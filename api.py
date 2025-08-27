from __future__ import annotations
import os, io, json, yaml, pathlib, time
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import joblib
import lightgbm as lgb
import pulp as pl

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# -------------------- config & utils --------------------

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

CFG = load_yaml("config.yaml")

def mk_run_dir(root="artifacts", season="unknown") -> str:
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(root, f"{season}_{run_id}")
    os.makedirs(out, exist_ok=True)
    return out

def save_parquet(df: pd.DataFrame, path: str) -> str:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path

def save_json(obj, path: str) -> str:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path

def minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)

SEASON = CFG.get("season", "unknown")

# -------------------- DiseaseAlert (PyTorch .pt) --------------------

class DiseaseConfig(BaseModel):
    district_name: Optional[str] = None

def _load_labels(labels_path: str) -> List[str]:
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            labels = json.load(f)
        if isinstance(labels, dict):
            # allow {"0":"Brown Spot", ...}
            ordered = [labels[str(i)] for i in range(len(labels))]
            return ordered
        return list(labels)
    return []

def _preprocess_image(img: Image.Image, size: int, mean: List[float], std: List[float]) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    t = torch.from_numpy(arr).unsqueeze(0)  # 1xCxHxW
    return t

def _softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=1)

def load_torch_model(pt_path: str):
    # Prefer TorchScript for server inference
    try:
        m = torch.jit.load(pt_path, map_location="cpu")
        m.eval()
        return m, True
    except Exception:
        # Fallback: try standard state_dict with a basic classifier shape (user should customize)
        return None, False

# cache global
_TORCH_MODEL = None
_TORCH_MODEL_IS_SCRIPT = False
_LABELS = _load_labels(CFG["paths"].get("disease_labels", ""))

def ensure_torch_model():
    global _TORCH_MODEL, _TORCH_MODEL_IS_SCRIPT
    if _TORCH_MODEL is None:
        _TORCH_MODEL, _TORCH_MODEL_IS_SCRIPT = load_torch_model(CFG["models"]["diseasealert_pt"])
    return _TORCH_MODEL, _TORCH_MODEL_IS_SCRIPT

# -------------------- YieldCast (LightGBM .joblib) --------------------

_LGB_MODEL = None
_LGB_FEATURE_LIST: Optional[List[str]] = None

def ensure_lgb_model():
    global _LGB_MODEL, _LGB_FEATURE_LIST
    if _LGB_MODEL is None:
        _LGB_MODEL = joblib.load(CFG["paths"]["yield_model_joblib"])  # lgb.Booster or sklearn API
        # Try to extract feature names if saved
        try:
            _LGB_FEATURE_LIST = _LGB_MODEL.feature_name()
        except Exception:
            _LGB_FEATURE_LIST = None
    return _LGB_MODEL, _LGB_FEATURE_LIST

def lgb_predict(features_csv: str, season: str) -> pd.DataFrame:
    model, feat_names = ensure_lgb_model()
    feats = pd.read_csv(features_csv)
    # filter current season if the file has multi-season rows
    if "season" in feats.columns:
        feats = feats[feats["season"] == season].copy()
    if "district_id" not in feats.columns:
        raise ValueError("features CSV must include 'district_id' column.")
    # build X matrix
    if feat_names:
        X = feats[feat_names]
    else:
        X = feats[[c for c in feats.columns if c not in ("district_id", "season")]]
    # predict
    if isinstance(model, lgb.Booster):
        y_pred = model.predict(X)
    else:
        y_pred = model.predict(X)  # sklearn-like
    # naive 15% CI band (replace with your calibrated approach when ready)
    ci_w = 0.15 * np.abs(y_pred)
    out = feats[["district_id"]].copy()
    out["season"] = season
    out["yield_pred_kg_ha"] = y_pred
    out["ci_low"] = y_pred - ci_w
    out["ci_high"] = y_pred + ci_w
    out["anomaly_flag"] = False
    return out

# -------------------- ResourceOpt (PuLP ILP) --------------------

class ResourceBody(BaseModel):
    total_supply_kg: Optional[int] = None
    bag_size_kg: Optional[int] = None
    lambda_cost: Optional[float] = None
    weights: Optional[Dict[str, float]] = None
    composite: Optional[str] = None
    region_caps: Optional[Dict[str, float]] = None
    # allow passing previously computed disease/yield paths (optional)
    disease_parquet: Optional[str] = None
    yield_parquet: Optional[str] = None

def resourceopt_solve(
    season: str,
    drivers_parquet: str,
    disease_df: Optional[pd.DataFrame],
    yield_df: pd.DataFrame,
    total_supply_kg: int,
    bag_size_kg: int,
    lambda_cost: float,
    weights: Dict[str, float],
    composite: str = "weighted",
    region_caps: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    drv = pd.read_parquet(drivers_parquet)
    if disease_df is not None:
        if "district_name" in disease_df.columns and "district_name" in drv.columns:
            drv = drv.merge(disease_df, on="district_name", how="left")
        elif "district_id" in disease_df.columns:
            drv = drv.merge(disease_df, on="district_id", how="left")
    if "disease_risk" not in drv.columns:
        drv["disease_risk"] = 0.5

    yc = yield_df
    if "season" in yc.columns:
        yc = yc[yc["season"] == season]
    drv = drv.merge(yc[["district_id","yield_pred_kg_ha"]], on="district_id", how="left")

    # scores
    drv["risk_n"]   = minmax(drv["disease_risk"])
    drv["rain_n"]   = minmax(drv["rain_deficit_mm"])
    drv["prior_n"]  = minmax(drv["prior_input_deficit_ratio"])
    drv["fert_n"]   = minmax(drv["soil_fertility_index"])
    drv["resp_n"]   = minmax(drv["yield_responsiveness"])

    drv["need"]      = weights["w_risk"]*drv["risk_n"] + weights["w_rain"]*drv["rain_n"] + weights["w_prior"]*drv["prior_n"]
    drv["potential"] = weights["v_fert"]*drv["fert_n"] + weights["v_resp"]*drv["resp_n"]

    if composite == "weighted":
        drv["score"] = weights["a"]*drv["need"] + weights["b"]*drv["potential"]
    else:
        alpha = weights.get("alpha", 1.0)
        beta  = weights.get("beta", 1.0)
        drv["score"] = ((drv["need"]+1e-9)**alpha) * ((drv["potential"]+1e-9)**beta)
        drv["score"] = minmax(drv["score"])

    # ILP
    model = pl.LpProblem("ResourceOpt", pl.LpMaximize)
    bag = max(1, int(bag_size_kg))
    x_units = {i: pl.LpVariable(f"x_{i}", lowBound=0, cat=pl.LpInteger) for i in drv.index}
    def xkg(i): return x_units[i] * bag
    cost = drv.get("transport_cost_per_kg", pd.Series(np.zeros(len(drv)), index=drv.index))

    model += pl.lpSum([(drv.loc[i,"score"] - lambda_cost*cost.iloc[i]) * xkg(i) for i in drv.index])
    model += pl.lpSum([xkg(i) for i in drv.index]) <= float(total_supply_kg)
    for i in drv.index:
        model += xkg(i) <= float(drv.loc[i, "max_absorption_kg"])
    if region_caps:
        assert "region" in drv.columns, "region column required for region caps"
        for r, cap in region_caps.items():
            idxs = drv.index[drv["region"] == r].tolist()
            if idxs:
                model += pl.lpSum([xkg(i) for i in idxs]) <= float(cap)

    model.solve(pl.PULP_CBC_CMD(msg=False))

    drv["allocation_kg"] = [pl.value(x_units[i]) * bag for i in drv.index]
    drv["rank"] = drv["allocation_kg"].rank(ascending=False, method="dense").astype(int)
    keep = ["district_id"]
    if "district_name" in drv.columns: keep.append("district_name")
    keep += ["allocation_kg","score","need","potential","rank"]
    return drv[keep].sort_values(["allocation_kg","score"], ascending=[False, False]).reset_index(drop=True)

# -------------------- FastAPI app --------------------

app = FastAPI(title="CropIntel API", version="0.1.0")

@app.get("/")
def root():
    return {"ok": True, "season": SEASON}

# 1) DiseaseAlert: upload images → per-image predictions + aggregated risk
@app.post("/run/diseasealert")
async def run_diseasealert(files: List[UploadFile] = File(...), district_name: Optional[str] = Form(None)):
    run_dir = mk_run_dir("artifacts", SEASON)
    model, is_script = ensure_torch_model()
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Torch model not loadable. Provide TorchScript .pt."})

    size = int(CFG["preprocess"]["img_size"])
    mean = CFG["preprocess"]["mean"]
    std  = CFG["preprocess"]["std"]

    preds = []
    for up in files:
        img = Image.open(io.BytesIO(await up.read()))
        t = _preprocess_image(img, size=size, mean=mean, std=std)
        with torch.no_grad():
            out = model(t) if is_script else None
            if out is None:
                return JSONResponse(status_code=500, content={"error": "Model not TorchScript. Please export scripted .pt."})
            prob = _softmax(out).cpu().numpy()[0]
        cls_idx = int(prob.argmax())
        cls_name = _LABELS[cls_idx] if _LABELS and cls_idx < len(_LABELS) else str(cls_idx)
        severity = float(prob[cls_idx])  # use winning prob as "severity" proxy (replace if you have a reg head)
        preds.append({"file": up.filename, "class": cls_name, "prob": float(prob[cls_idx]), "severity": severity})

    df = pd.DataFrame(preds)
    # aggregate district risk if provided
    if district_name:
        agg = pd.DataFrame([{"district_name": district_name, "disease_risk": float(df["severity"].mean())}])
    else:
        # if multiple districts are mixed, the client should call separately; here we only return per-image preds
        agg = pd.DataFrame(columns=["district_name","disease_risk"])

    risk_path = save_parquet(agg, os.path.join(run_dir, "disease_risk.parquet")) if len(agg) else None
    pred_path = save_parquet(df,  os.path.join(run_dir, "disease_predictions.parquet"))

    return {
        "season": SEASON,
        "run_dir": run_dir,
        "artifacts": {
            "disease_predictions": pred_path,
            "disease_risk": risk_path
        },
        "preview": preds[:5]
    }

# 2) YieldCast: run LightGBM model on features CSV
@app.post("/run/yieldcast")
def run_yieldcast():
    run_dir = mk_run_dir("artifacts", SEASON)
    out = lgb_predict(CFG["paths"]["yield_features_csv"], SEASON)
    p = save_parquet(out, os.path.join(run_dir, "yieldcast_predictions.parquet"))
    return {"season": SEASON, "run_dir": run_dir, "artifacts": {"yieldcast_predictions": p}, "rows": int(len(out))}

# 3) ResourceOpt: ILP using drivers + latest yield (or recompute) + optional disease risk parquet
@app.post("/run/resourceopt")
def run_resourceopt(body: ResourceBody):
    run_dir = mk_run_dir("artifacts", SEASON)

    # disease
    disease_df = pd.read_parquet(body.disease_parquet) if body.disease_parquet and os.path.exists(body.disease_parquet) else None
    # yield
    if body.yield_parquet and os.path.exists(body.yield_parquet):
        yield_df = pd.read_parquet(body.yield_parquet)
    else:
        yield_df = lgb_predict(CFG["paths"]["yield_features_csv"], SEASON)

    # params
    rcfg = CFG["resourceopt"].copy()
    if body.total_supply_kg is not None: rcfg["total_supply_kg"] = body.total_supply_kg
    if body.bag_size_kg    is not None: rcfg["bag_size_kg"]    = body.bag_size_kg
    if body.lambda_cost    is not None: rcfg["lambda_cost"]    = body.lambda_cost
    if body.weights        is not None: rcfg["weights"]        = body.weights
    if body.composite      is not None: rcfg["composite"]      = body.composite
    if body.region_caps    is not None: rcfg["region_caps"]    = body.region_caps

    alloc = resourceopt_solve(
        season=SEASON,
        drivers_parquet=CFG["paths"]["drivers_parquet"],
        disease_df=disease_df,
        yield_df=yield_df,
        total_supply_kg=rcfg["total_supply_kg"],
        bag_size_kg=rcfg["bag_size_kg"],
        lambda_cost=rcfg["lambda_cost"],
        weights=rcfg["weights"],
        composite=rcfg.get("composite","weighted"),
        region_caps=rcfg.get("region_caps", {})
    )

    p = save_parquet(alloc, os.path.join(run_dir, "allocations.parquet"))
    top = alloc.head(5).to_dict(orient="records")
    return {
        "season": SEASON,
        "run_dir": run_dir,
        "artifacts": {"allocations": p},
        "top5": top
    }

# 4) All stages (no images here; disease defaults to neutral risk unless you pass disease_parquet later)
@app.post("/run/all")
def run_all():
    run_dir = mk_run_dir("artifacts", SEASON)
    artifacts = {}

    # Yield first (deterministic)
    yc = lgb_predict(CFG["paths"]["yield_features_csv"], SEASON)
    artifacts["yieldcast_predictions"] = save_parquet(yc, os.path.join(run_dir, "yieldcast_predictions.parquet"))

    # Disease: neutral unless client uploaded earlier
    disease_df = None

    # ResourceOpt
    rcfg = CFG["resourceopt"]
    alloc = resourceopt_solve(
        season=SEASON,
        drivers_parquet=CFG["paths"]["drivers_parquet"],
        disease_df=disease_df,
        yield_df=yc,
        total_supply_kg=rcfg["total_supply_kg"],
        bag_size_kg=rcfg["bag_size_kg"],
        lambda_cost=rcfg["lambda_cost"],
        weights=rcfg["weights"],
        composite=rcfg.get("composite","weighted"),
        region_caps=rcfg.get("region_caps", {})
    )
    artifacts["allocations"] = save_parquet(alloc, os.path.join(run_dir, "allocations.parquet"))

    save_json({"season": SEASON, "run_dir": run_dir, "artifacts": artifacts}, os.path.join(run_dir, "summary.json"))
    return {"season": SEASON, "run_dir": run_dir, "artifacts": artifacts, "top5": alloc.head(5).to_dict(orient="records")}
