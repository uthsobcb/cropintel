import os, json, io
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="🌾 Rice Yield Prediction", page_icon="🌾", layout="wide")

DEFAULT_MODELS_DIR = "./models"
SEASONS = ["ALL", "Aman", "Aus", "Boro"]

# -----------------------------------------------------------------------------
# Helpers to load artifacts
# -----------------------------------------------------------------------------
@st.cache_resource
def load_manifest(models_dir: str, tag: str, featureset: str = "SELECTED") -> Optional[List[dict]]:
    path = os.path.join(models_dir, f"models_manifest_{tag}_{featureset}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

@st.cache_resource
def load_features(models_dir: str, tag: str, featureset: str = "SELECTED") -> List[str]:
    candidates = [
        os.path.join(models_dir, f"features_{tag}_{featureset}.csv"),
        os.path.join(models_dir, f"features_{tag}.csv"),
        os.path.join(models_dir, "features_WITHLAG_SELECTED.csv"),
        os.path.join(models_dir, "features_NOLAG_SELECTED.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            s = pd.read_csv(p)["feature"].tolist()
            # ensure uniqueness / order preservation
            seen, ordered = set(), []
            for f in s:
                if f not in seen:
                    seen.add(f); ordered.append(f)
            return ordered
    raise FileNotFoundError("Feature list CSV not found in models dir.")

@st.cache_resource
def load_models_by_season(models_dir: str, manifest: List[dict]) -> Dict[str, Any]:
    cache = {}
    for m in manifest:
        season = m["season"]  # "ALL", "Aman", "Aus", "Boro"
        p = m["path"]
        if not os.path.isabs(p):
            p = os.path.join(models_dir, os.path.basename(p))
        if not os.path.exists(p) and os.path.exists(m["path"]):
            p = m["path"]
        cache[season] = joblib.load(p)
    return cache

@st.cache_data
def load_training_medians(models_dir: str, feature_list: List[str]) -> Dict[str, float]:
    """Use training medians for imputation if available."""
    for fname in ["training_panel_richlags.csv", "training_panel.csv"]:
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                cols = [c for c in feature_list if c in df.columns]
                if cols:
                    return df[cols].median(numeric_only=True).to_dict()
            except Exception:
                pass
    return {}

def is_lag_feature(name: str) -> bool:
    return name.startswith("Yield_t_ha_lag") or name == "Yield_t_ha_ma3"

def coerce_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def build_feature_row(input_map: Dict[str, Any], feature_list: List[str], medians: Dict[str, float]) -> Tuple[pd.DataFrame, int]:
    vals, missing = [], 0
    for f in feature_list:
        v = coerce_num(input_map.get(f, np.nan))
        if (v is None) or (isinstance(v, float) and np.isnan(v)):
            missing += 1
            v = float(medians.get(f, 0.0))
        vals.append(v)
    X = pd.DataFrame([vals], columns=feature_list)
    return X, missing

def route_model(models_map: Dict[str, Any], season: str) -> Any:
    return models_map.get(season) or models_map.get("ALL")

# -----------------------------------------------------------------------------
# Feature glossary (show help + realistic ranges)
# Edit / extend as needed for your selected feature set
# -----------------------------------------------------------------------------
FEATURE_INFO = {
    # Lag memory
    "Yield_t_ha_lag1": ("Last year's yield (same district & season)", "t/ha", "Aman 2.2–3.2 | Aus 1.8–2.8 | Boro 3.5–5.5"),
    "Yield_t_ha_lag2": ("Yield two years ago", "t/ha", "Similar to lag1"),
    "Yield_t_ha_lag3": ("Yield three years ago", "t/ha", "Similar to lag1"),
    "Yield_t_ha_ma3": ("3-season moving average of yield", "t/ha", "Between lag1 and district-season mean"),
    # Vegetation
    "NDVI_mean_season": ("Mean NDVI over active crop period", "0–1", "Typical 0.25–0.85"),
    "NDVI_mean_lag1": ("Mean NDVI last year (same season)", "0–1", "0.25–0.85"),
    "NDVI_mean_lag2": ("Mean NDVI two years ago", "0–1", "0.25–0.85"),
    "NDVI_mean_trend2": ("NDVI change vs 2y ago (mean - lag2)", "Δ", "−0.2 to +0.2"),
    "NDVI_anom": ("NDVI anomaly vs long-term mean", "Δ", "−0.2 to +0.2"),
    "NDVI_max_season": ("Peak NDVI in season", "0–1", "0.3–0.9"),
    "NDVI_integral": ("Area under NDVI curve (season)", "unitless", "~(0.2–0.8) × season_days"),
    # Climate (seasonal aggregates over planting→harvest)
    "Rain": ("Total rainfall in season", "mm", "Aman 800–1600 | Aus 400–900 | Boro 150–600"),
    "Tmax": ("Mean daily maximum temperature", "°C", "30–36"),
    "Tmin": ("Mean daily minimum temperature", "°C", "16–26"),
    "Tmean": ("Mean temperature", "°C", "22–30"),
    "Sunshine": ("Total bright sunshine hours (season)", "hours", "350–800"),
    "RH": ("Mean relative humidity", "%", "60–90"),
    "Wind": ("Mean wind speed", "m/s", "0.5–3"),
    "Cloud": ("Cloudiness (okta/% proxy)", "okta/%", "2–7 okta (≈25–80%)"),
    "TempRange": ("Diurnal temperature range (Tmax−Tmin)", "°C", "6–14"),
    # Irrigation/cropping context
    "Pct_irrigated": ("Share of area irrigated", "0–1", "Aman 0.1–0.5 | Aus 0.2–0.6 | Boro 0.8–1.0"),
    "Area_ir": ("Irrigated area", "ha", "District-season context"),
    "Area_rf": ("Rainfed area", "ha", "District-season context"),
    "Area_total": ("Total cropped area", "ha", "District-season context"),
    "Planting_Month_w": ("Weighted planting month", "1–12", "Month number"),
    "Maturity_Month_w": ("Weighted maturity month", "1–12", "Month number"),
}

# -----------------------------------------------------------------------------
# Sidebar: bundle choice & mode (lag vs no-lag)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

models_dir = st.sidebar.text_input("Models directory", value=DEFAULT_MODELS_DIR)

# Load both bundles if available
withlag_manifest = load_manifest(models_dir, "WITHLAG", "SELECTED")
nolag_manifest = load_manifest(models_dir, "NOLAG", "SELECTED")

if withlag_manifest is None:
    st.error("Missing WITHLAG bundle in models/. Please copy your trained artifacts.")
    st.stop()

withlag_features = load_features(models_dir, "WITHLAG", "SELECTED")
withlag_models_map = load_models_by_season(models_dir, withlag_manifest)
withlag_medians = load_training_medians(models_dir, withlag_features)

nolag_available = nolag_manifest is not None
if nolag_available:
    nolag_features = load_features(models_dir, "NOLAG", "SELECTED")
    nolag_models_map = load_models_by_season(models_dir, nolag_manifest)
    nolag_medians = load_training_medians(models_dir, nolag_features)
else:
    nolag_features, nolag_models_map, nolag_medians = None, None, {}

use_no_lag = st.sidebar.toggle(
    "I don't know last year's yield (use NO-lag model)",
    value=False,
    help="If ON and NO-lag bundle exists, the app will route to the NOLAG model and hide lag fields."
)

active_tag = "NOLAG" if (use_no_lag and nolag_available) else "WITHLAG"
features = nolag_features if (active_tag == "NOLAG") else withlag_features
models_map = nolag_models_map if (active_tag == "NOLAG") else withlag_models_map
medians = nolag_medians if (active_tag == "NOLAG") else withlag_medians

with st.sidebar.expander("Loaded bundles", expanded=False):
    st.write(f"**WITHLAG** seasons: {sorted(list(withlag_models_map.keys()))} | #features: {len(withlag_features)}")
    st.write(f"**NOLAG** available: {bool(nolag_available)}")
    if nolag_available:
        st.write(f"NOLAG seasons: {sorted(list(nolag_models_map.keys()))} | #features: {len(nolag_features)}")

if use_no_lag and not nolag_available:
    st.warning("No NO-lag bundle found. We'll still predict using WITHLAG by imputing lag fields (lower accuracy).")

# -----------------------------------------------------------------------------
# Header & Education panel
# -----------------------------------------------------------------------------
st.title("🌾 Rice Yield Prediction")

st.markdown("""
Use this tool to predict **seasonal rice yield (t/ha)** per district-season using **climate, NDVI, irrigation**, and optionally **last year's yield**.

- **Don’t know last year’s yield?** Turn on *“I don’t know last year’s yield”* in the sidebar — we’ll use a **NO-lag model** if available, or impute missing lag fields if not.
- **Tip:** Yield memory (lag) is the single strongest predictor. Provide it when possible for best accuracy.
""")

with st.expander("📘 What each feature means (units & realistic ranges)"):
    info = []
    for f in features:
        desc, unit, rng = FEATURE_INFO.get(f, ("—", "—", "—"))
        info.append((f, desc, unit, rng))
    df_info = pd.DataFrame(info, columns=["Feature", "What it means", "Units", "Typical range"])
    st.dataframe(df_info, use_container_width=True)

# -----------------------------------------------------------------------------
# Tabs: Manual & Batch
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔢 Manual Input", "📄 Batch CSV"])

# -----------------------------------------------------------------------------
# Manual Input
# -----------------------------------------------------------------------------
with tab1:
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        season = st.selectbox("Season", options=SEASONS, index=0)
    with c2:
        ##st.caption("Missing values are filled with **training medians** (if available), else 0.")
        show_all = st.toggle("Show all feature inputs", value=False, help="If OFF: common features only. If ON: every feature.")

    # A compact, friendly set for casual users:
    common_fields = [
        "Yield_t_ha_lag1", "NDVI_mean_season", "Rain", "Tmax", "Tmin", "Sunshine", "RH", "Wind", "Cloud", "Pct_irrigated"
    ]
    to_render = features if show_all else [f for f in common_fields if f in features]

    # Hide lag fields if user chose NO-lag AND NOLAG bundle exists
    if active_tag == "NOLAG":
        to_render = [f for f in to_render if not is_lag_feature(f)]

    st.subheader("Inputs")
    inputs: Dict[str, Any] = {}
    grid_cols = st.columns(3)
    for i, fname in enumerate(to_render):
        with grid_cols[i % 3]:
            desc, unit, rng = FEATURE_INFO.get(fname, ("", "", ""))
            label = fname if not unit or unit == "—" else f"{fname} ({unit})"
            default = float(medians.get(fname, 0.0))
            help_txt = f"{desc}\n\nTypical: {rng}" if desc or rng else None
            val = st.number_input(label, value=default, step=0.01, format="%.4f", help=help_txt)
            inputs[fname] = val

    if st.button("Predict", type="primary"):
        mdl = route_model(models_map, season)
        if mdl is None:
            st.error("No model available for selected season or ALL.")
        else:
            # Build payload including only provided inputs; missing will be imputed
            X, miss = build_feature_row(inputs, features, medians)
            yhat = float(mdl.predict(X)[0])
            note = "NO-lag model" if active_tag == "NOLAG" else "WITH-lag model"
            st.success(f"**Predicted Yield:** {yhat:.3f} t/ha  ·  **Bundle:** {note}  ·  **Missing filled:** {miss}")

# -----------------------------------------------------------------------------
# Batch CSV
# -----------------------------------------------------------------------------
with tab2:
    st.write("Upload a CSV. Columns can be a subset of the current bundle's features; missing ones are imputed.")
    st.caption("Switch the sidebar toggle if you prefer the NO-lag mode template.")

    # Provide a downloadable template that matches the current mode:
    current_schema = [f for f in features if (active_tag != "NOLAG" or not is_lag_feature(f))]
    tpl = pd.DataFrame({"feature": current_schema})
    st.download_button("Download CSV template (current mode)", data=tpl.to_csv(index=False),
                       file_name=f"features_schema_{active_tag}.csv", mime="text/csv")

    up = st.file_uploader("CSV file", type=["csv"])
    season_b = st.selectbox("Season for batch", options=SEASONS, index=0, key="season_batch")

    if up is not None:
        df_in = pd.read_csv(up)
        st.write("Preview:", df_in.head())

        # Align to current mode's features (impute missing)
        aligned, miss_counts = [], []
        for _, row in df_in.iterrows():
            payload = {k: row.get(k, np.nan) for k in df_in.columns}
            X_row, miss = build_feature_row(payload, features, medians)
            aligned.append(X_row); miss_counts.append(miss)
        X_all = pd.concat(aligned, ignore_index=True)

        mdl_b = route_model(models_map, season_b)
        preds = mdl_b.predict(X_all).astype(float)
        out = df_in.copy()
        out["predicted_yield_t_ha"] = preds
        out["missing_filled"] = miss_counts

        st.success("Done. Showing first rows with predictions:")
        st.dataframe(out.head(), use_container_width=True)

        buff = io.StringIO()
        out.to_csv(buff, index=False)
        st.download_button("Download results CSV", data=buff.getvalue(),
                           file_name=f"predictions_{active_tag}.csv", mime="text/csv")
