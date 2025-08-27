import os
import io
import time
import json
import requests
import pandas as pd
import streamlit as st

# ----------------- Config -----------------
st.set_page_config(page_title="CropIntel Control Panel", layout="wide")

# Allow overriding if API is on another host/port
DEFAULT_API = os.getenv("CROPINTEL_API", "http://localhost:8000")

with st.sidebar:
    st.title("Settings")
    api_base = st.text_input("FastAPI base URL", value=DEFAULT_API)
    st.caption("Tip: set env var CROPINTEL_API to change default.")
    st.divider()
    st.caption("This UI calls the API and, when possible, reads returned artifacts from disk if running on the same machine.")

def api_url(path: str) -> str:
    return api_base.rstrip("/") + path

def read_parquet_if_exists(path: str) -> pd.DataFrame | None:
    try:
        if path and os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        pass
    return None

# Session state
if "last_run" not in st.session_state:
    st.session_state.last_run = {}

# ----------------- Tabs -----------------
tab1, tab2, tab3, tab4 = st.tabs(["DiseaseAlert", "YieldCast", "ResourceOpt", "Run All"])

# ----------------- DiseaseAlert -----------------
with tab1:
    st.header("DiseaseAlert: Upload leaf images")
    district = st.text_input("District name (optional, used to aggregate risk)", value="")
    files = st.file_uploader("Upload leaf images", type=["jpg","jpeg","png","webp"], accept_multiple_files=True)

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Run DiseaseAlert", disabled=not files):
            if not files:
                st.warning("Please upload at least one image.")
            else:
                # Build multipart form
                m = []
                for f in files:
                    m.append(("files", (f.name, f.getvalue(), f"type={f.type}" if hasattr(f, "type") else "application/octet-stream")))
                data = {}
                if district.strip():
                    data["district_name"] = district.strip()
                try:
                    resp = requests.post(api_url("/run/diseasealert"), files=m, data=data, timeout=180)
                    if resp.ok:
                        out = resp.json()
                        st.success("DiseaseAlert run complete.")
                        st.json(out)
                        st.session_state.last_run["disease"] = out
                        # Preview per-image predictions
                        preview = out.get("preview") or []
                        if preview:
                            st.subheader("Preview predictions")
                            st.dataframe(pd.DataFrame(preview))
                        # If a disease_risk parquet is present locally, load and show
                        dr_path = (out.get("artifacts") or {}).get("disease_risk")
                        df = read_parquet_if_exists(dr_path)
                        if df is not None and len(df):
                            st.subheader("Aggregated risk (from local artifact)")
                            st.dataframe(df)
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    with colB:
        st.info("If your API and Streamlit run on different machines, the returned artifact paths won’t exist locally. You’ll still see the JSON response.")

# ----------------- YieldCast -----------------
with tab2:
    st.header("YieldCast: Batch predict")
    if st.button("Run YieldCast"):
        try:
            resp = requests.post(api_url("/run/yieldcast"), timeout=180)
            if resp.ok:
                out = resp.json()
                st.success("YieldCast run complete.")
                st.json(out)
                st.session_state.last_run["yieldcast"] = out
                p = (out.get("artifacts") or {}).get("yieldcast_predictions")
                df = read_parquet_if_exists(p)
                if df is not None and len(df):
                    st.subheader("Predictions (from local artifact)")
                    st.dataframe(df)
                    st.bar_chart(df.set_index("district_id")["yield_pred_kg_ha"])
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# ----------------- ResourceOpt -----------------
with tab3:
    st.header("ResourceOpt: Optimize allocation")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_supply_kg = st.number_input("Total supply (kg)", min_value=0, value=200_000, step=10_000)
    with col2:
        bag_size_kg = st.number_input("Bag size (kg)", min_value=1, value=50, step=1)
    with col3:
        lambda_cost = st.number_input("Logistics penalty λ", min_value=0.0, value=0.2, step=0.1, format="%.2f")

    st.caption("Weights (Need/Potential). Leave blank to use API defaults.")
    cw1, cw2, cw3, cw4, cw5, cw6, cw7 = st.columns(7)
    with cw1: w_risk = st.text_input("w_risk", value="")
    with cw2: w_rain = st.text_input("w_rain", value="")
    with cw3: w_prior = st.text_input("w_prior", value="")
    with cw4: v_fert = st.text_input("v_fert", value="")
    with cw5: v_resp = st.text_input("v_resp", value="")
    with cw6: a_w    = st.text_input("a (Need)", value="")
    with cw7: b_w    = st.text_input("b (Potential)", value="")

    colL, colR = st.columns([1,1])
    with colL:
        use_last_disease = st.checkbox("Use disease_risk from last DiseaseAlert run (if available)")
        use_last_yield   = st.checkbox("Use yield predictions from last YieldCast run (if available)", value=True)

    req_body = {
        "total_supply_kg": int(total_supply_kg),
        "bag_size_kg": int(bag_size_kg),
        "lambda_cost": float(lambda_cost),
    }
    # attach weights only if user filled
    weights = {}
    def _maybe_add(k, v):
        if v.strip() != "":
            try:
                weights[k] = float(v.strip())
            except Exception:
                pass
    _maybe_add("w_risk", w_risk); _maybe_add("w_rain", w_rain); _maybe_add("w_prior", w_prior)
    _maybe_add("v_fert", v_fert); _maybe_add("v_resp", v_resp)
    _maybe_add("a", a_w); _maybe_add("b", b_w)
    if weights:
        req_body["weights"] = weights

    # pass parquet paths from last runs if local
    if use_last_disease and "disease" in st.session_state.last_run:
        p = (st.session_state.last_run["disease"].get("artifacts") or {}).get("disease_risk")
        if p and os.path.exists(p):  # only pass if readable locally
            req_body["disease_parquet"] = p
    if use_last_yield and "yieldcast" in st.session_state.last_run:
        p = (st.session_state.last_run["yieldcast"].get("artifacts") or {}).get("yieldcast_predictions")
        if p and os.path.exists(p):
            req_body["yield_parquet"] = p

    run_opt = st.button("Solve ResourceOpt")
    if run_opt:
        try:
            resp = requests.post(api_url("/run/resourceopt"), json=req_body, timeout=300)
            if resp.ok:
                out = resp.json()
                st.success("ResourceOpt run complete.")
                st.json(out)
                st.session_state.last_run["resourceopt"] = out

                alloc_path = (out.get("artifacts") or {}).get("allocations")
                df = read_parquet_if_exists(alloc_path)
                if df is not None and len(df):
                    st.subheader("Allocations (from local artifact)")
                    st.dataframe(df)
                    st.bar_chart(df.set_index("district_id")["allocation_kg"])
                else:
                    # fallback to top5 from API response
                    top5 = out.get("top5") or []
                    if top5:
                        st.subheader("Top 5 (from API response)")
                        st.dataframe(pd.DataFrame(top5))
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# ----------------- Run All -----------------
with tab4:
    st.header("Run entire pipeline")
    if st.button("Run ALL (YieldCast → ResourceOpt; Disease neutral)"):
        try:
            resp = requests.post(api_url("/run/all"), timeout=600)
            if resp.ok:
                out = resp.json()
                st.success("Pipeline run complete.")
                st.json(out)
                st.session_state.last_run["all"] = out
                # try to load artifacts
                yc_path = (out.get("artifacts") or {}).get("yieldcast_predictions")
                alloc_path = (out.get("artifacts") or {}).get("allocations")
                df_y = read_parquet_if_exists(yc_path)
                df_a = read_parquet_if_exists(alloc_path)
                if df_y is not None and len(df_y):
                    st.subheader("Yield predictions (from local artifact)")
                    st.dataframe(df_y)
                if df_a is not None and len(df_a):
                    st.subheader("Allocations (from local artifact)")
                    st.dataframe(df_a)
                    st.bar_chart(df_a.set_index("district_id")["allocation_kg"])
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
