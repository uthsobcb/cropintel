# Rice Yield Prediction (Bangladesh, Agro-climatic & Temporal Data)

This repository contains machine learning models for **seasonal rice yield prediction** in Bangladesh, built on multiple datasets:
- Climate (temperature, rainfall, humidity, sunshine, etc.)
- NDVI (vegetation indices)
- Irrigation and crop calendars
- Historical production and area
- Yield targets (t/ha)

## 📂 Datasets Used
- `clim_features.csv` → seasonal climate indicators
- `ndvi_features_seasonal.csv` → NDVI seasonal aggregates
- `irr.csv` → irrigation areas (% irrigated, planting & maturity windows)
- `yields_panel.csv` → yield targets (Production, Area, Yield_t_ha)
- `crop_production_dataset_10years2005-2014_mod.csv` → historical production

All merged into a **training panel** with added lagged, rolling, and interaction features.

---

## 🔑 Features Engineered
1. **Yield memory**
   - `Yield_t_ha_lag1`, `lag2`, `lag3`
   - `Yield_t_ha_ma3` (3-season moving average)

2. **NDVI lags & trends**
   - `NDVI_mean_lag1`, `lag2`
   - `NDVI_mean_trend2`

3. **Rolling climate/NDVI means**
   - `Rain_ma3`, `Tmax_ma3`, `Tmin_ma3`, `Sunshine_ma3`, `NDVI_mean_season_ma3`

4. **Interaction terms**
   - `lag1_rain_interaction`
   - `ndvi_rain_interaction`
   - `lag1_ndvi_interaction`

5. **Feature selection**
   - Variance filter → correlation pruning → model-based top-N (LightGBM/RandomForest)

---

## ⚙️ Models Trained
We benchmarked:
- **RandomForestRegressor**
- **XGBoost**
- **LightGBM**

Evaluation used:
- **Leave-One-Year-Out (LOYO)** cross-validation
- **KFold (5-fold)** cross-validation
- Per-season breakdown (Aman, Aus, Boro)

---

## 📊 Results (SELECTED features, WITH-lag)
| Model        | MAE   | RMSE  | R²   |
|--------------|-------|-------|------|
| **RandomForest** | **0.254** | **0.394** | **0.767** |
| XGBoost      | 0.280 | 0.427 | 0.726 |
| LightGBM     | 0.281 | 0.419 | 0.735 |

👉 **Best model: RandomForest WITH-lag, SELECTED features.**  
Explains ~77% of variance in yields.

---

## 📦 Deployment Artifacts
Inside `/content/model_outputs/`:

- **Best models**
  - `best_WITHLAG_RandomForest_SELECTED_ALL.joblib`  
  - `best_WITHLAG_RandomForest_SELECTED_Aman.joblib`  
  - `best_WITHLAG_RandomForest_SELECTED_Aus.joblib`  
  - `best_WITHLAG_RandomForest_SELECTED_Boro.joblib`  
  - (Optional) `best_NOLAG_*.joblib` for early-season forecasts without yield history

- **Feature schema**
  - `features_WITHLAG_SELECTED.csv`  
  - (Optional) `features_NOLAG_SELECTED.csv`

- **Model manifest**
  - `models_manifest_WITHLAG_SELECTED.json`  
  - (Optional) `models_manifest_NOLAG_SELECTED.json`

- **Evaluation reports** (useful for research, not deployment)
  - `cv_summary_*.csv`
  - `cv_per_year_*.csv`
  - `cv_per_season_*.csv`
  - `kfold_*.csv`

---

## 🚀 Example: Load & Predict

```python
import joblib, pandas as pd, json

BASE = "/content/model_outputs"

# Load manifest
with open(f"{BASE}/models_manifest_WITHLAG_SELECTED.json") as f:
    manifest = json.load(f)

# Load ALL-season best model (RandomForest)
all_model_path = [m["path"] for m in manifest if m["season"]=="ALL"][0]
model = joblib.load(all_model_path)

# Load feature schema
features = pd.read_csv(f"{BASE}/features_WITHLAG_SELECTED.csv")["feature"].tolist()

# Example payload (replace values with real observations)
payload = {f: 0.5 for f in features}  # dummy data
row = pd.DataFrame([payload], columns=features)

# Predict yield
y_pred = model.predict(row)[0]
print("Predicted Yield (t/ha):", y_pred)
