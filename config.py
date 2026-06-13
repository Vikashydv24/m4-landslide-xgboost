"""
config.py — central settings for M-4 LSM pipeline.
All scripts import from here.
"""
import json
from pathlib import Path

# ── Seeds & CV ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_CV_FOLDS  = 5

# ── Study area ────────────────────────────────────────────────────────────
KERALA_BBOX = (74.9, 8.3, 77.5, 12.8)

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
DEM_PATH    = ROOT / "data" / "dem" / "kerala_srtm30.tif"
PRED_DIR    = DATA_PROC / "predictors"
FEAT_PATH   = DATA_PROC / "features_all.parquet"
MODELS_DIR  = ROOT / "results" / "models"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
SHAP_DIR    = ROOT / "results" / "shap"
MAPS_DIR    = ROOT / "results" / "maps"
BEST_PARAMS = METRICS_DIR / "best_params.json"

# ── Predictor names ───────────────────────────────────────────────────────
PREDICTOR_NAMES = [
    "elevation", "slope", "aspect",
    "plan_curv", "prof_curv",
    "twi", "tri", "tpi",
    "spi", "sti",
    "flow_acc", "dist_river",
]

# ── Columns to exclude from model input ───────────────────────────────────
EXCLUDE_COLS = {
    # labels and coordinates
    "label", "POINT_X", "POINT_Y",
    # shapefile-only columns (leakage — only real for landslide points)
    "RASTERVALU", "Reclass_Sl",
    # ALL LU columns excluded — background has artificial LU values
    # so any LU one-hot column perfectly separates the two classes.
    # Only DEM-derived terrain features are used (no LU raster available).
    # LU columns are excluded dynamically in load_data() via startswith check.
    # metadata
    "Type_of_sl", "District", "geometry",
    "No", "NRSC", "GSI", "New", "Specific_r", "Remarks",
    "Building_I", "Road_impac", "Impact_Agr",
    "Length", "Width", "Area",
}

# ── Default hyperparameters ───────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators = 210,
    max_features = "sqrt",
    class_weight = "balanced",
    n_jobs       = -1,
    random_state = RANDOM_SEED,
)

GBM_PARAMS = dict(
    n_estimators  = 300,
    learning_rate = 0.1,
    max_depth     = 4,
    subsample     = 0.8,
    random_state  = RANDOM_SEED,
)

XGB_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.7,
    colsample_bytree = 0.5,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    eval_metric      = "auc",
    n_jobs           = -1,
    random_state     = RANDOM_SEED,
)

# ── Load tuned params if available ────────────────────────────────────────
def get_model_params():
    rf  = RF_PARAMS.copy()
    gbm = GBM_PARAMS.copy()
    xgb = XGB_PARAMS.copy()

    if BEST_PARAMS.exists():
        with open(BEST_PARAMS) as f:
            best = json.load(f)
        skip = {"top_features","random_state","n_jobs","eval_metric"}
        for k,v in best.get("RF",{}).get("params",{}).items():
            if k not in skip: rf[k] = v
        for k,v in best.get("GBM",{}).get("params",{}).items():
            if k not in skip: gbm[k] = v
        for k,v in best.get("XGBoost",{}).get("params",{}).items():
            if k not in skip: xgb[k] = v
        print("  Loaded tuned params from best_params.json")
    else:
        print("  Using default hyperparameters")

    rf["random_state"]  = RANDOM_SEED
    gbm["random_state"] = RANDOM_SEED
    xgb["random_state"] = RANDOM_SEED
    return rf, gbm, xgb
