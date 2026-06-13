#!/usr/bin/env python3
"""
03_train_models.py
Train RF, GBM, XGBoost with spatial-block CV on Kerala 2018.

Outputs:
  results/metrics/summary_table.csv
  results/metrics/wilcoxon_table.csv
  results/metrics/cv_fold_results.csv
  results/models/*.pkl
  results/figures/roc_curves.png
  results/figures/auc_boxplot.png
  results/figures/oa_barplot.png
  results/figures/feature_importance.png
  results/figures/spatial_cv_blocks.png
  results/shap/shap_beeswarm_*.png
  results/shap/shap_values_*.csv
"""
import sys, os, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute   import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap

import config as C
from src.spatial_cv   import SpatialBlockCV
from src.evaluation   import (compute_metrics, aggregate_cv_metrics,
                               run_wilcoxon_table)
from src.visualization import (plot_roc_curves, plot_auc_boxplot,
                                plot_oa_bar, plot_feature_importance,
                                plot_spatial_cv_blocks)


# ── Create output folders ─────────────────────────────────────────────────
for d in [C.MODELS_DIR, C.METRICS_DIR, C.FIGURES_DIR, C.SHAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────
def load_data():
    print(f"\n{'='*55}")
    print("  Loading feature matrix")
    print(f"{'='*55}")

    if not C.FEAT_PATH.exists():
        print(f"\n  ERROR: {C.FEAT_PATH} not found.")
        print("  Run: python scripts/02_prepare_samples.py first.")
        sys.exit(1)

    df = pd.read_parquet(C.FEAT_PATH)
    print(f"  Rows  : {len(df):,}")
    print(f"  Label : {df['label'].value_counts().to_dict()}")

    feature_cols = [c for c in df.columns
                    if c not in C.EXCLUDE_COLS
                    and not c.startswith("LU_")]   # LU columns excluded
    # Only terrain rasters are used as features — LU values for background
    # points are artificial and create perfect label leakage

    # Encode any leftover categoricals
    for col in df[feature_cols].select_dtypes(
            include=["object","category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    y = df["label"].values.astype(int)

    # Drop columns that are entirely NaN
    all_nan = [c for c in feature_cols if df[c].isna().all()]
    if all_nan:
        print(f"  Warning: dropping {len(all_nan)} all-NaN column(s): {all_nan}")
        feature_cols = [c for c in feature_cols if c not in all_nan]

    # Median imputation
    imp   = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imp.fit_transform(df[feature_cols]),
        columns=feature_cols)

    # Clip extreme outliers to 2nd-98th percentile for each feature
    # (fixes richdem curvature inflation from missing geotransform)
    clip_cols = ["plan_curv", "prof_curv", "sti", "spi", "flow_acc", "dist_river"]
    for col in clip_cols:
        if col in X_imp.columns:
            lo = X_imp[col].quantile(0.02)
            hi = X_imp[col].quantile(0.98)
            X_imp[col] = X_imp[col].clip(lo, hi)
    print(f"  Outlier clipping applied to: {[c for c in clip_cols if c in X_imp.columns]}")

    # Attach coords for splitter (not used as model features)
    X_imp["POINT_X"] = df["POINT_X"].values
    X_imp["POINT_Y"] = df["POINT_Y"].values

    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    return X_imp, y, feature_cols


# ── Spatial CV ────────────────────────────────────────────────────────────
def run_cv(X, y, feature_cols, models):
    cv = SpatialBlockCV(
        n_splits=C.N_CV_FOLDS,
        lat_col="POINT_Y", lon_col="POINT_X",
        random_state=C.RANDOM_SEED)

    results   = {n: [] for n in models}
    oof_preds = {n: np.full(len(y), np.nan) for n in models}
    X_feat    = X[feature_cols]

    print(f"\n{'='*55}")
    print(f"  Spatial Block CV  ({C.N_CV_FOLDS} folds)")
    print(f"{'='*55}")

    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X_feat.iloc[tr], X_feat.iloc[te]
        y_tr, y_te = y[tr], y[te]

        print(f"\n  Fold {fold}/{C.N_CV_FOLDS}"
              f"  train={len(tr):,}  test={len(te):,}"
              f"  pos%={y_te.mean()*100:.1f}")

        for name, model in models.items():
            clone = type(model)(**model.get_params())

            if name == "XGBoost":
                clone.fit(X_tr, y_tr,
                          eval_set=[(X_te, y_te)],
                          verbose=False)
            else:
                clone.fit(X_tr, y_tr)

            proba = clone.predict_proba(X_te)[:, 1]
            oof_preds[name][te] = proba

            m = compute_metrics(y_te, proba)
            m.update({"fold": fold, "model": name})
            results[name].append(m)

            print(f"    {name:<10}"
                  f"  AUC={m['AUC']:.4f}"
                  f"  OA={m['OA']:.4f}"
                  f"  RMSE={m['RMSE']:.4f}"
                  f"  Kappa={m['Kappa']:.4f}")

    return results, oof_preds


# ── Train final models ────────────────────────────────────────────────────
def train_final(X, y, feature_cols, models):
    print(f"\n{'='*55}")
    print("  Training final models on ALL data")
    print(f"{'='*55}")

    final  = {}
    X_feat = X[feature_cols]

    for name, model in models.items():
        print(f"  {name}...", end=" ", flush=True)
        if name == "XGBoost":
            model.fit(X_feat, y, verbose=False)
        else:
            model.fit(X_feat, y)
        final[name] = model
        out = C.MODELS_DIR / f"{name.lower()}_final.pkl"
        with open(out, "wb") as f:
            pickle.dump(model, f)
        print(f"saved  ({out.name})")

    return final


# ── SHAP ──────────────────────────────────────────────────────────────────
def run_shap(final_models, X, feature_cols, n_samples=500):
    print(f"\n{'='*55}")
    print("  SHAP Analysis")
    print(f"{'='*55}")

    import matplotlib.pyplot as plt

    rng    = np.random.default_rng(C.RANDOM_SEED)
    X_feat = X[feature_cols]
    idx    = rng.choice(len(X_feat),
                        size=min(n_samples, len(X_feat)),
                        replace=False)
    X_s    = X_feat.iloc[idx]

    for name, model in final_models.items():
        print(f"  {name}...", end=" ", flush=True)
        try:
            exp = shap.TreeExplainer(model)
            sv  = exp.shap_values(X_s)
            # Handle all SHAP output shapes:
            # list of arrays  → take index 1 (class 1)
            # 3D array (n,f,2) → take [:,:,1]  (class 1)
            # 2D array (n,f)   → use as-is
            if isinstance(sv, list):
                sv = sv[1]
            elif hasattr(sv, "ndim") and sv.ndim == 3:
                sv = sv[:, :, 1]
            print("done")
        except Exception as e:
            print(f"skipped ({e})")
            continue

        # Beeswarm
        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv, X_s, plot_type="dot",
                          show=False, max_display=15)
        plt.title(f"SHAP Beeswarm — {name}  (Kerala 2018)")
        plt.tight_layout()
        plt.savefig(C.SHAP_DIR / f"shap_beeswarm_{name.lower()}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # Bar
        plt.figure(figsize=(9, 6))
        shap.summary_plot(sv, X_s, plot_type="bar",
                          show=False, max_display=15)
        plt.title(f"SHAP Importance — {name}")
        plt.tight_layout()
        plt.savefig(C.SHAP_DIR / f"shap_bar_{name.lower()}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # CSV
        pd.DataFrame(sv, columns=feature_cols[:sv.shape[1]]
                     ).to_csv(
            C.SHAP_DIR / f"shap_values_{name.lower()}.csv",
            index=False)

        print(f"    Saved beeswarm + bar + CSV for {name}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("\n" + "█"*55)
    print("  M-4: XGBoost / GBM / RF — Kerala 2018")
    print("  Reference: Sahin (2020) SN Applied Sciences")
    print("█"*55)

    # 1. Load hyperparameters
    rf_p, gbm_p, xgb_p = C.get_model_params()

    # 2. Load data
    X, y, feature_cols = load_data()

    # 3. Build models
    models = {
        "RF":      RandomForestClassifier(**rf_p),
        "GBM":     GradientBoostingClassifier(**gbm_p),
        "XGBoost": xgb.XGBClassifier(**xgb_p),
    }

    # 4. Spatial CV
    cv_results, oof_preds = run_cv(X, y, feature_cols, models)

    # 5. Summary table (Table 5 — Sahin 2020)
    print(f"\n{'='*55}")
    print("  RESULTS SUMMARY")
    print(f"{'='*55}")

    summary_tables = {}
    table_rows     = []

    for name, folds in cv_results.items():
        s = aggregate_cv_metrics(folds)
        summary_tables[name] = s
        print(f"\n  {name}:\n{s.to_string()}")
        table_rows.append({
            "Model": name,
            **{col: f"{s.loc[col,'Mean']:.4f}±{s.loc[col,'Std']:.4f}"
               for col in ["OA","AUC","RMSE","Kappa"]},
        })

    pd.DataFrame(table_rows).to_csv(
        C.METRICS_DIR/"summary_table.csv", index=False)

    # Per-fold CSV
    rows = []
    for name, folds in cv_results.items():
        for f in folds:
            rows.append({"model": name, **f})
    pd.DataFrame(rows).to_csv(
        C.METRICS_DIR/"cv_fold_results.csv", index=False)

    # 6. Wilcoxon table (Table 6 — Sahin 2020)
    model_aucs  = {n:[m["AUC"] for m in cv_results[n]]
                   for n in cv_results}
    wilcoxon_df = run_wilcoxon_table(model_aucs)
    wilcoxon_df.to_csv(C.METRICS_DIR/"wilcoxon_table.csv", index=False)
    print(f"\n  Wilcoxon Table:\n{wilcoxon_df.to_string(index=False)}")

    # 7. Figures
    print(f"\n{'='*55}")
    print("  Generating figures")
    print(f"{'='*55}")

    p = plot_roc_curves(oof_preds, y, C.FIGURES_DIR)
    print(f"  {p.name}")
    p = plot_auc_boxplot(cv_results, C.FIGURES_DIR)
    print(f"  {p.name}")
    p = plot_oa_bar(summary_tables, C.FIGURES_DIR)
    print(f"  {p.name}")
    p = plot_spatial_cv_blocks(X, y, C.N_CV_FOLDS, C.FIGURES_DIR)
    print(f"  {p.name}")

    # 8. Final models
    final_models = train_final(X, y, feature_cols, models)

    # 9. Feature importance
    p = plot_feature_importance(
        final_models, feature_cols, C.FIGURES_DIR)
    print(f"  {p.name}")

    # 10. SHAP
    run_shap(final_models, X, feature_cols)

    print("\n" + "█"*55)
    print("  TRAINING COMPLETE")
    print(f"  Metrics  → {C.METRICS_DIR}")
    print(f"  Models   → {C.MODELS_DIR}")
    print(f"  Figures  → {C.FIGURES_DIR}")
    print(f"  SHAP     → {C.SHAP_DIR}")
    print("█"*55)


if __name__ == "__main__":
    main()
