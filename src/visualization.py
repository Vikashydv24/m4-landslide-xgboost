"""
Visualization for M-4 LSM results.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

COLORS = {"RF": "#2196F3", "GBM": "#FF9800", "XGBoost": "#F44336"}


def plot_roc_curves(oof_preds, y_true, out_dir):
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(7, 7))
    for name, preds in oof_preds.items():
        valid = ~np.isnan(preds)
        fpr, tpr, _ = roc_curve(y_true[valid], preds[valid])
        auc = roc_auc_score(y_true[valid], preds[valid])
        ax.plot(fpr, tpr, color=COLORS.get(name, "grey"),
                lw=2.5, label=f"{name}  (AUC={auc:.4f})")
    ax.plot([0,1],[0,1],"k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Spatial Block CV (Kerala 2018)")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    out = Path(out_dir) / "roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_auc_boxplot(all_results, out_dir):
    names  = list(all_results.keys())
    data   = [[m["AUC"] for m in all_results[n]] for n in names]
    colors = [COLORS.get(n, "#9E9E9E") for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=names, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.axhline(0.80, color="red", linestyle="--", lw=1.5,
               alpha=0.7, label="AUC=0.80 reference")
    ax.set_ylabel("AUC"); ax.set_ylim([0.5, 1.0])
    ax.set_title("AUC Across Spatial CV Folds")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    out = Path(out_dir) / "auc_boxplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_oa_bar(summary_tables, out_dir):
    names   = list(summary_tables.keys())
    means   = [summary_tables[n].loc["OA","Mean"] for n in names]
    stds    = [summary_tables[n].loc["OA","Std"]  for n in names]
    colors  = [COLORS.get(n, "#9E9E9E") for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=stds, color=colors,
                  alpha=0.85, capsize=6, edgecolor="black")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.007,
                f"{val*100:.2f}%", ha="center",
                fontsize=10, fontweight="bold")
    ax.set_ylim([0.6, 1.02]); ax.set_ylabel("Overall Accuracy")
    ax.set_title("Overall Accuracy — Kerala 2018 (Spatial Block CV)")
    ax.grid(axis="y", alpha=0.3)
    out = Path(out_dir) / "oa_barplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_importance(models, feature_cols, out_dir, top_n=15):
    # Clamp top_n to number of features available
    top_n = min(top_n, len(feature_cols))
    n     = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1: axes = [axes]
    for ax, (name, model) in zip(axes, models.items()):
        imp  = model.feature_importances_
        idx  = np.argsort(imp)[::-1][:top_n]
        vals = imp[idx][::-1]
        lbls = [feature_cols[i] for i in idx][::-1]
        ax.barh(range(len(vals)), vals,
                color=COLORS.get(name,"#9E9E9E"), alpha=0.8,
                edgecolor="black", lw=0.5)
        ax.set_yticks(range(len(lbls)))
        ax.set_yticklabels(lbls, fontsize=9)
        ax.set_xlabel("Importance"); ax.set_title(f"{name}")
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("Feature Importance — RF / GBM / XGBoost", fontsize=13)
    plt.tight_layout()
    out = Path(out_dir) / "feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_spatial_cv_blocks(X, y, n_splits, out_dir,
                           lat_col="POINT_Y", lon_col="POINT_X"):
    import matplotlib.cm as cm
    fig, ax = plt.subplots(figsize=(7, 9))
    lat   = X[lat_col].values
    edges = np.linspace(lat.min(), lat.max(), n_splits + 1)
    ids   = np.clip(np.digitize(lat, edges[:-1]) - 1, 0, n_splits-1)
    cmap  = cm.get_cmap("tab10", n_splits)
    bg    = y == 0
    ax.scatter(X.loc[bg, lon_col], X.loc[bg, lat_col],
               c="lightgrey", s=2, alpha=0.3)
    for b in range(n_splits):
        m = (y == 1) & (ids == b)
        ax.scatter(X.loc[m, lon_col], X.loc[m, lat_col],
                   color=cmap(b), s=8, alpha=0.8,
                   label=f"Block {b+1} (n={m.sum()})")
    for e in edges:
        ax.axhline(e, color="red", linestyle="--", lw=1.0, alpha=0.6)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial Block CV — {n_splits} bands")
    ax.legend(fontsize=8, loc="upper left", markerscale=2)
    ax.grid(alpha=0.2)
    out = Path(out_dir) / "spatial_cv_blocks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
