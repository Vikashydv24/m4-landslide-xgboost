#!/usr/bin/env python3
"""
05_produce_lsm_map.py  (v2 — with land mask)
Produce LSM for Kerala. Sea/ocean pixels are masked out
using elevation > 1m as land indicator.
"""
import sys, os, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import config as C

# ── Settings ──────────────────────────────────────────────────────────────
WEST, SOUTH, EAST, NORTH = C.KERALA_BBOX
GRID_RES   = 0.005
CHUNK_SIZE = 50_000
MODELS_TO_MAP = ["RF", "GBM", "XGBoost"]

FEATURE_COLS = [
    "elevation", "slope", "aspect",
    "plan_curv", "prof_curv",
    "twi", "tri",
    "spi", "sti",
    "flow_acc", "dist_river",
]
CLIP_COLS = ["plan_curv", "prof_curv", "sti", "spi", "flow_acc", "dist_river"]

C.MAPS_DIR.mkdir(parents=True, exist_ok=True)
C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Build grid ─────────────────────────────────────────────────────────────
def build_grid():
    lons = np.arange(WEST, EAST, GRID_RES)
    lats = np.arange(SOUTH, NORTH, GRID_RES)
    lon_g, lat_g = np.meshgrid(lons, lats)
    df = pd.DataFrame({
        "lon": lon_g.flatten(),
        "lat": lat_g.flatten(),
    })
    print(f"  Grid: {len(lons)} × {len(lats)} = {len(df):,} points")
    return df, lons, lats


# ── Extract features ───────────────────────────────────────────────────────
def extract_features(grid_df):
    print(f"  Extracting {len(FEATURE_COLS)} predictors...")
    coords    = list(zip(grid_df["lon"], grid_df["lat"]))
    feat_dict = {}

    for name in tqdm(FEATURE_COLS, desc="  Rasters"):
        rpath = C.PRED_DIR / f"{name}.tif"
        if not rpath.exists():
            feat_dict[name] = np.full(len(coords), np.nan, dtype=np.float32)
            continue
        vals = []
        with rasterio.open(rpath) as src:
            nodata = src.nodata
            for s in range(0, len(coords), CHUNK_SIZE):
                batch = coords[s:s + CHUNK_SIZE]
                vals.extend([v[0] for v in src.sample(batch)])
        arr = np.array(vals, dtype=np.float32)
        if nodata is not None:
            arr[arr == nodata] = np.nan
        feat_dict[name] = arr

    feat_df = pd.DataFrame(feat_dict)

    # ── Land mask: elevation > 1m means land ─────────────────────────────
    # Sea pixels have elevation = 0 or NaN → mask them out
    land_mask = (feat_df["elevation"].notna() &
                 (feat_df["elevation"] > 1.0))
    print(f"  Land pixels : {land_mask.sum():,} / {len(feat_df):,} "
          f"({land_mask.mean()*100:.1f}%)")

    # Apply clipping only on land pixels
    for col in CLIP_COLS:
        if col in feat_df.columns:
            land_vals = feat_df.loc[land_mask, col]
            lo = land_vals.quantile(0.02)
            hi = land_vals.quantile(0.98)
            feat_df[col] = feat_df[col].clip(lo, hi)

    # Fill NaN with median of land pixels only
    for col in feat_df.columns:
        median_val = feat_df.loc[land_mask, col].median()
        feat_df[col].fillna(median_val, inplace=True)

    return feat_df, land_mask.values


# ── Load model ─────────────────────────────────────────────────────────────
def load_model(name):
    path = C.MODELS_DIR / f"{name.lower()}_final.pkl"
    if not path.exists():
        print(f"  Not found: {path}")
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"  Loaded: {path.name}")
    return model


# ── Save rasters ───────────────────────────────────────────────────────────
def save_rasters(proba_full, land_mask, lons, lats, name):
    n_rows, n_cols = len(lats), len(lons)
    transform = from_bounds(
        lons[0], lats[0], lons[-1], lats[-1], n_cols, n_rows)

    # Mask sea pixels
    proba_masked = proba_full.copy()
    proba_masked[~land_mask] = np.nan

    prob_2d  = proba_masked.reshape(n_rows, n_cols)

    # Classification on LAND pixels only
    # Use fixed probability thresholds for interpretable classes
    # (quantile fails when most predictions cluster near 0 or 1)
    land_probs = proba_masked[land_mask & np.isfinite(proba_masked)]
    q_raw = np.quantile(land_probs, [0.2, 0.4, 0.6, 0.8])

    # If quantile thresholds are too compressed (range < 0.1), use fixed
    if (q_raw[-1] - q_raw[0]) < 0.1:
        q = np.array([0.20, 0.35, 0.50, 0.65])
        print("  Using fixed thresholds (quantile range too compressed)")
    else:
        q = q_raw

    print(f"\n  Class thresholds: "
          f"VL<{q[0]:.3f}  L<{q[1]:.3f}  "
          f"M<{q[2]:.3f}  H<{q[3]:.3f}  VH")

    class_2d = np.zeros((n_rows, n_cols), dtype=np.uint8)
    flat = prob_2d.flatten()
    cls_flat = np.zeros(len(flat), dtype=np.uint8)
    land_flat = land_mask.reshape(n_rows, n_cols).flatten()

    cls_flat[land_flat] = np.digitize(
        flat[land_flat], q) + 1            # 1–5
    cls_flat[cls_flat > 5] = 5
    class_2d = cls_flat.reshape(n_rows, n_cols)

    profile = {
        "driver":    "GTiff",
        "width":     n_cols,
        "height":    n_rows,
        "count":     1,
        "crs":       CRS.from_epsg(4326),
        "transform": transform,
        "compress":  "lzw",
        "nodata":    -9999,
    }

    # Probability raster
    prob_out = C.MAPS_DIR / f"lsm_{name.lower()}_prob.tif"
    arr = np.where(np.isnan(prob_2d), -9999, prob_2d).astype(np.float32)
    with rasterio.open(prob_out, "w", **{**profile, "dtype":"float32"}) as d:
        d.write(arr, 1)
    print(f"  Saved: {prob_out.name}")

    # Class raster
    cls_out = C.MAPS_DIR / f"lsm_{name.lower()}_class.tif"
    with rasterio.open(cls_out, "w",
                       **{**profile, "dtype":"uint8", "nodata":0}) as d:
        d.write(class_2d, 1)
    print(f"  Saved: {cls_out.name}")

    # Distribution (land pixels only)
    labels = ["Very Low","Low","Moderate","High","Very High"]
    print("\n  Susceptibility distribution (land only):")
    for i, lbl in enumerate(labels, start=1):
        n   = (class_2d[land_mask.reshape(n_rows,n_cols)] == i).sum()
        pct = n / land_mask.sum() * 100
        print(f"    {lbl:<12}: {pct:.1f}%")

    return prob_2d, class_2d, q, land_mask.reshape(n_rows, n_cols)


# ── Visualisation ──────────────────────────────────────────────────────────
def plot_map(prob_2d, class_2d, land_mask_2d, lons, lats, name, q):
    labels  = ["Very Low","Low","Moderate","High","Very High"]
    colors  = ["#1A9850","#91CF60","#FFFFBF","#FC8D59","#D73027"]

    lon_g, lat_g = np.meshgrid(lons, lats)
    land_flat    = land_mask_2d.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # ── Left: probability (land pixels only) ──────────────────────────────
    prob_flat = prob_2d.flatten()
    sc = axes[0].scatter(
        lon_g.flatten()[land_flat],
        lat_g.flatten()[land_flat],
        c=prob_flat[land_flat],
        cmap="RdYlGn_r", s=1, alpha=0.8, vmin=0, vmax=1)
    plt.colorbar(sc, ax=axes[0], label="Landslide Probability", shrink=0.8)
    axes[0].set_title(f"{name} — Probability Map", fontsize=13)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].grid(alpha=0.2)
    axes[0].set_facecolor("#d0e8f0")   # sea colour

    # ── Right: 5-class (land pixels only) ────────────────────────────────
    axes[1].set_facecolor("#d0e8f0")   # sea colour
    cls_flat = class_2d.flatten()

    for cls_id, (lbl, col) in enumerate(zip(labels, colors), start=1):
        m = land_flat & (cls_flat == cls_id)
        if m.sum() > 0:
            axes[1].scatter(
                lon_g.flatten()[m],
                lat_g.flatten()[m],
                c=col, s=1, alpha=0.8)

    # Clean legend
    patches = [Patch(facecolor=c, label=l)
               for c, l in zip(colors, labels)]
    axes[1].legend(handles=patches, title="Susceptibility",
                   loc="upper left", fontsize=9,
                   framealpha=0.9)

    # Clean percentage table — top right
    pct_lines = []
    for i, lbl in enumerate(labels, start=1):
        n   = (cls_flat[land_flat] == i).sum()
        pct = n / land_flat.sum() * 100
        pct_lines.append(f"{lbl}: {pct:.1f}%")

    axes[1].text(
        0.98, 0.98, "\n".join(pct_lines),
        transform=axes[1].transAxes,
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="white", alpha=0.85),
        family="monospace")

    axes[1].set_title(f"{name} — 5-Class Susceptibility Map", fontsize=13)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].grid(alpha=0.2)

    plt.suptitle(
        f"Landslide Susceptibility Map — Kerala 2018\n"
        f"Model: {name}  |  Spatial Block CV  |  "
        f"AUC (RF/GBM/XGB): 0.886 / 0.882 / 0.883",
        fontsize=12)
    plt.tight_layout()

    out = C.FIGURES_DIR / f"lsm_map_{name.lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out.name}")
    return out


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  LSM MAP PRODUCTION — Kerala 2018  (v2)")
    print("="*55)

    missing = [n for n in FEATURE_COLS
               if not (C.PRED_DIR / f"{n}.tif").exists()]
    if missing:
        print(f"ERROR: Missing rasters: {missing}")
        return

    print("\n[1/3] Building prediction grid")
    grid_df, lons, lats = build_grid()

    print("\n[2/3] Extracting terrain features + land mask")
    feat_df, land_mask = extract_features(grid_df)

    for name in MODELS_TO_MAP:
        print(f"\n[3/3] ── {name} ──────────────────────────")
        model = load_model(name)
        if model is None:
            continue

        # GBM (sklearn) does not accept NaN — fill with column median
        X_pred = feat_df[FEATURE_COLS].copy()
        X_pred.fillna(X_pred.median(), inplace=True)
        # Final safety check — replace any remaining NaN with 0
        X_pred.fillna(0, inplace=True)
        proba = model.predict_proba(X_pred.values)[:, 1]

        prob_2d, class_2d, q, lm2d = save_rasters(
            proba, land_mask, lons, lats, name)
        plot_map(prob_2d, class_2d, lm2d, lons, lats, name, q)

    print("\n" + "="*55)
    print("  COMPLETE")
    print(f"  Rasters → {C.MAPS_DIR}")
    print(f"  Figures → {C.FIGURES_DIR}")
    print("="*55)


if __name__ == "__main__":
    main()
