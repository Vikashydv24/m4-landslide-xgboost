#!/usr/bin/env python3
"""
02_prepare_samples.py
Build the training feature matrix for M-4 LSM:
  1. Load Kerala 2018 landslide inventory (4728 points, label=1)
  2. Generate equal-count non-landslide background points (label=0)
  3. Extract 12 terrain predictor values at all points from raster stack
  4. Encode land-use (LU_2018) with one-hot encoding
  5. Save as data/processed/features_all.parquet

Output: data/processed/features_all.parquet
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from shapely.geometry import Point

import config as C

# ── Config ────────────────────────────────────────────────────────────────
RAW_SHP    = C.DATA_RAW / "Kerela_landslide.shp"
OUT_DIR    = C.DATA_PROC
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEST, SOUTH, EAST, NORTH = C.KERALA_BBOX
MIN_SLOPE_FOR_BACKGROUND  = 5.0    # background points must have slope > 5°
MIN_DIST_FROM_LANDSLIDE   = 0.005  # ~0.5 km buffer around landslides
RANDOM_SEED               = C.RANDOM_SEED


def load_landslides():
    print(f"\n[1/5] Loading landslide inventory")
    gdf = gpd.read_file(RAW_SHP)
    gdf = gdf.dropna(subset=["POINT_X", "POINT_Y"])
    gdf["label"] = 1
    print(f"  Landslide points : {len(gdf):,}")
    print(f"  Type breakdown   : {gdf['Type_of_sl'].value_counts().to_dict()}")
    return gdf


def generate_background(n_points, landslide_gdf):
    """
    Generate n_points random background locations within Kerala.
    Constraints:
      - Not within MIN_DIST buffer of any landslide
      - Slope > MIN_SLOPE_FOR_BACKGROUND (only if slope raster exists)
    """
    print(f"\n[2/5] Generating {n_points:,} background (non-landslide) points")

    rng      = np.random.default_rng(RANDOM_SEED)
    ls_union = landslide_gdf.geometry.unary_union.buffer(
        MIN_DIST_FROM_LANDSLIDE)

    # Load slope raster for constraint (optional)
    slope_src = None
    slope_path = C.PRED_DIR / "slope.tif"
    if slope_path.exists():
        slope_src = rasterio.open(slope_path)
        print(f"  Using slope constraint (>{MIN_SLOPE_FOR_BACKGROUND}°)")
    else:
        print("  No slope raster found — skipping slope constraint")

    bg_lons, bg_lats = [], []
    attempts = 0
    max_attempts = n_points * 30

    while len(bg_lons) < n_points and attempts < max_attempts:
        batch_n  = min(20000, (n_points - len(bg_lons)) * 10)
        rand_lon = rng.uniform(WEST,  EAST,  batch_n)
        rand_lat = rng.uniform(SOUTH, NORTH, batch_n)

        for lon, lat in zip(rand_lon, rand_lat):
            if len(bg_lons) >= n_points:
                break
            pt = Point(lon, lat)
            if ls_union.contains(pt):
                continue
            # Check slope constraint
            if slope_src is not None:
                try:
                    slope_val = list(slope_src.sample([(lon, lat)]))[0][0]
                    if slope_val < MIN_SLOPE_FOR_BACKGROUND:
                        continue
                except Exception:
                    pass
            bg_lons.append(lon)
            bg_lats.append(lat)
        attempts += batch_n

    if slope_src:
        slope_src.close()

    bg_df = pd.DataFrame({
        "POINT_X":    bg_lons,
        "POINT_Y":    bg_lats,
        "label":      0,
        "LU_2018":    "UNKNOWN",
        "Type_of_sl": "NONE",
        "District":   "NA",
    })
    bg_gdf = gpd.GeoDataFrame(
        bg_df,
        geometry=gpd.points_from_xy(bg_lons, bg_lats),
        crs="EPSG:4326"
    )
    print(f"  Background points generated: {len(bg_gdf):,}")
    return bg_gdf


def extract_raster_values(gdf):
    """Extract terrain predictor values at each sample point."""
    print(f"\n[3/5] Extracting raster values at {len(gdf):,} points")

    coords    = list(zip(gdf["POINT_X"], gdf["POINT_Y"]))
    feat_dict = {}

    for name in C.PREDICTOR_NAMES:
        rpath = C.PRED_DIR / f"{name}.tif"
        if not rpath.exists():
            print(f"  WARNING: {name}.tif not found — filling with NaN")
            feat_dict[name] = np.full(len(gdf), np.nan, dtype=np.float32)
            continue

        with rasterio.open(rpath) as src:
            vals   = [v[0] for v in src.sample(coords)]
            nodata = src.nodata

        arr = np.array(vals, dtype=np.float32)
        if nodata is not None:
            arr[arr == nodata] = np.nan

        n_valid = int(np.sum(np.isfinite(arr)))
        print(f"  {name:<15}  valid={n_valid}/{len(gdf)}"
              f"  mean={np.nanmean(arr):.3f}")
        feat_dict[name] = arr

    return pd.DataFrame(feat_dict)


def encode_land_use(df):
    """One-hot encode LU_2018. Background gets LU_UNKNOWN (not LU_BACKGROUND)."""
    print(f"\n[4/5] Encoding land-use")

    # Background points have LU_2018='UNKNOWN' — NOT a label-leaking value
    # Keep top-10 real LU classes from landslide inventory
    ls_mask   = df["label"] == 1
    top10     = (df.loc[ls_mask, "LU_2018"]
                   .value_counts()
                   .head(10)
                   .index.tolist())
    top10.append("UNKNOWN")

    df["LU_2018"] = df["LU_2018"].where(
        df["LU_2018"].isin(top10), other="OTHER")

    dummies = pd.get_dummies(df["LU_2018"], prefix="LU", dtype=np.int8)
    df      = pd.concat([df.drop(columns=["LU_2018"]), dummies], axis=1)

    print(f"  LU classes: {sorted([c for c in df.columns if c.startswith('LU_')])}")
    return df


def main():
    print("="*55)
    print("  Preparing M-4 training dataset")
    print("="*55)

    # 1. Landslide points
    ls_gdf = load_landslides()

    # 2. Background points
    n_bg   = len(ls_gdf)
    bg_gdf = generate_background(n_bg, ls_gdf)

    # 3. Combine
    keep_cols = ["POINT_X", "POINT_Y", "label", "LU_2018",
                 "Type_of_sl", "District"]
    combined  = pd.concat([
        ls_gdf[keep_cols].copy(),
        bg_gdf[keep_cols].copy(),
    ], ignore_index=True)

    # 4. Extract raster features
    raster_df = extract_raster_values(
        gpd.GeoDataFrame(
            combined,
            geometry=gpd.points_from_xy(
                combined["POINT_X"], combined["POINT_Y"]),
            crs="EPSG:4326"))

    combined = pd.concat(
        [combined.reset_index(drop=True), raster_df], axis=1)

    # 5. Encode land-use
    combined = encode_land_use(combined)

    # 6. Drop rows where ALL raster features are NaN
    raster_cols = [c for c in C.PREDICTOR_NAMES if c in combined.columns]
    before = len(combined)
    combined.dropna(subset=raster_cols, how="all", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    dropped = before - len(combined)
    if dropped > 0:
        print(f"\n  Dropped {dropped} rows with all-NaN rasters "
              f"(outside DEM coverage)")

    # 7. Save
    out_parquet = OUT_DIR / "features_all.parquet"
    out_csv     = OUT_DIR / "features_all.csv"
    combined.to_parquet(out_parquet, index=False)
    combined.to_csv(out_csv, index=False)

    print(f"\n{'='*55}")
    print("  DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"  Total rows     : {len(combined):,}")
    print(f"  Landslide (1)  : {(combined['label']==1).sum():,}")
    print(f"  Background (0) : {(combined['label']==0).sum():,}")
    print(f"  Total features : {len(combined.columns)}")
    print(f"\n  Saved to:")
    print(f"    {out_parquet}")
    print(f"    {out_csv}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
