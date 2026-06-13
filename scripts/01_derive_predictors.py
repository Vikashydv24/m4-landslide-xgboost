#!/usr/bin/env python3
"""
01_derive_predictors.py
Derive 12 terrain predictor rasters from SRTM DEM for Kerala LSM.
Geotransform is passed to richdem so curvature values are correct.

Output: data/processed/predictors/*.tif
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import rasterio
from pathlib import Path
from scipy.ndimage import uniform_filter, distance_transform_edt

try:
    import richdem as rd
except ImportError:
    os.system("pip install richdem")
    import richdem as rd

# ── Paths ─────────────────────────────────────────────────────────────────
DEM_PATH = Path("data/dem/kerala_srtm30.tif")
OUT_DIR  = Path("data/processed/predictors")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_SIZE_M = 30.0   # SRTM 30m resolution in metres


def load_dem():
    """Load DEM and set proper geotransform on richdem array."""
    print("Loading DEM...")
    with rasterio.open(DEM_PATH) as src:
        profile   = src.profile.copy()
        arr       = src.read(1).astype(np.float32)
        nodata    = src.nodata if src.nodata else -9999
        transform = src.transform

    arr[arr == nodata] = np.nan

    dem_rd = rd.rdarray(arr, no_data=np.nan)

    # ── Critical fix: set geotransform so richdem uses correct cell size ──
    # Without this, richdem assumes 1x1 pixels and inflates curvature ~900x
    # Format: [west, cell_width, row_rotation, north, col_rotation, cell_height]
    dem_rd.geotransform = [
        transform.c,   # west  (x origin)
        transform.a,   # cell width  in degrees (~0.000278 for SRTM 30m)
        transform.b,   # row rotation (usually 0)
        transform.f,   # north (y origin)
        transform.d,   # col rotation (usually 0)
        transform.e,   # cell height in degrees (negative)
    ]

    print(f"  Shape     : {arr.shape}")
    print(f"  Elev range: {np.nanmin(arr):.0f} – {np.nanmax(arr):.0f} m")
    print(f"  Cell size : {transform.a:.6f} deg "
          f"(~{abs(transform.a) * 111320:.1f} m)")
    return dem_rd, arr, profile


def save_raster(array, name, profile, nodata=-9999):
    """Save numpy array as compressed GeoTIFF."""
    out = OUT_DIR / f"{name}.tif"
    p   = profile.copy()
    p.update(dtype="float32", count=1, nodata=nodata, compress="lzw")
    arr = array.astype(np.float32)
    arr[np.isnan(arr)] = nodata
    with rasterio.open(out, "w", **p) as dst:
        dst.write(arr, 1)
    valid = arr[arr != nodata]
    print(f"  Saved: {name:<15}  "
          f"min={np.nanmin(valid):.3f}  "
          f"max={np.nanmax(valid):.3f}  "
          f"mean={np.nanmean(valid):.3f}")
    return out


def main():
    print("\n" + "="*55)
    print("  Deriving 12 predictor rasters from DEM")
    print("="*55)

    if not DEM_PATH.exists():
        print(f"\nERROR: DEM not found at {DEM_PATH}")
        print("Run: python scripts/00_download_dem.py first.")
        sys.exit(1)

    dem_rd, dem_np, profile = load_dem()
    print()

    # ── 1. Elevation ──────────────────────────────────────────────────────
    print("[ 1/12] Elevation")
    save_raster(dem_np, "elevation", profile)

    # ── 2. Slope (degrees) ────────────────────────────────────────────────
    print("[ 2/12] Slope")
    slope_rd = rd.TerrainAttribute(dem_rd, attrib="slope_degrees")
    slope_np = np.array(slope_rd, dtype=np.float32)
    save_raster(slope_np, "slope", profile)

    # ── 3. Aspect (degrees) ───────────────────────────────────────────────
    print("[ 3/12] Aspect")
    aspect = rd.TerrainAttribute(dem_rd, attrib="aspect")
    save_raster(np.array(aspect), "aspect", profile)

    # ── 4. Plan Curvature ─────────────────────────────────────────────────
    print("[ 4/12] Plan Curvature")
    plan = rd.TerrainAttribute(dem_rd, attrib="planform_curvature")
    save_raster(np.array(plan), "plan_curv", profile)

    # ── 5. Profile Curvature ──────────────────────────────────────────────
    print("[ 5/12] Profile Curvature")
    prof = rd.TerrainAttribute(dem_rd, attrib="profile_curvature")
    save_raster(np.array(prof), "prof_curv", profile)

    # ── 6. TRI — computed manually (richdem has no roughness attribute) ───
    print("[ 6/12] TRI  (Riley et al. 1999)")
    tmp     = dem_np.copy()
    tmp[np.isnan(tmp)] = 0.0
    tri_sq  = np.zeros_like(tmp)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        shifted  = np.roll(np.roll(tmp, dy, axis=0), dx, axis=1)
        tri_sq  += (tmp - shifted) ** 2
    tri = np.sqrt(tri_sq / 8.0)
    tri[np.isnan(dem_np)] = np.nan
    save_raster(tri, "tri", profile)

    # ── 7. TPI — elevation minus local mean ───────────────────────────────
    print("[ 7/12] TPI")
    mean_elev = uniform_filter(dem_np, size=3)
    tpi       = dem_np - mean_elev
    save_raster(tpi, "tpi", profile)

    # ── 8. Flow Accumulation ──────────────────────────────────────────────
    print("[ 8/12] Flow Accumulation")
    dem_filled = rd.FillDepressions(dem_rd, epsilon=True)
    flow_acc   = rd.FlowAccumulation(dem_filled, method="D8")
    fa_np      = np.array(flow_acc, dtype=np.float32)
    save_raster(fa_np, "flow_acc", profile)

    # ── 9. TWI — Topographic Wetness Index ────────────────────────────────
    print("[ 9/12] TWI")
    slope_rad = np.deg2rad(slope_np)
    slope_rad = np.where(slope_rad < 0.001, 0.001, slope_rad)
    upslope   = np.maximum(fa_np * CELL_SIZE_M ** 2, 1.0)
    twi       = np.log(upslope / np.tan(slope_rad))
    save_raster(twi, "twi", profile)

    # ── 10. SPI — Stream Power Index ──────────────────────────────────────
    print("[10/12] SPI")
    spi = np.log1p(upslope * np.tan(slope_rad))
    save_raster(spi, "spi", profile)

    # ── 11. STI — Sediment Transport Index ───────────────────────────────
    print("[11/12] STI")
    sti = ((upslope / 22.13) ** 0.6) * ((np.sin(slope_rad) / 0.0896) ** 1.3)
    p99 = np.nanpercentile(sti[np.isfinite(sti)], 99)
    sti = np.clip(sti, 0, p99)
    save_raster(sti, "sti", profile)

    # ── 12. Distance to River ─────────────────────────────────────────────
    print("[12/12] Distance to River")
    river_mask = fa_np > 1000
    dist_px    = distance_transform_edt(~river_mask)
    dist_m     = dist_px * CELL_SIZE_M
    save_raster(dist_m, "dist_river", profile)

    print("\n" + "="*55)
    print(f"  All 12 rasters saved to: {OUT_DIR}")
    for t in sorted(OUT_DIR.glob("*.tif")):
        print(f"    {t.name}")
    print("="*55)


if __name__ == "__main__":
    main()
