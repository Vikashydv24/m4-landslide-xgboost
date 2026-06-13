#!/usr/bin/env python3
"""
Download SRTM 30m DEM for Kerala.
Tries 3 methods in order. Stops at first success.
Output: data/dem/kerala_srtm30.tif
"""
import os, sys, subprocess
from pathlib import Path

OUT_DIR  = Path("data/dem")
OUT_FILE = OUT_DIR / "kerala_srtm30.tif"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEST, SOUTH, EAST, NORTH = 74.8, 8.2, 77.6, 13.0

def check_valid(path):
    if not path.exists():
        return False
    size_mb = path.stat().st_size / 1e6
    if size_mb < 1.0:
        print(f"  File too small ({size_mb:.1f} MB) — likely failed.")
        return False
    print(f"  File size: {size_mb:.1f} MB  — looks good.")
    return True

def method_eio():
    print("\n[Method 1] elevation package...")
    try:
        import elevation
        elevation.clip(
            bounds=(WEST, SOUTH, EAST, NORTH),
            output=str(OUT_FILE.resolve()),
            product="SRTM3"
        )
        return check_valid(OUT_FILE)
    except Exception as e:
        print(f"  Failed: {e}")
        return False

def method_gdal():
    print("\n[Method 2] GDAL + OpenTopography tiles...")
    try:
        tile_files = []
        base = ("https://opentopography.s3.sdsc.edu/dataspace/"
                "OTDS.012022.4326.1/raster/"
                "Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM.tif")

        for lat in range(int(SOUTH), int(NORTH)+1):
            for lon in range(int(WEST), int(EAST)+1):
                url  = base.format(lat=lat, lon=lon)
                vsip = f"/vsicurl/{url}"
                tmp  = OUT_DIR / f"tile_{lat}_{lon}.tif"
                print(f"  Tile N{lat:02d}E{lon:03d}...", end=" ", flush=True)
                r = subprocess.run(
                    ["gdal_translate", "-q", vsip, str(tmp)],
                    capture_output=True, timeout=60
                )
                if r.returncode == 0 and tmp.exists():
                    print("ok")
                    tile_files.append(str(tmp))
                else:
                    print("skip")

        if not tile_files:
            return False

        vrt = OUT_DIR / "merged.vrt"
        subprocess.run(["gdalbuildvrt", str(vrt)] + tile_files,
                       check=True, capture_output=True)
        subprocess.run(["gdal_translate", "-q", "-of", "GTiff",
                        "-co", "COMPRESS=LZW",
                        str(vrt), str(OUT_FILE)], check=True)
        for t in tile_files:
            Path(t).unlink(missing_ok=True)
        vrt.unlink(missing_ok=True)
        return check_valid(OUT_FILE)
    except Exception as e:
        print(f"  Failed: {e}")
        return False

def method_manual():
    print("\n[Method 3] All automated methods failed.")
    print("  Manual download required.")
    print("  Go to: https://dwtkns.com/srtm30m/")
    print("  Download tiles: N08E075 N08E076 N09E075 N09E076")
    print("                  N10E075 N10E076 N11E075 N11E076 N12E075 N12E076")
    print("  Place .hgt files in: data/dem/")
    print("  Then run: python scripts/00_download_dem.py")
    return False

def main():
    print("="*50)
    print("  DEM Download for Kerala")
    print("="*50)

    if check_valid(OUT_FILE):
        print(f"\n  DEM already exists. Skipping download.")
        return

    for method in [method_eio, method_gdal, method_manual]:
        if method():
            print("\n  DEM download complete!")

            import rasterio
            with rasterio.open(OUT_FILE) as src:
                print(f"  Shape  : {src.shape}")
                print(f"  CRS    : {src.crs}")
                print(f"  Bounds : {src.bounds}")
            return

if __name__ == "__main__":
    main()
