"""
Cut Manning Raster to Match DEM Extent
======================================
This script clips the UNCUT_Manning.tif to match the Ho Chi Minh DEM extent,
resolution, and CRS exactly.

Usage:
    python cut_manning.py
"""

import os
import numpy as np

# Try rasterio first (preferred)
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not installed. Please install: pip install rasterio")

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input files
DEM_PATH = os.path.join(SCRIPT_DIR, "HoChiMinh_DEM.tif")
DEM_TFW_PATH = os.path.join(SCRIPT_DIR, "HoChiMinh_DEM.tfw")
MANNING_PATH = os.path.join(SCRIPT_DIR, "UNCUT_Manning.tif")
MANNING_TFW_PATH = os.path.join(SCRIPT_DIR, "UNCUT_Manning.tfw")

# Output
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "HoChiMinh_Manning.tif")

# DEM CRS (UTM Zone 48N for Ho Chi Minh City)
DEM_EPSG = 32648

# NoData value for output
NODATA_VALUE = -9999.0
# ---------------------


def read_world_file(tfw_path):
    """Read .tfw file for georeferencing"""
    if not os.path.exists(tfw_path):
        return None
    with open(tfw_path, 'r') as f:
        params = [float(line.strip()) for line in f.readlines() if line.strip()]
    return {
        'pixel_width': params[0],
        'rotation_y': params[1],
        'rotation_x': params[2],
        'pixel_height': params[3],
        'x_origin': params[4],
        'y_origin': params[5]
    }


def tfw_to_transform(tfw_info):
    """Convert TFW info to rasterio Affine transform"""
    from rasterio.transform import Affine
    return Affine(
        tfw_info['pixel_width'],
        tfw_info['rotation_x'],
        tfw_info['x_origin'],
        tfw_info['rotation_y'],
        tfw_info['pixel_height'],
        tfw_info['y_origin']
    )


def cut_manning():
    print("="*60)
    print("CUT MANNING RASTER TO MATCH DEM")
    print("="*60)
    
    if not RASTERIO_AVAILABLE:
        print("ERROR: rasterio is required. pip install rasterio")
        return
    
    # 1. Check files exist
    if not os.path.exists(DEM_PATH):
        print(f"ERROR: DEM not found at {DEM_PATH}")
        return
    if not os.path.exists(MANNING_PATH):
        print(f"ERROR: Manning raster not found at {MANNING_PATH}")
        return
    
    # 2. Open DEM to get target properties
    print("\n1. Loading DEM template...")
    with rasterio.open(DEM_PATH) as dem_src:
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        dem_width = dem_src.width
        dem_height = dem_src.height
        dem_bounds = dem_src.bounds
        dem_data = dem_src.read(1)
        
        # If DEM has no CRS, use TFW and set CRS manually
        if dem_crs is None:
            print("   DEM has no embedded CRS, using TFW file...")
            tfw_info = read_world_file(DEM_TFW_PATH)
            if tfw_info:
                dem_transform = tfw_to_transform(tfw_info)
                dem_crs = CRS.from_epsg(DEM_EPSG)
            else:
                print("   ERROR: No CRS and no TFW file found!")
                return
    
    print(f"   DEM Shape: {dem_width} x {dem_height}")
    print(f"   DEM CRS: {dem_crs}")
    print(f"   DEM Bounds: {dem_bounds}")
    
    # Create validity mask (where DEM > 0)
    dem_valid_mask = dem_data > 0
    print(f"   Valid pixels: {np.sum(dem_valid_mask):,} / {dem_valid_mask.size:,}")
    
    # 3. Open Manning raster
    print("\n2. Loading Manning raster...")
    with rasterio.open(MANNING_PATH) as man_src:
        man_crs = man_src.crs
        man_transform = man_src.transform
        man_data = man_src.read(1)
        
        # If Manning has no CRS, check for TFW
        if man_crs is None:
            print("   Manning has no embedded CRS, checking for TFW...")
            tfw_info = read_world_file(MANNING_TFW_PATH)
            if tfw_info:
                man_transform = tfw_to_transform(tfw_info)
                # Assume same CRS as DEM if not specified
                man_crs = CRS.from_epsg(DEM_EPSG)
                print(f"   Using TFW with assumed CRS: {man_crs}")
            else:
                print("   WARNING: No CRS info found. Assuming same CRS as DEM.")
                man_crs = dem_crs
        
        print(f"   Manning Shape: {man_src.width} x {man_src.height}")
        print(f"   Manning CRS: {man_crs}")
        print(f"   Manning Bounds: {man_src.bounds}")
        print(f"   Manning Value Range: {np.nanmin(man_data):.4f} to {np.nanmax(man_data):.4f}")
    
    # 4. Reproject Manning to match DEM exactly
    print("\n3. Reprojecting Manning to match DEM grid...")
    
    # Create output array
    manning_matched = np.empty((dem_height, dem_width), dtype=np.float32)
    manning_matched.fill(NODATA_VALUE)
    
    # Reproject
    with rasterio.open(MANNING_PATH) as man_src:
        # Use the detected/assumed CRS
        src_crs = man_crs if man_crs else dem_crs
        src_transform = man_transform if man_transform else man_src.transform
        
        reproject(
            source=man_data,
            destination=manning_matched,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear,
            src_nodata=man_src.nodata,
            dst_nodata=NODATA_VALUE
        )
    
    # 5. Apply DEM mask (set nodata where DEM is invalid)
    print("\n4. Applying DEM mask...")
    manning_matched[~dem_valid_mask] = NODATA_VALUE
    
    valid_manning = manning_matched[manning_matched != NODATA_VALUE]
    if valid_manning.size > 0:
        print(f"   Output Value Range: {np.min(valid_manning):.4f} to {np.max(valid_manning):.4f}")
        print(f"   Valid pixels: {valid_manning.size:,}")
    
    # 6. Save output
    print(f"\n5. Saving to {OUTPUT_PATH}...")
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': dem_width,
        'height': dem_height,
        'count': 1,
        'crs': dem_crs,
        'transform': dem_transform,
        'nodata': NODATA_VALUE,
        'compress': 'lzw'
    }
    
    with rasterio.open(OUTPUT_PATH, 'w', **profile) as dst:
        dst.write(manning_matched, 1)
    
    # Also create TFW sidecar file
    tfw_output = OUTPUT_PATH.replace('.tif', '.tfw')
    with open(tfw_output, 'w') as f:
        f.write(f"{dem_transform.a:.10f}\n")
        f.write(f"{dem_transform.d:.10f}\n")
        f.write(f"{dem_transform.b:.10f}\n")
        f.write(f"{dem_transform.e:.10f}\n")
        f.write(f"{dem_transform.c:.10f}\n")
        f.write(f"{dem_transform.f:.10f}\n")
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"Output: {OUTPUT_PATH}")
    print(f"TFW:    {tfw_output}")


if __name__ == "__main__":
    cut_manning()

