"""
Simulate AI Service Output Package for Ho Chi Minh City
=======================================================
This script simulates an AI service that:
1. Loads the Ho Chi Minh City DEM.
2. Generates a synthetic flood inundation map based on elevation (simple "bathtub model").
   - Pixels below a certain water level (e.g., 2.0m) are considered flooded.
   - Flood depth = Water Level - Ground Elevation.
3. Calculates the geospatial bounds (WGS84).
4. Packages this into a folder structure ready to be sent to a Rendering Service.

The "Package" (in 'mock_api_response_hochiminh/') contains:
- flood_depths.npy: The raw 2D matrix of flood depths (float32).
- metadata.json:  Geospatial bounds (WGS84), dimensions, and stats.
"""

import numpy as np
import tifffile
import json
from pathlib import Path
import shutil

# Try to import pyproj for coordinate conversion
try:
    from pyproj import Transformer, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    print("Error: pyproj is required. pip install pyproj")
    exit(1)

# Try to import rasterio for GeoTIFF writing (better than tifffile for this)
try:
    import rasterio
    from rasterio.transform import from_origin
    RASTERIO_AVAILABLE = True
except ImportError:
    print("Warning: rasterio not installed. Using tifffile fallback (less metadata).")
    RASTERIO_AVAILABLE = False

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# Paths
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "hochiminh"
DEM_PATH = DATA_DIR / "HoChiMinh_DEM.tif"
TFW_PATH = DATA_DIR / "HoChiMinh_DEM.tfw"

# The output package folder (relative to script location)
OUTPUT_PACKAGE_DIR = CURRENT_DIR.parent / "mock_api_response_hochiminh"

# Coordinate System of the source data (UTM Zone 48N for HCMC)
SOURCE_EPSG = 32648

# Flood Generation Parameters
# Simulating a flood event where water rises to 1.5 meters above reference 0
WATER_LEVEL_METERS = 1.5

# Value to use for NoData (pixels outside the DEM)
NODATA_VALUE = -9999.0

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def read_world_file(tfw_path):
    """Read .tfw file for georeferencing"""
    if not tfw_path.exists():
        return None
    with open(tfw_path, 'r') as f:
        params = [float(line.strip()) for line in f.readlines()]
    return {
        'pixel_width': params[0],
        'rotation_y': params[1],
        'rotation_x': params[2],
        'pixel_height': params[3],
        'x_origin': params[4],
        'y_origin': params[5]
    }

def get_wgs84_bounds(width, height, tfw_info, epsg_code):
    """Convert image pixel bounds to WGS84 Lat/Lon"""
    
    # Calculate projection coordinates (UTM)
    x_min = tfw_info['x_origin']
    y_max = tfw_info['y_origin']
    x_max = x_min + width * tfw_info['pixel_width']
    y_min = y_max + height * tfw_info['pixel_height']
    
    # Create transformer: UTM -> WGS84 (Lat/Lon)
    # always_xy=True means transform(long, lat)
    crs_source = CRS.from_epsg(epsg_code)
    transformer = Transformer.from_crs(crs_source, "EPSG:4326", always_xy=True)
    
    # Transform corners
    # Note: transformer.transform returns (lon, lat) because always_xy=True
    lon_min, lat_max = transformer.transform(x_min, y_max)
    lon_max, lat_min = transformer.transform(x_max, y_min)
    
    return {
        "north": lat_max,
        "south": lat_min,
        "east": lon_max,
        "west": lon_min,
        "center": {
            "lat": (lat_max + lat_min) / 2,
            "lon": (lon_max + lon_min) / 2
        }
    }

# ============================================================================
# 3. MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("SIMULATING AI SERVICE RESPONSE PACKAGE (HCMC)")
    print("="*60)

    # --- Step 1: Verify Data Availability ---
    if not DEM_PATH.exists() or not TFW_PATH.exists():
        print(f"Error: Missing source data at {DEM_PATH}")
        return

    print(f"1. Reading Source Configuration (DEM)...")
    # Get Dimensions from DEM
    dem = tifffile.imread(DEM_PATH)
    height, width = dem.shape
    print(f"   [OK] Grid Dimensions: {width} x {height}")
    print(f"   [OK] Elevation range: {np.min(dem):.2f}m to {np.max(dem):.2f}m")

    # Get Georeference info
    tfw_info = read_world_file(TFW_PATH)
    if not tfw_info:
        print("Error: Could not read .tfw file")
        return
    
    pixel_res = abs(tfw_info['pixel_width'])

    # --- Step 2: Calculate Map Bounds (The Context) ---
    print(f"2. Calculating WGS84 Bounds (Lat/Lon)...")
    bounds = get_wgs84_bounds(width, height, tfw_info, SOURCE_EPSG)
    print(f"   [OK] Bounds: N={bounds['north']:.4f}, S={bounds['south']:.4f}, E={bounds['east']:.4f}, W={bounds['west']:.4f}")

    # --- Step 3: Generate Synthetic Flood Prediction ---
    print(f"3. Generating Synthetic Flood Map...")
    print(f"   Simulating water level at: {WATER_LEVEL_METERS} meters")
    
    # Create flood array (same shape as DEM, explicitly float32)
    # Formula: Depth = Water_Level - Ground_Elevation
    # Only where Water_Level > Ground_Elevation
    flood_depths = np.maximum(0, WATER_LEVEL_METERS - dem).astype(np.float32)
    
    # Mask out areas that might be NoData in DEM
    # We identified that the padding/background value is exactly 0.
    # We mark these as NODATA_VALUE so they are completely ignored (transparent).
    invalid_mask = dem == 0
    flood_depths[invalid_mask] = NODATA_VALUE

    # Calculate stats (excluding NoData)
    valid_flood_pixels = flood_depths[flood_depths != NODATA_VALUE]
    
    if valid_flood_pixels.size > 0:
        max_depth = float(np.max(valid_flood_pixels))
        flooded_pixels = int(np.sum(valid_flood_pixels > 0.05)) # Count pixels with >5cm water
        total_valid_pixels = valid_flood_pixels.size
        flooded_percent = (flooded_pixels / total_valid_pixels) * 100
        mean_depth = float(np.mean(valid_flood_pixels[valid_flood_pixels > 0.05])) if flooded_pixels > 0 else 0.0
    else:
        max_depth = 0.0
        flooded_pixels = 0
        flooded_percent = 0.0
        mean_depth = 0.0

    print(f"   [OK] Flood Generated:")
    print(f"        Max Depth: {max_depth:.2f}m")
    print(f"        Flooded Area: {flooded_pixels:,} pixels ({flooded_percent:.2f}%)")
    print(f"        Mean Depth (flooded): {mean_depth:.2f}m")

    # --- Step 4: Package for Service ---
    print(f"4. Creating Service Payload in '{OUTPUT_PACKAGE_DIR}'...")
    
    # Clean/Create directory
    if OUTPUT_PACKAGE_DIR.exists():
        shutil.rmtree(OUTPUT_PACKAGE_DIR)
    OUTPUT_PACKAGE_DIR.mkdir(parents=True)

    # A. Save Metadata (JSON)
    metadata = {
        "request_id": "sim_hcm_001",
        "timestamp": "2025-11-29T10:00:00Z", # Current simulated time
        "location": "Ho Chi Minh City",
        "simulation_type": "static_inundation",
        "water_level_param": WATER_LEVEL_METERS,
        "bounds": bounds,
        "grid": {
            "width": width,
            "height": height,
            "resolution_meters": pixel_res
        },
        "data_stats": {
            "max_depth_meters": max_depth,
            "flooded_area_pixels": flooded_pixels,
            "flooded_percentage": flooded_percent,
            "unit": "meters",
            "nodata_value": NODATA_VALUE
        },
        "format": "geotiff"
    }
    
    with open(OUTPUT_PACKAGE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # B. Save as GeoTIFF
    # Using rasterio if available for proper georeferencing, else tifffile
    output_tif_path = OUTPUT_PACKAGE_DIR / "flood_depths.tif"
    
    if RASTERIO_AVAILABLE:
        # Create transform from .tfw info
        # TFW parameters: [pixel_width, rotation_y, rotation_x, pixel_height, x_origin, y_origin]
        # Affine(a, b, c, d, e, f) -> x' = ax + by + c, y' = dx + ey + f
        # Mapping: a=width, b=rot_x, c=x_org, d=rot_y, e=height, f=y_org
        transform = rasterio.Affine(
            tfw_info['pixel_width'], 
            tfw_info['rotation_x'], 
            tfw_info['x_origin'], 
            tfw_info['rotation_y'], 
            tfw_info['pixel_height'], 
            tfw_info['y_origin']
        )
        
        with rasterio.open(
            output_tif_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=flood_depths.dtype,
            crs=CRS.from_epsg(SOURCE_EPSG),
            transform=transform,
            nodata=NODATA_VALUE,
            compress='lzw'
        ) as dst:
            dst.write(flood_depths, 1)
            
    else:
        # Fallback to simple tifffile (no CRS metadata embedded, but readable)
        tifffile.imwrite(output_tif_path, flood_depths)
        print("Warning: Saved without embedded CRS (rasterio missing). Use sidecar metadata.")
    
    print("\n" + "="*60)
    print("[OK] PACKAGE CREATED SUCCESSFULLY")
    print("="*60)
    print(f"Location: {OUTPUT_PACKAGE_DIR.absolute()}")
    print("\nContents:")
    print("1. metadata.json    -> Contains bounds, stats, and simulation params")
    print("2. flood_depths.tif -> GeoTIFF with embedded georeferencing and NoData value")
    print("\nNext Step: You can use this package to test your visualization/rendering service.")

if __name__ == "__main__":
    main()
