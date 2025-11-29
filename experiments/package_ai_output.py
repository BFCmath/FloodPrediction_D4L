"""
Simulate AI Service Output Package
==================================
This script simulates an AI service that:
1. Generates/Loads a flood prediction (Input Data).
2. Calculates the geospatial bounds (Where it goes on the map).
3. Packages this into a folder structure ready to be sent to a Rendering Service.

The "Package" (in 'mock_api_response/') contains:
- flood_data.npy: The raw 2D matrix of flood depths (float32).
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

# ============================================================================
# 1. CONFIGURATION (Simulating the AI Service Environment)
# ============================================================================

# Paths to the "internal" data the AI uses to generate predictions
DATA_ROOT = Path("for_dev")
FLOOD_OUTPUT_DIR = DATA_ROOT / "High-fidelity_flood_forecasting" / "60m" / "Australia"
DEM_PATH = DATA_ROOT / "DEM" / "Australia_DEM.tif"
TFW_PATH = DATA_ROOT / "DEM" / "Australia_DEM.tfw"

# The output package folder
OUTPUT_PACKAGE_DIR = Path("mock_api_response")

# Coordinate System of the source data (UTM Zone 56S)
SOURCE_EPSG = 32756 

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
    crs_source = CRS.from_epsg(epsg_code)
    transformer = Transformer.from_crs(crs_source, "EPSG:4326", always_xy=True)
    
    # Transform corners
    # Note: standard transformer returns (lon, lat)
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
    print("SIMULATING AI SERVICE RESPONSE PACKAGE")
    print("="*60)

    # --- Step 1: Verify Data Availability ---
    if not DEM_PATH.exists() or not TFW_PATH.exists():
        print(f"Error: Missing source data at {DEM_PATH}")
        return

    print(f"1. Reading Source Configuration (DEM)...")
    # Get Dimensions from DEM
    dem = tifffile.imread(DEM_PATH)
    height, width = dem.shape
    print(f"   Grid Dimensions: {width} x {height}")

    # Get Georeference info
    tfw_info = read_world_file(TFW_PATH)
    if not tfw_info:
        print("Error: Could not read .tfw file")
        return

    # --- Step 2: Calculate Map Bounds (The Context) ---
    print(f"2. Calculating WGS84 Bounds (Lat/Lon)...")
    bounds = get_wgs84_bounds(width, height, tfw_info, SOURCE_EPSG)
    print(f"   Bounds: N={bounds['north']:.4f}, S={bounds['south']:.4f}, E={bounds['east']:.4f}, W={bounds['west']:.4f}")

    # --- Step 3: Load Flood Prediction (The AI Output) ---
    # We'll just grab the last available timestep to simulate a "prediction"
    print(f"3. Generating/Loading Flood Prediction...")
    flood_files = sorted(list(FLOOD_OUTPUT_DIR.glob("*.tif")))
    if not flood_files:
        print("Error: No flood files found to simulate prediction.")
        return
    
    # Pick the peak flood or last frame
    target_file = flood_files[-3] 
    print(f"   Using file: {target_file.name}")
    flood_data = tifffile.imread(target_file)

    # DEBUG: Inspect raw values to verify it's water depth
    print(f"   DEBUG: Raw Data Range -> Min: {np.min(flood_data):.4f}, Max: {np.max(flood_data):.4f}, Mean (wet): {np.mean(flood_data[flood_data>0.05]):.4f}")

    # Calculate simple stats for the renderer to use (e.g. for color scaling)
    max_depth = float(np.max(flood_data))
    flooded_pixels = int(np.sum(flood_data > 0.05)) # > 5cm
    
    # --- Step 4: Package for Service ---
    print(f"4. Creating Service Payload in '{OUTPUT_PACKAGE_DIR}'...")
    
    # Clean/Create directory
    if OUTPUT_PACKAGE_DIR.exists():
        shutil.rmtree(OUTPUT_PACKAGE_DIR)
    OUTPUT_PACKAGE_DIR.mkdir(parents=True)

    # A. Save Metadata (JSON)
    # This tells the renderer WHERE to put the map and HOW to scale colors
    metadata = {
        "request_id": "sim_req_001",
        "timestamp": "2023-10-27T12:00:00Z",
        "bounds": bounds,
        "grid": {
            "width": width,
            "height": height,
            "resolution_meters": 60
        },
        "data_stats": {
            "max_depth_meters": max_depth,
            "flooded_area_pixels": flooded_pixels,
            "unit": "meters"
        },
        "format": "npy_float32"
    }
    
    with open(OUTPUT_PACKAGE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # B. Save Raw Data (Numpy Binary)
    # This is efficient for transfer. The renderer loads this to generate the image.
    np.save(OUTPUT_PACKAGE_DIR / "flood_depths.npy", flood_data.astype(np.float32))

    print("\n" + "="*60)
    print("PACKAGE CREATED SUCCESSFULLY")
    print("="*60)
    print(f"Location: {OUTPUT_PACKAGE_DIR.absolute()}")
    print("\nContents:")
    print("1. metadata.json    -> Contains Lat/Lon bounds and stats")
    print("2. flood_depths.npy -> Raw 2D array of water depths")
    print("\nThis folder represents the API response your AI service would send")
    print("to the frontend/map-rendering service.")

if __name__ == "__main__":
    main()

