"""
Fill Missing Rainfall Time Steps
================================
This script fills in missing 30-minute intervals in rainfall data
by linear interpolation between existing timestamps.

Usage:
    python fill_gaps.py
"""

import os
import glob
import re
from datetime import datetime, timedelta
import numpy as np
import rasterio
from rasterio.crs import CRS

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, "tif")
OUTPUT_FOLDER = INPUT_FOLDER  # Save filled files in same folder

# Time interval in minutes
INTERVAL_MINUTES = 30
# ---------------------


def parse_filename(filename):
    """
    Parse filename like '20220220-S190000.tif' to datetime.
    Returns None if format doesn't match.
    """
    pattern = r"(\d{8})-S(\d{6})\.tif"
    match = re.match(pattern, filename)
    if match:
        date_str = match.group(1)  # 20220220
        time_str = match.group(2)  # 190000
        dt_str = f"{date_str}{time_str}"
        return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    return None


def datetime_to_filename(dt):
    """Convert datetime to filename format: YYYYMMDD-SHHMMSS.tif"""
    return dt.strftime("%Y%m%d-S%H%M%S.tif")


def read_raster(filepath):
    """Read raster and return data, profile"""
    with rasterio.open(filepath) as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def write_raster(filepath, data, profile):
    """Write raster data to file"""
    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(data, 1)


def interpolate_rasters(data1, data2, weight):
    """
    Linear interpolation between two rasters.
    weight=0 returns data1, weight=1 returns data2.
    """
    # Handle nodata values
    nodata = -9999.0
    
    # Create output array
    result = np.zeros_like(data1, dtype=np.float32)
    
    # Identify valid pixels in both rasters
    valid1 = data1 != nodata
    valid2 = data2 != nodata
    
    # Interpolate where both are valid
    both_valid = valid1 & valid2
    result[both_valid] = (1 - weight) * data1[both_valid] + weight * data2[both_valid]
    
    # Use single valid value where only one is valid
    only_valid1 = valid1 & ~valid2
    only_valid2 = ~valid1 & valid2
    result[only_valid1] = data1[only_valid1]
    result[only_valid2] = data2[only_valid2]
    
    # Set nodata where both are nodata
    neither_valid = ~valid1 & ~valid2
    result[neither_valid] = nodata
    
    return result


def fill_gaps():
    print("="*60)
    print("FILL MISSING RAINFALL TIME STEPS")
    print("="*60)
    
    # 1. Find all existing files
    tif_files = glob.glob(os.path.join(INPUT_FOLDER, "*.tif"))
    if not tif_files:
        print("No .tif files found in input folder.")
        return
    
    # 2. Parse timestamps and sort
    file_times = []
    for filepath in tif_files:
        filename = os.path.basename(filepath)
        dt = parse_filename(filename)
        if dt:
            file_times.append((dt, filepath))
    
    file_times.sort(key=lambda x: x[0])
    
    if len(file_times) < 2:
        print("Need at least 2 files to interpolate.")
        return
    
    print(f"Found {len(file_times)} rainfall files.")
    print(f"Time range: {file_times[0][0]} to {file_times[-1][0]}")
    
    # 3. Find gaps and interpolate
    interval = timedelta(minutes=INTERVAL_MINUTES)
    filled_count = 0
    
    for i in range(len(file_times) - 1):
        dt1, path1 = file_times[i]
        dt2, path2 = file_times[i + 1]
        
        # Calculate expected number of intervals
        time_diff = dt2 - dt1
        expected_intervals = int(time_diff.total_seconds() / (INTERVAL_MINUTES * 60))
        
        if expected_intervals > 1:
            # There are gaps to fill
            print(f"\nGap detected: {dt1} -> {dt2} ({expected_intervals - 1} missing steps)")
            
            # Load both rasters
            data1, profile = read_raster(path1)
            data2, _ = read_raster(path2)
            
            # Create intermediate files
            for step in range(1, expected_intervals):
                new_dt = dt1 + (interval * step)
                new_filename = datetime_to_filename(new_dt)
                new_filepath = os.path.join(OUTPUT_FOLDER, new_filename)
                
                # Check if file already exists
                if os.path.exists(new_filepath):
                    print(f"  Skip (exists): {new_filename}")
                    continue
                
                # Interpolation weight (0 to 1)
                weight = step / expected_intervals
                
                # Interpolate
                new_data = interpolate_rasters(data1, data2, weight)
                
                # Save
                write_raster(new_filepath, new_data, profile)
                filled_count += 1
                print(f"  Created: {new_filename} (weight={weight:.2f})")
    
    print("\n" + "="*60)
    print(f"Done! Created {filled_count} interpolated files.")
    print("="*60)


if __name__ == "__main__":
    fill_gaps()

