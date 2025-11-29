"""
Check DEM Corner Values
=======================
Reads the Ho Chi Minh City DEM and prints the values at the 4 corners
to inspect for NoData/Invalid values.
"""

import tifffile
import numpy as np
from pathlib import Path

# Path to DEM
DEM_PATH = Path("experiments/hochiminh/HoChiMinh_DEM.tif")

def main():
    if not DEM_PATH.exists():
        print(f"Error: File not found at {DEM_PATH}")
        return

    print(f"Reading DEM: {DEM_PATH}")
    dem = tifffile.imread(DEM_PATH)
    
    height, width = dem.shape
    print(f"Dimensions: {width} x {height}")
    print(f"Data Type: {dem.dtype}")
    print(f"Min Value: {np.min(dem)}")
    print(f"Max Value: {np.max(dem)}")
    
    print("\nCorner Values:")
    
    # Top-Left (0,0)
    tl = dem[0, 0]
    print(f"  Top-Left (0,0):      {tl}")
    
    # Top-Right (0, width-1)
    tr = dem[0, width-1]
    print(f"  Top-Right (0,{width-1}):   {tr}")
    
    # Bottom-Left (height-1, 0)
    bl = dem[height-1, 0]
    print(f"  Bottom-Left ({height-1},0): {bl}")
    
    # Bottom-Right (height-1, width-1)
    br = dem[height-1, width-1]
    print(f"  Bottom-Right ({height-1},{width-1}): {br}")

    # Check a few neighbors just in case edge is exactly 0 but nearby is nodata
    print("\nSample of first 5x5 pixels (Top-Left):")
    print(dem[:5, :5])

if __name__ == "__main__":
    main()


