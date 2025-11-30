import os
import glob
import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling

# --- CONFIGURATION ---
# Script is in: experiments/hochiminh/rainfall/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input Rainfall NetCDF files (in nc4 subfolder)
INPUT_FOLDER = os.path.join(SCRIPT_DIR, "nc4")

# Output Folder for TIFs
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "tif")

# Your Reference DEM (one level up from rainfall folder)
DEM_PATH = os.path.join(SCRIPT_DIR, "..", "HoChiMinh_DEM.tif")
# ---------------------

def process_rainfall():
    # 1. Setup Folders
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # 2. Load the DEM Template
    if not os.path.exists(DEM_PATH):
        print(f"CRITICAL ERROR: DEM not found at {DEM_PATH}")
        return

    print("Loading DEM template...")
    # mask_and_scale=True reads the nodata values correctly if set in metadata
    dem = rioxarray.open_rasterio(DEM_PATH)
    
    # Create the Validity Mask
    # You said 0 is invalid. So we create a True/False grid where DEM != 0.
    # We use .squeeze() to ensure it's 2D (Height, Width) not 3D (1, Height, Width)
    valid_mask = (dem > 0).squeeze()
    
    print(f"DEM Loaded. Shape: {dem.shape}. CRS: {dem.rio.crs}")

    # 3. Find Rainfall Files
    nc_files = glob.glob(os.path.join(INPUT_FOLDER, "*.nc4"))
    if not nc_files:
        print("No .nc4 files found.")
        return

    print(f"Found {len(nc_files)} rainfall files. Processing...")

    for file_path in nc_files:
        try:
            filename = os.path.basename(file_path)
            
            # A. Open Rainfall Data
            ds = xr.open_dataset(file_path, decode_coords="all")

            # B. Extract Precipitation Variable
            if 'precipitation' in ds:
                precip = ds['precipitation']
            elif 'precipitationCal' in ds:
                precip = ds['precipitationCal']
            else:
                print(f"Skipping {filename}: No precipitation variable.")
                continue

            # C. Prepare Rainfall Data
            precip = precip.transpose('time', 'lat', 'lon')
            precip.rio.write_crs("EPSG:4326", inplace=True)

            # D. Reproject and Match Grid (The "Scale" Step)
            # This aligns the pixels perfectly with the DEM using Bilinear Interpolation
            precip_matched = precip.rio.reproject_match(
                dem,
                resampling=Resampling.bilinear
            )

            # E. Apply the Mask (The "Filter" Step)
            # Logic: Where 'valid_mask' is True, keep the rain. 
            #        Where 'valid_mask' is False, set rain to 0.
            precip_final = precip_matched.where(valid_mask, 0)

            # F. Save Individual Time Steps
            for t in precip_final.time:
                ts = pd.Timestamp(t.values)
                # Format: YYYYMMDD-SHHMMSS.tif (e.g., 20220220-S190000.tif)
                tif_name = ts.strftime("%Y%m%d-S%H%M%S.tif")
                output_path = os.path.join(OUTPUT_FOLDER, tif_name)
                
                # Select single time step and save
                single_step = precip_final.sel(time=t)
                single_step.rio.to_raster(output_path)
                print(f"  Saved: {tif_name}")
            
            ds.close()
            print(f"[OK] Processed: {filename}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            continue

    print("\nDone! All rainfall files processed.")

if __name__ == "__main__":
    process_rainfall()