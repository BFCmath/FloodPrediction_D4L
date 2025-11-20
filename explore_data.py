"""
FloodCastBench Data Explorer
=============================
A simple script to visualize and understand the FloodCastBench dataset structure.

Usage:
    python explore_data.py
"""

import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# Configuration
# ============================================================================

# Base path to the data
BASE_PATH = Path("FloodCastBench_Dataset-and-Models-main/Data_Generation_Code/FloodCastBench")
RELEVANT_DATA_PATH = BASE_PATH / "Relevant_data"
OUTPUT_DATA_PATH = BASE_PATH / "High-fidelity_flood_forecasting"

# ============================================================================
# Utility Functions
# ============================================================================

def explore_folder_structure():
    """Print the complete folder structure with file counts."""
    print("=" * 80)
    print("📁 FLOODCASTBENCH FOLDER STRUCTURE")
    print("=" * 80)
    
    if not BASE_PATH.exists():
        print(f"❌ ERROR: Base path not found: {BASE_PATH}")
        print("   Please update BASE_PATH in the script to match your directory.")
        return
    
    for root, dirs, files in os.walk(BASE_PATH):
        level = root.replace(str(BASE_PATH), '').count(os.sep)
        indent = '  ' * level
        folder_name = os.path.basename(root)
        print(f"{indent}📂 {folder_name}/ ({len(files)} files, {len(dirs)} folders)")
        
        # Show first few files as examples
        sub_indent = '  ' * (level + 1)
        for i, file in enumerate(files[:3]):  # Show first 3 files
            size_mb = os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
            print(f"{sub_indent}📄 {file} ({size_mb:.2f} MB)")
        
        if len(files) > 3:
            print(f"{sub_indent}   ... and {len(files) - 3} more files")


def get_tif_info(filepath: Path) -> Dict:
    """Get information about a TIFF file."""
    if not filepath.exists():
        return {"error": "File not found"}
    
    try:
        data = tifffile.imread(filepath)
        return {
            "shape": data.shape,
            "dtype": data.dtype,
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
            "mean": float(np.nanmean(data)),
            "has_nan": bool(np.isnan(data).any()),
            "size_mb": filepath.stat().st_size / (1024 * 1024)
        }
    except Exception as e:
        return {"error": str(e)}


def print_data_summary():
    """Print summary statistics for each data type."""
    print("\n" + "=" * 80)
    print("📊 DATA SUMMARY")
    print("=" * 80)
    
    # DEM
    print("\n🗻 DEM (Digital Elevation Model)")
    print("-" * 80)
    dem_path = RELEVANT_DATA_PATH / "DEM" / "Australia_DEM.tif"
    if dem_path.exists():
        info = get_tif_info(dem_path)
        print(f"   File: {dem_path.name}")
        print(f"   Shape: {info.get('shape', 'N/A')}")
        print(f"   Data type: {info.get('dtype', 'N/A')}")
        print(f"   Elevation range: {info.get('min', 'N/A'):.2f}m to {info.get('max', 'N/A'):.2f}m")
        print(f"   Mean elevation: {info.get('mean', 'N/A'):.2f}m")
        print(f"   File size: {info.get('size_mb', 'N/A'):.2f} MB")
    else:
        print(f"   ❌ Not found: {dem_path}")
    
    # Rainfall
    print("\n🌧️ RAINFALL")
    print("-" * 80)
    rain_dir = RELEVANT_DATA_PATH / "Rainfall" / "Australia_flood"
    if rain_dir.exists():
        rain_files = sorted(list(rain_dir.glob("*.tif")))
        print(f"   Number of timesteps: {len(rain_files)}")
        if rain_files:
            first_file = rain_files[0]
            last_file = rain_files[-1]
            print(f"   First timestep: {first_file.name}")
            print(f"   Last timestep: {last_file.name}")
            
            info = get_tif_info(first_file)
            print(f"   Shape: {info.get('shape', 'N/A')}")
            print(f"   Rainfall range (first file): {info.get('min', 'N/A'):.2f} to {info.get('max', 'N/A'):.2f} mm/hr")
    else:
        print(f"   ❌ Not found: {rain_dir}")
    
    # Land Use (Manning)
    print("\n🌾 LAND USE / LAND COVER (Manning Coefficient)")
    print("-" * 80)
    manning_path = RELEVANT_DATA_PATH / "Land_use_and_land_cover" / "Australia.tif"
    if manning_path.exists():
        info = get_tif_info(manning_path)
        print(f"   File: {manning_path.name}")
        print(f"   Shape: {info.get('shape', 'N/A')}")
        print(f"   Manning range: {info.get('min', 'N/A'):.4f} to {info.get('max', 'N/A'):.4f}")
        print(f"   Mean roughness: {info.get('mean', 'N/A'):.4f}")
    else:
        print(f"   ❌ Not found: {manning_path}")
    
    # Initial Conditions
    print("\n💧 INITIAL CONDITIONS")
    print("-" * 80)
    init_30m = RELEVANT_DATA_PATH / "Initial_conditions" / "High-fidelity_flood_forecasting" / "Australia_30m.tif"
    init_60m = RELEVANT_DATA_PATH / "Initial_conditions" / "High-fidelity_flood_forecasting" / "Australia_60m.tif"
    
    for init_path, res in [(init_30m, "30m"), (init_60m, "60m")]:
        if init_path.exists():
            info = get_tif_info(init_path)
            print(f"   {res} Resolution:")
            print(f"      Shape: {info.get('shape', 'N/A')}")
            print(f"      Initial depth range: {info.get('min', 'N/A'):.2f}m to {info.get('max', 'N/A'):.2f}m")
        else:
            print(f"   ❌ {res} Not found: {init_path}")
    
    # Output Data
    print("\n📈 OUTPUT DATA (Simulation Results)")
    print("-" * 80)
    for resolution in ["30m", "60m"]:
        output_dir = OUTPUT_DATA_PATH / resolution / "Australia"
        if output_dir.exists():
            output_files = sorted(list(output_dir.glob("*.tif")))
            print(f"   {resolution} Resolution: {len(output_files)} timesteps")
            if output_files:
                # Get first and last timestep info
                first_info = get_tif_info(output_files[0])
                last_info = get_tif_info(output_files[-1])
                print(f"      First: {output_files[0].name} - Max depth: {first_info.get('max', 'N/A'):.2f}m")
                print(f"      Last: {output_files[-1].name} - Max depth: {last_info.get('max', 'N/A'):.2f}m")
        else:
            print(f"   ❌ {resolution} Not found: {output_dir}")


def visualize_data():
    """Create visualizations of the data."""
    print("\n" + "=" * 80)
    print("🎨 CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FloodCastBench Dataset Overview', fontsize=16, fontweight='bold')
    
    # 1. DEM
    dem_path = RELEVANT_DATA_PATH / "DEM" / "Australia_DEM.tif"
    if dem_path.exists():
        dem = tifffile.imread(dem_path)
        im1 = axes[0, 0].imshow(dem, cmap='terrain')
        axes[0, 0].set_title('DEM - Terrain Elevation')
        axes[0, 0].set_xlabel('X (pixels)')
        axes[0, 0].set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)')
        print("   ✓ DEM visualization created")
    else:
        axes[0, 0].text(0.5, 0.5, 'DEM not found', ha='center', va='center')
        axes[0, 0].set_title('DEM - Not Available')
    
    # 2. Rainfall (first timestep)
    rain_dir = RELEVANT_DATA_PATH / "Rainfall" / "Australia_flood"
    if rain_dir.exists():
        rain_files = sorted(list(rain_dir.glob("*.tif")))
        if rain_files:
            rain = tifffile.imread(rain_files[0])
            im2 = axes[0, 1].imshow(rain, cmap='Blues')
            axes[0, 1].set_title(f'Rainfall - {rain_files[0].stem}')
            axes[0, 1].set_xlabel('X (pixels)')
            axes[0, 1].set_ylabel('Y (pixels)')
            plt.colorbar(im2, ax=axes[0, 1], label='Rain (mm/hr)')
            print("   ✓ Rainfall visualization created")
    else:
        axes[0, 1].text(0.5, 0.5, 'Rainfall not found', ha='center', va='center')
        axes[0, 1].set_title('Rainfall - Not Available')
    
    # 3. Manning Coefficient
    manning_path = RELEVANT_DATA_PATH / "Land_use_and_land_cover" / "Australia.tif"
    if manning_path.exists():
        manning = tifffile.imread(manning_path)
        im3 = axes[0, 2].imshow(manning, cmap='YlGn')
        axes[0, 2].set_title('Land Use - Manning Coefficient')
        axes[0, 2].set_xlabel('X (pixels)')
        axes[0, 2].set_ylabel('Y (pixels)')
        plt.colorbar(im3, ax=axes[0, 2], label='Manning n')
        print("   ✓ Manning coefficient visualization created")
    else:
        axes[0, 2].text(0.5, 0.5, 'Manning not found', ha='center', va='center')
        axes[0, 2].set_title('Manning - Not Available')
    
    # 4-6. Flood depth at different times (30m resolution)
    output_dir = OUTPUT_DATA_PATH / "30m" / "Australia"
    if output_dir.exists():
        output_files = sorted(list(output_dir.glob("*.tif")))
        if len(output_files) >= 3:
            # Show first, middle, and last
            indices = [0, len(output_files) // 2, -1]
            for idx, ax_idx in zip(indices, range(3)):
                flood = tifffile.imread(output_files[indices[ax_idx]])
                im = axes[1, ax_idx].imshow(flood, cmap='Blues', vmin=0, vmax=np.nanmax(flood))
                timestamp = output_files[indices[ax_idx]].stem
                time_hours = int(timestamp) / 3600 if timestamp.isdigit() else 0
                axes[1, ax_idx].set_title(f'Flood Depth - t={time_hours:.1f}h')
                axes[1, ax_idx].set_xlabel('X (pixels)')
                axes[1, ax_idx].set_ylabel('Y (pixels)')
                plt.colorbar(im, ax=axes[1, ax_idx], label='Depth (m)')
            print("   ✓ Flood depth visualizations created")
    else:
        for ax_idx in range(3):
            axes[1, ax_idx].text(0.5, 0.5, 'Output not found', ha='center', va='center')
            axes[1, ax_idx].set_title(f'Flood Depth {ax_idx+1} - Not Available')
    
    plt.tight_layout()
    output_path = "FloodCastBench_Visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n   💾 Saved visualization to: {output_path}")
    plt.show()


def analyze_time_series():
    """Analyze the temporal evolution of the flood."""
    print("\n" + "=" * 80)
    print("📈 TIME SERIES ANALYSIS")
    print("=" * 80)
    
    output_dir = OUTPUT_DATA_PATH / "30m" / "Australia"
    if not output_dir.exists():
        print(f"   ❌ Output directory not found: {output_dir}")
        return
    
    output_files = sorted(list(output_dir.glob("*.tif")))
    if not output_files:
        print("   ❌ No output files found")
        return
    
    print(f"   Found {len(output_files)} timesteps")
    
    # Extract statistics over time
    times = []
    max_depths = []
    mean_depths = []
    flooded_area = []
    
    print("   Processing timesteps...")
    for f in output_files[:min(50, len(output_files))]:  # Analyze first 50 timesteps
        try:
            timestamp = int(f.stem) if f.stem.isdigit() else 0
            times.append(timestamp / 3600)  # Convert to hours
            
            data = tifffile.imread(f)
            max_depths.append(np.nanmax(data))
            mean_depths.append(np.nanmean(data[data > 0]))  # Mean of flooded areas only
            flooded_area.append(np.sum(data > 0.1))  # Count cells with >10cm water
        except Exception as e:
            print(f"   ⚠️  Error processing {f.name}: {e}")
    
    # Plot time series
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Flood Evolution Over Time', fontsize=14, fontweight='bold')
    
    axes[0].plot(times, max_depths, 'b-', linewidth=2)
    axes[0].set_ylabel('Max Depth (m)', fontsize=12)
    axes[0].set_title('Maximum Water Depth')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(times, mean_depths, 'g-', linewidth=2)
    axes[1].set_ylabel('Mean Depth (m)', fontsize=12)
    axes[1].set_title('Mean Water Depth (flooded areas only)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(times, flooded_area, 'r-', linewidth=2)
    axes[2].set_xlabel('Time (hours)', fontsize=12)
    axes[2].set_ylabel('Flooded Area (pixels)', fontsize=12)
    axes[2].set_title('Flooded Area (depth > 10cm)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = "FloodCastBench_TimeSeries.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n   💾 Saved time series analysis to: {output_path}")
    plt.show()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to explore the FloodCastBench dataset."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "FloodCastBench Data Explorer" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        # 1. Explore folder structure
        explore_folder_structure()
        
        # 2. Print data summary
        print_data_summary()
        
        # 3. Create visualizations
        visualize_data()
        
        # 4. Time series analysis
        analyze_time_series()
        
        print("\n" + "=" * 80)
        print("✅ EXPLORATION COMPLETE!")
        print("=" * 80)
        print("\nGenerated files:")
        print("   📊 FloodCastBench_Visualization.png")
        print("   📈 FloodCastBench_TimeSeries.png")
        print("\nNext steps:")
        print("   1. Review the generated visualizations")
        print("   2. Read FloodCastBench_Data_Structure_Guide.md for details")
        print("   3. Check FloodCastBench_Cheat_Sheet.md for quick reference")
        print()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
