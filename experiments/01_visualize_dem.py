"""
01 - Visualize DEM (Digital Elevation Model)
==============================================
Load and render the terrain elevation data for Australia flood region.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import tifffile
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Path to DEM file
DATA_ROOT = Path("..") / "FloodCastBench_Dataset-and-Models-main" / "Data_Generation_Code" / "FloodCastBench"
DEM_PATH = DATA_ROOT / "Relevant_data" / "DEM" / "Australia_DEM.tif"
TFW_PATH = DATA_ROOT / "Relevant_data" / "DEM" / "Australia_DEM.tfw"

# ============================================================================
# Functions
# ============================================================================

def read_world_file(tfw_path):
    """Read .tfw world file for georeferencing"""
    if not tfw_path.exists():
        return None
    
    with open(tfw_path, 'r') as f:
        params = [float(line.strip()) for line in f.readlines()]
    
    return {
        'pixel_width': params[0],       # Pixel width in meters
        'rotation_y': params[1],        # Rotation (usually 0)
        'rotation_x': params[2],        # Rotation (usually 0)
        'pixel_height': params[3],      # Pixel height (negative = north-up)
        'x_origin': params[4],          # X-coordinate of upper-left
        'y_origin': params[5]           # Y-coordinate of upper-left
    }


def visualize_dem(dem_path, tfw_path=None):
    """
    Comprehensive DEM visualization with multiple views.
    
    Args:
        dem_path: Path to DEM GeoTIFF file
        tfw_path: Optional path to .tfw world file
    """
    # Read DEM
    print("Loading DEM...")
    dem = tifffile.imread(dem_path)
    print(f"✓ DEM loaded: {dem.shape} pixels")
    
    # Read georeferencing if available
    geo_info = None
    if tfw_path and tfw_path.exists():
        geo_info = read_world_file(tfw_path)
        print(f"✓ Georeferencing found: {abs(geo_info['pixel_width'])}m resolution")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # 1. Basic Elevation Map (terrain colormap)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(dem, cmap='terrain', interpolation='bilinear')
    ax1.set_title('Elevation Map (Terrain)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Elevation (m)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 2. Hillshade (3D shaded relief)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ls = LightSource(azdeg=315, altdeg=45)
    pixel_size = abs(geo_info['pixel_width']) if geo_info else 30
    hillshade = ls.hillshade(dem, vert_exag=2, dx=pixel_size, dy=pixel_size)
    ax2.imshow(hillshade, cmap='gray')
    ax2.set_title('Hillshade (Shaded Relief)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    # ========================================================================
    # 3. Elevation + Hillshade Combined
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    # Blend hillshade with colored elevation
    rgb = ls.shade(dem, cmap=plt.cm.terrain, vert_exag=2, 
                   dx=pixel_size, dy=pixel_size, blend_mode='overlay')
    ax3.imshow(rgb)
    ax3.set_title('Combined (Elevation + Hillshade)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    
    # ========================================================================
    # 4. Contour Lines
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    contour_filled = ax4.contourf(dem, levels=30, cmap='viridis', alpha=0.8)
    contour_lines = ax4.contour(dem, levels=15, colors='black', linewidths=0.5, alpha=0.4)
    ax4.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f m')
    ax4.set_title('Elevation Contours', fontweight='bold', fontsize=12)
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    cbar4 = plt.colorbar(contour_filled, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('Elevation (m)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 5. Slope Map (terrain steepness)
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    # Calculate gradient (slope)
    gy, gx = np.gradient(dem)
    slope = np.sqrt(gx**2 + gy**2) * 100  # Convert to percentage
    im5 = ax5.imshow(slope, cmap='hot', vmin=0, vmax=np.percentile(slope, 95))
    ax5.set_title('Terrain Slope (%)', fontweight='bold', fontsize=12)
    ax5.set_xlabel('X (pixels)')
    ax5.set_ylabel('Y (pixels)')
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.set_label('Slope (%)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 6. Aspect Map (terrain orientation)
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    aspect = np.arctan2(gy, gx) * 180 / np.pi
    im6 = ax6.imshow(aspect, cmap='hsv', vmin=-180, vmax=180)
    ax6.set_title('Terrain Aspect (Direction)', fontweight='bold', fontsize=12)
    ax6.set_xlabel('X (pixels)')
    ax6.set_ylabel('Y (pixels)')
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.set_label('Aspect (degrees)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 7. Elevation Distribution Histogram
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(dem.flatten(), bins=100, color='brown', alpha=0.7, edgecolor='black')
    ax7.axvline(np.mean(dem), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(dem):.1f}m')
    ax7.axvline(np.median(dem), color='blue', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(dem):.1f}m')
    ax7.set_xlabel('Elevation (m)', fontweight='bold')
    ax7.set_ylabel('Frequency', fontweight='bold')
    ax7.set_title('Elevation Distribution', fontweight='bold', fontsize=12)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # ========================================================================
    # 8. Cross-Section Profile
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    # Take horizontal and vertical cross-sections through center
    center_y = dem.shape[0] // 2
    center_x = dem.shape[1] // 2
    
    profile_horizontal = dem[center_y, :]
    profile_vertical = dem[:, center_x]
    
    x_pixels = np.arange(len(profile_horizontal))
    y_pixels = np.arange(len(profile_vertical))
    
    ax8.plot(x_pixels, profile_horizontal, 'b-', linewidth=2, label=f'Horizontal (row {center_y})')
    ax8.plot(y_pixels, profile_vertical, 'r-', linewidth=2, label=f'Vertical (col {center_x})')
    ax8.set_xlabel('Distance (pixels)', fontweight='bold')
    ax8.set_ylabel('Elevation (m)', fontweight='bold')
    ax8.set_title('Cross-Section Profiles', fontweight='bold', fontsize=12)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # ========================================================================
    # 9. Statistics Summary
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    stats_text = "═══ DEM STATISTICS ═══\n\n"
    stats_text += f"Shape: {dem.shape[0]} × {dem.shape[1]} pixels\n"
    
    if geo_info:
        width_km = dem.shape[1] * abs(geo_info['pixel_width']) / 1000
        height_km = dem.shape[0] * abs(geo_info['pixel_height']) / 1000
        stats_text += f"Coverage: {width_km:.2f} × {height_km:.2f} km\n"
        stats_text += f"Resolution: {abs(geo_info['pixel_width'])}m\n"
    
    stats_text += f"\nElevation:\n"
    stats_text += f"  Min:    {np.min(dem):>8.2f} m\n"
    stats_text += f"  Max:    {np.max(dem):>8.2f} m\n"
    stats_text += f"  Mean:   {np.mean(dem):>8.2f} m\n"
    stats_text += f"  Median: {np.median(dem):>8.2f} m\n"
    stats_text += f"  Std:    {np.std(dem):>8.2f} m\n"
    stats_text += f"  Range:  {np.max(dem) - np.min(dem):>8.2f} m\n"
    
    stats_text += f"\nSlope:\n"
    stats_text += f"  Mean:   {np.mean(slope):>8.2f} %\n"
    stats_text += f"  Max:    {np.max(slope):>8.2f} %\n"
    
    if geo_info:
        stats_text += f"\nGeoreference:\n"
        stats_text += f"  Origin X: {geo_info['x_origin']:>12.2f}\n"
        stats_text += f"  Origin Y: {geo_info['y_origin']:>12.2f}\n"
    
    ax9.text(0.1, 0.95, stats_text, transform=ax9.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Main title
    fig.suptitle('DEM Visualization - Australia Flood Region', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = Path("outputs") / "01_dem_visualization.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Print statistics to console
    print("\n" + "="*60)
    print(stats_text)
    print("="*60)
    
    return fig, dem


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("DEM VISUALIZATION - Australia Flood Region")
    print("="*60)
    
    # Check if DEM file exists
    if not DEM_PATH.exists():
        print(f"\n✗ ERROR: DEM file not found at: {DEM_PATH}")
        print("Please update DATA_ROOT in the script to match your directory structure.")
        exit(1)
    
    # Visualize DEM
    fig, dem_data = visualize_dem(DEM_PATH, TFW_PATH)
    
    print("\n✓ Visualization complete!")
    print("Close the plot window to exit.")
    
    plt.show()
