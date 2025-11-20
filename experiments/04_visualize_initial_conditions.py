"""
04 - Visualize Initial Conditions
===================================
Load and render initial water depth conditions at t=0.
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Path to initial condition files
DATA_ROOT = Path("..") / "FloodCastBench_Dataset-and-Models-main" / "Data_Generation_Code" / "FloodCastBench"
IC_DIR = DATA_ROOT / "Relevant_data" / "Initial_conditions" / "High-fidelity_flood_forecasting"
DEM_PATH = DATA_ROOT / "Relevant_data" / "DEM" / "Australia_DEM.tif"

# ============================================================================
# Functions
# ============================================================================

def visualize_initial_conditions(ic_dir, dem_path=None):
    """
    Visualize initial water depth conditions.
    
    Args:
        ic_dir: Path to initial conditions directory
        dem_path: Optional path to DEM for overlay visualization
    """
    # Find initial condition files
    ic_files = list(ic_dir.glob("Australia*.tif"))
    
    if not ic_files:
        print(f"✗ No initial condition files found in {ic_dir}")
        return None, None
    
    print(f"Found {len(ic_files)} initial condition file(s):")
    for f in ic_files:
        print(f"  - {f.name}")
    
    # Load the first available IC file
    ic_path = ic_files[0]
    print(f"\nLoading: {ic_path.name}")
    h0 = tifffile.imread(ic_path)
    print(f"✓ Initial condition loaded: {h0.shape} pixels")
    
    # Load DEM if available
    dem = None
    if dem_path and dem_path.exists():
        dem = tifffile.imread(dem_path)
        print(f"✓ DEM loaded for overlay: {dem.shape} pixels")
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # ========================================================================
    # 1. Initial Water Depth
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    vmax = np.max(h0) if np.max(h0) > 0 else 0.1
    im1 = ax1.imshow(h0, cmap='Blues', vmin=0, vmax=vmax)
    ax1.set_title('Initial Water Depth (t=0)', fontweight='bold', fontsize=11)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Depth (m)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 2. Wet/Dry Classification
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    wet_threshold = 0.001  # 1mm threshold
    wet_mask = h0 > wet_threshold
    
    im2 = ax2.imshow(wet_mask, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax2.set_title(f'Wet Areas (depth > {wet_threshold}m)', fontweight='bold', fontsize=11)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    wet_pct = 100 * np.sum(wet_mask) / wet_mask.size
    ax2.text(0.02, 0.98, f'Wet: {wet_pct:.2f}%\nDry: {100-wet_pct:.2f}%',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========================================================================
    # 3. Water Depth Categories
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Classify water depth
    depth_classes = np.zeros_like(h0)
    depth_classes[h0 == 0] = 0  # Dry
    depth_classes[(h0 > 0) & (h0 <= 0.1)] = 1  # Very shallow
    depth_classes[(h0 > 0.1) & (h0 <= 0.5)] = 2  # Shallow
    depth_classes[(h0 > 0.5) & (h0 <= 1.0)] = 3  # Moderate
    depth_classes[h0 > 1.0] = 4  # Deep
    
    from matplotlib.colors import ListedColormap
    colors = ['tan', 'lightblue', 'blue', 'darkblue', 'navy']
    cmap_depth = ListedColormap(colors)
    
    im3 = ax3.imshow(depth_classes, cmap=cmap_depth, vmin=0, vmax=4)
    ax3.set_title('Water Depth Categories', fontweight='bold', fontsize=11)
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    
    # Legend
    import matplotlib.patches as mpatches
    legend_labels = ['Dry (0m)', 'Very Shallow (0-0.1m)', 'Shallow (0.1-0.5m)',
                     'Moderate (0.5-1.0m)', 'Deep (>1.0m)']
    patches = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(5)]
    ax3.legend(handles=patches, loc='upper right', fontsize=8, framealpha=0.9)
    
    # ========================================================================
    # 4. Overlay on DEM
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    if dem is not None:
        # Show DEM as background
        ax4.imshow(dem, cmap='terrain', alpha=0.7)
        # Overlay water depth with transparency
        water_overlay = np.ma.masked_where(h0 <= 0.001, h0)
        im4 = ax4.imshow(water_overlay, cmap='Blues', alpha=0.7, vmin=0, vmax=vmax)
        ax4.set_title('Initial Depth Overlaid on DEM', fontweight='bold', fontsize=11)
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label('Water Depth (m)', rotation=270, labelpad=20)
    else:
        ax4.text(0.5, 0.5, 'DEM not available', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('DEM Overlay (N/A)', fontweight='bold', fontsize=11)
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    
    # ========================================================================
    # 5. Water Depth Distribution
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Histogram of non-zero depths only
    h_nonzero = h0[h0 > wet_threshold]
    
    if len(h_nonzero) > 0:
        ax5.hist(h_nonzero, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax5.axvline(np.mean(h_nonzero), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(h_nonzero):.4f}m')
        ax5.axvline(np.median(h_nonzero), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(h_nonzero):.4f}m')
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'All cells are dry', ha='center', va='center',
                transform=ax5.transAxes, fontsize=12)
    
    ax5.set_xlabel('Water Depth (m)', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.set_title('Water Depth Distribution\n(non-zero values)', fontweight='bold', fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # ========================================================================
    # 6. Horizontal Cross-Section
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    
    center_y = h0.shape[0] // 2
    profile_h = h0[center_y, :]
    x_pixels = np.arange(len(profile_h))
    
    ax6.plot(x_pixels, profile_h, 'b-', linewidth=2, label='Water depth')
    ax6.fill_between(x_pixels, 0, profile_h, alpha=0.3)
    
    # Overlay DEM profile if available
    if dem is not None:
        profile_dem = dem[center_y, :]
        ax_dem = ax6.twinx()
        ax_dem.plot(x_pixels, profile_dem, 'brown', linestyle='--', linewidth=1.5, 
                    alpha=0.7, label='Terrain')
        ax_dem.set_ylabel('Elevation (m)', fontweight='bold', color='brown')
        ax_dem.tick_params(axis='y', labelcolor='brown')
    
    ax6.set_xlabel('X Position (pixels)', fontweight='bold')
    ax6.set_ylabel('Water Depth (m)', fontweight='bold', color='blue')
    ax6.tick_params(axis='y', labelcolor='blue')
    ax6.set_title(f'Horizontal Profile (row {center_y})', fontweight='bold', fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    # ========================================================================
    # 7. Vertical Cross-Section
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    
    center_x = h0.shape[1] // 2
    profile_v = h0[:, center_x]
    y_pixels = np.arange(len(profile_v))
    
    ax7.plot(y_pixels, profile_v, 'b-', linewidth=2)
    ax7.fill_between(y_pixels, 0, profile_v, alpha=0.3)
    ax7.set_xlabel('Y Position (pixels)', fontweight='bold')
    ax7.set_ylabel('Water Depth (m)', fontweight='bold')
    ax7.set_title(f'Vertical Profile (col {center_x})', fontweight='bold', fontsize=11)
    ax7.grid(True, alpha=0.3)
    
    # ========================================================================
    # 8. 3D Surface Plot (if water present)
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 1], projection='3d')
    
    if np.max(h0) > 0:
        # Downsample for plotting
        stride = max(1, h0.shape[0] // 50)
        X, Y = np.meshgrid(np.arange(0, h0.shape[1], stride), 
                           np.arange(0, h0.shape[0], stride))
        Z = h0[::stride, ::stride]
        
        surf = ax8.plot_surface(X, Y, Z, cmap='Blues', alpha=0.8, 
                                linewidth=0, antialiased=True)
        ax8.set_xlabel('X', fontweight='bold')
        ax8.set_ylabel('Y', fontweight='bold')
        ax8.set_zlabel('Depth (m)', fontweight='bold')
        ax8.set_title('3D Water Surface', fontweight='bold', fontsize=11)
    else:
        ax8.text2D(0.5, 0.5, 'No water present', ha='center', va='center',
                   transform=ax8.transAxes, fontsize=12)
        ax8.set_title('3D Surface (N/A)', fontweight='bold', fontsize=11)
    
    # ========================================================================
    # 9. Statistics Summary
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    stats_text = "═══ INITIAL CONDITIONS STATISTICS ═══\n\n"
    stats_text += f"File: {ic_path.name}\n"
    stats_text += f"Shape: {h0.shape[0]} × {h0.shape[1]} pixels\n\n"
    
    stats_text += "Water Depth (all cells):\n"
    stats_text += f"  Min:     {np.min(h0):>8.4f} m\n"
    stats_text += f"  Max:     {np.max(h0):>8.4f} m\n"
    stats_text += f"  Mean:    {np.mean(h0):>8.4f} m\n"
    stats_text += f"  Median:  {np.median(h0):>8.4f} m\n\n"
    
    if len(h_nonzero) > 0:
        stats_text += "Water Depth (wet cells only):\n"
        stats_text += f"  Mean:    {np.mean(h_nonzero):>8.4f} m\n"
        stats_text += f"  Median:  {np.median(h_nonzero):>8.4f} m\n\n"
    
    stats_text += "Wet/Dry Classification:\n"
    stats_text += f"  Wet cells:  {np.sum(wet_mask):>8d} ({wet_pct:>5.2f}%)\n"
    stats_text += f"  Dry cells:  {np.sum(~wet_mask):>8d} ({100-wet_pct:>5.2f}%)\n\n"
    
    stats_text += "Depth Categories:\n"
    for i, label in enumerate(legend_labels):
        count = np.sum(depth_classes == i)
        pct = 100 * count / depth_classes.size
        stats_text += f"  {label:20s}: {pct:5.2f}%\n"
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # Main title
    fig.suptitle('Initial Conditions - Water Depth at t=0', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = Path("outputs") / "04_initial_conditions_visualization.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Print statistics to console
    print("\n" + "="*60)
    print(stats_text)
    print("="*60)
    
    return fig, h0


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("INITIAL CONDITIONS VISUALIZATION")
    print("="*60)
    
    # Check if directory exists
    if not IC_DIR.exists():
        print(f"\n✗ ERROR: Initial conditions directory not found at: {IC_DIR}")
        print("Please update DATA_ROOT in the script to match your directory structure.")
        exit(1)
    
    # Visualize initial conditions
    fig, ic_data = visualize_initial_conditions(IC_DIR, DEM_PATH)
    
    if fig is not None:
        print("\n✓ Visualization complete!")
        print("Close the plot window to exit.")
        plt.show()
