"""
03 - Visualize Manning's Roughness Coefficient (Land Use)
===========================================================
Load and render land use/land cover data (Manning's n coefficient).
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm

# ============================================================================
# Configuration
# ============================================================================

# Path to Manning coefficient file
DATA_ROOT = Path("..") / "FloodCastBench_Dataset-and-Models-main" / "Data_Generation_Code" / "FloodCastBench"
MANNING_PATH = DATA_ROOT / "Relevant_data" / "Land_use_and_land_cover" / "Australia.tif"
TFW_PATH = DATA_ROOT / "Relevant_data" / "Land_use_and_land_cover" / "Australia.tfw"

# Land cover classification based on Manning values
LAND_COVER_CATEGORIES = {
    'Water/Smooth Surface': (0.000, 0.015),
    'Concrete/Pavement': (0.015, 0.020),
    'Short Grass': (0.020, 0.030),
    'Cropland/Farmland': (0.030, 0.050),
    'Shrubland': (0.050, 0.080),
    'Forest/Dense Vegetation': (0.080, 0.200)
}

# ============================================================================
# Functions
# ============================================================================

def visualize_manning(manning_path):
    """
    Comprehensive Manning's roughness coefficient visualization.
    
    Args:
        manning_path: Path to Manning coefficient GeoTIFF file
    """
    # Read Manning coefficient data
    print("Loading Manning's roughness coefficient...")
    manning = tifffile.imread(manning_path)
    print(f"✓ Manning data loaded: {manning.shape} pixels")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # ========================================================================
    # 1. Manning Coefficient Map (continuous)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(manning, cmap='YlOrRd', vmin=0, vmax=0.15)
    ax1.set_title("Manning's Roughness Coefficient (n)", fontweight='bold', fontsize=11)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Manning's n", rotation=270, labelpad=20)
    
    # ========================================================================
    # 2. Classified Land Cover Map
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create discrete colormap
    n_classes = len(LAND_COVER_CATEGORIES)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    cmap_discrete = ListedColormap(colors)
    
    # Classify Manning values
    classified = np.zeros_like(manning)
    for i, (label, (min_val, max_val)) in enumerate(LAND_COVER_CATEGORIES.items()):
        mask = (manning >= min_val) & (manning < max_val)
        classified[mask] = i
    
    im2 = ax2.imshow(classified, cmap=cmap_discrete, vmin=0, vmax=n_classes-1)
    ax2.set_title('Land Cover Classification', fontweight='bold', fontsize=11)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    # Add legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=colors[i], label=label) 
               for i, label in enumerate(LAND_COVER_CATEGORIES.keys())]
    ax2.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), 
               fontsize=9, framealpha=0.9)
    
    # ========================================================================
    # 3. Flow Resistance Zones
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Create 3-zone classification (Low, Medium, High resistance)
    resistance_zones = np.zeros_like(manning)
    resistance_zones[manning < 0.025] = 0  # Low resistance
    resistance_zones[(manning >= 0.025) & (manning < 0.050)] = 1  # Medium
    resistance_zones[manning >= 0.050] = 2  # High resistance
    
    colors_zones = ['lightblue', 'yellow', 'darkred']
    cmap_zones = ListedColormap(colors_zones)
    
    im3 = ax3.imshow(resistance_zones, cmap=cmap_zones, vmin=0, vmax=2)
    ax3.set_title('Flow Resistance Zones', fontweight='bold', fontsize=11)
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    
    # Legend for zones
    patches_zones = [
        mpatches.Patch(color='lightblue', label='Low (n < 0.025)'),
        mpatches.Patch(color='yellow', label='Medium (0.025 ≤ n < 0.050)'),
        mpatches.Patch(color='darkred', label='High (n ≥ 0.050)')
    ]
    ax3.legend(handles=patches_zones, loc='upper right', fontsize=9, framealpha=0.9)
    
    # ========================================================================
    # 4. Manning Coefficient Histogram
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(manning.flatten(), bins=100, color='brown', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(manning), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(manning):.4f}')
    ax4.axvline(np.median(manning), color='blue', linestyle='--', linewidth=2,
                label=f'Median: {np.median(manning):.4f}')
    ax4.set_xlabel("Manning's n", fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Distribution of Manning Coefficients', fontweight='bold', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # 5. Land Cover Pie Chart
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Calculate percentages for each land cover type
    percentages = []
    labels_with_pct = []
    for label, (min_val, max_val) in LAND_COVER_CATEGORIES.items():
        mask = (manning >= min_val) & (manning < max_val)
        pct = 100 * np.sum(mask) / manning.size
        percentages.append(pct)
        labels_with_pct.append(f'{label}\n({pct:.1f}%)')
    
    # Filter out zero/near-zero percentages for pie chart
    percentages_filtered = []
    labels_filtered = []
    colors_filtered = []
    for i, (pct, label) in enumerate(zip(percentages, labels_with_pct)):
        if pct > 0.01:  # Only include if > 0.01%
            percentages_filtered.append(pct)
            labels_filtered.append(label)
            colors_filtered.append(colors[i])
    
    if percentages_filtered:
        ax5.pie(percentages_filtered, labels=labels_filtered, colors=colors_filtered, 
                autopct='', startangle=90, textprops={'fontsize': 9})
        ax5.set_title('Land Cover Distribution', fontweight='bold', fontsize=11)
    else:
        ax5.text(0.5, 0.5, 'No valid data\nfor pie chart', ha='center', va='center',
                 transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Land Cover Distribution', fontweight='bold', fontsize=11)
    
    # ========================================================================
    # 6. Spatial Patterns (Texture Analysis)
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Calculate local variance (texture)
    from scipy.ndimage import generic_filter
    local_var = generic_filter(manning, np.var, size=5)
    
    im6 = ax6.imshow(local_var, cmap='viridis', vmin=0, vmax=np.percentile(local_var, 95))
    ax6.set_title('Surface Heterogeneity\n(local variance)', fontweight='bold', fontsize=11)
    ax6.set_xlabel('X (pixels)')
    ax6.set_ylabel('Y (pixels)')
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.set_label('Local Variance', rotation=270, labelpad=20)
    
    # ========================================================================
    # 7. Flow Velocity Potential (1/n)
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    
    # Higher 1/n means potentially faster flow
    # Handle invalid values (zeros, negatives, NaNs)
    manning_safe = np.where((manning > 0) & np.isfinite(manning), manning, 0.001)
    flow_potential = 1.0 / (manning_safe + 0.001)  # Add small value to avoid division by zero
    
    # Filter out invalid results
    flow_potential = np.where(np.isfinite(flow_potential), flow_potential, 0)
    
    im7 = ax7.imshow(flow_potential, cmap='RdYlGn', vmin=0, vmax=np.percentile(flow_potential, 95))
    ax7.set_title('Flow Velocity Potential (1/n)\nHigher = Faster Flow', 
                  fontweight='bold', fontsize=11)
    ax7.set_xlabel('X (pixels)')
    ax7.set_ylabel('Y (pixels)')
    cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    cbar7.set_label('1/n', rotation=270, labelpad=20)
    
    # ========================================================================
    # 8. Horizontal Profile
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    
    center_y = manning.shape[0] // 2
    profile = manning[center_y, :]
    x_pixels = np.arange(len(profile))
    
    ax8.plot(x_pixels, profile, 'b-', linewidth=2)
    ax8.fill_between(x_pixels, 0, profile, alpha=0.3)
    ax8.set_xlabel('X Position (pixels)', fontweight='bold')
    ax8.set_ylabel("Manning's n", fontweight='bold')
    ax8.set_title(f'Horizontal Profile (row {center_y})', fontweight='bold', fontsize=11)
    ax8.grid(True, alpha=0.3)
    
    # ========================================================================
    # 9. Statistics Summary
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    stats_text = "═══ MANNING COEFFICIENT STATISTICS ═══\n\n"
    stats_text += f"Shape: {manning.shape[0]} × {manning.shape[1]} pixels\n\n"
    
    stats_text += "Manning's n:\n"
    stats_text += f"  Min:    {np.min(manning):>8.4f}\n"
    stats_text += f"  Max:    {np.max(manning):>8.4f}\n"
    stats_text += f"  Mean:   {np.mean(manning):>8.4f}\n"
    stats_text += f"  Median: {np.median(manning):>8.4f}\n"
    stats_text += f"  Std:    {np.std(manning):>8.4f}\n\n"
    
    stats_text += "Land Cover Distribution:\n"
    for i, (label, (min_val, max_val)) in enumerate(LAND_COVER_CATEGORIES.items()):
        mask = (manning >= min_val) & (manning < max_val)
        pct = 100 * np.sum(mask) / manning.size
        pixels = np.sum(mask)
        stats_text += f"  {label[:20]:20s}: {pct:5.1f}%\n"
    
    stats_text += "\nResistance Zones:\n"
    low_pct = 100 * np.sum(resistance_zones == 0) / resistance_zones.size
    med_pct = 100 * np.sum(resistance_zones == 1) / resistance_zones.size
    high_pct = 100 * np.sum(resistance_zones == 2) / resistance_zones.size
    stats_text += f"  Low Resistance:    {low_pct:5.1f}%\n"
    stats_text += f"  Medium Resistance: {med_pct:5.1f}%\n"
    stats_text += f"  High Resistance:   {high_pct:5.1f}%\n"
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Main title
    fig.suptitle("Manning's Roughness Coefficient - Land Use/Land Cover Analysis", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = Path("outputs") / "03_manning_visualization.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Print statistics to console
    print("\n" + "="*60)
    print(stats_text)
    print("="*60)
    
    return fig, manning


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("MANNING'S ROUGHNESS COEFFICIENT VISUALIZATION")
    print("="*60)
    
    # Check if Manning file exists
    if not MANNING_PATH.exists():
        print(f"\n✗ ERROR: Manning file not found at: {MANNING_PATH}")
        print("Please update DATA_ROOT in the script to match your directory structure.")
        exit(1)
    
    # Visualize Manning coefficient
    fig, manning_data = visualize_manning(MANNING_PATH)
    
    print("\n✓ Visualization complete!")
    print("Close the plot window to exit.")
    
    plt.show()
