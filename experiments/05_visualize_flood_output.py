"""
05 - Visualize Flood Simulation Output
========================================
Load and render simulated flood depth evolution over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from glob import glob
import matplotlib.animation as animation

# ============================================================================
# Configuration
# ============================================================================

# Path to simulation output
DATA_ROOT = Path("..") / "FloodCastBench_Dataset-and-Models-main" / "Data_Generation_Code" / "FloodCastBench"
OUTPUT_30M = DATA_ROOT / "High-fidelity_flood_forecasting" / "30m" / "Australia"
OUTPUT_60M = DATA_ROOT / "High-fidelity_flood_forecasting" / "60m" / "Australia"
DEM_PATH = DATA_ROOT / "Relevant_data" / "DEM" / "Australia_DEM.tif"

# Use 30m by default, fallback to 60m
OUTPUT_DIR = OUTPUT_30M if OUTPUT_30M.exists() else OUTPUT_60M

# ============================================================================
# Functions
# ============================================================================

def load_flood_timeseries(output_dir, max_files=None):
    """Load flood depth time series"""
    flood_files = sorted(list(output_dir.glob("*.tif")))
    
    if not flood_files:
        raise FileNotFoundError(f"No flood output files found in {output_dir}")
    
    if max_files:
        flood_files = flood_files[:max_files]
    
    print(f"Loading {len(flood_files)} flood depth timesteps...")
    
    # Load all data
    all_floods = []
    timestamps = []
    
    for ff in flood_files:
        flood = tifffile.imread(ff)
        all_floods.append(flood)
        
        # Extract timestamp from filename
        try:
            timestamp = float(ff.stem)
            timestamps.append(timestamp)
        except:
            timestamps.append(0)
    
    return np.array(all_floods), timestamps, flood_files


def visualize_flood_output(output_dir, dem_path=None):
    """
    Comprehensive flood output visualization.
    
    Args:
        output_dir: Path to directory containing flood depth outputs
        dem_path: Optional path to DEM for overlay
    """
    # Load flood data (limit to first 100 for memory)
    all_floods, timestamps, flood_files = load_flood_timeseries(output_dir, max_files=100)
    print(f"✓ Loaded {len(flood_files)} timesteps, shape: {all_floods.shape}")
    
    # Load DEM if available
    dem = None
    if dem_path and dem_path.exists():
        dem = tifffile.imread(dem_path)
        print(f"✓ DEM loaded: {dem.shape}")
    
    # Convert timestamps to hours
    times_hours = np.array(timestamps) / 3600
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.35)
    
    # ========================================================================
    # 1. Initial flood depth (t=0)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    vmax_global = np.percentile(all_floods, 99)  # Use 99th percentile for better visualization
    im1 = ax1.imshow(all_floods[0], cmap='Blues', vmin=0, vmax=vmax_global)
    ax1.set_title(f'Flood Depth at t=0\n({flood_files[0].stem}s)', 
                  fontweight='bold', fontsize=10)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Depth (m)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 2. Mid-simulation
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    mid_idx = len(all_floods) // 2
    im2 = ax2.imshow(all_floods[mid_idx], cmap='Blues', vmin=0, vmax=vmax_global)
    ax2.set_title(f'Flood Depth at t={mid_idx}\n({times_hours[mid_idx]:.1f} hrs)', 
                  fontweight='bold', fontsize=10)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Depth (m)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 3. Final flood depth
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(all_floods[-1], cmap='Blues', vmin=0, vmax=vmax_global)
    ax3.set_title(f'Flood Depth at t={len(all_floods)-1}\n({times_hours[-1]:.1f} hrs)', 
                  fontweight='bold', fontsize=10)
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Depth (m)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 4. Maximum flood extent (envelope)
    # ========================================================================
    ax4 = fig.add_subplot(gs[0, 3])
    max_flood = np.max(all_floods, axis=0)
    im4 = ax4.imshow(max_flood, cmap='jet', vmin=0, vmax=np.max(max_flood))
    ax4.set_title('Maximum Flood Depth\n(all timesteps)', fontweight='bold', fontsize=10)
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('Max Depth (m)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 5. Flood depth change (final - initial)
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 0])
    depth_change = all_floods[-1] - all_floods[0]
    im5 = ax5.imshow(depth_change, cmap='RdBu_r', 
                     vmin=-np.max(np.abs(depth_change)), 
                     vmax=np.max(np.abs(depth_change)))
    ax5.set_title('Flood Depth Change\n(final - initial)', fontweight='bold', fontsize=10)
    ax5.set_xlabel('X (pixels)')
    ax5.set_ylabel('Y (pixels)')
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.set_label('Change (m)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 6. Flood inundation frequency
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 1])
    # How many timesteps was each pixel flooded?
    inundation_freq = np.sum(all_floods > 0.05, axis=0) / len(all_floods) * 100
    im6 = ax6.imshow(inundation_freq, cmap='YlOrRd', vmin=0, vmax=100)
    ax6.set_title('Inundation Frequency\n(% of time flooded)', fontweight='bold', fontsize=10)
    ax6.set_xlabel('X (pixels)')
    ax6.set_ylabel('Y (pixels)')
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.set_label('Frequency (%)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 7. Flood arrival time
    # ========================================================================
    ax7 = fig.add_subplot(gs[1, 2])
    # When did flooding first occur (depth > 5cm)?
    arrival_time = np.full(all_floods[0].shape, -1, dtype=float)
    for t_idx, flood in enumerate(all_floods):
        first_flood = (flood > 0.05) & (arrival_time < 0)
        arrival_time[first_flood] = times_hours[t_idx]
    
    arrival_time_masked = np.ma.masked_where(arrival_time < 0, arrival_time)
    im7 = ax7.imshow(arrival_time_masked, cmap='viridis')
    ax7.set_title('Flood Arrival Time\n(when depth > 5cm)', fontweight='bold', fontsize=10)
    ax7.set_xlabel('X (pixels)')
    ax7.set_ylabel('Y (pixels)')
    cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    cbar7.set_label('Time (hours)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 8. Overlay on DEM (final state)
    # ========================================================================
    ax8 = fig.add_subplot(gs[1, 3])
    if dem is not None:
        ax8.imshow(dem, cmap='terrain', alpha=0.6)
        flood_overlay = np.ma.masked_where(all_floods[-1] <= 0.01, all_floods[-1])
        im8 = ax8.imshow(flood_overlay, cmap='Blues', alpha=0.7, vmin=0, vmax=vmax_global)
        ax8.set_title('Final Flood Overlaid on DEM', fontweight='bold', fontsize=10)
        cbar8 = plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
        cbar8.set_label('Depth (m)', rotation=270, labelpad=20)
    else:
        ax8.text(0.5, 0.5, 'DEM not available', ha='center', va='center',
                transform=ax8.transAxes, fontsize=12)
        ax8.set_title('DEM Overlay (N/A)', fontweight='bold', fontsize=10)
    ax8.set_xlabel('X (pixels)')
    ax8.set_ylabel('Y (pixels)')
    
    # ========================================================================
    # 9. Maximum depth evolution
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 0:2])
    max_depths = np.max(all_floods, axis=(1, 2))
    mean_depths = np.mean(all_floods[all_floods > 0.01])  # Mean of flooded areas
    
    ax9.plot(times_hours, max_depths, 'b-', linewidth=2, label='Max Depth')
    ax9.set_xlabel('Time (hours)', fontweight='bold')
    ax9.set_ylabel('Water Depth (m)', fontweight='bold')
    ax9.set_title('Maximum Flood Depth Evolution', fontweight='bold', fontsize=11)
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    # ========================================================================
    # 10. Flooded area evolution
    # ========================================================================
    ax10 = fig.add_subplot(gs[2, 2:4])
    flooded_area = np.sum(all_floods > 0.05, axis=(1, 2))  # Count of cells > 5cm
    
    ax10.plot(times_hours, flooded_area, 'r-', linewidth=2)
    ax10.fill_between(times_hours, 0, flooded_area, alpha=0.3, color='red')
    ax10.set_xlabel('Time (hours)', fontweight='bold')
    ax10.set_ylabel('Flooded Area (pixels)', fontweight='bold')
    ax10.set_title('Flooded Area Evolution (depth > 5cm)', fontweight='bold', fontsize=11)
    ax10.grid(True, alpha=0.3)
    
    # ========================================================================
    # 11. Depth distribution at different times
    # ========================================================================
    ax11 = fig.add_subplot(gs[3, 0:2])
    
    # Plot histograms for initial, mid, and final states
    for idx, label, color in [(0, 'Initial', 'blue'), 
                               (mid_idx, 'Middle', 'green'), 
                               (-1, 'Final', 'red')]:
        depths = all_floods[idx][all_floods[idx] > 0.01].flatten()
        if len(depths) > 0:
            ax11.hist(depths, bins=50, alpha=0.5, label=label, color=color, edgecolor='black')
    
    ax11.set_xlabel('Water Depth (m)', fontweight='bold')
    ax11.set_ylabel('Frequency', fontweight='bold')
    ax11.set_title('Flood Depth Distribution at Different Times', fontweight='bold', fontsize=11)
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # ========================================================================
    # 12. Statistics Summary
    # ========================================================================
    ax12 = fig.add_subplot(gs[3, 2:4])
    ax12.axis('off')
    
    stats_text = "═══ FLOOD SIMULATION STATISTICS ═══\n\n"
    stats_text += f"Resolution: {output_dir.parent.name}\n"
    stats_text += f"Grid Size: {all_floods[0].shape[0]} × {all_floods[0].shape[1]}\n"
    stats_text += f"Timesteps: {len(all_floods)}\n"
    stats_text += f"Duration: {times_hours[-1]:.1f} hours ({times_hours[-1]/24:.2f} days)\n\n"
    
    stats_text += "Initial State:\n"
    stats_text += f"  Max depth: {np.max(all_floods[0]):>8.3f} m\n"
    stats_text += f"  Flooded:   {np.sum(all_floods[0] > 0.05):>8d} pixels\n\n"
    
    stats_text += "Final State:\n"
    stats_text += f"  Max depth: {np.max(all_floods[-1]):>8.3f} m\n"
    stats_text += f"  Mean depth: {np.mean(all_floods[-1][all_floods[-1] > 0.05]):>8.3f} m\n"
    stats_text += f"  Flooded:   {np.sum(all_floods[-1] > 0.05):>8d} pixels\n\n"
    
    stats_text += "Maximum Envelope:\n"
    stats_text += f"  Peak depth: {np.max(max_flood):>8.3f} m\n"
    stats_text += f"  Ever flooded: {np.sum(max_flood > 0.05):>8d} pixels\n"
    stats_text += f"  Percent wet: {100*np.sum(max_flood > 0.05)/max_flood.size:>7.2f}%\n\n"
    
    stats_text += "Temporal Dynamics:\n"
    stats_text += f"  Peak time: {times_hours[np.argmax(max_depths)]:>8.1f} hrs\n"
    stats_text += f"  Max area: {np.max(flooded_area):>8d} pixels\n"
    
    ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Main title
    resolution = output_dir.parent.name
    fig.suptitle(f'Flood Simulation Output Visualization - {resolution} Resolution', 
                 fontsize=16, fontweight='bold', y=0.99)
    
    # Save figure
    output_path = Path("outputs") / f"05_flood_output_{resolution}_visualization.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Print statistics to console
    print("\n" + "="*60)
    print(stats_text)
    print("="*60)
    
    return fig, all_floods


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("FLOOD SIMULATION OUTPUT VISUALIZATION")
    print("="*60)
    
    # Check if output directory exists
    if not OUTPUT_DIR.exists():
        print(f"\n✗ ERROR: Output directory not found at: {OUTPUT_DIR}")
        print("Tried:")
        print(f"  - {OUTPUT_30M}")
        print(f"  - {OUTPUT_60M}")
        print("\nPlease update DATA_ROOT in the script to match your directory structure.")
        exit(1)
    
    print(f"\nUsing output directory: {OUTPUT_DIR}")
    
    # Visualize flood output
    fig, flood_data = visualize_flood_output(OUTPUT_DIR, DEM_PATH)
    
    print("\n✓ Visualization complete!")
    print("Close the plot window to exit.")
    
    plt.show()
