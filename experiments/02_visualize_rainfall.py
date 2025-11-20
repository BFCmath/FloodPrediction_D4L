"""
02 - Visualize Rainfall Data
==============================
Load and render rainfall time series data for Australia flood event.
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from glob import glob
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

# Path to rainfall data
DATA_ROOT = Path("..") / "FloodCastBench_Dataset-and-Models-main" / "Data_Generation_Code" / "FloodCastBench"
RAINFALL_DIR = DATA_ROOT / "Relevant_data" / "Rainfall" / "Australia_flood"

# ============================================================================
# Functions
# ============================================================================

def load_rainfall_timeseries(rainfall_dir):
    """Load all rainfall files in temporal sequence"""
    rain_files = sorted(list(rainfall_dir.glob("*.tif")))
    
    if not rain_files:
        raise FileNotFoundError(f"No rainfall files found in {rainfall_dir}")
    
    print(f"Found {len(rain_files)} rainfall timesteps")
    
    # Load all data
    all_rain = []
    for rf in rain_files:
        rain = tifffile.imread(rf)
        all_rain.append(rain)
    
    return np.array(all_rain), rain_files


def visualize_rainfall(rainfall_dir):
    """
    Comprehensive rainfall visualization with temporal analysis.
    
    Args:
        rainfall_dir: Path to directory containing rainfall GeoTIFF files
    """
    print("Loading rainfall data...")
    all_rain, rain_files = load_rainfall_timeseries(rainfall_dir)
    print(f"✓ Loaded {len(rain_files)} timesteps, shape: {all_rain.shape}")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    
    # ========================================================================
    # 1. First timestep
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(all_rain[0], cmap='Blues', vmin=0, vmax=50)
    ax1.set_title(f'Rainfall at t=0\n{rain_files[0].stem}', fontweight='bold', fontsize=10)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Rain (mm/hr)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 2. Middle timestep
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    mid_idx = len(rain_files) // 2
    im2 = ax2.imshow(all_rain[mid_idx], cmap='Blues', vmin=0, vmax=50)
    ax2.set_title(f'Rainfall at t={mid_idx}\n{rain_files[mid_idx].stem}', 
                  fontweight='bold', fontsize=10)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Rain (mm/hr)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 3. Last timestep
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(all_rain[-1], cmap='Blues', vmin=0, vmax=50)
    ax3.set_title(f'Rainfall at t={len(rain_files)-1}\n{rain_files[-1].stem}', 
                  fontweight='bold', fontsize=10)
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Rain (mm/hr)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 4. Maximum rainfall (over all time)
    # ========================================================================
    ax4 = fig.add_subplot(gs[0, 3])
    max_rain = np.max(all_rain, axis=0)
    im4 = ax4.imshow(max_rain, cmap='jet', vmin=0, vmax=np.max(max_rain))
    ax4.set_title('Maximum Rainfall\n(all timesteps)', fontweight='bold', fontsize=10)
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('Max Rain (mm/hr)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 5. Mean rainfall (temporal average)
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 0])
    mean_rain = np.mean(all_rain, axis=0)
    im5 = ax5.imshow(mean_rain, cmap='YlGnBu', vmin=0, vmax=np.max(mean_rain))
    ax5.set_title('Mean Rainfall\n(temporal average)', fontweight='bold', fontsize=10)
    ax5.set_xlabel('X (pixels)')
    ax5.set_ylabel('Y (pixels)')
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.set_label('Mean Rain (mm/hr)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 6. Cumulative rainfall (total)
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 1])
    # Assuming 30-minute timesteps
    cumulative = np.sum(all_rain, axis=0) * 0.5  # Convert to total mm
    im6 = ax6.imshow(cumulative, cmap='YlOrRd', vmin=0, vmax=np.max(cumulative))
    ax6.set_title('Cumulative Rainfall\n(total event)', fontweight='bold', fontsize=10)
    ax6.set_xlabel('X (pixels)')
    ax6.set_ylabel('Y (pixels)')
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.set_label('Total Rain (mm)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 7. Rainfall variance (spatial)
    # ========================================================================
    ax7 = fig.add_subplot(gs[1, 2])
    rain_variance = np.var(all_rain, axis=0)
    im7 = ax7.imshow(rain_variance, cmap='Reds', vmin=0, vmax=np.percentile(rain_variance, 95))
    ax7.set_title('Rainfall Variability\n(temporal variance)', fontweight='bold', fontsize=10)
    ax7.set_xlabel('X (pixels)')
    ax7.set_ylabel('Y (pixels)')
    cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    cbar7.set_label('Variance (mm²/hr²)', rotation=270, labelpad=20)
    
    # ========================================================================
    # 8. Rainfall intensity distribution
    # ========================================================================
    ax8 = fig.add_subplot(gs[1, 3])
    # Histogram of all rainfall values
    rain_flat = all_rain.flatten()
    rain_nonzero = rain_flat[rain_flat > 0.1]  # Only non-zero rainfall
    
    ax8.hist(rain_nonzero, bins=100, color='blue', alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Rainfall Intensity (mm/hr)', fontweight='bold')
    ax8.set_ylabel('Frequency', fontweight='bold')
    ax8.set_title('Rainfall Intensity Distribution\n(non-zero values)', 
                  fontweight='bold', fontsize=10)
    ax8.axvline(np.mean(rain_nonzero), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(rain_nonzero):.2f} mm/hr')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # ========================================================================
    # 9. Time series at center point
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 0:2])
    center_y, center_x = all_rain.shape[1]//2, all_rain.shape[2]//2
    time_series_center = all_rain[:, center_y, center_x]
    time_hours = np.arange(len(time_series_center)) * 0.5  # 30-min intervals
    
    ax9.plot(time_hours, time_series_center, 'b-', linewidth=2, label='Center pixel')
    ax9.fill_between(time_hours, 0, time_series_center, alpha=0.3)
    ax9.set_xlabel('Time (hours)', fontweight='bold', fontsize=11)
    ax9.set_ylabel('Rainfall (mm/hr)', fontweight='bold', fontsize=11)
    ax9.set_title(f'Rainfall Time Series at Center ({center_x}, {center_y})', 
                  fontweight='bold', fontsize=11)
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    # ========================================================================
    # 10. Spatial average time series
    # ========================================================================
    ax10 = fig.add_subplot(gs[2, 2:4])
    spatial_mean = np.mean(all_rain, axis=(1, 2))
    spatial_max = np.max(all_rain, axis=(1, 2))
    
    ax10.plot(time_hours, spatial_mean, 'b-', linewidth=2, label='Spatial Mean')
    ax10.plot(time_hours, spatial_max, 'r--', linewidth=2, label='Spatial Max')
    ax10.fill_between(time_hours, 0, spatial_mean, alpha=0.3)
    ax10.set_xlabel('Time (hours)', fontweight='bold', fontsize=11)
    ax10.set_ylabel('Rainfall (mm/hr)', fontweight='bold', fontsize=11)
    ax10.set_title('Domain-Averaged Rainfall Evolution', fontweight='bold', fontsize=11)
    ax10.grid(True, alpha=0.3)
    ax10.legend()
    
    # Main title
    fig.suptitle('Rainfall Visualization - Australia Flood Event', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = Path("outputs") / "02_rainfall_visualization.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("RAINFALL STATISTICS")
    print("="*60)
    print(f"Spatial shape: {all_rain[0].shape}")
    print(f"Number of timesteps: {len(rain_files)}")
    print(f"Temporal resolution: 30 minutes")
    print(f"Total duration: {len(rain_files) * 0.5:.1f} hours ({len(rain_files) * 0.5 / 24:.1f} days)")
    print(f"\nRainfall Intensity:")
    print(f"  Max (any time/location): {np.max(all_rain):.2f} mm/hr")
    print(f"  Mean (all data): {np.mean(all_rain):.2f} mm/hr")
    print(f"  Mean (non-zero): {np.mean(rain_nonzero):.2f} mm/hr")
    print(f"\nCumulative Rainfall:")
    print(f"  Max (any location): {np.max(cumulative):.2f} mm")
    print(f"  Mean (domain avg): {np.mean(cumulative):.2f} mm")
    print(f"  Total volume: {np.sum(cumulative):.2f} mm×pixels")
    print("="*60)
    
    return fig, all_rain


def create_rainfall_animation(rainfall_dir, max_timesteps=None, output_gif='rainfall_animation.gif', fps=10):
    """
    Create animated visualization of rainfall evolution over time.
    
    Args:
        rainfall_dir: Path to directory containing rainfall GeoTIFF files
        max_timesteps: Maximum number of timesteps to animate (None = all)
        output_gif: Output filename for animation
        fps: Frames per second for animation
    """
    print("\n" + "="*60)
    print("CREATING RAINFALL ANIMATION")
    print("="*60)
    
    # Load all rainfall files
    rain_files = sorted(list(rainfall_dir.glob("*.tif")))
    
    if not rain_files:
        raise FileNotFoundError(f"No rainfall files found in {rainfall_dir}")
    
    # Limit to max_timesteps if specified
    if max_timesteps and max_timesteps < len(rain_files):
        rain_files = rain_files[:max_timesteps]
    
    print(f"\nFound {len(rain_files)} rainfall timesteps")
    print(f"Creating animation with {len(rain_files)} frames...")
    
    # Load first frame to get dimensions and max value
    first_rain = tifffile.imread(rain_files[0])
    
    # Calculate global max for consistent colorbar
    print("Calculating global maximum for color scale...")
    all_max_values = []
    for rf in tqdm(rain_files, desc="Scanning files"):
        rain = tifffile.imread(rf)
        all_max_values.append(np.max(rain))
    
    global_max = max(all_max_values)
    vmax = min(global_max, 100)  # Cap at 100 mm/hr for better visualization
    
    print(f"✓ Global max rainfall: {global_max:.2f} mm/hr")
    print(f"✓ Using colorbar max: {vmax:.2f} mm/hr")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[3, 1])
    
    # Main rainfall map
    ax_map = fig.add_subplot(gs[0, :])
    ax_time = fig.add_subplot(gs[1, 0])
    ax_stats = fig.add_subplot(gs[1, 1])
    
    # Initialize plots
    im = ax_map.imshow(first_rain, cmap='Blues', vmin=0, vmax=vmax, animated=True)
    ax_map.set_xlabel('X (pixels)', fontsize=12, fontweight='bold')
    ax_map.set_ylabel('Y (pixels)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label('Rainfall Intensity (mm/hr)', rotation=270, labelpad=25, fontsize=12, fontweight='bold')
    
    # Time series plot (will be updated)
    time_hours = np.arange(len(rain_files)) * 0.5
    spatial_mean = np.zeros(len(rain_files))
    spatial_max = np.zeros(len(rain_files))
    
    line_mean, = ax_time.plot([], [], 'b-', linewidth=2, label='Spatial Mean')
    line_max, = ax_time.plot([], [], 'r--', linewidth=2, label='Spatial Max')
    current_time_marker = ax_time.axvline(0, color='green', linestyle='--', linewidth=2, label='Current Time')
    ax_time.set_xlim(0, time_hours[-1])
    ax_time.set_ylim(0, vmax)
    ax_time.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
    ax_time.set_ylabel('Rainfall (mm/hr)', fontsize=11, fontweight='bold')
    ax_time.set_title('Domain-Averaged Rainfall', fontsize=12, fontweight='bold')
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc='upper right')
    
    # Statistics text
    stats_text = ax_stats.text(0.1, 0.9, '', transform=ax_stats.transAxes,
                               fontsize=11, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax_stats.axis('off')
    ax_stats.set_title('Current Statistics', fontsize=12, fontweight='bold')
    
    # Title
    title = fig.suptitle('', fontsize=16, fontweight='bold', y=0.98)
    
    # Pre-calculate all spatial statistics for time series
    print("\nCalculating statistics for all timesteps...")
    for i, rf in enumerate(tqdm(rain_files, desc="Processing")):
        rain = tifffile.imread(rf)
        spatial_mean[i] = np.mean(rain)
        spatial_max[i] = np.max(rain)
    
    def init():
        """Initialize animation"""
        im.set_data(first_rain)
        line_mean.set_data([], [])
        line_max.set_data([], [])
        return [im, line_mean, line_max, current_time_marker, title, stats_text]
    
    def update(frame):
        """Update function for animation"""
        # Load rainfall data for this frame
        rain = tifffile.imread(rain_files[frame])
        
        # Update rainfall map
        im.set_data(rain)
        
        # Update time series (show up to current frame)
        line_mean.set_data(time_hours[:frame+1], spatial_mean[:frame+1])
        line_max.set_data(time_hours[:frame+1], spatial_max[:frame+1])
        current_time_marker.set_xdata([time_hours[frame], time_hours[frame]])
        
        # Update title
        current_time = time_hours[frame]
        title.set_text(f'Rainfall Animation - t = {current_time:.1f} hours ({current_time/24:.2f} days) - Frame {frame+1}/{len(rain_files)}')
        
        # Update statistics
        wet_pixels = np.sum(rain > 0.1)
        total_pixels = rain.size
        wet_pct = 100 * wet_pixels / total_pixels
        
        stats_str = f"═══ TIMESTEP {frame} ═══\n\n"
        stats_str += f"Time: {current_time:.1f} hours\n"
        stats_str += f"      ({current_time/24:.2f} days)\n\n"
        stats_str += f"Rainfall:\n"
        stats_str += f"  Max:  {np.max(rain):>7.2f} mm/hr\n"
        stats_str += f"  Mean: {np.mean(rain):>7.2f} mm/hr\n"
        if wet_pixels > 0:
            stats_str += f"  Mean (wet): {np.mean(rain[rain>0.1]):>7.2f} mm/hr\n"
        stats_str += f"\nCoverage:\n"
        stats_str += f"  Wet area: {wet_pct:>5.1f}%\n"
        stats_str += f"  Wet pixels: {wet_pixels:>7,}\n"
        
        stats_text.set_text(stats_str)
        
        return [im, line_mean, line_max, current_time_marker, title, stats_text]
    
    # Create animation
    print(f"\nGenerating animation...")
    anim = FuncAnimation(
        fig, 
        update, 
        init_func=init,
        frames=len(rain_files), 
        interval=1000/fps,  # milliseconds per frame
        blit=True,
        repeat=True
    )
    
    # Save animation
    output_path = Path("outputs") / output_gif
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"\nSaving animation to: {output_path}")
    print("This may take a few minutes...")
    
    # Use PillowWriter for GIF
    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer, dpi=100,
              progress_callback=lambda i, n: print(f"  Saving frame {i+1}/{n}", end='\r'))
    
    print(f"\n\n✓ Animation saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"  Duration: {len(rain_files) / fps:.1f} seconds at {fps} fps")
    print(f"  Frames: {len(rain_files)}")
    
    return anim, fig


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("RAINFALL VISUALIZATION - Australia Flood Event")
    print("="*60)
    
    # Check if rainfall directory exists
    if not RAINFALL_DIR.exists():
        print(f"\n✗ ERROR: Rainfall directory not found at: {RAINFALL_DIR}")
        print("Please update DATA_ROOT in the script to match your directory structure.")
        exit(1)
    
    # Ask user what to do
    print("\nChoose visualization mode:")
    print("  1. Static visualization (default)")
    print("  2. Create animation (all timesteps)")
    print("  3. Create animation (first 100 timesteps - faster)")
    print("  4. Both static + animation")
    
    choice = input("\nEnter choice [1-4] (or press Enter for 1): ").strip()
    
    if choice == "" or choice == "1":
        # Static visualization only
        fig, rain_data = visualize_rainfall(RAINFALL_DIR)
        print("\n✓ Visualization complete!")
        print("Close the plot window to exit.")
        plt.show()
        
    elif choice == "2":
        # Full animation (all timesteps)
        anim, fig = create_rainfall_animation(
            RAINFALL_DIR, 
            max_timesteps=None,  # All timesteps
            output_gif='02_rainfall_animation_full.gif',
            fps=10
        )
        print("\n✓ Animation complete!")
        print("You can view the GIF file in outputs/ folder")
        
    elif choice == "3":
        # Animation with first 100 timesteps
        anim, fig = create_rainfall_animation(
            RAINFALL_DIR, 
            max_timesteps=100,
            output_gif='02_rainfall_animation_100frames.gif',
            fps=10
        )
        print("\n✓ Animation complete!")
        print("You can view the GIF file in outputs/ folder")
        
    elif choice == "4":
        # Both static and animation
        # First create static visualization
        fig_static, rain_data = visualize_rainfall(RAINFALL_DIR)
        plt.close(fig_static)  # Close without showing
        
        # Then create animation
        anim, fig_anim = create_rainfall_animation(
            RAINFALL_DIR, 
            max_timesteps=None,
            output_gif='02_rainfall_animation_full.gif',
            fps=10
        )
        
        print("\n✓ Both visualizations complete!")
        print("Check the outputs/ folder for all files.")
        
    else:
        print("\n✗ Invalid choice. Creating static visualization...")
        fig, rain_data = visualize_rainfall(RAINFALL_DIR)
        plt.show()
