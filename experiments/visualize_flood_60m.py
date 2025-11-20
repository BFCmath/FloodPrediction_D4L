"""
Visualize Flood Simulation Output - 60m Resolution
====================================================
Comprehensive visualization of flood depth evolution from the simulation.
Includes static plots, time series analysis, and optional animation.
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import warnings

# ============================================================================
# Configuration
# ============================================================================

# Path to flood simulation output
DATA_ROOT = Path("..") / "FloodCastBench_Dataset-and-Models-main" / "Data_Generation_Code" / "FloodCastBench"
FLOOD_OUTPUT_DIR = DATA_ROOT / "High-fidelity_flood_forecasting" / "60m" / "Australia"
DEM_PATH = DATA_ROOT / "Relevant_data" / "DEM" / "Australia_DEM.tif"

# ============================================================================
# Functions
# ============================================================================

def load_flood_timeseries(output_dir, max_files=None):
    """Load flood depth time series"""
    # Find all .tif files and sort them
    flood_files = sorted(list(output_dir.glob("*.tif")))
    
    if not flood_files:
        raise FileNotFoundError(f"No flood output files found in {output_dir}")
    
    print(f"Found {len(flood_files)} flood depth timesteps")
    
    if max_files:
        flood_files = flood_files[:max_files]
        print(f"Limiting to first {max_files} timesteps")
    
    # Extract timestamps from filenames (assuming format: timestamp.tif or flood_timestamp.tif)
    timestamps = []
    for ff in flood_files:
        try:
            # Try to extract numeric value from filename
            timestamp = float(ff.stem.replace('flood_', '').replace('_', ''))
            timestamps.append(timestamp)
        except:
            # If failed, use index
            timestamps.append(len(timestamps))
    
    return flood_files, timestamps


def visualize_flood_static(output_dir, dem_path=None, max_timesteps=100):
    """
    Create comprehensive static visualization of flood evolution.
    
    Args:
        output_dir: Path to directory containing flood depth outputs
        dem_path: Optional path to DEM for overlay
        max_timesteps: Maximum number of timesteps to load for analysis
    """
    print("\n" + "="*60)
    print("LOADING FLOOD DATA")
    print("="*60)
    
    # Load flood data
    flood_files, timestamps = load_flood_timeseries(output_dir, max_files=max_timesteps)
    
    print(f"\nLoading {len(flood_files)} timesteps...")
    all_floods = []
    for ff in tqdm(flood_files, desc="Reading files"):
        flood = tifffile.imread(ff)
        all_floods.append(flood)
    
    all_floods = np.array(all_floods)
    print(f"✓ Loaded flood data: {all_floods.shape}")
    
    # Load DEM if available
    dem = None
    if dem_path and dem_path.exists():
        dem = tifffile.imread(dem_path)
        print(f"✓ DEM loaded: {dem.shape}")
    
    # Convert timestamps to hours (assuming they're in seconds)
    times_hours = np.array(timestamps) / 3600
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(4, 5, hspace=0.4, wspace=0.4)
    
    # Calculate global max for consistent color scale
    vmax_global = np.percentile(all_floods, 99)
    
    # ========================================================================
    # Row 1: Snapshots at different times
    # ========================================================================
    
    # Initial state
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(all_floods[0], cmap='Blues', vmin=0, vmax=vmax_global)
    ax1.set_title(f'Initial (t=0)\n{times_hours[0]:.1f} hrs', fontweight='bold', fontsize=10)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Depth (m)')
    
    # 25% through simulation
    idx_25 = len(all_floods) // 4
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(all_floods[idx_25], cmap='Blues', vmin=0, vmax=vmax_global)
    ax2.set_title(f'25% ({idx_25})\n{times_hours[idx_25]:.1f} hrs', fontweight='bold', fontsize=10)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Depth (m)')
    
    # 50% through simulation
    idx_50 = len(all_floods) // 2
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(all_floods[idx_50], cmap='Blues', vmin=0, vmax=vmax_global)
    ax3.set_title(f'50% ({idx_50})\n{times_hours[idx_50]:.1f} hrs', fontweight='bold', fontsize=10)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Depth (m)')
    
    # 75% through simulation
    idx_75 = 3 * len(all_floods) // 4
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(all_floods[idx_75], cmap='Blues', vmin=0, vmax=vmax_global)
    ax4.set_title(f'75% ({idx_75})\n{times_hours[idx_75]:.1f} hrs', fontweight='bold', fontsize=10)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Depth (m)')
    
    # Final state
    ax5 = fig.add_subplot(gs[0, 4])
    im5 = ax5.imshow(all_floods[-1], cmap='Blues', vmin=0, vmax=vmax_global)
    ax5.set_title(f'Final ({len(all_floods)-1})\n{times_hours[-1]:.1f} hrs', fontweight='bold', fontsize=10)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label='Depth (m)')
    
    # ========================================================================
    # Row 2: Analysis maps
    # ========================================================================
    
    # Maximum flood depth (envelope)
    ax6 = fig.add_subplot(gs[1, 0])
    max_flood = np.max(all_floods, axis=0)
    im6 = ax6.imshow(max_flood, cmap='jet', vmin=0, vmax=np.max(max_flood))
    ax6.set_title('Maximum Flood Depth\n(all timesteps)', fontweight='bold', fontsize=10)
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label='Max Depth (m)')
    
    # Mean flood depth
    ax7 = fig.add_subplot(gs[1, 1])
    mean_flood = np.mean(all_floods, axis=0)
    im7 = ax7.imshow(mean_flood, cmap='YlGnBu', vmin=0, vmax=np.max(mean_flood))
    ax7.set_title('Mean Flood Depth\n(temporal average)', fontweight='bold', fontsize=10)
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04, label='Mean Depth (m)')
    
    # Flood arrival time
    ax8 = fig.add_subplot(gs[1, 2])
    arrival_time = np.full(all_floods[0].shape, -1, dtype=float)
    for t_idx, flood in enumerate(all_floods):
        first_flood = (flood > 0.05) & (arrival_time < 0)
        arrival_time[first_flood] = times_hours[t_idx]
    arrival_time_masked = np.ma.masked_where(arrival_time < 0, arrival_time)
    im8 = ax8.imshow(arrival_time_masked, cmap='viridis')
    ax8.set_title('Flood Arrival Time\n(depth > 5cm)', fontweight='bold', fontsize=10)
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04, label='Time (hours)')
    
    # Inundation frequency
    ax9 = fig.add_subplot(gs[1, 3])
    inundation_freq = np.sum(all_floods > 0.05, axis=0) / len(all_floods) * 100
    im9 = ax9.imshow(inundation_freq, cmap='YlOrRd', vmin=0, vmax=100)
    ax9.set_title('Inundation Frequency\n(% of time flooded)', fontweight='bold', fontsize=10)
    ax9.axis('off')
    plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04, label='Frequency (%)')
    
    # Depth change (final - initial)
    ax10 = fig.add_subplot(gs[1, 4])
    depth_change = all_floods[-1] - all_floods[0]
    vmax_change = np.max(np.abs(depth_change))
    im10 = ax10.imshow(depth_change, cmap='RdBu_r', vmin=-vmax_change, vmax=vmax_change)
    ax10.set_title('Depth Change\n(final - initial)', fontweight='bold', fontsize=10)
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04, label='Change (m)')
    
    # ========================================================================
    # Row 3: Time series analysis
    # ========================================================================
    
    # Maximum depth evolution
    ax11 = fig.add_subplot(gs[2, 0:2])
    max_depths = np.max(all_floods, axis=(1, 2))
    mean_depths = np.mean(all_floods, axis=(1, 2))
    
    ax11.plot(times_hours, max_depths, 'r-', linewidth=2, label='Max Depth', alpha=0.8)
    ax11.plot(times_hours, mean_depths, 'b-', linewidth=2, label='Mean Depth', alpha=0.8)
    ax11.fill_between(times_hours, 0, max_depths, alpha=0.2, color='red')
    ax11.set_xlabel('Time (hours)', fontweight='bold')
    ax11.set_ylabel('Water Depth (m)', fontweight='bold')
    ax11.set_title('Flood Depth Evolution', fontweight='bold', fontsize=11)
    ax11.legend(loc='best')
    ax11.grid(True, alpha=0.3)
    
    # Flooded area evolution
    ax12 = fig.add_subplot(gs[2, 2:4])
    flooded_area_05cm = np.sum(all_floods > 0.05, axis=(1, 2))
    flooded_area_10cm = np.sum(all_floods > 0.10, axis=(1, 2))
    flooded_area_50cm = np.sum(all_floods > 0.50, axis=(1, 2))
    
    ax12.plot(times_hours, flooded_area_05cm, 'b-', linewidth=2, label='> 5cm', alpha=0.8)
    ax12.plot(times_hours, flooded_area_10cm, 'g-', linewidth=2, label='> 10cm', alpha=0.8)
    ax12.plot(times_hours, flooded_area_50cm, 'r-', linewidth=2, label='> 50cm', alpha=0.8)
    ax12.fill_between(times_hours, 0, flooded_area_05cm, alpha=0.2, color='blue')
    ax12.set_xlabel('Time (hours)', fontweight='bold')
    ax12.set_ylabel('Flooded Area (pixels)', fontweight='bold')
    ax12.set_title('Flooded Area Evolution by Depth Threshold', fontweight='bold', fontsize=11)
    ax12.legend(loc='best')
    ax12.grid(True, alpha=0.3)
    
    # Depth distribution over time
    ax13 = fig.add_subplot(gs[2, 4])
    # Sample depths at different times
    for idx, label, color in [(0, 'Initial', 'lightblue'), 
                               (idx_50, 'Middle', 'blue'), 
                               (-1, 'Final', 'darkblue')]:
        depths = all_floods[idx][all_floods[idx] > 0.01].flatten()
        if len(depths) > 0:
            ax13.hist(depths, bins=30, alpha=0.5, label=label, color=color, edgecolor='black')
    ax13.set_xlabel('Depth (m)', fontweight='bold')
    ax13.set_ylabel('Frequency', fontweight='bold')
    ax13.set_title('Depth Distribution\n(wet areas)', fontweight='bold', fontsize=10)
    ax13.legend()
    ax13.grid(True, alpha=0.3)
    
    # ========================================================================
    # Row 4: Statistics and overlay
    # ========================================================================
    
    # Overlay on DEM (if available)
    ax14 = fig.add_subplot(gs[3, 0:2])
    if dem is not None:
        ax14.imshow(dem, cmap='terrain', alpha=0.6)
        flood_overlay = np.ma.masked_where(all_floods[-1] <= 0.01, all_floods[-1])
        im14 = ax14.imshow(flood_overlay, cmap='Blues', alpha=0.7, vmin=0, vmax=vmax_global)
        ax14.set_title('Final Flood Overlaid on DEM', fontweight='bold', fontsize=11)
        plt.colorbar(im14, ax=ax14, fraction=0.046, pad=0.04, label='Depth (m)')
    else:
        ax14.text(0.5, 0.5, 'DEM not available', ha='center', va='center',
                 transform=ax14.transAxes, fontsize=14)
        ax14.set_title('DEM Overlay (N/A)', fontweight='bold', fontsize=11)
    ax14.axis('off')
    
    # Cross-section profile
    ax15 = fig.add_subplot(gs[3, 2:4])
    center_y = all_floods.shape[1] // 2
    
    # Plot profiles at different times
    x_pixels = np.arange(all_floods.shape[2])
    for idx, label, color, style in [(0, 'Initial', 'lightblue', '-'), 
                                      (idx_50, 'Middle', 'blue', '--'), 
                                      (-1, 'Final', 'darkblue', '-')]:
        profile = all_floods[idx, center_y, :]
        ax15.plot(x_pixels, profile, style, linewidth=2, label=label, color=color, alpha=0.8)
    
    if dem is not None:
        profile_dem = dem[center_y, :]
        ax_dem = ax15.twinx()
        ax_dem.plot(x_pixels, profile_dem, 'brown', linestyle=':', linewidth=1.5, 
                   alpha=0.5, label='Terrain')
        ax_dem.set_ylabel('Elevation (m)', fontweight='bold', color='brown')
        ax_dem.tick_params(axis='y', labelcolor='brown')
    
    ax15.set_xlabel('X Position (pixels)', fontweight='bold')
    ax15.set_ylabel('Water Depth (m)', fontweight='bold', color='blue')
    ax15.tick_params(axis='y', labelcolor='blue')
    ax15.set_title(f'Cross-Section Profile (row {center_y})', fontweight='bold', fontsize=11)
    ax15.legend(loc='upper left')
    ax15.grid(True, alpha=0.3)
    
    # Statistics summary
    ax16 = fig.add_subplot(gs[3, 4])
    ax16.axis('off')
    
    stats_text = "═══ FLOOD STATISTICS ═══\n\n"
    stats_text += f"Grid: {all_floods[0].shape[0]}×{all_floods[0].shape[1]}\n"
    stats_text += f"Timesteps: {len(all_floods)}\n"
    stats_text += f"Duration: {times_hours[-1]:.1f} hrs\n"
    stats_text += f"         ({times_hours[-1]/24:.2f} days)\n\n"
    
    stats_text += "Initial State:\n"
    stats_text += f"  Max: {np.max(all_floods[0]):>7.3f} m\n"
    stats_text += f"  Flooded: {np.sum(all_floods[0]>0.05):>6,} px\n\n"
    
    stats_text += "Final State:\n"
    stats_text += f"  Max: {np.max(all_floods[-1]):>7.3f} m\n"
    stats_text += f"  Mean: {np.mean(all_floods[-1][all_floods[-1]>0.05]):>7.3f} m\n"
    stats_text += f"  Flooded: {np.sum(all_floods[-1]>0.05):>6,} px\n\n"
    
    stats_text += "Maximum Envelope:\n"
    stats_text += f"  Peak: {np.max(max_flood):>7.3f} m\n"
    stats_text += f"  Ever flooded:\n"
    stats_text += f"    {np.sum(max_flood>0.05):>6,} px\n"
    stats_text += f"    ({100*np.sum(max_flood>0.05)/max_flood.size:>5.1f}%)\n\n"
    
    peak_time_idx = np.argmax(max_depths)
    stats_text += "Peak Timing:\n"
    stats_text += f"  Time: {times_hours[peak_time_idx]:>7.1f} hrs\n"
    stats_text += f"  Depth: {max_depths[peak_time_idx]:>7.3f} m\n"
    
    ax16.text(0.05, 0.95, stats_text, transform=ax16.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Main title
    fig.suptitle('Flood Simulation Visualization - 60m Resolution Australia', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_path = Path("outputs") / "flood_60m_visualization.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print(stats_text)
    print("="*60)
    
    return fig, all_floods


def create_flood_animation(output_dir, max_timesteps=None, output_file='flood_animation.mp4', fps=10):
    """
    Create animated visualization of flood evolution as MP4 video.
    
    Args:
        output_dir: Path to directory containing flood outputs
        max_timesteps: Maximum number of timesteps (None = all)
        output_file: Output filename (.mp4)
        fps: Frames per second
    
    Note:
        Requires FFmpeg to be installed. Install with:
        conda install ffmpeg -c conda-forge
    """
    print("\n" + "="*60)
    print("CREATING FLOOD ANIMATION (MP4)")
    print("="*60)
    
    # Load flood files
    flood_files, timestamps = load_flood_timeseries(output_dir, max_files=max_timesteps)
    
    print(f"\nFound {len(flood_files)} timesteps")
    print(f"Creating animation with {len(flood_files)} frames at {fps} fps")
    print(f"Expected duration: {len(flood_files)/fps:.1f} seconds ({len(flood_files)/fps/60:.1f} minutes)")
    
    # Load first frame
    first_flood = tifffile.imread(flood_files[0])
    
    # Calculate global max for consistent colorbar
    print("\nScanning for global maximum...")
    all_max_values = []
    for ff in tqdm(flood_files, desc="Scanning"):
        flood = tifffile.imread(ff)
        all_max_values.append(np.max(flood))
    
    global_max = max(all_max_values)
    vmax = min(global_max, np.percentile(all_max_values, 99))
    
    print(f"✓ Global max: {global_max:.3f} m")
    print(f"✓ Using vmax: {vmax:.3f} m")
    
    # Convert timestamps
    times_hours = np.array(timestamps) / 3600
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[3, 1])
    
    # Main flood map
    ax_map = fig.add_subplot(gs[0, :])
    ax_time = fig.add_subplot(gs[1, 0])
    ax_stats = fig.add_subplot(gs[1, 1])
    
    # Initialize map
    im = ax_map.imshow(first_flood, cmap='Blues', vmin=0, vmax=vmax, animated=True)
    ax_map.set_xlabel('X (pixels)', fontsize=12, fontweight='bold')
    ax_map.set_ylabel('Y (pixels)', fontsize=12, fontweight='bold')
    ax_map.axis('off')
    cbar = plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label('Flood Depth (m)', rotation=270, labelpad=25, fontsize=12, fontweight='bold')
    
    # Time series data
    spatial_max = np.array(all_max_values)
    spatial_mean = np.zeros(len(flood_files))
    
    print("\nCalculating statistics...")
    for i, ff in enumerate(tqdm(flood_files, desc="Processing")):
        flood = tifffile.imread(ff)
        wet_mask = flood > 0.01
        if np.any(wet_mask):
            spatial_mean[i] = np.mean(flood[wet_mask])
        else:
            spatial_mean[i] = 0
    
    # Initialize time series
    line_max, = ax_time.plot([], [], 'r-', linewidth=2, label='Max Depth')
    line_mean, = ax_time.plot([], [], 'b-', linewidth=2, label='Mean Depth (wet)')
    current_marker = ax_time.axvline(0, color='green', linestyle='--', linewidth=2, label='Current')
    
    ax_time.set_xlim(0, times_hours[-1])
    ax_time.set_ylim(0, vmax)
    ax_time.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
    ax_time.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
    ax_time.set_title('Depth Evolution', fontsize=12, fontweight='bold')
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc='upper right')
    
    # Stats text
    stats_text = ax_stats.text(0.1, 0.9, '', transform=ax_stats.transAxes,
                               fontsize=11, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax_stats.axis('off')
    ax_stats.set_title('Current Statistics', fontsize=12, fontweight='bold')
    
    # Title
    title = fig.suptitle('', fontsize=16, fontweight='bold', y=0.98)
    
    def init():
        im.set_data(first_flood)
        line_max.set_data([], [])
        line_mean.set_data([], [])
        return [im, line_max, line_mean, current_marker, title, stats_text]
    
    def update(frame):
        # Load flood
        flood = tifffile.imread(flood_files[frame])
        
        # Update map
        im.set_data(flood)
        
        # Update time series
        line_max.set_data(times_hours[:frame+1], spatial_max[:frame+1])
        line_mean.set_data(times_hours[:frame+1], spatial_mean[:frame+1])
        current_marker.set_xdata([times_hours[frame], times_hours[frame]])
        
        # Update title
        current_time = times_hours[frame]
        title.set_text(f'Flood Animation - 60m Australia - t = {current_time:.1f} hrs ({current_time/24:.2f} days) - Frame {frame+1}/{len(flood_files)}')
        
        # Update stats
        wet_pixels = np.sum(flood > 0.05)
        total_pixels = flood.size
        wet_pct = 100 * wet_pixels / total_pixels
        
        stats_str = f"═══ TIMESTEP {frame} ═══\n\n"
        stats_str += f"Time: {current_time:.1f} hours\n"
        stats_str += f"      ({current_time/24:.2f} days)\n\n"
        stats_str += f"Flood Depth:\n"
        stats_str += f"  Max:  {np.max(flood):>7.3f} m\n"
        stats_str += f"  Mean: {np.mean(flood):>7.3f} m\n"
        if wet_pixels > 0:
            stats_str += f"  Mean (wet): {np.mean(flood[flood>0.01]):>7.3f} m\n"
        stats_str += f"\nInundation:\n"
        stats_str += f"  Wet area: {wet_pct:>5.1f}%\n"
        stats_str += f"  Wet pixels: {wet_pixels:>7,}\n"
        stats_str += f"\nDepth Categories:\n"
        stats_str += f"  > 10cm: {np.sum(flood>0.1):>7,} px\n"
        stats_str += f"  > 50cm: {np.sum(flood>0.5):>7,} px\n"
        stats_str += f"  > 1m:   {np.sum(flood>1.0):>7,} px\n"
        
        stats_text.set_text(stats_str)
        
        return [im, line_max, line_mean, current_marker, title, stats_text]
    
    # Create animation
    print(f"\nGenerating animation...")
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(flood_files),
        interval=1000/fps,
        blit=True,
        repeat=True
    )
    
    # Save
    output_path = Path("outputs") / output_file
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SAVING ANIMATION")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Format: MP4 (H.264)")
    print(f"\nThis will take several minutes...")
    print("Please wait...\n")
    
    try:
        # Use FFmpeg writer for MP4
        writer = FFMpegWriter(fps=fps, bitrate=2000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
        
        # Save with progress callback
        anim.save(
            str(output_path), 
            writer=writer, 
            dpi=100,
            progress_callback=lambda i, n: print(f"  Rendering frame {i+1}/{n} ({100*(i+1)/n:.1f}%)", end='\r')
        )
        
        print(f"\n\n{'='*60}")
        print("✓ ANIMATION SAVED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"File: {output_path}")
        print(f"Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"Duration: {len(flood_files) / fps:.1f} seconds ({len(flood_files) / fps / 60:.1f} minutes)")
        print(f"Frames: {len(flood_files)}")
        print(f"FPS: {fps}")
        print(f"{'='*60}")
        
    except FileNotFoundError as e:
        print(f"\n{'='*60}")
        print("✗ ERROR: FFmpeg not found!")
        print(f"{'='*60}")
        print("\nFFmpeg is required to create MP4 animations.")
        print("\nTo install FFmpeg:")
        print("  conda install ffmpeg -c conda-forge")
        print("\nOr on Windows, download from: https://ffmpeg.org/download.html")
        print("\nFull error details:")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}")
        raise
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ ERROR: Animation save failed!")
        print(f"{'='*60}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull error details:")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}")
        raise
    
    return anim, fig


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("FLOOD VISUALIZATION - 60m Resolution Australia")
    print("="*60)
    
    # Check if directory exists
    if not FLOOD_OUTPUT_DIR.exists():
        print(f"\n✗ ERROR: Output directory not found at:")
        print(f"  {FLOOD_OUTPUT_DIR}")
        print("\nPlease check the path and update DATA_ROOT if needed.")
        exit(1)
    
    print(f"\n✓ Found output directory: {FLOOD_OUTPUT_DIR.name}")
    
    # Menu
    print("\nChoose visualization mode:")
    print("  1. Static visualization (default, first 100 timesteps)")
    print("  2. Static visualization (all timesteps)")
    print("  3. Create animation (all timesteps)")
    print("  4. Create animation (first 100 timesteps - faster)")
    print("  5. Both static + animation (all timesteps)")
    
    choice = input("\nEnter choice [1-5] (or press Enter for 1): ").strip()
    
    if choice == "" or choice == "1":
        # Static with 100 timesteps
        fig, flood_data = visualize_flood_static(FLOOD_OUTPUT_DIR, DEM_PATH, max_timesteps=100)
        print("\n✓ Visualization complete!")
        plt.show()
        
    elif choice == "2":
        # Static with all timesteps
        fig, flood_data = visualize_flood_static(FLOOD_OUTPUT_DIR, DEM_PATH, max_timesteps=None)
        print("\n✓ Visualization complete!")
        plt.show()
        
    elif choice == "3":
        # Animation with all timesteps (MP4)
        anim, fig = create_flood_animation(
            FLOOD_OUTPUT_DIR,
            max_timesteps=None,
            output_file='flood_60m_animation_full.mp4',
            fps=10
        )
        print("\n✓ Animation complete!")
        
    elif choice == "4":
        # Animation with 100 timesteps (MP4)
        anim, fig = create_flood_animation(
            FLOOD_OUTPUT_DIR,
            max_timesteps=100,
            output_file='flood_60m_animation_100frames.mp4',
            fps=10
        )
        print("\n✓ Animation complete!")
        
    elif choice == "5":
        # Both static and animation
        fig_static, flood_data = visualize_flood_static(FLOOD_OUTPUT_DIR, DEM_PATH, max_timesteps=None)
        plt.close(fig_static)
        
        anim, fig_anim = create_flood_animation(
            FLOOD_OUTPUT_DIR,
            max_timesteps=None,
            output_file='flood_60m_animation_full.mp4',
            fps=10
        )
        
        print("\n✓ Both visualizations complete!")
        
    else:
        print("\n✗ Invalid choice. Creating static visualization...")
        fig, flood_data = visualize_flood_static(FLOOD_OUTPUT_DIR, DEM_PATH, max_timesteps=100)
        plt.show()
