"""
Flood Visualization on Google Maps
====================================
Overlay flood simulation data on DEM and render on Google Maps.
Converts raster coordinates to WGS84 and creates interactive HTML map.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tifffile
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

# Geospatial libraries
try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    print("⚠ Warning: rasterio not installed. Install with: pip install rasterio")
    RASTERIO_AVAILABLE = False

try:
    from pyproj import Transformer, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    print("⚠ Warning: pyproj not installed. Install with: pip install pyproj")
    PYPROJ_AVAILABLE = False

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    print("⚠ Warning: folium not installed. Install with: pip install folium")
    FOLIUM_AVAILABLE = False

# ============================================================================
# Configuration
# ============================================================================

DATA_ROOT = Path("..") / "FloodCastBench_Dataset-and-Models-main" / "Data_Generation_Code" / "FloodCastBench"
FLOOD_OUTPUT_DIR = DATA_ROOT / "High-fidelity_flood_forecasting" / "60m" / "Australia"
DEM_PATH = DATA_ROOT / "Relevant_data" / "DEM" / "Australia_DEM.tif"

DEFAULT_EPSG = 32756  # UTM Zone 56S (Brisbane/Australia)

# ============================================================================
# Helper Functions
# ============================================================================

def load_dem_with_georeference(dem_path):
    """Load DEM and extract georeferencing information"""
    print("\n" + "="*60)
    print("LOADING DEM")
    print("="*60)
    
    dem = tifffile.imread(dem_path)
    print(f"✓ DEM loaded: {dem.shape} pixels")
    
    if not RASTERIO_AVAILABLE:
        print("✗ Rasterio not available - cannot extract georeference")
        return dem, None, None
    
    # Read georeference info
    with rasterio.open(dem_path) as src:
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        
        print(f"✓ CRS: {crs}")
        print(f"✓ Bounds: {bounds}")
        
        return dem, transform, crs


def get_wgs84_bounds(transform, crs, shape):
    """Convert raster bounds to WGS84 (lat/lon)"""
    if not PYPROJ_AVAILABLE:
        print("✗ pyproj not available - cannot convert coordinates")
        return None
    
    height, width = shape
    
    # Get corners in original CRS
    x_min, y_max = transform * (0, 0)
    x_max, y_min = transform * (width, height)
    
    print(f"\nOriginal bounds (UTM):")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")
    
    # Transform to WGS84
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    
    lon_min, lat_max = transformer.transform(x_min, y_max)
    lon_max, lat_min = transformer.transform(x_max, y_min)
    
    bounds_wgs84 = {
        'north': lat_max,
        'south': lat_min,
        'east': lon_max,
        'west': lon_min,
        'center_lat': (lat_max + lat_min) / 2,
        'center_lon': (lon_max + lon_min) / 2
    }
    
    print(f"\nWGS84 bounds (Lat/Lon):")
    print(f"  Lat: [{lat_min:.6f}°, {lat_max:.6f}°]")
    print(f"  Lon: [{lon_min:.6f}°, {lon_max:.6f}°]")
    
    return bounds_wgs84


def load_flood_timestep(flood_dir, timestep_idx=None):
    """Load a specific flood timestep"""
    flood_files = sorted(list(flood_dir.glob("*.tif")))
    
    if not flood_files:
        raise FileNotFoundError(f"No flood files found in {flood_dir}")
    
    print(f"\nFound {len(flood_files)} flood timesteps")
    
    # Use last timestep if not specified
    if timestep_idx is None:
        timestep_idx = -1
    
    if timestep_idx < 0:
        timestep_idx = len(flood_files) + timestep_idx
    
    flood_file = flood_files[timestep_idx]
    print(f"Loading timestep {timestep_idx}: {flood_file.name}")
    
    flood = tifffile.imread(flood_file)
    print(f"✓ Flood data loaded: {flood.shape} pixels")
    
    return flood, flood_file.name


def create_flood_overlay_image(dem, flood, alpha=0.7):
    """Create RGBA image with DEM as background and flood as overlay"""
    print("\n" + "="*60)
    print("CREATING OVERLAY IMAGE")
    print("="*60)
    
    # Normalize DEM for visualization (terrain colormap)
    dem_normalized = (dem - np.min(dem)) / (np.max(dem) - np.min(dem))
    terrain_cmap = plt.cm.terrain
    dem_rgba = terrain_cmap(dem_normalized)
    
    # Create flood overlay (blue colormap)
    flood_max = np.percentile(flood[flood > 0], 99) if np.any(flood > 0) else 1.0
    flood_normalized = np.clip(flood / flood_max, 0, 1)
    
    # Blue colormap for water
    blues_cmap = plt.cm.Blues
    flood_rgba = blues_cmap(flood_normalized)
    
    # Set alpha based on flood depth
    flood_rgba[:, :, 3] = np.where(flood > 0.05, alpha, 0)
    
    # Composite: blend flood over DEM
    composite = dem_rgba.copy()
    flood_mask = flood > 0.05
    
    # Alpha blending
    for i in range(3):  # RGB channels
        composite[:, :, i] = np.where(
            flood_mask,
            flood_rgba[:, :, i] * flood_rgba[:, :, 3] + dem_rgba[:, :, i] * (1 - flood_rgba[:, :, 3]),
            dem_rgba[:, :, i]
        )
    
    print(f"✓ Overlay created")
    print(f"  DEM range: {np.min(dem):.1f} - {np.max(dem):.1f} m")
    print(f"  Flood max: {np.max(flood):.3f} m")
    print(f"  Flooded pixels: {np.sum(flood > 0.05):,} ({100*np.sum(flood>0.05)/flood.size:.2f}%)")
    
    return composite


def array_to_base64_png(array):
    """Convert numpy array to base64-encoded PNG"""
    # Convert to uint8 (0-255 range)
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = (array * 255).astype(np.uint8)
    
    # Create PIL Image
    img = Image.fromarray(array, mode='RGBA')
    
    # Save to bytes buffer
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode as base64
    img_base64 = base64.b64encode(buffer.read()).decode()
    
    return f"data:image/png;base64,{img_base64}"


def create_google_map(dem, flood, bounds_wgs84, flood_filename, output_html='flood_google_map.html'):
    """Create interactive Google Map with flood overlay"""
    
    if not FOLIUM_AVAILABLE:
        print("✗ Folium not available. Install with: pip install folium")
        return None
    
    print("\n" + "="*60)
    print("CREATING GOOGLE MAP")
    print("="*60)
    
    # Create overlay image
    overlay_img = create_flood_overlay_image(dem, flood, alpha=0.7)
    
    # Create base map
    m = folium.Map(
        location=[bounds_wgs84['center_lat'], bounds_wgs84['center_lon']],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add tile layers
    folium.TileLayer('CartoDB positron', name='Light Map', attr='CartoDB').add_to(m)
    
    # Google Satellite
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add DEM boundary
    folium.Rectangle(
        bounds=[[bounds_wgs84['south'], bounds_wgs84['west']],
                [bounds_wgs84['north'], bounds_wgs84['east']]],
        color='red',
        weight=2,
        fill=False,
        popup="DEM Boundary",
        tooltip="Click for details"
    ).add_to(m)
    
    # Convert overlay image to base64
    print("Converting overlay to PNG...")
    img_data = array_to_base64_png(overlay_img)
    
    # Add image overlay
    print("Adding flood overlay to map...")
    img_overlay = folium.raster_layers.ImageOverlay(
        image=img_data,
        bounds=[[bounds_wgs84['south'], bounds_wgs84['west']],
                [bounds_wgs84['north'], bounds_wgs84['east']]],
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        name='Flood Overlay'
    )
    img_overlay.add_to(m)
    
    # Add statistics popup
    stats_html = f"""
    <div style="font-family: monospace; width: 300px;">
        <h4>Flood Simulation Statistics</h4>
        <b>File:</b> {flood_filename}<br>
        <b>Grid Size:</b> {flood.shape[0]} × {flood.shape[1]} pixels<br>
        <br>
        <b>DEM:</b><br>
        &nbsp;&nbsp;Min Elevation: {np.min(dem):.1f} m<br>
        &nbsp;&nbsp;Max Elevation: {np.max(dem):.1f} m<br>
        <br>
        <b>Flood Depth:</b><br>
        &nbsp;&nbsp;Max Depth: {np.max(flood):.3f} m<br>
        &nbsp;&nbsp;Mean Depth: {np.mean(flood[flood>0.05]):.3f} m (wet areas)<br>
        <br>
        <b>Inundation:</b><br>
        &nbsp;&nbsp;Flooded Pixels: {np.sum(flood>0.05):,}<br>
        &nbsp;&nbsp;Flooded Area: {100*np.sum(flood>0.05)/flood.size:.2f}%<br>
        <br>
        <b>Depth Categories:</b><br>
        &nbsp;&nbsp;> 10 cm: {np.sum(flood>0.1):,} pixels<br>
        &nbsp;&nbsp;> 50 cm: {np.sum(flood>0.5):,} pixels<br>
        &nbsp;&nbsp;> 1 m: {np.sum(flood>1.0):,} pixels<br>
    </div>
    """
    
    folium.Marker(
        [bounds_wgs84['center_lat'], bounds_wgs84['center_lon']],
        popup=folium.Popup(stats_html, max_width=350),
        tooltip="Click for flood statistics",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright').add_to(m)
    
    # Add fullscreen
    plugins.Fullscreen(position='topleft').add_to(m)
    
    # Add mouse position
    plugins.MousePosition(
        position='bottomleft',
        separator=' | ',
        prefix='Coordinates:'
    ).add_to(m)
    
    # Add measurement tool
    plugins.MeasureControl(
        position='topleft',
        primary_length_unit='meters',
        primary_area_unit='sqmeters'
    ).add_to(m)
    
    # Add minimap
    minimap = plugins.MiniMap(toggle_display=True)
    m.add_child(minimap)
    
    # Save
    output_path = Path("outputs") / output_html
    output_path.parent.mkdir(exist_ok=True)
    m.save(str(output_path))
    
    print(f"\n✓ Google Map saved to: {output_path}")
    print(f"  Open in web browser to view interactive flood overlay!")
    
    return m


def create_static_visualization(dem, flood, flood_filename, output_png='flood_overlay_static.png'):
    """Create static PNG visualization"""
    print("\n" + "="*60)
    print("CREATING STATIC VISUALIZATION")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. DEM
    im1 = axes[0, 0].imshow(dem, cmap='terrain')
    axes[0, 0].set_title('Digital Elevation Model (DEM)', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04, label='Elevation (m)')
    
    # 2. Flood depth
    flood_max = np.percentile(flood[flood > 0], 99) if np.any(flood > 0) else 1.0
    im2 = axes[0, 1].imshow(flood, cmap='Blues', vmin=0, vmax=flood_max)
    axes[0, 1].set_title(f'Flood Depth\n{flood_filename}', fontweight='bold', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04, label='Depth (m)')
    
    # 3. Overlay
    overlay = create_flood_overlay_image(dem, flood, alpha=0.7)
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Flood Overlay on DEM', fontweight='bold', fontsize=12)
    axes[0, 2].axis('off')
    
    # 4. Flood mask
    flood_mask = flood > 0.05
    im4 = axes[1, 0].imshow(flood_mask, cmap='RdYlBu_r')
    axes[1, 0].set_title('Inundation Map (depth > 5cm)', fontweight='bold', fontsize=12)
    axes[1, 0].axis('off')
    wet_pct = 100 * np.sum(flood_mask) / flood_mask.size
    axes[1, 0].text(0.02, 0.98, f'Flooded: {wet_pct:.2f}%',
                    transform=axes[1, 0].transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Depth categories
    depth_categories = np.zeros_like(flood)
    depth_categories[flood <= 0.05] = 0  # Dry
    depth_categories[(flood > 0.05) & (flood <= 0.1)] = 1  # Very shallow
    depth_categories[(flood > 0.1) & (flood <= 0.5)] = 2  # Shallow
    depth_categories[(flood > 0.5) & (flood <= 1.0)] = 3  # Moderate
    depth_categories[flood > 1.0] = 4  # Deep
    
    from matplotlib.colors import ListedColormap
    colors = ['tan', 'lightblue', 'blue', 'darkblue', 'navy']
    cmap_depth = ListedColormap(colors)
    
    im5 = axes[1, 1].imshow(depth_categories, cmap=cmap_depth, vmin=0, vmax=4)
    axes[1, 1].set_title('Flood Depth Categories', fontweight='bold', fontsize=12)
    axes[1, 1].axis('off')
    
    import matplotlib.patches as mpatches
    legend_labels = ['Dry', '5-10cm', '10-50cm', '50cm-1m', '>1m']
    patches = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(5)]
    axes[1, 1].legend(handles=patches, loc='upper right', fontsize=10, framealpha=0.9)
    
    # 6. Statistics
    axes[1, 2].axis('off')
    stats_text = "═══ FLOOD STATISTICS ═══\n\n"
    stats_text += f"File: {flood_filename}\n\n"
    stats_text += f"Grid: {flood.shape[0]} × {flood.shape[1]} pixels\n\n"
    stats_text += "DEM:\n"
    stats_text += f"  Min: {np.min(dem):>8.1f} m\n"
    stats_text += f"  Max: {np.max(dem):>8.1f} m\n"
    stats_text += f"  Mean: {np.mean(dem):>8.1f} m\n\n"
    stats_text += "Flood Depth:\n"
    stats_text += f"  Max: {np.max(flood):>8.3f} m\n"
    if np.any(flood > 0.05):
        stats_text += f"  Mean (wet): {np.mean(flood[flood>0.05]):>8.3f} m\n"
    stats_text += f"\nInundation:\n"
    stats_text += f"  Flooded: {np.sum(flood>0.05):>8,} px\n"
    stats_text += f"  Percent: {100*np.sum(flood>0.05)/flood.size:>8.2f}%\n\n"
    stats_text += "Depth Categories:\n"
    for i, label in enumerate(legend_labels):
        count = np.sum(depth_categories == i)
        pct = 100 * count / depth_categories.size
        stats_text += f"  {label:10s}: {pct:>6.2f}%\n"
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle('Flood Overlay Visualization', fontsize=16, fontweight='bold', y=0.98)
    
    output_path = Path("outputs") / output_png
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Static visualization saved to: {output_path}")
    
    return fig


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution"""
    print("="*60)
    print("FLOOD OVERLAY ON GOOGLE MAPS")
    print("="*60)
    
    # Check dependencies
    if not RASTERIO_AVAILABLE:
        print("\n✗ ERROR: rasterio is required")
        print("  Install with: pip install rasterio")
        return
    
    if not PYPROJ_AVAILABLE:
        print("\n✗ ERROR: pyproj is required")
        print("  Install with: pip install pyproj")
        return
    
    if not FOLIUM_AVAILABLE:
        print("\n⚠ WARNING: folium not available (Google Maps won't be created)")
        print("  Install with: pip install folium")
    
    # Check paths
    if not DEM_PATH.exists():
        print(f"\n✗ ERROR: DEM not found at {DEM_PATH}")
        return
    
    if not FLOOD_OUTPUT_DIR.exists():
        print(f"\n✗ ERROR: Flood output directory not found at {FLOOD_OUTPUT_DIR}")
        return
    
    # Load DEM
    dem, transform, crs = load_dem_with_georeference(DEM_PATH)
    
    if transform is None or crs is None:
        print("\n✗ ERROR: Could not extract georeferencing from DEM")
        return
    
    # Get WGS84 bounds
    bounds_wgs84 = get_wgs84_bounds(transform, crs, dem.shape)
    
    if bounds_wgs84 is None:
        print("\n✗ ERROR: Could not convert to WGS84 coordinates")
        return
    
    # Load flood data
    print("\n" + "="*60)
    print("LOADING FLOOD DATA")
    print("="*60)
    
    # Ask user which timestep
    print("\nWhich flood timestep would you like to visualize?")
    print("  -1: Last timestep (final state)")
    print("  0: First timestep (initial state)")
    print("  N: Specific timestep number")
    
    choice = input("\nEnter timestep index (or press Enter for last): ").strip()
    
    if choice == "":
        timestep_idx = -1
    else:
        try:
            timestep_idx = int(choice)
        except ValueError:
            print("Invalid input, using last timestep")
            timestep_idx = -1
    
    flood, flood_filename = load_flood_timestep(FLOOD_OUTPUT_DIR, timestep_idx)
    
    # Check dimensions match
    if dem.shape != flood.shape:
        print(f"\n⚠ WARNING: DEM shape {dem.shape} != Flood shape {flood.shape}")
        print("  Resizing flood data to match DEM...")
        from scipy.ndimage import zoom
        zoom_factors = (dem.shape[0] / flood.shape[0], dem.shape[1] / flood.shape[1])
        flood = zoom(flood, zoom_factors, order=1)
        print(f"✓ Flood resized to {flood.shape}")
    
    # Create visualizations
    print("\n" + "="*60)
    print("GENERATING OUTPUTS")
    print("="*60)
    
    # Static PNG
    fig = create_static_visualization(dem, flood, flood_filename)
    plt.close(fig)
    
    # Google Map
    if FOLIUM_AVAILABLE:
        m = create_google_map(dem, flood, bounds_wgs84, flood_filename)
    
    # Summary
    print("\n" + "="*60)
    print("✓ COMPLETE!")
    print("="*60)
    print("\nGenerated files in 'outputs/' folder:")
    print("  1. flood_overlay_static.png  → Static visualization")
    if FOLIUM_AVAILABLE:
        print("  2. flood_google_map.html     → Interactive Google Maps")
        print("\n🌍 Open flood_google_map.html in your web browser!")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
