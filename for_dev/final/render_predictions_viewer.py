"""
Render All Prediction Files on OpenStreetMap
=============================================
Visualizes all flood depth predictions from db_storage with custom color gradient.

Color Scheme:
- 0-0.1m: Blue
- 0.1-0.2m: Yellow
- 0.2-0.5m: Orange
- 0.5m+: Red
"""

import numpy as np
import tifffile
import folium
from pathlib import Path
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import io
import base64
from scipy.ndimage import zoom

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_DIR = Path("d:/project/d4l/for_dev/final")
DEM_PATH = Path("d:/project/d4l/ai_service_hue/static_data/Hue_DEM.tif")
DEM_TFW = Path("d:/project/d4l/ai_service_hue/static_data/Hue_DEM.tfw")
DEM_SMALL_TFW = Path("d:/project/d4l/ai_service_hue/static_data/Hue_DEM_small.tfw")
PREDICTIONS_DIR = BASE_DIR  # Prediction files are in the same directory
OUTPUT_DIR = BASE_DIR / "output_maps"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Settings
MAX_IMG_SIZE = 1000  # Downsample for browser performance
FLOOD_ALPHA = 0.8    # Transparency

# ============================================================================
# CUSTOM COLORMAP
# ============================================================================

def create_flood_colormap():
    """
    Create custom colormap for flood depth visualization
    
    Depth ranges:
    - 0-0.1m: Blue
    - 0.1-0.2m: Yellow
    - 0.2-0.5m: Orange
    - 0.5m+: Red
    """
    colors = [
        (0.0, 'blue'),      # 0m
        (0.1, 'blue'),      # 0.1m
        (0.1, 'yellow'),    # 0.1m (transition)
        (0.2, 'yellow'),    # 0.2m
        (0.2, 'orange'),    # 0.2m (transition)
        (0.5, 'orange'),    # 0.5m
        (0.5, 'red'),       # 0.5m (transition)
        (1.0, 'red')        # 1.0m+
    ]
    
    # Normalize positions to [0, 1] based on max value of 1.0m
    positions = [c[0] for c in colors]
    color_names = [c[1] for c in colors]
    
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'flood_depth', 
        list(zip(positions, color_names))
    )
    
    return cmap

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_tfw(tfw_path):
    """Read World File for georeferencing"""
    if not tfw_path.exists():
        return None
    with open(tfw_path, 'r') as f:
        params = [float(line.strip()) for line in f.readlines()]
    return {
        'pixel_width': params[0],
        'rotation_y': params[1],
        'rotation_x': params[2],
        'pixel_height': params[3],
        'x_origin': params[4],
        'y_origin': params[5]
    }

def get_wgs84_bounds(tfw_info, shape, utm_zone=48, hemisphere='north'):
    """Calculate Lat/Lon bounds from TFW and Shape"""
    height, width = shape
    
    # Upper-left corner (from TFW)
    x_origin = tfw_info['x_origin']
    y_origin = tfw_info['y_origin']
    
    # Lower-right corner (calculated)
    x_max = x_origin + (width * tfw_info['pixel_width'])
    y_min = y_origin + (height * tfw_info['pixel_height'])  # pixel_height is negative
    
    utm_epsg = f"326{utm_zone}" if hemisphere == 'north' else f"327{utm_zone}"
    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    
    # Transform corners: bottom-left and top-right
    west, south = transformer.transform(x_origin, y_min)
    east, north = transformer.transform(x_max, y_origin)
    return [[south, west], [north, east]]

def resize_array(arr, target_shape):
    """Resize array to match target shape exactly"""
    if arr.shape == target_shape:
        return arr
    factors = (target_shape[0]/arr.shape[0], target_shape[1]/arr.shape[1])
    return zoom(arr, factors, order=1)  # Bilinear for smoother visualization

def generate_flood_image(flood_data, cmap, alpha):
    """
    Generate RGBA image from flood depth data
    
    Args:
        flood_data: 2D numpy array of flood depths (meters)
        cmap: Matplotlib colormap
        alpha: Transparency value
    
    Returns:
        Base64 encoded PNG string
    """
    # Mask out non-flooded areas
    mask = flood_data > 0.001
    
    if not np.any(mask):
        return None
    
    # Normalize to [0, 1] range (0 to 1.0m, anything above is clamped)
    data = np.copy(flood_data)
    data[~mask] = np.nan
    norm_data = np.clip(data / 1.0, 0, 1)
    
    # Apply colormap
    rgba = cmap(norm_data)
    
    # Set alpha channel (transparent where no flood)
    rgba[..., 3] = np.where(mask, alpha, 0)
    
    # Convert to PNG
    img_uint8 = (rgba * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG', optimize=True)
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

# ============================================================================
# MAIN PROCESS
# ============================================================================

def create_prediction_viewer():
    """Create interactive map with all predictions"""
    
    print("="*70)
    print("FLOOD PREDICTION VIEWER - OpenStreetMap")
    print("="*70)
    
    # 1. Load georeferencing info
    print("\n[1] Loading georeferencing data...")
    
    # First, check if predictions are downsampled by loading a sample
    # Try regular files first, then high-res files
    regular_sample_files = sorted(list(PREDICTIONS_DIR.glob("[0-9]*.tif")))
    high_sample_files = sorted(list(PREDICTIONS_DIR.glob("high_*.tif")))
    sample_files = regular_sample_files + high_sample_files
    
    if not sample_files:
        print("ERROR: No prediction files found!")
        return
        
    sample_pred = tifffile.imread(sample_files[0])
    pred_height = sample_pred.shape[0]
    print(f"Prediction shape: {sample_pred.shape}")
    
    # Determine which TFW to use based on prediction size
    # If predictions are around 670px high, use small TFW; if around 2681px, use full TFW
    if pred_height < 1000:  # Downsampled predictions
        tfw_path = DEM_SMALL_TFW
        print(f"Using SMALL TFW (predictions are downsampled)")
    else:  # Full resolution predictions
        tfw_path = DEM_TFW
        print(f"Using FULL TFW (predictions are full resolution)")
    
    tfw = read_tfw(tfw_path)
    if not tfw:
        print(f"ERROR: Cannot find TFW file: {tfw_path}")
        return
    
    # 2. Get all prediction files
    print("\n[2] Scanning prediction files...")
    # Look for numbered prediction files (e.g., 0000.tif, 0005.tif, etc.)
    regular_files = sorted(list(PREDICTIONS_DIR.glob("[0-9]*.tif")))
    # Also look for high-resolution files (e.g., high_0000.tif, high_0005.tif, etc.)
    high_res_files = sorted(list(PREDICTIONS_DIR.glob("high_*.tif")))
    pred_files = sorted(regular_files + high_res_files)
    print(f"Found {len(pred_files)} prediction files ({len(regular_files)} regular + {len(high_res_files)} high-res)")
    
    if not pred_files:
        print("ERROR: No prediction files found!")
        return
    
    # 3. Load first file to get dimensions and bounds
    print("\n[3] Setting up georeferencing...")
    original_shape = sample_pred.shape  # Use the already-loaded sample
    print(f"Prediction data shape: {original_shape}")
    
    # Calculate bounds from ORIGINAL prediction shape (not display shape)
    # This ensures the image overlays match the actual geographic extent
    bounds = get_wgs84_bounds(tfw, original_shape)
    center = [(bounds[0][0] + bounds[1][0])/2, (bounds[0][1] + bounds[1][1])/2]
    print(f"Map center: {center}")
    print(f"Geographic bounds: {bounds}")
    
    # Downsample if needed (for display only, bounds stay the same)
    if original_shape[0] > MAX_IMG_SIZE:
        print(f"Downsampling to {MAX_IMG_SIZE}px for browser performance...")
        scale = MAX_IMG_SIZE / original_shape[0]
        target_shape = (MAX_IMG_SIZE, int(original_shape[1] * scale))
        print(f"Display shape: {target_shape}")
    else:
        target_shape = original_shape
        print(f"Display shape: {target_shape} (no downsampling needed)")
    
    # 4. Create custom colormap
    print("\n[4] Creating custom colormap...")
    flood_cmap = create_flood_colormap()
    
    # 5. Generate images for all predictions
    print("\n[5] Generating flood visualizations...")
    flood_images = {}
    
    for i, pred_file in enumerate(pred_files):
        filename = pred_file.name
        print(f"  [{i+1}/{len(pred_files)}] Processing {filename}...", end='\r')
        
        # Load and resize
        flood = tifffile.imread(pred_file)
        if flood.shape != target_shape:
            flood = resize_array(flood, target_shape)
        
        # Generate image
        img_b64 = generate_flood_image(flood, flood_cmap, FLOOD_ALPHA)
        
        if img_b64:
            flood_images[filename] = {
                'image': img_b64,
                'max_depth': float(np.max(flood)),
                'flooded_area': int(np.sum(flood > 0.001))
            }
    
    print(f"\n  ✓ Generated {len(flood_images)} visualizations")
    
    # 6. Create interactive map
    print("\n[6] Creating interactive map...")
    m = folium.Map(
        location=center,
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add Google Satellite as option
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Get first prediction for initial display
    first_key = list(flood_images.keys())[0]
    initial_img = flood_images[first_key]['image']
    
    # Add image overlay
    img_layer = folium.raster_layers.ImageOverlay(
        image=initial_img,
        bounds=bounds,
        opacity=1,
        name="Flood Depth",
        interactive=True,
        cross_origin=False,
        zindex=10
    )
    img_layer.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # 7. Create dropdown selector with stats
    print("\n[7] Adding interactive controls...")
    
    import json
    images_json = json.dumps({k: v['image'] for k, v in flood_images.items()})
    stats_json = json.dumps({k: {'max': v['max_depth'], 'area': v['flooded_area']} 
                             for k, v in flood_images.items()})
    
    # Create sorted list of filenames for dropdown
    sorted_files = sorted(flood_images.keys())
    options_html = "\n".join([f'<option value="{f}">{f}</option>' for f in sorted_files])
    
    html_control = f"""
    <div id="control-panel" style="
        position: fixed; bottom: 30px; left: 30px; width: 400px;
        background: white; padding: 15px; border-radius: 8px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2); z-index: 9999; font-family: sans-serif;">
        
        <h4 style="margin: 0 0 10px 0;">🌊 Flood Prediction Viewer</h4>
        
        <label style="font-weight: bold; display: block; margin-bottom: 5px;">
            Select Prediction:
        </label>
        <select id="pred-selector" style="
            width: 100%; padding: 8px; margin-bottom: 10px; 
            border: 1px solid #ccc; border-radius: 4px; font-size: 13px;">
            {options_html}
        </select>
        
        <div id="stats" style="
            background: #f5f5f5; padding: 10px; border-radius: 4px; 
            font-size: 12px; margin-bottom: 10px;">
            <div><strong>Max Depth:</strong> <span id="max-depth">-</span> m</div>
            <div><strong>Flooded Pixels:</strong> <span id="flooded-area">-</span></div>
        </div>
        
        <div style="font-size: 11px; color: #666;">
            <strong>Color Legend:</strong><br>
            <span style="color: blue;">█</span> 0-0.1m (Blue)<br>
            <span style="color: #FFD700;">█</span> 0.1-0.2m (Yellow)<br>
            <span style="color: orange;">█</span> 0.2-0.5m (Orange)<br>
            <span style="color: red;">█</span> 0.5m+ (Red)
        </div>
    </div>

    <script>
        var floodImages = {images_json};
        var floodStats = {stats_json};
        
        var selector = document.getElementById('pred-selector');
        var maxDepthSpan = document.getElementById('max-depth');
        var floodedAreaSpan = document.getElementById('flooded-area');
        
        function updateImage(filename) {{
            var newSrc = floodImages[filename];
            var stats = floodStats[filename];
            
            if (newSrc) {{
                // Update image overlay
                var images = document.getElementsByClassName('leaflet-image-layer');
                if (images.length > 0) {{
                    images[0].src = newSrc;
                }}
                
                // Update stats
                maxDepthSpan.innerText = stats.max.toFixed(3);
                floodedAreaSpan.innerText = stats.area.toLocaleString();
            }}
        }}

        // Event listener
        selector.addEventListener('change', function() {{
            updateImage(this.value);
        }});
        
        // Initialize with first image
        updateImage(selector.value);
    </script>
    """
    
    m.get_root().html.add_child(folium.Element(html_control))
    
    # 8. Save map
    output_file = OUTPUT_DIR / "predictions_viewer.html"
    m.save(str(output_file))
    
    print(f"\n{'='*70}")
    print("✓ MAP CREATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\n📁 Output: {output_file}")
    print(f"📊 Total predictions: {len(flood_images)}")
    print(f"\n💡 Open in browser to explore all predictions")
    print(f"{'='*70}\n")
    
    return m

def main():
    """Main entry point"""
    create_prediction_viewer()

if __name__ == "__main__":
    main()
