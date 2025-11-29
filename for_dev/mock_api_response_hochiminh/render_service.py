"""
Mock Render Service
===================
This script simulates the "Frontend" or "Map Service".
It takes the standardized API response (metadata + npy) and renders the final Google Map.

Usage:
    python render_service.py
"""

import json
import numpy as np
import folium
from folium import plugins
from pathlib import Path
from matplotlib import cm
import base64
import io
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_DIR = Path(".") # Looking in current directory
METADATA_FILE = INPUT_DIR / "metadata.json"
DATA_FILE = INPUT_DIR / "flood_depths.npy"
OUTPUT_HTML = INPUT_DIR / "final_map_view.html"

# ============================================================================
# RENDER LOGIC
# ============================================================================

def load_package():
    """Load the API package data"""
    print(f"1. Loading API Package...")
    
    if not METADATA_FILE.exists() or not DATA_FILE.exists():
        raise FileNotFoundError("Missing API package files (metadata.json or flood_depths.npy)")

    # Load Metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    print(f"   - Metadata loaded (Bounds: {metadata['bounds']['north']:.4f}, {metadata['bounds']['east']:.4f})")
    
    # Load Binary Data
    flood_data = np.load(DATA_FILE)
    print(f"   - Data loaded (Shape: {flood_data.shape})")
    
    return metadata, flood_data

def generate_overlay_image(data, max_depth):
    """Convert raw data matrix to RGBA image string for the map"""
    print(f"2. Generating Overlay Image...")
    
    # COLOR SCALING STRATEGY:
    # The global max depth might be very deep (e.g. 15m in a river channel),
    # but most flooded areas are shallow (0.5m - 1m).
    # If we scale linearly to 15m, the 1m floods look white/invisible.
    # Solution: Cap the visual scale at a lower value (e.g. 4.0m).
    VISUAL_MAX = 4.0 # Meters. Anything deeper than this is darkest blue.
    
    # Normalize data (0 to 1) based on VISUAL max
    norm_data = np.clip(data / VISUAL_MAX, 0, 1)
    
    # Apply Blue Colormap (using matplotlib)
    try:
        # Using 'jet' or 'turbo' can give better distinction for depth
        # But for flood, Blues is standard. Let's stick to Blues but handle alpha better.
        cmap = cm.get_cmap('Blues')
    except AttributeError:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('Blues')
    
    rgba_image = cmap(norm_data)
    
    # Add Transparency based on Depth
    # - Dry pixels (<= 5cm) -> Transparent
    # - Shallow pixels -> More transparent (0.5)
    # - Deep pixels -> More opaque (0.9)
    
    is_wet = data > 0.01
    
    # Create an alpha channel that scales with depth
    # Start at 0.4 opacity for shallow water, ramp up to 0.9 for max depth
    alpha_channel = np.zeros_like(data, dtype=np.float32)
    alpha_channel[is_wet] = 0.4 + (0.5 * norm_data[is_wet])
    
    # Apply alpha
    rgba_image[:, :, 3] = alpha_channel
    
    # Convert to 8-bit integer (0-255)
    img_uint8 = (rgba_image * 255).astype(np.uint8)
    
    # Encode to PNG in memory
    img_pil = Image.fromarray(img_uint8, 'RGBA')
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Convert to Base64 string for embedding in HTML
    img_base64 = base64.b64encode(buffer.read()).decode()
    
    print(f"   - Image generated and encoded ({len(img_base64)} bytes)")
    return img_base64

def create_map(metadata, img_base64):
    """Create the Folium map with the overlay"""
    print(f"3. Composing Map...")
    
    bounds = metadata['bounds']
    center = [bounds['center']['lat'], bounds['center']['lon']]
    
    # Initialize Map
    m = folium.Map(
        location=center,
        zoom_start=12,
        tiles='OpenStreetMap' # Default base
    )
    
    # Add Satellite Layer
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # Add The Flood Overlay
    # Note: We map the image corners to the bounds provided in metadata
    image_bounds = [[bounds['south'], bounds['west']], [bounds['north'], bounds['east']]]
    
    folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_base64}',
        bounds=image_bounds,
        opacity=1, # We handled opacity in the image generation itself
        name='Flood Prediction',
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m)

    # Add Metadata Popup
    stats = metadata['data_stats']
    folium.Marker(
        center,
        popup=f"""
        <b>Flood Prediction</b><br>
        Max Depth: {stats['max_depth_meters']:.2f}m<br>
        Affected Area: {stats['flooded_area_pixels']} pixels
        """,
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

    # Add Color Scale Legend
    # We capped the visual scale at 4.0m in generate_overlay_image
    visual_max = 4.0 
    
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 160px; height: 100px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;">
        <p style="margin:0 0 5px; font-weight:bold;">Flood Depth</p>
        <div style="background: linear-gradient(to right, #d0e1f2, #08306b); width: 100%; height: 15px;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 12px;">
            <span>0m</span>
            <span>{visual_max}+ m</span>
        </div>
        <div style="margin-top: 5px; font-size: 11px; color: #666;">
            Global Max: {stats['max_depth_meters']:.1f}m
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add Controls
    folium.LayerControl().add_to(m)
    plugins.Fullscreen().add_to(m)
    
    return m

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        # 1. Get Data
        meta, data = load_package()
        
        # 2. Process Image
        max_depth = meta['data_stats']['max_depth_meters']
        img_str = generate_overlay_image(data, max_depth)
        
        # 3. Render Map
        m = create_map(meta, img_str)
        
        # 4. Save
        m.save(OUTPUT_HTML)
        print(f"\nSUCCESS: Map saved to {OUTPUT_HTML}")
        print("Open this file in your browser to view the result.")
        
    except Exception as e:
        print(f"\nERROR: {e}")

