"""
Ho Chi Minh City DEM Visualization on Google Maps
=================================================
Overlay DEM on Google Maps.
Transforms coordinates from UTM Zone 48N to WGS84 for web mapping.
"""

import numpy as np
import tifffile
from pathlib import Path
import io
import base64
import json

# Try to import geospatial libraries
try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    print("⚠ Warning: rasterio not installed")
    RASTERIO_AVAILABLE = False

try:
    from pyproj import Transformer, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    print("✗ ERROR: pyproj is required for coordinate conversion")
    print("  Install with: pip install pyproj")
    PYPROJ_AVAILABLE = False
    exit(1)

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

# Paths
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "hochiminh"
DEM_PATH = DATA_DIR / "HoChiMinh_DEM.tif"
TFW_PATH = DATA_DIR / "HoChiMinh_DEM.tfw"

# Coordinate system
# Ho Chi Minh City is in UTM Zone 48N
DEFAULT_EPSG = 32648  

# ============================================================================
# Helper Functions
# ============================================================================

def read_world_file(tfw_path):
    """Parse .tfw world file"""
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


class DEMMapOverlay:
    """Create DEM overlay on Google Maps"""
    
    def __init__(self, dem_path, tfw_path=None, epsg_code=None):
        self.dem_path = Path(dem_path)
        self.tfw_path = Path(tfw_path) if tfw_path else None
        self.epsg_code = epsg_code or DEFAULT_EPSG
        
        self.dem = None
        self.transform = None
        self.source_crs = None
        self.bounds_proj = None
        self.bounds_latlon = None
        
    def load_data(self):
        """Load DEM data"""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Load DEM
        print(f"\nLoading DEM: {self.dem_path.name}")
        if not self.dem_path.exists():
             raise FileNotFoundError(f"DEM file not found: {self.dem_path}")

        self.dem = tifffile.imread(self.dem_path)
        print(f"[OK] DEM loaded: {self.dem.shape} pixels")
        print(f"  Elevation range: {np.min(self.dem):.1f} - {np.max(self.dem):.1f} m")
        
        # Get georeferencing
        if RASTERIO_AVAILABLE:
            try:
                with rasterio.open(self.dem_path) as src:
                    self.transform = src.transform
                    self.source_crs = src.crs
                    if self.source_crs:
                        print(f"[OK] Found embedded CRS: {self.source_crs}")
                        
                        # Get bounds from rasterio
                        bounds = src.bounds
                        self.bounds_proj = {
                            'x_min': bounds.left,
                            'x_max': bounds.right,
                            'y_min': bounds.bottom,
                            'y_max': bounds.top
                        }
                        print(f"[OK] Bounds: X=[{self.bounds_proj['x_min']:.2f}, {self.bounds_proj['x_max']:.2f}]")
                        print(f"           Y=[{self.bounds_proj['y_min']:.2f}, {self.bounds_proj['y_max']:.2f}]")
            except Exception as e:
                print(f"⚠ Could not read GeoTIFF metadata with rasterio: {e}")
        
        # Fallback to .tfw
        if self.bounds_proj is None and self.tfw_path and self.tfw_path.exists():
            print(f"\nReading world file: {self.tfw_path.name}")
            geo = read_world_file(self.tfw_path)
            
            if geo:
                height, width = self.dem.shape
                x_min = geo['x_origin']
                y_max = geo['y_origin']
                x_max = x_min + width * geo['pixel_width']
                y_min = y_max + height * geo['pixel_height']
                
                self.bounds_proj = {
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max
                }
                
                self.source_crs = CRS.from_epsg(self.epsg_code)
                print(f"[OK] Using EPSG:{self.epsg_code} from configuration")
        
        if self.bounds_proj is None:
            raise ValueError("No georeferencing information found!")
        
        return True
    
    def convert_to_wgs84(self):
        """Convert bounds to WGS84 (Google Maps coordinates)"""
        print("\n" + "="*60)
        print("CONVERTING TO WGS84 (GOOGLE MAPS COORDINATES)")
        print("="*60)
        
        # Create transformer
        transformer = Transformer.from_crs(self.source_crs, "EPSG:4326", always_xy=True)
        
        # Transform corners
        lon_min, lat_max = transformer.transform(self.bounds_proj['x_min'], 
                                                   self.bounds_proj['y_max'])
        lon_max, lat_min = transformer.transform(self.bounds_proj['x_max'], 
                                                   self.bounds_proj['y_min'])
        
        self.bounds_latlon = {
            'north': lat_max,
            'south': lat_min,
            'east': lon_max,
            'west': lon_min,
            'center_lat': (lat_max + lat_min) / 2,
            'center_lon': (lon_max + lon_min) / 2
        }
        
        print("\n[OK] Conversion successful!")
        print(f"\nWGS84 Bounds (Lat/Lon):")
        print(f"  North: {lat_max:.6f} deg")
        print(f"  South: {lat_min:.6f} deg")
        print(f"  East:  {lon_max:.6f} deg")
        print(f"  West:  {lon_min:.6f} deg")
        print(f"\nCenter Point:")
        print(f"  Latitude:  {self.bounds_latlon['center_lat']:.6f} deg")
        print(f"  Longitude: {self.bounds_latlon['center_lon']:.6f} deg")
        
        return True

    def create_folium_map(self, output_html='hochiminh_dem_map.html', downsample=4):
        """Create interactive Folium map with DEM overlay"""
        if not FOLIUM_AVAILABLE:
            print("\n✗ Folium not installed. Install with: pip install folium")
            return None
        
        print("\n" + "="*60)
        print("CREATING INTERACTIVE MAP")
        print("="*60)
        
        # Create base map
        m = folium.Map(
            location=[self.bounds_latlon['center_lat'], self.bounds_latlon['center_lon']],
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Add different basemap options
        folium.TileLayer('Stamen Terrain', name='Terrain', attr='Stamen').add_to(m)
        folium.TileLayer('CartoDB positron', name='Light Map', attr='CartoDB').add_to(m)
        
        # Add Google Satellite
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='Google Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Downsample DEM for better performance
        if downsample > 1:
            dem_ds = self.dem[::downsample, ::downsample]
            print(f"[OK] Downsampled DEM: {self.dem.shape} -> {dem_ds.shape}")
        else:
            dem_ds = self.dem
        
        # Create DEM visualization
        from matplotlib import cm
        import matplotlib.colors as mcolors
        
        # Filter out nodata or very low values
        # Treating 0 as NoData/Transparent background
        valid_mask = dem_ds != 0
        
        if np.sum(valid_mask) == 0:
            print("[X] Warning: No valid pixels found (all 0). Check DEM data.")
            return None

        vmin = np.min(dem_ds[valid_mask])
        vmax = np.max(dem_ds[valid_mask])
        
        print(f"  Visualization Range: {vmin:.1f}m to {vmax:.1f}m")

        # Normalize elevation
        dem_normalized = np.clip((dem_ds - vmin) / (vmax - vmin), 0, 1)
        
        # Apply colormap (e.g., terrain)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('terrain')
        dem_rgba = cmap(dem_normalized)
        
        # Set transparency
        # Set global opacity for valid pixels
        dem_rgba[:, :, 3] = 0.7 * valid_mask
        # Explicitly make invalid pixels transparent (alpha=0)
        dem_rgba[~valid_mask, 3] = 0
        
        # Convert to uint8
        dem_rgba_uint8 = (dem_rgba * 255).astype(np.uint8)
        
        # Add overlay as image
        from PIL import Image
        
        img = Image.fromarray(dem_rgba_uint8, 'RGBA')
        
        # Save to buffer
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        
        # Add as image overlay
        folium.raster_layers.ImageOverlay(
            image=f'data:image/png;base64,{img_base64}',
            bounds=[
                [self.bounds_latlon['south'], self.bounds_latlon['west']],
                [self.bounds_latlon['north'], self.bounds_latlon['east']]
            ],
            opacity=0.7,
            interactive=True,
            name='Elevation (DEM)',
            overlay=True,
            control=True
        ).add_to(m)
        
        # Add boundary rectangle
        folium.Rectangle(
            bounds=[
                [self.bounds_latlon['south'], self.bounds_latlon['west']],
                [self.bounds_latlon['north'], self.bounds_latlon['east']]
            ],
            color='red',
            weight=2,
            fill=False,
            popup=f"""
                <b>DEM Coverage</b><br>
                Grid: {self.dem.shape[0]} × {self.dem.shape[1]} pixels<br>
                Min Elev: {vmin:.1f} m<br>
                Max Elev: {vmax:.1f} m
            """,
            tooltip="DEM Boundary"
        ).add_to(m)
        
        # Add corner markers
        corners = [
            ([self.bounds_latlon['north'], self.bounds_latlon['west']], 'NW', 'red'),
            ([self.bounds_latlon['north'], self.bounds_latlon['east']], 'NE', 'orange'),
            ([self.bounds_latlon['south'], self.bounds_latlon['west']], 'SW', 'purple'),
            ([self.bounds_latlon['south'], self.bounds_latlon['east']], 'SE', 'darkred'),
        ]
        
        for coord, label, color in corners:
            folium.Marker(
                coord,
                popup=f"<b>{label} Corner</b><br>Lat: {coord[0]:.6f}<br>Lon: {coord[1]:.6f}",
                icon=folium.Icon(color=color, icon='info-sign'),
                tooltip=f"{label} Corner"
            ).add_to(m)
        
        # Add colorbar legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
            <p style="margin:0; font-weight:bold;">Elevation (m)</p>
            <p style="margin:5px 0;">
                <span style="background-color: #F0F9E8; border: 1px solid #ccc; padding: 2px 10px;">&nbsp;</span> 
                High: {vmax:.1f} m
            </p>
             <p style="margin:5px 0;">
                <span style="background-color: #7BCCC4; border: 1px solid #ccc; padding: 2px 10px;">&nbsp;</span> 
                Mid: {(vmin+vmax)/2:.1f} m
            </p>
            <p style="margin:5px 0;">
                <span style="background-color: #0868AC; border: 1px solid #ccc; padding: 2px 10px;">&nbsp;</span> 
                Low: {vmin:.1f} m
            </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl(position='topright').add_to(m)
        
        # Add plugins
        plugins.Fullscreen(position='topleft').add_to(m)
        plugins.MousePosition(
            position='bottomleft',
            separator=' | ',
            prefix='Coordinates:'
        ).add_to(m)
        plugins.MeasureControl(
            position='topleft',
            primary_length_unit='meters',
            primary_area_unit='sqmeters'
        ).add_to(m)
        
        # Save
        output_path = Path("outputs") / output_html
        output_path.parent.mkdir(exist_ok=True)
        m.save(str(output_path))
        
        print(f"\n[OK] Interactive map saved: {output_path}")
        print("  Open this file in your web browser to view on Google Maps!")
        
        return m


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution"""
    print("="*60)
    print("HO CHI MINH CITY DEM VISUALIZATION")
    print("="*60)
    
    # Check if data exists
    if not DEM_PATH.exists():
        print(f"\n[X] ERROR: DEM file not found:")
        print(f"  {DEM_PATH}")
        print(f"  Expected location: {DATA_DIR}")
        exit(1)
    
    # Create overlay object
    # Defaulting to UTM Zone 48N (EPSG:32648)
    overlay = DEMMapOverlay(DEM_PATH, TFW_PATH, DEFAULT_EPSG)
    
    # Load data
    overlay.load_data()
    
    # Convert coordinates
    overlay.convert_to_wgs84()
    
    # Create outputs
    print("\n" + "="*60)
    print("GENERATING OUTPUTS")
    print("="*60)
    
    # Create interactive map
    # Using downsample=5 for better detail
    overlay.create_folium_map('hochiminh_dem_map.html', downsample=5)
    
    # Summary
    print("\n" + "="*60)
    print("[OK] COMPLETE!")
    print("="*60)
    print("\nGenerated file:")
    print("  outputs/hochiminh_dem_map.html  -> Open in web browser")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

