"""
Flood Map Overlay on Google Maps
==================================
Overlay flood simulation results on DEM and render on Google Maps.
Transforms coordinates from UTM to WGS84 for web mapping.
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
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
DATA_ROOT = Path("..") / "FloodCastBench_Dataset-and-Models-main" / "Data_Generation_Code" / "FloodCastBench"
FLOOD_OUTPUT_DIR = DATA_ROOT / "High-fidelity_flood_forecasting" / "60m" / "Australia"
DEM_PATH = DATA_ROOT / "Relevant_data" / "DEM" / "Australia_DEM.tif"
TFW_PATH = DATA_ROOT / "Relevant_data" / "DEM" / "Australia_DEM.tfw"

# Coordinate system
DEFAULT_EPSG = 32756  # UTM Zone 56S (Brisbane region, Australia)

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


class FloodMapOverlay:
    """Create flood map overlay on DEM with Google Maps export"""
    
    def __init__(self, dem_path, flood_dir, tfw_path=None, epsg_code=None):
        self.dem_path = Path(dem_path)
        self.flood_dir = Path(flood_dir)
        self.tfw_path = Path(tfw_path) if tfw_path else None
        self.epsg_code = epsg_code or DEFAULT_EPSG
        
        self.dem = None
        self.flood_data = None
        self.flood_files = []
        self.transform = None
        self.source_crs = None
        self.bounds_proj = None
        self.bounds_latlon = None
        
    def load_data(self):
        """Load DEM and flood data"""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Load DEM
        print(f"\nLoading DEM: {self.dem_path.name}")
        self.dem = tifffile.imread(self.dem_path)
        print(f"✓ DEM loaded: {self.dem.shape} pixels")
        print(f"  Elevation range: {np.min(self.dem):.1f} - {np.max(self.dem):.1f} m")
        
        # Get georeferencing
        if RASTERIO_AVAILABLE:
            try:
                with rasterio.open(self.dem_path) as src:
                    self.transform = src.transform
                    self.source_crs = src.crs
                    if self.source_crs:
                        print(f"✓ Found embedded CRS: {self.source_crs}")
                        
                        # Get bounds from rasterio
                        bounds = src.bounds
                        self.bounds_proj = {
                            'x_min': bounds.left,
                            'x_max': bounds.right,
                            'y_min': bounds.bottom,
                            'y_max': bounds.top
                        }
                        print(f"✓ Bounds: X=[{self.bounds_proj['x_min']:.2f}, {self.bounds_proj['x_max']:.2f}]")
                        print(f"           Y=[{self.bounds_proj['y_min']:.2f}, {self.bounds_proj['y_max']:.2f}]")
            except Exception as e:
                print(f"⚠ Could not read GeoTIFF metadata: {e}")
        
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
                print(f"✓ Using EPSG:{self.epsg_code}")
        
        if self.bounds_proj is None:
            raise ValueError("No georeferencing information found!")
        
        # Load flood files
        print(f"\nSearching for flood files in: {self.flood_dir.name}")
        self.flood_files = sorted(list(self.flood_dir.glob("*.tif")))
        
        if not self.flood_files:
            raise FileNotFoundError(f"No flood files found in {self.flood_dir}")
        
        print(f"✓ Found {len(self.flood_files)} flood timesteps")
        
        return True
    
    def load_flood_timestep(self, timestep_index=-1):
        """Load specific flood timestep (-1 = last/final state)"""
        if timestep_index < 0:
            timestep_index = len(self.flood_files) + timestep_index
        
        flood_file = self.flood_files[timestep_index]
        print(f"\nLoading flood timestep {timestep_index}: {flood_file.name}")
        
        self.flood_data = tifffile.imread(flood_file)
        print(f"✓ Flood data loaded: {self.flood_data.shape} pixels")
        print(f"  Depth range: {np.min(self.flood_data):.3f} - {np.max(self.flood_data):.3f} m")
        print(f"  Flooded area (>5cm): {np.sum(self.flood_data > 0.05):,} pixels ({100*np.sum(self.flood_data > 0.05)/self.flood_data.size:.2f}%)")
        
        return self.flood_data
    
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
        
        print("\n✓ Conversion successful!")
        print(f"\nWGS84 Bounds (Lat/Lon):")
        print(f"  North: {lat_max:.6f}°")
        print(f"  South: {lat_min:.6f}°")
        print(f"  East:  {lon_max:.6f}°")
        print(f"  West:  {lon_min:.6f}°")
        print(f"\nCenter Point:")
        print(f"  Latitude:  {self.bounds_latlon['center_lat']:.6f}°")
        print(f"  Longitude: {self.bounds_latlon['center_lon']:.6f}°")
        
        return True
    
    def create_overlay_image(self, output_path='flood_overlay.png', alpha=0.6):
        """Create PNG overlay image with transparency"""
        print("\n" + "="*60)
        print("CREATING OVERLAY IMAGE")
        print("="*60)
        
        if self.flood_data is None:
            print("⚠ Loading final flood timestep...")
            self.load_flood_timestep(-1)
        
        # Create RGBA image
        height, width = self.flood_data.shape
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Define color mapping for flood depth
        # Blue gradient: deeper = darker blue
        flood_depth_normalized = np.clip(self.flood_data / np.percentile(self.flood_data[self.flood_data > 0], 95), 0, 1)
        
        # Only show areas with depth > 5cm
        wet_mask = self.flood_data > 0.05
        
        # Create blue overlay
        overlay[:, :, 0] = 0  # Red
        overlay[:, :, 1] = (100 * flood_depth_normalized).astype(np.uint8)  # Green
        overlay[:, :, 2] = (255 * flood_depth_normalized).astype(np.uint8)  # Blue
        overlay[:, :, 3] = (alpha * 255 * wet_mask).astype(np.uint8)  # Alpha
        
        # Save
        output_file = Path("outputs") / output_path
        output_file.parent.mkdir(exist_ok=True)
        
        from PIL import Image
        img = Image.fromarray(overlay, 'RGBA')
        img.save(output_file)
        
        print(f"✓ Overlay image saved: {output_file}")
        print(f"  Size: {width} × {height} pixels")
        
        return output_file
    
    def create_folium_map(self, output_html='flood_google_map.html', downsample=4):
        """Create interactive Folium map with flood overlay"""
        if not FOLIUM_AVAILABLE:
            print("\n✗ Folium not installed. Install with: pip install folium")
            return None
        
        print("\n" + "="*60)
        print("CREATING INTERACTIVE GOOGLE MAP")
        print("="*60)
        
        if self.flood_data is None:
            print("⚠ Loading final flood timestep...")
            self.load_flood_timestep(-1)
        
        # Create base map
        m = folium.Map(
            location=[self.bounds_latlon['center_lat'], self.bounds_latlon['center_lon']],
            zoom_start=13,
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
        
        # Downsample flood data for better performance
        if downsample > 1:
            flood_ds = self.flood_data[::downsample, ::downsample]
            dem_ds = self.dem[::downsample, ::downsample]
            print(f"✓ Downsampled data: {self.flood_data.shape} → {flood_ds.shape}")
        else:
            flood_ds = self.flood_data
            dem_ds = self.dem
        
        # Create flood overlay with custom colormap
        from matplotlib import cm
        import matplotlib.colors as mcolors
        
        # Normalize flood depth
        vmax = np.percentile(self.flood_data[self.flood_data > 0.05], 95) if np.any(self.flood_data > 0.05) else 1.0
        flood_normalized = np.clip(flood_ds / vmax, 0, 1)
        
        # Apply colormap
        cmap = cm.get_cmap('Blues')
        flood_rgba = cmap(flood_normalized)
        
        # Set transparency based on flood depth
        wet_mask = flood_ds > 0.05
        flood_rgba[:, :, 3] = wet_mask * 0.6  # 60% opacity for flooded areas
        
        # Convert to uint8
        flood_rgba_uint8 = (flood_rgba * 255).astype(np.uint8)
        
        # Add flood overlay as image
        from PIL import Image
        import io
        import base64
        
        img = Image.fromarray(flood_rgba_uint8, 'RGBA')
        
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
            opacity=0.6,
            interactive=True,
            name='Flood Depth',
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
                <b>Flood Simulation Coverage</b><br>
                Grid: {self.flood_data.shape[0]} × {self.flood_data.shape[1]} pixels<br>
                Max Depth: {np.max(self.flood_data):.2f} m<br>
                Flooded Area: {100*np.sum(self.flood_data > 0.05)/self.flood_data.size:.2f}%
            """,
            tooltip="Simulation Boundary"
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
        
        # Add center marker
        center_y, center_x = self.flood_data.shape[0] // 2, self.flood_data.shape[1] // 2
        center_flood_depth = self.flood_data[center_y, center_x]
        center_elevation = self.dem[center_y, center_x]
        
        folium.Marker(
            [self.bounds_latlon['center_lat'], self.bounds_latlon['center_lon']],
            popup=f"""
                <b>Center Point</b><br>
                Lat: {self.bounds_latlon['center_lat']:.6f}<br>
                Lon: {self.bounds_latlon['center_lon']:.6f}<br>
                Elevation: {center_elevation:.1f} m<br>
                Flood Depth: {center_flood_depth:.2f} m
            """,
            icon=folium.Icon(color='green', icon='bullseye', prefix='fa'),
            tooltip="Center Point"
        ).add_to(m)
        
        # Add colorbar legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
            <p style="margin:0; font-weight:bold;">Flood Depth (m)</p>
            <p style="margin:5px 0;">
                <span style="background-color: rgba(0,100,255,0.6); padding: 2px 10px;">Deep</span> 
                {vmax:.2f} m
            </p>
            <p style="margin:5px 0;">
                <span style="background-color: rgba(0,50,200,0.4); padding: 2px 10px;">Medium</span> 
                {vmax/2:.2f} m
            </p>
            <p style="margin:5px 0;">
                <span style="background-color: rgba(100,150,255,0.3); padding: 2px 10px;">Shallow</span> 
                0.05 m
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
        
        print(f"\n✓ Interactive map saved: {output_path}")
        print("  Open this file in your web browser to view on Google Maps!")
        
        return m
    
    def export_geojson(self, output_file='flood_map.geojson', simplify_threshold=0.1):
        """Export flood extent as GeoJSON"""
        print("\n" + "="*60)
        print("EXPORTING GEOJSON")
        print("="*60)
        
        if self.flood_data is None:
            self.load_flood_timestep(-1)
        
        # Create simple bounding box for now
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": "Flood Simulation Area",
                        "max_depth": float(np.max(self.flood_data)),
                        "mean_depth": float(np.mean(self.flood_data[self.flood_data > 0.05])),
                        "flooded_pixels": int(np.sum(self.flood_data > 0.05)),
                        "total_pixels": int(self.flood_data.size),
                        "flooded_percent": float(100 * np.sum(self.flood_data > 0.05) / self.flood_data.size)
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [self.bounds_latlon['west'], self.bounds_latlon['north']],
                            [self.bounds_latlon['east'], self.bounds_latlon['north']],
                            [self.bounds_latlon['east'], self.bounds_latlon['south']],
                            [self.bounds_latlon['west'], self.bounds_latlon['south']],
                            [self.bounds_latlon['west'], self.bounds_latlon['north']]
                        ]]
                    }
                }
            ]
        }
        
        output_path = Path("outputs") / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"✓ GeoJSON saved: {output_path}")
        
        return output_path


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution"""
    print("="*60)
    print("FLOOD MAP OVERLAY ON GOOGLE MAPS")
    print("FloodCastBench - 60m Resolution Australia")
    print("="*60)
    
    # Check if data exists
    if not FLOOD_OUTPUT_DIR.exists():
        print(f"\n✗ ERROR: Flood output directory not found:")
        print(f"  {FLOOD_OUTPUT_DIR}")
        exit(1)
    
    if not DEM_PATH.exists():
        print(f"\n✗ ERROR: DEM file not found:")
        print(f"  {DEM_PATH}")
        exit(1)
    
    # Create overlay object
    overlay = FloodMapOverlay(DEM_PATH, FLOOD_OUTPUT_DIR, TFW_PATH, DEFAULT_EPSG)
    
    # Load data
    overlay.load_data()
    
    # Ask which timestep to visualize
    print("\n" + "="*60)
    print("SELECT FLOOD TIMESTEP")
    print("="*60)
    print(f"Available: {len(overlay.flood_files)} timesteps")
    print("\nOptions:")
    print("  1. Final state (last timestep)")
    print("  2. Peak flood (maximum depth)")
    print("  3. Custom timestep")
    
    choice = input("\nEnter choice [1-3] (or press Enter for 1): ").strip()
    
    if choice == "2":
        # Find peak
        print("\nScanning for peak flood...")
        max_depths = []
        for i, ff in enumerate(overlay.flood_files):
            if i % 50 == 0:  # Sample every 50th
                flood = tifffile.imread(ff)
                max_depths.append((i, np.max(flood)))
        peak_idx = max(max_depths, key=lambda x: x[1])[0]
        print(f"✓ Peak found at timestep {peak_idx}")
        overlay.load_flood_timestep(peak_idx)
        
    elif choice == "3":
        idx = int(input(f"Enter timestep index (0-{len(overlay.flood_files)-1}): "))
        overlay.load_flood_timestep(idx)
    else:
        # Default: final state
        overlay.load_flood_timestep(-1)
    
    # Convert coordinates
    overlay.convert_to_wgs84()
    
    # Create outputs
    print("\n" + "="*60)
    print("GENERATING OUTPUTS")
    print("="*60)
    
    # Create interactive map
    overlay.create_folium_map('flood_overlay_google_map.html', downsample=2)
    
    # Export GeoJSON
    overlay.export_geojson('flood_map.geojson')
    
    # Summary
    print("\n" + "="*60)
    print("✓ COMPLETE!")
    print("="*60)
    print("\nGenerated files in 'outputs/' folder:")
    print("  1. flood_overlay_google_map.html  → Open in web browser")
    print("  2. flood_map.geojson              → Use in GIS software")
    print("\n" + "="*60)
    print("\n💡 TIP: Open the HTML file to see the flood overlay on Google Satellite view!")


if __name__ == "__main__":
    main()
