"""
06 - Convert to Google Maps Coordinates
=========================================
Convert FloodCastBench data coordinates to WGS84 (Google Maps compatible).
Export as interactive HTML maps and KML files.
"""

import numpy as np
import tifffile
from pathlib import Path
import json

# Try to import required libraries
try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    print("⚠ Warning: rasterio not installed. Using fallback method.")
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

# Path to data
DATA_ROOT = Path("..") / "FloodCastBench_Dataset-and-Models-main" / "Data_Generation_Code" / "FloodCastBench"
DEM_PATH = DATA_ROOT / "Relevant_data" / "DEM" / "Australia_DEM.tif"
TFW_PATH = DATA_ROOT / "Relevant_data" / "DEM" / "Australia_DEM.tfw"

# Default EPSG code for Australia (you may need to adjust this)
# Common Australian EPSG codes:
#   32755: UTM Zone 55S (Sydney region)
#   32756: UTM Zone 56S (Brisbane region)
#   32754: UTM Zone 54S (Perth region)
DEFAULT_EPSG = 32756  # UTM Zone 56S

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


def detect_coordinate_system(tfw_info):
    """Detect likely coordinate system from .tfw values"""
    if tfw_info is None:
        return None
    
    x_origin = tfw_info['x_origin']
    y_origin = tfw_info['y_origin']
    
    print("\n--- Coordinate System Detection ---")
    print(f"X Origin: {x_origin:.2f}")
    print(f"Y Origin: {y_origin:.2f}")
    
    # Check for lat/lon (WGS84)
    if -180 <= x_origin <= 180 and -90 <= y_origin <= 90:
        print("✓ Detected: WGS84 Geographic (EPSG:4326)")
        print("  Coordinates appear to be latitude/longitude")
        return 4326
    
    # Check for Australian UTM zones
    elif 100000 < x_origin < 900000:
        if -10000000 < y_origin < -1000000:  # Southern hemisphere
            print("✓ Detected: UTM Southern Hemisphere")
            print(f"  Using default EPSG:{DEFAULT_EPSG}")
            print("  (You can adjust DEFAULT_EPSG in the script if needed)")
            return DEFAULT_EPSG
    
    print("⚠ Unknown coordinate system")
    print(f"  Using default EPSG:{DEFAULT_EPSG}")
    return DEFAULT_EPSG


# ============================================================================
# Main Conversion Class
# ============================================================================

class CoordinateConverter:
    """Convert DEM coordinates to Google Maps (WGS84)"""
    
    def __init__(self, dem_path, tfw_path=None, epsg_code=None):
        self.dem_path = Path(dem_path)
        self.tfw_path = Path(tfw_path) if tfw_path else None
        self.epsg_code = epsg_code
        
        self.dem = None
        self.transform = None
        self.source_crs = None
        self.bounds_latlon = None
        
    def load_data(self):
        """Load DEM and georeferencing information"""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Load DEM
        print(f"\nLoading DEM: {self.dem_path.name}")
        self.dem = tifffile.imread(self.dem_path)
        print(f"✓ DEM loaded: {self.dem.shape} pixels")
        
        # Try to read CRS from GeoTIFF metadata
        if RASTERIO_AVAILABLE:
            try:
                with rasterio.open(self.dem_path) as src:
                    self.transform = src.transform
                    self.source_crs = src.crs
                    if self.source_crs:
                        print(f"✓ Found embedded CRS: {self.source_crs}")
                        
                        # Calculate bounds from the transform
                        height, width = self.dem.shape
                        
                        # Use the transform to get bounds
                        try:
                            x_min, y_max = self.transform * (0, 0)
                            x_max, y_min = self.transform * (width, height)
                        except Exception as e:
                            print(f"⚠ Error calculating bounds from transform: {e}")
                            # Use rasterio bounds instead
                            bounds = src.bounds
                            x_min, y_min, x_max, y_max = bounds.left, bounds.bottom, bounds.right, bounds.top
                        
                        self.bounds_proj = {
                            'x_min': x_min,
                            'x_max': x_max,
                            'y_min': y_min,
                            'y_max': y_max
                        }
                        
                        print(f"✓ Bounds extracted from GeoTIFF")
                        print(f"  Bounds: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
                        
                        return True
            except Exception as e:
                print(f"⚠ Could not read GeoTIFF metadata: {e}")
        
        # Fallback to .tfw file
        if self.tfw_path and self.tfw_path.exists():
            print(f"\nReading world file: {self.tfw_path.name}")
            geo = read_world_file(self.tfw_path)
            
            if geo:
                height, width = self.dem.shape
                
                # Calculate bounds
                x_min = geo['x_origin']
                y_max = geo['y_origin']
                x_max = x_min + width * geo['pixel_width']
                y_min = y_max + height * geo['pixel_height']
                
                print(f"✓ World file loaded")
                print(f"  Resolution: {abs(geo['pixel_width'])}m × {abs(geo['pixel_height'])}m")
                print(f"  Bounds: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
                
                # Detect or use provided EPSG
                if self.epsg_code is None:
                    self.epsg_code = detect_coordinate_system(geo)
                
                if PYPROJ_AVAILABLE and self.epsg_code:
                    self.source_crs = CRS.from_epsg(self.epsg_code)
                    print(f"✓ Using EPSG:{self.epsg_code} as source CRS")
                
                # Store bounds
                self.bounds_proj = {
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max
                }
                
                return True
        
        print("✗ No georeferencing information found!")
        return False
    
    def convert_to_wgs84(self):
        """Convert bounds to WGS84 (lat/lon) for Google Maps"""
        if not PYPROJ_AVAILABLE:
            print("\n✗ ERROR: pyproj is required for coordinate conversion")
            print("  Install with: pip install pyproj")
            return False
        
        if self.source_crs is None:
            print("\n✗ ERROR: Source CRS not defined")
            return False
        
        print("\n" + "="*60)
        print("CONVERTING TO WGS84 (GOOGLE MAPS COORDINATES)")
        print("="*60)
        
        # Create transformer to WGS84
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
    
    def create_folium_map(self, output_html='google_map_interactive.html'):
        """Create interactive Folium map (Google Maps style)"""
        if not FOLIUM_AVAILABLE:
            print("\n⚠ Folium not installed. Install with: pip install folium")
            return None
        
        if self.bounds_latlon is None:
            print("\n✗ ERROR: Must convert to WGS84 first")
            return None
        
        print("\n" + "="*60)
        print("CREATING INTERACTIVE MAP")
        print("="*60)
        
        # Create base map
        m = folium.Map(
            location=[self.bounds_latlon['center_lat'], self.bounds_latlon['center_lon']],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add different basemap options (include attribution to satisfy folium)
        folium.TileLayer('Stamen Terrain', name='Terrain', attr='Stamen').add_to(m)
        folium.TileLayer('CartoDB positron', name='Light Map', attr='CartoDB').add_to(m)
        
        # Add Google Satellite layer
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='Google Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add DEM boundary rectangle
        folium.Rectangle(
            bounds=[
                [self.bounds_latlon['south'], self.bounds_latlon['west']],
                [self.bounds_latlon['north'], self.bounds_latlon['east']]
            ],
            color='red',
            weight=3,
            fill=True,
            fillColor='blue',
            fillOpacity=0.2,
            popup=f"""
                <b>DEM Coverage Area</b><br>
                Size: {self.dem.shape[0]} × {self.dem.shape[1]} pixels<br>
                Elevation: {np.min(self.dem):.1f} - {np.max(self.dem):.1f} m<br>
                Center: ({self.bounds_latlon['center_lat']:.6f}, {self.bounds_latlon['center_lon']:.6f})
            """,
            tooltip="DEM Boundary - Click for details"
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
        folium.Marker(
            [self.bounds_latlon['center_lat'], self.bounds_latlon['center_lon']],
            popup=f"""
                <b>DEM Center</b><br>
                Lat: {self.bounds_latlon['center_lat']:.6f}<br>
                Lon: {self.bounds_latlon['center_lon']:.6f}<br>
                Elevation: {self.dem[self.dem.shape[0]//2, self.dem.shape[1]//2]:.1f} m
            """,
            icon=folium.Icon(color='green', icon='mountain', prefix='fa'),
            tooltip="DEM Center Point"
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl(position='topright').add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen(position='topleft').add_to(m)
        
        # Add mouse position display
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
        
        # Save map
        output_path = Path("outputs") / output_html
        output_path.parent.mkdir(exist_ok=True)
        m.save(str(output_path))
        
        print(f"\n✓ Interactive map saved to: {output_path}")
        print("  Open this file in your web browser to view on Google Maps!")
        
        return m
    
    def export_kml(self, output_file='dem_coverage.kml'):
        """Export as KML for Google Earth"""
        if self.bounds_latlon is None:
            print("\n✗ ERROR: Must convert to WGS84 first")
            return None
        
        kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>FloodCastBench DEM - Australia</name>
    <description>DEM coverage area for flood forecasting simulation</description>
    
    <Style id="demBoundary">
      <LineStyle>
        <color>ff0000ff</color>
        <width>4</width>
      </LineStyle>
      <PolyStyle>
        <color>4d0000ff</color>
      </PolyStyle>
    </Style>
    
    <Placemark>
      <name>DEM Coverage Area</name>
      <description><![CDATA[
        <b>DEM Information:</b><br/>
        Grid Size: {self.dem.shape[0]} × {self.dem.shape[1]} pixels<br/>
        Elevation Range: {np.min(self.dem):.1f} - {np.max(self.dem):.1f} meters<br/>
        Mean Elevation: {np.mean(self.dem):.1f} meters<br/>
        <br/>
        <b>Coordinates:</b><br/>
        North: {self.bounds_latlon['north']:.6f}°<br/>
        South: {self.bounds_latlon['south']:.6f}°<br/>
        East: {self.bounds_latlon['east']:.6f}°<br/>
        West: {self.bounds_latlon['west']:.6f}°<br/>
      ]]></description>
      <styleUrl>#demBoundary</styleUrl>
      <Polygon>
        <extrude>1</extrude>
        <altitudeMode>clampToGround</altitudeMode>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              {self.bounds_latlon['west']},{self.bounds_latlon['north']},0
              {self.bounds_latlon['east']},{self.bounds_latlon['north']},0
              {self.bounds_latlon['east']},{self.bounds_latlon['south']},0
              {self.bounds_latlon['west']},{self.bounds_latlon['south']},0
              {self.bounds_latlon['west']},{self.bounds_latlon['north']},0
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
    
    <Placemark>
      <name>DEM Center Point</name>
      <description>Center of the DEM coverage area</description>
      <Point>
        <coordinates>{self.bounds_latlon['center_lon']},{self.bounds_latlon['center_lat']},0</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>"""
        
        output_path = Path("outputs") / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(kml_content)
        
        print(f"\n✓ KML file saved to: {output_path}")
        print("  Open in Google Earth to visualize!")
        
        return output_path
    
    def export_geojson(self, output_file='dem_coverage.geojson'):
        """Export as GeoJSON"""
        if self.bounds_latlon is None:
            print("\n✗ ERROR: Must convert to WGS84 first")
            return None
        
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": "DEM Coverage Area",
                        "grid_size": f"{self.dem.shape[0]} × {self.dem.shape[1]}",
                        "elevation_min": float(np.min(self.dem)),
                        "elevation_max": float(np.max(self.dem)),
                        "elevation_mean": float(np.mean(self.dem)),
                        "center_lat": self.bounds_latlon['center_lat'],
                        "center_lon": self.bounds_latlon['center_lon']
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
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"\n✓ GeoJSON saved to: {output_path}")
        
        return output_path


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution"""
    print("="*60)
    print("COORDINATE CONVERSION TO GOOGLE MAPS")
    print("FloodCastBench Australia DEM")
    print("="*60)
    
    # Check if files exist
    if not DEM_PATH.exists():
        print(f"\n✗ ERROR: DEM file not found at: {DEM_PATH}")
        print("Please update DATA_ROOT in the script.")
        return
    
    # Create converter
    converter = CoordinateConverter(DEM_PATH, TFW_PATH, epsg_code=DEFAULT_EPSG)
    
    # Load data
    if not converter.load_data():
        print("\n✗ Failed to load georeferencing data")
        return
    
    # Convert to WGS84
    if not converter.convert_to_wgs84():
        print("\n✗ Conversion failed")
        return
    
    # Create outputs
    print("\n" + "="*60)
    print("EXPORTING TO MULTIPLE FORMATS")
    print("="*60)
    
    # Interactive HTML map
    converter.create_folium_map('06_google_map_interactive.html')
    
    # KML for Google Earth
    converter.export_kml('06_dem_coverage.kml')
    
    # GeoJSON
    converter.export_geojson('06_dem_coverage.geojson')
    
    # Summary
    print("\n" + "="*60)
    print("✓ CONVERSION COMPLETE!")
    print("="*60)
    print("\nGenerated files in 'outputs/' folder:")
    print("  1. 06_google_map_interactive.html  → Open in web browser")
    print("  2. 06_dem_coverage.kml             → Open in Google Earth")
    print("  3. 06_dem_coverage.geojson         → Use in GIS software")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
