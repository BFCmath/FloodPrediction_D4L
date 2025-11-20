# FloodCastBench Data Flow - Complete Input/Output Reference

**Document Version:** 1.0  
**Date:** November 2025  
**Purpose:** Comprehensive guide to all input files, their contents, and output products

---

## 📋 Table of Contents

1. [Directory Structure](#directory-structure)
2. [Input Files - Detailed Reference](#input-files-detailed-reference)
3. [Data Loading Process](#data-loading-process)
4. [Output Files - Detailed Reference](#output-files-detailed-reference)
5. [Coordinate Systems](#coordinate-systems)
6. [Visualization Scripts Summary](#visualization-scripts-summary)

---

## 📁 Directory Structure

```
FloodCastBench_Dataset-and-Models-main/
└── Data_Generation_Code/
    └── FloodCastBench/
        ├── Relevant_data/
        │   ├── DEM/
        │   │   ├── Australia_DEM.tif          ← Digital Elevation Model
        │   │   └── Australia_DEM.tfw          ← Georeferencing (world file)
        │   │
        │   ├── Rainfall/
        │   │   └── Australia_flood/
        │   │       ├── rainfall_000.tif       ← t=0 hours
        │   │       ├── rainfall_001.tif       ← t=0.5 hours
        │   │       ├── rainfall_002.tif       ← t=1.0 hours
        │   │       └── ...                    (multiple timesteps)
        │   │
        │   ├── Initial_conditions/
        │   │   └── High-fidelity_flood_forecasting/
        │   │       ├── 30m/
        │   │       │   └── Australia/
        │   │       │       └── initial_condition.tif
        │   │       └── 60m/
        │   │           └── Australia/
        │   │               └── initial_condition.tif
        │   │
        │   └── Land_use_and_land_cover/
        │       ├── Australia.tif              ← Manning's roughness coefficient
        │       └── Australia.tfw
        │
        └── High-fidelity_flood_forecasting/
            ├── 30m/
            │   └── Australia/
            │       ├── 0.tif                  ← Flood depth at t=0 seconds
            │       ├── 1800.tif               ← Flood depth at t=1800 seconds
            │       ├── 3600.tif               ← Flood depth at t=3600 seconds
            │       └── ...                    (2881 timesteps)
            │
            └── 60m/
                └── Australia/
                    ├── 0.tif                  ← Same as 30m but lower resolution
                    ├── 1800.tif
                    └── ...
```

---

## 📥 Input Files - Detailed Reference

### 1. **Digital Elevation Model (DEM)**

**File Path:**
```
Relevant_data/DEM/Australia_DEM.tif
Relevant_data/DEM/Australia_DEM.tfw
```

**File Type:** GeoTIFF + World File (.tfw)

**What It Stores:**
```python
# When loaded:
import tifffile
dem = tifffile.imread("Australia_DEM.tif")

# Returns:
# - 2D NumPy array (rows × columns)
# - Each value = elevation in meters (above sea level)
# - dtype: float32 or float64
```

**Data Structure:**
```
Shape: (1073, 1073)  # pixels
Type: float32
Units: meters (m)
Range: [minimum_elevation, maximum_elevation]

Example values:
dem[0, 0]     = 52.3   # Elevation at top-left corner (meters)
dem[500, 500] = 45.1   # Elevation at center
dem[1072, 1072] = 38.7 # Elevation at bottom-right corner
```

**What You Get After Loading:**
- **Terrain elevation** at every pixel location
- **Slope information** (can be calculated from elevation gradient)
- **Flow direction** (water flows downhill)
- **Georeferencing** (coordinate system: UTM Zone 56S, EPSG:32756)

**Georeferencing (.tfw file):**
```
30.0          ← Pixel width (30 meters)
0.0           ← Rotation (row)
0.0           ← Rotation (column)
-30.0         ← Pixel height (-30 meters, negative = north-up)
517437.19     ← X-coordinate of top-left corner (UTM Easting)
6808613.86    ← Y-coordinate of top-left corner (UTM Northing)
```

**Coordinate Bounds:**
```
UTM Zone 56S (EPSG:32756):
  X (Easting):  517,437.19 to 549,627.19 meters (32,190 m width)
  Y (Northing): 6,776,423.86 to 6,808,613.86 meters (32,190 m height)

WGS84 (Lat/Lon):
  Latitude:  -29.139747° to -28.850054° S
  Longitude: 153.178769° to 153.510204° E
  
Location: Gold Coast Hinterland, Queensland, Australia
```

**Usage in Simulation:**
- Determines **flow direction** (water flows from high to low elevation)
- Affects **flow velocity** (steeper slopes = faster flow)
- Defines **topographic barriers** (hills/ridges block flow)

---

### 2. **Rainfall Time Series**

**File Path:**
```
Relevant_data/Rainfall/Australia_flood/
├── rainfall_000.tif  (t = 0.0 hours)
├── rainfall_001.tif  (t = 0.5 hours)
├── rainfall_002.tif  (t = 1.0 hours)
...
└── rainfall_XXX.tif
```

**File Type:** GeoTIFF (sequence)

**What It Stores:**
```python
# Load single timestep:
rainfall_t0 = tifffile.imread("rainfall_000.tif")

# Returns:
# - 2D array: rainfall intensity at t=0
# - Units: mm/hour
# - Same spatial extent as DEM
```

**Data Structure:**
```
Shape: (1073, 1073)  # Same as DEM
Type: float32
Units: mm/hour (millimeters per hour)
Range: 0 to ~200 mm/hr (extreme rainfall can be higher)

Example values:
rainfall[200, 300] = 0.0    # No rain at this pixel
rainfall[500, 500] = 15.3   # Moderate rain (15.3 mm/hr)
rainfall[600, 700] = 68.2   # Heavy rain (68.2 mm/hr)
```

**Temporal Resolution:**
- **Interval:** 30 minutes (0.5 hours) between files
- **Duration:** Total event duration = (number of files) × 0.5 hours
- **Naming:** Sequential numbers indicate time progression

**What You Get After Loading:**
```python
# Load all timesteps:
all_rainfall = []
for file in sorted(rainfall_dir.glob("*.tif")):
    rain = tifffile.imread(file)
    all_rainfall.append(rain)

all_rainfall = np.array(all_rainfall)  # Shape: (N_timesteps, 1073, 1073)

# Now you have:
# - Spatial-temporal rainfall field
# - Time series at each pixel location
# - Total precipitation over event (cumulative sum)
# - Peak rainfall intensity (max over time)
```

**Rainfall Intensity Categories:**
| Intensity (mm/hr) | Category | Description |
|-------------------|----------|-------------|
| 0 - 0.5 | Trace | Negligible |
| 0.5 - 2.5 | Light | Drizzle |
| 2.5 - 10.0 | Moderate | Normal rain |
| 10.0 - 50.0 | Heavy | Strong precipitation |
| 50.0+ | Very Heavy | Torrential downpour |

**Usage in Simulation:**
- **Water input** to the system at each timestep
- Converted to **depth added**: `depth_added = rainfall_intensity × dt`
- Example: 20 mm/hr × 0.5 hr = 10 mm = 0.01 m of water added

---

### 3. **Manning's Roughness Coefficient (Land Use/Land Cover)**

**File Path:**
```
Relevant_data/Land_use_and_land_cover/Australia.tif
Relevant_data/Land_use_and_land_cover/Australia.tfw
```

**File Type:** GeoTIFF

**What It Stores:**
```python
manning = tifffile.imread("Australia.tif")

# Returns:
# - 2D array: Manning's n coefficient at each pixel
# - Dimensionless friction parameter (s/m^(1/3))
```

**Data Structure:**
```
Shape: (1073, 1073)
Type: float32
Units: Dimensionless (s/m^(1/3))
Range: ~0.01 to ~0.15

Example values:
manning[100, 200] = 0.015  # Smooth concrete
manning[300, 400] = 0.035  # Grass
manning[500, 600] = 0.080  # Forest
```

**Manning's n Values by Land Type:**
| Land Cover | Manning's n | Flow Resistance |
|------------|-------------|-----------------|
| Water/Smooth | 0.010 - 0.015 | Very Low |
| Concrete/Pavement | 0.015 - 0.020 | Low |
| Short Grass | 0.025 - 0.035 | Low-Medium |
| Crops | 0.030 - 0.050 | Medium |
| Shrubland | 0.050 - 0.080 | Medium-High |
| Forest/Dense Vegetation | 0.080 - 0.150 | Very High |

**What You Get After Loading:**
- **Surface friction** affecting flow velocity
- **Land classification** (can infer land use from n values)
- **Flow resistance map**

**Physical Meaning:**
- **Higher n** = More friction = Slower flow
- **Lower n** = Less friction = Faster flow
- Used in **Manning's equation** for flow velocity:
  ```
  V = (1/n) × R^(2/3) × S^(1/2)
  
  Where:
    V = flow velocity (m/s)
    n = Manning's coefficient
    R = hydraulic radius (m)
    S = slope (m/m)
  ```

**Usage in Simulation:**
- Controls **flow velocity** in shallow water equations
- Affects **flood propagation speed**
- Represents **land cover impact** on flooding

---

### 4. **Initial Conditions**

**File Path:**
```
Relevant_data/Initial_conditions/High-fidelity_flood_forecasting/
├── 30m/Australia/initial_condition.tif
└── 60m/Australia/initial_condition.tif
```

**File Type:** GeoTIFF

**What It Stores:**
```python
initial = tifffile.imread("initial_condition.tif")

# Returns:
# - 2D array: initial water depth at t=0
# - Units: meters
```

**Data Structure:**
```
Shape: 
  - 30m resolution: (1073, 1073)
  - 60m resolution: (536, 536)
Type: float32
Units: meters (m)
Range: 0 to ~1.0 m (typically)

Example values:
initial[100, 200] = 0.0     # Dry at start
initial[300, 400] = 0.05    # 5 cm initial water
initial[500, 600] = 0.25    # 25 cm initial water
```

**What You Get After Loading:**
- **Starting water depth** at each location
- **Wet/dry initial state**
- **Pre-existing water bodies** (rivers, ponds)

**Usage in Simulation:**
- **Boundary condition** for flood propagation
- Represents **pre-event conditions** (soil moisture, existing water)
- Affects **initial flood spreading**

---

### 5. **Flood Simulation Output (30m Resolution)**

**File Path:**
```
High-fidelity_flood_forecasting/30m/Australia/
├── 0.tif        (t = 0 seconds = 0.00 hours)
├── 1800.tif     (t = 1800 seconds = 0.50 hours)
├── 3600.tif     (t = 3600 seconds = 1.00 hours)
├── 5400.tif     (t = 5400 seconds = 1.50 hours)
...
└── XXXXXX.tif   (final timestep)
```

**File Type:** GeoTIFF (sequence)

**What It Stores:**
```python
# Load single timestep:
flood_depth = tifffile.imread("1800.tif")

# Returns:
# - 2D array: simulated water depth at t=1800 seconds
# - Units: meters
```

**Data Structure:**
```
Shape: (1073, 1073)  # 30m resolution
Type: float32
Units: meters (m)
Range: 0 to ~15 m (maximum observed depth)

Example values:
flood[200, 300] = 0.0     # Dry (no flooding)
flood[400, 500] = 0.15    # 15 cm of water
flood[600, 700] = 2.34    # 2.34 meters of water (deep)
```

**Temporal Resolution:**
- **Interval:** 1800 seconds = 30 minutes
- **Total timesteps:** 2881 (for full simulation)
- **Total duration:** 2881 × 1800 sec = 5,185,800 sec ≈ 1441 hours ≈ 60 days

**Filename Convention:**
```
Filename: TTTTTT.tif
Where TTTTTT = seconds since start

Examples:
  0.tif      → t = 0 hours (initial)
  1800.tif   → t = 0.5 hours
  3600.tif   → t = 1.0 hours
  86400.tif  → t = 24 hours (1 day)
  172800.tif → t = 48 hours (2 days)
```

**What You Get After Loading All Timesteps:**
```python
all_floods = []
for file in sorted(flood_dir.glob("*.tif")):
    flood = tifffile.imread(file)
    all_floods.append(flood)

all_floods = np.array(all_floods)  # Shape: (2881, 1073, 1073)

# Now you can analyze:
# - Flood evolution over time
# - Maximum flood envelope: np.max(all_floods, axis=0)
# - Flood arrival time: first timestep where depth > threshold
# - Inundation duration: how long each pixel stays flooded
# - Flood recession: how quickly water drains
```

**Derived Metrics:**
```python
# Maximum flood depth at each location
max_flood = np.max(all_floods, axis=0)

# Mean flood depth (temporal average)
mean_flood = np.mean(all_floods, axis=0)

# Flood arrival time (first time depth > 5cm)
arrival_time = np.full(all_floods[0].shape, -1)
for t_idx, flood in enumerate(all_floods):
    first_flood = (flood > 0.05) & (arrival_time < 0)
    arrival_time[first_flood] = t_idx * 1800  # seconds

# Inundation frequency (% of time flooded)
inundation_freq = np.sum(all_floods > 0.05, axis=0) / len(all_floods) * 100

# Flooded area evolution
flooded_area = [np.sum(flood > 0.05) for flood in all_floods]
```

---

### 6. **Flood Simulation Output (60m Resolution)**

**File Path:**
```
High-fidelity_flood_forecasting/60m/Australia/
├── 0.tif
├── 1800.tif
...
└── XXXXXX.tif
```

**Same as 30m but:**
```
Shape: (536, 536)  # Lower resolution (60m per pixel)
Coverage: Same geographic area
Pixel size: 60m × 60m (vs 30m × 30m)
Computation: Faster but less detailed
```

**Why Two Resolutions?**
- **30m:** High detail, slower computation, larger file size
- **60m:** Lower detail, faster computation, smaller file size
- **Use case:** Downscaling experiments (predict 30m from 60m using ML)

---

## 🔄 Data Loading Process

### **Step-by-Step: What Happens When You Load Data**

#### **1. Loading DEM**
```python
import tifffile

# Load the file
dem = tifffile.imread("Australia_DEM.tif")

# What you get:
print(dem.shape)       # (1073, 1073)
print(dem.dtype)       # float32 or float64
print(np.min(dem))     # Minimum elevation (e.g., 15.3 m)
print(np.max(dem))     # Maximum elevation (e.g., 125.7 m)

# Derived information:
terrain_slope = np.gradient(dem)  # Elevation gradient
flow_direction = ...              # Computed from gradient
```

#### **2. Loading Rainfall Time Series**
```python
from pathlib import Path
import numpy as np

rainfall_dir = Path("Relevant_data/Rainfall/Australia_flood")
rain_files = sorted(list(rainfall_dir.glob("*.tif")))

# Load all timesteps
all_rain = []
for rf in rain_files:
    rain = tifffile.imread(rf)
    all_rain.append(rain)

all_rain = np.array(all_rain)

# What you get:
print(all_rain.shape)           # (N_timesteps, 1073, 1073)
print(np.max(all_rain))         # Maximum rainfall intensity (mm/hr)
cumulative = np.sum(all_rain, axis=0) * 0.5  # Total rainfall (mm)
```

#### **3. Loading Flood Output**
```python
flood_dir = Path("High-fidelity_flood_forecasting/60m/Australia")
flood_files = sorted(list(flood_dir.glob("*.tif")))

# Load all timesteps
all_floods = []
for ff in flood_files:
    flood = tifffile.imread(ff)
    all_floods.append(flood)

all_floods = np.array(all_floods)

# What you get:
print(all_floods.shape)         # (2881, 536, 536) for 60m
print(np.max(all_floods))       # Maximum water depth (m)
print(np.sum(all_floods[-1] > 0.05))  # Final flooded pixels
```

#### **4. Loading with Georeferencing**
```python
import rasterio

# Load with coordinate information
with rasterio.open("Australia_DEM.tif") as src:
    dem = src.read(1)              # Read band 1
    transform = src.transform       # Affine transform
    crs = src.crs                  # Coordinate system (EPSG:32756)
    bounds = src.bounds            # Geographic bounds
    
    # Convert pixel coordinates to real-world coordinates
    x, y = transform * (col_index, row_index)  # UTM coordinates
```

---

## 📤 Output Files - Detailed Reference

### **Outputs from Visualization Scripts**

All outputs are saved in the `experiments/outputs/` folder.

---

### **1. Static Visualizations (PNG)**

#### **01_dem_visualization.png**
```
File: outputs/01_dem_visualization.png
Size: ~2-3 MB
Dimensions: ~3300 × 2100 pixels (at 150 DPI)

Contains (9 subplots):
  ├─ Elevation map (terrain colormap)
  ├─ Hillshade (3D relief effect)
  ├─ Elevation + Hillshade overlay
  ├─ Contour lines
  ├─ Slope map (gradient steepness)
  ├─ Aspect map (slope direction)
  ├─ Elevation histogram
  ├─ Cross-section profiles (horizontal + vertical)
  └─ Statistics summary

What you learn:
  - Terrain topology
  - Flow pathways (valleys)
  - Steep vs flat areas
  - Elevation distribution
```

#### **02_rainfall_visualization.png**
```
File: outputs/02_rainfall_visualization.png
Size: ~2-3 MB
Dimensions: ~3000 × 1800 pixels

Contains (10 subplots):
  ├─ Rainfall at t=0 (initial)
  ├─ Rainfall at t=middle
  ├─ Rainfall at t=final
  ├─ Maximum rainfall (all time)
  ├─ Mean rainfall (temporal average)
  ├─ Cumulative rainfall (total mm)
  ├─ Rainfall variance (spatial)
  ├─ Intensity distribution histogram
  ├─ Time series at center pixel
  └─ Domain-averaged evolution

What you learn:
  - Rainfall patterns over time
  - Peak rainfall location/timing
  - Total water input to system
  - Temporal variability
```

#### **03_manning_visualization.png**
```
File: outputs/03_manning_visualization.png
Size: ~2-3 MB

Contains (8 subplots):
  ├─ Manning's coefficient map
  ├─ Land classification (by n value)
  ├─ Flow resistance zones
  ├─ Histogram of n values
  ├─ Pie chart (land cover %)
  ├─ Velocity potential map
  ├─ Cross-sections
  └─ Statistics

What you learn:
  - Land cover distribution
  - Flow resistance patterns
  - Surface friction effects
  - Heterogeneity of terrain
```

#### **04_initial_conditions_visualization.png**
```
File: outputs/04_initial_conditions_visualization.png
Size: ~2-3 MB

Contains:
  ├─ Initial water depth map
  ├─ Wet/dry classification
  ├─ Depth categories
  ├─ Histogram
  ├─ Statistics
  └─ Profiles

What you learn:
  - Pre-existing water
  - Starting flood conditions
  - Spatial distribution at t=0
```

#### **flood_60m_visualization.png**
```
File: outputs/flood_60m_visualization.png
Size: ~3-4 MB
Dimensions: ~3300 × 2100 pixels

Contains (16 subplots):
  Row 1: Snapshots at 0%, 25%, 50%, 75%, 100% progress (5 maps)
  Row 2: Maximum flood, Mean flood, Arrival time, Frequency, Change (5 maps)
  Row 3: Depth evolution, Flooded area, Depth distribution (3 plots)
  Row 4: DEM overlay, Cross-section, Statistics (3 panels)

What you learn:
  - Complete flood evolution
  - Peak flood extent
  - Temporal dynamics
  - Spatial patterns
  - Quantitative statistics
```

#### **flood_overlay_static.png**
```
File: outputs/flood_overlay_static.png
Size: ~3-4 MB

Contains (6 subplots):
  ├─ DEM (terrain colormap)
  ├─ Flood depth (blue gradient)
  ├─ Composite overlay (DEM + flood)
  ├─ Inundation map (binary wet/dry)
  ├─ Depth categories (5 classes)
  └─ Statistics summary

What you learn:
  - Flood overlaid on real terrain
  - Depth variations
  - Wet/dry boundaries
  - Quantitative metrics
```

---

### **2. Animations (GIF/MP4)**

#### **02_rainfall_animation_full.gif**
```
File: outputs/02_rainfall_animation_full.gif
Format: Animated GIF
Size: ~50-100 MB (depends on timesteps)
Frames: All rainfall timesteps
FPS: 10 frames/second
Duration: N_timesteps / 10 seconds

Contains:
  - Rainfall map (evolving)
  - Time series graph (building up)
  - Statistics panel (updating)
  - Progress indicator

What you learn:
  - How rainfall evolves over time
  - Spatial movement of storm
  - Intensity changes
  - Temporal patterns
```

#### **flood_60m_animation_full.mp4**
```
File: outputs/flood_60m_animation_full.mp4
Format: MP4 video (H.264)
Size: ~50-100 MB
Frames: 2881 (all timesteps)
FPS: 10
Duration: ~288 seconds = 4.8 minutes

Contains:
  - Flood depth map (evolving)
  - Max/mean depth graphs (building up)
  - Current statistics panel
  - Time indicator

What you learn:
  - Real-time flood propagation
  - How water spreads over terrain
  - Peak timing
  - Recession dynamics
```

---

### **3. Interactive Google Maps (HTML)**

#### **flood_google_map.html**
```
File: outputs/flood_google_map.html
Format: HTML with embedded JavaScript
Size: ~5-15 MB (includes base64 image)
Libraries: Folium, Leaflet.js

Opens in: Any web browser (Chrome, Firefox, Edge, Safari)

Features:
  ├─ Base layers:
  │   ├─ OpenStreetMap (default)
  │   ├─ CartoDB Positron (light)
  │   └─ Google Satellite (imagery)
  │
  ├─ Overlays:
  │   ├─ DEM + Flood composite (RGBA image)
  │   └─ Boundary rectangle (red outline)
  │
  ├─ Markers:
  │   ├─ Corner markers (NW, NE, SW, SE) with coordinates
  │   └─ Center point with statistics popup
  │
  └─ Tools:
      ├─ Layer control (toggle layers)
      ├─ Fullscreen button
      ├─ Mouse position tracker
      ├─ Measurement tool (distance/area)
      └─ Minimap (overview)

Coordinates shown:
  - WGS84 (Lat/Lon) format
  - Real-world location in Australia
  - Clickable for details

What you learn:
  - Exact geographic location of flood
  - Real-world context (roads, buildings, landscape)
  - Spatial extent in familiar map interface
  - Can zoom/pan for exploration
```

**How to Use:**
1. Open `flood_google_map.html` in web browser
2. Click layer control (top-right) to switch basemaps
3. Click center marker to see flood statistics
4. Use measurement tool to measure distances
5. Toggle flood overlay on/off
6. Zoom in to see details on satellite imagery

---

### **4. GeoJSON Files**

#### **flood_map.geojson**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "Flood Simulation Area",
        "max_depth": 15.014,
        "mean_depth": 0.876,
        "flooded_pixels": 125834,
        "total_pixels": 287296,
        "flooded_percent": 43.81
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [153.178769, -28.850054],  // West, North
          [153.510204, -28.850054],  // East, North
          [153.510204, -29.139747],  // East, South
          [153.178769, -29.139747],  // West, South
          [153.178769, -28.850054]   // Close polygon
        ]]
      }
    }
  ]
}
```

**What it contains:**
- Geographic boundary (WGS84 polygon)
- Flood statistics (metadata)
- GIS-compatible format

**How to use:**
- Import into QGIS, ArcGIS, or other GIS software
- Use in web mapping libraries (Mapbox, Leaflet)
- Overlay on other spatial data
- Export to other formats (Shapefile, KML)

---

### **5. KML Files (for Google Earth)**

#### **06_dem_coverage.kml**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>FloodCastBench DEM - Australia</name>
    <Placemark>
      <name>DEM Coverage Area</name>
      <Polygon>
        <coordinates>
          153.178769,-28.850054,0
          153.510204,-28.850054,0
          153.510204,-29.139747,0
          153.178769,-29.139747,0
          153.178769,-28.850054,0
        </coordinates>
      </Polygon>
    </Placemark>
  </Document>
</kml>
```

**What it contains:**
- Geographic boundary polygon
- DEM/flood metadata
- Google Earth compatible

**How to use:**
1. Double-click KML file
2. Opens in Google Earth (desktop or web)
3. Shows coverage area in 3D
4. Can measure, explore, add layers

---

## 🗺️ Coordinate Systems

### **UTM Zone 56S (EPSG:32756)**
```
Projection: Universal Transverse Mercator
Zone: 56 South
Hemisphere: Southern
Units: Meters (m)
Authority: EPSG

Coordinate range (for Australia DEM):
  Easting (X):  517,437 to 549,627 m
  Northing (Y): 6,776,424 to 6,808,614 m

Usage:
  - Native coordinate system of DEM
  - Preserves distances and areas
  - Used in simulation calculations
```

### **WGS84 (EPSG:4326)**
```
Datum: World Geodetic System 1984
Type: Geographic (Lat/Lon)
Units: Degrees
Authority: EPSG

Coordinate range (for Australia DEM):
  Latitude:  -29.139747° to -28.850054° (South)
  Longitude: 153.178769° to 153.510204° (East)

Usage:
  - Google Maps, web mapping
  - GPS coordinates
  - Global standard
```

### **Conversion Example**
```python
from pyproj import Transformer

# Create transformer
transformer = Transformer.from_crs("EPSG:32756", "EPSG:4326", always_xy=True)

# Convert UTM to Lat/Lon
utm_x, utm_y = 533532, 6792519  # UTM coordinates
lon, lat = transformer.transform(utm_x, utm_y)

print(f"UTM: ({utm_x}, {utm_y})")
print(f"Lat/Lon: ({lat:.6f}°, {lon:.6f}°)")

# Output:
# UTM: (533532, 6792519)
# Lat/Lon: (-28.994900°, 153.344487°)
```

---

## 📊 Visualization Scripts Summary

### **Script Inventory**

| Script | Input Files | Output Files | Purpose |
|--------|-------------|--------------|---------|
| `01_visualize_dem.py` | `Australia_DEM.tif` | `01_dem_visualization.png` | Terrain analysis |
| `02_visualize_rainfall.py` | `Rainfall/*.tif` | `02_rainfall_visualization.png`, `.gif` | Rainfall analysis |
| `03_visualize_manning.py` | `Land_use_and_land_cover/Australia.tif` | `03_manning_visualization.png` | Land cover analysis |
| `04_visualize_initial_conditions.py` | `Initial_conditions/*.tif` | `04_initial_conditions_visualization.png` | Initial state |
| `05_visualize_flood_output.py` | `High-fidelity_flood_forecasting/30m/*.tif` | `05_flood_output_30m_visualization.png` | 30m flood analysis |
| `visualize_flood_60m.py` | `High-fidelity_flood_forecasting/60m/*.tif` | `flood_60m_visualization.png`, `.mp4` | 60m flood analysis + animation |
| `06_convert_to_google_maps.py` | `Australia_DEM.tif` | `06_google_map_interactive.html`, `.kml`, `.geojson` | Coordinate conversion |
| `flood_on_google_maps.py` | `DEM + Flood` | `flood_google_map.html`, `flood_overlay_static.png` | Interactive flood map |

---

### **Complete Data Pipeline**

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT DATA                           │
└─────────────────────────────────────────────────────────┘
                           │
                           ├─→ DEM (Terrain)
                           ├─→ Rainfall (Forcing)
                           ├─→ Manning (Friction)
                           └─→ Initial Conditions (Boundary)
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              FLOOD SIMULATION                           │
│        (Saint-Venant Equations)                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ├─→ 30m resolution output (2881 timesteps)
                           └─→ 60m resolution output (2881 timesteps)
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│           VISUALIZATION SCRIPTS                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ├─→ Static PNG plots
                           ├─→ Animated GIF/MP4
                           ├─→ Interactive HTML maps
                           └─→ GeoJSON/KML exports
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              OUTPUT PRODUCTS                            │
│  (Use for analysis, presentations, publications)        │
└─────────────────────────────────────────────────────────┘
```

---

## 🎓 Quick Reference Tables

### **File Extensions & Formats**

| Extension | Format | Contains | Software |
|-----------|--------|----------|----------|
| `.tif` | GeoTIFF | Georeferenced raster data | QGIS, ArcGIS, Python |
| `.tfw` | World File | Coordinate transformation | Text editor |
| `.png` | PNG Image | Static visualization | Any image viewer |
| `.gif` | GIF Animation | Animated sequence | Web browser |
| `.mp4` | MP4 Video | Video animation | Video player |
| `.html` | HTML Document | Interactive web map | Web browser |
| `.json` / `.geojson` | GeoJSON | Vector geometry + attributes | QGIS, web maps |
| `.kml` | KML | Geographic markup | Google Earth |

---

### **Data Dimensions by Resolution**

| Resolution | Pixels | Pixel Size | Coverage | File Size (per frame) |
|------------|--------|------------|----------|----------------------|
| 30m | 1073 × 1073 | 30m × 30m | ~32 km × 32 km | ~4.4 MB (float32) |
| 60m | 536 × 536 | 60m × 60m | ~32 km × 32 km | ~1.1 MB (float32) |

---

### **Temporal Information**

| Data Type | Timesteps | Interval | Duration | Total Files |
|-----------|-----------|----------|----------|-------------|
| Rainfall | Varies | 30 min | Event-dependent | ~100-200 |
| Flood (30m) | 2881 | 30 min | ~60 days | 2881 |
| Flood (60m) | 2881 | 30 min | ~60 days | 2881 |

---

## 💾 Storage Requirements

```
Input Data:
  ├─ DEM:                    ~4.4 MB
  ├─ Rainfall (all):         ~880 MB (200 timesteps × 4.4 MB)
  ├─ Manning:                ~4.4 MB
  ├─ Initial Conditions:     ~4.4 MB
  └─ TOTAL INPUT:           ~900 MB

Output Data:
  ├─ Flood (30m, all):      ~12.7 GB (2881 × 4.4 MB)
  ├─ Flood (60m, all):      ~3.2 GB (2881 × 1.1 MB)
  └─ TOTAL OUTPUT:          ~16 GB

Visualization Products:
  ├─ PNG files (each):       2-4 MB
  ├─ GIF animations:         50-200 MB
  ├─ MP4 videos:             50-150 MB
  ├─ HTML maps:              5-20 MB
  └─ TOTAL VIZ:             ~500 MB - 1 GB
```

---

## 📝 Summary

This document provides a complete reference for:

✅ **All input files** - Location, format, contents, structure  
✅ **What you get** after loading each file  
✅ **All output files** - Generated by visualization scripts  
✅ **Coordinate systems** - UTM vs WGS84, conversion  
✅ **Data dimensions** - Spatial and temporal resolution  
✅ **Usage guide** - How to use each output product  

**Use this as a reference** when:
- Writing data loading code
- Understanding simulation inputs
- Interpreting visualization outputs
- Creating presentations/reports
- Building ML models on this data

---

**Document End** | Version 1.0 | November 2025
