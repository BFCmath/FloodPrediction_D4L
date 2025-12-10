# Hue City Mock API Response
Generated: 2025-12-11T00:52:39Z

## Contents

1. **metadata.json** - Complete API response with bounds, stats, and configuration
2. **flood_depth_20251211_005239.tif** - Flood depth prediction (GeoTIFF)

## Quick Info

- **Location**: Hue City, Vietnam
- **Center**: 16.366788°N, 107.605404°E
- **Grid Size**: 4109 × 2681 pixels
- **Resolution**: 30.73 meters
- **Max Flood Depth**: 1.00 meters
- **Flooded Area**: 42.57 km²

## Usage

### Load in Python
```python
import tifffile
import json

# Load metadata
with open('metadata.json') as f:
    meta = json.load(f)

# Load flood depths
flood = tifffile.imread('flood_depth_20251211_005239.tif')

print(f"Bounds: {meta['bounds']}")
print(f"Max depth: {flood.max()} meters")
```

### View in QGIS
1. Open QGIS
2. Drag and drop the .tif file
3. Style with blue color ramp for flood depth

## Data Format

- **Coordinate System**: UTM Zone 48N (EPSG:32648)
- **Values**: Flood depth in meters (Float32)
- **NoData**: -9999.0
- **Valid Range**: 0 to 1.00 meters

## Simulation Method

This is a simple "bathtub model" simulation:
- Water level set at: 2.0m
- Flood depth = max(0, Water Level - Ground Elevation)
- Areas with DEM ≤ 0 are masked as NoData

This is for testing/demo purposes only. Real flood predictions use
physics-based models considering rainfall, drainage, and flow dynamics.
