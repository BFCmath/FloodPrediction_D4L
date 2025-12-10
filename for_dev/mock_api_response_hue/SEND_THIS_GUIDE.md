# Hue City Mock API Response Package - Ready to Send

## 📦 Package Location
```
d:\project\d4l\for_dev\mock_api_response_hue\
```

## 📄 Files Included

1. **metadata.json** (1.3 KB) - Complete API response with all info
2. **flood_depth_20251211_004750.tif** (43 MB) - Flood depth GeoTIFF
3. **README.md** - Usage instructions

---

## 🎯 What's in the JSON (metadata.json)

### Complete API Response Structure

```json
{
  "request_id": "hue_mock_20251211_004750",
  "timestamp": "2025-12-11T00:47:50Z",
  "location": "Hue City, Vietnam",
  
  "bounds": {
    "north": 16.746028,
    "south": 15.987548,
    "east": 108.190940,
    "west": 107.019869,
    "center": {
      "lat": 16.366788,
      "lon": 107.605404
    }
  },
  
  "grid": {
    "width": 4109,
    "height": 2681,
    "resolution_meters": 30.73,
    "coordinate_system": "EPSG:32648"
  },
  
  "flood_statistics": {
    "max_depth_meters": 1.0,
    "mean_depth_meters": 1.0,
    "flooded_area_km2": 42.57,
    "flooded_pixels": 45093,
    "flooded_percentage": 0.89
  },
  
  "data": {
    "unit": "meters",
    "format": "GeoTIFF",
    "file_name": "flood_depth_20251211_004750.tif"
  }
}
```

---

## 📊 Data Summary

| Property | Value |
|----------|-------|
| **Location** | Hue City, Vietnam |
| **Center Coordinates** | 16.37°N, 107.61°E |
| **Grid Size** | 4,109 × 2,681 pixels (11M pixels) |
| **Resolution** | 30.73 meters per pixel |
| **Coverage Area** | ~126 km × 82 km |
| **Coordinate System** | UTM Zone 48N (EPSG:32648) |
| **Max Flood Depth** | 1.0 meter |
| **Flooded Area** | 42.57 km² |
| **Flooded Pixels** | 45,093 (0.89% of area) |

---

## 📤 How to Send to Your Colleague

### Option 1: Send Just JSON (Recommended for Quick Testing)
**File**: `metadata.json` (1.3 KB)

This contains all the information needed for:
- Displaying bounds on a map
- Showing flood statistics
- Understanding the data format

**Use case**: If they just need to test API parsing and UI display

### Option 2: Send Complete Package
**Files**: Entire `mock_api_response_hue` folder

This includes:
- metadata.json (API response)
- flood_depth_*.tif (actual flood data)
- README.md (instructions)

**Use case**: If they need to test full workflow including GeoTIFF rendering

### Option 3: Upload to Cloud
```powershell
# Zip the entire package
Compress-Archive -Path "d:\project\d4l\for_dev\mock_api_response_hue\*" -DestinationPath "hue_mock_data.zip"

# Then upload to Google Drive, Dropbox, etc.
```

---

## 🧪 What They Can Test With This Data

### 1. API Response Parsing
```javascript
// Your frontend can parse the JSON
const response = await fetch('/api/prediction');
const data = await response.json();

console.log(`Location: ${data.location}`);
console.log(`Max depth: ${data.flood_statistics.max_depth_meters}m`);
console.log(`Flooded area: ${data.flood_statistics.flooded_area_km2} km²`);
```

### 2. Map Display
```javascript
// Use bounds to center the map
const bounds = data.bounds;
map.fitBounds([
  [bounds.south, bounds.west],
  [bounds.north, bounds.east]
]);

// Show center marker
const marker = L.marker([
  data.bounds.center.lat,
  data.bounds.center.lon
]);
```

### 3. GeoTIFF Visualization
If they have the .tif file, they can:
- Load it in QGIS/ArcGIS
- Convert to PNG overlay for web maps
- Extract values at specific points

---

## 🎨 Visualization Preview

**Map Extent**:
- North: **16.75°N** (near Phong Điền)
- South: **15.99°N** (near A Lưới)
- East: **108.19°E** (near A Lưới)
- West: **107.02°E** (near Phú Lộc)

**Center Point**: 16.37°N, 107.61°E (approximately Hue city center)

**Flooded Areas**: 
- Small pockets (42.57 km² total)
- Max depth: 1.0 meter (low-lying areas near water level 2m)
- Represents ~0.89% of the total area

---

## 🔧 Technical Details

### Coordinate System
- **Name**: UTM Zone 48N
- **EPSG Code**: 32648
- **Projection**: Universal Transverse Mercator
- **Datum**: WGS84
- **Coverage**: All of Vietnam

### Data Format
- **Type**: GeoTIFF (Georeferenced TIFF)
- **Compression**: LZW
- **Data Type**: Float32
- **NoData Value**: -9999.0
- **Unit**: Meters (flood depth)

### Grid Information
- **Pixel Width**: 4,109
- **Pixel Height**: 2,681
- **Total Pixels**: 11,016,229
- **Valid Pixels**: 5,040,772 (rest is NoData/ocean)
- **Pixel Size**: 30.73m × 30.73m

---

## 📋 Sample Code for Loading Data

### Python
```python
import json
import tifffile
import numpy as np

# Load metadata
with open('metadata.json') as f:
    meta = json.load(f)

# Load flood depths
flood = tifffile.imread('flood_depth_20251211_004750.tif')

# Get bounds
print(f"Bounds: {meta['bounds']}")
print(f"Center: {meta['bounds']['center']}")

# Get statistics
valid_data = flood[flood != -9999.0]
print(f"Max depth: {np.max(valid_data):.2f}m")
print(f"Mean depth: {np.mean(valid_data[valid_data > 0]):.2f}m")
```

### JavaScript/Node.js
```javascript
const fs = require('fs');

// Load metadata
const metadata = JSON.parse(fs.readFileSync('metadata.json'));

console.log('Location:', metadata.location);
console.log('Bounds:', metadata.bounds);
console.log('Flooded area:', metadata.flood_statistics.flooded_area_km2, 'km²');
console.log('Max depth:', metadata.flood_statistics.max_depth_meters, 'm');

// For GeoTIFF, use libraries like geotiff.js
```

---

## ✅ Validation Checklist

Before sending, verify:

- [x] metadata.json is valid JSON (no syntax errors)
- [x] All required fields are present
- [x] Bounds make sense for Hue City (16-17°N, 107-108°E)
- [x] GeoTIFF file exists and is readable
- [x] README provides clear instructions
- [x] File sizes are reasonable (JSON: 1.3KB, TIF: 43MB)

---

## 🚀 Quick Test Commands

```powershell
# Verify JSON is valid
Get-Content "d:\project\d4l\for_dev\mock_api_response_hue\metadata.json" | ConvertFrom-Json

# Check file sizes
Get-ChildItem "d:\project\d4l\for_dev\mock_api_response_hue" | Format-Table Name, Length

# View JSON in formatted way
Get-Content "d:\project\d4l\for_dev\mock_api_response_hue\metadata.json" | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

---

## 📞 Support Information

If your colleague has questions:

1. **JSON structure**: Follows standard API response format
2. **Coordinate system**: UTM Zone 48N (same as HCMC example)
3. **Data units**: All depths in meters
4. **Bounds format**: WGS84 (lat/lon in degrees)

---

## 🎁 What You're Sending

> "Mock flood prediction data for Hue City, Vietnam. Includes complete API response JSON with geographic bounds, flood statistics, and a GeoTIFF file with synthetic flood depth data. Ready for testing map visualization and data display."

**File**: `metadata.json` (or entire folder)  
**Size**: 1.3 KB (JSON only) or 43 MB (with GeoTIFF)  
**Format**: Standard API response format  
**Status**: ✅ Ready to send
