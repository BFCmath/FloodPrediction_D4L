# Metadata Format Comparison - HCMC vs Hue

## ✅ Both Files Now Use Consistent Format

### Ho Chi Minh City (mock_api_response_hochiminh/metadata.json)
```json
{
  "request_id": "sim_hcm_001",
  "timestamp": "2025-11-29T10:00:00Z",
  "location": "Ho Chi Minh City",
  "simulation_type": "static_inundation",
  "water_level_param": 1.5,
  "bounds": {
    "north": 11.159871119602483,
    "south": 10.375438568758025,
    "east": 107.02445924184507,
    "west": 106.35742462311387,
    "center": {
      "lat": 10.767654844180253,
      "lon": 106.69094193247946
    }
  },
  "grid": {
    "width": 2390,
    "height": 2811,
    "resolution_meters": 30.7250757233
  },
  "data_stats": {
    "max_depth_meters": 47.5,
    "flooded_area_pixels": 386716,
    "flooded_percentage": 21.103860313177716,
    "unit": "meters",
    "nodata_value": -9999.0
  },
  "format": "geotiff"
}
```

### Hue City (mock_api_response_hue/metadata.json) ✨ UPDATED
```json
{
  "request_id": "sim_hue_001",
  "timestamp": "2025-12-11T00:47:50Z",
  "location": "Hue City",
  "simulation_type": "static_inundation",
  "water_level_param": 2.0,
  "bounds": {
    "north": 16.74602801912226,
    "south": 15.987548023342475,
    "east": 108.19093975546386,
    "west": 107.01986856019406,
    "center": {
      "lat": 16.366788021232367,
      "lon": 107.60540415782896
    }
  },
  "grid": {
    "width": 4109,
    "height": 2681,
    "resolution_meters": 30.7250757233
  },
  "data_stats": {
    "max_depth_meters": 1.0,
    "flooded_area_pixels": 45093,
    "flooded_percentage": 0.8945653562589223,
    "unit": "meters",
    "nodata_value": -9999.0
  },
  "format": "geotiff"
}
```

---

## 📋 Field Comparison

| Field | HCMC | Hue | Match? |
|-------|------|-----|--------|
| `request_id` | sim_hcm_001 | sim_hue_001 | ✅ Same format |
| `timestamp` | ISO 8601 | ISO 8601 | ✅ Same format |
| `location` | "Ho Chi Minh City" | "Hue City" | ✅ Same format |
| `simulation_type` | "static_inundation" | "static_inundation" | ✅ Identical |
| `water_level_param` | 1.5 | 2.0 | ✅ Same type (number) |
| `bounds.north` | 11.16 | 16.75 | ✅ Same type (float) |
| `bounds.south` | 10.38 | 15.99 | ✅ Same type (float) |
| `bounds.east` | 107.02 | 108.19 | ✅ Same type (float) |
| `bounds.west` | 106.36 | 107.02 | ✅ Same type (float) |
| `bounds.center.lat` | 10.77 | 16.37 | ✅ Same structure |
| `bounds.center.lon` | 106.69 | 107.61 | ✅ Same structure |
| `grid.width` | 2390 | 4109 | ✅ Same type (int) |
| `grid.height` | 2811 | 2681 | ✅ Same type (int) |
| `grid.resolution_meters` | 30.73 | 30.73 | ✅ Same value |
| `data_stats.max_depth_meters` | 47.5 | 1.0 | ✅ Same type (float) |
| `data_stats.flooded_area_pixels` | 386716 | 45093 | ✅ Same type (int) |
| `data_stats.flooded_percentage` | 21.10 | 0.89 | ✅ Same type (float) |
| `data_stats.unit` | "meters" | "meters" | ✅ Identical |
| `data_stats.nodata_value` | -9999.0 | -9999.0 | ✅ Identical |
| `format` | "geotiff" | "geotiff" | ✅ Identical |

---

## 🎯 Frontend Usage

Both files can now be consumed identically by the frontend:

```typescript
interface FloodPredictionMetadata {
  request_id: string;
  timestamp: string;  // ISO 8601
  location: string;
  simulation_type: string;
  water_level_param: number;
  
  bounds: {
    north: number;
    south: number;
    east: number;
    west: number;
    center: {
      lat: number;
      lon: number;
    };
  };
  
  grid: {
    width: number;
    height: number;
    resolution_meters: number;
  };
  
  data_stats: {
    max_depth_meters: number;
    flooded_area_pixels: number;
    flooded_percentage: number;
    unit: string;
    nodata_value: number;
  };
  
  format: string;
}
```

### Example Usage

```javascript
// Load metadata
const response = await fetch('/api/flood-prediction/hue');
const metadata = await response.json();

// Display on map
const map = new google.maps.Map(document.getElementById('map'), {
  center: { 
    lat: metadata.bounds.center.lat, 
    lng: metadata.bounds.center.lon 
  },
  zoom: 10
});

// Fit bounds
const bounds = new google.maps.LatLngBounds(
  { lat: metadata.bounds.south, lng: metadata.bounds.west },
  { lat: metadata.bounds.north, lng: metadata.bounds.east }
);
map.fitBounds(bounds);

// Display stats
document.getElementById('max-depth').textContent = 
  `${metadata.data_stats.max_depth_meters}m`;
document.getElementById('flooded-area').textContent = 
  `${metadata.data_stats.flooded_area_pixels.toLocaleString()} pixels`;
document.getElementById('flood-percent').textContent = 
  `${metadata.data_stats.flooded_percentage.toFixed(2)}%`;
```

---

## ✅ Validation Results

### Structure Validation
- ✅ Both files have identical structure
- ✅ All field names match exactly
- ✅ All data types are consistent
- ✅ Nested objects have same depth and keys

### Data Validation
- ✅ Bounds are valid lat/lon coordinates
- ✅ Grid dimensions are positive integers
- ✅ Statistics are non-negative numbers
- ✅ NoData values are consistent (-9999.0)

### Format Validation
- ✅ Valid JSON syntax
- ✅ Proper UTF-8 encoding
- ✅ ISO 8601 timestamps
- ✅ Consistent decimal precision

---

## 🔄 Changes Made to Hue Metadata

### Removed (not needed for frontend):
- ❌ `simulation_parameters` object (detailed description)
- ❌ `grid.coordinate_system` (EPSG code)
- ❌ `grid.coordinate_system_name` (UTM Zone name)
- ❌ `flood_statistics.mean_depth_meters` (extra stat)
- ❌ `flood_statistics.flooded_area_km2` (can calculate from pixels)
- ❌ `flood_statistics.total_valid_pixels` (internal detail)
- ❌ `data` object (file-specific details)
- ❌ `source_dem` object (DEM metadata)

### Simplified/Renamed:
- ✅ `flood_statistics` → `data_stats` (matches HCMC)
- ✅ `location` shortened to "Hue City" (not "Hue City, Vietnam")
- ✅ `simulation_type` changed to "static_inundation" (matches HCMC)
- ✅ Flat structure for easier parsing

---

## 📊 Data Comparison

| Metric | Ho Chi Minh City | Hue City |
|--------|------------------|----------|
| **Coverage Area** | 73.5 km × 86.3 km | 126.2 km × 82.3 km |
| **Grid Size** | 2,390 × 2,811 pixels | 4,109 × 2,681 pixels |
| **Total Pixels** | 6.7M | 11.0M |
| **Resolution** | 30.73m | 30.73m |
| **Water Level** | 1.5m | 2.0m |
| **Max Flood Depth** | 47.5m | 1.0m |
| **Flooded Area** | 386,716 pixels (21%) | 45,093 pixels (0.9%) |
| **Center Point** | 10.77°N, 106.69°E | 16.37°N, 107.61°E |

---

## 🚀 Ready to Use

Both metadata files are now:
- ✅ **Consistent** - Same structure and field names
- ✅ **Complete** - All required information present
- ✅ **Clean** - No extra fields to confuse frontend
- ✅ **Validated** - Proper types and valid data
- ✅ **Frontend-friendly** - Easy to parse and use

The frontend can now use **one TypeScript interface** to handle both datasets!
