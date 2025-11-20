# 🌊 FloodCastBench Experiments - Complete Setup

## ✅ What I Created For You

I've created **6 Python scripts** + supporting files in the `experiments/` folder to help you visualize and understand the FloodCastBench dataset:

```
d:\project\d4l\experiments\
├── 01_visualize_dem.py                    ← DEM terrain visualization
├── 02_visualize_rainfall.py               ← Rainfall time series
├── 03_visualize_manning.py                ← Land use/roughness
├── 04_visualize_initial_conditions.py     ← Initial water depth
├── 05_visualize_flood_output.py           ← Flood simulation results
├── 06_convert_to_google_maps.py           ← Google Maps conversion ⭐
├── README.md                              ← Detailed documentation
├── requirements.txt                       ← Package dependencies
├── run_all_visualizations.bat             ← Run all scripts at once
└── outputs/                               ← Output folder (auto-created)
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Required Packages
```bash
cd d:\project\d4l\experiments
pip install -r requirements.txt
```

### Step 2: Run All Visualizations
**Option A - Windows Batch File (Easiest):**
```bash
run_all_visualizations.bat
```

**Option B - Run Individually:**
```bash
python 01_visualize_dem.py
python 02_visualize_rainfall.py
python 03_visualize_manning.py
python 04_visualize_initial_conditions.py
python 05_visualize_flood_output.py
python 06_convert_to_google_maps.py
```

### Step 3: View Results
All outputs saved in `outputs/` folder:
- **PNG images** - Open in any image viewer
- **06_google_map_interactive.html** - Open in web browser 🌐
- **06_dem_coverage.kml** - Open in Google Earth 🌍

---

## 📊 Script Overview

| # | Script | What It Does | Key Output |
|---|--------|--------------|------------|
| 1 | **DEM Visualization** | Terrain elevation, hillshade, slopes, contours | 9 different elevation views |
| 2 | **Rainfall Visualization** | Precipitation time series, cumulative rain, evolution | Temporal rainfall analysis |
| 3 | **Manning Visualization** | Land cover, roughness coefficient, flow resistance | Land classification maps |
| 4 | **Initial Conditions** | Starting water depth, wet/dry areas | Initial state analysis |
| 5 | **Flood Output** | Simulated flood evolution, maximum extent, arrival times | Complete flood dynamics |
| 6 | **Google Maps Conversion** ⭐ | Convert to lat/lon, create interactive map | HTML map + KML + GeoJSON |

---

## 🗺️ Google Maps Integration (Script 06)

### What You Get:

**1. Interactive HTML Map** (`06_google_map_interactive.html`)
- DEM boundary overlay on Google Maps
- Switch between Street/Terrain/Satellite views
- Corner markers showing exact coordinates
- Mouse position tracker
- Measurement tools
- Fullscreen mode

**2. Google Earth KML** (`06_dem_coverage.kml`)
- Import into Google Earth
- 3D visualization of DEM coverage
- Metadata popup with elevation info

**3. GeoJSON** (`06_dem_coverage.geojson`)
- Use in web mapping frameworks (Mapbox, Leaflet)
- Import into GIS software (QGIS, ArcGIS)

### How It Works:

```
DEM GeoTIFF (.tif)
    ↓
Read .tfw world file
    ↓
Extract UTM coordinates (EPSG:32756 - Australia)
    ↓
Convert to WGS84 (Lat/Lon)
    ↓
Generate Google Maps overlay
```

**Coordinates Conversion:**
- **Input:** UTM Zone 56S (meters) - e.g., (500000, -3000000)
- **Output:** WGS84 Lat/Lon (degrees) - e.g., (-27.5°, 153.0°)

---

## 📦 Package Requirements

### Essential (Scripts 1-5)
```
numpy
matplotlib
tifffile
scipy
```

### Additional for Google Maps (Script 6)
```
pyproj      ← Coordinate transformations
folium      ← Interactive web maps
rasterio    ← GeoTIFF metadata reading
```

**Install all at once:**
```bash
pip install -r requirements.txt
```

---

## 🎯 Example Output Files

After running all scripts, you'll have:

```
outputs/
├── 01_dem_visualization.png                 (2-3 MB)
│   └─ Contains: Elevation map, hillshade, contours, slopes, statistics
│
├── 02_rainfall_visualization.png            (2-3 MB)
│   └─ Contains: Rain intensity at different times, cumulative rain, time series
│
├── 03_manning_visualization.png             (2-3 MB)
│   └─ Contains: Roughness map, land classification, flow resistance zones
│
├── 04_initial_conditions_visualization.png  (2-3 MB)
│   └─ Contains: Initial water depth, wet/dry classification, profiles
│
├── 05_flood_output_30m_visualization.png    (3-4 MB)
│   └─ Contains: Flood evolution, max extent, arrival times, statistics
│
├── 06_google_map_interactive.html           (10-20 KB) ⭐
│   └─ Interactive map with Google Satellite basemap
│
├── 06_dem_coverage.kml                      (2-5 KB) 🌍
│   └─ Google Earth file with DEM boundary
│
└── 06_dem_coverage.geojson                  (2-5 KB)
    └─ GeoJSON polygon for web mapping
```

---

## 🔧 Troubleshooting

### Issue: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "DEM file not found"
**Solution:**
Edit the `DATA_ROOT` variable in each script:
```python
DATA_ROOT = Path("..") / "FloodCastBench_Dataset-and-Models-main" / "Data_Generation_Code" / "FloodCastBench"
```

### Issue: Wrong location on Google Maps
**Solution:**
Update `DEFAULT_EPSG` in `06_convert_to_google_maps.py`:
- 32755: UTM Zone 55S (Sydney)
- 32756: UTM Zone 56S (Brisbane) ← Default
- 32754: UTM Zone 54S (Perth)

### Issue: Plots don't show up
**Solution:**
- They're saved automatically in `outputs/` folder
- Check for PNG files there
- Scripts use `plt.show()` which may be blocked in some environments

---

## 💡 Pro Tips

1. **Run in sequence** - Scripts 1-6 build understanding progressively
2. **Check console output** - Statistics are printed during execution
3. **Customize colormaps** - Edit `cmap='...'` parameters for different colors
4. **High-res exports** - Change `dpi=150` to `dpi=300` for publication quality
5. **Memory issues?** - Script 05 limits to 100 timesteps (adjust `max_files`)

---

## 📚 Additional Resources

In your main project folder (`d:\project\d4l\`), you also have:

- `FloodCastBench_Data_Structure_Guide.md` - Complete data explanation
- `FloodCastBench_Cheat_Sheet.md` - Quick reference
- `FloodCastBench_Visual_Summary.md` - Data flow diagrams
- `explore_data.py` - Alternative exploration script

---

## 🎓 What Each Script Teaches You

| Script | Key Concept |
|--------|-------------|
| **01** | Understanding terrain topology and its role in flood flow |
| **02** | Temporal dynamics of precipitation forcing |
| **03** | Surface friction and its impact on flow velocity |
| **04** | Initial conditions and their influence on simulation |
| **05** | Flood propagation patterns and inundation dynamics |
| **06** | Geospatial referencing for real-world applications |

---

## ✅ Checklist

Before running scripts:
- [ ] Data exists at: `d:\project\d4l\FloodCastBench_Dataset-and-Models-main\`
- [ ] Python installed (3.7+)
- [ ] Packages installed: `pip install -r requirements.txt`

After running scripts:
- [ ] Check `outputs/` folder for PNG files
- [ ] Open `06_google_map_interactive.html` in browser
- [ ] Try opening `06_dem_coverage.kml` in Google Earth
- [ ] Review console statistics printed by each script

---

## 🌟 Next Steps

After visualizing the data:

1. **Understand the physics:**
   - How does terrain slope affect flood propagation?
   - Why do certain areas flood first?
   - What's the relationship between rainfall and flood extent?

2. **Prepare for ML modeling:**
   - What features are most important? (DEM, Manning, rainfall)
   - What's the prediction target? (future flood depth)
   - How to handle temporal sequences?

3. **Build your first model:**
   - Start with simple LSTM for time series prediction
   - Try ConvLSTM for spatial-temporal modeling
   - Experiment with downscaling (60m → 30m)

---

## 📞 Support

If you encounter issues:

1. Check the `README.md` in this folder for detailed documentation
2. Review error messages in console output
3. Verify data paths in each script's configuration section
4. Ensure all packages are installed: `pip install -r requirements.txt`

---

**🎉 You're All Set! Start Exploring! 🌊📊🚀**

Run the batch file or individual scripts to begin visualizing the FloodCastBench dataset!

---

*Created: November 2025*
*Project: FloodCastBench Data Visualization Suite*
