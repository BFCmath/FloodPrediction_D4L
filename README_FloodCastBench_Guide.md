# 🌊 FloodCastBench Dataset Understanding Guide

This folder contains comprehensive documentation and tools to help you understand the **FloodCastBench** dataset structure and contents.

## 📚 Documentation Files

### 1. **FloodCastBench_Data_Structure_Guide.md** 
**→ START HERE! Comprehensive guide**

The complete reference guide that explains:
- ✅ Detailed breakdown of each folder
- ✅ Purpose and content of every file type
- ✅ How the data flows through the simulation
- ✅ Physics equations behind the data
- ✅ File formats and naming conventions
- ✅ Use cases for machine learning

**Best for:** Understanding the entire dataset structure in depth

---

### 2. **FloodCastBench_Visual_Summary.md**
**→ Visual diagrams and flowcharts**

Visual representation of:
- 📊 Complete data pipeline with ASCII art
- 📊 Spatial and temporal dimensions
- 📊 Typical data value ranges
- 📊 ML application patterns
- 📊 Physics equations explained
- 📊 Storage and performance metrics

**Best for:** Visual learners who prefer diagrams

---

### 3. **FloodCastBench_Cheat_Sheet.md**
**→ Quick reference guide**

One-page reference with:
- 🎯 Folder structure at a glance
- 🎯 Key parameters table
- 🎯 File naming conventions
- 🎯 Python code snippets
- 🎯 Common issues & solutions
- 🎯 Performance tips

**Best for:** Quick lookup when coding

---

## 🐍 Python Tools

### **explore_data.py**
**→ Interactive data exploration script**

Features:
- 📁 Automatically scans folder structure
- 📊 Prints summary statistics for all data types
- 🎨 Creates visualizations of DEM, rainfall, and flood maps
- 📈 Analyzes time series evolution
- 💾 Saves plots as PNG files

**Usage:**
```bash
python explore_data.py
```

**Outputs:**
- `FloodCastBench_Visualization.png` - Overview of all data types
- `FloodCastBench_TimeSeries.png` - Flood evolution over time

---

## 🚀 Quick Start Guide

### Step 1: Read the Documentation
```
1. FloodCastBench_Data_Structure_Guide.md  ← Read this first (15 min)
2. FloodCastBench_Visual_Summary.md        ← Browse the diagrams (10 min)
3. FloodCastBench_Cheat_Sheet.md           ← Keep for reference
```

### Step 2: Explore the Data
```bash
# Install required packages
pip install tifffile numpy matplotlib imageio

# Run the exploration script
python explore_data.py
```

### Step 3: Review Generated Visualizations
```
- Open FloodCastBench_Visualization.png
- Open FloodCastBench_TimeSeries.png
- Compare with the documentation
```

### Step 4: Start Working with the Data
```python
import tifffile
import numpy as np

# Example: Load and inspect DEM
dem = tifffile.imread('FloodCastBench_Dataset-and-Models-main/Data_Generation_Code/FloodCastBench/Relevant_data/DEM/Australia_DEM.tif')
print(f"DEM shape: {dem.shape}")
print(f"Elevation range: {dem.min():.1f}m to {dem.max():.1f}m")
```

---

## 📂 FloodCastBench Folder Structure Summary

```
FloodCastBench/
│
├── 📊 High-fidelity_flood_forecasting/  ← OUTPUT (Generated flood maps)
│   ├── 30m/Australia/   → Detailed simulation results
│   └── 60m/Australia/   → Coarser simulation results
│
└── 📥 Relevant_data/                    ← INPUT (Source data)
    ├── DEM/              → Terrain elevation (topography)
    ├── Rainfall/         → Precipitation time series
    ├── Land_use_and_land_cover/  → Surface roughness (Manning)
    ├── Initial_conditions/        → Starting water depth
    └── Georeferenced_files/       → GPS coordinates
```

---

## 🎯 What Each Data Type Does

| Data Type | Folder | Purpose | Example File |
|-----------|--------|---------|--------------|
| **Terrain Elevation** | `DEM/` | Gravity-driven flow direction | `Australia_DEM.tif` |
| **Rainfall** | `Rainfall/` | Water input source | `20220220-S193000.tif` |
| **Land Roughness** | `Land_use.../` | Flow resistance (friction) | `Australia.tif` |
| **Initial Water** | `Initial_conditions/` | Starting condition (t=0) | `Australia_30m.tif` |
| **Simulated Flood** | `High-fidelity.../30m/` | ML training data (output) | `100200.tif` |

---

## 📖 Key Concepts Explained

### Physics-Based Simulation
FloodCastBench uses the **Saint-Venant equations** (2D shallow water equations) to simulate flood propagation. Think of it as solving:

```
Water conservation:  
   ∂h/∂t = Rain - Outflow

Momentum conservation:
   Flow is driven by gravity (terrain slope)
   Flow is slowed by friction (Manning coefficient)
```

### Multi-Resolution Data
The dataset includes **three resolutions**:
- **30m**: High detail, slow to compute (~8-12 hours per simulation)
- **60m**: Medium detail, faster (~2-3 hours per simulation)
- **480m**: Coarse, very fast (for low-fidelity modeling)

### Time Series Structure
Each simulation runs for **6 days** (518,400 seconds):
- **Input rainfall** updates every **30 minutes** (1800 seconds)
- **Output flood maps** saved every **30 seconds**
- Result: ~17,000 timesteps per simulation!

---

## 🎓 Machine Learning Applications

### 1. Flood Forecasting
**Goal:** Predict future flood maps from past observations

```python
# Input: h(t-2), h(t-1), h(t), DEM, Future_Rain
# Output: h(t+1), h(t+2), h(t+3), ...
# Models: LSTM, ConvLSTM, Transformer
```

### 2. Spatial Downscaling
**Goal:** Enhance resolution from coarse to fine

```python
# Input: 60m resolution flood map
# Output: 30m resolution flood map
# Models: Super-resolution CNN, ESRGAN
```

### 3. Cross-Region Transfer
**Goal:** Train on one region, test on another

```python
# Train: Australia floods
# Test: UK / Pakistan / Mozambique floods
# Approach: Domain adaptation, meta-learning
```

---

## 🔧 Common Tasks

### Load and Visualize DEM
```python
import tifffile
import matplotlib.pyplot as plt

dem = tifffile.imread('path/to/Australia_DEM.tif')
plt.imshow(dem, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title('Digital Elevation Model')
plt.show()
```

### Create Flood Animation
```python
import glob
import imageio

# Load all flood depth files
files = sorted(glob.glob('High-fidelity_flood_forecasting/30m/Australia/*.tif'))
frames = [tifffile.imread(f) for f in files[::10]]  # Every 10th frame

# Save as GIF
imageio.mimsave('flood_evolution.gif', frames, fps=5)
```

### Extract Time Series at a Point
```python
import numpy as np

# Define location (pixel coordinates)
x, y = 150, 200

# Extract depth at this location over time
depths = []
for file in sorted(files):
    flood = tifffile.imread(file)
    depths.append(flood[y, x])

# Plot
plt.plot(depths)
plt.xlabel('Timestep')
plt.ylabel('Water Depth (m)')
plt.title(f'Flood Depth at ({x}, {y})')
plt.show()
```

---

## ❓ Frequently Asked Questions

### Q: How big is the dataset?
**A:** For a single 6-day simulation at 30m resolution: ~8-10 GB. The full dataset (multiple regions and resolutions) can be 50-100 GB.

### Q: What software do I need?
**A:** 
- Python 3.7+
- Libraries: `tifffile`, `numpy`, `matplotlib`, `torch` (for ML)
- Optional: GDAL for advanced GIS operations

### Q: Can I run the simulation myself?
**A:** Yes! Use `main.py` in the `Data_Generation_Code` folder. You'll need:
- GPU (NVIDIA recommended)
- 16+ GB RAM
- 8-12 hours for 30m resolution

### Q: What's the difference between High-fidelity and Low-fidelity?
**A:** 
- **High-fidelity** (30m/60m): Detailed, computationally expensive
- **Low-fidelity** (480m): Coarse, fast, good for training surrogate models

### Q: Where can I find more flood events?
**A:** The current dataset includes Australia, UK, Pakistan, and Mozambique. Check the README for updates on new regions.

---

## 📞 Support & Contact

- **Questions about the dataset:** qingsong(at)tum.de
- **GitHub Issues:** [FloodCastBench Repository](https://github.com/HydroPML/FloodCastBench)
- **Documentation Issues:** Check this folder first, then open a GitHub issue

---

## 🎉 You're Ready!

You now have:
- ✅ Comprehensive documentation (3 markdown files)
- ✅ Interactive exploration script (`explore_data.py`)
- ✅ Understanding of the data structure
- ✅ Code examples to get started

**Next Steps:**
1. Run `python explore_data.py` to see your data
2. Review the generated visualizations
3. Start building your ML models!

---

**Happy Flood Modeling! 🌊📊🚀**

*Last Updated: November 2025*
