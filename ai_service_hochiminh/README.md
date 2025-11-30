# AI Flood Forecasting Service - Ho Chi Minh City

## Overview

This service provides real-time flood depth predictions for Ho Chi Minh City using a ConvLSTM neural network trained on high-fidelity flood simulation data.

### Components

1. **AI Service (`main.py`)**: Core prediction engine that:
   - Maintains rolling buffers of rainfall and flood depth history
   - Runs the ConvLSTM model for inference
   - Stores predictions as GeoTIFF files

2. **Stream Service (`stream_service.py`)**: Data ingestion that:
   - Reads rainfall TIF files from `rainfall/` directory
   - Sends file paths to the AI Service API at configurable intervals

3. **Database (`database.py`)**: Storage layer for:
   - Rainfall file metadata and copies
   - Prediction GeoTIFF files with georeferencing

---

## Training Data Pattern (Critical!)

The model was trained with specific temporal alignment that **must be matched** during inference:

| Data Type | Interval | Training Pattern |
|-----------|----------|------------------|
| **Flood Depth** | 5 minutes | High-fidelity simulation output |
| **Rainfall** | 30 minutes | **Same rain repeated 6× to match 5-min depth** |
| **Input Buffer** | 12 frames | 12 × 5 min = **60 minutes of history** |
| **Output** | 6 frames | 6 × 5 min = **30 minutes prediction** |

### Training Data Structure
```
For each training sample:
  Rainfall:  [R₀, R₀, R₀, R₀, R₀, R₀, R₁, R₁, R₁, R₁, R₁, R₁]
              └──── 30 min (same) ────┘  └──── 30 min (same) ────┘
  
  Depth:     [D₀, D₁, D₂, D₃, D₄, D₅, D₆, D₇, D₈, D₉, D₁₀, D₁₁]
              └──────────────── 60 minutes ────────────────────┘
```

---

## Inference Flow (Training-Matched)

### Cold Start (Service Initialization)
```
buffer_rain  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 12× zero (no rain history)
buffer_depth = [D₀,D₀,D₀,D₀,D₀,D₀,D₀,D₀,D₀,D₀,D₀,D₀] # 12× initial condition
```

### When ONE Rainfall File Arrives (30-min interval)

**Step 1: Update Rain Buffer (6 copies)**
```python
# Repeat the same rain 6 times (matching training pattern)
for i in range(6):
    buffer_rain.pop(0)
    buffer_rain.append(new_rain)
```

**Step 2: Run 6 Inference Steps**
```python
for step in range(6):  # Simulate 5, 10, 15, 20, 25, 30 min
    # Build dynamic input from buffers
    dynamic_input = combine(buffer_rain, buffer_depth)  # (1, 12, 2, H, W)
    
    # Run model
    predictions = model(dynamic_input, static_input, future_steps=6)
    
    # Take 5-min prediction, update depth buffer
    buffer_depth.pop(0)
    buffer_depth.append(predictions[0])  # 5-min prediction
```

**Step 3: Save 30-min Prediction**
```python
# The 6th step's prediction = 30 minutes from start
save_prediction(predictions[-1])
```

### Buffer Evolution Example

```
Initial:     rain = [0,0,0,0,0,0,0,0,0,0,0,0]    depth = [D₀,D₀,D₀,D₀,D₀,D₀,D₀,D₀,D₀,D₀,D₀,D₀]

After R₁:    rain = [0,0,0,0,0,0,R₁,R₁,R₁,R₁,R₁,R₁]  depth = [D₀,D₀,D₀,D₀,D₀,D₀,P₁,P₂,P₃,P₄,P₅,P₆]

After R₂:    rain = [R₁,R₁,R₁,R₁,R₁,R₁,R₂,R₂,R₂,R₂,R₂,R₂]  depth = [P₁,P₂,P₃,P₄,P₅,P₆,P₇,P₈,P₉,P₁₀,P₁₁,P₁₂]
```

---

## Configuration

Key constants in `main.py`:

```python
# Timing (matching training)
RAIN_INTERVAL_MINUTES = 30   # Rainfall file interval
DEPTH_INTERVAL_MINUTES = 5   # Model's internal resolution
STEPS_PER_RAIN = 6           # 30 / 5 = 6 inferences per rainfall
BUFFER_SIZE = 12             # 12 frames × 5 min = 60 min history

# Normalization (matching training)
RAIN_NORM_FACTOR = 50.0      # Divide rainfall by 50
DEPTH_NORM_FACTOR = 5.0      # Divide/multiply depth by 5

# Performance
USE_DOWNSAMPLED = True/False # Auto-detect: use _small.tif on CPU
FUTURE_STEPS = 6             # Predict 6 steps per inference
# All 6 horizons are saved: 5, 10, 15, 20, 25, 30 minutes
```

---

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Static Data

Place these files in `static_data/`:
- `HoChiMinh_DEM.tif` - Digital Elevation Model
- `HoChiMinh_Manning.tif` - Manning roughness coefficients  
- `HoChiMinh_initial_condition.tif` - Initial water depth

For CPU inference, also create downsampled versions:
```bash
python downsample_data.py
```
This creates `*_small.tif` versions (~1/4 resolution) for faster CPU inference.

### 3. Prepare Rainfall Data

Place rainfall TIF files in `rainfall/`:
- Format: `YYYYMMDD-SHHMMSS.tif` (e.g., `20250628-S000000.tif`)
- Interval: 30 minutes between files

### 4. Model Weights

Place trained model weights at `weights/best_model.pth`

---

## Running

### 1. Start AI Service
```bash
python main.py
```
- Runs on http://localhost:5000
- Loads model, static data, and initializes buffers
- Shows detailed timing logs for each operation

### 2. Start Data Stream
```bash
python stream_service.py
```
- Reads rainfall files in chronological order
- Sends file paths to AI Service at configured intervals
- Default: 30-second intervals between files

---

## API Endpoints

### POST /receive_rainfall
Receive new rainfall data and generate **6 predictions** (5, 10, 15, 20, 25, 30 minutes).

**Request:**
```json
{
  "file_path": "/path/to/rainfall.tif",
  "filename": "20250628-S000000.tif",
  "timestamp": "20250628_000000",
  "file_size_bytes": 51234
}
```

**Response:**
```json
{
  "status": "success",
  "predictions": {
    "5": "/path/to/db_storage/pred_5min_20250628_000100.tif",
    "10": "/path/to/db_storage/pred_10min_20250628_000100.tif",
    "15": "/path/to/db_storage/pred_15min_20250628_000100.tif",
    "20": "/path/to/db_storage/pred_20min_20250628_000100.tif",
    "25": "/path/to/db_storage/pred_25min_20250628_000100.tif",
    "30": "/path/to/db_storage/pred_30min_20250628_000100.tif"
  }
}
```

### GET /prediction?time_horizon=N
Get latest prediction metadata and file path.
- **Valid horizons**: 5, 10, 15, 20, 25, 30 minutes

**Response:**
```json
{
  "metadata": {
    "request_id": "req_20250628_000100",
    "timestamp": "20250628_000100",
    "horizon_minutes": 30,
    "bounds": {"north": 10.88, "south": 10.65, "east": 106.88, "west": 106.55},
    "grid": {"width": 2811, "height": 2390},
    "data_stats": {"max_depth_meters": 1.23, "flooded_area_pixels": 45678}
  },
  "file_path": "/path/to/prediction.tif"
}
```

### GET /predictions_all
Get all 6 predictions at once.

**Response:**
```json
{
  "available_horizons": [5, 10, 15, 20, 25, 30],
  "predictions": {
    "5": {"file_path": "...", "timestamp": "...", "exists": true},
    "10": {"file_path": "...", "timestamp": "...", "exists": true},
    ...
  }
}
```

### GET /prediction_path?time_horizon=N
Get file path only (lightweight). Valid horizons: 5, 10, 15, 20, 25, 30.

### GET /download_prediction?time_horizon=N
Download the GeoTIFF file directly. Valid horizons: 5, 10, 15, 20, 25, 30.

### GET /rainfall_latest
Get info about most recent rainfall file.

---

## Storage

- **SQLite**: `ai_service_db.sqlite` - Metadata only
- **Files**: `db_storage/` - Rainfall and prediction GeoTIFFs
  - Predictions: `pred_{horizon}min_{timestamp}.tif`
  - Rainfall: `rainfall_{timestamp}.tif`

---

## Model Architecture

```
Input:
  dynamic_input: (1, 12, 2, H, W)  → [Rain, Depth] × 12 timesteps
  static_input:  (1, 2, H, W)      → [DEM, Manning]

Model:
  ConvLSTM Encoder → Process 12 input frames
  ConvLSTM Decoder → Generate 6 future frames (autoregressive)

Output:
  predictions: (1, 6, 1, H, W)  → Depth for next 5, 10, 15, 20, 25, 30 min
```

---

## Performance (CPU)

With downsampled data (~600×700 pixels):
- Model load: ~2s
- Per inference: ~500-800ms
- 6 inferences per rainfall: ~3-5s total
- GeoTIFF save: ~100ms

---

## Troubleshooting

### Out of Memory
- Enable `USE_DOWNSAMPLED = True` in main.py
- Run `python downsample_data.py` first

### Predictions All Zeros
- Check rainfall normalization: values should be 0-50 mm/hr
- Verify `RAIN_NORM_FACTOR = 50.0`

### Shape Mismatch Errors
- Ensure all static data has same resolution
- Rainfall is auto-resized to match DEM

### NaN/Inf in Predictions
- Check for invalid values in input data
- Verify model weights are correctly loaded
