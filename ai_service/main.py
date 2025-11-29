import os
import json
import base64
import io
import numpy as np
import torch
import rasterio
from flask import Flask, request, jsonify, send_file
from pathlib import Path
from datetime import datetime
from model import FloodForecastModel
from database import Database

app = Flask(__name__)

# --- CONFIG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path("weights/best_model.pth")
STATIC_DIR = Path("static_data")
OUTPUT_DIR = Path("output_predictions")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- GLOBAL STATE ---
class AIService:
    def __init__(self):
        self.db = Database()
        self.model = None
        self.static_input = None # Tensor (1, 2, H, W)
        self.buffer_rain = [] # List of tensors (1, 1, H, W)
        self.buffer_depth = [] # List of tensors (1, 1, H, W)
        self.dem_meta = None # rasterio profile
        self.bounds = None # WGS84 bounds
        
        self.load_resources()
    
    def load_resources(self):
        print("Loading resources...")
        # 1. Load Model
        self.model = FloodForecastModel(input_channels=4, hidden_channels=64, kernel_size=3).to(DEVICE)
        if MODEL_PATH.exists():
            try:
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                self.model.load_state_dict(state_dict)
                print("Model weights loaded.")
            except Exception as e:
                print(f"Warning: Could not load weights ({e}). Using random weights for MVP.")
        else:
            print("Warning: Model weights not found. Using random weights for MVP.")
        self.model.eval()

        # 2. Load Static Data
        dem_path = STATIC_DIR / "Australia_DEM.tif"
        manning_path = STATIC_DIR / "Australia_Manning.tif"
        init_cond_path = STATIC_DIR / "Australia_initial_condition.tif"
        
        # Check files
        if not dem_path.exists(): raise FileNotFoundError(f"Missing {dem_path}")
        
        # Load DEM
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            self.dem_meta = src.profile
            self.dem_bounds = src.bounds
            # Simple Normalize
            dem_norm = (dem - np.min(dem)) / (np.max(dem) - np.min(dem) + 1e-6)
        
        # Load Manning (if missing, use ones)
        if manning_path.exists():
            with rasterio.open(manning_path) as src:
                man = src.read(1)
        else:
            man = np.ones_like(dem) * 0.03

        # Static Tensor
        dem_t = torch.from_numpy(dem_norm).float().unsqueeze(0).unsqueeze(0)
        man_t = torch.from_numpy(man).float().unsqueeze(0).unsqueeze(0)
        self.static_input = torch.cat([dem_t, man_t], dim=1).to(DEVICE)

        # 3. Initialize Buffers (12 frames)
        # Initial condition
        if init_cond_path.exists():
             with rasterio.open(init_cond_path) as src:
                init_depth = src.read(1)
                init_depth = np.maximum(np.nan_to_num(init_depth), 0) / 5.0 # Norm
        else:
            init_depth = np.zeros_like(dem)

        init_depth_t = torch.from_numpy(init_depth).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        zero_rain_t = torch.zeros_like(init_depth_t).to(DEVICE)
        
        for _ in range(12):
            self.buffer_rain.append(zero_rain_t)
            self.buffer_depth.append(init_depth_t)
            
        print("Service Ready.")

    def process_incoming_rain(self, rain_npy):
        """
        rain_npy: 2D numpy array of rainfall
        """
        # Normalize Rain
        rain_norm = rain_npy / 50.0
        rain_t = torch.from_numpy(rain_norm).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Update Buffer
        self.buffer_rain.pop(0)
        self.buffer_rain.append(rain_t)
        
        # Prepare Input
        # Dynamic: Concat (Rain, PastDepth) for each step
        dynamic_list = []
        for r, d in zip(self.buffer_rain, self.buffer_depth):
            dynamic_list.append(torch.cat([r, d], dim=1))
            
        dynamic_input = torch.stack(dynamic_list, dim=1).to(DEVICE) # (1, 12, 2, H, W)
        
        # Run Inference
        # Horizons: 5 min (1 step), 30 min (6 steps), 120 min (24 steps)
        # We need max 24 steps
        with torch.no_grad():
            prediction_seq = self.model(dynamic_input, self.static_input, future_steps=24)
        
        # Save Results & Update Depth Buffer
        # Use the 5-min prediction (step 0) as the "Past Depth" for the next cycle
        next_depth = prediction_seq[:, 0:1, 0:1, :, :] # (1, 1, 1, H, W) -> remove time dim
        next_depth = next_depth.squeeze(1) # (1, 1, H, W)
        
        self.buffer_depth.pop(0)
        self.buffer_depth.append(next_depth)
        
        # Denormalize and Save predictions for requested horizons
        # 5 min = index 0
        # 30 min = index 5
        # 120 min = index 23
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = {}
        
        horizons_map = {
            5: 0,
            30: 5,
            120: 23
        }
        
        for mins, idx in horizons_map.items():
            if idx < prediction_seq.shape[1]:
                pred_map = prediction_seq[0, idx, 0].cpu().numpy()
                pred_map = pred_map * 5.0 # Denorm
                pred_map = np.maximum(pred_map, 0)
                
                # Save
                path = self.db.save_prediction(timestamp, mins, pred_map, {})
                saved_paths[mins] = path
                
        return saved_paths

service = AIService()

@app.route('/receive_rainfall', methods=['POST'])
def receive_rainfall():
    try:
        data = request.json
        # Assume data['image'] is base64 encoded TIF or NPY
        # For MVP simplicity, let's assume it sends a Base64 string of the TIF file bytes
        b64_string = data.get('image')
        if not b64_string:
            return jsonify({"error": "No image data provided"}), 400
            
        file_bytes = base64.b64decode(b64_string)
        
        # Convert bytes to numpy (assuming TIF)
        with rasterio.MemoryFile(file_bytes) as memfile:
            with memfile.open() as dataset:
                rain_arr = dataset.read(1)
                # Handle NaNs
                rain_arr = np.nan_to_num(rain_arr, nan=0.0)
        
        # Process
        paths = service.process_incoming_rain(rain_arr)
        
        return jsonify({"status": "success", "predictions": paths})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/prediction', methods=['GET'])
def get_prediction():
    try:
        horizon = int(request.args.get('time_horizon', 5))
        path, timestamp = service.db.get_latest_prediction(horizon)
        
        if not path:
            return jsonify({"error": "No prediction available"}), 404
            
        # Load the data to create the "Render Service" package structure response
        # In a real app, we might serve a zip. Here we return JSON metadata + Link to download npy
        
        flood_data = np.load(path)
        
        # Construct Metadata
        # Use Rasterio bounds to WGS84 (Approximation/Placeholder for MVP if logic not imported)
        # Ideally we use the transformation logic from experiments/
        
        # Quick WGS84 Bounds calculation (Assuming UTM56S like in experiments)
        # We will just mock the bounds here or assume they are known constants for the Demo
        bounds = {
             "north": -28.85, "south": -29.14, "east": 153.51, "west": 153.17,
             "center": {"lat": -29.0, "lon": 153.3}
        }
        
        metadata = {
            "request_id": f"req_{timestamp}",
            "timestamp": timestamp,
            "horizon_minutes": horizon,
            "bounds": bounds,
            "grid": {"width": flood_data.shape[1], "height": flood_data.shape[0]},
            "data_stats": {
                "max_depth_meters": float(np.max(flood_data)),
                "flooded_area_pixels": int(np.sum(flood_data > 0.05))
            },
            "format": "npy_float32"
        }
        
        # Check if user wants the file content or just metadata
        # We'll return metadata. The frontend can request the NPY via another endpoint if needed.
        # Or we can return the NPY in base64 if small enough (likely not for 1000x1000)
        
        return jsonify(metadata)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_npy', methods=['GET'])
def download_npy():
    horizon = int(request.args.get('time_horizon', 5))
    path, _ = service.db.get_latest_prediction(horizon)
    if path:
        return send_file(path)
    return "Not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

