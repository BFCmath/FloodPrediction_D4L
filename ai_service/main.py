import os
import json
import sqlite3
import numpy as np
import torch
import rasterio
import time
import logging
from flask import Flask, request, jsonify, send_file
from pathlib import Path
from datetime import datetime
from model import FloodForecastModel
from database import Database

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [AI-SERVICE] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- CONFIG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path("weights/best_model.pth")
STATIC_DIR = Path("static_data")
OUTPUT_DIR = Path("output_predictions")
OUTPUT_DIR.mkdir(exist_ok=True)

# CPU Optimization: Only predict 6 steps (30 minutes) instead of 24 steps (120 minutes)
# This makes inference ~4x faster on CPU
FUTURE_STEPS = 6  # 5, 10, 15, 20, 25, 30 minutes
TARGET_HORIZON = 30  # Only save 30-minute prediction

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
        logger.info("="*60)
        logger.info("LOADING RESOURCES")
        logger.info("="*60)
        
        # 1. Load Model
        t0 = time.time()
        logger.info("Step 1: Loading Neural Network Model...")
        logger.info(f"  → Device: {DEVICE}")
        logger.info(f"  → Model path: {MODEL_PATH}")
        
        self.model = FloodForecastModel(input_channels=4, hidden_channels=64, kernel_size=3).to(DEVICE)
        t1 = time.time()
        logger.info(f"  → Model architecture created: {(t1-t0)*1000:.1f}ms")
        
        if MODEL_PATH.exists():
            try:
                t2 = time.time()
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                t3 = time.time()
                logger.info(f"  → Weights loaded from disk: {(t3-t2)*1000:.1f}ms")
                
                self.model.load_state_dict(state_dict)
                t4 = time.time()
                logger.info(f"  → Weights applied to model: {(t4-t3)*1000:.1f}ms")
                logger.info("  ✓ Model weights loaded successfully")
            except Exception as e:
                logger.warning(f"  ✗ Could not load weights: {e}")
                logger.warning("  → Using random weights for MVP")
        else:
            logger.warning(f"  ✗ Model weights not found at {MODEL_PATH}")
            logger.warning("  → Using random weights for MVP")
        
        self.model.eval()
        t5 = time.time()
        logger.info(f"  → Model set to eval mode")
        logger.info(f"  → Total model load time: {(t5-t0)*1000:.1f}ms")

        # 2. Load Static Data
        logger.info("\nStep 2: Loading Static Geographic Data...")
        dem_path = STATIC_DIR / "Australia_DEM.tif"
        manning_path = STATIC_DIR / "Australia_Manning.tif"
        init_cond_path = STATIC_DIR / "Australia_initial_condition.tif"
        
        # Check files
        logger.info(f"  → Checking DEM file: {dem_path}")
        if not dem_path.exists(): 
            logger.error(f"  ✗ Missing DEM file: {dem_path}")
            raise FileNotFoundError(f"Missing {dem_path}")
        logger.info(f"  ✓ DEM file found")
        
        # Load DEM
        t_dem0 = time.time()
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            self.dem_meta = src.profile
            self.dem_bounds = src.bounds
            logger.info(f"  → DEM shape: {dem.shape}, CRS: {src.crs}")
            logger.info(f"  → DEM bounds: {self.dem_bounds}")
            logger.info(f"  → DEM value range: [{np.min(dem):.2f}, {np.max(dem):.2f}]")
            # Simple Normalize
            dem_norm = (dem - np.min(dem)) / (np.max(dem) - np.min(dem) + 1e-6)
        t_dem1 = time.time()
        logger.info(f"  → DEM loaded & normalized: {(t_dem1-t_dem0)*1000:.1f}ms")
        
        # Load Manning (if missing, use ones)
        t_man0 = time.time()
        if manning_path.exists():
            logger.info(f"  → Loading Manning file: {manning_path}")
            with rasterio.open(manning_path) as src:
                man = src.read(1)
            logger.info(f"  ✓ Manning loaded, range: [{np.min(man):.3f}, {np.max(man):.3f}]")
        else:
            logger.warning(f"  ✗ Manning file not found: {manning_path}")
            man = np.ones_like(dem) * 0.03
            logger.info(f"  → Using default Manning coefficient: 0.03")
        t_man1 = time.time()
        logger.info(f"  → Manning processing: {(t_man1-t_man0)*1000:.1f}ms")

        # Static Tensor
        t_tensor0 = time.time()
        dem_t = torch.from_numpy(dem_norm).float().unsqueeze(0).unsqueeze(0)
        man_t = torch.from_numpy(man).float().unsqueeze(0).unsqueeze(0)
        self.static_input = torch.cat([dem_t, man_t], dim=1).to(DEVICE)
        t_tensor1 = time.time()
        logger.info(f"  → Static tensor created: shape {self.static_input.shape}")
        logger.info(f"  → Tensor creation & GPU transfer: {(t_tensor1-t_tensor0)*1000:.1f}ms")

        # 3. Initialize Buffers (12 frames)
        logger.info("\nStep 3: Initializing State Buffers...")
        t_buf0 = time.time()
        
        # Initial condition
        if init_cond_path.exists():
            logger.info(f"  → Loading initial condition: {init_cond_path}")
            t_ic0 = time.time()
            with rasterio.open(init_cond_path) as src:
                init_depth = src.read(1)
                init_depth = np.maximum(np.nan_to_num(init_depth), 0) / 5.0 # Norm
            t_ic1 = time.time()
            logger.info(f"  ✓ Initial condition loaded: {(t_ic1-t_ic0)*1000:.1f}ms")
            logger.info(f"  → Initial depth range: [{np.min(init_depth):.4f}, {np.max(init_depth):.4f}] (normalized)")
        else:
            logger.warning(f"  ✗ Initial condition not found: {init_cond_path}")
            init_depth = np.zeros_like(dem)
            logger.info(f"  → Using zero initial depth")

        t_buf1 = time.time()
        init_depth_t = torch.from_numpy(init_depth).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        zero_rain_t = torch.zeros_like(init_depth_t).to(DEVICE)
        
        logger.info(f"  → Creating 12-frame buffers for rainfall and depth...")
        for _ in range(12):
            self.buffer_rain.append(zero_rain_t)
            self.buffer_depth.append(init_depth_t)
        t_buf2 = time.time()
        logger.info(f"  ✓ Buffers initialized: {(t_buf2-t_buf1)*1000:.1f}ms")
        logger.info(f"  → Total buffer setup: {(t_buf2-t_buf0)*1000:.1f}ms")
            
        logger.info("\n" + "="*60)
        logger.info("✓ SERVICE READY")
        logger.info("="*60)

    def process_incoming_rain(self, rain_npy):
        """
        rain_npy: 2D numpy array of rainfall
        """
        logger.info("  → Starting prediction pipeline...")
        t_start = time.time()
        
        # Normalize Rain
        t0 = time.time()
        logger.info(f"  → Input rain shape: {rain_npy.shape}, range: [{np.min(rain_npy):.2f}, {np.max(rain_npy):.2f}]")
        rain_norm = rain_npy / 50.0
        rain_t = torch.from_numpy(rain_norm).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        t1 = time.time()
        logger.info(f"    • Normalize & tensorize: {(t1-t0)*1000:.2f}ms")
        
        # Update Buffer
        t2 = time.time()
        self.buffer_rain.pop(0)
        self.buffer_rain.append(rain_t)
        logger.info(f"    • Update rain buffer: {(t2-t1)*1000:.2f}ms")
        
        # Prepare Input
        # Dynamic: Concat (Rain, PastDepth) for each step
        t3 = time.time()
        dynamic_list = []
        for r, d in zip(self.buffer_rain, self.buffer_depth):
            dynamic_list.append(torch.cat([r, d], dim=1))
            
        dynamic_input = torch.stack(dynamic_list, dim=1).to(DEVICE) # (1, 12, 2, H, W)
        t_prep = time.time()
        logger.info(f"    • Prepare dynamic input: {(t_prep-t3)*1000:.2f}ms")
        logger.info(f"    • Dynamic input shape: {dynamic_input.shape}")
        
        # Run Inference
        # Horizons: 5, 10, 15, 20, 25, 30 minutes (6 steps total)
        # Optimized for CPU - only predict up to 30 minutes
        logger.info(f"  → Running model inference ({FUTURE_STEPS} future steps = {TARGET_HORIZON} minutes)...")
        t_inf0 = time.time()
        
        with torch.no_grad():
            prediction_seq = self.model(dynamic_input, self.static_input, future_steps=FUTURE_STEPS)
        
        t_infer = time.time()
        total_inference_time = (t_infer - t_inf0) * 1000
        
        logger.info(f"    • Total model inference: {total_inference_time:.1f}ms")
        logger.info(f"    • Average per step: {total_inference_time/FUTURE_STEPS:.1f}ms")
        logger.info(f"    • Output shape: {prediction_seq.shape}")
        
        # Log timing breakdown for each horizon (5/10/15/20/25/30 min)
        logger.info("    • Predicted horizons with timing:")
        avg_time_per_step = total_inference_time / FUTURE_STEPS
        for i in range(FUTURE_STEPS):
            horizon_min = (i + 1) * 5
            logger.info(f"      - {horizon_min:2d} min (step {i}) - ~{avg_time_per_step:.1f}ms")
        
        # Save Results & Update Depth Buffer
        # Use the 5-min prediction (step 0) as the "Past Depth" for the next cycle
        t_buf0 = time.time()
        next_depth = prediction_seq[:, 0:1, 0:1, :, :] # (1, 1, 1, H, W) -> remove time dim
        next_depth = next_depth.squeeze(1) # (1, 1, H, W)
        
        self.buffer_depth.pop(0)
        self.buffer_depth.append(next_depth)
        t_buf1 = time.time()
        logger.info(f"    • Update depth buffer: {(t_buf1-t_buf0)*1000:.2f}ms")
        
        # Denormalize and Save predictions
        # We only save the 30-minute prediction (index 5, step 6)
        logger.info(f"  → Saving {TARGET_HORIZON}-minute prediction...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = {}
        
        # Only save 30-minute horizon (optimized for CPU performance)
        target_horizon_min = TARGET_HORIZON
        target_idx = (TARGET_HORIZON // 5) - 1  # 30 min = 6th step (index 5)
        
        t_save_start = time.time()
        
        if target_idx < prediction_seq.shape[1]:
            t_s0 = time.time()
            pred_map = prediction_seq[0, target_idx, 0].cpu().numpy()
            t_cpu = time.time()
            logger.info(f"    • GPU→CPU transfer: {(t_cpu-t_s0)*1000:.2f}ms")
            
            pred_map = pred_map * 5.0 # Denorm
            pred_map = np.maximum(pred_map, 0)
            t_s1 = time.time()
            
            logger.info(f"    • Horizon 30min: range [{np.min(pred_map):.3f}, {np.max(pred_map):.3f}]m")
            logger.info(f"    • Flooded pixels (>0.05m): {np.sum(pred_map > 0.05)}")
            
            # Save as GeoTIFF
            path = self.db.save_prediction(timestamp, target_horizon_min, pred_map, self.dem_meta)
            saved_paths[target_horizon_min] = path
            t_s2 = time.time()
            
            file_size = Path(path).stat().st_size if Path(path).exists() else 0
            logger.info(f"    • Saved: {Path(path).name} ({file_size/1024:.1f}KB)")
            logger.info(f"    • Save timing: denorm {(t_s1-t_cpu)*1000:.2f}ms + write {(t_s2-t_s1)*1000:.2f}ms = {(t_s2-t_s0)*1000:.1f}ms total")
        
        t_save = time.time()
        
        logger.info(f"  ✓ PIPELINE COMPLETE:")
        logger.info(f"    • Total: {(t_save-t_start)*1000:.1f}ms")
        logger.info(f"    • Prep: {(t_prep-t_start)*1000:.1f}ms")
        logger.info(f"    • Inference: {(t_infer-t_prep)*1000:.1f}ms")
        logger.info(f"    • Save: {(t_save-t_infer)*1000:.1f}ms")
                
        return saved_paths

service = AIService()

@app.route('/receive_rainfall', methods=['POST'])
def receive_rainfall():
    try:
        logger.info("\n" + "="*60)
        logger.info("RECEIVED RAINFALL DATA")
        logger.info("="*60)
        
        t_recv = time.time()
        data = request.json
        
        # Accept file_path instead of base64 image
        file_path_str = data.get('file_path')
        if not file_path_str:
            logger.error("✗ No file_path provided in request")
            return jsonify({"error": "No file_path provided"}), 400
        
        file_path = Path(file_path_str)
        filename = data.get('filename', file_path.name)
        timestamp = data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        file_size = data.get('file_size_bytes', 0)
        
        logger.info(f"Request metadata:")
        logger.info(f"  • File path: {file_path}")
        logger.info(f"  • Filename: {filename}")
        logger.info(f"  • Timestamp: {timestamp}")
        logger.info(f"  • File size: {file_size/1024:.1f}KB")
        
        # Replace : with _ in timestamp to avoid Windows path issues
        timestamp = timestamp.replace(':', '_')
        
        # Verify file exists
        t0 = time.time()
        if not file_path.exists():
            logger.error(f"✗ File does not exist: {file_path}")
            return jsonify({"error": f"File not found: {file_path}"}), 404
        
        actual_size = file_path.stat().st_size
        logger.info(f"  ✓ File exists, actual size: {actual_size/1024:.1f}KB")
        t1 = time.time()
        logger.info(f"  → File validation: {(t1-t0)*1000:.2f}ms")
        
        # Read file directly from disk
        logger.info("Reading rainfall TIF from disk...")
        t2 = time.time()
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        t3 = time.time()
        logger.info(f"  → File read: {(t3-t2)*1000:.1f}ms ({len(file_bytes)/1024:.1f}KB)")
        
        # Save rainfall file to database storage
        logger.info("Saving rainfall to database storage...")
        t4 = time.time()
        service.db.save_rainfall(timestamp, file_bytes, filename)
        t_store_rain = time.time()
        logger.info(f"  → Database storage: {(t_store_rain-t4)*1000:.1f}ms")

        # Convert bytes to numpy (assuming TIF)
        logger.info("Parsing GeoTIFF data...")
        t5 = time.time()
        with rasterio.MemoryFile(file_bytes) as memfile:
            with memfile.open() as dataset:
                rain_arr = dataset.read(1)
                logger.info(f"  → Array shape: {rain_arr.shape}")
                logger.info(f"  → CRS: {dataset.crs}")
                # Handle NaNs
                nan_count = np.isnan(rain_arr).sum()
                if nan_count > 0:
                    logger.info(f"  → Found {nan_count} NaN values, replacing with 0")
                rain_arr = np.nan_to_num(rain_arr, nan=0.0)
        t6 = time.time()
        logger.info(f"  → GeoTIFF parsing: {(t6-t5)*1000:.1f}ms")
        logger.info(f"  → Rain value range: [{np.min(rain_arr):.2f}, {np.max(rain_arr):.2f}]")
        
        # Process
        logger.info("\nStarting AI prediction pipeline...")
        t_proc_start = time.time()
        paths = service.process_incoming_rain(rain_arr)
        t_end = time.time()
        
        logger.info("\n" + "="*60)
        logger.info("✓ REQUEST COMPLETE")
        logger.info(f"Total timing breakdown:")
        logger.info(f"  • File validation: {(t1-t_recv)*1000:.1f}ms")
        logger.info(f"  • File read: {(t3-t2)*1000:.1f}ms")
        logger.info(f"  • Store rainfall: {(t_store_rain-t4)*1000:.1f}ms")
        logger.info(f"  • Parse GeoTIFF: {(t6-t5)*1000:.1f}ms")
        logger.info(f"  • AI processing: {(t_end-t_proc_start)*1000:.1f}ms")
        logger.info(f"  • TOTAL: {(t_end-t_recv)*1000:.1f}ms")
        logger.info("="*60 + "\n")
        
        return jsonify({"status": "success", "predictions": paths})
        
    except Exception as e:
        logger.error(f"✗ EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/prediction', methods=['GET'])
def get_prediction():
    try:
        t0 = time.time()
        horizon = int(request.args.get('time_horizon', TARGET_HORIZON))
        logger.info(f"GET /prediction?time_horizon={horizon}")
        
        # Only 30-minute predictions are available now
        if horizon != TARGET_HORIZON:
            logger.warning(f"  ⚠ Only {TARGET_HORIZON}-minute predictions available (requested: {horizon})")
            return jsonify({"error": f"Only {TARGET_HORIZON}-minute predictions are available. Use time_horizon={TARGET_HORIZON}"}), 400
        
        t1 = time.time()
        path, timestamp = service.db.get_latest_prediction(horizon)
        t2 = time.time()
        logger.info(f"  → DB query: {(t2-t1)*1000:.2f}ms")
        
        if not path:
            logger.warning(f"  ✗ No prediction available for horizon {horizon}min")
            return jsonify({"error": "No prediction available"}), 404
        
        logger.info(f"  → Found: {path}")
            
        # Load the data to create the "Render Service" package structure response
        t3 = time.time()
        flood_data = None
        if str(path).endswith('.tif'):
            with rasterio.open(path) as src:
                flood_data = src.read(1)
        else:
            flood_data = np.load(path)
        t4 = time.time()
        logger.info(f"  → Load data: {(t4-t3)*1000:.1f}ms")
        logger.info(f"  → Data shape: {flood_data.shape}")
        
        # Construct Metadata
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
            "format": "geotiff_lzw" if str(path).endswith('.tif') else "npy_float32"
        }
        
        t5 = time.time()
        logger.info(f"  ✓ Response ready: {(t5-t0)*1000:.1f}ms total")
        
        # Return both metadata and file path for other services
        return jsonify({
            "metadata": metadata,
            "file_path": str(path)
        })

    except Exception as e:
        logger.error(f"  ✗ Error in /prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_prediction', methods=['GET'])
def download_prediction():
    t0 = time.time()
    horizon = int(request.args.get('time_horizon', TARGET_HORIZON))
    logger.info(f"GET /download_prediction?time_horizon={horizon}")
    
    if horizon != TARGET_HORIZON:
        logger.warning(f"  ⚠ Only {TARGET_HORIZON}-minute predictions available (requested: {horizon})")
        return jsonify({"error": f"Only {TARGET_HORIZON}-minute predictions are available. Use time_horizon={TARGET_HORIZON}"}), 400
    
    path, _ = service.db.get_latest_prediction(horizon)
    if path:
        file_size = Path(path).stat().st_size if Path(path).exists() else 0
        logger.info(f"  → Sending file: {path} ({file_size/1024:.1f}KB)")
        t1 = time.time()
        logger.info(f"  ✓ Response ready: {(t1-t0)*1000:.1f}ms")
        return send_file(path)
    
    logger.warning(f"  ✗ Not found")
    return "Not found", 404

@app.route('/prediction_path', methods=['GET'])
def get_prediction_path():
    """Returns the file path for the latest prediction at a given horizon."""
    try:
        t0 = time.time()
        horizon = int(request.args.get('time_horizon', TARGET_HORIZON))
        logger.info(f"GET /prediction_path?time_horizon={horizon}")
        
        if horizon != TARGET_HORIZON:
            logger.warning(f"  ⚠ Only {TARGET_HORIZON}-minute predictions available (requested: {horizon})")
            return jsonify({"error": f"Only {TARGET_HORIZON}-minute predictions are available. Use time_horizon={TARGET_HORIZON}"}), 400
        
        t1 = time.time()
        path, timestamp = service.db.get_latest_prediction(horizon)
        t2 = time.time()
        logger.info(f"  → DB query: {(t2-t1)*1000:.2f}ms")
        
        if not path:
            logger.warning(f"  ✗ No prediction available")
            return jsonify({"error": "No prediction available"}), 404
        
        exists = os.path.exists(path)
        file_size = Path(path).stat().st_size if exists else 0
        logger.info(f"  → Path: {path}")
        logger.info(f"  → Exists: {exists}, Size: {file_size/1024:.1f}KB")
        t3 = time.time()
        logger.info(f"  ✓ Response ready: {(t3-t0)*1000:.1f}ms")
        
        return jsonify({
            "file_path": str(path),
            "timestamp": timestamp,
            "horizon_minutes": horizon,
            "exists": exists
        })
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/rainfall_latest', methods=['GET'])
def get_latest_rainfall():
    """Returns info about the most recent rainfall file."""
    try:
        t0 = time.time()
        logger.info("GET /rainfall_latest")
        
        t1 = time.time()
        conn = sqlite3.connect(service.db.db_path)
        c = conn.cursor()
        c.execute("SELECT file_path, timestamp FROM rainfall_log ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        t2 = time.time()
        logger.info(f"  → DB query: {(t2-t1)*1000:.2f}ms")
        
        if not row:
            logger.warning(f"  ✗ No rainfall data available")
            return jsonify({"error": "No rainfall data available"}), 404
        
        exists = os.path.exists(row[0])
        file_size = Path(row[0]).stat().st_size if exists else 0
        logger.info(f"  → Path: {row[0]}")
        logger.info(f"  → Timestamp: {row[1]}")
        logger.info(f"  → Exists: {exists}, Size: {file_size/1024:.1f}KB")
        t3 = time.time()
        logger.info(f"  ✓ Response ready: {(t3-t0)*1000:.1f}ms")
        
        return jsonify({
            "file_path": row[0],
            "timestamp": row[1],
            "exists": exists
        })
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("\n" + "="*60)
    logger.info("STARTING FLASK SERVER")
    logger.info("Host: 0.0.0.0")
    logger.info("Port: 5000")
    logger.info("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000)
