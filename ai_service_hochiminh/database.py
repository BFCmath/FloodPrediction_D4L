import sqlite3
import json
import numpy as np
import os
import rasterio
from datetime import datetime
from pathlib import Path

class Database:
    def __init__(self, db_path="ai_service_db.sqlite", storage_dir="db_storage"):
        self.db_path = db_path
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      horizon_minutes INTEGER,
                      file_path TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS rainfall_log
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      file_path TEXT)''')
        conn.commit()
        conn.close()

    def save_rainfall(self, timestamp, data_bytes, filename):
        # Save file
        file_path = self.storage_dir / f"rain_{timestamp}_{filename}"
        with open(file_path, "wb") as f:
            f.write(data_bytes)
        
        # Log to DB
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO rainfall_log (timestamp, file_path) VALUES (?, ?)",
                  (timestamp, str(file_path)))
        conn.commit()
        conn.close()
        return str(file_path)

    def save_prediction(self, timestamp, horizon, flood_data, metadata=None):
        """
        Save prediction as GeoTIFF (Compressed LZW)
        metadata: rasterio profile dictionary (driver, height, width, count, dtype, crs, transform, etc.)
        
        Raises:
            ValueError: If flood_data is invalid
            RuntimeError: If saving fails
        """
        filename = f"pred_{timestamp}_{horizon}min.tif"
        file_path = self.storage_dir / filename
        
        # Validate flood_data - EXPLICIT ERRORS
        if flood_data is None:
            raise ValueError("[DB] ERROR: flood_data is None!")
        
        if flood_data.size == 0:
            raise ValueError("[DB] ERROR: flood_data is empty (size=0)!")
        
        # Ensure 2D
        if flood_data.ndim == 3 and flood_data.shape[0] == 1:
            flood_data = flood_data[0]
        
        if flood_data.ndim != 2:
            raise ValueError(f"[DB] ERROR: flood_data must be 2D, got shape: {flood_data.shape}")
        
        height, width = flood_data.shape
        
        # Require metadata - NO FALLBACK
        if metadata is None:
            raise ValueError("[DB] ERROR: metadata is required to save GeoTIFF! Cannot save without georeferencing.")
        
        # Check for shape mismatch - EXPLICIT WARNING
        meta_height = metadata.get('height')
        meta_width = metadata.get('width')
        
        if meta_height != height or meta_width != width:
            print(f"[DB] ⚠ SHAPE MISMATCH: Data={height}x{width}, Metadata={meta_height}x{meta_width}")
            # Adjust transform if shapes differ
            if 'transform' in metadata and meta_height and meta_width:
                old_transform = metadata['transform']
                scale_x = meta_width / width
                scale_y = meta_height / height
                new_transform = rasterio.Affine(
                    old_transform.a * scale_x,
                    old_transform.b,
                    old_transform.c,
                    old_transform.d,
                    old_transform.e * scale_y,
                    old_transform.f
                )
                metadata = metadata.copy()
                metadata['transform'] = new_transform
                print(f"[DB] Transform adjusted: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
            else:
                raise ValueError(f"[DB] ERROR: Cannot adjust transform - missing required metadata fields!")
        
        # Check for invalid values - EXPLICIT
        nan_count = np.isnan(flood_data).sum()
        inf_count = np.isinf(flood_data).sum()
        if nan_count > 0:
            raise ValueError(f"[DB] ERROR: flood_data contains {nan_count} NaN values! Fix before saving.")
        if inf_count > 0:
            raise ValueError(f"[DB] ERROR: flood_data contains {inf_count} Inf values! Fix before saving.")
        
        # Update metadata for this specific file
        meta = metadata.copy()
        meta.update({
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'float32',
            'compress': 'lzw',
            'nodata': -9999.0
        })
        
        # Remove predictor if present (can cause issues)
        if 'predictor' in meta:
            del meta['predictor']
        
        # Save - NO FALLBACK, explicit error
        try:
            with rasterio.open(file_path, 'w', **meta) as dst:
                dst.write(flood_data.astype('float32'), 1)
            print(f"[DB] ✓ Saved: {file_path} ({height}x{width}, {file_path.stat().st_size/1024:.1f}KB)")
        except Exception as e:
            raise RuntimeError(f"[DB] ERROR: Failed to save GeoTIFF: {e}")

        # Log to DB
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO predictions (timestamp, horizon_minutes, file_path) VALUES (?, ?, ?)",
                  (timestamp, horizon, str(file_path)))
        conn.commit()
        conn.close()
        
        return str(file_path)

    def get_latest_prediction(self, horizon):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT file_path, timestamp FROM predictions WHERE horizon_minutes = ? ORDER BY id DESC LIMIT 1", (horizon,))
        row = c.fetchone()
        conn.close()
        
        if row:
            return row[0], row[1]
        return None, None
