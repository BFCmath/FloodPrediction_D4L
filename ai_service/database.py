import sqlite3
import json
import numpy as np
import os
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

    def save_prediction(self, timestamp, horizon, flood_data_npy, metadata):
        # Create a specific folder for this prediction packet (like mock_api_response)
        # But here we might just save the array and metadata linked by ID
        
        # Save npy
        filename = f"pred_{timestamp}_{horizon}min.npy"
        file_path = self.storage_dir / filename
        np.save(file_path, flood_data_npy)

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

