# AI Flood Forecasting MVP Service

## Overview
This service simulates a real-time flood forecasting system.
1. **Stream Service**: Reads historical rainfall data and sends it to the AI Service every T seconds.
2. **AI Service**: Maintains state (past rainfall/flood), runs the ConvLSTM model, and stores predictions.
3. **API**: Provides endpoints to receive data and retrieve predictions.

**CPU Optimization**: The model predicts only 6 future steps (5, 10, 15, 20, 25, 30 minutes) and saves only the 30-minute prediction. This is ~4x faster than predicting 24 steps (up to 120 minutes), making it suitable for CPU-only systems.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure:
   - Edit `config.env` to change `T` (interval in seconds).

## Running

1. **Start the AI Service (API)**:

   ```bash
   python main.py
   ```
   
   *Runs on http://localhost:5000*
   
   *Detailed logging will show: resource loading, request handling, model inference timing, file I/O operations*

2. **Start the Data Stream**:

   ```bash
   python stream_service.py
   ```
   
   *Will send file paths of TIF files from `rainfall/` to the API (no base64 encoding).*
   
   *Detailed logging will show: file discovery, validation, network requests, and timing for each step*

## API Endpoints

- **POST /receive_rainfall**
  - Input: JSON `{ "file_path": "/path/to/rainfall.tif", "filename": "...", "timestamp": "...", "file_size_bytes": ... }`
  - Output: JSON with status and saved prediction file paths.
  - **Note:** Services must run on the same machine with shared filesystem access. No base64 encoding is used for efficiency.

- **GET /prediction?time_horizon=30**
  - Input: `time_horizon` (only `30` is supported - optimized for CPU)
  - Output: JSON with metadata (Bounds, Stats) and file path.

- **GET /prediction_path?time_horizon=30**
  - Input: `time_horizon` (only `30` is supported - optimized for CPU)
  - Output: JSON with file path, timestamp, and existence check.
  - Use this endpoint when you only need the file path to read the prediction file directly from disk.

- **GET /rainfall_latest**
  - Output: JSON with file path and timestamp of the most recent rainfall data.
  - Returns the latest rainfall file stored in the system.

- **GET /download_prediction?time_horizon=30**
  - Input: `time_horizon` (only `30` is supported)
  - Output: The GeoTIFF file content for download.

## Database & Storage
- **SQLite database** (`ai_service_db.sqlite`): Stores metadata (timestamps, file paths) only.
- **Local file storage** (`db_storage/`): All prediction GeoTIFF files (~5MB each) and rainfall files (~50KB each) are stored locally.
- Other services can use the API endpoints to get file paths and read files directly from disk, avoiding large HTTP transfers.

## Logging & Performance Monitoring
Both services include comprehensive logging for performance optimization:
- **Stream Service**: Logs file scanning, validation, network requests, and timing for each phase
- **AI Service**: Logs model loading, data processing, inference, file I/O, and database operations
- All timing information is displayed in milliseconds for easy performance analysis
- Use these logs to identify bottlenecks and optimize the slowest operations


