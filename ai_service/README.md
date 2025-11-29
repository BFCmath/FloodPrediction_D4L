# AI Flood Forecasting MVP Service

## Overview
This service simulates a real-time flood forecasting system.
1. **Stream Service**: Reads historical rainfall data and sends it to the AI Service every T seconds.
2. **AI Service**: Maintains state (past rainfall/flood), runs the ConvLSTM model, and stores predictions.
3. **API**: Provides endpoints to receive data and retrieve predictions.

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

2. **Start the Data Stream**:
   ```bash
   python stream_service.py
   ```
   *Will start sending TIF files from `rainfall/` to the API.*

## API Endpoints

- **POST /receive_rainfall**
  - Input: JSON `{ "image": "base64_string...", "filename": "..." }`
  - Output: JSON with saved file paths.

- **GET /prediction?time_horizon=5**
  - Input: `time_horizon` (5, 30, 120)
  - Output: JSON Metadata (Bounds, Stats) + Link to NPY.

- **GET /download_npy?time_horizon=5**
  - Output: The .npy file content.

## Database
- A SQLite database `ai_service_db.sqlite` tracks all processed files.
- Predictions are stored in `db_storage/`.

