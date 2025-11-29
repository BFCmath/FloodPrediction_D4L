import os
import time
import base64
import requests
import glob
from pathlib import Path
from dotenv import load_dotenv

# Load Config
load_dotenv("config.env")
INTERVAL_SECONDS = int(os.getenv("T", 10))
API_URL = os.getenv("AI_SERVICE_URL", "http://localhost:5000") + "/receive_rainfall"

RAINFALL_DIR = Path("rainfall")

def main():
    print(f"Starting Data Stream Service...")
    print(f"Target: {API_URL}")
    print(f"Interval: {INTERVAL_SECONDS} seconds")

    # Get list of files (recursively or flat)
    # The task says "stored in rainfall/*.tif"
    # In the file structure I saw earlier, it was 'for_dev/Rainfall/...' 
    # But task says "Folder to work with: D:\project\d4l\ai_service" and "stored in rainfall\*.tif"
    # I assume the user will put files there. I'll check if I need to copy them from for_dev.
    
    files = sorted(list(RAINFALL_DIR.glob("*.tif")))
    
    if not files:
        print("No rainfall files found in 'rainfall/' folder.")
        print("Checking 'for_dev' backup...")
        backup_dir = Path("../for_dev/Rainfall/Australia_flood")
        if backup_dir.exists():
            files = sorted(list(backup_dir.glob("*.tif")))
            print(f"Found {len(files)} files in backup source.")
        else:
            print("No files found anywhere. Exiting.")
            return

    while True:
        for file_path in files:
            print(f"\n[STREAM] Sending: {file_path.name}")
            
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                    b64_data = base64.b64encode(file_bytes).decode('utf-8')
                
                payload = {
                    "filename": file_path.name,
                    "image": b64_data,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
                
                start = time.time()
                response = requests.post(API_URL, json=payload)
                duration = time.time() - start
                
                if response.status_code == 200:
                    print(f"   -> Success (took {duration:.2f}s)")
                    print(f"   -> Response: {response.json().get('predictions', 'OK')}")
                else:
                    print(f"   -> Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"   -> Exception: {e}")
            
            print(f"[STREAM] Waiting {INTERVAL_SECONDS}s...")
            time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()

