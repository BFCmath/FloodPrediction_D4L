import os
import time
import requests
import glob
from pathlib import Path
from dotenv import load_dotenv
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [STREAM] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load Config
load_dotenv("config.env")
INTERVAL_SECONDS = int(os.getenv("T", 10))
API_URL = os.getenv("AI_SERVICE_URL", "http://localhost:5000") + "/receive_rainfall"

RAINFALL_DIR = Path("rainfall")

def main():
    logger.info("="*60)
    logger.info("Starting Data Stream Service")
    logger.info(f"Target API: {API_URL}")
    logger.info(f"Interval: {INTERVAL_SECONDS} seconds")
    logger.info("="*60)

    # Get list of files
    t_scan_start = time.time()
    files = sorted(list(RAINFALL_DIR.glob("*.tif")))
    
    if not files:
        logger.warning(f"No rainfall files found in '{RAINFALL_DIR}/' folder")
        logger.info("Checking backup location: ../for_dev/Rainfall/Australia_flood")
        backup_dir = Path("../for_dev/Rainfall/Australia_flood")
        if backup_dir.exists():
            files = sorted(list(backup_dir.glob("*.tif")))
            logger.info(f"Found {len(files)} files in backup source")
        else:
            logger.error("No files found anywhere. Exiting.")
            return
    
    t_scan_end = time.time()
    logger.info(f"File scan completed: {len(files)} files found in {(t_scan_end-t_scan_start)*1000:.1f}ms")

    cycle_count = 0
    while True:
        cycle_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"CYCLE #{cycle_count} - Processing {len(files)} files")
        logger.info(f"{'='*60}")
        
        for idx, file_path in enumerate(files, 1):
            try:
                logger.info(f"\n[{idx}/{len(files)}] Processing: {file_path.name}")
                t_start = time.time()
                
                # Check file exists and get size
                t0 = time.time()
                if not file_path.exists():
                    logger.error(f"  ✗ File does not exist: {file_path}")
                    continue
                
                file_size = file_path.stat().st_size
                t1 = time.time()
                logger.info(f"  → File check: {(t1-t0)*1000:.2f}ms | Size: {file_size/1024:.1f}KB")
                
                # Prepare payload with absolute path
                absolute_path = str(file_path.absolute())
                payload = {
                    "filename": file_path.name,
                    "file_path": absolute_path,
                    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                    "file_size_bytes": file_size
                }
                
                t2 = time.time()
                logger.info(f"  → Payload prep: {(t2-t1)*1000:.2f}ms")
                logger.info(f"  → Sending file path: {absolute_path}")
                
                # Send request
                t3 = time.time()
                response = requests.post(API_URL, json=payload, timeout=30)
                t4 = time.time()
                
                logger.info(f"  → Network request: {(t4-t3)*1000:.1f}ms")
                logger.info(f"  → HTTP Status: {response.status_code}")
                
                if response.status_code == 200:
                    t5 = time.time()
                    resp_data = response.json()
                    t6 = time.time()
                    logger.info(f"  → Response parse: {(t6-t5)*1000:.2f}ms")
                    logger.info(f"  ✓ Success: {resp_data.get('status', 'OK')}")
                    if 'predictions' in resp_data:
                        logger.info(f"  → Predictions saved: {list(resp_data['predictions'].keys())}")
                else:
                    logger.error(f"  ✗ Server Error {response.status_code}")
                    logger.error(f"  → Response: {response.text[:200]}")
                
                t_end = time.time()
                logger.info(f"  → Total processing: {(t_end-t_start)*1000:.1f}ms")
                    
            except requests.exceptions.Timeout:
                logger.error(f"  ✗ Request timeout (30s)")
            except requests.exceptions.RequestException as e:
                logger.error(f"  ✗ Network error: {e}")
            except Exception as e:
                logger.error(f"  ✗ Exception: {type(e).__name__}: {e}")
            
            logger.info(f"\n⏳ Waiting {INTERVAL_SECONDS}s before next file...")
            time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()

