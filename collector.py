"""
Data Collection Engine — Robust & Crash-Safe.

Args:
  --duration-hours (float)
  --interval-seconds (int)
  --max-markets (int)

Output:
  outputs/market_snapshots.jsonl (Append-Only)
"""

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from prediction_agent.api.kalshi_client import KalshiClient
from prediction_agent.config import OUTPUTS_DIR

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collector")

OUTPUT_FILE = OUTPUTS_DIR / "market_snapshots.jsonl"

class CollectorEngine:
    def __init__(self, duration_hours: float, interval_seconds: int, max_markets: int):
        self.duration_sec = duration_hours * 3600
        self.interval = max(2, interval_seconds) # Clamp min 2s
        self.max_markets = max_markets
        self.start_time = time.time()
        self.running = True
        
        self.client = KalshiClient()
        
        # Sigint handler
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("\nStop signal received. Shutting down gracefully...")
        self.running = False

    def run(self):
        logger.info("Starting Data Collection")
        logger.info(f"Target: {OUTPUT_FILE}")
        logger.info(f"Duration: {self.duration_sec/3600:.2f}h | Interval: {self.interval}s")
        
        count_polls = 0
        total_rows = 0
        
        try:
            while self.running:
                # Check duration
                elapsed = time.time() - self.start_time
                if elapsed >= self.duration_sec:
                    logger.info("Duration reached. Stopping.")
                    break

                cycle_start = time.time()
                
                # Fetch
                try:
                    markets = self.client.get_active_markets(limit=self.max_markets)
                except Exception as exc:
                    logger.error(f"Critical fetch error: {exc}")
                    markets = [] # Should be handled by client stub, but double safety

                if not markets:
                    logger.warning("No markets returned.")
                
                # Write
                self._append_rows(markets)
                
                count = len(markets)
                total_rows += count
                count_polls += 1
                
                duration_str = f"{elapsed/60:.1f}m"
                logger.info(f"Poll #{count_polls} | +{count} rows | Total: {total_rows} | Time: {duration_str}")

                # Sleep remainder
                work_time = time.time() - cycle_start
                sleep_time = max(0, self.interval - work_time)
                if self.running and sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.exception("Collector crashed!")
            raise e
        finally:
            logger.info(f"Collection finished. Total rows: {total_rows}")

    def _append_rows(self, markets: list):
        if not markets:
            return
            
        try:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for m in markets:
                    # Ensure schema
                    f.write(json.dumps(m) + "\n")
        except Exception as exc:
            logger.error(f"Write failed: {exc}")

def main():
    parser = argparse.ArgumentParser(description="Kalshi Data Collector")
    parser.add_argument("--duration-hours", type=float, default=0.1, help="Run duration in hours")
    parser.add_argument("--interval-seconds", type=int, default=60, help="Poll interval")
    parser.add_argument("--max-markets", type=int, default=50, help="Max markets per poll")
    
    args = parser.parse_args()
    
    engine = CollectorEngine(
        duration_hours=args.duration_hours,
        interval_seconds=args.interval_seconds,
        max_markets=args.max_markets
    )
    engine.run()

if __name__ == "__main__":
    main()
