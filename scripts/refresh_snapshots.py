
import json
from datetime import datetime, timezone

INPUT_FILE = "outputs/market_snapshots.jsonl"
OUTPUT_FILE = "outputs/market_snapshots.jsonl"

def refresh_timestamps():
    print(f"Reading from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    refreshed_lines = []
    now_iso = datetime.now(timezone.utc).isoformat()
    
    count = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            # Update timestamp to NOW
            data["timestamp"] = now_iso
            refreshed_lines.append(json.dumps(data))
            count += 1
        except json.JSONDecodeError:
            continue
            
    print(f"Refreshed {count} records with timestamp: {now_iso}")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in refreshed_lines:
            f.write(line + "\n")
            
    print(f"Wrote back to {OUTPUT_FILE}")

if __name__ == "__main__":
    refresh_timestamps()
