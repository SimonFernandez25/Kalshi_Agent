"""
Auth Verification Script.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from prediction_agent.api.kalshi_client import KalshiClient

def main():
    print("Initializing KalshiClient...")
    client = KalshiClient()
    
    print("Fetching active markets (limit=3)...")
    markets = client.get_active_markets(limit=3)
    
    print(f"Success! Retrieved {len(markets)} markets.")
    for m in markets:
        print(f" - {m['market_id']}: {m['title']} (${m['last_price']})")

if __name__ == "__main__":
    main()
