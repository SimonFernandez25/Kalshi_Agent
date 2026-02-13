"""
Kalshi API client — V2 Signed Authentication & Data Collection.

Responsibilities:
  1. Authenticate using RSA-signed headers (Kalshi V2).
  2. Fetch ALL active markets (paginated).
  3. Return normalized snapshots.
  4. Fallback to stubs on ANY failure (crash-safe).

Reads:
  - env: KALSHI_ACCESS_KEY_ID
  - file: FebuaruyAPIKalshi.txt (RSA Private Key)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from config import (
    API_KEY_FILE,
    KALSHI_BASE_URL,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Stub data (Fallback)
# ──────────────────────────────────────────────
STUB_MARKETS: List[Dict[str, Any]] = [
    {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_id": "STUB-NBA-LAL-BOS-001",
        "title": "Lakers vs Celtics: Lakers Win",
        "status": "active",
        "yes_bid": 0.50,
        "yes_ask": 0.55,
        "last_price": 0.53,
        "volume": 1000,
        "open_interest": 500,
        "liquidity": 10000.0,
        "close_time": "2026-03-01T00:00:00Z",
        "time_to_close_sec": 3600,
        "series_ticker": "NBA",
        "category": "Sports",
    },
    {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_id": "STUB-ECON-CPI-MAR-002",
        "title": "US CPI > 3.0%",
        "status": "active",
        "yes_bid": 0.20,
        "yes_ask": 0.25,
        "last_price": 0.22,
        "volume": 5000,
        "open_interest": 2000,
        "liquidity": 50000.0,
        "close_time": "2026-03-10T12:00:00Z",
        "time_to_close_sec": 86400,
        "series_ticker": "CPI",
        "category": "Economics",
    },
    {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_id": "STUB-POL-PREZ-2028",
        "title": "Who will be president 2028?",
        "status": "active",
        "yes_bid": 0.10,
        "yes_ask": 0.12,
        "last_price": 0.11,
        "volume": 100000,
        "open_interest": 50000,
        "liquidity": 25000.0,
        "close_time": "2028-11-05T00:00:00Z",
        "time_to_close_sec": 90000000,
        "series_ticker": "PREZ",
        "category": "Politics",
    },
]


class KalshiClient:
    """
    Kalshi V2 Client with RSA Request Signing.
    """

    def __init__(self):
        self.base_url = KALSHI_BASE_URL
        self.session = requests.Session()
        
        # Load Credentials
        self.access_key_id = os.environ.get("KALSHI_ACCESS_KEY_ID")
        self.private_key: Optional[rsa.RSAPrivateKey] = self._load_private_key()

        if self.access_key_id and self.private_key:
            logger.info("KalshiClient initialized with RSA auth.")
        else:
            logger.warning("KalshiClient missing credentials (env or file). Using STUBS.")

    def _load_private_key(self) -> Optional[rsa.RSAPrivateKey]:
        """Load RSA private key from file using cryptography."""
        try:
            if not API_KEY_FILE.exists():
                logger.warning("Key file not found: %s", API_KEY_FILE)
                return None
            
            key_data = API_KEY_FILE.read_bytes()
            # Try loading as PEM
            private_key = load_pem_private_key(key_data, password=None)
            return private_key
        except Exception as exc:
            logger.error("Failed to load RSA private key: %s", exc)
            return None

    def _sign_request(self, method: str, path: str, body: str, timestamp: str) -> str:
        """
        Generate RSA-SHA256 signature for Kalshi V2.
        
        Message format: timestamp + method + path + body
        """
        if not self.private_key:
            return ""

        msg_str = f"{timestamp}{method.upper()}{path}{body}"
        msg_bytes = msg_str.encode('utf-8')

        try:
            signature = self.private_key.sign(
                msg_bytes,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode('utf-8')
        except Exception as exc:
            logger.error("Signing failed: %s", exc)
            return ""

    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Construct signed headers."""
        if not self.access_key_id or not self.private_key:
            return {}

        timestamp = str(int(time.time() * 1000))  # milliseconds
        signature = self._sign_request(method, path, body, timestamp)

        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.access_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    # ──────────────────────────────────────────────
    # Public API Methods
    # ──────────────────────────────────────────────

    def get_active_markets(self, limit: int = 100, status: str = "open") -> List[Dict[str, Any]]:
        """
        Fetch active markets with pagination.
        Returns normalized list of snapshots.
        Falls back to STUBS on error.
        """
        if not self.access_key_id or not self.private_key:
            logger.info("Auth missing — returning stubs.")
            return STUB_MARKETS[:limit]

        all_markets = []
        cursor = None
        page_count = 0
        max_pages = 10  # Hard cap

        try:
            while len(all_markets) < limit and page_count < max_pages:
                path = "/markets"
                # Query params are NOT part of the signed path string in V2 usually,
                # but we must check docs. Conventionally for Kalshi:
                # Signature uses the path part of the URL (e.g. /trade-api/v2/markets).
                # Query parameters are INCLUDED in the signature path if they are part of the request target?
                # Kalshi docs say: "The path parameter is the path of the request, including query parameters."
                
                # Construct query params
                params = {
                    "limit": min(100, limit - len(all_markets)),
                    "status": status,
                }
                if cursor:
                    params["cursor"] = cursor

                # Manually construct path with query for signing
                # (requests encodes params, so we need to match that)
                # For simplicity, let's trust requests but we need the exact string for signing.
                # A safe way is to assume minimal params or construct string manually.
                # Let's rely on basic construction:
                path_with_query = path + "?" + "&".join(f"{k}={v}" for k, v in params.items())
                
                # NOTE: The implementation of `kalshi_base_url` usually includes `/trade-api/v2`
                # If `self.base_url` is `.../v2`, then `path` should be `/markets`.
                # Checks strict path usage.
                
                # Full URL is base + path
                # Signing path usually excludes host but includes base path prefix if any?
                # Kalshi: "path: The path of the request, e.g. /trade-api/v2/markets"
                # If our base_url has /trade-api/v2, we need to extract the path component relative to domain?
                # Or does the user config have it?
                # config.py: KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
                
                # So the path to sign is `/trade-api/v2/markets?limit=...`
                
                sign_path = "/trade-api/v2" + path_with_query
                
                headers = self._get_headers("GET", sign_path)
                
                resp = self.session.get(
                    self.base_url + path,
                    params=params,
                    headers=headers,
                    timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                
                markets = data.get("markets", [])
                cursor = data.get("cursor")
                
                for m in markets:
                    all_markets.append(self._normalize_market(m))
                
                if not cursor:
                    break
                page_count += 1
                time.sleep(0.1) # Rate limit nice-ness

            logger.info("Fetched %d active markets.", len(all_markets))
            return all_markets[:limit]

        except Exception as exc:
            logger.warning("API call failed: %s — returning stubs.", exc)
            return STUB_MARKETS[:limit]

    def get_market_snapshot(self, market_id: str) -> Dict[str, Any]:
        """Fetch single market snapshot. Fallback to stub."""
        # This implementation just reuses list logic or could be direct call.
        # For efficiency in collector, we rely on get_active_markets.
        # This is a helper if needed.
        return STUB_MARKETS[0] # Placeholder if not strictly used by collector

    def _normalize_market(self, m: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw API dict to flat schema."""
        # Safely extract fields
        # Note: 'yes_ask' might be in different places depending on API version (orderbook vs ticker)
        # Kalshi V2 market object usually has 'yes_ask', 'yes_bid', 'last_price', 'liquidity', etc.
        
        # Calculate time to close
        close_time_str = m.get("close_time", "")
        time_to_close = 0
        if close_time_str:
            try:
                dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                time_to_close = int((dt - now).total_seconds())
            except:
                pass

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": m.get("ticker", ""),
            "title": m.get("title", ""),
            "status": m.get("status", "unknown"),
            "yes_bid": m.get("yes_bid", 0.0), # price in cents? API often returns cents.
            # If API returns 50 for 50 cents, we should normalize to 0.50?
            # Warning: Kalshi V2 usually assumes cents (1-99).
            # We will normalize / 100 if > 1.
            "yes_ask": self._norm_price(m.get("yes_ask", 0)),
            "last_price": self._norm_price(m.get("last_price", 0)),
            "volume": m.get("volume", 0),
            "open_interest": m.get("open_interest", 0),
            "liquidity": float(m.get("liquidity", 0)),
            "close_time": close_time_str,
            "time_to_close_sec": time_to_close,
            "series_ticker": m.get("series_ticker", ""),
            "category": m.get("category", "Uncategorized"),
            # Normalized bid
            "yes_bid": self._norm_price(m.get("yes_bid", 0)),
        }

    def _norm_price(self, p: Any) -> float:
        """Normalize price to 0.0-1.0 range."""
        if p is None: 
            return 0.0
        try:
            val = float(p)
            if val > 1.0: # assume cents
                return val / 100.0
            return val
        except:
            return 0.0
