"""
FRED Macro Collector
====================
Pulls series data from the Federal Reserve Economic Data (FRED) API.
Normalises into structured snapshots and appends to:
    outputs/external/fred_snapshots.jsonl

Key series collected:
  - FEDFUNDS   : Federal Funds Rate
  - UNRATE     : Unemployment Rate
  - CPIAUCSL   : CPI All Urban Consumers
  - T10Y2Y     : 10Y-2Y Treasury Spread (yield curve)
  - VIXCLS     : CBOE Volatility Index

Rules:
  - Deterministic: no randomness, same data -> same output
  - Idempotent: de-duplicates by series_id + observation_date
  - Fail-safe: any API error leaves existing JSONL untouched
  - No modifications to other pipeline components
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_API_KEY_FILE = _REPO_ROOT / "APIs" / "FRED.txt"
_OUTPUT_FILE = _REPO_ROOT / "outputs" / "external" / "fred_snapshots.jsonl"

# ── Config ─────────────────────────────────────────────────────────────────────
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = [
    "FEDFUNDS",   # Federal Funds Rate
    "UNRATE",     # Unemployment Rate
    "CPIAUCSL",   # CPI All Urban Consumers
    "T10Y2Y",     # 10Y-2Y Treasury Spread
    "VIXCLS",     # CBOE VIX
]

OBSERVATIONS_LIMIT = 12  # Last 12 observations per series


# ── Key loading ────────────────────────────────────────────────────────────────
def _load_api_key() -> Optional[str]:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key.strip()
    if _API_KEY_FILE.exists():
        key = _API_KEY_FILE.read_text(encoding="utf-8").strip()
        return key if key else None
    return None


# ── JSONL helpers ──────────────────────────────────────────────────────────────
def _load_existing_keys(path: Path) -> set:
    """Return set of (series_id, observation_date) already in file."""
    if not path.exists():
        return set()
    keys = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                keys.add((row.get("series_id"), row.get("observation_date")))
            except json.JSONDecodeError:
                continue
    return keys


def _append_jsonl(path: Path, records: List[Dict[str, Any]]) -> int:
    """Append records to JSONL. Returns count of records written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
            written += 1
    return written


# ── Core collection ────────────────────────────────────────────────────────────
def fetch_series(api_key: str, series_id: str) -> List[Dict[str, Any]]:
    """
    Fetch the most recent N observations for a FRED series.
    Returns list of normalised dicts or empty list on failure.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": OBSERVATIONS_LIMIT,
    }
    try:
        resp = requests.get(FRED_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])
        results = []
        for obs in observations:
            raw_val = obs.get("value", ".")
            if raw_val == ".":
                continue  # FRED uses "." for missing values
            try:
                value = float(raw_val)
            except (ValueError, TypeError):
                continue
            results.append({
                "series_id": series_id,
                "observation_date": obs.get("date"),
                "value": value,
            })
        return results
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return []


def collect(series_ids: Optional[List[str]] = None) -> int:
    """
    Collect FRED snapshots and append new records to JSONL.

    Args:
        series_ids: Override default series list (for testing).

    Returns:
        Number of new records written.
    """
    api_key = _load_api_key()
    if not api_key:
        logger.warning("FRED API key not found. Skipping collection.")
        return 0

    series_list = series_ids or FRED_SERIES
    existing_keys = _load_existing_keys(_OUTPUT_FILE)
    collected_at = datetime.now(timezone.utc).isoformat()

    new_records: List[Dict[str, Any]] = []
    for series_id in series_list:
        observations = fetch_series(api_key, series_id)
        for obs in observations:
            key = (obs["series_id"], obs["observation_date"])
            if key in existing_keys:
                continue
            obs["collected_at"] = collected_at
            obs["source"] = "fred"
            new_records.append(obs)
            existing_keys.add(key)

    written = _append_jsonl(_OUTPUT_FILE, new_records)
    logger.info("FRED collector: wrote %d new records", written)
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = collect()
    print(f"FRED: wrote {count} new records -> {_OUTPUT_FILE}")
