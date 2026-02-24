"""
BLS Labor / CPI Collector
=========================
Pulls data from the Bureau of Labor Statistics (BLS) Public Data API v2.
Normalises into structured snapshots and appends to:
    outputs/external/bls_snapshots.jsonl

Series collected:
  - LNS14000000  : Unemployment Rate (seasonally adjusted)
  - CUSR0000SA0  : CPI-U All Items (seasonally adjusted)
  - CES0000000001: Total Nonfarm Employees
  - LNS12000000  : Civilian Employment Level

Rules:
  - Deterministic: no randomness, same data -> same output
  - Idempotent: de-duplicates by series_id + year + period
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
_API_KEY_FILE = _REPO_ROOT / "APIs" / "USBLabor.txt"
_OUTPUT_FILE = _REPO_ROOT / "outputs" / "external" / "bls_snapshots.jsonl"

# ── Config ─────────────────────────────────────────────────────────────────────
BLS_BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

BLS_SERIES = [
    "LNS14000000",   # Unemployment Rate
    "CUSR0000SA0",   # CPI-U All Items
    "CES0000000001", # Total Nonfarm Payrolls
    "LNS12000000",   # Civilian Employment
]

YEARS_BACK = 2  # Request 2 years of data


# ── Key loading ────────────────────────────────────────────────────────────────
def _load_api_key() -> Optional[str]:
    key = os.environ.get("BLS_API_KEY")
    if key:
        return key.strip()
    if _API_KEY_FILE.exists():
        raw = _API_KEY_FILE.read_text(encoding="utf-8").strip()
        # Strip trailing period if present (seen in key file)
        key = raw.rstrip(".")
        return key if key else None
    return None


# ── JSONL helpers ──────────────────────────────────────────────────────────────
def _load_existing_keys(path: Path) -> set:
    """Return set of (series_id, year, period) already in file."""
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
                keys.add((row.get("series_id"), row.get("year"), row.get("period")))
            except json.JSONDecodeError:
                continue
    return keys


def _append_jsonl(path: Path, records: List[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
            written += 1
    return written


# ── Core collection ────────────────────────────────────────────────────────────
def fetch_series(api_key: str, series_ids: List[str]) -> Dict[str, List[Dict]]:
    """
    Fetch multiple BLS series in a single POST request.
    Returns dict of series_id -> list of observation dicts.
    """
    now = datetime.now(timezone.utc)
    start_year = str(now.year - YEARS_BACK)
    end_year = str(now.year)

    payload = {
        "seriesid": series_ids,
        "startyear": start_year,
        "endyear": end_year,
        "registrationkey": api_key,
    }

    try:
        resp = requests.post(
            BLS_BASE_URL,
            json=payload,
            headers={"Content-type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "REQUEST_SUCCEEDED":
            logger.warning("BLS API status: %s — %s", data.get("status"), data.get("message"))
            return {}

        results: Dict[str, List[Dict]] = {}
        for series in data.get("Results", {}).get("series", []):
            sid = series.get("seriesID", "")
            observations = []
            for obs in series.get("data", []):
                raw_val = obs.get("value", "")
                try:
                    value = float(raw_val)
                except (ValueError, TypeError):
                    continue
                observations.append({
                    "series_id": sid,
                    "year": obs.get("year"),
                    "period": obs.get("period"),       # e.g. "M01" = January
                    "period_name": obs.get("periodName"),
                    "value": value,
                    "footnotes": [f.get("text", "") for f in obs.get("footnotes", []) if f.get("text")],
                })
            results[sid] = observations
        return results

    except Exception as exc:
        logger.warning("BLS fetch failed: %s", exc)
        return {}


def collect(series_ids: Optional[List[str]] = None) -> int:
    """
    Collect BLS snapshots and append new records to JSONL.

    Returns:
        Number of new records written.
    """
    api_key = _load_api_key()
    if not api_key:
        logger.warning("BLS API key not found. Skipping collection.")
        return 0

    series_list = series_ids or BLS_SERIES
    existing_keys = _load_existing_keys(_OUTPUT_FILE)
    collected_at = datetime.now(timezone.utc).isoformat()

    series_data = fetch_series(api_key, series_list)

    new_records: List[Dict[str, Any]] = []
    for sid, observations in series_data.items():
        for obs in observations:
            key = (obs["series_id"], obs["year"], obs["period"])
            if key in existing_keys:
                continue
            obs["collected_at"] = collected_at
            obs["source"] = "bls"
            new_records.append(obs)
            existing_keys.add(key)

    written = _append_jsonl(_OUTPUT_FILE, new_records)
    logger.info("BLS collector: wrote %d new records", written)
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = collect()
    print(f"BLS: wrote {count} new records -> {_OUTPUT_FILE}")
