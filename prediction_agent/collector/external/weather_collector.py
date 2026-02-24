"""
WeatherAPI Collector
====================
Pulls current + 3-day forecast data from WeatherAPI.com.
Normalises into structured snapshots and appends to:
    outputs/external/weather_snapshots.jsonl

Default locations: NBA arena cities used in Kalshi markets.

Rules:
  - Deterministic: no randomness, same data -> same output
  - Idempotent: de-duplicates by location + forecast_date + collected_hour
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
_API_KEY_FILE = _REPO_ROOT / "APIs" / "WeatherAPI.txt"
_OUTPUT_FILE = _REPO_ROOT / "outputs" / "external" / "weather_snapshots.jsonl"

# ── Config ─────────────────────────────────────────────────────────────────────
WEATHER_BASE_URL = "https://api.weatherapi.com/v1/forecast.json"

# NBA arena cities — relevant for game-day weather context
DEFAULT_LOCATIONS = [
    "Boston,MA",
    "Los Angeles,CA",
    "Miami,FL",
    "Chicago,IL",
    "New York,NY",
    "Dallas,TX",
    "Phoenix,AZ",
    "Denver,CO",
]

FORECAST_DAYS = 3  # API supports 1-10 days on paid plans, 3 on free


# ── Key loading ────────────────────────────────────────────────────────────────
def _load_api_key() -> Optional[str]:
    key = os.environ.get("WEATHER_API_KEY")
    if key:
        return key.strip()
    if _API_KEY_FILE.exists():
        key = _API_KEY_FILE.read_text(encoding="utf-8").strip()
        return key if key else None
    return None


# ── JSONL helpers ──────────────────────────────────────────────────────────────
def _load_existing_keys(path: Path) -> set:
    """Return set of (location_key, forecast_date, collected_hour) already written."""
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
                keys.add((
                    row.get("location_key"),
                    row.get("forecast_date"),
                    row.get("collected_hour"),
                ))
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
def fetch_forecast(api_key: str, location: str) -> Optional[Dict[str, Any]]:
    """
    Fetch forecast data for a location.
    Returns normalised dict or None on failure.
    """
    params = {
        "key": api_key,
        "q": location,
        "days": FORECAST_DAYS,
        "aqi": "no",
        "alerts": "yes",
    }
    try:
        resp = requests.get(WEATHER_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("WeatherAPI fetch failed for %s: %s", location, exc)
        return None


def _normalise_forecast(raw: Dict[str, Any], location_query: str, collected_at: str) -> List[Dict[str, Any]]:
    """
    Extract per-day forecast records from raw WeatherAPI response.
    Each record covers one forecast day.
    """
    collected_hour = collected_at[:13]  # "YYYY-MM-DDTHH" — hourly bucket for dedup

    loc = raw.get("location", {})
    location_key = f"{loc.get('name','')},{loc.get('region','')}".lower().replace(" ", "_")

    records = []
    for day in raw.get("forecast", {}).get("forecastday", []):
        day_data = day.get("day", {})
        hour_data = day.get("hour", [])

        # Model disagreement proxy: std dev of hourly precip forecasts
        hourly_precip = [h.get("precip_mm", 0.0) for h in hour_data]
        if len(hourly_precip) > 1:
            mean_p = sum(hourly_precip) / len(hourly_precip)
            variance = sum((x - mean_p) ** 2 for x in hourly_precip) / len(hourly_precip)
            model_disagreement_proxy = variance ** 0.5
        else:
            model_disagreement_proxy = 0.0

        # Temperature anomaly proxy: deviation of max temp from seasonal average
        # We use (max_temp_c - 15.0) as a simple proxy; positive = warmer than baseline
        max_temp_c = day_data.get("maxtemp_c", 15.0)
        temp_anomaly = max_temp_c - 15.0

        record = {
            "location_key": location_key,
            "location_query": location_query,
            "location_name": loc.get("name"),
            "location_region": loc.get("region"),
            "location_country": loc.get("country"),
            "location_lat": loc.get("lat"),
            "location_lon": loc.get("lon"),
            "forecast_date": day.get("date"),
            "collected_hour": collected_hour,
            "collected_at": collected_at,
            "source": "weatherapi",
            # Forecast fields
            "max_temp_c": max_temp_c,
            "min_temp_c": day_data.get("mintemp_c"),
            "avg_temp_c": day_data.get("avgtemp_c"),
            "temp_anomaly_c": round(temp_anomaly, 2),
            "total_precip_mm": day_data.get("totalprecip_mm", 0.0),
            "avg_humidity": day_data.get("avghumidity"),
            "max_wind_kph": day_data.get("maxwind_kph"),
            "daily_chance_of_rain": day_data.get("daily_chance_of_rain", 0),
            "daily_chance_of_snow": day_data.get("daily_chance_of_snow", 0),
            "condition_text": day_data.get("condition", {}).get("text"),
            "uv_index": day_data.get("uv"),
            # Derived signals
            "forecast_probability": round(day_data.get("daily_chance_of_rain", 0) / 100.0, 4),
            "model_disagreement_proxy": round(model_disagreement_proxy, 4),
            # Alerts
            "has_weather_alert": len(raw.get("alerts", {}).get("alert", [])) > 0,
        }
        records.append(record)

    return records


def collect(locations: Optional[List[str]] = None) -> int:
    """
    Collect weather snapshots and append new records to JSONL.

    Returns:
        Number of new records written.
    """
    api_key = _load_api_key()
    if not api_key:
        logger.warning("WeatherAPI key not found. Skipping collection.")
        return 0

    location_list = locations or DEFAULT_LOCATIONS
    existing_keys = _load_existing_keys(_OUTPUT_FILE)
    collected_at = datetime.now(timezone.utc).isoformat()

    new_records: List[Dict[str, Any]] = []
    for location in location_list:
        raw = fetch_forecast(api_key, location)
        if raw is None:
            continue
        day_records = _normalise_forecast(raw, location, collected_at)
        for rec in day_records:
            key = (rec["location_key"], rec["forecast_date"], rec["collected_hour"])
            if key in existing_keys:
                continue
            new_records.append(rec)
            existing_keys.add(key)

    written = _append_jsonl(_OUTPUT_FILE, new_records)
    logger.info("Weather collector: wrote %d new records", written)
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = collect()
    print(f"WeatherAPI: wrote {count} new records -> {_OUTPUT_FILE}")
