"""
TheOddsAPI Collector
====================
Pulls live sportsbook odds from TheOddsAPI.com.
Normalises into structured snapshots and appends to:
    outputs/external/odds_snapshots.jsonl

Sports collected by default:
  - basketball_nba : NBA moneylines
  - americanfootball_nfl : NFL (cross-market context)

Rules:
  - Deterministic: no randomness, same data -> same output
  - Idempotent: de-duplicates by event_id + bookmaker + collected_hour
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
_API_KEY_FILE = _REPO_ROOT / "APIs" / "TheOddsAPI.txt"
_OUTPUT_FILE = _REPO_ROOT / "outputs" / "external" / "odds_snapshots.jsonl"

# ── Config ─────────────────────────────────────────────────────────────────────
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"

DEFAULT_SPORTS = [
    "basketball_nba",
    "americanfootball_nfl",
]

MARKETS = "h2h"         # head-to-head moneyline
ODDS_FORMAT = "decimal" # decimal odds (easier to convert to implied probability)
REGIONS = "us"          # US bookmakers


# ── Key loading ────────────────────────────────────────────────────────────────
def _load_api_key() -> Optional[str]:
    key = os.environ.get("THE_ODDS_API_KEY")
    if key:
        return key.strip()
    if _API_KEY_FILE.exists():
        key = _API_KEY_FILE.read_text(encoding="utf-8").strip()
        return key if key else None
    return None


# ── JSONL helpers ──────────────────────────────────────────────────────────────
def _load_existing_keys(path: Path) -> set:
    """Return set of (event_id, bookmaker, collected_hour) already in file."""
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
                    row.get("event_id"),
                    row.get("bookmaker"),
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


# ── Probability helpers ────────────────────────────────────────────────────────
def decimal_to_implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability. Clamps to [0, 1]."""
    if odds <= 0:
        return 0.0
    return min(1.0, max(0.0, round(1.0 / odds, 6)))


def remove_vig(home_prob: float, away_prob: float) -> tuple[float, float]:
    """Remove bookmaker vig (overround) to get true implied probabilities."""
    total = home_prob + away_prob
    if total <= 0:
        return home_prob, away_prob
    return round(home_prob / total, 6), round(away_prob / total, 6)


# ── Core collection ────────────────────────────────────────────────────────────
def fetch_odds(api_key: str, sport: str) -> List[Dict[str, Any]]:
    """
    Fetch current odds for a sport from TheOddsAPI.
    Returns list of raw event dicts or empty list on failure.
    """
    url = ODDS_BASE_URL.format(sport=sport)
    params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("TheOddsAPI fetch failed for %s: %s", sport, exc)
        return []


def _normalise_event(event: Dict[str, Any], sport: str, collected_at: str) -> List[Dict[str, Any]]:
    """
    Expand one event into per-bookmaker records.
    Each record contains implied probabilities with vig removed.
    """
    collected_hour = collected_at[:13]  # hourly bucket
    event_id = event.get("id", "")
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    commence_time = event.get("commence_time", "")

    records = []
    for bookmaker in event.get("bookmakers", []):
        bk_key = bookmaker.get("key", "")
        bk_title = bookmaker.get("title", "")
        last_update = bookmaker.get("last_update", "")

        # Find the h2h market
        h2h_outcomes: Dict[str, float] = {}
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price", 1.0)
                h2h_outcomes[name] = decimal_to_implied_prob(price)

        home_raw = h2h_outcomes.get(home_team, 0.0)
        away_raw = h2h_outcomes.get(away_team, 0.0)
        home_prob, away_prob = remove_vig(home_raw, away_raw)

        records.append({
            "event_id": event_id,
            "sport": sport,
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "bookmaker": bk_key,
            "bookmaker_title": bk_title,
            "bookmaker_last_update": last_update,
            "home_implied_prob": home_prob,
            "away_implied_prob": away_prob,
            "home_raw_implied": home_raw,
            "away_raw_implied": away_raw,
            "vig_total": round(home_raw + away_raw - 1.0, 6) if (home_raw + away_raw) > 0 else None,
            "collected_hour": collected_hour,
            "collected_at": collected_at,
            "source": "theoddsapi",
        })

    return records


def collect(sports: Optional[List[str]] = None) -> int:
    """
    Collect odds snapshots and append new records to JSONL.

    Returns:
        Number of new records written.
    """
    api_key = _load_api_key()
    if not api_key:
        logger.warning("TheOddsAPI key not found. Skipping collection.")
        return 0

    sport_list = sports or DEFAULT_SPORTS
    existing_keys = _load_existing_keys(_OUTPUT_FILE)
    collected_at = datetime.now(timezone.utc).isoformat()

    new_records: List[Dict[str, Any]] = []
    for sport in sport_list:
        events = fetch_odds(api_key, sport)
        for event in events:
            event_records = _normalise_event(event, sport, collected_at)
            for rec in event_records:
                key = (rec["event_id"], rec["bookmaker"], rec["collected_hour"])
                if key in existing_keys:
                    continue
                new_records.append(rec)
                existing_keys.add(key)

    written = _append_jsonl(_OUTPUT_FILE, new_records)
    logger.info("Odds collector: wrote %d new records", written)
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = collect()
    print(f"TheOddsAPI: wrote {count} new records -> {_OUTPUT_FILE}")
