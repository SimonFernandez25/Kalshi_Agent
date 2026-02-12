"""
Shared helpers for snapshot-based deterministic tools.
=====================================================
Common JSONL loading, timestamp parsing, price extraction, and spread
extraction used by all dataset-only tools.

This module is internal (prefixed with _) and not exported from the
tools package.  Individual tools import what they need.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default path to the snapshot JSONL file
DEFAULT_JSONL = Path(__file__).resolve().parent.parent / "outputs" / "market_snapshots.jsonl"


# ------------------------------------------------------------------
# Timestamp parsing
# ------------------------------------------------------------------
def parse_timestamp(raw: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string into a timezone-aware datetime."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=timezone.utc)
        return raw
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------
# JSONL loading with market_id + window filtering
# ------------------------------------------------------------------
def load_rows(
    jsonl_path: Path,
    market_id: str,
    window_minutes: int,
) -> List[Dict[str, Any]]:
    """
    Read JSONL and return rows matching *market_id* within the last
    *window_minutes*, sorted ascending by timestamp.

    Each returned row has an extra ``_parsed_ts`` key with the parsed
    datetime object for downstream use.
    """
    if not jsonl_path.exists():
        logger.warning("Snapshot file not found: %s", jsonl_path)
        return []

    now = datetime.now(timezone.utc)
    cutoff_seconds = window_minutes * 60
    matched: List[Dict[str, Any]] = []

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSON on line %d", line_num)
                continue

            if row.get("market_id") != market_id:
                continue

            ts = parse_timestamp(row.get("timestamp"))
            if ts is None:
                continue

            age_seconds = (now - ts).total_seconds()
            if age_seconds > cutoff_seconds:
                continue

            row["_parsed_ts"] = ts
            matched.append(row)

    matched.sort(key=lambda r: r["_parsed_ts"])
    return matched


# ------------------------------------------------------------------
# Price extraction  (PRICE RULE)
# ------------------------------------------------------------------
def row_price(row: Dict[str, Any]) -> Optional[float]:
    """
    Return usable price for a single row, or None.

    Priority: last_price > midpoint(yes_bid, yes_ask) > skip.
    """
    lp = row.get("last_price")
    if lp is not None and isinstance(lp, (int, float)) and lp > 0:
        return float(lp)

    bid = row.get("yes_bid")
    ask = row.get("yes_ask")
    if (
        bid is not None
        and ask is not None
        and isinstance(bid, (int, float))
        and isinstance(ask, (int, float))
        and (bid > 0 or ask > 0)
    ):
        return (float(bid) + float(ask)) / 2.0

    return None


def extract_prices(rows: List[Dict[str, Any]]) -> List[float]:
    """Extract price series from rows using the standard price rule."""
    prices: List[float] = []
    for r in rows:
        p = row_price(r)
        if p is not None:
            prices.append(p)
    return prices


# ------------------------------------------------------------------
# Spread extraction
# ------------------------------------------------------------------
def extract_spreads(rows: List[Dict[str, Any]]) -> List[float]:
    """
    Extract (yes_ask − yes_bid) for each row where both fields exist
    and are numeric.
    """
    spreads: List[float] = []
    for row in rows:
        bid = row.get("yes_bid")
        ask = row.get("yes_ask")
        if (
            bid is not None
            and ask is not None
            and isinstance(bid, (int, float))
            and isinstance(ask, (int, float))
        ):
            spreads.append(float(ask) - float(bid))
    return spreads


# ------------------------------------------------------------------
# Liquidity extraction
# ------------------------------------------------------------------
def extract_liquidity(rows: List[Dict[str, Any]]) -> List[float]:
    """
    Extract liquidity series: open_interest if present, else volume.
    Skips rows where neither field is available.
    """
    values: List[float] = []
    for row in rows:
        oi = row.get("open_interest")
        if oi is not None and isinstance(oi, (int, float)):
            values.append(float(oi))
        else:
            vol = row.get("volume")
            if vol is not None and isinstance(vol, (int, float)):
                values.append(float(vol))
    return values
