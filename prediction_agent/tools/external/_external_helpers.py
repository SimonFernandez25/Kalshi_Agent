"""
Shared helpers for external-data deterministic tools.
=====================================================
Mirrors the pattern of tools/_snapshot_helpers.py but targets
the external JSONL files in outputs/external/.

All functions:
  - Read-only (never write)
  - Deterministic (no randomness)
  - Fail-safe (return empty list on any IO error)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Default paths ──────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXTERNAL_DIR = _REPO_ROOT / "outputs" / "external"

FRED_JSONL    = EXTERNAL_DIR / "fred_snapshots.jsonl"
BLS_JSONL     = EXTERNAL_DIR / "bls_snapshots.jsonl"
WEATHER_JSONL = EXTERNAL_DIR / "weather_snapshots.jsonl"
ODDS_JSONL    = EXTERNAL_DIR / "odds_snapshots.jsonl"


# ── Timestamp parsing ──────────────────────────────────────────────────────────
def parse_ts(raw: Any) -> Optional[datetime]:
    """Parse ISO-8601 string into tz-aware datetime. Returns None on failure."""
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def recency_minutes(ts: datetime) -> float:
    """Return how many minutes ago ts was (vs now UTC). Negative = future."""
    now = datetime.now(timezone.utc)
    return (now - ts).total_seconds() / 60.0


# ── Generic JSONL loader ───────────────────────────────────────────────────────
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load all valid JSON lines from a JSONL file. Never raises."""
    if not path.exists():
        logger.debug("JSONL not found: %s", path)
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ── Statistics (pure stdlib, no numpy) ────────────────────────────────────────
def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def zscore(value: float, series: List[float]) -> float:
    """Z-score of value relative to series. Returns 0.0 if std is 0."""
    if not series:
        return 0.0
    s = std(series)
    if s == 0.0:
        return 0.0
    return (value - mean(series)) / s


def rolling_mean(series: List[float], window: int) -> List[float]:
    """Rolling mean with the given window size."""
    if not series or window <= 0:
        return []
    result = []
    for i in range(len(series)):
        start = max(0, i - window + 1)
        window_vals = series[start : i + 1]
        result.append(mean(window_vals))
    return result


def linear_trend(series: List[float]) -> float:
    """
    Slope of the OLS line fit through series (normalised by series length).
    Returns 0.0 for fewer than 2 points.
    Positive = upward trend, negative = downward.
    """
    n = len(series)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = mean(xs)
    my = mean(series)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, series))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


def volatility(series: List[float]) -> float:
    """Std dev of consecutive differences (realised volatility proxy)."""
    if len(series) < 2:
        return 0.0
    diffs = [series[i + 1] - series[i] for i in range(len(series) - 1)]
    return std(diffs)


# ── Confidence scoring ─────────────────────────────────────────────────────────
def data_completeness_confidence(
    actual: int,
    expected: int,
    recency_minutes_val: float,
    max_stale_minutes: float = 1440.0,  # 24 hours
) -> float:
    """
    Combine data completeness and recency into a [0, 1] confidence score.

    completeness_score = min(1, actual / expected)
    recency_score      = max(0, 1 - recency_minutes / max_stale_minutes)
    confidence         = 0.6 * completeness + 0.4 * recency
    """
    if expected <= 0:
        return 0.0
    completeness = min(1.0, actual / expected)
    recency = max(0.0, 1.0 - recency_minutes_val / max_stale_minutes)
    return round(0.6 * completeness + 0.4 * recency, 4)
