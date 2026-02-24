"""
FRED Macro Tool
===============
Deterministic tool that reads fred_snapshots.jsonl and returns
macro-economic signal vectors.

Reads  : outputs/external/fred_snapshots.jsonl
Writes : nothing
APIs   : none
Random : none

Output vector (5 elements):
    [latest_value, rolling_3_mean, rolling_3_trend, zscore_latest, volatility_last_6]

Each element is normalised to a [0, 1] range where possible.
Confidence is derived from data completeness and recency.

Usage in agent:
    Useful for understanding macro-economic context when predicting
    outcomes tied to economic events (CPI, rate decisions, employment).
    High-frequency trading signals for NBA markets: use to assess
    macro volatility that could affect market sentiment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas import EventInput, ToolOutput
from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

# Import helpers using absolute path resolution (works both as tool and in tests)
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

from _external_helpers import (
    FRED_JSONL,
    data_completeness_confidence,
    linear_trend,
    load_jsonl,
    mean,
    recency_minutes,
    rolling_mean,
    parse_ts,
    std,
    volatility,
    zscore,
)

_VECTOR_LEN = 5
_MIN_ROWS = 3
_DEFAULT_SERIES = "FEDFUNDS"


class FredMacroTool(BaseTool):
    """
    Reads FRED macro snapshots and returns a 5-element signal vector.

    Deterministic: identical snapshot file + identical series_id -> identical output.
    No API calls. No randomness.

    Output vector semantics:
      [0] latest_value         : most recent observation value (normalised 0-1 by /100)
      [1] rolling_3_mean       : mean of last 3 observations (normalised)
      [2] rolling_3_trend      : OLS slope of last 3 observations (signed, clamped)
      [3] zscore_latest        : z-score of latest vs last 12 obs (clamped to [-3,3] then /6 + 0.5)
      [4] volatility_last_6    : std of diffs of last 6 obs (normalised by /10)
    """

    def __init__(self, jsonl_path: Optional[Path] = None) -> None:
        self._jsonl_path = jsonl_path or FRED_JSONL

    @property
    def name(self) -> str:
        return "fred_macro_tool"

    @property
    def description(self) -> str:
        return (
            "Macro-economic signal from FRED data. "
            "Output: [latest_value, rolling_3_mean, rolling_3_trend, zscore_latest, volatility_last_6]. "
            "Useful for economic event markets (CPI prints, rate decisions, employment reports). "
            "Reads fred_snapshots.jsonl. No API calls."
        )

    def run(self, event: EventInput, **kwargs: Any) -> ToolOutput:
        series_id: str = kwargs.get("series_id", _DEFAULT_SERIES)

        rows = load_jsonl(self._jsonl_path)
        series_rows = sorted(
            [r for r in rows if r.get("series_id") == series_id],
            key=lambda r: (r.get("observation_date", ""), r.get("collected_at", "")),
        )

        values = [r["value"] for r in series_rows if isinstance(r.get("value"), (int, float))]

        n = len(values)
        if n < _MIN_ROWS:
            logger.info("%s: insufficient data for %s (n=%d) — returning zeros", self.name, series_id, n)
            return ToolOutput(
                tool_name=self.name,
                output_vector=[0.0] * _VECTOR_LEN,
                metadata={
                    "series_id": series_id,
                    "data_points_used": n,
                    "data_recency_minutes": None,
                    "confidence": 0.0,
                },
            )

        # Recency: time since last collected_at
        last_collected = parse_ts(series_rows[-1].get("collected_at"))
        recency_min = recency_minutes(last_collected) if last_collected else 99999.0

        # Compute signals
        latest = values[-1]
        last3 = values[-3:]
        last6 = values[-6:] if n >= 6 else values
        last12 = values[-12:] if n >= 12 else values

        r3_means = rolling_mean(last3, 3)
        r3_mean_val = r3_means[-1] if r3_means else latest

        r3_trend = linear_trend(last3)
        z = zscore(latest, last12)
        vol = volatility(last6)

        # Normalise to [0, 1]
        # latest_value: divide by 100 (works for rates 0-100, clamp)
        norm_latest = min(1.0, max(0.0, latest / 100.0))
        # rolling_3_mean: same normalisation
        norm_r3_mean = min(1.0, max(0.0, r3_mean_val / 100.0))
        # rolling_3_trend: signed slope, clamp to [-1, 1] then shift to [0, 1]
        norm_trend = min(1.0, max(0.0, (r3_trend / 5.0) + 0.5))
        # zscore: clamp [-3, 3], map to [0, 1]
        norm_z = min(1.0, max(0.0, (max(-3.0, min(3.0, z)) / 6.0) + 0.5))
        # volatility: divide by 10 as rough normaliser
        norm_vol = min(1.0, max(0.0, vol / 10.0))

        output_vector = [
            round(norm_latest, 6),
            round(norm_r3_mean, 6),
            round(norm_trend, 6),
            round(norm_z, 6),
            round(norm_vol, 6),
        ]

        confidence = data_completeness_confidence(
            actual=n,
            expected=12,
            recency_minutes_val=recency_min,
            max_stale_minutes=1440.0,  # 24h — macro data updates daily/monthly
        )

        logger.info(
            "%s: series=%s n=%d recency=%.1fmin confidence=%.3f vector=%s",
            self.name, series_id, n, recency_min, confidence, output_vector,
        )

        return ToolOutput(
            tool_name=self.name,
            output_vector=output_vector,
            metadata={
                "series_id": series_id,
                "data_points_used": n,
                "data_recency_minutes": round(recency_min, 1),
                "confidence": confidence,
                "latest_raw_value": latest,
            },
        )
