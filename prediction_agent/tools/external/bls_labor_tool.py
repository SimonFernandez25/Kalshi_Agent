"""
BLS Labor Tool
==============
Deterministic tool that reads bls_snapshots.jsonl and returns
labor market signal vectors.

Reads  : outputs/external/bls_snapshots.jsonl
Writes : nothing
APIs   : none
Random : none

Output vector (5 elements):
    [unemployment_rate, monthly_delta, 3m_trend, yoy_change, surprise_proxy]

All elements normalised to [0, 1].
Confidence derived from data completeness and recency.

Usage in agent:
    Useful for markets tied to employment reports, economic health,
    and consumer spending. BLS unemployment surprises move markets.
    Cross-signal with CPI data for inflation/employment correlation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas import EventInput, ToolOutput
from tools.base_tool import BaseTool

import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

from _external_helpers import (
    BLS_JSONL,
    data_completeness_confidence,
    linear_trend,
    load_jsonl,
    mean,
    recency_minutes,
    parse_ts,
    std,
)

logger = logging.getLogger(__name__)

_VECTOR_LEN = 5
_MIN_ROWS = 3
_DEFAULT_SERIES = "LNS14000000"  # Unemployment Rate

# BLS uses M01-M12 for monthly; sort by year+period for chronological order
def _period_sort_key(row: Dict) -> str:
    return f"{row.get('year', '0000')}{row.get('period', 'M00')}"


class BlsLaborTool(BaseTool):
    """
    Reads BLS labor market snapshots and returns a 5-element signal vector.

    Deterministic: identical snapshot file + identical series_id -> identical output.
    No API calls. No randomness.

    Output vector semantics:
      [0] unemployment_rate  : latest rate normalised (rate / 20.0, clamped to [0,1])
      [1] monthly_delta      : month-over-month change, shifted to [0,1] (0.5 = no change)
      [2] 3m_trend           : OLS slope of last 3 obs, shifted to [0,1]
      [3] yoy_change         : year-over-year change (last vs 12 months ago), shifted [0,1]
      [4] surprise_proxy     : deviation of latest from 3m rolling mean, shifted [0,1]
    """

    def __init__(self, jsonl_path: Optional[Path] = None) -> None:
        self._jsonl_path = jsonl_path or BLS_JSONL

    @property
    def name(self) -> str:
        return "bls_labor_tool"

    @property
    def description(self) -> str:
        return (
            "Labor market signal from BLS data. "
            "Output: [unemployment_rate, monthly_delta, 3m_trend, yoy_change, surprise_proxy]. "
            "Useful for employment-linked market predictions. "
            "Reads bls_snapshots.jsonl. No API calls."
        )

    def run(self, event: EventInput, **kwargs: Any) -> ToolOutput:
        series_id: str = kwargs.get("series_id", _DEFAULT_SERIES)

        rows = load_jsonl(self._jsonl_path)
        series_rows = sorted(
            [r for r in rows if r.get("series_id") == series_id],
            key=_period_sort_key,
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

        # Recency
        last_collected = parse_ts(series_rows[-1].get("collected_at"))
        recency_min = recency_minutes(last_collected) if last_collected else 99999.0

        latest = values[-1]

        # Monthly delta
        monthly_delta = values[-1] - values[-2] if n >= 2 else 0.0

        # 3-month trend
        last3 = values[-3:]
        trend_3m = linear_trend(last3)

        # Year-over-year (12 obs back)
        yoy = (values[-1] - values[-13]) if n >= 13 else (values[-1] - values[0])

        # Surprise proxy: deviation from 3-period rolling mean
        rm3 = mean(last3)
        surprise = latest - rm3

        # Normalise to [0, 1]
        norm_ur = min(1.0, max(0.0, latest / 20.0))             # unemployment rate 0-20%
        norm_delta = min(1.0, max(0.0, (monthly_delta / 2.0) + 0.5))  # ±1% typical range
        norm_trend = min(1.0, max(0.0, (trend_3m / 1.0) + 0.5))       # ±0.5 typical slope
        norm_yoy = min(1.0, max(0.0, (yoy / 4.0) + 0.5))              # ±2% typical range
        norm_surprise = min(1.0, max(0.0, (surprise / 1.0) + 0.5))    # ±0.5 typical

        output_vector = [
            round(norm_ur, 6),
            round(norm_delta, 6),
            round(norm_trend, 6),
            round(norm_yoy, 6),
            round(norm_surprise, 6),
        ]

        confidence = data_completeness_confidence(
            actual=n,
            expected=13,
            recency_minutes_val=recency_min,
            max_stale_minutes=43200.0,  # 30 days — BLS publishes monthly
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
