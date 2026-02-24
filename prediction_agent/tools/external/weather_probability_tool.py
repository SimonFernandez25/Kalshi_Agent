"""
Weather Probability Tool
========================
Deterministic tool that reads weather_snapshots.jsonl and returns
weather-based signal vectors.

Reads  : outputs/external/weather_snapshots.jsonl
Writes : nothing
APIs   : none
Random : none

Output vector (5 elements):
    [forecast_probability, precipitation_mm, temp_anomaly, forecast_confidence,
     model_disagreement_proxy]

All elements normalised to [0, 1].
Confidence derived from data completeness and recency.

Usage in agent:
    Useful for outdoor event markets (game attendance, weather-dependent
    outcomes). Rain probability and temperature extremes affect live sports
    markets. Cross-signal with sportsbook odds for weather-adjusted edge.
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
    WEATHER_JSONL,
    data_completeness_confidence,
    load_jsonl,
    mean,
    recency_minutes,
    parse_ts,
    std,
)

logger = logging.getLogger(__name__)

_VECTOR_LEN = 5
_MIN_ROWS = 1  # Weather is useful even with a single row


class WeatherProbabilityTool(BaseTool):
    """
    Reads weather forecast snapshots and returns a 5-element signal vector.

    Deterministic: identical snapshot file + identical location -> identical output.
    No API calls. No randomness.

    Location matching: searches for location_query or location_key containing
    the market_id's team city if available, otherwise uses the most recent
    entry regardless of location.

    Output vector semantics:
      [0] forecast_probability    : rain chance [0,1] from most relevant snapshot
      [1] precipitation_mm        : total precipitation normalised (/50mm, clamped [0,1])
      [2] temp_anomaly            : temp deviation from baseline, shifted to [0,1]
      [3] forecast_confidence     : daily_chance_of_rain / 100 (0=uncertain, 1=certain)
      [4] model_disagreement_proxy: std dev of hourly precip, normalised (/5mm)
    """

    def __init__(self, jsonl_path: Optional[Path] = None) -> None:
        self._jsonl_path = jsonl_path or WEATHER_JSONL

    @property
    def name(self) -> str:
        return "weather_probability_tool"

    @property
    def description(self) -> str:
        return (
            "Weather forecast signal. "
            "Output: [forecast_probability, precipitation_mm, temp_anomaly, "
            "forecast_confidence, model_disagreement_proxy]. "
            "Useful for outdoor events and weather-sensitive market outcomes. "
            "Reads weather_snapshots.jsonl. No API calls."
        )

    def run(self, event: EventInput, **kwargs: Any) -> ToolOutput:
        location_hint: str = kwargs.get("location", "")

        rows = load_jsonl(self._jsonl_path)

        if not rows:
            logger.info("%s: no weather data available — returning zeros", self.name)
            return ToolOutput(
                tool_name=self.name,
                output_vector=[0.0] * _VECTOR_LEN,
                metadata={
                    "data_points_used": 0,
                    "data_recency_minutes": None,
                    "confidence": 0.0,
                    "location_matched": None,
                },
            )

        # Filter to most relevant location if hint provided
        filtered = rows
        if location_hint:
            hint_lower = location_hint.lower()
            location_rows = [
                r for r in rows
                if hint_lower in (r.get("location_key") or "").lower()
                or hint_lower in (r.get("location_query") or "").lower()
                or hint_lower in (r.get("location_name") or "").lower()
            ]
            if location_rows:
                filtered = location_rows

        # Sort by forecast_date descending; take the most recent forecast day
        filtered_sorted = sorted(
            filtered,
            key=lambda r: (r.get("forecast_date", ""), r.get("collected_at", "")),
            reverse=True,
        )

        # Use the single most-relevant row for today's/tomorrow's forecast
        best = filtered_sorted[0]

        forecast_prob = float(best.get("forecast_probability", 0.0))
        precip_mm = float(best.get("total_precip_mm", 0.0))
        temp_anomaly_raw = float(best.get("temp_anomaly_c", 0.0))
        daily_rain_chance = float(best.get("daily_chance_of_rain", 0)) / 100.0
        model_disagree = float(best.get("model_disagreement_proxy", 0.0))

        # Normalise
        norm_precip = min(1.0, max(0.0, precip_mm / 50.0))
        norm_temp = min(1.0, max(0.0, (temp_anomaly_raw / 20.0) + 0.5))  # ±10°C typical
        norm_disagree = min(1.0, max(0.0, model_disagree / 5.0))

        output_vector = [
            round(forecast_prob, 6),
            round(norm_precip, 6),
            round(norm_temp, 6),
            round(daily_rain_chance, 6),
            round(norm_disagree, 6),
        ]

        # Recency
        last_collected = parse_ts(best.get("collected_at"))
        recency_min = recency_minutes(last_collected) if last_collected else 99999.0

        confidence = data_completeness_confidence(
            actual=len(filtered),
            expected=5,
            recency_minutes_val=recency_min,
            max_stale_minutes=360.0,  # 6h — weather updates frequently
        )

        location_matched = best.get("location_name") or best.get("location_key")
        logger.info(
            "%s: location=%s n=%d recency=%.1fmin confidence=%.3f vector=%s",
            self.name, location_matched, len(filtered), recency_min, confidence, output_vector,
        )

        return ToolOutput(
            tool_name=self.name,
            output_vector=output_vector,
            metadata={
                "data_points_used": len(filtered),
                "data_recency_minutes": round(recency_min, 1),
                "confidence": confidence,
                "location_matched": location_matched,
                "forecast_date": best.get("forecast_date"),
            },
        )
