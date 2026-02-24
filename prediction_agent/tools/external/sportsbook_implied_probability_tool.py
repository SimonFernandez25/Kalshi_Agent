"""
Sportsbook Implied Probability Tool
=====================================
Deterministic tool that reads odds_snapshots.jsonl and returns
cross-book implied probability signal vectors.

Reads  : outputs/external/odds_snapshots.jsonl
Writes : nothing
APIs   : none
Random : none

Output vector (4 elements):
    [mean_implied_probability, cross_book_variance, line_movement_rate,
     implied_volatility_proxy]

All elements normalised to [0, 1].
Confidence derived from data completeness and recency.

Usage in agent:
    PROBABILITY RECONCILIATION SIGNAL — when scoring in probability_edge mode,
    mean_implied_probability is used as the model's probability estimate.
    edge = mean_implied_probability - current_market_price
    Trigger when abs(edge) > threshold.

    Useful for any sports market (NBA, NFL) where Kalshi price may diverge
    from bookmaker consensus. High cross_book_variance = uncertain/moving line.
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
    ODDS_JSONL,
    data_completeness_confidence,
    load_jsonl,
    mean,
    recency_minutes,
    parse_ts,
    std,
    volatility,
)

logger = logging.getLogger(__name__)

_VECTOR_LEN = 4
_MIN_ROWS = 1


class SportsbookImpliedProbabilityTool(BaseTool):
    """
    Reads sportsbook odds snapshots and returns a 4-element signal vector.

    Deterministic: identical snapshot file + identical event context -> identical output.
    No API calls. No randomness.

    Event matching: searches odds_snapshots.jsonl for rows whose home_team or
    away_team appears in the market_title or market_id. Falls back to all rows
    for the most recent collect time if no match found.

    Output vector semantics:
      [0] mean_implied_probability : vig-removed consensus home win prob across books [0,1]
      [1] cross_book_variance      : variance in implied probs across bookmakers (normalised /0.1)
      [2] line_movement_rate       : change in mean_implied_prob between first/last snapshot [0,1]
      [3] implied_volatility_proxy : std dev of per-book implied probs [0,1]
    """

    def __init__(self, jsonl_path: Optional[Path] = None) -> None:
        self._jsonl_path = jsonl_path or ODDS_JSONL

    @property
    def name(self) -> str:
        return "sportsbook_implied_probability_tool"

    @property
    def description(self) -> str:
        return (
            "Sportsbook consensus implied probability signal. "
            "Output: [mean_implied_probability, cross_book_variance, "
            "line_movement_rate, implied_volatility_proxy]. "
            "output_vector[0] is a direct probability estimate for probability_edge scoring mode. "
            "Reads odds_snapshots.jsonl. No API calls."
        )

    def run(self, event: EventInput, **kwargs: Any) -> ToolOutput:
        target_team: str = kwargs.get("team", "")

        rows = load_jsonl(self._jsonl_path)

        if not rows:
            return ToolOutput(
                tool_name=self.name,
                output_vector=[0.0] * _VECTOR_LEN,
                metadata={
                    "data_points_used": 0,
                    "data_recency_minutes": None,
                    "confidence": 0.0,
                    "matched_event_id": None,
                },
            )

        # Try to match by market title / market_id content or explicit team kwarg
        search_terms = []
        if target_team:
            search_terms.append(target_team.lower())
        # Extract city-like tokens from market_id / title (e.g. "LAL" "BOS")
        for token in event.market_id.replace("-", " ").split() + event.market_title.replace("-", " ").split():
            if len(token) >= 3:
                search_terms.append(token.lower())

        matched = []
        if search_terms:
            for r in rows:
                home = (r.get("home_team") or "").lower()
                away = (r.get("away_team") or "").lower()
                event_id = (r.get("event_id") or "").lower()
                if any(t in home or t in away or t in event_id for t in search_terms):
                    matched.append(r)

        # Fallback: use all rows from the most recent collected_hour
        if not matched:
            hours = sorted(set(r.get("collected_hour", "") for r in rows), reverse=True)
            if hours:
                latest_hour = hours[0]
                matched = [r for r in rows if r.get("collected_hour") == latest_hour]
            else:
                matched = rows

        n = len(matched)

        # Pull home_implied_prob for each bookmaker row
        home_probs = [
            float(r["home_implied_prob"])
            for r in matched
            if isinstance(r.get("home_implied_prob"), (int, float))
        ]

        if not home_probs:
            return ToolOutput(
                tool_name=self.name,
                output_vector=[0.0] * _VECTOR_LEN,
                metadata={
                    "data_points_used": n,
                    "data_recency_minutes": None,
                    "confidence": 0.0,
                    "matched_event_id": None,
                },
            )

        mean_prob = mean(home_probs)
        variance = (std(home_probs) ** 2) if len(home_probs) > 1 else 0.0

        # Line movement: compare first vs last batch by collected_at
        sorted_matched = sorted(matched, key=lambda r: r.get("collected_at", ""))
        first_hour_probs = [
            float(r["home_implied_prob"])
            for r in sorted_matched[:max(1, n // 2)]
            if isinstance(r.get("home_implied_prob"), (int, float))
        ]
        last_hour_probs = [
            float(r["home_implied_prob"])
            for r in sorted_matched[max(1, n // 2):]
            if isinstance(r.get("home_implied_prob"), (int, float))
        ]
        if first_hour_probs and last_hour_probs:
            line_movement = mean(last_hour_probs) - mean(first_hour_probs)
        else:
            line_movement = 0.0

        impl_vol = std(home_probs)

        # Normalise
        norm_variance = min(1.0, max(0.0, variance / 0.01))      # variance up to 0.01 (10% disagreement)
        norm_movement = min(1.0, max(0.0, (line_movement / 0.1) + 0.5))  # ±5% movement
        norm_impl_vol = min(1.0, max(0.0, impl_vol / 0.1))       # std up to 10%

        output_vector = [
            round(min(1.0, max(0.0, mean_prob)), 6),
            round(norm_variance, 6),
            round(norm_movement, 6),
            round(norm_impl_vol, 6),
        ]

        # Recency
        last_collected = parse_ts(sorted_matched[-1].get("collected_at") if sorted_matched else None)
        recency_min = recency_minutes(last_collected) if last_collected else 99999.0
        matched_id = sorted_matched[-1].get("event_id") if sorted_matched else None

        confidence = data_completeness_confidence(
            actual=n,
            expected=10,  # expect ~10 bookmaker rows per event
            recency_minutes_val=recency_min,
            max_stale_minutes=120.0,  # 2h — odds move fast pre-game
        )

        logger.info(
            "%s: n=%d mean_prob=%.4f recency=%.1fmin confidence=%.3f vector=%s",
            self.name, n, mean_prob, recency_min, confidence, output_vector,
        )

        return ToolOutput(
            tool_name=self.name,
            output_vector=output_vector,
            metadata={
                "data_points_used": n,
                "data_recency_minutes": round(recency_min, 1),
                "confidence": confidence,
                "matched_event_id": matched_id,
                "mean_implied_probability": round(mean_prob, 6),
                "bookmakers_count": len(set(r.get("bookmaker") for r in matched)),
            },
        )
