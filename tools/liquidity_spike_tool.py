"""
Liquidity Spike Tool
====================
Detect liquidity spikes from stored JSONL snapshots.

Reads  : outputs/market_snapshots.jsonl
Writes : nothing (pure read-only computation)
APIs   : none
Random : none

Outputs a 4-element vector:
    [mean_liquidity, std_liquidity, latest_vs_mean_ratio, zscore_latest]
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Any, Optional

from schemas import EventInput, ToolOutput
from tools.base_tool import BaseTool
from tools._snapshot_helpers import (
    DEFAULT_JSONL,
    extract_liquidity,
    load_rows,
)

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 5
_VECTOR_LEN = 4


class LiquiditySpikeTool(BaseTool):
    """
    Detect and quantify liquidity spikes for a single market_id using
    stored snapshot data.  Pure math — no APIs, no randomness.

    Uses open_interest if present, else volume.

    Deterministic: identical JSONL + identical inputs → identical output.
    """

    deterministic: bool = True

    def __init__(self, jsonl_path: Optional[Path] = None) -> None:
        self._jsonl_path = jsonl_path or DEFAULT_JSONL

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "liquidity_spike_tool"

    @property
    def description(self) -> str:
        return (
            "Computes mean liquidity, std liquidity, latest-vs-mean ratio, "
            "and z-score of latest observation from local market snapshots. "
            "Deterministic numeric feature vector for agent weighting."
        )

    def run(self, event: EventInput, **kwargs: Any) -> ToolOutput:
        market_id: str = event.market_id
        window_minutes: int = int(kwargs.get("window_minutes", 120))

        rows = load_rows(self._jsonl_path, market_id, window_minutes)
        liq_series = extract_liquidity(rows)
        sample_count = len(liq_series)

        if sample_count < _MIN_SAMPLES:
            logger.info(
                "%s: only %d liquidity points for %s — returning zeros",
                self.name, sample_count, market_id,
            )
            return ToolOutput(
                tool_name=self.name,
                output_vector=[0.0] * _VECTOR_LEN,
                metadata={"confidence": 0.0, "sample_count": sample_count},
            )

        mean_liq = sum(liq_series) / len(liq_series)
        std_liq = statistics.stdev(liq_series) if len(liq_series) >= 2 else 0.0
        latest = liq_series[-1]

        latest_vs_mean_ratio = (latest / mean_liq) if mean_liq != 0 else 0.0
        zscore_latest = ((latest - mean_liq) / std_liq) if std_liq != 0 else 0.0
        confidence = min(1.0, sample_count / 50)

        output_vector = [
            round(mean_liq, 4),
            round(std_liq, 4),
            round(latest_vs_mean_ratio, 8),
            round(zscore_latest, 8),
        ]

        logger.info(
            "%s: market=%s samples=%d confidence=%.2f vector=%s",
            self.name, market_id, sample_count, confidence, output_vector,
        )

        return ToolOutput(
            tool_name=self.name,
            output_vector=output_vector,
            metadata={
                "confidence": round(confidence, 4),
                "sample_count": sample_count,
            },
        )
