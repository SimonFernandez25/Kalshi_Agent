"""
Spread Compression Tool
=======================
Compute bid-ask spread behaviour metrics from stored JSONL snapshots.

Reads  : outputs/market_snapshots.jsonl
Writes : nothing (pure read-only computation)
APIs   : none
Random : none

Outputs a 4-element vector:
    [mean_spread, spread_std, spread_trend, compression_ratio]
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Any, Optional

from prediction_agent.schemas import EventInput, ToolOutput
from prediction_agent.tools.base_tool import BaseTool
from prediction_agent.tools._snapshot_helpers import (
    DEFAULT_JSONL,
    extract_spreads,
    load_rows,
)

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 3
_VECTOR_LEN = 4


class SpreadCompressionTool(BaseTool):
    """
    Analyse bid-ask spread compression / expansion for a single market_id
    using stored snapshot data.  Pure math — no APIs, no randomness.

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
        return "spread_compression_tool"

    @property
    def description(self) -> str:
        return (
            "Computes mean spread, spread std-dev, spread trend, and "
            "compression ratio from local market snapshots. "
            "Deterministic numeric feature vector for agent weighting."
        )

    def run(self, event: EventInput, **kwargs: Any) -> ToolOutput:
        market_id: str = event.market_id
        window_minutes: int = int(kwargs.get("window_minutes", 120))

        rows = load_rows(self._jsonl_path, market_id, window_minutes)
        spreads = extract_spreads(rows)
        sample_count = len(spreads)

        if sample_count < _MIN_SAMPLES:
            logger.info(
                "%s: only %d spread points for %s — returning zeros",
                self.name, sample_count, market_id,
            )
            return ToolOutput(
                tool_name=self.name,
                output_vector=[0.0] * _VECTOR_LEN,
                metadata={"confidence": 0.0, "sample_count": sample_count},
            )

        mean_spread = sum(spreads) / len(spreads)
        spread_std = statistics.stdev(spreads) if len(spreads) >= 2 else 0.0
        spread_trend = spreads[-1] - spreads[0]
        compression_ratio = (spreads[-1] / mean_spread) if mean_spread != 0 else 0.0
        confidence = min(1.0, sample_count / 50)

        output_vector = [
            round(mean_spread, 8),
            round(spread_std, 8),
            round(spread_trend, 8),
            round(compression_ratio, 8),
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
