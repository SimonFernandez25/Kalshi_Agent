"""
Price Jump Detector Tool
========================
Detect and measure price jumps from stored JSONL snapshots.

Reads  : outputs/market_snapshots.jsonl
Writes : nothing (pure read-only computation)
APIs   : none
Random : none

Outputs a 4-element vector:
    [max_jump, mean_jump, jump_count, jump_density]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from schemas import EventInput, ToolOutput
from tools.base_tool import BaseTool
from tools._snapshot_helpers import (
    DEFAULT_JSONL,
    extract_prices,
    load_rows,
)

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 5
_VECTOR_LEN = 4
_JUMP_THRESHOLD = 0.05


class PriceJumpDetectorTool(BaseTool):
    """
    Detect and quantify consecutive price jumps for a single market_id
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
        return "price_jump_detector_tool"

    @property
    def description(self) -> str:
        return (
            "Detects price jumps and computes max jump, mean jump, "
            "jump count (>|0.05|), and jump density from local market "
            "snapshots. Deterministic numeric feature vector for agent weighting."
        )

    def run(self, event: EventInput, **kwargs: Any) -> ToolOutput:
        market_id: str = event.market_id
        window_minutes: int = int(kwargs.get("window_minutes", 120))

        rows = load_rows(self._jsonl_path, market_id, window_minutes)
        prices = extract_prices(rows)
        sample_count = len(prices)

        if sample_count < _MIN_SAMPLES:
            logger.info(
                "%s: only %d price points for %s — returning zeros",
                self.name, sample_count, market_id,
            )
            return ToolOutput(
                tool_name=self.name,
                output_vector=[0.0] * _VECTOR_LEN,
                metadata={"confidence": 0.0, "sample_count": sample_count},
            )

        diffs = self._compute_diffs(prices)
        abs_diffs = [abs(d) for d in diffs]
        n_steps = len(diffs)

        max_jump = max(abs_diffs) if abs_diffs else 0.0
        mean_jump = sum(abs_diffs) / n_steps if n_steps > 0 else 0.0
        jump_count = sum(1 for d in abs_diffs if d > _JUMP_THRESHOLD)
        jump_density = jump_count / n_steps if n_steps > 0 else 0.0
        confidence = min(1.0, sample_count / 50)

        output_vector = [
            round(max_jump, 8),
            round(mean_jump, 8),
            round(float(jump_count), 8),
            round(jump_density, 8),
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_diffs(prices: List[float]) -> List[float]:
        """Consecutive price differences: prices[i+1] - prices[i]."""
        return [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
