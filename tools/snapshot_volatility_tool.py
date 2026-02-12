"""
Snapshot Volatility Tool
========================
Compute deterministic market behavior metrics from stored JSONL snapshots.

Reads  : outputs/market_snapshots.jsonl
Writes : nothing (pure read-only computation)
APIs   : none
Random : none

Outputs a 5-element vector:
    [volatility, price_range, mean_spread, jump_rate, liquidity_proxy]
"""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from prediction_agent.schemas import EventInput, ToolOutput
from prediction_agent.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

# Default path relative to package root
_DEFAULT_JSONL = Path(__file__).resolve().parent.parent / "outputs" / "market_snapshots.jsonl"

# Minimum data points required for meaningful computation
_MIN_SAMPLES = 3


class SnapshotVolatilityTool(BaseTool):
    """
    Compute simple market behavior metrics for a single market_id
    using stored snapshot data.  Pure math — no APIs, no randomness.

    Deterministic: identical JSONL + identical inputs → identical output.
    """

    deterministic: bool = True

    def __init__(self, jsonl_path: Optional[Path] = None) -> None:
        self._jsonl_path = jsonl_path or _DEFAULT_JSONL

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "snapshot_volatility_tool"

    @property
    def description(self) -> str:
        return (
            "Computes volatility, price range, mean spread, jump rate, "
            "and liquidity proxy from local market snapshots. "
            "Deterministic numeric feature vector for agent weighting."
        )

    def run(self, event: EventInput, **kwargs: Any) -> ToolOutput:
        """
        Execute the tool.

        Parameters pulled from *event* and *kwargs*:
            market_id      — from event.market_id
            window_minutes — from kwargs (default 120)
        """
        market_id: str = event.market_id
        window_minutes: int = int(kwargs.get("window_minutes", 120))

        rows = self._load_rows(market_id, window_minutes)
        prices = self._extract_prices(rows)
        sample_count = len(prices)

        if sample_count < _MIN_SAMPLES:
            logger.info(
                "snapshot_volatility_tool: only %d price points for %s — returning zeros",
                sample_count,
                market_id,
            )
            return ToolOutput(
                tool_name=self.name,
                output_vector=[0.0, 0.0, 0.0, 0.0, 0.0],
                metadata={"confidence": 0.0, "sample_count": sample_count},
            )

        volatility = self._compute_volatility(prices)
        price_range = self._compute_price_range(prices)
        mean_spread = self._compute_mean_spread(rows)
        jump_rate = self._compute_jump_rate(prices)
        liquidity_proxy = self._compute_liquidity_proxy(rows)
        confidence = min(1.0, sample_count / 50)

        output_vector = [
            round(volatility, 8),
            round(price_range, 8),
            round(mean_spread, 8),
            round(jump_rate, 8),
            round(liquidity_proxy, 4),
        ]

        logger.info(
            "snapshot_volatility_tool: market=%s samples=%d confidence=%.2f vector=%s",
            market_id,
            sample_count,
            confidence,
            output_vector,
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
    # Data loading
    # ------------------------------------------------------------------
    def _load_rows(self, market_id: str, window_minutes: int) -> List[Dict[str, Any]]:
        """Read JSONL and return rows matching market_id within the time window."""
        if not self._jsonl_path.exists():
            logger.warning("Snapshot file not found: %s", self._jsonl_path)
            return []

        now = datetime.now(timezone.utc)
        cutoff_seconds = window_minutes * 60
        matched: List[Dict[str, Any]] = []

        with open(self._jsonl_path, "r", encoding="utf-8") as fh:
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

                # Parse timestamp
                ts = self._parse_timestamp(row.get("timestamp"))
                if ts is None:
                    continue

                # Window filter
                age_seconds = (now - ts).total_seconds()
                if age_seconds > cutoff_seconds:
                    continue

                row["_parsed_ts"] = ts
                matched.append(row)

        # Sort by timestamp ascending
        matched.sort(key=lambda r: r["_parsed_ts"])
        return matched

    # ------------------------------------------------------------------
    # Price extraction (PRICE RULE)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_prices(rows: List[Dict[str, Any]]) -> List[float]:
        """
        Extract price series.
        Priority: last_price > midpoint(yes_bid, yes_ask) > skip.
        """
        prices: List[float] = []
        for row in rows:
            price = SnapshotVolatilityTool._row_price(row)
            if price is not None:
                prices.append(price)
        return prices

    @staticmethod
    def _row_price(row: Dict[str, Any]) -> Optional[float]:
        """Return usable price for a single row, or None."""
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

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_volatility(prices: List[float]) -> float:
        """Standard deviation of price series."""
        if len(prices) < 2:
            return 0.0
        return statistics.stdev(prices)

    @staticmethod
    def _compute_price_range(prices: List[float]) -> float:
        """max(price) - min(price)."""
        if not prices:
            return 0.0
        return max(prices) - min(prices)

    @staticmethod
    def _compute_mean_spread(rows: List[Dict[str, Any]]) -> float:
        """Average of (yes_ask - yes_bid) across rows where both exist."""
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
        if not spreads:
            return 0.0
        return sum(spreads) / len(spreads)

    @staticmethod
    def _compute_jump_rate(prices: List[float]) -> float:
        """Fraction of consecutive price changes > 0.05."""
        if len(prices) < 2:
            return 0.0
        jumps = 0
        pairs = len(prices) - 1
        for i in range(pairs):
            if abs(prices[i + 1] - prices[i]) > 0.05:
                jumps += 1
        return jumps / pairs

    @staticmethod
    def _compute_liquidity_proxy(rows: List[Dict[str, Any]]) -> float:
        """Mean of open_interest (if present) else volume."""
        values: List[float] = []
        for row in rows:
            oi = row.get("open_interest")
            if oi is not None and isinstance(oi, (int, float)):
                values.append(float(oi))
            else:
                vol = row.get("volume")
                if vol is not None and isinstance(vol, (int, float)):
                    values.append(float(vol))
        if not values:
            return 0.0
        return sum(values) / len(values)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_timestamp(raw: Any) -> Optional[datetime]:
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
