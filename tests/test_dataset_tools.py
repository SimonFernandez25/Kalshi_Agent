"""
Tests for the three dataset-only deterministic tools:
  - SpreadCompressionTool
  - PriceJumpDetectorTool
  - LiquiditySpikeTool

Uses a small mock JSONL with 20 rows for market DST-001, written to a
temp file.  Verifies vector lengths, confidence > 0, determinism, edge
cases, and graceful handling of missing fields.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from prediction_agent.schemas import EventInput
from prediction_agent.tools.spread_compression_tool import SpreadCompressionTool
from prediction_agent.tools.price_jump_detector_tool import PriceJumpDetectorTool
from prediction_agent.tools.liquidity_spike_tool import LiquiditySpikeTool


# ------------------------------------------------------------------ #
# Helper: generate timestamps N minutes ago from now
# ------------------------------------------------------------------ #
def _ts(minutes_ago: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.isoformat()


# ------------------------------------------------------------------ #
# Build a 20-row mock JSONL
# ------------------------------------------------------------------ #
def _build_mock_jsonl(path: Path) -> None:
    """
    20 rows for market DST-001, spread over the last 100 minutes.
    Prices rise from 0.40 → 0.59, with a deliberate large jump at row 10.
    Spreads narrow from 0.08 down to 0.04 (compression).
    Open interest grows from 100 → 290.
    """
    rows = []
    base_price = 0.40
    base_spread = 0.08
    base_oi = 100

    for i in range(20):
        minutes_ago = 100 - i * 5  # 100, 95, 90, … 5
        price = base_price + i * 0.01
        # Inject a large jump at row 10
        if i == 10:
            price = base_price + 0.15  # jump from 0.49 → 0.55
        spread = base_spread - i * 0.002  # narrowing: 0.08 → 0.042
        if spread < 0.01:
            spread = 0.01
        bid = round(price - spread / 2, 4)
        ask = round(price + spread / 2, 4)
        oi = base_oi + i * 10

        rows.append({
            "timestamp": _ts(minutes_ago),
            "market_id": "DST-001",
            "last_price": round(price, 4),
            "yes_bid": bid,
            "yes_ask": ask,
            "volume": 500 + i * 20,
            "open_interest": oi,
        })

    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def mock_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "market_snapshots.jsonl"
    _build_mock_jsonl(p)
    return p


@pytest.fixture
def event() -> EventInput:
    return EventInput(
        event_id="evt-dst",
        market_id="DST-001",
        market_title="Dataset Test Market",
        current_price=0.50,
    )


@pytest.fixture
def unknown_event() -> EventInput:
    return EventInput(
        event_id="evt-unknown",
        market_id="NONEXISTENT-999",
        market_title="Ghost Market",
        current_price=0.50,
    )


# ================================================================== #
# SPREAD COMPRESSION TOOL
# ================================================================== #
class TestSpreadCompressionTool:

    def test_vector_length(self, mock_jsonl, event):
        tool = SpreadCompressionTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert len(result.output_vector) == 4

    def test_tool_name(self, mock_jsonl, event):
        tool = SpreadCompressionTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert result.tool_name == "spread_compression_tool"

    def test_confidence_positive(self, mock_jsonl, event):
        tool = SpreadCompressionTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert result.metadata["confidence"] > 0

    def test_confidence_value(self, mock_jsonl, event):
        tool = SpreadCompressionTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        # 20 samples → min(1.0, 20/50) = 0.4
        assert result.metadata["confidence"] == pytest.approx(0.4, abs=0.01)

    def test_sample_count(self, mock_jsonl, event):
        tool = SpreadCompressionTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert result.metadata["sample_count"] == 20

    def test_deterministic(self, mock_jsonl, event):
        tool = SpreadCompressionTool(jsonl_path=mock_jsonl)
        r1 = tool.run(event, window_minutes=999_999)
        r2 = tool.run(event, window_minutes=999_999)
        assert r1.output_vector == r2.output_vector

    def test_unknown_market_zeros(self, mock_jsonl, unknown_event):
        tool = SpreadCompressionTool(jsonl_path=mock_jsonl)
        result = tool.run(unknown_event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0]
        assert result.metadata["confidence"] == 0.0

    def test_mean_spread_positive(self, mock_jsonl, event):
        tool = SpreadCompressionTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert result.output_vector[0] > 0, "Mean spread should be positive"

    def test_compression_ratio_below_one(self, mock_jsonl, event):
        """Spreads narrow → last < mean → ratio < 1."""
        tool = SpreadCompressionTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        compression_ratio = result.output_vector[3]
        assert compression_ratio < 1.0, "Narrowing spreads should yield ratio < 1"

    def test_spread_trend_negative(self, mock_jsonl, event):
        """Spreads narrow → last − first < 0."""
        tool = SpreadCompressionTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        spread_trend = result.output_vector[2]
        assert spread_trend < 0, "Narrowing spreads should give negative trend"


# ================================================================== #
# PRICE JUMP DETECTOR TOOL
# ================================================================== #
class TestPriceJumpDetectorTool:

    def test_vector_length(self, mock_jsonl, event):
        tool = PriceJumpDetectorTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert len(result.output_vector) == 4

    def test_tool_name(self, mock_jsonl, event):
        tool = PriceJumpDetectorTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert result.tool_name == "price_jump_detector_tool"

    def test_confidence_positive(self, mock_jsonl, event):
        tool = PriceJumpDetectorTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert result.metadata["confidence"] > 0

    def test_confidence_value(self, mock_jsonl, event):
        tool = PriceJumpDetectorTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert result.metadata["confidence"] == pytest.approx(0.4, abs=0.01)

    def test_deterministic(self, mock_jsonl, event):
        tool = PriceJumpDetectorTool(jsonl_path=mock_jsonl)
        r1 = tool.run(event, window_minutes=999_999)
        r2 = tool.run(event, window_minutes=999_999)
        assert r1.output_vector == r2.output_vector

    def test_unknown_market_zeros(self, mock_jsonl, unknown_event):
        tool = PriceJumpDetectorTool(jsonl_path=mock_jsonl)
        result = tool.run(unknown_event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0]

    def test_max_jump_detects_large_jump(self, mock_jsonl, event):
        tool = PriceJumpDetectorTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        max_jump = result.output_vector[0]
        # Row 10 has a jump of ~0.06 (0.49 → 0.55)
        assert max_jump > 0.05, "Should detect the injected large jump"

    def test_jump_count_at_least_one(self, mock_jsonl, event):
        tool = PriceJumpDetectorTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        jump_count = result.output_vector[2]
        assert jump_count >= 1.0, "At least one jump > 0.05 was injected"

    def test_jump_density_in_range(self, mock_jsonl, event):
        tool = PriceJumpDetectorTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        density = result.output_vector[3]
        assert 0.0 <= density <= 1.0

    def test_mean_jump_positive(self, mock_jsonl, event):
        tool = PriceJumpDetectorTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        mean_jump = result.output_vector[1]
        assert mean_jump > 0, "Prices change, so mean jump must be > 0"


# ================================================================== #
# LIQUIDITY SPIKE TOOL
# ================================================================== #
class TestLiquiditySpikeTool:

    def test_vector_length(self, mock_jsonl, event):
        tool = LiquiditySpikeTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert len(result.output_vector) == 4

    def test_tool_name(self, mock_jsonl, event):
        tool = LiquiditySpikeTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert result.tool_name == "liquidity_spike_tool"

    def test_confidence_positive(self, mock_jsonl, event):
        tool = LiquiditySpikeTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert result.metadata["confidence"] > 0

    def test_confidence_value(self, mock_jsonl, event):
        tool = LiquiditySpikeTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        assert result.metadata["confidence"] == pytest.approx(0.4, abs=0.01)

    def test_deterministic(self, mock_jsonl, event):
        tool = LiquiditySpikeTool(jsonl_path=mock_jsonl)
        r1 = tool.run(event, window_minutes=999_999)
        r2 = tool.run(event, window_minutes=999_999)
        assert r1.output_vector == r2.output_vector

    def test_unknown_market_zeros(self, mock_jsonl, unknown_event):
        tool = LiquiditySpikeTool(jsonl_path=mock_jsonl)
        result = tool.run(unknown_event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0]

    def test_mean_liquidity_correct(self, mock_jsonl, event):
        tool = LiquiditySpikeTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        mean_liq = result.output_vector[0]
        # OI: 100, 110, 120, …, 290 → mean = 195
        assert mean_liq == pytest.approx(195.0, abs=1.0)

    def test_latest_vs_mean_ratio_above_one(self, mock_jsonl, event):
        """Latest OI (290) > mean (195) → ratio > 1."""
        tool = LiquiditySpikeTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        ratio = result.output_vector[2]
        assert ratio > 1.0

    def test_zscore_latest_positive(self, mock_jsonl, event):
        """Latest OI is well above mean → positive z-score."""
        tool = LiquiditySpikeTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        zscore = result.output_vector[3]
        assert zscore > 0

    def test_std_liquidity_positive(self, mock_jsonl, event):
        tool = LiquiditySpikeTool(jsonl_path=mock_jsonl)
        result = tool.run(event, window_minutes=999_999)
        std_liq = result.output_vector[1]
        assert std_liq > 0, "OI varies, so std should be > 0"


# ================================================================== #
# CROSS-CUTTING: missing fields
# ================================================================== #
class TestMissingFields:
    """Verify tools don't crash on rows with missing fields."""

    def test_spread_tool_no_bid_ask(self, tmp_path):
        """Rows without yes_bid / yes_ask → spread extraction yields empty → zeros."""
        p = tmp_path / "noask.jsonl"
        rows = [
            {"timestamp": _ts(i), "market_id": "NB-001", "last_price": 0.5, "volume": 100}
            for i in range(10, 0, -1)
        ]
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        tool = SpreadCompressionTool(jsonl_path=p)
        event = EventInput(event_id="e", market_id="NB-001", market_title="m", current_price=0.5)
        result = tool.run(event, window_minutes=999_999)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0]

    def test_jump_tool_no_price(self, tmp_path):
        """Rows without last_price or bid/ask → prices empty → zeros."""
        p = tmp_path / "noprice.jsonl"
        rows = [
            {"timestamp": _ts(i), "market_id": "NP-001", "volume": 100}
            for i in range(10, 0, -1)
        ]
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        tool = PriceJumpDetectorTool(jsonl_path=p)
        event = EventInput(event_id="e", market_id="NP-001", market_title="m", current_price=0.5)
        result = tool.run(event, window_minutes=999_999)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0]

    def test_liquidity_tool_no_oi_no_volume(self, tmp_path):
        """Rows without open_interest or volume → zeros."""
        p = tmp_path / "noliq.jsonl"
        rows = [
            {"timestamp": _ts(i), "market_id": "NL-001", "last_price": 0.5}
            for i in range(10, 0, -1)
        ]
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        tool = LiquiditySpikeTool(jsonl_path=p)
        event = EventInput(event_id="e", market_id="NL-001", market_title="m", current_price=0.5)
        result = tool.run(event, window_minutes=999_999)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0]

    def test_liquidity_tool_falls_back_to_volume(self, tmp_path):
        """When open_interest is absent, uses volume."""
        p = tmp_path / "volonly.jsonl"
        rows = [
            {"timestamp": _ts(i), "market_id": "VO-001", "last_price": 0.5, "volume": 100 + i * 10}
            for i in range(10, 0, -1)
        ]
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        tool = LiquiditySpikeTool(jsonl_path=p)
        event = EventInput(event_id="e", market_id="VO-001", market_title="m", current_price=0.5)
        result = tool.run(event, window_minutes=999_999)
        assert result.output_vector[0] > 0, "Should use volume as fallback"

    def test_missing_file(self, tmp_path):
        """All tools handle missing JSONL gracefully."""
        ghost = tmp_path / "ghost.jsonl"
        event = EventInput(event_id="e", market_id="X", market_title="m", current_price=0.5)

        for ToolClass in [SpreadCompressionTool, PriceJumpDetectorTool, LiquiditySpikeTool]:
            tool = ToolClass(jsonl_path=ghost)
            result = tool.run(event)
            assert result.output_vector == [0.0, 0.0, 0.0, 0.0]
            assert result.metadata["confidence"] == 0.0


# ================================================================== #
# REGISTRY INTEGRATION
# ================================================================== #
class TestRegistryIntegration:
    """All three tools appear in the default registry."""

    def test_all_tools_registered(self):
        from prediction_agent.tools.registry import build_default_registry
        registry = build_default_registry()
        for name in [
            "spread_compression_tool",
            "price_jump_detector_tool",
            "liquidity_spike_tool",
        ]:
            assert name in registry, f"{name} not in registry"
