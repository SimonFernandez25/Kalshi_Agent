"""
Tests for SnapshotVolatilityTool.

Uses a small mock JSONL (10 rows) written to a temp file.
Verifies:
  - output_vector has length 5
  - confidence > 0 when enough data
  - zero-output for unknown market
  - zero-output when fewer than 3 rows
  - deterministic (same input → same output)
  - price rule: falls back to midpoint when last_price missing
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from prediction_agent.schemas import EventInput
from prediction_agent.tools.snapshot_volatility_tool import SnapshotVolatilityTool


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

def _make_timestamp(minutes_ago: int) -> str:
    """ISO timestamp `minutes_ago` before now (UTC)."""
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.isoformat()


def _build_mock_jsonl(path: Path) -> None:
    """Write 10 rows of synthetic snapshot data for market TEST-MKT-001."""
    rows = [
        # 10 rows spread over last 60 minutes for TEST-MKT-001
        {"timestamp": _make_timestamp(55), "market_id": "TEST-MKT-001", "last_price": 0.50, "yes_bid": 0.48, "yes_ask": 0.52, "volume": 100, "open_interest": 200},
        {"timestamp": _make_timestamp(50), "market_id": "TEST-MKT-001", "last_price": 0.52, "yes_bid": 0.50, "yes_ask": 0.54, "volume": 120, "open_interest": 210},
        {"timestamp": _make_timestamp(45), "market_id": "TEST-MKT-001", "last_price": 0.48, "yes_bid": 0.46, "yes_ask": 0.50, "volume": 110, "open_interest": 205},
        {"timestamp": _make_timestamp(40), "market_id": "TEST-MKT-001", "last_price": 0.55, "yes_bid": 0.53, "yes_ask": 0.57, "volume": 130, "open_interest": 220},
        {"timestamp": _make_timestamp(35), "market_id": "TEST-MKT-001", "last_price": 0.53, "yes_bid": 0.51, "yes_ask": 0.55, "volume": 125, "open_interest": 215},
        {"timestamp": _make_timestamp(30), "market_id": "TEST-MKT-001", "last_price": 0.60, "yes_bid": 0.58, "yes_ask": 0.62, "volume": 140, "open_interest": 230},
        {"timestamp": _make_timestamp(25), "market_id": "TEST-MKT-001", "last_price": 0.58, "yes_bid": 0.56, "yes_ask": 0.60, "volume": 135, "open_interest": 225},
        {"timestamp": _make_timestamp(20), "market_id": "TEST-MKT-001", "last_price": 0.62, "yes_bid": 0.60, "yes_ask": 0.64, "volume": 150, "open_interest": 240},
        {"timestamp": _make_timestamp(15), "market_id": "TEST-MKT-001", "last_price": 0.61, "yes_bid": 0.59, "yes_ask": 0.63, "volume": 145, "open_interest": 235},
        {"timestamp": _make_timestamp(10), "market_id": "TEST-MKT-001", "last_price": 0.65, "yes_bid": 0.63, "yes_ask": 0.67, "volume": 160, "open_interest": 250},
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


@pytest.fixture
def mock_jsonl(tmp_path: Path) -> Path:
    """Create a temp JSONL with 10 mock rows and return path."""
    p = tmp_path / "market_snapshots.jsonl"
    _build_mock_jsonl(p)
    return p


@pytest.fixture
def tool(mock_jsonl: Path) -> SnapshotVolatilityTool:
    """Tool wired to the mock JSONL."""
    return SnapshotVolatilityTool(jsonl_path=mock_jsonl)


@pytest.fixture
def event() -> EventInput:
    """Minimal EventInput for TEST-MKT-001."""
    return EventInput(
        event_id="evt-test-001",
        market_id="TEST-MKT-001",
        market_title="Test Market",
        current_price=0.60,
    )


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestSnapshotVolatilityTool:
    """Core tool behaviour."""

    def test_output_vector_length(self, tool: SnapshotVolatilityTool, event: EventInput):
        result = tool.run(event)
        assert len(result.output_vector) == 5, "Expected 5-element output vector"

    def test_tool_name(self, tool: SnapshotVolatilityTool, event: EventInput):
        result = tool.run(event)
        assert result.tool_name == "snapshot_volatility_tool"

    def test_confidence_positive(self, tool: SnapshotVolatilityTool, event: EventInput):
        result = tool.run(event)
        confidence = result.metadata["confidence"]
        assert confidence > 0, "Expected confidence > 0 with 10 samples"

    def test_confidence_value(self, tool: SnapshotVolatilityTool, event: EventInput):
        result = tool.run(event)
        # 10 samples → confidence = min(1.0, 10/50) = 0.2
        assert result.metadata["confidence"] == pytest.approx(0.2, abs=0.01)

    def test_sample_count(self, tool: SnapshotVolatilityTool, event: EventInput):
        result = tool.run(event)
        assert result.metadata["sample_count"] == 10

    def test_deterministic(self, tool: SnapshotVolatilityTool, event: EventInput):
        r1 = tool.run(event)
        r2 = tool.run(event)
        assert r1.output_vector == r2.output_vector, "Tool must be deterministic"

    def test_unknown_market_returns_zeros(self, tool: SnapshotVolatilityTool):
        unknown = EventInput(
            event_id="evt-unknown",
            market_id="NONEXISTENT-999",
            market_title="No Such Market",
            current_price=0.50,
        )
        result = tool.run(unknown)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert result.metadata["confidence"] == 0.0

    def test_volatility_positive(self, tool: SnapshotVolatilityTool, event: EventInput):
        result = tool.run(event)
        volatility = result.output_vector[0]
        assert volatility > 0, "Volatility should be > 0 for varying prices"

    def test_price_range_correct(self, tool: SnapshotVolatilityTool, event: EventInput):
        result = tool.run(event)
        price_range = result.output_vector[1]
        # Prices span 0.48 to 0.65 → range = 0.17
        assert price_range == pytest.approx(0.17, abs=0.001)

    def test_mean_spread_correct(self, tool: SnapshotVolatilityTool, event: EventInput):
        result = tool.run(event)
        mean_spread = result.output_vector[2]
        # All spreads are 0.04 → mean = 0.04
        assert mean_spread == pytest.approx(0.04, abs=0.001)

    def test_jump_rate_in_range(self, tool: SnapshotVolatilityTool, event: EventInput):
        result = tool.run(event)
        jump_rate = result.output_vector[3]
        assert 0.0 <= jump_rate <= 1.0

    def test_liquidity_proxy_uses_open_interest(self, tool: SnapshotVolatilityTool, event: EventInput):
        result = tool.run(event)
        liquidity = result.output_vector[4]
        # open_interest values: 200,210,205,220,215,230,225,240,235,250 → mean = 223.0
        assert liquidity == pytest.approx(223.0, abs=1.0)


class TestPriceRule:
    """Verify price extraction fallback logic."""

    def test_midpoint_fallback(self, tmp_path: Path):
        """When last_price is missing, use midpoint of bid/ask."""
        p = tmp_path / "midpoint.jsonl"
        rows = [
            {"timestamp": _make_timestamp(5), "market_id": "MID-001", "yes_bid": 0.40, "yes_ask": 0.60, "volume": 100},
            {"timestamp": _make_timestamp(4), "market_id": "MID-001", "yes_bid": 0.42, "yes_ask": 0.58, "volume": 110},
            {"timestamp": _make_timestamp(3), "market_id": "MID-001", "yes_bid": 0.44, "yes_ask": 0.56, "volume": 120},
        ]
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        tool = SnapshotVolatilityTool(jsonl_path=p)
        event = EventInput(event_id="e", market_id="MID-001", market_title="m", current_price=0.5)
        result = tool.run(event)
        assert len(result.output_vector) == 5
        # Midpoints: 0.50, 0.50, 0.50 → volatility = 0, range = 0
        assert result.output_vector[0] == pytest.approx(0.0, abs=0.001)  # volatility
        assert result.output_vector[1] == pytest.approx(0.0, abs=0.001)  # price_range

    def test_skip_row_when_no_price(self, tmp_path: Path):
        """Rows with no price data are skipped; < 3 usable → zeros."""
        p = tmp_path / "noprice.jsonl"
        rows = [
            {"timestamp": _make_timestamp(5), "market_id": "NP-001", "volume": 100},
            {"timestamp": _make_timestamp(4), "market_id": "NP-001", "volume": 110},
        ]
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        tool = SnapshotVolatilityTool(jsonl_path=p)
        event = EventInput(event_id="e", market_id="NP-001", market_title="m", current_price=0.5)
        result = tool.run(event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert result.metadata["confidence"] == 0.0


class TestWindowFilter:
    """Verify that only rows within window_minutes are used."""

    def test_window_excludes_old_rows(self, tmp_path: Path):
        """Rows older than window_minutes are excluded."""
        p = tmp_path / "window.jsonl"
        rows = [
            # These 3 are within 30 minutes
            {"timestamp": _make_timestamp(10), "market_id": "WIN-001", "last_price": 0.50, "yes_bid": 0.48, "yes_ask": 0.52, "volume": 100, "open_interest": 200},
            {"timestamp": _make_timestamp(20), "market_id": "WIN-001", "last_price": 0.55, "yes_bid": 0.53, "yes_ask": 0.57, "volume": 110, "open_interest": 210},
            {"timestamp": _make_timestamp(25), "market_id": "WIN-001", "last_price": 0.52, "yes_bid": 0.50, "yes_ask": 0.54, "volume": 120, "open_interest": 220},
            # This one is outside 30-min window
            {"timestamp": _make_timestamp(60), "market_id": "WIN-001", "last_price": 0.90, "yes_bid": 0.88, "yes_ask": 0.92, "volume": 200, "open_interest": 300},
        ]
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        tool = SnapshotVolatilityTool(jsonl_path=p)
        event = EventInput(event_id="e", market_id="WIN-001", market_title="m", current_price=0.5)
        result = tool.run(event, window_minutes=30)
        # Only 3 rows within window, not the 0.90 outlier
        assert result.metadata["sample_count"] == 3
        price_range = result.output_vector[1]
        assert price_range == pytest.approx(0.05, abs=0.001)  # 0.55 - 0.50


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_missing_jsonl_file(self, tmp_path: Path):
        """Tool handles missing JSONL gracefully."""
        tool = SnapshotVolatilityTool(jsonl_path=tmp_path / "does_not_exist.jsonl")
        event = EventInput(event_id="e", market_id="X", market_title="m", current_price=0.5)
        result = tool.run(event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_empty_jsonl_file(self, tmp_path: Path):
        """Tool handles empty JSONL gracefully."""
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        tool = SnapshotVolatilityTool(jsonl_path=p)
        event = EventInput(event_id="e", market_id="X", market_title="m", current_price=0.5)
        result = tool.run(event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_liquidity_fallback_to_volume(self, tmp_path: Path):
        """When open_interest is absent, falls back to volume."""
        p = tmp_path / "novol.jsonl"
        rows = [
            {"timestamp": _make_timestamp(5), "market_id": "VOL-001", "last_price": 0.50, "yes_bid": 0.48, "yes_ask": 0.52, "volume": 300},
            {"timestamp": _make_timestamp(4), "market_id": "VOL-001", "last_price": 0.52, "yes_bid": 0.50, "yes_ask": 0.54, "volume": 400},
            {"timestamp": _make_timestamp(3), "market_id": "VOL-001", "last_price": 0.54, "yes_bid": 0.52, "yes_ask": 0.56, "volume": 500},
        ]
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        tool = SnapshotVolatilityTool(jsonl_path=p)
        event = EventInput(event_id="e", market_id="VOL-001", market_title="m", current_price=0.5)
        result = tool.run(event)
        liquidity = result.output_vector[4]
        assert liquidity == pytest.approx(400.0, abs=1.0)  # mean(300, 400, 500)
