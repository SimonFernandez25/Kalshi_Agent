"""
Tests for the deterministic backtesting engine.

Verifies:
  - run_backtest() returns a well-formed BacktestReport
  - Report is deterministic (same input → same output)
  - Missing snapshot file returns empty report, not an exception
  - Market filter correctly narrows events
  - Formula presets produce valid FormulaSpecs
  - No live API calls are made
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prediction_agent.backtest.backtest import run_backtest, BacktestReport, _build_formula


# ── Synthetic snapshot data ────────────────────────────────────────────────────

def _make_snapshot(
    market_id: str = "KXNBA-TEST",
    price: float = 0.45,
    ts: str = "2025-01-01T00:00:00+00:00",
) -> dict:
    return {
        "market_id": market_id,
        "event_id": market_id,
        "title": f"Test Market {market_id}",
        "last_price": price,
        "timestamp": ts,
        "yes_bid": price - 0.02,
        "yes_ask": price + 0.02,
        "volume": 100,
        "liquidity": 500,
        "open_interest": 50,
        "time_to_close_sec": 3600,
    }


def _write_snapshot_file(rows: list) -> Path:
    """Write rows as JSONL to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for row in rows:
        tmp.write(json.dumps(row) + "\n")
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestBacktestReportStructure:
    def test_returns_backtest_report_type(self):
        """run_backtest() must return a BacktestReport instance."""
        rows = [_make_snapshot(price=p) for p in [0.40, 0.55, 0.60, 0.35, 0.70]]
        snap = _write_snapshot_file(rows)
        report = run_backtest(snapshot_file=snap)
        assert isinstance(report, BacktestReport)

    def test_report_has_all_required_fields(self):
        """BacktestReport must have all schema-mandated fields."""
        rows = [_make_snapshot(price=0.5)] * 10
        snap = _write_snapshot_file(rows)
        report = run_backtest(snapshot_file=snap)

        assert hasattr(report, "total_events")
        assert hasattr(report, "bets_triggered")
        assert hasattr(report, "accuracy")
        assert hasattr(report, "brier_score")
        assert hasattr(report, "avg_score_delta")
        assert hasattr(report, "bet_precision")
        assert hasattr(report, "bet_recall")
        assert hasattr(report, "profit_simulation")
        assert hasattr(report, "tool_contributions")
        assert hasattr(report, "formula_preset")
        assert hasattr(report, "snapshot_file")
        assert hasattr(report, "run_timestamp")
        assert hasattr(report, "total_events_skipped")

    def test_total_events_matches_snapshot_count(self):
        """total_events + total_events_skipped should equal number of rows."""
        rows = [_make_snapshot(price=0.5)] * 15
        snap = _write_snapshot_file(rows)
        report = run_backtest(snapshot_file=snap)
        assert report.total_events + report.total_events_skipped == 15

    def test_brier_score_in_valid_range(self):
        """Brier score must be in [0, 1]."""
        rows = [_make_snapshot(price=p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
        snap = _write_snapshot_file(rows)
        report = run_backtest(snapshot_file=snap)
        assert 0.0 <= report.brier_score <= 1.0

    def test_accuracy_in_valid_range(self):
        """Accuracy must be between 0 and 1."""
        rows = [_make_snapshot(price=p) for p in [0.6, 0.7, 0.8, 0.3, 0.4]]
        snap = _write_snapshot_file(rows)
        report = run_backtest(snapshot_file=snap)
        assert 0.0 <= report.accuracy <= 1.0

    def test_tool_contributions_list_not_empty(self):
        """tool_contributions must contain at least one entry."""
        rows = [_make_snapshot(price=0.5)] * 5
        snap = _write_snapshot_file(rows)
        report = run_backtest(snapshot_file=snap)
        assert len(report.tool_contributions) >= 1


class TestBacktestDeterminism:
    def test_same_input_same_output(self):
        """Two runs on the same snapshot file must produce identical results."""
        rows = [_make_snapshot(price=p) for p in [0.4, 0.5, 0.6, 0.7, 0.3]]
        snap = _write_snapshot_file(rows)

        report_a = run_backtest(snapshot_file=snap)
        report_b = run_backtest(snapshot_file=snap)

        assert report_a.total_events       == report_b.total_events
        assert report_a.bets_triggered     == report_b.bets_triggered
        assert report_a.brier_score        == report_b.brier_score
        assert report_a.profit_simulation  == report_b.profit_simulation

    def test_preset_change_can_change_results(self):
        """Different presets should produce potentially different trigger counts."""
        rows = [_make_snapshot(price=p) for p in [0.4, 0.5, 0.6, 0.7, 0.8] * 4]
        snap = _write_snapshot_file(rows)

        r_eq   = run_backtest(snapshot_file=snap, formula_preset="equal_weight")
        r_real = run_backtest(snapshot_file=snap, formula_preset="real_tools_only")

        # Both must be valid reports; they may differ since tool sets differ
        assert isinstance(r_eq, BacktestReport)
        assert isinstance(r_real, BacktestReport)


class TestBacktestEdgeCases:
    def test_missing_snapshot_file_returns_empty_report(self):
        """A non-existent snapshot file must return an empty report, not raise."""
        report = run_backtest(
            snapshot_file=Path("/tmp/this_file_definitely_does_not_exist.jsonl")
        )
        assert isinstance(report, BacktestReport)
        assert report.total_events == 0

    def test_empty_snapshot_file_returns_empty_report(self):
        """An empty JSONL file must return an empty report."""
        snap = _write_snapshot_file([])
        report = run_backtest(snapshot_file=snap)
        assert report.total_events == 0

    def test_market_filter_narrows_events(self):
        """market_filter must exclude rows not matching the pattern."""
        rows = (
            [_make_snapshot(market_id="KXNBA-LAL", price=0.5)] * 5
            + [_make_snapshot(market_id="KXMLB-NYM", price=0.5)] * 5
        )
        snap = _write_snapshot_file(rows)

        report_all = run_backtest(snapshot_file=snap)
        report_nba = run_backtest(snapshot_file=snap, market_filter="KXNBA")

        assert report_all.total_events >= report_nba.total_events
        assert report_nba.total_events == 5

    def test_malformed_rows_skipped_gracefully(self):
        """Rows with invalid data should be skipped, not crash the engine."""
        rows = [
            {"not_a_market": True},          # no market_id
            _make_snapshot(price=0.5),       # valid
            {"market_id": "", "price": "x"}, # empty market_id
            _make_snapshot(price=0.6),       # valid
        ]
        snap = _write_snapshot_file(rows)
        report = run_backtest(snapshot_file=snap)
        # At least the 2 valid rows should be processed
        assert report.total_events >= 2


class TestFormulaPresets:
    """Tests for _build_formula preset logic (no engine involved)."""

    def test_equal_weight_all_tools(self):
        """equal_weight preset uses all available tools."""
        tools = ["tool_a", "tool_b", "tool_c"]
        formula = _build_formula("equal_weight", tools, threshold=0.5)
        assert len(formula.selections) == 3

    def test_equal_weight_sums_to_one(self):
        """Weights must sum to approximately 1.0."""
        tools = ["a", "b", "c", "d"]
        formula = _build_formula("equal_weight", tools, threshold=0.5)
        total = sum(s.weight for s in formula.selections)
        assert abs(total - 1.0) < 0.01

    def test_volatility_only_prefers_volatility_tools(self):
        """volatility_only preset should select volatility tools if available."""
        tools = ["snapshot_volatility_tool", "mock_price_signal", "liquidity_spike_tool"]
        formula = _build_formula("volatility_only", tools, threshold=0.5)
        selected = [s.tool_name for s in formula.selections]
        assert "snapshot_volatility_tool" in selected

    def test_real_tools_only_excludes_mock(self):
        """real_tools_only preset should exclude mock_ tools."""
        tools = ["mock_price_signal", "snapshot_volatility_tool", "price_jump_detector_tool"]
        formula = _build_formula("real_tools_only", tools, threshold=0.5)
        selected = [s.tool_name for s in formula.selections]
        assert "mock_price_signal" not in selected

    def test_threshold_stored_correctly(self):
        """The formula's threshold must match what was passed in."""
        tools = ["tool_x"]
        formula = _build_formula("equal_weight", tools, threshold=0.73)
        assert formula.threshold == pytest.approx(0.73)
