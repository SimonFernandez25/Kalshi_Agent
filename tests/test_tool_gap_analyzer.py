"""
Tests for tool_gap_analyzer.

Verifies:
  - Detects low-confidence runs
  - Returns None when no gap exceeds threshold
  - Detects implicit calculations in rationale
  - Handles insufficient data gracefully
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prediction_agent.evolution.tool_gap_analyzer import analyze_gaps


def _write_entries(path: Path, entries: list[dict]) -> None:
    """Write entries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _make_entry(
    run_id: str = "run-001",
    final_score: float = 0.62,
    threshold: float = 0.5,
    rationale: str = "Standard analysis.",
    tokens: int = 500,
) -> dict:
    """Create a synthetic execution log entry."""
    return {
        "run_id": run_id,
        "market_id": "MKT-001",
        "market_title": "Test",
        "selected_tools": ["mock_price_signal"],
        "tool_weights": [1.0],
        "tool_outputs": [{"tool_name": "mock_price_signal", "output_vector": [0.6]}],
        "final_score": final_score,
        "threshold": threshold,
        "bet_triggered": final_score >= threshold,
        "reasoning_segments": rationale,
        "failed_tool_attempts": [],
        "total_tokens_used": tokens,
        "timestamp": "2025-01-01T00:00:00+00:00",
    }


class TestGapAnalyzer:

    def test_insufficient_data_returns_none(self, tmp_path: Path):
        log_file = tmp_path / "logs.jsonl"
        _write_entries(log_file, [_make_entry(run_id=f"run-{i}") for i in range(3)])

        result = analyze_gaps(log_path=log_file, min_runs=5)
        assert result is None

    def test_no_gap_returns_none(self, tmp_path: Path):
        log_file = tmp_path / "logs.jsonl"
        entries = [
            _make_entry(run_id=f"run-{i}", final_score=0.8, threshold=0.5)
            for i in range(10)
        ]
        _write_entries(log_file, entries)

        result = analyze_gaps(log_path=log_file, min_runs=5, gap_threshold=0.9)
        assert result is None

    def test_detects_low_confidence_runs(self, tmp_path: Path):
        log_file = tmp_path / "logs.jsonl"
        # All runs have score very close to threshold
        entries = [
            _make_entry(run_id=f"run-{i}", final_score=0.51, threshold=0.50)
            for i in range(10)
        ]
        _write_entries(log_file, entries)

        result = analyze_gaps(log_path=log_file, min_runs=5, gap_threshold=0.1)
        assert result is not None
        assert "borderline" in result.problem_detected.lower() or result.priority_score > 0.0

    def test_detects_implicit_calculations(self, tmp_path: Path):
        log_file = tmp_path / "logs.jsonl"
        entries = [
            _make_entry(
                run_id=f"run-{i}",
                rationale="I need to calculate the probability and convert implied odds to a ratio.",
            )
            for i in range(10)
        ]
        _write_entries(log_file, entries)

        result = analyze_gaps(log_path=log_file, min_runs=5, gap_threshold=0.1)
        assert result is not None

    def test_missing_file_returns_none(self, tmp_path: Path):
        log_file = tmp_path / "nonexistent.jsonl"
        result = analyze_gaps(log_path=log_file, min_runs=1)
        assert result is None

    def test_empty_file_returns_none(self, tmp_path: Path):
        log_file = tmp_path / "empty.jsonl"
        log_file.write_text("")
        result = analyze_gaps(log_path=log_file, min_runs=1)
        assert result is None
