"""
Tests for execution_logger.

Verifies:
  - log_execution writes valid JSONL
  - All required fields are present
  - Repeated calls append (not overwrite)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prediction_agent.evolution.execution_logger import log_execution


def _make_run_result() -> dict:
    """Minimal run_pipeline return dict for testing."""
    return {
        "run_id": "test-001",
        "event": {
            "event_id": "evt-001",
            "market_id": "MKT-001",
            "market_title": "Test Market",
            "current_price": 0.55,
            "timestamp": "2025-01-01T00:00:00+00:00",
        },
        "formula": {
            "selections": [
                {"tool_name": "mock_price_signal", "tool_inputs": {}, "weight": 0.6},
                {"tool_name": "mock_random_context", "tool_inputs": {}, "weight": 0.4},
            ],
            "aggregation": "weighted_sum",
            "threshold": 0.5,
            "rationale": "Test rationale with some probability calculation.",
        },
        "score": {
            "final_score": 0.62,
            "tool_outputs": [
                {"tool_name": "mock_price_signal", "output_vector": [0.7]},
                {"tool_name": "mock_random_context", "output_vector": [0.5]},
            ],
            "weights": [0.6, 0.4],
            "threshold": 0.5,
            "bet_triggered": True,
        },
    }


class TestExecutionLogger:

    def test_writes_valid_jsonl(self, tmp_path: Path):
        log_file = tmp_path / "test_log.jsonl"
        log_execution(_make_run_result(), log_path=log_file)

        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert isinstance(record, dict)

    def test_required_fields_present(self, tmp_path: Path):
        log_file = tmp_path / "test_log.jsonl"
        entry = log_execution(_make_run_result(), log_path=log_file)

        assert entry.run_id == "test-001"
        assert entry.market_id == "MKT-001"
        assert entry.selected_tools == ["mock_price_signal", "mock_random_context"]
        assert entry.tool_weights == [0.6, 0.4]
        assert entry.final_score == 0.62
        assert entry.threshold == 0.5
        assert entry.bet_triggered is True

    def test_appends_not_overwrites(self, tmp_path: Path):
        log_file = tmp_path / "test_log.jsonl"

        log_execution(_make_run_result(), log_path=log_file)
        log_execution(_make_run_result(), log_path=log_file)

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_reasoning_segments_captured(self, tmp_path: Path):
        log_file = tmp_path / "test_log.jsonl"
        entry = log_execution(_make_run_result(), log_path=log_file)

        assert "probability" in entry.reasoning_segments.lower()

    def test_tool_outputs_captured(self, tmp_path: Path):
        log_file = tmp_path / "test_log.jsonl"
        entry = log_execution(_make_run_result(), log_path=log_file)

        assert len(entry.tool_outputs) == 2
        assert entry.tool_outputs[0]["tool_name"] == "mock_price_signal"
