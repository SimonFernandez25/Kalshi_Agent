"""
Tests for tool_lifecycle_manager.

Verifies:
  - Usage recording updates aggregated stats
  - Deprecation triggers after N consecutive underperforming runs
  - Deprecated status does not delete tool files
  - Active tools list excludes deprecated tools
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prediction_agent.evolution.tool_lifecycle_manager import ToolLifecycleManager
from prediction_agent.evolution.schemas import ToolStatus


class TestLifecycleManager:

    def test_record_usage_updates_stats(self, tmp_path: Path):
        lm = ToolLifecycleManager(lifecycle_path=tmp_path / "lifecycle.jsonl")

        lm.record_usage("test_tool", score_contribution=0.3, correct=True, latency_ms=50.0)
        lm.record_usage("test_tool", score_contribution=0.5, correct=False, latency_ms=30.0)

        record = lm.get_record("test_tool")
        assert record is not None
        assert record.usage_count == 2
        assert record.total_score_contribution == pytest.approx(0.8, abs=0.01)
        assert record.correct_predictions == 1
        assert record.total_predictions == 2
        assert record.avg_score_contribution == pytest.approx(0.4, abs=0.01)
        assert record.correct_prediction_rate == pytest.approx(0.5, abs=0.01)
        assert record.avg_latency_ms == pytest.approx(40.0, abs=0.01)

    def test_deprecation_triggers(self, tmp_path: Path):
        lm = ToolLifecycleManager(lifecycle_path=tmp_path / "lifecycle.jsonl")

        # Record 10 consecutive underperforming runs
        for _ in range(10):
            lm.record_usage(
                "bad_tool",
                score_contribution=-0.1,
                correct=False,
                latency_ms=100.0,
                underperformance_threshold=0.0,
            )

        assert lm.check_deprecation("bad_tool") is True
        record = lm.get_record("bad_tool")
        assert record.status == ToolStatus.DEPRECATED

    def test_deprecation_resets_on_good_run(self, tmp_path: Path):
        lm = ToolLifecycleManager(lifecycle_path=tmp_path / "lifecycle.jsonl")

        # 9 bad runs
        for _ in range(9):
            lm.record_usage("mixed_tool", score_contribution=-0.1, correct=False, latency_ms=50.0, underperformance_threshold=0.0)

        # 1 good run resets counter
        lm.record_usage("mixed_tool", score_contribution=0.5, correct=True, latency_ms=50.0, underperformance_threshold=0.0)

        assert lm.check_deprecation("mixed_tool") is False
        record = lm.get_record("mixed_tool")
        assert record.consecutive_underperformance == 0

    def test_active_tools_excludes_deprecated(self, tmp_path: Path):
        lm = ToolLifecycleManager(lifecycle_path=tmp_path / "lifecycle.jsonl")

        lm.record_usage("good_tool", score_contribution=0.5, correct=True, latency_ms=20.0)

        for _ in range(10):
            lm.record_usage("bad_tool", score_contribution=-0.1, correct=False, latency_ms=100.0, underperformance_threshold=0.0)
        lm.check_deprecation("bad_tool")

        active = lm.get_active_tools()
        assert "good_tool" in active
        assert "bad_tool" not in active

    def test_persistence_across_instances(self, tmp_path: Path):
        lifecycle_file = tmp_path / "lifecycle.jsonl"

        lm1 = ToolLifecycleManager(lifecycle_path=lifecycle_file)
        lm1.record_usage("persisted_tool", score_contribution=0.4, correct=True, latency_ms=25.0)

        # Create new instance -- should load persisted data
        lm2 = ToolLifecycleManager(lifecycle_path=lifecycle_file)
        record = lm2.get_record("persisted_tool")
        assert record is not None
        assert record.usage_count == 1
        assert record.correct_predictions == 1

    def test_unknown_tool_not_deprecated(self, tmp_path: Path):
        lm = ToolLifecycleManager(lifecycle_path=tmp_path / "lifecycle.jsonl")
        assert lm.check_deprecation("nonexistent_tool") is False
