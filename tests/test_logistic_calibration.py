"""
Tests for logistic calibration layer in engine/scorer.py.

Covers:
  - sigmoid function mathematical properties
  - probability_edge mode with USE_LOGISTIC_CALIBRATION=True
  - probability_edge mode with USE_LOGISTIC_CALIBRATION=False
  - z-space attribution preservation
  - raw_score_z logging
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from engine.scorer import _sigmoid, compute_score
from schemas import EventInput, FormulaSpec, ToolSelection, ToolOutput


# ── Sigmoid Function Tests ─────────────────────────────────────────────────────

class TestSigmoidFunction:
    """Test the deterministic sigmoid transform."""

    def test_sigmoid_at_zero_is_half(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-9

    def test_sigmoid_positive_approaches_one(self):
        assert _sigmoid(5.0) > 0.99
        assert _sigmoid(10.0) > 0.9999
        assert _sigmoid(20.0) >= 1.0 - 1e-9

    def test_sigmoid_negative_approaches_zero(self):
        assert _sigmoid(-5.0) < 0.01
        assert _sigmoid(-10.0) < 0.0001
        assert _sigmoid(-20.0) <= 1e-9

    def test_sigmoid_extreme_positive_clamped(self):
        # Numerical stability: x >= 20 → 1.0
        assert _sigmoid(100.0) == 1.0
        assert _sigmoid(1000.0) == 1.0

    def test_sigmoid_extreme_negative_clamped(self):
        # Numerical stability: x <= -20 → 0.0
        assert _sigmoid(-100.0) == 0.0
        assert _sigmoid(-1000.0) == 0.0

    def test_sigmoid_monotonic_increasing(self):
        values = [-10, -5, -1, 0, 1, 5, 10]
        results = [_sigmoid(x) for x in values]
        # Should be strictly increasing
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]

    def test_sigmoid_symmetric_around_half(self):
        # σ(x) + σ(-x) = 1 (logistic function property)
        x = 2.5
        assert abs(_sigmoid(x) + _sigmoid(-x) - 1.0) < 1e-9

    def test_sigmoid_deterministic(self):
        # Same input → same output (determinism check)
        x = 1.234567
        result1 = _sigmoid(x)
        result2 = _sigmoid(x)
        assert result1 == result2


# ── Integration Tests: probability_edge with Logistic Calibration ─────────────

class TestProbabilityEdgeWithCalibration:
    """Test probability_edge mode with USE_LOGISTIC_CALIBRATION=True."""

    @pytest.fixture
    def stub_event(self):
        return EventInput(
            event_id="TEST-001",
            market_id="TEST-001",
            market_title="Test Market",
            current_price=0.50,
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def stub_formula(self):
        return FormulaSpec(
            selections=[
                ToolSelection(tool_name="sportsbook_implied_probability_tool", tool_inputs={}, weight=0.6),
                ToolSelection(tool_name="snapshot_volatility_tool", tool_inputs={}, weight=0.4),
            ],
            aggregation="weighted_sum",
            threshold=0.05,
            rationale="Test formula",
        )

    def test_logistic_calibration_enabled_produces_bounded_probability(
        self, stub_event, stub_formula, monkeypatch
    ):
        """With calibration, p_model = sigmoid(z) ∈ (0,1)."""
        monkeypatch.setattr("config.USE_LOGISTIC_CALIBRATION", True)
        monkeypatch.setattr("config.PROBABILITY_SIGNAL_TOOLS", ["sportsbook_implied_probability_tool"])

        tool_outputs = [
            ToolOutput(tool_name="sportsbook_implied_probability_tool", output_vector=[0.65], metadata={}),
            ToolOutput(tool_name="snapshot_volatility_tool", output_vector=[0.2], metadata={}),
        ]

        score = compute_score(
            tool_outputs,
            stub_formula,
            scoring_mode="probability_edge",
            current_market_price=0.50,
        )

        # z = 0.6*0.65 + 0.4*0.2 = 0.39 + 0.08 = 0.47
        # p_model = sigmoid(0.47) ≈ 0.6155
        assert score.raw_score_z is not None
        assert abs(score.raw_score_z - 0.47) < 0.01
        assert score.model_probability is not None
        assert 0.0 < score.model_probability < 1.0
        # sigmoid(0.47) ≈ 0.6155
        assert abs(score.model_probability - _sigmoid(0.47)) < 1e-6

    def test_logistic_calibration_disabled_clamps_probability(
        self, stub_event, stub_formula, monkeypatch
    ):
        """Without calibration, p_model = clamp(z, 0, 1)."""
        monkeypatch.setattr("config.USE_LOGISTIC_CALIBRATION", False)
        monkeypatch.setattr("config.PROBABILITY_SIGNAL_TOOLS", ["sportsbook_implied_probability_tool"])

        tool_outputs = [
            ToolOutput(tool_name="sportsbook_implied_probability_tool", output_vector=[0.65], metadata={}),
            ToolOutput(tool_name="snapshot_volatility_tool", output_vector=[0.2], metadata={}),
        ]

        score = compute_score(
            tool_outputs,
            stub_formula,
            scoring_mode="probability_edge",
            current_market_price=0.50,
        )

        # z = 0.47
        # p_model = clamp(0.47, 0, 1) = 0.47
        assert score.raw_score_z is not None
        assert abs(score.raw_score_z - 0.47) < 0.01
        assert score.model_probability is not None
        assert abs(score.model_probability - 0.47) < 0.01

    def test_calibration_clamps_z_above_one(self, stub_event, monkeypatch):
        """Legacy mode clamps z > 1 → 1.0, calibration applies sigmoid."""
        monkeypatch.setattr("config.USE_LOGISTIC_CALIBRATION", False)
        monkeypatch.setattr("config.PROBABILITY_SIGNAL_TOOLS", ["sportsbook_implied_probability_tool"])

        formula = FormulaSpec(
            selections=[
                ToolSelection(tool_name="sportsbook_implied_probability_tool", tool_inputs={}, weight=1.0),
            ],
            aggregation="weighted_sum",
            threshold=0.05,
            rationale="Test",
        )

        tool_outputs = [
            ToolOutput(tool_name="sportsbook_implied_probability_tool", output_vector=[1.5], metadata={}),
        ]

        score = compute_score(
            tool_outputs,
            formula,
            scoring_mode="probability_edge",
            current_market_price=0.50,
        )

        # z = 1.5, clamp(1.5, 0, 1) = 1.0
        assert score.raw_score_z == 1.5
        assert score.model_probability == 1.0

    def test_calibration_transforms_z_above_one(self, stub_event, monkeypatch):
        """Calibration mode: z > 1 → sigmoid(z) < 1 but close to 1."""
        monkeypatch.setattr("config.USE_LOGISTIC_CALIBRATION", True)
        monkeypatch.setattr("config.PROBABILITY_SIGNAL_TOOLS", ["sportsbook_implied_probability_tool"])

        formula = FormulaSpec(
            selections=[
                ToolSelection(tool_name="sportsbook_implied_probability_tool", tool_inputs={}, weight=1.0),
            ],
            aggregation="weighted_sum",
            threshold=0.05,
            rationale="Test",
        )

        tool_outputs = [
            ToolOutput(tool_name="sportsbook_implied_probability_tool", output_vector=[2.0], metadata={}),
        ]

        score = compute_score(
            tool_outputs,
            formula,
            scoring_mode="probability_edge",
            current_market_price=0.50,
        )

        # z = 2.0, sigmoid(2.0) ≈ 0.8808
        assert score.raw_score_z == 2.0
        assert abs(score.model_probability - _sigmoid(2.0)) < 1e-6
        assert score.model_probability < 1.0

    def test_z_space_attribution_preserved(self, stub_event, stub_formula, monkeypatch):
        """Attribution remains linear in z-space (sum of w*s)."""
        monkeypatch.setattr("config.USE_LOGISTIC_CALIBRATION", True)
        monkeypatch.setattr("config.PROBABILITY_SIGNAL_TOOLS", ["sportsbook_implied_probability_tool"])

        tool_outputs = [
            ToolOutput(tool_name="sportsbook_implied_probability_tool", output_vector=[0.7], metadata={}),
            ToolOutput(tool_name="snapshot_volatility_tool", output_vector=[0.3], metadata={}),
        ]

        score = compute_score(
            tool_outputs,
            stub_formula,
            scoring_mode="probability_edge",
            current_market_price=0.50,
        )

        # z = 0.6*0.7 + 0.4*0.3 = 0.42 + 0.12 = 0.54
        expected_z = 0.6 * 0.7 + 0.4 * 0.3
        assert abs(score.raw_score_z - expected_z) < 1e-6

        # Tool contributions at z-level:
        contrib_1 = 0.6 * 0.7  # = 0.42
        contrib_2 = 0.4 * 0.3  # = 0.12
        assert abs(contrib_1 + contrib_2 - expected_z) < 1e-9

    def test_edge_computed_from_calibrated_probability(self, stub_event, stub_formula, monkeypatch):
        """Edge = p_model - p_market, where p_model is post-calibration."""
        monkeypatch.setattr("config.USE_LOGISTIC_CALIBRATION", True)
        monkeypatch.setattr("config.PROBABILITY_SIGNAL_TOOLS", ["sportsbook_implied_probability_tool"])

        tool_outputs = [
            ToolOutput(tool_name="sportsbook_implied_probability_tool", output_vector=[0.8], metadata={}),
            ToolOutput(tool_name="snapshot_volatility_tool", output_vector=[0.1], metadata={}),
        ]

        market_price = 0.40
        score = compute_score(
            tool_outputs,
            stub_formula,
            scoring_mode="probability_edge",
            current_market_price=market_price,
        )

        # z = 0.6*0.8 + 0.4*0.1 = 0.48 + 0.04 = 0.52
        # p_model = sigmoid(0.52) ≈ 0.6271
        # edge = 0.6271 - 0.40 = 0.2271
        expected_z = 0.52
        expected_p_model = _sigmoid(expected_z)
        expected_edge = expected_p_model - market_price

        assert abs(score.raw_score_z - expected_z) < 0.01
        assert abs(score.model_probability - expected_p_model) < 1e-6
        assert abs(score.edge - expected_edge) < 1e-6

    def test_bet_triggered_based_on_edge(self, stub_event, stub_formula, monkeypatch):
        """Bet triggers when abs(edge) >= threshold."""
        monkeypatch.setattr("config.USE_LOGISTIC_CALIBRATION", True)
        monkeypatch.setattr("config.PROBABILITY_SIGNAL_TOOLS", ["sportsbook_implied_probability_tool"])

        tool_outputs = [
            ToolOutput(tool_name="sportsbook_implied_probability_tool", output_vector=[0.9], metadata={}),
            ToolOutput(tool_name="snapshot_volatility_tool", output_vector=[0.1], metadata={}),
        ]

        market_price = 0.40
        threshold = 0.05

        formula_high_threshold = FormulaSpec(
            selections=stub_formula.selections,
            aggregation="weighted_sum",
            threshold=threshold,
            rationale="Test",
        )

        score = compute_score(
            tool_outputs,
            formula_high_threshold,
            scoring_mode="probability_edge",
            current_market_price=market_price,
        )

        # z = 0.6*0.9 + 0.4*0.1 = 0.54 + 0.04 = 0.58
        # p_model = sigmoid(0.58) ≈ 0.6413
        # edge = 0.6413 - 0.40 = 0.2413
        # abs(edge) = 0.2413 >= 0.05 → trigger
        assert score.bet_triggered is True
        assert abs(score.edge) >= threshold

    def test_raw_score_z_always_logged(self, stub_event, stub_formula, monkeypatch):
        """raw_score_z field is always populated in probability_edge mode."""
        monkeypatch.setattr("config.USE_LOGISTIC_CALIBRATION", True)
        monkeypatch.setattr("config.PROBABILITY_SIGNAL_TOOLS", ["sportsbook_implied_probability_tool"])

        tool_outputs = [
            ToolOutput(tool_name="sportsbook_implied_probability_tool", output_vector=[0.5], metadata={}),
            ToolOutput(tool_name="snapshot_volatility_tool", output_vector=[0.5], metadata={}),
        ]

        score = compute_score(
            tool_outputs,
            stub_formula,
            scoring_mode="probability_edge",
            current_market_price=0.50,
        )

        assert score.raw_score_z is not None
        # z = 0.6*0.5 + 0.4*0.5 = 0.5
        assert abs(score.raw_score_z - 0.5) < 1e-6

    def test_no_probability_tools_fallback(self, stub_event, monkeypatch):
        """When no probability tools selected, all tools contribute to z."""
        monkeypatch.setattr("config.USE_LOGISTIC_CALIBRATION", True)
        monkeypatch.setattr("config.PROBABILITY_SIGNAL_TOOLS", ["sportsbook_implied_probability_tool"])

        formula = FormulaSpec(
            selections=[
                ToolSelection(tool_name="snapshot_volatility_tool", tool_inputs={}, weight=0.5),
                ToolSelection(tool_name="price_jump_detector_tool", tool_inputs={}, weight=0.5),
            ],
            aggregation="weighted_sum",
            threshold=0.05,
            rationale="No probability tools",
        )

        tool_outputs = [
            ToolOutput(tool_name="snapshot_volatility_tool", output_vector=[0.6], metadata={}),
            ToolOutput(tool_name="price_jump_detector_tool", output_vector=[0.4], metadata={}),
        ]

        score = compute_score(
            tool_outputs,
            formula,
            scoring_mode="probability_edge",
            current_market_price=0.50,
        )

        # z = 0.5*0.6 + 0.5*0.4 = 0.5
        # p_model = sigmoid(0.5) = 0.6225
        assert abs(score.raw_score_z - 0.5) < 1e-6
        assert abs(score.model_probability - _sigmoid(0.5)) < 1e-6


# ── Regression Tests ───────────────────────────────────────────────────────────

class TestCalibrationBackwardCompatibility:
    """Ensure logistic calibration doesn't break signal_sum mode."""

    @pytest.fixture
    def stub_event(self):
        return EventInput(
            event_id="TEST-001",
            market_id="TEST-001",
            market_title="Test Market",
            current_price=0.50,
            timestamp=datetime.now(timezone.utc),
        )

    def test_signal_sum_mode_unchanged(self, stub_event):
        """signal_sum mode ignores USE_LOGISTIC_CALIBRATION."""
        formula = FormulaSpec(
            selections=[
                ToolSelection(tool_name="snapshot_volatility_tool", tool_inputs={}, weight=0.5),
                ToolSelection(tool_name="price_jump_detector_tool", tool_inputs={}, weight=0.5),
            ],
            aggregation="weighted_sum",
            threshold=0.6,
            rationale="Test",
        )

        tool_outputs = [
            ToolOutput(tool_name="snapshot_volatility_tool", output_vector=[0.7], metadata={}),
            ToolOutput(tool_name="price_jump_detector_tool", output_vector=[0.5], metadata={}),
        ]

        score = compute_score(
            tool_outputs,
            formula,
            scoring_mode="signal_sum",
        )

        # final_score = 0.5*0.7 + 0.5*0.5 = 0.6
        assert abs(score.final_score - 0.6) < 1e-6
        assert score.raw_score_z is None  # Not used in signal_sum mode
        assert score.model_probability is None
        assert score.edge is None
