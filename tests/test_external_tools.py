"""
Unit tests for all 4 external deterministic tools.

Tests:
  - Normal operation with stub data
  - Empty snapshot handling (safe zeros)
  - Missing file handling
  - No API calls (monkey-patched requests)
  - Determinism (same input -> same output)
  - Output vector length and value range
  - Confidence field presence in metadata
  - End-to-end run using stub snapshots
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

# ── Resolve repo root ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from schemas import EventInput, ToolOutput

# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def stub_event() -> EventInput:
    return EventInput(
        event_id="test-event-001",
        market_id="STUB-NBA-LAL-BOS-001",
        market_title="Lakers vs Celtics: Lakers Win",
        current_price=0.53,
        timestamp=datetime(2025, 1, 20, 19, 30, 0, tzinfo=timezone.utc),
    )


def _make_fred_jsonl(tmp_path: Path, n_rows: int = 12) -> Path:
    """Write synthetic FRED rows to a temp JSONL file."""
    p = tmp_path / "fred_snapshots.jsonl"
    collected = datetime.now(timezone.utc).isoformat()
    rows = []
    base = 5.33
    for i in range(n_rows):
        rows.append({
            "series_id": "FEDFUNDS",
            "observation_date": f"2024-{(i % 12) + 1:02d}-01",
            "value": round(base - i * 0.1, 2),
            "collected_at": collected,
            "source": "fred",
        })
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def _make_bls_jsonl(tmp_path: Path, n_rows: int = 13) -> Path:
    p = tmp_path / "bls_snapshots.jsonl"
    collected = datetime.now(timezone.utc).isoformat()
    months = [f"M{i:02d}" for i in range(1, 13)] + ["M01"]
    rows = []
    for i in range(n_rows):
        rows.append({
            "series_id": "LNS14000000",
            "year": "2024" if i < 12 else "2025",
            "period": months[i],
            "period_name": "January",
            "value": round(4.0 + (i % 3) * 0.1, 1),
            "footnotes": [],
            "collected_at": collected,
            "source": "bls",
        })
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def _make_weather_jsonl(tmp_path: Path, n_rows: int = 3) -> Path:
    p = tmp_path / "weather_snapshots.jsonl"
    collected = datetime.now(timezone.utc).isoformat()
    hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
    rows = []
    for i in range(n_rows):
        fd = (datetime.now(timezone.utc) + timedelta(days=i)).strftime("%Y-%m-%d")
        rows.append({
            "location_key": "boston,massachusetts",
            "location_query": "Boston,MA",
            "location_name": "Boston",
            "location_region": "Massachusetts",
            "forecast_date": fd,
            "collected_hour": hour,
            "collected_at": collected,
            "source": "weatherapi",
            "forecast_probability": 0.35,
            "total_precip_mm": 5.2,
            "temp_anomaly_c": -2.1,
            "daily_chance_of_rain": 35,
            "daily_chance_of_snow": 0,
            "model_disagreement_proxy": 1.3,
        })
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def _make_odds_jsonl(tmp_path: Path, n_books: int = 5) -> Path:
    p = tmp_path / "odds_snapshots.jsonl"
    collected = datetime.now(timezone.utc).isoformat()
    hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
    books = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet"]
    home_probs = [0.61, 0.60, 0.62, 0.61, 0.59]
    rows = []
    for i in range(n_books):
        hp = home_probs[i % len(home_probs)]
        rows.append({
            "event_id": "nba-cel-lal-20250120",
            "sport": "basketball_nba",
            "home_team": "Boston Celtics",
            "away_team": "Los Angeles Lakers",
            "commence_time": "2025-01-20T19:30:00Z",
            "bookmaker": books[i % len(books)],
            "bookmaker_title": books[i % len(books)].title(),
            "home_implied_prob": hp,
            "away_implied_prob": round(1.0 - hp, 6),
            "collected_hour": hour,
            "collected_at": collected,
            "source": "theoddsapi",
        })
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# FRED Macro Tool
# ══════════════════════════════════════════════════════════════════════════════

class TestFredMacroTool:
    def _tool(self, jsonl_path):
        from prediction_agent.tools.external.fred_macro_tool import FredMacroTool
        return FredMacroTool(jsonl_path=jsonl_path)

    def test_normal_run_returns_tool_output(self, stub_event, tmp_path):
        p = _make_fred_jsonl(tmp_path)
        result = self._tool(p).run(stub_event, series_id="FEDFUNDS")
        assert isinstance(result, ToolOutput)

    def test_output_vector_length(self, stub_event, tmp_path):
        p = _make_fred_jsonl(tmp_path)
        result = self._tool(p).run(stub_event, series_id="FEDFUNDS")
        assert len(result.output_vector) == 5

    def test_output_vector_values_in_range(self, stub_event, tmp_path):
        p = _make_fred_jsonl(tmp_path)
        result = self._tool(p).run(stub_event, series_id="FEDFUNDS")
        for v in result.output_vector:
            assert 0.0 <= v <= 1.0, f"Value {v} out of [0,1]"

    def test_metadata_has_confidence(self, stub_event, tmp_path):
        p = _make_fred_jsonl(tmp_path)
        result = self._tool(p).run(stub_event, series_id="FEDFUNDS")
        assert "confidence" in result.metadata
        assert 0.0 <= result.metadata["confidence"] <= 1.0

    def test_metadata_has_data_points_used(self, stub_event, tmp_path):
        p = _make_fred_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert "data_points_used" in result.metadata

    def test_empty_file_returns_zeros(self, stub_event, tmp_path):
        p = tmp_path / "fred_snapshots.jsonl"
        p.write_text("")
        result = self._tool(p).run(stub_event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert result.metadata["confidence"] == 0.0

    def test_missing_file_returns_zeros(self, stub_event, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        result = self._tool(p).run(stub_event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_determinism(self, stub_event, tmp_path):
        p = _make_fred_jsonl(tmp_path)
        tool = self._tool(p)
        r1 = tool.run(stub_event, series_id="FEDFUNDS")
        r2 = tool.run(stub_event, series_id="FEDFUNDS")
        assert r1.output_vector == r2.output_vector

    def test_tool_name(self, tmp_path):
        p = _make_fred_jsonl(tmp_path)
        assert self._tool(p).name == "fred_macro_tool"

    def test_no_api_calls(self, stub_event, tmp_path, monkeypatch):
        """Ensures the tool never calls requests.get."""
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("fred_macro_tool must not call requests.get")
        ))
        p = _make_fred_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert result.output_vector is not None


# ══════════════════════════════════════════════════════════════════════════════
# BLS Labor Tool
# ══════════════════════════════════════════════════════════════════════════════

class TestBlsLaborTool:
    def _tool(self, jsonl_path):
        from prediction_agent.tools.external.bls_labor_tool import BlsLaborTool
        return BlsLaborTool(jsonl_path=jsonl_path)

    def test_normal_run_returns_tool_output(self, stub_event, tmp_path):
        p = _make_bls_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert isinstance(result, ToolOutput)

    def test_output_vector_length(self, stub_event, tmp_path):
        p = _make_bls_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert len(result.output_vector) == 5

    def test_output_vector_values_in_range(self, stub_event, tmp_path):
        p = _make_bls_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        for v in result.output_vector:
            assert 0.0 <= v <= 1.0, f"Value {v} out of [0,1]"

    def test_metadata_has_confidence(self, stub_event, tmp_path):
        p = _make_bls_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert "confidence" in result.metadata
        assert 0.0 <= result.metadata["confidence"] <= 1.0

    def test_empty_file_returns_zeros(self, stub_event, tmp_path):
        p = tmp_path / "bls_snapshots.jsonl"
        p.write_text("")
        result = self._tool(p).run(stub_event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_missing_file_returns_zeros(self, stub_event, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        result = self._tool(p).run(stub_event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_determinism(self, stub_event, tmp_path):
        p = _make_bls_jsonl(tmp_path)
        tool = self._tool(p)
        r1 = tool.run(stub_event)
        r2 = tool.run(stub_event)
        assert r1.output_vector == r2.output_vector

    def test_tool_name(self, tmp_path):
        p = _make_bls_jsonl(tmp_path)
        assert self._tool(p).name == "bls_labor_tool"

    def test_no_api_calls(self, stub_event, tmp_path, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("bls_labor_tool must not call requests.post")
        ))
        p = _make_bls_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert result.output_vector is not None


# ══════════════════════════════════════════════════════════════════════════════
# Weather Probability Tool
# ══════════════════════════════════════════════════════════════════════════════

class TestWeatherProbabilityTool:
    def _tool(self, jsonl_path):
        from prediction_agent.tools.external.weather_probability_tool import WeatherProbabilityTool
        return WeatherProbabilityTool(jsonl_path=jsonl_path)

    def test_normal_run_returns_tool_output(self, stub_event, tmp_path):
        p = _make_weather_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert isinstance(result, ToolOutput)

    def test_output_vector_length(self, stub_event, tmp_path):
        p = _make_weather_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert len(result.output_vector) == 5

    def test_output_vector_values_in_range(self, stub_event, tmp_path):
        p = _make_weather_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        for v in result.output_vector:
            assert 0.0 <= v <= 1.0, f"Value {v} out of [0,1]"

    def test_forecast_probability_element(self, stub_event, tmp_path):
        """output_vector[0] should be the forecast_probability from the snapshot."""
        p = _make_weather_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        # Stub has forecast_probability=0.35
        assert abs(result.output_vector[0] - 0.35) < 0.01

    def test_metadata_has_confidence(self, stub_event, tmp_path):
        p = _make_weather_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert "confidence" in result.metadata

    def test_empty_file_returns_zeros(self, stub_event, tmp_path):
        p = tmp_path / "weather_snapshots.jsonl"
        p.write_text("")
        result = self._tool(p).run(stub_event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_missing_file_returns_zeros(self, stub_event, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        result = self._tool(p).run(stub_event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_location_filtering(self, stub_event, tmp_path):
        """location kwarg should match the boston stub."""
        p = _make_weather_jsonl(tmp_path)
        result_boston = self._tool(p).run(stub_event, location="boston")
        result_any    = self._tool(p).run(stub_event)
        # Both should succeed with same data since only boston rows exist
        assert result_boston.output_vector == result_any.output_vector

    def test_determinism(self, stub_event, tmp_path):
        p = _make_weather_jsonl(tmp_path)
        tool = self._tool(p)
        r1 = tool.run(stub_event)
        r2 = tool.run(stub_event)
        assert r1.output_vector == r2.output_vector

    def test_no_api_calls(self, stub_event, tmp_path, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("weather_probability_tool must not call requests.get")
        ))
        p = _make_weather_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert result.output_vector is not None


# ══════════════════════════════════════════════════════════════════════════════
# Sportsbook Implied Probability Tool
# ══════════════════════════════════════════════════════════════════════════════

class TestSportsbookImpliedProbabilityTool:
    def _tool(self, jsonl_path):
        from prediction_agent.tools.external.sportsbook_implied_probability_tool import (
            SportsbookImpliedProbabilityTool,
        )
        return SportsbookImpliedProbabilityTool(jsonl_path=jsonl_path)

    def test_normal_run_returns_tool_output(self, stub_event, tmp_path):
        p = _make_odds_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert isinstance(result, ToolOutput)

    def test_output_vector_length(self, stub_event, tmp_path):
        p = _make_odds_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert len(result.output_vector) == 4

    def test_output_vector_values_in_range(self, stub_event, tmp_path):
        p = _make_odds_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        for v in result.output_vector:
            assert 0.0 <= v <= 1.0, f"Value {v} out of [0,1]"

    def test_mean_implied_probability_reasonable(self, stub_event, tmp_path):
        """Stub probs are ~0.60-0.62 for Celtics; mean should be close."""
        p = _make_odds_jsonl(tmp_path)
        result = self._tool(p).run(stub_event, team="celtics")
        mean_prob = result.output_vector[0]
        assert 0.55 <= mean_prob <= 0.70, f"mean_implied_prob={mean_prob} out of expected range"

    def test_metadata_has_confidence(self, stub_event, tmp_path):
        p = _make_odds_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert "confidence" in result.metadata

    def test_metadata_has_bookmakers_count(self, stub_event, tmp_path):
        p = _make_odds_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert "bookmakers_count" in result.metadata
        assert result.metadata["bookmakers_count"] >= 1

    def test_empty_file_returns_zeros(self, stub_event, tmp_path):
        p = tmp_path / "odds_snapshots.jsonl"
        p.write_text("")
        result = self._tool(p).run(stub_event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0]

    def test_missing_file_returns_zeros(self, stub_event, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        result = self._tool(p).run(stub_event)
        assert result.output_vector == [0.0, 0.0, 0.0, 0.0]

    def test_determinism(self, stub_event, tmp_path):
        p = _make_odds_jsonl(tmp_path)
        tool = self._tool(p)
        r1 = tool.run(stub_event)
        r2 = tool.run(stub_event)
        assert r1.output_vector == r2.output_vector

    def test_no_api_calls(self, stub_event, tmp_path, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("sportsbook_implied_probability_tool must not call requests.get")
        ))
        p = _make_odds_jsonl(tmp_path)
        result = self._tool(p).run(stub_event)
        assert result.output_vector is not None


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: registry + scorer with stub snapshots
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_registry_has_all_required_tools(self):
        """Registry should contain all real (non-mock) tools.

        With ENABLE_MOCK_TOOLS=False (production default), MockRandomContext is
        excluded from the registry. All 9 real tools (4 external + 5 market-structure)
        must be present. The pure-noise mock_random_context must be absent in
        production mode.
        """
        from tools.registry import build_default_registry
        registry = build_default_registry()
        # External (Layer B) tools
        assert "fred_macro_tool" in registry
        assert "bls_labor_tool" in registry
        assert "weather_probability_tool" in registry
        assert "sportsbook_implied_probability_tool" in registry
        # Market-structure (Layer A) real tools
        assert "mock_price_signal" in registry
        assert "snapshot_volatility_tool" in registry
        assert "spread_compression_tool" in registry
        assert "price_jump_detector_tool" in registry
        assert "liquidity_spike_tool" in registry
        # At least 9 real tools registered
        assert len(registry) >= 9
        # Verify noise tool is excluded in production (ENABLE_MOCK_TOOLS=False)
        try:
            from config import ENABLE_MOCK_TOOLS
        except ImportError:
            ENABLE_MOCK_TOOLS = True
        if not ENABLE_MOCK_TOOLS:
            assert "mock_random_context" not in registry

    def test_signal_sum_scoring_with_external_tools(self, stub_event, tmp_path):
        from tools.registry import build_default_registry
        from engine.tool_runner import run_tools
        from engine.scorer import compute_score
        from schemas import FormulaSpec, ToolSelection, AggregationMethod

        # Use stub snapshot files
        fred_p = _make_fred_jsonl(tmp_path)
        bls_p  = _make_bls_jsonl(tmp_path)
        odds_p = _make_odds_jsonl(tmp_path)

        # Import tools with custom paths
        from prediction_agent.tools.external.fred_macro_tool import FredMacroTool
        from prediction_agent.tools.external.bls_labor_tool import BlsLaborTool
        from prediction_agent.tools.external.sportsbook_implied_probability_tool import (
            SportsbookImpliedProbabilityTool,
        )

        registry = build_default_registry()
        # Re-register with temp paths (for isolation)
        registry._tools["fred_macro_tool"] = FredMacroTool(jsonl_path=fred_p)
        registry._tools["bls_labor_tool"] = BlsLaborTool(jsonl_path=bls_p)
        registry._tools["sportsbook_implied_probability_tool"] = SportsbookImpliedProbabilityTool(jsonl_path=odds_p)

        formula = FormulaSpec(
            selections=[
                ToolSelection(tool_name="fred_macro_tool", weight=0.33),
                ToolSelection(tool_name="bls_labor_tool", weight=0.34),
                ToolSelection(tool_name="sportsbook_implied_probability_tool", weight=0.33),
            ],
            aggregation=AggregationMethod.WEIGHTED_SUM,
            threshold=0.55,
            rationale="End-to-end test with external tools",
        )

        tool_outputs, statuses = run_tools(stub_event, formula, registry)
        assert len(tool_outputs) == 3

        score = compute_score(tool_outputs, formula, scoring_mode="signal_sum")
        assert 0.0 <= score.final_score <= 1.0
        assert isinstance(score.bet_triggered, bool)

    def test_probability_edge_scoring_mode(self, stub_event, tmp_path):
        from engine.scorer import compute_score, _sigmoid
        from schemas import FormulaSpec, ToolSelection, ToolOutput, AggregationMethod

        # Simulate sportsbook tool returning mean_implied_prob = 0.61
        tool_outputs = [
            ToolOutput(
                tool_name="sportsbook_implied_probability_tool",
                output_vector=[0.61, 0.02, 0.50, 0.05],
                metadata={"confidence": 0.8},
            ),
        ]
        formula = FormulaSpec(
            selections=[ToolSelection(tool_name="sportsbook_implied_probability_tool", weight=1.0)],
            aggregation=AggregationMethod.WEIGHTED_SUM,
            threshold=0.05,
            rationale="Probability edge test",
        )

        score = compute_score(
            tool_outputs, formula,
            scoring_mode="probability_edge",
            current_market_price=0.53,  # Kalshi market price
        )

        # With USE_LOGISTIC_CALIBRATION=False (default: linear opinion pool):
        # z = 0.61, p_model = clamp(0.61, 0, 1) = 0.61
        # edge = 0.61 - 0.53 = 0.08
        # abs(edge) = 0.08 >= 0.05 → triggered
        assert score.raw_score_z is not None
        assert abs(score.raw_score_z - 0.61) < 0.01
        assert score.model_probability is not None
        assert abs(score.model_probability - 0.61) < 0.01  # Linear opinion pool (no calibration)
        assert score.edge is not None
        expected_edge = 0.61 - 0.53
        assert abs(score.edge - expected_edge) < 1e-6
        assert score.bet_triggered is True
        assert score.scoring_mode == "probability_edge"
