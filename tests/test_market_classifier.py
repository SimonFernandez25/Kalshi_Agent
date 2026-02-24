"""
Tests for engine/market_classifier.py and agent/nodes._deterministic_fallback.

Covers:
  - Domain classification accuracy for each domain
  - Fallback to "other" for unrecognised markets
  - get_domain_tools: filtering and safety net
  - get_domain_tools_with_weather: outdoor-sport detection
  - _deterministic_fallback: domain-aware equal-weight distribution
  - _deterministic_fallback: last-resort (no tools)
  - _market_is_valid (from main.py): price sanity checks
"""

from __future__ import annotations

import pytest

from engine.market_classifier import (
    DOMAIN_TOOLS,
    classify_market,
    get_domain_tools,
    get_domain_tools_with_weather,
    is_outdoor_sport,
)


# ── classify_market ────────────────────────────────────────────────────────────

class TestClassifyMarket:
    def test_nba_market_is_sports(self):
        assert classify_market("NBA-LAL-BOS-001", "Lakers vs Celtics: Lakers Win") == "sports"

    def test_nfl_market_is_sports(self):
        assert classify_market("NFL-KC-PHI", "Chiefs vs Eagles: Total Points Over 44.5") == "sports"

    def test_ncaa_market_is_sports(self):
        assert classify_market("NCAAB-2025", "NCAA College basketball championship winner") == "sports"

    def test_mlb_market_is_sports(self):
        assert classify_market("MLB-NYY-BOS", "Yankees vs Red Sox: Yankees Win") == "sports"

    def test_player_prop_is_sports(self):
        assert classify_market("PROP-NBA-001", "LeBron James prop bet double-double tonight") == "sports"

    def test_cpi_market_is_economics(self):
        assert classify_market("CPI-MAY2025", "Will CPI exceed 3% in May 2025?") == "economics"

    def test_fed_rate_is_economics(self):
        assert classify_market("FOMC-2025-07", "Will the Fed raise interest rates in July?") == "economics"

    def test_unemployment_is_economics(self):
        assert classify_market("BLS-UNEMP-2025", "Will unemployment exceed 4.5%?") == "economics"

    def test_gdp_is_economics(self):
        assert classify_market("GDP-Q2-2025", "Will Q2 GDP come in above 2%?") == "economics"

    def test_election_is_politics(self):
        assert classify_market("PRES-2028", "Who will win the 2028 presidential election?") == "politics"

    def test_senate_vote_is_politics(self):
        assert classify_market("SENATE-VOTE-001", "Will the Senate pass the bill?") == "politics"

    def test_approval_rating_is_politics(self):
        assert classify_market("APPROVAL-001", "Will presidential approval exceed 50%?") == "politics"

    def test_hurricane_is_weather(self):
        assert classify_market("WTHR-2025", "Will a hurricane make landfall in Florida?") == "weather"

    def test_rain_market_is_weather(self):
        assert classify_market("RAIN-NYC", "Will it rain in New York this weekend?") == "weather"

    def test_temperature_is_weather(self):
        assert classify_market("TEMP-CHI", "Will temperature exceed 90F in Chicago?") == "weather"

    def test_bitcoin_is_crypto(self):
        assert classify_market("BTC-100K", "Will Bitcoin exceed $100,000?") == "crypto"

    def test_ethereum_is_crypto(self):
        assert classify_market("ETH-5K", "Will Ethereum reach $5,000?") == "crypto"

    def test_crypto_keyword(self):
        assert classify_market("CRYPTO-001", "Crypto market cap above 3 trillion") == "crypto"

    def test_unknown_market_is_other(self):
        assert classify_market("XYZ-999", "Some completely unrelated topic") == "other"

    def test_empty_strings_is_other(self):
        assert classify_market("", "") == "other"

    def test_case_insensitive(self):
        # Uppercase NBA should still classify as sports (text is lowercased internally)
        assert classify_market("NBA-GAME-001", "NBA PLAYOFF GAME") == "sports"

    def test_mixed_keywords_prefers_first_domain(self):
        # "win" is sports but "bitcoin" is crypto — first match (sports) wins
        result = classify_market("GAME-001", "Will the warriors win today? Bitcoin irrelevant")
        assert result == "sports"


# ── get_domain_tools ───────────────────────────────────────────────────────────

class TestGetDomainTools:
    _ALL_TOOLS = [
        "mock_price_signal",
        "snapshot_volatility_tool",
        "spread_compression_tool",
        "price_jump_detector_tool",
        "liquidity_spike_tool",
        "fred_macro_tool",
        "bls_labor_tool",
        "weather_probability_tool",
        "sportsbook_implied_probability_tool",
    ]

    def test_sports_gets_sportsbook_tool(self):
        tools = get_domain_tools("sports", self._ALL_TOOLS)
        assert "sportsbook_implied_probability_tool" in tools

    def test_sports_excludes_fred_macro(self):
        tools = get_domain_tools("sports", self._ALL_TOOLS)
        assert "fred_macro_tool" not in tools

    def test_sports_excludes_bls_labor(self):
        tools = get_domain_tools("sports", self._ALL_TOOLS)
        assert "bls_labor_tool" not in tools

    def test_economics_gets_fred_macro(self):
        tools = get_domain_tools("economics", self._ALL_TOOLS)
        assert "fred_macro_tool" in tools

    def test_economics_gets_bls_labor(self):
        tools = get_domain_tools("economics", self._ALL_TOOLS)
        assert "bls_labor_tool" in tools

    def test_economics_excludes_sportsbook(self):
        tools = get_domain_tools("economics", self._ALL_TOOLS)
        assert "sportsbook_implied_probability_tool" not in tools

    def test_weather_domain_gets_weather_tool(self):
        tools = get_domain_tools("weather", self._ALL_TOOLS)
        assert "weather_probability_tool" in tools

    def test_weather_domain_excludes_sportsbook(self):
        tools = get_domain_tools("weather", self._ALL_TOOLS)
        assert "sportsbook_implied_probability_tool" not in tools

    def test_unknown_domain_falls_to_other(self):
        # "other" should not include sportsbook or macro tools
        tools = get_domain_tools("other", self._ALL_TOOLS)
        assert "sportsbook_implied_probability_tool" not in tools
        assert "fred_macro_tool" not in tools

    def test_safety_net_when_no_match(self):
        # Registry has only tools not in any allowlist
        sparse = ["mock_price_signal"]
        tools = get_domain_tools("sports", sparse)
        # Safety net: returns something rather than empty list
        assert len(tools) >= 1

    def test_returns_only_subset_of_available(self):
        # Should not invent tools not in available_tools
        tools = get_domain_tools("sports", ["snapshot_volatility_tool"])
        for t in tools:
            assert t in ["snapshot_volatility_tool"]

    def test_unrecognised_domain_returns_other_allowlist(self):
        tools = get_domain_tools("galactic_markets", self._ALL_TOOLS)
        # Falls back to "other" domain
        for t in tools:
            assert t in DOMAIN_TOOLS["other"]


# ── is_outdoor_sport / get_domain_tools_with_weather ──────────────────────────

class TestOutdoorSportDetection:
    def test_nfl_is_outdoor(self):
        assert is_outdoor_sport("NFL-KC-PHI", "Chiefs vs Eagles NFL Game") is True

    def test_mlb_is_outdoor(self):
        assert is_outdoor_sport("MLB-NYY-BOS", "Yankees vs Red Sox baseball game") is True

    def test_golf_is_outdoor(self):
        assert is_outdoor_sport("GOLF-MASTERS", "Masters Golf Tournament Winner") is True

    def test_nba_is_not_outdoor(self):
        # NBA = indoor — no outdoor keywords
        assert is_outdoor_sport("NBA-LAL-BOS", "Lakers vs Celtics NBA game") is False

    def test_weather_added_for_outdoor_sports(self):
        all_tools = [
            "sportsbook_implied_probability_tool",
            "snapshot_volatility_tool",
            "weather_probability_tool",
        ]
        tools = get_domain_tools_with_weather(
            "sports", all_tools, "NFL-KC-PHI", "Chiefs vs Eagles NFL game"
        )
        assert "weather_probability_tool" in tools

    def test_weather_not_added_for_indoor_sports(self):
        all_tools = [
            "sportsbook_implied_probability_tool",
            "snapshot_volatility_tool",
            "weather_probability_tool",
        ]
        tools = get_domain_tools_with_weather(
            "sports", all_tools, "NBA-LAL-BOS", "Lakers vs Celtics NBA game"
        )
        assert "weather_probability_tool" not in tools

    def test_weather_not_double_added(self):
        # weather already in domain allowlist for "weather" domain — no duplicates
        all_tools = ["weather_probability_tool", "snapshot_volatility_tool"]
        tools = get_domain_tools_with_weather(
            "weather", all_tools, "RAIN-NYC", "Will it rain in New York?"
        )
        assert tools.count("weather_probability_tool") == 1


# ── _deterministic_fallback ────────────────────────────────────────────────────

class TestDeterministicFallback:
    """Test agent.nodes._deterministic_fallback with the updated signature."""

    def _fallback(self, tools_list):
        from agent.nodes import _deterministic_fallback
        return _deterministic_fallback({}, tools_list)

    def test_equal_weight_two_tools(self):
        tools = [
            {"name": "sportsbook_implied_probability_tool", "description": "..."},
            {"name": "snapshot_volatility_tool", "description": "..."},
        ]
        spec = self._fallback(tools)
        weights = [s["weight"] for s in spec["selections"]]
        assert abs(sum(weights) - 1.0) < 1e-6
        assert all(abs(w - 0.5) < 1e-6 for w in weights)

    def test_equal_weight_three_tools(self):
        tools = [
            {"name": "sportsbook_implied_probability_tool", "description": ""},
            {"name": "snapshot_volatility_tool", "description": ""},
            {"name": "price_jump_detector_tool", "description": ""},
        ]
        spec = self._fallback(tools)
        weights = [s["weight"] for s in spec["selections"]]
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_caps_at_four_tools(self):
        tools = [{"name": f"tool_{i}", "description": ""} for i in range(6)]
        spec = self._fallback(tools)
        assert len(spec["selections"]) == 4

    def test_excludes_mock_random_context(self):
        tools = [
            {"name": "mock_random_context", "description": "noise"},
            {"name": "snapshot_volatility_tool", "description": "real"},
        ]
        spec = self._fallback(tools)
        names = [s["tool_name"] for s in spec["selections"]]
        assert "mock_random_context" not in names

    def test_last_resort_no_tools(self):
        spec = self._fallback([])
        # Should produce mock_price_signal as last resort
        names = [s["tool_name"] for s in spec["selections"]]
        assert names == ["mock_price_signal"]
        weights = [s["weight"] for s in spec["selections"]]
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_last_resort_only_mock_tools(self):
        # Only mock_random_context in registry (should be excluded → last resort)
        tools = [{"name": "mock_random_context", "description": "noise"}]
        spec = self._fallback(tools)
        names = [s["tool_name"] for s in spec["selections"]]
        assert names == ["mock_price_signal"]

    def test_none_tools_list(self):
        # Old call sites that pass None (backward compat)
        spec = self._fallback(None)
        assert len(spec["selections"]) >= 1

    def test_valid_pydantic_schema(self):
        from schemas import FormulaSpec
        tools = [
            {"name": "sportsbook_implied_probability_tool", "description": ""},
            {"name": "snapshot_volatility_tool", "description": ""},
        ]
        spec = self._fallback(tools)
        # Should not raise
        validated = FormulaSpec(**spec)
        assert abs(sum(s.weight for s in validated.selections) - 1.0) < 1e-6


# ── _market_is_valid (from main.py) ───────────────────────────────────────────

class TestMarketIsValid:
    """Integration test for the market sanity pre-check in main.py."""

    def _make_event(self, price: float):
        from schemas import EventInput
        from datetime import datetime, timezone
        return EventInput(
            event_id="TEST-001",
            market_id="TEST-001",
            market_title="Test Market",
            current_price=price,
            timestamp=datetime.now(timezone.utc),
        )

    def test_valid_price_passes(self):
        from main import _market_is_valid
        event = self._make_event(0.52)
        valid, reason = _market_is_valid(event)
        assert valid is True
        assert reason == ""

    def test_zero_price_abstains(self):
        from main import _market_is_valid
        event = self._make_event(0.0)
        valid, reason = _market_is_valid(event)
        assert valid is False
        assert "no signal" in reason

    def test_below_min_price_abstains(self):
        from main import _market_is_valid
        event = self._make_event(0.005)
        valid, reason = _market_is_valid(event)
        assert valid is False

    def test_at_min_valid_price_abstains(self):
        # Exactly at MIN_VALID_PRICE (0.01) should still be <= → abstain
        from main import _market_is_valid
        event = self._make_event(0.01)
        valid, reason = _market_is_valid(event)
        assert valid is False

    def test_just_above_min_passes(self):
        from main import _market_is_valid
        event = self._make_event(0.02)
        valid, reason = _market_is_valid(event)
        assert valid is True

    def test_near_one_abstains(self):
        # 0.995 = 99.5c — market likely resolved
        from main import _market_is_valid
        event = self._make_event(0.995)
        valid, reason = _market_is_valid(event)
        assert valid is False
        assert "resolved" in reason

    def test_at_exactly_099_abstains(self):
        from main import _market_is_valid
        event = self._make_event(0.99)
        valid, reason = _market_is_valid(event)
        assert valid is False

    def test_boundary_just_below_099_passes(self):
        from main import _market_is_valid
        event = self._make_event(0.98)
        valid, reason = _market_is_valid(event)
        assert valid is True
