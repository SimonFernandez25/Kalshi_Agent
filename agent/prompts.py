"""
Prompt templates for the LangGraph prediction agent.

The agent's ONLY job is to select tools, weights, and threshold.
It must NOT compute scores — that's the deterministic engine's job.
"""

# ── Tool schema definitions ────────────────────────────────────────────────────
# Structured metadata about each tool: output vector semantics, use case,
# relevant market types. The agent uses this to make informed selections.

TOOL_SCHEMAS = {
    # ── Layer A: Market Structure ──────────────────────────────────────────────
    "mock_price_signal": {
        "output_vector": ["current_yes_price [0,1]"],
        "use_case": "Direct Kalshi price signal. Baseline reference for any market.",
        "market_types": ["all"],
        "is_probability_signal": False,
    },
    "mock_random_context": {
        "output_vector": ["seeded_random [0,1]"],
        "use_case": "Deterministic random context placeholder. Low signal quality.",
        "market_types": ["all"],
        "is_probability_signal": False,
    },
    "snapshot_volatility_tool": {
        "output_vector": [
            "volatility [0,1]", "mean_price [0,1]", "price_range [0,1]",
            "recency_weight [0,1]", "sample_confidence [0,1]"
        ],
        "use_case": "Price volatility from recent snapshots. Higher = more uncertain market.",
        "market_types": ["sports", "economics", "politics"],
        "is_probability_signal": False,
    },
    "spread_compression_tool": {
        "output_vector": [
            "mean_spread [0,1]", "spread_std [0,1]",
            "spread_trend [-1,1 -> 0,1]", "compression_ratio [0,1]"
        ],
        "use_case": "Bid-ask spread dynamics. Compressing spread = market conviction growing.",
        "market_types": ["sports", "economics"],
        "is_probability_signal": False,
    },
    "price_jump_detector_tool": {
        "output_vector": [
            "max_jump [0,1]", "mean_jump [0,1]",
            "jump_count [normalised]", "jump_density [0,1]"
        ],
        "use_case": "Detects sudden price moves. High jumps = news/information shock.",
        "market_types": ["sports", "economics", "politics"],
        "is_probability_signal": False,
    },
    "liquidity_spike_tool": {
        "output_vector": [
            "mean_liquidity [normalised]", "std_liquidity [normalised]",
            "latest_vs_mean_ratio [0,1]", "zscore_latest [normalised]"
        ],
        "use_case": "Liquidity surge detection. Spike = informed traders entering.",
        "market_types": ["sports", "economics"],
        "is_probability_signal": False,
    },
    # ── Layer B: External Fundamentals ────────────────────────────────────────
    "fred_macro_tool": {
        "output_vector": [
            "latest_value [0,1]", "rolling_3_mean [0,1]",
            "rolling_3_trend [0,1]", "zscore_latest [0,1]", "volatility_last_6 [0,1]"
        ],
        "use_case": (
            "Macro-economic context (Fed funds rate, CPI, VIX). "
            "Useful when predicting economic event outcomes (rate decisions, CPI prints). "
            "Pass series_id kwarg to select series: FEDFUNDS, UNRATE, CPIAUCSL, T10Y2Y, VIXCLS."
        ),
        "market_types": ["economics", "politics"],
        "is_probability_signal": False,
        "tool_inputs_example": {"series_id": "FEDFUNDS"},
    },
    "bls_labor_tool": {
        "output_vector": [
            "unemployment_rate [0,1]", "monthly_delta [0,1]",
            "3m_trend [0,1]", "yoy_change [0,1]", "surprise_proxy [0,1]"
        ],
        "use_case": (
            "BLS labor market data. Useful for employment report markets, "
            "consumer spending forecasts. Surprise proxy signals beat/miss. "
            "Pass series_id: LNS14000000 (unemployment), CUSR0000SA0 (CPI-U)."
        ),
        "market_types": ["economics"],
        "is_probability_signal": False,
        "tool_inputs_example": {"series_id": "LNS14000000"},
    },
    "weather_probability_tool": {
        "output_vector": [
            "forecast_probability [0,1]", "precipitation_mm [0,1]",
            "temp_anomaly [0,1]", "forecast_confidence [0,1]",
            "model_disagreement_proxy [0,1]"
        ],
        "use_case": (
            "Weather forecast for relevant arena cities. "
            "output_vector[0] is a direct rain probability signal. "
            "Useful for outdoor markets or game-attendance-sensitive outcomes. "
            "Pass location kwarg (e.g. 'Boston') to target a city."
        ),
        "market_types": ["sports", "weather"],
        "is_probability_signal": True,
        "probability_element": 0,
        "tool_inputs_example": {"location": "Boston"},
    },
    "sportsbook_implied_probability_tool": {
        "output_vector": [
            "mean_implied_probability [0,1]", "cross_book_variance [0,1]",
            "line_movement_rate [0,1]", "implied_volatility_proxy [0,1]"
        ],
        "use_case": (
            "Consensus sportsbook implied home-win probability. "
            "output_vector[0] IS the model probability for probability_edge scoring. "
            "edge = output_vector[0] - current_market_price. "
            "High cross_book_variance = bookmakers disagree = uncertain line. "
            "Best for NBA/NFL markets where sportsbook data is available."
        ),
        "market_types": ["sports"],
        "is_probability_signal": True,
        "probability_element": 0,
    },
}


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a prediction tool selector for Kalshi prediction markets.

Your ONLY job is to:
1. Review the available tools, their output vector semantics, and the market event.
2. Select which tools to use (at least one).
3. Assign a weight to each selected tool (weights must sum to 1.0).
4. Choose a threshold (0.0 to 1.0).
5. Provide a brief rationale citing which signals drove your selections.

STRICT RULES:
- You can ONLY select tools from the available tools list provided.
- You CANNOT invent new tools.
- You CANNOT compute scores — the engine does that.
- Weights must be between 0.0 and 1.0 and sum to 1.0.
- Threshold must be between 0.0 and 1.0.
- Aggregation is ALWAYS "weighted_sum".
- You CANNOT call APIs, modify data, or override tool outputs.
- You CANNOT fabricate data — use only signals the tools provide.

TOOL SELECTION GUIDANCE:
- For SPORTS markets (NBA, NFL): prefer sportsbook_implied_probability_tool (direct probability signal),
  snapshot_volatility_tool, price_jump_detector_tool.
- For ECONOMICS markets (CPI, rates, employment): prefer fred_macro_tool, bls_labor_tool,
  snapshot_volatility_tool.
- For WEATHER-sensitive markets: prefer weather_probability_tool.
- For any market: spread_compression_tool and liquidity_spike_tool add structural context.
- Tools marked is_probability_signal=true have output_vector[0] as a direct probability estimate.
  These are especially powerful when the scoring_mode is probability_edge.

Respond with ONLY valid JSON matching the FormulaSpec schema. No markdown, no explanation outside the JSON.
"""

USER_PROMPT_TEMPLATE = """## Market Event
- Event ID: {event_id}
- Market ID: {market_id}
- Title: {market_title}
- Current YES Price: {current_price}
- Timestamp: {timestamp}
- Scoring Mode: {scoring_mode}

## Available Tools
{tools_description}

## Instructions
Select tools appropriate for this market type.
Assign weights (must sum to 1.0). Set threshold. Provide rationale.

Respond with JSON matching this exact schema:
{{
  "selections": [
    {{
      "tool_name": "<name from available tools>",
      "tool_inputs": {{}},
      "weight": <float 0-1>
    }}
  ],
  "aggregation": "weighted_sum",
  "threshold": <float 0-1>,
  "rationale": "<brief explanation citing tool choices>"
}}
"""


# ── Formatters ────────────────────────────────────────────────────────────────

def format_tools_description(tools_list: list[dict]) -> str:
    """
    Format registry tools into a detailed prompt block using TOOL_SCHEMAS.
    Falls back to plain description if schema not found.
    """
    lines = []
    for i, t in enumerate(tools_list, 1):
        name = t["name"]
        desc = t["description"]
        schema = TOOL_SCHEMAS.get(name, {})

        lines.append(f"{i}. **{name}**")
        lines.append(f"   Description: {desc}")

        if schema.get("output_vector"):
            vec_str = ", ".join(schema["output_vector"])
            lines.append(f"   Output vector: [{vec_str}]")

        if schema.get("market_types"):
            lines.append(f"   Relevant markets: {', '.join(schema['market_types'])}")

        if schema.get("is_probability_signal"):
            lines.append(
                f"   ** PROBABILITY SIGNAL: output_vector[{schema.get('probability_element', 0)}] "
                f"is a direct [0,1] probability estimate **"
            )

        if schema.get("tool_inputs_example"):
            lines.append(f"   Example tool_inputs: {schema['tool_inputs_example']}")

        lines.append("")

    return "\n".join(lines)


def build_user_prompt(event_dict: dict, tools_list: list[dict], scoring_mode: str = "signal_sum") -> str:
    """Build the full user prompt from event data and tool list."""
    return USER_PROMPT_TEMPLATE.format(
        event_id=event_dict["event_id"],
        market_id=event_dict["market_id"],
        market_title=event_dict["market_title"],
        current_price=event_dict["current_price"],
        timestamp=event_dict["timestamp"],
        scoring_mode=scoring_mode,
        tools_description=format_tools_description(tools_list),
    )
