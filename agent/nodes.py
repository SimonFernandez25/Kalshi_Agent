"""
LangGraph node functions.

Each node is a pure function: State → State.
The graph wires them together.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, TypedDict

from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from config import LLM_MODEL, LLM_TEMPERATURE, SCORING_MODE
from engine.market_classifier import classify_market, get_domain_tools_with_weather
from schemas import FormulaSpec, ToolSelection

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Graph State
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    """State that flows through the LangGraph."""
    event_input: Dict[str, Any]
    tools_list: List[Dict[str, str]]
    formula_spec: Dict[str, Any] | None
    error: str | None
    market_domain: str | None        # classified domain (sports/economics/politics/…)
    scoring_mode: str | None         # signal_sum or probability_edge


# ──────────────────────────────────────────────
# Node: Call LLM to select tools
# ──────────────────────────────────────────────

def select_tools_node(state: AgentState) -> AgentState:
    """
    Call the LLM (AWS Bedrock) to produce a FormulaSpec.

    Before calling the LLM:
      1. Classify the market domain (sports/economics/politics/…)
      2. Filter tools_list to domain-relevant tools only
      3. Determine scoring_mode (probability_edge for sports, signal_sum otherwise)

    Falls back to a deterministic domain-aware default if LLM fails.
    """
    event_dict = state["event_input"]
    all_tools_list = state["tools_list"]

    # ── Domain classification ──────────────────────────────────────────────────
    market_id    = event_dict.get("market_id", "")
    market_title = event_dict.get("market_title", "")
    domain = classify_market(market_id, market_title)
    state["market_domain"] = domain

    all_tool_names = [t["name"] for t in all_tools_list]
    domain_tool_names = get_domain_tools_with_weather(domain, all_tool_names, market_id, market_title)
    tools_list = [t for t in all_tools_list if t["name"] in domain_tool_names]

    # ── Scoring mode ───────────────────────────────────────────────────────────
    # Use probability_edge for sports (sportsbook tool provides direct probability)
    scoring_mode = "probability_edge" if domain == "sports" else SCORING_MODE
    state["scoring_mode"] = scoring_mode

    logger.info("Market domain: %s | Tools filtered: %s | Scoring mode: %s",
                domain, [t["name"] for t in tools_list], scoring_mode)

    try:
        from langchain_aws import ChatBedrock
        from langchain_core.messages import HumanMessage

        llm = ChatBedrock(
            model_id="arn:aws:bedrock:us-east-1:770182232673:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            provider="anthropic",
            region_name="us-east-1",
            temperature=LLM_TEMPERATURE,
        )

        user_prompt = build_user_prompt(event_dict, tools_list, scoring_mode=scoring_mode)

        response = llm.invoke([
            HumanMessage(content=SYSTEM_PROMPT + "\n\n" + user_prompt)
        ])

        raw_text = response.content.strip()

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]
            if raw_text.endswith("```"):
                raw_text = raw_text[: raw_text.rfind("```")]
            raw_text = raw_text.strip()

        spec_dict = json.loads(raw_text)
        spec = FormulaSpec(**spec_dict)

        # Validate tool names against domain-filtered list
        allowed_names = [t["name"] for t in tools_list]
        for sel in spec.selections:
            if sel.tool_name not in allowed_names:
                raise ValueError(f"Agent selected out-of-domain tool: {sel.tool_name}")

        logger.info("Agent selected tools: %s", [s.tool_name for s in spec.selections])
        state["formula_spec"] = spec.model_dump()
        state["error"] = None

    except Exception as exc:
        logger.warning("LLM call failed (%s) — using deterministic fallback.", exc)
        state["formula_spec"] = _deterministic_fallback(event_dict, tools_list)
        state["error"] = str(exc)

    return state


# ──────────────────────────────────────────────
# Node: Validate formula spec
# ──────────────────────────────────────────────

def validate_spec_node(state: AgentState) -> AgentState:
    """Ensure the FormulaSpec is well-formed. Fix minor issues."""
    spec = state.get("formula_spec")
    tools_list = state.get("tools_list", [])

    if spec is None:
        state["formula_spec"] = _deterministic_fallback(state["event_input"], tools_list)
        state["error"] = "No spec produced — applied fallback."
        return state

    # Re-validate through Pydantic
    try:
        validated = FormulaSpec(**spec)
        state["formula_spec"] = validated.model_dump()
    except Exception as exc:
        logger.warning("Spec validation failed: %s — using fallback.", exc)
        state["formula_spec"] = _deterministic_fallback(state["event_input"], tools_list)
        state["error"] = str(exc)

    return state


# ──────────────────────────────────────────────
# Deterministic Fallback
# ──────────────────────────────────────────────

def _deterministic_fallback(
    event_dict: Dict[str, Any],
    tools_list: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """
    If LLM is unavailable or produces garbage, produce a domain-aware safe default.

    Uses the domain-filtered tools_list if provided, distributing equal weight
    across all available tools (up to 4). Falls back to mock_price_signal only
    when no real tools are available.

    Args:
        event_dict: The event dict (used for logging only).
        tools_list: Domain-filtered list of tool dicts from the registry.
                    Each dict has keys "name" and "description".
    """
    # Prefer real tools over mock fallbacks
    _MOCK_TOOLS = {"mock_random_context"}

    available = []
    if tools_list:
        available = [t["name"] for t in tools_list if t["name"] not in _MOCK_TOOLS]

    # Cap at 4 tools to keep weights reasonable
    available = available[:4]

    if available:
        n = len(available)
        weight = round(1.0 / n, 6)
        # Adjust last weight to ensure exact sum of 1.0
        weights = [weight] * n
        weights[-1] = round(1.0 - weight * (n - 1), 6)

        selections = [
            ToolSelection(tool_name=name, tool_inputs={}, weight=w)
            for name, w in zip(available, weights)
        ]
        rationale = (
            f"Deterministic fallback: equal-weight {n} domain tools "
            f"({', '.join(available)}), threshold 0.55."
        )
        threshold = 0.55
    else:
        # Last resort: mock_price_signal is always registered
        selections = [
            ToolSelection(tool_name="mock_price_signal", tool_inputs={}, weight=1.0),
        ]
        rationale = "Deterministic fallback (no tools): mock_price_signal only, threshold 0.5."
        threshold = 0.5

    logger.info("Fallback spec: %s", [s.tool_name for s in selections])

    spec = FormulaSpec(
        selections=selections,
        aggregation="weighted_sum",
        threshold=threshold,
        rationale=rationale,
    )
    return spec.model_dump()
