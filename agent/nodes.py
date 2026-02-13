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
from config import AVAILABLE_TOOLS, LLM_MODEL, LLM_TEMPERATURE
from schemas import EventInput, FormulaSpec, ToolSelection

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


# ──────────────────────────────────────────────
# Node: Call LLM to select tools
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# Node: Call LLM to select tools
# ──────────────────────────────────────────────

def select_tools_node(state: AgentState) -> AgentState:
    """
    Call the LLM (AWS Bedrock) to produce a FormulaSpec.
    Falls back to a deterministic default if LLM fails.
    """
    event_dict = state["event_input"]
    tools_list = state["tools_list"]

    try:
        from langchain_aws import ChatBedrock
        from langchain_core.messages import HumanMessage

        # Initialize Bedrock client with specific model
        llm = ChatBedrock(
            model_id="arn:aws:bedrock:us-east-1:770182232673:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            provider="anthropic",
            region_name="us-east-1",
            temperature=LLM_TEMPERATURE,
        )

        user_prompt = build_user_prompt(event_dict, tools_list)
        
        # Invoke Bedrock
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

        # Validate tool names against registry
        for sel in spec.selections:
            if sel.tool_name not in [t["name"] for t in tools_list]:
                raise ValueError(f"Agent selected unknown tool: {sel.tool_name}")

        logger.info("Agent selected tools: %s", [s.tool_name for s in spec.selections])
        state["formula_spec"] = spec.model_dump()
        state["error"] = None

    except Exception as exc:
        logger.warning("LLM call failed (%s) — using deterministic fallback.", exc)
        state["formula_spec"] = _deterministic_fallback(event_dict)
        state["error"] = str(exc)

    return state


# ──────────────────────────────────────────────
# Node: Validate formula spec
# ──────────────────────────────────────────────

def validate_spec_node(state: AgentState) -> AgentState:
    """Ensure the FormulaSpec is well-formed. Fix minor issues."""
    spec = state.get("formula_spec")
    if spec is None:
        state["formula_spec"] = _deterministic_fallback(state["event_input"])
        state["error"] = "No spec produced — applied fallback."
        return state

    # Re-validate through Pydantic
    try:
        validated = FormulaSpec(**spec)
        state["formula_spec"] = validated.model_dump()
    except Exception as exc:
        logger.warning("Spec validation failed: %s — using fallback.", exc)
        state["formula_spec"] = _deterministic_fallback(state["event_input"])
        state["error"] = str(exc)

    return state


# ──────────────────────────────────────────────
# Deterministic Fallback
# ──────────────────────────────────────────────

def _deterministic_fallback(event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    If LLM is unavailable or produces garbage, use this
    safe default: both tools, equal weights, threshold = 0.6.
    """
    spec = FormulaSpec(
        selections=[
            ToolSelection(tool_name="mock_price_signal", tool_inputs={}, weight=0.5),
            ToolSelection(tool_name="mock_random_context", tool_inputs={}, weight=0.5),
        ],
        aggregation="weighted_sum",
        threshold=0.6,
        rationale="Deterministic fallback: equal-weight both mock tools, threshold 0.6.",
    )
    return spec.model_dump()
