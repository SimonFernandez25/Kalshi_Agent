"""
LangGraph definition for the prediction agent.

Two graphs:
  1. Agent graph (main pipeline):
     select_tools -> validate_spec -> END

  2. Evolution graph (optional, post-run):
     analyze_logs -> [conditional] -> propose_tool -> build_tool -> verify_tool -> update_registry -> END

The agent graph does NOT execute tools or compute scores.
It only produces a FormulaSpec that the deterministic engine consumes.

The evolution graph runs AFTER the main pipeline completes,
gated by ENABLE_EVOLUTION config flag.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent.nodes import (
    AgentState,
    select_tools_node,
    validate_spec_node,
)

logger = logging.getLogger(__name__)


def build_agent_graph() -> StateGraph:
    """
    Construct and compile the agent graph.

    Returns a compiled StateGraph ready to invoke.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("select_tools", select_tools_node)
    workflow.add_node("validate_spec", validate_spec_node)

    # Wire edges: linear flow
    workflow.set_entry_point("select_tools")
    workflow.add_edge("select_tools", "validate_spec")
    workflow.add_edge("validate_spec", END)

    compiled = workflow.compile()
    logger.info("Agent graph compiled: select_tools -> validate_spec -> END")
    return compiled


# ------------------------------------------------------------------
# Evolution Graph (optional post-run branch)
# ------------------------------------------------------------------

class EvolutionState(TypedDict):
    """State that flows through the evolution graph."""
    log_path: str
    gap_report: Optional[Dict[str, Any]]
    tool_spec: Optional[Dict[str, Any]]
    tool_path: Optional[str]
    verification_result: Optional[Dict[str, Any]]
    registry_updated: bool
    error: Optional[str]


def _analyze_logs_node(state: EvolutionState) -> EvolutionState:
    """Analyze execution logs for capability gaps."""
    from prediction_agent.evolution.tool_gap_analyzer import analyze_gaps

    log_path = Path(state["log_path"])

    try:
        gap = analyze_gaps(log_path=log_path)
        if gap is not None:
            state["gap_report"] = gap.model_dump(mode="json")
            logger.info("Evolution: gap detected -- '%s' (priority=%.3f)",
                        gap.problem_detected, gap.priority_score)
        else:
            state["gap_report"] = None
            logger.info("Evolution: no actionable gap detected.")
    except Exception as exc:
        state["gap_report"] = None
        state["error"] = f"Gap analysis failed: {exc}"
        logger.warning("Evolution: gap analysis failed: %s", exc)

    return state


def _should_propose(state: EvolutionState) -> str:
    """Conditional edge: propose only if a gap was detected."""
    if state.get("gap_report") is not None:
        return "propose_tool"
    return END


def _propose_tool_node(state: EvolutionState) -> EvolutionState:
    """Generate a ToolSpec from the gap report."""
    from prediction_agent.evolution.schemas import GapReport
    from prediction_agent.evolution.tool_spec_generator import generate_tool_spec

    try:
        gap = GapReport(**state["gap_report"])
        spec = generate_tool_spec(gap)
        state["tool_spec"] = spec.model_dump(mode="json")
        logger.info("Evolution: proposed tool '%s'", spec.tool_name)
    except Exception as exc:
        state["tool_spec"] = None
        state["error"] = f"Tool spec generation failed: {exc}"
        logger.warning("Evolution: spec generation failed: %s", exc)

    return state


def _build_tool_node(state: EvolutionState) -> EvolutionState:
    """Build a Python tool file from the spec."""
    from prediction_agent.evolution.schemas import ToolSpec
    from prediction_agent.evolution.tool_builder import build_tool

    if state.get("tool_spec") is None:
        state["error"] = "No tool spec available for building."
        return state

    try:
        spec = ToolSpec(**state["tool_spec"])
        tool_path = build_tool(spec)
        state["tool_path"] = str(tool_path)
        logger.info("Evolution: built tool at %s", tool_path)
    except Exception as exc:
        state["tool_path"] = None
        state["error"] = f"Tool build failed: {exc}"
        logger.warning("Evolution: build failed: %s", exc)

    return state


def _verify_tool_node(state: EvolutionState) -> EvolutionState:
    """Verify the built tool passes all safety and correctness checks."""
    from prediction_agent.evolution.schemas import ToolSpec
    from prediction_agent.evolution.tool_verifier import verify_tool

    if state.get("tool_path") is None or state.get("tool_spec") is None:
        state["error"] = "No tool to verify."
        return state

    try:
        spec = ToolSpec(**state["tool_spec"])
        result = verify_tool(Path(state["tool_path"]), spec)
        state["verification_result"] = result.model_dump(mode="json")

        if result.passed:
            logger.info("Evolution: tool '%s' PASSED verification.", spec.tool_name)
        else:
            logger.warning(
                "Evolution: tool '%s' FAILED verification: %s",
                spec.tool_name,
                result.rejection_reason,
            )
    except Exception as exc:
        state["verification_result"] = None
        state["error"] = f"Verification failed: {exc}"
        logger.warning("Evolution: verification error: %s", exc)

    return state


def _update_registry_node(state: EvolutionState) -> EvolutionState:
    """Update the approved.json manifest if tool passed verification."""
    from config import GENERATED_TOOLS_DIR, EVOLUTION_REQUIRE_MANUAL_APPROVAL

    verification = state.get("verification_result")
    spec_dict = state.get("tool_spec")

    if verification is None or not verification.get("passed", False):
        state["registry_updated"] = False
        logger.info("Evolution: tool not approved. Registry unchanged.")
        return state
    
    # Check manual approval gate
    if EVOLUTION_REQUIRE_MANUAL_APPROVAL:
        state["registry_updated"] = False
        logger.info("Evolution: manual approval required - tool awaits review in pending/")
        return state

    tool_name = verification.get("tool_name", "")
    manifest_path = GENERATED_TOOLS_DIR / "approved.json"

    try:
        # Load existing manifest
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = []

        # Check for duplicates
        existing_names = {e.get("tool_name") for e in manifest if isinstance(e, dict)}
        if tool_name in existing_names:
            logger.info("Evolution: tool '%s' already in approved.json.", tool_name)
            state["registry_updated"] = False
            return state

        # Add new entry
        manifest.append({
            "tool_name": tool_name,
            "version": "0.1.0",
        })

        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        state["registry_updated"] = True
        logger.info("Evolution: tool '%s' added to approved.json.", tool_name)

    except Exception as exc:
        state["registry_updated"] = False
        state["error"] = f"Registry update failed: {exc}"
        logger.warning("Evolution: registry update failed: %s", exc)

    return state


def build_evolution_graph() -> StateGraph:
    """
    Construct and compile the evolution graph.

    This graph runs AFTER the main pipeline and is gated by ENABLE_EVOLUTION.
    Maximum 1 tool per cycle.

    Returns a compiled StateGraph ready to invoke.
    """
    workflow = StateGraph(EvolutionState)

    # Add nodes
    workflow.add_node("analyze_logs", _analyze_logs_node)
    workflow.add_node("propose_tool", _propose_tool_node)
    workflow.add_node("build_tool", _build_tool_node)
    workflow.add_node("verify_tool", _verify_tool_node)
    workflow.add_node("update_registry", _update_registry_node)

    # Wire edges
    workflow.set_entry_point("analyze_logs")
    workflow.add_conditional_edges("analyze_logs", _should_propose)
    workflow.add_edge("propose_tool", "build_tool")
    workflow.add_edge("build_tool", "verify_tool")
    workflow.add_edge("verify_tool", "update_registry")
    workflow.add_edge("update_registry", END)

    compiled = workflow.compile()
    logger.info(
        "Evolution graph compiled: analyze_logs -> [conditional] -> "
        "propose_tool -> build_tool -> verify_tool -> update_registry -> END"
    )
    return compiled
