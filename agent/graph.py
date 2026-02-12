"""
LangGraph definition for the prediction agent.

Linear graph:
  select_tools → validate_spec → END

The agent does NOT execute tools or compute scores.
It only produces a FormulaSpec that the deterministic engine consumes.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from prediction_agent.agent.nodes import (
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
    logger.info("Agent graph compiled: select_tools → validate_spec → END")
    return compiled
