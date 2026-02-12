"""
Deterministic tool runner.

Takes the FormulaSpec (agent output) + EventInput,
executes the selected tools through the registry,
and returns raw ToolOutputs.

NO LLM logic here. Pure execution.
"""

from __future__ import annotations

import logging
from typing import List

from prediction_agent.schemas import EventInput, FormulaSpec, ToolOutput
from prediction_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def run_tools(
    event: EventInput,
    formula: FormulaSpec,
    registry: ToolRegistry,
) -> List[ToolOutput]:
    """
    Execute each tool selected by the agent.

    Args:
        event: The market event to analyze.
        formula: Agent's tool selections and weights.
        registry: The tool registry.

    Returns:
        Ordered list of ToolOutputs (same order as formula.selections).
    """
    outputs: List[ToolOutput] = []

    for selection in formula.selections:
        tool = registry.get(selection.tool_name)
        logger.info("Running tool: %s (weight=%.2f)", selection.tool_name, selection.weight)

        output = tool.run(event, **selection.tool_inputs)
        outputs.append(output)

        logger.info(
            "  -> %s output_vector=%s",
            selection.tool_name,
            output.output_vector,
        )

    return outputs
