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

from schemas import EventInput, FormulaSpec, ToolOutput
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def run_tools(
    event: EventInput,
    formula: FormulaSpec,
    registry: ToolRegistry,
) -> tuple[List[ToolOutput], List[ToolExecutionStatus]]:
    """
    Execute each tool selected by the agent with strict validation.

    Args:
        event: The market event to analyze.
        formula: Agent's tool selections and weights.
        registry: The tool registry.

    Returns:
        Tuple of (outputs, statuses) - ordered lists matching formula.selections.
    """
    import time
    from schemas import ToolExecutionStatus
    
    outputs: List[ToolOutput] = []
    statuses: List[ToolExecutionStatus] = []

    for selection in formula.selections:
        tool = registry.get(selection.tool_name)
        logger.info("Running tool: %s (weight=%.2f)", selection.tool_name, selection.weight)
        
        start_time = time.perf_counter()
        
        try:
            output = tool.run(event, **selection.tool_inputs)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            statuses.append(ToolExecutionStatus(
                tool_name=selection.tool_name,
                success=True,
                latency_ms=latency_ms,
                error=None
            ))
            outputs.append(output)
            
            logger.info(
                "  -> %s output_vector=%s (%.1fms)",
                selection.tool_name,
                output.output_vector,
                latency_ms
            )
            
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"{type(exc).__name__}: {str(exc)}"
            
            statuses.append(ToolExecutionStatus(
                tool_name=selection.tool_name,
                success=False,
                latency_ms=latency_ms,
                error=error_msg
            ))
            
            # Fallback: zero vector
            outputs.append(ToolOutput(
                tool_name=selection.tool_name,
                output_vector=[0.0],
                metadata={"error": error_msg}
            ))
            
            logger.warning("Tool '%s' failed: %s", selection.tool_name, error_msg)
            print(f"[WARNING] Tool execution failed: {selection.tool_name} - {error_msg}")

    return outputs, statuses

