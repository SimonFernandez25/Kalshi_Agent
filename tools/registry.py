"""
Tool registry — single source of truth for available tools.

The agent can ONLY pick from tools registered here.
No dynamic discovery, no plugins, no auto-detection.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from prediction_agent.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Holds all registered tools. Agent queries this to know what's available."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool. Raises if duplicate name."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered.")
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> BaseTool:
        """Retrieve a tool by name. Raises KeyError if not found."""
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not in registry. Available: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def list_tools(self) -> List[Dict[str, str]]:
        """Return tool metadata for the agent prompt."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def build_default_registry() -> ToolRegistry:
    """Create and populate the registry with all available tools."""
    from prediction_agent.tools.mock_tools import MockPriceSignal, MockRandomContext
    from prediction_agent.tools.snapshot_volatility_tool import SnapshotVolatilityTool
    from prediction_agent.tools.spread_compression_tool import SpreadCompressionTool
    from prediction_agent.tools.price_jump_detector_tool import PriceJumpDetectorTool
    from prediction_agent.tools.liquidity_spike_tool import LiquiditySpikeTool

    registry = ToolRegistry()
    registry.register(MockPriceSignal())
    registry.register(MockRandomContext())
    registry.register(SnapshotVolatilityTool())
    registry.register(SpreadCompressionTool())
    registry.register(PriceJumpDetectorTool())
    registry.register(LiquiditySpikeTool())
    return registry
