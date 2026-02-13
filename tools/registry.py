"""
Tool registry — single source of truth for available tools.

The agent can ONLY pick from tools registered here.
No dynamic discovery, no plugins, no auto-detection.

Generated tools from the evolution pipeline are loaded through
load_approved_generated_tools() using the approved.json manifest.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Dict, List

from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Holds all registered tools. Agent queries this to know what's available."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._generated_tools: Dict[str, str] = {}  # name -> version

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

    def register_generated_tool(self, tool: BaseTool, version: str = "0.1.0") -> None:
        """
        Register a verified generated tool.

        Only called after the verifier has approved the tool.
        Does NOT allow runtime self-registration by tools themselves.

        Raises:
            ValueError: If the tool name conflicts with an existing tool.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Generated tool '{tool.name}' conflicts with existing tool."
            )
        self._tools[tool.name] = tool
        self._generated_tools[tool.name] = version
        logger.info("Registered generated tool: %s v%s", tool.name, version)

    def load_approved_generated_tools(self) -> None:
        """
        Scan tools/generated/approved.json for approved tool modules.

        For each entry in the manifest, dynamically import the module,
        find the BaseTool subclass, instantiate, and register.
        """
        from config import GENERATED_TOOLS_DIR

        manifest_path = GENERATED_TOOLS_DIR / "approved.json"
        if not manifest_path.exists():
            return

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read approved.json: %s", exc)
            return

        if not isinstance(manifest, list):
            logger.warning("approved.json must be a JSON array.")
            return

        for entry in manifest:
            tool_name = entry.get("tool_name", "") if isinstance(entry, dict) else ""
            version = entry.get("version", "0.1.0") if isinstance(entry, dict) else "0.1.0"

            if not tool_name:
                continue

            tool_path = GENERATED_TOOLS_DIR / f"{tool_name}.py"
            if not tool_path.exists():
                logger.warning("Approved tool file missing: %s", tool_path)
                continue

            try:
                tool_class = _load_tool_class(tool_path, tool_name)
                if tool_class is not None:
                    self.register_generated_tool(tool_class(), version=version)
            except Exception as exc:
                logger.warning("Failed to load generated tool '%s': %s", tool_name, exc)

    @property
    def generated_tool_names(self) -> List[str]:
        """Return names of all registered generated tools."""
        return list(self._generated_tools.keys())


def _load_tool_class(tool_path: Path, tool_name: str):
    """
    Dynamically import a tool module and return the BaseTool subclass.

    Returns None if no subclass is found.
    """
    spec = importlib.util.spec_from_file_location(
        f"generated_{tool_name}", str(tool_path)
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, BaseTool)
            and attr is not BaseTool
        ):
            return attr

    return None


def build_default_registry() -> ToolRegistry:
    """Create and populate the registry with all available tools."""
    from tools.mock_tools import MockPriceSignal, MockRandomContext
    from tools.snapshot_volatility_tool import SnapshotVolatilityTool
    from tools.spread_compression_tool import SpreadCompressionTool
    from tools.price_jump_detector_tool import PriceJumpDetectorTool
    from tools.liquidity_spike_tool import LiquiditySpikeTool

    registry = ToolRegistry()
    registry.register(MockPriceSignal())
    registry.register(MockRandomContext())
    registry.register(SnapshotVolatilityTool())
    registry.register(SpreadCompressionTool())
    registry.register(PriceJumpDetectorTool())
    registry.register(LiquiditySpikeTool())

    # Load approved generated tools (if any)
    registry.load_approved_generated_tools()

    return registry
