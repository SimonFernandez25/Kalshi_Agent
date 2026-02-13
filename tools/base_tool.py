"""
Abstract base class for all deterministic tools.

Every tool must:
  - Accept an EventInput
  - Return a ToolOutput with a numeric output_vector
  - Be fully deterministic (same input → same output)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from schemas import EventInput, ToolOutput


class BaseTool(ABC):
    """Interface that every prediction tool must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier (must match registry key)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description shown to the agent."""
        ...

    @abstractmethod
    def run(self, event: EventInput, **kwargs) -> ToolOutput:
        """
        Execute the tool deterministically.

        Args:
            event: The market event to analyze.
            **kwargs: Extra params from tool_inputs.

        Returns:
            ToolOutput with output_vector of floats.
        """
        ...

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"
