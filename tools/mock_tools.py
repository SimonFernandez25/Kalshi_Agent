"""
Mock deterministic tools for testing the tool plumbing.

These are NOT real signals — they exist to prove the framework works.
Replace with real tools once the plumbing is validated.
"""

from __future__ import annotations

import hashlib
import random

from schemas import EventInput, ToolOutput
from tools.base_tool import BaseTool


class MockPriceSignal(BaseTool):
    """
    Returns the current market price as a normalized signal.
    Deterministic: same price in → same vector out.
    """

    @property
    def name(self) -> str:
        return "mock_price_signal"

    @property
    def description(self) -> str:
        return (
            "Returns the current Kalshi YES price as a [0,1] signal vector. "
            "Higher price = market thinks event is more likely."
        )

    def run(self, event: EventInput, **kwargs) -> ToolOutput:
        normalized_price = max(0.0, min(1.0, event.current_price))
        return ToolOutput(
            tool_name=self.name,
            output_vector=[normalized_price],
            metadata={"source": "kalshi_price", "raw_price": event.current_price},
        )


class MockRandomContext(BaseTool):
    """
    Returns a seeded pseudo-random number derived from event_id.
    Deterministic: same event_id → same random output every time.
    """

    @property
    def name(self) -> str:
        return "mock_random_context"

    @property
    def description(self) -> str:
        return (
            "Returns a deterministic pseudo-random [0,1] value seeded by event_id. "
            "Simulates an external context signal (placeholder for real data)."
        )

    def run(self, event: EventInput, **kwargs) -> ToolOutput:
        # Derive a stable seed from event_id
        seed = int(hashlib.sha256(event.event_id.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        value = rng.random()

        return ToolOutput(
            tool_name=self.name,
            output_vector=[round(value, 6)],
            metadata={"seed": seed, "event_id": event.event_id},
        )
