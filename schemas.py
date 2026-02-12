"""
Strict schemas for the prediction agent framework.
All data flowing through the system must conform to these Pydantic models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
# Input Schema  (comes ONLY from Kalshi)
# ──────────────────────────────────────────────

class EventInput(BaseModel):
    """Single basketball market pulled from Kalshi."""
    event_id: str = Field(..., description="Kalshi event identifier")
    market_id: str = Field(..., description="Kalshi market identifier")
    market_title: str = Field(..., description="Human-readable market title")
    current_price: float = Field(..., ge=0.0, le=1.0, description="Current YES price (0-1)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when price was captured",
    )

    @field_validator("current_price")
    @classmethod
    def clamp_price(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# ──────────────────────────────────────────────
# Tool Output Schema
# ──────────────────────────────────────────────

class ToolOutput(BaseModel):
    """Result vector from a single deterministic tool."""
    tool_name: str
    output_vector: List[float] = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)


# ──────────────────────────────────────────────
# Agent Formula Specification
# ──────────────────────────────────────────────

class AggregationMethod(str, Enum):
    WEIGHTED_SUM = "weighted_sum"


class ToolSelection(BaseModel):
    """One tool + its weight chosen by the agent."""
    tool_name: str
    tool_inputs: dict = Field(default_factory=dict)
    weight: float = Field(..., ge=0.0, le=1.0)


class FormulaSpec(BaseModel):
    """
    The agent's full decision output.
    LLM chooses tools, weights, threshold — but NEVER computes scores.
    """
    selections: List[ToolSelection] = Field(..., min_length=1)
    aggregation: AggregationMethod = AggregationMethod.WEIGHTED_SUM
    threshold: float = Field(..., ge=0.0, le=1.0, description="Bet trigger threshold")
    rationale: str = Field(..., min_length=1, description="Agent reasoning for selections")

    @field_validator("selections")
    @classmethod
    def weights_must_sum_to_one(cls, v: List[ToolSelection]) -> List[ToolSelection]:
        total = sum(s.weight for s in v)
        if abs(total - 1.0) > 0.01:
            # Auto-normalize
            for s in v:
                s.weight = s.weight / total
        return v


# ──────────────────────────────────────────────
# Scoring Output
# ──────────────────────────────────────────────

class ScoreResult(BaseModel):
    """Deterministic engine output."""
    final_score: float
    tool_outputs: List[ToolOutput]
    weights: List[float]
    threshold: float
    bet_triggered: bool


# ──────────────────────────────────────────────
# Watcher State
# ──────────────────────────────────────────────

class WatcherTick(BaseModel):
    """Single poll from the watcher loop."""
    market_id: str
    polled_price: float
    threshold: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    triggered: bool = False


# ──────────────────────────────────────────────
# Paper Bet / Run Log
# ──────────────────────────────────────────────

class PaperBet(BaseModel):
    """Simulated bet logged to JSONL."""
    run_id: str
    event_input: EventInput
    formula_spec: FormulaSpec
    score_result: ScoreResult
    watcher_ticks: List[WatcherTick] = Field(default_factory=list)
    bet_placed: bool = False
    bet_side: Optional[str] = None  # "YES" or "NO"
    bet_amount: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
