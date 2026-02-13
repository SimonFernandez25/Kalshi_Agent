"""
Pydantic schemas for the evolution pipeline.

Strict typing for every data structure flowing through the
tool gap analysis, spec generation, verification, and lifecycle tracking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ------------------------------------------------------------------
# Execution Log
# ------------------------------------------------------------------

class ExecutionLogEntry(BaseModel):
    """Structured record of a single pipeline run."""

    run_id: str
    market_id: str
    market_title: str = ""
    selected_tools: List[str] = Field(default_factory=list)
    tool_weights: List[float] = Field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    final_score: float = 0.0
    threshold: float = 0.0
    bet_triggered: bool = False
    reasoning_segments: str = ""
    failed_tool_attempts: List[str] = Field(default_factory=list)
    total_tokens_used: int = 0
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    # Live data tracking
    kalshi_response_timestamp: Optional[datetime] = None
    kalshi_response_hash: Optional[str] = None
    
    # Tool execution validation
    tool_execution_statuses: List[Dict[str, Any]] = Field(default_factory=list)



# ------------------------------------------------------------------
# Gap Analysis
# ------------------------------------------------------------------

class GapReport(BaseModel):
    """Output of the tool gap analyzer."""

    problem_detected: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    estimated_token_waste: float = 0.0
    priority_score: float = Field(0.0, ge=0.0, le=1.0)


# ------------------------------------------------------------------
# Tool Specification
# ------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolSpec(BaseModel):
    """LLM-generated specification for a new tool."""

    tool_name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    inputs: Dict[str, str] = Field(default_factory=dict)
    output_type: str = Field(default="float")
    deterministic: bool = True
    data_sources: List[str] = Field(default_factory=list)
    expected_token_reduction: float = Field(0.0, ge=0.0)
    expected_accuracy_gain: float = Field(0.0, ge=0.0)
    risk_level: RiskLevel = RiskLevel.LOW

    @field_validator("deterministic")
    @classmethod
    def must_be_deterministic(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Generated tools must be deterministic.")
        return v

    @field_validator("risk_level")
    @classmethod
    def reject_high_risk(cls, v: RiskLevel) -> RiskLevel:
        if v == RiskLevel.HIGH:
            raise ValueError("High-risk tools are rejected at spec level.")
        return v


# ------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------

class VerificationResult(BaseModel):
    """Output of the tool verifier."""

    tool_name: str
    passed: bool
    checks: Dict[str, bool] = Field(default_factory=dict)
    rejection_reason: Optional[str] = None


# ------------------------------------------------------------------
# Lifecycle Tracking
# ------------------------------------------------------------------

class ToolStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    PENDING = "pending"


class ToolLifecycleRecord(BaseModel):
    """Per-tool performance tracking over time."""

    tool_name: str
    version: str = "0.1.0"
    usage_count: int = 0
    total_score_contribution: float = 0.0
    correct_predictions: int = 0
    total_predictions: int = 0
    total_latency_ms: float = 0.0
    consecutive_underperformance: int = 0
    status: ToolStatus = ToolStatus.ACTIVE
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_used_at: Optional[datetime] = None

    @property
    def avg_score_contribution(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.total_score_contribution / self.usage_count

    @property
    def correct_prediction_rate(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    @property
    def avg_latency_ms(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.total_latency_ms / self.usage_count
