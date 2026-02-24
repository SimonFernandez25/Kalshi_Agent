"""
Correlation pruner — prevents redundant tool generation.

Before a newly verified tool is registered, this module runs it on the last N
historical snapshots and computes the Pearson correlation between its output
vectors and those of every existing tool.

If the new tool's outputs are correlated at > CORRELATION_REJECTION_THRESHOLD
with ANY existing tool, the tool is REJECTED (not registered) and the rejection
reason is logged.

This prevents the evolution pipeline from generating a cascade of tools that all
measure the same market signal with different names.

Design constraints:
  - Pure Python + stdlib only (no numpy/scipy required).
  - Deterministic — always produces the same decision for the same inputs.
  - Does not modify any production files or registry state.
  - Returns a structured PruningResult dataclass, not side effects.

Usage (called by tool_lifecycle_manager or directly by verify_tool):
    from prediction_agent.evolution.correlation_pruner import check_correlation
    result = check_correlation(new_tool_path, registry, snapshot_file, n_samples=50)
    if not result.approved:
        # reject tool
        print(result.rejection_reason)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
CORRELATION_REJECTION_THRESHOLD = 0.95   # Reject if |r| >= this with any existing tool
CORRELATION_MIN_SAMPLES         = 5      # Need at least this many valid samples
CORRELATION_DEFAULT_N_SAMPLES   = 50     # How many historical snapshots to use


@dataclass
class CorrelationEntry:
    """Correlation result between the new tool and one existing tool."""
    existing_tool_name: str
    pearson_r: float
    n_samples: int
    exceeds_threshold: bool


@dataclass
class PruningResult:
    """
    Outcome of a correlation pruning check.

    Attributes:
        approved:           True if the tool passed the pruning check.
        new_tool_name:      Name of the tool being evaluated.
        correlations:       Per-existing-tool correlation values.
        rejection_reason:   Human-readable reason if not approved.
        n_snapshots_used:   How many snapshot rows were evaluated.
        checked_at:         UTC timestamp of the check.
    """
    approved: bool
    new_tool_name: str
    correlations: List[CorrelationEntry] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    n_snapshots_used: int = 0
    checked_at: str = ""


# ── Public API ─────────────────────────────────────────────────────────────────

def check_correlation(
    new_tool_path: Path,
    registry: "ToolRegistry",
    snapshot_file: Optional[Path] = None,
    n_samples: int = CORRELATION_DEFAULT_N_SAMPLES,
    threshold: float = CORRELATION_REJECTION_THRESHOLD,
) -> PruningResult:
    """
    Check whether a new tool's output is redundant with any existing tool.

    Args:
        new_tool_path:  Path to the new tool's .py file.
        registry:       The current ToolRegistry (for existing tool instances).
        snapshot_file:  Path to market_snapshots.jsonl.
                        Defaults to outputs/market_snapshots.jsonl.
        n_samples:      How many snapshots to evaluate on.
        threshold:      Pearson |r| threshold above which tool is rejected.

    Returns:
        PruningResult with approved=True if tool is unique enough.
    """
    from config import OUTPUTS_DIR
    from schemas import EventInput

    if snapshot_file is None:
        snapshot_file = OUTPUTS_DIR / "market_snapshots.jsonl"

    now_str = datetime.now(timezone.utc).isoformat()
    tool_stem = new_tool_path.stem

    # Load the new tool from file using the sandbox runner for safety
    # We call it via subprocess to stay consistent with verifier policy
    from prediction_agent.evolution.sandbox_runner import run_tool_in_sandbox

    # Load a sample of EventInputs from snapshots
    events = _load_sample_events(snapshot_file, n_samples)

    if len(events) < CORRELATION_MIN_SAMPLES:
        logger.info(
            "Correlation check skipped — only %d snapshots (need %d). Tool approved by default.",
            len(events), CORRELATION_MIN_SAMPLES
        )
        return PruningResult(
            approved=True,
            new_tool_name=tool_stem,
            rejection_reason=None,
            n_snapshots_used=len(events),
            checked_at=now_str,
        )

    # Collect new tool's outputs across all sample events
    new_tool_outputs: List[float] = _run_tool_on_events(new_tool_path, events)

    if len(new_tool_outputs) < CORRELATION_MIN_SAMPLES:
        logger.info(
            "Correlation check: new tool '%s' produced fewer than %d valid outputs. "
            "Approving by default (insufficient data).",
            tool_stem, CORRELATION_MIN_SAMPLES
        )
        return PruningResult(
            approved=True,
            new_tool_name=tool_stem,
            n_snapshots_used=len(new_tool_outputs),
            checked_at=now_str,
        )

    # Collect existing tools' outputs
    correlations: List[CorrelationEntry] = []

    for existing_name in registry.tool_names:
        try:
            existing_tool = registry.get(existing_name)
        except Exception:
            continue

        existing_outputs: List[float] = []
        for event in events[:len(new_tool_outputs)]:
            try:
                out = existing_tool.run(event)
                vec = out.output_vector
                mean_val = sum(vec) / len(vec) if vec else 0.0
                existing_outputs.append(mean_val)
            except Exception:
                existing_outputs.append(0.0)

        if len(existing_outputs) < CORRELATION_MIN_SAMPLES:
            continue

        r = _pearson_r(new_tool_outputs[:len(existing_outputs)], existing_outputs)
        exceeds = abs(r) >= threshold

        entry = CorrelationEntry(
            existing_tool_name=existing_name,
            pearson_r=round(r, 4),
            n_samples=len(existing_outputs),
            exceeds_threshold=exceeds,
        )
        correlations.append(entry)

        if exceeds:
            reason = (
                f"New tool '{tool_stem}' has Pearson r={r:.4f} "
                f"with existing tool '{existing_name}' "
                f"(threshold={threshold}). Tool rejected as redundant."
            )
            logger.warning(reason)
            return PruningResult(
                approved=False,
                new_tool_name=tool_stem,
                correlations=correlations,
                rejection_reason=reason,
                n_snapshots_used=len(new_tool_outputs),
                checked_at=now_str,
            )

    logger.info(
        "Correlation check passed for '%s' — max |r|=%.4f across %d existing tools.",
        tool_stem,
        max((abs(c.pearson_r) for c in correlations), default=0.0),
        len(correlations),
    )

    return PruningResult(
        approved=True,
        new_tool_name=tool_stem,
        correlations=correlations,
        n_snapshots_used=len(new_tool_outputs),
        checked_at=now_str,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_sample_events(snapshot_file: Path, n: int) -> List["EventInput"]:
    """Load up to n EventInputs from the snapshot file (most recent first)."""
    from schemas import EventInput

    if not snapshot_file.exists():
        return []

    rows = []
    with open(snapshot_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Take last n rows (most recent)
    rows = rows[-n:]

    events = []
    for row in rows:
        try:
            ts = row.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif ts is None:
                ts = datetime.now(timezone.utc)
            market_id = row.get("market_id", "")
            if not market_id:
                continue
            price = float(row.get("last_price", row.get("current_price", 0.0)))
            events.append(EventInput(
                event_id=row.get("event_id", market_id),
                market_id=market_id,
                market_title=row.get("title", market_id),
                current_price=price,
                timestamp=ts,
            ))
        except Exception:
            continue

    return events


def _run_tool_on_events(tool_path: Path, events: List["EventInput"]) -> List[float]:
    """
    Run a tool (via subprocess sandbox) on a list of events.
    Returns a list of scalar means (one per event). Failures produce no entry.
    """
    from prediction_agent.evolution.sandbox_runner import run_tool_in_sandbox

    outputs: List[float] = []
    for event in events:
        event_dict = event.model_dump(mode="json")
        result = run_tool_in_sandbox(
            tool_path=tool_path,
            event_input_dict=event_dict,
        )
        if result.success and result.output_vector:
            mean_val = sum(result.output_vector) / len(result.output_vector)
            outputs.append(mean_val)
        # Skip on failure — don't pad with zeros (would distort correlation)

    return outputs


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """
    Compute Pearson correlation coefficient between two equal-length lists.
    Returns 0.0 if standard deviation is zero (constant signal).
    """
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0

    xs = xs[:n]
    ys = ys[:n]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov   = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if std_x == 0.0 or std_y == 0.0:
        return 0.0  # Constant output — no meaningful correlation

    return cov / (std_x * std_y)
