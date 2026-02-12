"""
Deterministic scorer.

Computes a weighted sum dot product from tool outputs and weights.
No LLM involvement. Pure arithmetic.
"""

from __future__ import annotations

import logging
from typing import List

from prediction_agent.schemas import FormulaSpec, ScoreResult, ToolOutput

logger = logging.getLogger(__name__)


def compute_score(
    tool_outputs: List[ToolOutput],
    formula: FormulaSpec,
) -> ScoreResult:
    """
    Weighted sum: score = sum(weight_i * mean(output_vector_i))

    Each tool produces a vector; we take its mean as the scalar signal,
    then dot-product with the weight.

    Args:
        tool_outputs: Results from tool_runner.
        formula: Agent's selections (carries weights + threshold).

    Returns:
        ScoreResult with final_score and bet_triggered flag.
    """
    if len(tool_outputs) != len(formula.selections):
        raise ValueError(
            f"Mismatch: {len(tool_outputs)} outputs vs "
            f"{len(formula.selections)} selections."
        )

    weights: List[float] = []
    signals: List[float] = []

    for output, selection in zip(tool_outputs, formula.selections):
        w = selection.weight
        # Scalar signal = mean of the output vector
        vec = output.output_vector
        signal = sum(vec) / len(vec) if vec else 0.0

        weights.append(w)
        signals.append(signal)

    # Weighted sum dot product
    final_score = sum(w * s for w, s in zip(weights, signals))
    final_score = round(final_score, 6)

    bet_triggered = final_score >= formula.threshold

    logger.info(
        "Score: %.4f | Threshold: %.4f | Triggered: %s",
        final_score,
        formula.threshold,
        bet_triggered,
    )

    return ScoreResult(
        final_score=final_score,
        tool_outputs=tool_outputs,
        weights=weights,
        threshold=formula.threshold,
        bet_triggered=bet_triggered,
    )
