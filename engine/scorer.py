"""
Deterministic scorer.

Supports two scoring modes (controlled by SCORING_MODE in config.py):

  signal_sum (default, backward-compatible):
      score = sum(weight_i * mean(output_vector_i))
      bet_triggered when score >= threshold

  probability_edge:
      z = weighted sum of probability-signal tools' output_vector[0]
      model_probability = sigmoid(z)   [if USE_LOGISTIC_CALIBRATION=True]
      edge = model_probability - current_market_price
      bet_triggered when abs(edge) >= threshold

Logistic calibration layer (sigmoid) is a fixed mathematical transform applied
post-aggregation. No LLM involvement. Pure arithmetic.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from schemas import FormulaSpec, ScoreResult, ToolOutput

logger = logging.getLogger(__name__)


# ── Logistic Calibration ───────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """
    Logistic sigmoid function: σ(x) = 1 / (1 + exp(-x))

    Maps (-∞, +∞) → (0, 1). Used for probability calibration in probability_edge mode.

    Deterministic, no external libraries, numerically stable for typical ranges.
    For extreme values:
      x >= 20  → ≈ 1.0
      x <= -20 → ≈ 0.0
    """
    # Numerical stability: clamp to avoid overflow in exp()
    if x >= 20.0:
        return 1.0
    if x <= -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def compute_score(
    tool_outputs: List[ToolOutput],
    formula: FormulaSpec,
    scoring_mode: Optional[str] = None,
    current_market_price: Optional[float] = None,
) -> ScoreResult:
    """
    Compute a final score from tool outputs and formula weights.

    Args:
        tool_outputs:         Results from tool_runner (ordered, matches formula.selections).
        formula:              Agent's selections (carries weights + threshold).
        scoring_mode:         Override config SCORING_MODE. None = use config default.
        current_market_price: Required for 'probability_edge' mode.

    Returns:
        ScoreResult with final_score, edge (if applicable), and bet_triggered flag.
    """
    if len(tool_outputs) != len(formula.selections):
        raise ValueError(
            f"Mismatch: {len(tool_outputs)} outputs vs "
            f"{len(formula.selections)} selections."
        )

    # Resolve scoring mode
    if scoring_mode is None:
        try:
            from config import SCORING_MODE
            scoring_mode = SCORING_MODE
        except ImportError:
            scoring_mode = "signal_sum"

    if scoring_mode == "probability_edge":
        return _compute_probability_edge(
            tool_outputs, formula, current_market_price
        )
    else:
        return _compute_signal_sum(tool_outputs, formula)


# ── Mode: signal_sum ───────────────────────────────────────────────────────────

def _compute_signal_sum(
    tool_outputs: List[ToolOutput],
    formula: FormulaSpec,
) -> ScoreResult:
    """
    Original weighted-sum mode (backward compatible).
    score = sum(weight_i * mean(output_vector_i))
    """
    weights: List[float] = []
    signals: List[float] = []

    for output, selection in zip(tool_outputs, formula.selections):
        vec = output.output_vector
        signal = sum(vec) / len(vec) if vec else 0.0
        weights.append(selection.weight)
        signals.append(signal)

    final_score = round(sum(w * s for w, s in zip(weights, signals)), 6)
    bet_triggered = final_score >= formula.threshold

    logger.info(
        "Score [signal_sum]: %.4f | Threshold: %.4f | Triggered: %s",
        final_score, formula.threshold, bet_triggered,
    )

    return ScoreResult(
        final_score=final_score,
        tool_outputs=tool_outputs,
        weights=weights,
        threshold=formula.threshold,
        bet_triggered=bet_triggered,
    )


# ── Mode: probability_edge ────────────────────────────────────────────────────

def _compute_probability_edge(
    tool_outputs: List[ToolOutput],
    formula: FormulaSpec,
    current_market_price: Optional[float],
) -> ScoreResult:
    """
    Probability reconciliation mode with optional logistic calibration.

    Flow:
      1. Compute z = weighted sum of signals (attribution preserved at z-level)
      2. Apply logistic calibration: p_model = sigmoid(z)  [if USE_LOGISTIC_CALIBRATION=True]
      3. Compute edge = p_model - current_market_price
      4. Trigger when abs(edge) >= threshold

    Logistic calibration is a deterministic, fixed mathematical transform.
    No LLM numeric computation. Attribution remains linear in z-space.
    """
    try:
        from config import PROBABILITY_SIGNAL_TOOLS
    except ImportError:
        PROBABILITY_SIGNAL_TOOLS = [
            "sportsbook_implied_probability_tool",
            "weather_probability_tool",
        ]

    try:
        from config import USE_LOGISTIC_CALIBRATION
    except ImportError:
        USE_LOGISTIC_CALIBRATION = True  # Default to enabled

    if current_market_price is None:
        logger.warning(
            "probability_edge mode requires current_market_price; falling back to signal_sum"
        )
        return _compute_signal_sum(tool_outputs, formula)

    weights: List[float] = []
    signals: List[float] = []

    for output, selection in zip(tool_outputs, formula.selections):
        vec = output.output_vector
        w = selection.weight
        weights.append(w)

        if selection.tool_name in PROBABILITY_SIGNAL_TOOLS:
            # Use output_vector[0] as direct probability signal
            prob_signal = vec[0] if vec else 0.0
            signals.append(prob_signal)
        else:
            # Non-probability tools contribute their mean as a signal
            signal = sum(vec) / len(vec) if vec else 0.0
            signals.append(signal)

    # Step 1: Compute raw weighted sum (z) — this is the attribution level
    z = sum(w * s for w, s in zip(weights, signals))

    # Step 2: Apply logistic calibration if enabled
    if USE_LOGISTIC_CALIBRATION:
        model_probability = _sigmoid(z)
        logger.debug("Logistic calibration: z=%.6f → p_model=%.6f", z, model_probability)
    else:
        # Legacy behavior: treat z as direct probability (clamp to [0,1])
        model_probability = max(0.0, min(1.0, z))
        logger.debug("No calibration: z=%.6f → p_model=%.6f (clamped)", z, model_probability)

    # Step 3: Compute edge
    edge = round(model_probability - current_market_price, 6)
    final_score = round(abs(edge), 6)  # final_score = |edge| for threshold comparison
    bet_triggered = final_score >= formula.threshold

    logger.info(
        "Score [probability_edge]: z=%.4f → p_model=%.4f | market=%.4f edge=%.4f "
        "Threshold: %.4f | Triggered: %s",
        z, model_probability, current_market_price, edge,
        formula.threshold, bet_triggered,
    )

    return ScoreResult(
        final_score=final_score,
        tool_outputs=tool_outputs,
        weights=weights,
        threshold=formula.threshold,
        bet_triggered=bet_triggered,
        scoring_mode="probability_edge",
        model_probability=round(model_probability, 6),
        edge=edge,
        raw_score_z=round(z, 6),  # NEW: log the pre-calibration score
    )
