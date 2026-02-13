"""
Tool gap analyzer -- statistical analysis of execution logs.

No LLM calls. Pure statistics over execution_logs.jsonl to detect
repeated reasoning patterns, low-confidence runs, high token usage,
and implicit calculations that a tool could handle.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import EVOLUTION_GAP_THRESHOLD, EXECUTION_LOG_FILE
from prediction_agent.evolution.schemas import GapReport

logger = logging.getLogger(__name__)

# Keywords that suggest implicit calculations in rationale text
_IMPLICIT_CALC_KEYWORDS = [
    "probability",
    "convert",
    "implied odds",
    "calculate",
    "compute",
    "multiply",
    "divide",
    "percentage",
    "ratio",
    "normalize",
    "weighted average",
    "expected value",
]

# Minimum runs required before analysis is meaningful
_MIN_RUNS = 5


def analyze_gaps(
    log_path: Path | None = None,
    min_runs: int = _MIN_RUNS,
    gap_threshold: float = EVOLUTION_GAP_THRESHOLD,
) -> Optional[GapReport]:
    """
    Analyze execution logs and identify the highest-priority capability gap.

    Args:
        log_path: Path to execution_logs.jsonl. Defaults to config value.
        min_runs: Minimum number of logged runs before analysis starts.
        gap_threshold: Minimum priority_score to return a report.

    Returns:
        The highest-priority GapReport, or None if no gap exceeds threshold.
    """
    target = log_path or EXECUTION_LOG_FILE
    entries = _load_entries(target)

    if len(entries) < min_runs:
        logger.info(
            "Gap analyzer: only %d runs logged (need %d). Skipping.",
            len(entries),
            min_runs,
        )
        return None

    candidates: List[GapReport] = []

    # Pass 1: Low-confidence runs (score close to threshold)
    low_conf = _detect_low_confidence(entries)
    if low_conf is not None:
        candidates.append(low_conf)

    # Pass 2: High token usage runs
    high_tokens = _detect_high_token_usage(entries)
    if high_tokens is not None:
        candidates.append(high_tokens)

    # Pass 3: Repeated reasoning patterns
    repeated = _detect_repeated_reasoning(entries)
    if repeated is not None:
        candidates.append(repeated)

    # Pass 4: Implicit calculations in rationale
    implicit = _detect_implicit_calculations(entries)
    if implicit is not None:
        candidates.append(implicit)

    if not candidates:
        logger.info("Gap analyzer: no gaps detected.")
        return None

    # Return highest priority gap that exceeds threshold
    candidates.sort(key=lambda g: g.priority_score, reverse=True)
    best = candidates[0]

    if best.priority_score < gap_threshold:
        logger.info(
            "Gap analyzer: best gap score %.3f below threshold %.3f.",
            best.priority_score,
            gap_threshold,
        )
        return None

    logger.info(
        "Gap analyzer: detected gap '%s' with priority %.3f",
        best.problem_detected,
        best.priority_score,
    )
    return best


# ------------------------------------------------------------------
# Analysis passes
# ------------------------------------------------------------------

def _detect_low_confidence(entries: List[Dict[str, Any]]) -> Optional[GapReport]:
    """Find runs where final_score is within 0.05 of threshold."""
    close_calls = 0
    total = len(entries)

    for e in entries:
        score = e.get("final_score", 0.0)
        threshold = e.get("threshold", 0.5)
        if abs(score - threshold) <= 0.05:
            close_calls += 1

    ratio = close_calls / total if total > 0 else 0.0

    if ratio < 0.3:
        return None

    return GapReport(
        problem_detected="High frequency of borderline predictions (score near threshold)",
        evidence={
            "close_call_count": close_calls,
            "total_runs": total,
            "ratio": round(ratio, 4),
        },
        estimated_token_waste=0.0,
        priority_score=min(1.0, ratio * 1.2),
    )


def _detect_high_token_usage(entries: List[Dict[str, Any]]) -> Optional[GapReport]:
    """Detect runs above the 90th percentile of token usage."""
    token_counts = [e.get("total_tokens_used", 0) for e in entries]
    token_counts = [t for t in token_counts if t > 0]

    if len(token_counts) < 3:
        return None

    sorted_tokens = sorted(token_counts)
    p90_idx = int(len(sorted_tokens) * 0.9)
    p90 = sorted_tokens[min(p90_idx, len(sorted_tokens) - 1)]
    mean_tokens = sum(token_counts) / len(token_counts)

    high_count = sum(1 for t in token_counts if t > p90)
    waste = sum(t - mean_tokens for t in token_counts if t > p90)

    if high_count < 2:
        return None

    return GapReport(
        problem_detected="Repeated high token usage runs detected",
        evidence={
            "p90_tokens": p90,
            "mean_tokens": round(mean_tokens, 1),
            "high_usage_count": high_count,
        },
        estimated_token_waste=round(waste, 1),
        priority_score=min(1.0, 0.4 + (waste / max(mean_tokens * len(token_counts), 1)) * 0.6),
    )


def _detect_repeated_reasoning(entries: List[Dict[str, Any]]) -> Optional[GapReport]:
    """Find repeated n-gram patterns across rationale strings."""
    all_ngrams: Counter = Counter()
    n = 3  # trigrams

    for e in entries:
        rationale = e.get("reasoning_segments", "")
        if not rationale:
            continue
        words = re.findall(r"\w+", rationale.lower())
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i : i + n])
            all_ngrams[gram] += 1

    # Find trigrams that appear in more than 50% of runs
    threshold_count = len(entries) * 0.5
    repeated = {gram: count for gram, count in all_ngrams.items() if count >= threshold_count}

    if len(repeated) < 2:
        return None

    top_repeated = dict(sorted(repeated.items(), key=lambda x: x[1], reverse=True)[:5])

    return GapReport(
        problem_detected="Repeated reasoning patterns across runs suggest missing tooling",
        evidence={
            "top_repeated_trigrams": top_repeated,
            "unique_repeated_count": len(repeated),
        },
        estimated_token_waste=float(sum(repeated.values()) * 3),  # rough estimate
        priority_score=min(1.0, len(repeated) / 10.0),
    )


def _detect_implicit_calculations(entries: List[Dict[str, Any]]) -> Optional[GapReport]:
    """Detect keywords in rationale suggesting manual computation."""
    keyword_hits: Counter = Counter()
    runs_with_hits = 0

    for e in entries:
        rationale = e.get("reasoning_segments", "").lower()
        if not rationale:
            continue
        found_any = False
        for kw in _IMPLICIT_CALC_KEYWORDS:
            count = rationale.count(kw)
            if count > 0:
                keyword_hits[kw] += count
                found_any = True
        if found_any:
            runs_with_hits += 1

    ratio = runs_with_hits / len(entries) if entries else 0.0

    if ratio < 0.4:
        return None

    return GapReport(
        problem_detected="Agent frequently performs implicit calculations that a tool could handle",
        evidence={
            "keyword_hits": dict(keyword_hits.most_common(5)),
            "runs_with_calculations": runs_with_hits,
            "total_runs": len(entries),
        },
        estimated_token_waste=float(sum(keyword_hits.values()) * 5),
        priority_score=min(1.0, ratio),
    )


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def _load_entries(path: Path) -> List[Dict[str, Any]]:
    """Load all entries from execution_logs.jsonl."""
    if not path.exists():
        return []

    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return entries
