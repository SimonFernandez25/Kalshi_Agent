"""
Paper broker — logs simulated bets to JSONL.

No real execution. No money. Just logging for analysis.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from prediction_agent.config import DEFAULT_BET_AMOUNT, PAPER_BET_LOG, RUN_LOG
from prediction_agent.schemas import (
    EventInput,
    FormulaSpec,
    PaperBet,
    ScoreResult,
    WatcherTick,
)

logger = logging.getLogger(__name__)


def log_paper_bet(
    event: EventInput,
    formula: FormulaSpec,
    score: ScoreResult,
    watcher_ticks: List[WatcherTick],
    bet_amount: float = DEFAULT_BET_AMOUNT,
) -> PaperBet:
    """
    Create and log a paper bet to the JSONL file.

    Args:
        event: The market event.
        formula: The agent's formula spec.
        score: The deterministic score result.
        watcher_ticks: Price poll history.
        bet_amount: Simulated bet size.

    Returns:
        The PaperBet record.
    """
    # Determine if bet should be placed
    any_triggered = any(t.triggered for t in watcher_ticks) or score.bet_triggered
    bet_side = "YES" if score.final_score >= score.threshold else "NO"

    paper_bet = PaperBet(
        run_id=str(uuid.uuid4())[:8],
        event_input=event,
        formula_spec=formula,
        score_result=score,
        watcher_ticks=watcher_ticks,
        bet_placed=any_triggered,
        bet_side=bet_side if any_triggered else None,
        bet_amount=bet_amount if any_triggered else 0.0,
    )

    # Append to JSONL log
    _append_jsonl(PAPER_BET_LOG, paper_bet.model_dump(mode="json"))

    # Also write a simpler run log entry
    run_entry = {
        "run_id": paper_bet.run_id,
        "market_id": event.market_id,
        "market_title": event.market_title,
        "current_price": event.current_price,
        "tools_used": [s.tool_name for s in formula.selections],
        "weights": [s.weight for s in formula.selections],
        "threshold": formula.threshold,
        "final_score": score.final_score,
        "bet_triggered": paper_bet.bet_placed,
        "bet_side": paper_bet.bet_side,
        "bet_amount": paper_bet.bet_amount,
        "watcher_ticks_count": len(watcher_ticks),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(RUN_LOG, run_entry)

    logger.info(
        "Paper bet logged: run=%s market=%s score=%.4f triggered=%s side=%s",
        paper_bet.run_id,
        event.market_id,
        score.final_score,
        paper_bet.bet_placed,
        paper_bet.bet_side,
    )

    return paper_bet


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
