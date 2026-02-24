"""
Deterministic backtesting engine.

Replays market_snapshots.jsonl chronologically, applies the deterministic
scoring engine (NOT the LLM), computes simulated bets, and measures
prediction quality.

Design constraints:
  - No live API calls — reads only from local JSONL snapshots.
  - No LLM involvement — uses a fixed FormulaSpec or the default registry
    equal-weight formula so results are reproducible.
  - Deterministic and reproducible from the same snapshot file.
  - Does NOT modify any output files in outputs/ — results are returned
    as a BacktestReport dataclass and optionally written to a separate
    backtest_results.jsonl.

Usage:
    python -m prediction_agent.backtest.backtest
    python -m prediction_agent.backtest.backtest --snapshot-file outputs/market_snapshots.jsonl
    python -m prediction_agent.backtest.backtest --formula-preset equal_weight --market-filter KXNBA
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── BacktestReport dataclass ───────────────────────────────────────────────────

@dataclass
class ToolContribution:
    """Per-tool summary stats from a backtest run."""
    tool_name: str
    times_selected: int = 0
    avg_output_mean: float = 0.0
    total_score_contribution: float = 0.0


@dataclass
class BacktestReport:
    """
    Full report from one backtest run.

    Attributes:
        total_events:       Total snapshot rows processed.
        bets_triggered:     Number of events where score >= threshold.
        accuracy:           Fraction of triggered bets that were "correct"
                            (price moved toward YES after bet).
        brier_score:        Mean Brier score (lower = better) measuring
                            calibration of the final_score as a probability.
        avg_score_delta:    Mean |final_score - current_price| across all events.
        bet_precision:      Of all triggered bets, fraction where outcome == YES.
        bet_recall:         Of all YES outcomes, fraction where bet was triggered.
        profit_simulation:  Simulated P&L in paper dollars.
        tool_contributions: Per-tool performance breakdown.
        formula_preset:     Which formula preset was used.
        snapshot_file:      Path to snapshot file used.
        market_filter:      Optional market ticker filter applied.
        run_timestamp:      When this backtest was run.
        total_events_skipped: Rows skipped due to missing/malformed data.
    """
    total_events: int = 0
    bets_triggered: int = 0
    accuracy: float = 0.0
    brier_score: float = 0.0
    avg_score_delta: float = 0.0
    bet_precision: float = 0.0
    bet_recall: float = 0.0
    profit_simulation: float = 0.0
    tool_contributions: List[ToolContribution] = field(default_factory=list)
    formula_preset: str = "equal_weight"
    snapshot_file: str = ""
    market_filter: Optional[str] = None
    run_timestamp: str = ""
    total_events_skipped: int = 0


# ── Backtest engine ────────────────────────────────────────────────────────────

def run_backtest(
    snapshot_file: Optional[Path] = None,
    formula_preset: str = "equal_weight",
    market_filter: Optional[str] = None,
    threshold: float = 0.55,
    bet_amount: float = 10.0,
    output_file: Optional[Path] = None,
    verbose: bool = False,
) -> BacktestReport:
    """
    Execute a full deterministic backtest.

    Args:
        snapshot_file:   Path to market_snapshots.jsonl.
                         Defaults to outputs/market_snapshots.jsonl.
        formula_preset:  "equal_weight" (all tools, equal weights) or
                         "volatility_only" (snapshot_volatility_tool only).
        market_filter:   Optional substring to filter market_id (e.g. "KXNBA").
        threshold:       Score threshold for triggering a simulated bet.
        bet_amount:      Paper dollar amount per bet.
        output_file:     Optional path to write BacktestReport as JSONL line.
        verbose:         Print per-event details.

    Returns:
        BacktestReport
    """
    from config import OUTPUTS_DIR
    from schemas import EventInput, FormulaSpec, ToolSelection, AggregationMethod
    from engine.tool_runner import run_tools
    from engine.scorer import compute_score
    from tools.registry import build_default_registry

    # Resolve snapshot file
    if snapshot_file is None:
        snapshot_file = OUTPUTS_DIR / "market_snapshots.jsonl"

    if not snapshot_file.exists():
        logger.warning(
            "Snapshot file not found: %s — returning empty BacktestReport", snapshot_file
        )
        return BacktestReport(
            snapshot_file=str(snapshot_file),
            run_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # Build registry (uses same default registry as production — deterministic)
    registry = build_default_registry()
    available_tools = registry.tool_names

    # Build FormulaSpec from preset
    formula = _build_formula(formula_preset, available_tools, threshold)

    # Load and sort snapshots chronologically
    rows = _load_snapshots(snapshot_file, market_filter)
    if not rows:
        logger.info("No rows matched filter. Returning empty report.")
        return BacktestReport(
            snapshot_file=str(snapshot_file),
            market_filter=market_filter,
            formula_preset=formula_preset,
            run_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    rows.sort(key=lambda r: r.get("timestamp", ""))

    # Per-event state
    brier_sum = 0.0
    score_delta_sum = 0.0
    bets_triggered = 0
    correct_bets = 0
    total_yes_outcomes = 0
    correct_triggered_yes = 0
    profit = 0.0
    skipped = 0
    tool_contrib_map: Dict[str, ToolContribution] = {
        s.tool_name: ToolContribution(tool_name=s.tool_name)
        for s in formula.selections
    }

    for i, row in enumerate(rows):
        event = _row_to_event(row)
        if event is None:
            skipped += 1
            continue

        # Run tools deterministically
        try:
            tool_outputs, _ = run_tools(event, formula, registry)
            score_result = compute_score(tool_outputs, formula)
        except Exception as exc:
            logger.debug("Backtest row %d failed engine: %s", i, exc)
            skipped += 1
            continue

        final_score = score_result.final_score
        triggered    = score_result.bet_triggered
        current_price = event.current_price

        # Outcome: treat current_price as the "ground truth" probability.
        # A bet is "correct" if: triggered=True AND price >= 0.5 (market favors YES)
        # or triggered=False AND price < 0.5.
        # NOTE: In real backtesting you'd compare against a future snapshot;
        # here we use current_price as a proxy since realized outcomes are not
        # in the snapshot data.
        outcome_yes = current_price >= 0.5
        if outcome_yes:
            total_yes_outcomes += 1

        # Brier score: (final_score - outcome)^2
        outcome_float = 1.0 if outcome_yes else 0.0
        brier_sum += (final_score - outcome_float) ** 2
        score_delta_sum += abs(final_score - current_price)

        if triggered:
            bets_triggered += 1
            # Profit: if bet YES and outcome YES → +bet_amount * (1/price - 1)
            # if bet YES and outcome NO → -bet_amount
            if outcome_yes:
                correct_bets += 1
                correct_triggered_yes += 1
                if current_price > 0.0:
                    profit += bet_amount * ((1.0 / current_price) - 1.0)
            else:
                profit -= bet_amount

        # Tool contribution tracking
        for tool_out, sel in zip(tool_outputs, formula.selections):
            tc = tool_contrib_map[sel.tool_name]
            tc.times_selected += 1
            vec_mean = sum(tool_out.output_vector) / len(tool_out.output_vector) if tool_out.output_vector else 0.0
            tc.avg_output_mean = (
                (tc.avg_output_mean * (tc.times_selected - 1) + vec_mean)
                / tc.times_selected
            )
            tc.total_score_contribution += sel.weight * vec_mean

        if verbose:
            print(
                f"[{i:4d}] market={event.market_id[:20]:<20s} "
                f"price={current_price:.3f} score={final_score:.4f} "
                f"bet={'YES' if triggered else ' no'} "
                f"outcome={'YES' if outcome_yes else ' NO'}"
            )

    total_processed = len(rows) - skipped
    accuracy      = (correct_bets / bets_triggered)       if bets_triggered > 0         else 0.0
    brier_score   = (brier_sum / total_processed)          if total_processed > 0        else 0.0
    avg_delta     = (score_delta_sum / total_processed)    if total_processed > 0        else 0.0
    bet_precision = (correct_bets / bets_triggered)        if bets_triggered > 0         else 0.0
    bet_recall    = (correct_triggered_yes / total_yes_outcomes) if total_yes_outcomes > 0 else 0.0

    report = BacktestReport(
        total_events=total_processed,
        bets_triggered=bets_triggered,
        accuracy=round(accuracy, 4),
        brier_score=round(brier_score, 6),
        avg_score_delta=round(avg_delta, 4),
        bet_precision=round(bet_precision, 4),
        bet_recall=round(bet_recall, 4),
        profit_simulation=round(profit, 2),
        tool_contributions=list(tool_contrib_map.values()),
        formula_preset=formula_preset,
        snapshot_file=str(snapshot_file),
        market_filter=market_filter,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        total_events_skipped=skipped,
    )

    logger.info(
        "Backtest complete: %d events, %d bets, accuracy=%.3f, "
        "brier=%.4f, profit=$%.2f",
        report.total_events, report.bets_triggered,
        report.accuracy, report.brier_score, report.profit_simulation,
    )

    # Optionally persist
    if output_file is not None:
        _write_report(report, output_file)

    return report


# ── Formula presets ────────────────────────────────────────────────────────────

def _build_formula(
    preset: str,
    available_tools: List[str],
    threshold: float,
) -> "FormulaSpec":
    """
    Build a deterministic FormulaSpec for the backtest.
    Never calls the LLM — uses fixed, reproducible weights.
    """
    from schemas import FormulaSpec, ToolSelection, AggregationMethod

    if preset == "volatility_only":
        tools_to_use = [t for t in available_tools if "volatility" in t]
        if not tools_to_use:
            tools_to_use = available_tools[:1]
    elif preset == "real_tools_only":
        # Exclude mock tools
        tools_to_use = [t for t in available_tools if not t.startswith("mock_")]
        if not tools_to_use:
            tools_to_use = available_tools
    else:
        # equal_weight — all tools, uniform weights
        tools_to_use = list(available_tools)

    n = len(tools_to_use)
    equal_w = round(1.0 / n, 6)

    selections = [
        ToolSelection(tool_name=t, weight=equal_w, tool_inputs={})
        for t in tools_to_use
    ]

    return FormulaSpec(
        selections=selections,
        aggregation=AggregationMethod.WEIGHTED_SUM,
        threshold=threshold,
        rationale=f"[BACKTEST:{preset}] Deterministic fixed formula — no LLM",
    )


# ── Snapshot loading helpers ───────────────────────────────────────────────────

def _load_snapshots(
    snapshot_file: Path,
    market_filter: Optional[str],
) -> List[Dict[str, Any]]:
    """Load JSONL rows, optionally filtered by market_id substring."""
    rows = []
    with open(snapshot_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if market_filter and market_filter not in row.get("market_id", ""):
                continue
            rows.append(row)
    return rows


def _row_to_event(row: Dict[str, Any]) -> "Optional[EventInput]":
    """Convert a snapshot row dict to an EventInput. Returns None if invalid."""
    try:
        from schemas import EventInput

        ts = row.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif ts is None:
            ts = datetime.now(timezone.utc)

        price = float(row.get("last_price", row.get("current_price", 0.0)))
        market_id = row.get("market_id", "")
        if not market_id:
            return None

        return EventInput(
            event_id=row.get("event_id", market_id),
            market_id=market_id,
            market_title=row.get("title", row.get("market_title", market_id)),
            current_price=price,
            timestamp=ts,
        )
    except Exception as exc:
        logger.debug("Could not convert row to EventInput: %s — %s", row, exc)
        return None


# ── Report persistence ─────────────────────────────────────────────────────────

def _write_report(report: BacktestReport, output_file: Path) -> None:
    """Append BacktestReport as a JSON line to output_file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_dict = asdict(report)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(report_dict, default=str) + "\n")
    logger.info("Backtest report written to %s", output_file)


def print_report(report: BacktestReport) -> None:
    """Print a human-readable BacktestReport summary."""
    print("\n" + "=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)
    print(f"Snapshot file:      {report.snapshot_file}")
    print(f"Formula preset:     {report.formula_preset}")
    print(f"Market filter:      {report.market_filter or '(none)'}")
    print(f"Run timestamp:      {report.run_timestamp}")
    print()
    print(f"Total events:       {report.total_events}")
    print(f"Events skipped:     {report.total_events_skipped}")
    print(f"Bets triggered:     {report.bets_triggered}")
    print()
    print(f"Accuracy:           {report.accuracy:.2%}")
    print(f"Bet precision:      {report.bet_precision:.2%}")
    print(f"Bet recall:         {report.bet_recall:.2%}")
    print(f"Brier score:        {report.brier_score:.4f}  (lower = better)")
    print(f"Avg score delta:    {report.avg_score_delta:.4f}")
    print(f"Simulated P&L:      ${report.profit_simulation:.2f}")
    print()
    print("Tool contributions:")
    for tc in report.tool_contributions:
        print(
            f"  {tc.tool_name:<30s}  selected={tc.times_selected:4d}  "
            f"avg_output={tc.avg_output_mean:.4f}  "
            f"total_contrib={tc.total_score_contribution:.4f}"
        )
    print("=" * 60 + "\n")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    # Allow running from repo root with python -m prediction_agent.backtest.backtest
    from pathlib import Path as _P
    repo_root = _P(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Deterministic Backtesting Engine")
    parser.add_argument(
        "--snapshot-file",
        type=Path,
        default=None,
        help="Path to market_snapshots.jsonl (default: outputs/market_snapshots.jsonl)",
    )
    parser.add_argument(
        "--formula-preset",
        choices=["equal_weight", "volatility_only", "real_tools_only"],
        default="equal_weight",
        help="Which fixed formula preset to use (no LLM)",
    )
    parser.add_argument(
        "--market-filter",
        type=str,
        default=None,
        help="Filter snapshots by market_id substring (e.g. KXNBA)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Bet trigger threshold (default: 0.55)",
    )
    parser.add_argument(
        "--bet-amount",
        type=float,
        default=10.0,
        help="Paper dollar amount per bet (default: 10.0)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional JSONL file to append the BacktestReport to",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-event details",
    )
    args = parser.parse_args()

    report = run_backtest(
        snapshot_file=args.snapshot_file,
        formula_preset=args.formula_preset,
        market_filter=args.market_filter,
        threshold=args.threshold,
        bet_amount=args.bet_amount,
        output_file=args.output_file,
        verbose=args.verbose,
    )
    print_report(report)


if __name__ == "__main__":
    main()
