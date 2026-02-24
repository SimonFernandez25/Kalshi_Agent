"""
Main orchestrator — runs the full prediction pipeline.

Flow:
  1. Kalshi client → top 10 basketball markets
  2. Pick first market → build EventInput
  2a. Market sanity check — abstain if price == 0 or below MIN_VALID_PRICE
  3. LangGraph agent → FormulaSpec (tools + weights + threshold)
  4. Deterministic engine → run tools → compute score
  5. Watcher loop → poll Kalshi odds
  6. Paper broker → log simulated bet
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import List, Optional

from api.kalshi_client import KalshiClient
from agent.graph import build_agent_graph
from engine.tool_runner import run_tools
from engine.scorer import compute_score
from engine.watcher import watch_market
from engine.paper_broker import log_paper_bet
from schemas import EventInput, FormulaSpec, ScoreResult, ToolOutput
from tools.registry import build_default_registry
from config import ENABLE_EVOLUTION, EXECUTION_LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Sanity check ────────────────────────────────────────────────────────────────

def _market_is_valid(event: EventInput) -> tuple[bool, str]:
    """
    Return (True, "") if the market is tradeable, or (False, reason) if not.

    Abstain conditions:
      - current_price <= MIN_VALID_PRICE  →  no market signal (stub / illiquid)
      - current_price >= 1 - MIN_VALID_PRICE  →  market already resolved (99c+)
    """
    try:
        from config import ABSTAIN_ON_ZERO_PRICE, MIN_VALID_PRICE
    except ImportError:
        ABSTAIN_ON_ZERO_PRICE = True
        MIN_VALID_PRICE = 0.01

    if not ABSTAIN_ON_ZERO_PRICE:
        return True, ""

    price = event.current_price
    if price <= MIN_VALID_PRICE:
        return False, f"price={price:.4f} <= MIN_VALID_PRICE={MIN_VALID_PRICE} (no signal)"
    if price >= 1.0 - MIN_VALID_PRICE:
        return False, f"price={price:.4f} >= {1.0 - MIN_VALID_PRICE:.4f} (market resolved)"
    return True, ""


# ── Probabilistic run summary ───────────────────────────────────────────────────

def _print_run_summary(
    event: EventInput,
    formula: FormulaSpec,
    score: ScoreResult,
    registry,
    tool_statuses: list,
    tool_outputs: List[ToolOutput],
    mode: str = "LIVE",
    market_domain: Optional[str] = None,
    scoring_mode: Optional[str] = None,
) -> None:
    """
    Print a human-readable probabilistic run summary.

    Shows:
      - Domain + scoring mode
      - Market price
      - Model probability (probability_edge mode) or weighted score (signal_sum)
      - Edge (model_prob - market_price)
      - Per-tool contribution to probability / score
      - Bet decision
    """
    SEP = "=" * 62
    sep = "-" * 62

    print(f"\n{SEP}")
    print(f"  RUN SUMMARY [{mode} MODE]")
    print(SEP)

    domain_str = market_domain or "unknown"
    sm_str = scoring_mode or "signal_sum"
    print(f"  Market : {event.market_title}")
    print(f"  Domain : {domain_str}   |   Scoring : {sm_str}")
    print(sep)

    market_price = event.current_price

    if sm_str == "probability_edge" and score.model_probability is not None:
        model_prob = score.model_probability
        edge = score.edge if score.edge is not None else (model_prob - market_price)
        edge_sign = "+" if edge >= 0 else ""
        bet_side = "YES" if edge > 0 else "NO"

        # Show raw score z (pre-calibration) if available
        if score.raw_score_z is not None:
            try:
                from config import USE_LOGISTIC_CALIBRATION
            except ImportError:
                USE_LOGISTIC_CALIBRATION = True

            calib_status = "sigmoid(z)" if USE_LOGISTIC_CALIBRATION else "clamp(z)"
            print(f"  Raw Score (z)       : {score.raw_score_z:.4f}  (pre-calibration)")
            print(f"  Model Probability   : {model_prob:.4f}  ({model_prob*100:.1f}%) [{calib_status}]")
        else:
            print(f"  Model Probability   : {model_prob:.4f}  ({model_prob*100:.1f}%)")

        print(f"  Market Implied Prob : {market_price:.4f}  ({market_price*100:.1f}%)")
        print(f"  Edge                : {edge_sign}{edge:.4f}  ({edge_sign}{edge*100:.1f}pp)")
        print(f"  Threshold (min edge): {score.threshold:.4f}")
        print(f"  Bet Triggered       : {score.bet_triggered}  →  {bet_side if score.bet_triggered else 'PASS'}")
        print(sep)

        # Per-tool contribution to model probability
        try:
            from config import PROBABILITY_SIGNAL_TOOLS
        except ImportError:
            PROBABILITY_SIGNAL_TOOLS = [
                "sportsbook_implied_probability_tool",
                "weather_probability_tool",
            ]

        print("  Tool Contribution to Raw Score (z-space attribution):")
        for output, sel in zip(tool_outputs, formula.selections):
            vec = output.output_vector
            w = sel.weight
            if sel.tool_name in PROBABILITY_SIGNAL_TOOLS:
                p_signal = vec[0] if vec else 0.0
                contribution = w * p_signal
                tag = "(probability signal)"
            else:
                mean_sig = sum(vec) / len(vec) if vec else 0.0
                contribution = w * mean_sig
                tag = "(mean signal)"
            contrib_sign = "+" if contribution >= 0 else ""
            print(f"    {sel.tool_name:<40s} {contrib_sign}{contribution:.4f}  {tag}")

    else:
        # signal_sum mode
        print(f"  Market Price        : {market_price:.4f}  ({market_price*100:.1f}%)")
        print(f"  Final Score         : {score.final_score:.4f}")
        print(f"  Threshold           : {score.threshold:.4f}")
        print(f"  Bet Triggered       : {score.bet_triggered}")
        print(sep)

        print("  Tool Signal Contributions:")
        for output, sel in zip(tool_outputs, formula.selections):
            vec = output.output_vector
            mean_sig = sum(vec) / len(vec) if vec else 0.0
            contribution = sel.weight * mean_sig
            contrib_sign = "+" if contribution >= 0 else ""
            print(f"    {sel.tool_name:<40s} {contrib_sign}{contribution:.4f}  (w={sel.weight:.2f})")

    print(sep)

    # Tool execution status
    print("  Tool Execution:")
    for status in tool_statuses:
        icon = "OK" if status.success else "!!"
        latency = f"{status.latency_ms:.0f}ms"
        err = f"  ERROR: {status.error}" if status.error else ""
        print(f"    [{icon}] {status.tool_name:<38s} {latency:>6s}{err}")

    print(f"\n  Registry Size: {len(registry)} tools")
    print(SEP + "\n")


# ── Pipeline ────────────────────────────────────────────────────────────────────

def run_pipeline(
    market_index: int = 0,
    skip_watcher: bool = False,
    watcher_max_ticks: int = 3,
) -> dict:
    """
    Execute the full prediction pipeline.

    Args:
        market_index: Which market from the top-10 to use (0-based).
        skip_watcher: If True, skip the watcher loop (for fast testing).
        watcher_max_ticks: Cap watcher polls for V1.

    Returns:
        Dict summary of the run.
    """
    # ── Step 1: Fetch markets ────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching top basketball markets from Kalshi")
    logger.info("=" * 60)

    client = KalshiClient()
    markets = client.get_active_markets(limit=10)

    logger.info("Got %d markets:", len(markets))
    for i, m in enumerate(markets):
        logger.info("  [%d] %s — price=%.2f", i, m.get("title", "Unknown"), m.get("last_price", 0.0))

    # ── Step 2: Build EventInput ─────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Selecting market [%d] as EventInput", market_index)
    logger.info("=" * 60)

    chosen = markets[min(market_index, len(markets) - 1)]
    event = EventInput(
        event_id=chosen.get("event_id", chosen["market_id"]),
        market_id=chosen["market_id"],
        market_title=chosen.get("title", "Unknown Market"),
        current_price=chosen.get("last_price", 0.0),
        timestamp=datetime.fromisoformat(chosen["timestamp"]) if isinstance(chosen["timestamp"], str)
                  else chosen["timestamp"],
    )
    logger.info("EventInput: %s", event.model_dump())

    # ── Step 2a: Market sanity check ─────────
    valid, reason = _market_is_valid(event)
    if not valid:
        logger.warning("ABSTAIN — market sanity check failed: %s", reason)
        print(f"\n[ABSTAIN] {event.market_title}")
        print(f"  Reason: {reason}")
        print("  No bet placed. Skipping pipeline.\n")
        return {
            "run_id": None,
            "event": event.model_dump(mode="json"),
            "abstain": True,
            "abstain_reason": reason,
            "formula": None,
            "score": None,
            "watcher_ticks": [],
            "paper_bet": None,
        }

    # ── Step 3: Run Agent Graph ──────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Running LangGraph agent to select tools")
    logger.info("=" * 60)

    registry = build_default_registry()
    graph = build_agent_graph()

    agent_input = {
        "event_input": event.model_dump(mode="json"),
        "tools_list": registry.list_tools(),
        "formula_spec": None,
        "error": None,
        "market_domain": None,
        "scoring_mode": None,
    }

    agent_output = graph.invoke(agent_input)
    formula = FormulaSpec(**agent_output["formula_spec"])

    # Pull domain + scoring_mode from agent state
    market_domain = agent_output.get("market_domain")
    scoring_mode = agent_output.get("scoring_mode")

    logger.info("FormulaSpec:")
    logger.info("  Tools: %s", [(s.tool_name, s.weight) for s in formula.selections])
    logger.info("  Threshold: %.4f", formula.threshold)
    logger.info("  Rationale: %s", formula.rationale)
    logger.info("  Domain: %s | Scoring: %s", market_domain, scoring_mode)

    if agent_output.get("error"):
        logger.warning("  Agent error (used fallback): %s", agent_output["error"])

    # ── Step 4: Run Tools + Score ────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Running deterministic tools and computing score")
    logger.info("=" * 60)

    tool_outputs, tool_statuses = run_tools(event, formula, registry)
    score = compute_score(
        tool_outputs,
        formula,
        scoring_mode=scoring_mode,
        current_market_price=event.current_price,
    )

    logger.info("Score Result:")
    logger.info("  Final Score: %.4f", score.final_score)
    logger.info("  Threshold: %.4f", score.threshold)
    logger.info("  Bet Triggered: %s", score.bet_triggered)
    if score.model_probability is not None:
        logger.info("  Model Probability: %.4f", score.model_probability)
    if score.edge is not None:
        logger.info("  Edge: %.4f", score.edge)

    # Print probabilistic run summary for human oversight
    try:
        from config import LIVE_MODE
    except ImportError:
        LIVE_MODE = True

    _print_run_summary(
        event=event,
        formula=formula,
        score=score,
        registry=registry,
        tool_statuses=tool_statuses,
        tool_outputs=tool_outputs,
        mode="LIVE" if LIVE_MODE else "REPLAY",
        market_domain=market_domain,
        scoring_mode=scoring_mode,
    )

    # ── Step 5: Watcher Loop ─────────────────
    watcher_ticks = []
    if not skip_watcher:
        logger.info("=" * 60)
        logger.info("STEP 5: Running watcher loop")
        logger.info("=" * 60)

        watcher_ticks = watch_market(
            client=client,
            market_id=event.market_id,
            threshold=formula.threshold,
            max_ticks=watcher_max_ticks,
        )
    else:
        logger.info("STEP 5: Watcher skipped (--skip-watcher)")

    # ── Step 6: Log Paper Bet ────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Logging paper bet")
    logger.info("=" * 60)

    paper_bet = log_paper_bet(
        event=event,
        formula=formula,
        score=score,
        watcher_ticks=watcher_ticks,
    )

    # ── Step 7: Evolution (optional) ────────
    if ENABLE_EVOLUTION:
        logger.info("=" * 60)
        logger.info("STEP 7: Running evolution pipeline")
        logger.info("=" * 60)

        from prediction_agent.evolution.execution_logger import log_execution
        from agent.graph import build_evolution_graph

        run_result = {
            "run_id": paper_bet.run_id,
            "event": event.model_dump(mode="json"),
            "formula": formula.model_dump(mode="json"),
            "score": score.model_dump(mode="json"),
        }

        log_execution(run_result)

        try:
            evo_graph = build_evolution_graph()
            evo_result = evo_graph.invoke({
                "log_path": str(EXECUTION_LOG_FILE),
                "gap_report": None,
                "tool_spec": None,
                "tool_path": None,
                "verification_result": None,
                "registry_updated": False,
                "error": None,
            })

            if evo_result.get("registry_updated"):
                logger.info("Evolution: new tool registered for next run.")
            elif evo_result.get("error"):
                logger.warning("Evolution error: %s", evo_result["error"])
            else:
                logger.info("Evolution: no new tool this cycle.")
        except Exception as exc:
            logger.warning("Evolution pipeline failed (non-fatal): %s", exc)
    else:
        logger.info("STEP 7: Evolution skipped (ENABLE_EVOLUTION=False)")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Run ID: %s", paper_bet.run_id)
    logger.info("  Market: %s", event.market_title)
    logger.info("  Score: %.4f", score.final_score)
    logger.info("  Bet Placed: %s", paper_bet.bet_placed)
    logger.info("  Bet Side: %s", paper_bet.bet_side)
    logger.info("=" * 60)

    return {
        "run_id": paper_bet.run_id,
        "event": event.model_dump(mode="json"),
        "abstain": False,
        "formula": formula.model_dump(mode="json"),
        "score": score.model_dump(mode="json"),
        "watcher_ticks": [t.model_dump(mode="json") for t in watcher_ticks],
        "paper_bet": paper_bet.model_dump(mode="json"),
        "market_domain": market_domain,
        "scoring_mode": scoring_mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Prediction Agent Pipeline")
    parser.add_argument(
        "--market-index",
        type=int,
        default=0,
        help="Index of market to pick from top-10 (default: 0)",
    )
    parser.add_argument(
        "--skip-watcher",
        action="store_true",
        help="Skip the watcher loop for fast testing",
    )
    parser.add_argument(
        "--watcher-ticks",
        type=int,
        default=3,
        help="Max watcher polls (default: 3)",
    )
    args = parser.parse_args()

    run_pipeline(
        market_index=args.market_index,
        skip_watcher=args.skip_watcher,
        watcher_max_ticks=args.watcher_ticks,
    )


if __name__ == "__main__":
    main()
