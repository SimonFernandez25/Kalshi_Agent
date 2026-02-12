"""
Main orchestrator — runs the full prediction pipeline.

Flow:
  1. Kalshi client → top 10 basketball markets
  2. Pick first market → build EventInput
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

from prediction_agent.api.kalshi_client import KalshiClient
from prediction_agent.agent.graph import build_agent_graph
from prediction_agent.engine.tool_runner import run_tools
from prediction_agent.engine.scorer import compute_score
from prediction_agent.engine.watcher import watch_market
from prediction_agent.engine.paper_broker import log_paper_bet
from prediction_agent.schemas import EventInput, FormulaSpec
from prediction_agent.tools.registry import build_default_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    }

    agent_output = graph.invoke(agent_input)
    formula = FormulaSpec(**agent_output["formula_spec"])

    logger.info("FormulaSpec:")
    logger.info("  Tools: %s", [(s.tool_name, s.weight) for s in formula.selections])
    logger.info("  Threshold: %.4f", formula.threshold)
    logger.info("  Rationale: %s", formula.rationale)

    if agent_output.get("error"):
        logger.warning("  Agent error (used fallback): %s", agent_output["error"])

    # ── Step 4: Run Tools + Score ────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Running deterministic tools and computing score")
    logger.info("=" * 60)

    tool_outputs = run_tools(event, formula, registry)
    score = compute_score(tool_outputs, formula)

    logger.info("Score Result:")
    logger.info("  Final Score: %.4f", score.final_score)
    logger.info("  Threshold: %.4f", score.threshold)
    logger.info("  Bet Triggered: %s", score.bet_triggered)

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
        "formula": formula.model_dump(mode="json"),
        "score": score.model_dump(mode="json"),
        "watcher_ticks": [t.model_dump(mode="json") for t in watcher_ticks],
        "paper_bet": paper_bet.model_dump(mode="json"),
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
