"""
Single Pass Execution Script
============================

Running this script performs one complete pass of the prediction agent:
1. Validates API connection.
2. Fetches ONE active market from Kalshi.
3. Invokes the Agent (AWS Bedrock) to analyze the market and select tools.
4. Executes the selected tools to get signal vectors.
5. Computes the final weighted score.
6. Prints a detailed review of the decision.

Usage:
    python -m prediction_agent.one_pass.run_pipeline
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from prediction_agent.api.kalshi_client import KalshiClient
from prediction_agent.schemas import EventInput, FormulaSpec
from prediction_agent.agent.nodes import select_tools_node
from prediction_agent.engine.tool_runner import run_tools
from prediction_agent.engine.scorer import compute_score
from prediction_agent.tools.registry import build_default_registry

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simplified format for readability
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("one_pass")

def main():
    print("\n" + "="*60)
    print(" PREDICTION AGENT: SINGLE PASS EXECUTION ")
    print("="*60 + "\n")

    # 1. Initialize Client
    print(">>> Initializing Kalshi Client...")
    try:
        client = KalshiClient()
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        return

    # 2. Setup Tool Registry
    print(">>> Building Tool Registry...")
    registry = build_default_registry()

    # 3. Fetch One Market
    print(">>> Fetching active markets...")
    try:
        markets = client.get_active_markets(limit=5)
    except Exception as e:
        logger.error(f"Failed to fetch markets: {e}")
        markets = []
        
    if not markets:
        logger.error("No active markets found!")
        return
    
    # Pick the first one
    chosen = markets[0]
    print(f"    Selected Market: {chosen.get('title', 'Unknown')}")
    print(f"    Current Price:   {chosen.get('last_price', 'N/A')}")
    print(f"    Market ID:       {chosen['market_id']}")

    # 4. Create EventInput
    event = EventInput(
        event_id=chosen.get("event_id", chosen["market_id"]),
        market_id=chosen["market_id"],
        market_title=chosen.get("title", "Unknown"),
        current_price=chosen.get("last_price", 0.0),
        timestamp=datetime.now() # Use current time for the run
    )

    # 5. Agent Analysis (Bedrock)
    print("\n>>> invoking Agent (Bedrock)...")
    tools_list = registry.list_tools()
    
    # Construct state for the node
    state = {
        "event_input": event.model_dump(),
        "tools_list": tools_list,
        "formula_spec": None,
        "error": None
    }

    # Run the node
    state = select_tools_node(state)

    if state.get("error"):
        logger.warning(f"Agent Warning: {state['error']}")

    spec_data = state.get("formula_spec")
    if not spec_data:
        logger.error("Agent failed to produce a specification.")
        return

    spec = FormulaSpec(**spec_data)
    
    print(f"    Rationale: {spec.rationale}")
    print(f"    Threshold: {spec.threshold}")
    print("    Selected Tools:")
    for sel in spec.selections:
        print(f"      - {sel.tool_name} (Weight: {sel.weight:.2f})")

    # 6. Execute Tools
    print("\n>>> Appending current snapshot to history (for tool freshness)...")
    import json
    # Use 'chosen' dict to get full market details (bids, asks, etc.)
    snapshot_entry = {
        "timestamp": event.timestamp.isoformat(),
        "market_id": event.market_id,
        "title": event.market_title,
        "last_price": event.current_price,
        "status": chosen.get("status", "active"),
        "yes_bid": chosen.get("yes_bid", 0.0),
        "yes_ask": chosen.get("yes_ask", 0.0),
        "volume": chosen.get("volume", 0),
        "open_interest": chosen.get("open_interest", 0),
        "liquidity": chosen.get("liquidity", 0.0),
        "close_time": chosen.get("close_time"),
    }
    
    # Path to snapshots file
    snapshots_path = PROJECT_ROOT / "prediction_agent" / "outputs" / "market_snapshots.jsonl"
    try:
        with open(snapshots_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot_entry) + "\n")
        print("    Snapshot saved.")
    except Exception as e:
        logger.warning(f"Failed to save snapshot: {e}")

    print("\n>>> Executing Selected Tools...")
    tool_outputs, tool_statuses = run_tools(event, spec, registry)
    for out in tool_outputs:
        print(f"    Result '{out.tool_name}': {out.output_vector}")

    # 7. Compute Score
    print("\n>>> Computing Final Score...")
    score_result = compute_score(tool_outputs, spec)

    # 8. Final Review
    print("\n" + "="*60)
    print(" EXECUTION REVIEW ")
    print("="*60)
    print(f"Market:      {event.market_title}")
    print(f"Final Score: {score_result.final_score:.4f}")
    print(f"Threshold:   {score_result.threshold:.4f}")
    print(f"Decision:    {'BET PLACED (YES)' if score_result.bet_triggered else 'NO BET'}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
