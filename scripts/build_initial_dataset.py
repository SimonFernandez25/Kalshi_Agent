"""
Build initial dataset from 25 closed markets.

Purpose: Run the full pipeline on closed markets to:
1. Build training data
2. Identify failure modes
3. Collect tool outputs for analysis

Settings:
- 25 closed markets
- Threshold: 0.30 (low, to trigger more bets)
- Bet amount: $10 per market
- Full domain-aware pipeline with all tools
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
from datetime import datetime, timezone

from api.kalshi_client import KalshiClient
from engine.market_classifier import classify_market, get_domain_tools_with_weather
from engine.tool_runner import run_tools
from engine.scorer import compute_score
from main import _market_is_valid
from schemas import EventInput, FormulaSpec, ToolSelection
from tools.registry import build_default_registry
from prediction_agent.storage.sqlite_store import SQLiteStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# -- Configuration --------------------------------------------------------------
THRESHOLD = 0.30           # Low threshold to trigger more bets
BET_AMOUNT = 10.0          # $10 per bet
MAX_MARKETS = 25           # Number of closed markets to process
MARKET_STATUS = "closed"   # Only closed markets (we know outcomes)
# -------------------------------------------------------------------------------


def build_formula_for_domain(domain: str, registry, market_id: str, market_title: str) -> FormulaSpec:
    """Build equal-weight formula with domain-filtered tools."""
    all_tool_names = registry.tool_names
    domain_tool_names = get_domain_tools_with_weather(domain, all_tool_names, market_id, market_title)

    if not domain_tool_names:
        # Fallback to all tools if no domain match
        domain_tool_names = all_tool_names[:5]

    n = len(domain_tool_names)
    weight = 1.0 / n
    selections = [ToolSelection(tool_name=name, tool_inputs={}, weight=weight) for name in domain_tool_names]

    return FormulaSpec(
        selections=selections,
        aggregation='weighted_sum',
        threshold=THRESHOLD,
        rationale=f'Equal-weight {n} {domain} tools (initial dataset build)'
    )


def main():
    print("=" * 80)
    print("  INITIAL DATASET BUILD — 25 CLOSED MARKETS")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Threshold       : {THRESHOLD} (low, to trigger more bets)")
    print(f"  Bet Amount      : ${BET_AMOUNT}")
    print(f"  Max Markets     : {MAX_MARKETS}")
    print(f"  Market Status   : {MARKET_STATUS}")
    print("=" * 80)
    print()

    # Initialize
    client = KalshiClient()
    registry = build_default_registry()
    store = SQLiteStore()

    # Fetch closed markets
    print(f"[1/4] Fetching {MAX_MARKETS} closed markets from Kalshi...")
    try:
        markets = client.get_markets(limit=MAX_MARKETS, status=MARKET_STATUS)
    except Exception as e:
        logger.error(f"Failed to fetch markets: {e}")
        print(f"ERR API Error: {e}")
        return

    print(f"OK Fetched {len(markets)} closed markets")
    print()

    # Process each market
    print(f"[2/4] Processing markets with full pipeline...")
    print("-" * 80)

    results = []
    abstained = []
    failed = []

    for i, mkt in enumerate(markets, 1):
        market_id = mkt.get('market_id', f'unknown_{i}')
        title = mkt.get('title', 'Unknown')
        price = mkt.get('last_price', mkt.get('yes_sub_title', 0.0))

        # Handle price
        if isinstance(price, str):
            try:
                price = float(price)
            except (ValueError, TypeError):
                price = 0.0

        result_ticker = mkt.get('result', 'unknown')

        print(f"[{i}/{MAX_MARKETS}] {market_id}")
        print(f"      Title : {title[:70]}")
        print(f"      Price : {price:.3f}  |  Result: {result_ticker}")

        # Build EventInput
        try:
            event = EventInput(
                event_id=mkt.get('event_id', market_id),
                market_id=market_id,
                market_title=title,
                current_price=price,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            print(f"      ERR Failed to build EventInput: {e}")
            failed.append({'market_id': market_id, 'reason': f'EventInput error: {e}'})
            print()
            continue

        # Market sanity check
        valid, reason = _market_is_valid(event)
        if not valid:
            print(f"      X ABSTAIN: {reason}")
            abstained.append({'market_id': market_id, 'reason': reason, 'price': price})
            print()
            continue

        # Classify domain
        domain = classify_market(market_id, title)
        print(f"      Domain: {domain}")

        # Build formula
        formula = build_formula_for_domain(domain, registry, market_id, title)
        print(f"      Tools : {[s.tool_name for s in formula.selections]}")

        # Run tools
        try:
            tool_outputs, tool_statuses = run_tools(event, formula, registry)
        except Exception as e:
            print(f"      ERR Tool execution failed: {e}")
            failed.append({'market_id': market_id, 'reason': f'Tool error: {e}'})
            print()
            continue

        # Score
        scoring_mode = 'probability_edge' if domain == 'sports' else 'signal_sum'
        try:
            score = compute_score(
                tool_outputs,
                formula,
                scoring_mode=scoring_mode,
                current_market_price=price
            )
        except Exception as e:
            print(f"      ERR Scoring failed: {e}")
            failed.append({'market_id': market_id, 'reason': f'Scoring error: {e}'})
            print()
            continue

        # Determine bet
        bet_triggered = score.bet_triggered
        bet_side = "YES" if score.final_score >= THRESHOLD else "NO"

        # Show result
        bet_flag = "[BET]" if bet_triggered else "     "
        print(f"      {bet_flag} Score: {score.final_score:.4f}  |  Bet: {bet_side if bet_triggered else 'PASS'}")

        if scoring_mode == 'probability_edge' and score.model_probability:
            print(f"      p_model={score.model_probability:.3f}  edge={score.edge:.4f}  z={score.raw_score_z:.3f}")

        # Log to SQLite
        run_id = f"dataset_{i:03d}"
        try:
            # Determine outcome: 'yes' -> 1.0, 'no' -> 0.0, else None
            outcome = None
            if result_ticker == 'yes':
                outcome = 1.0
            elif result_ticker == 'no':
                outcome = 0.0

            store.insert_run(
                run_id=run_id,
                event=event,
                formula=formula,
                score=score,
                outcome=outcome
            )

            print(f"      OK Logged to SQLite (run_id={run_id})")
        except Exception as e:
            print(f"      ERR SQLite write failed: {e}")

        # Collect result
        results.append({
            'run_id': run_id,
            'market_id': market_id,
            'title': title,
            'price': price,
            'domain': domain,
            'score': score.final_score,
            'bet_triggered': bet_triggered,
            'bet_side': bet_side if bet_triggered else None,
            'result': result_ticker,
            'scoring_mode': scoring_mode,
            'model_probability': score.model_probability,
            'edge': score.edge,
            'raw_score_z': score.raw_score_z,
        })

        print()

    # Summary
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"Total markets       : {len(markets)}")
    print(f"Processed           : {len(results)}")
    print(f"Abstained           : {len(abstained)}")
    print(f"Failed              : {len(failed)}")
    print()

    if results:
        bets_triggered = sum(1 for r in results if r['bet_triggered'])
        print(f"Bets triggered      : {bets_triggered}/{len(results)} ({100*bets_triggered/len(results):.1f}%)")
        print(f"Total bet amount    : ${bets_triggered * BET_AMOUNT:.2f}")
        print()

        # Show bet breakdown
        yes_bets = sum(1 for r in results if r['bet_triggered'] and r['bet_side'] == 'YES')
        no_bets = sum(1 for r in results if r['bet_triggered'] and r['bet_side'] == 'NO')
        print(f"YES bets            : {yes_bets}")
        print(f"NO bets             : {no_bets}")
        print()

        # Domain breakdown
        from collections import Counter
        domain_counts = Counter(r['domain'] for r in results)
        print("Domains processed:")
        for domain, count in domain_counts.most_common():
            print(f"  {domain:<15s} : {count}")
        print()

    # Show abstained reasons
    if abstained:
        print("Abstained markets:")
        for a in abstained[:5]:  # Show first 5
            print(f"  {a['market_id']}: {a['reason']}")
        if len(abstained) > 5:
            print(f"  ... and {len(abstained) - 5} more")
        print()

    # Show failures
    if failed:
        print("Failed markets:")
        for f in failed:
            print(f"  {f['market_id']}: {f['reason']}")
        print()

    print("=" * 80)
    print(f"Dataset saved to SQLite: {store.db_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
