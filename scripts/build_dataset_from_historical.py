"""
Build initial dataset from HISTORICAL (settled) markets using Kalshi Historical API.

Purpose: Run the full pipeline on markets with KNOWN OUTCOMES to validate the system.

This script:
1. Fetches settled markets from the Kalshi Historical API (settled in last 3 months)
2. Runs the full domain-aware pipeline on each market
3. Logs predictions to SQLite with known outcomes
4. Shows where the system succeeds/fails with real data

Key advantage over live markets:
- Historical markets have known outcomes (result='yes' or 'no')
- Can validate prediction accuracy immediately
- No dependency on live data collection

Settings:
- 50 historical settled markets
- Threshold: 0.30 (low, to trigger more bets)
- Bet amount: $10 per market
- Uses ALL available tools (snapshot + external)

See: https://docs.kalshi.com/historical-data
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
from datetime import datetime, timezone, timedelta

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
MAX_MARKETS = 50           # Number of historical markets to process
LOOKBACK_DAYS = 90         # Fetch markets settled in last 90 days
# -------------------------------------------------------------------------------


def build_domain_formula(market_id: str, market_title: str, registry) -> FormulaSpec:
    """
    Build equal-weight formula with domain-filtered tools.

    Uses the same domain-aware tool selection as the live system:
    - Classify market domain (sports/economics/politics/weather/crypto/other)
    - Filter tools to those relevant for the domain
    - Equal weight across selected tools
    """
    # Get all probability signal tools
    from config import PROBABILITY_SIGNAL_TOOLS
    available_tools = [t for t in PROBABILITY_SIGNAL_TOOLS if t in registry.tool_names]

    # Classify domain and filter tools
    domain = classify_market(market_id, market_title)
    domain_tools = get_domain_tools_with_weather(domain, available_tools, market_id, market_title)

    if not domain_tools:
        # Fallback to first 4 available tools
        domain_tools = available_tools[:4]

    # Limit to 4 tools for equal weighting
    tool_list = domain_tools[:4]
    n = len(tool_list)
    weight = 1.0 / n

    selections = [ToolSelection(tool_name=name, tool_inputs={}, weight=weight) for name in tool_list]

    return FormulaSpec(
        selections=selections,
        aggregation='weighted_sum',
        threshold=THRESHOLD,
        rationale=f'Domain-aware ({domain}): {n} equal-weight tools'
    )


def main():
    print("=" * 80)
    print("  DATASET BUILD FROM HISTORICAL MARKETS")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Threshold       : {THRESHOLD} (low, to trigger more bets)")
    print(f"  Bet Amount      : ${BET_AMOUNT}")
    print(f"  Max Markets     : {MAX_MARKETS}")
    print(f"  Lookback        : {LOOKBACK_DAYS} days")
    print(f"  Data Source     : Kalshi Historical API (settled markets)")
    print("=" * 80)
    print()

    # Initialize
    client = KalshiClient()
    registry = build_default_registry()
    store = SQLiteStore()

    # Check historical cutoff
    print("[1/5] Checking historical data cutoff...")
    cutoff = client.get_historical_cutoff()
    if cutoff:
        market_cutoff = cutoff.get('market_settled_ts', 'unknown')
        print(f"OK Historical market cutoff: {market_cutoff}")
    else:
        print("WARN Could not fetch cutoff (continuing anyway)")
    print()

    # Calculate time range for historical markets
    # Note: Don't constrain by date - let the API return whatever historical markets it has
    # The historical cutoff is managed by Kalshi (currently 2024-07-10)
    print(f"[2/5] Fetching {MAX_MARKETS} historical markets...")
    print(f"      Fetching markets settled before the historical cutoff")
    try:
        markets = client.get_historical_markets(
            limit=MAX_MARKETS,
            status="settled",
            # Don't constrain by date - get whatever historical markets exist
        )
    except Exception as e:
        logger.error(f"Failed to fetch historical markets: {e}")
        print(f"ERR API Error: {e}")
        return

    if not markets:
        print("ERR No historical markets returned")
        print("    This could mean:")
        print("    1. No markets settled in the last 90 days")
        print("    2. Historical API not yet enabled on your account")
        print("    3. Authentication issue")
        return

    print(f"OK Fetched {len(markets)} historical markets")
    print()

    # Process each market
    print(f"[3/5] Processing markets with full pipeline...")
    print("-" * 80)

    results = []
    abstained = []
    failed = []

    for i, mkt in enumerate(markets, 1):
        market_id = mkt.get('market_id', f'unknown_{i}')
        title = mkt.get('title', 'Unknown')
        price = mkt.get('last_price', 0.0)
        result_ticker = mkt.get('result', 'unknown')
        close_time = mkt.get('close_time', 'unknown')

        print(f"[{i}/{len(markets)}] {market_id}")
        print(f"      Title  : {title[:70]}")
        print(f"      Price  : {price:.3f}  |  Result: {result_ticker}")
        print(f"      Closed : {close_time}")

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

        # Market sanity check (still useful to filter resolved/zero-price)
        valid, reason = _market_is_valid(event)
        if not valid:
            print(f"      X ABSTAIN: {reason}")
            abstained.append({'market_id': market_id, 'reason': reason, 'price': price})
            print()
            continue

        # Classify domain
        domain = classify_market(market_id, title)
        print(f"      Domain : {domain}")

        # Build formula with domain-filtered tools
        try:
            formula = build_domain_formula(market_id, title, registry)
        except Exception as e:
            print(f"      ERR Formula build failed: {e}")
            failed.append({'market_id': market_id, 'reason': f'Formula error: {e}'})
            print()
            continue

        print(f"      Tools  : {[s.tool_name for s in formula.selections]}")

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
        bet_side = "YES" if score.final_score >= THRESHOLD else "NO" if score.final_score <= -THRESHOLD else "PASS"

        # Show result
        bet_flag = "[BET]" if bet_triggered else "     "
        print(f"      {bet_flag} Score: {score.final_score:+.4f}  |  Bet: {bet_side}")

        if scoring_mode == 'probability_edge' and score.model_probability is not None:
            print(f"             p_model={score.model_probability:.3f}  edge={score.edge:+.4f}  z={score.raw_score_z:.3f}")

        # Calculate correctness if bet was triggered
        correct = None
        if bet_triggered and result_ticker in ['yes', 'no']:
            outcome_value = 1.0 if result_ticker == 'yes' else 0.0
            if bet_side == 'YES':
                correct = (result_ticker == 'yes')
            elif bet_side == 'NO':
                correct = (result_ticker == 'no')

            if correct is not None:
                correct_str = "CORRECT" if correct else "WRONG"
                print(f"             Prediction: {correct_str}")

        # Log to SQLite
        run_id = f"historical_{i:03d}"
        try:
            # Determine outcome
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
            'outcome_value': 1.0 if result_ticker == 'yes' else 0.0 if result_ticker == 'no' else None,
            'correct': correct,
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

        # Accuracy calculation
        bets_with_outcome = [r for r in results if r['bet_triggered'] and r['correct'] is not None]
        if bets_with_outcome:
            correct_count = sum(1 for r in bets_with_outcome if r['correct'])
            accuracy = 100 * correct_count / len(bets_with_outcome)
            print(f"Prediction Accuracy : {correct_count}/{len(bets_with_outcome)} ({accuracy:.1f}%)")
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

        # Outcome breakdown
        outcome_counts = Counter(r['result'] for r in results)
        print("Outcomes:")
        for outcome, count in outcome_counts.most_common():
            print(f"  {outcome:<15s} : {count}")
        print()

    # Show abstained reasons
    if abstained:
        print("Abstained markets:")
        for a in abstained[:5]:
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
    print(f"Dataset saved to SQLite: {store._path}")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Query SQLite to analyze tool performance:")
    print("     SELECT tool_id, AVG(contribution), COUNT(*) FROM tool_outputs GROUP BY tool_id")
    print("  2. Run backtest evaluation notebook:")
    print("     notebooks/03_backtest_eval.ipynb")
    print("  3. Adjust threshold based on accuracy/trigger rate")
    print("=" * 80)


if __name__ == "__main__":
    main()
