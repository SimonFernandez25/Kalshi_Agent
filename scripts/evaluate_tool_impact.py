"""
Tool Impact Evaluation Pipeline

Purpose: Deterministic evaluation of tool predictive power on historical markets.

This script:
1. Fetches N historical markets with known outcomes
2. Re-runs the full scoring pipeline on each
3. Logs per-tool contributions to SQLite
4. Computes tool-level predictive metrics
5. Outputs ranked tool effectiveness

Research Goal: Answer "Which tools are actually predictive?"

Metrics Computed:
- Market-level: Accuracy, Brier Score, Trigger Rate, ROI
- Tool-level: Correlation, IC, Mean Signal, Win Rate

Constraints:
- Deterministic execution (no randomness)
- No LLM involvement
- No external data refresh during evaluation
- Tools use data consistent with market timestamp

Output: Console table ranking tools by predictive power
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import math

from api.kalshi_client import KalshiClient
from engine.market_classifier import classify_market, get_domain_tools_with_weather
from engine.tool_runner import run_tools
from engine.scorer import compute_score
from main import _market_is_valid
from schemas import EventInput, FormulaSpec, ToolSelection
from tools.registry import build_default_registry
from prediction_agent.storage.sqlite_store import SQLiteStore

logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO logs for clean output
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
N_HISTORICAL_MARKETS = 200    # Number of historical markets to evaluate
THRESHOLD = 0.30              # Bet threshold (must match training/production)
BET_AMOUNT = 10.0             # Simulated bet size for ROI calculation
# ───────────────────────────────────────────────────────────────────────────────


def build_domain_formula(market_id: str, market_title: str, registry) -> FormulaSpec:
    """
    Build equal-weight formula with domain-filtered tools.
    Same logic as production pipeline.
    """
    from config import PROBABILITY_SIGNAL_TOOLS
    available_tools = [t for t in PROBABILITY_SIGNAL_TOOLS if t in registry.tool_names]

    domain = classify_market(market_id, market_title)
    domain_tools = get_domain_tools_with_weather(domain, available_tools, market_id, market_title)

    if not domain_tools:
        domain_tools = available_tools[:4]

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


def compute_brier_score(predictions: List[Tuple[float, float]]) -> float:
    """
    Compute Brier score: mean((p_model - outcome)^2)

    Args:
        predictions: List of (p_model, outcome) tuples

    Returns:
        Brier score (lower is better, 0 = perfect, 1 = worst)
    """
    if not predictions:
        return float('nan')
    return sum((p - o) ** 2 for p, o in predictions) / len(predictions)


def compute_correlation(x: List[float], y: List[float]) -> float:
    """
    Compute Pearson correlation coefficient.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Correlation coefficient (-1 to 1, 0 = no correlation)
    """
    if not x or not y or len(x) != len(y):
        return float('nan')

    n = len(x)
    if n < 2:
        return float('nan')

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))

    if denominator_x == 0 or denominator_y == 0:
        return float('nan')

    return numerator / math.sqrt(denominator_x * denominator_y)


def main():
    print("=" * 80)
    print("  TOOL IMPACT EVALUATION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  N Markets       : {N_HISTORICAL_MARKETS}")
    print(f"  Threshold       : {THRESHOLD}")
    print(f"  Bet Amount      : ${BET_AMOUNT}")
    print(f"  Mode            : Deterministic evaluation (no randomness)")
    print("=" * 80)
    print()

    # Initialize
    client = KalshiClient()
    registry = build_default_registry()
    store = SQLiteStore()

    # Fetch historical markets
    print(f"[1/4] Fetching {N_HISTORICAL_MARKETS} historical markets...")
    try:
        markets = client.get_historical_markets(
            limit=N_HISTORICAL_MARKETS,
            status="settled",
        )
    except Exception as e:
        logger.error(f"Failed to fetch historical markets: {e}")
        print(f"ERR API Error: {e}")
        return

    if not markets:
        print("ERR No historical markets returned")
        return

    print(f"OK Fetched {len(markets)} historical markets")
    print()

    # Filter to markets with valid outcomes
    valid_markets = []
    for mkt in markets:
        result = mkt.get('result', 'unknown')
        price = mkt.get('last_price', 0.0)

        # Skip markets without outcomes
        if result not in ['yes', 'no']:
            continue

        # Skip markets with invalid prices (will be abstained anyway)
        if price <= 0.01 or price >= 0.99:
            continue

        valid_markets.append(mkt)

    print(f"[2/4] Filtered to {len(valid_markets)} markets with valid outcomes and prices")
    if len(valid_markets) == 0:
        print("ERR No valid markets to evaluate")
        return
    print()

    # Evaluation loop
    print(f"[3/4] Running evaluation pipeline on {len(valid_markets)} markets...")
    print()

    processed = 0
    abstained = 0
    failed = 0

    for i, mkt in enumerate(valid_markets, 1):
        market_id = mkt.get('market_id', f'unknown_{i}')
        title = mkt.get('title', 'Unknown')
        price = mkt.get('last_price', 0.0)
        result_ticker = mkt.get('result', 'unknown')
        close_time = mkt.get('close_time', 'unknown')

        # Progress indicator (every 20 markets)
        if i % 20 == 0 or i == len(valid_markets):
            print(f"  Processing market {i}/{len(valid_markets)}...")

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
            logger.warning(f"EventInput failed for {market_id}: {e}")
            failed += 1
            continue

        # Market sanity check
        valid, reason = _market_is_valid(event)
        if not valid:
            abstained += 1
            continue

        # Classify domain
        domain = classify_market(market_id, title)

        # Build formula
        try:
            formula = build_domain_formula(market_id, title, registry)
        except Exception as e:
            logger.warning(f"Formula build failed for {market_id}: {e}")
            failed += 1
            continue

        # Run tools
        try:
            tool_outputs, tool_statuses = run_tools(event, formula, registry)
        except Exception as e:
            logger.warning(f"Tool execution failed for {market_id}: {e}")
            failed += 1
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
            logger.warning(f"Scoring failed for {market_id}: {e}")
            failed += 1
            continue

        # Determine outcome
        outcome = 1.0 if result_ticker == 'yes' else 0.0

        # Determine decision
        if score.bet_triggered:
            if score.final_score >= THRESHOLD:
                decision = "YES"
            elif score.final_score <= -THRESHOLD:
                decision = "NO"
            else:
                decision = "PASS"
        else:
            decision = "PASS"

        # Log to evaluation table
        run_id = f"eval_{i:04d}"

        for tool_out, sel in zip(score.tool_outputs, formula.selections):
            vec = tool_out.output_vector
            tool_signal_mean = sum(vec) / len(vec) if vec else 0.0
            z_contribution = sel.weight * tool_signal_mean

            try:
                store.insert_evaluation(
                    run_id=run_id,
                    market_id=market_id,
                    domain=domain,
                    p_model=score.model_probability,
                    p_market=price,
                    edge=score.edge,
                    decision=decision,
                    outcome=outcome,
                    raw_score_z=score.raw_score_z,
                    tool_name=sel.tool_name,
                    tool_signal_mean=tool_signal_mean,
                    weight=sel.weight,
                    z_contribution=z_contribution,
                    market_timestamp=close_time,
                )
            except Exception as e:
                logger.error(f"SQLite insert failed for {market_id}: {e}")
                failed += 1
                break

        processed += 1

    print()
    print(f"Processed: {processed}  |  Abstained: {abstained}  |  Failed: {failed}")
    print()

    # Compute metrics
    print(f"[4/4] Computing evaluation metrics...")
    print()

    # Market-level metrics
    market_results = store.query("""
        SELECT DISTINCT
            run_id,
            p_model,
            p_market,
            edge,
            decision,
            outcome
        FROM historical_tool_evaluation
    """)

    if not market_results:
        print("ERR No evaluation results in database")
        return

    # Accuracy (bets only)
    bets = [r for r in market_results if r['decision'] != 'PASS']
    if bets:
        correct = sum(1 for r in bets if (r['decision'] == 'YES' and r['outcome'] == 1.0) or
                                          (r['decision'] == 'NO' and r['outcome'] == 0.0))
        accuracy = 100 * correct / len(bets)
    else:
        accuracy = float('nan')

    # Brier score (all markets with p_model)
    predictions = [(r['p_model'], r['outcome']) for r in market_results if r['p_model'] is not None]
    brier = compute_brier_score(predictions)

    # Trigger rate
    trigger_rate = 100 * len(bets) / len(market_results) if market_results else 0

    # Simulated ROI
    roi = 0.0
    for r in bets:
        if r['decision'] == 'YES':
            roi += BET_AMOUNT if r['outcome'] == 1.0 else -BET_AMOUNT
        elif r['decision'] == 'NO':
            roi += BET_AMOUNT if r['outcome'] == 0.0 else -BET_AMOUNT

    print("=" * 80)
    print("  MARKET-LEVEL METRICS")
    print("=" * 80)
    print(f"Total Markets       : {len(market_results)}")
    print(f"Bets Triggered      : {len(bets)} ({trigger_rate:.1f}%)")
    print(f"Accuracy            : {accuracy:.1f}% ({correct}/{len(bets)} correct)" if bets else "Accuracy            : N/A (no bets)")
    print(f"Brier Score         : {brier:.4f} (lower is better)" if not math.isnan(brier) else "Brier Score         : N/A")
    print(f"Simulated ROI       : ${roi:.2f} (${BET_AMOUNT}/bet)")
    print("=" * 80)
    print()

    # Tool-level metrics
    tool_data = store.query("""
        SELECT
            tool_name,
            tool_signal_mean,
            z_contribution,
            outcome
        FROM historical_tool_evaluation
        ORDER BY tool_name, run_id
    """)

    # Group by tool
    tools = {}
    for row in tool_data:
        tool_name = row['tool_name']
        if tool_name not in tools:
            tools[tool_name] = {
                'signals': [],
                'contributions': [],
                'outcomes': [],
            }
        tools[tool_name]['signals'].append(row['tool_signal_mean'])
        tools[tool_name]['contributions'].append(row['z_contribution'])
        tools[tool_name]['outcomes'].append(row['outcome'])

    # Compute tool metrics
    tool_metrics = []
    for tool_name, data in tools.items():
        signals = data['signals']
        contribs = data['contributions']
        outcomes = data['outcomes']

        mean_signal = sum(signals) / len(signals) if signals else 0.0
        mean_z = sum(contribs) / len(contribs) if contribs else 0.0

        # Correlation between signal and outcome
        corr_signal = compute_correlation(signals, outcomes)

        # Correlation between z_contribution and outcome (Information Coefficient)
        ic = compute_correlation(contribs, outcomes)

        # Win rate when signal > 0.5
        bullish = [(s, o) for s, o in zip(signals, outcomes) if s > 0.5]
        if bullish:
            wins = sum(1 for s, o in bullish if o == 1.0)
            win_rate = 100 * wins / len(bullish)
        else:
            win_rate = float('nan')

        tool_metrics.append({
            'tool_name': tool_name,
            'corr': corr_signal,
            'ic': ic,
            'mean_signal': mean_signal,
            'mean_z': mean_z,
            'win_rate': win_rate,
            'n_samples': len(signals),
        })

    # Sort by correlation (descending)
    tool_metrics.sort(key=lambda x: x['corr'] if not math.isnan(x['corr']) else -999, reverse=True)

    # Print tool ranking table
    print("=" * 80)
    print("  TOOL-LEVEL PREDICTIVE METRICS")
    print("=" * 80)
    print(f"{'Tool Name':<45} {'Corr':>7} {'IC':>7} {'Mean Signal':>12} {'Mean Z':>8}")
    print("-" * 80)

    for tm in tool_metrics:
        tool_display = tm['tool_name'][:44]  # Truncate long names
        corr_str = f"{tm['corr']:>7.3f}" if not math.isnan(tm['corr']) else "    N/A"
        ic_str = f"{tm['ic']:>7.3f}" if not math.isnan(tm['ic']) else "    N/A"
        mean_sig_str = f"{tm['mean_signal']:>12.4f}"
        mean_z_str = f"{tm['mean_z']:>8.4f}"

        print(f"{tool_display:<45} {corr_str} {ic_str} {mean_sig_str} {mean_z_str}")

    print("=" * 80)
    print()
    print("Legend:")
    print("  Corr        : Correlation(tool_signal, outcome) - higher is better")
    print("  IC          : Information Coefficient (Correlation(z_contrib, outcome))")
    print("  Mean Signal : Average tool output across all markets")
    print("  Mean Z      : Average z-space contribution (weight * signal)")
    print()
    print("Interpretation:")
    print("  Corr > 0.10  : Predictive tool (positive signal)")
    print("  Corr < -0.10 : Negative signal (may need inversion)")
    print("  |Corr| < 0.05: No meaningful predictive power")
    print()
    print("=" * 80)
    print(f"Evaluation results saved to SQLite: {store._path}")
    print(f"Table: historical_tool_evaluation ({len(tool_data)} records)")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Review tool rankings - remove tools with Corr near 0")
    print("  2. Investigate negative-correlation tools for potential bugs")
    print("  3. Design new tools targeting weak domains")
    print("  4. Consider weight optimization based on IC")
    print("=" * 80)


if __name__ == "__main__":
    main()
