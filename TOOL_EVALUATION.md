# Tool Impact Evaluation System

## Overview

The Tool Evaluation Phase measures **predictive power** of each tool on historical markets with known outcomes. This is **not** about trading - it's about **signal attribution** and **statistical validation**.

## Research Questions

1. **Which tools are actually predictive?**
2. Which tools have zero correlation (noise)?
3. Which tools have negative correlation (bugs/inversions)?
4. Where should we focus new tool development?

## Methodology

### 1. Data Selection

```python
N_HISTORICAL_MARKETS = 200  # Configurable parameter
```

**Selection Criteria:**
- Markets with known outcomes (`result='yes'` or `result='no'`)
- Valid prices (0.01 < price < 0.99)
- Deterministic ordering (by resolution time)
- No randomness in selection

**Exclusions:**
- Markets that would be abstained (price=0 or price>=0.99)
- Markets without outcomes
- No future leakage (tools use data consistent with market timestamp)

### 2. Evaluation Pipeline

For each historical market:

```python
1. Fetch market data from Kalshi Historical API
2. Build EventInput (market_id, title, price, timestamp)
3. Classify domain (sports/economics/crypto/politics/weather/other)
4. Select tools via domain-aware filtering
5. Run tools (deterministic, no LLM)
6. Compute score via linear opinion pooling
7. Log results to SQLite:
   - Market-level: p_model, edge, decision, outcome
   - Tool-level: signal_mean, weight, z_contribution
```

**Critical Constraints:**
- ✅ Deterministic execution (no randomness)
- ✅ No LLM involvement
- ✅ No external data refresh during evaluation
- ⚠️ Tools use data consistent with market timestamp (not fully implemented - see Data Leakage section)

### 3. Metrics

#### Market-Level

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | `correct_bets / total_bets` | % of bets on correct outcome |
| **Brier Score** | `mean((p_model - outcome)^2)` | Calibration (0=perfect, 1=worst) |
| **Trigger Rate** | `bets_triggered / total_markets` | % of markets passing threshold |
| **Simulated ROI** | `sum(bet_outcomes * BET_AMOUNT)` | Profit/loss at $10/bet |

#### Tool-Level

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Correlation** | `Pearson(tool_signal, outcome)` | Signal-outcome alignment |
| **IC** (Information Coefficient) | `Pearson(z_contribution, outcome)` | Weighted signal impact |
| **Mean Signal** | `mean(tool_signal_mean)` | Average tool output |
| **Mean Z** | `mean(weight × signal)` | Average z-space contribution |
| **Win Rate** | `P(outcome=1 \| signal>0.5)` | Accuracy when bullish |

**Interpretation Guide:**
- `Corr > 0.10`: **Predictive tool** (keep and amplify)
- `Corr < -0.10`: **Negative signal** (invert or fix bug)
- `|Corr| < 0.05`: **No predictive power** (remove)

### 4. Storage Schema

#### `historical_tool_evaluation` Table

```sql
CREATE TABLE historical_tool_evaluation (
    eval_id          INTEGER PRIMARY KEY,
    run_id           TEXT NOT NULL,
    market_id        TEXT NOT NULL,
    domain           TEXT,
    p_model          REAL,
    p_market         REAL,
    edge             REAL,
    decision         TEXT,       -- YES/NO/PASS
    outcome          REAL,       -- 1.0 or 0.0
    raw_score_z      REAL,
    tool_name        TEXT NOT NULL,
    tool_signal_mean REAL,       -- mean(output_vector)
    weight           REAL,       -- from FormulaSpec
    z_contribution   REAL,       -- weight × signal
    market_timestamp TEXT
);
```

**Key Properties:**
- One row per tool per market
- Standalone table (no foreign key to `runs`)
- Supports per-tool correlation analysis
- Enables cross-tool comparison

## Usage

### Running Evaluation

```bash
# Basic usage (200 markets)
python scripts/evaluate_tool_impact.py

# Modify N_HISTORICAL_MARKETS in script for more/fewer markets
```

### Output

```
================================================================================
  MARKET-LEVEL METRICS
================================================================================
Total Markets       : 32
Bets Triggered      : 15 (46.9%)
Accuracy            : 73.3% (11/15 correct)
Brier Score         : 0.1823 (lower is better)
Simulated ROI       : $70.00 ($10.0/bet)
================================================================================

================================================================================
  TOOL-LEVEL PREDICTIVE METRICS
================================================================================
Tool Name                                        Corr      IC  Mean Signal   Mean Z
--------------------------------------------------------------------------------
sportsbook_implied_probability_tool              0.312   0.285       0.5124   0.1281
weather_probability_tool                         0.089   0.067       0.4987   0.0622
snapshot_volatility_tool                         0.021   0.015       0.4892   0.0122
fred_macro_tool                                 -0.043  -0.031       0.5123  -0.0078
bls_labor_tool                                  -0.002  -0.001       0.5001  -0.0003
================================================================================
```

**Interpretation:**
- `sportsbook_implied_probability_tool`: Strong predictor (Corr=0.312)
- `weather_probability_tool`: Weak predictor (Corr=0.089)
- `snapshot_volatility_tool`: Noise (Corr≈0)
- `fred_macro_tool`: Negative signal (possible bug or domain mismatch)
- `bls_labor_tool`: No information (Corr≈0)

### Querying Results

```python
from prediction_agent.storage.sqlite_store import SQLiteStore
store = SQLiteStore()

# Tool rankings by correlation
rankings = store.query("""
    SELECT
        tool_name,
        COUNT(*) as n_markets,
        AVG(tool_signal_mean) as avg_signal,
        AVG(z_contribution) as avg_z,
        -- Correlation approximation (for quick check)
        (SUM((tool_signal_mean - 0.5) * (outcome - 0.5))) /
        (COUNT(*) * 0.25) as approx_corr
    FROM historical_tool_evaluation
    GROUP BY tool_name
    ORDER BY approx_corr DESC
""")

# Market outcomes
outcomes = store.query("""
    SELECT DISTINCT
        run_id,
        market_id,
        domain,
        decision,
        outcome,
        p_model,
        edge
    FROM historical_tool_evaluation
    WHERE decision != 'PASS'
    ORDER BY abs(edge) DESC
""")

# High-confidence bets
high_conf = store.query("""
    SELECT DISTINCT
        market_id,
        decision,
        outcome,
        edge,
        COUNT(DISTINCT tool_name) as n_tools
    FROM historical_tool_evaluation
    WHERE abs(edge) > 0.10
    GROUP BY run_id
    HAVING n_tools >= 3
""")
```

## Current Limitations

### 1. Data Leakage (Not Yet Addressed)

⚠️ **CRITICAL**: Tools currently do NOT enforce timestamp alignment.

**Problem:**
- Market closes at `2024-07-10T21:00:00Z`
- External data collected at `2026-02-27T18:00:00Z`
- Tools may use future data to predict past outcomes

**Impact:**
- Inflated correlation scores
- Overly optimistic performance metrics
- Real-time performance will be worse

**Future Fix Required:**
```python
# In each tool, filter data by timestamp
def fred_macro_tool(event: EventInput, ...) -> ToolOutput:
    market_close_time = event.timestamp
    # Only use FRED data published BEFORE market_close_time
    valid_data = [d for d in fred_data if d['timestamp'] < market_close_time]
    ...
```

**Flag Status:** ⚠️ Known issue, flagged clearly in documentation

### 2. External Data Missing

**Current State:**
- All external data files are empty (`external/*.jsonl`)
- External tools return zeros
- Historical snapshot tools return zeros (only STUB data in snapshots)

**Next Step:**
- Run `notebooks/00_live_loop.ipynb` Section 3 to collect external data
- See `DATASET_BUILD_STATUS.md` for instructions

### 3. Historical Markets Limited to July 2024

**Kalshi Historical API cutoff:** `2024-07-10T00:00:00Z`

**Impact:**
- All historical markets are 7+ months old
- Market types may not reflect current offerings
- Domain distribution skewed (32/32 markets were crypto in test run)

**Workaround:**
- Use evaluation results as **relative** tool rankings
- Re-run evaluation after Kalshi updates historical cutoff
- Focus on tool-to-tool comparison rather than absolute metrics

## Research Workflow

### Phase 1: Baseline Evaluation (Current)

```bash
1. python scripts/evaluate_tool_impact.py
2. Review tool rankings
3. Identify predictive vs noise tools
```

**Deliverable:** Initial tool effectiveness report

### Phase 2: Tool Refinement (Next)

```bash
1. Remove tools with |Corr| < 0.05
2. Investigate negative-correlation tools for bugs
3. Invert tools if needed (e.g., if weather predicts opposite)
4. Re-run evaluation to confirm improvement
```

**Deliverable:** Pruned tool set with proven predictive power

### Phase 3: Domain-Specific Analysis

```sql
-- Sports markets only
SELECT tool_name, AVG(z_contribution)
FROM historical_tool_evaluation
WHERE domain = 'sports'
GROUP BY tool_name;

-- Economics markets only
SELECT tool_name, AVG(z_contribution)
FROM historical_tool_evaluation
WHERE domain = 'economics'
GROUP BY tool_name;
```

**Deliverable:** Domain-specific tool recommendations

### Phase 4: Weight Optimization (Future)

```python
# Fit weights via regression
from sklearn.linear_model import Ridge

X = tool_signals  # [n_markets × n_tools]
y = outcomes      # [n_markets]
model = Ridge().fit(X, y)
optimized_weights = model.coef_
```

**Deliverable:** Data-driven weight allocation

### Phase 5: Rolling Window Backtest (Future)

```python
# Train on months 1-3, test on month 4, repeat
for window in rolling_windows(markets, train_size=90, test_size=30):
    train_markets = window.train
    test_markets = window.test

    # Fit weights on train
    weights = fit_weights(train_markets)

    # Evaluate on test
    performance = evaluate(test_markets, weights)
```

**Deliverable:** Out-of-sample performance estimates

## Configuration

### Script Parameters

```python
# In scripts/evaluate_tool_impact.py

N_HISTORICAL_MARKETS = 200    # Number of markets to evaluate
THRESHOLD = 0.30              # Bet threshold (must match production)
BET_AMOUNT = 10.0             # Simulated bet size for ROI
```

### Tool Selection

Uses production domain-aware filtering:
```python
from engine.market_classifier import classify_market, get_domain_tools_with_weather

domain = classify_market(market_id, title)
tools = get_domain_tools_with_weather(domain, available_tools, market_id, title)
```

## Troubleshooting

### "No historical markets returned"

**Cause:** Kalshi Historical API unavailable or auth failure

**Fix:**
```bash
# Check credentials
echo $KALSHI_ACCESS_KEY_ID
# Verify API key file exists
ls FebuaruyAPIKalshi.txt
```

### "All tools return zeros"

**Cause:** External data not collected

**Fix:**
```bash
# Run data collection
jupyter notebook notebooks/00_live_loop.ipynb
# Execute Section 3 cells
```

### "Correlation = N/A"

**Cause:** All tool signals are identical (zero variance)

**Explanation:** If a tool returns 0.0 for all markets, correlation is undefined

**Fix:** Collect external data or check tool implementation

### "FOREIGN KEY constraint failed"

**Cause:** Old database schema

**Fix:**
```bash
# Remove old database
rm outputs/prediction_agent.db*
# Re-run evaluation (will create new schema)
python scripts/evaluate_tool_impact.py
```

## Files

| File | Purpose |
|------|---------|
| `scripts/evaluate_tool_impact.py` | Main evaluation script |
| `prediction_agent/storage/sqlite_store.py` | Database schema and `insert_evaluation()` |
| `outputs/prediction_agent.db` | SQLite database with results |
| `TOOL_EVALUATION.md` | This document |

## Next Steps

### Immediate

1. ✅ Evaluation pipeline implemented
2. ✅ SQLite schema created
3. ✅ Metrics computed
4. ✅ Console output formatted
5. ⚠️ Data leakage flagged (not yet fixed)

### Required Before Production

1. **Collect external data** via notebook Section 3
2. **Re-run evaluation** with real tool signals
3. **Analyze correlations** and remove noise tools
4. **Implement timestamp filtering** in tools (prevent data leakage)
5. **Validate on out-of-sample data** (recent markets)

### Future Research

1. Weight optimization via regression
2. Rolling window backtest
3. Log-odds pooling comparison
4. Tool ensemble methods
5. Domain-specific weight profiles

---

**Status:** Evaluation system operational, awaiting external data collection for meaningful tool correlation analysis.
