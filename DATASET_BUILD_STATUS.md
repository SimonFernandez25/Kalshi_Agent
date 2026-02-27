# Dataset Build Status

## Summary

Created three dataset build scripts to process markets and identify failure modes:

1. **`scripts/build_initial_dataset.py`** - Full pipeline with historical snapshot tools
2. **`scripts/build_initial_dataset_external.py`** - External tools only (no snapshot dependency)
3. **`scripts/build_dataset_from_historical.py`** - ✅ **Uses Kalshi Historical API** (settled markets with outcomes)

## Latest Update: Historical API Working! 🎉

**Successfully implemented Kalshi Historical API support:**
- Added `get_historical_cutoff()` and `get_historical_markets()` to `KalshiClient`
- Fetched 50 settled markets from July 2024 with known outcomes
- System correctly processed 11/50 markets (39 abstained on price filters)
- Historical cutoff timestamp: `2024-07-10T00:00:00Z`

## Current Status

**What's Working ✓**
- Kalshi Historical API integration functional
- Market fetching, processing, and SQLite logging all work
- Sanity checks correctly filter 39/50 markets (price=0 or price>=0.99)
- Known outcomes available (`result='yes'` or `result='no'`)

**Current Limitation ⚠️**
- **All tools return zeros** (no data collected yet):
  - External tools need FRED/BLS/Weather/Odds data
  - Historical tools need market snapshot data
  - 0 bets triggered out of 11 processed markets

## Findings

### What Works ✓
- SQLite logging successfully stores runs with tool outputs
- Market sanity checks correctly filter zero-price markets (abstained 25/25)
- Domain classification working (identified sports/economics/other)
- Tool execution pipeline runs without errors
- External tools script properly filters by tool type

### What Doesn't Work Yet ✗
- **No historical snapshot data** for real Kalshi markets
  - `outputs/market_snapshots.jsonl` contains only STUB markets from mock mode
  - Historical tools (snapshot_volatility, spread_compression, price_jump_detector) return zeros
  - Real markets fetched from API have never been collected by live loop

- **Kalshi API returning only zero-price markets**
  - All 100 fetched markets have `last_price=0.000`
  - Cannot test full pipeline on liquid markets
  - Need real market data with price > 0.01

- **Outcome data not available**
  - API returns `result='unknown'` for all markets
  - Cannot validate predictions against known outcomes yet

## Critical Next Step: Collect External Data

**You must collect external data before the system can generate meaningful signals.**

### Step 1: Collect External Data (REQUIRED)

Open `notebooks/00_live_loop.ipynb` and run **Section 3** to populate external data:

```python
# Section 3: External Data Collection
from prediction_agent.collector.external.fred_collector import collect_fred_data
from prediction_agent.collector.external.bls_collector import collect_bls_data
from prediction_agent.collector.external.weather_collector import collect_weather_data
from prediction_agent.collector.external.odds_collector import collect_odds_data

# Run all collectors
collect_fred_data()    # Macro indicators (GDP, inflation, unemployment)
collect_bls_data()     # Labor statistics and CPI
collect_weather_data() # NBA arena city forecasts
collect_odds_data()    # Sportsbook lines for NBA/NFL
```

**Expected output:**
- `external/fred_snapshots.jsonl` - FRED economic indicators
- `external/bls_snapshots.jsonl` - BLS labor/CPI data
- `external/weather_snapshots.jsonl` - Weather forecasts for NBA cities
- `external/odds_snapshots.jsonl` - Sportsbook implied probabilities

This takes ~2-5 minutes to collect fresh data from all APIs.

### Step 2: Re-run Historical Dataset Build

Once external data is collected, run:

```bash
python scripts/build_dataset_from_historical.py
```

**Expected results:**
- Tools will return non-zero signals (using external data)
- Some markets will trigger bets (threshold=0.30)
- Can evaluate prediction accuracy against known outcomes

### Step 3: Analyze Performance

```python
from prediction_agent.storage.sqlite_store import SQLiteStore
store = SQLiteStore()

# Check tool contributions
tools = store.query("""
    SELECT tool_id,
           AVG(contribution) as avg_contrib,
           COUNT(*) as n_runs
    FROM tool_outputs
    GROUP BY tool_id
    ORDER BY avg_contrib DESC
""")

# Check bet accuracy
bets = store.query("""
    SELECT
        COUNT(*) as total_bets,
        SUM(CASE WHEN (final_score > 0 AND outcome = 1.0) OR
                      (final_score < 0 AND outcome = 0.0)
                 THEN 1 ELSE 0 END) as correct_bets
    FROM runs
    WHERE bet_triggered = 1 AND outcome IS NOT NULL
""")
```

### Optional: Collect Live Snapshot Data

For historical tools (snapshot_volatility, spread_compression, price_jump_detector):

```bash
# Run live loop for several hours to build price history
python -m prediction_agent.live.run_live_loop --poll-interval 300 --max-markets 100
```

This populates `outputs/market_snapshots.jsonl` with real Kalshi price data over time.

## Configuration

Current dataset build settings:
- **Threshold**: 0.30 (low, to trigger more bets)
- **Bet amount**: $10 per market
- **Max markets**: 25
- **Market status**: "open" (active markets)

These can be adjusted in the script configuration section.

## File Locations

### Scripts
- `scripts/build_dataset_from_historical.py` - ✅ **RECOMMENDED** Uses Kalshi Historical API
- `scripts/build_initial_dataset.py` - Full pipeline with all tools (needs snapshot data)
- `scripts/build_initial_dataset_external.py` - External tools only

### Data Files
- `outputs/prediction_agent.db` - SQLite database with runs and tool outputs
- `outputs/market_snapshots.jsonl` - Kalshi price history (612 STUB entries currently)
- `external/fred_snapshots.jsonl` - ⚠️ **MISSING** - FRED macro data
- `external/bls_snapshots.jsonl` - ⚠️ **MISSING** - BLS labor/CPI data
- `external/weather_snapshots.jsonl` - ⚠️ **MISSING** - Weather forecasts
- `external/odds_snapshots.jsonl` - ⚠️ **MISSING** - Sportsbook odds

### Notebooks
- `notebooks/00_live_loop.ipynb` - **Section 3: Run this first to collect external data**
- `notebooks/03_backtest_eval.ipynb` - Backtest analysis

### API Clients
- `api/kalshi_client.py` - Kalshi API client with historical data support
  - `get_historical_cutoff()` - Get current historical data cutoff timestamps
  - `get_historical_markets()` - Fetch settled markets with known outcomes
  - Working as of 2026-02-27 (cutoff: 2024-07-10)

## Known Issues

1. **market_snapshots.jsonl contains only STUB data**
   - Need to run live loop to populate with real Kalshi data
   - Current 612 lines are from mock mode testing

2. **Kalshi API returns only zero-price markets**
   - May be timing issue (low liquidity period)
   - May need different API query parameters
   - Check API status: https://api.elections.kalshi.com/trade-api/v2/markets

3. **No outcome data for validation**
   - API `result` field returns 'unknown'
   - Need to track markets over time and update outcomes manually
   - Or query historical closed markets with known results

## Success Criteria

Dataset build will be successful when:
- [x] Kalshi Historical API integration working
- [x] Fetches settled markets with known outcomes
- [x] SQLite logging functional
- [x] Market sanity checks filter correctly
- [ ] **External data collected** ← **YOU ARE HERE**
- [ ] At least 5/50 historical markets trigger bets
- [ ] Tool outputs are non-zero (using external data)
- [ ] Prediction accuracy > 50% (better than random)

## Git Status

Latest commits:
```bash
4a5677c Add Kalshi Historical API support and build_dataset_from_historical.py
7d71f52 Add DATASET_BUILD_STATUS.md documenting current state and blockers
b4fbed9 Add build_initial_dataset_external.py for external tools only
d53716b Fix SQLiteStore._path attribute reference in build_initial_dataset
b4e0b8d Fix SQLiteStore.insert_run() call in build_initial_dataset.py
```

Branch: `main` (consolidated, adoring-tharp branch merged and deleted)

---

## Quick Start Guide

**To build your first dataset with known outcomes:**

1. Open `notebooks/00_live_loop.ipynb`
2. Run **Section 3** to collect external data (~5 minutes)
3. Run: `python scripts/build_dataset_from_historical.py`
4. Check results in `outputs/prediction_agent.db`
5. Analyze: `SELECT * FROM runs WHERE bet_triggered=1`

That's it! The historical markets have known outcomes so you can immediately see accuracy.
