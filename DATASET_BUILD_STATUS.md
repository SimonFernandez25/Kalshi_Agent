# Dataset Build Status

## Summary

Created two dataset build scripts to process 25 markets and identify failure modes:

1. **`scripts/build_initial_dataset.py`** - Full pipeline with historical snapshot tools
2. **`scripts/build_initial_dataset_external.py`** - External tools only (no snapshot dependency)

## Current Blocker

**All Kalshi API markets have price=0.000** (illiquid/stub markets)
- Fetched 100 markets with `status="open"`, all have `last_price=0.000`
- System correctly abstains on these (price <= MIN_VALID_PRICE=0.01)
- Cannot generate meaningful signals without liquid markets

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

## Next Steps

### Immediate (Required before dataset build)

1. **Run live loop to collect snapshot data**
   ```bash
   # From notebook `00_live_loop.ipynb` Section 4
   # Or from CLI:
   python -m prediction_agent.live.run_live_loop --poll-interval 300 --max-markets 100
   ```
   - Run for several hours/days to build up price history
   - Creates snapshots in `outputs/market_snapshots.jsonl`
   - Enables historical tools to produce meaningful signals

2. **Collect external data**
   ```python
   # From notebook `00_live_loop.ipynb` Section 3
   from prediction_agent.collector.external.fred_collector import collect_fred_data
   from prediction_agent.collector.external.bls_collector import collect_bls_data
   from prediction_agent.collector.external.weather_collector import collect_weather_data
   from prediction_agent.collector.external.odds_collector import collect_odds_data

   collect_fred_data()    # Macro indicators (GDP, inflation, etc.)
   collect_bls_data()     # Labor/CPI data
   collect_weather_data() # NBA arena city forecasts
   collect_odds_data()    # Sportsbook lines (NBA/NFL)
   ```
   - Populates `external/*.jsonl` files
   - Enables external tools to provide signals

3. **Wait for liquid Kalshi markets**
   - Current API returns only price=0.000 markets
   - Check periodically for markets with `last_price > 0.01`
   - Or use historical/closed markets if API provides them

### After Data Collection

4. **Re-run dataset build scripts**
   ```bash
   # Option A: Full pipeline (requires snapshot history)
   python scripts/build_initial_dataset.py

   # Option B: External tools only
   python scripts/build_initial_dataset_external.py
   ```

5. **Analyze results from SQLite**
   ```python
   from prediction_agent.storage.sqlite_store import SQLiteStore
   store = SQLiteStore()

   # Query runs
   runs = store.query("SELECT * FROM runs WHERE bet_triggered=1")

   # Query tool performance
   tools = store.query("""
       SELECT tool_id, AVG(contribution) as avg_contrib, COUNT(*) as n_runs
       FROM tool_outputs
       GROUP BY tool_id
       ORDER BY avg_contrib DESC
   """)
   ```

6. **Backtest evaluation**
   - Use `notebooks/03_backtest_eval.ipynb` to evaluate performance
   - Compare predicted bets vs actual outcomes (when available)
   - Calculate Brier score, calibration, profit/loss

## Configuration

Current dataset build settings:
- **Threshold**: 0.30 (low, to trigger more bets)
- **Bet amount**: $10 per market
- **Max markets**: 25
- **Market status**: "open" (active markets)

These can be adjusted in the script configuration section.

## File Locations

### Scripts
- `scripts/build_initial_dataset.py` - Full pipeline with all tools
- `scripts/build_initial_dataset_external.py` - External tools only

### Data
- `outputs/market_snapshots.jsonl` - Historical Kalshi price data (STUB data only currently)
- `outputs/prediction_agent.db` - SQLite database with runs and tool outputs
- `external/fred_snapshots.jsonl` - FRED macro data
- `external/bls_snapshots.jsonl` - BLS labor/CPI data
- `external/weather_snapshots.jsonl` - Weather forecasts
- `external/odds_snapshots.jsonl` - Sportsbook odds

### Notebooks
- `notebooks/00_live_loop.ipynb` - Live data collection control panel
- `notebooks/03_backtest_eval.ipynb` - Backtest analysis

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
- [ ] Live loop has collected 1000+ market snapshots over 24+ hours
- [ ] External data collectors have recent (<1 hour old) data
- [ ] Kalshi API returns markets with price > 0.01
- [ ] At least 10/25 markets trigger bets (with threshold=0.30)
- [ ] SQLite contains runs with non-zero tool outputs
- [ ] At least 2 different domains represented (sports, economics, etc.)

## Git Status

Latest commits:
```bash
b4fbed9 Add build_initial_dataset_external.py for external tools only
d53716b Fix SQLiteStore._path attribute reference in build_initial_dataset
b4e0b8d Fix SQLiteStore.insert_run() call in build_initial_dataset.py
31673fe Add abstain check to live loop _process_market
d236b55 Update live loop notebook with domain-aware pipeline and external data collection
```

Branch: `main` (consolidated, adoring-tharp branch merged and deleted)
