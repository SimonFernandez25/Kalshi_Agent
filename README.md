# Prediction Agent Framework

Minimal deterministic LangGraph agent for tool-weighted prediction experiments on Kalshi basketball markets.

## Architecture

```
prediction_agent/
  schemas.py          # Pydantic models (EventInput, FormulaSpec, ScoreResult, PaperBet)
  config.py           # Central configuration
  api/
    kalshi_client.py  # ONLY live data source — Kalshi basketball markets
  tools/
    base_tool.py      # Abstract BaseTool interface
    registry.py       # Tool registry (agent can only pick registered tools)
    mock_tools.py     # Two mock deterministic tools for plumbing tests
  agent/
    graph.py          # LangGraph definition (select_tools → validate_spec → END)
    nodes.py          # Node functions + deterministic fallback
    prompts.py        # System/user prompts for tool selection
  engine/
    tool_runner.py    # Executes selected tools through registry
    scorer.py         # Weighted sum dot product
    watcher.py        # Polls Kalshi price until threshold hit or timeout
    paper_broker.py   # Logs simulated bets to JSONL
  outputs/            # JSONL logs (paper_bets.jsonl, run_log.jsonl)
  main.py             # Pipeline orchestrator
```

## Flow

1. Pull top 10 basketball markets from Kalshi
2. Pick one as EventInput
3. LangGraph agent selects mock tools + weights + threshold → FormulaSpec
4. Deterministic engine runs tools, computes weighted score
5. Watcher loop polls Kalshi odds for the same market
6. Paper broker logs simulated bet to JSONL

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Place your Kalshi API key in the project root
echo "your-api-key" > FebuaruyAPIKalshi.txt

# Run the pipeline (uses stubs if no API key)
python -m prediction_agent.main --skip-watcher

# Full run with watcher
python -m prediction_agent.main --watcher-ticks 3
```

## Strict Rules

- **ONE live data source**: Kalshi API only
- **ONE market type**: Basketball (NBA)
- **TOP 10 lines** only
- **NO** dataset scraping, historical data, ESPN, Twitter, news APIs, web browsing
- LLM selects tools but **never computes scores**
- Engine is **fully deterministic** (same input → same output)
