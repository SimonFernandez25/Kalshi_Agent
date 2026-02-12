"""
Central configuration for the prediction agent framework.
All tunables live here — no magic numbers elsewhere.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTION_AGENT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PREDICTION_AGENT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Kalshi API key file — expected in project root
API_KEY_FILE = PREDICTION_AGENT_DIR / "FebuaruyAPIKalshi.txt"

# ──────────────────────────────────────────────
# Kalshi API
# ──────────────────────────────────────────────
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_BASKETBALL_SERIES = "NBA"  # Series ticker prefix for basketball
KALSHI_TOP_N = 10  # Only pull top 10 markets

# ──────────────────────────────────────────────
# Agent / LLM
# ──────────────────────────────────────────────
LLM_MODEL = "claude-sonnet-4-5-20250929"  # Default model for agent
LLM_TEMPERATURE = 0.0  # Deterministic output

# ──────────────────────────────────────────────
# Watcher
# ──────────────────────────────────────────────
WATCHER_POLL_INTERVAL_SEC = 30  # Seconds between price polls
WATCHER_TIMEOUT_SEC = 300  # Max seconds to watch before giving up

# ──────────────────────────────────────────────
# Paper Broker
# ──────────────────────────────────────────────
DEFAULT_BET_AMOUNT = 10.0  # Paper dollars
PAPER_BET_LOG = OUTPUTS_DIR / "paper_bets.jsonl"
RUN_LOG = OUTPUTS_DIR / "run_log.jsonl"

# ──────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────
AVAILABLE_TOOLS = [
    "mock_price_signal",
    "mock_random_context",
    "snapshot_volatility_tool",
    "spread_compression_tool",
    "price_jump_detector_tool",
    "liquidity_spike_tool",
]
