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

# Try loading .env file manually (if present, to avoid requiring python-dotenv)
_env_file = PREDICTION_AGENT_DIR / ".env"
try:
    if not _env_file.exists():
         _env_file = PROJECT_ROOT / ".env"

    if _env_file.exists():
        import os
        with open(_env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    # Remove quotes if present
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        v = v[1:-1]
                    # Only set if not already set (env vars take precedence)
                    if k not in os.environ:
                        os.environ[k] = v
except Exception:
    pass  # Best effort loader


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
# Signal Integrity
# ──────────────────────────────────────────────
ENABLE_MOCK_TOOLS = False             # Disable noise tools (MockRandomContext etc.) in production
ABSTAIN_ON_ZERO_PRICE = True          # Abstain from scoring if market price <= MIN_VALID_PRICE
MIN_VALID_PRICE = 0.01                # Prices at or below this are considered invalid

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

# ──────────────────────────────────────────────
# Evolution (Recursive Tool Growth)
# ──────────────────────────────────────────────
ENABLE_EVOLUTION = False                          # Master switch — off by default
EVOLUTION_GAP_THRESHOLD = 0.5                     # Minimum priority_score to trigger proposal
EVOLUTION_MAX_TOOLS_PER_CYCLE = 1                 # Cap new tools per evolution cycle
EVOLUTION_DEPRECATION_RUNS = 10                   # Consecutive underperforming runs before deprecation
EXECUTION_LOG_FILE = OUTPUTS_DIR / "execution_logs.jsonl"
TOOL_LIFECYCLE_FILE = OUTPUTS_DIR / "tool_lifecycle.jsonl"
GENERATED_TOOLS_DIR = PREDICTION_AGENT_DIR / "tools" / "generated"
TOOL_VERIFY_TIMEOUT_SEC = 10
TOOL_VERIFY_DETERMINISM_RUNS = 3

# Evolution Oversight
LIVE_MODE = True                                  # Enforce live API calls, warn on stale data
FORCE_REFRESH_MARKETS = True                      # Skip Kalshi API cache
EVOLUTION_REQUIRE_MANUAL_APPROVAL = True          # If True, tools go to pending/ and require manual promotion

# ──────────────────────────────────────────────
# Sandbox (subprocess tool execution)
# ──────────────────────────────────────────────
SANDBOX_MAX_MEMORY_MB = 256                       # RSS memory limit inside sandbox child (POSIX only)
SANDBOX_MAX_CPU_SEC   = 8                         # CPU-time limit inside sandbox child (POSIX only)

# ──────────────────────────────────────────────
# Correlation Pruning
# ──────────────────────────────────────────────
CORRELATION_REJECTION_THRESHOLD = 0.95            # Reject new tool if |Pearson r| >= this with any existing tool
CORRELATION_N_SAMPLES           = 50              # Number of recent snapshots to use for pruning check
ENABLE_CORRELATION_PRUNING      = True            # Master switch for correlation-based tool rejection

# ──────────────────────────────────────────────
# SQLite Storage
# ──────────────────────────────────────────────
SQLITE_DB_FILE = OUTPUTS_DIR / "prediction_agent.db"
SQLITE_ENABLED = True                             # If False, SQLite writes are skipped (JSONL-only mode)

# ──────────────────────────────────────────────
# Backtesting
# ──────────────────────────────────────────────
BACKTEST_RESULTS_FILE     = OUTPUTS_DIR / "backtest_results.jsonl"
BACKTEST_DEFAULT_THRESHOLD = 0.55
BACKTEST_DEFAULT_PRESET    = "equal_weight"       # "equal_weight" | "volatility_only" | "real_tools_only"

# ──────────────────────────────────────────────
# External Data Collectors (Layer B)
# ──────────────────────────────────────────────
EXTERNAL_DIR = OUTPUTS_DIR / "external"
EXTERNAL_DIR.mkdir(exist_ok=True)

FRED_SNAPSHOT_FILE    = EXTERNAL_DIR / "fred_snapshots.jsonl"
BLS_SNAPSHOT_FILE     = EXTERNAL_DIR / "bls_snapshots.jsonl"
WEATHER_SNAPSHOT_FILE = EXTERNAL_DIR / "weather_snapshots.jsonl"
ODDS_SNAPSHOT_FILE    = EXTERNAL_DIR / "odds_snapshots.jsonl"

# API key files
APIS_DIR              = PREDICTION_AGENT_DIR / "APIs"
FRED_API_KEY_FILE     = APIS_DIR / "FRED.txt"
BLS_API_KEY_FILE      = APIS_DIR / "USBLabor.txt"
WEATHER_API_KEY_FILE  = APIS_DIR / "WeatherAPI.txt"
ODDS_API_KEY_FILE     = APIS_DIR / "TheOddsAPI.txt"

# ──────────────────────────────────────────────
# Scorer
# ──────────────────────────────────────────────
# "signal_sum"      : original weighted-sum dot product (default)
# "probability_edge": edge = model_probability - current_market_price
#                     trigger when abs(edge) > threshold
SCORING_MODE = "signal_sum"

# Tools that carry a direct probability signal in output_vector[0]
# These are used in probability_edge mode.
PROBABILITY_SIGNAL_TOOLS = [
    "sportsbook_implied_probability_tool",
    "weather_probability_tool",
]

# Logistic calibration layer (probability_edge mode only)
# When enabled: z = weighted_sum → p_model = sigmoid(z)
# When disabled: z = weighted_sum → p_model = clamp(z, 0, 1)  [DEFAULT: linear opinion pool]
#
# NOTE: Since z is already a convex combination of probabilities (z ∈ [0,1]),
# applying sigmoid introduces structural bias (sigmoid(z) ∈ [0.5, 0.73] for z ∈ [0,1]).
# We use linear opinion pooling (p_model = Σ w_i p_i) as the mathematically correct default.
# Sigmoid is reserved for future log-odds pooling modes.
USE_LOGISTIC_CALIBRATION = False

