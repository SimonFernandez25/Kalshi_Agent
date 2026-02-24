"""
Status CLI - system health and pipeline telemetry.

All data derived from SQLite (primary) with JSONL fallback.
No live API calls.

Usage:
    python -m prediction_agent.live.status
    python -m prediction_agent.live.status --json
    python -m prediction_agent.live.status --watch 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (
    OUTPUTS_DIR,
    SQLITE_DB_FILE,
    TOOL_LIFECYCLE_FILE,
    ENABLE_EVOLUTION,
)


# -- Data collection ------------------------------------------------------------

def _collect_status() -> dict:
    """
    Gather all status metrics from SQLite + JSONL files.
    Returns a flat dict of metrics.
    """
    metrics = {}

    # -- SQLite metrics ---------------------------------------------------------
    db_exists = SQLITE_DB_FILE.exists()
    metrics["sqlite_available"]  = db_exists
    metrics["sqlite_size_kb"]    = round(SQLITE_DB_FILE.stat().st_size / 1024, 1) if db_exists else 0

    if db_exists:
        try:
            from prediction_agent.storage.sqlite_store import SQLiteStore
            store = SQLiteStore()

            r = store.query("SELECT COUNT(*) AS cnt FROM runs")
            metrics["total_runs"] = r[0]["cnt"] if r else 0

            r = store.query("SELECT COUNT(*) AS cnt FROM runs WHERE bet_triggered=1")
            metrics["bets_triggered"] = r[0]["cnt"] if r else 0

            r = store.query("SELECT AVG(score) AS avg FROM runs")
            avg_score = r[0]["avg"] if r else None
            metrics["avg_score"] = round(avg_score, 4) if avg_score is not None else None

            r = store.query("SELECT MAX(timestamp) AS ts FROM runs")
            metrics["last_run_timestamp"] = r[0]["ts"] if r else None

            r = store.query(
                "SELECT COUNT(*) AS cnt FROM runs WHERE timestamp >= "
                "datetime('now', '-1 hour')"
            )
            metrics["runs_last_1h"] = r[0]["cnt"] if r else 0

            r = store.query(
                "SELECT COUNT(*) AS cnt FROM runs WHERE timestamp >= "
                "datetime('now', '-24 hours')"
            )
            metrics["runs_last_24h"] = r[0]["cnt"] if r else 0

            r = store.query("SELECT COUNT(DISTINCT market_id) AS cnt FROM runs")
            metrics["unique_markets"] = r[0]["cnt"] if r else 0

            r = store.query("SELECT COUNT(*) AS cnt FROM tools")
            metrics["registered_tools_db"] = r[0]["cnt"] if r else 0

            r = store.query(
                "SELECT AVG(score) AS avg FROM runs WHERE "
                "timestamp >= datetime('now', '-24 hours')"
            )
            avg24 = r[0]["avg"] if r else None
            metrics["avg_score_24h"] = round(avg24, 4) if avg24 is not None else None

        except Exception as exc:
            metrics["sqlite_error"] = str(exc)

    # -- Snapshot file ----------------------------------------------------------
    snap_path = OUTPUTS_DIR / "market_snapshots.jsonl"
    if snap_path.exists():
        snap_lines = _count_jsonl_lines(snap_path)
        metrics["total_snapshots"] = snap_lines
        metrics["snapshot_file_kb"] = round(snap_path.stat().st_size / 1024, 1)
    else:
        metrics["total_snapshots"] = 0
        metrics["snapshot_file_kb"] = 0

    # -- Paper bets JSONL -------------------------------------------------------
    pb_path = OUTPUTS_DIR / "paper_bets.jsonl"
    metrics["paper_bets_jsonl_rows"] = _count_jsonl_lines(pb_path) if pb_path.exists() else 0

    # -- Run log JSONL ----------------------------------------------------------
    rl_path = OUTPUTS_DIR / "run_log.jsonl"
    metrics["run_log_rows"] = _count_jsonl_lines(rl_path) if rl_path.exists() else 0

    # -- Active tools from registry ---------------------------------------------
    try:
        from tools.registry import build_default_registry
        reg = build_default_registry()
        metrics["active_tools"] = reg.tool_names
        metrics["active_tool_count"] = len(reg.tool_names)
        metrics["generated_tools"] = reg.generated_tool_names
    except Exception as exc:
        metrics["active_tools"] = []
        metrics["active_tool_count"] = 0
        metrics["registry_error"] = str(exc)

    # -- Lifecycle JSONL --------------------------------------------------------
    if TOOL_LIFECYCLE_FILE.exists():
        metrics["lifecycle_rows"] = _count_jsonl_lines(TOOL_LIFECYCLE_FILE)
    else:
        metrics["lifecycle_rows"] = 0

    # -- Evolution status -------------------------------------------------------
    metrics["evolution_enabled"] = ENABLE_EVOLUTION

    # -- Outputs directory size -------------------------------------------------
    try:
        total_bytes = sum(
            f.stat().st_size
            for f in OUTPUTS_DIR.rglob("*")
            if f.is_file()
        )
        metrics["outputs_dir_mb"] = round(total_bytes / (1024 * 1024), 2)
    except Exception:
        metrics["outputs_dir_mb"] = None

    metrics["status_timestamp"] = datetime.now(timezone.utc).isoformat()
    return metrics


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file (efficient, no JSON parsing)."""
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except OSError:
        pass
    return count


# -- Formatters -----------------------------------------------------------------

def _fmt_optional(val, fmt=None, suffix="") -> str:
    if val is None:
        return "-"
    if fmt:
        return f"{val:{fmt}}{suffix}"
    return f"{val}{suffix}"


def print_status(metrics: dict) -> None:
    """Print a formatted status dashboard to stdout."""
    ts = metrics.get("status_timestamp", "-")
    evo = "ON  [!]" if metrics.get("evolution_enabled") else "OFF [ok]"

    print()
    print("=" * 60)
    print("         PREDICTION AGENT -- SYSTEM STATUS")
    print("=" * 60)
    print(f"  Status time:       {ts}")
    print(f"  Evolution mode:    {evo}")
    print()

    print("  -- Data Volume " + "-" * 44)
    print(f"  Total snapshots:   {_fmt_optional(metrics.get('total_snapshots'))}")
    print(f"  Total runs:        {_fmt_optional(metrics.get('total_runs'))}")
    print(f"  Bets triggered:    {_fmt_optional(metrics.get('bets_triggered'))}")
    print(f"  Unique markets:    {_fmt_optional(metrics.get('unique_markets'))}")
    print(f"  Paper bets (JSONL):{_fmt_optional(metrics.get('paper_bets_jsonl_rows'))}")
    print()

    print("  -- Recent Activity " + "-" * 40)
    print(f"  Runs (last 1h):    {_fmt_optional(metrics.get('runs_last_1h'))}")
    print(f"  Runs (last 24h):   {_fmt_optional(metrics.get('runs_last_24h'))}")
    print(f"  Last run:          {metrics.get('last_run_timestamp') or '-'}")
    print()

    print("  -- Score Summary " + "-" * 42)
    print(f"  Avg score (all):   {_fmt_optional(metrics.get('avg_score'), '.4f')}")
    print(f"  Avg score (24h):   {_fmt_optional(metrics.get('avg_score_24h'), '.4f')}")
    print()

    print("  -- Tools " + "-" * 50)
    print(f"  Active tools:      {_fmt_optional(metrics.get('active_tool_count'))}")
    tools = metrics.get("active_tools", [])
    for t in tools:
        tag = " [generated]" if t in metrics.get("generated_tools", []) else ""
        print(f"    * {t}{tag}")
    print()

    print("  -- Storage " + "-" * 48)
    print(f"  SQLite available:  {'Yes' if metrics.get('sqlite_available') else 'No'}")
    print(f"  Database size:     {_fmt_optional(metrics.get('sqlite_size_kb'), '.1f', ' KB')}")
    print(f"  Outputs dir:       {_fmt_optional(metrics.get('outputs_dir_mb'), '.2f', ' MB')}")
    print(f"  Snapshots file:    {_fmt_optional(metrics.get('snapshot_file_kb'), '.1f', ' KB')}")
    print()

    if metrics.get("sqlite_error"):
        print(f"  [!] SQLite error:  {metrics['sqlite_error']}")
    if metrics.get("registry_error"):
        print(f"  [!] Registry error: {metrics['registry_error']}")

    print("  " + "-" * 57)
    print()


# -- CLI ------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prediction agent status CLI - all data from SQLite"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw metrics as JSON",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Refresh status every N seconds (0 = run once)",
    )
    args = parser.parse_args()

    try:
        while True:
            metrics = _collect_status()
            if args.json:
                print(json.dumps(metrics, default=str, indent=2))
            else:
                # Clear terminal if watching
                if args.watch:
                    print("\033[2J\033[H", end="")  # ANSI clear screen
                print_status(metrics)

            if args.watch <= 0:
                break
            time.sleep(args.watch)

    except KeyboardInterrupt:
        print("\nStatus monitor stopped.")


if __name__ == "__main__":
    main()
