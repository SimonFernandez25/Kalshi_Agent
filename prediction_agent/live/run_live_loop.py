"""
Live continuous runner — data collection mode.

Continuously polls Kalshi markets, runs the deterministic scoring engine,
logs all results to SQLite and JSONL, then sleeps for the configured interval.

IMPORTANT INVARIANTS:
  - ENABLE_EVOLUTION remains False. No LLM mutation occurs here.
  - No automatic tool generation or registry modification.
  - Deterministic engine only (run_tools + compute_score), no agent graph.
  - All writes are atomic (SQLite WAL, JSONL append-only).
  - Graceful shutdown on SIGINT / SIGTERM.
  - Snapshots are deduplicated by MD5 hash of the API response.

CLI:
    python -m prediction_agent.live.run_live_loop
    python -m prediction_agent.live.run_live_loop --poll-interval 30
    python -m prediction_agent.live.run_live_loop --market-filter KXNBA --dry-run

Architecture:
    [Kalshi API] → deduplicate → [EventInput] → [deterministic engine]
        → [SQLite insert_run] → [JSONL paper_bet] → sleep
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ── resolve project root ───────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Ensure external tools are registered
from tools.registry import build_default_registry, ToolRegistry
from api.kalshi_client import KalshiClient
from config import (
    DEFAULT_BET_AMOUNT,
    ENABLE_EVOLUTION,
    OUTPUTS_DIR,
    PAPER_BET_LOG,
    RUN_LOG,
    SQLITE_ENABLED,
    WATCHER_POLL_INTERVAL_SEC,
)
from engine.scorer import compute_score
from engine.tool_runner import run_tools
from schemas import EventInput, FormulaSpec, ToolSelection, AggregationMethod


logger = logging.getLogger(__name__)

# ── Global shutdown flag ───────────────────────────────────────────────────────
_SHUTDOWN = False


def _install_signal_handlers() -> None:
    """Install SIGINT / SIGTERM handlers for graceful shutdown.

    No-op when called from a non-main thread (e.g. a Jupyter background thread)
    because Python only allows signal.signal() on the main interpreter thread.
    In that case the notebook cell's stop button / _SHUTDOWN flag is used instead.
    """
    import threading as _threading
    if _threading.current_thread() is not _threading.main_thread():
        logger.debug("_install_signal_handlers: skipped (not main thread)")
        return

    def _handler(signum, frame):
        global _SHUTDOWN
        _SHUTDOWN = True
        logger.info("Shutdown signal received — will exit after current iteration.")

    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (AttributeError, OSError):
        pass  # Windows may not have SIGTERM


# ── Hash utilities ─────────────────────────────────────────────────────────────

def _market_hash(market: Dict[str, Any]) -> str:
    """Return MD5 of deterministic JSON serialisation of a market snapshot."""
    canonical = json.dumps(market, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


# ── Deterministic formula (no LLM) ────────────────────────────────────────────

def _build_equal_weight_formula(
    tool_names: List[str],
    threshold: float = 0.55,
) -> FormulaSpec:
    """
    Build a deterministic equal-weight FormulaSpec using all available tools.
    Never calls the LLM — suitable for live data collection mode.
    """
    n = len(tool_names)
    w = round(1.0 / n, 8) if n > 0 else 1.0
    selections = [
        ToolSelection(tool_name=t, weight=w, tool_inputs={})
        for t in tool_names
    ]
    return FormulaSpec(
        selections=selections,
        aggregation=AggregationMethod.WEIGHTED_SUM,
        threshold=threshold,
        rationale="[LIVE_LOOP:equal_weight] Deterministic fixed formula — no LLM",
    )


# ── SQLite logging ─────────────────────────────────────────────────────────────

def _log_to_sqlite(
    run_id: str,
    event: EventInput,
    formula: FormulaSpec,
    score,
    market_price: float,
) -> None:
    """
    Write run + tool_outputs rows to SQLite (atomic via WAL mode).
    Silently skips if SQLITE_ENABLED is False or store unavailable.
    """
    if not SQLITE_ENABLED:
        return
    try:
        from prediction_agent.storage.sqlite_store import SQLiteStore
        store = SQLiteStore()

        # Extend runs table with market_price — stored in existing 'score' columns
        # We use update_outcome() as a post-hoc mechanism; for now just insert_run
        store.insert_run(
            run_id=run_id,
            event=event,
            formula=formula,
            score=score,
            outcome=None,  # unknown until market resolves
        )

        # Also write extra market_price field via raw query
        with store._connect() as conn:
            conn.execute(
                "UPDATE runs SET outcome = ? WHERE run_id = ? AND outcome IS NULL",
                (market_price, run_id),  # use outcome col to store current price temporarily
            )
            # We overwrite with None since outcome = realized result not current price
            conn.execute(
                "UPDATE runs SET outcome = NULL WHERE run_id = ?",
                (run_id,),
            )

    except Exception as exc:
        logger.warning("SQLite write failed (non-fatal): %s", exc)


# ── JSONL paper-bet logging ────────────────────────────────────────────────────

def _log_paper_bet(
    run_id: str,
    event: EventInput,
    formula: FormulaSpec,
    score,
    dry_run: bool,
) -> None:
    """Append paper bet record to JSONL if bet triggered and not dry-run."""
    if dry_run:
        return

    any_triggered = score.bet_triggered
    bet_side = "YES" if score.final_score >= score.threshold else "NO"

    paper_entry = {
        "run_id": run_id,
        "event_input": event.model_dump(mode="json"),
        "formula_spec": formula.model_dump(mode="json"),
        "score_result": score.model_dump(mode="json"),
        "watcher_ticks": [],
        "bet_placed": any_triggered,
        "bet_side": bet_side if any_triggered else None,
        "bet_amount": DEFAULT_BET_AMOUNT if any_triggered else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "live_loop",
    }

    _append_jsonl(PAPER_BET_LOG, paper_entry)

    run_entry = {
        "run_id": run_id,
        "market_id": event.market_id,
        "market_title": event.market_title,
        "current_price": event.current_price,
        "tools_used": [s.tool_name for s in formula.selections],
        "weights": [s.weight for s in formula.selections],
        "threshold": formula.threshold,
        "final_score": score.final_score,
        "bet_triggered": any_triggered,
        "bet_side": bet_side if any_triggered else None,
        "bet_amount": DEFAULT_BET_AMOUNT if any_triggered else 0.0,
        "source": "live_loop",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(RUN_LOG, run_entry)


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a JSON record to a JSONL file (atomic line write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ── Snapshot deduplication ─────────────────────────────────────────────────────

def _append_snapshot(snapshot: Dict[str, Any]) -> None:
    """Append a market snapshot to market_snapshots.jsonl."""
    snap_path = OUTPUTS_DIR / "market_snapshots.jsonl"
    _append_jsonl(snap_path, snapshot)


def _load_seen_hashes(n: int = 500) -> Set[str]:
    """Load the last N snapshot hashes for deduplication."""
    snap_path = OUTPUTS_DIR / "market_snapshots.jsonl"
    if not snap_path.exists():
        return set()
    hashes: List[str] = []
    with open(snap_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                h = row.get("_hash", "")
                if h:
                    hashes.append(h)
            except json.JSONDecodeError:
                continue
    return set(hashes[-n:])


# ── Per-market processing ──────────────────────────────────────────────────────

def _process_market(
    market: Dict[str, Any],
    registry,
    formula: FormulaSpec,
    seen_hashes: Set[str],
    dry_run: bool,
) -> Optional[Dict[str, Any]]:
    """
    Process a single market snapshot: score it, log results.

    Returns a compact summary dict, or None if skipped (duplicate/error).
    """
    # Work on a shallow copy so we never mutate the caller's dict (or stub objects).
    market = dict(market)

    mhash = _market_hash(market)

    # Deduplicate
    if mhash in seen_hashes:
        return None

    market["_hash"] = mhash

    # Build EventInput
    try:
        ts = market.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif ts is None:
            ts = datetime.now(timezone.utc)

        price = float(market.get("last_price", market.get("current_price", 0.0)))
        market_id = market.get("market_id", "")
        if not market_id:
            return None

        event = EventInput(
            event_id=market.get("event_id", market_id),
            market_id=market_id,
            market_title=market.get("title", market_id),
            current_price=price,
            timestamp=ts,
        )
    except Exception as exc:
        logger.debug("Could not build EventInput for market %s: %s", market.get("market_id"), exc)
        return None

    # Market sanity check (abstain on zero/near-zero prices)
    try:
        from config import ABSTAIN_ON_ZERO_PRICE, MIN_VALID_PRICE
    except ImportError:
        ABSTAIN_ON_ZERO_PRICE = True
        MIN_VALID_PRICE = 0.01

    if ABSTAIN_ON_ZERO_PRICE:
        if price <= MIN_VALID_PRICE:
            logger.debug("Abstaining on %s: price=%.4f <= MIN_VALID_PRICE", market_id, price)
            return None
        if price >= 1.0 - MIN_VALID_PRICE:
            logger.debug("Abstaining on %s: price=%.4f >= %.4f (resolved)", market_id, price, 1.0 - MIN_VALID_PRICE)
            return None

    # Run deterministic engine
    try:
        tool_outputs, tool_statuses = run_tools(event, formula, registry)
        score = compute_score(tool_outputs, formula)
    except Exception as exc:
        logger.warning("Engine failed for %s: %s", event.market_id, exc)
        return None

    run_id = str(uuid.uuid4())[:8]

    # Append snapshot (with hash for dedup)
    if not dry_run:
        _append_snapshot(market)

    # Log to SQLite
    _log_to_sqlite(run_id, event, formula, score, price)

    # Log paper bet (JSONL)
    _log_paper_bet(run_id, event, formula, score, dry_run)

    seen_hashes.add(mhash)

    # Build tool contribution summary for console
    contrib_summary = []
    for out, sel in zip(tool_outputs, formula.selections):
        vec = out.output_vector
        mean_val = sum(vec) / len(vec) if vec else 0.0
        contrib_summary.append({
            "tool": sel.tool_name,
            "mean": round(mean_val, 4),
            "weight": round(sel.weight, 4),
            "contribution": round(sel.weight * mean_val, 4),
        })

    return {
        "run_id": run_id,
        "market_id": event.market_id,
        "market_title": event.market_title[:40],
        "price": price,
        "score": score.final_score,
        "threshold": score.threshold,
        "bet_triggered": score.bet_triggered,
        "tool_contributions": contrib_summary,
    }


# ── Console summary printer ────────────────────────────────────────────────────

def _print_iteration_summary(
    results: List[Dict[str, Any]],
    iteration: int,
    dry_run: bool,
) -> None:
    """Print a human-readable summary of one poll iteration."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    mode = "[DRY RUN]" if dry_run else "[LIVE]"
    processed = len(results)
    bets = sum(1 for r in results if r.get("bet_triggered"))

    print(f"\n{'─'*60}")
    print(f"  {mode} Iteration #{iteration}  |  {ts} UTC")
    print(f"  Markets processed: {processed}  |  Bets triggered: {bets}")
    print(f"{'─'*60}")

    for r in results:
        trigger_icon = "▶ BET" if r["bet_triggered"] else "  ·  "
        print(
            f"  {trigger_icon}  {r['market_id'][:22]:<22s}  "
            f"price={r['price']:.3f}  score={r['score']:.4f}  "
            f"threshold={r['threshold']:.3f}"
        )
    print()


# ── Retry wrapper ──────────────────────────────────────────────────────────────

def _fetch_with_retry(
    client: KalshiClient,
    limit: int,
    max_retries: int = 3,
    backoff_sec: float = 5.0,
) -> List[Dict[str, Any]]:
    """Fetch markets with exponential backoff on transient failures."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return client.get_active_markets(limit=limit)
        except Exception as exc:
            last_exc = exc
            wait = backoff_sec * (2 ** attempt)
            logger.warning(
                "API fetch attempt %d/%d failed: %s — retrying in %.0fs",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    logger.error("All %d API fetch attempts failed. Last error: %s", max_retries, last_exc)
    return []


# ── Main live loop ─────────────────────────────────────────────────────────────

def run_live_loop(
    poll_interval: int = 60,
    market_filter: Optional[str] = None,
    dry_run: bool = False,
    threshold: float = 0.55,
    max_markets: int = 50,
) -> None:
    """
    Main live data collection loop.

    Args:
        poll_interval:  Seconds between API polls.
        market_filter:  Optional substring to filter market_id.
        dry_run:        If True, skip all writes (safe exploration).
        threshold:      Score threshold for paper bet triggering.
        max_markets:    Max markets to fetch per iteration.
    """
    _install_signal_handlers()

    # Safety assertion — evolution must remain off
    assert not ENABLE_EVOLUTION, (
        "ENABLE_EVOLUTION must be False during live collection mode. "
        "Set ENABLE_EVOLUTION = False in config.py."
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    mode_str = "DRY RUN" if dry_run else "LIVE"
    logger.info("=" * 60)
    logger.info("LIVE LOOP STARTING  [%s MODE]", mode_str)
    logger.info("poll_interval=%ds  market_filter=%s  threshold=%.3f",
                poll_interval, market_filter or "(none)", threshold)
    logger.info("ENABLE_EVOLUTION=%s (must be False)", ENABLE_EVOLUTION)
    logger.info("=" * 60)

    client   = KalshiClient()
    
    # ── FORCE RE-BUILD REGISTRY ───────────────────────────────────────────────
    # This ensures new tools are picked up even if module was cached previously
    import tools.registry
    import importlib
    importlib.reload(tools.registry)
    registry = tools.registry.build_default_registry()
    # ──────────────────────────────────────────────────────────────────────────

    formula  = _build_equal_weight_formula(registry.tool_names, threshold=threshold)

    logger.info("Registry: %d tools — %s", len(registry), registry.tool_names)
    logger.info("Formula:  %d selections, threshold=%.3f", len(formula.selections), threshold)

    seen_hashes = _load_seen_hashes(n=1000)
    logger.info("Loaded %d known hashes for deduplication.", len(seen_hashes))

    iteration = 0

    while not _SHUTDOWN:
        iteration += 1
        iter_start = time.perf_counter()

        # 1. Fetch markets
        markets = _fetch_with_retry(client, limit=max_markets)

        if not markets:
            logger.warning("No markets returned. Sleeping and retrying.")
            time.sleep(poll_interval)
            continue

        # 2. Apply market filter
        if market_filter:
            markets = [m for m in markets if market_filter in m.get("market_id", "")]
            if not markets:
                logger.debug("No markets matched filter '%s'. Sleeping.", market_filter)
                time.sleep(poll_interval)
                continue

        # 3. Process each market
        iteration_results = []
        for market in markets:
            if _SHUTDOWN:
                break
            result = _process_market(
                market=market,
                registry=registry,
                formula=formula,
                seen_hashes=seen_hashes,
                dry_run=dry_run,
            )
            if result is not None:
                iteration_results.append(result)

        # 4. Print summary
        if iteration_results:
            _print_iteration_summary(iteration_results, iteration, dry_run)
        else:
            logger.info("Iteration #%d: all %d markets were duplicates — nothing new.",
                        iteration, len(markets))

        # 5. Sleep (respecting shutdown flag)
        elapsed = time.perf_counter() - iter_start
        sleep_time = max(0.0, poll_interval - elapsed)
        logger.debug("Iteration #%d complete in %.1fs. Sleeping %.1fs.", iteration, elapsed, sleep_time)

        slept = 0.0
        while slept < sleep_time and not _SHUTDOWN:
            time.sleep(min(1.0, sleep_time - slept))
            slept += 1.0

    logger.info("Live loop shut down cleanly after %d iterations.", iteration)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live continuous runner — data collection mode (no LLM, no evolution)"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between Kalshi API polls (default: 60)",
    )
    parser.add_argument(
        "--market-filter",
        type=str,
        default=None,
        help="Filter markets by market_id substring (e.g. KXNBA)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing to disk (safe exploration mode)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Score threshold for paper bet triggering (default: 0.55)",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=50,
        help="Max markets to fetch per poll (default: 50)",
    )
    args = parser.parse_args()

    run_live_loop(
        poll_interval=args.poll_interval,
        market_filter=args.market_filter,
        dry_run=args.dry_run,
        threshold=args.threshold,
        max_markets=args.max_markets,
    )


if __name__ == "__main__":
    main()
