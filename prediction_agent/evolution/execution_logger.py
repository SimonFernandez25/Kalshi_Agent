"""
Execution logger -- captures structured data from every pipeline run.

Appends ExecutionLogEntry records to outputs/execution_logs.jsonl
so the gap analyzer can detect missing capabilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from config import EXECUTION_LOG_FILE
from prediction_agent.evolution.schemas import ExecutionLogEntry

logger = logging.getLogger(__name__)


def log_execution(
    run_result: Dict[str, Any],
    log_path: Path | None = None,
) -> ExecutionLogEntry:
    """
    Extract structured execution data from a pipeline run result
    and append it to the execution log JSONL.

    Args:
        run_result: The dict returned by run_pipeline().
        log_path: Override path for the log file (used in tests).

    Returns:
        The ExecutionLogEntry that was logged.
    """
    import hashlib
    from datetime import datetime, timezone
    from config import LIVE_MODE
    
    target = log_path or EXECUTION_LOG_FILE

    formula = run_result.get("formula", {})
    score = run_result.get("score", {})
    event = run_result.get("event", {})

    selections = formula.get("selections", [])
    tool_outputs_raw = score.get("tool_outputs", [])
    
    # Compute API response hash for stale data detection
    api_response_hash = None
    api_response_timestamp = None
    if event:
        event_str = json.dumps(event, sort_keys=True)
        api_response_hash = hashlib.md5(event_str.encode()).hexdigest()
        api_response_timestamp = event.get("timestamp", datetime.now(timezone.utc))
        if isinstance(api_response_timestamp, str):
            api_response_timestamp = datetime.fromisoformat(api_response_timestamp)

    entry = ExecutionLogEntry(
        run_id=run_result.get("run_id", "unknown"),
        market_id=event.get("market_id", ""),
        market_title=event.get("market_title", ""),
        selected_tools=[s.get("tool_name", "") for s in selections],
        tool_weights=[s.get("weight", 0.0) for s in selections],
        tool_outputs=[
            {
                "tool_name": to.get("tool_name", ""),
                "output_vector": to.get("output_vector", []),
            }
            for to in tool_outputs_raw
        ],
        final_score=score.get("final_score", 0.0),
        threshold=score.get("threshold", 0.0),
        bet_triggered=score.get("bet_triggered", False),
        reasoning_segments=formula.get("rationale", ""),
        failed_tool_attempts=[],
        total_tokens_used=0,
        kalshi_response_timestamp=api_response_timestamp,
        kalshi_response_hash=api_response_hash,
        tool_execution_statuses=run_result.get("tool_statuses", []),
    )
    
    # Stale data detection
    if LIVE_MODE and api_response_hash:
        _check_stale_data(target, api_response_hash)

    _append_jsonl(target, entry.model_dump(mode="json"))
    logger.info(
        "Execution logged: run=%s market=%s score=%.4f hash=%s",
        entry.run_id,
        entry.market_id,
        entry.final_score,
        api_response_hash[:8] if api_response_hash else "N/A"
    )
    return entry


def _check_stale_data(log_path: Path, current_hash: str) -> None:
    """Warn if the same API response hash appears in consecutive runs."""
    if not log_path.exists():
        return
    
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) < 2:
        return
    
    # Check last 3 entries
    recent_hashes = []
    for line in lines[-3:]:
        try:
            entry = json.loads(line)
            if hash_val := entry.get("kalshi_response_hash"):
                recent_hashes.append(hash_val)
        except json.JSONDecodeError:
            continue
    
    if recent_hashes and all(h == current_hash for h in recent_hashes):
        logger.warning(
            "STALE API DATA DETECTED: hash %s seen in %d consecutive runs",
            current_hash[:8],
            len(recent_hashes) + 1
        )
        print(f"[WARNING] Stale API data detected: hash {current_hash[:8]} " 
              f"seen in {len(recent_hashes) + 1} consecutive runs")


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
