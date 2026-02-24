"""
Tool lifecycle manager -- tracks per-tool performance over time.

Records usage, score contribution, correctness correlation, and latency.
Marks tools as deprecated after sustained underperformance.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import EVOLUTION_DEPRECATION_RUNS, TOOL_LIFECYCLE_FILE
from prediction_agent.evolution.schemas import ToolLifecycleRecord, ToolStatus

logger = logging.getLogger(__name__)


class ToolLifecycleManager:
    """
    Tracks generated tool performance over time.

    Persists data to a JSONL file. Each record represents the current
    aggregate state for one tool.
    """

    def __init__(self, lifecycle_path: Path | None = None) -> None:
        self._path = lifecycle_path or TOOL_LIFECYCLE_FILE
        self._records: Dict[str, ToolLifecycleRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_tool_with_provenance(
        self,
        tool_name: str,
        namespace: str = "built-in",
        version: int = 1,
        parent_tool_id: Optional[str] = None,
        trigger_gap_id: Optional[str] = None,
        trigger_run_ids: Optional[List[str]] = None,
        capability_tag: Optional[str] = None,
        backtest_delta_score: Optional[float] = None,
        correlation_checked: bool = False,
        verification_checks: Optional[Dict[str, bool]] = None,
    ) -> ToolLifecycleRecord:
        """
        Create or update a tool's lifecycle record with full provenance metadata.

        Call this when a new tool is registered (either built-in or evolved).
        Idempotent — updates the record if it already exists.

        Returns the ToolLifecycleRecord.
        """
        record = self._records.get(tool_name)
        if record is None:
            record = ToolLifecycleRecord(
                tool_name=tool_name,
                namespace=namespace,
                version=version,
                parent_tool_id=parent_tool_id,
                trigger_gap_id=trigger_gap_id,
                trigger_run_ids=trigger_run_ids or [],
                capability_tag=capability_tag,
                backtest_delta_score=backtest_delta_score,
                correlation_checked=correlation_checked,
                verification_checks=verification_checks or {},
            )
            self._records[tool_name] = record
        else:
            # Update provenance fields on existing record
            record.namespace          = namespace
            if version > record.version:
                record.version        = version
            if parent_tool_id:
                record.parent_tool_id = parent_tool_id
            if trigger_gap_id:
                record.trigger_gap_id = trigger_gap_id
            if trigger_run_ids:
                record.trigger_run_ids = trigger_run_ids
            if capability_tag:
                record.capability_tag = capability_tag
            if backtest_delta_score is not None:
                record.backtest_delta_score = backtest_delta_score
            record.correlation_checked = correlation_checked
            if verification_checks:
                record.verification_checks = verification_checks

        self._save()

        # Mirror to SQLite if enabled
        try:
            from config import SQLITE_ENABLED
            if SQLITE_ENABLED:
                from prediction_agent.storage.sqlite_store import SQLiteStore
                SQLiteStore().upsert_tool_lineage(record)
        except Exception as exc:
            logger.debug("SQLite upsert for tool '%s' skipped: %s", tool_name, exc)

        logger.info(
            "Registered tool provenance: %s (namespace=%s, version=%d)",
            tool_name, namespace, version,
        )
        return record

    def record_usage(
        self,
        tool_name: str,
        score_contribution: float,
        correct: bool,
        latency_ms: float,
        underperformance_threshold: float = 0.0,
    ) -> None:
        """
        Record a single usage of a tool.

        Args:
            tool_name: Name of the tool.
            score_contribution: How much this tool contributed to the final score.
            correct: Whether the prediction was correct.
            latency_ms: Execution time in milliseconds.
            underperformance_threshold: Score contribution below this is considered underperformance.
        """
        record = self._records.get(tool_name)
        if record is None:
            record = ToolLifecycleRecord(tool_name=tool_name)
            self._records[tool_name] = record

        record.usage_count += 1
        record.total_score_contribution += score_contribution
        record.total_predictions += 1
        record.total_latency_ms += latency_ms
        record.last_used_at = datetime.now(timezone.utc)

        if correct:
            record.correct_predictions += 1

        # Track consecutive underperformance
        if score_contribution < underperformance_threshold:
            record.consecutive_underperformance += 1
        else:
            record.consecutive_underperformance = 0

        self._save()

    def check_deprecation(self, tool_name: str) -> bool:
        """
        Check if a tool should be deprecated based on sustained underperformance.

        Returns:
            True if the tool should be deprecated.
        """
        record = self._records.get(tool_name)
        if record is None:
            return False

        if record.consecutive_underperformance >= EVOLUTION_DEPRECATION_RUNS:
            if record.status != ToolStatus.DEPRECATED:
                record.status = ToolStatus.DEPRECATED
                self._save()
                logger.info(
                    "Tool '%s' marked as DEPRECATED after %d consecutive underperforming runs.",
                    tool_name,
                    record.consecutive_underperformance,
                )
            return True

        return False

    def get_record(self, tool_name: str) -> Optional[ToolLifecycleRecord]:
        """Get the lifecycle record for a specific tool."""
        return self._records.get(tool_name)

    def get_all_records(self) -> List[ToolLifecycleRecord]:
        """Get all lifecycle records."""
        return list(self._records.values())

    def get_active_tools(self) -> List[str]:
        """Get names of all active (non-deprecated) tools."""
        return [
            name
            for name, record in self._records.items()
            if record.status == ToolStatus.ACTIVE
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load lifecycle records from JSONL."""
        if not self._path.exists():
            return

        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = ToolLifecycleRecord(**data)
                    self._records[record.tool_name] = record
                except (json.JSONDecodeError, Exception):
                    continue

    def _save(self) -> None:
        """Overwrite the lifecycle JSONL with current state."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            for record in self._records.values():
                f.write(json.dumps(record.model_dump(mode="json"), default=str) + "\n")
