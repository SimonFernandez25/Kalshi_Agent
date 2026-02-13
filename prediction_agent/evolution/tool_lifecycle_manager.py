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
from typing import Dict, List, Optional

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
