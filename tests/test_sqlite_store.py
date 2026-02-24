"""
Tests for the SQLite storage layer.

Verifies:
  - Schema initialisation creates all required tables
  - insert_run() persists a run and tool_outputs
  - upsert_tool_lineage() persists provenance metadata
  - query() returns correct data
  - migrate_jsonl_to_sqlite() is idempotent
  - Missing JSONL file returns 0, not an error
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prediction_agent.storage.sqlite_store import SQLiteStore, migrate_jsonl_to_sqlite


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_store() -> SQLiteStore:
    """Return an in-memory-equivalent SQLiteStore using a temp file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return SQLiteStore(db_path=Path(tmp.name))


def _make_event():
    from schemas import EventInput
    return EventInput(
        event_id="evt-001",
        market_id="KXNBA-TEST",
        market_title="Test NBA Market",
        current_price=0.55,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def _make_formula():
    from schemas import FormulaSpec, ToolSelection, AggregationMethod
    return FormulaSpec(
        selections=[
            ToolSelection(tool_name="mock_price_signal", weight=0.6),
            ToolSelection(tool_name="mock_random_context", weight=0.4),
        ],
        aggregation=AggregationMethod.WEIGHTED_SUM,
        threshold=0.55,
        rationale="test formula",
    )


def _make_score(formula):
    from schemas import ScoreResult, ToolOutput
    return ScoreResult(
        final_score=0.62,
        tool_outputs=[
            ToolOutput(tool_name="mock_price_signal", output_vector=[0.55]),
            ToolOutput(tool_name="mock_random_context", output_vector=[0.72]),
        ],
        weights=[s.weight for s in formula.selections],
        threshold=formula.threshold,
        bet_triggered=True,
    )


def _make_lifecycle_record(tool_name: str = "test_tool"):
    from prediction_agent.evolution.schemas import ToolLifecycleRecord, ToolStatus
    return ToolLifecycleRecord(
        tool_name=tool_name,
        namespace="evolved/v1",
        version=2,
        parent_tool_id="snapshot_volatility_tool",
        trigger_gap_id="gap-abc123",
        trigger_run_ids=["run-1", "run-2", "run-3"],
        capability_tag="volatility",
        backtest_delta_score=0.012,
        correlation_checked=True,
        verification_checks={"ast_inspection": True, "runtime_sandbox": True, "determinism": True},
        status=ToolStatus.ACTIVE,
    )


# ── Schema tests ───────────────────────────────────────────────────────────────

class TestSchemaInit:
    def test_tables_exist_after_init(self):
        """All three required tables must be created on first connect."""
        store = _make_store()
        tables = store.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {r["name"] for r in tables}
        assert "runs" in table_names
        assert "tool_outputs" in table_names
        assert "tools" in table_names

    def test_init_is_idempotent(self):
        """Creating the store twice on the same file should not raise."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        p = Path(tmp.name)
        s1 = SQLiteStore(db_path=p)
        s2 = SQLiteStore(db_path=p)  # Should not raise


# ── Run insertion tests ────────────────────────────────────────────────────────

class TestInsertRun:
    def test_insert_run_creates_row(self):
        store  = _make_store()
        event  = _make_event()
        formula = _make_formula()
        score   = _make_score(formula)

        store.insert_run("run-test-001", event, formula, score)

        row = store.get_run("run-test-001")
        assert row is not None
        assert row["run_id"] == "run-test-001"
        assert row["market_id"] == "KXNBA-TEST"
        assert abs(row["score"] - 0.62) < 0.001
        assert row["bet_triggered"] == 1

    def test_insert_run_creates_tool_output_rows(self):
        store   = _make_store()
        event   = _make_event()
        formula = _make_formula()
        score   = _make_score(formula)

        store.insert_run("run-test-002", event, formula, score)

        tool_rows = store.get_tool_outputs_for_run("run-test-002")
        assert len(tool_rows) == 2
        tool_names = {r["tool_id"] for r in tool_rows}
        assert "mock_price_signal" in tool_names
        assert "mock_random_context" in tool_names

    def test_insert_run_is_idempotent(self):
        """INSERT OR REPLACE — second insert with same run_id should not raise."""
        store   = _make_store()
        event   = _make_event()
        formula = _make_formula()
        score   = _make_score(formula)

        store.insert_run("run-dup-001", event, formula, score)
        store.insert_run("run-dup-001", event, formula, score)  # Should not raise

        rows = store.query("SELECT * FROM runs WHERE run_id = 'run-dup-001'")
        assert len(rows) == 1

    def test_update_outcome(self):
        store   = _make_store()
        event   = _make_event()
        formula = _make_formula()
        score   = _make_score(formula)

        store.insert_run("run-outcome-001", event, formula, score)
        store.update_outcome("run-outcome-001", 1.0)

        row = store.get_run("run-outcome-001")
        assert row["outcome"] == pytest.approx(1.0)


# ── Tool lineage tests ─────────────────────────────────────────────────────────

class TestToolLineage:
    def test_upsert_tool_lineage_persists_provenance(self):
        store  = _make_store()
        record = _make_lifecycle_record("my_evolved_tool")

        store.upsert_tool_lineage(record)

        row = store.get_tool_lineage("my_evolved_tool")
        assert row is not None
        assert row["namespace"]       == "evolved/v1"
        assert row["version"]         == 2
        assert row["parent_tool_id"]  == "snapshot_volatility_tool"
        assert row["trigger_gap_id"]  == "gap-abc123"
        assert row["capability_tag"]  == "volatility"
        assert abs(row["backtest_delta_score"] - 0.012) < 0.001

    def test_trigger_run_ids_round_trip(self):
        store  = _make_store()
        record = _make_lifecycle_record("provenance_tool")

        store.upsert_tool_lineage(record)

        row = store.get_tool_lineage("provenance_tool")
        run_ids = row.get("trigger_run_ids", [])
        assert isinstance(run_ids, list)
        assert "run-1" in run_ids
        assert "run-2" in run_ids
        assert "run-3" in run_ids

    def test_upsert_is_idempotent(self):
        store  = _make_store()
        record = _make_lifecycle_record("idempotent_tool")

        store.upsert_tool_lineage(record)
        store.upsert_tool_lineage(record)

        rows = store.query("SELECT * FROM tools WHERE tool_id = 'idempotent_tool'")
        assert len(rows) == 1


# ── Migration tests ────────────────────────────────────────────────────────────

class TestMigration:
    def _make_paper_bet_jsonl(self, n: int) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for i in range(n):
            record = {
                "run_id": f"migrated-run-{i:04d}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_input": {
                    "event_id": f"evt-{i}",
                    "market_id": f"KXNBA-MIG-{i}",
                    "market_title": f"Migration Test {i}",
                    "current_price": 0.5,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "formula_spec": {
                    "selections": [
                        {"tool_name": "mock_price_signal", "weight": 1.0, "tool_inputs": {}}
                    ],
                    "aggregation": "weighted_sum",
                    "threshold": 0.55,
                    "rationale": "test",
                },
                "score_result": {
                    "final_score": 0.5 + i * 0.01,
                    "tool_outputs": [
                        {"tool_name": "mock_price_signal", "output_vector": [0.5], "metadata": {}}
                    ],
                    "weights": [1.0],
                    "threshold": 0.55,
                    "bet_triggered": i % 2 == 0,
                },
            }
            tmp.write(json.dumps(record) + "\n")
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def test_migration_inserts_all_rows(self):
        jsonl = self._make_paper_bet_jsonl(10)
        tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_db.close()

        n = migrate_jsonl_to_sqlite(
            paper_bets_path=jsonl,
            db_path=Path(tmp_db.name),
        )
        assert n == 10

    def test_migration_is_idempotent(self):
        """Running migration twice must not create duplicate rows."""
        jsonl = self._make_paper_bet_jsonl(5)
        tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_db.close()
        db_path = Path(tmp_db.name)

        migrate_jsonl_to_sqlite(paper_bets_path=jsonl, db_path=db_path)
        migrate_jsonl_to_sqlite(paper_bets_path=jsonl, db_path=db_path)

        store = SQLiteStore(db_path=db_path)
        rows  = store.query("SELECT COUNT(*) AS cnt FROM runs")
        assert rows[0]["cnt"] == 5

    def test_missing_jsonl_returns_zero(self):
        """A missing JSONL file should return 0, not raise."""
        tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_db.close()

        n = migrate_jsonl_to_sqlite(
            paper_bets_path=Path("/tmp/this_does_not_exist_abc.jsonl"),
            db_path=Path(tmp_db.name),
        )
        assert n == 0
