"""
SQLite-backed storage layer for the prediction agent.

Provides structured, queryable storage for:
  - runs:         One row per pipeline execution (score, bet, outcome)
  - tool_outputs: Per-tool output scalars for each run
  - tools:        Tool registry metadata including full provenance lineage

The JSONL paper_bets.jsonl remains append-only as the audit trail.
This database is complementary — designed for aggregation, backtesting queries,
and lineage traversal.

Schema:
    runs(
        run_id        TEXT PRIMARY KEY,
        timestamp     TEXT,          -- ISO 8601 UTC
        event_id      TEXT,
        market_id     TEXT,
        market_title  TEXT,
        score         REAL,
        threshold     REAL,
        bet_triggered INTEGER,       -- 0 or 1
        outcome       REAL           -- realized outcome if known, else NULL
    )

    tool_outputs(
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        TEXT,
        tool_id       TEXT,
        output_mean   REAL,          -- mean(output_vector)
        weight        REAL,          -- weight used in FormulaSpec
        contribution  REAL           -- weight * output_mean
    )

    tools(
        tool_id       TEXT PRIMARY KEY,
        namespace     TEXT,
        version       INTEGER,
        birth_timestamp TEXT,        -- ISO 8601 UTC
        parent_tool_id  TEXT,        -- NULL for built-in tools
        trigger_gap_id  TEXT,
        capability_tag  TEXT,
        status        TEXT,          -- active / deprecated / pending
        backtest_delta_score REAL
    )

Usage:
    from prediction_agent.storage.sqlite_store import SQLiteStore
    store = SQLiteStore()
    store.insert_run(run_id, event, formula, score)
    store.upsert_tool_lineage(lifecycle_record)

    # Query
    rows = store.query("SELECT * FROM runs WHERE bet_triggered=1 ORDER BY timestamp DESC LIMIT 20")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = None  # Resolved at runtime from config


# ── SQLiteStore class ──────────────────────────────────────────────────────────

class SQLiteStore:
    """
    Thin wrapper around sqlite3 providing the prediction agent's
    structured storage schema.

    Thread-safety: each method opens and closes its own connection
    (SQLite WAL mode) so concurrent writers are safe on the same file.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            try:
                from config import OUTPUTS_DIR
                db_path = OUTPUTS_DIR / "prediction_agent.db"
            except ImportError:
                db_path = Path("outputs/prediction_agent.db")
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Schema init ────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Create tables if they don't exist. Safe to call multiple times."""
        with self._connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS runs (
                    run_id        TEXT PRIMARY KEY,
                    timestamp     TEXT NOT NULL,
                    event_id      TEXT,
                    market_id     TEXT,
                    market_title  TEXT,
                    score         REAL,
                    threshold     REAL,
                    bet_triggered INTEGER DEFAULT 0,
                    outcome       REAL
                );

                CREATE TABLE IF NOT EXISTS tool_outputs (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id       TEXT NOT NULL,
                    tool_id      TEXT NOT NULL,
                    output_mean  REAL,
                    weight       REAL,
                    contribution REAL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS tools (
                    tool_id              TEXT PRIMARY KEY,
                    namespace            TEXT DEFAULT 'built-in',
                    version              INTEGER DEFAULT 1,
                    birth_timestamp      TEXT,
                    parent_tool_id       TEXT,
                    trigger_gap_id       TEXT,
                    capability_tag       TEXT,
                    status               TEXT DEFAULT 'active',
                    backtest_delta_score REAL,
                    trigger_run_ids      TEXT  -- JSON array stored as TEXT
                );

                CREATE TABLE IF NOT EXISTS historical_tool_evaluation (
                    eval_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id        TEXT NOT NULL,
                    market_id     TEXT NOT NULL,
                    domain        TEXT,
                    p_model       REAL,
                    p_market      REAL,
                    edge          REAL,
                    decision      TEXT,
                    outcome       REAL,
                    raw_score_z   REAL,
                    tool_name     TEXT NOT NULL,
                    tool_signal_mean REAL,
                    weight        REAL,
                    z_contribution REAL,
                    market_timestamp TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_runs_market ON runs(market_id);
                CREATE INDEX IF NOT EXISTS idx_runs_bet ON runs(bet_triggered);
                CREATE INDEX IF NOT EXISTS idx_tool_outputs_run ON tool_outputs(run_id);
                CREATE INDEX IF NOT EXISTS idx_tool_outputs_tool ON tool_outputs(tool_id);
                CREATE INDEX IF NOT EXISTS idx_eval_market ON historical_tool_evaluation(market_id);
                CREATE INDEX IF NOT EXISTS idx_eval_tool ON historical_tool_evaluation(tool_name);
                CREATE INDEX IF NOT EXISTS idx_eval_run ON historical_tool_evaluation(run_id);
            """)

    # ── Run insertion ──────────────────────────────────────────────────────────

    def insert_run(
        self,
        run_id: str,
        event: "EventInput",
        formula: "FormulaSpec",
        score: "ScoreResult",
        outcome: Optional[float] = None,
    ) -> None:
        """
        Insert a completed pipeline run into the runs + tool_outputs tables.

        Args:
            run_id:   Unique run identifier (from PaperBet).
            event:    EventInput used in the run.
            formula:  FormulaSpec (carries tools, weights, threshold).
            score:    ScoreResult with final_score and tool_outputs.
            outcome:  Realized outcome [0,1] if known, else None.
        """
        ts = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, timestamp, event_id, market_id, market_title,
                     score, threshold, bet_triggered, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    ts,
                    event.event_id,
                    event.market_id,
                    event.market_title,
                    score.final_score,
                    score.threshold,
                    int(score.bet_triggered),
                    outcome,
                ),
            )

            for tool_out, sel in zip(score.tool_outputs, formula.selections):
                vec = tool_out.output_vector
                output_mean = sum(vec) / len(vec) if vec else 0.0
                contribution = sel.weight * output_mean
                conn.execute(
                    """
                    INSERT INTO tool_outputs
                        (run_id, tool_id, output_mean, weight, contribution)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (run_id, sel.tool_name, output_mean, sel.weight, contribution),
                )

        logger.debug("SQLiteStore: inserted run %s", run_id)

    # ── Tool lineage upsert ────────────────────────────────────────────────────

    def upsert_tool_lineage(self, record: "ToolLifecycleRecord") -> None:
        """
        Insert or replace a tool's lineage record in the tools table.

        Args:
            record: ToolLifecycleRecord with full provenance metadata.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tools
                    (tool_id, namespace, version, birth_timestamp,
                     parent_tool_id, trigger_gap_id, capability_tag,
                     status, backtest_delta_score, trigger_run_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.tool_name,
                    record.namespace,
                    record.version,
                    record.created_at.isoformat(),
                    record.parent_tool_id,
                    record.trigger_gap_id,
                    record.capability_tag,
                    record.status.value,
                    record.backtest_delta_score,
                    json.dumps(record.trigger_run_ids),
                ),
            )
        logger.debug("SQLiteStore: upserted tool lineage for %s", record.tool_name)

    # ── Outcome update ─────────────────────────────────────────────────────────

    def update_outcome(self, run_id: str, outcome: float) -> None:
        """
        Set the realized outcome for a past run (used when market resolves).

        Args:
            run_id:  The run to update.
            outcome: Realized YES probability (1.0 = yes, 0.0 = no).
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET outcome = ? WHERE run_id = ?",
                (outcome, run_id),
            )

    # ── Query helpers ──────────────────────────────────────────────────────────

    def query(self, sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a raw SELECT query and return rows as list of dicts.

        Args:
            sql:    SQL SELECT statement.
            params: Optional parameter tuple.

        Returns:
            List of row dicts.
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single run by run_id."""
        rows = self.query("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        return rows[0] if rows else None

    def get_tool_lineage(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Fetch lineage record for a single tool."""
        rows = self.query("SELECT * FROM tools WHERE tool_id = ?", (tool_id,))
        if not rows:
            return None
        row = rows[0]
        # Decode trigger_run_ids JSON
        if row.get("trigger_run_ids"):
            try:
                row["trigger_run_ids"] = json.loads(row["trigger_run_ids"])
            except (json.JSONDecodeError, TypeError):
                row["trigger_run_ids"] = []
        return row

    def get_tool_outputs_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Fetch all tool output rows for a given run_id."""
        return self.query(
            "SELECT * FROM tool_outputs WHERE run_id = ? ORDER BY id",
            (run_id,),
        )

    def recent_runs(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the n most recent runs ordered by timestamp."""
        return self.query(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?",
            (n,),
        )

    def tool_accuracy(self, tool_id: str) -> Dict[str, Any]:
        """
        Compute prediction accuracy and contribution stats for one tool.
        Only includes runs where outcome is known (not NULL).
        """
        rows = self.query(
            """
            SELECT
                t.tool_id,
                COUNT(*) AS times_used,
                AVG(t.output_mean) AS avg_output,
                AVG(t.contribution) AS avg_contribution,
                SUM(CASE WHEN r.bet_triggered=1 AND r.outcome >= 0.5 THEN 1 ELSE 0 END)
                    AS correct_triggered,
                SUM(CASE WHEN r.bet_triggered=1 THEN 1 ELSE 0 END)
                    AS total_triggered
            FROM tool_outputs t
            JOIN runs r ON t.run_id = r.run_id
            WHERE t.tool_id = ?
              AND r.outcome IS NOT NULL
            """,
            (tool_id,),
        )
        return rows[0] if rows else {}

    def insert_evaluation(
        self,
        run_id: str,
        market_id: str,
        domain: str,
        p_model: Optional[float],
        p_market: float,
        edge: Optional[float],
        decision: str,
        outcome: float,
        raw_score_z: Optional[float],
        tool_name: str,
        tool_signal_mean: float,
        weight: float,
        z_contribution: float,
        market_timestamp: str,
    ) -> None:
        """
        Insert a single evaluation record into historical_tool_evaluation table.

        This is called once per tool per market during evaluation runs.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO historical_tool_evaluation
                    (run_id, market_id, domain, p_model, p_market, edge, decision,
                     outcome, raw_score_z, tool_name, tool_signal_mean, weight,
                     z_contribution, market_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    market_id,
                    domain,
                    p_model,
                    p_market,
                    edge,
                    decision,
                    outcome,
                    raw_score_z,
                    tool_name,
                    tool_signal_mean,
                    weight,
                    z_contribution,
                    market_timestamp,
                ),
            )

    # ── Connection factory ─────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Open a database connection. Use as a context manager."""
        conn = sqlite3.connect(str(self._path), timeout=10)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


# ── JSONL migration script ─────────────────────────────────────────────────────

def migrate_jsonl_to_sqlite(
    paper_bets_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Backfill existing paper_bets.jsonl records into SQLite.

    Idempotent — skips rows that already exist (INSERT OR IGNORE).
    Returns the number of rows inserted.

    Args:
        paper_bets_path: Path to paper_bets.jsonl. Defaults to outputs/paper_bets.jsonl.
        db_path:         Path to target SQLite DB. Defaults to outputs/prediction_agent.db.

    Returns:
        Number of new rows inserted.
    """
    try:
        from config import OUTPUTS_DIR
        if paper_bets_path is None:
            paper_bets_path = OUTPUTS_DIR / "paper_bets.jsonl"
    except ImportError:
        if paper_bets_path is None:
            paper_bets_path = Path("outputs/paper_bets.jsonl")

    store = SQLiteStore(db_path)

    if not paper_bets_path.exists():
        logger.info("Migration: paper_bets.jsonl not found at %s. Nothing to migrate.", paper_bets_path)
        return 0

    inserted = 0
    skipped  = 0

    with open(paper_bets_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                _migrate_paper_bet_record(store, record)
                inserted += 1
            except Exception as exc:
                logger.debug("Migration: skipping line %d: %s", line_num, exc)
                skipped += 1

    logger.info(
        "Migration complete: %d rows inserted, %d skipped.", inserted, skipped
    )
    return inserted


def _migrate_paper_bet_record(store: SQLiteStore, record: Dict[str, Any]) -> None:
    """
    Insert a single paper_bets.jsonl record into SQLite.
    Uses raw SQL with INSERT OR IGNORE to stay idempotent.
    """
    run_id       = record.get("run_id", "")
    event_data   = record.get("event_input", {})
    formula_data = record.get("formula_spec", {})
    score_data   = record.get("score_result", {})

    if not run_id:
        return

    ts = record.get("timestamp", datetime.now(timezone.utc).isoformat())

    with store._connect() as conn:
        # Insert run row (ignore if already exists)
        conn.execute(
            """
            INSERT OR IGNORE INTO runs
                (run_id, timestamp, event_id, market_id, market_title,
                 score, threshold, bet_triggered, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                ts,
                event_data.get("event_id", ""),
                event_data.get("market_id", ""),
                event_data.get("market_title", ""),
                score_data.get("final_score", 0.0),
                score_data.get("threshold", 0.0),
                int(score_data.get("bet_triggered", False)),
                None,  # outcome unknown from paper bets
            ),
        )

        # Insert tool_output rows
        tool_outputs  = score_data.get("tool_outputs", [])
        selections    = formula_data.get("selections", [])

        for out, sel in zip(tool_outputs, selections):
            vec = out.get("output_vector", [])
            output_mean  = sum(vec) / len(vec) if vec else 0.0
            weight       = sel.get("weight", 0.0)
            contribution = weight * output_mean

            conn.execute(
                """
                INSERT OR IGNORE INTO tool_outputs
                    (run_id, tool_id, output_mean, weight, contribution)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, sel.get("tool_name", ""), output_mean, weight, contribution),
            )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import sys
    from pathlib import Path as _P

    repo_root = _P(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="SQLite storage for prediction agent")
    sub = parser.add_subparsers(dest="cmd")

    migrate_cmd = sub.add_parser("migrate", help="Backfill paper_bets.jsonl → SQLite")
    migrate_cmd.add_argument("--paper-bets", type=Path, default=None)
    migrate_cmd.add_argument("--db", type=Path, default=None)

    query_cmd = sub.add_parser("query", help="Run a SELECT query")
    query_cmd.add_argument("sql", type=str, help="SQL SELECT statement")
    query_cmd.add_argument("--db", type=Path, default=None)

    args = parser.parse_args()

    if args.cmd == "migrate":
        n = migrate_jsonl_to_sqlite(
            paper_bets_path=args.paper_bets,
            db_path=args.db,
        )
        print(f"Migrated {n} records.")

    elif args.cmd == "query":
        store = SQLiteStore(args.db)
        rows  = store.query(args.sql)
        for row in rows:
            print(json.dumps(row, default=str))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
