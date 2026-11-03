import json
import sqlite3

from datetime import datetime, timezone
from pathlib import Path


class CleaningHistory:

    def __init__(self, database_path=None):
        self.database_path = database_path or (
            Path(__file__).resolve().parent.parent
            / "datadoctor_history.db"
        )

        self._initialize()

    def _connect(self):
        return sqlite3.connect(self.database_path)

    def _initialize(self):
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS cleaning_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    sheet_name TEXT,
                    fingerprint_before TEXT NOT NULL,
                    fingerprint_after TEXT NOT NULL,
                    rows_before INTEGER NOT NULL,
                    rows_after INTEGER NOT NULL,
                    approved_issue_ids TEXT NOT NULL,
                    audit_log TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)

            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(cleaning_runs)"
                ).fetchall()
            }

            if "metadata" not in columns:
                connection.execute(
                    "ALTER TABLE cleaning_runs ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'"
                )

    def save_run(
        self,
        file_name,
        sheet_name,
        fingerprint_before,
        fingerprint_after,
        rows_before,
        rows_after,
        approved_issue_ids,
        audit_log,
        metadata=None
    ):
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO cleaning_runs (
                    created_at,
                    file_name,
                    sheet_name,
                    fingerprint_before,
                    fingerprint_after,
                    rows_before,
                    rows_after,
                    approved_issue_ids,
                    audit_log,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    file_name,
                    sheet_name,
                    fingerprint_before,
                    fingerprint_after,
                    rows_before,
                    rows_after,
                    json.dumps(sorted(approved_issue_ids)),
                    json.dumps(audit_log, default=str),
                    json.dumps(metadata or {}, default=str)
                )
            )

            return cursor.lastrowid

    def get_recent_runs(self, limit=20):
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                SELECT *
                FROM cleaning_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

            return [dict(row) for row in rows]

    def get_run(self, run_id):
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                "SELECT * FROM cleaning_runs WHERE id = ?",
                (run_id,)
            ).fetchone()

            return dict(row) if row else None