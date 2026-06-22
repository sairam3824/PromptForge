from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class RunTracker:
    """SQLite-backed store for every candidate prompt + its val score."""

    def __init__(self, db_path: str | Path = "promptforge.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT    NOT NULL,
                    optimizer   TEXT    NOT NULL,
                    iteration   INTEGER NOT NULL,
                    instruction TEXT    NOT NULL,
                    demos_json  TEXT,
                    val_score   REAL,
                    meta_json   TEXT,
                    ts          REAL    DEFAULT (unixepoch('now'))
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_run ON candidates(run_id)"
            )
            conn.commit()

    def log(
        self,
        run_id: str,
        optimizer: str,
        iteration: int,
        instruction: str,
        demos: list | None,
        val_score: float,
        metadata: dict | None = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO candidates
                    (run_id, optimizer, iteration, instruction, demos_json, val_score, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    optimizer,
                    iteration,
                    instruction,
                    json.dumps(demos) if demos is not None else None,
                    val_score,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()

    def leaderboard(self, run_id: str, top_n: int = 10) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT optimizer, iteration, instruction, demos_json, val_score, ts
                FROM candidates
                WHERE run_id = ?
                ORDER BY val_score DESC
                LIMIT ?
                """,
                (run_id, top_n),
            ).fetchall()
        return [dict(r) for r in rows]

    def best(self, run_id: str) -> dict | None:
        results = self.leaderboard(run_id, top_n=1)
        return results[0] if results else None

    def all_runs(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT run_id FROM candidates ORDER BY rowid DESC"
            ).fetchall()
        return [r[0] for r in rows]
