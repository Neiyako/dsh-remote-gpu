"""SQLite persistence for tasks (contract §4).

A single connection is shared between the FastAPI request threads and the task
manager thread, guarded by a re-entrant lock.  WAL mode keeps reads cheap.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Iterable

from .errors import task_not_found
from .models import TERMINAL_STATUSES
from .utils import utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id          TEXT PRIMARY KEY,
    task_type        TEXT NOT NULL,
    status           TEXT NOT NULL,
    name             TEXT,
    idempotency_key  TEXT,
    payload_json     TEXT NOT NULL,
    payload_hash     TEXT,
    config_json      TEXT,
    progress_json    TEXT,
    result_json      TEXT,
    error_json       TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT,
    timeout_seconds  INTEGER NOT NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    worker_pid       INTEGER,
    log_path         TEXT,
    experiment_dir   TEXT,
    updated_at       TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_idempotency
    ON tasks(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_ALLOWED_UPDATE_COLUMNS = frozenset(
    {
        "status",
        "progress_json",
        "result_json",
        "error_json",
        "started_at",
        "finished_at",
        "cancel_requested",
        "worker_pid",
        "log_path",
        "updated_at",
    }
)


class Store:
    """Thread-safe task store."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()

    # -- meta -----------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def worker_id(self) -> str:
        existing = self.get_meta("worker_id")
        if existing:
            return existing
        created = f"worker-{uuid.uuid4().hex[:12]}"
        self.set_meta("worker_id", created)
        # A freshly created id is only authoritative if nobody raced us to it.
        return self.get_meta("worker_id") or created

    # -- create / read --------------------------------------------------------

    def create_task(
        self,
        *,
        task_id: str,
        task_type: str,
        name: str | None,
        idempotency_key: str | None,
        payload_json: str,
        payload_hash: str,
        config_json: str,
        timeout_seconds: int,
        created_at: str,
        experiment_dir: str,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO tasks (
                    task_id, task_type, status, name, idempotency_key,
                    payload_json, payload_hash, config_json, timeout_seconds,
                    created_at, updated_at, experiment_dir, cancel_requested
                ) VALUES (?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    task_id,
                    task_type,
                    name,
                    idempotency_key,
                    payload_json,
                    payload_hash,
                    config_json,
                    timeout_seconds,
                    created_at,
                    created_at,
                    experiment_dir,
                ),
            )

    def get_task(self, task_id: str) -> sqlite3.Row:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise task_not_found(task_id)
        return row

    def find_by_idempotency(self, key: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM tasks WHERE idempotency_key = ?", (key,)
            ).fetchone()

    def list_tasks(self, status: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
        query = "SELECT * FROM tasks"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return list(self._conn.execute(query, params).fetchall())

    def list_pending(self, limit: int) -> list[sqlite3.Row]:
        with self._lock:
            return list(
                self._conn.execute(
                    "SELECT * FROM tasks WHERE status = 'PENDING' "
                    "ORDER BY created_at ASC, rowid ASC LIMIT ?",
                    (limit,),
                ).fetchall()
            )

    def count_by_status(self, status: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM tasks WHERE status = ?", (status,)
            ).fetchone()
        return int(row["n"])

    def running_task_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT task_id FROM tasks WHERE status = 'RUNNING'"
            ).fetchall()
        return [row["task_id"] for row in rows]

    # -- update ---------------------------------------------------------------

    def update_fields(self, task_id: str, **fields: Any) -> None:
        unknown = set(fields) - _ALLOWED_UPDATE_COLUMNS
        if unknown:
            raise ValueError(f"refusing to update unknown columns: {unknown}")
        if not fields:
            return
        fields.setdefault("updated_at", utcnow())
        assignments = ", ".join(f"{column} = ?" for column in fields)
        values = list(fields.values()) + [task_id]
        with self._lock, self._conn:
            cursor = self._conn.execute(
                f"UPDATE tasks SET {assignments} WHERE task_id = ?", values
            )
        if cursor.rowcount == 0:
            raise task_not_found(task_id)

    def set_progress(self, task_id: str, progress: dict[str, Any]) -> None:
        now = utcnow()
        enriched = {**progress, "updated_at": now}
        self.update_fields(task_id, progress_json=json.dumps(enriched, ensure_ascii=False))

    def mark_started(self, task_id: str, *, started_at: str, worker_pid: int, log_path: str) -> None:
        self.update_fields(
            task_id,
            status="RUNNING",
            started_at=started_at,
            worker_pid=worker_pid,
            log_path=log_path,
            updated_at=started_at,
        )

    def request_cancel(self, task_id: str) -> None:
        self.update_fields(task_id, cancel_requested=1)

    def finalize(
        self,
        task_id: str,
        *,
        status: str,
        error: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        finished_at: str | None = None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"not a terminal status: {status}")
        self.update_fields(
            task_id,
            status=status,
            error_json=json.dumps(error, ensure_ascii=False) if error is not None else None,
            result_json=json.dumps(result, ensure_ascii=False) if result is not None else None,
            finished_at=finished_at or utcnow(),
        )

    # -- restart recovery -----------------------------------------------------

    def recover_interrupted_tasks(self) -> list[dict[str, Any]]:
        """Mark interrupted tasks on startup; never report them as SUCCEEDED."""
        recovered: list[dict[str, Any]] = []
        now = utcnow()
        with self._lock, self._conn:
            rows = list(
                self._conn.execute(
                    "SELECT task_id, status, cancel_requested FROM tasks "
                    "WHERE status IN ('PENDING', 'RUNNING')"
                ).fetchall()
            )
            for row in rows:
                task_id = row["task_id"]
                if row["cancel_requested"]:
                    status = "CANCELLED"
                    error = {
                        "code": "CANCELLED",
                        "message": "Task was cancel-requested when the worker restarted",
                    }
                else:
                    status = "FAILED"
                    error = {
                        "code": "WORKER_RESTARTED",
                        "message": (
                            "The worker process restarted while this task was "
                            f"{row['status']}; it cannot be resumed and is reported as FAILED."
                        ),
                    }
                self._conn.execute(
                    "UPDATE tasks SET status = ?, error_json = ?, finished_at = ?, "
                    "updated_at = ?, cancel_requested = ? WHERE task_id = ?",
                    (
                        status,
                        json.dumps(error, ensure_ascii=False),
                        now,
                        now,
                        row["cancel_requested"],
                        task_id,
                    ),
                )
                recovered.append(
                    {
                        "task_id": task_id,
                        "previous_status": row["status"],
                        "status": status,
                        "error_code": error["code"],
                    }
                )
            self._conn.commit()
        return recovered


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [row_to_dict(row) for row in rows]
