"""Task runner: subprocess isolation, log capture, progress, cancel, timeout.

Design:
- ``POST /tasks`` only inserts a PENDING row and creates the experiment dir.
- One daemon thread starts up to ``max_concurrent_tasks`` job subprocesses and
  reaps them.  The FastAPI process never blocks on a training run.
- Each subprocess is ``<python> -m worker.job_entry <root>/experiments/<id>/job.json``
  started in its own process group (POSIX) / via ``taskkill /T`` (Windows), so
  cancellation kills the whole tree.
- ``@@PROGRESS@@ {json}`` lines on stdout are parsed into the DB; they are never
  written to the public log file.
- Terminal state is derived from cancel -> timeout -> exit code -> result.json.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import scan_artifacts
from .config import PROJECT_DIR, Settings
from .db import Store
from .errors import conflict, forbidden
from .logging_utils import (
    PROGRESS_MARKER,
    RESULT_MARKER,
    LogWriter,
    sanitize_log_text,
)
from .models import TERMINAL_STATUSES, SubmitRequest, parse_task_config
from .paths import resolve_generic_script
from .utils import (
    canonical_json,
    relative_posix,
    sha256_text,
    utcnow,
    write_json_atomic,
)

logger = logging.getLogger("worker.runner")

# ---------------------------------------------------------------------------------
# view helpers (used by the API)
# ---------------------------------------------------------------------------------


def _loads(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:  # pragma: no cover - DB integrity issue
        return None


def row_to_task_view(row) -> dict[str, Any]:
    """Build the contract §2.4 TaskView from a DB row."""
    status = row["status"]
    terminal = status in TERMINAL_STATUSES
    return {
        "task_id": row["task_id"],
        "task_type": row["task_type"],
        "status": status,
        "name": row["name"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "timeout_seconds": row["timeout_seconds"],
        "progress": _loads(row["progress_json"]),
        "error": _loads(row["error_json"]),
        "result": _loads(row["result_json"]) if terminal else None,
        "cancel_requested": bool(row["cancel_requested"]),
        "idempotency_key": row["idempotency_key"],
        "experiment_dir": row["experiment_dir"] or f"experiments/{row['task_id']}",
    }


# ---------------------------------------------------------------------------------
# process-tree termination
# ---------------------------------------------------------------------------------


def _terminate_tree(proc: subprocess.Popen) -> None:
    """Ask the whole process tree to terminate (soft, escalates later)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":  # pragma: no cover - Windows only
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except OSError:
            pass


def _kill_tree(proc: subprocess.Popen) -> None:
    """Hard-kill the whole process tree."""
    if proc.poll() is not None:
        return
    if os.name == "nt":  # pragma: no cover - Windows only
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except OSError:
            pass


# ---------------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------------


@dataclass
class RunningTask:
    task_id: str
    proc: subprocess.Popen
    log: LogWriter
    readers: list[threading.Thread] = field(default_factory=list)
    timeout_seconds: int = 3600
    deadline: float = field(default_factory=time.monotonic)
    started_at: str = field(default_factory=utcnow)
    kill_started_at: float | None = None
    timed_out: bool = False
    result_marker: dict[str, Any] | None = None


def _child_environment() -> dict[str, str]:
    env = os.environ.copy()
    # Job scripts must never see the worker token.
    env.pop("DSH_GPU_WORKER_TOKEN", None)
    project = str(PROJECT_DIR)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = project + (os.pathsep + existing if existing else "")
    # Real-time logs: the job must not buffer stdout in a pipe.
    env["PYTHONUNBUFFERED"] = "1"
    return env


class TaskRunner:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self._running: dict[str, RunningTask] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_monotonic = time.monotonic()
        self.recovery: list[dict[str, Any]] = []

    # -- lifecycle ------------------------------------------------------------

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_monotonic

    @property
    def active_tasks(self) -> int:
        with self._lock:
            return len(self._running)

    def start(self) -> None:
        self.recovery = self.store.recover_interrupted_tasks()
        if self.recovery:
            logger.warning(
                "recovered %d interrupted task(s): %s",
                len(self.recovery),
                [entry["task_id"] for entry in self.recovery],
            )
        self._thread = threading.Thread(target=self._loop, name="task-runner", daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 15.0) -> None:
        self._stop.set()
        with self._lock:
            entries = list(self._running.values())
        for entry in entries:
            try:
                entry.log.write("stderr", "[runner] worker shutting down; terminating task")
            except Exception:  # pragma: no cover
                pass
            _terminate_tree(entry.proc)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                remaining = [entry for entry in self._running.values() if entry.proc.poll() is None]
            if not remaining:
                break
            for entry in remaining:
                _kill_tree(entry.proc)
            time.sleep(0.05)
        for entry in entries:
            self._finalize(entry, reason="shutdown")
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # -- public API -----------------------------------------------------------

    def submit(self, request: SubmitRequest) -> tuple[dict[str, Any], bool]:
        """Create a task; returns ``(TaskView, created)``.

        Idempotent replay returns the original task with ``created=False``;
        a different payload for the same key raises 409 IDEMPOTENCY_CONFLICT.
        """
        if request.task_type == "generic_python" and not self.settings.allow_generic_python:
            raise forbidden("generic_python is disabled on this worker")

        config_model = parse_task_config(request.task_type, request.config)
        config = config_model.model_dump(mode="json")
        if request.task_type == "generic_python":
            # Reject escaping script paths at submit time (400 PATH_ESCAPE)
            # instead of failing the task later.
            resolve_generic_script(self.settings.root, str(config.get("script", "")))
        payload = {
            "task_type": request.task_type,
            "config": config,
            "name": request.name,
            "timeout_seconds": request.timeout_seconds,
        }
        payload_json = canonical_json(payload)
        payload_hash = sha256_text(payload_json)

        with self._lock:
            if request.idempotency_key:
                existing = self.store.find_by_idempotency(request.idempotency_key)
                if existing is not None:
                    if existing["payload_hash"] == payload_hash:
                        return row_to_task_view(existing), False
                    raise conflict(
                        "IDEMPOTENCY_CONFLICT",
                        "idempotency_key was already used with a different payload",
                        {"task_id": existing["task_id"]},
                    )

            task_id = uuid.uuid4().hex
            created_at = utcnow()
            experiment = self.settings.experiments_dir / task_id
            experiment.mkdir(parents=True, exist_ok=True)
            self.store.create_task(
                task_id=task_id,
                task_type=request.task_type,
                name=request.name,
                idempotency_key=request.idempotency_key,
                payload_json=payload_json,
                payload_hash=payload_hash,
                config_json=canonical_json(config),
                timeout_seconds=request.timeout_seconds,
                created_at=created_at,
                experiment_dir=relative_posix(experiment, self.settings.root),
            )
            row = self.store.get_task(task_id)
        return row_to_task_view(row), True

    def cancel(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.store.get_task(task_id)
            status = row["status"]
            if status in TERMINAL_STATUSES:
                raise conflict(
                    "TASK_ALREADY_FINISHED",
                    f"Task {task_id} already finished with status {status}",
                    {"status": status},
                )
            self.store.request_cancel(task_id)
            entry = self._running.get(task_id)
            if entry is not None and entry.proc.poll() is None:
                try:
                    entry.log.write(
                        "stderr", "[runner] cancel requested; terminating process tree"
                    )
                except Exception:  # pragma: no cover
                    pass
                self._begin_kill(entry)
            elif status == "PENDING":
                self.store.finalize(
                    task_id,
                    status="CANCELLED",
                    error={
                        "code": "CANCELLED",
                        "message": "Task was cancelled before it started",
                    },
                )
            row = self.store.get_task(task_id)
        return row_to_task_view(row)

    # -- scheduling loop ------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._start_pending()
                self._poll()
            except Exception:  # pragma: no cover - defensive, keep the thread alive
                logger.exception("task runner loop iteration failed")
            self._stop.wait(self.settings.poll_seconds)

    def _start_pending(self) -> None:
        with self._lock:
            capacity = self.settings.max_concurrent_tasks - len(self._running)
            if capacity <= 0:
                return
            pending = self.store.list_pending(capacity + 8)
            for row in pending:
                if capacity <= 0:
                    break
                task_id = row["task_id"]
                if task_id in self._running:
                    continue
                if row["cancel_requested"]:
                    self.store.finalize(
                        task_id,
                        status="CANCELLED",
                        error={
                            "code": "CANCELLED",
                            "message": "Task was cancelled before it started",
                        },
                    )
                    continue
                self._start_task(row)
                capacity -= 1

    def _start_task(self, row) -> None:
        task_id = row["task_id"]
        experiment = self.settings.experiments_dir / task_id
        experiment.mkdir(parents=True, exist_ok=True)
        config = _loads(row["config_json"]) or {}
        spec = {
            "task_id": task_id,
            "task_type": row["task_type"],
            "config": config,
            "root": str(self.settings.root),
            "experiment_dir": str(experiment),
            "timeout_seconds": row["timeout_seconds"],
            "created_at": row["created_at"],
        }
        job_path = experiment / "job.json"
        write_json_atomic(job_path, spec)
        log = LogWriter(experiment / "worker.log")

        command = [sys.executable, "-u", "-m", "worker.job_entry", str(job_path)]
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(self.settings.root),
                env=_child_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=(os.name != "nt"),
            )
        except OSError as exc:
            log.write("stderr", f"[runner] failed to spawn job process: {exc}")
            log.close()
            self.store.finalize(
                task_id,
                status="FAILED",
                error={
                    "code": "SPAWN_ERROR",
                    "message": f"Failed to start job process: {exc}",
                },
            )
            return

        started_at = utcnow()
        entry = RunningTask(
            task_id=task_id,
            proc=proc,
            log=log,
            timeout_seconds=int(row["timeout_seconds"]),
            deadline=time.monotonic() + int(row["timeout_seconds"]),
            started_at=started_at,
        )
        self.store.mark_started(
            task_id,
            started_at=started_at,
            worker_pid=proc.pid,
            log_path=relative_posix(log.path, self.settings.root),
        )
        self._running[task_id] = entry
        for stream, name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
            reader = threading.Thread(
                target=self._read_stream,
                args=(entry, stream, name),
                name=f"reader-{task_id[:8]}-{name}",
                daemon=True,
            )
            entry.readers.append(reader)
            reader.start()

    # -- output capture -------------------------------------------------------

    def _read_stream(self, entry: RunningTask, stream, name: str) -> None:
        try:
            assert stream is not None
            for raw in stream:
                self._handle_line(entry, name, raw.rstrip("\n"))
        except (ValueError, OSError):  # pragma: no cover - pipe closed during shutdown
            pass
        finally:
            try:
                stream.close()
            except Exception:  # pragma: no cover
                pass

    def _handle_line(self, entry: RunningTask, name: str, line: str) -> None:
        if name == "stdout" and line.startswith(PROGRESS_MARKER):
            payload = line[len(PROGRESS_MARKER) :].strip()
            data = _loads(payload)
            if isinstance(data, dict):
                try:
                    self.store.set_progress(entry.task_id, data)
                except Exception:  # pragma: no cover - task row may be gone
                    logger.exception("failed to persist progress for %s", entry.task_id)
            return
        if name == "stdout" and line.startswith(RESULT_MARKER):
            data = _loads(line[len(RESULT_MARKER) :].strip())
            if isinstance(data, dict):
                entry.result_marker = data
            return
        try:
            entry.log.write(name, sanitize_log_text(line))
        except Exception:  # pragma: no cover
            pass

    # -- reaping / timeout ----------------------------------------------------

    def _poll(self) -> None:
        now = time.monotonic()
        with self._lock:
            entries = list(self._running.values())
        for entry in entries:
            proc = entry.proc
            if proc.poll() is not None:
                self._finalize(entry)
                continue
            if entry.kill_started_at is not None:
                if now - entry.kill_started_at >= self.settings.kill_grace_seconds:
                    _kill_tree(proc)
                continue
            if now >= entry.deadline:
                entry.timed_out = True
                try:
                    entry.log.write(
                        "stderr",
                        f"[runner] task exceeded timeout of {entry.timeout_seconds}s; "
                        "terminating process tree",
                    )
                except Exception:  # pragma: no cover
                    pass
                self._begin_kill(entry)

    def _begin_kill(self, entry: RunningTask) -> None:
        if entry.kill_started_at is None:
            entry.kill_started_at = time.monotonic()
        _terminate_tree(entry.proc)

    # -- terminal state -------------------------------------------------------

    def _finalize(self, entry: RunningTask, reason: str = "exit") -> None:
        with self._lock:
            if self._running.get(entry.task_id) is not entry:
                return
            del self._running[entry.task_id]

        for reader in entry.readers:
            reader.join(timeout=5.0)

        experiment = self.settings.experiments_dir / entry.task_id
        result_file = experiment / "result.json"
        error_file = experiment / "error.json"
        result_raw = _loads(result_file.read_text(encoding="utf-8")) if result_file.is_file() else None
        if result_raw is None and entry.result_marker is not None:
            result_raw = entry.result_marker
        error_raw = _loads(error_file.read_text(encoding="utf-8")) if error_file.is_file() else None

        try:
            row = self.store.get_task(entry.task_id)
            config = _loads(row["config_json"]) or {}
            cancel_requested = bool(row["cancel_requested"])
        except Exception:  # pragma: no cover - row missing
            config = {}
            cancel_requested = False

        return_code = entry.proc.returncode
        status: str
        error: dict[str, Any] | None = None
        result: dict[str, Any] | None = None

        if cancel_requested:
            status = "CANCELLED"
            error = {"code": "CANCELLED", "message": "Task was cancelled by request"}
        elif entry.timed_out:
            status = "FAILED"
            error = {
                "code": "TASK_TIMEOUT",
                "message": (
                    f"Task exceeded timeout of {entry.timeout_seconds} seconds and was terminated"
                ),
            }
        elif reason == "shutdown":
            status = "FAILED"
            error = {
                "code": "WORKER_SHUTDOWN",
                "message": "The worker shut down while this task was running",
            }
        elif return_code == 0 and result_raw is not None and not _has_error(error_raw, result_raw):
            status = "SUCCEEDED"
        else:
            status = "FAILED"
            if _has_error(error_raw, result_raw):
                error = _extract_error(error_raw, result_raw)
            elif return_code == 0:
                error = {
                    "code": "MISSING_RESULT",
                    "message": "Job process exited successfully but produced no result.json",
                }
            else:
                error = {
                    "code": "PROCESS_EXIT",
                    "message": f"Job process exited with code {return_code}",
                }

        if status == "SUCCEEDED" and result_raw is not None:
            result = {
                "metrics": result_raw.get("metrics") or {},
                "artifacts": scan_artifacts(experiment, entry.task_id),
                "experiment_dir": f"experiments/{entry.task_id}",
                "config": config,
                "summary": result_raw.get("summary"),
            }
        try:
            entry.log.write("stdout", f"[runner] task finished with status {status}")
            entry.log.close()
        except Exception:  # pragma: no cover
            pass

        if status == "SUCCEEDED" and result is not None:
            # rescan after the log writer is closed so artifacts are complete
            result["artifacts"] = scan_artifacts(experiment, entry.task_id)

        self.store.finalize(
            entry.task_id,
            status=status,
            error=error,
            result=result,
            finished_at=utcnow(),
        )
        logger.info("task %s finished: %s", entry.task_id, status)


def _has_error(error_raw: Any, result_raw: Any) -> bool:
    if isinstance(error_raw, dict) and isinstance(error_raw.get("error"), dict):
        return bool(error_raw["error"].get("code"))
    if isinstance(result_raw, dict) and isinstance(result_raw.get("error"), dict):
        return bool(result_raw["error"].get("code"))
    return False


def _extract_error(error_raw: Any, result_raw: Any) -> dict[str, Any]:
    if isinstance(error_raw, dict) and isinstance(error_raw.get("error"), dict):
        return error_raw["error"]
    if isinstance(result_raw, dict) and isinstance(result_raw.get("error"), dict):
        return result_raw["error"]
    return {"code": "EXECUTOR_ERROR", "message": "Unknown executor failure"}


__all__ = ["TaskRunner", "RunningTask", "row_to_task_view"]
