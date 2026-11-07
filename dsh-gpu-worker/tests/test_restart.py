"""Restart recovery: interrupted tasks become FAILED WORKER_RESTARTED or CANCELLED."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

TOKEN = "restart-recovery-token-123"


def _seed_task(store, task_id: str, *, status: str, cancel_requested: int = 0) -> None:
    from worker.utils import canonical_json

    config = {"script": "jobs/hello.py", "args": [], "env": {}}
    store.create_task(
        task_id=task_id,
        task_type="generic_python",
        name=None,
        idempotency_key=None,
        payload_json=canonical_json(config),
        payload_hash="hash-" + task_id,
        config_json=canonical_json(config),
        timeout_seconds=3600,
        created_at="2026-01-01T00:00:00.000000Z",
        experiment_dir=f"experiments/{task_id}",
    )
    store.update_fields(task_id, status=status, cancel_requested=cancel_requested)


def test_restart_marks_interrupted_tasks(tmp_path: Path, monkeypatch):
    from worker.api import create_app
    from worker.config import Settings
    from worker.db import Store

    monkeypatch.setenv("DSH_GPU_WORKER_TOKEN", TOKEN)
    monkeypatch.setenv("DSH_GPU_WORKER_ROOT", str(tmp_path))
    settings = Settings.from_env()
    settings.ensure_directories()
    jobs = tmp_path / "scripts" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "hello.py").write_text("print('hi')\n", encoding="utf-8")

    store = Store(settings.resolved_db_path)
    _seed_task(store, "pendingtask0001", status="PENDING")
    _seed_task(store, "runningtask0001", status="RUNNING")
    _seed_task(store, "cancellingtask01", status="RUNNING", cancel_requested=1)
    store.close()

    app = create_app(settings)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        health = client.get("/health", headers=headers).json()
        assert health["recovered_tasks"] == 3

        pending = client.get("/tasks/pendingtask0001", headers=headers).json()
        assert pending["status"] == "FAILED"
        assert pending["error"]["code"] == "WORKER_RESTARTED"
        assert pending["finished_at"]

        running = client.get("/tasks/runningtask0001", headers=headers).json()
        assert running["status"] == "FAILED"
        assert running["error"]["code"] == "WORKER_RESTARTED"

        cancelling = client.get("/tasks/cancellingtask01", headers=headers).json()
        assert cancelling["status"] == "CANCELLED"
        assert cancelling["cancel_requested"] is True

        # history stays queryable after the restart
        failed = client.get("/tasks?status=FAILED", headers=headers).json()
        failed_ids = {task["task_id"] for task in failed["tasks"]}
        assert {"pendingtask0001", "runningtask0001"}.issubset(failed_ids)

        # a new task still runs on the recovered store
        response = client.post(
            "/tasks",
            json={
                "task_type": "generic_python",
                "config": {"script": "jobs/hello.py"},
            },
            headers=headers,
        )
        assert response.status_code == 201
        task_id = response.json()["task_id"]
        import time

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            view = client.get(f"/tasks/{task_id}", headers=headers).json()
            if view["status"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.02)
        assert view["status"] == "SUCCEEDED"


def test_restart_recovery_survives_reopen(tmp_path: Path, monkeypatch):
    """Restarting twice keeps terminal history intact."""
    import time

    from worker.api import create_app
    from worker.config import Settings

    monkeypatch.setenv("DSH_GPU_WORKER_TOKEN", TOKEN)
    monkeypatch.setenv("DSH_GPU_WORKER_ROOT", str(tmp_path))
    settings = Settings.from_env()
    settings.ensure_directories()
    (tmp_path / "scripts" / "jobs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "jobs" / "hello.py").write_text(
        "print('hi')\n", encoding="utf-8"
    )
    headers = {"Authorization": f"Bearer {TOKEN}"}
    task_id = None
    for _ in range(2):
        with TestClient(create_app(settings)) as client:
            if task_id is None:
                response = client.post(
                    "/tasks",
                    json={"task_type": "generic_python", "config": {"script": "jobs/hello.py"}},
                    headers=headers,
                )
                task_id = response.json()["task_id"]
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    view = client.get(f"/tasks/{task_id}", headers=headers).json()
                    if view["status"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                        break
                    time.sleep(0.02)
                assert view["status"] == "SUCCEEDED"
            else:
                again = client.get(f"/tasks/{task_id}", headers=headers).json()
                assert again["status"] == "SUCCEEDED"
                assert again["result"]["metrics"]["exit_code"] == 0
