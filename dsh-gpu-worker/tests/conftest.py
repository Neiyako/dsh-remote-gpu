"""Shared pytest fixtures.

Tests run WITHOUT torch installed: training/inference success paths are covered
by executor-specific tests when torch is available, while the API-level CUDA
unavailable semantics are asserted here.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

TOKEN = "test-token-0123456789abcdef"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}


class Api:
    """Tiny HTTP helper around the TestClient."""

    def __init__(self, client: TestClient, root: Path) -> None:
        self.client = client
        self.root = root
        self.headers = dict(AUTH)

    # -- submissions ----------------------------------------------------------

    def submit(
        self,
        task_type: str,
        config: dict | None = None,
        *,
        name: str | None = None,
        timeout_seconds: int | None = None,
        idempotency_key: str | None = None,
    ):
        payload: dict = {"task_type": task_type, "config": config or {}}
        if name is not None:
            payload["name"] = name
        if timeout_seconds is not None:
            payload["timeout_seconds"] = timeout_seconds
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        return self.client.post("/tasks", json=payload, headers=self.headers)

    def submit_view(self, task_type: str, config: dict | None = None, **kwargs) -> dict:
        response = self.submit(task_type, config, **kwargs)
        assert response.status_code == 201, response.text
        return response.json()

    # -- reads ----------------------------------------------------------------

    def task(self, task_id: str):
        return self.client.get(f"/tasks/{task_id}", headers=self.headers)

    def task_view(self, task_id: str) -> dict:
        response = self.task(task_id)
        assert response.status_code == 200, response.text
        return response.json()

    def logs(self, task_id: str, offset: int = 0, limit: int = 500):
        return self.client.get(
            f"/tasks/{task_id}/logs",
            params={"offset": offset, "limit": limit},
            headers=self.headers,
        )

    def result(self, task_id: str):
        return self.client.get(f"/tasks/{task_id}/result", headers=self.headers)

    def artifacts(self, task_id: str):
        return self.client.get(f"/tasks/{task_id}/artifacts", headers=self.headers)

    # -- waiting --------------------------------------------------------------

    def wait_status(self, task_id: str, status: str, timeout: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout
        last: dict = {}
        while time.monotonic() < deadline:
            last = self.task_view(task_id)
            if last["status"] == status:
                return last
            if last["status"] in TERMINAL:
                raise AssertionError(
                    f"task {task_id} reached {last['status']} while waiting for {status}: {last}"
                )
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for {status}: last={last}")

    def wait_terminal(self, task_id: str, timeout: float = 40.0) -> dict:
        deadline = time.monotonic() + timeout
        last: dict = {}
        while time.monotonic() < deadline:
            last = self.task_view(task_id)
            if last["status"] in TERMINAL:
                return last
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for task {task_id}: last={last}")


@pytest.fixture()
def worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DSH_GPU_WORKER_TOKEN", TOKEN)
    monkeypatch.setenv("DSH_GPU_WORKER_ROOT", str(tmp_path))
    monkeypatch.delenv("DSH_GPU_WORKER_DB", raising=False)
    monkeypatch.delenv("DSH_GPU_WORKER_MAX_CONCURRENT_TASKS", raising=False)

    jobs_dir = tmp_path / "scripts" / "jobs"
    jobs_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROJECT_DIR / "scripts" / "jobs", jobs_dir, dirs_exist_ok=True)
    (tmp_path / "datasets").mkdir(exist_ok=True)

    from worker.api import create_app
    from worker.config import Settings

    settings = Settings.from_env()
    app = create_app(settings)
    with TestClient(app) as client:
        yield {
            "client": client,
            "app": app,
            "settings": settings,
            "root": tmp_path,
        }


@pytest.fixture()
def api(worker) -> Api:
    return Api(worker["client"], worker["root"])


@pytest.fixture()
def make_job(worker):
    """Write an extra whitelisted job script into the temp worker root."""

    jobs_dir = worker["root"] / "scripts" / "jobs"

    def _make(name: str, source: str) -> str:
        path = jobs_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return f"jobs/{name}"

    return _make
