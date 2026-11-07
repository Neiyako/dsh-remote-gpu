"""Logs endpoint: pagination, persistence, internal marker hiding."""

from __future__ import annotations

import json

NOISY_SCRIPT = """
import sys
for index in range(12):
    print(f"line-{index:02d}")
print("@@PROGRESS@@ " + '{"epoch": 1, "total_epochs": 2, "train_loss": 0.5}')
print("@@RESULT@@ " + '{"ok": false, "code": "LEAKED"}')
print("last-line")
"""


def test_logs_offset_and_limit(worker, api, make_job):
    script = make_job("noisy.py", NOISY_SCRIPT)
    task_id = api.submit_view("generic_python", {"script": script})["task_id"]
    final = api.wait_terminal(task_id)
    assert final["status"] == "SUCCEEDED", final

    first = api.logs(task_id, offset=0, limit=5).json()
    assert first["offset"] == 0
    assert len(first["lines"]) == 5
    assert first["next_offset"] == 5
    assert first["eof"] is False
    seqs = [line["seq"] for line in first["lines"]]
    assert seqs == sorted(seqs)
    assert all(line["stream"] in {"stdout", "stderr", "runner"} for line in first["lines"])

    rest = api.logs(task_id, offset=first["next_offset"], limit=2000).json()
    assert len(rest["lines"]) >= 8
    assert rest["eof"] is True

    all_text = "\n".join(line["text"] for line in first["lines"] + rest["lines"])
    assert "line-00" in all_text and "line-11" in all_text and "last-line" in all_text


def test_logs_offset_past_end(worker, api):
    task_id = api.submit_view("generic_python", {"script": "jobs/hello.py"})["task_id"]
    api.wait_terminal(task_id)
    page = api.logs(task_id, offset=100000, limit=10).json()
    assert page["lines"] == []
    assert page["eof"] is True


def test_logs_markers_never_leak(worker, api, make_job):
    script = make_job("markers.py", NOISY_SCRIPT)
    task_id = api.submit_view("generic_python", {"script": script})["task_id"]
    final = api.wait_terminal(task_id)
    assert final["status"] == "SUCCEEDED", final

    # progress marker was parsed into the DB, not logged
    assert final["progress"]["epoch"] == 1
    assert final["progress"]["total_epochs"] == 2
    assert final["progress"]["updated_at"].endswith("Z")

    logs = api.logs(task_id, limit=2000).json()
    text = "\n".join(line["text"] for line in logs["lines"])
    assert "@@PROGRESS@@" not in text
    assert "@@RESULT@@" not in text
    assert "LEAKED" not in text


def test_logs_persist_after_restart(tmp_path, monkeypatch):
    """The log file lives on disk under the experiment directory."""
    from fastapi.testclient import TestClient

    from worker.api import create_app
    from worker.config import Settings

    monkeypatch.setenv("DSH_GPU_WORKER_TOKEN", "restart-token-1234")
    monkeypatch.setenv("DSH_GPU_WORKER_ROOT", str(tmp_path))
    jobs = tmp_path / "scripts" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "hello.py").write_text("print('hello file')\n", encoding="utf-8")

    settings = Settings.from_env()
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/tasks",
            json={"task_type": "generic_python", "config": {"script": "jobs/hello.py"}},
            headers={"Authorization": "Bearer restart-token-1234"},
        )
        task_id = response.json()["task_id"]
        import time

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            view = client.get(
                f"/tasks/{task_id}", headers={"Authorization": "Bearer restart-token-1234"}
            ).json()
            if view["status"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.02)
        assert view["status"] == "SUCCEEDED"

    log_file = tmp_path / "experiments" / task_id / "worker.log"
    assert log_file.is_file()
    records = [json.loads(line) for line in log_file.read_text().splitlines() if line]
    assert any("hello file" in record["text"] for record in records)
