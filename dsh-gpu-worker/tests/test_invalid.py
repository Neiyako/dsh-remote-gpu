"""Invalid requests, unknown endpoints, listing and datasets."""

from __future__ import annotations

import pytest

from conftest import AUTH


@pytest.mark.parametrize(
    "payload",
    [
        {"task_type": "shell", "config": {}},
        {"task_type": "training", "config": {}, "timeout_seconds": 5},
        {"task_type": "training", "config": {}, "timeout_seconds": 90000},
        {"task_type": "generic_python", "config": {}},
        {"task_type": "generic_python", "config": {"script": "jobs/hello.py", "args": "nope"}},
        {"task_type": "generic_python", "config": {"script": "jobs/hello.py", "env": {"lower": "x"}}},
        {"task_type": "generic_python", "config": {"script": "jobs/hello.py", "env": {"PATH": "/evil"}}},
        {"task_type": "training", "config": {"epochs": "many"}},
        {"task_type": "inference", "config": {}},
        {"nope": True},
    ],
)
def test_invalid_submissions_are_400(worker, payload):
    response = worker["client"].post("/tasks", json=payload, headers=AUTH)
    assert response.status_code == 400, (payload, response.text)
    body = response.json()
    assert body["error"]["code"] == "INVALID_REQUEST"
    assert "details" in body["error"]


def test_unknown_task_endpoints_404(worker, api):
    for suffix in ("", "/logs", "/result", "/artifacts"):
        response = worker["client"].get(f"/tasks/unknown-task{suffix}", headers=AUTH)
        assert response.status_code == 404, suffix
        assert response.json()["error"]["code"] == "TASK_NOT_FOUND"


def test_unknown_route_unified_error(worker):
    response = worker["client"].get("/nope", headers=AUTH)
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "ROUTE_NOT_FOUND"
    assert isinstance(body["error"]["details"], dict)


def test_wrong_method_unified_error(worker):
    response = worker["client"].post("/health", headers=AUTH)
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_list_tasks_filtering_and_limit(worker, api):
    first = api.submit_view("generic_python", {"script": "jobs/hello.py"})["task_id"]
    api.wait_terminal(first)

    response = worker["client"].get("/tasks?limit=10", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["tasks"])
    assert first in {task["task_id"] for task in body["tasks"]}

    succeeded = worker["client"].get("/tasks?status=SUCCEEDED", headers=AUTH).json()
    assert all(task["status"] == "SUCCEEDED" for task in succeeded["tasks"])

    bad = worker["client"].get("/tasks?status=NOPE", headers=AUTH)
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "INVALID_REQUEST"

    bad_limit = worker["client"].get("/tasks?limit=0", headers=AUTH)
    assert bad_limit.status_code == 400


def test_logs_query_validation(worker, api):
    task_id = api.submit_view("generic_python", {"script": "jobs/hello.py"})["task_id"]
    api.wait_terminal(task_id)
    assert worker["client"].get(
        f"/tasks/{task_id}/logs?limit=0", headers=AUTH
    ).status_code == 400
    assert worker["client"].get(
        f"/tasks/{task_id}/logs?limit=5000", headers=AUTH
    ).status_code == 400
    assert worker["client"].get(
        f"/tasks/{task_id}/logs?offset=-1", headers=AUTH
    ).status_code == 400


def test_datasets_endpoint_empty(worker):
    response = worker["client"].get("/datasets", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["datasets"] == []
    assert body["count"] == 0


def test_datasets_endpoint_lists_files(worker):
    dataset_dir = worker["root"] / "datasets" / "toy"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.csv").write_text("x1,x2,target\n1,2,0\n", encoding="utf-8")

    body = worker["client"].get("/datasets", headers=AUTH).json()
    assert body["count"] == 1
    entry = body["datasets"][0]
    assert entry["dataset_id"] == "toy"
    assert entry["formats"] == ["csv"]
    assert entry["files"][0]["name"] == "data.csv"
    assert entry["files"][0]["size_bytes"] > 0


def test_submit_returns_immediately(worker, api, make_job):
    """Submission must not block until the job completes."""
    import time

    script = make_job(
        "slow_submit.py",
        "import time\nprint('started', flush=True)\ntime.sleep(5)\n",
    )
    started = time.monotonic()
    response = api.submit("generic_python", {"script": script})
    elapsed = time.monotonic() - started
    assert response.status_code == 201
    assert elapsed < 2.0, f"submit blocked for {elapsed:.2f}s"
    task_id = response.json()["task_id"]
    assert api.task_view(task_id)["status"] in {"PENDING", "RUNNING"}
    api.client.post(f"/tasks/{task_id}/cancel", headers=api.headers)
    api.wait_terminal(task_id)
