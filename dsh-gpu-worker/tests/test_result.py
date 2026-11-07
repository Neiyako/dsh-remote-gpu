"""Result endpoint semantics (contract §2.7)."""

from __future__ import annotations

LONG_SCRIPT = """
import time
print("started", flush=True)
time.sleep(60)
"""


def test_result_not_finished_returns_409(worker, api, make_job):
    script = make_job("slow_result.py", LONG_SCRIPT)
    task_id = api.submit_view("generic_python", {"script": script})["task_id"]
    api.wait_status(task_id, "RUNNING")

    response = api.result(task_id)
    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "TASK_NOT_FINISHED"
    assert body["error"]["details"]["status"] == "RUNNING"

    api.client.post(f"/tasks/{task_id}/cancel", headers=api.headers)
    api.wait_terminal(task_id)


def test_result_structure_after_success(worker, api):
    task_id = api.submit_view("generic_python", {"script": "jobs/hello.py"})["task_id"]
    final = api.wait_terminal(task_id)
    assert final["status"] == "SUCCEEDED", final
    assert final["result"] is not None
    assert final["result"]["metrics"]["exit_code"] == 0
    assert final["result"]["experiment_dir"] == f"experiments/{task_id}"

    response = api.result(task_id)
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == task_id
    assert body["status"] == "SUCCEEDED"
    assert body["error"] is None
    result = body["result"]
    for key in ("metrics", "artifacts", "experiment_dir", "config", "summary"):
        assert key in result, key
    assert isinstance(result["artifacts"], list) and result["artifacts"]
    for item in result["artifacts"]:
        assert set(item) >= {"path", "size_bytes", "sha256", "download_url"}
    assert result["config"]["script"] == "jobs/hello.py"


def test_result_failed_task_returns_200_with_error(worker, api):
    task_id = api.submit_view(
        "generic_python", {"script": "jobs/hello.py", "args": ["--fail"]}
    )["task_id"]
    final = api.wait_terminal(task_id)
    assert final["status"] == "FAILED"

    response = api.result(task_id)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAILED"
    assert body["result"] is None
    assert body["error"]["code"] == "SCRIPT_FAILED"


def test_result_unknown_task(worker, api):
    response = api.result("no-such-task")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TASK_NOT_FOUND"


def test_task_progress_cleared_shape(worker, api):
    task_id = api.submit_view("generic_python", {"script": "jobs/hello.py"})["task_id"]
    final = api.wait_terminal(task_id)
    # tasks without epoch progress simply have no progress object
    assert final["progress"] is None or isinstance(final["progress"], dict)
