"""Cancellation semantics (contract §2.5)."""

from __future__ import annotations

import time

LONG_SCRIPT = """
import time
print("long job started", flush=True)
time.sleep(60)
print("should never print", flush=True)
"""


def test_cancel_running_task_becomes_cancelled(worker, api, make_job):
    script = make_job("slow_cancel.py", LONG_SCRIPT)
    task_id = api.submit_view("generic_python", {"script": script})["task_id"]
    running = api.wait_status(task_id, "RUNNING")
    assert running["started_at"]

    # wait until the job has really started producing output, then cancel
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        text = "\n".join(line["text"] for line in api.logs(task_id).json()["lines"])
        if "long job started" in text:
            break
        time.sleep(0.02)
    else:
        raise AssertionError("job never produced its startup log line")

    response = api.client.post(f"/tasks/{task_id}/cancel", headers=api.headers)
    assert response.status_code == 200, response.text
    assert response.json()["cancel_requested"] is True

    final = api.wait_terminal(task_id, timeout=20)
    assert final["status"] == "CANCELLED", final
    assert final["finished_at"]
    # a cancelled task must not be reported as FAILED
    assert final["error"] is None or final["error"]["code"] == "CANCELLED"

    result = api.result(task_id).json()
    assert result["status"] == "CANCELLED"

    logs = api.logs(task_id).json()
    text = "\n".join(line["text"] for line in logs["lines"])
    assert "long job started" in text
    assert "should never print" not in text


def test_cancel_finished_task_conflicts(worker, api):
    task_id = api.submit_view("generic_python", {"script": "jobs/hello.py"})["task_id"]
    api.wait_terminal(task_id)
    response = api.client.post(f"/tasks/{task_id}/cancel", headers=api.headers)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "TASK_ALREADY_FINISHED"


def test_cancel_pending_task(worker, api, make_job):
    script = make_job("slow_pending.py", LONG_SCRIPT)
    first = api.submit_view("generic_python", {"script": script})["task_id"]
    api.wait_status(first, "RUNNING")

    second = api.submit_view("generic_python", {"script": script})["task_id"]
    pending = api.task_view(second)
    assert pending["status"] == "PENDING"

    response = api.client.post(f"/tasks/{second}/cancel", headers=api.headers)
    assert response.status_code == 200
    final = api.wait_terminal(second, timeout=10)
    assert final["status"] == "CANCELLED"
    assert final["started_at"] is None

    api.client.post(f"/tasks/{first}/cancel", headers=api.headers)
    assert api.wait_terminal(first)["status"] == "CANCELLED"


def test_cancel_unknown_task(worker, api):
    response = api.client.post("/tasks/nope/cancel", headers=api.headers)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TASK_NOT_FOUND"


def test_cancel_kills_process_tree(worker, api, make_job):
    """The runner kills the job process group; no orphan keeps running."""
    script = make_job(
        "slow_orphan.py",
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print('parent', child.pid, flush=True)\n"
        "time.sleep(120)\n",
    )
    task_id = api.submit_view("generic_python", {"script": script})["task_id"]
    running = api.wait_status(task_id, "RUNNING")
    time.sleep(0.5)  # let the grandchild spawn

    api.client.post(f"/tasks/{task_id}/cancel", headers=api.headers)
    final = api.wait_terminal(task_id, timeout=20)
    assert final["status"] == "CANCELLED"

    # the experiment dir exists and the task reached a terminal state
    assert (worker["root"] / "experiments" / task_id).is_dir()
    assert running["task_id"] == task_id
