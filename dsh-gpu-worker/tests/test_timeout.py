"""Task timeout kills the process tree and fails with TASK_TIMEOUT."""

from __future__ import annotations

import time

TIMEOUT_SCRIPT = """
import time
print("timeout job started", flush=True)
time.sleep(120)
print("should never print", flush=True)
"""


def test_task_timeout_fails_with_task_timeout(worker, api, make_job):
    script = make_job("slow_timeout.py", TIMEOUT_SCRIPT)
    task_id = api.submit_view(
        "generic_python", {"script": script}, timeout_seconds=10
    )["task_id"]
    assert api.task_view(task_id)["timeout_seconds"] == 10

    started = time.monotonic()
    final = api.wait_terminal(task_id, timeout=45)
    elapsed = time.monotonic() - started

    assert final["status"] == "FAILED", final
    assert final["error"]["code"] == "TASK_TIMEOUT", final["error"]
    assert final["result"] is None
    assert 9.0 <= elapsed <= 40.0

    result = api.result(task_id).json()
    assert result["status"] == "FAILED"
    assert result["error"]["code"] == "TASK_TIMEOUT"

    logs = api.logs(task_id).json()
    text = "\n".join(line["text"] for line in logs["lines"])
    assert "timeout job started" in text
    assert "should never print" not in text


def test_timeout_bounds_enforced(worker, api):
    for bad in (9, 0, -5, 86401):
        response = api.submit(
            "generic_python", {"script": "jobs/hello.py"}, timeout_seconds=bad
        )
        assert response.status_code == 400, bad
        assert response.json()["error"]["code"] == "INVALID_REQUEST"
