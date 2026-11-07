"""generic_python executor: success path, rejection rules, token isolation."""

from __future__ import annotations

import hashlib
import json

import pytest

from conftest import AUTH


def test_generic_job_succeeds(worker, api):
    response = api.submit(
        "generic_python",
        {"script": "jobs/hello.py", "args": ["--x", "1"], "env": {"FOO": "bar"}},
        name="hello job",
    )
    assert response.status_code == 201, response.text
    view = response.json()
    assert view["status"] in {"PENDING", "RUNNING"}
    assert view["name"] == "hello job"
    assert view["task_id"]

    final = api.wait_terminal(view["task_id"])
    assert final["status"] == "SUCCEEDED", final
    assert final["error"] is None
    assert final["started_at"] and final["finished_at"]

    # logs are persisted and readable
    logs = api.logs(view["task_id"]).json()
    text = "\n".join(line["text"] for line in logs["lines"])
    assert "hello from scripts/jobs/hello.py" in text
    assert "bar" in text  # FOO env passed through
    assert "wrote hello.json and hello.txt" in text
    assert "token_visible" in text and '"token_visible": false' in text

    # result payload
    result = api.result(view["task_id"]).json()
    assert result["status"] == "SUCCEEDED"
    assert result["error"] is None
    payload = result["result"]
    assert payload["metrics"]["exit_code"] == 0
    assert payload["metrics"]["script"] == "jobs/hello.py"
    assert payload["experiment_dir"] == f"experiments/{view['task_id']}"
    assert payload["config"]["args"] == ["--x", "1"]
    assert payload["summary"]

    # artifacts incl. files the job created in its experiment dir
    artifacts = {item["path"]: item for item in api.artifacts(view["task_id"]).json()["artifacts"]}
    for expected in ("hello.txt", "hello.json", "metrics.json", "result.json", "worker.log"):
        assert expected in artifacts, sorted(artifacts)

    hello = artifacts["hello.txt"]
    download = worker["client"].get(hello["download_url"], headers=AUTH)
    assert download.status_code == 200
    assert download.content == b"hello world\n"
    assert hashlib.sha256(download.content).hexdigest() == hello["sha256"]


def test_generic_job_failure_reports_script_error(worker, api):
    response = api.submit("generic_python", {"script": "jobs/hello.py", "args": ["--fail"]})
    assert response.status_code == 201
    final = api.wait_terminal(response.json()["task_id"])
    assert final["status"] == "FAILED"
    assert final["error"]["code"] == "SCRIPT_FAILED"
    assert "exited with code 3" in final["error"]["message"]

    result = api.result(response.json()["task_id"]).json()
    assert result["status"] == "FAILED"
    assert result["error"]["code"] == "SCRIPT_FAILED"
    assert result["result"] is None


def test_generic_env_cannot_override_pythonpath(worker, api):
    response = api.submit(
        "generic_python",
        {"script": "jobs/hello.py", "env": {"PYTHONPATH": "/tmp/evil"}},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_generic_args_must_be_strings(worker, api):
    response = api.submit("generic_python", {"script": "jobs/hello.py", "args": [1, 2]})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_generic_missing_script_file_fails(worker, api):
    response = api.submit("generic_python", {"script": "jobs/does_not_exist.py"})
    assert response.status_code == 201
    final = api.wait_terminal(response.json()["task_id"])
    assert final["status"] == "FAILED"
    assert final["error"]["code"] == "SCRIPT_NOT_FOUND"


@pytest.mark.parametrize(
    "script",
    [
        "../hello.py",
        "jobs/../../hello.py",
        "/etc/passwd",
        "jobs/hello.txt",
        "jobs/hello",
        "..\\..\\hello.py",
    ],
)
def test_generic_script_path_escape_rejected(worker, api, script):
    response = api.submit("generic_python", {"script": script})
    assert response.status_code == 400, (script, response.text)
    assert response.json()["error"]["code"] == "PATH_ESCAPE"
