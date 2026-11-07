"""Artifacts listing and safe download (contract §2.8 / §2.9)."""

from __future__ import annotations

import hashlib

from conftest import AUTH


def test_artifacts_list_and_download(worker, api):
    task_id = api.submit_view("generic_python", {"script": "jobs/hello.py"})["task_id"]
    assert api.wait_terminal(task_id)["status"] == "SUCCEEDED"

    response = api.artifacts(task_id)
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == task_id
    artifacts = body["artifacts"]
    assert artifacts
    for item in artifacts:
        assert item["path"] == item["path"].replace("\\", "/")
        assert item["size_bytes"] >= 0
        assert len(item["sha256"]) == 64
        assert item["download_url"] == f"/tasks/{task_id}/artifacts/{item['path']}"

    listed = {item["path"] for item in artifacts}
    assert {"hello.txt", "hello.json", "result.json", "worker.log"}.issubset(listed)

    for path in ("hello.txt", "hello.json"):
        item = next(entry for entry in artifacts if entry["path"] == path)
        download = worker["client"].get(item["download_url"], headers=AUTH)
        assert download.status_code == 200
        assert "attachment" in download.headers.get("content-disposition", "")
        assert len(download.content) == item["size_bytes"]
        assert hashlib.sha256(download.content).hexdigest() == item["sha256"]


def test_artifacts_unknown_task(worker, api):
    response = api.artifacts("missing-task-id")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TASK_NOT_FOUND"


def test_artifact_missing_file(worker, api):
    task_id = api.submit_view("generic_python", {"script": "jobs/hello.py"})["task_id"]
    api.wait_terminal(task_id)
    response = worker["client"].get(
        f"/tasks/{task_id}/artifacts/nope.bin", headers=AUTH
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ARTIFACT_NOT_FOUND"


def test_artifact_download_unknown_task(worker, api):
    response = worker["client"].get("/tasks/nope/artifacts/hello.txt", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "TASK_NOT_FOUND"


def test_artifact_directories_are_not_listed(worker, api):
    task_id = api.submit_view("generic_python", {"script": "jobs/hello.py"})["task_id"]
    api.wait_terminal(task_id)
    experiment = worker["root"] / "experiments" / task_id
    (experiment / "subdir").mkdir()
    (experiment / "subdir" / "nested.txt").write_text("nested", encoding="utf-8")

    artifacts = api.artifacts(task_id).json()["artifacts"]
    paths = {item["path"] for item in artifacts}
    assert "subdir/nested.txt" in paths
    assert "subdir" not in paths
