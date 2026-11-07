"""Path traversal must yield 400 PATH_ESCAPE, never a file outside the jail."""

from __future__ import annotations

import urllib.parse

import pytest

from conftest import AUTH

TRAVERSAL_PATHS = [
    "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    "..%2f..%2fetc%2fpasswd",
    "%2e%2e%5c%2e%2e%5cwindows%5cwin.ini",
    "..%2F..%2F..%2F..%2Fetc%2Fhosts",
    "%2Fetc%2Fpasswd",
    "%2e%2e/evil.txt",
]


@pytest.fixture()
def finished_task(api):
    task_id = api.submit_view("generic_python", {"script": "jobs/hello.py"})["task_id"]
    assert api.wait_terminal(task_id)["status"] == "SUCCEEDED"
    return task_id


@pytest.mark.parametrize("path", TRAVERSAL_PATHS)
def test_artifact_download_path_escape(worker, finished_task, path):
    url = f"/tasks/{finished_task}/artifacts/{path}"
    response = worker["client"].get(url, headers=AUTH)
    assert response.status_code == 400, (path, response.status_code, response.text)
    assert response.json()["error"]["code"] == "PATH_ESCAPE"


def test_artifact_download_encoded_dotdot(worker, finished_task):
    encoded = urllib.parse.quote("../worker.db", safe="")
    response = worker["client"].get(
        f"/tasks/{finished_task}/artifacts/{encoded}", headers=AUTH
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "PATH_ESCAPE"


def test_artifact_download_engineered_symlink_escape(worker, finished_task):
    """A symlink inside the experiment dir must not expose outside files."""
    experiment = worker["root"] / "experiments" / finished_task
    target = worker["root"] / "secret.txt"
    target.write_text("top secret", encoding="utf-8")
    link = experiment / "leak.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this filesystem")

    response = worker["client"].get(
        f"/tasks/{finished_task}/artifacts/leak.txt", headers=AUTH
    )
    # resolve() follows the symlink -> outside experiment dir -> PATH_ESCAPE
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "PATH_ESCAPE"


def test_unknown_task_cannot_be_probed(worker):
    response = worker["client"].get("/tasks/../../etc/passwd", headers=AUTH)
    assert response.status_code in {404, 400}
