"""Authentication: every endpoint requires a valid worker token."""

from __future__ import annotations

import pytest

from conftest import AUTH, TOKEN

PROTECTED_GETS = ["/health", "/gpu", "/version", "/tasks", "/datasets"]
PROTECTED_EXTRA = [
    ("POST", "/tasks"),
    ("GET", "/tasks/does-not-exist"),
    ("GET", "/tasks/does-not-exist/logs"),
    ("GET", "/tasks/does-not-exist/result"),
    ("GET", "/tasks/does-not-exist/artifacts"),
]


def test_health_requires_token(worker):
    response = worker["client"].get("/health")
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "UNAUTHORIZED"
    assert "message" in body["error"]
    assert isinstance(body["error"]["details"], dict)


@pytest.mark.parametrize("path", PROTECTED_GETS)
def test_get_endpoints_require_token(worker, path):
    response = worker["client"].get(path)
    assert response.status_code == 401, path
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


@pytest.mark.parametrize("method,path", PROTECTED_EXTRA)
def test_other_endpoints_require_token(worker, method, path):
    response = worker["client"].request(method, path)
    assert response.status_code == 401, (method, path)
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_wrong_token_rejected(worker):
    for headers in (
        {"Authorization": "Bearer wrong-token"},
        {"X-Worker-Token": "wrong-token"},
        {"Authorization": "Basic abc"},
    ):
        response = worker["client"].get("/health", headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_bearer_and_x_worker_token_accepted(worker):
    client = worker["client"]
    assert client.get("/health", headers=AUTH).status_code == 200
    assert client.get("/health", headers={"X-Worker-Token": TOKEN}).status_code == 200


def test_settings_refuse_missing_token(monkeypatch):
    from worker.config import ConfigurationError, Settings

    monkeypatch.delenv("DSH_GPU_WORKER_TOKEN", raising=False)
    with pytest.raises(ConfigurationError):
        Settings.from_env()

    monkeypatch.setenv("DSH_GPU_WORKER_TOKEN", "short")
    with pytest.raises(ConfigurationError):
        Settings.from_env()
