"""Health / version endpoints."""

from __future__ import annotations

from conftest import AUTH


def test_health_payload(worker):
    response = worker["client"].get("/health", headers=AUTH)
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert isinstance(body["worker_id"], str) and body["worker_id"]
    assert body["version"] == "0.1.0"
    assert body["time"].endswith("Z")
    assert body["uptime_seconds"] >= 0
    assert isinstance(body["cuda_available"], bool)
    assert body["active_tasks"] == 0
    assert body["queued_tasks"] == 0
    assert body["max_concurrent_tasks"] == 1


def test_health_stays_ok_without_cuda(worker):
    body = worker["client"].get("/health", headers=AUTH).json()
    if not body["cuda_available"]:
        # contract §2.1: status remains ok, cuda_available is reported honestly
        assert body["status"] == "ok"


def test_version_endpoint(worker):
    body = worker["client"].get("/version", headers=AUTH).json()
    assert body["name"] == "dsh-gpu-worker"
    assert body["version"] == "0.1.0"
    assert body["worker_id"]
    assert body["time"].endswith("Z")


def test_worker_id_is_stable(worker):
    first = worker["client"].get("/health", headers=AUTH).json()["worker_id"]
    second = worker["client"].get("/health", headers=AUTH).json()["worker_id"]
    assert first == second
