"""Idempotency semantics for POST /tasks (contract §2.3)."""

from __future__ import annotations

GENERIC_CONFIG = {"script": "jobs/hello.py", "args": ["--x", "1"]}


def test_same_key_same_payload_returns_original(worker, api):
    first = api.submit("generic_python", GENERIC_CONFIG, idempotency_key="key-abc")
    assert first.status_code == 201
    payload = first.json()
    assert payload["idempotency_key"] == "key-abc"

    second = api.submit("generic_python", GENERIC_CONFIG, idempotency_key="key-abc")
    assert second.status_code == 200, second.text
    assert second.json()["task_id"] == payload["task_id"]

    final = api.wait_terminal(payload["task_id"])
    assert final["status"] == "SUCCEEDED"

    # replay after completion still returns the original task
    third = api.submit("generic_python", GENERIC_CONFIG, idempotency_key="key-abc")
    assert third.status_code == 200
    assert third.json()["task_id"] == payload["task_id"]
    assert third.json()["status"] == "SUCCEEDED"


def test_same_key_different_payload_conflicts(worker, api):
    first = api.submit("generic_python", GENERIC_CONFIG, idempotency_key="key-conflict")
    assert first.status_code == 201

    different = api.submit(
        "generic_python", {"script": "jobs/hello.py", "args": ["--x", "2"]},
        idempotency_key="key-conflict",
    )
    assert different.status_code == 409
    body = different.json()
    assert body["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    assert body["error"]["details"]["task_id"] == first.json()["task_id"]

    api.wait_terminal(first.json()["task_id"])


def test_different_task_type_same_key_conflicts(worker, api):
    first = api.submit("generic_python", GENERIC_CONFIG, idempotency_key="key-type")
    assert first.status_code == 201
    second = api.submit(
        "training",
        {"epochs": 1, "dataset": {"format": "synthetic", "num_samples": 16}},
        idempotency_key="key-type",
    )
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    api.wait_terminal(first.json()["task_id"])


def test_without_key_each_submit_is_new(worker, api):
    first = api.submit("generic_python", GENERIC_CONFIG)
    second = api.submit("generic_python", GENERIC_CONFIG)
    assert first.status_code == 201 and second.status_code == 201
    assert first.json()["task_id"] != second.json()["task_id"]
    api.wait_terminal(first.json()["task_id"])
    api.wait_terminal(second.json()["task_id"])
