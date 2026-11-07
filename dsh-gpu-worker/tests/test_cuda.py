"""CUDA forced-device semantics at the API level (must fail, never CPU fallback).

These tests need the training executor module.  They run without torch: the
expected error code is CUDA_UNAVAILABLE when ``device: "cuda"`` is forced on a
machine without CUDA (even when torch itself is missing, CUDA is unavailable).
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "worker.executors.training",
    reason="training executor module not available in this checkout yet",
)

TRAINING_CPU_CONFIG = {
    "model": {"type": "mlp", "input_dim": 4, "hidden_dims": [8], "num_classes": 2},
    "dataset": {
        "format": "synthetic",
        "num_samples": 32,
        "num_features": 4,
        "num_classes": 2,
    },
    "epochs": 1,
    "batch_size": 8,
    "device": "cuda",
}


def _cuda_available() -> bool:
    from worker import gpu

    return bool(gpu.gpu_info(refresh=True)["cuda_available"])


def test_forced_cuda_fails_with_cuda_unavailable(worker, api):
    if _cuda_available():
        pytest.skip("this machine has CUDA; the unavailable-device path cannot be exercised")

    task_id = api.submit_view("training", TRAINING_CPU_CONFIG)["task_id"]
    final = api.wait_terminal(task_id, timeout=60)
    assert final["status"] == "FAILED", final
    assert final["error"] is not None
    assert final["error"]["code"] == "CUDA_UNAVAILABLE", final["error"]
    # it must not pretend to have produced a result
    assert final["result"] is None

    result = api.result(task_id).json()
    assert result["status"] == "FAILED"
    assert result["error"]["code"] == "CUDA_UNAVAILABLE"


def test_training_without_torch_fails_explicitly(worker, api):
    from worker import gpu

    info = gpu.gpu_info(refresh=True)
    if info["torch_installed"]:
        pytest.skip("torch is installed; the torch-unavailable path cannot be exercised")
    if info["cuda_available"]:
        pytest.skip("CUDA available; auto device would try cuda")

    config = dict(TRAINING_CPU_CONFIG)
    config["device"] = "cpu"
    task_id = api.submit_view("training", config)["task_id"]
    final = api.wait_terminal(task_id, timeout=60)
    assert final["status"] == "FAILED", final
    assert final["error"]["code"] == "TORCH_UNAVAILABLE", final["error"]


@pytest.mark.parametrize("device", ["tpu", "CUDA", 0])
def test_invalid_device_rejected(worker, api, device):
    config = dict(TRAINING_CPU_CONFIG)
    config["device"] = device
    response = api.submit("training", config)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
