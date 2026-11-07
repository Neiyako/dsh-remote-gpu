"""GET /gpu contract shape (works with and without torch)."""

from __future__ import annotations

from conftest import AUTH


def test_gpu_payload_shape(worker):
    response = worker["client"].get("/gpu", headers=AUTH)
    assert response.status_code == 200
    body = response.json()

    for key in ("cuda_available", "device_count", "devices", "nvidia_smi_available", "error"):
        assert key in body, key
    assert isinstance(body["cuda_available"], bool)
    assert isinstance(body["device_count"], int)
    assert isinstance(body["devices"], list)
    assert body["device_count"] == len(body["devices"])

    if not body["cuda_available"]:
        assert body["device_count"] == 0
        assert body["devices"] == []
        assert body["error"], "CUDA-unavailable payload must explain why"


def test_gpu_device_fields_when_present(worker):
    body = worker["client"].get("/gpu", headers=AUTH).json()
    for device in body["devices"]:
        for key in (
            "index",
            "name",
            "capability",
            "total_memory_mb",
            "free_memory_mb",
            "used_memory_mb",
            "utilization_percent",
            "temperature_c",
            "torch_version",
            "driver_version",
        ):
            assert key in device, key
