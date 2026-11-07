"""Torch-dependent helpers shared by the training and inference executors.

This module is imported lazily: generic-python tasks never import numpy/torch.
"""

from __future__ import annotations

import csv
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..errors import ExecutorError
from ..models import DatasetSpec, ModelSpec
from ..paths import ensure_under
from ..utils import sha256_file


# ---------------------------------------------------------------------------------
# torch / device
# ---------------------------------------------------------------------------------


def require_torch():
    try:
        import torch  # noqa: PLC0415 - optional dependency by design
    except Exception as exc:  # pragma: no cover - depends on host environment
        raise ExecutorError(
            "TORCH_UNAVAILABLE",
            "PyTorch is required for training/inference tasks but is not installed "
            f"({exc}). Install it on the GPU worker, e.g. `pip install torch`.",
        ) from exc
    return torch


def resolve_device(torch, requested: str) -> str:
    cuda_ok = bool(torch.cuda.is_available())
    if requested == "cuda":
        if not cuda_ok:
            raise ExecutorError(
                "CUDA_UNAVAILABLE",
                "device='cuda' was requested but torch.cuda.is_available() is false. "
                "The worker refuses to silently fall back to CPU.",
            )
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cuda" if cuda_ok else "cpu"


def seed_everything(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _nvidia_smi_metrics() -> dict[str, Any]:
    """Fallback for drivers where pynvml is unavailable (common on Windows)."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        first = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        parts = [part.strip() for part in first.split(",")]
        if len(parts) < 2:
            return {}
        return {
            "gpu_utilization_percent": float(parts[0]),
            "gpu_memory_used_mb": round(float(parts[1]), 1),
        }
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return {}


def gpu_metrics(torch, device: str) -> dict[str, Any]:
    """Best-effort live GPU utilization/memory for progress payloads.

    Preference order: torch runtime -> pynvml -> ``nvidia-smi``. The last one
    matters on Windows, where pynvml is often not installed but nvidia-smi is.
    """
    if not device.startswith("cuda"):
        return {}
    metrics: dict[str, Any] = {}
    try:
        metrics["gpu_memory_used_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 1)
    except Exception:
        metrics["gpu_memory_used_mb"] = None
    utilization = None
    try:
        utilization = float(torch.cuda.utilization())
    except Exception:
        try:  # pragma: no cover - depends on nvml availability
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            utilization = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            pynvml.nvmlShutdown()
        except Exception:
            utilization = None
    if utilization is None or metrics.get("gpu_memory_used_mb") is None:
        smi = _nvidia_smi_metrics()
        if utilization is None:
            utilization = smi.get("gpu_utilization_percent")
        if metrics.get("gpu_memory_used_mb") is None:
            metrics["gpu_memory_used_mb"] = smi.get("gpu_memory_used_mb")
    metrics["gpu_utilization_percent"] = utilization
    return metrics


def reproducibility(
    torch,
    device: str,
    seed: int,
    started_at: str,
    finished_at: str | None = None,
) -> dict[str, Any]:
    cudnn_version = None
    try:
        if torch.backends.cudnn.is_available():
            cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None
    device_name = None
    if device.startswith("cuda"):
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = None
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": cudnn_version,
        "device": device,
        "device_name": device_name,
        "seed": seed,
        "started_at": started_at,
        "finished_at": finished_at,
        "python_executable": sys.executable,
    }


def torch_load(torch, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - very old torch without weights_only
        return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------------


def _dataset_dir(root: Path, dataset_id: str) -> Path:
    return ensure_under(root, root / "datasets" / dataset_id, what="dataset directory")


def load_dataset(
    spec: DatasetSpec,
    root: Path,
    *,
    classification: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a dataset, returning float32 features and typed labels."""
    if spec.format == "synthetic":
        rng = np.random.default_rng(spec.seed)
        centers = rng.normal(0.0, 2.0, size=(spec.num_classes, spec.num_features))
        labels = rng.integers(0, spec.num_classes, size=spec.num_samples).astype(np.int64)
        features = centers[labels] + rng.normal(
            0.0, 1.0, size=(spec.num_samples, spec.num_features)
        ) * float(spec.noise)
        meta = {
            "format": "synthetic",
            "dataset_id": spec.dataset_id,
            "num_samples": int(spec.num_samples),
            "num_features": int(spec.num_features),
            "num_classes": int(spec.num_classes),
            "noise": float(spec.noise),
            "seed": int(spec.seed),
            "source": "generated",
        }
        return features.astype(np.float32), labels, meta

    dataset_dir = _dataset_dir(root, spec.dataset_id)
    if not dataset_dir.is_dir():
        raise ExecutorError(
            "DATASET_NOT_FOUND", f"Dataset directory does not exist: {dataset_dir}"
        )

    if spec.format == "csv":
        path = ensure_under(dataset_dir, dataset_dir / spec.file, what="dataset file")
        if not path.is_file():
            raise ExecutorError("DATASET_NOT_FOUND", f"CSV file does not exist: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ExecutorError("INVALID_DATASET", f"CSV file is empty: {path}") from exc
            if spec.target_column not in header:
                raise ExecutorError(
                    "INVALID_DATASET",
                    f"target_column {spec.target_column!r} not found in CSV header",
                )
            feature_columns = spec.feature_columns or [
                column for column in header if column != spec.target_column
            ]
            missing = [column for column in feature_columns if column not in header]
            if missing:
                raise ExecutorError(
                    "INVALID_DATASET", f"feature_columns missing from CSV: {missing}"
                )
            feature_indices = [header.index(column) for column in feature_columns]
            target_index = header.index(spec.target_column)
            feature_rows: list[list[float]] = []
            label_rows: list[float] = []
            for row_number, row in enumerate(reader, start=2):
                if not row or all(not cell.strip() for cell in row):
                    continue
                if len(row) <= target_index:
                    raise ExecutorError(
                        "INVALID_DATASET", f"Malformed CSV row {row_number}: {row!r}"
                    )
                try:
                    feature_rows.append([float(row[i]) for i in feature_indices])
                    label_rows.append(float(row[target_index]))
                except (ValueError, IndexError) as exc:
                    raise ExecutorError(
                        "INVALID_DATASET", f"Malformed CSV row {row_number}: {exc}"
                    ) from exc
        if not feature_rows:
            raise ExecutorError("INVALID_DATASET", f"CSV file has no data rows: {path}")
        features = np.asarray(feature_rows, dtype=np.float32)
        labels = np.asarray(label_rows, dtype=np.int64 if classification else np.float32)
        meta = {
            "format": "csv",
            "dataset_id": spec.dataset_id,
            "path": str(path.relative_to(root.resolve())),
            "sha256": sha256_file(path),
            "num_samples": int(features.shape[0]),
            "num_features": int(features.shape[1]),
            "target_column": spec.target_column,
            "feature_columns": feature_columns,
        }
        return features, labels, meta

    if spec.format == "npy":
        x_path = ensure_under(dataset_dir, dataset_dir / "X.npy", what="dataset file")
        if not x_path.is_file():
            raise ExecutorError("DATASET_NOT_FOUND", f"X.npy not found in {dataset_dir}")
        features = np.load(x_path).astype(np.float32)
        label_name = "y.npy" if (dataset_dir / "y.npy").exists() else "labels.npy"
        y_path = dataset_dir / label_name
        labels = (
            np.load(y_path).astype(np.int64 if classification else np.float32)
            if y_path.exists()
            else None
        )
        if labels is not None and len(labels) != len(features):
            raise ExecutorError(
                "INVALID_DATASET",
                f"Feature/label length mismatch: {len(features)} vs {len(labels)}",
            )
        meta = {
            "format": "npy",
            "dataset_id": spec.dataset_id,
            "path": str(x_path.relative_to(root.resolve())),
            "sha256": sha256_file(x_path),
            "num_samples": int(features.shape[0]),
            "num_features": int(features.shape[1]),
            "labels_present": labels is not None,
        }
        return features, labels if labels is not None else np.empty(0, dtype=np.int64), meta

    raise ExecutorError("INVALID_DATASET", f"Unsupported dataset format: {spec.format}")


def train_val_split(
    features: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    n = int(features.shape[0])
    if val_ratio <= 0 or n < 4:
        return features, labels, None, None
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(n)
    val_n = max(1, int(round(n * val_ratio)))
    val_idx, train_idx = order[:val_n], order[val_n:]
    return (
        features[train_idx],
        labels[train_idx],
        features[val_idx],
        labels[val_idx],
    )


# ---------------------------------------------------------------------------------
# models / optimisation
# ---------------------------------------------------------------------------------


def build_model(torch, spec: ModelSpec, input_dim: int, output_dim: int):
    if spec.type == "linear":
        return torch.nn.Linear(input_dim, output_dim)
    layers: list[Any] = []
    previous = input_dim
    for hidden in spec.hidden_dims:
        layers.append(torch.nn.Linear(previous, hidden))
        layers.append(torch.nn.ReLU())
        if spec.dropout > 0:
            layers.append(torch.nn.Dropout(spec.dropout))
        previous = hidden
    layers.append(torch.nn.Linear(previous, output_dim))
    return torch.nn.Sequential(*layers)


def build_optimizer(torch, spec, parameters, learning_rate: float):
    if spec.name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=spec.momentum,
            weight_decay=spec.weight_decay,
        )
    if spec.name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=spec.weight_decay
        )
    return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=spec.weight_decay)


def build_criterion(torch, name: str):
    if name == "mse":
        return torch.nn.MSELoss()
    return torch.nn.CrossEntropyLoss()


def minibatches(
    torch,
    features,
    labels,
    batch_size: int,
    *,
    shuffle: bool,
    generator=None,
) -> Iterator[tuple[Any, Any]]:
    n = int(features.shape[0])
    if n == 0:
        return
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        index = order[start : start + batch_size]
        yield features[index], labels[index]


def save_checkpoint(
    torch,
    path: Path,
    *,
    model,
    optimizer,
    epoch: int,
    model_spec: ModelSpec,
    metrics: dict[str, Any],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "model_spec": model_spec.model_dump(mode="json"),
            "metrics": metrics,
            "seed": seed,
        },
        path,
    )
