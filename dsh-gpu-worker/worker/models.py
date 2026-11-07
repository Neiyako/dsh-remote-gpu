"""Pydantic models for the public API and for executor configuration."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .config import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    MIN_TIMEOUT_SECONDS,
)
from .errors import bad_request

TaskType = Literal["training", "inference", "generic_python"]
TaskStatus = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"]

TERMINAL_STATUSES: frozenset[str] = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})

_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_RESERVED_ENV = frozenset({"PATH", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"})


class ApiModel(BaseModel):
    """Base model: reject nothing, but keep validation errors friendly."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------------


class SubmitRequest(ApiModel):
    task_type: TaskType
    config: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS)
    idempotency_key: str | None = None

    @field_validator("timeout_seconds")
    @classmethod
    def _timeout_range(cls, value: int) -> int:
        if not MIN_TIMEOUT_SECONDS <= value <= MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout_seconds must be within {MIN_TIMEOUT_SECONDS}..{MAX_TIMEOUT_SECONDS}"
            )
        return value

    @field_validator("idempotency_key")
    @classmethod
    def _idempotency_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if len(value) > 256:
            raise ValueError("idempotency_key must be at most 256 characters")
        return value

    @field_validator("name")
    @classmethod
    def _name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value[:200] if value else None


class Progress(ApiModel):
    epoch: int | None = None
    total_epochs: int | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    accuracy: float | None = None
    learning_rate: float | None = None
    gpu_utilization_percent: float | None = None
    gpu_memory_used_mb: float | None = None
    elapsed_seconds: float | None = None
    updated_at: str | None = None


class ErrorInfo(ApiModel):
    code: str
    message: str
    traceback: str | None = None


class ArtifactInfo(ApiModel):
    path: str
    size_bytes: int
    sha256: str
    download_url: str | None = None


class TaskView(ApiModel):
    task_id: str
    task_type: TaskType
    status: TaskStatus
    name: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    timeout_seconds: int
    progress: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    cancel_requested: bool = False
    idempotency_key: str | None = None


# ---------------------------------------------------------------------------------
# Executor configuration models
# ---------------------------------------------------------------------------------


class ModelSpec(ApiModel):
    type: Literal["mlp", "linear"] = "mlp"
    input_dim: int = Field(default=8, ge=1)
    hidden_dims: list[int] = Field(default_factory=lambda: [64, 32])
    num_classes: int = Field(default=3, ge=2)
    dropout: float = Field(default=0.0, ge=0.0, le=0.9)

    @field_validator("hidden_dims")
    @classmethod
    def _hidden_dims(cls, value: list[int]) -> list[int]:
        if any(dim < 1 for dim in value):
            raise ValueError("hidden_dims must all be >= 1")
        return value


class DatasetSpec(ApiModel):
    dataset_id: str = "synthetic"
    format: Literal["synthetic", "csv", "npy"] = "synthetic"
    num_samples: int = Field(default=600, ge=1)
    num_features: int = Field(default=8, ge=1)
    num_classes: int = Field(default=3, ge=2)
    noise: float = Field(default=0.1, ge=0.0)
    val_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    seed: int = 42
    file: str = "data.csv"
    target_column: str = "target"
    feature_columns: list[str] | None = None

    @field_validator("dataset_id")
    @classmethod
    def _dataset_id(cls, value: str) -> str:
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", value or ""):
            raise ValueError("dataset_id must be a simple directory name")
        return value

    @field_validator("file")
    @classmethod
    def _file(cls, value: str) -> str:
        if "/" in value or "\\" in value or value in {"", ".", ".."}:
            raise ValueError("dataset file must be a plain file name")
        return value


class OptimizerSpec(ApiModel):
    name: Literal["adam", "adamw", "sgd"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)


class LossSpec(ApiModel):
    name: Literal["cross_entropy", "mse"] = "cross_entropy"


class CheckpointSpec(ApiModel):
    enabled: bool = True
    every_n_epochs: int = Field(default=1, ge=1)
    save_best: bool = True
    resume_from: str | None = None


class EarlyStoppingSpec(ApiModel):
    enabled: bool = False
    patience: int = Field(default=3, ge=1)
    min_delta: float = Field(default=0.0, ge=0.0)
    monitor: Literal["val_loss", "accuracy"] = "val_loss"


class ValidationSpec(ApiModel):
    enabled: bool = True
    batch_size: int = Field(default=64, ge=1)


class TrainingConfig(ApiModel):
    model: ModelSpec = Field(default_factory=ModelSpec)
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)
    epochs: int = Field(default=10, ge=1, le=100000)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)
    loss: LossSpec = Field(default_factory=LossSpec)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    checkpoint: CheckpointSpec = Field(default_factory=CheckpointSpec)
    early_stopping: EarlyStoppingSpec = Field(default_factory=EarlyStoppingSpec)
    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    seed: int = 42

    @model_validator(mode="after")
    def _consistency(self) -> "TrainingConfig":
        if self.dataset.format == "synthetic" and self.dataset.num_classes != self.model.num_classes:
            raise ValueError(
                "dataset.num_classes and model.num_classes must match for synthetic data"
            )
        return self


class InferenceConfig(ApiModel):
    source_task_id: str
    checkpoint: str = "best_model.pt"
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)
    batch_size: int = Field(default=64, ge=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 42

    @field_validator("source_task_id")
    @classmethod
    def _source_task_id(cls, value: str) -> str:
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$", value or ""):
            raise ValueError("source_task_id must be a task id")
        return value

    @field_validator("checkpoint")
    @classmethod
    def _checkpoint(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("checkpoint must be a plain file name")
        return value


class GenericPythonConfig(ApiModel):
    script: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("args")
    @classmethod
    def _args(cls, value: list[str]) -> list[str]:
        for item in value:
            if not isinstance(item, str):
                raise ValueError("args must be a list of strings")
        return value

    @field_validator("env")
    @classmethod
    def _env(cls, value: dict[str, str]) -> dict[str, str]:
        for key, val in value.items():
            if not _ENV_NAME_RE.match(key):
                raise ValueError(f"invalid env name: {key!r}")
            if key in _RESERVED_ENV:
                raise ValueError(f"env {key} may not be overridden")
            if not isinstance(val, str):
                raise ValueError("env values must be strings")
        return value


CONFIG_MODELS: dict[str, type[ApiModel]] = {
    "training": TrainingConfig,
    "inference": InferenceConfig,
    "generic_python": GenericPythonConfig,
}


def parse_task_config(task_type: str, raw_config: dict[str, Any]) -> ApiModel:
    """Validate ``config`` for ``task_type``, raising a 400 INVALID_REQUEST."""
    model = CONFIG_MODELS.get(task_type)
    if model is None:
        raise bad_request(
            "INVALID_REQUEST",
            f"Unsupported task_type: {task_type!r}",
            {"supported": sorted(CONFIG_MODELS)},
        )
    try:
        return model.model_validate(raw_config or {})
    except ValidationError as exc:
        raise bad_request(
            "INVALID_REQUEST",
            f"Invalid {task_type} config",
            {"errors": exc.errors(include_url=False, include_input=False, include_context=False)},
        ) from exc
