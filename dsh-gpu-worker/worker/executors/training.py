"""``training`` executor (contract §3.1).

Trains a small PyTorch model (``mlp`` or ``linear``) on a synthetic/CSV/NPY
dataset and writes every artifact inside ``ctx.experiment_dir``.

Design notes
------------
* ``device="cuda"`` with no CUDA fails loudly with ``CUDA_UNAVAILABLE``; the
  executor never falls back to CPU behind the caller's back.
* One progress record per epoch is emitted through ``ctx.emit_progress`` (the
  parent worker turns it into the persisted ``progress`` object); human
  readable lines go through ``ctx.log`` and also into ``train.log``.
* Checkpoints store CPU tensors so ``torch.load(..., map_location="cpu")``
  works on any host, and they carry the model config, feature normalisation
  statistics and class mapping required by the inference executor.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ExecutorError
from ..models import TrainingConfig
from ..paths import ensure_under
from ..utils import utcnow, write_json_atomic, write_text_atomic
from .context import ExecutorContext
from .torch_utils import (
    build_criterion,
    build_model,
    build_optimizer,
    gpu_metrics,
    load_dataset,
    reproducibility,
    require_torch,
    resolve_device,
    seed_everything,
    torch_load,
    train_val_split,
)

#: Features whose standard deviation is below this are treated as constant.
_NORMALIZATION_EPS = 1e-8


# ---------------------------------------------------------------------------------
# logging helpers
# ---------------------------------------------------------------------------------


class _TrainLog:
    """Append-only ``train.log`` writer (plain text, one line per event)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        if self._handle.closed:  # pragma: no cover - defensive
            return
        self._handle.write(f"{utcnow()} {message}\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


def _round(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


# ---------------------------------------------------------------------------------
# configuration / data
# ---------------------------------------------------------------------------------


def _parse_training_config(raw: dict[str, Any]) -> TrainingConfig:
    from pydantic import ValidationError  # local import keeps import cost low

    try:
        return TrainingConfig.model_validate(raw or {})
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error.get('loc', ())) or '<root>'}: "
            f"{error.get('msg', 'invalid value')}"
            for error in exc.errors(include_url=False, include_input=False)
        )
        raise ExecutorError("INVALID_CONFIG", f"Invalid training config: {problems}") from exc


def _compute_normalization(features: np.ndarray) -> dict[str, Any]:
    """Per-feature mean/std of the training split (used by inference too)."""
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std = np.where(std < _NORMALIZATION_EPS, np.float32(1.0), std).astype(np.float32)
    return {
        "mean": [float(value) for value in mean.tolist()],
        "std": [float(value) for value in std.tolist()],
        "num_features": int(features.shape[1]),
        "method": "zscore",
    }


def _apply_normalization(features: np.ndarray, normalization: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    return ((features - mean) / std).astype(np.float32)


def _prepare_targets(
    labels: np.ndarray,
    *,
    classification: bool,
    num_classes: int,
) -> tuple[np.ndarray, list[int] | None]:
    """Return targets in ``0..k-1`` index space plus the class label mapping."""
    flat = np.asarray(labels).reshape(-1)
    if flat.size == 0:
        raise ExecutorError(
            "INVALID_DATASET",
            "Training requires labelled data but the dataset does not provide labels",
        )
    if not classification:
        return flat.astype(np.float32), None

    raw = flat.astype(np.int64)
    classes = sorted({int(value) for value in raw.tolist()})
    if len(classes) > num_classes:
        raise ExecutorError(
            "INVALID_CONFIG",
            f"Dataset contains {len(classes)} distinct classes but "
            f"model.num_classes={num_classes}",
        )
    lookup = {label: index for index, label in enumerate(classes)}
    mapped = np.fromiter(
        (lookup[int(value)] for value in raw.tolist()),
        dtype=np.int64,
        count=raw.size,
    )
    return mapped, classes


def _make_loader(
    torch,
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
):
    """Build a ``TensorDataset``/``DataLoader`` pair over in-memory arrays."""
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(features)),
        torch.from_numpy(np.ascontiguousarray(targets)),
    )
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


# ---------------------------------------------------------------------------------
# optimisation loops
# ---------------------------------------------------------------------------------


def _train_epoch(
    torch,
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    *,
    classification: bool,
) -> tuple[float | None, float | None]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features)
        loss = criterion(outputs, labels) if classification else criterion(outputs.squeeze(-1), labels)
        loss.backward()
        optimizer.step()

        count = int(features.shape[0])
        total_loss += float(loss.detach().item()) * count
        total += count
        if classification:
            correct += int((outputs.argmax(dim=1) == labels).sum().item())

    if total == 0:
        return None, None
    accuracy = correct / total if classification else None
    return total_loss / total, accuracy


def _evaluate(
    torch,
    model,
    loader,
    criterion,
    device: str,
    *,
    classification: bool,
) -> tuple[float | None, float | None]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = (
                criterion(outputs, labels)
                if classification
                else criterion(outputs.squeeze(-1), labels)
            )
            count = int(features.shape[0])
            total_loss += float(loss.item()) * count
            total += count
            if classification:
                correct += int((outputs.argmax(dim=1) == labels).sum().item())

    if total == 0:
        return None, None
    accuracy = correct / total if classification else None
    return total_loss / total, accuracy


# ---------------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------------


def _save_checkpoint(
    torch,
    path: Path,
    *,
    ctx: ExecutorContext,
    cfg: TrainingConfig,
    model,
    optimizer,
    epoch: int,
    classes: list[int] | None,
    normalization: dict[str, Any],
    dataset_meta: dict[str, Any],
    history: list[dict[str, Any]],
    best: dict[str, Any] | None,
    device: str,
    classification: bool,
    output_dim: int,
) -> None:
    # CPU tensors keep the checkpoint portable (map_location="cpu" friendly).
    state_dict = {
        name: tensor.detach().to("cpu") for name, tensor in model.state_dict().items()
    }
    payload = {
        "checkpoint_version": 1,
        "task_id": ctx.task_id,
        "task_type": "training",
        "epoch": int(epoch),
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "model_spec": cfg.model.model_dump(mode="json"),
        "model_config": {
            "type": cfg.model.type,
            "input_dim": int(cfg.model.input_dim),
            "hidden_dims": [int(dim) for dim in cfg.model.hidden_dims],
            "num_classes": int(cfg.model.num_classes),
            "dropout": float(cfg.model.dropout),
            "output_dim": int(output_dim),
            "classification": bool(classification),
            "loss": cfg.loss.name,
        },
        "normalization": normalization,
        "classes": [int(label) for label in classes] if classes is not None else None,
        "loss": cfg.loss.name,
        "dataset": dataset_meta,
        "seed": int(cfg.seed),
        "device": device,
        # ``str()`` keeps the checkpoint loadable with ``weights_only=True``.
        "torch_version": str(getattr(torch, "__version__", "")),
        "metrics": {"history": history, "best": best},
        "saved_at": utcnow(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(payload, path)
    except OSError as exc:
        raise ExecutorError(
            "CHECKPOINT_SAVE_FAILED", f"Failed to write checkpoint {path.name}: {exc}"
        ) from exc


def _resolve_resume_checkpoint(ctx: ExecutorContext, cfg: TrainingConfig) -> Path:
    raw = (cfg.checkpoint.resume_from or "").strip()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        resolved = ensure_under(ctx.root, candidate, what="resume checkpoint")
    else:
        in_experiment = ctx.experiment_dir / raw
        if in_experiment.exists():
            resolved = ensure_under(ctx.root, in_experiment, what="resume checkpoint")
        else:
            resolved = ensure_under(ctx.root, ctx.root / raw, what="resume checkpoint")
    if not resolved.is_file():
        raise ExecutorError(
            "CHECKPOINT_NOT_FOUND",
            f"resume_from checkpoint does not exist: {raw!r} "
            f"(resolved to {ctx.path_relative_root(resolved)})",
        )
    return resolved


def _restore_from_checkpoint(ctx: ExecutorContext, cfg: TrainingConfig, torch, model, optimizer) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    resume_path = _resolve_resume_checkpoint(ctx, cfg)
    checkpoint = torch_load(torch, resume_path)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ExecutorError(
            "INVALID_CHECKPOINT",
            f"Checkpoint {resume_path.name} has no model_state_dict",
        )
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as exc:
        raise ExecutorError(
            "CHECKPOINT_INCOMPATIBLE",
            f"Checkpoint {resume_path.name} is incompatible with the configured model: {exc}",
        ) from exc
    if checkpoint.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as exc:
            raise ExecutorError(
                "CHECKPOINT_INCOMPATIBLE",
                f"Checkpoint {resume_path.name} has an incompatible optimizer state: {exc}",
            ) from exc

    start_epoch = int(checkpoint.get("epoch") or 0)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    previous = checkpoint.get("metrics")
    if isinstance(previous, dict):
        history = [dict(entry) for entry in previous.get("history") or [] if isinstance(entry, dict)]
        if isinstance(previous.get("best"), dict):
            best = dict(previous["best"])
    info = {
        "path": ctx.path_relative_root(resume_path),
        "epoch": start_epoch,
        "task_id": checkpoint.get("task_id"),
    }
    return start_epoch, history, best, info


# ---------------------------------------------------------------------------------
# monitoring / reporting
# ---------------------------------------------------------------------------------


def _select_monitor(
    cfg: TrainingConfig, *, classification: bool, use_validation: bool
) -> tuple[str, bool]:
    monitor = cfg.early_stopping.monitor
    if monitor == "accuracy" and classification:
        return ("val_accuracy" if use_validation else "train_accuracy"), True
    return ("val_loss" if use_validation else "train_loss"), False


def _is_improvement(
    current: float,
    best_value: float | None,
    *,
    higher_better: bool,
    min_delta: float,
) -> bool:
    if best_value is None:
        return True
    if higher_better:
        return current > best_value + float(min_delta)
    return current < best_value - float(min_delta)


def _render_report(
    *,
    ctx: ExecutorContext,
    cfg: TrainingConfig,
    device: str,
    metrics: dict[str, Any],
) -> str:
    history: list[dict[str, Any]] = metrics.get("history") or []
    best = metrics.get("best") or {}
    final = metrics.get("final") or {}
    repro = metrics.get("reproducibility") or {}
    classes = metrics.get("model", {}).get("classes")
    lines: list[str] = [
        f"# Training report — `{ctx.task_id}`",
        "",
        "## Run",
        "",
        f"- Device: `{device}`",
        f"- Model: `{cfg.model.type}` (input_dim={cfg.model.input_dim}, "
        f"hidden_dims={cfg.model.hidden_dims}, num_classes={cfg.model.num_classes}, "
        f"parameters={metrics.get('model', {}).get('parameters')})",
        f"- Loss: `{cfg.loss.name}`",
        f"- Optimizer: `{cfg.optimizer.name}` (lr={cfg.learning_rate}, "
        f"weight_decay={cfg.optimizer.weight_decay}, momentum={cfg.optimizer.momentum})",
        f"- Batch size: {cfg.batch_size}",
        f"- Epochs completed: {metrics.get('epochs_completed')} / {metrics.get('epochs_requested')}",
        f"- Early stopped: {metrics.get('early_stopped')} ({metrics.get('stop_reason')})",
        f"- Classes: {classes}",
        "",
        "## Dataset",
        "",
        f"- `{json_safe(metrics.get('dataset'))}`",
        "",
        "## Per-epoch metrics",
        "",
        "| epoch | train_loss | val_loss | train_acc | val_acc | lr | elapsed_s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in history:
        lines.append(
            "| {epoch} | {train_loss} | {val_loss} | {train_acc} | {val_acc} | "
            "{lr} | {elapsed} |".format(
                epoch=entry.get("epoch"),
                train_loss=_fmt(entry.get("train_loss")),
                val_loss=_fmt(entry.get("val_loss")),
                train_acc=_fmt(entry.get("train_accuracy")),
                val_acc=_fmt(entry.get("val_accuracy")),
                lr=_fmt(entry.get("learning_rate"), 6),
                elapsed=_fmt(entry.get("elapsed_seconds"), 3),
            )
        )
    lines += [
        "",
        "## Best checkpoint",
        "",
        f"- Metric: `{best.get('metric')}` = {_fmt(best.get('value'))} at epoch {best.get('epoch')}",
        "",
        "## Final epoch",
        "",
        f"- train_loss={_fmt(final.get('train_loss'))}, val_loss={_fmt(final.get('val_loss'))}, "
        f"accuracy={_fmt(final.get('accuracy'))}",
        "",
        "## Reproducibility",
        "",
        f"- seed={repro.get('seed')}, torch={repro.get('torch_version')}, "
        f"cuda={repro.get('cuda_version')}, device_name={repro.get('device_name')}",
        f"- python={repro.get('python_version')} ({repro.get('python_executable')})",
        f"- started_at={repro.get('started_at')}, finished_at={repro.get('finished_at')}",
        "",
        "## Artifacts",
        "",
        "- `model.pt` (last checkpoint)",
        "- `best_model.pt` (best checkpoint, if `save_best`)",
        "- `metrics.json`, `config.json`, `train.log`, `report.md`",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def json_safe(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------


def run_training(ctx: ExecutorContext) -> dict[str, Any]:
    """Run a training job and return ``{"metrics": dict, "summary": str}``."""
    cfg = _parse_training_config(ctx.config)

    started_at = utcnow()
    wall_start = time.monotonic()
    torch = require_torch()
    device = resolve_device(torch, cfg.device)
    seed_everything(torch, cfg.seed)

    experiment_dir = ctx.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)
    train_log = _TrainLog(experiment_dir / "train.log")

    def info(message: str) -> None:
        train_log.write(message)
        ctx.log(message)

    try:
        info(
            f"[training] task={ctx.task_id} device={device} epochs={cfg.epochs} "
            f"batch_size={cfg.batch_size} seed={cfg.seed} model={cfg.model.type} "
            f"loss={cfg.loss.name} optimizer={cfg.optimizer.name}"
        )

        classification = cfg.loss.name == "cross_entropy"

        # --- data ------------------------------------------------------------
        features, labels, dataset_meta = load_dataset(
            cfg.dataset, ctx.root, classification=classification
        )
        if int(features.shape[1]) != int(cfg.model.input_dim):
            raise ExecutorError(
                "INVALID_CONFIG",
                f"model.input_dim={cfg.model.input_dim} does not match the dataset "
                f"feature dimension {features.shape[1]}",
            )
        targets, classes = _prepare_targets(
            labels, classification=classification, num_classes=cfg.model.num_classes
        )
        if features.shape[0] != targets.shape[0]:
            raise ExecutorError(
                "INVALID_DATASET",
                f"Feature/label length mismatch: {features.shape[0]} vs {targets.shape[0]}",
            )
        info(
            f"[training] dataset={dataset_meta.get('format')} samples={features.shape[0]} "
            f"features={features.shape[1]} classes={classes}"
        )

        use_validation = bool(cfg.validation.enabled and cfg.dataset.val_ratio > 0)
        if use_validation:
            x_train, y_train, x_val, y_val = train_val_split(
                features, targets, cfg.dataset.val_ratio, cfg.seed
            )
        else:
            x_train, y_train = features, targets
            x_val = y_val = None
        if x_train.shape[0] == 0:
            raise ExecutorError(
                "INVALID_DATASET", "Training split is empty after applying val_ratio"
            )

        normalization = _compute_normalization(x_train)
        x_train = _apply_normalization(x_train, normalization)
        if x_val is not None:
            x_val = _apply_normalization(x_val, normalization)

        train_loader = _make_loader(
            torch, x_train, y_train, cfg.batch_size, shuffle=True, seed=cfg.seed
        )
        val_loader = None
        if use_validation and x_val is not None and y_val is not None:
            val_loader = _make_loader(
                torch,
                x_val,
                y_val,
                cfg.validation.batch_size,
                shuffle=False,
                seed=cfg.seed,
            )

        # --- model / optimiser ----------------------------------------------
        output_dim = int(cfg.model.num_classes) if classification else 1
        model = build_model(torch, cfg.model, int(cfg.model.input_dim), output_dim).to(device)
        criterion = build_criterion(torch, cfg.loss.name)
        optimizer = build_optimizer(
            torch, cfg.optimizer, model.parameters(), float(cfg.learning_rate)
        )
        parameters = int(sum(parameter.numel() for parameter in model.parameters()))
        info(f"[training] model parameters={parameters} output_dim={output_dim}")

        # --- optional resume ------------------------------------------------
        history: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        start_epoch = 0
        resume_info: dict[str, Any] | None = None
        if cfg.checkpoint.resume_from:
            start_epoch, history, best, resume_info = _restore_from_checkpoint(
                ctx, cfg, torch, model, optimizer
            )
            info(
                f"[training] resumed from {resume_info['path']} "
                f"at epoch {start_epoch} (previous best={best})"
            )
            if start_epoch >= cfg.epochs:
                info(
                    f"[training] checkpoint already covers {start_epoch} epochs >= "
                    f"requested {cfg.epochs}; skipping training loop"
                )

        monitor, higher_better = _select_monitor(
            cfg, classification=classification, use_validation=use_validation
        )
        patience_counter = 0
        early_stopped = False
        stop_reason = "completed"
        gpu: dict[str, Any] = {}

        for epoch in range(start_epoch + 1, int(cfg.epochs) + 1):
            epoch_started = time.monotonic()
            train_loss, train_accuracy = _train_epoch(
                torch,
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                classification=classification,
            )
            val_loss: float | None = None
            val_accuracy: float | None = None
            if val_loader is not None:
                val_loss, val_accuracy = _evaluate(
                    torch,
                    model,
                    val_loader,
                    criterion,
                    device,
                    classification=classification,
                )

            gpu = gpu_metrics(torch, device)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            elapsed = round(time.monotonic() - wall_start, 3)
            accuracy = val_accuracy if val_accuracy is not None else train_accuracy
            entry: dict[str, Any] = {
                "epoch": int(epoch),
                "train_loss": _round(train_loss),
                "val_loss": _round(val_loss),
                "train_accuracy": _round(train_accuracy),
                "val_accuracy": _round(val_accuracy),
                "accuracy": _round(accuracy),
                "learning_rate": learning_rate,
                "gpu_utilization_percent": gpu.get("gpu_utilization_percent"),
                "gpu_memory_used_mb": gpu.get("gpu_memory_used_mb"),
                "elapsed_seconds": elapsed,
                "epoch_seconds": round(time.monotonic() - epoch_started, 3),
                "timestamp": utcnow(),
            }
            history.append(entry)

            ctx.emit_progress(
                {
                    "epoch": int(epoch),
                    "total_epochs": int(cfg.epochs),
                    "train_loss": _round(train_loss),
                    "val_loss": _round(val_loss),
                    "accuracy": _round(accuracy),
                    "learning_rate": learning_rate,
                    "gpu_utilization_percent": gpu.get("gpu_utilization_percent"),
                    "gpu_memory_used_mb": gpu.get("gpu_memory_used_mb"),
                    "elapsed_seconds": elapsed,
                }
            )
            info(
                f"[epoch {epoch}/{cfg.epochs}] train_loss={_fmt(train_loss)} "
                f"val_loss={_fmt(val_loss)} accuracy={_fmt(accuracy)} "
                f"lr={learning_rate:g} elapsed={elapsed}s"
            )

            current = entry.get(monitor)
            improved = current is not None and _is_improvement(
                float(current),
                None if best is None else _coerce_float(best.get("value")),
                higher_better=higher_better,
                min_delta=float(cfg.early_stopping.min_delta),
            )
            if improved:
                best = {
                    "epoch": int(epoch),
                    "metric": monitor,
                    "value": _round(float(current)),
                    "higher_is_better": higher_better,
                    "train_loss": entry["train_loss"],
                    "val_loss": entry["val_loss"],
                    "train_accuracy": entry["train_accuracy"],
                    "val_accuracy": entry["val_accuracy"],
                }
                patience_counter = 0
                if cfg.checkpoint.enabled and cfg.checkpoint.save_best:
                    _save_checkpoint(
                        torch,
                        experiment_dir / "best_model.pt",
                        ctx=ctx,
                        cfg=cfg,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        classes=classes,
                        normalization=normalization,
                        dataset_meta=dataset_meta,
                        history=history,
                        best=best,
                        device=device,
                        classification=classification,
                        output_dim=output_dim,
                    )
                    info(
                        f"[checkpoint] saved best_model.pt at epoch {epoch} "
                        f"({monitor}={_fmt(current)})"
                    )
            else:
                patience_counter += 1

            if cfg.checkpoint.enabled and epoch % cfg.checkpoint.every_n_epochs == 0:
                _save_checkpoint(
                    torch,
                    experiment_dir / "model.pt",
                    ctx=ctx,
                    cfg=cfg,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    classes=classes,
                    normalization=normalization,
                    dataset_meta=dataset_meta,
                    history=history,
                    best=best,
                    device=device,
                    classification=classification,
                    output_dim=output_dim,
                )

            if (
                cfg.early_stopping.enabled
                and patience_counter >= cfg.early_stopping.patience
            ):
                early_stopped = True
                stop_reason = (
                    f"early_stopping:{monitor}:patience={cfg.early_stopping.patience}:"
                    f"min_delta={cfg.early_stopping.min_delta}"
                )
                info(
                    f"[training] early stopping at epoch {epoch} "
                    f"(no {monitor} improvement for {patience_counter} epochs)"
                )
                break

        # Always persist a "last" checkpoint with the final model state.
        if cfg.checkpoint.enabled:
            _save_checkpoint(
                torch,
                experiment_dir / "model.pt",
                ctx=ctx,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                epoch=int(history[-1]["epoch"]) if history else start_epoch,
                classes=classes,
                normalization=normalization,
                dataset_meta=dataset_meta,
                history=history,
                best=best,
                device=device,
                classification=classification,
                output_dim=output_dim,
            )
            info("[checkpoint] saved model.pt (last)")

        finished_at = utcnow()
        total_elapsed = round(time.monotonic() - wall_start, 3)
        final = history[-1] if history else None
        if best is None and final is not None:
            best = {
                "epoch": final["epoch"],
                "metric": monitor,
                "value": final.get(monitor),
                "higher_is_better": higher_better,
                "train_loss": final.get("train_loss"),
                "val_loss": final.get("val_loss"),
                "train_accuracy": final.get("train_accuracy"),
                "val_accuracy": final.get("val_accuracy"),
            }

        metrics: dict[str, Any] = {
            "task_type": "training",
            "status": "succeeded",
            "device": device,
            "resolved_device": device,
            "epochs_requested": int(cfg.epochs),
            "epochs_completed": len(history),
            "early_stopped": early_stopped,
            "stop_reason": stop_reason,
            "best": best,
            "final": final,
            "history": history,
            "dataset": dataset_meta,
            "model": {
                "spec": cfg.model.model_dump(mode="json"),
                "input_dim": int(cfg.model.input_dim),
                "output_dim": output_dim,
                "parameters": parameters,
                "classification": classification,
                "classes": classes,
            },
            "optimizer": cfg.optimizer.model_dump(mode="json"),
            "loss": cfg.loss.name,
            "learning_rate": float(cfg.learning_rate),
            "batch_size": int(cfg.batch_size),
            "normalization": normalization,
            "checkpoints": {
                "last": "model.pt" if cfg.checkpoint.enabled else None,
                "best": "best_model.pt"
                if (cfg.checkpoint.enabled and cfg.checkpoint.save_best)
                else None,
                "resumed_from": resume_info,
            },
            "reproducibility": reproducibility(
                torch, device, cfg.seed, started_at, finished_at
            ),
            "elapsed_seconds": total_elapsed,
            "summary": "",
        }
        summary = (
            f"training {ctx.task_id} finished on {device}: "
            f"{len(history)}/{cfg.epochs} epochs"
            + (" (early stopped)" if early_stopped else "")
            + f", best {monitor}={_fmt(None if best is None else best.get('value'))}"
            + f", final train_loss={_fmt(None if final is None else final.get('train_loss'))}"
            + f", val_loss={_fmt(None if final is None else final.get('val_loss'))}"
            + f" in {total_elapsed}s"
        )
        metrics["summary"] = summary

        config_payload = {
            "task_id": ctx.task_id,
            "task_type": "training",
            "config": cfg.model_dump(mode="json"),
            "resolved_device": device,
            "started_at": started_at,
            "finished_at": finished_at,
            "seed": int(cfg.seed),
        }
        write_json_atomic(experiment_dir / "config.json", config_payload)
        write_json_atomic(experiment_dir / "metrics.json", metrics)
        write_text_atomic(
            experiment_dir / "report.md",
            _render_report(ctx=ctx, cfg=cfg, device=device, metrics=metrics),
        )

        info(f"[training] done: {summary}")
        return {"metrics": metrics, "summary": summary}
    finally:
        train_log.close()


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
