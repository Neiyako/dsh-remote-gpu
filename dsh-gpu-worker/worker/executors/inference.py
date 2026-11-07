"""``inference`` executor (contract §3.2).

Loads a checkpoint produced by the training executor and runs evaluation /
prediction on a synthetic, CSV or NPY dataset.

Safety
------
The checkpoint is resolved strictly as
``<root>/experiments/<source_task_id>/<checkpoint>`` and re-validated with
:func:`worker.paths.ensure_under`, so a crafted task cannot read files outside
the experiment jail.  ``device="cuda"`` without CUDA yields
``CUDA_UNAVAILABLE`` — never a silent CPU fallback.
"""

from __future__ import annotations

import csv
import io
import time
from typing import Any

import numpy as np

from ..errors import ExecutorError, WorkerError
from ..models import InferenceConfig, ModelSpec
from ..paths import ensure_under, experiment_dir
from ..utils import utcnow, write_json_atomic, write_text_atomic
from .context import ExecutorContext
from .torch_utils import (
    build_criterion,
    build_model,
    load_dataset,
    reproducibility,
    require_torch,
    resolve_device,
    seed_everything,
    torch_load,
)


def _round(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


# ---------------------------------------------------------------------------------
# configuration / checkpoint resolution
# ---------------------------------------------------------------------------------


def _parse_inference_config(raw: dict[str, Any]) -> InferenceConfig:
    from pydantic import ValidationError  # local import keeps import cost low

    try:
        return InferenceConfig.model_validate(raw or {})
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error.get('loc', ())) or '<root>'}: "
            f"{error.get('msg', 'invalid value')}"
            for error in exc.errors(include_url=False, include_input=False)
        )
        raise ExecutorError("INVALID_CONFIG", f"Invalid inference config: {problems}") from exc


def _resolve_checkpoint(
    root, source_task_id: str, checkpoint: str
) -> tuple[Any, str]:
    """Resolve ``experiments/<source_task_id>/<checkpoint>`` inside the jail."""
    try:
        source_dir = experiment_dir(root, source_task_id)
        resolved = ensure_under(source_dir, source_dir / checkpoint, what="checkpoint")
    except WorkerError as exc:
        raise ExecutorError("PATH_ESCAPE", exc.message) from exc
    if not resolved.is_file():
        raise ExecutorError(
            "CHECKPOINT_NOT_FOUND",
            f"Checkpoint does not exist: experiments/{source_task_id}/{checkpoint}",
        )
    return resolved, f"experiments/{source_task_id}/{checkpoint}"


def _load_model(
    ctx: ExecutorContext, torch, cfg: InferenceConfig, checkpoint_path
) -> tuple[Any, ModelSpec, dict[str, Any], dict[str, Any]]:
    checkpoint = torch_load(torch, checkpoint_path)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ExecutorError(
            "INVALID_CHECKPOINT",
            f"Checkpoint {checkpoint_path.name} has no model_state_dict",
        )
    raw_spec = checkpoint.get("model_spec")
    if not isinstance(raw_spec, dict):
        raise ExecutorError(
            "INVALID_CHECKPOINT",
            f"Checkpoint {checkpoint_path.name} has no model_spec; it was not "
            "produced by the training executor",
        )
    try:
        model_spec = ModelSpec.model_validate(raw_spec)
    except Exception as exc:
        raise ExecutorError(
            "INVALID_CHECKPOINT", f"Checkpoint model_spec is invalid: {exc}"
        ) from exc

    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        model_config = {
            "input_dim": int(model_spec.input_dim),
            "output_dim": int(model_spec.num_classes),
            "classification": str(checkpoint.get("loss", "cross_entropy")) != "mse",
            "loss": checkpoint.get("loss", "cross_entropy"),
            "num_classes": int(model_spec.num_classes),
        }
    model_config = {
        "input_dim": int(model_config.get("input_dim", model_spec.input_dim)),
        "output_dim": int(
            model_config.get(
                "output_dim",
                model_spec.num_classes if model_config.get("classification", True) else 1,
            )
        ),
        "classification": bool(model_config.get("classification", True)),
        "loss": str(model_config.get("loss", checkpoint.get("loss", "cross_entropy"))),
        "num_classes": int(model_config.get("num_classes", model_spec.num_classes)),
        "type": model_config.get("type", model_spec.type),
    }
    model_config["checkpoint_epoch"] = checkpoint.get("epoch")
    model_config["checkpoint_task_id"] = checkpoint.get("task_id")

    model = build_model(
        torch, model_spec, model_config["input_dim"], model_config["output_dim"]
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as exc:
        raise ExecutorError(
            "CHECKPOINT_INCOMPATIBLE",
            f"Checkpoint {checkpoint_path.name} is incompatible with its model_spec: {exc}",
        ) from exc
    return model, model_spec, model_config, checkpoint


def _apply_normalization(features: np.ndarray, normalization: Any) -> tuple[np.ndarray, bool]:
    if not isinstance(normalization, dict):
        return features, False
    mean = normalization.get("mean")
    std = normalization.get("std")
    if mean is None or std is None:
        return features, False
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    if mean_array.size != features.shape[1] or std_array.size != features.shape[1]:
        raise ExecutorError(
            "INVALID_CONFIG",
            f"Checkpoint normalization has {mean_array.size} features but the "
            f"inference dataset has {features.shape[1]}",
        )
    return ((features - mean_array) / std_array).astype(np.float32), True


def _map_labels(raw_labels: np.ndarray, classes: list[int] | None) -> np.ndarray | None:
    flat = np.asarray(raw_labels).reshape(-1)
    if flat.size == 0:
        return None
    if classes is None:
        return flat.astype(np.int64)
    lookup = {int(label): index for index, label in enumerate(classes)}
    unknown = sorted({int(value) for value in flat.tolist()} - set(lookup))
    if unknown:
        raise ExecutorError(
            "INVALID_DATASET",
            f"Labels {unknown} are not present in the checkpoint classes {classes}",
        )
    return np.fromiter(
        (lookup[int(value)] for value in flat.tolist()),
        dtype=np.int64,
        count=flat.size,
    )


# ---------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------


def _render_report(
    *,
    ctx: ExecutorContext,
    cfg: InferenceConfig,
    metrics: dict[str, Any],
) -> str:
    repro = metrics.get("reproducibility") or {}
    lines: list[str] = [
        f"# Inference report — `{ctx.task_id}`",
        "",
        "## Run",
        "",
        f"- Source task: `{cfg.source_task_id}`",
        f"- Checkpoint: `{cfg.checkpoint}` (epoch={metrics.get('checkpoint_epoch')})",
        f"- Device: `{metrics.get('device')}`",
        f"- Samples: {metrics.get('num_samples')} (batch_size={cfg.batch_size})",
        f"- Task: {'classification' if metrics.get('classification') else 'regression'}",
        f"- Classes: {metrics.get('classes')}",
        "",
        "## Metrics",
        "",
    ]
    if metrics.get("classification"):
        lines.append(f"- accuracy: {_fmt(metrics.get('accuracy'))}")
        lines.append(f"- loss (cross_entropy): {_fmt(metrics.get('loss'))}")
        lines.append("")
        lines.append("| class | count | correct |")
        lines.append("|---:|---:|---:|")
        for label, stats in sorted((metrics.get("per_class") or {}).items(), key=lambda item: str(item[0])):
            lines.append(
                f"| {label} | {stats.get('count')} | {stats.get('correct')} |"
            )
    else:
        lines.append(f"- mse: {_fmt(metrics.get('mse'))}")
        lines.append(f"- rmse: {_fmt(metrics.get('rmse'))}")
        lines.append(f"- mae: {_fmt(metrics.get('mae'))}")
    lines += [
        "",
        "## Dataset",
        "",
        f"- `{_json_safe(metrics.get('dataset'))}`",
        "",
        "## Reproducibility",
        "",
        f"- seed={repro.get('seed')}, torch={repro.get('torch_version')}, "
        f"cuda={repro.get('cuda_version')}, device_name={repro.get('device_name')}",
        f"- python={repro.get('python_version')}, elapsed={metrics.get('elapsed_seconds')}s",
        "",
        "## Artifacts",
        "",
        "- `predictions.csv`",
        "- `metrics.json`, `config.json`, `report.md`",
        "",
    ]
    return "\n".join(lines)


def _json_safe(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------


def run_inference(ctx: ExecutorContext) -> dict[str, Any]:
    """Run an inference/evaluation job and return ``{"metrics", "summary"}``."""
    cfg = _parse_inference_config(ctx.config)

    started_at = utcnow()
    wall_start = time.monotonic()
    torch = require_torch()
    device = resolve_device(torch, cfg.device)
    seed_everything(torch, cfg.seed)

    experiment_dir_path = ctx.experiment_dir
    experiment_dir_path.mkdir(parents=True, exist_ok=True)

    ctx.log(
        f"[inference] task={ctx.task_id} source_task={cfg.source_task_id} "
        f"checkpoint={cfg.checkpoint} device={device} batch_size={cfg.batch_size}"
    )

    checkpoint_path, checkpoint_rel = _resolve_checkpoint(
        ctx.root, cfg.source_task_id, cfg.checkpoint
    )
    model, model_spec, model_config, checkpoint = _load_model(
        ctx, torch, cfg, checkpoint_path
    )
    classification = bool(model_config["classification"])
    input_dim = int(model_config["input_dim"])

    classes: list[int] | None = None
    if classification:
        raw_classes = checkpoint.get("classes")
        if isinstance(raw_classes, list) and raw_classes:
            classes = [int(value) for value in raw_classes]
        else:  # checkpoint without a class mapping: assume 0..K-1
            classes = list(range(int(model_config["num_classes"])))
    normalization = checkpoint.get("normalization")
    ctx.log(
        f"[inference] checkpoint={checkpoint_rel} "
        f"epoch={model_config.get('checkpoint_epoch')} classes={classes}"
    )

    features, labels, dataset_meta = load_dataset(
        cfg.dataset, ctx.root, classification=classification
    )
    if int(features.shape[1]) != input_dim:
        raise ExecutorError(
            "INVALID_CONFIG",
            f"Checkpoint expects {input_dim} features but the inference dataset has "
            f"{features.shape[1]}",
        )
    if classification and cfg.dataset.format == "synthetic":
        checkpoint_classes = int(model_config["num_classes"])
        if int(cfg.dataset.num_classes) != checkpoint_classes:
            raise ExecutorError(
                "INVALID_CONFIG",
                f"dataset.num_classes={cfg.dataset.num_classes} does not match the "
                f"checkpoint model num_classes={checkpoint_classes}",
            )

    features, normalized = _apply_normalization(features, normalization)
    if classification:
        assert classes is not None
        target_indices = _map_labels(labels, classes)
        regression_targets: np.ndarray | None = None
    else:
        target_indices = None
        flat_labels = np.asarray(labels).reshape(-1)
        regression_targets = (
            flat_labels.astype(np.float32) if flat_labels.size else None
        )
    has_labels = (
        target_indices is not None if classification else regression_targets is not None
    )

    model.to(device)
    model.eval()

    features_tensor = torch.from_numpy(np.ascontiguousarray(features))
    total = int(features_tensor.shape[0])
    if total == 0:
        raise ExecutorError("INVALID_DATASET", "Inference dataset is empty")

    if classification:
        criterion = build_criterion(torch, "cross_entropy")
    else:
        criterion = build_criterion(torch, "mse")

    batch_size = int(cfg.batch_size)
    total_batches = (total + batch_size - 1) // batch_size
    progress_stride = max(1, total_batches // 10)

    prediction_labels: list[int] = []
    confidences: list[float] = []
    regression_predictions: list[float] = []
    loss_total = 0.0
    loss_samples = 0
    correct = 0
    per_class: dict[str, dict[str, int]] = {}
    class_counts: dict[str, int] = {}

    with torch.no_grad():
        for batch_index, start in enumerate(range(0, total, batch_size), start=1):
            stop = min(start + batch_size, total)
            batch = features_tensor[start:stop].to(device)
            outputs = model(batch)
            count = int(batch.shape[0])

            if classification:
                if has_labels and target_indices is not None:
                    targets = torch.from_numpy(
                        np.ascontiguousarray(target_indices[start:stop])
                    ).to(device)
                    loss = criterion(outputs, targets)
                    loss_total += float(loss.item()) * count
                    loss_samples += count
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = probabilities.max(dim=1)
                predicted_cpu = predicted.detach().to("cpu").tolist()
                confidence_cpu = confidence.detach().to("cpu").tolist()
                if classes is not None:
                    mapped_labels = [classes[int(index)] for index in predicted_cpu]
                else:  # pragma: no cover - classes always set for classification
                    mapped_labels = [int(index) for index in predicted_cpu]
                prediction_labels.extend(int(label) for label in mapped_labels)
                confidences.extend(float(value) for value in confidence_cpu)

                if has_labels and target_indices is not None:
                    expected = target_indices[start:stop]
                    predicted_array = np.asarray(predicted_cpu, dtype=np.int64)
                    correct += int((predicted_array == expected).sum())
                    for target_index, predicted_index in zip(
                        expected.tolist(), predicted_cpu
                    ):
                        actual_label = (
                            classes[int(target_index)] if classes is not None else int(target_index)
                        )
                        key = str(actual_label)
                        stats = per_class.setdefault(key, {"count": 0, "correct": 0})
                        stats["count"] += 1
                        if int(target_index) == int(predicted_index):
                            stats["correct"] += 1
                for label in prediction_labels[-count:]:
                    key = str(label)
                    class_counts[key] = class_counts.get(key, 0) + 1
            else:
                flat = outputs.squeeze(-1)
                if has_labels and regression_targets is not None:
                    targets = torch.from_numpy(
                        np.ascontiguousarray(regression_targets[start:stop])
                    ).to(device)
                    loss = criterion(flat, targets)
                    loss_total += float(loss.item()) * count
                    loss_samples += count
                regression_predictions.extend(
                    float(value) for value in flat.detach().to("cpu").tolist()
                )

            if batch_index % progress_stride == 0 or batch_index == total_batches:
                elapsed = round(time.monotonic() - wall_start, 3)
                accuracy = correct / loss_samples if classification and loss_samples else None
                ctx.emit_progress(
                    {
                        "epoch": batch_index,
                        "total_epochs": total_batches,
                        "accuracy": _round(accuracy),
                        "learning_rate": None,
                        "elapsed_seconds": elapsed,
                    }
                )

    finished_at = utcnow()
    elapsed_total = round(time.monotonic() - wall_start, 3)
    accuracy = correct / loss_samples if classification and loss_samples else None
    predicted_count = len(prediction_labels) if classification else len(regression_predictions)

    metrics: dict[str, Any] = {
        "task_type": "inference",
        "status": "succeeded",
        "device": device,
        "resolved_device": device,
        "source_task_id": cfg.source_task_id,
        "checkpoint": cfg.checkpoint,
        "checkpoint_path": checkpoint_rel,
        "checkpoint_epoch": model_config.get("checkpoint_epoch"),
        "model": model_spec.model_dump(mode="json"),
        "classification": classification,
        "classes": classes,
        "num_samples": total,
        "num_predictions": predicted_count,
        "batch_size": batch_size,
        "normalization_applied": normalized,
        "dataset": dataset_meta,
        "predictions_file": "predictions.csv",
    }
    if classification:
        metrics.update(
            {
                "accuracy": _round(accuracy),
                "loss": _round(loss_total / loss_samples) if loss_samples else None,
                "per_class": per_class,
                "prediction_counts": class_counts,
                "mean_confidence": _round(
                    float(np.mean(confidences)) if confidences else None
                ),
            }
        )
    else:
        if has_labels and regression_targets is not None:
            predicted_array = np.asarray(regression_predictions, dtype=np.float32)
            errors = predicted_array - regression_targets
            mse = float(np.mean(np.square(errors)))
            metrics.update(
                {
                    "loss": _round(mse),
                    "mse": _round(mse),
                    "rmse": _round(float(np.sqrt(mse))),
                    "mae": _round(float(np.mean(np.abs(errors)))),
                }
            )
        else:
            metrics.update({"loss": None, "mse": None, "rmse": None, "mae": None})
        metrics["mean_prediction"] = _round(
            float(np.mean(regression_predictions)) if regression_predictions else None
        )

    # --- predictions.csv ----------------------------------------------------
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    if classification:
        writer.writerow(["row_index", "prediction", "probability", "label", "correct"])
        has_label_column = has_labels and target_indices is not None
        for index, (prediction, confidence) in enumerate(
            zip(prediction_labels, confidences)
        ):
            label_value: Any = ""
            correct_value: Any = ""
            if has_label_column and target_indices is not None:
                expected_index = int(target_indices[index])
                label_value = (
                    classes[expected_index] if classes is not None else expected_index
                )
                correct_value = int(prediction == int(label_value))
            writer.writerow(
                [index, prediction, f"{confidence:.6f}", label_value, correct_value]
            )
    else:
        writer.writerow(["row_index", "prediction", "label", "error"])
        for index, prediction in enumerate(regression_predictions):
            if has_labels and regression_targets is not None:
                actual = float(regression_targets[index])
                writer.writerow([index, f"{prediction:.6f}", f"{actual:.6f}", f"{prediction - actual:.6f}"])
            else:
                writer.writerow([index, f"{prediction:.6f}", "", ""])
    write_text_atomic(experiment_dir_path / "predictions.csv", buffer.getvalue())

    metrics["reproducibility"] = reproducibility(
        torch, device, cfg.seed, started_at, finished_at
    )
    metrics["elapsed_seconds"] = elapsed_total

    config_payload = {
        "task_id": ctx.task_id,
        "task_type": "inference",
        "config": cfg.model_dump(mode="json"),
        "resolved_device": device,
        "checkpoint": checkpoint_rel,
        "started_at": started_at,
        "finished_at": finished_at,
        "seed": int(cfg.seed),
    }

    if classification:
        summary = (
            f"inference {ctx.task_id} finished on {device}: {total} samples from "
            f"{cfg.source_task_id}/{cfg.checkpoint}, accuracy={_fmt(accuracy)}, "
            f"loss={_fmt(metrics['loss'])} in {elapsed_total}s"
        )
    else:
        summary = (
            f"inference {ctx.task_id} finished on {device}: {total} samples from "
            f"{cfg.source_task_id}/{cfg.checkpoint}, mse={_fmt(metrics.get('mse'))}, "
            f"rmse={_fmt(metrics.get('rmse'))} in {elapsed_total}s"
        )
    metrics["summary"] = summary

    write_json_atomic(experiment_dir_path / "config.json", config_payload)
    write_json_atomic(experiment_dir_path / "metrics.json", metrics)
    write_text_atomic(
        experiment_dir_path / "report.md",
        _render_report(ctx=ctx, cfg=cfg, metrics=metrics),
    )

    ctx.log(f"[inference] done: {summary}")
    return {"metrics": metrics, "summary": summary}
