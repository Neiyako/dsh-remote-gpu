"""Subprocess entry point for one task.

Invoked by the runner as::

    python -m worker.job_entry <root>/experiments/<task_id>/job.json

The process is fully isolated from the FastAPI server: it loads the JSON spec,
dispatches to the executor for the task type, writes ``result.json`` on success
or ``error.json`` on failure, and exits non-zero on failure.
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

PROJECT_DIR = Path(__file__).resolve().parent.parent

#: task_type -> (module, callable).  Note ``generic_python`` maps to run_generic.
EXECUTOR_TARGETS: dict[str, tuple[str, str]] = {
    "generic_python": ("worker.executors.generic", "run_generic"),
    "training": ("worker.executors.training", "run_training"),
    "inference": ("worker.executors.inference", "run_inference"),
}


def _ensure_import_path() -> None:
    project = str(PROJECT_DIR)
    if project not in sys.path:
        sys.path.insert(0, project)


def _write_json(path: Path, value: Any) -> None:
    from worker.utils import write_json_atomic

    write_json_atomic(path, value)


def _enforce_cuda_request(task_type: str, config: dict[str, Any]) -> None:
    """A forced ``device="cuda"`` task must fail loudly when CUDA is absent.

    Checked at the job boundary so the contract holds even on a torch-less
    worker: we never let a cuda-forced training/inference task fall back to CPU.
    """
    if task_type not in {"training", "inference"}:
        return
    from worker.errors import ExecutorError

    device = str((config or {}).get("device", "auto"))
    if device != "cuda":
        return
    from worker import gpu

    if not gpu.cuda_available(refresh=True):
        raise ExecutorError(
            "CUDA_UNAVAILABLE",
            "device='cuda' was requested but CUDA is unavailable on this worker "
            f"({gpu.cuda_unavailable_reason()}); refusing to fall back to CPU.",
        )


def _resolve_executor(task_type: str) -> Callable:
    from worker.errors import ExecutorError

    target = EXECUTOR_TARGETS.get(task_type)
    if target is None:
        raise ExecutorError(
            "UNKNOWN_TASK_TYPE",
            f"Unsupported task_type: {task_type!r}",
        )
    module_name, function_name = target

    # Preferred path: the executors package exposes get_executor() with lazy imports.
    try:
        from worker.executors import get_executor

        return get_executor(task_type)
    except ImportError:
        pass

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ExecutorError(
            "EXECUTOR_UNAVAILABLE",
            f"Executor module {module_name} could not be imported: {exc}",
        ) from exc
    function = getattr(module, function_name, None)
    if function is None:
        raise ExecutorError(
            "EXECUTOR_UNAVAILABLE",
            f"Executor {module_name}.{function_name} is not implemented",
        )
    return function


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m worker.job_entry <job.json>", file=sys.stderr)
        return 2

    _ensure_import_path()

    from worker.errors import ExecutorError
    from worker.executors.context import ExecutorContext
    from worker.paths import ensure_under

    spec_path = Path(argv[0]).resolve()
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[job] cannot read job spec {spec_path}: {exc}", file=sys.stderr)
        return 2

    task_id = str(spec.get("task_id") or "")
    task_type = str(spec.get("task_type") or "")
    experiment_dir = Path(str(spec.get("experiment_dir") or "")).resolve()
    root = Path(str(spec.get("root") or "")).resolve()
    if not task_id or not task_type or not spec.get("experiment_dir") or not spec.get("root"):
        print(f"[job] malformed job spec: {spec!r}", file=sys.stderr)
        return 2

    # Defense in depth: the runner should never write an escaping spec.
    try:
        ensure_under(root, experiment_dir, what="experiment directory")
    except Exception as exc:
        print(f"[job] refusing unsafe experiment dir: {exc}", file=sys.stderr)
        return 2
    experiment_dir.mkdir(parents=True, exist_ok=True)

    context = ExecutorContext(
        task_id=task_id,
        task_type=task_type,
        config=spec.get("config") or {},
        root=root,
        experiment_dir=experiment_dir,
    )

    try:
        _enforce_cuda_request(task_type, context.config)
        executor = _resolve_executor(task_type)
        outcome = executor(context) or {}
        metrics = outcome.get("metrics") or {}
        summary = outcome.get("summary") or f"{task_type} task finished"
        from worker.artifacts import scan_artifacts

        result = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "SUCCEEDED",
            "metrics": metrics,
            "summary": summary,
            "artifacts": scan_artifacts(experiment_dir, task_id),
            "config": context.config,
        }
        _write_json(experiment_dir / "result.json", result)
        context.emit_result({"ok": True, "metrics": metrics, "summary": summary})
        return 0
    except ExecutorError as exc:
        error = {
            "code": exc.code,
            "message": exc.message,
            "traceback": traceback.format_exc(),
        }
        _write_json(experiment_dir / "error.json", {"error": error, "task_id": task_id})
        print(f"[job] FAILED {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:  # unexpected executor bug
        error = {
            "code": "EXECUTOR_ERROR",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _write_json(experiment_dir / "error.json", {"error": error, "task_id": task_id})
        print(f"[job] FAILED EXECUTOR_ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover - process entry
    raise SystemExit(main())
