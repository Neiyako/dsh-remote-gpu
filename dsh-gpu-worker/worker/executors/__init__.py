"""Executor package: one callable per task type.

Imports are deliberately lazy.  ``training``/``inference`` pull in numpy and
(through ``torch_utils``) require PyTorch at run time, while ``generic_python``
tasks must never import torch at all.  Use :func:`get_executor` to resolve the
callable for a task type without eagerly importing heavy dependencies, or
import the individual ``run_*`` functions directly (they are resolved lazily
through :func:`__getattr__`).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

from ..errors import ExecutorError
from .context import ExecutorContext

Executor = Callable[[ExecutorContext], dict[str, Any]]

#: ``task_type`` -> (module suffix below ``worker.executors``, callable name)
_EXECUTOR_TARGETS: dict[str, tuple[str, str]] = {
    "training": ("training", "run_training"),
    "inference": ("inference", "run_inference"),
    "generic_python": ("generic", "run_generic"),
}

#: Public lazy re-exports: attribute name -> (module suffix, callable name)
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "run_training": _EXECUTOR_TARGETS["training"],
    "run_inference": _EXECUTOR_TARGETS["inference"],
    "run_generic": _EXECUTOR_TARGETS["generic_python"],
}

__all__ = ["Executor", "ExecutorContext", "available_task_types", "get_executor", "run_generic", "run_inference", "run_training"]


def available_task_types() -> tuple[str, ...]:
    """Return every task type that has an executor implementation."""
    return tuple(_EXECUTOR_TARGETS)


def get_executor(task_type: str) -> Executor:
    """Resolve the executor callable for ``task_type`` (importing it lazily)."""
    target = _EXECUTOR_TARGETS.get(task_type)
    if target is None:
        raise ExecutorError(
            "UNSUPPORTED_TASK_TYPE",
            f"No executor registered for task_type={task_type!r}; "
            f"supported types: {', '.join(sorted(_EXECUTOR_TARGETS))}",
        )
    module_name, function_name = target
    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, function_name)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:  # pragma: no cover - normal Python attribute protocol
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, function_name = target
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, function_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
