"""Path resolution helpers that enforce the worker root / experiment jail."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from .errors import path_escape

_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def looks_like_task_id(value: str) -> bool:
    return bool(_TASK_ID_RE.match(value or ""))


def ensure_under(root: Path, candidate: Path, *, what: str = "path") -> Path:
    """Resolve ``candidate`` and require it to stay inside ``root``."""
    root_resolved = root.resolve()
    resolved = candidate.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise path_escape(f"{what} does not resolve inside {root_resolved}: {resolved}")
    return resolved


def experiment_dir(settings_root: Path, task_id: str) -> Path:
    if not looks_like_task_id(task_id):
        raise path_escape(f"Invalid task id: {task_id!r}")
    return ensure_under(settings_root, settings_root / "experiments" / task_id, what="experiment dir")


def resolve_artifact_path(root: Path, task_id: str, raw_path: str) -> Path:
    """Resolve a user supplied artifact path, rejecting escapes (PATH_ESCAPE)."""
    exp = experiment_dir(root, task_id)
    if not raw_path or raw_path.strip() == "":
        raise path_escape("Empty artifact path")

    posix = PurePosixPath(raw_path)
    windows = PureWindowsPath(raw_path)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise path_escape(f"Absolute artifact paths are not allowed: {raw_path!r}")
    if ".." in posix.parts or ".." in windows.parts:
        raise path_escape(f"Artifact path may not contain '..': {raw_path!r}")

    resolved = (exp / raw_path).resolve()
    if resolved != exp.resolve() and exp.resolve() not in resolved.parents:
        raise path_escape(f"Artifact path escapes the experiment directory: {raw_path!r}")
    return resolved


def resolve_generic_script(root: Path, raw_script: str) -> Path:
    """Resolve a generic-python script, restricted to ``scripts/jobs/**.py``."""
    if not raw_script or raw_script.strip() == "":
        raise path_escape("Generic python script is required")

    posix = PurePosixPath(raw_script)
    windows = PureWindowsPath(raw_script)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise path_escape(f"Absolute script paths are not allowed: {raw_script!r}")
    if ".." in posix.parts or ".." in windows.parts:
        raise path_escape(f"Script path may not contain '..': {raw_script!r}")
    if not raw_script.endswith(".py"):
        raise path_escape("Generic python scripts must end with .py")

    scripts_dir = (root / "scripts").resolve()
    jobs_dir = (scripts_dir / "jobs").resolve()
    # Accept both "hello.py" and the contract example form "jobs/hello.py",
    # as long as the resolved target stays inside scripts/jobs.
    if posix.parts and posix.parts[0] == "jobs":
        candidate = scripts_dir / raw_script
    else:
        candidate = jobs_dir / raw_script
    resolved = candidate.resolve()
    if resolved != jobs_dir and jobs_dir not in resolved.parents:
        raise path_escape(f"Script escapes scripts/jobs: {raw_script!r}")
    return resolved
