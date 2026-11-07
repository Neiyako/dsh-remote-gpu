"""Artifact discovery / hashing for existing experiment directories only."""

from __future__ import annotations

import threading
from pathlib import Path
from urllib.parse import quote

from .utils import sha256_file

_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_CACHE_LOCK = threading.Lock()


def _cached_sha256(path: Path, stat) -> str:
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _CACHE_LOCK:
        cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = sha256_file(path)
    with _CACHE_LOCK:
        _HASH_CACHE[key] = digest
        if len(_HASH_CACHE) > 4096:
            _HASH_CACHE.clear()
            _HASH_CACHE[key] = digest
    return digest


def scan_artifacts(experiment_dir: Path, task_id: str) -> list[dict[str, object]]:
    """List ordinary files below ``experiment_dir`` with size/sha256/download url.

    Symlinks are skipped and directories never appear in the listing.
    """
    if not experiment_dir.is_dir():
        return []
    entries: list[dict[str, object]] = []
    for path in sorted(experiment_dir.rglob("*")):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
        except OSError:  # pragma: no cover - file vanished mid scan
            continue
        relative = path.relative_to(experiment_dir).as_posix()
        if relative.startswith("."):
            continue
        entries.append(
            {
                "path": relative,
                "size_bytes": int(stat.st_size),
                "sha256": _cached_sha256(path, stat),
                "download_url": f"/tasks/{task_id}/artifacts/{quote(relative)}",
            }
        )
    return entries
