"""Execution context shared by all executors.

The child process talks to the parent worker over stdout using two internal
markers.  Everything else printed on stdout/stderr becomes a log line.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..logging_utils import PROGRESS_MARKER, RESULT_MARKER


@dataclass
class ExecutorContext:
    task_id: str
    task_type: str
    config: dict[str, Any]
    root: Path
    experiment_dir: Path

    def log(self, message: str) -> None:
        print(message, flush=True)

    def emit_progress(self, payload: dict[str, Any]) -> None:
        print(f"{PROGRESS_MARKER} {json.dumps(payload, default=str)}", flush=True)

    def emit_result(self, payload: dict[str, Any]) -> None:
        print(f"{RESULT_MARKER} {json.dumps(payload, default=str)}", flush=True)

    def path_relative_root(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path)
