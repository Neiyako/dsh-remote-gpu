"""``generic_python`` executor: run a whitelisted script without a shell.

Security rules (contract §3.3):
- the script must resolve inside ``<root>/scripts/jobs`` and end with ``.py``;
- ``args`` is a ``list[str]`` passed as argv (never concatenated into a string);
- ``env`` may not override ``PATH`` / ``PYTHONPATH`` and names are validated;
- the subprocess is started with argv only (``shell=False``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

from ..errors import ExecutorError
from ..models import GenericPythonConfig
from ..paths import resolve_generic_script
from ..utils import write_json_atomic
from .context import ExecutorContext


def _safe_child_env() -> dict[str, str]:
    env = os.environ.copy()
    # Never leak the worker token to job scripts.
    env.pop("DSH_GPU_WORKER_TOKEN", None)
    return env


def run_generic(ctx: ExecutorContext) -> dict[str, Any]:
    config = GenericPythonConfig.model_validate(ctx.config)
    script_path = resolve_generic_script(ctx.root, config.script)
    if not script_path.is_file():
        raise ExecutorError("SCRIPT_NOT_FOUND", f"Script not found: {config.script!r}")

    env = _safe_child_env()
    env.update(config.env)

    command = [sys.executable, str(script_path), *config.args]
    ctx.log(f"[generic] cwd={ctx.experiment_dir}")
    ctx.log(f"[generic] argv={command!r}")

    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(ctx.experiment_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=False,
        )
    except OSError as exc:
        raise ExecutorError("SCRIPT_SPAWN_FAILED", f"Failed to start script: {exc}") from exc

    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip("\n"), flush=True)
    return_code = process.wait()
    duration = round(time.monotonic() - started, 3)

    if return_code != 0:
        raise ExecutorError(
            "SCRIPT_FAILED",
            f"{config.script} exited with code {return_code}",
        )

    metrics = {
        "exit_code": return_code,
        "duration_seconds": duration,
        "script": config.script,
        "num_args": len(config.args),
    }
    write_json_atomic(ctx.experiment_dir / "metrics.json", metrics)
    summary = f"generic_python {config.script} finished with exit code {return_code} in {duration}s"
    ctx.log(f"[generic] {summary}")
    return {"metrics": metrics, "summary": summary}
