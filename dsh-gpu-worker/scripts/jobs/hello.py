#!/usr/bin/env python3
"""Example generic_python job.

Only scripts under ``scripts/jobs/`` can be executed by the worker.  The
worker starts this file with ``python <script> [args...]`` (no shell) and the
job's working directory is its isolated experiment directory.

Examples (submitted through POST /tasks):
    {"task_type": "generic_python", "config": {"script": "jobs/hello.py", "args": ["--x", "1"]}}
    {"task_type": "generic_python", "config": {"script": "hello.py", "args": ["--sleep", "30"]}}
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _parse_args(argv: list[str]) -> dict[str, object]:
    options: dict[str, object] = {"sleep": 0.0, "fail": False}
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--sleep" and index + 1 < len(argv):
            options["sleep"] = float(argv[index + 1])
            index += 2
            continue
        if item == "--fail":
            options["fail"] = True
        index += 1
    return options


def main() -> int:
    argv = sys.argv[1:]
    options = _parse_args(argv)
    print("hello from scripts/jobs/hello.py")
    print(
        json.dumps(
            {
                "args": argv,
                "cwd": os.getcwd(),
                "pid": os.getpid(),
                "foo_env": os.environ.get("FOO"),
                "token_visible": "DSH_GPU_WORKER_TOKEN" in os.environ,
            },
            ensure_ascii=False,
        )
    )

    sleep_seconds = float(options["sleep"] or 0.0)
    if sleep_seconds > 0:
        deadline = time.monotonic() + sleep_seconds
        tick = 0
        while time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            tick += 1
            print(f"tick {tick}")

    if options["fail"]:
        print("intentional failure requested via --fail", file=sys.stderr)
        return 3

    payload = {
        "message": "hello from the DSH GPU worker generic_python executor",
        "args": argv,
        "foo_env": os.environ.get("FOO"),
        "cwd": os.getcwd(),
        "timestamp": time.time(),
    }
    Path("hello.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path("hello.txt").write_text("hello world\n", encoding="utf-8")
    print("wrote hello.json and hello.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
