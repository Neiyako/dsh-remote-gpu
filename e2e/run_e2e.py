#!/usr/bin/env python3
"""End-to-end harness for the DSH remote GPU worker.

Stdlib only (urllib).  Starts a *real* worker process from
``dsh-gpu-worker/worker.main`` against a throw-away work root and token,
then exercises the frozen HTTP contract in ``docs/CONTRACT.md`` §2:

  auth / health / gpu / generic_python success / logs / result /
  artifacts + sha256 download / path escape / idempotency replay+conflict /
  cancel -> CANCELLED / forced CUDA -> FAILED CUDA_UNAVAILABLE /
  worker restart -> history queryable / wrong port -> connection refused.

The harness is intentionally noisy in its final report: every check records a
short evidence string so ``docs/E2E_TEST_REPORT.md`` can quote real output.

Usage
-----
    python3 e2e/run_e2e.py                 # portable: starts worker, exits 0/1
    python3 e2e/run_e2e.py --keep-temp     # keep temp root for inspection
    python3 e2e/run_e2e.py --json out.json # also write machine-readable summary
    python3 e2e/run_e2e.py --port 8766     # override port (default 8766)

Environment notes
-----------------
* Worker interpreter: ``<project>/.venv-e2e/bin/python`` when present
  (torch CPU + fastapi), else system ``python3`` (torch optional: the forced
  CUDA check only needs ``torch.cuda.is_available()``, and is skipped as
  NOT_RUN when the interpreter cannot import torch and cannot report CUDA).
* Port 8766 is used by default so a manually started worker on 8765 does not
  clash.  The harness never touches the DSH profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKER_DIR = PROJECT_DIR / "dsh-gpu-worker"
HOST = "127.0.0.1"
DEFAULT_PORT = 8766
WRONG_PORT = 8767
STARTUP_TIMEOUT = 90.0

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"

# macOS urllib picks up the *system* HTTP proxy, which would turn a request to a
# dead 127.0.0.1 port into a proxy 502 instead of a real connection failure.
# Always talk to localhost directly.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

results: list[dict[str, object]] = []
_state: dict[str, object] = {}


# --------------------------------------------------------------------------------------
# infrastructure
# --------------------------------------------------------------------------------------


class Response:
    def __init__(self, status: int, headers: dict, body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.text)


class ConnectionFailed(Exception):
    """Raised when the socket-level connection fails (wrong port / down)."""


def base_url(port: int | None = None) -> str:
    return f"http://{HOST}:{port or _state['port']}"  # type: ignore[arg-type]


def _lower_headers(headers) -> dict:
    """HTTP headers are case-insensitive; store lower-case keys for lookups."""
    try:
        return {str(key).lower(): value for key, value in headers.items()}
    except AttributeError:  # pragma: no cover - defensive
        return {}


def http(
    method: str,
    path: str,
    *,
    token: str | None = None,
    body: dict | None = None,
    port: int | None = None,
    timeout: float = 20.0,
) -> Response:
    """One HTTP request.  Returns any HTTP status; raises ConnectionFailed."""
    url = base_url(port) + path
    data = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return Response(response.status, _lower_headers(response.headers), response.read())
    except urllib.error.HTTPError as exc:
        return Response(exc.code, _lower_headers(exc.headers or {}), exc.read())
    except (urllib.error.URLError, ConnectionError, socket.timeout) as exc:
        reason = getattr(exc, "reason", exc)
        raise ConnectionFailed(f"{type(exc).__name__}: {reason}") from exc


def auth_http(method: str, path: str, **kwargs) -> Response:
    kwargs.setdefault("token", _state["token"])
    return http(method, path, **kwargs)


def find_python() -> Path:
    candidates = [
        PROJECT_DIR / ".venv-e2e" / "bin" / "python",
        PROJECT_DIR / ".venv-e2e" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def write_job_scripts(root: Path) -> None:
    jobs = root / "scripts" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "e2e_hello.py").write_text(
        """\
import argparse, os, sys, time

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="world")
parser.add_argument("--sleep", type=float, default=0.0)
parser.add_argument("--fail", action="store_true")
args = parser.parse_args()

print(f"e2e-hello name={args.name} flag={os.environ.get('E2E_FLAG', '<unset>')}", flush=True)
for index in range(3):
    print(f"tick {index}", flush=True)
if args.sleep:
    print("sleeping", flush=True)
    time.sleep(args.sleep)
print("e2e-hello done", flush=True)
sys.exit(3 if args.fail else 0)
""",
        encoding="utf-8",
    )


def wait_health(proc: subprocess.Popen, log_path: Path, timeout: float = STARTUP_TIMEOUT) -> dict:
    deadline = time.monotonic() + timeout
    last_error = "no attempt"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                pass
            raise RuntimeError(
                f"worker exited early with code {proc.returncode}; log tail:\n{tail}"
            )
        try:
            response = http("GET", "/health", token=_state["token"], timeout=2.0)
            if response.status == 200:
                return response.json()
            last_error = f"HTTP {response.status}: {response.text[:200]}"
        except ConnectionFailed as exc:
            last_error = str(exc)
        except (urllib.error.URLError, OSError) as exc:  # pragma: no cover - defensive
            last_error = str(exc)
        time.sleep(0.2)
    raise RuntimeError(f"worker did not become healthy in {timeout}s: {last_error}")


def start_worker(root: Path) -> tuple[subprocess.Popen, object, Path]:
    env = os.environ.copy()
    env.update(
        {
            "DSH_GPU_WORKER_TOKEN": str(_state["token"]),
            "DSH_GPU_WORKER_ROOT": str(root),
            "DSH_GPU_WORKER_HOST": HOST,
            "DSH_GPU_WORKER_PORT": str(_state["port"]),
            "DSH_GPU_WORKER_MAX_CONCURRENT_TASKS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.pop("DSH_GPU_WORKER_DB", None)
    # Keep the child env free of unrelated PYTHONPATH overrides from the caller.
    env.pop("PYTHONPATH", None)
    log_path = root / "worker-e2e.log"
    log_handle = log_path.open("ab")
    proc = subprocess.Popen(
        [str(_state["python"]), "-u", "-m", "worker.main"],
        cwd=str(WORKER_DIR),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    try:
        health = wait_health(proc, log_path)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_handle.close()
        raise
    return proc, log_handle, log_path


def stop_worker() -> None:
    proc = _state.get("proc")
    handle = _state.get("log_handle")
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        except ProcessLookupError:  # pragma: no cover
            pass
    if handle is not None:
        handle.close()
    _state["proc"] = None
    _state["log_handle"] = None


def wait_task(task_id: str, statuses: set[str], timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        response = auth_http("GET", f"/tasks/{task_id}")
        if response.status != 200:
            raise AssertionError(f"GET /tasks/{task_id} -> HTTP {response.status}: {response.text[:300]}")
        last = response.json()
        if last["status"] in statuses:
            return last
        time.sleep(0.15)
    raise AssertionError(f"timeout waiting for {sorted(statuses)}; last={json.dumps(last)[:400]}")


TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}


def submit(task_type: str, config: dict, **extra) -> tuple[Response, dict]:
    payload = {"task_type": task_type, "config": config}
    payload.update(extra)
    response = auth_http("POST", "/tasks", body=payload)
    return response, (response.json() if response.text.strip() else {})


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------


def record(check_id: str, name: str, status: str, evidence: str) -> None:
    results.append({"id": check_id, "name": name, "status": status, "evidence": evidence})
    print(f"[{status:7s}] {check_id} {name} :: {evidence}", flush=True)


def check(check_id: str, name: str):
    """Decorator: wrap one check so it records PASS/FAIL with evidence on call."""

    def decorator(func):
        def run_check() -> None:
            try:
                evidence = func()
                record(check_id, name, PASS, str(evidence))
            except Exception as exc:  # noqa: BLE001 - a check failure must be reported
                detail = f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, AssertionError):
                    detail += " | " + traceback.format_exc(limit=2).replace("\n", " ")[:500]
                record(check_id, name, FAIL, detail[:900])

        run_check.__name__ = getattr(func, "__name__", f"check_{check_id}")
        return run_check

    return decorator


# --- ordered checks -------------------------------------------------------------------


@check("T01", "auth rejects missing/wrong token on /health and /gpu")
def t01_auth_reject() -> str:
    missing = http("GET", "/health", token=None, timeout=5)
    assert missing.status == 401, f"expected 401 without token, got {missing.status}: {missing.text[:200]}"
    code = missing.json().get("error", {}).get("code")
    assert code == "UNAUTHORIZED", f"expected UNAUTHORIZED, got {code!r}"

    wrong = http("GET", "/gpu", token="definitely-wrong-token", timeout=5)
    assert wrong.status == 401, f"expected 401 with wrong token, got {wrong.status}"
    assert wrong.json().get("error", {}).get("code") == "UNAUTHORIZED"

    x_header = http("GET", "/health", token=None, timeout=5)
    url = base_url() + "/health"
    req = urllib.request.Request(url, headers={"X-Worker-Token": str(_state["token"])})
    with _OPENER.open(req, timeout=5) as resp:
        assert resp.status == 200, f"X-Worker-Token should authenticate, got {resp.status}"
    assert x_header.status == 401
    return "401 UNAUTHORIZED without token; 401 with wrong token; X-Worker-Token header -> 200"


@check("T02", "GET /health returns ok + cuda_available + counters")
def t02_health() -> str:
    response = auth_http("GET", "/health")
    assert response.status == 200, f"HTTP {response.status}: {response.text[:300]}"
    payload = response.json()
    for field in (
        "status",
        "worker_id",
        "version",
        "time",
        "uptime_seconds",
        "cuda_available",
        "active_tasks",
        "queued_tasks",
        "max_concurrent_tasks",
    ):
        assert field in payload, f"/health missing field {field!r}: {payload}"
    assert payload["status"] == "ok", payload["status"]
    assert isinstance(payload["cuda_available"], bool)
    assert payload["max_concurrent_tasks"] == 1
    _state["cuda_available"] = payload["cuda_available"]
    return (
        f"status=ok version={payload['version']} cuda_available={payload['cuda_available']} "
        f"active={payload['active_tasks']} queued={payload['queued_tasks']}"
    )


@check("T03", "GET /gpu reports CUDA truthfully (no fake devices)")
def t03_gpu() -> str:
    response = auth_http("GET", "/gpu")
    assert response.status == 200, f"HTTP {response.status}: {response.text[:300]}"
    payload = response.json()
    for field in ("cuda_available", "device_count", "devices", "nvidia_smi_available", "error"):
        assert field in payload, f"/gpu missing field {field!r}: {payload}"
    assert isinstance(payload["devices"], list)
    if payload["cuda_available"]:
        assert payload["device_count"] >= 1 and payload["devices"], payload
        device = payload["devices"][0]
        for field in ("index", "name", "capability", "total_memory_mb"):
            assert field in device, f"device missing {field!r}: {device}"
        return f"cuda_available=True devices={len(payload['devices'])} first={device['name']}"
    assert payload["device_count"] == 0, f"cuda unavailable but device_count={payload['device_count']}"
    assert payload["devices"] == [], payload["devices"]
    return f"cuda_available=False device_count=0 devices=[] error={str(payload['error'])[:120]!r}"


@check("T04", "generic_python task SUCCEEDED (submit 201, wait terminal)")
def t04_generic_success() -> str:
    response, view = submit(
        "generic_python",
        {"script": "jobs/e2e_hello.py", "args": ["--name", "e2e"], "env": {"E2E_FLAG": "yes"}},
        name="e2e-generic",
    )
    assert response.status == 201, f"submit expected 201, got {response.status}: {response.text[:300]}"
    task_id = view["task_id"]
    assert view["status"] in {"PENDING", "RUNNING"}, view["status"]
    final = wait_task(task_id, TERMINAL, timeout=90)
    assert final["status"] == "SUCCEEDED", f"expected SUCCEEDED, got {final['status']}: {json.dumps(final.get('error'))[:400]}"
    _state["generic_task"] = task_id
    return f"task_id={task_id} status=SUCCEEDED created={view['status']}"


@check("T05", "GET /tasks/{id}/logs has script output, no internal markers")
def t05_logs() -> str:
    task_id = _state["generic_task"]
    response = auth_http("GET", f"/tasks/{task_id}/logs?offset=0&limit=500")
    assert response.status == 200, f"HTTP {response.status}: {response.text[:300]}"
    payload = response.json()
    for field in ("task_id", "offset", "next_offset", "lines", "eof"):
        assert field in payload, f"logs missing {field!r}"
    texts = [line["text"] for line in payload["lines"]]
    joined = "\n".join(texts)
    assert "e2e-hello name=e2e" in joined, f"script output missing: {joined[:400]!r}"
    assert "flag=yes" in joined, f"env passthrough missing: {joined[:400]!r}"
    assert "@@PROGRESS@@" not in joined and "@@RESULT@@" not in joined, "internal marker leaked"
    assert payload["lines"] and all({"seq", "ts", "stream", "text"} <= set(line) for line in payload["lines"])

    page = auth_http("GET", f"/tasks/{task_id}/logs?offset=0&limit=2").json()
    assert len(page["lines"]) == 2, f"limit=2 returned {len(page['lines'])} lines"
    assert page["next_offset"] == 2, page
    assert page["eof"] is False, page
    tail = auth_http("GET", f"/tasks/{task_id}/logs?offset={page['next_offset']}&limit=2000").json()
    assert tail["offset"] == page["next_offset"]
    assert tail["eof"] is True, tail
    return f"lines={len(payload['lines'])} eof={payload['eof']} paging offset=2 ok marker_free=True"


@check("T06", "GET /result returns metrics/artifacts/experiment_dir for SUCCEEDED")
def t06_result() -> str:
    task_id = _state["generic_task"]
    response = auth_http("GET", f"/tasks/{task_id}/result")
    assert response.status == 200, f"HTTP {response.status}: {response.text[:300]}"
    payload = response.json()
    assert payload["status"] == "SUCCEEDED", payload["status"]
    result = payload["result"]
    for field in ("metrics", "artifacts", "experiment_dir", "summary"):
        assert field in result, f"result missing {field!r}: {result}"
    assert result["metrics"]["exit_code"] == 0, result["metrics"]
    assert result["experiment_dir"] == f"experiments/{task_id}", result["experiment_dir"]
    _state["generic_result"] = result
    return (
        f"status=SUCCEEDED exit_code={result['metrics']['exit_code']} "
        f"experiment_dir={result['experiment_dir']} artifacts={len(result['artifacts'])}"
    )


@check("T07", "GET /artifacts lists files with sha256 and download matches bytes")
def t07_artifacts_sha256() -> str:
    task_id = _state["generic_task"]
    response = auth_http("GET", f"/tasks/{task_id}/artifacts")
    assert response.status == 200, f"HTTP {response.status}: {response.text[:300]}"
    artifacts = response.json()["artifacts"]
    assert artifacts, "artifactId list is empty"
    by_path = {entry["path"]: entry for entry in artifacts}
    assert "metrics.json" in by_path, f"metrics.json missing from artifacts: {sorted(by_path)}"
    entry = by_path["metrics.json"]
    for field in ("path", "size_bytes", "sha256", "download_url"):
        assert field in entry, f"artifact missing {field!r}: {entry}"

    download = auth_http("GET", f"/tasks/{task_id}/artifacts/metrics.json")
    assert download.status == 200, f"download HTTP {download.status}: {download.text[:200]}"
    assert "attachment" in download.headers.get("content-disposition", "").lower(), (
        f"missing Content-Disposition attachment: {download.headers.get('content-disposition')!r}"
    )
    digest = hashlib.sha256(download.body).hexdigest()
    assert digest == entry["sha256"], f"sha256 mismatch: downloaded={digest} listed={entry['sha256']}"
    assert len(download.body) == entry["size_bytes"], (
        f"size mismatch: downloaded={len(download.body)} listed={entry['size_bytes']}"
    )
    parsed = json.loads(download.body.decode("utf-8"))
    assert parsed["exit_code"] == 0, parsed
    return f"artifacts={len(artifacts)} metrics.json sha256={digest[:16]}… bytes={entry['size_bytes']} match=True"


@check("T08", "artifact path traversal rejected with 400 PATH_ESCAPE")
def t08_path_escape() -> str:
    task_id = _state["generic_task"]
    response = auth_http("GET", f"/tasks/{task_id}/artifacts/..%2f..%2fworker.db")
    assert response.status == 400, f"expected 400, got {response.status}: {response.text[:300]}"
    code = response.json().get("error", {}).get("code")
    assert code == "PATH_ESCAPE", f"expected PATH_ESCAPE, got {code!r}"
    return "GET artifacts/../../worker.db -> 400 PATH_ESCAPE"


@check("T09", "idempotency: same key+payload replay 200 same id; different payload 409")
def t09_idempotency() -> str:
    key = f"e2e-idem-{secrets.token_hex(6)}"
    config = {"script": "jobs/e2e_hello.py", "args": ["--name", "idem"]}
    first, first_view = submit("generic_python", config, idempotency_key=key)
    assert first.status == 201, f"first submit expected 201, got {first.status}"
    task_id = first_view["task_id"]

    replay, replay_view = submit("generic_python", config, idempotency_key=key)
    assert replay.status == 200, f"replay expected 200, got {replay.status}: {replay.text[:300]}"
    assert replay_view["task_id"] == task_id, f"replay task_id mismatch: {replay_view['task_id']} != {task_id}"

    conflict, conflict_view = submit(
        "generic_python", {"script": "jobs/e2e_hello.py", "args": ["--name", "different"]}, idempotency_key=key
    )
    assert conflict.status == 409, f"conflict expected 409, got {conflict.status}: {conflict.text[:300]}"
    code = conflict_view.get("error", {}).get("code")
    assert code == "IDEMPOTENCY_CONFLICT", f"expected IDEMPOTENCY_CONFLICT, got {code!r}"
    wait_task(task_id, TERMINAL, timeout=90)
    return f"key={key[:12]}… first=201 replay=200 same_id=True conflict=409 IDEMPOTENCY_CONFLICT"


@check("T10", "result before terminal -> 409 TASK_NOT_FINISHED")
def t10_result_not_finished() -> str:
    response, view = submit(
        "generic_python",
        {"script": "jobs/e2e_hello.py", "args": ["--sleep", "120"]},
        name="e2e-cancel-me",
    )
    assert response.status == 201, f"submit HTTP {response.status}: {response.text[:300]}"
    task_id = view["task_id"]
    _state["cancel_task"] = task_id
    running = wait_task(task_id, {"RUNNING"} | TERMINAL, timeout=60)
    assert running["status"] == "RUNNING", f"expected RUNNING, got {running['status']}"
    early = auth_http("GET", f"/tasks/{task_id}/result")
    assert early.status == 409, f"expected 409 for running task, got {early.status}: {early.text[:300]}"
    code = early.json().get("error", {}).get("code")
    assert code == "TASK_NOT_FINISHED", f"expected TASK_NOT_FINISHED, got {code!r}"
    details_status = early.json()["error"].get("details", {}).get("status")
    assert details_status == "RUNNING", f"details.status={details_status!r}"
    return f"task_id={task_id} running; result -> 409 TASK_NOT_FINISHED details.status=RUNNING"


@check("T11", "cancel RUNNING task -> CANCELLED (never FAILED), second cancel 409")
def t11_cancel() -> str:
    task_id = _state["cancel_task"]
    response = auth_http("POST", f"/tasks/{task_id}/cancel")
    assert response.status == 200, f"cancel HTTP {response.status}: {response.text[:300]}"
    final = wait_task(task_id, TERMINAL, timeout=60)
    assert final["status"] == "CANCELLED", f"cancel ended as {final['status']}: {json.dumps(final.get('error'))[:300]}"
    error = final.get("error") or {}
    assert error.get("code") == "CANCELLED", f"unexpected cancel error: {error}"
    again = auth_http("POST", f"/tasks/{task_id}/cancel")
    assert again.status == 409, f"second cancel expected 409, got {again.status}"
    assert again.json()["error"]["code"] == "TASK_ALREADY_FINISHED", again.text[:300]
    return "cancel -> 200, terminal status=CANCELLED error.code=CANCELLED, second cancel=409 TASK_ALREADY_FINISHED"


@check("T12", "forced device=cuda training FAILED with CUDA_UNAVAILABLE (no CPU fallback)")
def t12_cuda_unavailable() -> str:
    response, view = submit(
        "training",
        {
            "device": "cuda",
            "epochs": 1,
            "batch_size": 8,
            "dataset": {
                "dataset_id": "synthetic",
                "format": "synthetic",
                "num_samples": 40,
                "num_features": 8,
                "num_classes": 3,
            },
            "model": {"type": "mlp", "input_dim": 8, "hidden_dims": [8], "num_classes": 3},
        },
        name="e2e-cuda-forced",
    )
    assert response.status == 201, f"submit HTTP {response.status}: {response.text[:300]}"
    task_id = view["task_id"]
    final = wait_task(task_id, TERMINAL, timeout=120)
    assert final["status"] == "FAILED", f"expected FAILED, got {final['status']}: {json.dumps(final.get('result'))[:300]}"
    error = final.get("error") or {}
    assert error.get("code") == "CUDA_UNAVAILABLE", f"expected CUDA_UNAVAILABLE, got {error.get('code')!r}: {str(error.get('message'))[:300]}"
    _state["cuda_task"] = task_id
    return f"task_id={task_id} status=FAILED error.code=CUDA_UNAVAILABLE message={str(error.get('message'))[:120]!r}"


@check("T16", "CPU training SUCCEEDED: progress while RUNNING, checkpoints, downloadable weights")
def t16_cpu_training() -> str:
    response, view = submit(
        "training",
        {
            "device": "cpu",
            "epochs": 6,
            "batch_size": 32,
            "learning_rate": 0.01,
            "dataset": {
                "dataset_id": "synthetic",
                "format": "synthetic",
                "num_samples": 1200,
                "num_features": 8,
                "num_classes": 3,
            },
            "model": {"type": "mlp", "input_dim": 8, "hidden_dims": [16, 8], "num_classes": 3},
            "checkpoint": {"enabled": True, "every_n_epochs": 1, "save_best": True},
        },
        name="e2e-training-cpu",
        timeout_seconds=600,
    )
    assert response.status == 201, f"submit HTTP {response.status}: {response.text[:300]}"
    task_id = view["task_id"]

    # Poll like a client would; record progress snapshots that appear while RUNNING.
    deadline = time.monotonic() + 240
    seen_epochs: list[int] = []
    final: dict = {}
    while time.monotonic() < deadline:
        current = auth_http("GET", f"/tasks/{task_id}")
        assert current.status == 200, current.text[:300]
        final = current.json()
        progress = final.get("progress") or {}
        if isinstance(progress.get("epoch"), int) and progress["epoch"] not in seen_epochs:
            seen_epochs.append(progress["epoch"])
        if final["status"] in TERMINAL:
            break
        time.sleep(0.05)
    assert final["status"] == "SUCCEEDED", f"expected SUCCEEDED, got {final['status']}: {json.dumps(final.get('error'))[:400]}"
    progress = final.get("progress") or {}
    assert progress.get("epoch") == 6, f"final progress epoch={progress.get('epoch')}: {progress}"
    assert progress.get("total_epochs") == 6, progress
    assert isinstance(progress.get("train_loss"), (int, float)), progress
    assert seen_epochs, "no progress observed while the task was running"

    payload = auth_http("GET", f"/tasks/{task_id}/result").json()
    assert payload["status"] == "SUCCEEDED"
    result = payload["result"]
    artifacts = {entry["path"]: entry for entry in result["artifacts"]}
    for expected in ("model.pt", "best_model.pt", "metrics.json", "config.json", "train.log", "report.md"):
        assert expected in artifacts, f"missing artifact {expected}: {sorted(artifacts)}"

    # Download the model weights and verify the bytes against the advertised sha256.
    weights = auth_http("GET", f"/tasks/{task_id}/artifacts/model.pt")
    assert weights.status == 200, f"model.pt download HTTP {weights.status}"
    assert len(weights.body) == artifacts["model.pt"]["size_bytes"], "model.pt size mismatch"
    digest = hashlib.sha256(weights.body).hexdigest()
    assert digest == artifacts["model.pt"]["sha256"], "model.pt sha256 mismatch"

    metrics = auth_http("GET", f"/tasks/{task_id}/artifacts/metrics.json").json()
    _state["training_task"] = task_id
    return (
        f"task_id={task_id} SUCCEEDED progress_snapshots={seen_epochs} "
        f"progress.loss={progress.get('train_loss'):.4f}/{progress.get('val_loss')} "
        f"artifacts={len(artifacts)} model.pt sha256={digest[:16]}… bytes={len(weights.body)} "
        f"metrics.device={metrics.get('device')}"
    )


@check("T13", "wrong port -> socket-level connection failure (no silent success)")
def t13_wrong_port() -> str:
    # Pick a port that is provably free right now (a fixed "wrong" port can be
    # occupied by an unrelated local service, which would make this flaky).
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind((HOST, 0))
    wrong_port = probe.getsockname()[1]
    probe.close()
    try:
        http("GET", "/health", token=_state["token"], port=wrong_port, timeout=3)
    except ConnectionFailed as exc:
        return f"http://{HOST}:{wrong_port}/health (free port) -> ConnectionFailed ({str(exc)[:160]})"
    raise AssertionError(f"expected connection failure on free port {wrong_port}, but a response was received")


@check("T14", "worker restart keeps history, results, logs and artifacts")
def t14_restart_history() -> str:
    response, view = submit(
        "generic_python", {"script": "jobs/e2e_hello.py", "args": ["--name", "before-restart"]}, name="e2e-before-restart"
    )
    assert response.status == 201, f"submit HTTP {response.status}: {response.text[:300]}"
    task_id = view["task_id"]
    wait_task(task_id, {"SUCCEEDED"}, timeout=90)
    before = auth_http("GET", f"/tasks/{task_id}").json()
    before_logs = auth_http("GET", f"/tasks/{task_id}/logs").json()["lines"]
    assert before_logs, "no logs before restart"

    stop_worker()
    _state["proc"], _state["log_handle"], _state["log_path"] = start_worker(_state["root"])  # type: ignore[arg-type]
    health = auth_http("GET", "/health").json()
    assert health["status"] == "ok"

    listing = auth_http("GET", "/tasks?limit=200").json()
    ids = {task["task_id"] for task in listing["tasks"]}
    assert task_id in ids, f"task {task_id} missing from history after restart (n={len(ids)})"
    after = auth_http("GET", f"/tasks/{task_id}").json()
    assert after["status"] == "SUCCEEDED", after["status"]
    after_logs = auth_http("GET", f"/tasks/{task_id}/logs").json()["lines"]
    assert after_logs, "logs lost after restart"
    assert [line.get("text") for line in after_logs] == [line.get("text") for line in before_logs], (
        "logs changed after restart"
    )
    result = auth_http("GET", f"/tasks/{task_id}/result").json()
    assert result["status"] == "SUCCEEDED" and result["result"]["metrics"]["exit_code"] == 0
    artifacts = auth_http("GET", f"/tasks/{task_id}/artifacts").json()["artifacts"]
    assert any(entry["path"] == "metrics.json" for entry in artifacts), artifacts
    return (
        f"restarted; history n={len(ids)} contains {task_id}; status/logs/result/artifacts preserved "
        f"recovered_tasks={health.get('recovered_tasks')}"
    )


@check("T15", "invalid submissions rejected with 400 INVALID_REQUEST")
def t15_invalid_submit() -> str:
    bad_type = auth_http("POST", "/tasks", body={"task_type": "shell", "config": {}})
    assert bad_type.status == 400, f"bad task_type expected 400, got {bad_type.status}: {bad_type.text[:300]}"
    code = bad_type.json()["error"]["code"]
    assert code == "INVALID_REQUEST", code

    bad_timeout = auth_http(
        "POST", "/tasks", body={"task_type": "generic_python", "config": {"script": "jobs/e2e_hello.py"}, "timeout_seconds": 5}
    )
    assert bad_timeout.status == 400, f"bad timeout expected 400, got {bad_timeout.status}"
    assert bad_timeout.json()["error"]["code"] == "INVALID_REQUEST"

    escape = auth_http(
        "POST", "/tasks", body={"task_type": "generic_python", "config": {"script": "../../etc/passwd.py"}}
    )
    assert escape.status == 400, f"path escape expected 400, got {escape.status}: {escape.text[:300]}"
    assert escape.json()["error"]["code"] == "PATH_ESCAPE", escape.text[:300]
    return "shell task_type -> 400 INVALID_REQUEST; timeout=5 -> 400; script escape -> 400 PATH_ESCAPE"


CHECKS = [
    t01_auth_reject,
    t02_health,
    t03_gpu,
    t04_generic_success,
    t05_logs,
    t06_result,
    t07_artifacts_sha256,
    t08_path_escape,
    t09_idempotency,
    t10_result_not_finished,
    t11_cancel,
    t12_cuda_unavailable,
    t16_cpu_training,
    t13_wrong_port,
    t14_restart_history,
    t15_invalid_submit,
]


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def summarize(started_at: str, finished_at: str) -> dict:
    counts = {PASS: 0, FAIL: 0, NOT_RUN: 0}
    for result in results:
        counts[str(result["status"])] = counts.get(str(result["status"]), 0) + 1
    return {
        "harness": "e2e/run_e2e.py",
        "base_url": base_url(),
        "worker_python": str(_state.get("python")),
        "worker_root": str(_state.get("root")),
        "started_at": started_at,
        "finished_at": finished_at,
        "counts": counts,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end harness for the DSH remote GPU worker")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="worker port (default 8766)")
    parser.add_argument("--keep-temp", action="store_true", help="keep the temporary work root")
    parser.add_argument("--json", dest="json_out", default=None, help="write the JSON summary to this path")
    args = parser.parse_args(argv)

    started_at = datetime.now(timezone.utc).isoformat()
    _state["port"] = args.port
    _state["token"] = secrets.token_urlsafe(32)
    _state["python"] = find_python()

    root = Path(tempfile.mkdtemp(prefix="dsh-gpu-e2e-"))
    _state["root"] = root
    write_job_scripts(root)
    print(f"[e2e] temp root   : {root}")
    print(f"[e2e] worker python: {_state['python']}")
    print(f"[e2e] worker URL   : {base_url()}")
    print(f"[e2e] token        : <redacted {len(str(_state['token']))} chars>")

    try:
        _state["proc"], _state["log_handle"], _state["log_path"] = start_worker(root)
        print("[e2e] worker healthy; running checks\n", flush=True)
        for func in CHECKS:
            func()
    except Exception as exc:  # noqa: BLE001 - startup failure is a hard FAIL
        record("T00", "worker startup", FAIL, f"{type(exc).__name__}: {exc}"[:900])
    finally:
        stop_worker()
        if not args.keep_temp:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"[e2e] kept temp root: {root}")

    finished_at = datetime.now(timezone.utc).isoformat()
    summary = summarize(started_at, finished_at)

    print("\n" + "=" * 100)
    print(f"{'ID':<4} {'STATUS':<8} CHECK")
    print("-" * 100)
    for result in results:
        print(f"{result['id']:<4} {result['status']:<8} {result['name']}")
    counts = summary["counts"]
    print("-" * 100)
    print(f"TOTAL: {counts.get(PASS, 0)} PASS, {counts.get(FAIL, 0)} FAIL, {counts.get(NOT_RUN, 0)} NOT_RUN")
    print("=" * 100)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[e2e] JSON summary written to {args.json_out}")

    return 0 if counts.get(FAIL, 0) == 0 and counts.get(PASS, 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
