# dsh-gpu-worker

Independent DSH remote GPU worker: a small FastAPI service that accepts
`training` / `inference` / `generic_python` tasks, runs each one in its own
subprocess, and exposes logs, progress, results and downloadable artifacts.

It is the server side of the `dsh-remote-gpu` DSH plugin. The frozen interface
contract is [`../docs/CONTRACT.md`](../docs/CONTRACT.md); this README covers
installation, operation and security.

- Python 3.10+ (tested with 3.13), FastAPI + Pydantic v2 + SQLite
- PyTorch is an **optional runtime dependency**: the server, generic jobs, health
  and all tests work without it; only `training` / `inference` need torch.
- macOS / Linux / Windows (PowerShell scripts provided)

## Quick start (macOS / Linux)

```bash
cd dsh-gpu-worker

python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt          # add -r requirements-dev.txt for tests

cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # paste into .env

python -m worker                          # listens on 127.0.0.1:8765
```

Or `./scripts/start.sh`, which loads `.env` and refuses to start without a token.

## Quick start (Windows PowerShell)

```powershell
cd dsh-gpu-worker

# 1) install: creates .venv, installs deps, creates .env, prints CUDA diagnostic
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1 -WithTorch

# 2) edit .env and set DSH_GPU_WORKER_TOKEN (never commit .env)

# 3) start
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

`install.ps1` without `-WithTorch` installs only the web stack; training and
inference then fail with `TORCH_UNAVAILABLE` until torch is installed.

## CUDA / GPU check

```bash
# inside the venv on the GPU machine
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
nvidia-smi
```

RTX 50-series (Blackwell, including RTX 5060 Laptop) needs a CUDA 12.8+ PyTorch wheel; CUDA 12.4 wheels do not contain sm_120 kernels. Use cu128 (or newer) and a recent NVIDIA driver:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then verify through the worker:

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/gpu | python -m json.tool
```

`cuda_available: false` is a normal state on CPU-only machines; `/health` stays
`status: "ok"` and reports `cuda_available` honestly. A task that explicitly sets
`"device": "cuda"` on such a machine fails with `CUDA_UNAVAILABLE` — it never
silently runs on the CPU.

## Configuration

All configuration is environment variables (or a `.env` file loaded by the start
scripts):

| Variable | Default | Meaning |
|---|---|---|
| `DSH_GPU_WORKER_TOKEN` | **required** | Bearer token for every API call. Startup is refused when missing or shorter than 8 chars. |
| `DSH_GPU_WORKER_HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` only for LAN access with a firewall. |
| `DSH_GPU_WORKER_PORT` | `8765` | HTTP port. |
| `DSH_GPU_WORKER_ROOT` | project dir | Work root. Experiments, datasets, logs, DB live below it; nothing may escape it. |
| `DSH_GPU_WORKER_DB` | `<root>/worker.db` | SQLite database path. |
| `DSH_GPU_WORKER_MAX_CONCURRENT_TASKS` | `1` | Parallel job processes. |
| `DSH_GPU_WORKER_ALLOW_GENERIC_PYTHON` | `true` | Enable/disable the `generic_python` task type. |

Directory layout below the root:

```
<root>/
├── worker.db                 # task history (SQLite, WAL)
├── experiments/<task_id>/    # one isolated dir per task: job.json, worker.log,
│                             #   result.json / error.json, artifacts (model.pt, metrics.json, ...)
├── datasets/                 # csv / npy datasets (read-only input)
└── scripts/jobs/             # ONLY place generic_python scripts may live
```

## API summary

Every endpoint requires the token: `Authorization: Bearer <token>` or
`X-Worker-Token: <token>`. All errors share one body:
`{"error":{"code":"...","message":"...","details":{}}}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, version, uptime, CUDA flag, active/queued tasks |
| GET | `/gpu` | CUDA devices (memory, utilization, temperature, driver) |
| GET | `/version` | worker version/id |
| POST | `/tasks` | submit a task, returns immediately with `TaskView` (201 / 200 on idempotent replay / 409 on conflict) |
| GET | `/tasks?status=&limit=` | task history (survives restarts) |
| GET | `/tasks/{id}` | `TaskView` (status, progress, error, result) |
| POST | `/tasks/{id}/cancel` | cancel PENDING/RUNNING (kills the process tree) |
| GET | `/tasks/{id}/logs?offset=&limit=` | persisted JSONL logs, paginated |
| GET | `/tasks/{id}/result` | terminal result (409 `TASK_NOT_FINISHED` while running) |
| GET | `/tasks/{id}/artifacts` | artifact list with size + sha256 |
| GET | `/tasks/{id}/artifacts/{path}` | binary download (`attachment`; path escape → 400 `PATH_ESCAPE`) |
| GET | `/datasets` | datasets below `datasets/` |

Task states: `PENDING → RUNNING → SUCCEEDED | FAILED | CANCELLED`.

### Submit examples

```bash
TOKEN=...; BASE=http://127.0.0.1:8765
H="Authorization: Bearer $TOKEN"

# generic python (only whitelisted scripts under scripts/jobs/)
curl -s -X POST "$BASE/tasks" -H "$H" -H 'Content-Type: application/json' -d '{
  "task_type": "generic_python",
  "config": {"script": "jobs/hello.py", "args": ["--x", "1"], "env": {"FOO": "bar"}},
  "timeout_seconds": 120, "idempotency_key": "demo-1"
}'

# training (needs torch)
curl -s -X POST "$BASE/tasks" -H "$H" -H 'Content-Type: application/json' -d '{
  "task_type": "training",
  "config": {
    "model": {"type": "mlp", "input_dim": 8, "hidden_dims": [64, 32], "num_classes": 3},
    "dataset": {"format": "synthetic", "num_samples": 600, "num_features": 8, "num_classes": 3},
    "epochs": 10, "batch_size": 32, "learning_rate": 0.001,
    "device": "auto",
    "checkpoint": {"enabled": true, "every_n_epochs": 1, "save_best": true},
    "early_stopping": {"enabled": true, "patience": 3, "monitor": "val_loss"}
  },
  "timeout_seconds": 3600
}'

# inference from a previous training task
curl -s -X POST "$BASE/tasks" -H "$H" -H 'Content-Type: application/json' -d '{
  "task_type": "inference",
  "config": {"source_task_id": "<training-task-id>", "checkpoint": "best_model.pt",
             "dataset": {"format": "synthetic", "num_samples": 100, "num_features": 8, "num_classes": 3}}
}'
```

Training writes `model.pt`, `best_model.pt` (when `save_best`), `metrics.json`,
`config.json`, `train.log`, `report.md` and one persisted progress record per
epoch. Inference writes `metrics.json`, `predictions.csv`, `report.md`.

## Security

- **Token required on every request.** The worker refuses to start without
  `DSH_GPU_WORKER_TOKEN`; use a long random value (`secrets.token_urlsafe(32)`).
- **Local bind by default.** `127.0.0.1` means only the local machine can reach
  the worker. For LAN access set `DSH_GPU_WORKER_HOST=0.0.0.0` *and* add a
  firewall rule, then restrict it to the trusted subnet. Prefer an SSH tunnel /
  reverse proxy with TLS over exposing plain HTTP.
- **No shell interface.** Jobs run as `python -m worker.job_entry job.json` with
  argv only; `generic_python` scripts are executed as
  `python <root>/scripts/jobs/<name>.py <args...>` (`shell=False`). Nothing the
  API accepts is ever concatenated into a command string.
- **generic_python is code execution by design.** The whitelist (`scripts/jobs/`
  only, `.py` only, no `..`, no absolute paths) limits the blast radius but any
  script there runs with the worker user's privileges. Keep the directory
  writable only by trusted administrators, or set
  `DSH_GPU_WORKER_ALLOW_GENERIC_PYTHON=false` to disable the task type.
- **Path jail.** All artifact/log/script paths are resolved and must stay inside
  the configured root; escapes return 400 `PATH_ESCAPE`.
- **Token is not inherited by jobs.** The runner strips
  `DSH_GPU_WORKER_TOKEN` from the child environment.
- **Windows firewall** (open 8765 only when using LAN mode):

  ```powershell
  New-NetFirewallRule -DisplayName "DSH GPU Worker" -Direction Inbound `
    -Protocol TCP -LocalPort 8765 -Action Allow -Profile Private
  ```

## Running as a Windows service / at boot

Option A — Task Scheduler (built in):

```powershell
schtasks /Create /TN "DSH GPU Worker" /SC ONSTART /RU SYSTEM /RL HIGHEST `
  /TR "powershell -ExecutionPolicy Bypass -File C:\dsh-gpu-worker\scripts\start.ps1"
schtasks /Run /TN "DSH GPU Worker"
```

Option B — NSSM:

```powershell
nssm install "DSH GPU Worker" "C:\dsh-gpu-worker\.venv\Scripts\python.exe" "-m worker"
nssm set "DSH GPU Worker" AppDirectory "C:\dsh-gpu-worker"
nssm set "DSH GPU Worker" AppEnvironmentExtra "DSH_GPU_WORKER_TOKEN=<token>" "DSH_GPU_WORKER_HOST=0.0.0.0" "DSH_GPU_WORKER_PORT=8765"
nssm set "DSH GPU Worker" AppStdout "C:\dsh-gpu-worker\logs\service.out.log"
nssm set "DSH GPU Worker" AppStderr "C:\dsh-gpu-worker\logs\service.err.log"
nssm start "DSH GPU Worker"
```

Use the same recipe with `systemd` on Linux (`ExecStart=<venv>/bin/python -m worker`,
`EnvironmentFile=<root>/.env`) or `launchd` on macOS.

## Persistence and restart behaviour

Task history lives in SQLite (`tasks` table). On startup the worker marks tasks
that are still `PENDING`/`RUNNING` as `FAILED` with `error.code =
WORKER_RESTARTED` (cancel-requested ones become `CANCELLED`), so an interrupted
run is never reported as `SUCCEEDED`. Logs are plain JSONL files inside each
experiment directory and remain readable after restart.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q          # 90 tests, no torch / GPU required
```

Coverage: auth on every endpoint, health, GPU payload, generic job success and
rejection rules, log pagination and marker hiding, result/artifacts/download
sha256, cancel (running + pending + process tree), idempotency, restart
recovery, path traversal, invalid requests, task timeout, and forced-CUDA
failure semantics. Tests use FastAPI `TestClient` and a temporary work root;
they never bind a port.

## Known limitations

- A hard worker crash cannot resume a running training job; it is reported as
  `FAILED`/`WORKER_RESTARTED` (the contract forbids faking success).
- Training/inference require PyTorch on the worker. On a CPU-only machine the
  `device: "cuda"` path is still verified to fail with `CUDA_UNAVAILABLE`.
- `generic_python` is deliberately a code-execution endpoint; its security model
  is "trusted scripts in one directory", not sandboxing.
- SQLite + a single process: fine for one worker; do not run two workers against
  the same database file.
- No built-in TLS or user management. For anything beyond localhost use a tunnel
  or a TLS-terminating reverse proxy and a strong token.
