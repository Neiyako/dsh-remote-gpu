# Smoke evidence — dsh-gpu-worker

Real-run evidence captured on 2026-11-07 (macOS arm64, Python 3.13.13).
All commands below were executed against a real `python -m worker` process on
`127.0.0.1:8765`; the server was stopped afterwards.

## 1. Test suite

```bash
cd dsh-gpu-worker
python3 -m pytest -q
# 90 passed in 34.38s

# same suite inside the torch-enabled venv (.venv-e2e, used only for a smoke run):
/Users/apple/Desktop/provider/.venv-e2e/bin/python -m pytest -q
# 89 passed, 1 skipped (skipped: torch is installed; TORCH_UNAVAILABLE path n/a)
```

Tests never bind a port and never need torch or a GPU.

## 2. curl smoke (generic_python)

| Step | Request | Result |
|---|---|---|
| health | `GET /health` | `{"status":"ok","version":"0.1.0","cuda_available":false,...}` |
| gpu | `GET /gpu` | `{"cuda_available":false,"device_count":0,"devices":[],"error":"CUDA unavailable: ..."}` |
| auth | `GET /health` without token | `401 {"error":{"code":"UNAUTHORIZED",...}}` |
| submit | `POST /tasks` generic_python `jobs/hello.py` | `201` + `TaskView` `PENDING` (returned immediately) |
| idempotency | same payload + same key | `200`, same `task_id` |
| idempotency conflict | same key, different payload | `409 IDEMPOTENCY_CONFLICT` |
| logs | `GET /tasks/<id>/logs?offset=0&limit=20` | JSONL lines incl. `hello from scripts/jobs/hello.py`, `"token_visible": false` |
| result | `GET /tasks/<id>/result` | `SUCCEEDED`, `metrics.exit_code = 0`, `experiment_dir = experiments/<id>` |
| artifacts | `GET /tasks/<id>/artifacts` | `hello.json`, `hello.txt`, `job.json`, `metrics.json`, `result.json`, `worker.log` |
| download | `GET /tasks/<id>/artifacts/hello.txt` | `200`, `Content-Disposition: attachment` |
| sha256 | compare artifact sha256 with downloaded bytes | `expected == actual`, `sha256_match=True` |
| path escape | `GET .../artifacts/%2e%2e%2f%2e%2e%2fetc%2fpasswd` | `400 {"error":{"code":"PATH_ESCAPE",...}}` |
| CUDA forced | `POST /tasks` training `device:"cuda"` (no CUDA) | `FAILED` + `{"code":"CUDA_UNAVAILABLE"}` — no CPU fallback |
| cancel | long generic job → `POST /tasks/<id>/cancel` | `cancel_requested=true` → final `CANCELLED` |
| history | `GET /tasks?limit=10` | `SUCCEEDED`/`FAILED`/`CANCELLED` tasks listed |

Selected raw output:

```json
GET /health
{"status": "ok", "worker_id": "worker-7e6cec5a6333", "version": "0.1.0",
 "time": "2026-11-07T14:15:22.916489Z", "uptime_seconds": 0.149,
 "cuda_available": false, "active_tasks": 0, "queued_tasks": 0,
 "max_concurrent_tasks": 1, "allow_generic_python": true,
 "torch_installed": false, "recovered_tasks": 0}

GET /gpu
{"cuda_available": false, "device_count": 0, "devices": [],
 "nvidia_smi_available": false,
 "error": "CUDA unavailable: PyTorch is not installed in this environment and no nvidia-smi device was detected.",
 "torch_installed": false, "torch_version": null,
 "time": "2026-11-07T14:15:22.979409Z"}

forced cuda training task
cuda_task_status=FAILED
error= {"code": "CUDA_UNAVAILABLE",
        "message": "device='cuda' was requested but CUDA is unavailable on this worker (...); refusing to fall back to CPU.",
        "traceback": "..."}

cancel
long_task=3afc3b6f5a57449b815e12196b4b31ae status_before_cancel=RUNNING
cancel_requested= True status= RUNNING
cancel_final_status=CANCELLED
```

## 3. Real CPU training + inference smoke (torch venv)

`python -m worker` was started with `/Users/apple/Desktop/provider/.venv-e2e/bin/python`
(torch 2.15.0, CPU only) and driven with curl. The venv was only used to run the
smoke; it is not a project dependency and was not modified.

Training request: `mlp [32,16]`, synthetic 240 samples / 8 features / 3 classes,
5 epochs, batch 32, lr 0.01, CPU, checkpoint + early stopping + validation.

```
status=RUNNING ...
status=SUCCEEDED epoch 5 / 5 train_loss 0.001966 val_loss 0.000156 acc 1.0
final_training_status=SUCCEEDED

persisted progress:
{"epoch": 5, "total_epochs": 5, "train_loss": 0.001966, "val_loss": 0.000156,
 "accuracy": 1.0, "learning_rate": 0.01, "gpu_utilization_percent": null,
 "gpu_memory_used_mb": null, "elapsed_seconds": 3.134, "updated_at": "..."}

training metrics (result):
{"task_type": "training", "status": "succeeded", "device": "cpu",
 "epochs_requested": 5, "epochs_completed": 5, "early_stopped": false,
 "stop_reason": "completed",
 "best": {"epoch": 5, "metric": "val_loss", "value": 0.000156, ...},
 "final": {"epoch": 5, "train_loss": 0.001966, "val_loss": 0.000156,
           "accuracy": 1.0, "learning_rate": 0.01, ...}}

training artifacts:
config.json
job.json
metrics.json
model.pt
best_model.pt
report.md
result.json
train.log
worker.log

train.log head:
[training] task=... device=cpu epochs=5 batch_size=32 seed=7 model=mlp loss=cross_entropy optimizer=adam
[training] dataset=synthetic samples=240 features=8 classes=[0, 1, 2]
[training] model parameters=867 output_dim=3
[epoch 1/5] train_loss=0.936067 val_loss=0.574306 accuracy=1.000000 lr=0.01 elapsed=3.086s
[checkpoint] saved best_model.pt at epoch 1 (val_loss=0.574306)
[epoch 2/5] train_loss=0.366101 val_loss=0.131355 accuracy=1.000000 lr=0.01 elapsed=3.102s
[checkpoint] saved best_model.pt at epoch 2 (val_loss=0.131355)
[epoch 3/5] train_loss=0.072072 val_loss=0.015704 accuracy=1.000000 lr=0.01 elapsed=3.114s

worker.log contains @@PROGRESS@@ markers: 0   (markers are parsed, never logged)
```

Inference from `source_task_id` + `best_model.pt`:

```
final_inference_status=SUCCEEDED
metrics.accuracy = 0.311111   # synthetic eval data is regenerated with a
                              # different seed -> different class centers, so
                              # this number is data-dependent, not a failure
inference artifacts: config.json, job.json, metrics.json, predictions.csv,
                     report.md, result.json, worker.log
predictions.csv head:
row_index,prediction,probability,label,correct
0,2,0.999999,0,0
1,1,0.667000,1,1
2,2,0.999999,0,0
3,1,0.673650,1,1
```

Restart (graceful stop + start, same DB):

```
restarted_ok=1
inference_task_after_restart: SUCCEEDED
recovered_tasks_on_restart= 0     # no task was running at shutdown
history of training/inference tasks remains queryable via GET /tasks/<id>
```

## 4. Environment

| Item | Value |
|---|---|
| OS | macOS 27.0.1 arm64 |
| Python (tests/server) | 3.13.13 |
| fastapi / pydantic | 0.115.6 / 2.13.2 (system), 0.162.1 / 2.14.3 (torch venv smoke) |
| torch | 2.15.0 CPU (smoke venv only) |
| CUDA / nvidia-smi | not available (`cuda_available=false`); real-GPU training is `NOT_RUN` on this machine |
