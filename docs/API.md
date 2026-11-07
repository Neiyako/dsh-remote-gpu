# DSH Remote GPU Worker — HTTP API

> 冻结契约见 [`CONTRACT.md`](./CONTRACT.md)。本文是**面向调用方**的完整 API 文档，
> 由实现（`dsh-gpu-worker/worker/api.py`、`models.py`、`runner.py`）与
> 端到端测试（`e2e/run_e2e.py`）核对，Worker 版本 `0.1.0`。
>
> DSH 侧不直接调 HTTP，而是使用插件 `dsh-remote-gpu` 的 8 个 `remote_gpu_*` 工具
> （见根 [`README.md`](../README.md) 与 [`../dsh-remote-gpu/README.md`](../dsh-remote-gpu/README.md)）。
> 本文档描述工具底层对应的 HTTP 接口，便于用 curl / 其它语言直接调用或排障。

## 1. 基础约定

| 项 | 约定 |
|---|---|
| Base URL | `http://<host>:<port>`，默认端口 `8765` |
| 认证 | 每个请求都要 `Authorization: Bearer <token>` 或 `X-Worker-Token: <token>`；**包括 `/health`、`/gpu`** |
| Token 来源 | Worker 进程环境变量 `DSH_GPU_WORKER_TOKEN`，缺失/过短时拒绝启动 |
| 编码 | JSON UTF-8；除产物下载外响应均为 `application/json` |
| 时间 | ISO-8601、UTC、带 `Z` |
| 任务状态 | `PENDING` / `RUNNING` / `SUCCEEDED` / `FAILED` / `CANCELLED` |
| 并发 | 默认 `max_concurrent_tasks=1`，超出排队为 `PENDING` |
| 超时 | `timeout_seconds` 10..86400，默认 3600；超时任务被终止并标记 `FAILED` + `TASK_TIMEOUT` |

### 1.1 统一错误体

```json
{
  "error": {
    "code": "TASK_NOT_FOUND",
    "message": "Task 'abc' does not exist",
    "details": {"task_id": "abc"}
  }
}
```

HTTP 状态码与 `code` 对应关系见 §7。

### 1.2 认证示例

```bash
TOKEN='<long-random-token>'
BASE='http://127.0.0.1:8765'

curl -s -H "Authorization: Bearer $TOKEN" "$BASE/health"
# 等价：curl -s -H "X-Worker-Token: $TOKEN" "$BASE/health"
```

未带/错 token 一律 `401`：

```json
{"error":{"code":"UNAUTHORIZED","message":"Missing or invalid worker token","details":{}}}
```

## 2. 健康与设备

### 2.1 `GET /health`

```bash
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/health"
```

```json
{
  "status": "ok",
  "worker_id": "5b0e...",
  "version": "0.1.0",
  "time": "2026-11-07T14:05:01.123456Z",
  "uptime_seconds": 12.3,
  "cuda_available": false,
  "active_tasks": 0,
  "queued_tasks": 0,
  "max_concurrent_tasks": 1,
  "allow_generic_python": true,
  "torch_installed": true,
  "recovered_tasks": 0
}
```

- 即使 CUDA 不可用，`status` 仍为 `ok`；`cuda_available` 如实返回。
- `recovered_tasks`：本次进程启动时按“重启恢复”规则处理的未完成任务数。
- `torch_installed` 表示 Worker 解释器能否 import torch（训练/推理任务需要）。

### 2.2 `GET /gpu`

CUDA 可用时：

```json
{
  "cuda_available": true,
  "device_count": 1,
  "devices": [{
    "index": 0,
    "name": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "capability": "12.0",
    "total_memory_mb": 8151,
    "free_memory_mb": 7000,
    "used_memory_mb": 1151,
    "utilization_percent": 0,
    "temperature_c": 45,
    "torch_version": "2.x",
    "driver_version": "580.xx"
  }],
  "nvidia_smi_available": true,
  "error": null,
  "torch_installed": true,
  "time": "..."
}
```

CUDA 不可用时：

```json
{
  "cuda_available": false,
  "device_count": 0,
  "devices": [],
  "nvidia_smi_available": true,
  "error": "CUDA unavailable: torch.cuda.is_available() is false and nvidia-smi found no device.",
  "torch_installed": true,
  "time": "..."
}
```

`error` 是给人读的原因字符串（不是错误体）。

### 2.3 `GET /version`

```json
{"name":"dsh-gpu-worker","version":"0.1.0","api_version":"1","worker_id":"...",
 "time":"...","root":"/abs/work/root","max_concurrent_tasks":1,"allow_generic_python":true}
```

### 2.4 `GET /datasets`

只列出固定工作根下 `datasets/` 的目录元数据，不读文件内容：

```json
{"datasets":[{"dataset_id":"iris","path":"datasets/iris","num_files":1,
  "formats":["csv"],"files":[{"name":"data.csv","size_bytes":455}]}],"count":1}
```

## 3. 任务：提交与查询

### 3.1 `POST /tasks`

请求体：

```json
{
  "task_type": "training",
  "config": {},
  "name": "optional human label",
  "timeout_seconds": 3600,
  "idempotency_key": "optional"
}
```

- `task_type`：`training` | `inference` | `generic_python`。
- `config`：按 §4 对应任务类型的结构；未知字段被接受但执行器会忽略/校验。
- `timeout_seconds`：10..86400，越界 → `400 INVALID_REQUEST`。
- `idempotency_key`：可选，最长 256 字符。
  - 相同 key + **相同 payload**（`task_type/config/name/timeout_seconds` 规范化后）→ `200` + 原 TaskView，**不再新建任务**；
  - 相同 key + payload 不同 → `409 IDEMPOTENCY_CONFLICT`，`details.task_id` 指向原任务。
- **提交立即返回**（`201`），不等待训练完成。

```bash
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"task_type":"generic_python",
       "config":{"script":"jobs/hello.py","args":["--x","1"]},
       "name":"smoke","timeout_seconds":300}' \
  "$BASE/tasks"
```

`201` 响应 = TaskView（§3.3）。

### 3.2 `GET /tasks?status=&limit=`

分页列出任务，用于重启后查历史。

- `status`：可选，取值 `PENDING|RUNNING|SUCCEEDED|FAILED|CANCELLED`，其它值 → `400 INVALID_REQUEST`。
- `limit`：1..500，默认 50。

```json
{"tasks":[TaskView, ...],"count":5}
```

### 3.3 `GET /tasks/{task_id}` → TaskView

```json
{
  "task_id": "24cf875254e5495ab7abc70c92c99482",
  "task_type": "generic_python",
  "status": "RUNNING",
  "name": "smoke",
  "created_at": "2026-11-07T14:05:01Z",
  "started_at": "2026-11-07T14:05:01Z",
  "finished_at": null,
  "timeout_seconds": 300,
  "progress": {
    "epoch": 3, "total_epochs": 10, "train_loss": 0.42, "val_loss": 0.51,
    "accuracy": 0.88, "learning_rate": 0.001,
    "gpu_utilization_percent": 91, "gpu_memory_used_mb": 1024,
    "elapsed_seconds": 12.5, "updated_at": "..."
  },
  "error": null,
  "result": null,
  "cancel_requested": false,
  "idempotency_key": null
}
```

- 终态保留 `progress` 最后值。
- `error` 非空时结构为 `{"code","message","traceback?"}`。
- `result` 仅终态出现（结构与 `/result` 相同）。
- 不存在 → `404 TASK_NOT_FOUND`。

### 3.4 `POST /tasks/{task_id}/cancel`

- `PENDING`/`RUNNING` → 请求取消（杀子树），返回 TaskView；终态最终必须为 `CANCELLED`。
- 已完成任务 → `409 TASK_ALREADY_FINISHED`，`details.status` 给终态。
- `PENDING` 直接被取消（不会启动）。

```bash
curl -s -X POST -H "Authorization: Bearer $TOKEN" "$BASE/tasks/$TASK_ID/cancel"
```

### 3.5 `GET /tasks/{task_id}/logs?offset=0&limit=500`

```json
{
  "task_id": "24cf...",
  "offset": 0,
  "next_offset": 5,
  "lines": [
    {"seq": 1, "ts": "2026-11-07T14:05:02Z", "stream": "stdout", "text": "hello"},
    {"seq": 2, "ts": "2026-11-07T14:05:02Z", "stream": "stderr", "text": "warn"}
  ],
  "eof": false
}
```

- `offset` 为行偏移，`limit` 1..2000（默认 500）。
- 日志持久化在实验目录（`experiments/<task_id>/worker.log`），**Worker 重启后仍可读**。
- 内部进度/结果标记（`@@PROGRESS@@` / `@@RESULT@@`）不会出现在 `text` 中。
- 任务存在但还没有日志 → `lines: []`、`eof: true`。

### 3.6 `GET /tasks/{task_id}/result`

- 未终态 → `409 TASK_NOT_FINISHED`，`details.status` 给当前状态。
- 终态一律 `200`，`status` 如实：

```json
{
  "task_id": "...",
  "status": "SUCCEEDED",
  "result": {
    "metrics": {"exit_code": 0, "duration_seconds": 1.23, "script": "jobs/hello.py"},
    "artifacts": [{"path": "metrics.json", "size_bytes": 123, "sha256": "..."}],
    "experiment_dir": "experiments/<task_id>",
    "config": {},
    "summary": "generic_python jobs/hello.py finished with exit code 0 in 1.23s"
  },
  "error": null
}
```

`FAILED`/`CANCELLED` 时 `result` 为 `null`，`error` 非空，例如：

```json
{"task_id":"...","status":"FAILED","result":null,
 "error":{"code":"CUDA_UNAVAILABLE","message":"device='cuda' was requested but CUDA is unavailable ..."}}
```

### 3.7 `GET /tasks/{task_id}/artifacts`

```json
{
  "task_id": "...",
  "artifacts": [
    {"path": "metrics.json", "size_bytes": 123,
     "sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
     "download_url": "/tasks/<task_id>/artifacts/metrics.json"}
  ]
}
```

- 只列实验目录内的普通文件；符号链接、目录、点文件不列出。
- 按需计算 sha256（带缓存）。

### 3.8 `GET /tasks/{task_id}/artifacts/{path}` — 二进制下载

```bash
curl -s -H "Authorization: Bearer $TOKEN" -OJ \
  "$BASE/tasks/$TASK_ID/artifacts/model.pt"
```

- 响应 `Content-Type: application/octet-stream`，`Content-Disposition: attachment; filename="..."`。
- 路径逃逸（`..`、绝对路径、Windows 盘符）→ `400 PATH_ESCAPE`。
- 不存在 → `404 ARTIFACT_NOT_FOUND`。

## 4. 任务类型配置

### 4.1 `training`

```json
{
  "model": {"type": "mlp", "input_dim": 8, "hidden_dims": [64, 32], "num_classes": 3, "dropout": 0.0},
  "dataset": {
    "dataset_id": "synthetic", "format": "synthetic",
    "num_samples": 600, "num_features": 8, "num_classes": 3,
    "noise": 0.1, "val_ratio": 0.2, "seed": 42,
    "file": "data.csv", "target_column": "target", "feature_columns": null
  },
  "epochs": 10, "batch_size": 32, "learning_rate": 0.001,
  "optimizer": {"name": "adam", "weight_decay": 0.0, "momentum": 0.9},
  "loss": {"name": "cross_entropy"},
  "device": "auto",
  "checkpoint": {"enabled": true, "every_n_epochs": 1, "save_best": true, "resume_from": null},
  "early_stopping": {"enabled": false, "patience": 3, "min_delta": 0.0, "monitor": "val_loss"},
  "validation": {"enabled": true, "batch_size": 64},
  "seed": 42
}
```

| 字段 | 说明 |
|---|---|
| `model.type` | `mlp`（默认）或 `linear` |
| `dataset.format` | `synthetic`（默认，按参数现场生成）、`csv`、`npy` |
| `dataset.dataset_id` | 对应 `datasets/<dataset_id>/`；`csv` 默认文件名 `data.csv`，`npy` 需 `X.npy`/`y.npy` |
| `loss.name` | `cross_entropy`（分类）或 `mse`（回归） |
| `device` | `auto`（默认，有 CUDA 用 CUDA）、`cpu`、`cuda`。**`cuda` 且 CUDA 不可用 → 任务 FAILED `CUDA_UNAVAILABLE`，绝不静默回退 CPU** |
| `optimizer.name` | `adam` / `adamw` / `sgd` |
| `checkpoint.resume_from` | 实验目录内已有 checkpoint 文件名，用于续训 |

进度：每个 epoch 通过内部标记落库，`GET /tasks/{id}` 的 `progress` 可见
`epoch / total_epochs / train_loss / val_loss / accuracy / learning_rate / gpu_* / elapsed_seconds`。

产物（`experiments/<task_id>/`）：`model.pt`（last）、`best_model.pt`（`save_best` 时）、
`metrics.json`、`config.json`、`train.log`、`report.md`。
`metrics.json` / `config.json` 同时保存可复现信息：完整配置、seed、torch/cuda 版本、
设备名、Python 版本、开始/结束时间。

### 4.2 `inference`

```json
{
  "source_task_id": "<training task id>",
  "checkpoint": "best_model.pt",
  "dataset": {"dataset_id": "synthetic", "format": "synthetic",
              "num_samples": 100, "num_features": 8, "num_classes": 3},
  "batch_size": 64,
  "device": "auto",
  "seed": 42
}
```

- `source_task_id` 必须是合法的 training 任务 id；`checkpoint` 不得含路径分隔符。
- 产物：`metrics.json`、`predictions.csv`、`report.md`。
- 相关错误码：`CHECKPOINT_NOT_FOUND`、`INVALID_CHECKPOINT`、`INVALID_DATASET`。

### 4.3 `generic_python`

```json
{"script": "jobs/hello.py", "args": ["--x", "1"], "env": {"FOO": "bar"}}
```

安全规则（实现强约束）：

- `script` 必须解析到 `<root>/scripts/jobs/` 内且以 `.py` 结尾；`jobs/hello.py` 与 `hello.py` 等价；
- 绝对路径、`..`、非 `.py` → 提交阶段即 `400 PATH_ESCAPE`；
- `args` 必须全部是字符串，作为 argv 直接传给子进程，**不经过 shell、不做字符串拼接**；
- `env` 名必须匹配 `^[A-Z_][A-Z0-9_]*$`，且不得覆盖 `PATH`/`PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`；
- 子进程环境不会继承 `DSH_GPU_WORKER_TOKEN`；
- 禁用整个任务类型：`DSH_GPU_WORKER_ALLOW_GENERIC_PYTHON=false`（提交返回 `403 FORBIDDEN`）。

产物：`metrics.json`（exit_code / duration_seconds / script / num_args）、`worker.log` 等。
脚本退出码非 0 → 任务 `FAILED`，`error.code="SCRIPT_FAILED"`。

## 5. Worker 侧环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSH_GPU_WORKER_TOKEN` | （无，必填） | 所有 API 的 Bearer token；缺失或 <8 字符拒绝启动 |
| `DSH_GPU_WORKER_ROOT` | Worker 项目目录 | 固定工作根：`experiments/`、`datasets/`、`scripts/jobs/`、`worker.db` 都在其下；所有路径 resolve 后必须仍在根内 |
| `DSH_GPU_WORKER_HOST` | `127.0.0.1` | 监听地址；跨机需显式 `0.0.0.0` + 防火墙白名单 |
| `DSH_GPU_WORKER_PORT` | `8765` | 监听端口 |
| `DSH_GPU_WORKER_DB` | `<root>/worker.db` | SQLite 路径 |
| `DSH_GPU_WORKER_MAX_CONCURRENT_TASKS` | `1` | 并行任务上限 |
| `DSH_GPU_WORKER_ALLOW_GENERIC_PYTHON` | `true` | 是否允许 `generic_python`（仅白名单脚本目录） |

## 6. 持久化与重启语义

- SQLite 表 `tasks` 持久化任务状态、payload、progress、result、error、日志路径。
- 启动时：`PENDING`/`RUNNING` 且未请求取消的任务 → 置 `FAILED`，`error.code="WORKER_RESTARTED"`（**禁止误报 SUCCEEDED**）；`cancel_requested=1` → `CANCELLED`。
- 日志文件在实验目录中，重启后 `GET /logs` 仍可读；历史任务可继续查询 `/tasks`、`/tasks/{id}`、`/result`、`/artifacts`。

## 7. 错误码总表

### 7.1 HTTP / API 错误

| HTTP | code | 触发 |
|---|---|---|
| 400 | `INVALID_REQUEST` | 请求体/`config` 校验失败、非法 status 过滤、非法 timeout 等 |
| 400 | `PATH_ESCAPE` | 脚本/产物/实验目录路径逃逸 |
| 401 | `UNAUTHORIZED` | 无 token 或 token 不匹配 |
| 403 | `FORBIDDEN` | `generic_python` 被禁用 |
| 404 | `TASK_NOT_FOUND` | 任务不存在 |
| 404 | `ARTIFACT_NOT_FOUND` | 产物文件不存在 |
| 404 | `ROUTE_NOT_FOUND` | 路径不存在 |
| 405 | `METHOD_NOT_ALLOWED` | 方法不支持 |
| 409 | `IDEMPOTENCY_CONFLICT` | 同 key 不同 payload |
| 409 | `TASK_ALREADY_FINISHED` | 取消已终态任务 |
| 409 | `TASK_NOT_FINISHED` | 未终态取 result |
| 500 | `INTERNAL_ERROR` | 未预期异常（有日志） |

### 7.2 任务 `error.code`（任务终态 `FAILED`）

| code | 说明 |
|---|---|
| `CUDA_UNAVAILABLE` | `device:"cuda"` 但 CUDA 不可用（绝不回退 CPU） |
| `TORCH_UNAVAILABLE` | 训练/推理需要 PyTorch，但 Worker 解释器未安装 |
| `TASK_TIMEOUT` | 超过 `timeout_seconds`，进程被终止 |
| `CANCELLED` | 请求取消（终态 `CANCELLED`） |
| `WORKER_RESTARTED` | 进程重启时任务未完成，无法恢复 |
| `WORKER_SHUTDOWN` | 优雅停机时任务仍在运行 |
| `PROCESS_EXIT` | 子进程非 0 退出且无结构化 error |
| `MISSING_RESULT` | 子进程 0 退出但没有 `result.json` |
| `INVALID_CONFIG` / `INVALID_DATASET` / `DATASET_NOT_FOUND` | 训练/推理配置或数据集问题 |
| `INVALID_CHECKPOINT` / `CHECKPOINT_NOT_FOUND` | 推理 checkpoint 问题 |
| `SCRIPT_NOT_FOUND` / `SCRIPT_FAILED` / `SCRIPT_SPAWN_FAILED` | generic_python 脚本问题 |
| `SPAWN_ERROR` | 任务子进程无法启动 |
| `EXECUTOR_UNAVAILABLE` / `UNKNOWN_TASK_TYPE` / `EXECUTOR_ERROR` | 执行器解析/未预期异常 |

## 8. 最小调用序列

```bash
# 1) 健康检查（含认证）
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/health"

# 2) 提交 generic_python 任务
TASK_ID=$(curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"task_type":"generic_python","config":{"script":"jobs/hello.py"}}' \
  "$BASE/tasks" | python3 -c 'import sys,json;print(json.load(sys.stdin)["task_id"])')

# 3) 轮询状态直到终态
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/tasks/$TASK_ID"

# 4) 读日志 / 结果 / 产物
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/tasks/$TASK_ID/logs?limit=200"
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/tasks/$TASK_ID/result"
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/tasks/$TASK_ID/artifacts"
curl -s -H "Authorization: Bearer $TOKEN" -OJ "$BASE/tasks/$TASK_ID/artifacts/metrics.json"
```

## 9. 已知限制

- 无 SSE/WebSocket 推送：进度与日志靠轮询。
- 无内置鉴权重放保护/速率限制；跨机暴露时建议放在反向代理/VPN 后。
- `generic_python` 只允许 `scripts/jobs/` 白名单脚本，不是通用代码执行服务。
- 单 worker 单 SQLite；不支持多 worker 共享队列。
- 产物下载为单文件流式响应，无断点续传。
