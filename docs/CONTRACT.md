# DSH Remote GPU Worker — 实现契约 v1

> 本文件是 Lead 与各实现者之间的唯一接口契约。实现可以加字段，但不能改已有字段语义。
> 项目根：`/Users/apple/Desktop/provider`

## 1. 目录与写作用域

| 路径 | Owner | 说明 |
|---|---|---|
| `dsh-gpu-worker/**` | worker-dev | 独立 Python Worker 项目 |
| `dsh-remote-gpu/**` | plugin-dev | DSH Host 插件（8 个工具） |
| `docs/**`, `e2e/**`, 根 `README.md` | verifier | 文档与端到端脚本 |
| `docs/CONTRACT.md` | lead | 冻结契约，其他人只读 |

禁止改 DSH Core、`~/.dsh` 下的 profile 配置；插件由 Lead 用 `plugin_manager` 安装。

## 2. Worker HTTP API

- Base：`http://<host>:8765`
- JSON UTF-8；除下载外全部 `Content-Type: application/json`
- 认证：`Authorization: Bearer <token>` 或 `X-Worker-Token: <token>`。**所有** API（含 `/health`、`/gpu`）都要认证。
- 错误体统一：
  ```json
  {"error":{"code":"TASK_NOT_FOUND","message":"...","details":{}}}
  ```
- 时间统一 ISO-8601 带时区（UTC，`Z`）。
- 任务状态枚举：`PENDING | RUNNING | SUCCEEDED | FAILED | CANCELLED`。

### 2.1 `GET /health`
```json
{"status":"ok","worker_id":"...","version":"0.1.0","time":"...","uptime_seconds":12.3,
 "cuda_available":false,"active_tasks":0,"queued_tasks":0,"max_concurrent_tasks":1}
```
即使 CUDA 不可用，`status` 仍为 `ok`，`cuda_available` 如实返回。

### 2.2 `GET /gpu`
```json
{"cuda_available":true,"device_count":1,
 "devices":[{"index":0,"name":"NVIDIA GeForce RTX 5060 Laptop GPU","capability":"12.0",
   "total_memory_mb":8151,"free_memory_mb":7000,"used_memory_mb":1151,
   "utilization_percent":0,"temperature_c":45,"torch_version":"2.x","driver_version":"..."}],
 "nvidia_smi_available":true,"error":null}
```
CUDA 不可用时 `cuda_available:false`、`device_count:0`、`devices:[]`，`error` 给人读原因。

### 2.3 `POST /tasks`
请求：
```json
{"task_type":"training|inference|generic_python","config":{},
 "name":"optional","timeout_seconds":3600,"idempotency_key":"optional"}
```
- 201 新建；相同 `idempotency_key` 且 payload 相同返回 200 + 原 Task；payload 不同返回 409 `IDEMPOTENCY_CONFLICT`。
- `timeout_seconds` 范围 10..86400，默认 3600。
- 返回 TaskView（见 2.4）。**提交后立即返回，不阻塞等待训练。**

### 2.4 `GET /tasks/{task_id}` → TaskView
```json
{"task_id":"...","task_type":"training","status":"RUNNING","name":"...",
 "created_at":"...","started_at":"...","finished_at":null,"timeout_seconds":3600,
 "progress":{"epoch":3,"total_epochs":10,"train_loss":0.42,"val_loss":0.51,
   "accuracy":0.88,"learning_rate":0.001,"gpu_utilization_percent":91,
   "gpu_memory_used_mb":1024,"elapsed_seconds":12.5,"updated_at":"..."},
 "error":null,"result":null,"cancel_requested":false,"idempotency_key":null}
```
- 终态任务保留 `progress` 最后值；`error` 为 `{code,message,traceback?}`。
- `result` 仅在终态出现，结构见 2.7。

### 2.5 `POST /tasks/{task_id}/cancel`
- PENDING/RUNNING → 请求取消，尽量杀子进程；返回 TaskView。终态返回 409 `TASK_ALREADY_FINISHED`。
- 取消成功最终状态必须是 `CANCELLED`，不能是 `FAILED`。

### 2.6 `GET /tasks/{task_id}/logs?offset=0&limit=500`
```json
{"task_id":"...","offset":0,"next_offset":120,"lines":[{"seq":1,"ts":"...","stream":"stdout","text":"..."}],"eof":false}
```
- 日志持久化到实验目录，Worker 重启后可读。
- `offset` 为行偏移；`limit` 1..2000。
- 返回的文本不得包含内部 `@@PROGRESS@@` / `@@RESULT@@` 标记。

### 2.7 `GET /tasks/{task_id}/result`
- 未终态：409 `TASK_NOT_FINISHED`，`details.status` 给当前状态。
- 终态：
```json
{"task_id":"...","status":"SUCCEEDED","result":{
  "metrics":{},"artifacts":[{"path":"metrics.json","size_bytes":123,"sha256":"..."}],
  "experiment_dir":"experiments/<task_id>","config":{},"summary":"..."},
 "error":null}
```
FAILED/CANCELLED 也返回 200，`status` 如实，`error` 非空。

### 2.8 `GET /tasks/{task_id}/artifacts`
```json
{"task_id":"...","artifacts":[{"path":"model.pt","size_bytes":123,"sha256":"...",
  "download_url":"/tasks/<id>/artifacts/model.pt"}]}
```
只列实验目录内普通文件；路径不得逃逸。

### 2.9 `GET /tasks/{task_id}/artifacts/{path:path}`
- 二进制下载，`Content-Disposition: attachment`。
- 路径逃逸 → 400 `PATH_ESCAPE`；不存在 → 404。

### 2.10 建议附加
- `GET /tasks?status=&limit=` 列表（用于重启后查询历史）。
- `GET /datasets` 只列 `datasets/` 下数据集元数据。
- `GET /version`。

## 3. Executor 配置

### 3.1 training
```json
{"model":{"type":"mlp","input_dim":8,"hidden_dims":[64,32],"num_classes":3,"dropout":0.0},
 "dataset":{"dataset_id":"synthetic","format":"synthetic","num_samples":600,
   "num_features":8,"num_classes":3,"noise":0.1,"val_ratio":0.2,"seed":42},
 "epochs":10,"batch_size":32,"learning_rate":0.001,
 "optimizer":{"name":"adam","weight_decay":0.0,"momentum":0.9},
 "loss":{"name":"cross_entropy"},
 "device":"cuda|cpu|auto",
 "checkpoint":{"enabled":true,"every_n_epochs":1,"save_best":true,"resume_from":null},
 "early_stopping":{"enabled":true,"patience":3,"min_delta":0.0,"monitor":"val_loss"},
 "validation":{"enabled":true,"batch_size":64},
 "seed":42}
```
- 数据集支持 `synthetic`、`csv`（`datasets/<dataset_id>/<file>`，`target_column`、`feature_columns`）、`npy`（`X.npy`/`y.npy`）。
- 模型至少支持 `mlp`、`linear`。
- `device` 缺省 `auto`：有 CUDA 用 CUDA，否则 CPU。
- `device:"cuda"` 且 CUDA 不可用 → 任务 FAILED，`error.code="CUDA_UNAVAILABLE"`，**不得静默 CPU**。
- 每 epoch 输出进度（stdout 内部标记，由父进程落库）：
  `{"epoch":1,"total_epochs":10,"train_loss":...,"val_loss":...,"accuracy":...,
    "learning_rate":...,"gpu_utilization_percent":...,"gpu_memory_used_mb":...,"elapsed_seconds":...}`
- 产物：`model.pt`（last）、`best_model.pt`（若 save_best）、`metrics.json`、`config.json`、`train.log`、`report.md`。
- 必须保存可复现信息：完整配置、seed、torch/cuda 版本、设备名、Python 版本、开始/结束时间。

### 3.2 inference
```json
{"source_task_id":"<training task id>","checkpoint":"best_model.pt",
 "dataset":{"dataset_id":"synthetic","format":"synthetic","num_samples":100,"num_features":8,"num_classes":3},
 "batch_size":64,"device":"auto"}
```
- 产物：`metrics.json`、`predictions.csv`、`report.md`。

### 3.3 generic_python
```json
{"script":"jobs/hello.py","args":["--x","1"],"env":{"FOO":"bar"}}
```
- `script` 必须解析到 `scripts/jobs/` 内、`.py` 结尾；只允许白名单目录，不经过 shell，不拼接字符串命令。
- `args` 必须是字符串数组；`env` 只允许 `^[A-Z_][A-Z0-9_]*$` 且不得覆盖 `PATH`、`PYTHONPATH`。
- 默认禁用任意代码；配置 `allow_generic_python` 默认 `true` 但仅限白名单脚本目录（本机演示需要）。文档必须写清楚风险。

## 4. SQLite 持久化

表 `tasks` 至少包含：
`task_id TEXT PK, task_type TEXT, status TEXT, name TEXT, idempotency_key TEXT UNIQUE,
payload_json TEXT, config_json TEXT, progress_json TEXT, result_json TEXT, error_json TEXT,
created_at TEXT, started_at TEXT, finished_at TEXT, timeout_seconds INTEGER,
cancel_requested INTEGER, worker_pid INTEGER, log_path TEXT, updated_at TEXT`

- 启动时恢复：`PENDING/RUNNING` 且非取消中的任务 → `FAILED`，`error.code="WORKER_RESTARTED"`（禁止误报 SUCCEEDED）；`cancel_requested=1` → `CANCELLED`。
- 历史任务重启后可查。

## 5. 安全底线

- Token 从环境变量 `DSH_GPU_WORKER_TOKEN` 读取；缺失则拒绝启动。
- 默认监听 `127.0.0.1`，允许显式 `DSH_GPU_WORKER_HOST=0.0.0.0`；文档提示 LAN 防火墙。
- 固定工作根 `DSH_GPU_WORKER_ROOT`（默认项目根）；所有路径 `resolve` 后必须仍在根内。
- 无任意 Shell 接口；子进程只用 `sys.executable -m worker.job_entry`。
- 超时、并发上限（默认 1）、实验目录隔离。

## 6. DSH 插件工具

包名 `dsh-remote-gpu`，Host 插件，`ctx.tools.register`。工具名严格为：
`remote_gpu_health`、`remote_gpu_gpu_info`、`remote_gpu_submit`、`remote_gpu_status`、
`remote_gpu_logs`、`remote_gpu_cancel`、`remote_gpu_result`、`remote_gpu_artifacts`。
工具名只能含 `[A-Za-z0-9_-]`（DeepSeek API 对 `tools[].name` 的校验），不得用点号。

配置：
```yaml
- id: remote-gpu
  name: dsh-remote-gpu
  config:
    baseUrl: http://127.0.0.1:8765
    tokenEnv: DSH_GPU_WORKER_TOKEN
    tokenFile: ~/.dsh-remote-gpu/token
    timeoutMs: 30000
```
- 凭证优先级：`tokenEnv` → `tokenFile`；不得硬编码。
- 网络错误必须返回 `{ok:false,error:{code:"WORKER_UNREACHABLE"|"WORKER_TIMEOUT",...}}`，**不得把网络失败报告为任务成功**。
- `submit` 必须支持 `idempotency_key`，透传 `timeout_seconds`。
- 输出统一 `{ok, result?, error?}`，并带 text render。

## 7. 验收口径

必须实际跑并记录证据：
1. `remote_gpu_health` / `gpu_info`（本机 Worker 可 PASS；真实 Windows CUDA 需另测）
2. 提交 generic_python 任务成功并拿到日志
3. `device:"cuda"` 任务明确 FAILED + `CUDA_UNAVAILABLE`（本机无 CUDA）
4. 提交长任务后 cancel → CANCELLED
5. result / artifacts / 下载文件校验 sha256
6. Worker 重启后历史任务仍可查询
7. 断网/错端口 → 工具返回 `WORKER_UNREACHABLE`，不报成功
8. 真实 Windows RTX 5060 跨机 CUDA 训练：有机器则 PASS，无机器则 `NOT_RUN`（禁止编造）

测试报告用 `PASS / FAIL / NOT_RUN / INSUFFICIENT_EVIDENCE`，每条附命令、退出码或响应片段。
