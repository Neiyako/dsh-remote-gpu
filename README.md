# DSH Remote GPU

把一台 Windows NVIDIA 机器变成 DSH 的远程 GPU 执行端：Mac 侧的 DSH 通过
**8 个 `remote_gpu_*` 工具**提交训练/推理任务、读日志与进度、取消任务、取结果、
下载模型权重。Worker 是一个独立的 FastAPI 服务，DSH 插件只是它的 HTTP 客户端。

> 契约唯一来源：[`docs/CONTRACT.md`](docs/CONTRACT.md)（冻结）·
> HTTP API：[`docs/API.md`](docs/API.md) ·
> Windows 部署：[`docs/DEPLOY-WINDOWS.md`](docs/DEPLOY-WINDOWS.md) ·
> 端到端验收证据：[`docs/E2E_TEST_REPORT.md`](docs/E2E_TEST_REPORT.md)

## 架构

```
┌──────────────────────────── Mac (DSH Host) ────────────────────────────┐
│  DSH 会话                                                              │
│    │ 调用工具                                                          │
│    ▼                                                                    │
│  dsh-remote-gpu 插件 (Host-only)                                       │
│    8 个工具: remote_gpu_health / gpu_info / submit / status            │
│             / logs / cancel / result / artifacts                       │
│    │  Node 内置 fetch + Bearer token + 结构化错误                       │
└────┼───────────────────────────────────────────────────────────────────┘
     │  HTTP/JSON  (默认 127.0.0.1:8765；跨机走内网/VPN)
     ▼
┌────────────────── Windows / Linux GPU Worker ──────────────────────────┐
│  FastAPI  (worker/api.py)                                              │
│    ├─ 统一认证 (Authorization: Bearer … | X-Worker-Token)              │
│    ├─ SQLite 任务库 (worker.db, 重启恢复/历史查询)                      │
│    └─ 任务调度线程 (并发上限默认 1, 超时/取消杀进程树)                   │
│           │  subprocess: python -m worker.job_entry <job.json>          │
│           ▼                                                             │
│  执行器: training (PyTorch/CUDA) │ inference │ generic_python (白名单)   │
│           │                                                             │
│           ▼                                                             │
│  <root>/experiments/<task_id>/  model.pt, best_model.pt, metrics.json, │
│                                 config.json, train.log, report.md, …   │
└─────────────────────────────────────────────────────────────────────────┘
```

**工具名规范**：DeepSeek 的 `tools[].name` 只允许 `[A-Za-z0-9_-]`，所以实际工具名是
下划线形式 `remote_gpu_health` 等（不是 `remote_gpu.health`）。

## 目录

| 路径 | 内容 |
|---|---|
| `dsh-gpu-worker/` | 独立 Python Worker（FastAPI + SQLite + 执行器 + Windows 脚本） |
| `dsh-remote-gpu/` | DSH Host 插件（8 个工具、无第三方依赖） |
| `e2e/run_e2e.py` | HTTP 层端到端 harness（stdlib urllib，自起 worker） |
| `e2e/plugin_e2e.mjs` | 插件层端到端 harness（真实 `index.js` + 假 ctx 注册） |
| `deploy.ps1` | Windows 一键部署脚本（下载/venv/cu128/token/防火墙/启动） |
| `docs/` | 契约、API、Windows 部署、E2E 验收报告 |

## 快速开始

### A. 本机（macOS/Linux，CPU 即可跑通用与训练语义）

```bash
cd dsh-gpu-worker
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # fastapi/uvicorn/pydantic
# 训练/推理需要 PyTorch；CUDA 机器按 docs/DEPLOY-WINDOWS.md §2.2 装对应 wheel
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

export DSH_GPU_WORKER_TOKEN="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"
export DSH_GPU_WORKER_ROOT="$PWD/../data"
mkdir -p "$DSH_GPU_WORKER_ROOT/scripts/jobs" "$DSH_GPU_WORKER_ROOT/datasets"
cp scripts/jobs/hello.py "$DSH_GPU_WORKER_ROOT/scripts/jobs/"

.venv/bin/python -m worker            # 默认 127.0.0.1:8765
```

冒烟：

```bash
TOKEN="$DSH_GPU_WORKER_TOKEN"
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/health
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/gpu
```

### B. Windows RTX 5060 一键部署（推荐）

在 Windows 上**以管理员身份打开 PowerShell**（管理员只用于自动加防火墙规则），执行：

```powershell
powershell -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; iex (irm 'https://api.github.com/repos/Neiyako/dsh-remote-gpu/contents/deploy.ps1?ref=main' -Headers @{'User-Agent'='dsh';'Accept'='application/vnd.github.raw'})"
```

脚本会自动：下载仓库 → 建 venv → 安装 FastAPI + **cu128 PyTorch**（RTX 50 系列 Blackwell）
→ 生成 token 并写 `.env`（HOST=0.0.0.0，工作根 `%USERPROFILE%\dsh-remote-gpu\data`）
→ 后台启动 Worker → 加 8765/TCP 防火墙规则 → 打印 Mac 侧要执行的两行连接命令。

常用参数（完整说明见 [`docs/DEPLOY-WINDOWS.md`](docs/DEPLOY-WINDOWS.md)）：

```powershell
# 复用 Mac 上已有的 token（推荐，省去两端改 token）
.\deploy.ps1 -Token '<token>'
# 只装不启动 / CPU-only / 不改防火墙
.\deploy.ps1 -StartWorker:$false
.\deploy.ps1 -InstallTorch:$false
.\deploy.ps1 -OpenFirewall:$false
```

也支持手动部署，按 [`docs/DEPLOY-WINDOWS.md`](docs/DEPLOY-WINDOWS.md) 执行：

1. 装 Python 3.10+ → `scripts/install.ps1`；
2. 装 **CUDA 12.8+ 版 PyTorch**（RTX 50 系列 Blackwell 必须，默认 CPU wheel 或 cu124 可能不匹配）；
3. 复制 `.env.example` → `.env`，设置 `DSH_GPU_WORKER_TOKEN` / `HOST=0.0.0.0` / `ROOT`；
4. `scripts/start.ps1` 启动，用 `/health`、`/gpu` 确认 `cuda_available: true`；
5. 需要 Mac 访问时开防火墙白名单（8765/TCP），并按需配任务计划程序或 NSSM 自启。

### C. Mac 侧接入 DSH 插件

```bash
# 凭证（二选一；插件按 tokenEnv -> tokenFile 顺序解析，不硬编码）
export DSH_GPU_WORKER_TOKEN='<worker token>'
# 或：mkdir -p ~/.dsh-remote-gpu && printf '%s\n' '<token>' > ~/.dsh-remote-gpu/token && chmod 600 ~/.dsh-remote-gpu/token

# 跨机时可用环境变量覆盖插件 baseUrl（默认 tokenEnv=DSH_GPU_WORKER_TOKEN、baseUrlEnv=DSH_GPU_WORKER_URL）
export DSH_GPU_WORKER_URL='http://<windows-ip>:8765'
```

`dsh-remote-gpu/cordis.patch.yml` 默认 `baseUrl: http://127.0.0.1:8765`；
跨机改成 Windows 内网 IP：`http://192.168.1.20:8765`。由 DSH owner 用 `plugin_manager`
安装/启用该 bundle（本仓库不自动改 profile）。启用后工具列表出现 8 个 `remote_gpu_*`。

最小会话用法（由 agent 调用工具，下例为工具语义）：

1. `remote_gpu_health` / `remote_gpu_gpu_info` 确认连通与 `cuda_available`；
2. `remote_gpu_submit` 提交 `training`（`device:"cuda"`）或 `generic_python`；
3. 轮询 `remote_gpu_status` 看 `progress`（epoch/train_loss/val_loss/…）；
4. `remote_gpu_logs` 读日志，`remote_gpu_cancel` 取消；
5. 终态后 `remote_gpu_result` 取 metrics/artifacts，`remote_gpu_artifacts` 列
   `path/size_bytes/sha256/download_url`；模型权重用 Worker 的
   `GET /tasks/{id}/artifacts/<path>` 下载（插件当前只返回清单）。

## 8 个工具

| 工具（实际注册名） | 底层 HTTP | 用途 |
|---|---|---|
| `remote_gpu_health` | `GET /health` | 状态、版本、`cuda_available`、活跃/排队任务 |
| `remote_gpu_gpu_info` | `GET /gpu` | CUDA 可用性、设备列表、显存/利用率/温度/driver |
| `remote_gpu_submit` | `POST /tasks` | 提交 training / inference / generic_python，立即返回 TaskView |
| `remote_gpu_status` | `GET /tasks/{id}` | status/progress/error |
| `remote_gpu_logs` | `GET /tasks/{id}/logs` | 分页日志（offset/limit） |
| `remote_gpu_cancel` | `POST /tasks/{id}/cancel` | 取消 PENDING/RUNNING 任务 |
| `remote_gpu_result` | `GET /tasks/{id}/result` | 终态 metrics/artifacts/experiment_dir |
| `remote_gpu_artifacts` | `GET /tasks/{id}/artifacts` | 产物清单（含 sha256/download_url） |

统一返回 `{ok:true,result}` 或 `{ok:false,error:{code,message,details?}}`；
网络失败/超时/非 2xx/坏 JSON/缺凭证都是 `ok:false`，**不会把失败报成成功**。

## 安全模型

- **所有** Worker 接口（含 `/health`、`/gpu`）都要 token；token 从
  `DSH_GPU_WORKER_TOKEN` 读取，缺失/过短拒绝启动；插件侧只从 `tokenEnv`/`tokenFile`
  读取，不硬编码、不回显。
- Worker 默认只监听 `127.0.0.1`；跨机需显式 `0.0.0.0` + 防火墙白名单/内网/VPN。
- `generic_python` 不是任意代码执行：只允许 `<root>/scripts/jobs/*.py` 白名单脚本，
  argv 直传、不经过 shell；`env` 不能覆盖 `PATH`/`PYTHONPATH`；子进程看不到 worker token。
  不需要时可用 `DSH_GPU_WORKER_ALLOW_GENERIC_PYTHON=false` 关闭。
- 固定工作根 + 路径 resolve 校验：实验目录、产物、脚本路径逃逸一律 `400 PATH_ESCAPE`。
- `device:"cuda"` 且 CUDA 不可用 → 任务 `FAILED` + `CUDA_UNAVAILABLE`，**绝不静默回退 CPU**。
- 重启恢复：启动时把未完成的 `PENDING/RUNNING` 任务标记为 `FAILED`
  `WORKER_RESTARTED`（取消中的标记 `CANCELLED`），禁止误报 `SUCCEEDED`。

## 测试

```bash
# Worker 单元/接口测试（system python 或 .venv）
cd dsh-gpu-worker && python3 -m pytest -q          # 90 passed
cd dsh-gpu-worker && ../.venv-e2e/bin/python -m pytest -q   # 89 passed, 1 skipped（torch 已装时跳过 torch-unavailable 用例）

# 插件单元测试（假 HTTP server，不联网）
cd dsh-remote-gpu && node --check index.js && node --test    # 20 cases

# HTTP 层端到端（自起 worker 于 127.0.0.1:8766，临时 root/token，15 项检查）
python3 e2e/run_e2e.py

# 插件层端到端（导入真实 index.js，假 ctx 注册，8 个工具打真 worker + 不可达端口）
node e2e/plugin_e2e.mjs
```

两个 E2E harness 均退出码 0 / 非 0 表示通过/失败，会自动清理 worker 进程和临时目录
（`--keep-temp` 保留现场）。E2E 只用 8766/动态空闲端口，不碰 8765。

## 已知限制

- **真实 Windows RTX 5060 跨机 CUDA 训练未在本机验证**（无 Windows/CUDA 机器）：
  验收项状态 `NOT_RUN`，需要真实硬件按 [`docs/DEPLOY-WINDOWS.md`](docs/DEPLOY-WINDOWS.md)
  与 [`docs/E2E_TEST_REPORT.md`](docs/E2E_TEST_REPORT.md) 复核。
- 插件无轮询/编排/重试；进度与日志靠调用方轮询。
- 插件只返回产物清单，不提供二进制下载工具（用 Worker `download_url`/curl）。
- Worker 单进程单 SQLite，不支持多 worker 共享队列；无鉴权重放保护/限流，
  跨机暴露应放在 VPN/反向代理后。
- `generic_python` 仅白名单脚本目录，不能当通用代码执行服务。
- 产物下载不支持断点续传。

## 外部参照

- FastAPI 官方文档（应用生命周期、`FileResponse`、异常处理）；
- PyTorch 官方安装矩阵（CUDA 12.8 wheel 与 Blackwell/sm_120 兼容性）；
- Node.js 内置 `fetch` + `AbortSignal.timeout` 文档（无依赖 HTTP 客户端与超时语义）。
