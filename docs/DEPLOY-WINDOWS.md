# 在 Windows RTX 5060 上部署 DSH GPU Worker

> 目标：在一台 Windows 机器（示例：RTX 5060 Laptop，Blackwell，compute capability 12.0）上
> 常驻运行 `dsh-gpu-worker`，让 Mac 侧的 DSH 通过 `dsh-remote-gpu` 插件的 8 个
> `remote_gpu_*` 工具提交训练/推理任务。
>
> 本机（macOS，无 CUDA）验证的是 CPU 语义；**真实 RTX 5060 跨机 CUDA 训练属于
> 需要真实 Windows 硬件才能完成的验收项**，没有机器就是 `NOT_RUN`，不要用本机结果冒充。
> HTTP 契约见 [`API.md`](./API.md)，冻结契约见 [`CONTRACT.md`](./CONTRACT.md)。

## 一键部署（推荐）

Windows 上以**管理员身份**打开 PowerShell，执行：

```powershell
powershell -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; iex (irm 'https://api.github.com/repos/Neiyako/dsh-remote-gpu/contents/deploy.ps1?ref=main' -Headers @{'User-Agent'='dsh';'Accept'='application/vnd.github.raw'})"
```

脚本会完成：下载公开仓库 → `%USERPROFILE%\dsh-remote-gpu` → Python venv + FastAPI
→ **cu128 PyTorch**（RTX 50 系列必需）→ 生成 token/写 `.env`（0.0.0.0:8765，
工作根 `%USERPROFILE%\dsh-remote-gpu\data`）→ 后台启动 → 防火墙规则（Private）
→ 打印 Mac 侧连接命令。

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `-Token '<token>'` | 自动生成 | 复用 Mac 上 `~/.dsh-remote-gpu/token` 的内容，省去两端改 token |
| `-InstallDir <path>` | `%USERPROFILE%\dsh-remote-gpu` | 安装目录 |
| `-InstallTorch:$false` | 安装 | 跳过 PyTorch（只跑 generic/健康检查） |
| `-StartWorker:$false` | 启动 | 只安装不启动 |
| `-OpenFirewall:$false` | 开启 | 不改 Windows 防火墙 |
| `-Force` | 关闭 | 更新已存在的安装目录（覆盖代码，保留 `data/`、`logs/`、`.env`） |

> 非管理员运行时脚本仍会安装/启动 Worker，但不会加防火墙规则；此时 Mac 无法跨机连接，
> 需按 §5 手动执行 `New-NetFirewallRule` 或把网络配置为 Private。

## 0. 前置检查清单

| 项 | 要求 | 检查命令（PowerShell） |
|---|---|---|
| Windows | 10 19041+ / 11 / Server 2019+ | `winver` |
| GPU 驱动 | 支持 CUDA 12.8+（RTX 50 系列 Blackwell 需要新驱动） | `nvidia-smi`（右上角 CUDA Version ≥ 12.8） |
| Python | 3.10–3.13，64 位 | `py -3 --version` |
| Git | 任意较新版本（拉代码用） | `git --version` |
| 磁盘 | ≥ 15 GB（CUDA 版 PyTorch 约 3–4 GB） | — |
| 网络 | 首次安装需访问 pypi / download.pytorch.org | — |

> **RTX 5060 / Blackwell 注意**：`compute capability 12.0`（sm_120）只有
> **CUDA 12.8 及以上**的 PyTorch wheel 才带对应 kernel。仓库里
> `scripts/install.ps1 -WithTorch` 默认安装 `cu128`（适配 RTX 50 系列）；30/40 系老卡
> 如需保守可选 `cu124` wheel（见 §2.2 手动命令）。装完后必须按 §2.3 自检，
> 若报 `no kernel image is available` 就用 §2.2 的 cu128 命令重装 torch。

## 1. 获取代码

```powershell
# 建议放短路径，避免 Windows 260 字符路径限制
cd C:\
git clone <repo-url> dsh-gpu-worker
cd C:\dsh-gpu-worker\dsh-gpu-worker
```

目录结构（部署视角）：

```
dsh-gpu-worker/
├─ worker/                 # FastAPI 服务 + 执行器
├─ scripts/
│  ├─ install.ps1          # 建 .venv、装依赖、可选装 torch
│  ├─ start.ps1            # 读 .env 后启动 worker
│  └─ jobs/hello.py        # generic_python 白名单脚本目录
├─ requirements.txt
├─ .env.example
└─ worker.db               # 首次启动自动生成
```

## 2. 安装 Python 环境

### 2.1 建 venv + 运行时依赖

```powershell
cd C:\dsh-gpu-worker\dsh-gpu-worker
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # 可选：跑测试
```

或直接使用仓库脚本（会顺便复制 `.env`）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

### 2.2 安装 CUDA 版 PyTorch（RTX 5060 关键步骤）

* Blackwell / RTX 50 系列（推荐，CUDA 12.8 wheel）：

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

* 若驱动只能到 CUDA 12.6/12.4 且 GPU 是较老的卡（30/40 系），可改用对应 wheel：

```powershell
# 40 系（sm_89）
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

> 不要用默认 `pip install torch`（Windows 上默认是 CPU wheel），否则 Worker 会报
> `cuda_available=false`，且 `device:"cuda"` 任务会按契约直接 `FAILED CUDA_UNAVAILABLE`。

### 2.3 验证 CUDA 可用（必须通过再继续）

```powershell
.\.venv\Scripts\python.exe -c @"
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_version', torch.version.cuda)
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    print('capability', torch.cuda.get_device_capability(0))
    x = torch.randn(8, 8, device='cuda'); print('matmul', (x @ x).sum().item())
"@
```

期望：`cuda_available True`、`device NVIDIA GeForce RTX 5060 ...`、`capability (12, 0)`、
`matmul` 正常输出。若 `is_available()` 为 False 或 matmul 报
`no kernel image is available for execution on the device` → 回到 §2.2 换 cu128 wheel。

同时确认 `nvidia-smi` 能看到 GPU、显存、驱动版本：

```powershell
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

## 3. 配置 token 与工作根

### 3.1 生成 token

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

复制输出，写入 `.env`（不要提交到 git）：

```powershell
Copy-Item .env.example .env
notepad .env
```

`.env` 最小内容：

```ini
DSH_GPU_WORKER_TOKEN=<粘贴上面生成的 token>
DSH_GPU_WORKER_HOST=0.0.0.0
DSH_GPU_WORKER_PORT=8765
DSH_GPU_WORKER_ROOT=C:\dsh-gpu-worker\data
DSH_GPU_WORKER_MAX_CONCURRENT_TASKS=1
DSH_GPU_WORKER_ALLOW_GENERIC_PYTHON=true
```

要点：

- `DSH_GPU_WORKER_TOKEN` **必填**，少于 8 字符 Worker 会拒绝启动。
- `DSH_GPU_WORKER_ROOT` 是固定工作根：`experiments/`、`datasets/`、`scripts/jobs/`、
  `worker.db` 全部在其下，所有路径 resolve 后必须仍在根内。
  首次启动前把白名单脚本目录建好并把 `scripts\jobs\hello.py` 拷进去：
  ```powershell
  New-Item -ItemType Directory -Force C:\dsh-gpu-worker\data\scripts\jobs | Out-Null
  Copy-Item .\scripts\jobs\hello.py C:\dsh-gpu-worker\data\scripts\jobs\hello.py
  New-Item -ItemType Directory -Force C:\dsh-gpu-worker\data\datasets | Out-Null
  ```
- 只想本机调试就保持 `DSH_GPU_WORKER_HOST=127.0.0.1`；要给 Mac 调用才用 `0.0.0.0`（见 §5 防火墙）。

### 3.2 启动并本机冒烟

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

另开一个 PowerShell 窗口：

```powershell
$TOKEN = (Get-Content .env | Where-Object { $_ -match '^DSH_GPU_WORKER_TOKEN=' }) -replace '^DSH_GPU_WORKER_TOKEN=',''
$H = @{ Authorization = "Bearer $TOKEN" }

Invoke-RestMethod -Headers $H http://127.0.0.1:8765/health | ConvertTo-Json
Invoke-RestMethod -Headers $H http://127.0.0.1:8765/gpu    | ConvertTo-Json -Depth 5
```

期望 `/health` 的 `cuda_available: true`，`/gpu` 的 `devices[0].name` 是 RTX 5060。

再提交一个真实 CUDA 训练任务（10 epoch，几百样本，几十秒内完成）：

```powershell
$body = @{
  task_type = 'training'
  name = 'windows-cuda-smoke'
  config = @{
    device = 'cuda'
    epochs = 2
    batch_size = 32
    dataset = @{ dataset_id='synthetic'; format='synthetic'; num_samples=600; num_features=8; num_classes=3 }
    model   = @{ type='mlp'; input_dim=8; hidden_dims=@(64,32); num_classes=3 }
  }
} | ConvertTo-Json -Depth 6

$task = Invoke-RestMethod -Method Post -Headers $H -ContentType 'application/json' -Body $body http://127.0.0.1:8765/tasks
$task.task_id
```

轮询直到 `SUCCEEDED`，确认 `progress` 里有 epoch 且 `result.metrics` 含 device 名：

```powershell
$id = $task.task_id
Invoke-RestMethod -Headers $H "http://127.0.0.1:8765/tasks/$id" | ConvertTo-Json -Depth 5
Invoke-RestMethod -Headers $H "http://127.0.0.1:8765/tasks/$id/result" | ConvertTo-Json -Depth 6
Invoke-RestMethod -Headers $H "http://127.0.0.1:8765/tasks/$id/artifacts" | ConvertTo-Json -Depth 4
```

产物目录：`C:\dsh-gpu-worker\data\experiments\<task_id>\`（`model.pt`、`best_model.pt`、
`metrics.json`、`config.json`、`train.log`、`report.md`）。

## 4. 防火墙与网络

只有需要 Mac 跨机调用时才开放入站；不要直接暴露到公网。

```powershell
# 仅允许内网网段访问 8765/TCP（按实际网段修改）
New-NetFirewallRule -DisplayName "DSH GPU Worker 8765" -Direction Inbound `
  -Protocol TCP -LocalPort 8765 -Action Allow -Profile Private `
  -RemoteAddress 192.168.1.0/24

# 确认监听
Get-NetTCPConnection -LocalPort 8765 -State Listen
```

推荐拓扑：

- 同一 Wi-Fi/交换机下的私有网段 + 上面这条防火墙白名单；
- 需要跨公网时走 Tailscale/ZeroTier/WireGuard 之类的 VPN，而不是端口映射到公网；
- token 定期轮换（改 `.env` 后重启 Worker，同时更新 Mac 侧凭证）。

## 5. 开机自启

### 方案 A：任务计划程序（推荐，无第三方依赖）

以**管理员**身份运行 PowerShell：

```powershell
$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument '-NoProfile -ExecutionPolicy Bypass -File "C:\dsh-gpu-worker\dsh-gpu-worker\scripts\start.ps1"'
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName "DSH GPU Worker" -Action $action -Trigger $trigger `
  -Settings $settings -Principal $principal

# 立即启动并验证
Start-ScheduledTask -TaskName "DSH GPU Worker"
Start-Sleep 5
(Invoke-RestMethod -Headers $H http://127.0.0.1:8765/health).status
```

- 注意：用 `SYSTEM` 运行时 `.env` 所在路径必须可读、工作根目录可写。
- 若要 GPU 进程在“用户登录后”才用完整驱动能力，可把 `$principal` 改为
  `-UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive`，并把 trigger 换成 `-AtLogOn`。
- 卸载：`Unregister-ScheduledTask -TaskName "DSH GPU Worker" -Confirm:$false`。

### 方案 B：NSSM（把 worker 注册成 Windows 服务）

```powershell
# 下载 nssm 后（或已有 choco install nssm）
nssm install DSHGpuWorker "C:\dsh-gpu-worker\dsh-gpu-worker\.venv\Scripts\python.exe" "-m worker"
nssm set DSHGpuWorker AppDirectory "C:\dsh-gpu-worker\dsh-gpu-worker"
nssm set DSHGpuWorker AppEnvironmentExtra ^
  DSH_GPU_WORKER_TOKEN=<token> ^
  DSH_GPU_WORKER_HOST=0.0.0.0 ^
  DSH_GPU_WORKER_PORT=8765 ^
  DSH_GPU_WORKER_ROOT=C:\dsh-gpu-worker\data
nssm set DSHGpuWorker AppStdout "C:\dsh-gpu-worker\logs\worker.out.log"
nssm set DSHGpuWorker AppStderr "C:\dsh-gpu-worker\logs\worker.err.log"
nssm set DSHGpuWorker AppRotateFiles 1
nssm set DSHGpuWorker Start SERVICE_AUTO_START
nssm start DSHGpuWorker
```

- 与 `start.ps1` 不同，NSSM 直接启动 venv python，需自己通过 `AppEnvironmentExtra` 注入变量
  （不要把 token 写进服务描述）。
- 卸载：`nssm stop DSHGpuWorker & nssm remove DSHGpuWorker confirm`。

> 只选一种方案，不要同时用任务计划和 NSSM 抢同一端口。

## 6. 数据与产物目录约定

```
C:\dsh-gpu-worker\data\
├─ worker.db                 # SQLite 任务库（重启后历史查询）
├─ scripts\jobs\             # generic_python 白名单脚本（只允许 .py）
├─ datasets\                 # 训练/推理数据集，按 dataset_id 建子目录
│  └─ iris\data.csv          # csv: 默认 data.csv + target_column
│  └─ custom\X.npy  y.npy    # npy 格式
└─ experiments\<task_id>\    # 每个任务的隔离目录：配置、日志、产物
   ├─ job.json
   ├─ worker.log
   ├─ metrics.json / config.json / train.log / report.md
   └─ model.pt / best_model.pt
```

- 训练用 `csv`：`dataset.get("file")` 默认 `data.csv`，`target_column` 默认 `target`，
  `feature_columns` 省略则用其余数值列。
- 训练用 `npy`：`datasets/<dataset_id>/X.npy`、`y.npy`。
- 产物只能从 `experiments/<task_id>/` 内下载；路径逃逸返回 400 `PATH_ESCAPE`。
- 清理磁盘：只删已终态任务的 `experiments\<task_id>\`（DB 里的历史记录会保留但产物下载 404）。

## 7. Mac 侧对接

### 7.1 凭证

```bash
export DSH_GPU_WORKER_TOKEN='<与 Windows .env 相同的 token>'
# 或落盘给长期使用（600 权限）
mkdir -p ~/.dsh-remote-gpu
printf '%s\n' '<token>' > ~/.dsh-remote-gpu/token
chmod 600 ~/.dsh-remote-gpu/token
```

### 7.2 插件配置

`dsh-remote-gpu` 的 bundle patch（`dsh-remote-gpu/cordis.patch.yml`）默认指向
`http://127.0.0.1:8765`，跨机时改成 Windows 的局域网 IP（也可以不动 patch，
用环境变量 `DSH_GPU_WORKER_URL` 覆盖 `baseUrl`，见下表 `baseUrlEnv`）：

```yaml
- insert:
    - id: remote-gpu
      name: dsh-remote-gpu
      config:
        baseUrl: http://192.168.1.20:8765   # Windows 机器内网 IP
        tokenEnv: DSH_GPU_WORKER_TOKEN
        tokenFile: ~/.dsh-remote-gpu/token
        timeoutMs: 30000
```

插件配置字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `baseUrl` | `http://127.0.0.1:8765` | Worker 基地址（尾部 `/` 自动去掉） |
| `baseUrlEnv` | `DSH_GPU_WORKER_URL` | 该环境变量非空时**优先于** `baseUrl`（跨机切换 IP 不必改 patch） |
| `tokenEnv` | `DSH_GPU_WORKER_TOKEN` | 凭证环境变量名 |
| `tokenFile` | `~/.dsh-remote-gpu/token` | `tokenEnv` 缺失时的兜底凭证文件（`~` 展开、内容 trim） |
| `timeoutMs` | `30000` | 单次请求超时（毫秒） |

跨机快速切换（推荐给桌面端长期使用）：

```bash
export DSH_GPU_WORKER_URL='http://192.168.1.20:8765'
export DSH_GPU_WORKER_TOKEN='<token>'
```

启用/修改后重启 DSH 宿主，工具列表应出现 8 个 `remote_gpu_*` 工具。
用 `remote_gpu_health`、`remote_gpu_gpu_info` 先确认 `cuda_available: true` 和 RTX 5060 设备名。

### 7.3 跨机最小验证

```bash
# Mac 上直接打 Windows worker（绕过 DSH，先验证网络/端口/token）
TOKEN='<token>'
curl -s -H "Authorization: Bearer $TOKEN" http://192.168.1.20:8765/health
curl -s -H "Authorization: Bearer $TOKEN" http://192.168.1.20:8765/gpu
```

若 curl 能通而 DSH 工具报 `WORKER_UNREACHABLE`：检查 DSH 宿主进程是否继承了
`DSH_GPU_WORKER_TOKEN`（GUI 启动的进程常不继承 shell 的环境变量，改用 token 文件）；
若报 `WORKER_HTTP_ERROR 401`：token 不一致（注意 `.env` 里的引号/换行）。

## 8. 诊断与常见故障

| 症状 | 可能原因 | 处理 |
|---|---|---|
| Worker 启动即退出，日志 `FATAL: DSH_GPU_WORKER_TOKEN is not set` | `.env` 没加载/变量名拼错/start.ps1 读不到 | 用 `$env:DSH_GPU_WORKER_TOKEN` 检查；直接 `.\scripts\start.ps1` 看报错 |
| `/health` 的 `cuda_available: false` | 装了 CPU 版 torch / 驱动太旧 / venv 不对 | §2.3 自检；重装 cu128；确认 `sys.executable` 是 `.venv\Scripts\python.exe` |
| `/gpu` 有设备但训练报 `no kernel image is available` | torch wheel 与 sm_120 不匹配 | 换 CUDA 12.8+ wheel |
| 训练任务 `FAILED CUDA_UNAVAILABLE` | 提交了 `device:"cuda"` 但 Worker CUDA 不可用（契约要求不静默回退 CPU） | 先修 CUDA 环境，或把 `device` 改成 `auto`/`cpu` 跑通流程 |
| 端口被占用 `[Errno 10048]` | 已有 worker/NSSM 服务在跑 | `Get-NetTCPConnection -LocalPort 8765`，停掉旧实例 |
| Mac 端 `WORKER_UNREACHABLE` | 防火墙未放行 / Host 仍是 127.0.0.1 / IP 变化 | §4 防火墙 + `ping`/`Test-NetConnection`，`Test-NetConnection 192.168.1.20 -Port 8765` |
| Mac 端 `WORKER_HTTP_ERROR` 401 | token 不一致 | 比对两端 token；注意别把 `.env` 的 CRLF 带进 token |
| Mac 端 `CONFIG_MISSING` | DSH 宿主没环境变量且 token 文件不存在 | 用 `~/.dsh-remote-gpu/token` 文件方案 |
| 任务一直 `PENDING` | 并发上限 1 且已有任务在跑 | 查 `/tasks?status=RUNNING`，等或调大 `MAX_CONCURRENT_TASKS` |
| 重启后任务变 `FAILED` + `WORKER_RESTARTED` | 重启时任务未完成（契约规定不能误报成功） | 重新提交；查看 `worker.log` 判断是否被重启打断 |
| generic_python 脚本找不到 | 脚本不在 `<root>\scripts\jobs\` 或路径写成绝对路径/带 `..` | 放白名单目录，`config.script` 用 `jobs/xxx.py` 或 `xxx.py` |

日志位置：

- Worker 服务端日志：`start.ps1` 前台输出的 stdout；NSSM 方案在 `C:\dsh-gpu-worker\logs\worker.*.log`
- 每任务日志：`data\experiments\<task_id>\worker.log`（也可通过 API `GET /tasks/<id>/logs` 读）
- 训练明细：`data\experiments\<task_id>\train.log`

## 9. 安全 checklist

- [ ] `DSH_GPU_WORKER_TOKEN` 至少 32 字符随机串，只存 `.env`（ACL 限本人）或服务环境变量。
- [ ] 跨机只监听在可信内网/VPN，防火墙白名单到具体网段，不映射公网。
- [ ] 不用 `generic_python` 执行任意代码；只放可信的 `.py` 到 `scripts\jobs\`；不需要时设 `DSH_GPU_WORKER_ALLOW_GENERIC_PYTHON=false`。
- [ ] `data\` 目录 ACL 只给运行账户，不要让其他用户可写（脚本目录等价于代码执行入口）。
- [ ] 定期轮换 token（改 `.env` → 重启 Worker → 更新 Mac 侧环境变量/token 文件）。
- [ ] 验收真实 CUDA 时记录 `/gpu` 输出、任务 `result.metrics` 的 device 信息与产物 sha256，作为跨机证据。
