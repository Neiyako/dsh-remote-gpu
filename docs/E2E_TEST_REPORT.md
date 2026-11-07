# E2E 验收报告 — DSH Remote GPU

> 报告生成时间：2026-11-07T22:25+08:00（UTC 2026-11-07T14:25Z）
> 代码基线：`d280a2a`（最终 HEAD；含 `nvidia-smi` GPU 指标 fallback）；插件 `DSH_GPU_WORKER_URL` 覆盖见 `be42e54`；Worker 核心实现见 `5707b78`
> 本机：macOS Darwin arm64 27.0.1，**无 NVIDIA GPU**，`torch.cuda.is_available() == False`。
> 本报告所有 PASS 均来自本机真实进程/真实 HTTP 调用，可复现命令见 §9；
> **真实 Windows RTX 5060 跨机 CUDA 没有机器，严格记为 `NOT_RUN`，未用本机结果冒充。**

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| HTTP 层 E2E（`e2e/run_e2e.py`） | **16 / 16 PASS，0 FAIL，exit 0** |
| 插件层 E2E（`e2e/plugin_e2e.mjs`） | **16 / 16 PASS，0 FAIL，exit 0** |
| Worker 接口/单元测试 | `python3 -m pytest -q` **90 passed**；`.venv-e2e` **89 passed, 1 skipped** |
| 插件单元测试 | `node --test` **20 passed, 0 fail** |
| 用户验收 1-10 | **8 PASS，2 NOT_RUN**（1、3 需要真实 Windows/CUDA 机器） |

用户验收逐条状态：

| # | 验收项 | 状态 | 一句话证据 |
|---|---|---|---|
| 1 | Mac 能查询 Windows GPU | **NOT_RUN** | 无 Windows 机器；本机 `gpu_info` 等价 PASS（`cuda_available=false`，不伪造设备） |
| 2 | Mac 能提交训练任务 | **PASS**（本机等价） | CPU training 提交 201 → `SUCCEEDED`（T16）；插件 `remote_gpu_submit` P04/P05 |
| 3 | Windows 能使用 CUDA 执行训练 | **NOT_RUN** | 本机无 CUDA（`torch.cuda.is_available()==False`），无 Windows 主机 |
| 4 | Mac 能实时获取训练进度 | **PASS** | 轮询捕获 `progress_snapshots=[1,2,4,6]`，终态 `epoch=6/6`（T16） |
| 5 | Mac 能取消任务 | **PASS** | RUNNING 任务 cancel → `CANCELLED`，二次 cancel 409（T11/P10） |
| 6 | 训练结束后能够获取结果 | **PASS** | SUCCEEDED 后 `result` 含 metrics/artifacts/experiment_dir（T16/T06/P07） |
| 7 | 能下载模型权重 | **PASS** | `model.pt` 下载字节 sha256 与清单一致，12767 字节（T16） |
| 8 | Worker 重启后能够查询历史任务 | **PASS** | 重启后历史 n=6，状态/日志/结果/产物均保留（T14） |
| 9 | CUDA 不可用时能够正确报错 | **PASS** | `device:"cuda"` → `FAILED` + `CUDA_UNAVAILABLE`，无 CPU 回退（T12/P11） |
| 10 | 网络中断时不会错误报告任务成功 | **PASS** | 死端口 `ConnectionFailed`；插件 `ok:false WORKER_UNREACHABLE`；401 也不报成功（T13/P12/P13/P14） |

## 2. 环境与版本

| 项 | 值 |
|---|---|
| 主机 | macOS Darwin arm64 27.0.1（Apple Silicon，无 NVIDIA GPU） |
| Worker 解释器 | `/Users/apple/Desktop/provider/.venv-e2e/bin/python`，Python 3.13.13 |
| PyTorch | 2.15.0（CPU wheel），`torch.cuda.is_available() == False` |
| Node | v24.14.1（插件自带 `node --test`、内置 `fetch`） |
| Worker | `dsh-gpu-worker` v0.1.0，监听 `127.0.0.1:8766`（自起；8765 留给 DSH 演示 server） |
| 代码基线 | `d280a2a`（最终 HEAD）；Worker 核心 commit `5707b78`；插件 env 覆盖 `be42e54`；另有 `632eb3f` 文档、`634148d` 工具名规范化 |
| 端口隔离 | E2E 只用 8766 / 动态空闲端口；错误端口测试用 `bind(0)` 现取的未被占用端口，避免与 8765 冲突 |

## 3. 运行命令与退出码（最终一轮）

| 命令 | 时间（UTC） | 退出码 | 结果 |
|---|---|---|---|
| `python3 e2e/run_e2e.py --json /tmp/e2e-final2.json` | 2026-11-07T14:22:29Z 起 | **0** | 16 PASS / 0 FAIL |
| `node e2e/plugin_e2e.mjs` | 2026-11-07T14:23:31Z–14:24:06Z | **0** | 16 PASS / 0 FAIL |
| **d280a2a 后复跑** `python3 e2e/run_e2e.py` / `node e2e/plugin_e2e.mjs` | 2026-11-07T14:29Z | **0 / 0** | 16 PASS / 16 PASS（GPU 指标 fallback 不影响 CPU E2E） |
| `cd dsh-gpu-worker && python3 -m pytest -q` | 2026-11-07T14:19:08Z 起（`634148d`，worker 树其后未改） | 0 | 90 passed |
| `cd dsh-gpu-worker && ../.venv-e2e/bin/python -m pytest -q` | 同上 | 0 | 89 passed, 1 skipped |
| `cd dsh-remote-gpu && node --check index.js && node --test` | 2026-11-07T14:22Z 起 | 0 | 20 passed, 0 fail |
| **提交后复跑** `python3 e2e/run_e2e.py`（HEAD=`2e26ba5`） | 2026-11-07T14:24:53Z 起 | **0** | 16 PASS / 0 FAIL（已提交的 harness 版本） |
| **提交后复跑** `node e2e/plugin_e2e.mjs`（HEAD=`2e26ba5`） | 同上 | **0** | 16 PASS / 0 FAIL（`TOTAL: 16 PASS, 0 FAIL`） |

跳过说明：`.venv-e2e` 里装了 torch，`tests/test_cuda.py:59` 的 “torch 不可用” 断言无法构造，故 skip
（system python 无 torch，该路径由 90 passed 的完整跑覆盖）。

## 4. HTTP 层 E2E 明细 — `e2e/run_e2e.py`（16/16 PASS）

命令：`python3 e2e/run_e2e.py`（自起 worker：临时 root + 随机 token，`127.0.0.1:8766`，结束后杀进程并清理临时目录）

| ID | 检查 | 状态 | 真实输出（截取自 stdout） |
|---|---|---|---|
| T01 | 认证拒绝缺/错 token（/health、/gpu） | PASS | `401 UNAUTHORIZED without token; 401 with wrong token; X-Worker-Token header -> 200` |
| T02 | `/health` 字段与计数 | PASS | `status=ok version=0.1.0 cuda_available=False active=0 queued=0` |
| T03 | `/gpu` 如实报告 CUDA（不伪造设备） | PASS | `cuda_available=False device_count=0 devices=[] error='CUDA unavailable: torch.cuda.is_available() is false and nvidia-smi found no device.'` |
| T04 | generic_python 提交 201 → SUCCEEDED | PASS | `task_id=c691fdee4dbe416cbe7111d483ccd765 status=SUCCEEDED created=PENDING` |
| T05 | `/logs` 有脚本输出、分页、无内部标记 | PASS | `lines=9 eof=True paging offset=2 ok marker_free=True` |
| T06 | `/result` 返回 metrics/artifacts/experiment_dir | PASS | `status=SUCCEEDED exit_code=0 experiment_dir=experiments/c691fdee... artifacts=4` |
| T07 | `/artifacts` 列出 sha256 且下载字节一致 | PASS | `artifacts=4 metrics.json sha256=c399e8aa2a473e65… bytes=100 match=True` |
| T08 | 产物路径逃逸 → 400 | PASS | `GET artifacts/../../worker.db -> 400 PATH_ESCAPE` |
| T09 | 幂等：同 key+payload 复用、异 payload 冲突 | PASS | `first=201 replay=200 same_id=True conflict=409 IDEMPOTENCY_CONFLICT` |
| T10 | 未终态取 result → 409 | PASS | `running; result -> 409 TASK_NOT_FINISHED details.status=RUNNING` |
| T11 | 取消 RUNNING → CANCELLED，二次 409 | PASS | `cancel -> 200, terminal status=CANCELLED error.code=CANCELLED, second cancel=409 TASK_ALREADY_FINISHED` |
| T12 | 强制 `device:"cuda"` → FAILED CUDA_UNAVAILABLE | PASS | `task_id=373ca4ca... status=FAILED error.code=CUDA_UNAVAILABLE message="device='cuda' was requested but CUDA is unavailable on this worker..."` |
| T13 | 错端口 → 连接失败（不误报成功） | PASS | `http://127.0.0.1:63339/health (free port) -> ConnectionFailed (URLError: [Errno 61] Connection refused)` |
| T14 | Worker 重启后历史/日志/结果/产物保留 | PASS | `restarted; history n=6 contains d5e92f69...; status/logs/result/artifacts preserved recovered_tasks=0` |
| T15 | 非法提交 → 400 | PASS | `shell task_type -> 400 INVALID_REQUEST; timeout=5 -> 400; script escape -> 400 PATH_ESCAPE` |
| T16 | CPU 训练成功 + 进度 + checkpoint + 权重下载 | PASS | `task_id=2a1ef627... SUCCEEDED progress_snapshots=[1, 2, 4, 6] progress.loss=0.0001/0.00014 artifacts=9 model.pt sha256=06fe56b0e7daa552… bytes=12767 metrics.device=cpu` |

T16 产物完整性：`model.pt`、`best_model.pt`、`metrics.json`、`config.json`、`train.log`、`report.md` 等
共 9 个文件；下载 `model.pt` 的字节数与 sha256 与 `/artifacts` 清单**逐字节比对一致**。

## 5. 插件层 E2E 明细 — `e2e/plugin_e2e.mjs`（16/16 PASS）

命令：`node e2e/plugin_e2e.mjs`。该 harness **导入真实插件源码** `dsh-remote-gpu/index.js`，
用假 `ctx.tools.register` 走真实 `apply()` 注册路径，并启动真实 worker（临时 root/token，8766）。

| ID | 检查 | 状态 | 真实输出（截取） |
|---|---|---|---|
| P01 | `apply()` 恰好注册 8 个下划线工具名 | PASS | `names=remote_gpu_health,remote_gpu_gpu_info,remote_gpu_submit,remote_gpu_status,remote_gpu_logs,remote_gpu_cancel,remote_gpu_result,remote_gpu_artifacts`（无点号，全部匹配 `^[A-Za-z0-9_-]+$`） |
| P02 | `remote_gpu_health` | PASS | `status=ok version=0.1.0 cuda_available=false` |
| P03 | `remote_gpu_gpu_info` CUDA 真实 | PASS | `cuda_available=false device_count=0 error=CUDA unavailable: ...` |
| P04 | `remote_gpu_submit` generic_python | PASS | `task_id=01405fcb... status=PENDING` |
| P05 | `remote_gpu_status` 轮询至 SUCCEEDED | PASS | `status=SUCCEEDED` |
| P06 | `remote_gpu_logs` | PASS | `lines=9 eof=true marker_free=True` |
| P07 | `remote_gpu_result` | PASS | `status=SUCCEEDED exit_code=0 artifacts=4` |
| P08 | `remote_gpu_artifacts` | PASS | `artifacts=4 metrics.sha256=13dc56a9e85af2d6… size=100` |
| P09 | `remote_gpu_submit` 幂等 | PASS | `replay_same_id=True conflict=WORKER_HTTP_ERROR(409 IDEMPOTENCY_CONFLICT)` |
| P10 | `remote_gpu_cancel` RUNNING → CANCELLED | PASS | `task_id=217a5fb6... RUNNING -> cancel -> CANCELLED` |
| P11 | 强制 CUDA 训练 → CUDA_UNAVAILABLE | PASS | `task_id=5f97abfc... FAILED CUDA_UNAVAILABLE` |
| P12 | 不可达端口 | PASS | `http://127.0.0.1:63466/health -> WORKER_UNREACHABLE (ECONNREFUSED ...)`（`ok:false`，未报成功） |
| P13 | 错误 token | PASS | `401 -> WORKER_HTTP_ERROR workerCode=UNAUTHORIZED (no false success)` |
| P14 | 缺凭证 | PASS | `CONFIG_MISSING (tokenEnv unset, tokenFile absent)`（不发请求） |
| P15 | 本地非法参数 | PASS | `task_type=shell / limit=9999 / missing task_id -> INVALID_ARGUMENT` |
| P16 | `baseUrlEnv` 环境变量覆盖 baseUrl | PASS | `E2E_WORKER_URL=http://127.0.0.1:8766 wins over config.baseUrl; env unset -> falls back to config.baseUrl and reports WORKER_UNREACHABLE` |

P16 说明：`baseUrlEnv`（默认 `DSH_GPU_WORKER_URL`）在 `apply()` 注册工具时解析一次；
跨机切换 IP 后需重启 DSH 宿主/重新加载插件才生效（已写入 `docs/DEPLOY-WINDOWS.md` §7.2 配置表）。

## 6. Worker 接口/单元测试

| 命令 | 结果 |
|---|---|
| `cd dsh-gpu-worker && python3 -m pytest -q` | `90 passed in 56.47s`（system Python，无 torch，覆盖 CUDA-unavailable 分支） |
| `cd dsh-gpu-worker && ../.venv-e2e/bin/python -m pytest -q` | `89 passed, 1 skipped in 49.73s`（skip 见 §3） |
| `cd dsh-remote-gpu && node --test` | `tests 20 / pass 20 / fail 0` |

## 7. DSH 侧真实工具调用证据（Lead 提供，未由本 harness 独立复跑）

Lead 在 `127.0.0.1:8765` 用真实 DSH 工具跑过（本机 CPU，无 CUDA）：
`remote_gpu_health`/`gpu_info` ok（`cuda_available=false`, `torch_installed=true`）；
generic task `8a2e994f...` SUCCEEDED（logs 7 行、result/artifacts 完整、下载 `hello.txt` sha256 匹配）；
forced cuda training `f7c97072...` FAILED `CUDA_UNAVAILABLE`；cancel training `f882f81f...` → `CANCELLED`；
CPU training 3 epochs `18bd8dcd...` SUCCEEDED（progress epoch 3/3、val_loss=0.006867、产物齐全）；
重启后 `8a2e994f...` 仍可查、产物 sha256 不变。

以上属于 DSH 宿主层佐证，**本报告不把它当作 PASS 依据**（PASS 依据是 §4/§5 本机 harness 输出）；
仅用于说明插件在真实宿主里也工作。

## 8. NOT_RUN 与已知限制（诚实声明）

| 项 | 状态 | 原因 / 待办 |
|---|---|---|
| 真实 Windows RTX 5060 跨机 CUDA 训练（用户验收 1、3） | **NOT_RUN** | 本机是 Apple Silicon + CPU torch，无 Windows/CUDA 主机，无法产生真实 GPU 训练证据。需要在 Windows 机器上按 `docs/DEPLOY-WINDOWS.md` 部署后补测并记录：`/gpu` 设备名、`result.metrics` 的 device/torch/cuda 版本、epoch 进度、`model.pt` sha256。 |
| 本机 CUDA 成功路径 | NOT_RUN（同一原因） | 本机只验证了 “CUDA 不可用 → 明确失败” 的契约语义（验收 9 PASS）。 |
| 插件二进制下载工具 | 限制 | `remote_gpu_artifacts` 只返回清单/`download_url`；本报告验收 7 的下载走 Worker HTTP `GET /tasks/{id}/artifacts/model.pt`（curl/urllib），不是插件工具。 |
| 跨机网络故障（真断网/Wi-Fi 掉线） | 未模拟 | 用 “不可达端口/错误 token/缺凭证” 等价覆盖（T13/P12/P13/P14）；真实链路层断网未在本机复现。 |
| 多 worker / 高并发 | 未覆盖 | Worker 单进程单 SQLite，并发默认 1；不在本次验收范围。 |

## 9. 复现方法

```bash
cd /Users/apple/Desktop/provider

# 1) HTTP 层（16 项，自起 worker 于 8766，退出码 0=全 PASS）
python3 e2e/run_e2e.py
# python3 e2e/run_e2e.py --keep-temp   # 保留临时 root 与 worker 日志排查
# python3 e2e/run_e2e.py --json /tmp/e2e.json

# 2) 插件层（16 项，导入真实 index.js，假 ctx 注册，8 工具打真 worker）
node e2e/plugin_e2e.mjs
# E2E_PLUGIN_JSON=/tmp/plugin.json node e2e/plugin_e2e.mjs

# 3) 单元/接口测试
cd dsh-gpu-worker && python3 -m pytest -q && cd ..
cd dsh-remote-gpu && node --check index.js && node --test && cd ..
```

两个 E2E harness 的约定：全部检查 PASS 才 `exit 0`（任一 FAIL 非 0）；
每个检查打印 `[PASS]/[FAIL] <id> <名称> :: <证据>`，末尾打印汇总表与完整 JSON；
结束时杀 worker 进程并清理临时目录（`--keep-temp` 可保留）。

## 10. 结论

- 本机可控范围内（Mac CPU、无 CUDA）：HTTP 契约、认证、任务生命周期（提交/进度/日志/取消/结果/产物/下载/幂等/重启恢复）、
  插件 8 工具与全部失败路径（不可达/超时语义/401/缺凭证/参数非法）**全部按契约工作，16+16 项全 PASS**。
- 用户验收 10 项中 **8 项 PASS**；验收 1、3 必须依赖真实 Windows RTX 5060 机器，
  当前严格 `NOT_RUN`，未编造任何 GPU 训练结果。补齐硬件后按本报告 §9 同样的命令重跑即可得到可对照的证据。
