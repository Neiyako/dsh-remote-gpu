# dsh-remote-gpu

DSH Host 插件：把独立 GPU Worker 的 HTTP API 暴露成 8 个 `remote_gpu_*` 工具。
Worker API 契约见 [`../docs/CONTRACT.md`](../docs/CONTRACT.md) §2、§6。

- **Host-only**：只通过 `ctx.tools.register` 注册工具，不新增路由、不改内置包、不改 provider/profile 配置；移除 bundle 即恢复原样。
- **无依赖**：只用 Node 内置 `fetch` + `AbortSignal.timeout`（Node ≥ 20；已在 v24.14.1 验证）。
- **失败不伪装成功**：网络错误、超时、非 2xx、非法 JSON、缺凭证一律返回 `{ok:false,error:{code,message,details?}}`。

## 工具清单（严格 8 个）

| 工具 | HTTP 调用 | 说明 |
|---|---|---|
| `remote_gpu_health` | `GET /health` | Worker 状态、版本、`cuda_available`、活跃/排队任务 |
| `remote_gpu_gpu_info` | `GET /gpu` | CUDA 可用性、设备列表、显存/利用率/温度 |
| `remote_gpu_submit` | `POST /tasks` | 提交 `training` / `inference` / `generic_python`，立即返回 TaskView |
| `remote_gpu_status` | `GET /tasks/{task_id}` | 状态、progress、error |
| `remote_gpu_logs` | `GET /tasks/{task_id}/logs?offset=&limit=` | 分页日志（offset ≥ 0，limit 1..2000） |
| `remote_gpu_cancel` | `POST /tasks/{task_id}/cancel` | 取消 PENDING/RUNNING 任务 |
| `remote_gpu_result` | `GET /tasks/{task_id}/result` | 终态结果（metrics/artifacts/experiment_dir） |
| `remote_gpu_artifacts` | `GET /tasks/{task_id}/artifacts` | 产物列表（path/size/sha256/download_url） |

工具名只能含 `[A-Za-z0-9_-]`（DeepSeek API 对 tools[].name 的校验），不能用点号。

所有工具输出统一为：

```json
{"ok": true, "result": { ...Worker 原始 JSON... }}
```

或

```json
{"ok": false, "error": {"code": "WORKER_HTTP_ERROR", "message": "...", "details": {...}}}
```

并带 text render（JSON 美化），便于直接在对话里阅读。

`remote_gpu_submit` 参数：`task_type`（枚举）、`config`（对象）、`name`、`timeout_seconds`（10..86400）、`idempotency_key`，后三者按需透传，不做本地改名。

## 安装（由 Lead/owner 执行）

本仓库不自动安装到 DSH profile。安装与启用由 Lead 用 `plugin_manager` 完成：

1. 把本目录作为本地包安装/链接到当前 profile（`plugin_manager` action `install_bundle` 或 profile 的本地插件机制）；
2. 确认 bundle patch 生效：`cordis.patch.yml` 会插入

   ```yaml
   - insert:
       - id: remote-gpu
         name: dsh-remote-gpu
         config:
           baseUrl: http://127.0.0.1:8765
           tokenEnv: DSH_GPU_WORKER_TOKEN
           tokenFile: ~/.dsh-remote-gpu/token
           timeoutMs: 30000
   ```

3. 重启/刷新 DSH 宿主后，工具列表应出现上述 8 个 `remote_gpu_*` 工具。

安装前可选离线自检（本目录内，不联网、不写 profile）：

```bash
cd dsh-remote-gpu
node --check index.js
node --test
```

## 配置

| 字段 | 默认 | 说明 |
|---|---|---|
| `baseUrl` | `http://127.0.0.1:8765` | Worker 基地址（尾部 `/` 自动去掉） |
| `tokenEnv` | `DSH_GPU_WORKER_TOKEN` | 凭证环境变量名 |
| `tokenFile` | `~/.dsh-remote-gpu/token` | 环境变量缺失时的兜底凭证文件（`~` 会展开，内容 trim） |
| `timeoutMs` | `30000` | 单次请求超时（毫秒），超时返回 `WORKER_TIMEOUT` |

配置示例（`cordis.patch.yml` 中的 `config:` 块）：

```yaml
- insert:
    - id: remote-gpu
      name: dsh-remote-gpu
      config:
        baseUrl: http://192.168.1.20:8765
        tokenEnv: DSH_GPU_WORKER_TOKEN
        tokenFile: ~/.dsh-remote-gpu/token
        timeoutMs: 30000
```

## 凭证与环境变量

凭证解析顺序（**不得硬编码 token**）：

1. `process.env[tokenEnv]`（默认 `DSH_GPU_WORKER_TOKEN`），非空并 trim 后使用；
2. 否则读 `tokenFile`（默认 `~/.dsh-remote-gpu/token`，展开 `~`，trim）；
3. 两者都不可用 → 工具返回 `CONFIG_MISSING`，**不会发出任何无认证请求**。

配置凭证（任选其一）：

```bash
# 方式 A：环境变量（推荐给桌面端长期运行）
export DSH_GPU_WORKER_TOKEN='<worker-token>'

# 方式 B：token 文件（权限收紧）
mkdir -p ~/.dsh-remote-gpu
printf '%s\n' '<worker-token>' > ~/.dsh-remote-gpu/token
chmod 600 ~/.dsh-remote-gpu/token
```

Worker 侧所有接口（含 `/health`、`/gpu`）都需要认证；本插件每个请求都带
`Authorization: Bearer <token>`。

## 错误码

| code | 触发条件 |
|---|---|
| `CONFIG_MISSING` | `tokenEnv` 未设置且 `tokenFile` 缺失/为空/不可读（不发请求） |
| `WORKER_UNREACHABLE` | DNS/连接失败/连接被拒/读取响应体失败 |
| `WORKER_TIMEOUT` | 超过 `timeoutMs`（`AbortSignal.timeout`） |
| `WORKER_HTTP_ERROR` | 非 2xx；`message` 与 `details` 保留 HTTP status 和 Worker 错误 body |
| `WORKER_BAD_RESPONSE` | 2xx 但响应体为空或不是合法 JSON |
| `INVALID_ARGUMENT` | 本地参数校验失败（如非法 `task_type`、`limit` 越界） |
| `INTERNAL_ERROR` | 未预期的本地异常（不会把异常抛给宿主） |

## 安全

- 凭证只存于环境变量或权限为 `600` 的 token 文件；插件不打印 token，错误信息不回显 token。
- 工具只访问配置的 `baseUrl`；没有任意 URL / shell / 文件读写参数。
- Worker 默认监听 `127.0.0.1`；跨机使用时应走内网 + 防火墙白名单，并换强 token（见 Worker 文档）。
- 本插件不修改 DSH Core、`~/.dsh` profile 或其它插件。

## 回滚

1. 用 `plugin_manager` 禁用或移除 `dsh-remote-gpu` bundle（profile 配置改动由 Lead 执行）；
2. 移除后 `cordis.patch.yml` 的 insert 项不再生效，8 个工具消失，DSH 恢复原行为；
3. 如手工把 patch 写进了 profile，删除 `id: remote-gpu` 对应 insert 块，重启宿主；
4. 重启后确认工具列表不再包含 `remote_gpu_*`。

回滚不影响 Worker 侧任务与数据。

## 测试

```bash
cd dsh-remote-gpu
node --check index.js
node --test            # 或 npm test
```

`test/plugin.test.mjs` 用 `node:test` + 本地 fake HTTP server（`127.0.0.1:0`，不联网）覆盖：
8 个工具注册与 schema/text render、Bearer 认证头、`submit` 透传、`status`/`logs`/`cancel`/`result`/`artifacts` 路径映射、
非 2xx、401、不可达、超时（服务端延迟 > timeoutMs）、坏 JSON、空响应体、`tokenFile` 兜底、env 优先、无凭证不发请求。

## 已知限制

- 只做 API 客户端，不做任务编排/重试/轮询；需要时由调用方（agent）轮询 `remote_gpu_status`。
- 不支持二进制产物下载：`remote_gpu_artifacts` 只返回列表与 `download_url`，下载请用 Worker 提供的 URL（或后续工具扩展）。
- 无本地缓存 / 无离线模式；每次工具调用都实时请求 Worker。
- `INTERNAL_ERROR` 分支只兜底未预期异常，正常契约路径不会触发。
