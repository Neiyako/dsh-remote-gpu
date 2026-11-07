/**
 * dsh-remote-gpu — Host half.
 *
 * 把独立 GPU Worker 的 HTTP API（docs/CONTRACT.md §2）暴露成 8 个 DSH 工具：
 *   remote_gpu_health / gpu_info / submit / status / logs / cancel / result / artifacts
 *
 * 设计边界：
 *   - 只注册工具，不改内置包、不改 provider/profile 配置；禁用 bundle 即恢复原样。
 *   - 凭证只从环境变量 tokenEnv（默认 DSH_GPU_WORKER_TOKEN）或 tokenFile 读取，
 *     绝不硬编码；无凭证时返回 CONFIG_MISSING 且不发任何请求。
 *   - 所有失败（网络/超时/非 2xx/坏 JSON/缺凭证）一律返回
 *     {ok:false,error:{code,message,details?}}，绝不把失败报成成功。
 *   - 使用 Node 内置 fetch + AbortSignal.timeout，不引入依赖。
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

export const name = 'dsh-remote-gpu'
export const inject = ['tools']

export const CONFIG_MISSING = 'CONFIG_MISSING'
export const WORKER_UNREACHABLE = 'WORKER_UNREACHABLE'
export const WORKER_TIMEOUT = 'WORKER_TIMEOUT'
export const WORKER_HTTP_ERROR = 'WORKER_HTTP_ERROR'
export const WORKER_BAD_RESPONSE = 'WORKER_BAD_RESPONSE'
export const INVALID_ARGUMENT = 'INVALID_ARGUMENT'
export const INTERNAL_ERROR = 'INTERNAL_ERROR'

const TASK_TYPES = ['training', 'inference', 'generic_python']
const DEFAULT_TIMEOUT_MS = 30000
const MAX_BODY_CHARS = 2000

const TOOL_OUTPUT_SCHEMA = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    result: { type: 'object', additionalProperties: true },
    error: {
      type: 'object',
      properties: {
        code: { type: 'string' },
        message: { type: 'string' },
        details: { type: 'object', additionalProperties: true },
      },
      required: ['code', 'message'],
    },
  },
  required: ['ok'],
}

/** text render：统一 JSON 美化，便于直接在对话里读。 */
export function textRender(_args, value) {
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2)
  return [{ type: 'text', text }]
}

/** 展开 `~` / `~/...`（只用 homedir，不读 shell 环境）。 */
export function expandHome(file, home = os.homedir()) {
  if (typeof file !== 'string') return file
  if (file === '~') return home
  if (file.startsWith('~/')) return path.join(home, file.slice(2))
  return file
}

/** 归一化配置：baseUrl/tokenEnv/tokenFile/timeoutMs。 */
export function resolveConfig(raw = {}) {
  const config = raw ?? {}
  const baseUrl = typeof config.baseUrl === 'string' && config.baseUrl.trim() !== ''
    ? config.baseUrl.trim().replace(/\/+$/, '')
    : 'http://127.0.0.1:8765'
  const tokenEnv = typeof config.tokenEnv === 'string' && config.tokenEnv.trim() !== ''
    ? config.tokenEnv.trim()
    : 'DSH_GPU_WORKER_TOKEN'
  const tokenFile = typeof config.tokenFile === 'string' && config.tokenFile.trim() !== ''
    ? expandHome(config.tokenFile.trim())
    : expandHome('~/.dsh-remote-gpu/token')
  const rawTimeout = Number(config.timeoutMs)
  const timeoutMs = Number.isFinite(rawTimeout) && rawTimeout > 0 ? Math.floor(rawTimeout) : DEFAULT_TIMEOUT_MS
  return { baseUrl, tokenEnv, tokenFile, timeoutMs }
}

function fail(code, message, details) {
  const error = { code, message }
  if (details !== undefined) error.details = details
  return { ok: false, error }
}

/**
 * 凭证解析：tokenEnv 优先，其次 tokenFile。返回 {ok:true,token,source} 或 {ok:false,error}。
 * 读文件失败/内容为空都视为该来源不可用，最终 CONFIG_MISSING；不回显 token 内容。
 */
export function resolveToken(config, env = process.env) {
  const fromEnv = env?.[config.tokenEnv]
  if (typeof fromEnv === 'string' && fromEnv.trim() !== '') {
    return { ok: true, token: fromEnv.trim(), source: `env:${config.tokenEnv}` }
  }
  let fileError
  try {
    const text = fs.readFileSync(config.tokenFile, 'utf8')
    if (text.trim() !== '') {
      return { ok: true, token: text.trim(), source: `file:${config.tokenFile}` }
    }
    fileError = 'token 文件为空'
  } catch (error) {
    fileError = error?.code ?? String(error?.message ?? error)
  }
  return fail(
    CONFIG_MISSING,
    `缺少 GPU Worker 凭证：环境变量 ${config.tokenEnv} 未设置，且 token 文件不可用（${config.tokenFile}：${fileError}）`,
    { tokenEnv: config.tokenEnv, tokenFile: config.tokenFile },
  )
}

function truncate(text) {
  const value = String(text ?? '')
  return value.length > MAX_BODY_CHARS ? `${value.slice(0, MAX_BODY_CHARS)}…[truncated]` : value
}

function describeFailure(method, url, error) {
  const code = error?.cause?.code ?? error?.code
  const reason = error?.cause?.message ?? error?.message ?? String(error)
  return code !== undefined ? `${code}: ${reason}` : reason
}

/**
 * 发一次 Worker 请求并归一化结果。
 * @returns {Promise<{ok:true,result:any}|{ok:false,error:{code,message,details?}}>}
 */
export async function workerRequest(config, { method = 'GET', route, query, body, fetchImpl = fetch } = {}) {
  const credential = resolveToken(config)
  if (!credential.ok) return credential

  const url = new URL(`${config.baseUrl}${route}`)
  if (query !== undefined) {
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null) url.searchParams.set(key, String(value))
    }
  }

  const headers = {
    Authorization: `Bearer ${credential.token}`,
    Accept: 'application/json',
  }
  let payload
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
    payload = JSON.stringify(body)
  }

  let response
  try {
    response = await fetchImpl(url, {
      method,
      headers,
      body: payload,
      signal: AbortSignal.timeout(config.timeoutMs),
    })
  } catch (error) {
    const isTimeout =
      error?.name === 'TimeoutError' ||
      error?.name === 'AbortError' ||
      error?.cause?.code === 'UND_ERR_CONNECT_TIMEOUT' ||
      error?.cause?.code === 'ETIMEDOUT' ||
      error?.code === 'ETIMEDOUT'
    if (isTimeout) {
      return fail(
        WORKER_TIMEOUT,
        `GPU Worker 请求超时（${config.timeoutMs}ms）：${method} ${url.origin}${url.pathname}`,
        { method, url: `${url.origin}${url.pathname}`, timeoutMs: config.timeoutMs },
      )
    }
    return fail(
      WORKER_UNREACHABLE,
      `GPU Worker 不可达：${method} ${url.origin}${url.pathname} — ${describeFailure(method, url, error)}`,
      { method, url: `${url.origin}${url.pathname}`, cause: describeFailure(method, url, error) },
    )
  }

  let text
  try {
    text = await response.text()
  } catch (error) {
    return fail(
      WORKER_UNREACHABLE,
      `读取 GPU Worker 响应失败：${method} ${url.origin}${url.pathname} — ${describeFailure(method, url, error)}`,
      { method, status: response.status },
    )
  }

  if (!response.ok) {
    // 非 2xx 一律失败，保留 HTTP status 与 Worker 错误 body。
    let workerCode
    let workerMessage
    let parsedBody
    try {
      parsedBody = JSON.parse(text)
      workerCode = parsedBody?.error?.code
      workerMessage = parsedBody?.error?.message
    } catch {
      parsedBody = undefined
    }
    const detail = workerCode !== undefined
      ? `${workerCode}: ${workerMessage ?? ''}`.trim()
      : truncate(text) || '(empty body)'
    return fail(
      WORKER_HTTP_ERROR,
      `GPU Worker 返回 HTTP ${response.status}：${method} ${url.pathname} — ${detail}`,
      {
        status: response.status,
        method,
        path: url.pathname,
        ...(workerCode !== undefined ? { workerCode } : {}),
        ...(workerMessage !== undefined ? { workerMessage } : {}),
        body: truncate(text),
      },
    )
  }

  if (text.trim() === '') {
    return fail(
      WORKER_BAD_RESPONSE,
      `GPU Worker 返回空响应体：${method} ${url.pathname}（HTTP ${response.status}）`,
      { status: response.status, method, path: url.pathname },
    )
  }

  try {
    return { ok: true, result: JSON.parse(text) }
  } catch {
    return fail(
      WORKER_BAD_RESPONSE,
      `GPU Worker 返回非法 JSON：${method} ${url.pathname}（HTTP ${response.status}）— ${truncate(text)}`,
      { status: response.status, method, path: url.pathname, body: truncate(text) },
    )
  }
}

function requireTaskId(args) {
  const taskId = args?.task_id
  if (typeof taskId !== 'string' || taskId.trim() === '') {
    return fail(INVALID_ARGUMENT, '参数 task_id 必须是非空字符串')
  }
  return { ok: true, taskId: taskId.trim() }
}

function taskRoute(taskId) {
  return `/tasks/${encodeURIComponent(taskId)}`
}

// ---- 参数 schema ----

const NO_ARGS = {
  type: 'object',
  properties: {},
  additionalProperties: false,
}

const TASK_ID_PARAMS = {
  type: 'object',
  properties: {
    task_id: { type: 'string', description: 'Worker 返回的 task_id' },
  },
  required: ['task_id'],
  additionalProperties: false,
}

const SUBMIT_PARAMS = {
  type: 'object',
  properties: {
    task_type: {
      type: 'string',
      enum: TASK_TYPES,
      description: '任务类型：training | inference | generic_python',
    },
    config: {
      type: 'object',
      additionalProperties: true,
      description: '任务配置，按 CONTRACT.md §3 对应 task_type 的结构透传',
    },
    name: { type: 'string', description: '可选任务名' },
    timeout_seconds: {
      type: 'integer',
      minimum: 10,
      maximum: 86400,
      description: '任务超时秒数，范围 10..86400（Worker 默认 3600）',
    },
    idempotency_key: {
      type: 'string',
      description: '幂等键：相同 key + 相同 payload 返回原任务，payload 不同返回 409',
    },
  },
  required: ['task_type'],
  additionalProperties: false,
}

const LOGS_PARAMS = {
  type: 'object',
  properties: {
    task_id: { type: 'string', description: 'Worker 返回的 task_id' },
    offset: { type: 'integer', minimum: 0, description: '行偏移，默认 0' },
    limit: { type: 'integer', minimum: 1, maximum: 2000, description: '返回行数上限 1..2000，默认 500' },
  },
  required: ['task_id'],
  additionalProperties: false,
}

/** 组装工具定义（供 apply 注册，也便于测试单独检查）。 */
export function toolDefinitions(config) {
  const call = (options) => workerRequest(config, options)

  return [
    {
      name: 'remote_gpu_health',
      description: '查询远程 GPU Worker 健康状态（GET /health）：status/version/cuda_available/active_tasks 等。注意 Worker 全部接口都需要 Bearer token。',
      parameters: NO_ARGS,
      output: { schema: TOOL_OUTPUT_SCHEMA, render: textRender },
      async execute(args = {}) {
        return call({ method: 'GET', route: '/health' })
      },
    },
    {
      name: 'remote_gpu_gpu_info',
      description: '查询远程 GPU 信息（GET /gpu）：CUDA 是否可用、设备列表、显存/利用率/温度/driver 版本。',
      parameters: NO_ARGS,
      output: { schema: TOOL_OUTPUT_SCHEMA, render: textRender },
      async execute(args = {}) {
        return call({ method: 'GET', route: '/gpu' })
      },
    },
    {
      name: 'remote_gpu_submit',
      description: '向远程 GPU Worker 提交任务（POST /tasks）：training | inference | generic_python。提交后立即返回 TaskView，不等待训练完成；透传 timeout_seconds 与 idempotency_key。',
      parameters: SUBMIT_PARAMS,
      output: { schema: TOOL_OUTPUT_SCHEMA, render: textRender },
      async execute(args = {}) {
        const taskType = args?.task_type
        if (!TASK_TYPES.includes(taskType)) {
          return fail(INVALID_ARGUMENT, `参数 task_type 必须是 ${TASK_TYPES.join(' | ')} 之一，收到：${String(taskType)}`)
        }
        if (args.config !== undefined && (args.config === null || typeof args.config !== 'object' || Array.isArray(args.config))) {
          return fail(INVALID_ARGUMENT, '参数 config 必须是 JSON 对象')
        }
        if (args.name !== undefined && typeof args.name !== 'string') {
          return fail(INVALID_ARGUMENT, '参数 name 必须是字符串')
        }
        if (args.timeout_seconds !== undefined) {
          const seconds = args.timeout_seconds
          if (!Number.isInteger(seconds) || seconds < 10 || seconds > 86400) {
            return fail(INVALID_ARGUMENT, '参数 timeout_seconds 必须是 10..86400 的整数')
          }
        }
        if (args.idempotency_key !== undefined && typeof args.idempotency_key !== 'string') {
          return fail(INVALID_ARGUMENT, '参数 idempotency_key 必须是字符串')
        }
        const body = { task_type: taskType, config: args.config ?? {} }
        if (args.name !== undefined) body.name = args.name
        if (args.timeout_seconds !== undefined) body.timeout_seconds = args.timeout_seconds
        if (args.idempotency_key !== undefined) body.idempotency_key = args.idempotency_key
        return call({ method: 'POST', route: '/tasks', body })
      },
    },
    {
      name: 'remote_gpu_status',
      description: '查询远程任务状态（GET /tasks/{task_id}）：status/progress/error/result 摘要。',
      parameters: TASK_ID_PARAMS,
      output: { schema: TOOL_OUTPUT_SCHEMA, render: textRender },
      async execute(args = {}) {
        const checked = requireTaskId(args)
        if (!checked.ok) return checked
        return call({ method: 'GET', route: taskRoute(checked.taskId) })
      },
    },
    {
      name: 'remote_gpu_logs',
      description: '分页读取远程任务日志（GET /tasks/{task_id}/logs?offset=&limit=），返回 lines/next_offset/eof。',
      parameters: LOGS_PARAMS,
      output: { schema: TOOL_OUTPUT_SCHEMA, render: textRender },
      async execute(args = {}) {
        const checked = requireTaskId(args)
        if (!checked.ok) return checked
        const query = {}
        if (args.offset !== undefined) {
          if (!Number.isInteger(args.offset) || args.offset < 0) {
            return fail(INVALID_ARGUMENT, '参数 offset 必须是 >=0 的整数')
          }
          query.offset = args.offset
        }
        if (args.limit !== undefined) {
          if (!Number.isInteger(args.limit) || args.limit < 1 || args.limit > 2000) {
            return fail(INVALID_ARGUMENT, '参数 limit 必须是 1..2000 的整数')
          }
          query.limit = args.limit
        }
        return call({ method: 'GET', route: `${taskRoute(checked.taskId)}/logs`, query })
      },
    },
    {
      name: 'remote_gpu_cancel',
      description: '请求取消远程任务（POST /tasks/{task_id}/cancel）：PENDING/RUNNING 可取消，终态返回 409 TASK_ALREADY_FINISHED。',
      parameters: TASK_ID_PARAMS,
      output: { schema: TOOL_OUTPUT_SCHEMA, render: textRender },
      async execute(args = {}) {
        const checked = requireTaskId(args)
        if (!checked.ok) return checked
        return call({ method: 'POST', route: `${taskRoute(checked.taskId)}/cancel` })
      },
    },
    {
      name: 'remote_gpu_result',
      description: '读取远程任务结果（GET /tasks/{task_id}/result）：终态返回 metrics/artifacts/experiment_dir；未终态返回 409 TASK_NOT_FINISHED。',
      parameters: TASK_ID_PARAMS,
      output: { schema: TOOL_OUTPUT_SCHEMA, render: textRender },
      async execute(args = {}) {
        const checked = requireTaskId(args)
        if (!checked.ok) return checked
        return call({ method: 'GET', route: `${taskRoute(checked.taskId)}/result` })
      },
    },
    {
      name: 'remote_gpu_artifacts',
      description: '列出远程任务产物（GET /tasks/{task_id}/artifacts）：path/size_bytes/sha256/download_url。',
      parameters: TASK_ID_PARAMS,
      output: { schema: TOOL_OUTPUT_SCHEMA, render: textRender },
      async execute(args = {}) {
        const checked = requireTaskId(args)
        if (!checked.ok) return checked
        return call({ method: 'GET', route: `${taskRoute(checked.taskId)}/artifacts` })
      },
    },
  ]
}

/** 兜底：execute 内任何未预期异常都转成结构化失败，绝不把异常抛给宿主。 */
function guardedExecute(execute) {
  return async (args = {}) => {
    try {
      return await execute(args)
    } catch (error) {
      return fail(INTERNAL_ERROR, `dsh-remote-gpu 内部错误：${error?.message ?? String(error)}`)
    }
  }
}

/** 注册 8 个工具；返回 disposer 便于手动卸载/测试。 */
export function apply(ctx, rawConfig) {
  const config = resolveConfig(rawConfig)
  const disposers = []
  for (const definition of toolDefinitions(config)) {
    const disposer = ctx.tools.register({ ...definition, execute: guardedExecute(definition.execute) })
    if (typeof disposer === 'function') disposers.push(disposer)
  }
  return () => {
    for (const dispose of disposers) {
      try {
        dispose()
      } catch {
        // 卸载失败不影响其它工具；DSH 侧仍有自己的生命周期清理
      }
    }
  }
}
