/**
 * dsh-remote-gpu —— 离线测试（node:test + 本地 fake HTTP worker）。
 *
 * 覆盖：8 个工具注册、认证头、凭证优先级（env → tokenFile → CONFIG_MISSING）、
 * submit 透传、status/logs/cancel/result/artifacts 路径映射、非 2xx、不可达、
 * 超时、坏 JSON、text render。
 *
 * 全程不发外网请求：所有请求打到 127.0.0.1 的临时 fake worker。
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import { apply, currentBaseUrl, resolveConfig, resolveToken } from '../index.js'

const TOKEN = 'test-token-abc123'
const ENV_NAME = 'DSH_REMOTE_GPU_TEST_TOKEN'
const URL_ENV_NAME = 'DSH_GPU_WORKER_URL'
delete process.env[URL_ENV_NAME]
const TOOL_NAMES = [
  'remote_gpu_health',
  'remote_gpu_gpu_info',
  'remote_gpu_submit',
  'remote_gpu_status',
  'remote_gpu_logs',
  'remote_gpu_cancel',
  'remote_gpu_result',
  'remote_gpu_artifacts',
]

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'dsh-remote-gpu-test-'))
const MISSING_FILE = path.join(TMP, 'no-such-token')
const TOKEN_FILE = path.join(TMP, 'token')

process.on('exit', () => {
  try {
    fs.rmSync(TMP, { recursive: true, force: true })
  } catch {
    // 清理失败不影响测试结果
  }
})

function setEnvToken(value) {
  if (value === undefined) delete process.env[ENV_NAME]
  else process.env[ENV_NAME] = value
}

/** 启动本地 fake worker，记录每个请求（method/url/headers/body）。 */
async function startFakeWorker(handler) {
  const requests = []
  const server = http.createServer(async (req, res) => {
    const chunks = []
    for await (const chunk of req) chunks.push(chunk)
    const record = {
      method: req.method,
      url: req.url,
      headers: req.headers,
      body: Buffer.concat(chunks).toString('utf8'),
    }
    requests.push(record)
    try {
      await handler(req, res, record)
    } catch (error) {
      res.writeHead(500, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: { code: 'FAKE_HANDLER_CRASH', message: String(error?.message ?? error) } }))
    }
  })
  await new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })
  const { port } = server.address()
  const close = () =>
    new Promise((resolve) => {
      server.close(resolve)
      server.closeAllConnections?.()
    })
  return { server, port, baseUrl: `http://127.0.0.1:${port}`, requests, close }
}

function json(res, status, value) {
  res.writeHead(status, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify(value))
}

/** 用假 ctx 装配插件，返回 name → 工具定义。 */
function loadPlugin(config) {
  const registered = new Map()
  const ctx = {
    tools: {
      register(definition) {
        assert.equal(registered.has(definition.name), false, `重复注册工具：${definition.name}`)
        registered.set(definition.name, definition)
        return () => registered.delete(definition.name)
      },
    },
  }
  const dispose = apply(ctx, config)
  return { registered, dispose }
}

function configFor(baseUrl, extra = {}) {
  return {
    baseUrl,
    baseUrlFile: MISSING_FILE,
    tokenEnv: ENV_NAME,
    tokenFile: MISSING_FILE,
    timeoutMs: 2000,
    ...extra,
  }
}

function getTool(registered, toolName) {
  const definition = registered.get(toolName)
  assert.ok(definition, `工具未注册：${toolName}`)
  return definition
}

// ---------------------------------------------------------------------------
// 1. 注册与 schema
// ---------------------------------------------------------------------------

test('注册且仅注册 8 个工具，schema 与 text render 完整', () => {
  const { registered, dispose } = loadPlugin(configFor('http://127.0.0.1:1'))
  assert.deepEqual([...registered.keys()], TOOL_NAMES)
  assert.equal(registered.size, 8)

  for (const definition of registered.values()) {
    assert.equal(typeof definition.description, 'string')
    assert.ok(definition.description.length > 0)
    assert.equal(definition.parameters.type, 'object')
    assert.equal(definition.output.schema.type, 'object')
    assert.equal(typeof definition.output.render, 'function')
    assert.equal(typeof definition.execute, 'function')

    const rendered = definition.output.render({}, { ok: true, result: { hello: 'world' } })
    assert.equal(Array.isArray(rendered), true)
    assert.equal(rendered[0].type, 'text')
    assert.deepEqual(JSON.parse(rendered[0].text), { ok: true, result: { hello: 'world' } })
    assert.match(rendered[0].text, /\n  "ok": true/)
  }

  dispose()
  assert.equal(registered.size, 0, 'disposer 应卸载全部工具')
})

test('resolveConfig 默认值与 ~ 展开正确', () => {
  const config = resolveConfig({})
  assert.equal(config.baseUrl, 'http://127.0.0.1:8765')
  assert.equal(config.baseUrlEnv, 'DSH_GPU_WORKER_URL')
  assert.equal(config.baseUrlFile, path.join(os.homedir(), '.dsh-remote-gpu', 'url'))
  assert.equal(config.tokenEnv, 'DSH_GPU_WORKER_TOKEN')
  assert.equal(config.tokenFile, path.join(os.homedir(), '.dsh-remote-gpu', 'token'))
  assert.equal(config.timeoutMs, 30000)

  const custom = resolveConfig({
    baseUrl: 'http://10.0.0.5:8765/',
    tokenEnv: 'MY_TOKEN',
    tokenFile: '~/secret/token',
    timeoutMs: 500,
  })
  assert.equal(custom.baseUrl, 'http://10.0.0.5:8765')
  assert.equal(custom.tokenEnv, 'MY_TOKEN')
  assert.equal(custom.tokenFile, path.join(os.homedir(), 'secret', 'token'))
  assert.equal(custom.timeoutMs, 500)
})

test('baseUrlFile 每次调用时重新读取（Mac 写文件即可切 Windows IP）', () => {
  const urlFile = path.join(TMP, 'worker-url.txt')
  fs.writeFileSync(urlFile, 'http://127.0.0.1:1/\n', 'utf8')
  const config = resolveConfig({ baseUrl: 'http://127.0.0.1:8765', baseUrlFile: urlFile })
  assert.equal(currentBaseUrl(config), 'http://127.0.0.1:1')
  fs.writeFileSync(urlFile, 'http://127.0.0.1:2\n', 'utf8')
  assert.equal(currentBaseUrl(config), 'http://127.0.0.1:2')
})

test('DSH_GPU_WORKER_URL 环境变量覆盖 baseUrl（Mac 侧无需改配置）', () => {
  const saved = process.env[URL_ENV_NAME]
  try {
    process.env[URL_ENV_NAME] = 'http://172.20.10.99:8765/'
    const config = resolveConfig({ baseUrl: 'http://127.0.0.1:8765' })
    assert.equal(config.baseUrl, 'http://172.20.10.99:8765')
    assert.equal(config.baseUrlEnv, URL_ENV_NAME)
  } finally {
    if (saved === undefined) delete process.env[URL_ENV_NAME]
    else process.env[URL_ENV_NAME] = saved
  }
})

// ---------------------------------------------------------------------------
// 2. 认证头 / health / gpu_info
// ---------------------------------------------------------------------------

test('health 携带 Bearer 认证头并原样返回结果', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((req, res) => {
    assert.equal(req.url, '/health')
    json(res, 200, {
      status: 'ok',
      worker_id: 'w-1',
      version: '0.1.0',
      cuda_available: false,
      active_tasks: 0,
    })
  })
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl, { tokenFile: TOKEN_FILE }))
  const result = await getTool(registered, 'remote_gpu_health').execute({})

  assert.equal(worker.requests.length, 1)
  assert.equal(worker.requests[0].method, 'GET')
  assert.equal(worker.requests[0].headers.authorization, `Bearer ${TOKEN}`)
  assert.deepEqual(result, {
    ok: true,
    result: { status: 'ok', worker_id: 'w-1', version: '0.1.0', cuda_available: false, active_tasks: 0 },
  })
})

test('gpu_info 请求 GET /gpu', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((req, res) => {
    assert.equal(req.url, '/gpu')
    json(res, 200, { cuda_available: false, device_count: 0, devices: [], error: 'no CUDA' })
  })
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl))
  const result = await getTool(registered, 'remote_gpu_gpu_info').execute({})
  assert.equal(result.ok, true)
  assert.deepEqual(result.result, { cuda_available: false, device_count: 0, devices: [], error: 'no CUDA' })
})

// ---------------------------------------------------------------------------
// 3. submit 透传
// ---------------------------------------------------------------------------

test('submit 以 JSON POST /tasks 透传 config/name/timeout_seconds/idempotency_key', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((_req, res) => {
    json(res, 201, { task_id: 'task-1', task_type: 'training', status: 'PENDING' })
  })
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const configInput = { epochs: 2, dataset: { dataset_id: 'synthetic' }, device: 'cpu' }
  const { registered } = loadPlugin(configFor(worker.baseUrl))
  const result = await getTool(registered, 'remote_gpu_submit').execute({
    task_type: 'training',
    config: configInput,
    name: 'smoke',
    timeout_seconds: 120,
    idempotency_key: 'idem-1',
  })

  assert.equal(result.ok, true)
  assert.equal(result.result.task_id, 'task-1')
  const request = worker.requests[0]
  assert.equal(request.method, 'POST')
  assert.equal(request.url, '/tasks')
  assert.equal(request.headers.authorization, `Bearer ${TOKEN}`)
  assert.match(request.headers['content-type'], /application\/json/)
  assert.deepEqual(JSON.parse(request.body), {
    task_type: 'training',
    config: configInput,
    name: 'smoke',
    timeout_seconds: 120,
    idempotency_key: 'idem-1',
  })
})

test('submit 缺省只发送 task_type 与空 config', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((_req, res) => json(res, 201, { task_id: 'task-2', status: 'PENDING' }))
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl))
  const result = await getTool(registered, 'remote_gpu_submit').execute({ task_type: 'generic_python' })
  assert.equal(result.ok, true)
  assert.deepEqual(JSON.parse(worker.requests[0].body), { task_type: 'generic_python', config: {} })
})

test('submit 本地校验非法参数且不发请求', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((_req, res) => json(res, 201, {}))
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl))
  const submit = getTool(registered, 'remote_gpu_submit')
  for (const args of [
    { task_type: 'not_a_type' },
    { task_type: 'training', config: [] },
    { task_type: 'training', timeout_seconds: 5 },
    { task_type: 'training', timeout_seconds: 90000 },
    { task_type: 'training', name: 42 },
    { task_type: 'training', idempotency_key: 7 },
  ]) {
    const result = await submit.execute(args)
    assert.equal(result.ok, false, `应拒绝参数：${JSON.stringify(args)}`)
    assert.equal(result.error.code, 'INVALID_ARGUMENT')
  }
  assert.equal(worker.requests.length, 0, '非法参数不应发出任何 HTTP 请求')
})

// ---------------------------------------------------------------------------
// 4. status / logs / cancel / result / artifacts 路径映射
// ---------------------------------------------------------------------------

test('status/logs/cancel/result/artifacts 映射到契约路径与查询参数', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((req, res) => {
    if (req.url.startsWith('/tasks/task%201%2F2/logs')) {
      return json(res, 200, { task_id: 'task 1/2', offset: 120, next_offset: 130, lines: [], eof: false })
    }
    if (req.url === '/tasks/task%201%2F2/cancel') {
      return json(res, 200, { task_id: 'task 1/2', status: 'CANCELLED' })
    }
    if (req.url === '/tasks/task%201%2F2/result') {
      return json(res, 200, { task_id: 'task 1/2', status: 'SUCCEEDED', result: { metrics: {} }, error: null })
    }
    if (req.url === '/tasks/task%201%2F2/artifacts') {
      return json(res, 200, { task_id: 'task 1/2', artifacts: [] })
    }
    if (req.url === '/tasks/task%201%2F2') {
      return json(res, 200, { task_id: 'task 1/2', status: 'RUNNING', progress: { epoch: 1 } })
    }
    return json(res, 404, { error: { code: 'TASK_NOT_FOUND', message: `unexpected ${req.url}` } })
  })
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl))
  const taskId = 'task 1/2'

  const status = await getTool(registered, 'remote_gpu_status').execute({ task_id: taskId })
  assert.equal(status.ok, true)
  assert.equal(status.result.status, 'RUNNING')
  assert.equal(worker.requests.at(-1).method, 'GET')
  assert.equal(worker.requests.at(-1).url, '/tasks/task%201%2F2')

  const logs = await getTool(registered, 'remote_gpu_logs').execute({ task_id: taskId, offset: 120, limit: 10 })
  assert.equal(logs.ok, true)
  assert.equal(worker.requests.at(-1).url, '/tasks/task%201%2F2/logs?offset=120&limit=10')

  const cancel = await getTool(registered, 'remote_gpu_cancel').execute({ task_id: taskId })
  assert.equal(cancel.ok, true)
  assert.equal(cancel.result.status, 'CANCELLED')
  assert.equal(worker.requests.at(-1).method, 'POST')
  assert.equal(worker.requests.at(-1).url, '/tasks/task%201%2F2/cancel')
  assert.equal(worker.requests.at(-1).body, '')

  const result = await getTool(registered, 'remote_gpu_result').execute({ task_id: taskId })
  assert.equal(result.ok, true)
  assert.equal(result.result.status, 'SUCCEEDED')
  assert.equal(worker.requests.at(-1).url, '/tasks/task%201%2F2/result')

  const artifacts = await getTool(registered, 'remote_gpu_artifacts').execute({ task_id: taskId })
  assert.equal(artifacts.ok, true)
  assert.equal(worker.requests.at(-1).url, '/tasks/task%201%2F2/artifacts')
})

test('logs 本地校验 offset/limit 与缺失 task_id', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((_req, res) => json(res, 200, {}))
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl))
  const logs = getTool(registered, 'remote_gpu_logs')
  for (const args of [
    {},
    { task_id: '' },
    { task_id: 't', offset: -1 },
    { task_id: 't', limit: 0 },
    { task_id: 't', limit: 2001 },
    { task_id: 't', limit: 1.5 },
  ]) {
    const result = await logs.execute(args)
    assert.equal(result.ok, false, `应拒绝参数：${JSON.stringify(args)}`)
    assert.equal(result.error.code, 'INVALID_ARGUMENT')
  }
  assert.equal(worker.requests.length, 0)
})

// ---------------------------------------------------------------------------
// 5. 失败路径：非 2xx / 不可达 / 超时 / 坏 JSON
// ---------------------------------------------------------------------------

test('非 2xx 返回 WORKER_HTTP_ERROR，保留 status 与 worker 错误 body', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((_req, res) => {
    json(res, 409, { error: { code: 'TASK_NOT_FINISHED', message: 'task is RUNNING', details: { status: 'RUNNING' } } })
  })
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl))
  const result = await getTool(registered, 'remote_gpu_result').execute({ task_id: 'task-x' })

  assert.equal(result.ok, false, '非 2xx 绝不能 ok:true')
  assert.equal(result.error.code, 'WORKER_HTTP_ERROR')
  assert.match(result.error.message, /HTTP 409/)
  assert.match(result.error.message, /TASK_NOT_FINISHED/)
  assert.match(result.error.message, /task is RUNNING/)
  assert.equal(result.error.details.status, 409)
  assert.equal(result.error.details.workerCode, 'TASK_NOT_FINISHED')
  assert.match(result.error.details.body, /TASK_NOT_FINISHED/)
})

test('401 也算失败并透出 Worker 错误码', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((_req, res) => {
    json(res, 401, { error: { code: 'UNAUTHORIZED', message: 'bad token' } })
  })
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl))
  const result = await getTool(registered, 'remote_gpu_health').execute({})
  assert.equal(result.ok, false)
  assert.equal(result.error.code, 'WORKER_HTTP_ERROR')
  assert.match(result.error.message, /HTTP 401/)
  assert.equal(result.error.details.workerCode, 'UNAUTHORIZED')
})

test('不可达端口返回 WORKER_UNREACHABLE', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((_req, res) => json(res, 200, {}))
  const deadPort = worker.port
  await worker.close()

  const { registered } = loadPlugin(configFor(`http://127.0.0.1:${deadPort}`, { timeoutMs: 1000 }))
  t.after(() => setEnvToken(undefined))

  const result = await getTool(registered, 'remote_gpu_health').execute({})
  assert.equal(result.ok, false, '不可达绝不能 ok:true')
  assert.equal(result.error.code, 'WORKER_UNREACHABLE')
  assert.equal(result.error.details.method, 'GET')
})

test('fake server 延迟超过 timeoutMs 返回 WORKER_TIMEOUT', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((_req, res) => {
    setTimeout(() => {
      if (!res.writableEnded) json(res, 200, { status: 'ok' })
    }, 400)
  })
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl, { timeoutMs: 80 }))
  const startedAt = Date.now()
  const result = await getTool(registered, 'remote_gpu_health').execute({})
  const elapsed = Date.now() - startedAt

  assert.equal(result.ok, false, '超时绝不能 ok:true')
  assert.equal(result.error.code, 'WORKER_TIMEOUT')
  assert.match(result.error.message, /超时/)
  assert.ok(elapsed < 350, `应在超时后尽快返回，实际 ${elapsed}ms`)
})

test('200 但响应体不是 JSON 返回 WORKER_BAD_RESPONSE', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end('not-json{')
  })
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl))
  const result = await getTool(registered, 'remote_gpu_status').execute({ task_id: 'task-1' })
  assert.equal(result.ok, false, '坏 JSON 绝不能 ok:true')
  assert.equal(result.error.code, 'WORKER_BAD_RESPONSE')
  assert.match(result.error.message, /not-json/)
})

test('200 但响应体为空也返回 WORKER_BAD_RESPONSE', async (t) => {
  setEnvToken(TOKEN)
  const worker = await startFakeWorker((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end()
  })
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl))
  const result = await getTool(registered, 'remote_gpu_health').execute({})
  assert.equal(result.ok, false)
  assert.equal(result.error.code, 'WORKER_BAD_RESPONSE')
})

// ---------------------------------------------------------------------------
// 6. 凭证：env 优先 / tokenFile 兜底 / CONFIG_MISSING 不发请求
// ---------------------------------------------------------------------------

test('tokenFile 兜底：env 未设置时读取文件并 trim', async (t) => {
  setEnvToken(undefined)
  fs.writeFileSync(TOKEN_FILE, `  file-token-xyz\n`, { mode: 0o600 })
  const worker = await startFakeWorker((_req, res) => json(res, 200, { status: 'ok' }))
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
    fs.rmSync(TOKEN_FILE, { force: true })
  })

  const config = resolveConfig(configFor(worker.baseUrl, { tokenFile: TOKEN_FILE }))
  const credential = resolveToken(config)
  assert.equal(credential.ok, true)
  assert.equal(credential.token, 'file-token-xyz')
  assert.equal(credential.source, `file:${TOKEN_FILE}`)

  const { registered } = loadPlugin(config)
  const result = await getTool(registered, 'remote_gpu_health').execute({})
  assert.equal(result.ok, true)
  assert.equal(worker.requests[0].headers.authorization, 'Bearer file-token-xyz')
})

test('env 凭证优先于 tokenFile', async (t) => {
  setEnvToken(TOKEN)
  fs.writeFileSync(TOKEN_FILE, 'file-token-should-lose\n', { mode: 0o600 })
  const worker = await startFakeWorker((_req, res) => json(res, 200, { status: 'ok' }))
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
    fs.rmSync(TOKEN_FILE, { force: true })
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl, { tokenFile: TOKEN_FILE }))
  const result = await getTool(registered, 'remote_gpu_health').execute({})
  assert.equal(result.ok, true)
  assert.equal(worker.requests[0].headers.authorization, `Bearer ${TOKEN}`)
})

test('无凭证返回 CONFIG_MISSING 且不发出任何请求', async (t) => {
  setEnvToken(undefined)
  const worker = await startFakeWorker((_req, res) => json(res, 200, { status: 'ok' }))
  t.after(async () => {
    await worker.close()
    setEnvToken(undefined)
  })

  const { registered } = loadPlugin(configFor(worker.baseUrl))
  const result = await getTool(registered, 'remote_gpu_health').execute({})
  assert.equal(result.ok, false, '缺凭证绝不能 ok:true')
  assert.equal(result.error.code, 'CONFIG_MISSING')
  assert.match(result.error.message, new RegExp(ENV_NAME))
  assert.match(result.error.message, /token/)
  assert.equal(worker.requests.length, 0, '无凭证不得发送无认证请求')
})

// ---------------------------------------------------------------------------
// 7. 兜底：execute 未预期异常不外抛
// ---------------------------------------------------------------------------

test('非法 baseUrl 触发内部异常时返回 INTERNAL_ERROR 而不是抛出', async (t) => {
  setEnvToken(TOKEN)
  t.after(() => setEnvToken(undefined))

  const { registered } = loadPlugin(configFor('not-a-valid-url'))
  const result = await getTool(registered, 'remote_gpu_health').execute({})
  assert.equal(result.ok, false)
  assert.equal(result.error.code, 'INTERNAL_ERROR')
  assert.equal(typeof result.error.message, 'string')
})

