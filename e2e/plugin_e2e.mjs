#!/usr/bin/env node
/**
 * Plugin-level E2E: imports the real DSH plugin (`../dsh-remote-gpu/index.js`),
 * registers its 8 tool definitions through a fake `ctx.tools.register`, and calls
 * every tool against a *real* worker process.
 *
 * This exercises the actual production code path used by the DSH host:
 *   `apply(ctx, config)` -> `toolDefinitions(config)` -> `workerRequest(...)`
 * with the real `fetch`, real Bearer auth, real JSON envelope, and the real
 * underscore tool names (`remote_gpu_health` etc. — DeepSeek tool names may not
 * contain dots).
 *
 * The harness starts its own worker (temp root + token, port 8766 by default)
 * unless `--reuse` is given and something already answers on the port.
 *
 * Usage:
 *   node e2e/plugin_e2e.mjs
 *   node e2e/plugin_e2e.mjs --keep-temp
 *   E2E_PLUGIN_PORT=8770 node e2e/plugin_e2e.mjs
 *
 * Exit code: 0 only when every check PASSes.
 */

import { spawn } from 'node:child_process'
import crypto from 'node:crypto'
import net from 'node:net'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { apply, toolDefinitions } from '../dsh-remote-gpu/index.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PROJECT_DIR = path.resolve(__dirname, '..')
const WORKER_DIR = path.join(PROJECT_DIR, 'dsh-gpu-worker')
const HOST = '127.0.0.1'
const PORT = Number(process.env.E2E_PLUGIN_PORT || 8766)
const UNREACHABLE_PORT = Number(process.env.E2E_PLUGIN_UNREACHABLE_PORT || 8769)
const BASE = `http://${HOST}:${PORT}`
const STARTUP_TIMEOUT_MS = 90_000

const MISSING_URL_FILE = path.join(os.tmpdir(), 'dsh-remote-gpu-e2e-no-url-file')

const EXPECTED_TOOLS = [
  'remote_gpu_health',
  'remote_gpu_gpu_info',
  'remote_gpu_submit',
  'remote_gpu_status',
  'remote_gpu_logs',
  'remote_gpu_cancel',
  'remote_gpu_result',
  'remote_gpu_artifacts',
]

const results = []
const state = {}

function pass(id, name, evidence) {
  results.push({ id, name, status: 'PASS', evidence: String(evidence) })
  console.log(`[PASS   ] ${id} ${name} :: ${evidence}`)
}

function fail(id, name, error) {
  const evidence = `${error?.name ?? 'Error'}: ${error?.message ?? String(error)}`.slice(0, 900)
  results.push({ id, name, status: 'FAIL', evidence })
  console.log(`[FAIL   ] ${id} ${name} :: ${evidence}`)
}

async function check(id, name, fn) {
  try {
    pass(id, name, await fn())
  } catch (error) {
    fail(id, name, error)
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/** Ask the OS for a currently-free TCP port (used for the unreachable case). */
function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.once('error', reject)
    server.listen(0, HOST, () => {
      const { port } = server.address()
      server.close(() => resolve(port))
    })
  })
}

// ---------------------------------------------------------------------------------
// worker process management
// ---------------------------------------------------------------------------------

function findPython() {
  const candidates = [
    path.join(PROJECT_DIR, '.venv-e2e', 'bin', 'python'),
    path.join(PROJECT_DIR, '.venv-e2e', 'Scripts', 'python.exe'),
  ]
  for (const candidate of candidates) if (fs.existsSync(candidate)) return candidate
  return process.env.PYTHON || 'python3'
}

function writeJobScripts(root) {
  const jobs = path.join(root, 'scripts', 'jobs')
  fs.mkdirSync(jobs, { recursive: true })
  fs.writeFileSync(
    path.join(jobs, 'e2e_hello.py'),
    `import argparse, os, sys, time

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="world")
parser.add_argument("--sleep", type=float, default=0.0)
args = parser.parse_args()

print(f"plugin-e2e hello name={args.name} flag={os.environ.get('E2E_FLAG', '<unset>')}", flush=True)
for index in range(3):
    print(f"tick {index}", flush=True)
if args.sleep:
    print("sleeping", flush=True)
    time.sleep(args.sleep)
print("plugin-e2e done", flush=True)
sys.exit(0)
`,
    'utf8',
  )
}

async function workerHealthy(timeoutMs = STARTUP_TIMEOUT_MS) {
  const deadline = Date.now() + timeoutMs
  let lastError = 'not attempted'
  while (Date.now() < deadline) {
    if (state.proc && state.proc.exitCode !== null) {
      let tail = ''
      try {
        tail = fs.readFileSync(state.logPath, 'utf8').slice(-2000)
      } catch {}
      throw new Error(`worker exited early with code ${state.proc.exitCode}; log tail:\n${tail}`)
    }
    try {
      const response = await fetch(`${BASE}/health`, {
        headers: { Authorization: `Bearer ${state.token}` },
        signal: AbortSignal.timeout(2000),
      })
      if (response.ok) return await response.json()
      lastError = `HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`
    } catch (error) {
      lastError = `${error?.cause?.code ?? error?.name}: ${error?.message ?? error}`
    }
    await sleep(200)
  }
  throw new Error(`worker did not become healthy in ${timeoutMs}ms: ${lastError}`)
}

async function startWorker(root) {
  const env = {
    ...process.env,
    DSH_GPU_WORKER_TOKEN: state.token,
    DSH_GPU_WORKER_ROOT: root,
    DSH_GPU_WORKER_HOST: HOST,
    DSH_GPU_WORKER_PORT: String(PORT),
    DSH_GPU_WORKER_MAX_CONCURRENT_TASKS: '1',
    PYTHONUNBUFFERED: '1',
  }
  delete env.DSH_GPU_WORKER_DB
  delete env.PYTHONPATH
  state.logPath = path.join(root, 'worker-plugin-e2e.log')
  const logHandle = fs.openSync(state.logPath, 'a')
  state.proc = spawn(state.python, ['-u', '-m', 'worker.main'], {
    cwd: WORKER_DIR,
    env,
    stdio: ['ignore', logHandle, logHandle],
  })
  try {
    return await workerHealthy()
  } catch (error) {
    await stopWorker()
    throw error
  }
}

async function stopWorker() {
  const proc = state.proc
  if (!proc) return
  state.proc = null
  if (proc.exitCode === null) {
    proc.kill('SIGTERM')
    const deadline = Date.now() + 30_000
    while (proc.exitCode === null && Date.now() < deadline) await sleep(100)
    if (proc.exitCode === null) proc.kill('SIGKILL')
    await sleep(200)
  }
}

function makeFakeCtx(store) {
  return {
    tools: {
      register(definition) {
        store.push(definition)
        return () => {}
      },
    },
  }
}

function toolsMap(definitions) {
  return new Map(definitions.map((definition) => [definition.name, definition]))
}

async function callTool(tools, name, args = {}) {
  const tool = tools.get(name)
  assert(tool && typeof tool.execute === 'function', `tool ${name} is not registered`)
  return tool.execute(args)
}

async function waitStatus(tools, taskId, targets, timeoutMs = 120_000) {
  const deadline = Date.now() + timeoutMs
  let last
  while (Date.now() < deadline) {
    const response = await callTool(tools, 'remote_gpu_status', { task_id: taskId })
    assert(response.ok, `status failed: ${JSON.stringify(response.error)}`)
    last = response.result
    if (targets.includes(last.status)) return last
    await sleep(150)
  }
  throw new Error(`timeout waiting for ${targets.join('|')} on ${taskId}; last=${JSON.stringify(last)}`)
}

// ---------------------------------------------------------------------------------
// checks
// ---------------------------------------------------------------------------------

async function main() {
  state.token = crypto.randomBytes(24).toString('hex')
  state.python = findPython()
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'dsh-gpu-plugin-e2e-'))
  state.root = root
  writeJobScripts(root)
  process.env.DSH_GPU_WORKER_TOKEN = state.token

  const startedAt = new Date().toISOString()
  console.log(`[e2e] temp root : ${root}`)
  console.log(`[e2e] python    : ${state.python}`)
  console.log(`[e2e] worker    : ${BASE}`)
  console.log(`[e2e] token     : <redacted ${state.token.length} chars>\n`)

  try {
    const health = await startWorker(root)
    console.log(`[e2e] worker healthy (version=${health.version}, cuda_available=${health.cuda_available}); running checks\n`)

    // -- registration ---------------------------------------------------------
    const registered = []
    const ctx = makeFakeCtx(registered)
    const dispose = apply(ctx, {
      baseUrl: BASE,
      baseUrlFile: MISSING_URL_FILE,
      tokenEnv: 'DSH_GPU_WORKER_TOKEN',
      timeoutMs: 20000,
    })
    const tools = toolsMap(registered)

    await check('P01', 'apply() registers exactly the 8 underscore tool names', () => {
      assert(typeof dispose === 'function', 'apply() must return a disposer')
      const names = registered.map((definition) => definition.name)
      assert(names.length === 8, `expected 8 tools, registered ${names.length}: ${names.join(',')}`)
      assert(new Set(names).size === 8, `duplicate tool names: ${names.join(',')}`)
      for (const expected of EXPECTED_TOOLS) assert(names.includes(expected), `missing tool ${expected}: ${names.join(',')}`)
      for (const name of names) {
        assert(!name.includes('.'), `tool name must not contain a dot: ${name}`)
        assert(/^[A-Za-z0-9_-]+$/.test(name), `tool name has invalid characters: ${name}`)
      }
      const sample = registered.find((definition) => definition.name === 'remote_gpu_submit')
      assert(sample.parameters?.required?.includes('task_type'), 'submit parameters missing required task_type')
      assert(sample.parameters.properties?.task_type?.enum?.length === 3, 'submit task_type enum must have 3 values')
      for (const definition of registered) {
        assert(definition.output?.schema?.type === 'object', `${definition.name} missing output.schema`)
        assert(typeof definition.output?.render === 'function', `${definition.name} missing output.render`)
        const rendered = definition.output.render({}, { ok: true, result: { ping: 1 } })
        assert(Array.isArray(rendered) && rendered[0]?.type === 'text' && rendered[0].text.includes('ping'), `${definition.name} text render broken`)
      }
      // tool name list is exactly the documented production names
      return `names=${names.join(',')}`
    })

    // -- health / gpu ---------------------------------------------------------
    await check('P02', 'remote_gpu_health -> ok:true, status=ok', async () => {
      const response = await callTool(tools, 'remote_gpu_health')
      assert(response.ok === true, `expected ok:true, got ${JSON.stringify(response)}`)
      assert(response.result.status === 'ok', `status=${response.result.status}`)
      assert(typeof response.result.cuda_available === 'boolean', 'missing cuda_available')
      state.cudaAvailable = response.result.cuda_available
      return `status=ok version=${response.result.version} cuda_available=${response.result.cuda_available}`
    })

    await check('P03', 'remote_gpu_gpu_info -> ok:true with CUDA truth', async () => {
      const response = await callTool(tools, 'remote_gpu_gpu_info')
      assert(response.ok === true, `expected ok:true, got ${JSON.stringify(response.error)}`)
      const gpu = response.result
      assert(typeof gpu.cuda_available === 'boolean', 'missing cuda_available')
      assert(Array.isArray(gpu.devices), 'missing devices array')
      if (gpu.cuda_available) assert(gpu.device_count >= 1 && gpu.devices.length >= 1, 'CUDA true but no devices')
      else assert(gpu.device_count === 0 && gpu.devices.length === 0, 'CUDA false but devices reported')
      return `cuda_available=${gpu.cuda_available} device_count=${gpu.device_count} error=${String(gpu.error).slice(0, 100)}`
    })

    // -- submit / status / logs / result / artifacts --------------------------
    let genericTaskId = ''

    await check('P04', 'remote_gpu_submit(generic_python) -> ok:true TaskView', async () => {
      const response = await callTool(tools, 'remote_gpu_submit', {
        task_type: 'generic_python',
        config: { script: 'jobs/e2e_hello.py', args: ['--name', 'plugin'], env: { E2E_FLAG: 'yes' } },
        name: 'plugin-e2e-generic',
      })
      assert(response.ok === true, `submit failed: ${JSON.stringify(response.error)}`)
      assert(typeof response.result.task_id === 'string' && response.result.task_id.length > 0, 'missing task_id')
      assert(['PENDING', 'RUNNING'].includes(response.result.status), `unexpected initial status ${response.result.status}`)
      genericTaskId = response.result.task_id
      return `task_id=${genericTaskId} status=${response.result.status}`
    })

    await check('P05', 'remote_gpu_status polls to SUCCEEDED', async () => {
      const final = await waitStatus(tools, genericTaskId, ['SUCCEEDED', 'FAILED', 'CANCELLED'])
      assert(final.status === 'SUCCEEDED', `expected SUCCEEDED, got ${final.status}: ${JSON.stringify(final.error)}`)
      state.genericTaskId = genericTaskId
      return `task_id=${genericTaskId} status=SUCCEEDED`
    })

    await check('P06', 'remote_gpu_logs returns lines, no internal markers', async () => {
      const response = await callTool(tools, 'remote_gpu_logs', { task_id: genericTaskId, limit: 500 })
      assert(response.ok === true, `logs failed: ${JSON.stringify(response.error)}`)
      const payload = response.result
      assert(Array.isArray(payload.lines) && payload.lines.length > 0, 'no log lines')
      const text = payload.lines.map((line) => line.text).join('\n')
      assert(text.includes('plugin-e2e hello name=plugin'), `script output missing: ${text.slice(0, 400)}`)
      assert(text.includes('flag=yes'), 'E2E_FLAG env passthrough missing')
      assert(!text.includes('@@PROGRESS@@') && !text.includes('@@RESULT@@'), 'internal marker leaked')
      return `lines=${payload.lines.length} eof=${payload.eof} marker_free=True`
    })

    await check('P07', 'remote_gpu_result returns metrics + artifacts for SUCCEEDED', async () => {
      const response = await callTool(tools, 'remote_gpu_result', { task_id: genericTaskId })
      assert(response.ok === true, `result failed: ${JSON.stringify(response.error)}`)
      const payload = response.result
      assert(payload.status === 'SUCCEEDED', `status=${payload.status}`)
      assert(payload.result.metrics.exit_code === 0, `exit_code=${payload.result.metrics.exit_code}`)
      assert(Array.isArray(payload.result.artifacts), 'result.artifacts not an array')
      return `status=SUCCEEDED exit_code=0 artifacts=${payload.result.artifacts.length}`
    })

    await check('P08', 'remote_gpu_artifacts lists metrics.json with sha256', async () => {
      const response = await callTool(tools, 'remote_gpu_artifacts', { task_id: genericTaskId })
      assert(response.ok === true, `artifacts failed: ${JSON.stringify(response.error)}`)
      const artifacts = response.result.artifacts
      assert(Array.isArray(artifacts) && artifacts.length > 0, 'empty artifact list')
      const metrics = artifacts.find((entry) => entry.path === 'metrics.json')
      assert(metrics, `metrics.json missing: ${artifacts.map((entry) => entry.path).join(',')}`)
      assert(/^[0-9a-f]{64}$/.test(metrics.sha256), `bad sha256: ${metrics.sha256}`)
      assert(typeof metrics.download_url === 'string' && metrics.download_url.includes('/artifacts/'), 'missing download_url')
      return `artifacts=${artifacts.length} metrics.sha256=${metrics.sha256.slice(0, 16)}… size=${metrics.size_bytes}`
    })

    // -- idempotency ----------------------------------------------------------
    await check('P09', 'remote_gpu_submit idempotency: replay same id, conflict -> WORKER_HTTP_ERROR', async () => {
      const key = `plugin-idem-${crypto.randomBytes(4).toString('hex')}`
      const config = { script: 'jobs/e2e_hello.py', args: ['--name', 'idem'] }
      const first = await callTool(tools, 'remote_gpu_submit', { task_type: 'generic_python', config, idempotency_key: key })
      assert(first.ok === true, `first submit failed: ${JSON.stringify(first.error)}`)
      const replay = await callTool(tools, 'remote_gpu_submit', { task_type: 'generic_python', config, idempotency_key: key })
      assert(replay.ok === true, `replay failed: ${JSON.stringify(replay.error)}`)
      assert(replay.result.task_id === first.result.task_id, `replay returned a different task ${replay.result.task_id}`)
      const conflict = await callTool(tools, 'remote_gpu_submit', {
        task_type: 'generic_python',
        config: { script: 'jobs/e2e_hello.py', args: ['--name', 'other'] },
        idempotency_key: key,
      })
      assert(conflict.ok === false, 'conflicting payload must not report success')
      assert(conflict.error.code === 'WORKER_HTTP_ERROR', `expected WORKER_HTTP_ERROR, got ${conflict.error.code}`)
      assert(conflict.error.details?.workerCode === 'IDEMPOTENCY_CONFLICT', `workerCode=${conflict.error.details?.workerCode}`)
      assert(conflict.error.details?.status === 409, `http status=${conflict.error.details?.status}`)
      await waitStatus(tools, first.result.task_id, ['SUCCEEDED', 'FAILED', 'CANCELLED'])
      return `key=${key} replay_same_id=True conflict=WORKER_HTTP_ERROR(409 IDEMPOTENCY_CONFLICT)`
    })

    // -- cancel ---------------------------------------------------------------
    await check('P10', 'remote_gpu_cancel RUNNING task -> CANCELLED', async () => {
      const submitted = await callTool(tools, 'remote_gpu_submit', {
        task_type: 'generic_python',
        config: { script: 'jobs/e2e_hello.py', args: ['--sleep', '120'] },
        name: 'plugin-e2e-cancel-me',
        timeout_seconds: 300,
      })
      assert(submitted.ok === true, `submit failed: ${JSON.stringify(submitted.error)}`)
      const taskId = submitted.result.task_id
      const running = await waitStatus(tools, taskId, ['RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED'])
      assert(running.status === 'RUNNING', `expected RUNNING before cancel, got ${running.status}`)
      const cancelled = await callTool(tools, 'remote_gpu_cancel', { task_id: taskId })
      assert(cancelled.ok === true, `cancel failed: ${JSON.stringify(cancelled.error)}`)
      const final = await waitStatus(tools, taskId, ['CANCELLED', 'FAILED', 'SUCCEEDED'], 60_000)
      assert(final.status === 'CANCELLED', `expected CANCELLED, got ${final.status}`)
      return `task_id=${taskId} RUNNING -> cancel -> CANCELLED`
    })

    // -- CUDA forced ----------------------------------------------------------
    await check('P11', 'remote_gpu_submit training device=cuda -> FAILED CUDA_UNAVAILABLE', async () => {
      const submitted = await callTool(tools, 'remote_gpu_submit', {
        task_type: 'training',
        config: {
          device: 'cuda',
          epochs: 1,
          batch_size: 8,
          dataset: { dataset_id: 'synthetic', format: 'synthetic', num_samples: 40, num_features: 8, num_classes: 3 },
          model: { type: 'mlp', input_dim: 8, hidden_dims: [8], num_classes: 3 },
        },
        name: 'plugin-e2e-cuda-forced',
      })
      assert(submitted.ok === true, `submit failed: ${JSON.stringify(submitted.error)}`)
      const final = await waitStatus(tools, submitted.result.task_id, ['SUCCEEDED', 'FAILED', 'CANCELLED'], 120_000)
      assert(final.status === 'FAILED', `expected FAILED, got ${final.status}`)
      assert(final.error?.code === 'CUDA_UNAVAILABLE', `expected CUDA_UNAVAILABLE, got ${JSON.stringify(final.error).slice(0, 300)}`)
      return `task_id=${submitted.result.task_id} FAILED CUDA_UNAVAILABLE`
    })

    // -- failure isolation ----------------------------------------------------
    await check('P12', 'unreachable port -> ok:false WORKER_UNREACHABLE', async () => {
      const unreachablePort = Number(process.env.E2E_PLUGIN_UNREACHABLE_PORT) || (await freePort())
      const store = []
      apply(makeFakeCtx(store), {
        baseUrl: `http://${HOST}:${unreachablePort}`,
        baseUrlFile: MISSING_URL_FILE,
        tokenEnv: 'DSH_GPU_WORKER_TOKEN',
        timeoutMs: 3000,
      })
      const broken = toolsMap(store)
      const response = await callTool(broken, 'remote_gpu_health')
      assert(response.ok === false, `expected ok:false, got ${JSON.stringify(response)}`)
      assert(response.error.code === 'WORKER_UNREACHABLE', `expected WORKER_UNREACHABLE, got ${response.error.code}: ${response.error.message}`)
      return `http://${HOST}:${unreachablePort}/health -> ${response.error.code} (${String(response.error.details?.cause).slice(0, 120)})`
    })

    await check('P13', 'wrong token -> ok:false WORKER_HTTP_ERROR(401 UNAUTHORIZED)', async () => {
      process.env.E2E_BAD_TOKEN = `${state.token}-wrong`
      const store = []
      apply(makeFakeCtx(store), { baseUrl: BASE, baseUrlFile: MISSING_URL_FILE, tokenEnv: 'E2E_BAD_TOKEN', timeoutMs: 5000 })
      const bad = toolsMap(store)
      const response = await callTool(bad, 'remote_gpu_health')
      assert(response.ok === false, 'wrong token must not report success')
      assert(response.error.code === 'WORKER_HTTP_ERROR', `expected WORKER_HTTP_ERROR, got ${response.error.code}`)
      assert(response.error.details?.status === 401, `http status=${response.error.details?.status}`)
      assert(response.error.details?.workerCode === 'UNAUTHORIZED', `workerCode=${response.error.details?.workerCode}`)
      return `401 -> WORKER_HTTP_ERROR workerCode=UNAUTHORIZED (no false success)`
    })

    await check('P14', 'missing credential -> ok:false CONFIG_MISSING before any request', async () => {
      delete process.env.E2E_MISSING_TOKEN
      const store = []
      apply(makeFakeCtx(store), {
        baseUrl: BASE,
        baseUrlFile: MISSING_URL_FILE,
        tokenEnv: 'E2E_MISSING_TOKEN',
        tokenFile: '~/.dsh-remote-gpu-e2e-does-not-exist/token',
      })
      const missing = toolsMap(store)
      const response = await callTool(missing, 'remote_gpu_health')
      assert(response.ok === false, 'missing credential must not report success')
      assert(response.error.code === 'CONFIG_MISSING', `expected CONFIG_MISSING, got ${response.error.code}`)
      return `CONFIG_MISSING (tokenEnv unset, tokenFile absent)`
    })

    await check('P15', 'invalid local argument -> ok:false INVALID_ARGUMENT', async () => {
      const badType = await callTool(tools, 'remote_gpu_submit', { task_type: 'shell', config: {} })
      assert(badType.ok === false && badType.error.code === 'INVALID_ARGUMENT', `bad task_type -> ${JSON.stringify(badType.error)}`)
      const badLimit = await callTool(tools, 'remote_gpu_logs', { task_id: genericTaskId, limit: 9999 })
      assert(badLimit.ok === false && badLimit.error.code === 'INVALID_ARGUMENT', `bad limit -> ${JSON.stringify(badLimit.error)}`)
      const noId = await callTool(tools, 'remote_gpu_status', {})
      assert(noId.ok === false && noId.error.code === 'INVALID_ARGUMENT', `missing task_id -> ${JSON.stringify(noId.error)}`)
      return `task_type=shell / limit=9999 / missing task_id -> INVALID_ARGUMENT`
    })

    await check('P16', 'baseUrlEnv override (DSH_GPU_WORKER_URL) beats config.baseUrl', async () => {
      // Mac 跨机常用：保持 bundle 里的 baseUrl 不动，用环境变量指向 Windows IP。
      const envName = 'E2E_WORKER_URL'
      process.env[envName] = BASE
      const store = []
      apply(makeFakeCtx(store), {
        baseUrl: `http://${HOST}:1`, // deliberately wrong; env must win
        baseUrlFile: MISSING_URL_FILE,
        baseUrlEnv: envName,
        tokenEnv: 'DSH_GPU_WORKER_TOKEN',
        timeoutMs: 5000,
      })
      const viaEnv = toolsMap(store)
      const good = await callTool(viaEnv, 'remote_gpu_health')
      assert(good.ok === true, `env baseUrl should override config: ${JSON.stringify(good.error)}`)
      assert(good.result.status === 'ok', `status=${good.result.status}`)

      // The override is read once when apply() registers the tools (a DSH host
      // restart / re-apply is needed to pick up a changed env var). A second,
      // independent apply with the env var unset must fall back to config.baseUrl.
      const fallbackStore = []
      apply(makeFakeCtx(fallbackStore), {
        baseUrl: `http://${HOST}:1`,
        baseUrlFile: MISSING_URL_FILE,
        baseUrlEnv: `${envName}_UNSET`,
        tokenEnv: 'DSH_GPU_WORKER_TOKEN',
        timeoutMs: 3000,
      })
      const bad = await callTool(toolsMap(fallbackStore), 'remote_gpu_health')
      assert(bad.ok === false, 'without the env override the bad config.baseUrl must fail')
      assert(bad.error.code === 'WORKER_UNREACHABLE', `expected WORKER_UNREACHABLE, got ${bad.error.code}`)
      delete process.env[envName]
      return `${envName}=${BASE} wins over config.baseUrl; env unset -> falls back to config.baseUrl and reports WORKER_UNREACHABLE`
    })

    dispose()
  } catch (error) {
    fail('P00', 'worker startup / harness', error)
    console.error(error)
  } finally {
    await stopWorker()
    if (!process.argv.includes('--keep-temp')) {
      fs.rmSync(root, { recursive: true, force: true })
    } else {
      console.log(`[e2e] kept temp root: ${root}`)
    }
    const finishedAt = new Date().toISOString()
    const counts = results.reduce((acc, result) => {
      acc[result.status] = (acc[result.status] || 0) + 1
      return acc
    }, {})
    const summary = {
      harness: 'e2e/plugin_e2e.mjs',
      base_url: BASE,
      started_at: startedAt,
      finished_at: finishedAt,
      counts,
      results,
    }
    console.log('\n' + '='.repeat(100))
    console.log('ID'.padEnd(4) + ' ' + 'STATUS'.padEnd(8) + ' CHECK')
    console.log('-'.repeat(100))
    for (const result of results) console.log(result.id.padEnd(4) + ' ' + result.status.padEnd(8) + ' ' + result.name)
    console.log('-'.repeat(100))
    console.log(`TOTAL: ${counts.PASS || 0} PASS, ${counts.FAIL || 0} FAIL`)
    console.log('='.repeat(100))
    console.log(JSON.stringify(summary, null, 2))
    if (process.env.E2E_PLUGIN_JSON) {
      fs.writeFileSync(process.env.E2E_PLUGIN_JSON, JSON.stringify(summary, null, 2), 'utf8')
    }
    process.exit(counts.FAIL ? 1 : 0)
  }
}

main()
