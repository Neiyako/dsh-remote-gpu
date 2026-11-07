#!/usr/bin/env bash
# Start the DSH remote GPU worker (macOS / Linux dev helper).
# Loads KEY=VALUE pairs from .env, then runs `python -m worker`.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${DSH_GPU_WORKER_TOKEN:-}" ]]; then
  echo "DSH_GPU_WORKER_TOKEN is not set. Copy .env.example to .env and set a token." >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi

exec "$PYTHON" -m worker
