#Requires -Version 5.1
<#
.SYNOPSIS
    Install the DSH remote GPU worker on Windows.

.DESCRIPTION
    Creates .venv, installs runtime dependencies, copies .env.example to .env
    when no .env exists, and prints a CUDA/Torch diagnostic.

.PARAMETER WithTorch
    Also install PyTorch (CUDA 12.8 wheel, required for RTX 50-series / Blackwell) into the venv.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1 -WithTorch
#>
[CmdletBinding()]
param(
    [switch]$WithTorch
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "== dsh-gpu-worker install ==" -ForegroundColor Cyan
Write-Host "project root: $ProjectRoot"

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "creating virtual environment .venv ..."
    py -3 -m venv .venv
    if (-not (Test-Path $python)) {
        throw "Failed to create .venv. Install Python 3.10+ and ensure the 'py' launcher is available."
    }
}

& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
& $python -m pip install -r (Join-Path $ProjectRoot "requirements-dev.txt")

if ($WithTorch) {
    Write-Host "installing PyTorch (CUDA 12.8 / cu128, required for RTX 50-series) ..." -ForegroundColor Yellow
    & $python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
}

if (-not (Test-Path (Join-Path $ProjectRoot ".env"))) {
    Copy-Item (Join-Path $ProjectRoot ".env.example") (Join-Path $ProjectRoot ".env")
    Write-Host "created .env from .env.example — EDIT IT and set DSH_GPU_WORKER_TOKEN" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "== python / CUDA diagnostic ==" -ForegroundColor Cyan
& $python -c @"
import sys
print('python', sys.version.split()[0], sys.executable)
try:
    import torch
    print('torch', torch.__version__)
    print('cuda_available', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('device', torch.cuda.get_device_name(0))
        print('capability', torch.cuda.get_device_capability(0))
except Exception as exc:
    print('torch NOT available:', exc)
"@
Write-Host ""
Write-Host "Next: edit .env, then run scripts\start.ps1" -ForegroundColor Green
