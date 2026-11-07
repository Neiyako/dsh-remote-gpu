#Requires -Version 5.1
<#
.SYNOPSIS
    Start the DSH remote GPU worker on Windows.

.DESCRIPTION
    Loads KEY=VALUE pairs from .env into the current process environment, then
    runs ``<venv python> -m worker``.  DSH_GPU_WORKER_TOKEN is mandatory.

    To run it at boot use the Windows Task Scheduler (recommended) or NSSM:
      schtasks /Create /TN "DSH GPU Worker" /SC ONSTART /RU SYSTEM ^
        /TR "powershell -ExecutionPolicy Bypass -File C:\dsh-gpu-worker\scripts\start.ps1"
    See README.md for the NSSM variant.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$envFile = Join-Path $ProjectRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        $pair = $line -split "=", 2
        if ($pair.Count -ne 2) { return }
        $name = $pair[0].Trim()
        $value = $pair[1].Trim().Trim('"').Trim("'")
        if ($name -ne "") { Set-Item -Path "Env:$name" -Value $value }
    }
}

if (-not $env:DSH_GPU_WORKER_TOKEN) {
    throw "DSH_GPU_WORKER_TOKEN is not set. Edit .env and set a long random token."
}

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host ".venv not found, falling back to system python" -ForegroundColor Yellow
    $python = "python"
}

Write-Host "starting dsh-gpu-worker on $($env:DSH_GPU_WORKER_HOST):$($env:DSH_GPU_WORKER_PORT)" -ForegroundColor Cyan
& $python -m worker
exit $LASTEXITCODE
