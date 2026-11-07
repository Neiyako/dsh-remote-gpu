#Requires -Version 5.1
<#
.SYNOPSIS
    One-click installer and launcher for the DSH remote GPU Worker on Windows.

.DESCRIPTION
    Downloads the public repository (or uses the current checkout), creates a
    Python venv, installs FastAPI + CUDA PyTorch (cu128 for RTX 50-series),
    generates/uses a bearer token, writes .env, starts the worker in the
    background, optionally opens the Windows firewall, and prints the exact
    Mac-side connection snippet.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/Neiyako/dsh-remote-gpu/main/deploy.ps1 | iex"

.EXAMPLE
    .\deploy.ps1 -Token 'paste-existing-token' -OpenFirewall:$false

.EXAMPLE
    .\deploy.ps1 -InstallTorch:$false -StartWorker:$false
#>
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:USERPROFILE 'dsh-remote-gpu'),

    # Optional: reuse an existing token (for example the one already stored on
    # the Mac in ~/.dsh-remote-gpu/token). Omit to generate a new one.
    [string]$Token = '',

    [string]$RepoUrl = 'https://github.com/Neiyako/dsh-remote-gpu',
    [string]$Branch = 'main',

    # RTX 5060 / Blackwell needs cu128. Pass -InstallTorch:$false for CPU-only.
    [bool]$InstallTorch = $true,
    [bool]$StartWorker = $true,
    [bool]$OpenFirewall = $true,

    # Overwrite InstallDir when it already exists (upgrade / reinstall).
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function New-RandomToken {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($bytes)
    } finally {
        $rng.Dispose()
    }
    return [Convert]::ToBase64String($bytes).Replace('+', '-').Replace('/', '_').TrimEnd('=')
}

function Get-ExistingToken {
    param([string]$EnvFile)
    if (-not (Test-Path $EnvFile)) { return '' }
    foreach ($line in Get-Content $EnvFile) {
        if ($line -match '^\s*DSH_GPU_WORKER_TOKEN\s*=\s*(.+?)\s*$') {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return ''
}

function Get-LanIPv4 {
    try {
        $config = Get-NetIPConfiguration |
            Where-Object { $_.IPv4DefaultGateway -ne $null -and $_.IPv4Address -ne $null } |
            Select-Object -First 1
        if ($config -ne $null -and $config.IPv4Address -ne $null) {
            return ($config.IPv4Address | Select-Object -First 1).IPAddress
        }
    } catch {
        # fall through
    }
    try {
        return (Get-NetIPAddress -AddressFamily IPv4 |
            Where-Object {
                $_.IPAddress -notlike '127.*' -and
                $_.IPAddress -notlike '169.254.*' -and
                $_.PrefixOrigin -ne 'WellKnown'
            } |
            Select-Object -First 1 -ExpandProperty IPAddress)
    } catch {
        return '127.0.0.1'
    }
}

function Wait-Health {
    param([string]$Url, [string]$TokenValue, [int]$Seconds = 90)
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $headers = @{ Authorization = "Bearer $TokenValue" }
            $health = Invoke-RestMethod -Uri "$Url/health" -Headers $headers -TimeoutSec 2
            return $health
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $null
}

Write-Host "DSH Remote GPU Worker - Windows one-click deploy" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 1. Resolve repository (current checkout or download the public zip)
# ---------------------------------------------------------------------------
Write-Step '1/6 Resolve project files'
$repoRoot = $null
if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot 'dsh-gpu-worker\worker\main.py'))) {
    $repoRoot = $PSScriptRoot
} elseif (Test-Path (Join-Path (Get-Location) 'dsh-gpu-worker\worker\main.py')) {
    $repoRoot = (Get-Location).Path
}

if ($repoRoot -ne $null) {
    Write-Host "Using local checkout: $repoRoot"
    $workerDir = Join-Path $repoRoot 'dsh-gpu-worker'
    $projectDir = $repoRoot
} else {
    if ((Test-Path $InstallDir) -and (-not $Force)) {
        throw "InstallDir already exists: $InstallDir. Re-run with -Force to update code (data/ and .env are preserved), or choose another -InstallDir."
    }
    $repoPath = ([Uri]$RepoUrl).AbsolutePath.Trim('/')
    $zipUrl = "https://codeload.github.com/$repoPath/zip/refs/heads/$Branch"
    $tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("dsh-gpu-worker-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmpRoot -Force | Out-Null
    $zipPath = Join-Path $tmpRoot 'repo.zip'
    Write-Host "Downloading $zipUrl"
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
    Expand-Archive -LiteralPath $zipPath -DestinationPath $tmpRoot -Force
    $extracted = Get-ChildItem -LiteralPath $tmpRoot -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName 'dsh-gpu-worker\worker\main.py') } |
        Select-Object -First 1
    if ($extracted -eq $null) {
        throw "Downloaded archive does not contain dsh-gpu-worker/worker/main.py"
    }
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    # Overwrite code only; data/, logs/ and .env keep working across re-runs.
    Copy-Item -Path (Join-Path $extracted.FullName '*') -Destination $InstallDir -Recurse -Force
    Remove-Item -LiteralPath $tmpRoot -Recurse -Force
    $projectDir = $InstallDir
    $workerDir = Join-Path $InstallDir 'dsh-gpu-worker'
    Write-Host "Installed to: $InstallDir"
}

$dataDir = Join-Path $projectDir 'data'
$jobsDir = Join-Path $dataDir 'scripts\jobs'
$datasetsDir = Join-Path $dataDir 'datasets'
New-Item -ItemType Directory -Path $jobsDir -Force | Out-Null
New-Item -ItemType Directory -Path $datasetsDir -Force | Out-Null
Copy-Item -Path (Join-Path $workerDir 'scripts\jobs\hello.py') -Destination $jobsDir -Force

# ---------------------------------------------------------------------------
# 2. Token and .env
# ---------------------------------------------------------------------------
Write-Step '2/6 Configure token'
$envFile = Join-Path $workerDir '.env'
if ([string]::IsNullOrWhiteSpace($Token)) {
    $Token = Get-ExistingToken -EnvFile $envFile
}
if ([string]::IsNullOrWhiteSpace($Token)) {
    $Token = New-RandomToken
    Write-Host "Generated a new worker token."
} else {
    Write-Host "Using the provided/existing worker token."
}

$envLines = @(
    "DSH_GPU_WORKER_TOKEN=$Token",
    "DSH_GPU_WORKER_HOST=0.0.0.0",
    "DSH_GPU_WORKER_PORT=8765",
    "DSH_GPU_WORKER_ROOT=$dataDir",
    "DSH_GPU_WORKER_MAX_CONCURRENT_TASKS=1",
    "DSH_GPU_WORKER_ALLOW_GENERIC_PYTHON=true"
)
Set-Content -Path $envFile -Value $envLines -Encoding ASCII

# `python -m worker` reads real environment variables (it does not parse .env),
# so export the same values into this process before starting the child.
$env:DSH_GPU_WORKER_TOKEN = $Token
$env:DSH_GPU_WORKER_HOST = '0.0.0.0'
$env:DSH_GPU_WORKER_PORT = '8765'
$env:DSH_GPU_WORKER_ROOT = $dataDir
$env:DSH_GPU_WORKER_MAX_CONCURRENT_TASKS = '1'
$env:DSH_GPU_WORKER_ALLOW_GENERIC_PYTHON = 'true'
try {
    & icacls $envFile /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null
} catch {
    Write-Host "WARN: could not restrict .env ACL; keep it in a trusted user profile." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 3. Python environment
# ---------------------------------------------------------------------------
Write-Step '3/6 Install Python dependencies'
$python = Join-Path $workerDir '.venv\Scripts\python.exe'
$installArgs = @('-ExecutionPolicy', 'Bypass', '-File', (Join-Path $workerDir 'scripts\install.ps1'))
if ($InstallTorch) { $installArgs += '-WithTorch' }
& powershell @installArgs
if ($LASTEXITCODE -ne 0) { throw "install.ps1 failed with exit code $LASTEXITCODE" }
if (-not (Test-Path $python)) { throw "Python venv not found at $python" }

# ---------------------------------------------------------------------------
# 4. Start worker
# ---------------------------------------------------------------------------
Write-Step '4/6 Start worker'
$started = $false
$health = $null
$port = 8765
$listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener -ne $null) {
    Write-Host "Port $port is already in use (PID $($listener.OwningProcess)); probing it instead of starting a second worker."
    $health = Wait-Health -Url "http://127.0.0.1:$port" -TokenValue $Token -Seconds 5
} elseif ($StartWorker) {
    $logDir = Join-Path $projectDir 'logs'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $outLog = Join-Path $logDir 'worker.out.log'
    $errLog = Join-Path $logDir 'worker.err.log'
    $process = Start-Process -FilePath $python -ArgumentList @('-m', 'worker') `
        -WorkingDirectory $workerDir -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    Set-Content -Path (Join-Path $workerDir 'worker.pid') -Value $process.Id -Encoding ASCII
    Write-Host "Worker started, PID $($process.Id)"
    Write-Host "Logs: $outLog / $errLog"
    $started = $true
    $health = Wait-Health -Url "http://127.0.0.1:$port" -TokenValue $Token -Seconds 120
    if ($health -eq $null) {
        Write-Host "Worker did not become healthy. Last error log:" -ForegroundColor Red
        if (Test-Path $errLog) { Get-Content $errLog -Tail 30 }
        throw "Worker health check failed"
    }
} else {
    Write-Host "Skipped starting the worker (-StartWorker:`$false)."
}

# ---------------------------------------------------------------------------
# 5. Firewall
# ---------------------------------------------------------------------------
Write-Step '5/6 Firewall'
if ($OpenFirewall) {
    if (Test-Admin) {
        try {
            $existing = Get-NetFirewallRule -DisplayName 'DSH GPU Worker 8765' -ErrorAction SilentlyContinue
            if ($existing -eq $null) {
                New-NetFirewallRule -DisplayName 'DSH GPU Worker 8765' -Direction Inbound `
                    -Protocol TCP -LocalPort 8765 -Action Allow -Profile Private | Out-Null
                Write-Host "Added inbound firewall rule for TCP 8765 on the Private profile."
            } else {
                Write-Host "Firewall rule already exists."
            }
        } catch {
            Write-Host "WARN: failed to add firewall rule: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Not running as Administrator; firewall rule was not added." -ForegroundColor Yellow
        Write-Host 'Run this in an elevated PowerShell if the Mac cannot connect:'
        Write-Host '  New-NetFirewallRule -DisplayName "DSH GPU Worker 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow -Profile Private'
    }
} else {
    Write-Host "Skipped firewall change (-OpenFirewall:`$false)."
}

# ---------------------------------------------------------------------------
# 6. Connection info
# ---------------------------------------------------------------------------
Write-Step '6/6 Connection info'
$lanIp = Get-LanIPv4
$workerUrl = "http://$lanIp`:8765"
Write-Host ""
Write-Host "Worker URL : $workerUrl" -ForegroundColor Green
Write-Host "Worker token: $Token" -ForegroundColor Green
if ($health -ne $null) {
    Write-Host "Health     : status=$($health.status) cuda_available=$($health.cuda_available) torch_installed=$($health.torch_installed)"
}
Write-Host ""
Write-Host "On the Mac (DSH host), run:" -ForegroundColor Cyan
Write-Host "  export DSH_GPU_WORKER_URL='$workerUrl'"
Write-Host "  mkdir -p ~/.dsh-remote-gpu && printf '%s\n' '$Token' > ~/.dsh-remote-gpu/token && chmod 600 ~/.dsh-remote-gpu/token"
Write-Host ""
Write-Host "Then restart/reload the DSH host so the plugin re-reads baseUrlEnv."
Write-Host ""
Write-Host "GPU check on Windows:" -ForegroundColor Cyan
Write-Host "  $python -c `"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')`""
Write-Host "  curl.exe -H `"Authorization: Bearer $Token`" http://127.0.0.1:8765/gpu"
Write-Host ""
Write-Host "If the Mac still cannot connect, allow Python through Windows Defender Firewall for the Private network." -ForegroundColor Yellow
