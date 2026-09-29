# Start the ARDY model server from ARDY's own venv.
#   scripts\run_ardy_server.ps1 [-Ardy <ARDY clone>] [-Port 8002]
# Needs the Kimodo text-encoder container up:
#   docker compose -f docker-compose.bridge.yaml up text-encoder -d
param(
    [string]$Ardy = "$env:USERPROFILE\Documents\GitHub\ardy",
    [int]$Port = 8002
)
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $Ardy ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "No ARDY venv at $python" }
# before ErrorActionPreference Stop: Windows PowerShell 5.1 would turn the
# import error on stderr into a terminating error and hide the message below
& $python -c "import fastapi, uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "FastAPI and uvicorn are missing from ARDY's venv. Install them with: uv pip install --python `"$python`" fastapi uvicorn requests"
}
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$root\server;$root\houdini\python"
$env:TEXT_ENCODER_MODE = "api"
if (-not $env:TEXT_ENCODER_URL) { $env:TEXT_ENCODER_URL = "http://127.0.0.1:9550/" }
& $python -m uvicorn ardy_backend:make_app --factory --host 127.0.0.1 --port $Port
