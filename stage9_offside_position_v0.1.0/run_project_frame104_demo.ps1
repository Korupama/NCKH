param(
    [string]$Video = "",
    [string]$Image = "",
    [int]$Port = 8107
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonPath)) { $PythonPath = "python" }
$ArgsList = @((Join-Path $PSScriptRoot "web_demo.py"), "--project", "--port", "$Port")
if ($Video) { $ArgsList += @("--video", $Video) }
if ($Image) { $ArgsList += @("--image", $Image) }
Write-Host "Stage 1-7 explorer: http://127.0.0.1:$Port"
& $PythonPath @ArgsList
