$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    throw "Run .\setup.ps1 first."
}

& ".\.venv\Scripts\python.exe" ".\start.py"
