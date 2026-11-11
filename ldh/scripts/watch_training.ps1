# Shows training progress live. Read-only: closing this window does NOT stop training.
# Usage (any folder): powershell -NoProfile -ExecutionPolicy Bypass -File D:\LDH\ML\4ir\ldh\scripts\watch_training.ps1 [log name]
#   No log name: watches the most recently written log in ldh\logs. Example log name: retrain_big_lstm.log
param([string]$Log = '')
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$host.UI.RawUI.WindowTitle = 'Training monitor'
try { $host.UI.RawUI.BufferSize = New-Object Management.Automation.Host.Size(160, 5000) } catch {}
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)  # project root (ldh/scripts -> root)
if (-not $Log) {
    $latest = Get-ChildItem ldh/logs -Filter *.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $Log = if ($latest) { $latest.Name } else { 'run_models.log' }
}
$path = Join-Path 'ldh/logs' (Split-Path $Log -Leaf)
if (-not (Test-Path $path)) { throw "Log not found: $path (logs are in ldh\logs)" }
Write-Host "=== Training monitor: $path (closing this window does NOT stop training) ===" -ForegroundColor Cyan
Get-Content $path -Tail 40 -Wait -Encoding UTF8 | ForEach-Object {
    if ($_ -match '^\d\d-\d\d .*(===|---)') { Write-Host $_ -ForegroundColor Yellow }
    elseif ($_ -match 'Traceback|Error|실패') { Write-Host $_ -ForegroundColor Red }
    elseif ($_ -match '완료') { Write-Host $_ -ForegroundColor Green }
    else { Write-Host $_ }
}
