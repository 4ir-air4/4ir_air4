# Shows training progress live. Read-only: closing this window does NOT stop training.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File watch_training.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$host.UI.RawUI.WindowTitle = 'Training monitor'
try { $host.UI.RawUI.BufferSize = New-Object Management.Automation.Host.Size(160, 5000) } catch {}
Set-Location $PSScriptRoot
Write-Host '=== Training monitor: closing this window does NOT stop training ===' -ForegroundColor Cyan
if (-not (Test-Path run_models.log)) { New-Item -ItemType File run_models.log | Out-Null }
Get-Content run_models.log -Tail 40 -Wait -Encoding UTF8 | ForEach-Object {
    if ($_ -match '^\d\d-\d\d .*(===|---)') { Write-Host $_ -ForegroundColor Yellow }
    elseif ($_ -match 'Traceback|Error|실패') { Write-Host $_ -ForegroundColor Red }
    elseif ($_ -match '완료') { Write-Host $_ -ForegroundColor Green }
    else { Write-Host $_ }
}
