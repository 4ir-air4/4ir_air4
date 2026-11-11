# Copies the project to Google Drive for Desktop for colab_run.ipynb. Never deletes anything at the target.
# Only listed files go up, so .env and .venv stay on this PC. /XO keeps newer results already on Drive.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File ldh/scripts/upload_to_drive.ps1 "G:\My Drive\4ir\ldh" [extra folder ...]
param(
    [Parameter(Mandatory = $true)][string]$Target,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra = @()
)
# Temporary split (2026-11-05): home trains on Colab from Drive, this PC only preprocesses. Delete this line when the user asks to merge.
throw 'Upload is blocked while training (Colab) and preprocessing (this PC) are split. See WORK_LOG.md.'
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)  # project root (ldh/scripts -> root)
if (-not (Test-Path (Split-Path $Target -Parent))) { throw "Drive folder not found: $(Split-Path $Target -Parent)" }

$failed = $false
robocopy . $Target *.py *.md requirements.txt /XO /R:2 /W:5 /NP /NDL /NFL
if ($LASTEXITCODE -ge 8) { $failed = $true }
foreach ($dir in @('ldh', 'models', 'data') + $Extra) {
    Write-Host "=== $dir"
    # Keeps data before START_YEAR (2020) and unused folders off Drive. They were removed there and stay on this PC.
    robocopy $dir (Join-Path $Target $dir) /E /XO /XD __pycache__ bts_zip opensky_full pilot recent_2023_2025 /XF *.tmp *.copying ontime_19* ontime_200* ontime_201* flightlist_2019* Pre2008.zip PRE1982.zip ntsb_Pre2008.parquet /R:2 /W:5 /NP /NDL /NFL
    if ($LASTEXITCODE -ge 8) { $failed = $true }  # robocopy: 0-7 is success, 8 or more is failure
}
if ($failed) { Write-Host 'FAILED: see the errors above, then run again.' -ForegroundColor Red; exit 1 }
Write-Host 'Done. Wait until Google Drive finishes syncing before opening Colab.' -ForegroundColor Green
exit 0
