param(
    [string]$Out = "tensorchat_native.exe"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$zig = "zig"
$src = "tensorchat_native.c"

& $zig cc -O2 -municode $src -o $Out -luser32 -lgdi32 -lcomdlg32 -lcomctl32 -lwinhttp
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "Built $Out" -ForegroundColor Green
