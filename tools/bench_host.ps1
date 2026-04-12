param(
    [string]$Exe = "C:\Users\legom\HyperTensor\build_host\hypertensor.exe",
    [string]$Model = "C:\Users\legom\TensorOS\models\google_gemma-4-E2B-it-Q4_0.gguf",
    [string]$Cases = "C:\Users\legom\TensorOS\tests\runtime\bench_cases.txt",
    [int]$MaxTokens = 64,
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

function Parse-Metrics([string]$text) {
    $m = [regex]::Match($text, "\[(\d+) tok,\s*(\d+) ms/tok,\s*prefill\s*(\d+) ms,\s*(\d+) cpus\]")
    if (-not $m.Success) {
        return $null
    }
    return [pscustomobject]@{
        Tokens = [int]$m.Groups[1].Value
        MsPerTok = [int]$m.Groups[2].Value
        PrefillMs = [int]$m.Groups[3].Value
        Cpus = [int]$m.Groups[4].Value
        TokPerSec = if ([int]$m.Groups[2].Value -gt 0) { [math]::Round(1000.0 / [int]$m.Groups[2].Value, 2) } else { 0.0 }
    }
}

if (-not (Test-Path $Exe)) {
    throw "Executable not found: $Exe"
}
if (-not (Test-Path $Model)) {
    throw "Model not found: $Model"
}
if (-not (Test-Path $Cases)) {
    throw "Cases file not found: $Cases"
}

$rows = Get-Content $Cases | Where-Object { $_ -and -not $_.StartsWith("#") }
if ($rows.Count -eq 0) {
    throw "No benchmark rows in: $Cases"
}

$results = @()
$failed = 0

Write-Host "Running host benchmark..." -ForegroundColor Cyan
Write-Host "Exe:   $Exe"
Write-Host "Model: $Model"
Write-Host "Cases: $Cases"
Write-Host ""

foreach ($line in $rows) {
    $parts = $line.Split("|", 2)
    $prompt = $parts[0].Trim()
    $mustContain = if ($parts.Count -gt 1) { $parts[1].Trim() } else { "" }

    $cmd = "`"$Exe`" `"$Model`" -p `"$prompt`" -n $MaxTokens"
    $raw = cmd /c $cmd 2>&1 | Out-String

    $metrics = Parse-Metrics $raw
    $ok = $true

    if ($mustContain -ne "") {
        if ($raw.ToLower().Contains($mustContain.ToLower()) -eq $false) {
            $ok = $false
        }
    }

    if (-not $metrics) {
        $ok = $false
    }

    if (-not $ok) { $failed++ }

    $results += [pscustomobject]@{
        Prompt = $prompt
        Expect = $mustContain
        Pass = $ok
        Tokens = if ($metrics) { $metrics.Tokens } else { -1 }
        MsPerTok = if ($metrics) { $metrics.MsPerTok } else { -1 }
        TokPerSec = if ($metrics) { $metrics.TokPerSec } else { 0.0 }
        PrefillMs = if ($metrics) { $metrics.PrefillMs } else { -1 }
        Cpus = if ($metrics) { $metrics.Cpus } else { -1 }
    }
}

$results | Format-Table -AutoSize

$valid = $results | Where-Object { $_.MsPerTok -gt 0 }
if ($valid.Count -gt 0) {
    $avgMs = [math]::Round((($valid | Measure-Object -Property MsPerTok -Average).Average), 2)
    $avgTps = [math]::Round((($valid | Measure-Object -Property TokPerSec -Average).Average), 2)
    $avgPref = [math]::Round((($valid | Measure-Object -Property PrefillMs -Average).Average), 2)

    Write-Host ""
    Write-Host "Summary:" -ForegroundColor Yellow
    Write-Host "  Cases:        $($results.Count)"
    Write-Host "  Passed:       $($results.Count - $failed)"
    Write-Host "  Failed:       $failed"
    Write-Host "  Avg ms/tok:   $avgMs"
    Write-Host "  Avg tok/s:    $avgTps"
    Write-Host "  Avg prefill:  $avgPref ms"
}

if ($Strict -and $failed -gt 0) {
    exit 1
}
exit 0
