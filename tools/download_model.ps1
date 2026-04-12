# TensorOS Model Download Script
# Downloads quantized GGUF models from HuggingFace for LLM inference
#
# Usage:
#   .\tools\download_model.ps1 -Model qwen2.5-0.5b
#   .\tools\download_model.ps1 -Ref https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF
#   .\tools\download_model.ps1 -Ref https://huggingface.co/<org>/<repo>/resolve/main/model.gguf
#   .\tools\download_model.ps1 -Ref ollama:llama3.1:8b
#   .\tools\download_model.ps1 -Ref https://ollama.com/library/llama3.1
#   .\tools\download_model.ps1 -List

param(
    [string]$Model,
    [string]$Ref,
    [string]$Url,
    [string]$Output,
    [switch]$List,
    [switch]$AllowSafeTensors
)

$ModelsDir = Join-Path $PSScriptRoot "..\models"
if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory $ModelsDir | Out-Null
}

function Get-RepoPathFromHfUrl([string]$Text) {
    if (-not $Text) { return $null }
    $m = [regex]::Match($Text, "https?://huggingface\.co/([^/]+/[^/?#]+)")
    if ($m.Success) { return $m.Groups[1].Value }
    return $null
}

function Resolve-HfDownload([string]$InputRef, [bool]$AllowSafe) {
    # Direct resolve/download URL
    if ($InputRef -match "https?://huggingface\.co/.+/(resolve|blob)/.+") {
        $outName = ($InputRef -split "/")[-1]
        return @{ Url = $InputRef -replace "/blob/", "/resolve/"; File = $outName; Desc = "HuggingFace direct file" }
    }

    # Repo URL or repo id
    $repo = $InputRef
    if ($InputRef -match "https?://") {
        $repo = Get-RepoPathFromHfUrl $InputRef
    }
    if (-not $repo) { return $null }

    $api = "https://huggingface.co/api/models/$repo"
    Write-Host "Resolving Hugging Face repo: $repo" -ForegroundColor Cyan
    $meta = Invoke-RestMethod -Uri $api -Method Get
    if (-not $meta.siblings) {
        throw "No files found in Hugging Face repo metadata: $repo"
    }

    $files = @($meta.siblings | ForEach-Object { $_.rfilename })
    $gguf = @($files | Where-Object { $_ -match "\\.gguf$" })
    $safe = @($files | Where-Object { $_ -match "\\.safetensors$" })

    $selected = $null
    if ($gguf.Count -gt 0) {
        # Prefer strong defaults for CPU runtime
        $pref = @("Q4_K_M", "Q4_0", "Q5_K_M", "Q8_0")
        foreach ($p in $pref) {
            $hit = $gguf | Where-Object { $_ -imatch $p } | Select-Object -First 1
            if ($hit) { $selected = $hit; break }
        }
        if (-not $selected) { $selected = $gguf | Sort-Object Length | Select-Object -First 1 }
    } elseif ($AllowSafe -and $safe.Count -gt 0) {
        $selected = $safe | Sort-Object Length | Select-Object -First 1
    }

    if (-not $selected) {
        $msg = "No GGUF file found in repo '$repo'."
        if (-not $AllowSafe) { $msg += " Re-run with -AllowSafeTensors to permit .safetensors download." }
        throw $msg
    }

    $dl = "https://huggingface.co/$repo/resolve/main/$selected"
    return @{ Url = $dl; File = ($selected -split "/")[-1]; Desc = "HuggingFace auto-resolved ($repo)" }
}

function Resolve-OllamaTag([string]$InputRef) {
    if (-not $InputRef) { return $null }
    if ($InputRef -match "^ollama:(.+)$") {
        return $Matches[1]
    }
    if ($InputRef -match "https?://ollama\.com/library/([^/?#]+)") {
        return $Matches[1]
    }
    if ($InputRef -match "^[a-z0-9._-]+(:[a-z0-9._-]+)?$") {
        # Accept direct tag form like llama3.1:8b
        return $InputRef
    }
    return $null
}

function Download-ModelFile([string]$SourceUrl, [string]$TargetPath) {
    Write-Host "Downloading: $SourceUrl" -ForegroundColor Cyan
    Write-Host "Target: $TargetPath" -ForegroundColor DarkGray
    try {
        Invoke-WebRequest -Uri $SourceUrl -OutFile $TargetPath -UseBasicParsing
    } catch {
        # Fallback for systems where PowerShell's curl alias causes issues
        & curl.exe -L -o $TargetPath $SourceUrl
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed via Invoke-WebRequest and curl.exe"
        }
    }
    $sz = (Get-Item $TargetPath).Length
    Write-Host "Done: $([math]::Round($sz / 1MB)) MB" -ForegroundColor Green
}

function Import-OllamaModel([string]$Tag) {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        throw "Ollama CLI not found in PATH. Install Ollama or pass a Hugging Face GGUF URL/repo."
    }
    Write-Host "Pulling Ollama model: $Tag" -ForegroundColor Cyan
    & ollama pull $Tag
    if ($LASTEXITCODE -ne 0) {
        throw "ollama pull failed for tag '$Tag'"
    }
    Write-Host "Pulled into Ollama local store." -ForegroundColor Green
    Write-Host "Note: exporting Ollama blobs to standalone GGUF is model-dependent and not universal yet." -ForegroundColor Yellow
}

# Model registry: name -> (URL, filename, description, size)
$Registry = @{
    "qwen2.5-0.5b" = @{
        Url  = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf"
        File = "qwen2.5-0.5b-instruct-q4_0.gguf"
        Desc = "Qwen2.5-0.5B-Instruct Q4_0 - Best math for size, 494M params"
        Size = "352 MB"
    }
    "qwen2.5-0.5b-q8" = @{
        Url  = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"
        File = "qwen2.5-0.5b-instruct-q8_0.gguf"
        Desc = "Qwen2.5-0.5B-Instruct Q8_0 - Higher quality, slower"
        Size = "531 MB"
    }
    "smollm2-135m" = @{
        Url  = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"
        File = "smollm2-135m-instruct-q8_0.gguf"
        Desc = "SmolLM2-135M-Instruct Q8_0 - Fastest, tiny model"
        Size = "145 MB"
    }
    "smollm2-360m" = @{
        Url  = "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"
        File = "smollm2-360m-instruct-q4km.gguf"
        Desc = "SmolLM2-360M-Instruct Q4_K_M - Good balance"
        Size = "230 MB"
    }
    "tinyllama" = @{
        Url  = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
        File = "tinyllama-1.1b-chat-q4_0.gguf"
        Desc = "TinyLlama-1.1B-Chat Q4_0 - Classic small LLM"
        Size = "600 MB"
    }
    "gemma-2-2b" = @{
        Url  = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_0.gguf"
        File = "gemma-2-2b-it-q4_0.gguf"
        Desc = "Gemma-2-2B-IT Q4_0 - Google's model, strong math"
        Size = "1.4 GB"
    }
}

if ($List) {
    Write-Host "`nAvailable Models for TensorOS:" -ForegroundColor Cyan
    Write-Host ("=" * 65)
    foreach ($key in $Registry.Keys | Sort-Object) {
        $m = $Registry[$key]
        Write-Host "  $key" -ForegroundColor Yellow -NoNewline
        Write-Host "  ($($m.Size))" -ForegroundColor DarkGray
        Write-Host "    $($m.Desc)"
    }
    Write-Host "`nUsage: .\tools\download_model.ps1 -Model <name>" -ForegroundColor Green
    Write-Host "Models are saved to: models\*.gguf" -ForegroundColor DarkGray
    Write-Host "The build script auto-detects them when using -Run`n" -ForegroundColor DarkGray
    exit 0
}

if ($Url -and -not $Ref) {
    $Ref = $Url
}

if ($Ref) {
    try {
        $hf = Resolve-HfDownload -InputRef $Ref -AllowSafe:$AllowSafeTensors
        if ($hf) {
            if (-not $Output) {
                $Output = Join-Path $ModelsDir $hf.File
            }
            Download-ModelFile -SourceUrl $hf.Url -TargetPath $Output
            exit 0
        }

        $tag = Resolve-OllamaTag $Ref
        if ($tag) {
            Import-OllamaModel -Tag $tag
            exit 0
        }

        throw "Unsupported reference. Pass Hugging Face URL/repo, Ollama tag, or registry model name."
    } catch {
        Write-Host "Reference resolution failed: $_" -ForegroundColor Red
        exit 1
    }
}

if (-not $Model) {
    Write-Host "Usage: .\tools\download_model.ps1 -Model <name>" -ForegroundColor Yellow
    Write-Host "       .\tools\download_model.ps1 -Ref <url|repo|ollama-tag>" -ForegroundColor Yellow
    Write-Host "       .\tools\download_model.ps1 -List" -ForegroundColor Yellow
    exit 1
}

$key = $Model.ToLower()
if (-not $Registry.ContainsKey($key)) {
    # Fall back to universal reference mode when model is not in the registry
    try {
        $hf = Resolve-HfDownload -InputRef $Model -AllowSafe:$AllowSafeTensors
        if ($hf) {
            if (-not $Output) {
                $Output = Join-Path $ModelsDir $hf.File
            }
            Download-ModelFile -SourceUrl $hf.Url -TargetPath $Output
            exit 0
        }

        $tag = Resolve-OllamaTag $Model
        if ($tag) {
            Import-OllamaModel -Tag $tag
            exit 0
        }

        Write-Host "Unknown model: $Model" -ForegroundColor Red
        Write-Host "Use -List, or pass -Ref <hf-url|hf-repo|ollama-tag>" -ForegroundColor Yellow
        exit 1
    } catch {
        Write-Host "Model resolution failed: $_" -ForegroundColor Red
        exit 1
    }
}

$info = $Registry[$key]
$target = Join-Path $ModelsDir $info.File

if (Test-Path $target) {
    $sz = (Get-Item $target).Length
    Write-Host "Model already downloaded: $($info.File) ($([math]::Round($sz / 1MB)) MB)" -ForegroundColor Green
    exit 0
}

Write-Host "`n=== Downloading $($info.Desc) ===" -ForegroundColor Cyan
Write-Host "  URL: $($info.Url)" -ForegroundColor DarkGray
Write-Host "  Target: $target" -ForegroundColor DarkGray
Write-Host "  Expected size: $($info.Size)" -ForegroundColor DarkGray
Write-Host ""

try {
    $ProgressPreference = 'SilentlyContinue'  # Speed up download
    Download-ModelFile -SourceUrl $info.Url -TargetPath $target
    $sz = (Get-Item $target).Length
    Write-Host "`n=== Download complete: $([math]::Round($sz / 1MB)) MB ===" -ForegroundColor Green
    Write-Host "  Run: .\build.ps1 -Run" -ForegroundColor Yellow
    Write-Host "  The model will be auto-detected and loaded at boot.`n" -ForegroundColor DarkGray
} catch {
    Write-Host "`nDownload failed: $_" -ForegroundColor Red
    Write-Host "You can manually download from:" -ForegroundColor Yellow
    Write-Host "  $($info.Url)" -ForegroundColor DarkGray
    Write-Host "Save to: models\$($info.File)`n" -ForegroundColor DarkGray
    exit 1
}
