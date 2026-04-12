# Build CUDA kernels into a dynamic library
# Requires: NVIDIA CUDA Toolkit (nvcc)
#
# Usage: .\scripts\build_cuda.ps1
# Output: build_host\cuda_kernels.dll

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  CUDA Kernel Build System" -ForegroundColor Cyan
Write-Host "  Compiling GPU inference kernels" -ForegroundColor DarkCyan
Write-Host ""

# Find nvcc
$nvcc = $null
$cudaPaths = @(
    "$env:CUDA_PATH\bin\nvcc.exe",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe"
)

foreach ($p in $cudaPaths) {
    if (Test-Path $p) { $nvcc = $p; break }
}

if (-not $nvcc) {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
}

if (-not $nvcc) {
    Write-Host "  ERROR: nvcc not found. Install CUDA Toolkit." -ForegroundColor Red
    exit 1
}

Write-Host "  nvcc: $nvcc" -ForegroundColor Green

# Detect GPU arch
$gpu_arch = "sm_89"  # RTX 4070 = Ada Lovelace = SM 8.9
Write-Host "  Target: $gpu_arch (Ada Lovelace)" -ForegroundColor Green

# Create output dir
if (-not (Test-Path "build_host")) { New-Item -ItemType Directory -Path "build_host" | Out-Null }

# Compile
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$src = "runtime\nn\cuda_kernels.cu"
$out = "build_host\cuda_kernels.dll"

Write-Host "  Compiling $src..."

$arch_flag = "-arch=$gpu_arch"
$args_list = @("-shared", "-o", $out, $arch_flag, "-O2", "-DCUDA_KERNELS_EXPORTS", "--compiler-options", "/MD", "-I.", $src)
& $nvcc @args_list 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  BUILD FAILED" -ForegroundColor Red
    exit 1
}

$sw.Stop()
$size = (Get-Item $out).Length / 1KB

Write-Host ""
Write-Host "  BUILD SUCCESS" -ForegroundColor Green
Write-Host "  Output: $out ($([math]::Round($size)) KB)" -ForegroundColor Green
Write-Host "  Time: $([math]::Round($sw.Elapsed.TotalSeconds, 1))s" -ForegroundColor Green
