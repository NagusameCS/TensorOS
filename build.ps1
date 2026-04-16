# TensorOS Build Script
# Builds the TensorOS kernel for x86_64 and creates a bootable QEMU image
#
# Requirements: zig (0.15+), NASM
# Usage: .\build.ps1 [-Run] [-Clean]

param(
    [switch]$Run,
    [switch]$Interactive,
    [switch]$Clean
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$NASM = "C:\Users\legom\AppData\Local\bin\NASM\nasm.exe"
$QEMU = "C:\Program Files\qemu\qemu-system-x86_64.exe"
$BUILD = "build"

$CFLAGS = @(
    "-target", "x86_64-freestanding-none",
    "-ffreestanding", "-fno-builtin", "-fno-stack-protector",
    "-nostdlib", "-mno-red-zone", "-fno-pic", "-fno-pie",
    "-O2", "-fno-sanitize=all", "-msse2",
    "-Wno-unused-function", "-Wno-unused-variable", "-Wno-format",
    "-Wno-incompatible-pointer-types", "-Wno-int-conversion",
    "-I.", "-S"
)

$SOURCES = @(
    "kernel\core\main.c",
    "kernel\core\klib.c",
    "kernel\core\perf.c",
    "kernel\core\exception.c",
    "kernel\core\cpu_features.c",
    "kernel\core\watchdog.c",
    "kernel\core\selftest.c",
    "kernel\core\syscall.c",
    "kernel\sched\tensor_sched.c",
    "kernel\mm\tensor_mm.c",
    "kernel\drivers\gpu\gpu.c",
    "kernel\drivers\tpu\tpu.c",
    "kernel\fs\git.c",
    "kernel\fs\tensorfs.c",
    "kernel\security\sandbox.c",
    "kernel\security\crypto.c",
    "kernel\security\ssh.c",
    "kernel\security\security.c",
    "kernel\ipc\tensor_ipc.c",
    "virt\virt.c",
    "runtime\pseudocode\pseudocode_jit.c",
    "runtime\tensor\tensor_engine.c",
    "runtime\tensor\tensor_cpu.c",
    "runtime\tensor\tensor_avx2.c",
    "runtime\jit\x86_jit.c",
    "runtime\jit\llm_jit.c",
    "runtime\nn\inference.c",
    "runtime\nn\quantize.c",
    "runtime\nn\evolution.c",
    "runtime\nn\train.c",
    "runtime\nn\speculative.c",
    "runtime\nn\transformer.c",
    "runtime\nn\quantize4.c",
    "kernel\mm\tensor_arena.c",
    "runtime\nn\gguf.c",
    "runtime\nn\math_llm.c",
    "runtime\nn\llm.c",
    "kernel\core\smp.c",
    "kernel\drivers\net\virtio_net.c",
    "kernel\drivers\net\e1000.c",
    "kernel\net\netstack.c",
    "kernel\net\tls.c",
    "kernel\drivers\blk\virtio_blk.c",
    "kernel\drivers\blk\ahci.c",
    "pkg\modelpkg.c",
    "userland\shell\aishell.c",
    "userland\monitor\tensor_monitor.c",
    "userland\deploy\deploy_service.c",
    "userland\train\train_service.c",
    "runtime\nn\braniac.c",
    "runtime\nn\backend.c",
    "runtime\nn\model_meta.c",
    "runtime\nn\tensor_bridge.c",
    "kernel\drivers\bt\rpi_bt.c",
    "kernel\drivers\blk\rpi_sd.c",
    "kernel\update\ota.c",
    "kernel\drivers\dma\pcie_dma.c",
    "runtime\nn\flash_attn.c",
    "runtime\nn\paged_attn.c",
    "runtime\nn\safetensors.c",
    "runtime\nn\onnx.c",
    "runtime\nn\axiom_linalg.c",
    "runtime\nn\axiom_geo.c",
    "runtime\nn\axiom_beta.c",
    "runtime\pseudocode\pseudo_stdlib.c",
    "runtime\compute\vulkan_compute.c",
    "kernel\net\distributed.c"
)

if ($Clean) {
    Write-Host "Cleaning build directory..."
    Remove-Item "$BUILD\*" -Force -ErrorAction SilentlyContinue
    exit 0
}

if (-not (Test-Path $BUILD)) { New-Item -ItemType Directory $BUILD | Out-Null }

# Step 1: Compile C -> asm -> fix string sections -> .o
Write-Host "=== Compiling $($SOURCES.Count) C sources ===" -ForegroundColor Cyan
foreach ($src in $SOURCES) {
    $name = ($src -replace '\\','_' -replace '\.c$','')
    $sfile = "$BUILD\${name}.s"
    $ofile = "$BUILD\${name}.o"
    $out = zig cc @CFLAGS -o $sfile $src 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Host "  ERROR compiling $src" -ForegroundColor Red; $out | Write-Host; exit 1 }
    # Workaround: zig lld corrupts relocations to SHF_MERGE sections
    $content = Get-Content $sfile -Raw
    $content = $content -replace '\.section\s+\.rodata\.[^,]+,"aM[S]?",@progbits,\d+', '.section .rodata,"a",@progbits'
    Set-Content $sfile $content -NoNewline
    $out = zig cc -target x86_64-freestanding-none -c -o $ofile $sfile 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Host "  ERROR assembling $sfile" -ForegroundColor Red; $out | Write-Host; exit 1 }
    Write-Host "  $src" -ForegroundColor DarkGray
}

# Step 2: Assemble 64-bit entry point (must be first object linked)
Write-Host "=== Assembling entry64.asm ===" -ForegroundColor Cyan
$out = & $NASM -f elf64 -o "$BUILD\entry64.o" boot\entry64.asm 2>&1
if ($LASTEXITCODE -ne 0) { Write-Host "  ERROR assembling entry64.asm" -ForegroundColor Red; $out | Write-Host; exit 1 }

# Step 3: Link 64-bit kernel ELF
Write-Host "=== Linking kernel64.elf ===" -ForegroundColor Cyan
$OBJS = @("$BUILD\entry64.o")
foreach ($src in $SOURCES) {
    $name = ($src -replace '\\','_' -replace '\.c$','')
    $OBJS += "$BUILD\${name}.o"
}
$out = zig cc -target x86_64-freestanding-none -nostdlib -static -fno-pic -fno-pie `
    "-Wl,-T,boot/kernel64.ld" "-Wl,--entry=long_mode_entry" `
    -o "$BUILD\kernel64.elf" @OBJS 2>&1
if ($LASTEXITCODE -ne 0) { Write-Host "  LINK ERROR" -ForegroundColor Red; $out | Write-Host; exit 1 }

# Step 4: ELF to flat binary (custom - zig objcopy has segment mapping bug)
Write-Host "=== Creating kernel64.bin ===" -ForegroundColor Cyan
$elf = [System.IO.File]::ReadAllBytes("$PWD\$BUILD\kernel64.elf")
$phoff = [BitConverter]::ToInt64($elf, 0x20)
$phentsize = [BitConverter]::ToUInt16($elf, 0x36)
$phnum = [BitConverter]::ToUInt16($elf, 0x38)
$segments = @(); $minVA = [UInt64]::MaxValue; $maxVA = [UInt64]0
for ($p = 0; $p -lt $phnum; $p++) {
    $o = [int]$phoff + $p * $phentsize
    if ([BitConverter]::ToUInt32($elf, $o) -ne 1) { continue }
    $foff = [BitConverter]::ToUInt64($elf, $o + 8)
    $vaddr = [BitConverter]::ToUInt64($elf, $o + 16)
    $filesz = [BitConverter]::ToUInt64($elf, $o + 32)
    if ($filesz -eq 0) { continue }
    if ($vaddr -lt $minVA) { $minVA = $vaddr }
    if (($vaddr + $filesz) -gt $maxVA) { $maxVA = $vaddr + $filesz }
    $segments += @{v=$vaddr; f=[int]$foff; s=[int]$filesz}
}
$bin = New-Object byte[] ([int]($maxVA - $minVA))
foreach ($seg in $segments) {
    [Array]::Copy($elf, $seg.f, $bin, [int]($seg.v - $minVA), $seg.s)
}
[System.IO.File]::WriteAllBytes("$PWD\$BUILD\kernel64.bin", $bin)
Write-Host "  kernel64.bin: $($bin.Length) bytes"

# Step 5: Multiboot stub (embeds kernel64.bin via incbin)
Write-Host "=== Building multiboot stub ===" -ForegroundColor Cyan
$out = & $NASM -f elf32 -o "$BUILD\multiboot_stub.o" boot\multiboot_stub.asm 2>&1
if ($LASTEXITCODE -ne 0) { Write-Host "  ERROR assembling multiboot_stub.asm" -ForegroundColor Red; $out | Write-Host; exit 1 }

# Step 6: Final link
Write-Host "=== Linking tensoros.elf ===" -ForegroundColor Cyan
$out = zig cc -target x86-freestanding-none -nostdlib -static `
    "-Wl,-T,boot/stub32.ld" "-Wl,--entry=_start" `
    -o "$BUILD\tensoros.elf" "$BUILD\multiboot_stub.o" 2>&1
if ($LASTEXITCODE -ne 0) { Write-Host "  ERROR linking tensoros.elf" -ForegroundColor Red; $out | Write-Host; exit 1 }

$sz = (Get-Item "$BUILD\tensoros.elf").Length
Write-Host "=== Build complete: tensoros.elf ($sz bytes) ===" -ForegroundColor Green

# Auto-detect GGUF model files for LLM inference
$ModelDrive = @()
$ModelFile = Get-ChildItem -Path "models\*.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($ModelFile) {
    Write-Host "  Model found: $($ModelFile.Name) ($([math]::Round($ModelFile.Length / 1MB)) MB)" -ForegroundColor Yellow
    $ModelDrive = @("-drive", "file=$($ModelFile.FullName),format=raw,if=virtio,readonly=on")
}

if ($Interactive) {
    Write-Host "`n=== Booting TensorOS (Interactive, WHPX) ===" -ForegroundColor Yellow
    Write-Host "  Serial output only (VGA disabled for WHPX compatibility)." -ForegroundColor DarkGray
    Write-Host "  Type 'exit' in the shell to shut down." -ForegroundColor DarkGray
    Remove-Item "$BUILD\serial.log" -Force -ErrorAction SilentlyContinue
    $qemuArgs = @(
        "-machine", "q35,accel=whpx",
        "-cpu", "EPYC-v4",
        "-smp", "4,cores=4,threads=1",
        "-kernel", "$BUILD\tensoros.elf",
        "-serial", "file:$BUILD\serial.log",
        "-display", "none", "-vga", "none",
        "-no-reboot", "-m", "8G",
        "-device", "isa-debug-exit,iobase=0x501,iosize=2"
    ) + $ModelDrive
    & $QEMU @qemuArgs
    Write-Host "`n=== TensorOS exited ===" -ForegroundColor Green
    if (Test-Path "$BUILD\serial.log") {
        $log = [System.IO.File]::ReadAllBytes("$PWD\$BUILD\serial.log")
        Write-Host "Serial log: $($log.Length) bytes saved to build\serial.log"
        [System.Text.Encoding]::ASCII.GetString($log)
    }
} elseif ($Run) {
    Write-Host "`n=== Booting TensorOS in QEMU (WHPX) ===" -ForegroundColor Yellow
    Remove-Item "$BUILD\serial.log" -Force -ErrorAction SilentlyContinue
    $qemuArgs = @(
        "-machine", "q35,accel=whpx",
        "-cpu", "EPYC-v4",
        "-smp", "4,cores=4,threads=1",
        "-kernel", "$BUILD\tensoros.elf",
        "-serial", "file:$BUILD\serial.log",
        "-display", "none", "-vga", "none",
        "-no-reboot", "-m", "8G",
        "-device", "isa-debug-exit,iobase=0x501,iosize=2",
        "-nic", "user,model=virtio-net-pci"
    ) + $ModelDrive
    $proc = Start-Process -FilePath $QEMU -ArgumentList $qemuArgs -PassThru
    $timeout = if ($ModelFile -and $ModelFile.Length -gt 500MB) { 600 } elseif ($ModelFile) { 300 } else { 90 }
    Write-Host "  WHPX: near-native speed. Waiting ${timeout}s for boot" -ForegroundColor DarkGray
    Start-Sleep $timeout
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep 1
    if (Test-Path "$BUILD\serial.log") {
        $log = [System.IO.File]::ReadAllBytes("$PWD\$BUILD\serial.log")
        Write-Host "`n=== Serial Output ($($log.Length) bytes) ===" -ForegroundColor Cyan
        [System.Text.Encoding]::ASCII.GetString($log)
    }
}
