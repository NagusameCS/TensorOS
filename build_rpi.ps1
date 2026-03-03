# TensorOS ARM64 Build Script (Raspberry Pi 4)
# Builds the TensorOS kernel for AArch64 and creates kernel8.img for RPi boot
#
# Requirements: zig (0.15+)
# Usage: .\build_rpi.ps1 [-Flash] [-Clean]
# -Flash: Copy kernel8.img + config.txt to SD card (F:\)

param(
    [switch]$Flash,
    [switch]$Clean,
    [string]$SDDrive = "F:"
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$BUILD = "build_arm64"

$CFLAGS = @(
    "-target", "aarch64-freestanding-none",
    "-ffreestanding", "-fno-builtin", "-fno-stack-protector",
    "-nostdlib", "-fno-pic", "-fno-pie",
    "-O2", "-fno-sanitize=all",
    "-Wall",
    "-I.", "-S"
)

# Same source list as x86 build — all files have #ifdef __aarch64__ guards
$SOURCES = @(
    "kernel\core\main.c",
    "kernel\core\klib.c",
    "kernel\core\perf.c",
    "kernel\core\exception.c",
    "kernel\core\cpu_features.c",
    "kernel\core\watchdog.c",
    "kernel\core\selftest.c",
    "kernel\sched\tensor_sched.c",
    "kernel\mm\tensor_mm.c",
    "kernel\drivers\gpu\gpu.c",
    "kernel\drivers\tpu\tpu.c",
    "kernel\fs\git.c",
    "kernel\fs\tensorfs.c",
    "kernel\security\sandbox.c",
    "kernel\ipc\tensor_ipc.c",
    "virt\virt.c",
    "runtime\pseudocode\pseudocode_jit.c",
    "runtime\tensor\tensor_engine.c",
    "runtime\tensor\tensor_cpu.c",
    "runtime\tensor\tensor_avx2.c",
    "runtime\jit\x86_jit.c",
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
    "kernel\net\netstack.c",
    "kernel\drivers\blk\virtio_blk.c",
    "pkg\modelpkg.c",
    "userland\shell\aishell.c",
    "userland\monitor\tensor_monitor.c",
    "userland\deploy\deploy_service.c",
    "userland\train\train_service.c",
    "kernel\drivers\bt\rpi_bt.c",
    "kernel\drivers\blk\rpi_sd.c",
    "kernel\drivers\gpu\rpi_fb.c",
    "kernel\update\ota.c"
)

if ($Clean) {
    Write-Host "Cleaning ARM64 build directory..."
    Remove-Item "$BUILD\*" -Force -ErrorAction SilentlyContinue
    exit 0
}

if (-not (Test-Path $BUILD)) { New-Item -ItemType Directory $BUILD | Out-Null }

# Step 0: Generate BT firmware header from .hcd blob
$hcdFile = Join-Path $PSScriptRoot "firmware\BCM4345C0.hcd"
$fwHeader = Join-Path $PSScriptRoot "kernel\drivers\bt\bt_firmware.h"
$needRegen = $true
if (Test-Path $fwHeader) {
    if (Test-Path $hcdFile) {
        $needRegen = (Get-Item $hcdFile).LastWriteTime -gt (Get-Item $fwHeader).LastWriteTime
    } else {
        $needRegen = $false  # stub already exists
    }
}
if ($needRegen) {
    Write-Host "=== Generating BT firmware header ===" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "tools\hcd2header.ps1") -InputFile $hcdFile -OutputFile $fwHeader
}

# Step 1: Compile C -> asm -> fix string sections -> .o
Write-Host "=== Compiling $($SOURCES.Count) C sources (AArch64) ===" -ForegroundColor Cyan
$failed = 0
foreach ($src in $SOURCES) {
    $name = ($src -replace '\\','_' -replace '\.c$','')
    $sfile = "$BUILD\${name}.s"
    $ofile = "$BUILD\${name}.o"
    $stderr = zig cc @CFLAGS -o $sfile $src 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAIL: $src" -ForegroundColor Red
        Write-Host "  $stderr" -ForegroundColor DarkRed
        $failed++
        continue
    }
    # Workaround: zig lld corrupts relocations to SHF_MERGE sections
    $content = Get-Content $sfile -Raw
    $content = $content -replace '\.section\s+\.rodata\.[^,]+,"aM[S]?",@progbits,\d+', '.section .rodata,"a",@progbits'
    Set-Content $sfile $content -NoNewline
    $stderr2 = zig cc -target aarch64-freestanding-none -c -o $ofile $sfile 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAIL (asm): $src" -ForegroundColor Red
        $failed++
        continue
    }
    Write-Host "  $src" -ForegroundColor DarkGray
}

if ($failed -gt 0) {
    Write-Host "=== $failed file(s) failed to compile ===" -ForegroundColor Red
    exit 1
}

# Step 2: Assemble ARM64 boot stub
Write-Host "=== Assembling ARM64 boot.S ===" -ForegroundColor Cyan
$stderr = zig cc -target aarch64-freestanding-none -c -o "$BUILD\boot_arm64.o" boot\arm64\boot.S 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  FAIL: boot.S" -ForegroundColor Red
    Write-Host "  $stderr" -ForegroundColor DarkRed
    exit 1
}
Write-Host "  boot/arm64/boot.S" -ForegroundColor DarkGray

# Step 3: Link AArch64 kernel ELF
Write-Host "=== Linking kernel_arm64.elf ===" -ForegroundColor Cyan
$OBJS = @("$BUILD\boot_arm64.o")
foreach ($src in $SOURCES) {
    $name = ($src -replace '\\','_' -replace '\.c$','')
    $OBJS += "$BUILD\${name}.o"
}
$stderr = zig cc -target aarch64-freestanding-none -nostdlib -static -fno-pic -fno-pie `
    "-Wl,-T,boot/arm64/kernel.ld" "-Wl,--entry=_start" `
    -o "$BUILD\kernel_arm64.elf" @OBJS 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  FAIL: link" -ForegroundColor Red
    Write-Host "  $stderr" -ForegroundColor DarkRed
    exit 1
}

# Step 4: ELF to raw binary (kernel8.img) using zig objcopy
Write-Host "=== Creating kernel8.img ===" -ForegroundColor Cyan
$stderr = zig objcopy -O binary "$BUILD\kernel_arm64.elf" "$BUILD\kernel8.img" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  FAIL: objcopy" -ForegroundColor Red
    Write-Host "  $stderr" -ForegroundColor DarkRed
    exit 1
}
$sz = (Get-Item "$BUILD\kernel8.img").Length
Write-Host "  kernel8.img: $sz bytes ($([math]::Round($sz/1024))KB)" -ForegroundColor Green

# Step 5: Flash to SD card
if ($Flash) {
    Write-Host "`n=== Flashing to SD card ($SDDrive) ===" -ForegroundColor Yellow

    if (-not (Test-Path $SDDrive)) {
        Write-Host "  ERROR: SD card not found at $SDDrive" -ForegroundColor Red
        exit 1
    }

    # Copy our kernel
    Copy-Item "$BUILD\kernel8.img" "$SDDrive\kernel8.img" -Force
    Write-Host "  [OK] kernel8.img copied" -ForegroundColor Green

    # Pre-create BOOTLOG.TXT (32 KB of spaces) for the SD boot logger.
    # The kernel finds this file on FAT32 and overwrites it with log data.
    $logSize = 32 * 1024
    $logContent = [byte[]]::new($logSize)
    for ($i = 0; $i -lt $logSize; $i++) { $logContent[$i] = 0x20 }  # spaces
    [System.IO.File]::WriteAllBytes("$SDDrive\BOOTLOG.TXT", $logContent)
    Write-Host "  [OK] BOOTLOG.TXT pre-created (32 KB)" -ForegroundColor Green

    # Create bare-metal config.txt
    $config = @"
# TensorOS ARM64 bare-metal configuration
# Raspberry Pi 4 (BCM2711)

# Force 64-bit mode (loads kernel8.img at 0x80000)
arm_64bit=1

# Enable UART for serial console (115200 baud)
enable_uart=1

# BT uses PL011 UART0 on GPIO 30-33 (ALT3), debug uses mini UART on GPIO 14/15
# No dtoverlay needed — bare-metal code configures GPIO directly

# Fixed core frequency for stable UART baud rate
core_freq=500
core_freq_min=500

# Disable Linux-specific features
kernel=kernel8.img
disable_commandline_tags=1
disable_overscan=1

# GPU memory (for HDMI framebuffer)
gpu_mem=64

# No splash screen
disable_splash=1

# Boot immediately
boot_delay=0
"@
    Set-Content "$SDDrive\config.txt" $config -NoNewline
    Write-Host "  [OK] config.txt written" -ForegroundColor Green

    # Ensure required firmware files exist — auto-copy from firmware/ if available
    $firmware = @("bootcode.bin", "start4.elf", "fixup4.dat")
    $fwDir = Join-Path $PSScriptRoot "firmware"
    $missing = @()
    foreach ($f in $firmware) {
        if (-not (Test-Path "$SDDrive\$f")) {
            $local = Join-Path $fwDir $f
            if (Test-Path $local) {
                Copy-Item $local "$SDDrive\$f" -Force
                Write-Host "  [OK] $f copied from firmware/" -ForegroundColor Green
            } else {
                $missing += $f
            }
        } else {
            Write-Host "  [OK] $f already on SD" -ForegroundColor Green
        }
    }
    if ($missing.Count -gt 0) {
        Write-Host "  WARNING: Missing firmware files: $($missing -join ', ')" -ForegroundColor Yellow
        Write-Host "  Run: Invoke-WebRequest https://github.com/raspberrypi/firmware/raw/master/boot/<file> -OutFile firmware\<file>" -ForegroundColor Yellow
    }

    Write-Host "`n=== SD card ready! Insert into RPi4 and power on. ===" -ForegroundColor Green
    Write-Host "  Connect USB-UART adapter to GPIO 14 (TX) / 15 (RX) for serial console." -ForegroundColor DarkGray
    Write-Host "  Baud: 115200, 8N1" -ForegroundColor DarkGray
}

Write-Host "`n=== ARM64 build complete ===" -ForegroundColor Green
