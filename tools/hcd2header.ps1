# Convert BCM .hcd firmware file to a C header with byte array
# Usage: .\tools\hcd2header.ps1 -InputFile firmware\BCM4345C0.hcd -OutputFile kernel\drivers\bt\bt_firmware.h

param(
    [string]$InputFile = "firmware\BCM4345C0.hcd",
    [string]$OutputFile = "kernel\drivers\bt\bt_firmware.h"
)

if (-not (Test-Path $InputFile)) {
    Write-Host "Firmware not found: $InputFile" -ForegroundColor Yellow
    Write-Host "Generating stub (BT firmware upload will be skipped)" -ForegroundColor Yellow
    $stub = @"
/* Auto-generated stub — no BT firmware available.
 * Place BCM4345C0.hcd in firmware/ and rebuild to enable BT. */
#ifndef BT_FIRMWARE_H
#define BT_FIRMWARE_H
static const uint8_t bt_firmware[] = { 0 };
static const uint32_t bt_firmware_size = 0;
#endif
"@
    Set-Content $OutputFile $stub -NoNewline
    exit 0
}

$bytes = [System.IO.File]::ReadAllBytes($InputFile)
$size = $bytes.Length

$sb = [System.Text.StringBuilder]::new(($size * 6) + 512)

[void]$sb.AppendLine("/* Auto-generated from BCM4345C0.hcd -- do not edit */")
[void]$sb.AppendLine("/* Source: https://github.com/RPi-Distro/bluez-firmware */")
[void]$sb.AppendLine("/* SPDX-License-Identifier: Broadcom-RPi (redistributable) */")
[void]$sb.AppendLine("#ifndef BT_FIRMWARE_H")
[void]$sb.AppendLine("#define BT_FIRMWARE_H")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("static const uint8_t bt_firmware[] = {")

$lineWidth = 0
for ($i = 0; $i -lt $size; $i++) {
    $hex = ('0x{0:X2}' -f $bytes[$i])
    if ($i -lt $size - 1) { $hex += "," }
    [void]$sb.Append($hex)
    $lineWidth += $hex.Length
    if ($lineWidth -ge 76) {
        [void]$sb.AppendLine("")
        $lineWidth = 0
    }
}
[void]$sb.AppendLine("")
[void]$sb.AppendLine("};")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("static const uint32_t bt_firmware_size = $size;")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("#endif /* BT_FIRMWARE_H */")

Set-Content $OutputFile $sb.ToString() -NoNewline
Write-Host "Generated $OutputFile ($size bytes firmware, $($sb.Length) bytes header)" -ForegroundColor Green
