# TensorChat Native (Windows C App)

Modular native C desktop studio with:
- warmup loading screen on boot
- OBS-style module controls (Add Chat, Add Editor, Add Terminal, Add Outline)
- default module is Chat (history + input + send)
- optional editor + structure outline module
- optional sandbox terminal module
- Jekyll-inspired theme presets (Minima/Cayman/Architect/Slate/Midnight/Hacker/Dinky)

## Build

From this folder:

```
./build_native.ps1
```

Output: `tensorchat_native.exe`

## Notes

- This is a hosted Windows tool and is not part of kernel build.
- Chat now calls TensorOS HTTP API at `http://127.0.0.1:8080/v1/chat/completions` with local fallback when runtime is unavailable.
- Outline parsing currently tracks common symbols (`int`, `void`, `class`, `struct`, `fn`, `def`).
- Terminal module runs in sandbox mode with built-in commands (`help`, `clear`, `pwd`, `status`).
- Intended as a productionizable shell for upcoming runtime wiring and richer theming.
