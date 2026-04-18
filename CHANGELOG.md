# Changelog

All notable changes to TensorOS are documented in this file.
The focus here is code and measured behavior, not release-note marketing.

---

## [0.3.0] — 2026-03-24

### Summary

**2.8× performance improvement.** Decode speed improved from 454 ms/tok to 162 ms/tok
through SMP parallel GEMV, JIT-compiled forward kernels, and critical bug fixes in
both the JIT loop counters and the SMP trampoline.

### Critical Fixes

#### SMP Page Table Relocation
Page tables relocated from `0x1000` to `0x10000` (18 pages). The original address
collided with the BIOS Data Area, causing AP bootstrap failures. All 4 CPUs now
come online reliably.

#### JIT Loop Counter Bugs
All JIT forward kernels (`vadd`, `dot`, `vmul`, `rmsnorm`) emitted `vecs/2` as the
loop count instead of `vecs`. This halved the computation, producing incorrect results
for every JIT-accelerated operation.

#### SMP Trampoline Stack Alignment
Changed AP entry from `jmp rax` to `call rax` to ensure 16-byte stack alignment
required by the System V ABI. Misalignment caused SSE2 `movaps` faults on APs.

### New Features

#### JIT Forward Kernels
Six native x86_64 kernels lazy-compiled on first LLM inference:
- `vadd` (dim=3072) — residual connections
- `dot` (head_dim=96) — attention score computation
- `axpy` (head_dim=96) — attention value accumulation
- `fused_silu_mul` (ff_dim=8192) — FFN gate ⊙ up projection
- `rope` (head_dim=96) — rotary position encoding
- `rmsnorm` (dim=3072) — RMS normalization

Emitted into a 2 MB W^X code pool (max 64 concurrent buffers).

#### SMP Parallel GEMV
- `smp_dispatch()` partitions GEMV rows across all online CPUs
- Supports Q4_0 and Q8_0 fused AVX2 GEMV paths
- Dispatches when `ncpu > 1 && out_dim >= 64`
- BSP + APs synchronized via `smp_wait_all()`

#### AVX2 Integer SIMD Emitters
7 new AVX2 integer SIMD instruction emitters added to the JIT engine.
Integer Q4×Q8 GEMV compiler implemented (disabled pending correctness verification).

### Performance

| Metric | v0.2.0 | v0.3.0 | Improvement |
|--------|--------|--------|-------------|
| Decode speed | 454 ms/tok | 162 ms/tok | 2.8× faster |
| CPUs used | 1 | 4 | SMP dispatch |
| JIT kernels | 0 | 6 | Forward pass JIT |
| JIT pool | 1 MB | 2 MB | Doubled capacity |

---

## [0.2.0] — 2026-03-23

### Summary

**First coherent LLM inference achieved.** Phi-3.5 Mini Instruct (3.8B params, Q4_0)
now generates correct English text on bare-metal x86_64, running under QEMU WHPX at
~800 ms/tok (454 ms/tok decode, 5.5s prefill for 12 tokens).

Prompt: `"What is an operating system?"`
Output: `"An operating system (OS) is a complex piece of software that man..."`

This release fixes critical numerical bugs in quantized inference that produced
garbage output since the initial LLM integration.

### Critical Fixes

#### Q4_0 Dequantization Layout (Root Cause of Garbage Output)
All Q4_0 (4-bit quantized) dequantization code used an **interleaved nibble layout**
instead of the GGML standard layout. For each 32-element block packed into 16 bytes:

- **Wrong (interleaved):** `out[2*j] = lo_nibble, out[2*j+1] = hi_nibble`
- **Correct (GGML standard):** `out[j] = lo_nibble, out[j+16] = hi_nibble`

This caused element-order corruption at every F32 boundary (RMSNorm multiply,
residual connections), compounding through all 32 transformer layers. Aggregate
statistics (min/max/sum) were identical since values were just permuted within
blocks, making the bug invisible to earlier verification.

**Files fixed:**
- `runtime/nn/llm.c` — `llm_embed()` Q4_0 case
- `runtime/nn/llm.c` — `q4_0_dot32()` (SSE2 + aarch64 paths)
- `runtime/nn/llm.c` — `q4_1_dot32()` (SSE2 + aarch64 paths)
- `runtime/nn/llm.c` — `q4_0_dot32_avx2()` (AVX2 8-wide path)
- `runtime/nn/llm.c` — `llm_gemv_q4_fused_avx2()` (4-row batched GEMV)
- `runtime/nn/llm.c` — `llm_gemv_q4_fused_range_avx2()` (parallel worker GEMV)
- `runtime/nn/llm.c` — AVX2 helper replaced: `q4_unpack_v8f` → `q4_unpack_lo_v8f` + `q4_unpack_hi_v8f`

#### RMSNorm Epsilon Mismatch
Hardcoded `1e-6f` replaced with model-specific epsilon loaded from GGUF metadata
(`general.rms_norm_eps`). Phi-3.5 uses `1e-5`.

#### Math Library Precision (Prior Session)
Complete rewrites of `sinf`, `cosf`, `expf`, `logf`, `sqrtf` — custom bare-metal
implementations had catastrophic precision errors affecting RoPE frequency computation
and softmax normalization.

### New Features

#### LongRoPE Scaling Factors
- Loaded `rope_factors_short` and `rope_factors_long` tensors from GGUF
- Applied frequency scaling in `llm_rope_precompute()` based on position vs
  original context length (4096 for Phi-3.5)
- Enables correct positional encoding for extended-context models

#### New Subsystems (Scaffolding)
- `runtime/nn/flash_attn.c` — Flash Attention kernel interface
- `runtime/nn/paged_attn.c` — PagedAttention (vLLM-style) interface
- `runtime/nn/safetensors.c` — Safetensors format loader
- `runtime/nn/onnx.c` — ONNX Runtime integration interface
- `runtime/compute/vulkan_compute.c` — Vulkan/WebGPU compute backend interface
- `runtime/pseudocode/pseudo_stdlib.c` — Pseudocode standard library
- `kernel/drivers/dma/pcie_dma.c` — PCIe DMA engine
- `kernel/net/distributed.c` — Distributed inference networking
- `boot/uefi_stub.c` — UEFI boot support

### Performance

| Metric | Before | After |
|--------|--------|-------|
| Decode speed | N/A (garbage) | 454 ms/tok |
| Prefill (12 tok) | N/A | 5,475 ms |
| End-to-end (16 tok) | N/A | 12,793 ms |
| Model load | 5.5s | 5.5s |
| Binary size | ~808 KB | ~808 KB |

### Numerical Verification

All values now match a Python/NumPy reference implementation exactly:

| Checkpoint | Python | TensorOS | Match |
|------------|--------|----------|-------|
| Embedding abssum | 72.35 | 72.35 | ✅ |
| Embed[0] | -0.009949 | -0.009949 | ✅ |
| Q[0] after GEMV | -0.419630 | -0.419630 | ✅ |
| L0 output min | -3.5961 | -3.5961 | ✅ |
| L0 output max | 2.5063 | 2.5063 | ✅ |
| L0 output abssum | 135.21 | 135.21 | ✅ |

Previously, Q[0] was -0.014042 (30× too small) and L0 output range was
[-0.37, 0.37] instead of [-3.60, 2.51] — a 10× dynamic range loss.

---

## [0.1.0] — 2025 (Initial Release)

### Features
- Multiboot1 bootloader with x86_64 long mode + SSE2
- SMP bootstrap (INIT-SIPI-SIPI)
- Tensor-aware memory manager (heap + arena + slab)
- GGUF model format parser
- Complete transformer forward pass
- Q4_0, Q4_1, Q6_K, Q8_0 quantization support
- AVX2+FMA SIMD acceleration with 4-row batched GEMV
- x86_64 JIT code generator for Q8_0 GEMV kernels
- SentencePiece and BPE tokenizers
- Temperature sampling with top-k and nucleus filtering
- Virtio-blk and virtio-net drivers
- ARP/IPv4/UDP/ICMP network stack
- AI shell with 20+ commands
- Pseudocode JIT compiler (lexer, parser, IR, 4-tier optimization)
- Model package manager
- ARM64 / Raspberry Pi 4 boot stub
