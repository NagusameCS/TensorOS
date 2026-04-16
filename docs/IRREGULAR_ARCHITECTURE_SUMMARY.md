# What's Irregular: TensorOS & HyperTensor

A factual inventory of every custom, non-standard, or from-scratch component across both projects.

---

## TensorOS — Bare-Metal AI-First Operating System

### 1. Custom Bootloader & CPU Bring-Up

| Component | Detail |
|-----------|--------|
| **Multiboot1 → Long Mode** | `boot/entry64.asm` transitions 32-bit Multiboot entry to 64-bit long mode. SSE2 is force-enabled (CR0/CR4) *before* any C code runs so the entire C runtime can assume SIMD availability. |
| **Identity-Mapped Paging** | Kernel linked at `0x200000` with VA = PA — no virtual memory isolation. Intentional: eliminates TLB misses for large tensor buffers at the cost of all process/memory protection. |
| **ARM64 Boot** | Separate `boot/arm64/` stubs handle EL2→EL1 transition with PSCI for multiprocessor startup on Raspberry Pi. |
| **Custom Linker Scripts** | `boot/kernel64.ld`, `boot/linker.ld` hand-tune section placement for tensor-specific memory layout. |
| **Exception Handler** | `kernel/core/exception.c` attempts best-effort crash logging to SD card (`BOOTLOG.TXT`) during fault — most bare-metal kernels just halt. |

### 2. Custom SMP Implementation

- **Manual INIT-SIPI-SIPI trampoline** in `kernel/core/smp.c` — the trampoline is a **static byte array** of hand-encoded 16-bit→32-bit→64-bit transition code (not assembled from source).
- **IPI work-dispatch model**, not preemptive scheduling. BSP enqueues work → sends IPI 0xFE → APs execute to completion → sleep. No context switching, no time slicing.
- Achieves **2.8× parallel GEMV speedup** across 4 cores on Q4×Q8 integer dot products.

### 3. Custom Tensor-Aware Memory Manager

- `kernel/mm/tensor_mm.c` parses Multiboot memory map into **four dedicated zones**:
  - `MM_ZONE_KERNEL` — 4 KB pages, slab allocator for kernel objects.
  - `MM_ZONE_TENSOR` — 2 MB pages for tensor operations (TLB-friendly).
  - `MM_ZONE_MODEL_CACHE` — LRU model weight cache.
  - `MM_ZONE_GPU_DMA` — pinned memory for GPU transfers.
- Slab allocator deliberately does **not recycle freed slabs** (assumes long-running inference workloads).
- No virtual memory, no page tables, no user/kernel split — everything runs in Ring 0.

### 4. Custom ML Inference Runtime (Ring 0)

The entire transformer pipeline runs in kernel mode:

| Subsystem | File(s) | What's Custom |
|-----------|---------|---------------|
| **GGUF Loader** | `runtime/nn/gguf.c` | Full binary GGUF v2/v3 parser in bare-metal C. No Python, no libc. Supports Q4_0, Q4_1, Q6_K, Q8_0, BF16, FP32. |
| **Transformer Forward** | `runtime/nn/transformer.c` | Complete transformer: RoPE, multi-head self-attention, KV-cache, RMSNorm, SwiGLU FFN, causal masking. Runs in Ring 0. |
| **Q4 Quantization** | `runtime/nn/quantize4.c` | Custom fused dequant-and-dot kernel: unpacks INT4 nibbles to float AND computes dot product in one SSE2 pass. 6.4× compression vs FP32. |
| **Flash Attention** | `runtime/nn/flash_attn.c` | FlashAttention v1 (Dao et al.) in bare-metal C. Tiled 32×32 blocks, running softmax, O(N) memory, SSE2 dot products, GQA support. |
| **Speculative Neural Execution** | `runtime/nn/speculative.c` | Five custom techniques: Adaptive Precision Cascade (INT16→FP32 escalation), Speculative Layer Fusion (cached-signature layer skip), Entropy-Aware Neuron Pruning, Compute DAG Scheduling (Tomasulo's), Confidence-Gated Early Exit. |
| **Neuroevolution** | `runtime/nn/evolution.c` | Genetic algorithm evolves network architectures + weights at boot time. Population of 16, tournament selection, solves XOR from scratch. |
| **Braniac** | `runtime/nn/braniac.c` | Brain-inspired predictive coding with recognition/generative pathways, precision neurons, lateral connections. Not standard backprop. |
| **Training** | `runtime/nn/train.c` | Full backpopagation + Adam optimizer + REINFORCE policy gradient. Cross-entropy and MSE loss. SSE2 SIMD on all matrix ops. |
| **Model Metadata** | `runtime/nn/model_meta.c` | 53 HuggingFace→GGUF tensor name mapping rules. Auto-detects LLaMA, Gemma, Qwen2, Phi-2/3, Mistral, GPT-2. |
| **Tensor Bridge** | `runtime/nn/tensor_bridge.c` | Hidden-state capture/injection at any transformer layer. Linear projection for dimension mismatch. Enables model chaining in kernel. |
| **Paged Attention** | `runtime/nn/paged_attn.c` | vLLM-style block allocator: 2048 blocks × 16 tokens, copy-on-write for beam search, block swap-out, 256 concurrent sequences. |
| **Backend Abstraction** | `runtime/nn/backend.c` | 17-op compute vtable with CPU reference, CUDA skeleton, MLIR skeleton. |

### 5. Custom JIT Compiler

`runtime/jit/x86_jit.c` — **hand-written x86_64 instruction emitter**, no LLVM or external codegen:

- Byte-accurate ModRM/SIB/REX/VEX encoding (~2000 lines).
- SSE2 SIMD emission: `movups`, `addps`, `mulps`, `haddps` emitted as raw bytes.
- System V AMD64 ABI compliance (RDI/RSI/RDX/RCX/R8/R9).
- 2 MB static code buffer in BSS with W^X enforcement.
- **6 JIT-compiled kernels**: vadd, dot product, fused SiLU-mul, RoPE, RMSNorm, fast-exp.
- Pseudocode DSL (`runtime/pseudocode/pseudocode_jit.c`) with `git commit` and `deploy` as language keywords.

### 6. Custom Filesystem & Kernel Git

| Component | Detail |
|-----------|--------|
| **TensorFS** | `kernel/fs/tensorfs.c` — in-memory inode array, metadata only, no data storage, flat (non-hierarchical). Placeholder. |
| **Kernel Git** | `kernel/fs/git.c` — SHA-256 object store, reference management, repository init — all in kernel space. Designed for atomic model weight commits without userland. |

### 7. Custom Network Stack

- `kernel/net/netstack.c` — ARP, IPv4 (fragmentation/reassembly), ICMP, UDP, minimal TCP (3-way handshake).
- **Ollama-compatible HTTP REST API** built into the kernel: `/api/generate`, `/api/chat`, `/api/tags`. Curl from another machine and get live model outputs.
- RFC 6528-compliant ISN randomization via FNV-1a hash.
- Drivers: `virtio-blk` (model loading from Qemu disk), `virtio-net` (packet I/O), `rpi_sd` (ARM64 SD card).

### 8. Custom Drivers (Mixed Real/Stub)

| Driver | Status | Detail |
|--------|--------|--------|
| **GPU (PCI)** | Detection only | Real PCI config space scan, enumerates NVIDIA/AMD/Intel, reads capability flags. Compute kernels are no-ops. |
| **TPU** | Stub | Returns -1 on all operations. |
| **virtio-blk** | Working | Full virtio block device spec, polling I/O. |
| **virtio-net** | Working | Rx/tx descriptor rings, integrated with netstack. |
| **RPi SD** | Working | SDHOST controller commands for ARM64. |
| **E1000** | Partial | Intel NIC driver scaffold. |
| **AHCI** | Partial | SATA controller scaffold. |

### 9. Custom Security

- `kernel/security/crypto.c` — SHA-256 + ChaCha20 in bare-metal C (no OpenSSL).
- `kernel/security/sandbox.c` — permission bits and audit logging, but trivially bypassable (everything is Ring 0).
- `kernel/security/ssh.c` — SSH protocol scaffold.
- **No virtual memory isolation**, no user/kernel boundary, no ASLR.

### 10. Custom Virtualization (Stub)

- `virt/virt.c` — container lifecycle management, GPU resource allocation, shared memory regions, VT-x/SVM detection via CPUID. **No actual VM launch or container execution.**

### 11. Custom Userland (Ring 0)

- **AI Shell** (`userland/shell/aishell.c`): `model load`, `infer`, `deploy`, `git commit` as shell commands.
- **Tensor Monitor** (`userland/monitor/tensor_monitor.c`): token rate, memory utilization profiling.
- **OTA Updates** (`kernel/update/ota.c`): over-the-air update mechanism.
- All "userland" code runs in Ring 0.

### 12. Custom Build System

- `Makefile`: `-ffreestanding -nostdlib -nostdinc -mcmodel=kernel -fno-pic`.
- SSE is disabled at compile time (`-mno-sse -mno-sse2`) to prevent compiler-generated SIMD before boot code enables it, but SSE2 is force-enabled at runtime.
- GRUB `grub-mkrescue` for bootable ISO generation.
- Cross-compile support: `x86_64-elf-gcc` with system `gcc` fallback.

### 13. Custom Package Format

- `pkg/modelpkg.c` — `mod_pkg_t` structure: model name, version, architecture, weights, quantization format, dependencies, author, license. Kernel-level model packaging.

---

## HyperTensor — Hosted Inference Engine

HyperTensor shares the same runtime codebase as TensorOS but compiles as a **hosted userland binary** (Windows/Linux) instead of a bare-metal kernel.

### 1. Custom Quantized Arithmetic

- **Fused dequant-and-dot**: Q4_0 dequantization and matrix multiply happen in a single SIMD pass (SSE2/AVX2). Most engines separate the two.
- **Integer-only Q4×Q8 GEMV**: `int8` arithmetic path with AVX2 8-element/iteration, 2× k-unroll. No float promotion.
- **Q4_0 layout**: GGML-compatible `(lo | (hi << 4))` nibble packing, verified byte-exact against reference.

### 2. Custom JIT (Same Emitter, Different ABI)

- Same hand-written x86_64 emitter as TensorOS.
- **JIT disabled on Windows hosted mode** due to SysV vs Windows calling convention mismatch (SysV: rdi/rsi; Windows: rcx/rdx). Works on bare-metal and Linux.
- Six compiled kernels: fast-exp, SiLU, RMSNorm, softmax, RoPE, GELU.

### 3. Custom Multi-Format Model Loading

| Format | File | Status |
|--------|------|--------|
| **GGUF** | `runtime/nn/gguf.c` | Complete. All 18 GGML quant types. |
| **SafeTensors** | `runtime/nn/safetensors.c` | ~80%. JSON header parse, 9 dtype conversions (BF16→FP32 via bit shift). |
| **ONNX** | `runtime/nn/onnx.c` | Minimal protobuf decoder. 8 operators (Gemm, MatMul, Add, ReLU, Sigmoid, Softmax, Transpose). |
| **HuggingFace Auto-Download** | `runtime/nn/hf_download.c` | Downloads GGUF from HF Hub via HTTPS (WinHTTP on Windows, curl on Linux). |

No other single-binary engine combines all four formats with runtime format detection.

### 4. Custom Token-Space Communication

- `runtime/nn/token_comm.c` — inter-model communication in **distributional space**, not text:
  - **HARD** mode: argmax tokens.
  - **SOFT** mode: full logit distributions.
  - **STOCHASTIC** mode: sampling + soft transfers.
- Cross-vocabulary mapping via hash table. Models with different tokenizers can exchange logits without text serialization.

### 5. Custom Hosted HAL

- `host/hal.c` maps kernel abstractions to OS calls:
  - `kmalloc()` → `malloc()` + 64-byte alignment.
  - Pinned memory → `VirtualLock` (Windows) / `mlock` (Linux).
  - Console I/O → `WriteConsoleA` / `write`.
  - Disk I/O → `CreateFileA` / `open` with `mmap`.
  - Timer → `QueryPerformanceCounter` / `clock_gettime`.
  - Networking → Winsock2 / POSIX sockets.

### 6. Custom Axiom Beta (Geometric Model Analysis)

- `runtime/nn/axiom_beta.c` — 5-phase model geometry survey:
  1. Manifold dimension estimation.
  2. Symmetry extraction.
  3. Curvature proxy computation.
  4. Axiom-set formalization.
  5. Native inference complexity projection.
- Reports intrinsic dimension, metric rank, symmetry invariance score, projected optimal inference cost.

### 7. Custom Build Toolchain

- **Zig C compiler** (`zig cc`) as primary compiler for unified cross-platform targeting.
- Single binary ~346 KB, compiles in ~1.3s.
- Optional `-DENABLE_CUDA` and `-DENABLE_MLIR` backends.
- Links: `advapi32`, `winhttp`, `ws2_32` on Windows.

### 8. Custom Architecture Auto-Detection

- Auto-detects model family from GGUF metadata and tensor names.
- Auto-selects: RMSNorm vs LayerNorm, SwiGLU vs GELU, RoPE frequency base, LongRoPE factors.
- Verified on: LLaMA/2/3, Gemma/2/4, Qwen2/2.5, Phi-3/3.5, Mistral, SmolLM/2, TinyLlama.

---

## Shared Custom Components (Both Projects)

These components exist in the shared `runtime/` codebase and compile into both TensorOS (bare-metal) and HyperTensor (hosted):

| Component | What's Custom |
|-----------|---------------|
| **x86_64 JIT Emitter** | Hand-written instruction encoder, no LLVM. ModRM/SIB/REX byte emission. |
| **Flash Attention** | Cache-conscious tiled implementation with running softmax, SSE2 accelerated. |
| **Speculative Neural Execution** | Five combined techniques (APC, SLF, EANP, DAG scheduling, early exit). |
| **Neuroevolution** | Genetic architecture + weight search at runtime. |
| **Braniac** | Predictive coding with precision neurons and lateral connections. |
| **Fused Q4 Kernels** | Single-pass dequant+dot in SIMD. |
| **Token-Native Pipeline** | KV cache, agent loops, RAG injection, speculative verification, RL rollouts, structured output validation — all operate on token IDs without text decode/re-encode. |
| **Multi-Format Loader** | GGUF + SafeTensors + ONNX + HF auto-download in one binary. |
| **Pseudocode DSL** | Custom ML language with `git commit` and `deploy` as keywords. |

---

## What's NOT Custom (Standard / External)

- **GGUF binary format** — same spec as llama.cpp / GGML.
- **GGML quantization type IDs** — compatible numbering.
- **Multiboot1 boot protocol** — GRUB standard.
- **virtio device spec** — standard OASIS virtio.
- **SHA-256 / ChaCha20** — standard algorithms (custom implementation, not custom algorithm).
- **Adam optimizer** — standard formulation.
- **BPE tokenization** — standard algorithm (custom implementation).
- **RoPE, RMSNorm, SwiGLU** — published techniques (custom bare-metal implementations).

---

## Summary

**TensorOS** is a bare-metal operating system where the kernel *is* the inference engine. Custom boot, custom memory zones, custom SMP dispatch, custom network stack with built-in Ollama API, custom JIT, kernel-level git — all in Ring 0 with no libc.

**HyperTensor** is the same runtime compiled as a small hosted binary. Adds multi-format model loading, HuggingFace auto-download, token-space inter-model communication, and geometric model analysis. Builds with Zig in ~1 second to a ~346 KB executable.

Both share: fused quantized SIMD kernels, hand-written JIT, flash attention, speculative neural execution, neuroevolution, and a fully token-native processing pipeline that avoids text decode/re-encode at every internal boundary.
