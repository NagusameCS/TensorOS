# ==============================================================================
# TensorOS — Architecture Deep Dive
# ==============================================================================

This is the reference architecture file for the bare-metal side of TensorOS. It describes the boot path, memory layout, and major runtime subsystems as they exist in the current tree.

## Boot Sequence

```
BIOS → multiboot_stub.asm (Multiboot1) → boot.asm → entry64.asm → kernel_main() → AI Shell
```

### Phase 1: Hardware Init (multiboot_stub.asm + boot.asm)
1. Multiboot1 header validated (magic `0x1BADB002`)
2. `multiboot_stub.asm`: serial checkpoints, CPUID checks, builds 16 GB identity map
   with 2 MB huge pages (PML4 → PDPT → PD at `0x10000`, 18 pages)
3. Loads `kernel64.bin` at `0x200000`, jumps to 32-bit `boot.asm`
4. `boot.asm`: GDT loaded, PAE + long mode (IA32_EFER.LME) enabled
5. Jump to 64-bit `entry64.asm` → `kernel_main`

### Phase 2: Kernel Subsystem Init (main.c)
1. `tensor_mm_init()` — physical bitmap, tensor heap, model cache, slab allocator
2. `smp_init()` — LAPIC enable, AP bootstrap via INIT-SIPI-SIPI (up to 64 CPUs)
3. `tensor_sched_init()` — priority queues, GPU device state
4. `gpu_init()` / `tpu_init()` — PCI bus scan, capability detection
5. `git_init()` — kernel-level git object store
6. `tensorfs_init()` — AI-aware virtual filesystem
7. `sandbox_init()` — security subsystem
8. `tensor_ipc_init()` — IPC channels
9. `virt_init()` — VT-x/AMD-V detection, container support

### Phase 3: Runtime Init
1. `tensor_engine_init()` — eager ops, compute graph engine
2. `pseudo_runtime_init()` — Pseudocode JIT (lexer, parser ready)
3. `modelpkg_init()` — model registries

### Phase 4: Userland
1. `monitor_daemon_main()` — background monitoring (future: separate MEU)
2. `deploy_init()` / `train_init()` — services ready
3. `aishell_main()` — interactive shell starts, banner displayed


## Memory Layout

Page tables are built by `multiboot_stub.asm` at physical address `0x10000` (18 pages).
The first 16 GB of physical memory is identity-mapped using 2 MB huge pages.

```
0x0000_0000_0000_0000  ┌──────────────────────────────┐
                       │  Kernel code + data           │
                       │  (loaded at 0x200000)         │
0x0000_0000_0001_0000  │  Page tables (18 pages)       │
                       │  PML4, PDPT, 8× PD            │
                       ├──────────────────────────────┤
                       │  SMP trampoline (0x8000)      │
                       │  AP stacks (65 KB each)       │
                       ├──────────────────────────────┤
                       │  Identity-mapped first 16 GB  │
                       │  (2 MB huge pages)            │
                       │  ┌────────────────────────┐   │
                       │  │ Tensor Heap            │   │
                       │  │ (dynamic, 2MB pages)   │   │
                       │  ├────────────────────────┤   │
                       │  │ Model Cache (LRU, 64)  │   │
                       │  ├────────────────────────┤   │
                       │  │ JIT Code Pool (2 MB)   │   │
                       │  │ W^X, max 64 buffers    │   │
                       │  ├────────────────────────┤   │
                       │  │ Git Object Store       │   │
                       │  ├────────────────────────┤   │
                       │  │ IPC Shared Buffers     │   │
                       │  └────────────────────────┘   │
0x0000_0004_0000_0000  └──────────────────────────────┘  (16 GB)
```

Actual heap/cache sizes depend on available RAM (detected from Multiboot1 mmap):
- **8 GB config**: tensor heap ~4992 MB, model cache ~2976 MB
- **4 GB config**: tensor heap ~256 MB, model cache ~512 MB

### Tensor Heap
- Bump allocator with free-list fallback
- Coalesces adjacent free blocks
- 2MB huge page alignment for GPU DMA

### Model Weight Cache
- LRU eviction with 64 entry slots
- Each entry: name hash → physical address + reference count
- Cache hit = instant model load; cache miss = load from TensorFS

### Slab Allocator
8 size classes: 16, 32, 64, 128, 256, 512, 1024, 2048 bytes.
Used for kernel structures (MEU descriptors, scheduler queues, git objects).


## Scheduler Design

```
Priority Queues:
  [REALTIME]  ──→ Safety-critical inference (medical, autonomous)
  [CRITICAL]  ──→ Low-latency serving
  [HIGH]      ──→ Interactive inference
  [NORMAL]    ──→ Batch inference, training
  [LOW]       ──→ Background optimization
  [IDLE]      ──→ Model prefetching, cache warming
```

### Dispatch Policies
- **THROUGHPUT**: Maximize tensor ops/sec (batch-friendly)
- **LATENCY**: Minimize time-to-first-token
- **EFFICIENCY**: Minimize power consumption
- **FAIR**: Equal GPU time across MEUs

### GPU Scoring
When assigning an MEU to a GPU, the scheduler computes:
```
score = (available_VRAM × 4)
      + ((100 - utilization%) × 2)
      + ((100 - temperature°C) × 1)
      + (weight_locality_bonus × 8)
```
Weight locality bonus rewards GPUs that already have the model's weights cached.

### Batch Coalescing
If multiple MEUs request the same operation (e.g., matmul with same shapes),
the scheduler coalesces them into a single batched GPU dispatch for throughput.


## Tensor IR (TIR)

The intermediate representation used by the Pseudocode JIT and tensor engine:

| Opcode | Description |
|--------|-------------|
| TIR_LOAD | Load tensor from memory |
| TIR_STORE | Store tensor to memory |
| TIR_MATMUL | Matrix multiplication |
| TIR_ADD / MUL / DIV / SUB | Elementwise arithmetic |
| TIR_RELU / GELU / SILU / TANH | Activations |
| TIR_SOFTMAX | Softmax normalization |
| TIR_LAYERNORM | Layer normalization |
| TIR_ATTENTION | Fused multi-head attention |
| TIR_CONV2D | 2D convolution |
| TIR_POOL | Pooling (max/avg) |
| TIR_EMBEDDING | Embedding lookup |
| TIR_TRANSPOSE | Tensor transpose |
| TIR_RESHAPE | Tensor reshape |
| TIR_CONCAT | Tensor concatenation |
| TIR_SPLIT | Tensor split |
| TIR_REDUCE | Reduction (sum/mean/max) |
| TIR_CAST | Dtype conversion |
| TIR_ALLOC / FREE | Memory management |
| TIR_BRANCH / CALL / RET | Control flow |

### Optimization Passes
1. **Op Fusion**: MATMUL → ADD (bias) → RELU fused into single kernel
2. **Precision Auto-Downgrade**: FP32 → FP16 for compute, FP32 for accumulation
3. **Dead Code Elimination**: Remove unused tensor computations
4. **Memory Planning**: Static allocation of tensor buffers, reuse across ops


## Security Model

Three sandbox policies:

| Policy | Tensor Ops | GPU | Network | Filesystem | IPC |
|--------|-----------|-----|---------|------------|-----|
| STRICT | Allowed | No | No | No | No |
| STANDARD | Allowed | Allowed | Read-only | Read-only | Allowed |
| PERMISSIVE | Allowed | Allowed | Allowed | Read/Write | Allowed |

- Every MEU has a permission bitmask (11 flags)
- Audit ring buffer logs all security-relevant operations
- Deterministic mode: fixed random seeds, no timing side channels
- Resource accounting: per-MEU memory and compute time tracking


## SMP Architecture

### Bootstrap
- BSP enables LAPIC, copies trampoline to physical `0x8000`
- Sends INIT-SIPI-SIPI to all APs (up to `MAX_CPUS=64`)
- Each AP: enters real mode at `0x8000`, transitions to protected → long mode,
  gets a 65 KB stack, increments `smp.ap_started`, enters idle loop

### Work Dispatch
```
BSP                     AP 0                  AP 1                  AP 2
  │                      │                     │                     │
  ├─ smp_dispatch(fn) ──►│ wake via IPI 0xFE  │                     │
  │                      ├─ execute fn(arg)    │                     │
  ├─ smp_dispatch(fn) ──►│                     ├─ execute fn(arg)    │
  ├─ smp_dispatch(fn) ──►│                     │                     ├─ execute fn(arg)
  │  (BSP does own share)│                     │                     │
  ├─ smp_wait_all() ─────┤─── barrier ─────────┤─── barrier ─────────┤
  │                      │                     │                     │
```

### Parallel GEMV
When `ncpu > 1 && out_dim >= 64`, GEMV rows are partitioned across CPUs:
- CPU `c` processes rows `[c * rows_per_cpu, (c+1) * rows_per_cpu)`
- BSP executes its share directly, APs receive work via `smp_dispatch()`
- All CPUs join via `smp_wait_all()` before the result is used
- Supports both Q4_0 and Q8_0 fused AVX2 GEMV paths


## JIT Compilation

### x86_64 JIT Engine (`x86_jit.c`)
- 2 MB executable code pool with W^X protection
- Max 64 concurrent JIT buffers
- Full x86_64 instruction encoder: REX, ModR/M, SIB, SSE2, AVX2 opcodes
- Register allocator using System V calling convention

### LLM Forward Kernels (`llm_jit.c`)
Lazy-compiled on first inference call. Kernel cache holds up to 32 entries.

| Kernel | Op | Vector Size | Used For |
|--------|----|-------------|----------|
| vadd | a[i] + b[i] | dim (3072) | Residual connections |
| dot | Σ a[i]×b[i] | head_dim (96) | Attention score computation |
| axpy | a[i] + α×b[i] | head_dim (96) | Attention value accumulation |
| fused_silu_mul | silu(a[i])×b[i] | ff_dim (8192) | FFN gate ⊙ up projection |
| rope | Rotary encoding | head_dim (96) | Position encoding for Q/K |
| rmsnorm | RMS normalize | dim (3072) | Layer normalization |

Softmax is not JIT-compiled because sequence length varies per token.
