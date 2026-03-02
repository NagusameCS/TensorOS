<img width="1280" height="640" alt="IRIS-MD" src="https://github.com/user-attachments/assets/3fb51941-b10b-4f92-b482-f0e8ff81d97d" />
# TensorOS


**An AI-First Operating System**

TensorOS is an operating system built from scratch with a single goal: **run AI workloads faster and cheaper, without losing accuracy.** Every layer — from the bootloader to the shell — is designed around tensors, models, and inference as first-class primitives.

Traditional OSes treat AI as just another application. TensorOS treats AI as *the* application.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        AI Shell (aishell)                          │
│  model load llama3 │ infer gpt4 "hello" │ deploy bert --port 8080 │
├───────────┬────────┴───────┬──────────────┬───────────────────────┤
│  Training │   Deployment   │   Monitor    │   Pseudocode JIT      │
│  Service  │   Service      │   Daemon     │   Runtime             │
├───────────┴────────────────┴──────────────┴───────────────────────┤
│                     Tensor Execution Engine                        │
│            (Eager ops, Compute Graphs, Auto-backend)              │
├───────────────────────────────────────────────────────────────────┤
│  Tensor     │  Memory      │  Native   │  Virtual-  │ Security   │
│  Scheduler  │  Manager     │  Git      │  ization   │ Sandbox    │
│  (MEU-based)│  (AI-aware)  │  (SHA-256)│  (VT-x)   │            │
├─────────────┼──────────────┼───────────┼────────────┼────────────┤
│  GPU Driver │  TPU Driver  │  TensorFS │  IPC       │ Model Pkg  │
│  (NVIDIA/   │              │  (AI-aware│  (zero-    │ Manager    │
│   AMD/Intel)│              │   VFS)    │   copy)    │            │
├─────────────┴──────────────┴───────────┴────────────┴────────────┤
│                     Multiboot2 Bootloader                         │
│              (x86_64 Long Mode, Tensor Memory Regions)            │
└───────────────────────────────────────────────────────────────────┘
```

## Key Innovations

### 1. Model Execution Units (MEUs) Replace Processes

Traditional OSes schedule processes and threads. TensorOS schedules **Model Execution Units** — each MEU encapsulates a model with its weights, compute graph, and I/O. The scheduler understands tensor operations and can:

- **Batch-coalesce** similar operations across MEUs
- **GPU-score** devices based on VRAM, utilization, temperature, and weight locality
- **Priority-schedule** with 6 levels: REALTIME → CRITICAL → HIGH → NORMAL → LOW → IDLE

### 2. Tensor-Aware Memory Manager

Memory is organized into zones optimized for AI:

| Zone | Purpose | Page Size |
|------|---------|-----------|
| `TENSOR` | Active tensor computation | 2MB huge pages |
| `MODEL` | Model weight cache (LRU) | 2MB huge pages |
| `DMA` | GPU/TPU DMA transfers | 4KB, pinned |
| `GIT` | Git object store | 4KB |
| `KERNEL` | Kernel data structures | Slab allocator |

The model weight cache uses LRU eviction with 64 entries, so switching between models is near-instant when weights are already cached.

### 3. Native Kernel-Level Git

Git is not an application — it's a kernel subsystem. Benefits:

- SHA-256 (not SHA-1) computed in kernel space
- Extended object types: `GIT_OBJ_TENSOR` and `GIT_OBJ_MODEL` for native versioning of tensors and model checkpoints
- Training runs automatically create git commits at checkpoint intervals
- Zero-copy: git objects share memory with the tensor heap

### 4. Pseudocode as Default Language

The default runtime uses **Pseudocode** (inspired by [NaguSamecs' Pseudocode](https://github.com/NaguSamecs/Pseudocode)), a language designed to look like natural algorithmic descriptions but compile to efficient tensor operations:

```pseudocode
model transformer:
    layer attention(Q, K, V):
        scores = matmul(Q, transpose(K))
        weights = softmax(scores / sqrt(dim))
        return matmul(weights, V)

    layer feedforward(x):
        h = relu(matmul(x, W1) + b1)
        return matmul(h, W2) + b2

load "llama-3-8b" as llm
result = infer llm with "Explain quantum computing"
print result

train llm on "dataset.jsonl":
    epochs = 3
    learning_rate = 0.0003
    optimizer = adamw
    save every 500 steps

deploy llm on port 8080

git commit "trained llama-3 on custom data"
```

The Pseudocode runtime includes:
- **60+ token types** with AI-specific keywords (`model`, `layer`, `tensor`, `train`, `infer`, `deploy`)
- **Recursive descent parser** producing an AST
- **Tensor IR** with 28 opcodes (MATMUL, CONV2D, ATTENTION, SOFTMAX, etc.)
- **4-tier JIT**: Interpreter → Basic compilation → Optimized → Fully optimized
- **Optimization passes**: Op fusion (matmul+bias+relu), precision auto-downgrade (FP32→FP16)

### 5. Near-Zero-Cost Virtualization

VT-x/AMD-V with EPT/NPT for hardware-accelerated containers:

- **Paravirtualized tensor hypercalls** — guest VMs can request tensor operations from the host without full device emulation
- **IOMMU GPU passthrough** — direct GPU access for containers with near-native performance
- **Shared tensor memory** — containers share a mapped memory region for zero-copy tensor transfer

### 6. Model Package Manager

Like apt/npm but for AI models:

```
tensor> pkg install llama-3-8b
[PKG] Resolving llama-3-8b from tensoros-hub...
[PKG] Downloading: llama-3-8b (4.5 GB, Q4_K_M quantized)
[PKG] Verifying SHA-256...
[PKG] Installing to /models/llama-3-8b/
[PKG] Auto-optimizing for detected hardware (NVIDIA RTX 4090)...
[PKG] Done.

tensor> pkg search "code generation"
Found 12 packages:
  codellama-34b     34B params  Code generation  ★★★★★
  starcoder2-15b    15B params  Code generation  ★★★★☆
  deepseek-coder-v2 16B params  Code + reasoning ★★★★★
```

Registries: `tensoros-hub` (default), `huggingface`.
Supports automatic quantization and hardware-specific optimization on install.

---

## Building

### Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| `x86_64-elf-gcc` | Cross-compiler | See [OSDev GCC Cross-Compiler](https://wiki.osdev.org/GCC_Cross-Compiler) |
| `nasm` | Assembler | `apt install nasm` / `choco install nasm` |
| `qemu-system-x86_64` | Emulator | `apt install qemu-system-x86` / `choco install qemu` |
| `grub-mkrescue` | ISO builder (optional) | `apt install grub-pc-bin xorriso` |
| `gdb` | Debugger (optional) | `apt install gdb` |

### Build & Run

```bash
# Build the kernel
make

# Run in QEMU (4GB RAM, 4 CPUs, virtio-gpu)
make run

# Build bootable ISO
make iso

# Debug with GDB
make debug
# In another terminal:
gdb -x .gdbinit build/tensoros.bin
```

### Windows (PowerShell)

```powershell
# Run in QEMU
.\scripts\run-qemu.ps1

# Debug mode
.\scripts\run-qemu.ps1 -Debug

# Boot from ISO
.\scripts\run-qemu.ps1 -Iso
```

---

## Project Structure

```
TensorOS/
├── boot/
│   ├── boot.asm              # Multiboot2 bootloader (x86_64 long mode)
│   └── linker.ld             # Linker script with tensor memory regions
├── kernel/
│   ├── core/
│   │   ├── kernel.h          # Core types (tensor_desc_t, MEU, kernel_state)
│   │   └── main.c            # Kernel entry, 4-phase boot
│   ├── sched/
│   │   ├── tensor_sched.h    # Tensor-aware scheduler
│   │   └── tensor_sched.c    # MEU scheduling, GPU scoring, batch coalescing
│   ├── mm/
│   │   ├── tensor_mm.h       # Memory manager
│   │   └── tensor_mm.c       # Tensor heap, model cache, slab allocator
│   ├── drivers/
│   │   ├── gpu/
│   │   │   ├── gpu.h         # GPU driver interface
│   │   │   └── gpu.c         # PCI detection, tensor op dispatch
│   │   └── tpu/
│   │       ├── tpu.h         # TPU driver interface
│   │       └── tpu.c         # TPU driver stub
│   ├── fs/
│   │   ├── git.h             # Native git (SHA-256, tensor objects)
│   │   ├── git.c             # Git implementation
│   │   ├── tensorfs.h        # AI-aware filesystem
│   │   └── tensorfs.c        # TensorFS implementation
│   ├── security/
│   │   ├── sandbox.h         # Security sandbox
│   │   └── sandbox.c         # Permissions, audit, deterministic mode
│   └── ipc/
│       ├── tensor_ipc.h      # Inter-MEU communication
│       └── tensor_ipc.c      # Zero-copy channels, pipelines
├── virt/
│   ├── virt.h                # Virtualization (VT-x, containers)
│   └── virt.c                # EPT/NPT, GPU passthrough, hypercalls
├── runtime/
│   ├── pseudocode/
│   │   ├── pseudocode_jit.h  # Pseudocode language runtime
│   │   └── pseudocode_jit.c  # Lexer, parser, IR, optimizer, interpreter
│   └── tensor/
│       ├── tensor_engine.h   # Tensor execution engine
│       └── tensor_engine.c   # Eager ops, compute graphs, backend selection
├── pkg/
│   ├── modelpkg.h            # Model package manager
│   └── modelpkg.c            # Registry, install, quantize, verify
├── userland/
│   ├── shell/
│   │   ├── aishell.h         # AI Shell
│   │   └── aishell.c         # Interactive shell with AI commands
│   ├── monitor/
│   │   ├── tensor_monitor.h  # System monitor daemon
│   │   └── tensor_monitor.c  # GPU/memory/MEU monitoring, alerts
│   ├── deploy/
│   │   ├── deploy_service.h  # Model deployment service
│   │   └── deploy_service.c  # Auto-scaling, health checks, A/B testing
│   └── train/
│       ├── train_service.h   # Training service
│       └── train_service.c   # Distributed training, checkpointing, LR scheduling
├── scripts/
│   ├── run-qemu.sh           # QEMU launcher (Linux/macOS)
│   └── run-qemu.ps1          # QEMU launcher (Windows)
├── Makefile                   # Build system
├── .gdbinit                   # GDB configuration
└── README.md                  # This file
```

---

## AI Shell Quick Reference

```
tensor> model load llama-3-8b          # Load model into MEU
tensor> model list                      # Show running MEUs
tensor> infer llama-3-8b "Hello"        # Run inference
tensor> train bert dataset.json         # Launch training
tensor> deploy llama-3-8b --port 8080   # Deploy as service
tensor> git init                        # Initialize git repo
tensor> git commit -m "checkpoint"      # Commit state
tensor> pkg install mistral-7b          # Install model
tensor> monitor                         # System dashboard
tensor> run script.pseudo               # Execute Pseudocode file
tensor> help                            # Full command list
```

Any text that isn't a built-in command is automatically JIT-compiled as Pseudocode.

---

## Design Principles

1. **Tensors are first-class** — Memory, scheduling, IPC, and filesystems all understand tensor shapes and dtypes natively.

2. **Models are the unit of execution** — No processes, threads, or PIDs. Everything is an MEU with a model, weights, and a compute graph.

3. **Zero-copy everywhere** — IPC uses shared memory, git objects live in the tensor heap, GPU passthrough avoids host copies.

4. **Git is infrastructure** — Every training run, deployment, and model change is automatically version-controlled at the kernel level.

5. **Hardware-aware by default** — The scheduler, memory manager, and package manager all auto-optimize for detected hardware (GPU VRAM, tensor cores, thermal limits).

6. **Pseudocode is the interface** — Write what you mean, not how the machine wants it. The JIT figures out the rest.

---

## Roadmap

- [ ] Interrupt handler (IDT, PIC/APIC, timer)
- [ ] Full PS/2 keyboard driver with scancode set 2
- [ ] PCI Express enumeration for modern GPUs
- [ ] NVIDIA GPU driver (MMIO register interface)
- [ ] AMD ROCm-compatible GPU driver
- [ ] Real DMA engine for PCIe transfers
- [ ] Network stack (TCP/IP for model serving)
- [ ] Distributed training across multiple machines
- [ ] UEFI boot support
- [ ] Filesystem persistence (disk I/O)
- [ ] Pseudocode standard library
- [ ] WebGPU/Vulkan compute backend
- [ ] ONNX Runtime integration
- [ ] safetensors / GGUF native loaders
- [ ] Model quantization engine (GPTQ, AWQ, GGML)
- [ ] Speculative decoding support
- [ ] Flash Attention kernel
- [ ] PagedAttention (vLLM-style) for serving

---

## License

TensorOS is an experimental research OS. Use at your own risk. Run only in QEMU or a virtual machine.

---

*Built for a world where AI is not an app — it's the operating system.*
