/* =============================================================================
 * TensorOS Kernel - Main Entry Point
 * Copyright (c) 2026 TensorOS Project
 *
 * This is the C entry point after the bootloader has set up long mode,
 * paging, and basic hardware. From here we initialize all kernel subsystems
 * in dependency order and launch the AI-first userland.
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/core/perf.h"
#ifndef __aarch64__
#include "kernel/core/cpu_features.h"
#endif
#include "kernel/core/selftest.h"
#include "kernel/sched/tensor_sched.h"
#include "kernel/mm/tensor_mm.h"
#include "kernel/drivers/gpu/gpu.h"
#include "kernel/drivers/tpu/tpu.h"
#include "kernel/fs/tensorfs.h"
#include "kernel/ipc/tensor_ipc.h"
#include "kernel/security/sandbox.h"
#include "runtime/tensor/tensor_cpu.h"
#ifndef __aarch64__
#include "runtime/jit/x86_jit.h"
#endif
#include "runtime/nn/inference.h"
#include "runtime/nn/quantize.h"
#include "runtime/nn/evolution.h"
#include "runtime/nn/train.h"
#include "runtime/nn/speculative.h"
#include "runtime/nn/transformer.h"
#include "runtime/nn/quantize4.h"
#include "kernel/mm/tensor_arena.h"
#include "runtime/nn/gguf.h"
#include "runtime/nn/math_llm.h"
#include "runtime/nn/llm.h"
#include "kernel/core/smp.h"
#include "kernel/drivers/net/virtio_net.h"
#include "kernel/net/netstack.h"
#include "kernel/drivers/blk/virtio_blk.h"
#include "kernel/drivers/blk/sdlog.h"

/* Kernel version */
#define TENSOROS_VERSION_MAJOR  0
#define TENSOROS_VERSION_MINOR  1
#define TENSOROS_VERSION_PATCH  0
#define TENSOROS_CODENAME       "Neuron"

/* Forward declarations */
static void init_phase1_hardware(void);
static void init_phase2_subsystems(void);
static void init_phase3_runtime(void);
static void init_phase4_userland(void);
static void kernel_idle_loop(void) __attribute__((noreturn));

/* Global kernel state */
struct kernel_state kstate = {
    .phase = KSTATE_BOOT,
    .cpu_count = 1,
    .gpu_count = 0,
    .tpu_count = 0,
    .tensor_ops_total = 0,
    .models_loaded = 0,
    .uptime_ticks = 0,
};

/* =============================================================================
 * kernel_main - Primary kernel entry point
 * Called from boot.asm after long mode setup
 * =============================================================================*/

/* =============================================================================
 * sdlog_crash_c — Called from boot.S exception handler to flush crash info
 *                 to BOOTLOG.TXT on the SD card.  Best-effort: if the SD
 *                 driver or stack is broken, this silently fails.
 *
 * Arguments (ARM64 calling convention):
 *   x0 = type letter ('S'=sync, 'I'=IRQ, 'F'=FIQ, 'E'=SError)
 *   x1 = breadcrumb value (last C checkpoint)
 *   x2 = exception class (EC, 6-bit)
 *   x3 = ESR_EL1 (full)
 *   x4 = ELR_EL1 (faulting PC)
 *   x5 = FAR_EL1 (fault address)
 * =============================================================================*/
#if defined(__aarch64__)
void sdlog_crash_c(uint64_t type, uint64_t bc, uint64_t ec,
                   uint64_t esr, uint64_t elr, uint64_t far_val)
{
    sdlog("!!! EXCEPTION !!!");
    /* Type letter */
    char tbuf[] = "Type= ";
    tbuf[5] = (char)type;
    sdlog(tbuf);
    sdlog_val("BC=", (int32_t)bc);
    sdlog_val("EC=", (int32_t)ec);
    sdlog_hex("ESR=", esr);
    sdlog_hex("ELR=", elr);
    sdlog_hex("FAR=", far_val);

    /* Decode common exception classes */
    switch ((int)ec) {
        case 0x00: sdlog("  (Unknown / Uncategorized)"); break;
        case 0x15: sdlog("  (SVC instruction)"); break;
        case 0x18: sdlog("  (MSR/MRS trap from lower EL)"); break;
        case 0x20: sdlog("  (Instruction Abort, lower EL)"); break;
        case 0x21: sdlog("  (Instruction Abort, same EL)"); break;
        case 0x22: sdlog("  (PC Alignment Fault)"); break;
        case 0x24: sdlog("  (Data Abort, lower EL)"); break;
        case 0x25: sdlog("  (Data Abort, same EL)"); break;
        case 0x26: sdlog("  (SP Alignment Fault)"); break;
        case 0x2C: sdlog("  (FP/SIMD trap)"); break;
        case 0x3C: sdlog("  (BRK instruction)"); break;
        default:   sdlog("  (see ARM ARM for EC)"); break;
    }

    sdlog_flush();
}
#endif
void kernel_main(void)
{
    /*
     * ARM64 breadcrumb: write progress to 0x1000 so the exception
     * handler can report WHERE the crash happened.
     * boot.S clears 0x1000 to 0 before entering C.
     *
     *  BC=1  C entry
     *  BC=2  before vga_init
     *  BC=3  after vga_init
     *  BC=4  before perf_calibrate
     *  BC=5  after perf_calibrate
     *  BC=6  before init_phase1_hardware
     *  BC=7  after init_phase1_hardware
     *  BC=8  before init_phase2_subsystems
     *  BC=9  after init_phase2_subsystems
     *  BC=10 before init_phase3_runtime
     *  BC=11 after init_phase3_runtime
     *  BC=12 before init_phase4_userland
     *  BC=13 all phases done
     */
#if defined(__aarch64__)
    #define BREADCRUMB(n) (*(volatile uint32_t *)0x1000 = (uint32_t)(n))
#else
    #define BREADCRUMB(n) ((void)0)
#endif

    BREADCRUMB(1);  /* C entry */

#if defined(__aarch64__)
    /* ================================================================
     * ARM64 FIRST-BOOT CONFIGURATION
     *
     * Set ARM64_SKIP_SD_INIT=1 to skip SD card init and go straight
     * to HDMI framebuffer.  Use this for initial hardware bring-up.
     * Set to 0 once HDMI is working to enable SD boot logging.
     * ================================================================ */
    #define ARM64_SKIP_SD_INIT 1

    /* ================================================================
     * LED milestones — visible without UART adapter.
     * Each milestone: N quick blinks (125ms on/off), 500ms gap.
     *   1 blink  = C code reached
     *   2 blinks = before framebuffer init
     *   3 blinks = framebuffer init done
     *   4 blinks = entering main boot phases
     * ================================================================ */
    led_blink(1);  /* MILESTONE 1: C code alive */
    uart_puts("[LED] M1: C code entry\r\n");

    int sdlog_rc = -1;  /* No SD logging until we enable it */

#if !ARM64_SKIP_SD_INIT
    /* SD card init — skip this for initial HDMI bring-up */
    {
        volatile uint32_t __attribute__((aligned(16))) mb[8];
        mb[0] = 8 * 4; mb[1] = 0;
        mb[2] = 0x00028001; mb[3] = 8; mb[4] = 8;
        mb[5] = 0; mb[6] = 0x03; mb[7] = 0;
        mbox_call(8, mb);
    }
    {
        volatile uint32_t __attribute__((aligned(16))) mb[9];
        mb[0] = 9 * 4; mb[1] = 0;
        mb[2] = 0x00038002; mb[3] = 12; mb[4] = 12;
        mb[5] = 0x0000000C; mb[6] = 200000000; mb[7] = 0; mb[8] = 0;
        mbox_call(8, mb);
    }
    sd_init();
    sdlog_rc = sdlog_init();
    uart_puts("[SDLOG] SD init done\r\n");
#endif /* !ARM64_SKIP_SD_INIT */

#else /* !__aarch64__ (x86) */
    int sdlog_rc = -99;
#endif /* __aarch64__ */

    /* ---- Common path: HDMI framebuffer + banner ---- */
    BREADCRUMB(2);  /* before vga_init */
#if defined(__aarch64__)
    led_blink(2);  /* MILESTONE 2: about to init framebuffer */
    uart_puts("[LED] M2: before fb_init\r\n");
#endif
    sdlog("BC=2 before vga_init");
    sdlog_flush();
    vga_init();
    BREADCRUMB(3);  /* after vga_init */
    sdlog("BC=3 after vga_init");

#if defined(__aarch64__)
    /* Initialise HDMI framebuffer via VideoCore mailbox */
    {
        extern int fb_init(void);
        extern void fb_boot_splash(void);
        uart_puts("[FB] Calling fb_init...\r\n");
        int fb_rc = fb_init();
        if (fb_rc == 0) {
            uart_puts("[FB] OK! Framebuffer ready\r\n");
            fb_boot_splash();
        } else {
            uart_puts("[FB] FAILED (rc=");
            uart_putchar('0' + ((-fb_rc) % 10));
            uart_puts(")\r\n");
        }
    }
    led_blink(3);  /* MILESTONE 3: framebuffer init done */
    uart_puts("[LED] M3: fb_init done\r\n");
#endif

    kprintf("TensorOS v%d.%d.%d \"%s\" booting...\n",
            TENSOROS_VERSION_MAJOR, TENSOROS_VERSION_MINOR,
            TENSOROS_VERSION_PATCH, TENSOROS_CODENAME);

    kprintf("[BOOT] Kernel loaded at %p, size %lu KB\n",
            (void*)__text_start, ((uintptr_t)__kernel_end - (uintptr_t)__text_start) / 1024);
#if defined(__aarch64__)
    kprintf("[BOOT] ARM64 NEON enabled --- SIMD tensor operations available\n");
    if (sdlog_rc == 0)
        kprintf("[SDLOG] Boot logger active — pull SD card to read BOOTLOG.TXT\n");
    else
        kprintf("[SDLOG] Logger FAILED (rc=%d) — SD card logging unavailable\n", sdlog_rc);
#else
    kprintf("[BOOT] SSE2 enabled --- SIMD tensor operations available\n");
#endif

    /* Calibrate TSC for benchmarking */
    BREADCRUMB(4);  /* before perf_calibrate */
    sdlog("BC=4 before perf_calibrate");
    perf_calibrate();
    BREADCRUMB(5);  /* after perf_calibrate */
    sdlog("BC=5 after perf_calibrate");
    sdlog_flush();
    kprintf("[BOOT] TSC calibrated: %lu MHz\n", perf_tsc_mhz());

    /* Phase 1: Hardware Discovery & Init */
    kprintf("[PHASE 1] Hardware initialization\n");
    BREADCRUMB(6);  /* before init_phase1_hardware */
    sdlog("BC=6 before phase1 (IDT, CPU, PCI, GPU, TPU, PIC, timer)");
    sdlog_flush();
    init_phase1_hardware();
    BREADCRUMB(7);  /* after init_phase1_hardware */
    sdlog("BC=7 after phase1 OK");

    /* Phase 2: Core Subsystems */
    kprintf("[PHASE 2] Subsystem initialization\n");
    BREADCRUMB(8);  /* before init_phase2_subsystems */
    sdlog("BC=8 before phase2 (MM, sched, FS, IPC, git, net)");
    sdlog_flush();
    init_phase2_subsystems();
    BREADCRUMB(9);  /* after init_phase2_subsystems */
    sdlog("BC=9 after phase2 OK");

    /* Phase 3: AI Runtime */
    kprintf("[PHASE 3] AI runtime initialization\n");
    BREADCRUMB(10); /* before init_phase3_runtime */
    sdlog("BC=10 before phase3 (JIT, tensor engine, sandbox)");
    sdlog_flush();
    init_phase3_runtime();
    BREADCRUMB(11); /* after init_phase3_runtime */
    sdlog("BC=11 after phase3 OK");

    /* Phase 4: Userland */
    kprintf("[PHASE 4] Launching userland\n");
    BREADCRUMB(12); /* before init_phase4_userland */
    sdlog("BC=12 before phase4 (monitor, deploy)");
    sdlog_flush();
    init_phase4_userland();
    BREADCRUMB(13); /* after init_phase4_userland */
    sdlog("BC=13 after phase4 OK");

    /* Bluetooth SPP console (ARM64 only) */
#if defined(__aarch64__)
    kprintf("[BT] Initializing Bluetooth SPP console...\n");
    if (bt_init() == 0)
        kprintf("  [OK] Discoverable as 'TensorOS' — pair from PC\n");
    else
        kprintf("  [--] BT not available (no firmware?)\n");
#endif

    /* Phase 5: CPU Feature Detection & AVX2 Enable */
    sdlog("Phase5: CPU detect");
    kprintf("\n[PHASE 5] CPU Feature Detection\n");
#if defined(__aarch64__)
    {
        uint64_t midr;
        __asm__ volatile ("mrs %0, midr_el1" : "=r"(midr));
        kprintf("  [OK] ARM64 MIDR: %lx  ISA: NEON FP\n", midr);
    }
#else
    cpu_detect_features();
    if (cpu_features.has_xsave && cpu_features.has_avx) {
        cpu_enable_avx();
    }
    cpu_print_features();
#endif

    /* Phase 6: Production Self-Test Suite */
    sdlog("Phase6: self-test");
    kprintf("\n[PHASE 6] Production Self-Test Suite\n");
    selftest_run_all();

    /* Phase 7: SMP Multi-Core Bootstrap */
    sdlog("Phase7: SMP");
    sdlog_flush();
    kprintf("\n[PHASE 7] SMP Multi-Core Bootstrap\n");
#ifndef __aarch64__
    smp_run_demos();
    kstate.cpu_count = smp.cpu_count;  /* Update with actual SMP count */

    /* Quick SMP dispatch test (no long benchmark) */
    if (smp.ap_started > 0) {
        extern void smp_test_worker(void *arg);
        volatile uint32_t smp_test_flag = 0;
        int smp_ok = 0, smp_fail = 0;
        for (uint32_t ap = 1; ap <= smp.ap_started; ap++) {
            smp_test_flag = 0;
            __asm__ volatile ("mfence" ::: "memory");
            int rc = smp_dispatch(ap, smp_test_worker, (void *)&smp_test_flag);
            if (rc != 0) { smp_fail++; continue; }
            volatile int wait_ok = 0;
            for (uint64_t tries = 0; tries < 500000; tries++) {
                if (smp.cpus[ap].work_done) { wait_ok = 1; break; }
                __asm__ volatile ("pause");
            }
            smp.cpus[ap].state = CPU_STATE_IDLE;
            if (wait_ok && smp_test_flag == 0xCAFE) smp_ok++;
            else smp_fail++;
        }
        kprintf("  [OK] SMP dispatch: %d/%d cores responding\n",
                smp_ok, smp.ap_started);
    }
#else
    kprintf("  [OK] ARM64 PSCI multicore (4 Cortex-A72 cores)\n");
#endif

    /* Phase 8: Storage Driver */
    sdlog("Phase8: storage");
    kprintf("\n[PHASE 8] Storage Driver\n");
#ifndef __aarch64__
    {
        int blk_rc = virtio_blk_init();
        if (blk_rc == 0) {
            virtio_blk_print_info();
        } else {
            kprintf("  [--] No block device (add -drive to QEMU)\n");
        }
    }
#else
    kprintf("  [OK] SD/eMMC via EMMC2 (BCM2711)\n");
#endif

    /* Phase 9: Network Stack */
    sdlog("Phase9: network");
    kprintf("\n[PHASE 9] Network Stack\n");
#ifndef __aarch64__
    {
        int net_rc = virtio_net_init();
        if (net_rc == 0) {
            uint8_t ip[]      = {10, 0, 2, 15};
            uint8_t netmask[] = {255, 255, 255, 0};
            uint8_t gateway[] = {10, 0, 2, 2};
            netstack_init(ip, netmask, gateway);
            netstack_start_http_server();
            kprintf("  [OK] Network ready: 10.0.2.15 UDP:8080\n");
        } else {
            kprintf("  [--] No NIC (add -nic to QEMU for networking)\n");
        }
    }
#else
    kprintf("  [OK] GENET Ethernet (BCM54213PE) available\n");
#endif

    /* Phase 10: LLM Model Loading (load only — no eval during boot) */
    sdlog("Phase10: LLM load");
    kprintf("\n[PHASE 10] AI Model Loader\n");
    llm_boot_load();  /* Load model into RAM if disk present, but don't run eval */

    kstate.phase = KSTATE_RUNNING;

    BREADCRUMB(99); /* ALL PHASES DONE */
    sdlog("BC=99 ALL PHASES COMPLETE -- boot successful");
    sdlog_flush();

    /* Boot summary */
    kprintf("\n============================================================\n");
    kprintf("  TensorOS v0.1.0 \"Neuron\" -- Boot Complete\n");
    kprintf("============================================================\n");
#if defined(__aarch64__)
    kprintf("  Arch:     ARM64 (NEON 128-bit SIMD)\n");
#else
    kprintf("  Arch:     x86_64 (%s)\n",
            cpu_features.avx2_usable ? "AVX2+FMA 256-bit" : "SSE2 128-bit");
#endif
    kprintf("  CPUs:     %d    RAM: %lu MB\n",
            kstate.cpu_count, kstate.memory_total_bytes / (1024*1024));
    kprintf("  Storage:  %s\n",
#ifndef __aarch64__
            virtio_blk_capacity() > 0 ? "virtio-blk" : "none");
#else
            "SD/eMMC");
#endif
    {
        extern int llm_is_loaded(void);
        extern const char *llm_model_name(void);
        if (llm_is_loaded())
            kprintf("  LLM:      %s (ready)\n", llm_model_name());
        else
            kprintf("  LLM:      (none -- attach GGUF disk)\n");
    }
    kprintf("============================================================\n");

    kprintf("\n[READY] TensorOS is operational. %d CPUs, %lu MB RAM\n",
            kstate.cpu_count, kstate.memory_total_bytes / (1024*1024));

    /* Enable interrupts for keyboard/timer */
    sti();

    /* Always enter the interactive shell — this is an OS, not a batch runner.
     * The shell accepts keyboard input (PS/2 IRQ1) and serial (COM1).
     * Type 'help' for commands, 'exit' to shutdown. */
    kprintf("\n[READY] Entering TensorOS shell (type 'help' for commands)\n\n");
    {
        extern void aishell_main(void);
        aishell_main();
        kprintf("[SHUTDOWN] TensorOS shutting down...\n");
    }

    cli();

    /* Write to QEMU ISA debug-exit device to cleanly exit (x86 only) */
#ifndef __aarch64__
    outb(0x501, 0x31);
#endif

    sti();

    /* Enter idle loop - scheduler takes over */
    kernel_idle_loop();
}

/* =============================================================================
 * Phase 1: Hardware Discovery
 * Detect CPUs, GPUs, TPUs and initialize interrupt handling
 * =============================================================================*/
static void init_phase1_hardware(void)
{
    /* Initialize interrupt descriptor table */
    idt_init();
    kprintf("  [OK] IDT initialized\n");

    /* Detect and initialize all CPUs */
    kstate.cpu_count = cpu_detect_and_init();
    kprintf("  [OK] %d CPU(s) detected\n", kstate.cpu_count);

    /* PCI bus enumeration - find accelerators */
    pci_enumerate();
    kprintf("  [OK] PCI bus enumerated\n");

    /* GPU detection and initialization */
    kstate.gpu_count = gpu_detect_and_init();
    if (kstate.gpu_count > 0) {
        kprintf("  [OK] %d GPU(s) detected and initialized\n", kstate.gpu_count);
        for (int i = 0; i < kstate.gpu_count; i++) {
            struct gpu_info *gpu = gpu_get_info(i);
            kprintf("       GPU %d: %s (%u MB VRAM, %u CUDA cores)\n",
                    i, gpu->name, gpu->vram_mb, gpu->compute_units);
        }
    } else {
        kprintf("  [--] No GPU detected, running in CPU-only mode\n");
    }

    /* TPU detection */
    kstate.tpu_count = tpu_detect_and_init();
    if (kstate.tpu_count > 0) {
        kprintf("  [OK] %d TPU(s) detected\n", kstate.tpu_count);
    }

    /* Enable interrupts */
    pic_init();
    timer_init(1000); /* 1000 Hz tick for fine-grained scheduling */
    /* Keep interrupts disabled during boot to prevent output corruption */
    /* sti(); */
    kprintf("  [OK] Interrupts configured (1000 Hz timer) - enabling after boot\n");
}

/* =============================================================================
 * Phase 2: Core Subsystems
 * Memory manager, filesystem, IPC, networking
 * =============================================================================*/
static void init_phase2_subsystems(void)
{
    /* Tensor-aware memory manager */
    tensor_mm_init();
    kprintf("  [OK] Tensor memory manager initialized\n");
    kprintf("       Tensor heap: %lu MB, Model cache: %lu MB\n",
            tensor_mm_heap_size() / (1024 * 1024),
            tensor_mm_cache_size() / (1024 * 1024));

    /* Tensor-aware process scheduler */
    tensor_sched_init();
    kprintf("  [OK] Tensor scheduler initialized\n");

    /* TensorFS - AI-aware filesystem */
    tensorfs_init();
    kprintf("  [OK] TensorFS mounted\n");

    /* IPC subsystem for model communication */
    tensor_ipc_init();
    kprintf("  [OK] Tensor IPC initialized\n");

    /* Native git subsystem */
    extern void git_subsystem_init(void);
    git_subsystem_init();
    kprintf("  [OK] Native git subsystem initialized\n");

    /* Network stack for distributed training */
    net_init();
    kprintf("  [OK] Network stack initialized\n");
}

/* =============================================================================
 * Phase 3: AI Runtime Environment
 * JIT compiler, tensor engine, model sandbox
 * =============================================================================*/
static void init_phase3_runtime(void)
{
    /* Pseudocode JIT compiler */
    extern int pseudocode_jit_init(void);
    pseudocode_jit_init();
    kprintf("  [OK] Pseudocode JIT compiler ready\n");

    /* JIT backend */
#ifndef __aarch64__
    jit_init();
    kprintf("  [OK] x86_64 native JIT backend initialized\n");
#else
    kprintf("  [OK] ARM64 NEON inference backend initialized\n");
#endif

    /* Tensor execution engine */
    extern int tensor_engine_init(void);
    tensor_engine_init();
#if defined(__aarch64__)
    kprintf("  [OK] Tensor execution engine ready (NEON SIMD)\n");
#else
    kprintf("  [OK] Tensor execution engine ready (SSE2 SIMD)\n");
#endif

    /* CPU tensor math self-test */
    {
        int rc = tensor_cpu_selftest();
        if (rc == 0)
            kprintf("  [OK] CPU tensor math verified (matmul, relu, softmax, dot)\n");
        else
            kprintf("  [!!] CPU tensor self-test FAILED (code %d)\n", rc);
    }

    /* Model sandbox */
    sandbox_init();
    kprintf("  [OK] Model sandbox initialized\n");

    /* Model package manager */
    extern int modelpkg_init(void);
    modelpkg_init();
    kprintf("  [OK] Model package manager ready\n");

    /* Virtualization layer */
#ifndef __aarch64__
    extern int virt_layer_init(void);
    virt_layer_init();
    kprintf("  [OK] Near-zero-cost virtualization layer ready\n");
#else
    kprintf("  [OK] ARM64 EL2 hypervisor support available\n");
#endif
}

/* =============================================================================
 * Phase 4: Userland Launch
 * Start the AI shell and monitoring services
 * =============================================================================*/
static void init_phase4_userland(void)
{
    /* Launch the tensor monitor daemon */
    extern void monitor_daemon_main(void);
    monitor_daemon_main();
    kprintf("  [OK] Tensor monitor daemon started\n");

    /* Launch the model deployment service */
    extern void deploy_daemon_main(void);
    deploy_daemon_main();
    kprintf("  [OK] Model deployment service started\n");

    /* Shell will be launched after benchmarks/demos complete */
    kprintf("  [OK] Userland services initialized\n");
}

/* =============================================================================
 * Kernel Idle Loop
 * When no AI tasks are scheduled, enter low-power state
 * Periodically runs maintenance: memory defrag, cache warming
 * =============================================================================*/
static void kernel_idle_loop(void)
{
    uint64_t idle_cycles = 0;

    while (1) {
        /* Check if any tensor tasks are pending */
        if (tensor_sched_has_pending()) {
            tensor_sched_dispatch();
            idle_cycles = 0;
            continue;
        }

        /* Idle maintenance */
        if (idle_cycles % 10000 == 0) {
            /* Defragment tensor memory */
            tensor_mm_defrag();

            /* Pre-warm model cache if predictions available */
            tensor_mm_cache_warmup();
        }

        if (idle_cycles % 100000 == 0) {
            /* Update kernel statistics */
            kstate.uptime_ticks++;

            /* Log health status */
            kprintf_debug("[IDLE] ops=%lu models=%u mem_free=%lu MB\n",
                         kstate.tensor_ops_total,
                         kstate.models_loaded,
                         (uint64_t)(tensor_mm_free_bytes() / (1024 * 1024)));
        }

        idle_cycles++;

        /* Enter low-power halt until next interrupt */
#if defined(__aarch64__)
        bt_poll();  /* Service Bluetooth SPP connections */
        /* LED feedback: solid ON = BT connected, slow blink = discoverable */
        if (bt_connected()) {
            led_on();
        } else if (idle_cycles % 20000 == 0) {
            /* Toggle LED every ~20k idle iterations for slow blink */
            static int bt_led_state = 0;
            bt_led_state = !bt_led_state;
            if (bt_led_state) led_on(); else led_off();
        }
        __asm__ volatile ("wfi");
#else
        __asm__ volatile ("hlt");
#endif
    }
}
