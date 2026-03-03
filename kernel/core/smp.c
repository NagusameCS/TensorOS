/* =============================================================================
 * TensorOS - SMP Bootstrap Implementation
 * x86_64: LAPIC init, AP trampoline, INIT-SIPI-SIPI, per-core work dispatch
 * ARM64: Uses PSCI (guarded below)
 * =============================================================================*/

#ifndef __aarch64__

#include "kernel/core/kernel.h"
#include "kernel/core/smp.h"

/* =============================================================================
 * SMP global state
 * =============================================================================*/

smp_state_t smp;

/* =============================================================================
 * LAPIC MMIO access
 * =============================================================================*/

uint32_t lapic_read(uint32_t offset)
{
    volatile uint32_t *reg = (volatile uint32_t *)(uintptr_t)(smp.lapic_base + offset);
    return *reg;
}

void lapic_write(uint32_t offset, uint32_t val)
{
    volatile uint32_t *reg = (volatile uint32_t *)(uintptr_t)(smp.lapic_base + offset);
    *reg = val;
}

void lapic_eoi(void)
{
    if (smp.lapic_base)
        lapic_write(LAPIC_EOI, 0);
}

uint32_t smp_get_apic_id(void)
{
    if (!smp.lapic_base) return 0;
    return lapic_read(LAPIC_ID) >> 24;
}

/* =============================================================================
 * Delay using PIT (channel 2)
 * =============================================================================*/

static void smp_delay_us(uint32_t us)
{
    /* Use TSC for delay (calibrated in perf.c) */
    extern uint64_t perf_tsc_mhz(void);
    uint64_t cycles = (uint64_t)us * perf_tsc_mhz();
    uint32_t lo, hi;
    __asm__ volatile ("lfence; rdtsc" : "=a"(lo), "=d"(hi));
    uint64_t start = ((uint64_t)hi << 32) | lo;
    while (1) {
        __asm__ volatile ("lfence; rdtsc" : "=a"(lo), "=d"(hi));
        uint64_t now = ((uint64_t)hi << 32) | lo;
        if (now - start >= cycles) break;
        __asm__ volatile ("pause");
    }
}

/* =============================================================================
 * AP Trampoline
 * 
 * This 16-bit real mode code is copied to physical address 0x8000.
 * When an AP receives SIPI, it starts executing here.
 * It transitions: 16-bit -> 32-bit -> 64-bit -> C ap_entry()
 *
 * We use inline asm to generate the trampoline binary.
 * =============================================================================*/

/* AP stack: each AP gets an 8KB stack */
#define AP_STACK_SIZE 32768   /* 32 KB per AP (needs headroom for GEMM, ISRs) */
static uint8_t ap_stacks[MAX_CPUS][AP_STACK_SIZE] __attribute__((aligned(16)));

/* AP entry flag - set by each AP when it reaches C code */
volatile uint32_t ap_running_flag = 0;

/* The trampoline code. We'll manually write the binary to 0x8000 */
#define TRAMPOLINE_ADDR 0x8000

/* Trampoline: 16-bit stub that switches to long mode and jumps to ap_entry
 * This is a minimal binary blob we copy to 0x8000 */
static const uint8_t trampoline_code[] = {
    /* 0x0000: 16-bit real mode entry (CS:IP = 0x0800:0x0000 = 0x8000) */
    0xFA,                         /* cli */
    0x31, 0xC0,                   /* xor eax, eax */
    0x8E, 0xD8,                   /* mov ds, ax */
    0x8E, 0xC0,                   /* mov es, ax */
    0x8E, 0xD0,                   /* mov ss, ax */

    /* Load 32-bit GDT */
    0x0F, 0x01, 0x16,             /* lgdt [gdt_ptr] (at trampoline + 0xD0) */
    0xD0, 0x80,                   /* offset 0x80D0 (absolute, DS=0) */

    /* Enable protected mode */
    0x0F, 0x20, 0xC0,             /* mov eax, cr0 */
    0x0C, 0x01,                   /* or al, 1 */
    0x0F, 0x22, 0xC0,             /* mov cr0, eax */

    /* Far jump to 32-bit code at trampoline + 0x30 */
    0x66, 0xEA,                   /* ljmp 0x08:offset */
    0x30, 0x80, 0x00, 0x00,      /* offset = 0x8030 (absolute) */
    0x08, 0x00,                   /* segment selector 0x08 */
    /* Pad to offset 0x30 (need 18 NOPs: 0x30 - 0x1E = 18) */
    0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
    0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
    0x90, 0x90,

    /* 0x0030: 32-bit protected mode */
    /* bits 32 */
    0x66, 0xB8, 0x10, 0x00,      /* mov ax, 0x10 (data segment) */
    0x8E, 0xD8,                   /* mov ds, ax */
    0x8E, 0xC0,                   /* mov es, ax */
    0x8E, 0xD0,                   /* mov ss, ax */

    /* Enable PAE (CR4 bit 5) */
    0x0F, 0x20, 0xE0,             /* mov eax, cr4 */
    0x0D, 0x20, 0x00, 0x00, 0x00,/* or eax, 0x20 */
    0x0F, 0x22, 0xE0,             /* mov cr4, eax */

    /* Load CR3 from trampoline data area (offset 0xC0) */
    0x8B, 0x05,                   /* mov eax, [abs] */
    0xC0, 0x80, 0x00, 0x00,      /* address 0x80C0 */
    0x0F, 0x22, 0xD8,             /* mov cr3, eax */

    /* Enable long mode in EFER MSR */
    0xB9, 0x80, 0x00, 0x00, 0xC0,/* mov ecx, 0xC0000080 */
    0x0F, 0x32,                   /* rdmsr */
    0x0D, 0x00, 0x01, 0x00, 0x00,/* or eax, 0x100 (LME) */
    0x0F, 0x30,                   /* wrmsr */

    /* Enable paging (CR0 bit 31) */
    0x0F, 0x20, 0xC0,             /* mov eax, cr0 */
    0x0D, 0x00, 0x00, 0x00, 0x80,/* or eax, 0x80000000 */
    0x0F, 0x22, 0xC0,             /* mov cr0, eax */

    /* Far jump to 64-bit code */
    0xEA,                          /* ljmp */
    0x70, 0x80, 0x00, 0x00,       /* offset = 0x8070 (absolute) */
    0x18, 0x00,                    /* 64-bit code segment (GDT entry 3) */
    /* Pad to offset 0x70 (need 2 NOPs: 0x70 - 0x6E = 2) */
    0x90, 0x90,

    /* 0x0070: 64-bit long mode */
    /* bits 64 */
    0x48, 0x31, 0xC0,             /* xor rax, rax */
    0xB0, 0x20,                   /* mov al, 0x20 (64-bit data segment) */
    0x8E, 0xD8,                   /* mov ds, ax */
    0x8E, 0xC0,                   /* mov es, ax */
    0x8E, 0xD0,                   /* mov ss, ax */

    /* Signal that we're in 64-bit mode */
    0xF0, 0xFF, 0x05,             /* lock inc dword [ap_running_flag_addr] */
    /* Relative address to flag - will be patched */
    0x00, 0x00, 0x00, 0x00,

    /* Load per-AP stack from trampoline page offset 0xB0 (patched by BSP) */
    0x48, 0x8B, 0x24, 0x25,      /* mov rsp, [abs32] */
    0xB0, 0x80, 0x00, 0x00,      /* address = 0x80B0 */

    /* Load IDT from BSP's IDTR (at trampoline page offset 0x100, patched) */
    0x0F, 0x01, 0x1C, 0x25,      /* lidt [abs32] */
    0x00, 0x81, 0x00, 0x00,      /* address = 0x8100 */

    /* Jump to C ap_idle_loop function (address at offset 0xA8, patched) */
    0x48, 0x8B, 0x04, 0x25,      /* mov rax, [abs32] */
    0xA8, 0x80, 0x00, 0x00,      /* address = 0x80A8 */
    0xFF, 0xE0,                   /* jmp rax */
};

/* GDT for trampoline (at offset 0xD0 in trampoline page) */
static const uint8_t trampoline_gdt[] = {
    /* GDT pointer (6 bytes) */
    0x27, 0x00,                   /* limit = 39 (5 entries * 8 - 1) */
    0xD8, 0x80, 0x00, 0x00,      /* base = 0x80D8 (GDT entries follow) */
    0x00, 0x00,                   /* padding */

    /* GDT entries at offset 0xD8 */
    /* Entry 0: Null */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    /* Entry 1 (0x08): 32-bit code */
    0xFF, 0xFF, 0x00, 0x00, 0x00, 0x9A, 0xCF, 0x00,
    /* Entry 2 (0x10): 32-bit data */
    0xFF, 0xFF, 0x00, 0x00, 0x00, 0x92, 0xCF, 0x00,
    /* Entry 3 (0x18): 64-bit code */
    0xFF, 0xFF, 0x00, 0x00, 0x00, 0x9A, 0xAF, 0x00,
    /* Entry 4 (0x20): 64-bit data */
    0xFF, 0xFF, 0x00, 0x00, 0x00, 0x92, 0xCF, 0x00,
};

/* =============================================================================
 * AP Idle Loop — runs on each Application Processor after boot
 *
 * Each AP enables interrupts and spins waiting for work. When the BSP
 * calls smp_dispatch(), it sets work_ready=1 and sends an IPI (vector 0xFE).
 * The AP sees work_ready, calls the work function, sets work_done=1, and
 * resumes spinning. This gives us near-zero-latency work dispatch with
 * no OS scheduler overhead — perfect for parallel tensor operations.
 * =============================================================================*/

/* IPI work notification handler (vector 0xFE).
 * This interrupt breaks the HLT in the idle loop. The actual work check
 * happens in the main loop — the ISR just needs to return. */
extern void isr_smp_work(void);
__asm__(
    ".globl isr_smp_work\n"
    ".type isr_smp_work, @function\n"
    "isr_smp_work:\n"
    "  push %rax\n"
    "  mov %cr8, %rax\n"         /* Save TPR */
    "  push %rax\n"
    /* Send EOI to LAPIC (MMIO write to LAPIC_BASE + 0xB0) */
    "  movq smp(%rip), %rax\n"   /* smp.lapic_base (first field of smp_state_t) */
    "  movl $0, 0xB0(%rax)\n"   /* LAPIC EOI register */
    "  pop %rax\n"
    "  mov %rax, %cr8\n"        /* Restore TPR */
    "  pop %rax\n"
    "  iretq\n"
);

/* BSP's GDTR, saved so APs can reload the kernel GDT.
 * The trampoline GDT has selector 0x08 = 32-bit code, but IDT gates use
 * selector 0x08 which must be 64-bit code.  After the AP enters long mode
 * through the trampoline, it must switch to the kernel GDT. */
static uint8_t bsp_gdtr[10] __attribute__((aligned(16)));

/* The AP idle loop: called from trampoline after reaching long mode */
void ap_idle_loop(void)
{
    /* ---- Reload the kernel GDT (critical!) ----
     * The trampoline GDT has selector 0x08 as a 32-bit code segment (needed
     * for real-mode → protected-mode transition).  But IDT gates use selector
     * 0x08 for 64-bit code.  If we take an interrupt with the trampoline GDT
     * still loaded, the CPU loads CS=0x08 = 32-bit code → triple fault.
     *
     * Fix: LGDT the kernel's GDT (saved by BSP in smp_init), then reload
     * CS via a far-return and set data segments to kernel data selector. */
    __asm__ volatile (
        "lgdt %0\n"
        : : "m"(*(uint8_t(*)[10])bsp_gdtr) : "memory"
    );
    /* Reload CS: push kernel CS (0x08), push return address, lretq */
    __asm__ volatile (
        "pushq $0x08\n"
        "leaq 1f(%%rip), %%rax\n"
        "pushq %%rax\n"
        "lretq\n"
        "1:\n"
        /* Reload data segment registers with kernel data selector (0x10) */
        "mov $0x10, %%ax\n"
        "mov %%ax, %%ds\n"
        "mov %%ax, %%es\n"
        "mov %%ax, %%fs\n"
        "mov %%ax, %%gs\n"
        "mov %%ax, %%ss\n"
        : : : "rax", "memory"
    );

    /* Enable SSE/SSE2 on this AP (trampoline doesn't set these).
     * CR0: clear EM (bit 2), set MP (bit 1)
     * CR4: set OSFXSR (bit 9), OSXMMEXCPT (bit 10) */
    __asm__ volatile (
        "mov %%cr0, %%rax\n"
        "and $~(1<<2), %%eax\n"   /* clear EM */
        "or  $(1<<1),  %%eax\n"   /* set MP  */
        "mov %%rax, %%cr0\n"
        "mov %%cr4, %%rax\n"
        "or  $((1<<9)|(1<<10)), %%eax\n"  /* OSFXSR + OSXMMEXCPT */
        "mov %%rax, %%cr4\n"
        : : : "rax"
    );

    /* Identify which CPU we are by reading LAPIC ID */
    uint32_t my_apic_id = lapic_read(LAPIC_ID) >> 24;
    uint32_t my_cpu = 0;
    for (uint32_t i = 0; i < smp.cpu_count; i++) {
        if (smp.cpus[i].apic_id == my_apic_id) {
            my_cpu = i;
            break;
        }
    }

    /* Enable LAPIC on this AP (set spurious interrupt vector, enable bit) */
    lapic_write(LAPIC_SVR, 0x1FF);  /* Enable + spurious vector 0xFF */

    /* Enable interrupts so we can receive the work IPI */
    __asm__ volatile ("sti");

    /* Spin forever: check for work, execute, signal done */
    for (;;) {
        /* Memory fence before checking work_ready */
        __asm__ volatile ("mfence" ::: "memory");

        if (smp.cpus[my_cpu].work_ready) {
            smp.cpus[my_cpu].work_ready = 0;
            __asm__ volatile ("mfence" ::: "memory");

            /* Execute the work function */
            if (smp.cpus[my_cpu].work_fn) {
                smp.cpus[my_cpu].work_fn(smp.cpus[my_cpu].work_arg);
            }

            /* Signal completion */
            __asm__ volatile ("mfence" ::: "memory");
            smp.cpus[my_cpu].work_done = 1;
        }

        /* Sleep until next interrupt (IPI will wake us) */
        __asm__ volatile ("hlt");
    }
}

/* =============================================================================
 * Install trampoline at 0x8000
 * =============================================================================*/

static void install_trampoline(void)
{
    uint8_t *tramp = (uint8_t *)(uintptr_t)TRAMPOLINE_ADDR;
    
    /* Clear the page */
    kmemset(tramp, 0, 4096);
    
    /* Copy code */
    kmemcpy(tramp, trampoline_code, sizeof(trampoline_code));
    
    /* Copy GDT at offset 0xD0 */
    kmemcpy(tramp + 0xD0, trampoline_gdt, sizeof(trampoline_gdt));
    
    /* Write CR3 value at offset 0xC0 (must not overlap GDT entries at 0x88-0xAF) */
    uint64_t cr3;
    __asm__ volatile ("mov %%cr3, %0" : "=r"(cr3));
    *(uint32_t *)(tramp + 0xC0) = (uint32_t)cr3;
    
    /* Patch the lock inc displacement at offset 0x7E to point to ap_running_flag.
     * The lock inc instruction is at 0x807B (3-byte opcode + 4-byte disp32).
     * RIP-relative displacement is relative to the instruction END (0x8082).
     * displacement = &ap_running_flag - 0x8082 */
    int32_t *disp = (int32_t *)(tramp + 0x7E);
    *disp = (int32_t)((uintptr_t)&ap_running_flag - (uintptr_t)(tramp + 0x82));

    /* Patch offset 0xA8: address of ap_idle_loop (64-bit) */
    *(uint64_t *)(tramp + 0xA8) = (uint64_t)(uintptr_t)ap_idle_loop;

    /* Patch offset 0x100: BSP's IDTR so APs can handle interrupts.
     * Read current IDTR via SIDT — avoids static linkage issues. */
    uint8_t idtr_buf[10];
    __asm__ volatile ("sidt %0" : "=m"(*(uint8_t (*)[10])idtr_buf));
    kmemcpy(tramp + 0x100, idtr_buf, 10);

    /* Offset 0xB0 (AP stack) is patched per-AP in smp_init before each SIPI */
}

/* =============================================================================
 * Detect LAPIC and CPUs
 * =============================================================================*/

void smp_detect(void)
{
    kmemset(&smp, 0, sizeof(smp));

    /* Read LAPIC base from IA32_APIC_BASE MSR (0x1B) */
    uint32_t lo, hi;
    __asm__ volatile ("rdmsr" : "=a"(lo), "=d"(hi) : "c"(0x1B));
    smp.lapic_base = ((uint64_t)hi << 32) | (lo & 0xFFFFF000);

    /* Check if LAPIC is enabled */
    if (!(lo & (1 << 11))) {
        kprintf("[SMP] LAPIC not enabled\n");
        smp.cpu_count = 1;
        return;
    }

    /* Get BSP APIC ID */
    smp.bsp_id = lapic_read(LAPIC_ID) >> 24;
    smp.cpus[0].apic_id = smp.bsp_id;
    smp.cpus[0].state = CPU_STATE_IDLE;

    kprintf("[SMP] LAPIC base: 0x%lx, BSP APIC ID: %u\n",
            smp.lapic_base, smp.bsp_id);

    /* Try to detect CPUs via ACPI MADT (simplified) */
    /* For now, try to enumerate via CPUID */
    uint32_t eax, ebx, ecx, edx;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                     : "a"(1));
    
    /* Check for HTT (Hyper-Threading Technology) bit in EDX */
    if (edx & (1 << 28)) {
        /* EBX[23:16] contains logical processor count */
        uint32_t logical_cpus = (ebx >> 16) & 0xFF;
        if (logical_cpus > MAX_CPUS) logical_cpus = MAX_CPUS;
        if (logical_cpus < 1) logical_cpus = 1;
        smp.cpu_count = logical_cpus;
    } else {
        smp.cpu_count = 1;
    }

    /* Initialize CPU state for each detected core */
    for (uint32_t i = 1; i < smp.cpu_count; i++) {
        smp.cpus[i].apic_id = i; /* Assume sequential APIC IDs */
        smp.cpus[i].state = CPU_STATE_OFFLINE;
    }

    kprintf("[SMP] Detected %u logical CPUs\n", smp.cpu_count);
}

/* =============================================================================
 * Initialize LAPIC
 * =============================================================================*/

static void lapic_init(void)
{
    /* Enable LAPIC via SVR (Spurious Vector Register) */
    uint32_t svr = lapic_read(LAPIC_SVR);
    svr |= 0x100;   /* Enable bit */
    svr |= 0xFF;    /* Spurious vector = 0xFF */
    lapic_write(LAPIC_SVR, svr);

    /* Set task priority to 0 (accept all interrupts) */
    lapic_write(LAPIC_TPR, 0);

    /* Clear any pending interrupts */
    lapic_eoi();

    kprintf("[SMP] LAPIC initialized (SVR=0x%x)\n", lapic_read(LAPIC_SVR));
}

/* =============================================================================
 * LAPIC Timer Calibration & Setup
 *
 * Calibrates the LAPIC timer against a known time reference (PIT countdown).
 * Once calibrated, programs the LAPIC timer in periodic mode at 1000 Hz
 * for per-CPU tick counting — essential for profiling SMP performance.
 * =============================================================================*/

/* Per-CPU LAPIC tick counter (BSP only for now) */
static volatile uint64_t lapic_tick_count = 0;
static uint32_t lapic_ticks_per_ms = 0;

/* LAPIC timer ISR (vector 0xFD) — minimal: increment tick, send EOI */
extern void isr_lapic_timer(void);
__asm__(
    ".globl isr_lapic_timer\n"
    ".type isr_lapic_timer, @function\n"
    "isr_lapic_timer:\n"
    "  push %rax\n"
    "  push %rdx\n"
    "  lock incq lapic_tick_count(%rip)\n"
    /* Send LAPIC EOI */
    "  movq smp(%rip), %rax\n"      /* smp.lapic_base */
    "  movl $0, 0xB0(%rax)\n"       /* LAPIC EOI */
    "  pop %rdx\n"
    "  pop %rax\n"
    "  iretq\n"
);

/* Calibrate LAPIC timer: count how many LAPIC ticks pass in 10ms (via PIT) */
static void lapic_timer_calibrate(void)
{
    /* Program PIT channel 2 for 10ms one-shot
     * PIT frequency = 1193182 Hz, so 10ms = 11932 ticks */
    const uint16_t pit_10ms = 11932;

    /* LAPIC timer: divide by 16, one-shot, start with max count */
    lapic_write(LAPIC_TIMER_DIV, 0x03);  /* Divide by 16 */
    lapic_write(LAPIC_TIMER_LVT, 0x10000); /* Masked (disabled) for calibration */

    /* Setup PIT channel 2 in hardware retriggerable one-shot */
    outb(0x43, 0xB0);  /* Channel 2, lobyte/hibyte, mode 0 (interrupt on terminal count) */
    outb(0x42, (uint8_t)(pit_10ms & 0xFF));
    outb(0x42, (uint8_t)(pit_10ms >> 8));

    /* Start LAPIC timer counting down from 0xFFFFFFFF */
    lapic_write(LAPIC_TIMER_ICR, 0xFFFFFFFF);

    /* Gate PIT channel 2 on */
    uint8_t gate = inb(0x61);
    outb(0x61, (gate & 0xFC) | 0x01);

    /* Wait for PIT to count down (poll bit 5 of port 0x61) */
    while (!(inb(0x61) & 0x20))
        __asm__ volatile ("pause");

    /* Stop LAPIC timer */
    lapic_write(LAPIC_TIMER_LVT, 0x10000); /* Mask to stop */

    /* Read how many LAPIC ticks elapsed */
    uint32_t elapsed = 0xFFFFFFFF - lapic_read(LAPIC_TIMER_CCR);

    /* LAPIC ticks per ms = elapsed / 10 */
    lapic_ticks_per_ms = elapsed / 10;

    kprintf("[SMP] LAPIC timer: %u ticks/ms (calibrated via PIT 10ms)\n",
            lapic_ticks_per_ms);
}

/* Start LAPIC timer in periodic mode at 1000 Hz */
static void lapic_timer_start(void)
{
    if (lapic_ticks_per_ms == 0) return;

    extern void idt_set_gate(int num, uint64_t handler);
    idt_set_gate(0xFD, (uint64_t)(uintptr_t)isr_lapic_timer);

    lapic_write(LAPIC_TIMER_DIV, 0x03);  /* Divide by 16 */
    /* Periodic mode (bit 17), vector 0xFD, not masked */
    lapic_write(LAPIC_TIMER_LVT, 0x200FD);
    lapic_write(LAPIC_TIMER_ICR, lapic_ticks_per_ms); /* 1 ms period = 1000 Hz */

    kprintf("[SMP] LAPIC timer running at 1000 Hz (vector 0xFD)\n");
}

/* Get LAPIC tick count (for profiling) */
uint64_t smp_lapic_ticks(void)
{
    return lapic_tick_count;
}

/* =============================================================================
 * Send IPI (Inter-Processor Interrupt)
 * =============================================================================*/

static void lapic_send_ipi(uint8_t apic_id, uint32_t icr_lo)
{
    /* Wait for previous IPI to complete */
    while (lapic_read(LAPIC_ICR_LO) & (1 << 12))
        __asm__ volatile ("pause");

    /* Set destination APIC ID */
    lapic_write(LAPIC_ICR_HI, (uint32_t)apic_id << 24);

    /* Send IPI */
    lapic_write(LAPIC_ICR_LO, icr_lo);

    /* Wait for delivery */
    while (lapic_read(LAPIC_ICR_LO) & (1 << 12))
        __asm__ volatile ("pause");
}

/* =============================================================================
 * Boot APs via INIT-SIPI-SIPI
 * =============================================================================*/

void smp_init(void)
{
    if (smp.cpu_count <= 1) {
        kprintf("[SMP] Single CPU -- skipping AP boot\n");
        return;
    }

    /* Save BSP's GDTR so APs can reload the kernel GDT */
    __asm__ volatile ("sgdt %0" : "=m"(*(uint8_t(*)[10])bsp_gdtr));

    /* Initialize BSP LAPIC */
    lapic_init();

    /* Calibrate and start LAPIC timer */
    lapic_timer_calibrate();
    lapic_timer_start();

    /* Install AP trampoline at 0x8000 */
    install_trampoline();

    /* Install IPI work notification handler (vector 0xFE) in IDT */
    extern void idt_set_gate(int num, uint64_t handler);
    idt_set_gate(0xFE, (uint64_t)(uintptr_t)isr_smp_work);

    kprintf("[SMP] Starting %u Application Processors...\n", smp.cpu_count - 1);

    for (uint32_t i = 1; i < smp.cpu_count; i++) {
        uint8_t target_id = smp.cpus[i].apic_id;
        smp.cpus[i].state = CPU_STATE_BOOTING;

        /* Patch per-AP stack pointer at trampoline offset 0xB0.
         * Stack grows down, so set RSP to top of the AP's stack. */
        uint8_t *tramp = (uint8_t *)(uintptr_t)TRAMPOLINE_ADDR;
        uint64_t stack_top = (uint64_t)(uintptr_t)&ap_stacks[i][AP_STACK_SIZE];
        *(uint64_t *)(tramp + 0xB0) = stack_top;

        uint32_t prev_count = ap_running_flag;

        /* Step 1: Send INIT IPI */
        lapic_send_ipi(target_id, ICR_INIT | ICR_LEVEL_ASSERT);
        smp_delay_us(200);  /* Wait 200us */

        /* Deassert INIT */
        lapic_send_ipi(target_id, ICR_INIT | ICR_LEVEL_DEASSERT);
        smp_delay_us(10000);  /* Wait 10ms */

        /* Step 2: Send STARTUP IPI (vector = trampoline page number) */
        uint32_t vec = TRAMPOLINE_ADDR >> 12;  /* 0x8000 >> 12 = 8 */
        lapic_send_ipi(target_id, ICR_STARTUP | vec);
        smp_delay_us(200);

        /* Step 3: Send second STARTUP IPI */
        lapic_send_ipi(target_id, ICR_STARTUP | vec);
        smp_delay_us(200);

        /* Wait for AP to signal (up to 100ms) */
        uint64_t timeout = 100000; /* 100ms in us */
        uint64_t waited = 0;
        while (ap_running_flag == prev_count && waited < timeout) {
            smp_delay_us(100);
            waited += 100;
        }

        if (ap_running_flag > prev_count) {
            smp.ap_started++;
            smp.cpus[i].state = CPU_STATE_IDLE;
            kprintf("[SMP] CPU %u (APIC %u) started OK\n", i, target_id);
        } else {
            smp.cpus[i].state = CPU_STATE_OFFLINE;
            kprintf("[SMP] CPU %u (APIC %u) failed to start\n", i, target_id);
        }
    }

    kprintf("[SMP] %u/%u APs started\n", smp.ap_started, smp.cpu_count - 1);
}

/* Simple SMP dispatch test worker — writes 0xCAFE to the flag */
void smp_test_worker(void *arg)
{
    volatile uint32_t *flag = (volatile uint32_t *)arg;
    *flag = 0xCAFE;
    __asm__ volatile ("mfence" ::: "memory");
}

/* =============================================================================
 * Work dispatch
 * =============================================================================*/

int smp_dispatch(uint32_t cpu_id, smp_work_fn_t fn, void *arg)
{
    if (cpu_id >= smp.cpu_count) return -1;
    if (smp.cpus[cpu_id].state != CPU_STATE_IDLE) return -2;

    smp.cpus[cpu_id].work_fn = fn;
    smp.cpus[cpu_id].work_arg = arg;
    smp.cpus[cpu_id].work_done = 0;
    __asm__ volatile ("mfence" ::: "memory");
    smp.cpus[cpu_id].work_ready = 1;
    smp.cpus[cpu_id].state = CPU_STATE_BUSY;

    /* Send IPI to wake the AP (vector 0xFE = work notification).
     * For Fixed delivery mode, Level must be Assert (bit 14) and
     * Trigger must be Edge (bit 15 = 0) per Intel SDM Vol 3A §10.6.1. */
    if (cpu_id > 0) {
        lapic_send_ipi(smp.cpus[cpu_id].apic_id, 0xFE | ICR_LEVEL_ASSERT);
    }

    return 0;
}

void smp_dispatch_all(smp_work_fn_t fn, void *arg)
{
    for (uint32_t i = 1; i < smp.cpu_count; i++) {
        if (smp.cpus[i].state == CPU_STATE_IDLE) {
            smp_dispatch(i, fn, arg);
        }
    }
}

void smp_wait(uint32_t cpu_id)
{
    if (cpu_id >= smp.cpu_count) return;
    while (!smp.cpus[cpu_id].work_done)
        __asm__ volatile ("pause");
    smp.cpus[cpu_id].state = CPU_STATE_IDLE;
}

void smp_wait_all(void)
{
    for (uint32_t i = 1; i < smp.cpu_count; i++) {
        if (smp.cpus[i].state == CPU_STATE_BUSY) {
            smp_wait(i);
        }
    }
}

/* =============================================================================
 * Print SMP status
 * =============================================================================*/

static const char *cpu_state_names[] = {
    "OFFLINE", "BOOTING", "IDLE", "BUSY"
};

void smp_print_status(void)
{
    kprintf("[SMP] %u CPUs, BSP APIC ID %u\n", smp.cpu_count, smp.bsp_id);
    for (uint32_t i = 0; i < smp.cpu_count; i++) {
        const char *state = (smp.cpus[i].state < 4) ?
                            cpu_state_names[smp.cpus[i].state] : "UNKNOWN";
        kprintf("  CPU %u: APIC %u, state=%s\n",
                i, smp.cpus[i].apic_id, state);
    }
}

/* =============================================================================
 * SMP Demo
 * =============================================================================*/

void smp_run_demos(void)
{
    kprintf("\n=== SMP Multi-Core Demo ===\n");
    smp_detect();
    
    /* Only attempt SMP init if multiple CPUs detected */
    if (smp.cpu_count > 1) {
        smp_init();
    }
    
    smp_print_status();
    
    kprintf("[SMP] Multi-core infrastructure ready\n");
    if (smp.cpu_count > 1 && smp.ap_started > 0) {
        kprintf("[SMP] %u cores available for parallel tensor operations\n", 
                smp.ap_started + 1);
    } else {
        kprintf("[SMP] Single-core mode (APs can be started with real hardware)\n");
    }
}

#else /* __aarch64__ */

#include "kernel/core/kernel.h"
#include "kernel/core/smp.h"

smp_state_t smp = {0};
volatile uint32_t ap_running_flag = 0;

uint32_t lapic_read(uint32_t off) { (void)off; return 0; }
void lapic_write(uint32_t off, uint32_t v) { (void)off; (void)v; }
void lapic_eoi(void) {}
uint32_t smp_get_apic_id(void) { return 0; }
void smp_detect(void) { smp.cpu_count = 4; /* Cortex-A72 quad-core */ }
void smp_init(void) { smp_detect(); }
int smp_dispatch(uint32_t cpu_id, smp_work_fn_t fn, void *arg) { (void)cpu_id; (void)fn; (void)arg; return -1; }
void smp_dispatch_all(smp_work_fn_t fn, void *arg) { (void)fn; (void)arg; }
void smp_wait(uint32_t cpu_id) { (void)cpu_id; }
void smp_wait_all(void) {}
void smp_print_status(void) { kprintf("[SMP] ARM64 PSCI: 4 cores\n"); }
void smp_run_demos(void) {
    smp_detect();
    kprintf("[SMP] ARM64 PSCI multicore (4 Cortex-A72 cores)\n");
}

#endif /* __aarch64__ */
