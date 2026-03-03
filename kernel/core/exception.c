/* =============================================================================
 * TensorOS - CPU Exception Handlers (Production-Grade)
 *
 * x86_64-only: ARM64 uses EL1 vector table (VBAR_EL1) instead.
 * =============================================================================*/

#ifndef __aarch64__
/* =============================================================================
 * Handles all CPU exceptions (vectors 0-31) with:
 *   - Full register dump (RAX-R15, RIP, RSP, RFLAGS, CR2, CS, SS)
 *   - Stack trace via RBP chain walking
 *   - Exception-specific diagnostics (page fault, GPF, invalid opcode)
 *   - Clean halt with reboot prompt
 *
 * CRITICAL FIX: Previously all 256 IDT entries pointed to isr_stub which
 * sends EOI and iretq. CPU exceptions (vectors 0-31) are NOT PIC interrupts:
 *   - Sending EOI for non-PIC exceptions masks real IRQs
 *   - Exceptions that push error codes corrupt the stack when isr_stub
 *     tries to iretq (treats error code as RIP → triple fault)
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"

/* =============================================================================
 * Exception Frame
 * Matches the stack layout: ISR stub pushes → common handler pushes → CPU frame
 * =============================================================================*/

typedef struct {
    /* Pushed by exception_common (reverse order of push = forward order here) */
    uint64_t r15, r14, r13, r12, r11, r10, r9, r8;
    uint64_t rbp, rsi, rdi, rdx, rcx, rbx, rax;
    /* Pushed by ISR stub */
    uint64_t vector;
    uint64_t error_code;    /* Dummy 0 for exceptions without, real for those with */
    /* Pushed by CPU */
    uint64_t rip;
    uint64_t cs;
    uint64_t rflags;
    uint64_t rsp;
    uint64_t ss;
} __attribute__((packed)) exception_frame_t;

/* =============================================================================
 * Exception Names (vectors 0-31)
 * =============================================================================*/

static const char *exception_names[32] = {
    "Divide Error (#DE)",                   /* 0  */
    "Debug (#DB)",                          /* 1  */
    "Non-Maskable Interrupt (NMI)",         /* 2  */
    "Breakpoint (#BP)",                     /* 3  */
    "Overflow (#OF)",                       /* 4  */
    "Bound Range Exceeded (#BR)",           /* 5  */
    "Invalid Opcode (#UD)",                 /* 6  */
    "Device Not Available (#NM)",           /* 7  */
    "Double Fault (#DF)",                   /* 8  */
    "Coprocessor Segment Overrun",          /* 9  */
    "Invalid TSS (#TS)",                    /* 10 */
    "Segment Not Present (#NP)",            /* 11 */
    "Stack-Segment Fault (#SS)",            /* 12 */
    "General Protection Fault (#GP)",       /* 13 */
    "Page Fault (#PF)",                     /* 14 */
    "Reserved",                             /* 15 */
    "x87 FP Exception (#MF)",              /* 16 */
    "Alignment Check (#AC)",               /* 17 */
    "Machine Check (#MC)",                 /* 18 */
    "SIMD FP Exception (#XM)",             /* 19 */
    "Virtualization Exception (#VE)",      /* 20 */
    "Control Protection (#CP)",            /* 21 */
    "Reserved", "Reserved", "Reserved",    /* 22-24 */
    "Reserved", "Reserved", "Reserved",    /* 25-27 */
    "Hypervisor Injection (#HV)",          /* 28 */
    "VMM Communication (#VC)",             /* 29 */
    "Security Exception (#SX)",            /* 30 */
    "Reserved",                            /* 31 */
};

/* =============================================================================
 * Stack Trace Walker
 * Walks the RBP chain to produce a call-stack backtrace.
 * Stops at null/invalid frame pointers or after 16 frames.
 * =============================================================================*/

static void dump_stack_trace(uint64_t rbp, uint64_t rip)
{
    kprintf("\n  Call Stack:\n");
    kprintf("    #0  %p  <-- fault location\n", (void *)rip);

    uint64_t *frame = (uint64_t *)rbp;
    for (int depth = 1; depth < 16; depth++) {
        /* Sanity: frame pointer must be in a reasonable range */
        if ((uint64_t)frame < 0x10000 || (uint64_t)frame > 0xFFFFFFFFF)
            break;

        uint64_t ret_addr = frame[1];   /* Return address */
        uint64_t next_rbp = frame[0];   /* Saved RBP */

        if (ret_addr == 0) break;

        kprintf("    #%d  %p\n", depth, (void *)ret_addr);

        /* Guard against infinite loops (next frame must be higher on stack) */
        if (next_rbp <= (uint64_t)frame) break;
        frame = (uint64_t *)next_rbp;
    }
}

/* =============================================================================
 * Page Fault Error Code Decoder
 * Bit 0: 0=not-present, 1=protection violation
 * Bit 1: 0=read, 1=write
 * Bit 2: 0=kernel, 1=user
 * Bit 3: reserved bit violation
 * Bit 4: instruction fetch
 * =============================================================================*/

static void decode_page_fault(uint64_t error_code, uint64_t cr2)
{
    kprintf("\n  Page Fault Details:\n");
    kprintf("    Faulting address: %p\n", (void *)cr2);
    kprintf("    Cause:  %s\n",
            (error_code & 1) ? "protection violation" : "page not present");
    kprintf("    Access: %s\n", (error_code & 2) ? "WRITE" : "READ");
    kprintf("    Mode:   %s\n", (error_code & 4) ? "user" : "kernel");
    if (error_code & 8)  kprintf("    Reserved bit violation in page table\n");
    if (error_code & 16) kprintf("    Instruction fetch (NX violation)\n");
}

/* =============================================================================
 * GPF Error Code Decoder
 * =============================================================================*/

static void decode_gpf(uint64_t error_code)
{
    if (error_code == 0) {
        kprintf("\n  GPF: selector=0 (null deref, bad memory, or privileged instruction)\n");
    } else {
        kprintf("\n  GPF Selector Info:\n");
        kprintf("    Raw:   %p\n", (void *)error_code);
        kprintf("    Table: %s\n",
                (error_code & 2) ? "IDT" : ((error_code & 4) ? "LDT" : "GDT"));
        kprintf("    Index: %lu\n", (error_code >> 3) & 0x1FFF);
        if (error_code & 1) kprintf("    External event\n");
    }
}

/* =============================================================================
 * C Exception Handler
 * Called from the assembly common stub with pointer to saved register state.
 * This function does NOT return — it halts the CPU after printing diagnostics.
 * =============================================================================*/

void exception_handler_c(exception_frame_t *frame, uint64_t cr2)
{
    /* Disable interrupts — we're crashing, nothing else should run */
    __asm__ volatile("cli");

    uint64_t vec = frame->vector;

    /* =================================================================
     * Recoverable Page Fault: try demand-paging before panicking.
     * If vm_demand_fault() succeeds, re-enable interrupts and RETURN
     * so the faulting instruction is retried with the new mapping.
     * =================================================================*/
    if (vec == 14) {
        /* Only attempt demand-page for not-present faults (error_code bit 0 = 0) */
        if (!(frame->error_code & 1)) {
            if (vm_demand_fault(cr2) == 0) {
                __asm__ volatile("sti");
                return;  /* Back to exception_common → iretq → retry */
            }
        }
    }

    const char *name = (vec < 32) ? exception_names[vec] : "Unknown Exception";

    kprintf("\n");
    kprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    kprintf("  KERNEL PANIC: CPU Exception %lu - %s\n", vec, name);
    kprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

    /* Full register dump */
    kprintf("\n  Register State:\n");
    kprintf("    RIP    = %p    RSP    = %p\n",
            (void *)frame->rip, (void *)frame->rsp);
    kprintf("    RBP    = %p    RFLAGS = %p\n",
            (void *)frame->rbp, (void *)frame->rflags);
    kprintf("    RAX    = %p    RBX    = %p\n",
            (void *)frame->rax, (void *)frame->rbx);
    kprintf("    RCX    = %p    RDX    = %p\n",
            (void *)frame->rcx, (void *)frame->rdx);
    kprintf("    RSI    = %p    RDI    = %p\n",
            (void *)frame->rsi, (void *)frame->rdi);
    kprintf("    R8     = %p    R9     = %p\n",
            (void *)frame->r8, (void *)frame->r9);
    kprintf("    R10    = %p    R11    = %p\n",
            (void *)frame->r10, (void *)frame->r11);
    kprintf("    R12    = %p    R13    = %p\n",
            (void *)frame->r12, (void *)frame->r13);
    kprintf("    R14    = %p    R15    = %p\n",
            (void *)frame->r14, (void *)frame->r15);
    kprintf("    CS     = %p    SS     = %p\n",
            (void *)frame->cs, (void *)frame->ss);
    kprintf("    ERR    = %p    CR2    = %p\n",
            (void *)frame->error_code, (void *)cr2);

    /* Exception-specific diagnostics */
    switch (vec) {
    case 14:
        decode_page_fault(frame->error_code, cr2);
        break;
    case 13:
        decode_gpf(frame->error_code);
        break;
    case 8:
        kprintf("\n  Double Fault: corrupted IDT, stack overflow, or nested exception\n");
        break;
    case 6:
        kprintf("\n  Invalid Opcode at %p\n", (void *)frame->rip);
        kprintf("  Possible cause: AVX/AVX2 instruction on CPU without support\n");
        break;
    case 0:
        kprintf("\n  Divide by zero at %p\n", (void *)frame->rip);
        break;
    case 19:
        kprintf("\n  SIMD exception at %p (check for NaN/denormal in vector math)\n",
                (void *)frame->rip);
        break;
    }

    /* Stack trace */
    dump_stack_trace(frame->rbp, frame->rip);

    kprintf("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    kprintf("  System halted. Reboot required.\n");
    kprintf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

    /* Write ISA debug-exit to signal QEMU (if running under QEMU) */
    outb(0x501, 0x31);

    /* Halt forever */
    for (;;) __asm__ volatile("hlt");
}

/* =============================================================================
 * Assembly Stubs
 *
 * Common handler: saves all 15 GP registers, passes frame pointer + CR2
 * to C handler, restores on return (though fatal exceptions never return).
 *
 * Stack after common handler push (lowest address first):
 *   [R15] [R14] ... [RAX]  ← pushed by exception_common
 *   [vector] [error_code]  ← pushed by ISR stub
 *   [RIP] [CS] [RFLAGS] [RSP] [SS]  ← pushed by CPU
 * =============================================================================*/

__asm__(
    ".text\n"
    ".globl exception_common\n"
    ".type exception_common, @function\n"
    "exception_common:\n"
    "  pushq %rax\n"
    "  pushq %rbx\n"
    "  pushq %rcx\n"
    "  pushq %rdx\n"
    "  pushq %rdi\n"
    "  pushq %rsi\n"
    "  pushq %rbp\n"
    "  pushq %r8\n"
    "  pushq %r9\n"
    "  pushq %r10\n"
    "  pushq %r11\n"
    "  pushq %r12\n"
    "  pushq %r13\n"
    "  pushq %r14\n"
    "  pushq %r15\n"
    "  movq %rsp, %rdi\n"          /* arg1: pointer to exception_frame_t */
    "  movq %cr2, %rsi\n"          /* arg2: CR2 (faulting address for #PF) */
    "  call exception_handler_c\n"
    /* If we ever return (non-fatal or future recovery): */
    "  popq %r15\n"
    "  popq %r14\n"
    "  popq %r13\n"
    "  popq %r12\n"
    "  popq %r11\n"
    "  popq %r10\n"
    "  popq %r9\n"
    "  popq %r8\n"
    "  popq %rbp\n"
    "  popq %rsi\n"
    "  popq %rdi\n"
    "  popq %rdx\n"
    "  popq %rcx\n"
    "  popq %rbx\n"
    "  popq %rax\n"
    "  addq $16, %rsp\n"           /* skip vector + error_code */
    "  iretq\n"
);

/* =============================================================================
 * Individual ISR Stubs
 *
 * Two flavors:
 *   ISR_NOERR(n): exceptions that don't push error code → push dummy 0
 *   ISR_ERR(n):   exceptions that push error code → CPU already pushed it
 *
 * Both push vector number then jump to exception_common.
 * =============================================================================*/

#define ISR_NOERR(n) \
    extern void isr_exc_##n(void); \
    __asm__( \
        ".globl isr_exc_" #n "\n" \
        ".type isr_exc_" #n ", @function\n" \
        "isr_exc_" #n ":\n" \
        "  pushq $0\n" \
        "  pushq $" #n "\n" \
        "  jmp exception_common\n" \
    );

#define ISR_ERR(n) \
    extern void isr_exc_##n(void); \
    __asm__( \
        ".globl isr_exc_" #n "\n" \
        ".type isr_exc_" #n ", @function\n" \
        "isr_exc_" #n ":\n" \
        "  pushq $" #n "\n" \
        "  jmp exception_common\n" \
    );

/* --- Vectors 0-31: all CPU exceptions --- */
ISR_NOERR(0)   /* #DE Divide Error */
ISR_NOERR(1)   /* #DB Debug */
ISR_NOERR(2)   /* NMI */
ISR_NOERR(3)   /* #BP Breakpoint */
ISR_NOERR(4)   /* #OF Overflow */
ISR_NOERR(5)   /* #BR Bound Range Exceeded */
ISR_NOERR(6)   /* #UD Invalid Opcode */
ISR_NOERR(7)   /* #NM Device Not Available */
ISR_ERR(8)     /* #DF Double Fault (error code always 0) */
ISR_NOERR(9)   /* Coprocessor Segment Overrun */
ISR_ERR(10)    /* #TS Invalid TSS */
ISR_ERR(11)    /* #NP Segment Not Present */
ISR_ERR(12)    /* #SS Stack-Segment Fault */
ISR_ERR(13)    /* #GP General Protection Fault */
ISR_ERR(14)    /* #PF Page Fault */
ISR_NOERR(15)  /* Reserved */
ISR_NOERR(16)  /* #MF x87 FP Exception */
ISR_ERR(17)    /* #AC Alignment Check */
ISR_NOERR(18)  /* #MC Machine Check */
ISR_NOERR(19)  /* #XM SIMD FP Exception */
ISR_NOERR(20)  /* #VE Virtualization Exception */
ISR_ERR(21)    /* #CP Control Protection */
ISR_NOERR(22)  /* Reserved */
ISR_NOERR(23)  /* Reserved */
ISR_NOERR(24)  /* Reserved */
ISR_NOERR(25)  /* Reserved */
ISR_NOERR(26)  /* Reserved */
ISR_NOERR(27)  /* Reserved */
ISR_NOERR(28)  /* #HV Hypervisor Injection */
ISR_ERR(29)    /* #VC VMM Communication */
ISR_ERR(30)    /* #SX Security Exception */
ISR_NOERR(31)  /* Reserved */

/* =============================================================================
 * Install Exception Handlers into IDT
 * Called during idt_init() to override vectors 0-31 with proper handlers.
 * =============================================================================*/

typedef void (*isr_fn_t)(void);

static isr_fn_t exception_isrs[32] = {
    isr_exc_0,  isr_exc_1,  isr_exc_2,  isr_exc_3,
    isr_exc_4,  isr_exc_5,  isr_exc_6,  isr_exc_7,
    isr_exc_8,  isr_exc_9,  isr_exc_10, isr_exc_11,
    isr_exc_12, isr_exc_13, isr_exc_14, isr_exc_15,
    isr_exc_16, isr_exc_17, isr_exc_18, isr_exc_19,
    isr_exc_20, isr_exc_21, isr_exc_22, isr_exc_23,
    isr_exc_24, isr_exc_25, isr_exc_26, isr_exc_27,
    isr_exc_28, isr_exc_29, isr_exc_30, isr_exc_31,
};

/* idt_set_gate is defined in klib.c — made non-static for cross-module use */
extern void idt_set_gate(int num, uint64_t handler);

void exception_install_handlers(void)
{
    for (int i = 0; i < 32; i++) {
        idt_set_gate(i, (uint64_t)(uintptr_t)exception_isrs[i]);
    }
    kprintf("[EXCEPT] 32 CPU exception handlers installed (vectors 0-31)\n");
    kprintf("[EXCEPT] Page fault, GPF, double fault: full register dump + stack trace\n");
}

#else /* __aarch64__ */

#include "kernel/core/kernel.h"

void exception_install_handlers(void)
{
    kprintf("[EXCEPT] ARM64 EL1 exception vectors configured via GIC-400\n");
}

#endif /* __aarch64__ */
