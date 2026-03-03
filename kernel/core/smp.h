/* =============================================================================
 * TensorOS - SMP Bootstrap & LAPIC Driver
 * Wakes Application Processors via INIT-SIPI-SIPI
 * Provides per-core work dispatch for parallel tensor operations
 * =============================================================================*/

#ifndef TENSOROS_SMP_H
#define TENSOROS_SMP_H

#include <stdint.h>

#define MAX_CPUS 64

/* LAPIC register offsets (from LAPIC base) */
#define LAPIC_ID        0x020
#define LAPIC_VERSION   0x030
#define LAPIC_TPR       0x080
#define LAPIC_EOI       0x0B0
#define LAPIC_SVR       0x0F0
#define LAPIC_ICR_LO    0x300
#define LAPIC_ICR_HI    0x310
#define LAPIC_TIMER_LVT 0x320
#define LAPIC_TIMER_ICR 0x380
#define LAPIC_TIMER_CCR 0x390
#define LAPIC_TIMER_DIV 0x3E0

/* ICR delivery modes */
#define ICR_INIT        0x00000500
#define ICR_STARTUP     0x00000600
#define ICR_LEVEL_ASSERT 0x00004000
#define ICR_LEVEL_DEASSERT 0x00000000
#define ICR_ALL_EXCL_SELF 0x000C0000

/* Per-CPU state */
typedef enum {
    CPU_STATE_OFFLINE = 0,
    CPU_STATE_BOOTING = 1,
    CPU_STATE_IDLE    = 2,
    CPU_STATE_BUSY    = 3,
} cpu_state_t;

/* Work item for SMP dispatch */
typedef void (*smp_work_fn_t)(void *arg);

typedef struct {
    uint8_t       apic_id;
    cpu_state_t   state;
    volatile int  work_ready;   /* Set by BSP, cleared by AP */
    smp_work_fn_t work_fn;
    void         *work_arg;
    volatile int  work_done;    /* Set by AP when complete */
    uint64_t      ticks;        /* Per-CPU tick counter */
} smp_cpu_t;

/* SMP system state */
typedef struct {
    uint64_t   lapic_base;      /* LAPIC MMIO base address */
    uint32_t   bsp_id;          /* BSP APIC ID */
    uint32_t   cpu_count;       /* Total CPUs (BSP + APs) */
    smp_cpu_t  cpus[MAX_CPUS];
    volatile uint32_t ap_started; /* Count of APs that have started */
} smp_state_t;

extern smp_state_t smp;

/* =============================================================================
 * API
 * =============================================================================*/

/**
 * Detect LAPIC and enumerate CPUs.
 * Must be called early in boot before SMP init.
 */
void smp_detect(void);

/**
 * Initialize LAPIC and start all Application Processors.
 * APs enter idle loop waiting for work.
 */
void smp_init(void);

/**
 * Dispatch work to a specific CPU.
 * Returns 0 on success, -1 if CPU busy.
 */
int smp_dispatch(uint32_t cpu_id, smp_work_fn_t fn, void *arg);

/**
 * Dispatch work to all APs (parallel execution).
 * BSP waits for all APs to complete.
 */
void smp_dispatch_all(smp_work_fn_t fn, void *arg);

/**
 * Wait for a specific CPU to complete its work.
 */
void smp_wait(uint32_t cpu_id);

/**
 * Wait for all CPUs to complete.
 */
void smp_wait_all(void);

/**
 * Get the current CPU's APIC ID.
 */
uint32_t smp_get_apic_id(void);

/**
 * Send EOI to LAPIC (called at end of interrupt handlers).
 */
void lapic_eoi(void);

/**
 * Read/write LAPIC registers.
 */
uint32_t lapic_read(uint32_t offset);
void lapic_write(uint32_t offset, uint32_t val);

/**
 * Print SMP status.
 */
void smp_print_status(void);

/**
 * Run SMP demos.
 */
void smp_run_demos(void);

/**
 * Get LAPIC timer tick count (for profiling).
 */
uint64_t smp_lapic_ticks(void);

#endif /* TENSOROS_SMP_H */
