/* =============================================================================
 * TensorOS ARM64 Hardware Abstraction Layer
 *
 * BCM2711 (Raspberry Pi 4) peripheral access:
 *   - PL011 UART (serial console)
 *   - ARM Generic Timer (replaces x86 TSC/PIT)
 *   - GIC-400 interrupt controller (replaces x86 PIC/APIC)
 *   - GPIO for UART pin mux
 *   - Mailbox for firmware queries
 *
 * Zero-overhead: all functions are static inline or direct MMIO.
 * No abstraction penalty — compiles to raw load/store instructions.
 * =============================================================================*/

#ifndef TENSOROS_ARM64_HAL_H
#define TENSOROS_ARM64_HAL_H

#include <stdint.h>

/* HDMI framebuffer console — real implementation in kernel/drivers/gpu/rpi_fb.c.
 * Declarations available via kernel/drivers/gpu/rpi_fb.h (included below). */

/* =============================================================================
 * BCM2711 Peripheral Base Addresses
 * =============================================================================*/

#define BCM2711_PERI_BASE       0xFE000000UL
#define BCM2711_GPIO_BASE       (BCM2711_PERI_BASE + 0x200000)
#define BCM2711_UART0_BASE      (BCM2711_PERI_BASE + 0x201000)  /* PL011 */
#define BCM2711_AUX_BASE        (BCM2711_PERI_BASE + 0x215000)  /* Mini UART */
#define BCM2711_SYSTIMER_BASE   (BCM2711_PERI_BASE + 0x003000)
#define BCM2711_MBOX_BASE       (BCM2711_PERI_BASE + 0x00B880)

/* ARM local peripherals (GIC, timers) */
#define BCM2711_LOCAL_BASE      0xFF800000UL
#define BCM2711_GIC_DIST_BASE   0xFF841000UL  /* GIC-400 Distributor */
#define BCM2711_GIC_CPU_BASE    0xFF842000UL  /* GIC-400 CPU Interface */

/* =============================================================================
 * MMIO Access (volatile, with memory barriers)
 * =============================================================================*/

static inline void mmio_write(uint64_t reg, uint32_t val) {
    *(volatile uint32_t *)reg = val;
}

static inline uint32_t mmio_read(uint64_t reg) {
    return *(volatile uint32_t *)reg;
}

/* Full memory barrier */
static inline void dmb(void) {
    __asm__ volatile ("dmb sy" ::: "memory");
}

static inline void dsb(void) {
    __asm__ volatile ("dsb sy" ::: "memory");
}

static inline void isb_barrier(void) {
    __asm__ volatile ("isb" ::: "memory");
}

/* =============================================================================
 * GPIO (for UART pin mux)
 * =============================================================================*/

#define GPIO_GPFSEL1            (BCM2711_GPIO_BASE + 0x04)
#define GPIO_PUP_PDN_CNTRL0    (BCM2711_GPIO_BASE + 0xE4)

static inline void gpio_setup_uart(void) {
    /* No-op: boot.S already configured GPIO 14 for mini UART (ALT5).
     * We intentionally do NOT switch to PL011 ALT0 here, because
     * that would kill the boot diagnostic UART output from boot.S. */
}

/* =============================================================================
 * PL011 UART (Primary serial console)
 *
 * UART clock on BCM2711 = 48 MHz
 * For 115200 baud: IBRD = 26, FBRD = 3
 * =============================================================================*/

/* PL011 Register offsets */
#define UART_DR         (BCM2711_UART0_BASE + 0x00)  /* Data */
#define UART_FR         (BCM2711_UART0_BASE + 0x18)  /* Flag */
#define UART_IBRD       (BCM2711_UART0_BASE + 0x24)  /* Integer baud rate */
#define UART_FBRD       (BCM2711_UART0_BASE + 0x28)  /* Fractional baud rate */
#define UART_LCRH       (BCM2711_UART0_BASE + 0x2C)  /* Line control */
#define UART_CR         (BCM2711_UART0_BASE + 0x30)  /* Control */
#define UART_IMSC       (BCM2711_UART0_BASE + 0x38)  /* Interrupt mask */
#define UART_ICR        (BCM2711_UART0_BASE + 0x44)  /* Interrupt clear */

/* Flag register bits */
#define UART_FR_RXFE    (1 << 4)    /* RX FIFO empty */
#define UART_FR_TXFF    (1 << 5)    /* TX FIFO full */
#define UART_FR_BUSY    (1 << 3)    /* UART busy */

static inline void uart_init(void) {
    /* Mini UART already initialized by boot.S at 115200 8N1.
     * We use the same mini UART (AUX) for all C serial output,
     * avoiding PL011 init that would reconfigure GPIO 14. */
}

static inline void uart_putchar(char c) {
    /* Mini UART: wait for TX empty (AUX_MU_LSR bit 5) */
    while (!(mmio_read(BCM2711_AUX_BASE + 0x54) & 0x20)) {}
    mmio_write(BCM2711_AUX_BASE + 0x40, (uint32_t)c);
}

static inline void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n') uart_putchar('\r');
        uart_putchar(*s++);
    }
}

static inline int uart_has_data(void) {
    return mmio_read(BCM2711_AUX_BASE + 0x54) & 0x01;
}

static inline char uart_getchar(void) {
    while (!uart_has_data()) {}
    return (char)(mmio_read(BCM2711_AUX_BASE + 0x40) & 0xFF);
}

/* =============================================================================
 * ARM Generic Timer (replaces x86 TSC + PIT)
 *
 * CNTFRQ_EL0: Timer frequency (typically 54 MHz on BCM2711)
 * CNTPCT_EL0: Physical counter value (always-incrementing)
 * =============================================================================*/

static inline uint64_t arm_timer_freq(void) {
    uint64_t freq;
    __asm__ volatile ("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

static inline uint64_t arm_timer_count(void) {
    uint64_t cnt;
    __asm__ volatile ("isb; mrs %0, cntpct_el0" : "=r"(cnt));
    return cnt;
}

/* Timer delay in microseconds */
static inline void arm_timer_delay_us(uint32_t us) {
    uint64_t freq = arm_timer_freq();
    uint64_t target = arm_timer_count() + (freq * us / 1000000);
    while (arm_timer_count() < target) {}
}

/* Timer delay in milliseconds */
static inline void arm_timer_delay_ms(uint32_t ms) {
    arm_timer_delay_us(ms * 1000);
}

/* =============================================================================
 * ACT LED Control (GPIO 42 on RPi4)
 *
 * boot.S already configures GPIO 42 as output before entering C,
 * so these functions can be called immediately from kernel_main.
 * Useful for debugging without HDMI/UART.
 * =============================================================================*/

#define GPIO_GPFSEL4    (BCM2711_GPIO_BASE + 0x10)
#define GPIO_GPSET1     (BCM2711_GPIO_BASE + 0x20)
#define GPIO_GPCLR1     (BCM2711_GPIO_BASE + 0x2C)

static inline void led_on(void) {
    mmio_write(GPIO_GPSET1, (1 << 10));     /* GPIO 42 = bit 10 of SET1 */
}

static inline void led_off(void) {
    mmio_write(GPIO_GPCLR1, (1 << 10));     /* GPIO 42 = bit 10 of CLR1 */
}

/* Blink the ACT LED n times (125ms on, 125ms off, 500ms gap after) */
static inline void led_blink(int n) {
    for (int i = 0; i < n; i++) {
        led_on();
        arm_timer_delay_ms(125);
        led_off();
        arm_timer_delay_ms(125);
    }
    arm_timer_delay_ms(500);
}

/* =============================================================================
 * GIC-400 (Generic Interrupt Controller)
 * Replaces x86 PIC + APIC
 * =============================================================================*/

/* GIC Distributor registers */
#define GICD_CTLR       (BCM2711_GIC_DIST_BASE + 0x000)
#define GICD_ISENABLER  (BCM2711_GIC_DIST_BASE + 0x100)
#define GICD_ICENABLER  (BCM2711_GIC_DIST_BASE + 0x180)
#define GICD_IPRIORITYR (BCM2711_GIC_DIST_BASE + 0x400)
#define GICD_ITARGETSR  (BCM2711_GIC_DIST_BASE + 0x800)
#define GICD_ICFGR      (BCM2711_GIC_DIST_BASE + 0xC00)

/* GIC CPU Interface registers */
#define GICC_CTLR       (BCM2711_GIC_CPU_BASE + 0x000)
#define GICC_PMR        (BCM2711_GIC_CPU_BASE + 0x004)
#define GICC_IAR        (BCM2711_GIC_CPU_BASE + 0x00C)
#define GICC_EOIR       (BCM2711_GIC_CPU_BASE + 0x010)

static inline void gic_init(void) {
    /* Enable GIC distributor */
    mmio_write(GICD_CTLR, 1);

    /* Enable GIC CPU interface, allow all priority levels */
    mmio_write(GICC_CTLR, 1);
    mmio_write(GICC_PMR, 0xFF);

    dsb();
}

static inline void gic_enable_irq(uint32_t irq) {
    uint32_t reg = GICD_ISENABLER + (irq / 32) * 4;
    mmio_write(reg, 1 << (irq % 32));
}

static inline uint32_t gic_acknowledge_irq(void) {
    return mmio_read(GICC_IAR);
}

static inline void gic_end_irq(uint32_t irq) {
    mmio_write(GICC_EOIR, irq);
}

/* =============================================================================
 * ARM64 Interrupt Control (replaces x86 cli/sti)
 * =============================================================================*/

static inline void arm_disable_irq(void) {
    __asm__ volatile ("msr daifset, #2" ::: "memory");  /* Mask IRQ */
}

static inline void arm_enable_irq(void) {
    __asm__ volatile ("msr daifclr, #2" ::: "memory");  /* Unmask IRQ */
}

/* =============================================================================
 * Cache Maintenance (needed for DMA coherency with VideoCore GPU)
 *
 * The GPU reads/writes physical RAM directly, but the ARM CPU caches data.
 * Before sending a buffer to the GPU: clean (flush cache → RAM).
 * After GPU writes response:        invalidate (discard stale cache).
 * DC CIVAC: Clean + Invalidate by VA to Point of Coherency.
 * =============================================================================*/

static inline int arm_mmu_enabled(void) {
    uint64_t sctlr;
    __asm__ volatile ("mrs %0, sctlr_el1" : "=r"(sctlr));
    return (sctlr & 1);  /* SCTLR_EL1.M = bit 0 */
}

static inline void arm_cache_clean_invalidate(volatile void *addr, uint32_t size)
{
    /* DC CIVAC faults on Cortex-A72 when MMU is off (IMPLEMENTATION DEFINED).
     * Without MMU+caches, RAM is accessed directly — no stale cache lines.
     * Just issue a DSB to ensure all prior stores are visible. */
    if (!arm_mmu_enabled()) {
        __asm__ volatile ("dsb sy" ::: "memory");
        return;
    }
    uint64_t line = 64;  /* Cortex-A72 cache line = 64 bytes */
    uint64_t start = (uint64_t)addr & ~(line - 1);
    uint64_t end   = (uint64_t)addr + size;
    for (uint64_t va = start; va < end; va += line)
        __asm__ volatile ("dc civac, %0" :: "r"(va) : "memory");
    __asm__ volatile ("dsb sy" ::: "memory");
}

/* =============================================================================
 * Mailbox Interface (for querying RPi firmware)
 * =============================================================================*/

#define MBOX_READ       (BCM2711_MBOX_BASE + 0x00)
#define MBOX_STATUS     (BCM2711_MBOX_BASE + 0x18)
#define MBOX_WRITE      (BCM2711_MBOX_BASE + 0x20)
#define MBOX_FULL       0x80000000
#define MBOX_EMPTY      0x40000000
#define MBOX_CHANNEL_PM 8  /* Property tags (ARM → VideoCore) */

static inline int mbox_call(uint32_t channel, volatile uint32_t *mbox_buf) {
    uint32_t buf_size = mbox_buf[0];  /* first word = total buffer size */
    uint32_t phys = (uint32_t)(uint64_t)mbox_buf;

    /* Flush ARM CPU cache to RAM so the GPU sees our data */
    arm_cache_clean_invalidate(mbox_buf, buf_size);

    /* Use 0xC0000000 bus alias so the GPU bypasses its own L2 cache
     * and reads directly from SDRAM (guaranteed coherent after our flush) */
    uint32_t bus_addr = phys | 0xC0000000;
    uint32_t msg = (bus_addr & ~0xF) | (channel & 0xF);

    /* Wait for mailbox to be not full */
    while (mmio_read(MBOX_STATUS) & MBOX_FULL) {}
    mmio_write(MBOX_WRITE, msg);

    /* Wait for response */
    while (1) {
        while (mmio_read(MBOX_STATUS) & MBOX_EMPTY) {}
        uint32_t resp = mmio_read(MBOX_READ);
        if ((resp & 0xF) == channel) {
            /* Invalidate stale cache so we read the GPU's response from RAM */
            arm_cache_clean_invalidate(mbox_buf, buf_size);
            return (mbox_buf[1] == 0x80000000);  /* Check response code */
        }
    }
}

/* =============================================================================
 * Board Info via Mailbox
 * =============================================================================*/

static inline uint32_t rpi_get_arm_memory(void) {
    volatile uint32_t __attribute__((aligned(16))) mbox[8];
    mbox[0] = 8 * 4;           /* Buffer size */
    mbox[1] = 0;               /* Request */
    mbox[2] = 0x00010005;      /* Tag: Get ARM memory */
    mbox[3] = 8;               /* Response size */
    mbox[4] = 0;               /* Request indicator */
    mbox[5] = 0;               /* Base address (filled by response) */
    mbox[6] = 0;               /* Size (filled by response) */
    mbox[7] = 0;               /* End tag */

    if (mbox_call(MBOX_CHANNEL_PM, mbox)) {
        return mbox[6];         /* Memory size in bytes */
    }
    return 0;
}

static inline uint32_t rpi_get_board_revision(void) {
    volatile uint32_t __attribute__((aligned(16))) mbox[7];
    mbox[0] = 7 * 4;
    mbox[1] = 0;
    mbox[2] = 0x00010002;      /* Tag: Get board revision */
    mbox[3] = 4;
    mbox[4] = 0;
    mbox[5] = 0;
    mbox[6] = 0;

    if (mbox_call(MBOX_CHANNEL_PM, mbox)) {
        return mbox[5];
    }
    return 0;
}

/* Include HDMI framebuffer console declarations */
#include "kernel/drivers/gpu/rpi_fb.h"

#endif /* TENSOROS_ARM64_HAL_H */
