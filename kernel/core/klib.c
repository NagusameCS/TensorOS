/* =============================================================================
 * TensorOS - Kernel Library (klib)
 *
 * Provides core runtime functions: VGA console, kprintf, string/memory utils,
 * interrupt setup stubs, and panic handler.
 * =============================================================================*/

#include "kernel/core/kernel.h"

/* =============================================================================
 * VGA Text Mode Console (x86 only — RPi has no VGA)
 * =============================================================================*/

#if defined(__aarch64__)
/* ARM64: HDMI framebuffer console + UART serial output.
 * fb_init() requests a 1280x720 framebuffer from the VideoCore GPU
 * via the mailbox, then all text is rendered with an 8x8 bitmap font. */
void vga_init(void) {
    uart_init();
    if (fb_init() == 0)
        uart_puts("[FB] HDMI framebuffer 1280x720@32bpp ready\r\n");
    else
        uart_puts("[FB] HDMI init failed — serial only\r\n");
}
static void vga_putchar(char c) { fb_putchar(c); }
static void vga_puts(const char *s) { fb_puts(s); }
__attribute__((unused))
static void vga_scroll(void) { /* handled internally by fb_putchar */ }
#else
#define VGA_ADDR   0xB8000
#define VGA_WIDTH  80
#define VGA_HEIGHT 25
#define VGA_ATTR   0x0F  /* White on black */

static uint16_t *vga_buffer = (uint16_t *)VGA_ADDR;
static int vga_col = 0;
static int vga_row = 0;

void vga_init(void)
{
    vga_buffer = (uint16_t *)VGA_ADDR;
    vga_col = 0;
    vga_row = 0;
    /* Clear screen */
    for (int i = 0; i < VGA_WIDTH * VGA_HEIGHT; i++)
        vga_buffer[i] = (VGA_ATTR << 8) | ' ';
}

static void vga_scroll(void)
{
    for (int i = 0; i < VGA_WIDTH * (VGA_HEIGHT - 1); i++)
        vga_buffer[i] = vga_buffer[i + VGA_WIDTH];
    for (int i = VGA_WIDTH * (VGA_HEIGHT - 1); i < VGA_WIDTH * VGA_HEIGHT; i++)
        vga_buffer[i] = (VGA_ATTR << 8) | ' ';
    vga_row = VGA_HEIGHT - 1;
}

static void vga_putchar(char c)
{
    if (c == '\n') {
        vga_col = 0;
        vga_row++;
    } else if (c == '\r') {
        vga_col = 0;
    } else if (c == '\t') {
        vga_col = (vga_col + 8) & ~7;
    } else {
        vga_buffer[vga_row * VGA_WIDTH + vga_col] = (VGA_ATTR << 8) | (uint8_t)c;
        vga_col++;
    }
    if (vga_col >= VGA_WIDTH) {
        vga_col = 0;
        vga_row++;
    }
    if (vga_row >= VGA_HEIGHT)
        vga_scroll();
}

static void vga_puts(const char *s)
{
    while (*s)
        vga_putchar(*s++);
}
#endif /* __aarch64__ */

/* Also output to serial port COM1 (0x3F8) for QEMU -serial stdio */
#if defined(__aarch64__)
/* ARM64: Serial output via mini UART + Bluetooth SPP mirror + HDMI framebuffer */
#include "kernel/drivers/gpu/rpi_fb.h"
static void serial_init(void) { /* Already done in vga_init() */ }
static void serial_putchar(char c) { uart_putchar(c); bt_putchar(c); fb_putchar(c); }
static void serial_puts(const char *s) { uart_puts(s); while (*s) { bt_putchar(*s); fb_putchar(*s); s++; } }
#else
#define COM1 0x3F8

static void serial_init(void)
{
    outb(COM1 + 1, 0x00);  /* Disable interrupts */
    outb(COM1 + 3, 0x80);  /* Enable DLAB */
    outb(COM1 + 0, 0x03);  /* 38400 baud (divisor = 3) */
    outb(COM1 + 1, 0x00);
    outb(COM1 + 3, 0x03);  /* 8 bits, no parity, 1 stop bit */
    outb(COM1 + 2, 0xC7);  /* Enable FIFO */
    outb(COM1 + 4, 0x0B);  /* IRQs enabled, RTS/DSR set */
}

static void serial_putchar(char c)
{
    /* Also write to QEMU debug console port 0xE9 (always works) */
    outb(0xE9, (uint8_t)c);
    while (!(inb(COM1 + 5) & 0x20))
        ;
    outb(COM1, (uint8_t)c);
}

static void serial_puts(const char *s)
{
    while (*s) {
        if (*s == '\n')
            serial_putchar('\r');
        serial_putchar(*s++);
    }
}
#endif /* __aarch64__ */

/* =============================================================================
 * kprintf - Minimal kernel printf
 *
 * Supports: %s, %d, %u, %x, %lx, %lu, %ld, %p, %c, %%
 * =============================================================================*/

static void kprint_uint64(uint64_t val, int base, int is_upper)
{
    char buf[24];
    int  i = 0;
    const char *digits = is_upper ? "0123456789ABCDEF" : "0123456789abcdef";

    if (val == 0) {
        vga_putchar('0');
        serial_putchar('0');
        return;
    }
    while (val > 0) {
        buf[i++] = digits[val % base];
        val /= base;
    }
    while (--i >= 0) {
        vga_putchar(buf[i]);
        serial_putchar(buf[i]);
    }
}

static void kprint_int64(int64_t val)
{
    if (val < 0) {
        vga_putchar('-');
        serial_putchar('-');
        kprint_uint64((uint64_t)(-val), 10, 0);
    } else {
        kprint_uint64((uint64_t)val, 10, 0);
    }
}

/* Float formatting: integer.fraction with rounding */
static void kprint_float(double val, int precision)
{
    if (val < 0.0) {
        vga_putchar('-');
        serial_putchar('-');
        val = -val;
    }
    /* Round to given precision */
    double mult = 1.0;
    for (int p = 0; p < precision; p++) mult *= 10.0;
    if (mult > 0.0) val += 0.5 / mult;
    uint64_t integer_part = (uint64_t)val;
    kprint_uint64(integer_part, 10, 0);
    if (precision > 0) {
        vga_putchar('.');
        serial_putchar('.');
        double frac = val - (double)integer_part;
        for (int p = 0; p < precision; p++) {
            frac *= 10.0;
            int digit = (int)frac;
            if (digit > 9) digit = 9;
            if (digit < 0) digit = 0;
            vga_putchar('0' + digit);
            serial_putchar('0' + digit);
            frac -= digit;
        }
    }
}

int kprintf(const char *fmt, ...)
{
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    int count = 0;

    while (*fmt) {
        if (*fmt != '%') {
            vga_putchar(*fmt);
            if (*fmt == '\n') serial_putchar('\r');
            serial_putchar(*fmt);
            fmt++;
            count++;
            continue;
        }
        fmt++; /* skip '%' */

        /* Parse optional precision (.N) for %f */
        int precision = 4; /* default 4 decimal places */
        if (*fmt == '.') {
            fmt++;
            precision = 0;
            while (*fmt >= '0' && *fmt <= '9') {
                precision = precision * 10 + (*fmt - '0');
                fmt++;
            }
        }

        /* Check for 'l' modifier */
        int is_long = 0;
        if (*fmt == 'l') { is_long = 1; fmt++; }

        switch (*fmt) {
        case 'f': {
            double val = __builtin_va_arg(ap, double);
            kprint_float(val, precision);
            break;
        }
        case 's': {
            const char *s = __builtin_va_arg(ap, const char *);
            if (!s) s = "(null)";
            vga_puts(s);
            serial_puts(s);
            break;
        }
        case 'd': {
            if (is_long) {
                int64_t v = __builtin_va_arg(ap, int64_t);
                kprint_int64(v);
            } else {
                int32_t v = __builtin_va_arg(ap, int32_t);
                kprint_int64(v);
            }
            break;
        }
        case 'u': {
            if (is_long) {
                uint64_t v = __builtin_va_arg(ap, uint64_t);
                kprint_uint64(v, 10, 0);
            } else {
                uint32_t v = __builtin_va_arg(ap, uint32_t);
                kprint_uint64(v, 10, 0);
            }
            break;
        }
        case 'x': {
            if (is_long) {
                uint64_t v = __builtin_va_arg(ap, uint64_t);
                kprint_uint64(v, 16, 0);
            } else {
                uint32_t v = __builtin_va_arg(ap, uint32_t);
                kprint_uint64(v, 16, 0);
            }
            break;
        }
        case 'X': {
            if (is_long) {
                uint64_t v = __builtin_va_arg(ap, uint64_t);
                kprint_uint64(v, 16, 1);
            } else {
                uint32_t v = __builtin_va_arg(ap, uint32_t);
                kprint_uint64(v, 16, 1);
            }
            break;
        }
        case 'p': {
            uint64_t v = (uint64_t)__builtin_va_arg(ap, void *);
            vga_puts("0x");
            serial_puts("0x");
            kprint_uint64(v, 16, 0);
            break;
        }
        case 'c': {
            char c = (char)__builtin_va_arg(ap, int);
            vga_putchar(c);
            serial_putchar(c);
            break;
        }
        case '%':
            vga_putchar('%');
            serial_putchar('%');
            break;
        default:
            vga_putchar('%');
            serial_putchar('%');
            vga_putchar(*fmt);
            serial_putchar(*fmt);
            break;
        }
        fmt++;
        count++;
    }

    __builtin_va_end(ap);
    return count;
}

static void kvprintf_serial(const char *fmt, __builtin_va_list ap)
{
    while (*fmt) {
        if (*fmt != '%') {
            serial_putchar(*fmt);
            fmt++;
            continue;
        }
        fmt++;
        /* Parse optional precision (.N) for %f */
        int precision = 4;
        if (*fmt == '.') {
            fmt++;
            precision = 0;
            while (*fmt >= '0' && *fmt <= '9') {
                precision = precision * 10 + (*fmt - '0');
                fmt++;
            }
        }
        int is_long = 0;
        if (*fmt == 'l') { is_long = 1; fmt++; }
        if (*fmt == 'l') { is_long = 2; fmt++; }  /* ll */
        switch (*fmt) {
        case 'f': {
            double val = __builtin_va_arg(ap, double);
            if (val < 0.0) { serial_putchar('-'); val = -val; }
            double m = 1.0;
            for (int p = 0; p < precision; p++) m *= 10.0;
            if (m > 0.0) val += 0.5 / m;
            uint64_t ip = (uint64_t)val;
            char nb[24]; int ni = 0;
            if (ip == 0) { serial_putchar('0'); }
            else { while (ip > 0) { nb[ni++] = '0' + (ip % 10); ip /= 10; } while (--ni >= 0) serial_putchar(nb[ni]); }
            if (precision > 0) {
                serial_putchar('.');
                double fr = val - (double)((uint64_t)val);
                for (int p = 0; p < precision; p++) {
                    fr *= 10.0; int d = (int)fr;
                    if (d > 9) d = 9; if (d < 0) d = 0;
                    serial_putchar('0' + d); fr -= d;
                }
            }
            break;
        }
        case 's': {
            const char *s = __builtin_va_arg(ap, const char *);
            serial_puts(s ? s : "(null)");
            break;
        }
        case 'd': {
            int64_t v = is_long ? __builtin_va_arg(ap, int64_t) : (int64_t)__builtin_va_arg(ap, int32_t);
            if (v < 0) { serial_putchar('-'); v = -v; }
            char buf[24]; int i = 0;
            if (v == 0) { serial_putchar('0'); }
            else { while (v > 0) { buf[i++] = '0' + (v % 10); v /= 10; } while (--i >= 0) serial_putchar(buf[i]); }
            break;
        }
        case 'u': case 'x': case 'X': {
            uint64_t v = is_long ? __builtin_va_arg(ap, uint64_t) : (uint64_t)__builtin_va_arg(ap, uint32_t);
            int base = (*fmt == 'u') ? 10 : 16;
            const char *dig = (*fmt == 'X') ? "0123456789ABCDEF" : "0123456789abcdef";
            char buf[24]; int i = 0;
            if (v == 0) { serial_putchar('0'); }
            else { while (v > 0) { buf[i++] = dig[v % base]; v /= base; } while (--i >= 0) serial_putchar(buf[i]); }
            break;
        }
        case 'p': {
            uint64_t v = (uint64_t)__builtin_va_arg(ap, void *);
            serial_puts("0x");
            char buf[24]; int i = 0;
            if (v == 0) { serial_putchar('0'); }
            else { while (v > 0) { buf[i++] = "0123456789abcdef"[v % 16]; v /= 16; } while (--i >= 0) serial_putchar(buf[i]); }
            break;
        }
        case 'c': serial_putchar((char)__builtin_va_arg(ap, int)); break;
        case '%': serial_putchar('%'); break;
        default: serial_putchar('%'); serial_putchar(*fmt); break;
        }
        fmt++;
    }
}

int kprintf_debug(const char *fmt, ...)
{
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    serial_puts("[DBG] ");
    kvprintf_serial(fmt, ap);
    __builtin_va_end(ap);
    return 0;
}

/* =============================================================================
 * Panic
 * =============================================================================*/

void kpanic(const char *msg)
{
    kprintf("\n\n*** KERNEL PANIC ***\n%s\n", msg);
    kprintf("System halted.\n");
    cli();
    for (;;)
#if defined(__aarch64__)
        __asm__ volatile ("wfi");
#else
        __asm__ volatile ("hlt");
#endif
    __builtin_unreachable();
}

/* =============================================================================
 * Memory Utilities — Architecture-Optimized
 *
 * Uses x86_64 REP STOSB/MOVSB (Enhanced REP MOVSB on modern CPUs).
 * ERMS is available on Ivy Bridge+ and provides ~32 bytes/cycle throughput,
 * compared to ~1 byte/cycle for the naive loop. For large copies the CPU
 * microcode uses 256-bit internal stores, matching or beating SSE2/AVX2.
 *
 * Fallback: for very small sizes (<= 64 bytes) we use direct 8-byte stores
 * which avoid the REP setup overhead (~35 cycles).
 * =============================================================================*/

void *kmemset(void *s, int c, size_t n)
{
#ifndef __aarch64__
    /* Fast path: small fills with 8-byte stores */
    if (n <= 64) {
        uint8_t *p = (uint8_t *)s;
        while (n >= 8) {
            uint64_t fill8 = (uint8_t)c * 0x0101010101010101ULL;
            *(uint64_t *)p = fill8;
            p += 8; n -= 8;
        }
        while (n--) *p++ = (uint8_t)c;
        return s;
    }
    /* REP STOSB: RDI=dest, RCX=count, AL=value */
    __asm__ volatile (
        "rep stosb"
        : "+D"(s), "+c"(n)
        : "a"((uint8_t)c)
        : "memory"
    );
    return s;
#else
    uint8_t *p = (uint8_t *)s;
    while (n--) *p++ = (uint8_t)c;
    return s;
#endif
}

void *kmemcpy(void *dest, const void *src, size_t n)
{
#ifndef __aarch64__
    void *ret = dest;
    /* Fast path: small copies with 8-byte loads/stores */
    if (n <= 64) {
        uint8_t *d = (uint8_t *)dest;
        const uint8_t *s = (const uint8_t *)src;
        while (n >= 8) {
            *(uint64_t *)d = *(const uint64_t *)s;
            d += 8; s += 8; n -= 8;
        }
        while (n--) *d++ = *s++;
        return ret;
    }
    /* REP MOVSB: RDI=dest, RSI=src, RCX=count (direction flag clear by ABI) */
    __asm__ volatile (
        "rep movsb"
        : "+D"(dest), "+S"(src), "+c"(n)
        :
        : "memory"
    );
    return ret;
#else
    uint8_t *d = (uint8_t *)dest;
    const uint8_t *s = (const uint8_t *)src;
    while (n--) *d++ = *s++;
    return dest;
#endif
}

/* Standard libc symbols — compiler may emit calls at -O2 */
void *memcpy(void *dest, const void *src, size_t n)  { return kmemcpy(dest, src, n); }
void *memset(void *s, int c, size_t n)               { return kmemset(s, c, n); }
void *memmove(void *dest, const void *src, size_t n) {
#ifndef __aarch64__
    if (dest == src || n == 0) return dest;
    if ((uint8_t *)dest < (const uint8_t *)src) {
        return kmemcpy(dest, src, n);
    }
    /* Backward copy: set direction flag, use REP MOVSB backward */
    uint8_t *d = (uint8_t *)dest + n - 1;
    const uint8_t *s2 = (const uint8_t *)src + n - 1;
    __asm__ volatile (
        "std\n\t"
        "rep movsb\n\t"
        "cld"
        : "+D"(d), "+S"(s2), "+c"(n)
        :
        : "memory"
    );
    return dest;
#else
    uint8_t *d = (uint8_t *)dest;
    const uint8_t *s2 = (const uint8_t *)src;
    if (d < s2) { while (n--) *d++ = *s2++; }
    else { d += n; s2 += n; while (n--) *--d = *--s2; }
    return dest;
#endif
}
int memcmp(const void *a, const void *b, size_t n) {
    const uint8_t *p = a, *q = b;
    for (size_t i = 0; i < n; i++) { if (p[i] != q[i]) return p[i] - q[i]; }
    return 0;
}

/* =============================================================================
 * String Utilities
 * =============================================================================*/

int kstrcmp(const char *s1, const char *s2)
{
    while (*s1 && *s1 == *s2) {
        s1++;
        s2++;
    }
    return (unsigned char)*s1 - (unsigned char)*s2;
}

int kstrncmp(const char *s1, const char *s2, size_t n)
{
    while (n && *s1 && *s1 == *s2) {
        s1++;
        s2++;
        n--;
    }
    if (n == 0) return 0;
    return (unsigned char)*s1 - (unsigned char)*s2;
}

size_t kstrlen(const char *s)
{
    size_t len = 0;
    while (*s++) len++;
    return len;
}

char *kstrcpy(char *dest, const char *src)
{
    char *d = dest;
    while (*src)
        *d++ = *src++;
    *d = '\0';
    return dest;
}

char *kstrncpy(char *dest, const char *src, size_t n)
{
    char *d = dest;
    while (n && *src) {
        *d++ = *src++;
        n--;
    }
    while (n--)
        *d++ = '\0';
    return dest;
}

/* =============================================================================
 * Platform-Specific Hardware: Interrupt, Keyboard, IDT, PIC, PIT, CPU, PCI
 * =============================================================================*/

#if defined(__aarch64__)
/* =========================================================================
 * ARM64 Platform Implementation
 * =========================================================================*/

/* Simple keyboard buffer for UART serial input */
#define KB_BUF_SIZE 256
static volatile char kb_buf[KB_BUF_SIZE];
static volatile int  kb_head = 0;
static volatile int  kb_tail = 0;

char keyboard_getchar(void) {
    while (1) {
        if (kb_tail != kb_head) {
            char c = kb_buf[kb_tail];
            kb_tail = (kb_tail + 1) % KB_BUF_SIZE;
            return c;
        }
        if (uart_has_data()) {
            char c = uart_getchar();
            if (c == '\r') c = '\n';
            return c;
        }
        /* Also accept input from Bluetooth SPP console */
        bt_poll();
        if (bt_has_data()) {
            char c = bt_getchar();
            if (c == '\r') c = '\n';
            return c;
        }
        __asm__ volatile ("wfi");
    }
}

int keyboard_has_key(void) {
    if (kb_tail != kb_head) return 1;
    if (uart_has_data()) return 1;
    bt_poll();
    if (bt_has_data()) return 1;
    return 0;
}

/* ARM64 IDT equivalent: exception vector table */
void idt_init(void) {
    gic_init();
    kprintf("[GIC] GIC-400 initialized\n");
    kprintf("[IDT] ARM64 exception vectors installed\n");
}

void idt_set_gate(int num, uint64_t handler) {
    (void)num; (void)handler;
}

/* ARM64: no PIC, we use GIC */
void pic_init(void) {
    kprintf("[GIC] Interrupt controller ready\n");
}

/* ARM64: use generic timer instead of PIT */
void timer_init(uint32_t freq_hz) {
    (void)freq_hz;
    kprintf("[TIMER] ARM generic timer (%lu MHz)\n",
            arm_timer_freq() / 1000000);
}

/* ARM64 CPU detection via MIDR_EL1 */
int cpu_detect_and_init(void) {
    uint64_t midr;
    __asm__ volatile ("mrs %0, midr_el1" : "=r"(midr));
    uint32_t implementer = (midr >> 24) & 0xFF;
    uint32_t part = (midr >> 4) & 0xFFF;
    const char *impl_name = "Unknown";
    if (implementer == 0x41) impl_name = "ARM";
    else if (implementer == 0x42) impl_name = "Broadcom";
    const char *core_name = "Unknown";
    if (part == 0xD08) core_name = "Cortex-A72";
    else if (part == 0xD0B) core_name = "Cortex-A76";
    kprintf("[CPU] %s %s (MIDR: %lx)\n", impl_name, core_name, midr);
    return 4;  /* RPi 4 has 4 cores */
}

/* ARM64: no port-I/O PCI — use ECAM MMIO or skip */
void pci_enumerate(void) {
    kprintf("[PCI] ARM64: PCIe ECAM (not enumerated, using DT)\n");
}

#else
/* =========================================================================
 * x86_64 Platform Implementation
 * =========================================================================*/

/* IDT entry structure */
struct idt_entry {
    uint16_t offset_lo;
    uint16_t selector;
    uint8_t  ist;
    uint8_t  type_attr;
    uint16_t offset_mid;
    uint32_t offset_hi;
    uint32_t reserved;
} __attribute__((packed));

struct idt_ptr {
    uint16_t limit;
    uint64_t base;
} __attribute__((packed));

static struct idt_entry idt[256];
static struct idt_ptr   idtr;

/* =============================================================================
 * Keyboard Ring Buffer
 * =============================================================================*/

#define KB_BUF_SIZE 256
static volatile char kb_buf[KB_BUF_SIZE];
static volatile int  kb_head = 0;
static volatile int  kb_tail = 0;
static volatile int  kb_shift = 0;
static volatile int  kb_caps = 0;

static const char scancode_to_ascii[128] = {
    0,  27, '1','2','3','4','5','6','7','8','9','0','-','=','\b',
    '\t','q','w','e','r','t','y','u','i','o','p','[',']','\n',
     0, 'a','s','d','f','g','h','j','k','l',';','\'','`',
     0, '\\','z','x','c','v','b','n','m',',','.','/', 0,
    '*', 0, ' ', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};

static const char scancode_to_ascii_shift[128] = {
    0,  27, '!','@','#','$','%','^','&','*','(',')','_','+','\b',
    '\t','Q','W','E','R','T','Y','U','I','O','P','{','}','\n',
     0, 'A','S','D','F','G','H','J','K','L',':','\"','~',
     0, '|','Z','X','C','V','B','N','M','<','>','?', 0,
    '*', 0, ' ', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};

/* Called from keyboard ISR — reads scancode, translates, buffers */
void keyboard_irq_handler(void)
{
    uint8_t sc = inb(0x60);

    /* Track shift keys (make=0x2A/0x36, break=0xAA/0xB6) */
    if (sc == 0x2A || sc == 0x36) { kb_shift = 1; return; }
    if (sc == 0xAA || sc == 0xB6) { kb_shift = 0; return; }

    /* Caps Lock toggle (make=0x3A) */
    if (sc == 0x3A) { kb_caps ^= 1; return; }

    /* Ignore key release (bit 7 set) */
    if (sc & 0x80) return;
    if (sc >= 128) return;

    char c;
    if (kb_shift)
        c = scancode_to_ascii_shift[sc];
    else
        c = scancode_to_ascii[sc];

    /* Apply caps lock to letters */
    if (kb_caps && c >= 'a' && c <= 'z') c -= 32;
    else if (kb_caps && c >= 'A' && c <= 'Z') c += 32;

    if (c == 0) return;

    /* Insert into ring buffer */
    int next = (kb_head + 1) % KB_BUF_SIZE;
    if (next != kb_tail) {
        kb_buf[kb_head] = c;
        kb_head = next;
    }
}

/* Public API: blocking read one character from keyboard buffer or serial */
char keyboard_getchar(void)
{
    while (1) {
        /* Check keyboard buffer first */
        if (kb_tail != kb_head) {
            char c = kb_buf[kb_tail];
            kb_tail = (kb_tail + 1) % KB_BUF_SIZE;
            return c;
        }
        /* Check serial port (COM1) for input — supports serial console */
        if (inb(COM1 + 5) & 0x01) {
            char c = (char)inb(COM1);
            if (c == '\r') c = '\n';
            return c;
        }
        __asm__ volatile ("hlt");  /* Sleep until next IRQ */
    }
}

/* Public API: non-blocking check if key available */
int keyboard_has_key(void)
{
    return kb_tail != kb_head;
}

/* =============================================================================
 * ISR stubs via inline asm
 * =============================================================================*/

/* Default ISR: just send EOI and return */
extern void isr_stub(void);
__asm__(
    ".globl isr_stub\n"
    "isr_stub:\n"
    "  push %rax\n"
    "  mov $0x20, %al\n"
    "  outb %al, $0x20\n"
    "  pop %rax\n"
    "  iretq\n"
);

/* Keyboard ISR (IRQ1 = INT 0x21): call C handler, send EOI */
extern void isr_keyboard(void);
__asm__(
    ".globl isr_keyboard\n"
    "isr_keyboard:\n"
    "  push %rax\n"
    "  push %rcx\n"
    "  push %rdx\n"
    "  push %rsi\n"
    "  push %rdi\n"
    "  push %r8\n"
    "  push %r9\n"
    "  push %r10\n"
    "  push %r11\n"
    "  call keyboard_irq_handler\n"
    "  mov $0x20, %al\n"
    "  outb %al, $0x20\n"
    "  pop %r11\n"
    "  pop %r10\n"
    "  pop %r9\n"
    "  pop %r8\n"
    "  pop %rdi\n"
    "  pop %rsi\n"
    "  pop %rdx\n"
    "  pop %rcx\n"
    "  pop %rax\n"
    "  iretq\n"
);

void idt_set_gate(int num, uint64_t handler)
{
    idt[num].offset_lo  = (uint16_t)(handler & 0xFFFF);
    idt[num].selector   = 0x08;
    idt[num].ist        = 0;
    idt[num].type_attr  = 0x8E;
    idt[num].offset_mid = (uint16_t)((handler >> 16) & 0xFFFF);
    idt[num].offset_hi  = (uint32_t)((handler >> 32) & 0xFFFFFFFF);
    idt[num].reserved   = 0;
}

void idt_init(void)
{
    kmemset(idt, 0, sizeof(idt));

    /* Set all 256 entries to default stub (IRQs) */
    for (int i = 0; i < 256; i++)
        idt_set_gate(i, (uint64_t)(uintptr_t)isr_stub);

    /* Install production-grade CPU exception handlers (vectors 0-31) */
    exception_install_handlers();

    /* Install watchdog timer ISR (IRQ0 = vector 0x20) */
    watchdog_install();

    /* Override IRQ1 (keyboard) with dedicated handler */
    idt_set_gate(0x21, (uint64_t)(uintptr_t)isr_keyboard);

    idtr.limit = sizeof(idt) - 1;
    idtr.base  = (uint64_t)(uintptr_t)&idt[0];
    __asm__ volatile ("lidt %0" : : "m"(idtr));

    kprintf("[IDT] Loaded with 256 entries (32 exception + 224 IRQ)\n");
}

/* PIC remapping */
void pic_init(void)
{
    /* ICW1: begin init sequence */
    outb(0x20, 0x11);
    outb(0xA0, 0x11);

    /* ICW2: remap IRQs */
    outb(0x21, 0x20);  /* Master PIC: IRQ 0-7 → INT 0x20-0x27 */
    outb(0xA1, 0x28);  /* Slave PIC:  IRQ 8-15 → INT 0x28-0x2F */

    /* ICW3 */
    outb(0x21, 0x04);
    outb(0xA1, 0x02);

    /* ICW4: 8086 mode */
    outb(0x21, 0x01);
    outb(0xA1, 0x01);

    /* Mask all IRQs except timer (IRQ0), keyboard (IRQ1), and COM1 (IRQ4) */
    outb(0x21, 0xEC);  /* 0xEC = 11101100b: unmask IRQ0, IRQ1, IRQ4 */
    outb(0xA1, 0xFF);

    kprintf("[PIC] Remapped IRQs 0x20-0x2F\n");
}

/* PIT Timer */
void timer_init(uint32_t freq_hz)
{
    uint32_t divisor = 1193180 / freq_hz;
    outb(0x43, 0x36);                   /* Channel 0, lo/hi byte, square wave */
    outb(0x40, (uint8_t)(divisor & 0xFF));
    outb(0x40, (uint8_t)((divisor >> 8) & 0xFF));
    kprintf("[TIMER] PIT configured at %u Hz\n", freq_hz);
}

/* =============================================================================
 * CPU Detection (stub)
 * =============================================================================*/

int cpu_detect_and_init(void)
{
    /* CPUID basic detection */
    uint32_t eax, ebx, ecx, edx;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0));

    char vendor[13];
    *(uint32_t *)&vendor[0] = ebx;
    *(uint32_t *)&vendor[4] = edx;
    *(uint32_t *)&vendor[8] = ecx;
    vendor[12] = '\0';
    kprintf("[CPU] Vendor: %s\n", vendor);

    return 1;  /* At least the BSP; SMP updates later */
}

/* =============================================================================
 * PCI Enumeration (stub)
 * =============================================================================*/

void pci_enumerate(void)
{
    kprintf("[PCI] Enumerating devices...\n");
    /* Scan bus 0 for devices */
    for (uint32_t dev = 0; dev < 32; dev++) {
        uint32_t addr = (1 << 31) | (0 << 16) | (dev << 11) | (0 << 8);
        outl(0xCF8, addr);
        uint32_t vendor_device = inl(0xCFC);
        if ((vendor_device & 0xFFFF) != 0xFFFF) {
            kprintf("[PCI] %u:%u.0 vendor=%x device=%x\n",
                    0, dev,
                    vendor_device & 0xFFFF,
                    (vendor_device >> 16) & 0xFFFF);
        }
    }
}

#endif /* __aarch64__ / x86_64 platform split */

/* =============================================================================
 * Networking (stub)
 * =============================================================================*/

void net_init(void)
{
    kprintf("[NET] Network init (stub)\n");
}

/* =============================================================================
 * Global Kernel State
 * =============================================================================*/

/* kstate is defined in main.c */

/* =============================================================================
 * Serial init helper called early
 * =============================================================================*/

void klib_early_init(void)
{
    serial_init();
}

/* long_mode_entry is now in kernel/core/entry.c (must be first object linked) */
