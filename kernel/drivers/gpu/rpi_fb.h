/* =============================================================================
 * TensorOS — Raspberry Pi 4 HDMI Framebuffer Driver
 * VideoCore mailbox + 8x16 text console
 * =============================================================================*/

#ifndef RPI_FB_H
#define RPI_FB_H

#ifdef __aarch64__

#include <stdint.h>

/* Framebuffer info (populated after fb_init) */
typedef struct {
    volatile uint32_t *base;    /* Framebuffer pixel base address */
    uint32_t width;             /* Physical width (pixels) */
    uint32_t height;            /* Physical height (pixels) */
    uint32_t pitch;             /* Bytes per row */
    uint32_t depth;             /* Bits per pixel (32) */
    uint32_t size;              /* Total framebuffer size in bytes */

    /* Text console state */
    uint32_t cols;              /* Characters per row */
    uint32_t rows;              /* Character rows */
    uint32_t cursor_x;         /* Current text column */
    uint32_t cursor_y;         /* Current text row */
    uint32_t fg_color;         /* Foreground (0xAARRGGBB) */
    uint32_t bg_color;         /* Background */
    int      ready;            /* 1 if framebuffer is available */
} fb_state_t;

extern fb_state_t fb;

/* Initialise HDMI framebuffer via VideoCore mailbox.
 * Returns 0 on success, -1 on failure.
 * On success, fb.ready=1 and text console is usable. */
int  fb_init(void);

/* Console output */
void fb_putchar(char c);
void fb_puts(const char *s);
void fb_clear(void);

/* Pixel-level access */
void fb_fill_rect(uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint32_t color);
void fb_draw_char(uint32_t col, uint32_t row, char c, uint32_t fg, uint32_t bg);

#endif /* __aarch64__ */
#endif /* RPI_FB_H */
