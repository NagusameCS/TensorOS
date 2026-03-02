/* =============================================================================
 * TensorOS — SD Card Boot Logger
 *
 * Writes verbose boot logs to BOOTLOG.TXT on the FAT32 boot partition.
 * The file must be pre-created (32 KB of spaces) by the flash script.
 * At boot, we parse FAT32 minimally to find the file's data sectors,
 * then overwrite them with log text.  Pull the SD card to read on PC.
 *
 * API:
 *   sdlog_init()           — init SD, find BOOTLOG.TXT
 *   sdlog(msg)             — append a line to the log buffer
 *   sdlog_flush()          — write buffer to SD card
 *   sdlog_panic(msg)       — log + flush + LED rapid blink
 * =============================================================================*/

#ifndef TENSOROS_SDLOG_H
#define TENSOROS_SDLOG_H

#include <stdint.h>

/* Forward declarations from rpi_sd.h */
int  sd_init(void);
int  sd_read_sector(uint32_t lba, void *buf);
int  sd_write_sector(uint32_t lba, const void *buf);

/* ---- Configuration ---- */
#define SDLOG_MAX_BYTES   (32 * 1024)   /* 32 KB log file (64 sectors) */
#define SDLOG_MAX_SECTORS (SDLOG_MAX_BYTES / 512)

/* ---- State ---- */
static uint32_t sdlog_file_sector = 0;      /* First sector of BOOTLOG.TXT data */
static uint32_t sdlog_file_sectors = 0;     /* Number of contiguous sectors */
static char     sdlog_buf[SDLOG_MAX_BYTES]; /* RAM log buffer */
static uint32_t sdlog_pos = 0;             /* Current write position in buffer */
static int      sdlog_ready = 0;           /* 1 if init succeeded */

/* Temp sector buffer (512 bytes, 4-byte aligned) */
static uint8_t  sdlog_sector_buf[512] __attribute__((aligned(4)));

/* ---- Helpers ---- */

/* Simple strlen */
__attribute__((unused))
static uint32_t sdlog_strlen(const char *s) {
    uint32_t n = 0;
    while (*s++) n++;
    return n;
}

/* Simple memcmp */
__attribute__((unused))
static int sdlog_memcmp(const void *a, const void *b, uint32_t n) {
    const uint8_t *pa = (const uint8_t *)a;
    const uint8_t *pb = (const uint8_t *)b;
    for (uint32_t i = 0; i < n; i++) {
        if (pa[i] != pb[i]) return (int)pa[i] - (int)pb[i];
    }
    return 0;
}

/* Read uint16 LE from buffer */
__attribute__((unused))
static inline uint16_t rd16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

/* Read uint32 LE from buffer */
__attribute__((unused))
static inline uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* Append a decimal number to the log buffer */
__attribute__((unused))
static void sdlog_put_num(int32_t v) {
    if (sdlog_pos >= SDLOG_MAX_BYTES - 16) return;
    if (v < 0) {
        sdlog_buf[sdlog_pos++] = '-';
        v = -v;
    }
    char tmp[12];
    int len = 0;
    if (v == 0) { tmp[len++] = '0'; }
    else { while (v > 0) { tmp[len++] = '0' + (v % 10); v /= 10; } }
    for (int i = len - 1; i >= 0; i--)
        sdlog_buf[sdlog_pos++] = tmp[i];
}

/* Append a hex number to the log buffer */
__attribute__((unused))
static void sdlog_put_hex(uint64_t v) {
    if (sdlog_pos >= SDLOG_MAX_BYTES - 20) return;
    static const char hex[] = "0123456789ABCDEF";
    sdlog_buf[sdlog_pos++] = '0';
    sdlog_buf[sdlog_pos++] = 'x';
    /* Find highest non-zero nibble */
    int started = 0;
    for (int i = 60; i >= 0; i -= 4) {
        int digit = (v >> i) & 0xF;
        if (digit || started || i == 0) {
            sdlog_buf[sdlog_pos++] = hex[digit];
            started = 1;
        }
    }
}

/* =============================================================================
 * sdlog_init — Initialize SD card and find BOOTLOG.TXT
 *
 * Parses:
 *   1. MBR (sector 0) → partition 1 start LBA
 *   2. FAT32 BPB (VBR) → cluster geometry
 *   3. Root directory → find BOOTLOG.TXT entry → starting cluster
 *   4. Walk FAT chain for contiguous sectors (up to 64 sectors / 32 KB)
 *
 * Returns 0 on success, negative on failure.
 * =============================================================================*/
__attribute__((unused))
static int sdlog_init(void)
{
    sdlog_pos = 0;
    sdlog_ready = 0;

    /* Init SD card hardware — propagate the exact error code.
     * sd_init returns negative values; we shift them to -100..-199
     * so sdlog-level errors (-1..-20) are distinguishable. */
    int rc = sd_init();
    if (rc != 0) return -100 + rc;  /* e.g. sd_init=-3 → sdlog=-103 */

    /* Read MBR (sector 0) */
    if (sd_read_sector(0, sdlog_sector_buf) != 0) return -2;

    /* Check MBR signature */
    if (sdlog_sector_buf[510] != 0x55 || sdlog_sector_buf[511] != 0xAA)
        return -3;

    /* Get partition 1 start LBA (offset 0x1BE + 0x08 = 0x1C6) */
    uint32_t part_start = rd32(&sdlog_sector_buf[0x1C6]);
    if (part_start == 0) return -4;

    /* Read VBR / BPB (first sector of partition) */
    if (sd_read_sector(part_start, sdlog_sector_buf) != 0) return -5;

    /* Parse FAT32 BPB fields */
    uint16_t bytes_per_sec  = rd16(&sdlog_sector_buf[0x0B]);
    uint8_t  sec_per_clus   = sdlog_sector_buf[0x0D];
    uint16_t reserved_sec   = rd16(&sdlog_sector_buf[0x0E]);
    uint8_t  num_fats       = sdlog_sector_buf[0x10];
    uint32_t fat_size       = rd32(&sdlog_sector_buf[0x24]);
    uint32_t root_cluster   = rd32(&sdlog_sector_buf[0x2C]);

    if (bytes_per_sec != 512) return -6;
    if (sec_per_clus == 0 || num_fats == 0) return -7;

    /* Data region starts after reserved + FATs */
    uint32_t fat_start    = part_start + reserved_sec;
    uint32_t data_start   = fat_start + (num_fats * fat_size);

    /* Helper: cluster number → first sector */
    #define CLUS_TO_SEC(c) (data_start + ((uint32_t)(c) - 2) * sec_per_clus)

    /* Search root directory for BOOTLOG.TXT */
    /* FAT32 8.3 name: "BOOTLOG TXT" (8+3, space-padded) */
    static const uint8_t target_name[11] = {
        'B','O','O','T','L','O','G',' ', 'T','X','T'
    };

    uint32_t dir_cluster = root_cluster;
    uint32_t file_cluster = 0;
    uint32_t file_size = 0;  (void)file_size;
    int found = 0;

    /* Walk directory clusters (up to 16 clusters = lots of entries) */
    for (int dc = 0; dc < 16 && !found; dc++) {
        uint32_t dir_sec = CLUS_TO_SEC(dir_cluster);

        for (uint32_t s = 0; s < (uint32_t)sec_per_clus && !found; s++) {
            if (sd_read_sector(dir_sec + s, sdlog_sector_buf) != 0)
                return -8;

            /* 16 directory entries per 512-byte sector */
            for (int e = 0; e < 16; e++) {
                uint8_t *ent = &sdlog_sector_buf[e * 32];

                if (ent[0] == 0x00) { found = -1; break; }  /* End of dir */
                if (ent[0] == 0xE5) continue;                /* Deleted */
                if (ent[11] & 0x08) continue;                /* Volume label */
                if (ent[11] & 0x10) continue;                /* Subdirectory */

                if (sdlog_memcmp(ent, target_name, 11) == 0) {
                    /* Found BOOTLOG.TXT */
                    uint16_t clus_hi = rd16(&ent[0x14]);
                    uint16_t clus_lo = rd16(&ent[0x1A]);
                    file_cluster = ((uint32_t)clus_hi << 16) | clus_lo;
                    file_size = rd32(&ent[0x1C]);
                    found = 1;
                    break;
                }
            }
        }

        if (!found || found == -1) {
            /* Follow cluster chain for directory (read FAT) */
            uint32_t fat_offset = dir_cluster * 4;
            uint32_t fat_sec = fat_start + (fat_offset / 512);
            uint32_t fat_idx = (fat_offset % 512) / 4;
            if (sd_read_sector(fat_sec, sdlog_sector_buf) != 0)
                return -9;
            uint32_t next = rd32(&sdlog_sector_buf[fat_idx * 4]);
            if (next >= 0x0FFFFFF8) break;  /* End of chain */
            dir_cluster = next;
        }
    }

    if (!found || found == -1) return -10;  /* BOOTLOG.TXT not found */
    if (file_cluster < 2) return -11;

    /* Calculate file's starting sector and how many contiguous sectors we can use.
     * The file was pre-created as 32KB, so it should be 1 cluster minimum.
     * We'll follow the FAT chain to count contiguous sectors. */
    sdlog_file_sector = CLUS_TO_SEC(file_cluster);

    /* Walk FAT chain to count contiguous clusters */
    uint32_t cur_clus = file_cluster;
    uint32_t contig_clusters = 1;
    for (int i = 0; i < 128; i++) {
        uint32_t fat_offset = cur_clus * 4;
        uint32_t fat_sec = fat_start + (fat_offset / 512);
        uint32_t fat_idx = (fat_offset % 512) / 4;
        if (sd_read_sector(fat_sec, sdlog_sector_buf) != 0) break;
        uint32_t next = rd32(&sdlog_sector_buf[fat_idx * 4]);
        if (next >= 0x0FFFFFF8) break;  /* End of chain */
        if (next == cur_clus + 1) {
            contig_clusters++;
            cur_clus = next;
        } else {
            break;  /* Not contiguous — stop here */
        }
    }

    sdlog_file_sectors = contig_clusters * sec_per_clus;
    if (sdlog_file_sectors > SDLOG_MAX_SECTORS)
        sdlog_file_sectors = SDLOG_MAX_SECTORS;

    #undef CLUS_TO_SEC

    /* Clear the log buffer */
    for (uint32_t i = 0; i < SDLOG_MAX_BYTES; i++)
        sdlog_buf[i] = 0;
    sdlog_pos = 0;

    /* Write header */
    static const char hdr[] = "=== TensorOS Boot Log ===\r\n";
    for (uint32_t i = 0; hdr[i]; i++)
        sdlog_buf[sdlog_pos++] = hdr[i];

    sdlog_ready = 1;
    return 0;

}

/* =============================================================================
 * sdlog — Append a log message (with newline)
 * =============================================================================*/
__attribute__((unused))
static void sdlog(const char *msg)
{
    if (!sdlog_ready) return;
    while (*msg && sdlog_pos < SDLOG_MAX_BYTES - 2)
        sdlog_buf[sdlog_pos++] = *msg++;
    if (sdlog_pos < SDLOG_MAX_BYTES - 2) {
        sdlog_buf[sdlog_pos++] = '\r';
        sdlog_buf[sdlog_pos++] = '\n';
    }
}

/* Log message with a decimal value: "msg: value" */
__attribute__((unused))
static void sdlog_val(const char *msg, int32_t val)
{
    if (!sdlog_ready) return;
    while (*msg && sdlog_pos < SDLOG_MAX_BYTES - 20)
        sdlog_buf[sdlog_pos++] = *msg++;
    sdlog_put_num(val);
    if (sdlog_pos < SDLOG_MAX_BYTES - 2) {
        sdlog_buf[sdlog_pos++] = '\r';
        sdlog_buf[sdlog_pos++] = '\n';
    }
}

/* Log message with a hex value: "msg: 0xVALUE" */
__attribute__((unused))
static void sdlog_hex(const char *msg, uint64_t val)
{
    if (!sdlog_ready) return;
    while (*msg && sdlog_pos < SDLOG_MAX_BYTES - 24)
        sdlog_buf[sdlog_pos++] = *msg++;
    sdlog_put_hex(val);
    if (sdlog_pos < SDLOG_MAX_BYTES - 2) {
        sdlog_buf[sdlog_pos++] = '\r';
        sdlog_buf[sdlog_pos++] = '\n';
    }
}

/* =============================================================================
 * sdlog_flush — Write log buffer to BOOTLOG.TXT sectors on SD card
 * =============================================================================*/
__attribute__((unused))
static void sdlog_flush(void)
{
    if (!sdlog_ready || sdlog_file_sector == 0) return;

    /* Save current write position so we can keep appending after flush */
    uint32_t saved_pos = sdlog_pos;

    /* Pad remainder of buffer with spaces + final newline (local only) */
    for (uint32_t p = sdlog_pos; p < SDLOG_MAX_BYTES; p++)
        sdlog_buf[p] = ' ';
    if (SDLOG_MAX_BYTES >= 4) {
        sdlog_buf[SDLOG_MAX_BYTES - 2] = '\r';
        sdlog_buf[SDLOG_MAX_BYTES - 1] = '\n';
    }

    /* Write ALL sectors (padded region included for clean file) */
    uint32_t secs = SDLOG_MAX_BYTES / 512;
    if (secs > sdlog_file_sectors)
        secs = sdlog_file_sectors;

    for (uint32_t i = 0; i < secs; i++) {
        sd_write_sector(sdlog_file_sector + i,
                        (const uint8_t *)sdlog_buf + i * 512);
    }

    /* Restore position — subsequent sdlog() calls append where we left off */
    sdlog_pos = saved_pos;
}

/* =============================================================================
 * sdlog_panic — Log a message, flush to SD, then blink LED rapidly
 * =============================================================================*/
__attribute__((unused))
static void sdlog_panic(const char *msg)
{
    sdlog(msg);
    sdlog_flush();

    /* Rapid-blink LED forever */
    while (1) {
        led_on();
        arm_timer_delay_ms(60);
        led_off();
        arm_timer_delay_ms(60);
    }
}

#endif /* TENSOROS_SDLOG_H */
