/* =============================================================================
 * TensorOS — OTA Update Protocol Implementation
 *
 * Receives a new kernel binary over the serial/BT link and either:
 *   (a) chain-loads it from RAM (fast dev iteration), or
 *   (b) writes it to the SD card FAT32 boot partition (persistent)
 *
 * The FAT32 writer does the minimal work: parse MBR → find FAT32 partition →
 * search root directory for "KERNEL8 IMG" → overwrite its clusters.
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/update/ota.h"
#include "kernel/drivers/blk/rpi_sd.h"

#if defined(__aarch64__)

/* =============================================================================
 * CRC-32 (IEEE 802.3 polynomial 0xEDB88320, reflected)
 * =============================================================================*/

static uint32_t crc32_update(uint32_t crc, const uint8_t *buf, uint32_t len)
{
    crc = ~crc;
    for (uint32_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
    }
    return ~crc;
}

/* =============================================================================
 * Serial/BT IO helpers (use UART + BT simultaneously)
 *
 * Output goes to both UART and BT.  Input comes from whichever has data first.
 * =============================================================================*/

static void ota_puts(const char *s)
{
    uart_puts(s);
    while (*s) bt_putchar(*s++);
    bt_poll();  /* flush BT TX */
}

__attribute__((unused))
static int ota_has_data(void)
{
    if (uart_has_data()) return 1;
    bt_poll();
    if (bt_has_data()) return 1;
    return 0;
}

__attribute__((unused))
static uint8_t ota_getc(void)
{
    while (1) {
        if (uart_has_data()) return (uint8_t)uart_getchar();
        bt_poll();
        if (bt_has_data()) return (uint8_t)bt_getchar();
        __asm__ volatile ("wfi");
    }
}

__attribute__((unused))
static uint8_t ota_getc_timeout(uint32_t ms)
{
    uint64_t deadline = arm_timer_count() + (arm_timer_freq() * ms / 1000);
    while (arm_timer_count() < deadline) {
        if (uart_has_data()) return (uint8_t)uart_getchar();
        bt_poll();
        if (bt_has_data()) return (uint8_t)bt_getchar();
    }
    return 0;  /* timeout — caller must handle */
}

/* Read exact number of bytes.  Returns 0 on success, -1 on timeout */
static int ota_read_exact(uint8_t *buf, uint32_t len, uint32_t timeout_ms)
{
    uint64_t deadline = arm_timer_count() +
                        (arm_timer_freq() * timeout_ms / 1000);
    for (uint32_t i = 0; i < len; i++) {
        while (1) {
            if (arm_timer_count() > deadline) return -1;
            if (uart_has_data()) { buf[i] = (uint8_t)uart_getchar(); break; }
            bt_poll();
            if (bt_has_data()) { buf[i] = (uint8_t)bt_getchar(); break; }
        }
    }
    return 0;
}

/* =============================================================================
 * Receive kernel binary
 * Stores into receive_buf (address passed in).  Returns size, or <0 on error.
 * =============================================================================*/

/* We receive into high RAM (256 MB offset) to avoid stomping on ourselves */
#define OTA_RECV_ADDR   0x10000000UL   /* 256 MB */
#define OTA_MAX_SIZE    (64 * 1024 * 1024)  /* 64 MB max kernel */

static int ota_receive_kernel(uint32_t *out_size)
{
    uint8_t *recv_buf = (uint8_t *)OTA_RECV_ADDR;

    ota_puts("RDY\n");
    kprintf("[OTA] Waiting for kernel binary...\n");
    kprintf("[OTA] Protocol: 'OTA!' + uint32 size + data + uint32 crc32\n");

    /* Wait for magic: "OTA!" */
    uint8_t magic[4];
    if (ota_read_exact(magic, 4, 60000) != 0) {  /* 60s timeout */
        ota_puts("ERR:timeout waiting for magic\n");
        return -1;
    }
    if (magic[0] != 'O' || magic[1] != 'T' || magic[2] != 'A' || magic[3] != '!') {
        ota_puts("ERR:bad magic\n");
        return -2;
    }

    /* Read size (4 bytes, little-endian) */
    uint8_t szb[4];
    if (ota_read_exact(szb, 4, 5000) != 0) {
        ota_puts("ERR:timeout reading size\n");
        return -3;
    }
    uint32_t size = szb[0] | ((uint32_t)szb[1] << 8) |
                    ((uint32_t)szb[2] << 16) | ((uint32_t)szb[3] << 24);

    if (size == 0 || size > OTA_MAX_SIZE) {
        ota_puts("ERR:invalid size\n");
        return -4;
    }

    kprintf("[OTA] Receiving %u bytes", size);

    /* Read data with progress */
    uint32_t received = 0;
    uint32_t last_pct = 0;
    uint64_t timeout_per_block = 30000;  /* 30s timeout per 4KB block */

    while (received < size) {
        uint32_t chunk = size - received;
        if (chunk > 4096) chunk = 4096;

        if (ota_read_exact(recv_buf + received, chunk, timeout_per_block) != 0) {
            kprintf("\n");
            ota_puts("ERR:timeout during transfer\n");
            return -5;
        }
        received += chunk;

        /* Progress dots */
        uint32_t pct = (received * 100) / size;
        if (pct / 10 > last_pct / 10) {
            kprintf(".");
            last_pct = pct;
        }
    }
    kprintf(" done\n");

    /* Read CRC32 (4 bytes) */
    uint8_t crcb[4];
    if (ota_read_exact(crcb, 4, 5000) != 0) {
        ota_puts("ERR:timeout reading crc\n");
        return -6;
    }
    uint32_t expected_crc = crcb[0] | ((uint32_t)crcb[1] << 8) |
                            ((uint32_t)crcb[2] << 16) | ((uint32_t)crcb[3] << 24);

    /* Verify CRC */
    uint32_t actual_crc = crc32_update(0, recv_buf, size);
    if (actual_crc != expected_crc) {
        kprintf("[OTA] CRC mismatch: expected 0x%x got 0x%x\n",
                expected_crc, actual_crc);
        ota_puts("ERR:crc mismatch\n");
        return -7;
    }

    kprintf("[OTA] CRC OK (0x%x), %u bytes verified\n", actual_crc, size);
    ota_puts("OK!\n");

    *out_size = size;
    return 0;
}

/* =============================================================================
 * Chain-load: copy received kernel to 0x80000 and jump
 * =============================================================================*/

/* This function attribute prevents inlining — we need to absolutely control
 * the jump sequence.  The function copies the kernel then branches to it. */
static void __attribute__((noinline, noreturn))
ota_chainload(const uint8_t *src, uint32_t size)
{
    /* Disable interrupts */
    __asm__ volatile ("msr daifset, #0xF" ::: "memory");

    /* Copy kernel to 0x80000 */
    uint8_t *dst = (uint8_t *)0x80000;
    for (uint32_t i = 0; i < size; i++)
        dst[i] = src[i];

    /* DSB + ISB to ensure copied data is visible */
    __asm__ volatile ("dsb sy; isb" ::: "memory");

    /* Invalidate instruction cache (we're overwriting code) */
    __asm__ volatile ("ic iallu; dsb sy; isb" ::: "memory");

    /* Jump to new kernel at 0x80000 */
    __asm__ volatile ("br %0" :: "r"(0x80000UL));
    __builtin_unreachable();
}

int ota_receive_and_chainload(void)
{
    uint32_t size;
    int r = ota_receive_kernel(&size);
    if (r != 0) return r;

    kprintf("[OTA] Chain-loading %u bytes to 0x80000...\n", size);
    ota_puts("BOOT\n");

    /* Small delay to let the "BOOT" message flush over BT */
    arm_timer_delay_ms(100);
    bt_poll();

    ota_chainload((const uint8_t *)OTA_RECV_ADDR, size);
    /* Does not return */
}

/* =============================================================================
 * FAT32 Minimal: Find kernel8.img on boot partition and overwrite it
 * =============================================================================*/

/* MBR partition entry */
typedef struct {
    uint8_t  status;
    uint8_t  chs_first[3];
    uint8_t  type;
    uint8_t  chs_last[3];
    uint32_t lba_start;
    uint32_t sectors;
} __attribute__((packed)) mbr_part_t;

/* FAT32 BPB (BIOS Parameter Block) */
typedef struct {
    uint8_t  jmp[3];
    char     oem[8];
    uint16_t bytes_per_sector;
    uint8_t  sectors_per_cluster;
    uint16_t reserved_sectors;
    uint8_t  num_fats;
    uint16_t root_entry_count;  /* 0 for FAT32 */
    uint16_t total_sectors_16;
    uint8_t  media_type;
    uint16_t fat_size_16;       /* 0 for FAT32 */
    uint16_t sectors_per_track;
    uint16_t num_heads;
    uint32_t hidden_sectors;
    uint32_t total_sectors_32;
    /* FAT32 specific */
    uint32_t fat_size_32;
    uint16_t ext_flags;
    uint16_t fs_version;
    uint32_t root_cluster;
} __attribute__((packed)) fat32_bpb_t;

/* FAT32 directory entry (32 bytes) */
typedef struct {
    char     name[11];          /* 8.3 format, padded with spaces */
    uint8_t  attr;
    uint8_t  nt_reserved;
    uint8_t  create_time_10th;
    uint16_t create_time;
    uint16_t create_date;
    uint16_t access_date;
    uint16_t cluster_hi;
    uint16_t mod_time;
    uint16_t mod_date;
    uint16_t cluster_lo;
    uint32_t file_size;
} __attribute__((packed)) fat32_dirent_t;

/* Compare 11 bytes (8.3 filename) */
static int fat32_namecmp(const char *a, const char *b)
{
    for (int i = 0; i < 11; i++)
        if (a[i] != b[i]) return 1;
    return 0;
}

/* Find and overwrite kernel8.img on the first FAT32 partition.
 * Returns 0 on success. */
static int ota_flash_to_sd(const uint8_t *data, uint32_t size)
{
    uint8_t sector[512];

    /* Step 1: Read MBR (sector 0) */
    if (sd_read_sector(0, sector) != 0) {
        kprintf("[OTA] Failed to read MBR\n");
        return -1;
    }

    /* Check MBR signature */
    if (sector[510] != 0x55 || sector[511] != 0xAA) {
        kprintf("[OTA] Invalid MBR signature\n");
        return -2;
    }

    /* Find first FAT32 partition (type 0x0B or 0x0C) */
    mbr_part_t *parts = (mbr_part_t *)(sector + 446);
    uint32_t part_lba = 0;
    for (int i = 0; i < 4; i++) {
        if (parts[i].type == 0x0B || parts[i].type == 0x0C) {
            part_lba = parts[i].lba_start;
            break;
        }
    }
    if (part_lba == 0) {
        kprintf("[OTA] No FAT32 partition found\n");
        return -3;
    }

    /* Step 2: Read FAT32 BPB (first sector of partition) */
    if (sd_read_sector(part_lba, sector) != 0) {
        kprintf("[OTA] Failed to read BPB\n");
        return -4;
    }

    fat32_bpb_t *bpb = (fat32_bpb_t *)sector;
    uint32_t spc = bpb->sectors_per_cluster;
    uint32_t fat_start = part_lba + bpb->reserved_sectors;
    uint32_t fat_size = bpb->fat_size_32;
    uint32_t data_start = fat_start + bpb->num_fats * fat_size;
    uint32_t root_cluster = bpb->root_cluster;

    kprintf("[OTA] FAT32: spc=%u fat_start=%u data_start=%u root_clust=%u\n",
            spc, fat_start, data_start, root_cluster);

    /* Step 3: Search root directory for "KERNEL8 IMG" */
    /* Follow the cluster chain of the root directory */
    uint32_t cluster = root_cluster;
    uint32_t found_cluster_lo = 0, found_cluster_hi = 0;
    uint32_t found_size = 0;
    uint32_t dir_entry_sector = 0;
    uint32_t dir_entry_offset = 0;
    int found = 0;

    /* FAT32 8.3 name for "kernel8.img" = "KERNEL8 IMG" (8+3, space-padded) */
    const char target_name[11] = {'K','E','R','N','E','L','8',' ','I','M','G'};

    for (int chain = 0; chain < 64 && !found; chain++) {  /* max 64 clusters */
        uint32_t cluster_lba = data_start + (cluster - 2) * spc;

        for (uint32_t s = 0; s < spc && !found; s++) {
            if (sd_read_sector(cluster_lba + s, sector) != 0) break;

            fat32_dirent_t *entries = (fat32_dirent_t *)sector;
            for (int e = 0; e < 16; e++) {  /* 512/32 = 16 entries per sector */
                if (entries[e].name[0] == 0x00) goto dir_end;  /* End of directory */
                if ((uint8_t)entries[e].name[0] == 0xE5) continue;  /* Deleted */
                if (entries[e].attr & 0x08) continue;  /* Volume label */
                if (entries[e].attr & 0x0F) continue;  /* LFN entry, skip */

                if (fat32_namecmp(entries[e].name, target_name) == 0) {
                    found_cluster_lo = entries[e].cluster_lo;
                    found_cluster_hi = entries[e].cluster_hi;
                    found_size = entries[e].file_size;
                    dir_entry_sector = cluster_lba + s;
                    dir_entry_offset = e;
                    found = 1;
                    break;
                }
            }
        }

        /* Follow FAT chain */
        uint32_t fat_sector_idx = cluster / 128;  /* 512/4 = 128 entries per sector */
        if (sd_read_sector(fat_start + fat_sector_idx, sector) != 0) break;
        uint32_t *fat = (uint32_t *)sector;
        uint32_t next = fat[cluster % 128] & 0x0FFFFFFF;
        if (next >= 0x0FFFFFF8) break;  /* End of chain */
        cluster = next;
    }
dir_end:

    if (!found) {
        kprintf("[OTA] kernel8.img not found on SD card\n");
        return -5;
    }

    uint32_t file_cluster = found_cluster_lo | ((uint32_t)found_cluster_hi << 16);
    kprintf("[OTA] Found kernel8.img: cluster=%u size=%u\n", file_cluster, found_size);

    /* Step 4: Write new kernel data over the file's clusters.
     * Follow the FAT chain, writing sector by sector.
     * If new kernel > old size, we may need to allocate new clusters.
     * For safety, require new kernel <= old allocated space. */
    uint32_t old_clusters = (found_size + spc * 512 - 1) / (spc * 512);
    uint32_t new_clusters = (size + spc * 512 - 1) / (spc * 512);

    if (new_clusters > old_clusters + 8) {
        /* We could allocate more clusters from the FAT, but for safety just
         * reject if it's way bigger.  +8 clusters (~32KB) slack is fine. */
        kprintf("[OTA] New kernel too large (%u > %u clusters)\n",
                new_clusters, old_clusters);
        return -6;
    }

    /* Write data following the cluster chain */
    cluster = file_cluster;
    uint32_t written = 0;
    uint32_t cluster_bytes = spc * 512;  (void)cluster_bytes;

    while (written < size) {
        uint32_t cluster_lba = data_start + (cluster - 2) * spc;

        for (uint32_t s = 0; s < spc && written < size; s++) {
            /* Prepare a 512-byte sector (zero-padded at the end) */
            uint8_t wbuf[512];
            uint32_t remain = size - written;
            uint32_t copy = remain > 512 ? 512 : remain;
            for (uint32_t i = 0; i < copy; i++)
                wbuf[i] = data[written + i];
            for (uint32_t i = copy; i < 512; i++)
                wbuf[i] = 0;

            if (sd_write_sector(cluster_lba + s, wbuf) != 0) {
                kprintf("[OTA] SD write error at LBA %u\n", cluster_lba + s);
                return -7;
            }
            written += 512;
        }

        /* Progress */
        kprintf("[OTA] Written %u / %u bytes\r", written > size ? size : written, size);

        /* Follow FAT chain to next cluster */
        uint32_t fat_sector_idx = cluster / 128;
        if (sd_read_sector(fat_start + fat_sector_idx, sector) != 0) {
            kprintf("\n[OTA] FAT read error\n");
            return -8;
        }
        uint32_t *fat = (uint32_t *)sector;
        uint32_t next = fat[cluster % 128] & 0x0FFFFFFF;
        if (next >= 0x0FFFFFF8) {
            if (written < size) {
                /* Need to allocate a new cluster — find a free one in this FAT sector */
                int alloc_found = 0;
                for (int i = 0; i < 128; i++) {
                    if ((fat[i] & 0x0FFFFFFF) == 0) {
                        /* Free cluster! Link it */
                        fat[cluster % 128] = (fat_sector_idx * 128 + i) | 0x00000000;
                        fat[i] = 0x0FFFFFF8;  /* End of chain */
                        sd_write_sector(fat_start + fat_sector_idx, sector);
                        next = fat_sector_idx * 128 + i;
                        alloc_found = 1;
                        break;
                    }
                }
                if (!alloc_found) {
                    kprintf("\n[OTA] No free clusters\n");
                    return -9;
                }
            } else {
                break;  /* All written */
            }
        }
        cluster = next;
    }
    kprintf("\n");

    /* Step 5: Update directory entry with new file size */
    if (sd_read_sector(dir_entry_sector, sector) != 0) {
        kprintf("[OTA] Failed to re-read directory sector\n");
        return -10;
    }
    fat32_dirent_t *entries = (fat32_dirent_t *)sector;
    entries[dir_entry_offset].file_size = size;
    if (sd_write_sector(dir_entry_sector, sector) != 0) {
        kprintf("[OTA] Failed to update directory entry\n");
        return -11;
    }

    kprintf("[OTA] kernel8.img updated: %u -> %u bytes\n", found_size, size);
    return 0;
}

int ota_receive_and_flash(void)
{
    uint32_t size;
    int r = ota_receive_kernel(&size);
    if (r != 0) return r;

    kprintf("[OTA] Initializing SD card...\n");
    if (sd_init() != 0) {
        kprintf("[OTA] SD card init failed — falling back to chain-load\n");
        ota_puts("WARN:sd_fail,chainloading\n");
        ota_chainload((const uint8_t *)OTA_RECV_ADDR, size);
    }

    kprintf("[OTA] Writing kernel8.img to SD card...\n");
    r = ota_flash_to_sd((const uint8_t *)OTA_RECV_ADDR, size);
    if (r != 0) {
        kprintf("[OTA] SD flash failed (%d) — falling back to chain-load\n", r);
        ota_puts("WARN:flash_fail,chainloading\n");
        ota_chainload((const uint8_t *)OTA_RECV_ADDR, size);
    }

    kprintf("[OTA] Flash complete! Rebooting...\n");
    ota_puts("BOOT\n");
    arm_timer_delay_ms(200);
    bt_poll();

    /* Reboot via watchdog (PM_RSTC) */
    #define PM_BASE     0xFE100000UL
    #define PM_RSTC     (PM_BASE + 0x1C)
    #define PM_WDOG     (PM_BASE + 0x24)
    #define PM_PASSWORD 0x5A000000
    mmio_write(PM_WDOG, PM_PASSWORD | 1);              /* Watchdog = 1 tick */
    mmio_write(PM_RSTC, PM_PASSWORD | 0x20);            /* Full reset */
    while (1) __asm__ volatile ("wfi");  /* Wait for reset */
}

#else /* x86 stubs */

int ota_receive_and_chainload(void) { kprintf("[OTA] Not available on x86\n"); return -1; }
int ota_receive_and_flash(void)     { kprintf("[OTA] Not available on x86\n"); return -1; }

#endif
