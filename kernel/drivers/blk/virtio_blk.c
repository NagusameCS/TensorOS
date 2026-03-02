/* =============================================================================
 * TensorOS - Virtio Block Device Driver Implementation
 * Legacy virtio (PCI BAR0 I/O) for QEMU disk I/O
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/drivers/blk/virtio_blk.h"

/* Reuse virtio PCI constants from net driver */
#define VIRTIO_PCI_VENDOR      0x1AF4
#define VIRTIO_BLK_DEVICE_LO   0x1000
#define VIRTIO_BLK_DEVICE_HI   0x103F

/* Legacy virtio PCI register offsets */
#define VIRTIO_REG_DEVICE_FEATURES  0x00
#define VIRTIO_REG_GUEST_FEATURES   0x04
#define VIRTIO_REG_QUEUE_ADDR       0x08
#define VIRTIO_REG_QUEUE_SIZE       0x0C
#define VIRTIO_REG_QUEUE_SELECT     0x0E
#define VIRTIO_REG_QUEUE_NOTIFY     0x10
#define VIRTIO_REG_DEVICE_STATUS    0x12
#define VIRTIO_REG_ISR_STATUS       0x13

/* Device-specific config starts at 0x14 for legacy virtio-blk */
#define VIRTIO_BLK_CFG_OFFSET       0x14

#define VIRTIO_STATUS_ACK       1
#define VIRTIO_STATUS_DRIVER    2
#define VIRTIO_STATUS_DRIVER_OK 4

/* Virtqueue desc flags */
#define VRING_DESC_F_NEXT   1
#define VRING_DESC_F_WRITE  2

/* Virtqueue structures */
struct vring_desc_blk {
    uint64_t addr;
    uint32_t len;
    uint16_t flags;
    uint16_t next;
} __attribute__((packed));

struct vring_avail_blk {
    uint16_t flags;
    uint16_t idx;
    uint16_t ring[256];
} __attribute__((packed));

struct vring_used_elem_blk {
    uint32_t id;
    uint32_t len;
} __attribute__((packed));

struct vring_used_blk {
    uint16_t flags;
    uint16_t idx;
    struct vring_used_elem_blk ring[256];
} __attribute__((packed));

/* Global state */
static virtio_blk_dev_t blk_dev;

/* Virtqueue memory (page-aligned) */
static uint8_t blk_vq_mem[32768] __attribute__((aligned(4096)));

/* DMA buffers for requests */
static struct virtio_blk_req blk_req_hdr __attribute__((aligned(16)));
static uint8_t               blk_status  __attribute__((aligned(16)));
static uint8_t               blk_dma_buf[512 * 128] __attribute__((aligned(4096))); /* 64KB */

/* Virtqueue state */
static struct vring_desc_blk  *blk_desc;
static struct vring_avail_blk *blk_avail;
static struct vring_used_blk  *blk_used;
static uint16_t                blk_last_used;

/* =============================================================================
 * PCI helpers
 * =============================================================================*/

static uint32_t blk_pci_read32(uint8_t bus, uint8_t slot, uint8_t func, uint8_t offset)
{
    uint32_t addr = (1u << 31) | ((uint32_t)bus << 16) | ((uint32_t)slot << 11) |
                    ((uint32_t)func << 8) | (offset & 0xFC);
    outl(0xCF8, addr);
    return inl(0xCFC);
}

static void blk_pci_write32(uint8_t bus, uint8_t slot, uint8_t func, uint8_t offset, uint32_t val)
{
    uint32_t addr = (1u << 31) | ((uint32_t)bus << 16) | ((uint32_t)slot << 11) |
                    ((uint32_t)func << 8) | (offset & 0xFC);
    outl(0xCF8, addr);
    outl(0xCFC, val);
}

/* =============================================================================
 * Find virtio-blk on PCI
 * =============================================================================*/

static int find_virtio_blk(uint8_t *out_bus, uint8_t *out_slot)
{
    for (uint32_t bus = 0; bus < 8; bus++) {
        for (uint32_t slot = 0; slot < 32; slot++) {
            uint32_t id = blk_pci_read32(bus, slot, 0, 0);
            uint16_t vendor = id & 0xFFFF;
            uint16_t device = (id >> 16) & 0xFFFF;

            if (vendor != VIRTIO_PCI_VENDOR) continue;

            if (device >= VIRTIO_BLK_DEVICE_LO && device <= VIRTIO_BLK_DEVICE_HI) {
                uint32_t subsys = blk_pci_read32(bus, slot, 0, 0x2C);
                uint16_t subsys_id = (subsys >> 16) & 0xFFFF;
                if (subsys_id == VIRTIO_BLK_SUBSYS) {
                    *out_bus = bus;
                    *out_slot = slot;
                    return 0;
                }
            }
            /* Modern virtio-blk: device 0x1042 */
            if (device == 0x1042) {
                *out_bus = bus;
                *out_slot = slot;
                return 0;
            }
        }
    }
    return -1;
}

/* =============================================================================
 * Initialize driver
 * =============================================================================*/

int virtio_blk_init(void)
{
    kmemset(&blk_dev, 0, sizeof(blk_dev));

    uint8_t bus, slot;
    if (find_virtio_blk(&bus, &slot) != 0) {
        kprintf("[VBLK] No virtio-blk device found\n");
        return -1;
    }

    blk_dev.pci_bus = bus;
    blk_dev.pci_slot = slot;
    kprintf("[VBLK] Found virtio-blk at PCI %u:%u.0\n", bus, slot);

    /* Enable PCI bus mastering + I/O space */
    uint32_t cmd = blk_pci_read32(bus, slot, 0, 0x04);
    cmd |= 0x05;
    blk_pci_write32(bus, slot, 0, 0x04, cmd);

    /* Read BAR0 */
    uint32_t bar0 = blk_pci_read32(bus, slot, 0, 0x10);
    if (!(bar0 & 1)) {
        kprintf("[VBLK] BAR0 is not I/O space\n");
        return -2;
    }
    blk_dev.io_base = (uint16_t)(bar0 & 0xFFFC);
    kprintf("[VBLK] I/O base: 0x%x\n", blk_dev.io_base);

    /* Read IRQ */
    uint32_t irq_reg = blk_pci_read32(bus, slot, 0, 0x3C);
    blk_dev.irq = irq_reg & 0xFF;

    uint16_t io = blk_dev.io_base;

    /* Reset */
    outb(io + VIRTIO_REG_DEVICE_STATUS, 0);
    outb(io + VIRTIO_REG_DEVICE_STATUS, VIRTIO_STATUS_ACK);
    outb(io + VIRTIO_REG_DEVICE_STATUS, VIRTIO_STATUS_ACK | VIRTIO_STATUS_DRIVER);

    /* Accept all features (we don't need any specific ones) */
    uint32_t features = inl(io + VIRTIO_REG_DEVICE_FEATURES);
    (void)features;
    outl(io + VIRTIO_REG_GUEST_FEATURES, 0);

    /* Read device config: capacity */
    uint32_t cap_lo = inl(io + VIRTIO_BLK_CFG_OFFSET);
    uint32_t cap_hi = inl(io + VIRTIO_BLK_CFG_OFFSET + 4);
    blk_dev.capacity_sectors = ((uint64_t)cap_hi << 32) | cap_lo;
    blk_dev.blk_size = 512;

    kprintf("[VBLK] Capacity: %lu sectors (%lu MB)\n",
            blk_dev.capacity_sectors,
            (blk_dev.capacity_sectors * 512) / (1024 * 1024));

    /* Setup virtqueue 0 */
    outw(io + VIRTIO_REG_QUEUE_SELECT, 0);
    uint16_t qsize = inw(io + VIRTIO_REG_QUEUE_SIZE);
    if (qsize == 0 || qsize > 256) qsize = 256;
    blk_dev.vq_size = qsize;
    blk_dev.vq_mem = blk_vq_mem;

    kmemset(blk_vq_mem, 0, sizeof(blk_vq_mem));
    blk_desc = (struct vring_desc_blk *)blk_vq_mem;
    blk_avail = (struct vring_avail_blk *)(blk_vq_mem + qsize * sizeof(struct vring_desc_blk));

    uint64_t avail_end = (uint64_t)(uintptr_t)blk_avail + sizeof(uint16_t) * (3 + qsize);
    uint64_t used_off = (avail_end + 4095) & ~4095ULL;
    blk_used = (struct vring_used_blk *)(uintptr_t)used_off;
    blk_last_used = 0;

    /* Tell device the queue address (page frame number) */
    uint64_t vq_phys = (uint64_t)(uintptr_t)blk_vq_mem;
    outl(io + VIRTIO_REG_QUEUE_ADDR, (uint32_t)(vq_phys / 4096));

    /* Mark driver OK */
    outb(io + VIRTIO_REG_DEVICE_STATUS,
         VIRTIO_STATUS_ACK | VIRTIO_STATUS_DRIVER | VIRTIO_STATUS_DRIVER_OK);

    blk_dev.initialized = 1;
    kprintf("[VBLK] Virtio-blk initialized successfully\n");
    return 0;
}

/* =============================================================================
 * Submit a 3-descriptor request: header -> data -> status
 * =============================================================================*/

static int blk_do_request(uint32_t type, uint64_t sector, void *buf, uint32_t nbytes)
{
    if (!blk_dev.initialized) return -1;

    /* Setup request header */
    blk_req_hdr.type = type;
    blk_req_hdr.reserved = 0;
    blk_req_hdr.sector = sector;
    blk_status = 0xFF;

    /* Descriptor 0: request header (device reads) */
    blk_desc[0].addr = (uint64_t)(uintptr_t)&blk_req_hdr;
    blk_desc[0].len = sizeof(struct virtio_blk_req);
    blk_desc[0].flags = VRING_DESC_F_NEXT;
    blk_desc[0].next = 1;

    /* Descriptor 1: data buffer */
    blk_desc[1].addr = (uint64_t)(uintptr_t)buf;
    blk_desc[1].len = nbytes;
    blk_desc[1].flags = VRING_DESC_F_NEXT;
    if (type == VIRTIO_BLK_T_IN) {
        blk_desc[1].flags |= VRING_DESC_F_WRITE; /* Device writes to this buffer */
    }
    blk_desc[1].next = 2;

    /* Descriptor 2: status byte (device writes) */
    blk_desc[2].addr = (uint64_t)(uintptr_t)&blk_status;
    blk_desc[2].len = 1;
    blk_desc[2].flags = VRING_DESC_F_WRITE;
    blk_desc[2].next = 0;

    /* Submit to available ring */
    uint16_t avail_idx = blk_avail->idx;
    blk_avail->ring[avail_idx % blk_dev.vq_size] = 0; /* Descriptor chain starts at 0 */
#if defined(__aarch64__)
    __asm__ volatile ("dmb sy" ::: "memory");
#else
    __asm__ volatile ("mfence" ::: "memory");
#endif
    blk_avail->idx = avail_idx + 1;

    /* Notify device */
    outw(blk_dev.io_base + VIRTIO_REG_QUEUE_NOTIFY, 0);

    /* Poll for completion */
    uint64_t timeout = 10000000; /* ~10M iterations */
    while (blk_last_used == blk_used->idx && timeout > 0) {
#if defined(__aarch64__)
        __asm__ volatile ("yield");
#else
        __asm__ volatile ("pause");
#endif
        timeout--;
    }

    if (timeout == 0) {
        kprintf("[VBLK] Request timeout!\n");
        return -2;
    }

    blk_last_used = blk_used->idx;

    if (blk_status != VIRTIO_BLK_S_OK) {
        kprintf("[VBLK] Request failed: status=%u\n", blk_status);
        return -3;
    }

    return 0;
}

/* =============================================================================
 * Public API
 * =============================================================================*/

int virtio_blk_read(uint64_t sector, uint32_t count, void *buf)
{
    if (!blk_dev.initialized) return -1;
    if (count == 0) return 0;

    /* Read in chunks that fit our DMA buffer */
    uint32_t max_sectors = sizeof(blk_dma_buf) / 512;
    uint8_t *dst = (uint8_t *)buf;

    while (count > 0) {
        uint32_t chunk = count < max_sectors ? count : max_sectors;
        uint32_t bytes = chunk * 512;

        int rc = blk_do_request(VIRTIO_BLK_T_IN, sector, blk_dma_buf, bytes);
        if (rc != 0) return rc;

        kmemcpy(dst, blk_dma_buf, bytes);
        dst += bytes;
        sector += chunk;
        count -= chunk;
        blk_dev.reads++;
        blk_dev.bytes_read += bytes;
    }

    return 0;
}

int virtio_blk_write(uint64_t sector, uint32_t count, const void *buf)
{
    if (!blk_dev.initialized) return -1;
    if (count == 0) return 0;

    uint32_t max_sectors = sizeof(blk_dma_buf) / 512;
    const uint8_t *src = (const uint8_t *)buf;

    while (count > 0) {
        uint32_t chunk = count < max_sectors ? count : max_sectors;
        uint32_t bytes = chunk * 512;

        kmemcpy(blk_dma_buf, src, bytes);
        int rc = blk_do_request(VIRTIO_BLK_T_OUT, sector, blk_dma_buf, bytes);
        if (rc != 0) return rc;

        src += bytes;
        sector += chunk;
        count -= chunk;
        blk_dev.writes++;
        blk_dev.bytes_written += bytes;
    }

    return 0;
}

uint64_t virtio_blk_capacity(void)
{
    return blk_dev.capacity_sectors * 512;
}

void virtio_blk_print_info(void)
{
    if (!blk_dev.initialized) {
        kprintf("[VBLK] Not initialized\n");
        return;
    }
    kprintf("[VBLK] PCI %u:%u.0, I/O 0x%x, IRQ %u\n",
            blk_dev.pci_bus, blk_dev.pci_slot, blk_dev.io_base, blk_dev.irq);
    kprintf("[VBLK] Capacity: %lu MB (%lu sectors)\n",
            (blk_dev.capacity_sectors * 512) / (1024 * 1024),
            blk_dev.capacity_sectors);
    kprintf("[VBLK] Stats: %lu reads (%lu KB), %lu writes (%lu KB)\n",
            blk_dev.reads, blk_dev.bytes_read / 1024,
            blk_dev.writes, blk_dev.bytes_written / 1024);
}
