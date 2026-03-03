/* =============================================================================
 * TensorOS - Tensor-Aware Memory Manager Implementation
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────┐
 * │                    Unified Virtual Address Space             │
 * ├────────────┬──────────────┬──────────────┬─────────────────┤
 * │  Kernel    │  Tensor Heap │ Model Cache  │   GPU Mapped    │
 * │  (4K pgs)  │  (2MB pgs)   │  (2MB pgs)   │   (MMIO)        │
 * └────────────┴──────────────┴──────────────┴─────────────────┘
 *
 * Key design decisions:
 * 1. Tensor heap uses 2MB pages by default to reduce TLB misses during
 *    large matrix operations.
 * 2. Model weights are cached with an LRU policy, avoiding re-loading
 *    from disk for frequently used models.
 * 3. DMA allocations are aligned and pinned for zero-copy GPU transfers.
 * 4. A slab allocator handles small kernel allocations efficiently.
 * =============================================================================*/

#include "kernel/mm/tensor_mm.h"

/* =============================================================================
 * Physical Memory Bitmap
 * Simple bitmap allocator for physical pages
 * =============================================================================*/

#define PHYS_MEM_MAX_GB     64
#define PHYS_PAGES_4K       (PHYS_MEM_MAX_GB * 1024 * 256) /* 4K pages */
#define BITMAP_SIZE         (PHYS_PAGES_4K / 8)

/* Bitmap for 4GB / 4KB = 1M pages → 128KB bitmap.
 * We allocate enough for 4GB (reasonable for QEMU). */
#define PHYS_BITMAP_4GB  (1024 * 1024 / 8)   /* 128 KB */
static uint8_t phys_bitmap[PHYS_BITMAP_4GB];
static uint64_t phys_mem_total = 0;
static uint64_t phys_mem_free = 0;

/* Track page allocations for kfree: (ptr → page_count) */
#define KFREE_TRACK_MAX 512
static struct { uint64_t addr; uint32_t pages; } kfree_track[KFREE_TRACK_MAX];
static int kfree_track_count = 0;

static void phys_page_mark_used(uint64_t page_index)
{
    phys_bitmap[page_index / 8] |= (1 << (page_index % 8));
    phys_mem_free -= MM_PAGE_SIZE_4K;
}

static void phys_page_mark_free(uint64_t page_index)
{
    phys_bitmap[page_index / 8] &= ~(1 << (page_index % 8));
    phys_mem_free += MM_PAGE_SIZE_4K;
}

static bool phys_page_is_free(uint64_t page_index)
{
    return !(phys_bitmap[page_index / 8] & (1 << (page_index % 8)));
}

static uint64_t phys_alloc_pages(uint64_t count, uint64_t alignment_pages)
{
    uint64_t consecutive = 0;
    uint64_t start = 0;

    for (uint64_t i = 0; i < PHYS_PAGES_4K; i++) {
        if (phys_page_is_free(i)) {
            if (consecutive == 0) {
                /* Check alignment */
                if ((i % alignment_pages) != 0) continue;
                start = i;
            }
            consecutive++;
            if (consecutive == count) {
                /* Mark all pages as used */
                for (uint64_t j = start; j < start + count; j++)
                    phys_page_mark_used(j);
                return start * MM_PAGE_SIZE_4K;
            }
        } else {
            consecutive = 0;
        }
    }
    return 0; /* Out of memory */
}

static void phys_free_pages(uint64_t phys_addr, uint64_t count)
{
    uint64_t start = phys_addr / MM_PAGE_SIZE_4K;
    for (uint64_t i = start; i < start + count; i++)
        phys_page_mark_free(i);
}

/* =============================================================================
 * Tensor Heap - Bump allocator with free list
 * Optimized for large, aligned tensor allocations
 * =============================================================================*/

typedef struct heap_block {
    uint64_t            size;
    bool                free;
    struct heap_block  *next;
    struct heap_block  *prev;
} heap_block_t;

static uint8_t *tensor_heap_base;
static uint64_t tensor_heap_size;
static uint64_t tensor_heap_used;
static heap_block_t *tensor_heap_first;

static void tensor_heap_init(void)
{
    tensor_heap_base = (uint8_t *)__tensor_heap_start;
    tensor_heap_size = __tensor_heap_end - __tensor_heap_start;
    tensor_heap_used = 0;

    /* Initialize with single free block */
    tensor_heap_first = (heap_block_t *)tensor_heap_base;
    tensor_heap_first->size = tensor_heap_size - sizeof(heap_block_t);
    tensor_heap_first->free = true;
    tensor_heap_first->next = NULL;
    tensor_heap_first->prev = NULL;
}

static void *tensor_heap_alloc(uint64_t size, uint32_t alignment)
{
    /* Round up size to alignment */
    size = (size + alignment - 1) & ~(alignment - 1);

    /* First-fit search */
    heap_block_t *block = tensor_heap_first;
    while (block) {
        if (block->free && block->size >= size) {
            /* Calculate aligned address */
            uint64_t addr = (uint64_t)(block + 1);
            uint64_t aligned = (addr + alignment - 1) & ~(alignment - 1);
            uint64_t padding = aligned - addr;
            uint64_t total_needed = size + padding;

            if (block->size >= total_needed) {
                /* Split block if enough space remaining */
                if (block->size > total_needed + sizeof(heap_block_t) + 64) {
                    heap_block_t *new_block = (heap_block_t *)
                        ((uint8_t *)block + sizeof(heap_block_t) + total_needed);
                    new_block->size = block->size - total_needed - sizeof(heap_block_t);
                    new_block->free = true;
                    new_block->next = block->next;
                    new_block->prev = block;
                    if (block->next)
                        block->next->prev = new_block;
                    block->next = new_block;
                    block->size = total_needed;
                }

                block->free = false;
                tensor_heap_used += block->size;
                return (void *)aligned;
            }
        }
        block = block->next;
    }

    return NULL; /* Out of tensor heap memory */
}

static void tensor_heap_free_block(void *ptr)
{
    if (!ptr) return;

    /* Find the block header */
    heap_block_t *block = tensor_heap_first;
    while (block) {
        uint64_t block_start = (uint64_t)(block + 1);
        uint64_t block_end = block_start + block->size;
        if ((uint64_t)ptr >= block_start && (uint64_t)ptr < block_end) {
            block->free = true;
            tensor_heap_used -= block->size;

            /* Coalesce with next block */
            if (block->next && block->next->free) {
                block->size += sizeof(heap_block_t) + block->next->size;
                block->next = block->next->next;
                if (block->next)
                    block->next->prev = block;
            }

            /* Coalesce with previous block */
            if (block->prev && block->prev->free) {
                block->prev->size += sizeof(heap_block_t) + block->size;
                block->prev->next = block->next;
                if (block->next)
                    block->next->prev = block->prev;
            }
            return;
        }
        block = block->next;
    }
}

/* =============================================================================
 * Model Weight Cache
 * LRU cache for model weights to avoid re-loading from storage
 * =============================================================================*/

static model_cache_t model_cache;

static void model_cache_init(void)
{
    kmemset(&model_cache, 0, sizeof(model_cache));
    model_cache.max_size = __model_cache_end - __model_cache_start;
}

void *model_cache_get(uint64_t model_hash, uint64_t *size)
{
    for (uint32_t i = 0; i < model_cache.count; i++) {
        if (model_cache.entries[i].model_hash == model_hash) {
            model_cache.entries[i].last_access = kstate.uptime_ticks;
            model_cache.entries[i].ref_count++;
            model_cache.hits++;
            if (size) *size = model_cache.entries[i].size;
            return model_cache.entries[i].data;
        }
    }
    model_cache.misses++;
    return NULL;
}

int model_cache_put(uint64_t model_hash, void *data, uint64_t size)
{
    /* Check if already cached */
    for (uint32_t i = 0; i < model_cache.count; i++) {
        if (model_cache.entries[i].model_hash == model_hash)
            return 0; /* Already cached */
    }

    /* Evict if necessary */
    while (model_cache.total_size + size > model_cache.max_size) {
        model_cache_evict_lru();
    }

    if (model_cache.count >= MODEL_CACHE_MAX_ENTRIES)
        model_cache_evict_lru();

    /* Add entry */
    model_cache_entry_t *entry = &model_cache.entries[model_cache.count++];
    entry->model_hash = model_hash;
    entry->data = data;
    entry->size = size;
    entry->last_access = kstate.uptime_ticks;
    entry->ref_count = 1;
    entry->pin_count = 0;
    entry->on_gpu = false;

    model_cache.total_size += size;
    return 0;
}

void model_cache_pin(uint64_t model_hash)
{
    for (uint32_t i = 0; i < model_cache.count; i++) {
        if (model_cache.entries[i].model_hash == model_hash) {
            model_cache.entries[i].pin_count++;
            return;
        }
    }
}

void model_cache_unpin(uint64_t model_hash)
{
    for (uint32_t i = 0; i < model_cache.count; i++) {
        if (model_cache.entries[i].model_hash == model_hash) {
            if (model_cache.entries[i].pin_count > 0)
                model_cache.entries[i].pin_count--;
            return;
        }
    }
}

void model_cache_evict_lru(void)
{
    if (model_cache.count == 0) return;

    /* Find least recently used unpinned entry */
    int lru_idx = -1;
    uint64_t oldest = UINT64_MAX;

    for (uint32_t i = 0; i < model_cache.count; i++) {
        if (model_cache.entries[i].pin_count == 0 &&
            model_cache.entries[i].ref_count == 0 &&
            model_cache.entries[i].last_access < oldest) {
            oldest = model_cache.entries[i].last_access;
            lru_idx = i;
        }
    }

    if (lru_idx < 0) return; /* Everything is pinned */

    model_cache.total_size -= model_cache.entries[lru_idx].size;

    /* Shift entries down */
    for (uint32_t i = lru_idx; i < model_cache.count - 1; i++)
        model_cache.entries[i] = model_cache.entries[i + 1];
    model_cache.count--;
}

/* =============================================================================
 * Slab Allocator for small kernel allocations
 * =============================================================================*/

#define SLAB_SIZES      8
#define SLAB_PER_PAGE   4096

typedef struct slab {
    void          *free_list;
    uint32_t       obj_size;
    uint32_t       total;
    uint32_t       used;
    struct slab   *next;
} slab_t;

static slab_t *slab_caches[SLAB_SIZES]; /* 16, 32, 64, 128, 256, 512, 1024, 2048 */
static const uint32_t slab_sizes[SLAB_SIZES] = {16, 32, 64, 128, 256, 512, 1024, 2048};

static int slab_size_index(uint64_t size)
{
    for (int i = 0; i < SLAB_SIZES; i++) {
        if (size <= slab_sizes[i])
            return i;
    }
    return -1; /* Too large for slab */
}

/* Create a new slab page for a given size class.
 * Each slab uses one 4K page: the slab_t header lives at the start,
 * followed by as many objects as fit. Each free object stores a pointer
 * to the next free object (freelist threading). */
static slab_t *slab_create(int idx)
{
    uint64_t phys = phys_alloc_pages(1, 1);
    if (phys == 0) return NULL;

    slab_t *slab = (slab_t *)phys;
    uint32_t obj_size = slab_sizes[idx];

    /* Ensure obj_size can hold a pointer for freelist threading */
    if (obj_size < sizeof(void *)) obj_size = sizeof(void *);

    slab->obj_size = obj_size;
    slab->next = slab_caches[idx];
    slab_caches[idx] = slab;

    /* Carve objects from the page, starting after the slab_t header */
    uint8_t *base = (uint8_t *)slab + ((sizeof(slab_t) + obj_size - 1) & ~(obj_size - 1));
    uint8_t *end = (uint8_t *)slab + MM_PAGE_SIZE_4K;
    slab->free_list = NULL;
    slab->total = 0;
    slab->used = 0;

    while (base + obj_size <= end) {
        *(void **)base = slab->free_list;
        slab->free_list = base;
        slab->total++;
        base += obj_size;
    }

    return slab;
}

/* Check if a pointer falls within a slab page */
static bool ptr_in_slab(slab_t *slab, void *ptr)
{
    uint64_t slab_base = (uint64_t)slab;
    uint64_t addr = (uint64_t)ptr;
    return addr >= slab_base && addr < slab_base + MM_PAGE_SIZE_4K;
}

void *kmalloc(uint64_t size)
{
    int idx = slab_size_index(size);
    if (idx >= 0) {
        /* Walk slab chain looking for a slab with free objects */
        slab_t *slab = slab_caches[idx];
        while (slab) {
            if (slab->free_list) {
                void *obj = slab->free_list;
                slab->free_list = *(void **)obj;
                slab->used++;
                return obj;
            }
            slab = slab->next;
        }
        /* No free objects — create a new slab page */
        slab = slab_create(idx);
        if (slab && slab->free_list) {
            void *obj = slab->free_list;
            slab->free_list = *(void **)obj;
            slab->used++;
            return obj;
        }
    }

    /* Fall through to page allocator for large allocations */
    uint64_t pages = (size + MM_PAGE_SIZE_4K - 1) / MM_PAGE_SIZE_4K;
    uint64_t phys = phys_alloc_pages(pages, 1);
    if (phys == 0) return NULL;

    /* Track allocation for kfree */
    if (kfree_track_count < KFREE_TRACK_MAX) {
        kfree_track[kfree_track_count].addr = phys;
        kfree_track[kfree_track_count].pages = (uint32_t)pages;
        kfree_track_count++;
    }

    return (void *)phys; /* Identity mapped in early boot */
}

void kfree(void *ptr)
{
    if (!ptr) return;
    uint64_t addr = (uint64_t)ptr;

    /* Check if it's a slab allocation — walk each size class chain */
    for (int si = 0; si < SLAB_SIZES; si++) {
        slab_t *slab = slab_caches[si];
        while (slab) {
            if (ptr_in_slab(slab, ptr)) {
                /* Return object to this slab's free list */
                *(void **)ptr = slab->free_list;
                slab->free_list = ptr;
                if (slab->used > 0) slab->used--;
                return;
            }
            slab = slab->next;
        }
    }

    /* Check page allocation tracker */
    for (int i = 0; i < kfree_track_count; i++) {
        if (kfree_track[i].addr == addr) {
            phys_free_pages(addr, kfree_track[i].pages);
            kfree_track[i] = kfree_track[--kfree_track_count];
            return;
        }
    }
}

/* =============================================================================
 * Public API Implementation
 * =============================================================================*/

void tensor_mm_init(void)
{
    /* Detect physical memory from multiboot info */
    /* For now, assume 4GB */
    phys_mem_total = 4ULL * 1024 * 1024 * 1024;
    phys_mem_free = phys_mem_total;

    /* Mark first 2MB as reserved: BIOS data, VGA framebuffer, ISA ROM,
     * page tables (0x1000-0x4FFF), multiboot stub, stack, etc. */
    for (uint64_t i = 0; i < 0x200; i++)
        phys_page_mark_used(i);

    /* Mark kernel region as used (kernel is loaded at 2MB).
     * __kernel_end includes .text, .data, .bss, tensor heap, model cache,
     * and git object store. Everything in this range is managed by the
     * kernel's own allocators, not the physical page allocator. */
    uint64_t kernel_start_page = (uint64_t)(uintptr_t)__text_start / MM_PAGE_SIZE_4K;
    uint64_t kernel_end_page = ((uint64_t)(uintptr_t)__kernel_end + MM_PAGE_SIZE_4K - 1) / MM_PAGE_SIZE_4K;
    for (uint64_t i = kernel_start_page; i < kernel_end_page; i++)
        phys_page_mark_used(i);

    /* Initialize tensor heap */
    tensor_heap_init();

    /* Initialize model cache */
    model_cache_init();

    /* Initialize slab caches */
    kmemset(slab_caches, 0, sizeof(slab_caches));

    kstate.memory_total_bytes = phys_mem_total;
    kstate.memory_used_bytes = phys_mem_total - phys_mem_free;

    kprintf_debug("[MM] Initialized: %lu MB total, %lu MB free\n",
                  phys_mem_total / (1024 * 1024), phys_mem_free / (1024 * 1024));
    kprintf_debug("[MM] Tensor heap: %lu MB, Model cache: %lu MB\n",
                  tensor_heap_size / (1024 * 1024), model_cache.max_size / (1024 * 1024));
}

void *tensor_mm_alloc(mm_alloc_request_t *req)
{
    if (!req) return NULL;

    /* Determine alignment */
    uint32_t align = req->alignment;
    if (align < MM_ALIGN_TENSOR)
        align = MM_ALIGN_TENSOR;

    switch (req->zone) {
    case MM_ZONE_TENSOR:
        return tensor_heap_alloc(req->size, align);

    case MM_ZONE_KERNEL:
        return kmalloc(req->size);

    case MM_ZONE_DMA:
        /* DMA allocations must be physically contiguous and page-aligned */
        {
            uint64_t pages = (req->size + MM_PAGE_SIZE_4K - 1) / MM_PAGE_SIZE_4K;
            return (void *)phys_alloc_pages(pages, 1);
        }

    case MM_ZONE_MODEL:
        return tensor_heap_alloc(req->size, align);

    default:
        return kmalloc(req->size);
    }
}

void tensor_mm_free(void *ptr)
{
    if (!ptr) return;

    /* Determine which zone this came from */
    uint64_t addr = (uint64_t)ptr;
    uint64_t heap_start = (uint64_t)tensor_heap_base;
    uint64_t heap_end = heap_start + tensor_heap_size;

    if (addr >= heap_start && addr < heap_end) {
        tensor_heap_free_block(ptr);
    } else {
        kfree(ptr);
    }
}

void *tensor_alloc(uint64_t size)
{
    return tensor_heap_alloc(size, MM_ALIGN_TENSOR);
}

void *tensor_alloc_pinned(uint64_t size)
{
    mm_alloc_request_t req = {
        .size = size,
        .alignment = MM_ALIGN_TENSOR,
        .zone = MM_ZONE_TENSOR,
        .flags = MM_ALLOC_PINNED | MM_ALLOC_PREFAULT,
    };
    return tensor_mm_alloc(&req);
}

void *tensor_alloc_dma(uint64_t size)
{
    mm_alloc_request_t req = {
        .size = size,
        .alignment = MM_ALIGN_PAGE,
        .zone = MM_ZONE_DMA,
        .flags = MM_ALLOC_DMA | MM_ALLOC_PINNED,
    };
    return tensor_mm_alloc(&req);
}

void *tensor_alloc_shared(uint64_t size)
{
    mm_alloc_request_t req = {
        .size = size,
        .alignment = MM_ALIGN_TENSOR,
        .zone = MM_ZONE_TENSOR,
        .flags = MM_ALLOC_SHARED,
    };
    return tensor_mm_alloc(&req);
}

void tensor_free(void *ptr)
{
    tensor_mm_free(ptr);
}

/* =============================================================================
 * Memory Maintenance
 * =============================================================================*/

void tensor_mm_defrag(void)
{
    /* Walk the free list and coalesce adjacent free blocks */
    heap_block_t *block = tensor_heap_first;
    while (block && block->next) {
        if (block->free && block->next->free) {
            block->size += sizeof(heap_block_t) + block->next->size;
            block->next = block->next->next;
            if (block->next)
                block->next->prev = block;
        } else {
            block = block->next;
        }
    }
}

void tensor_mm_cache_warmup(void)
{
    /* Prefetch model weights that are predicted to be needed soon */
    /* Based on historical access patterns */
    /* TODO: implement predictive prefetching */
}

/* =============================================================================
 * Statistics
 * =============================================================================*/

uint64_t tensor_mm_heap_size(void)
{
    return tensor_heap_size;
}

uint64_t tensor_mm_cache_size(void)
{
    return model_cache.max_size;
}

uint64_t tensor_mm_free_bytes(void)
{
    return tensor_heap_size - tensor_heap_used;
}

void tensor_mm_get_stats(mm_stats_t *stats)
{
    if (!stats) return;

    stats->total_phys = phys_mem_total;
    stats->free_phys = phys_mem_free;
    stats->tensor_heap_size = tensor_heap_size;
    stats->tensor_heap_used = tensor_heap_used;
    stats->model_cache_size = model_cache.max_size;
    stats->model_cache_used = model_cache.total_size;
}

#ifndef __aarch64__
/* =============================================================================
 * Virtual Memory: 4K Page Mapping
 *
 * The boot loader identity-maps 4GB using 2MB huge pages.
 * This module can "split" a 2MB huge page into 512 × 4K pages and then
 * individually control each 4K mapping.  Primary use: demand paging for
 * model weights &mdash; fault on first access, allocate physical page, map it,
 * and resume.
 *
 * Page table layout (set up by multiboot_stub.asm):
 *   PML4   @ 0x1000
 *   PDPT   @ 0x2000
 *   PD0-3  @ 0x3000-0x6000  (each PD covers 1GB with 512 × 2MB entries)
 * =============================================================================*/

#define PT_PRESENT   0x001ULL
#define PT_WRITE     0x002ULL
#define PT_USER      0x004ULL
#define PT_HUGE      0x080ULL  /* 2MB page in PD entry */
#define PT_NX        (1ULL << 63)
#define PT_ADDR_MASK 0x000FFFFFFFFFF000ULL

/* Pool of pre-allocated 4K page table pages for splitting huge pages */
#define VM_PT_POOL_MAX 16
static uint64_t vm_pt_pool[VM_PT_POOL_MAX]; /* physical addresses of PT pages */
static int vm_pt_pool_count = 0;

/* Split a 2MB huge page into 512 × 4K pages.
 * Returns the physical address of the new page table, or 0 on failure. */
static uint64_t vm_split_huge_page(uint64_t huge_phys_base)
{
    /* Allocate a page for the new page table */
    uint64_t pt_phys;
    if (vm_pt_pool_count > 0) {
        pt_phys = vm_pt_pool[--vm_pt_pool_count];
    } else {
        pt_phys = phys_alloc_pages(1, 1);
        if (pt_phys == 0) return 0;
    }

    /* Fill 512 4K entries mapping the same 2MB region */
    volatile uint64_t *pt = (volatile uint64_t *)(uintptr_t)pt_phys;
    for (int i = 0; i < 512; i++) {
        pt[i] = (huge_phys_base + (uint64_t)i * 4096) | PT_PRESENT | PT_WRITE;
    }

    return pt_phys;
}

/* Map a single 4K page: vaddr → paddr with given flags.
 * If the PD entry is still a 2MB huge page, splits it first.
 * Returns 0 on success, -1 on failure. */
int vm_map_4k(uint64_t vaddr, uint64_t paddr, uint64_t flags)
{
    /* Only works within first 4GB identity-mapped region */
    if (vaddr >= 0x100000000ULL) return -1;

    /* Navigate page tables */
    uint64_t pd_index  = (vaddr >> 21) & 0x1FF;    /* Which 2MB slot */
    uint64_t gb_index  = (vaddr >> 30) & 0x3;       /* Which GB (0-3) */
    uint64_t pt_index  = (vaddr >> 12) & 0x1FF;     /* Which 4K slot within 2MB */

    /* PD base addresses: 0x3000 + gb_index * 0x1000 */
    volatile uint64_t *pd = (volatile uint64_t *)(uintptr_t)(0x3000 + gb_index * 0x1000);
    uint64_t pd_entry = pd[pd_index];

    volatile uint64_t *pt;

    if (pd_entry & PT_HUGE) {
        /* Split the 2MB huge page */
        uint64_t huge_phys = pd_entry & 0x000FFFFFFFE00000ULL; /* 2MB-aligned phys addr */
        uint64_t pt_phys = vm_split_huge_page(huge_phys);
        if (pt_phys == 0) return -1;

        /* Replace PD entry: point to new PT, remove HUGE flag */
        pd[pd_index] = pt_phys | PT_PRESENT | PT_WRITE;

        /* Invalidate TLB for the entire 2MB region */
        for (int i = 0; i < 512; i++) {
            uint64_t inv_addr = (gb_index << 30) | (pd_index << 21) | ((uint64_t)i << 12);
            __asm__ volatile ("invlpg (%0)" : : "r"(inv_addr) : "memory");
        }

        pt = (volatile uint64_t *)(uintptr_t)pt_phys;
    } else if (pd_entry & PT_PRESENT) {
        /* Already split — get existing PT */
        pt = (volatile uint64_t *)(uintptr_t)(pd_entry & PT_ADDR_MASK);
    } else {
        return -1; /* PD entry not present */
    }

    /* Set the 4K page table entry */
    pt[pt_index] = (paddr & PT_ADDR_MASK) | flags;

    /* Invalidate TLB for this page */
    __asm__ volatile ("invlpg (%0)" : : "r"(vaddr) : "memory");

    return 0;
}

/* Unmap a 4K page (set entry to not-present). */
void vm_unmap_4k(uint64_t vaddr)
{
    if (vaddr >= 0x100000000ULL) return;

    uint64_t pd_index = (vaddr >> 21) & 0x1FF;
    uint64_t gb_index = (vaddr >> 30) & 0x3;
    uint64_t pt_index = (vaddr >> 12) & 0x1FF;

    volatile uint64_t *pd = (volatile uint64_t *)(uintptr_t)(0x3000 + gb_index * 0x1000);
    uint64_t pd_entry = pd[pd_index];

    if ((pd_entry & PT_PRESENT) && !(pd_entry & PT_HUGE)) {
        volatile uint64_t *pt = (volatile uint64_t *)(uintptr_t)(pd_entry & PT_ADDR_MASK);
        pt[pt_index] = 0; /* Not present */
        __asm__ volatile ("invlpg (%0)" : : "r"(vaddr) : "memory");
    }
}

/* Handle a demand-page fault: allocate a physical page, map it, return success.
 * Returns 0 if the fault was handled (caller should IRETQ), -1 if not ours. */
int vm_demand_fault(uint64_t fault_addr)
{
    /* Only handle faults in the upper 2-3 GB range (where model data lives) */
    if (fault_addr < 0x40000000ULL || fault_addr >= 0x100000000ULL) return -1;

    /* Allocate a physical page */
    uint64_t phys = phys_alloc_pages(1, 1);
    if (phys == 0) return -1;

    /* Zero it */
    kmemset((void *)(uintptr_t)phys, 0, 4096);

    /* Map it at the faulting address (4K-aligned) */
    uint64_t page_addr = fault_addr & ~0xFFFULL;
    if (vm_map_4k(page_addr, phys, PT_PRESENT | PT_WRITE) != 0) {
        phys_free_pages(phys, 1);
        return -1;
    }

    return 0;
}
#endif /* !__aarch64__ */
