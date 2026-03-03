/* =============================================================================
 * TensorOS - Tensor-Aware Memory Manager Header
 *
 * Memory management designed for AI workloads:
 * - Huge page support for model weights (2MB/1GB pages)
 * - Tensor-aligned allocations (cache-line, SIMD-width aligned)
 * - Model weight caching with LRU eviction
 * - Zero-copy tensor sharing between MEUs
 * - Predictive prefetching based on model architecture
 * - GPU/TPU memory unified addressing
 * =============================================================================*/

#ifndef TENSOROS_TENSOR_MM_H
#define TENSOROS_TENSOR_MM_H

#include "kernel/core/kernel.h"

/* =============================================================================
 * Memory Regions
 * =============================================================================*/

#define MM_PAGE_SIZE_4K     0x1000
#define MM_PAGE_SIZE_2M     0x200000
#define MM_PAGE_SIZE_1G     0x40000000

#define MM_ALIGN_CACHELINE  64
#define MM_ALIGN_SIMD_256   32
#define MM_ALIGN_SIMD_512   64
#define MM_ALIGN_TENSOR     256     /* Default tensor alignment */
#define MM_ALIGN_PAGE       4096

/* Memory zone types */
typedef enum {
    MM_ZONE_KERNEL    = 0,  /* Kernel code and data */
    MM_ZONE_TENSOR    = 1,  /* Tensor computation workspace */
    MM_ZONE_MODEL     = 2,  /* Model weight storage/cache */
    MM_ZONE_DMA       = 3,  /* DMA-capable memory for GPU transfers */
    MM_ZONE_GIT       = 4,  /* Git object store */
    MM_ZONE_USER      = 5,  /* General userland allocations */
} mm_zone_type_t;

/* =============================================================================
 * Tensor Allocation Descriptor
 * =============================================================================*/

typedef struct {
    uint64_t    size;           /* Requested size */
    uint32_t    alignment;      /* Required alignment */
    mm_zone_type_t zone;        /* Memory zone */
    uint32_t    flags;          /* MM_ALLOC_* flags */
    uint32_t    numa_node;      /* NUMA node preference (-1 = any) */
    uint32_t    device_id;      /* Target device for DMA */
} mm_alloc_request_t;

/* Allocation flags */
#define MM_ALLOC_PINNED     (1 << 0)  /* Non-swappable */
#define MM_ALLOC_HUGE_2M    (1 << 1)  /* Use 2MB pages */
#define MM_ALLOC_HUGE_1G    (1 << 2)  /* Use 1GB pages */
#define MM_ALLOC_ZEROED     (1 << 3)  /* Zero-fill */
#define MM_ALLOC_DMA        (1 << 4)  /* DMA-capable */
#define MM_ALLOC_SHARED     (1 << 5)  /* Shareable between MEUs */
#define MM_ALLOC_NOCACHE    (1 << 6)  /* Uncacheable (for MMIO) */
#define MM_ALLOC_PREFAULT   (1 << 7)  /* Pre-fault all pages */

/* =============================================================================
 * Model Weight Cache
 * Caches model weights in memory to avoid reloading from disk
 * =============================================================================*/

#define MODEL_CACHE_MAX_ENTRIES  64

typedef struct {
    uint64_t    model_hash;     /* SHA-256 hash of model */
    void       *data;           /* Pointer to cached weights */
    uint64_t    size;           /* Size in bytes */
    uint64_t    last_access;    /* Tick of last access */
    uint32_t    ref_count;      /* Reference count */
    uint32_t    pin_count;      /* Pin count (prevents eviction) */
    bool        on_gpu;         /* Also resident on GPU? */
    uint32_t    gpu_id;         /* Which GPU */
} model_cache_entry_t;

typedef struct {
    model_cache_entry_t entries[MODEL_CACHE_MAX_ENTRIES];
    uint32_t    count;
    uint64_t    total_size;
    uint64_t    max_size;
    uint64_t    hits;
    uint64_t    misses;
} model_cache_t;

/* =============================================================================
 * Memory Statistics
 * =============================================================================*/

typedef struct {
    uint64_t    total_phys;         /* Total physical memory */
    uint64_t    free_phys;          /* Free physical memory */
    uint64_t    tensor_heap_size;   /* Tensor heap size */
    uint64_t    tensor_heap_used;   /* Tensor heap used */
    uint64_t    model_cache_size;   /* Model cache size */
    uint64_t    model_cache_used;   /* Model cache used */
    uint64_t    gpu_mem_total;      /* Total GPU memory across all GPUs */
    uint64_t    gpu_mem_used;       /* Used GPU memory */
    uint64_t    alloc_count;        /* Total allocations */
    uint64_t    free_count;         /* Total frees */
    uint64_t    page_faults;        /* Page faults */
    uint64_t    huge_pages_used;    /* 2MB pages in use */
} mm_stats_t;

/* =============================================================================
 * Memory Manager API
 * =============================================================================*/

/* Initialization */
void tensor_mm_init(void);

/* Core allocation */
void *tensor_mm_alloc(mm_alloc_request_t *req);
void  tensor_mm_free(void *ptr);
void *tensor_mm_realloc(void *ptr, uint64_t new_size);

/* Convenience allocators */
void *tensor_alloc(uint64_t size);           /* Tensor-aligned allocation */
void *tensor_alloc_pinned(uint64_t size);    /* Pinned, non-swappable */
void *tensor_alloc_dma(uint64_t size);       /* DMA-capable for GPU xfer */
void *tensor_alloc_shared(uint64_t size);    /* Shared between MEUs */
void  tensor_free(void *ptr);

/* Kernel allocations */
void *kmalloc(uint64_t size);
void  kfree(void *ptr);

/* Model weight cache */
void *model_cache_get(uint64_t model_hash, uint64_t *size);
int   model_cache_put(uint64_t model_hash, void *data, uint64_t size);
void  model_cache_pin(uint64_t model_hash);
void  model_cache_unpin(uint64_t model_hash);
void  model_cache_evict_lru(void);

/* GPU memory management */
void *gpu_mem_alloc(uint32_t gpu_id, uint64_t size);
void  gpu_mem_free(uint32_t gpu_id, void *ptr);
int   gpu_mem_copy_h2d(uint32_t gpu_id, void *dst, const void *src, uint64_t size);
int   gpu_mem_copy_d2h(void *dst, uint32_t gpu_id, const void *src, uint64_t size);

/* Memory maintenance */
void  tensor_mm_defrag(void);
void  tensor_mm_cache_warmup(void);

/* Statistics */
uint64_t tensor_mm_heap_size(void);
uint64_t tensor_mm_cache_size(void);
uint64_t tensor_mm_free_bytes(void);
void     tensor_mm_get_stats(mm_stats_t *stats);

#ifndef __aarch64__
/* Virtual Memory: 4K page mapping (splits 2MB huge pages on demand) */
int  vm_map_4k(uint64_t vaddr, uint64_t paddr, uint64_t flags);
void vm_unmap_4k(uint64_t vaddr);
int  vm_demand_fault(uint64_t fault_addr); /* Returns 0 if handled, -1 if not */
#endif /* !__aarch64__ */

#endif /* TENSOROS_TENSOR_MM_H */
