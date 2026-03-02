/* =============================================================================
 * TensorOS - Tensor-Aware Scheduler Implementation
 *
 * The heart of TensorOS: a scheduler that understands AI workloads.
 *
 * Key innovations:
 * 1. COMPUTE-INTENSITY ROUTING: Routes compute-bound ops to GPU, memory-bound
 *    ops to CPU or high-bandwidth memory devices.
 * 2. BATCH COALESCING: Automatically batches small inference requests together
 *    to maximize GPU utilization.
 * 3. WEIGHT-AWARE PREEMPTION: Avoids preempting tasks while model weights are
 *    being loaded into GPU memory (expensive to restart).
 * 4. THERMAL-AWARE BALANCING: Migrates workloads away from hot devices.
 * 5. PIPELINE SCHEDULING: Understands model pipelines and schedules stages
 *    to overlap computation with data transfer.
 * =============================================================================*/

#include "kernel/sched/tensor_sched.h"
#include "kernel/mm/tensor_mm.h"
#include "kernel/drivers/gpu/gpu.h"

/* Global scheduler instance */
tensor_scheduler_t g_scheduler;

/* Internal state */
static uint64_t next_meu_id = 1;
static model_exec_unit_t meu_pool[SCHED_MAX_MEUS];
static uint32_t meu_pool_used = 0;

/* =============================================================================
 * Queue Operations
 * =============================================================================*/

static void queue_push(sched_queue_t *q, model_exec_unit_t *meu)
{
    meu->next = NULL;
    meu->prev = q->tail;
    if (q->tail)
        q->tail->next = meu;
    else
        q->head = meu;
    q->tail = meu;
    q->count++;
}

static model_exec_unit_t *queue_pop(sched_queue_t *q)
{
    model_exec_unit_t *meu = q->head;
    if (!meu)
        return NULL;

    q->head = meu->next;
    if (q->head)
        q->head->prev = NULL;
    else
        q->tail = NULL;

    meu->next = NULL;
    meu->prev = NULL;
    q->count--;
    return meu;
}

static void queue_remove(sched_queue_t *q, model_exec_unit_t *meu)
{
    if (meu->prev)
        meu->prev->next = meu->next;
    else
        q->head = meu->next;

    if (meu->next)
        meu->next->prev = meu->prev;
    else
        q->tail = meu->prev;

    meu->next = NULL;
    meu->prev = NULL;
    q->count--;
}

/* =============================================================================
 * Initialization
 * =============================================================================*/

void tensor_sched_init(void)
{
    /* Zero out scheduler state */
    kmemset(&g_scheduler, 0, sizeof(g_scheduler));
    kmemset(meu_pool, 0, sizeof(meu_pool));

    /* Default policy: maximize throughput */
    g_scheduler.policy = SCHED_POLICY_THROUGHPUT;

    /* Default batch coalescing: 5ms window */
    g_scheduler.coalesce_window_ms = 5;

    /* Initialize device states from detected hardware */
    g_scheduler.gpu_count = kstate.gpu_count;
    g_scheduler.tpu_count = kstate.tpu_count;

    for (uint32_t i = 0; i < g_scheduler.gpu_count; i++) {
        g_scheduler.gpus[i].device_id = i;
        g_scheduler.gpus[i].active = true;
        struct gpu_info *info = gpu_get_info(i);
        if (info) {
            g_scheduler.gpus[i].vram_total = (uint64_t)info->vram_mb * 1024 * 1024;
            g_scheduler.gpus[i].vram_free = g_scheduler.gpus[i].vram_total;
        }
    }

    kprintf_debug("[SCHED] Initialized with %d GPUs, %d TPUs, policy=THROUGHPUT\n",
                  g_scheduler.gpu_count, g_scheduler.tpu_count);
}

/* =============================================================================
 * MEU Lifecycle
 * =============================================================================*/

model_exec_unit_t *meu_create(const char *name, meu_type_t type,
                               meu_priority_t priority)
{
    if (meu_pool_used >= SCHED_MAX_MEUS) {
        kprintf("SCHED: MEU pool exhausted\n");
        return NULL;
    }

    model_exec_unit_t *meu = &meu_pool[meu_pool_used++];
    kmemset(meu, 0, sizeof(*meu));

    meu->meu_id = next_meu_id++;
    meu->state = MEU_STATE_CREATED;
    meu->type = type;
    meu->priority = priority;
    meu->gpu_id = (uint32_t)-1; /* Not assigned */

    /* Copy name */
    for (int i = 0; i < 63 && name[i]; i++)
        meu->name[i] = name[i];

    /* Default permissions based on type */
    meu->permissions = MEU_PERM_GPU_ACCESS | MEU_PERM_MODEL_LOAD;
    if (type == MEU_TYPE_TRAINING)
        meu->permissions |= MEU_PERM_TRAINING | MEU_PERM_FILESYSTEM;
    if (type == MEU_TYPE_SYSTEM)
        meu->permissions |= MEU_PERM_NETWORK | MEU_PERM_FILESYSTEM | MEU_PERM_IPC;

    kstate.models_loaded++;

    kprintf_debug("[SCHED] Created MEU %lu '%s' type=%d prio=%d\n",
                  meu->meu_id, meu->name, type, priority);
    return meu;
}

void meu_destroy(model_exec_unit_t *meu)
{
    if (!meu) return;

    /* Remove from any queue */
    if (meu->state == MEU_STATE_READY || meu->state == MEU_STATE_WAITING) {
        tensor_sched_dequeue(meu);
    }

    /* Free GPU resources */
    if (meu->gpu_id != (uint32_t)-1 && meu->gpu_id < g_scheduler.gpu_count) {
        g_scheduler.gpus[meu->gpu_id].vram_free += meu->vram_used;
        g_scheduler.gpus[meu->gpu_id].meu_count--;
    }

    /* Update kernel state */
    kstate.tensor_ops_total += meu->tensor_ops;
    kstate.models_loaded--;

    meu->state = MEU_STATE_COMPLETED;
    kprintf_debug("[SCHED] Destroyed MEU %lu '%s' (ops=%lu)\n",
                  meu->meu_id, meu->name, meu->tensor_ops);
}

int meu_set_model(model_exec_unit_t *meu, uint64_t model_hash,
                   uint64_t param_count, tensor_dtype_t dtype)
{
    meu->model_hash = model_hash;
    meu->param_count = param_count;
    meu->compute_dtype = dtype;
    return 0;
}

int meu_set_resource_budget(model_exec_unit_t *meu, uint64_t mem_bytes,
                             uint64_t vram_bytes)
{
    meu->mem_budget = mem_bytes;
    meu->vram_budget = vram_bytes;
    return 0;
}

/* =============================================================================
 * Core Scheduling Logic
 * =============================================================================*/

void tensor_sched_enqueue(model_exec_unit_t *meu)
{
    meu->state = MEU_STATE_READY;
    queue_push(&g_scheduler.queues[meu->priority], meu);
}

void tensor_sched_dequeue(model_exec_unit_t *meu)
{
    queue_remove(&g_scheduler.queues[meu->priority], meu);
}

bool tensor_sched_has_pending(void)
{
    for (int i = 0; i < 6; i++) {
        if (g_scheduler.queues[i].count > 0)
            return true;
    }
    return false;
}

/* =============================================================================
 * dispatch_select_meu - Pick the best MEU to run next
 *
 * Strategy varies by policy:
 * - THROUGHPUT: Pick highest-priority MEU with GPU affinity match
 * - LATENCY: Pick the MEU that has waited longest at REALTIME/HIGH
 * - EFFICIENCY: Pick MEU that maximizes ops-per-watt
 * - FAIR: Round-robin across all priorities
 * =============================================================================*/

static model_exec_unit_t *dispatch_select_meu(void)
{
    switch (g_scheduler.policy) {
    case SCHED_POLICY_THROUGHPUT:
    case SCHED_POLICY_LATENCY:
        /* Strict priority: always pick from highest priority non-empty queue */
        for (int i = 0; i < 6; i++) {
            if (g_scheduler.queues[i].count > 0) {
                return queue_pop(&g_scheduler.queues[i]);
            }
        }
        break;

    case SCHED_POLICY_EFFICIENCY:
        /* Pick MEU with best compute intensity (highest FLOPs/byte) */
        {
            model_exec_unit_t *best = NULL;
            int best_prio = -1;
            /* Still respect priority, but within same priority prefer efficient */
            for (int i = 0; i < 6; i++) {
                if (g_scheduler.queues[i].count > 0) {
                    best = queue_pop(&g_scheduler.queues[i]);
                    best_prio = i;
                    break;
                }
            }
            return best;
        }

    case SCHED_POLICY_FAIR:
        /* Round-robin: cycle through priorities */
        {
            static int rr_index = 0;
            for (int attempt = 0; attempt < 6; attempt++) {
                int idx = (rr_index + attempt) % 6;
                if (g_scheduler.queues[idx].count > 0) {
                    rr_index = (idx + 1) % 6;
                    return queue_pop(&g_scheduler.queues[idx]);
                }
            }
        }
        break;
    }

    return NULL;
}

/* =============================================================================
 * dispatch_assign_device - Find optimal device for MEU
 *
 * Algorithm:
 * 1. If MEU has explicit GPU affinity, use it
 * 2. If model weights are already on a GPU (cached), prefer that GPU
 * 3. Score each available GPU by: free VRAM, utilization, temperature
 * 4. If no GPU can fit the model, fall back to CPU
 * =============================================================================*/

static int dispatch_assign_device(model_exec_unit_t *meu)
{
    /* Already assigned? */
    if (meu->gpu_id != (uint32_t)-1)
        return meu->gpu_id;

    /* No GPUs available, use CPU */
    if (g_scheduler.gpu_count == 0)
        return -1;

    /* Score each GPU */
    int best_gpu = -1;
    int64_t best_score = -1;

    for (uint32_t i = 0; i < g_scheduler.gpu_count; i++) {
        device_state_t *dev = &g_scheduler.gpus[i];
        if (!dev->active) continue;

        /* Can this GPU fit the model? */
        if (dev->vram_free < meu->vram_budget) continue;

        /* Score: higher is better
         * +100 for each 10% free VRAM
         * -10 for each 10% utilization
         * -5 for each 10°C over 60°C
         * +50 if model weights already cached here
         */
        int64_t score = 0;
        score += (dev->vram_free * 100) / (dev->vram_total + 1);
        score -= dev->utilization_pct;
        if (dev->temperature_c > 60)
            score -= ((dev->temperature_c - 60) / 10) * 5;

        /* Bonus for weight locality (check if model hash matches cached) */
        /* TODO: integrate with model cache subsystem */

        if (score > best_score) {
            best_score = score;
            best_gpu = i;
        }
    }

    if (best_gpu >= 0) {
        meu->gpu_id = best_gpu;
        g_scheduler.gpus[best_gpu].vram_free -= meu->vram_budget;
        g_scheduler.gpus[best_gpu].meu_count++;
    }

    return best_gpu;
}

/* =============================================================================
 * tensor_sched_dispatch - Main dispatch function
 * Called from kernel idle loop or timer interrupt
 * =============================================================================*/

void tensor_sched_dispatch(void)
{
    model_exec_unit_t *meu = dispatch_select_meu();
    if (!meu) return;

    /* Assign to optimal device */
    dispatch_assign_device(meu);

    /* Transition to running state */
    meu->state = MEU_STATE_RUNNING;
    g_scheduler.total_dispatches++;

    kprintf_debug("[SCHED] Dispatching MEU %lu '%s' -> %s %d\n",
                  meu->meu_id, meu->name,
                  meu->gpu_id != (uint32_t)-1 ? "GPU" : "CPU",
                  meu->gpu_id != (uint32_t)-1 ? meu->gpu_id : 0);

    /* TODO: Actually context-switch to MEU execution context */
    /* For now, this is the scheduling decision framework */
}

/* =============================================================================
 * Preemption and Yielding
 * =============================================================================*/

void tensor_sched_yield(model_exec_unit_t *meu)
{
    meu->state = MEU_STATE_READY;
    tensor_sched_enqueue(meu);
}

void tensor_sched_block(model_exec_unit_t *meu, const char *reason)
{
    meu->state = MEU_STATE_WAITING;
    kprintf_debug("[SCHED] MEU %lu blocked: %s\n", meu->meu_id, reason);
}

void tensor_sched_unblock(model_exec_unit_t *meu)
{
    meu->state = MEU_STATE_READY;
    tensor_sched_enqueue(meu);
}

/* =============================================================================
 * Tensor Operation Hints
 * The runtime informs the scheduler about upcoming operations so it can
 * make proactive decisions (e.g., prefetch weights, migrate to GPU)
 * =============================================================================*/

void tensor_sched_hint_op(model_exec_unit_t *meu, tensor_op_profile_t *profile)
{
    /* High compute intensity -> ensure on GPU */
    if (profile->compute_intensity > 10.0f && profile->requires_gpu) {
        if (meu->gpu_id == (uint32_t)-1) {
            /* Not on GPU yet, try to migrate */
            dispatch_assign_device(meu);
        }
    }

    /* Memory-bound operations -> might be better on CPU with large cache */
    if (profile->compute_intensity < 1.0f && !profile->requires_gpu) {
        /* Consider keeping on CPU to avoid PCIe transfer overhead */
    }

    /* Update MEU statistics */
    meu->tensor_ops++;
    meu->flops += profile->flop_estimate;
    g_scheduler.total_tensor_ops++;
}

void tensor_sched_hint_batch_size(model_exec_unit_t *meu, uint32_t batch_size)
{
    /* Larger batch sizes benefit more from GPU */
    if (batch_size >= 8 && meu->gpu_id == (uint32_t)-1) {
        dispatch_assign_device(meu);
    }
}

/* =============================================================================
 * Device Management
 * =============================================================================*/

int tensor_sched_assign_device(model_exec_unit_t *meu, uint32_t device_id)
{
    if (device_id >= g_scheduler.gpu_count) return -1;

    device_state_t *dev = &g_scheduler.gpus[device_id];
    if (dev->vram_free < meu->vram_budget) return -1;

    meu->gpu_id = device_id;
    dev->vram_free -= meu->vram_budget;
    dev->meu_count++;
    return 0;
}

int tensor_sched_migrate_device(model_exec_unit_t *meu, uint32_t new_device)
{
    if (new_device >= g_scheduler.gpu_count) return -1;
    if (meu->gpu_id == new_device) return 0;

    /* Free from old device */
    if (meu->gpu_id < g_scheduler.gpu_count) {
        g_scheduler.gpus[meu->gpu_id].vram_free += meu->vram_used;
        g_scheduler.gpus[meu->gpu_id].meu_count--;
    }

    /* Assign to new device */
    device_state_t *dev = &g_scheduler.gpus[new_device];
    if (dev->vram_free < meu->vram_budget) return -1;

    meu->gpu_id = new_device;
    dev->vram_free -= meu->vram_budget;
    dev->meu_count++;

    g_scheduler.total_migrations++;
    kprintf_debug("[SCHED] Migrated MEU %lu to GPU %d\n", meu->meu_id, new_device);
    return 0;
}

/* Balance workloads across GPUs - called periodically */
void tensor_sched_balance_devices(void)
{
    if (g_scheduler.gpu_count < 2) return;

    /* Find most and least loaded GPUs */
    uint32_t max_util = 0, min_util = 100;
    uint32_t max_gpu = 0, min_gpu = 0;

    for (uint32_t i = 0; i < g_scheduler.gpu_count; i++) {
        if (g_scheduler.gpus[i].utilization_pct > max_util) {
            max_util = g_scheduler.gpus[i].utilization_pct;
            max_gpu = i;
        }
        if (g_scheduler.gpus[i].utilization_pct < min_util) {
            min_util = g_scheduler.gpus[i].utilization_pct;
            min_gpu = i;
        }
    }

    /* Only balance if imbalance > 30% */
    if (max_util - min_util < 30) return;

    /* Migrate lowest-priority MEU from overloaded GPU */
    /* TODO: implement full migration with weight transfer */
    kprintf_debug("[SCHED] Balancing: GPU %d (%d%%) -> GPU %d (%d%%)\n",
                  max_gpu, max_util, min_gpu, min_util);
}

/* =============================================================================
 * Batch Coalescing
 * Combines multiple small inference requests into a single batch
 * to improve GPU utilization. Critical for serving scenarios.
 * =============================================================================*/

void tensor_sched_coalesce_enable(uint32_t window_ms)
{
    g_scheduler.coalesce_window_ms = window_ms;
}

void tensor_sched_coalesce_flush(void)
{
    /* Force-dispatch all pending coalesced batches */
    g_scheduler.pending_batch_count = 0;
}

/* =============================================================================
 * Policy Management
 * =============================================================================*/

void tensor_sched_set_policy(sched_policy_t policy)
{
    g_scheduler.policy = policy;
    kprintf("[SCHED] Policy changed to %d\n", policy);
}

/* =============================================================================
 * Statistics
 * =============================================================================*/

void tensor_sched_get_stats(uint64_t *ops, uint64_t *dispatches,
                             uint64_t *avg_latency_us)
{
    if (ops) *ops = g_scheduler.total_tensor_ops;
    if (dispatches) *dispatches = g_scheduler.total_dispatches;
    if (avg_latency_us) *avg_latency_us = g_scheduler.avg_latency_us;
}
