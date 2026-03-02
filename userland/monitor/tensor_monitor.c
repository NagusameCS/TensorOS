/* =============================================================================
 * TensorOS - Tensor Monitor Daemon Implementation
 * =============================================================================*/

#include "userland/monitor/tensor_monitor.h"

void monitor_init(tensor_monitor_t *mon)
{
    kmemset(mon, 0, sizeof(*mon));
    mon->gpu_temp_warn       = 80;
    mon->gpu_temp_critical   = 95;
    mon->vram_pressure_thresh = 0.9f;
    mon->cache_thrash_thresh  = 0.3f;
    mon->running             = true;
    mon->last_tensor_ops     = 0;
    mon->last_ipc_count      = 0;
}

static void monitor_add_alert(tensor_monitor_t *mon, alert_type_t type,
                               uint32_t device, const char *msg)
{
    if (mon->alert_count >= MONITOR_MAX_ALERTS) return;
    monitor_alert_t *a = &mon->alerts[mon->alert_count++];
    a->type = type;
    a->device_id = device;
    a->timestamp = kstate.uptime_ticks;
    /* Copy message */
    int i = 0;
    while (msg[i] && i < 127) { a->message[i] = msg[i]; i++; }
    a->message[i] = '\0';
}

void monitor_tick(tensor_monitor_t *mon)
{
    /* Collect sample */
    uint32_t idx = mon->sample_cursor % MONITOR_MAX_SAMPLES;
    monitor_sample_t *s = &mon->samples[idx];

    s->timestamp = kstate.uptime_ticks;
    s->meu_running_count = kstate.meu_count;
    s->meu_queued_count  = 0; /* TODO: query scheduler */
    s->memory_used       = kstate.memory_used_bytes;
    s->memory_total      = kstate.memory_total_bytes;

    /* Tensor ops/sec since last tick */
    uint64_t ops_now = kstate.tensor_ops_total;
    s->tensor_ops_per_sec = ops_now - mon->last_tensor_ops;
    mon->last_tensor_ops = ops_now;

    /* GPU metrics (first GPU for simplicity) */
    if (kstate.gpu_count > 0) {
        s->gpu_util_percent = 0;  /* TODO: query driver */
        s->gpu_temp_celsius = 0;
        s->gpu_vram_used    = 0;
        s->gpu_vram_total   = 0;
        s->gpu_power_watts  = 0;
        s->gpu_fan_percent  = 0;
    }

    /* Cache hit rate from mm stats */
    /* TODO: retrieve from tensor_mm */
    s->cache_hit_rate = 0.0f;

    mon->sample_cursor++;
    if (mon->sample_count < MONITOR_MAX_SAMPLES)
        mon->sample_count++;
}

void monitor_check_alerts(tensor_monitor_t *mon)
{
    uint32_t idx = (mon->sample_cursor - 1) % MONITOR_MAX_SAMPLES;
    monitor_sample_t *s = &mon->samples[idx];

    /* GPU temperature */
    if (s->gpu_temp_celsius >= mon->gpu_temp_critical) {
        monitor_add_alert(mon, ALERT_GPU_TEMP_CRITICAL, 0,
                          "GPU temperature CRITICAL - throttling imminent");
        kprintf("[ALERT] GPU temperature critical: %dC\n", s->gpu_temp_celsius);
    } else if (s->gpu_temp_celsius >= mon->gpu_temp_warn) {
        monitor_add_alert(mon, ALERT_GPU_TEMP_HIGH, 0,
                          "GPU temperature elevated");
    }

    /* VRAM pressure */
    if (s->gpu_vram_total > 0) {
        float vram_usage = (float)s->gpu_vram_used / s->gpu_vram_total;
        if (vram_usage >= mon->vram_pressure_thresh) {
            monitor_add_alert(mon, ALERT_VRAM_PRESSURE, 0,
                              "VRAM pressure high — consider model offloading");
        }
    }

    /* System memory OOM */
    if (s->memory_total > 0) {
        float mem_usage = (float)s->memory_used / s->memory_total;
        if (mem_usage > 0.95f) {
            monitor_add_alert(mon, ALERT_OOM_IMMINENT, 0,
                              "System memory near exhaustion");
            kprintf("[ALERT] Memory usage > 95%%!\n");
        }
    }

    /* Cache thrashing */
    if (s->cache_hit_rate < mon->cache_thrash_thresh && s->cache_hit_rate > 0.0f) {
        monitor_add_alert(mon, ALERT_CACHE_THRASHING, 0,
                          "Model cache hit rate critically low");
    }
}

void monitor_print_dashboard(tensor_monitor_t *mon)
{
    uint32_t idx = (mon->sample_cursor - 1) % MONITOR_MAX_SAMPLES;
    monitor_sample_t *s = &mon->samples[idx];

    kprintf("\033[2J\033[H");  /* Clear screen (ANSI) */
    kprintf("+--------------------------------------------------------------+\n");
    kprintf("|             TensorOS System Monitor v0.1                     |\n");
    kprintf("+--------------------------------------------------------------+\n");

    /* System Overview */
    kprintf("|  Uptime: %lu ticks                                          |\n",
            s->timestamp);
    kprintf("|  MEUs Running: %-4d    Queued: %-4d                          |\n",
            s->meu_running_count, s->meu_queued_count);
    kprintf("|  Tensor Ops/s: %-10lu                                     |\n",
            s->tensor_ops_per_sec);

    /* Memory */
    kprintf("+--------------------------------------------------------------+\n");
    kprintf("|  Memory: %lu / %lu MB  ",
            s->memory_used / (1024*1024), s->memory_total / (1024*1024));
    /* Progress bar */
    uint32_t bar_width = 30;
    float usage = s->memory_total > 0 ?
                  (float)s->memory_used / s->memory_total : 0.0f;
    uint32_t filled = (uint32_t)(usage * bar_width);
    kprintf("[");
    for (uint32_t i = 0; i < bar_width; i++)
        kprintf(i < filled ? "#" : "-");
    kprintf("]    |\n");

    /* GPU */
    if (kstate.gpu_count > 0) {
        kprintf("+--------------------------------------------------------------+\n");
        kprintf("|  GPU #0: %dC  Util: %d%%  VRAM: %lu/%lu MB  Power: %dW     |\n",
                s->gpu_temp_celsius, s->gpu_util_percent,
                s->gpu_vram_used / (1024*1024), s->gpu_vram_total / (1024*1024),
                s->gpu_power_watts);
    }

    /* Alerts */
    if (mon->alert_count > 0) {
        kprintf("+--------------------------------------------------------------+\n");
        kprintf("|  Recent Alerts:                                               |\n");
        uint32_t start = mon->alert_count > 5 ? mon->alert_count - 5 : 0;
        for (uint32_t i = start; i < mon->alert_count; i++) {
            kprintf("|  [!] %s\n", mon->alerts[i].message);
        }
    }

    kprintf("+--------------------------------------------------------------+\n");
    kprintf("\nPress 'q' to exit monitor, 'r' to refresh\n");
}

void monitor_daemon_main(void)
{
    static tensor_monitor_t mon;
    monitor_init(&mon);

    /* Do one tick to gather initial stats */
    monitor_tick(&mon);

    kprintf("[MONITOR] Tensor monitor daemon initialized\n");
    /* Returns - real daemon scheduling handled by tensor_sched */
}
