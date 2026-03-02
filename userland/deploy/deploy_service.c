/* =============================================================================
 * TensorOS - Deployment Service Implementation
 * =============================================================================*/

#include "userland/deploy/deploy_service.h"

static deploy_service_t services[DEPLOY_MAX_SERVICES];
static uint32_t         service_count = 0;

static int kstrcmp_d(const char *a, const char *b)
{
    while (*a && *a == *b) { a++; b++; }
    return *(unsigned char *)a - *(unsigned char *)b;
}

static void kstrcpy_d(char *dst, const char *src)
{
    while (*src) *dst++ = *src++;
    *dst = 0;
}

int deploy_init(void)
{
    kmemset(services, 0, sizeof(services));
    service_count = 0;
    kprintf("[DEPLOY] Deployment service initialized\n");
    return 0;
}

static deploy_service_t *find_service(const char *name)
{
    for (uint32_t i = 0; i < service_count; i++) {
        if (kstrcmp_d(services[i].name, name) == 0)
            return &services[i];
    }
    return NULL;
}

int deploy_create(const char *name, const char *model, uint16_t port)
{
    if (service_count >= DEPLOY_MAX_SERVICES) return -1;
    if (find_service(name)) return -2; /* Already exists */

    deploy_service_t *svc = &services[service_count++];
    kmemset(svc, 0, sizeof(*svc));
    kstrcpy_d(svc->name, name);
    kstrcpy_d(svc->model_name, model);
    svc->port = port;
    svc->state = DEPLOY_STATE_STOPPED;
    svc->target_replicas = 1;
    svc->min_replicas = 1;
    svc->max_replicas = DEPLOY_MAX_REPLICAS;
    svc->batch_size = 1;
    svc->batch_timeout_ms = 50;
    svc->scale_up_threshold = 100.0f;   /* 100ms avg → scale up */
    svc->scale_down_threshold = 10.0f;  /* 10ms avg → scale down */
    svc->scale_cooldown_ticks = 30000;

    kprintf("[DEPLOY] Service '%s' created (model=%s, port=%d)\n",
            name, model, port);
    return 0;
}

int deploy_start(const char *name)
{
    deploy_service_t *svc = find_service(name);
    if (!svc) return -1;

    svc->state = DEPLOY_STATE_STARTING;
    kprintf("[DEPLOY] Starting service '%s'...\n", name);

    /* Create initial replicas (MEUs) */
    for (uint32_t i = 0; i < svc->target_replicas; i++) {
        deploy_replica_t *rep = &svc->replicas[svc->replica_count++];
        rep->meu_id = 0; /* TODO: scheduler_submit_meu() */
        rep->requests_served = 0;
        rep->total_latency_us = 0;
        rep->healthy = true;
        kprintf("[DEPLOY] Replica %d started (MEU #%d)\n", i, rep->meu_id);
    }

    svc->state = DEPLOY_STATE_RUNNING;
    svc->uptime_ticks = kstate.uptime_ticks;
    kprintf("[DEPLOY] Service '%s' is RUNNING on port %d (%d replicas)\n",
            name, svc->port, svc->replica_count);
    return 0;
}

int deploy_stop(const char *name)
{
    deploy_service_t *svc = find_service(name);
    if (!svc) return -1;

    svc->state = DEPLOY_STATE_DRAINING;
    kprintf("[DEPLOY] Draining service '%s'...\n", name);

    /* Kill all replicas */
    for (uint32_t i = 0; i < svc->replica_count; i++) {
        /* TODO: scheduler_kill_meu(svc->replicas[i].meu_id) */
    }
    svc->replica_count = 0;
    svc->state = DEPLOY_STATE_STOPPED;
    kprintf("[DEPLOY] Service '%s' stopped\n", name);
    return 0;
}

int deploy_scale(const char *name, uint32_t replicas)
{
    deploy_service_t *svc = find_service(name);
    if (!svc) return -1;
    if (replicas > DEPLOY_MAX_REPLICAS) replicas = DEPLOY_MAX_REPLICAS;

    svc->target_replicas = replicas;
    svc->state = DEPLOY_STATE_SCALING;

    /* Scale up */
    while (svc->replica_count < replicas) {
        deploy_replica_t *rep = &svc->replicas[svc->replica_count++];
        rep->meu_id = 0;
        rep->healthy = true;
        kprintf("[DEPLOY] Scaling up: new replica %d\n", svc->replica_count - 1);
    }

    /* Scale down */
    while (svc->replica_count > replicas) {
        svc->replica_count--;
        kprintf("[DEPLOY] Scaling down: removed replica %d\n", svc->replica_count);
    }

    svc->state = DEPLOY_STATE_RUNNING;
    svc->last_scale_tick = kstate.uptime_ticks;
    kprintf("[DEPLOY] Service '%s' scaled to %d replicas\n", name, replicas);
    return 0;
}

int deploy_enable_autoscale(const char *name, uint32_t min, uint32_t max)
{
    deploy_service_t *svc = find_service(name);
    if (!svc) return -1;

    svc->autoscale_enabled = true;
    svc->min_replicas = min;
    svc->max_replicas = max;
    kprintf("[DEPLOY] Autoscaling enabled for '%s': min=%d, max=%d\n",
            name, min, max);
    return 0;
}

int deploy_submit_request(const char *name, const tensor_desc_t *input,
                           tensor_desc_t *output)
{
    deploy_service_t *svc = find_service(name);
    if (!svc || svc->state != DEPLOY_STATE_RUNNING) return -1;

    deploy_request_queue_t *q = &svc->requests;
    if (q->queue_count >= DEPLOY_MAX_QUEUE) return -2; /* Queue full */

    uint32_t idx = q->queue_tail % DEPLOY_MAX_QUEUE;
    q->queue[idx].input = *input;
    q->queue[idx].submitted_at = kstate.uptime_ticks;
    q->queue[idx].completed = false;
    q->queue_tail++;
    q->queue_count++;
    svc->total_requests++;

    /* Round-robin select replica */
    uint32_t rep_idx = (uint32_t)(svc->total_requests % svc->replica_count);
    deploy_replica_t *rep = &svc->replicas[rep_idx];

    /* TODO: Actually dispatch inference to the MEU and write output */
    q->queue[idx].completed = true;
    q->queue[idx].completed_at = kstate.uptime_ticks;
    q->queue_head++;
    q->queue_count--;

    rep->requests_served++;
    return 0;
}

static void deploy_autoscale_check(deploy_service_t *svc)
{
    if (!svc->autoscale_enabled) return;
    if (svc->state != DEPLOY_STATE_RUNNING) return;
    if (kstate.uptime_ticks - svc->last_scale_tick < svc->scale_cooldown_ticks)
        return;

    /* Calculate average latency across replicas */
    float avg_lat = 0.0f;
    uint32_t active = 0;
    for (uint32_t i = 0; i < svc->replica_count; i++) {
        if (svc->replicas[i].requests_served > 0) {
            avg_lat += svc->replicas[i].avg_latency_ms;
            active++;
        }
    }
    if (active > 0) avg_lat /= active;

    if (avg_lat > svc->scale_up_threshold && svc->replica_count < svc->max_replicas) {
        deploy_scale(svc->name, svc->replica_count + 1);
    } else if (avg_lat < svc->scale_down_threshold &&
               svc->replica_count > svc->min_replicas) {
        deploy_scale(svc->name, svc->replica_count - 1);
    }
}

void deploy_health_check(void)
{
    for (uint32_t i = 0; i < service_count; i++) {
        deploy_service_t *svc = &services[i];
        if (svc->state != DEPLOY_STATE_RUNNING) continue;

        for (uint32_t j = 0; j < svc->replica_count; j++) {
            deploy_replica_t *rep = &svc->replicas[j];
            /* TODO: Ping MEU, check if alive */
            if (!rep->healthy) {
                kprintf("[DEPLOY] Replica %d of '%s' is unhealthy -- restarting\n",
                        j, svc->name);
                rep->healthy = true;
                /* TODO: restart MEU */
            }
        }

        deploy_autoscale_check(svc);
    }
}

void deploy_print_status(void)
{
    kprintf("\n=== Deployment Services ===\n");
    kprintf("%-16s %-10s %-6s %-8s %-12s %-12s\n",
            "NAME", "STATE", "PORT", "REPLICAS", "REQUESTS", "ERRORS");
    kprintf("%-16s %-10s %-6s %-8s %-12s %-12s\n",
            "----", "-----", "----", "--------", "--------", "------");

    for (uint32_t i = 0; i < service_count; i++) {
        deploy_service_t *svc = &services[i];
        const char *state_str =
            svc->state == DEPLOY_STATE_RUNNING  ? "RUNNING"  :
            svc->state == DEPLOY_STATE_STOPPED  ? "STOPPED"  :
            svc->state == DEPLOY_STATE_STARTING ? "STARTING" :
            svc->state == DEPLOY_STATE_SCALING  ? "SCALING"  :
            svc->state == DEPLOY_STATE_DRAINING ? "DRAINING" :
            svc->state == DEPLOY_STATE_FAILED   ? "FAILED"   : "UNKNOWN";

        kprintf("%-16s %-10s %-6d %-8d %-12lu %-12lu\n",
                svc->name, state_str, svc->port,
                svc->replica_count, svc->total_requests, svc->total_errors);
    }
    kprintf("\n");
}

void deploy_daemon_main(void)
{
    deploy_init();
    kprintf("[DEPLOY] Deployment daemon initialized\n");
    /* Returns - real daemon scheduling handled by tensor_sched */
}
