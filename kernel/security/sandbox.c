/* =============================================================================
 * TensorOS - Security Sandbox Implementation
 * =============================================================================*/

#include "kernel/security/sandbox.h"

static sandbox_t sandboxes[SANDBOX_MAX];
static uint32_t sandbox_count = 0;
static uint64_t next_sandbox_id = 1;

void sandbox_init(void)
{
    kmemset(sandboxes, 0, sizeof(sandboxes));
    sandbox_count = 0;
}

sandbox_t *sandbox_create(const char *name, sandbox_policy_t policy)
{
    if (sandbox_count >= SANDBOX_MAX) return NULL;

    sandbox_t *sb = &sandboxes[sandbox_count++];
    kmemset(sb, 0, sizeof(*sb));

    sb->id = next_sandbox_id++;
    sb->active = false;
    sb->policy = policy;
    sb->audit_enabled = true;

    for (int i = 0; i < 63 && name[i]; i++)
        sb->name[i] = name[i];

    /* Set default permissions based on policy */
    switch (policy) {
    case SANDBOX_POLICY_STRICT:
        sb->permissions = SANDBOX_PERM_READ_MODEL | SANDBOX_PERM_GPU_ACCESS;
        break;
    case SANDBOX_POLICY_STANDARD:
        sb->permissions = SANDBOX_PERM_READ_MODEL | SANDBOX_PERM_WRITE_MODEL |
                          SANDBOX_PERM_GPU_ACCESS | SANDBOX_PERM_FS_READ |
                          SANDBOX_PERM_FS_WRITE | SANDBOX_PERM_GIT;
        break;
    case SANDBOX_POLICY_PERMISSIVE:
        sb->permissions = 0xFFFFFFFF; /* All permissions */
        break;
    }

    /* Default resource limits */
    sb->mem_limit = 8ULL * 1024 * 1024 * 1024;  /* 8GB */
    sb->gpu_mem_limit = 4ULL * 1024 * 1024 * 1024;  /* 4GB */
    sb->tensor_ops_limit = 0; /* Unlimited */

    kprintf_debug("[SANDBOX] Created sandbox %lu '%s' policy=%d\n",
                  sb->id, sb->name, policy);
    return sb;
}

int sandbox_destroy(uint64_t sandbox_id)
{
    for (uint32_t i = 0; i < sandbox_count; i++) {
        if (sandboxes[i].id == sandbox_id) {
            sandboxes[i].active = false;
            sandboxes[i].id = 0;
            return 0;
        }
    }
    return -1;
}

int sandbox_activate(uint64_t sandbox_id)
{
    for (uint32_t i = 0; i < sandbox_count; i++) {
        if (sandboxes[i].id == sandbox_id) {
            sandboxes[i].active = true;
            sandbox_audit_log(sandbox_id, AUDIT_TENSOR_ALLOC, "Sandbox activated");
            return 0;
        }
    }
    return -1;
}

/* =============================================================================
 * Permission Checking
 * =============================================================================*/

static sandbox_t *find_sandbox(uint64_t id)
{
    for (uint32_t i = 0; i < sandbox_count; i++) {
        if (sandboxes[i].id == id)
            return &sandboxes[i];
    }
    return NULL;
}

int sandbox_grant_permission(uint64_t sandbox_id, uint32_t permissions)
{
    sandbox_t *sb = find_sandbox(sandbox_id);
    if (!sb) return -1;
    sb->permissions |= permissions;
    return 0;
}

int sandbox_revoke_permission(uint64_t sandbox_id, uint32_t permissions)
{
    sandbox_t *sb = find_sandbox(sandbox_id);
    if (!sb) return -1;
    sb->permissions &= ~permissions;
    return 0;
}

bool sandbox_check_permission(uint64_t sandbox_id, uint32_t permission)
{
    sandbox_t *sb = find_sandbox(sandbox_id);
    if (!sb) return false;
    return (sb->permissions & permission) != 0;
}

/* =============================================================================
 * Enforcement Functions
 * Called by kernel subsystems before performing operations
 * =============================================================================*/

bool sandbox_allow_tensor_op(uint64_t sandbox_id, tensor_desc_t *tensor)
{
    sandbox_t *sb = find_sandbox(sandbox_id);
    if (!sb || !sb->active) return true; /* No sandbox = allowed */

    /* Check GPU permission */
    if ((tensor->flags & TENSOR_FLAG_DEVICE_MEM) &&
        !(sb->permissions & SANDBOX_PERM_GPU_ACCESS)) {
        sandbox_audit_log(sandbox_id, AUDIT_PERMISSION_DENIED,
                          "GPU access denied");
        return false;
    }

    /* Check memory limits */
    if (sb->mem_allocated + tensor->size_bytes > sb->mem_limit) {
        sandbox_audit_log(sandbox_id, AUDIT_PERMISSION_DENIED,
                          "Memory limit exceeded");
        return false;
    }

    /* Check ops limit */
    if (sb->tensor_ops_limit > 0 && sb->tensor_ops_count >= sb->tensor_ops_limit) {
        sandbox_audit_log(sandbox_id, AUDIT_PERMISSION_DENIED,
                          "Tensor ops limit exceeded");
        return false;
    }

    sb->tensor_ops_count++;
    sb->mem_allocated += tensor->size_bytes;
    return true;
}

bool sandbox_allow_network(uint64_t sandbox_id, const char *host, uint16_t port)
{
    sandbox_t *sb = find_sandbox(sandbox_id);
    if (!sb || !sb->active) return true;

    /* Check network permissions */
    if (!(sb->permissions & SANDBOX_PERM_NETWORK_LOCAL) &&
        !(sb->permissions & SANDBOX_PERM_NETWORK_REMOTE)) {
        sandbox_audit_log(sandbox_id, AUDIT_PERMISSION_DENIED,
                          "Network access denied");
        return false;
    }

    /* If only local, check host */
    if ((sb->permissions & SANDBOX_PERM_NETWORK_LOCAL) &&
        !(sb->permissions & SANDBOX_PERM_NETWORK_REMOTE)) {
        /* Only allow localhost/127.0.0.1 */
        if (kstrcmp(host, "localhost") != 0 && kstrcmp(host, "127.0.0.1") != 0) {
            sandbox_audit_log(sandbox_id, AUDIT_PERMISSION_DENIED,
                              "Remote network access denied");
            return false;
        }
    }

    sandbox_audit_log(sandbox_id, AUDIT_NETWORK_ACCESS, host);
    return true;
}

bool sandbox_allow_fs_access(uint64_t sandbox_id, const char *path, bool write)
{
    sandbox_t *sb = find_sandbox(sandbox_id);
    if (!sb || !sb->active) return true;

    if (write && !(sb->permissions & SANDBOX_PERM_FS_WRITE)) {
        sandbox_audit_log(sandbox_id, AUDIT_PERMISSION_DENIED,
                          "Filesystem write denied");
        return false;
    }

    if (!write && !(sb->permissions & SANDBOX_PERM_FS_READ)) {
        sandbox_audit_log(sandbox_id, AUDIT_PERMISSION_DENIED,
                          "Filesystem read denied");
        return false;
    }

    return true;
}

/* =============================================================================
 * Audit Logging
 * =============================================================================*/

void sandbox_audit_log(uint64_t sandbox_id, audit_event_type_t type,
                        const char *description)
{
    sandbox_t *sb = find_sandbox(sandbox_id);
    if (!sb || !sb->audit_enabled) return;

    audit_entry_t *entry = &sb->audit_log[sb->audit_head % 256];
    kmemset(entry, 0, sizeof(*entry));

    entry->type = type;
    entry->timestamp = kstate.uptime_ticks;
    entry->sandbox_id = sandbox_id;

    for (int i = 0; i < 255 && description[i]; i++)
        entry->description[i] = description[i];

    sb->audit_head++;
    if (sb->audit_count < 256) sb->audit_count++;
}

int sandbox_audit_dump(uint64_t sandbox_id, audit_entry_t *entries,
                        uint32_t max, uint32_t *count)
{
    sandbox_t *sb = find_sandbox(sandbox_id);
    if (!sb) return -1;

    uint32_t to_copy = sb->audit_count < max ? sb->audit_count : max;
    uint32_t start = (sb->audit_head - sb->audit_count + 256) % 256;

    for (uint32_t i = 0; i < to_copy; i++) {
        entries[i] = sb->audit_log[(start + i) % 256];
    }

    if (count) *count = to_copy;
    return 0;
}

/* =============================================================================
 * Reproducibility
 * =============================================================================*/

int sandbox_set_deterministic(uint64_t sandbox_id, bool deterministic,
                                uint64_t seed)
{
    sandbox_t *sb = find_sandbox(sandbox_id);
    if (!sb) return -1;

    sb->deterministic_mode = deterministic;
    sb->random_seed = seed;

    if (deterministic) {
        kprintf_debug("[SANDBOX] Sandbox %lu set to deterministic mode (seed=%lu)\n",
                      sandbox_id, seed);
    }
    return 0;
}

/* Resource limits */
int sandbox_set_mem_limit(uint64_t sandbox_id, uint64_t bytes)
{
    sandbox_t *sb = find_sandbox(sandbox_id);
    if (!sb) return -1;
    sb->mem_limit = bytes;
    return 0;
}

int sandbox_set_gpu_limit(uint64_t sandbox_id, uint64_t bytes)
{
    sandbox_t *sb = find_sandbox(sandbox_id);
    if (!sb) return -1;
    sb->gpu_mem_limit = bytes;
    return 0;
}

int sandbox_set_ops_limit(uint64_t sandbox_id, uint64_t max_ops)
{
    sandbox_t *sb = find_sandbox(sandbox_id);
    if (!sb) return -1;
    sb->tensor_ops_limit = max_ops;
    return 0;
}
