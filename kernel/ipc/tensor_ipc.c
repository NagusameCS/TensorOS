/* =============================================================================
 * TensorOS - Tensor IPC Implementation
 * =============================================================================*/

#include "kernel/ipc/tensor_ipc.h"
#include "kernel/mm/tensor_mm.h"

static ipc_channel_t channels[IPC_MAX_CHANNELS];
static uint32_t channel_count = 0;
static uint32_t next_channel_id = 1;

void tensor_ipc_init(void)
{
    kmemset(channels, 0, sizeof(channels));
    channel_count = 0;
}

int ipc_channel_create(uint64_t sender_meu, uint64_t receiver_meu,
                         ipc_chan_type_t type, uint64_t buffer_size)
{
    if (channel_count >= IPC_MAX_CHANNELS) return -1;

    ipc_channel_t *ch = &channels[channel_count++];
    kmemset(ch, 0, sizeof(*ch));

    ch->id = next_channel_id++;
    ch->type = type;
    ch->sender_meu = sender_meu;
    ch->receiver_meu = receiver_meu;
    ch->buffer_size = buffer_size;
    ch->active = true;

    /* Allocate shared buffer for zero-copy transfers */
    ch->shared_buffer = tensor_alloc_shared(buffer_size);

    kprintf_debug("[IPC] Channel %d created: MEU %lu -> MEU %lu (%lu KB buffer)\n",
                  ch->id, sender_meu, receiver_meu, buffer_size / 1024);
    return ch->id;
}

int ipc_channel_destroy(uint32_t channel_id)
{
    for (uint32_t i = 0; i < channel_count; i++) {
        if (channels[i].id == channel_id) {
            if (channels[i].shared_buffer)
                tensor_free(channels[i].shared_buffer);
            channels[i].active = false;
            return 0;
        }
    }
    return -1;
}

int ipc_send_tensor(uint32_t channel_id, const tensor_desc_t *tensor,
                     const void *data)
{
    for (uint32_t i = 0; i < channel_count; i++) {
        if (channels[i].id == channel_id && channels[i].active) {
            /* Copy tensor data to shared buffer */
            uint64_t size = tensor->size_bytes;
            if (size > channels[i].buffer_size) return -1;

            kmemcpy(channels[i].shared_buffer, data, size);
            channels[i].messages_sent++;
            channels[i].bytes_transferred += size;
            return 0;
        }
    }
    return -1;
}

void *ipc_send_tensor_zerocopy(uint32_t channel_id, const tensor_desc_t *tensor)
{
    for (uint32_t i = 0; i < channel_count; i++) {
        if (channels[i].id == channel_id && channels[i].active) {
            channels[i].messages_sent++;
            return channels[i].shared_buffer;
        }
    }
    return NULL;
}

const void *ipc_recv_tensor_zerocopy(uint32_t channel_id, tensor_desc_t *tensor)
{
    for (uint32_t i = 0; i < channel_count; i++) {
        if (channels[i].id == channel_id && channels[i].active) {
            return channels[i].shared_buffer;
        }
    }
    return NULL;
}

int ipc_pipeline_create(uint64_t *meu_ids, uint32_t count)
{
    /* Create a chain of channels: MEU[0] -> MEU[1] -> ... -> MEU[n-1] */
    for (uint32_t i = 0; i < count - 1; i++) {
        int ch = ipc_channel_create(meu_ids[i], meu_ids[i + 1],
                                     IPC_CHAN_PIPE, 16 * 1024 * 1024);
        if (ch < 0) return -1;
    }
    return 0;
}

/* External declarations */
extern void *tensor_alloc_shared(uint64_t size);
extern void tensor_free(void *ptr);
