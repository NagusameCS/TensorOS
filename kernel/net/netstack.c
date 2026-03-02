/* =============================================================================
 * TensorOS - Network Stack Implementation
 * ARP responder, IPv4, UDP, ICMP echo, and simple HTTP inference server
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/drivers/net/virtio_net.h"
#include "kernel/net/netstack.h"

/* =============================================================================
 * State
 * =============================================================================*/

static net_config_t net_cfg;
static arp_entry_t arp_cache[ARP_CACHE_SIZE];

/* Statistics */
static uint64_t stat_rx_frames;
static uint64_t stat_tx_frames;
static uint64_t stat_arp_req;
static uint64_t stat_arp_rep;
static uint64_t stat_ip_rx;
static uint64_t stat_udp_rx;
static uint64_t stat_icmp_rx;
static uint64_t stat_http_req;

/* UDP port handlers */
#define MAX_UDP_HANDLERS 16
static struct {
    uint16_t     port;
    udp_handler_t handler;
} udp_handlers[MAX_UDP_HANDLERS];
static int n_udp_handlers;

/* TX frame buffer */
static uint8_t tx_frame[2048] __attribute__((aligned(16)));

/* =============================================================================
 * IP checksum
 * =============================================================================*/

static uint16_t ip_checksum(const void *data, uint32_t len)
{
    const uint16_t *p = (const uint16_t *)data;
    uint32_t sum = 0;
    while (len > 1) {
        sum += *p++;
        len -= 2;
    }
    if (len == 1) sum += *(const uint8_t *)p;
    sum = (sum >> 16) + (sum & 0xFFFF);
    sum += (sum >> 16);
    return (uint16_t)(~sum);
}

/* =============================================================================
 * Byte comparison helpers
 * =============================================================================*/

static int ip_eq(const uint8_t a[4], const uint8_t b[4])
{
    return a[0]==b[0] && a[1]==b[1] && a[2]==b[2] && a[3]==b[3];
}

__attribute__((unused))
static int mac_eq(const uint8_t a[6], const uint8_t b[6])
{
    return a[0]==b[0] && a[1]==b[1] && a[2]==b[2] &&
           a[3]==b[3] && a[4]==b[4] && a[5]==b[5];
}

static const uint8_t broadcast_mac[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
__attribute__((unused))
static const uint8_t zero_mac[6] = {0,0,0,0,0,0};

/* =============================================================================
 * ARP cache
 * =============================================================================*/

static void arp_cache_add(const uint8_t ip[4], const uint8_t mac[6])
{
    /* Check if already exists */
    for (int i = 0; i < ARP_CACHE_SIZE; i++) {
        if (arp_cache[i].valid && ip_eq(arp_cache[i].ip, ip)) {
            kmemcpy(arp_cache[i].mac, mac, 6);
            return;
        }
    }
    /* Find empty slot */
    for (int i = 0; i < ARP_CACHE_SIZE; i++) {
        if (!arp_cache[i].valid) {
            kmemcpy(arp_cache[i].ip, ip, 4);
            kmemcpy(arp_cache[i].mac, mac, 6);
            arp_cache[i].valid = 1;
            return;
        }
    }
    /* Cache full — overwrite first entry */
    kmemcpy(arp_cache[0].ip, ip, 4);
    kmemcpy(arp_cache[0].mac, mac, 6);
}

static const uint8_t *arp_cache_lookup(const uint8_t ip[4])
{
    for (int i = 0; i < ARP_CACHE_SIZE; i++) {
        if (arp_cache[i].valid && ip_eq(arp_cache[i].ip, ip))
            return arp_cache[i].mac;
    }
    return NULL;
}

/* =============================================================================
 * Send Ethernet frame
 * =============================================================================*/

static int send_eth(const uint8_t dst_mac[6], uint16_t ethertype,
                    const void *payload, uint32_t payload_len)
{
    if (payload_len + 14 > sizeof(tx_frame)) return -1;

    struct eth_hdr *eth = (struct eth_hdr *)tx_frame;
    kmemcpy(eth->dst, dst_mac, 6);
    kmemcpy(eth->src, net_cfg.mac, 6);
    eth->ethertype = htons(ethertype);
    kmemcpy(tx_frame + 14, payload, payload_len);

    stat_tx_frames++;
    return virtio_net_send(tx_frame, 14 + payload_len);
}

/* =============================================================================
 * ARP handler
 * =============================================================================*/

static void handle_arp(const uint8_t *frame, uint32_t len)
{
    if (len < 14 + sizeof(struct arp_hdr)) return;
    const struct arp_hdr *arp = (const struct arp_hdr *)(frame + 14);

    uint16_t op = ntohs(arp->opcode);

    /* Always learn sender */
    arp_cache_add(arp->sender_ip, arp->sender_mac);

    if (op == 1) { /* ARP Request */
        stat_arp_req++;
        /* Is this for our IP? */
        if (!ip_eq(arp->target_ip, net_cfg.ip)) return;

        /* Build ARP reply */
        struct arp_hdr reply;
        reply.hw_type = htons(1);
        reply.proto_type = htons(0x0800);
        reply.hw_len = 6;
        reply.proto_len = 4;
        reply.opcode = htons(2); /* Reply */
        kmemcpy(reply.sender_mac, net_cfg.mac, 6);
        kmemcpy(reply.sender_ip, net_cfg.ip, 4);
        kmemcpy(reply.target_mac, arp->sender_mac, 6);
        kmemcpy(reply.target_ip, arp->sender_ip, 4);

        send_eth(arp->sender_mac, ETH_TYPE_ARP, &reply, sizeof(reply));
        stat_arp_rep++;
    }
    else if (op == 2) { /* ARP Reply — already cached above */
        stat_arp_rep++;
    }
}

/* =============================================================================
 * ICMP handler (ping reply)
 * =============================================================================*/

static void handle_icmp(const uint8_t *ip_pkt, uint32_t ip_len,
                        const struct ip_hdr *iph)
{
    uint32_t hdr_len = (iph->version_ihl & 0x0F) * 4;
    if (ip_len < hdr_len + sizeof(struct icmp_hdr)) return;

    const struct icmp_hdr *icmp = (const struct icmp_hdr *)(ip_pkt + hdr_len);
    stat_icmp_rx++;

    if (icmp->type == 8 && icmp->code == 0) {
        /* Echo request — build echo reply */
        uint32_t icmp_total = ip_len - hdr_len;
        uint8_t reply_buf[1500];
        if (icmp_total > sizeof(reply_buf)) return;

        kmemcpy(reply_buf, icmp, icmp_total);
        struct icmp_hdr *rep = (struct icmp_hdr *)reply_buf;
        rep->type = 0; /* Echo reply */
        rep->checksum = 0;
        rep->checksum = ip_checksum(reply_buf, icmp_total);

        netstack_send_ip(iph->src, IP_PROTO_ICMP, reply_buf, icmp_total);
    }
}

/* =============================================================================
 * UDP handler
 * =============================================================================*/

static void handle_udp(const uint8_t *ip_pkt, uint32_t ip_len,
                       const struct ip_hdr *iph)
{
    uint32_t hdr_len = (iph->version_ihl & 0x0F) * 4;
    if (ip_len < hdr_len + sizeof(struct udp_hdr)) return;

    const struct udp_hdr *udp = (const struct udp_hdr *)(ip_pkt + hdr_len);
    uint16_t dst_port = ntohs(udp->dst_port);
    uint16_t src_port = ntohs(udp->src_port);
    uint32_t udp_data_len = ntohs(udp->length) - sizeof(struct udp_hdr);
    const uint8_t *udp_data = (const uint8_t *)udp + sizeof(struct udp_hdr);

    stat_udp_rx++;

    /* Dispatch to registered handler */
    for (int i = 0; i < n_udp_handlers; i++) {
        if (udp_handlers[i].port == dst_port && udp_handlers[i].handler) {
            udp_handlers[i].handler(iph->src, src_port, udp_data, udp_data_len);
            return;
        }
    }
}

/* =============================================================================
 * IP handler
 * =============================================================================*/

static void handle_ip(const uint8_t *frame, uint32_t len)
{
    if (len < 14 + sizeof(struct ip_hdr)) return;
    const struct ip_hdr *iph = (const struct ip_hdr *)(frame + 14);

    /* Verify it's IPv4 */
    if ((iph->version_ihl >> 4) != 4) return;

    /* Verify it's for us (or broadcast) */
    uint8_t bcast[4] = {255,255,255,255};
    if (!ip_eq(iph->dst, net_cfg.ip) && !ip_eq(iph->dst, bcast)) return;

    uint32_t ip_total = ntohs(iph->total_len);
    const uint8_t *ip_pkt = frame + 14;
    stat_ip_rx++;

    switch (iph->proto) {
    case IP_PROTO_ICMP:
        handle_icmp(ip_pkt, ip_total, iph);
        break;
    case IP_PROTO_UDP:
        handle_udp(ip_pkt, ip_total, iph);
        break;
    default:
        break;
    }
}

/* =============================================================================
 * Public API
 * =============================================================================*/

void netstack_init(const uint8_t ip[4], const uint8_t netmask[4],
                   const uint8_t gateway[4])
{
    kmemset(&net_cfg, 0, sizeof(net_cfg));
    kmemcpy(net_cfg.ip, ip, 4);
    kmemcpy(net_cfg.netmask, netmask, 4);
    kmemcpy(net_cfg.gateway, gateway, 4);
    net_cfg.http_port = 8080;

    virtio_net_dev_t *dev = virtio_net_get_dev();
    if (dev && dev->initialized) {
        kmemcpy(net_cfg.mac, dev->mac, 6);
    }

    kmemset(arp_cache, 0, sizeof(arp_cache));
    n_udp_handlers = 0;

    net_cfg.configured = 1;

    kprintf("[NET] Stack configured: %u.%u.%u.%u/%u.%u.%u.%u gw %u.%u.%u.%u\n",
            ip[0], ip[1], ip[2], ip[3],
            netmask[0], netmask[1], netmask[2], netmask[3],
            gateway[0], gateway[1], gateway[2], gateway[3]);
}

void netstack_rx(const uint8_t *frame, uint32_t len)
{
    if (len < 14) return;
    stat_rx_frames++;

    const struct eth_hdr *eth = (const struct eth_hdr *)frame;
    uint16_t ethertype = ntohs(eth->ethertype);

    switch (ethertype) {
    case ETH_TYPE_ARP:
        handle_arp(frame, len);
        break;
    case ETH_TYPE_IP:
        handle_ip(frame, len);
        break;
    default:
        break;
    }
}

int netstack_send_ip(const uint8_t dst_ip[4], uint8_t proto,
                     const void *data, uint32_t len)
{
    if (!net_cfg.configured) return -1;
    if (len + sizeof(struct ip_hdr) > 1500) return -2;

    /* Build IP packet */
    uint8_t pkt[1500];
    struct ip_hdr *iph = (struct ip_hdr *)pkt;
    iph->version_ihl = 0x45;  /* IPv4, 20-byte header */
    iph->tos = 0;
    iph->total_len = htons(sizeof(struct ip_hdr) + len);
    iph->id = 0;
    iph->flags_frag = 0;
    iph->ttl = 64;
    iph->proto = proto;
    iph->checksum = 0;
    kmemcpy(iph->src, net_cfg.ip, 4);
    kmemcpy(iph->dst, dst_ip, 4);
    iph->checksum = ip_checksum(iph, sizeof(struct ip_hdr));

    kmemcpy(pkt + sizeof(struct ip_hdr), data, len);

    /* Resolve dest MAC via ARP cache */
    const uint8_t *dst_mac = arp_cache_lookup(dst_ip);
    if (!dst_mac) {
        /* Send ARP request (lazy — just use broadcast) */
        dst_mac = broadcast_mac;
    }

    return send_eth(dst_mac, ETH_TYPE_IP, pkt, sizeof(struct ip_hdr) + len);
}

int netstack_send_udp(const uint8_t dst_ip[4], uint16_t src_port,
                      uint16_t dst_port, const void *data, uint32_t len)
{
    if (len + sizeof(struct udp_hdr) > 1400) return -2;

    uint8_t udp_pkt[1500];
    struct udp_hdr *udp = (struct udp_hdr *)udp_pkt;
    udp->src_port = htons(src_port);
    udp->dst_port = htons(dst_port);
    udp->length = htons(sizeof(struct udp_hdr) + len);
    udp->checksum = 0; /* Optional for UDP over IPv4 */

    kmemcpy(udp_pkt + sizeof(struct udp_hdr), data, len);

    return netstack_send_ip(dst_ip, IP_PROTO_UDP, udp_pkt,
                            sizeof(struct udp_hdr) + len);
}

void netstack_register_udp(uint16_t port, udp_handler_t handler)
{
    if (n_udp_handlers >= MAX_UDP_HANDLERS) return;
    udp_handlers[n_udp_handlers].port = port;
    udp_handlers[n_udp_handlers].handler = handler;
    n_udp_handlers++;
    kprintf("[NET] Registered UDP handler on port %u\n", port);
}

/* =============================================================================
 * Simple HTTP inference server (over UDP for simplicity)
 * Real HTTP would need TCP; this is a lightweight REST-like API over UDP
 * for maximum latency & simplicity.
 *
 * Protocol:
 *   Request:  "INFER <model_name> <input_json>\n"
 *   Response: "OK <output_json>\n"  or  "ERR <message>\n"
 *
 * Also responds to:
 *   "PING\n"  -> "PONG TensorOS v0.1\n"
 *   "INFO\n"  -> JSON system info
 * =============================================================================*/

static void inference_udp_handler(const uint8_t src_ip[4], uint16_t src_port,
                                  const uint8_t *data, uint32_t len)
{
    stat_http_req++;

    /* Simple command parser */
    if (len >= 4 && data[0] == 'P' && data[1] == 'I' && data[2] == 'N' && data[3] == 'G') {
        const char *resp = "PONG TensorOS v0.1.0 Neuron\n";
        netstack_send_udp(src_ip, net_cfg.http_port, src_port,
                          resp, kstrlen(resp));
        return;
    }

    if (len >= 4 && data[0] == 'I' && data[1] == 'N' && data[2] == 'F' && data[3] == 'O') {
        /* Send system info */
        char info[512];
        int pos = 0;
        pos += kprintf_to_buf(info + pos, sizeof(info) - pos,
            "{\"os\":\"TensorOS\",\"version\":\"0.1.0\","
            "\"cpus\":%u,\"gpus\":%u,"
            "\"features\":[\"SSE2\",\"GEMM\",\"Q4_0\",\"KV-cache\",\"SNE\","
            "\"Winograd\",\"Arena\",\"GGUF\"]}\n",
            kstate.cpu_count, kstate.gpu_count);
        netstack_send_udp(src_ip, net_cfg.http_port, src_port, info, pos);
        return;
    }

    if (len >= 5 && data[0] == 'I' && data[1] == 'N' && data[2] == 'F' &&
        data[3] == 'E' && data[4] == 'R') {
        /* Inference request — placeholder */
        const char *resp = "ERR no model loaded (use GGUF loader)\n";
        netstack_send_udp(src_ip, net_cfg.http_port, src_port,
                          resp, kstrlen(resp));
        return;
    }

    /* Unknown command */
    const char *resp = "ERR unknown command. Use: PING, INFO, INFER\n";
    netstack_send_udp(src_ip, net_cfg.http_port, src_port,
                      resp, kstrlen(resp));
}

void netstack_start_http_server(void)
{
    netstack_register_udp(net_cfg.http_port, inference_udp_handler);
    kprintf("[NET] Inference server listening on UDP port %u\n", net_cfg.http_port);
}

void netstack_print_stats(void)
{
    kprintf("[NET] Stats: RX=%lu TX=%lu ARP=%lu/%lu IP=%lu UDP=%lu ICMP=%lu HTTP=%lu\n",
            stat_rx_frames, stat_tx_frames, stat_arp_req, stat_arp_rep,
            stat_ip_rx, stat_udp_rx, stat_icmp_rx, stat_http_req);
}

/* =============================================================================
 * kprintf_to_buf - format string to buffer (subset of kprintf)
 * Simple implementation supporting %s, %u, %d
 * =============================================================================*/

int kprintf_to_buf(char *buf, int buflen, const char *fmt, ...)
{
    /* Minimal snprintf-like implementation */
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);

    int pos = 0;
    while (*fmt && pos < buflen - 1) {
        if (*fmt != '%') {
            buf[pos++] = *fmt++;
            continue;
        }
        fmt++;
        switch (*fmt) {
        case 's': {
            const char *s = __builtin_va_arg(ap, const char *);
            if (!s) s = "(null)";
            while (*s && pos < buflen - 1) buf[pos++] = *s++;
            break;
        }
        case 'u': {
            uint32_t v = __builtin_va_arg(ap, uint32_t);
            char tmp[12];
            int ti = 0;
            if (v == 0) { tmp[ti++] = '0'; }
            else { while (v > 0) { tmp[ti++] = '0' + (v % 10); v /= 10; } }
            for (int i = ti - 1; i >= 0 && pos < buflen - 1; i--)
                buf[pos++] = tmp[i];
            break;
        }
        case 'd': {
            int32_t v = __builtin_va_arg(ap, int32_t);
            if (v < 0) { buf[pos++] = '-'; v = -v; }
            char tmp[12];
            int ti = 0;
            if (v == 0) { tmp[ti++] = '0'; }
            else { while (v > 0) { tmp[ti++] = '0' + (v % 10); v /= 10; } }
            for (int i = ti - 1; i >= 0 && pos < buflen - 1; i--)
                buf[pos++] = tmp[i];
            break;
        }
        case '%':
            buf[pos++] = '%';
            break;
        default:
            buf[pos++] = '%';
            if (pos < buflen - 1) buf[pos++] = *fmt;
            break;
        }
        fmt++;
    }

    __builtin_va_end(ap);
    buf[pos] = '\0';
    return pos;
}
