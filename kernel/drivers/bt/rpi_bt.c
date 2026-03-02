/* =============================================================================
 * TensorOS — Bare-Metal Bluetooth SPP Console for Raspberry Pi 4
 *
 * Implements a complete Bluetooth Serial Port Profile (SPP) stack:
 *   PL011 UART (UART0) → HCI H4 → L2CAP → SDP + RFCOMM
 *
 * Hardware: BCM43455 (CYW43455) combo chip on RPi4
 * Transport: PL011 UART via GPIO 30-33 (ALT3) at 115200 baud
 *
 * Mini UART (UART1) is used for debug console on GPIO 14/15,
 * so we MUST use the separate PL011 (UART0) peripheral for BT.
 *
 * Usage: pair to "TensorOS" from Windows/Mac/Linux → COM port appears →
 *        open with PuTTY or any serial terminal at any baud rate.
 * =============================================================================*/

#include "kernel/core/kernel.h"

#if defined(__aarch64__)

/* =============================================================================
 * BCM2711 PL011 UART (UART0) — Physical transport to BCM43455
 *
 * The BCM43455 BT chip is physically wired to GPIO 30-33 on the RPi4.
 * PL011 registers (UART_DR, UART_FR, etc.) are defined in arm64_hal.h
 * at BCM2711_UART0_BASE = 0xFE201000.
 *
 * GPIO mapping (ALT3 = 0b111):
 *   GPIO 30 = CTS0    GPIO 31 = RTS0
 *   GPIO 32 = TXD0    GPIO 33 = RXD0
 * =============================================================================*/

/* GPIO registers for BT UART pins (GPIO 30-33) */
#define BT_GPIO_BASE        0xFE200000UL
#define BT_GPFSEL3          (BT_GPIO_BASE + 0x0C)   /* GPIO 30-39 */
#define BT_GPIO_PUP_PDN1    (BT_GPIO_BASE + 0xE8)   /* GPIO 16-31 */
#define BT_GPIO_PUP_PDN2    (BT_GPIO_BASE + 0xEC)   /* GPIO 32-47 */

/* PL011 UART clock rate (set via mailbox in mu_init) */
#define BT_UART_CLOCK       48000000U

/* ---------- Ring buffers ---------- */
#define BT_TX_SIZE  2048
#define BT_RX_SIZE  256

static char     bt_txbuf[BT_TX_SIZE];
static volatile int bt_tx_head = 0;
static volatile int bt_tx_tail = 0;

static char     bt_rxbuf[BT_RX_SIZE];
static volatile int bt_rx_head = 0;
static volatile int bt_rx_tail = 0;

/* ---------- PL011 UART low-level (UART_* macros from arm64_hal.h) ---------- */
static void mu_putc(uint8_t c)
{
    while (mmio_read(UART_FR) & UART_FR_TXFF) {}    /* TX FIFO full? */
    mmio_write(UART_DR, c);
}

static int mu_has_data(void)
{
    return !(mmio_read(UART_FR) & UART_FR_RXFE);    /* RX FIFO not empty */
}

static uint8_t mu_getc(void)
{
    while (!mu_has_data()) {}
    return (uint8_t)(mmio_read(UART_DR) & 0xFF);
}

/* Read with timeout (ms).  Returns -1 on timeout. */
static int mu_getc_timeout(uint32_t ms)
{
    uint64_t freq = arm_timer_freq();
    uint64_t deadline = arm_timer_count() + (freq * ms / 1000);
    while (arm_timer_count() < deadline) {
        if (mu_has_data()) return (int)mu_getc();
    }
    return -1;
}

static void mu_write(const uint8_t *buf, int len)
{
    for (int i = 0; i < len; i++) mu_putc(buf[i]);
}

static void mu_init(uint32_t baud)
{
    /* 1. Set UART clock to 48 MHz via VideoCore mailbox */
    {
        volatile uint32_t __attribute__((aligned(16))) mb[9];
        mb[0] = 9 * 4;          /* buffer size */
        mb[1] = 0;              /* request */
        mb[2] = 0x00038002;     /* SET_CLOCK_RATE */
        mb[3] = 12;             /* value buffer size */
        mb[4] = 12;             /* request size */
        mb[5] = 0x00000002;     /* clock ID: UART */
        mb[6] = BT_UART_CLOCK;  /* 48 MHz */
        mb[7] = 0;              /* skip turbo */
        mb[8] = 0;              /* end tag */
        mbox_call(8, mb);
    }

    /* 2. Setup GPIO 30-33 as ALT3 for PL011 (CTS0/RTS0/TXD0/RXD0) */
    uint32_t sel = mmio_read(BT_GPFSEL3);
    sel &= ~0xFFF;                          /* clear bits [11:0] for GPIO 30-33 */
    sel |= (7<<0)|(7<<3)|(7<<6)|(7<<9);    /* ALT3 = 0b111 for each */
    mmio_write(BT_GPFSEL3, sel);

    /* No pull-up/down on GPIO 30-33 */
    uint32_t pud1 = mmio_read(BT_GPIO_PUP_PDN1);
    pud1 &= ~((3u<<28)|(3u<<30));           /* GPIO 30, 31 */
    mmio_write(BT_GPIO_PUP_PDN1, pud1);
    uint32_t pud2 = mmio_read(BT_GPIO_PUP_PDN2);
    pud2 &= ~((3u<<0)|(3u<<2));             /* GPIO 32, 33 */
    mmio_write(BT_GPIO_PUP_PDN2, pud2);

    for (volatile int i = 0; i < 150; i++) {}  /* GPIO settle time */

    /* 3. Configure PL011 (UART0 at 0xFE201000) */
    mmio_write(UART_CR, 0);                    /* disable UART during config */
    mmio_write(UART_ICR, 0x7FF);               /* clear all pending interrupts */
    mmio_write(UART_IMSC, 0);                  /* mask all interrupts */

    /* Baud rate: UART_CLK / (16 * baud)
     * 48 MHz / (16 * 115200) = 26.0417  →  IBRD=26, FBRD=3 */
    uint32_t div_int  = BT_UART_CLOCK / (16 * baud);
    uint32_t div_rem  = BT_UART_CLOCK % (16 * baud);
    uint32_t div_frac = ((div_rem * 64) + (8 * baud)) / (16 * baud);
    mmio_write(UART_IBRD, div_int);
    mmio_write(UART_FBRD, div_frac);

    /* Line control: 8 data bits, FIFO enabled, no parity, 1 stop bit */
    mmio_write(UART_LCRH, (3 << 5) | (1 << 4));   /* WLEN=8, FEN=1 */

    /* Enable UART with TX, RX, and hardware flow control (CTS + RTS) */
    mmio_write(UART_CR, (1 << 0)  |    /* UARTEN */
                        (1 << 8)  |    /* TXE */
                        (1 << 9)  |    /* RXE */
                        (1 << 14) |    /* CTSEn */
                        (1 << 15));    /* RTSEn */

    dmb();
}

/* =============================================================================
 * HCI H4 Transport Layer
 * =============================================================================*/

/* Packet indicators */
#define HCI_CMD_PKT     0x01
#define HCI_ACL_PKT     0x02
#define HCI_EVT_PKT     0x04

/* HCI command opcodes (OGF | OCF) */
#define HCI_RESET                   0x0C03
#define HCI_READ_BD_ADDR            0x1009
#define HCI_WRITE_LOCAL_NAME        0x0C13
#define HCI_WRITE_CLASS_OF_DEVICE   0x0C24
#define HCI_WRITE_SCAN_ENABLE      0x0C1A
#define HCI_SET_EVENT_MASK          0x0C01
#define HCI_WRITE_SSP_MODE         0x0C56
#define HCI_WRITE_PAGE_TIMEOUT     0x0C18
#define HCI_ACCEPT_CONN_REQ        0x0409
#define HCI_REJECT_CONN_REQ        0x040A
#define HCI_LINK_KEY_NEG_REPLY     0x040C
#define HCI_PIN_CODE_REPLY         0x040D
#define HCI_IO_CAP_REQ_REPLY      0x042B
#define HCI_USER_CONFIRM_REPLY    0x042C
#define HCI_WRITE_AUTH_ENABLE      0x0C20

/* HCI event codes */
#define HCI_EVT_INQUIRY_COMPLETE     0x01
#define HCI_EVT_CONN_COMPLETE        0x03
#define HCI_EVT_CONN_REQUEST         0x04
#define HCI_EVT_DISCONN_COMPLETE     0x05
#define HCI_EVT_CMD_COMPLETE         0x0E
#define HCI_EVT_CMD_STATUS           0x0F
#define HCI_EVT_PIN_CODE_REQ        0x16
#define HCI_EVT_LINK_KEY_REQ        0x17
#define HCI_EVT_LINK_KEY_NOTIF      0x18
#define HCI_EVT_IO_CAP_REQ         0x31
#define HCI_EVT_IO_CAP_RESP        0x32
#define HCI_EVT_USER_CONFIRM_REQ   0x33
#define HCI_EVT_SSP_COMPLETE       0x36
#define HCI_EVT_NUM_COMP_PKTS      0x13

/* Connection state */
static uint16_t hci_conn_handle = 0;
static int      hci_connected   = 0;
static uint8_t  hci_remote_addr[6];

/* Receive buffer for HCI events/ACL */
#define HCI_BUF_SIZE 1024
static uint8_t hci_buf[HCI_BUF_SIZE];

/* ---------- Send HCI command ---------- */
static void hci_send_cmd(uint16_t opcode, const uint8_t *params, uint8_t plen)
{
    mu_putc(HCI_CMD_PKT);
    mu_putc(opcode & 0xFF);
    mu_putc(opcode >> 8);
    mu_putc(plen);
    for (int i = 0; i < plen; i++) mu_putc(params[i]);
}

static void hci_send_cmd0(uint16_t opcode)
{
    hci_send_cmd(opcode, (const uint8_t *)0, 0);
}

/* ---------- Read one HCI packet (blocking with timeout) ---------- */
/* Returns packet type, fills hci_buf with the payload, sets *out_len.
 * Returns 0 on timeout. */
static int hci_read_packet(int *out_len, uint32_t timeout_ms)
{
    int t = mu_getc_timeout(timeout_ms);
    if (t < 0) { *out_len = 0; return 0; }

    uint8_t type = (uint8_t)t;
    *out_len = 0;

    if (type == HCI_EVT_PKT) {
        /* Event: code(1) + plen(1) + params(plen) */
        int code = mu_getc_timeout(timeout_ms);
        int plen = mu_getc_timeout(timeout_ms);
        if (code < 0 || plen < 0) return 0;
        hci_buf[0] = (uint8_t)code;
        hci_buf[1] = (uint8_t)plen;
        for (int i = 0; i < plen && i < HCI_BUF_SIZE - 2; i++)
            hci_buf[2 + i] = (uint8_t)mu_getc_timeout(timeout_ms);
        *out_len = 2 + plen;
        return HCI_EVT_PKT;

    } else if (type == HCI_ACL_PKT) {
        /* ACL: handle(2) + dlen(2) + data(dlen) */
        uint8_t h0 = (uint8_t)mu_getc_timeout(timeout_ms);
        uint8_t h1 = (uint8_t)mu_getc_timeout(timeout_ms);
        uint8_t d0 = (uint8_t)mu_getc_timeout(timeout_ms);
        uint8_t d1 = (uint8_t)mu_getc_timeout(timeout_ms);
        uint16_t dlen = d0 | ((uint16_t)d1 << 8);
        hci_buf[0] = h0; hci_buf[1] = h1;
        hci_buf[2] = d0; hci_buf[3] = d1;
        for (int i = 0; i < dlen && i < HCI_BUF_SIZE - 4; i++)
            hci_buf[4 + i] = (uint8_t)mu_getc_timeout(timeout_ms);
        *out_len = 4 + (dlen < HCI_BUF_SIZE - 4 ? dlen : HCI_BUF_SIZE - 4);
        return HCI_ACL_PKT;
    }
    return 0;
}

/* Wait for a Command Complete event for a specific opcode. */
static int hci_wait_cmd_complete(uint16_t opcode, uint32_t timeout_ms)
{
    int len;
    uint64_t freq = arm_timer_freq();
    uint64_t deadline = arm_timer_count() + (freq * timeout_ms / 1000);

    while (arm_timer_count() < deadline) {
        int type = hci_read_packet(&len, 200);
        if (type == HCI_EVT_PKT && len >= 6 && hci_buf[0] == HCI_EVT_CMD_COMPLETE) {
            uint16_t op = hci_buf[3] | ((uint16_t)hci_buf[4] << 8);
            if (op == opcode) return hci_buf[5]; /* status */
        }
        if (type == HCI_EVT_PKT && len >= 5 && hci_buf[0] == HCI_EVT_CMD_STATUS) {
            uint16_t op = hci_buf[4] | ((uint16_t)hci_buf[5] << 8);
            if (op == opcode) return hci_buf[2]; /* status */
        }
    }
    return -1; /* timeout */
}

/* =============================================================================
 * Send HCI ACL data (L2CAP packet)
 * =============================================================================*/
static void hci_send_acl(uint16_t handle, uint16_t cid,
                         const uint8_t *data, uint16_t len)
{
    uint16_t hfl  = handle | 0x2000;    /* PB=10 (first auto-flush), BC=00 */
    uint16_t dlen = len + 4;            /* L2CAP header = 4 bytes */

    mu_putc(HCI_ACL_PKT);
    mu_putc(hfl & 0xFF); mu_putc(hfl >> 8);
    mu_putc(dlen & 0xFF); mu_putc(dlen >> 8);
    /* L2CAP header */
    mu_putc(len & 0xFF); mu_putc(len >> 8);
    mu_putc(cid & 0xFF); mu_putc(cid >> 8);
    /* L2CAP payload */
    mu_write(data, len);
}

/* =============================================================================
 * L2CAP — Logical Link Control and Adaptation Protocol
 * =============================================================================*/

/* L2CAP signaling codes */
#define L2CAP_CMD_REJECT        0x01
#define L2CAP_CONN_REQ          0x02
#define L2CAP_CONN_RSP          0x03
#define L2CAP_CONF_REQ          0x04
#define L2CAP_CONF_RSP          0x05
#define L2CAP_DISCONN_REQ       0x06
#define L2CAP_DISCONN_RSP       0x07
#define L2CAP_INFO_REQ          0x0A
#define L2CAP_INFO_RSP          0x0B

/* Well-known CIDs */
#define L2CAP_CID_SIGNALING     0x0001
#define L2CAP_CID_CONNLESS      0x0002

/* Well-known PSMs */
#define L2CAP_PSM_SDP           0x0001
#define L2CAP_PSM_RFCOMM        0x0003

/* Channel tracking (max 4 simultaneous channels) */
#define L2CAP_MAX_CHANNELS 4

typedef struct {
    uint16_t local_cid;
    uint16_t remote_cid;
    uint16_t psm;
    int      active;
} l2cap_channel_t;

static l2cap_channel_t l2cap_channels[L2CAP_MAX_CHANNELS];
static uint16_t l2cap_next_cid = 0x0040;

static l2cap_channel_t *l2cap_alloc(uint16_t psm, uint16_t remote_cid)
{
    for (int i = 0; i < L2CAP_MAX_CHANNELS; i++) {
        if (!l2cap_channels[i].active) {
            l2cap_channels[i].local_cid  = l2cap_next_cid++;
            l2cap_channels[i].remote_cid = remote_cid;
            l2cap_channels[i].psm        = psm;
            l2cap_channels[i].active     = 1;
            return &l2cap_channels[i];
        }
    }
    return (l2cap_channel_t *)0;
}

static l2cap_channel_t *l2cap_find_local(uint16_t cid)
{
    for (int i = 0; i < L2CAP_MAX_CHANNELS; i++)
        if (l2cap_channels[i].active && l2cap_channels[i].local_cid == cid)
            return &l2cap_channels[i];
    return (l2cap_channel_t *)0;
}

__attribute__((unused))
static l2cap_channel_t *l2cap_find_psm(uint16_t psm)
{
    for (int i = 0; i < L2CAP_MAX_CHANNELS; i++)
        if (l2cap_channels[i].active && l2cap_channels[i].psm == psm)
            return &l2cap_channels[i];
    return (l2cap_channel_t *)0;
}

static void l2cap_free(l2cap_channel_t *ch)
{
    if (ch) ch->active = 0;
}

/* Send L2CAP signaling command */
static void l2cap_send_sig(uint8_t code, uint8_t id,
                           const uint8_t *data, uint16_t len)
{
    uint8_t sig[256];
    sig[0] = code;
    sig[1] = id;
    sig[2] = len & 0xFF;
    sig[3] = len >> 8;
    for (int i = 0; i < len && i < 252; i++) sig[4+i] = data[i];
    hci_send_acl(hci_conn_handle, L2CAP_CID_SIGNALING, sig, 4 + len);
}

/* Handle L2CAP signaling packet */
static void l2cap_handle_signaling(const uint8_t *data, int len)
{
    if (len < 4) return;
    uint8_t  code = data[0];
    uint8_t  id   = data[1];
    uint16_t clen = data[2] | ((uint16_t)data[3] << 8);
    const uint8_t *params = data + 4;

    if (code == L2CAP_CONN_REQ && clen >= 4) {
        /* Connection Request: PSM(2) + source_CID(2) */
        uint16_t psm      = params[0] | ((uint16_t)params[1] << 8);
        uint16_t src_cid   = params[2] | ((uint16_t)params[3] << 8);

        if (psm == L2CAP_PSM_SDP || psm == L2CAP_PSM_RFCOMM) {
            l2cap_channel_t *ch = l2cap_alloc(psm, src_cid);
            if (ch) {
                /* Connection Response: dest(2)+src(2)+result(2)+status(2) */
                uint8_t rsp[8];
                rsp[0] = ch->local_cid & 0xFF; rsp[1] = ch->local_cid >> 8;
                rsp[2] = src_cid & 0xFF;       rsp[3] = src_cid >> 8;
                rsp[4] = 0; rsp[5] = 0;  /* result=success */
                rsp[6] = 0; rsp[7] = 0;  /* status=no info */
                l2cap_send_sig(L2CAP_CONN_RSP, id, rsp, 8);
                return;
            }
        }
        /* Reject: no resources */
        uint8_t rsp[8] = {0,0, 0,0, 0x04,0x00, 0,0}; /* result=refused */
        rsp[2] = src_cid & 0xFF; rsp[3] = src_cid >> 8;
        l2cap_send_sig(L2CAP_CONN_RSP, id, rsp, 8);

    } else if (code == L2CAP_CONF_REQ && clen >= 4) {
        /* Configuration Request: dest_CID(2) + flags(2) + options... */
        uint16_t dst_cid = params[0] | ((uint16_t)params[1] << 8);
        /* Accept with default config */
        uint8_t rsp[6];
        rsp[0] = dst_cid & 0xFF; rsp[1] = dst_cid >> 8; /* source_CID */
        rsp[2] = 0; rsp[3] = 0;  /* flags */
        rsp[4] = 0; rsp[5] = 0;  /* result=success */
        l2cap_send_sig(L2CAP_CONF_RSP, id, rsp, 6);

        /* Also send our own Config Request to complete handshake */
        l2cap_channel_t *ch = l2cap_find_local(dst_cid);
        if (ch) {
            uint8_t req[4];
            req[0] = ch->remote_cid & 0xFF; req[1] = ch->remote_cid >> 8;
            req[2] = 0; req[3] = 0; /* flags=no continuation */
            l2cap_send_sig(L2CAP_CONF_REQ, id + 1, req, 4);
        }

    } else if (code == L2CAP_CONF_RSP) {
        /* Config response — just accept, channel is now open */

    } else if (code == L2CAP_DISCONN_REQ && clen >= 4) {
        uint16_t dst_cid = params[0] | ((uint16_t)params[1] << 8);
        uint16_t src_cid = params[2] | ((uint16_t)params[3] << 8);
        l2cap_channel_t *ch = l2cap_find_local(dst_cid);
        if (ch) l2cap_free(ch);
        uint8_t rsp[4];
        rsp[0] = dst_cid & 0xFF; rsp[1] = dst_cid >> 8;
        rsp[2] = src_cid & 0xFF; rsp[3] = src_cid >> 8;
        l2cap_send_sig(L2CAP_DISCONN_RSP, id, rsp, 4);

    } else if (code == L2CAP_INFO_REQ && clen >= 2) {
        uint16_t info_type = params[0] | ((uint16_t)params[1] << 8);
        if (info_type == 0x0002) {
            /* Extended Features — report none */
            uint8_t rsp[8] = {0x02,0x00, 0x00,0x00, 0,0,0,0};
            l2cap_send_sig(L2CAP_INFO_RSP, id, rsp, 8);
        } else if (info_type == 0x0003) {
            /* Fixed Channels — only signaling (CID 1) */
            uint8_t rsp[12];
            rsp[0]=0x03; rsp[1]=0x00; rsp[2]=0x00; rsp[3]=0x00;
            rsp[4]=0x02; rsp[5]=0; rsp[6]=0; rsp[7]=0;
            rsp[8]=0; rsp[9]=0; rsp[10]=0; rsp[11]=0;
            l2cap_send_sig(L2CAP_INFO_RSP, id, rsp, 12);
        } else {
            /* Not supported */
            uint8_t rsp[4] = {0,0, 0x01,0x00}; /* result=not supported */
            rsp[0] = info_type & 0xFF; rsp[1] = info_type >> 8;
            l2cap_send_sig(L2CAP_INFO_RSP, id, rsp, 4);
        }
    }
}

/* =============================================================================
 * SDP — Service Discovery Protocol (static SPP record)
 * =============================================================================*/

/* Pre-computed SPP service record as SDP attribute list.
 * Service Record Handle: 0x00010001
 * Service Class: SerialPort (0x1101)
 * Protocol: L2CAP + RFCOMM channel 1
 * Browse Group: PublicBrowseRoot
 * Service Name: "TensorOS" */
static const uint8_t sdp_spp_record[] = {
    /* Attribute 0x0000 — ServiceRecordHandle */
    0x09, 0x00, 0x00,              /* UINT16 attr id = 0x0000 */
    0x0A, 0x00, 0x01, 0x00, 0x01,  /* UINT32 = 0x00010001 */

    /* Attribute 0x0001 — ServiceClassIDList */
    0x09, 0x00, 0x01,
    0x35, 0x03,                    /* Seq(3) */
      0x19, 0x11, 0x01,            /*   UUID16 SerialPort */

    /* Attribute 0x0004 — ProtocolDescriptorList */
    0x09, 0x00, 0x04,
    0x35, 0x0C,                    /* Seq(12) */
      0x35, 0x03,                  /*   Seq(3): L2CAP */
        0x19, 0x01, 0x00,          /*     UUID16 L2CAP */
      0x35, 0x05,                  /*   Seq(5): RFCOMM */
        0x19, 0x00, 0x03,          /*     UUID16 RFCOMM */
        0x08, 0x01,                /*     UINT8 channel=1 */

    /* Attribute 0x0005 — BrowseGroupList */
    0x09, 0x00, 0x05,
    0x35, 0x03,
      0x19, 0x10, 0x02,            /*   UUID16 PublicBrowseRoot */

    /* Attribute 0x0100 — ServiceName */
    0x09, 0x01, 0x00,
    0x25, 0x08,                    /* Text(8) */
      'T','e','n','s','o','r','O','S',
};

#define SDP_RECORD_LEN  ((int)sizeof(sdp_spp_record))

/* Handle an SDP request on the given L2CAP channel */
static void sdp_handle(const uint8_t *data, int len, l2cap_channel_t *ch)
{
    if (len < 5) return;
    uint8_t  pdu_id = data[0];
    uint16_t txn_id = ((uint16_t)data[1] << 8) | data[2]; /* big-endian */

    uint8_t rsp[256];
    int rlen = 0;

    if (pdu_id == 0x02) {
        /* ServiceSearchRequest → respond with handle 0x00010001 */
        rsp[0] = 0x03;  /* ServiceSearchResponse */
        rsp[1] = (uint8_t)(txn_id >> 8); rsp[2] = (uint8_t)txn_id;
        /* param len = 9 */
        rsp[3] = 0; rsp[4] = 9;
        /* TotalServiceRecordCount = 1 */
        rsp[5] = 0; rsp[6] = 1;
        /* CurrentServiceRecordCount = 1 */
        rsp[7] = 0; rsp[8] = 1;
        /* Handle */
        rsp[9] = 0x00; rsp[10] = 0x01; rsp[11] = 0x00; rsp[12] = 0x01;
        /* ContinuationState = 0 */
        rsp[13] = 0;
        rlen = 14;

    } else if (pdu_id == 0x04) {
        /* ServiceAttributeRequest → respond with full record */
        rsp[0] = 0x05;  /* ServiceAttributeResponse */
        rsp[1] = (uint8_t)(txn_id >> 8); rsp[2] = (uint8_t)txn_id;
        /* Build attribute list as a sequence */
        int seq_len = SDP_RECORD_LEN;
        int total_attr_bytes = seq_len + 2; /* 0x35 + len byte + data */
        int param_len = 2 + total_attr_bytes + 1; /* byte count(2) + data + cont(1) */
        rsp[3] = (uint8_t)(param_len >> 8); rsp[4] = (uint8_t)param_len;
        /* AttributeListByteCount */
        rsp[5] = (uint8_t)(total_attr_bytes >> 8);
        rsp[6] = (uint8_t)total_attr_bytes;
        /* Sequence header */
        rsp[7] = 0x35;
        rsp[8] = (uint8_t)seq_len;
        /* Attribute data */
        for (int i = 0; i < seq_len && i < 240; i++)
            rsp[9 + i] = sdp_spp_record[i];
        /* ContinuationState = 0 */
        rsp[9 + seq_len] = 0;
        rlen = 10 + seq_len;

    } else if (pdu_id == 0x06) {
        /* ServiceSearchAttributeRequest → respond with full record */
        rsp[0] = 0x07;  /* ServiceSearchAttributeResponse */
        rsp[1] = (uint8_t)(txn_id >> 8); rsp[2] = (uint8_t)txn_id;
        /* Wrap record in an outer sequence (list of attribute lists) */
        int inner_len = SDP_RECORD_LEN;
        int inner_seq = inner_len + 2;      /* 0x35 + len + data */
        int outer_seq = inner_seq + 2;      /* 0x35 + len + inner_seq */
        int param_len = 2 + outer_seq + 1;  /* byte_count(2) + outer + cont(1) */
        rsp[3] = (uint8_t)(param_len >> 8); rsp[4] = (uint8_t)param_len;
        /* AttributeListsByteCount */
        rsp[5] = (uint8_t)(outer_seq >> 8);
        rsp[6] = (uint8_t)outer_seq;
        /* Outer sequence */
        rsp[7] = 0x35; rsp[8] = (uint8_t)(inner_seq);
        /* Inner sequence (one record's attributes) */
        rsp[9] = 0x35; rsp[10] = (uint8_t)inner_len;
        for (int i = 0; i < inner_len && i < 230; i++)
            rsp[11 + i] = sdp_spp_record[i];
        rsp[11 + inner_len] = 0; /* ContinuationState */
        rlen = 12 + inner_len;
    }

    if (rlen > 0)
        hci_send_acl(hci_conn_handle, ch->local_cid, rsp, rlen);
}

/* =============================================================================
 * RFCOMM — Serial Port Emulation
 * =============================================================================*/

/* RFCOMM frame types */
#define RFCOMM_SABM  0x3F
#define RFCOMM_UA    0x73
#define RFCOMM_DM    0x0F
#define RFCOMM_DISC  0x53
#define RFCOMM_UIH   0xEF

#define RFCOMM_SERVER_CHANNEL 1
static int rfcomm_dlci_data = 0;   /* data DLCI (set when SABM received) */
static int rfcomm_open      = 0;   /* 1 if data channel is open */
static l2cap_channel_t *rfcomm_l2cap = (l2cap_channel_t *)0;

/* CRC-8 table for RFCOMM FCS (polynomial 0xE0, reversed from 0x07) */
static const uint8_t rfcomm_crc8[256] = {
    0x00,0x91,0xE3,0x72,0x07,0x96,0xE4,0x75,0x0E,0x9F,0xED,0x7C,0x09,0x98,0xEA,0x7B,
    0x1C,0x8D,0xFF,0x6E,0x1B,0x8A,0xF8,0x69,0x12,0x83,0xF1,0x60,0x15,0x84,0xF6,0x67,
    0x38,0xA9,0xDB,0x4A,0x3F,0xAE,0xDC,0x4D,0x36,0xA7,0xD5,0x44,0x31,0xA0,0xD2,0x43,
    0x24,0xB5,0xC7,0x56,0x23,0xB2,0xC0,0x51,0x2A,0xBB,0xC9,0x58,0x2D,0xBC,0xCE,0x5F,
    0x70,0xE1,0x93,0x02,0x77,0xE6,0x94,0x05,0x7E,0xEF,0x9D,0x0C,0x79,0xE8,0x9A,0x0B,
    0x6C,0xFD,0x8F,0x1E,0x6B,0xFA,0x88,0x19,0x62,0xF3,0x81,0x10,0x65,0xF4,0x86,0x17,
    0x48,0xD9,0xAB,0x3A,0x4F,0xDE,0xAC,0x3D,0x46,0xD7,0xA5,0x34,0x41,0xD0,0xA2,0x33,
    0x54,0xC5,0xB7,0x26,0x53,0xC2,0xB0,0x21,0x5A,0xCB,0xB9,0x28,0x5D,0xCC,0xBE,0x2F,
    0xE0,0x71,0x03,0x92,0xE7,0x76,0x04,0x95,0xEE,0x7F,0x0D,0x9C,0xE9,0x78,0x0A,0x9B,
    0xFC,0x6D,0x1F,0x8E,0xFB,0x6A,0x18,0x89,0xF2,0x63,0x11,0x80,0xF5,0x64,0x16,0x87,
    0xD8,0x49,0x3B,0xAA,0xDF,0x4E,0x3C,0xAD,0xD6,0x47,0x35,0xA4,0xD1,0x40,0x32,0xA3,
    0xC4,0x55,0x27,0xB6,0xC3,0x52,0x20,0xB1,0xCA,0x5B,0x29,0xB8,0xCD,0x5C,0x2E,0xBF,
    0x90,0x01,0x73,0xE2,0x97,0x06,0x74,0xE5,0x9E,0x0F,0x7D,0xEC,0x99,0x08,0x7A,0xEB,
    0x8C,0x1D,0x6F,0xFE,0x8B,0x1A,0x68,0xF9,0x82,0x13,0x61,0xF0,0x85,0x14,0x66,0xF7,
    0xA8,0x39,0x4B,0xDA,0xAF,0x3E,0x4C,0xDD,0xA6,0x37,0x45,0xD4,0xA1,0x30,0x42,0xD3,
    0xB4,0x25,0x57,0xC6,0xB3,0x22,0x50,0xC1,0xBA,0x2B,0x59,0xC8,0xBD,0x2C,0x5E,0xCF,
};

static uint8_t rfcomm_fcs(const uint8_t *d, int n)
{
    uint8_t fcs = 0xFF;
    for (int i = 0; i < n; i++) fcs = rfcomm_crc8[fcs ^ d[i]];
    return 0xFF - fcs;
}

/* Send an RFCOMM frame via L2CAP */
static void rfcomm_send_frame(uint8_t addr, uint8_t ctrl,
                              const uint8_t *data, int dlen)
{
    if (!rfcomm_l2cap) return;
    uint8_t frame[512];
    int pos = 0;

    frame[pos++] = addr;
    frame[pos++] = ctrl;

    /* Length: 1 byte if < 128, else 2 bytes */
    if (dlen < 128) {
        frame[pos++] = (uint8_t)((dlen << 1) | 1);  /* EA=1 */
    } else {
        frame[pos++] = (uint8_t)((dlen & 0x7F) << 1); /* EA=0 */
        frame[pos++] = (uint8_t)(dlen >> 7);
    }

    /* Credits (for UIH with PF bit set) */
    /* We won't send credit-based flow for simplicity */

    for (int i = 0; i < dlen && pos < 500; i++)
        frame[pos++] = data[i];

    /* FCS: over address + control only for UIH, over addr+ctrl for others */
    uint8_t fcs;
    if ((ctrl & ~0x10) == RFCOMM_UIH) {
        fcs = rfcomm_fcs(frame, 2);   /* addr + ctrl only */
    } else {
        fcs = rfcomm_fcs(frame, 2);   /* SABM/UA/DM: addr + ctrl */
    }
    frame[pos++] = fcs;

    hci_send_acl(hci_conn_handle, rfcomm_l2cap->local_cid, frame, pos);
}

/* Respond to RFCOMM multiplexer commands (PN, MSC) on DLCI 0 */
static void rfcomm_handle_mux(uint8_t addr, const uint8_t *data, int dlen)
{
    if (dlen < 2) return;
    uint8_t type   = data[0];
    uint8_t mlen   = data[1] >> 1; /* length in EA format */
    const uint8_t *mdata = data + 2;

    uint8_t type_val = type & 0xFC; /* strip EA and CR bits */

    if (type_val == 0x80 && mlen >= 8) {
        /* PN (Parameter Negotiation) — 0x20 << 2 = 0x80 */
        uint8_t rsp[10 + 2];
        /* Type: PN response (CR=0) */
        rsp[0] = 0x81; /* type=0x80 | EA=1, CR=0 for response */
        rsp[1] = (8 << 1) | 1;  /* length=8, EA=1 */
        /* Echo back DLCI */
        rsp[2] = mdata[0]; /* DLCI */
        rsp[3] = 0;        /* CL/flow: no credit-based flow */
        rsp[4] = 0;        /* priority */
        rsp[5] = 0;        /* timer */
        /* MTU: accept what they propose, up to 512 */
        uint16_t mtu = mdata[4] | ((uint16_t)mdata[5] << 8);
        if (mtu > 512) mtu = 512;
        if (mtu == 0) mtu = 127;
        rsp[6] = mtu & 0xFF;
        rsp[7] = mtu >> 8;
        rsp[8] = 0;  /* max retransmissions (N1) */
        rsp[9] = 0;  /* initial credits */
        rfcomm_send_frame(addr | 0x02, RFCOMM_UIH, rsp, 10);

    } else if (type_val == 0xE0 && mlen >= 2) {
        /* MSC (Modem Status Command) — 0x38 << 2 = 0xE0 */
        /* Respond with same signals */
        uint8_t rsp[4 + 2];
        rsp[0] = 0xE1; /* MSC response: type=0xE0 | EA=1, CR=0 */
        rsp[1] = (2 << 1) | 1; /* length=2 */
        rsp[2] = mdata[0];     /* DLCI address */
        rsp[3] = mdata[1];     /* modem signals */
        rfcomm_send_frame(addr | 0x02, RFCOMM_UIH, rsp, 4);
    }
    /* Other mux commands silently ignored */
}

/* Handle a received RFCOMM frame */
static void rfcomm_handle(const uint8_t *data, int len, l2cap_channel_t *ch)
{
    if (len < 3) return;

    rfcomm_l2cap = ch;

    uint8_t  addr = data[0];
    uint8_t  ctrl = data[1];
    int      dlci = (addr >> 2) & 0x3F;
    /* Length */
    int      doff = 2;
    int      dlen;
    if (data[2] & 1) {
        dlen = data[2] >> 1;
        doff = 3;
    } else {
        dlen = (data[2] >> 1) | ((int)data[3] << 7);
        doff = 4;
    }

    const uint8_t *payload = data + doff;
    /* FCS is at data[doff + dlen] */

    uint8_t resp_ctrl = ctrl;  (void)resp_ctrl;

    if (ctrl == RFCOMM_SABM || ctrl == (RFCOMM_SABM | 0x10)) {
        /* Respond with UA */
        uint8_t resp_addr = addr;
        rfcomm_send_frame(resp_addr, RFCOMM_UA | 0x10, (const uint8_t *)0, 0);

        if (dlci == 0) {
            /* Multiplexer opened */
        } else {
            /* Data channel opened */
            rfcomm_dlci_data = dlci;
            rfcomm_open = 1;
        }

    } else if ((ctrl & ~0x10) == RFCOMM_UIH) {
        if (dlci == 0) {
            /* Multiplexer command */
            rfcomm_handle_mux(addr, payload, dlen);
        } else if (dlci == rfcomm_dlci_data && rfcomm_open) {
            /* Data! Push into receive ring buffer */
            for (int i = 0; i < dlen; i++) {
                int next = (bt_rx_head + 1) % BT_RX_SIZE;
                if (next != bt_rx_tail) {
                    bt_rxbuf[bt_rx_head] = (char)payload[i];
                    bt_rx_head = next;
                }
            }
        }

    } else if (ctrl == RFCOMM_DISC || ctrl == (RFCOMM_DISC | 0x10)) {
        /* Disconnect — respond with UA */
        rfcomm_send_frame(addr, RFCOMM_UA | 0x10, (const uint8_t *)0, 0);
        if (dlci == rfcomm_dlci_data) rfcomm_open = 0;
        if (dlci == 0) rfcomm_open = 0;
    }
}

/* Send data over RFCOMM (called from bt_poll to flush TX buffer) */
static void rfcomm_flush_tx(void)
{
    if (!rfcomm_open || !rfcomm_l2cap) return;

    uint8_t chunk[128];
    int n = 0;
    while (bt_tx_tail != bt_tx_head && n < 120) {
        chunk[n++] = (uint8_t)bt_txbuf[bt_tx_tail];
        bt_tx_tail = (bt_tx_tail + 1) % BT_TX_SIZE;
    }
    if (n == 0) return;

    /* Build address for our outgoing data */
    uint8_t addr = (uint8_t)((rfcomm_dlci_data << 2) | 0x02 | 0x01);
    /* EA=1, CR=0 for responder data */
    rfcomm_send_frame(addr, RFCOMM_UIH, chunk, n);
}

/* =============================================================================
 * HCI Event Handling
 * =============================================================================*/

static void handle_hci_event(const uint8_t *buf, int len)
{
    if (len < 2) return;
    uint8_t code = buf[0];
    /* uint8_t plen = buf[1]; */
    const uint8_t *p = buf + 2;

    switch (code) {
    case HCI_EVT_CONN_REQUEST: {
        /* Accept incoming connection */
        /* p: bd_addr(6) + class(3) + link_type(1) */
        if (len < 12) break;
        for (int i = 0; i < 6; i++) hci_remote_addr[i] = p[i];
        uint8_t params[7];
        for (int i = 0; i < 6; i++) params[i] = p[i]; /* BD_ADDR */
        params[6] = 0x01; /* Role: remain slave */
        hci_send_cmd(HCI_ACCEPT_CONN_REQ, params, 7);
        break;
    }

    case HCI_EVT_CONN_COMPLETE: {
        /* p: status(1) + handle(2) + bd_addr(6) + link_type(1) + enc(1) */
        if (len < 13) break;
        if (p[0] == 0) {
            hci_conn_handle = p[1] | ((uint16_t)p[2] << 8);
            hci_conn_handle &= 0x0FFF;
            hci_connected = 1;
        }
        break;
    }

    case HCI_EVT_DISCONN_COMPLETE: {
        if (len >= 6) {
            uint16_t h = p[1] | ((uint16_t)p[2] << 8);
            h &= 0x0FFF;
            if (h == hci_conn_handle) {
                hci_connected = 0;
                rfcomm_open = 0;
                rfcomm_l2cap = (l2cap_channel_t *)0;
                /* Free all L2CAP channels */
                for (int i = 0; i < L2CAP_MAX_CHANNELS; i++)
                    l2cap_channels[i].active = 0;
            }
        }
        break;
    }

    case HCI_EVT_PIN_CODE_REQ: {
        /* Legacy PIN pairing — respond with "0000" */
        uint8_t params[10];
        for (int i = 0; i < 6; i++) params[i] = p[i]; /* BD_ADDR */
        params[6] = 4;             /* PIN length */
        params[7] = '0';           /* PIN code */
        params[8] = '0';
        params[9] = '0';
        /* HCI_Pin_Code_Request_Reply has PIN up to 16 bytes, padded with 0 */
        uint8_t full[23];
        for (int i = 0; i < 23; i++) full[i] = 0;
        for (int i = 0; i < 6; i++) full[i] = p[i];
        full[6] = 4;
        full[7] = '0'; full[8] = '0'; full[9] = '0'; full[10] = '0';
        hci_send_cmd(HCI_PIN_CODE_REPLY, full, 23);
        break;
    }

    case HCI_EVT_LINK_KEY_REQ: {
        /* We don't store link keys — send negative reply */
        hci_send_cmd(HCI_LINK_KEY_NEG_REPLY, p, 6);
        break;
    }

    case HCI_EVT_IO_CAP_REQ: {
        /* SSP IO Capability Request — reply NoInputNoOutput (Just Works) */
        uint8_t params[9];
        for (int i = 0; i < 6; i++) params[i] = p[i]; /* BD_ADDR */
        params[6] = 0x03; /* IO capability: NoInputNoOutput */
        params[7] = 0x00; /* OOB data: not present */
        params[8] = 0x00; /* Auth requirement: no MITM */
        hci_send_cmd(HCI_IO_CAP_REQ_REPLY, params, 9);
        break;
    }

    case HCI_EVT_USER_CONFIRM_REQ: {
        /* SSP User Confirmation — auto-accept (Just Works) */
        hci_send_cmd(HCI_USER_CONFIRM_REPLY, p, 6);
        break;
    }

    case HCI_EVT_LINK_KEY_NOTIF:
    case HCI_EVT_IO_CAP_RESP:
    case HCI_EVT_SSP_COMPLETE:
    case HCI_EVT_CMD_COMPLETE:
    case HCI_EVT_CMD_STATUS:
    case HCI_EVT_NUM_COMP_PKTS:
        /* Informational — no action needed */
        break;

    default:
        break;
    }
}

/* Handle received ACL data (L2CAP routing) */
static void handle_hci_acl(const uint8_t *buf, int len)
{
    if (len < 8) return; /* HCI ACL header(4) + L2CAP header(4) minimum */

    /* uint16_t handle = (buf[0] | ((uint16_t)buf[1] << 8)) & 0x0FFF; */
    /* uint16_t dlen   = buf[2] | ((uint16_t)buf[3] << 8); */
    uint16_t l2len  = buf[4] | ((uint16_t)buf[5] << 8);
    uint16_t l2cid  = buf[6] | ((uint16_t)buf[7] << 8);

    const uint8_t *l2data = buf + 8;
    int l2data_len = (len - 8 < (int)l2len) ? len - 8 : (int)l2len;

    if (l2cid == L2CAP_CID_SIGNALING) {
        l2cap_handle_signaling(l2data, l2data_len);
        return;
    }

    /* Find which channel this belongs to */
    l2cap_channel_t *ch = l2cap_find_local(l2cid);
    if (!ch) return;

    if (ch->psm == L2CAP_PSM_SDP) {
        sdp_handle(l2data, l2data_len, ch);
    } else if (ch->psm == L2CAP_PSM_RFCOMM) {
        rfcomm_handle(l2data, l2data_len, ch);
    }
}

/* =============================================================================
 * Public API
 * =============================================================================*/

int bt_init(void)
{
    /* LED ON = starting BT init (will turn OFF on success) */
    led_on();
    uart_puts("[BT] Initializing PL011 UART for BCM43455...\r\n");

    /* Initialize PL011 UART at 115200 for BCM43455 */
    mu_init(115200);

    /* Small delay for BT chip to be ready after power-on */
    arm_timer_delay_ms(200);

    /* Drain any stale data */
    while (mu_has_data()) (void)mu_getc();

    uart_puts("[BT] Sending HCI_RESET...\r\n");

    /* HCI Reset */
    hci_send_cmd0(HCI_RESET);
    int status = hci_wait_cmd_complete(HCI_RESET, 3000);
    if (status != 0) {
        /* LED stays ON = HCI_RESET failed */
        uart_puts("[BT] HCI_RESET FAILED — chip not responding\r\n");

        /* Try once more after toggling PL011 and longer delay */
        mmio_write(UART_CR, 0);       /* disable UART */
        arm_timer_delay_ms(500);
        mu_init(115200);               /* re-init */
        arm_timer_delay_ms(200);
        while (mu_has_data()) (void)mu_getc();

        hci_send_cmd0(HCI_RESET);
        status = hci_wait_cmd_complete(HCI_RESET, 3000);
        if (status != 0) {
            uart_puts("[BT] HCI_RESET retry FAILED\r\n");
            return -1;  /* LED stays ON = failure */
        }
        uart_puts("[BT] HCI_RESET retry OK\r\n");
    }

    uart_puts("[BT] HCI_RESET OK\r\n");

    arm_timer_delay_ms(50);

    /* Set Event Mask — enable all interesting events */
    {
        uint8_t mask[8] = {0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0x3F,0x20};
        hci_send_cmd(HCI_SET_EVENT_MASK, mask, 8);
        hci_wait_cmd_complete(HCI_SET_EVENT_MASK, 1000);
    }

    /* Write Local Name: "TensorOS" */
    {
        uint8_t name[248];
        for (int i = 0; i < 248; i++) name[i] = 0;
        const char *n = "TensorOS";
        for (int i = 0; n[i]; i++) name[i] = (uint8_t)n[i];
        hci_send_cmd(HCI_WRITE_LOCAL_NAME, name, 248);
        hci_wait_cmd_complete(HCI_WRITE_LOCAL_NAME, 1000);
    }

    /* Write Class of Device: Computer (0x000100) */
    {
        uint8_t cod[3] = {0x00, 0x01, 0x00};
        hci_send_cmd(HCI_WRITE_CLASS_OF_DEVICE, cod, 3);
        hci_wait_cmd_complete(HCI_WRITE_CLASS_OF_DEVICE, 1000);
    }

    /* Enable SSP (Secure Simple Pairing) */
    {
        uint8_t ssp = 1;
        hci_send_cmd(HCI_WRITE_SSP_MODE, &ssp, 1);
        hci_wait_cmd_complete(HCI_WRITE_SSP_MODE, 1000);
    }

    /* Disable authentication requirement for now */
    {
        uint8_t auth = 0;
        hci_send_cmd(HCI_WRITE_AUTH_ENABLE, &auth, 1);
        hci_wait_cmd_complete(HCI_WRITE_AUTH_ENABLE, 1000);
    }

    /* Write Scan Enable: Inquiry + Page scan (discoverable + connectable) */
    {
        uint8_t scan = 0x03;
        hci_send_cmd(HCI_WRITE_SCAN_ENABLE, &scan, 1);
        hci_wait_cmd_complete(HCI_WRITE_SCAN_ENABLE, 1000);
    }

    uart_puts("[BT] Discoverable as 'TensorOS'\r\n");

    /* Read BD_ADDR for info */
    hci_send_cmd0(HCI_READ_BD_ADDR);
    {
        int len;
        int type = hci_read_packet(&len, 1000);
        if (type == HCI_EVT_PKT && len >= 12 && hci_buf[0] == HCI_EVT_CMD_COMPLETE) {
            /* BD_ADDR is at bytes 6-11 of the event params */
            uart_puts("[BT] Address: ");
            for (int i = 11; i >= 6; i--) {
                static const char hex[] = "0123456789ABCDEF";
                uart_putchar(hex[hci_buf[i] >> 4]);
                uart_putchar(hex[hci_buf[i] & 0xF]);
                if (i > 6) uart_putchar(':');
            }
            uart_puts("\r\n");
        }
    }

    /* Initialize L2CAP channels */
    for (int i = 0; i < L2CAP_MAX_CHANNELS; i++)
        l2cap_channels[i].active = 0;
    l2cap_next_cid = 0x0040;

    /* LED OFF = BT init succeeded, now discoverable */
    led_off();
    return 0;
}

void bt_poll(void)
{
    /* Process any incoming HCI packets */
    while (mu_has_data()) {
        int len;
        int type = hci_read_packet(&len, 50);
        if (type == HCI_EVT_PKT) {
            handle_hci_event(hci_buf, len);
        } else if (type == HCI_ACL_PKT) {
            handle_hci_acl(hci_buf, len);
        }
        if (type == 0) break; /* timeout, no more data */
    }

    /* Flush outgoing RFCOMM data */
    rfcomm_flush_tx();
}

void bt_putchar(char c)
{
    if (!rfcomm_open) return;
    int next = (bt_tx_head + 1) % BT_TX_SIZE;
    if (next == bt_tx_tail) return; /* buffer full, drop */
    bt_txbuf[bt_tx_head] = c;
    bt_tx_head = next;
}

int bt_has_data(void)
{
    return bt_rx_head != bt_rx_tail;
}

char bt_getchar(void)
{
    while (bt_rx_head == bt_rx_tail) {
        bt_poll();
    }
    char c = bt_rxbuf[bt_rx_tail];
    bt_rx_tail = (bt_rx_tail + 1) % BT_RX_SIZE;
    return c;
}

int bt_connected(void)
{
    return rfcomm_open;
}

#else /* ===== x86 — Bluetooth not applicable ===== */

int  bt_init(void)      { return -1; }
void bt_poll(void)      {}
void bt_putchar(char c) { (void)c; }
int  bt_has_data(void)  { return 0; }
char bt_getchar(void)   { return 0; }
int  bt_connected(void) { return 0; }

#endif
