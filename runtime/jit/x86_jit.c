/* =============================================================================
 * TensorOS - x86_64 JIT Compiler Implementation
 *
 * x86_64-only: ARM64 would need an AArch64 JIT backend.
 * =============================================================================*/

#ifndef __aarch64__
/* =============================================================================
 * Generates native machine code at runtime for tensor operations.
 * This is the core differentiator: an OS that JIT-compiles AI workloads
 * directly to hardware instructions with zero overhead.
 *
 * Architecture:
 *   1. Low-level emitter: byte-accurate x86_64 instruction encoding
 *   2. SSE2 SIMD emission: packed float operations
 *   3. High-level compiler: tensor ops → native code
 *
 * All generated code uses System V AMD64 ABI:
 *   Args: rdi, rsi, rdx, rcx, r8, r9  |  XMM0-7 for floats
 *   Callee-saved: rbx, rbp, r12-r15   |  Caller-saved: rax,rcx,rdx,rsi,rdi,r8-r11
 * =============================================================================*/

#include "runtime/jit/x86_jit.h"
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"

/* =============================================================================
 * Buffer Management
 *
 * Uses a static pool instead of kmalloc to avoid allocator issues
 * and ensure JIT code is always in executable kernel memory.
 * Pool lives in BSS — zero-initialized, within the kernel's identity map.
 * =============================================================================*/

#define JIT_POOL_SIZE  (2 * 1024 * 1024)  /* 2MB for all JIT code */
static uint8_t jit_pool[JIT_POOL_SIZE] __attribute__((aligned(4096)));
static int jit_pool_offset = 0;

/* Statically allocate jit_buf_t structs (max 64 concurrent buffers) */
#define JIT_MAX_BUFS 64
static jit_buf_t jit_buf_storage[JIT_MAX_BUFS];
static int jit_buf_count = 0;
static bool jit_buf_active[JIT_MAX_BUFS]; /* Track which slots are in use */

jit_buf_t *jit_create(int capacity)
{
    /* Align capacity to 16 bytes */
    capacity = (capacity + 15) & ~15;

    /* Try to find a recycled slot first */
    for (int i = 0; i < jit_buf_count; i++) {
        if (!jit_buf_active[i] && jit_buf_storage[i].cap >= capacity) {
            jit_buf_active[i] = true;
            jit_buf_storage[i].len = 0;
            /* Ensure buffer is writable (may have been marked RX previously) */
            vmm_mark_rw(jit_buf_storage[i].code, jit_buf_storage[i].cap);
            return &jit_buf_storage[i];
        }
    }

    if (jit_buf_count >= JIT_MAX_BUFS) return NULL;
    if (jit_pool_offset + capacity > JIT_POOL_SIZE) return NULL;

    /* Ensure new region is writable (previous vmm_mark_rx may have made
     * the enclosing page read-only / execute-only) */
    vmm_mark_rw(jit_pool + jit_pool_offset, capacity);

    int idx = jit_buf_count++;
    jit_buf_t *b = &jit_buf_storage[idx];
    b->code = jit_pool + jit_pool_offset;
    b->len = 0;
    b->cap = capacity;
    jit_pool_offset += capacity;
    jit_buf_active[idx] = true;
    return b;
}

void jit_destroy(jit_buf_t *buf)
{
    if (!buf) return;
    /* Mark slot as inactive so jit_create can reuse it */
    for (int i = 0; i < jit_buf_count; i++) {
        if (&jit_buf_storage[i] == buf) {
            jit_buf_active[i] = false;
            buf->len = 0;
            return;
        }
    }
}

void jit_reset(jit_buf_t *buf)
{
    if (buf) buf->len = 0;
}

jit_void_fn jit_get_fn(jit_buf_t *buf)
{
    if (!buf || !buf->code) return NULL;
    /* Ensure pages are executable (W^X: flip from RW to RX) */
    vmm_mark_rx(buf->code, buf->cap);
    return (jit_void_fn)(void *)buf->code;
}

/* =============================================================================
 * Low-Level Byte Emission
 * =============================================================================*/

void jit_emit8(jit_buf_t *b, uint8_t v)
{
    if (b->len < b->cap)
        b->code[b->len++] = v;
}

void jit_emit16(jit_buf_t *b, uint16_t v)
{
    jit_emit8(b, v & 0xFF);
    jit_emit8(b, (v >> 8) & 0xFF);
}

void jit_emit32(jit_buf_t *b, uint32_t v)
{
    jit_emit8(b, v & 0xFF);
    jit_emit8(b, (v >> 8) & 0xFF);
    jit_emit8(b, (v >> 16) & 0xFF);
    jit_emit8(b, (v >> 24) & 0xFF);
}

void jit_emit64(jit_buf_t *b, uint64_t v)
{
    jit_emit32(b, (uint32_t)v);
    jit_emit32(b, (uint32_t)(v >> 32));
}

/* =============================================================================
 * x86_64 Encoding Helpers
 * =============================================================================*/

/* REX prefix: 0100WRXB */
static void emit_rex(jit_buf_t *b, int w, int r, int x, int bv)
{
    uint8_t rex = 0x40;
    if (w) rex |= 0x08;
    if (r) rex |= 0x04;
    if (x) rex |= 0x02;
    if (bv) rex |= 0x01;
    jit_emit8(b, rex);
}

/* REX.W for 64-bit operation with potential extended registers */
static void emit_rex_w(jit_buf_t *b, int reg, int rm)
{
    emit_rex(b, 1, reg >= 8, 0, rm >= 8);
}

/* ModR/M byte */
static void emit_modrm(jit_buf_t *b, int mod, int reg, int rm)
{
    jit_emit8(b, (uint8_t)((mod << 6) | ((reg & 7) << 3) | (rm & 7)));
}

/* ModR/M with displacement (handles 0, int8, int32 cases) */
static void emit_modrm_disp(jit_buf_t *b, int reg, int base, int32_t disp)
{
    if (base == RSP || base == R12) {
        /* RSP/R12 needs SIB byte */
        if (disp == 0 && base != RBP && base != R13) {
            emit_modrm(b, 0, reg, 4);
            jit_emit8(b, 0x24); /* SIB: scale=0, index=RSP(none), base=RSP */
        } else if (disp >= -128 && disp <= 127) {
            emit_modrm(b, 1, reg, 4);
            jit_emit8(b, 0x24);
            jit_emit8(b, (uint8_t)(int8_t)disp);
        } else {
            emit_modrm(b, 2, reg, 4);
            jit_emit8(b, 0x24);
            jit_emit32(b, (uint32_t)disp);
        }
    } else if (disp == 0 && base != RBP && base != R13) {
        emit_modrm(b, 0, reg, base);
    } else if (disp >= -128 && disp <= 127) {
        emit_modrm(b, 1, reg, base);
        jit_emit8(b, (uint8_t)(int8_t)disp);
    } else {
        emit_modrm(b, 2, reg, base);
        jit_emit32(b, (uint32_t)disp);
    }
}

/* =============================================================================
 * GP Register Instructions
 * =============================================================================*/

void jit_push(jit_buf_t *b, int reg)
{
    if (reg >= 8) jit_emit8(b, 0x41); /* REX.B for r8-r15 */
    jit_emit8(b, 0x50 + (reg & 7));
}

void jit_pop(jit_buf_t *b, int reg)
{
    if (reg >= 8) jit_emit8(b, 0x41);
    jit_emit8(b, 0x58 + (reg & 7));
}

void jit_ret(jit_buf_t *b)
{
    jit_emit8(b, 0xC3);
}

/* mov r64, r64 */
void jit_mov_reg_reg(jit_buf_t *b, int dst, int src)
{
    emit_rex_w(b, src, dst);
    jit_emit8(b, 0x89);
    emit_modrm(b, 3, src, dst);
}

/* mov r64, imm64 */
void jit_mov_reg_imm64(jit_buf_t *b, int reg, uint64_t imm)
{
    emit_rex(b, 1, 0, 0, reg >= 8);
    jit_emit8(b, 0xB8 + (reg & 7));
    jit_emit64(b, imm);
}

/* mov r64, [base + disp] */
void jit_mov_reg_mem(jit_buf_t *b, int dst, int base, int32_t disp)
{
    emit_rex_w(b, dst, base);
    jit_emit8(b, 0x8B);
    emit_modrm_disp(b, dst, base, disp);
}

/* mov [base + disp], r64 */
void jit_mov_mem_reg(jit_buf_t *b, int base, int32_t disp, int src)
{
    emit_rex_w(b, src, base);
    jit_emit8(b, 0x89);
    emit_modrm_disp(b, src, base, disp);
}

/* lea r64, [base + disp] */
void jit_lea(jit_buf_t *b, int dst, int base, int32_t disp)
{
    emit_rex_w(b, dst, base);
    jit_emit8(b, 0x8D);
    emit_modrm_disp(b, dst, base, disp);
}

/* add r64, r64 */
void jit_add_reg_reg(jit_buf_t *b, int dst, int src)
{
    emit_rex_w(b, src, dst);
    jit_emit8(b, 0x01);
    emit_modrm(b, 3, src, dst);
}

/* add r64, imm32 (sign-extended) */
void jit_add_reg_imm32(jit_buf_t *b, int dst, int32_t imm)
{
    emit_rex_w(b, 0, dst);
    if (imm >= -128 && imm <= 127) {
        jit_emit8(b, 0x83);
        emit_modrm(b, 3, 0, dst);
        jit_emit8(b, (uint8_t)(int8_t)imm);
    } else {
        jit_emit8(b, 0x81);
        emit_modrm(b, 3, 0, dst);
        jit_emit32(b, (uint32_t)imm);
    }
}

/* sub r64, imm32 */
void jit_sub_reg_imm32(jit_buf_t *b, int dst, int32_t imm)
{
    emit_rex_w(b, 0, dst);
    if (imm >= -128 && imm <= 127) {
        jit_emit8(b, 0x83);
        emit_modrm(b, 3, 5, dst);  /* /5 = SUB */
        jit_emit8(b, (uint8_t)(int8_t)imm);
    } else {
        jit_emit8(b, 0x81);
        emit_modrm(b, 3, 5, dst);
        jit_emit32(b, (uint32_t)imm);
    }
}

/* imul r64, r64 */
void jit_imul_reg_reg(jit_buf_t *b, int dst, int src)
{
    emit_rex_w(b, dst, src);
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0xAF);
    emit_modrm(b, 3, dst, src);
}

/* cmp r64, r64 */
void jit_cmp_reg_reg(jit_buf_t *b, int a, int reg_b)
{
    emit_rex_w(b, reg_b, a);
    jit_emit8(b, 0x39);
    emit_modrm(b, 3, reg_b, a);
}

/* cmp r64, imm32 */
void jit_cmp_reg_imm32(jit_buf_t *b, int reg, int32_t imm)
{
    emit_rex_w(b, 0, reg);
    if (imm >= -128 && imm <= 127) {
        jit_emit8(b, 0x83);
        emit_modrm(b, 3, 7, reg);  /* /7 = CMP */
        jit_emit8(b, (uint8_t)(int8_t)imm);
    } else {
        jit_emit8(b, 0x81);
        emit_modrm(b, 3, 7, reg);
        jit_emit32(b, (uint32_t)imm);
    }
}

/* xor r64, r64 (used to zero a register) */
void jit_xor_reg_reg(jit_buf_t *b, int dst, int src)
{
    emit_rex_w(b, src, dst);
    jit_emit8(b, 0x31);
    emit_modrm(b, 3, src, dst);
}

/* inc r64 */
void jit_inc_reg(jit_buf_t *b, int reg)
{
    emit_rex_w(b, 0, reg);
    jit_emit8(b, 0xFF);
    emit_modrm(b, 3, 0, reg);  /* /0 = INC */
}

/* =============================================================================
 * Control Flow
 * =============================================================================*/

/* call via rax (mov rax, addr; call rax) */
void jit_call_abs(jit_buf_t *b, uint64_t addr)
{
    jit_mov_reg_imm64(b, RAX, addr);
    /* call rax: FF D0 */
    jit_emit8(b, 0xFF);
    emit_modrm(b, 3, 2, RAX); /* /2 = CALL, mod=11 for register */
}

/* Forward jump (placeholder — returns patch offset) */
int jit_jmp_fwd(jit_buf_t *b)
{
    jit_emit8(b, 0xE9); /* JMP rel32 */
    int patch = b->len;
    jit_emit32(b, 0); /* placeholder */
    return patch;
}

int jit_jl_fwd(jit_buf_t *b)
{
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0x8C); /* JL rel32 */
    int patch = b->len;
    jit_emit32(b, 0);
    return patch;
}

int jit_jge_fwd(jit_buf_t *b)
{
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0x8D); /* JGE rel32 */
    int patch = b->len;
    jit_emit32(b, 0);
    return patch;
}

/* Patch a forward jump to land at current position */
void jit_patch_jump(jit_buf_t *b, int patch_offset)
{
    int32_t rel = b->len - (patch_offset + 4);
    b->code[patch_offset + 0] = (uint8_t)(rel & 0xFF);
    b->code[patch_offset + 1] = (uint8_t)((rel >> 8) & 0xFF);
    b->code[patch_offset + 2] = (uint8_t)((rel >> 16) & 0xFF);
    b->code[patch_offset + 3] = (uint8_t)((rel >> 24) & 0xFF);
}

/* Backward jump: JMP to a known target */
void jit_jmp_back(jit_buf_t *b, int target_offset)
{
    jit_emit8(b, 0xE9);
    int32_t rel = target_offset - (b->len + 4);
    jit_emit32(b, (uint32_t)rel);
}

void jit_jl_back(jit_buf_t *b, int target_offset)
{
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0x8C);
    int32_t rel = target_offset - (b->len + 4);
    jit_emit32(b, (uint32_t)rel);
}

void jit_jge_back(jit_buf_t *b, int target_offset)
{
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0x8D);
    int32_t rel = target_offset - (b->len + 4);
    jit_emit32(b, (uint32_t)rel);
}

/* =============================================================================
 * SSE2 Packed Float Instructions
 * =============================================================================*/

/* Encode SSE prefix + opcode with REX for extended regs */
static void emit_sse_op(jit_buf_t *b, uint8_t op, int dst_xmm, int src_or_base,
                         int is_mem, int32_t disp)
{
    /* REX prefix if either register is xmm8+ or base is r8+ */
    int need_rex = (dst_xmm >= 8 || src_or_base >= 8);
    if (need_rex) {
        emit_rex(b, 0, dst_xmm >= 8, 0, src_or_base >= 8);
    }
    jit_emit8(b, 0x0F);
    jit_emit8(b, op);
    if (is_mem) {
        emit_modrm_disp(b, dst_xmm, src_or_base, disp);
    } else {
        emit_modrm(b, 3, dst_xmm, src_or_base);
    }
}

/* movups xmm, [base+disp] — unaligned load */
void jit_movups_load(jit_buf_t *b, int xmm, int base, int32_t disp)
{
    emit_sse_op(b, 0x10, xmm, base, 1, disp);
}

/* movups [base+disp], xmm — unaligned store */
void jit_movups_store(jit_buf_t *b, int base, int32_t disp, int xmm)
{
    emit_sse_op(b, 0x11, xmm, base, 1, disp);
}

/* movaps xmm, [base+disp] — aligned load (data MUST be 16-byte aligned) */
void jit_movaps_load(jit_buf_t *b, int xmm, int base, int32_t disp)
{
    emit_sse_op(b, 0x28, xmm, base, 1, disp);
}

/* movaps [base+disp], xmm — aligned store (data MUST be 16-byte aligned) */
void jit_movaps_store(jit_buf_t *b, int base, int32_t disp, int xmm)
{
    emit_sse_op(b, 0x29, xmm, base, 1, disp);
}

/* addps xmm, xmm */
void jit_addps(jit_buf_t *b, int dst_xmm, int src_xmm)
{
    emit_sse_op(b, 0x58, dst_xmm, src_xmm, 0, 0);
}

/* mulps xmm, xmm */
void jit_mulps(jit_buf_t *b, int dst_xmm, int src_xmm)
{
    emit_sse_op(b, 0x59, dst_xmm, src_xmm, 0, 0);
}

/* xorps xmm, xmm (used to zero a register) */
void jit_xorps(jit_buf_t *b, int dst_xmm, int src_xmm)
{
    emit_sse_op(b, 0x57, dst_xmm, src_xmm, 0, 0);
}

/* maxps xmm, xmm */
void jit_maxps(jit_buf_t *b, int dst_xmm, int src_xmm)
{
    emit_sse_op(b, 0x5F, dst_xmm, src_xmm, 0, 0);
}

/* shufps xmm, xmm, imm8 */
void jit_shufps(jit_buf_t *b, int dst_xmm, int src_xmm, uint8_t imm)
{
    emit_sse_op(b, 0xC6, dst_xmm, src_xmm, 0, 0);
    jit_emit8(b, imm);
}

/* movss xmm, [base+disp] — load single float */
void jit_movss_load(jit_buf_t *b, int xmm, int base, int32_t disp)
{
    jit_emit8(b, 0xF3);
    emit_sse_op(b, 0x10, xmm, base, 1, disp);
}

/* movss [base+disp], xmm — store single float */
void jit_movss_store(jit_buf_t *b, int base, int32_t disp, int xmm)
{
    jit_emit8(b, 0xF3);
    emit_sse_op(b, 0x11, xmm, base, 1, disp);
}

/* movaps xmm, xmm — register copy (aligned) */
void jit_movaps_reg(jit_buf_t *b, int dst_xmm, int src_xmm)
{
    emit_sse_op(b, 0x28, dst_xmm, src_xmm, 0, 0);
}

/* Scalar SSE2: addss, mulss, maxss, subss, divss */
void jit_addss(jit_buf_t *b, int d, int s) { jit_emit8(b, 0xF3); emit_sse_op(b, 0x58, d, s, 0, 0); }
void jit_mulss(jit_buf_t *b, int d, int s) { jit_emit8(b, 0xF3); emit_sse_op(b, 0x59, d, s, 0, 0); }
void jit_maxss(jit_buf_t *b, int d, int s) { jit_emit8(b, 0xF3); emit_sse_op(b, 0x5F, d, s, 0, 0); }
void jit_subss(jit_buf_t *b, int d, int s) { jit_emit8(b, 0xF3); emit_sse_op(b, 0x5C, d, s, 0, 0); }
void jit_divss(jit_buf_t *b, int d, int s) { jit_emit8(b, 0xF3); emit_sse_op(b, 0x5E, d, s, 0, 0); }
void jit_addss_mem(jit_buf_t *b, int xmm, int base, int32_t disp)
{
    jit_emit8(b, 0xF3);
    emit_sse_op(b, 0x58, xmm, base, 1, disp);
}

/* =============================================================================
 * Additional Packed SSE2 Instructions
 * =============================================================================*/

void jit_subps(jit_buf_t *b, int d, int s) { emit_sse_op(b, 0x5C, d, s, 0, 0); }
void jit_divps(jit_buf_t *b, int d, int s) { emit_sse_op(b, 0x5E, d, s, 0, 0); }
void jit_andps(jit_buf_t *b, int d, int s) { emit_sse_op(b, 0x54, d, s, 0, 0); }
void jit_orps(jit_buf_t *b, int d, int s) { emit_sse_op(b, 0x56, d, s, 0, 0); }
void jit_andnps(jit_buf_t *b, int d, int s) { emit_sse_op(b, 0x55, d, s, 0, 0); }
void jit_minps(jit_buf_t *b, int d, int s) { emit_sse_op(b, 0x5D, d, s, 0, 0); }
void jit_rcpps(jit_buf_t *b, int d, int s) { emit_sse_op(b, 0x53, d, s, 0, 0); }
void jit_sqrtps(jit_buf_t *b, int d, int s) { emit_sse_op(b, 0x51, d, s, 0, 0); }
void jit_rsqrtps(jit_buf_t *b, int d, int s) { emit_sse_op(b, 0x52, d, s, 0, 0); }

/* cmpps xmm, xmm, imm8 — predicate: 0=eq,1=lt,2=le,3=unord,4=neq,5=nlt,6=nle */
void jit_cmpps(jit_buf_t *b, int d, int s, uint8_t pred)
{
    emit_sse_op(b, 0xC2, d, s, 0, 0);
    jit_emit8(b, pred);
}

/* Memory operand packed ops */
void jit_mulps_mem(jit_buf_t *b, int xmm, int base, int32_t disp)
{ emit_sse_op(b, 0x59, xmm, base, 1, disp); }
void jit_addps_mem(jit_buf_t *b, int xmm, int base, int32_t disp)
{ emit_sse_op(b, 0x58, xmm, base, 1, disp); }
void jit_subps_mem(jit_buf_t *b, int xmm, int base, int32_t disp)
{ emit_sse_op(b, 0x5C, xmm, base, 1, disp); }

/* =============================================================================
 * SSE2 Integer & Conversion Instructions
 * =============================================================================*/

/* cvtdq2ps xmm, xmm — int32 to float (packed) */
void jit_cvtdq2ps(jit_buf_t *b, int d, int s)
{ emit_sse_op(b, 0x5B, d, s, 0, 0); }

/* cvttps2dq xmm, xmm — float to int32 truncated (prefix F3) */
void jit_cvttps2dq(jit_buf_t *b, int d, int s)
{ jit_emit8(b, 0xF3); emit_sse_op(b, 0x5B, d, s, 0, 0); }

/* paddd xmm, xmm */
void jit_paddd(jit_buf_t *b, int d, int s)
{
    jit_emit8(b, 0x66);
    int need_rex = (d >= 8 || s >= 8);
    if (need_rex) emit_rex(b, 0, d >= 8, 0, s >= 8);
    jit_emit8(b, 0x0F); jit_emit8(b, 0xFE);
    emit_modrm(b, 3, d, s);
}

/* psubd xmm, xmm */
void jit_psubd(jit_buf_t *b, int d, int s)
{
    jit_emit8(b, 0x66);
    int need_rex = (d >= 8 || s >= 8);
    if (need_rex) emit_rex(b, 0, d >= 8, 0, s >= 8);
    jit_emit8(b, 0x0F); jit_emit8(b, 0xFA);
    emit_modrm(b, 3, d, s);
}

/* pslld xmm, imm8 — logical shift left int32 */
void jit_pslld_imm(jit_buf_t *b, int xmm, uint8_t imm)
{
    jit_emit8(b, 0x66);
    if (xmm >= 8) emit_rex(b, 0, 0, 0, 1);
    jit_emit8(b, 0x0F); jit_emit8(b, 0x72);
    emit_modrm(b, 3, 6, xmm); /* /6 = pslld */
    jit_emit8(b, imm);
}

/* psrld xmm, imm8 — logical shift right int32 */
void jit_psrld_imm(jit_buf_t *b, int xmm, uint8_t imm)
{
    jit_emit8(b, 0x66);
    if (xmm >= 8) emit_rex(b, 0, 0, 0, 1);
    jit_emit8(b, 0x0F); jit_emit8(b, 0x72);
    emit_modrm(b, 3, 2, xmm); /* /2 = psrld */
    jit_emit8(b, imm);
}

/* psrad xmm, imm8 — arithmetic shift right int32 */
void jit_psrad_imm(jit_buf_t *b, int xmm, uint8_t imm)
{
    jit_emit8(b, 0x66);
    if (xmm >= 8) emit_rex(b, 0, 0, 0, 1);
    jit_emit8(b, 0x0F); jit_emit8(b, 0x72);
    emit_modrm(b, 3, 4, xmm); /* /4 = psrad */
    jit_emit8(b, imm);
}

/* movd xmm, r32 */
void jit_movd_to_xmm(jit_buf_t *b, int xmm, int gpr)
{
    jit_emit8(b, 0x66);
    int need_rex = (xmm >= 8 || gpr >= 8);
    if (need_rex) emit_rex(b, 0, xmm >= 8, 0, gpr >= 8);
    jit_emit8(b, 0x0F); jit_emit8(b, 0x6E);
    emit_modrm(b, 3, xmm, gpr);
}

/* movd r32, xmm */
void jit_movd_from_xmm(jit_buf_t *b, int gpr, int xmm)
{
    jit_emit8(b, 0x66);
    int need_rex = (xmm >= 8 || gpr >= 8);
    if (need_rex) emit_rex(b, 0, xmm >= 8, 0, gpr >= 8);
    jit_emit8(b, 0x0F); jit_emit8(b, 0x7E);
    emit_modrm(b, 3, xmm, gpr);
}

/* =============================================================================
 * Additional Control Flow
 * =============================================================================*/

int jit_jle_fwd(jit_buf_t *b)
{ jit_emit8(b, 0x0F); jit_emit8(b, 0x8E); int p = b->len; jit_emit32(b, 0); return p; }

int jit_jne_fwd(jit_buf_t *b)
{ jit_emit8(b, 0x0F); jit_emit8(b, 0x85); int p = b->len; jit_emit32(b, 0); return p; }

int jit_je_fwd(jit_buf_t *b)
{ jit_emit8(b, 0x0F); jit_emit8(b, 0x84); int p = b->len; jit_emit32(b, 0); return p; }

void jit_jle_back(jit_buf_t *b, int t)
{ jit_emit8(b, 0x0F); jit_emit8(b, 0x8E); int32_t r = t - (b->len + 4); jit_emit32(b, (uint32_t)r); }

void jit_jne_back(jit_buf_t *b, int t)
{ jit_emit8(b, 0x0F); jit_emit8(b, 0x85); int32_t r = t - (b->len + 4); jit_emit32(b, (uint32_t)r); }

/* =============================================================================
 * Additional GP Instructions
 * =============================================================================*/

void jit_shl_reg_imm(jit_buf_t *b, int reg, uint8_t imm)
{ emit_rex_w(b, 0, reg); jit_emit8(b, 0xC1); emit_modrm(b, 3, 4, reg); jit_emit8(b, imm); }

void jit_shr_reg_imm(jit_buf_t *b, int reg, uint8_t imm)
{ emit_rex_w(b, 0, reg); jit_emit8(b, 0xC1); emit_modrm(b, 3, 5, reg); jit_emit8(b, imm); }

void jit_and_reg_imm32(jit_buf_t *b, int reg, int32_t imm)
{
    emit_rex_w(b, 0, reg);
    jit_emit8(b, 0x81); emit_modrm(b, 3, 4, reg); /* /4 = AND */
    jit_emit32(b, (uint32_t)imm);
}

void jit_or_reg_reg(jit_buf_t *b, int dst, int src)
{ emit_rex_w(b, src, dst); jit_emit8(b, 0x09); emit_modrm(b, 3, src, dst); }

void jit_sub_reg_reg(jit_buf_t *b, int dst, int src)
{ emit_rex_w(b, src, dst); jit_emit8(b, 0x29); emit_modrm(b, 3, src, dst); }

/* =============================================================================
 * Code Data Embedding
 * =============================================================================*/

/* Embed 4 floats in the code stream (16-byte aligned). Returns offset. */
int jit_embed_f32x4(jit_buf_t *b, float v0, float v1, float v2, float v3)
{
    /* Align to 16 bytes */
    while (b->len & 15) jit_emit8(b, 0xCC); /* int3 padding */
    int off = b->len;
    union { float f; uint32_t u; } u;
    u.f = v0; jit_emit32(b, u.u);
    u.f = v1; jit_emit32(b, u.u);
    u.f = v2; jit_emit32(b, u.u);
    u.f = v3; jit_emit32(b, u.u);
    return off;
}

/* movaps xmm, [rip + offset_to_data] — load from embedded constant */
void jit_movaps_riprel(jit_buf_t *b, int xmm, int data_offset)
{
    /* 0F 28 /r with ModR/M mod=00 rm=101 (RIP-relative) */
    if (xmm >= 8) emit_rex(b, 0, 1, 0, 0);
    jit_emit8(b, 0x0F); jit_emit8(b, 0x28);
    emit_modrm(b, 0, xmm, 5); /* mod=00, rm=101 = RIP-relative */
    int32_t rel = data_offset - (b->len + 4);
    jit_emit32(b, (uint32_t)rel);
}

/* =============================================================================
 * Function Prologue / Epilogue
 * =============================================================================*/

void jit_prologue(jit_buf_t *b)
{
    jit_push(b, RBP);
    jit_mov_reg_reg(b, RBP, RSP);
    jit_push(b, RBX);
    jit_push(b, R12);
    jit_push(b, R13);
    jit_push(b, R14);
    jit_push(b, R15);
#if defined(_WIN32) || defined(__MINGW32__) || defined(__MINGW64__)
    /* Windows x64 ABI: RSI and RDI are callee-saved (unlike System V).
     * Save them before we remap args, restore in epilogue. */
    jit_push(b, RSI);
    jit_push(b, RDI);
    /* 7 pushes + return addr = 8*8 = 64 → 16-aligned, but we have
     * push rbp + push rbx + push r12-r15 + push rsi + push rdi = 8
     * plus return addr = 9 * 8 = 72 → 8-mod-16. Need sub rsp,8 */
    jit_sub_reg_imm32(b, RSP, 8);
    /* Windows x64 ABI → System V ABI register mapping */
    jit_mov_reg_reg(b, RDI, RCX);
    jit_mov_reg_reg(b, RSI, RDX);
    jit_mov_reg_reg(b, RDX, R8);
    jit_mov_reg_reg(b, RCX, R9);
#else
    /* Align stack to 16 bytes (5 pushes + return addr = 6*8 = 48, need 16-align) */
    jit_sub_reg_imm32(b, RSP, 8);
#endif
}

void jit_epilogue(jit_buf_t *b)
{
#if defined(_WIN32) || defined(__MINGW32__) || defined(__MINGW64__)
    jit_add_reg_imm32(b, RSP, 8);
    jit_pop(b, RDI);
    jit_pop(b, RSI);
#else
    jit_add_reg_imm32(b, RSP, 8);
#endif
    jit_pop(b, R15);
    jit_pop(b, R14);
    jit_pop(b, R13);
    jit_pop(b, R12);
    jit_pop(b, RBX);
    jit_pop(b, RBP);
    jit_ret(b);
}

/* =============================================================================
 * High-Level JIT: Compile MatMul Kernel
 *
 * Generates: void matmul(float *C, float *A, float *B, int M, int N, int K)
 * System V ABI: C=rdi, A=rsi, B=rdx  (M, N, K are compile-time constants)
 *
 * Strategy: i-k-j order with SSE2 4-wide inner j loop.
 * All address computation uses simple pointer arithmetic (no SIB).
 *
 * Register allocation:
 *   R12 = C_base,  R13 = A_base,  R14 = B_base    (callee-saved, persist)
 *   R15 = i_counter,  RBX = k_counter              (callee-saved loops)
 *   RAX = A_row pointer (A + i*K*4)
 *   RCX = B_row pointer (B + k*N*4)
 *   RDX = C_row pointer (C + i*N*4)
 *   RDI = j_byte offset (inner loop)
 * =============================================================================*/

/* Cached compiled kernels */
#define JIT_MAX_KERNELS 16
static struct {
    int M, N, K;
    jit_matmul_fn fn;
    int code_size;
} jit_kernel_cache[JIT_MAX_KERNELS];
static int jit_num_kernels = 0;
static int jit_total_bytes = 0;

/* Helper: emit shl reg, imm8 */
static void jit_shl_imm(jit_buf_t *b, int reg, uint8_t imm)
{
    emit_rex_w(b, 0, reg);
    jit_emit8(b, 0xC1);
    emit_modrm(b, 3, 4, reg);  /* /4 = SHL */
    jit_emit8(b, imm);
}

/* Helper: emit imul dst, src, imm32 */
static void jit_imul_imm32(jit_buf_t *b, int dst, int src, int32_t imm)
{
    emit_rex_w(b, dst, src);
    jit_emit8(b, 0x69);
    emit_modrm(b, 3, dst, src);
    jit_emit32(b, (uint32_t)imm);
}

jit_matmul_fn jit_compile_matmul_kernel(int M, int N, int K)
{
    /* Check cache first */
    for (int i = 0; i < jit_num_kernels; i++) {
        if (jit_kernel_cache[i].M == M && jit_kernel_cache[i].N == N &&
            jit_kernel_cache[i].K == K)
            return jit_kernel_cache[i].fn;
    }

    int j_vecs = N / 4;
    if (j_vecs < 1) return NULL; /* N must be >= 4 and multiple of 4 */

    jit_buf_t *b = jit_create(4096);
    if (!b) return NULL;

    jit_prologue(b);

    /* Save base pointers to callee-saved regs */
    jit_mov_reg_reg(b, R12, RDI); /* C */
    jit_mov_reg_reg(b, R13, RSI); /* A */
    jit_mov_reg_reg(b, R14, RDX); /* B */

    /* ---- Step 1: Zero C[M*N] using xorps + movups ---- */
    jit_xorps(b, XMM7, XMM7);
    jit_mov_reg_reg(b, RAX, R12);   /* RAX = walk pointer through C */
    jit_xor_reg_reg(b, RCX, RCX);   /* RCX = counter */
    int ztop = b->len;
    jit_movups_store(b, RAX, 0, XMM7);
    jit_add_reg_imm32(b, RAX, 16);
    jit_inc_reg(b, RCX);
    jit_cmp_reg_imm32(b, RCX, (M * N + 3) / 4);
    jit_jl_back(b, ztop);

    /* ---- Step 2: Triple loop i-k-j ---- */
    /* for (i = 0; i < M; i++) */
    jit_xor_reg_reg(b, R15, R15);   /* i = 0 */
    int i_top = b->len;

    /*   Compute C_row = C + i * N * 4  (DX = C_row pointer, persists across k) */
    jit_mov_reg_reg(b, RDX, R15);
    jit_imul_imm32(b, RDX, RDX, N * 4);
    jit_add_reg_reg(b, RDX, R12);    /* RDX = &C[i * N] */

    /*   Compute A_row_base = A + i * K * 4 */
    jit_mov_reg_reg(b, RAX, R15);
    jit_imul_imm32(b, RAX, RAX, K * 4);
    jit_add_reg_reg(b, RAX, R13);    /* RAX = &A[i * K] */
    /* Save A_row to stack since we need RAX for scratch */
    jit_push(b, RAX);

    /*   for (k = 0; k < K; k++) */
    jit_xor_reg_reg(b, RBX, RBX);   /* k = 0 */
    int k_top = b->len;

    /*     Load A[i*K + k]: address = A_row + k*4 */
    /*     RAX = stack-saved A_row */
    jit_mov_reg_mem(b, RAX, RSP, 0); /* reload A_row from stack */
    /*     movss xmm0, [RAX + RBX*4]  --- but avoid SIB encoding.
     *     Instead: compute temp = A_row + k*4 in RSI */
    jit_mov_reg_reg(b, RSI, RBX);   /* RSI = k */
    jit_shl_imm(b, RSI, 2);         /* RSI = k * 4 */
    jit_add_reg_reg(b, RSI, RAX);   /* RSI = &A[i*K + k] */
    jit_movss_load(b, XMM0, RSI, 0); /* xmm0 = A[i*K+k] (scalar) */

    /*     Broadcast xmm0 to all 4 lanes */
    jit_shufps(b, XMM0, XMM0, 0x00);

    /*     Compute B_row = B + k * N * 4 */
    jit_mov_reg_reg(b, RCX, RBX);
    jit_imul_imm32(b, RCX, RCX, N * 4);
    jit_add_reg_reg(b, RCX, R14);   /* RCX = &B[k * N] */

    /*     Inner j loop using pointer increment (no SIB needed) */
    /*     RDI = j_byte_offset, starts at 0, increments by 16 */
    jit_xor_reg_reg(b, RDI, RDI);
    int j_top = b->len;

    /*       Load B[k*N + j..j+3]: movups xmm1, [RCX + RDI] */
    /*       Use: add RCX, RDI → load [RCX] → sub RCX, RDI
     *       Actually simpler: compute temp address in RSI */
    jit_mov_reg_reg(b, RSI, RCX);
    jit_add_reg_reg(b, RSI, RDI);   /* RSI = &B[k*N] + j_bytes */
    jit_movups_load(b, XMM1, RSI, 0);

    /*       Load C[i*N + j..j+3]: movups xmm2, [RDX + RDI] */
    jit_mov_reg_reg(b, RSI, RDX);
    jit_add_reg_reg(b, RSI, RDI);   /* RSI = &C[i*N] + j_bytes */
    jit_movups_load(b, XMM2, RSI, 0);

    /*       xmm2 += xmm0 * xmm1 */
    jit_mulps(b, XMM1, XMM0);
    jit_addps(b, XMM2, XMM1);

    /*       Store C[i*N + j..j+3] */
    jit_movups_store(b, RSI, 0, XMM2);

    /*       j_byte += 16 */
    jit_add_reg_imm32(b, RDI, 16);
    jit_cmp_reg_imm32(b, RDI, j_vecs * 16);
    jit_jl_back(b, j_top);

    /*   k++ */
    jit_inc_reg(b, RBX);
    jit_cmp_reg_imm32(b, RBX, K);
    jit_jl_back(b, k_top);

    /* Pop saved A_row */
    jit_pop(b, RAX);

    /* i++ */
    jit_inc_reg(b, R15);
    jit_cmp_reg_imm32(b, R15, M);
    jit_jl_back(b, i_top);

    jit_epilogue(b);

    /* Cache the compiled kernel */
    vmm_mark_rx(b->code, b->cap);
    jit_matmul_fn fn = (jit_matmul_fn)(void *)b->code;
    if (jit_num_kernels < JIT_MAX_KERNELS) {
        jit_kernel_cache[jit_num_kernels].M = M;
        jit_kernel_cache[jit_num_kernels].N = N;
        jit_kernel_cache[jit_num_kernels].K = K;
        jit_kernel_cache[jit_num_kernels].fn = fn;
        jit_kernel_cache[jit_num_kernels].code_size = b->len;
        jit_total_bytes += b->len;
        jit_num_kernels++;
    }

    /* Don't destroy buf — the code must stay alive! */
    return fn;
}

/* =============================================================================
 * High-Level JIT: Compile ReLU Kernel
 * void relu(float *out, float *in, int n)
 * Args: out=rdi, in=rsi, n=edx
 * =============================================================================*/

jit_unary_fn jit_compile_relu_kernel(int n)
{
    jit_buf_t *b = jit_create(1024);
    if (!b) return NULL;

    jit_prologue(b);

    /* R12 = out, R13 = in */
    jit_mov_reg_reg(b, R12, RDI);
    jit_mov_reg_reg(b, R13, RSI);

    /* xmm7 = {0,0,0,0} for max(0, x) */
    jit_xorps(b, XMM7, XMM7);

    int vecs = n / 4;
    jit_xor_reg_reg(b, RBX, RBX);  /* byte offset = 0 */

    if (vecs > 0) {
        int loop_top = b->len;

        /* Load 4 floats from in */
        jit_movups_load(b, XMM0, R13, 0);
        /* max(xmm0, xmm7) */
        jit_maxps(b, XMM0, XMM7);
        /* Store to out */
        jit_movups_store(b, R12, 0, XMM0);

        jit_add_reg_imm32(b, R12, 16);
        jit_add_reg_imm32(b, R13, 16);
        jit_inc_reg(b, RBX);
        jit_cmp_reg_imm32(b, RBX, vecs);
        jit_jl_back(b, loop_top);
    }

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_unary_fn fn = (jit_unary_fn)(void *)b->code;
    jit_total_bytes += b->len;
    jit_num_kernels++;
    return fn;
}

/* =============================================================================
 * High-Level JIT: Compile Q8_0 GEMV Kernel
 *
 * Generates: void gemv_q8(float *out, const void *weight, const float *x,
 *                         int rows, int cols)
 * System V ABI: out=rdi, weight=rsi, x=rdx  (rows, cols baked in as constants)
 *
 * For each output row:
 *   1. Walk Q8_0 blocks (34 bytes each: fp16 scale + 32 int8 values)
 *   2. For each block: load scale, process 32 int8×float products
 *   3. Accumulate into SSE2 v4f registers
 *   4. Horizontal sum + store result
 *
 * This is the HOT PATH for LLM inference — every GEMV in the transformer
 * forward pass goes through here when weights are Q8_0 quantized.
 * =============================================================================*/

/* Cache for Q8_0 GEMV kernels */
#define JIT_MAX_GEMV_KERNELS 16
static struct {
    int rows, cols;
    jit_gemv_q8_fn fn;
} jit_gemv_cache[JIT_MAX_GEMV_KERNELS];
static int jit_num_gemv = 0;

jit_gemv_q8_fn jit_compile_q8_gemv(int rows, int cols)
{
    /* Check cache */
    for (int i = 0; i < jit_num_gemv; i++) {
        if (jit_gemv_cache[i].rows == rows && jit_gemv_cache[i].cols == cols)
            return jit_gemv_cache[i].fn;
    }

    /* Q8_0: 32 elements per block, 34 bytes per block */
    int n_blocks = cols / 32;
    if (n_blocks < 1) return NULL;
    int row_bytes = n_blocks * 34;

    /* Estimate code size: ~50 bytes per block iteration + overhead */
    int code_est = rows * 32 + n_blocks * 64 + 256;
    if (code_est > 65536) code_est = 65536; /* Cap at 64K per kernel */

    jit_buf_t *b = jit_create(code_est);
    if (!b) return NULL;

    jit_prologue(b);

    /* Save base pointers:
     * R12 = out (float *), R13 = weight (void *), R14 = x (float *) */
    jit_mov_reg_reg(b, R12, RDI);  /* out */
    jit_mov_reg_reg(b, R13, RSI);  /* weight */
    jit_mov_reg_reg(b, R14, RDX);  /* x */

    /* Row loop: R15 = row counter */
    jit_xor_reg_reg(b, R15, R15);
    int row_top = b->len;

    /* Zero accumulator: xmm6 = {0,0,0,0} */
    jit_xorps(b, XMM6, XMM6);

    /* RCX = pointer to current weight row = R13 + R15 * row_bytes */
    jit_mov_reg_reg(b, RAX, R15);
    jit_imul_imm32(b, RAX, RAX, row_bytes);
    jit_add_reg_reg(b, RAX, R13);
    /* RAX = weight row pointer */

    /* RBX = block counter */
    jit_xor_reg_reg(b, RBX, RBX);
    int blk_top = b->len;

    /* --- Process one Q8_0 block (34 bytes) --- */
    /* Load fp16 scale from block[0..1], convert to float */
    /* movzx ecx, word [RAX + RBX*34]  — but we avoid SIB, compute address */
    jit_mov_reg_reg(b, RCX, RBX);
    jit_imul_imm32(b, RCX, RCX, 34);  /* RCX = block_idx * 34 */
    jit_add_reg_reg(b, RCX, RAX);     /* RCX = &block[block_idx] */

    /* Load fp16 scale at [RCX+0]:
     * We'll call our fp16_to_fp32 conversion inline:
     * For JIT, we use a simpler approach: load the 16-bit value,
     * convert using x87 or SSE2 bit manipulation.
     * Simpler: just use movss to load as int, then do the shift trick.
     * Actually, for bare-metal JIT, let's do the int→float and scale
     * by processing 4 bytes at a time from qs[2..33]. */

    /* Load scale bytes and convert FP16→FP32 using integer ops:
     * We use movzx to load the 16-bit fp16 value into a GPR,
     * then do the bit manipulation to convert to fp32,
     * then movd to XMM. This is complex, so instead we'll
     * process the entire block in float-safe way:
     * 
     * Faster approach: process 4 int8 values at a time,
     * convert to float, multiply by x[], accumulate.
     * Apply scale at end of block.
     */

    /* For simplicity, we load 4 int8 values at a time,
     * sign-extend using the punpcklbw/punpcklwd/psrad trick,
     * convert to float, multiply by x[j..j+3], accumulate.
     * 8 iterations of 4 = 32 elements per block. */

    /* RDI = x pointer offset for this block = R14 + block_idx * 32 * 4 */
    jit_mov_reg_reg(b, RDI, RBX);
    jit_imul_imm32(b, RDI, RDI, 128); /* block_idx * 32 floats * 4 bytes */
    jit_add_reg_reg(b, RDI, R14);    /* RDI = &x[block_idx * 32] */

    /* RSI = pointer to qs[0] = RCX + 2 (skip fp16 scale) */
    jit_lea(b, RSI, RCX, 2);

    /* Process 8 groups of 4 int8s using xmm0-xmm5 */
    /* xmm5 = block accumulator (separate from row acc xmm6) */
    jit_xorps(b, XMM5, XMM5);

    for (int g = 0; g < 8; g++) {
        /* movd xmm0, [RSI + g*4]  — load 4 int8 bytes */
        jit_emit8(b, 0x66); /* SSE2 prefix for integer ops */
        if (0) {} /* xmm0, RSI */
        {
            int need_rex = (RSI >= 8);
            if (need_rex) emit_rex(b, 0, 0, 0, RSI >= 8);
            jit_emit8(b, 0x0F);
            jit_emit8(b, 0x6E); /* movd xmm0, r/m32 */
            emit_modrm_disp(b, XMM0, RSI, g * 4);
        }

        /* punpcklbw xmm0, xmm0 — bytes to words (duplicated) */
        jit_emit8(b, 0x66);
        jit_emit8(b, 0x0F);
        jit_emit8(b, 0x60); /* punpcklbw */
        emit_modrm(b, 3, XMM0, XMM0);

        /* punpcklwd xmm0, xmm0 — words to dwords (duplicated) */
        jit_emit8(b, 0x66);
        jit_emit8(b, 0x0F);
        jit_emit8(b, 0x61); /* punpcklwd */
        emit_modrm(b, 3, XMM0, XMM0);

        /* psrad xmm0, 24 — arithmetic right shift to sign-extend bytes */
        jit_emit8(b, 0x66);
        jit_emit8(b, 0x0F);
        jit_emit8(b, 0x72); /* psrad xmm, imm8 */
        emit_modrm(b, 3, 4, XMM0); /* /4 for psrad */
        jit_emit8(b, 24);

        /* cvtdq2ps xmm0, xmm0 — int32 to float */
        jit_emit8(b, 0x0F);
        jit_emit8(b, 0x5B); /* cvtdq2ps */
        emit_modrm(b, 3, XMM0, XMM0);

        /* movups xmm1, [RDI + g*16] — load 4 floats from x */
        jit_movups_load(b, XMM1, RDI, g * 16);

        /* mulps xmm0, xmm1 */
        jit_mulps(b, XMM0, XMM1);

        /* addps xmm5, xmm0 — accumulate */
        jit_addps(b, XMM5, XMM0);
    }

    /* Now xmm5 has the unscaled dot for this block.
     * We need to multiply by the fp16 scale.
     * Load fp16 from [RCX+0], convert to float in xmm0, broadcast, mulps.
     *
     * FP16→FP32 in JIT: movzx eax, word [RCX]; then bit-shift convert.
     * sign = (eax >> 15) & 1
     * exp  = (eax >> 10) & 0x1F
     * mant = eax & 0x3FF
     * if exp != 0 && exp != 31: result = (sign<<31) | ((exp+112)<<23) | (mant<<13)
     *
     * For typical Q8_0 scales (positive, normal), this simplifies.
     * Let's emit the full conversion: */

    /* movzx eax, word [RCX] */
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0xB7); /* movzx r32, r/m16 */
    emit_modrm_disp(b, RAX, RCX, 0);

    /* Save the fp16 bits in EAX. Now convert to fp32 bits in EDX. */
    /* EDX = sign << 31 */
    jit_mov_reg_reg(b, RDX, RAX);
    jit_emit8(b, 0xC1); emit_modrm(b, 3, 5, RDX & 7); jit_emit8(b, 16); /* shr edx, 16 — but we need >> 15 & 1 << 31 */

    /* Actually, simpler approach: use the call to a helper.
     * Even simpler: since we're in JIT and performance here is
     * per-block (not per-element), use the fast fp16 approximation:
     * extract exp/mantissa, shift, OR together.
     * 
     * But this adds 20+ instructions per block for the conversion.
     * Alternative: precompute scale as float BEFORE the block loop.
     * 
     * Better idea: Apply scale at the end via a callback, or
     * just keep it simple and multiply by scale_float from C.
     * 
     * BEST approach: The JIT kernel accumulates int8*float products
     * without the scale, and the C wrapper multiplies by scale.
     * But this requires restructuring...
     *
     * For now, let's do a compact FP16→FP32 conversion: */

    /* Clear upper bits of EAX (already done by movzx) */
    /* Extract: sign in bit 15, exp in bits 14-10, mant in bits 9-0 */

    /* mov ecx_scratch, eax; and ecx_scratch, 0x7C00; shr ecx_scratch, 10 → exp */
    /* For the JIT, let's use a fast approximate conversion that works
     * for normal fp16 values (not denorms/inf/NaN, which are rare for scales):
     * fp32_bits = ((fp16 & 0x8000) << 16) | (((fp16 & 0x7C00) + 0x1C000) << 13) | ((fp16 & 0x03FF) << 13)
     * This works for normal FP16 values. */

    /* mov edx, eax; and edx, 0x8000; shl edx, 16 → sign bit at position 31 */
    jit_mov_reg_reg(b, RDX, RAX);
    /* and edx, 0x8000 */
    jit_emit8(b, 0x81); emit_modrm(b, 3, 4, RDX & 7); jit_emit32(b, 0x8000);
    /* shl edx, 16 */
    jit_emit8(b, 0xC1); emit_modrm(b, 3, 4, RDX & 7); jit_emit8(b, 16);

    /* mov esi_tmp, eax; and esi_tmp, 0x7C00; add esi_tmp, 0x1C000; shl esi_tmp, 13 */
    jit_push(b, RSI); /* save RSI */
    jit_mov_reg_reg(b, RSI, RAX);
    jit_emit8(b, 0x81); emit_modrm(b, 3, 4, RSI & 7); jit_emit32(b, 0x7C00);
    jit_add_reg_imm32(b, RSI, 0x1C000);
    /* shl esi, 13 */
    jit_emit8(b, 0xC1); emit_modrm(b, 3, 4, RSI & 7); jit_emit8(b, 13);
    /* or edx, esi */
    jit_emit8(b, 0x09); emit_modrm(b, 3, RSI & 7, RDX & 7);

    /* mov esi_tmp, eax; and esi_tmp, 0x03FF; shl esi_tmp, 13 */
    jit_mov_reg_reg(b, RSI, RAX);
    jit_emit8(b, 0x81); emit_modrm(b, 3, 4, RSI & 7); jit_emit32(b, 0x03FF);
    jit_emit8(b, 0xC1); emit_modrm(b, 3, 4, RSI & 7); jit_emit8(b, 13);
    /* or edx, esi */
    jit_emit8(b, 0x09); emit_modrm(b, 3, RSI & 7, RDX & 7);
    jit_pop(b, RSI); /* restore */

    /* EDX now has fp32 bits of the scale. Move to XMM0 and broadcast. */
    /* movd xmm0, edx */
    jit_emit8(b, 0x66);
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0x6E);
    emit_modrm(b, 3, XMM0, RDX & 7);

    /* shufps xmm0, xmm0, 0x00 — broadcast to all 4 lanes */
    jit_shufps(b, XMM0, XMM0, 0x00);

    /* mulps xmm5, xmm0 — scale the block accumulator */
    jit_mulps(b, XMM5, XMM0);

    /* addps xmm6, xmm5 — add to row accumulator */
    jit_addps(b, XMM6, XMM5);

    /* Next block */
    jit_inc_reg(b, RBX);
    jit_cmp_reg_imm32(b, RBX, n_blocks);
    jit_jl_back(b, blk_top);

    /* Horizontal sum of xmm6 → single float */
    /* movaps xmm0, xmm6; shufps xmm0, xmm6, 0x4E; addps xmm6, xmm0 */
    jit_movaps_reg(b, XMM0, XMM6);
    jit_shufps(b, XMM0, XMM6, 0x4E); /* swap high/low 64-bit halves */
    jit_addps(b, XMM6, XMM0);
    /* movaps xmm0, xmm6; shufps xmm0, xmm6, 0xB1; addss xmm6, xmm0 */
    jit_movaps_reg(b, XMM0, XMM6);
    jit_shufps(b, XMM0, XMM6, 0xB1); /* swap within 64-bit halves */
    jit_addss(b, XMM6, XMM0);

    /* Store result: out[row] = xmm6[0] */
    /* movss [R12 + R15*4], xmm6 — compute address without SIB */
    jit_mov_reg_reg(b, RAX, R15);
    jit_shl_imm(b, RAX, 2);        /* RAX = row * 4 */
    jit_add_reg_reg(b, RAX, R12);   /* RAX = &out[row] */
    jit_movss_store(b, RAX, 0, XMM6);

    /* Next row */
    jit_inc_reg(b, R15);
    jit_cmp_reg_imm32(b, R15, rows);
    jit_jl_back(b, row_top);

    jit_epilogue(b);

    vmm_mark_rx(b->code, b->cap);
    jit_gemv_q8_fn fn = (jit_gemv_q8_fn)(void *)b->code;
    if (jit_num_gemv < JIT_MAX_GEMV_KERNELS) {
        jit_gemv_cache[jit_num_gemv].rows = rows;
        jit_gemv_cache[jit_num_gemv].cols = cols;
        jit_gemv_cache[jit_num_gemv].fn = fn;
        jit_num_gemv++;
    }
    jit_total_bytes += b->len;
    jit_num_kernels++;

    return fn;
}

/* =============================================================================
 * High-Level JIT: SiLU Kernel
 * Implemented in llm_jit.c (mature fused path)
 * =============================================================================*/

/* =============================================================================
 * Fused MatMul + ReLU (eliminates intermediate memory write)
 * =============================================================================*/

jit_matmul_fn jit_compile_fused_matmul_relu(int M, int N, int K)
{
    /* First compile a regular matmul */
    jit_matmul_fn base = jit_compile_matmul_kernel(M, N, K);
    if (!base) return NULL;
    /* For now, return the base matmul — a full fused kernel would interleave
     * the relu into the innermost loop. TODO: implement true fusion. */
    return base;
}

/* =============================================================================
 * JIT Subsystem Init & Stats
 * =============================================================================*/

void jit_init(void)
{
    jit_num_kernels = 0;
    jit_total_bytes = 0;
    kprintf_debug("[JIT] x86_64 JIT compiler initialized\n");
}

int jit_kernel_count(void)  { return jit_num_kernels; }
int jit_code_bytes(void)    { return jit_total_bytes; }

/* =============================================================================
 * Self-Test: Compile a small matmul, execute, verify
 * =============================================================================*/

int jit_selftest(void)
{
    /* Test: 4×4 identity × B = B  (same test as tensor_cpu_selftest) */
    static float A[16] __attribute__((aligned(16)));
    static float B[16] __attribute__((aligned(16)));
    static float C[16] __attribute__((aligned(16)));

    /* Identity matrix */
    kmemset(A, 0, sizeof(A));
    A[0] = 1.0f; A[5] = 1.0f; A[10] = 1.0f; A[15] = 1.0f;

    /* Sequential values */
    for (int i = 0; i < 16; i++)
        B[i] = (float)(i + 1);
    kmemset(C, 0, sizeof(C));

    /* JIT compile a 4×4 matmul */
    jit_matmul_fn fn = jit_compile_matmul_kernel(4, 4, 4);
    if (!fn) return -1;

    /* Execute JIT-compiled code */
    fn(C, A, B, 4, 4, 4);

    /* Verify: C should equal B */
    for (int i = 0; i < 16; i++) {
        float diff = C[i] - B[i];
        if (diff > 0.01f || diff < -0.01f)
            return -(i + 2);  /* Return negative index for debugging */
    }

    return 0;  /* Success */
}

/* =============================================================================
 * AVX2 VEX Prefix Encoding
 *
 * VEX replaces legacy REX+opcode prefixes with a compact 2 or 3 byte prefix:
 *   2-byte: C5 [R~ vvvv L pp]            — for 0F-map, no REX.{X,B,W}
 *   3-byte: C4 [R~ X~ B~ mmmmm] [W vvvv L pp] — all maps, full REX support
 *
 * R~/X~/B~ are INVERTED REX bits.  vvvv is COMPLEMENTED NDS register.
 * L=0 -> 128-bit (XMM), L=1 -> 256-bit (YMM).
 * pp: 00=none, 01=66h, 10=F3h, 11=F2h.
 * mmmmm: 00001=0F, 00010=0F38, 00011=0F3A.
 * =============================================================================*/

/* 2-byte VEX prefix (map must be 0F, no REX.X/B/W) */
static void emit_vex2(jit_buf_t *b, int r_ext, int vvvv_reg, int L, int pp)
{
    jit_emit8(b, 0xC5);
    uint8_t byte1 = 0;
    byte1 |= (r_ext ? 0 : 0x80);         /* R~ = NOT(REX.R) */
    byte1 |= ((~vvvv_reg & 0xF) << 3);   /* complemented NDS */
    byte1 |= (L ? 0x04 : 0);
    byte1 |= (pp & 3);
    jit_emit8(b, byte1);
}

/* 3-byte VEX prefix (any map, full REX support) */
static void emit_vex3(jit_buf_t *b, int r_ext, int x_ext, int b_ext,
                       int map, int W, int vvvv_reg, int L, int pp)
{
    jit_emit8(b, 0xC4);
    uint8_t byte1 = 0;
    byte1 |= (r_ext ? 0 : 0x80);
    byte1 |= (x_ext ? 0 : 0x40);
    byte1 |= (b_ext ? 0 : 0x20);
    byte1 |= (map & 0x1F);
    jit_emit8(b, byte1);
    uint8_t byte2 = 0;
    byte2 |= (W ? 0x80 : 0);
    byte2 |= ((~vvvv_reg & 0xF) << 3);
    byte2 |= (L ? 0x04 : 0);
    byte2 |= (pp & 3);
    jit_emit8(b, byte2);
}

/* VEX prefix for 0F map: uses 2-byte form when possible */
static void emit_vex_0f(jit_buf_t *b, int reg, int rm, int vvvv, int L, int pp)
{
    int r_ext = (reg >= 8);
    int b_ext = (rm >= 8);
    if (!b_ext)
        emit_vex2(b, r_ext, vvvv, L, pp);
    else
        emit_vex3(b, r_ext, 0, b_ext, 1/*0F*/, 0, vvvv, L, pp);
}

/* VEX prefix for 0F38 map (always 3-byte) */
static void emit_vex_0f38(jit_buf_t *b, int reg, int rm, int vvvv, int L, int pp)
{
    emit_vex3(b, (reg >= 8), 0, (rm >= 8), 2/*0F38*/, 0, vvvv, L, pp);
}

/* VEX prefix for 0F3A map (always 3-byte) */
static void emit_vex_0f3a(jit_buf_t *b, int reg, int rm, int vvvv, int L, int pp)
{
    emit_vex3(b, (reg >= 8), 0, (rm >= 8), 3/*0F3A*/, 0, vvvv, L, pp);
}

/* =============================================================================
 * AVX2 VEX-Encoded Instruction Implementations
 * =============================================================================*/

void jit_vzeroupper(jit_buf_t *b)
{
    jit_emit8(b, 0xC5);
    jit_emit8(b, 0xF8);
    jit_emit8(b, 0x77);
}

/* vmovups ymm, [base+disp]: VEX.256.NP.0F.WIG 10 /r */
void jit_vmovups_load256(jit_buf_t *b, int ymm, int base, int32_t disp)
{
    emit_vex_0f(b, ymm, base, 0, 1/*256*/, 0/*NP*/);
    jit_emit8(b, 0x10);
    emit_modrm_disp(b, ymm, base, disp);
}

/* vmovups [base+disp], ymm: VEX.256.NP.0F.WIG 11 /r */
void jit_vmovups_store256(jit_buf_t *b, int base, int32_t disp, int ymm)
{
    emit_vex_0f(b, ymm, base, 0, 1, 0);
    jit_emit8(b, 0x11);
    emit_modrm_disp(b, ymm, base, disp);
}

/* vxorps ymm, ymm, ymm: VEX.NDS.256.NP.0F.WIG 57 /r */
void jit_vxorps256(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f(b, dst, src2, src1, 1, 0);
    jit_emit8(b, 0x57);
    emit_modrm(b, 3, dst, src2);
}

/* vaddps ymm, ymm, ymm: VEX.NDS.256.NP.0F.WIG 58 /r */
void jit_vaddps256(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f(b, dst, src2, src1, 1, 0);
    jit_emit8(b, 0x58);
    emit_modrm(b, 3, dst, src2);
}

/* vmulps ymm, ymm, ymm: VEX.NDS.256.NP.0F.WIG 59 /r */
void jit_vmulps256(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f(b, dst, src2, src1, 1, 0);
    jit_emit8(b, 0x59);
    emit_modrm(b, 3, dst, src2);
}

/* vbroadcastss ymm, [base+disp]: VEX.256.66.0F38.W0 18 /r */
void jit_vbroadcastss(jit_buf_t *b, int ymm, int base, int32_t disp)
{
    emit_vex_0f38(b, ymm, base, 0, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0x18);
    emit_modrm_disp(b, ymm, base, disp);
}

/* vbroadcastss ymm, xmm: VEX.256.66.0F38.W0 18 /r (AVX2 reg source) */
void jit_vbroadcastss_reg(jit_buf_t *b, int ymm_dst, int xmm_src)
{
    emit_vex_0f38(b, ymm_dst, xmm_src, 0, 1, 1);
    jit_emit8(b, 0x18);
    emit_modrm(b, 3, ymm_dst, xmm_src);
}

/* vpmovsxbd ymm, [base+disp]: VEX.256.66.0F38.WIG 21 /r */
void jit_vpmovsxbd256(jit_buf_t *b, int ymm, int base, int32_t disp)
{
    emit_vex_0f38(b, ymm, base, 0, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0x21);
    emit_modrm_disp(b, ymm, base, disp);
}

/* vcvtdq2ps ymm, ymm: VEX.256.NP.0F.WIG 5B /r */
void jit_vcvtdq2ps256(jit_buf_t *b, int dst, int src)
{
    emit_vex_0f(b, dst, src, 0, 1, 0);
    jit_emit8(b, 0x5B);
    emit_modrm(b, 3, dst, src);
}

/* vfmadd231ps ymm, ymm, ymm: VEX.NDS.256.66.0F38.W0 B8 /r */
void jit_vfmadd231ps256(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f38(b, dst, src2, src1, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0xB8);
    emit_modrm(b, 3, dst, src2);
}

/* vextractf128 xmm, ymm, imm8: VEX.256.66.0F3A.W0 19 /r ib */
void jit_vextractf128(jit_buf_t *b, int xmm_dst, int ymm_src, uint8_t imm)
{
    emit_vex_0f3a(b, ymm_src, xmm_dst, 0, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0x19);
    emit_modrm(b, 3, ymm_src, xmm_dst);
    jit_emit8(b, imm);
}

/* vaddps xmm, xmm, xmm: VEX.NDS.128.NP.0F.WIG 58 /r */
void jit_vaddps128(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f(b, dst, src2, src1, 0/*128*/, 0);
    jit_emit8(b, 0x58);
    emit_modrm(b, 3, dst, src2);
}

/* vhaddps xmm, xmm, xmm: VEX.NDS.128.F2.0F.WIG 7C /r */
void jit_vhaddps128(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f(b, dst, src2, src1, 0/*128*/, 3/*F2*/);
    jit_emit8(b, 0x7C);
    emit_modrm(b, 3, dst, src2);
}

/* vmovss [base+disp], xmm: VEX.LIG.F3.0F.WIG 11 /r */
void jit_vmovss_store_vex(jit_buf_t *b, int base, int32_t disp, int xmm)
{
    emit_vex_0f(b, xmm, base, 0, 0/*128*/, 2/*F3*/);
    jit_emit8(b, 0x11);
    emit_modrm_disp(b, xmm, base, disp);
}

/* vmovd xmm, r32: VEX.128.66.0F.W0 6E /r */
void jit_vmovd_to_xmm_vex(jit_buf_t *b, int xmm, int gpr)
{
    emit_vex_0f(b, xmm, gpr, 0, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0x6E);
    emit_modrm(b, 3, xmm, gpr);
}

/* prefetcht0 [base+disp]: 0F 18 /1 */
void jit_prefetcht0(jit_buf_t *b, int base, int32_t disp)
{
    if (base >= 8) emit_rex(b, 0, 0, 0, 1);
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0x18);
    emit_modrm_disp(b, 1, base, disp);
}

/* =============================================================================
 * AVX2 Integer SIMD Emitters (for Q4_0×Q8_0 integer dot product)
 * =============================================================================*/

/* vpand ymm, ymm, ymm: VEX.256.66.0F.WIG DB /r */
void jit_vpand256(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f(b, dst, src2, src1, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0xDB);
    emit_modrm(b, 3, dst, src2);
}

/* vpsrlw ymm, ymm, imm8: VEX.256.66.0F.WIG 71 /2 ib */
void jit_vpsrlw256_imm(jit_buf_t *b, int ymm, int src, uint8_t imm)
{
    emit_vex_0f(b, 2, src, ymm, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0x71);
    emit_modrm(b, 3, 2, src);
    jit_emit8(b, imm);
}

/* vpunpcklbw ymm, ymm, ymm: VEX.256.66.0F.WIG 60 /r */
void jit_vpunpcklbw256(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f(b, dst, src2, src1, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0x60);
    emit_modrm(b, 3, dst, src2);
}

/* vpunpckhbw ymm, ymm, ymm: VEX.256.66.0F.WIG 68 /r */
void jit_vpunpckhbw256(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f(b, dst, src2, src1, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0x68);
    emit_modrm(b, 3, dst, src2);
}

/* vpmaddubsw ymm, ymm, ymm: VEX.256.66.0F38.WIG 04 /r */
void jit_vpmaddubsw256(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f38(b, dst, src2, src1, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0x04);
    emit_modrm(b, 3, dst, src2);
}

/* vpmaddwd ymm, ymm, ymm: VEX.256.66.0F.WIG F5 /r */
void jit_vpmaddwd256(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f(b, dst, src2, src1, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0xF5);
    emit_modrm(b, 3, dst, src2);
}

/* vpaddd ymm, ymm, ymm: VEX.256.66.0F.WIG FE /r */
void jit_vpaddd256(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f(b, dst, src2, src1, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0xFE);
    emit_modrm(b, 3, dst, src2);
}

/* vmovdqu ymm, [base+disp]: VEX.256.F3.0F.WIG 6F /r */
void jit_vmovdqu_load256(jit_buf_t *b, int ymm, int base, int32_t disp)
{
    emit_vex_0f(b, ymm, base, 0, 1/*256*/, 2/*F3*/);
    jit_emit8(b, 0x6F);
    emit_modrm_disp(b, ymm, base, disp);
}

/* vmovdqu [base+disp], ymm: VEX.256.F3.0F.WIG 7F /r */
void jit_vmovdqu_store256(jit_buf_t *b, int base, int32_t disp, int ymm)
{
    emit_vex_0f(b, ymm, base, 0, 1/*256*/, 2/*F3*/);
    jit_emit8(b, 0x7F);
    emit_modrm_disp(b, ymm, base, disp);
}

/* vpxor ymm, ymm, ymm: VEX.256.66.0F.WIG EF /r */
void jit_vpxor256(jit_buf_t *b, int dst, int src1, int src2)
{
    emit_vex_0f(b, dst, src2, src1, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0xEF);
    emit_modrm(b, 3, dst, src2);
}

/* vcvtdq2ps ymm, ymm (just vpaddd for completeness, int32->float already exists) */

/* =============================================================================
 * AVX2+FMA Q8_0 GEMV Kernel Compiler
 *
 * Generates: void gemv(float *out, const void *weight, const float *x,
 *                      int rows, int cols)  [rows/cols baked in]
 *
 * Key optimizations over the compiler-generated AVX2 path:
 *  1. Loop trip counts baked as immediate constants
 *  2. Row byte stride baked into address computation
 *  3. 4-row batching with shared x loads (load once, reuse 4x)
 *  4. vfmadd231ps for single-cycle fused multiply-accumulate
 *  5. vpmovsxbd for 8 simultaneous int8->int32 conversions
 *  6. Branchless fp16->fp32 scale conversion inlined (7 ops)
 *  7. Row accumulators held in ymm8-11 across all blocks (1 hsum/row)
 *
 * Register allocation:
 *   R12=out, R13=weight, R14=x  (callee-saved, lifetime=entire kernel)
 *   R15=row counter  RBX=block counter
 *   RAX,RCX,RDX,RSI,RDI=scratch
 *   Stack: [RSP+0..31] = 4 weight row base pointers
 *   YMM0-3=x data  YMM4-7=weights  YMM8-11=accumulators  YMM12=scale
 * =============================================================================*/

static void emit_avx2_gemv_row_block(jit_buf_t *b, int stack_off, int ymm_acc)
{
    /* Load row base from stack, add block byte offset (in RAX) */
    jit_mov_reg_mem(b, RDI, RSP, stack_off);
    jit_add_reg_reg(b, RDI, RAX);

    /* FP16 -> FP32 scale (branchless, for normal values)
     * fp32 = ((h & 0x7FFF) << 13) + 0x38000000 | ((h & 0x8000) << 16) */
    jit_emit8(b, 0x0F); jit_emit8(b, 0xB7); /* movzx edx, word [rdi] */
    emit_modrm_disp(b, RDX, RDI, 0);

    jit_mov_reg_reg(b, RSI, RDX);
    jit_and_reg_imm32(b, RDX, 0x7FFF);
    jit_shl_reg_imm(b, RDX, 13);
    jit_add_reg_imm32(b, RDX, 0x38000000);
    jit_and_reg_imm32(b, RSI, 0x8000);
    jit_shl_reg_imm(b, RSI, 16);
    jit_or_reg_reg(b, RDX, RSI);

    jit_vmovd_to_xmm_vex(b, YMM12, RDX);
    jit_vbroadcastss_reg(b, YMM12, YMM12);

    /* vpmovsxbd: 8 int8 -> 8 int32, 4 times for 32 weights */
    jit_vpmovsxbd256(b, YMM4, RDI, 2);
    jit_vpmovsxbd256(b, YMM5, RDI, 10);
    jit_vpmovsxbd256(b, YMM6, RDI, 18);
    jit_vpmovsxbd256(b, YMM7, RDI, 26);

    jit_vcvtdq2ps256(b, YMM4, YMM4);
    jit_vcvtdq2ps256(b, YMM5, YMM5);
    jit_vcvtdq2ps256(b, YMM6, YMM6);
    jit_vcvtdq2ps256(b, YMM7, YMM7);

    /* weight * x */
    jit_vmulps256(b, YMM4, YMM4, YMM0);
    jit_vmulps256(b, YMM5, YMM5, YMM1);
    jit_vmulps256(b, YMM6, YMM6, YMM2);
    jit_vmulps256(b, YMM7, YMM7, YMM3);

    /* Reduce 4 products -> 1 */
    jit_vaddps256(b, YMM4, YMM4, YMM5);
    jit_vaddps256(b, YMM6, YMM6, YMM7);
    jit_vaddps256(b, YMM4, YMM4, YMM6);

    /* FMA: acc += product * scale */
    jit_vfmadd231ps256(b, ymm_acc, YMM4, YMM12);
}

static void emit_avx2_hsum_store(jit_buf_t *b, int ymm_acc, int tmp_xmm,
                                  int base, int32_t disp)
{
    jit_vextractf128(b, tmp_xmm, ymm_acc, 1);
    jit_vaddps128(b, ymm_acc, ymm_acc, tmp_xmm);
    jit_vhaddps128(b, ymm_acc, ymm_acc, ymm_acc);
    jit_vhaddps128(b, ymm_acc, ymm_acc, ymm_acc);
    jit_vmovss_store_vex(b, base, disp, ymm_acc);
}

#define JIT_MAX_GEMV_AVX2 16
static struct {
    int rows, cols;
    jit_gemv_q8_fn fn;
} jit_gemv_avx2_cache[JIT_MAX_GEMV_AVX2];
static int jit_num_gemv_avx2 = 0;

jit_gemv_q8_fn jit_compile_q8_gemv_avx2(int rows, int cols)
{
    for (int i = 0; i < jit_num_gemv_avx2; i++) {
        if (jit_gemv_avx2_cache[i].rows == rows &&
            jit_gemv_avx2_cache[i].cols == cols)
            return jit_gemv_avx2_cache[i].fn;
    }

    int nb = cols / 32;
    if (nb < 1) return NULL;
    int row_bytes = nb * 34;

    jit_buf_t *buf = jit_create(2048);
    if (!buf) return NULL;

    jit_prologue(buf);
    jit_sub_reg_imm32(buf, RSP, 32);

    jit_mov_reg_reg(buf, R12, RDI);  /* out */
    jit_mov_reg_reg(buf, R13, RSI);  /* weight */
    jit_mov_reg_reg(buf, R14, RDX);  /* x */

    /* === 4-row batched main loop === */
    jit_xor_reg_reg(buf, R15, R15);
    int row4_top = buf->len;

    jit_mov_reg_reg(buf, RAX, R15);
    jit_add_reg_imm32(buf, RAX, 3);
    jit_cmp_reg_imm32(buf, RAX, rows);
    int row4_exit = jit_jge_fwd(buf);

    /* 4 row base pointers on stack */
    jit_mov_reg_reg(buf, RAX, R15);
    jit_imul_imm32(buf, RAX, RAX, row_bytes);
    jit_add_reg_reg(buf, RAX, R13);
    jit_mov_mem_reg(buf, RSP, 0, RAX);
    jit_lea(buf, RCX, RAX, row_bytes);
    jit_mov_mem_reg(buf, RSP, 8, RCX);
    jit_lea(buf, RCX, RCX, row_bytes);
    jit_mov_mem_reg(buf, RSP, 16, RCX);
    jit_lea(buf, RCX, RCX, row_bytes);
    jit_mov_mem_reg(buf, RSP, 24, RCX);

    /* Zero accumulators */
    jit_vxorps256(buf, YMM8,  YMM8,  YMM8);
    jit_vxorps256(buf, YMM9,  YMM9,  YMM9);
    jit_vxorps256(buf, YMM10, YMM10, YMM10);
    jit_vxorps256(buf, YMM11, YMM11, YMM11);

    /* Block loop */
    jit_xor_reg_reg(buf, RBX, RBX);
    int blk_top = buf->len;

    jit_imul_imm32(buf, RAX, RBX, 34);

    /* Load x[b*32..+31] shared across 4 rows */
    jit_imul_imm32(buf, RCX, RBX, 128);
    jit_add_reg_reg(buf, RCX, R14);
    jit_vmovups_load256(buf, YMM0, RCX, 0);
    jit_vmovups_load256(buf, YMM1, RCX, 32);
    jit_vmovups_load256(buf, YMM2, RCX, 64);
    jit_vmovups_load256(buf, YMM3, RCX, 96);

    emit_avx2_gemv_row_block(buf, 0,  YMM8);
    emit_avx2_gemv_row_block(buf, 8,  YMM9);
    emit_avx2_gemv_row_block(buf, 16, YMM10);
    emit_avx2_gemv_row_block(buf, 24, YMM11);

    jit_inc_reg(buf, RBX);
    jit_cmp_reg_imm32(buf, RBX, nb);
    jit_jl_back(buf, blk_top);

    /* Horizontal reduce + store 4 results */
    jit_mov_reg_reg(buf, RAX, R15);
    jit_shl_reg_imm(buf, RAX, 2);
    jit_add_reg_reg(buf, RAX, R12);

    emit_avx2_hsum_store(buf, YMM8,  XMM13, RAX, 0);
    emit_avx2_hsum_store(buf, YMM9,  XMM13, RAX, 4);
    emit_avx2_hsum_store(buf, YMM10, XMM13, RAX, 8);
    emit_avx2_hsum_store(buf, YMM11, XMM13, RAX, 12);

    jit_add_reg_imm32(buf, R15, 4);
    jit_jmp_back(buf, row4_top);

    /* === Tail: single rows === */
    jit_patch_jump(buf, row4_exit);

    int tail_top = buf->len;
    jit_cmp_reg_imm32(buf, R15, rows);
    int tail_exit = jit_jge_fwd(buf);

    jit_mov_reg_reg(buf, RAX, R15);
    jit_imul_imm32(buf, RAX, RAX, row_bytes);
    jit_add_reg_reg(buf, RAX, R13);
    jit_mov_mem_reg(buf, RSP, 0, RAX);

    jit_vxorps256(buf, YMM8, YMM8, YMM8);

    jit_xor_reg_reg(buf, RBX, RBX);
    int tail_blk_top = buf->len;

    jit_imul_imm32(buf, RAX, RBX, 34);
    jit_imul_imm32(buf, RCX, RBX, 128);
    jit_add_reg_reg(buf, RCX, R14);
    jit_vmovups_load256(buf, YMM0, RCX, 0);
    jit_vmovups_load256(buf, YMM1, RCX, 32);
    jit_vmovups_load256(buf, YMM2, RCX, 64);
    jit_vmovups_load256(buf, YMM3, RCX, 96);

    emit_avx2_gemv_row_block(buf, 0, YMM8);

    jit_inc_reg(buf, RBX);
    jit_cmp_reg_imm32(buf, RBX, nb);
    jit_jl_back(buf, tail_blk_top);

    jit_mov_reg_reg(buf, RAX, R15);
    jit_shl_reg_imm(buf, RAX, 2);
    jit_add_reg_reg(buf, RAX, R12);
    emit_avx2_hsum_store(buf, YMM8, XMM13, RAX, 0);

    jit_inc_reg(buf, R15);
    jit_jmp_back(buf, tail_top);

    jit_patch_jump(buf, tail_exit);

    jit_vzeroupper(buf);
    jit_add_reg_imm32(buf, RSP, 32);
    jit_epilogue(buf);

    vmm_mark_rx(buf->code, buf->cap);
    jit_gemv_q8_fn fn = (jit_gemv_q8_fn)(void *)buf->code;
    if (jit_num_gemv_avx2 < JIT_MAX_GEMV_AVX2) {
        jit_gemv_avx2_cache[jit_num_gemv_avx2].rows = rows;
        jit_gemv_avx2_cache[jit_num_gemv_avx2].cols = cols;
        jit_gemv_avx2_cache[jit_num_gemv_avx2].fn = fn;
        jit_num_gemv_avx2++;
    }
    jit_total_bytes += buf->len;
    jit_num_kernels++;

    kprintf("[JIT] AVX2 GEMV %dx%d compiled (%d bytes)\n", rows, cols, buf->len);
    return fn;
}

/* =============================================================================
 * AVX2 Q4_0×Q8_0 Integer GEMV Kernel Compiler
 *
 * Generates: void gemv_q4q8(float *out, const void *q4_weight,
 *                           const void *q8_input, int rows, int cols)
 *   q8_input = array of {float d; int8_t qs[32]} blocks (36 bytes each)
 *   q4_weight = array of rows, each row = nb blocks of {uint16_t d; uint8_t qs[16]} (18 bytes)
 *
 * Algorithm per block (llama.cpp integer approach):
 *   1. Load 16 Q4 bytes → unpack to 32 unsigned nibbles [0,15]
 *   2. vpmaddubsw(q4_unsigned, q8_signed) → 16×int16
 *   3. vpmaddwd(result, all_ones) → 8×int32
 *   4. Accumulate int32 sum
 *   5. Also track xq_sum for bias correction: dot - 8*sum(xq)
 *   6. At end: float result = scale_w * scale_x * (isum - 8*xsum)
 * =============================================================================*/

/* Helper: emit Q4×Q8 block processing for one row.
 * Inputs: RDI = current block ptr of Q4 row, R9 = current Q8 block ptr
 * Uses: YMM0 (low nibbles), YMM1 (high nibbles), YMM2 (q8 bytes),
 *        YMM3 (ones constant, pre-loaded), YMM4 (0x0F mask, pre-loaded)
 * Outputs: adds block int32 sum to ymm_acc (int32 accumulator)
 *          adds block float contribution to ymm_facc (float accumulator)
 * Clobbers: YMM5, YMM6, YMM7, RDX, RSI */
static void emit_q4q8_row_block(jit_buf_t *b, int ymm_iacc, int ymm_facc)
{
    /* Load 16 Q4 bytes from [RDI+2] (skip 2-byte scale) into XMM5 low 128 bits,
     * then unpack nibbles to 32 unsigned bytes in YMM0.
     * Strategy: load 16 bytes, vpunpcklbw with shifted version to interleave,
     * then mask. Actually simpler: put same 16B in both halves, mask lo, shift+mask hi.
     *
     * Simpler approach: load 16B into low 128 of YMM5, copy to YMM6,
     * mask YMM5 with 0x0F → lo nibbles in low 128
     * shift YMM6 right by 4, mask with 0x0F → hi nibbles in low 128
     * We need [lo0..lo15, hi0..hi15] in one YMM register.
     * Use vinserti128 to combine them.
     *
     * Actually, even simpler: broadcast the 16 bytes to both halves,
     * then lo half: vpand with 0x0F → lo nibbles
     * hi half: shift+mask → hi nibbles
     * But we can't differentiate halves easily.
     *
     * SIMPLEST: scalar unpack to stack, load back. For a JIT kernel with baked
     * constants, the overhead is small vs the memory bandwidth bottleneck.
     * BUT: we're generating code, so let's do it properly in registers.
     *
     * Correct approach:
     * - vmovdqu xmm5, [rdi+2]  (16 bytes into low 128)
     * - vpand xmm0, xmm5, xmm4_lo128  (lo nibbles, 16 bytes, low lane)
     * - vpsrlw xmm1, xmm5, 4; vpand xmm1, xmm1, xmm4_lo128  (hi nibbles, 16 bytes, low lane)
     * - vinserti128 ymm0, ymm0, xmm1, 1  (combine: lo in low128, hi in hi128)
     * Now YMM0 has 32 unsigned bytes [lo0..lo15, hi0..hi15]
     *
     * For vinserti128: VEX.256.66.0F3A.W0 38 /r ib
     */

    /* Load 16 Q4 weight bytes from [RDI+2] — use 128-bit load into low half of YMM5 */
    /* vmovdqu xmm5, [rdi+2]: VEX.128.F3.0F.WIG 6F /r */
    emit_vex_0f(b, YMM5, RDI, 0, 0/*128*/, 2/*F3*/);
    jit_emit8(b, 0x6F);
    emit_modrm_disp(b, YMM5, RDI, 2);

    /* vpand xmm0, xmm5, xmm4 (lo nibbles): VEX.128.66.0F DB /r */
    emit_vex_0f(b, YMM0, YMM4 & 7, YMM5, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0xDB);
    emit_modrm(b, 3, YMM0, YMM4 & 7);

    /* vpsrlw xmm1, xmm5, 4 */
    emit_vex_0f(b, 2, YMM5, YMM1, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0x71);
    emit_modrm(b, 3, 2, YMM5);
    jit_emit8(b, 4);

    /* vpand xmm1, xmm1, xmm4 (hi nibbles) */
    emit_vex_0f(b, YMM1, YMM4 & 7, YMM1, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0xDB);
    emit_modrm(b, 3, YMM1, YMM4 & 7);

    /* vinserti128 ymm0, ymm0, xmm1, 1: VEX.256.66.0F3A.W0 38 /r 01 */
    emit_vex_0f3a(b, YMM0, YMM1, YMM0, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0x38);
    emit_modrm(b, 3, YMM0, YMM1);
    jit_emit8(b, 1);

    /* Now YMM0 = 32 unsigned Q4 nibbles [lo0..lo15, hi0..hi15] */

    /* Load 32 Q8 signed bytes from [R9+8] (skip 4B float d + 4B int32 isum) */
    /* vmovdqu ymm2, [r9+8] */
    jit_vmovdqu_load256(b, YMM2, R9, 8);

    /* vpmaddubsw ymm5, ymm0, ymm2 → 16×int16 (unsigned Q4 × signed Q8) */
    jit_vpmaddubsw256(b, YMM5, YMM0, YMM2);

    /* vpmaddwd ymm5, ymm5, ymm3 (all-ones) → 8×int32 */
    jit_vpmaddwd256(b, YMM5, YMM5, YMM3);

    /* vpaddd ymm_iacc, ymm_iacc, ymm5 — accumulate int32 partial sums */
    jit_vpaddd256(b, ymm_iacc, ymm_iacc, YMM5);

    /* We also need the bias correction: sum of Q8 bytes for this block.
     * Use vpmaddubsw with all-ones to sum bytes pairwise.
     * Actually, we need sum(xq[0..31]) as a scalar int.
     * For efficiency: vpmaddubsw(all_ones_unsigned, xq_signed) gives pair sums,
     * then vpmaddwd with int16 all-ones gives 8 int32 partial sums.
     * Or we can fold the bias into the final scalar calculation.
     *
     * For now: compute xq_sum separately per block and accumulate a float correction.
     * The scale product scale_w * scale_x * (-8 * xq_sum) adds per block.
     *
     * Actually, let's compute it: xq_sum = sum of 32 signed bytes.
     * vpmaddubsw(all_1_unsigned, signed_xq) = pairwise: 1*xq[0]+1*xq[1] as int16
     * Then vpmaddwd(result, all_1_int16) = sum pairs of int16 → 8 int32
     * This gives a vector of partial sums that we accumulate.
     */

    /* Load constant pointers — YMM14 is all-1-unsigned-bytes (pre-loaded) */
    /* vpmaddubsw ymm6, ymm14, ymm2 → pairwise byte sums as int16 */
    jit_vpmaddubsw256(b, YMM6, YMM14, YMM2);

    /* vpmaddwd ymm6, ymm6, ymm3 → 8×int32 partial sums of Q8 bytes */
    jit_vpmaddwd256(b, YMM6, YMM6, YMM3);

    /* Now ymm6 has 8 int32s whose total = sum of all 32 Q8 signed bytes this block */
    /* Multiply by -8: we'll do this as float later. For now accumulate int sum. */
    /* Actually, let's keep a separate int32 accumulator for xq_sums. */
    /* But we need per-block scale weighting... hmm. */

    /* REVISED APPROACH: compute the per-block float contribution directly.
     * result_block = scale_w * scale_x * (isum - 8 * xqsum)
     * = scale_w * scale_x * isum - 8 * scale_w * scale_x * xqsum
     *
     * Since we need per-block scale factors anyway, let's:
     * 1. Horizontal-sum the int32 iacc for THIS block (ymm5 value)
     * 2. Horizontal-sum the xqsum for THIS block (ymm6 value)
     * 3. dot_block = scale_w * scale_x * (isum_block - 8 * xqsum_block)
     * 4. Accumulate as float scalar
     *
     * But horizontal sums per block are expensive. Better approach:
     * Keep int32 accumulators across all blocks, then do ONE h-sum at end.
     * BUT: scale varies per block, so we can't just accumulate raw int32s.
     *
     * OK, llama.cpp approach: float result = sum over blocks of:
     *   fp16_to_f32(w->d) * xq->d * (isum_block - 8 * xqsum_block)
     * Each block has different scales. So we MUST convert to float per block.
     *
     * Efficient per-block approach:
     * 1. h-sum ymm5 (8 int32) → scalar isum
     * 2. h-sum ymm6 (8 int32) → scalar xqsum
     * 3. isum - 8*xqsum → scalar int
     * 4. cvt to float, multiply by scale_w * scale_x
     * 5. accumulate into float scalar
     */

    /* Horizontal sum of ymm5 into eax (isum) */
    /* vextracti128 xmm7, ymm5, 1 */
    emit_vex_0f3a(b, YMM7 & 7, YMM5, 0, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0x39);            /* VEXTRACTI128 */
    emit_modrm(b, 3, YMM7 & 7, YMM5);
    jit_emit8(b, 1);

    /* vpaddd xmm5, xmm5, xmm7 (128-bit) */
    emit_vex_0f(b, YMM5, YMM7 & 7, YMM5, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0xFE);
    emit_modrm(b, 3, YMM5, YMM7 & 7);

    /* pshufd xmm7, xmm5, 0x4E (swap hi64/lo64) */
    emit_vex_0f(b, YMM7, YMM5, 0, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0x70);
    emit_modrm(b, 3, YMM7, YMM5);
    jit_emit8(b, 0x4E);

    /* vpaddd xmm5, xmm5, xmm7 */
    emit_vex_0f(b, YMM5, YMM7, YMM5, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0xFE);
    emit_modrm(b, 3, YMM5, YMM7);

    /* pshufd xmm7, xmm5, 0x01 */
    emit_vex_0f(b, YMM7, YMM5, 0, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0x70);
    emit_modrm(b, 3, YMM7, YMM5);
    jit_emit8(b, 0x01);

    /* vpaddd xmm5, xmm5, xmm7 → xmm5[0] = total isum */
    emit_vex_0f(b, YMM5, YMM7, YMM5, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0xFE);
    emit_modrm(b, 3, YMM5, YMM7);

    /* vmovd edx, xmm5 → isum in EDX */
    jit_movd_from_xmm(b, RDX, YMM5);

    /* Same horizontal sum for ymm6 → xqsum in ESI */
    /* vextracti128 xmm7, ymm6, 1 */
    emit_vex_0f3a(b, YMM7 & 7, YMM6, 0, 1/*256*/, 1/*66*/);
    jit_emit8(b, 0x39);
    emit_modrm(b, 3, YMM7 & 7, YMM6);
    jit_emit8(b, 1);

    /* vpaddd xmm6, xmm6, xmm7 */
    emit_vex_0f(b, YMM6, YMM7, YMM6, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0xFE);
    emit_modrm(b, 3, YMM6, YMM7);

    emit_vex_0f(b, YMM7, YMM6, 0, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0x70);
    emit_modrm(b, 3, YMM7, YMM6);
    jit_emit8(b, 0x4E);

    emit_vex_0f(b, YMM6, YMM7, YMM6, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0xFE);
    emit_modrm(b, 3, YMM6, YMM7);

    emit_vex_0f(b, YMM7, YMM6, 0, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0x70);
    emit_modrm(b, 3, YMM7, YMM6);
    jit_emit8(b, 0x01);

    emit_vex_0f(b, YMM6, YMM7, YMM6, 0/*128*/, 1/*66*/);
    jit_emit8(b, 0xFE);
    emit_modrm(b, 3, YMM6, YMM7);

    /* vmovd esi, xmm6 → xqsum in ESI */
    jit_movd_from_xmm(b, RSI, YMM6);

    /* EDX = isum, ESI = xqsum.  Compute isum - 8*xqsum */
    jit_shl_reg_imm(b, RSI, 3);      /* ESI *= 8 */
    jit_sub_reg_reg(b, RDX, RSI);     /* EDX = isum - 8*xqsum */

    /* Convert to float: vcvtsi2ss xmm5, xmm5, edx */
    /* VEX.LIG.F3.0F.W0 2A /r */
    emit_vex_0f(b, YMM5, RDX, YMM5, 0, 2/*F3*/);
    jit_emit8(b, 0x2A);
    emit_modrm(b, 3, YMM5, RDX);

    /* Load Q4 scale: fp16 at [RDI+0] → float */
    /* movzx edx, word [rdi] */
    jit_emit8(b, 0x0F); jit_emit8(b, 0xB7);
    emit_modrm_disp(b, RDX, RDI, 0);

    /* Branchless fp16→fp32 */
    jit_mov_reg_reg(b, RSI, RDX);
    jit_and_reg_imm32(b, RDX, 0x7FFF);
    jit_shl_reg_imm(b, RDX, 13);
    jit_add_reg_imm32(b, RDX, 0x38000000);
    jit_and_reg_imm32(b, RSI, 0x8000);
    jit_shl_reg_imm(b, RSI, 16);
    jit_or_reg_reg(b, RDX, RSI);

    /* vmovd xmm6, edx → fp32 weight scale */
    jit_vmovd_to_xmm_vex(b, YMM6, RDX);

    /* Load Q8 scale: float at [R9+0] */
    /* vmovss xmm7, [r9+0] */
    jit_movss_load(b, YMM7, R9, 0);

    /* xmm6 = scale_w * scale_x */
    jit_mulss(b, YMM6, YMM7);

    /* xmm5 = (isum - 8*xqsum) as float, xmm6 = scale_w * scale_x */
    jit_mulss(b, YMM5, YMM6);

    /* Accumulate into ymm_facc (scalar float in lane 0) */
    jit_addss(b, ymm_facc, YMM5);
}

/* Q4×Q8 GEMV JIT cache */
#define JIT_MAX_GEMV_Q4Q8 16
static struct {
    int rows, cols;
    jit_gemv_q8_fn fn;
} jit_gemv_q4q8_cache[JIT_MAX_GEMV_Q4Q8];
static int jit_num_gemv_q4q8 = 0;

/* Compile a Q4_0×Q8_0 integer GEMV kernel.
 * Signature: void gemv(float *out, const void *q4_weight, const void *q8_input,
 *                      int rows, int cols)
 * q8_input is a q8_input_t array: {float d; int8_t qs[32]} = 36 bytes per block
 * q4_weight rows: nb blocks of {uint16_t d; uint8_t qs[16]} = 18 bytes per block */
jit_gemv_q8_fn jit_compile_q4_q8_gemv_avx2(int rows, int cols)
{
    /* Cache check */
    for (int i = 0; i < jit_num_gemv_q4q8; i++) {
        if (jit_gemv_q4q8_cache[i].rows == rows &&
            jit_gemv_q4q8_cache[i].cols == cols)
            return jit_gemv_q4q8_cache[i].fn;
    }

    int nb = cols / 32;
    if (nb < 1) return NULL;
    int q4_row_bytes = nb * 18;       /* Q4_0: 18 bytes per block */
    int q8_block_bytes = 40;          /* q8_input_t: 4B float d + 4B int32 isum + 32B qs = 40 */

    /* This kernel is large due to per-block scalar h-sum. Allocate generously. */
    jit_buf_t *buf = jit_create(16384);
    if (!buf) return NULL;

    jit_prologue(buf);
    jit_sub_reg_imm32(buf, RSP, 8);   /* align stack */

    /* Save args: R12=out, R13=q4_weight, R14=q8_input */
    jit_mov_reg_reg(buf, R12, RDI);
    jit_mov_reg_reg(buf, R13, RSI);
    jit_mov_reg_reg(buf, R14, RDX);

    /* Pre-load constants into callee-usable YMMs */
    /* YMM3 = all int16 ones {1,1,...,1} for vpmaddwd */
    /* YMM4 = all bytes 0x0F for nibble masking (low 128 only needed) */
    /* YMM14 = all bytes 0x01 for xqsum calculation */

    /* Build 0x0F mask: load 0x0F0F0F0F into EDX, broadcast */
    jit_mov_reg_imm64(buf, RAX, 0x0F0F0F0F0F0F0F0FULL);
    jit_vmovd_to_xmm_vex(buf, YMM4, RAX);
    /* Need to broadcast to full xmm. Use vpbroadcastq or just movq+unpack.
     * Simpler: push to stack, load 128-bit. */
    jit_push(buf, RAX);
    jit_push(buf, RAX);
    /* vmovdqu xmm4, [rsp] */
    emit_vex_0f(buf, YMM4, RSP, 0, 0/*128*/, 2/*F3*/);
    jit_emit8(buf, 0x6F);
    emit_modrm_disp(buf, YMM4, RSP, 0);
    jit_add_reg_imm32(buf, RSP, 16);

    /* Build all-ones int16 {1,1,...,1}: 0x0001 repeated 16 times */
    jit_mov_reg_imm64(buf, RAX, 0x0001000100010001ULL);
    jit_push(buf, RAX);
    jit_push(buf, RAX);
    jit_push(buf, RAX);
    jit_push(buf, RAX);
    /* vmovdqu ymm3, [rsp] */
    jit_vmovdqu_load256(buf, YMM3, RSP, 0);
    jit_add_reg_imm32(buf, RSP, 32);

    /* Build all-ones unsigned bytes for xqsum: 0x01 repeated 32 */
    jit_mov_reg_imm64(buf, RAX, 0x0101010101010101ULL);
    jit_push(buf, RAX);
    jit_push(buf, RAX);
    jit_push(buf, RAX);
    jit_push(buf, RAX);
    jit_vmovdqu_load256(buf, YMM14, RSP, 0);
    jit_add_reg_imm32(buf, RSP, 32);

    /* === Row loop === */
    jit_xor_reg_reg(buf, R15, R15);   /* row = 0 */

    int row_top = buf->len;
    jit_cmp_reg_imm32(buf, R15, rows);
    int row_exit = jit_jge_fwd(buf);

    /* Compute Q4 row pointer: RCX = weight + row * q4_row_bytes */
    jit_mov_reg_reg(buf, RAX, R15);
    jit_imul_imm32(buf, RAX, RAX, q4_row_bytes);
    jit_add_reg_reg(buf, RAX, R13);
    /* Store row base in RDI for the block helper */

    /* Zero float accumulator for this row (scalar in XMM8 lane 0) */
    jit_xorps(buf, YMM8, YMM8);

    /* Block loop */
    jit_xor_reg_reg(buf, RBX, RBX);   /* block = 0 */
    jit_mov_reg_reg(buf, RCX, RAX);    /* RCX = Q4 row base */

    int blk_top = buf->len;

    /* RDI = Q4 block ptr = row_base + block * 18 */
    jit_mov_reg_reg(buf, RDI, RCX);
    jit_mov_reg_reg(buf, RAX, RBX);
    jit_imul_imm32(buf, RAX, RAX, 18);
    jit_add_reg_reg(buf, RDI, RAX);

    /* R9 = Q8 block ptr = q8_input + block * 36 */
    jit_mov_reg_reg(buf, R9, R14);
    jit_imul_imm32(buf, RAX, RBX, q8_block_bytes);
    jit_add_reg_reg(buf, R9, RAX);

    /* Process one block */
    emit_q4q8_row_block(buf, YMM15, YMM8);

    jit_inc_reg(buf, RBX);
    jit_cmp_reg_imm32(buf, RBX, nb);
    jit_jl_back(buf, blk_top);

    /* Store result: out[row] = xmm8 lane 0 */
    jit_mov_reg_reg(buf, RAX, R15);
    jit_shl_reg_imm(buf, RAX, 2);     /* row * 4 */
    jit_add_reg_reg(buf, RAX, R12);    /* &out[row] */
    jit_vmovss_store_vex(buf, RAX, 0, YMM8);

    jit_inc_reg(buf, R15);
    jit_jmp_back(buf, row_top);

    jit_patch_jump(buf, row_exit);

    jit_vzeroupper(buf);
    jit_add_reg_imm32(buf, RSP, 8);
    jit_epilogue(buf);

    vmm_mark_rx(buf->code, buf->cap);
    jit_gemv_q8_fn fn = (jit_gemv_q8_fn)(void *)buf->code;
    if (jit_num_gemv_q4q8 < JIT_MAX_GEMV_Q4Q8) {
        jit_gemv_q4q8_cache[jit_num_gemv_q4q8].rows = rows;
        jit_gemv_q4q8_cache[jit_num_gemv_q4q8].cols = cols;
        jit_gemv_q4q8_cache[jit_num_gemv_q4q8].fn = fn;
        jit_num_gemv_q4q8++;
    }
    jit_total_bytes += buf->len;
    jit_num_kernels++;

    kprintf("[JIT] AVX2 Q4xQ8 GEMV %dx%d compiled (%d bytes)\n", rows, cols, buf->len);
    return fn;
}

#else /* __aarch64__ */

/* ARM64: JIT stubs — JIT not available, runtime uses interpreter path */
#include "runtime/jit/x86_jit.h"
#include "kernel/core/kernel.h"
#include <stddef.h>

jit_buf_t *jit_create(int cap) { (void)cap; return NULL; }
void jit_destroy(jit_buf_t *b) { (void)b; }
void jit_prologue(jit_buf_t *b) { (void)b; }
void jit_epilogue(jit_buf_t *b) { (void)b; }
void jit_init(void) { kprintf("[JIT] ARM64 mode: using NEON interpreter\n"); }
int  jit_selftest(void) { return 0; }
jit_gemv_q8_fn jit_compile_q8_gemv(int r, int c) { (void)r; (void)c; return NULL; }
jit_gemv_q8_fn jit_compile_q8_gemv_avx2(int r, int c) { (void)r; (void)c; return NULL; }
jit_gemv_q8_fn jit_compile_q4_q8_gemv_avx2(int r, int c) { (void)r; (void)c; return NULL; }
jit_silu_fn jit_compile_silu_kernel(int n) { (void)n; return NULL; }
int jit_kernel_count(void) { return 0; }
int jit_code_bytes(void) { return 0; }

#endif /* __aarch64__ */
