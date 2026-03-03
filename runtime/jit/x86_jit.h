/* =============================================================================
 * TensorOS - x86_64 JIT Compiler
 *
 * Generates native x86_64 machine code at runtime for tensor operations.
 * Emits SSE2-vectorized kernels specialized for specific dimensions.
 * All generated code follows System V AMD64 calling convention.
 *
 * The JIT compiles tensor computation graphs into optimized native functions
 * that eliminate dispatch overhead and enable operator fusion.
 * =============================================================================*/

#ifndef TENSOROS_X86_JIT_H
#define TENSOROS_X86_JIT_H

#include <stdint.h>
#include <stddef.h>

/* =============================================================================
 * x86_64 Register IDs
 * =============================================================================*/

enum x86_gpr {
    RAX = 0, RCX = 1, RDX = 2, RBX = 3,
    RSP = 4, RBP = 5, RSI = 6, RDI = 7,
    R8  = 8, R9  = 9, R10 = 10, R11 = 11,
    R12 = 12, R13 = 13, R14 = 14, R15 = 15
};

/* XMM register IDs (same numbering) */
#define XMM0  0
#define XMM1  1
#define XMM2  2
#define XMM3  3
#define XMM4  4
#define XMM5  5
#define XMM6  6
#define XMM7  7

/* =============================================================================
 * JIT Code Buffer
 * =============================================================================*/

#define JIT_DEFAULT_CAP  8192
#define JIT_MAX_CAP      (1 << 20)  /* 1 MB max */

typedef struct {
    uint8_t *code;      /* Executable code buffer */
    int      len;       /* Current code length */
    int      cap;       /* Buffer capacity */
} jit_buf_t;

/* =============================================================================
 * Function pointer types for JIT-compiled kernels
 * =============================================================================*/

typedef void (*jit_void_fn)(void);
typedef void (*jit_matmul_fn)(float *C, const float *A, const float *B,
                               int M, int N, int K);
typedef void (*jit_unary_fn)(float *out, const float *in, int n);
typedef void (*jit_binary_fn)(float *out, const float *a, const float *b, int n);

/* GEMV: out[rows] = W[rows × cols_quant] · x[cols], W is Q8_0 blocked */
typedef void (*jit_gemv_q8_fn)(float *out, const void *weight, const float *x,
                                int rows, int cols);

/* Fused SiLU: x[i] = x[i] * sigmoid(x[i]) */
typedef void (*jit_silu_fn)(float *x, int n);

/* RMSNorm: out[dim] = normalize(x) * weight */
typedef void (*jit_rmsnorm_fn)(float *out, const float *x, const float *w, int dim);

/* =============================================================================
 * Buffer Management
 * =============================================================================*/

jit_buf_t *jit_create(int capacity);
void       jit_destroy(jit_buf_t *buf);
void       jit_reset(jit_buf_t *buf);

/* Finalize: returns function pointer to generated code */
jit_void_fn jit_get_fn(jit_buf_t *buf);

/* =============================================================================
 * Low-Level Byte Emission
 * =============================================================================*/

void jit_emit8(jit_buf_t *b, uint8_t v);
void jit_emit16(jit_buf_t *b, uint16_t v);
void jit_emit32(jit_buf_t *b, uint32_t v);
void jit_emit64(jit_buf_t *b, uint64_t v);

/* =============================================================================
 * x86_64 Instruction Emission
 * =============================================================================*/

/* Function frame */
void jit_prologue(jit_buf_t *b);
void jit_epilogue(jit_buf_t *b);

/* GP register operations */
void jit_push(jit_buf_t *b, int reg);
void jit_pop(jit_buf_t *b, int reg);
void jit_ret(jit_buf_t *b);
void jit_mov_reg_reg(jit_buf_t *b, int dst, int src);
void jit_mov_reg_imm64(jit_buf_t *b, int reg, uint64_t imm);
void jit_mov_reg_mem(jit_buf_t *b, int dst, int base, int32_t disp);
void jit_mov_mem_reg(jit_buf_t *b, int base, int32_t disp, int src);
void jit_lea(jit_buf_t *b, int dst, int base, int32_t disp);
void jit_add_reg_reg(jit_buf_t *b, int dst, int src);
void jit_add_reg_imm32(jit_buf_t *b, int dst, int32_t imm);
void jit_sub_reg_imm32(jit_buf_t *b, int dst, int32_t imm);
void jit_imul_reg_reg(jit_buf_t *b, int dst, int src);
void jit_cmp_reg_reg(jit_buf_t *b, int a, int reg_b);
void jit_cmp_reg_imm32(jit_buf_t *b, int reg, int32_t imm);
void jit_xor_reg_reg(jit_buf_t *b, int dst, int src);
void jit_inc_reg(jit_buf_t *b, int reg);

/* Control flow */
void jit_call_abs(jit_buf_t *b, uint64_t addr);
int  jit_jmp_fwd(jit_buf_t *b);
int  jit_jl_fwd(jit_buf_t *b);
int  jit_jge_fwd(jit_buf_t *b);
void jit_patch_jump(jit_buf_t *b, int patch_offset);
void jit_jmp_back(jit_buf_t *b, int target_offset);
void jit_jl_back(jit_buf_t *b, int target_offset);
void jit_jge_back(jit_buf_t *b, int target_offset);

/* SSE2 packed float instructions */
void jit_movups_load(jit_buf_t *b, int xmm, int base, int32_t disp);
void jit_movups_store(jit_buf_t *b, int base, int32_t disp, int xmm);
void jit_movaps_load(jit_buf_t *b, int xmm, int base, int32_t disp);
void jit_movaps_store(jit_buf_t *b, int base, int32_t disp, int xmm);
void jit_addps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_mulps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_xorps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_maxps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_shufps(jit_buf_t *b, int dst_xmm, int src_xmm, uint8_t imm);
void jit_movss_load(jit_buf_t *b, int xmm, int base, int32_t disp);
void jit_movss_store(jit_buf_t *b, int base, int32_t disp, int xmm);
void jit_movaps_reg(jit_buf_t *b, int dst_xmm, int src_xmm);

/* Scalar SSE2 instructions */
void jit_addss(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_mulss(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_maxss(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_addss_mem(jit_buf_t *b, int xmm, int base, int32_t disp);

/* =============================================================================
 * High-Level JIT Compilation
 * =============================================================================*/

/* Initialize JIT subsystem (called during boot) */
void jit_init(void);

/* Compile optimized matmul kernel for given dimensions */
jit_matmul_fn jit_compile_matmul_kernel(int M, int N, int K);

/* Compile vectorized relu kernel */
jit_unary_fn jit_compile_relu_kernel(int n);

/* Compile fused matmul+relu (single pass, no intermediate storage) */
jit_matmul_fn jit_compile_fused_matmul_relu(int M, int N, int K);

/* Compile Q8_0 GEMV kernel specialized for given dimensions */
jit_gemv_q8_fn jit_compile_q8_gemv(int rows, int cols);

/* Compile vectorized SiLU kernel for given size */
jit_silu_fn jit_compile_silu_kernel(int n);

/* Get number of JIT-compiled kernels cached */
int jit_kernel_count(void);

/* Get total bytes of JIT-compiled code */
int jit_code_bytes(void);

/* Self-test: compile, execute, verify */
int jit_selftest(void);

#endif /* TENSOROS_X86_JIT_H */
