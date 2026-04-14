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
#define XMM8  8
#define XMM9  9
#define XMM10 10
#define XMM11 11
#define XMM12 12
#define XMM13 13
#define XMM14 14
#define XMM15 15

/* YMM register IDs (AVX2 256-bit, same encoding as XMM) */
#define YMM0  0
#define YMM1  1
#define YMM2  2
#define YMM3  3
#define YMM4  4
#define YMM5  5
#define YMM6  6
#define YMM7  7
#define YMM8  8
#define YMM9  9
#define YMM10 10
#define YMM11 11
#define YMM12 12
#define YMM13 13
#define YMM14 14
#define YMM15 15

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

/* Softmax: in-place softmax over n elements */
typedef void (*jit_softmax_fn)(float *x, int n);

/* RoPE: rotary position encoding for one head vector */
typedef void (*jit_rope_fn)(float *vec, int pos, int head_dim, const float *freqs);

/* Dot product: returns sum(a[i]*b[i]) */
typedef float (*jit_dot_fn)(const float *a, const float *b, int n);

/* Element-wise ops */
typedef void (*jit_ewise_fn)(float *dst, const float *src, int n);

/* Scaled add: dst += scale * src */
typedef void (*jit_axpy_fn)(float *dst, float scale, const float *src, int n);

/* LayerNorm: out = normalize(x) * w + b */
typedef void (*jit_layernorm_fn)(float *out, const float *x, const float *w,
                                  const float *b, int dim);

/* Fused SiLU*multiply: gate[i] = SiLU(gate[i]) * up[i] */
typedef void (*jit_fused_silu_mul_fn)(float *gate, const float *up, int n);

/* Tokenizer/detokenizer helpers */
typedef int (*jit_mem_equal_fn)(const char *a, const char *b, int len);
typedef int (*jit_bpe_escape_scan_fn)(const char *s, int len);
typedef int (*jit_find_byte_fn)(const char *s, int len, char needle);

/* BPE merge candidate comparison: returns first index where a[i] > b[i], or -1 */
typedef int (*jit_merge_cmp_fn)(const int32_t *scores_a, const int32_t *scores_b, int n);

/* Token append with bounds check: returns new length or -1 if at capacity */
typedef int (*jit_token_append_fn)(int32_t *ids, int cur_len, int max_len,
                                    int32_t token_id);

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
void jit_subss(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_divss(jit_buf_t *b, int dst_xmm, int src_xmm);

/* Additional packed SSE2 instructions */
void jit_subps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_divps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_andps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_orps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_andnps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_minps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_rcpps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_sqrtps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_rsqrtps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_cmpps(jit_buf_t *b, int dst_xmm, int src_xmm, uint8_t pred);
void jit_mulps_mem(jit_buf_t *b, int xmm, int base, int32_t disp);
void jit_addps_mem(jit_buf_t *b, int xmm, int base, int32_t disp);
void jit_subps_mem(jit_buf_t *b, int xmm, int base, int32_t disp);

/* SSE2 integer and conversion instructions */
void jit_cvtdq2ps(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_cvttps2dq(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_paddd(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_psubd(jit_buf_t *b, int dst_xmm, int src_xmm);
void jit_pslld_imm(jit_buf_t *b, int xmm, uint8_t imm);
void jit_psrld_imm(jit_buf_t *b, int xmm, uint8_t imm);
void jit_psrad_imm(jit_buf_t *b, int xmm, uint8_t imm);
void jit_movd_to_xmm(jit_buf_t *b, int xmm, int gpr);
void jit_movd_from_xmm(jit_buf_t *b, int gpr, int xmm);

/* Additional control flow */
int  jit_jle_fwd(jit_buf_t *b);
int  jit_jne_fwd(jit_buf_t *b);
int  jit_je_fwd(jit_buf_t *b);
void jit_jle_back(jit_buf_t *b, int target_offset);
void jit_jne_back(jit_buf_t *b, int target_offset);

/* Additional GP operations */
void jit_shl_reg_imm(jit_buf_t *b, int reg, uint8_t imm);
void jit_shr_reg_imm(jit_buf_t *b, int reg, uint8_t imm);
void jit_and_reg_imm32(jit_buf_t *b, int reg, int32_t imm);
void jit_or_reg_reg(jit_buf_t *b, int dst, int src);
void jit_sub_reg_reg(jit_buf_t *b, int dst, int src);

/* Embed raw 4-float constant in code stream, return offset of data */
int jit_embed_f32x4(jit_buf_t *b, float v0, float v1, float v2, float v3);

/* Load XMM from RIP-relative (data embedded in code) */
void jit_movaps_riprel(jit_buf_t *b, int xmm, int data_offset);

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

/* Compile RMSNorm kernel for given dimension */
jit_rmsnorm_fn jit_compile_rmsnorm_kernel(int dim);

/* Compile softmax kernel for given size */
jit_softmax_fn jit_compile_softmax_kernel(int n);

/* Compile RoPE kernel for given head dimension */
jit_rope_fn jit_compile_rope_kernel(int head_dim);

/* Compile dot product kernel for given size */
jit_dot_fn jit_compile_dot_kernel(int n);

/* Compile element-wise multiply: dst[i] *= src[i] */
jit_ewise_fn jit_compile_vmul_kernel(int n);

/* Compile element-wise add: dst[i] += src[i] */
jit_ewise_fn jit_compile_vadd_kernel(int n);

/* Compile scaled add: dst += scale * src */
jit_axpy_fn jit_compile_axpy_kernel(int n);

/* Compile fused SiLU(gate) * up */
jit_fused_silu_mul_fn jit_compile_fused_silu_mul_kernel(int n);

/* Compile GELU activation */
jit_silu_fn jit_compile_gelu_kernel(int n);

/* Compile LayerNorm kernel */
jit_layernorm_fn jit_compile_layernorm_kernel(int dim);

/* Compile tokenizer/detokenizer helper kernels */
jit_mem_equal_fn jit_compile_mem_equal_kernel(void);
jit_bpe_escape_scan_fn jit_compile_bpe_escape_scan_kernel(void);
jit_find_byte_fn jit_compile_find_byte_kernel(void);
jit_merge_cmp_fn jit_compile_merge_cmp_kernel(void);
jit_token_append_fn jit_compile_token_append_kernel(void);

/* =============================================================================
 * AVX2 VEX-Encoded Instructions (256-bit SIMD)
 *
 * All use VEX prefix encoding (2-byte C5 or 3-byte C4).
 * 3-operand form: dst = src1 OP src2 (non-destructive).
 * =============================================================================*/

void jit_vzeroupper(jit_buf_t *b);

/* 256-bit packed float loads/stores (unaligned) */
void jit_vmovups_load256(jit_buf_t *b, int ymm, int base, int32_t disp);
void jit_vmovups_store256(jit_buf_t *b, int base, int32_t disp, int ymm);

/* 256-bit packed float arithmetic (3-operand) */
void jit_vxorps256(jit_buf_t *b, int dst, int src1, int src2);
void jit_vaddps256(jit_buf_t *b, int dst, int src1, int src2);
void jit_vmulps256(jit_buf_t *b, int dst, int src1, int src2);

/* Broadcast single float to all 8 YMM lanes */
void jit_vbroadcastss(jit_buf_t *b, int ymm, int base, int32_t disp);
void jit_vbroadcastss_reg(jit_buf_t *b, int ymm_dst, int xmm_src);

/* Sign-extend 8 packed int8 from memory to 8 packed int32 in YMM */
void jit_vpmovsxbd256(jit_buf_t *b, int ymm, int base, int32_t disp);

/* Convert 8 packed int32 to 8 packed float (YMM) */
void jit_vcvtdq2ps256(jit_buf_t *b, int dst, int src);

/* Fused multiply-add: dst += src1 * src2 (AVX2+FMA) */
void jit_vfmadd231ps256(jit_buf_t *b, int dst, int src1, int src2);

/* Extract upper 128 bits of YMM to XMM */
void jit_vextractf128(jit_buf_t *b, int xmm_dst, int ymm_src, uint8_t imm);

/* 128-bit VEX-encoded ops (for horizontal sum after reduction) */
void jit_vaddps128(jit_buf_t *b, int dst, int src1, int src2);
void jit_vhaddps128(jit_buf_t *b, int dst, int src1, int src2);

/* VEX-encoded scalar/utility */
void jit_vmovss_store_vex(jit_buf_t *b, int base, int32_t disp, int xmm);
void jit_vmovd_to_xmm_vex(jit_buf_t *b, int xmm, int gpr);
void jit_prefetcht0(jit_buf_t *b, int base, int32_t disp);

/* AVX2 integer SIMD emitters (for Q4_0×Q8_0 integer dot product) */
void jit_vpand256(jit_buf_t *b, int dst, int src1, int src2);
void jit_vpsrlw256_imm(jit_buf_t *b, int ymm, int src, uint8_t imm);
void jit_vpunpcklbw256(jit_buf_t *b, int dst, int src1, int src2);
void jit_vpunpckhbw256(jit_buf_t *b, int dst, int src1, int src2);
void jit_vpmaddubsw256(jit_buf_t *b, int dst, int src1, int src2);
void jit_vpmaddwd256(jit_buf_t *b, int dst, int src1, int src2);
void jit_vpaddd256(jit_buf_t *b, int dst, int src1, int src2);
void jit_vmovdqu_load256(jit_buf_t *b, int ymm, int base, int32_t disp);
void jit_vmovdqu_store256(jit_buf_t *b, int base, int32_t disp, int ymm);
void jit_vpxor256(jit_buf_t *b, int dst, int src1, int src2);

/* Compile AVX2+FMA fused Q8_0 GEMV kernel (8-wide, 4-row batched) */
jit_gemv_q8_fn jit_compile_q8_gemv_avx2(int rows, int cols);

/* Compile AVX2 Q4_0×Q8_0 integer GEMV kernel */
jit_gemv_q8_fn jit_compile_q4_q8_gemv_avx2(int rows, int cols);

/* Get number of JIT-compiled kernels cached */
int jit_kernel_count(void);

/* Get total bytes of JIT-compiled code */
int jit_code_bytes(void);

/* Self-test: compile, execute, verify */
int jit_selftest(void);

#endif /* TENSOROS_X86_JIT_H */
