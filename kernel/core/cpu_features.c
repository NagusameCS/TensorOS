/* =============================================================================
 * TensorOS - CPU Feature Detection via CPUID
 *
 * x86_64-only: ARM64 uses MIDR_EL1 / ID_AA64ISARx_EL1 registers instead.
 * =============================================================================*/

#ifndef __aarch64__
/* =============================================================================
 * Detects ISA extensions (SSE, AVX, AVX2, FMA, etc.) and enables OS-level
 * support for extended state saving (XSAVE/XRSTOR for AVX registers).
 *
 * This enables runtime dispatch: if AVX2+FMA are available, GEMM uses
 * 8-wide YMM registers with fused multiply-add for ~2x over SSE2.
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/core/cpu_features.h"

/* Global feature flags — checked at runtime for ISA dispatch */
cpu_features_t cpu_features;

/* =============================================================================
 * CPUID / XGETBV Wrappers
 * =============================================================================*/

static void cpuid(uint32_t leaf, uint32_t subleaf,
                  uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx)
{
    __asm__ volatile("cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(leaf), "c"(subleaf));
}

static uint64_t xgetbv(uint32_t xcr)
{
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(xcr));
    return ((uint64_t)hi << 32) | lo;
}

/* =============================================================================
 * Feature Detection
 * =============================================================================*/

void cpu_detect_features(void)
{
    kmemset(&cpu_features, 0, sizeof(cpu_features));

    uint32_t eax, ebx, ecx, edx;

    /* Leaf 0: max standard leaf + vendor string */
    cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    uint32_t max_leaf = eax;
    *(uint32_t *)&cpu_features.vendor[0] = ebx;
    *(uint32_t *)&cpu_features.vendor[4] = edx;
    *(uint32_t *)&cpu_features.vendor[8] = ecx;
    cpu_features.vendor[12] = '\0';

    /* Leaf 1: basic feature flags */
    if (max_leaf >= 1) {
        cpuid(1, 0, &eax, &ebx, &ecx, &edx);

        /* EDX features */
        cpu_features.has_sse      = (edx >> 25) & 1;
        cpu_features.has_sse2     = (edx >> 26) & 1;
        cpu_features.has_tsc      = (edx >>  4) & 1;

        /* ECX features */
        cpu_features.has_sse3     = (ecx >>  0) & 1;
        cpu_features.has_ssse3    = (ecx >>  9) & 1;
        cpu_features.has_fma      = (ecx >> 12) & 1;
        cpu_features.has_sse41    = (ecx >> 19) & 1;
        cpu_features.has_sse42    = (ecx >> 20) & 1;
        cpu_features.has_popcnt   = (ecx >> 23) & 1;
        cpu_features.has_aes      = (ecx >> 25) & 1;
        cpu_features.has_xsave    = (ecx >> 26) & 1;
        cpu_features.has_osxsave  = (ecx >> 27) & 1;
        cpu_features.has_avx      = (ecx >> 28) & 1;
    }

    /* Leaf 7: extended features (AVX2, BMI, AVX-512) */
    if (max_leaf >= 7) {
        cpuid(7, 0, &eax, &ebx, &ecx, &edx);

        cpu_features.has_bmi1     = (ebx >>  3) & 1;
        cpu_features.has_avx2     = (ebx >>  5) & 1;
        cpu_features.has_bmi2     = (ebx >>  8) & 1;
        cpu_features.has_avx512f  = (ebx >> 16) & 1;
    }

    /* Extended leaf 0x80000001: RDTSCP */
    cpuid(0x80000000, 0, &eax, &ebx, &ecx, &edx);
    uint32_t max_ext = eax;

    if (max_ext >= 0x80000001) {
        cpuid(0x80000001, 0, &eax, &ebx, &ecx, &edx);
        cpu_features.has_rdtscp = (edx >> 27) & 1;
    }

    /* Extended leaf 0x80000007: Invariant TSC */
    if (max_ext >= 0x80000007) {
        cpuid(0x80000007, 0, &eax, &ebx, &ecx, &edx);
        cpu_features.has_invariant_tsc = (edx >> 8) & 1;
    }

    /* Check if OS has enabled AVX state saving */
    if (cpu_features.has_avx && cpu_features.has_osxsave) {
        uint64_t xcr0 = xgetbv(0);
        /* Bits 1+2 = SSE state + AVX state must both be set */
        cpu_features.avx_usable = ((xcr0 & 0x6) == 0x6);
    }

    /* AVX2 is usable only if AVX state saving is enabled AND CPU has AVX2+FMA */
    cpu_features.avx2_usable = cpu_features.avx_usable &&
                                cpu_features.has_avx2 &&
                                cpu_features.has_fma;
}

/* =============================================================================
 * Enable AVX State Saving
 *
 * Must be called before any AVX/AVX2 instruction or you get #UD.
 * Sets CR4.OSXSAVE and enables SSE+AVX state in XCR0.
 * =============================================================================*/

void cpu_enable_avx(void)
{
    if (!cpu_features.has_xsave || !cpu_features.has_avx) {
        kprintf("[CPU] AVX not supported by this CPU\n");
        return;
    }

    /* Step 1: Set CR4.OSXSAVE (bit 18) to enable XGETBV/XSETBV */
    uint64_t cr4;
    __asm__ volatile("mov %%cr4, %0" : "=r"(cr4));
    cr4 |= (1ULL << 18);
    __asm__ volatile("mov %0, %%cr4" : : "r"(cr4));

    /* Step 2: Enable SSE state + AVX state in XCR0 */
    uint64_t xcr0 = xgetbv(0);
    xcr0 |= (1 << 1) | (1 << 2);   /* Bit 1 = SSE, Bit 2 = AVX */
    uint32_t lo = (uint32_t)xcr0;
    uint32_t hi = (uint32_t)(xcr0 >> 32);
    __asm__ volatile("xsetbv" : : "a"(lo), "d"(hi), "c"(0));

    /* Step 3: Re-verify */
    xcr0 = xgetbv(0);
    cpu_features.avx_usable = ((xcr0 & 0x6) == 0x6);
    cpu_features.avx2_usable = cpu_features.avx_usable &&
                                cpu_features.has_avx2 &&
                                cpu_features.has_fma;

    if (cpu_features.avx_usable)
        kprintf("[CPU] AVX state saving enabled (XCR0=%p)\n", (void *)xcr0);
    else
        kprintf("[CPU] WARNING: Failed to enable AVX state saving\n");
}

/* =============================================================================
 * Pretty-Print Features
 * =============================================================================*/

void cpu_print_features(void)
{
    kprintf("[CPU] Vendor: %s\n", cpu_features.vendor);
    kprintf("[CPU] ISA:");
    if (cpu_features.has_sse)     kprintf(" SSE");
    if (cpu_features.has_sse2)    kprintf(" SSE2");
    if (cpu_features.has_sse3)    kprintf(" SSE3");
    if (cpu_features.has_ssse3)   kprintf(" SSSE3");
    if (cpu_features.has_sse41)   kprintf(" SSE4.1");
    if (cpu_features.has_sse42)   kprintf(" SSE4.2");
    if (cpu_features.has_avx)     kprintf(" AVX");
    if (cpu_features.has_avx2)    kprintf(" AVX2");
    if (cpu_features.has_fma)     kprintf(" FMA");
    if (cpu_features.has_aes)     kprintf(" AES-NI");
    if (cpu_features.has_popcnt)  kprintf(" POPCNT");
    if (cpu_features.has_bmi1)    kprintf(" BMI1");
    if (cpu_features.has_bmi2)    kprintf(" BMI2");
    if (cpu_features.has_avx512f) kprintf(" AVX-512F");
    kprintf("\n");

    if (cpu_features.has_invariant_tsc)
        kprintf("[CPU] Invariant TSC: yes (reliable cycle counter)\n");

    if (cpu_features.avx2_usable)
        kprintf("[CPU] GEMM dispatch: AVX2+FMA 8-wide (256-bit)\n");
    else if (cpu_features.has_sse2)
        kprintf("[CPU] GEMM dispatch: SSE2 4-wide (128-bit)\n");
    else
        kprintf("[CPU] GEMM dispatch: scalar fallback\n");
}

#else /* __aarch64__ */

#include "kernel/core/cpu_features.h"

cpu_features_t cpu_features = {0};

void cpu_detect_features(void) {}
void cpu_enable_avx(void) {}
void cpu_print_features(void) {}

#endif /* __aarch64__ */
