// Host half of the CUDA fallback probe. Loads libcuda.so.1 through dlopen, so it needs neither
// cuda_runtime.h nor libcudart: the box has the driver and nvcc, not the CUDA dev package.
//
// Build: gcc -O2 cuda_ternary_host.c -o cuda_ternary -ldl
// Run:   ./cuda_ternary cuda_ternary.cubin
//
// Weight/activation data are synthetic; the arithmetic is the 1-bit ternary dot, and rows are
// checked against a CPU evaluation of the same bytes before any timing is reported.

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef unsigned int CUdevice_attribute;
typedef void *CUcontext, *CUmodule, *CUfunction, *CUevent, *CUstream;

static void *libcuda;

static CUresult (*p_cuInit)(unsigned);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuDeviceGetName)(char *, int, CUdevice);
static CUresult (*p_cuDeviceGetAttribute)(int *, CUdevice_attribute, CUdevice);
static CUresult (*p_cuCtxCreate_v2)(CUcontext *, unsigned, CUdevice);
static CUresult (*p_cuModuleLoad)(CUmodule *, const char *);
static CUresult (*p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*p_cuMemAlloc_v2)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemFree_v2)(CUdeviceptr);
static CUresult (*p_cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
static CUresult (*p_cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
static CUresult (*p_cuMemsetD8_v2)(CUdeviceptr, unsigned char, size_t);
static CUresult (*p_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                                    unsigned, unsigned, unsigned, CUstream, void **, void **);
static CUresult (*p_cuCtxSynchronize)(void);
static CUresult (*p_cuEventCreate)(CUevent *, unsigned);
static CUresult (*p_cuEventRecord)(CUevent, CUstream);
static CUresult (*p_cuEventSynchronize)(CUevent);
static CUresult (*p_cuEventElapsedTime)(float *, CUevent, CUevent);
static CUresult (*p_cuEventDestroy_v2)(CUevent);
static CUresult (*p_cuGetErrorString)(CUresult, const char **);

#define SYM(f, name)                                                                     \
    do {                                                                                 \
        *(void **)(&f) = dlsym(libcuda, name);                                           \
        if (!f) {                                                                        \
            fprintf(stderr, "missing driver symbol %s\n", name);                         \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

static void ck(CUresult r, const char *what) {
    if (r != 0) {
        const char *s = "?";
        if (p_cuGetErrorString) p_cuGetErrorString(r, &s);
        fprintf(stderr, "CUDA %s: %d (%s)\n", what, r, s);
        exit(1);
    }
}

static uint64_t rng_state = 0x00c00b4a7e110b01ull;
static uint64_t rng64(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return rng_state;
}

typedef struct {
    CUfunction f;
    void **args;
    unsigned gx, gy, gz, bx, by, bz;
} Launch;

static void dispatch(const Launch *l) {
    ck(p_cuLaunchKernel(l->f, l->gx, l->gy, l->gz, l->bx, l->by, l->bz, 0, 0, l->args, 0),
       "cuLaunchKernel");
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// 5 warmup dispatches, 5 timed dispatches; median and [min..max] over the timed five.
static void bench(const char *label, const Launch *l, size_t bytes, int warmup, int reps,
                  double *median_out) {
    for (int i = 0; i < warmup; ++i) dispatch(l);
    ck(p_cuCtxSynchronize(), "sync");
    double ms[64];
    for (int i = 0; i < reps; ++i) {
        CUevent a, b;
        ck(p_cuEventCreate(&a, 0), "eventCreate");
        ck(p_cuEventCreate(&b, 0), "eventCreate");
        ck(p_cuEventRecord(a, 0), "eventRecord");
        dispatch(l);
        ck(p_cuEventRecord(b, 0), "eventRecord");
        ck(p_cuEventSynchronize(b), "eventSync");
        float t = 0;
        ck(p_cuEventElapsedTime(&t, a, b), "elapsed");
        ms[i] = t;
        ck(p_cuEventDestroy_v2(a), "eventDestroy");
        ck(p_cuEventDestroy_v2(b), "eventDestroy");
    }
    qsort(ms, reps, sizeof(double), cmp_double);
    double med = ms[reps / 2];
    double gbps = (double)bytes / (med * 1e6);
    printf("%s bytes=%zu median_ms=%.4f [%.4f..%.4f] GBps=%.1f tok_s=%.1f\n", label, bytes,
           med, ms[0], ms[reps - 1], gbps, 1e3 / med);
    if (median_out) *median_out = gbps;
}

int main(int argc, char **argv) {
    const char *cubin = argc > 1 ? argv[1] : "cuda_ternary.cubin";
    libcuda = dlopen("libcuda.so.1", RTLD_NOW);
    if (!libcuda) {
        fprintf(stderr, "dlopen libcuda.so.1: %s\n", dlerror());
        return 1;
    }
    SYM(p_cuInit, "cuInit");
    SYM(p_cuDeviceGet, "cuDeviceGet");
    SYM(p_cuDeviceGetName, "cuDeviceGetName");
    SYM(p_cuDeviceGetAttribute, "cuDeviceGetAttribute");
    SYM(p_cuCtxCreate_v2, "cuCtxCreate_v2");
    SYM(p_cuModuleLoad, "cuModuleLoad");
    SYM(p_cuModuleGetFunction, "cuModuleGetFunction");
    SYM(p_cuMemAlloc_v2, "cuMemAlloc_v2");
    SYM(p_cuMemFree_v2, "cuMemFree_v2");
    SYM(p_cuMemcpyHtoD_v2, "cuMemcpyHtoD_v2");
    SYM(p_cuMemcpyDtoH_v2, "cuMemcpyDtoH_v2");
    SYM(p_cuMemsetD8_v2, "cuMemsetD8_v2");
    SYM(p_cuLaunchKernel, "cuLaunchKernel");
    SYM(p_cuCtxSynchronize, "cuCtxSynchronize");
    SYM(p_cuEventCreate, "cuEventCreate");
    SYM(p_cuEventRecord, "cuEventRecord");
    SYM(p_cuEventSynchronize, "cuEventSynchronize");
    SYM(p_cuEventElapsedTime, "cuEventElapsedTime");
    SYM(p_cuEventDestroy_v2, "cuEventDestroy_v2");
    SYM(p_cuGetErrorString, "cuGetErrorString");

    ck(p_cuInit(0), "cuInit");
    CUdevice dev;
    ck(p_cuDeviceGet(&dev, 0), "cuDeviceGet");
    char name[256] = {0};
    ck(p_cuDeviceGetName(name, sizeof name, dev), "name");
    int sms = 0, clk_khz = 0, mem_khz = 0, bus = 0;
    ck(p_cuDeviceGetAttribute(&sms, 16, dev), "sms");
    ck(p_cuDeviceGetAttribute(&clk_khz, 13, dev), "clock");
    ck(p_cuDeviceGetAttribute(&mem_khz, 36, dev), "memclock");
    ck(p_cuDeviceGetAttribute(&bus, 37, dev), "buswidth");
    CUcontext ctx;
    ck(p_cuCtxCreate_v2(&ctx, 0, dev), "ctx");
    printf("device: %s SMs=%d clock=%.0f MHz mem_clock=%.0f MHz bus=%d bits\n", name, sms,
           clk_khz / 1000.0, mem_khz / 1000.0, bus);
    const double spec_gbps = 2.0 * (mem_khz * 1e3) * (bus / 8.0) / 1e9;
    printf("spec_bw_GBps: %.1f\n", spec_gbps);

    CUmodule mod;
    ck(p_cuModuleLoad(&mod, cubin), "moduleLoad");

    // ── 1. popcount bandwidth over 1 GiB ──────────────────────────────────────────────────
    const size_t n_bytes = (size_t)1 << 30;
    unsigned long long n_words = n_bytes / 8;
    const unsigned grid = 8192, block = 256;
    CUfunction fbw;
    ck(p_cuModuleGetFunction(&fbw, mod, "popcount_bw"), "popcount_bw");
    CUdeviceptr bw, bw_out;
    ck(p_cuMemAlloc_v2(&bw, n_bytes), "alloc bw");
    ck(p_cuMemAlloc_v2(&bw_out, (size_t)grid * block / 32 * 4), "alloc bw_out");
    ck(p_cuMemsetD8_v2(bw, 0x5a, n_bytes), "memset bw");
    ck(p_cuMemsetD8_v2(bw_out, 0, (size_t)grid * block / 32 * 4), "memset bw_out");
    void *args_bw[] = {&bw, &n_words, &bw_out};
    Launch lbw = {fbw, args_bw, grid, 1, 1, block, 1, 1};
    double med_bw = 0;
    bench("popcount_bw", &lbw, n_bytes, 5, 5, &med_bw);
    ck(p_cuMemFree_v2(bw), "free bw");
    ck(p_cuMemFree_v2(bw_out), "free bw_out");

    // ── 2. the 7B ffn_down ternary GEMV, checked against the CPU ──────────────────────────
    const int m = 3584, k = 18944, groups = k / 64;
    const size_t wn = (size_t)m * groups;
    unsigned long long *W = malloc(wn * 8), *P = malloc(groups * 8), *N = malloc(groups * 8);
    for (size_t i = 0; i < wn; ++i) W[i] = rng64();
    for (int g = 0; g < groups; ++g) {
        unsigned long long p = 0, n = 0;
        for (int j = 0; j < 64; ++j) {
            unsigned r = (unsigned)(rng64() % 3);
            if (r == 0) p |= 1ull << j;
            if (r == 1) n |= 1ull << j;
        }
        P[g] = p;
        N[g] = n;
    }
    CUdeviceptr dW, dP, dN, dout;
    ck(p_cuMemAlloc_v2(&dW, wn * 8), "alloc W");
    ck(p_cuMemAlloc_v2(&dP, groups * 8), "alloc P");
    ck(p_cuMemAlloc_v2(&dN, groups * 8), "alloc N");
    ck(p_cuMemAlloc_v2(&dout, (size_t)m * 4), "alloc out");
    ck(p_cuMemcpyHtoD_v2(dW, W, wn * 8), "copy W");
    ck(p_cuMemcpyHtoD_v2(dP, P, groups * 8), "copy P");
    ck(p_cuMemcpyHtoD_v2(dN, N, groups * 8), "copy N");
    CUfunction fgemv;
    ck(p_cuModuleGetFunction(&fgemv, mod, "ternary_gemv"), "ternary_gemv");
    void *args_gemv[] = {&dW, &dP, &dN, (void *)&m, (void *)&groups, &dout};
    Launch lgemv = {fgemv, args_gemv, (unsigned)((m + 7) / 8), 1, 1, 256, 1, 1};
    dispatch(&lgemv);
    ck(p_cuCtxSynchronize(), "sync gemv");
    float *got = malloc((size_t)m * 4);
    ck(p_cuMemcpyDtoH_v2(got, dout, (size_t)m * 4), "copy out");
    int bad = 0;
    const int rows[4] = {0, 1, m / 2, m - 1};
    for (int ri = 0; ri < 4; ++ri) {
        int r = rows[ri];
        long long want = 0;
        for (int g = 0; g < groups; ++g) {
            unsigned long long w = W[(size_t)r * groups + g];
            want += __builtin_popcountll(w & P[g]) - __builtin_popcountll(w & N[g]);
        }
        if ((long long)got[r] != want) {
            printf("CHECK_FAIL row=%d got=%lld want=%lld\n", r, (long long)got[r], want);
            ++bad;
        }
    }
    printf("ternary_gemv_check rows=4 bad=%d\n", bad);
    if (bad) return 1;
    bench("ternary_gemv", &lgemv, wn * 8, 5, 5, NULL);
    ck(p_cuMemFree_v2(dW), "free W");
    ck(p_cuMemFree_v2(dP), "free P");
    ck(p_cuMemFree_v2(dN), "free N");
    ck(p_cuMemFree_v2(dout), "free out");
    free(W);
    free(P);
    free(N);
    free(got);

    // ── 3. the 7B 1-bit pack size: 0.998 GB/token through the same kernel ────────────────
    const int big_groups = 34816; // 3584 * 34816 * 64 bits = 998,244,352 B
    const size_t big_bytes = (size_t)m * big_groups * 8;
    unsigned long long *bigP = malloc((size_t)big_groups * 8);
    unsigned long long *bigN = malloc((size_t)big_groups * 8);
    for (int g = 0; g < big_groups; ++g) {
        unsigned long long p = rng64(), n = rng64();
        bigP[g] = p & ~n;
        bigN[g] = n & ~p;
    }
    CUdeviceptr dBig, dBP, dBN, dBigOut;
    ck(p_cuMemAlloc_v2(&dBig, big_bytes), "alloc big");
    ck(p_cuMemsetD8_v2(dBig, 0x5a, big_bytes), "memset big");
    ck(p_cuMemAlloc_v2(&dBP, (size_t)big_groups * 8), "alloc bigP");
    ck(p_cuMemAlloc_v2(&dBN, (size_t)big_groups * 8), "alloc bigN");
    ck(p_cuMemAlloc_v2(&dBigOut, (size_t)m * 4), "alloc bigOut");
    ck(p_cuMemcpyHtoD_v2(dBP, bigP, (size_t)big_groups * 8), "copy bigP");
    ck(p_cuMemcpyHtoD_v2(dBN, bigN, (size_t)big_groups * 8), "copy bigN");
    void *args_big[] = {&dBig, &dBP, &dBN, (void *)&m, (void *)&big_groups, &dBigOut};
    Launch lbig = {fgemv, args_big, (unsigned)((m + 7) / 8), 1, 1, 256, 1, 1};
    // The fold's weights are a known 0x5a byte pattern, so a row is checkable without a 1 GB
    // host copy: every word is 0x5a5a5a5a5a5a5a5a.
    dispatch(&lbig);
    ck(p_cuCtxSynchronize(), "sync big");
    float *big_got = malloc((size_t)m * 4);
    ck(p_cuMemcpyDtoH_v2(big_got, dBigOut, (size_t)m * 4), "copy bigOut");
    const unsigned long long pat = 0x5a5a5a5a5a5a5a5aull;
    int big_bad = 0;
    const int big_rows[2] = {0, m - 1};
    for (int ri = 0; ri < 2; ++ri) {
        int r = big_rows[ri];
        long long want = 0;
        for (int g = 0; g < big_groups; ++g)
            want += __builtin_popcountll(pat & bigP[g]) - __builtin_popcountll(pat & bigN[g]);
        if ((long long)big_got[r] != want) {
            printf("FOLD_CHECK_FAIL row=%d got=%lld want=%lld\n", r, (long long)big_got[r],
                   want);
            ++big_bad;
        }
    }
    printf("ternary_fold_check rows=2 bad=%d\n", big_bad);
    if (big_bad) return 1;
    double med_fold = 0;
    bench("ternary_fold", &lbig, big_bytes, 5, 5, &med_fold);
    ck(p_cuMemFree_v2(dBig), "free big");
    ck(p_cuMemFree_v2(dBP), "free bigP");
    ck(p_cuMemFree_v2(dBN), "free bigN");
    ck(p_cuMemFree_v2(dBigOut), "free bigOut");
    free(bigP);
    free(bigN);
    free(big_got);

    // Ceiling: measured streaming GB/s divided by the 7B 1-bit pack's GB/token.
    const double pack_gb = 0.998244352;
    const double ceil_bw = med_bw / pack_gb;
    const double ceil_fold = med_fold / pack_gb;
    printf("ceiling_popcount_bw tok_s=%.2f gap_to_20=%.2fx fraction=%.3f\n", ceil_bw,
           20.0 / ceil_bw, ceil_bw / 20.0);
    printf("ceiling_ternary_fold tok_s=%.2f gap_to_20=%.2fx fraction=%.3f\n", ceil_fold,
           20.0 / ceil_fold, ceil_fold / 20.0);
    return 0;
}
