// CUDA fallback for the Vulkan path on a rented RTX 2080 Ti.
//
// The host's NVIDIA open kernel module (570.133.07) exposes no usable Vulkan ICD: the library
// exports vk_icdNegotiateLoaderICDInterfaceVersion, but calling it returns -3
// (VK_ERROR_INITIALIZATION_FAILED) and vk_icdGetInstanceProcAddr(NULL, "vkCreateInstance") is
// NULL, so `vulkaninfo` reports ERROR_INCOMPATIBLE_DRIVER. This file measures the 1-bit ternary
// carrier on the same GPU with CUDA instead. It is NOT the repository's Vulkan kernel, and the
// 1-bit carrier is not implemented in src/gpu.
//
// The host box has nvcc 12.0 but no CUDA headers (nvidia-cuda-dev is a 668 MB download), so this
// translation unit is header-free: the keywords come from a local stub, and the two device
// operations that need declarations are written as PTX (`popc.b64`, `shfl.sync.down.b32`).
//
// Build (device half):
//   nvcc -ccbin gcc -I stubs -O3 -arch=sm_75 -cubin cuda_ternary.cu -o cuda_ternary.cubin
// Build (host half):
//   gcc -O2 cuda_ternary_host.c -o cuda_ternary -ldl
//
// Three measurements, each 5 warmup dispatches then 5 timed dispatches:
//   1. popcount_bw: 1 GiB of 64-bit words, one popc per word -> raw 1-bit read bandwidth.
//   2. ternary_gemv: the 7B ffn_down shape (m=3584, k=18944, 296 groups of 64), one warp per
//      row, checked against a CPU evaluation of the same packed data.
//   3. ternary_fold: the same kernel at the 7B 1-bit pack size (m=3584, groups=34816,
//      7,985,995,776 terms = 998,244,352 B) -> the 0.998 GB/token ceiling.
//
// The ternary dot here is sum(w_i * a_i) with w in {0,1} and a in {-1,0,+1}, encoded as the
// positive mask P and negative mask N: popcount(w & P) - popcount(w & N). Two popc per 64
// terms. The CPU cell's four-popcount form uses a different bit encoding; the bytes, not the
// popcount count, bind this GPU probe.
//
// Accumulators are 32-bit: a lane sees at most ceil(34816/32) = 1088 groups of 64 terms, so its
// partial sum is bounded by 1088 * 64 = 69632, and a row sum by 2,225,152. Both fit an int32 and
// are exact in f32.

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __forceinline__ inline __attribute__((always_inline))

__device__ __forceinline__ unsigned popc64(unsigned long long w) {
    unsigned r;
    asm volatile("popc.b64 %0, %1;" : "=r"(r) : "l"(w));
    return r;
}

__device__ __forceinline__ unsigned warp_sum(unsigned v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        unsigned t;
        asm volatile("shfl.sync.down.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                     : "=r"(t)
                     : "r"(v), "r"(off));
        v += t;
    }
    return v;
}

extern "C" __global__ void popcount_bw(const unsigned long long *__restrict__ w,
                                       unsigned long long n, unsigned *__restrict__ out) {
    unsigned long long i =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (; i < n; i += stride) acc += popc64(w[i]);
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0)
        out[((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5] = acc;
}

// One warp per row. W is row-major groups of 64 bits; P/N are the shared activation masks.
extern "C" __global__ void ternary_gemv(const unsigned long long *__restrict__ W,
                                        const unsigned long long *__restrict__ P,
                                        const unsigned long long *__restrict__ N, int m,
                                        int groups, float *__restrict__ out) {
    int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (warp >= m) return;
    const unsigned long long *row = W + (unsigned long long)warp * groups;
    int acc = 0;
    for (int g = lane; g < groups; g += 32) {
        unsigned long long w = row[g];
        acc += (int)popc64(w & P[g]) - (int)popc64(w & N[g]);
    }
    acc = (int)warp_sum((unsigned)acc);
    if (lane == 0) out[warp] = (float)acc;
}
