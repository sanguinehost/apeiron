// Streaming-read bandwidth probe for the decode-ceiling measurement.
//
// Build (through the workspace guard):
//   agent-guard run -- cc -O3 -march=native -pthread -o /tmp/membw tools/membw.c
// Run:
//   /tmp/membw 4 128 60      # four threads, 128 MiB each, 60 timed passes
//   /tmp/membw 1 256 60      # one thread
//
// Thread t pins to CPU t. On the ThinkPad L13 Yoga 20R6 the four physical cores
// are 0,1,2,3; the SMT siblings are 0/4, 1/5, 2/6, 3/7. Pinning to 0,2,4,6
// would put two threads on each of two cores and halve the aggregate.
//
// Each pass reads one chunk with eight independent 64-bit loads per iteration
// and four accumulators. The two warmup passes are discarded. The chunk is far
// larger than the 6 MiB L3, so the timed passes read DRAM.
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define MAXT 8

static int nthread = 4;
static size_t chunk = 128u << 20;
static int passes = 25;
static uint8_t *buf[MAXT];
static double gbps[MAXT];
static double best[MAXT];
static uint64_t sink;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *reader(void *arg) {
    long t = (long)arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(t, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

    const uint64_t *p = (const uint64_t *)buf[t];
    const size_t words = chunk / sizeof(uint64_t);
    uint64_t acc = 0;
    for (int pass = 0; pass < passes + 2; pass++) {
        double t0 = now();
        uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (size_t i = 0; i < words; i += 8) {
            a0 += p[i + 0];
            a1 += p[i + 1];
            a2 += p[i + 2];
            a3 += p[i + 3];
            a0 += p[i + 4];
            a1 += p[i + 5];
            a2 += p[i + 6];
            a3 += p[i + 7];
        }
        double dt = now() - t0;
        acc += a0 + a1 + a2 + a3;
        if (pass >= 2) {
            double g = (double)chunk / dt / 1e9;
            gbps[t] += g;
            if (g > best[t])
                best[t] = g;
        }
    }
    sink += acc;
    gbps[t] /= (double)passes;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc > 1)
        nthread = atoi(argv[1]);
    if (argc > 2)
        chunk = (size_t)atoi(argv[2]) << 20;
    if (argc > 3)
        passes = atoi(argv[3]);
    if (nthread < 1 || nthread > MAXT || chunk < (8u << 20) || passes < 1) {
        fprintf(stderr, "usage: %s [threads 1..8] [chunk_MiB >=8] [passes >=1]\n", argv[0]);
        return 2;
    }
    for (int t = 0; t < nthread; t++) {
        if (posix_memalign((void **)&buf[t], 2u << 20, chunk) != 0) {
            perror("posix_memalign");
            return 1;
        }
        madvise(buf[t], chunk, MADV_HUGEPAGE);
        memset(buf[t], 0x5a, chunk);
    }
    pthread_t th[MAXT];
    for (long t = 0; t < nthread; t++)
        pthread_create(&th[t], NULL, reader, (void *)t);
    for (int t = 0; t < nthread; t++)
        pthread_join(th[t], NULL);
    double total = 0, total_best = 0;
    for (int t = 0; t < nthread; t++) {
        printf("thread %d: mean %.2f GB/s  best %.2f GB/s\n", t, gbps[t], best[t]);
        total += gbps[t];
        total_best += best[t];
    }
    printf("aggregate: mean %.2f GB/s  best-pass %.2f GB/s  "
           "(chunk %zu MiB, %d passes, sink %llu)\n",
           total, total_best, chunk >> 20, passes, (unsigned long long)sink);
    return 0;
}
