// Exact-token prefill comparator against a llama.cpp build (CPU or GPU).
//
// The campaign contract (docs/evidence/gpu-prefill/compare.md, readiness report) requires both
// engines to consume the SAME raw token ids from an empty state, with model weights resident before
// timing. `llama-bench` cannot do this: it synthesizes ids (llama-bench.cpp) and derives n_ctx
// internally, so it is a shape proxy, not a matched comparator. This runner loads the same GGUF,
// reads a whitespace-separated id file, prefills it in one batch from a fresh context, synchronizes,
// and prints a JSON receipt with the timing and the greedy next token.
//
// Usage:
//   llama_cmp <model.gguf> <ids.txt> <n_gpu_layers> <n_ctx> <n_batch> [<n_ubatch>]
//
// n_gpu_layers < 0 offloads all layers; 0 forces CPU. Timing is device-resident prefill only:
// the timed region is llama_decode() of the whole id list plus llama_synchronize(), after a warm-up
// prefill on a discarded context. Model load, context creation and warm-up are outside the timed
// region. Complexity is dominated by llama.cpp's own GEMM; the harness adds one batch and O(N) ids.

#define _POSIX_C_SOURCE 200809L
#include "llama.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void die(const char *what) {
    fprintf(stderr, "llama_cmp: %s\n", what);
    exit(2);
}

// Whitespace-separated unsigned integers; the id files are text (engine_profile reads them with
// split_whitespace), not binary.
static llama_token *read_ids(const char *path, int32_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) die("cannot open ids file");
    if (fseek(f, 0, SEEK_END) != 0) die("cannot size ids file");
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) die("cannot size ids file");
    char *buf = malloc((size_t)n + 1);
    if (buf == NULL) die("out of memory for ids");
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) die("cannot read ids file");
    fclose(f);
    buf[n] = '\0';

    size_t cap = 1024;
    int32_t count = 0;
    llama_token *ids = malloc(cap * sizeof(*ids));
    if (ids == NULL) die("out of memory for ids");
    char *p = buf;
    while (*p != '\0') {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        if (*p == '\0') break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || v < 0 || v > INT32_MAX) die("malformed id");
        if ((size_t)count == cap) {
            cap *= 2;
            ids = realloc(ids, cap * sizeof(*ids));
            if (ids == NULL) die("out of memory for ids");
        }
        ids[count++] = (llama_token)v;
        p = end;
    }
    free(buf);
    if (count == 0) die("ids file has no tokens");
    *n_out = count;
    return ids;
}

static int32_t argmax(const float *v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

// One full prefill of `ids` from a fresh context. Returns tok/s through *tps and the greedy next
// token through *next.
static void prefill_timed(struct llama_model *model, const llama_token *ids, int32_t n,
                          int32_t n_ctx, int32_t n_batch, int32_t n_ubatch, int32_t n_threads,
                          double *elapsed, double *tps, int32_t *next) {
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)n_ctx;
    cp.n_batch = (uint32_t)n_batch;
    cp.n_ubatch = (uint32_t)n_ubatch;
    if (n_threads > 0) {
        cp.n_threads = n_threads;
        cp.n_threads_batch = n_threads;
    }
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (ctx == NULL) die("llama_init_from_model failed");

    struct llama_batch batch = llama_batch_get_one((llama_token *)ids, n);
    double t0 = now_s();
    if (llama_decode(ctx, batch) != 0) die("llama_decode failed");
    llama_synchronize(ctx);
    double t1 = now_s();

    *elapsed = t1 - t0;
    *tps = (double)n / (*elapsed > 0.0 ? *elapsed : 1e-9);

    const float *logits = llama_get_logits_ith(ctx, n - 1);
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    *next = argmax(logits, n_vocab);

    llama_free(ctx);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s model.gguf ids.txt n_gpu_layers n_ctx n_batch [n_ubatch] [n_threads]\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *ids_path = argv[2];
    int32_t ngl = (int32_t)strtol(argv[3], NULL, 10);
    int32_t n_ctx = (int32_t)strtol(argv[4], NULL, 10);
    int32_t n_batch = (int32_t)strtol(argv[5], NULL, 10);
    int32_t n_ubatch = argc > 6 ? (int32_t)strtol(argv[6], NULL, 10) : n_batch;
    int32_t n_threads = argc > 7 ? (int32_t)strtol(argv[7], NULL, 10) : 0;
    if (n_ctx <= 0 || n_batch <= 0 || n_ubatch <= 0 || n_ubatch > n_batch) die("bad geometry");

    int32_t n = 0;
    llama_token *ids = read_ids(ids_path, &n);
    if (n > n_batch) die("id count exceeds n_batch; raise n_batch");

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (model == NULL) die("llama_model_load_from_file failed");

    // Warm-up on a discarded context: pages weights in without entering the timed region.
    double w_elapsed, w_tps;
    int32_t w_next;
    prefill_timed(model, ids, n, n_ctx, n_batch, n_ubatch, n_threads, &w_elapsed, &w_tps, &w_next);

    double elapsed, tps;
    int32_t next;
    prefill_timed(model, ids, n, n_ctx, n_batch, n_ubatch, n_threads, &elapsed, &tps, &next);

    printf("{\"model\":\"%s\",\"ids\":\"%s\",\"n_tokens\":%d,\"n_gpu_layers\":%d,"
           "\"n_ctx\":%d,\"n_batch\":%d,\"n_ubatch\":%d,\"n_threads\":%d,\"warmup_tps\":%.4f,"
           "\"elapsed_s\":%.6f,\"tok_per_s\":%.4f,\"next_token\":%d,\"system_info\":\"%s\"}\n",
           model_path, ids_path, n, ngl, n_ctx, n_batch, n_ubatch, n_threads, w_tps,
           elapsed, tps, next, llama_print_system_info());

    llama_model_free(model);
    llama_backend_free();
    free(ids);
    return 0;
}
