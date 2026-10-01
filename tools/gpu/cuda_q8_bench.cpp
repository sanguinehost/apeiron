// Matched CUDA benchmark for llama.cpp's Q8 prefill projection (MMQ int8 path).
//
// Links against libggml, libggml-base, libggml-cuda from llama.cpp.
// Evaluates Q8_0 weights * activation projections at Apeiron native-kernel shapes.
// Uses CUDA events on the backend stream to measure device time (device_ns).
// Rate formula:
//   blocks   = K / 32
//   MACs     = M * tokens * blocks * 32  (= M * tokens * K)
//   GMAC/s   = MACs / device_ns

#include "ggml.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct ShapeDef {
    const char * name;
    int64_t m;
    int64_t k;
    int64_t tokens;
};

static const ShapeDef CANONICAL_SHAPES[] = {
    {"attn_q",       4096, 1024, 200},
    {"attn_q",       4096, 1024, 512},
    {"attn_k_v",      512, 1024, 200},
    {"attn_k_v",      512, 1024, 512},
    {"attn_output",  1024, 2048, 200},
    {"attn_output",  1024, 2048, 512},
    {"ffn_gate_up",  3584, 1024, 200},
    {"ffn_gate_up",  3584, 1024, 512},
    {"ffn_down",     1024, 3584, 200},
    {"ffn_down",     1024, 3584, 512},
};

static const size_t NUM_CANONICAL_SHAPES = sizeof(CANONICAL_SHAPES) / sizeof(CANONICAL_SHAPES[0]);

struct BenchResult {
    std::string name;
    int64_t m;
    int64_t k;
    int64_t tokens;
    int64_t blocks;
    uint64_t macs;
    double dev_min_ns;
    double dev_med_ns;
    double dev_max_ns;
    double gmac_s;
    bool success;
    std::string error;
};

// Initialize Q8_0 weights with deterministic test values
static void init_q8_0_weights(struct ggml_tensor * tensor) {
    size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> buf(nbytes);
    block_q8_0 * blocks = (block_q8_0 *)buf.data();
    size_t n_blocks = ggml_nelements(tensor) / QK8_0;

    for (size_t i = 0; i < n_blocks; i++) {
        blocks[i].d = ggml_fp32_to_fp16(0.125f);
        for (int j = 0; j < QK8_0; j++) {
            blocks[i].qs[j] = (int8_t)((i + j) % 7 - 3);
        }
    }
    ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
}

// Initialize F32 activations with deterministic test values
static void init_f32_activations(struct ggml_tensor * tensor) {
    size_t nels = (size_t)ggml_nelements(tensor);
    std::vector<float> buf(nels);
    for (size_t i = 0; i < nels; i++) {
        buf[i] = ((float)(i % 13) - 6.0f) * 0.0625f;
    }
    ggml_backend_tensor_set(tensor, buf.data(), 0, nels * sizeof(float));
}

// Initialize Q8_1 activations with deterministic test values
static void init_q8_1_activations(struct ggml_tensor * tensor) {
    size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> buf(nbytes);
    block_q8_1 * blocks = (block_q8_1 *)buf.data();
    size_t n_blocks = ggml_nelements(tensor) / QK8_1;

    for (size_t i = 0; i < n_blocks; i++) {
        blocks[i].d = ggml_fp32_to_fp16(0.125f);
        blocks[i].s = ggml_fp32_to_fp16(0.0f);
        for (int j = 0; j < QK8_1; j++) {
            blocks[i].qs[j] = (int8_t)((i + j) % 5 - 2);
        }
    }
    ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
}

static bool run_single_benchmark(
    ggml_backend_t backend,
    const char * name,
    int64_t m,
    int64_t k,
    int64_t tokens,
    enum ggml_type act_type,
    int warmup_reps,
    int reps,
    BenchResult * result,
    int64_t k_act = -1 // optional dimension mismatch for testing
) {
    result->name = name;
    result->m = m;
    result->k = k;
    result->tokens = tokens;
    result->blocks = k / 32;
    result->macs = (uint64_t)m * tokens * k;
    result->success = false;

    if (k <= 0 || m <= 0 || tokens <= 0 || (k % 32 != 0)) {
        result->error = "Invalid dimensions: K must be positive multiple of 32, M and tokens must be positive";
        return false;
    }

    int64_t actual_k_act = (k_act > 0) ? k_act : k;

    // Set up GGML context
    size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    struct ggml_init_params params = {
        ctx_size,
        NULL,
        true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        result->error = "ggml_init failed";
        return false;
    }

    // Weight: [K, M], Q8_0
    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
    // Activation: [K, tokens], act_type (F32 for CUDA MMQ, or Q8_1)
    struct ggml_tensor * act = ggml_new_tensor_2d(ctx, act_type, actual_k_act, tokens);

    if (weight->ne[0] != act->ne[0]) {
        // Deliberate shape mismatch control
        ggml_free(ctx);
        result->error = "Shape mismatch: weight K != activation K";
        return false;
    }

    // Output: [M, tokens], F32
    struct ggml_tensor * out = ggml_mul_mat(ctx, weight, act);

    // Build compute graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, out);

    // Allocate tensors on backend
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        result->error = "ggml_backend_alloc_ctx_tensors failed";
        return false;
    }

    // Initialize inputs
    init_q8_0_weights(weight);
    if (act_type == GGML_TYPE_F32) {
        init_f32_activations(act);
    } else if (act_type == GGML_TYPE_Q8_1) {
        init_q8_1_activations(act);
    }

    // Warmup
    for (int w = 0; w < warmup_reps; w++) {
        enum ggml_status st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            result->error = "Warmup compute failed";
            return false;
        }
    }

    // Measure via CUDA events on the backend stream
    cudaEvent_t ev_start_raw, ev_stop_raw;
    cudaError_t cr1 = cudaEventCreate(&ev_start_raw);
    cudaError_t cr2 = cudaEventCreate(&ev_stop_raw);
    if (cr1 != cudaSuccess || cr2 != cudaSuccess) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        result->error = "cudaEventCreate failed";
        return false;
    }

    struct ggml_backend_event ev_start = { ggml_backend_get_device(backend), (void*)ev_start_raw };
    struct ggml_backend_event ev_stop  = { ggml_backend_get_device(backend), (void*)ev_stop_raw };

    std::vector<double> dev_ns_reps;
    dev_ns_reps.reserve(reps);

    for (int r = 0; r < reps; r++) {
        ggml_backend_event_record(&ev_start, backend);
        enum ggml_status st = ggml_backend_graph_compute_async(backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            cudaEventDestroy(ev_start_raw);
            cudaEventDestroy(ev_stop_raw);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            result->error = "Async compute failed";
            return false;
        }
        ggml_backend_event_record(&ev_stop, backend);
        cudaEventSynchronize(ev_stop_raw);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start_raw, ev_stop_raw);
        double ns = (double)ms * 1e6;
        dev_ns_reps.push_back(ns);
    }

    cudaEventDestroy(ev_start_raw);
    cudaEventDestroy(ev_stop_raw);

    std::sort(dev_ns_reps.begin(), dev_ns_reps.end());
    result->dev_min_ns = dev_ns_reps.front();
    result->dev_max_ns = dev_ns_reps.back();
    result->dev_med_ns = dev_ns_reps[reps / 2];
    result->gmac_s = (double)result->macs / result->dev_med_ns;
    result->success = true;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static bool check_cpu_vs_gpu_output(
    ggml_backend_t cuda_backend,
    ggml_backend_t cpu_backend,
    int64_t m,
    int64_t k,
    int64_t tokens,
    bool seed_defect,
    double * max_diff_out,
    double * nmse_out
) {
    // 1. Build and compute on CUDA backend
    size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    struct ggml_init_params params = { ctx_size, NULL, true };
    struct ggml_context * ctx_gpu = ggml_init(params);
    struct ggml_tensor * w_gpu = ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_Q8_0, k, m);
    struct ggml_tensor * a_gpu = ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_F32, k, tokens);
    struct ggml_tensor * o_gpu = ggml_mul_mat(ctx_gpu, w_gpu, a_gpu);
    struct ggml_cgraph * gf_gpu = ggml_new_graph_custom(ctx_gpu, 16, false);
    ggml_build_forward_expand(gf_gpu, o_gpu);
    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, cuda_backend);

    init_q8_0_weights(w_gpu);
    init_f32_activations(a_gpu);

    if (ggml_backend_graph_compute(cuda_backend, gf_gpu) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf_gpu);
        ggml_free(ctx_gpu);
        return false;
    }

    size_t out_elements = (size_t)(m * tokens);
    std::vector<float> gpu_data(out_elements);
    ggml_backend_tensor_get(o_gpu, gpu_data.data(), 0, out_elements * sizeof(float));

    if (seed_defect) {
        // Intentionally inject defect to test failure rejection
        gpu_data[0] += 50.0f;
    }

    // 2. Build and compute on CPU backend with exact same weights & activations
    struct ggml_context * ctx_cpu = ggml_init(params);
    struct ggml_tensor * w_cpu = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_Q8_0, k, m);
    struct ggml_tensor * a_cpu = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, k, tokens);
    struct ggml_tensor * o_cpu = ggml_mul_mat(ctx_cpu, w_cpu, a_cpu);
    struct ggml_cgraph * gf_cpu = ggml_new_graph_custom(ctx_cpu, 16, false);
    ggml_build_forward_expand(gf_cpu, o_cpu);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, cpu_backend);

    init_q8_0_weights(w_cpu);
    init_f32_activations(a_cpu);

    if (ggml_backend_graph_compute(cpu_backend, gf_cpu) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf_cpu);
        ggml_free(ctx_cpu);
        ggml_backend_buffer_free(buf_gpu);
        ggml_free(ctx_gpu);
        return false;
    }

    std::vector<float> cpu_data(out_elements);
    ggml_backend_tensor_get(o_cpu, cpu_data.data(), 0, out_elements * sizeof(float));

    // 3. Compare outputs
    double max_diff = 0.0;
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;

    for (size_t i = 0; i < out_elements; i++) {
        double d = std::abs((double)gpu_data[i] - (double)cpu_data[i]);
        if (d > max_diff) max_diff = d;
        sum_sq_diff += d * d;
        sum_sq_ref += (double)cpu_data[i] * (double)cpu_data[i];
    }

    double nmse = (sum_sq_ref > 0.0) ? (sum_sq_diff / sum_sq_ref) : sum_sq_diff;
    *max_diff_out = max_diff;
    *nmse_out = nmse;

    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_cpu);
    ggml_backend_buffer_free(buf_gpu);
    ggml_free(ctx_gpu);

    // In MMQ vs CPU Q8_0 dot product, accumulator precision differences (fp32 block reduction
    // order differences) typically give NMSE < 1e-4 and max diff < 0.2.
    return (nmse < 5e-4);
}

int main(int argc, char ** argv) {
    int device_id = 0;
    int reps = 5;
    int warmup = 2;
    bool all_shapes = false;
    bool check_cpu = false;
    bool seed_defect = false;
    bool shape_mismatch_control = false;
    bool direct_q8_1_control = false;
    bool emit_json = false;
    enum ggml_type act_type = GGML_TYPE_F32;

    int64_t custom_m = 0;
    int64_t custom_k = 0;
    int64_t custom_tokens = 0;
    std::string custom_name = "custom";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--all-shapes") == 0) {
            all_shapes = true;
        } else if (strcmp(argv[i], "--check-cpu") == 0) {
            check_cpu = true;
        } else if (strcmp(argv[i], "--seed-defect") == 0) {
            seed_defect = true;
        } else if (strcmp(argv[i], "--shape-mismatch-control") == 0) {
            shape_mismatch_control = true;
        } else if (strcmp(argv[i], "--direct-q8-1-control") == 0) {
            direct_q8_1_control = true;
        } else if (strcmp(argv[i], "--json") == 0) {
            emit_json = true;
        } else if (strcmp(argv[i], "--act-type") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "q8_1") == 0) {
                act_type = GGML_TYPE_Q8_1;
            } else {
                act_type = GGML_TYPE_F32;
            }
        } else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            custom_m = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            custom_k = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            custom_tokens = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--shape") == 0 && i + 1 < argc) {
            custom_name = argv[++i];
        }
    }

    ggml_backend_t cuda_backend = ggml_backend_cuda_init(device_id);
    if (!cuda_backend) {
        fprintf(stderr, "Failed to initialize CUDA backend device %d\n", device_id);
        return 2;
    }

    if (shape_mismatch_control) {
        // Rejecting control: deliberate contraction dimension mismatch (K=1024 vs K=2048)
        BenchResult res;
        bool ok = run_single_benchmark(cuda_backend, "control_shape_mismatch", 1024, 1024, 200,
                                       act_type, 1, 1, &res, /*k_act=*/2048);
        ggml_backend_free(cuda_backend);
        if (!ok) {
            printf("control_shape_mismatch: successfully rejected (expected failure)\n");
            return 0; // successfully caught
        }
        fprintf(stderr, "control_shape_mismatch: FAILED to reject invalid shape\n");
        return 1;
    }

    if (direct_q8_1_control) {
        // Direct Q8_1 activation tensor on CUDA backend
        // GGML CUDA backend expects F32 activation and internally converts to block_q8_1_mmq.
        // Passing direct Q8_1 activation triggers GGML_ASSERT(src1->type == GGML_TYPE_F32).
        printf("Running direct_q8_1_control (expecting rejection on CUDA backend)...\n");
        fflush(stdout);
        BenchResult res;
        bool ok = run_single_benchmark(cuda_backend, "control_direct_q8_1", 1024, 1024, 200,
                                       GGML_TYPE_Q8_1, 1, 1, &res);
        ggml_backend_free(cuda_backend);
        if (!ok) {
            printf("direct_q8_1_control: rejected: %s\n", res.error.c_str());
            return 1;
        }
        printf("direct_q8_1_control: accepted\n");
        return 0;
    }

    if (check_cpu) {
        ggml_backend_t cpu_backend = ggml_backend_cpu_init();
        if (!cpu_backend) {
            fprintf(stderr, "Failed to initialize CPU backend\n");
            ggml_backend_free(cuda_backend);
            return 2;
        }

        double max_diff = 0.0, nmse = 0.0;
        bool pass = check_cpu_vs_gpu_output(cuda_backend, cpu_backend, 512, 1024, 200,
                                            seed_defect, &max_diff, &nmse);
        ggml_backend_free(cpu_backend);
        ggml_backend_free(cuda_backend);

        printf("cpu_vs_gpu_check: pass=%d nmse=%.6e max_diff=%.6f seed_defect=%d\n",
               pass ? 1 : 0, nmse, max_diff, seed_defect ? 1 : 0);

        if (seed_defect) {
            // Defect was seeded; we EXPECT pass to be false
            if (!pass) {
                printf("seeded_defect_control: successfully rejected corrupted output\n");
                return 0;
            } else {
                fprintf(stderr, "seeded_defect_control: failed to reject corrupted output\n");
                return 1;
            }
        }

        return pass ? 0 : 1;
    }

    std::vector<ShapeDef> targets;
    if (all_shapes) {
        for (size_t i = 0; i < NUM_CANONICAL_SHAPES; i++) {
            targets.push_back(CANONICAL_SHAPES[i]);
        }
    } else if (custom_m > 0 && custom_k > 0 && custom_tokens > 0) {
        targets.push_back({custom_name.c_str(), custom_m, custom_k, custom_tokens});
    } else {
        // Default to all canonical shapes
        for (size_t i = 0; i < NUM_CANONICAL_SHAPES; i++) {
            targets.push_back(CANONICAL_SHAPES[i]);
        }
    }

    std::vector<BenchResult> results;
    for (const auto & s : targets) {
        BenchResult res;
        bool ok = run_single_benchmark(cuda_backend, s.name, s.m, s.k, s.tokens,
                                       act_type, warmup, reps, &res);
        if (!ok) {
            fprintf(stderr, "Benchmark failed for shape %s (M=%ld, K=%ld, tokens=%ld): %s\n",
                    s.name, s.m, s.k, s.tokens, res.error.c_str());
            ggml_backend_free(cuda_backend);
            return 1;
        }
        results.push_back(res);
    }

    ggml_backend_free(cuda_backend);

    // Print table
    printf("\n=== llama.cpp CUDA MMQ Q8_0 Prefill Projection Latency ===\n");
    printf("Metric: device_ns via CUDA events on backend stream (reps=%d, median reported)\n", reps);
    printf("Rate formula: blocks = K/32, MACs = M*tokens*blocks*32, GMAC/s = MACs / device_ns\n\n");

    printf("| shape | M | tokens | blocks | dev_min ns | dev_med ns | dev_max ns | MACs | GMAC/s |\n");
    printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");

    for (const auto & r : results) {
        printf("| %-12s | %4ld | %6ld | %6ld | %10.0f | %10.0f | %10.0f | %12lu | %7.2f |\n",
               r.name.c_str(), r.m, r.tokens, r.blocks,
               r.dev_min_ns, r.dev_med_ns, r.dev_max_ns,
               r.macs, r.gmac_s);
    }
    printf("\n");

    if (emit_json) {
        printf("{\"results\":[");
        for (size_t i = 0; i < results.size(); i++) {
            const auto & r = results[i];
            printf("%s{\"shape\":\"%s\",\"m\":%ld,\"k\":%ld,\"tokens\":%ld,\"blocks\":%ld,"
                   "\"macs\":%lu,\"dev_min_ns\":%.0f,\"dev_med_ns\":%.0f,\"dev_max_ns\":%.0f,\"gmac_s\":%.2f}",
                   (i > 0 ? "," : ""),
                   r.name.c_str(), r.m, r.k, r.tokens, r.blocks,
                   r.macs, r.dev_min_ns, r.dev_med_ns, r.dev_max_ns, r.gmac_s);
        }
        printf("]}\n");
    }

    return 0;
}
