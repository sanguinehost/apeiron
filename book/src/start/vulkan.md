# Run on the GPU

The `gpu` feature adds a Vulkan runtime. It loads `libvulkan.so.1` at run time with `dlopen`. The
feature adds no crates. It is off by default.

```sh
cargo build --release --locked --features gpu --examples
```

The runtime creates a Vulkan 1.1 instance. Some kernels need device features beyond that, such as
cooperative matrices. The `simt_check` example runs all 12 SIMT ops on the device and compares each
with its CPU reference. It exits 0 and prints a skip when no Vulkan loader or device is present, so
it is safe to run anywhere:

```sh
cargo run --release --locked --features gpu --example simt_check
```

## Decode throughput

```sh
cargo run --release --locked --features gpu --example gpu_decode_bench -- <model.gguf> --gen 128 --reps 5 --carrier q4
cargo run --release --locked --features gpu --example gpu_decode_bench -- <model.gguf> --prefill 512 --reps 5 --carrier q4
```

`--gen N` times `N` greedy steps per repetition after a 16-step warm-up. `--prefill N` times a
whole-prompt prefill of `N` pseudo-random tokens. An untimed pass afterwards compares the device
argmax with a host scan of the full logit row.

## Other GPU examples

| example | what it does |
|---|---|
| `simt_gemv` | `--check` compares the device GEMV with the CPU fold; `--bench` reports GFLOP/s |
| `simt_batch` | the batch-tiled integer-dot GEMM at prefill shapes |
| `gemm_bench` | the DP4A batch GEMM against the cooperative-matrix GEMM on Qwen2.5-0.5B layer-0 projections |
| `gpu_ppl` | perplexity of the GPU decode path on Qwen2.5-0.5B, 512-token chunks |
| `gpu_dump` | the f32 forward pass on the device, dumped per layer for Qwen2-family models |

## Limits

The GPU path is single-tenant. Buffers run without `robustBufferAccess` and workgroup memory is not
cleared between dispatches. Do not share one device between mutually untrusted users through this
crate. The [trust boundary](../verification/trust.md) states what is and is not covered.
