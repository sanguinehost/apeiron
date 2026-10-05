# GPU prefill vs llama.cpp (Qwen2.5-0.5B Q4_K_M, pp512)

Grade: `measured` (speed), `tested` (match). Contended repeated measurement (shared box, load average 17.8 at start), not a quiet-host number. apeiron loses at pp512.

## Identity

| item | value |
|---|---|
| apeiron | `7b27c18` + dirty (uncommitted; worktree `/home/socol/oc-tmp/wt-inf-prefill`); `src/gpu/decode/prefill.rs` sha256 `510a56f5bdcedfcd…`, `src/simt_nn/prefill_glue.rs` `29c95983d91fa5a9…` |
| model | `/home/socol/models/gguf/qwen05b-Q4_K_M.gguf`, sha256 `ebd3d6faed5ef9dfa6770a9d36fbb91f421e67ff08b102c17244970fd11cec19` |
| machine | i9-14900KF, RTX 5080 (driver 615.71.09); the Vulkan device also drives the desktop |
| llama.cpp Vulkan | `build-vulkan/bin/llama-bench`, build `0324696b8 (10152)` (checkout HEAD is `ece963f41`; the binary is older than the checkout) |
| llama.cpp CUDA | `build-cuda/bin/llama-bench`, build `ece963f41 (10450)` (reference only) |
| rustc | 1.97.0-nightly (17584a181 2026-04-13) |

## Commands

```sh
cd /home/socol/oc-tmp/wt-inf-prefill
CARGO_TARGET_DIR=$PWD/target cargo build --release --locked --features gpu --example gpu_decode_bench
B=target/release/examples/gpu_decode_bench; M=/home/socol/models/gguf/qwen05b-Q4_K_M.gguf
$B $M --prefill 512 --reps 5 --carrier q4          # and --carrier q8
PREFILL_PROFILE=1 $B $M --prefill 512 --reps 3 --carrier q4    # per-node GPU-active us
L=/home/socol/Workspace/llama.cpp
$L/build-vulkan/bin/llama-bench -m $M -ngl 99 -p 512 -n 0 -r 5 -o jsonl
$L/build-cuda/bin/llama-bench   -m $M -ngl 99 -p 512 -n 0 -r 5 -o jsonl
```

Campaign: `/home/socol/oc-tmp/pp-campaign/run.sh` (3 interleaved rounds of apeiron-q4, apeiron-q8, llama-Vulkan, llama-CUDA; 40 s wall; every arm exit 0; raw output `/home/socol/oc-tmp/pp-campaign/`). Each arm-round is 5 samples; 15 samples per arm. apeiron: one construct per process, one untimed warm-up prefill, then 5 timed `GpuDecode::prefill` calls of 512 pseudo-random token ids (`tmax 528`), each timed wall (host token write, one submit, readback). llama-bench `-p 512` feeds random ids too.

## Result (tok/s, 15 samples per arm)

| arm | mean | min-max | vs llama Vulkan | vs llama CUDA |
|---|---:|---|---:|---:|
| apeiron `--carrier q4` | 26587.8 | 22003.5-29373.1 | 0.525 | 0.422 |
| apeiron `--carrier q8` | 24495.7 | 21963.9-28736.5 | 0.483 | 0.389 |
| llama.cpp Vulkan | 50682.8 | 41526-56110 | 1.000 | 0.804 |
| llama.cpp CUDA | 63018.5 | 42882-70377 | 1.244 | 1.000 |

Round means (tok/s): apeiron-q4 23476 / 27977 / 28310; apeiron-q8 27746 / 22750 / 22992; Vulkan 44967 / 52469 / 54613; CUDA 63202 / 64309 / 61544.

## Reading

- apeiron/llama-Vulkan 0.53x (q4), 0.48x (q8); apeiron/llama-CUDA 0.42x, 0.39x. Every apeiron sample (22.0k-29.4k) is below every llama-Vulkan sample (41.5k-56.1k): a loss in every round.
- The apeiron run-level mean is bimodal: runs sit near 23k or near 28k, tight within a run (spread inside a run <= 7 %), flipping between processes independent of carrier and of the `DEC_PREFILL_BT`/`DEC_PREFILL_RT` tile (7 successive runs over 7 tile settings alternated 23.5-23.9k / 28.9k). The cause is not isolated; a tile sweep under this variance resolves nothing.
- Node profile, one pp512 call, GPU-active (`PREFILL_PROFILE=1`, q4, one run, 22156 us total, summed over 24 layers): GEMMs 18.2 ms (82 %: gate/up 6486, down 4017+4007, qkv 1646, o 1551, lm head 502 us), attention (`attn_prefill_tiled` 32:4:2) 2367 us (11 %), RMSNorm+quantize 423, activation quantize 535, SwiGLU 283, RoPE+KV 166, KV re-layout 101 us.
- Body GEMM work is 183 GMAC (358 M weights x 512); 18.2 ms is 10 TMAC/s = 20 TOPS int8 DP4A, against the 133 TOPS DP4A ceiling stated in `src/simt/superblock.rs` and the 51.7 TOPS standalone gate/up figure in the `prefill_tile` comment of fold `bedrock/src/serve/gpu.rs` (d497161f); both quoted, not re-measured. In this checkpoint `attn_q/k/o` and `ffn_gate/up` are Q5_0 and 12 `ffn_down` are Q6_K, so `Carrier::Q4` keeps only 12 of 96 projection charts as Q4_K and packs the rest at Q8 (as in the decode receipt).
- Not reproduced here: llama-Vulkan at the checkout HEAD, 8B/27B rows, a quiet-host run.

## Match (`tests/gpu_prefill_match.rs`, release, `--ignored`)

Qwen2.5-0.5B-Q4_K_M, five prompts, `TMAX 256`, both carriers (q4, q8). First token from whole-prompt GPU prefill equals token-by-token `GpuDecode::step` and the CPU engine (`KernelMode::Exact`) on all 10 (prompt, carrier) cases. Hidden max relative error at the last prompt position, prefill vs token-by-token, defined `max|d| / max|ref|`: 0.14-0.24 (q8 worst 0.2145, q4 worst 0.2434); relative L2 0.27-0.40. Logit relative L2 against the CPU exact logits at the last position: prefill 0.050-0.254, token-by-token 0.064-0.233, so the two GPU paths deviate from the CPU by the same order and from each other by 0.07-0.14. Both GPU paths are W8A8 (int8 activations, Q8/Q4 charts); the path-to-path hidden difference is not attributed to a stage (an n=1 prompt, where attention is trivial, shows 0.28-0.42 relative L2).
