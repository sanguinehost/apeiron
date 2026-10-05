# GPU decode vs llama.cpp (Qwen2.5-0.5B Q4_K_M, tg128)

Grade: `measured`. Contended repeated measurement (shared box, load average 49 at start), not a quiet-host number.

## Identity

| item | value |
|---|---|
| apeiron | `a090aaa` + dirty (waves 2-5 uncommitted; worktree `/home/socol/oc-tmp/wt-inf-w5`); `src/gpu/decode/*`, `examples/gpu_decode_bench.rs` |
| model | `/home/socol/models/gguf/qwen05b-Q4_K_M.gguf`, sha256 `ebd3d6faed5ef9dfa6770a9d36fbb91f421e67ff08b102c17244970fd11cec19` |
| machine | i9-14900KF, RTX 5080 (driver 615.71.09), cpu governor `powersave`; Vulkan device also drives the desktop (sunshine, vivaldi) |
| llama.cpp Vulkan | `/home/socol/Workspace/llama.cpp/build-vulkan/bin/llama-bench`, build `0324696b8 (10152)` (binary dated Aug 6; checkout HEAD is `ece963f41`, so this binary is older than the checkout) |
| llama.cpp CUDA | `build-cuda/bin/llama-bench`, build `ece963f41 (10450)` (reference only) |
| rustc | 1.97.0-nightly (17584a181 2026-04-13) |
| 8B / 27B rows | not run: `/mnt/storage` unmounted |

## Commands

```sh
# apeiron (built once: cargo build --release --locked --features gpu --example gpu_decode_bench)
target-w5/release/examples/gpu_decode_bench /home/socol/models/gguf/qwen05b-Q4_K_M.gguf --gen 128 --reps 5 --carrier q4
target-w5/release/examples/gpu_decode_bench /home/socol/models/gguf/qwen05b-Q4_K_M.gguf --gen 128 --reps 5 --carrier q8
# llama.cpp
build-vulkan/bin/llama-bench -m <gguf> -ngl 99 -p 0 -n 128 -r 5 -o jsonl
build-cuda/bin/llama-bench   -m <gguf> -ngl 99 -p 0 -n 128 -r 5 -o jsonl
```

Interleaved: 3 rounds of apeiron-q4, apeiron-q8, llama-Vulkan, llama-CUDA (`/home/socol/oc-tmp/w5-campaign.sh`, 58 s wall,
every arm exit 0; raw output `/home/socol/oc-tmp/w5-campaign/`). Each arm-round is 5 samples; 15 samples per arm.
apeiron: one construct per process, 16 warm-up steps, then 5 timed reps of 128 greedy device-argmax steps from the same
prefix (`tmax 512`, prompt "The capital of France is"). Device argmax vs host scan over the 128 steps: 0 mismatches in all 6 runs.

## Result (tok/s, 15 samples per arm)

| arm | mean | min-max | vs llama Vulkan | vs llama CUDA |
|---|---:|---|---:|---:|
| apeiron `--carrier q4` | 728.4 | 544.7-777.4 | 1.266 | 0.889 |
| apeiron `--carrier q8` | 694.2 | 487.3-757.0 | 1.207 | 0.848 |
| llama.cpp Vulkan | 575.4 | 516.1-622.0 | 1.000 | 0.703 |
| llama.cpp CUDA | 818.9 | 671.9-861.7 | 1.423 | 1.000 |

## Reading

- apeiron/llama-Vulkan 1.27x (q4), 1.21x (q8) on this model at tg128. Every apeiron rep after rep 0 (711-777 q4, 702-757 q8) is above every llama-Vulkan sample (516-622); rep 0 of each run (487-642) is the first timed replay after construct and overlaps llama-Vulkan.
- apeiron vs llama-CUDA: 0.89x (q4), 0.85x (q8): apeiron loses to CUDA.
- q4 vs q8 (apeiron): +4.9% on the means, inside the observed spread; the carrier effect is not resolved by this run. In this checkpoint only `ffn_down` of 12 of 24 layers is Q4_K (`k=4864`, a multiple of 256). `attn_q/k/o` and `ffn_gate/up` are Q5_0, `attn_v` is Q8_0 (12 layers) or Q5_0 (12), `token_embd` Q8_0, the other 12 `ffn_down` Q6_K (tensor types from the GGUF header). `Carrier::Q4` therefore keeps 12 of 96 projection charts as Q4_K and packs the rest at Q8. The Q4-everywhere byte traffic llama.cpp reads is not reproduced for this file.
- Clock samples (`nvidia-smi`) were taken before each arm and show 2917-2932 MHz SM in 11 of 12 samples (one 795 MHz idle sample, before round 1 apeiron-q4); utilization in those samples is the previous arm's tail, not a quiet-host baseline. Other GPU clients: sunshine, vivaldi.
- Not reproduced here: prefill (`pp`), 8B/27B rows, llama-Vulkan at the checkout HEAD (binary is `0324696b8`).
