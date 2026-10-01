# Dense prefill reuse attribution: Apeiron vs CPU llama.cpp

Status: `measured` observation. One machine, two models, prefill lengths 200 and 512, two
repetitions. No speedup and no optimization is claimed.

```text
SCOPE:      examples/engine_profile.rs (CPU device pin for the llama.cpp child),
            docs/evidence/gpu-prefill/dense-reuse-attribution.md (this file). Completion: the child
            reports a CPU comparator and the per-model prefill numbers are recorded.
REGISTER:   measurement; no algebra, refinement, state or test-contract change.
CORPUS:     src/engine/step/cols.rs, src/engine/kern/matmat.rs read for the reuse mechanism; no
            canonical theorem read or changed.
REFERENCE:  /home/socol/Workspace/llama.cpp/build-cpu/bin/llama-bench;
            /home/socol/Workspace/llama.cpp/tools/llama-bench/llama-bench.cpp.
GATE:       cargo test --locked --example engine_profile; cargo clippy --locked --example
            engine_profile -- -D warnings; cargo fmt -- --check examples/engine_profile.rs;
            two `timeout 540 cargo run --release ... --example engine_profile` invocations per
            model, budget 540 s each.
GRADE:      measured. Before: the comparator child had no device pin; after: GPUs hidden and the
            receipt carries llama-bench's own `gpu_info` / `backends`.
```

## 1 · Comparator device pin

`run_llama` spawns `llama-bench`. Without a pin, a GPU-visible `llama-bench` silently compares the
Apeiron CPU engine against a GPU. The child now runs with every GPU backend hidden
(`CUDA_VISIBLE_DEVICES=""` and `GGML_VK_VISIBLE_DEVICES=""`, both cheap on this machine), and the
run fails rather than writes a receipt if `llama-bench` reports a device. The receipt records the
comparator's own report:

| receipt field | Qwen3-8B pp200 | Qwen3.5-0.8B pp200 | Qwen3.5-0.8B pp512 |
|---|---|---|---|
| `identity.llama_commit` | `0324696b8` | `0324696b8` | `0324696b8` |
| `identity.llama_backends` | `CPU` | `CPU` | `CPU` |
| `identity.llama_gpu_info` | `` (empty) | `` (empty) | `` (empty) |

Comparator binary: `/home/socol/Workspace/llama.cpp/build-cpu/bin/llama-bench` (CPU build).
`--list-devices` on it prints `(none)`.

## 2 · Machine and identity

| item | value |
|---|---|
| CPU | Intel(R) Core(TM) i9-14900KF, 32 online CPUs |
| governor / kernel | `powersave` / `7.2.8-1-cachyos` |
| `rustc` | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| engine `src/` sha256 | `def07db15dc1e7d73853868153fa2501ce80ef8cf8409b28817fc8fbc74f027c` (identical in all three runs) |
| HEAD at run time | 8B pp200 `dcb0a13e4d77c56b03fa00a387265fc28c5896fb`; 0.8B runs `23089351c31cee87252eddd477a0639c6e1e485f`; tree dirty (99 paths) |
| read ceiling reported by the harness | 70.9 / 83.9 / 86.1 GB/s at 8 / 16 / 32 threads (8B pp200 run); 66.7 / 85.6 / 89.6 GB/s (0.8B pp512 run) |

HEAD moved during the slice (another workstream commits in this checkout); `src_sha256` is
unchanged, so the numbers bind to the source digest, not to HEAD.

## 3 · Command

```sh
timeout 540 cargo run --release --locked --example engine_profile -- \
  --model <model.gguf> --ids /home/socol/oc-tmp/w7-fixtures/qwen15.ids \
  --pp <200|512> --tg 8 --reps 2 \
  --llama /home/socol/Workspace/llama.cpp/build-cpu/bin/llama-bench \
  --llama-threads 8,16,24,32 --out <dir>
```

| model | path | sha256 | weight payload |
|---|---|---|---|
| Qwen3-8B-Q4_K_M | `/home/socol/models/gguf/q3-8b/Qwen3-8B-Q4_K_M.gguf` | `120307ba529eb2439d6c430d94104dabd578497bc7bfe7e322b5d9933b449bd4` | 5,027,784,512 B = 5.028 GB / 4.683 GiB |
| Qwen3.5-0.8B-Q8_0 | `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` | `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f` | 833,592,096 B = 0.834 GB / 0.776 GiB |

Prompt ids `/home/socol/oc-tmp/w7-fixtures/qwen15.ids`, sha256
`0e7b28afb1d1aca440ddb0a7aa0987ba7818d97f7b1a3fa2ba0e5b052d6b3c1e`, used for both models.

## 4 · Attribution

Definitions:

- per-token ms = 1000 / (median Apeiron prefill tok/s).
- weights bytes/token = the model file's total payload, i.e. the bytes a per-token, no-reuse stream
  would read for every token. It includes embeddings and norms; they are a small fraction.
- implied GB/s = (median Apeiron prefill tok/s) × (weights bytes/token) / 10^9.
- ratio = median per-round Apeiron prefill tok/s / llama.cpp prefill tok/s at llama's best thread
  count in that round.

| model | pp | Apeiron pp tok/s (median [min, max]) | ms/token | weights B/token | implied GB/s | llama.cpp CPU pp tok/s (best threads) | per-round ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B-Q4_K_M | 200 | 15.18 [14.66, 15.70] | 65.9 | 5,027,784,512 | 76.3 | 81.31 (32t) | 0.19 [0.18, 0.20] |
| Qwen3-8B-Q4_K_M | 512 | blocked | — | — | — | — | — |
| Qwen3.5-0.8B-Q8_0 | 200 | 470.72 [468.37, 473.06] | 2.12 | 833,592,096 | 392.4 | 572.86 (24t) | 0.82 [0.82, 0.83] |
| Qwen3.5-0.8B-Q8_0 | 512 | 434.88 [434.29, 435.47] | 2.30 | 833,592,096 | 362.5 | 467.65 (24t) | 0.94 [0.85, 1.03] |

Run cost:

| model | pp | elapsed | result |
|---|---|---:|---|
| Qwen3-8B-Q4_K_M | 200 | 271 s | receipt `1790845323`, exit 0 |
| Qwen3-8B-Q4_K_M | 512 | 540.04 s | budget exhausted, no receipt |
| Qwen3.5-0.8B-Q8_0 | 200 | 17.25 s | receipt `1790845898`, exit 0 |
| Qwen3.5-0.8B-Q8_0 | 512 | 34.45 s | receipt `1790845932`, exit 0 |

## 5 · Flat or drop

| model | ms/token at pp200 | ms/token at pp512 | change |
|---|---:|---:|---|
| Qwen3-8B-Q4_K_M | 65.9 | blocked | not established |
| Qwen3.5-0.8B-Q8_0 | 2.12 | 2.30 | +8.3 % (no drop) |

The 8B pp512 invocation reached the 540 s ceiling with one of two rounds reported and no receipt
written; it was not retried. So the 8B is reported at pp200 only, and no pp200→pp512 comparison
exists for it.

## 6 · Reading

Two independent signals, each stated with its own limit:

| model / pp | implied GB/s vs measured read ceiling | per-token time vs pp | reading |
|---|---|---|---|
| Qwen3-8B pp200 | 76.3 of 86.1 GB/s at 32 threads — below the ceiling | no pp512 | consistent with streaming the whole payload once per token: no reuse across the 200 columns |
| Qwen3.5-0.8B pp200 | 392.4 GB/s, 4.6× the 86.1 GB/s ceiling | — | full-weight-per-token streaming is excluded on this hardware; weight bytes are reused across the prefill block |
| Qwen3.5-0.8B pp512 | 362.5 GB/s, 4.0× the 89.6 GB/s ceiling | +8.3 % | reuse is present but does not grow with prompt length |

The 8B reading agrees with the epic's open item `docs/epics/gpu-prefill.md:435:Dense Q4_K prefill
has no weight reuse`. At 76.3 GB/s the dense Q4_K path sits under the measured DRAM ceiling, the
shape a per-token weight stream takes.

For the 0.8B, moving the full 0.834 GB payload 392 times per second exceeds the 89.6 GB/s the
harness measures for a 1 GiB sum, so the engine does not read the full payload from DRAM for every
token. The source has the mechanism: the batched `Q8_0` branch in `proj_cols`
(`src/engine/step/cols.rs:224:if ggml_type == quant::GGML_TYPE_Q8_0`) reads the weight payload once
per call for a block of columns, and the block width is default 256
(`src/engine/step.rs:360:.unwrap_or(256)`) with a column cap `MAX_K = 16`
(`src/engine/kern/matmat.rs:47:const MAX_K: usize = 16;`). That cap bounds the reuse factor per
block, so pp200 (one block) and pp512 (two blocks) carry the same per-column reuse factor and the
per-token time does not drop. Which of the block width or the cap sets the observed rate is
`unmeasured` here.

No speedup and no optimization is claimed. This is an observation on one machine with two
repetitions.

## 7 · What this does not cover

- The comparator is a CPU `llama-bench`; no GPU number and no GPU comparison is reported.
- Two models only (dense Q4_K 8B, hybrid Q8_0 0.8B), two prefill lengths, `--reps 2`, governor
  `powersave`.
- The 8B pp512 case is blocked at the 540 s ceiling.
- The per-phase trace byte model in the receipts is not used for `weights bytes/token`; that figure
  is the model file's payload.
- rustc/LLVM codegen, the kernel ABI and every GPU layer are outside this evidence.
