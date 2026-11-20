# The `Q5_0` prefill hole: 54% of the demo file read once per token

Criterion: **T2** (turn latency), partial. The 1,000-token-at-200k turn is not measured; this
receipt covers the 42-token first turn on the demo file. Grade: `measured` for the timing table,
`tested` for the kernel equality. The 7B pack, the 30 tok/s target and the maintainer gate are not
covered.

## 1. The hole

The demo file is Qwen2.5-0.5B-Instruct Q4_K_M, 491,400,032 B, sha256
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`. `Q4_K_M` is not mostly
`Q4_K`:

| ggml type | bytes | share of file | pooled before |
|---|---:|---:|---|
| `Q5_0` | 266,747,904 | 54% | **no** |
| `Q8_0` | 146,145,280 | 30% | yes |
| `Q6_K` | 42,926,080 | 9% | yes |
| `Q4_K` | 29,360,128 | 6% | yes |

`step::cols::proj_cols` admitted `Q8_0`, `Q4_K`, `Q6_K` and `F16` to the multi-column prefill. Its
`Q5_0` arm did not exist, and `matmat::pooled` called only the `Q8_0` row kernel. A 42-token prompt
therefore took one single-column projection per token for the majority of the weights: 266.7 MB
read 42 times, **11.2 GB**, against 0.27 GB if pooled. That was the first-token wait.

## 2. The fix

One row loop, a carrier dial, in two stages:

| stage | change | file |
|---|---|---|
| 1. pooled admission | `Q5_0` admitted; `Family` on `pooled` / `PooledRowsCtx`; `Q8_0` and `Q5_0` share one loop | `src/engine/kern/matmat.rs`, `src/engine/step/cols.rs` |
| 2. decode once | `q5_0_codes4` / `q8_0_codes4` decode four blocks once per group; `dot32x4_codes` folds them against each column | `src/engine/chart/x86.rs`, `src/engine/chart/mod.rs` |

Stage 1 removed the repeated DRAM reads. Stage 2 removed the repeated block decode: the first
`q5_0_row_kcols_avx2` called `q5_0_dot32x4_ptr` once per column, and that function decoded the
four blocks inside the call, so a 42-column prefill decoded the same weights 42 times. The
disassembly is in `bench/cpu/1794793277-kernel-asm.md`: 139 instructions per column call, about
100 of them decode. The split decodes once and leaves 55 instructions per column.

The `Q5_0` block dot already existed (`q5_0_dot32x4_ptr`, `q5_0_dot32_ptr`); the new functions
are the row wrapper and the split. No arithmetic was restated.

## 3. Equality

```
cargo test --lib --locked engine::kern::matmat
```

8 passed, 0 failed, 3 ignored. `columns_past_max_k_match_per_column` runs both carriers over
`k ∈ {17, 24, 33, 42, 64, 96}` and asserts that the pooled kernel takes the call and that every
column equals the per-column `matvec_q8act` bit for bit. The test is `tested`; the AVX2 core is not
extracted to Lean.

## 4. Timing

Machine: Intel Core i5-10210U, 4 cores / 8 threads, 15 W, 6.7 GiB RAM, Ubuntu 24.04, rustc
1.99.0-nightly `0e29c21d9` (2026-07-21), source `735a7b06` plus the working tree. Command:
`./demo/start.sh`, then `./demo/ask.sh "<42-token prompt>"`, `APEIRON_THREADS=4`,
`--kernel q8act`, 96 max tokens, greedy. The prompt is the client-note summarisation prompt in
`demo/recorded/live-answer.txt`; the prefill is 42 tokens and the answer is 18 tokens.

| arm | TTFT (s) | decode (tok/s) | end-to-end (s) |
|---|---:|---:|---:|
| before, one run | 3.71 | 6.81 | 6.36 |
| stage 1 (pooled), 3-run mean | 2.26 | 9.86 | 4.09 |
| stage 2 (decode once), run 1 | 1.82 | 10.31 | 3.57 |
| stage 2, run 2 | 1.79 | 10.17 | 3.56 |
| stage 2, run 3 | 1.90 | 10.24 | 3.65 |
| stage 2, mean | **1.84** | **10.24** | **3.59** |

Prefill rate: 42 / 1.84 = **22.8 prompt tok/s**, from 42 / 3.71 = 11.3. The before arm is one run;
the after arms are three. The prefill change is 2.02× and is larger than the run-to-run spread. The
stage-1 to stage-2 change (2.26 → 1.84 s, 1.23×) is the decode-once split. The decode arm also
measured higher, but a `k > 1` prefill change does not explain decode; that difference is not
attributed to this fix.

A second, 50-token prompt (`Write a warm, two-sentence welcome message...`) measured 1.33 s TTFT,
9.97 tok/s decode, 6.34 s end-to-end. It is one run and is reported as a check, not as a mean.

The same box, llama.cpp b12428 on the same file at `-t 2`: 27.7 prompt tok/s and 10.5 gen tok/s.
The fixed engine is 1.21× slower on this prefill and 1.03× slower on decode. Before the fix it was
2.45× slower on prefill.

## 5. What this does not cover

- **G2.** `src/engine/kern/matmat.rs` is on the extracted path. `proof/regen.sh --check` and
  `lake build` are owed on a host with the pinned Aeneas, Charon and Lean. The existing kernel had
  no committed refinement theorem for the pooled row kernel; the new `Q5_0` row is in the same
  position.
- **30 tok/s.** The 7B pack at 1.174 b/w is 957,615,232 B; 30 tok/s reads 28.7 GB/s. This receipt
  measures a 0.49 GB 0.5B file and does not establish the 7B rate.
- **The fold.** `Q8_0` still takes the VNNI pooled entry, and `Q4_K`/`Q6_K` still take
  `chart::kcols`. They are the next one-path fold, not this receipt.
- **The oracle.** The per-column `matvec_q8act` remains the correctness oracle and the fallback for
  a refusal. It is not a production route.
- **The rest of the turn.** Tokenisation, sampling, the serving loop and the network listener are
  outside the timed kernel.
