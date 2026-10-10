# ONEBIT W9: whole-model one-bit conversion of Qwen2.5-0.5B, paired KLD against f16

Board row `onebit-exact-one-bit-and-FubKIC3prjKrgnryWWxrF1`. Lean: `proof/corpus/Perf/OnebitW9.lean`.

| field | value |
|---|---|
| model | `/home/socol/models/gguf/qwen05b-f16.gguf`, sha256 `73f932c0a3a3daf1…` |
| text | wikitext-2-raw `wiki.test.raw`, sha256 `173c87a53759e020…`; 64 chunks of 512 |
| eval | llama.cpp `ece963f41` `build-cuda/bin/llama-perplexity -c 512 -b 512 --chunks 64 -ngl 0 -t 32`, `CUDA_VISIBLE_DEVICES=` (CPU), `--kl-divergence-base f16-wt64.kld --kl-divergence` |
| calibration | C4 (`/home/socol/oc-tmp/seal/calib-c4.txt`) for both sides: our Grams (51200 tokens, `/home/socol/oc-tmp/gptq-w2/gram`, 24 blocks × 4 inputs) and llama.cpp's imatrix (100 chunks) |
| writer | `examples/onebit_model.rs` sha256 `3eb76645b0808704…`, over `src/learn/onebit.rs` `47d0d42f9dd2c474…` |
| machine | i9-14900KF |

## Population

llama.cpp `IQ1_S` and `IQ1_M` need rows divisible by 256. On Qwen2.5-0.5B only the 21 `ffn_down` tensors of
blocks 3–23 get `IQ1`. The width-896 tensors fall back to `iq4_nl`, `attn_v` is `q5_0`, `ffn_down` 0–2 is `q2_K`, and
`token_embd` is `q8_0`. The files are 5.02 and 5.05 BPW. An "IQ1_S model" here is mostly 4.5-bit, so a whole-model
comparison against it is not like-for-like. Two matched populations:

- **P1**: the 21 `ffn_down` tensors (91,521,024 weights). Every other tensor is taken byte-for-byte from the `IQ1_S`
  file. The `IQ1_S` and `IQ1_M` files differ only in these 21 tensors (checked tensor by tensor).
- **P2**: all 168 linear tensors at one bit (357,826,560 weights). The base is llama.cpp's own `Q1_0` file, so
  `token_embd` `q8_0` and the norms are identical in every arm.

## Result

| arm | bits/weight on the population | P1 KLD | P1 PPL | P2 KLD | P2 PPL |
|---|---:|---:|---:|---:|---:|
| f16 against itself (control) | 16 | 0.000000 | 13.4203 | — | — |
| `Q4_K_M` imatrix, whole model (397,807,776 B file) | — | 0.0243 | 13.6602 | — | — |
| llama.cpp `IQ1_M` imatrix | 1.75 | **0.5191** | 22.0078 | — | — |
| llama.cpp `IQ1_S` imatrix | 1.5625 | 0.6315 | 24.3413 | — | — |
| `had-gptq1` (ours) | 1.125 + online R₁₂₈ | 0.8345 | 30.0687 | **6.8496** | 12500.0 |
| `gptq1` (ours) | 1.125 | 5.2429 | 2426.96 | 15.3005 | 56.5M |
| llama.cpp `Q1_0` | 1.125 | 10.3846 | 396012 | 13.1667 | 6.55M |
| `rtn` (ours, `Q1_0` writer) | 1.125 | 10.3846 | 396012 | 13.1667 | 6.55M |

KLD ± (standard error of the mean) is at most 0.027 on every row. Same-top-token rates: `IQ1_M` 65.9%,
`IQ1_S` 62.0%, `had-gptq1` P1 58.2%, P2 1.4%.

- **Same 1.125 b/w:** `had-gptq1` beats llama.cpp's `Q1_0` by 12.4× in KLD on P1 (0.8345 against 10.3846), and by
  1.92× on P2.
- **Against `IQ1`, P1:** `IQ1_S` is 1.32× lower KLD with 1.39× our bits, and `IQ1_M` 1.61× lower with 1.56× our bits.
- **Whole model, P2:** no post-training one-bit arm is usable. The best KLD, 6.85 nats, is 282× the `Q4_K_M`
  reference. One-bit weights on every linear tensor need training (`LEARN-HATTN` T4 and the fold training path), not
  post-training rounding.
- `gptq1` beats `rtn` on P1 but loses on P2 (15.30 against 13.17). Its Grams come from f16 activations; with every
  layer at one bit the activations it was fitted on are not the ones it receives.
- `gptq1-act` cannot be written as `Q1_0`. Its act-order scale groups are 128 consecutive columns in permuted order,
  so a natural-order 128-block mixes scales (refused by the writer: `blk.3.ffn_down` block 0, |w₁| ≠ a). Natural-order
  `gptq1` replaces it.

## Storage of `had-gptq1`

`had-gptq1` quantises `W·R` with R the block-diagonal `H₁₂₈/√128`, then stores `R·(±a)` rotated back. Here it is
written as F16 (P1 file 480,996,000 bytes, P2 866,530,464 bytes). Rotating each 128-block of the stored weights by
`R` gives `±a` to within 0.29% (f16 rounding) on all 715,008 P1 and 2,795,520 P2 blocks. So the arm is a `Q1_0`
tensor at 1.125 b/w plus an online R₁₂₈ (7 butterfly stages) on that tensor's input activations. As `Q1_0` the files
would be the size of the `rtn` files: P1 310,824,096 bytes, P2 201,196,704 bytes.

## Controls

| control | result |
|---|---|
| identical weights | f16 against itself: KLD 0.000000, same top 99.982% |
| writer (P1) | our `rtn` as `Q1_0` is byte-identical to llama-quantize's `Q1_0` on the same 21 tensors (`cmp`; 715,008 blocks; sha256 `6ca38c104d5e0ff6…`); KLD equal to 6 decimals |
| writer (P2) | 2,795,520 blocks: 357,826,560 sign bits identical; 1 scale differs by one f16 ulp (`blk.11.attn_q` block 5690, 0.0141449 against 0.0141373; f64 against f32 mean); KLD 13.166692 against 13.166693 |
| storage | `rtn` as `Q1_0` against as F16 on P1: KLD 10.384596 against 10.385443 |
| regeneration | the committed writer regenerates `p1-rtn-q1` and `p1-gptq1-q1` byte-identical |
| Lean | `lake build Perf.OnebitW9 AxiomAudit` green (9167 jobs, 54 s), every verdict audits with an empty axiom set; mutating `p2HadGptq1` to 6000000 and `p1Rtn` to 10384597 in a throwaway copy turns two `decide`s red; restored file green |

## Not covered

- One model (0.5B), one text, 512-token windows.
- No one-bit kernel runs here; llama.cpp's CPU path evaluates every arm. The online R₁₂₈ is not implemented in a
  kernel; the F16 file stands in for it.
- `Q4_K_M` covers the whole model, not a matched population.
- Blocks 4–23 of the W7 per-tensor table remain open.

Artifacts: `/home/socol/oc-tmp/onebit-w9/` (`kld-*.log`, `gen-*.log`, `p1-*.gguf`, `p2-*.gguf`, `kld.sh`).
File sha256: `IQ1_S` `381b31c7d15e5c1d…`, `IQ1_M` `ac82c0d2cc705be1…`, llama `Q1_0` `0dc798dbbb86b173…`,
`Q4_K_M` `4089649c04adccf3…`, P1 `gptq1` `543a4b111faea2ff…`, P1 `had-gptq1` `4fc43a1d81d06e5f…`,
P2 `gptq1` `6344714b1d089c77…`, P2 `had-gptq1` `48f91fd2363f029b…`, P2 `rtn` `36ceab375ee29e5d…`.
