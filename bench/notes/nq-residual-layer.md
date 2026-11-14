# A two-layer dense sign residual comes within 1.1% of Q4_K_M at 80.5% of its bytes

Status: the sizes, reconstruction errors and perplexities below are `measured` on the 8B pack and
the version-3 packs through the engine, on one rented CPU instance. The previous version of this
note carried a PPL projection (`model`) that assumed the excess scales with total squared weight
error; the runs falsify it, so the projection is removed rather than revised. Nothing here is a
proof.

## Result

1. The served NQ format is a binary sign-rank product `s_post·U·diag(s_mid)·V·s_pre` plus a
   `k`-column sparse term. Its error is **not rank-limited**. `k` has 1024 output rows and rank
   2528, beyond the 1024 rows of the product, and its relative error is 0.4727; gate at rank 3040
   of 4096 is 0.4697. Exact Eckart–Young floors are 0.1454 (gate, rank 3040) and 0.3424 (down,
   rank 3328), so the fit is 3.2× and 1.6× above the unconstrained floor in raw Frobenius, but
   extra rank buys almost nothing on the served metric.
2. The format's bit ceiling for a 3:1 FFN matrix is `(r+16)(a+b)` bits with `r ≤ min(a,b)`:
   **1.3336 b/w** for gate/up/down at 12288×4096. The pack spends 1.007/0.987/1.059. There is no
   way to spend the size headroom inside this cell without adding a new section.
3. The residual after the served factor is white. Sign quantization captures 0.6308 of its energy
   on L0 gate and 0.6290 on L0 down, against the Gaussian `2/π = 0.6366`; per row 0.6123–0.6509.
   No low-rank structure is left to extract.
4. One dense 1-bit residual layer with a per-row bf16 scale cuts the squared error to 0.369×:
   gate 0.4684 → 0.2846, down 0.5506 → 0.3354. Two layers reach 0.1730 / 0.2043. The fit is
   closed-form — sign of the residual, `α = mean|residual|` per row — not an ADMM refit.
5. **The sparse columns must be refit after the residual.** The pack stores `k = 5` or `13`
   salient columns per row in f16 because they dominate the loss. The sign step adds `±α` at
   every column; on the sampled sections `α / mean|S|` is 4.98 (L0 gate), 6.87 (L0 down), 7.48
   (L10 q) and 14.21 (L35 down). Without the refit the first layer measured 1,376,970 PPL on the
   first window and the Q8_0 bake 2,646,480 over 1008 tokens. With `S = W_sal − (U·V)_sal − R_sal`
   the first window is 14.146. Decoding the refit packs and comparing `(U·V)_sal + S + R_sal` with
   the official weights gives a maximum error of 2.7e-5, 0.13% of mean |W|, against alpha-level
   before.
6. **One layer is worse than none; two layers recover.** After the refit, r1 is 51.153 against
   r0's 28.308 over 1008 tokens, although its raw Frobenius error is lower. The unweighted sign
   step is not aligned with the calibration-weighted metric the served factor minimises, and the
   first bit costs 0.592 nats/token. The second step fits what is left and brings the same run to
   16.295, 0.552 nats/token better than r0.
7. The bar is official `Qwen3-8B-Q4_K_M.gguf` at **16.124694** PPL over 1008 tokens. The earlier
   context-128 bar, 10.988133, is a different checkpoint (Qwen3-8B-Base); the NQ source is the
   official instruct checkpoint, whose `output.weight` is byte-identical to the official Q8_0.
   Comparing the pack to the base checkpoint understated its ratio by about 1.47×.
8. Measured at ctx 128, 1008 scored tokens: r0 (v2, 2,305,675,376 B, 45.9% of Q4_K_M)
   **28.307562** = 1.756×; r1 (one refit layer, 3,176,698,976 B, 63.2%) **51.153269** = 3.172×;
   r2 (two refit layers, 4,047,721,568 B, 80.5%) **16.295301** = **1.011×**. The 1.15–1.5× band
   is 18.543–24.187; r2 is below the band's lower edge (better), with a 1.06% PPL excess over
   Q4_K_M against the 15% the edge allows. On the same full span, r2 is 0.980 GB (19.5%) smaller
   than Q4_K_M.

## PPL

Protocol: raw wikitext-2 test file, Qwen3 tokenizer, consecutive non-overlapping windows,
`--second-half` scores `ctx/2 .. ctx-2`, `Q8Act` trunk and head, exact packed linears. The first
window is the first 128 tokens (63 scored); the full run is the first 2048 tokens (1008 scored).
The receipt is `bench/cpu/1794653685-nq8b-residual-ppl.md`.

| config | b/w on linears | served bytes | % of Q4_K_M | first window | full 1008 |
|---|---:|---:|---:|---:|---:|
| r0, v2 | 1.124186 | 2,305,675,376 | 45.9% | 9.359791 | 28.307562 |
| r1, v3 no refit | 2.127413 | 3,176,698,976 | 63.2% | 1,376,970.012849 | — |
| r1, v3 refit | 2.127413 | 3,176,698,976 | 63.2% | 14.146016 | 51.153269 |
| r2, v3 no refit | 3.130640 | 4,047,721,568 | 80.5% | 6.953788 | — |
| r2, v3 refit | 3.130640 | 4,047,721,568 | 80.5% | 5.785161 | 16.295301 |
| Q4_K_M, official | — | 5,027,783,488 | 100% | 5.040663 | 16.124694 |

The 63-token first window is a screen, not an estimate of the full ratio: it puts r2 at 1.148×
and r1 at 2.806×, while the 1008-token row gives 1.011× and 3.172×. The ranking of r0 and r2 is
stable; the ratio is not.

The same-engine no-refit rows separate the refit from the residual: one layer 1,376,970 → 14.146,
two layers 6.954 → 5.785 on the first window. The Q8_0 bake agrees with the engine where both
exist: r0 9.332 vs 9.360 and r2-no-refit 7.008 vs 6.954 on the first window, and r0 28.294 vs
28.308 on the full span. The bake reproduced the r1 catastrophe over 1008 tokens (2,646,480), so
the catastrophe is not a Q8_0 requantisation artifact. The exact and Q8Act kernels are both
catastrophic on the unrefit r1 image; the kernel mode is not the cause.

## Size and rate

A residual layer is 1 bit per fitted weight plus one bf16 row scale: **871,022,592 B** of factor
payload (868,220,928 B of signs and 2,801,664 B of row scales). Two layers add 1,742,045,184 B of
factors; the r2 pack is 1,742,046,192 B larger than r0, which is 34.6% of Q4_K_M's size.

| pack | linears | factor bytes | b/w | pack + skeleton | % of Q4_K_M |
|---|---:|---:|---:|---:|---:|
| r0 v2 | 252 | 976,041,584 | 1.124186 | 2,305,675,376 | 45.9% |
| r1 v3 | 252 | 1,847,064,176 | 2.127413 | 3,176,698,976 | 63.2% |
| r2 v3 | 252 | 2,718,086,768 | 3.130640 | 4,047,721,568 | 80.5% |

r2 reaches Q4_K_M within 1.06% at 80.5% of its bytes. Including the 1.330 GB skeleton, r2 stores the
8,709,518,112 weights at 3.718 b/w, against Q4_K_M's 4.618 b/w: 0.90 b/w cheaper overall at a 1.06%
PPL excess. The last 0.980 GB of Q4_K_M buys 0.0105 nats/token over r2, while r0's 2.306 GB leaves
it 0.5628 nats/token behind Q4_K_M. In this cell the two residual layers are the efficient part of
the budget, not the base factor.

## What it changes

- **Pack v3** (`src/format/nq.rs`): a per-linear residual section — `out × ⌈inn/32⌉` u32 sign
  words and one bf16 row scale per layer, a 36-byte section header, `stored_bits` including the
  residual, and a v2 reader/writer for packs with no residual. The v2 writer is byte-identical to
  the served pack `1f4cc4b4…`.
- **Kernel** (`src/engine/kern/nq.rs`): one more sign dot per output row,
  `y += Σ_l s_res[l,i]·Σ_j (2·bit−1)·x[j]`, after the low-rank and sparse terms. The dot is
  negated because `sign_dot!` maps bit 1 to `−1`; `U` and `V` cancel that inversion as a product,
  a single residual plane does not. The accumulator bound is checked in `i64`/`i128` and refuses
  on overflow; there is no new multiply path.
- **Fit** (`tools/nq_convert.py`): `--residual N` and `--dense-ref`, the closed-form sign steps,
  and the sparse refit. The refit is what makes the first layer usable; the no-refit variant is
  kept only as the measured counterexample.
- **Bake** (`tools/nq_residual_bake.py`): refits the salient columns too, and is documented as a
  Q8_0 proxy. The engine v3 pack, not the bake, is the measurement.

## What is not covered

- One dataset and one context length; the 63-token first window is a screen, the 1008-token row is
  the comparable number. Context 512 was not re-run against the corrected bar.
- The residual is fit against the official Q8_0 weights and is unweighted; a calibration-weighted
  refit or a teacher would fit a different objective and is not measured.
- The residual is two independent binary planes (2 b/w). A 2-bit four-level residual at the same
  2 b/w has not been fitted; it may do better or worse than two sign steps.
- The kernel is Rust. No Lean refinement theorem for the residual dot is proved here; the
  maintainer gate (pinned Aeneas/Charon and `proof/`) is not run. No claim in this note is
  `proved`.
- The wasm32 crate compiles against the v3 engine, but no v3 pack has been loaded there.
- rustc and LLVM code generation, the kernel ABI, the GPU path and the `unsafe` container are
  outside the measurement.

## Proof

The maintained extraction unit does not contain the quantiser or this kernel, and this machine has
no pinned Aeneas/Charon install. The property worth proving is local and integer-valued: the sign
dot `Σ (2·bit−1)·x[j]` is the exact sum in `i64`/`i128` under the checked `n·max|x|` bound, and the
only rounding is the documented `mul_q32` epilogue. That is a refinement theorem over
`src/engine/kern/nq.rs`, not over the fit.

The loss is not a kernel defect: the served factors reconstruct to the exporter's own relative
error, the residual is white at the Gaussian constant, and the refit pack reproduces the sparse
columns to f16 precision. The residual layers are what move the perplexity.
