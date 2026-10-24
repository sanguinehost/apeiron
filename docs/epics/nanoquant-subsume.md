# NanoQuant subsumption

## Intent — NanoQuant subsumption

- Owner: lead. Board: `APEIRON-QUALITY-118-W1-T01`.
- Source revision: `SamsungLabs/NanoQuant` @ `a9e0a43` (Apache-2.0), checked out at `../NanoQuant`;
  paper arXiv 2602.06694.
- Status: W1 (ADMM port), W2 (layer-scope STE tune), W3 (KL arm) and the sketched tune measured; W4 verdicts
  in `proof/corpus/Learn/Quant/NanoQuant.lean` (Aeneas link open); W5 open.

| reference file | sha256 |
|---|---|
| `src/nanoquant/core/admm_nq.py` | `7145ba305ad6f3e1303dc618722c71e210309160ea3a1c6f6b16bb646009c1a6` |
| `src/nanoquant/compress_block.py` | `23e817f6abef65f902e5a5a7e9399018cd0f32f462f3809887643af23cc19a8f` |
| `src/nanoquant/compress_model.py` | `2acecfe5413b3bd2d680b3e9826074f5d373e44201c127020e9e4dee0bd94212` |
| `src/nanoquant/importance.py` | `1daa708c3763ede072aae021710ee5cf8d86aaa49e6f6d5b47e1fe550284d0a3` |

### Problem

The goal is a Qwen-class model in ≤ 4.0 GB of resident weights (KV excluded), ≤ 1.18 b/w at
Q4_K_M-like quality. NanoQuant is the published sub-1.2 b/w post-training method; per §3a it is
subsumed as the base before any further derivation. The c6x29 gate-code line stops as the base.

### Why the alternatives were rejected

- c6x29 / c6x29-sel: in-house, no published comparison; kept only as a measured incumbent.
- Training from scratch: excluded by owner rule (convert, never retrain).

### Completion condition

The Rust port reproduces the reference pipeline (ADMM, `tune_fact`, global KD) on the engine path, and
each stage's quality is recorded as relH and held-out KL against same-scope Q4_K_M.

### Invariants

- Deployed layer `Ŵ = diag(s_post)·U·V·diag(s_pre)`, `U ∈ {±1}^{o×r}`, `V ∈ {±1}^{r×n}`.
- Stored bits `(r+16)(o+n)`; rank `r = ⌊o·n·b/(o+n)⌋ − 16`, floored to a multiple of 32
  (`admm_nq.py`; `nanoquant::rank_for_bits`, `stored_bits`).
- Every product runs on `engine::kern::limb` (§0 K2), 3 limbs unless a measured metric licenses fewer.

### Verification hypotheses

- The ADMM export equals its dense form within 1e-6·max|w| (tested).
- The H-weighted latent tune (W2) lowers relH below the ADMM-only figure at fixed b/w.

### Non-goals

Convergence theorems for ADMM or the STE tune; CUDA/torch execution.

### Trust boundary

rustc/LLVM; the `limb` kernel's own error bound (§0 K4, open); RNG differs from torch (results are not
bit-reproducible against the reference).

### Open questions

Whether block-scope `tune_fact` (needs the block forward with gradients) beats the layer-scope
analogue enough to pay for it.

## Specification — NanoQuant subsumption

### Scope and admissible domain

`factorize(w, out, inn, i_norm, o_norm, cfg)` over Q32.32 words; `out, inn ≥ 1`, `rank ≥ 1`. A wide
matrix (`inn > out`) is factorised on its transpose. Errors are `NqError` (budget below 16 + 32 rank
rows, size mismatch, overflow); no panic on the contract domain.

### Representation relation (Rust to Lean)

Planned (W4): `Learn.NanoQuant` states `rank_for_bits` and `stored_bits` over `Nat` and links the
Aeneas extraction of the same two functions; the factored forward `diag(s_post)·U·(V·(diag(s_pre)·x))
= Ŵx`.

### Contracts

| function | contract | grade |
|---|---|---|
| `rank_for_bits` | `stored_bits(r) ≤ budget`, `r` maximal multiple of 32 | tested (`rank_fits_budget_and_is_maximal`) |
| `factorize` | signs ±1; `weight()` = dense `diag(s_post)UVdiag(s_pre)`; loss falls with iterations | tested (`export_matches_its_dense_form_and_loss_falls`, `wide_matrix_runs_on_the_transpose`) |
| `tune` | latents carry the exported signs (≥ 99%); `tr(EHEᵀ)` falls; signs stay ±1 | tested (`latents_carry_the_signs_and_tune_lowers_the_h_energy`) |

### Failure semantics

`Result<NqFactor, NqError>`; checked arithmetic on sizes.

### Evidence plan

Layer-scope relH `tr(EHEᵀ)/tr(WHWᵀ)` on the cached calibration Gram, then held-out KL through the
`learn_gate_code` KL arm against same-scope Q4_K_M.

### Deviations from the reference

| item | reference | port |
|---|---|---|
| schedule | CLI default linear `ρ = t/T` (`main.py:81`); function default cubic | `Schedule::Linear` default, `Cubic` selectable |
| `i_norm` | input activation norm from calibration | `diag(H)` of the cached Gram |
| `o_norm` | `importance.py` output importance | 1 (W5) |
| precision | fp32 / TF32 on CUDA (`admm_nq.py:9`) | Q32.32 on the 3-limb integer kernel |
| RNG | torch `randn` | crate RNG; not bit-reproducible |
| tune | block-scope `tune_fact` on the block output, 128 samples × 8 epochs | layer scope: analytic `L = tr(EHEᵀ)` on the cached Gram, `h_scale = 2048`; same optimiser, rates, schedule and step count |
| KD | global KD after the blocks | not ported |

## Plan — NanoQuant subsumption

### Measured (l23 `down`, Qwen3.5-0.8B, layer scope, grade measured)

Command: `APEIRON_NQ_KIND=down APEIRON_NQ_LAYERS=23 … cargo test --release --all-features --locked
--test learn_nanoquant nq_fit -- --ignored --nocapture`; machine i9-14900KF, 32 threads.

| arm | b/w | relH | × Q4_K_M (0.000151) |
|---|---:|---:|---:|
| c6x29 | 1.175 | 0.019483 | 129 |
| c6x29-sel | 1.175 | 0.012146 | 80 |
| NQ ADMM, linear, 400 it | 1.1451 | 0.02446 | 162 |
| NQ ADMM, cubic, 400 it | 1.1451 | 0.02493 | 165 |
| NQ ADMM, cubic, 400 it | 0.9844 | 0.03010 | 199 |
| NQ ADMM, cubic, 150 it | 0.9844 | 0.0322 | 213 |
| NQ ADMM 400 it + layer tune, 1024 steps (W2) | 1.1451 | 0.01727 | 114 |
| NQ ADMM 400 it + sketched tune, `s = 256`, 1024 steps | 1.1451 | 0.01678 | 111 |
| NQ ADMM 400 it + sketched tune, `s = 128`, 1024 steps | 1.1451 | 0.01719 | 114 |

### Held-out KL (W3, same scope: only `blk.23.ffn_down` replaced; chunks 8–9, 510 tokens)

Command: `APEIRON_GATE_ARM=<arm> APEIRON_GATE_SKIP=8 APEIRON_GATE_CHUNKS=2 cargo test --release --all-features
--locked --test learn_gate_code -- --ignored --exact gate_code_kl --nocapture`; logs
`/home/socol/oc-tmp/escalation/lead-1023/nq/kl-<arm>-89.log`.

| arm | b/w | KL (sd) | × Q4_K_M | top-1 |
|---|---:|---:|---:|---:|
| Q4_K_M (`down-q4kmL23`) | 4.75 | 0.000952 (0.000037) | 1 | 0.980 |
| c6x29-sel (`down-c6x29selL23`) | 1.175 | 0.050752 (0.000676) | 53.3 | 0.882 |
| W2 (`down-nqw2b`) | 1.1451 | 0.075035 (0.004615) | 78.8 | 0.855 |
| NQ ADMM only (`down-nq-b1175lin`) | 1.1451 | 0.077613 (0.003045) | 81.5 | 0.855 |

W2's 29% relH gain over ADMM is a 3% KL gain, inside W2's sd. c6x29-sel leads in both relH and KL. System
effect: this one W2 tensor costs 5.7× the KL of all seven `o` tensors at Q4_K_M (0.0131). Lean:
`Learn.NanoQuant.kl_l23_down`.

### Sketched tune (`tune_sketch`, `H ≈ Q·H_s·Qᵀ`, rank `s` subspace iteration, 3 passes)

| s | relH | wall (1024 steps) | MAC/step | × full-step MACs |
|---:|---:|---:|---:|---:|
| full (W2) | 0.01727 | 403.9 s | 23.02 G | 1 |
| 256 | 0.01678 | 110.1 s | 3.24 G | 0.141 |
| 128 | 0.01719 | 105.0 s | 1.60 G | 0.070 |

The rank-256 sketch holds 43.5% of W2's error energy (92.0% of Q4_K_M's, `nq_sketch_share`); the
sketched tune still lands below the full tune. Halving `s` halves the MACs and saves 5% wall: a per-step
cost independent of `s` dominates (profile: `nq-sk-floor-1`). Lean: `sketch_share_tail`,
`sketch_tune_l23_down`, `sketch_rank_floor`. Logs `tune-sk{256,128}.log`, `sketch.log`.
Control (`APEIRON-QUALITY-118-W1-T01`): `- self.wq` → `+ self.wq` in `SkFrame::step` turns
`full_width_sketch_tune_matches_the_full_tune` red (exit 101); the unmutated copy exits 0
(`/home/socol/oc-tmp/escalation/lead-1023/nq/mut.log`).

ADMM alone is NanoQuant's initialisation; the paper's quality comes after `tune_fact` and KD. The
ADMM-only arm is 1.26× c6x29 and 2.01× c6x29-sel in relH. The W2 tune (reference rates `1e-5`,
cosine, 1024 steps; 66 665 of 4 128 768 signs flipped; best at the last step, so still falling)
is 0.886× c6x29 and 1.42× c6x29-sel. Logs: `/home/socol/oc-tmp/escalation/lead-1023/nq/w2a-micro.log`,
`w2b-1024.log`; board `APEIRON-QUALITY-118-W1-T01`.

Small-matrix control (96×64, `r = 32`, 60 steps): scales alone converge to 0.987× the ADMM energy;
latent rates `1e-3` / `1e-2` flip 135 / 507 signs and raise it to 1.02× / 1.56×. The STE step is a
linearisation at the continuous latent, so a rate large against the latent magnitude (`|ul| ≈ 0.05`)
flips signs the discrete loss does not reward.

### Cost equation (measured)

ADMM, blk.23 `ffn_down` 1024×3584, r = 768, 3 limbs, 20 iterations, seeds 0–2: 3.3, 3.3, 3.2 s per fit
(was 0.34 s per iteration), `admm_loss` 0.16482, 0.16522, 0.16484. Per iteration ≈ 165 ms: `W·Z` 41.5, Gram
25.4, right-hand side 19, `H⁻¹` 34 → 29, `rank1` 29.5 ms; the first three run at the kernel rate (≈ 815 G
limb-MAC/s). `H⁻¹` is [`spd_inverse`]: block Schur-complement recursion, `2n³/3` in kernel products, base
order 32 (order 768: 12.2 ms against 17.4 at 192; residual at order 300 7.7e-6 against 3.4e-5 unrecursed).
2 limbs: fit 2.7–2.8 s, `admm_loss` 0.168 (+1.9%, seeds 0–4), so 3 limbs stay (§0 K3). 400 iterations
(the reference's `outer_iters`), seeds 2 and 0: fit 61.3 s per tensor, `admm_loss` 0.10344 / 0.10341
against 0.16484 at 20; after the same 128-step tune (s = 256) relH 0.02478 / 0.02471 against 0.03321
(−25%), best 0.01377 / 0.01370 against 0.01721, 16 358 / 15 569 flips against 50 042; `i400.log`; verdict
`Learn.NanoQuant.admm_iters_l23_down`. Iteration curve, seed 0, same 128-step tune: 50 / 100 / 200 / 400 iterations give relH 0.02787 / 0.02622 / 0.02521 / 0.02471 at fit 7.9 / 15.5 / 30.9 / 61.3 s (`icurve.log`). 1024 further tune steps (34.4 s) from the 100-iteration factor reach 0.02157 (× Q4_K_M 142.9) at 49.9 s total; from the 400-iteration factor 0.02039 (× 135.1) at 95.4 s (`budget.log`). Past 100 iterations a second of tune cuts relH 4.1× more than a second of ADMM; verdict `Learn.NanoQuant.admm_budget_l23_down`, control `mut-budget.log`. The fit runs 100 iterations and spends the wall on tune steps. Tune: 0.39 s per step at r = 896 (403.9 s for 1024); sketched,
110 s for 1024 at 1024×3584. Sketched at r = 768, s = 256, 128 steps from the `b32` factor (sketch
included): 10.0/9.9/9.8 s → 7.1/7.1/7.1 s (1.39×) with packed Adam moments, parallel sign and real maps,
and transposes on words instead of reals; the trace is bit-identical (relH 0.04919 → 0.03321, best
0.01721 at step 128, 50 042 flips; `tsb.log`, `tsw.log`). `∂s_pre[j] = ⟨Q_j, (VᵀA)_j⟩ = Σ_i sign(V_ij)·(A·Qᵀ)_ij`
is one exact `i128` signed column sum over the `A·Qᵀ` words; it replaces the `sign(V)ᵀ`
transpose (1.60 ms per step) and the `VᵀA·Q` product (6.45 ms) and cuts the row dots 5.46 → 1.31 ms:
6.71/6.60/6.57 s → 5.25/5.23/5.27 s (1.26×), relH 0.04919 → 0.03321, final 0.03320, best 0.01722, 50 103
flips (+0.12%); `tss.log`, `colsum-clean.log`, per-phase laps `phase.log`, `colsum.log`; control: the sign
negated in the column sum turns `full_width_sketch_tune_matches_the_full_tune` red (`mut-colsum.log`);
verdict `Learn.NanoQuant.tune_colsum_l23_down`. `A·Qᵀ` against `Q` as a prepared kernel operand (one plane
per tune; the kernel returns `(A·Qᵀ)ᵀ`, 10.7 → 7.1 ms) and `∂V` with the column sum in one pass blocked by eight
rows ([`col_grads`], 4.5 → 2.97 ms): 5.25/5.23/5.27 s → 4.79/4.80/4.77 s (1.10×), trace identical;
`colgrads-clean.log`, laps `aqswap.log`, `colgrads.log`, `colblock.log`; control: the sign negated in
`col_grads` turns `full_width_sketch_tune_matches_the_full_tune` red (`mut-colgrads.log`); verdict
`Learn.NanoQuant.tune_colgrads_l23_down`. Remaining per step (≈ 37 ms): `P` 8.2, `A·Qᵀ` 7.1, `col_grads` 3.0,
`∂U` 2.4, `A` 2.45, Adam 2.4, `sign(V)·diag(sq)` 2.15, `U·P` 1.8. Per tensor at 400 iterations the ADMM
(61.3 s) is the dominant term; 128 tune steps are 4.8 s; at 100 iterations ADMM is 15.5 s against 34.4 s for 1024 steps. Logs `/home/socol/oc-tmp/escalation/lead-1023/nq/`
`b32c.log`, `seed-mixed.log`, `invsmall.log`, `invres.log`, control `invmut.log`; board `APEIRON-QUALITY-118-W1-T01`.

Device limb product ([`src/gpu/limb.rs`], feature `gpu`, routed inside `engine::kern::limb` by `product()` and
`Prepared::syrk`; `APEIRON_LIMB_DEVICE=0` disables it): int8 M16N16K32 cooperative matrices, one 64-bit lo/hi
accumulator per pair group, host bound check (out of range → CPU), host finish `round_shift_32` per group
summed in `i128`, pooled across rows. Routed when `n ≥ 256` and `m·p·n ≥ 2²⁶`: at m·p = 1024·768 the routed
wall is ≈ 0.89 ms + 2.1e-9 ms/MAC against 1.0e-8 ms/MAC on the CPU, break-even n ≈ 143 (n = 128 ties, 1.10
against 1.07 ms, so it stays on the CPU). Wall per call, mean of 3 (CPU → routed): `W·Z` 1024×768×3584 31.73
→ 6.88 ms (4.6×), Gram 768×768×3584 20.83 → 5.37 ms (3.9×), solve 1024×768×768 4.46 → 1.97 ms (2.3×), tune
768×128×3584 3.55 → 2.40 ms (1.5×); 0 differing entries at all eight test shapes (`gfp.log`). Fit, 100
iterations, 128-step tune (s = 128), three interleaved runs per arm: ADMM 15.43 → 12.47 s (0.808×), tune
3.17 → 2.74 s (0.864×); `admm_loss` 0.11172, relH 0.02631, 19 161 flips identical in all six runs
(`dfit2.log`). Control: the shift dropped from `limb_gemm.comp:93` makes 786 432 of 786 432 entries differ
(`mut-limb.log`). Verdicts `Learn.NanoQuant.limb_device_route`, `Learn.NanoQuant.limb_device_fit_l23_down`.
Remaining per iteration ≈ 125 ms: `rank1` (CPU prepares and one-column products), `spd_inverse` (order-384
sub-products under `2²⁶`), the per-call plane build and upload of `W`, and the f64 maps.

### Same-model comparison (Qwen3-0.6B-Base, wikitext2 test, 146 × 2048 tokens, grade measured)

Qwen3.5 is outside the reference's `get_decoder_layers` (`utils/utils.py:107`, as is `qwen2`), so the
shared model is `Qwen/Qwen3-0.6B-Base@da87bfb6`. Instrument: the reference's `evaluate_ppl` protocol
plus KL(FP16‖arm) and top-1 agreement over the same 298 862 tokens. Receipts `/home/socol/oc-tmp/nq-ref/`
`base.log`, `run3.log`; board `APEIRON-QUALITY-118-W1-T01`.

| arm | linear b/w | ppl | × FP16 ppl | KL | top-1 |
|---|---:|---:|---:|---:|---:|
| FP16 | 16 | 12.660 | 1.000 | 0 | 1 |
| Q4_K_M (llama.cpp, whole file 5.2446 b/w, 372.65 MiB) | 4.7750 | 13.750 | 1.086 | 0.0860 | 0.852 |
| NanoQuant reference, 1.18 b/w, after block 0 only | 1.18 (1 of 28 blocks) | 14.997 | 1.185 | — | — |
| same, after blocks 0–1 | 1.18 (2 of 28) | 13.819 | 1.092 | — | — |
| same, after blocks 0–2 | 1.18 (3 of 28) | 16.995 | 1.342 | — | — |

Owner metric (perplexity and task accuracy), same harness and run (`eval_arm.py`, NanoQuant's `evaluate_model`; receipts
`arm-bf16.json`, `arm-q4km.json`, `run4.log`; `docs/evidence/perf-predictions.md` P131):

| arm | ppl | × Q4_K_M | mean 0-shot acc | × Q4_K_M |
|---|---:|---:|---:|---:|
| bf16 | 12.6705 | 0.921 | 0.5635 | 1.014 |
| Q4_K_M | 13.7633 | 1.000 | 0.5556 | 1.000 |
| NanoQuant reference, 1.18 b/w, as shipped (run4) | 24.3317 | 1.768 | 0.4235 | 0.762 |
| bar (ppl ≤ 1.10×, accuracy ≥ 0.90×) | 15.14 | 1.10 | 0.500 | 0.90 |

The reference as shipped misses both bars on this model. The port has no row until a Qwen3-0.6B student exists.

The tied embedding (151936 × 1024) stays FP16 in the reference; at 0.6B it is 26% of the parameters, so
the reference's whole-model rate is ≈ 5.05 b/w, while at 27B the embedding share is small.

Reference wall: 113.6 s per block (116, 138, 100, 100 s on one RTX 5080, including the per-block ppl
eval); 28 blocks = 53 min plus model KD, above the 600 s ceiling. ADMM is 0.56 s per tensor on the GPU;
the tune epochs dominate. Our path at 1024×3584 measures 61.3 s ADMM (400 iterations, CPU) plus 4.8 s per
128 sketched tune steps (≈ 38 s for 1024) per tensor (above): the dominant term is our ADMM per-iteration
cost (≈ 154 ms on the CPU, ≈ 125 ms with the device limb product; 89× the reference's 1.4 ms GPU
iteration), not the algorithm.

### Wave map

| wave | task | files | gate |
|---|---|---|---|
| W1 | ADMM port | `src/learn/nanoquant.rs`, `src/learn/mod.rs`, `tests/learn_nanoquant.rs` | `cargo test --release --all-features --locked --test learn_nanoquant` |
| W2 | layer-scope STE tune: export latents; Adam on `L = tr((Ŵ−W)H(Ŵ−W)ᵀ)` through `sign` | `src/learn/nanoquant.rs`, `tests/learn_nanoquant.rs` | same; relH below the W1 row at 1.1451 b/w |
| W3 | KL arm vs same-scope Q4_K_M (chunks 8–9) | `tests/learn_gate_code.rs` | KL log |
| W4 | Lean/Aeneas: budget, factored forward, SVID identity, ridge minimiser | `proof/corpus/Learn/Quant/NanoQuant.lean`, `proof/corpus/AxiomAudit.lean` | `cd proof/corpus && lake build` |
| W5 | `o_norm` importance, block-scope tune | `src/learn/nanoquant.rs` | relH, then KL |

W2 step cost: `o·n² + 3·r·o·n` MACs ≈ 13.2 + 9.9 GMAC at `down`, 0.39 s measured.

### W6: device-resident ADMM iteration

Intent: the iteration's operands stay on the device between products. Per iteration at blk.23 `down`
(o = 3584, n = 1024, r = 768) the limb products are ≈ 11.6 G MAC × 6 kept pairs ≈ 70 G int8 MAC, ≈ 1.4 ms
at 50 effective int8 TOPS. Measured with the device on (110 ms per iteration): `rank1` 27.3 (prepare 11.3,
power 12.2, sign 3.8), inverse 25.1, `wz` 23.9, Gram 14.7, `f` 10.8, `zt` 4.1, `rt` 2.9, `hw` 1.2 ms. The
host plane build, upload and readback carry most of that. Target: ≤ 10 ms per iteration.

Spec: every task is bit-identical to the CPU `factorize`, which stays the oracle. A device range flag
(block exponent e > 62 in the plane build, or a pair-group bound ≥ 2⁶³) reruns that iteration on the CPU.
The elementwise f64 maps (`from_grid`, `to_grid`, the `d_i` divide, `sqrt`, `rho·(z − du)`) run in binary64
on the device, written `precise` with unfused multiply and add, and rounded half away from zero with
`trunc` (GLSL `round` is not used). Receipt: `gpu::limb::tests::f64_maps_match_cpu`, 2²⁰ inputs at
magnitudes 2⁻³²..2³¹ with every 7th on a half grid point, 0 differing results for divide, `sqrt`, multiply,
unfused multiply-add, the grid round and i64→f64. Control: `fma(a, b, c)` in place of the `precise`
unfused form gives 120 741 differing results.

| task | files | gate | control |
|---|---|---|---|
| T1 plane build on device: absmax → `block_exp` → `atomicMax(e[b])`, then the digits by `split_digits` | `src/gpu/spirv/limb_plane.comp`, `src/gpu/limb.rs` | digits and `e` equal `Plane::build` at the fit shapes | first-digit rounding bias dropped → red |
| T2 placement on device: `T`, `tmin`, shifts and the bound check from `e_a`, `e_b` | `src/gpu/spirv/limb_place.comp`, `src/gpu/limb.rs` | meta equal to the host `LimbGemm::run` meta | `MIN_PAIR_EXP` off by one → red |
| T3 finish on device: `round_shift_32` per group to an i64 plane, with a range flag | `src/gpu/spirv/limb_finish.comp`, `src/gpu/limb.rs` | i64 plane equal to `gemm_q32` | the rounding constant dropped → red |
| T4 resident `x_update` (inverse on the host) | `src/gpu/admm.rs`, `src/learn/nanoquant.rs` | `x_update` output bit-identical | `stab` term dropped → red |
| T5 resident `rank1` (power steps, `fro2`, sign product) | `src/gpu/admm.rs`, `src/learn/nanoquant.rs` | `rank1` bit-identical | sign product negated → red |
| T6 resident inverse (Schur levels on device; base-32 `inv_factor_upper` stays i128 on the host) | `src/gpu/admm.rs`, `src/learn/solve/inv.rs` | `spd_inverse` bit-identical | Schur symmetrisation dropped → red |

T1, T2, T3 and T4 pass their gates and controls; receipts are P107, P108, P109 and P110 in
`docs/evidence/perf-predictions.md`. The T4 frame is `src/gpu/limb/admm.rs`, not `src/gpu/admm.rs`:
the limb device code moved to `src/gpu/limb/` under the 1500-line cap, and T5 and T6 extend that file.
T4 moves ADMM from 11.90 to 9.57 s per 100 iterations (0.804×) with every fit number unchanged. The T2 range fixture uses 56-bit words, because 51-bit words reach
only `T = 90`. The fit shapes place every pair group at `Tmin ≥ 32`, so the T3 rounding branch is
reached only by the synthetic planes; its control is seen there.

T5 passes its gate and control (P111): the resident `rank1` is `src/gpu/limb/rank1.rs` with
`src/gpu/spirv/rank1_map.comp`, and ADMM falls from 9.57 to 6.80 s per 100 iterations (0.711×; 0.596×
the CPU path) with every fit number unchanged. With the device `rank1` alone the fit got slower: the
pool (`src/engine/chart/pool.rs`) parks its workers after 5 ms idle, each device wait parked them,
and the host inverse ran cold (65.0 against 24.0 ms per iteration). `pool::Hot`, held by `factorize`
over the ADMM loop, keeps them spinning; T4's 9.57 s also carried that cold start. The host inverse
(T6) is the largest remaining host stage. Board row `APEIRON-QUALITY-118-W1-T01`.

T6 is re-scoped by its cost equation (P112). Measured on the RTX 5080 (`device_unit_costs`), an empty
`submit_wait` costs 19.9 µs and two plane builds plus one product cost 457.4, 86.9, 37.5, 30.7 and 29.2 µs
at half order 384, 192, 96, 48 and 24. The order-768 inverse has 124 products over 31 internal nodes,
so the device-resident inverse projects 8.33 ms (`Perf.LimbDevice.schurDevice_fit`) against 10.21 ms for
the host inverse with its block copies and assembly pooled: 0.38 s per 100 iterations, so the device levels
are not built. The host change is bit-identical to the serial body, and its control (symmetrisation of `S`
dropped) turns red. ADMM falls from 6.73 to 6.43 s per 100 iterations (0.955×) with every fit number
unchanged. The next host cost is the inverse's small products: about 3.7 ms at half order 24–96, overhead-bound in
the limb kernel's plane builds and pool dispatch.

T7 overlaps the two `x`-updates (P113). Both read only the previous `Z` and `U`, so `factorize` runs them
on two threads; one side's 10.21 ms host inverse proceeds while the other holds the device mutex. Measured
split per iteration at `629767bc`: `x_A` 22.20, `x_B` 18.41, `rank1` A 10.09, B 4.26, `t` 2.84, duals
2.45 ms. The pair holds 20.42 ms of host inverse and 20.19 ms of device work, so overlap hides at most
20.19 ms (`Perf.LimbDevice.xPairOverlap_fit`). ADMM falls from 6.40 to 5.63 s per 100 iterations (0.880×),
7.67 ms per iteration; the saved factor is byte-identical, and the control (`B` reading `X_A` in place of
`Z_A`) differs in 28 179 933 of 31 887 392 bytes. The pair now takes 12.52 ms more than its 20.42 ms
host total (grade `model`).

T8 runs each side's whole step on its own thread (P114): `X`, `T = X + U`, `rank1` and the dual read
only that side's `X` and `U` and the previous `Z`s, and both `rank1` starts are drawn first in the
serial order. Split per call at `4be01388`: `x_A` 31.76 and `x_B` 26.54 ms (their two inverses contend
in the pool, 14.72 and 15.22 ms against 10.21 alone), `rank1` A 9.99 ms (8.05 on the device), B
4.28 ms. `A`'s chain stays the longer one, so the iteration saves at most `B`'s moved work, 5.46 ms
(`Perf.LimbDevice.sideStep_fit`, grade `model`). ADMM falls from 5.67 to 5.20 s per 100 iterations
(0.917×), 4.67 ms per iteration; the saved factor is byte-identical, and the control (starts drawn `B`
first) differs in 28 079 495 of 31 887 392 bytes. The host stages and reads back about 9.8 ms of
iterate per iteration and holds the two contending inverses; the next costs are those, and the
inverses' contention.

T9 keeps the iterate `[Z_x, Z, U, X]` on the device between outer iterations (P115,
`src/gpu/limb/resident.rs`). `cb1` uploads only `[rho, reg]`; `cb2` and `rank1` read nothing back;
`rank1` forms its target `X + U` on the device (`rank1_map.comp` mode 6); one commit buffer per side
runs the dual `U + (X − Z)` (mode 7) and copies the new `Z` into the side's own iterate and the peer's
`Z_x`. A refusal reads `Z` and `U` back and resumes the host loop at that iteration with the same
starts; `APEIRON_ADMM_RESIDENT=0` keeps the host-staged path. Removed from `A`'s chain per iteration:
9.86 ms of device transfer, 6.23 ms of host staging and readback, and `A`'s 4.11 ms share of `T` and
the duals, 20.21 ms (`Perf.LimbDevice.residentIterate_fit`, grade `model`). ADMM falls from 5.37 to
3.43 s per 100 iterations against the interleaved host-staged arm (0.639×), 53.7 → 34.3 ms per
iteration; the saved factor is byte-identical. Controls: the dual's sign flipped (`U − (X − Z)`)
makes the fit refuse (`factorize: Kernel`); `U` dropped from the target (mode 6 `T = X`) differs in
28 752 128 of 31 887 392 factor bytes (relH 4137.7). What remains per iteration is the two `x`-updates' host inverses
(about 15 ms each under contention), the `H⁻¹` uploads and the `rank1` power steps.

T10 splits the digit tail block zero-padded to a full block (P116, `DigitsTask::body`); the
order-768 inverse falls from 10.06 to 9.33 ms and ADMM from 3.50 to 3.33 s, factor byte-identical
(`Perf.LimbDevice.digitsPad_fit`).

T11 moves `spd_inverse` onto the device above order 96 (P117, `src/gpu/limb/inverse.rs`,
`inv_map.comp`). Each node of order above the cut runs its block extracts, `S = C − BᵀX`, the
assembly and four limb products on the device; nodes of order 96 are read back, inverted by the host,
and staged into the parent, one submit per leaf. A flag or a leaf refusal returns to the host
`spd_inverse` on the same `H`, so the result is the host one or its refusal; `cb2[2 + v]` omits the
`H⁻¹` upload when the device produced it. ADMM per 100 iterations by cut (3 interleaved runs): host
3.3, 192 2.6, 96 2.4, 48 2.47, 32 2.9 s; 96 is the default (`DEV_INV_CUT`). The saved factor is
byte-identical in all 18 runs. Controls: the `S` base offset set to 0 makes the device refuse and the
host fall back; `Y` read from `X` in the assembly differs in 6400 words. 9.0 ms per iteration saved
leaves 8.46 ms of `A`'s former 17.46 ms inverse on its chain (`Perf.LimbDevice.deviceInverse_fit`,
grade `model`). What remains: the leaf round trips, the `hw` copy to the host on every step (needed
only on refusal), `B`'s `cb1`/`cb2` (9.2 ms) and the `rank1` power steps (`A` 4.7, `B` 2.8 ms).

A second queue per side (T12, throwaway, not installed) is refuted: ADMM 2.4 / 2.5 s against one queue,
factor unchanged. Run one side after the other, each side's fence wait is its own device time, `A`
8.25 and `B` 6.35 ms per iteration, 14.6 ms in all: the device runs the two sides' work one after the
other in either case. Timestamps inside `rank1` put `A`'s device time there at 7.3 ms per
iteration, of which the 1×1 self-dot (`u.norm2`, 1.96 ms) and the matvec (`m.prod`, 2.11 ms) are
latency-bound: one workgroup per 64×64 tile walks all 112 blocks.

T13 splits each tile's block loop of `limb_gemm.comp` across `splits` workgroups (P118,
`Product::splits`, `SPLIT_WG` = 192): split `c` writes its own lo/hi planes and `limb_finish.comp`
sums a group's splits in 64 bits before it rounds by `Tmin_g`. Addition modulo 2⁶⁴ is associative,
so the group word is the one-workgroup word for every split count. ADMM falls from 2.43 to 1.9 s per 100
iterations (3 interleaved runs; `SPLIT_WG` 768 also 1.9 s), factor byte-identical in all 9.
Control: the finish reads only the first split, and `admm_rank1_matches_cpu` and
`admm_x_update_matches_cpu` go red (`Perf.LimbDevice.splitK_fit`).

T14 runs the order-96 host leaves of the device inverse inline on their side thread
(`pool::inline`, P119): no dispatch lock, no pin, no worker. A pooled leaf took 0.37–0.39 ms
against 0.327 ms on one thread, and the two sides' leaves queued on the pool's one dispatch lock.
ADMM falls from 1.855 to 1.784 s per 100 iterations (3 interleaved runs), factor byte-identical in
all 6. With inline leaves side `A` still spends 4.29 ms per iteration in its eight leaves against
2.62 ms single-thread (`p15.log`). `spd_inverse_inline_matches_pooled` holds 0 differing words at
orders 96 and 301; control: the inline loop one task short, and the inline inverse refuses, test red
(`Perf.LimbDevice.inlineLeaf_fit`). Side `B`'s thread inherits `A`'s CPU 0 pin, so both sides' leaves
share one core; `B` on CPU 3 gives loop 11.50 against 12.92 ms (`p16.log`, single runs).

T15 pins the side-`B` thread to the hyperthread sibling of the first performance worker's core
(`pool::pin_beside_worker`, P120; CPU 3 on this box). The worker's spin keeps that core out of its
idle states, and CPU 0 stays side `A`'s. ADMM falls from 1.788 to 1.608 s per 100 iterations
(3 interleaved runs), factor byte-identical in all 6. Placement profile (`p16.log`, `p17.log`, single
runs): inherited CPU 0 12.92 ms per iteration, CPU 1 12.60, CPU 3 11.50 and 11.31, unpinned 11.90;
with the pool's spin off (`Hot` dropped) CPU 2 gives 15.20 ms. `beside_worker_takes_the_first_workers_sibling`
checks the sibling choice on the host's layout; control: the worker index 1 → 0 chooses CPU 1, test
red (`Perf.LimbDevice.sideBPin_fit`).

T16 draws the initial `A` and `Bᵀ` normals on the pool (`Rng::normals`, P121). SplitMix64's state
after `w` words is `s₀ + w·γ`, so draw `i` is computed alone from `s₀ + 2i·γ`; `Bᵀ` is filled in its
transposed slot order, which removes the `transpose`. The fit falls from 1.614 to 1.561 s (3
interleaved runs per arm), factor byte-identical in all 6. `normals_match_sequential` holds every bit
of both slot orders and the state after; control: the state advanced by `len` words instead of
`2·len`, test red (`Perf.LimbDevice.rngPool_fit`).

T17 runs the limb product segment-outer (`src/gpu/spirv/limb_gemm.comp`, P122). Each segment walks
its blocks innermost; the int32 fragment accumulates across blocks of one shift and enters the
64-bit accumulator only when the shift changes or the segment ends, so the per-block 64-bit
read-modify-write of every fragment element is gone. A segment block adds at most 2^20 and at most
512 blocks (`MAX_INNER / 32`) run in one fragment, under 2^29. Product time at the frame shapes falls
14–34% (`A` `wz` 3584×768×1024 501 → 394 µs, `B` `gram` 768×768×3584 507 → 336 µs; 56–86 int8 TOPS
from 44–67). The fit falls from 1.576 to 1.532 s (8 interleaved pairs, 7 faster by 43–122 ms, one
+138 ms outlier), factor byte-identical in all 16. Control: the final flush dropped (`blk < blk1`),
four device tests red (`Perf.LimbDevice.segRun_fit`).

T18 (resident ADMM tail) is a loss and is not installed: on 1.70–1.74 s against off 1.49–1.51 s, five
interleaved pairs, factor sha `ce1a2b76f09c622b` in all ten (`/home/socol/oc-tmp/escalation/lead-1023/nq/t18-ab.log`).
Attribution (`attr18.log`): the loop is 1.09 s of a 1.49 s fit; its products run at 58–86 int8 TOPS.

The held-out instrument (`src/learn/fold/stats/heldout.rs`) scores a chunk's student columns with one
`lm_head_cols` call (APEIRON-QUALITY-118-W1-T01); `learn_fold_heldout` 11 passed, 3 ignored.

Measurement after each task: `nq_fit`, three interleaved runs per arm, ADMM seconds per 100 iterations
(baseline 11.9–12.5 s), `admm_loss`, relH and flips unchanged.

### W7: the reference block loop on the device

The reference's quality comes from its sequential block loop (`core/compress_model.py:83-101`): per
block, for each of q, v, o, k, gate, up, down, `tune_nonfact` over the block's dense linears, ADMM, then
`tune_fact` over the factorised ones, both streams advancing through the block. The device fit
(`src/gpu/nn/train/block.rs` `BlockTune`) tapes o, gate, up and down only. Board
`APEIRON-QUALITY-118-W1-T01`.

| task | change | gate on the live tree |
|---|---|---|
| T01 | `Mixer::Softmax` carries `gated`; the ungated face (Qwen3: no gate half, `attn_map` mode bit 2, `attn_map.comp:41`) records and runs | `layer_fwd_matches_cpu`: Qwen3.5 l0 rel 5.688e-3 / 2.814e-2; Qwen3-0.6B l0 1.100e-2 / 1.556e-2, l3 2.475e-2 / 6.864e-4 (bound 3e-2). Control: mode bit dropped → l0 9.723e-1, red |
| T02 | the per-slot builders (factor allocation, materialise, factor adjoint, Adam, upload) move to `src/gpu/nn/train/slot.rs`, the tapes unchanged | `gpu::nn::train` 15 passed; loss device 347.652154 / host 347.652157, Adam 347.6522 → 307.0150, identical before and after. Control: `ds_post` reads `w` for `m` → red |
| T03 | `BlockTune`'s backward ends `da = dh·Wo` (`linear_bwd_input_dispatch` after the `dwt(0)` weight gradient), the gradient at the attention output the q/k/v slots read; accessor `BlockTune::da` | `block_tune_matches_host`: `da` worst relative error 1.79e-7 against f64 central differences (bound 1e-4); loss and scale gradient unchanged. Control: `wt(1)` in place of `wt(0)` → 3.29e0, red |

### Negative controls

Dropping `s_pre` in `NqFactor::weight` in a throwaway copy turns
`export_matches_its_dense_form_and_loss_falls` red; the restored tree passes.

W2: negating `∂s_post` before its Adam step in a throwaway copy turns
`latents_carry_the_signs_and_tune_lowers_the_h_energy` red at its descent assertion; the live tree
passes.

### Stop and rollback conditions

A W2 tune that does not beat the W1 relH at equal b/w is recorded as a failure in the quantisation
model and stops the layer-scope route in favour of block scope.
