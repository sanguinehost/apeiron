# Original NanoQuant (`SamsungLabs/NanoQuant` @ `a9e0a43`) vs `apeiron-nq`: Parity & Single-Path Consolidation

Criteria: **Q1**, **Q2**, **Q3**, **T1**, **T2**, **V1**.
Grades: `model` / source audit for the code comparison, `measured` for the Phase 0b export A/B and the reference ADMM band (§3.1, §3.4: `Qwen3-8B` layer 0 and `admm_nq.py` on the box), `tested` for the unit gates. Includes DeepSeek worker review corrections and Gemini 4 Argon architectural consolidation review.

## 1. Current Repository State & Provenance

- **Original reference clone:** `/home/lucas/Workspace/NanoQuant` (`https://github.com/SamsungLabs/NanoQuant`, `HEAD` at `a9e0a43`, Apache-2.0) — the exact commit cited throughout `apeiron-nq`. Ships full PyTorch/CUDA PTQ source and custom CUDA kernels; ships no pre-quantized weights.
- **Provenance of the benchmarked 8B packs:** The 1.124 b/w and 1.1788 b/w packs scored in `GOAL.md` (v1 five-window screen: `1.462553×` official Q4_K_M PPL, `1.525440×` Q8_0 PPL, v0 0-shot `0.7534×` Q4_K_M) come from converting the third-party `arelath/Qwen3-8B-nanoquant-GGUF` via `tools/nq_convert.py`. That upstream file carries fused `attn_qkv`, `nq_scale_mid`, and sparse salient columns (`nq_salient_idx` / `nq_salient_weight`). None of those scored packs came from `SamsungLabs/NanoQuant` directly or from `apeiron-nq`'s `examples/nq_fit.rs`.
- **Four fragmented pack producers currently exist:**
  1. Third-party `arelath` GGUFs via `tools/nq_convert.py`.
  2. Standalone CLI `examples/nq_fit.rs` + `src/learn/nanoquant.rs` (ADMM-only, independent per linear, no backward Fisher `o_norm`, no clipping/shrinkage, no block tuning, no model KD).
  3. Vulkan GPU training tests `tests/learn_block_tune.rs` + `tests/learn_kd.rs` (`src/gpu/nn/train/{front,block,chain,head,kd}.rs`): port OG's `compress_block_recon` and `compress_model_recon`, but are `#[ignore]` 0.6B-only tests; `tests/learn_block_tune.rs:29-30`'s `0.3–3.9%` parity claim has no receipt in `bench/`, and `:1300-1301,1477-1478` print `(run4 q 0.2589 / v 0.3043)` as literal hardcoded labels — **asserted, not measured**.
  4. Untouched original `SamsungLabs/NanoQuant` (`/home/lucas/Workspace/NanoQuant`, Apache-2.0).

---

## 2. Stage-by-Stage Code Comparison (`SamsungLabs/NanoQuant` vs `apeiron-nq`)

### 2.1 Calibration & Importance Statistics

| aspect | `SamsungLabs/NanoQuant` (`core/importance.py`) | `apeiron-nq` (`tests/learn_calib.rs`, `examples/nq_fit.rs`) | consequence |
|---|---|---|---|
| Input statistic `i_norm` | Forward hook on `nn.Linear`, `x_clipped.square().mean(dim=0)` over 128 × 2048 tokens (`wikitext2` train) | Tapped `Q8Act` prefill Gram diagonal (`g{l}.{tag}.{gram}.diag.i64`); falls back to weight column RMS when `--diag-dir` is omitted | Weight column-RMS fallback collapses quality (185k–1.2M PPL on 0.6B layer 0) |
| Output statistic `o_norm` | Full backward hook (`linear_cross_entropy`), squared output gradients scaled by `GRAD_SCALE_FACTOR = 1e6` | Hardcoded `vec![1.0; out]` in `examples/nq_fit.rs:367` | ADMM weights all output rows equally; ignores downstream loss sensitivity |
| Outlier clipping | 99.9th-percentile L2 row-norm clipping (`PERCENTILE = 0.999`, `online` cumulative or `two_phase` fixed) | None (`--salient-frac` zeroes top-`k` columns in `nq_fit.rs:347-366` while `packed_linear()` writes empty `salient`/`salient_idx`) | Extreme channels dominate `W_norm`, or get silently dropped on export |
| Covariance shrinkage | `get_shrunk_stats`: `H_new = 0.6·H + 0.4·mean(H)` (`calib_shrinkage = 0.4`) | None; only dead-coordinate floor `FLOOR_REL = 1e-6 * mean(sqrt(diag))` (`src/learn/nanoquant.rs:55`) | Unshrunk diagonals increase ADMM condition spread |

### 2.2 Rank Allocation & Scale Count

- **Formula:** `utils/utils.py:128-182` (`calculate_ranks`) and `src/learn/nanoquant.rs:98-107` (`rank_for_bits`) use the same 2-scale formula `floor((a·b·bits / (a + b) - 16) / 32) * 32`, minimum `32`.
- **Differences:** OG caps rank at `min(in_features, out_features)` (`_validate_rank`, `utils.py:161-167`), while `rank_for_bits` does not; OG `admm_nq` stores **2 scales** (`scale_pre`, `scale_post`, **no `scale_mid`**; 3 scales are `admm_dbf` only), whereas `APNQPK01` v2 always charges `16·r` bits for `s_mid` even when it is `1`: ~1.08 MB over the 252 fitted sections, i.e. ~0.010 b/w of the 1.000 pack spent on a plane the reference does not have.

### 2.3 ADMM, Block Reconstruction & Model KD

| stage | `SamsungLabs/NanoQuant` (`a9e0a43`) | `apeiron-nq` |
|---|---|---|
| **ADMM** | `core/admm_nq.py`: fp32/TF32 CUDA, `cholesky_ex` + `linalg.solve` fallback, `linear` ρ default (CLI), T=400, 2-scale export | `src/learn/nanoquant.rs`: Q32.32 3-limb GEMM + recursive Schur `spd_inverse`; leaf refusal on 8B `l06 down` `r=3040` and the f64 leaf fallback (§3.2); export fixed to `col(A_z)`, `s_mid = 1` (§3.1) |
| **Block Recon** | `compress_model.py:19-122`, `compress_block.py:52-201`: sequential per linear in order `q, v, o, k, gate, up, down` on compressed stream `xs` vs FP `yt`, weighted by `mlp.down_proj.o_norm`: 8 ep `tune_nonfact` (lr `1e-4`, bs 4) → ADMM → 8 ep `tune_fact` STE (lr `1e-5`, bs 1) → `finalize()` | `examples/nq_fit.rs` skips all three. `nanoquant.rs::tune` is single-linear Gram-MSE only. `tests/learn_block_tune.rs` ports the 3-phase loop to Vulkan (f32 vs OG bf16+Kahan), but is `#[ignore]` 0.6B-only and unreceipted |
| **Model KD** | `compress_model.py:125-286`: 8 ep forward-KL distillation (`AdamW`, lr `1e-5`, bs 1, cosine LR) tuning **only `scale_*`** globally with `U`/`V` frozen | Absent from `nq_fit`. Ported in `src/gpu/nn/train/kd.rs` + `tests/learn_kd.rs` (0.6B step 0 only), not wired to 7B/8B pack export |

### 2.4 Format, Kernels & Evaluation

- **Format/Kernels:** OG uses PyTorch `.pt`/`.safetensors`, BF16 embed/head (~2.49 GB on 8B), `+1→0/-1→1` `half2` packing, custom CUDA GEMV (`gemv_kernel.cu`), Marlin binary GEMM, and GemLite Triton (no CPU kernel). `apeiron-nq` uses mmap `APNQPK01` v2/v3/v4 (`+1→1/-1→0`), quantized GGUF skeletons (`Q4_K` embed + `Q6_K` head), CPU AVX2 `q8act` + exact Q32.32 fallback + Lean-refined `Cell<i64>` word core, and no Vulkan 1-bit serving kernel yet (**V1 open**).
- **Evaluation:** OG (`eval_utils.py`) scores full WikiText-2 test (`seqlen=2048`, all `seqlen - 1` positions) + full `lm-eval` 6-task 0-shot. `apeiron-nq` (`bench/PROTOCOL.md` v1) matches the full `seqlen - 1` span (no `--second-half`) with paired Q4_K_M / Q8_0 arms; OG comparison remains cross-harness until OG weights are converted and scored on v1.

---

## 3. Critical Defects Found

### 3.1 `export()` divided by the wrong column norm and `s_mid` restored the wrong product

Two independent defects, both measured on `Qwen3-8B` layer 0 with one fit per row
(`target/release/examples/nq_fit --model Qwen3-8B-Q8_0.gguf --layers 0 --iters 200 --reg 0.3 --diag-tag c6x29 --rel`,
RTX 4090 box, `--rel` prints `relF` = `‖W − Ŵ‖/‖W‖` for both arms on the same trajectory):

1. `col[k] = ‖a[:,k]‖` with `a = A_z/no`. The reference divides by `‖A_z[:,k]‖`
   (`admm_nq.py:235`); `no` is a row scale and cannot enter a column norm. The error is invisible
   on the square linears (`no ≡ 1`) and lands on exactly the transposed ones (`out < in`: `k`,
   `v`, and 7B's `down`).
2. `s_mid = col` re-multiplied the divisor, so the served matrix was `A_z·B_z` instead of the
   B-step product `A_z·diag(1/c)·B_z` (`admm_nq.py:172-177`).

| kind | shape | old: `col(a)`, `s_mid = c` | `col(A_z)`, `s_mid = 1` |
|---|---:|---:|---:|
| q | 4096² | 0.99144 | 0.58224 |
| k | 1024×4096 | 0.99237 | 0.58809 |
| v | 1024×4096 | 0.99392 | 0.61225 |
| o | 4096² | 0.98891 | 0.61890 |
| gate | 12288×4096 | 0.92035 | 0.51606 |
| up | 12288×4096 | 0.92766 | 0.53648 |
| down | 4096×12288 | 0.97277 | 0.65589 |

`m = 1` is the reference's served form: `factorize_admm_nanoquant` returns no `scale_mid`
(`admm_nq.py:233-252`), and `init_for_inference(rank, has_scale_mid=False)` deletes the attribute
when a checkpoint has one (`linear.py:86-99`); `_compute_forward` applies mid only if present
(`linear.py:245-253`). A third arm, `s_mid[k] = col-mean(|B_z|)/mean` ("brank"), was measured on
all seven kinds: no row moves by more than 4e-5, and the committed form stays the reference's
(§3.1's `s_mid` is now `1`). Reverted to `col(A_z)` and `s_mid = 1` in `src/learn/nanoquant.rs`;
`export_serves_the_b_step_product_not_its_column_rescaling` pins both, and the 1.00 b/w pack must
be rebuilt before it is scored again (the 223,222 PPL row was served by the old form).

Our v2 pack still stores `s_mid` (`16·r` bits per linear, about 1.08 MB over the 252 sections at
the fitted ranks), which the reference does not store at all. The 1.000000 b/w pack therefore
carries ~0.010 b/w less rank than the reference at the same nominal rate.

### 3.2 Q32.32 `spd_inverse` failure at rank 3040, and the f64 leaf fallback

`Solve(NotPositiveDefinite)` reproduces on `Qwen3-8B` `l06 down` at `r = 3040`
(`examples/nq_fit --layers 6 --kinds down`, `/workspace/logs/arms-l06.log`), not only on the 7B
shapes. The refusal is in the recursion's leaf (order `≤ INV_BASE = 32`): Q32.32 loses a positive
diagonal entry or a Cholesky pivot, though the block is SPD by the Schur argument in exact
arithmetic. The reference guards the same operation the same way: `cholesky_ex`, and on
`info != 0` `torch.linalg.solve` (`admm_nq.py:83-89`).

`base_inverse` now takes that fallback at the leaf only: `inv_factor_upper`, and on either
`NotPositiveDefinite` or `NonPositiveDiagonal` an `f64` Gauss–Jordan inverse with partial pivoting
rounded back to Q32.32. The bulk of the recursion stays on the limb kernel, so the measured
inversion wall is unchanged on every block that factors.

The leaf repair was not sufficient. With it in place the run passes the pivot, then fails with
`Error: "Range"`: the leaf's diagonal is healthy (`0.368..0.466`, from the refusal's own print), so
the block is not near-singular on its diagonal, but the assembled `A⁻¹` can still be large enough
that the level's `A⁻¹ + Y·Xᵀ` (`src/learn/nanoquant.rs`, the `checked_add` in the assembly) leaves
Q32.32. The failure is therefore structural: forming `H⁻¹` and then multiplying by it needs `H⁻¹`
itself to be representable, which the reference never asks — it solves `H·X = R` with
`cholesky_ex` and `linalg.solve` (`admm_nq.py:83-89`), so only the *solution* has to fit.

`spd_inverse` now falls back at the top level: on `Solve` or `Range` from the recursion at order
`> INV_BASE` it calls `base_inverse(h, n)`, a direct factor of the whole matrix with no Schur
subtraction, and prints `nq solve: order {n} recursion refused; direct factor` once. A magnitude
the direct factor also cannot hold still surfaces, with the offending `max |entry|` printed by
`inverse_f64`. The swap's invariant — the direct factor equals the recursion where the recursion
completes — is pinned by `the_direct_factor_agrees_with_the_recursion`. Status: code written, not
yet compiled or run; the box is stopped (§5, blocker).

### 3.3 Missing calibration and tuning in `examples/nq_fit.rs`

`examples/nq_fit.rs` still lacks `o_norm` (`vec![1.0; out]`), 99.9th-percentile clipping, `0.4`
covariance shrinkage, `tune_nonfact`, `tune_fact` and global scale KD. `--salient-frac` is fixed:
it records the removed columns as `salient_idx`/`salient` and zeroes `s_pre` on them, so the
served form `y += Σ_j sal[·,j]·x[idx[j]]` (`src/engine/kern/nq.rs`) restores exactly what the
factor drops.

### 3.4 A sign factor is not in the ADMM's model class, for the reference either

`W = A·Bᵀ/√r` with `A`, `B` `±1` is representable by the served two-scale form, but not by the
ADMM's class `(A_z·diag(1/c_k))·B_z` with `A_z`, `B_z` of `rank1_approx` shape
`sign ⊙ outer(u, v)` (`admm_nq.py:55-59`): the magnitudes of a random sign matrix are not
separable per rank. Measured on the reference (`factorize_admm_nanoquant`, CPU fp32, 200 outer
iters, `linear`, `reg = 3e-2`, `print_admm_steps` on):

| shape | reference `Loss` at it 1 → 200 | reference served `rel` | ours `admm_loss` |
|---|---:|---:|---:|
| 64×48, r=24 | 1.606 → 0.180 | 0.440 | 0.211 |
| 64×64, r=24 | 1.402 → 0.221 | 0.476 | — |
| 128×96, r=24 | 1.237 → 0.351 | 0.670 | — |

So the in-tree assertion `admm_loss < 0.02` was a claim neither implementation meets; the test is
now `admm_loss_on_a_sign_factor_matches_the_reference_band`, with the reference's band as the
criterion and the served `rel < 0.50` beside it.

### 3.5 Inference split-path blind spots

1. **`NqLinear` bypasses `matmat_q8act` on prefill (`k > 1`):** `src/engine/step/cols.rs:503-511` loops `step_proj_sched` one column at a time whenever `nq_linear(key).is_some()`, re-reading the ~1 GiB `U`/`V` pack **`k` times** (2,048× on a 2k window).
2. **v4 `shared = true` dedupes disk bytes (`src/format/nq.rs:361`), not compute:** `NqLinear::project_with` (`src/engine/kern/nq.rs:1082-1141`) still computes `V·(s_pre ⊙ x)` three separate times for `q`, `k`, `v`, and `group_cols` (`src/engine/step/cols.rs:638-640`) bails out on any NQ layer.

---

## 4. Split-Path & Kernel-Jungle Consolidation Audit

`src/engine/{kern,chart,step}` + `src/learn/nanoquant` has grown to ~37.7k LOC across parallel siblings. Target architecture: **two execution paths only — CPU (`matmat_q8act`) and SPIR-V/Vulkan (`GemvSpec`) — sharing one canonical `Carrier` geometry and safe-Rust arithmetic spec**, with **CPU primary** for the 6.6 GiB L13 Yoga (4C/8T AVX2, no GPU) and mobile groundwork (ARM64 NEON).

| Path / File | LOC | Status | Decision | Action |
|---|---:|---|---|---|
| `matmat_q8act` (`kern/matmat.rs` + `chart/{mod,kcols,x86,portable}.rs`) | 6,882 | Shared `k=1..4096` row loop (`Q8_0/Q5_0/Q4_K/Q6_K`) | **KEEP as THE single CPU projection path** | Every CPU projection (decode `k=1` and prefill `k>1`) routes here. |
| `NqLinear` (`kern/nq.rs`, `loader/nq.rs`, `step/cols.rs:503-511`) | 2,580+ | Parallel `k=1`-only island with 4 internal readers | **FOLD into `matmat_q8act` as `Carrier::Sign1`** | Express a 1-bit sign plane (`u32` packed `±1` rows) as a carrier on `matmat_q8act`. An NQ linear becomes two `matmat_q8act(Sign1, ...)` calls (`V` then `U`) + diagonal epilogues (`s_pre`, `s_mid`, `s_post`). Unlocks `k>1` batched prefill (1 pack read per chunk) and shared-`qkv` single `V` pass. Keep `NqLinear::project` (exact `i128`) as test oracle only. |
| `KernelMode::Ternary` / `Reader::Ternary` (`nq.rs`, `step.rs`, `cols.rs`, `loader/nq.rs`) | ~450 | Exploratory, quality-refuted (77.7M PPL / 1,065,125×); leaks into 14 match arms | **DELETE from production code** | Keep receipts in `bench/` only. |
| `gemm_q8_0` (`kern/gemm.rs` + `gemm_attrib.rs` + `gemm_probe.rs`) | 5,726 | Named sibling debt (`matmat.rs:174`); AVX-VNNI only (dead on L13 Yoga AVX2) | **DELETE probes; FOLD into `matmat_q8act` dial** | Delete `gemm_attrib.rs` and `gemm_probe.rs` (2,513 LOC). Fold 8-col `kcols` panel + pow2 `i64` accumulator into `matmat_q8act`, then delete `gemm.rs`. |
| `Pq2Lut` (`kern/lut.rs` + `lut_tests.rs`) | 1,918 | Separate T-MAC `PQ2_0` tile engine | **QUARANTINE / DELETE** | Off product path; revisit only as a `matmat_q8act` carrier + dot dial. |
| `tp.rs` (`ffn_tp`, `#![allow(dead_code)]`) | 691 | Unwired dead code | **DELETE** | Re-add only if `phase_split` proves FFN dispatch dominates. |
| `native_gpu_probe.rs` | 813 | Probe inside `src/engine/kern/` | **MOVE to `examples/` or DELETE** | Probes do not belong in library kernel modules. |
| `proj_f16_limb` (`step/cols.rs:549-582`) | — | Special `if ggml_type == F16` sibling branch | **FOLD `F16` into `matmat_q8act`** | Keep `kern/limb.rs` strictly for `learn/` if needed; remove `F16` branches from `cols.rs`. |
| `bitplane.rs` (`Q8_0` argmax pruning) | 1,275 | Special-case greedy `Q8_0` head pruning (Lean-proved) | **KEEP isolated as optional head dial** | Do not expand to other dtypes. |
| GPU Decode (`src/gpu/simt/gemv.rs` + `src/gpu/decode/`) | ~2,800 | Declarative `GemvSpec` + `Carrier` enum emitting SPIR-V from one builder | **KEEP as the single SPIR-V/Vulkan path** | Share `Carrier` geometry + safe-Rust spec with CPU `matmat_q8act`; add `Carrier::Sign1` when V1–V4 resumes. |
| Fit/Training stack (4 fragmented fitters) | — | `arelath` converter vs `nq_fit.rs` vs `#[ignore]` 0.6B Vulkan tests vs OG repo | **CONSOLIDATE to OG `SamsungLabs/NanoQuant` + `.pt → APNQPK01`** | Do not maintain two competing training frameworks. Adopt OG (Apache-2.0) + thin converter as authoritative producer until OG parity is proven and beaten; freeze `learn_block_tune.rs`/`learn_kd.rs`. |

---

## 5. Match-or-Beat Original NanoQuant FIRST (Ordered Action Plan)

**Principle:** Reproduce and match `SamsungLabs/NanoQuant` (`a9e0a43`) on our v1 harness **first**. Only after CPU parity is proven, iterate on improvements for the 6.6 GiB L13 Yoga and mobile.

| Step | Host | Exact Files / Command | Exit Criterion |
|---|---|---|---|
| **0a** | Local | This document (`bench/notes/nanoquant-og-analysis.md`) | Grade corrected (`model`), `439k→223k` & unreceipted `learn_block_tune` noted, `NqLinear` prefill/qkv gaps & consolidation table added; `check-claims.py` and `book-check.py` exit 0. *(Done.)* |
| **0b** | Box (`ssh3.vast.ai:10194`) | `cargo test --release --lib --features gpu learn::nanoquant` + `examples/nq_fit --layers 0 --rel` (7 kinds, both export arms on one fit) + the reference's `factorize_admm_nanoquant` on the same sign-factor class | **Done.** Export: `col(A_z)` + `s_mid = 1` moves the served `relF` from 0.920–0.994 to 0.516–0.656 on all seven kinds (§3.1); `s_mid = brank` within 4e-5 of `1`; reference band for an exactly representable sign factor `0.180–0.351` (§3.4). Tests: 8 passed. |
| **0c** | Rented GPU | `/workspace/NanoQuant`: `python -m nanoquant.main --model_id Qwen/Qwen3-8B --bits 1.0` (and `1.18`) + `tools/nq_from_og.py --checkpoint … --out … --og-src /workspace/NanoQuant/src` | OG `.pt` checkpoint + native OG PPL & 6-task 0-shot log + `APNQPK01` pack whose `U`/`V` words the reference's own `binary_unpacker` reproduces bit-for-bit. |
| **0d** | Rented host / Local | `bench/run-ppl.sh` and `bench/run-screen.sh` (`bench/PROTOCOL.md` v1) on converted OG pack vs `Q8_0` and `Q4_K_M` (official + bartowski) | `apeiron-nq` CPU matches OG PyTorch within ~1% (`q8act` bound); v1 JSON record in `bench/results/` passes `tools/bench-check.py`. **Closes Q3 cross-harness gap.** |
| **1a** | Local | Delete `KernelMode::Ternary` / `Reader::Ternary` (`src/engine/{kern/nq.rs,step.rs,step/cols.rs,loader/nq.rs}`) + remove `gemm_attrib.rs`, `gemm_probe.rs`, `native_gpu_probe.rs`, `tp.rs`; quarantine `lut.rs` | Clean diff; contributor gate (`cargo test`, `clippy -j 1`, `fmt`, `check-claims.py`, `book-check.py`) green. |
| **1b** | Local | Fold `Carrier::Sign1` into `src/engine/kern/matmat.rs` (`k = 1..4096`) + shared-`qkv` single `V` pass; remove `NqLinear` bypass in `src/engine/step/cols.rs:503-511` | Bit-exact vs `NqLinear::project_q8` (`k=1` and `k>1`); prefill reads pack once per chunk; shared-`qkv` computes `V` once per layer. |
| **1c** | Local | Fold `gemm_q8_0` and `F16` into `matmat_q8act` dials; delete `src/engine/kern/gemm.rs` | Bit-exact vs `matvec_q8act`; contributor gate green; zero sibling GEMM loops remain. |
| **2a** | Local (L13 Yoga) | `examples/phase_split --steps 24` + 3-run decode at ctx 2048 on Phase 0 OG pack | Measured post-consolidation baseline on L13 Yoga naming the dominant term and its `1/(1−s)` ceiling. |
| **2b** | Local / Box | **Improve from OG's frontier (only after 0d + 1c + 2a):** (1) Quality: verify if OG 1.18 b/w clears Q2 (`≤1.5×` PPL, `≥0.90×` 0-shot); if not, iterate *inside* OG's 4-stage pipeline (shared-`qkv` v4 freeing ~0.105 b/w for sensitive layers, 3-scale `s_mid`, broader calib). (2) CPU speed (T1) & mobile: fit LM head onto `Sign1` (1.406→0.998 GB/tok, bytes ceiling 19.9→28.0 tok/s); replace 12-instr `code_select_avx2` dot dial with a **quality-exact 4-term LUT (`vpshufb` AVX2 / `vtbl` ARM64 NEON)** tested equal to `code_select_twin` (0% quality loss vs `q8act`). (3) 200k KV (C1/C2) and Vulkan `Carrier::Sign1` (V1–V4). | Every change backed by `phase_split --target`, 3-run L13 Yoga spread, and PROTOCOL v1 PPL + 6-task 0-shot record. |

*(The in-tree fitter keeps `spd_inverse` for the bulk, falls back to `f64` at the Schur leaf, and falls back to a direct factor of the whole matrix when the recursion refuses at all (§3.2). Remaining parity work is the calibration statistic set (`o_norm` backward hook, 99.9% clip, `0.4` shrinkage) and the two tuning stages, not the linear algebra: do not block Phase 0–2 on `learn_block_tune`/`learn_kd`.)*

**Blocker (measured, owner action).** Instance `61176577` (RTX 4090, `ssh3.vast.ai:10194`) is
`exited` with `intended_status: stopped`; `vastai show user` reports `balance = -0.011056`, below
the `-0.01` threshold, and `vastai create instance` answers `Your account lacks credit`. The disk is
retained (packs, `diag/`, `hf/`, `venv/`, logs). Until the balance is positive nothing on the box
can be built or run, and no local `cargo` run is admitted (`MemAvailable` ~1.4 GiB against the
guard's derived cap). Every row above that names the box is therefore written-but-unbuilt or
measured-before-the-last-edit, and says which.

---

## 6. Non-Coverage

- Does not execute `cargo test`, `clippy`, `fmt`, `lake build`, or `SamsungLabs/NanoQuant` on a GPU (`agent-guard` refused local cargo at 850 MiB < 1024 MiB available).
- Does not cover rustc/LLVM code generation, the kernel ABI, Vulkan drivers, or `unsafe` containers.
