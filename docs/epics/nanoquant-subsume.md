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

0.34 s per ADMM iteration at 1024×3584, r = 768; 0.43 s at r = 896 (171.1 s for 400). Tune: 0.39 s per
step at r = 896 (403.9 s for 1024), against the projected `o·n² + 3·o·n·r` = 23.1 GMAC. 400 iterations = 137–170 s per
tensor; 28 tensors ≈ 4000 s, one kind (7 tensors) ≈ 1000–1200 s: both above the 600 s ceiling. Levers:
iterations (150 it costs +7% relH), 2 limbs (K3: needs a measured metric), the GPU path.

### Wave map

| wave | task | files | gate |
|---|---|---|---|
| W1 | ADMM port | `src/learn/nanoquant.rs`, `src/learn/mod.rs`, `tests/learn_nanoquant.rs` | `cargo test --release --all-features --locked --test learn_nanoquant` |
| W2 | layer-scope STE tune: export latents; Adam on `L = tr((Ŵ−W)H(Ŵ−W)ᵀ)` through `sign` | `src/learn/nanoquant.rs`, `tests/learn_nanoquant.rs` | same; relH below the W1 row at 1.1451 b/w |
| W3 | KL arm vs same-scope Q4_K_M (chunks 8–9) | `tests/learn_gate_code.rs` | KL log |
| W4 | Lean/Aeneas: budget, factored forward, SVID identity, ridge minimiser | `proof/corpus/Learn/Quant/NanoQuant.lean`, `proof/corpus/AxiomAudit.lean` | `cd proof/corpus && lake build` |
| W5 | `o_norm` importance, block-scope tune | `src/learn/nanoquant.rs` | relH, then KL |

W2 step cost: `o·n² + 3·r·o·n` MACs ≈ 13.2 + 9.9 GMAC at `down`, 0.39 s measured.

### Negative controls

Dropping `s_pre` in `NqFactor::weight` in a throwaway copy turns
`export_matches_its_dense_form_and_loss_falls` red; the restored tree passes.

W2: negating `∂s_post` before its Adam step in a throwaway copy turns
`latents_carry_the_signs_and_tune_lowers_the_h_energy` red at its descent assertion; the live tree
passes.

### Stop and rollback conditions

A W2 tune that does not beat the W1 relH at equal b/w is recorded as a failure in the quantisation
model and stops the layer-scope route in favour of block scope.
