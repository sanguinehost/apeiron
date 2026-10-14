# Quantum register: gate stages over `R[x]/(x²−σ)` and the learned rotation

Board row: `quantum-register-learned-LD7oW6M8EVQgAkWyMtEOs`. Evidence for runs lands in
[`docs/evidence/onebit-w15.md`](../evidence/onebit-w15.md) under its own section; Lean in
`proof/corpus/Numeric/Register.lean` and `proof/corpus/Perf/`.

## Intent

- **Owner:** lead (Claude Code), owner go given for this epic.
- **Source revision:** `f4bb03e`; working tree dirty only in `.intent/` bookkeeping and `.held-control/`.
- **Status:** approved.

### Problem

The one-bit quantizer rotates every 128-column group by a fixed Sylvester Hadamard `H^⊗7 / √128`
(`tools/onebit/fold_transfer.py:440`, `fwht128`) before the 2-plane E8 code solve. Finishing trained only the per-group
scales (`r, c, d`, norms) and took held-out KL from 1.1637 to 0.8484 at 1.178 b/w (`Perf/PVFinish.lean`,
`hadmm_lower`). Read in the cell representation, the rotation and the scales are two of the three faces of a 2×2 real
block:

| factor of a 2×2 block | face | corpus statement | trained today |
|---|---|---|---|
| K: rotation `rep (−1) (cos t, sin t)` | elliptic, `σ < 0` | `rep_elliptic_is_rotation`, `elliptic_one_param` (sanguine `Apeiron/Physics/Quadratic.lean`) | no: fixed at the Hadamard angle |
| A: squeeze `rep 1 (cosh t, sinh t)` | hyperbolic, `σ > 0` | `hyperbolic_one_param`, `unitNorm_hyperbolic` | yes: `r, c, d`, norms |
| N: shear `rep 0 (1, v)` | parabolic, `σ = 0` | `parabolic_one_param`, `shear_additive_iff_parabolic` | no: the low-rank corrector is frozen |

The Hadamard butterfly `(u, v) ↦ (u + v, u − v)` is `channelPair 1` (`sylvester_one_channelPair`,
`proof/corpus/Numeric/Sylvester.lean:186`): the change to the idempotent coordinates of the hyperbolic face. A 128-group
is a register of `2^7` real amplitudes; `fwht128` is seven stages of 64 such butterflies, i.e. the circuit `H^⊗7`
(Clifford, classically simulable: Gottesman–Knill, pointer, not retrieved). Replacing the fixed butterfly with a learned
elliptic angle per butterfly makes the K factor trainable at no stored-weight cost: `W x = (W T)(Tᵀ x)`, the code
quantizes `W T` and `Tᵀ` runs on activations as the Hadamard does now.

The quantum-classical reading this rests on is in the corpus: Lohmiller–Slotine (arXiv:2405.06328) and the Vattay
exchange (arXiv:2605.02621, 2605.20443, 2606.05197) at `../sanguine/theory/LITERATURE.md:838–880`;
`proof/corpus/Substrate/Algebra/QuantumField/QuantumClassicalIsomorphism.lean` (`quantum_potential_is_the_gap`,
`stencil4_reconstructs_laplacian`); sanguine `actual_schrodinger_iff_madelung`, `det_rep`, `generator_trichotomy`. The
semiclassical trace formula is exact iff the Hamiltonian is at most quadratic (Vattay, arXiv:2606.05197); the gate set
here (cell multiplies, `channelPair`, sign diagonals) stays inside that window.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Keep the fixed Hadamard, train more scale steps | Token-limited and falling by ≈ 0.04 per doubling (`hadmm_lower`); it does not touch the rotation factor |
| Dense learned orthogonal 128×128 per group (SpinQuant arXiv:2405.16406, OSTQuant arXiv:2501.13987, Cayley-parametrised) | Incumbents to subsume, not yet retrieved; a dense rotation is 128² parameters per group and 128 FLOPs per word at inference against `7 · 2` for a butterfly stage. The canonical Cayley isometry (`cayley_isometry`, sanguine `Trichotomy/Derived/Elliptic.lean:637`) covers the dense case if the butterfly loses |
| Generic `Matrix` rotation primitive in Rust | One implementation per math object: the stage is the existing `Cell<i64>` multiply (`src/cell.rs:127`, `word_mul_denote`) swept over a register |

### Completion condition

The learned-K rotation either lowers held-out KL before finishing below 1.1637 by more than the seed spread at
≤ 1.18 b/w with the angle bits counted, and finishing on it ends below 0.8484, or the epic records the refutation as a
Lean verdict; in both cases the register primitive is proved and refined.

```sh
cd proof/corpus && lake build Numeric.Register Perf.QuantumRotation
cargo test --workspace --all-features --locked register
proof/regen.sh --check
```

### Invariants

1. The register stage is the cell multiply at a value `σ`; no separate rotation formula.
2. Angles cost bits: every receipt counts them in `bits` (16-bit angles: 1,792 per layer for the four input sites of
   Qwen2.5-0.5B, 688,128 bits in all, 0.00192 b/w; the 1.178061 b/w arm has 0.00194 b/w of slack under 1.18).
3. At `θ = π/4` with the sign diagonal the learned stage equals `fwht128` exactly (float64 control `< 10⁻¹²`).
4. Checked `i64` arithmetic in Rust: overflow is `Err`, never wrapping.

### Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| An elliptic stage scales the register's sum of squares by exactly `normForm (−1) w` | proved | `register_stage_normForm` |
| A stage by `w` then by its conjugate is the scalar `normForm σ w` | proved | `register_stage_conj` |
| Seven `channelPair 1` stages are `sylvesterH 7` | proved | `register_hadamard` (from `fwhtRef_getElem`) |
| Every `SL(2,ℝ)` block is elliptic · (Hadamard-conjugated hyperbolic) · parabolic | proved | `iwasawa_faces` |
| Cost: `n · 2^(n−1)` cell multiplies per register, 4 multiplies and 2 adds each | proved | `register_cost` |
| Rust `stage_mul` refines the Lean stage | proved | `stage_mul_denote` via `word_mul_denote` |
| Learned K lowers the Hessian-weighted E8 error of the code at fixed bits | measured | `tools/onebit/kan_probe.py`, RTX 5080 |
| Learned K lowers held-out KL before finishing below 1.1637 | measured | `fold_transfer.py --rotation kan` |
| The register is a quantum circuit simulator for real Clifford-plus-rotation circuits | model | the mapping above, stated as a mapping |

### Non-goals

- Complex amplitudes and non-real gates; the register is real (`σ = −1` cells carry the complex pairs).
- Hardware quantum execution or a claim of quantum advantage.
- Any gate outside the quadratic (Gaussian/Clifford/matchgate) window.

### Trust boundary

rustc/LLVM codegen, the torch/CUDA training path (floating point, `measured` only), the bf16 cast (measured in
`hadmm_bf16_amplification`), and the fixed-point angle table (`src/numeric/fixed_trig.rs`) beyond its own tests.

### Open questions

- Angle tying: per (stage, butterfly) shared across a linear's groups (448 per input site, the default), or per stage
  only (7). Decided by the W1 probe.

## Specification

### Scope and admissible domain

| Item | Value |
|---|---|
| Operation | `stage_mul(x: &mut [i64], span: usize, w: Cell<i64>, sigma: Sigma) -> Result<(), RegisterError>` |
| Domain | `x.len()` a multiple of `2 · span`; `span = 2^s`, `0 ≤ s < 16`; any `w`, any `sigma` |
| Totality | `Result`: `Err(LengthMismatch)`, `Err(NotPowerOfTwo)`, `Err(Overflow)` on a checked multiply/add |
| Postcondition | for every block base `b` and lane `i < span`: `(x[b+i], x[b+i+span]) ← w · (x[b+i], x[b+i+span])` in `R[x]/(x²−σ)` |

The training path uses float angles `θ` with `w = (cos θ, sin θ)`, `σ = −1`; the Rust path takes `w` from
`fixed_sin_cos` (`src/numeric/fixed_trig.rs:149`) and the stage scales by `normForm(−1) w ≈ 2^(2q)`, exact on
integers.

### Representation relation

| Layer | Symbol | Path |
|---|---|---|
| Production Rust | `stage_mul` | `src/numeric/register.rs` |
| Extraction wrapper | `word_stage_mul` | `proof/extract/lib.rs` |
| Extracted body | `register.stage_mul` | `proof/ApeironWord.lean` (generated) |
| Cell multiply | `word_mul_denote` | `proof/corpus/WordRefinement.lean` |
| Canonical stage | `Numeric.Register.stage` | `proof/corpus/Numeric/Register.lean` |
| Refinement | `stage_mul_denote` | `proof/corpus/Numeric/RegisterRefinement.lean` |

### Contracts

| Contract | Statement | Grade target |
|---|---|---|
| `register_stage_normForm` | `σ = −1`: `Σ (stage w x)_i² = normForm (−1) w · Σ x_i²` | proved |
| `register_stage_conj` | `stage (conj w) (stage w x) = normForm σ w • x` | proved |
| `register_hadamard` | `fwhtRef 7 = sylvesterH 7 *ᵥ ·` on a 128-register | proved |
| `iwasawa_faces` | `det M = 1 → ∃ t a v, 0 < a ∧ M = rep (−1) (cos t, sin t) · diag(a, a⁻¹) · (rep 0 (1, v))ᵀ`; `diag_eq_hadamard_conj_rep_hyperbolic`: `diag(eᵗ, e⁻ᵗ) = ½ H · rep 1 (cosh t, sinh t) · H` (`iwasawa_faces_hadamard`) | proved |
| `register_stage_diagonal` | the Hadamard stage is not a cell multiply: `(1,1)` at `σ = 1` sends `(u, v)` to `(u+v, u+v)`. The butterfly is `channelPair 1`, and it diagonalises the `σ = 1` cell stage | proved |
| `register_cost` | `n` stages × `2^(n−1)` pairs, 4 mul + 2 add per pair | proved |
| overflow | an `i64` overflow returns `Err(Overflow)` and leaves no partial claim | tested |

### Evidence plan

1. Lean: `cd proof/corpus && lake build Numeric.Register`; one changed-constant control per theorem.
2. Probe: `kan_probe.py` learns angles per input site on the calibration Grams by minimising the Hessian-weighted error of
   the E8 nearest-point code (straight-through), then re-scores with the real `gptq_e8` solve; receipt JSON, Lean verdict.
3. Transfer: `fold_transfer.py --rotation kan` re-solves the arm with the learned `T`; held-out KL before finishing vs
   1.1637 over three seeds.
4. Finishing: `pv_finish.py` on the learned-`T` arm, against 0.8484.
5. Rust: `cargo test` positives and overflow; mutation of the multiply in a throwaway copy turns `stage_mul_denote` red.

## Plan

Hardest and highest-contribution first: the measurement decides whether the rest of the K work ships.

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | T1 probe: learned K vs fixed Hadamard on the code error | `tools/onebit/kan_probe.py`, receipt, `Perf/QuantumRotation.lean` | lead | `kan_probe.py` exit 0 under its prediction; `lake build Perf.QuantumRotation` | angles at `π/4` reproduce `fwht128` (`< 10⁻¹²`); changed constant per theorem |
| W1 | T2 register theorems | `proof/corpus/Numeric/Register.lean` | Sonnet lane | `lake build Numeric.Register` | changed constant / swapped `σ` per theorem |
| W2 | T3 transfer with learned `T` (blocked on T1 lowering the error) | `tools/onebit/fold_transfer.py`, receipt, `Perf/QuantumRotation.lean` | lead | held-out KL receipt, three seeds | `--rotation kan` at Hadamard angles equals the `-nb` arm bit for bit |
| W2 | T4 Rust stage and refinement | `src/numeric/register.rs`, `proof/extract/lib.rs`, `Numeric/RegisterRefinement.lean` | Sonnet lane | `cargo test register`; `proof/regen.sh --check`; `lake build Numeric.RegisterRefinement` | mutate the multiply in a copy: refinement red |
| W3 | T5 finish on learned `T`, then train N (the corrector); refuted, see W3-T5 result | `tools/onebit/pv_finish.py`, receipts | lead | held-out KL vs the frozen-corrector pair | as `hadmm` controls; step-0 `torch.equal`, `compose_only` loss rel 0 |

Stop: W1 refutes (no error drop beyond the seed spread at fixed bits) → record the verdict theorem, keep T2/T4 (the
primitive stands on its own), and move K to the dense Cayley route (`cayley_isometry`) as the next row.

### W2-T4 Rust stage (`tested`) and its extraction (`proved`)

`src/numeric/register.rs` `stage_mul` multiplies each pair `(x[b+i], x[b+i+span])` by `w` through `Cell::<i64>::checked_mul`
(`src/cell.rs`, appended after every extracted item so `proof/regen.sh --check` is unchanged): `Err(Overflow)` on any
`i64` overflow, `Err(NotPowerOfTwo)`, `Err(LengthMismatch)`. Loops are flag loops, one per function, index carried by
value (the shapes the pinned Aeneas accepts). `cargo test --lib register`: 6 passed (Hadamard-angle pair, `normForm`
scaling by 25, conjugate stage = `normForm σ w • x` on all three faces, agreement with `Cell::mul`, overflow and domain
refusals). Controls: lo/hi swap in the write-back fails 2; dropping `σ` from `checked_mul` fails 5.

The extraction root `word_stage_mul` (`proof/extract/lib.rs`, `#[path]` to the real `src/numeric/register.rs`) is added to
`proof/regen.sh` (`--start-from crate::word_stage_mul`, source list); `proof/regen.sh --check`: `extraction is fresh`,
1331 → 1339 generated defs, the 8 added being `cell.CellI64.checked_mul`, `register.stage_{mul,block}{,_loop,_loop.body}`
and the wrapper. `proof/corpus/Numeric/RegisterRefinement.lean` (`lake build Numeric.RegisterRefinement`, 3686 jobs):

| Theorem | Statement |
|---|---|
| `checked_mul_spec` | on `MulFits` (the five intermediates fit `i64`) the extracted multiply returns `some c` with `denoteCell c = ringMul σ w p` |
| `checked_mul_none`, `checked_mul_none_of_a` | an out-of-range product or sum returns `none` |
| `stage_mul_not_pow`, `stage_mul_too_large`, `stage_mul_length_mismatch` | the three refusals, slice unchanged |
| `stage_block_spec`, `stageRef_denote` | one block writes each pair's checked product; the block fold denotes `cellStage` at `span = 2^k` |
| `stage_mul_denote` | span `2^k < 2^16`, length a multiple of `2·span`, every pair in `MulFits`: `Ok ()` and entry `i` denotes `Register.cellStage (denoteSigma σ) (denoteCell w) k` |
| `stage_mul_denote_witness` | the hypotheses hold jointly on `[3, 5]`, span 1, `w = (2, 1)`, `σ = 1` |

All ten are in `AxiomAudit.lean` (`{propext, Classical.choice, Quot.sound}`). Controls: in a renamed copy of the file,
swapping `w`'s slots or negating `σ` in `mulCell_denote` fails at line 617 (exit 1 each). Generated-body control
`bash proof/controls/register.sh` (copies under `/home/socol/oc-tmp`, never the working tree; 12G cap, lane lock): baseline
green; swapping the lo/hi write-back in `register.stage_block_loop.body` fails at `RegisterRefinement.lean:484`; dropping
`σ` from `cell.CellI64.checked_mul` (`I64.checked_mul s self.b` → `1#i64`) fails at `RegisterRefinement.lean:159`; restored
green; exit 0 in 489 s of 580. The two mutations build in parallel copies, because one fresh chain after an
`ApeironWord.lean` change is ~245 s (`Numeric.Exp.Part2` 186 s). With both mutations made identity edits, the script
exits 1 with "mutation not detected" on both steps, so its own refusal fires.
Board row `quantum-register-learned-LD7oW6M8EVQgAkWyMtEOs`.

### W1-T1 result (prediction `kan-probe-1`, refuted)

Receipt [`docs/evidence/receipts/onebit/qreg-kan-probe.json`](../evidence/receipts/onebit/qreg-kan-probe.json)
(sha256 `e13b8356…`, `kan_probe.py` `981e9fc7…`, RTX 5080, 447 s); verdicts in `proof/corpus/Perf/QuantumRotation.lean`
(controls: flipped sum, `92 → 91`, `194 → 192`, each red at its line).

| site | J learned / J Hadamard, calibration | held-out |
|---|---|---|
| q/k/v input | 0.99481 | 0.99623 |
| o input | 0.91704 | 0.91504 |
| gate/up input | 1.00429 | 1.00494 |
| down input | 1.00232 | 1.00191 |
| all, summed | 1.00298 | 1.00368 |
| all, geometric mean of 96 | 0.97857 | 0.97842 |

The angles cost `0.00192` b/w (`688128` bits). The surrogate is per-layer with the code detached; the wins sit on
small-J solves and on the o input. Next row: angles trained on end-to-end KL inside `pv_finish`, starting from the
all-`π/4` register (bit-equal to the Hadamard arm), and the o-input-only register as its cheaper control.

### W2 result: angles on end-to-end KL (prediction `kan-kl-exact-1`, refuted on wall, wins per step)

Receipt [`docs/evidence/receipts/onebit/qreg-kan-exact.json`](../evidence/receipts/onebit/qreg-kan-exact.json)
(sha256 `4bf34898…`, `pv_finish.py` `17586d6d…`, RTX 5080, 542 s); verdicts `kan_exact_protocol`, `kan_exact_step_cost`,
`kan_exact_loses_wall`, `butterfly_curve_convex`, `kan_exact_wins_matched_step` in
`proof/corpus/Perf/QuantumRotation.lean`, audited in `AxiomAudit.lean` (controls: register KL `0.85`, `0.9`, the 529-step
KL `0.91`, register train time `215` s, each red at its own theorem).

| quantity | register (`--rotation kan`, butterfly) | fixed Hadamard |
|---|---|---|
| step-0 weights | `torch.equal` to `fwht128`'s on all 168 | — |
| s/step | 1.050 | 0.571 (`w18_pvfast`); ratio 1.84 |
| steps in the 580 s run | 377 | 729 |
| held-out KL after | 0.8979 | 0.8577 |
| butterfly KL at 377 steps, convex bracket | — | [0.9012, 0.9062] |
| b/w | 1.179984 (+688128 angle bits) | 1.178061 |

Per step the register is below the butterfly curve's convex lower bound (`anchor`: convexity between the measured
points; `kan_exact_wins_matched_step`); at equal wall it loses because each of its seven
stages runs as a separate kernel. Next row: the register as one `128 × 128` matmul `T(δ)` per (layer, site) per step
(`kan_matrix`, `RegisterComposeMM`), the same cost shape as `had_mm`; its gate is the held-out KL against `hadmm`'s 0.8484.

### W2 result: the register as one matmul (prediction `kan-kl-mm-1`, refuted on wall, wins per step)

Receipt [`docs/evidence/receipts/onebit/qreg-kan-mm.json`](../evidence/receipts/onebit/qreg-kan-mm.json)
(sha256 `62d597a6…`, `pv_finish.py` `4611904f…`, RTX 5080, 542 s); verdicts `kan_mm_protocol`, `kan_mm_step_cost`,
`kan_mm_loses_wall`, `kan_mm_wins_matched_step` in `proof/corpus/Perf/QuantumRotation.lean`, audited in `AxiomAudit.lean`
(controls: register KL `0.84`, `0.8785`, train time `263` s, offset drift `0.08`, each red at its own theorem).
`--selftest-kan` gates the hand adjoint `RegisterComposeMM` in float64 against the butterfly path (adjoints
`≤ 8.4·10⁻¹⁶`) and seeds two defects it must catch: `G` dropped from the column-scale adjoint (rel 1.57) and `T`
untransposed in the input adjoint (rel 0.39; `H128` is symmetric, so that defect is visible only at `δ ≠ 0`).

| quantity | register, matmul | register, butterfly | fixed Hadamard, matmul (`w18_hadmm`) |
|---|---|---|---|
| s/step | 0.707 | 1.050 | 0.479 |
| steps in the 580 s run | 550 | 377 | 871 |
| held-out KL after | 0.8779 | 0.8979 | 0.8484 |
| butterfly KL at the register's steps, convex bracket | [0.8791, 0.8798] | [0.9012, 0.9062] | — |
| b/w | 1.179984 | 1.179984 | 1.178061 |

The matmul path is `0.673×` the butterfly register's step and still `1.47×` the fixed matmul's: the 96 `T(δ)` builds
run seven stages of small kernels each, forward and backward. Next row: all 96 built in one batched register pass
(`kan_matrix_batch`, seven stage launches per step; bitwise equal to `kan_matrix` per key, `--selftest-kan` controls 9–10).

### W2 result: the batched register (prediction `kan-kl-mmb-1`, refuted on speed and steps, beats the butterfly at equal wall)

Receipt [`docs/evidence/receipts/onebit/qreg-kan-mmb.json`](../evidence/receipts/onebit/qreg-kan-mmb.json)
(sha256 `c24380bb…`, `pv_finish.py` `302ae284…`, RTX 5080, 542 s); verdicts `kan_mmb_protocol`, `kan_mmb_step_cost`,
`kan_mmb_beats_butterfly_wall`, `kan_mmb_loses_hadmm`, `kan_mmb_wins_matched_step` in
`proof/corpus/Perf/QuantumRotation.lean`, audited in `AxiomAudit.lean` (controls: offset drift `0.08`, train time `300` s,
per-key register KL `0.86`, register KL `0.85`, `0.8565`, each red at its own theorem). `--selftest-kan` 10/10: the batch
is `torch.equal` to `kan_matrix` per key in fp32 and fp64, `dL/dδ` rel `2.7·10⁻¹⁶` against the stacked per-key
gradients; the `a ↔ b` swap mutant moves it by rel `0.95`.

| quantity | register, batched | register, per-key matmul | fixed butterfly (`w18_pvfast`) | fixed Hadamard, matmul (`w18_hadmm`) |
|---|---|---|---|---|
| s/step | 0.570 | 0.707 | 0.571 | 0.479 |
| steps in the 580 s run | 683 | 550 | 729 | 871 |
| held-out KL after | **0.8547** | 0.8779 | 0.8577 | 0.8484 |
| top-1 after | 0.5475 | 0.5424 | — | — |
| butterfly KL at the register's steps, convex bracket | [0.8583, 0.8634] | [0.8791, 0.8798] | — | — |
| b/w | 1.179984 | 1.179984 | 1.178061 | 1.178061 |

The learned register now runs at the fixed butterfly's step cost and beats it at equal wall (`0.8547 < 0.8577`). The
per-step margin under the butterfly's convex lower bound grows from `0.00125` at 550 steps to `0.0036` at 683
(`anchor`: convexity). The fixed Hadamard matmul is still ahead by `0.0063` at 188 more steps. Its `0.84×` step was
measured on a different host state; the pair below puts the register within `1.04×` of it on one host state, so the
`0.091` s/step gap is not the angle gradient. Next row: W3-T5, train the corrector `N` (`Corr = U Vᵀ`, rank 2 on q,
rank 1 on down; `fold_transfer.py` `lowrank_corrector`, `pv_finish.py` `Plane.dense`) as factors at unchanged bits
(`bits_corr` 1798656), on the batched register; gate: held-out KL against `0.8484`.

### W2 result: step-time pairs on one host state (prediction `qreg-step-pair-1`, confirmed)

Receipt [`docs/evidence/receipts/onebit/qreg-step-pair.json`](../evidence/receipts/onebit/qreg-step-pair.json)
(sha256 `e192abcb…`); verdicts `kan_pair_step_parity`, `kan_pair_each`, `pair_host_slowdown` in
`proof/corpus/Perf/QuantumRotation.lean` (constants generated by `tools/rooting/receipt_lean.py`, namespace
`qreg_step_pair`; control: one arm's `t100` moved by 1 s in a copy, `--check` exit 1 on the stale sha, restored exit 0).
Six 101-step runs, `n-cal 50`, `mm`/`step`/`fused`, ABBA order `kan1 had1 had2 kan2 kan3 had3`, a co-tenant process at
~570% CPU throughout; steady s/step is `(t100 − t25)/75`.

| pair | register (`--rotation kan`) | fixed Hadamard matmul | ratio |
|---|---|---|---|
| 1 | 0.5648 | 0.5525 | 1.022 |
| 2 | 0.5637 | 0.5521 | 1.021 |
| 3 | 0.5589 | 0.5275 | 1.060 |
| mean | 0.5625 | 0.5440 | 1.034 |

The register costs `1.02–1.06×` the fixed Hadamard matmul per step, not `1.19×`. On this host state the matmul arm ran
`1.10–1.15×` its `w18_hadmm` step, and `qreg_kan_mmb`'s mean step is within `1.02×` of the pair's register step: the
`1.19×` in `kan_mmb_step_cost` compared two host states. Rule taken from this: a step-cost ratio is measured as
interleaved pairs on one host state, never across receipts.

### W3-T5 result: the trainable corrector (prediction `qreg-t5-corr-3`, refuted; base `qreg-t5-base-1`, confirmed)

Receipt [`docs/evidence/receipts/onebit/qreg-t5-corr.json`](../evidence/receipts/onebit/qreg-t5-corr.json)
(sha256 `a9b456eb…`); verdicts `t5_corr_store_refines`, `t5_corr_loses` in `proof/corpus/Perf/QuantumRotation.lean`,
audited in `AxiomAudit.lean` (constants from `tools/rooting/receipt_lean.py`, namespace `qreg_t5_corr`; control: the corr
arm's KL moved by `0.01` in a copy, `--check` exit 1 on the stale sha, restored exit 0). The corrector trains as its
stored factors, `corr_weight = u8(U) @ u8(V)` (`fold_transfer.py` `lowrank_corrector(factors=True)` keeps them;
`pv_finish.py` `factor_dense` refuses factors that are not `u8_ste` fixed points). Its store is the extracted count:
`t5_corr_store_refines` equates it with `t3_nb_3seed.bitsCorr`, which `BitsRefine.qwen_bits_corr` equates with
`bits_corr` of the extracted Rust on the Qwen configuration. Both arms carry the same 48 corrector planes; the base arm freezes them.

| quantity | corrector frozen (base) | corrector trained, `lr_corr 1e-4` |
|---|---|---|
| steps in the 580 s run | 740 | 732 |
| trainable parameters | 3949952 | 4174208 |
| held-out KL after | **0.8491** | 0.8539 |
| held-out top-1 after | 0.5497 | 0.5486 |
| b/w | 1.179984 | 1.179984 |

Training the corrector raises held-out KL by `0.0048` at equal bits and equal wall (one pair, base then corr). Two
earlier runs were refuted before training, each by a gate that did its job: `qreg-t5-corr-1` re-factored the dense plane
by SVD and re-coded it (step-0 weights 120/168 `torch.equal`); `qreg-t5-corr-2` composed `u8(U) @ u8(V)` in bf16 inside
the reference path's autocast and in fp32 on the step path (`compose_only` loss rel `1.66·10⁻³`). `corr_weight` now runs
the product with autocast off; both arms show `compose_only` loss rel `0`. The base arm is also the register's best
equal-wall point: `0.8491` at 740 steps against `w18_hadmm`'s `0.8484` at 871. Next row: the corrector's rate, not its
training — the 48 planes spend `1798656` bits on a rank-1/2 fix whose trained form loses; the question for that row is
whether those bits buy more as sign-plane or `r` bits at the same total.

### Attribution: where the held-out KL lives (prediction `qreg-attrib-1`, refuted)

Receipt [`docs/evidence/receipts/onebit/qreg-attrib.json`](../evidence/receipts/onebit/qreg-attrib.json)
(sha256 `fa668251…`); verdicts `attrib_no_layer_holds_kl`, `attrib_kl_nonadditive`, `attrib_kinds_coadapted` in
`proof/corpus/Perf/QuantumRotation.lean`, audited in `AxiomAudit.lean` (constants from `tools/rooting/receipt_lean.py`,
namespace `qreg_attrib`; controls: the before-phase base against the after-phase layers leaves layer 0's goal `False`,
exit 1; the base moved by `0.01` in a receipt copy, `--check` exit 1 on the stale sha; restored, both exit 0).
`pv_finish.py --attrib N` restores the teacher's linears one group at a time on the first `N` held-out chunks, before
and after finishing in one run: each layer's seven, each kind across all 24 layers, and all of them; the teacher as its
own student is the zero control (KL `0`), and a no-op restore fails the run's "some layer moves the KL" assert.
Arm: 1.178 b/w, n-cal 50, `mm/step/fused`, `--attrib 8`, 683 wall-budgeted steps (the attribution's 34 s comes out of
the finishing wall), held-out KL `1.1637 → 0.8660`.

| on 8 held-out chunks | before finishing | after finishing |
|---|---|---|
| student KL (base) | 0.7957 | 0.6626 |
| largest single-layer restore delta | 0.0839 (L23) | 0.0463 (L23, 7.0%) |
| sum of the 24 single-layer deltas | 0.4713 (0.59× base) | 0.2666 (0.40× base) |
| cross-layer part, base − sum | 0.3244 | 0.3960 |
| restore `down` in all layers | +0.6045 | +0.1854 |
| restore `gate` / `o` / `up` in all layers | +0.0135 / +0.0785 / +0.0212 | +0.0657 / +0.0563 / +0.0372 |
| restore `q` / `k` / `v` in all layers | −0.0060 / +0.0014 / +0.0138 | +0.0023 / +0.0016 / +0.0144 |
| restore every linear | 0.0112 | 0.0106 |

(Kind rows are the KL change from restoring that kind, so `+` means the teacher's kind alone is worse than the
student's.) No layer holds the KL: the largest is 7% of it, so a per-layer rate reallocation reaches at most the additive
part. 60% of the finished KL is cross-layer, and finishing moves KL into that part (additive `0.471 → 0.267`, cross-layer
`0.324 → 0.396`). After finishing every kind is co-adapted: restoring any one kind raises the KL, `down` most. The
error to remove is a property of the composed network, so the lever is the end-to-end objective and its rate per
second of finishing, not a per-tensor fit. Next row: restore contiguous depth blocks (prefix and suffix), to see whether
the cross-layer part grows with depth as compounded residual-stream error does.

### Attribution: layer errors saturate, they do not compound (prediction `qreg-iso-1`, confirmed)

Receipt [`docs/evidence/receipts/onebit/qreg-iso.json`](../evidence/receipts/onebit/qreg-iso.json)
(sha256 `c3790895…`); verdicts `iso_kl_superadditive`, `iso_groups_saturate`, `iso_finishing_saturates`,
`iso_every_layer_costs` in `proof/corpus/Perf/QuantumRotation.lean`, audited in `AxiomAudit.lean` (namespace
`qreg_iso`; controls: the after prefix-12 ratio bounded by `0.57` and the before/after suffix-18 ratios swapped each leave
`⊢ False`, exit 1; layer 11's iso KL moved by `0.01` in a receipt copy, `--check` exit 1 on the stale sha; restored, both
exit 0). Board row `quantum-register-learned-LD7oW6M8EVQgAkWyMtEOs`. The complementary view to the restore rows
above: `pv_finish.py --attrib N` also starts from the all-linear configuration (every linear the teacher's) and puts the
student's linears back into one layer (iso), into layers `0..k−1` (prefix) or `k..23` (suffix), `k ∈ {6, 12, 18}`; a
no-op put-back fails the run's "no layer's student linears cost KL in isolation" assert. Cost of a group is its KL minus
all_linear. Arm as above, 589 wall-budgeted steps (the two attribution phases take 34 s and 33 s of the finishing wall),
held-out KL `1.1637 → 0.8776`.

| on 8 held-out chunks | before finishing | after finishing |
|---|---|---|
| base / all_linear | 0.7957 / 0.0112 | 0.6709 / 0.0107 |
| sum of 24 isolated costs ÷ (base − all_linear) | 1.446 | 1.651 |
| smallest / largest isolated cost | 0.0228 (L11) / 0.1387 (L23) | 0.0229 (L11) / 0.1096 (L23) |
| prefix group ÷ its layers' isolated sum, k = 6 / 12 / 18 | 0.671 / 0.633 / 0.690 | 0.612 / 0.571 / 0.616 |
| suffix group ÷ its layers' isolated sum, k = 6 / 12 / 18 | 0.825 / 0.881 / 0.915 | 0.722 / 0.753 / 0.786 |

Every group costs less than the sum of its layers' isolated costs: the errors saturate through depth instead of
compounding, early-layer errors saturate most (later layers absorb them), and finishing strengthens it in all six
groups. The "cross-layer part" of the restore rows is this saturation, not growth with depth. Each layer's isolated cost
is at least `0.0229` nats, above Q4_K_M's whole-model `0.0197` (`anchor`: different evaluation sets). That makes the
isolated cost of one layer, with the rest at the teacher, a one-forward inner-loop objective for a per-layer quantizer:
the per-layer target that reaches the Q4_K_M level is roughly 30–45× below the present costs.

### Block-wise fine-tuning (prediction `qreg-blockft-1`, refuted)

Receipt [`docs/evidence/receipts/onebit/qreg-blockft.json`](../evidence/receipts/onebit/qreg-blockft.json)
(sha256 `b9da33a4…`); verdicts `blockft_refuted`, `blockft_phase_raises_kl`, `blockft_block_step_cost` in
`proof/corpus/Perf/QuantumRotation.lean`, audited in `AxiomAudit.lean` (namespace `qreg_blockft`; controls: the
reversed KL order and a block step under `3/4` of a KL step each leave `⊢ False`, exit 1; `blockLossLast` moved by one
unit in a copy, `receipt_lean.py --check` exit 1 on drift; restored, exit 0). Board row
`quantum-register-learned-LD7oW6M8EVQgAkWyMtEOs`.
`pv_finish.py --block-s S` (off by default) spends the first `S` s of the train budget on AQLM's phase 3 (Egiazarian et
al., arXiv 2401.06118, Sec. 3.4): each block fits its relative squared residual update
`‖f_l^s(h_l^t) − h_{l+1}^t‖² / ‖h_{l+1}^t − h_l^t‖²` on the teacher's hidden state `h_l^t`, no LM head; at `S` the Adam
state is cleared, the warmup restarts and the end-to-end KL phase runs to the same wall. One pair, same transfer
(step-0 quick KL `0.7954` in both), same source, n-cal 50, mm/step/fused, 1.178 b/w.

| arm | steps | train s | held-out KL `1.1637 →` | held-out top-1 |
|---|---|---|---|---|
| `--block-s 120` | 813 (296 block, 517 KL) | 402.2 | 0.8817 | 0.5404 |
| KL only | 730 | 392.9 | **0.8588** | **0.5473** |

The block phase lowers its own loss (`0.3120 → 0.2803` on the same batch, every layer's error falls) while the quick
held-out KL rises `0.7954 → 0.9635`; the KL phase then recovers most but not all of it. The Euclidean residual of each
block on teacher inputs is not aligned with the end-to-end KL: it scores each block in isolation, while the iso rows
above show the composed chain costs `0.57–0.79×` the sum of isolated costs, i.e. later layers compensate earlier
errors; fitting each block to the teacher's states alone spends parameters on error the chain already absorbs. Cost model: a block
step is `0.406` s against `0.538` s for a KL step (ratio `0.754`, it skips the LM head), so the phase buys 83 extra steps
and still loses `0.023` nats. With the iso result (errors saturate, every layer `≥ 0.0229` alone), continuous-parameter
finishing is not where the gap is; the next arm re-rounds the discrete codes under a full-model-KL Hessian.

### Arc Block (W1-T1)

```text
SCOPE:      tools/onebit/kan_probe.py (new), its receipt, Perf/QuantumRotation.lean; done when the receipt and verdict build
REGISTER:   measurement; excludes the code solve (fold_transfer), finishing, Rust
CORPUS:     rep_elliptic_is_rotation, elliptic_one_param (sanguine Apeiron/Physics/Quadratic.lean, sha256 b90770cb6a43e58b…,
            rev abcf09e3, proved); sylvester_one_channelPair (proof/corpus/Numeric/Sylvester.lean:186, proved)
REFERENCE:  /home/socol/Workspace/apeiron/tools/onebit/fold_transfer.py:440 (fwht128), :266 (gptq_e8)
GATE:       PREDICTION-admitted kan_probe.py run, cwd repo root, 580 s budget; red: MISSING (gate-first task)
GRADE:      measured; before = fixed-Hadamard code error from the same run
```

### Subsume search (recorded)

Command: `rg -l -i 'SpinQuant|OSTQuant|QuaRot|Cayley'`, `'Givens'`, `'qubit|Bloch'`, `'hypergraph state|stabilizer
state|Gottesman'` over `apeiron sanguine sanguine-ref song fold spinor-ttl intent`, and `rg -c` for
`Iwasawa|KAN decomposition`, Hadamard/butterfly theorems, `unitary|orthogonal`, `rotation` over the four Lean trees.

- No learned-rotation quantizer, Givens register or Iwasawa factorisation exists in any tree: `not-subsumable` in-tree;
  the empirical incumbents are SpinQuant and OSTQuant (external, to retrieve before W2).
- Ported, not re-derived: `Numeric/Sylvester.lean`, `Numeric/HadamardStage.lean` (`fwhtRef_getElem`),
  `Numeric/HadamardRefinement.lean` (`fwht_int_spec`, the lane/base/stage loop pattern `stage_mul` follows),
  `Nous/Model/RotationStage.lean` (`hadamardRef_isometry_l2`), sanguine `Physics/Quadratic.lean` (the three one-parameter
  faces), sanguine `Trichotomy/Derived/Elliptic.lean` (`cayley_isometry`), `src/cell.rs` `Cell::mul`,
  `src/numeric/fixed_trig.rs` `fixed_sin_cos`.
- `song` matches for Givens are Iota checker identifiers, not rotations; `sanguine-ref` qubit matches are paper prose
  and `Hopfion.lean`, no register.
- The Hopf map `ℝ⁴ → ℝ³` over `ℍ` (spinor to Bloch vector) is in the canonical corpus as the Kustaanheimo–Stiefel map:
  `../sanguine/proof/Apeiron/Physics/KustaanheimoStiefel.lean` (sanguine `abcf09e3`, last touched `ebc45479`)
  `ks_norm` (`|x| = |u|²`), points only. It is not in the in-tree `proof/canonical` pin (`880aa57e`): a consumer ports it.

## Building blocks: the Lohmiller–Slotine / Vattay exchange against the corpus

Retrieved and read, `/home/socol/oc-tmp/refs/quantum-classical/` (PDF and `pdftotext -layout` text):

| arXiv | Paper | sha256 (PDF) |
|---|---|---|
| 2405.06328v11 | Lohmiller, Slotine, *On computing quantum waves exactly from classical and relativistic action* | `365a99fb…9de9e3` |
| 2605.02621v1 | Vattay, *Comment* | `78e623b2…40e527` |
| 2605.20443v5 | Lohmiller, Slotine, *Reply to second comment* (with extended Lemma 3.1) | `b61c1113…6ec2` |
| 2606.05197v1 | Vattay, *Reply* | `5a578ea0…bfac` |

What the exchange settles, with both sides' own words:

- Substituting `ψ = √ρ e^{iφ/ℏ}` into Schrödinger gives continuity plus Hamilton–Jacobi with the Madelung–Bohm term
  `Q = −(ℏ²/2M) ∇²√ρ / √ρ` (Vattay 2605.02621 eq. 2.3–2.6). The extended Lemma 3.1 of 2605.20443 concedes this and
  restates the claim as conditional: the classical action is exact on a branch when `∇²√ρ_j = 0` on that branch.
- Lohmiller–Slotine: `∇²√ρ_j = 0` holds when `Δφ_j` is independent of position, which covers the Pauli, Dirac and
  Maxwell derivations and every example except hydrogen; hydrogen needs a Duru–Kleinert time rescaling `t′(x, t)`.
- Vattay 2606.05197: in physical coordinates `∇²R(t′(x,t)) = R″(∇t′)² + R′∇²t′`, which does not vanish, so the
  rescaled construction is the Van Vleck–Pauli–Morette propagator, exact if and only if the Hamiltonian is at most
  quadratic. Lohmiller–Slotine answer that the condition is on a Laplacian (a scalar), not the Van Vleck Hessian.
- Spin (2405.06328 §4.2, eq. 36–39): the action is a pure-imaginary quaternion `ℏ s γ/2 · Σ·n`, the wave
  `e^{iΣ·n sγ/2}` is a unit quaternion acting on the Bloch sphere, `Σ·n = χ↑χ↑† − χ↓χ↓†`. Here `Δφ = 0`, `ρ` is constant
  along the path, `Q = 0` exactly: no dispute covers this case.

**The contribution here: four branches.** Vattay's objection is that one classical branch cannot supply `∇²√ρ`, so
Lohmiller–Slotine need `∇²√ρ = 0`. Sample four branches and `Q` is computed, not assumed zero; the harmonic condition
and the time-rescaling term both drop out of the argument, because the stencil samples the amplitude in physical
coordinates and so includes `R″|∇t′|² + R′∇²t′` whatever `t′` is.
`Substrate/Algebra/QuantumField/QuantumClassicalIsomorphism.lean` (vendored from sanguine):

- proved: `single_branch_underdetermines`, `¬ ∃ F, F (jet1 (quad …)) = p + q`. This is Vattay's obstruction: the
  1-jet of one branch does not determine the Laplacian.
- proved: `stencil4_reconstructs_laplacian`. The four branches `(x ± h, y)`, `(x, y ± h)`, which are the orbit
  `{±1, ±i}` (`stencil_offsets_are_the_orbit`), give `∇²R = p + q` exactly for every `h ≠ 0` on a quadratic amplitude.
- open, in the file's own list: exactness beyond the quadratic amplitude (the `h → 0` limit), minimality, and the
  joint theorem with block 1.

Proved in `proof/corpus/Numeric/BranchLaplacian.lean` (controls: centre weight `4 → 3`, divisor `2h² → h²`, bound `4 → 5`, each red). In ℝ³, where the hydrogen case lives, four branches at
the vertices of a regular tetrahedron `v ∈ {(1,1,1), (1,−1,−1), (−1,1,−1), (−1,−1,1)}` give
`Σₖ f(x + h vₖ) − 4 f(x) = 2h² ∇²f` exactly on quadratics, since `Σ vₖ = 0` and `Σ vₖ vₖᵀ = 4I`. Four is also the
fewest. A weighted stencil needs `Σ wₖ vₖ = 0` and `Σ wₖ vₖ vₖᵀ = cI` with `c ≠ 0`. The second forces three linearly
independent `vₖ`; with exactly three, the first forces `w = 0`. So `k ≥ d + 1 = 4`. In ℝ³, then, `K = 4` is the
minimum branch count that reconstructs `Q` exactly on quadratic amplitudes, with no antipodal-symmetry premise.

Consequence for the register: a `2^n`-amplitude register has no spatial Laplacian, so `Q` never arises; its dynamics
`iψ′ = Hψ` is the finite-dimensional unitary flow, inside the exactness window on both sides' accounts. The quantum
computation primitives are gates, tensor structure, entangling stages and measurement, not the propagator question.

Each block, mapped to the corpus (`canonical` = in-tree `proof/canonical`, pin `880aa57e`; `sanguine` = `../sanguine/proof`, rev `abcf09e3`):

| # | Block | Paper | Corpus declaration | Grade |
|---|---|---|---|---|
| 1 | Schrödinger ⟺ HJ + `Q` and continuity, 1-D, `ρ > 0` | Vattay 2.1–2.5; L–S Lemma 3.1 | canonical `Physics/Models/Schrodinger.lean` `actual_schrodinger_iff_madelung` | proved |
| 2 | `Q = −(ℏ²/2M) ∇²√ρ/√ρ`; the gap to classical HJ is `R″/R` | Vattay 2.6 | canonical `quantumPotential_eq_actual_laplacian_sqrt_density`; `Substrate/Algebra/QuantumField/QuantumClassicalIsomorphism.lean` `quantum_potential_is_the_gap` | proved |
| 3 | harmonic amplitude (`∇²√ρ = 0`) ⇒ Schrödinger ⟺ classical HJ + continuity | L–S reply Lemma 3.1 | superseded by 3a–3d: with four branches the premise is not needed | — |
| 3a | one branch's 1-jet does not determine `∇²R` | Vattay 2605.02621 | `QuantumClassicalIsomorphism.lean` `single_branch_underdetermines` | proved |
| 3b | four branches `{±1, ±i}` reconstruct `∇²R` exactly on a quadratic amplitude, 2-D | — (ours) | `QuantumClassicalIsomorphism.lean` `stencil4_reconstructs_laplacian` | proved |
| 3c | four tetrahedral branches reconstruct `∇²R` exactly on a quadratic amplitude in ℝ³; `k ≥ d + 1` | — (ours) | `Numeric/BranchLaplacian.lean` `tetra_stencil_reconstructs_laplacian`, `branch_count_ge_four`, `tetra_family_attains_four` | proved |
| 3d | four-branch stencil → `∇²R` as `h → 0` for `R` twice differentiable (beyond quadratics) | — (ours) | `Numeric/BranchLaplacian.lean` `secondDiff_tendsto`, `stencil4_tendsto_laplacian` (limit, no rate); `Numeric/TetraLimit.lean` `tetra4_tendsto_field`: the four-branch tetrahedral stencil of any `C²` field on `ℝ³` (no quadratic premise) tends to the mean of the second derivatives along the four branch lines; `Numeric/TetraLaplacian.lean` `tetra_sum_eq_lap`, `tetra4_tendsto_laplacian`: that mean is `∂xx+∂yy+∂zz`, because `Σₙ vₙ vₙᵀ = 4·1` gives `Σₙ D²F(vₙ, vₙ) = 4 tr D²F` for any bilinear `D²F`, symmetric or not (`line_second_deriv`: `(t ↦ F(p+tv))″(0) = D²F(p)(v, v)`) | proved |
| 4 | time rescaling leaves `R″(t′)·|∇t′|² + R′·∇²t′` in `∇²R(t′(x))`, and the four-branch stencil of `R ∘ t′` returns it | Vattay 2606.05197 eq. 1–2 | `Numeric/BranchLaplacian.lean` `second_derivative_comp`, `stencil4_quad_comp_affine`, `stencil4_affine_comp_quad` (each term read exactly) | proved |
| 5 | Van Vleck kernel exact on the three quadratic faces | Vattay 2606.05197 §3 | canonical `VanVleck/{Elliptic,Parabolic,Hyperbolic}.lean` `*_vanVleck_pointwise_exact` (pointwise residual, not propagator) | proved |
| 6 | Van Vleck exact **only if** the Hamiltonian is at most quadratic | Vattay 2606.05197 §3 (cited theorem) | absent; cited, not proved | anchor |
| 7 | Duru–Kleinert / Kustaanheimo–Stiefel flow equivalence under `dt = |x| dτ` | L–S Example 6, reply Annex | sanguine `Physics/KustaanheimoStiefel.lean` states it is not claimed (points only) | open |
| 8 | quadratic flow is the face's one-parameter subgroup | L–S Thm 1 on quadratic `H` | canonical `Physics/Quadratic.lean` `flow_is_one_param`, `flow_is_one_param_unique` | proved |
| 9 | density along a path `ρ(t) = ρ₀ e^{−∫Δφ}` | L–S eq. 12 | `Numeric/PathDensity.lean` `pathDensity_hasDerivAt`, `pathDensity_zero`, `pathDensity_unique` (any `C¹` solution of `ρ' = −ρa`, `a` continuous, equals the closed form), `pathDensity_pos`; `a = Δφ/m` along a Hamiltonian path not linked | proved |
| 10 | quantization: `Σ_{κ<K} e^{iκφ/ℏ}/K → 0` unless `φ/ℏ ∈ 2πℤ` | L–S Lemma 3 | `Numeric/BranchQuantization.lean` `branchAvg_tendsto_zero`, `branchAvg_tendsto_one`, `norm_branchSum_le` (`‖Σ‖ ≤ 2/‖e^{iθ}−1‖`), `branchSum_eq_zero_iff` (finite `K`: zero exactly at nontrivial `K`-th roots), `exp_eq_one_iff_mem`, `branchSum_four_quarter` (four branches at `θ = π/2` cancel) | proved |
| 1+3 | `Q` read from the branches, joined to block 1: the 1-D two-branch `Q_h` tends to Bohm's `Q`, and the canonical Schrödinger ⟺ Madelung iff holds with that limit | Vattay 2.1–2.5; L–S Lemma 3.1 | `Numeric/BranchMadelung.lean` `branchQ_tendsto`, `schrodinger_iff_branch_madelung` (1-D; the corpus has no 3-D Madelung) | proved |
| 11 | spinor ↦ Bloch vector, `|n| = |ψ|²` | L–S Fig. 4, eq. 39 | sanguine `Physics/KustaanheimoStiefel.lean` `ks_norm`; in-tree port `Numeric/Qubit.lean` `bloch_norm`, `bloch_eq_ks` (Bloch vector = KS map of `(Re α, Im β, Re β, Im α)`; KS defs copied from sanguine lines 53–71) | proved |
| 12 | SU(2) gate on the spinor = SO(3) rotation of the Bloch vector (Hopf equivariance) | L–S eq. 36–37 | `Numeric/Qubit.lean` `bloch_rz`, `bloch_ry`, `suR_blochVec` (any `U(a,b) = [[a,−b̄],[b,ā]]`), `suR_orthogonal` (`|a|²+|b|² = 1`) | proved |
| 13 | `Σ·n = χ↑χ↑† − χ↓χ↓†`, `(Σ·n)² = 1`; real meridian `[[c,s],[s,−c]] = c·σ_z + s·σ_x` (the `kan_probe` butterfly) | L–S eq. 38 | `Numeric/Qubit.lean` `sigmaN_sq`, `sigmaN_sq_unit`, `projP_idem`, `projP_add_projM`, `projP_mul_projM`, `sigmaN_meridian`, `sigmaN_meridian_pi_four` | proved |
| 14 | unitary step conserves total Born density | L–S eq. 18 | `Substrate/StationaryPhase/SpinorLoop.lean` `cayley_isometry`, `cayley_conserves_born_density`, `schrodinger_generator_skew` | proved |
| 15 | single-qubit gate on every pair of a `2^n` register; `H^⊗n` | L–S eq. 53 (product state) | `Numeric/Register.lean` `pairStage`, `register_hadamard`, `register_stage_normForm` | proved |
| 16 | entangling two-qubit stage (controlled-Z as a sign diagonal) | L–S Example 7 (EPR) | `Numeric/Register.lean` `cz_involutive`, `cz_sum_sq`, `cz_entangles` (`y₀y₃ ≠ y₁y₂` on the all-ones register; every product state has equality) | proved |
| 17 | measurement: qubit projectors idempotent, sum to identity, Born weights sum to `‖x‖²` | L–S Lemma 2 (Born rule a postulate there) | `Numeric/Register.lean` `measProj_idem`, `measProj_add`, `measProj_orth`, `born_sum` | proved |
| 18 | reflection stage is an involution, so `Tᵀ` is the stages in reverse order | — (W2 inverse for `fold_transfer`) | `Numeric/Register.lean` `reflPair_involutive`, `reflStage_involutive`, `reflStage_sum_sq`, `reflStages_reverse_cancel`, `refl_quarter` | proved |

Landed: 3c, 3d, 4 (`Numeric/BranchLaplacian.lean`, `Numeric/TetraLimit.lean`, `Numeric/TetraLaplacian.lean`), 1+3 (`Numeric/BranchMadelung.lean`), 9 (`Numeric/PathDensity.lean`), 10 (`Numeric/BranchQuantization.lean`), 11–13 (`Numeric/Qubit.lean`), 16–18 (`Numeric/Register.lean`), each
with three changed-constant controls red at the edited line and the real file green.

Every row above is registered in `proof/corpus/AxiomAudit.lean` (closed cone).

Open: a 3-D Madelung (the canonical one is 1-D), so the joint row is 1-D; a rate for 3d; the link `a = Δφ/m` for row 9. Row 6 stays `anchor` (cited, not proved); 7 is the step 3c and 3d bypass, not one they need.
