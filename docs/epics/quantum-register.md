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
| `iwasawa_faces` | `det M = 1 → ∃ t a v, M = rep (−1) (rot t) · (H · diag(a, a⁻¹) · H / 2) · (rep 0 (1, v))ᵀ` with `a > 0`; the middle factor is `rep 1 (cosh s, sinh s)` for `a = e^s` | proved |
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
| W3 | T5 finish on learned `T`, then train N (the corrector) | `tools/onebit/pv_finish.py`, receipts | lead | held-out KL vs 0.8484 | as `hadmm` controls |

Stop: W1 refutes (no error drop beyond the seed spread at fixed bits) → record the verdict theorem, keep T2/T4 (the
primitive stands on its own), and move K to the dense Cayley route (`cayley_isometry`) as the next row.

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
