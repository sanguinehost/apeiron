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
| W3 | T5 finish on learned `T`, then train N (the corrector) | `tools/onebit/pv_finish.py`, receipts | lead | held-out KL vs 0.8484 | as `hadmm` controls |

Stop: W1 refutes (no error drop beyond the seed spread at fixed bits) → record the verdict theorem, keep T2/T4 (the
primitive stands on its own), and move K to the dense Cayley route (`cayley_isometry`) as the next row.

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
| 3d | four-branch stencil → `∇²R` as `h → 0` for `R` twice differentiable (beyond quadratics) | — (ours) | `Numeric/BranchLaplacian.lean` `secondDiff_tendsto`, `stencil4_tendsto_laplacian` (limit, no rate); `Numeric/TetraLimit.lean` `tetra4_tendsto_field`: the four-branch tetrahedral stencil of any `C²` field on `ℝ³` (no quadratic premise) tends to the mean of the second derivatives along the four branch lines; identifying that mean with `(∂xx+∂yy+∂zz)/3` needs the symmetric Hessian (open) | proved |
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

Landed: 3c, 3d, 4 (`Numeric/BranchLaplacian.lean`, `Numeric/TetraLimit.lean`), 1+3 (`Numeric/BranchMadelung.lean`), 9 (`Numeric/PathDensity.lean`), 10 (`Numeric/BranchQuantization.lean`), 11–13 (`Numeric/Qubit.lean`), 16–18 (`Numeric/Register.lean`), each
with three changed-constant controls red at the edited line and the real file green.

Every row above is registered in `proof/corpus/AxiomAudit.lean` (closed cone).

Open: a 3-D Madelung (the canonical one is 1-D), so the joint row is 1-D; the symmetric-Hessian step that turns the
tetra limit into `∇²/3`; a rate for 3d; the link `a = Δφ/m` for row 9. Row 6 stays `anchor` (cited, not proved); 7 is the step 3c and 3d bypass, not one they need.
