# Spinor coverage and transport

Merged epic record: `plan/spinor-coverage.md`, `plan/spinor-transport.md` (one file per epic; see `docs/README.md`).

## Spinor Coverage - Second Epic Lifecycle

- **Owner:** deepseek-flash (SANGUINE-SPINORCOVERAGE-W0-T1, documentation).
- **Status:** E1, E2, C1, N1, M1 and C2 are proved at the canonical gate and integrated uncommitted
  into the canonical sibling. E1/E2/C2 file `proof/Apeiron/Trichotomy/Derived/Elliptic.lean`,
  sha256 `23d591d29095ce2de54ac4027d9e91ac73c88bfae89e26834aaa67cf54cb1535`, 644 lines; C1/M1
  file `proof/Apeiron/Trichotomy/AnyRing.lean`, sha256
  `decbba9dc965bbf94f83beb982d468e27ddee69c024ecf923740a30217445428`; M1 file
  `proof/Apeiron/Physics/Quadratic.lean`, sha256
  `b90770cb6a43e58b6a92cb3b618190eee389dfebafd1e23928d343866da1848a`; N1 file
  `proof/Apeiron/Numeric/Accumulator.lean`, sha256
  `ffe878895360ba6549a434d41144492c5acbef7a2cb57b7753d06dba938a32cf`. M2 and the additive-route
  row remain deferred with the reasons below. The canonical tree is dirty; nothing is committed.
  Epic-level acceptance still requires a commit decision by the lead.
- **Self-contained:** Intent, Specification, Plan, Evidence, Reference register, Non-claims.

### Source identity

- Consuming tree `/home/socol/Workspace/apeiron`, corpus pin `88866412`, unchanged. No Rust,
  Aeneas, Slang, Song or Fold change.
- Live canonical proof source `/home/socol/Workspace/sanguine-apeiron`, `336ce8d3`, dirty. Old
  read-only reference `/home/socol/Workspace/sanguine`, `e830d3f`.
- Lean `leanprover/lean4:v4.31.0`, commit `68218e876d2a38b1985b8590fff244a83c321783`; Mathlib
  `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`.
- E1/E2/C2 file `proof/Apeiron/Trichotomy/Derived/Elliptic.lean`, sha256
  `23d591d29095ce2de54ac4027d9e91ac73c88bfae89e26834aaa67cf54cb1535`, 644 lines; E1/E2
  `ellipticEquiv` and its laws, C2 `skew_re_zero`, `cayley_isometry_of_re_zero`, `cayley_isometry`.
- C1/M1 file `proof/Apeiron/Trichotomy/AnyRing.lean`, sha256
  `decbba9dc965bbf94f83beb982d468e27ddee69c024ecf923740a30217445428`; C1 new
  `ringNormForm_conj`, `ringCayley_normForm`; M1 new `MatrixAffineStep`, `matrixAffine`,
  `matrixStepComp`, `matrixStepId`, `matrixStepFold` and their laws.
- M1 file `proof/Apeiron/Physics/Quadratic.lean`, sha256
  `b90770cb6a43e58b6a92cb3b618190eee389dfebafd1e23928d343866da1848a`; new `ringRep`, `cellVec`,
  `toMatrixStep`, `ringRep_mul`, `matrixStep_apply`, `matrixStep_comp`, `ringRep_real_eq_rep`;
  `Apeiron.Physics.rep` (`Physics/Quadratic.lean:75`) unchanged.
- N1 file `proof/Apeiron/Numeric/Accumulator.lean`, sha256
  `ffe878895360ba6549a434d41144492c5acbef7a2cb57b7753d06dba938a32cf`; new `osFusePair`, `osDen`,
  `osNum`, `osTile`, `osScan`, `osDen_fuse`, `osNum_fuse`, `osDen_scan_acc`, `osNum_scan_acc`,
  `softmax_attention_is_online_chart`.
- Integrated audit `proof/Apeiron/Audit.lean`, sha256
  `abacd1a6f1bbe1e8cc40c07035b36c8cbd3fee62d383a6414ac1a2bc63375543` (was
  `f7fe3b05d632d6751c8ff61e62187e3a28274a05d389d4914b3d57ac4ba9c55c` after C1/N1); +22 headlines
  (12 AnyRing matrix, 7 Physics, 3 Trichotomy Cayley).
- Barrier on this dirty state: `lake build` exit 0 at 3522 jobs; `check-lints`, `check-style`
  (181 files) and `check-imports` exit 0; E1/E2 headlines audit inside
  `{propext, Classical.choice, Quot.sound}` (11 `#audit` targets); `ringNormForm_conj` and
  `ringCayley_normForm` audit inside `{propext, Quot.sound}`; the five Numeric headlines and all 22
  M1/C2 headlines inside `{propext, Classical.choice, Quot.sound}`. Nothing committed.
- The pin carries `Physics/Quadratic.lean`, `Trichotomy/Flow.lean`,
  `Trichotomy/unit_loop_isometric` and `cstarNormNeg` in `Trichotomy/CStar.lean`. The live
  sibling moved the C* material to `Trichotomy/Derived/CStar.lean`; the pin has no
  `Trichotomy/Derived/`. Every name cited below resolves in the live sibling.

### Intent

**Problem.** The first epic (`docs/epics/spinor.md`) ported the canonical analogue of
the sigma-free real scalar affine loop and left the sigma < 0 elliptic complex rotor, the
complex isometries, the matrix action and the term-algebra initiality outside it. Its ledger
named elliptic representation as the row carrying the old spinor rotation content. That row is
this epic.

**Minimal scope.** Canonical-only coverage. Add the representation link between `Cell` at
`σ < 0` and `Complex`, then read the existing elliptic declarations through it. Do not duplicate
`Physics.Quadratic`, `Trichotomy.Flow`, `Trichotomy.Derived.CStar` or `Trichotomy.Trace`.

**Invariants.** (1) `σ` is explicit in every executing signature. (2) `normForm` stays the
indefinite algebraic form; only at `σ < 0` does `cstarNormNeg` supply a positive definite norm.
(3) One authority: the canonical declarations are cited, not restated. (4) The cone is
`{propext, Classical.choice, Quot.sound}`; the empty Slang cone is not claimed. (5) No O(1),
backpropagation, performance or source-binding claim. (6) No corpus move, pin, extraction,
Rust, Aeneas, Kani or source-binding edit; no all-corpus closure; no consumer uptake.

**Trust boundary.** Evidence reaches the Lean declarations below and their `#audit` cones, not
the old repository's build, rustc/LLVM, the kernel ABI, GPU/SPIR-V or any executing consumer.
The old sources are read references, not extracted evidence.

### Scope boundary: already canonical, not to be duplicated

| Declaration | File | Statement |
|---|---|---|
| `rot` | `Physics/Quadratic.lean` | size-one point `(cos t, sin t)` at `σ = -1` |
| `elliptic_one_param` | `Physics/Quadratic.lean` | `mul (-1)` adds the angle on the circle |
| `unitNorm_elliptic` | `Physics/Quadratic.lean` | `normForm (-1) (cos t, sin t) = 1` |
| `elliptic_rotors_commute` | `Physics/Quadratic.lean` | pure rotors commute |
| `flow` | `Trichotomy/Flow.lean` | one size-one flow; `σ` selects the face |
| `flow_neg` | `Trichotomy/Flow.lean` | below zero the flow is the circle |
| `normForm_flow` | `Trichotomy/Flow.lean` | `normForm σ (flow σ t) = 1` |
| `flow_add` | `Trichotomy/Flow.lean` | `mul σ (flow σ s) (flow σ t) = flow σ (s + t)` |
| `cstarNormNeg` | `Trichotomy/Derived/CStar.lean` | `sqrt (normForm σ z)`, positive definite |
| `unit_loop_isometric` | `Trichotomy/Trace.lean` | `normForm σ d = 1` survives `cpow σ d n` |

The gap is the representation link: no canonical declaration maps `Cell` at `σ < 0` onto
`Complex` and carries `ringMul`, `ringNormForm`, `ringConj`, `flow` and `ringAffineFold` across it.

### Specification

#### Wave 1 (E1): the elliptic representation, integrated

File `proof/Apeiron/Trichotomy/Derived/Elliptic.lean`, namespace `Apeiron.Trichotomy` (module
`Apeiron.Trichotomy.Derived.Elliptic`). The map `z ↦ ⟨z.1, Real.sqrt (-σ) * z.2⟩` is real-linear
and invertible exactly when `σ < 0`.

```lean
def ellipticEquiv (σ : ℝ) (hσ : σ < 0) : Cell ≃ₗ[ℝ] ℂ
theorem ellipticEquiv_mul   -- multiplicative: ringMul maps to complex multiplication
theorem ellipticEquiv_normSq -- Complex.normSq (ellipticEquiv ...) = normForm σ z
theorem ellipticEquiv_conj  -- ringConj maps to Complex.conjugate
theorem ellipticEquiv_flow  -- flow σ t maps to the complex exponential at parameter t
theorem ellipticEquiv_affineFold -- chronological ringAffineFold maps to the complex fold
theorem ellipticEquiv_neg_four_witness
theorem elliptic_quarter_turn_order_witness
```

The `σ = -1` coverage specialization of the old `AttentionSpinor.spinApply` quarter-turn order
example over `ringAffineApply`: with `b₁ = (1, 0)` and `b₂ = (0, 0)`, the two orders differ.
Integrated uncommitted; grade proved at the canonical gate below.

#### Wave 2 (E2): the damped elliptic fold preserves an inhabited ball, integrated

The canonical analogue of the old `TrainingCollapse.rotor_state_bounded`, which proves for
`ψ' = ρ·e^{iθ}·ψ + (1 - ρ)·u` with `0 ≤ ρ ≤ 1`, `‖ψ‖ ≤ M` and `‖u‖ ≤ M`, that `‖ψ'‖ ≤ M`.

Actual: `dampedStep σ p = ⟨p.1 • flow σ p.2.1, (1 - p.1) • p.2.2⟩`, and `damped_fold_bounded`
takes `(σ : ℝ) (hσ : σ < 0) (M : ℝ) (z : Cell) (hz : cstarNormNeg σ z ≤ M)`
`(ps : List (ℝ × ℝ × Cell))` `(hps : ∀ p ∈ ps, 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ cstarNormNeg σ p.2.2 ≤ M)`
and returns
`cstarNormNeg σ (ringAffineApply σ (ringAffineFold σ (ps.map (dampedStep σ))) z) ≤ M`.
`damped_step_joint_witness` evaluates one step at `σ = -4`, weight `1/2`, time `π/4`, injection
`(1, 0)`: the state `(1, 0)` maps to `(1/2, 1/4)`, `normForm (-4) (1/2, 1/4) = 1/2`, and its
elliptic size is at most one. Integrated uncommitted; grade proved at the canonical gate below.
The negative control (mutate `1 - ρ` to `1 + ρ`) is not recorded in this handoff.

#### Wave 3 (C1): scalar Cayley factor over `AnyRing`, integrated

File `proof/Apeiron/Trichotomy/AnyRing.lean`, namespace `Apeiron.Trichotomy.AnyRing`. New
`ringNormForm_conj` (conjugation leaves the norm form alone) and `ringCayley_normForm`: for
`σ θ : R` and `w : RingCell R`, a carried inverse witness `hw : ringMul σ (1, θ) w = ringOne`
gives `ringNormForm σ (ringMul σ (ringConj (1, θ)) w) = 1`. A general `CommRing` is not a field, so
division is carried, not total. The carried-inverse form is the implemented statement; the choice
versus a division-free cross-multiplied variant is an unconfirmed spec choice, not proved here.
Integrated uncommitted; grade proved at the canonical gate below.

#### Wave 4 (N1): the online numerator/denominator chart, integrated

File `proof/Apeiron/Numeric/Accumulator.lean`, namespace `Apeiron.Numeric`. New `osFusePair`,
`osDen`, `osNum`, `osTile`, `osScan`, `osDen_fuse`, `osNum_fuse`, `osDen_scan_acc`,
`osNum_scan_acc` and the headline `softmax_attention_is_online_chart`: for `l : List (ℝ × ℝ)`,
`osNum (osScan l)` is the value-weighted exponential sum and `osDen (osScan l)` the exponential
sum, and both channels merge as exact `+`. This transports the old three-conjunct
`softmax_attention_is_online_spinor` at `/home/socol/Workspace/sanguine` rev `e830d3fb`,
`proof/Substrate/Algebra/Neural/AttentionSpinor.lean:403-469`, not the doc's paraphrase. The chart
is not `Trichotomy.Cell`; no `ringMul` or `ringNormForm` is read. Integrated uncommitted; grade
proved at the canonical gate below.

#### Wave 5 (M1): the matrix/module affine action, integrated

File `proof/Apeiron/Trichotomy/AnyRing.lean` adds `MatrixAffineStep`, `matrixAffine`,
`matrixStepComp`, `matrixStep_apply`, `matrixStep_comp`, `matrixStepId`, `matrixStepId_apply`,
`matrixStepId_comp`, `matrixStepComp_id`, `matrixStepFold`, `matrixStepFold_apply` and
`matrixStepFold_append`, all under `[Fintype ι] [DecidableEq ι] [Semiring R]`. File
`proof/Apeiron/Physics/Quadratic.lean` adds `ringRep`, `cellVec`, `toMatrixStep`, `ringRep_mul`,
`matrixStep_apply`, `matrixStep_comp` and `ringRep_real_eq_rep`; `Apeiron.Physics.rep`
(`Physics/Quadratic.lean:75`) is unchanged and `ringRep_real_eq_rep` is `rfl`. Only AnyRing adds
the import `Mathlib.Data.Matrix.Mul`. Integrated uncommitted; grade proved at the canonical gate.

#### Wave 6 (C2): the general complex-linear Cayley isometry, integrated

File `proof/Apeiron/Trichotomy/Derived/Elliptic.lean` (644 lines) adds `skew_re_zero`,
`cayley_isometry_of_re_zero` and `cayley_isometry` over an arbitrary complex inner-product space.
The implication `φ + Aφ = ψ - Aψ → ‖φ‖ = ‖ψ‖` needs neither `(1 + A)` invertible nor finite
dimension; the sharp local premise is `RCLike.re ⟪A (φ + ψ), φ + ψ⟫ = 0`, which `skew_re_zero`
discharges for skew-adjoint `A`. This replaces the earlier "may need invertibility or finite
dimension" note with Sol's verdict. The Cayley theorems carry no `@cite`: `Linter/Citations.lean`
has no Cayley or spinor key. Integrated uncommitted; grade proved at the canonical gate below.

### Plan

#### Barrier and gate commands

Every Lean gate is serialized and bounded, with no `maxHeartbeats 0`, no raised `maxRecDepth`
and no whole-source `decide`. Commands run individually from the stated cwd.

```sh
# cwd /home/socol/Workspace/sanguine-apeiron/proof
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope \
  -p MemoryMax=6G -p MemorySwapMax=0 -- env LEAN_NUM_THREADS=1 timeout 1800 \
  lake build <module> Apeiron.Audit
# cwd /home/socol/Workspace/sanguine-apeiron
scripts/check-lints.py
python3 scripts/check-style.py
python3 scripts/check-imports.py
```

`<module>` is `Apeiron.Trichotomy.Derived.Elliptic` for E1. Each new headline declaration needs
an `#audit` line in `Audit.lean`; a `sorry` is a build failure. The barrier runs all four gates
on the integrated state; no E1/E2 mutation control is recorded in this handoff. Rust, Kani, Aeneas
and Slang gates are outside. Wave discipline: a concurrent task pair must not share a dependency
module under edit, because the lane lock serializes builds but not edits and a build can observe a
mid-edit red state; observed once, C2's first build saw a transient AnyRing.lean red while M1 was
mid-edit, and the retry was green.

#### Task blocks

```text
SCOPE:      E1 add the elliptic representation and its laws to a new Derived/Elliptic.lean.
REGISTER:   local algebra/refinement (representation relation Cell at σ < 0 to ℂ).
CORPUS:     AnyRing ringMul, ringNormForm, ringConj, ringAffineFold; Quadratic rot,
            elliptic_one_param, unitNorm_elliptic; Flow flow, flow_neg, normForm_flow, flow_add.
REFERENCE:  .../Neural/AttentionSpinor.lean at e830d3f; spinApply, spinor_scan_is_order_sensitive.
GATE:       cd proof && lake build Apeiron.Trichotomy.Derived.Elliptic Apeiron.Audit; exit 0.
GRADE:      proved; integrated uncommitted at the barrier gate; namespace Apeiron.Trichotomy.
SCOPE:      E2 damped elliptic fold preserves an inhabited ball.
REGISTER:   local state obligation (ball preservation under a damped elliptic fold).
CORPUS:     Derived/CStar cstarNormNeg; Flow flow; AnyRing ringAffineApply.
REFERENCE:  .../Neural/TrainingCollapse.lean at e830d3f; rotor_state_bounded.
GATE:       cd proof && lake build Apeiron.Trichotomy.Derived.Elliptic Apeiron.Audit; exit 0.
GRADE:      proved; integrated uncommitted; witness σ = -4, weight 1/2, time π/4, injection
            (1, 0); control 1 - ρ to 1 + ρ not recorded.
SCOPE:      C1 add the scalar Cayley norm-form law to AnyRing; gate exits 0.
REGISTER:   local algebra obligation (norm form of the scalar Cayley factor is one).
CORPUS:     AnyRing ringMul_conj, ringNormForm_mul, ringConj, ringOne.
REFERENCE:  NONE - inventing (the canonical scalar Cayley norm form over AnyRing).
GATE:       cd proof && lake build Apeiron.Trichotomy.AnyRing Apeiron.Audit; exit 0.
GRADE:      proved; carried inverse witness hw; control conclusion `= 1` to `= 2` red, restored.
SCOPE:      N1 add the online numerator/denominator chart to Numeric/Accumulator; gate exits 0.
REGISTER:   local algebra transport (pair chart; numerator and denominator merge as exact +).
CORPUS:     Numeric osFuse, osVal, osFuse_is_add, osFuse_fold_eq_add.
REFERENCE:  /home/socol/Workspace/sanguine rev e830d3fb,
            proof/Substrate/Algebra/Neural/AttentionSpinor.lean:403-469.
GATE:       cd proof && lake build Apeiron.Numeric.Accumulator Apeiron.Audit; exit 0.
GRADE:      proved; control numerator channel swapped to the denominator channel red, restored.
SCOPE:      M1 add the matrix/module affine action and the real representation link.
REGISTER:   local algebra transport (a ring step read as a grid and vector step).
CORPUS:     AnyRing ringMul, ringAffineApply, ringAffineCompose, ringAffineFold; Physics rep.
REFERENCE:  .../StationaryPhase/SpinorLoop.lean at e830d3f; AffineStepN.apply, composeN_assoc.
GATE:       cd proof && lake build Apeiron.Trichotomy.AnyRing Apeiron.Physics.Quadratic
            Apeiron.Audit; exit 0.
GRADE:      proved; control dropped `+ outer.injection` and swapped `cellVec` slots, red, restored.
SCOPE:      C2 add the general complex-linear Cayley isometry to Derived/Elliptic.
REGISTER:   local algebra obligation (length preservation under a Cayley step).
CORPUS:     Derived/Elliptic skew_re_zero; RCLike inner product; norm.
REFERENCE:  .../StationaryPhase/SpinorLoop.lean at e830d3f; skew_re_zero, cayley_isometry.
GATE:       cd proof && lake build Apeiron.Trichotomy.Derived.Elliptic Apeiron.Audit; exit 0.
GRADE:      proved; sharp local premise RCLike.re ⟪A (φ + ψ), φ + ψ⟫ = 0, no invertibility or
            finite dimension; control dropped the hskew binder, red, restored.
```

#### Deferred rows, with the exact reason

- **M2 `ComposeExpr` naturality.** Dropped: it is a σ-free duplicate of Mathlib's
  `AffineMap.comp_assoc` (`Mathlib/LinearAlgebra/AffineSpace/AffineMap.lean:395`) unless a
  consumer names it.
- **Additive GoI route and Aitken on path sums.** Out of scope. `Combinator/RouteAnyRing.lean`
  defines `Route R` as a partial injection `ℕ → Option (ℕ × RingCell R)`: nonbranching, with no
  route addition. A path-sum Aitken needs its own epic.

### Old readings that must not be restated

- **Exact O(1) attention for arbitrary queries.** `softmax_attention_is_online_spinor` proves,
  for a `List (ℝ × ℝ)`, `oaNum (oaScan l) = Σ e^{sc}·v` and `oaDen (oaScan l) = Σ e^{sc}`; the
  canonical `softmax_attention_is_online_chart` transports that declaration, not the paraphrase.
  It does not prove that a query produces those scores, that the normalized read equals softmax
  attention for arbitrary queries, or any runtime cost.
- **Rotor adjoint as backpropagation.** `rotor_adjoint_forward` proves
  `conj (exp (θ·i)) = (exp (θ·i))⁻¹`. It does not prove a backward pass or a gradient exists.
- **`newton_step_lands` as a complete forward-only trainer.** It proves, for `quadLoss f0 g c x`
  with `c > 0`, that `x - (g + c·x)/c` minimizes it over `y` in one dimension. It does not prove
  a multi-dimensional optimizer, that the loss is quadratic, that gradient and curvature are
  available, or that a training run converges.
- **Cayley as exact Schrödinger evolution.** `cayley_isometry` proves `‖φ‖ = ‖ψ‖` for
  skew-adjoint `A` and `φ + Aφ = ψ - Aψ`; `cayley_conserves_born_density` sums it. Neither
  proves the Cayley step equals `e^{-iHt}`, that the generator is a physical Hamiltonian, or
  that time evolution is exact.
- **Matrix sizes as Pauli/Dirac tiers.** `SpinorLoop`'s `AffineStepN` is parametric in `n` with
  `composeN_assoc`; only the Tier-1 scalar case is formalised. It does not prove that 2×2 is the
  Pauli/weak-force tier and 4×4 the Dirac/Spin(1,3) tier; the old file labels that an anchor.

### Evidence

The canonical tree is barrier-green on the dirty state: `lake build` exit 0 at 3522 jobs;
`check-lints`, `check-style` (181 files) and `check-imports` exit 0; the E1/E2 headlines audit
inside `{propext, Classical.choice, Quot.sound}` (11 `#audit` targets), `ringNormForm_conj` and
`ringCayley_normForm` inside `{propext, Quot.sound}`, and the five N1 headlines plus all 22 M1/C2
headlines inside `{propext, Classical.choice, Quot.sound}`. Nothing is committed.

| Row | Command (cwd proof) | Result |
|---|---|---|
| E1 | `lake build Apeiron.Trichotomy.Derived.Elliptic Apeiron.Audit` | exit 0; grade proved |
| E2 | same module plus `Audit` | exit 0; grade proved |
| M1, integrated | `lake build Apeiron.Physics.Quadratic Apeiron.Audit` | exit 0; proved |
| M2, dropped | no gate; a σ-free duplicate of `AffineMap.comp_assoc` | no result |
| C1, integrated | `lake build Apeiron.Trichotomy.AnyRing Apeiron.Audit` | exit 0; proved |
| C2, integrated | `lake build Apeiron.Trichotomy.Derived.Elliptic Apeiron.Audit` | exit 0; proved |
| N1, integrated | `lake build Apeiron.Numeric.Accumulator Apeiron.Audit` | exit 0; proved |
| Additive route, out of scope | no gate; it needs its own epic | no result |

A green build, a sampled test and a universal proof are three different pieces of evidence. E1, E2,
C1, N1, M1 and C2 are proved at the kernel gate. C1's `ringNormForm_conj`/`ringCayley_normForm`
audit inside `[propext, Quot.sound]`; the five N1 headlines and all 22 M1/C2 headlines inside
`[propext, Classical.choice, Quot.sound]`. Negative controls red then restored on throwaway copies:
C1 conclusion `= 1` to `= 2`; N1 numerator channel swapped to the denominator channel; M1 dropped
`+ outer.injection` and swapped `cellVec` slots; C2 dropped the `hskew` binder. No E1/E2 mutation
control is recorded in this handoff. C1's carried-inverse form versus a division-free
cross-multiplied variant remains an unconfirmed spec choice; the Cayley theorems carry no `@cite`
because `Linter/Citations.lean` has no Cayley or spinor key. Nothing is committed.

### Reference register

- Consuming tree `/home/socol/Workspace/apeiron`, pin `88866412`, unchanged.
- Live canonical `/home/socol/Workspace/sanguine-apeiron`, `336ce8d3`, dirty:
  `Physics/Quadratic.lean` (`rot`, `elliptic_one_param`, `unitNorm_elliptic`,
  `elliptic_rotors_commute`, all sigma < 0; M1 `ringRep`, `cellVec`, `toMatrixStep`, `ringRep_mul`,
  `matrixStep_apply`, `matrixStep_comp`, `ringRep_real_eq_rep`); `Trichotomy/Flow.lean` (`flow`,
  `flow_neg`, `normForm_flow`, `flow_add`); `Trichotomy/Derived/CStar.lean` (`cstarNormNeg`);
  `Trichotomy/Derived/Elliptic.lean` (`ellipticEquiv`, `ellipticEquiv_mul`, `dampedStep`,
  `damped_fold_bounded`; C2 `skew_re_zero`, `cayley_isometry_of_re_zero`, `cayley_isometry`);
  `Trichotomy/Trace.lean` (`unit_loop_isometric`); `Trichotomy/AnyRing.lean` (`ringMul`,
  `ringConj`, `ringNormForm`, `ringAffineApply`, `ringAffineFold`, `ringNormForm_conj`,
  `ringCayley_normForm`; M1 `MatrixAffineStep`, `matrixAffine`, `matrixStepComp`, `matrixStepId`,
  `matrixStepFold`); `Numeric/Accumulator.lean` (authorities
  `osFuse`/`osFuse_is_add`/`osFuse_fold_eq_add`/`osVal`; pair chart `osFusePair`, `osDen`, `osNum`,
  `osTile`, `osScan`, `softmax_attention_is_online_chart`); `Combinator/RouteAnyRing.lean`
  (`Route`).
- Old read-only `/home/socol/Workspace/sanguine`, `e830d3f` (faces verified at the declarations):
  `Substrate/Algebra/Neural/AttentionSpinor.lean:340` (`rotor`, `spinApply`; sigma < 0),
  `:403` (`oaFuse`, `softmax_attention_is_online_spinor`; sigma < 0);
  `Substrate/Algebra/Neural/TrainingCollapse.lean:290` (`rotorStep`, `rotor_isometry`; sigma < 0);
  `Substrate/StationaryPhase/SpinorLoop.lean:524` (`cayley_isometry`; sigma < 0),
  `:153` (`AffineStep`; sigma-free), `:304` (`AffineStepN`; sigma-free);
  `Substrate/Algebra/NullCell/NullCell.lean:110` (`nmul`; sigma > 0);
  `Substrate/Algebra/NullCell/FlawVarieties.lean:233` (`parabolic_flaw`; sigma = 0).
- Mathlib `fabf563a`: `AffineMap.lean:395` (`AffineMap.comp_assoc`), `Complex/Basic.lean`
  (`Complex.normSq`). First epic and ledger `docs/epics/spinor.md`.

### Non-claims

1. No all-corpus closure and no source-binding closure; the deferred rows above stay open.
2. No empty Slang cone: the cone is `{propext, Classical.choice, Quot.sound}`.
3. No performance, cost or allocation claim. No O(1) claim, whole-source or otherwise;
   `ringAffineFold` is O(n) in its list.
4. No Rust, Aeneas, Charon, Kani or Slang execution refinement; the epic is Lean-only.
5. No consumer pin uptake: `../apeiron` stays at `88866412`; nothing is committed or pushed.
6. The old sources at `e830d3f` are read-only references, not extracted evidence.
7. This file is a lifecycle record. It carries no proof and no `#audit` line of its own.

## Spinor Transport - Lifecycle

- **Owner:** deepseek-flash (SANGUINE-SPINORTRICHOTOMY-W6-T1, documentation)
- **Status:** full local proof transport. W1, W2, W3, W4 and W5 are proved at their focused gates
  and integrated uncommitted into the canonical sibling, at grade proved against the barrier gate
  below. Sol corrected one false canonical claim: the old sentence "at zero the failing loops are
  exactly the ones that die out" is replaced by the witness theorem
  `parabolic_failing_and_dying_loops`. The four canonical gates are green on this state:
  `lake build` exit 0 at 3522 jobs, `check-lints`, `check-style` (181 files) and `check-imports`
  exit 0. Epic 2 (E1/E2), Epic 3 (C1/N1) and Epic 4 (M1/C2) are integrated uncommitted in the same
  tree; see `docs/epics/spinor.md`.
  Epic-level acceptance still requires a commit decision by the lead; the consumer `../apeiron` pin
  stays `88866412`. No commit, no push. Not an all-corpus or source-binding closure.
- **Self-contained:** Intent, Spec, Plan, coverage ledger and evidence in this file.

### Source identity

- Consuming Rust/apeiron tree: `/home/socol/Workspace/apeiron`, revision
  `c450c222905c050f793a23332708c230c658f82c`.
- Resolved pinned corpus package:
  `/home/socol/Workspace/apeiron/proof/corpus/.lake/packages/apeiron/proof`, revision
  `8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f` (consumer pin unchanged).
- Canonical proof source base (live sibling): `/home/socol/Workspace/sanguine-apeiron`, revision
  `336ce8d3d7ef7d7be3be343e4b678ee16a5c321b`, dirty.
- W1/M1 file `proof/Apeiron/Trichotomy/AnyRing.lean`, sha256
  `decbba9dc965bbf94f83beb982d468e27ddee69c024ecf923740a30217445428`.
- W2 file `proof/Apeiron/Trichotomy/ChannelsAnyRing.lean`, sha256
  `52aaf5f09b8fbba44699adea79870dd2b05e05c4a68fa430aa4a9b7bffd2a361`.
- W2-W4 file `proof/Apeiron/Trichotomy/Trace.lean`, sha256
  `770255ab5f58c43d2227e36b17489fb559568d698e2192bbfbe34385b3aa3b27`.
- W5 file `proof/Apeiron/Table/Basic.lean`, sha256
  `4682563b466562c14b5a1fe78a91370d6beb9f4907fa7468af15f6f1143ccc90`.
- Integrated audit `proof/Apeiron/Audit.lean`, sha256
  `abacd1a6f1bbe1e8cc40c07035b36c8cbd3fee62d383a6414ac1a2bc63375543`.
- Epic 2 E1/E2 and Epic 4 C2 file `proof/Apeiron/Trichotomy/Derived/Elliptic.lean`, sha256
  `23d591d29095ce2de54ac4027d9e91ac73c88bfae89e26834aaa67cf54cb1535`, 644 lines; see
  `docs/epics/spinor.md`.
- Epic 3 C1/N1 and Epic 4 M1 files `proof/Apeiron/Trichotomy/AnyRing.lean`,
  `proof/Apeiron/Numeric/Accumulator.lean` and `proof/Apeiron/Physics/Quadratic.lean` (sha256
  `b90770cb6a43e58b6a92cb3b618190eee389dfebafd1e23928d343866da1848a`); hashes in
  `docs/epics/spinor.md`.
- Old source (read-only reference): `/home/socol/Workspace/sanguine`, revision
  `e830d3fbdeb40319d1be0d1fc803e706748b6a3c`.
- Lean toolchain `lean-toolchain`: `leanprover/lean4:v4.31.0`, commit
  `68218e876d2a38b1985b8590fff244a83c321783`.
- Mathlib `lake-manifest.json`: `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`.

The pinned package and the live sibling are different paths at different revisions: the consuming
bridge reads the `.lake/packages` copy at `88866412`, while the port edits the sibling at
`336ce8d3`. All named files are integrated byte-identical under lead-approved delegated integration;
the hashes above are the dirty attributable sources. The canonical dirty paths `AGENTS.md`,
`scripts/check-imports.py`, `theory/POTENTIAL_CLAIMS.md`, `theory/ROADMAP.md` and the untracked
`scripts/check-a5-control.py`, `scripts/check-f2-control.py` are untouched. The seven Combinator
style files were wrapped to <=100 chars in a separate pass; `check-style.py` is clean at 181 files.

### Intent

**Problem.** The old `SpinorLoop.lean` at `e830d3f` carried a monoid of scalar affine steps, the
iterate, and `aitken_exact`: three consecutive iterates of a constant affine loop return the fixed
point exactly. That composition had no canonical home. The canonical corpus had the two-slot
algebra `AnyRing` (free `sigma` over a commutative ring) and the real `Trace.lean`'s
`restPoint_fixed` and `aitken_exact`, but no step monoid and no link from the old iterate to a
canonical fold. The port supplied the missing home; Sol repaired one canonical prose sentence.

**Minimal scope.** No new module: the canonical affine cell composition over `AnyRing` suffices for
the old scalar and `NullCell` spinor material. `coefficient` and `injection` are the ordinary affine
coefficient and injection; `rotor` is not used because it names ordinary weights physical.
Restating `restPoint_fixed` or `aitken_exact` would break one authority.

**Invariants.** (1) `sigma` is explicit in every executing declaration; no declaration fixes it
silently. (2) `normForm` is the algebraic norm form of the two-slot algebra (indefinite in general),
not a positive norm; no theorem identifies it with an SO(2) Euclidean isometry or a length.
(3) Canonical `AnyRing`, `ChannelsAnyRing` and `Trace` statements are not duplicated; the new
declarations derive from `ringMul_assoc`, `ringMul_distrib_add`, `ringMul_one`, `ringMul_zero`,
`ringUpper_mul`, `ringUpper_add`, `ringLower_mul`, `ringLower_add` and `Trace.lean`'s
`aitken_exact`. (4) The fold is the left fold
`steps.foldl (fun acc step => ringAffineCompose sigma step acc) ringAffineId`; no right fold.
(5) The cone is `{propext, Classical.choice, Quot.sound}`; the empty Slang cone is not claimed.
(6) No new global O(1), backpropagation or source-binding claim; the generic ring laws remain the
authority.

**Non-goals.** No corpus move; no apeiron pin, package, extraction, Rust, Aeneas, Kani or
source-binding edit. No new generic claim outside the five task scopes; no test asserting blanket
injectivity.

**Trust boundary.** Evidence reaches the Lean declarations named below and their `#audit` cones. It
does not reach the old repository's build, rustc/LLVM, the kernel ABI, GPU/SPIR-V, or any executing
consumer. The old `SpinorLoop.lean` and its siblings are read references, not extracted evidence.

### Transport boundary (Astra)

**What the port covers.** The scalar affine monoid over `AnyRing.RingCell` is the canonical
analogue of the old scalar loop and is parametric in `sigma : R`; its laws hold for every
`CommRing`, so it instantiates at every `sigma`. The old `SpinorLoop.AffineStep`
(`SP/SpinorLoop.lean:153`) is a real scalar `rotor_decay`/`injection` pair with no `sigma` field:
its face is **sigma-free**, and it is not the `sigma = 1` face. The old `NullCell` spinor
material (`SpinorFold.upper_nmul`, `lower_nmul`, `nmul_is_diagonal`, `spinorFold_schedule_free`,
`rotor_isometry`, `one_half_decides`) rests on `NullCell.nmul`/`nNorm`
(`NC/NullCell.lean:110`, `:115`), the split algebra `x^2 = +1`: its face is **sigma > 0**
(hyperbolic; `sigma = 1` is one representative). No theorem in the port relates the old
declarations to the canonical ones, so these are canonical analogues/reads, not exact transports.

**Per-face attribution of the old declarations.** Paths are relative to
`/home/socol/Workspace/sanguine/proof/`; `SP` = `Substrate/StationaryPhase`,
`NC` = `Substrate/Algebra/NullCell`, `NA` = `Substrate/Algebra/Neural`,
`AS` = `Substrate/Algebra/Security`, `AT` = `Substrate/Algebra/Translation`,
`SU` = `Substrate`.

| Old row | Old declarations (read-only, file:line) | Face |
|---|---|---|
| Scalar affine step | `SP/SpinorLoop.lean:153` (`AffineStep`) | sigma-free |
| Matrix affine step | `SP/SpinorLoop.lean:304` (`AffineStepN`) | sigma-free |
| Complex rotor | `NA/AttentionSpinor.lean:340`, `:346`, `:486`, `:659` | sigma < 0 |
| Complex isometry | `SP/SpinorLoop.lean:524`, `:508` | sigma < 0 |
| Neural collapse | `NA/TrainingCollapse.lean:290`, `:296` | sigma < 0 |
| Neural ridge | `NA/TrainingCollapse.lean:193` (`drag_posDef`) | sigma < 0 |
| Split-cell fold | `NC/SpinorFold.lean:1` imports `NullCell` only | sigma > 0 |
| Parabolic edge | `NC/FlawVarieties.lean:233` (`parabolic_flaw`) | sigma = 0 |
| Split-cell identity | `NC/FlawVarieties.lean:139` (`cmul_one_eq_nmul`) | sigma > 0 |
| Generic fold | `SU/Collapse.lean:114`, `:122` | sigma-free |
| Generic table | `AS/HashTrinityCore.lean:83`, `:165` | sigma-free |
| Initiality | `AT/UniversalLexing.lean:110` | sigma-free |

The old corpus already spans the parabolic edge: `FlawVarieties.parabolic_flaw`
(`NC/FlawVarieties.lean:233`) exhibits the nonzero nilpotent `x` with `x^2 = 0` at `sigma = 0`,
and `FlawVarieties.cmul_one_eq_nmul` (`NC/FlawVarieties.lean:139`) states `cmul 1 = nmul`, the
split `sigma > 0` case. The canonical analogue of the parabolic fold is
`parabolic_square_zero_prefix_fold`; the sigma-free scalar loop is not the `sigma = 1` face.

**What is not transported here.** The complex rotor and its neural collapse numerics, and the
term-algebra initiality, have no canonical analogue in this port; the matrix/module action and the
complex isometry are integrated in `docs/epics/spinor.md` (Epic 4). A claim that the old
corpus spinor is entirely hyperbolic is false and is not made here.

**Forward task (separate named task, not silently imported).** A checked consumer/empty-cone
transport under the Slang empty cone (`tools/slang/zero_axioms.py`,
`proof/corpus/Slang/Audit/Empty.lean`) runs only when a consumer requires it. Parabolic and elliptic
readings use `Trace.lean`'s `depth_two_trichotomy` and `collapse_always_exact_iff_disc_neg`.

### Specification

#### Scope and admissible domain

| Item | Value |
|---|---|
| Operation | `RingAffineStep` apply/compose/fold; channel laws; `Trace` links |
| Domain | `R` a `CommRing`; `sigma : R` free; `z : RingCell R`; `steps : List ...` |
| Totality | total; `recip` returns the origin on the cone; collapse stays conditional |
| Preconditions | `r ^ 2 = sigma`; collapse needs three `normForm` terms nonzero |
| Postcondition | the named equality; no value fabricated on refusal |

#### Exact signatures (actual, public)

`Apeiron.Trichotomy.AnyRing` and `Apeiron.Trichotomy` are declaration namespaces.
`Apeiron.Trichotomy.ChannelsAnyRing` and `Apeiron.Trichotomy.Trace` in gate commands are module
and target names from filenames, not namespaces. Private helpers are named, not listed as public.

W1, namespace `Apeiron.Trichotomy.AnyRing`, file `Apeiron/Trichotomy/AnyRing.lean`:

```lean
structure RingAffineStep (R : Type*) [CommRing R] where
  coefficient : RingCell R
  injection   : RingCell R
def ringAffineId (R : Type*) [CommRing R] : RingAffineStep R
def ringAffineApply (sigma : R) (step : RingAffineStep R) (z : RingCell R) : RingCell R
def ringAffineCompose (sigma : R) (outer inner : RingAffineStep R) : RingAffineStep R
def ringAffineFold (sigma : R) (steps : List (RingAffineStep R)) : RingAffineStep R
-- laws: ringAffineId_apply, ringAffineId_compose, ringAffineCompose_id,
-- ringAffineCompose_apply, ringAffineCompose_assoc, ringAffineFold_apply, ringAffineFold_append
-- integer witness: ringAffine_hyperbolic_witness
-- private helpers: ringAffineFold_accum, ringAffineApply_fold
```

The witness `ringAffine_hyperbolic_witness`: `weighted` is `(3,1)`/`0`; `inject` is
`(1,1)`/`(1,1)`. Forward at `sigma = 1` gives `(4,4)`, reverse `(1,1)`, forward at `sigma = 0`
gives `(3,4)`, folds unequal.

W2, namespace `Apeiron.Trichotomy.AnyRing`, file `Apeiron/Trichotomy/ChannelsAnyRing.lean`:

```lean
theorem ringUpper_ringAffineApply {sigma r : R} (hr : r ^ 2 = sigma)
    (step : RingAffineStep R) (z : RingCell R) :
    ringUpper r (ringAffineApply sigma step z) =
      ringUpper r step.coefficient * ringUpper r z + ringUpper r step.injection
theorem ringLower_ringAffineApply {sigma r : R} (hr : r ^ 2 = sigma)
    (step : RingAffineStep R) (z : RingCell R) :
    ringLower r (ringAffineApply sigma step z) =
      ringLower r step.coefficient * ringLower r z + ringLower r step.injection
theorem ringUpper_ringAffineFold {sigma r : R} (hr : r ^ 2 = sigma)
    (steps : List (RingAffineStep R)) (z : RingCell R) :
    ringUpper r (ringAffineApply sigma (ringAffineFold sigma steps) z) =
      steps.foldl (fun acc step => ringUpper r step.coefficient * acc +
        ringUpper r step.injection) (ringUpper r z)
theorem ringLower_ringAffineFold {sigma r : R} (hr : r ^ 2 = sigma)
    (steps : List (RingAffineStep R)) (z : RingCell R) :
    ringLower r (ringAffineApply sigma (ringAffineFold sigma steps) z) =
      steps.foldl (fun acc step => ringLower r step.coefficient * acc +
        ringLower r step.injection) (ringLower r z)
```

W2-W4, namespace `Apeiron.Trichotomy`, file `Apeiron/Trichotomy/Trace.lean`:

```lean
theorem affineIter_eq_ringAffineFold (sigma : ℝ) (d c s : Cell) (n : ℕ) :
    affineIter sigma d c s n =
      AnyRing.ringAffineApply sigma
        (AnyRing.ringAffineFold sigma (List.replicate n ⟨d, c⟩)) s
theorem aitken_exact_of_prefix_fold (sigma : ℝ) (d c s : Cell) (n : ℕ)
    (hclose : normForm sigma (one - d) ≠ 0) (hloop : normForm sigma d ≠ 0)
    (hseed : normForm sigma (s - restPoint sigma d c) ≠ 0) :
    aitken sigma (three replicate-fold prefixes) = restPoint sigma d c
theorem parabolic_square_zero_prefix_fold (d c s : Cell) (hd : normForm 0 d = 0) :
    AnyRing.ringAffineApply 0
      (AnyRing.ringAffineFold 0 (List.replicate 2 (⟨d, c⟩ : AnyRing.RingAffineStep ℝ))) s =
    mul 0 d c + c
theorem collapse_always_exact_iff_disc_neg_prefix_fold (σ : ℝ) :
    (∀ (d c s : Cell) (n : ℕ), normForm σ (one - d) ≠ 0 → d ≠ 0 → s ≠ restPoint σ d c →
      aitken σ (three replicate-fold prefixes) = restPoint σ d c) ↔ σ < 0
theorem positive_face_prefix_fold_failure :
    ∃ sigma : ℝ, 0 < sigma ∧ ∃ (d c s : Cell) (n : ℕ),
      normForm sigma (one - d) ≠ 0 ∧ d ≠ 0 ∧ s ≠ restPoint sigma d c ∧
      aitken sigma (three replicate-fold prefixes) ≠ restPoint sigma d c
theorem aitken_hyperbolic_witness (n : ℕ) :
    normForm (1 : ℝ) (one - ((3 : ℝ), (1 : ℝ))) = 3 ∧
    normForm (1 : ℝ) ((3 : ℝ), (1 : ℝ)) = 8 ∧
    restPoint (1 : ℝ) ((3 : ℝ), (1 : ℝ)) ((1 : ℝ), (0 : ℝ)) = ((-2 / 3 : ℝ), (1 / 3 : ℝ)) ∧
    aitken (1 : ℝ) (affineIter ... n) (affineIter ... (n + 1)) (affineIter ... (n + 2)) =
      restPoint (1 : ℝ) ((3 : ℝ), (1 : ℝ)) ((1 : ℝ), (0 : ℝ))
-- private helper: affineIter_succ_apply
```

`positive_face_prefix_fold_failure`'s witness is `sigma = 1`, `d = (1, 1)`, `c = 0`, `s = one`,
`n = 1`: the three prefix folds are `(1, 1)`, `(2, 2)`, `(4, 4)`, their second difference has size
zero, so the extrapolation returns the first state rather than the resting point.

W5, namespace `Apeiron.Table`, file `Apeiron/Table/Basic.lean`:

```lean
theorem get_build_ringAffineCompose {K R : Type} [DecidableEq K] [CommRing R]
    (sigma : R) (kvs : List (K × AnyRing.RingAffineStep R)) (key : K) :
    get (build (AnyRing.ringAffineCompose sigma) (AnyRing.ringAffineId R) kvs) key =
      AnyRing.ringAffineFold sigma
        ((kvs.filter (fun kv => kv.1 = key)).map Prod.snd)
```

**Representation link, not a cost equation.** `affineIter_eq_ringAffineFold` is the representation
link: `Trace.lean`'s executable `affineIter` equals the replicate `AnyRing.ringAffineFold` applied
once. `aitken_exact_of_prefix_fold` is the separate prefix form that reads `Trace.lean`'s existing
`aitken_exact` through that link; it is not a second authority. `get_build_ringAffineCompose` reads
`get_build_acc` through the same fold. Neither statement is an O(1) or cost claim: `ringAffineFold`
is O(n) in the list length per its own docstring, and no executable cost is asserted.

### Plan

#### Barrier and gate commands

Every Lean gate is serialized and bounded, with no `maxHeartbeats 0`, no raised `maxRecDepth` and no
whole-source `decide`. Commands run individually from the stated cwd:

```sh
# cwd /home/socol/Workspace/sanguine-apeiron/proof
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope \
  -p MemoryMax=6G -p MemorySwapMax=0 -- env LEAN_NUM_THREADS=1 timeout 1800 lake build <targets>
lake build
# cwd /home/socol/Workspace/sanguine-apeiron
scripts/check-lints.py
python3 scripts/check-style.py
python3 scripts/check-imports.py
```

Every `lake build` named in the task blocks runs through the wrapper above from cwd `proof`; the
bare form is shorthand for that wrapper, not a second execution path. Rust and source-binding
barriers are not part of this port.

#### File ownership and integration

W1 owned `AnyRing.lean` and `Audit.lean`; W2 owned `ChannelsAnyRing.lean`, `Trace.lean` and
`Audit.lean`; W3 and W4 owned `Trace.lean` and `Audit.lean`; W5 owned `Table/Basic.lean` and
`Audit.lean`. W3 and W4 serialized after W2 on the shared `Trace.lean`; no two tasks held one file
at once. The lead integrated only the reviewed owned diffs into the supplied canonical checkout,
with no commit and no branch left behind; all five hashes are the integrated dirty sources.

#### Task blocks

```text
SCOPE:      W1 add the canonical RingAffineStep monoid and its laws to AnyRing; gate exits 0.
REGISTER:   local algebra obligation with a semantic state witness (the order- and
            sigma-distinguishing fold witness); no refinement, measurement or Rust obligation.
CORPUS:     Apeiron/Trichotomy/AnyRing.lean; derives from ringMul_assoc, ringMul_distrib_add,
            ringMul_one, ringMul_zero.
REFERENCE:  /home/socol/Workspace/sanguine/proof/Substrate/StationaryPhase/SpinorLoop.lean at
            e830d3f; AffineStep.apply/compose/compose_apply/compose_assoc resolve there.
GATE:       cd proof && lake build Apeiron.Trichotomy.AnyRing Apeiron.Audit; exit 0, 3519 jobs.
GRADE:      proved; witness values (4,4), (1,1), (3,4), folds unequal.

SCOPE:      W2 add the apply and fold channel laws and the Trace iterate/Aitken link; gate exits 0.
REGISTER:   local algebra transport (channel read of apply and of the fold; fold/Aitken link).
CORPUS:     ChannelsAnyRing.lean and Trace.lean; derives from ringUpper_mul/ringUpper_add,
            ringLower_mul/ringLower_add, restPoint_fixed, aitken_exact.
REFERENCE:  SpinorLoop.lean at e830d3f; AffineStep.iterate and AffineStep.aitken_exact resolve
            there; the old scalar loop is sigma-free, and the SO(2) tier is the separate
            AttentionSpinor rotor (sigma < 0), not this port.
GATE:       cd proof && lake build Apeiron.Trichotomy.ChannelsAnyRing Apeiron.Trichotomy.Trace
            Apeiron.Audit; exit 0.
GRADE:      proved; witness norms 3, 8, 1/3; restPoint (-2/3, 1/3).

SCOPE:      W3 add the parabolic square-zero prefix fold to Trace; gate exits 0.
REGISTER:   local algebra transport of the parabolic face onto the replicate fold.
CORPUS:     Trace.lean; derives from ringMul_distrib_add, ringMul_assoc, ringMul_comm,
            ringMul_zero and the replicate fold.
REFERENCE:  SpinorLoop.lean at e830d3f; the sigma-free scalar loop read at the canonical sigma = 0.
GATE:       cd proof && lake build Apeiron.Trichotomy.Trace Apeiron.Audit; exit 0.
GRADE:      proved; `normForm 0 d = 0` gives `d · d = 0` and the two-trip collapse `d · c + c`.

SCOPE:      W4 read the collapse iff and the positive-face failure on the replicate fold.
REGISTER:   local algebra transport of `collapse_always_exact_iff_disc_neg` plus its counterexample.
CORPUS:     Trace.lean; derives from collapse_always_exact_iff_disc_neg and
            affineIter_eq_ringAffineFold.
REFERENCE:  SpinorLoop.lean at e830d3f; the sigma-free scalar loop read at the canonical
            sigma > 0, where the collapse fails.
GATE:       cd proof && lake build Apeiron.Trichotomy.Trace Apeiron.Audit; exit 0.
GRADE:      proved; witness sigma = 1, d = (1,1), c = 0, s = one, n = 1.

SCOPE:      W5 add the per-key affine build to Table/Basic; gate exits 0.
REGISTER:   local state obligation: the built table's key read is the key's affine fold.
CORPUS:     Table/Basic.lean; derives from get_build_acc and AnyRing.ringAffineFold.
REFERENCE:  NONE - inventing (the old corpus has no table build over affine steps).
GATE:       cd proof && lake build Apeiron.Table.Basic Apeiron.Audit; exit 0.
GRADE:      proved; `rfl` after `get_build_acc`.

SCOPE:      Sol defect fix: correct the false canonical sentence and add the witness theorem.
REGISTER:   local canonical-claim correction, not a port scope.
CORPUS:     Trace.lean; new `parabolic_failing_and_dying_loops` from normForm_cpow and
            cpow_ne_zero_of_off_cone.
REFERENCE:  SpinorLoop.lean at e830d3f; the sigma-free scalar loop read at the canonical
            parabolic sigma = 0.
GATE:       cd proof && lake build Apeiron.Trichotomy.Trace Apeiron.Audit; exit 0.
GRADE:      proved; failing loop (1,1) never dies; every dying loop has first slot zero.
```

#### Sol correction (before / after)

Before, in `Trace.lean`'s `divergent_loop_of_disc_pos` docstring: "Below zero this cannot happen,
`loop_always_solvable_iff_disc_neg`; at zero the failing loops are exactly the ones that die out,
`nilpotent_iff_disc_zero`." After: "Below zero this cannot happen,
`loop_always_solvable_iff_disc_neg`. At zero `(1, 1)` fails to close and never dies, while every
loop that dies has first slot zero and closes, `parabolic_failing_and_dying_loops`."

The new witness theorem `Apeiron.Trichotomy.parabolic_failing_and_dying_loops` proves
`normForm 0 (one - ((1,1) : Cell)) = 0` and `∀ n, cpow 0 ((1,1) : Cell) n ≠ 0`; any dying loop
`cpow 0 d n = 0` has `d.1 = 0` and `normForm 0 (one - d) ≠ 0`, so the sets differ at `sigma = 0`.

### Coverage and outstanding ledger

This ledger records the old-source rows the port does not close. It is prose, not a gate: the
decision is **not** to add a new checker script, and closing this epic does not close the ledger.
The four canonical gates plus the `#audit` cone remain the acceptance surface.

- **Matrix/module affine action (sigma-free).** Old `AffineStepN.apply`, `compose`,
  `composeN_assoc`. Integrated: `AnyRing.matrixAffine`, `matrixStepComp`, `matrixStep_apply`,
  `matrixStep_comp`, `matrixStepFold`; `Physics.ringRep`, `cellVec`, `toMatrixStep`, `ringRep_mul`.
- **Elliptic complex rotor (sigma < 0).** Old `AttentionSpinor.rotor`, `rotor_isometry`,
  `rotor_compose`. Partial cover: `Trace.elliptic_never_terminates` for the disc-below-zero face;
  no complex rotor.
- **Complex inner-product isometry (sigma < 0).** Old `UnitaryLoop.skew_re_zero`,
  `cayley_isometry`, `cayley_conserves_born_density`. Integrated: `Derived/Elliptic.skew_re_zero`,
  `cayley_isometry_of_re_zero` and `cayley_isometry` over any complex inner-product space; the
  sharp local premise is `RCLike.re ⟪A (φ + ψ), φ + ψ⟫ = 0`. `cayley_conserves_born_density` stays
  uncovered.
- **`TrainingCollapse` contracts (sigma < 0).** Old forward-only/backprop and ridge/IRLS numerics:
  `ridgeHead_solves`, `ridgeHead_minimizes`, `newton_step_lands`, `training_collapse`; its real
  ridge sections are positive-definite, hence sigma < 0. Partial cover: `Apeiron.Numeric.*` numeric
  contracts only; no ridge, IRLS or backprop theorem.
- **Additive route trace for Aitken (sigma-free).** Old `AffineStep.iterate`, `second_difference`,
  `aitken_exact`, `one_step_error_identity`, `safe_error_bound`. Partial cover:
  `Trace.affineIter_telescopes`, `resolvent_from_three_partial_sums`, `loopSum_eq_sum`,
  `loopSum_telescopes`, `execution_formula`; the port adds `aitken_exact_of_prefix_fold`.
- **`Collapse` cost/memory (sigma-free).** Old `Substrate/Collapse.lean` `foldKey` over
  `AddCommMonoid`, `streaming_exact`, `streaming_order_free`, `foldKey_is_streamFold`; its cost and
  memory claims are prose and anchors. Partial cover: `Table.get_build_acc` and
  `build_eq_of_key_streams_eq` for the stream algebra.
- **`UniversalLexing` / `UniverseSyntax` initiality (sigma-free).** Old
  `UniverseSyntax.fromTerm_unique`, `fromTerm_ext`; `UniversalLexing.lexer_is_forced`,
  `lexing_collapses_witness`. Partial cover: none; the `Combinator` route algebra is not a
  term-algebra initiality theorem.
- **Completed-channel completion (sigma < 0).** The two channel reads of a completed step list.
  Partial cover: `ringUpper_ringAffineFold` and `ringLower_ringAffineFold`; the old
  `TrainingCollapse.qform_completion` (completing the square) has no canonical analogue.

### Evidence

#### Barrier on the integrated state

| Row | Command (cwd proof) | Result | Log |
|---|---|---|---|
| E8a | `lake build` | exit 0, 3522 jobs | `spinor-barrier-1-lake-build.log` |
| E8b | `scripts/check-lints.py` | exit 0, 3.81s | `spinor-barrier-2-check-lints.log` |
| E8c | `scripts/check-imports.py` | exit 0 | `spinor-barrier-4-check-imports.log` |
| E8d | `scripts/check-style.py` | exit 0, 181 files | `spinor-barrier-3-check-style.log` |

The pre-wrap `spinor-barrier-3` log is the before image. Post-W5 rebuilds (`gate1.log`, `gate2.log`,
`final-green.log`, `w4-restore.log`, `w5-restore.log` under `/tmp/opencode`) each exit 0 at 3519
jobs; adding theorems to existing modules leaves the module set unchanged. The barrier includes
Epic 2 E1/E2, Epic 3 C1/N1 and Epic 4 M1/C2.

#### Audit cones (read from `/tmp/opencode/final-green.log`)

25 new Epic 1 `#audit` targets, all inside `{propext, Classical.choice, Quot.sound}`; not the empty
Slang cone. Distribution: 5 `[]`, 8 `[propext]`, 5 `[propext, Quot.sound]`, 7 full cone. Epic 2
E1/E2 add 11; Epic 3 C1/N1 add 7; Epic 4 M1/C2 add 22 (12 AnyRing matrix, 7 Physics, 3 Cayley).

- `[]`: `RingAffineStep`, `ringAffineId`, `ringAffineApply`, `ringAffineCompose`, `ringAffineFold`.
- `[propext]`: the six unit/apply/assoc/fold laws, `ringAffine_hyperbolic_witness`,
  `get_build_ringAffineCompose`.
- `[propext, Quot.sound]`: `ringAffineFold_append`, `ringUpper_ringAffineApply`,
  `ringLower_ringAffineApply`, `ringUpper_ringAffineFold`, `ringLower_ringAffineFold`.
- `[propext, Classical.choice, Quot.sound]`: `parabolic_failing_and_dying_loops`,
  `affineIter_eq_ringAffineFold`, `aitken_exact_of_prefix_fold`, `aitken_hyperbolic_witness`,
  `parabolic_square_zero_prefix_fold`, `collapse_always_exact_iff_disc_neg_prefix_fold`,
  `positive_face_prefix_fold_failure`; all 22 Epic 4 M1/C2 headlines sit inside this cone.

#### Negative controls

| Wave | Mutation (throwaway copy) | Red | Restore |
|---|---|---|---|
| W1 | injection propagation -> `inner.injection` | `spinor-w1-neg.log` exit 1 | gate exits 0 |
| W1 | scratch `sorry` audit | `spinor-w1-sorry.log` exit 1 | not committed |
| W2 | left fold -> `List.foldr` | `w2_mut.log` error | `w2_base.log` |
| W3 | `replicate 2` -> `replicate 1` | `w3_mut.log` error | `w3_base.log` |
| W4 | `sigma = 1` -> `0` | `w4_mut.log`, `w4-red.log` | `w4-restore.log` exit 0 |
| W5 | filter `=` -> `≠` | `w5-red.log` exit 1 | `w5-restore.log` exit 0 |
| M1 | drop `+ outer.injection`; swap `cellVec` slots | red | restored |
| C2 | drop the `hskew` binder | red | restored |

The W1, W2, W3, W4, W5, M1 and C2 controls each turn the named declaration red and the restore
turns it green. All log paths are under `/tmp/opencode`.

### Non-claims

1. No O(1), cost, memory or allocation claim: `ringAffineFold` is O(n) in the list length.
2. No Rust, Aeneas, Charon, Kani or Slang execution refinement; the port is Lean-only.
3. No empty Slang cone: the cone is `{propext, Classical.choice, Quot.sound}`.
4. No all-corpus closure and no source-binding closure; the ledger above stays open.
5. No consumer pin uptake: `../apeiron` stays at `88866412` and nothing here is committed or pushed.
6. No claim that the old corpus spinor is entirely hyperbolic; see the Astra boundary.

### Lifecycle status

W1-W5, Epic 2 E1/E2, Epic 3 C1/N1 and Epic 4 M1/C2 are proved at their focused gates and integrated
byte-identical into the canonical sibling, uncommitted, at grade proved against the barrier gate
above. Sol corrected the false parabolic sentence and added `parabolic_failing_and_dying_loops`; the
C2 Cayley step needs no invertibility or finite dimension, only
`RCLike.re ⟪A (φ + ψ), φ + ψ⟫ = 0`. Open: C1's carried-inverse form versus a division-free
cross-multiplied variant is an unconfirmed spec choice, and the Cayley theorems carry no `@cite`
because `Linter/Citations.lean` has no Cayley or spinor key. Epic-level acceptance is not closed: it
requires a commit decision by the lead. The coverage ledger is open by decision, with no new checker
script.
Documentation-task gate: `git diff --no-index --check /dev/null docs/epics/spinor.md` (cwd
`/home/socol/Workspace/apeiron`; expected exit 1 for an untracked new file with no whitespace error;
exit 3 means a whitespace error) plus verification that every cited path resolves at its revision.

## Faces review series (FACES-R1 to FACES-R3)

Owner direction 2026-10-07: study LLM mathematics from the topological root (first cause, spinor,
torus), find where it is identical to other things, and say what each implementation is on the
elliptic, parabolic and hyperbolic base. Every statement below is grade `model` until its row's
test runs; none is `proved` by this section.

### First cause

- A point (the fingertip placed in an unbounded plane): no structure.
- A second placement gives a displacement `x`: one axis, an origin and a direction.
- Compounding a displacement with itself is the one new law, `x² = σ`. The sign of `σ` is the whole
  classification: `σ<0` the displacement returns (a rotation, period), `σ=0` it never turns (a
  shear: `x²=0`, nilpotent), `σ>0` it splits into two independent directions (idempotents
  `(1±x/√σ)/2`, a boost). The one-parameter orbit `exp(θx)` is `cos/sin`, `1+θε`, `cosh/sinh`.
- Consequence tested in R1: any model component that is a one-parameter group acting on positions
  or states has exactly these three orbit shapes.

### Hypotheses (each carries its falsifier)

| id | claim | face | falsifier | row |
|---|---|---|---|---|
| H1 | RoPE (RoFormer, Su et al. 2104.09864) is `exp(θx)` at `σ=-1` acting per frequency pair; relative position falls out because `exp(a)exp(b)=exp(a+b)` | elliptic | the relative-position identity fails numerically on a real checkpoint's rotated Q,K | R1 |
| H2 | ALiBi (Press et al., 2108.12409) is the `σ=0` orbit: a linear bias `m·(j−i)` is the dual-number translation `1+θε` read in log space | parabolic | a head's bias is not linear in distance in the shipped code | R1 |
| H3 | RetNet/SSM decay (`λ^(n−m)`, Mamba's `exp(Δ·A)` with real `A`) is the `σ>0` orbit with the two idempotent lanes as the growing and decaying modes | hyperbolic | a real-`A` decay cannot be written as `cosh/sinh` of one generator on a two-lane split | R1 |
| H4 | xPos (Sun et al., 2212.10554) is the product of H1 and H3: a rotation with a decay, i.e. a point of the complexified group, not a new object | elliptic × hyperbolic | the xPos operator is not `exp(z·x)` for complex `z` | R1 |
| H5 | The dependent dispatch chain (150 per token) is a nilpotent chain: each stage consumes the previous output and nothing else; `Perf.Chain.chainLB` is a sum for that reason | parabolic | a measured overlap between two dependent dispatches | R1 |
| H6 | Splitting into independent lanes (DP4A-packed bytes, split-K partials, the two idempotent modes of H3) is the hyperbolic idempotent decomposition: independent components, combined once | hyperbolic | a split whose lanes interact before the combine | R1 |
| H7 | Ring addressing, the all-reduce ring and RoPE's period are one circle: wraparound is the elliptic orbit's closure | elliptic | a ring address with no period | R2 |

### Reviews

| row | question | deliverable | home |
|---|---|---|---|
| FACES-R1 | For H1–H6, does the claimed identity hold on the shipped code? | a table row per hypothesis with the command, the number and the verdict; a refuted row stays | this section |
| FACES-R2 | What is each implementation on the other two bases? (position code, attention score, decay/state, hash-table probe, decode chain) | per component: its orbit on `σ<0`, `σ=0`, `σ>0`, what stays invariant, what breaks (range, period, definiteness of the norm `x²−σy²`), and one cheap experiment | this section |
| FACES-R3 | Which connections are in no source we hold? | a searched list: each candidate, the search command and result, `known (citation)` or `not found`; absence is stated with its search | this section |

Reading rules: a hypothesis is graded by its falsifier alone. The split-complex face is
`j²=+1`, `j≠±1`, not the complex numbers; no identity is imported from the complex case without
its own check. `docs/evidence/torus-spinor-sources.md` and `docs/evidence/atlas-lens-corpus.md`
hold the sources retrieved so far; a source not retrieved is a pointer, not evidence.

### FACES-R1 results (toy representation; shipped code not yet read)

Row: `faces-r1-do-the-three-on-HCnA8ZlHN3emV1nzkVH7V0`.

Command: `python3 /home/socol/oc-tmp/escalation/FACES-R1-orbits.py` (regular representation of
`R[x]/(x²−σ)` on `(a,b)`, `x·(a+bx)=σb+ax`; matrix exponential by scaling and squaring). Grade
`tested` for the toy identities; `model` for every statement about a shipped model.

| id | result | grade |
|---|---|---|
| H1–H3 | With the `SL₂`-invariant form `J` (every orbit has det 1, so `S(m)ᵀJS(n)=J·S(n−m)`), the score depends on `d=n−m` only, on all three faces: `ω(q,S(d)k)=a·g₀(d)+b·g₁(d)` with `(g₀,g₁)=(cos θd, sin θd)` at `σ=−1`, `(1, θd)` at `σ=0`, `(cosh θd, sinh θd)` at `σ=1`. Least-squares fit exact on `d<60` for each | tested |
| H1 | elliptic score is periodic in `d` (period `2π/θ`), bounded by `‖q‖‖k‖`: RoPE's aliasing and boundedness | tested (toy) |
| H2 | parabolic score is affine in `d` (`a + bθd`): the ALiBi form. The shear acts on logits; `exp` carries the additive group to the multiplicative one, so the same bias is a decay on weights. That `ALiBi ≈ exponential decay` is not claimed new | tested (toy) |
| H3 | hyperbolic score grows as `e^{θd}` (1.8e6 at `d=59`, `θ=0.31`); the orbit is `e^{θ}e₊ + e^{−θ}e₋` with `e₊e₋=0`. A stable hyperbolic position code keeps one idempotent lane (`|λ|<1`): this is the form RetNet and SSM decay take | tested (toy); the reading of RetNet/SSM is `model` |
| H4 | rotation × scalar decay is a one-parameter group (`R(2)D(2)R(3)D(3)=R(5)D(5)`), a point of the product of the elliptic and hyperbolic orbits | tested (toy) |
| H1 (shipped rotation) | `cargo test --lib rope_score_depends_on_the_relative_offset_only` on `RopeTable::new(1.0e6, 64)` with synthetic `q`,`k` (LCG, 64 elements, `i64<<8`): prediction written before the run, shifted score `⟨R(m+s)q,R(n+s)k⟩` differs from `⟨R(m)q,R(n)k⟩` by at most `2^-20·‖q‖‖k‖`, moving `q` alone changes it by more than `2^-6·‖q‖‖k‖`. Measured over 5 `(m,n)` × 4 shifts `s`: worst shift error `4.86e12` vs scale `1.06e20` (`4.6e-8` of `‖q‖‖k‖`, `2^-24.4`); worst move change `0.314` of scale (the control). Prediction held; log `/home/socol/oc-tmp/escalation/FACES-R1-H1-test.log`, row `faces-r1-do-the-three-on-HCnA8ZlHN3emV1nzkVH7V0`. Not covered: real checkpoint `q`,`k`; only the shipped fixed-point rotation, synthetic inputs | tested |
| H7 (shipped phase) | `RopeTable` stores the per-pair phase as `turn_j` in `Z/2^64`; `rope_turn_phase_is_additive_mod_2_to_64` checks it is a homomorphism `Z→Z/2^64`, so wraparound of the phase is the closure of the circle orbit by construction, not a separate mechanism. A ring address is the same quotient `Z/N`; the identification is a structural reading of shipped code, not a measured effect on any model | model, test-backed on the phase homomorphism |
| H6 (shipped split-K) | Row `faces-r1-do-the-three-on-HCnA8ZlHN3emV1nzkVH7V0`. Prediction (formed from the P25 receipt, before reading the shader lines): the `z` lanes of `coopmat_gemm.comp` read disjoint K ranges, write disjoint planes, share nothing before `splitk_reduce`/the norm fold, so the integer face is `S`-invariant to the bit and the `f32` face is not. Checks: (1) `rg -n 'atomic' src/spirv/coopmat_gemm.comp` returns no line; the only `z` uses are `bbeg/bend = (z*units/KSPLIT)*KB, ((z+1)*units/KSPLIT)*KB` (lines 359-360) and the store offset `poff = z*gl_NumWorkGroups.y*BN` (line 421); each lane is its own workgroup, so `shared` tiles are not common to lanes. (2) `python3` partition check: `floor((z+1)U/S)` telescopes, so the ranges cover `0..U` exactly once; `U=76`: `S=2` 38+38, `S=3` 25+25+26, `S=5` 15,15,15,15,16; `U=14` (the o GEMM): `S=3` 4+5+5. (3) `cargo test --locked --features gpu --test gpu_splitk` on the live tree at `6fdcc8e`: `splitk_down_gemm_sums_every_k_range_once` ok (8.87 s, `FOLD-live-splitk.log`); it asserts, per `S` in 2, 3, 5, that every plane cell equals the host sum of its own K range and that the summed integer face has 0 bit differences from the unsplit kernel. Reading (`model`): the lane projections `e_z` onto the K ranges satisfy `e_z e_w = δ_zw e_z` and `Σ e_z = 1`; for `S=2` with `j = e_0 − e_1` this is `j²=+1`, `j≠±1`, `e_{0,1} = (1±j)/2`, the split-complex idempotent pair acting diagonally on the K axis; for `S>2` it is `R^S`, not the 2-dimensional algebra. Held: no interaction before the combine in the shipped kernel. What the face does not give: the combine is exact over `Z` (the integer face, partials below `2^24`) and is **not** exact over `f32`: the same decomposition at `S=3` changes the bits of 396k of 459k output cells (max 3.9e-7 of the output scale, P25), because floating-point addition does not preserve the resolution; this is why the P25 floor moved 43 to 29. Split-K itself is standard GEMM practice (not searched here for prior art beyond the R3 digest, which has no row for it); the content is the naming, not a method. Not covered: GPU timing overlap of the lanes (the `decattr_probe chain` experiment, waits for the quiet GPU), lanes of a different kernel | tested (existing test re-run; shader read), reading `model` |
| H4 (shipped xPos) | Row `faces-r1-do-the-three-on-HCnA8ZlHN3emV1nzkVH7V0`. Reference: `fla/modules/rotary.py` (`~/Workspace/ref/flash-linear-attention` @ `1837094`, lines 410–463; found by local `rg`; the xPos paper 2212.10554 was not retrieved). The numpy reimplementation is `python3 /home/socol/oc-tmp/escalation/FACES-R1-H4-xpos.py`, with output in `FACES-R1-H4-xpos.out`. Inputs: `dim=64`, `base=1e4`, `scale_base=512`, `seqlen` 256 and 64, synthetic `q`,`k`. The prediction was written in the script header before the run. Results: (P1) the score depends on `n−m` only, worst error `2.1e-16` of `‖q‖‖k‖`; (P2) per pair `j`, the q-side operator is `exp(tZ)` with `Z=(ln ζ_j/γ)I+θ_jJ`, and the k side is `exp(tZ')` with `Z'=−Zᵀ≠Z`, worst error `8.2e-14` (the first run used a bare 60-term Taylor series that diverged at `‖tZ‖=255`; that was a defect in the instrument, replaced by scaling and squaring); (P3, control) putting the q operator on both sides breaks relativity, changing the score by `2.3e-2` of scale under a shift; (P4) the `seqlen//2` centring cancels, with caches of length 64 and 256 agreeing to `1.5e-17`; (P5) under a shift `s`, the scalar factors `(u,v)=(ζ^{p(n)},ζ^{−p(m)})` move as `diag(e^{sa},e^{−sa})` and `uv` is invariant to `2.2e-16`. Held: H4 is refined. xPos is not one group element acting on both q and k. Each pair is `C^× = R_{>0}×SO(2)` (elliptic rotation × real dilation, `z=ln ζ/γ+iθ`), and k carries the contragredient `−z̄`. Reading (`model`): the dilation is the hyperbolic face only on the (q-lane, k-lane) pair, where it is the boost that preserves the null-coordinate product `uv` (the split-complex norm form `x²−y²=uv`). Within a pair it is a real scalar, not `j`. That the k side takes the inverse scale is xPos's own construction, so the content here is the naming only. Not covered: no shipped model here uses xPos (qwen uses RoPE); synthetic inputs | tested (identities against the fla formula); reading `model` |
| H3 (real checkpoint) | Row `faces-r1-do-the-three-on-HCnA8ZlHN3emV1nzkVH7V0`. Qwen3.5-0.8B Gated DeltaNet (`/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf`; 18 linear-attention layers, 16 heads each). Decay per head and token `g = −exp(A_log)·softplus(a_t + dt_bias)`, `λ = e^g` (`ref/GatedDeltaNet` `lit_gpt/gated_delta_net.py:159`); at layer 0 `a_t` is a function of the token alone, so all 248320 vocabulary ids are samples. A head pair is `diag(λ_i, λ_j) = exp(a + b x)` with `a = (g_i+g_j)/2`, `b = (g_i−g_j)/2`. Command `python3 /home/socol/oc-tmp/escalation/FACES-R1-H3-gdn.py` (sha256 `b627a272036e8c19`, output `FACES-R1-H3-gdn.out`; prediction in the header, written before the run). (P1) `ssm_a < 0` in 288 of 288 heads (max −0.00137, min −10.58): `λ ∈ (0,1]` everywhere, `a ≤ −|b|`; held. (P2) one generator per pair would make `g_i/g_j` constant over tokens; median CV over 120 pairs 2.885 (min 0.168, threshold 0.1); robust spread added after the run: IQR of `ln(g_i/g_j)` median 1.79, min 0.195; held. Control: a shared `Δ` (Mamba-1 form) gives CV 2.6e-7 and IQR 2.4e-7. Layer-0 `λ` over tokens × heads: p1 0.0071, p50 0.9971, max 1; `|b|/|a|` median 0.983, so a pair sits near the null cone: one lane near `λ=1`, the other decaying. Verdict: **H3 refuted as stated** on this checkpoint, on both counts. No lane grows. The pair is not `cosh/sinh` of one generator: it lies in `R_{>0} × SO(1,1)`, scale times boost, the split-complex counterpart of xPos's `C^× = R_{>0} × SO(2)` (H4). The generator direction `(a, b)` changes per token, so a sequence gives `exp(Σ_t (a_t + b_t x))`: an element of the two-parameter group, not a point on one orbit. The one-parameter form holds exactly only where `Δ` is shared across lanes. Mamba-1 shares it (`ref/mamba` @ `e9594ce`, `selective_scan_interface.py:162`, `exp(einsum('bdl,dn->bdln', delta, A))`); the direction there is still `a < −|b|`, not a pure boost. `β = sigmoid(·) ∈ (0,1)` makes the delta-rule eigenvalue `1−β` positive, so no layer here carries the `−1` element that FACES-R2 decay/state needed for parity (structural; arXiv 2411.12537 is the pointer for negative eigenvalues, not retrieved). Not covered: layers past 0 (their `a_t` needs a forward pass), real text frequencies (every vocabulary id weighted equally), the conv and short-conv paths, and any effect on model quality | tested (weights, layer 0); reading `model` |
| H5 | Row `faces-r1-do-the-three-on-HCnA8ZlHN3emV1nzkVH7V0`. `examples/chain_probe.rs` (sha256 `4a9fa4e39859dcba`), RTX 5080, arms E/S/D2/I2/D16/I16 of `add_inplace_module`, `PREFILL_BAR=0` (barrier between dependent dispatches) vs `9` (none); lane 3 runs plus one live re-run, all agree. Marginal slope per dependent dispatch: 1.046 µs with the barrier vs 0.071 µs without at `n=2^14`; 12.80 vs 11.79 µs at `n=2^22`. The barrier adds ≈1.0 µs per dependent dispatch and the chain adds strictly, so the chain is nilpotent and its cost is a sum, as H5 states. Correctness: D16 at `BAR=0` has 0 mismatches at both sizes; without the barrier 16384 mismatches at `2^14` and 224–736 at `2^22`, so the barrier is the ordering, not overhead. Prediction P1 as written is refuted: its statistic `o_D = c − b` counts the graph-startup term (0.46 / 1.61 µs), not overlap. P3 and P5 held; P4 held at `2^14` and was refuted at `2^22` (cold buffer); P2's unit was ambiguous. Cross-check: P53 measured 1.13 µs per removed dispatch. Decode: 126 dispatches × ≈1 µs ≈ 126 µs per token is the barrier term that fusion removes | measured; P1 refuted |

Caveat: the norm form of `σ=0` is degenerate (`a²`), so the norm-form relative-position check is
trivial there and `J` is the form that carries position; the first script draft used the transpose
convention for `x`, which put the shear on the wrong coordinate and reported a norm violation that
was a convention error. Not covered: whether any checkpoint's heads use these forms (needs the
shipped rotation code and a trained model), and the line comparing `softmax(l−s·d)` with a
renormalised decay in the script is an identity by construction and carries no weight.

#### FACES-R1 H1/H2 bridge: RoPE pairs on the parabolic face (row `faces-r1-do-the-three-on-HCnA8ZlHN3emV1nzkVH7V0`)

Command: `python3 /home/socol/oc-tmp/escalation/FACES-R1-rope-contraction.py` (schedule `theta_j = theta^(-2j/hd)` read from
`src/engine/rope.rs` `RopeTable::new`; `theta = 1e6`, `hd = 64`, 32 pairs; prediction in the script header, written before the run).
Statement: a pair with `theta_j·L < sqrt(2ε)` stays within `ε` of the `σ=0` orbit `(1, θd)` for every `d ≤ L`
(`sup|cos θd − 1| = (θL)²/2`, `sup|sin θd − θd| = (θL)³/6`; direct grid check passed for every counted pair).

| context `L` | pairs within ε=1e-2 of parabolic | pairs with period `< L` (wrapped) | intermediate | prediction |
|---|---|---|---|---|
| 512 | 13 (j ≥ 19) | 11 | 8 | 13 / 11, matched |
| 2048 | 9 (j ≥ 23) | 14 | 9 | 9, matched |
| 32768 | 3 (j ≥ 29) | 20 | 9 | 3, matched |
| 131072 | 0 | 24 | 8 | 0, matched |

Grade `tested` (numeric, the shipped schedule; no checkpoint weights read). Reading (`model`): inside its training context RoPE
is not one face; high-frequency pairs are elliptic and aliased, low-frequency pairs are the ALiBi (parabolic) face, and the
parabolic share shrinks from 13 of 32 pairs at `L=512` to 0 at `L=131072`: the elliptic-to-parabolic limit `θ→0` is
the Inönü–Wigner contraction (Inönü and Wigner 1953; Yaglom's treatment of the three Cayley–Klein geometries; pointers, not
retrieved by the lead, not claimed new). Trained Q,K energy in the low-frequency pairs: see the next paragraph. A `docs/` search for `inonu|yaglom|contraction` hit only unrelated uses of "contraction" (files not opened).

**Trained Q,K energy in the parabolic pairs (row `faces-r1-do-the-three-on-HCnA8ZlHN3emV1nzkVH7V0`).** Command:
`QK_ENERGY=1 /home/socol/oc-tmp/shared-target/release/examples/attn_cut_probe` (`examples/attn_cut_probe.rs`, hook `qk_energy_layer`),
qwen05b Q4_K_M, prompt 0, `T=512`, 24 layers × 14 q heads × 32 pairs, all causal `(t,s)`; log
`/home/socol/oc-tmp/escalation/FACES-R1-QKENERGY.out`. Pair `j` is `(x[j], x[j+32])` of the roped vector; the share is that of
`j ≥ 19` (13 of 32 pairs, uniform `0.4062`). Prediction (`FACES-R1-QKENERGY-PREDICTION.md`, written before each run): share of
the squared pair contribution `(q_j·k_j)²` at least 0.55, refuted at or below 0.4062; after the first run, the per-query-centred form
at least 0.5.

| measure (layer mean [min, max]) | share of `j ≥ 19` |
|---|---|
| bound `Σ|q_j||k_j|` | 0.7389 [0.6126, 0.9054] |
| second moment `Σ(q_j·k_j)²` | 0.9463 [0.8132, 0.9982] |
| per-query centred second moment (`Σ_s c² − (Σ_s c)²/n`, the part softmax can see) | 0.5278 [0.0497, 0.9188]; pooled 0.4552 |

Reading: trained Q,K do put energy in the parabolic pairs, but 0.85 of the total second moment sits in `j = 29, 30, 31`
(0.14, 0.25, 0.46) and mostly as a per-query constant in `s` (a softmax-invariant offset; centring drops `j = 31` from 0.46 to 0.10). With that removed the excess over uniform is small
and layer-dependent: 16 of 24 layers exceed 0.4062; layers 0, 1, 2, 4, 7, 8, 12, 19 do not (layer 0: 0.05); layers 9, 11, 16, 20, 21, 22 exceed 0.80.
Prediction: held on the second moment and on the centred layer mean (0.5278 against 0.5, a margin inside the layer spread). The
claim that attention logits are mostly shaped by the parabolic (ALiBi-like) face is not supported: the centred pooled share is 0.455. Grade `tested`.
Not covered: other checkpoints and prompts, `T>512` (the parabolic set shrinks with `L`), the `d`-dependence inside those pairs
(affine in `d` is not measured), the Q4_K_M dequantised weights rather than the f16 checkpoint.

**The `d`-dependence inside the parabolic pairs (row `faces-r1-do-the-three-on-HCnA8ZlHN3emV1nzkVH7V0`).** Command:
`QK_DUMP=/home/socol/oc-tmp/faces-qk /home/socol/oc-tmp/shared-target/release/examples/attn_cut_probe` (hook in
`examples/attn_cut_probe.rs`: prompt 0's unrotated `q`,`k` per layer and the engine's own cos/sin table), then
`python3 /home/socol/oc-tmp/escalation/FACES-R1-H2-slope.py` (log `FACES-R1-H2-slope.out`); qwen05b Q4_K_M, `T=512`, 24 layers
× 14 q heads, queries `t ≥ 31`, keys `s ≤ t`. Per pair, `conj(z_q) z_k = A + iB` from the unrotated vectors (`A` the norm
bilinear form, position-free; `B` the `J` form) and the roped contribution is `c = Re((A+iB)·e_t·conj(e_s))` with `e_p` the
engine table; the positional part is `p = c − A`, which on a parabolic pair is `≈ −Bθd`, an ALiBi slope with a content-set
coefficient. Sanity: `Σ_j c` equals the roped dot product to `4.2e-16`. Prediction
(`FACES-R1-H2-SLOPE-PREDICTION.md`, written before the run): (1) positional share of the per-query centred variance over
`j ≥ 19` at most 0.10; (2) at most 0.01 for `j = 29..31`; (3) `p` linear in `d`, variance-weighted `r²` at least 0.95; (4) slope
set by the mean key, `r²` against `Re(conj(z_q) k̄ (e_t conj(e_s) − 1))` at least 0.5. Control: the elliptic pairs `j ≤ 10`.

| band | `Var(p)/Var(c)`, per pair | band sum | `Var(P_band)/Var(score)` | `Var(C_band)/Var(score)` | `r²(P, d)` | `r²(P, mean-key slope)` |
|---|---|---|---|---|---|---|
| parabolic `j ≥ 19` | 0.0010 | 0.0011 | 0.0005 | 0.4241 | 0.882 | 0.882 |
| far `j = 29..31` | 0.0000 | 0.0000 | 0.0000 | 0.2237 | 0.966 | 0.966 |
| elliptic `j ≤ 10` (control) | 1.089 | 1.161 | 0.5074 | 0.4370 | 0.100 | 0.786 |
| all 32 pairs | 0.535 | 0.672 | 0.672 | 1 | 0.242 | 0.762 |

Per layer, the parabolic band's positional share is at most 0.033 (layer 0); every other layer is ≤ 0.004. Predictions (1), (2),
(4) held; (3) is **refuted**: `r²(P,d) = 0.882 < 0.95`, so the slope varies with the key (`B` is not constant over `s`). The
control behaves as stated (`r² = 0.10` on the wrapped elliptic pairs). Reading: on this checkpoint the parabolic pairs carry
0.42 of the softmax-visible score variance, and 0.999 of that is the position-free norm form `A`. Geometrically they sit on the
`σ=0` orbit (the bridge table above); functionally they are content channels with a slope `≈ 5e-4` of the score variance.
Position is carried by the elliptic pairs (0.51 of the score variance is their positional part). The "low-frequency pairs are the
ALiBi face" reading above therefore holds for the group action and fails for what trained heads put on it. The result is
**known**: Barbero, Vitvitskyi, Perivolaropoulos, Pascanu, Veličković, *Round and Round We Go! What makes Rotary Positional
Encodings useful?* (arXiv 2410.06205v3; abstract fetched by the lead: Gemma "greatly prefers to use the lowest frequencies of
RoPE, which we suspect are used to carry semantic information", positional patterns use "the highest frequencies"). This
measurement reproduces it on qwen2.5-0.5B and adds the variance split (positional share 0.0011 of the band). Further pointers
from the `FACES-R3-ROPE-LOWFREQ` digest, not opened by the lead, are in the FACES-R3 table below. Not covered: the f16 checkpoint, other prompts and models, `T>512` (at `L=32768`, `θ_19·L ≈
9` rad and the same pairs are elliptic), the effect of the slope on the softmax output rather than on the logit variance. Grade
`tested`.

### FACES-R3 results (prior-art search)

Row: `faces-r3-searched-list-of-HFjIFdmrgQqByikqw7ayoE`. Digest `/home/socol/oc-tmp/escalation/FACES-R3-DIGEST.md`
(agy, Gemini 3.8 Flash; control passed: it re-found 2104.09864 and 2108.12409). Lead check: the abstract
pages of two cited papers were fetched; nothing else was retrieved. A digest item the lead did not
open is a pointer.

| candidate | result | status |
|---|---|---|
| RoPE and ALiBi as one group family | **Known in part.** GRAPE, arXiv 2512.07805: multiplicative rotations in `SO(d)` plus additive logit biases from unipotent actions in `GL`, recovering RoPE and ALiBi as special cases (abstract fetched). The abstract names no hyperbolic orbit, no `SO(1,1)`, no decay and no `R[x]/(x²−σ)`; the full text is not read | elliptic + parabolic known; the hyperbolic third face and the one-ring statement: not found in four queries |
| ALiBi equals a fixed exponential forget gate | Digest cites Forgetting Transformer, arXiv 2503.02130, §3, with a quoted sentence. Title and authors confirmed from the abstract page; the sentence itself is not verified | known (citation unverified at section level) |
| dual or split-complex numbers in position encoding | Second search (FACES-R3-IW-B, agy): **hyperbolic face known** — HoPE, Dai, Shan, Song, Liang, arXiv 2509.05218, §3.1 eqs. 5–7, the 2×2 Lorentz boost `[[cosh mθ, sinh mθ],[sinh mθ, cosh mθ]]` for monotone decay (title and authors confirmed from the abstract page by the lead). **Parabolic face known** as a nilpotent generator: GRAPE §4.1–4.2 (`A²=0`, `I+mA` gives exact ALiBi) and Jordan-RoPE, Yaobo Zhang, arXiv 2605.04217, §3.2–3.4 (`J = iωI + N`, `N²=0`, reducing at `ω=0` to `I+dN`; title confirmed by the lead). The names `R[ε]/(ε²)` and split-complex: not found | elliptic, parabolic and hyperbolic orbits each known separately; the one-ring statement `R[x]/(x²−σ)` over all three: not found; not a novelty claim |
| RoPE long-period pairs as an Inönü–Wigner contraction to the parabolic face | FACES-R3-IW-A/B (agy, row `faces-r3-searched-list-of-HFjIFdmrgQqByikqw7ayoE`): contraction named in a transformer, attention or position-encoding paper: not found (arXiv API `all:"Inonu-Wigner" AND (transformer OR RoPE)`, `all:"group contraction" AND ("positional encoding" OR RoPE)`; 2507.16871 uses contraction outside attention). Small-angle `sin θd ≈ θd` emulating ALiBi: known, Litzenberger (2026) blog "Emulating ALiBi with RoPE". Long-wavelength pairs as non-rotating: known, YaRN arXiv 2309.00071 §3.2. Digests `escalation/FACES-R3-IW-A.md`, `-B.md` | contraction reading: not found; its small-angle content: known |
| lowest RoPE pairs as position-free content channels in trained models (H2 result above) | **Known.** Barbero et al., arXiv 2410.06205v3, abstract fetched by the lead. Digest `agy-research/faces-r3-lowfreq/DIGEST.md` (agy, timed out at 20 m after writing it) also lists 2502.01563 (massive values in low-frequency Q/K), 2502.11276, 2410.08703, 2603.11611 (partial RoPE), 2606.21249 (zeroing lowest-frequency dims degrades retrieval in five models including Qwen-2.5) and 2607.19363 (AdaRoPE: semantic heads use the slowest frequencies); none opened by the lead (row `faces-r3-searched-list-of-HFjIFdmrgQqByikqw7ayoE`). ALiBi, arXiv 2108.12409v2: no small-angle relation to RoPE (digest; RoPE appears only as a baseline) | known; the measured variance split on qwen05b is a reproduction, not a novelty claim |
| real SSM diagonal as a two-idempotent split | not found in four queries; S4D (2206.11893) and LRU (2303.06349) contrast real decay with complex rotation without the idempotent reading | not found; not a novelty claim |
| dependent dispatch chain as nilpotent/Jordan | not found in four queries. Latency-sum framing is known: Event Tensor, arXiv 2604.13327, §1 (5–10 µs launch against 2 µs kernels), not retrieved by the lead | Jordan reading not found; latency sum known |

Consequence for the series: H1+H2 are known as a pair and the hyperbolic orbit is known on its own
(HoPE), so the open content narrows to the single ring over all three faces with the idempotent split
(H3), the contraction reading, H4 and H5/H6. A "not found" row shows a search
that found the two seeded papers; it does not show absence from the literature.

### FACES-R2 results (per component; model unless a command is named)

Row: `faces-r2-each-on-the-and-BegLX7J1BhlO8MfCu6nnZd`. Shared fact (tested in R1): a score that
depends on relative position through a one-parameter group obeys `f''(d) = σθ² f(d)`, so its
solution space is `{cos θd, sin θd}` (`σ<0`), `{1, d}` (`σ=0`), `{e^{θd}, e^{−θd}}` (`σ>0`). What
each face can express follows from that space.

| component | elliptic `σ<0` | parabolic `σ=0` | hyperbolic `σ>0` | what differs |
|---|---|---|---|---|
| position code (score vs `d`) | RoPE: periodic, bounded; can peak at a nonzero offset (`cos(θd−φ)`) | ALiBi: affine in `d`, no period, no peak | two exponentials; one idempotent lane kept (`|λ|<1`) for stability | only the elliptic face expresses "attend at fixed offset k" in one pair; parabolic extrapolates in length because nothing wraps; hyperbolic overflows (`e^{θd}`: 1.8e6 at `d=59`, `θ=0.31`) |
| attention score (`conj(q)k`, conjugation `a+bx ↦ a−bx`) | scalar part `q0k0 + q1k1`: the positive-definite dot product; times `e^{iθd}` it is `A cos θd − J sin θd` | scalar part `q0k0`: degenerate, and position-free (`(w)(1+θdε)` moves only the `ε` part) | scalar part `q0k0 − q1k1`: Minkowski; times `e^{θdx}` it is `A cosh θd + J sinh θd` | `conj(R_m q)R_n k = conj(q)k·exp((n−m)θx)` on every face (`N(exp θx)=1`); the `x` part is the `J` form on every face. ALiBi's affine `A + Jθd` is the small-`θd` limit of the elliptic or hyperbolic scalar part, not the `σ=0` scalar part. `python3 /home/socol/oc-tmp/escalation/FACES-R2-score.py`: relativity error ≤ `2.7e-7` relative, product and scalar-part forms exact, 1000 draws per face; `σ=0` scalar part constant in `d` (`−0.396` at every `d`); trained qwen05b puts `5e-4` of the score variance on that slope (R1 paragraph above). Grade `tested` |
| decay / state (`h←Ah+Bx`) | complex diagonal `A`: oscillator, phase memory, counts mod k | Jordan block `1+Nε`: state grows linearly, an integrator/counter | real diagonal `A`: exponential forgetting (RetNet, real-`A` Mamba) | memory length is `∞` (elliptic, `|λ|=1`), unbounded growth (parabolic), `1/(1−λ)` (hyperbolic) |
| norm (RMSNorm) | `x²+y²`, positive definite: the normaliser RMSNorm uses | `x²`, degenerate: kills the second coordinate | `x²−y²`, indefinite: zero on the null cone `x=±y`, where the idempotent lanes lie | RMSNorm presupposes the elliptic face. Toy: normalising 448 Gaussian pairs by the hyperbolic form gives inverse-rms gain median 5.6, p99 41.6, max 725.9 (elliptic ≈ 1.00) and a random sign (P(Q<0)=0.502) |
| hash-table probe | address ring `Z/2⁶⁴`, address determines key (regenerated); ring wraps | key carried beside the address; no wrap into the flat region (spill) | `Z/m` with CRT idempotent channels; address does not determine the key | already decided per face in `docs/epics/algebra-native-addressing.md` (lines 62–67, 127); the CRT split of `Z/m` is the discrete idempotent decomposition `(1±x)/2` |
| decode chain (cost) | the token loop: output token re-enters as input, period one token | dependent dispatches inside a token: strictly forward, nothing returns (nilpotent), cost adds: `chainLB = Σ dispLB` | independent lanes (heads, split-K, DP4A bytes): cost is the max, combined once | the chain model is sum for a dependent composition and max for a parallel one; the face names which composition applies |

Command for the norm row: `python3 /home/socol/oc-tmp/escalation/FACES-R2-norm.py` (numpy, seed 0,
`N=200000` samples, `P=448` pairs). Grade `tested` for the toy numbers; the reading of RMSNorm as an
elliptic-norm normaliser is `model` (it is the standard definition read on the face); nothing here
says a trained network would work better or worse with another form.

One cheap experiment per row (the decay/state row is run below; the others have not run, grade `model` until run):

| component | experiment | refuted if |
|---|---|---|
| position code | one head, fixed random `q,k`, three position codes; train on offset-retrieval at length `L`, test at `4L` | parabolic does not extrapolate better than elliptic, or elliptic cannot peak at offset |
| decay / state | 2-state linear recurrence on parity (mod-2 count), unbounded count, and moving average; one face per state | a face solves a task outside its solution space above chance |
| norm | replace RMSNorm in a small trained block by the `|x²−y²|` form with the same eps; compare loss and gain tail | the hyperbolic normaliser trains to the elliptic loss within noise, so the null cone does not bind |
| hash probe | read the existing per-face address decision against `Coord` measurements | none new: row defers to the addressing epic |
| decode chain | two independent GEMVs in one submission, with and without a barrier, against the dependent pair (`decattr_probe chain`) | independent pair time is not near the max of the two (overlap fails), refuting H6 on this driver |

#### FACES-R2 decay/state experiment (row `faces-r2-each-on-the-and-BegLX7J1BhlO8MfCu6nnZd`)

Command: `python3 /home/socol/oc-tmp/escalation/FACES-R2-decay.py` (numpy; all `2^12` binary strings; input 1 applies
`exp(θx)` or `-exp(θx)` in the regular representation, input 0 the identity; readout is the best linear functional of
the 2-d state plus threshold; prediction written in the script header before the run). Grade `tested` (toy).

| face, generator | parity accuracy | count R² (linear readout) |
|---|---|---|
| elliptic `θ=π` | 1.000 | 0.000 |
| elliptic `θ=0.3` | 0.726 | 0.991 |
| parabolic `θ=0.3` | 0.613 | 1.000 |
| hyperbolic `θ=0.5`, connected | 0.626 | 0.937 |
| hyperbolic `θ=0.1`, connected | 0.726 | 1.000 |
| hyperbolic `θ=0.5`, sign flip `-exp(θx)` | 1.000 | 0.000 |

Baselines computed in-session: the best one-threshold rule on the count gets 0.613 (parabolic hits it exactly,
its state is affine in the count); the best window `c1 ≤ count < c2` gets 0.7256 (elliptic `0.3` and hyperbolic
`0.1` hit it: a two-term state `a·e^{kθ}+b·e^{−kθ}` or `a·cos kθ+b·sin kθ` can be non-monotone in `k`). Verdict:
no orbit generated by a small `θ` exceeds the window ceiling; parity reaches 1.000 only where the state carries the
element `−1` (elliptic `θ=π`, inside the connected group; hyperbolic only through the sign flip, which is outside
the connected group `{exp(θx)}`). My refutation threshold of 0.7 was set below the window ceiling and was wrong;
the criterion that stands is "above the window ceiling". Count is solved linearly by the parabolic face exactly
(R² 1.000), by the others only for small `θ`, and the hyperbolic `θ=0.5` loses 6% to curvature.

Reading (`model`): the hyperbolic face needs the disconnected component (`−1`, equivalently a negative real
eigenvalue) for parity, the elliptic face has it at `θ=π`. Prior art not retrieved: negative-eigenvalue state
tracking in linear recurrences (arXiv 2411.12537, title from memory; a pointer, not evidence). Not covered:
input-dependent `θ` per token, trained gates, more than two states, readout beyond a threshold.

#### FACES-R2 decay contraction on a real checkpoint (row `faces-r2-each-on-the-and-BegLX7J1BhlO8MfCu6nnZd`)

Command: `python3 /home/socol/oc-tmp/escalation/FACES-R2-gdn-contraction.py` (sha256 `93dfde6095e8c103…`, output
`FACES-R2-gdn-contraction.out`; prediction in the script header and `FACES-R2-DECAY-CONTRACT-PREDICTION.md`, both written
before the run). Object: the Gated DeltaNet decay `λ = e^g`, `g = ssm_a·softplus(x·W_α + dt_bias) < 0` of
Qwen3.5-0.8B-Q8_0 layer 0, all 248320 vocabulary ids × 16 heads, from the weights alone (as `FACES-R1-H3-gdn.py`).
Criterion: the orbit `d ↦ e^{gd}` stays within `ε` of the parabolic orbit `1 + gd` for every `d ≤ L` iff
`(|g|L)²/2 ≤ ε`, because `0 ≤ e^x − 1 − x ≤ x²/2` for `x ≤ 0`; the lane is forgotten within `L` when `|g|L ≥ ln(1/ε)`.
`ε = 10⁻²`, as in the RoPE contraction table above. This is the hyperbolic-side counterpart of that table. Grade `tested`.

| L | parabolic | forgotten | between | head medians: parabolic / forgotten / between |
|---|---|---|---|---|
| 512 | 0.2148 | 0.3993 | 0.3859 | 2 / 6 / 8 |
| 2048 | 0.1110 | 0.5276 | 0.3614 | 1 / 7 / 8 |
| 32768 | 0.0645 | 0.8463 | 0.0892 | 1 / 14 / 1 |
| 131072 | 0.0627 | 0.9127 | 0.0246 | 1 / 15 / 0 |

- Controls:
  - Grid check: at the largest counted parabolic `|g|` (2.762e-4), `max_{d≤512} |e^{gd} − 1 − gd| = 0.009545 ≤ ε`.
  - Synthetic `|g| = 0.01`: parabolic fraction at `L = 512` is 0.
- Head 8 is an integrator:
  - `ssm_a = −0.00137`, `dt_bias = −13.1`;
  - median `|g| = 2.19e-10`, so `|g|·131072 ≈ 3e-5`;
  - its decay is the identity to `ε` at every tested length.
  - The gate on that head is inert: the recurrence there is the ungated delta rule.
- Heads 0, 3, 11 and 13 have `|g| ≥ 0.05` for every token (per-head minimum). They forget within about 90 tokens.
- Verdicts against the prediction:
  - P1 refuted: the `L = 512` fraction 0.2148 is inside [0.02, 0.40], but 0.0645 at `L = 32768` is above 0.05. The
    residue is head 8 (1/16 = 0.0625) plus a tail.
  - P2 held: 2 head medians are parabolic at 512.
  - P3 held: 6 head medians are forgotten within 512.
- Reading (`model`):
  - One layer splits its heads across the faces' limits by length: a σ=0 lane (an integrator, the shear/translation orbit),
    short-memory lanes, and lanes between.
  - The contraction parameter is `|g|L`, as `θL` is for RoPE. The same quadratic bound decides both faces.
  - Not shown: that the parabolic heads compute a count. That needs the state and the readout, not the decay alone.
- Search: none run for this check. The decay formula is read from `ref/GatedDeltaNet` `lit_gpt/gated_delta_net.py:159`
  (cited in the H3 script). No claim of novelty.
- Not covered:
  - layers past 0, which need a forward pass;
  - vocabulary ids weighted equally rather than by corpus frequency;
  - Q8_0 weights, not the BF16 release;
  - the delta-rule update `I − βkkᵀ`, classified in the next section.

#### FACES-R2 delta-rule transition (row `faces-r2-each-on-the-and-BegLX7J1BhlO8MfCu6nnZd`)

**Command.** `python3 /home/socol/oc-tmp/escalation/FACES-R2-deltarule.py`. Output: `FACES-R2-deltarule.out`. The prediction is in the script header, written before the run. The first run crashed on an embedding-shape defect in the instrument before reaching P3. It was fixed by a one-line `reshape`, and the predictions were not changed; the script was `3619a2b9c9c8bccc` before that fix and is sha256 `f6d76f95a56d06e7…` as run.

**Object.** The key-side factor of Gated DeltaNet, `H_t = I − β_t k_t k_tᵀ`, with `‖k_t‖ = 1` and `β_t = sigmoid(·) ∈ (0,1)`. The scalar decay `α_t` is the object of the two sections above. It is positive, so it changes no eigenvalue argument.

**Layer 0 of Qwen3.5-0.8B-Q8_0 from the weights.**
- Input: `rmsnorm(embed)·attn_norm`, then `attn_qkv`.
- A causal 4-tap depthwise conv, then silu.
- `k` is L2-normalised per head (16 heads × 128).
- `β = sigmoid(ssm_beta·x)`.
- Text: the first 2048 tokens of `wiki.valid.raw`. They are produced by the `tokenizer.json` BPE merges, with an ASCII approximation of its pre-tokenizer regex (`regex` is not installed). The ids are real vocabulary ids; a few split boundaries may differ from the reference tokenizer.

**Measured inputs.**
- `β`: p50 0.470, p99 0.9955, max 0.9993.
- `|k_t·k_{t−1}|`: p50 0.204, p90 0.544.

**Single factor (analytic).** Eigenvalues are `1` (× 127) and `1−β` along `k`. It is symmetric positive definite: a scale along one axis, with no rotation and no `−1`.

| check | result | prediction |
|---|---|---|
| P1: two factors, 2×10⁴ random draws each at `d = 128` and `d = 2`, `β ∈ (0,1)` | max `|Im λ|/|λ|` = 1.3e-15, min `Re λ` = 1.3e-6 > 0 | real and positive: held (a product of two SPD matrices is similar to an SPD matrix) |
| P2: three factors, `d = 2`, keys at 0°, 60°, 120°, `β = 0.999` | eigenvalues `0`, `−0.1239` (the projection limit is `cos60·cos60·cos120 = −0.125`) | a negative eigenvalue: held |
| P3, `T = 2` (checkpoint windows) | off-axis 0 / 32736 | 0: held |
| P3, `T = 3` | off-axis 1026 / 32720 = 0.031; largest surviving rotation 0.668 rad | > 0: held |
| P3, `T = 4` | 0.055; 1.081 rad | |
| P3, `T = 8` | 0.435; 1.297 rad | |
| P3, `T = 16` | 0.835; 1.255 rad | |
| P3, `T = 64` | 0.994; 0.697 rad | |
| control: `β → 2β` (range `(0,2)`) | `T = 1`: off-axis 0.446, surviving `|arg|` = π (a negative eigenvalue); `T = 3`: 0.611 | fires |

How the P3 statistics are defined:
- A window is *off-axis* when its product has some eigenvalue with `|λ| ≥ 1e-6` and `|arg λ| > 1e-6`.
- The *surviving rotation* is the largest `|arg λ|` among eigenvalues with `|λ| ≥ 0.1`.

**Verdict.** Held on every line.

**Reading** (`model`).
- One delta-rule step with `β < 1` sits on the scale face: real and positive, like the decay.
- Two steps cannot leave the positive axis.
- Three or more steps with non-collinear keys reach complex spectra. That is a rotation, the elliptic face, and in the projection limit a negative eigenvalue.
- On this checkpoint that happens in 3% of 3-token windows and in 99% of 64-token windows, with rotations up to 1.30 rad on eigenvalues of magnitude ≥ 0.1.
- So the H3 row's statement "no layer here carries the `−1` element" holds per step, not for the composed state map.
- The elliptic face is generated from the hyperbolic one by non-commuting composition. One step lies in `R_{>0}` along `k`; the key directions supply the rotation.

**Prior art.**
- Grazzi, Siems, Zela, Franke, Hutter, Pontil, *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*, arXiv 2411.12537; abstract fetched by the lead. It attributes the parity failure to *diagonal* transitions restricted to `[0,1]`, and states that products of `I − vvᵀ`-type matrices with eigenvalues in `[−1,1]` reach every regular language.
- The result above is consistent with that abstract. It does not show that `β < 1` solves parity: one symbol repeated gives powers of one SPD factor, which stay positive.
- The classical fact (products of positive definite matrices, Ballantine 1967–68) is a pointer from memory, not retrieved.
- No novelty is claimed. No search beyond the abstract was run.

**Not covered.**
- Layers past 0.
- The value side and the readout.
- Whether trained heads use the composed rotation.
- Windows longer than 64.
- Real-tokenizer boundary differences.

**Grade.** `tested` (identities and the layer-0 checkpoint spectrum).

#### FACES-R2 position-code experiment (row `faces-r2-each-on-the-and-BegLX7J1BhlO8MfCu6nnZd`)

Command: `python3 /home/socol/oc-tmp/escalation/FACES-R2-poscode.py` (numpy; output `FACES-R2-poscode.out`; prediction in the
script header, written before the run). Score as a function of offset `d` is a linear combination of a 16-pair bank of one
face's orbit (`cos/sin`, `1/d`, `cosh/sinh` of `θ_j d`); minimum-norm least squares (SVD, rank cut `1e-9`) fits a target on
`d<L=64`; the test takes the argmax over `d<4L`. Peak task: Gaussian bump (width 2) at offset `k`, 10 values of `k` in 5..59.
Recency task: target `−d/L`, success is argmax at `d=0`. Grade `tested` (toy; a least-squares fit, not a trained network).

| face | peak argmax=k at `L` | peak at `4L` | recency (`L`, `4L`) | numerical rank |
|---|---|---|---|---|
| elliptic | 1.00 | 0.00 | (1, 0) | 23 |
| parabolic | 0.00 | 0.00 | (1, 1) | 2 |
| hyperbolic | 0.60 | 0.30 | (1, 1) | 10 |

Against the prediction: held for parabolic (0.00 peak, recency 1/1: an affine score has its argmax at an edge) and for elliptic
peak at `L` (1.00) and elliptic recency at `4L` (0 against the predicted ≤0.5). Missed: elliptic peak at `4L` is 0.00 against
the predicted ≥0.7, and hyperbolic peak at `L` is 0.60 against the predicted ≤0.5 (a bank of ten effective exponentials can
form a bump; the claim that exponentials cannot peak holds for one pair only). The refutation conditions as written
(parabolic peak >0.1; elliptic peak at `L` <0.9; parabolic recency at `4L` <1.0 while elliptic is 1.0) did not fire, so the
row stands; the extrapolation prediction for the elliptic bank is refuted. Reading (`model`): on this task only the elliptic bank
fits a peak exactly in range, the parabolic bank cannot peak and is the only one whose recency survives `4L` together with
the hyperbolic bank; the elliptic bank's failure at `4L` follows from a fit with rank 23 and no constraint outside `d<L`, so the
out-of-range score is whatever the bank does there. Not covered: gradient training, a constrained fit (e.g. only pairs with
period above `4L`, which is the Inönü–Wigner limit of the contraction table above), other `N`, other bump widths, softmax
normalisation over the keys, any trained checkpoint. The decode-chain experiment still waits for a quiet GPU.

#### FACES-R2 position code, Inönü–Wigner restriction (row `faces-r2-each-on-the-and-BegLX7J1BhlO8MfCu6nnZd`)

Command: `python3 /home/socol/oc-tmp/escalation/FACES-R2-poscode-IW.py` (output `FACES-R2-poscode-IW.out`; prediction in the script
header, written before the run). Same fit as above (`L=64`, 16 pairs, minimum-norm least squares, rank cut `1e-9`); the restricted bank keeps only
elliptic pairs with period `≥ 4L` (`θ ∈ [2π/(8·4L), 2π/(4L)]`), where `cos θd → 1`, `sin θd → θd` and the bank contracts toward the parabolic
face. Grade `tested` (toy).

| bank | bump width | peak@`L` | peak@`4L` | recency (`L`, `4L`) | rank |
|---|---|---|---|---|---|
| elliptic, full | 2 | 1.00 | 0.00 | (1, 0) | 23 |
| elliptic, full | 16 | 1.00 | 1.00 | | 23 |
| elliptic, period `≥ 4L` | 2 | 0.40 | 0.20 | (1, 1) | 8 |
| elliptic, period `≥ 4L` | 16 | 1.00 | 0.40 | | 8 |

Against the prediction: held on every line (restricted width 2: peak@`L` 0.40 ≤ 0.5, peak@`4L` 0.20 ≤ 0.40, recency `(1,1)`; width 16: peak@`4L` 0.40 ≤ 0.5; the full
bank reproduces the committed 1.00 / 0.00). The refutation conditions (restricted peak@`4L` > 0.5 at width 2; restricted recency@`4L` < 1) did not fire.
Reading (`model`): restricting the periods removes the elliptic bank's out-of-range failure on recency (`(1,0)` to `(1,1)`) and costs the narrow
peak (1.00 to 0.40), which is the contraction table's direction; it is not a collapse to the parabolic face, since the restricted bank has rank 8 against the
parabolic bank's 2 and a width-16 bump is still fitted exactly in range. The full bank's width-16 peak@`4L` of 1.00 against 0.00 at width 2 shows the
`4L` failure above is a narrow-bump property of the fit, not of the elliptic face as such. Not covered: gradient training, softmax normalisation, a trained
checkpoint, other `N`. Prior art for the contraction: not searched in this slice; FACES-R3 holds the search record.

#### FACES-R2 norm experiment (row `faces-r2-each-on-the-and-BegLX7J1BhlO8MfCu6nnZd`)

Command: `python3 /home/socol/oc-tmp/escalation/FACES-R2-norm-train.py` (numpy; sha256 `130f2c0c478f23b3…`; output
`FACES-R2-NORM-v2.out`). It trains a pre-norm residual MLP stack with manual backward passes, 2000 Adam steps at lr `1e-3`,
batch 128, 5 seeds per normaliser. The task is an order-2 Markov chain over 64 symbols. Its entropy-rate floor is 3.6836 nats;
predicting uniformly costs 4.1589 nats. The whole-model gradient check passed on all 24 tensors (max relative error ≤ `1e-5`).
The run hit the 40-minute worker timeout after the first parabolic seed. Grade `tested` (toy).

| normaliser | final loss, mean of 5 seeds | range |
|---|---|---|
| elliptic (RMS) | 4.1542 | 4.1527–4.1551 |
| hyperbolic (`|x²−y²|`) | 4.2091 | 4.2003–4.2164 |
| parabolic | 4.1543 (1 seed) | — |

Verdict: **inconclusive**. The refutation condition did not fire: hyperbolic is 0.055 nats above elliptic, beyond both
spreads. That gap does not test the norm, though. No arm learned the task: elliptic ends 0.005 nats below uniform, with
0.47 nats still to the floor. Hyperbolic ends above uniform. The gap is a difference in failure to train, not in trained loss.
Next falsifier: raise the steps or the learning rate until the elliptic arm closes most of the gap to the floor, then rerun
all three arms.

Consequence: the faces are not interchangeable per component. Each row lists a capability one face
has and the others lack (peak at offset, extrapolation, forgetting, definite normaliser, key
recovery from address, overlap), so a variant that kept all invariants on every base would be flagged
not face-dependent; none did. The decode-chain experiment ran; its result is below.

#### FACES-R2 decode-chain experiment (row `faces-r2-each-on-the-and-BegLX7J1BhlO8MfCu6nnZd`)

**Command.**
`PREFILL_BAR=<9|0> CHAIN_LOG2N=14,16,18,20,22 CHAIN_REPS=300 chain_probe`, where `examples/chain_probe.rs` (measured at sha256 `e2b2ba92240cdb2d`; committed as its rustfmt, `952e8723a3e7ac33`, whitespace only)
gained a `CHAIN_LOG2N` size override.

**Runs and outputs.**
- 3 processes per barrier mode, RTX 5080, serialised on `/tmp/opencode/gpu-time.lock`.
- Log: `/home/socol/oc-tmp/escalation/FACES-R2-chain-runs.log`. Prediction, written before the run:
  `FACES-R2-CHAIN-PREDICTION.md`.

**Statistic.** `r16 = I16/S`, where `I16` is 16 `add_inplace` dispatches on 16 distinct `x` buffers and `S` is one
dispatch. The table gives the mean of the per-run medians, with min–max in brackets.

| `log2 n` | `S` µs | `I16` µs, no barrier | `r16`, no barrier | `r16`, shipped barrier (control) | `I16` bytes / time |
|---|---|---|---|---|---|
| 14 | 1.50 | 2.40 | 1.60 [1.60, 1.60] | 11.44 | — (latency) |
| 16 | 1.66 | 5.00 | 3.01 [3.00, 3.02] | 12.04 | 2.5 TB/s |
| 18 | 2.72 | 14.71 | 5.50 [5.01, 6.43] | 10.92 | 3.4 TB/s |
| 20 | 5.41 | 142.18 | 26.29 [26.28, 26.30] | 27.21 | 1.42 TB/s |
| 22 | 14.92 | 737.45 | 49.44 [49.07, 49.65] | 48.53 | 1.09 TB/s |

**Predictions.**
- P1 (`r16 ≤ 2` at `2^14`): held.
- P2 (`r16 ≥ 8` at `2^18`): **refuted**, at 5.50.
- P3 (`r16 ≥ 16` at `2^22`): held.
- P4 (monotone in `n`): held.
- C1 (the barrier forces `r16 ≥ 8` at every size): held, so the probe sees serialisation.

**Verdict.** The table row's claim, "independent lanes cost the max, combined once", is **refuted as stated**.
- It holds only in the latency regime: at `2^14`, 16 lanes cost 1.60× one lane.
- Past that, `r16` climbs to 49×, which is beyond the sum (16×).
- At `2^20` and `2^22` the barrier no longer matters (the two `r16` columns agree within 4 %). Throughput, not
  ordering, binds there.

The lead's own alternative (a sum-like cost once one lane fills the SM slots) is refuted by P2: at `2^18`, `S` is
still latency-dominated. The binding resource is L2 bandwidth (`16·12n` bytes in 14.7 µs = 3.4 TB/s), not
occupancy.

**Measured law.** On this device, the cost of independent lanes is
`max(latency floor, Σ_lanes bytes / BW(level holding the working set))`.
- Each lane moves `12n` bytes. Past `2^20`, the 16-buffer working set (≥ 68 MB) exceeds the 64 MB L2 (an anchor).
- `S` stays L2-resident (3.4 TB/s at `2^22`), while `I16` runs near the DRAM rate (1.09 TB/s).
- That residency gap is why `r16` exceeds 16.

**Reading** (`model`).
- The idempotent split is a property of the **data**: disjoint `x`, so `e_z e_w = 0` on the operands (H6 holds
  for the shipped split-K at the bit level, R1).
- Cost is a functional on the **device's resources**, and on a shared resource it adds. The lanes are therefore
  coupled before the combine, through bandwidth and cache capacity.
- The hyperbolic "max" is the cost law of the latency term only. The throughput term composes as a sum, the same
  composition as the dependent chain (H5).
- This is max across resources and sum within one resource, the max-plus form of a roofline.

**Prior art.** Standard. Pointers, not retrieved: the roofline model (Williams, Waterman, Patterson, *CACM* 52(4),
2009) and max-plus scheduling (Baccelli, Cohen, Olsder, Quadrat, *Synchronization and Linearity*, 1992). Novelty is
not claimed; the content here is the measured regime boundary on this device and the face reading.

**Consequence for decode.** Splitting decode work into independent lanes buys latency overlap only while the
summed working set stays in the level that serves `S`. It does not turn bandwidth into a max.

**Not covered.**
- Other kernels (GEMV with Q4 weights).
- The `D16` arm at bar 9, which is invalid: it shows 256–142752 mismatches, a race, so it is not a dependent chain.
- Device specs, which are anchors (64 MB L2, 960 GB/s DRAM), not measured here.

**Grade.** `measured` (RTX 5080, device timestamps, 3 runs, spread above); the reading is `model`.
