# Spinor Coverage - Second Epic Lifecycle

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

## Source identity

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

## Intent

**Problem.** The first epic (`docs/plan/spinor-transport.md`) ported the canonical analogue of
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

## Scope boundary: already canonical, not to be duplicated

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

## Specification

### Wave 1 (E1): the elliptic representation, integrated

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

### Wave 2 (E2): the damped elliptic fold preserves an inhabited ball, integrated

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

### Wave 3 (C1): scalar Cayley factor over `AnyRing`, integrated

File `proof/Apeiron/Trichotomy/AnyRing.lean`, namespace `Apeiron.Trichotomy.AnyRing`. New
`ringNormForm_conj` (conjugation leaves the norm form alone) and `ringCayley_normForm`: for
`σ θ : R` and `w : RingCell R`, a carried inverse witness `hw : ringMul σ (1, θ) w = ringOne`
gives `ringNormForm σ (ringMul σ (ringConj (1, θ)) w) = 1`. A general `CommRing` is not a field, so
division is carried, not total. The carried-inverse form is the implemented statement; the choice
versus a division-free cross-multiplied variant is an unconfirmed spec choice, not proved here.
Integrated uncommitted; grade proved at the canonical gate below.

### Wave 4 (N1): the online numerator/denominator chart, integrated

File `proof/Apeiron/Numeric/Accumulator.lean`, namespace `Apeiron.Numeric`. New `osFusePair`,
`osDen`, `osNum`, `osTile`, `osScan`, `osDen_fuse`, `osNum_fuse`, `osDen_scan_acc`,
`osNum_scan_acc` and the headline `softmax_attention_is_online_chart`: for `l : List (ℝ × ℝ)`,
`osNum (osScan l)` is the value-weighted exponential sum and `osDen (osScan l)` the exponential
sum, and both channels merge as exact `+`. This transports the old three-conjunct
`softmax_attention_is_online_spinor` at `/home/socol/Workspace/sanguine` rev `e830d3fb`,
`proof/Substrate/Algebra/Neural/AttentionSpinor.lean:403-469`, not the doc's paraphrase. The chart
is not `Trichotomy.Cell`; no `ringMul` or `ringNormForm` is read. Integrated uncommitted; grade
proved at the canonical gate below.

### Wave 5 (M1): the matrix/module affine action, integrated

File `proof/Apeiron/Trichotomy/AnyRing.lean` adds `MatrixAffineStep`, `matrixAffine`,
`matrixStepComp`, `matrixStep_apply`, `matrixStep_comp`, `matrixStepId`, `matrixStepId_apply`,
`matrixStepId_comp`, `matrixStepComp_id`, `matrixStepFold`, `matrixStepFold_apply` and
`matrixStepFold_append`, all under `[Fintype ι] [DecidableEq ι] [Semiring R]`. File
`proof/Apeiron/Physics/Quadratic.lean` adds `ringRep`, `cellVec`, `toMatrixStep`, `ringRep_mul`,
`matrixStep_apply`, `matrixStep_comp` and `ringRep_real_eq_rep`; `Apeiron.Physics.rep`
(`Physics/Quadratic.lean:75`) is unchanged and `ringRep_real_eq_rep` is `rfl`. Only AnyRing adds
the import `Mathlib.Data.Matrix.Mul`. Integrated uncommitted; grade proved at the canonical gate.

### Wave 6 (C2): the general complex-linear Cayley isometry, integrated

File `proof/Apeiron/Trichotomy/Derived/Elliptic.lean` (644 lines) adds `skew_re_zero`,
`cayley_isometry_of_re_zero` and `cayley_isometry` over an arbitrary complex inner-product space.
The implication `φ + Aφ = ψ - Aψ → ‖φ‖ = ‖ψ‖` needs neither `(1 + A)` invertible nor finite
dimension; the sharp local premise is `RCLike.re ⟪A (φ + ψ), φ + ψ⟫ = 0`, which `skew_re_zero`
discharges for skew-adjoint `A`. This replaces the earlier "may need invertibility or finite
dimension" note with Sol's verdict. The Cayley theorems carry no `@cite`: `Linter/Citations.lean`
has no Cayley or spinor key. Integrated uncommitted; grade proved at the canonical gate below.

## Plan

### Barrier and gate commands

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

### Task blocks

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

### Deferred rows, with the exact reason

- **M2 `ComposeExpr` naturality.** Dropped: it is a σ-free duplicate of Mathlib's
  `AffineMap.comp_assoc` (`Mathlib/LinearAlgebra/AffineSpace/AffineMap.lean:395`) unless a
  consumer names it.
- **Additive GoI route and Aitken on path sums.** Out of scope. `Combinator/RouteAnyRing.lean`
  defines `Route R` as a partial injection `ℕ → Option (ℕ × RingCell R)`: nonbranching, with no
  route addition. A path-sum Aitken needs its own epic.

## Old readings that must not be restated

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

## Evidence

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

## Reference register

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
  (`Complex.normSq`). First epic and ledger `docs/plan/spinor-transport.md`.

## Non-claims

1. No all-corpus closure and no source-binding closure; the deferred rows above stay open.
2. No empty Slang cone: the cone is `{propext, Classical.choice, Quot.sound}`.
3. No performance, cost or allocation claim. No O(1) claim, whole-source or otherwise;
   `ringAffineFold` is O(n) in its list.
4. No Rust, Aeneas, Charon, Kani or Slang execution refinement; the epic is Lean-only.
5. No consumer pin uptake: `../apeiron` stays at `88866412`; nothing is committed or pushed.
6. The old sources at `e830d3f` are read-only references, not extracted evidence.
7. This file is a lifecycle record. It carries no proof and no `#audit` line of its own.
