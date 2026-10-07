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
| H5, H6, H7 | not yet run | model |

Caveat: the norm form of `σ=0` is degenerate (`a²`), so the norm-form relative-position check is
trivial there and `J` is the form that carries position; the first script draft used the transpose
convention for `x`, which put the shear on the wrong coordinate and reported a norm violation that
was a convention error. Not covered: whether any checkpoint's heads use these forms (needs the
shipped rotation code and a trained model), and the line comparing `softmax(l−s·d)` with a
renormalised decay in the script is an identity by construction and carries no weight.

### FACES-R3 results (prior-art search)

Row: `faces-r3-searched-list-of-HFjIFdmrgQqByikqw7ayoE`. Digest `/home/socol/oc-tmp/escalation/FACES-R3-DIGEST.md`
(agy, Gemini 3.8 Flash; control passed: it re-found 2104.09864 and 2108.12409). Lead check: the abstract
pages of two cited papers were fetched; nothing else was retrieved. A digest item the lead did not
open is a pointer.

| candidate | result | status |
|---|---|---|
| RoPE and ALiBi as one group family | **Known in part.** GRAPE, arXiv 2512.07805: multiplicative rotations in `SO(d)` plus additive logit biases from unipotent actions in `GL`, recovering RoPE and ALiBi as special cases (abstract fetched). The abstract names no hyperbolic orbit, no `SO(1,1)`, no decay and no `R[x]/(x²−σ)`; the full text is not read | elliptic + parabolic known; the hyperbolic third face and the one-ring statement: not found in four queries |
| ALiBi equals a fixed exponential forget gate | Digest cites Forgetting Transformer, arXiv 2503.02130, §3, with a quoted sentence. Title and authors confirmed from the abstract page; the sentence itself is not verified | known (citation unverified at section level) |
| dual or split-complex numbers in position encoding | not found in five queries | not found; not a novelty claim |
| real SSM diagonal as a two-idempotent split | not found in four queries; S4D (2206.11893) and LRU (2303.06349) contrast real decay with complex rotation without the idempotent reading | not found; not a novelty claim |
| dependent dispatch chain as nilpotent/Jordan | not found in four queries. Latency-sum framing is known: Event Tensor, arXiv 2604.13327, §1 (5–10 µs launch against 2 µs kernels), not retrieved by the lead | Jordan reading not found; latency sum known |

Consequence for the series: H1+H2 are known as a pair, so the open content is H3 (the hyperbolic
orbit as the third face with the idempotent split), H4 and H5/H6. A "not found" row shows a search
that found the two seeded papers; it does not show absence from the literature.

### FACES-R2 results (per component; model unless a command is named)

Row: `faces-r2-each-on-the-and-BegLX7J1BhlO8MfCu6nnZd`. Shared fact (tested in R1): a score that
depends on relative position through a one-parameter group obeys `f''(d) = σθ² f(d)`, so its
solution space is `{cos θd, sin θd}` (`σ<0`), `{1, d}` (`σ=0`), `{e^{θd}, e^{−θd}}` (`σ>0`). What
each face can express follows from that space.

| component | elliptic `σ<0` | parabolic `σ=0` | hyperbolic `σ>0` | what differs |
|---|---|---|---|---|
| position code (score vs `d`) | RoPE: periodic, bounded; can peak at a nonzero offset (`cos(θd−φ)`) | ALiBi: affine in `d`, no period, no peak | two exponentials; one idempotent lane kept (`|λ|<1`) for stability | only the elliptic face expresses "attend at fixed offset k" in one pair; parabolic extrapolates in length because nothing wraps; hyperbolic overflows (`e^{θd}`: 1.8e6 at `d=59`, `θ=0.31`) |
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

Consequence: the faces are not interchangeable per component. Each row lists a capability one face
has and the others lack (peak at offset, extrapolation, forgetting, definite normaliser, key
recovery from address, overlap), so a variant that kept all invariants on every base would be flagged
not face-dependent; none did. The decode-chain experiment needs the GPU and waits for the running
lane.
