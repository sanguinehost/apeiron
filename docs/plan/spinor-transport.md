# Spinor Transport - Lifecycle

- **Owner:** deepseek-flash (SANGUINE-SPINORTRICHOTOMY-W6-T1, documentation)
- **Status:** full local proof transport. W1, W2, W3, W4 and W5 are proved at their focused gates
  and integrated uncommitted into the canonical sibling, at grade proved against the barrier gate
  below. Sol corrected one false canonical claim: the old sentence "at zero the failing loops are
  exactly the ones that die out" is replaced by the witness theorem
  `parabolic_failing_and_dying_loops`. The four canonical gates are green on this state:
  `lake build` exit 0 at 3522 jobs, `check-lints`, `check-style` (181 files) and `check-imports`
  exit 0. Epic 2 (E1/E2), Epic 3 (C1/N1) and Epic 4 (M1/C2) are integrated uncommitted in the same
  tree; see `docs/plan/spinor-coverage.md`.
  Epic-level acceptance still requires a commit decision by the lead; the consumer `../apeiron` pin
  stays `88866412`. No commit, no push. Not an all-corpus or source-binding closure.
- **Self-contained:** Intent, Spec, Plan, coverage ledger and evidence in this file.

## Source identity

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
  `docs/plan/spinor-coverage.md`.
- Epic 3 C1/N1 and Epic 4 M1 files `proof/Apeiron/Trichotomy/AnyRing.lean`,
  `proof/Apeiron/Numeric/Accumulator.lean` and `proof/Apeiron/Physics/Quadratic.lean` (sha256
  `b90770cb6a43e58b6a92cb3b618190eee389dfebafd1e23928d343866da1848a`); hashes in
  `docs/plan/spinor-coverage.md`.
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

## Intent

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

## Transport boundary (Astra)

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
complex isometry are integrated in `docs/plan/spinor-coverage.md` (Epic 4). A claim that the old
corpus spinor is entirely hyperbolic is false and is not made here.

**Forward task (separate named task, not silently imported).** A checked consumer/empty-cone
transport under the Slang empty cone (`tools/slang/zero_axioms.py`,
`proof/corpus/Slang/Audit/Empty.lean`) runs only when a consumer requires it. Parabolic and elliptic
readings use `Trace.lean`'s `depth_two_trichotomy` and `collapse_always_exact_iff_disc_neg`.

## Specification

### Scope and admissible domain

| Item | Value |
|---|---|
| Operation | `RingAffineStep` apply/compose/fold; channel laws; `Trace` links |
| Domain | `R` a `CommRing`; `sigma : R` free; `z : RingCell R`; `steps : List ...` |
| Totality | total; `recip` returns the origin on the cone; collapse stays conditional |
| Preconditions | `r ^ 2 = sigma`; collapse needs three `normForm` terms nonzero |
| Postcondition | the named equality; no value fabricated on refusal |

### Exact signatures (actual, public)

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

## Plan

### Barrier and gate commands

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

### File ownership and integration

W1 owned `AnyRing.lean` and `Audit.lean`; W2 owned `ChannelsAnyRing.lean`, `Trace.lean` and
`Audit.lean`; W3 and W4 owned `Trace.lean` and `Audit.lean`; W5 owned `Table/Basic.lean` and
`Audit.lean`. W3 and W4 serialized after W2 on the shared `Trace.lean`; no two tasks held one file
at once. The lead integrated only the reviewed owned diffs into the supplied canonical checkout,
with no commit and no branch left behind; all five hashes are the integrated dirty sources.

### Task blocks

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

### Sol correction (before / after)

Before, in `Trace.lean`'s `divergent_loop_of_disc_pos` docstring: "Below zero this cannot happen,
`loop_always_solvable_iff_disc_neg`; at zero the failing loops are exactly the ones that die out,
`nilpotent_iff_disc_zero`." After: "Below zero this cannot happen,
`loop_always_solvable_iff_disc_neg`. At zero `(1, 1)` fails to close and never dies, while every
loop that dies has first slot zero and closes, `parabolic_failing_and_dying_loops`."

The new witness theorem `Apeiron.Trichotomy.parabolic_failing_and_dying_loops` proves
`normForm 0 (one - ((1,1) : Cell)) = 0` and `∀ n, cpow 0 ((1,1) : Cell) n ≠ 0`; any dying loop
`cpow 0 d n = 0` has `d.1 = 0` and `normForm 0 (one - d) ≠ 0`, so the sets differ at `sigma = 0`.

## Coverage and outstanding ledger

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

## Evidence

### Barrier on the integrated state

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

### Audit cones (read from `/tmp/opencode/final-green.log`)

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

### Negative controls

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

## Non-claims

1. No O(1), cost, memory or allocation claim: `ringAffineFold` is O(n) in the list length.
2. No Rust, Aeneas, Charon, Kani or Slang execution refinement; the port is Lean-only.
3. No empty Slang cone: the cone is `{propext, Classical.choice, Quot.sound}`.
4. No all-corpus closure and no source-binding closure; the ledger above stays open.
5. No consumer pin uptake: `../apeiron` stays at `88866412` and nothing here is committed or pushed.
6. No claim that the old corpus spinor is entirely hyperbolic; see the Astra boundary.

## Lifecycle status

W1-W5, Epic 2 E1/E2, Epic 3 C1/N1 and Epic 4 M1/C2 are proved at their focused gates and integrated
byte-identical into the canonical sibling, uncommitted, at grade proved against the barrier gate
above. Sol corrected the false parabolic sentence and added `parabolic_failing_and_dying_loops`; the
C2 Cayley step needs no invertibility or finite dimension, only
`RCLike.re ⟪A (φ + ψ), φ + ψ⟫ = 0`. Open: C1's carried-inverse form versus a division-free
cross-multiplied variant is an unconfirmed spec choice, and the Cayley theorems carry no `@cite`
because `Linter/Citations.lean` has no Cayley or spinor key. Epic-level acceptance is not closed: it
requires a commit decision by the lead. The coverage ledger is open by decision, with no new checker
script.
Documentation-task gate: `git diff --no-index --check /dev/null docs/plan/spinor-transport.md` (cwd
`/home/socol/Workspace/apeiron`; expected exit 1 for an untracked new file with no whitespace error;
exit 3 means a whitespace error) plus verification that every cited path resolves at its revision.
