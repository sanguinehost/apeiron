# Rigor baseline — TorchLean claim-by-claim

Reference: `/home/socol/oc-tmp/torchlean-de3192df`, a read-only detached worktree of
`/home/socol/Workspace/TorchLean` at `de3192df4779877ceb65cfe470a4d4ce9d480a5b` ("Fix FloatLib API
documentation link", 2026-09-21). The tree is **not built**. Apeiron revision of this receipt:
`74c037b1cfc79b0d6be6d3907398fe926d78d12d`. Grades: `cited` where a theorem or line is quoted,
`anchor` for counts and inferences. Appendix A retains the earlier `12f5c651…` snapshot findings,
now superseded.

## Retrieval

| Item | Value |
|---|---|
| `git -C /home/socol/Workspace/TorchLean rev-parse de3192df…` | `de3192df4779877ceb65cfe470a4d4ce9d480a5b` (exit 0) |
| Worktree | `/home/socol/oc-tmp/torchlean-de3192df`, detached HEAD, clean, not built |
| Toolchain | `leanprover/lean4:v4.34.0` (`lean-toolchain`) |
| Snapshot → pin delta | `1818 files changed, 219943 insertions(+), 177824 deletions(-)` (`git diff --shortstat 12f5c651… de3192df…`) |

The D4 receipt read `12f5c651…` because `de3192df…` was then absent from the checkout. The pin now
resolves, the worktree is clean, and this receipt re-binds every citation to the pinned revision.
The `12f5c651…` checkout remains at `/home/socol/Workspace/TorchLean`; its findings are in
Appendix A.

## Verdict at the pin

Apeiron **exceeds** TorchLean on the two axes that are a standard rather than a breadth count: the
refinement of the *executing* artifact and the mechanised axiom cone. The reference has grown across
the snapshot→pin delta (2,497 → **3,007** `theorem`/`lemma` declarations) and added a genuine
execution-binding headline that ties the executed Lean tape to the Fréchet adjoint
(`backwardDenseAll_lowerGraphToTape_adjoint_fderiv`, `NN/Proofs/Autograd/Runtime/Link/BackwardDenseGraph.lean:389`).
But the pin still has **no source extraction** (no Aeneas/Charon), **no transitive cone gate**, and
**no mutation control**; `docs/TRUST_BOUNDARIES.md:31-33` still states that the executed-sweep
theorem "says nothing about `Float` rounding, compiled code, or CUDA", and the native path remains a
documented external assumption. What Apeiron lacks is **surface**: TorchLean carries ≈3,007
theorems across models, autograd, optimizers, floats and verification, while the Nous model surface
is four landed ℝ-generic modules plus two creation targets. "Same rigor as TorchLean or better" is
therefore not a claim about matching theorem count; it is a claim that each headline we land is *at
least as tightly bound to its executing artifact* as theirs. The comparison table and the gap list
below turn that into named gates.

## 1 · Reference identity

| Item | Value |
|---|---|
| Requested revision | `de3192df4779877ceb65cfe470a4d4ce9d480a5b` |
| Present and resolvable? | **Yes** — `git rev-parse` returns the pin; worktree checked out detached |
| Subject / date | "Fix FloatLib API documentation link", 2026-09-21 12:37:20 −0700 |
| Checkout at `/home/socol/Workspace/TorchLean` | `12f5c651f03b3890ec012d0a6bb45e3ea698c8d3` (`main`); the pin is a separate object |
| Working tree | clean; one prior worktree plus this read-only pinned worktree |

`formalization.yaml` records paper arXiv:2602.22631, `status.sorry_free: true`, toolchain
`leanprover/lean4:v4.34.0`, and `trust_boundaries.source_of_truth: docs/TRUST_BOUNDARIES.md` with
`lean_axioms: []`. The stale snapshot's `formalization.yaml` named a non-existent "lean axiom"
`Runtime.Autograd.Cuda.instNonemptyBuffer`; at the pin that name is gone and the list is empty.

## 2 · Scope

`formalization.yaml` (scope block) and `README.md:293-299`:

- **Included:** shape-indexed tensors/interfaces; model APIs; graph IR + denotation + graph/spec
  bridge; executable autograd, optimizers, training loops; IEEE-style finite-precision semantics;
  IBP/CROWN/spline/certificate verification; native CPU/CUDA entry points with documented FFI
  assumptions.
- **Excluded by the reference itself** (`formalization.yaml` `scope.excludes`): proof of CUDA,
  cuBLAS, cuFFT, compiler, driver or GPU-hardware correctness; correctness of external Python,
  Julia, PyTorch, Arb or CROWN producers; "a claim that every runnable model command is fully
  verified end to end".
- README trust statement: "TorchLean proves properties of explicit Lean definitions… None of these
  checks certifies the program that produced the certificate" (`README.md:293-299`).

Module inventory (Lean files): **1,459** total, **1,395** under `NN/`. Largest surfaces —
`NN/Runtime` 250, `NN/Proofs` 210, `NN/MLTheory` 156, `NN/Tensor` 132, `NN/Spec` 116, `NN/API` 103,
`NN/Examples` 100, `NN/Verification` 95, `NN/Tests` 95, `NN/GraphSpec` 31.

Structural drift vs `12f5c65`: `NN/Floats` fell 123 → 14 (the FloatLib dependency was extracted:
`formalization.yaml` `numeric_dependency` pins `lean-dojo/FloatLib@52ab504b…`); `NN/Tensor` (132),
`NN/IR` (10), `NN/Backend` (15) are new surfaces; the CUDA engine and `csrc/cuda` remain.

## 3 · Proof surface (quoted)

Headline autograd correctness, `NN/Proofs/Autograd/Tape/Core/Soundness.lean:353` (was `:349`):

```lean
theorem backprop_correct {ss : List Shape} (g : Graph Γ ss) :
    ∀ x dx seed,
      TensorPack.dotList (jvpCtx (ss := ss) g x dx) seed =
        TensorPack.dotList dx (backpropCtx (ss := ss) g x seed)
```

The adjacent analytic upgrade, `NN/Proofs/Autograd/Tape/Core/FDeriv.lean:1204` (was `:1264`):

```lean
theorem backpropVec_eq_adjoint_fderiv
    {ss : List Shape} (g : Graph Γ ss) (hg : GraphFDerivCorrect (Γ := Γ) g) :
    ∀ (xV : CtxVec Γ) (seedV : CtxVec (Γ ++ ss)),
      backpropVec (Γ := Γ) (ss := ss) g xV seedV
        =
      (fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g) xV).adjoint seedV
```

Both remain **conditional on a supplied per-node obligation**: `GraphFDerivCorrect` is a `Type`
(proof-relevant), defined structurally at `NN/Proofs/Autograd/Tape/Core/FDeriv.lean:987` as
`GraphFDerivCorrect g × NodeFDerivCorrect node`; the vectorized inner-product kernel is
`backprop_correct_inner` (`NN/Proofs/Autograd/Tape/Core/FDeriv.lean:870`, was `:933`). There is now
also a pointwise form `GraphFDerivCorrectAt`/`backpropVec_eq_adjoint_fderiv_at`
(`FDeriv.lean:1025,1365`) and an algebra-carrier restatement `backprop_correct`
(`NN/Proofs/Autograd/Tape/Algebra/Soundness.lean:386`). The global soundness statement still
inherits, per node, whatever local JVP/VJP law the operation provides — the same architecture as
Nous' `OpSpec` + per-op cotangent laws.

**New execution-binding headline**, `NN/Proofs/Autograd/Runtime/Link/BackwardDenseGraph.lean:389`
(quoted, abridged at the `∧`):

```lean
theorem backwardDenseAll_lowerGraphToTape_adjoint_fderiv {Δ : Type} {Γ ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (x : TorchLean.TensorPack ℝ Γ) (d0 : Δ)
    {τ : Shape} (output : Idx (Γ ++ ss) τ) (seed : Tensor ℝ τ)
    (hg : GraphFDerivCorrect (Γ := Γ) (toReal g d0)) :
    Runtime.Autograd.Tape.backwardDenseAll
        (t := (lowerGraphToTape (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1) output.i.1
        (Spec.SomeTensor.ofTensor seed) =
      .ok (TorchLean.TensorPack.toShapeErasedArray (α := ℝ) (ss := Γ ++ ss)
        (backpropAllCtx (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0
          (TensorPack.single output seed)))
    ∧ flattenCtx (TensorPack.takeLeft
        (backpropAllCtx (α := ℝ) g x d0 (TensorPack.single output seed)))
      = (fderiv ℝ (Proofs.Autograd.Graph.evalVec (toReal g d0))
          (flattenCtx x)).adjoint (flattenCtx (TensorPack.single output seed))
```

This is the strongest single execution link in the tree: the *executed* `Tape.backwardDenseAll` on a
tape lowered from a proof-carrying `Graph` returns `backpropAllCtx` of the seed, and over `ℝ` its
input block is the adjoint of the Fréchet derivative of the forward map. Its own trust doc bounds
it: "neither says anything about `Float` rounding, compiled code, or CUDA"
(`docs/TRUST_BOUNDARIES.md:31-33`). The typed runtime–tape equality remains at
`NN/Runtime/Autograd/TypedGraph/Core.lean:199` (was `:118`):

```lean
theorem backwardDenseAllFrom_lowerToTape_eq_backpropAllCtx
    (g : GraphData α Γ ss) (x : TorchLean.TensorPack α Γ) (output : Idx (Γ ++ ss) τ)
    (seed : Tensor α τ) :
    backwardDenseAllFrom (lowerToTape g x).1 output seed =
      .ok (TorchLean.TensorPack.toShapeErasedArray
            (GraphData.backpropAllCtx g x () (TensorPack.single output seed)))
```

with the same docstring caveat: "states fidelity to the executable VJP stored in `GraphData`;
derivative correctness requires the separate local laws carried by `Proofs.Autograd.Algebra.Node`"
(`NN/Runtime/Autograd/TypedGraph/Core.lean:194-198`).

New attention VJP chain, `NN/Proofs/Autograd/Tape/Ops/Attention/SpecBridge.lean`: `:288`
`backpropVec_eq_adjoint_fderiv_sdpaSpec`, `:307`
`backpropVec_eq_adjoint_fderiv_scaledDotProductAttention`, `:319`
`…_allTrueMask` — the tape backprop seeded on the output block equals the adjoint of the Fréchet
derivative of `Spec.scaledDotProductAttention`. This is the nearest analogue on the reference side
to the Nous attention target, but it is stated about the *Lean* forward map, not an extracted body.

Robustness/margin (closest to a decision-margin result),
`NN/MLTheory/Proofs/Verification/Robustness/LipschitzCertified.lean`: `:164`
`argmaxClassifier_eq_of_linf_distance_lt_half_margin`, `:192`
`argmaxClassifier_eq_of_hasLogitMargin`, `:213`
`is_certified_robust_of_lipschitz_of_logitMargin` (were `:185,213,234`). These bound a *classifier's*
decision under an L∞ perturbation; they are not a per-operation `ε` ledger composed through a layer
fold.

Totals: **3,007** `theorem`/`lemma` declarations and **10,060** `def`/`abbrev` declarations
(line-anchored counts; were 2,497 and 8,240 at the snapshot).

## 4 · Hygiene

Counts are `rg`-derived at `de3192df` and are an `anchor` for the mechanism, `cited` where a line is
quoted. They are not kernel-checked.

| Mechanism | Count | Location / note |
|---|---|---|
| `sorry` declarations | **0** | 2 word-hits, both non-proof: `home_page/blueprint/TorchLeanBlueprint/Guide/Ch1_Introduction/TheLeanLanguage.lean:960` (prose), `NN/Tensor/Internal/Elab/Einsum/Index.lean:554` (a diagnostic string) |
| `admit` declarations | **0** | 4 word-hits, all prose: e.g. `NN/Spec/Layers/Attention.lean:135`, `NN/Backend/ContractCheck.lean:98` |
| `axiom` declarations | **0** | `scripts/checks/repo_lint.py:66` `ALLOWED_AXIOMS: dict[str,set[str]] = {}`; lint rejects any non-allowlisted `axiom` (`:3109-3120`) |
| custom Lean axioms | **0** | `formalization.yaml` (`trust_boundaries.lean_axioms: []`); `docs/TRUST_BOUNDARIES.md:63-79` |
| `native_decide` | **0** | banned by `scripts/checks/repo_lint.py:2096` |
| `set_option maxHeartbeats` | **0** | banned by `scripts/checks/repo_lint.py:2499-2500` |
| `unsafe` declarations | **1** | `NN/Runtime/Autograd/Torch/Core/Session/State.lean:62` `private unsafe def storageKeyImpl`; the remaining 29 `unsafe` word-hits are prose |
| FFI externs (`extern "`) | **125** (121 in `NN/`) | CUDA: `Buffer.lean` 67, `Kernels.lean` 34, `ConvPool.lean` 10, `DGemm.lean` 1; CPU native: `NN/Tensor/Internal/Elab/Native/Pointwise.lean` 6, `Transpose.lean` 2, `NN/Tensor/Internal/Representation/Storage.lean` 1; blueprint 4. CUDA entry points use the multi-attribute `@[never_extract, extern "…"]` form (111 of the 121 NN externs). `@[extern` *as the first attribute* matches only 12; the D4 pattern under-counts the pin |
| `@[implemented_by]` | **30** | `NN/Tensor/Internal/Representation/Storage.lean` (ByteArray/FloatArray fast paths, `:554-809`), `NN/Spec/Core/Tensor/Factorizations.lean:157,253,282`, `NN/Tensor/Internal/Representation/Basic/Pointwise.lean:45,62,84,313`, `NN/Runtime/Autograd/Torch/Core/Session/State.lean:72` |
| `opaque` declarations | **181** | CUDA trust boundary: `Buffer.lean` 49, `Kernels.lean` 34, `ConvPool.lean` 10; plus non-FFI opaques (`docs/TRUST_BOUNDARIES.md:184` "Opaque Non-FFI Declarations") |
| `#print axioms` | **10** | only in `home_page/blueprint/TorchLeanBlueprint/Guide` prose (e.g. `home_page/blueprint/TorchLeanBlueprint/Guide/Ch1_Introduction/TheLeanLanguage.lean:952`); no CI or lint gate reads them |
| `#audit` / cone gate | **0** | none — no `Lean.collectAxioms`-based gate anywhere |

**Trusted declaration.** The CUDA handle type is carried by `Runtime.Autograd.Cuda.BufferImpl`, an
`opaque NonemptyType` at `NN/Runtime/Autograd/Engine/Cuda/Trusted.lean:117`, with
`instance : Nonempty Buffer := BufferImpl.property` (`:128`). `docs/TRUST_BOUNDARIES.md:63-70` states
this is data-carrier-derived, not an asserted proposition.

**Axiom cone.** No `#print axioms`/`#audit` gate exists in CI or the lint suite, so the transitive
cone is **not mechanised**. Grep establishes the *absence* of `axiom` declarations (strong), but not
the cone of any headline (e.g. `Classical.choice` from `fderiv`, `propext`, `Quot.sound`).
Inference "core proofs use only the three Mathlib standard axioms" is an `anchor`, not evidence.
This is unchanged from the snapshot.

## 5 · Execution binding / trust boundary

Four distinct execution relations, none of which is a source extraction:

1. **Lean-compiled eager runtime.** The eager runtime executes Lean bodies; the executed sweep is
   linked to the algebra by `backwardDenseAll_lowerGraphToTape_adjoint_fderiv`
   (`NN/Proofs/Autograd/Runtime/Link/BackwardDenseGraph.lean:389`) and by the typed-graph equality
   (`NN/Runtime/Autograd/TypedGraph/Core.lean:199`).
2. **`@[implemented_by]`** (30 hooks). Cholesky/triangular/ridge solve
   (`NN/Spec/Core/Tensor/Factorizations.lean:157,253,282`), the `ByteArray`/`FloatArray` buffer fast
   paths (`NN/Tensor/Internal/Representation/Storage.lean:554-809`), pointwise fast paths, and the
   parameter-alias lookup key (`NN/Runtime/Autograd/Torch/Core/Session/State.lean:62,72`).
   `docs/TRUST_BOUNDARIES.md:115-135` states the logical theorems do **not** transfer to the compiled
   body without a separate agreement argument; the alias-key replacement additionally depends on
   stable reference addresses and is covered by regression tests, not a proof.
3. **FFI / native CPU.** 9 CPU externs in `NN/Tensor/Internal/Elab/Native/{Pointwise,Transpose}.lean`
   and `NN/Tensor/Internal/Representation/Storage.lean`; each has a proof-visible reference body
   (`Pointwise.lean:26`).
4. **FFI / CUDA.** 112 CUDA externs (`Buffer` 67, `Kernels` 34, `ConvPool` 10, `DGemm` 1) backed by
   `csrc/cuda/**` translation units, marked `@[never_extract]`, and explicitly outside the kernel:
   "Lean does not prove the compiled CUDA binary correct… memory safety, launch behavior, compiler
   behavior, and floating-point behavior of native CUDA/C/C++ code are outside Lean's proof kernel"
   (`NN/Runtime/Autograd/Engine/Cuda/Trusted.lean:24-45`, `docs/TRUST_BOUNDARIES.md:191-293`,
   `formalization.yaml` `trust_boundaries.native_runtime`).

Trust-boundary documents at the pin: `docs/TRUST_BOUNDARIES.md` (moved from the repository root
`TRUST_BOUNDARIES.md`, which no longer exists; sections `Levels of Assurance:11`, `Lean Axioms and
Hidden Implementations:63`, `Prop-Valued Contracts:81`, `Executable Replacements:115`, `Native Host
Tensors:136`, `Native Host Allocation:163`, `Opaque Non-FFI Declarations:184`, `CUDA Runtime:191`,
`Executable Floating Point:294`, `External Numeric Oracles:409`), plus `README.md:293-299` and the
`formalization.yaml` `trust_boundaries` block.

There is no Aeneas/Charon extraction of a Rust artifact; there is no Rust. The relation to executed
native code is a **documented assumption**, not a refinement theorem. This is the categorical gap
versus Apeiron.

## 6 · Comparison against Apeiron's standard

| Axis | TorchLean (`de3192df`) | Apeiron (`74c037b`) |
|---|---|---|
| Refinement of the **executing** artifact | Theorems about Lean defs; the executed tape is linked to the algebra *and* to the Fréchet adjoint (`NN/Proofs/Autograd/Runtime/Link/BackwardDenseGraph.lean:389`); native path outside the kernel (`docs/TRUST_BOUNDARIES.md:31-33,191-293`). No extraction. | Charon/Aeneas extracts the real bodies by `#[path]` (`proof/extract/lib.rs:14-30`); drift gate `proof/regen.sh --check` diffs regenerated `proof/ApeironWord.lean`; `word_mul_denote`/`word_norm_denote` (`proof/corpus/WordRefinement.lean:45,56`) equate the extracted body to canonical `ringMul`/`ringNormForm`. |
| Axiom discipline | 0 `axiom` decls, lint-enforced (`scripts/checks/repo_lint.py:66,3109-3120`); **no cone gate**; cone unverified. | `proof/corpus/AxiomAudit.lean` `#audit` 1,070 headlines against `{propext, Classical.choice, Quot.sound}` with `throwError` on deviation (`proof/corpus/AxiomAudit.lean:63-84`); Slang chain keeps a separate empty cone (`tools/slang/zero_axioms.py`). |
| Controls | None found. CI = repo lint + `lake build NN NNCI NNExamples NNTests …` + `lake test` + native command smoke (`.github/workflows/ci.yml:85-122`); `scripts/checks/check.sh` = build/test/lint. | Source-mutation controls in a throwaway copy: `proof/controls/relocation.sh` (baseline/red/restore, per-step timeouts, lane lock); receipt `docs/evidence/relocation-controls.md:1-40`. |
| Error / margin composition | Perturbation margin on a classifier (`NN/MLTheory/Proofs/Verification/Robustness/LipschitzCertified.lean:164,192,213`). No per-op `ε` ledger folded through layers. | `proof/corpus/Nous/Model/ErrorComposition.lean:127` `norm_execComp_sub_refComp_le`, `:218` `strictWinner_perturb`, `:256` `decision_margin_preserved`: a reference decision gap `> 2·(lip·budget)` selects the same argmax under the executing fold. |
| Model-level mechanisms | Full model zoo (`NN/Examples` 100 files), transformer VJP chain including SDPA (`NN/Proofs/Autograd/Tape/Ops/Attention/SpecBridge.lean:288,307,319`), FlashAttention denotation (`NN/Spec/Layers/FlashAttention.lean`), optimizers. | `proof/corpus/Nous/Model/{Model,Decomposition,Weighting,State,Memory}.lean`: `run_eq`, `run_state_eq_tokenFold`, `hyperbolic_decomposition`, `channel_diagonal_fold`, `channelWeight_eq`. Attention spinor scans are ℝ-valued in `proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean`. |

Net: **refinement and controls are strictly stronger here; axiom discipline is stronger in
mechanism, weaker in breadth of registration.** Breadth of proved model surface is far larger there
and is not the standard being matched.

## 7 · Gap list — what Nous must state, and its gate

Mapped to the creation targets and calibrated on our own receipts
(`docs/epics/model-algebra.md:106-112`, `docs/epics/inference-parity.md:33-41`).

| # | Gap to close | Artifact | Gate | Control | Slices |
|---|---|---|---|---|---|
| G1 | **CLOSED** — `Nous.Model.attention_channel_form` proved (`proof/corpus/Nous/Model/Attention.lean`, `ecdd61fe…`, 301 lines): the running `(Σ wᵢ, Σ wᵢvᵢ)` scan equals `Σ (wᵢ/Σ wⱼ)vᵢ` and the factorized-score form under `ChannelFormDomain` (inhabited; joint witness `channel_form_domain_witness`). The correspondence to the ℝ scans is a stated reading, not a refinement. | `proof/corpus/Nous/Model/Attention.lean` | `cd proof/corpus && lake build Nous.Model.Attention` (exit 0, 3052 jobs); `AxiomAudit` registration pending the barrier | drop normalizer → red (attributed) | 0 |
| G2 | **CLOSED** — `Nous/Model/Instances.lean` (`c958bed3…`, 346 lines) fixes the hyperbolic/parabolic/elliptic faces over `ℤ` and the `ZMod (2^64)` word face via `denoteSigma`, with the 2-adic caveat explicit (`zmod_two_not_unit`); ternary excluded as `anchor` (no extracted kernel). Gate 3355 jobs; wrong-σ control red; 29 headlines registered. | `proof/corpus/Nous/Model/Instances.lean` | `cd proof/corpus && lake build Nous.Model.Instances AxiomAudit` (exit 0) | wrong σ → instance goal red | 0 |
| G3 | Per-kernel executing-body refinement for engine/numeric kernels | `proof/extract/lib.rs` roots + `proof/corpus/Numeric/ExpRefinement.lean`, `proof/corpus/Numeric/RsqrtRefinement.lean`, `proof/corpus/Numeric/RopeRefinement.lean`, `proof/corpus/Numeric/FormatRefinement.lean` | `proof/regen.sh --check`; `lake build Numeric.<X> AxiomAudit` | perturb coefficient → bound red | 1–2 per kernel |
| G4 | Written plan for the abstraction (currently "to be written") | `docs/epics/model-algebra.md` (creation target) | path resolution + `git diff --check` | n/a (planning) | 1 |
| G5 | **CLOSED** — the original `channelWeight_binary_softmax` phrasing is declined; the repaired log-substitution identity is proved in `Nous/Model/Weighting.lean` (`e2ebfac4…`, 194): `ratio_eq_exp_log`, `ratio_eq_softmax2_log`, `channelWeight_binary_softmax` (domain `0 < u`, `0 < l`). Gate 2974 jobs; control red. `AxiomAudit` registration pending the barrier. | `proof/corpus/Nous/Model/Weighting.lean` | `lake build Nous.Model.Weighting` (exit 0) | remove `0 < u` → unprovable | 0 |
| G6 | ~~Register the landed-but-unregistered attention headline in the cone~~ **CLOSED** — `#audit Sanguine.AttentionSpinor.softmax_attention_is_online_spinor` is present at `proof/corpus/AxiomAudit.lean:1372`; `AxiomAudit.lean` now carries 1,070 `#audit` lines. The `001ccab5…` cited by the lead is the sha256 of Apeiron's `proof/corpus/AxiomAudit.lean` (an Apeiron artifact digest), not a TorchLean git object; the registration is present in the working tree and audited. | `proof/corpus/AxiomAudit.lean:1372` | `cd proof/corpus && lake build AxiomAudit` | inject `sorry` → build red | 0 |
| G7 | Plan must carry the landed composed-margin module and its numeric control | `docs/epics/model-algebra.md` (creation target) row for `proof/corpus/Nous/Model/ErrorComposition.lean`; add `decision_margin_preserved` (`:256`) to the barrier | `lake build Nous.Model.ErrorComposition` + control | perturb a per-layer `ε` → refusal/decision red | within G4 |

To **match** the baseline, only **G3** remains a proof obligation (engine-kernel executing-body
refinements); G4 and G7 are registration and planning; G1, G2, G5 and G6 are closed. To **exceed** it, each must additionally remain (a) an extracted-body
refinement rather than a model-only statement, (b) inside the audited closed cone, and (c) paired
with a mutation control — all three of which the Nous surface already satisfies for its landed
modules (`proof/corpus/Nous/Model/ErrorComposition.lean:47-52` states its per-layer hypotheses as an
explicit trust boundary; the word face is registered in `proof/corpus/AxiomAudit.lean:88-89`).

## 8 · Trust boundary and residual

- TorchLean claims here are read from its source at `de3192df`; none is kernel-checked by this task.
  Grades: `cited` where a theorem/line is quoted, `anchor` for counts and inferences.
- The pinned worktree is **left in place** for the reader at
  `/home/socol/oc-tmp/torchlean-de3192df`; it is **not built** (no `.lake`/`build`). No build would
  be "clearly cheap and useful" for this comparison, since the claim is about the stated proof
  surface and its documented boundary.
- Hygiene counts are `rg`-derived; the fold pattern is named per row because the CUDA externs use the
  multi-attribute `@[never_extract, extern]` form. A transitive axiom-cone check for TorchLean would
  need an unbounded `lake build` and is out of scope. The absence of `axiom`/`sorry` declarations is
  established; the cone of their Mathlib-core proofs is not. The absence of a cone gate is
  established by the absent `#audit`-equivalent and the blueprint-only `#print axioms` lines.
- G6 closed; G5 corrected against source. G1–G4 and G7 remain open; this receipt raises no claim
  grade for TorchLean and changes none for Apeiron.

---

## Appendix A — snapshot `12f5c651…` findings (superseded)

The D4 receipt read `/home/socol/Workspace/TorchLean` at
`12f5c651f03b3890ec012d0a6bb45e3ea698c8d3` ("Small complex fix", 2026-08-27), toolchain
`leanprover/lean4:v4.33.0`, because `de3192df…` was then absent from the checkout
(`git cat-file -t` → `fatal: could not get object info`). Those findings are retained here for the
delta; do not cite them as current.

- **Requested revision:** `de3192df…`; **checkout HEAD:** `12f5c651…`; tree did not contain the pin.
- **Module counts:** 1,263 Lean files, 1,201 under `NN/`; largest surfaces `NN/Runtime` 207,
  `NN/Proofs` 162, `NN/MLTheory` 134, `NN/Floats` 123, `NN/Spec` 115, `NN/API` 105,
  `NN/Examples` 110, `NN/Verification` 88.
- **Headline autograd:** `backprop_correct` (`NN/Proofs/Autograd/Tape/Core/Soundness.lean:349`),
  `backpropVec_eq_adjoint_fderiv` (`NN/Proofs/Autograd/Tape/Core/FDeriv.lean:1264`),
  `GraphFDerivCorrect` (`FDeriv.lean:1049-1051`), `backprop_correct_inner` (`FDeriv.lean:933`),
  `backwardDenseAllFrom_lowerToTape_eq_backpropAllCtx`
  (`NN/Runtime/Autograd/TypedGraph/Core.lean:118`).
- **Headline robustness:** `argmaxClassifier_eq_of_linf_distance_lt_half_margin`
  (`NN/MLTheory/Proofs/Verification/Robustness/LipschitzCertified.lean:185`),
  `argmaxClassifier_eq_of_hasLogitMargin` (`:213`),
  `is_certified_robust_of_lipschitz_of_logitMargin` (`:234`); plus
  `fp32_le_of_real_le_sub_margin` (`NN/Proofs/RuntimeApprox/FP32/CROWN.lean:72`).
- **Totals:** 2,497 `theorem`/`lemma`, 8,240 `def`/`abbrev`.
- **Hygiene:** `sorry` 0 declarations (prose `NN/Examples/Factorization.lean:19`); `admit` 0
  declarations (prose `NN/Spec/Layers/Attention.lean:131`); `axiom` 0
  (`scripts/checks/repo_lint.py:37` `ALLOWED_AXIOMS = {}`); `native_decide` 0; `maxHeartbeats` 0;
  `unsafe` 0; `opaque` 132; `@[extern "…"]` 109 (`NN/Runtime/Autograd/Engine/Cuda/Buffer.lean` 62,
  `Kernels.lean` 33, `ConvPool.lean` 10, `DGemm.lean` 1, plus 3 blueprint); `@[implemented_by]` 3
  (`NN/Spec/Core/Tensor/Factorizations.lean:151,242,300`); `#print axioms`/cone gate 0.
- **Prior drift note:** `formalization.yaml` then listed a "lean axiom"
  `Runtime.Autograd.Cuda.instNonemptyBuffer` that occurred in no Lean file; the real declaration was
  `Runtime.Autograd.Cuda.BufferImpl` (`NN/Runtime/Autograd/Engine/Cuda/Trusted.lean:41`) with
  `Nonempty Buffer` as `BufferImpl.property` (`:48`). At the pin the stale name is gone and
  `lean_axioms: []`.
- **Prior execution binding:** three relations (Lean-compiled defs; `@[implemented_by]`;
  FFI/native CUDA at `TRUST_BOUNDARIES.md:102-103`, root path, now `docs/TRUST_BOUNDARIES.md`).
- **Prior verdict:** Apeiron exceeds TorchLean on executing-artifact refinement and mechanised cone;
  reference had no extraction, no cone gate, no mutation control; surface gap large.
