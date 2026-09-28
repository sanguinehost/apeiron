# Resource and security continuation

- Owner: Astra boundary spike `APEIRON-OPEN-SPIKE-boundary-design`; limit 20 minutes.
- Base: `262bb0a94eff2cc7d5655122a954f44d74de96b6`, plus accepted dirty identity
  `target/open-areas/baseline.json`; worktree `worktree-agent-APEIRON-OPEN-SPIKE-boundary-design`.
- Owned path: this file. Build lane: none; no Rust or Lean build runs in this spike.
- Grade: `anchor`. Decisions and source findings below establish no new conformance or security proof.
- Question: smallest executable task at each boundary, and what prevents actual closure?

```text
SCOPE: docs/epics/resource-security-continuation.md; bounded resource/security decisions and tasks.
REGISTER: planning/refinement; excludes production, backend, canonical-source and pin changes.
CORPUS: ringComplementary_idem requires root, inverse and half witnesses; old otp_perfect_secrecy
        says only forall c m, exists unique k, m + k = c. Source grades retained; no new proof.
REFERENCE: /home/socol/Workspace/apeiron/target/aeneas-model/lean/Aeneas/Std/Vec.lean;
           /home/socol/Workspace/sanguine/proof/Substrate/Algebra/Trinity/RecoveryTrinity.lean.
GATE: git diff --check -- docs/epics/resource-security-continuation.md,
      cwd assigned worktree. Semantic gates MISSING/unrun as specified below; no invented red.
GRADE: anchor before/after; no runtime measurement or proof build in a documentation spike.
```

## Resource decision

Proceed with the standalone same-input relation already selected in
[allocator-may-fail](allocator-model.md). Select a backend effect as the target architecture
for actual allocator choice, subject to the capacity and effect feasibility gates below. Retain the
existing production reservation path. An explicit Boolean relation is a model only. Existing
feature-gated failure injection supplies Rust branch/ordering tests; it does not prove the global
allocator returns the injected result. No public arena or new reservation-capability API is selected.

Source findings on this identity:

| Artifact | Read result | Consequence |
|---|---|---|
| Active `target/aeneas-model/lean/Aeneas/Std/Vec.lean:35-39,101-117` | `Vec` has `slice` and `cap`; `try_reserve` succeeds whenever `length + additional ≤ Usize.max`, sets `max cap need`, otherwise returns `.Err`. | Capacity exists. Ordinary refusal on a valid growing request is absent. `Usize` arithmetic is not a layout check. |
| Active `Vec.lean:203-210,450-451,453-466` | `push` calls `Vec.from`, which sets capacity to new length; `extend_from_slice` also reconstructs via `.from`; `with_capacity` sets its field to the request. | A reserved empty vector with cap 8 followed by one push has model cap 1. Correct length alone cannot justify receipt attachment or no further allocation. This is a source-derived counterexample, not a run. |
| Active `Primitives.lean:66-75`; `WP.lean:22-31,873-920` | Sole effect is `fail`, response `PEmpty`; `spec` accepts only `ok`; `dspec` additionally accepts divergence. `mvcgen` handles other effects as a dummy exception. | Add a returning allocator effect and its semantics; merely extending `TryReserveError` or `.vis` is insufficient. |
| Active `Vec.lean:463` | `extend_from_slice` eliminates every effect response with `nomatch`. | Returning effects require changing this eliminator as well as WP. Direct response/eliminator search finds `Primitives`, `WP`, `Vec`; broader transitive proof consumers still require build inventory. |
| `proof/ApeironWord.lean:9055-9143` | `materialize_rebuild`, `prepare_from_plan`, `prepare_rebuild`, reservation and capacity observations are generated. Locator failure returns the modified `self1`. | Historical blanket claims that these bodies are unextracted are stale. Generated presence is not adapter refinement or fresh extraction acceptance. |
| `proof/corpus/Growable/Resources.lean:704-760` | `ResourceAdapterSimulates.refused` takes `committedAfter = state` and zero counters as premises. | Projection theorem `resourceAdapter_refusal_stutter` supplies no execution trace or proof of the premise. |
| `src/table/locator.rs:403-484` | Before later failures, `reserve_path` can set `directory := Some`, reserve link capacity, and initialize a profile. Arena rollback preserves committed contents, not necessarily all representation fields/capacities. | Whole generated-table equality may be stronger than production stutter. Sol must decide the exact observation relation or require restoration; do not silently weaken the contract. |
| `src/table/tests.rs:571-635` | The final-reservation test uses `usize::MAX` in the last shard and returns `SizeOverflow`. | This is not same-valid-layout allocator refusal. |
| `src/coord/locator_core.rs:337-375,1169-1178`; `src/table/locator.rs:1090-1160` | An `injection` late-edge seam exists in the real reservation path; the table test calls actual preparation but uses a synthetic plan and checks selected observations, without clone/merge counters. | Reuse and strengthen this seam. Do not call it ordinary allocator conformance or unconditional public-path coverage. |

The durable backend edit surface is `proof/patches/aeneas-lean-model.patch`, materialized by
`proof/model-patches.sh`, selected by `proof/lakefile.lean`. Installed `target/aeneas*` trees are
build products; do not edit them as source. The model patch currently spans Vec and three Core
modules. A new effect must specify whether its response is refusal or successful capacity, preserve
contents on both outcomes, require valid layout, distinguish no-growth and zero-sized cases, and
quantify over every admitted response. A success capacity may exceed the request; identifying it
with `max old need` is an abstraction requiring a capacity simulation, not exact allocator behavior.

### Smallest resource tasks

Each row owns at most three files. Future file names and commands are proposed artifacts, not claims
that a gate already exists. Before dispatch, record exact base, worktree, owner and build lane. Sol
ratifies new theorem premises; the lead serializes `lakefile.lean`/`AxiomAudit.lean` ownership with
other proof work. Use a new `Resource/` directory to avoid adding another immediate `Growable/` file.

| Task / owner | Owned files and completion | One focused gate / semantic control |
|---|---|---|
| `APEIRON-RESOURCE-W1-T1` / Sol | `proof/corpus/Resource/AllocatorModel.lean`, `proof/corpus/lakefile.lean`, `proof/corpus/AxiomAudit.lean`. Define contents/capacity invariant, explicit byte layout and relation. For one fixed nonzero-size byte vector `([],0)` and additional `1`, prove valid layout, growth, unchanged contents, success cap ≥1 and refusal stutter jointly. Keep production simulation absent. | Bounded `lake build Resource.AllocatorModel AxiomAudit`. In a throwaway copy require overflow for refusal: the same-input theorem must fail semantically; restored proof passes. Grade `model`, kernel-checked relation only. |
| `APEIRON-RESOURCE-W1-T2` / Flash after Sol fixture verdict | `src/table/locator.rs`, `src/table/tests.rs`, new `proof/controls/resource-refusal.sh`. Strengthen the existing late-edge test with counted hold/clone/merge observations and first-use plus already-initialized locator cases; run success and forced refusal from equivalent small valid states. Record logical contents, staging and capacity changes separately. | New runner invokes `cargo test --locked --features injection --lib locator_refusal` in baseline/mutant/restored copies. Mutation moves a real hold/clone before the late reservation; assert it is detected. Runner MISSING until authored. Grade `tested`; no allocator proof. |
| `APEIRON-RESOURCE-W2-T1` / Sol | New `proof/corpus/Resource/CapacityProbe.lean`, `proof/corpus/lakefile.lean`, `proof/corpus/AxiomAudit.lean`. State active-model facts for reserve→push, initialization, truncate/clear, insert/extend and receipt observations. First prove the concrete cap-shrink counterexample; specify desired preservation separately. | Bounded `lake build Resource.CapacityProbe AxiomAudit`; reserve 8 then push 1 must expose cap 1 in the current model. A throwaway capacity-preserving mutant must invalidate that counterexample. This diagnoses the active semantics, not production correctness. |
| `APEIRON-RESOURCE-W2-T2` / Astra + Sol | `docs/epics/allocator-model.md`, `docs/epics/allocator-model.md`, `docs/epics/allocator-model.md`. Freeze capacity simulation, layout/ZST domain, returning-effect response and WP interpretation; inventory all affected constructors, handlers, tactics and generated proof dependencies. Decide exact-state restoration versus an explicit committed-observation relation using the preceding fixture. | Complete-file/diff review; implementation gate MISSING until the bounded prototype and its runner are specified. No backend patch is dispatched merely from this row. |

The byte witness may instantiate an explicit 64-bit layout domain with element size/alignment 1,
`need = 1`, and bytes ≤ `2^63 - 1`; it must name that domain, not infer Rust layout from Lean's
abstract type `α`. Generic layout fidelity requires the actual target type layout and Rust
`Layout::array` correspondence. A 32-bit or zero-sized population remains a separately named
obligation, not silently excluded from a generic theorem.

After capacity/effect feasibility, the integration order is durable patch and its rejecting model
controls → regeneration → generated adapter branch lemmas → source-linked effect trace/certificates
→ construction of both `ResourceAdapterSimulates` branches → public insert/build composition.
Split every implementation slice to ≤3 files; extraction/generated output and shared caches are
serial lanes. No production conformance gate exists yet. Its required runner must compose real-source
freshness, bounded adapter proof/audit and a real-body early-hold or lost-refusal mutant; a grep,
synthetic tuple or passing abstract oracle cannot be that runner.

The prepared proof must bind the exact actual plan, capacities, initialized storage, held carrier,
locator receipt and published tuple. The refused proof must derive stutter and zero effects from the
actual execution. Callback counters are not present in the current Aeneas result; a separately
specified source-linked trace/ghost instrumentation or equivalent execution theorem is necessary.
Arbitrary callback panic/allocation, global allocator realization, executor, codegen, ABI and GPU
remain outside existing evidence; none waives source-level ordering or capacity correctness.

## Security decision

The canonical corpus has the algebra but no canonical security/protocol substrate matching the old
headlines. This is established by the pinned Git tree and the named theorem searches, not by assuming
all security-related content is absent: `Apeiron/Capability/Basic.lean` is present, and its own scope
excludes real compiler grammar correspondence. The missing bridge is semantic, not a missing import.

Read source identities: old Sanguine HEAD `e830d3fbdeb40319d1be0d1fc803e706748b6a3c`; canonical pin
`3d66ea3aa2fe1de5997b6b757247454a2f771358`. Canonical worktree is dirty; pin-specific statements below
were reread with `git show <pin>:proof/Apeiron/...`. Do not import dirty sibling definitions.

| Primary statement | Exact available content | Required counterhypothesis / missing link |
|---|---|---|
| Old `NullCell/NullCell.lean`, `NCell`, `nmul`, `nj_sq` | `ℝ × ℝ`, split multiplication and `j²=1`. Canonical `Basic.zeroDivisorPair_of_disc_one` supplies the split zero-divisor pair; `nilpotent_of_disc_zero` supplies a different square-zero witness. | Preserve exact transport, not a word-ring real classification. Generic collision or a dropped coordinate alone does not identify σ=0; hyperbolic channel projections also discard information. The existing plan's broad collapse→parabolic rule needs Sol attribution review. |
| Canonical `IdempotentsAnyRing.ringComplementary_idem` | Premises `r*r=σ`, `r*s=1`, `2*h=1`. `ChannelsAnyRing.ringUpper_channel` / `ringLower_channel` require `r^2=σ`. | Joint witnesses over the selected scalar are mandatory. `2*h=1` cannot hold in the wrapping word ring. No secrecy follows from complementary projections alone. |
| Old `RecoveryTrinity.otp_perfect_secrecy` | `∀ c m, ∃! k, m+k=c` for any additive commutative group; `keyed k` is itself an equivalence. | Unique compatible keys do not specify their distribution. Finite full-support biased pad countermodel (weights 3:1 on two keys) keeps unique fibers but changes ciphertext probabilities by plaintext. Require independent uniform fresh key plus observer experiment for a secrecy headline. |
| Old `EncryptionProjection.cipher_fiber_is_a_singleton`; `SanguineSecurity.cipher_secrecy_is_only_cost` | `(f ⁻¹' {f a₀}).ncard=1` under fixed `Injective f`. | Fixed-key injectivity does not characterize the observer who lacks the key. Compare the family `keyed k` with the public joint map `(m,k) ↦ m+k`; distinguish their domains before any cost/secrecy verdict. |
| Old `EncryptionProjection.decryptRecovery` | Construction takes the target plaintext `a₀`; its residual has unique zero there. Source marks AES/RSA bridge “named, not re-proven”. | A unique minimum with supplied target supplies no algorithm finding an unknown target, complexity bound or standard-specific attack. Require an executable reduction and stated cost model; no standard-break conclusion. |
| Old `SecurityCore.xorStream` | Empty key passes through the remaining data; involution remains true. | A short-key fixture must expose the unchanged suffix. Round-trip does not imply full coverage or secrecy. |
| Old `Federation.honesty_decided_at_the_cell` / `law_violation_is_detectable` | Universal intertwining iff all one-cell probes agree; detection assumes a supplied violating pair. | A finite sampled detector can miss an unqueried bad pair. No protocol, delivery/partition model, exhaustive decision procedure or adversarial-completeness theorem follows. |
| Old `Federation.federation_reseeds_the_genome` | Decode a survivor's component of a diagonal copy using the supplied codec. | Need deployment distribution, survivor availability/authenticity, corruption and network failure assumptions; not Byzantine recovery merely from a projection equation. |

These are source-grounded tests of interpretations. They do not refute the exact old algebraic
statements or silently change their native/riding grades. Their proposed finite countermodels need
kernel checks before promotion. No SEC theorem, gate, implementation standard or crypto repository
is fabricated. Target standards and trusted reference revisions remain unselected.

### Smallest security tasks and prerequisites

The current user request permits this bounded review alongside storage work. The existing rebase
plan still gates canonical transport behind storage closure and lead opening; this spike does not
claim either. It also lacks a separate specification and has transport rows with compound gates and
an oversized prose row. Those are dispatch blockers, not reasons to abandon the security outcome.

| Task / owner | Owned paths and completion | Gate / control |
|---|---|---|
| `APEIRON-SECURITY-W1-T1` / Sol | New `docs/epics/security-algebra-rebase.md`, existing `docs/epics/security-algebra-rebase.md`, existing `docs/epics/security-algebra-rebase.md`. Freeze exact statements/grades, selected scalar domains, observer/key model, confirm/refute pairs; correct only source-supported attribution errors. Split every later row to ≤3 paths and one runner. | Complete-file/diff review against pinned source; reject a fixture inferring secrecy from compatible-key existence or forcing every projection onto σ=0. Documentary control only, no security acceptance. |
| `APEIRON-SECURITY-W2-T1` / Sol after spec | New `proof/corpus/Security/InterpretationWitnesses.lean`, `proof/corpus/lakefile.lean`, `proof/corpus/AxiomAudit.lean`. Kernel-check finite biased-pad and fixed-key/public-map witnesses plus short-key passthrough using pinned definitions or separately checked statement transport. | Proposed bounded `lake build Security.InterpretationWitnesses AxiomAudit`; change the biased weights to equal weights in a disposable copy and require the named bias witness to fail. Grade `model`; not a theorem about a deployed cipher. |
| `APEIRON-SECURITY-W2-T2` / corpus owner, conditional | At most one new canonical face-transport module plus the corpus's audit/root registration (≤3 actual files selected after reading that repository's AGENTS). Transport the exact `NCell` multiplication and split/idempotent observations. | Canonical package's bounded build/audit runner, MISSING until its local lifecycle selects the module. Mutate the split parameter to zero and require the orthogonal/complementary witness to reject. Do not silently add the missing half premise. |

The first executable security proof is the finite interpretation-witness task after its specification,
not a security-standard implementation. Later OTP security needs a probability/observation contract;
federation needs a current joint protocol profile; AES/RSA/SHA work needs selected standards, retrieved
reference implementations, exact reductions or documented residuals, empirical vectors, source
extraction and both semantic and implementation controls. Absence of these is a finite prerequisite
list, not a waiver of the end-to-end goal.

## Cross-repository and integration boundary

Canonical generic mathematics remains in the pinned corpus until a separately gated migration into
Apeiron. Transporting old Sanguine statements to that corpus requires its owner and local gates;
consuming a new release requires a separate three-file pin task (`proof/corpus/lakefile.lean`,
`proof/corpus/lake-manifest.json`, `proof/README.md`) and fresh bridge acceptance. No migration,
canonical edit, commit, pin change or secondary crypto repository is performed here.

Song stays an independent implementation. A future security consumer must share a checked canonical
specification through explicit source-connected correspondence, matched scalar/domain, state,
admission/refusal, resources and observer assumptions. Existing `Joint.WordBridge` map transport
cannot supply cryptographic, protocol or allocator correspondence. Slang retains its empty cone;
Rust/canonical proofs retain `{propext, Classical.choice, Quot.sound}`. Security, allocator, size,
performance and full integration gates remain separate and jointly required where applicable.

## Identity and acceptance

| Artifact | SHA-256 |
|---|---|
| Active `Vec.lean` | `5ff3818d75d3f831959c0e02baaae46ab281934d922c4691ad0a38421eaa4ba0` |
| Active `Primitives.lean` | `702946866c83fa0bb8b9d2a248c14164d38aa15e1eebd67caf4f39b0076d0ff8` |
| Active `WP.lean` | `28d78a34e3033cb1856a6a1029f3c187abe14ef597ffcc6865910306c250928f` |
| `proof/ApeironWord.lean` | `92c771510c259bc56c9e437e7d3f3108fa9e2796e6ba41ca2cd826f4127ff74e` |
| `proof/corpus/Growable/Resources.lean` | `43a438c68ef8ae98fb36e11801ca0f003bf0aa1bef38e778c50bbaf4e2cf5a93` |

The generated digest is checked against the baseline manifest before handback. Because this file is
untracked, complete-file trailing-whitespace and final-newline checks supplement `git diff --check`. No build, test,
benchmark, global-allocator run or mutation control ran in this spike. Its acceptance is path/source
review and `git diff --check -- docs/epics/resource-security-continuation.md`.
The next resource work is the model witness plus real-path refusal regression; the next security
work is the missing specification and attribution verdict. Whole adapter and security closure remain
open for the exact reasons above.
