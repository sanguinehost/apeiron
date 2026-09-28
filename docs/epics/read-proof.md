# Read-path proof (composition, next steps, classifier continuation)

Merged epic record: `spec/read-proof-composition.md`, `plan/read-proof-next.md`, `plan/read-classifier-continuation.md` (one file per epic; see `docs/README.md`).

## Public read proof composition contract

Status: proposed and uncompiled. This document fixes the downstream theorem interface and proof
dependencies; it establishes no semantic grade. The generated roots, current bridge declarations and
import graph were read at the baseline recorded in `target/readlink/baseline.json`.

```text
SCOPE: exact Stored/Regen public-read targets, missing premises and producers, joint witnesses,
       controls, vector-update inventory and an acyclic module placement; complete when the artifact
       is source-linked and reviewable without changing a theorem premise
REGISTER: refinement planning; excludes proof implementation, source changes, extraction changes,
          SearchDecisionCorrect closure, allocator conformance and performance
CORPUS: proof/ApeironWord.lean word_{stored,regen}_table_get_at and table.probe.spill_read;
        Growable/Representation.lean RepR; Growable/Search.lean SearchDecisionCorrect and
        word_{stored,regen}_search_decision_simulates; canonical Apeiron.Table.get
REFERENCE: proof/extract/lib.rs, proof/ApeironWord.lean and proof/corpus/Growable/
           {Scan,Search,Relocation,Refinement,Representation,Resources}.lean read from this checkout
GATE: git diff --check -- docs/epics/read-proof.md plus source-linked Sol/lead review;
      no build and no semantic control run
GRADE: planning only; every signature below is proposed/uncompiled
```

### Existing roots and exact target surface

The real wrappers in `proof/extract/lib.rs` call `Table::get_at`; the generated definitions are:

- `word_stored_table_get_at : WordTable (Stored I64) → Sigma → I64 → Result I64`
- `word_regen_table_get_at : WordTable (Regen I64WordCodec) → Sigma → I64 → Result I64`

The following are the exact proposed public-read theorem signatures. They are deliberately stated
from `RepR` plus named source-link certificates, rather than from a desired classifier result.
Names and syntax are proposals until a later bounded Lean task compiles them.

```lean
theorem word_stored_table_get_at_refines
    (state : WordTable (table.keyrep.Stored I64))
    (sigma : cell.Sigma) (key : I64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hrep : RepR state decodeStoredOption direct live logical)
    (hread : StoredReadSourceCorrect state (sigma, key)) :
    word_stored_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key))

theorem word_regen_table_get_at_refines
    (state : WordTable (table.keyrep.Regen table.keyrep.I64WordCodec))
    (sigma : cell.Sigma) (key : I64)
    (direct : LogicalKey → Option Location) (live : LogicalKey → Prop)
    (logical : Apeiron.Table.HMap LogicalKey I64)
    (hrep : RepR state decodeRegenOption direct live logical)
    (hread : RegenReadSourceCorrect state (sigma, key)) :
    word_regen_table_get_at state sigma key =
      ok (Apeiron.Table.get logical (sigma, key))
```

`StoredReadSourceCorrect` and `RegenReadSourceCorrect` are unimplemented packaging placeholders, not
assumptions to be postulated permanently. The targets are not proof-ready until every field below is
made precise and produced by checked lemmas about the
generated `get_at` call graph. They may package carrier-specific address/decode facts, but their
shared fields must expose the following facts without containing the desired conclusion:

1. generated address succeeds and agrees with `RepR.address`/`RepR.directIndex`;
2. resident classification is exact: a live full-key match yields that flat location; an empty
   admitted direct slot uses `RepR.directCover`; an occupied non-match continues to spill;
3. the selected shard agrees with generated `table.shard` and `RepR.address`;
4. when the generated summary query returns `ok false`, checked summary shape/recordedness makes
   that result exclude the routed spill after direct exclusion; the invariants do not force the
   query to return false;
5. generated `get_at` calls `spill_read`, which calls `summary_maybe`; on `ok true`, its finite sweep
   is complete for the admitted geometry and a tag hit is confirmed by full `(Sigma,key)` equality;
6. the returned index, when present, is in bounds and denotes the unique entry supplied by
   `RepR.liveSet`, `coverage`, `unique` and `observations`;
7. when no match is returned, one of direct-empty, direct-excluded plus sound summary miss, or
   complete finite coverage proves logical absence. No vacancy or first-fit conclusion follows.

The structures must not contain `word_*_table_get_at ... = ok (Apeiron.Table.get ...)`, an assumed
`Found`/`Absent` result, `SearchDecisionCorrect`, or a caller-provided desired location. Those would
make the public theorem circular. The insertion classifier and its five cases remain separate.

### Missing premises and producer map

| Required fact | Present source | Producer required before public-read proof |
|---|---|---|
| Whole-table observation, uniqueness, physical coverage, address, direct-index and direct-empty certificate | `GrowableRepresentation.RepR` fields | Existing projections; add only missing projections, not stronger premises |
| Summary index is within the selected summary word vector | `GrowableScan.summary_index_word_lt_len` in `proof/corpus/Growable/Scan.lean` | Proved and audited; [execution evidence](../evidence/read-proof-next.md) |
| Set bit is observed by the same-place query | `GrowableScan.summary_set_implies_maybe` links generated `Region.summary_set`/`summary_maybe` | Proved and audited, with nonempty witness and generated mutation control; [execution evidence](../evidence/read-proof-next.md) |
| Every live spill publication records its summary bit | generated `publish_live` calls `summary_set` before tag publication | Representation construction/preservation theorem over the real write path |
| Summary miss excludes all live matching payloads in the routed region | not supplied by `RepR.directory` alone | `SpillSummaryShape`, `SpillSummaryRecorded`, and a derived miss-soundness theorem carried by `RepR` or a source-linked extension |
| Tag/fingerprint hit is only a filter; full key decides | generated `tag`, `scan.match_tag`, `matches`; `entryAt` decodes full key | Existing scalar scan facts plus carrier-specific `matches` exactness |
| `spill_read` visits every candidate required by the admitted layout and returns the first full match | generated nested loops exist; no accepted loop theorem | New Stored/Regen-neutral spill coverage theorem over `WINDOWS + groups`, prefix groups, and all tag-hit lanes |
| Resident hit/empty/non-match control flow matches the three certificate cases | generated `Table.get_at`; `DirectCover` covers direct-empty | New source-linked resident classifier theorem; occupied non-match must retain spill fallback |
| Stored/Regen decoding and address agree with concrete dictionaries | `Growable.Identity`, `decodeStoredOption`, `decodeRegenOption`, `RepR.address` | Reuse checked carrier/address lemmas; joint theorem instantiates the same logical query |
| Canonical result after a located decision | `word_{stored,regen}_search_decision_simulates` under `SearchDecisionCorrect` | Reuse only if a separately proved generated classifier supplies `SearchDecisionCorrect`; public read cannot assume it |

`RepR` currently contains directory length agreement but no summary invariant or spill-loop coverage.
Consequently `RepR` alone cannot justify either summary-negative absence or summary-positive
completeness. Adding the desired public-read equality to `RepR` is forbidden.

### Joint witnesses and semantic controls

Each accepted theorem needs paired Stored/Regen states representing the same `logical` map and using
the same `sigma`, `key`, address class and expected canonical value. This is a joint fixture across
two carrier-typed states, not one state at two carrier types. Unrelated carrier fixtures do not
establish joint scope.

| Case | Required reachable fixture | Future rejecting control |
|---|---|---|
| Direct live hit | Constructor-reachable table with the queried `(Sigma,key)` in the admitted flat slot for both carriers | Compare only `Sigma` or only address; full-key collision fixture must reject |
| Direct empty, summary negative | Direct slot empty, routed spill contains other live keys, queried summary bit absent and query logically absent | Treat summary miss as global absence without direct exclusion; occupied direct match makes result wrong |
| Occupied direct non-match, spill hit | Colliding direct occupant and queried key in routed spill | Return default on occupied direct non-match; expected spill value rejects |
| Prefix vacancy, late spill hit | Summary positive; an earlier swept lane is vacant and a later group/lane contains the query | Terminate `spill_read` at first vacancy or omit the final required group; late-hit assertion rejects |
| Summary/tag collision, absent | Summary positive and tag hit exists for another full key, with complete sweep and no query | Treat tag equality as presence; default/read identity rejects |
| Unequal `Sigma`, same key/address | Other face is live while queried face is absent or has another value | Drop `Sigma` from `matches`; both carriers return the wrong face |

Fixture satisfiability in a hand-built Lean state is insufficient for constructor reachability. The
future producer task must either derive each fixture from the real generated constructor/write roots
or explicitly keep it as a logical-model witness and supply a separate reachable Rust/source witness.
No witness may assume the classifier outcome it is intended to test.

### Vector update lemma inventory

The generated `summary_set` computes `index % 64`, reads `index / 64` with
`alloc.vec.Vec.index_mut`, writes `old ||| bit`, and installs the returned vector. The checked composition in
`Growable/Scan.lean` uses the active Vec specifications below; list lemmas remain references for
later representation proofs:

- Resolved and used in this corpus: `Array.set_val_eq`, `List.getElem?_set_self`,
  `List.getElem?_set_eq_of_lt`, and `List.getElem?_set_of_lt'`.
- A local unequal-index lemma is present in `Growable/Relocation.lean` as the private
  `getElem?_set_eq_of_ne`; it cannot be imported and must be reproved or promoted deliberately.
- Resolved in pinned Lean core `Init/Data/List/Basic.lean`: `List.length_set` proves that list
  update preserves length.
- Resolved in active `target/aeneas-model/lean/Aeneas/Std/Vec.lean`:
  `Vec.index_mut_slice_index` reduces the generated generic mutable-index call to
  `Vec.index_mut_usize`; `Vec.index_mut_usize_spec` returns the old element and `Vec.set` back-function;
  `Vec.index_slice_index` supplies the corresponding read normalization; `Vec.set_val_eq` exposes
  the backing-list update. `Vec.set_length` and `Vec.index_usize_spec` establish readback at the
  updated index in `summary_set_implies_maybe`. The active `Vec.set` preserves `cap` explicitly.
- Accepted scalar facts in `Growable/Scan.lean`: `summary_index_word_lt_len`,
  `summary_remainder_lt64`, `summary_mask_nonzero`, and `u64_or_mask_and_mask`.

Existing patterns occur in `Growable/Relocation.lean` (`getElem?_set_eq_of_ne`, nested payload/set
readback) and the older `TableRefinement*` modules (`Array.index_mut_usize`, `Array.set_val_eq`,
`List.getElem?_set_*`, `List.set_set`). They remain implementation references. The checked `summary_set_implies_maybe` theorem and
its controls establish the local summary composition; they do not establish representation
preservation or complete spill reads.

### Acyclic module placement

The current graph is `Search` and `Relocation` → `Refinement` → `Representation`; specifically,
`Representation` imports `Geometry` and `Refinement`, while `Refinement` imports `Relocation` and
`Search`. Therefore `Search` cannot import `Representation` without a cycle.

Place the public theorems in a new `proof/corpus/Growable/Read.lean` importing
`Growable.Representation` and `Growable.Scan`. Put source-neutral spill-loop lemmas either in a new
`Growable/ReadLoop.lean` importing `ApeironWord`, `Growable.Scan`, and `Growable.Search`, or directly
in `Read.lean` if the file stays within task limits. `Representation` must not import either module.
The resulting path is:

```text
ApeironWord + Geometry + Identity + Scan
                  ↓
Search + Relocation
        ↓
Refinement
        ↓
Representation
        ↓
Read
        ↓
AxiomAudit
```

A later implementation task may own at most `Growable/Read.lean`, `lakefile.lean`, and
`AxiomAudit.lean`. If a separate `ReadLoop.lean` is necessary, split it into a preceding task rather
than exceeding three files or creating a cycle.

### Acceptance boundary

The local index, mask and summary write/read prerequisites are proved. This public-read contract
remains planning scope: kernel-checked public-read refinement still requires representation
preservation, complete spill coverage, both carrier theorems, joint
reachable witnesses, audit registration and semantic mutations. It does not close
`SearchDecisionCorrect`, insertion/build refinement, allocator behavior, Rust/LLVM code generation,
unsafe-container correctness, ABI/hardware execution, performance, or G1/REG-1.

## Next leg: source-linked summary facts

Status: contracts and the shift, index, mask and summary write/read proofs are accepted.
Rust, extraction freshness and both Lean builds pass; seven existing assurance size breaches remain.
Lead and resumed proof implementation: Astra. Independent proof/contract review: Sol. This is a bounded
continuation of [read intent](read-path-specialization.md) and
[read specification](read-path-specialization.md), not a new runtime design.
It supersedes the old handoff branch table and bundled summary/read proof dispatch.

### Outcome and limits

Establish small source-linked summary lemmas and their rejecting controls, then the generated
summary write/read link. Prepare the downstream read contract and allocator architecture in
parallel. Do not promise whole-table `get`, insertion classification or allocator conformance in
this leg. `SearchDecisionCorrect` remains a separate insertion obligation.

The previous summary task failed three materially different approaches. Its patch is rejected;
`target/handoff-run/summary/verdict.md` records the exact residual. Do not restart that bundled task.
The new decomposition separates index arithmetic, mask arithmetic and vector update composition.

```text
SCOPE: docs/epics/read-proof.md and docs/archive/handoff-openai.md; replace stale dispatch with bounded source-grounded tasks and current evidence.
REGISTER: planning; no new proof or runtime result.
CORPUS: generated table.Region.summary_index/summary_set/summary_maybe; Scan.summary_trailing_zeros_eq; Representation.RepR; Search.SearchDecisionCorrect.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean and proof/corpus/Growable/{Scan,Representation,Search,Refinement,Resources}.lean; existing read intent/spec retained.
GATE: git diff --check and resolve named paths/declarations; no documentary red invented.
GRADE: anchor plan; proof tasks earn proved only through kernel checks, audits and controls.
```

### Baseline

HEAD `262bb0a94eff2cc7d5655122a954f44d74de96b6` plus accepted uncommitted continuation and
retirement changes. Preserve the dirty tree; HEAD alone is not the task baseline.
Generated SHA256 `92c771510c259bc56c9e437e7d3f3108fa9e2796e6ba41ca2cd826f4127ff74e`;
Geometry SHA256 `627ab73df9cd88120bf886f07b5f6cf11a71a84d1fb42702491e453bb3ea6cc6`.
Canonical resolved package stays `3d66ea3aa2fe1de5997b6b757247454a2f771358`; tool pins and
18 extraction-source hashes resolve through `proof/README.md`. Compare actual hashes at dispatch.

[Retirement evidence](../evidence/kani-retirement.md) records post-change library tests 92/one ignored,
Clippy/fmt, coverage controls, fresh extraction 1708 and bridge 3482. These are existing receipts,
not reruns in this planning pass. Kani is retired. Seven size breaches and unresolved claim-checker
references keep overall assurance open. Performance has no new measurement.

### First parallel deployment

| Task | Owner and owned files | Deliverable and gate | Control and dependency |
|---|---|---|---|
| APEIRON-READLINK-W1-T1 | Sol scalar: `proof/corpus/Growable/Scan.lean`, `proof/corpus/AxiomAudit.lean` | Prove/audit `summary_index_word_lt_len`; bounded `lake build Growable.Scan AxiomAudit` | Disposable generated index-mask `power - 1` to `power` mutation must refute the bound on a concrete successful input; restored gate passes. Needs unchanged fresh extraction. |
| APEIRON-READLINK-W1-T2 | Sol read contract: `docs/epics/read-proof.md` | Exact Stored/Regen generated-root theorem signatures, missing premises/producers, vector update lemma inventory, joint witness cases, acyclic imports; `git diff --check -- docs/epics/read-proof.md` plus lead/Sol artifact review | Planning only: no semantic gate claimed. Map summary-negative/direct-live and prefix-vacancy/late-hit cases to the future semantic controls. Read-only source access; independent of scalar edits. |
| APEIRON-READLINK-W1-T3 | Sol allocator: `docs/epics/allocator-model.md` | Same-input success/refusal relation, capacity/effect limitations, implementation alternatives and exact downstream file decomposition; `git diff --check -- docs/epics/allocator-model.md` plus lead/Sol review | Planning/model only; semantic controls inapplicable until executable implementation. Real allocation refusal differs from size overflow. No backend, Rust signature or premise change. |

The documentation gates establish well-formed artifacts only. Specialist decisions and source
references are required before their plans can authorize implementation; a green diff check proves
no contract. Do not add a keyword checker or call an abstract oracle production conformance.

Exact scalar target (no extra nonempty, power-of-two or caller-supplied index bound):

```lean
theorem summary_index_word_lt_len (R V : Type) (place : U64)
    (lenWords index : Usize)
    (hindex : table.Region.summary_index R V place lenWords = ok (some index)) :
    index.val / 64 < lenWords.val
```

Before accepting its mutation, construct a bounded counterexample showing that the altered
successful result violates the conclusion. A syntax/type/import failure is not a semantic control.
Candidate counterexample: `lenWords = 1`, `place = 256`; the altered mask can select index 64, outside the single word. Confirm this by a bounded checked witness before accepting the control. If the proposed mutation survives or lacks an executable witness, stop for Sol adjudication.

### Sequential proof tasks after the first barrier

Each row owns `proof/corpus/Growable/Scan.lean` and `proof/corpus/AxiomAudit.lean` only; the previous
owner releases both paths first. One gate each: bounded `lake build Growable.Scan AxiomAudit`.

| Task | Exact scope | Acceptance/control |
|---|---|---|
| APEIRON-READLINK-W2-T1 | Isolate the generated `U64` shift/mask law: for a shift below 64, the single-bit mask is nonzero and `(word OR mask) AND mask` is nonzero; derive the shift bound from `% 64` rather than adding it to the summary caller | Kernel-checked scalar law, audit and zero-word witness. In a disposable scalar fixture replace the law’s OR with AND: the zero-word witness rejects, then restore. This checks scalar-law sensitivity only; the production generated mutation belongs to the next task. No vector or whole-summary reduction. |
| APEIRON-READLINK-W3-T1 | Compose accepted index/mask facts with actual `Vec.index_mut` update and unchanged vector length to prove the statement below | Nonempty successful generated write/query witness on the same input; generated summary-set OR-to-AND mutation rejects, restored gate passes. No desired-result or classifier premise. |

```lean
theorem summary_set_implies_maybe {R V : Type} (before after : table.Region R V)
    (place : U64) (hset : table.Region.summary_set before place = ok after) :
    table.Region.summary_maybe after place = ok true
```

The nonempty witness is required, not optional: the empty-summary branch alone cannot close this
claim. If the witness needs another substantive lemma, split it into a new bounded task before
closing this one. Do not quietly extend scope or weaken the domain.

### Downstream work held at this boundary

`Representation -> Refinement -> Search` is the current import graph. A public read theorem that
uses `RepR` must live downstream (candidate `Growable/Read.lean`); importing Representation into
Search would cycle. Future packaging can own `Growable/Read.lean`, `lakefile.lean`, `AxiomAudit.lean`,
but is not yet dispatch-ready. First establish summary/fingerprint facts, representation
construction/preservation and complete generated spill coverage. `RepR` alone is insufficient.
Do not convert the conditional apply-get theorems into public-read claims by assumption.

Allocator implementation stays held until the design adjudicates capacity/layout invariants in the
active capacity-carrying model and its missing allocator choice, with a real-source refusing-branch control. No edit to
`target/aeneas`, no deterministic model called may-fail, no oracle called `ResourceAdapterSimulates`.
Size-remediation implementation, security rebase and strict benchmarking are separate later waves;
none is added just to fill an agent slot. Measurement needs an exclusive quiet host and frozen inputs.

### Ownership, resources and stop rules

At implementation dispatch, the lead creates `worktree-agent-<task-id>` / `codex/readlink-<task-id>`
from the recorded HEAD and copies the required accepted dirty baseline with a digest manifest.
Worktrees are created only for explicitly dispatched tasks. Declare owner/base/lane and Arc before each
edit. Integrate only reviewed owned files, preserve receipts, then remove task worktrees/branches.
No commit/push or unrelated git mutation is authorized.

Three workers prepare concurrently. Only the proof worker initially leases compilation; contract
workers read source and write disjoint documents. Shared generated file, target and Lake caches
have one owner. All Lean commands, including scratch/control commands, use:

```sh
flock /tmp/opencode/apeiron-lane.lock systemd-run --user --scope \
  -p MemoryMax=8G -p MemorySwapMax=0 env LEAN_NUM_THREADS=1 timeout 900s \
  lake build Growable.Scan AxiomAudit
```

Cwd: `proof/corpus` in the owning worktree. No unbounded heartbeats/reduction, raised recursion
limits or whole-source `decide`. Controls use disposable workspace-disk copies, never the live tree.
Each proof task has a 30-minute active-work budget (queue time excluded), at most three materially
different failed approaches, and a 900-second cap per gate. Stop at whichever limit comes first.
A timeout/resource failure is inconclusive, not the required negative result. Record the smallest
remaining goal and release the lane; no unattended retries.

### Closure

A local task closes on its focused gate, exact cone `{propext, Classical.choice, Quot.sound}`,
joint witness and applicable semantic control. The Slang empty cone is separate. After the proof
sequence, run one full integrated bridge build and the existing shared barrier commands required
by AGENTS: Rust test/clippy/fmt, extraction freshness, both bounded Lean package builds, applicable
audits and controls. Freeze source/tool digests; do not repeat unchanged gates between every edit.
Record all failures, including the existing size/claim obligations; do not call the epic closed.

Deliver: named checked local summary facts, controls and source digests; reviewed read/allocator
contracts; explicit remaining universal read/classifier and resource obligations. If a proof task
blocks, return its bounded verdict plus the independent completed contract artifacts. The next
leg does not silently expand into a backend rewrite or unlimited proof search.

### Planning review

Sol read review accepts the index-bound decomposition and identifies the import cycle. It rejects
using a generated-write mutation as the scalar lemma’s control; the table above separates scalar
sensitivity from the later generated semantic control. Host arithmetic confirms the candidate
mutant selects index 64 for length 1/place 256; this is not a Lean witness or an executed proof gate.
The implementation must still kernel-check its witness. Allocator review identifies missing effect
choice; execution review distinguishes stock capacity absence from the active patch’s capacity field
and missing capacity/layout invariants; a proposed wording checker is rejected as no semantic evidence.

Documentation validation checks local links, all 18 recorded extraction-source hashes, unchanged
non-document files against the retirement manifest, and `git diff --check`. No proof, source edit,
benchmark, Kani run or new acceptance result is part of this planning pass.

Final reference-only slice: lead owns `docs/flash-handoff.md`; expand existing short Growable file
paths and the generated apply-insert name to their actual repository artifacts. Register:
documentation attribution, no theorem or grade change. Gate: `git diff --check` plus the existing
claim checker; preserve unrelated historical test-command findings. Reference: actual generated
`table.Table.apply_insert` and named corpus files, read before editing.

Final existing claim-checker result: exit1, five findings remain in historical/contextual
attribution: Rust `is_intact` and `address_held` are classified as unresolved Lean names; their
paragraphs lack resolvable test commands; the generated `apeiron_word.table.Table.apply_insert`
name is also unresolved by the checker despite its definition in `proof/ApeironWord.lean`.
No checker change or invented gate clears these findings. They remain outside proof acceptance;
raw output is `target/next-leg/claims-final.log`. Documentation diff-check is exit0.

### Execution ownership

Lead documentation exception: this plan, `docs/evidence/read-proof-next.md`, and
`docs/archive/handoff-openai.md`. Register: gate receipts, reviewed integration decisions and exact residuals;
no new mathematical claim without its worker gate/control. Gate: diff-check and receipt inspection.
The three first-wave worktrees are based on HEAD plus the copied accepted dirty manifest
`target/readlink/baseline.json`. Branches: `codex/readlink-w1-t1`, `codex/readlink-w1-t2`,
`codex/readlink-w1-t3`; directories `../worktree-agent-APEIRON-READLINK-W1-T{1,2,3}`.
Sol scalar exclusively holds compilation/Lean lanes. Contract workers own only their one document.
No commit/push. The lead reviews/copies owned outputs and removes each completed task worktree.

### Execution decision

First-wave scalar task rejected at its three-approach bound; no proof diff integrated. Read and
allocator contracts accepted at planning/model scope. The first barrier is not green, so the mask
and summary write/read tasks do not open. See [execution evidence](../evidence/read-proof-next.md).
The smallest next prerequisite is an exact machine-width shift success/value lemma; it must be a
new bounded task, not an unattended retry of the rejected index theorem. No full epic closure.

### Astra shift prerequisite

User-directed Astra escalation `APEIRON-READLINK-W2-T0` owns only
`proof/corpus/Growable/Scan.lean` and `proof/corpus/AxiomAudit.lean`. Isolated worktree
`../worktree-agent-APEIRON-READLINK-W2-T0`, branch `codex/readlink-astra-shift`, accepted dirty
baseline manifest `target/readlink/astra-shift/baseline.json`. The task proves exact success/value
normalization of generated `core.num.Usize.MAX >>> 6#i32`, without caller-added premises.
It does not retry the whole index theorem. Architecture verdict and residual accompany the proof.

Gate: bounded `lake build Growable.Scan AxiomAudit`, with exact exit captured. Disposable scalar
fixture control changes shift6 to shift5 while retaining the expected division by64; it must reject
semantically, followed by restored green. This is prerequisite-law sensitivity, not the generated
summary-index mutation. The existing 30-minute/three-approach bounds and one heavy lane apply.
Sol reviews theorem/control sufficiency before lead integration. No commit/push or backend change.
Lead owns this plan, execution evidence and handoff for receipt-only updates; gate diff-check.

Astra prerequisite accepted after focused/audit/control gates and independent Sol review; see the
execution evidence. The next index task can obtain the exact successful shift bound from
`Apeiron.GrowableScan.summary_max_shift_six`. This replaces the former immediate normalization
blocker with the still-open fast-path/fallback arithmetic; it does not open the dependent mask and
summary composition waves until the index task passes its own gate/control.

### Resumed index execution

User authorizes the remaining unblocked work. `APEIRON-READLINK-W3-T0` resumes the original
index-bound target on the accepted shift-lemma baseline. Owner: the user-deployed Astra worker;
Sol supplies independent arithmetic review and final proof/control adjudication. Owned files remain
`Growable/Scan.lean` and `AxiomAudit.lean`; no extra caller premises. Worktree
`../worktree-agent-APEIRON-READLINK-W3-T0`, branch `codex/readlink-index-resume`, digest manifest
`target/readlink/index-resume/baseline.json`. One heavy lane; existing 30-minute/three-approach
bounds. The original focused gate and generated mask mutation remain required.

A second Sol performs read-only mask-lemma inventory while index work proceeds; this is preparation,
not opening the dependent proof task. Only an accepted index gate/control opens the existing
mask task, then accepted mask facts open generated summary write/read composition. Lead continues
the existing three-document receipt scope. No commit/push or installed-model/source mutation.

### Mask execution ownership

The index task passes its gate, semantic control and Sol review. `APEIRON-READLINK-W2-T1`
opens with the same Astra owner and independent Sol adjudication, owning only
`proof/corpus/Growable/Scan.lean` and `proof/corpus/AxiomAudit.lean`. Worktree
`../worktree-agent-APEIRON-READLINK-W2-T1`, branch `codex/readlink-mask`; accepted dirty
baseline manifest `target/readlink/mask/baseline.json`. The focused gate, zero-word scalar
control and 30-minute/three-approach limits above apply. One heavy lane; no summary composition
work until this task passes. Read-only lemma preparation: `target/readlink/mask-preparation.md`.

### Summary composition execution ownership

The mask task passes its focused gate, scalar control and Sol review. `APEIRON-READLINK-W3-T1`
opens with Astra owning only `proof/corpus/Growable/Scan.lean` and `proof/corpus/AxiomAudit.lean`,
with independent Sol adjudication. Worktree `../worktree-agent-APEIRON-READLINK-W3-T1`, branch
`codex/readlink-summary`; copied accepted dirty manifest `target/readlink/summary/baseline.json`.
The exact theorem, mandatory nonempty witness, generated OR-to-AND control, focused gate and
30-minute/three-approach bounds above apply. One heavy lane. Read-only vector/witness inventory:
`target/readlink/summary-preparation.md`. No public-read or allocator implementation opens here.

### Read-contract reference refresh

Supplied-checkout documentation exception, lead-owned task `APEIRON-READLINK-W4-T0`:

```text
SCOPE: docs/epics/read-proof.md; refresh accepted-lemma references.
REGISTER: documentation attribution only.
CORPUS: summary_index_word_lt_len; summary_set_implies_maybe.
REFERENCE: /home/socol/Workspace/apeiron/proof/corpus/Growable/Scan.lean
GATE: git diff --check and declaration/reference checks.
GRADE: documented proved results; no new proof claim.
```

The contract now records accepted index/mask/summary facts while retaining the separate
representation, complete spill-loop and public-read obligations. No contract premise changes.

### Local proof completion

All three resumed proof tasks pass their focused gates, applicable semantic controls and independent
Sol reviews. Shared Rust, extraction and both-package builds pass on the integrated identity;
[execution evidence](../evidence/read-proof-next.md) records exact exits, digests and residuals.
The assurance checker remains red on seven pre-existing size breaches, so this completes local
proof work without full assurance or epic closure. No task worktree/branch remains. Next proof
boundary: live publication/representation recordedness, then complete spill traversal and
downstream public-read linkage. Allocator and insertion-classifier work remain separately held.

## Read and classifier continuation map

Status: source-pinned planning verdict, not a checked theorem or a W7 reopening. Owner: Sol proof escalation. Base `262bb0a94eff2cc7d5655122a954f44d74de96b6` plus the accepted dirty tree; compare source hashes before dispatch. This spike owns this file and `target/open-areas/proof-verdict.md` only. No Lean lane or shared build product was used.

```text
SCOPE: finite source-linked read/classifier/representation map, one ready next proof task,
       and an explicit W7 reopening verdict; no implementation
REGISTER: refinement planning; excludes proof acceptance, allocator realization,
          canonical pin changes, benchmark claims and W7 closure
CORPUS: Growable.Scan.summary_set_implies_maybe; Growable.Representation.RepR;
        Growable.Search.SearchDecisionCorrect; canonical Apeiron.Table.get/insertWith
REFERENCE: proof/ApeironWord.lean generated table.Region.publish_live,
           table.probe.spill_read, table.Table.get_at/search_decision/apply_insert;
           proof/extract/lib.rs real Stored/Regen wrappers; src/table.rs and
           src/table/{state,probe}.rs read in this checkout
GATE: git diff --no-index --check /dev/null docs/epics/read-proof.md;
      exit 1 means new-file difference, no whitespace diagnostics; semantic gates not run
GRADE: anchor/planning; prior Scan receipts are historical, not fresh acceptance
```

### Boundary found in source

`Growable/Representation.lean:109` defines `RepR` with live set, observations, uniqueness, physical coverage, address, direct index, direct cover, directory and length. It has no summary shape/recordedness, tag-to-payload agreement, constructor theorem or whole-state preservation. `Growable/Search.lean:120` defines `SearchDecisionCorrect` as a premise to `word_{stored,regen}_search_decision_simulates`; `absenceCovered` currently offers a direct case or checked-location case, with no summary-miss case. Its five required insertion outcomes are found-flat, found-spill, vacant-flat, vacant-spill and exhausted/growth. `src/table/probe.rs:212` and generated `table.probe.spill_read` search only after valid grouped geometry and a positive summary; the finite loop is `WINDOWS + groups` with all tag-hit lanes checked by full `matches`. `table.Table.get_at` handles direct-word absent/out-of-extent, flat live match, flat vacant, and occupied non-match spill fallback. `table.Table.search_decision` still calls `spill_locate`, so public read and insertion classification require different theorems. `proof/extract/lib.rs` contains both `word_{stored,regen}_table_new_at` wrappers, but the current generated file contains only their real generic callee `table.Table.new`, not either thin wrapper. That callee suffices for a concrete instance constructor theorem; a claim about either wrapper itself needs extraction selection.

The checked local fact is `GrowableScan.summary_set_implies_maybe`: a successful generated summary write makes that same-place query positive. Generated `Region.publish_live` writes payload, invokes `summary_set`, then publishes a nonzero tag. The implication is now checked through `publish_live` by `GrowableScan.publish_live_implies_summary_maybe`, with a successful-publication/false-query mutant and restored green gate (`target/open-areas/APEIRON-READ-W2-PUBLISH`). Other publication/rebuild paths and representation recordedness remain open. A summary miss excludes routed spill only after shape, recordedness, routing and equality-respecting address facts; it does not exclude a flat match. A positive summary or matching tag proves no presence until full `(Sigma,key)` equality is checked.

### Finite discharge order

Each row below is a dependency stage; rows 4, 7 and 8 must be split into separate implementation tasks before dispatch. Each task has at most three owned files, one bounded focused gate and a semantic control in a disposable copy. `flock /tmp/opencode/apeiron-lane.lock`, `LEAN_NUM_THREADS=1`, `systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0`, and `timeout` apply to every Lean gate. The lead assigns numeric task IDs and one writer per path when opening a wave. Rows sharing a file are serial. Gate syntax is the intended command from `proof/corpus`; each command additionally runs the matching `AxiomAudit` target. These are proposed targets, not signatures asserted to compile.

| Order; owner; files | Exact source and target statement | Gate and rejecting control; dependency |
|---|---|---|
| 1, Sol; `Growable/Scan.lean`, `AxiomAudit.lean` | Generated `table.Region.publish_live`: for every `before after index payload place`, `publish_live before index payload place = ok after` implies `summary_maybe after place = ok true`. The exact candidate is `theorem publish_live_implies_summary_maybe {R V : Type} (before after : table.Region R V) (index : Usize) (payload : cell.Sigma × R × V) (place : U64) (hpub : table.Region.publish_live before index payload place = ok after) : table.Region.summary_maybe after place = ok true`. It composes `summary_set_implies_maybe` with final tag update framing. | `lake build Growable.Scan AxiomAudit`; mutate generated `publish_live` in a throwaway extraction to omit `summary_set` and use a nonempty one-word, one-slot witness whose query was false before publication. The witness must fail the theorem semantically, then restore green. Existing Scan theorem is prerequisite. This is the next ready proof task. |
| 2, Sol; `Growable/Scan.lean`, `AxiomAudit.lean` | Generated `summary_index`, `summary_maybe`, `tag`, `publish_live`, `scan.match_tag`, `least_set_lane`, `matches`: define `SpillSummaryShape`, `SpillFingerprintExact` and prove `summary_maybe = ok false` excludes a recorded same-place bit; prove publication tag readback, equal full keys imply equal tag, and a tag hit still requires full equality. | `lake build Growable.Scan AxiomAudit`; flip a published tag bit and use distinct full keys with equal fingerprints. Requires row 1 only for later recordedness; scalar work can be sequenced in the same Scan lane. |
| 3, Sol; `Growable/Representation.lean`, `AxiomAudit.lean` | Extend `RepR` and `ReservedRepR` by `SpillSummaryShape`, `SpillSummaryRecorded`, `SpillFingerprintExact` for every search-visible spill region; derive `SpillSummaryMissSound` from those fields, `RepR.address`, equality-respecting carrier dictionaries and generated `summary_maybe`. No desired `get_at` result or assumed output invariant is a field. | `lake build Growable.Representation AxiomAudit`; clear a live bit in a disposable generated state and require miss-soundness to fail on a concrete live query. Rows 1-2 and carrier address facts precede it. Constructor and preservation are still unproved after this definition task. |
| 4, Sol; `Growable/Representation.lean`, `Growable/Resources.lean`, `AxiomAudit.lean` | Generated real `table.Table.new`, `Region.reserve_empty`, `publish_live`, prepared-shard apply, growth, relocation and pending commit: prove both concrete `N=16` constructor instances establish full `RepR` for empty canonical maps, and each successful write/rebuild preserves all old fields plus the new summary/tag fields. Refusal stutters only under the exact resource branch. Split this row into serial <=3-file tasks by constructor, normal publication and rebuild before dispatch. | Constructor core gate `lake build Growable.Representation AxiomAudit` on the existing generated `table.Table.new`; no wrapper prerequisite for that theorem. A later public-wrapper claim requires a separate extraction-selection task and `proof/regen.sh --check`. Per subsequent subtask, focused `lake build Growable.Representation AxiomAudit` or `lake build Growable.Resources AxiomAudit`; omit the summary write in each real publication path, then mutate a moved payload's place on growth. Requires row 3. Production adapter effect and allocator-may-fail correspondence need their own realization result. |
| 5, Sol; new `Growable/ReadLoop.lean`, `AxiomAudit.lean`, `lakefile.lean` | Generated `table.probe.spill_read_loop0` and inner loop: under admitted `N=16`, grouped power-of-two region geometry, equal tag/payload lengths and exact tag/address relation, a positive summary performs a complete finite match sweep, including after vacancies; returned index is in bounds and full-key equal; `none` means no full match. The `WINDOWS + groups` order must be related to every possible routed representative. | `lake build Growable.ReadLoop AxiomAudit`; mutate `WINDOWS + groups` to omit the last required group, using a late-hit fixture. If group coverage or Aeneas loop measure cannot be established with checked primitives, stop for a separate feasibility task. Rows 2-3 are inputs. |
| 6, Sol; new `Growable/Read.lean`, `AxiomAudit.lean`, `lakefile.lean` | Generated `word_{stored,regen}_table_get_at`/`Table.get_at`: from a constructor-reachable `RepR` and derived source certificates, prove each root equals `ok (Apeiron.Table.get logical (sigma,key))`. Partition direct hit, direct empty, occupied non-match, routed summary miss, and complete positive-summary sweep. No caller-supplied found/absent result, desired location or public-read equality in a premise. | `lake build Growable.Read AxiomAudit`; mutate occupied non-match to return default and direct empty to search an unrelated shard; paired Stored/Regen collision and late-hit fixtures reject. Requires rows 3-5 and actual reachable `RepR` in row 4. Import `Representation`/`Scan`/`ReadLoop` into `Read`; never import `Representation` into `Search` (cycle). |
| 7, Sol; interface task `Growable/Search.lean`, `AxiomAudit.lean`; producer task new `Growable/Classifier.lean`, `AxiomAudit.lean`, `lakefile.lean` | First replace `SearchDecisionCorrect.absenceCovered` by a three-way direct-empty, direct-excluded plus sound summary miss, or complete match coverage certificate while preserving existing apply implications. Then, in a downstream module importing `Representation` and `Search`, prove from generated `spill_locate`/`Table.search_decision` exact `Found`, first-fit `Vacant`, and `Exhausted` for found-flat, found-spill, vacant-flat, vacant-spill and exhaustion/growth. Preserve existing `word_{stored,regen}_search_decision_simulates` at their actual premises. `SpillVacantDirectGuard` must be proved from the real direct route, not supplied as an outcome. | Interface gate `lake build Growable.Search AxiomAudit`; producer gate `lake build Growable.Classifier AxiomAudit`. Mutate `spill_locate` to stop matching at a prefix vacancy and use a later matching key; separately reverse vacancy order. Requires rows 3 and 5 plus a finite insertion-order lemma; public-read success alone does not give first fit. `Search` must not import `Representation`. |
| 8, Sol; `Growable/Representation.lean`, `Growable/Resources.lean`, `AxiomAudit.lean` | Generated `word_{stored,regen}_apply_insert` and production `Table.apply_insert`: prove full `RepR` preservation against canonical `insertWith` for found/fresh flat/spill, unchanged unrelated keys, exact length, direct cover, summary/tag fields and both carriers. State success and refusal separately. Joint reachable witnesses must use the same logical map/query and distinct carrier-typed states for all four branches. | Focused `lake build Growable.Representation AxiomAudit` and source-mutant control deleting the final commit or reversing new-first merge; split per branch into serial <=3-file tasks. Requires rows 4 and 7, `GrowableCallbackLaws`, `QueryAddressCorrect`, `FreshLengthBound`, `DecisionSymbolicCorrespondence`, and the exact resource branch. A proof assuming the post-state `RepR` does not count. |

The proposal in row 1 is a conditional local lemma over the generated successful `publish_live` result. It does not assert that a real constructor reaches that result. The semantic witness needs a source-pinned nonempty summary and in-bounds payload/tag slot; `oneWordZeroRegion` has an empty slot vector and cannot itself witness publication. Build that finite fixture from the generated region constructor or an explicit region whose shape matches it, and confirm baseline/ red/ restored exits before crediting the control.

### Lane execution status (integrator)

- `Growable/Scan.lean` `18ef9cc8…`: the marker chain (`mapBytes_lane`, `from_le_bytes_lane`, `from_le_bytes_end_lanes`, `tagMarkers_from_le_bytes_lane`, `tagMarkers_from_le_bytes_word`, `tagMarkers_asymmetric_witness`), the positive fixtures `generated_clear_hits_strict`/`generated_tail_steps_in_bound`, and `match_tag_window_exact` (one eight-byte window: generated `word_at` to `eq_swar` to `gather` returns that window's eight equality bits under `off + 8 <= length`), `match_tag_exact` (whole scan: for every `line` of length ≤ 64 the generated `table.scan.match_tag` returns `bits` with `bits.getLsbD i = decide (i < length ∧ line[i]? = some wanted)`), and the removal trio `generated_clear_hits_exact`/`generated_clear_hits_live_decrease`/`generated_spill_read_clears_selected` (exactly the selected lowest lane cleared, every other bit unchanged, `BitVec.cpop` down one) are checked and registered in `AxiomAudit.lean`.
- `least_set_lane` is proved for every nonzero `U64` — `least_set_lane_selects`, `least_set_lane_exact`, `least_set_lane_in_range`, `least_set_lane_high_fixture` (registered). The source-correct invariant is the **lowest** set lane, not the highest. Control: generated `proof/ApeironWord.lean:6049` `then lane4 + 1#usize` to `then ok 64#usize` rejected the concrete `32770 -> 1` guard (exit 1); restore verified at `2a1138e4…`.
- Recorded controls for `match_tag_window_exact`: the generated word-loop advance `base + 8#u32` to `+7#u32` rejected the late-hit and post-vacancy guards; `eq_swar word tag` to `eq_swar word 42#u8` rejected the absent-tag guards; both restored at `2a1138e4…`.
- Recorded controls for `match_tag_exact`: the disposable word-shift mutation (`proof/ApeironWord.lean:6127` `i5 <<< base` to `i5 <<< 0#u32`) rejected with an unsolved word-mask goal and the `lastGroupLine` guard; the tail-shift mutation (`:6165` `1#u64 <<< base` to `1#u64 <<< 0#u32`) rejected with the tail `rfl` failure and the `wordTailBoundaryLine`/`secondWordTailLine` guards; baseline/red/restore exits 0/1/0, restored generated digest `2a1138e4…`, logs `/tmp/opencode/readlink-t7-control/*-scoped.log`.
- Recorded control for the removal trio: the disposable generated `hits - 1#u64` to `hits - 2#u64` kept types valid and failed the source-binding theorem with the expected-versus-mutated diagnostic; baseline/red/restore exits 0/1/0, restored generated digest `2a1138e4…`; logs `/tmp/opencode/readlink-w5-t8-control-*.log`, cone `/tmp/opencode/readlink-w5-t8-cone.log`.
- Row 5 `Growable.ReadLoop` (`454efdb7…`; the state/invariant/relocation block lives in `Growable/ReadState.lean` `e30382b4…`; root in `lakefile.lean`; earlier digests `5fb48b9c…`, `f6c375c9…`): the bound/candidate facts `generated_outer_stops_on_hit`, `generated_outer_stops_at_limit`, `generated_tail_indices_eligible` and `stored_candidate_full_match` are checked and registered; module gate `lake build Growable.ReadLoop` exit 0 (3350 jobs) and integrated `lake build AxiomAudit` exit 0 (3370 jobs, with the new `import Growable.ReadLoop`). Two recorded controls: a proof-level generated mutation rejected with unsolved goals, and a late-placement mutation rejected by the fixture (`Expression did not evaluate to \`true\``), both restored green; logs `/tmp/opencode/readlink-w5-t6/*`; then the inner sweep `generated_inner_sweep_outcome` (conditional on `InnerReady`), the outer sweep `generated_outer_sweep_outcome` (variant `WINDOWS + groups - step`), `generated_outer_none_covers_steps`, tail surjectivity `generated_tail_group_surjective` and the per-group join `generated_outer_none_covers_groups` are checked and registered, with lane-zero, outer-bound and masked-tail-offset controls (group 7 of 8 unreachable) each baseline/red/restore 0/1/0.
- Shared-gate note: package-wide bridge and `AxiomAudit` are green at generated `8e76b917…` (`lake build` 3817 jobs; `AxiomAudit` 3692 jobs, no errors, closed cone). The readiness derivations landed (`inner_ready_of_scan_geometry`, `outer_step_ready_of_geometry`, with `generated_outer_increment_checked` and `generated_group_bounded`), controlled by a payload-length mutation, and `repR_region_outer_ready`/`repR_stored_region_outer_ready` wire the admitted representation into the outer readiness (control rejects three malformed-region fixtures). Admitted grouped sizing/headroom and the positive-summary branch landed (`grouped_admission_of_checked_branch`, `repR_summary_read_branch`, `repR_live_spill_summary_positive`; control rejects four malformed fixtures); The global invariant landed partially (`StoredLiveTagAddress`, `StoredTaggedRepR`, `word_stored_table_new_tagged_repR` on the real generated empty constructor, `word_stored_apply_insert_found_spill_keeps_tags`, `generated_publish_live_frames`, `stored_live_tag_address_publish_spill`; control omits the generated `publish_live` tag commit and the readback proof rejects, restored). The rebuild traversal landed (xiaomi-pro; `Representation.lean` `b58877c8…`, `ReadLoop.lean` `d82c4683…`): per-step relocation run/publish/readback with the unchanged-destination frame, and the traversal induction `word_stored_relocate_all_keeps_reserved_tags_from_prepared` from the initial invariant; gate 3351 jobs; two mutation controls (the second rejects inside the new per-step theorem at `Representation.lean:1041`/`:1079`), restore green. Residuals: (1) the `hstep` certificate supply: `retarget_held` totality from `sourcePayload` alone is **disproved** — the success characterization `retarget_held_succeeds_iff`, the five branch equations and the refusal witness `retarget_held_refuses_of_linked_handle_fail` are proved and registered (`Relocation.lean` `75ed8969…`, control red at `:859`, restored), and the traversal therefore needs a locator well-formedness premise making `linked_handle` total, or a repaired plan-side model — **lead decision (user direction): repair the structure and prove it out; no weakening** (W5-T14 proves the well-formedness predicate; if totality requires the plan model to carry the generated summary commit, repair it there rather than weaken); (2) the `ReadLoop.lean` size breach is closed: the state/invariant/relocation block moved verbatim into the new `Growable/ReadState.lean` (`e30382b4…`, 526 lines; `ReadLoop.lean` now `676bf5ab…`, 1152 lines; lakefile root added; all names/statements unchanged, gate 3352 jobs, `AxiomAudit` green unchanged at 3693 jobs, and the mutation control rejects inside the moved block); (3) finding for the lead: `GrowableRelocation.RelocationEffectsPrepared.publish` is `∀ region`-quantified and unsatisfiable at spill destinations (`publish_live` fails out of bounds), so the landed `Growable/Relocation.lean` spill theorems are vacuous there — the new theorem chain uses explicit satisfiable premises instead; **lead decision (user direction): repair the `publish` premise structurally and prove the spill theorems at satisfiable premises, no record-and-relabel** (queued behind W5-T14 on the same file); (4) the traversal task ran ≈2 h 15 min active, over its 45-minute working budget (recorded deviation; no per-command budget exceeded). Logical absence composes after the `hstep` supply.
- Row 5 feasibility verdict (spike, no file change): feasible with new generated-body theorems, no new extraction. Residuals: (1) outer-loop measure `WINDOWS + groups - step` with the "every earlier group completed while `hit = none`" invariant; (2) inner-loop coverage via `liveLaneCount` and `loop.spec_decr_nat`, since neither generated loop carries a variant; (3) tail-group surjectivity for `mask = groups - 1` with power-of-two `groups`, joined to the outer invariant. `match_tag_exact` plus the removal trio then enumerate each group's set bits in increasing lane order; the stored matcher must still connect each comparison to full `(Sigma, key)` equality, and `none` from `spill_read` needs the admitted-geometry/successful-index and summary-recordedness premises. Outer-step coverage landed too (`generated_outer_sweep_outcome` with variant `WINDOWS + groups - step`, and `generated_outer_none_covers_steps`; control exits 0/1/0, restored digest `2a1138e4…`). Row 5 is **conditionally closed**: `generated_spill_read_unfold`, `spill_read_none_implies_no_full_match` and `spill_read_some_implies_full_match` (plus the whole-shard corollary and the rebuild-certificate wrapper `spill_read_after_relocate_all_none_implies_no_full_match`) are proved and registered; the tag/address-disagreement control rejects five fixtures semantically. Both decisions are resolved by lead direction (repair structurally and prove out; no weakening, no relabel): the traversal `hstep` is supplied by the W5-T14 well-formedness predicate, and the `RelocationEffectsPrepared.publish` spill premise is repaired at satisfiable premises. Row 6 (`Growable.Read`) — the flat/resident read face and public read refinement — is next.
- Locator loop totality (`Growable/RelocationWF.lean` `3d0f9897…`, new; root in `lakefile.lean`): the width premise `4 * UScalar.max .U32 + 3 ≤ UScalar.max .Usize` (false on 32-bit `usize`; stated in the module docstring), `cast_usize_val` via `UScalar.cast_inBounds_spec`, projection normalization, `find_edge_loop_succeeds`, `find_edge_succeeds`, `walk_residue_loop_succeeds`, `walk_residue_succeeds`, `identity_terminal_val`, `residue_identity_val`, `vec_deref_val`, `walk_identity_loop_succeeds`, `walk_identity_succeeds`, and `directory_walk_succeeds` (discharging the `List.setSlice!` readback obstacle) are checked. Continuation 3 (`APEIRON-READLINK-W5-T14-CONT3`, Gemini Flash) proved `linked_handle_succeeds` and `retarget_held_hstep_of_wf` (closing the traversal `hstep` from locator well-formedness), with the mandatory mutation control executed in `/tmp/opencode/control_w5_t14/` (injected panic in throwaway `ApeironWord.lean` rejected semantically with application type mismatch; baseline/red/restore 0/1/0, live digest `8e76b917…` verified); all 15 declarations registered in `AxiomAudit.lean` and checked green at 3696 jobs. The locator well-formedness and traversal `hstep` supply is closed. Open on the relocation front: the `RelocationEffectsPrepared.publish` satisfiable-premise repair.
- Row 6 first slice (`Growable/Read.lean` `1eb64f08…`, new; root in `lakefile.lean` `bb15f648…`): `ConstructorReachableRepR` and the direct-hit partition at both public roots (`word_stored_table_get_at_direct_hit`, `word_regen_table_get_at_direct_hit`) are checked and registered. Continuation 1 landed the localization core (`query_rep`, `absent_of_exclusions`, `default_of_absent`; gate 3353 jobs). Continuation 4 landed `spill_read_none_of_summary_miss` (`268842a2…`). Continuation 5 (`APEIRON-READLINK-W5-T13-CONT5`, Gemini Flash) landed flat index pinning (`flat_index_none_pin`, `flat_index_some_pin`, `flat_index_compute`), Partition 2 Shape A direct-word none (`word_stored_table_get_at_direct_none`), Shape B flat none (`word_stored_table_get_at_flat_none`), and Shape C vacant flat slot (`flat_vacant_entry_none`, `flat_vacant_exclusion_of_prepared`, `word_stored_table_get_at_flat_vacant`) at `1eb64f08…`; module gate exit 0 (3353 jobs, 1.9 s); cone clean over all 19 declarations; all registered in `AxiomAudit.lean` and checked green at 3696 jobs. The mandatory mutation control ran in a disposable workspace copy against `table.Table.get_at` returning 7/5 mutations, rejected semantically with `Expression did not evaluate to \`true\`` and proof mismatches, restored clean (0/1/0, logs in `/tmp/opencode/readlink-w5-t13-control-*.log`, `8e76b917…` verified). Continuation 6 (`APEIRON-READLINK-W5-T13-CONT6`, Gemini Flash) landed Partition 3 (occupied non-match at the stored root): `flat_occupied_ne_exclusion`, `flat_occupied_exclusion_of_prepared`, and `word_stored_table_get_at_flat_occupied_miss` at `cb3bfa49…`; module gate exit 0 (3353 jobs, 2.79 s); cone clean over all 22 declarations, zero `sorryAx`; all registered in `AxiomAudit.lean` and checked green at 3696 jobs. Residual: Partition 5 (complete positive sweep) at the stored root.
- Row 7 interface (`Growable/Search.lean` `a52502ad…`; no consumer file forced): `absenceCovered` is replaced by the three-way `AbsenceCertificate` (`directEmpty`; `directExcludedSummaryMiss` with the cycle-safe `SummaryMissSound` mirror; `coverage` byte-identical), plus the route-bridge fields `flatMatchAdmitted`/`spillMatchRouted` on `SearchDecisionCorrect`; `word_{stored,regen}_search_decision_simulates` statements unchanged and `absent_of_not_found` re-proved over the three routes; `Search` still imports only `ApeironWord`/`Table.Basic`/`Geometry`/`Identity`. Gate `lake build Growable.Search` exit 0 (3345 jobs); control in a disposable copy 0/1/0 (cleared live bit with a live payload; limitation: `summary_maybe` is not eval-reducible, so the check inspects the concrete summary word and `entryAt`), restored digest `8e76b917…`. Consumer rebuild, package audit and the row 7b producer (`Growable/Classifier.lean`) are the next tasks.
- Row 7 producer first slice (`Growable/Classifier.lean` `fd16f901…`, new; lakefile root `23866265…` with `Growable.Search`/`Growable.Read`/`Growable.Classifier` added): `summaryMissSound_of_repR`, `flatMatchAdmitted_of_repR`, `spillMatchRouted_of_repR` (the latter two from `RepR.address` via `preparedFlatIndex` and `RepR.directIndex`), `repR_absenceCertificate_of_summaryMiss` (the `directExcludedSummaryMiss` constructor) and `repR_absent_of_summaryMiss` are checked and registered; module gate `lake build Growable.Classifier` exit 0 (3350 jobs), in-build cone per declaration; control in a disposable copy perturbs generated `table.shard` arithmetic and rejects the live-spill routing fixture semantically (0/1/0), restored digest `8e76b917…`. `summaryMissSound_of_repR` is a `def` (Type-valued structure, compiler-forced). Residual: `foundExact`/`vacantExact`/`exhaustedExact`/`complete`, first fit, `FreshLengthBound`, `GrowableCallbackLaws`, `DecisionSymbolicCorrespondence`, `SpillVacantDirectGuard`. W7-T3 (`Classifier.lean` `de0706fa…`) wrote the generated-decision provenance casework and the spill-route missing-lemma statement (`SpillLocateFoundFullMatch`: a `spill_locate` `.Found` carries no full-match certificate today — the inner loop computes `matches` but does not return it); re-run at the green Relocation state, the slice has real errors (phantom `F` in `DecisionProvenance`, two `Result.ok_injective` rewrite motives, unused simp args, `:333` no-goals, and an auto-sorry caught by the in-module cone check) — `W7-T3-CONT` repaired it at `Classifier.lean` `ffbdb84a…`: the phantom `F` binder is removed, decision equalities use `table.SearchDecision.mk` plus `subst`, the named unused simp arguments are dropped, and the in-module cone check reports all 14 declarations inside `{propext, Classical.choice, Quot.sound}`, no `sorryAx`; gate `lake build Growable.Classifier` exit 0 (3350 jobs, 2.89 s); a disposable copy with the generated flat `matches` guard dropped is refuted by the concrete non-match fixture through `search_decision_found_flat_guards` (baseline/red/restore 0/1/0, live tree untouched, `8e76b917…` verified). Registered: `DecisionProvenance`, `search_decision_provenance`, `search_decision_wellFormed`, `search_decision_found_flat_guards`; `SpillLocateFoundFullMatch` stays a statement (the `spill_locate` inner loop computes `matches` but returns no certificate). Open: `entryAt`/`foundDenotes` packaging, `vacantExact`/`exhaustedExact`/`complete`, first fit, `SpillVacantDirectGuard`, `SearchDecisionCorrect` production.
- Attempt history: earlier retries blocked on concrete generated reductions (`decide +revert` without `Decidable` synthesis, `simp` leaving the six-stage body, `rfl` not definitionally equal); the source-linked stage invariant closed `least_set_lane`. The same class in the address lane was closed by `spread_zero` (`Native/LocatorSpec.lean` `ec2f4e0d…`).
- Generated identity: `proof/ApeironWord.lean` is now `8e76b917…` after `proof/regen.sh` and `--check` exit 0; control digests `2a1138e4…` and `d0a34ee5…` are historical.
- Row 7 producer `W7-T4` (`Growable/Classifier.lean` `94161ef2304aeec433d54b08b36b8562719f6de762345748760ee75e7ac1c999`, 641 lines): flat found packaging `stored_entryAt_flat`/`regen_entryAt_flat` (weakening through `vec_index_readback` and `table_matches_of_guards`), logical denotation `foundDenotes`/`foundDenotes_of_repR`, the generated-decision links `search_decision_found_flat_denotes_stored`/`_regen`, and spill packaging `stored_entryAt_spill`/`regen_entryAt_spill` are checked; module gate `lake build Growable.Classifier` exit 0 (3350 jobs) with the in-module cone reporting the new declarations inside `{propext, Classical.choice, Quot.sound}`; controls in `/tmp/opencode/control_w7t4/` (flat value mutation, matcher-polarity mutation) baseline/red/restore 0/1/0 with proof diagnostics plus the in-module cone rejecting `sorryAx`, restored `8e76b917…`. All ten registered in `AxiomAudit.lean` (`b27f0c4bd7c9aec8c3eb8020da114b2d9224c9a5071227b9e0d4ed7d8d4c0bfe`); the barrier re-run is pending the in-flight `Growable/Read.lean` partition-5 lane (mid-edit red at `Read.lean:809`, `:819` `omega` goals). Residual: the `SpillLocateFoundFullMatch` completion invariant for `spill_locate` `.Found` (inner loop computes `matches` but does not return it), then `vacantExact`/`exhaustedExact`/`complete`, first fit, `QueryAddressCorrect`, `FreshLengthBound`, `GrowableCallbackLaws`, `DecisionSymbolicCorrespondence`, `SpillVacantDirectGuard`.
- Row 6 continuation (`APEIRON-READLINK-W5-T13-CONT7`, budget-stopped at 48m): landed
`spill_read_some_tagged_full_match`, `spill_read_some_entryAt`, `spill_read_none_entryAt_ne` (and two
private index-readback helpers) in `Growable/Read.lean`, now `866d035d…` (1,019 lines, gate 3353 jobs);
all three registered in `AxiomAudit.lean`. The stored-root composition (`valid_n_16`, the six
`_hit`/`_sweep_none` route theorems, `StoredReadRoute`, `word_stored_table_get_at_root`) was cut before
the file could go green: `table.valid_n 16#usize = ok true` leaves
`core.num.Usize.is_power_of_two 16#usize` after `unfold table.valid_n table.MAX_GROUP` and `decide`
cannot synthesize the residual `Decidable`; the flatVacant route additionally needs the
`word_stored_table_get_at_flat_vacant` argument set. Draft preserved at
`/tmp/opencode/read-composition-draft.lean`; continuation `W5-T13-CONT8` in
`Growable/ReadComposition.lean` (out of roots until green; lead adds the root). With Read green the
barrier passed: `lake build AxiomAudit` exit 0 (3877 jobs) and bare `lake build` exit 0 (4003 jobs).
- Row 6 continuation `APEIRON-READLINK-W5-T13-CONT8` (Gemini, ~12 min): the stored-root composition is
closed in the new module `Growable/ReadComposition.lean`
(`f5f29731797d9d3ca59a5e539b5267eec1431c6fa4141a35d9f38bceb20036fd`, 529 lines): `valid_n_16` (unfold
`table.valid_n table.MAX_GROUP core.num.Usize.is_power_of_two`, then `rfl`), the six route theorems,
`StoredReadRoute` and `word_stored_table_get_at_root`; gate exit 0 (3354 jobs; root added to
`lakefile.lean` `4072b7c5…`), local cone closed, seeded root-equality mutation red with type
mismatches, restore green. All nine registered in `AxiomAudit.lean` (`5006a792…`, 3880 jobs). The
public read refinement now covers all partitions and the composition at the stored root; next:
paired fixtures and the composition control, then the `RelocationEffectsPrepared.publish` repair.
- Row 6 continuation `APEIRON-READLINK-W5-T13-CONT9` (DeepSeek Flash, ~30 min):
`Growable/ReadComposition.lean` is now
`70d90c215aa78fee5897dcfd63f32681f5bc82e962aae316e468b091c37ec46a` (737 lines): 10 concrete
Stored-carrier fixtures, 12 value definitions, and 19 kernel-evaluated `#guard` checks
(positive/negative pairs on the direct route, `entryAt`, flat index, `summary_maybe`, `spill_read`
and the public read), each matched through `Result.match` because `Result I64` has no `BEq`. Gate
`lake build Growable.ReadComposition` exit 0 (3354 jobs) with the ten new declarations inside
`{propext, Classical.choice, Quot.sound}` and the local cone list intact; control in
`/tmp/opencode/w5t13-control/` (spill payload `42#i64 → 43#i64`) baseline 0 / red 1 / restore 0,
red attributable at `ReadCompositionRed.lean:693:0` (`did not evaluate to true`); imported digests
unchanged (`8e76b917…`, `866d035d…`). **Blocker named:** the six route theorems and
`word_stored_table_get_at_root` admit no concrete instantiation because no concrete
`ConstructorReachableRepR` value exists — the sole producer `word_stored_table_new_repR` is a WP
triple over `table.Table.new` whose empty state is exposed only through the private
`table_new_empty_repR` (`direct = fun _ => none`, `live = fun _ => False`); non-empty reachability
is row 4/8 work. The fixtures land the strongest evaluable evidence instead; no theorem was
weakened. Barrier re-run pending the in-flight `Relocation` repair lane.
- `APEIRON-READLINK-W5-T15` (Gemini, ~18 min; independent verifier xiaomi-pro, verdict **accept with
residual**): `Growable/Relocation.lean` `24a8eeb66fd68ad10b387f5e44d5db1540c9704a954918547ae76b8cc752e192`
repairs `RelocationEffectsPrepared.publish` at the prepared spill facts (added `reserved`; premises
`sourcePayload`, `address_held`, `reserved.regions[shard]`), four consumer names and conclusions
unchanged; gate 3344 jobs, controls red (dropped live-tag condition; out-of-bounds region premise) and
restored; consumers `Growable.Read`/`ReadComposition` 3354 jobs and
`RelocationWF`/`Refinement`/`Rebuild` + `AxiomAudit` 3880 jobs green. The verifier re-ran the gates and
its own region-index mutation (`/tmp/opencode/verify-w5t15/`; red attributable at
`Relocation.lean:520:33`/`635:33`, restore `24a8eeb6…`) and confirmed the premises are satisfiable,
non-circular and pinned to the prepared facts. Residuals: the stale note at `Growable/Read.lean:30`
was corrected in the same lead pass (`Growable/Read.lean` now
`311db56531fd7cac2bde6251454a828471f9b6d6df26223816201a31fa0cc946`; gate `Growable.Read`
+`ReadComposition`+`AxiomAudit` 3880 jobs green); the pre-change baseline `0a8ac79c…` is
unrecoverable from git (HEAD `74c037b`, no commit), so binder-level comparison rests on the recorded
old statement; `RelocationEffectsPrepared` remains a caller-supplied effect certificate by design.
- Locate feasibility spike (`APEIRON-READLINK-SPIKE-LOCATE`, xiaomi-pro, ~26 min of 45; report
`/tmp/opencode/readlink-spike-locate.md`, probe
`/tmp/opencode/spike-locate/proof/corpus/Growable/LocateLoopProbe.lean` `4c99c366…`, 260 lines):
**yes** — the `spill_locate` `.Found` completion connects using existing generated-code infrastructure.
Proved in the probe: the inner completion (`generated_locate_inner_outcome` with
`LocateInnerInv`/`Occurrence`) and `generated_spill_locate_clears_selected`; one statement-only link
remains, `least_set_lane_loop_selects` over the loop-based `table.least_set_lane_loop` (distinct from
`table.scan.least_set_lane` — the key structural find and duplication hazard). Minimal set: L1 scanner
spec, L2/L3 proved, L4 outer 6-tuple sweep, L5 unfold (no `valid_n`/summary gate), L6 projection via
`InnerCompared` + `vec_index_readback`; module plan `Growable/LocateLoop.lean` 450–600 lines;
`Found` forces `may_match = true`. Follow-up `APEIRON-READLINK-W8-T1` (same session) promotes the probe
into the live module and proves L1; lakefile root and `AxiomAudit` registration are lead steps at
integration.
- Locate-loop slice 1 (`APEIRON-READLINK-W8-T1`, xiaomi-pro, ~32.5 min of 45): `Growable/LocateLoop.lean`
`bd9b9af2bf9a4c4cfbf0e22e55c2b2ed233a79f1a0c72aacf91f81f68bbaeda5` (456 lines) proves
`least_set_lane_loop_selects` via `loop.spec_decr_nat` over `table.least_set_lane_loop`, plus
`least_set_lane_loop_in_range` and `scan_lane_of_loop_lane`, carrying
`generated_spill_locate_clears_selected` and `generated_locate_inner_outcome`; gate
`lake env lean -DautoImplicit=false -DwarningAsError=true` exit 0 (zero sorry/warnings), control
0/1/0 with the matched-step mutation attributable at `LocateLoop.lean:322:53`. Lead integration: root
in `lakefile.lean` `9b43f92d…`; five declarations registered in `AxiomAudit.lean`
`31a612255ba28f366b4949d022e9cbb8cbef3d540ef042f07eb83eaab520656a` (3890 jobs, cone closed);
`ApeironWord.lean` unchanged.
- Locate-loop slice 2 (`APEIRON-READLINK-W8-T2`, xiaomi-pro, ~48–50 min of a 45-min budget —
  overrun, elapsed estimated): `Growable/LocateLoop.lean`
  `fdc127ccb58071ca8a64654142564d34b28ee9279f33d0f00abac8ff0fcb85ed` (795 lines) proves L4
  `generated_locate_outer_outcome` through `LocateMatchWitness`, `LocateVacantWitness`,
  `LocateOuterStepMiss`, `LocateOuterStepReady`, `LocateOuterInv` and `LocateOuterOutcome`; gate
  `lake build Growable.LocateLoop` exit 0 (3353 jobs), control 0/1/0 with the matched-step mutation
  attributable at `LocateLoop.lean:672:59`, no forbidden constructs, `ApeironWord.lean` unchanged.
  Lead integration: nine declarations registered in `AxiomAudit.lean`
  `04dd2c9b0fb85e76a89c4ae6c792ceb340b834134f3542dbd1c6ec8364b8e7db` (3890 jobs, cone closed).
- Locate-loop slice 3 (`APEIRON-READLINK-W8-T3`, xiaomi-pro, 22 min of 45 — within budget):
  `Growable/LocateLoop.lean` `062d2a449924592472ebc809e7200f9588eb41579747b4ae9d0f7d686132526d`
  (990 lines) proves **L5** `generated_spill_locate_unfold` (a `SpillSite.Found index` result unfolds
  to `tag`/`summary_maybe` ok and a completed inner sweep with `b1 = true, i2 = index`; `valid_n` and
  summary gating absent, mirroring `generated_spill_read_unfold`) and the **geometry discharge**
  `locate_outer_step_ready_of_geometry` for `LocateOuterStepReady`, with two explicit premises
  (`hmatcher` callback termination; `hmatchEmpty` per-line `table.scan.match_empty` totality — no
  empty-scan spec exists in `Growable/Scan.lean`) plus `hwide : 64 ≤ n.val`. Gate
  `lake build Growable.LocateLoop Growable.Classifier` exit 0 (3354 jobs), control 0/1/0 (mutation
  drops the `b2 = true` pin in the L5 Found link) attributable at `LocateLoop.lean:899:14`,
  `ApeironWord.lean` unchanged; `Growable/Classifier.lean` untouched. Lead integration: both
  declarations registered in `AxiomAudit.lean`
  `8fb29916b23095a63076506fee5e6bd2de6835176bc53bda031b33ae9fac9f1e` (3890 jobs, cone closed).
- L6 blocker (why `spill_locate_found_full_match` does not follow from L3–L5). The `⦃ ⦄`/`spec`
  framework asserts ok-ness, so `generated_locate_inner_outcome`/`generated_locate_outer_outcome`
  carry per-lane readiness premises (`InnerReady`). A `Found` run cannot satisfy `InnerReady` at its
  found step — the inner sweep stops at the matching lane and never compares the remaining selected
  lanes — so the outcome theorems cannot be instantiated on a `Found` run. The certificate must come
  from run-relative inversion: (1) inner-body found inversion, (2) inner-loop found extraction
  (equation-lemma induction on `liveLaneCount`), (3) outer-body found inversion, (4) outer-loop
  extraction, (5) assembly with L5 + `InnerCompared` + `vec_index_readback` (`payload := entry`, plus
  a structure-eta step for `table.matches`'s address argument); est. 250–300 lines.
- Locate-loop slice 4 (`APEIRON-READLINK-W8-T4`, xiaomi-pro, 27 min of 45 — within budget):
  `Growable/LocateLoop.lean` `ba61417af6708c0f7289eb9d3ffc121cfc6c81b56cb45381767bc6d3bc0027d8`
  (1126 lines) proves step 1 of the L6 chain: `LocateSlotCertified` (`∃ entry, Vec.index … slots out =
  ok entry ∧ table.matches … entry … = ok true`) and `locate_inner_body_shape` — full inner-body
  inversion over all four `Result` outcomes with `liveLaneCount next + 1 = liveLaneCount hits` via
  `scan_lane_of_loop_lane` + `generated_clear_hits_live_decrease`. Gate exit 0 (3354 jobs), control
  0/1/0 (mutation `hmatch` → `hidx` in the found-link certificate) attributable at
  `LocateLoop.lean:1124:48`, `ApeironWord.lean` unchanged, `Classifier.lean` untouched. Lead
  integration: both declarations registered in `AxiomAudit.lean`
  `a47765e826d4c600d0dc7c9903db0f247e2ed4273ef93082e24f34ef721644f9` (3890 jobs, cone closed).
  Steps 2–5 are blocked on the `loop.eq_def` residue: after `unfold` + `rw [loop.eq_def]` + `dsimp
  only`, `hres : Std.bind (body …) (fun r => match r with | cont x => loop … | done x => ok x) = ok
  result` refuses term-level coercion against an alpha-identical printed type (`LocateLoop.lean:1157:60`;
  hidden implicit difference suspected, `pp.all` not yet run), and `bind_tc_ok`/`bind_tc_vis` patterns
  do not match the residue. `APEIRON-READLINK-W8-T5` (same session, running) resolves the residue —
  `pp.all` diagnosis, `convert`/`show`, or a wrapper-vs-`loop` equation lemma — then lands steps 2–5;
  if the residue resists inside a 15-min bound, the alternative is `loop.spec_decr_nat` with a custom
  found-flag postcondition (the W8-T1 recipe). Est. 150–200 lines remaining.
- Locate-loop slice 5 (`APEIRON-READLINK-W8-T5`, xiaomi-pro, 22 min of 45 — within budget):
  `Growable/LocateLoop.lean` `7908c76468ae8330e0a0e1f0b1e85da0bdf220376ef6e88e81c73db17e94cd64`
  (1310 lines) resolves both residues and proves step 2. Residue diagnosis (persist this):
  (1) the `loop.eq_def` RHS continuation carries a `ControlFlow` match motive that no hand-written
  restatement reproduces — `exact`/`have`/`rfl` refused on printed-identical types; `convert hres using 2`
  decomposes the equation and the continuation equality closes by `funext r; cases r` — case-splitting
  the scrutinee dissolves the motive gap; packaged as `wrapper_loop_form` (proved `rfl`) plus a
  raw-to-`do`-form bridge. (2) `bind_tc_vis`/`bind_tc_div` are theorem-level over the ITree
  coinductive bind, not defeq, so raw `Std.bind` hypotheses cannot be coerced and `bind_tc_ok`
  patterns never match; bridge to the `do`-notation form first and case-split the run variable before
  any rewriting (`rw … at hres` pollutes the motive: "motive is not type correct"). Proved:
  `wrapper_loop_form` and step 2 `generated_locate_inner_found` (`spill_locate_loop0_loop0 … false
  prior base hits = ok (true, out)` implies `LocateSlotCertified … out`, by `Nat` induction on
  `liveLaneCount` over `loop.eq_def`, consuming `locate_inner_body_shape` in the `done`/`cont` cases).
  Gate exit 0 (3354 jobs), control 0/1/0 (found-link `hmatch` → `hidx`) attributable at
  `LocateLoop.lean:1124:48`, `ApeironWord.lean` unchanged, `Classifier.lean` untouched. Lead
  integration: both declarations registered in `AxiomAudit.lean`
  `798d26338d81cfbddafdc5a8e327882df85671f542ea44a39c0af33f10e31ba4` (3890 jobs, cone closed).
- Remaining steps 3–5 (~150–200 lines mechanical): outer-body inversion for the 6-tuple
  `spill_locate_loop0.body` (found-step implies the inner-call equation; `done` carries `(b1, i2)`;
  `cont` advances the cursor by one), outer-loop extraction on `limit − step`, and the Classifier
  assembly from L5 + (4) + `vec_index_readback` (`payload := entry`) + the `table.matches`
  structure-eta step. `APEIRON-READLINK-W8-T6` (same session, running) executes them.
- Locate-loop slice 6 (`APEIRON-READLINK-W8-T6`, xiaomi-pro, 32.3 min of 45 — within budget):
  **`SpillLocateFoundFullMatch` is proved.** `Growable/LocateLoop.lean`
  `a745a9172ed773b7a8d71727d4a7c8ac920a27a5f96f31278fa9b0b70c4efb20` (1666 lines) adds
  `bind_ret_inv` (private, generic `do`-bind inversion), `outer_wrapper_loop_form`,
  `locate_outer_body_shape` (step 3: `done` carries `(b, i1, b1, i2)`; every `cont` advances the
  cursor by exactly one and certifies a recorded match slot via `generated_locate_inner_found`; the
  vacancy-scan internals need no inversion because the pair value is existential in the conclusion),
  `generated_locate_outer_found` (step 4: `spill_locate_loop0 … = ok (b, i1, true, index)` implies
  `LocateSlotCertified … index`, by `Nat` induction on `Usize.max − step.val` over `loop.eq_def`),
  and `generated_spill_locate_unfold_run` (step 5a: premise-free run-relative unfold; every
  `Grow`-shaped branch closes by `Result.ok_injective` constructor injection, `Found index` pins the
  sweep's `(b1 = true, i2 = index)` run). `Growable/Classifier.lean`
  `9a04fbce34232df3a26149e22070e06734731572cf7c52d725fa739df974de53` (659 lines) adds
  `spill_locate_found_full_match` (step 5b: L5 unfold-run → outer-found → `LocateSlotCertified` →
  `vec_index_readback` with `payload := entry` plus the `table.matches` structure-eta step for the
  address argument; the `SpillLocateFoundFullMatch` statement is unchanged). Gate
  `lake build Growable.LocateLoop Growable.Classifier` exit 0 (3354 jobs), control 0/1/0 (mutation
  drops the `vec_index_readback` step in the assembly) attributable at `Classifier.lean:657:16`,
  `ApeironWord.lean` unchanged. Lead integration: five declarations registered in `AxiomAudit.lean`
  `8cf72c7839ce23606cb4a9752e41045eef63375ca1efd316f84de392ddaeee57` (3890 jobs, cone closed).
  Residuals: the `LocateOuterStepReady` geometry premises stay explicit, and the per-lane
  miss-content claim is asserted nowhere.

### W7 discharge and stop condition

The seven ledger names in `docs/kernel-boundary.md` are not seven proved inputs:

| Boundary premise | Source-pinned producer and required exact content | State before a W7 task can open |
|---|---|---|
| `SearchDecisionCorrect` | Generated `table.Table.search_decision`/`table.probe.spill_locate`; derive every field of the existing structure for each returned decision, including first fit and corrected absence certificate. | Row 7 producer, `lake build Growable.Classifier AxiomAudit`, semantic late-hit/order controls. |
| `DecisionSymbolicCorrespondence` | Generated `table.Table.search_decision` and `Growable.Resources.DecisionSymbolicCorrespondence`; prove generated runtime class/location equals the symbolic certificate used by `apply_insert`. | No producer theorem; exact runtime/symbolic witness construction needs a feasibility result, then focused `lake build Growable.Resources AxiomAudit` and class/location mutant. |
| `QueryAddressCorrect` | Generated `WordHome.address`, `table.shard`, `Table.search_decision`; derive actual query route and direct index from the same address, with no supplied desired location. | `Growable.Search` interface exists, source-bound producer absent; `lake build Growable.Classifier AxiomAudit` and wrong-shard mutant. |
| `FreshLengthBound` | Generated `table.Table.apply_insert` fresh branches; prove checked `len + 1` succeeds at every admitted vacancy and has exact resulting value. | Predicate exists in `Growable.Search`; source-bound producer absent; `lake build Growable.Classifier AxiomAudit` and length mutant. |
| `GrowableCallbackLaws` | Generated Stored/Regen hold and `WordMerge.merge` calls in `table.Table.apply_insert`; prove concrete new-first result and exactly one merge for each successful branch. | Structure exists in `Growable.Search`; callback trace producer absent; `lake build Growable.Classifier AxiomAudit` and reverse-order mutant. |
| `SpillVacantDirectGuard` | Generated direct route plus `spill_locate`; establish fresh spill publication cannot leave the query's admitted direct slot vacant under `DirectCover`. | No declaration or source-bound theorem; first obtain a checked feasibility/witness task, then `lake build Growable.Classifier AxiomAudit` and direct-vacancy mutant. |
| `ResourceAdapterSimulates` | Generated `table.Table.plan_rebuild`, `reserve_rebuild`, `observe_receipt`, `materialize_rebuild`, `prepare_from_plan`, `prepare_rebuild`, and locator staging plus real Rust reservation; show same-input prepared and refusal branches match the `Growable.Resources` relation, including actual capacity, initialization, no callback on refusal and stuttering state. | These generated declarations exist and the active Vec model carries `cap`; syntax/root absence is not the blocker. `Vec.try_reserve_exact` selects success on representable need, while `Vec.push` reconstructs through `Vec.from`, resetting `cap` to new length after a reserve. Missing are ordinary allocator may-failure, reserve/content/capacity and later-push preservation, and source-to-adapter effect correspondence. Generated `prepare_from_plan` returns `self1` on locator reservation `Err`, while `ResourceAdapterSimulates.refused` requires `committedAfter = state`; prove logical/physical stutter for this branch or repair the source/model contract before asserting the relation. A separately accepted feasibility result must adjudicate these exact counterexamples; only then can a focused `lake build Growable.Resources AxiomAudit` and real refusal/late-capacity mutants gate a source-bound proof. Rust refusal tests are `tested` only. |

The final row is the decisive missing primitive for unconditional production W7 composition. The absent generated constructor wrappers limit only a claim about those public wrappers; generated `table.Table.new` is available for concrete constructor instance refinement. Existing `Resources.lean` constructors are conditional certificates, not the production adapter effect link. The table identifies producer obligations but is **not** a finite *discharge* map under the governing reopening rule because several exact compiled producer statements, checked primitives and an accepted may-fail/effect correspondence gate are absent.

Do not reopen `W7-SearchCommitBoundary` on this map. The governing reopening condition in `docs/direction.md` requires a Sol-ratified source-pinned **task** for full `apply_insert` preservation, an acyclic target, joint found/fresh flat/spill witnesses and a checked primitive/tool dependency for every boundary premise. Rows 4, 7 and 8 are not dispatch-ready until the missing constructor/loop/adapter primitives have feasibility results and exact compiled signatures. Once they do, put the whole-state theorem in a module after `Representation`/`Resources` and its dependencies, with `AxiomAudit` last; do not make `Search` import `Representation`. The W7 theorem and G1/REG-1 remain open. No claim here reaches rustc/LLVM, ABI, GPU, unsafe container, generic divergent callbacks, allocator behavior or performance.
