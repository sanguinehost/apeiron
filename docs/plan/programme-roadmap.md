# Programme documentation consolidation

## Intent

The user directs the programme; AI writes its code. Commit authors and optional trailers do not
partition that direction. The approved decision retains Lean and consolidates canonical mathematics,
reusable Rust primitives and implementation refinements in Apeiron. Song remains an adjacent,
independent assembly-up implementation. Correspondence between implemented primitives is required;
Rust linkage or replacing Lean is not the method.

Reuse `docs/direction.md` as the programme authority. Do not create a second programme roadmap.
Local roadmaps retain implementation detail, gates and release acceptance. A programme outcome may
be required without being privileged, an immediate task, or a prerequisite for every local release.

Rejected alternatives: retiring Lean loses a retained asset and contradicts the user; merging Song
into the Rust implementation loses its independent assembly-up boundary; making every programme
outcome an A0 prerequisite prevents bounded foundational work; maintaining duplicate master ledgers
recreates the current drift.

## Specification

- Retain Lean sources, packages, history and attribution. Current source locations and Git/Lake
  pins remain unchanged until a separately gated migration.
- Apeiron owns canonical definitions and reusable Rust/refinement work. Song owns its assembly,
  Slang implementation, contracts and local build/runtime acceptance. Applications retain policy.
- A primitive correspondence relates both executing implementations to the same canonical
  definition on matched domains, including representation, state/effects, refusal and resources.
  Equal names, example outputs, citations or identical layouts are not that proof.
- Preserve programme outcomes for algebraic computation, combinators, lift, checked subsumption,
  realization, measured transformation benefit, active test-time learning, physical models and
  federation. Scope/gate gaps remain open; no new software completion is claimed.
- Preserve Song A0-A6/B6, local Rust/Lean gates, axiom cones, controls, existing evidence and bounds.
  Foundational verification precedes the next large AI implementation. The joint readiness profile
  is an explicit specification obligation, not satisfied by A0 alone.
- Remove stale current-direction assertions, not mathematical source. Historical RFC publication
  artifacts stay frozen; their entry page points to the current programme authority.

This is an approved documentation pass on the supplied checkouts. No task branch/worktree is
created, no source/build lane is occupied, and no commit or push is authorized. Temporary checking
code lives under `/tmp/opencode/programme-docs/`, not on a production acceptance path. This combined
intent/spec/plan is the one retained lifecycle record for the pass.

## Input identities

| Checkout | Base revision | Pre-existing work to preserve |
|---|---|---|
| `apeiron` | `1403767e671c81737525702946a3eb1eae4f0b0b` | Clean |
| `sanguine-apeiron` | `8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f` | Clean; shares Git history with `sanguine` |
| `sanguine` | `e830d3fbdeb40319d1be0d1fc803e706748b6a3c` | Clean; predecessor worktree, not canonical root |
| `song` | `47869e76f982cfa3085a8618a82b80463e22ef46` | Clean; `song-ref` remains untouched |
| `penglai` | `ed11d3b34c148eb35050b615dc47e8340a9e224f` | Clean |
| `whispering-tides` | `e62ae9bca358bacb9fc5e042830f6a5059442c7d` | Clean |
| `fold` | `d497161f32e0d366220f2065ec36e680bd9aed86` | Modified `modelfold/src/arch.rs`, `lib.rs`; untracked CQ source/examples |
| `lyrium` | `665ccfe1da85e6b5e8d312cb324cc61211ca1640` | Modified `src/lib.rs`, `main.rs`, `tools/mod.rs`, `ui/render.rs`; untracked `.cargo/`, `examples/vfsreal.rs`, `src/render/` |

Existing unrelated branches/worktrees are inventoried, not removed. The baseline checker records
working-source digests, including nonignored untracked files, rather than attributing dirty inputs
to HEAD alone.

## Ownership and waves

Task IDs use `APEIRON-PROGRAMME-` followed by the table key. Paths are relative to the workspace.
All lanes are documentation-only. One owner per file; the lead alone maintains this plan.

| Task | Owner | Owned files | Completion |
|---|---|---|---|
| W0-T1 | Lead | `apeiron/docs/plan/programme-roadmap.md` | Intent/spec/plan recorded; `git diff --check` |
| W0-T2 | Flash gate worker | `/tmp/opencode/programme-docs/check.py`, `baseline.json` | Checker built, snapshot captured, expected red baseline observed |
| W1-T1 | Flash central worker; lead architecture review | `apeiron/docs/direction.md`, `docs/kernel-boundary.md`, `docs/flash-handoff.md` | Central authority, outcomes and local ownership consistent |
| W2-T1 | Flash Apeiron worker; Sol claim review | `apeiron/AGENTS.md`, `README.md`, `proof/README.md` | Entry points and current API/evidence agree with source |
| W2-T2 | Flash corpus worker; Sol claim review | `sanguine-apeiron/AGENTS.md`, `theory/ROADMAP.md`, `theory/POTENTIAL_CLAIMS.md` | Lean retained; current delivery/primitive correspondence distinguished |
| W2-T3 | Flash Song roadmap worker | `song/AGENTS.md`, `docs/ROADMAP.md`, `docs/research/ROADMAP.md` | Programme obligations coexist with unchanged local release barriers |
| W2-T4 | Flash Song architecture worker | `song/docs/DESIGN.md`, `docs/ARCHITECTURE.md`, `plans/ROADMAP.md` | Independent assembly-up boundary; obsolete Lean-retirement goals superseded |
| W2-T5 | Flash predecessor worker | `sanguine/AGENTS.md`, `fold/AGENTS.md`, `lyrium/AGENTS.md` | Current authority pointers, local autonomy and historical scope explicit |
| W2-T6 | Flash Penglai worker | `penglai/AGENTS.md`, `docs/substrate-port-scope.md` | Programme linkage; gameplay/adapter ownership and local gates preserved |
| W2-T7 | Flash application/network worker | `whispering-tides/AGENTS.md`, `ROADMAP.md`, `song/docs/network/ROADMAP.md` | Application science and federation remain programme outcomes, not core privilege |
| W2-T8 | Flash RFC worker | `apeiron/docs/rfcs/0001-core-architecture/README.md`, `ownership.md` | Frozen proposal clearly separate from approved programme; attribution corrected |
| W3-T1 | Lead | `apeiron/docs/plan/programme-roadmap.md` | Combined checks, review and preservation evidence recorded |

The central wording lands before the disjoint propagation tasks. W2 may run in parallel. Checker
snapshots and final evidence publication serialize; no worker writes another task's file or this
plan. A newly required path is sequenced and recorded before editing.

## Arc

```text
SCOPE:      The owned documentation above; one programme authority with independent local implementation gates.
REGISTER:   Planning and documentation; excludes implementation, proof promotion, source migration and publication.
CORPUS:     Governing documents and named combinator/table sources read; no theorem establishes repository policy.
REFERENCE:  /home/socol/Workspace/apeiron/AGENTS.md; /home/socol/Workspace/song/docs/ROADMAP.md
GATE:       python3 /tmp/opencode/programme-docs/check.py --task <ID> --controls; initially MISSING, W0-T2 creates it.
GRADE:      Architecture decision and checked documentation consistency; no new software or performance evidence.
```

Each worker emits this arc specialized to its owned files and reads its actual red result before
editing. W0-T1 uses `git diff --check`; W0-T2 uses the checker's self-test plus expected-red task
baseline. All later tasks use the single command above; final integration uses `--task all`.

## Documentation gate

The temporary checker captures HEAD, worktree files/digests and original owned text before W1.
It checks exact write allowlists, preservation of nonowned files and source/tool identities,
newly broken local Markdown links/anchors, retained Song checklists and A0/B6 rows, and per-task
programme authority/outcome requirements. Existing broken historical links are recorded separately,
not silently promoted or rewritten. Directory links and line anchors are distinguished from headings.

Controls operate on scratch document copies, never working sources: remove the authority pointer or
a required outcome, remove a retained acceptance requirement, and introduce the opposite boundary
(retire Lean or require Song to execute through Rust). Baseline and restoration must pass; each
intended mutation must fail for its named reason. These checks establish limited documentary
consistency, not semantic truth of arbitrary prose. Lead review remains required.

The central outcome ledger must cover ownership, consumer, dependencies, existing evidence, missing
specification/gate, and completion conditions for each outcome. It is not a dispatch list: missing
gates and profiles require bounded design tasks before software implementation.

Sol reviews storage/benchmark scope, F2 delivery and stale combinator/evaluator statements before
acceptance. No grade, premise, theorem or attribution is inferred from a title. Astra's frozen
boundary retains all local gates. Runtime/Lean rebuilds are inapplicable to this documentation-only
pass: no code, extraction, package, pin or theorem changes are allowed. A software change would
require a separate task and its local barrier.

## Acceptance record

- Architecture decision: frozen by `openai-astra`, session `ses_f66026973ffejJuS8pXwtdbGkM`.
- Preparation: `git diff --check` exits 0. Checker unit checks exit 0; lead reran the
  pre-edit `--task all` baseline: exit 1, 50 documentary findings. Live controls were not ready
  until the relevant documents were green; synthetic detector tests are not acceptance controls.
- Baseline: `/tmp/opencode/programme-docs/baseline.json`, sha256
  `399382c1d0ced58c4c024495a01cbcc15157064b7a65f7001a41537d92bb8150`; eight repositories,
  4,811 files, 26 owned documents, 100 unique pre-existing broken-link keys. Do not recapture.
- Sol claim wording approved in `ses_f65dc7229ffeTavtqB83fwFrVS`: current fixed-shard table
  witnesses are not universal preservation/subsumption; capped certificates are not uncapped
  `CertOp` completeness; `normalize`/`verdict` exist, with exactness only on decided outcomes.
  No current-table performance or Rust-Slang runtime correspondence is claimed.
- Central task W1-T1: lead read all three outputs, corrected unsupported absence/identity claims
  and separated scientific, application and learning acceptance. The final focused command exits 0:
  zero findings, two live baseline/red/restore controls. This opened W2 propagation on disjoint files.
- Propagation tasks pass their focused commands after review corrections. Each command is
  `python3 /tmp/opencode/programme-docs/check.py --task APEIRON-PROGRAMME-<task> --controls`,
  run from the Apeiron checkout:

| Task | Initial findings / exit | Accepted findings / exit | Live controls passed |
|---|---|---|---|
| W2-T1 | 4 / 1 | 0 / 0 | 1/1 |
| W2-T2 | 5 / 1 | 0 / 0 | 1/1 |
| W2-T3 | 3 / 1 | 0 / 0 | 2/2 |
| W2-T4 | 5 / 1 | 0 / 0 | 2/2 |
| W2-T5 | 6 / 1 | 0 / 0 | 1/1 |
| W2-T6 | 2 / 1 | 0 / 0 | 1/1 |
| W2-T7 | 5 / 1 | 0 / 0 | 1/1 |
| W2-T8 | 2 / 1 | 0 / 0 | 1/1 |

- Final integration: `python3 /tmp/opencode/programme-docs/check.py --task all --controls`
  exits 0, zero findings, 5/5 live baseline/red/restore controls. The mutations remove a programme
  pointer, a required outcome or a retained Song requirement, or introduce Lean deletion or Song
  runtime linkage to Rust. All fail for their named reason; restoration passes.
- Lead read the diffs and checked the affected source. Sol approves the corrected theorem/refinement
  wording in `ses_f65dc7229ffeTavtqB83fwFrVS`; Astra approves the documentation architecture in
  `ses_f66026973ffejJuS8pXwtdbGkM`. Neither verdict claims fresh software acceptance.
- Review corrections restore the capped-certificate existential, distinguish a bounded reduct from
  a normal form, scope the Engeler singleton limitation rather than applying it to weighted routes,
  and retain the checked table witnesses without claiming universal preservation. Architecture
  corrections retain the secure OS-language-proof outcome, independent learning and scientific
  acceptance, local engine ownership, and complete Penglai coverage obligations.
- `git diff --check` exits 0 in all eight checkouts. `git status --short` confirms only the 24
  intended tracked documents plus the new plan differ from the recorded dirty baseline. The other
  edited document, `song/plans/ROADMAP.md`, is already ignored and is checked directly, not included
  in a Git diff. Its 275 original checkboxes remain; the current Song roadmap retains all 49
  checkbox lines, 34 A0 rows and four B6 rows.
- Source/pin/generated-artifact preservation passes against the immutable baseline: all eight HEADs
  and every captured nonowned file remain unchanged, including the dirty Fold/Lyrium source and
  frozen RFC/publication artifacts. The baseline records tracked and nonignored untracked files;
  it is not an inventory of ignored build caches. No build lane was used.
- The baseline contained 100 unique pre-existing broken-link keys. No newly broken local Markdown
  link or anchor is introduced; this is not a claim that all historical links are repaired. Review
  also replaced the old Song design links to deleted Map source and benchmarks with the current
  `apeiron/src/table.rs` reference.
- Checker sha256: `92e8ff0fbc8aac095b55d49b3fb3db06047f6b262ec7cfd2663954954134b44a`.
  The baseline hash above is unchanged. The ignored historical Song plan's final sha256 is
  `909bae260644bf2b8c9e9db967fbeb8edfb5cdc7a7bb9a222a123030ca222ddf`.
- Decision: documentation consolidation accepted, 26 documents across eight repositories. This is
  checked documentary consistency and a reviewed architecture decision, not a new `proved`,
  software-`tested` or performance-`measured` claim. Rust/Lean/Kani barriers are inapplicable to this
  source-frozen documentation pass; implementation epics retain all their required gates.
- No commit, push or physical migration in this pass.

## Remaining implementation

The next bounded design task specifies the joint primitive/readiness profile: selected executing
primitives, their domains and representations, success/refusal/effects, resource bounds, exact
theorem links and rejecting gates. Existing Apeiron universal table preservation/admission work and
Song's local foundation continue under their own plans. Canonical source migration needs a separate
identity-preserving package task. None of these implementation obligations is closed by this pass,
and the next large AI implementation remains behind joint foundational readiness.
