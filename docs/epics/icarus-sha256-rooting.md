# SHA-256 / SHA-256d rooting onto the trichotomy

Rolled up from the Icarus campaign (`lucas/icarus`, workspace staging) into this repository's
lifecycle: an epic with intent, specification and plan in one file (AGENTS §2–§3).

## Intent — SHA-256 / SHA-256d rooting

- **Owner:** lead (Icarus campaign); Sol for grades; corpus lead for transport.
- **Source revision:** `549d00ca62df50802db47f3e58ec4ee11c40c4b6`; working tree dirty, list in the
  commit body of the port tasks (11 pre-existing doc changes, untouched by this epic).
- **Status:** active — W1–W4 ported and bound; the closing full-library barrier is green; W5 blocked on the owner's regen window (see the Status section at the end).
- **Spec / Plan:** the sections below; staging evidence lives in the campaign workspace
  (`lucas/icarus`, `work/algebra-rooting/` and `work/proofs/`, 87 bound claim blocks at the time of
  writing, `python3 work/check_receipts.py --check` exit 0).

### Problem

SHA-256 and SHA-256d are the programme's named cryptographic targets, and
`docs/direction.md:242` (security / cryptography rebase) already contemplates a "SHA-256d reading"
and "message-schedule localisation". The campaign has proved, at declared grades, what each operation
of SHA-256 admits and refuses on the trichotomy faces — plus a formal cost model and its bridge into
`proof/resource/CostAlgebra.lean` — but those proofs live as workspace staging files verified by
direct `lake env lean` calls; they are not modules of this repository's build, not discoverable by
`lake build`, and not registered in the audit. The epic moves that body of work into the tree as a
first-class library, without changing any existing corpus statement.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| One face (e.g. elliptic) under one polynomial expression | Refuted in the staging evidence: rotation is not any one-`ringMul` read at any face (`no_read_computes_rotation`, `ringMul_family_additive` vs `rotWrap_not_additive`); the additive layer forces σ=1 for the round (`one_face_serves_the_round`, at 3 ≤ k). |
| A single word-ring encoding for everything | Refuted twice: no ring polynomial of any degree reads the shift (`shr_not_ringpoly`, every width, any face); the halves pairing serves exactly one amount (`span_rotr_iff`), and its action property fails with no equivariant multiplier and no multiplicative repair (AR-36, bound). |
| Transcribe/restate the canonical corpus structures inside the ported files | Rejected by the corpus's own rule: the staging files already **import** `Apeiron.Trichotomy.AnyRing` / `ChannelsAnyRing` / `Crypto.*` (57 of 60 staging files); the port re-homes them and proves the same statements from the same imports. |
| Keep the staging tree as the deliverable | It is design under AGENTS §2 ("an epic without a written intent, spec and plan is still design"): no build target, no audit registration, no tree ownership. |

### Completion condition

```sh
cd proof/corpus && lake build Icarus
```

green over the ported modules, with every ported declaration's statement byte-identical to its
staging source (modulo namespace/import adaptation), its cone inside
`{propext, Classical.choice, Quot.sound}`, zero `sorryAx`, and its task receipt bound as in the
staging convention. The epic closes when waves W1–W4 are green and W5's extraction obligation is
either discharged or recorded as a blocker with its exact regen requirement.

### Invariants

1. No existing corpus theorem statement changes; the port is additive (new directory, new library,
   one lakefile stanza, one docs index row).
2. No ported theorem's grade changes in either direction without a new named gate; refutations
   (`not_*`, `no_*`, `*_not_*`) port at their status as first-class outcomes.
3. The face dial σ appears only where the staging proofs put it: the product law of `ringMul` and
   the channel/readout structure; no new per-operation σ-dependence is introduced by porting.
4. No `sorry`/`admit`/axiom/`native_decide`/raised `maxRecDepth`/`maxHeartbeats 0`; lanes as AGENTS
   §6 (`cd proof/corpus` for Lean, bounded, `flock`).
5. The extraction (`ApeironWord.lean`) and `regen.sh` are untouched until an exclusive window;
   W5 changes only what imports or models them.
6. Claims without a bound receipt are ungraded, not false — ported files carry their staging claim
   ids in their receipts.

### Verification hypotheses

| hypothesis | grade | staging evidence (claim id) |
|---|---|---|
| Carry decomposition: both slots recompose the modular sum, every width, every σ | proved | `ar18-carry-general-p9-green`, `ar4-carry-*` |
| ANF degree `j+1` and term count `2^j + 1`, every width | proved | `ar29-*`, `ar62-term-count` |
| Shift exactness `{n : k ∣ n}`; no ring polynomial of any degree reads a shift | proved | `ar23-shr-exactness`, `ar43-shr-opens` |
| Rotation: halves read exact; action refuted; one amount per pairing; norm boundary movable by σ | proved | `ar17-*`, `ar36-rot-bridge-refutation`, `ar25-*`, `ar26-*` |
| One face (σ=1) serves the round; elliptic face has no operation-preserving map out | proved | `ar19-*` |
| Width-32 carrier: packing, wiring, carry, one round step `rfl`-linked to the canonical body | proved | `ar37-*`, `ar53-*`, `ar60-*` |
| SHA-256d canonical composition and unconditional byte transport | proved | `ar40-*`, `ar46-*` |
| Cost model: attempt/range figures, face-blindness, `CostAlgebra.Cost` bridge | proved (map graded `model`) | `ar15-*`, `ar20-*`, `ar27-*`, `ar41-*`, `ar58-*`, `ar59-*` |
| Spec-web edges (pad, schedule, absorb, squeeze) | proved | `ar32-*`, `ar42-*`, `ar44-*`, `ar49-*` |
| Extraction evaluator fidelity on published vectors | tested | `ar31-*` |

## Specification — rooting attribution

### Algebra and selected domains

- Carrier: `RingCell R = R × R` with `ringMul σ`, `ringNormForm σ` from
  `Apeiron.Trichotomy.AnyRing` / `ChannelsAnyRing`; Boolean carrier `BoolRing k = Fin k → ZMod 2`
  with `boolEquiv : BitVec k ≃ BoolRing k`.
- The four ported primitive families: first-axis `ringMul` (AND), word addition via the carry cell
  (XOR + residual), coordinate wiring (rotations and shifts), and readout (`ringUpper`, channels).
- Widths: the staging lattice includes 2, 4, 8, 32, 64; the executed targets are 32 (SHA-256 word)
  and 64 (`Cell<i64>` where reachable). Domains/partiality per file, unchanged on port.
- Cost: the staging measures (`RingCost` 6, `RoundCost` 11, `ShaCost` 23 fields), the
  source-bound compression cost, and the `shaToCost` bridge into `CostAlgebra.Cost` with its
  dropped fields named.

### Admission, controls and trust

- Admission: a port task touches ≤3 files, runs one build gate and one negative control, and is
  closed by the gate's exit status. Seeded controls edit throwaway copies only.
- Trust: the port does not re-verify the extraction; extraction claims stay at their staging grades
  until W5. `regen.sh --check` is run only in the exclusive window.
- Evidence: each task appends an arc block and, on green, a receipt with the build log's digest;
  the staging checker remains the campaign-side index.

## Plan — waves and bounded tasks

Waves are barriers: all gates green with controls before the next opens. A task owns ≤3 files; the
port tasks are `APEIRON-ICARUS-W<wave>-T<task>`, assigned to the lead's workers with staging files
read-only; each runs in an isolated worktree per AGENTS §3 and merges to `main` locally (no push
without the owner).

- **W1 — foundation core** (carry, degree, shift): port `AlgebraCarry`, `AlgebraCarryGeneral`,
  `AlgebraCarryComputable`, `AlgebraCarryTermCount`, `AlgebraShrExact`, `AlgebraShrExactOpen`.
  Gate: `cd proof/corpus && lake build Icarus`; control: a statement mutation in a throwaway copy
  must fail the build.
- **W2 — carrier and round**: port `AlgebraBoolCarrier{,-32,-32Round}`, `AlgebraRotorSha`,
  `AlgebraRotExact`, `AlgebraRotate32`, `AlgebraRotorShift`, `AlgebraRotorUnit`, `AlgebraSigmaHalves`,
  `AlgebraFaceComposition*`. Same gate; controls per file (already staged).
- **W3 — composition**: port `AlgebraSha256d`, `AlgebraSha256dCompose`,
  `AlgebraSha256dBlockArray`, `Sha256BE`, and the sponge spec-web files that import `Crypto.*`.
- **W4 — cost**: port `AlgebraCost`, `AlgebraCostRound`, `AlgebraCostPow`, `AlgebraCostComplete`,
  `AlgebraCostSource`, `AlgebraCostPowCompare`, `AlgebraCostKeccak`, `AlgebraCostBridge`, plus the
  bridge freshness gate (`work/cost_bridge_gate.py` → `tools/` on acceptance).
- **W5 — extraction obligation**: port the `WordSha256Denotes` prep; open the exclusive regen window
  for `word_sha256d` and the `paddedBlock` un-private (E1); or record the exact blocker.

### Bounded first task

`APEIRON-ICARUS-W1-T1`: the lakefile stanza (`lean_lib Icarus where globs := #[.submodules `Icarus]`)
plus the four carry/degree modules; gate `cd proof/corpus && lake build Icarus`; control: in a
disposable copy, mutate `bitPoly`'s recurrence so the term-count theorem must fail. Files: the new
directory's first members and `proof/corpus/lakefile.lean` (one stanza).

### What this plan is not

Not a claim that the staging tree is wrong; not a change to any existing corpus statement; not an
extraction refresh; not a security, performance or mining claim. The port is bookkeeping of proved
things into their proper tree.

## Status (2026-10-04)

`Icarus` = `proof/corpus/Icarus/` (42 `.lean` modules), built by `lean_lib Icarus` in
`proof/corpus/lakefile.lean`. All commits local, unpushed, pending the owner.

### Per wave

| wave | state | modules landed |
|---|---|---|
| W1 — foundation core | done (`7821a8a`) | `AlgebraCarry`, `AlgebraCarryComputable`, `AlgebraCarryGeneral`, `AlgebraCarryTermCount`, `AlgebraShrExact`, `AlgebraShrExactOpen` |
| W2 — carrier and round | done (`b04da83`, `d49b7ee`, `c7c97e6`, `b554776`; AR-36 `6d53961`) | `AlgebraBoolCarrier`, `AlgebraBoolCarrier32`, `AlgebraBoolCarrier32Round`, `AlgebraRotorSha`, `AlgebraRotExact`, `AlgebraRotate32`, `AlgebraRotorShift`, `AlgebraRotorUnit`, `AlgebraSigmaHalves`, `AlgebraFaceComposition`, `AlgebraFaceCompositionRound`, `AlgebraFaceCompositionSwitch`, `AlgebraRotBridge32` |
| W3 — composition | done (`b1a2376`, `ad5d3c3`) | `Composition/AlgebraSha256d`, `Composition/AlgebraSha256dBlockArray`, `Composition/AlgebraSha256dCompose`, `Composition/Sha256BE`, `Composition/SpecImplicationWeb`, `Composition/SpongeBlocks` |
| W4 — cost | done except the two modules named below (`c275aee`, `30cd3a8`, `dd2bd71`) | `Cost/AlgebraCostSource`, `Cost/AlgebraCostRound`, `Cost/AlgebraCostPow`, `Cost/AlgebraCostCompletePart1`, `Cost/AlgebraCostCompletePart2`, `Cost/AlgebraCostPowComparePart1`, `Cost/AlgebraCostPowComparePart2`, `Cost/AlgebraCostBridge`, `Cost/CostAlgebra` (vendored) |
| W5 — extraction obligation | **blocked** (owner regen window); the conditional half is ported | `Extraction/DenotesPrep` (`b2ec840`, rewired `d9b5164`), `Extraction/Store` (`9206dc5`), `Extraction/Schedule`, `Extraction/Compression` (`1230b89`) |
| beyond the W1–W5 plan | done | keystone `Keystone/RootedSha256` (`sha256_round_rooted`, `711df4b`); AR-33 `RingModelWeb` (`5b643d0`); width-32 reads `Sha256/CarrierSigma32` (`575989d`, `43b2eb6`), `Sha256/ReadTrichotomy32` (AR-12, `f37f8a2`) |
| plan item E1 | done | `Crypto/Sha2.lean` `paddedBlock` un-private (`1f39140`) |

### Barrier

`cd proof/corpus && lake build Icarus` green over the full library: log
`receipts/logs/review/sol7b-remark-barrier.log`, 3348 jobs, `LEAN_RC=0`, `SOURCE_SHA256=66e62397…`,
1942 `#print axioms` rows all inside `{propext, Classical.choice, Quot.sound}`, `sorryAx` 0,
0 error lines. Rows name 41 of the 42 modules; `Cost/CostAlgebra.lean` carries no `#print axioms`
row (vendored, `Lean`-import only). Not re-run for this status note; the log is the evidence.

### W5 block

`apeiron_word.word_sha256d` is absent from `proof/ApeironWord.lean` (0 occurrences, 24899 lines).
The prerequisite is the owner's exclusive `proof/regen.sh` window; `proof/extract/lib.rs:1109`
already defines the wrapper and `proof/regen.sh:242` already selects it, so nothing but the window
is missing. The handoff is `docs/evidence/icarus-regen-manifest.md`: §1 the required emitted symbol
and its line-span consequence (`word_sha256d` inserts **before** the trailing `chacha` items, so the
regen shifts every later span and re-binds the AR-14 / AR-31 / AR-40 / AR-73 pins that hold the
pre-window digest `855528a8…` **in the same commit** as the new `ApeironWord.lean`), §2 the
obligations that discharge on arrival (`AR-73-R1` and its `rfl` identification), §3 what the window
does not discharge (`WordSha256Denotes` stays an unasserted `def`; `DenotesPrep`'s headlines stay
conditional), §4 the freshness requirements (pre-run digest, derived-olean digest, marker + cone,
import route, pin citation, red/restore control; `APEIRON_LAKE=1` set so the corpus build runs).

### Remaining

| item | owner | state / next gate |
|---|---|---|
| Rebase + push of the port | — | **done** 2026-10-04: `origin/main` `dc1c114` (rebased onto `688f041`, post-rebase barrier 3348 jobs); both checkouts synced |
| `AlgebraCost.lean` / `AlgebraCostKeccak.lean` | — | **done** 2026-10-04: both ported, gated, controlled and bound (`sol15-*`, `sol16-*`); library at 44 modules |
| Regen window for `word_sha256d` | extraction owner | open — one exclusive `regen.sh` run + `regen.sh --check` with the `rfl` discharge check (manifest §2); then `word_sha256d_denotes` composes the corpus conditional with itself; the parent obligation stays `WordSha256Denotes` |
| Certificate arcs O11–O15 | certificate owner | open — wave-1 dispatch 2026-10-04 (schedule totality, pad tail + length write, loading-loop invariant, compression fold, store-loop value link) |
| Public write-up of the rooting results | lead | open — `work/algebra-rooting/final-report-draft.md` + `mining-analysis.md`; needs its own bound claim with a gate |
| Gate-tool hygiene | lead | **promoted** 2026-10-04: `tools/cost_bridge_gate.py`, guard closed, check + 47/47 suite green; residual: the fresh-elaboration (oleans-removed) audit exceeds this box budget |
