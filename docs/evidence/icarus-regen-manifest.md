# Icarus extraction handoff manifest — the exclusive regen window for `word_sha256d`

Task `SOL-6 w5-keystone-prep` of the SHA-256/SHA-256d rooting epic
(`docs/epics/icarus-sha256-rooting.md`, W5). This file is the handoff to the exclusive
`proof/regen.sh` window: what the window must produce, what that discharges, and the
freshness/identity discipline the window's own evidence must satisfy. The typed conditional
headline it hands off lives in `proof/corpus/Icarus/Extraction/DenotesPrep.lean`
(AR-73, ported from the staging receipt `work/proofs/AlgebraSha256dDenotes.md`).

Nothing in the window has been run. No `proof/ApeironWord.lean` line, no `paddedBlock`
declaration and no `regen.sh` line was edited by this task; the two artifacts this task added
are the module above and this manifest.

## Identity of the handoff

| role | path | sha256 at this manifest |
|---|---|---|
| ported conditional module | `proof/corpus/Icarus/Extraction/DenotesPrep.lean` | `b43c7babd952c0c16bb353eea8b8850ed554dab35b0ec0be165593d64f315526` (SOL-6, chain re-declared) |
| the same module, rewired onto W3 | `proof/corpus/Icarus/Extraction/DenotesPrep.lean` | `d5323d757c788e839d25b7befc016befd30f8dbb459e091bbcd64cee362bc605` (SOL-8, chain imported) |
| imported canonical chain (AR-40) | `proof/corpus/Icarus/Composition/AlgebraSha256dCompose.lean` | `d7eef60f59722007af296e942802e243b1fd1fff7a82ae3ac0aa888789fa0dd0` |
| gate log (SOL-8 rewire) | `receipts/logs/review/icarus-sol8-gate.log` | `5eb529568cc498d367c8b7b10ca50a65b25fcd7923f5ee67a62774d8945f106a` |
| control log (SOL-8 rewire) | `receipts/logs/review/icarus-sol8-control.log` | `b909eac8311a54f61aeede95bc52922eb8528f2b9987c20cb093dd54ea35673b` |
| extraction root, **pre-window** | `proof/ApeironWord.lean` | `855528a8a05b76328c99ad1ba1dd2e36458e5fe8aaccdc2c2596d7a58899d27c` |
| staging source of the port | `work/proofs/AlgebraSha256dDenotes.lean` | `852ea73ad6ae12cb6d8828368edb6430c57b5e27b8e7ee3392f3e36b3de67c65` |
| re-declared canonical chain (AR-40), staging | `work/proofs/AlgebraSha256dCompose.lean` | `d7eef60f59722007af296e942802e243b1fd1fff7a82ae3ac0aa888789fa0dd0` |
| gate log | `receipts/logs/review/icarus-w5prep-gate.log` | `208deedf0f5e63f50af0e4533a09337626f8917f14c490d2209850d2ee977c3f` |
| control log | `receipts/logs/review/icarus-w5prep-control.log` | `e174b19545ed37c79a92dc6223d5ab27defa05ea6b04e5b9ae28a8906ee71d5e` |

apeiron `HEAD` at this manifest: `b5547763b5aeaf1eb37b04ec94767559664da05d`.
Gate (SOL-6 port): `cd proof/corpus && bin/lane lake build +Icarus.Extraction.DenotesPrep` → rc 0, 19.8 s,
1709 jobs, 18 `#print axioms` rows all inside `{propext, Classical.choice, Quot.sound}`,
`sorryAx` 0, error lines 0. The one warning in the log is the transitive
`Aeneas/Std/Vec.lean:322` simp-argument lint already recorded in the W1 port receipt.
Gate (SOL-8 rewire, same command): rc 0, 20.0 s, 1710 jobs, 18 rows in the same cone, `sorryAx` 0.

## 1 · What the window must produce

`proof/ApeironWord.lean` currently contains no `word_sha256d` (`grep -c word_sha256d` = 0). The
wrapper is already prepared on the Rust side and already selected by the extraction:

| item | location | state |
|---|---|---|
| `pub fn word_sha256d(msg: &[u8]) -> [u8; 32] { sha256::sha256d(msg) }` | `proof/extract/lib.rs:1109` | present |
| `--start-from crate::word_sha256d` | `proof/regen.sh:243` | present |
| `apeiron_word.word_sha256` | `proof/ApeironWord.lean:24896`, body `sha256.sha256 msg` | present |
| `apeiron_word.word_sha256d` | — | **absent; this is the window's deliverable** |

Required symbol shape, exactly:

```lean
/-- [apeiron_word::word_sha256d]:
    Source: 'lib.rs', lines 1109:0-1111:1
    Visibility: public -/
def word_sha256d (msg : Slice Std.U8) : Result (Array Std.U8 32#usize) := do
  sha256.sha256d msg
```

with the emitted body definitionally the two-call composition the module already states:

```lean
def wordSha256dStrength (msg : Slice Std.U8) : Result (Array Std.U8 32#usize) := do
  let inner ← apeiron_word.word_sha256 msg
  apeiron_word.word_sha256 (Array.to_slice inner)
```

The window is a binding requirement, not a shape guess: the wrapper calls the real
`src/crypto/sha256.rs` `sha256d`, which applies the real `sha256::sha256` twice, and the extracted
body of `word_sha256` is the single call `sha256.sha256 msg`
(`proof/ApeironWord.lean:24896`, body line 24897). The inner digest must reach the outer application as
`Array.to_slice inner`. If the emitted conversion differs, restate `wordSha256dStrength` at the
emitted conversion and re-prove the two composition lines; the byte-view bridge
(`sliceBytes_to_slice_eq_stateToBytes_digView`) is unchanged either way.

### Line-span consequence to expect

`word_sha256d` sits at `lib.rs:1109`, **before** the trailing `chacha` items, which are declared
last only so that no earlier span moves. Inserting `word_sha256d` therefore shifts the recorded
source spans — and the line numbers — of every extracted item after it. `proof/ApeironWord.lean`
does not stay at `855528a8…`: it is a new revision, and every receipt that pins that digest
(AR-14, AR-31, AR-40, AR-73) must be re-bound against the new one in the same commit. The
pre-window digest is recorded above precisely so the re-binding is a diff and not a recollection.

## 2 · Obligations that discharge on the window's arrival

| obligation | statement | discharges when |
|---|---|---|
| **AR-73-R1** | the regen emits `apeiron_word.word_sha256d` at the shape of §1 | the emitted `ApeironWord.lean` is bound to a digest and `proof/regen.sh --check` is green on it |
| **AR-73-R1 discharge check** | `example : apeiron_word.word_sha256d = wordSha256dStrength := rfl` (or the pointwise form) in a small post-regen file, with `#print axioms` rows and `sorryAx` = 0 | that file builds green in `cd proof/corpus && lake build` |
| **instantiation** | `word_sha256d_strength_denotes h` and `word_sha256d_strength_eq_outer_compress_H0 h` then state about the named root, and `WordSha256dStrengthDenotes` reduces to `WordSha256DenotesAll`-based claims about `apeiron_word.word_sha256d` | the `rfl` identification holds |

Everything else in `DenotesPrep.lean` is unconditional at its stated grade and is **not**
waiting on the window: the byte-view bridge, AR-40's canonical chain
(`pad_stateToBytes_length`, `hash_one_block_eq_compress_H0`, `hash_outer_eq_compress_H0`,
`sha256d_eq_outer_compress_H0`), now imported from
`proof/corpus/Icarus/Composition/AlgebraSha256dCompose.lean`, and the composition's own
run/refinement pair. The window adds one identification; it does not re-open them.

## 3 · Obligations the window does **not** discharge

- **AR-73-R2 — the certificate hypothesis.** `apeiron_word.WordSha256Denotes`
  (`proof/corpus/Crypto/Sha256Certificate.lean:908`) remains a `def` of a `Prop`, deliberately
  unasserted. `WordSha256DenotesAll` is its universal closure and is **carried**, never
  asserted. Discharging it is the certificate owner's stage 1–4 obligation (schedule,
  compression, padding, digest-store lemmas): AR-31's published-vector evaluation of
  `word_sha256` is `tested`, not this kernel theorem. A window that emits `word_sha256d` leaves
  both headline theorems of `DenotesPrep.lean` conditional.
- **No freshness, extraction-fidelity, security, collision-resistance, preimage, constant-time,
  cost or mining claim** follows from the window, from this module or from this manifest. AR-31's
  vectors are evaluator evidence on the *previous* revision.
- **The W3 cross-reference — discharged.** `DenotesPrep.lean` used to re-declare AR-40's
  canonical-side chain verbatim because the W3 port of `AlgebraSha256dCompose` was not in the
  tree. W3 has landed at `proof/corpus/Icarus/Composition/AlgebraSha256dCompose.lean`, sha256
  `d7eef60f59722007af296e942802e243b1fd1fff7a82ae3ac0aa888789fa0dd0`, byte-identical to the
  staging source, and the re-declarations are deleted: the module imports it and opens the eleven
  canonical names by name, so AR-40 is the single authority for the chain and the module's grade
  is a kernel check over that import. Module digest after the rewire
  `d5323d757c788e839d25b7befc016befd30f8dbb459e091bbcd64cee362bc605`; gate
  `cd proof/corpus && bin/lane lake build +Icarus.Extraction.DenotesPrep` → rc 0, 20.0 s, 1710
  jobs, 18 `#print axioms` rows in `{propext, Classical.choice, Quot.sound}`, `sorryAx` 0
  (log `receipts/logs/review/icarus-sol8-gate.log`,
  `5eb529568cc498d367c8b7b10ca50a65b25fcd7923f5ee67a62774d8945f106a`; control
  `receipts/logs/review/icarus-sol8-control.log`,
  `b909eac8311a54f61aeede95bc52922eb8528f2b9987c20cb093dd54ea35673b`). The conditional
  headlines keep `WordSha256DenotesAll` carried and never asserted; no statement changed.

## 4 · Freshness and identity requirements for the window's evidence

The window's evidence is only worth its name if it binds source identities before, across and
after the run. The discipline is the one `work/cost_bridge_gate.py` enforces for AR-58; the same
six requirements, expressed for this window:

1. **Pre-run digest.** `proof/ApeironWord.lean` and `proof/corpus/Icarus/Extraction/DenotesPrep.lean`
   are hashed before anything runs, and the run is refused unless the digests equal the ones
   recorded in the receipt that claims them. A source that moved under the gate is `SOURCE_DRIFT`,
   whether it moved before the compile, between the compile and the run, or during the run.
2. **Derived-artifact digest.** Any scratch `.olean` the discharge check compiles is hashed after
   the compile that produced it and again after the run, and compared to the recorded digest.
   A scratch olean that is missing, truncated or built from something else is `STALE_OLEAN`, not
   evidence. Lean v4.31.0 oleans are byte-deterministic for a fixed toolchain and option set, so
   this comparison is a real check; it is not a proof that the bytes came from this run.
3. **Marker and cone.** The discharge run prints its marker exactly once; every
   `#print axioms` row is inside `{propext, Classical.choice, Quot.sound}`; `sorryAx` occurs 0
   times. Zero `#print axioms` rows is a refusal, not a pass: the cone was not audited.
4. **Import route.** `printenv LEAN_PATH` under the same lane shows which directory the module
   name resolved from, and any earlier entry holding the same module's `.olean` is `OLEAN_SHADOW`
   — the run would have audited a different artifact than the one digested.
5. **Pin citation.** Every digest the record pins appears in the receipt text that claims it. The
   gate's record and the receipt are separate documents; if they drift, neither is evidence for
   the other.
6. **Red control, then restore.** The window's own claim — "the emitted root is the two-call
   composition" — needs a mutation control on a throwaway copy that reddens attributably, and a
   restore that is green again, both with the tree unchanged.

Additionally, for this window specifically:

- the window is **exclusive**; no other Lean/extraction work shares `proof/.lake/`, `target/` or
  the lane while `regen.sh` runs, and no agent edits `proof/ApeironWord.lean`;
- `proof/regen.sh` skips the Lake build unless `APEIRON_LAKE=1` is set, so a window that reports
  green without it has checked the extraction only, not the corpus build;
- `proof/regen.sh --check` is run inside the window and its output is part of the evidence;
- the post-regen tree must be committed with the re-bindings of every receipt that pinned
  `855528a8…`, in the same commit as the new `ApeironWord.lean` (§1).

## 5 · Residuals

| residual | owner | next gate |
|---|---|---|
| `apeiron_word.word_sha256d` absent from the extraction root | extraction owner, exclusive window | `regen.sh --check` + the `rfl` discharge check of §2 |
| `WordSha256Denotes` unasserted (`WordSha256DenotesAll` carried) | certificate owner | stage 1–4 certificate lemmas in `Crypto/Sha256Certificate.lean` |
| AR-40 chain re-declared, not imported, pending W3 — **closed by SOL-8**: imported from `proof/corpus/Icarus/Composition/AlgebraSha256dCompose.lean`, module re-gated at `d5323d75…` (rc 0, 1710 jobs) | lead | `lake build Icarus` at the next barrier |
| no freshness or extraction-fidelity claim | lead | a separate bound claim with its own gate; not implied here |