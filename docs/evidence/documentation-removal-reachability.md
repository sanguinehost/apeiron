# Documentation-removal reachability gate: discharge receipt

Discharges the open discharge *A generic reachability gate for removals* of `AGENTS.md` §10
row *Consolidation lost files*, at grade `tested`: a rejecting gate with a seeded refusal, run
on this tree at the identities below. Owner acceptance of the row is not this receipt's claim;
the row stays in the register (§10 removes a row only on an owner-reviewed discharge).

## 1 · Mechanism

| item | value |
|---|---|
| gate | `python3 tools/rooting/doc_reach.py --check` |
| source | `tools/rooting/doc_reach.py`, sha256 `1fd7ed89a2416edb933edcf741e7264c54d060dfd0cfcf8f2d76d90028cb0d1b`, 1157 lines |
| registry | `docs/moved.tsv`, 105 `removed<TAB>successor` rows, one per documentation removal |
| rules | `S` surface absent; `D1` no successor; `D2` successor does not resolve under the root or is the removed path; `D3` removed path still on disk; `D4` a live document names a removed path on a line that does not also name the successor; `D5` registry row structure (exit 2) |
| exit | 0 clean; 1 a rule fired; 2 usage or registry structure |
| modes | `--check`, `--self-test`, `--controls`, `--inventory PATH`, `--list`, `--root PATH`, `--registry PATH` |

`tools/rooting/retire_check.py` (rules M1–M4) checks a removal **record**; this gate checks the
**tree** for the documentation removals the repository registers in `docs/moved.tsv`. The three
legs are the row's own requirements: a named replacement (D1/D2), a computed consumer inventory
printed per removal and by `--inventory` (D4), and a fail-closed reachability check (D4).
Migration-note exemption: a reference whose line also names the row's successor is recorded, not
refused; there is deliberately no successor-self exemption.

## 2 · Covered population, and where it is on the mandatory path

`python3 tools/rooting/doc_reach.py --check`, cwd `/home/socol/Workspace/apeiron`, dirty tree at
`apeiron_head=1fb651af9d28260517352d006bbf76031a6c76eb` (working tree carries unrelated in-flight
Rust edits; the gate reads Markdown and the registry only, so no Rust source is in its input):

```text
CHECK CLEAN: docs/moved.tsv records 105 removal(s); every replacement resolves, every removed path is absent and no document names one
  rows=105 documents=197 gated=102 non_markdown=3 consumers=0 migration_notes=11
```

| population | count | read as |
|---|---|---|
| registered removals | 105 | `docs/moved.tsv` rows |
| of which Markdown (`.md`) | 102 | `gated`, each refused by D4 while a live document names it |
| of which non-Markdown | 3 | `non_markdown`: `docs/plan/state-ledger.tsv` (replacement `docs/state-ledger.tsv`), `docs/evidence/perf-global-baseline.log` and `docs/evidence/perf-global-baseline.stderr.log` (replacement `docs/evidence/perf-ledger.md`); still D1/D2/D3-checked, and a document naming one is exempt from the consumer rule only when the line names the replacement |
| documentation-surface documents scanned | 197 | every `*.md` under the root outside `.git`, `target`, `__pycache__`, `*.lake`, `.tracks`, `.agents`, `.claude`, `.codex`, `.aws`, `.licenses`, `.held-control`, `.held-scratch` |
| live consumers of a removed path | 0 | `consumers`; `migration_notes=11` = 2 repaired rows in `docs/evidence/perf-ledger.md` §8 + 9 preimage references in this receipt, each on a line that names its replacement |

Mandatory path: `AGENTS.md` §1a carries the command in the enforcement table, and `AGENTS.md` §10
row *Consolidation lost files* names it in its present-safeguard cell. §10 requires a removal to
name its replacement, inventory its consumers and gate reachability; this gate is the third leg
for every documentation removal the repository registers. It is a Python gate: no `cargo`, no
`lake`, no network, no clock.

## 3 · Seeded refusal and restoration (attributable, fresh, within budget)

Two pre-existing `D4` violations failed the gate before this task (`consumers=2`):

```text
RULE D4 docs/evidence/perf-ledger.md:693: live document still names the removed path docs/evidence/perf-global-baseline.log at line(s) 693; its named successor is docs/evidence/perf-ledger.md
RULE D4 docs/evidence/perf-ledger.md:694: live document still names the removed path docs/evidence/perf-global-baseline.stderr.log at line(s) 694; its named successor is docs/evidence/perf-ledger.md
  rows=105 documents=195 gated=102 non_markdown=3 consumers=2 migration_notes=0
CHECK FAIL: 2 violation(s) (D4=2)      # exit 1
```

They were the raw-log table of the `perf-global-baseline` receipt: two digests presented against
files the consolidation had deleted, with the row for the receipt text
(`docs/evidence/perf-global-baseline.md`, replacement `docs/evidence/perf-ledger.md`) deleted in the
same change. Repaired as a §10 removal
rather than by deleting the evidence: each row keeps its digest and line count and names the
replacement (`docs/evidence/perf-ledger.md`), so `migration_notes=2` and `consumers=0`. The digests
`92badfed…` and `9f7a6ec3…` now appear nowhere else in the tree, so the rows are their only record.

**Control — a seeded reference is refused by name, and restoring it is clean.** Throwaway copy of
the documentation tree at `/home/socol/oc-tmp/reach-control-APEIRON-ENFORCER-REACHABILITY-2/docs`
(never the working tree); `TMPDIR=/home/socol/oc-tmp`; `cp -a` of `docs/` only.

| step | command | exit | output |
|---|---|---|---|
| baseline | `python3 tools/rooting/doc_reach.py --root $SCRATCH --check` | 0 | `CHECK CLEAN: … 105 removal(s) … rows=105 documents=151 gated=102 non_markdown=3 consumers=0 migration_notes=11`; pre-seed digest of the seeded file `65f5af04848323640d01aad0bd78c35dc799491e70c9ed1d4c734267928fd734` |
| seed | append ``Seed line for the control: see `docs/autonomous-run-log.md`.`` to `$SCRATCH/docs/evidence/perf-ledger.md` (removed path; replacement `docs/archive/autonomous-run-log.md`) | — | — |
| refusal | same command | 1 | `RULE D4 docs/evidence/perf-ledger.md:2570: live document still names the removed path docs/autonomous-run-log.md at line(s) 2570; its named successor is docs/archive/autonomous-run-log.md` · `consumers=1` · `CHECK FAIL: 1 violation(s) (D4=1)` |
| restore | truncate the seeded line, then `sha256sum` | — | `65f5af04848323640d01aad0bd78c35dc799491e70c9ed1d4c734267928fd734`, equal to the pre-seed digest of the same file |
| restore | same `--check` | 0 | `CHECK CLEAN: … rows=105 documents=151 gated=102 non_markdown=3 consumers=0 migration_notes=11` |

**Gate self-tests.** `python3 tools/rooting/doc_reach.py --self-test` exit 0: 18 rejection
fixtures fired their rule and 4 acceptance fixtures were accepted —

```text
SELF-TEST PASS: rejected: S migration table absent, S documentation root absent, D1 no replacement named, D2 unreachable replacement, D2 replacement is the removed path, D4 live consumer still names the removed document, D4 successor document still names its preimage, D4 self-successor row does not exempt its references, D3 registered removal still on disk, D5 row is not a tab-separated pair, D5 row carries three fields, D5 removed path is empty, D5 duplicate removed path, D5 removed path is absolute, D5 removed path escapes the root, D5 successor escapes the root, D5 registry records no removal, D4 consumer inventory
```

`python3 tools/rooting/doc_reach.py --controls` exit 0: baseline GREEN on the repaired live tree,
each seeded defect red by name, restore reproduces it, repair skipped —

```text
CONTROL baseline: GREEN (105 removal(s), 151 document(s), 0 consumer ref)
CONTROL seed-reference: RED :: RULE D4 docs/README.md:119: live document still names the removed path docs/autonomous-run-log.md at line(s) 119; its named successor is docs/archive/autonomous-run-log.md
CONTROL remove-referenced: RED :: RULE D4 docs/epics/ai-track-reanchor.md:3: live document still names the removed path docs/README.md at line(s) 3; its named successor is docs/plan.md
CONTROL restore: GREEN (byte-identical to baseline)
CONTROL repair: SKIPPED (the baseline carries no violation to repair)
CONTROLS PASS: baseline reported, every seeded defect red by name, restore reproduces it
```
The same command before the repair is the positive leg on the live tree: `CONTROL baseline: RED
(105 removal(s), 149 document(s), 2 consumer ref)` with both `docs/evidence/perf-ledger.md` lines
named, then `CONTROL repair: GREEN (105 removal(s), 0 consumer ref, 0 violation)`. One run refuses
the real tree, the other refuses a seeded fixture; neither is a self-test.

**Non-regression of the neighbouring gates** (cwd the repository root, same dirty tree):

| gate | before | after |
|---|---|---|
| `python3 tools/rooting/index_check.py --check` | exit 1, `E1=6` (six unindexed receipts, pre-existing; `docs/plan.md` records the remainder open), `rows=70 files=76` | exit 1, `E1=7`, `E5=0`, `rows=71 files=78` — the same six paths plus `docs/evidence/m-m3a-address-half.md`, a file another lane materialized on disk at 23:31:27 (tracked at HEAD `1fb651af`, absent from disk when this lane's baseline ran). This receipt is indexed (`id documentation-removal-reachability`), so it adds no `E1`. |
| `python3 tools/check-claims.py` | exit 0, 31 graded claims | exit 0, 31 graded claims |
| `python3 tools/rooting/read_gate.py --check` | exit 0, 6 stage documents | exit 0, 6 stage documents (the `AGENTS.md` digest moves, as a dispatch expects) |
| `python3 tools/rooting/check.py --check` | exit 1, `24 violation(s) (A=1, B=4, C=2, D=17)` | exit 1, `24 violation(s) (A=1, B=4, C=2, D=17)` |

Two rows of `docs/evidence/ledger.jsonl` are mandatory registrations of the artefacts this
receipt records, not added scope: the new `id documentation-removal-reachability` row, and the
refreshed `bytes`/`sha256` of the `perf-ledger` row after §3 repaired that file — E5 compares a
row against the bytes on disk, so an edited receipt left unregistered makes `E5` red. Both were
written with `json.dumps(row, sort_keys=True)`, the format all 70 pre-existing rows round-trip
through.

Identities: Python 3.14.7; no tool revision beyond the gate file is named because no other tool
runs in this gate.

## 4 · Exclusions and open residual

- **Unregistered removals are invisible.** The gate refuses what `docs/moved.tsv` records; a
  documentation deletion that was never registered has no row to refuse and no previous-tree
  baseline to diff. The population is not empty on this tree, found by
  `for p in docs/extraction-requirements.md docs/apeiron-intent-escalation.md; do test -e $p …; done`
  plus `rg -l --glob '*.md' -F "$p" .`: both paths are absent from the tree, both are named by
  live documents (`docs/extraction-requirements.md` by `docs/README.md` and
  `docs/archive/handoff-openai.md`; `docs/apeiron-intent-escalation.md` by `docs/plan.md`), and
  neither has a registry row. Closing this needs a git-baseline discovery leg or a row per
  unregistered removal; both are outside this gate.
- **The surface is Markdown only.** A tool, a receipt or a removal record naming a removed path is
  a record of the removal, not a consumer, so a live *program* reading a removed document is not
  seen here.
- **Resolution is existence, not correctness.** A successor that resolves to a file is not shown
  to carry the removed content; `D4` bounds references, not fidelity of the roll-up.
- **Registration is not review.** A `docs/moved.tsv` row is a claim; `--check` holds the row to
  the tree, not to the owner's judgement that the successor is right.
- Retained per §10: every removal still requires its consumer inventory and its task-specific
  reachability gate; this receipt discharges the gate's existence, not the per-removal obligation.

## 5 · Related

`AGENTS.md` §1a (enforcement table row and the MISSING paragraph), `AGENTS.md` §10 row
*Consolidation lost files*, [`tools/rooting/retire_check.py`](../../tools/rooting/retire_check.py)
(record side), [`tools/rooting/index_check.py`](../../tools/rooting/index_check.py) (evidence
index side), `docs/moved.tsv` (the registry).

---

Row: `APEIRON-ENFORCER-REACHABILITY-2` (`apeiron-enforcer-2-the-DoniUgsOm7EBIMgZWNTZWn`).
