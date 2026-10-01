# Post-compaction continuation — programme dispatch (mid-campaign)

Repo `/home/socol/Workspace/apeiron`, branch `main`. Read `AGENTS.md`, `../AGENTS.md`, `docs/direction.md`,
`docs/plan.md`, `docs/reviews/multi-repo-fleet-brief.md` (§9 phasing, §9.1–9.3 decisions). Source and
gates are the record; a document is corrected to them. The live fleet state is mirrored at
`/home/socol/oc-tmp/fleet-state.md`.

## 1 · Routes

- **Workers: `openrouter/stealth/space-bunny-alpha`** via `agent: general`. Free, window ends
  **2026-10-05**. `deepseek-flash` is the paid escalation, on a mandatory trigger only; anything beyond
  escalates manually to the owner.
- **Escalation: `/home/socol/oc-tmp/bin/sol.sh <prompt-file> [pro|<model>]`**, answers under
  `/home/socol/oc-tmp/escalation/`. Default `openai/gpt-6-luna-pro` ($0.10/Mtok in, $0.50/Mtok out);
  `pro` selects `openai/gpt-6.1-sol-pro` at twenty times the price. **The owner's OpenRouter balance was
  $7.31 of $77.45 on 2026-10-01.** Attach excerpts, never whole trees; `sol.sh` refuses a prompt over
  400 KB. The lead does **not** idle waiting for the owner — it escalates, records, acts, reports.
- **Lane caps.** ≤3 simultaneous heavy Lean checks (12 GiB each, `flock /tmp/opencode/apeiron-lane.lock`).
  Everything cargo/doc/read-only is uncapped; ~50 agents is the target width and the Lean cap is the
  only thing that has held it below that.

## 2 · Environment traps (learned the hard way this session)

- **`find` and `grep` are broken shims.** `/home/socol/.local/bin/find` (findutils 4.4.2) exits 0 and
  returns **nothing** on a directory holding 160 files; `grep` likewise returns nothing on files that
  contain the string. Use `rg`, `rg --files`, `ls -R`, globs — and a claim of absence must name the
  command behind it. This already cost one false "the corpus has 0 Lean files" finding.
- **Corpus directories are fine to read** (`../sanguine-apeiron/proof/Apeiron` holds 162 `.lean` files,
  `rg --files` counts them). Absence claims are admissible only from `rg`.
- **git's `%s` is the first *paragraph*.** A commit message with no blank line after the subject makes
  the whole body the subject. Song additionally caps the subject at 72 characters, requires an
  `<area>: ` prefix, allows one paragraph, and **bans phrases including "not yet"** — write messages to
  a file and use `git commit -F <file>`.
- **One owner per file.** Two Song workers held `docs/ROADMAP.md` at once and one arc rode into another's
  commit. Mechanically intersect file sets before dispatch.

## 3 · Programme invariants added this session (owner direction, recorded in the authorities)

- **The substrate is shared and Song's data operations run on it** (`docs/direction.md` §Ownership,
  `AGENTS.md` §4). A private probe, lookup or merge the canonical table already supplies is a rebasing
  obligation, not an alternative; where Song's architecture cannot host the table, the architecture
  changes. Fleet brief §9 gained Song wave **S11** for the convergence.
- **Corpus lineage.** `../sanguine` is the original corpus, `../sanguine-apeiron` its successor, **this
  repository is the canonical root** with both as predecessors. A predecessor is provenance at its
  revision, not a second authority.
- **Search the lineage and the sibling repos before commissioning a primitive** (`AGENTS.md` §4). Most
  likely in `../sanguine` or `../sanguine-apeiron`, sometimes solved in another sibling repo. A form
  found there is a **port with a named target**. Only absence everywhere authorizes making one, and the
  spec records the search and its command.

## 4 · Landed this session (lead re-ran every gate before commit)

| repo | commits |
|---|---|
| apeiron | `52cb47b` extraction root resolves `dequant_f32_row` · `405df80` RowDotAdmission + gate of record · `da5915c` loop-brief objective + E3 decomposition · `003bfad` evidence index (66 rows) · `360b5ff` escalation pricing · `028c8aa` gate manifest + adapter + conformance record (38 rows) · `8eed015` substrate invariant + corpus lineage · `3f419db` `row_dot_fixed` isolated baseline (measured: 0.58 ns/element at 4096, slowed-copy control moves 3.5x) · `9250932` gates refuse early on a missing artifact · `3074187` name the `find` shim · `b42018b` record the measured Song table blocker · `af06014` removal reachability gate (M1–M4) |
| intent | `e499153` D7 is an epoch transition · `df33ee9` Ring 6 gate, four refusing controls · `2eb6aed` Ring 2 descriptor isolation via nested `bwrap` |
| song | `fd7e5777` B6.1 comparison profile gate (green-capable, currently RED: no CBMC/rustc/solver artefact exists) · `a1308ff4` eight corpus renames resolved · `0f219f51` build route states its real R5 status · `f3614eba` false four-merge comment · `c4583f1e` dead second validation site removed |
| penglai | `4d1a5a5` carrier citations re-pointed at the corpus of record (14 pre-edit failures → 0) |
| webfold | `5882db84` merge of `origin/main` (27 conflicts, `cargo check --all-targets` green) · `bdc4134a` process-monitor units gate |

## 5 · Escalation verdicts already acted on (do not re-derive)

- `e5w1-answer.md` — **D7's rebase as written cannot rebase** (`max(S_fresh, M_t)` preserves the stale
  high-water mark); corrected to a receipted epoch transition. **D2's `history_hi` is an observation,
  not a bound**; corrected to a justified bound or `None` + refusal. The loop-brief objective was
  **unattributable** to `row_dot_fixed` (the GPU figures time a Vulkan `q8_tile` dispatch and CUDA MMQ).
- `e3-bind-answer.md` — the reported `bind_tc_ok` root cause was **wrong** (its continuation is
  arbitrary, so pattern binders are fine). Real defects: premature witness selection,
  equality-vs-existential, a reset proof state, unjustified `vis`/`div` exclusion, and
  `row_visits_closed_form`'s prose claiming total executed work where the theorem proves the initial
  rank.
- `song-table-merge-answer.md` — **decision (a)**: Song's `insertWith` must expose the canonical merge
  freedom (`apeiron/src/table.rs:134-143` admits any `Fn(V,V) -> V`; `song/src/table/table.slang:155`
  narrows to three codes). The three codes survive as named merges; the `table.slang:33`/`:36-38`
  assertions stay as regressions rewritten to `mergeSet`; the rejecting control is
  `keepFirst(new,old) == old` vs `MERGE_MAP(new,old) == new`. **Prerequisite: if Slang cannot represent
  first-class callables, the API change waits for that support.**
- A measured falsifier backs it: two intern records with identical bytes give one fold key; the table
  returns last-wins, `internFindWalk` returns first-wins, `agree=0`.

## 6 · Known blockers and their owners

- **Charon refuses `Returns inside of nested loops`** for `src/engine/attn.rs:887-951` and
  `attn/f32_core.rs:129-218,342-432`, which blocks both `proof/regen.sh --check` (E4-W3, E0514 already
  cleared) and `KvAppendResource`. One tool limitation, two obligations; a task holds Charon's own
  exclusion mechanism, with an order to stop and name the contract options rather than invent one.
- `KvAppendResource.lean` and `RegenRelocateAllResource.lean` were **never committed** (disposition (b),
  decided by commands); both gates now refuse in 14 ms with an attributable diagnostic.
- **Intent's D4 is implemented nowhere** (no `stratif|pareto` in the tree); the Ring 6 D4 arm is an
  absence pin that must redden when D4 lands. `intent_core::selection` has no production caller.
- **MAX-PAYLOAD-FORGE** is still open in Intent (a candidate reaching the host's own protocol pipe
  scores 15/15); Ring 2 as a nonce is open (`admit_frame` called with `tag_ok` wired `true`).
- **Webfold's `CLI-*` rows have no located evidence** (empty Source, `path: null`); one task is
  re-locating them and must report the unlocatable count rather than invent fixes.
- Song's `plans/ROADMAP.md` marked K12.3/K12.4 DONE on artefacts absent from `src/`.

## 7 · Next actions

1. Keep integrating returns: re-run each gate, commit with a message the repo's hooks accept, record the
   overrun when a run exceeds its budget (one already did: E5-W2 at ~13 min against a 420 s declaration,
   closed on the lead's fresh 0 s re-run).
2. Drain the heavy Lean queue as slots free: `matvec`, `LmHead`, the Charon exclusion, then Penglai's
   `carrier3` refinement lemma and Intent's E5.W4 rooting.
3. Widen with Webfold's 59 deduped groups (unblocked by the merge) and Song's A1 R3/R4, A4 consumers and
   S11 convergence.
4. Answer the callable-representation question (`/home/socol/oc-tmp/song-callable/answer.md`) before the
   table API change; then intern, `sort.slang` and `region.slang` rebase onto the canonical table.
5. Search `../sanguine` for the five Penglai claims that lost their carrier (16 `Product*.lean` files
   exist there) before any of them is re-derived.
