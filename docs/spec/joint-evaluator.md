# Joint Evaluator and Certificate-Readiness Profile

This file is the frozen specification of the joint evaluator and certificate-readiness subprofile
for `APEIRON-PROGRAMME-W18-T2`. It names the canonical evaluator and decider, the Song entries that
are to be related to them, the eight obligations each profile row carries, the selected boundary,
the unmatched counterparts recorded as blockers, the known hazards, the required controls, the
integration order and what is explicitly not covered.

Authority: [`docs/direction.md`](../direction.md#primitive-correspondence) is the programme
authority; [`docs/kernel-boundary.md`](../kernel-boundary.md) owns the Rust representation and trust
boundary; [`proof/README.md`](../../proof/README.md) owns extraction and bridge scope. The canonical
corpus remains at `../sanguine-apeiron/proof/` at Git
`50c94192399add3b9f8e64bd14e36d8cba270705`, Mathlib `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`.

This document specifies obligations. It does not claim the correspondence exists. No row is closed:
every row below is `selected-open` or `blocker`. The machine-readable form is
[`tools/joint/evaluator_profile_rows.json`](../../tools/joint/evaluator_profile_rows.json); the
checker [`tools/joint/evaluator_profile_check.py`](../../tools/joint/evaluator_profile_check.py)
establishes specification consistency only, never correspondence.

## Required fields

Every joint evaluator profile row carries these eight fields. A row that drops one is incomplete and
the checker rejects it.

| Field | Obligation |
|---|---|
| `canonical` | The canonical object, the observation relation and the exact types. |
| `entry_points` | The concrete canonical and Song entry points, their source identity, and the Rust absence statement. |
| `representation_invariant` | The representation invariant established by the actual constructor or initializer, with the named terms it must state. |
| `input_grammar` | The accepted byte grammar, byte consumption and trailing-input policy. |
| `admission` | Separate canonical and Song admission predicates; joint success requires both, and unequal capacities need not give identical refusals. |
| `behavior` | Success, unsupported-input and resource-exhaustion behaviour, including the raw return encoding and the post-state. |
| `resources` | Separate canonical and Song resource accounting with distinct kinds; no generic `fuel`. |
| `refinement` | A source-connected refinement and a rejecting control. A copied model or a differential suite alone does not close a row. |

## Selected boundary

The selected boundary is the executing combinator evaluator and the certificate replayer. Both have
a canonical definition in the pinned corpus and a Song execution, and neither has an executing Rust
counterpart. They are therefore not paired Rust-Slang rows: each row records a Song side and a
canonical side, and the missing Rust side is an unmatched blocker (`BE1`, `BE2`).

| Row | Canonical object | Canonical entry | Song entry | Rust side | Status |
|---|---|---|---|---|---|
| E1 | `Apeiron.Combinator.Comb` evaluator | `Comb.headReduce`, `Comb.reduceDeep`, `Comb.normalize`, `Comb.hasRedex`, `Comb.verdict` | `prover0.slang::{headReduce,pass,normalize,hasRedex,structEq,verdict}` | NONE | selected-open |
| E2 | `Apeiron.Decider.checkCert` | `Decider.checkCert`, `Decider.certStep`, `Decider.CertOp` | `certcheck.slang::{main,ccBatch,ccRun}` | NONE | selected-open |

Canonical authority for E1 is `Apeiron/Combinator/Basic.lean` (`Comb`, `Comb.I`, `Comb.iota`,
`Comb.I_app`, `Comb.iota_app`), `Apeiron/Combinator/Verdict.lean` (`headReduce`, `reduceDeep`,
`normalize`, `hasRedex`, `Joinable`, `verdict`, `reduceDeep_sound`, `reduces_normalize`,
`normal_of_hasRedex_false`, `verdict_decided_exact`) and `Apeiron/Combinator/Confluence.lean`
(`confluence`, `normalForm_unique`). Canonical authority for E2 is `Apeiron/Decider/Cert.lean`
(`CRow`, `CertOp`, `certStep`, `checkCert`, `checkCert_sound`).

## Rows

### E1 · Combinator evaluator: normalize-and-compare on SKI/iota

**1 · Canonical object and observation relation.** `Apeiron.Combinator.Comb` with `Step`, `Reduces`,
`NormalForm`, `Joinable`, `headReduce`, `reduceDeep`, `hasRedex`, `normalize` and `verdict`.
Observation: `verdict fuel a b ≠ .fuel → (verdict fuel a b = .equal ↔ Joinable a b)`
(`Comb.verdict_decided_exact`); `Reduces c (normalize fuel c)` (`Comb.reduces_normalize`); a clean
`hasRedex` report is a `NormalForm` (`Comb.normal_of_hasRedex_false`); floors are unique under
confluence (`Comb.normalForm_unique`). Exact types: `Comb = S | K | var : ℕ → Comb | app : Comb →
Comb → Comb`; `Reduces = Relation.ReflTransGen Step`; `Verdict = equal | distinct | fuel`;
`fuel : ℕ`. The canonical evaluator does not carry `I` or `iota` as primitive constructors: `Comb.I`
is `app (app S K) K` and `Comb.iota` is a term built from `S` and `K` (`Comb.I_app`,
`Comb.iota_app`).

**2 · Entry points and source identity.** Canonical: `Comb.headReduce`, `Comb.reduceDeep`,
`Comb.normalize`, `Comb.hasRedex` and `Comb.verdict` in `Apeiron/Combinator/Verdict.lean`. Song:
`song/src/prover/ring/prover0.slang::headReduce` (one head redex on already-reduced children),
`pass` (one bottom-up memoized pass), `normalize` (pass composition to a fixed point or the budget),
`hasRedex` (the decidable floor detector), `structEq` (structural equality over the shared arena)
and `verdict` (parse, normalize, detect, compare, report). The inline laws `k_select`, `s_rule_open`,
`i_rule_open` and `iota_unfold` are Song laws, not canonical Lean.

**3 · Representation invariant.** Song represents a term as an acyclic initialized three-cell-node
DAG. Each node is `(tag, left, right)` at `CTR[C_NODES] + n*NODE_STRIDE + {0,1,2}`; `tag` is one of
`TAG_K`, `TAG_S`, `TAG_I`, `TAG_IOTA`, `TAG_VAR`, `TAG_APP`; `left` and `right` are node indices or
0. Every node is allocated before it is referenced and its children are fixed at allocation, so the
DAG is acyclic and initialized. Variable identity is index identity through `VCACHE[v]` for
`v ∈ 0..9`, which is what makes the `structEq` index-equal shortcut exact on variables. Memo
validity is per pass: `MEMO[n]` is valid exactly when `MSTAMP[n] = CTR[C_PASS]`. The walk stack
invariant is `0 ≤ CTR[C_SP] ≤ STACK_CAP` with the top at `STK[CTR[C_SP]-1]`; the parser stack
invariant is `0 ≤ CTR[C_PSP] ≤ PSTACK_CAP` with pending App nodes at `PSTK[0..C_PSP)`. There is no
rollback: a `SPENT` flag makes exhaustion a reported value, not a state restoration. No Song `Rep`
theorem relates this DAG to `Comb`.

**4 · Input grammar, byte consumption and trailing-input policy.** Prefix terms: `*` is application;
`k`, `s`, `i`, `u` are `TAG_K`, `TAG_S`, `TAG_I`, `TAG_IOTA`; digits `0..9` are `TAG_VAR` with
index `0..9`; whitespace is skipped. `readToken()` consumes one byte per call through `inByte()` and
increments `CTR[C_POS]` for each byte read; EOF is not counted. `verdict` reads exactly two prefix
terms and does not require EOF: bytes after the second term are left unconsumed and are not reported.
An unknown token byte, an incomplete prefix term or a parse-stack overflow yields `PARSE <pos>` with
exit 2.

**5 · Admission predicates.** Canonical: `Comb` is total; `Step`, `Reduces`, `NormalForm` and
`Joinable` are total; `normalize fuel c` returns `c` when `fuel = 0`. Song: `verdict` admits a byte
string that parses as two prefix terms; `parseStep` returns 0 on EOF before the term is complete, on
an unknown token byte or on parse-stack overflow, and `verdict` maps 0 to `PARSE`. The matched
domain excludes terms whose parse exceeds `PSTACK_CAP` or whose construction exceeds `NODE_CAP`.

**6 · Behaviour.** Success: `EQUAL` (exit 0) when both reached floors are redex-free and `structEq`
reports index-equal; `DISTINCT` (exit 1) when both floors are redex-free and `structEq` reports
unequal. Unsupported input: `PARSE <pos>` (exit 2). Resource exhaustion: `FUEL` (exit 3) when
`normalize` runs out of passes without a fixed point, or when `SPENT` is set by arena
(`NODE_CAP`), walk-stack (`STACK_CAP`) or parse-stack (`PSTACK_CAP`) exhaustion; `RING` (exit 4)
when `ringClaim` fails in `main()`. Raw return encoding: stdout `EQUAL|DISTINCT|FUEL|PARSE <pos>`
then the meter line, and the process exit code. Post-state: the arena, memo, stamps, counters and
stacks are mutated in place; no rollback.

**7 · Resource accounting.** Canonical: `normalize fuel c` applies `reduceDeep` at most `fuel`
times and `reduceDeep` visits each subterm once; these are definitional recursion counts, not
machine budgets. Song: `FUEL_PASSES = 2048` passes, `NODE_CAP = 1048576` nodes, `STACK_CAP = 65536`
walk-stack entries, `PSTACK_CAP = 4096` parser-stack entries, plus input bytes and the small
counter arithmetic. These are source budgets and are not Lean reduction-step counts.

**8 · Source-connected refinement and rejecting control.** Canonical: `Comb.reduceDeep_sound`,
`Comb.reduces_normalize`, `Comb.normal_of_hasRedex_false` and `Comb.verdict_decided_exact`. Song: no
checked Song-to-Lean refinement of `headReduce`, `pass`, `normalize`, `hasRedex`, `structEq` or
`verdict` exists. The joint refinement is OPEN. Rejecting control: `C6`, the `i`/`**skk` regression.

### E2 · Certificate replay: checkCert over the cone0 alphabet

**1 · Canonical object and observation relation.** `Apeiron.Decider.CRow`, `CertOp` and `checkCert`.
`checkCert raw [] log = true` exactly when the log ends on a `close x` naming a row whose bounds
cross, with every prior step naming an already-built row. Observation: `checkCert raw [] log = true
→ ¬ ∃ ρ, SatLines ρ raw` (`Decider.checkCert_sound`); every step maps holding rows to holding rows
(`Decider.certStep_holds`); an impossible row holds at no assignment
(`Decider.crImpossible_empty`). Exact types: `CRow n = { lo : Option ℤ, up : Option ℤ, coeff :
Fin n → ℤ }`; `CertOp = load k | combo i j a b | neg i | meet i j | tidy i | close i`;
`checkCert : List (Line n) → List (CRow n) → List CertOp → Bool`.

**2 · Entry points and source identity.** Canonical: `Decider.checkCert`, `Decider.certStep` and
`Decider.CertOp` in `Apeiron/Decider/Cert.lean`. Song:
`song/src/prover/cone/certcheck.slang::main` (batch entry), `ccBatch` (next-system loop) and `ccRun`
(certificate op loop). The five ops `r`, `c`, `n`, `m`, `t` and the closing `x` are `ccStepR`,
`ccStepC`, `ccStepN`, `ccStepM`, `ccStepT`, `ccStepX`. The in-tree `arith.slang` laws
(`combovalid_nonneg`, `farkas_bound`, `xsound`, `meetlo_sound`, `negeq_sound`) are Song laws.

**3 · Representation invariant.** A Song certificate row is a stride-106 record: bound-presence
flags `F_HASLO` and `F_HASUP`, then `F_LO` and `F_UP`, then `F_C + k*WF_W` per column. A wide value
is `WF_W = 4` cells: a tri-state sign `WF_S ∈ {WF_ZERO, WF_POS, WF_NEG}` then three unsigned 64-bit
magnitude limbs, least significant first, so the value is `±(m0 + m1·2^64 + m2·2^128)` under the
sign. This is a bound-presence flag plus a signed three-limb magnitude, not a wrapping ring value:
the limbs carry no law of their own and the encoding is not Apeiron's `ZMod (2^64)` word. `CROWS` is
`CERTCAP = 4096` rows and `RAWS` is `RAWCAP = 1024` rows; `CS.S_NROW` and `CS.S_NRAW` bound the
replay. A step index must be below `CS.S_NROW`.

**4 · Input grammar, byte consumption and trailing-input policy.** `certcheck` reads a system header
`n ∈ 0..MAXV = 24`, then `g`/`e` wire rows, then a `# cert` line, then the ops `r k`, `c i j a b`,
`n i`, `m i j`, `t i` and `x i`. `satcheck` separately reads `g`/`e` rows and a `w <v0> … <v(n-1)>`
witness with `MAXV = 12` and `ROWCAP = 256`. The reader consumes tokens and integers; `ccSeek` skips
everything before `# cert`. `certcheck` is batch-oriented: after a certificate closes, `ccBatch`
continues with the next system, so trailing input is the next batch, not an error. Within one
certificate, bytes after the closing `x` are the next system; a certificate that ends without `x` is
refused and an unknown op is refused by `ccRun`.

**5 · Admission predicates.** Canonical: `certStep` refuses a step naming a row index at or beyond
the built rows, a `combo` with a multiplier below one, and a `meet` over unequal coefficients;
`checkCert` refuses a log that does not end on `close` or has anything after it; `load k` refuses `k`
at or beyond `raw.length`. Song: `ccRow()` refuses a step naming `i ≥ CS.S_NROW` (a future row);
`ccNew()` refuses when `n ≥ CERTCAP`; `rs()` and `qs()` refuse an out-of-range row or field rather
than truncating; `ccStepC` requires both multipliers at least one; `ccSameCoefs` guards the meet;
`ccStepX` requires `ccGroundBad` (`lo > up`). The variable count is admitted in `0..MAXV = 24`.

**6 · Behaviour.** Success: exit 0 accept when the log ends on `x` and the named row is impossible
under `ccGroundBad`. Unsupported input: exit 2 parse for a malformed stream, an unknown op or a
variable count outside `0..MAXV`. Resource exhaustion: exit 1 reject for a future row, an unequal
meet, a multiplier below one or a certificate beyond `CERTCAP`; the reader refuses a malformed
stream. Raw return encoding: exit 0 accept, 1 reject, 2 parse; `ccBad(why)` prints the reason. No
boolean result crosses the process boundary. Post-state: the control counters and `CROWS`/`RAWS` are
mutated in place; no rollback.

**7 · Resource accounting.** Canonical: one `CRow` appended per non-closing step; one combination,
negation, meet or tidy per step; the Lean model computes in `ℤ` with no limb budget. Song:
`CERTCAP = 4096` certificate rows, `RAWCAP = 1024` system rows, `MAXV = 24` columns per row,
`satcheck` `MAXV = 12` and `ROWCAP = 256`, and `WF_W = 4` limbs per wide value. The limb arithmetic
(`wAddc`, `wMulWordA`, `wMulWordB`, `wCombineTAB`, `wMagSub3`) carries no law of its own and is
trusted like an ISA primitive. The source capacities are not Lean reduction-step counts.

**8 · Source-connected refinement and rejecting control.** Canonical: `Decider.checkCert_sound` is
the headline, with `Decider.certStep_holds` and `Decider.crImpossible_empty`. Song: no checked
Song-to-Lean refinement of `checkCert` exists, and the wide limb arithmetic carries no law of its
own. The joint refinement is OPEN. `Apeiron/Decider/Stream.lean` and the executing stream refinement
are MISSING. Rejecting controls: `C1` (corrupt a carry or sign), `C2` (future row reference), `C3`
(trailing records), `C4` (relabel SAT/UNSAT).

## Unmatched counterparts

These counterparts have no matched primitive on one side. They are recorded as blockers, not
assumed, and the checker requires every one of them to be present.

| ID | Counterpart | Reason | Status |
|---|---|---|---|
| BE1 | Rust side NONE for the canonical `Apeiron.Combinator` evaluator; Song `prover0.slang` exists. | There is no executing Rust evaluator, so E1 is not a paired Rust-Slang row. | blocker |
| BE2 | Rust side NONE for `Apeiron.Decider.checkCert`; Song `certcheck.slang` exists. | There is no Rust decider or replayer, so E2 is not a paired Rust-Slang row. | blocker |
| BE3 | Song `satcheck.slang::{main,batchLoop}` at `MAXV = 12`, `ROWCAP = 256`; canonical side NONE. | No canonical SAT-witness checker for this alphabet exists; `capped_certificate_complete` is a capped-box existence theorem, not a completeness theorem for this alphabet. | blocker |
| BE4 | `Apeiron/Decider/Stream.lean` and the executing stream refinement; both MISSING. | Certificate readiness requires a stream representation and a checked executing stream refinement; neither exists. | blocker |
| BE5 | Canonical `Comb.I = app (app S K) K`; Song `i` (`TAG_I`) and `**skk` (`app (app S K) K`) are distinct tags and shapes. | The pair `i`/`**skk` is a source-derived regression fixture candidate against the canonical `Joinable` reading of `DISTINCT`; it is not resolved by dropping `i`/`u`, changing canonical equality, or conflating the primitive with the normal form. | blocker |

## Hazards

| ID | Hazard | Consequence |
|---|---|---|
| HE1 | Canonical `I` is `S K K`, while Song treats primitive `i` (`TAG_I`) and the normal form `**skk` (`app (app S K) K`) as distinct tags and shapes. | Song `structEq` reports `DISTINCT` for the pair while canonical `Joinable` holds; the `C6` regression fixture must pin both readings. |
| HE2 | Song `FUEL` conflates Lean's `.fuel` (a reached floor that still carries a redex) with arena, walk-stack, parser-stack and ring exhaustion. | A `FUEL` verdict does not identify which budget ran out, there is no rollback, and no precise resource-classification theorem exists. |
| HE3 | The source budgets `2048` passes, `1048576` nodes, `65536` walk-stack entries and `4096` parser-stack entries are machine budgets. | They are not Lean reduction-step counts and must not be converted to `fuel`. |
| HE4 | `certcheck`'s wide row values are a tri-state sign plus three 64-bit magnitude limbs, and the limb arithmetic carries no law of its own. | The encoding is not `ZMod (2^64)` and is trusted like an ISA primitive; `C1` is a required control, not a proved absence. |
| HE5 | `capped_certificate_complete` is a capped-box existence theorem, and `Apeiron/Decider/Stream.lean` is MISSING. | It is not a completeness theorem for the `certcheck` alphabet and does not establish certificate readiness. |

## Required controls

These controls are specification-level obligations. They are not executed by
`evaluator_profile_check.py`, which checks consistency only; each names a source-derived fixture or
mutation that the executing refinement must reject.

| ID | Target | Control |
|---|---|---|
| C1 | E2 | Corrupt a carry or a sign in the wide row encoding (flip `WF_POS` to `WF_NEG`, or drop a carry in `wAddc`) and require the replayed verdict to change. |
| C2 | E2 | Permit a certificate step to name a row index at or beyond the current row count; `ccRow()` must refuse it and the replay must reject. |
| C3 | E2 | Accept trailing records after the closing `x`; `checkCert` requires the rest to be empty, so the replay must reject. |
| C4 | E2 | Relabel a SAT witness as UNSAT or an UNSAT certificate as SAT; `satcheck` and `certcheck` must each refuse the relabelled claim. |
| C5 | E2 | Omit the `w` witness record while claiming SAT; `satcheck` must refuse a SAT claim that carries no replayed witness. |
| C6 | E1 | Pin the Song `DISTINCT` of primitive `i` (`TAG_I`) against the normal form `**skk` (`app (app S K) K`) alongside the canonical `Joinable i **skk` reading; the fixture records the divergence and does not drop `i`/`u`, change canonical equality, or conflate the primitive with the normal form. |

## Integration order

1. E1 combinator evaluator: fix the DAG representation, variable identity, memo validity and stack
   invariants, and state the input grammar and trailing-input policy.
2. E1 refinement: define a Song-to-Lean refinement of `headReduce`, `pass`, `normalize`, `hasRedex`,
   `structEq` and `verdict` on the matched domain, and run the `C6` `i`/`**skk` regression.
3. E2 certificate replay: fix the bound-presence and signed three-limb representation, the row
   capacities and the refusal paths, and state the reader grammar.
4. E2 refinement: define a checked Song-to-Lean refinement of `checkCert`, add
   `Apeiron/Decider/Stream.lean` and the executing stream refinement, and run controls `C1`–`C5`.
5. Only after E1 and E2 are refined: a bounded vertical lift/reduction demonstration.

## Not covered

- rustc/LLVM code generation and the generated machine code.
- The kernel ABI, stack and OS.
- GPU / SPIR-V.
- Runtime linkage between Apeiron and Song; the required result is a checked correspondence, not
  linkage.
- Song local registers, ontology and commands.
- Semantic correspondence itself. `tools/joint/evaluator_profile_check.py` establishes
  specification consistency only; it does not report correspondence.
- No executing Rust evaluator and no Rust decider or replayer (`BE1`, `BE2`).
- A canonical SAT-witness checker for the `satcheck` alphabet (`BE3`).
- `Apeiron/Decider/Stream.lean` and the executing stream refinement (`BE4`).
- Any resolution of the `i`/`**skk` compatibility blocker (`BE5`, `HE1`).

## Checker

`python3 tools/joint/evaluator_profile_check.py --check` validates
`tools/joint/evaluator_profile_rows.json` against this document and the pinned sources: every row
carries all eight fields, every named canonical theorem and canonical entry point resolves as a
declaration in the pinned corpus, every named Song entry exists as a declaration, every required
unmatched blocker, hazard and control is present, the representation statement names its declared
terms, and the resource accounting names the distinct kinds with no generic `fuel`. It exits nonzero
on a missing field, row, blocker, hazard, control, theorem or source entry. `--controls` and
`--self-test` mutate a deep copy of the loaded profile in memory (remove a required field, remove an
unmatched blocker, remove a row, rename a theorem to a nonexistent name, point a source entry at a
missing file, rename a resource kind to `fuel`, remove a hazard, drop a fields entry, remove a
required control, drop a representation term, remove the trailing-input policy) and require each
mutation to be rejected; the working tree is never touched. The checker establishes specification
consistency only, never correspondence.
