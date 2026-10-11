# Bounded authorization decision core

Merged epic record: `intent/policy-engine.md`, `spec/policy-engine.md`, `plan/policy-engine.md` (one file per epic; see `docs/README.md`). Study: [`docs/study/policy-engines.md`](../study/policy-engines.md).

## Intent — Bounded authorization decision core

- **Epic:** `APEIRON-POLICY`
- **Owner:** Apeiron lead; Song correspondence with the Song lead; first consumer with the intent lead
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty (unrelated
  storage/AI/read lanes; the MPL-2.0 relicence in `LICENSE`, the four `Cargo.toml` manifests and
  `docs/hash-comparison.md`; this documentation pass)
- **Status:** draft. D1–D7 resolved (derived from the algebra, owner direction); W0 spikes closed;
  S6 (relationships) and S7 (compiler) open; no repository source is changed by this epic yet
- **Study:** [policy engines](../study/policy-engines.md)
- **Spec:** [specification](#specification--bounded-authorization-decision-core)
- **Plan:** [plan](#plan--bounded-authorization-decision-core)

### Problem

Three consumers need the same authorization decision and none has a checked one:

| Consumer | Present decision | Gap |
|---|---|---|
| `../intent` dispatch seam | `approvals::decide(mode, name) → Allow \| Deny(String)`, a 3×2 table (`crates/intent-agent/src/approvals.rs:35`). "Guarded + mutating" is `Deny` because no approver is connected, which makes it an `ask` that falls back to deny. `policy::decide` is registered as owed at W16 (`docs/PROOF.md:58`; `docs/ROADMAP.md` §4 and W16). | No rule language, no named cause beyond a free string, no refinement. |
| Song capability admission | `ringGrantSlice` attenuates a grant to a sub-range and refuses widening; unauthorised resolution returns 0 (`../song/src/kernel/mem/ring.slang:188-191`; `src/kernel/tests/grant.slang`). Song's authority boundary: "explicit grants; attenuation and transfer through the core interface" (`../song/docs/ARCHITECTURE.md` §1). | No canonical attenuation law shared with Apeiron; A7.5/A7.6 capability admission is blocked. |
| telos / agent gate (shape evidence only) | 110 Cedar rules, 337 `like` string patterns; the deny reason is the constant `"cedar policy denies"` (study). | The firing rule is lost; string matching sits inside the decision. |

The canonical corpus already has the value-level half: `Apeiron.Capability.guardedRead` with
`guardedRead_ungranted`/`guardedRead_granted`, and `forge_reaches_only_scope`
(`../sanguine/proof/Apeiron/Capability/{MaskedTable,Basic}.lean` at `880aa57e`). It has
no rule-set decision, no verdict with a cause and no attenuation law.
`Capability.Extent.ungranted_reads_granted_extent` also proves that a masked lookup is containment,
not authentication, so a compiled decision index must be keyed by the full key.

### Where σ enters

**Evaluation: it does not.** `decide` on a concrete fact record is a fold over the rights lattice
𝔽₂⁶⁴ (x² = x). No law about it needs `ringMul` or `ringNormForm`. Over a base of characteristic 2,
x² − 1 = (x + 1)², so σ = 1 and σ = 0 give the same ring; the rights have no face to select
(`anchor`: elementary algebra, not yet a corpus theorem).

**Analysis: through the canonical decider.** The guards `eq`/`le`/`ge` on integer facts are linear
integer rows (`Apeiron.Fragment.Row`). A rule guard is a box, a product of intervals.
`Apeiron.Decider.face_is_forced` (`Decider/Face.lean:274`) proves that an interval representation
(two ordered reads per point) exists exactly on the hyperbolic face (σ > 0);
`disc_one_of_integer_basis` forces σ = 1 for whole numbers. Meet is `meet_intervalOf`; an empty
guard is `crossing_empty`. Quantified policy questions (dead permits, "never allows", equivalence,
resolution of unknown facts) are integer-row satisfiability, decided by `completeSat`
(`completeSat_eq_true_iff`/`_false_iff`, `Decider/CompleteDecision.lean:102,129`), with UNSAT
proved (`Decider/Model.lean:198` `decide_unsat_sound`) and certificates replayed by
`checkCert_sound` (`Decider/Cert.lean:809`). Song's `ncone0.slang` and `certcheck.slang` execute the
same decider. Cedar answers the same questions by a Lean-verified encoding into SMT and an external
CVC5 process whose verdict is trusted (`cedar-lean/Cedar/SymCC/Solver.lean:27`).

| Layer | Algebra | Cost | Grade today |
|---|---|---|---|
| Evaluation `decide` | rights lattice + Kleene logic | O(R·(1+A)) atom evaluations | proved at spike scope (S3a) |
| Analysis (`Apeiron.Policy.Analysis`) | σ = 1 interval cell; integer rows | DNF size × `completeSat` (unbounded residue search, stated in `CompleteDecision.lean`) | proved at spike scope (S5: both soundness directions, 1-bit) |

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Adopt cedar-spec as the canonical definition | Lean `v4.34.1` + batteries against the bridge's `4.31.0` + Mathlib pin; strings, extensions and entity stores are outside the fragment; errors are skipped, so an erroring `forbid` permits (`cedar-policy-core/src/authorizer.rs:236-258`). See the study. |
| Link `cedar-policy` (Rust) as the evaluator | Its link to the Lean spec is differential testing, not proof; the same error-skip semantics applies. It is kept as a `tested` oracle. |
| Zanzibar/Polar/Rego-style recursive evaluation | Termination comes from configured depth or timeout (OpenFGA 25, SpiceDB 50, Keto 5, Polar stack 10,000 / 30 s), not from the language. |
| Casbin PERM | Interpreted matcher strings; allow-by-default and rule-order-dependent effects are selectable. |
| Keep per-consumer decisions | Three unrelated decisions cannot share one refinement or one Song correspondence. |
| Carry the compiled index in `Table` now | `Table::get`'s universal read link is open ([read-proof-next](../epics/read-proof.md)); a Table-carried decision cannot be refined end-to-end yet. Deferred to v2. |

### Completion condition

The extracted `policy::decide` refines the canonical `Apeiron.Policy.decide` on its whole
(unrestricted) input domain; the canonical laws, the four verdict witnesses and the Song
attenuation correspondence are kernel-checked with their controls red under mutation.

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
proof/regen.sh --check
cd proof/corpus && lake build Policy
python3 tools/slang/zero_axioms.py --stage policy-attenuation --controls
cd tools/policy-oracle && cargo test --locked
```

Each Lean command runs bounded (AGENTS §6), with a 600 s ceiling per invocation.

### Invariants

1. **Totality and determinism.** `decide` is a total function of `(rules, atoms, facts, known,
   action, want)`; the Rust body never panics on any input (every index is range-checked and
   out-of-range reads are unknown facts).
2. **Default deny.** No relevant permit whose guard evaluates true ⇒ `deny noPermit`.
3. **Forbid wins.** A relevant forbid whose guard is true **or unknown** removes its rights from the
   grant.
4. **Order independence.** Permuting or duplicating rules leaves the verdict unchanged; named rule
   ids are the minimum over the contributing set.
5. **Monotonic restriction.** The granted rights are a subset of the requested rights; adding a
   forbid never widens the grant; attenuation never widens its parent.
6. **Named cause.** Every deny carries `forbid id` for an applicable forbid in the set, or
   `noPermit`. There is no free-text reason in the decision.
7. **Unknown ≠ allow.** A permit with an unknown guard grants nothing; filling unknown facts
   never removes rights already granted (`granted` is monotone under completion).
8. **Non-vacuity.** Checked witnesses reach `allow`, `attenuate`, `ask` and `deny` at the joint
   domain, from the real extracted body.
9. **Mechanism, not policy.** The core evaluates rules; rule content belongs to applications
   (`docs/direction.md:31`; Song ARCHITECTURE §1 seL4 split).

### Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| Laws 1–7 over the canonical definition | proved | `Apeiron.Policy.{decide_default_deny, decide_forbid_wins, decide_perm, decide_granted_le_want, decide_add_forbid_le, decide_deny_named, decide_granted_mono_completion}`, `lake build Policy` |
| Extracted Rust `decide` refines canonical `decide`, total | proved | `Policy.Refinement.word_decide_denote`, `proof/regen.sh --check` + `lake build Policy` |
| Allow/attenuate/ask/deny reachable from the extracted body | proved | `Policy.Refinement.word_decide_{allow,attenuate,ask,deny}_witness` |
| Song `ringGrantSlice` attenuation corresponds to canonical `sliceExtent` under the empty cone | proved (empty cone) | `Slang.Policy.Attenuation.*`, `zero_axioms.py --stage policy-attenuation` |
| Agreement with `cedar-policy` on the 1-bit, error-free fragment | tested | `tools/policy-oracle` differential run |
| Surface compiler emits images that decide as the Cedar-subset source | tested | `tools/policy-oracle` |
| Per-decision cost O(R·(1+A)) atom evaluations | model/anchor, then measured if a performance claim is made | comment at `decide` (AGENTS §8) |

### Non-goals

- Dynamic attribute fetching, entity stores and I/O inside `decide`.
- Strings, regex, `like`, floats, time, IP/decimal extensions in the proved core. Their
  classification into bounded facts is a tested encoding boundary owned by the consumer.
- Unbounded or depth-configured recursion. Reachability is in-core only through a certified
  budget (`BoundedRoutes.TraceBudget`); without a certificate the verdict is `ask`.
- Transitive or cross-principal delegation chains, side effects, obligations/advice. Bounded
  delegation is in scope: a sub-agent's grant is the meet of its parent's grant and its own request
  (`att(att(g, a), b) = att(g, a ∧ b)`), required by `docs/epics/frontier-consumer.md`
  ("Delegation attenuates").
- Policy administration, distribution and consistency tokens (OPAL/zookie class).
- Moving the canonical corpus. The canonical definition is staged in `proof/corpus/Policy/Core.lean`
  (decision 5) and migrates with the corpus-release pin; `../sanguine` is unchanged.
- Displacing [read-proof-next](../epics/read-proof.md) (G1/REG-1 OPEN), the AI-stack lanes or
  Penglai.

### Trust boundary

Not covered: the surface compiler and parser (tested); the consumer's fact encoding, including
string/path classification (tested); `cedar-policy` (oracle only); rustc/LLVM codegen; the Song
`.slang` source binding beyond the modelled attenuation kernel; the intent seam's single-dispatch
property (intent `TRUST.md` #1, structurally checked there). The Aeneas bridge keeps the
`{propext, Classical.choice, Quot.sound}` cone; the Slang face keeps the empty cone.

### Decisions, derived from the algebra (owner direction)

Owner direction: each decision follows from the canonical object and its principles (defence in
depth, auditability, reversibility), not from preference. Sources are canonical-corpus theorems at
`880aa57e` unless marked as old Sanguine (`../sanguine-ref` `e830d3fb`, pre-trichotomy, rebase open in
`docs/epics/security-algebra-rebase.md`). A derivation step that is not yet a theorem is labelled
`anchor`.

**Principles, as the corpus states them.**

| Principle | Source | Consequence here |
|---|---|---|
| Schedule-freedom needs a merge whose left translations commute | `Table.OrderBlind.build_orderBlind`, `orderBlind_iff_leftTranslations_commute`; `Table.build_gauge_invariance` | the rule store is built with an order-blind merge, so order independence is an instance, not a new law |
| Commutative merges (presence `∨`, count `+`) are schedule-free; replace and append are order-bearing | `Table/Adjacency.lean`: `append_not_gauge`, `build_replace_swap_ne`, `build_append_perm_pointwise` | decisions read the presence face; the audit log is the append face, where order is meaning |
| Reversibility: state is a build over a key stream; any prefix rebuilds | `Table.build_eq_of_key_streams_eq`; old Sanguine `round_trip_id` (Wire fusion) | policy state = `build(log)`; rollback = rebuild a prefix |
| A decision that may not land within budget forces a third verdict | `Combinator.Verdict.verdict_decided_exact` with `Combinator/Omega.lean`; `Decider/Model.lean` (UNSAT/SAT/MAYBE) | `ask` is the honest undecided value, never folded into allow |
| Containment is not authentication | `Capability.Extent.ungranted_reads_granted_extent`; `Capability.forge_reaches_only_scope` | full-key lookup only; defence in depth = independent layers, each with its own theorem |

**D1 · Verdict.** `allow g | attenuate g | ask g p | deny cause`, rights as a W-bit mask.
- Default deny: the empty store is the unit of the order-blind merge, and its decision is the least
  verdict (`decide_default_deny`, S3a).
- Forbid wins, derived: an order-blind store only accumulates (a join), so a revocation cannot delete
  a rule without an order-bearing merge (replace), which would make decisions depend on arrival order
  (`build_replace_swap_ne`). A revocation is therefore an added rule that restricts, and its effect
  must hold at every position. That is forbid-wins (`decide_forbid_wins`, `decide_add_forbid_le`).
  The implication "order-blind + revocable ⇒ forbid-wins" is `anchor` until stated as a theorem
  (W1 target `revocation_is_forbid`).
- `ask`: the forced third value. It arises from unknown facts (Kleene `u`), from analysis `MAYBE`,
  and from an uncertified relationship budget (D2). D7 resolves some `ask`s exactly.
- Attenuation: the meet on the rights order and on extents (`sliceExtent`); bounded delegation is
  `att(att(g, a), b) = att(g, a ∧ b)`.
- The two consumer vocabularies remain specialisations (`oneBit_no_attenuate`, `noAsk_no_ask`).

**D2 · Relationships: in-core, on the Table's graph face (reverses the earlier bitset recommendation).**
- Relationship tuples are the Table's adjacency face: `insertWith (++)` records an edge
  (`Table/Adjacency.lean`, `get_build_append`). Decisions read edge presence, which is order-blind
  (`build_append_perm_pointwise`: which edges a key holds is schedule-free).
- The Web (old Sanguine `Substrate/Algebra/Web/Web.lean`) is exactly two Table maps, node states and
  adjacency; facts are node states and relationships are adjacency. `RelationalWeb.lean` reads
  select/join/project onto these operations (`anchor` there; rebase open).
- Reachability ("member of G through groups") is `Combinator.RelationalPaths.RelRuns` over a
  `StepRel`, with path weights composed by `ringMul σ`.
- Termination comes from a certificate, not a configured depth: `Combinator.BoundedRoutes`
  (`TraceBudget`, `traces_iff_traceRouteOfBudget`, `traceRouteOfBudget_sound`; audited
  `#audit_constructive`). With a certified budget, `exhausted` means no path exists; without one,
  the verdict is `ask`, never a truncation read as deny or allow (the OpenFGA/SpiceDB/Keto failure mode
  in the study).
- Cross-boundary requests (federation peers, remote agents, Song's "remote credentials cannot write a
  grant row") meet the Wire (old Sanguine `Web/Wire.lean`: `boundary_gateway`, fuse / detect / relay /
  zone; `foreign_contact_contained`). The legs read onto allow / deny cause / attenuate / contain.
  That reading is `anchor` (riding grade, pre-trichotomy); it enters the design only with the
  Federation rebase.
- Open, and a spike (S6): which σ carries authorization path weights. The corpus fixes the weight
  composition (`ringMul σ`) but not the face for this use; no face is chosen here by preference.

**D3 · Compiler.** Surface rules → flat image → proved total `decide`. Where Aeneas can eliminate the
compiler bug class, it does (owner: optional): the lowering `compile : Ast → Image` is written in the
fragment and refined (`decide (compile ast) = denote ast`, spike S7). The parser follows the Slang
precedent when it is worth it: tokenizer and parser are proved under the empty cone in
`proof/corpus/Slang/Source/{Tokenizer,Parsing}`. Until then the parser is tested trust.

**D4 · First consumer.** In-repo first: `docs/epics/frontier-consumer.md` R1 memory write/forget,
once Track B's Table-backed store exists; then the same seam for R2 (Penglai world actions), R4 (tool
calls) and R5 (actuators). Cross-repo first: intent's dispatch seam (W16), now linkable under Apache-2.0.

**D5 · Host.** This repository (`src/policy/`, `proof/corpus/Policy/`, `tools/policy-oracle/`), the
`spirv` precedent, under the shared-file protocol with the AI-stack lead. The canonical definition is
staged in `proof/corpus/Policy/Core.lean` and moves into the canonical corpus beside `Capability`
and `Table` with the corpus migration.

**D6 · Oracle.** `cedar-policy` 4.x, grade `tested`, on the 1-bit, error-free, unknown-free
fragment, in its own workspace root. Never cited as proof.

**D7 · Analysis (approved).** `Apeiron.Policy.Analysis` over `Apeiron.Decider.completeSat`: the
integer rows live on the σ = 1 interval cell (`face_is_forced`). It runs at compile/admission time and
on the `ask` path. Proved at spike scope in S5.

**Defence in depth, as layers with independent theorems.**

| Layer | Theorem | Fails closed by |
|---|---|---|
| Capability provenance (grammar) | `Capability.forge_reaches_only_scope` | no term builds a capability |
| Full-key authorization table | `Capability.guardedRead_ungranted`; `Extent.checked_no_alias` | sink on ungranted |
| Rule decision | `decide_default_deny`, `decide_forbid_wins`, `decide_granted_mono_completion` | unknown forbid applies |
| Relationship reachability | `traceRouteOfBudget_sound` | uncertified budget ⇒ `ask` |
| Static analysis | `neverAllows_of_canAllow_false` (S5) | MAYBE never reads as allow |
| Boundary (federation) | Wire `foreign_contact_contained` (old, `anchor` until rebased) | foreign contact zoned |

## Specification — Bounded authorization decision core

- **Owner:** Apeiron lead
- **Intent:** [intent](#intent--bounded-authorization-decision-core)
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty (see intent)
- **Status:** draft. Canonical names are fixed by spike S3a (`byId` replaces the keyword `by`);
  no repository declaration exists yet.

### Scope and admissible domain

| Item | Value |
|---|---|
| Operation | Rust `apeiron::policy::decide`; Lean `Apeiron.Policy.decide` |
| Domain | `rules: &[Rule]`, `atoms: &[Atom]`, `facts: &[u64]`, `known: u64`, `action: u32`, `want: u64`, **all values** |
| Totality | total; no panic, no `Result`, no allocation |
| Preconditions | none |
| Postcondition | `VerdictRep (decide …) (Policy.decide (rulesRep rules atoms) (reqRep facts known action want))` |

Fact slot `s` is **known** iff `s < 64 ∧ (known >> s) & 1 = 1 ∧ s < facts.len()`; otherwise it is
unknown. A rule's atom range `[atom_start, atom_start + atom_len)` that leaves `atoms` makes that
rule's guard unknown. Neither case narrows the domain; both are fail-closed through the unknown value.

### Canonical definition (`proof/corpus/Policy/Core.lean`, namespace `Apeiron.Policy`)

Imports: Lean core only (no Mathlib, no `Finset`, no `ZMod`, no `Classical`, no `noncomputable`),
so the Aeneas bridge and the empty-cone Slang chain can both import it.

```lean
inductive Tri | t | f | u                               -- Kleene (1938) three-valued logic
inductive Op  | eq | le | ge | all | any                 -- all/any: mask ⊆ v, mask ∩ v ≠ 0
structure Atom where op : Op; slot : Nat; value : Nat
inductive Effect | permit | forbid | ask
structure Rule where id : Nat; effect : Effect; action : Nat; rights : Nat; guard : Option (List Atom)
                                                          -- none = out-of-range guard (unknown)
structure Req  where facts : Nat → Option Nat; action : Nat; want : Nat
inductive Cause   | forbid (id : Nat) | noPermit
inductive Verdict | allow (g byId : Nat) | attenuate (g byId : Nat) | ask (g pending byId : Nat) | deny (c : Cause)

def Atom.eval : (Nat → Option Nat) → Atom → Tri          -- none ⇒ .u
def guardEval : (Nat → Option Nat) → Option (List Atom) → Tri   -- Kleene conjunction; none ⇒ .u
def W : Nat := 64
def mask (x : Nat) : Nat := x % 2 ^ W
def decide (rs : List Rule) (q : Req) : Verdict
--  rel r      := r.action = q.action
--  P          := ⋁ {r.rights | rel r, permit, guard = t, r.rights ∧ want ≠ 0};  pBy := min id
--  F          := ⋁ {r.rights | rel r, forbid, guard ≠ f};    fBy := min id with rights ∧ want ∧ P ≠ 0
--  A          := ⋁ {r.rights | rel r, ask,    guard ≠ f};    aBy := min id
--  g          := mask q.want ∧ P ∧ ¬F
--  g = 0      ⇒ deny (if mask q.want ∧ P = 0 then noPermit else forbid fBy)
--  g ∧ A ≠ 0  ⇒ ask g (g ∧ A) aBy
--  g = mask q.want ⇒ allow g pBy   (W1-T1 counterexample: comparing raw want makes want = 2^64 + 1 attenuate)
--  otherwise  ⇒ attenuate g pBy
def granted : Verdict → Nat                               -- deny ⇒ 0; ask ⇒ g
def Completes (q q' : Req) : Prop  -- same action and want; q' agrees on known slots, may fill unknown ones (S3a)

-- Song face: extent attenuation (no bitmask)
structure Extent where base : Nat; size : Nat
def sliceExtent (p : Extent) (off sz : Nat) : Option Extent :=
  if 0 < sz ∧ off + sz ≤ p.size then some ⟨p.base + off, sz⟩ else none
```

`⋁`, `∧` and `¬` are 64-step fuel-bounded bitwise operations, not core `Nat.lor`/`Nat.land`: core
`Nat.land` depends on `propext` under Lean `4.31.0` (lead probe `/home/socol/oc-tmp/policy-lead/Probe.lean`),
which puts every definition that uses it outside the empty cone. Spike S3a used `Nat.land`; its
`decide`, `guardEval` and `Atom.eval` depend on `[propext]`. The precedent is `Slang.Word.bitwise`
(`proof/corpus/Slang/Word.lean:37-50`). `¬F` is `(2^W - 1) - mask F`. The min id is an
`Option Nat` fold, so it is commutative and idempotent. The definitions must audit empty; the laws
may use the closed cone (S3a: `decide_default_deny` `[propext, Quot.sound]`, the others
`[propext, Classical.choice, Quot.sound]`).

**Complexity (AGENTS §8).** `decide` is time O(R·(1+A)) atom evaluations and O(1) words of space,
where R = `rules.len()` and A = the longest guard; derivation: one pass over `rules`, and for each
relevant rule one pass over its atom range; every atom is O(1) (one range check, one shift, one
comparison). `sliceExtent` is O(1).

### Laws (theorem sketches, all in `Apeiron.Policy`)

| Theorem | Statement | Control (throwaway copy of `Core.lean`) |
|---|---|---|
| `decide_default_deny` | `(∀ r ∈ rs, rel r → r.effect = permit → guardEval … ≠ t ∨ r.rights ∧ want = 0) → decide rs q = deny noPermit` | `g = 0 ⇒ allow` arm |
| `decide_forbid_wins` | `r ∈ rs → rel r → r.effect = forbid → guardEval … ≠ f → granted (decide rs q) ∧ r.rights = 0` | drop `¬F` from `g` |
| `decide_perm` | `rs.Perm rs' → decide rs q = decide rs' q` | replace min by first-seen id |
| `decide_dup` | `decide (r :: r :: rs) q = decide (r :: rs) q` | as above |
| `decide_granted_le_want` | `granted (decide rs q) ∧ ¬(mask q.want) = 0` | drop `mask q.want` from `g` |
| `decide_add_forbid_le` | `r.effect = forbid → granted (decide (r :: rs) q) ⊆ granted (decide rs q)` (⊆ as `a ∧ ¬b = 0`) | forbid contributes to `P` |
| `decide_deny_named` | `decide rs q = deny (.forbid i) → ∃ r ∈ rs, r.id = i ∧ rel r ∧ r.effect = forbid ∧ guardEval … ≠ f` | set `fBy := 0` |
| `decide_noPermit_sound` | `decide rs q = deny .noPermit → ∀ r ∈ rs, rel r → r.effect = permit → guardEval … = t → r.rights ∧ mask q.want = 0` | — |
| `decide_unknown_permit_inert` | a permit whose guard is `u` does not change `granted` | permit guard `≠ f` instead of `= t` |
| `decide_granted_mono_completion` | `Completes q q' → granted (decide rs q) ⊆ granted (decide rs q')` | forbid guard `= t` instead of `≠ f` |
| `oneBit_no_attenuate` | `mask q.want ≤ 1 → ∀ g b, decide rs q ≠ attenuate g b` | — |
| `noAsk_no_ask` | `(∀ r ∈ rs, r.effect ≠ ask) → ∀ g p b, decide rs q ≠ ask g p b` | — |
| `sliceExtent_le` | `sliceExtent p o s = some c → p.base ≤ c.base ∧ c.base + c.size ≤ p.base + p.size` | `off + sz ≤ p.size + 1` |
| `sliceExtent_refuses_widen` | `p.size < o + s → sliceExtent p o s = none` | same |
| witnesses | `witness_{allow,attenuate,ask,deny_noPermit,deny_forbid}` : concrete closed `rs`, `q`, proved by `rfl` on terms of ≤ 3 rules | each fails when its arm is mutated |

`rfl` is admitted only on these closed witnesses (≤ 3 rules, ≤ 2 atoms each), never over a source
or a quantified statement (AGENTS §6).

### Analysis layer (D7; `proof/corpus/Policy/Analysis.lean`, namespace `Apeiron.Policy.Analysis`)

| Item | Value |
|---|---|
| Imports | `Policy.Core`; `Apeiron.Decider.CompleteDecision` (corpus `880aa57e`, the bridge pin in `proof/corpus/lakefile.lean`) |
| Encoding | fact slot `s < n` ↦ column `s`; `boundRows n` (0 ≤ f ≤ 2⁶⁴ − 1); `atomRows`/`negAtomRows` give the DNF of an atom true/false, with integer strictness (¬(f ≤ v) ⇔ f ≥ v + 1) |
| Queries | `canAllow rs action n` (1-bit): `completeSat` over the DNF branches of `guard(p) ∧ ⋀ ¬guard(f) ∧ bounds` |
| Domain | `eq`/`le`/`ge` atoms with slot < n; `all`/`any` are not encoded (query returns the conservative answer, never allow) |
| Cost | branches = Π over forbids of their atom counts; each branch one `completeSat` call, whose residue search has no stated bound |
| Cone | closed (`completeSat` depends on `[propext, Classical.choice, Quot.sound]`) |

| Theorem | Statement | Control |
|---|---|---|
| `atomRows_iff` | for an `eq`/`le`/`ge` atom and ρ within bounds, `Atom.eval` is `t` ⇔ some branch of `atomRows` has `SatRows ρ` | `negAtomRows le` off by one |
| `neverAllows_of_canAllow_false` | `canAllow = false` → no in-bounds fully known request is allowed | same |
| `shadowed_witness`, `live_witness`, `unknown_exact_witness` | concrete systems (spike S5) | same |

### Rust face (`src/policy/mod.rs`, `src/policy/decide.rs`)

```rust
##[derive(Clone, Copy, PartialEq, Eq)] pub enum Op { Eq, Le, Ge, All, Any }
##[derive(Clone, Copy, PartialEq, Eq)] pub enum Effect { Permit, Forbid, Ask }
##[derive(Clone, Copy)] pub struct Atom { pub op: Op, pub slot: u32, pub value: u64 }
##[derive(Clone, Copy)] pub struct Rule { pub id: u32, pub effect: Effect, pub action: u32,
                                         pub rights: u64, pub atom_start: u32, pub atom_len: u32 }
##[derive(Clone, Copy, PartialEq, Eq)] pub enum Cause { Forbid(u32), NoPermit }
##[derive(Clone, Copy, PartialEq, Eq)] pub enum Verdict {
    Allow { granted: u64, by: u32 }, Attenuate { granted: u64, by: u32 },
    Ask { granted: u64, pending: u64, by: u32 }, Deny(Cause) }
pub fn decide(rules: &[Rule], atoms: &[Atom], facts: &[u64], known: u64,
              action: u32, want: u64) -> Verdict
```

Loops are `while i < len` with index access after the bound check (the `ty_get` shape). No
iterators, closures, traits or allocation. `u64::MAX` masks the width. Comparisons decide through
`if`, never through a `bool`-valued `match` (fragment table).

### Representation relation (Rust to Lean)

| Layer | Symbol | Path |
|---|---|---|
| Production Rust | `policy::decide` | `src/policy/decide.rs` |
| Extraction include | `#[path = "../../src/policy/mod.rs"] mod policy;` | `proof/extract/lib.rs` |
| Extraction root | `--start-from crate::policy::decide::decide` (no wrapper; the function is monomorphic) | `proof/regen.sh` |
| Extracted body | `policy.decide.decide`, `…decide_loop`, `…guard_loop` | `proof/ApeironWord.lean` (generated) |
| Canonical definition | `Apeiron.Policy.decide` | `proof/corpus/Policy/Core.lean` (staged; D5) |
| Representation | `ruleRep : Slice Rule → Slice Atom → List Policy.Rule`; `reqRep : Slice U64 → U64 → U32 → U64 → Policy.Req`; `VerdictRep` | `proof/corpus/Policy/Refinement.lean` |
| Refinement theorem | `word_decide_denote : ∃ v, decide … = ok v ∧ VerdictRep v (Policy.decide (ruleRep …) (reqRep …))` | `proof/corpus/Policy/Refinement.lean` |
| Song model | `Slang.Policy.Attenuation.slice` over `Slang.Word`, related to `sliceExtent` | `proof/corpus/Slang/Policy/Attenuation.lean` |

`ruleRep` maps an out-of-range atom range to `guard := none`. `reqRep` maps slot `s` to
`some facts[s].val` when known, else `none`. `VerdictRep` maps `u64`/`u32` by `.val`.

The loop proofs follow `Spirv.CheckScan.ty_get_spec` (`proof/corpus/Spirv/CheckScan.lean:173`,
`loop.spec_decr_nat` on a prefix invariant) and `table_build_loop_readRep`
(`proof/corpus/TableRefinementBuild.lean:1269`).

### Fragment fitness

| Needed feature | Existing extraction witness (`proof/ApeironWord.lean`, sha256 `72a8b7d4…`) | Status |
|---|---|---|
| Field-less enum + `match` | `coord.locator.locator_core.Refusal` `:834` | witnessed |
| Enum with data | `spirv.check.CheckError` `:13319`; `spirv.check.TagSpec` `:13391` | witnessed |
| `match` returning `Option` | `spirv.check.resolve_tag` `:13406` | witnessed |
| Loop over a slice, early exit, `Option` result | `spirv.check.ty_get_loop.body` `:13354`, proved by `ty_get_spec` | witnessed + proved |
| Nested loops | `coord.locator.locator_core.Directory.link_loop1` `:2752` | witnessed |
| `u64` `&&&`, `\|\|\|`, `^^^`, `~~~`, shifts | `:1213` (`&&&`), `:355` (`<<<`), `:18656` (`~~~`, `table.scan`) | witnessed |
| `Slice Std.U32` argument | `spirv.check.defs_contains` (proved `defs_contains_spec`, `CheckScan.lean:250`) | witnessed |
| Slice of a user enum | `Slice gguf.GVal` `:10555` | witnessed |
| Slice of a user struct (`&[Rule]`, `&[Atom]`) | spike S1: `Slice decide.Rule` / `Slice decide.Atom` read by `Slice.index_usize` | witnessed (S1) |
| `bool`-valued `match` whose arms are comparisons | spike S1: extracts to a `Prop`-typed bind; Lean fails `failed to synthesize Decidable holds` (Aeneas `nightly-2026.09.09-505b6ca`, patched model). A `let holds: bool` annotation and arm reordering both still fail. | **excluded by construction**: each arm decides with its own `if … { T } else { F }` (candidate sha256 `271d0e6c…`, S1 gate green) |
| Struct return with enum field (`Verdict` payloads) | spike S1: `inductive decide.Verdict` with `Allow : Std.U64 → Std.U32 → …` | witnessed (S1) |
| `Slice (Option _)` | none (grep 0) | avoided by design: `known: u64` mask |

The generated file is `noncomputable section`, so the extracted body stays in the closed cone. The
Slang face uses its own constructive model and shares only `Policy/Core.lean` definitions.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| Laws | table above | proved | `lake build Policy` |
| Refinement | `word_decide_denote`, total `ok` | proved | `Policy.Refinement` |
| Reachable verdicts | `word_decide_{allow,attenuate,ask,deny}_witness` on concrete slices | proved | `Policy.Refinement` |
| Song attenuation | `Slang.Policy.Attenuation.slice_denote`, `slice_le`, `slice_refuses_widen` | proved (empty cone) | `zero_axioms.py --stage policy-attenuation` |
| Rust nominal + boundaries | empty rules; `want = 0`; `u64::MAX`; slot 63/64; out-of-range atom range; unknown forbid; duplicate ids | tested | `src/policy/decide.rs` `#[cfg(test)]` |
| Cedar agreement | 1-bit, error-free fragment: decision equal; allow `by` = min Cedar reason; `deny (forbid i)` has i = min Cedar reason. Documented divergence: when no permit matches, Cedar's reason lists satisfied forbids and ours is `noPermit` (spike S2: 0 mismatches, 4,772 divergences in 20,000 cases, `cedar-policy` 4.13.0; control 6,581 mismatches) | tested | `tools/policy-oracle`; receipt `/home/socol/oc-tmp/policy-spike-s2/report.md` |

### Failure semantics

| Input class | Result | Condition |
|---|---|---|
| No relevant permit true | `Deny(NoPermit)` | law 2 |
| All permitted bits forbidden | `Deny(Forbid(min id))` | law 3 |
| Unknown slot / out-of-range atom range | atom `u`: permit inert, forbid/ask applies | laws 3, 7 |
| `want = 0` | `Deny(NoPermit)` | `g = 0` |
| Rule id collision | allowed; cause names the minimum id | no uniqueness precondition |

No panic path exists; the refinement theorem states `ok` for every input.

### Evidence plan

1. **Positive tests.** `src/policy/decide.rs` unit tests; `tools/policy-oracle` differential run.
2. **Proof.** `proof/regen.sh` regenerates the extraction; `cd proof/corpus && lake build Policy`.
3. **Axiom cone.** `proof/corpus/AxiomAudit.lean` `#audit`s every theorem above (closed cone);
   `tools/slang/zero_axioms.json` stage `policy-attenuation` audits the Song face with the empty cone.
4. **Negative controls.** Canonical: the per-law mutations above. Extraction: swap the permit and
   forbid arms of the generated `decide` in a throwaway `ApeironWord.lean` copy; `word_decide_denote`
   and `decide_forbid_wins`-transported lemmas must go red, and restore green. Source: in a throwaway
   copy of `src/policy/decide.rs`, change forbid applicability `!= F` to `== T`; the unknown-forbid
   unit test and, after re-extraction, the refinement must go red. Empty cone: add `Classical.choice`
   to a registered Song declaration; the stage must go red.
5. **Digests.** `src/policy/*.rs`, `proof/extract/lib.rs`, generated `ApeironWord.lean`; Aeneas
   `nightly-2026.09.09-505b6ca`, Charon `0.1.254`, Lean `4.31.0`, Mathlib `v4.31.0`, as run.
6. **Measurement.** None; no performance claim is made. The complexity comment is a derivation.

### Trust boundary

Not covered: surface compiler/parser; consumer fact encoding (strings, paths, modes); the
`cedar-policy` oracle; rustc/LLVM; Song `.slang` source binding of `ringGrantSlice` beyond its Lean
model; intent's single-dispatch property; any Table-carried index (v2).

### Open questions

- Is `ask` precedence (ask over partial grant) the intent approver semantics? Owner: intent lead.
- Rights width 64: sufficient for the intent tool vocabulary (danger classes, not tool names)?
- Should `attenuate` carry the removed mask as well as the granted one (for receipts)?

## Plan — Bounded authorization decision core

- **Owner:** Apeiron lead (closure); workers `deepseek-flash` via OpenCode
- **Intent:** [intent](#intent--bounded-authorization-decision-core)
- **Spec:** [specification](#specification--bounded-authorization-decision-core)
- **Study:** [`docs/study/policy-engines.md`](../study/policy-engines.md)
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty (see intent)
- **Status:** W0, W1 and W2 closed and lead-verified (every gate and control re-run by the lead).
  D1–D7 were derived and recorded; the user accepted them. Extraction identity `a4599646…`.
  Policy modules `Core`, `Laws` and `Analysis` are registered in `AxiomAudit.lean` (1,536 audits).
  Nothing is committed.
- **Next:** W3 refinement `word_decide_denote`. Proposed split into two tasks: (a) `Policy/RefinementGuard.lean`
  covers `atom_eval`/`guard_eval` against `Atom.eval`/`guardEval`, the U64 `&&&`/`|||`/`~~~` ↔
  `band`/`bor`/`ncomp` bridge (`UScalar.val_and`, `UScalar.val_or` in Aeneas `Std/Scalar/Bitwise.lean:340-341`) and
  `reqRep` (slot < 64 ∧ slot < len ∧ known bit ⇒ `some`); (b) `Policy/Refinement.lean` covers the
  `decide_loop0`/`decide_loop1` prefix invariants and the headline. The one nontrivial correspondence:
  Rust returns `Deny(NoPermit)` when `forbid_by = none`, but canonical returns `forbid ((forbBy …).getD 0)`.
  (b) must prove `forbBy ≠ none` when `granted = 0 ∧ want & permit ≠ 0`. Gates and briefs are not written yet.
- **Remaining scope after W3:** W4 Song attenuation (empty cone); W5 intent seam; relationship module (D2;
  S6 residuals: converse bridge, vertex-count certificate, visited-set search); compiler (D3; S7
  residuals: dead `CompileError::Cycle`, `decide (compile ast) = denote ast`); `revocation_is_forbid`
  (D1); analysis beyond 1-bit `eq`/`le`/`ge`; oracle coverage of `ask`, attenuation and unknown facts.
  Exit rows P1–P9 are in `docs/plan.md`, "Policy engine exit".
- **Process findings (for the AGENTS §10 register; owner edit):**
  - A worker's mutation control edited the working tree and was stranded by an unrelated compile error. Controls now run on a repo copy.
  - Three lead gates had defects that workers found: `pipefail` masking; `tail -1` on libtest output; `ugrep` as `grep`.
  - A worker's interactive question killed its session. Briefs now forbid it.

### Coordination with the wider roadmap

| Active lane | Shared resource | Rule for this epic |
|---|---|---|
| [read-proof-next](../epics/read-proof.md) (G1/REG-1 OPEN) | `proof/ApeironWord.lean`, `proof/corpus/.lake` | not displaced; W2/W3 queue behind its current task on the extraction lane |
| AI stack (inference parity, dequant/Hadamard lanes; `opencode --yolo` session; `/home/socol/oc-tmp/ds-dispatch/common.md` ownership map) | `proof/ApeironWord.lean`, `proof/regen.sh`, `proof/extract/lib.rs`, `AxiomAudit.lean`, `lakefile.lean`, lane lock | W0 uses none of them; W2-T1 (extraction) takes the lane only between AI-lane tasks, by lead scheduling |
| ANATIVE W4 frontier (`docs/state-ledger.tsv`) | `src/table*.rs`, `target/` | no overlap; every cargo run uses its own `CARGO_TARGET_DIR` under `/home/socol/oc-tmp` |
| Penglai | Aeneas toolchain (`target/aeneas → ../penglai/target/aeneas`) | read-only use |

Shared-file protocol agreed with the AI-stack lead (session `apeiron-c6`):

| File | Rule |
|---|---|
| `proof/corpus/lakefile.lean` (`lean_lib Policy`) | policy lead edits once the W1 Lean files exist; not during an announced full `lake build` barrier |
| `src/lib.rs` (`pub mod policy;`) | policy lead adds it in the same edit that creates `src/policy/mod.rs`, never before: a dangling `pub mod` breaks every cargo lane |
| `proof/corpus/AxiomAudit.lean` | the AI-stack lead folds the policy `#audit` and import lines into its registration batch after each module gate is green, and gates `AxiomAudit` |
| `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` | one writer at a time; message the AI-stack lead before regenerating (its next extraction is MoE W2). The regenerator runs `lake build` in `proof/corpus` (~385 s cold) and fixes its own modules; breakage in other modules goes to their owner. Identity at agreement: `ApeironWord.lean` sha256 `72a8b7d4…`, 23,436 lines. After the AI-stack barrier (MoE W2 + FFN include): `206e96ba…`, 24,009 lines; `proof/extract/lib.rs` `bd559395…`, `proof/regen.sh` `7f6f0d29…`; corpus 4,115 jobs, AxiomAudit 1,501 audits, `regen.sh --check` clean (`docs/evidence/ai-wave-receipt.md`). The policy include and root append after `ffn`/`moe`. The extraction lane is the policy lead's on request; message the AI-stack lead before and after regenerating |

Lean runs take `/tmp/opencode/apeiron-lane.lock`, `MemoryMax=12G`, `MemorySwapMax=0`,
`LEAN_NUM_THREADS=1`, and `timeout 540` (below the 600 s ceiling).

### Goal and completion condition

The intent's completion commands, all exit 0 on one recorded source/tool identity, with every
listed control red then green.

### Non-goals

As in the intent. In particular: no Table-carried index, no in-core reachability, no string
predicates, no intent-repository edit from this plan (W5 is a cross-repo task drafted for the
intent lead).

### Wave map

| Wave | Task | Files (owned) | Gate | Control | Grade |
|---|---|---|---|---|---|
| W0 | S1 fragment fitness | `/home/socol/oc-tmp/policy-spike-s1/{lib.rs,Witness.lean,report.md}` | `s1/gate.sh` (extract + audit + elaborate) | swap the forbid/permit arms in a copy of the generated Lean; `Witness.lean` red | tested (extraction) / proved (witness) |
| W0 | S2 Cedar alignment | `/home/socol/oc-tmp/policy-spike-s2/{Cargo.toml,src/main.rs,report.md}` | `cargo run --release -- 20000` | make the candidate ignore forbids (copy); mismatches > 0 | tested |
| W0 | S3a canonical laws | `/home/socol/oc-tmp/policy-spike-s3a/{Core.lean,Laws.lean,report.md}` | `s3a/gate.sh` (elaborate + axiom report) | per-law mutation in a copy | proved (spike scope) |
| W0 | S3b Song attenuation, empty cone | `/home/socol/oc-tmp/policy-spike-s3b/{Attenuation.lean,Axioms.lean,report.md}` | `s3b/gate.sh` (elaborate; every `#print axioms` empty) | insert `Classical.choice` use; gate red | proved (spike scope, empty cone) |
| W0 | S4 cedar-lean rebuild (optional) | `/home/socol/oc-tmp/policy-spike-s4/report.md` | `lake build Cedar.Thm.Authorization` in the clone | — | upgrades the study pointer to a local kernel check |
| W1 | T1 canonical core (fuel-bounded bitwise; definitions audit empty) | `proof/corpus/Policy/Core.lean` | `lake build Policy.Core` | S3a mutations | proved |
| W1 | T2 Rust core | `src/policy/mod.rs`, `src/policy/decide.rs` | `CARGO_TARGET_DIR=… cargo test --lib policy` | forbid `!= F` → `== T` in a copy | tested |
| W1 | T3 lakefile + lib.rs registration (lead) | `proof/corpus/lakefile.lean`, `src/lib.rs` | `cargo check`; `lake build Policy.Core` resolves | remove the glob; build fails | — |
| W2 | T1 extraction | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/ApeironWord.lean` | `proof/regen.sh --check` | delete the root; `--check` fails | tested (fresh) |
| W2 | T2 oracle crate | `tools/policy-oracle/{Cargo.toml,Cargo.lock,src/main.rs}` (own workspace root; usage in the `//!` header, no README) | `/home/socol/oc-tmp/policy-w2-t2/gate.sh` (fmt, clippy, `cargo test`, 20,000-case run) | forbids dropped in a copy of the adapter; mismatches > 0 | tested |
| W2 | T4 analysis layer (D7) | `proof/corpus/Policy/Analysis.lean` | `/home/socol/oc-tmp/policy-w1-t4/gate.sh` (`lake build Policy.Analysis`, 11 audited names) | S5 control (`negAtomRows le` off by one) | proved |
| W3 | T1a guard refinement; T1b decide refinement | `proof/corpus/Policy/RefinementGuard.lean`; `proof/corpus/Policy/Refinement.lean` | `lake build Policy.RefinementGuard` / `Policy.Refinement` (lead gate scripts to be written) | swap arms in a copy of the generated body; drop the known-bit test in `reqRep` (copy) | proved |
| W3 | T2 axiom audit | `proof/corpus/AxiomAudit.lean` | `lake build AxiomAudit` | register a `sorry` lemma in a copy; red | proved (cone) |
| W4 | T1 Song attenuation | `proof/corpus/Slang/Policy/Attenuation.lean`, `tools/slang/zero_axioms.json` | `python3 tools/slang/zero_axioms.py --stage policy-attenuation --controls` | built-in controls | proved (empty cone) |
| W5 | T1 intent seam adoption (cross-repo draft) | `../intent` files named by the intent lead | intent's own gates + `scripts/proof-register.py` | flipping `Guarded` to `Allow` reddens `guarded_denies_the_mutating_write_at_the_boundary` | tested → proved via the register |

W1-T3 is lead-owned because both files are shared with other lanes. W2-T1 and W3-T2 are
serialized on the extraction/corpus lanes.

### Task blocks

#### APEIRON-POLICY-SPIKE-S1 · fragment fitness on the real candidate

```text
SCOPE:      /home/socol/oc-tmp/policy-spike-s1/{lib.rs,Witness.lean,report.md}; extract
            policy-spike-common/decide.rs through Charon/Aeneas; complete when the generated Lean
            elaborates with no axiom/sorry and Witness.lean proves the empty-rules deny.
REGISTER:   refinement (fragment fitness); excludes the canonical link and repository edits.
CORPUS:     proof/corpus/Spirv/CheckScan.lean:173 `ty_get_spec` (loop.spec_decr_nat pattern);
            no policy declaration exists.
REFERENCE:  /home/socol/Workspace/apeiron/proof/regen.sh:116-140 (charon/aeneas flags);
            /home/socol/oc-tmp/policy-spike-common/decide.rs.
GATE:       /home/socol/oc-tmp/policy-spike-s1/gate.sh, cwd s1; red before lib.rs exists; 540 s.
GRADE:      tested (extraction elaborates) + proved (one witness, spike scope); before: no witness
            for a slice of a user struct.
```

#### APEIRON-POLICY-SPIKE-S2 · Cedar semantics alignment

```text
SCOPE:      /home/socol/oc-tmp/policy-spike-s2/{Cargo.toml,src/main.rs,report.md}; random 1-bit
            permit/forbid sets over Eq/Le/Ge atoms on integer context facts, decided by cedar-policy
            4.x and by decide.rs (#[path]); complete at 0 mismatches over 20,000 cases.
REGISTER:   test (differential); excludes ask, attenuation, unknowns and Cedar errors.
CORPUS:     files read, no match (oracle only); Cedar spec `isAuthorized`
            (/home/socol/oc-tmp/study-cedar-spec/cedar-lean/Cedar/Spec/Authorizer.lean).
REFERENCE:  /home/socol/oc-tmp/study-cedar/cedar-policy (API); decide.rs.
GATE:       cd s2 && CARGO_TARGET_DIR=/home/socol/oc-tmp/policy-spike-s2/target cargo run --release -- 20000;
            red before main.rs exists; 540 s.
GRADE:      tested; before: none.
```

#### APEIRON-POLICY-SPIKE-S3a · canonical laws

```text
SCOPE:      /home/socol/oc-tmp/policy-spike-s3a/{Core.lean,Laws.lean,report.md}; the spec's
            canonical definition (core Lean only) and at least decide_default_deny,
            decide_forbid_wins, decide_granted_le_want, decide_granted_mono_completion and the five
            witnesses; complete when the gate passes with the axiom report recorded.
REGISTER:   algebra (canonical laws); excludes the Rust link.
CORPUS:     ../sanguine/proof/Apeiron/Capability/MaskedTable.lean `guardedRead_ungranted`
            (proved, native) is the closest; no rule-set decision exists.
REFERENCE:  this file, "Canonical definition" and "Laws".
GATE:       /home/socol/oc-tmp/policy-spike-s3a/gate.sh (lean v4.31.0, no lake); 540 s.
GRADE:      proved at spike scope; the report lists each theorem's axioms.
```

#### APEIRON-POLICY-SPIKE-S3b · Song attenuation under the empty cone

```text
SCOPE:      /home/socol/oc-tmp/policy-spike-s3b/{Attenuation.lean,Axioms.lean,report.md}; a
            constructive model of ../song/src/kernel/mem/ring.slang `ringGrantSlice` over UInt64,
            related to `sliceExtent`, with slice_le and slice_refuses_widen; complete when every
            registered declaration prints no axioms.
REGISTER:   refinement (model) under the empty cone; excludes .slang source binding.
CORPUS:     proof/corpus/Slang/Audit/Empty.lean (`#audit_empty` mechanism); no match for extents.
REFERENCE:  /home/socol/Workspace/song/src/kernel/mem/ring.slang:138-200; src/kernel/tests/grant.slang.
GATE:       /home/socol/oc-tmp/policy-spike-s3b/gate.sh; 540 s.
GRADE:      proved (empty cone, spike scope).
```

#### APEIRON-POLICY-SPIKE-S4 · local kernel check of cedar-lean (optional)

```text
SCOPE:      /home/socol/oc-tmp/policy-spike-s4/report.md; build Cedar.Thm.Authorization in the
            cedar-spec clone with its pinned toolchain; complete on exit 0 or a recorded blocker.
REGISTER:   test of a citation; excludes adoption.
CORPUS:     cedar-lean/Cedar/Thm/Authorization.lean:64-249.
REFERENCE:  /home/socol/oc-tmp/study-cedar-spec/cedar-lean.
GATE:       timeout 540 lake build Cedar.Thm.Authorization (cwd cedar-lean); 540 s.
GRADE:      proved (upstream, locally rebuilt) or blocked.
```

W1-T3 (the `lean_lib Policy` glob) was done by the lead with the seed of `Policy/{Core,Laws}.lean`. W1–W4 task blocks are written at the W0 barrier from the spike reports, with the Arc Block filled
from observed red output. A task whose gate cannot be named is not dispatched.

### W0 results (lead re-run of every gate and control)

| Spike | Gate | Control | Result | Grade | Receipt |
|---|---|---|---|---|---|
| S1 fragment fitness | `policy-spike-s1/gate.sh` exit 0 (3 s) | extracted `NoPermit` → `Forbid 0`: `Witness.lean:36:58` unsolved goals; restore exit 0 | extraction elaborates; slice of a user struct witnessed. Defect found: a `bool`-valued `match` over comparisons extracts to a `Prop` bind (`failed to synthesize Decidable holds`); the candidate now decides each arm with `if` (sha256 `271d0e6c…`) | tested; witness `empty_deny_noPermit` proved (closed cone) | `/home/socol/oc-tmp/policy-spike-s1/report.md` |
| S2 Cedar alignment | `cargo run --release -- 20000` exit 0: 0 mismatches, 4,656 allow / 15,344 deny, 0 Cedar errors | forbids ignored: 6,581 mismatches, exit 1 | agreement on the 1-bit error-free fragment; 4,772 documented divergences (Cedar names forbids when no permit matched) | tested | `/home/socol/oc-tmp/policy-spike-s2/report.md` |
| S3a canonical laws | `policy-spike-s3a/gate.sh` exit 0 | `¬F` removed: `Laws.lean:44:21`, `:52:18`, `:62:20` red; restore exit 0 | `decide_default_deny`, `decide_forbid_wins`, `decide_granted_le_want`, `decide_granted_mono_completion`, five witnesses | proved (closed cone, spike scope) | `/home/socol/oc-tmp/policy-spike-s3a/report.md` |
| S3a definitions | lead probe | — | `decide`, `guardEval`, `Atom.eval` depend on `[propext]` via core `Nat.land`; the definitions need fuel-bounded bitwise operations (`Slang/Word.lean:37-50` precedent) before the empty-cone chain can import them | finding | `/home/socol/oc-tmp/policy-lead/{Probe,DefAxioms}.lean` |
| S3b (`Int` model) | red | — | core `Int` order lemmas carry `propext`; superseded | — | `/home/socol/oc-tmp/policy-spike-s3b/report.md` |
| S3b2 Song slice on `Slang.Word` | `policy-spike-s3b2/gate.sh` exit 0 | signed `gt` replaced by an unsigned comparison: `SongSlice.lean:194:73` red | `songSliceW` transcribes `ring.slang:190-197`; `songSliceW_widen_witness` proves the wrap: `offset = 2⁶³ − 1`, `subSize = 1` mints a grant outside the parent. Fixed comparison `songSliceFixed_le` (stretch) | proved (empty cone) for the required set; stretch `[propext, Quot.sound]` | `/home/socol/oc-tmp/policy-spike-s3b2/report.md` |
| S4 cedar-lean | `lake build Cedar.Thm.Authorization` exit 0 (119 jobs) | citation check | four authorizer theorems rebuilt locally, closed cone | proved upstream (pointer for Apeiron claims) | `/home/socol/oc-tmp/policy-spike-s4/report.md` |
| S5 analysis (D7) | `policy-spike-s5/gate.sh` exit 0 (3 s) | `negAtomRows le` off by one (`f ≥ v` for `f ≥ v + 1`): `Analysis.lean:295:61` (`unknown_exact_witness`) and `:521:55` (`negAtomRows_iff`) red; restore exit 0 | `atomRows_iff`, `negAtomRows_iff`, `shadowed_witness`, `live_witness`, `unknown_exact_witness` (analysis allows every value of an unknown `f0`; Kleene `decide` returns `deny noPermit` on the same request), `neverAllows_of_canAllow_false` and `exists_allow_of_canAllow_true` (the latter assumes no relevant `ask` rule). Scope: 1-bit rights, `eq`/`le`/`ge` atoms, slots < n ≤ 64, fully known bounded facts; the exactness witness flattens single-branch negations, so the general unknown-resolution theorem is open | proved (closed cone, spike scope) | `/home/socol/oc-tmp/policy-spike-s5/report.md` (Analysis.lean sha256 `7ff859ec…`) |

**W1 barrier (T1–T3).** Focused gates and controls are in the table below. The shared gates ran on the AI lane's barrier at
`proof/ApeironWord.lean` `a4599646…`, with `src/policy/decide.rs` `4f7eace2…` unchanged since: `cargo test` 438 passed,
including the 12 `policy::` tests; clippy; fmt; `regen.sh --check` 1,708 jobs; corpus `lake build` 4,115 jobs; `AxiomAudit`
(`/tmp/opencode/barrier-w5/`). `lean_lib Policy` is not a default target, so `Policy.*` is on the barrier path only
through the `AxiomAudit` import (lines handed over). T4 (analysis) moved to W2: nothing in W1 depends on it,
and it depends only on the W1-T1 core.

### W1 and follow-up spike results (lead re-run of every gate and control)

| Task | Gate | Control | Result | Grade | Receipt |
|---|---|---|---|---|---|
| W1-T2 Rust core | `policy-w1-t2/gate.sh` exit 0; 12 `policy::` tests | forbid applicability `!= F` → `== T` in a copy: `unknown_forbid_applies_and_unknown_permit_is_inert` fails | `src/policy/{mod,decide}.rs` = the S1-verified candidate plus rustfmt layout and three reasoned lint attributes; `src/lib.rs` gains only `pub mod policy;`. Gate defect found by the worker and fixed: under `pipefail` the clippy check could not go red while another lane's clippy failure was present; it now greps a captured log | tested | `/home/socol/oc-tmp/policy-w1-t2/report.md`; `decide.rs` sha256 `4f7eace2…` |
| W1-T1 canonical core | — | — | first run escalated a kernel-checked counterexample: `allow` compared raw `want`; `want = 2^64 + 1` gave `attenuate`. Spec corrected to `g = mask q.want`; re-dispatched | — | `/home/socol/oc-tmp/policy-w1-t1/scratch/Check.lean` |
| W1-T1 canonical core (final) | `policy-w1-t1/gate.sh` exit 0: `lake build Policy.Laws`; 20 audited names | `gOf` drops `ncomp (forbRights rs q)` in a copy: `Laws.lean` exit 1, first red `Policy/Laws.lean:106:28` (`rewrite` failed), `decide_forbid_wins` at `:145:18` | `band`/`bor`/`ncomp` are fuel-bounded (`Slang.Word.bitwise` construction); allow compares `gOf rs q = mask q.want`. Definitions `Atom.eval`, `guardEval`, `mask`, `decide`, `granted`, `Completes` and the five witnesses: no axioms. Laws `decide_default_deny`, `decide_forbid_wins`, `decide_granted_le_want`, `decide_granted_mono_completion`, `decide_perm`, `decide_add_forbid_le`, `decide_deny_named`, `oneBit_no_attenuate`, `noAsk_no_ask`: within `{propext, Classical.choice, Quot.sound}`. `Core.lean` `c09cc0cd…` (874 lines; lead corrected the `decide` complexity comment), `Laws.lean` `a32770f6…` (509 lines). registered in `proof/corpus/AxiomAudit.lean` (`import Policy.Laws`, 20 `#audit`); `lake build AxiomAudit` exit 0, 3,988 jobs (`/tmp/opencode/axiomaudit-policy.log`) | proved (kernel; cone as listed); Rust correspondence `anchor` until `word_decide_denote` | `/home/socol/oc-tmp/policy-w1-t1/report.md`, `axioms.txt`; `/tmp/opencode/policy-w1-t1-lead-{gate2,control}.log` |
| S6 relationships (D2) | `policy-spike-s6/gate.sh` exit 0 | exhaustion read as found: `Relations.lean:127:55` (`reachWithin_t_sound`) red | `ReachBudget` (mirror of `TraceBudget`), `reachWithin_t_sound`, `reachWithin_certified_f_sound`, three witnesses; `[propext, Quot.sound]` at most. Corpus facts: `RelRuns σ g Set.univ` is empty (`RunsN.exit` needs the final vertex outside `S`, `RouteAnyRing.lean:578`), so soundness is stated against reflexive-transitive reachability, bridged one way; with unit weights the path weight is `ringOne` on every face (`ringOne_mul`, `ringMul_one`, `Trichotomy/AnyRing.lean:203,213`), so σ does not enter reachability. Open: converse bridge, vertex-count certificate constructor, visited-set search | proved (closed cone, spike scope) | `/home/socol/oc-tmp/policy-spike-s6/report.md` |
| S7 compiler lowering (D3) | `policy-spike-s7/gate.sh` exit 0 | ≤/≥ complements swapped: 4 of 17 tests fail | arena-encoded guard → DNF clauses in caller buffers; fuel-bounded recursion extracts as `partial_fixpoint` and elaborates; 17 tests incl. 1,792 differential checks against direct evaluation; `single_leaf_compiles` proved. Gate defect found by the worker and fixed (`tail -1` read libtest's trailing blank line). Open: `CompileError::Cycle` is dead (cycles exhaust fuel as `TooLarge`); the refinement `decide (compile ast) = denote ast` | tested; witness proved (closed cone) | `/home/socol/oc-tmp/policy-spike-s7/report.md` |
| W2-T1 extraction | `policy-w2-t1/gate.sh` exit 0 (`regen.sh --check` fresh; `proof/` build 1,708 jobs) | `src/policy/decide.rs` mutated in a repo copy (`want & permit & !forbid` → `want & permit`): the copy's `regen.sh --check` exits 1 with `stale: ApeironWord.lean does not match regeneration from current source`; restored copy prints `extraction is fresh`, exit 0 | `proof/ApeironWord.lean` `a4599646…` (24,413 lines) carries `policy.decide.*` and the `word_policy_decide` root; `proof/extract/lib.rs` `448702be…` (the AI lane's `par`/`fused` mirrors appended after every extracted span), `proof/regen.sh` `b7246021…`. The first worker control was invalid (overtaken by an unrelated E0432) and was superseded by the copy control; the mutation never touches the working tree. Gate defect fixed: `grep` here is `ugrep`, so source checks use fixed strings | tested (extraction fresh at this identity) | `/tmp/opencode/policy-w2-t1-lead-gate.log`; control `/home/socol/oc-tmp/policy-w2-t1/control-copy.sh` |

| W2-T4 analysis layer (D7) | `policy-w1-t4/gate.sh` exit 0: `lake build Policy.Analysis`; 11 audited names, closed cone | `negAtomRows` `Op.le` `f ≥ v+1` → `f ≥ v` in a copy: exit 1 at `unknown_exact_witness` and `negAtomRows_iff` | S5 ported onto the W1-T1 core; the only statement change is `r.rights &&& 1` → `band r.rights 1` in `hnoask` (lead diff against S5). `Policy/Analysis.lean` `ff1fc782…`; registered in `AxiomAudit.lean` (`import Policy.Analysis`, 11 `#audit`), `lake build AxiomAudit` exit 0, 4,001 jobs (`/tmp/opencode/axiomaudit-policy2.log`). Scope: 1-bit want, `eq`/`le`/`ge` atoms, slots < n; `completeSat` search unbounded | proved | `/home/socol/oc-tmp/policy-w1-t4/report.md`; `/tmp/opencode/policy-w1-t4-lead-{gate,control}.log` |
| W2-T2 Cedar oracle (D6) | `policy-w2-t2/gate.sh` exit 0: fmt, clippy, `cargo test`, 20,000 cases: 0 mismatches, 0 Cedar errors, 4,656 allow / 15,344 deny, 4,772 documented divergences | forbids dropped before `apeiron::policy::decide` in a copy: exit 1, 6,581 mismatches | `tools/policy-oracle/{Cargo.toml `89431524…`, Cargo.lock `b3d89221…`, src/main.rs `760d22b0…`}`, own workspace root, `cedar-policy =4.13.0`, calls the real crate. Divergence class: `Deny(NoPermit)` where Cedar names satisfied forbids. Not covered: `ask`, attenuation, unknown facts, `all`/`any` | tested | worker log `/home/socol/oc-tmp/policy-dispatch/w2-t2.out` |

**Song-lead note (not applied to `../song`).** `ringGrantSlice` (`../song/src/kernel/mem/ring.slang:195`,
rev `aeaf3baa`) tests `offset + subSize > parentSize` on wrapping 64-bit words
(`src/prover/cone/ncone0.slang:136-137`). Model theorem `Apeiron.Policy.songSliceW_widen_witness`
(empty cone): parent `⟨20000000, 1024⟩`, `offset = 2⁶³ − 1`, `subSize = 1` returns a grant based at
`20000000 + 2⁶³ − 1`. Proposed comparison: refuse `offset > parentSize`, then
`subSize > parentSize − offset`, never forming the sum (`songSliceFixed_le`). The Song source change
and its execution test belong to the Song lead; this is a model result, not an execution receipt.

### Worker task table (deepseek-flash)

| ID | Artifact | Owned paths | Gate | Evidence path | Budget |
|---|---|---|---|---|---|
| APEIRON-POLICY-SPIKE-S1 | fragment fitness | `/home/socol/oc-tmp/policy-spike-s1/*` | `gate.sh` | `s1/report.md`, `/tmp/opencode/policy-s1.log` | 540 s/cmd, 60 min |
| APEIRON-POLICY-SPIKE-S2 | Cedar DRT | `/home/socol/oc-tmp/policy-spike-s2/*` | `cargo run --release -- 20000` | `s2/report.md`, `/tmp/opencode/policy-s2.log` | 540 s/cmd, 60 min |
| APEIRON-POLICY-SPIKE-S3a | canonical laws | `/home/socol/oc-tmp/policy-spike-s3a/*` | `gate.sh` | `s3a/report.md`, `/tmp/opencode/policy-s3a.log` | 540 s/cmd, 90 min |
| APEIRON-POLICY-SPIKE-S3b | Song attenuation | `/home/socol/oc-tmp/policy-spike-s3b/*` | `gate.sh` | `s3b/report.md`, `/tmp/opencode/policy-s3b.log` | 540 s/cmd, 60 min |
| APEIRON-POLICY-SPIKE-S3b2 | Song slice on `Slang.Word` | `/home/socol/oc-tmp/policy-spike-s3b2/*` | `gate.sh` | `s3b2/report.md` | 540 s/cmd, 60 min |
| APEIRON-POLICY-SPIKE-S5 | analysis on the decider (D7) | `/home/socol/oc-tmp/policy-spike-s5/*` | `gate.sh` | `s5/report.md`, `/tmp/opencode/policy-s5.log` | 540 s/cmd, 90 min |
| APEIRON-POLICY-SPIKE-S6 | relationships on the adjacency face, certified budget (D2) | `/home/socol/oc-tmp/policy-spike-s6/*` | `gate.sh` | `s6/report.md` | 540 s/cmd, 90 min |
| APEIRON-POLICY-SPIKE-S7 | compiler lowering in the fragment (D3) | `/home/socol/oc-tmp/policy-spike-s7/*` | `gate.sh` | `s7/report.md` | 540 s/cmd, 90 min |
| APEIRON-POLICY-W1-T1 | canonical core, empty-cone definitions, remaining laws | `proof/corpus/Policy/{Core,Laws}.lean` (seeded from S3a by the lead) | `/home/socol/oc-tmp/policy-w1-t1/gate.sh` | `policy-w1-t1/report.md` | 540 s/cmd, 120 min |
| APEIRON-POLICY-W1-T2 | Rust core | `src/policy/{mod,decide}.rs` + one line in `src/lib.rs` (added after the files exist) | `/home/socol/oc-tmp/policy-w1-t2/gate.sh` | `policy-w1-t2/report.md` | 540 s/cmd, 60 min |
| APEIRON-POLICY-W1-T4 | analysis layer over `Apeiron.Decider` (D7), ported from S5 to the verified core | `proof/corpus/Policy/Analysis.lean` | `/home/socol/oc-tmp/policy-w1-t4/gate.sh` | `policy-w1-t4/report.md` | 540 s/cmd, 120 min |
| APEIRON-POLICY-SPIKE-S4 | cedar-lean rebuild | `/home/socol/oc-tmp/policy-spike-s4/*` | `lake build Cedar.Thm.Authorization` | `s4/report.md` | 540 s/cmd, 30 min |

Escalation stays with the lead: a changed contract premise, theorem statement, attribution, grade
or control, more than three files or one gate, or a governing-document conflict stops the worker.

### Barrier checklist (W0)

- [ ] Each spike report has gate exit status, elapsed seconds from `date`, and the control's red line.
- [ ] No file under `/home/socol/Workspace/apeiron` changed (`git status --short` identical before and after).
- [ ] Fragment blockers, if any, named with the failing Charon/Aeneas message.
- [ ] Empty-cone feasibility of `Policy/Core.lean` definitions recorded (S3a axiom report).

### Stop and rollback conditions

- A spike needs a repository edit: stop and report.
- Charon/Aeneas rejects a slice of a user struct: redesign to struct-of-slices (parallel `&[u32]`/`&[u64]`), recorded as a spec change for the lead.
- Three materially different failed approaches: report the smallest unresolved goal.

### Owner-decision draft for `docs/direction.md` (not applied)

> **Authorization decision core (owner decision pending).** Apeiron hosts one bounded
> authorization decision, the canonical `Apeiron.Policy.decide` over a fixed fact record and a W-bit
> rights order. Its verdict is `allow | attenuate | ask | deny(cause)`, with default deny,
> forbid-wins, order independence, fail-closed unknowns, a named cause on every deny, and
> attenuation as monotone restriction. The Rust `src/policy` core is extracted and refined through
> the Aeneas lane. Song's capability attenuation corresponds under the empty cone. The intent
> dispatch seam is the first consumer, as its 1-bit specialisation. Rule content, fact encoding
> and administration stay application policy outside the core. `cedar-policy` is a `tested`
> differential oracle, never a proof. The definition is staged in `proof/corpus/Policy/Core.lean`
> and moves with the corpus migration. This thread does not displace G1/REG-1, the read/search leg,
> the AI stack or Penglai. An analysis layer decides quantified policy questions on the canonical
> integer-row decider (`Apeiron.Decider.completeSat`), with no external solver in the trust base. Epic: [policy-engine](epics/policy-engine.md).
