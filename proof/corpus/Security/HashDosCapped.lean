import Security.HashDos
import Map.ExecutedProbe
import Map.ExecutedOverflowAvlBase
import Mathlib.Tactic

/-!
# HashDOS over the SHIPPED capped probe — the bound that needs no hypothesis on the key set

`Security/HashDos.lean` models the **uncapped** descent: `plen` walks until it finds a free lane, so
its length is a function of the occupied set, `descent_can_be_wholly_blocked` fills every lane of a
victim's whole prefix with `PHI⁻¹`-aimed keys, and the only worst case the model admits is `Θ(n²)`
(`build_bounded_by_n_squared`). That model is correct about the uncapped table and is kept as such.

The shipped table is not that table. `src/map/hashtrinity/overflow.rs` reads at most
`probe_limit = min PROBE_CAP (LINEAR_G + 64 / bits - 1)` primary slots
(`probe_rest_with`'s `while j < limit`), and a key unresolved inside that budget is routed to the
AVL overflow tree instead of a terminal sweep. This file states and proves the three bounds that
replace the `Θ(n²)` ceiling, over the shipped objects:

* **§A — the cap, for every key set including an adversary's.** The probe's examination count is at
  most the budget, and the budget is at most `PROBE_CAP = 16`
  (`exams_le_cap`, `extracted_probe_limit_le_cap`). No hypothesis on the occupied set, no
  termination premise: the bound is a property of the *loop*, so the filling construction of
  `descent_can_be_wholly_blocked` cannot raise it. It can only send the key to §B.
* **§B — the overflow tree is `O(log n)` in AVL height.** `size_ge_two_pow_half` is the classic AVL
  size/height inequality `2^(ht/2) ≤ size + 1`; `avl_height_le` reads it as
  `ht ≤ 2·log2(size + 1) + 1`. `Bal` is preserved by the extracted bodies
  (`insert_node_spec` in `Map/ExecutedOverflowInsert.lean:363` returns `Bal T'` and
  `(idxs T').Perm (i :: idxs T)`; `rebalance_spec` in `Map/ExecutedOverflowAvl.lean:103`).
* **§C — the aggregate.** `capped_build_cost_le`: `n` operations, each costing at most `PROBE_CAP`
  primary examinations plus the overflow height at that moment, sum to
  `n·(16 + 2·log2(n + 1) + 1)` — `O(n·(PROBE_CAP + log n))` for **any** key stream.

## What is linked to the extracted body, and what is not

Linked. `Apeiron.MapExecuted.limN` is the extraction's own cap expression, and
`probe_limit_val` (`Map/ExecutedProbe.lean:178`) proves the extracted
`hashtrinity.overflow.Table.probe_limit` returns exactly `limN bits`; `probe_rest_spec`,
`pcc_spec` and `pcw_spec` (`Map/ExecutedProbe.lean:286`, `:660`, `:796`) prove the extracted
`probe_rest_with` / `probe_capped_cold` / `probe_capped_with` loops satisfy `CappedOK`, whose
`Blocked` arm is "every step below `limN bits` passed". §A's `extracted_probe_limit_le_cap`,
`blocked_all_steps_pass` and `vacant_stop_below_cap` are stated over those objects. §B's `Bal`,
`ht`, `idxs` are the extracted arena's shape predicates (`Emb` of `o.nodes.val`), and the AVL
invariant's preservation is the cited `insert_node_spec` / `rebalance_spec`, not a new model.

Not linked, named. §A's `exams` count is a model of the loop's *count*, not a ghost counter on the
extracted loop body: the extracted loop is proved to satisfy `CappedOK` (the step set), and the
count is the model's; a total-correctness variant on `probe_rest_with_loop` that returns the number
of examinations is open, and no theorem here claims it. The shipped `Table::descend` body *is* in
the extraction (`proof/ApeironWord.lean:109330`) — `Security/HashDos.lean` §6(a) said it was absent,
which was true of the extraction it was written against and is stale now (§7 below). §C's cost list
takes the per-step AVL shape as given: the induction over a *stream* of `insert_node` calls that
builds that list from a key stream (using `insert_node_spec`'s `Bal` and `Perm` results) is open;
what is proved is the aggregate arithmetic plus the per-step bound, both structural.

No `sorry`, no `axiom`, no `maxHeartbeats`: every headline carries an in-module `#print axioms` row.
-/

set_option linter.unusedVariables false
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Finset
open Apeiron.MapExecuted
open Aeneas Aeneas.Std Result
open apeiron_word

namespace Apeiron.Security.HashDos

-- ═══════════════════════════════════════════════════════════════════
-- §A · THE CAPPED PROBE — a bounded walk, counted
-- ═══════════════════════════════════════════════════════════════════
-- The probe is `while j < limit { examine descend(coord, j); if hit/vacant stop; j += 1 }`
-- (`src/map/hashtrinity/overflow.rs:695-708`, extracted as `probe_rest_with_loop.body`). One
-- examination per step, so the examination count is a prefix length: the walk examines step `j`
-- and stops, or examines the whole budget `limit`. `examsFrom` is that count, and it is a count of
-- the loop, not of the address function — it is therefore the same number for every `seq` and
-- every occupied set.

/-- The slot examinations of a capped walk over `n` steps from step `j`: one per step, stopping at
the first slot outside `O`. Temporal: `O(n)` slot membership tests, one per step, the recursion
closing because `n` strictly decreases; spatial: `O(n)` stack, the same bound. -/
def examsFrom (seq : ℕ → ℕ) (O : Finset ℕ) : ℕ → ℕ → ℕ
  | 0, _ => 0
  | n + 1, j => if seq j ∈ O then 1 + examsFrom seq O n (j + 1) else 1

/-- The slot examinations of a capped walk with budget `limit`. -/
def exams (seq : ℕ → ℕ) (O : Finset ℕ) (limit : ℕ) : ℕ := examsFrom seq O limit 0

/-- A walk never examines more steps than its budget. -/
theorem examsFrom_le (seq : ℕ → ℕ) (O : Finset ℕ) (n : ℕ) :
    ∀ j, examsFrom seq O n j ≤ n := by
  induction n with
  | zero => intro j; exact Nat.le_of_eq rfl
  | succ n ih =>
    intro j
    unfold examsFrom
    by_cases h : seq j ∈ O
    · rw [if_pos h]; have := ih (j + 1); omega
    · rw [if_neg h]; omega

theorem exams_le (seq : ℕ → ℕ) (O : Finset ℕ) (limit : ℕ) : exams seq O limit ≤ limit :=
  examsFrom_le seq O limit 0

/-- **The budget is `PROBE_CAP` or less.** `limN` is the extraction's own cap expression
(`min PROBE_CAP (LINEAR_G + 64 / bits - 1)`, `Map/ExecutedProbe.lean:49`); it never exceeds the
shipped `PROBE_CAP = 16`. -/
theorem limN_le_cap (b : ℕ) : limN b ≤ 16 :=
  Nat.min_le_left 16 _

/-- **§A headline — the cap holds for every key set.** For every address sequence `seq`, every
occupied set `O` (including one an adversary filled by `PHI⁻¹` aiming) and every window width `b`,
a capped probe examines at most `limN b ≤ PROBE_CAP = 16` primary slots. This is the statement the
uncapped model could not make: it carries **no** hypothesis on `O` and **no** termination premise,
because the bound is the loop's, not the walk's. -/
theorem exams_le_cap (seq : ℕ → ℕ) (O : Finset ℕ) (b : ℕ) : exams seq O (limN b) ≤ 16 :=
  le_trans (exams_le seq O (limN b)) (limN_le_cap b)

/-- A walk that examines its whole budget found no free slot. This is the `Blocked` verdict's
meaning (`CappedOK`'s `Blocked` arm), and it is what routes the key to the overflow. -/
theorem examsFrom_eq_of_occupied (seq : ℕ → ℕ) (O : Finset ℕ) (n : ℕ) :
    ∀ j, (∀ i < n, seq (j + i) ∈ O) → examsFrom seq O n j = n := by
  induction n with
  | zero => intro j _; rfl
  | succ n ih =>
    intro j h
    unfold examsFrom
    have h0 : seq j ∈ O := by simpa using h 0 (by omega)
    rw [if_pos h0]
    have h' : ∀ i < n, seq (j + 1 + i) ∈ O := by
      intro i hi
      have := h (i + 1) (by omega)
      simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using this
    rw [ih (j + 1) h']
    omega

theorem exams_eq_limit_of_occupied (seq : ℕ → ℕ) (O : Finset ℕ) (limit : ℕ)
    (h : ∀ i < limit, seq i ∈ O) : exams seq O limit = limit := by
  unfold exams
  exact examsFrom_eq_of_occupied seq O limit 0 (by simpa using h)

/-- **The cap is attained, so the bound is not slack-by-vacuity.** Sixteen distinct occupied slots
cost exactly sixteen examinations — the `Blocked` case the `crafted_keys_block_a_victims_whole_capped_prefix`
test executes (`tests/hashdos.rs`). -/
theorem cap_is_attained : exams (fun j => j) (Finset.range 16) 16 = 16 :=
  exams_eq_limit_of_occupied _ _ _ (fun i hi => Finset.mem_range.mpr hi)

/-- The shipped 14-bit window geometry: the budget is `min 16 (8 + 64/14 - 1) = 11`. -/
theorem shipped_limit_eq : limN 14 = 11 := by unfold limN; norm_num

/-- The model's walk over the shipped probe's own addresses (`seqLin`, the transcription of
`Table::descend` inside the window set) is capped at 16 for every occupied set. -/
theorem shipped_probe_exams_le (p : ℕ) (O : Finset ℕ) :
    exams (fun j => Sanguine.RadixDescent.seqLin 14 8 p j) O (limN 14) ≤ 16 :=
  exams_le_cap _ O 14

-- ═══════════════════════════════════════════════════════════════════
-- §A′ · THE SAME BOUND AT THE EXTRACTED BODY
-- ═══════════════════════════════════════════════════════════════════
-- `limN` is not a transcription: `probe_limit_val` proves the extracted `probe_limit` returns it,
-- and `CappedOK` (proved of `probe_rest_with` by `probe_rest_spec`) is the extracted loop's step
-- set. The three theorems below are stated over those objects directly.

/-- **The extracted `probe_limit` is at most `PROBE_CAP`, for every table state.** This is the
quantity the extracted loop compares against (`while j < limit`), so the loop's step index never
reaches `PROBE_CAP`. -/
theorem extracted_probe_limit_le_cap {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (s : Apeiron.MapExecuted.Tbl KS) (l : U32) (hb : 0 < s.bits.val)
    (h : hashtrinity.overflow.Table.probe_limit core.cmp.EqU64 Apeiron.MapExecuted.CI ks s = ok l) :
    l.val ≤ 16 := by
  rw [Apeiron.MapExecuted.probe_limit_val ks s l hb h]
  exact limN_le_cap s.bits.val

/-- **A `Blocked` verdict examined exactly the budget.** `CappedOK`'s `Blocked` arm: every step
below `limN bits` passed — the walk was full, and the key goes to the AVL overflow. -/
theorem blocked_all_steps_pass {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (s : Apeiron.MapExecuted.Tbl KS) (c : U64) (h : Apeiron.MapExecuted.CappedOK ks s c .Blocked) :
    ∀ j : U32, j.val < Apeiron.MapExecuted.limN s.bits.val →
      Apeiron.MapExecuted.Pass ks s c j :=
  h

/-- **A `Vacant` verdict stopped below the cap.** The vacant slot is the `j`-th probe slot for some
`j.val < limN bits ≤ 16`, so the walk to it cost at most `j.val + 1 ≤ 16` examinations — the
`Vacant` arm of the extracted loop's contract. -/
theorem vacant_stop_below_cap {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (s : Apeiron.MapExecuted.Tbl KS) (c : U64) (i : Usize)
    (h : Apeiron.MapExecuted.CappedOK ks s c (.Vacant i)) :
    ∃ j : U32, j.val < 16 ∧ Apeiron.MapExecuted.Desc ks s c j i := by
  obtain ⟨-, j, hj, hd, -⟩ := h
  exact ⟨j, lt_of_lt_of_le hj (limN_le_cap s.bits.val), hd⟩

end Apeiron.Security.HashDos

-- ═══════════════════════════════════════════════════════════════════
-- §B · THE OVERFLOW TREE — AVL height is `O(log n)`
-- ═══════════════════════════════════════════════════════════════════
-- Stated over the extracted arena's shape predicates: `ht`/`Bal`/`idxs` describe the `Emb` shape of
-- `o.nodes.val` (`Map/ExecutedOverflow.lean`), and the invariant's preservation is
-- `insert_node_spec` (`Map/ExecutedOverflowInsert.lean:363`) and `rebalance_spec`
-- (`Map/ExecutedOverflowAvl.lean:103`). What is new here is the size/height inequality.

namespace Apeiron.MapExecuted

/-- The height arithmetic of one AVL step: two subtrees whose heights differ by at most one carry at
least `2^((max+1)/2)` nodes between them. -/
theorem pow_half_add_le {a b : ℕ} (h1 : a ≤ b + 1) (h2 : b ≤ a + 1) :
    2 ^ ((max a b + 1) / 2) ≤ 2 ^ (a / 2) + 2 ^ (b / 2) := by
  have key : ∀ c d : ℕ, c ≤ d → d ≤ c + 1 →
      2 ^ ((d + 1) / 2) ≤ 2 ^ (c / 2) + 2 ^ (d / 2) := by
    intro c d hcd hcd1
    have hstep : d = c ∨ d = c + 1 := by omega
    rcases hstep with h | h
    · rw [h]
      have hle : (c + 1) / 2 ≤ c / 2 + 1 := by omega
      have hp : 2 ^ ((c + 1) / 2) ≤ 2 ^ (c / 2 + 1) := Nat.pow_le_pow_right (by norm_num) hle
      have hsum : 2 ^ (c / 2 + 1) = 2 ^ (c / 2) + 2 ^ (c / 2) := by rw [pow_succ]; ring
      rw [hsum] at hp
      exact hp
    · rw [h]
      have heq : (c + 1 + 1) / 2 = c / 2 + 1 := by omega
      rw [heq]
      have hp : 2 ^ (c / 2 + 1) = 2 ^ (c / 2) + 2 ^ (c / 2) := by rw [pow_succ]; ring
      have hmono : 2 ^ (c / 2) ≤ 2 ^ ((c + 1) / 2) :=
        Nat.pow_le_pow_right (by norm_num) (by omega)
      rw [hp]
      omega
  rcases le_total a b with hab | hba
  · rw [Nat.max_eq_right hab]
    exact key a b hab h2
  · rw [Nat.max_eq_left hba]
    have := key b a hba h1
    omega

/-- **The AVL size/height inequality.** An AVL tree of height `h` has at least `2^(h/2) - 1` nodes:
the minimal AVL tree of height `h` has `1 + m(h-1) + m(h-2)` nodes, and that recurrence dominates
`2^(h/2)`. -/
theorem size_ge_two_pow_half : ∀ t : OTree, Bal t → 2 ^ (ht t / 2) ≤ (idxs t).length + 1
  | .leaf, _ => by simp [ht, idxs]
  | .node l i r, hBal => by
    obtain ⟨hBl, hBr, hb1, hb2⟩ := hBal
    have h1 := size_ge_two_pow_half l hBl
    have h2 := size_ge_two_pow_half r hBr
    have hcore := pow_half_add_le hb1 hb2
    have hsum : 2 ^ (ht l / 2) + 2 ^ (ht r / 2)
        ≤ ((idxs l).length + 1) + ((idxs r).length + 1) := Nat.add_le_add h1 h2
    show 2 ^ ((max (ht l) (ht r) + 1) / 2) ≤ (idxs (.node l i r)).length + 1
    calc 2 ^ ((max (ht l) (ht r) + 1) / 2)
        ≤ 2 ^ (ht l / 2) + 2 ^ (ht r / 2) := hcore
      _ ≤ ((idxs l).length + 1) + ((idxs r).length + 1) := hsum
      _ = (idxs (.node l i r)).length + 1 := by
          simp only [idxs, List.length_append, List.length_cons]; omega

/-- **§B headline — the overflow tree's height is `2·log2(size + 1) + 1` or less.** For an AVL
shape (`Bal`, the invariant `insert_node_spec` and `rebalance_spec` preserve), the height the
overflow's own `height` field stores (`HOK`) is logarithmic in the node count. -/
theorem avl_height_le (t : OTree) (hBal : Bal t) :
    ht t ≤ 2 * Nat.log2 ((idxs t).length + 1) + 1 := by
  have hge : 2 ^ (ht t / 2) ≤ (idxs t).length + 1 := size_ge_two_pow_half t hBal
  have hle : ht t / 2 ≤ Nat.log 2 ((idxs t).length + 1) :=
    Nat.le_log_of_pow_le (by norm_num) hge
  rw [← Nat.log2_eq_log_two] at hle
  omega

/-- The same bound against any bound `n` on the node count — the form the aggregate uses, where the
tree at the `m`-th insert has at most `n` nodes. -/
theorem avl_height_le_of_len (t : OTree) (hBal : Bal t) {n : ℕ}
    (hn : (idxs t).length ≤ n) : ht t ≤ 2 * Nat.log2 (n + 1) + 1 := by
  have hb := avl_height_le t hBal
  have hmono : Nat.log2 ((idxs t).length + 1) ≤ Nat.log2 (n + 1) := by
    rw [Nat.log2_eq_log_two, Nat.log2_eq_log_two]
    exact Nat.log_mono_right (by omega)
  omega

end Apeiron.MapExecuted

namespace Apeiron.Security.HashDos

-- ═══════════════════════════════════════════════════════════════════
-- §C · THE AGGREGATE — `n·(PROBE_CAP + log n)` for any key stream
-- ═══════════════════════════════════════════════════════════════════

/-- **One operation's cost.** A lookup, insert, update or remove costs its capped primary walk
(`prim ≤ PROBE_CAP`) plus a descent of the overflow tree (`≤ ht t`). Given the AVL invariant and a
node count at most the stream length `n`, the sum is `PROBE_CAP + 2·log2(n + 1) + 1`. -/
theorem op_cost_le_capped (prim : ℕ) (t : OTree) (n : ℕ)
    (hp : prim ≤ 16) (hBal : Bal t) (hn : (idxs t).length ≤ n) :
    prim + ht t ≤ 16 + (2 * Nat.log2 (n + 1) + 1) := by
  have h1 := Apeiron.MapExecuted.avl_height_le_of_len t hBal hn
  omega

/-- **§C headline — the aggregate build cost, for ANY key stream.** `n` operations, each carrying
the primary count it examined (`s.2 ≤ PROBE_CAP`) and the AVL shape of the overflow at that moment
(`Bal`, at most `n` nodes), cost at most `n·(PROBE_CAP + 2·log2(n + 1) + 1)` in total:
`O(n·(PROBE_CAP + log n))`, not `Θ(n²)`.

The stream enters only through `n`: the primary term is bounded by the loop (`exams_le_cap`) and
does not read the keys at all, and the overflow term is bounded by the node count. Nothing here
depends on the distribution of the keys, on the occupied set, or on a termination premise — the
three things `descent_can_be_wholly_blocked` removes from the uncapped model. -/
theorem capped_build_cost_le (steps : List (OTree × ℕ)) (n : ℕ)
    (hlen : steps.length = n)
    (hBal : ∀ s ∈ steps, Bal s.1)
    (hsize : ∀ s ∈ steps, (idxs s.1).length ≤ n)
    (hcap : ∀ s ∈ steps, s.2 ≤ 16) :
    buildCost (steps.map (fun s => s.2 + ht s.1))
      ≤ n * (16 + (2 * Nat.log2 (n + 1) + 1)) := by
  have hb : ∀ c ∈ steps.map (fun s => s.2 + ht s.1),
      c ≤ 16 + (2 * Nat.log2 (n + 1) + 1) := by
    intro c hc
    obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hc
    exact op_cost_le_capped s.2 s.1 n (hcap s hs) (hBal s hs) (hsize s hs)
  have hmain := build_sum_le_length_mul (steps.map (fun s => s.2 + ht s.1))
    (16 + (2 * Nat.log2 (n + 1) + 1)) hb
  simpa [buildCost, hlen] using hmain

/-- The capped aggregate is below the uncapped model's `n²` ceiling whenever the log term and the cap
fit inside one key's worth of work — the comparison that replaces `build_bounded_by_n_squared` for
the shipped table. -/
theorem capped_below_quadratic {n : ℕ} (h : 16 + (2 * Nat.log2 (n + 1) + 1) ≤ n) :
    n * (16 + (2 * Nat.log2 (n + 1) + 1)) ≤ n * n :=
  Nat.mul_le_mul_left n h

-- ═══════════════════════════════════════════════════════════════════
-- §D · HEADLINES
-- ═══════════════════════════════════════════════════════════════════

/-- **The shipped table's HashDOS bound, conjoined.** For every key set, including an adversarial
one: (i) a primary probe examines at most `PROBE_CAP = 16` slots; (ii) the overflow tree's height is
`2·log2(size + 1) + 1` or less under the AVL invariant; (iii) a stream of `n` operations costs at
most `n·(PROBE_CAP + 2·log2(n + 1) + 1)`. -/
theorem capped_table_bounds :
    (∀ (seq : ℕ → ℕ) (O : Finset ℕ) (b : ℕ), exams seq O (limN b) ≤ 16) ∧
    (∀ t : OTree, Bal t → ht t ≤ 2 * Nat.log2 ((idxs t).length + 1) + 1) ∧
    (∀ (steps : List (OTree × ℕ)) (n : ℕ), steps.length = n →
      (∀ s ∈ steps, Bal s.1) → (∀ s ∈ steps, (idxs s.1).length ≤ n) →
      (∀ s ∈ steps, s.2 ≤ 16) →
      buildCost (steps.map (fun s => s.2 + ht s.1))
        ≤ n * (16 + (2 * Nat.log2 (n + 1) + 1))) :=
  ⟨fun seq O b => exams_le_cap seq O b,
   fun t h => Apeiron.MapExecuted.avl_height_le t h,
   fun steps n hlen hBal hsize hcap => capped_build_cost_le steps n hlen hBal hsize hcap⟩

end Apeiron.Security.HashDos

#print axioms Apeiron.Security.HashDos.exams_le
#print axioms Apeiron.Security.HashDos.exams_le_cap
#print axioms Apeiron.Security.HashDos.exams_eq_limit_of_occupied
#print axioms Apeiron.Security.HashDos.cap_is_attained
#print axioms Apeiron.Security.HashDos.shipped_probe_exams_le
#print axioms Apeiron.Security.HashDos.extracted_probe_limit_le_cap
#print axioms Apeiron.Security.HashDos.blocked_all_steps_pass
#print axioms Apeiron.Security.HashDos.vacant_stop_below_cap
#print axioms Apeiron.MapExecuted.pow_half_add_le
#print axioms Apeiron.MapExecuted.size_ge_two_pow_half
#print axioms Apeiron.MapExecuted.avl_height_le
#print axioms Apeiron.MapExecuted.avl_height_le_of_len
#print axioms Apeiron.Security.HashDos.op_cost_le_capped
#print axioms Apeiron.Security.HashDos.capped_build_cost_le
#print axioms Apeiron.Security.HashDos.capped_table_bounds
