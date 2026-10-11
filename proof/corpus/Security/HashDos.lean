import Closures.Engineering.RadixDescent
import ApeironWord
import Mathlib.Tactic

/-!
# HashDOS for the shipped address function — the structural bound, and the exact hole

The adversarial property, stated in the form the threat needs.

**Threat.** An adversary *chooses the key stream*. The attempt is to make the aggregate work
superlinear by (a) forcing many live keys onto **one address**, or (b) **lengthening a probe** — the
number of occupied lanes a key's descent walks through before it finds a free one. The measures are

* **lane occupancy** — how many coordinates share an address lane (`homeSlot b s`);
* **probe length** — `plen seq O h`, the first step at which the walk leaves the occupied set `O`;
* **build cost** — `buildCost (keys.map plen)`, the total probe work of inserting the stream.

The address is a mixing function of the full key coordinate: `c ↦ ⌊c · s / 2^(W−b)⌋` with `s` a unit
of `ZMod (2^W)` (odd). What that buys, and what it does not, is below.

## What is proved here, over which object, and what is not

* **Proved (structural, every key set including the adversary's).** Distinct keys separate within the
  radix depth `D` (`chain_separates_within_radix`) and within `G + D − 1` for the shipped probe with
  its linear prefix (`linear_prefix_chain_separates`); the address map is *exactly balanced* — every
  lane is the home of exactly `2^(W−b)` coordinates (`home_balanced`), so no lane is structurally
  favoured and occupancy is a property of the population, never of the address; a **terminating**
  probe costs at most `D` steps (`plen_addr_le_radix`), `G + D − 1` shipped (`plen_seqLin_le_radix`);
  and forcing a probe of length `L` costs the adversary at least as many *distinct* occupied lanes
  (`plen_image_card_le`). The pile on a whole window set is `2^(W−b·F)` keys (`window_pile_card`),
  256 at the shipped 14-bit four-window geometry (`shipped_pile_card`).
* **Proved here, and it is the hole.** Universal probe-length immunity is **FALSE**. `≤ D` crafted
  keys, none of them the victim and each homing where `PHI⁻¹` says, occupy *every* lane the victim's
  descent will try (`descent_can_be_wholly_blocked`; `shipped_probe_can_be_wholly_blocked` for the
  probe as shipped). Termination itself is what the craft removes, not a short walk, so the only
  worst-case aggregate left is `Θ(n²)` (`build_bounded_by_n_squared`), the same ceiling a flooded
  `hash & mask` pays (`flood_is_quadratic`). This section is carried over from the upstream corpus
  (`../sanguine-ref` `Substrate/Algebra/Security/AggregationBound.lean`); the apeiron
  `Closures/Engineering/RadixDescent.lean` cites those declarations but does not carry them.
* **Trust boundary — the seed story, stated.** The multipliers are fixed source constants, not
  per-instance secrets: `PHI = 0x9E37_79B9_7F4A_7C15` (`src/coord.rs:59`, the fold, the ported
  `hashtrinity::Table` `home`/`descend`, `dense`, and `table.TAG_SPREAD` at `src/table.rs:93`) and
  `HOME_SPREAD = 0x51_7C_C1_B7_27_22_0A_95` (`src/table/state.rs:39`, `Region::home`). Both are odd,
  hence units; the multiply-shift is a bijection on the word. Because the constant is fixed and the
  source is public, the aiming attack above is available to any adversary who reads it, and the
  bounds are therefore claims about the mixing's *structure*, never about secrecy. Grade of "the
  constant is secret": **not claimed, and false** — no random seed exists on this face.
* **Where the extraction does not reach.** `addr`/`homeSlot`/`seqLin`/`plen` are the hand-transcribed
  model of `Table::home`/`Table::descend` (`src/hashtrinity/mod.rs`); no refinement theorem binds them
  to the Rust. The generated body `table.Region.home` (`proof/ApeironWord.lean`, from
  `src/table/state.rs`) reads the same multiplier this model does — `table.HOME_SPREAD` equals
  `homeSpread` (§0b) — and the generated `table.Region.home` spread arithmetic is the model's
  `homeSlot` at `bits = b`, `bound = 2^b` (§0b). What stays open is the input plumbing: `Region::home`
  folds `coord_of place` (not the key) and casts the top window through `Usize`, and the extraction
  carries no `descend` body at all, so `seqLin`/`plen` have nothing to bind to. The *link* from the
  model to the generated body is therefore partial: constants and slot arithmetic discharged, input
  and probe blocked, named at the end.

No `sorry`, no `axiom`, no `maxHeartbeats`: every headline carries an in-module `#print axioms` row.
-/

namespace Apeiron
namespace Security
namespace HashDos

open Sanguine.RadixDescent
open Finset

set_option linter.unusedVariables false

-- ═══════════════════════════════════════════════════════════════════
-- §0 · THE SHIPPED MULTIPLIERS — fixed constants, not a seed
-- ═══════════════════════════════════════════════════════════════════

/-- `crate::coord::PHI` (`src/coord.rs:59`): the fold / `hashtrinity::Table` home / `Dense` /
`table.TAG_SPREAD` multiplier. Odd, so the multiply-shift is a bijection on `ZMod (2^64)`. -/
def phiSpread : ℕ := 0x9E37_79B9_7F4A_7C15

/-- `HOME_SPREAD` (`src/table/state.rs:39`), the apeiron `Region::home` multiply (fxhash 0.2.1's
`SEED64`). Odd. Distinct from `phiSpread`, which carries the tag fingerprint. -/
def homeSpread : ℕ := 0x51_7C_C1_B7_27_22_0A_95

theorem phiSpread_odd : Odd phiSpread := by decide

theorem homeSpread_odd : Odd homeSpread := by decide

/-- The two shipped multipliers as units of `ZMod (2^64)` — the fold is a bijection. -/
def phiUnit : (ZMod (2 ^ 64))ˣ := ZMod.unitOfCoprime phiSpread (by decide)

/-- The `Region::home` multiplier as a unit of `ZMod (2^64)`. -/
def homeSpreadUnit : (ZMod (2 ^ 64))ˣ := ZMod.unitOfCoprime homeSpread (by decide)

/-- **The multipliers are fixed constants.** Both are literal `def`s equal to the source words; there
is no constructor argument, no environment read, and no seed. This is the whole of the "keyed?"
answer: the face is **unkeyed**, so an adversary who knows the source knows the multiplier. -/
theorem shipped_multipliers_are_fixed :
    phiSpread = 0x9E37_79B9_7F4A_7C15 ∧ homeSpread = 0x51_7C_C1_B7_27_22_0A_95 :=
  ⟨rfl, rfl⟩

-- ═══════════════════════════════════════════════════════════════════
-- §0b · THE EXECUTED CONSTANTS AND THE EXECUTED SPREAD ARITHMETIC
-- ═══════════════════════════════════════════════════════════════════
-- Discharged from the generated body, not from a reading of the Rust.
--
-- The generated addressing carries the multipliers as its own definitions: `Region::home`'s spread
-- branch multiplies by `table.HOME_SPREAD` (`src/table/state.rs:39`, body `proof/ApeironWord.lean`,
-- `table.Region.home`), and `table.tag` multiplies by `table.TAG_SPREAD := coord.PHI`
-- (`src/table.rs:95`). The two constants are §0's literals. What closes here is the transcription's
-- CONSTANT and SLOT-ARITHMETIC hypotheses: the model's `homeSlot` is the body's
-- `((c * HOME_SPREAD) >>> (64 - bits)) &&& mask` value at `bits = b` and `bound = 2^b`. What does
-- NOT close here is the input plumbing — `Region::home` folds `coord_of place`, not the key, and
-- casts the top window through `Usize` — which the model link still owes (§6(a)).

section ExecutedConstants

open Aeneas Aeneas.Std

/-- The generated `table.HOME_SPREAD` constant is the model's `homeSpread` — the multiplier the
    `Region::home` spread branch reads. -/
theorem generated_HOME_SPREAD_val : (apeiron_word.table.HOME_SPREAD : U64).val = homeSpread := by
  rw [apeiron_word.table.HOME_SPREAD]; rfl

/-- The generated `table.TAG_SPREAD` constant is `coord.PHI`, the model's `phiSpread` — the
    fingerprint multiplier `table.tag` reads. -/
theorem generated_TAG_SPREAD_val : (apeiron_word.table.TAG_SPREAD : U64).val = phiSpread := by
  rw [apeiron_word.table.TAG_SPREAD, apeiron_word.coord.PHI]; rfl

/-- The model's `homeSpreadUnit`, as an element of `ZMod (2^64)`, is the generated multiplier. -/
theorem homeSpreadUnit_val : ((homeSpreadUnit : ZMod (2 ^ 64))).val = homeSpread := by
  rw [homeSpreadUnit, ZMod.coe_unitOfCoprime, ZMod.val_natCast_of_lt (by decide)]

/-- **The executed spread arithmetic is the model's `homeSlot`.** At slot count `2^b` (the body's
    `mask = bound - 1 = 2^b - 1`) and window width `b`, the value the generated spread branch
    computes from a coordinate `c` — `((c * HOME_SPREAD) >>> (64 - b)) &&& mask` — is exactly the
    model's top-window address `homeSlot b homeSpreadUnit c`, over the generated constant. The `&&&`
    drops out because the top `b` bits are already `< 2^b`; the model's `/ 2^(64-b)` is the body's
    `>>> (64-b)`; `homeSpreadUnit`'s value is `table.HOME_SPREAD`'s. -/
theorem region_home_spread_arithmetic (c : U64) (b : ℕ) (hb : 0 < b) (hb64 : b ≤ 64) :
    ((c.val * (apeiron_word.table.HOME_SPREAD : U64).val) % 2 ^ 64 / 2 ^ (64 - b))
        &&& (2 ^ b - 1)
      = homeSlot b homeSpreadUnit (c.val : ZMod (2 ^ 64)) := by
  have hc : c.val < 2 ^ 64 := by
    simpa [UScalar.size, UScalarTy.numBits] using c.hBounds
  have hlt : (c.val * (apeiron_word.table.HOME_SPREAD : U64).val) % 2 ^ 64 / 2 ^ (64 - b)
      < 2 ^ b := by
    rw [Nat.div_lt_iff_lt_mul (by positivity)]
    calc (c.val * (apeiron_word.table.HOME_SPREAD : U64).val) % 2 ^ 64 < 2 ^ 64 :=
          Nat.mod_lt _ (by positivity)
      _ = 2 ^ b * 2 ^ (64 - b) := by rw [← pow_add]; congr 1; omega
  rw [Nat.and_two_pow_sub_one_of_lt_two_pow hlt, generated_HOME_SPREAD_val]
  unfold homeSlot
  rw [ZMod.val_mul, homeSpreadUnit_val, ZMod.val_natCast_of_lt hc]

end ExecutedConstants

-- ═══════════════════════════════════════════════════════════════════
-- §1 · THE MEASURES — lane occupancy, probe length, build cost
-- ═══════════════════════════════════════════════════════════════════

/-- **The build cost** — the total probe work of a key stream (build / insert / GROUP BY are each
this). -/
def buildCost (probes : List ℕ) : ℕ := probes.sum

/-- Bounded per-probe elements give a bounded sum (restated locally, axiom-free). -/
theorem build_sum_le_length_mul (l : List ℕ) (P : ℕ) (hle : ∀ x ∈ l, x ≤ P) :
    l.sum ≤ l.length * P := by
  induction l with
  | nil => simp
  | cons a t ih =>
    have ha : a ≤ P := hle a (List.mem_cons_self ..)
    have ht : t.sum ≤ t.length * P := ih (fun x hx => hle x (List.mem_cons_of_mem a hx))
    have hsm : (t.length + 1) * P = t.length * P + P := Nat.succ_mul _ _
    calc (a :: t).sum = a + t.sum := List.sum_cons
      _ ≤ P + t.length * P := Nat.add_le_add ha ht
      _ = (t.length + 1) * P := by rw [hsm]; exact Nat.add_comm _ _
      _ = (a :: t).length * P := by rw [List.length_cons]

/-- **Bounded per-probe ⇒ LINEAR build.** If every key's probe is bounded by `P`, the aggregate is
`≤ n·P` — the positive half. `p ≤ P` is a HYPOTHESIS here and stays one: §3 refutes its worst-case
discharge. -/
theorem build_linear_of_bounded_probes (probes : List ℕ) (P : ℕ)
    (h : ∀ p ∈ probes, p ≤ P) : buildCost probes ≤ probes.length * P :=
  build_sum_le_length_mul probes P h

-- ═══════════════════════════════════════════════════════════════════
-- §2 · THE STRUCTURAL BOUNDS, AT THE SHIPPED GEOMETRY (W = 64)
-- ═══════════════════════════════════════════════════════════════════

/-- **Worked instance, clean window: no two distinct keys alias.** On the 64-bit word split into four
16-bit windows, distinct coordinates separate at some window `< 4` — a constant fixed by the word,
independent of the key count, and holding for every key set including an adversarial one. (The
shipped table's window width is the runtime `bits`; §2's `shipped_seqLin_plen_bound` and
`shipped_window_pile` use the shipped `14`.) -/
theorem shipped_separation (s : (ZMod (2 ^ 64))ˣ) {k₁ k₂ : ZMod (2 ^ 64)} (hk : k₁ ≠ k₂) :
    ∃ j < 4, addr 16 j s k₁ ≠ addr 16 j s k₂ :=
  chain_separates_within_radix (b := 16) (D := 4) (W := 64) (by norm_num) (by norm_num) s hk

/-- **No hot lane — the keys-per-address fibre is exactly `2^48`.** Every one of the `2^16` lanes is
the home of exactly the same number of coordinates, so the fold distributes the coordinate space with
no structurally favoured lane: collisions come only from how many keys are present (the load), never
from a lane an adversary can bias. This is the "many keys on one address" cap, and it holds for every
odd multiplier. -/
theorem shipped_lane_balanced (s : (ZMod (2 ^ 64))ˣ) {lane : ℕ} (hl : lane < 2 ^ 16) :
    (univ.filter (fun c : ZMod (2 ^ 64) => homeSlot 16 s c = lane)).card = 2 ^ (64 - 16) :=
  home_balanced (b := 16) (W := 64) (by norm_num) s hl

/-- **The shipped probe bound, shipped numbers.** With 14-bit windows (`D = 5` covers the word,
`14 · 5 = 70 ≥ 64`) and the shipped `LINEAR_G = 8`, a descent that finds a free slot at all costs at
most `8 + 5 − 1 = 12` group steps. This is `plen_seqLin_le_radix` at the shipped geometry; it bounds
the walk GIVEN it terminates. §3 shows termination can be removed. -/
theorem shipped_seqLin_plen_bound (s : (ZMod (2 ^ 64))ˣ) (k : ZMod (2 ^ 64)) (O : Finset ℕ)
    (h : ∃ j, seqLin 14 8 ((k * (s : ZMod (2 ^ 64))).val) j ∉ O) :
    plen (fun j => seqLin 14 8 ((k * (s : ZMod (2 ^ 64))).val) j) O h ≤ 8 + 5 - 1 :=
  plen_seqLin_le_radix (b := 14) (G := 8) (D := 5) (W := 64)
    (by norm_num) (by norm_num) s k O h

/-- **The shipped pile is 256.** At the 14-bit four-window geometry exactly `2^(64−56) = 256`
coordinates agree with any key on its entire consumed window set — bounded by the word, never by the
key count. -/
theorem shipped_window_pile (s : (ZMod (2 ^ 64))ˣ) (k : ZMod (2 ^ 64)) :
    (univ.filter (fun c : ZMod (2 ^ 64) => homeSlot 56 s c = homeSlot 56 s k)).card = 256 :=
  shipped_pile_card s k

-- ═══════════════════════════════════════════════════════════════════
-- §3 · THE HOLE — the descent can be wholly blocked (carried over)
-- ═══════════════════════════════════════════════════════════════════
-- `chain_separates_within_radix` bounds a PAIR: two distinct keys differ in some digit below the
-- radix depth. It forbids no SINGLE collision, and a probe length is a function of the OCCUPIED SET.
-- So for every step of a key's descent there is a distinct key colliding at exactly that step, and
-- `D` aimed keys occupy the victim's whole probe-address set — while the separation bound holds of
-- every pair involved. The fold is a bijection, so `PHI⁻¹` converts any named lane into a key that
-- homes there: no search, no hardness assumption, only the public constant.

/-- **A distinct key colliding at a chosen step.** Flip one bit of the fold in a different digit
window: a different key whose step-`j` address is unchanged. -/
theorem exists_collision_at_step {b D W : ℕ} (hb : 0 < b) (hW : W = b * D)
    (s : (ZMod (2 ^ W))ˣ) (k : ZMod (2 ^ W)) {j j' : ℕ} (hj' : j' < D) (hjj : j ≠ j') :
    ∃ k' : ZMod (2 ^ W), k' ≠ k ∧ addr b j s k' = addr b j s k := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  set p := (k * (s : ZMod (2 ^ W))).val with hp
  have hplt : p < 2 ^ W := ZMod.val_lt _
  have hbj : b * j' < W := by rw [hW]; exact mul_lt_mul_of_pos_left hj' hb
  have hpow : (2 : ℕ) ^ (b * j') < 2 ^ W := Nat.pow_lt_pow_right (by norm_num) hbj
  set q := p ^^^ 2 ^ (b * j') with hq
  have hqlt : q < 2 ^ W := Nat.xor_lt_two_pow hplt hpow
  have hfold : (((q : ZMod (2 ^ W)) * (↑s⁻¹ : ZMod (2 ^ W))) * (s : ZMod (2 ^ W))).val = q := by
    rw [mul_assoc, ← Units.val_mul, inv_mul_cancel, Units.val_one, mul_one,
        ZMod.val_natCast_of_lt hqlt]
  have hqp : q ≠ p := by
    intro h
    have hbit : q.testBit (b * j') = p.testBit (b * j') := by rw [h]
    rw [hq, Nat.testBit_xor, Nat.testBit_two_pow_self] at hbit
    cases hpb : p.testBit (b * j') <;> rw [hpb] at hbit <;> simp at hbit
  refine ⟨(q : ZMod (2 ^ W)) * (↑s⁻¹ : ZMod (2 ^ W)), ?_, ?_⟩
  · intro hk'
    apply hqp
    rw [← hfold, hk', ← hp]
  · show digit b j _ = digit b j _
    rw [hfold, ← hp]
    apply Nat.eq_of_testBit_eq
    intro r
    rcases lt_or_ge r b with hr | hr
    · rw [digit_testBit b j q r hr, digit_testBit b j p r hr, hq, Nat.testBit_xor]
      have hne : b * j' ≠ b * j + r := by
        intro h
        apply hjj
        have h1 : (b * j + r) / b = j := by
          rw [Nat.mul_add_div hb, Nat.div_eq_of_lt hr, Nat.add_zero]
        rw [← h, Nat.mul_div_cancel_left j' hb] at h1
        exact h1.symm
      rw [Nat.testBit_two_pow_of_ne hne]
      simp
    · have h1 : digit b j q < 2 ^ b := digit_lt b j q hb
      have h2 : digit b j p < 2 ^ b := digit_lt b j p hb
      have hpw : (2 : ℕ) ^ b ≤ 2 ^ r := Nat.pow_le_pow_right (by norm_num) hr
      rw [Nat.testBit_eq_false_of_lt (lt_of_lt_of_le h1 hpw),
          Nat.testBit_eq_false_of_lt (lt_of_lt_of_le h2 hpw)]

/-- **Separation does NOT bound the probe.** At EVERY step of `k`'s descent there is a key `k' ≠ k`
landing on `k`'s lane at that step, and `k'` still separates from `k` within the radix depth — the
two facts are simultaneous, which is exactly why the pair bound does not give a set bound. -/
theorem separation_does_not_bound_probe {b D W : ℕ} (hb : 0 < b) (hD : 2 ≤ D) (hW : W = b * D)
    (s : (ZMod (2 ^ W))ˣ) (k : ZMod (2 ^ W)) (j : ℕ) :
    ∃ k' : ZMod (2 ^ W), k' ≠ k
      ∧ addr b j s k' = addr b j s k
      ∧ ∃ i < D, addr b i s k' ≠ addr b i s k := by
  have hstep : ∃ k' : ZMod (2 ^ W), k' ≠ k ∧ addr b j s k' = addr b j s k := by
    by_cases h0 : j = 0
    · exact exists_collision_at_step hb hW s k (j' := 1) (by omega) (by omega)
    · exact exists_collision_at_step hb hW s k (j' := 0) (by omega) (by omega)
  obtain ⟨k', hne, hcol⟩ := hstep
  exact ⟨k', hne, hcol, chain_separates_within_radix hb hW s hne⟩

/-- **A key that homes onto a chosen lane, and is not the victim.** The fold is a unit, so `q · s⁻¹`
is the key whose fold is `q`; two candidates congruent to the named lane mod `2^b` differ, so one
avoids the victim's own fold. No search — the inverse is arithmetic. -/
theorem exists_key_homing_at {b D W : ℕ} (hb : 0 < b) (hD : 2 ≤ D) (hW : W = b * D)
    (s : (ZMod (2 ^ W))ˣ) (k : ZMod (2 ^ W)) {lane : ℕ} (hs : lane < 2 ^ b) :
    ∃ a : ZMod (2 ^ W), a ≠ k ∧ addr b 0 s a = lane := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  set p := (k * (s : ZMod (2 ^ W))).val with hp
  have h2 : b * 2 ≤ b * D := Nat.mul_le_mul_left b hD
  have hbW : b + 1 ≤ W := hW ▸ le_trans (by omega : b + 1 ≤ b * 2) h2
  have h2W : (2 : ℕ) ^ b * 2 ≤ 2 ^ W := by
    calc (2 : ℕ) ^ b * 2 = 2 ^ (b + 1) := by ring
      _ ≤ 2 ^ W := Nat.pow_le_pow_right (by norm_num) hbW
  have hfold : ∀ q : ℕ, q < 2 ^ W →
      (((q : ZMod (2 ^ W)) * (↑s⁻¹ : ZMod (2 ^ W))) * (s : ZMod (2 ^ W))).val = q := by
    intro q hq
    rw [mul_assoc, ← Units.val_mul, inv_mul_cancel, Units.val_one, mul_one,
        ZMod.val_natCast_of_lt hq]
  obtain ⟨q, hqlt, hqmod, hqp⟩ : ∃ q, q < 2 ^ W ∧ q % 2 ^ b = lane ∧ q ≠ p := by
    have hpos : 0 < (2 : ℕ) ^ b := by positivity
    by_cases h : lane = p
    · exact ⟨lane + 2 ^ b, by omega, by rw [Nat.add_mod_right, Nat.mod_eq_of_lt hs], by omega⟩
    · exact ⟨lane, by omega, Nat.mod_eq_of_lt hs, h⟩
  refine ⟨(q : ZMod (2 ^ W)) * (↑s⁻¹ : ZMod (2 ^ W)), ?_, ?_⟩
  · intro hk
    apply hqp
    rw [← hfold q hqlt, hk, ← hp]
  · show digit b 0 _ = lane
    rw [hfold q hqlt]
    simpa [digit] using hqmod

/-- **Any probe prefix is fillable — the general shape of the defect.** For ANY sequence of `L` lane
addresses and any victim, there are `L` keys, none of them the victim, whose own home addresses are
exactly that sequence. Stated over `seq`, so no re-choice of windows repairs it. -/
theorem crafted_keys_fill_any_probe_prefix {b D W : ℕ} (hb : 0 < b) (hD : 2 ≤ D)
    (hW : W = b * D) (s : (ZMod (2 ^ W))ˣ) (k : ZMod (2 ^ W)) (seq : ℕ → ℕ)
    (hseq : ∀ j, seq j < 2 ^ b) (L : ℕ) :
    ∃ adv : ℕ → ZMod (2 ^ W),
      (∀ j, adv j ≠ k)
      ∧ (∀ j, addr b 0 s (adv j) = seq j)
      ∧ (∀ j < L, seq j ∈ (range L).image (fun i => addr b 0 s (adv i)))
      ∧ ((range L).image (fun i => addr b 0 s (adv i))).card ≤ L := by
  choose adv h1 h2 using fun j : ℕ => exists_key_homing_at hb hD hW s k (hseq j)
  refine ⟨adv, h1, h2, fun j hj => Finset.mem_image.mpr ⟨j, Finset.mem_range.mpr hj, h2 j⟩, ?_⟩
  exact le_trans Finset.card_image_le (by simp)

/-- **THE HOLE — the descent can be wholly blocked.** An adversary key set of `≤ D` keys, none equal
to the victim, whose home lanes contain EVERY one of the victim's `D` probe addresses. The `Open:`
that would give `plen ≤ D` unconditionally reads `∀ k, ∃ j < D, addr b j s k ∉ O`; this `O` refutes
it. It costs one key per step and needs nothing but the public multiplier. At `W = 64, b = 16, D = 4`
that is 4 lanes of 65536. -/
theorem descent_can_be_wholly_blocked {b D W : ℕ} (hb : 0 < b) (hD : 2 ≤ D) (hW : W = b * D)
    (s : (ZMod (2 ^ W))ˣ) (k : ZMod (2 ^ W)) :
    ∃ adv : ℕ → ZMod (2 ^ W),
      (∀ j, adv j ≠ k)
      ∧ ((range D).image (fun i => addr b 0 s (adv i))).card ≤ D
      ∧ ¬ ∃ j < D, addr b j s k ∉ (range D).image (fun i => addr b 0 s (adv i)) := by
  obtain ⟨adv, h1, _, h3, h4⟩ :=
    crafted_keys_fill_any_probe_prefix hb hD hW s k (fun j => addr b j s k)
      (fun j => digit_lt b j _ hb) D
  refine ⟨adv, h1, h4, ?_⟩
  rintro ⟨j, hj, hjO⟩
  exact hjO (h3 j hj)

/-- **The same for the probe AS SHIPPED.** `seqLin` — home, then the `G−1` linear prefix, then the
fresh windows — separates within `G + D − 1`, and that prefix is filled by `G + D − 1` crafted keys
the same way. The linear prefix is not a mitigation; it is `G − 1` more lanes to buy. -/
theorem shipped_probe_can_be_wholly_blocked {b D G W : ℕ} (hb : 0 < b) (hG : 1 ≤ G) (hD : 2 ≤ D)
    (hW : W = b * D) (s : (ZMod (2 ^ W))ˣ) (k : ZMod (2 ^ W)) :
    ∃ adv : ℕ → ZMod (2 ^ W),
      (∀ j, adv j ≠ k)
      ∧ (∀ j, seqLin b G ((adv j * (s : ZMod (2 ^ W))).val) 0 = addr b 0 s (adv j))
      ∧ ((range (G + D - 1)).image (fun i => addr b 0 s (adv i))).card ≤ G + D - 1
      ∧ ¬ ∃ j < G + D - 1, seqLin b G ((k * (s : ZMod (2 ^ W))).val) j
            ∉ (range (G + D - 1)).image (fun i => addr b 0 s (adv i)) := by
  obtain ⟨adv, h1, _, h3, h4⟩ :=
    crafted_keys_fill_any_probe_prefix hb hD hW s k
      (fun j => seqLin b G ((k * (s : ZMod (2 ^ W))).val) j) (fun j => seqLin_lt hb j) (G + D - 1)
  refine ⟨adv, h1, fun j => ?_, h4, ?_⟩
  · show seqLin b G _ 0 = digit b 0 _
    unfold seqLin
    rw [if_pos (by omega : 0 < G), Nat.add_zero, Nat.mod_eq_of_lt (digit_lt b 0 _ hb)]
  · rintro ⟨j, hj, hjO⟩
    exact hjO (h3 j hj)

-- ═══════════════════════════════════════════════════════════════════
-- §4 · WHAT THE ADVERSARY PAYS — the exchange rate and the ceiling
-- ═══════════════════════════════════════════════════════════════════

/-- **Forcing a probe of length `L` costs one occupied lane per DISTINCT address it crosses.** No
hypothesis: the addresses of the first `L` steps all lie in the occupied set, so the adversary owns
at least as many lanes as the probe has distinct addresses. This is what the block above does NOT get
to amplify. -/
theorem probe_prefix_forces_distinct_occupied_slots {α : Type} [DecidableEq α]
    (seq : ℕ → α) (O : Finset α) (h : ∃ j, seq j ∉ O) {L : ℕ}
    (hL : L ≤ plen seq O h) :
    ((range L).image seq).card ≤ O.card := by
  have hsub : range L ⊆ range (plen seq O h) :=
    Finset.range_subset.mpr (fun x hx => Finset.mem_range.mpr (lt_of_lt_of_le hx hL))
  exact le_trans (Finset.card_le_card (Finset.image_subset_image hsub))
    (plen_image_card_le seq O h)

/-- **The same at one lane per step, when the walk does not revisit.** With the probe's addresses
distinct on its own prefix, `L ≤ plen` gives `L ≤ O.card` outright. The `Set.InjOn` hypothesis is
genuine and not discharged for the shipped probe (`addr_not_injOn`); where it fails, the
distinct-address form above is the bound. -/
theorem probe_length_forces_occupied_slots {α : Type} [DecidableEq α]
    (seq : ℕ → α) (O : Finset α) (h : ∃ j, seq j ∉ O)
    (hinj : Set.InjOn seq (range (plen seq O h))) {L : ℕ}
    (hL : L ≤ plen seq O h) : L ≤ O.card :=
  le_trans hL (plen_le_card seq O h hinj)

/-- **The ceiling the exchange rate buys is `n²`** — the same `Θ(n²)` a flooded `hash & mask` pays.
`n` residents occupy at most `n` lanes, so no probe exceeds `n`, and the build is `≤ n·n`. -/
theorem build_bounded_by_n_squared (probes : List ℕ) (n : ℕ)
    (hlen : probes.length ≤ n) (hp : ∀ p ∈ probes, p ≤ n) :
    buildCost probes ≤ n * n :=
  le_trans (build_sum_le_length_mul probes n hp) (Nat.mul_le_mul_right n hlen)

/-- **The `if` discharged, at its only price: TERMINATION.** A build whose every descent finds a free
slot costs `≤ n·D` — linear, no distribution, no load hypothesis. The hypothesis is exactly what
`descent_can_be_wholly_blocked` removes. -/
theorem build_linear_of_terminating_descents {b D W : ℕ} (hW : W ≤ b * D)
    (s : (ZMod (2 ^ W))ˣ) (keys : List (ZMod (2 ^ W))) (O : ZMod (2 ^ W) → Finset ℕ)
    (h : ∀ k, ∃ j, addr b j s k ∉ O k) :
    buildCost (keys.map (fun k => plen (fun j => addr b j s k) (O k) (h k)))
      ≤ keys.length * D := by
  have hbd : ∀ p ∈ keys.map (fun k => plen (fun j => addr b j s k) (O k) (h k)), p ≤ D := by
    intro p hp
    obtain ⟨k, _, rfl⟩ := List.mem_map.mp hp
    exact plen_addr_le_radix hW s k (O k) (h k)
  simpa using build_linear_of_bounded_probes _ D hbd

/-- The flooded `hash & mask` build, as probe lengths. -/
def floodCost (n : ℕ) : ℕ := (List.range n).sum

/-- **The flood is quadratic** — `2·floodCost n = n·(n−1)`. -/
theorem flood_is_quadratic (n : ℕ) : 2 * floodCost n = n * (n - 1) := by
  induction n with
  | zero => rfl
  | succ m ih =>
    have hstep : floodCost (m + 1) = floodCost m + m := by
      unfold floodCost; rw [List.range_succ, List.sum_append]; simp
    rw [hstep, Nat.mul_add, ih, Nat.add_sub_cancel]
    cases m with
    | zero => rfl
    | succ k => simp only [Nat.add_sub_cancel]; ring

/-- Once `n ≥ 2·P + 1` the flood's build already meets the whole `n·P` of a `P`-bounded build. -/
theorem flood_dominates_bounded {n P : ℕ} (hn : 2 * P + 1 ≤ n) :
    n * P ≤ floodCost n := by
  have h2 : 2 * floodCost n = n * (n - 1) := flood_is_quadratic n
  have hle : 2 * P ≤ n - 1 := by omega
  have hmul : n * (2 * P) ≤ n * (n - 1) := Nat.mul_le_mul_left n hle
  have hassoc : n * (2 * P) = 2 * (n * P) := by ring
  omega

-- ═══════════════════════════════════════════════════════════════════
-- §5 · HEADLINES
-- ═══════════════════════════════════════════════════════════════════

/-- **The structural HashDOS bound, for every odd multiplier and every key set.** (i) distinct keys
separate within the radix depth; (ii) every lane's coordinate fibre is exactly `2^(W−b)` — no hot
lane; (iii) a terminating descent costs at most `D` steps; (iv) a forced probe of length `L` crosses
at most `|O|` distinct occupied lanes. All four are `proved` about the address model, with no
distribution, hardness, or secrecy hypothesis. -/
theorem hashdos_structure {b D W : ℕ} (hb : 0 < b) (hD : 1 ≤ D) (hbW : b ≤ W)
    (hW : W = b * D) (s : (ZMod (2 ^ W))ˣ) :
    (∀ k₁ k₂ : ZMod (2 ^ W), k₁ ≠ k₂ → ∃ j < D, addr b j s k₁ ≠ addr b j s k₂) ∧
    (∀ lane < 2 ^ b, (univ.filter (fun c : ZMod (2 ^ W) => homeSlot b s c = lane)).card
        = 2 ^ (W - b)) ∧
    (∀ (k : ZMod (2 ^ W)) (O : Finset ℕ) (h : ∃ j, addr b j s k ∉ O),
        plen (fun j => addr b j s k) O h ≤ D) ∧
    (∀ (α : Type) [DecidableEq α] (seq : ℕ → α) (O : Finset α) (h : ∃ j, seq j ∉ O),
        ((range (plen seq O h)).image seq).card ≤ O.card) :=
  ⟨fun _ _ hk => chain_separates_within_radix hb hW s hk,
   fun _ hl => home_balanced hbW s hl,
   fun k O h => plen_addr_le_radix (le_of_eq hW) s k O h,
   fun _ _ seq O h => plen_image_card_le seq O h⟩

/-- **The honest limit — the invocation of the structural bound is FALSE.** There is an adversarial
occupied set of at most `D` lanes, no key equal to the victim, that contains every lane the victim's
descent will try; so `∀ k, ∃ j < D, addr b j s k ∉ O` — the hypothesis that would make the build
linear — is refutable, and the worst-case aggregate is only `Θ(n²)`. This is the HashDOS
vulnerability stated as a theorem, carried over into the apeiron corpus. -/
theorem worst_case_probe_is_not_bounded {b D W : ℕ} (hb : 0 < b) (hD : 2 ≤ D) (hW : W = b * D)
    (s : (ZMod (2 ^ W))ˣ) (k : ZMod (2 ^ W)) :
    ∃ adv : ℕ → ZMod (2 ^ W),
      (∀ j, adv j ≠ k)
      ∧ ((range D).image (fun i => addr b 0 s (adv i))).card ≤ D
      ∧ ¬ ∃ j < D, addr b j s k ∉ (range D).image (fun i => addr b 0 s (adv i)) :=
  descent_can_be_wholly_blocked hb hD hW s k

/-- **The headline, conjoined, with its exact scope.** For the shipped-style fold (`s` a fixed odd
multiplier, unkeyed): the structural bounds hold over every key set (`hashdos_structure`); and the
universal probe-immunity reading is refuted (`worst_case_probe_is_not_bounded`). Both are `proved`
about the address model; the seed is public (`shipped_multipliers_are_fixed`), so the bound is a
statement about mixing structure, not secrecy. -/
theorem hashdos_invulnerability_and_its_limit {b D W : ℕ} (hb : 0 < b) (hD : 2 ≤ D)
    (hbW : b ≤ W) (hW : W = b * D) (s : (ZMod (2 ^ W))ˣ) :
    (∀ k₁ k₂ : ZMod (2 ^ W), k₁ ≠ k₂ → ∃ j < D, addr b j s k₁ ≠ addr b j s k₂)
    ∧ (∀ lane < 2 ^ b, (univ.filter (fun c : ZMod (2 ^ W) => homeSlot b s c = lane)).card
        = 2 ^ (W - b))
    ∧ (∀ k : ZMod (2 ^ W), ∃ adv : ℕ → ZMod (2 ^ W),
        (∀ j, adv j ≠ k)
        ∧ ((range D).image (fun i => addr b 0 s (adv i))).card ≤ D
        ∧ ¬ ∃ j < D, addr b j s k ∉ (range D).image (fun i => addr b 0 s (adv i))) :=
  ⟨(hashdos_structure hb (by omega) hbW hW s).1,
   (hashdos_structure hb (by omega) hbW hW s).2.1,
   fun k => worst_case_probe_is_not_bounded hb hD hW s k⟩

-- ═══════════════════════════════════════════════════════════════════
-- §6 · THE RESIDUAL — where the extraction does not reach
-- ═══════════════════════════════════════════════════════════════════
-- (a) MODEL LINK, PARTIAL. `addr`/`homeSlot`/`seqLin`/`plen` are a hand transcription of
--     `src/hashtrinity/mod.rs` (`Table::home` at `:602-610`, `Table::descend` at `:626-654`); no
--     Aeneas refinement theorem binds them to the Rust. §0b discharges the CONSTANT and SLOT
--     ARITHMETIC: `table.HOME_SPREAD` equals `homeSpread` and `Region::home`'s spread value is
--     `homeSlot` at `bits = b`, `bound = 2^b`. Still OPEN and precisely blocked:
--       · `homeSlot`'s INPUT: the generated body folds `table.Region.coord_of R V place`
--         (`result.result` — `place >>> SHARD_BITS`), not the key `k` the model folds. Discharging
--         the remainder needs the `Result` do-block inversion of `table.Region.home`
--         (`proof/ApeironWord.lean`, `table.Region.home`), including its `UScalar.cast .Usize` of
--         the top window (platform-`Usize`-width) and its `c < len` / `len = 0` branches.
--       · `seqLin`/`plen`: the extraction carries NO `Table::descend` body, so these have no
--         generated definition to bind to. Blocked on a `descend` extraction, not on a lemma.
--     The `home_lt_bound` fact the earlier note cited lives in `Growable/HomeBound.lean`, which is
--     NOT a `lakefile.lean` root and still inverts `table.Region.home` against `table.TAG_SPREAD`
--     (lines 74/94/113/127) — the body now reads `table.HOME_SPREAD`, so that historical file no
--     longer matches the current extraction (reading, not a run gate; a sibling lane owns it).
--     No generated-body separation, balance or probe-length theorem is proved here or anywhere in tree.
-- (b) CONSTANT DRIFT, CLOSED. The current extraction's `table.Region.home` spread branch multiplies
--     by `table.HOME_SPREAD` (`5871781006564002453#u64`) and `table.tag` by
--     `table.TAG_SPREAD := coord.PHI`; §0b proves both equal the model's `homeSpread`/`phiSpread`.
--     The source/extraction identity is still established only by `proof/regen.sh --check`, which is
--     a regeneration gate, not this file.
-- (c) SHIPPED SWEEP, NOT CARRIED. The shipped `Table::descend` falls back to a terminal sweep past
--     the `D` windows; its termination theorem (`GroupProbe.seqSweep_terminates`) and the aggregate
--     `build_bounded_by_sweep` live only in `../sanguine-ref`, not in this corpus. §4's
--     `build_bounded_by_n_squared` is the ceiling that remains.

end HashDos
end Security
end Apeiron

#print axioms Apeiron.Security.HashDos.shipped_multipliers_are_fixed
#print axioms Apeiron.Security.HashDos.shipped_separation
#print axioms Apeiron.Security.HashDos.shipped_lane_balanced
#print axioms Apeiron.Security.HashDos.shipped_seqLin_plen_bound
#print axioms Apeiron.Security.HashDos.shipped_window_pile
#print axioms Apeiron.Security.HashDos.descent_can_be_wholly_blocked
#print axioms Apeiron.Security.HashDos.shipped_probe_can_be_wholly_blocked
#print axioms Apeiron.Security.HashDos.probe_prefix_forces_distinct_occupied_slots
#print axioms Apeiron.Security.HashDos.build_bounded_by_n_squared
#print axioms Apeiron.Security.HashDos.build_linear_of_terminating_descents
#print axioms Apeiron.Security.HashDos.hashdos_structure
#print axioms Apeiron.Security.HashDos.worst_case_probe_is_not_bounded
#print axioms Apeiron.Security.HashDos.hashdos_invulnerability_and_its_limit
#print axioms Apeiron.Security.HashDos.generated_HOME_SPREAD_val
#print axioms Apeiron.Security.HashDos.generated_TAG_SPREAD_val
#print axioms Apeiron.Security.HashDos.homeSpreadUnit_val
#print axioms Apeiron.Security.HashDos.region_home_spread_arithmetic
