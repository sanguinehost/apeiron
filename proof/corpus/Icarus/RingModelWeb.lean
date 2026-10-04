/-
  `RingModelWeb.lean` — AR-33, an implication web over the ring models.

  Task AR-33 of the algebra-rooting wave, from finding 1 of
  `work/algebra-rooting/catcrypt-analysis.md` (§5, row "5+6"). Owned files: this one and
  `RingModelWeb.md`. No production, pin, extraction-root or API edit; no commit.

  **Port (SOL-12).** Ported verbatim from the staging proof `work/proofs/RingModelWeb.lean`
  (SHA256 `2b2e3d7f…`) into the corpus library as `Icarus.RingModelWeb`. Imports are unchanged and
  resolve in-tree: `WordRefinement`, `Apeiron.Trichotomy.AnyRing` and
  `Apeiron.Trichotomy.ChannelsAnyRing` all exist here, and the sibling port
  `Icarus/AlgebraRotate32.lean` already uses the first and third. **No statement, signature or
  proof body changed**; the only edits are the option header below, the `autoImplicit`/
  `warningAsError` pairs this package already sets globally, and `linter.unusedVariables false`
  for the deliberately-unused hypothesis binders (`_hm0`, `_hn0`, `_hnw`, `hw`), matching
  `Icarus/AlgebraRotate32.lean:44`.

  **The method, and where ours differs.** CatCrypt §2.1 (`:126-140`) builds "a web of formally
  proved implications between alternative formulations of the same security notion" and states the
  purpose: "A bug in any single definition would break at least one implication in the web,
  providing an additional layer of assurance beyond what any individual cross-check could offer"
  (`reference/catcrypt/catcrypt.txt:139-140`). Their members are *security definitions* (IND-CPA,
  IND$, semantic security; SUF-CMA/EUF-CMA; CR/TCR/preimage). Ours are *ring models*: the shared
  definitions are `Apeiron.Trichotomy.AnyRing.{ringMul, ringNormForm}` and
  `Apeiron.Trichotomy.ChannelsAnyRing.ringUpper`, and the members are carriers (`BitVec w` versus
  `ZMod (2 ^ w)`), σ-faces (`ringUpper`'s read table), widths (`w = 64` against `w = 32`), and the
  ops-map candidate formulations. No security notion, hardness assumption, game or ε-bound
  appears here, and none is claimed (`catcrypt-analysis.md` §6, "The web's members").

  What transfers is the *shape*: a closed axiom cone (`#print axioms` rows at the end of this file)
  says every row was proved, and says nothing about whether the shared definitions are the right
  ones. Every obstruction and candidate in `work/algebra-rooting/ops-map.md` is a single-witness,
  single-formulation claim graded `model`, so a wrong witness or a wrong bridge definition passes
  the cone untouched. The web below is arranged so that a defect in a shared definition breaks a
  *composed* lemma rather than one row.

  **The six edges, in order.**

    §1  BitVec ⇄ ZMod agreement. The lift `(x.toNat : ZMod (2 ^ w))` is `bvNat w`; round-trip,
         `toNat mod 2 ^ w` and `ZMod.natCast_mod` discipline, injectivity, and additivity. Every
         later row is stated in `ZMod (2 ^ w)` and is reachable from a `BitVec w` row by the named
         lemmas of this section, not by re-witnessing.
    §2  The σ-face read table as ONE parametric theorem (`face_read`, `face_channel_mul`) over the
         structure `FaceRoot w`, specialised at `faceZero w` (σ = 0, blind), `faceOne w` (σ = 1,
         `z.1 + z.2`) and the elliptic face (σ = −1, no root for `w ≥ 2`).
    §3  The rotation identity as ONE statement in the width `w`. `w = 64` (Keccak) and `w = 32`
         (SHA-2) are instances of `bvNat_rotateLeft_eq`, not two witnesses.
    §4  The ops-map candidate expression `rotExpr w x n`, its exact offset, and its agreement set —
         again one statement in `w`.
    §5  Cross-candidate agreement. Ops-map rows S7 (SHA-2 `rotr`) and K4/K2 (Keccak `rotl`) are
         related by `rotr n = rotl (w − n)` and by the amount swap `n ↦ w − n` in `rotExpr`; the two
         rows' defect functions are therefore one function at two arguments.
    §6  The composed obstructions: no member of the two-slot read family reads a rotation, at every
         width, and the AND/XOR non-affinity witnesses at every width.

  Nothing here is a security, collision-resistance, break or constant-time claim.

  Every shift amount is written as a natural number read into the ring, `((2 ^ n : ℕ) : ZMod
  (2 ^ w))`, the element `Nat.cast_pow` produces; `two_pow_cast_eq` identifies it with the ring
  power `(2 : ZMod (2 ^ w)) ^ n` so the spelling cannot drift. The one `decide` is over the four
  elements of `Fin 4`. No `sorry`, no `axiom`, no `admit`, no `native_decide`, no raised
  `maxRecDepth`, no `maxHeartbeats 0`.

  Grades: `proved` for a named theorem whose `#print axioms` row at the end of this file is inside
  `{propext, Classical.choice, Quot.sound}`. Nothing in the ops map changes grade: rows S7, K2, K4
  move from `model` to having a `proved` parametric statement behind them, and O5 gains a
  parametric statement. The ops-map *candidates* C5/C5b as printed remain **refuted**, which
  §4 proves parametrically.
-/

import WordRefinement
import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

namespace RingModelWeb

open apeiron_word
open Apeiron.Trichotomy.AnyRing

/-! ## §0 The shared definitions this web protects

Everything below is stated over

* `RingCell R` (`AnyRing.lean:98`), `ringMul` (`:109`), `ringNormForm` (`:128`) — one definition each,
* `ChannelsAnyRing.ringUpper` (`ChannelsAnyRing.lean:139`) with its single hypothesis
  `r ^ 2 = σ` (`ringUpper_mul`, `:160`),
* the carriers `BitVec w` (Lean core `Init/Data/BitVec`) and `ZMod (2 ^ w)` (Mathlib `Data/ZMod`).

and over `denoteSigma` (`WordRefinement.lean:35`), which is what fixes a σ for the executing
`Cell<i64>` bodies. -/

/-! ## §1 BitVec ⇄ ZMod agreement

The carrier denotation. It is read two ways below: `bvNat_val` recovers the word's unsigned value,
`bvNat_injective` says no two words share an image, and `bvNat_natCast_mod` is the `ZMod` reduction
discipline (`ZMod.natCast_mod`, `Mathlib/Data/ZMod/Basic.lean:746`). Any theorem proved about one
carrier is a theorem about the other after these three, by name. -/

/-- The `BitVec w`-level lift of a word into the ring model `ZMod (2 ^ w)`: its unsigned value read
into the ring. Width-parametric, so the `BitVec 64` carrier of the ops map's Keccak rows and the
`BitVec 32` carrier of its SHA-2 rows are the same definition at two arguments. -/
def bvNat (w : ℕ) (x : BitVec w) : ZMod (2 ^ w) := (x.toNat : ZMod (2 ^ w))

/-- The lift is the unsigned value read into the ring. -/
@[simp] theorem bvNat_def (w : ℕ) (x : BitVec w) : bvNat w x = (x.toNat : ZMod (2 ^ w)) := rfl

set_option linter.unnecessarySimpa false in
/-- `2 ^ w` is `0` in its own modulus: the fact that makes every `ZMod` statement at width `w` a
statement modulo `2 ^ w`. -/
theorem two_pow_natCast_zero (w : ℕ) : ((2 ^ w : ℕ) : ZMod (2 ^ w)) = 0 := by
  have h := ZMod.natCast_mod (2 ^ w) (2 ^ w)
  rw [Nat.mod_self] at h
  simpa using h.symm

/-- **Round-trip.** The lift is faithful in both directions: its canonical representative is the
word's unsigned value. This is the named lemma that lets a theorem about `ZMod (2 ^ w)` values be
read back as a statement about `BitVec w` words. -/
theorem bvNat_val (w : ℕ) (x : BitVec w) : (bvNat w x).val = x.toNat := by
  rw [bvNat_def, ZMod.val_natCast, Nat.mod_eq_of_lt x.isLt]

/-- The `toNat mod 2 ^ w` discipline: reading a residue class or its representative gives the same
element of `ZMod (2 ^ w)`. -/
theorem bvNat_natCast_mod (w k : ℕ) :
    ((k % 2 ^ w : ℕ) : ZMod (2 ^ w)) = ((k : ℕ) : ZMod (2 ^ w)) :=
  ZMod.natCast_mod k (2 ^ w)

/-- The same discipline with the representative returned: the lift of a residue class has `k`'s own
remainder as its representative. -/
theorem bvNat_val_natCast (w k : ℕ) : ((k : ZMod (2 ^ w))).val = k % 2 ^ w := ZMod.val_natCast _ _

/-- Ground evaluation: the lift of `BitVec.ofNat w k` is `k` in the ring, for every `k`, with no
bound on `k`. A row witnessed at a numeral therefore transfers to the ring by this lemma alone. -/
theorem bvNat_ofNat (w k : ℕ) : bvNat w (BitVec.ofNat w k) = ((k : ℕ) : ZMod (2 ^ w)) := by
  rw [bvNat_def, BitVec.toNat_ofNat, ZMod.natCast_mod]

/-- **Injectivity.** Equal denotations mean equal words: the carrier identification has no kernel. -/
theorem bvNat_injective (w : ℕ) {x y : BitVec w} (h : bvNat w x = bvNat w y) : x = y := by
  refine BitVec.eq_of_toNat_eq ?_
  have h1 := congrArg ZMod.val h
  simpa only [bvNat_val] using h1

/-- The injectivity statement in the `iff` form a transfer between carriers is normally written. -/
theorem bvNat_eq_iff (w : ℕ) (x y : BitVec w) : bvNat w x = bvNat w y ↔ x = y :=
  ⟨bvNat_injective w, fun h => by rw [h]⟩

/-- **Addition agrees across the two carriers.** The `BitVec` addition is the `2 ^ w`-modular one
(`BitVec.toNat_add`, `Init/Data/BitVec/Lemmas.lean:3521`), so its lift is the ring addition of
`ZMod (2 ^ w)`. This is the bridge that lets ops-map row S1 (`wrapping_add`, `F2`) be settled by a
`BitVec` lemma and used in the ring, and it holds at every width. -/
theorem bvNat_add (w : ℕ) (x y : BitVec w) :
    bvNat w (x + y) = bvNat w x + bvNat w y := by
  simp only [bvNat_def, BitVec.toNat_add, ZMod.natCast_mod, Nat.cast_add]

/-- Subtraction agrees the same way. -/
theorem bvNat_sub (w : ℕ) (x y : BitVec w) :
    bvNat w (x - y) = bvNat w x - bvNat w y := by
  simp only [bvNat_def, BitVec.toNat_sub, ZMod.natCast_mod]
  rw [Nat.cast_add]
  have hsub : ((2 ^ w - y.toNat : ℕ) : ZMod (2 ^ w))
      = ((2 ^ w : ℕ) : ZMod (2 ^ w)) - ((y.toNat : ℕ) : ZMod (2 ^ w)) := by
    rw [Nat.cast_sub (Nat.le_of_lt y.isLt), two_pow_natCast_zero]
  rw [hsub, two_pow_natCast_zero, zero_sub, add_comm, sub_eq_add_neg]

/-- The two spellings of the shift factor are the same element at every width. Removing this
discrepancy is what lets a row be re-stated with `(2 : ZMod (2 ^ w)) ^ n` or with `((2 ^ n : ℕ) :
ZMod (2 ^ w))` without changing what it says. -/
theorem two_pow_cast_eq (w n : ℕ) : (2 : ZMod (2 ^ w)) ^ n = ((2 ^ n : ℕ) : ZMod (2 ^ w)) := by
  rw [Nat.cast_pow, Nat.cast_ofNat]

/-- `2 ^ m` is non-zero in `ZMod (2 ^ w)` whenever `0 < m < w`. The bound on `m` is exactly the
`n < w` hypothesis of every rotation row, so this is the non-vanishing obligation for a whole
family at once rather than per amount. -/
theorem two_pow_natCast_ne_zero (w m : ℕ) (_hm0 : 0 < m) (hmw : m < w) :
    ((2 ^ m : ℕ) : ZMod (2 ^ w)) ≠ 0 := by
  intro h
  have hlt : 2 ^ m < 2 ^ w := Nat.pow_lt_pow_right (by omega) (by omega)
  rw [← ZMod.val_eq_zero, bvNat_val_natCast, Nat.mod_eq_of_lt hlt] at h
  exact absurd h.symm (Nat.ne_of_lt (Nat.two_pow_pos m))
/-! ## §2 The σ-face read table as one parametric theorem

`ChannelsAnyRing.ringUpper r z = z.1 + r * z.2` (`ChannelsAnyRing.lean:139`) carries exactly one
hypothesis, `r ^ 2 = σ`, and that is the hypothesis of `ringUpper_mul` (`:160`). Everything in this
section is derived from those two declarations, at an arbitrary width. -/

/-- The read of a point along the direction `(r, −1)`, over an arbitrary commutative ring: the body
of `ChannelsAnyRing.ringUpper`. -/
theorem ringUpper_def {R : Type*} [CommRing R] (r : R) (z : RingCell R) :
    ringUpper r z = z.1 + r * z.2 := rfl

/-- **σ = 0 is blind.** The parabolic read is the first slot alone, over every commutative ring: the
second slot never reaches the read. This is the ops map's "σ = 0 blind" clause (row A5). -/
theorem ringUpper_zero_blind {R : Type*} [CommRing R] (z : RingCell R) :
    ringUpper (0 : R) z = z.1 := by
  simp only [ringUpper_def, zero_mul, add_zero]

/-- **σ = 1 reads the sum.** The hyperbolic read is `z.1 + z.2`. -/
theorem ringUpper_one_sum {R : Type*} [CommRing R] (z : RingCell R) :
    ringUpper (1 : R) z = z.1 + z.2 := by
  simp only [ringUpper_def, one_mul]

/-- The read is additive in the point, for every root and every commutative ring. Together with
`ringUpper_zero_blind` this says the parabolic blindness is a statement about the *shared*
`ringUpper`, not about a face-specific formula. -/
theorem ringUpper_add {R : Type*} [CommRing R] (r : R) (z w : RingCell R) :
    ringUpper r (z + w) = ringUpper r z + ringUpper r w := by
  simp only [ringUpper_def, Prod.fst_add, Prod.snd_add, add_mul]
  ring

/-- A σ together with a root of it: precisely the data `ringUpper` and `ringUpper_mul` need. -/
structure FaceRoot (w : ℕ) where
  /-- the parameter σ, read in `ZMod (2 ^ w)` -/
  σ : ZMod (2 ^ w)
  /-- the channel root r -/
  r : ZMod (2 ^ w)
  /-- the root squares to the parameter: the single hypothesis of `ringUpper_mul` -/
  sq : r ^ 2 = σ

/-- The parabolic face: σ = 0, root r = 0. -/
def faceZero (w : ℕ) : FaceRoot w where
  σ := 0
  r := 0
  sq := by simp

/-- The hyperbolic face: σ = 1, root r = 1. -/
def faceOne (w : ℕ) : FaceRoot w where
  σ := 1
  r := 1
  sq := by simp

/-- **The read table, as ONE parametric theorem.** For every width, every σ that has a root, and
every point, the read is `z.1 + r * z.2`. Each face-specific read below is this equation at a named
face, so a wrong face-specific statement does not get a row of its own. -/
theorem face_read (w : ℕ) (F : FaceRoot w) (z : RingCell (ZMod (2 ^ w))) :
    ringUpper F.r z = z.1 + F.r * z.2 :=
  ringUpper_def _ _

/-- **Channel multiplicativity, as ONE parametric theorem.** The read of a product is the product of
the reads, for every face with a root, at every width. This is `ChannelsAnyRing.ringUpper_mul` at
the hypothesis `F.sq`. -/
theorem face_channel_mul (w : ℕ) (F : FaceRoot w) (z w' : RingCell (ZMod (2 ^ w))) :
    ringUpper F.r (ringMul F.σ z w') = ringUpper F.r z * ringUpper F.r w' :=
  ringUpper_mul F.sq z w'

/-- **Face σ = 0, blind.** The read of a product at the parabolic face is the product of the first
slots: the second slot is invisible on both sides. -/
theorem face_parabolic_read_mul (w : ℕ) (z w' : RingCell (ZMod (2 ^ w))) :
    ringUpper (faceZero w).r (ringMul (faceZero w).σ z w') = z.1 * w'.1 := by
  rw [face_channel_mul, show (faceZero w).r = 0 from rfl, ringUpper_zero_blind,
    ringUpper_zero_blind]

/-- **Face σ = 1, the sum read.** The read of a product at the hyperbolic face is the product of
the sums. This is the instance ops-map rows S7, K2 and K4 need. -/
theorem face_hyperbolic_read_mul (w : ℕ) (z w' : RingCell (ZMod (2 ^ w))) :
    ringUpper (faceOne w).r (ringMul (faceOne w).σ z w')
      = (z.1 + z.2) * (w'.1 + w'.2) := by
  rw [face_channel_mul, show (faceOne w).r = 1 from rfl, ringUpper_one_sum, ringUpper_one_sum]

/-- **Face σ = −1, no root at width ≥ 2.** `2 ^ w` factors as `4 · 2 ^ (w − 2)`, so `4` is already
`0` in the ring. -/
theorem four_mul_two_pow_sub_two (w : ℕ) (hw : 2 ≤ w) : (4 : ℕ) * 2 ^ (w - 2) = 2 ^ w := by
  calc (4 : ℕ) * 2 ^ (w - 2) = 2 ^ 2 * 2 ^ (w - 2) := by rw [show (2 : ℕ) ^ 2 = 4 from by decide]
    _ = 2 ^ (2 + (w - 2)) := (pow_add 2 2 (w - 2)).symm
    _ = 2 ^ w := by rw [show (2 + (w - 2)) = w from by omega]

/-- Squares modulo `4` are `0` or `1`. The four-element ground check behind the elliptic absence;
this is the only `decide` in the file and it ranges over `Fin 4`. -/
theorem fin4_sq_mod (a : Fin 4) : (a.val ^ 2) % 4 ≤ 1 := by
  have h : ∀ a : Fin 4, (a.val ^ 2) % 4 ≤ 1 := by decide
  exact h a

/-- No square is `−1` modulo `4`: the arithmetic behind the elliptic absence at every word width. -/
theorem no_four_dvd_sq_add_one (k : ℕ) : ¬ (4 : ℕ) ∣ k ^ 2 + 1 := by
  intro h
  rw [Nat.dvd_iff_mod_eq_zero] at h
  have hlt : k % 4 < 4 := Nat.mod_lt _ (by decide)
  have hsq : (k % 4) ^ 2 % 4 ≤ 1 := fin4_sq_mod ⟨k % 4, hlt⟩
  have hmod : k ^ 2 % 4 = (k % 4) ^ 2 % 4 := by
    calc k ^ 2 % 4 = (k * k) % 4 := by rw [pow_two]
      _ = (k % 4) * (k % 4) % 4 := Nat.mul_mod k k 4
      _ = (k % 4) ^ 2 % 4 := by rw [← pow_two]
  have hsmall : (k % 4) ^ 2 % 4 + 1 < 4 := by omega
  rw [Nat.add_mod, show (1 : ℕ) % 4 = 1 by rfl, hmod, Nat.mod_eq_of_lt hsmall] at h
  omega

/-- **Face σ = −1 has no root at any width `w ≥ 2`.** If `r ^ 2 = −1` in `ZMod (2 ^ w)`, then
`2 ^ w ∣ r.val ^ 2 + 1`, hence `4 ∣ r.val ^ 2 + 1`, which `no_four_dvd_sq_add_one` refutes. This is
the ops map's clause "no root of σ for the elliptic face at word width" (row A5, obstruction O6),
proved once for all widths instead of per width. -/
theorem face_elliptic_no_root (w : ℕ) (hw : 2 ≤ w) : ¬ ∃ r : ZMod (2 ^ w), r ^ 2 = -1 := by
  rintro ⟨r, hr⟩
  have hrv : r = ((r.val : ℕ) : ZMod (2 ^ w)) := (ZMod.natCast_zmod_val r).symm
  rw [hrv] at hr
  have hz : (((r.val ^ 2 + 1 : ℕ)) : ZMod (2 ^ w)) = 0 := by
    have h : ((r.val ^ 2 + 1 : ℕ) : ZMod (2 ^ w))
        = ((r.val : ZMod (2 ^ w))) ^ 2 + 1 := by
      simp only [Nat.cast_add, Nat.cast_one, Nat.cast_pow]
    rw [h, hr]
    ring
  obtain ⟨q, hq⟩ := ZMod.natCast_eq_zero_iff _ _ |>.mp hz
  refine (no_four_dvd_sq_add_one r.val) ?_
  rw [hq, ← four_mul_two_pow_sub_two w hw, Nat.mul_assoc]
  exact dvd_mul_right 4 (2 ^ (w - 2) * q)

/-- **Composed face statement.** There is no `FaceRoot` at the elliptic face for `w ≥ 2`, so the
parametric channel lemma `face_channel_mul` has no elliptic instance at any word width. This is the
composed lemma a wrong face-specific elliptic claim would break. -/
theorem face_elliptic_absent (w : ℕ) (hw : 2 ≤ w) : ¬ ∃ F : FaceRoot w, F.σ = -1 := by
  rintro ⟨F, hF⟩
  exact face_elliptic_no_root w hw ⟨F.r, hF ▸ F.sq⟩

/-- The read at the elliptic face, for a formal root: the shape is `z.1 − r * z.2`. It has no
instance at any word width, by `face_elliptic_no_root`. -/
theorem face_elliptic_read_shape (w : ℕ) (r : ZMod (2 ^ w)) (z : RingCell (ZMod (2 ^ w))) :
    ringUpper (-r) z = z.1 - r * z.2 := by
  rw [ringUpper_def]
  ring

/-- The whole read table at width `w`, as one conjunction: the parabolic read is blind, the
hyperbolic read is the sum, and the elliptic read is the difference shape with no root available. -/
theorem face_read_table (w : ℕ) (z : RingCell (ZMod (2 ^ w))) :
    ringUpper (faceZero w).r z = z.1
      ∧ ringUpper (faceOne w).r z = z.1 + z.2
      ∧ ∀ r : ZMod (2 ^ w), ringUpper (-r) z = z.1 - r * z.2 := by
  refine ⟨?_, ringUpper_one_sum z, ?_⟩
  rw [show (faceZero w).r = 0 from rfl]
  exact ringUpper_zero_blind z
  exact fun r => face_elliptic_read_shape w r z

/-! ### §2b The same three faces for the norm form

Ops-map rows S2 and A3 read the norm form `a² − σ b²` off a paired round/state. The face table for
that second shared definition is the same three-case statement. -/

/-- The norm form is `z.1 ^ 2 − σ * z.2 ^ 2`: the body of `AnyRing.ringNormForm`. -/
theorem ringNormForm_def {R : Type*} [CommRing R] (σ : R) (z : RingCell R) :
    ringNormForm σ z = z.1 ^ 2 - σ * z.2 ^ 2 := rfl

/-- **The norm-form face table, as one parametric theorem over any commutative ring.** At σ = 0 it
is the square of the first slot; at σ = 1 the difference of squares; at σ = −1 the sum of two
squares. -/
theorem normForm_face_table {R : Type*} [CommRing R] (a b : R) :
    ringNormForm (0 : R) (a, b) = a ^ 2
      ∧ ringNormForm (1 : R) (a, b) = a ^ 2 - b ^ 2
      ∧ ringNormForm (-1 : R) (a, b) = a ^ 2 + b ^ 2 := by
  refine ⟨?_, ?_, ?_⟩
  · rw [ringNormForm_def, zero_mul, sub_zero]
  · rw [ringNormForm_def, one_mul]
  · simp only [ringNormForm_def, Prod.fst, Prod.snd]
    ring

/-- Multiplication stays on the axis of second slot zero, for every σ: the two-slot pairing is
`ringMul`-shaped, which is what makes a paired round/state readable. -/
theorem ringMul_axis {R : Type*} [CommRing R] (σ u v : R) :
    ringMul σ (u, (0 : R)) (v, (0 : R)) = (u * v, (0 : R)) := by
  simp [ringMul, Prod.ext_iff]

/-- The norm form on a paired point, for every σ. This is the shared definition both the
parabolic and hyperbolic clauses of the face table specialise. -/
theorem ringNormForm_axis {R : Type*} [CommRing R] (σ a b : R) :
    ringNormForm σ (a, b) = a ^ 2 - σ * b ^ 2 :=
  ringNormForm_def _ _

/-- **Ops-map row S2 as a face instance:** the hyperbolic read of the pairing `(t1, t2)` is
`ringNormForm 1 (t1, t2) = t1^2 − t2^2`, over any commutative ring. -/
theorem ringNormForm_hyperbolic_axis {R : Type*} [CommRing R] (a b : R) :
    ringNormForm (1 : R) (a, b) = a ^ 2 - b ^ 2 := by
  rw [ringNormForm_def, one_mul]

/-- **Ops-map row A3 as a shared hypothesis:** the norm form of the product of two axis points is
the product of their norm forms. Together with `ringMul_axis` this says the two-slot pairing is
`ringMul`-shaped and the readout is multiplicative, in one equation. -/
theorem ringNormForm_hyperbolic_mul {R : Type*} [CommRing R] (u v : R) :
    ringNormForm (1 : R) (ringMul (1 : R) (u, (0 : R)) (v, (0 : R)))
      = ringNormForm (1 : R) (u, (0 : R)) * ringNormForm (1 : R) (v, (0 : R)) :=
  ringNormForm_mul 1 (u, (0 : R)) (v, (0 : R))

/-- The value of that readout on the axis: `u² · v²`. -/
theorem ringNormForm_hyperbolic_mul_axis {R : Type*} [CommRing R] (u v : R) :
    ringNormForm (1 : R) (ringMul (1 : R) (u, (0 : R)) (v, (0 : R))) = u ^ 2 * v ^ 2 := by
  rw [ringMul_axis]
  simp only [ringNormForm_def, Prod.fst, Prod.snd, one_mul, pow_two, zero_mul, sub_zero]
  ring

/-! ### §2c The executing dial agrees with the face table

`denoteSigma` (`WordRefinement.lean:35`) is what fixes a σ for the extracted `Cell<i64>` bodies, so
it is the point where the abstract face table meets the executing code. -/

/-- The executing dial's hyperbolic σ is the hyperbolic face's parameter. -/
theorem dial_hyperbolic_is_faceOne : denoteSigma cell.Sigma.Hyperbolic = (faceOne 64).σ := rfl

/-- The executing dial's parabolic σ is the parabolic face's parameter. -/
theorem dial_parabolic_is_faceZero : denoteSigma cell.Sigma.Parabolic = (faceZero 64).σ := rfl

/-- The executing dial's elliptic σ has no root in the 64-bit ring, by the width-parametric elliptic
absence. So the parametric channel lemma has no elliptic instance on the executing carrier. -/
theorem dial_elliptic_no_root :
    ¬ ∃ r : ZMod (2 ^ 64), r ^ 2 = denoteSigma cell.Sigma.Elliptic := by
  have h : denoteSigma cell.Sigma.Elliptic = (-1 : ZMod (2 ^ 64)) := rfl
  rw [h]
  exact face_elliptic_no_root 64 (by decide)

/-- The channel law on the executing carrier at its hyperbolic face: `ringUpper` at
`denoteSigma cell.Sigma.Hyperbolic` is multiplicative. -/
theorem dial_hyperbolic_channel_mul (z w' : RingCell (ZMod (2 ^ 64))) :
    ringUpper (faceOne 64).r (ringMul (faceOne 64).σ z w')
      = ringUpper (faceOne 64).r z * ringUpper (faceOne 64).r w' :=
  face_channel_mul 64 (faceOne 64) z w'

/-! ## §3 The rotation identity as ONE statement in the width

`BitVec.toNat_rotateLeft` (`Init/Data/BitVec/Lemmas.lean:4948`) states the rotate-left value as a
bitwise `|||`; `Nat.two_pow_add_eq_or_of_lt` (`Init/Data/Nat/Bitwise/Lemmas.lean:745`) turns that
into a sum, because the wrapped part is below `2 ^ n` while the shift is a multiple of `2 ^ n`. The
result is one statement in `w`. -/

/-- **The rotation value, one statement in the width.** Rotating a `w`-bit word left by `n < w`
gives the truncated left shift plus the wrapped high bits, at every width. -/
theorem toNat_rotateLeft_eq_sum (w : ℕ) (x : BitVec w) (n : ℕ) (hn : n < w) :
    (x.rotateLeft n).toNat = (x.toNat * 2 ^ n) % 2 ^ w + x.toNat / 2 ^ (w - n) := by
  have hxpow : 2 ^ (w - n) * 2 ^ n = 2 ^ w := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hA : (x.toNat * 2 ^ n) % 2 ^ w = 2 ^ n * (x.toNat % 2 ^ (w - n)) := by
    have hmm := Nat.mul_mod_mul_left (2 ^ n) x.toNat (2 ^ (w - n))
    rw [mul_comm (2 ^ n) x.toNat, mul_comm (2 ^ n) (2 ^ (w - n)), hxpow] at hmm
    exact hmm
  have hq : x.toNat / 2 ^ (w - n) < 2 ^ n := by
    refine (Nat.div_lt_iff_lt_mul (by positivity)).mpr ?_
    rw [mul_comm (2 ^ n) (2 ^ (w - n)), hxpow]
    exact x.isLt
  rw [BitVec.toNat_rotateLeft, Nat.mod_eq_of_lt hn, Nat.shiftLeft_eq,
    Nat.shiftRight_eq_div_pow, hA]
  exact (Nat.two_pow_add_eq_or_of_lt hq (x.toNat % 2 ^ (w - n))).symm

/-- **The rotation identity in the ring, one statement in the width.** This is the corrected form of
ops-map candidate C5: the first summand is the *truncated* left shift (already reduced in
`ZMod (2 ^ w)`) and the second is the wrapped high bits. Ops-map rows S7, K2 and K4 are instances of
it at `w = 32`, `n = 1` and `n = ROT[i]` respectively. -/
theorem bvNat_rotateLeft_eq (w : ℕ) (x : BitVec w) (n : ℕ) (hn : n < w) :
    bvNat w (x.rotateLeft n)
      = bvNat w x * ((2 ^ n : ℕ) : ZMod (2 ^ w))
        + ((x.toNat / 2 ^ (w - n) : ℕ) : ZMod (2 ^ w)) := by
  have h := congrArg (fun t : ℕ => (t : ZMod (2 ^ w))) (toNat_rotateLeft_eq_sum w x n hn)
  rw [Nat.cast_add, ZMod.natCast_mod, Nat.cast_mul] at h
  simpa only [bvNat_def] using h

/-- The same statement with the shift factor written as the ring power `(2 : ZMod (2 ^ w)) ^ n`. The
two spellings are identified by `two_pow_cast_eq`, so a row may be restated in either without
changing what it says. -/
theorem bvNat_rotateLeft_eq_pow (w : ℕ) (x : BitVec w) (n : ℕ) (hn : n < w) :
    bvNat w (x.rotateLeft n)
      = bvNat w x * (2 : ZMod (2 ^ w)) ^ n
        + ((x.toNat / 2 ^ (w - n) : ℕ) : ZMod (2 ^ w)) := by
  rw [bvNat_rotateLeft_eq w x n hn, two_pow_cast_eq]

/-- `2` is below `2 ^ w` as soon as `w > 1`: the reduction bound used by every ground evaluation
below. -/
theorem two_lt_two_pow (w : ℕ) (hw : 1 < w) : (2 : ℕ) < 2 ^ w := by
  have h := Nat.pow_lt_pow_right (a := (2 : ℕ)) (by decide) (by omega)
  simpa using h

/-- The word `1` at width `w`, as `BitVec.ofNat`. -/
theorem bitvec_one_toNat (w : ℕ) (hw : 0 < w) : (BitVec.ofNat w 1).toNat = 1 := by
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (Nat.one_lt_two_pow (Nat.ne_of_gt hw))]

/-- The word `2` at width `w`, as `BitVec.ofNat`. -/
theorem bitvec_two_toNat (w : ℕ) (hw : 1 < w) : (BitVec.ofNat w 2).toNat = 2 := by
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (two_lt_two_pow w hw)]

/-- The `BitVec w` numeral `1` is `BitVec.ofNat w 1`: the two spellings of the witness agree. -/
theorem bitvecLit_one (w : ℕ) : (1 : BitVec w) = BitVec.ofNat w 1 := rfl

/-- The `BitVec w` numeral `2` is `BitVec.ofNat w 2`. -/
theorem bitvecLit_two (w : ℕ) : (2 : BitVec w) = BitVec.ofNat w 2 := rfl

/-- The `BitVec w` numeral `1` has unsigned value `1`. -/
theorem bitvecLit_one_toNat (w : ℕ) (hw : 0 < w) : (1 : BitVec w).toNat = 1 := by
  rw [bitvecLit_one, bitvec_one_toNat w hw]

/-- The `BitVec w` numeral `2` has unsigned value `2`. -/
theorem bitvecLit_two_toNat (w : ℕ) (hw : 1 < w) : (2 : BitVec w).toNat = 2 := by
  rw [bitvecLit_two, bitvec_two_toNat w hw]

theorem bitvec_two_pow_sub_one_toNat (w : ℕ) (hw : 0 < w) :
    (BitVec.ofNat w (2 ^ (w - 1))).toNat = 2 ^ (w - 1) := by
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (a := (2 : ℕ)) (by decide)
    (by omega))]

theorem bvNat_zero (w : ℕ) (_hw : 0 < w) : bvNat w ((0 : BitVec w)) = 0 := by
  rw [bvNat_def, show (0 : BitVec w) = BitVec.ofNat w 0 from rfl, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt (Nat.two_pow_pos w), Nat.cast_zero]

/-- The zero word rotates to the zero word. -/
theorem bvNat_rotateLeft_zero (w : ℕ) (n : ℕ) (hn : n < w) :
    bvNat w ((0 : BitVec w).rotateLeft n) = 0 := by
  rw [bvNat_rotateLeft_eq w _ n hn, bvNat_zero w (by omega)]
  simp

/-- `w − (w − n) = n` when `n ≤ w`: the amount exchange used in §5. -/
theorem sub_sub_eq (w n : ℕ) (hnw : n ≤ w) : w - (w - n) = n := by omega

/-! ## §4 The ops-map candidate expression, its value, and its agreement set

`ops-map.md` writes the C5 candidate as `ringUpper 1 (ringMul 1 (x,0) (2^n, 2^(64−n)))`, at
`ZMod (2 ^ 64)` for the Keccak rows and `ZMod (2 ^ 32)` for SHA-2 row S7. Below it is `rotExpr w`,
defined at the hyperbolic *face* of §2 (`faceOne w`) and at an arbitrary width: the two rows'
formulations are one definition at two arguments. -/

/-- The ops-map C5 candidate expression, at the hyperbolic face, at an arbitrary width. -/
def rotExpr (w x n : ℕ) : ZMod (2 ^ w) :=
  ringUpper (faceOne w).r
    (ringMul (faceOne w).σ (((x : ℕ) : ZMod (2 ^ w)), (0 : ZMod (2 ^ w)))
      (((2 ^ n : ℕ) : ZMod (2 ^ w)), ((2 ^ (w - n) : ℕ) : ZMod (2 ^ w))))

/-- The shared evaluation identity of the whole two-slot family, over every commutative ring and
with **no hypothesis**: `ringUpper r (ringMul σ (x,0) (a,b))` is `x ↦ x * (a + r * b)`. `σ` never
reaches the right-hand side, so the arithmetic of §4 needs no root; only channel multiplicativity
(`face_channel_mul`) does. -/
theorem ringUpper_mul_eval {R : Type*} [CommRing R] (σ r a b x : R) :
    ringUpper r (ringMul σ (x, 0) (a, b)) = x * (a + r * b) := by
  simp only [ringUpper_def, ringMul, Prod.fst, Prod.snd, zero_mul, add_zero, mul_zero, zero_add]
  ring

/-- **What the candidate computes, one statement in the width:** a single multiplication by
`2 ^ n + 2 ^ (w − n)`. This is the refutation of C5 as printed, at every width at once. -/
theorem rotExpr_eq (w x n : ℕ) :
    rotExpr w x n
      = ((x : ℕ) : ZMod (2 ^ w)) * ((2 ^ n : ℕ) : ZMod (2 ^ w))
        + ((x : ℕ) : ZMod (2 ^ w)) * ((2 ^ (w - n) : ℕ) : ZMod (2 ^ w)) := by
  simp only [rotExpr, show (faceOne w).r = 1 from rfl, show (faceOne w).σ = 1 from rfl,
    ringUpper_mul_eval, one_mul]
  ring

/-- **The offset, one statement in the width:** the difference between the candidate and the
rotation is exactly `x · 2 ^ (w − n) − ⌊x / 2 ^ (w − n)⌋`. -/
theorem rotExpr_offset (w : ℕ) (x : BitVec w) (n : ℕ) (hnw : n < w) :
    rotExpr w x.toNat n - bvNat w (x.rotateLeft n)
      = ((x.toNat * 2 ^ (w - n) : ℕ) : ZMod (2 ^ w))
        - ((x.toNat / 2 ^ (w - n) : ℕ) : ZMod (2 ^ w)) := by
  have h1 := rotExpr_eq w x.toNat n
  have h2 := bvNat_rotateLeft_eq w x n hnw
  simp only [bvNat_def] at h1 h2 ⊢
  rw [h1, h2, Nat.cast_mul]
  ring

/-- The candidate is right on a quarter of the word space at `n = 1`, at every width: if `2 ^ n`
divides the word and the word is below `2 ^ (w − n)`, the two slots cancel in the offset. One
criterion covering both `ZMod (2 ^ 64)` and `ZMod (2 ^ 32)`. -/
theorem rotExpr_agrees (w : ℕ) (x : BitVec w) (n : ℕ) (_hn0 : 0 < n) (hnw : n < w)
    (hdiv : 2 ^ n ∣ x.toNat) (hlt : x.toNat < 2 ^ (w - n)) :
    rotExpr w x.toNat n = bvNat w (x.rotateLeft n) := by
  refine sub_eq_zero.mp ?_
  rw [rotExpr_offset w x n hnw]
  have hpow : 2 ^ (w - n) * 2 ^ n = 2 ^ w := by
    rw [← Nat.pow_add]
    congr 1
    omega
  obtain ⟨k, hk⟩ := hdiv
  have hdv : 2 ^ w ∣ x.toNat * 2 ^ (w - n) := by
    refine ⟨k, ?_⟩
    rw [hk]
    calc 2 ^ n * k * 2 ^ (w - n) = 2 ^ (w - n) * (2 ^ n * k) := by ring
      _ = (2 ^ (w - n) * 2 ^ n) * k := by ring
      _ = 2 ^ w * k := by rw [hpow]
  have hz : ((x.toNat * 2 ^ (w - n) : ℕ) : ZMod (2 ^ w)) = 0 :=
    ZMod.natCast_eq_zero_iff _ _ |>.mpr hdv
  rw [hz, Nat.div_eq_of_lt hlt, Nat.cast_zero, sub_zero]

/-- The word `0` is agreed on, at every width and every amount. -/
theorem rotExpr_agrees_zero (w : ℕ) (n : ℕ) (hnw : n < w) :
    rotExpr w 0 n = bvNat w ((BitVec.ofNat w 0).rotateLeft n) := by
  have hrot : bvNat w ((BitVec.ofNat w 0).rotateLeft n) = 0 := by
    rw [bvNat_rotateLeft_eq w _ n hnw]
    simp only [bvNat_def, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (Nat.two_pow_pos w),
      Nat.cast_zero, zero_mul, zero_add, Nat.zero_div]
  have hexp : rotExpr w 0 n = 0 := by
    rw [rotExpr_eq]
    simp
  rw [hexp, hrot]

/-- The candidate is wrong at the word `1` and amount `1`, at every width: the offset is
`2 ^ (w − 1)`, which `two_pow_natCast_ne_zero` forbids from being `0`. One witness, one
non-vanishing lemma, all widths. -/
theorem rotExpr_offset_one (w : ℕ) (hw : 2 ≤ w) :
    rotExpr w 1 1 - bvNat w ((BitVec.ofNat w 1).rotateLeft 1)
      = ((2 ^ (w - 1) : ℕ) : ZMod (2 ^ w)) := by
  have key : rotExpr w (BitVec.ofNat w 1).toNat 1 - bvNat w ((BitVec.ofNat w 1).rotateLeft 1)
      = rotExpr w 1 1 - bvNat w ((BitVec.ofNat w 1).rotateLeft 1) := by
    rw [bitvec_one_toNat w (by omega)]
  rw [← key, rotExpr_offset w (BitVec.ofNat w 1) 1 (by omega),
    bitvec_one_toNat w (by omega)]
  have hq : (1 : ℕ) / 2 ^ (w - 1) = 0 := Nat.div_eq_of_lt (Nat.one_lt_two_pow (by omega : w - 1 ≠ 0))
  rw [hq, Nat.cast_zero, sub_zero, one_mul]

theorem rotExpr_offset_one_ne (w : ℕ) (hw : 2 ≤ w) :
    rotExpr w 1 1 - bvNat w ((BitVec.ofNat w 1).rotateLeft 1) ≠ 0 := by
  rw [rotExpr_offset_one w hw]
  exact two_pow_natCast_ne_zero w (w - 1) (by omega) (by omega)

/-- The candidate's value does not depend on which end of the rotation is written: exchanging the
amount `n` for `w − n` gives the same function. This is what ties ops-map row S7 (`rotr`) to rows K2
and K4 (`rotl`). -/
theorem rotExpr_amount_swap (w x n : ℕ) (hnw : n ≤ w) : rotExpr w x (w - n) = rotExpr w x n := by
  simp only [rotExpr, show (faceOne w).r = 1 from rfl, show (faceOne w).σ = 1 from rfl,
    ringUpper_mul_eval, one_mul, sub_sub_eq w n hnw]
  rw [← Nat.cast_add, ← Nat.cast_add, Nat.add_comm]

/-! ## §5 Cross-candidate agreement between ops-map rows

Row S7 is SHA-2's `rotr n` at width 32. Rows K2 and K4 are Keccak's `rotl` by `1` and by `ROT[i]` at
width 64. The ops map writes the same candidate expression for both, so the two rows agree only if
the amount is exchanged. That exchange is what §5 proves. -/

/-- Rotating right by `n` is rotating left by `w − n`, at every width. -/
theorem rotateRight_eq_rotateLeft (w : ℕ) (x : BitVec w) (n : ℕ) (_hn0 : 0 < n) (hn : n < w) :
    x.rotateRight n = x.rotateLeft (w - n) := by
  have hwn : w - n < w := by omega
  have h1 : n % w = n := Nat.mod_eq_of_lt hn
  have h2 : (w - n) % w = w - n := Nat.mod_eq_of_lt hwn
  simp only [BitVec.rotateRight_def, BitVec.rotateLeft_def, h1, h2]
  rw [sub_sub_eq w n (Nat.le_of_lt hn)]
  exact BitVec.or_comm _ _

/-- The same statement read in the ring models, through the carrier agreement of §1. -/
theorem bvNat_rotateRight_eq_rotateLeft (w : ℕ) (x : BitVec w) (n : ℕ) (_hn0 : 0 < n)
    (hn : n < w) : bvNat w (x.rotateRight n) = bvNat w (x.rotateLeft (w - n)) := by
  rw [bvNat_def, bvNat_def, rotateRight_eq_rotateLeft w x n _hn0 hn]

/-- **Ops-map row S7's offset, at every width:** for the *printed* multiplier `(2 ^ n,
2 ^ (w − n))` the defect of `rotr n` is `x · 2 ^ n − ⌊x / 2 ^ n⌋`, not the `rotl` defect
`x · 2 ^ (w − n) − ⌊x / 2 ^ (w − n)⌋`. -/
theorem rotExpr_offset_rotateRight (w : ℕ) (x : BitVec w) (n : ℕ) (hn0 : 0 < n) (hnw : n < w) :
    rotExpr w x.toNat n - bvNat w (x.rotateRight n)
      = ((x.toNat * 2 ^ n : ℕ) : ZMod (2 ^ w))
        - ((x.toNat / 2 ^ n : ℕ) : ZMod (2 ^ w)) := by
  calc rotExpr w x.toNat n - bvNat w (x.rotateRight n)
      = rotExpr w x.toNat (w - n) - bvNat w (x.rotateLeft (w - n)) := by
        rw [bvNat_rotateRight_eq_rotateLeft w x n hn0 hnw,
          ← rotExpr_amount_swap w x.toNat n (Nat.le_of_lt hnw)]
    _ = ((x.toNat * 2 ^ (w - (w - n)) : ℕ) : ZMod (2 ^ w))
        - ((x.toNat / 2 ^ (w - (w - n)) : ℕ) : ZMod (2 ^ w)) :=
      rotExpr_offset w x (w - n) (by omega)
    _ = ((x.toNat * 2 ^ n : ℕ) : ZMod (2 ^ w))
        - ((x.toNat / 2 ^ n : ℕ) : ZMod (2 ^ w)) := by
        rw [sub_sub_eq w n (Nat.le_of_lt hnw)]

/-- **Cross-candidate agreement, rows S7 and K4.** The defect of the printed candidate against
SHA-2's `rotr n` equals its defect against Keccak's `rotl (w − n)`, for every width, word and
amount. A wrong multiplier in either row's transcription breaks this equation, and a wrong `rotl`
reading of row S7 breaks `bvNat_rotateRight_eq_rotateLeft`. -/
theorem s7_k4_offset_agree (w : ℕ) (x : BitVec w) (n : ℕ) (hn0 : 0 < n) (hnw : n < w) :
    rotExpr w x.toNat n - bvNat w (x.rotateRight n)
      = rotExpr w x.toNat (w - n) - bvNat w (x.rotateLeft (w - n)) := by
  rw [bvNat_rotateRight_eq_rotateLeft w x n hn0 hnw,
    ← rotExpr_amount_swap w x.toNat n (Nat.le_of_lt hnw)]

/-- **Row S7's own witness, at every width.** At `x = 1`, `n = 1`, the printed candidate differs
from `rotr 1` by exactly `2`, which is non-zero in `ZMod (2 ^ w)` for `w ≥ 2`. Row K4's witness
differs by `2 ^ (w − 1)` (`rotExpr_offset_one`): the two rows fail by *different* amounts at the
same word, which is exactly why both needed checking. -/
theorem s7_offset_one (w : ℕ) (hw : 2 ≤ w) :
    rotExpr w 1 1 - bvNat w ((BitVec.ofNat w 1).rotateRight 1)
      = ((2 : ℕ) : ZMod (2 ^ w)) := by
  calc rotExpr w 1 1 - bvNat w ((BitVec.ofNat w 1).rotateRight 1)
      = rotExpr w 1 (w - 1) - bvNat w ((BitVec.ofNat w 1).rotateLeft (w - 1)) := by
        rw [bvNat_rotateRight_eq_rotateLeft w _ 1 (by decide) (by omega),
          ← rotExpr_amount_swap w 1 1 (by omega)]
    _ = ((1 * 2 ^ (w - (w - 1)) : ℕ) : ZMod (2 ^ w))
        - ((1 / 2 ^ (w - (w - 1)) : ℕ) : ZMod (2 ^ w)) := by
        have key1 : rotExpr w (BitVec.ofNat w 1).toNat (w - 1)
              - bvNat w ((BitVec.ofNat w 1).rotateLeft (w - 1))
            = rotExpr w 1 (w - 1) - bvNat w ((BitVec.ofNat w 1).rotateLeft (w - 1)) := by
          rw [bitvec_one_toNat w (by omega)]
        rw [← key1, rotExpr_offset w (BitVec.ofNat w 1) (w - 1) (by omega),
          bitvec_one_toNat w (by omega)]
    _ = ((2 : ℕ) : ZMod (2 ^ w)) := by
        rw [sub_sub_eq w 1 (by omega)]
        have hq : (1 : ℕ) / 2 ^ 1 = 0 := Nat.div_eq_of_lt (by decide)
        rw [hq, Nat.cast_zero, one_mul, sub_zero, ← two_pow_cast_eq w 1, pow_one]
        norm_num

theorem s7_offset_one_ne (w : ℕ) (hw : 2 ≤ w) :
    rotExpr w 1 1 - bvNat w ((BitVec.ofNat w 1).rotateRight 1) ≠ 0 := by
  rw [s7_offset_one w hw]
  exact two_pow_natCast_ne_zero w 1 (by omega) (by omega)

/-- **Row K2 evaluated, width 64, `n = 1`, word `1`.** The ops map's printed right-hand side is
right here; its printed left-hand side is not, by `rotExpr_offset_one_ne 64`. -/
theorem k2_candidate_one : rotExpr 64 1 1 = ((2 + 2 ^ 63 : ℕ) : ZMod (2 ^ 64)) := by
  have hrot : bvNat 64 ((BitVec.ofNat 64 1).rotateLeft 1)
      = ((2 : ℕ) : ZMod (2 ^ 64)) := by
    rw [bvNat_rotateLeft_eq 64 _ 1 (by decide)]
    simp only [bvNat_def, bitvec_one_toNat 64 (by decide),
      Nat.div_eq_of_lt (by decide : 1 < 2 ^ 63), Nat.cast_zero, zero_add, Nat.cast_mul,
      Nat.cast_one, one_mul]
    push_cast
    ring
  have hoff := rotExpr_offset_one 64 (by decide)
  rw [hrot] at hoff
  calc rotExpr 64 1 1 = ((2 : ℕ) : ZMod (2 ^ 64)) + ((2 ^ 63 : ℕ) : ZMod (2 ^ 64)) := by
        rw [← hoff]
        ring
    _ = ((2 + 2 ^ 63 : ℕ) : ZMod (2 ^ 64)) := by rw [Nat.cast_add]

/-- **Width agreement, first form.** The 64-bit identity (rows K2, K4) and the 32-bit identity
(row S7) are instances of one statement, and the refutation of the printed candidate is proved at
both widths by one witness and one non-vanishing lemma. -/
theorem c5_refuted_at_both_widths :
    rotExpr 64 1 1 - bvNat 64 ((BitVec.ofNat 64 1).rotateLeft 1) ≠ 0
      ∧ rotExpr 32 1 1 - bvNat 32 ((BitVec.ofNat 32 1).rotateLeft 1) ≠ 0
      ∧ rotExpr 32 1 1 - bvNat 32 ((BitVec.ofNat 32 1).rotateRight 1) ≠ 0 :=
  ⟨rotExpr_offset_one_ne 64 (by decide), rotExpr_offset_one_ne 32 (by decide),
    s7_offset_one_ne 32 (by decide)⟩

/-- **Width agreement, second form.** The `ZMod (2 ^ 64)` and `ZMod (2 ^ 32)` evaluations of the
candidate are instances of one identity, so a width-specific factor cannot drift. -/
theorem rotExpr_eq_at_both_widths (x n : ℕ) :
    rotExpr 64 x n = ((x : ℕ) : ZMod (2 ^ 64)) * ((2 ^ n : ℕ) : ZMod (2 ^ 64))
        + ((x : ℕ) : ZMod (2 ^ 64)) * ((2 ^ (64 - n) : ℕ) : ZMod (2 ^ 64))
      ∧ rotExpr 32 x n = ((x : ℕ) : ZMod (2 ^ 32)) * ((2 ^ n : ℕ) : ZMod (2 ^ 32))
        + ((x : ℕ) : ZMod (2 ^ 32)) * ((2 ^ (32 - n) : ℕ) : ZMod (2 ^ 32)) :=
  ⟨rotExpr_eq 64 x n, rotExpr_eq 32 x n⟩

/-! ## §6 The composed obstructions, at every width

Everything here is derived from the shared definitions of §1–§2 and the identities of §3–§4, so a
defect in `ringUpper`, in `ringMul`, or in the carrier lift breaks a *composed* statement rather
than a single row. -/

/-- `2 ^ m` is non-zero in `ZMod (2 ^ w)` for every `m < w`, including `m = 0`. This is the
non-vanishing obligation used by the second witness of `no_read_computes_rotation`. -/
theorem two_pow_natCast_ne_zero_of_lt (w m : ℕ) (hmw : m < w) :
    ((2 ^ m : ℕ) : ZMod (2 ^ w)) ≠ 0 := by
  intro hz
  have hdvd : 2 ^ w ∣ 2 ^ m := ZMod.natCast_eq_zero_iff _ _ |>.mp hz
  rw [Nat.dvd_iff_mod_eq_zero,
    Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (a := (2 : ℕ)) (by decide) hmw)] at hdvd
  exact absurd hdvd (Nat.ne_of_gt (Nat.two_pow_pos m))

/-- `2` is non-zero in `ZMod (2 ^ w)` as soon as `2 ≤ w`. -/
theorem two_natCast_ne_zero (w : ℕ) (hw : 2 ≤ w) : ((2 : ℕ) : ZMod (2 ^ w)) ≠ 0 :=
  two_pow_natCast_ne_zero w 1 (by omega) (by omega)

/-- **Shared hypothesis of rows S7, K2 and K4:** the two-slot read family is additive in the word,
over every commutative ring. Being additive is what makes it unusable as a rotation. -/
theorem read_family_additive {R : Type*} [CommRing R] (σ r a b x y : R) :
    ringUpper r (ringMul σ (x, 0) (a, b)) + ringUpper r (ringMul σ (y, 0) (a, b))
      = ringUpper r (ringMul σ (x + y, 0) (a, b)) := by
  simp only [ringUpper_mul_eval]
  ring

/-- **Composed obstruction, at every width.** The denotation of `rotateLeft 1` is not additive in
the word for `w ≥ 2`, witnessed at the word `2 ^ (w − 1)`: the two images sum to `2`, while the
image of the sum is `0`. -/
theorem rotateLeft_denotation_not_additive (w : ℕ) (hw : 2 ≤ w) :
    ¬ (∀ x y : BitVec w,
        bvNat w (x.rotateLeft 1) + bvNat w (y.rotateLeft 1)
          = bvNat w ((x + y).rotateLeft 1)) := by
  intro h
  have htv : (BitVec.ofNat w (2 ^ (w - 1))).toNat = 2 ^ (w - 1) :=
    bitvec_two_pow_sub_one_toNat w (by omega)
  have hrot : bvNat w ((BitVec.ofNat w (2 ^ (w - 1))).rotateLeft 1)
      = (1 : ZMod (2 ^ w)) := by
    rw [bvNat_rotateLeft_eq w _ 1 (by omega), bvNat_def, htv]
    have hpow : 2 ^ (w - 1) * 2 ^ 1 = 2 ^ w := by
      rw [← Nat.pow_add]
      congr 1
      omega
    rw [← Nat.cast_mul, hpow, two_pow_natCast_zero,
      Nat.div_self (Nat.two_pow_pos (w - 1)), Nat.cast_one, zero_add]
  have hbv : bvNat w (BitVec.ofNat w (2 ^ (w - 1))) = ((2 ^ (w - 1) : ℕ) : ZMod (2 ^ w)) := by
    rw [bvNat_def, htv]
  have hsum : BitVec.ofNat w (2 ^ (w - 1)) + BitVec.ofNat w (2 ^ (w - 1)) = (0 : BitVec w) := by
    refine bvNat_injective w ?_
    have hpow : 2 ^ (w - 1) + 2 ^ (w - 1) = 2 ^ w := by
      calc 2 ^ (w - 1) + 2 ^ (w - 1) = 2 ^ (w - 1) * 2 := by ring
        _ = 2 ^ w := by rw [← pow_succ]; congr 1; omega
    rw [bvNat_add, hbv, bvNat_zero w (by omega)]
    simp only [← Nat.cast_add, hpow, two_pow_natCast_zero]
  have htwo : (2 : ZMod (2 ^ w)) = 1 + 1 := by ring
  have h1 := h (BitVec.ofNat w (2 ^ (w - 1))) (BitVec.ofNat w (2 ^ (w - 1)))
  rw [hrot, hsum, bvNat_rotateLeft_zero w 1 (by omega), ← htwo] at h1
  exact two_natCast_ne_zero w hw h1

/-- **Composed obstruction: no member of the two-slot read family reads a rotation**, at any σ, any
root and any width, for every amount `0 < n < w`. Ops-map obstruction O5 together with the
additivity of `read_family_additive`, in one statement. -/
theorem no_read_computes_rotation (w : ℕ) (n : ℕ) (_hn0 : 0 < n) (hnw : n < w)
    (σ r a b : ZMod (2 ^ w)) :
    ¬ ∀ k : ℕ, ringUpper r (ringMul σ (((k : ℕ) : ZMod (2 ^ w)), 0) (a, b))
        = bvNat w ((BitVec.ofNat w k).rotateLeft n) := by
  intro hall
  have h1v : (BitVec.ofNat w 1).toNat = 1 := bitvec_one_toNat w (by omega)
  have htv : (BitVec.ofNat w (2 ^ (w - 1))).toNat = 2 ^ (w - 1) :=
    bitvec_two_pow_sub_one_toNat w (by omega)
  have hpow : 2 ^ (w - n) * 2 ^ n = 2 ^ w := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hdv : 2 ^ w ∣ 2 ^ (w - 1) * 2 ^ n := by
    refine ⟨2 ^ (n - 1), ?_⟩
    have h1 : 2 ^ (w - 1) * 2 ^ n = 2 ^ (w - 1 + n) := (pow_add _ _ _).symm
    have h2 : w - 1 + n = w + (n - 1) := by omega
    have h3 : 2 ^ (w + (n - 1)) = 2 ^ w * 2 ^ (n - 1) := pow_add _ _ _
    rw [h1, h2, h3]

  have hrot1 : bvNat w ((BitVec.ofNat w 1).rotateLeft n)
      = ((2 ^ n : ℕ) : ZMod (2 ^ w)) := by
    rw [bvNat_rotateLeft_eq w _ n hnw]
    simp only [bvNat_def, bitvec_one_toNat w (by omega),
      Nat.div_eq_of_lt (Nat.one_lt_two_pow (by omega : w - n ≠ 0)), Nat.cast_mul,
      Nat.cast_one, one_mul, Nat.cast_zero, add_zero]
  have hdiv : 2 ^ (w - n) * 2 ^ (n - 1) = 2 ^ (w - 1) := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hrot2 : bvNat w ((BitVec.ofNat w (2 ^ (w - 1))).rotateLeft n)
      = ((2 ^ (n - 1) : ℕ) : ZMod (2 ^ w)) := by
    rw [bvNat_rotateLeft_eq w _ n hnw, bvNat_def, htv]
    have hz' : ((2 ^ (w - 1) : ℕ) : ZMod (2 ^ w)) * ((2 ^ n : ℕ) : ZMod (2 ^ w)) = 0 := by
      obtain ⟨q, hq⟩ := hdv
      rw [← Nat.cast_mul, hq, Nat.cast_mul, two_pow_natCast_zero, zero_mul]
    rw [hz', ← hdiv, Nat.mul_div_cancel_left (2 ^ (n - 1)) (Nat.two_pow_pos (w - n)),
      zero_add]
  have hA : a + r * b = ((2 ^ n : ℕ) : ZMod (2 ^ w)) := by
    have h := hall 1
    rw [ringUpper_mul_eval] at h
    simp only [Nat.cast_one, one_mul] at h
    rw [hrot1] at h
    exact h
  have hzero : ((2 ^ (w - 1) * 2 ^ n : ℕ) : ZMod (2 ^ w)) = 0 := by
    obtain ⟨q, hq⟩ := hdv
    rw [hq, Nat.cast_mul, two_pow_natCast_zero, zero_mul]
  have hB := hall (2 ^ (w - 1))
  rw [ringUpper_mul_eval, hA, hrot2] at hB
  rw [← Nat.cast_mul, hzero] at hB
  exact two_pow_natCast_ne_zero_of_lt w (n - 1) (by omega) hB.symm

/-- **Composed obstruction, at every width: `x &&& 1` is not affine over `ZMod (2 ^ w)`.** An
affine map is additive, so the images of `1` and `1` would have to sum to the image of `2`; they sum
to `2` and the image of `2` is `0`, and `2 ≠ 0` in `ZMod (2 ^ w)` for `w ≥ 2`. Ops-map obstruction
O2 at `ZMod 16`, lifted to every width. -/
theorem and_one_denotation_not_additive (w : ℕ) (hw : 2 ≤ w) :
    ¬ (∀ x y : BitVec w, bvNat w (x &&& 1) + bvNat w (y &&& 1)
        = bvNat w ((x + y) &&& 1)) := by
  intro h
  have hwn0 : w ≠ 0 := by omega
  have hwn1 : 1 < w := by omega
  have hone2 : BitVec.ofNat w 1 + BitVec.ofNat w 1 = BitVec.ofNat w 2 := by
    refine BitVec.eq_of_toNat_eq ?_
    simp only [BitVec.toNat_add, BitVec.toNat_ofNat,
      Nat.mod_eq_of_lt (Nat.one_lt_two_pow hwn0),
      Nat.mod_eq_of_lt (two_lt_two_pow w hwn1), show (1 + 1) = (2 : ℕ) from rfl]
  have hone : (1 : BitVec w).toNat = 1 := bitvecLit_one_toNat w (by omega)
  have h1 : ((BitVec.ofNat w 1) &&& 1).toNat = 1 := by
    rw [BitVec.toNat_and, bitvec_one_toNat w (by omega), hone]
    decide
  have h2 : ((BitVec.ofNat w 2) &&& 1).toNat = 0 := by
    rw [BitVec.toNat_and, bitvec_two_toNat w (by omega), hone]
    decide
  have htwo : (2 : ZMod (2 ^ w)) = 1 + 1 := by ring
  have hA := h (BitVec.ofNat w 1) (BitVec.ofNat w 1)
  rw [hone2] at hA
  simp only [bvNat_def, h1, h2, Nat.cast_add, Nat.cast_one, Nat.cast_zero] at hA
  rw [← htwo] at hA
  exact two_natCast_ne_zero w hw hA

/-- **Composed obstruction, at every width: XOR by `1` is not affine over `ZMod (2 ^ w)`.** The
images of `1` and `1` sum to `0` and the image of `2` is `3`, and `3 ≠ 0` in `ZMod (2 ^ w)` for
`w ≥ 2`. This covers ops-map obstruction O4 at `c = 1` and obstruction O1's `1 ^^^ 1 = 0`. -/
theorem xor_one_denotation_not_additive (w : ℕ) (hw : 2 ≤ w) :
    ¬ (∀ x y : BitVec w, bvNat w (x ^^^ 1) + bvNat w (y ^^^ 1)
        = bvNat w ((x + y) ^^^ 1)) := by
  intro h
  have hwn0 : w ≠ 0 := by omega
  have hwn1 : 1 < w := by omega
  have hone2 : BitVec.ofNat w 1 + BitVec.ofNat w 1 = BitVec.ofNat w 2 := by
    refine BitVec.eq_of_toNat_eq ?_
    simp only [BitVec.toNat_add, BitVec.toNat_ofNat,
      Nat.mod_eq_of_lt (Nat.one_lt_two_pow hwn0),
      Nat.mod_eq_of_lt (two_lt_two_pow w hwn1), show (1 + 1) = (2 : ℕ) from rfl]
  have hone : (1 : BitVec w).toNat = 1 := bitvecLit_one_toNat w (by omega)
  have h1 : ((BitVec.ofNat w 1) ^^^ 1).toNat = 0 := by
    rw [BitVec.toNat_xor, bitvec_one_toNat w (by omega), hone]
    decide
  have h2 : ((BitVec.ofNat w 2) ^^^ 1).toNat = 3 := by
    rw [BitVec.toNat_xor, bitvec_two_toNat w (by omega), hone]
    decide
  have hthree : ((3 : ℕ) : ZMod (2 ^ w)) ≠ 0 := by
    intro hz
    obtain ⟨q, hq⟩ := ZMod.natCast_eq_zero_iff _ _ |>.mp hz
    rcases Nat.eq_zero_or_pos q with hq' | hq'
    · rw [hq', Nat.mul_zero] at hq
      exact absurd hq (by decide)
    · have hle : 2 ^ w ≤ q * 2 ^ w := Nat.le_mul_of_pos_left (2 ^ w) hq'
      rw [Nat.mul_comm q (2 ^ w), ← hq] at hle
      have hle2 : 2 ≤ 2 ^ w := Nat.le_of_lt (two_lt_two_pow w hwn1)
      omega
  have hA := h (BitVec.ofNat w 1) (BitVec.ofNat w 1)
  rw [hone2] at hA
  simp only [bvNat_def, h1, h2, Nat.cast_zero, zero_add] at hA
  exact hthree hA.symm

/-! ## Axiom cone

Every theorem above audits inside `{propext, Classical.choice, Quot.sound}` and none
uses `sorryAx`. Rows reading `'does not depend on any axioms'` are the `BitVec`-level facts that
are closed by `rfl`/`decide` alone. -/

#print axioms RingModelWeb.bvNat_def
#print axioms RingModelWeb.two_pow_natCast_zero
#print axioms RingModelWeb.bvNat_val
#print axioms RingModelWeb.bvNat_natCast_mod
#print axioms RingModelWeb.bvNat_val_natCast
#print axioms RingModelWeb.bvNat_ofNat
#print axioms RingModelWeb.bvNat_injective
#print axioms RingModelWeb.bvNat_eq_iff
#print axioms RingModelWeb.bvNat_add
#print axioms RingModelWeb.bvNat_sub
#print axioms RingModelWeb.two_pow_cast_eq
#print axioms RingModelWeb.two_pow_natCast_ne_zero
#print axioms RingModelWeb.ringUpper_def
#print axioms RingModelWeb.ringUpper_zero_blind
#print axioms RingModelWeb.ringUpper_one_sum
#print axioms RingModelWeb.ringUpper_add
#print axioms RingModelWeb.face_read
#print axioms RingModelWeb.face_channel_mul
#print axioms RingModelWeb.face_parabolic_read_mul
#print axioms RingModelWeb.face_hyperbolic_read_mul
#print axioms RingModelWeb.four_mul_two_pow_sub_two
#print axioms RingModelWeb.fin4_sq_mod
#print axioms RingModelWeb.no_four_dvd_sq_add_one
#print axioms RingModelWeb.face_elliptic_no_root
#print axioms RingModelWeb.face_elliptic_absent
#print axioms RingModelWeb.face_elliptic_read_shape
#print axioms RingModelWeb.face_read_table
#print axioms RingModelWeb.ringNormForm_def
#print axioms RingModelWeb.normForm_face_table
#print axioms RingModelWeb.ringMul_axis
#print axioms RingModelWeb.ringNormForm_axis
#print axioms RingModelWeb.ringNormForm_hyperbolic_axis
#print axioms RingModelWeb.ringNormForm_hyperbolic_mul
#print axioms RingModelWeb.ringNormForm_hyperbolic_mul_axis
#print axioms RingModelWeb.dial_hyperbolic_is_faceOne
#print axioms RingModelWeb.dial_parabolic_is_faceZero
#print axioms RingModelWeb.dial_elliptic_no_root
#print axioms RingModelWeb.dial_hyperbolic_channel_mul
#print axioms RingModelWeb.toNat_rotateLeft_eq_sum
#print axioms RingModelWeb.bvNat_rotateLeft_eq
#print axioms RingModelWeb.bvNat_rotateLeft_eq_pow
#print axioms RingModelWeb.two_lt_two_pow
#print axioms RingModelWeb.bitvec_one_toNat
#print axioms RingModelWeb.bitvec_two_toNat
#print axioms RingModelWeb.bitvecLit_one
#print axioms RingModelWeb.bitvecLit_two
#print axioms RingModelWeb.bitvecLit_one_toNat
#print axioms RingModelWeb.bitvecLit_two_toNat
#print axioms RingModelWeb.bitvec_two_pow_sub_one_toNat
#print axioms RingModelWeb.bvNat_zero
#print axioms RingModelWeb.bvNat_rotateLeft_zero
#print axioms RingModelWeb.sub_sub_eq
#print axioms RingModelWeb.ringUpper_mul_eval
#print axioms RingModelWeb.rotExpr_eq
#print axioms RingModelWeb.rotExpr_offset
#print axioms RingModelWeb.rotExpr_agrees
#print axioms RingModelWeb.rotExpr_agrees_zero
#print axioms RingModelWeb.rotExpr_offset_one
#print axioms RingModelWeb.rotExpr_offset_one_ne
#print axioms RingModelWeb.rotExpr_amount_swap
#print axioms RingModelWeb.rotateRight_eq_rotateLeft
#print axioms RingModelWeb.bvNat_rotateRight_eq_rotateLeft
#print axioms RingModelWeb.rotExpr_offset_rotateRight
#print axioms RingModelWeb.s7_k4_offset_agree
#print axioms RingModelWeb.s7_offset_one
#print axioms RingModelWeb.s7_offset_one_ne
#print axioms RingModelWeb.k2_candidate_one
#print axioms RingModelWeb.c5_refuted_at_both_widths
#print axioms RingModelWeb.rotExpr_eq_at_both_widths
#print axioms RingModelWeb.two_pow_natCast_ne_zero_of_lt
#print axioms RingModelWeb.two_natCast_ne_zero
#print axioms RingModelWeb.read_family_additive
#print axioms RingModelWeb.rotateLeft_denotation_not_additive
#print axioms RingModelWeb.no_read_computes_rotation
#print axioms RingModelWeb.and_one_denotation_not_additive
#print axioms RingModelWeb.xor_one_denotation_not_additive
