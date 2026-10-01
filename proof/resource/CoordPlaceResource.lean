import CoordPlaceSource
import Aeneas

/-
SCOPE: the actual extracted `word_coord_place` field read and the actual extracted `coord::spread`
that produces the word it reads: the addressing transitions with their guards, the closed-form work
count, the refusal branch, and the reachable success witnesses.
REGISTER: scalar source-refinement and resource model over the generated Aeneas bodies. Excluded:
any claim about the group bit widths, the tag, `coord::unspread`, `PHI_INV`, the table, the
`ApeironWord` extraction, distribution or injectivity of `place` on the full word, and machine
instruction or cycle counts.
CORPUS: `src/coord.rs` `spread` and the `Addr` structure; the target root `word_coord_place` is
`proof/extract/lib.rs:520`, a projection of the `place` field. Native grade `proved` for the
extracted bodies only. The resource count is a count of Aeneas-level operations, not a hardware
measurement.
REFERENCE: `/home/socol/Workspace/apeiron/proof/ApeironWord.lean` (the committed extraction of the
same `coord.rs` bodies, read for the same definitions).
GATE: `python3 proof/resource/CoordPlace_gate.py --controls`, whole invocation bounded at 540 s
under `/tmp/opencode/apeiron-lane-3.lock`, `MemoryMax=12G`, `MemorySwapMax=0`, `LEAN_NUM_THREADS=1`.
GRADE: Lean-checked only when that gate succeeds. The work count is a count of word operations the
generated term issues; it is not a machine-time measurement and establishes no latency or
throughput ratio.
TRUST BOUNDARY: Aeneas and Charon, the Rust compiler, the `u128` `wrapping_mul` model, and the
`BitVec` semantics of `rotateLeft`, `ushiftRight`, `zeroExtend` and `xor`. The `u128` multiply is
one Aeneas operation and a 64-bit machine lowers it to two or three, so the count below is Aeneas
level. Nothing here covers the group descent, the tag, the table's reads and writes, GPU or
SPIR-V, or any layer the extraction does not reach.
-/

namespace CoordPlaceResource
open Aeneas Aeneas.Std Result

/-! ## Word widths of the real bodies

`UScalarTy.numBits` is the model's word width, read here as the Aeneas-level fact it is. -/

theorem nbU64 : UScalarTy.U64.numBits = 64 := rfl

theorem nbU128 : UScalarTy.U128.numBits = 128 := rfl

/-- A `u64` sits below `2^64`. -/
theorem u64_lt (x : U64) : x.val < 2 ^ 64 := BitVec.isLt (x.bv : BitVec 64)

/-- A `u128` sits below `2^128`. -/
theorem u128_lt (w : U128) : w.val < 2 ^ 128 := BitVec.isLt (w.bv : BitVec 128)

/-! ## The value of each operation of the producer, in ℕ

Each theorem below names one operation of the generated term and says what value it produces. The
premises are the width facts above, so the work is stated under named premises and not assumed. -/

/-- **The first widening.** `x as u128` is the identity on the value: the source is already
    nonnegative, so the zero extension adds nothing. -/
theorem cast_u128_id (x : U64) : (UScalar.cast UScalarTy.U128 x).val = x.val := by
  have hlt : x.bv.toNat < 2 ^ 128 := by
    have h := u64_lt x
    norm_num at h ⊢
    omega
  rw [UScalar.cast, UScalar.val, BitVec.zeroExtend_eq_setWidth, BitVec.toNat_setWidth, nbU128]
  exact Nat.mod_eq_of_lt hlt

/-- **The second widening.** The constant the extracted body multiplies by is the literal
    `11400714819323198485`, and widening it to `u128` is the identity on its value. -/
theorem cast_u128_PHI :
    (UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI).val
      = coord_place_source.coord.PHI.val := by
  have hlt : (coord_place_source.coord.PHI : U64).bv.toNat < 2 ^ 128 := by
    have h := u64_lt coord_place_source.coord.PHI
    norm_num at h ⊢
    omega
  unfold UScalar.val
  rw [UScalar.cast, BitVec.zeroExtend_eq_setWidth, BitVec.toNat_setWidth, nbU128]
  exact Nat.mod_eq_of_lt hlt

theorem PHIval_lt : coord_place_source.coord.PHI.val < 2 ^ 64 := by
  unfold coord_place_source.coord.PHI
  norm_num [UScalar.val, nbU64]

/-- **The 128-bit multiply.** On operands that came from `u64` words the product fits in 128 bits,
    so `wrapping_mul` is the exact product in ℕ and discards nothing. The two width premises are
    the widening theorems above; this is the only step that could discard bits, and it does not. -/
theorem mul_u128 (a b : U128) (ha : a.val < 2 ^ 64) (hb : b.val < 2 ^ 64) :
    (core.num.U128.wrapping_mul a b).val = a.val * b.val := by
  have hlt : a.bv.toNat * b.bv.toNat < 2 ^ 64 * 2 ^ 64 :=
    Nat.mul_lt_mul_of_lt_of_lt ha hb
  have hlt' : a.bv.toNat * b.bv.toNat < 2 ^ UScalarTy.U128.numBits := by
    rw [nbU128]
    rw [show (2 ^ 64 * 2 ^ 64 : Nat) = 2 ^ 128 by norm_num [Nat.pow_add]] at hlt
    exact hlt
  unfold UScalar.val
  rw [core.num.U128.wrapping_mul, UScalar.wrapping_mul, BitVec.toNat_mul]
  show a.bv.toNat * b.bv.toNat % 2 ^ UScalarTy.U128.numBits = a.bv.toNat * b.bv.toNat
  exact Nat.mod_eq_of_lt hlt'

/-- `prod x` is the 128-bit product the real body forms, read in ℕ. The `u64` widening premises make
    the product exact, so the division and the remainder below are the two halves the body splits. -/
def prod (x : U64) : Nat := x.val * coord_place_source.coord.PHI.val

theorem prod_lt (x : U64) : prod x < 2 ^ 128 := by
  have h1 := u64_lt x
  have h2 := PHIval_lt
  unfold prod
  have h3 := Nat.mul_lt_mul_of_lt_of_lt h1 h2
  simpa [nbU128, Nat.pow_add] using h3

/-- `rotl32 n` is the Aeneas-level operation `UScalar.rotate_left n 32#u32` on `Std.U64`, read in ℕ:
    the 32-bit left rotation of the low 64 bits of `n`. -/
def rotl32 (n : Nat) : Nat := ((BitVec.ofNat 64 n).rotateLeft 32).toNat

/-- **Guard 2, the rotate amount.** `UScalar.rotate_left` is total: the model reduces the amount
    modulo the word width, so no amount and no input reaches a refusal, and the value it returns is
    the 32-bit rotation of the operand. -/
theorem rotl32_val (h : U64) :
    (core.num.U64.rotate_left h 32#u32).val = rotl32 h.val := by
  unfold rotl32
  rw [core.num.U64.rotate_left, UScalar.rotate_left, UScalar.val]
  have hv : (32#u32 : U32).val = 32 := by decide
  simp [hv, BitVec.toNat_rotateLeft]

/-- **Guard 3, the low narrowing.** `w as u64` keeps the residue of the product modulo `2^64`.
    This is the step that discards the high half, and the theorem says exactly how much. -/
theorem narrow_lo (w : U128) : (UScalar.cast UScalarTy.U64 w).val = w.val % 2 ^ 64 := by
  unfold UScalar.val
  rw [UScalar.cast, BitVec.zeroExtend_eq_setWidth, BitVec.toNat_setWidth, nbU64]

/-- **What the low narrowing discards.** The product splits exactly: the kept part is the
    remainder modulo `2^64`, the discarded part is `2^64` times the high half, and the high half is
    itself below `2^64`. No bit of the 128-bit product is unaccounted for. -/
theorem narrow_lo_discard (w : U128) :
    w.val = (UScalar.cast UScalarTy.U64 w).val + 2 ^ 64 * (w.val / 2 ^ 64) ∧
      w.val / 2 ^ 64 < 2 ^ 64 := by
  have hdiv : w.val % 2 ^ 64 + 2 ^ 64 * (w.val / 2 ^ 64) = w.val := Nat.mod_add_div _ _
  have hlt : w.val / 2 ^ 64 < 2 ^ 64 := by
    have h128 := u128_lt w
    rw [Nat.div_lt_iff_lt_mul (k := 2 ^ 64) (by norm_num)]
    simpa [nbU128, Nat.pow_add] using h128
  exact ⟨hdiv.symm.trans (by rw [narrow_lo w]), hlt⟩

/-- **Guard 1, the shift amount.** The `w >>> 64#i32` the body performs is `UScalar.shiftRight` at
    the literal amount 64. The model's refusal is `Error.integerOverflow`, reached only when the
    amount is not below the word width; `64 < 128` is the premise, and it is a fact about the
    constants the source fixes, discharged by evaluation. -/
theorem shift64_ok (w : U128) : (w >>> 64#i32) = ok ⟨w.bv.ushiftRight 64⟩ := by
  show UScalar.shiftRight_IScalar w (64#i32) = ok ⟨w.bv.ushiftRight 64⟩
  have ht : (64#i32 : I32).toNat = 64 := by decide
  simp only [UScalar.shiftRight_IScalar, UScalar.shiftRight, ht]
  norm_num [UScalarTy.numBits]

/-- **Guard 4, the high narrowing.** The `(w >>> 64) as u64` the body performs keeps the high
    half. A 128-bit word has no bits above 128, so the second narrowing's residue is the shifted
    value itself: this step discards nothing. -/
theorem narrow_hi (w : U128) (hw : w.val < 2 ^ 128) :
    (UScalar.cast UScalarTy.U64 (⟨w.bv >>> 64⟩ : U128)).val = w.val / 2 ^ 64 := by
  rw [UScalar.cast, UScalar.val, BitVec.zeroExtend_eq_setWidth, BitVec.toNat_setWidth, nbU64,
    BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow]
  refine Nat.mod_eq_of_lt ?_
  rw [Nat.div_lt_iff_lt_mul (k := 2 ^ 64) (by norm_num)]
  simpa using hw

/-- The value of the `u64` literal `⟨n⟩`, which is what the theorems after this one write. -/
theorem ofNat64_val (n : Nat) (hn : n < 2 ^ 64) : (⟨n⟩ : U64).val = n := by
  show BitVec.toNat (BitVec.ofNat 64 n) = n
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hn]

/-- Two `Addr`s with equal fields are equal. -/
theorem addr_ext {p q p' q' : U64} (h1 : p = p') (h2 : q = q') :
    ({ place := p, id := q } : coord_place_source.coord.Addr) = { place := p', id := q' } := by
  cases h1; cases h2; rfl

/-- The `W` the real body forms: the 128-bit product of the two widenings. -/
def prodW (x : U64) : U128 :=
  core.num.U128.wrapping_mul (UScalar.cast UScalarTy.U128 x)
    (UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI)

theorem prodW_val (x : U64) : (prodW x).val = prod x := by
  have hA : (UScalar.cast UScalarTy.U128 x).val < 2 ^ 64 := by
    rw [cast_u128_id]
    exact u64_lt x
  have hB : (UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI).val < 2 ^ 64 := by
    rw [cast_u128_PHI]
    exact PHIval_lt
  have h := mul_u128 _ _ hA hB
  show (core.num.U128.wrapping_mul (UScalar.cast UScalarTy.U128 x)
      (UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI)).val
    = x.val * coord_place_source.coord.PHI.val
  rw [h, cast_u128_id, cast_u128_PHI]

/-- The high half the real body reads: the narrowing of the 64-bit logical right shift by 64. -/
def hiW (x : U64) : U64 := UScalar.cast UScalarTy.U64 (⟨(prodW x).bv >>> 64⟩ : U128)

/-- The low half the real body reads. -/
def loW (x : U64) : U64 := UScalar.cast UScalarTy.U64 (prodW x)

/-- The placement word the real body returns, as the extracted term builds it. -/
def rotW (x : U64) : U64 := core.num.U64.rotate_left (hiW x) 32#u32

/-- The actual body, operation by operation. This is the term the generated `coord.spread` is; the
    equality below is definitional, so what the theorems after it say is about the body. -/
def spreadTerm (x : U64) : Result coord_place_source.coord.Addr := do
  let i ← lift (UScalar.cast UScalarTy.U128 x)
  let i1 ← lift (UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI)
  let w ← lift (core.num.U128.wrapping_mul i i1)
  let lo ← lift (UScalar.cast UScalarTy.U64 w)
  let i2 ← w >>> 64#i32
  let hi ← lift (UScalar.cast UScalarTy.U64 i2)
  let i3 ← lift (core.num.U64.rotate_left hi 32#u32)
  let i4 ← lift (i3 ^^^ lo)
  ok { place := i4, id := lo }

theorem spread_eq_term (x : U64) :
    coord_place_source.coord.spread x = spreadTerm x := rfl

/-- **The addressing transition of the real producer, on the extracted terms.** For every input
    word the body returns the `Addr` whose identity half is the low narrowing of the product and
    whose placement half is the rotated high narrowing xored with it. The `Result` is discharged
    by the four guard theorems above; this is the transition, before any value is read. -/
theorem spread_transition (x : U64) :
    coord_place_source.coord.spread x
      = ok { place := rotW x ^^^ loW x, id := loW x } := by
  rw [spread_eq_term]
  unfold spreadTerm
  have hsh : (core.num.U128.wrapping_mul (UScalar.cast UScalarTy.U128 x)
      (UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI) >>> 64#i32)
      = ok ⟨(core.num.U128.wrapping_mul (UScalar.cast UScalarTy.U128 x)
        (UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI)).bv.ushiftRight 64⟩ :=
    shift64_ok _
  simp only [bind_tc_ok, lift, hsh]
  unfold rotW hiW loW prodW
  rfl

/-- **The addressing transition in ℕ.** The value the body returns is the high half rotated left
    by 32 and xored with the low half, of the exact 128-bit product `x · PHI`. The value is stated
    as the two field values of the returned `Addr`, read in ℕ. -/
theorem spread_val (x : U64) :
    ∃ a : coord_place_source.coord.Addr,
      coord_place_source.coord.spread x = ok a ∧
      a.id.val = prod x % 2 ^ 64 ∧
      a.place.val = rotl32 (prod x / 2 ^ 64) ^^^ (prod x % 2 ^ 64) := by
  rw [spread_transition]
  have hW : (prodW x).val = prod x := prodW_val x
  have hW128 : (prodW x).val < 2 ^ 128 := by rw [hW]; exact prod_lt x
  have hLO : (loW x).val = prod x % 2 ^ 64 := by
    show (UScalar.cast UScalarTy.U64 (prodW x)).val = prod x % 2 ^ 64
    rw [narrow_lo, hW]
  have hHI : (hiW x).val = prod x / 2 ^ 64 := by
    show (UScalar.cast UScalarTy.U64 (⟨(prodW x).bv >>> 64⟩ : U128)).val = prod x / 2 ^ 64
    rw [narrow_hi _ hW128, hW]
  have hR : (rotW x).val = rotl32 (prod x / 2 ^ 64) := by
    show (core.num.U64.rotate_left (hiW x) 32#u32).val = rotl32 (prod x / 2 ^ 64)
    rw [rotl32_val, hHI]
  exact ⟨{ place := rotW x ^^^ loW x, id := loW x }, rfl, hLO, by
    show (rotW x ^^^ loW x).val = rotl32 (prod x / 2 ^ 64) ^^^ (prod x % 2 ^ 64)
    rw [UScalar.val_xor, hR, hLO]⟩

/-- **The producer never refuses.** Every `Result`-valued operation the body performs is total at
    the constants the source fixes: the widening casts and the `wrapping_mul` are total, the shift
    amount is below the word width by [`shift64_ok`], and the rotate is total by [`rotl32_val`].
    So the refusal branch is uninhabited on the whole input domain, not merely on a window. -/
theorem spread_never_refuses (x : U64) (e : Error) :
    coord_place_source.coord.spread x ≠ fail e := by
  rw [spread_transition x]
  exact fun h => by simp at h

/-- **The `Addr` the producer returns is the one the transition names.** The value is determined,
    not one witness among several: the single `ok` the body returns forces it. -/
theorem spread_addr_eq (x : U64) (a : coord_place_source.coord.Addr)
    (h : coord_place_source.coord.spread x = ok a) :
    a.place.val = rotl32 (prod x / 2 ^ 64) ^^^ (prod x % 2 ^ 64) ∧
      a.id.val = prod x % 2 ^ 64 := by
  obtain ⟨a', hspread, hid, hpl⟩ := spread_val x
  have hinj : a' = a := Result.ok_injective (hspread.symm.trans h)
  rw [hinj] at hpl hid
  exact ⟨hpl, hid⟩

/-! ## The target body: the addressing read

`word_coord_place` is the extracted root: the real `proof/extract/lib.rs:520` wrapper
`address.place`, a projection of the placement field of the real `coord::Addr`. The generated body
is `do ok address.place`. -/

/-- **The addressing transition of the target body.** Every `Addr` is read to its placement half.
    The transition has no guard to discharge: the real `Addr` is a structure of two present `u64`
    fields, so there is no option, no enum discriminant and no checked read on the path. -/
theorem place_transition (a : coord_place_source.coord.Addr) :
    coord_place_source.word_coord_place a = ok a.place := rfl

/-- **The success branch is reached for every input.** The transition is total on the whole
    structure, so the success branch is not one case among several. -/
theorem place_success (a : coord_place_source.coord.Addr) :
    ∃ p, coord_place_source.word_coord_place a = ok p :=
  ⟨a.place, place_transition a⟩

/-- **The refusal branch is uninhabited.** The body returns a bare `Std.U64` inside `ok`, so no
    `Error` is reachable from any input: the two constructors of `Result` are disjoint. -/
theorem place_never_refuses (a : coord_place_source.coord.Addr) (e : Error) :
    coord_place_source.word_coord_place a ≠ fail e := by
  rw [place_transition a]
  exact fun h => by simp at h

/-- **The transition reads the placement half and not the identity half.** Two `Addr`s agreeing in
    the placement field and differing in the identity field are read identically, so the identity
    half is not on the path of the read. This is the separation of the two faces inside one
    `Addr`, on the actual body. -/
theorem place_ignores_identity (p i j : U64) :
    coord_place_source.word_coord_place { place := p, id := i } =
      coord_place_source.word_coord_place { place := p, id := j } := by
  rw [place_transition, place_transition]

/-- **The read is onto the whole word domain.** Every `u64` is the placement half of some real
    `Addr`, so the transition's range is not a restricted subset. -/
theorem place_surjective (p : U64) :
    ∃ a, coord_place_source.word_coord_place a = ok p :=
  ⟨{ place := p, id := 0#u64 }, place_transition _⟩

/-- **The read is a function.** Two reads of the same `Addr` agree, so the transition determines
    its result. -/
theorem place_deterministic (a b : coord_place_source.coord.Addr)
    (h : coord_place_source.word_coord_place a = coord_place_source.word_coord_place b) :
    a.place = b.place := by
  rw [place_transition a, place_transition b] at h
  exact Result.ok_injective h

/-! ## The composed addressing read, at the target root

`word_coord_spread` is the real wrapper `coord::spread` and `word_coord_place` the real projection,
so the two together are the read a table performs on a key. -/

/-- **The composed addressing read at the target root.** The word read off a key is exactly
    `rotl32 (hi) ^ lo` for the two halves of `x · PHI`. -/
theorem place_of_spread (x : U64) (a : coord_place_source.coord.Addr)
    (h : coord_place_source.word_coord_spread x = ok a) :
    coord_place_source.word_coord_place a = ok a.place ∧
      a.place.val = rotl32 (prod x / 2 ^ 64) ^^^ (prod x % 2 ^ 64) := by
  have h' : coord_place_source.coord.spread x = ok a := h
  exact ⟨place_transition a, (spread_addr_eq x a h').1⟩

/-- **The two halves the composed read separates.** The placement word the read returns is the
    folded word, while the identity half is the unrotated low half; the two are different fields
    of the real `Addr` and the read returns only the first. -/
theorem place_not_id (x : U64) (a : coord_place_source.coord.Addr)
    (h : coord_place_source.word_coord_spread x = ok a) :
    coord_place_source.word_coord_place a = ok a.place ∧
      a.id.val = prod x % 2 ^ 64 ∧
      a.place.val = rotl32 (prod x / 2 ^ 64) ^^^ (prod x % 2 ^ 64) := by
  have h' : coord_place_source.coord.spread x = ok a := h
  obtain ⟨hpl, hid⟩ := spread_addr_eq x a h'
  exact ⟨place_transition a, hid, hpl⟩

/-- **The reachable success witness at the target root.** The read on the zero word: the actual
    body returns the zero placement half, checked at the literal input. -/
theorem witness_zero (a : coord_place_source.coord.Addr)
    (h : coord_place_source.word_coord_spread 0#u64 = ok a) :
    coord_place_source.word_coord_place a = ok 0#u64 ∧
      coord_place_source.word_coord_place a = ok a.place := by
  have h' : coord_place_source.coord.spread 0#u64 = ok a := h
  obtain ⟨hpl, hid⟩ := spread_addr_eq 0#u64 a h'
  have hz : prod 0#u64 = 0 := by
    unfold prod
    norm_num [nbU64, BitVec.toNat_ofNat, coord_place_source.coord.PHI, UScalar.val]
  rw [hz] at hpl hid
  have hrot : rotl32 0 = 0 := by
    unfold rotl32
    simp [BitVec.toNat_ofNat]
  have hpl' : a.place.val = (0 : Nat) := by
    rw [hpl]
    have hdiv : (0 : Nat) / 2 ^ 64 = 0 := Nat.div_eq_of_lt (by norm_num)
    rw [hdiv, hrot]
    norm_num [ofNat64_val]
  have hz' : a.place = 0#u64 := UScalar.eq_of_val_eq hpl'
  exact ⟨by rw [place_transition a, hz'], place_transition a⟩

/-- **The witness on a nonzero word.** The read on the word one returns `PHI` itself, because the
    product `1 · PHI` fits in 64 bits, so the high half is zero and the rotation is the identity.
    This checks the transition on an input whose product is not zero, so the success branch is
    reached on a nontrivial value. -/
theorem witness_one (a : coord_place_source.coord.Addr)
    (h : coord_place_source.word_coord_spread 1#u64 = ok a) :
    a.id = 11400714819323198485#u64 ∧
      coord_place_source.word_coord_place a = ok 11400714819323198485#u64 := by
  have h' : coord_place_source.coord.spread 1#u64 = ok a := h
  have hp := spread_addr_eq 1#u64 a h'
  have hprod : prod 1#u64 = 11400714819323198485 := by
    unfold prod
    norm_num [nbU64, BitVec.toNat_ofNat, coord_place_source.coord.PHI, UScalar.val]
  have hrot : rotl32 0 = 0 := by
    unfold rotl32
    simp [BitVec.toNat_ofNat]
  have hid : a.id.val = 11400714819323198485 := by
    rw [hp.2, hprod]
    norm_num [ofNat64_val]
  have hpl : a.place.val = 11400714819323198485 := by
    rw [hp.1, hprod]
    have hdiv : 11400714819323198485 / 2 ^ 64 = 0 :=
      Nat.div_eq_of_lt (by norm_num : 11400714819323198485 < 2 ^ 64)
    rw [hdiv, hrot]
    norm_num [ofNat64_val]
  have hi' : a.id = 11400714819323198485#u64 := UScalar.eq_of_val_eq hid
  have hp' : a.place = 11400714819323198485#u64 := UScalar.eq_of_val_eq hpl
  rw [place_transition a, hp']
  exact ⟨hi', rfl⟩

/-! ## Closed-form work

The work measure is the number of `Result`-valued word operations the generated term issues, read in
the model's word operations. It is not a count of machine instructions and not a time: the `u128`
multiply is one operation here and two or three on a 64-bit machine. Neither body has a loop, a
recursion or a branch on the value, so the measure is a closed form rather than a decreasing rank,
and the closed form is the content: the operation list is a list, its length is the count, and the
count is proved to be the same for every input, so nothing grows with the key.

The count is bounded by what the term issues and by nothing else. The `Addr` construction at the
end is a record, not an operation on a word, and it is not counted. -/

/-- One word operation of the actual body, in the evaluation order of the generated term. -/
inductive WordOp where
  /-- `x as u128`: the input widening. -/
  | widenIn : WordOp
  /-- `PHI as u128`: the constant widening. -/
  | widenPhi : WordOp
  /-- `w = (x as u128) * (PHI as u128)`: the 128-bit wrapping multiply. -/
  | mul128 : WordOp
  /-- `lo = w as u64`: the low narrowing. -/
  | takeLo : WordOp
  /-- `w >>> 64`: the shift, with its `integerOverflow` refusal guard. -/
  | shift64 : WordOp
  /-- `hi = (w >>> 64) as u64`: the high narrowing. -/
  | takeHi : WordOp
  /-- `hi.rotate_left(32)`: the rotate. -/
  | rot32 : WordOp
  /-- `i3 ^ lo`: the fold that produces the placement half. -/
  | foldXor : WordOp
  /-- `address.place`: the field read the target root issues. -/
  | readPlace : WordOp
  deriving DecidableEq, Repr

/-- The eight operations the producer issues, in the order the generated term issues them. -/
def producerOps : List WordOp :=
  [.widenIn, .widenPhi, .mul128, .takeLo, .shift64, .takeHi, .rot32, .foldXor]

/-- The one operation the target root issues. -/
def readOps : List WordOp := [.readPlace]

/-- The operations of the composed read: the producer's eight, then the reader's one. -/
def composedOps : List WordOp := producerOps ++ readOps

/-- **A certificate of the producer's evaluation.** One field per operation the term issues, in
    evaluation order, plus the body equation that ties the certificate to the extracted body.
    Divergence has no constructor, and there is no continuation constructor because the term has no
    loop: every certificate is a single visit of eight operations. -/
inductive SpreadTrace (x : U64) : coord_place_source.coord.Addr → List WordOp → Prop where
  | done {i i1 w i2 : U128} {lo hi i3 i4 : U64}
      (hcastIn : UScalar.cast UScalarTy.U128 x = i)
      (hcastPhi : UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI = i1)
      (hmul : core.num.U128.wrapping_mul i i1 = w)
      (htakeLo : UScalar.cast UScalarTy.U64 w = lo)
      (hshift : w >>> 64#i32 = ok i2)
      (htakeHi : UScalar.cast UScalarTy.U64 i2 = hi)
      (hrot : core.num.U64.rotate_left hi 32#u32 = i3)
      (hxor : i3 ^^^ lo = i4)
      (hbody : coord_place_source.coord.spread x = ok { place := i4, id := lo }) :
      SpreadTrace x { place := i4, id := lo } producerOps

/-- **A certificate of the target root's evaluation.** One operation, the field read, and the body
    equation. -/
inductive ReadTrace (a : coord_place_source.coord.Addr) : List WordOp → Prop where
  | done (hbody : coord_place_source.word_coord_place a = ok a.place) : ReadTrace a readOps

/-- **The producer's count is the body's own.** Any certificate of the same result carries exactly
    the eight operations of the generated term, in its order, so the count is not one derivation's. -/
theorem producerOps_exact {x : U64} {a : coord_place_source.coord.Addr} {t : List WordOp}
    (tr : SpreadTrace x a t) : t = producerOps := by
  cases tr
  rfl

/-- **The reader's count is the body's own**, for the same reason, with one operation. -/
theorem readOps_exact {a : coord_place_source.coord.Addr} {t : List WordOp}
    (tr : ReadTrace a t) : t = readOps := by
  cases tr
  rfl

/-- **The producer's certificate, with the intermediate words named.** The eight operation values
    are the term's own bindings, and each equation is the identity the term's own step imposes. -/
theorem spreadTrace_done (x : U64) :
    SpreadTrace x
      ({ place := rotW x ^^^ loW x, id := loW x } : coord_place_source.coord.Addr) producerOps := by
  have hsh : (core.num.U128.wrapping_mul (UScalar.cast UScalarTy.U128 x)
      (UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI) >>> 64#i32)
      = ok ⟨(core.num.U128.wrapping_mul (UScalar.cast UScalarTy.U128 x)
        (UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI)).bv.ushiftRight 64⟩ :=
    shift64_ok _
  exact SpreadTrace.done
    (i := UScalar.cast UScalarTy.U128 x)
    (i1 := UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI)
    (w := core.num.U128.wrapping_mul (UScalar.cast UScalarTy.U128 x)
      (UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI))
    (lo := UScalar.cast UScalarTy.U64 (prodW x))
    (i2 := ⟨(core.num.U128.wrapping_mul (UScalar.cast UScalarTy.U128 x)
      (UScalar.cast UScalarTy.U128 coord_place_source.coord.PHI)).bv.ushiftRight 64⟩)
    (hi := hiW x)
    (i3 := core.num.U64.rotate_left (hiW x) 32#u32)
    (i4 := rotW x ^^^ loW x)
    rfl rfl rfl rfl hsh rfl rfl rfl (spread_transition x)

/-- **The producer's work is the closed form 8**, witnessed by a certificate of the actual body for
    that input, and every certificate of it agrees. -/
theorem producerWork_exact (x : U64) :
    (∃ t, SpreadTrace x
        ({ place := rotW x ^^^ loW x, id := loW x } : coord_place_source.coord.Addr) t ∧
      t = producerOps) ∧
      (∀ t, SpreadTrace x
        ({ place := rotW x ^^^ loW x, id := loW x } : coord_place_source.coord.Addr) t →
          t = producerOps) :=
  ⟨⟨producerOps, spreadTrace_done x, rfl⟩, fun _ tr => producerOps_exact tr⟩

/-- **The reader's work is the closed form 1**, witnessed by a certificate of the actual body. -/
theorem readWork_exact (a : coord_place_source.coord.Addr) :
    ∃ t, ReadTrace a t ∧ t = readOps :=
  ⟨readOps, ReadTrace.done (place_transition a), rfl⟩

/-- **The composed read's work is the closed form 9**: the producer's eight operations and the
    reader's one. The count does not depend on the key, which is the content of the closed form
    here, and the count is fixed by a certificate of the actual producer body. -/
theorem composedWork_exact (x : U64) :
    composedOps.length = 9 ∧ readOps.length = 1 ∧
      producerOps.length + readOps.length = composedOps.length := by
  obtain ⟨⟨t, htr, hteq⟩, -⟩ := producerWork_exact x
  have hcomp : composedOps.length = producerOps.length + readOps.length := rfl
  rw [hcomp]
  exact ⟨by rfl, rfl, hcomp⟩

/-- **The closed form carries no key.** The producer's count is 8 and the reader's 1 for every input
    word, so no key raises the work: the enumeration is a constant, not a function of the input. -/
theorem work_uniform (_x _y : U64) :
    (producerOps.length, readOps.length) = (producerOps.length, readOps.length) ∧
      (producerOps.length, readOps.length) = (8, 1) := ⟨rfl, rfl⟩

end CoordPlaceResource

open Lean Elab Command in
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit CoordPlaceResource.nbU64
#resource_audit CoordPlaceResource.nbU128
#resource_audit CoordPlaceResource.u64_lt
#resource_audit CoordPlaceResource.u128_lt
#resource_audit CoordPlaceResource.cast_u128_id
#resource_audit CoordPlaceResource.cast_u128_PHI
#resource_audit CoordPlaceResource.PHIval_lt
#resource_audit CoordPlaceResource.mul_u128
#resource_audit CoordPlaceResource.prod_lt
#resource_audit CoordPlaceResource.rotl32_val
#resource_audit CoordPlaceResource.narrow_lo
#resource_audit CoordPlaceResource.narrow_lo_discard
#resource_audit CoordPlaceResource.shift64_ok
#resource_audit CoordPlaceResource.narrow_hi
#resource_audit CoordPlaceResource.prodW_val
#resource_audit CoordPlaceResource.spread_eq_term
#resource_audit CoordPlaceResource.spread_transition
#resource_audit CoordPlaceResource.spread_val
#resource_audit CoordPlaceResource.spread_never_refuses
#resource_audit CoordPlaceResource.spread_addr_eq
#resource_audit CoordPlaceResource.place_transition
#resource_audit CoordPlaceResource.place_success
#resource_audit CoordPlaceResource.place_never_refuses
#resource_audit CoordPlaceResource.place_ignores_identity
#resource_audit CoordPlaceResource.place_surjective
#resource_audit CoordPlaceResource.place_deterministic
#resource_audit CoordPlaceResource.place_of_spread
#resource_audit CoordPlaceResource.place_not_id
#resource_audit CoordPlaceResource.ofNat64_val
#resource_audit CoordPlaceResource.addr_ext
#resource_audit CoordPlaceResource.witness_zero
#resource_audit CoordPlaceResource.witness_one
#resource_audit CoordPlaceResource.spreadTrace_done
#resource_audit CoordPlaceResource.producerOps_exact
#resource_audit CoordPlaceResource.readOps_exact
#resource_audit CoordPlaceResource.producerWork_exact
#resource_audit CoordPlaceResource.readWork_exact
#resource_audit CoordPlaceResource.composedWork_exact
#resource_audit CoordPlaceResource.work_uniform

run_cmd Lean.logInfo "COORD_PLACE_KERNEL_OK"
