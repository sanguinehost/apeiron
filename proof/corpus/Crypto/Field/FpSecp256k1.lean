/-
`GF(p)` for secp256k1, `p = 2^256 - 2^32 - 977`, as `ZMod p` over the pinned Mathlib
`Mathlib.Algebra.Field.ZMod`.

This module mirrors `work/crypto-field-secp256k1` (`src/lib.rs`), whose elements are four
little-endian 64-bit limbs held canonically in `[0, p)` and whose intermediate products use
`u128` accumulators. What is modelled here is the mathematical content, not the limb
arithmetic: the constant `p`, the reduction constant `C = 2^32 + 977` with `2^256 = C (mod p)`,
the `invert` exponent `p - 2`, and the canonical 32-byte big-endian encode/decode
(`to_bytes` / `from_bytes`). Curve constants (`b = 7`, `work/crypto-curve-secp256k1/src/lib.rs:17`)
and affine curve points, scalar multiplication and the generator `G` live outside this crate and
are deliberately not restated as curve objects; `curveB` below is the field-side reading of the
constant only. The Rust crate contains **no** square root: `secpSqrt` here is the standard
`p = 3 (mod 4)` algorithm stated for the same field, not a mirror of a Rust function.

Two things are separated throughout:

* Everything about `ℕ`, byte lists and `ZMod p` **values** (canonicalize, encode, decode, the
  reduction pins, `p`'s shape) is proved unconditionally, because it needs no field instance.
* Everything that needs a `Field (ZMod p)` instance (`inv`, `pow`, `sqrt`, the field laws) is
  stated for an arbitrary prime `q` under `[Fact q.Prime]`, in the same generic style as
  `Apeiron.Crypto.Field.ZModPrime`, and instantiated at `q := p` only through the documented
  `Prop` `secpPrimeCert`. Supplying that certificate needs `Nat.Prime (2^256 - 2^32 - 977)`,
  which this file does not prove; see `secpPrimeCert` for the obligation and the route. No
  inhabitant is fabricated, and nothing below is proved by assuming it.
-/
import Mathlib.Algebra.Field.ZMod
import Mathlib.Tactic

set_option autoImplicit false
set_option warningAsError true

-- The pins below are `decide` proofs over 256-bit numerals read through `UInt64.toNat` and
-- `UInt8.toNat`; the default kernel recursion limit is too small to evaluate them. Raising the
-- limit is not a proof-weakening option: `decide` still reduces to `True`/`False` under the same
-- kernel, and no `sorry`, `axiom` or `admit` is involved.
set_option maxRecDepth 40000

namespace Apeiron.Crypto.Field.Secp256k1

/-! ## The modulus and its shape

`P` in the Rust crate is little-endian, `P_BYTES` big-endian; both are restated here as the
literal 4-limb and 32-byte forms, and each is proved to reconstruct `p` under the corresponding
reading. -/

/-- The secp256k1 base field modulus `p = 2^256 - 2^32 - 977`. Cost: `O(1)`, the value is a
closed numeral. -/
def p : ℕ := 2 ^ 256 - 2 ^ 32 - 977

/-- The reduction constant `C = 2^32 + 977` (`src/lib.rs:27`, `0x1_0000_03D1`), so that
`2^256 ≡ C (mod p)`. Cost: `O(1)`. -/
def reductionConst : ℕ := 2 ^ 32 + 977

/-- Curve constant `b` of `y^2 = x^3 + ax + b` with `a = 0`, as the Rust `B` of
`work/crypto-curve-secp256k1/src/lib.rs:17`. It is a field element of `ZMod p`; nothing in this
file is a curve. Cost: `O(1)`. -/
def curveB : ℕ := 7

/-- Reading a little-endian limb list as a natural number: `Σ lᵢ * 2^(64 i)`. Cost: `O(k)` steps
for a `k`-limb list. -/
def limbsToNat : List UInt64 → ℕ
  | [] => 0
  | l :: ls => l.toNat + limbsToNat ls * 2 ^ 64

/-- Reading a big-endian byte list as a natural number: `Σ bᵢ * 2^(8 (k - 1 - i))` over a
`k`-byte list, `b₀` the most significant byte, which is exactly the `u64::from_be_bytes` reading
of `from_bytes`. Recurrence `f [] = 0`, `f (b :: bs) = b * 2^(8 * |bs|) + f bs`.

Cost: `k` steps for a `k`-byte list, each one `Nat` multiplication by `2^(8 j)` and one addition,
counted in `Nat` multiplications of at most 256-bit operands and one live `Nat` per recursion
level, so `O(k)` multiplications and `O(k)` stack words; the multiplication by `2^(8 j)` is itself
`O(log j)` shifts, so a full 32-byte read is 32 such products. `O(1)` heap words beyond the
result. -/
def beNat : List UInt8 → ℕ
  | [] => 0
  | b :: bs => b.toNat * 2 ^ (8 * bs.length) + beNat bs

@[simp] theorem beNat_nil : beNat ([] : List UInt8) = 0 := rfl

@[simp] theorem beNat_cons (b : UInt8) (bs : List UInt8) :
    beNat (b :: bs) = b.toNat * 2 ^ (8 * bs.length) + beNat bs := rfl

@[simp] theorem beNat_singleton (b : UInt8) : beNat [b] = b.toNat := by
  rw [beNat_cons]
  simp

@[simp] theorem limbsToNat_nil : limbsToNat ([] : List UInt64) = 0 := rfl

@[simp] theorem limbsToNat_cons (l : UInt64) (ls : List UInt64) :
    limbsToNat (l :: ls) = l.toNat + limbsToNat ls * 2 ^ 64 := rfl

/-- `p` in the hex form the Rust `P` constant encodes. -/
theorem p_eq_hex :
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F := by
  decide

/-- The little-endian limb image `P` (`src/lib.rs:13`) reconstructs `p`. This is the pin that ties
the literal in `src/lib.rs` to the modulus used everywhere below. The kernel interpreter needs
the raised recursion limit to evaluate the 64-bit limb products. -/
theorem limbsToNat_pLimbs :
    limbsToNat [0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF]
      = p := by
  decide

/-- The big-endian byte image `P_BYTES` (`src/lib.rs:21`) reconstructs `p`. -/
theorem beNat_pBytes :
    beNat [
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2F
    ] = p := by
  decide

/-- The `invert` exponent constant `E` (`src/lib.rs:177`, `p - 2` little-endian) reconstructs
`p - 2`. This is the pin on the exponent the Rust double-and-add loop walks over. -/
theorem limbsToNat_eLimbs :
    limbsToNat [0xFFFFFFFEFFFFFC2D, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF]
      = p - 2 := by
  decide

/-- The reduction identity behind `reduce_wide`: `2^256 = p + C`. -/
theorem two_pow_256_eq_add : 2 ^ 256 = p + reductionConst := by
  decide

/-- Known-answer pin for the Rust reduction constant: `2^256 mod p = 2^32 + 977`. -/
theorem two_pow_256_mod_p : 2 ^ 256 % p = reductionConst := by
  decide

theorem p_ne_zero : p ≠ 0 := by decide

theorem two_le_p : 2 ≤ p := by decide

theorem p_lt_two_pow_256 : p < 2 ^ 256 := by decide

/-- The pin the 32-byte decode bound uses: a `2^255 ≤ p` modulus means one conditional
subtraction is enough for any 256-bit input. -/
theorem two_pow_255_le_p : 2 ^ 255 ≤ p := by decide

/-- `2 * p ≥ 2^256`, so any 256-bit input needs at most one conditional subtraction. -/
theorem two_pow_256_le_two_mul_p : 2 ^ 256 ≤ 2 * p := by decide

theorem odd_p : Odd p := by decide

theorem mod_four_p : p % 4 = 3 := by decide

theorem p_mod_two_pow_32 : p % 2 ^ 32 = 2 ^ 32 - 977 := by decide

/-- File-local: `ZMod p`'s `natCast` lemmas need it, and it is available unconditionally from
`p_ne_zero`. It is deliberately not a global `NeZero` instance for every modulus. -/
local instance instNeZeroP : NeZero p := ⟨p_ne_zero⟩

/-! ## Primality: the one input this file does not supply

`Field (ZMod p)` and the `Nat.Prime`-derived laws all require `[Fact p.Prime]`. Mathlib's
`Decidable (Nat.Prime n)` is trial division, so `by decide` cannot decide primality of a 256-bit
number in a gate. The obligation is therefore left open and stated, not filled in:
`secpPrimeCert` is the exact goal, and `secpFieldLawsAtP` is the documented `Prop` that names
everything which becomes available once it is inhabited. -/

/-- The open obligation: `p = 2^256 - 2^32 - 977` is prime. Nothing in this file proves it, and
no later declaration is allowed to consume it without that proof.

The route is a certificate, not a search: `Mathlib.NumberTheory.LucasPrimality.lucas_primality`
discharges `Nat.Prime p` from a generator `a` of `(ZMod p)ˣ`, i.e. from
`a^(p-1) = 1 (mod p)` together with `a^((p-1)/q) ≠ 1 (mod p)` for every prime `q` dividing
`p - 1 = 2 * (2^255 - 2^31 - 489)`. Both sides are decidable by repeated squaring modulo `p`,
and the `q`s are the prime divisors of that odd cofactor. The factorization is not carried out
here, and no inhabitant is fabricated. -/
def secpPrimeCert : Prop := Nat.Prime p

/-- The only constructor of the missing `[Fact p.Prime]` instance: an inhabitant of
`secpPrimeCert` turns into `Fact (Nat.Prime p)`, and with it the field laws of `ZMod p` become
available. Marked `@[reducible]` so that this `def` is not itself collected as an instance, which
would make the open obligation look discharged. -/
@[reducible] def secpFact (h : secpPrimeCert) : Fact (Nat.Prime p) := ⟨h⟩

/-! ## Reading bytes as natural numbers

`beNat` is unbounded over long lists, so the 32-byte facts are stated against a length bound and
the bound is what the encode side proves. -/

/-- Byte shift: one more byte multiplies the read value by `256`. -/
theorem two_pow_8_succ_mul (j : ℕ) : 2 ^ (8 * (j + 1)) = 2 ^ (8 * j) * 256 := by
  rw [show 8 * (j + 1) = 8 * j + 8 by omega, pow_add, show (2 : ℕ) ^ 8 = 256 by norm_num]

/-- `beNat` of a byte list of its own length fits in `2^(8 * length)`: a `k`-byte read is below
`2^(8 k)`. Induction on the list, the step closing because the leading byte is below `2^8`. -/
theorem beNat_lt_pow_len (bs : List UInt8) : beNat bs < 2 ^ (8 * bs.length) := by
  induction bs with
  | nil => simp
  | cons b bs ih =>
      rw [beNat_cons, List.length_cons, two_pow_8_succ_mul]
      have hb8 : b.toNat < 256 := by
        have := UInt8.toNat_lt b
        norm_num at this
        exact this
      have hle : b.toNat * 2 ^ (8 * bs.length) ≤ 255 * 2 ^ (8 * bs.length) :=
        Nat.mul_le_mul_right _ (by omega)
      omega

/-- The 32-byte bound the decode side uses. -/
theorem beNat_lt_two_pow_256 (bs : List UInt8) (h : bs.length ≤ 32) : beNat bs < 2 ^ 256 := by
  have hb := beNat_lt_pow_len bs
  have hexp : 8 * bs.length ≤ 256 := by omega
  have hp := Nat.pow_le_pow_right (n := 2) (by norm_num) hexp
  omega

/-- The bound the single conditional subtraction of `canonicalize` rests on: any value below
`2^256` is below `2 * p`, so `v - p < p` whenever `p ≤ v < 2 * p`. Stated with `p` left abstract so
the later `omega` steps see `p` as an atom rather than a reduced numeral. -/
theorem lt_two_mul_p_of_lt_two_pow_256 {v : ℕ} (h : v < 2 ^ 256) : v < 2 * p := by
  have h255 : 2 ^ 255 ≤ p := two_pow_255_le_p
  omega

/-! ## Canonical 32-byte encode and decode -/

/-- Split a natural number into bytes, least significant first, each `n mod 256` with the
remaining `n / 256`; `natToBytes 0 = []`. Cost: one division and one remainder per byte,
`O(log₂ n / 8)` steps. `UInt8.ofNat` is the `% 256` the recurrence already applies. -/
def natToBytes : ℕ → List UInt8
  | 0 => []
  | n + 1 => UInt8.ofNat ((n + 1) % 256) :: natToBytes ((n + 1) / 256)

/-- Reading a list least significant byte first: `Σ bᵢ * 2^(8 i)` with `b₀` the least
significant byte. `beNat` reads the same list most significant byte first, so `beNat (l.reverse)`
is `leNat l`; `natToBytes` produces least-significant-first order. -/
def leNat : List UInt8 → ℕ
  | [] => 0
  | b :: bs => b.toNat + 256 * leNat bs

@[simp] theorem leNat_nil : leNat ([] : List UInt8) = 0 := rfl

@[simp] theorem leNat_cons (b : UInt8) (bs : List UInt8) :
    leNat (b :: bs) = b.toNat + 256 * leNat bs := rfl

/-- Appending the least significant byte: a big-endian read of `l ++ [b]` is `beNat l` shifted
one byte left plus `b`. This is the positional weighting of `beNat`, proved by induction on `l`. -/
theorem beNat_append_singleton (l : List UInt8) (b : UInt8) :
    beNat (l ++ [b]) = beNat l * 256 + b.toNat := by
  induction l with
  | nil => rw [List.nil_append, beNat_singleton, beNat_nil]; norm_num
  | cons a l ih =>
      rw [List.cons_append, beNat_cons, beNat_cons, ih]
      have hlen : (l ++ [b]).length = l.length + 1 := by simp
      rw [hlen, two_pow_8_succ_mul]
      ring

/-- Reversing the byte order turns the little-endian reading into the big-endian one. -/
theorem beNat_reverse (bs : List UInt8) : beNat bs.reverse = leNat bs := by
  induction bs with
  | nil => simp
  | cons b bs ih =>
      rw [List.reverse_cons, beNat_append_singleton, ih, leNat_cons]
      omega

/-- Splitting a natural number into little-endian bytes and reading it back little-endian is the
identity: `leNat (natToBytes n) = n`. Strong induction on `n`, the measure decreasing because
`(n + 1) / 256 < n + 1`. -/
theorem leNat_natToBytes (n : ℕ) : leNat (natToBytes n) = n := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
      rcases Nat.eq_zero_or_pos n with rfl | hpos
      · simp [natToBytes]
      · obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hpos)
        have hrec : (m + 1) / 256 < m + 1 := by
          have := Nat.div_lt_self (n := m + 1) (k := 256) (by norm_num) (by norm_num)
          omega
        have htail := ih ((m + 1) / 256) hrec
        have hdiv : (m + 1) % 256 + 256 * ((m + 1) / 256) = m + 1 := Nat.mod_add_div _ _
        have hhead : (UInt8.ofNat ((m + 1) % 256)).toNat = (m + 1) % 256 := by
          have := UInt8.toNat_ofNat (n := (m + 1) % 256)
          norm_num at this
          exact this
        rw [natToBytes, leNat_cons, hhead, htail, hdiv]

/-- The encode round trip: the canonical bytes of `n` read back big-endian are `n`. -/
theorem beNat_natToBytes_reverse (n : ℕ) : beNat (natToBytes n).reverse = n :=
  beNat_reverse _ |>.trans (leNat_natToBytes n)

/-- Canonical reduction of a 256-bit integer, mirroring `from_bytes`: the value itself when it is
below `p`, otherwise `value - p`. Cost: one comparison and at most one subtraction, `O(1)`. -/
def canonicalize (v : ℕ) : ℕ := if v < p then v else v - p

@[simp] theorem canonicalize_of_lt {v : ℕ} (h : v < p) : canonicalize v = v := if_pos h

@[simp] theorem canonicalize_of_not_lt {v : ℕ} (h : ¬ v < p) : canonicalize v = v - p :=
  if_neg h

/-- The canonical 32-byte big-endian encoding of a field element, mirroring `to_bytes`: the
canonical representative `a.val` split into bytes and reversed into big-endian order. Cost:
`O(log₂ p / 8) = 32` split steps plus one reversal. -/
def encode (a : ZMod p) : List UInt8 := (natToBytes a.val).reverse

/-- Decoding 32 big-endian bytes, mirroring `from_bytes`: read the bytes big-endian and apply the
single conditional subtraction. Cost: 32 additions and multiplications, `O(1)` extra words. -/
def decodeBytes (bs : List UInt8) : ZMod p := (canonicalize (beNat bs) : ZMod p)

/-- Byte length of `natToBytes n`: at most one byte per eight bits of `n`, i.e.
`n < 2^(8 k)` gives `≤ k` bytes. Strong induction on `n`; the tail `(n + 1) / 256` is below
`2^(8 k')` because `n + 1 < 256 * 2^(8 k')`, so the measure decreases. -/
theorem natToBytes_length_le (n : ℕ) (k : ℕ) (h : n < 2 ^ (8 * k)) : (natToBytes n).length ≤ k := by
  induction n using Nat.strong_induction_on generalizing k with
  | _ n ih =>
      rcases Nat.eq_zero_or_pos n with rfl | hpos
      · simp [natToBytes]
      · obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hpos)
        rcases Nat.eq_zero_or_pos k with rfl | hk
        · rw [show 8 * 0 = 0 by norm_num, pow_zero] at h
          omega
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hk)
          rw [show 8 * (k' + 1) = 8 * k' + 8 by omega, pow_add,
            show (2 : ℕ) ^ 8 = 256 by norm_num] at h
          have hrec : (m + 1) / 256 < m + 1 := by
            have := Nat.div_lt_self (n := m + 1) (k := 256) (by norm_num) (by norm_num)
            omega
          have hlt : (m + 1) / 256 < 2 ^ (8 * k') :=
            Nat.div_lt_of_lt_mul (by simpa [Nat.mul_comm] using h)
          have htail := ih ((m + 1) / 256) hrec k' hlt
          rw [natToBytes]
          simp only [List.length_cons, Nat.succ_le_succ_iff]
          omega

/-- The canonical representative of a field element is at most 32 bytes wide, because
`a.val < p < 2^256`. -/
theorem encode_length_le (a : ZMod p) : (encode a).length ≤ 32 := by
  have h := natToBytes_length_le a.val 32
    (lt_of_lt_of_le (ZMod.val_lt a) (le_of_lt p_lt_two_pow_256))
  simpa [encode, List.length_reverse] using h

/-- The encode side is canonical: those bytes read back to the representative. -/
theorem beNat_encode (a : ZMod p) : beNat (encode a) = a.val := beNat_natToBytes_reverse a.val

/-- Encoding then decoding is the identity on field elements. This is the round-trip law of the
Rust `from_bytes ∘ to_bytes`, and it is unconditional: it only uses `a.val < p`. -/
theorem decodeBytes_encode (a : ZMod p) : decodeBytes (encode a) = a := by
  have hcanon : canonicalize a.val = a.val := canonicalize_of_lt (ZMod.val_lt a)
  simp [decodeBytes, beNat_encode, hcanon]

/-- A decoded value is canonical: its representative is below `p`. This is the `≤ 32`-byte
hypothesis, which `decodeBytes` records by taking a `List` and reading it with `beNat`; the
single conditional subtraction of `canonicalize` suffices because `2^255 ≤ p`. -/
theorem decodeBytes_val_lt (bs : List UInt8) (h : bs.length ≤ 32) : (decodeBytes bs).val < p := by
  have hb2 : beNat bs < 2 * p :=
    lt_two_mul_p_of_lt_two_pow_256 (beNat_lt_two_pow_256 bs h)
  have hcast : ((canonicalize (beNat bs) : ℕ) : ZMod p).val = canonicalize (beNat bs) % p :=
    ZMod.val_natCast _ _
  unfold decodeBytes
  rw [hcast]
  rcases Nat.lt_or_ge (beNat bs) p with hc | hc
  · rw [canonicalize_of_lt hc, Nat.mod_eq_of_lt hc]
    omega
  · rw [canonicalize_of_not_lt (Nat.not_lt.mpr hc), Nat.mod_eq_of_lt (by omega)]
    omega

/-- Decoding reduces its input modulo `p`, which is what the Rust `from_bytes` comment claims
("reducing mod `p`"): for any 256-bit input the single conditional subtraction agrees with the
remainder, because the input is below `2 * p`. -/
theorem decodeBytes_beNat_mod (bs : List UInt8) (h : bs.length ≤ 32) :
    (decodeBytes bs).val = beNat bs % p := by
  have hb2 : beNat bs < 2 * p :=
    lt_two_mul_p_of_lt_two_pow_256 (beNat_lt_two_pow_256 bs h)
  have hcast : ((canonicalize (beNat bs) : ℕ) : ZMod p).val = canonicalize (beNat bs) % p :=
    ZMod.val_natCast _ _
  unfold decodeBytes
  rw [hcast]
  rcases Nat.lt_or_ge (beNat bs) p with hc | hc
  · rw [canonicalize_of_lt hc, Nat.mod_eq_of_lt hc]
  · rw [canonicalize_of_not_lt (Nat.not_lt.mpr hc), Nat.mod_eq_sub_mod hc,
      Nat.mod_eq_of_lt (by omega)]

/-- Known-answer pin on the modulus encoding: `P_BYTES` reads back as `p`, and since `p` is not
below itself the decode subtracts it away. In the Rust crate `Fp::from_bytes(&P_BYTES) == 0`. -/
theorem decodeBytes_pBytes :
    decodeBytes [
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2F
    ] = 0 := by
  rw [decodeBytes, beNat_pBytes]
  rw [canonicalize_of_not_lt (Nat.lt_irrefl p)]
  simp

/-- `curveB = 7` as a field element. -/
def curveBField : ZMod p := (curveB : ZMod p)

@[simp] theorem curveBField_val : curveBField.val = curveB := by
  have hcast : ((curveB : ℕ) : ZMod p).val = curveB % p := ZMod.val_natCast p curveB
  rw [curveBField, hcast]
  have h7 : curveB < 2 ^ 255 := by norm_num [curveB]
  rw [Nat.mod_eq_of_lt (lt_of_lt_of_le h7 two_pow_255_le_p)]

/-! ## Field layer: `inv`, `pow`, `sqrt` at an arbitrary prime

Generic in the prime `q`, exactly as `Apeiron.Crypto.Field.ZModPrime` is: the `Field (ZMod q)`
instance is the pinned Mathlib one, and the statements below are the field laws of the operations
this file defines. Instantiating at `q := p` requires `secpPrimeCert`. -/

section Generic

variable {q : ℕ} [Fact q.Prime]

/-- Executable inverse in the prime field: `ZMod.inv`, the extended-Euclid witness
`Nat.gcdA a.val q`. Cost: one extended-Euclid run, `O(log q)` `Nat` remainder steps. The Rust
`invert` returns zero at zero, and so does this, since a field inverse of `0` is not definable;
see `secpInv_zero`. -/
def secpInv (a : ZMod q) : ZMod q := a⁻¹

@[simp] theorem secpInv_zero : secpInv (0 : ZMod q) = 0 := inv_zero

theorem secpInv_mul_cancel (a : ZMod q) (ha : a ≠ 0) : secpInv a * a = 1 := by
  rw [mul_comm]
  exact mul_inv_cancel₀ ha

theorem secpInv_mul_cancel' (a : ZMod q) (ha : a ≠ 0) : a * secpInv a = 1 := mul_inv_cancel₀ ha

/-- Executable exponentiation by structural recursion, mirroring the left-to-right
double-and-add of the Rust `invert` loop: recurrence `f a 0 = 1`, `f a (e + 1) = a * f a e`, so
`e` multiplications in `ZMod q` and stack depth `e` if not turned into a loop. -/
def secpPow (a : ZMod q) : ℕ → ZMod q
  | 0 => 1
  | e + 1 => a * secpPow a e

@[simp] theorem secpPow_zero (a : ZMod q) : secpPow a 0 = 1 := rfl

theorem secpPow_succ (a : ZMod q) (e : ℕ) : secpPow a (e + 1) = a * secpPow a e := rfl

theorem secpPow_eq (a : ZMod q) (e : ℕ) : secpPow a e = a ^ e := by
  induction e with
  | zero => exact (secpPow_zero a).trans (Monoid.npow_zero a).symm
  | succ e ih =>
      exact (secpPow_succ a e).trans (congrArg (fun z => a * z) ih) |>.trans (pow_succ' a e).symm

theorem secpPow_one (a : ZMod q) : secpPow a 1 = a := by
  rw [secpPow_succ, secpPow_zero, mul_one]

/-- Exponentiation is a monoid homomorphism in the exponent. -/
theorem secpPow_add (a : ZMod q) (m n : ℕ) :
    secpPow a (m + n) = secpPow a m * secpPow a n := by
  simp only [secpPow_eq, pow_add]

theorem secpPow_mul (a : ZMod q) (m n : ℕ) : (secpPow a m) ^ n = secpPow a (m * n) := by
  simp only [secpPow_eq, pow_mul]

/-- Fermat: `a^(q-2)` is the inverse of a nonzero `a`. This is the exponent the Rust `invert`
walks over `E` limb by limb, and `limbsToNat_eLimbs` pins that constant to `p - 2`. -/
theorem secpPow_sub_two_inv (a : ZMod q) (ha : a ≠ 0) : secpPow a (q - 2) = secpInv a := by
  rw [secpPow_eq]
  have hsub : q - 2 + 1 = q - 1 := by
    have hq2 : 2 ≤ q := (Fact.out : Nat.Prime q).two_le
    omega
  refine mul_left_cancel₀ ha ?_
  calc a * a ^ (q - 2) = a ^ (q - 2 + 1) := by rw [pow_succ', mul_comm]
    _ = a ^ (q - 1) := by rw [hsub]
    _ = a * a⁻¹ := by rw [ZMod.pow_card_sub_one_eq_one ha, mul_inv_cancel₀ ha]
    _ = a * secpInv a := by rw [secpInv]

/-- The `q ≡ 3 (mod 4)` square root: `a^((q+1)/4)`. Cost: `O(log q)` multiplications, against
`O(log² q)` for a general modular square root. Not a Rust function: the field crate has none. -/
def secpSqrt (a : ZMod q) : ZMod q := secpPow a ((q + 1) / 4)

/-- The exponent identity the `q ≡ 3 (mod 4)` square root rests on: `2 * ((q+1)/4) = q/2 + 1`. -/
theorem two_mul_sqrt_exp (h : q % 4 = 3) : 2 * ((q + 1) / 4) = q / 2 + 1 := by
  have hq : q ≠ 0 := (Fact.out : Nat.Prime q).ne_zero
  omega

/-- At a modulus `≡ 3 (mod 4)` the value `4` is a unit, so `sqrt` of it has two roots. -/
theorem four_ne_zero_of_mod_four_three (h3 : q % 4 = 3) : (4 : ZMod q) ≠ 0 := by
  have hq2 : 2 ≤ q := (Fact.out : Nat.Prime q).two_le
  intro hz
  have hd4 : q ∣ 4 := (ZMod.natCast_eq_zero_iff 4 q).mp hz
  obtain ⟨k, hk2le, hk2⟩ :=
    (Nat.dvd_prime_pow (p := 2) (by norm_num : Nat.Prime 2) (m := 2)).mp hd4
  interval_cases k <;> norm_num at hk2 <;> omega

/-- **Conditional correctness of `secpSqrt`.** For a modulus with `q ≡ 3 (mod 4)` and an element
`a` that is a quadratic residue, `a^((q+1)/4)` is a square root of `a`.

The residue side condition is the hypothesis `a^(q/2) = 1`, which is Euler's criterion and is
equivalent to `IsSquare a` at `a ≠ 0` (`ZMod.euler_criterion`). It is carried here rather than
assumed away because `secpSqrt` is total: at a nonresidue it returns some value that is not a
root, and the file does not claim otherwise. -/
theorem secpSqrt_sq (a : ZMod q) (h3 : q % 4 = 3) (ha : a ≠ 0) (hres : a ^ (q / 2) = 1) :
    secpSqrt a ^ 2 = a := by
  rw [secpSqrt, secpPow_eq]
  have hexp : ((q + 1) / 4) * 2 = q / 2 + 1 :=
    calc ((q + 1) / 4) * 2 = 2 * ((q + 1) / 4) := Nat.mul_comm _ _
      _ = q / 2 + 1 := two_mul_sqrt_exp h3
  calc (a ^ ((q + 1) / 4)) ^ 2 = a ^ (((q + 1) / 4) * 2) := (pow_mul a _ _).symm
    _ = a ^ (q / 2 + 1) := by rw [hexp]
    _ = a ^ (q / 2) * a ^ 1 := pow_add a _ _
    _ = (a * a⁻¹) * a := by rw [hres]; rw [mul_inv_cancel₀ ha]; ring
    _ = a := by rw [mul_inv_cancel₀ ha]; ring

/-- The residue hypothesis is exactly the `IsSquare` side of Euler's criterion, so this is the
same statement with the hypothesis phrased as squareness. -/
theorem secpSqrt_sq_of_isSquare (a : ZMod q) (h3 : q % 4 = 3) (ha : a ≠ 0)
    (hs : IsSquare a) : secpSqrt a ^ 2 = a :=
  secpSqrt_sq a h3 ha ((ZMod.euler_criterion q ha).mp hs)

/-- Euler's criterion in this file's terms: a nonzero element is a square exactly when
`secpPow a (q / 2) = 1`. -/
theorem isSquare_iff_euler (a : ZMod q) (ha : a ≠ 0) :
    IsSquare a ↔ secpPow a (q / 2) = 1 := by
  rw [secpPow_eq]
  exact ZMod.euler_criterion q ha

/-- Roots of `x^2 = 4` at a modulus where `4` is nonzero: the two roots are `2` and `-2`. This
is the general form of the `sqrt(4) = 2 ∨ sqrt(4) = q - 2` pin. -/
theorem secpSqrt_four_choice (h3 : q % 4 = 3) :
    secpSqrt (4 : ZMod q) = 2 ∨ secpSqrt (4 : ZMod q) = -(2 : ZMod q) := by
  have h4 : (4 : ZMod q) ≠ 0 := four_ne_zero_of_mod_four_three h3
  have hsq : secpSqrt (4 : ZMod q) ^ 2 = 4 :=
    secpSqrt_sq 4 h3 h4
      ((ZMod.euler_criterion q h4).mp ⟨(2 : ZMod q), by norm_num⟩)
  have hkey : (secpSqrt (4 : ZMod q) - 2) * (secpSqrt (4 : ZMod q) + 2) = 0 := by
    calc (secpSqrt (4 : ZMod q) - 2) * (secpSqrt (4 : ZMod q) + 2)
        = secpSqrt (4 : ZMod q) ^ 2 - 4 := by ring
      _ = 0 := by rw [hsq]; ring
  rcases mul_eq_zero.mp hkey with h | h
  · exact Or.inl (sub_eq_zero.mp h)
  · exact Or.inr (add_eq_zero_iff_eq_neg.mp h)

/-- `-2` at a prime modulus `q ≥ 2` is the element read from the representative `q - 2`, so the
disjuncts of `secpSqrt_four_choice` are exactly `sqrt(4) = 2 ∨ sqrt(4) = q - 2`, which at
`q := p` is the pinned statement `sqrt(4) = 2 ∨ sqrt(4) = p - 2`. -/
theorem neg_two_eq_natCast_sub_two (hq2 : 2 ≤ q) :
    (-(2 : ZMod q)) = ((q - 2 : ℕ) : ZMod q) := by
  symm
  rw [Nat.cast_sub hq2, ZMod.natCast_self, zero_sub]
  norm_num

/-- The `sqrt(4)` pin at an arbitrary prime `≡ 3 (mod 4)`, with `q - 2` read as a representative. -/
theorem secpSqrt_four_choice_val (h3 : q % 4 = 3) :
    secpSqrt (4 : ZMod q) = 2 ∨ secpSqrt (4 : ZMod q) = ((q - 2 : ℕ) : ZMod q) :=
  (secpSqrt_four_choice h3).elim Or.inl
    (fun h => Or.inr (h.trans (neg_two_eq_natCast_sub_two (Fact.out : Nat.Prime q).two_le)))

/-- The `sqrt(4)` pin with the representative made explicit: the representative of the second
root is the numeral `q - 2`. -/
theorem secpSqrt_four_val (h3 : q % 4 = 3) :
    secpSqrt (4 : ZMod q) = 2 ∨ (secpSqrt (4 : ZMod q)).val = q - 2 := by
  have hq2 : 2 ≤ q := (Fact.out : Nat.Prime q).two_le
  rcases secpSqrt_four_choice_val h3 with h | h
  · exact Or.inl h
  · refine Or.inr ?_
    rw [h, ZMod.val_natCast, Nat.mod_eq_of_lt (by omega)]

end Generic

/-! ## The concrete instantiation at `p`: stated, not assumed

The theorems below are the generic ones read at `q := p`. They are hypotheses of the section, not
theorems about this modulus: an inhabitant of `secpPrimeCert` is required. `secpFieldLawsAtP`
names the whole conditional content as a `Prop`, and `secpFieldLawsAtP_holds` proves that `Prop`
from the certificate alone without registering any instance. -/

section Concrete

variable [Fact (Nat.Prime p)]

/-- Inversion at `p`: `secpInv (q := p)` is the extended-Euclid inverse, zero at zero. -/
theorem secpInv_zero_p : secpInv (0 : ZMod p) = 0 := secpInv_zero

/-- Fermat at `p`: the Rust `invert` exponent `p - 2` inverts every nonzero element. -/
theorem secpPow_sub_two_inv_p (a : ZMod p) (ha : a ≠ 0) : secpPow a (p - 2) = secpInv a :=
  secpPow_sub_two_inv a ha

/-- Square root at `p`: `mod_four_p` is the only modulus-side input, and the residue hypothesis
`a^(p/2) = 1` is carried, not discharged. `secpPrimeCert` is not used in this theorem beyond the
`Field (ZMod p)` instance it makes available. -/
theorem secpSqrt_sq_p (a : ZMod p) (ha : a ≠ 0) (hres : a ^ (p / 2) = 1) : secpSqrt a ^ 2 = a :=
  secpSqrt_sq a mod_four_p ha hres

/-- The `sqrt(4)` pin at this modulus: `sqrt(4) = 2 ∨ sqrt(4) = p - 2`, with the second root read
from the representative `p - 2`. -/
theorem secpSqrt_four_val_p :
    secpSqrt (4 : ZMod p) = 2 ∨ (secpSqrt (4 : ZMod p)).val = p - 2 :=
  secpSqrt_four_val mod_four_p

/-- The documented conditional content of this file at the concrete modulus: inversion,
Fermat, and the two-root statement for `sqrt(4)`. It is a `Prop`, so an inhabitant of
`secpPrimeCert` is exactly what turns it from a statement into a proved theorem. -/
def secpFieldLawsAtP : Prop :=
  ∀ _cert : secpPrimeCert,
    (∀ a : ZMod p, a ≠ 0 → secpInv (q := p) a * a = 1)
      ∧ (∀ a : ZMod p, secpInv (q := p) a * a = 1 → secpPow a (p - 2) = secpInv (q := p) a)
      ∧ (secpSqrt (q := p) (4 : ZMod p) = 2
        ∨ secpSqrt (q := p) (4 : ZMod p) = ((p - 2 : ℕ) : ZMod p))

/-- `secpFieldLawsAtP` holds given the primality certificate: the proof installs the `Fact`
locally through `secpFact` and instantiates the generic section. It does not register a global
`[Fact (Nat.Prime p)]`, so no theorem in this file is proved by assuming the open obligation. -/
theorem secpFieldLawsAtP_holds : secpFieldLawsAtP := by
  intro h
  letI : Fact (Nat.Prime p) := secpFact h
  refine ⟨fun a ha => ?_, fun a h1 => ?_, ?_⟩
  · exact secpInv_mul_cancel a ha
  · refine secpPow_sub_two_inv a ?_
    intro hz
    have hzero : secpInv a * a = 0 := by rw [hz, secpInv_zero, mul_zero]
    rw [hzero] at h1
    exact absurd h1 (by simp)
  · rcases secpSqrt_four_choice_val mod_four_p with h' | h'
    · exact Or.inl h'
    · exact Or.inr h'

end Concrete

end Apeiron.Crypto.Field.Secp256k1