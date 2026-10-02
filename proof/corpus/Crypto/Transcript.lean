/-
A Fiat–Shamir transcript over the FIPS-202 sponge of `Crypto.Sponge`.

The transcript carries the absorbed byte list and a challenge counter. `squeeze` is a function of the
absorbed bytes alone, so the transcript is a *function* of its own history: two transcripts with the
same absorbed list squeeze to the same bytes at every explicit output length. The theorems below are
that determinism, the append behaviour of `absorbBytes`, and a non-identity witness that a nonempty
absorb changes the absorbed list and the squeezed bytes.

Not claimed here: collision resistance, second-preimage resistance, unforgeability, domain separation
between transcripts with different counters, or that the challenge binds to a verifier statement.
The sponge itself is green in `Crypto.Sponge`; nothing about its cryptographic strength is used by the
theorems below.
-/

import Crypto.Sponge

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.Transcript

/-! ## The transcript -/

/-- A Fiat–Shamir transcript: the bytes absorbed so far, in order, and the number of challenges
already squeezed. The absorbed list is the only data a challenge depends on; the counter is
transcript bookkeeping. -/
structure Transcript where
  /-- The absorbed message bytes, in absorption order. -/
  absorbed : List (BitVec 8)
  /-- How many challenges have been squeezed from this transcript. -/
  counter : Nat
  deriving DecidableEq, Repr

/-- The empty transcript. O(1). -/
def init : Transcript := ⟨[], 0⟩

/-- Append bytes to the absorbed list. O(|bs|) time, O(|bs|) space for the appended list. -/
def absorbBytes (t : Transcript) (bs : List (BitVec 8)) : Transcript :=
  ⟨t.absorbed ++ bs, t.counter⟩

/-- The `n` squeezed challenge bytes of `t` from `shake128` (FIPS-202 §6.2), an explicit output
length. A function of `t.absorbed` alone: O(n + ⌈n / 168⌉ · permute) time, O(n) space. -/
def squeeze (t : Transcript) (n : Nat) : Vector (BitVec 8) n :=
  Keccak.shake128 t.absorbed n

/-- Absorb the `n` bytes of a previously squeezed challenge `c` and increment the counter. The
counter byte in front of `c` keeps two absorptions of equal `c` from producing equal transcripts.
O(n) time, O(n) space. -/
def absorbChallenge (t : Transcript) {n : Nat} (c : Vector (BitVec 8) n) : Transcript where
  absorbed := t.absorbed ++ BitVec.ofNat 8 t.counter :: c.toList
  counter := t.counter + 1

/-! ## Structure: the absorbed list is the transcript -/

/-- `squeeze` is a function of the absorbed list. `rfl`; stated so the dependency is checkable. -/
theorem squeeze_eq_shake128 (t : Transcript) (n : Nat) :
    squeeze t n = Keccak.shake128 t.absorbed n := rfl

/-- `absorbBytes` appends, and leaves the counter alone. `rfl`. -/
theorem absorbBytes_absorbed (t : Transcript) (bs : List (BitVec 8)) :
    (absorbBytes t bs).absorbed = t.absorbed ++ bs := rfl

/-- Absorption composes: absorbing `bs₁` then `bs₂` leaves the same wire as absorbing `bs₁ ++ bs₂` in
one call, so the challenge of a transcript does not depend on how the bytes were chunked. -/
theorem absorbBytes_assoc (t : Transcript) (bs₁ bs₂ : List (BitVec 8)) :
    (absorbBytes (absorbBytes t bs₁) bs₂).absorbed
      = (absorbBytes t (bs₁ ++ bs₂)).absorbed := by
  simp only [absorbBytes, List.append_assoc]

/-- `absorbBytes` does not move the counter. `rfl`. -/
theorem absorbBytes_counter (t : Transcript) (bs : List (BitVec 8)) :
    (absorbBytes t bs).counter = t.counter := rfl

/-- Determinism of `squeeze`: two transcripts with the same absorbed bytes squeeze the same `n`
challenge bytes, for every explicit output length `n`. -/
theorem squeeze_congr_absorbed {t₁ t₂ : Transcript} (h : t₁.absorbed = t₂.absorbed) (n : Nat) :
    squeeze t₁ n = squeeze t₂ n := by
  rw [squeeze, squeeze, Keccak.shake128, Keccak.shake128, h]

/-- `absorbChallenge` appends the counter byte and the challenge, and increments the counter. -/
theorem absorbChallenge_absorbed (t : Transcript) {n : Nat} (c : Vector (BitVec 8) n) :
    (absorbChallenge t c).absorbed = t.absorbed ++ BitVec.ofNat 8 t.counter :: c.toList :=
  rfl

theorem absorbChallenge_counter (t : Transcript) {n : Nat} (c : Vector (BitVec 8) n) :
    (absorbChallenge t c).counter = t.counter + 1 := rfl

/-- Absorption is observable in the transcript: a nonempty absorb changes the absorbed list. -/
theorem changes_the_wire_absorbed (b : BitVec 8) :
    (absorbBytes init [b]).absorbed ≠ init.absorbed := by
  simp [absorbBytes, init]

/-! ## The evaluator pins

`#guard` runs the untrusted evaluator, so these are regression pins on the concrete challenge bytes,
not kernel proofs of any separation property. -/

/- `shake128 ""` at 32 bytes, via `hashlib.shake_128(b"").digest(32)`. -/
#guard (squeeze init 32).toList ==
  [0x7f, 0x9c, 0x2b, 0xa4, 0xe8, 0x8f, 0x82, 0x7d, 0x61, 0x60, 0x45, 0x50, 0x76, 0x05, 0x85, 0x3e,
   0xd7, 0x3b, 0x80, 0x93, 0xf6, 0xef, 0xbc, 0x88, 0xeb, 0x1a, 0x6e, 0xac, 0xfa, 0x66, 0xef, 0x26]

/- `shake128 "a"` at 32 bytes. Differs from the empty transcript above: the absorb reaches the
squeezed output, not only the absorbed list. -/
#guard (squeeze (absorbBytes init [0x61]) 32).toList ==
  [0x85, 0xc8, 0xde, 0x88, 0xd2, 0x88, 0x66, 0xbf, 0x08, 0x68, 0x09, 0x0b, 0x39, 0x61, 0x16, 0x2b,
   0xf8, 0x23, 0x92, 0xf6, 0x90, 0xd9, 0xe4, 0x73, 0x09, 0x10, 0xf4, 0xaf, 0x7c, 0x6a, 0xb3, 0xee]

end Apeiron.Crypto.Transcript