/-
Canonical port of the two `SecurityCore` keystream theorems
(`Sanguine.Security.xorStream_involutive`, `Sanguine.Security.otp_changes_the_wire`) to the
`BitVec n` / `List (BitVec n)` carrier of this repository.

Face: **all-σ** (generic). Neither statement mentions `σ`, a `Cell`, a norm form or a face
coordinate; nothing σ-specific is assumed, and no face may be read into either conclusion.

Port, not transport. `xorStream` is a new definition on a fixed-width carrier, not the
`List Nat` function of the old corpus; the two theorems are re-proved here from the group laws
and from the core `BitVec` xor lemmas. The two lanes are separate because they are separate
contracts: the `AddCommGroup` lane is the additive (Boolean-ring) reading in which the keystream
XOR *is* the group operation, the `BitVec` lane is the bitwise reading in which the keystream XOR
is `^^^`. This module does **not** prove `(x ^^^ y : BitVec n) = x + y`; that bridge is not
established here and no conclusion below depends on it.

Honest limits carried over from the source's own warrant block, which are not weakened:

* `xorStream_involutive` is **round-trip fidelity over the masked prefix only**. The
  `[] , d :: ds` arm passes the tail through unmasked, so the theorem says the masked positions
  round-trip, *not* that the whole stream was covered. `Crypto.Security.Core.short_key_passthrough`
  is the falsifier for a coverage reading. It does not imply coverage.
* `otp_changes_the_wire` is **one witness of non-identity** of the encoder. It is not secrecy:
  there is no observer, no key law, no observation relation and no advantage statement here.
  "The encoder is not the identity function" is the whole content.
-/

import Mathlib.Data.BitVec

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.Security

/-! ## The `AddCommGroup` lane: the keystream XOR is the group operation

On a Boolean ring (and in particular on a `BitVec n` read through its `CommRing` instance) `+`
*is* exclusive-or, so a keystream is a word added twice. Cost: one cons cell and one `HAdd.hAdd`
per masked position, no allocation; the passthrough tail costs one cons cell.
-/

/--
Keystream masking: XOR `key` into `data` positionwise over the masked prefix, leaving the
remaining tail of `data` untouched when the key runs out.

The third argument is spelled as `data` so the `[] , d :: ds` arm is the passthrough tail, not a
length admission condition. Whether the key covers the data is a separate question, settled by
`short_key_passthrough` and not by `xorStream_involutive`.
-/
def xorStream {α : Type} [AddCommGroup α] : List α → List α → List α
  | _,       []      => []
  | [],      d :: ds => d :: ds
  | k :: ks, d :: ds => (d + k) :: xorStream ks ds

/--
The masked prefix round-trips: masking with `key` twice is the identity on `data`.

`hSelfInv` is the hypothesis that `+` is exclusive-or on `α`, i.e. every element is its own
inverse. It is **not** a consequence of `AddCommGroup`; in an abelian group `x + x = 0` is the
Boolean-ring property, so it is stated rather than assumed. The step is `d + k + k = d` by
`add_assoc`, `hSelfInv` and `add_zero`. This is losslessness of the encoder over the masked
prefix. It is not a statement about secrecy, coverage or key length.
-/
theorem xorStream_involutive {α : Type} [AddCommGroup α] (hSelfInv : ∀ x : α, x + x = 0)
    (key data : List α) :
    xorStream key (xorStream key data) = data := by
  induction key generalizing data with
  | nil => cases data <;> rfl
  | cons k ks ih =>
      cases data with
      | nil => rfl
      | cons d ds =>
          show ((d + k) + k) :: xorStream ks (xorStream ks ds) = d :: ds
          rw [add_assoc, hSelfInv, add_zero, ih ds]

/--
The encoder is not the identity function.

`hne` is the explicit hypothesis that some keystream element differs from the group zero; the
existential conclusion needs it, and it is not a secrecy statement. The witness is one position
masked by a nonzero element against a zero word.
-/
theorem otp_changes_the_wire {α : Type} [AddCommGroup α] (hne : ∃ w : α, w ≠ 0) :
    ∃ key data : List α, xorStream key data ≠ data := by
  obtain ⟨w, hw⟩ := hne
  refine ⟨[w], [0], ?_⟩
  intro h
  have hh : (some w : Option α) = some 0 := by
    simpa [xorStream] using congrArg List.head? h
  exact hw (Option.some.inj hh)

/--
Falsifier for a coverage reading of `xorStream_involutive`: a one-element key leaves the rest of
the data unmasked, so the round trip is vacuous there. This is the limit stated in the file header,
kept next to the theorem it bounds.
-/
theorem short_key_passthrough {α : Type} [AddCommGroup α] (key : α) (datum : α) (tail : List α) :
    xorStream [key] (datum :: tail) = (datum + key) :: tail := by
  simp only [xorStream]
  cases tail <;> rfl

/-! ## The `BitVec n` lane: the keystream XOR is `^^^`

Fixed width `n`; the three laws used are core `BitVec` lemmas, not the `AddCommGroup` ones, so
this lane does not depend on the `+`/`^^^` bridge mentioned in the file header.
-/

/--
Bitwise keystream masking over a fixed width `n`. Same three arms as `xorStream`, with `^^^` in
place of `+`. Cost: one cons cell per masked position, no allocation.
-/
def bitVecXorStream (n : Nat) : List (BitVec n) → List (BitVec n) → List (BitVec n)
  | _,       []      => []
  | [],      d :: ds => d :: ds
  | k :: ks, d :: ds => (d ^^^ k) :: bitVecXorStream n ks ds

/--
Bitwise round-trip fidelity over the masked prefix: `^^^` is an involution, so masking twice with
the same keystream recovers `data`. Uses `BitVec.xor_assoc`, `BitVec.xor_self`, `BitVec.xor_zero`.

Width that matters: exactly the masked prefix, for the same reason as `xorStream_involutive`.
-/
theorem bitVecXorStream_involutive (n : Nat) (key data : List (BitVec n)) :
    bitVecXorStream n key (bitVecXorStream n key data) = data := by
  induction key generalizing data with
  | nil => cases data <;> rfl
  | cons k ks ih =>
      cases data with
      | nil => rfl
      | cons d ds =>
          show ((d ^^^ k) ^^^ k) :: bitVecXorStream n ks (bitVecXorStream n ks ds) = d :: ds
          rw [BitVec.xor_assoc, BitVec.xor_self, BitVec.xor_zero, ih ds]

/--
The bitwise encoder is not the identity function. `hne` is the explicit hypothesis that some
`BitVec n` differs from `0#n`; no secrecy claim is made. One witness, decided.
-/
theorem bitVecOtp_changes_the_wire (n : Nat) (hne : 1#n ≠ 0#n) :
    ∃ key data : List (BitVec n), bitVecXorStream n key data ≠ data := by
  refine ⟨[1#n], [0#n], ?_⟩
  intro h
  have hh : (1#n : BitVec n) ^^^ 0#n = 0#n := by
    simpa [bitVecXorStream] using congrArg List.head? h
  exact absurd hh (by simpa using hne)

/--
Coverage falsifier in the bitwise lane, matching `short_key_passthrough`: a one-element keystream
leaves the tail unmasked, so the round trip carries no information about it.
-/
theorem bitVecShortKeyPassthrough (n : Nat) (key : BitVec n) (datum : BitVec n) (tail : List (BitVec n)) :
    bitVecXorStream n [key] (datum :: tail) = (datum ^^^ key) :: tail := by
  simp only [bitVecXorStream]
  cases tail <;> rfl

end Apeiron.Crypto.Security