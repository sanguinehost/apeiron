import Apeiron.AuditClosed
import Apeiron.Table.Basic
import Apeiron.Table.Home
import Native.HyperbolicBase
import Native.LocatorSpec
import Mathlib.Tactic

/-!
# hashtrinity's resident index, read from the trichotomy basis

`../fold` at `d497161f32e0d366220f2065ec36e680bd9aed86` (fold HEAD) and at `4cc14471ae24f3f204fb0b0e91736ef75c909358`,
the G1 revision `apeiron/Cargo.toml:76-80` pins. `measured` — `git -C ../fold diff --stat
4cc14471 d497161f -- hashtrinity/src`: of the five files this module transcribes, `freq.rs`,
`algebra.rs`, `dense.rs`, `face.rs` and `par.rs` are byte-identical and only `lib.rs` differs (2
lines, `pub mod secret;` moved across a `#[cfg]` gate); the line ranges cited below were read at
**both** revisions and are equal. So every `../fold` citation here resolves at the G1 baseline.

This module states and proves the
**end-to-end basis correspondence**: from `Apeiron.Trichotomy.AnyRing`'s cell and ring laws alone,
which hashtrinity surface realises each face's index, when the resident index operation is correct
for that face, when it is exclusion-only, where hashtrinity is *not* exact, and what the canonical
presence predicate needs that the index does not carry.

## Search first (§4)

`rg` for a hashtrinity↔trichotomy correspondence over `proof/corpus/`,
`../sanguine/proof/`, `../sanguine-ref/proof/`, `../fold/hashtrinity/proof/`:

```
$ rg -l "hashtrinity|Hashtrinity" proof/ ../sanguine/proof/ ../sanguine-ref/proof/
$ rg -l "Trichotomy" ../fold/hashtrinity/proof/                     # no hits
$ rg -l "FaceHome|H3_40" ../fold/hashtrinity/ ../fold/hashtrinity/proof/   # no hits
```

One declaration exists, and it is the port target for the read half:
`Apeiron.Table.one_half_decides` / `Apeiron.Table.opaque_needs_lower`,
`../sanguine/proof/Apeiron/Table/Home.lean` at corpus revision
`880aa57e47e782ed7a9d9ee1974cfc87360b446b`. It reads hashtrinity's `get` at `σ > 0` over `ℝ`, with
`Trichotomy.Cell` and `Trichotomy.upper`/`lower`, and says one coordinate decides when a decryptor
recovers the other. It carries **no `ringMul`, no `ringNormForm`, no `RingCell`, no `RingChannel`**
— the ring-universal statement and the non-injectivity direction occur in none of the four trees.
Those are this module's derivations, over the in-tree target
`Apeiron.NativeAddress.quadratic_split` / `h3_40_quadratic_split`
(`proof/corpus/Native/HyperbolicBase.lean`).

## What is a proof here and what is not

Everything after §0 is `proved`: a Lean kernel check over the pinned canonical corpus
(`880aa57e`) and the pinned Mathlib (`v4.31.0`). The hashtrinity bodies in §0 are `model`: a
transcription of pinned Rust source into `UInt64`, with no Aeneas extraction binding them to
`../fold`'s LLVM. **No theorem here is about hashtrinity's machine code.** The trust boundary is
named in `ht_resident_index_correspondence`'s docstring and in the receipt.

## The three faces and the surface that realises each

| face | has a two-channel index? | hashtrinity surface |
|---|---|---|
| `σ > 0` hyperbolic (split-complex; Cockle 1848, Clifford 1873) | yes, at a unit root `r` | `home`'s **direct-index branch**, `lib.rs:565-572` (`if coord < len { coord }` at `:567-568`) — the address *is* the slot, no fold; and `Dense`, `dense.rs:162` (`add`) / `dense.rs:173` (`get`), the same identity taken to its limit (no coordinate column, no probe) |
| `σ = 0` parabolic (dual numbers, `R[ε]/ε²`) | no: `√0 = 0` is nilpotent, so never a unit | **none** — `ht_face_zero_has_no_index` |
| `σ < 0` elliptic | no: no real root of `x² − σ` | **none** — `ht_face_elliptic_has_no_index` |

The hyperbolic row's realiser is `Dense`, i.e. the coordinate-key (`Regen`) case of `Table`:
`algebra.rs:20` makes an integer its own address, `CoordInv`/`Regen` (`algebra.rs:29-38`) regenerate
the key from the slot. Its in-tree reading as a cell is
`Apeiron.NativeLocator.hyperbolicScalarEmbedding` (`Native/LocatorSpec.lean:525`) — `(a, 0)` as
slots. `ht_face_positive_is_dense` states that reading: the coordinate-key address is the slot
`(a, 0)`, and the slot reading reads `a` on **both** channels, so the direct branch is not the
one-channel read and is exact.

## Honesty

`coord == wanted` is exclusion-only on the hyperbolic face
(`ht_index_read_not_injective`, `ht_h3_40_index_read_collapses_keys`). The canonical presence
predicate needs the shared full key (`ht_canonical_presence_is_full_key`), and `FreqMap` — which
stores no key at all (`freq.rs:710-712`, `_k : PhantomData<K>`, and the module header `freq.rs:3`
"no key column") — merges distinct keys at one coordinate (`ht_freqmap_merges_equal_coords`,
counterexample `coord((-12,38,11)) == coord((12,34,-9)) == 0x134e261b3f5655c1`).
-/

namespace Apeiron
namespace HashtrinityBasis

open Trichotomy.AnyRing
open Apeiron.NativeAddress

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-! ## §0 · The incumbent's bodies, transcribed

Pinned to `../fold` at `d497161f32e0d366220f2065ec36e680bd9aed86`. Grade `model`: no extraction
binds these definitions to the shipped code. -/

/-- `../fold/hashtrinity/src/lib.rs:292`, the golden-ratio 64-bit constant the fold multiplies by. -/
def htPHI : UInt64 := 0x9E37_79B9_7F4A_7C15

/-- `../fold/hashtrinity/src/lib.rs:298`, the reserved "slot holds nothing" coordinate. -/
def htEMPTY : UInt64 := 0xFFFF_FFFF_FFFF_FFFF

/-- `../fold/hashtrinity/src/lib.rs:378`, the reserved tombstone coordinate. -/
def htTOMB : UInt64 := htEMPTY - 1

/-- All 64 bits set, the mask a two's-complement sign extension uses. -/
def htAllOnes : UInt64 := htEMPTY

/-- Rust's `i32 as u64`: the same bits, sign-extended — `algebra.rs:20`'s `*self as u64`. -/
def htI32Coord (i : Int) : UInt64 :=
  if i < 0 then (htAllOnes - i.natAbs.toUInt64) + 1 else i.natAbs.toUInt64

/-- `../fold/hashtrinity/src/lib.rs:553-560`, `coord_of`: nudge the two reserved coordinates aside
so a live coordinate is never a sentinel. The fold sends `EMPTY`, `TOMB` **and** `TOMB − 1` to the
same word (`lib.rs:548-550`). -/
def htCoordOf (c : UInt64) : UInt64 :=
  if c = htEMPTY || c = htTOMB then htTOMB - 1 else c

/-- `../fold/hashtrinity/src/algebra.rs:352-357`, `<(A, B) as Coord>::coord`. -/
def htTupleFold2 (a b : UInt64) : UInt64 := (a * htPHI) ^^^ b

/-- `../fold/hashtrinity/src/algebra.rs:358-362`, `<(A, B, C) as Coord>::coord` — the **lossy**
tuple-cell fold. -/
def htTupleFold3 (a b c : UInt64) : UInt64 := (htTupleFold2 a b * htPHI) ^^^ c

/-- The `coord_of` of an `i32` triple: the hashtrinity address of a tuple key. -/
def htTupleCoord (k : Int × Int × Int) : UInt64 :=
  htCoordOf (htTupleFold3 (htI32Coord k.1) (htI32Coord k.2.1) (htI32Coord k.2.2))

/-- `../fold/hashtrinity/src/lib.rs:565-572`, `home`: a coordinate below the slot count **is** its
own slot; anything larger takes the top `bits` of `coord·PHI`. -/
def htHomeDirect (len bits coord : UInt64) : UInt64 :=
  if coord < len then coord else (coord * htPHI) >>> (64 - bits)

/-- `../fold/hashtrinity/src/dense.rs:162` / `dense.rs:173`: `Dense`'s address function is the
identity, so `add i` writes slot `i` and `get i` reads slot `i`. -/
def htDenseIndex (i : UInt64) : UInt64 := i

/-- `../fold/hashtrinity/src/freq.rs:737-739`, `FreqMap::freq`: branchless clamp keeping the fold off
the two sentinels; its whole distortion is one three-point fibre. -/
def htFreqClamp (c : UInt64) : UInt64 := min c (htEMPTY - 2)

/-- `coord_of` is the identity off the reserved range. -/
theorem htCoordOf_of_ne (c : UInt64) (h1 : c ≠ htEMPTY) (h2 : c ≠ htTOMB) : htCoordOf c = c := by
  simp [htCoordOf, h1, h2]

/-- **`coord_of` collapses a three-point fibre.** `EMPTY`, `TOMB` and `TOMB − 1` are three
distinct reserved-side coordinates and the fold sends all of them to one word, so `EMPTY` and
`TOMB` are indistinguishable after the fold. -/
theorem htCoordOf_three_way_fiber :
    htCoordOf htEMPTY = htTOMB - 1 ∧ htCoordOf htTOMB = htTOMB - 1 ∧
      htCoordOf (htTOMB - 1) = htTOMB - 1 := by
  refine ⟨?_, ?_, ?_⟩ <;> decide

/-- `home`'s direct-index branch: a coordinate below the slot count is its own slot. No fold, no
hash, no collision — this is the branch that realises the hyperbolic face's index. -/
theorem ht_home_direct_is_identity {len bits coord : UInt64} (h : coord < len) :
    htHomeDirect len bits coord = coord := by
  simp [htHomeDirect, h]

/-- `home`'s fold branch: a coordinate at or above the slot count is truncated through `coord·PHI`. -/
theorem ht_home_fold_of_large {len bits coord : UInt64} (h : len ≤ coord) :
    htHomeDirect len bits coord = (coord * htPHI) >>> (64 - bits) := by
  have h' : ¬ coord < len := Nat.not_lt.mpr h
  simp [htHomeDirect, h']

/-- `Dense`'s address function is the identity. -/
theorem ht_dense_index_is_identity (i : UInt64) : htDenseIndex i = i := rfl

/-- The hyperbolic face's coordinate-key index never collides: it is the identity, and the in-tree
declaration `Apeiron.Table.coord_home_collisionFree` is that fact for `home = id`. -/
theorem ht_coordinate_key_index_never_collides (a b : ℕ) :
    ¬ Apeiron.Table.collides id a b :=
  Apeiron.Table.coord_home_collisionFree a b

/-! ## §1 · The face trichotomy as the existence of an index -/

/-- A face of `R[x]/(x² − σ)` carries a **two-channel index** exactly when the quadratic has a
unit root and a unit half: then `RingCell R ≃ R × R` is the canonical `ringOfHalves` change of
coordinates, and the two channels read every cell. -/
def HtIndexWitness {R : Type*} [CommRing R] (σ : R) : Prop :=
  ∃ r s h : R, r ^ 2 = σ ∧ r * s = 1 ∧ 2 * h = 1

theorem htIndexWitness_iff {R : Type*} [CommRing R] (σ : R) :
    HtIndexWitness σ ↔ ∃ r s h : R, r ^ 2 = σ ∧ r * s = 1 ∧ 2 * h = 1 :=
  Iff.rfl

/-- **FACE `σ = 0`, parabolic: no index.** A root of `x² = 0` is nilpotent, hence not a unit, so
there is no `s`, no `h`, no channel split and no index map. Canonical contrast over `ℝ`:
`Trichotomy.upper_eq_lower_of_disc_zero`, where `√0 = 0` collapses both channels onto the first
slot — `R[ε]/ε²` is a filtration, not a product.

`[Nontrivial R]` is required and is not redundant: in the trivial ring every element is `0 = 1`,
every `r` is a unit, and `HtIndexWitness 0` holds. `NoZeroDivisors` alone does not exclude it. -/
theorem ht_face_zero_has_no_index {R : Type*} [CommRing R] [NoZeroDivisors R] [Nontrivial R] :
    ¬ HtIndexWitness (0 : R) := by
  rintro ⟨r, s, h, hr, hrs, hh⟩
  rw [sq_eq_zero_iff.mp hr, zero_mul] at hrs
  exact absurd hrs one_ne_zero.symm

/-- **FACE `σ < 0`, elliptic: no index.** Over `ℝ` the quadratic `x² − σ` has no root, so the
algebra is a field extension with no channel split and no zero divisors. Canonical reading:
`Trichotomy.Roots.exists_extra_sqrt_one_iff_disc_pos`, the extra square root of one existing
exactly when `σ > 0`. -/
theorem ht_face_elliptic_has_no_index (σ : ℝ) (hσ : σ < 0) : ¬ HtIndexWitness σ := by
  rintro ⟨r, s, h, hr, hrs, hh⟩
  have hnonneg : (0 : ℝ) ≤ σ := by rw [← hr]; exact sq_nonneg r
  exact absurd hnonneg (not_le.mpr hσ)

/-- **FACE `σ > 0`, hyperbolic: the index exists.** The sealed profile's `H3_40 = ZMod (3^40)` at
`σ = 1` has root `r = 1`, its own inverse, and `2⁻¹` — the joint witness. -/
theorem ht_face_positive_has_index : HtIndexWitness (1 : H3_40) :=
  ⟨1, 1, h3_40Half, by norm_num, by norm_num, h3_40_two_mul_half⟩

/-- **The hyperbolic realiser is `Dense` at the slot reading.** `Dense`'s identity address
function is `htDenseIndex`, and an `i32` key is its own coordinate (`algebra.rs:20`), so the
resident index of a coordinate key is the coordinate. The slot reading of that coordinate is
`(a, 0)`, which reads `a` on **both** channels — so this realiser is exact, not one-channel. -/
theorem ht_face_positive_is_dense (a : H3_40) :
    Apeiron.NativeLocator.hyperbolicScalarEmbedding a = (a, 0) ∧
      ringUpper 1 (Apeiron.NativeLocator.hyperbolicScalarEmbedding a) = a ∧
        ringLower 1 (Apeiron.NativeLocator.hyperbolicScalarEmbedding a) = a := by
  have hEq := Apeiron.NativeLocator.hyperbolic_scalar_channels_equal a
  refine ⟨rfl, congrArg Prod.fst hEq, congrArg Prod.snd hEq⟩

/-! ## §2 · The end-to-end correspondence

`Apeiron.NativeAddress.quadraticChannelEquiv` is the in-tree port target
(`proof/corpus/Native/HyperbolicBase.lean`), itself `ringUpper`/`ringLower` plus the round trip. -/

/-- The canonical two-channel change of coordinates, re-exported so the headline reads on one
line. Same definition as `Apeiron.NativeAddress.quadraticChannelEquiv`. -/
def htChannelEquiv {R : Type*} [CommRing R] (r s h : R) (hrs : r * s = 1) (hh : 2 * h = 1) :
    RingCell R ≃ R × R :=
  Apeiron.NativeAddress.quadraticChannelEquiv r s h hrs hh

/-- The two readouts together decide the cell. Derived from the canonical round trip
`AnyRing.ringOfHalves_upper_lower`. -/
theorem ht_both_readouts_decide {R : Type*} [CommRing R] {r s h : R} (hrs : r * s = 1)
    (hh : 2 * h = 1) (z : RingCell R) (w : RingCell R) (h1 : ringUpper r z = ringUpper r w)
    (h2 : ringLower r z = ringLower r w) : z = w :=
  calc z = ringOfHalves (ringUpper r z) (ringLower r z) s h :=
        ringOfHalves_upper_lower hrs hh z |>.symm
    _ = ringOfHalves (ringUpper r w) (ringLower r w) s h := by rw [h1, h2]
    _ = w := ringOfHalves_upper_lower hrs hh w

/-- **Without a second readout the cell is not decided.** `opaque_needs_lower`'s ring-universal
form; the ℝ/σ>0 port target is `Apeiron.Table.opaque_needs_lower`. -/
theorem ht_one_readout_leaves_the_second {R : Type*} [CommRing R] {r s h : R} (hrs : r * s = 1)
    (hh : 2 * h = 1) {z w : RingCell R} (h1 : ringUpper r z = ringUpper r w) (hne : z ≠ w) :
    ringLower r z ≠ ringLower r w :=
  fun h2 => hne (ht_both_readouts_decide hrs hh z w h1 h2)

/-- **HEADLINE. The end-to-end basis correspondence for hashtrinity's resident index operation.**

From the trichotomy cell and ring laws alone, at the joint witness `r² = σ`, `r·s = 1`, `2h = 1`:

* the two channel reads are ring homomorphisms and the two idempotents are complementary, and
  the change of coordinates is an equivalence — the hyperbolic face's index, canonical;
* the norm form is the product of the two channel reads, so the collapsed direction is null;
* **the resident index operation is correct**: one index readout *together with the key's own
  second readout* decides the cell, which is the soundness of hashtrinity's hit branch at
  `../fold/hashtrinity/src/lib.rs:676-678` (`get_coord`: `if sc == coord { return … }`, `get_coord`
  itself at `:671-683`, reached from `get` at `:662`) and of `get_exact` at
  `../fold/hashtrinity/src/lib.rs:1124-1131` (`found && ((KS::REGEN && coord != TOMB - 1) ||
  self.keys.matches(i, coord, k))`);
* **and it is not correct on the index alone**: `(1, 0)` and `(0, s)` are distinct cells with one
  index readout whose difference is a **null cell**, so the one-line rule `coord == wanted ⇒ hit`
  is exclusion-only.

Refines, body by body: `lib.rs:565-572` (`home`), `lib.rs:552-560` (`coord_of`),
`algebra.rs:352-362` (the tuple `Coord` fold), `dense.rs:162`/`dense.rs:173` (`Dense`),
`freq.rs:737-739` (`FreqMap::freq`), `lib.rs:662` (`get`), `lib.rs:1124` (`get_exact`). Proven from
`AnyRing.ringUpper_channel`, `ringLower_channel`, `ringComplementary_idem`, `ringMul`,
`ringNormForm`, `ringNormForm_eq_upper_mul_lower`, `ringOfHalves_upper_lower` and the in-tree
`Apeiron.NativeAddress.quadratic_split`.

**Machine-code leg, `measured`.** `disasm.txt`
sha256 `61bc3365ac8c71d55b4935d215cc57b3fbd02b59427e1b1b1fc1e25c2352340c`, from the `4cc14471`
build, `probe_get` at `18210`:

| address | instruction | source |
|---|---|---|
| `18213`,`18217`,`1821e` | `cmp $0xfffffffffffffffd` / `mov $0xfffffffffffffffd` / `cmovb` | `coord_of`'s nudge, `TOMB − 1 = 0xFFFF_FFFF_FFFF_FFFD`; **branchless**, the three-point fibre of `htCoordOf_three_way_fiber` |
| `18225`,`18229` | `cmp 0x8(%rdi),%rsi` / `jb` | `home`'s direct-index branch: `coord < len` ⇒ the address *is* the slot, the first arm of `htHomeDirect` |
| `1822b`,`18235`,`1823b`,`1823e` | `movabs $0x9e3779b97f4a7c15` / `imul` / `sub` / `shr %cl` | `home`'s fold arm, `coord·PHI >>> (64 − bits)`, `htPHI` |
| `18249`,`1824c`,`18252`,`18256` | `cmp %rsi,%rcx` / `jne` / `shl $0x3` / `add` | the hit branch `sc == coord ⇒ return`: one equality **on the index readout alone** |
| `1825b` | `cmp $0xffffffffffffffff` | the `sc == EMPTY ⇒ None` miss |
| `1878b`,`1878f` (`probe_get_exact`) | `cmp $0xfffffffffffffffe` / `jb` | `get_exact`'s `coord != TOMB − 1` guard — the one coordinate where `Regen` skips the key compare |

So the shipped read really is `coord == wanted ⇒ hit` as a **necessary** condition, and the
instruction-level witness of the exclusion-only clause is the bare `cmp/jne` at `18249` with no key
compare on the hot path.

**Trust boundary:** the last two conjuncts are statements about the canonical algebra under the
face's joint witness. They are statements *about* hashtrinity's read discipline through the
transcribed bodies in §0; no extraction binds §0 to `../fold`'s Rust, and nothing here is about
`../fold`'s LLVM, `rustc` codegen, the SIMD prefix scan, or the probe walk. -/
theorem ht_resident_index_correspondence {R : Type*} [CommRing R] [Nontrivial R] {σ r s h : R}
    (hrr : r ^ 2 = σ) (hrs : r * s = 1) (hh : 2 * h = 1) :
    RingChannel σ (ringUpper r) ∧
      RingChannel σ (ringLower r) ∧
      RingComplementary σ (ringIdemPlus σ r s h) (ringIdemMinus σ r s h) ∧
      Function.Injective (htChannelEquiv r s h hrs hh) ∧
      (∀ z : RingCell R, ringNormForm σ z = ringUpper r z * ringLower r z) ∧
      (∀ z w : RingCell R, ringUpper r z = ringUpper r w → ringLower r z = ringLower r w →
        z = w) ∧
      (∃ z w : RingCell R,
        z ≠ w ∧ ringUpper r z = ringUpper r w ∧ ringNormForm σ (z - w) = 0) ∧
      ¬ (∀ z w : RingCell R, ringUpper r z = ringUpper r w → z = w) := by
  have hrr' : r * r = σ := by simpa [pow_two] using hrr
  have hinj : Function.Injective (htChannelEquiv r s h hrs hh) := by
    intro z w hzw
    exact ht_both_readouts_decide hrs hh z w (congrArg Prod.fst hzw) (congrArg Prod.snd hzw)
  -- The witness `(1, 0)` and `(0, s)`: distinct cells, one index readout, a null difference.
  have hne1 : ((1, 0) : RingCell R) ≠ (0, s) := by
    intro hEq
    have hs0 : s = (0 : R) := by simpa using (congrArg Prod.snd hEq).symm
    have hzero : (0 : R) = 1 := by
      calc (0 : R) = r * 0 := (mul_zero r).symm
        _ = r * s := by rw [hs0]
        _ = 1 := hrs
    exact one_ne_zero hzero.symm
  have hup : ringUpper r (1, 0) = ringUpper r (0, s) := by simp [ringUpper, hrs]
  refine ⟨ringUpper_channel hrr, ringLower_channel hrr, ringComplementary_idem hrr' hrs hh,
    hinj, fun z => ringNormForm_eq_upper_mul_lower hrr z,
    fun _ _ h1 h2 => ht_both_readouts_decide hrs hh _ _ h1 h2, ?_, ?_⟩
  · refine ⟨(1, 0), (0, s), hne1, hup, ?_⟩
    rw [ringNormForm_eq_upper_mul_lower hrr]
    have hu : ringUpper r ((1, 0) - (0, s)) = 0 := by simp [ringUpper, hrs]
    rw [hu, zero_mul]
  · intro hall
    exact hne1 (hall (1, 0) (0, s) hup)

/-! ## §3 · Honesty (a): `coord == wanted` is exclusion-only -/

/-- **One index readout does not decide the cell, and the lost direction is a null cell.**

The witness is `(1, 0)` and `(0, s)`: `ringUpper r (0, s) = 0 + r·s = 1 = ringUpper r (1, 0)`, the
cells differ because `s` is a unit, and — under the joint witness — the difference `(1, −s)` has
`ringNormForm σ (1, −s) = 1 − σs² = 1 − (rs)² = 0`. So the index collapses a zero divisor, which
is why `coord == wanted ⇒ hit` is a necessary condition and not a presence criterion. The
hypothesis `2 ≠ 0` is needed only for the second-readout clause: at `2 = 0` the two readouts agree
on this pair as well, and the collapse is total rather than one-dimensional.

The H3_40 instance of this witness is the pair `(1, 0)`, `(0, 1)` — the same pair as
`Apeiron.NativeAddress.h3_40_index_read_not_injective` in
`proof/corpus/Native/HyperbolicFaceNullCell.lean`. That module is **not in the live tree** at this
revision (`git ls-files proof/corpus/Native/` returns only `HyperbolicBase.lean` and
`LocatorSpec.lean`; the module exists only in the scratch `tree`,
from ), so it could not be imported and this general-`σ` form was proved
instead. Residual: when that module lands, `ht_h3_40_index_read_collapses_keys` is its
specialisation and one of the two should be re-expressed in terms of the other. -/
theorem ht_index_read_not_injective {R : Type*} [CommRing R] [Nontrivial R] {r s : R}
    (hrs : r * s = 1) (h2 : (2 : R) ≠ 0) :
    ∃ z w : RingCell R,
      z ≠ w ∧ ringUpper r z = ringUpper r w ∧ ringLower r z ≠ ringLower r w := by
  refine ⟨(1, 0), (0, s), ?_, ?_, fun hl => ?_⟩
  · intro hEq
    have hs0 : s = (0 : R) := by
      simpa using (congrArg Prod.snd hEq).symm
    rw [hs0, mul_zero] at hrs
    exact absurd hrs one_ne_zero.symm
  · simp [ringUpper, hrs]
  · have h1 : ringLower r (0, s) = -(1 : R) := by
      simp [ringLower, hrs]
    have hone : ringLower r (1, 0) = (1 : R) := by simp [ringLower]
    have hne : ¬((1 : R) = -(1 : R)) := by
      intro hEq
      exact h2 (by
        calc (2 : R) = (1 : R) - (-(1 : R)) := by ring
          _ = 1 - (-1) := rfl
          _ = 0 := sub_eq_zero.mpr hEq)
    rw [h1, hone] at hl
    exact hne hl

/-! ## §4 · Honesty (b): the canonical presence predicate needs the full key -/

/-- **`Apeiron.Table.get` / `insertWith` decide presence by the shared full key.**
`../sanguine/proof/Apeiron/Table/Basic.lean:71` and `:80` at `880aa57e`:
`insertWith f k v m = fun x => if x = k then f v (m x) else m x`, with `[DecidableEq K]`. So a
write is witnessed by comparing the *whole* key, and a table is unchanged everywhere else
**exactly** when the merge at the written key is the identity — the canonical
`get_insertWith_ne` (`Basic.lean:97`).

This is the step hashtrinity's index-only read cannot perform: it has the index readout, not the
key, and `ht_index_read_not_injective` says the index does not carry the key. -/
theorem ht_canonical_presence_is_full_key {K V : Type} [DecidableEq K]
    (f : V → V → V) (k : K) (v : V) (m : Apeiron.Table.HMap K V) :
    (∀ x : K, Apeiron.Table.get (Apeiron.Table.insertWith f k v m) x = Apeiron.Table.get m x) ↔
      f v (Apeiron.Table.get m k) = Apeiron.Table.get m k := by
  constructor
  · intro h
    have hk := h k
    rw [Apeiron.Table.get_insertWith_self] at hk
    exact hk
  · intro h x
    by_cases hx : x = k
    · subst hx
      rw [Apeiron.Table.get_insertWith_self]
      exact h
    · exact Apeiron.Table.get_insertWith_ne f k v m hx

/-! ## §5 · The incumbent is not exact: `FreqMap` merges equal coordinates -/

/-- **THE COUNTEREXAMPLE.** Two distinct `i32` triple keys, one resident coordinate.
`../fold/hashtrinity/src/algebra.rs:358-362` folds a tuple by `c ← (c·PHI) ⊕ xᵢ`; with `i32` keys
sign-extended (`algebra.rs:20`), `coord((-12,38,11))` and `coord((12,34,-9))` are both
`0x134e261b3f5655c1`. `coord_of`'s nudge (`lib.rs:553`) is the identity on this word, so the
resident coordinate is genuinely one.

The value `0x134e261b3f5655c1` is `measured`, recomputed by a command on the pinned tree; the
equality below is `proved` about the transcribed fold. -/
theorem ht_freqmap_merges_equal_coords :
    htTupleCoord (-12, 38, 11) = htTupleCoord (12, 34, -9) ∧
      htTupleCoord (-12, 38, 11) = 0x134E_261B_3F56_55C1 := by
  decide

/-- `FreqMap`'s clamp is the identity off the two reserved coordinates and one below them.
`UInt64`'s `min` is `minOfLe`, i.e. an `if`, not the `LinearOrder` `min`. -/
theorem ht_freq_clamp_untouched (c : UInt64) (h : c ≤ htEMPTY - 2) : htFreqClamp c = c := by
  show (if c ≤ htEMPTY - 2 then c else htEMPTY - 2) = c
  exact if_pos h

/-- **THEOREM ABOUT THE INCUMBENT.** `FreqMap` stores frequencies, not keys
(`../fold/hashtrinity/src/freq.rs:710-712`: `_k : PhantomData<K>`, and the module header
`freq.rs:3` "no key column"), so its resident predicate is the folded coordinate alone and the
merge `f` of `FreqMap::insert_with` (`freq.rs:773-776`) is applied **across distinct keys**.

The index is not a decision procedure: `(1, 0)` and `(0, 1)` are distinct cells at `H3_40` with
one index readout and a null difference. Instantiated at `σ = 1` over the sealed profile's ring. -/
theorem ht_freqmap_index_is_not_a_decision_procedure :
    ∃ z w : RingCell H3_40,
      z ≠ w ∧ ringUpper (1 : H3_40) z = ringUpper (1 : H3_40) w ∧
        ringNormForm (1 : H3_40) (z - w) = 0 := by
  have hone : (1 : H3_40) ≠ (0 : H3_40) := by decide
  refine ⟨(1, 0), (0, 1), ?_, ?_, ?_⟩
  · intro hEq
    exact hone (congrArg Prod.fst hEq)
  · rfl
  · rw [ringNormForm_eq_upper_mul_lower (σ := (1 : H3_40)) (r := (1 : H3_40)) (by norm_num)]
    have hu : ringUpper (1 : H3_40) ((1, 0) - (0, 1)) = 0 := by
      simp [ringUpper]
    rw [hu, zero_mul]

/-- The H3_40 specialisation of the index collapse, with the second readout named: the pair
`(1, 0)`, `(0, 1)` reads `1` on the index channel and `1` against `-1` on the second, and the
difference is a null cell. `ht_index_read_not_injective` at `r = s = 1`, `2 ≠ 0` in `H3_40`. -/
theorem ht_h3_40_index_read_collapses_keys :
    ∃ z w : RingCell H3_40,
      z ≠ w ∧ ringUpper (1 : H3_40) z = ringUpper (1 : H3_40) w ∧
        ringLower (1 : H3_40) z ≠ ringLower (1 : H3_40) w := by
  have h2 : (2 : H3_40) ≠ (0 : H3_40) := by decide
  have hone : (1 : H3_40) ≠ (0 : H3_40) := by
    intro h1
    exact h2 (by
      calc (2 : H3_40) = (1 : H3_40) + (1 : H3_40) := by norm_num
        _ = (0 : H3_40) + (0 : H3_40) := by rw [h1]
        _ = 0 := by norm_num)
  have hnt : Nontrivial H3_40 := ⟨(1 : H3_40), (0 : H3_40), hone⟩
  obtain ⟨z, w, hne, hu, hl⟩ :=
    ht_index_read_not_injective (R := H3_40) (r := (1 : H3_40)) (s := (1 : H3_40))
      (by norm_num) h2
  exact ⟨z, w, hne, hu, hl⟩

/-! ## Headline audit rows

Each row is the closed-cone gate of `Apeiron.closedAxioms`: `#audit` collects the declaration's
transitive axiom dependencies and fails the build on anything outside
`{propext, Classical.choice, Quot.sound}`, so a `sorryAx` anywhere in the cone is a build failure.
They sit in this module rather than in `AxiomAudit/Core.lean` so they run in the scoped build of
`Native.HashtrinityBasis` itself. -/

#audit Apeiron.HashtrinityBasis.htCoordOf_three_way_fiber
#audit Apeiron.HashtrinityBasis.ht_home_direct_is_identity
#audit Apeiron.HashtrinityBasis.ht_coordinate_key_index_never_collides
#audit Apeiron.HashtrinityBasis.ht_face_zero_has_no_index
#audit Apeiron.HashtrinityBasis.ht_face_elliptic_has_no_index
#audit Apeiron.HashtrinityBasis.ht_face_positive_has_index
#audit Apeiron.HashtrinityBasis.ht_face_positive_is_dense
#audit Apeiron.HashtrinityBasis.ht_both_readouts_decide
#audit Apeiron.HashtrinityBasis.ht_one_readout_leaves_the_second
#audit Apeiron.HashtrinityBasis.ht_resident_index_correspondence
#audit Apeiron.HashtrinityBasis.ht_index_read_not_injective
#audit Apeiron.HashtrinityBasis.ht_canonical_presence_is_full_key
#audit Apeiron.HashtrinityBasis.ht_freqmap_merges_equal_coords
#audit Apeiron.HashtrinityBasis.ht_freqmap_index_is_not_a_decision_procedure
#audit Apeiron.HashtrinityBasis.ht_h3_40_index_read_collapses_keys

end HashtrinityBasis
end Apeiron
