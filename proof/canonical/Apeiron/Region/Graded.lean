import Apeiron.Linter.Notation

/-!
# The graded region -- the strided chart, proved once

@name Graded
@grade proved

A flat region is a base and a run of cells. A graded region is the strided one: `count` records
laid end to end, each `stride` cells wide, addressed `base + i*stride + k`. That address map is a
mixed-radix chart -- record `i` in the high channel, offset `k` in the low one -- and every fact a
strided access needs is a fact about the chart. This file proves those facts once, so no strided
module re-derives its own bound against a global invariant.

Two facts carry the rest. Constant-shift linearity (Rule D): stepping the record index by a
constant translates the address by exactly that many strides -- the chart is a homomorphism in the
index. Index monotonicity (Rule M): a larger record index gives a larger address -- the chart
embeds the index order. `chart_shift_up`, `chart_shift_down` and `chart_mono` are these; from them
`graded_access_in_region` (any in-range `(i,k)` lands inside `[base, base+extent)`) and
`graded_no_overflow_into_neighbor` (two graded regions laid end to end never overlap) follow.

The chart is also a bijection onto its extent. `decode` splits an address back into its two
channels by the division algorithm; `chart_decode` is that the split reconstructs the address
exactly (every address in the region is some channel pair), and `decode_chart` is that decoding an
in-range `(i,k)` recovers it (the channels are independent). Together the two channels are a clean
record/offset coordinate for the region.

Every obligation reduces to linear facts over `Nat`: the shift is by a constant, the bound is
`i*stride + k < count*stride`, and `i*stride` stays a single opaque channel with linear arithmetic
around it. It is `Nat`-only (core `Nat` plus `omega`), Mathlib-free.

## What this does not settle

`decode_chart` needs a positive stride (`0 < stride`); at `stride = 0` the region has no cells and
decoding is degenerate -- the theorem is stated for the non-degenerate case and does not cover
`stride = 0`. `chart_decode` reconstructs any address at or above `base`, including addresses past
the extent -- it is the arithmetic identity of the division algorithm, not a claim that the address
was in range; in-range membership is `graded_access_in_region`, proved separately. Nothing here
says the stride is the record's true size, that records do not overlap in some external layout, or
that `count` and `stride` are the ones a given allocation was granted -- those are facts about a
program, cited against this chart, not established by it. The region is a coordinate system with
checked laws, not a proof that a particular program uses it honestly.
-/

namespace Apeiron
namespace Region

/-- A strided region: a base address, a count of records, and the stride between records.

    Graded                   [a strided region, addressed base + i*stride + k]
    base   : Nat             [the address the region starts at]
    count  : Nat             [how many records the region holds]
    stride : Nat             [how many cells wide each record is] -/
structure Graded where
  base   : Nat
  count  : Nat
  stride : Nat

/-- The region's total extent: `count` records of `stride` cells.

    (g : Graded)             [a strided region: base, count, stride]
    extent g                 [its total width, count records of stride cells each] -/
def extent (g : Graded) : Nat := g.count * g.stride

/-- The mixed-radix chart: record `i`, offset `k` within it.

    (g : Graded) (i k : Nat)     [a region, a record index, an offset within the record]
    chart g i k                  [the address of that cell, base + i*stride + k] -/
def chart (g : Graded) (i k : Nat) : Nat := g.base + i * g.stride + k

/-- **Rule D (constant-shift linearity).** Stepping the record index by a constant translates the
    address by exactly that many strides -- the chart is a `Nat`-module homomorphism in the index.

    (g : Graded) (i k c : Nat)                [a region, a record index, an offset, a constant step]
    chart g (i + c) k = chart g i k + c * g.stride   [shifts the address by c strides] -/
theorem chart_shift_up (g : Graded) (i k c : Nat) :
    chart g (i + c) k = chart g i k + c * g.stride := by
  unfold chart
  have h : (i + c) * g.stride = i * g.stride + c * g.stride := by rw [Nat.add_mul]
  omega

/-- The downward form: `(i-c)*stride = i*stride - c*stride`, the shape a recursion stepping the
    index down by `c` needs. Stated additively (Nat subtraction is total) via `chart_shift_up`.

    (g : Graded) (i k c : Nat)                [a region, a record index, an offset, a constant step]
    (h : c ≤ i)                                      [the step does not run the index below zero]
    ≤                                                [is smaller than or equal to]
    chart g i k = chart g (i - c) k + c * g.stride   [c strides above the stepped-down one] -/
theorem chart_shift_down (g : Graded) (i k c : Nat) (h : c ≤ i) :
    chart g i k = chart g (i - c) k + c * g.stride := by
  have e := chart_shift_up g (i - c) k c
  rw [Nat.sub_add_cancel h] at e
  omega

/-- **Rule M (index monotonicity).** A larger record index gives a larger address; the chart embeds
    the index order. This is what carries a cross-index bound like `(i-1)*stride ≤ count*stride`.

    (g : Graded) (i i' k : Nat)              [a region, two record indices, a shared offset]
    (h : i ≤ i')                             [the first index is at or below the second]
    chart g i k ≤ chart g i' k               [so the first address is at or below the second]
    ≤                                        [is smaller than or equal to] -/
theorem chart_mono (g : Graded) (i i' k : Nat) (h : i ≤ i') :
    chart g i k ≤ chart g i' k := by
  unfold chart
  have hm : i * g.stride ≤ i' * g.stride := Nat.mul_le_mul h (Nat.le_refl g.stride)
  omega

/-- **The access lands in the region -- for any in-range `(i,k)`.** Discharged ONCE; every strided
    access inherits it. Both hypotheses are load-bearing: `hi` bounds the record channel and `hk`
    the offset channel, and together they place the address strictly below `base + extent`.

    (g : Graded) (i k : Nat)                               [a region, a record index, an offset]
    (hi : i < g.count)                                     [the record index is in range]
    (hk : k < g.stride)                                    [the offset is inside its record]
    <                                                      [is smaller than]
    g.base ≤ chart g i k ∧ chart g i k < g.base + extent g [at or above base, below base+extent]
    ≤                                                      [is smaller than or equal to]
    ∧                                                      [both are true] -/
theorem graded_access_in_region (g : Graded) (i k : Nat)
    (hi : i < g.count) (hk : k < g.stride) :
    g.base ≤ chart g i k ∧ chart g i k < g.base + extent g := by
  unfold chart extent
  refine ⟨by omega, ?_⟩
  have h1 : i + 1 ≤ g.count := hi
  have h2 : (i + 1) * g.stride ≤ g.count * g.stride := Nat.mul_le_mul h1 (Nat.le_refl g.stride)
  have h3 : (i + 1) * g.stride = i * g.stride + g.stride := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- **Regions disjoint -- no overflow into a neighbour.** Two graded regions laid end to end: no
    in-range access of the first reaches the base of the second. The separation hypothesis `h`
    (the first region's extent stops at or before the second's base) is what forces it.

    (g1 g2 : Graded) (i k : Nat)             [two regions, a record index, an offset into the first]
    (hi : i < g1.count)                              [the record index is in range for the first]
    (hk : k < g1.stride)                             [the offset is inside its record in the first]
    (h : g1.base + extent g1 ≤ g2.base)       [the first region ends at or before the second begins]
    ≤                                                [is smaller than or equal to]
    chart g1 i k < g2.base                    [so the access stays strictly below the second's base]
    <                                                [is smaller than] -/
theorem graded_no_overflow_into_neighbor (g1 g2 : Graded) (i k : Nat)
    (hi : i < g1.count) (hk : k < g1.stride)
    (h : g1.base + extent g1 ≤ g2.base) :
    chart g1 i k < g2.base := by
  have hb := (graded_access_in_region g1 i k hi hk).2
  omega

/-- The channel split: an address decomposes into (record, offset) by the division algorithm.

    (g : Graded) (addr : Nat)    [a region, an address]
    decode g addr               [the pair (record index, offset), by dividing addr - base by stride]
    Nat × Nat                    [a pair: the record channel and the offset channel] -/
def decode (g : Graded) (addr : Nat) : Nat × Nat :=
  ((addr - g.base) / g.stride, (addr - g.base) % g.stride)

/-- **`chart ∘ decode = id`** on any address at or above `base`: the decomposition
    reconstructs the address exactly (the surjective half of the chart bijection -- every
    address is some channel pair).

    (g : Graded) (addr : Nat)                                      [a region, an address]
    (hb : g.base ≤ addr)                                           [the address is at or above base]
    ≤                                                              [is smaller than or equal to]
    chart g (decode g addr).1 (decode g addr).2 = addr             [recovers the address] -/
theorem chart_decode (g : Graded) (addr : Nat) (hb : g.base ≤ addr) :
    chart g (decode g addr).1 (decode g addr).2 = addr := by
  unfold chart decode
  show g.base + (addr - g.base) / g.stride * g.stride + (addr - g.base) % g.stride = addr
  have hdm : g.stride * ((addr - g.base) / g.stride) + (addr - g.base) % g.stride = addr - g.base :=
    Nat.div_add_mod (addr - g.base) g.stride
  have hc : (addr - g.base) / g.stride * g.stride = g.stride * ((addr - g.base) / g.stride) :=
    Nat.mul_comm _ _
  omega

/-- **`decode ∘ chart = id`** on an in-range offset: the two channels are independent and
    recover the original `(i,k)` (the injective half -- the channel split is exact). Both
    hypotheses are used: a positive stride and an offset below it are what make the division
    recover `i` and the remainder `k`.

    (g : Graded) (i k : Nat)                 [a region, a record index, an offset]
    (hs : 0 < g.stride)                      [the stride is positive, so the region has cells]
    (hk : k < g.stride)                      [the offset is inside its record]
    <                                        [is smaller than]
    decode g (chart g i k) = (i, k)    [decoding the charted cell recovers the record and offset] -/
theorem decode_chart (g : Graded) (i k : Nat) (hs : 0 < g.stride) (hk : k < g.stride) :
    decode g (chart g i k) = (i, k) := by
  unfold decode chart
  have hsub : g.base + i * g.stride + k - g.base = i * g.stride + k := by omega
  rw [hsub]
  have hcomm : i * g.stride = g.stride * i := Nat.mul_comm _ _
  have hd : (i * g.stride + k) / g.stride = i := by
    rw [hcomm, Nat.mul_add_div hs, Nat.div_eq_of_lt hk, Nat.add_zero]
  have hr : (i * g.stride + k) % g.stride = k := by
    rw [hcomm, Nat.mul_add_mod, Nat.mod_eq_of_lt hk]
  rw [hd, hr]

end Region
end Apeiron
