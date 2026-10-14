import Spirv.Semantics
import Nous.Model.State
import Apeiron.Trichotomy.AnyRing
import Mathlib.Tactic

/-!
# The shared integer GEMV spec, its GPU-side refinement and the in-bounds obligations

The v0 GPU refinement of `docs/epics/gpu-spv.md`. Four pieces, all at the canonical face — the
`Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` object carried by the word face
`W32 = ZMod (2 ^ 32)`, the face `word_mul_denote`/`word_norm_denote`
(`proof/corpus/WordRefinement.lean`) fix for the extracted kernels. No standalone `ZMod (2 ^ w)`
foundation is introduced: every accumulation below is `ringMul`/`ringNormForm` of
`Apeiron.Trichotomy.AnyRing` read at the word face.

1. **The shared spec `dp4a_gemv_spec face`.** One row/segment output word of the integer GEMV: the
   emitted row guard `OpULessThan(r, M)`, the segment range of column chunks, the canonical
   channel-0 read of the per-byte-lane `ringMul` products, and the integer fixed-point requant
   `2 ^ scaleShift`. It is the single object both lanes refine.
2. **`gpu_dp4a_word_refines_cpu_word face`.** The GPU-side computation — the emitted `OpSUDot` phase
   pairs accumulated as `eval` denotes them (`eval_sudot`, `Spirv.Semantics`) and requantized by
   the emitted `OpConstant` `scaleConst` — refines the shared spec by `sdot_split_ringMul` and the
   fixed-point law `scale_fixed_point`; the CPU-side reading refines the same spec; the two links
   compose through `Refines.trans`, not by a second correspondence.
3. **The split-K fold reassociation.** `splitK_fold_reassoc`/`splitK_row_reassoc`: the `ringMul`
   fold over a split block list reassociates — `ringMul_assoc` through
   `Nous.Model.token_step_fold`; `splitK_normForm_fold` carries its `ringNormForm` read
   (`ringNormForm_mul`), the composition law of the per-block fixed-point scales.
4. **In bounds.** `accessChain_in_bounds`: every `OpAccessChain` index the guarded body emits
   (`WIDX = CC·M + R`, `DM2 = BB·M + R`, `bbx = BB`, `xqi = dpw·CC + l`, `OI2 = S·M + R`,
   `simt/dp4a.rs`) is below its buffer length under the dispatch parameters, the loop bounds and
   the row guard. Removing the guard leaves the obligation unsolved.

## What this file does not claim

The CPU *body* (the Rust GEMV) refining `cpu_dp4a_word`/`dp4a_gemv_spec` is the Aeneas lane's
obligation in the style of `proof/corpus/Numeric/Engine/Exp.lean`; `gpu_dp4a_word_refines_cpu_word`
proves the GPU-side link and the transitivity composition with that link, at the shared spec. The
module-level statement `denote (decode ws) = dp4a_gemv_spec …` over the concrete
`gemv_dp4a_module_dyn` words is not proved here: the emitter `Kb` is unproved and named, the
emitted words are not decoded in this corpus (`Spirv.Decode` proves the decoder itself), and the
float epilogue is the named migration item (`docs/epics/gpu-spv.md` epilogue premise). The
`GlobalInvocationId` grid assembly and the structured-CFG termination premise remain the
well-formedness premise of `docs/epics/gpu-spv.md`; the indices the loop nest computes are what
`accessChain_in_bounds` bounds. The face ledger stands: the realized face is
`Spirv.Semantics.realizedSigma` (`σ = +1`, `sdot_split_ringMul`), only channel 0 is computed
(`cross_channel_not_realized`, the emitter migration item), and the 2-adic caveat
(`two_not_unit`; `hyperbolic_decomposition`'s `2 * h = 1` does not transport to `W32`) is
unchanged.

## Controls

Throwaway copies: (a) `rowGuard` rewritten to `true` — the model of the
removed `OpULessThan(r, M)` row guard — leaves `accessChain_in_bounds` unsolved at `r < d.M`;
(b) `scaleConst` flipped off `2 ^ 15` turns `scale_fixed_point`, `gpu_dp4a_refines_shared`,
`gpu_dp4a_word_refines_cpu_word` and the `#guard` refinement comparison red. Restored: green.

## Complexity

`dp4aChunkWord`/`dp4aChunkCell`/`dp4aChunkDot` are O(1): four byte lanes, fixed (`Fin 4`), in lane
products and additions. `dp4aSegWord`/`dp4aSegWordDot` are O(K / segs) additions in column chunks
(the `Finset.Ico` sum over the segment's chunk range); `dp4a_gemv_spec`/`gpu_dp4a_word`/
`cpu_dp4a_word` inherit that count. `splitK_normForm_fold` is O(|xs|) in `ringNormForm` calls and
multiplications (the list induction). `accessChain_in_bounds` and the `Refines` links are proofs,
no runtime cost.
-/

namespace Spirv

open Apeiron.Trichotomy.AnyRing
open Nous.Model
open BigOperators

/-- The packed phase pair of one column chunk: the even-phase and the odd-phase word of the
emitted `OpSUDot` pair. -/
structure PhasePair where
  /-- The even-phase packed word (the first `OpSUDot` operand pair slot). -/
  ev : Word32
  /-- The odd-phase packed word (the second `OpSUDot` operand pair slot). -/
  od : Word32

/-- The chunk contribution at the face, as the canonical object: the channel-0 read of the
`ringMul` products of the four byte-lane cells (`actCell`/`wgtCell`, `Spirv.Semantics`).

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def dp4aChunkWord (face : Word32) (x w : PhasePair) : Word32 :=
  ∑ i : Fin 4, (ringMul face (actCell x.ev x.od i) (wgtCell w.ev w.od i)).1

/-- The chunk contribution cell at the face: the summed `ringMul` products of the four byte-lane
cells, before the channel-0 read. O(1): four fixed lanes.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def dp4aChunkCell (face : Word32) (x w : PhasePair) : RingCell W32 :=
  ∑ i : Fin 4, ringMul face (actCell x.ev x.od i) (wgtCell w.ev w.od i)

/-- The contribution cells of row `r`, one per column chunk. O(K): one `dp4aChunkCell` per chunk.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def chunkCells (face : Word32) (chunks : Nat) (X W : Nat → Nat → PhasePair) (r : Nat) :
    List (RingCell W32) :=
  (List.range chunks).map (fun c => dp4aChunkCell face (X r c) (W r c))

/-- The emitted row guard `OpULessThan(r, M)` and its `OpBranchConditional`: the guarded body runs
exactly for `r < M`. The control removes this guard.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def rowGuard (r M : Nat) : Bool := decide (r < M)

/-- The fixed-point requant multiplier the module carries as `OpConstant` (Q15).

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def scaleConst : Word32 := 2 ^ 15

/-- The CPU-side fixed-point requant shift.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def scaleShift : Nat := 15

/-- The fixed-point law binding the emitted constant to the CPU-side shift. The GPU-side
refinement carries this law; flipping `scaleConst` turns it and the refinement red.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem scale_fixed_point : scaleConst = (2 ^ scaleShift : Word32) := rfl

/-- The word of one split-K segment of row `r`: the channel-0 reads of the canonical `ringMul`
contributions summed over the segment's column-chunk range. O(K / segs) additions in chunks.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def dp4aSegWord (face : Word32) (K segs y : Nat) (X W : Nat → Nat → PhasePair) (r : Nat) :
    Word32 :=
  ∑ c ∈ Finset.Ico (y * (K / segs)) ((y + 1) * (K / segs)), dp4aChunkWord face (X r c) (W r c)

/-- The emitted `OpSUDot` chunk contribution: the two phase dots as `eval` denotes them
(`eval_sudot`, `Spirv.Semantics`), the accumulator update `ACCQVAR += dot`. O(1).

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def dp4aChunkDot (x w : PhasePair) : Word32 :=
  evalSUDot x.ev w.ev + evalSUDot x.od w.od

/-- The emitted segment computation before requant: the `OpSUDot` phase dots summed over the
segment's chunk range. O(K / segs) additions in chunks.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def dp4aSegWordDot (K segs y : Nat) (X W : Nat → Nat → PhasePair) (r : Nat) : Word32 :=
  ∑ c ∈ Finset.Ico (y * (K / segs)) ((y + 1) * (K / segs)), dp4aChunkDot (X r c) (W r c)

/-- **The shared spec `dp4a_gemv_spec face`.** The output word of row `r`, segment `y`: under the
emitted row guard and the segment bound, the fixed-point requant `2 ^ scaleShift` — the shift
form of the module's one scale, the emitted `OpConstant` `scaleConst` (`scale_fixed_point`) — of
the canonical channel-0 read of the per-lane `ringMul` products; `0` outside the guarded region.
This is the face-parameterized instance of
`Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` both lanes refine. O(K / segs).

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def dp4a_gemv_spec (face : Word32) (M K segs : Nat)
    (X W : Nat → Nat → PhasePair) (r y : Nat) : Word32 :=
  if rowGuard r M = true ∧ y < segs then
    (2 ^ scaleShift : Word32) * dp4aSegWord face K segs y X W r
  else 0

/-- The GPU-side row/segment word: the emitted `OpSUDot` accumulation requantized by the emitted
constant `scaleConst`. O(K / segs).

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def gpu_dp4a_word (M K segs : Nat) (X W : Nat → Nat → PhasePair) (r y : Nat) : Word32 :=
  if rowGuard r M = true ∧ y < segs then
    scaleConst * dp4aSegWordDot K segs y X W r
  else 0

/-- The CPU-side reading: the same `OpSUDot` accumulation requantized by the shift form
`2 ^ scaleShift`, the body the Aeneas lane refines. The module carries one fixed-point scale, so
the spec fixes it rather than parameterizing; a per-block scale composition is the `ringNormForm`
law `splitK_normForm_fold`. O(K / segs).

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def cpu_dp4a_word (M K segs : Nat) (X W : Nat → Nat → PhasePair) (r y : Nat) :
    Word32 :=
  if rowGuard r M = true ∧ y < segs then
    (2 ^ scaleShift : Word32) * dp4aSegWordDot K segs y X W r
  else 0

/-- Refinement at the word face: equality of the denoted output words. The two lanes meet at the
shared spec, so refinement composes by transitivity, not by a second correspondence. -/
abbrev Refines (a b : Word32) : Prop := a = b

/-- Refinement is transitive.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem Refines.trans {a b c : Word32} (hab : Refines a b) (hbc : Refines b c) : Refines a c :=
  Eq.trans hab hbc

/-- Refinement is symmetric.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem Refines.symm {a b : Word32} (hab : Refines a b) : Refines b a := Eq.symm hab

/-- The chunk-level link: the two `OpSUDot` phase dots equal the canonical channel-0 read of the
`ringMul` products of the lane cells. This is `Spirv.Semantics.sdot_split_ringMul`, the layout
theorem at the realized face `σ = +1`.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem chunk_dot_refines (x w : PhasePair) :
    dp4aChunkDot x w = dp4aChunkWord realizedSigma x w :=
  sdot_split_ringMul x.ev x.od w.ev w.od

/-- The segment-level link at the realized face: the emitted phase-dot sum equals the canonical
channel-0 read of the `ringMul` contributions over the same chunk range.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem seg_dot_refines (face : Word32) (hface : face = realizedSigma) (K segs y : Nat)
    (X W : Nat → Nat → PhasePair) (r : Nat) :
    dp4aSegWordDot K segs y X W r = dp4aSegWord face K segs y X W r := by
  rw [hface]
  exact Finset.sum_congr rfl fun c _ => chunk_dot_refines (X r c) (W r c)

/-- The GPU-side link into the shared spec: the emitted computation refines
`dp4a_gemv_spec face`, through `sdot_split_ringMul` and the fixed-point law.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem gpu_dp4a_refines_shared (face : Word32) (hface : face = realizedSigma)
    (M K segs : Nat) (X W : Nat → Nat → PhasePair) (r y : Nat) :
    Refines (gpu_dp4a_word M K segs X W r y)
      (dp4a_gemv_spec face M K segs X W r y) := by
  unfold gpu_dp4a_word dp4a_gemv_spec
  by_cases h : rowGuard r M = true ∧ y < segs
  · rw [if_pos h, if_pos h, seg_dot_refines face hface K segs y X W r, scale_fixed_point]
  · rw [if_neg h, if_neg h]

/-- The CPU-side link into the shared spec: the shift-form reading refines `dp4a_gemv_spec face`
through `sdot_split_ringMul`. The CPU *body* refining this reading is the Aeneas lane's
obligation, named and not claimed here.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem cpu_dp4a_refines_shared (face : Word32) (hface : face = realizedSigma)
    (M K segs : Nat) (X W : Nat → Nat → PhasePair) (r y : Nat) :
    Refines (cpu_dp4a_word M K segs X W r y)
      (dp4a_gemv_spec face M K segs X W r y) := by
  unfold cpu_dp4a_word dp4a_gemv_spec
  by_cases h : rowGuard r M = true ∧ y < segs
  · rw [if_pos h, if_pos h, seg_dot_refines face hface K segs y X W r]
  · rw [if_neg h, if_neg h]

/-- **The GPU-side word refinement `gpu_dp4a_word_refines_cpu_word face`.** The GPU-side computation
refines the CPU-side reading of the shared spec, composed by transitivity of refinement through
`dp4a_gemv_spec face` — one correspondence at the canonical object, not two. With the CPU body's
link into `cpu_dp4a_word` (the Aeneas lane), this is the transitive claim of
`docs/epics/gpu-spv.md`'s contract at the word model. The contract name
`gpu_dp4a_refines_cpu_spec` is reserved for the unproved decode-level statement
`denote (decode ws) = dp4a_gemv_spec …`.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem gpu_dp4a_word_refines_cpu_word (face : Word32) (hface : face = realizedSigma)
    (M K segs : Nat) (X W : Nat → Nat → PhasePair) (r y : Nat) :
    Refines (gpu_dp4a_word M K segs X W r y)
      (cpu_dp4a_word M K segs X W r y) :=
  Refines.trans (gpu_dp4a_refines_shared face hface M K segs X W r y)
    (Refines.symm (cpu_dp4a_refines_shared face hface M K segs X W r y))

/-- **The split-K fold reassociation.** The `ringMul` fold over a split block list is the
`ringMul` of the two sub-folds: `ringMul_assoc` through `Nous.Model.token_step_fold`. Block
accumulation and host-side split-K over the same block list meet here.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem splitK_fold_reassoc (face : Word32) (xs ys : List (RingCell W32)) :
    tokenFold face (xs ++ ys) =
      ringMul face (tokenFold face xs) (tokenFold face ys) :=
  token_step_fold face xs ys

/-- The split-K fold over the row's own contribution-cell list: any split of the block list
reassociates the `ringMul` fold (`List.take_append_drop`, `token_step_fold`).

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem splitK_row_reassoc (face : Word32) (chunks : Nat) (X W : Nat → Nat → PhasePair)
    (r n : Nat) :
    tokenFold face (chunkCells face chunks X W r) =
      ringMul face (tokenFold face ((chunkCells face chunks X W r).take n))
        (tokenFold face ((chunkCells face chunks X W r).drop n)) := by
  have h : (chunkCells face chunks X W r).take n ++ (chunkCells face chunks X W r).drop n
      = chunkCells face chunks X W r := by
    simp [List.take_append_drop]
  conv_lhs => rw [← h]
  exact token_step_fold face _ _

/-- The `ringNormForm` read of the fold: the norm form of the `ringMul` fold is the product of the
norm forms — `ringNormForm_mul` through the fold. This is the composition law of the per-block
fixed-point scales under split-K.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem splitK_normForm_fold (face : Word32) (xs : List (RingCell W32)) :
    ringNormForm face (tokenFold face xs) = (xs.map (ringNormForm face)).prod := by
  induction xs with
  | nil => simp [tokenFold, ringNormForm, ringOne]
  | cons x xs ih =>
    have hfold : tokenFold face (x :: xs) = ringMul face x (tokenFold face xs) := by
      simp only [tokenFold, List.foldl_cons]
      rw [foldl_ringMul_accum, ringOne_mul]
    rw [hfold, ringNormForm_mul face x (tokenFold face xs), ih, List.map_cons, List.prod_cons]

/-! ## In bounds: every emitted `OpAccessChain` index under the dispatch parameters -/

/-- The dispatch parameters of the served module (`gemv_dp4a_module_dyn`): rows `M`, 32-blocks
`blocks`, chunks per block `cpb`, dots per weight word `dpw`, split-K segments `segs`. -/
structure Dispatch where
  /-- Rows of the output, the bound of the emitted row guard `OpULessThan(r, M)`. -/
  M : Nat
  /-- 32-blocks of the reduction. -/
  blocks : Nat
  /-- Column chunks per 32-block. -/
  cpb : Nat
  /-- `OpSUDot` phases per weight word (1 for Q8, 2 for Q4). -/
  dpw : Nat
  /-- Split-K segments. -/
  segs : Nat

/-- One emitted `OpAccessChain` site: the index the body computes and the length of the buffer it
indexes. -/
structure ChainSite where
  /-- The computed index. -/
  idx : Nat
  /-- The length of the indexed buffer. -/
  len : Nat

/-- The `OpAccessChain` sites the guarded body emits for one invocation, with the index
expressions as the emitter computes them (`simt/dp4a.rs`, served path): weight `W[CC·M + R]` at
`CC = BB·CPB + JJ`, scales `DM[BB·M + R]`, block sums `XS[BB]`, activations
`XQ[dpw·CC + l]`, output `OUT[S·M + R]`. O(1): five sites.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def emittedAccessChains (d : Dispatch) (r y bb jj ll : Nat) : List (Nat × Nat) :=
  [ ((bb * d.cpb + jj) * d.M + r, d.blocks * d.cpb * d.M),
    (bb * d.M + r, d.blocks * d.M),
    (bb, d.blocks),
    ((bb * d.cpb + jj) * d.dpw + ll, d.blocks * d.cpb * d.dpw),
    (y * d.M + r, d.segs * d.M) ]

/-- A guarded row index is below the row-block bound. The row guard supplies `r < M`.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem row_index_lt {c cd M r : Nat} (hc : c < cd) (hr : r < M) : c * M + r < cd * M := by
  have h1 : c * M + r < (c + 1) * M := by
    rw [Nat.succ_mul]
    exact Nat.add_lt_add_left hr (c * M)
  have h2 : (c + 1) * M ≤ cd * M :=
    Nat.mul_le_mul (Nat.succ_le_of_lt hc) (Nat.le_refl M)
  exact Nat.lt_of_lt_of_le h1 h2

/-- A block-local chunk index is below the block-strided bound.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem chunk_index_lt {bb blocks cpb jj : Nat} (hbb : bb < blocks) (hjj : jj < cpb) :
    bb * cpb + jj < blocks * cpb := by
  have h1 : bb * cpb + jj < (bb + 1) * cpb := by
    rw [Nat.succ_mul]
    exact Nat.add_lt_add_left hjj (bb * cpb)
  have h2 : (bb + 1) * cpb ≤ blocks * cpb :=
    Nat.mul_le_mul (Nat.succ_le_of_lt hbb) (Nat.le_refl cpb)
  exact Nat.lt_of_lt_of_le h1 h2

/-- **In bounds.** Every `OpAccessChain` index the guarded body emits is below its buffer length,
under the dispatch parameters, the loop bounds and the emitted row guard `OpULessThan(r, M)`. The
guard is the load-bearing term: removing it leaves `r < d.M` undischarged and the obligation
unsolved.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
theorem accessChain_in_bounds (d : Dispatch) (r y bb jj ll : Nat)
    (hg : rowGuard r d.M = true) (hy : y < d.segs) (hbb : bb < d.blocks)
    (hjj : jj < d.cpb) (hll : ll < d.dpw) :
    ∀ p ∈ emittedAccessChains d r y bb jj ll, p.1 < p.2 := by
  have hrow : r < d.M := by
    unfold rowGuard at hg
    exact of_decide_eq_true hg
  intro p hp
  simp only [emittedAccessChains, List.mem_cons] at hp
  rcases hp with hp | hp | hp | hp | hp | hp
  · subst hp
    exact row_index_lt (chunk_index_lt hbb hjj) hrow
  · subst hp
    exact row_index_lt hbb hrow
  · subst hp
    exact hbb
  · subst hp
    exact row_index_lt (chunk_index_lt hbb hjj) hll
  · subst hp
    exact row_index_lt hy hrow
  · simp at hp

/-! ## Executable comparisons (the controls read these) -/

/-- Sample activation data for the executable comparisons.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def sampleX : Nat → Nat → PhasePair := fun _ _ => ⟨1, 0⟩

/-- Sample weight data for the executable comparisons.

`→` [takes the left and gives the right] `×` [paired with] `<` [is smaller than] `≤` [is smaller than or equal to] `∧` [both are true] `¬` [this is not true] `∀` [for every single one] `∃` [there is at least one] `∈` [is one of the things in] `++` [the two lists joined, the left one's entries first] `List` [a finite sequence of things, in order] `Bool` [true or false, nothing else] `RingCell` [a point: two ring elements side by side, a + b·x] `ringMul` [multiply two points together, the σ way] `ringNormForm` [the norm form: a point times its conjugate, which can come out negative] -/
def sampleW : Nat → Nat → PhasePair := fun _ _ => ⟨1, 0⟩

-- The guard's polarity: the emitted `OpULessThan(r, M)` accepts `r < M` and refuses `r = M`.
#guard rowGuard 3 4 = true
#guard rowGuard 4 4 = false

-- The fixed-point scale comparison: the emitted constant equals the CPU-side shift form.
#guard scaleConst = 2 ^ scaleShift

-- The chunk-level comparison: the phase-dot pair equals the canonical `ringMul` read.
#guard dp4aChunkDot ⟨1, 0⟩ ⟨1, 0⟩ = dp4aChunkWord realizedSigma ⟨1, 0⟩ ⟨1, 0⟩

-- The segment datapath on the sample: four chunks, one contribution each.
#guard dp4aSegWordDot 4 1 0 sampleX sampleW 0 = 4
#guard dp4aSegWord realizedSigma 4 1 0 sampleX sampleW 0 = 4

-- The refinement comparison on a concrete instance: flipping `scaleConst` turns this red.
#guard gpu_dp4a_word 1 4 1 sampleX sampleW 0 0
    = cpu_dp4a_word 1 4 1 sampleX sampleW 0 0

end Spirv
