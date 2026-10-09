import Apeiron.Linter.Notation
import Apeiron.Machine.Encoding
import Apeiron.Numeric.Generic
import Apeiron.Table.Basic
import Apeiron.Trichotomy.AnyRing
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The numerics seam: the real model, error and calibration layers

@name Numeric Accumulator
@grade proved

Song's Track T3 seed. `song`'s runtime forward pass runs on finite-precision hardware: a
GEMV contraction and an attention score are integer or floating reductions over a fixed
width, not the exact-ℝ semantics the rest of this corpus reasons about.

`Apeiron/Numeric/Generic.lean` carries the exact algebra. This file is the real model layer on
top of it: the exponential online-softmax chart, the square-root recalibration, the relative-error
hypothesis and the integer token-margin check, and the instantiations of the generic reductions
and fusion at `ℝ` and `ℤ`.

Four facts, read top to bottom:

- **The accumulator must be sized to the operand range.** A reduction over `n` terms each
  bounded by `M` is bounded by `n·M` (`dot_abs_bounded`), and choosing an integer width whose
  capacity clears that bound is what makes the reduction never wrap (`accumulator_fits`). Both
  live in `Generic`; overflow is not a rounding error inside the model, it is a violation of the
  premise every bound below assumes.
- **Once in range, the rounding residue is carried and bounded.** `lossy_plus_carry_is_exact`
  is the exact bookkeeping, in `Generic`. The relative error model is the hypothesis of
  `lossy_residue_bounded` here, where `ε` and the real line enter.
- **The residue composes additively, not multiplicatively.** `lossy_error_is_additive` is in
  `Generic`: the forward pass's numeric error is itself a bounded accumulator, one level up.
- **The emitted token survives the residue when the reference margin clears it.**
  `token_survives_margin` is the runtime-checkable bridge from the additive bound to
  token-identity. Whether a given quantization scheme's `ε` clears a given margin is
  data-dependent and stays open; this file proves only the conditional.

Two further sections extend the seam to specific kernels a quantized forward leans on.

`§2` proves the quantization-dual facts a weight-only calibration floor cannot see: the
forward feels a residual through the calibration stream's per-channel energy, not through
`‖W − Ŵ‖²`, so a seal computed from the weights alone is blind to the one thing that decides
its output error. The exact identity is `Generic.output_error_is_function_distortion`; the
square-root chart and the order it is read through stay here.

`§3` proves the online-softmax (flash / streaming) attention kernel is exact, not an
approximation of the serial reference. The schedule-freedom is a generic additive-monoid fact,
`Generic.foldTotal_perm` and `Generic.foldTotal_chunks`; this file instantiates it on the tiles'
real values. The pair state is a numerically stable chart on a quantity that itself has the
identity the schedule argument needs. The pair carries a second channel: the drag and the
value-weighted numerator ride one max, and their two sums are the softmax attention ratio
(`osFusePair`, `osNum`, `osDen`, `softmax_attention_is_online_chart`).

## What is not ported

`numerics_seam`, the predecessor's bundling theorem, is a plain conjunction of the four
facts above with no content beyond the conjunction. `Apeiron/Table/Basic.lean` already
records why that shape does not belong in the corpus (the `Emergence.lean` precedent); it is
not ported here either.

The predecessor's `§N5` output-error identity assumed a diagonal Gram matrix (decoupled
calibration channels) and is ported at exactly that hypothesis; it is not claimed for a
general Gram matrix, and no attempt is made to remove the hypothesis here.

`§N6`'s online-softmax merge on `ℝ × ℝ` has no identity element: there is no finite `m`
making `(m, 0)` a two-sided unit, since a true "no contribution yet" state needs `m = -∞`.
The generic fusion is applied to the real-valued total after the merge is read off by `osVal`,
which lands in `(ℝ, +, 0)`, a monoid that does have an identity; `osFuse_fold_eq_add` is the
bridge between the two. The pair is a chart, not `Trichotomy.Cell`: no `ringMul` or
`ringNormForm` is read here.
-/

namespace Apeiron
namespace Numeric

/- ── §1. the real error model: a bounded relative residue ── -/

/-- The carried residue is bounded by the relative error.

    (rnd : ℝ → ℝ)                                    [a rounding map]
    (ε : ℝ)                                          [a relative error rate]
    (h : ∀ x, |rnd x - x| ≤ ε * |x|)                 [rnd stays within that rate at every point]
    (a b : ℝ)                                        [two reals]
    |(a + b) - rnd (a + b)| ≤ ε * |a + b|            [the dropped residue is within the rate]

    `ℝ` [the real numbers: every number on the number line]
    `→` [takes the left and gives the right; between statements, if the left holds so does
        the right]
    `∀` [for every single one]
    `≤` [is smaller than or equal to]

    The IEEE float guarantee for `rnd`, or a quantizer's codebook step, is exactly this
    hypothesis; the residue of one merge inherits its bound. This is the model layer: the
    exact bookkeeping `lossy_plus_carry_is_exact` is in `Generic`. -/
theorem lossy_residue_bounded (rnd : ℝ → ℝ) (ε : ℝ) (h : ∀ x, |rnd x - x| ≤ ε * |x|) (a b : ℝ) :
    |(a + b) - rnd (a + b)| ≤ ε * |a + b| := by
  have hb := h (a + b)
  rw [abs_sub_comm]
  exact hb

/-- The emitted token survives a bounded residue when the reference margin clears it.

    (ra rb sa sb ε : ℤ)                              [reference and perturbed logits, and a
                                                       shared error bound]
    (ha : |sa - ra| ≤ ε) (hb : |sb - rb| ≤ ε)        [each perturbed logit stays within ε of
                                                       its reference]
    (hgap : 2 * ε < ra - rb)                         [the reference winner clears the rival
                                                       by more than 2ε]
    sb < sa                                          [the perturbed winner is unchanged]

    `≤` [is smaller than or equal to]
    `<` [is smaller than]

    The runtime-checkable bridge from the additive error bound to token-identity: whether a
    given quantization's `ε` clears a given reference margin is data-dependent and stays
    open, but whenever it does, the argmax is provably unchanged. The logits are integers here,
    so this is a model instance of the additive bound, not a floating-point correspondence. -/
theorem token_survives_margin (ra rb sa sb ε : ℤ)
    (ha : |sa - ra| ≤ ε) (hb : |sb - rb| ≤ ε) (hgap : 2 * ε < ra - rb) : sb < sa := by
  rw [abs_le] at ha hb
  omega

/- ── §2. the quantization dual: the forward feels the calibration stream's channel energy ── -/

/-- The `√τ`-scaled chart turns the weighted distortion into plain squared error.

    {n : ℕ} (τ r : Fin n → ℝ)                        [channel energy and residual]
    (hτ : ∀ p, 0 ≤ τ p)                              [energy is nonnegative]
    functionDistortion τ r = ∑ p, (Real.sqrt (τ p) * r p) ^ 2
                                                       [the weighted distortion equals plain
                                                       squared error after scaling by √τ]

    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `ℝ` [the real numbers: every number on the number line]
    `∀` [for every single one]
    `≤` [is smaller than or equal to]
    `Real.sqrt` [the number that squares to give this]
    `functionDistortion` [the precision-weighted squared residual, over a per-channel energy]

    Minimizing ordinary MSE after quantizing `W ⊙ √τ` and folding `1/√τ` back minimizes the
    true function distortion; the α = 1/2 scaling exponent a calibration-aware quantizer uses
    is forced by this identity, not a tuned constant. `Real.sqrt` is the model layer; the exact
    identity `Generic.output_error_is_function_distortion` is what it is read through. -/
theorem aware_scale_is_the_chart {n : ℕ} (τ r : Fin n → ℝ) (hτ : ∀ p, 0 ≤ τ p) :
    functionDistortion τ r = ∑ p, (Real.sqrt (τ p) * r p) ^ 2 := by
  unfold functionDistortion
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [mul_pow, Real.sq_sqrt (hτ p)]

/-- A seal built from the weights alone cannot read the channel energy.

    {n : ℕ} (r : Fin n → ℝ)                          [a fixed nonzero residual]
    (∀ τ τ' : Fin n → ℝ, functionDistortion τ r - functionDistortion τ' r
        = ∑ p, (τ p - τ' p) * r p ^ 2)               [two duals' distortions differ by exactly
                                                       the energy difference, weighted by r]
    ∧ (∀ p0 : Fin n, r p0 ≠ 0 → ∀ C : ℝ, ∃ τ : Fin n → ℝ,
        (∀ p, 0 ≤ τ p) ∧ C < functionDistortion τ r)  [and for any nonzero residual channel,
                                                       some dual pushes the distortion past
                                                       any bound]

    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `ℝ` [the real numbers: every number on the number line]
    `∀` [for every single one]
    `∃` [there is at least one]
    `≠` [is not equal to]
    `≤` [is smaller than or equal to]
    `<` [is smaller than]
    `∧` [both are true]
    `functionDistortion` [the precision-weighted squared residual, over a per-channel energy]

    A floor computed from `W` alone fixes one `r` for every possible calibration dual; since
    the distortion is unbounded across duals whenever `r` has a nonzero channel, no
    weight-only floor bounds the true output error the forward pass will see. The division by
    `r p0 ^ 2` and the absolute value are the real order layer. -/
theorem weight_only_is_dual_blind {n : ℕ} (r : Fin n → ℝ) :
    (∀ τ τ' : Fin n → ℝ, functionDistortion τ r - functionDistortion τ' r
        = ∑ p, (τ p - τ' p) * r p ^ 2)
    ∧ (∀ p0 : Fin n, r p0 ≠ 0 → ∀ C : ℝ, ∃ τ : Fin n → ℝ,
        (∀ p, 0 ≤ τ p) ∧ C < functionDistortion τ r) := by
  refine ⟨fun τ τ' => ?_, fun p0 hr C => ?_⟩
  · unfold functionDistortion
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun p _ => by ring
  · have hr2 : (0 : ℝ) < r p0 ^ 2 := by positivity
    refine ⟨fun p => if p = p0 then (|C| + 1) / r p0 ^ 2 else 0, fun p => ?_, ?_⟩
    · by_cases h : p = p0 <;> simp only [h, if_true, if_false] <;> positivity
    · unfold functionDistortion
      rw [Finset.sum_eq_single p0
        (fun q _ hk => by simp [hk])
        (fun h => absurd (Finset.mem_univ p0) h)]
      simp only [if_true]
      rw [div_mul_cancel₀ _ (ne_of_gt hr2)]
      exact lt_of_le_of_lt (le_abs_self C) (lt_add_one _)

/- ── §3. flash / streaming attention: the tile merge is exact, and schedule-free ── -/

/-- The online-softmax state `(m, s)`: a running max paired with a max-shifted running sum.

    ℝ × ℝ → ℝ × ℝ                                    [combines two states into one]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]

    A chart on the single number `osVal`, chosen so no intermediate exponential overflows. The
    pair is not `Trichotomy.Cell`; it is read through `(ℝ, +, 0)`, and the carrier's own
    multiplication is not consulted. -/
noncomputable def osFuse (a b : ℝ × ℝ) : ℝ × ℝ :=
  (max a.1 b.1, a.2 * Real.exp (a.1 - max a.1 b.1) + b.2 * Real.exp (b.1 - max a.1 b.1))

/-- What the chart denotes: the un-normalized softmax drag `s · eᵐ`.

    ℝ × ℝ → ℝ                                         [a state to the real number it stands for]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side] -/
noncomputable def osVal (a : ℝ × ℝ) : ℝ := a.2 * Real.exp a.1

/-- The tile merge denotes addition, read through the chart.

    (a b : ℝ × ℝ)                                    [two online-softmax states]
    osVal (osFuse a b) = osVal a + osVal b            [merging the states denotes summing what
                                                       they stand for]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `osFuse` [combines two online-softmax states into one]
    `osVal` [an online-softmax state, to the un-normalized drag it stands for]

    The numerical-stability shift never leaves the monoid `(ℝ, +)`; it only changes which
    chart the merge is read through. This is the exponential model layer. -/
theorem osFuse_is_add (a b : ℝ × ℝ) : osVal (osFuse a b) = osVal a + osVal b := by
  simp only [osVal, osFuse, add_mul, mul_assoc, ← Real.exp_add]
  ring_nf

/-- Folding a list of tiles into a running state denotes the running value plus the tiles'
    total.

    (l : List (ℝ × ℝ)) (init : ℝ × ℝ)                [a list of tile states, a running state]
    osVal (l.foldl osFuse init)
      = osVal init + (l.map osVal).foldl (· + ·) 0    [the folded state's value is the running
                                                       value plus the plain sum of the tiles'
                                                       own values]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `List` [a finite sequence of things, in order]
    `osFuse` [combines two online-softmax states into one]
    `osVal` [an online-softmax state, to the un-normalized drag it stands for]

    The bridge from the pair-valued kernel state to the plain real-valued total that `§3`'s
    schedule-freedom theorems are proved about. -/
theorem osFuse_fold_eq_add (l : List (ℝ × ℝ)) (init : ℝ × ℝ) :
    osVal (l.foldl osFuse init) = osVal init + (l.map osVal).foldl (· + ·) 0 := by
  induction l generalizing init with
  | nil => simp
  | cons t rest ih =>
    rw [List.foldl_cons, ih (osFuse init t), osFuse_is_add init t, List.map_cons,
      List.foldl_cons, zero_add,
      Machine.foldl_from_acc (· + ·) (0 : ℝ) add_assoc zero_add add_zero (osVal t)
        (rest.map osVal)]
    ring

/-- The total value a list of tiles carries, independent of any running state.

    (l : List (ℝ × ℝ)) : ℝ                            [a list of tile states, to their total]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `List` [a finite sequence of things, in order]

    So: the tiles' values are read through `osVal` and summed by the generic `foldTotal`, which
    is where the additive monoid enters. -/
noncomputable def tileTotal (l : List (ℝ × ℝ)) : ℝ := foldTotal (l.map osVal)

/-- The legacy left-fold over the tiles' values is the generic tile total.

    (l : List (ℝ × ℝ))                               [a list of tile states]
    (l.map osVal).foldl (· + ·) 0 = tileTotal l       [the legacy fold is the generic total]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `List` [a finite sequence of things, in order]
    `tileTotal` [a list of tile states, to their total]

    So: this is `Generic.foldl_add_eq` at accumulator zero, read through the definition of
    `tileTotal`. It is the one place the legacy `List.foldl` spelling is related to the generic
    `foldTotal`, so a later proof rewrites through this theorem rather than asking a simplifier
    to unfold `foldTotal`. -/
theorem tileFold_eq_tileTotal (l : List (ℝ × ℝ)) :
    (l.map osVal).foldl (· + ·) 0 = tileTotal l := by
  unfold tileTotal foldTotal
  rw [foldl_add_eq, zero_add]

/-- Any permutation of the tile order leaves the total unchanged.

    {l₁ l₂ : List (ℝ × ℝ)}                            [two lists of tile states]
    (hperm : l₁.Perm l₂)                             [one a rearrangement of the other]
    tileTotal l₁ = tileTotal l₂                       [the same total]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `List` [a finite sequence of things, in order]
    `Perm` [is a rearrangement of]
    `tileTotal` [a list of tile states, to their total]

    So: this is `Generic.foldTotal_perm` at the tiles' real values. The generic fact is the
    additive monoid's, and `Table.build_gauge_invariance` is the hashmap-level statement of the
    same fact; no second bracketing argument is written. -/
theorem tileTotal_perm {l₁ l₂ : List (ℝ × ℝ)} (hperm : l₁.Perm l₂) :
    tileTotal l₁ = tileTotal l₂ :=
  foldTotal_perm (hperm.map osVal)

/-- Splitting the tile list into contiguous chunks, folding each independently and combining
    the partials, gives the same total as folding sequentially.

    (chunks : List (List (ℝ × ℝ)))                   [the tile list, split into contiguous
                                                       chunks]
    (chunks.map tileTotal).foldl (· + ·) 0 = tileTotal chunks.flatten
                                                       [combining each chunk's own total
                                                       matches the total of the flattened list]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `List` [a finite sequence of things, in order]
    `flatten` [the inner lists joined into one, in order]
    `tileTotal` [a list of tile states, to their total]

    So: this is `Generic.foldTotal_chunks` at the tiles' real values. This is what licenses a
    workgroup's per-lane partial reduction, each starting its own accumulator from `0`, to be
    recombined without error. -/
theorem tileTotal_chunks (chunks : List (List (ℝ × ℝ))) :
    (chunks.map tileTotal).foldl (· + ·) 0 = tileTotal chunks.flatten := by
  have h := foldTotal_chunks (chunks.map (List.map osVal))
  have hmap : (chunks.map (List.map osVal)).map foldTotal = chunks.map tileTotal := by
    simp [tileTotal]
  have hflat : (chunks.map (List.map osVal)).flatten = chunks.flatten.map osVal := by
    simp [List.map_flatten]
  rw [hmap, hflat] at h
  exact h

/-- The trichotomy carrier is an instance of the same additive fusion.

    {R : Type*} [CommRing R]                          [the ring the carrier is built over]
    {l₁ l₂ : List (Trichotomy.AnyRing.RingCell R)}    [two lists of carrier points]
    (hperm : l₁.Perm l₂)                             [one a rearrangement of the other]
    foldTotal l₁ = foldTotal l₂                       [the carrier-valued totals agree]

    `List` [a finite sequence of things, in order]
    `Perm` [is a rearrangement of]

    So: `RingCell R` is the two-slot carrier of `Apeiron.Trichotomy.AnyRing`, and its additive
    structure is one instance of the generic fusion. The online-softmax pair above is a chart on
    `(ℝ, +, 0)` and is not this carrier. The carrier's multiplicative merge is `ringMul`, whose
    schedule-freedom is `Table.build_gauge_invariance_mul`. -/
theorem ringCell_foldTotal_perm {R : Type*} [CommRing R]
    {l₁ l₂ : List (Trichotomy.AnyRing.RingCell R)} (hperm : l₁.Perm l₂) :
    foldTotal l₁ = foldTotal l₂ :=
  foldTotal_perm hperm

/-- The carrier's norm-form ledger of an order-independent total is order-independent.

    {R : Type*} [CommRing R] (σ : R)                  [the ring and the dial]
    {l₁ l₂ : List (Trichotomy.AnyRing.RingCell R)}    [two lists of carrier points]
    (hperm : l₁.Perm l₂)                              [one a rearrangement of the other]
    ringNormForm σ (foldTotal l₁) = ringNormForm σ (foldTotal l₂)
                                                       [the two ledger values agree]

    `σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    `List` [a finite sequence of things, in order]
    `Perm` [is a rearrangement of]

    So: `ringNormForm` is the carrier's size, and it reads the total the additive fusion
    produced. The accumulation and the ledger are the carrier's own operations, not a second
    pair. The carrier's multiplicative merge is `ringMul`, whose schedule-freedom is
    `Table.build_gauge_invariance_mul`. -/
theorem ringNormForm_foldTotal_perm {R : Type*} [CommRing R] (σ : R)
    {l₁ l₂ : List (Trichotomy.AnyRing.RingCell R)} (hperm : l₁.Perm l₂) :
    Trichotomy.AnyRing.ringNormForm σ (foldTotal l₁)
      = Trichotomy.AnyRing.ringNormForm σ (foldTotal l₂) := by
  rw [ringCell_foldTotal_perm hperm]

/-- Flash attention is exact and schedule-free (headline).

    (l₁ l₂ : List (ℝ × ℝ))                            [two tile orderings]
    (init : ℝ × ℝ)                                    [a running kernel state]
    (chunks : List (List (ℝ × ℝ)))                    [the same tiles, split into contiguous
                                                       chunks]
    l₁.Perm l₂ → osVal (l₁.foldl osFuse init) = osVal (l₂.foldl osFuse init)
                                                       [any permutation of the tile order
                                                       denotes the same accumulated drag]
    ∧ (chunks.map (fun c => osVal (c.foldl osFuse init) - osVal init)).foldl (· + ·) 0
        = osVal (chunks.flatten.foldl osFuse init) - osVal init
                                                       [and any contiguous per-lane chunking,
                                                       recombined, denotes the same drag]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `List` [a finite sequence of things, in order]
    `Perm` [is a rearrangement of]
    `→` [if the left holds so does the right]
    `∧` [both are true]
    `flatten` [the inner lists joined into one, in order]
    `osFuse` [combines two online-softmax states into one]
    `osVal` [an online-softmax state, to the un-normalized drag it stands for]

    No approximation enters the parallelization of the kernel: a workgroup's lane-vs-tree
    schedule and its chunk boundaries change nothing the kernel computes. The only numeric
    error left is the finite-precision residue of `§1`, unaffected by the fusion. This is an
    exact-algebra statement about the chart; it is not a claim about floating-point timing or
    correspondence. -/
theorem flash_attention_exact (l₁ l₂ : List (ℝ × ℝ)) (init : ℝ × ℝ)
    (chunks : List (List (ℝ × ℝ))) :
    (l₁.Perm l₂ → osVal (l₁.foldl osFuse init) = osVal (l₂.foldl osFuse init))
    ∧ (chunks.map (fun c => osVal (c.foldl osFuse init) - osVal init)).foldl (· + ·) 0
        = osVal (chunks.flatten.foldl osFuse init) - osVal init := by
  constructor
  · intro hperm
    rw [osFuse_fold_eq_add, osFuse_fold_eq_add]
    congr 1
    rw [tileFold_eq_tileTotal l₁, tileFold_eq_tileTotal l₂]
    exact tileTotal_perm hperm
  · have hchunk := tileTotal_chunks chunks
    have hL : (chunks.map (fun c => osVal (c.foldl osFuse init) - osVal init)).foldl (· + ·) 0
        = (chunks.map tileTotal).foldl (· + ·) 0 := by
      congr 1
      apply List.map_congr_left
      intro c _
      rw [osFuse_fold_eq_add]; unfold tileTotal foldTotal; ring
    rw [hL, hchunk, osFuse_fold_eq_add]
    unfold tileTotal foldTotal
    ring

/-- The max-shift is exact: subtracting a shared constant from the logits changes no softmax
    weight.

    {n : ℕ} (x : Fin n → ℝ) (c : ℝ) (i : Fin n)      [logits, a shared shift, a lane]
    Real.exp (x i + c) / ∑ p, Real.exp (x p + c) = Real.exp (x i) / ∑ p, Real.exp (x p)
                                                       [the shifted softmax weight equals the
                                                       unshifted one]

    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `ℝ` [the real numbers: every number on the number line]

    The kernel's per-workgroup running max is one such shift; the guarded `exp(score − m)` a
    pass-2 kernel computes is the exact softmax weight, not an approximation of it. -/
theorem softmax_shift_exact {n : ℕ} (x : Fin n → ℝ) (c : ℝ) (i : Fin n) :
    Real.exp (x i + c) / ∑ p, Real.exp (x p + c)
      = Real.exp (x i) / ∑ p, Real.exp (x p) := by
  simp only [Real.exp_add, ← Finset.sum_mul]
  rw [mul_div_mul_right _ _ (Real.exp_ne_zero c)]

/- ── §3b. the online numerator and denominator chart: one scan, both channels ── -/

/-- The numerator/denominator pair chart: one running max, a rescaled drag and a rescaled
    value-weighted numerator, both riding the same max.

    ℝ × ℝ × ℝ → ℝ × ℝ × ℝ → ℝ × ℝ × ℝ             [combines two pair states into one]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `osFuse` [combines two online-softmax states into one]

    So: the pair is the drag chart and the numerator chart side by side on one max. The max is
    data: `osVal` reads either channel without it. -/
noncomputable def osFusePair (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let d := osFuse (a.1, a.2.1) (b.1, b.2.1)
  let n := osFuse (a.1, a.2.2) (b.1, b.2.2)
  (d.1, d.2, n.2)

/-- The drag the pair chart denotes: `(m, den, num) ↦ den·eᵐ = Σ e^{sc}`.

    ℝ × ℝ × ℝ → ℝ                                     [a pair state to its un-normalized drag]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `osVal` [an online-softmax state, to the un-normalized drag it stands for]

    So: `osVal` on the drag channel, so the pair adds no second chart. -/
noncomputable def osDen (a : ℝ × ℝ × ℝ) : ℝ := osVal (a.1, a.2.1)

/-- The numerator the pair chart denotes: `(m, den, num) ↦ num·eᵐ = Σ e^{sc}·v`.

    ℝ × ℝ × ℝ → ℝ                                     [a pair state to its value-weighted
                                                       numerator]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `osVal` [an online-softmax state, to the un-normalized drag it stands for]

    So: `osVal` on the numerator channel, the same chart as the drag. -/
noncomputable def osNum (a : ℝ × ℝ × ℝ) : ℝ := osVal (a.1, a.2.2)

/-- The drag channel merges as exact `+`.

    (a b : ℝ × ℝ × ℝ)                                 [two pair states]
    osDen (osFusePair a b) = osDen a + osDen b        [merging the states adds their drags]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `osFuse` [combines two online-softmax states into one]

    So: `osFuse_is_add` on the drag channel. The pair merge is the single-channel merge twice,
    so the chart law is the single-channel law. -/
theorem osDen_fuse (a b : ℝ × ℝ × ℝ) :
    osDen (osFusePair a b) = osDen a + osDen b :=
  osFuse_is_add (a.1, a.2.1) (b.1, b.2.1)

/-- The numerator channel merges as exact `+`.

    (a b : ℝ × ℝ × ℝ)                                 [two pair states]
    osNum (osFusePair a b) = osNum a + osNum b        [merging the states adds their numerators]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `osFuse` [combines two online-softmax states into one]

    So: `osFuse_is_add` on the numerator channel, riding the same max. -/
theorem osNum_fuse (a b : ℝ × ℝ × ℝ) :
    osNum (osFusePair a b) = osNum a + osNum b :=
  osFuse_is_add (a.1, a.2.2) (b.1, b.2.2)

/-- One token as a pair tile `(sc, 1, v)`: drag `e^{sc}`, numerator `v·e^{sc}`.

    ℝ → ℝ → ℝ × ℝ × ℝ                                 [a score and a value, to one tile]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]

    So: the drag channel is one and the numerator channel is the value, so one tile carries
    both. -/
def osTile (sc v : ℝ) : ℝ × ℝ × ℝ := (sc, 1, v)

/-- The streaming scan: fold the score-value stream into one pair state, one `osFusePair` per
    token, from the state `(0, 0, 0)`.

    List (ℝ × ℝ) → ℝ × ℝ × ℝ                          [a stream of score-value pairs, to the
                                                       running pair state]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `List` [a finite sequence of things, in order]

    So: the single-pass flash traversal, carrying the numerator and the denominator together. -/
noncomputable def osScan (l : List (ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  l.foldl (fun acc p => osFusePair acc (osTile p.1 p.2)) (0, 0, 0)

/-- The drag of the scan is the running `Σ e^{sc}`.

    (l : List (ℝ × ℝ)) (acc : ℝ × ℝ × ℝ)             [a stream and a running pair state]
    osDen (l.foldl (fun a p => osFusePair a (osTile p.1 p.2)) acc)
      = osDen acc + (l.map (fun p => Real.exp p.1)).sum
                                                       [the folded drag is the running drag plus
                                                       the plain sum of the tile drags]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `List` [a finite sequence of things, in order]

    So: `osDen_fuse` along the fold. The drag accumulates exactly, tile by tile. -/
theorem osDen_scan_acc (l : List (ℝ × ℝ)) (acc : ℝ × ℝ × ℝ) :
    osDen (l.foldl (fun a p => osFusePair a (osTile p.1 p.2)) acc)
      = osDen acc + (l.map (fun p => Real.exp p.1)).sum := by
  induction l generalizing acc with
  | nil => simp
  | cons p ps ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons, ih]
    rw [osDen_fuse]
    simp only [osDen, osTile, osVal, one_mul]
    ring

/-- The numerator of the scan is the running `Σ e^{sc}·v`.

    (l : List (ℝ × ℝ)) (acc : ℝ × ℝ × ℝ)             [a stream and a running pair state]
    osNum (l.foldl (fun a p => osFusePair a (osTile p.1 p.2)) acc)
      = osNum acc + (l.map (fun p => Real.exp p.1 * p.2)).sum
                                                       [the folded numerator is the running
                                                       numerator plus the value-weighted sum]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `List` [a finite sequence of things, in order]

    So: `osNum_fuse` along the fold. The numerator accumulates exactly, tile by tile. -/
theorem osNum_scan_acc (l : List (ℝ × ℝ)) (acc : ℝ × ℝ × ℝ) :
    osNum (l.foldl (fun a p => osFusePair a (osTile p.1 p.2)) acc)
      = osNum acc + (l.map (fun p => Real.exp p.1 * p.2)).sum := by
  induction l generalizing acc with
  | nil => simp
  | cons p ps ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons, ih]
    rw [osNum_fuse]
    simp only [osNum, osTile, osVal]
    ring

/-- Softmax attention is the online numerator/denominator chart (headline).

    (l : List (ℝ × ℝ))                                [a stream of score-value pairs]
    osNum (osScan l) = (l.map (fun p => Real.exp p.1 * p.2)).sum
                                                       [the scan's numerator is the
                                                       value-weighted exponential sum]
    ∧ osDen (osScan l) = (l.map (fun p => Real.exp p.1)).sum
                                                       [and its drag is the exponential sum]
    ∧ (∀ a b : ℝ × ℝ × ℝ, osDen (osFusePair a b) = osDen a + osDen b
        ∧ osNum (osFusePair a b) = osNum a + osNum b)  [and both channels merge as exact `+`]

    `ℝ` [the real numbers: every number on the number line]
    `×` [paired with, so ℝ × ℝ means two real numbers side by side]
    `List` [a finite sequence of things, in order]
    `∀` [for every single one]
    `∧` [both are true]

    So: streaming the score-value tokens through one pair state, one `osFusePair` per token,
    reads the numerator and the drag as the two sums whose ratio is the softmax attention
    output. This is an exact-algebra statement about the chart, not a claim about
    floating-point timing, a query distribution or a runtime cost. -/
theorem softmax_attention_is_online_chart (l : List (ℝ × ℝ)) :
    osNum (osScan l) = (l.map (fun p => Real.exp p.1 * p.2)).sum
      ∧ osDen (osScan l) = (l.map (fun p => Real.exp p.1)).sum
      ∧ (∀ a b : ℝ × ℝ × ℝ, osDen (osFusePair a b) = osDen a + osDen b
            ∧ osNum (osFusePair a b) = osNum a + osNum b) := by
  refine ⟨?_, ?_, fun a b => ⟨osDen_fuse a b, osNum_fuse a b⟩⟩
  · rw [osScan, osNum_scan_acc]
    simp [osNum, osVal]
  · rw [osScan, osDen_scan_acc]
    simp [osDen, osVal]

end Numeric
end Apeiron
