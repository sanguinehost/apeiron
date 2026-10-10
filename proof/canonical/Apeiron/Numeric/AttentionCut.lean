import Apeiron.Fragment.Basic
import Apeiron.Linter.Notation
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.Complex.Exponential
import Mathlib.LinearAlgebra.LinearIndependent.Basic
import Mathlib.LinearAlgebra.Matrix.DotProduct

/-!
# The certified cut: when skipping keys leaves softmax attention exact

@name Attention Cut
@grade proved
@cite farkas-1902

Softmax attention at one query reads every key: the output is `N / Z`, with
`Z = Σ e^{sᵢ}` and `N = Σ e^{sᵢ}·vᵢ` over all keys. A cheaper reader skips some keys and
summarises others. This file proves when the cheaper reader is exact to a stated bound, and
proves the floor no cheaper reader can go under.

**Any certified pair bounds the output** (`ratio_error`, `certified_read`). If a reader returns
`N'` and `Z'` with `|N − N'| ≤ εN` and `|Z − Z'| ≤ εZ`, its output is within
`(εN + V·εZ) / Z'` of softmax, where `V` bounds the values. Every quantity on the right is
known to the reader, so the bound is checked at run time, per query.

**Dropped keys cost at most their count times `e^{−Δ}`** (`dropped_mass_bound`). When every
dropped score sits `Δ` below a kept score, the output moves by at most `2·V·|D|·e^{−Δ}`.

**A Farkas certificate proves the drop without reading the keys** (`farkas_prunes`). A node
stores its keys' region as rows. A nonnegative combination of those rows and the score row
that comes out a negative constant proves no key in the region scores at or above the
threshold. This is `Fragment.unsat_of_negative_combo`, the checker `../song`'s prover uses.

**Close keys are carried by moments** (`moment_remainder`, `moment_one_stored`,
`moment_two_stored`). When a node's scores lie within `B ≤ 1` of a centre, its exponential
sum is its first `p` moments up to `e^c·(Σ|vᵢ|)·B^p·(p+1)/(p!·p)`, and the first two moments
are stored once, independent of the query.

**The errors of a partition add** (`partition_error`). A reader that splits the keys into
parts, each read exactly, dropped or summarised, is off by at most the sum of the parts'
bounds.

**The floor** (`keys_determined`). A state that returns `Σ cₖ·e^{q·k}` exactly for every
query determines every weight `cₖ`: the maps `q ↦ e^{q·k}` for distinct keys are distinct
characters of the additive group, and distinct characters are linearly independent
(`linearIndependent_monoidHom`). An exact reader for all queries therefore stores the keys.
Exactness with less memory holds only per query, through the certificates above.

## What this does not settle

How many keys a real query must read once the certificates are applied: the size of the cut.
That is a measurement on a model, not a theorem here. Floating-point rounding of the reader
itself is `Accumulator.lean`'s layer; the bounds here are in exact reals.
-/

namespace Apeiron
namespace Numeric
namespace AttentionCut

open Finset Matrix

/- ── §1. any certified pair bounds the output ── -/

/-- The output moves by the numerator error plus `V` times the denominator error.

    (N Z N' Z' V : ℝ)                        [a numerator, a denominator, their stand-ins, a
                                              bound on the values]
    (hZ : 0 < Z) (hZ' : 0 < Z')              [both denominators are above zero]
    (hN : |N| ≤ V * Z)                       [the true output is at most V in size]
    |N / Z - N' / Z'| ≤ (|N - N'| + V * |Z - Z'|) / Z'
                                             [the two outputs differ by at most the numerator
                                              error plus V times the denominator error, over
                                              the stand-in denominator]
    `ℝ` [the real numbers: every number on the number line]
    `<` [is smaller than]
    `≤` [is smaller than or equal to]

    So: `N/Z − N'/Z' = (Z·(N − N') − N·(Z − Z')) / (Z·Z')`, and `|N| ≤ V·Z` cancels the `Z`. -/
theorem ratio_error (N Z N' Z' V : ℝ) (hZ : 0 < Z) (hZ' : 0 < Z') (hN : |N| ≤ V * Z) :
    |N / Z - N' / Z'| ≤ (|N - N'| + V * |Z - Z'|) / Z' := by
  have hid : N / Z - N' / Z' = (Z * (N - N') - N * (Z - Z')) / (Z * Z') := by
    field_simp
    ring
  have hnum : |Z * (N - N') - N * (Z - Z')| ≤ Z * (|N - N'| + V * |Z - Z'|) := by
    calc |Z * (N - N') - N * (Z - Z')|
        ≤ |Z * (N - N')| + |N * (Z - Z')| := abs_sub _ _
      _ = Z * |N - N'| + |N| * |Z - Z'| := by
          rw [abs_mul, abs_mul, abs_of_pos hZ]
      _ ≤ Z * |N - N'| + V * Z * |Z - Z'| := by
          gcongr
      _ = Z * (|N - N'| + V * |Z - Z'|) := by ring
  rw [hid, abs_div, abs_of_pos (mul_pos hZ hZ'), div_le_div_iff₀ (mul_pos hZ hZ') hZ']
  calc |Z * (N - N') - N * (Z - Z')| * Z'
      ≤ Z * (|N - N'| + V * |Z - Z'|) * Z' := by gcongr
    _ = (|N - N'| + V * |Z - Z'|) * (Z * Z') := by ring

/-- A reader whose two sums carry stated error bounds is within a bound it can compute.

    (N Z N' Z' V εN εZ : ℝ)                  [the true sums, the reader's sums, a value bound,
                                              and the two error bounds]
    (hZ : 0 < Z) (hZ' : 0 < Z')              [both denominators are above zero]
    (hN : |N| ≤ V * Z)                       [the true output is at most V in size]
    (hV : 0 ≤ V)                             [the value bound is not negative]
    (heN : |N - N'| ≤ εN)                    [the numerator error is within its bound]
    (heZ : |Z - Z'| ≤ εZ)                    [the denominator error is within its bound]
    |N / Z - N' / Z'| ≤ (εN + V * εZ) / Z'   [the output error is within the computable bound]
    `ℝ` [the real numbers: every number on the number line]
    `<` [is smaller than]
    `≤` [is smaller than or equal to]

    So: `ratio_error` with the two errors replaced by their bounds. The right side uses only
    what the reader holds, which is what makes it a run-time certificate. -/
theorem certified_read (N Z N' Z' V εN εZ : ℝ) (hZ : 0 < Z) (hZ' : 0 < Z')
    (hN : |N| ≤ V * Z) (hV : 0 ≤ V) (heN : |N - N'| ≤ εN) (heZ : |Z - Z'| ≤ εZ) :
    |N / Z - N' / Z'| ≤ (εN + V * εZ) / Z' := by
  refine (ratio_error N Z N' Z' V hZ hZ' hN).trans ?_
  gcongr

/- ── §2. dropped keys cost at most their count times e^{−Δ} ── -/

/-- The value-weighted exponential sum of a set of keys: softmax's numerator.

    {ι : Type*} (S : Finset ι) (s v : ι → ℝ) [a set of keys, their scores, their values]
    : ℝ                                      [the sum of e^{score} times value]
    `ℝ` [the real numbers: every number on the number line]
    `→` [takes the left to the right; between statements, the left gives the right] -/
noncomputable def num {ι : Type*} (S : Finset ι) (s v : ι → ℝ) : ℝ :=
  ∑ i ∈ S, Real.exp (s i) * v i

/-- The exponential sum of a set of keys: softmax's denominator.

    {ι : Type*} (S : Finset ι) (s : ι → ℝ)   [a set of keys and their scores]
    : ℝ                                      [the sum of e^{score}]
    `ℝ` [the real numbers: every number on the number line]
    `→` [takes the left to the right; between statements, the left gives the right] -/
noncomputable def den {ι : Type*} (S : Finset ι) (s : ι → ℝ) : ℝ :=
  ∑ i ∈ S, Real.exp (s i)

/-- A set's numerator is at most the value bound times its denominator.

    {ι : Type*} (S : Finset ι) (s v : ι → ℝ) (V : ℝ)
                                             [keys, scores, values, a value bound]
    (hv : ∀ i ∈ S, |v i| ≤ V)                [every value in the set is at most V in size]
    |num S s v| ≤ V * den S s                [the numerator is at most V times the denominator]
    `ℝ` [the real numbers: every number on the number line]
    `→` [takes the left to the right; between statements, the left gives the right]
    `∀` [for every single one]
    `∈` [is one of the things in]
    `≤` [is smaller than or equal to] -/
theorem num_le {ι : Type*} (S : Finset ι) (s v : ι → ℝ) (V : ℝ)
    (hv : ∀ i ∈ S, |v i| ≤ V) : |num S s v| ≤ V * den S s := by
  unfold num den
  rw [mul_sum]
  refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun i hi => ?_)
  rw [abs_mul, abs_of_pos (Real.exp_pos _), mul_comm V]
  exact mul_le_mul_of_nonneg_left (hv i hi) (Real.exp_pos _).le

/-- Dropping keys that score `Δ` below a kept key moves the output by at most
    `2·V·|D|·e^{−Δ}`.

    {ι : Type*} [DecidableEq ι] (K D : Finset ι) (s v : ι → ℝ) (V m Δ : ℝ)
                                             [kept and dropped keys, scores, values, a value
                                              bound, a kept score level, a margin]
    (hKD : Disjoint K D)                     [no key is both kept and dropped]
    (hv : ∀ i ∈ K ∪ D, |v i| ≤ V)            [every value is at most V in size]
    (j : ι) (hj : j ∈ K) (hm : m ≤ s j)      [some kept key scores at least m]
    (hD : ∀ i ∈ D, s i ≤ m - Δ)              [every dropped key scores at most m − Δ]
    |num (K ∪ D) s v / den (K ∪ D) s - num K s v / den K s|
      ≤ 2 * V * D.card * Real.exp (-Δ)       [the full output and the kept output differ by
                                              at most 2·V·|D|·e^{−Δ}]
    `ℝ` [the real numbers: every number on the number line]
    `→` [takes the left to the right; between statements, the left gives the right]
    `∀` [for every single one]
    `∈` [is one of the things in]
    `∪` [the keys in either set]
    `≤` [is smaller than or equal to]
    `Real.exp` [e raised to this power]

    So: `ratio_error` with the dropped keys as both errors, then each dropped term is at most
    `e^{m−Δ}` and the kept denominator is at least `e^m`. -/
theorem dropped_mass_bound {ι : Type*} [DecidableEq ι] (K D : Finset ι) (s v : ι → ℝ)
    (V m Δ : ℝ) (hKD : Disjoint K D) (hv : ∀ i ∈ K ∪ D, |v i| ≤ V)
    (j : ι) (hj : j ∈ K) (hm : m ≤ s j) (hD : ∀ i ∈ D, s i ≤ m - Δ) :
    |num (K ∪ D) s v / den (K ∪ D) s - num K s v / den K s|
      ≤ 2 * V * D.card * Real.exp (-Δ) := by
  have hpos : ∀ S : Finset ι, j ∈ S → 0 < den S s := fun S hS =>
    sum_pos (fun i _ => Real.exp_pos _) ⟨j, hS⟩
  have hK := hpos K hj
  have hKD' := hpos (K ∪ D) (mem_union_left _ hj)
  have hnu : num (K ∪ D) s v - num K s v = num D s v := by
    simp [num, sum_union hKD]
  have hdu : den (K ∪ D) s - den K s = den D s := by
    simp [den, sum_union hKD]
  have hV : 0 ≤ V := (abs_nonneg _).trans (hv j (mem_union_left _ hj))
  have hND : |num D s v| ≤ V * den D s :=
    num_le D s v V fun i hi => hv i (mem_union_right _ hi)
  have hDD : 0 ≤ den D s := sum_nonneg fun i _ => (Real.exp_pos _).le
  have hDc : den D s ≤ D.card * Real.exp (m - Δ) := by
    unfold den
    calc ∑ i ∈ D, Real.exp (s i) ≤ ∑ _i ∈ D, Real.exp (m - Δ) :=
          sum_le_sum fun i hi => Real.exp_le_exp.mpr (hD i hi)
      _ = D.card * Real.exp (m - Δ) := by simp
  have hKm : Real.exp m ≤ den K s := by
    unfold den
    exact (Real.exp_le_exp.mpr hm).trans
      (single_le_sum (f := fun i => Real.exp (s i)) (fun i _ => (Real.exp_pos _).le) hj)
  refine (ratio_error _ _ _ _ V hKD' hK (num_le _ s v V hv)).trans ?_
  rw [hnu, hdu, abs_of_nonneg hDD, div_le_iff₀ hK]
  have hem : Real.exp (m - Δ) = Real.exp (-Δ) * Real.exp m := by
    rw [← Real.exp_add]; ring_nf
  calc |num D s v| + V * den D s ≤ 2 * V * den D s := by linarith
    _ ≤ 2 * V * (D.card * Real.exp (m - Δ)) := by gcongr
    _ = 2 * V * D.card * Real.exp (-Δ) * Real.exp m := by rw [hem]; ring
    _ ≤ 2 * V * D.card * Real.exp (-Δ) * den K s := by gcongr

/- ── §3. a Farkas certificate proves the drop without reading the keys ── -/

/-- The row `q·k − t ≥ 0`: a key scores at least `t` against the query `q`.

    {d : ℕ} (q : Fin d → ℤ) (t : ℤ)          [a query of d whole numbers, a threshold]
    : Fragment.Row d                         [the row whose value at k is q·k − t]
    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `→` [takes the left to the right; between statements, the left gives the right] -/
def scoreRow {d : ℕ} (q : Fin d → ℤ) (t : ℤ) : Fragment.Row d := ⟨-t, q⟩

/-- A Farkas certificate over a node's region and the score row proves every key in the
    region scores below the threshold.

    {d : ℕ} (rs : List (Fragment.Row d))     [the rows bounding a node's keys]
    (q : Fin d → ℤ) (t : ℤ)                  [a query, a threshold]
    (ws : List (ℤ × Fragment.Row d))         [the certificate: weighted rows]
    (hw : ∀ p ∈ ws, 0 ≤ p.1)                 [every weight is at least zero]
    (hm : ∀ p ∈ ws, p.2 ∈ scoreRow q t :: rs)
                                             [every weighted row is the score row or a
                                              region row]
    (hz : ∀ i, (Fragment.combo ws).coeff i = 0)
                                             [the combination has no columns left]
    (hn : (Fragment.combo ws).const < 0)     [and its constant is below zero]
    (k : Fin d → ℤ)                          [a key]
    (hk : ∀ r ∈ rs, Fragment.Row.Holds k r)  [the key lies in the node's region]
    ∑ i, q i * k i < t                       [the key scores below the threshold]
    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `→` [takes the left to the right; between statements, the left gives the right]
    `List` [a finite sequence of things, in order]
    `×` [paired with, so ℤ × Row means a weight beside a row]
    `∀` [for every single one]
    `∈` [is one of the things in]
    `≤` [is smaller than or equal to]
    `<` [is smaller than]
    `∑` [added up over every coordinate]

    So: if some key scored at or above `t`, it would satisfy the score row and every region
    row together, and `Fragment.unsat_of_negative_combo` says nothing does. The check costs
    one combination and one sign; no key is read. -/
theorem farkas_prunes {d : ℕ} (rs : List (Fragment.Row d)) (q : Fin d → ℤ) (t : ℤ)
    (ws : List (ℤ × Fragment.Row d)) (hw : ∀ p ∈ ws, 0 ≤ p.1)
    (hm : ∀ p ∈ ws, p.2 ∈ scoreRow q t :: rs)
    (hz : ∀ i, (Fragment.combo ws).coeff i = 0) (hn : (Fragment.combo ws).const < 0)
    (k : Fin d → ℤ) (hk : ∀ r ∈ rs, Fragment.Row.Holds k r) :
    ∑ i, q i * k i < t := by
  by_contra h
  refine Fragment.unsat_of_negative_combo _ ws hw hm hz hn ⟨k, fun r hr => ?_⟩
  rcases List.mem_cons.mp hr with rfl | hr
  · simp only [Fragment.Row.Holds, Fragment.Row.value, scoreRow]
    linarith
  · exact hk r hr

/-- Non-vacuity: the region `k ≤ 3`, query `2`, threshold `7`. The certificate
    `2·(3 − k) + (2k − 7) = −1` proves every key in the region scores below `7`.
    `∀` [for every single one]
    `∈` [is one of the things in]
    `<` [is smaller than]
    `∑` [added up over every coordinate]
    `→` [takes the left to the right; between statements, the left gives the right] -/
example (k : Fin 1 → ℤ) (hk : ∀ r ∈ [(⟨3, fun _ => -1⟩ : Fragment.Row 1)],
    Fragment.Row.Holds k r) : ∑ i, (fun _ => (2 : ℤ)) i * k i < 7 :=
  farkas_prunes [⟨3, fun _ => -1⟩] (fun _ => 2) 7
    [(2, ⟨3, fun _ => -1⟩), (1, scoreRow (fun _ => 2) 7)]
    (by decide) (by simp [scoreRow])
    (fun i => by simp [Fragment.combo, Fragment.Row.add, Fragment.Row.scale,
      Fragment.Row.ofConst, scoreRow])
    (by simp [Fragment.combo, Fragment.Row.add, Fragment.Row.scale, Fragment.Row.ofConst,
      scoreRow]) k hk

/- ── §4. close keys are carried by moments ── -/

/-- A node whose scores lie within `B ≤ 1` of a centre is its first `p` moments, up to a
    remainder that shrinks as `B^p`.

    {ι : Type*} (S : Finset ι) (s v : ι → ℝ) (c B : ℝ) (p : ℕ)
                                             [keys, scores, values, a centre, a radius, an
                                              order]
    (hp : 0 < p) (hB : B ≤ 1)                [at least one moment; the radius is at most one]
    (hs : ∀ i ∈ S, |s i - c| ≤ B)            [every score lies within B of the centre]
    |num S s v - Real.exp c * ∑ m ∈ range p, (∑ i ∈ S, (s i - c) ^ m * v i) / m.factorial|
      ≤ Real.exp c * (∑ i ∈ S, |v i|) * (B ^ p * (p.succ / (p.factorial * p)))
                                             [the numerator and its moment expansion differ
                                              by at most e^c times the value mass times the
                                              Taylor remainder]
    `ℝ` [the real numbers: every number on the number line]
    `→` [takes the left to the right; between statements, the left gives the right]
    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `∀` [for every single one]
    `∈` [is one of the things in]
    `<` [is smaller than]
    `≤` [is smaller than or equal to]
    `Real.exp` [e raised to this power]
    `∑` [added up over]

    So: `e^{sᵢ} = e^c·e^{sᵢ−c}`, and `Real.exp_bound` bounds each `e^{sᵢ−c}` by its first `p`
    Taylor terms. Swapping the two sums makes the expansion a sum of moments
    `Σᵢ (sᵢ − c)^m·vᵢ`. With every `vᵢ = 1` this is the denominator. -/
theorem moment_remainder {ι : Type*} (S : Finset ι) (s v : ι → ℝ) (c B : ℝ) (p : ℕ)
    (hp : 0 < p) (hB : B ≤ 1) (hs : ∀ i ∈ S, |s i - c| ≤ B) :
    |num S s v - Real.exp c * ∑ m ∈ range p, (∑ i ∈ S, (s i - c) ^ m * v i) / m.factorial|
      ≤ Real.exp c * (∑ i ∈ S, |v i|) * (B ^ p * (p.succ / (p.factorial * p))) := by
  set C : ℝ := (p.succ : ℝ) / (p.factorial * p)
  have hC : 0 ≤ C := by positivity
  have hswap : ∑ m ∈ range p, (∑ i ∈ S, (s i - c) ^ m * v i) / m.factorial
      = ∑ i ∈ S, (∑ m ∈ range p, (s i - c) ^ m / m.factorial) * v i := by
    simp_rw [sum_div, sum_mul]
    rw [sum_comm]
    refine sum_congr rfl fun i _ => sum_congr rfl fun m _ => by ring
  have hterm : ∀ i ∈ S, |Real.exp (s i) * v i
      - Real.exp c * ((∑ m ∈ range p, (s i - c) ^ m / m.factorial) * v i)|
      ≤ Real.exp c * (|v i| * (B ^ p * C)) := by
    intro i hi
    have hx : |s i - c| ≤ 1 := (hs i hi).trans hB
    have hb := Real.exp_bound hx hp
    have hsplit : Real.exp (s i) = Real.exp c * Real.exp (s i - c) := by
      rw [← Real.exp_add]; ring_nf
    have hpow : |s i - c| ^ p ≤ B ^ p := pow_le_pow_left₀ (abs_nonneg _) (hs i hi) p
    rw [hsplit, show Real.exp c * Real.exp (s i - c) * v i
        - Real.exp c * ((∑ m ∈ range p, (s i - c) ^ m / m.factorial) * v i)
        = Real.exp c * ((Real.exp (s i - c)
          - ∑ m ∈ range p, (s i - c) ^ m / m.factorial) * v i) by ring,
      abs_mul, abs_of_pos (Real.exp_pos _), abs_mul]
    refine mul_le_mul_of_nonneg_left ?_ (Real.exp_pos _).le
    rw [mul_comm]
    exact mul_le_mul_of_nonneg_left (hb.trans (mul_le_mul_of_nonneg_right hpow hC))
      (abs_nonneg _)
  unfold num
  rw [hswap, mul_sum, ← sum_sub_distrib, mul_assoc, sum_mul, mul_sum]
  exact (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun i hi => by
    have := hterm i hi
    calc _ ≤ _ := this
      _ = Real.exp c * (|v i| * (B ^ p * C)) := rfl)

/-- The first moment is stored once: it reads any query through one dot product.

    {ι : Type*} {d : ℕ} (S : Finset ι) (u : ι → Fin d → ℝ) (v : ι → ℝ) (q : Fin d → ℝ)
                                             [keys, their offsets from the node centre, values,
                                              a query]
    ∑ i ∈ S, (q ⬝ᵥ u i) * v i = q ⬝ᵥ ∑ i ∈ S, v i • u i
                                             [the query's first moment is the query against
                                              one stored vector]
    `ℝ` [the real numbers: every number on the number line]
    `→` [takes the left to the right; between statements, the left gives the right]
    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `∈` [is one of the things in]
    `∑` [added up over]
    `⬝ᵥ` [the dot product: multiply coordinate by coordinate and add]
    `•` [stretch by, a number times a point] -/
theorem moment_one_stored {ι : Type*} {d : ℕ} (S : Finset ι) (u : ι → Fin d → ℝ)
    (v : ι → ℝ) (q : Fin d → ℝ) :
    ∑ i ∈ S, (q ⬝ᵥ u i) * v i = q ⬝ᵥ ∑ i ∈ S, v i • u i := by
  rw [dotProduct_sum]
  refine sum_congr rfl fun i _ => ?_
  rw [dotProduct_smul, smul_eq_mul, mul_comm]

/-- The second moment is stored once: it reads any query through one `d × d` table.

    {ι : Type*} {d : ℕ} (S : Finset ι) (u : ι → Fin d → ℝ) (v : ι → ℝ) (q : Fin d → ℝ)
                                             [keys, their offsets from the node centre, values,
                                              a query]
    ∑ i ∈ S, (q ⬝ᵥ u i) ^ 2 * v i
      = ∑ a, ∑ b, q a * q b * ∑ i ∈ S, v i * u i a * u i b
                                             [the query's second moment is the query twice
                                              against one stored table]
    `ℝ` [the real numbers: every number on the number line]
    `→` [takes the left to the right; between statements, the left gives the right]
    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `∈` [is one of the things in]
    `∑` [added up over]
    `⬝ᵥ` [the dot product: multiply coordinate by coordinate and add] -/
theorem moment_two_stored {ι : Type*} {d : ℕ} (S : Finset ι) (u : ι → Fin d → ℝ)
    (v : ι → ℝ) (q : Fin d → ℝ) :
    ∑ i ∈ S, (q ⬝ᵥ u i) ^ 2 * v i
      = ∑ a, ∑ b, q a * q b * ∑ i ∈ S, v i * u i a * u i b := by
  simp only [dotProduct, sq, sum_mul, mul_sum]
  conv_lhs => rw [Finset.sum_comm]
  refine sum_congr rfl fun a _ => ?_
  conv_lhs => rw [Finset.sum_comm]
  exact sum_congr rfl fun b _ => sum_congr rfl fun i _ => by ring

/- ── §5. the errors of a partition add ── -/

/-- A reader that splits the keys into parts is off by at most the sum of the parts' bounds.

    {ι κ : Type*} [DecidableEq ι] (P : Finset κ) (part : κ → Finset ι)
                                             [the parts, and the keys in each]
    (hdisj : (P : Set κ).PairwiseDisjoint part) [no key is in two parts]
    (f : ι → ℝ) (approx ε : κ → ℝ)           [a term per key, a stand-in and a bound per part]
    (h : ∀ b ∈ P, |∑ i ∈ part b, f i - approx b| ≤ ε b)
                                             [each part's stand-in is within its bound]
    |∑ i ∈ P.biUnion part, f i - ∑ b ∈ P, approx b| ≤ ∑ b ∈ P, ε b
                                             [the whole is within the sum of the bounds]
    `ℝ` [the real numbers: every number on the number line]
    `→` [takes the left to the right; between statements, the left gives the right]
    `∀` [for every single one]
    `∈` [is one of the things in]
    `≤` [is smaller than or equal to]
    `∑` [added up over]

    So: the sum over a disjoint union is the sum of the parts' sums, and the triangle
    inequality does the rest. Exact parts carry bound `0`, dropped parts the §2 mass, and
    summarised parts the §4 remainder. -/
theorem partition_error {ι κ : Type*} [DecidableEq ι] (P : Finset κ) (part : κ → Finset ι)
    (hdisj : (P : Set κ).PairwiseDisjoint part) (f : ι → ℝ) (approx ε : κ → ℝ)
    (h : ∀ b ∈ P, |∑ i ∈ part b, f i - approx b| ≤ ε b) :
    |∑ i ∈ P.biUnion part, f i - ∑ b ∈ P, approx b| ≤ ∑ b ∈ P, ε b := by
  rw [sum_biUnion hdisj, ← sum_sub_distrib]
  exact (abs_sum_le_sum_abs _ _).trans (sum_le_sum h)

/- ── §6. the floor: exactness for every query stores the keys ── -/

/-- A key read as a character: `q ↦ e^{q·k}` turns adding queries into multiplying.

    {d : ℕ} (k : Fin d → ℝ)                  [a key]
    : Multiplicative (Fin d → ℝ) →* ℝ        [a map from queries that sends sums to products]
    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `ℝ` [the real numbers: every number on the number line]
    `→` [takes the left to the right; between statements, the left gives the right]
    `→*` [a map that sends products to products and one to one] -/
noncomputable def keyChar {d : ℕ} (k : Fin d → ℝ) : Multiplicative (Fin d → ℝ) →* ℝ where
  toFun q := Real.exp (Multiplicative.toAdd q ⬝ᵥ k)
  map_one' := by simp
  map_mul' a b := by simp [add_dotProduct, Real.exp_add]

/-- Distinct keys are distinct characters.

    {d : ℕ}                                  [the key width]
    Function.Injective (keyChar (d := d))    [two keys with the same character are equal]
    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `Function.Injective` [different inputs give different outputs]

    So: at the query `k − k'`, equal characters give `(k − k')·(k − k') = 0`, and a dot
    product of a real vector with itself is zero only at zero. -/
theorem keyChar_injective {d : ℕ} : Function.Injective (keyChar (d := d)) := by
  intro k k' h
  have he := DFunLike.congr_fun h (Multiplicative.ofAdd (k - k'))
  simp only [keyChar, MonoidHom.coe_mk, OneHom.coe_mk, toAdd_ofAdd, Real.exp_eq_exp] at he
  have h0 : (k - k') ⬝ᵥ (k - k') = 0 := by
    rw [dotProduct_sub, he, sub_self]
  exact sub_eq_zero.mp (dotProduct_self_eq_zero.mp h0)

/-- The floor: a weighted exponential sum known for every query determines every weight.

    {d : ℕ} (S : Finset (Fin d → ℝ))          [a finite set of keys]
    (c c' : (Fin d → ℝ) → ℝ)                  [two weightings of the keys]
    (h : ∀ q : Fin d → ℝ,
      ∑ k ∈ S, c k * Real.exp (q ⬝ᵥ k) = ∑ k ∈ S, c' k * Real.exp (q ⬝ᵥ k))
                                             [the two sums agree at every query]
    ∀ k ∈ S, c k = c' k                       [then the two weightings agree at every key]
    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `ℝ` [the real numbers: every number on the number line]
    `→` [takes the left to the right; between statements, the left gives the right]
    `∀` [for every single one]
    `∈` [is one of the things in]
    `∑` [added up over]
    `Real.exp` [e raised to this power]
    `⬝ᵥ` [the dot product: multiply coordinate by coordinate and add]

    So: the characters `keyChar k` are linearly independent (`linearIndependent_monoidHom`,
    composed with `keyChar_injective`), and the difference of the two weightings combines
    them to zero. With `c` the key counts this is the denominator alone: a state exact for
    every query determines the key multiset, so it is at least as large as the keys. -/
theorem keys_determined {d : ℕ} (S : Finset (Fin d → ℝ)) (c c' : (Fin d → ℝ) → ℝ)
    (h : ∀ q : Fin d → ℝ,
      ∑ k ∈ S, c k * Real.exp (q ⬝ᵥ k) = ∑ k ∈ S, c' k * Real.exp (q ⬝ᵥ k)) :
    ∀ k ∈ S, c k = c' k := by
  have hli : LinearIndependent ℝ (fun k : Fin d → ℝ => (keyChar k : Multiplicative (Fin d → ℝ)
      → ℝ)) :=
    (linearIndependent_monoidHom (Multiplicative (Fin d → ℝ)) ℝ).comp _ keyChar_injective
  have hz := linearIndependent_iff'.mp hli S (fun k => c k - c' k) (by
    funext q
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply, keyChar,
      MonoidHom.coe_mk, OneHom.coe_mk, sub_mul, sum_sub_distrib]
    rw [h (Multiplicative.toAdd q), sub_self])
  intro k hk
  exact sub_eq_zero.mp (hz k hk)

end AttentionCut
end Numeric
end Apeiron
