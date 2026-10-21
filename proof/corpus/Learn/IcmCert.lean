import Mathlib
import ApeironWord
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Learn.IcmError

/-!
# The low-limb argmin certificate

`src/learn/fold/fit/icm.rs` computes the E8 code dots and their flush at `L < 5` limbs and keeps the
`L`-limb argmin only while the winner clears the runner-up by more than `2·δ`; otherwise it recomputes
the row at 5 limbs. Both executions read the same carried state (`f_res`, `s_cur`), so `δ` bounds the
gap between the `L`-limb words and the 5-limb words, and not the gap from either to the exact reals.
The residual-state error `εres` of `Learn.IcmError` does not enter this certificate, and the static
margin `311287510032769024/2^64 ≈ 1.69e-2` of `word_argmin_cost_bound_end2end_derived_const`
(`Learn/IcmError.lean:315`) is not the relevant figure.

## What is certified

Given, per code `c`, a bound `|costL c − cost c| ≤ δ` between the `L`-limb block cost and the 5-limb
block cost, and a strict clearance `costL cstar + 2·δ < costL c` at every other code, the 5-limb
argmin at `cstar` is the `L`-limb argmin too (`argmin_certified`) and it is the only one
(`argmin_certified_unique`). `argmin_certified_words` is the same rule in the integer words the Rust
check compares, in the form that check evaluates (`2·δ < costL c − costL cstar`). The factor `2·δ` is
tight: `argmin_gap_tight` exhibits admissible costs at equality where the conclusion fails, and at
`δ` the clearance holds while the conclusion fails.

`δ` is built from two pieces: the two `nt_q` call bounds at `L` and 5 limbs and the per-entry flush
difference `εv` (`dot_gap_two_limb`), carried through the block cost `q c − 2·d c`
(`blockCost_gap`, `argmin_certified_limbs`). `dots_end2end_state_G` is `Learn.Limb.dots_end2end_state`
with the call bound as a parameter: the shipped lemma fixes it at `24842/2^32`, the 5-limb value, and
a call at `L` limbs has a larger one.

## Port

`gap_lt_of_close` is `token_survives_margin` of
`../sanguine/proof/Apeiron/Numeric/Accumulator.lean:122`
(`(ra rb sa sb ε : ℤ) … (hgap : 2 * ε < ra - rb) : sb < sa`, proof `rw [abs_le] at ha hb; omega`)
restated over `ℝ`: the same inequality by the same route, in the structure the block costs live in.
The sanguine corpus is not imported; nothing else of it is used.

## Boundary

Not claimed: the Rust check's per-row evaluation of `δ`, which is a computation on the block's data
and is not refined here; that the two executions read the same carried state, which is a property of
the `sweep_rows` body; the `add_q`/`sub_q` overflow branches, which abort the pass; and the
`i64`/`i128` grid. `argmin_certified_words` carries the integer words; the rest carry reals.
The Rust side is board row `APEIRON-QUALITY-118-W1-T01`, lane `CERT-W1-T01`.
-/

namespace Learn.IcmCert

/-! ## The one-pair gap -/

/-- **A clearance of `2·δ` survives a perturbation of `δ` on each side.** If `a'` is within `δ` of
`a`, `b'` within `δ` of `b`, and `a` beats `b` by more than `2·δ`, then `b' < a'`. This is the
`ℤ` inequality of `token_survives_margin` (`../sanguine/proof/Apeiron/Numeric/Accumulator.lean:122`,
proof `rw [abs_le] at ha hb; omega`) over `ℝ`; `abs_le` then `linarith` is the same route.
`≤` [is smaller than or equal to]; `<` [is smaller than];
`ℝ` [the real numbers: every number on the number line]. -/
theorem gap_lt_of_close {a a' b b' δ : ℝ}
    (ha : |a' - a| ≤ δ) (hb : |b' - b| ≤ δ) (hgap : 2 * δ < a - b) : b' < a' := by
  have h1 := abs_le.mp ha
  have h2 := abs_le.mp hb
  linarith [h1.1, h1.2, h2.1, h2.2]

/-! ## The argmin rule -/

/-- **The certified `L`-limb argmin is the 5-limb argmin.** With every `L`-limb cost within `δ` of its
5-limb cost, and `cstar` clearing every other code by more than `2·δ` in the `L`-limb costs, `cstar`
has strictly the least 5-limb cost. `δ` is the gap between the two word executions, not a distance to
the reals; no hypothesis on the carried state appears.
`→` [takes the left and gives the right]; `∀` [for every single one];
`∈` [is one of the things in]; `≠` [is not equal to]; `≤` [is smaller than or equal to];
`<` [is smaller than]; `ℝ` [the real numbers: every number on the number line]. -/
theorem argmin_certified {ι : Type} (s : Finset ι) (cost costL : ι → ℝ) (δ : ℝ) (cstar : ι)
    (hcstar : cstar ∈ s)
    (hclose : ∀ c ∈ s, |costL c - cost c| ≤ δ)
    (hgap : ∀ c ∈ s, c ≠ cstar → costL cstar + 2 * δ < costL c) :
    ∀ c ∈ s, c ≠ cstar → cost cstar < cost c := by
  intro c hc hne
  have h1 : |cost c - costL c| ≤ δ := by
    rw [abs_sub_comm]
    exact hclose c hc
  have h2 : |cost cstar - costL cstar| ≤ δ := by
    rw [abs_sub_comm]
    exact hclose cstar hcstar
  have h3 : 2 * δ < costL c - costL cstar := by
    have hc' : costL cstar + 2 * δ < costL c := hgap c hc hne
    linarith
  exact gap_lt_of_close h1 h2 h3

/-- **`cstar` is the unique 5-limb minimiser**, so a tie rule (the first index of `best_code_q`)
selects it whatever it does on the codes it never compares. The first conjunct is the argmin
property, the second the uniqueness: any code that minimises the 5-limb cost is `cstar`.
`→` [takes the left and gives the right]; `∀` [for every single one];
`∈` [is one of the things in]; `≠` [is not equal to]; `≤` [is smaller than or equal to];
`<` [is smaller than]; `∧` [both are true];
`ℝ` [the real numbers: every number on the number line]. -/
theorem argmin_certified_unique {ι : Type} (s : Finset ι) (cost costL : ι → ℝ) (δ : ℝ)
    (cstar : ι) (hcstar : cstar ∈ s)
    (hclose : ∀ c ∈ s, |costL c - cost c| ≤ δ)
    (hgap : ∀ c ∈ s, c ≠ cstar → costL cstar + 2 * δ < costL c) :
    (∀ c ∈ s, cost cstar ≤ cost c) ∧
      (∀ d ∈ s, (∀ c ∈ s, cost d ≤ cost c) → d = cstar) := by
  have hmin := argmin_certified s cost costL δ cstar hcstar hclose hgap
  refine ⟨?_, ?_⟩
  · intro c hc
    by_cases h : c = cstar
    · subst h
      exact le_refl _
    · exact le_of_lt (hmin c hc h)
  · intro d hd hdle
    by_contra hne
    have hlt := hmin d hd hne
    have hle := hdle cstar hcstar
    linarith

/-- **The rule in the integer words the Rust check compares.** `2 * δ < costL c - costL cstar` is
the test `icm.rs` evaluates, with `cstar` the `L`-limb winner: at every other code the `L`-limb cost
exceeds the winner's by more than `2·δ`, and every word cost is within `δ` of its 5-limb word. The
conclusion is the strict 5-limb order at `cstar`.
`→` [takes the left and gives the right]; `∀` [for every single one];
`∈` [is one of the things in]; `≠` [is not equal to]; `≤` [is smaller than or equal to];
`<` [is smaller than]. -/
theorem argmin_certified_words {ι : Type} (s : Finset ι) (cost costL : ι → ℤ) (δ : ℤ) (cstar : ι)
    (hcstar : cstar ∈ s)
    (hclose : ∀ c ∈ s, |costL c - cost c| ≤ δ)
    (hcheck : ∀ c ∈ s, c ≠ cstar → 2 * δ < costL c - costL cstar) :
    ∀ c ∈ s, c ≠ cstar → cost cstar < cost c := by
  intro c hc hne
  have h1 := abs_le.mp (hclose c hc)
  have h2 := abs_le.mp (hclose cstar hcstar)
  have h3 := hcheck c hc hne
  omega

/-! ## The word gap between `L` and 5 limbs -/

/-- **A perturbation of one operand moves the eight-term inner product by at most `8·εa·Mb`**, the
`Learn.Limb.dot8_error_le` at the codebook column bound `12100000000/2^32`.
`→` [takes the left and gives the right]; `∀` [for every single one];
`≤` [is smaller than or equal to]; `ℝ` [the real numbers: every number on the number line]. -/
theorem dot8_shift_le (f fL C : Fin 8 → ℝ) (εv : ℝ)
    (hflush : ∀ k, |fL k - f k| ≤ εv)
    (hC : ∀ k, |C k| ≤ 12100000000 / 2 ^ 32) (hεv : 0 ≤ εv) :
    |Limb.dot8 fL C - Limb.dot8 f C| ≤ 8 * εv * (12100000000 / 2 ^ 32) :=
  Limb.dot8_error_le hflush hC hεv

/-- **`dots` end to end with the call bound as a parameter.** `Learn.Limb.dots_end2end_state` fixes
the `nt_q` call bound at `24842/2^32`, the value at the 5-limb production call; a call at `L < 5`
limbs has a larger one, so `G` is a parameter here and the conclusion carries `G/2^32`. The proof is
that lemma's composition unchanged — `Learn.Limb.fblk_error_le`, then `Learn.Limb.dot8_comp_le` — and
the one thing that changed is `24842` replaced by `G`. No nonnegativity of `G` is needed.
`→` [takes the left and gives the right]; `∀` [for every single one];
`≤` [is smaller than or equal to]; `ℝ` [the real numbers: every number on the number line]. -/
theorem dots_end2end_state_G (v vh fr frh C : Fin 8 → ℝ) (a dh εv εres G : ℝ)
    (hv : ∀ k, |vh k - v k| ≤ εv) (hr : ∀ k, |frh k - fr k| ≤ εres)
    (ha : |a| ≤ 1) (hεv : 0 ≤ εv) (hεres : 0 ≤ εres)
    (hC : ∀ k, |C k| ≤ 12100000000 / 2 ^ 32)
    (hcall : |dh - Limb.dot8 (fun k => frh k + a * vh k) C| ≤ G / 2 ^ 32) :
    |dh - Limb.dot8 (fun k => fr k + a * v k) C|
      ≤ G / 2 ^ 32 + 8 * (εres + εv) * (12100000000 / 2 ^ 32) :=
  Limb.dot8_comp_le hcall (Limb.fblk_error_le v vh fr frh a εv εres hv hr ha) hC
    (by linarith)

/-- **The dot gap between the `L`-limb and 5-limb executions.** `f` is the 5-limb flush output and
`fL` the `L`-limb one, differing per entry by at most `εv`; the two dots calls are within `GLimb/2^32`
and `G5/2^32` of `dot8` of their own inputs. Then the two computed dots differ by at most
`GLimb/2^32 + G5/2^32 + 8·εv·(12100000000/2^32)`.

The carried state does not appear: both runs read the same `f_res` and `s_cur`, so the `εres` term of
`Learn.Limb.dots_end2end_state` is absent, and what remains is the flush difference carried through
the codebook column (`dot8_shift_le`) plus the two call bounds. The `2^32` denominators are the
`Q32.32` word scale.
`→` [takes the left and gives the right]; `∀` [for every single one];
`≤` [is smaller than or equal to]; `ℝ` [the real numbers: every number on the number line]. -/
theorem dot_gap_two_limb (f fL C : Fin 8 → ℝ) (dhL dh5 GLimb G5 εv : ℝ)
    (hflush : ∀ k, |fL k - f k| ≤ εv)
    (hC : ∀ k, |C k| ≤ 12100000000 / 2 ^ 32) (hεv : 0 ≤ εv)
    (hcallL : |dhL - Limb.dot8 fL C| ≤ GLimb / 2 ^ 32)
    (hcall5 : |dh5 - Limb.dot8 f C| ≤ G5 / 2 ^ 32) :
    |dhL - dh5| ≤ GLimb / 2 ^ 32 + G5 / 2 ^ 32 + 8 * εv * (12100000000 / 2 ^ 32) := by
  have hcross : |Limb.dot8 fL C - Limb.dot8 f C| ≤ 8 * εv * (12100000000 / 2 ^ 32) :=
    dot8_shift_le f fL C εv hflush hC hεv
  have h5 : |Limb.dot8 f C - dh5| ≤ G5 / 2 ^ 32 := by
    rw [abs_sub_comm]
    exact hcall5
  have hsplit : dhL - dh5
      = (dhL - Limb.dot8 fL C) + (Limb.dot8 fL C - Limb.dot8 f C)
        + (Limb.dot8 f C - dh5) := by ring
  rw [hsplit]
  have htri : |(dhL - Limb.dot8 fL C) + (Limb.dot8 fL C - Limb.dot8 f C)
      + (Limb.dot8 f C - dh5)|
      ≤ |dhL - Limb.dot8 fL C| + |Limb.dot8 fL C - Limb.dot8 f C|
        + |Limb.dot8 f C - dh5| := by
    refine le_trans (abs_add_le _ _) ?_
    have h2 := abs_add_le (dhL - Limb.dot8 fL C) (Limb.dot8 fL C - Limb.dot8 f C)
    linarith
  linarith [htri, hcallL, hcross, h5]

/-! ## The block cost -/

/-- The block cost `q c − 2·d c` of code `c`: `q` is the quadratic form, equal at both limb counts
(`a²·q` is exact), and `d` is the code dot, which the limb count moves. This is
`fit::icm::block_cost` (`aa·q − two_a·(dot << 32)`) with the row scale `a²`/`2a` absorbed into `q`
and `d`. `O(1)`, two multiplications and one subtraction.
`→` [takes the left and gives the right]; `ℝ` [the real numbers: every number on the number line]. -/
def blockCost {ι : Type} (q d : ι → ℝ) (c : ι) : ℝ := q c - 2 * d c

/-- **The block cost gap is twice the dot gap.** `q` is the same at both limb counts, so the cost
difference is `−2·(dL c − d c)` and its magnitude at most `2·εd` when the dots are within `εd`.
`→` [takes the left and gives the right]; `≤` [is smaller than or equal to];
`ℝ` [the real numbers: every number on the number line]. -/
theorem blockCost_gap {ι : Type} (q d dL : ι → ℝ) (c : ι) (εd : ℝ)
    (hd : |dL c - d c| ≤ εd) :
    |blockCost q dL c - blockCost q d c| ≤ 2 * εd := by
  have h : blockCost q dL c - blockCost q d c = -2 * (dL c - d c) := by
    unfold blockCost
    ring
  rw [h, abs_mul]
  have h2 : |(-2 : ℝ)| = 2 := by norm_num
  rw [h2]
  exact mul_le_mul_of_nonneg_left hd (by norm_num)

/-- **The certificate at the block cost.** `εd` is the dot gap of the two limb counts, and the
clearance is `2·(2·εd)`: the cost gap is `2·εd` (`blockCost_gap`) and the rule needs twice it
(`argmin_certified`). At the `icm.rs` call site `εd` is
`GLimb/2^32 + G5/2^32 + 8·εv·(12100000000/2^32)` (`dot_gap_two_limb`).
`→` [takes the left and gives the right]; `∀` [for every single one];
`∈` [is one of the things in]; `≠` [is not equal to]; `≤` [is smaller than or equal to];
`<` [is smaller than]; `ℝ` [the real numbers: every number on the number line]. -/
theorem argmin_certified_limbs {ι : Type} (s : Finset ι) (q d dL : ι → ℝ) (εd : ℝ) (cstar : ι)
    (hcstar : cstar ∈ s)
    (hd : ∀ c ∈ s, |dL c - d c| ≤ εd)
    (hgap : ∀ c ∈ s, c ≠ cstar →
      blockCost q dL cstar + 2 * (2 * εd) < blockCost q dL c) :
    ∀ c ∈ s, c ≠ cstar → blockCost q d cstar < blockCost q d c := by
  refine argmin_certified s (blockCost q d) (blockCost q dL) (2 * εd) cstar hcstar ?_ hgap
  intro c hc
  exact blockCost_gap q d dL c εd (hd c hc)

/-! ## Non-vacuity and tightness (§4 obligation 2) -/

/-- **The joint hypotheses are satisfiable.** `s = univ` on `Fin 3`, `cstar = 0`, `q ≡ 0`,
`d = ![0, -1, -2]`, the `L`-limb dots `dL = ![1/4, -1, -2]` and `εd = 1/4`: closeness holds at all
three codes, the clearance `blockCost q dL 0 + 2·(2·εd) < blockCost q dL c` holds at `c = 1, 2`, and
the conclusion is obtained by applying `argmin_certified_limbs`, not by evaluating it. All three
conjuncts are closed by `norm_num` from the definitions.
`→` [takes the left and gives the right]; `∀` [for every single one];
`∈` [is one of the things in]; `≠` [is not equal to]; `≤` [is smaller than or equal to];
`<` [is smaller than]; `∧` [both are true];
`ℝ` [the real numbers: every number on the number line]. -/
theorem argmin_joint_witness :
    (∀ c ∈ (Finset.univ : Finset (Fin 3)),
        |(![1 / 4, -1, -2] : Fin 3 → ℝ) c - (![0, -1, -2] : Fin 3 → ℝ) c| ≤ 1 / 4)
      ∧ (∀ c ∈ (Finset.univ : Finset (Fin 3)), c ≠ 0 →
          blockCost (fun _ => (0 : ℝ)) (![1 / 4, -1, -2] : Fin 3 → ℝ) 0 + 2 * (2 * (1 / 4))
            < blockCost (fun _ => (0 : ℝ)) (![1 / 4, -1, -2] : Fin 3 → ℝ) c)
      ∧ (∀ c ∈ (Finset.univ : Finset (Fin 3)), c ≠ 0 →
          blockCost (fun _ => (0 : ℝ)) (![0, -1, -2] : Fin 3 → ℝ) 0
            < blockCost (fun _ => (0 : ℝ)) (![0, -1, -2] : Fin 3 → ℝ) c) := by
  have hd : ∀ c ∈ (Finset.univ : Finset (Fin 3)),
      |(![1 / 4, -1, -2] : Fin 3 → ℝ) c - (![0, -1, -2] : Fin 3 → ℝ) c| ≤ 1 / 4 := by
    intro c _
    fin_cases c <;> norm_num
  have hgap : ∀ c ∈ (Finset.univ : Finset (Fin 3)), c ≠ 0 →
      blockCost (fun _ => (0 : ℝ)) (![1 / 4, -1, -2] : Fin 3 → ℝ) 0 + 2 * (2 * (1 / 4))
        < blockCost (fun _ => (0 : ℝ)) (![1 / 4, -1, -2] : Fin 3 → ℝ) c := by
    intro c _ hne
    fin_cases c
    · exact absurd rfl hne
    · norm_num [blockCost]
    · norm_num [blockCost]
  refine ⟨hd, hgap, ?_⟩
  exact argmin_certified_limbs (Finset.univ : Finset (Fin 3)) (fun _ => (0 : ℝ))
    (![0, -1, -2] : Fin 3 → ℝ) (![1 / 4, -1, -2] : Fin 3 → ℝ) (1 / 4) 0 (Finset.mem_univ 0)
    hd hgap

/-- **The strict `2·δ` is tight.** On `Fin 2` with `δ = 1`, `costL = ![0, 2]` and `cost = ![1, 1]`,
the closeness hypothesis holds at both codes (`|costL c − cost c| = 1 ≤ δ`), the clearance is exactly
`2·δ` and not more (`costL 0 + 2·1 = costL 1`, so the strict hypothesis fails at `c = 1`), and the
conclusion fails (`cost 0 = cost 1`). At `δ` the clearance `costL 0 + δ < costL 1` does hold while the
conclusion fails, which is the weakening this rule must not admit.
`→` [takes the left and gives the right]; `∀` [for every single one];
`∈` [is one of the things in]; `≤` [is smaller than or equal to]; `<` [is smaller than];
`∧` [both are true]; `¬` [this is not true];
`ℝ` [the real numbers: every number on the number line]. -/
theorem argmin_gap_tight :
    (∀ c ∈ (Finset.univ : Finset (Fin 2)),
        |(![0, 2] : Fin 2 → ℝ) c - (![1, 1] : Fin 2 → ℝ) c| ≤ 1)
      ∧ ¬ ((![0, 2] : Fin 2 → ℝ) 0 + 2 * (1 : ℝ) < (![0, 2] : Fin 2 → ℝ) 1)
      ∧ ¬ ((![1, 1] : Fin 2 → ℝ) 0 < (![1, 1] : Fin 2 → ℝ) 1)
      ∧ ((![0, 2] : Fin 2 → ℝ) 0 + (1 : ℝ) < (![0, 2] : Fin 2 → ℝ) 1) := by
  refine ⟨?_, by norm_num, by norm_num, by norm_num⟩
  intro c _
  fin_cases c <;> norm_num

end Learn.IcmCert
