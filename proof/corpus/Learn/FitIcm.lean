import Learn.E8Code
import Perf.CostCorrespondence
import Substrate.Algebra.Neural.CountLMFloor

/-!
# The `E8 ∪ O8` ICM step: specification, and the conditional link to the extracted body

The executed body is `fit::icm::e8_icm_step` (`src/learn/fold/fit/icm.rs:948`), rooted as
`apeiron_word.fit_port.word_fit_icm_e8_icm_step` (`proof/extract/ports/fit_port.rs:94`). Per block it
computes `q_all[m][j] = C_jᵀ H_blk C_j` and `dot[j] = f·C_j`, then takes the `CODE_LEN`-way argmin of
`block_cost aa two_a q dot = a²·q − 2·(2a)·(dot << 32)` (`fit::icm::best_code_q`, `:243`), where `f` is
the metric-applied residual (`f_res` initialised `G_rot − B·H_rot` and updated by `−a·(s·H_rot)`, so
`f = H·x` for the block residual `x`). `q_all` is the diagonal of `block_x`, i.e. `Cᵀ H C`.

This module states the specification that decision must meet and proves it from the canonical
identities: the block cost is the quadratic `Perf.FaceTypedE8.joint_block_cost_expand` at `a₂ = 0`; the
selected code is the argmin of that cost over the 256-row table; the codebook rows are the half-cube
spinors `= ONE • Perf.FaceTypedE8.Spin8`, the `D₈` roots `±2` and the `O₈` vertices `±√8`; and the
sweep's `J` is non-increasing per decision (`Perf.CostCorrespondence.e8_icm_global_descent_dominance`).

## The open link to the extracted body

`#audit apeiron_word.fit_port.word_fit_icm_e8_icm_step` does not close. Measured with the line enabled
(`lake build AxiomAudit`, exit 1) it names 31 constants outside `{propext, Classical.choice, Quot.sound}`,
led by:

* the pool: `apeiron_word.chart.pool.width_global` (`proof/ApeironWord.lean:8439`, `axiom …`),
  `.run_global`, `.chunks_mut_global`, `.Flag`, `.Flag.get/.new/.set`, `.chunks.for_chunks`,
  `.chunks.for_chunks4`, `.chunks.set_written_lens`
* the AVX2 surface: `apeiron_word.chart.q8_0_row_kcols_avx2`, `chart.x86.avx2`, `.dot16_i8`,
  `.dot32_i8`, `.q4_k_window_dot`, `.tiled2`, `kern.matmat.q5_0_block_kdots_avx2`,
  `kern.gemm.x86.rows`, `std_detect.detect.arch.x86.__is_feature_detected.avxvnni`
* the four `Apeiron_wordChartFastBlockDecoder.decode_step_avx2` instances (`Q4_K`, `Q5_0`, `Q6_K`,
  `Q8_0`)
* `apeiron_word.alloc.vec.Vec.spare_capacity_mut`, `core.slice.raw.from_raw_parts`,
  `core.ptr.mut_ptr.RawPtrMutT.add`, `core.ptr.mut_ptr.RawPtrMutT.write`,
  `core.option.Option.map`, `core.option.OptionShared0T.copied`, `par.online_threads`

The body's call graph is the quantiser's whole dispatch path, so the audit is not a handful of holes.
Of the two facts that link the generated body to this module's `blockReduce`, the first is proved about
the extracted definitions and the second is a hypothesis:

* `best_code_q` returns an argmin index of its Q96.96 word cost, with all 256 entry costs `Ok`:
  `best_code_q_argmin` (`proved`; the extracted loop, `hq`/`hd` are the caller's slice lengths).
* That word cost is `2^96 · blockReduce`: `WordCostCorresponds` states it row by row,
  `best_code_q_argmin_real` carries it through the argmin, and `icm_step_spec_of_extracted_argmin`
  concludes `IcmStepSpec` from `hbest` alone. `2^96 > 0` carries the order (`block_cost_ok` and
  `block_cost_total` are the per-entry word facts). Both theorems are `proved` **conditional on
  `WordCostCorresponds`**; its word form `aa·q − two_a·(dot << 32)` is `fit::icm::block_cost`, and
  `blockReduce_eq_word` gives the real identity, not the identity on the generated `i128` words.

Open, and not claimed: (a) the caller's Q32.32 slice correspondence — that the `aa`, `two_a`, `q`,
`dot` slices `e8_icm_step` builds are `(a·2^32)²`, `2a·2^32`, `Q·2^32`, `D·2^32` for the block's `H`,
`x`, `a`, i.e. an instance of `WordCostCorresponds`; (b) the pool/AVX2 dispatch path around the call,
whose 31 externals keep `#audit …word_fit_icm_e8_icm_step` open; (c) that body audit itself. No axiom is
added: the interface stays a theorem with an explicit hypothesis, so a lane that roots a pool-free
`best_code_q` and proves the slice correspondence discharges it.

Grades: every theorem below is `proved` (Lean kernel, closed axiom cone) about the specification; the
link to the generated body is `tested`-grade at best (the body is extracted and compiled, not refined),
and its audit is open as stated.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

noncomputable section

namespace Learn.FitIcm

open Perf.FaceTypedE8

/-- The 256-row codebook index set, `fit::icm::CODE_LEN`. -/
abbrev CodeIndex : Type := Fin 256

/-- The extracted codebook in Q32.32 units: row `c`, coordinate `k` is `codewordSpec c k / 2^32`.
This is the table `fit::icm::codebook` returns (`word_fit_icm_codebook`). -/
def codebook (c : CodeIndex) (k : Fin 8) : ℝ :=
  (Learn.E8Code.codewordSpec c.val k.val : ℝ) / 2 ^ 32

/-- The joint block cost of code `c` against the block residual `x` in the metric block `H` at group
scale `a`: `Σ_p Σ_q H_pq (a·C_c(p) − x_p)(a·C_c(q) − x_q)`, the quantity the sweep minimises. -/
def blockQuad (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ) (c : CodeIndex) : ℝ :=
  ∑ p, ∑ q, H p q * (a * codebook c p - x p) * (a * codebook c q - x q)

/-- The code-dependent part of `blockQuad`: `a²·q_c − 2a·dot_c` with `q_c = C_cᵀ H C_c` and
`dot_c = (H x)ᵀ C_c`. This is the real form of `fit::icm::block_cost`. -/
def blockReduce (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ) (c : CodeIndex) : ℝ :=
  a ^ 2 * (∑ p, ∑ q, H p q * codebook c p * codebook c q)
    - 2 * a * (∑ p, ∑ q, H p q * codebook c p * x q)

/-- **The block cost is the expansion with `a₂ = 0`.** `Perf.FaceTypedE8.joint_block_cost_expand` at
`g₁ = C_c`, `g₂ = 0`, `a₁ = a`, `a₂ = 0`: the two-plane identity degenerates to the one-plane block
cost plus the code-independent `xᵀHx`. -/
theorem blockQuad_eq_blockReduce_add (H : Fin 8 → Fin 8 → ℝ) (hH : ∀ p q, H p q = H q p)
    (x : Fin 8 → ℝ) (a : ℝ) (c : CodeIndex) :
    blockQuad H x a c = blockReduce H x a c + ∑ p, ∑ q, H p q * x p * x q := by
  have h := joint_block_cost_expand H hH (codebook c) (fun _ => 0) x a 0
  simpa [blockQuad, blockReduce, zero_mul, mul_zero, zero_add, add_zero] using h

/-- **The reduction is the body's word cost.** With `dot_p = Σ_q H_pq x_q` — the metric-applied
residual the sweep carries in `f_res` — `blockReduce` is `a²·q_c − 2a·dot_c`, the value
`fit::icm::block_cost` computes in Q96.96 (`aa·q − two_a·(dot << 32)`). -/
theorem blockReduce_eq_word (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ) (c : CodeIndex) :
    blockReduce H x a c
      = a ^ 2 * (∑ p, ∑ q, H p q * codebook c p * codebook c q)
        - 2 * a * (∑ p, codebook c p * (∑ q, H p q * x q)) := by
  have hlin : ∑ p, ∑ q, H p q * codebook c p * x q
      = ∑ p, codebook c p * (∑ q, H p q * x q) := by
    refine Finset.sum_congr rfl fun p _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun q _ => by ring
  rw [blockReduce, hlin]

/-- The codebook decision of one block: the `CODE_LEN`-way argmin of the block cost, first index on a
tie. `best_code_q` computes the same map in Q96.96. -/
def icmStep (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ) : CodeIndex :=
  (Finset.exists_min_image (Finset.univ : Finset CodeIndex) (fun c => blockReduce H x a c)
    ⟨0, Finset.mem_univ 0⟩).choose

/-- **The specification of one block decision.** The returned index minimises the block cost over the
whole 256-row codebook; its membership in the table is structure, not a side condition
(`CodeIndex` indexes `codebook`). -/
structure IcmStepSpec (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ) (c : CodeIndex) : Prop where
  /-- the decision is a codebook argmin of the block cost -/
  argmin : ∀ c' : CodeIndex, blockReduce H x a c ≤ blockReduce H x a c'

/-- **The canonical step meets the specification.** -/
theorem icmStep_spec (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ) :
    IcmStepSpec H x a (icmStep H x a) := by
  have h := (Finset.exists_min_image (Finset.univ : Finset CodeIndex)
    (fun c => blockReduce H x a c) ⟨0, Finset.mem_univ 0⟩).choose_spec
  exact ⟨fun c' => h.2 c' (Finset.mem_univ c')⟩

/-- **Any index meeting the specification attains the canonical argmin's cost.** The specification
pins the cost, not the index (ties are left to `best_code_q`'s first-index rule). -/
theorem spec_cost_eq_argmin (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ) (c : CodeIndex)
    (h : IcmStepSpec H x a c) :
    blockReduce H x a c = blockReduce H x a (icmStep H x a) :=
  le_antisymm (h.argmin _) ((icmStep_spec H x a).argmin c)

/-- **A specification-conforming decision never increases the block cost** (in the metric `H`). -/
theorem spec_quad_le (H : Fin 8 → Fin 8 → ℝ) (hH : ∀ p q, H p q = H q p) (x : Fin 8 → ℝ)
    (a : ℝ) (c : CodeIndex) (h : IcmStepSpec H x a c) (c' : CodeIndex) :
    blockQuad H x a c ≤ blockQuad H x a c' := by
  rw [blockQuad_eq_blockReduce_add H hH x a c, blockQuad_eq_blockReduce_add H hH x a c']
  linarith [h.argmin c']

/-- The sweep's `J` over a finite block set. -/
def sweepTotal {ι : Type} (s : Finset ι) (cost : ι → ℝ) : ℝ := ∑ i ∈ s, cost i

/-- A sweep whose every decision is non-increasing leaves `J` non-increasing. -/
theorem sweep_step_le {ι : Type} (s : Finset ι) (cost cost' : ι → ℝ)
    (h : ∀ i ∈ s, cost' i ≤ cost i) : sweepTotal s cost' ≤ sweepTotal s cost :=
  Finset.sum_le_sum h

/-- A sweep with one strictly improving decision leaves `J` strictly decreasing. -/
theorem sweep_step_lt {ι : Type} (s : Finset ι) (cost cost' : ι → ℝ)
    (hle : ∀ i ∈ s, cost' i ≤ cost i) (i₀ : ι) (hi₀ : i₀ ∈ s) (hlt : cost' i₀ < cost i₀) :
    sweepTotal s cost' < sweepTotal s cost :=
  Finset.sum_lt_sum hle ⟨i₀, hi₀, hlt⟩

/-- **The per-block `J` update of the step is a strict decrease**
(`Perf.CostCorrespondence.e8_icm_global_descent_dominance`). -/
theorem block_delta_descends {j_old c_old c_new : ℝ} (h : c_new < c_old) :
    j_old + (c_new - c_old) < j_old :=
  Perf.CostCorrespondence.e8_icm_global_descent_dominance h

/-- **A specification-conforming sweep never increases `J`.** Each block's decision is an argmin, so
each block cost weakly falls; the sweep total falls with it. -/
theorem spec_sweep_le {ι : Type} (s : Finset ι) (H : ι → Fin 8 → Fin 8 → ℝ)
    (hH : ∀ i ∈ s, ∀ p q, H i p q = H i q p) (x : ι → Fin 8 → ℝ) (a : ι → ℝ)
    (c c' : ι → CodeIndex) (h : ∀ i ∈ s, IcmStepSpec (H i) (x i) (a i) (c' i)) :
    sweepTotal s (fun i => blockQuad (H i) (x i) (a i) (c' i))
      ≤ sweepTotal s (fun i => blockQuad (H i) (x i) (a i) (c i)) :=
  sweep_step_le s _ _ fun i hi => spec_quad_le (H i) (hH i hi) (x i) (a i) (c' i) (h i hi) (c i)

/-- **A sweep with one strictly improving block strictly lowers `J`.** -/
theorem spec_sweep_lt {ι : Type} (s : Finset ι) (H : ι → Fin 8 → Fin 8 → ℝ)
    (hH : ∀ i ∈ s, ∀ p q, H i p q = H i q p) (x : ι → Fin 8 → ℝ) (a : ι → ℝ)
    (c c' : ι → CodeIndex) (h : ∀ i ∈ s, IcmStepSpec (H i) (x i) (a i) (c' i))
    (i₀ : ι) (hi₀ : i₀ ∈ s)
    (hlt : blockQuad (H i₀) (x i₀) (a i₀) (c' i₀) < blockQuad (H i₀) (x i₀) (a i₀) (c i₀)) :
    sweepTotal s (fun i => blockQuad (H i) (x i) (a i) (c' i))
      < sweepTotal s (fun i => blockQuad (H i) (x i) (a i) (c i)) :=
  sweep_step_lt s _ _ (fun i hi => spec_quad_le (H i) (hH i hi) (x i) (a i) (c' i) (h i hi) (c i))
    i₀ hi₀ hlt

/-! ## The joint two-plane decision (`e8_joint_icm_2plane`) -/

/-- The joint pair cost `2·a₁a₂·X_{c₁c₂} + A_{c₁} + B_{c₂}` of `fit::icm::pair_cost`, with
`X_{c₁c₂} = C_{c₁}ᵀ H C_{c₂}`. -/
def pairReduce (H : Fin 8 → Fin 8 → ℝ) (x₁ x₂ : Fin 8 → ℝ) (a₁ a₂ : ℝ)
    (p : CodeIndex × CodeIndex) : ℝ :=
  2 * a₁ * a₂ * (∑ i, ∑ j, H i j * codebook p.1 i * codebook p.2 j)
    + blockReduce H x₁ a₁ p.1 + blockReduce H x₂ a₂ p.2

/-- A finite nonempty index set has an argmin (`Finset.exists_min_image` at `Finset.univ`). -/
theorem exists_min {ι : Type} [Fintype ι] [Nonempty ι] (g : ι → ℝ) :
    ∃ i, ∀ j, g i ≤ g j :=
  Exists.elim
    (Finset.exists_min_image (s := (Finset.univ : Finset ι)) (f := g) Finset.univ_nonempty)
    (fun i hp => ⟨i, fun j => hp.2 j (Finset.mem_univ j)⟩)

/-- The joint pair minimum exists: the codebook product is a nonempty finite set. -/
theorem exists_pair_min (H : Fin 8 → Fin 8 → ℝ) (x₁ x₂ : Fin 8 → ℝ) (a₁ a₂ : ℝ) :
    ∃ p : CodeIndex × CodeIndex,
      ∀ q : CodeIndex × CodeIndex, pairReduce H x₁ x₂ a₁ a₂ p ≤ pairReduce H x₁ x₂ a₁ a₂ q := by
  let g : CodeIndex × CodeIndex → ℝ := fun p => pairReduce H x₁ x₂ a₁ a₂ p
  obtain ⟨p, hp⟩ := exists_min g
  exact ⟨p, hp⟩

/-- The joint pair decision: the `CODE_LEN²`-way argmin of `pairReduce` over the product codebook
(`best_pair_q`). -/
def pairStep (H : Fin 8 → Fin 8 → ℝ) (x₁ x₂ : Fin 8 → ℝ) (a₁ a₂ : ℝ) : CodeIndex × CodeIndex :=
  (exists_pair_min H x₁ x₂ a₁ a₂).choose

/-- **The joint decision is the pair-cost argmin over the product codebook.** -/
theorem pairStep_spec (H : Fin 8 → Fin 8 → ℝ) (x₁ x₂ : Fin 8 → ℝ) (a₁ a₂ : ℝ)
    (p : CodeIndex × CodeIndex) :
    pairReduce H x₁ x₂ a₁ a₂ (pairStep H x₁ x₂ a₁ a₂) ≤ pairReduce H x₁ x₂ a₁ a₂ p :=
  (exists_pair_min H x₁ x₂ a₁ a₂).choose_spec p

/-- **The joint minimum is at most each fixed-first-plane slice**
(`Perf.FaceTypedE8.joint_move_le_planewise`): joint optimisation is never worse than holding one
plane's code fixed. -/
theorem joint_pair_le_slice_left (f : CodeIndex → CodeIndex → ℝ) (a : CodeIndex) :
    ((Finset.univ : Finset CodeIndex) ×ˢ Finset.univ).inf'
        (Finset.univ_nonempty.product Finset.univ_nonempty) (fun x => f x.1 x.2)
      ≤ Finset.univ.inf' Finset.univ_nonempty (fun b => f a b) :=
  joint_move_le_planewise Finset.univ Finset.univ Finset.univ_nonempty Finset.univ_nonempty f a
    (Finset.mem_univ a)

/-- **And at most each fixed-second-plane slice**
(`Perf.FaceTypedE8.joint_move_le_planewise'`). -/
theorem joint_pair_le_slice_right (f : CodeIndex → CodeIndex → ℝ) (b : CodeIndex) :
    ((Finset.univ : Finset CodeIndex) ×ˢ Finset.univ).inf'
        (Finset.univ_nonempty.product Finset.univ_nonempty) (fun x => f x.1 x.2)
      ≤ Finset.univ.inf' Finset.univ_nonempty (fun a => f a b) :=
  joint_move_le_planewise' Finset.univ Finset.univ Finset.univ_nonempty Finset.univ_nonempty f b
    (Finset.mem_univ b)

/-- **The pair code rate.** The joint decision indexes `2^16 = 65536` pairs
(`Perf.FaceTypedE8.pair_rate_eq`). -/
theorem pair_codebook_rate : Fintype.card (CodeIndex × CodeIndex) = 2 ^ 16 :=
  pair_rate_eq

/-! ## The codebook rows -/

/-- Row `c < 128` of the table is the spinor row `c`. -/
theorem codewordSpec_spinor {i k : ℕ} (hi : i < 128) :
    Learn.E8Code.codewordSpec i k = Learn.E8Code.spinorSpec i k := by
  unfold Learn.E8Code.codewordSpec
  rw [if_pos hi]

/-- Row `128 ≤ c < 240` of the table is the `D₈` root `c − 128`. -/
theorem codewordSpec_root {i k : ℕ} (h1 : 128 ≤ i) (h2 : i < 240) :
    Learn.E8Code.codewordSpec i k = Learn.E8Code.rootSpec (i - 128) k := by
  unfold Learn.E8Code.codewordSpec
  rw [if_neg (by omega), if_pos h2]

/-- Row `240 ≤ c` of the table is the `O₈` vertex `c − 240`. -/
theorem codewordSpec_o8 {i k : ℕ} (h : 240 ≤ i) :
    Learn.E8Code.codewordSpec i k = Learn.E8Code.o8Spec (i - 240) k := by
  unfold Learn.E8Code.codewordSpec
  rw [if_neg (by omega), if_neg (by omega)]

/-- A `D₈` root coordinate is `0` or `±2`. -/
theorem rootSpec_val (j k : ℕ) :
    Learn.E8Code.rootSpec j k = 0 ∨ Learn.E8Code.rootSpec j k = 2 * Learn.E8Code.ONE
      ∨ Learn.E8Code.rootSpec j k = -(2 * Learn.E8Code.ONE) := by
  unfold Learn.E8Code.rootSpec
  by_cases h1 : k = (Learn.E8Code.rootPairs.getD (j / 4) (0, 0)).1
  · rw [if_pos h1]
    by_cases h2 : j % 4 / 2 = 0
    · rw [if_pos h2]; exact Or.inr (Or.inr rfl)
    · rw [if_neg h2]; exact Or.inr (Or.inl rfl)
  · rw [if_neg h1]
    by_cases h2 : k = (Learn.E8Code.rootPairs.getD (j / 4) (0, 0)).2
    · rw [if_pos h2]
      by_cases h3 : j % 2 = 0
      · rw [if_pos h3]; exact Or.inr (Or.inr rfl)
      · rw [if_neg h3]; exact Or.inr (Or.inl rfl)
    · rw [if_neg h2]; exact Or.inl rfl

/-- An `O₈` vertex coordinate is `0` or `±SQRT8`. -/
theorem o8Spec_val (j k : ℕ) :
    Learn.E8Code.o8Spec j k = 0 ∨ Learn.E8Code.o8Spec j k = Learn.E8Code.SQRT8
      ∨ Learn.E8Code.o8Spec j k = -Learn.E8Code.SQRT8 := by
  unfold Learn.E8Code.o8Spec
  by_cases h1 : k = j / 2
  · rw [if_pos h1]
    by_cases h2 : j % 2 = 0
    · rw [if_pos h2]; exact Or.inr (Or.inr rfl)
    · rw [if_neg h2]; exact Or.inr (Or.inl rfl)
  · rw [if_neg h1]; exact Or.inl rfl

/-- The half-cube spinor row `i < 128`, in real Q32.32 units. -/
theorem codebook_spinor (i : ℕ) (hi : i < 128) :
    ∃ s : Spin8, ∀ k : Fin 8, codebook ⟨i, by omega⟩ k = ((s.1 k : ℤˣ) : ℝ) := by
  obtain ⟨s, hs⟩ := Learn.E8Code.e8_codeword_spin8 i hi
  refine ⟨s, fun k => ?_⟩
  have hk : Learn.E8Code.codewordSpec i k.val = Learn.E8Code.ONE * ((s.1 k : ℤˣ) : ℤ) := hs k
  simp only [codebook, hk, Learn.E8Code.ONE]
  push_cast
  norm_num

/-- A codebook row is a half-cube spinor (`±1`, even parity), a `D₈` root (`0`, `±2`) or an `O₈`
vertex (`0`, `±√8` in Q32.32 units). The three families are the `E₈ ∪ O₈` table. -/
def RowShape (c : CodeIndex) : Prop :=
  (c.val < 128 ∧ ∃ s : Spin8, ∀ k : Fin 8, codebook c k = ((s.1 k : ℤˣ) : ℝ))
  ∨ (128 ≤ c.val ∧ c.val < 240
      ∧ ∀ k : Fin 8, codebook c k = 0 ∨ codebook c k = 2 ∨ codebook c k = -2)
  ∨ (240 ≤ c.val
      ∧ ∀ k : Fin 8, codebook c k = 0
          ∨ codebook c k = (Learn.E8Code.SQRT8 : ℝ) / 2 ^ 32
          ∨ codebook c k = -((Learn.E8Code.SQRT8 : ℝ) / 2 ^ 32))

/-- **Every codebook row has `E₈ ∪ O₈` shape.** -/
theorem codebook_row_shape (c : CodeIndex) : RowShape c := by
  rcases Nat.lt_or_ge c.val 128 with h | h
  · obtain ⟨s, hs⟩ := codebook_spinor c.val h
    exact Or.inl ⟨h, s, fun k => by rw [hs k]⟩
  rcases Nat.lt_or_ge c.val 240 with h2 | h2
  · refine Or.inr (Or.inl ⟨h, h2, fun k => ?_⟩)
    rw [codebook, codewordSpec_root h h2]
    rcases rootSpec_val (c.val - 128) k.val with h0 | h0 | h0
    · exact Or.inl (by rw [h0]; norm_num)
    · exact Or.inr (Or.inl (by rw [h0, Learn.E8Code.ONE]; norm_num))
    · exact Or.inr (Or.inr (by rw [h0, Learn.E8Code.ONE]; norm_num))
  · refine Or.inr (Or.inr ⟨h2, fun k => ?_⟩)
    rw [codebook, codewordSpec_o8 h2]
    rcases o8Spec_val (c.val - 240) k.val with h0 | h0 | h0
    · exact Or.inl (by rw [h0]; norm_num)
    · exact Or.inr (Or.inl (by rw [h0]))
    · exact Or.inr (Or.inr (by rw [h0]; push_cast; ring))

/-- **The selected code has `E₈ ∪ O₈` shape**: any decision over `CodeIndex` lands on a table row. -/
theorem icmStep_row_shape (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ) :
    RowShape (icmStep H x a) :=
  codebook_row_shape _

/-- **The table's cardinality.** `256 = 128 + 112 + 16`
(`Perf.CostCorrespondence.e8_o8_codebook_cardinality`). -/
theorem codebook_card : Fintype.card CodeIndex = 128 + 112 + 16 := by
  rw [Fintype.card_fin, (Perf.CostCorrespondence.e8_o8_codebook_cardinality).1]

/-! ## The conditional refinement of the extracted body -/

/-- **From the body's asserted argmin to the specification.** `h` is the single fact the generated
body's externals (the 31 `axiom`s listed in the module header, led by `chart.pool.width_global` and the
AVX2 decode path) keep out of the closed cone: that `best_code_q`'s returned index minimises
`blockReduce` over the codebook. Under it
the decision meets `IcmStepSpec` and attains the canonical argmin's block cost; through
`spec_quad_le` and `spec_sweep_le` that gives the block-cost and sweep-`J` non-increase. -/
theorem icm_step_spec_of_body_argmin (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ)
    (c : CodeIndex) (h : ∀ c' : CodeIndex, blockReduce H x a c ≤ blockReduce H x a c') :
    IcmStepSpec H x a c ∧ blockReduce H x a c = blockReduce H x a (icmStep H x a) :=
  ⟨⟨h⟩, le_antisymm (h _) ((icmStep_spec H x a).argmin c)⟩

/-! ## Joint witness: the hypotheses are reachable and the specification is non-vacuous

`H = I`, `x = 1`, `a = 1`, on the extracted codebook. The block cost is then
`Σ_p (C_c(p) − 1)² − 8`, so every code costs at least `−8`, and the all-`+1` spinor row `c = 0`
(`codewordSpec 0 k = ONE`, the key-`0` sign vector) attains `−8`. -/

/-- The diagonal metric block. -/
def diagBlock : Fin 8 → Fin 8 → ℝ := fun p q => if p = q then 1 else 0

/-- The all-`+1` residual vector. -/
def onesVec : Fin 8 → ℝ := fun _ => 1

theorem diagBlock_symm : ∀ p q, diagBlock p q = diagBlock q p := by
  intro p q; unfold diagBlock; by_cases h : p = q <;> simp [h, eq_comm]

/-- At `H = I`, `x = 1`, `a = 1` the reduction is `Σ_p C_c(p)² − 2·Σ_p C_c(p)`. -/
theorem diagBlock_reduce (c : CodeIndex) :
    blockReduce diagBlock onesVec 1 c
      = ∑ p, codebook c p * codebook c p - 2 * ∑ p, codebook c p := by
  have hq : ∀ p : Fin 8,
      ∑ q, diagBlock p q * codebook c p * codebook c q = codebook c p * codebook c p := by
    intro p
    rw [Finset.sum_eq_single p]
    · simp [diagBlock]
    · intro q _ hq
      have hpq : ¬ (p = q) := fun h => hq h.symm
      simp only [diagBlock, if_neg hpq, zero_mul]
    · intro hp; exact absurd (Finset.mem_univ p) hp
  have hq1 : ∀ p : Fin 8,
      ∑ q, diagBlock p q * codebook c p * onesVec q = codebook c p := by
    intro p
    rw [Finset.sum_eq_single p]
    · simp [diagBlock, onesVec]
    · intro q _ hq
      have hpq : ¬ (p = q) := fun h => hq h.symm
      simp only [diagBlock, if_neg hpq, zero_mul]
    · intro hp; exact absurd (Finset.mem_univ p) hp
  unfold blockReduce
  rw [show (∑ p, ∑ q, diagBlock p q * codebook c p * codebook c q)
      = ∑ p, codebook c p * codebook c p from Finset.sum_congr rfl fun p _ => hq p]
  rw [show (∑ p, ∑ q, diagBlock p q * codebook c p * onesVec q)
      = ∑ p, codebook c p from Finset.sum_congr rfl fun p _ => hq1 p]
  norm_num

/-- Row `0` of the table is the all-`+1` spinor. -/
theorem codebook_zero (k : Fin 8) : codebook 0 k = 1 := by
  have hkey : Learn.E8Code.spinorKey 0 = 0 := by
    rw [Learn.E8Code.spinorKey_eq 0 (by norm_num)]
    simp [Learn.E8Code.parityNat, Learn.E8Code.bitN]
  have hcw : Learn.E8Code.codewordSpec 0 k.val = Learn.E8Code.ONE := by
    rw [codewordSpec_spinor (i := 0) (k := k.val) (by norm_num), Learn.E8Code.spinorSpec, hkey]
    simp [Learn.E8Code.signAt]
  simp only [codebook, Fin.val_zero, hcw, Learn.E8Code.ONE]
  norm_num

/-- **Every code costs at least `−8`** at this witness: `Σ_p (C_c(p) − 1)² ≥ 0`. -/
theorem diagBlock_reduce_ge (c : CodeIndex) : -8 ≤ blockReduce diagBlock onesVec 1 c := by
  have hnn : 0 ≤ ∑ p : Fin 8, (codebook c p - 1) * (codebook c p - 1) :=
    Finset.sum_nonneg fun p _ => mul_self_nonneg _
  have hid : blockReduce diagBlock onesVec 1 c
      = (∑ p : Fin 8, (codebook c p - 1) * (codebook c p - 1)) - 8 := by
    rw [diagBlock_reduce]
    have hsq : ∑ p : Fin 8, (codebook c p - 1) * (codebook c p - 1)
        = ∑ p : Fin 8, (codebook c p * codebook c p - 2 * codebook c p + 1) :=
      Finset.sum_congr rfl fun p _ => by ring
    rw [hsq, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
    simp
  rw [hid]; linarith

/-! ## The extracted `best_code_q`: the argmin spec in the word arithmetic

`fit::icm::best_code_q` (`src/learn/fold/fit/icm.rs:243`) is the `CODE_LEN`-way argmin of the exact
Q96.96 block cost `fit::icm::block_cost`, `aa·q − two_a·(dot << 32)`, over one row-block. Everything
below is about the *extracted* definitions (`ApeironWord`), so the theorem is a statement about the body
that executes, not about a restatement of it. -/

open Aeneas Aeneas.Std Result

/-- The integer value `fit::icm::block_cost` computes: `aa·q − two_a·(dot·2^32)`, every operation
checked `i128`. At the sweep's call site (`icm.rs:966`) `aa = (a·2^32)²`, `two_a = 2a·2^32`,
`q = Q·2^32`, `dot = D·2^32`, so this is `2^96·(a²Q − 2aD)` — `2^96` times `blockReduce`. -/
def wordCost (aa two_a q dot : ℤ) : ℤ := aa * q - two_a * (dot * 2 ^ 32)

/-- `qmul` returns `Ok` only on a product inside `i128`, and then the value is the product. -/
theorem qmul_ok {a b z : Std.I128}
    (h : apeiron_word.fold_port.fit.icm.qmul a b = ok (core.result.Result.Ok z)) :
    z.val = a.val * b.val := by
  unfold apeiron_word.fold_port.fit.icm.qmul at h
  simp only [lift, bind_tc_ok] at h
  cases hm : a.checked_mul b with
  | none => simp [hm] at h
  | some v =>
    have hspec := core.num.checked_mul_IScalar_bv_spec a b
    rw [show core.num.checked_mul_IScalar a b = some v from hm] at hspec
    simp only [hm] at h
    rw [show z = v from by simpa using h.symm]
    exact hspec.2.2.1

/-- `qsub` returns `Ok` only on a difference inside `i128`, and then the value is the difference. -/
theorem qsub_ok {a b z : Std.I128}
    (h : apeiron_word.fold_port.fit.icm.qsub a b = ok (core.result.Result.Ok z)) :
    z.val = a.val - b.val := by
  unfold apeiron_word.fold_port.fit.icm.qsub at h
  simp only [lift, bind_tc_ok] at h
  cases hm : a.checked_sub b with
  | none => simp [hm] at h
  | some v =>
    have hspec := core.num.checked_sub_IScalar_bv_spec a b
    rw [show core.num.checked_sub_IScalar a b = some v from hm] at hspec
    simp only [hm] at h
    rw [show z = v from by simpa using h.symm]
    exact hspec.2.2.1

/-- The `i128` left shift of a widened `i64` by `32` is exact: `d·2^32` fits (`|d| < 2^63`). -/
theorem shl_i128_32 (v : Std.I128) (hlo : -(2 ^ 95) < v.val) (hhi : v.val < 2 ^ 95) :
    ∃ z, (v <<< 32#i32) = ok z ∧ z.val = v.val * 2 ^ 32 := by
  have hsz : Std.I128.size = 2 ^ 128 := by simp [Std.I128.size, Std.I128.numBits_def]
  have hm2 : Std.I128.size / 2 = 2 ^ 127 := by rw [hsz]; norm_num
  have hm3 : (Std.I128.size + 1) / 2 = 2 ^ 127 := by rw [hsz]; norm_num
  have h := Std.I128.ShiftLeft_IScalar_spec v (32#i32) (by scalar_tac) (by scalar_tac)
  obtain ⟨z, hz, hvz, _hbv⟩ := WP.spec_imp_exists h
  refine ⟨z, hz, ?_⟩
  have h32 : (32#i32).toNat = 32 := rfl
  rw [hvz, h32, Int.shiftLeft_eq]
  apply Int.bmod_eq_of_le
  · have h2 : ((Std.I128.size : ℤ) / 2) = 2 ^ 127 := by rw [hsz]; norm_num
    rw [h2]; omega
  · have h3 : (((Std.I128.size : ℤ) + 1) / 2) = 2 ^ 127 := by rw [hsz]; norm_num
    rw [h3]; omega

/-- **The extracted `block_cost` computes `wordCost`.** Whenever the body returns `Ok z` — both checked
products and the checked difference succeeded — `z` is exactly `aa·q − two_a·(dot·2^32)`. -/
theorem block_cost_ok {aa two_a : Std.I128} {q dot : Std.I64} {z : Std.I128}
    (h : apeiron_word.fold_port.fit.icm.block_cost aa two_a q dot = ok (core.result.Result.Ok z)) :
    z.val = wordCost aa.val two_a.val q.val dot.val := by
  unfold apeiron_word.fold_port.fit.icm.block_cost at h
  simp only [lift, bind_tc_ok] at h
  cases hm : aa.checked_mul (core.convert.num.FromI128I64.from q) with
  | none =>
    have hq : apeiron_word.fold_port.fit.icm.qmul aa (core.convert.num.FromI128I64.from q)
        = ok (core.result.Result.Err apeiron_word.fold_port.fit.icm.IcmError.Overflow) := by
      unfold apeiron_word.fold_port.fit.icm.qmul
      simp only [lift, bind_tc_ok, hm, core.option.Option.ok_or_none]
    rw [hq] at h
    simp only [core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok] at h
    simp [core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual] at h
  | some val =>
    have hspec := core.num.checked_mul_IScalar_bv_spec aa (core.convert.num.FromI128I64.from q)
    rw [show core.num.checked_mul_IScalar aa (core.convert.num.FromI128I64.from q) = some val
      from hm] at hspec
    have hvalv : val.val = aa.val * q.val := by
      rw [hspec.2.2.1, core.convert.num.FromI128I64.from_val_eq]
    have hq : apeiron_word.fold_port.fit.icm.qmul aa (core.convert.num.FromI128I64.from q)
        = ok (core.result.Result.Ok val) := by
      unfold apeiron_word.fold_port.fit.icm.qmul
      simp only [lift, bind_tc_ok, hm, core.option.Option.ok_or_some]
    rw [hq] at h
    simp only [core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok] at h
    have h64 : (IScalarTy.I64.numBits : ℕ) = 64 := rfl
    have hbnd := dot.hBounds
    rw [h64] at hbnd
    obtain ⟨i2, hi2, hi2v⟩ := shl_i128_32 (core.convert.num.FromI128I64.from dot)
      (by rw [core.convert.num.FromI128I64.from_val_eq]; omega)
      (by rw [core.convert.num.FromI128I64.from_val_eq]; omega)
    have hi2v' : i2.val = dot.val * 2 ^ 32 := by
      rw [hi2v, core.convert.num.FromI128I64.from_val_eq]
    rw [hi2] at h
    simp only [bind_tc_ok] at h
    cases hn : two_a.checked_mul i2 with
    | none =>
      have hr : apeiron_word.fold_port.fit.icm.qmul two_a i2
          = ok (core.result.Result.Err apeiron_word.fold_port.fit.icm.IcmError.Overflow) := by
        unfold apeiron_word.fold_port.fit.icm.qmul
        simp only [lift, bind_tc_ok, hn, core.option.Option.ok_or_none]
      rw [hr] at h
      simp only [bind_tc_ok] at h
      simp [core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual] at h
    | some val1 =>
      have hspec1 := core.num.checked_mul_IScalar_bv_spec two_a i2
      rw [show core.num.checked_mul_IScalar two_a i2 = some val1 from hn] at hspec1
      have hval1v : val1.val = two_a.val * i2.val := hspec1.2.2.1
      have hr : apeiron_word.fold_port.fit.icm.qmul two_a i2
          = ok (core.result.Result.Ok val1) := by
        unfold apeiron_word.fold_port.fit.icm.qmul
        simp only [lift, bind_tc_ok, hn, core.option.Option.ok_or_some]
      rw [hr] at h
      simp only [bind_tc_ok] at h
      cases hs : val.checked_sub val1 with
      | none =>
        simp only [apeiron_word.fold_port.fit.icm.qsub, lift, bind_tc_ok, hs,
          core.option.Option.ok_or_none] at h
        exact absurd h (by simp)
      | some w =>
        have hspecs := core.num.checked_sub_IScalar_bv_spec val val1
        rw [show core.num.checked_sub_IScalar val val1 = some w from hs] at hspecs
        simp only [apeiron_word.fold_port.fit.icm.qsub, lift, bind_tc_ok, hs,
          core.option.Option.ok_or_some] at h
        rw [show z = w from by simpa using h.symm]
        rw [hspecs.2.2.1, hvalv, hval1v, hi2v']
        unfold wordCost
        ring

/-- `block_cost` is total: every operation in it is checked or total, and the shift by `32` of a
widened `i64` cannot fail. -/
theorem block_cost_total (aa two_a : Std.I128) (q dot : Std.I64) :
    ∃ r, apeiron_word.fold_port.fit.icm.block_cost aa two_a q dot = ok r := by
  have h64 : (IScalarTy.I64.numBits : ℕ) = 64 := rfl
  have hbnd := dot.hBounds
  rw [h64] at hbnd
  obtain ⟨i2, hi2, _hi2v⟩ := shl_i128_32 (core.convert.num.FromI128I64.from dot)
    (by rw [core.convert.num.FromI128I64.from_val_eq]; omega)
    (by rw [core.convert.num.FromI128I64.from_val_eq]; omega)
  unfold apeiron_word.fold_port.fit.icm.block_cost apeiron_word.fold_port.fit.icm.qmul
    apeiron_word.fold_port.fit.icm.qsub
  simp only [lift, bind_tc_ok, core.option.Option.ok_or]
  rw [hi2]
  simp only [bind_tc_ok]
  cases h1 : aa.checked_mul (core.convert.num.FromI128I64.from q) with
  | none =>
    exact ⟨core.result.Result.Err apeiron_word.fold_port.fit.icm.IcmError.Overflow, by
      simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch,
        core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
        core.convert.FromSame.from]⟩
  | some val =>
    dsimp only []
    cases h2 : two_a.checked_mul i2 with
    | none =>
      exact ⟨core.result.Result.Err apeiron_word.fold_port.fit.icm.IcmError.Overflow, by
        simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch,
          core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
          core.convert.FromSame.from]⟩
    | some val1 =>
      dsimp only []
      cases h3 : val.checked_sub val1 with
      | none =>
        exact ⟨core.result.Result.Err apeiron_word.fold_port.fit.icm.IcmError.Overflow, by
          simp only [h3, bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch,
            core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
            core.convert.FromSame.from]⟩
      | some w =>
        exact ⟨core.result.Result.Ok w, by
          simp only [h3, bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch,
            core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
            core.convert.FromSame.from]⟩

/-- `l[i]!` is `l[i]` in bounds. -/
theorem list_getElem!_eq {α : Type} [Inhabited α] {l : List α} {i : ℕ} (h : i < l.length) :
    l[i]! = l[i] := by
  rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem h]
  rfl

/-- **The extracted argmin loop.** On a run that ends without an error, the returned index is below
`256` and its exact word cost is no larger than any entry's. The invariant keeps the running best
index below `256`, every visited entry cost `Ok`, the running best cost attained at the running best
index, and the running best cost no larger than every visited entry cost. -/
theorem best_code_q_loop_spec (aa two_a : Std.I128) (q dot : Slice Std.I64)
    (hq : 256 ≤ q.val.length) (hd : 256 ≤ dot.val.length) :
    apeiron_word.fold_port.fit.icm.best_code_q_loop aa two_a q dot 0#u16 core.num.I128.MAX none
        0#usize ⦃ fun r : Std.U16 × Option apeiron_word.fold_port.fit.icm.IcmError =>
      r.2 = none → r.1.val < 256 ∧
        ∃ c : Std.I128,
          apeiron_word.fold_port.fit.icm.block_cost aa two_a q.val[r.1.val]! dot.val[r.1.val]!
              = ok (core.result.Result.Ok c) ∧
          ∀ i < 256, ∃ d : Std.I128,
            apeiron_word.fold_port.fit.icm.block_cost aa two_a q.val[i]! dot.val[i]!
              = ok (core.result.Result.Ok d) ∧ c.val ≤ d.val ⦄ := by
  unfold apeiron_word.fold_port.fit.icm.best_code_q_loop
  apply loop.spec_decr_nat
    (measure := fun s : Std.U16 × Std.I128 ×
        Option apeiron_word.fold_port.fit.icm.IcmError × Std.Usize => 256 - s.2.2.2.val)
    (inv := fun s : Std.U16 × Std.I128 × Option apeiron_word.fold_port.fit.icm.IcmError × Std.Usize =>
      s.1.val < 256 ∧
        (s.2.2.1 = none →
          s.2.2.2.val ≤ 256 ∧
          (∀ i < s.2.2.2.val, ∃ c : Std.I128,
              apeiron_word.fold_port.fit.icm.block_cost aa two_a q.val[i]! dot.val[i]!
                = ok (core.result.Result.Ok c) ∧ s.2.1.val ≤ c.val) ∧
          (s.2.2.2.val = 0 ∨ ∃ c : Std.I128,
              apeiron_word.fold_port.fit.icm.block_cost aa two_a q.val[s.1.val]! dot.val[s.1.val]!
                = ok (core.result.Result.Ok c) ∧ c.val = s.2.1.val) ∧
          (s.2.2.2.val = 0 → s.1.val = 0 ∧ s.2.1.val = core.num.I128.MAX)))
    (post := fun r : Std.U16 × Option apeiron_word.fold_port.fit.icm.IcmError =>
      r.2 = none → r.1.val < 256 ∧
        ∃ c : Std.I128,
          apeiron_word.fold_port.fit.icm.block_cost aa two_a q.val[r.1.val]! dot.val[r.1.val]!
              = ok (core.result.Result.Ok c) ∧
          ∀ i < 256, ∃ d : Std.I128,
            apeiron_word.fold_port.fit.icm.block_cost aa two_a q.val[i]! dot.val[i]!
              = ok (core.result.Result.Ok d) ∧ c.val ≤ d.val)
  · rintro ⟨b, bc, e, j'⟩ ⟨hb, hinv⟩
    dsimp only at hb hinv ⊢
    unfold apeiron_word.fold_port.fit.icm.best_code_q_loop.body
    have hcl : apeiron_word.fold_port.fit.icm.CODE_LEN = 256#usize := by
      unfold apeiron_word.fold_port.fit.icm.CODE_LEN apeiron_word.learn_port.e8.CODEBOOK_LEN
      rfl
    rw [hcl]
    by_cases hj : j' < 256#usize
    · rw [if_pos hj]
      have hjv : j'.val < 256 := by scalar_tac
      have hcast : (UScalar.cast UScalarTy.U16 j').val = j'.val :=
        UScalar.cast_val_mod_pow_of_inBounds_eq .U16 j' (by scalar_tac)
      cases e with
      | none =>
        rw [if_pos (by rfl : core.option.Option.is_none (none : Option apeiron_word.fold_port.fit.icm.IcmError) = true)]
        step as ⟨x, hx⟩
        step as ⟨y, hy⟩
        obtain ⟨r0, hr0⟩ := block_cost_total aa two_a x y
        rw [hr0]
        simp only [bind_tc_ok]
        cases r0 with
        | Ok cost =>
          dsimp only
          have hr_cost : apeiron_word.fold_port.fit.icm.block_cost aa two_a q.val[j'.val]! dot.val[j'.val]!
              = ok (core.result.Result.Ok cost) := by
            rw [list_getElem!_eq (by omega), list_getElem!_eq (by omega), ← hx, ← hy]
            exact hr0
          by_cases hlt : cost < bc
          · rw [if_pos hlt]
            simp only [lift, bind_tc_ok]
            step as ⟨j1, hj1⟩
            obtain ⟨hjle, hcost, hatt, hz⟩ := hinv rfl
            refine ⟨by rw [hcast]; omega, ?_, by omega⟩
            intro _
            refine ⟨by omega, ?_, Or.inr ?_, fun h0 => absurd h0 (by omega)⟩
            · intro i hi
              rcases Nat.lt_succ_iff_lt_or_eq.mp (by omega : i < j'.val + 1) with h | h
              · obtain ⟨c, hc, hbc⟩ := hcost i h
                exact ⟨c, hc, by scalar_tac⟩
              · subst h
                exact ⟨cost, hr_cost, by scalar_tac⟩
            · refine ⟨cost, ?_, rfl⟩
              rw [hcast]; exact hr_cost
          · rw [if_neg hlt]
            simp only [bind_tc_ok]
            step as ⟨j1, hj1⟩
            obtain ⟨hjle, hcost, hatt, hz⟩ := hinv rfl
            refine ⟨hb, ?_, by omega⟩
            intro _
            refine ⟨by omega, ?_, ?_, fun h0 => absurd h0 (by omega)⟩
            · intro i hi
              rcases Nat.lt_succ_iff_lt_or_eq.mp (by omega : i < j'.val + 1) with h | h
              · exact hcost i h
              · subst h
                exact ⟨cost, hr_cost, by scalar_tac⟩
            · rcases hatt with h0 | ⟨c, hc, hcv⟩
              · obtain ⟨hb0, hbcMAX⟩ := hz h0
                have hbj : b.val = j'.val := by omega
                have h1 : bc.val ≤ cost.val := by scalar_tac
                have h2 : cost.val ≤ bc.val := by
                  rw [hbcMAX]
                  have hle : cost.val ≤ I128.rMax := by
                    have hb2 : cost.val < 170141183460469231731687303715884105728 :=
                      cost.hBounds.2
                    have hr : I128.rMax = 170141183460469231731687303715884105727 := rfl
                    rw [hr]
                    omega
                  simpa using hle
                exact Or.inr ⟨cost, by rw [hbj]; exact hr_cost, by omega⟩
              · exact Or.inr ⟨c, hc, hcv⟩
        | Err e' =>
          dsimp only
          step as ⟨j1, hj1⟩
          exact ⟨hb, (fun h => by cases h), by omega⟩
      | some e' =>
        simp only [core.option.Option.is_none, Option.isNone_some, Bool.false_eq_true, if_false]
        simp only [WP.spec_ok]
        intro h; cases h
    · rw [if_neg hj]
      simp only [WP.spec_ok]
      intro he
      obtain ⟨hjle, hcost, hatt, hz⟩ := hinv he
      have hj256 : j'.val = 256 := by scalar_tac
      refine ⟨hb, ?_⟩
      rcases hatt with h0 | ⟨c, hc, hcv⟩
      · exact absurd h0 (by omega)
      · refine ⟨c, hc, ?_⟩
        intro i hi
        obtain ⟨d, hd, hcd⟩ := hcost i (by omega)
        exact ⟨d, hd, by omega⟩
  · exact ⟨by simp, fun _ => ⟨by simp, by simp, Or.inl rfl, by simp⟩⟩

/-- **The argmin.** When the extracted `best_code_q` returns `Ok i`, the index is below `256` and the
exact word cost at `i` is no larger than every entry's. -/
theorem best_code_q_argmin (aa two_a : Std.I128) (q dot : Slice Std.I64)
    (hq : 256 ≤ q.val.length) (hd : 256 ≤ dot.val.length) (i : Std.Usize)
    (h : apeiron_word.fold_port.fit.icm.best_code_q aa two_a q dot
      = ok (core.result.Result.Ok i)) :
    i.val < 256 ∧
      ∃ c : Std.I128,
        apeiron_word.fold_port.fit.icm.block_cost aa two_a q.val[i.val]! dot.val[i.val]! = ok (core.result.Result.Ok c) ∧
        ∀ j < 256, ∃ d : Std.I128,
          apeiron_word.fold_port.fit.icm.block_cost aa two_a q.val[j]! dot.val[j]! = ok (core.result.Result.Ok d) ∧ c.val ≤ d.val := by
  obtain ⟨⟨b, e⟩, hloop, hpost⟩ := WP.spec_imp_exists (best_code_q_loop_spec aa two_a q dot hq hd)
  unfold apeiron_word.fold_port.fit.icm.best_code_q at h
  rw [hloop] at h
  simp only [bind_tc_ok] at h
  cases e with
  | some e' => simp at h
  | none =>
    simp only [lift, bind_tc_ok] at h
    obtain ⟨hb256, c, hc, hall⟩ := hpost rfl
    have hb : core.convert.num.FromUsizeU16.from b = i := by simpa using h
    subst hb
    have hv : (core.convert.num.FromUsizeU16.from b).val = b.val :=
      core.convert.num.FromUsizeU16.from_val_eq b
    rw [hv]
    exact ⟨hb256, c, hc, hall⟩

/-- **The caller's slice correspondence.** For every one of the 256 codebook rows, the `i128` value the
extracted `fit::icm::block_cost` returns is exactly `2^96` times this module's `blockReduce` at that
row: the Q96.96 scale of `aa = (a·2^32)²`, `two_a = 2a·2^32`, `q = Q·2^32`, `dot = D·2^32`. This is
the one fact the generated body does not carry — the slices it builds are `i64` arrays, not real
vectors — so it is a hypothesis of the interface, not a theorem here. -/
def WordCostCorresponds (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ)
    (aa two_a : Std.I128) (q dot : Slice Std.I64) : Prop :=
  ∀ j : Fin 256, ∃ c : Std.I128,
    apeiron_word.fold_port.fit.icm.block_cost aa two_a q.val[j.val]! dot.val[j.val]! = ok (core.result.Result.Ok c) ∧
    (c.val : ℝ) = 2 ^ 96 * blockReduce H x a j

/-- **The extracted argmin is the `blockReduce` argmin.** `best_code_q_argmin` gives the extracted
`i128` comparison; `WordCostCorresponds` and the positive scale `2^96` carry it to `ℝ`. The bound on
`i` and the witness index are the ones `best_code_q_argmin` returns. -/
theorem best_code_q_argmin_real (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ)
    (aa two_a : Std.I128) (q dot : Slice Std.I64)
    (hq : 256 ≤ q.val.length) (hd : 256 ≤ dot.val.length)
    (hcorr : WordCostCorresponds H x a aa two_a q dot) (i : Std.Usize)
    (hbest : apeiron_word.fold_port.fit.icm.best_code_q aa two_a q dot
      = ok (core.result.Result.Ok i)) :
    ∃ h : i.val < 256, ∀ j : Fin 256,
      blockReduce H x a ⟨i.val, h⟩ ≤ blockReduce H x a j := by
  obtain ⟨hb, c0, hc0, hall⟩ := best_code_q_argmin aa two_a q dot hq hd i hbest
  obtain ⟨c, hc, hcv⟩ := hcorr ⟨i.val, hb⟩
  have hcc : c = c0 := by
    rw [hc] at hc0
    simpa using hc0
  refine ⟨hb, fun j => ?_⟩
  obtain ⟨d, hdok, hdv⟩ := hcorr j
  obtain ⟨d', hd'ok, hle⟩ := hall j.val j.isLt
  have hdd : d = d' := by
    rw [hdok] at hd'ok
    simpa using hd'ok
  subst hdd
  have hcast : (c0.val : ℝ) ≤ (d.val : ℝ) := by exact_mod_cast hle
  have hi : (c0.val : ℝ) = 2 ^ 96 * blockReduce H x a ⟨i.val, hb⟩ := by
    rw [← hcc]; exact hcv
  have h96 : (0 : ℝ) < 2 ^ 96 := by norm_num
  have hAB : 2 ^ 96 * blockReduce H x a ⟨i.val, hb⟩ ≤ 2 ^ 96 * blockReduce H x a j := by
    rw [← hi, ← hdv]; exact hcast
  exact le_of_mul_le_mul_left hAB h96

/-- **The extracted `best_code_q` meets the block-decision specification.** Composing
`best_code_q_argmin_real` with `icm_step_spec_of_body_argmin`: the real-valued argmin hypothesis of the
latter is supplied by the former, so the only hypothesis left is the caller's slice correspondence
`hcorr`. The conclusion is `IcmStepSpec`, hence through `spec_quad_le` and `spec_sweep_le` the block-cost
and sweep-`J` non-increase. -/
theorem icm_step_spec_of_extracted_argmin (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ)
    (aa two_a : Std.I128) (q dot : Slice Std.I64)
    (hq : 256 ≤ q.val.length) (hd : 256 ≤ dot.val.length)
    (hcorr : WordCostCorresponds H x a aa two_a q dot) (i : Std.Usize)
    (hbest : apeiron_word.fold_port.fit.icm.best_code_q aa two_a q dot
      = ok (core.result.Result.Ok i)) :
    ∃ h : i.val < 256, IcmStepSpec H x a ⟨i.val, h⟩ := by
  obtain ⟨h, hargmin⟩ := best_code_q_argmin_real H x a aa two_a q dot hq hd hcorr i hbest
  exact ⟨h, (icm_step_spec_of_body_argmin H x a ⟨i.val, h⟩ hargmin).1⟩

/-- **The argmin is attained at the all-`+1` spinor row**, so the specification is inhabited on the
extracted codebook with a strictly negative block cost. -/
theorem icmStep_witness :
    IcmStepSpec diagBlock onesVec 1 (icmStep diagBlock onesVec 1)
      ∧ blockReduce diagBlock onesVec 1 (icmStep diagBlock onesVec 1) = -8 := by
  have h0 : blockReduce diagBlock onesVec 1 (0 : CodeIndex) = -8 := by
    rw [diagBlock_reduce]
    have h1 : ∑ p : Fin 8, codebook (0 : CodeIndex) p = 8 := by
      simp [codebook_zero]
    have h2 : ∑ p : Fin 8, codebook (0 : CodeIndex) p * codebook (0 : CodeIndex) p = 8 := by
      simp [codebook_zero]
    rw [h1, h2]; norm_num
  refine ⟨icmStep_spec _ _ _, ?_⟩
  have hle := (icmStep_spec diagBlock onesVec 1).argmin (0 : CodeIndex)
  have hge := diagBlock_reduce_ge (icmStep diagBlock onesVec 1)
  rw [h0] at hle
  linarith


/-! ## A concrete witness for the slice correspondence

`WordCostCorresponds` is a hypothesis of the interface above because the extracted body builds its own
`i64` slices and no theorem here can name them. This section fixes one concrete pair of slices built
from the codebook itself and proves the correspondence on the rows where it can hold.

The `q` coordinate of a row is the Q32.32 value of `Σ_p C_j(p)^2`. For the 128 spinor rows and the 112
`D₈` rows that sum is `8·2^32 = 2^35`. The 16 `O₈` rows have one nonzero coordinate `±√8`, i.e.
`±SQRT8/2^32` with `SQRT8 = 12148002000`, so their `Σ_p C_j(p)^2` is `SQRT8²/2^64` and the exact Q32.32
`q` would be `SQRT8²/2^32`, which is not an integer: `SQRT8` has 2-adic valuation 4, so `SQRT8²` has
valuation 8 and is not divisible by `2^32` (`not_dvd_sqrt8_sq` below, proved by `decide`). No `i64`
slice can therefore satisfy the row identity at an `O₈` row, and the correspondence is stated on the
240 exact rows (`exactRows`). -/

/-- `qmul` succeeds with the checked product. -/
theorem qmul_ok_of_checked {a b z : Std.I128} (h : I128.checked_mul a b = some z) :
    apeiron_word.fold_port.fit.icm.qmul a b = ok (core.result.Result.Ok z) := by
  unfold apeiron_word.fold_port.fit.icm.qmul
  simp only [lift, bind_tc_ok, h, core.option.Option.ok_or_some]

/-- `qsub` succeeds with the checked difference. -/
theorem qsub_ok_of_checked {a b z : Std.I128} (h : I128.checked_sub a b = some z) :
    apeiron_word.fold_port.fit.icm.qsub a b = ok (core.result.Result.Ok z) := by
  unfold apeiron_word.fold_port.fit.icm.qsub
  simp only [lift, bind_tc_ok, h, core.option.Option.ok_or_some]

/-- **`block_cost` is total on bounded inputs**, and its value is the exact Q96.96 form
`2^64·q − 2^65·dot` when `|q| ≤ 2^36` and `|dot| ≤ 2^37`: every intermediate product stays inside
`i128` (`2^64·2^36 = 2^100`, `2^33·(2^37·2^32) = 2^102`). -/
theorem block_cost_ok_bounded (aa two_a : Std.I128) (q dot : Std.I64)
    (haa : aa.val = 2 ^ 64) (htwo : two_a.val = 2 ^ 33)
    (hq : |q.val| ≤ 2 ^ 36) (hd : |dot.val| ≤ 2 ^ 37) :
    ∃ c : Std.I128,
      apeiron_word.fold_port.fit.icm.block_cost aa two_a q dot = ok (core.result.Result.Ok c) ∧
      c.val = 2 ^ 64 * q.val - 2 ^ 65 * dot.val := by
  have hI : Std.I128.min = -2 ^ 127 ∧ Std.I128.max = 2 ^ 127 - 1 := by
    constructor <;> simp [Std.I128.min_def, Std.I128.max_def, Std.I128.numBits_def]
  have hqv := abs_le.mp hq
  have hdv := abs_le.mp hd
  have hfq : (core.convert.num.FromI128I64.from q).val = q.val :=
    core.convert.num.FromI128I64.from_val_eq q
  have hfd : (core.convert.num.FromI128I64.from dot).val = dot.val :=
    core.convert.num.FromI128I64.from_val_eq dot
  obtain ⟨i2, hi2, hi2v⟩ := shl_i128_32 (core.convert.num.FromI128I64.from dot)
    (by rw [hfd]; omega) (by rw [hfd]; omega)
  obtain ⟨z0, hz0, hz0v⟩ := Learn.E8Code.checked_mul_some128 (a := aa)
    (b := core.convert.num.FromI128I64.from q) (by
      rw [haa, hfq, hI.1, hI.2]; constructor <;> nlinarith [hqv.1, hqv.2])
  obtain ⟨z1, hz1, hz1v⟩ := Learn.E8Code.checked_mul_some128 (a := two_a) (b := i2) (by
      rw [htwo, hi2v, hI.1, hI.2]; constructor <;> nlinarith [hdv.1, hdv.2])
  obtain ⟨z, hz, hzv⟩ := Learn.E8Code.checked_sub_some128 (a := z0) (b := z1) (by
      rw [hz0v, hz1v, hI.1, hI.2]; constructor <;> nlinarith [hqv.1, hqv.2, hdv.1, hdv.2])
  refine ⟨z, ?_, ?_⟩
  · unfold apeiron_word.fold_port.fit.icm.block_cost
    simp only [lift, bind_tc_ok]
    rw [qmul_ok_of_checked hz0]
    simp only [core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok]
    rw [hi2]
    simp only [bind_tc_ok]
    rw [qmul_ok_of_checked hz1]
    simp only [bind_tc_ok]
    rw [qsub_ok_of_checked hz]
  · rw [hzv, hz0v, hz1v, hi2v, hfq, hfd, haa, htwo]
    ring

/-- Every codeword coordinate is bounded by `SQRT8`: `±2^32` on the spinor rows, `±2·2^32` on the
`D₈` rows, `±SQRT8` on the `O₈` rows. -/
theorem codewordSpec_abs_le (i k : ℕ) :
    |Learn.E8Code.codewordSpec i k| ≤ Learn.E8Code.SQRT8 := by
  unfold Learn.E8Code.codewordSpec
  by_cases h1 : i < 128
  · rw [if_pos h1]
    unfold Learn.E8Code.spinorSpec Learn.E8Code.signAt
    split <;> simp [Learn.E8Code.ONE, Learn.E8Code.SQRT8]
  · rw [if_neg h1]
    by_cases h2 : i < 240
    · rw [if_pos h2]
      rcases rootSpec_val (i - 128) k with h | h | h <;>
        rw [h] <;> simp [Learn.E8Code.ONE, Learn.E8Code.SQRT8]
    · rw [if_neg h2]
      rcases o8Spec_val (i - 240) k with h | h | h <;>
        rw [h] <;> simp [Learn.E8Code.SQRT8]

/-- The row sum `Σ_p C_j(p)` is bounded by `8·SQRT8`. -/
theorem codewordSpec_sum_abs_le (j : Fin 256) :
    |∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val| ≤ 8 * Learn.E8Code.SQRT8 := by
  calc |∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val|
      ≤ ∑ p : Fin 8, |Learn.E8Code.codewordSpec j.val p.val| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _p : Fin 8, Learn.E8Code.SQRT8 :=
        Finset.sum_le_sum fun p _ => codewordSpec_abs_le _ _
    _ = 8 * Learn.E8Code.SQRT8 := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
        norm_num

/-- The `q` coordinate of the witness lies inside `i64` on every row. -/
theorem witnessQBound (j : Fin 256) :
    -(2 : ℤ) ^ (IScalarTy.I64.numBits - 1)
        ≤ (if j.val < 240 then (2 : ℤ) ^ 35 else Learn.E8Code.SQRT8 ^ 2 / 2 ^ 32)
      ∧ (if j.val < 240 then (2 : ℤ) ^ 35 else Learn.E8Code.SQRT8 ^ 2 / 2 ^ 32)
        < 2 ^ (IScalarTy.I64.numBits - 1) := by
  split <;> decide

/-- The `dot` coordinate of the witness lies inside `i64` on every row. -/
theorem witnessDotBound (j : Fin 256) :
    -(2 : ℤ) ^ (IScalarTy.I64.numBits - 1)
        ≤ ∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val
      ∧ (∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val)
        < 2 ^ (IScalarTy.I64.numBits - 1) := by
  have h := abs_le.mp (codewordSpec_sum_abs_le j)
  have hb : 8 * Learn.E8Code.SQRT8 < 2 ^ 63 := by decide
  have hbits : (2 : ℤ) ^ (IScalarTy.I64.numBits - 1) = 2 ^ 63 := by
    norm_num [Std.I64.numBits]
  constructor <;> omega

/-- `Σ_p C_j(p)^2 = 2^67` on the 128 spinor rows, by a bounded kernel reduction over `List.range`. -/
theorem codewordSpec_sq_sum_spinor :
    (List.range 128).all (fun j =>
      ((∑ p : Fin 8, Learn.E8Code.codewordSpec j p.val ^ 2) == 2 ^ 67)) = true := by decide

/-- `Σ_p C_j(p)^2 = 2^67` on the 112 `D₈` rows, by a bounded kernel reduction over `List.range`. -/
theorem codewordSpec_sq_sum_root :
    (List.range 112).all (fun j =>
      ((∑ p : Fin 8, Learn.E8Code.codewordSpec (j + 128) p.val ^ 2) == 2 ^ 67)) = true := by
  decide

/-- **`Σ_p C_j(p)^2 = 2^67` on the 240 exact rows** (`8` in Q32.32). -/
theorem codewordSpec_sq_sum (j : Fin 256) (hj : j.val < 240) :
    (∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val ^ 2) = 2 ^ 67 := by
  rcases Nat.lt_or_ge j.val 128 with h | h
  · have hh := List.all_eq_true.mp codewordSpec_sq_sum_spinor j.val (List.mem_range.mpr h)
    simpa using hh
  · have hh := List.all_eq_true.mp codewordSpec_sq_sum_root (j.val - 128)
      (List.mem_range.mpr (by omega))
    have hsub : j.val - 128 + 128 = j.val := by omega
    simpa [hsub] using hh

/-- The witness `aa = 2^64 = (1·2^32)^2` for `a = 1`. -/
def witnessAA : Std.I128 := I128.ofIntCore (2 ^ 64) (by decide)

/-- The witness `two_a = 2^33 = 2·1·2^32` for `a = 1`. -/
def witnessTwoA : Std.I128 := I128.ofIntCore (2 ^ 33) (by decide)

/-- `witnessAA` is `2^64`. -/
theorem witnessAA_val : witnessAA.val = 2 ^ 64 := I128.ofInt_val_eq _

/-- `witnessTwoA` is `2^33`. -/
theorem witnessTwoA_val : witnessTwoA.val = 2 ^ 33 := I128.ofInt_val_eq _

/-- The `q` slice of the witness: `2^35` on the 240 exact rows, the `>> 32` truncation of `SQRT8²` on
the 16 `O₈` rows outside them. -/
def witnessQList : List Std.I64 :=
  List.ofFn (fun j : Fin 256 =>
    I64.ofIntCore (if j.val < 240 then (2 : ℤ) ^ 35 else Learn.E8Code.SQRT8 ^ 2 / 2 ^ 32)
      (witnessQBound j))

/-- The `q` list has `256` entries, which fits a `usize` length. -/
theorem witnessQList_length : witnessQList.length ≤ Usize.max := by
  rw [witnessQList, List.length_ofFn]
  have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by simp [Usize.max, Usize.numBits]
  rw [hmax]
  rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> norm_num

/-- The witness `q` slice. -/
def witnessQ : Slice Std.I64 := Slice.from witnessQList witnessQList_length

/-- The `dot` slice of the witness: `Σ_p C_j(p)`, exact on every row including the `O₈` rows. -/
def witnessDotList : List Std.I64 :=
  List.ofFn (fun j : Fin 256 =>
    I64.ofIntCore (∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val) (witnessDotBound j))

/-- The `dot` list has `256` entries, which fits a `usize` length. -/
theorem witnessDotList_length : witnessDotList.length ≤ Usize.max := by
  rw [witnessDotList, List.length_ofFn]
  have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by simp [Usize.max, Usize.numBits]
  rw [hmax]
  rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> norm_num

/-- The witness `dot` slice. -/
def witnessDot : Slice Std.I64 := Slice.from witnessDotList witnessDotList_length

/-- The witness `q` slice has the list's value. -/
theorem witnessQ_val_eq : witnessQ.val = witnessQList := Slice.from_val _ _

/-- The witness `dot` slice has the list's value. -/
theorem witnessDot_val_eq : witnessDot.val = witnessDotList := Slice.from_val _ _

/-- The witness `q` slice has length `256`. -/
theorem witnessQ_val_length : witnessQ.val.length = 256 := by
  rw [witnessQ_val_eq]
  unfold witnessQList
  rw [List.length_ofFn]

/-- The witness `dot` slice has length `256`. -/
theorem witnessDot_val_length : witnessDot.val.length = 256 := by
  rw [witnessDot_val_eq]
  unfold witnessDotList
  rw [List.length_ofFn]

/-- The witness `q` list has length `256`. -/
theorem witnessQList_length_eq : witnessQList.length = 256 := by
  unfold witnessQList
  rw [List.length_ofFn]

/-- The witness `dot` list has length `256`. -/
theorem witnessDotList_length_eq : witnessDotList.length = 256 := by
  unfold witnessDotList
  rw [List.length_ofFn]

/-- **The `q` entry is `2^35` on every exact row.** -/
theorem witnessQ_val (j : Fin 256) (hj : j.val < 240) : (witnessQ.val[j.val]!).val = 2 ^ 35 := by
  rw [witnessQ_val_eq]
  unfold witnessQList
  rw [list_getElem!_eq (by rw [List.length_ofFn]; exact j.isLt)]
  rw [List.getElem_ofFn (by rw [List.length_ofFn]; exact j.isLt)]
  rw [I64.ofInt_val_eq _]
  rw [if_pos hj]

/-- **The `dot` entry is `Σ_p C_j(p)` on every row.** -/
theorem witnessDot_val (j : Fin 256) :
    (witnessDot.val[j.val]!).val = ∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val := by
  rw [witnessDot_val_eq]
  unfold witnessDotList
  rw [list_getElem!_eq (by rw [List.length_ofFn]; exact j.isLt)]
  rw [List.getElem_ofFn (by rw [List.length_ofFn]; exact j.isLt)]
  exact I64.ofInt_val_eq _

/-- **The row identity in `ℝ`.** If the `i64` pair `(q, dot)` carries `Q·2^32 = Σ_p C_j(p)^2` and
`dot = Σ_p C_j(p)`, then the extracted `block_cost` returns a value that is `2^96` times
`blockReduce` at row `j`: the Q96.96 scale of `aa = (a·2^32)^2`, `two_a = 2a·2^32`. -/
theorem wordCostCorresponds_row (j : Fin 256) (q dot : Std.I64)
    (hq : q.val * 2 ^ 32 = ∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val ^ 2)
    (hd : dot.val = ∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val)
    (hqb : |q.val| ≤ 2 ^ 35) (hdb : |dot.val| ≤ 2 ^ 37) :
    ∃ c : Std.I128,
      apeiron_word.fold_port.fit.icm.block_cost witnessAA witnessTwoA q dot
        = ok (core.result.Result.Ok c) ∧
      (c.val : ℝ) = 2 ^ 96 * blockReduce diagBlock onesVec 1 j := by
  obtain ⟨c, hc, hcv⟩ :=
    block_cost_ok_bounded witnessAA witnessTwoA q dot witnessAA_val witnessTwoA_val
      (by omega) hdb
  refine ⟨c, hc, ?_⟩
  have hqR : (q.val : ℝ) * 4294967296
      = ∑ p : Fin 8, (Learn.E8Code.codewordSpec j.val p.val : ℝ) ^ 2 := by
    have h := congrArg (fun z : ℤ => (z : ℝ)) hq
    push_cast at h
    exact h
  have hdR : (dot.val : ℝ) = ∑ p : Fin 8, (Learn.E8Code.codewordSpec j.val p.val : ℝ) := by
    have h := congrArg (fun z : ℤ => (z : ℝ)) hd
    push_cast at h
    exact h
  have hR : 2 ^ 96 * blockReduce diagBlock onesVec 1 j
      = 2 ^ 32 * (∑ p : Fin 8, (Learn.E8Code.codewordSpec j.val p.val : ℝ) ^ 2)
        - 2 ^ 65 * (∑ p : Fin 8, (Learn.E8Code.codewordSpec j.val p.val : ℝ)) := by
    rw [diagBlock_reduce]
    have hsq : (∑ p : Fin 8, codebook j p * codebook j p)
        = ∑ p : Fin 8, ((Learn.E8Code.codewordSpec j.val p.val : ℝ) ^ 2) / 2 ^ 64 :=
      Finset.sum_congr rfl fun p _ => by simp only [codebook]; ring
    have hlin : (∑ p : Fin 8, codebook j p)
        = ∑ p : Fin 8, ((Learn.E8Code.codewordSpec j.val p.val : ℝ)) / 2 ^ 32 :=
      Finset.sum_congr rfl fun p _ => by simp only [codebook]
    rw [hsq, hlin, ← Finset.sum_div, ← Finset.sum_div]
    field_simp
  have h1 : (18446744073709551616 : ℝ) * (q.val : ℝ)
      = 2 ^ 32 * (∑ p : Fin 8, (Learn.E8Code.codewordSpec j.val p.val : ℝ) ^ 2) := by
    calc (18446744073709551616 : ℝ) * (q.val : ℝ)
        = 2 ^ 32 * ((q.val : ℝ) * 4294967296) := by ring
      _ = 2 ^ 32 * (∑ p : Fin 8, (Learn.E8Code.codewordSpec j.val p.val : ℝ) ^ 2) := by rw [hqR]
  rw [hcv, hR]
  push_cast
  rw [h1, hdR]
  ring

/-- `2^32` does not divide `SQRT8²`: `SQRT8 = 16·759250125`, so `SQRT8²` has 2-adic valuation `8`. -/
theorem not_dvd_sqrt8_sq : ¬ (2 ^ 32 ∣ Learn.E8Code.SQRT8 ^ 2) := by
  have hmod : Learn.E8Code.SQRT8 ^ 2 % 2 ^ 32 ≠ 0 := by decide
  intro h
  exact hmod (Int.emod_eq_zero_of_dvd h)

/-- **No `i64` can carry an `O₈` row's `q` coordinate**: the exact Q32.32 value `SQRT8²/2^32` of
`Σ_p C_j(p)^2` is not an integer. -/
theorem o8_q_not_integral (q : ℤ) : q * 2 ^ 32 ≠ Learn.E8Code.SQRT8 ^ 2 := by
  intro h
  exact not_dvd_sqrt8_sq ⟨q, by rw [← h, mul_comm]⟩

/-- The 240 codebook rows whose codewords are exact Q32.32 integers: the 128 spinor rows and the 112
`D₈` roots. The 16 `O₈` rows are excluded because no `i64` `q` satisfies the row identity there
(`o8_q_not_integral`). -/
def exactRows : Finset (Fin 256) := Finset.univ.filter (fun j => j.val < 240)

/-- Membership in `exactRows` is `j.val < 240`. -/
theorem mem_exactRows (j : Fin 256) : j ∈ exactRows ↔ j.val < 240 := by
  simp [exactRows]

/-- Row-restricted form of `WordCostCorresponds`: the word cost of the extracted `block_cost` equals
`2^96` times `blockReduce` on the rows of `s`. -/
def WordCostCorrespondsOn (s : Finset (Fin 256)) (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ)
    (aa two_a : Std.I128) (q dot : Slice Std.I64) : Prop :=
  ∀ j ∈ s, ∃ c : Std.I128,
    apeiron_word.fold_port.fit.icm.block_cost aa two_a q.val[j.val]! dot.val[j.val]! =
      ok (core.result.Result.Ok c) ∧
    (c.val : ℝ) = 2 ^ 96 * blockReduce H x a j

/-- **The witness slices meet the correspondence on the 240 exact rows.** `q = 2^35` is the Q32.32
value of `Σ_p C_j(p)^2 = 8` on both the spinor and the `D₈` rows, and `dot = Σ_p C_j(p)`. -/
theorem witnessWordCostCorrespondsOn :
    WordCostCorrespondsOn exactRows diagBlock onesVec 1 witnessAA witnessTwoA witnessQ witnessDot := by
  intro j hj
  have hj240 : j.val < 240 := (mem_exactRows j).mp hj
  refine wordCostCorresponds_row j (witnessQ.val[j.val]!) (witnessDot.val[j.val]!) ?_ ?_ ?_ ?_
  · rw [witnessQ_val j hj240, codewordSpec_sq_sum j hj240]
    norm_num
  · exact witnessDot_val j
  · rw [witnessQ_val j hj240]
    norm_num
  · rw [witnessDot_val j]
    have h := abs_le.mp (codewordSpec_sum_abs_le j)
    have hb : 8 * Learn.E8Code.SQRT8 ≤ 2 ^ 37 := by decide
    rw [abs_le]
    constructor <;> omega

/-- **Restricted argmin transfer.** On a row set `s` where the word cost corresponds, the extracted
`best_code_q` index is a `blockReduce` argmin over `s`, provided the returned row itself lies in `s`. -/
theorem best_code_q_argmin_real_on (s : Finset (Fin 256)) (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ)
    (a : ℝ) (aa two_a : Std.I128) (q dot : Slice Std.I64)
    (hq : 256 ≤ q.val.length) (hd : 256 ≤ dot.val.length)
    (hcorr : WordCostCorrespondsOn s H x a aa two_a q dot) (i : Std.Usize)
    (hbest : apeiron_word.fold_port.fit.icm.best_code_q aa two_a q dot
      = ok (core.result.Result.Ok i))
    (hiS : ∀ (h : i.val < 256), (⟨i.val, h⟩ : Fin 256) ∈ s) :
    ∃ h : i.val < 256, ∀ j ∈ s, blockReduce H x a ⟨i.val, h⟩ ≤ blockReduce H x a j := by
  obtain ⟨hb, c0, hc0, hall⟩ := best_code_q_argmin aa two_a q dot hq hd i hbest
  obtain ⟨c, hc, hcv⟩ := hcorr ⟨i.val, hb⟩ (hiS hb)
  have hcc : c = c0 := by
    rw [hc] at hc0
    simpa using hc0
  refine ⟨hb, fun j hj => ?_⟩
  obtain ⟨d, hdok, hdv⟩ := hcorr j hj
  obtain ⟨d', hd'ok, hle⟩ := hall j.val j.isLt
  have hdd : d = d' := by
    rw [hdok] at hd'ok
    simpa using hd'ok
  subst hdd
  have hcast : (c0.val : ℝ) ≤ (d.val : ℝ) := by exact_mod_cast hle
  have hi : (c0.val : ℝ) = 2 ^ 96 * blockReduce H x a ⟨i.val, hb⟩ := by
    rw [← hcc]; exact hcv
  have h96 : (0 : ℝ) < 2 ^ 96 := by norm_num
  have hAB : 2 ^ 96 * blockReduce H x a ⟨i.val, hb⟩ ≤ 2 ^ 96 * blockReduce H x a j := by
    rw [← hi, ← hdv]; exact hcast
  exact le_of_mul_le_mul_left hAB h96

/-- **The witness argmin over the 240 exact rows**: whichever index the extracted `best_code_q`
returns below `240`, it is a `blockReduce` argmin over the exact rows. -/
theorem witnessArgmin (i : Std.Usize)
    (hbest : apeiron_word.fold_port.fit.icm.best_code_q witnessAA witnessTwoA witnessQ witnessDot
      = ok (core.result.Result.Ok i))
    (hi : i.val < 240) :
    ∃ h : i.val < 256, ∀ j : Fin 256, j.val < 240 →
      blockReduce diagBlock onesVec 1 ⟨i.val, h⟩ ≤ blockReduce diagBlock onesVec 1 j := by
  have hq : 256 ≤ witnessQ.val.length := by rw [witnessQ_val_length]
  have hd : 256 ≤ witnessDot.val.length := by rw [witnessDot_val_length]
  obtain ⟨h, harg⟩ :=
    best_code_q_argmin_real_on exactRows diagBlock onesVec 1 witnessAA witnessTwoA witnessQ witnessDot
      hq hd witnessWordCostCorrespondsOn i hbest (fun _ => (mem_exactRows _).mpr hi)
  exact ⟨h, fun j hj => harg j ((mem_exactRows j).mpr hj)⟩

/-- **The all-`+1` spinor row 0 costs `-8`**: `∑_p C_0(p)² = ∑_p C_0(p) = 8`. -/
theorem blockReduce_zero : blockReduce diagBlock onesVec 1 (0 : CodeIndex) = -8 := by
  rw [diagBlock_reduce]
  have h1 : ∑ p : Fin 8, codebook (0 : CodeIndex) p = 8 := by simp [codebook_zero]
  have h2 : ∑ p : Fin 8, codebook (0 : CodeIndex) p * codebook (0 : CodeIndex) p = 8 := by
    simp [codebook_zero]
  rw [h1, h2]; norm_num

/-- **The witness argmin is at most the all-`+1` spinor row's cost**, `−8`, without knowing the index. -/
theorem witnessArgmin_le (i : Std.Usize)
    (hbest : apeiron_word.fold_port.fit.icm.best_code_q witnessAA witnessTwoA witnessQ witnessDot
      = ok (core.result.Result.Ok i))
    (hi : i.val < 240) :
    ∃ h : i.val < 256, blockReduce diagBlock onesVec 1 ⟨i.val, h⟩ ≤ -8 := by
  obtain ⟨h, harg⟩ := witnessArgmin i hbest hi
  exact ⟨h, le_trans (harg 0 (by decide)) (le_of_eq blockReduce_zero)⟩



/-- The witness `q` entry on every row, including the 16 `O₈` rows. -/
theorem witnessQ_val_all (j : Fin 256) :
    (witnessQ.val[j.val]!).val
      = (if j.val < 240 then (2 : ℤ) ^ 35 else Learn.E8Code.SQRT8 ^ 2 / 2 ^ 32) := by
  rw [witnessQ_val_eq]
  unfold witnessQList
  rw [list_getElem!_eq (by rw [List.length_ofFn]; exact j.isLt)]
  rw [List.getElem_ofFn (by rw [List.length_ofFn]; exact j.isLt)]
  exact I64.ofInt_val_eq _

/-- The witness `q` entry is inside `2^36` on every row: `2^35` on the exact rows, and
`SQRT8²/2^32 < 2^36` on the `O₈` rows. -/
theorem witnessQ_abs_le (j : Fin 256) : |(witnessQ.val[j.val]!).val| ≤ 2 ^ 36 := by
  rw [witnessQ_val_all]
  split <;> decide

/-- The witness `dot` entry is inside `2^37` on every row. -/
theorem witnessDot_abs_le (j : Fin 256) : |(witnessDot.val[j.val]!).val| ≤ 2 ^ 37 := by
  rw [witnessDot_val]
  have h := abs_le.mp (codewordSpec_sum_abs_le j)
  have hb : 8 * Learn.E8Code.SQRT8 ≤ 2 ^ 37 := by decide
  rw [abs_le]
  constructor <;> omega

/-- **`block_cost` succeeds on every witness row**, including the `O₈` rows. -/
theorem witness_block_cost_ok (i : ℕ) (hi : i < 256) :
    ∃ c : Std.I128,
      apeiron_word.fold_port.fit.icm.block_cost witnessAA witnessTwoA
        witnessQ.val[i]! witnessDot.val[i]! = ok (core.result.Result.Ok c) := by
  obtain ⟨c, hc, -⟩ :=
    block_cost_ok_bounded witnessAA witnessTwoA witnessQ.val[i]! witnessDot.val[i]!
      witnessAA_val witnessTwoA_val (witnessQ_abs_le ⟨i, hi⟩) (witnessDot_abs_le ⟨i, hi⟩)
  exact ⟨c, hc⟩

/-- **The extracted argmin loop never records an error** when every entry's `block_cost` succeeds.
The loop carries the error slot `err : Option IcmError`; the only write to it is the `Err` branch of
`block_cost`, so `err = none` is an invariant (`loop.spec_decr_nat` with the measure
`256 - j.val`). -/
theorem best_code_q_loop_noerr (aa two_a : Std.I128) (q dot : Slice Std.I64)
    (hq : 256 ≤ q.val.length) (hd : 256 ≤ dot.val.length)
    (hok : ∀ i < 256, ∃ c : Std.I128,
      apeiron_word.fold_port.fit.icm.block_cost aa two_a q.val[i]! dot.val[i]!
        = ok (core.result.Result.Ok c)) :
    apeiron_word.fold_port.fit.icm.best_code_q_loop aa two_a q dot 0#u16 core.num.I128.MAX none
        0#usize ⦃ fun r : Std.U16 × Option apeiron_word.fold_port.fit.icm.IcmError =>
      r.2 = none ⦄ := by
  unfold apeiron_word.fold_port.fit.icm.best_code_q_loop
  apply loop.spec_decr_nat
    (measure := fun s : Std.U16 × Std.I128 ×
        Option apeiron_word.fold_port.fit.icm.IcmError × Std.Usize => 256 - s.2.2.2.val)
    (inv := fun s : Std.U16 × Std.I128 ×
        Option apeiron_word.fold_port.fit.icm.IcmError × Std.Usize => s.2.2.1 = none)
    (post := fun r : Std.U16 × Option apeiron_word.fold_port.fit.icm.IcmError => r.2 = none)
  · rintro ⟨b, bc, e, j'⟩ he
    dsimp only at he ⊢
    unfold apeiron_word.fold_port.fit.icm.best_code_q_loop.body
    have hcl : apeiron_word.fold_port.fit.icm.CODE_LEN = 256#usize := by
      unfold apeiron_word.fold_port.fit.icm.CODE_LEN apeiron_word.learn_port.e8.CODEBOOK_LEN
      rfl
    rw [hcl]
    by_cases hj : j' < 256#usize
    · rw [if_pos hj]
      have his : core.option.Option.is_none e = true := by rw [he]; rfl
      rw [if_pos his]
      step as ⟨x, hx⟩
      step as ⟨y, hy⟩
      have hjv : j'.val < 256 := by scalar_tac
      obtain ⟨c, hc⟩ := hok j'.val hjv
      have hxy : apeiron_word.fold_port.fit.icm.block_cost aa two_a x y
          = ok (core.result.Result.Ok c) := by
        rw [list_getElem!_eq (by omega : j'.val < q.val.length),
            list_getElem!_eq (by omega : j'.val < dot.val.length)] at hc
        rw [← hx, ← hy] at hc
        exact hc
      rw [hxy]
      simp only [bind_tc_ok]
      by_cases hlt : c < bc
      · rw [if_pos hlt]
        simp only [lift, bind_tc_ok]
        step as ⟨j1, hj1⟩
        refine ⟨he, by scalar_tac⟩
      · rw [if_neg hlt]
        simp only [bind_tc_ok]
        step as ⟨j1, hj1⟩
        refine ⟨he, by scalar_tac⟩
    · rw [if_neg hj]
      simp only [WP.spec_ok]
      exact he
  · rfl

/-- **The extracted `best_code_q` returns an index on the witness slices**: the run records no error,
so the result is `Ok i` for the running best index. -/
theorem witness_best_code_q_ok :
    ∃ i : Std.Usize,
      apeiron_word.fold_port.fit.icm.best_code_q witnessAA witnessTwoA witnessQ witnessDot
        = ok (core.result.Result.Ok i) := by
  have hq : 256 ≤ witnessQ.val.length := by rw [witnessQ_val_length]
  have hd : 256 ≤ witnessDot.val.length := by rw [witnessDot_val_length]
  obtain ⟨⟨b, e⟩, hrun, he⟩ :=
    WP.spec_imp_exists (best_code_q_loop_noerr witnessAA witnessTwoA witnessQ witnessDot hq hd
      (fun i hi => witness_block_cost_ok i hi))
  refine ⟨core.convert.num.FromUsizeU16.from b, ?_⟩
  unfold apeiron_word.fold_port.fit.icm.best_code_q
  have he' : e = none := he
  subst he'
  rw [hrun]
  simp only [lift, bind_tc_ok]
  rfl

/-- `Σ_p o8Spec(m, p)² = SQRT8²` on all 16 `O₈` vertices, by a bounded kernel reduction. -/
theorem o8_sq_sum_all :
    (List.range 16).all (fun m =>
      ((∑ p : Fin 8, Learn.E8Code.o8Spec m p.val ^ 2) == Learn.E8Code.SQRT8 ^ 2)) = true := by
  decide

/-- `-SQRT8 ≤ Σ_p o8Spec(m, p) ≤ SQRT8` on all 16 `O₈` vertices. -/
theorem o8_sum_bounds_all :
    (List.range 16).all (fun m =>
      ((-Learn.E8Code.SQRT8 ≤ ∑ p : Fin 8, Learn.E8Code.o8Spec m p.val) &&
        ((∑ p : Fin 8, Learn.E8Code.o8Spec m p.val) ≤ Learn.E8Code.SQRT8))) = true := by
  decide

/-- **`Σ_p C_j(p)² = SQRT8²` on the 16 `O₈` rows.** -/
theorem o8_sq_sum (j : Fin 256) (hj : 240 ≤ j.val) :
    (∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val ^ 2) = Learn.E8Code.SQRT8 ^ 2 := by
  have ho8 : ∀ k : Fin 8, Learn.E8Code.codewordSpec j.val k.val
      = Learn.E8Code.o8Spec (j.val - 240) k.val := fun k => codewordSpec_o8 hj
  simp only [ho8]
  have h := List.all_eq_true.mp o8_sq_sum_all (j.val - 240) (List.mem_range.mpr (by omega))
  simpa using h

/-- **`-SQRT8 ≤ Σ_p C_j(p) ≤ SQRT8` on the 16 `O₈` rows.** -/
theorem o8_sum_bounds (j : Fin 256) (hj : 240 ≤ j.val) :
    -(Learn.E8Code.SQRT8 : ℝ) ≤ (∑ p : Fin 8, (Learn.E8Code.codewordSpec j.val p.val : ℝ))
      ∧ (∑ p : Fin 8, (Learn.E8Code.codewordSpec j.val p.val : ℝ))
        ≤ (Learn.E8Code.SQRT8 : ℝ) := by
  have ho8 : ∀ k : Fin 8, Learn.E8Code.codewordSpec j.val k.val
      = Learn.E8Code.o8Spec (j.val - 240) k.val := fun k => codewordSpec_o8 hj
  have h := List.all_eq_true.mp o8_sum_bounds_all (j.val - 240) (List.mem_range.mpr (by omega))
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  have hsum : (∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val)
      = ∑ p : Fin 8, Learn.E8Code.o8Spec (j.val - 240) p.val :=
    Finset.sum_congr rfl fun p _ => ho8 p
  have hcast : ((∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val : ℤ) : ℝ)
      = ∑ p : Fin 8, (Learn.E8Code.codewordSpec j.val p.val : ℝ) := by
    push_cast
    rfl
  constructor
  · rw [← hcast, hsum]
    exact_mod_cast h.1
  · rw [← hcast, hsum]
    exact_mod_cast h.2

/-- **Every `O₈` row costs at least `0`** at `H = I`, `x = 1`, `a = 1`: the row is a single
`±√8` coordinate, so the cost is `(SQRT8² ∓ 2^33·SQRT8)/2^64 > 0`. -/
theorem diagBlock_reduce_o8_nonneg (j : Fin 256) (hj : 240 ≤ j.val) :
    0 ≤ blockReduce diagBlock onesVec 1 j := by
  obtain ⟨hb1, hb2⟩ := o8_sum_bounds j hj
  rw [diagBlock_reduce]
  have hsq : (∑ p : Fin 8, codebook j p * codebook j p)
      = (Learn.E8Code.SQRT8 : ℝ) ^ 2 / 2 ^ 64 := by
    have hstep : (∑ p : Fin 8, codebook j p * codebook j p)
        = ∑ p : Fin 8, ((Learn.E8Code.codewordSpec j.val p.val : ℝ) ^ 2) / 2 ^ 64 :=
      Finset.sum_congr rfl fun p _ => by simp only [codebook]; ring
    have hsumR : (∑ p : Fin 8, ((Learn.E8Code.codewordSpec j.val p.val : ℝ) ^ 2))
        = (Learn.E8Code.SQRT8 : ℝ) ^ 2 := by
      have h := congrArg (fun z : ℤ => (z : ℝ)) (o8_sq_sum j hj)
      push_cast at h
      exact h
    rw [hstep, ← Finset.sum_div, hsumR]
  have hlin : (∑ p : Fin 8, codebook j p)
      = (∑ p : Fin 8, (Learn.E8Code.codewordSpec j.val p.val : ℝ)) / 2 ^ 32 := by
    have h1 : (∑ p : Fin 8, codebook j p)
        = ∑ p : Fin 8, ((Learn.E8Code.codewordSpec j.val p.val : ℝ) / 2 ^ 32) :=
      Finset.sum_congr rfl fun p _ => by simp only [codebook]
    rw [h1, Finset.sum_div]
  rw [hsq, hlin]
  have hnum : (0 : ℝ) < (Learn.E8Code.SQRT8 : ℝ) ^ 2 - 2 ^ 33 * (Learn.E8Code.SQRT8 : ℝ) := by
    norm_num [Learn.E8Code.SQRT8]
  nlinarith [hb1, hb2, hnum]

/-- **`-SQRT8 ≤ Σ_p C_j(p) ≤ SQRT8` on the 16 `O₈` rows**, in `ℤ`. -/
theorem o8_sum_bounds_Z (j : Fin 256) (hj : 240 ≤ j.val) :
    -Learn.E8Code.SQRT8 ≤ (∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val)
      ∧ (∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val) ≤ Learn.E8Code.SQRT8 := by
  have h := List.all_eq_true.mp o8_sum_bounds_all (j.val - 240) (List.mem_range.mpr (by omega))
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  have hsum : (∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val)
      = ∑ p : Fin 8, Learn.E8Code.o8Spec (j.val - 240) p.val :=
    Finset.sum_congr rfl fun p _ => codewordSpec_o8 hj
  rw [hsum]
  exact h

/-- **The word cost at every `O₈` row is non-negative.** There `q = 2^35` and `dot = ∑_p C_j(p)` is
`±SQRT8`, so the extracted `c.val = 2^64·q − 2^65·dot ≥ 2^65·(2^34 − SQRT8) > 0`: the `O₈` rows can
never beat the `−2^96·8` cost of row 0, whatever the truncation does to `q`. -/
theorem o8_block_cost_nonneg (i : ℕ) (h240 : 240 ≤ i) (hi : i < 256) {c : Std.I128}
    (hc : apeiron_word.fold_port.fit.icm.block_cost witnessAA witnessTwoA
      witnessQ.val[i]! witnessDot.val[i]! = ok (core.result.Result.Ok c)) :
    0 ≤ c.val := by
  obtain ⟨c', hc', hcv⟩ :=
    block_cost_ok_bounded witnessAA witnessTwoA witnessQ.val[i]! witnessDot.val[i]!
      witnessAA_val witnessTwoA_val (witnessQ_abs_le ⟨i, hi⟩) (witnessDot_abs_le ⟨i, hi⟩)
  have hcc : c = c' := by rw [hc] at hc'; simpa using hc'
  subst hcc
  have hq : (witnessQ.val[i]!).val = (2 : ℤ) ^ 35 := by
    have hv := witnessQ_val_all (⟨i, hi⟩ : Fin 256)
    dsimp only at hv
    rw [hv, if_neg (by omega)]
    decide
  have hS : (∑ p : Fin 8, Learn.E8Code.codewordSpec i p.val) ≤ Learn.E8Code.SQRT8 := by
    have hs := (o8_sum_bounds_Z (⟨i, hi⟩ : Fin 256) h240).2
    simpa only [Fin.val_mk] using hs
  have hdv : (witnessDot.val[i]!).val = ∑ p : Fin 8, Learn.E8Code.codewordSpec i p.val := by
    have hv := witnessDot_val (⟨i, hi⟩ : Fin 256)
    simpa only [Fin.val_mk] using hv
  rw [hcv, hq, hdv]
  have hpos : (0 : ℤ) < 2 ^ 99 - 2 ^ 65 * Learn.E8Code.SQRT8 := by decide
  omega

/-- **The extracted argmin lies in the 240 exact rows.** If it lay in an `O₈` row its word cost would
be `≥ 0` (`o8_block_cost_nonneg`), while `best_code_q_argmin` bounds it by row 0's cost `−2^96·8`. -/
theorem witnessArgmin_index_lt (i : Std.Usize)
    (hbest : apeiron_word.fold_port.fit.icm.best_code_q witnessAA witnessTwoA witnessQ witnessDot
      = ok (core.result.Result.Ok i)) :
    i.val < 240 := by
  have hq : 256 ≤ witnessQ.val.length := by rw [witnessQ_val_length]
  have hd : 256 ≤ witnessDot.val.length := by rw [witnessDot_val_length]
  obtain ⟨hb256, c, hc, hall⟩ :=
    best_code_q_argmin witnessAA witnessTwoA witnessQ witnessDot hq hd i hbest
  obtain ⟨d, hdok, hle⟩ := hall 0 (by decide)
  obtain ⟨d0, hd0ok, hd0v⟩ := witnessWordCostCorrespondsOn 0 ((mem_exactRows 0).mpr (by decide))
  have hdok' : apeiron_word.fold_port.fit.icm.block_cost witnessAA witnessTwoA
      witnessQ.val[0]! witnessDot.val[0]! = ok (core.result.Result.Ok d) := by simpa using hdok
  have hd0ok' : apeiron_word.fold_port.fit.icm.block_cost witnessAA witnessTwoA
      witnessQ.val[0]! witnessDot.val[0]! = ok (core.result.Result.Ok d0) := by simpa using hd0ok
  have hdd : d = d0 := by rw [hdok'] at hd0ok'; simpa using hd0ok'
  subst hdd
  have hdneg : d.val < 0 := by
    have hcast : (d.val : ℝ) < 0 := by
      rw [hd0v, blockReduce_zero]
      norm_num
    exact_mod_cast hcast
  by_contra hcon
  have h240 : 240 ≤ i.val := by omega
  have hnn := o8_block_cost_nonneg i.val h240 hb256 hc
  omega

/-- **The full real argmin on the witness**: the extracted index minimises `blockReduce` over all 256
codebook rows at `H = I`, `x = 1`, `a = 1`, not only over the 240 exact rows. The `O₈` rows cost at
least `0` while the index's row costs at most `-8` (`diagBlock_reduce_o8_nonneg`, `blockReduce_zero`),
and the index cannot be an `O₈` row (`witnessArgmin_index_lt`). -/
theorem witnessArgmin_all (i : Std.Usize)
    (hbest : apeiron_word.fold_port.fit.icm.best_code_q witnessAA witnessTwoA witnessQ witnessDot
      = ok (core.result.Result.Ok i)) :
    ∃ h : i.val < 256, ∀ j : Fin 256,
      blockReduce diagBlock onesVec 1 ⟨i.val, h⟩ ≤ blockReduce diagBlock onesVec 1 j := by
  have hi : i.val < 240 := witnessArgmin_index_lt i hbest
  obtain ⟨h, harg⟩ := witnessArgmin i hbest hi
  refine ⟨h, fun j => ?_⟩
  rcases Nat.lt_or_ge j.val 240 with hlt | hge
  · exact harg j hlt
  · have hle : blockReduce diagBlock onesVec 1 ⟨i.val, h⟩ ≤ -8 := by
      have h0 := harg 0 (by decide)
      rwa [blockReduce_zero] at h0
    have h0 := diagBlock_reduce_o8_nonneg j hge
    linarith

/-- **The witness meets the block-decision specification**: the extracted argmin satisfies
`IcmStepSpec` at `H = I`, `x = 1`, `a = 1`, and its cost is the canonical argmin's cost. -/
theorem witnessStepSpec (i : Std.Usize)
    (hbest : apeiron_word.fold_port.fit.icm.best_code_q witnessAA witnessTwoA witnessQ witnessDot
      = ok (core.result.Result.Ok i)) :
    ∃ h : i.val < 256,
      IcmStepSpec diagBlock onesVec 1 ⟨i.val, h⟩
        ∧ blockReduce diagBlock onesVec 1 ⟨i.val, h⟩
            = blockReduce diagBlock onesVec 1 (icmStep diagBlock onesVec 1)
        ∧ ∀ c' : CodeIndex, blockQuad diagBlock onesVec 1 ⟨i.val, h⟩ ≤ blockQuad diagBlock onesVec 1 c' := by
  obtain ⟨h, harg⟩ := witnessArgmin_all i hbest
  obtain ⟨hspec, hcost⟩ := icm_step_spec_of_body_argmin diagBlock onesVec 1 ⟨i.val, h⟩ harg
  exact ⟨h, hspec, hcost, fun c' => spec_quad_le diagBlock diagBlock_symm onesVec 1 ⟨i.val, h⟩ hspec c'⟩

/-- **An unconditional `IcmStepSpec` witness**: the extracted `best_code_q` returns an index
(`witness_best_code_q_ok`), and that index meets the block-decision specification
(`witnessStepSpec`), with no hypothesis on the index. -/
theorem witnessStepSpec_exists :
    ∃ i : Std.Usize, ∃ h : i.val < 256, IcmStepSpec diagBlock onesVec 1 ⟨i.val, h⟩ := by
  obtain ⟨i, hbest⟩ := witness_best_code_q_ok
  obtain ⟨h, hspec, -⟩ := witnessStepSpec i hbest
  exact ⟨i, h, hspec⟩



/-- **The extracted witness decision carried into a sweep.** Over any finite set of blocks whose
residual is the witness block `onesVec`, with `H = I` and `a = 1`, the extracted argmin index has a
sweep total no larger than any constant code assignment's: each block's decision is a codebook
argmin, so `spec_sweep_le` applies block by block. -/
theorem witnessSweep_le {ι : Type} (s : Finset ι) (x : ι → Fin 8 → ℝ)
    (hx : ∀ i ∈ s, x i = onesVec) (i₀ : Std.Usize)
    (hbest : apeiron_word.fold_port.fit.icm.best_code_q witnessAA witnessTwoA witnessQ witnessDot
      = ok (core.result.Result.Ok i₀)) (c : CodeIndex) :
    ∃ h : i₀.val < 256,
      sweepTotal s (fun i => blockQuad diagBlock (x i) 1 ⟨i₀.val, h⟩)
        ≤ sweepTotal s (fun i => blockQuad diagBlock (x i) 1 c) := by
  obtain ⟨h, hspec, -⟩ := witnessStepSpec i₀ hbest
  refine ⟨h, ?_⟩
  have hper : ∀ i ∈ s, IcmStepSpec diagBlock (x i) 1 ⟨i₀.val, h⟩ := by
    intro i hi
    rw [hx i hi]
    exact hspec
  exact spec_sweep_le s (fun _ => diagBlock) (fun _ _ p q => diagBlock_symm p q) x (fun _ => 1)
    (fun _ => c) (fun _ => ⟨i₀.val, h⟩) hper

/-- **A strictly improving sweep.** When the comparison code's block cost is non-negative — every
`O₈` row is (`diagBlock_reduce_o8_nonneg`) — the extracted decision's sweep total is strictly below
it, because the witness row costs at most `-8`. This is the strict branch of `spec_sweep_lt`. -/
theorem witnessSweep_lt {ι : Type} (s : Finset ι) (x : ι → Fin 8 → ℝ)
    (hx : ∀ i ∈ s, x i = onesVec) (i₁ : ι) (hi₁ : i₁ ∈ s) (i₀ : Std.Usize)
    (hbest : apeiron_word.fold_port.fit.icm.best_code_q witnessAA witnessTwoA witnessQ witnessDot
      = ok (core.result.Result.Ok i₀)) (c : CodeIndex)
    (hc : 0 ≤ blockReduce diagBlock onesVec 1 c) :
    ∃ h : i₀.val < 256,
      sweepTotal s (fun i => blockQuad diagBlock (x i) 1 ⟨i₀.val, h⟩)
        < sweepTotal s (fun i => blockQuad diagBlock (x i) 1 c) := by
  obtain ⟨h, hspec, -⟩ := witnessStepSpec i₀ hbest
  have hi : i₀.val < 240 := witnessArgmin_index_lt i₀ hbest
  obtain ⟨h', hle8⟩ := witnessArgmin_le i₀ hbest hi
  have hfin : (⟨i₀.val, h'⟩ : Fin 256) = ⟨i₀.val, h⟩ := Fin.ext rfl
  have hlt8 : blockReduce diagBlock onesVec 1 ⟨i₀.val, h⟩ ≤ -8 := by
    rw [← hfin]; exact hle8
  refine ⟨h, ?_⟩
  have hper : ∀ i ∈ s, IcmStepSpec diagBlock (x i) 1 ⟨i₀.val, h⟩ := by
    intro i hi2
    rw [hx i hi2]
    exact hspec
  refine spec_sweep_lt s (fun _ => diagBlock) (fun _ _ p q => diagBlock_symm p q) x (fun _ => 1)
    (fun _ => c) (fun _ => ⟨i₀.val, h⟩) hper i₁ hi₁ ?_
  rw [hx i₁ hi₁,
    blockQuad_eq_blockReduce_add diagBlock diagBlock_symm onesVec 1 ⟨i₀.val, h⟩,
    blockQuad_eq_blockReduce_add diagBlock diagBlock_symm onesVec 1 c]
  linarith

/-- **The strict sweep at a concrete `O₈` comparison code** (`240`, cost `≥ 0`). -/
theorem witnessSweep_lt_o8 {ι : Type} (s : Finset ι) (x : ι → Fin 8 → ℝ)
    (hx : ∀ i ∈ s, x i = onesVec) (i₁ : ι) (hi₁ : i₁ ∈ s) (i₀ : Std.Usize)
    (hbest : apeiron_word.fold_port.fit.icm.best_code_q witnessAA witnessTwoA witnessQ witnessDot
      = ok (core.result.Result.Ok i₀)) :
    ∃ h : i₀.val < 256,
      sweepTotal s (fun i => blockQuad diagBlock (x i) 1 ⟨i₀.val, h⟩)
        < sweepTotal s (fun i => blockQuad diagBlock (x i) 1 (240 : CodeIndex)) :=
  witnessSweep_lt s x hx i₁ hi₁ i₀ hbest 240
    (diagBlock_reduce_o8_nonneg (240 : CodeIndex) (by decide))



/-- The residual after deciding block code `c` at scale `a`: `x − a·C_c`, the vector the next pass
sees. `fit::icm::e8_icm_step`'s caller forms this; the model had no such update. -/
def residualStep (a : ℝ) (c : CodeIndex) (x : Fin 8 → ℝ) : Fin 8 → ℝ :=
  fun p => x p - a * codebook c p

/-- **Two residual steps add their codes**: `x − a·C_c − a·C_c' = x − a·(C_c + C_c')`. -/
theorem residualStep_comp (a : ℝ) (c c' : CodeIndex) (x : Fin 8 → ℝ) :
    residualStep a c' (residualStep a c x)
      = fun p => x p - a * (codebook c p + codebook c' p) := by
  funext p
  simp only [residualStep]
  ring

/-- **The block cost is the quadratic form of the residual it leaves**: `blockQuad H x a c` is
`Σ_p Σ_q H_pq r_p r_q` at `r = residualStep a c x`. This is the identity that makes the code
decision and the residual update the same object. -/
theorem blockQuad_residual_energy (H : Fin 8 → Fin 8 → ℝ) (x : Fin 8 → ℝ) (a : ℝ)
    (c : CodeIndex) :
    blockQuad H x a c
      = ∑ p, ∑ q, H p q * residualStep a c x p * residualStep a c x q := by
  refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
  simp only [residualStep]
  ring

/-- At `H = I` the block cost is the sum of squares of the residual coordinates. -/
theorem blockQuad_diagBlock (x : Fin 8 → ℝ) (a : ℝ) (c : CodeIndex) :
    blockQuad diagBlock x a c = ∑ p, (a * codebook c p - x p) ^ 2 := by
  unfold blockQuad
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [Finset.sum_eq_single p]
  · simp [diagBlock]
    ring
  · intro q _ hq
    simp [diagBlock, Ne.symm hq]
  · intro hp
    exact absurd (Finset.mem_univ p) hp

/-- **The exact one-pass-to-two-pass identity** at `H = I`, `a = 1`: the second pass's cost is the
first pass's cost plus `Σ_p C_{c'}² − 2·Σ_p r_p·C_{c'}(p)`, where `r = residualStep 1 c x`. -/
theorem twoPass_quad (x : Fin 8 → ℝ) (c c' : CodeIndex) :
    blockQuad diagBlock (residualStep 1 c x) 1 c'
      = blockQuad diagBlock x 1 c
        - 2 * (∑ p, residualStep 1 c x p * codebook c' p)
        + ∑ p, codebook c' p ^ 2 := by
  rw [blockQuad_diagBlock, blockQuad_diagBlock]
  have hpt : ∀ p : Fin 8,
      (1 * codebook c' p - residualStep 1 c x p) ^ 2
        = (1 * codebook c p - x p) ^ 2
          - 2 * (residualStep 1 c x p * codebook c' p) + codebook c' p ^ 2 := by
    intro p
    simp only [residualStep]
    ring
  calc ∑ p : Fin 8, (1 * codebook c' p - residualStep 1 c x p) ^ 2
      = ∑ p : Fin 8, ((1 * codebook c p - x p) ^ 2
          - 2 * (residualStep 1 c x p * codebook c' p) + codebook c' p ^ 2) :=
        Finset.sum_congr rfl fun p _ => hpt p
    _ = ∑ p : Fin 8, (1 * codebook c p - x p) ^ 2
          - 2 * (∑ p : Fin 8, residualStep 1 c x p * codebook c' p)
          + ∑ p : Fin 8, codebook c' p ^ 2 := by
        rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.mul_sum]

/-- **When the second pass descends, exactly**: `blockQuad diagBlock r 1 c' ≤ blockQuad diagBlock x 1 c`
holds iff `Σ_p C_{c'}² ≤ 2·Σ_p r_p·C_{c'}(p)`, i.e. iff the code's squared norm is dominated by
twice its correlation with the residual `r = residualStep 1 c x`. -/
theorem twoPass_le_iff (x : Fin 8 → ℝ) (c c' : CodeIndex) :
    blockQuad diagBlock (residualStep 1 c x) 1 c' ≤ blockQuad diagBlock x 1 c
      ↔ (∑ p, codebook c' p ^ 2) ≤ 2 * (∑ p, residualStep 1 c x p * codebook c' p) := by
  rw [twoPass_quad]
  constructor <;> intro h <;> linarith

/-- **The proposed two-pass descent is false.** At `x = onesVec`, `c = c' = 0` (`C_0 = onesVec`) the
first pass costs `0` and the second costs `32`: the residual update subtracts `C_0` again, so the
composite residual is `−1` and `Σ_p (1 − (−1))² = 32`. `witness_onePass_zero` shows the same
exhaustion happens at the extracted optimum, so the failure is not an artefact of `c = 0`. -/
theorem twoPass_not_le :
    ¬ (blockQuad diagBlock
          (residualStep 1 (0 : CodeIndex) (residualStep 1 (0 : CodeIndex) onesVec))
          1 (0 : CodeIndex)
        ≤ blockQuad diagBlock onesVec 1 (0 : CodeIndex)) := by
  have h1 : residualStep 1 (0 : CodeIndex) (residualStep 1 (0 : CodeIndex) onesVec)
      = fun _ => (-1 : ℝ) := by
    funext p
    simp [residualStep, onesVec, codebook_zero]
  have h2 : blockQuad diagBlock
        (residualStep 1 (0 : CodeIndex) (residualStep 1 (0 : CodeIndex) onesVec))
        1 (0 : CodeIndex) = 32 := by
    rw [h1, blockQuad_diagBlock]
    simp [codebook_zero]
    norm_num
  have h3 : blockQuad diagBlock onesVec 1 (0 : CodeIndex) = 0 := by
    rw [blockQuad_diagBlock]
    simp [codebook_zero, onesVec]
  rw [h2, h3]
  norm_num

/-- **At the extracted optimum the first pass exhausts the block**: the witness row's `blockQuad` is
`0`, so any further pass on that residual adds energy (`twoPass_quad`). -/
theorem witness_onePass_zero (i₀ : Std.Usize)
    (hbest : apeiron_word.fold_port.fit.icm.best_code_q witnessAA witnessTwoA witnessQ witnessDot
      = ok (core.result.Result.Ok i₀)) :
    ∃ h : i₀.val < 256, blockQuad diagBlock onesVec 1 ⟨i₀.val, h⟩ = 0 := by
  obtain ⟨h, hspec, -⟩ := witnessStepSpec i₀ hbest
  refine ⟨h, ?_⟩
  have hi : i₀.val < 240 := witnessArgmin_index_lt i₀ hbest
  obtain ⟨h', hle8⟩ := witnessArgmin_le i₀ hbest hi
  have hfin : (⟨i₀.val, h'⟩ : Fin 256) = ⟨i₀.val, h⟩ := Fin.ext rfl
  have hle : blockReduce diagBlock onesVec 1 ⟨i₀.val, h⟩ ≤ -8 := by
    rw [← hfin]; exact hle8
  have hq : blockQuad diagBlock onesVec 1 ⟨i₀.val, h⟩ ≤ 0 := by
    rw [blockQuad_eq_blockReduce_add diagBlock diagBlock_symm onesVec 1 ⟨i₀.val, h⟩]
    have hxx : ∑ p : Fin 8, ∑ q : Fin 8, diagBlock p q * onesVec p * onesVec q = 8 := by
      simp [diagBlock, onesVec]
    rw [hxx]
    linarith
  have hnn : 0 ≤ blockQuad diagBlock onesVec 1 ⟨i₀.val, h⟩ := by
    rw [blockQuad_diagBlock]
    exact Finset.sum_nonneg fun p _ => sq_nonneg _
  linarith



/-! ## The shipped state update (`fit::icm::sweep_rows`, `joint_band`)

`e8_icm_step` keeps `s_cur` (the codes) and `f_res` (the residual gradient). A block decision sets
`s_cur[b0+k] = cb[best][k]` and then applies `f_res -= a·(H·Δ)` with `Δ_k = cb[best][k] − s_old[k]`.
The three objects below name that state: `stateResid` is the residual `x − a·s`, `blockGrad` is the
gradient `H·(x − a·s)` the sweep carries in `f_res`, and `stateQuad` is the row objective
`Σ_p Σ_q H_pq r_p r_q` at an arbitrary state `s`. -/

/-- The block residual of the state `s` at group scale `a`: `x − a·s`. -/
def stateResid (a : ℝ) (s x : Fin 8 → ℝ) : Fin 8 → ℝ := fun p => x p - a * s p

/-- The residual gradient the sweep carries in `f_res`: `f_p = Σ_q H_pq·(x_q − a·s_q)`. -/
def blockGrad (H : Fin 8 → Fin 8 → ℝ) (a : ℝ) (s x : Fin 8 → ℝ) : Fin 8 → ℝ :=
  fun p => ∑ q, H p q * stateResid a s x q

/-- The block's contribution to the row objective at an arbitrary state `s`:
`Σ_p Σ_q H_pq (a·s_p − x_p)(a·s_q − x_q)`. At `s = codebook c` this is `blockQuad`. -/
def stateQuad (H : Fin 8 → Fin 8 → ℝ) (u : Fin 8 → ℝ) (a : ℝ) (s : Fin 8 → ℝ) : ℝ :=
  ∑ p, ∑ q, H p q * (a * s p - u p) * (a * s q - u q)

/-- The code-dependent part of `stateQuad` at an arbitrary state: `a²·sᵀHs − 2a·sᵀ(H u)`. At
`s = codebook c` this is `blockReduce`, the quantity `fit::icm::block_cost` computes in Q96.96. -/
def stateReduce (H : Fin 8 → Fin 8 → ℝ) (u : Fin 8 → ℝ) (a : ℝ) (s : Fin 8 → ℝ) : ℝ :=
  a ^ 2 * (∑ p, ∑ q, H p q * s p * s q) - 2 * a * (∑ p, ∑ q, H p q * s p * u q)

/-- The state objective is the quadratic form of the residual: `Σ_p Σ_q H_pq r_p r_q` at
`r = stateResid a s u`. -/
theorem stateQuad_eq_resid_form (H : Fin 8 → Fin 8 → ℝ) (u : Fin 8 → ℝ) (a : ℝ)
    (s : Fin 8 → ℝ) :
    stateQuad H u a s = ∑ p, ∑ q, H p q * stateResid a s u p * stateResid a s u q := by
  refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
  unfold stateResid
  ring

/-- At a codebook row the state objective is the model's block cost. -/
theorem stateQuad_codebook (H : Fin 8 → Fin 8 → ℝ) (u : Fin 8 → ℝ) (a : ℝ) (c : CodeIndex) :
    stateQuad H u a (codebook c) = blockQuad H u a c := rfl

/-- At a codebook row the state reduction is the model's `blockReduce`. -/
theorem stateReduce_codebook (H : Fin 8 → Fin 8 → ℝ) (u : Fin 8 → ℝ) (a : ℝ)
    (c : CodeIndex) :
    stateReduce H u a (codebook c) = blockReduce H u a c := rfl

/-- The state objective splits into the code-dependent part and the state-independent `uᵀHu`. -/
theorem stateQuad_eq_stateReduce_add (H : Fin 8 → Fin 8 → ℝ) (hH : ∀ p q, H p q = H q p)
    (u : Fin 8 → ℝ) (a : ℝ) (s : Fin 8 → ℝ) :
    stateQuad H u a s = stateReduce H u a s + ∑ p, ∑ q, H p q * u p * u q := by
  have h := joint_block_cost_expand H hH s (fun _ => 0) u a 0
  simpa [stateQuad, stateReduce, zero_mul, mul_zero, zero_add, add_zero] using h

/-- **The state objective's change is exactly the argmin's cost change**: the `uᵀHu` constant
cancels, so moving the state from `s'` to `s` changes the objective by the same amount the block
cost changes. -/
theorem stateQuad_delta (H : Fin 8 → Fin 8 → ℝ) (hH : ∀ p q, H p q = H q p)
    (u : Fin 8 → ℝ) (a : ℝ) (s s' : Fin 8 → ℝ) :
    stateQuad H u a s - stateQuad H u a s' = stateReduce H u a s - stateReduce H u a s' := by
  rw [stateQuad_eq_stateReduce_add H hH u a s, stateQuad_eq_stateReduce_add H hH u a s']
  ring

/-- One residual step at the state: `x − a·(s + δ) = (x − a·s) − a·δ`. -/
theorem stateResid_update (a : ℝ) (s δ x : Fin 8 → ℝ) :
    stateResid a (s + δ) x = fun p => stateResid a s x p - a * δ p := by
  funext p
  unfold stateResid
  simp only [Pi.add_apply]
  ring

/-- **The invariant update**: the metric image of the residual after the state moves by `δ` is the
old image minus `a·H·δ`. This is the arithmetic of `f_res -= a·(H·Δ)`. -/
theorem lin_update (H : Fin 8 → Fin 8 → ℝ) (a : ℝ) (s δ x : Fin 8 → ℝ) :
    ∀ p, (∑ q, H p q * stateResid a (s + δ) x q)
      = (∑ q, H p q * stateResid a s x q) - a * (∑ q, H p q * δ q) := by
  intro p
  rw [stateResid_update, Finset.mul_sum, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun q _ => ?_
  ring

/-- **The gradient invariant is preserved by the shipped update**: `f` at the new state equals `f`
at the old state minus `a·(H·δ)`. -/
theorem blockGrad_update (H : Fin 8 → Fin 8 → ℝ) (a : ℝ) (s δ x : Fin 8 → ℝ) :
    blockGrad H a (s + δ) x
      = fun p => blockGrad H a s x p - a * (∑ q, H p q * δ q) :=
  funext fun p => lin_update H a s δ x p

/-- The concrete form used by `sweep_rows`: with `δ = C_{c'} − C_c` the update reproduces the
gradient at the new code exactly. -/
theorem blockGrad_update_code (H : Fin 8 → Fin 8 → ℝ) (a : ℝ) (x : Fin 8 → ℝ)
    (c c' : CodeIndex) :
    ∀ p, blockGrad H a (codebook c) x p
          - a * (∑ q, H p q * (codebook c' q - codebook c q))
        = blockGrad H a (codebook c') x p := by
  intro p
  have h := lin_update H a (codebook c) (fun q => codebook c' q - codebook c q) x p
  have hfun : (codebook c + fun q => codebook c' q - codebook c q) = codebook c' := by
    funext q
    simp only [Pi.add_apply]
    ring
  rw [hfun] at h
  exact h.symm

/-- **The joint two-plane update** (`joint_band`): the plane-1 gradient is
`H₁₁·r₁ + H₁₂·r₂`, and after both planes move it reads
`f₁ -= a₁·(H₁₁Δ₁) + a₂·(H₁₂Δ₂)`. -/
theorem jointGrad_update_left (H11 H12 : Fin 8 → Fin 8 → ℝ) (a₁ a₂ : ℝ)
    (s₁ s₂ δ₁ δ₂ x₁ x₂ : Fin 8 → ℝ) :
    ∀ p, blockGrad H11 a₁ (s₁ + δ₁) x₁ p
          + (∑ q, H12 p q * stateResid a₂ (s₂ + δ₂) x₂ q)
        = (blockGrad H11 a₁ s₁ x₁ p + ∑ q, H12 p q * stateResid a₂ s₂ x₂ q)
          - a₁ * (∑ q, H11 p q * δ₁ q) - a₂ * (∑ q, H12 p q * δ₂ q) := by
  intro p
  have h1 := congrFun (blockGrad_update H11 a₁ s₁ δ₁ x₁) p
  have h2 := lin_update H12 a₂ s₂ δ₂ x₂ p
  rw [h1, h2]
  ring

/-- **One block decision never increases the state objective**: the old code is an admissible
candidate, so the argmin's block cost is at most the old code's. -/
theorem state_step_descends (H : Fin 8 → Fin 8 → ℝ) (hH : ∀ p q, H p q = H q p)
    (u : Fin 8 → ℝ) (a : ℝ) (c_old c_new : CodeIndex) (h : IcmStepSpec H u a c_new) :
    stateQuad H u a (codebook c_new) ≤ stateQuad H u a (codebook c_old) := by
  rw [stateQuad_eq_stateReduce_add H hH u a (codebook c_new),
    stateQuad_eq_stateReduce_add H hH u a (codebook c_old)]
  have hb : stateReduce H u a (codebook c_new) ≤ stateReduce H u a (codebook c_old) :=
    h.argmin c_old
  linarith

/-- **A sweep of decisions never increases the summed state objective** (the `sweepTotal` form,
with the effective data `u` fixed across the sweep). -/
theorem sweep_state_descent {ι : Type} (s : Finset ι) (H : ι → Fin 8 → Fin 8 → ℝ)
    (hH : ∀ i ∈ s, ∀ p q, H i p q = H i q p) (u : ι → Fin 8 → ℝ) (a : ι → ℝ)
    (c_old c_new : ι → CodeIndex) (h : ∀ i ∈ s, IcmStepSpec (H i) (u i) (a i) (c_new i)) :
    sweepTotal s (fun i => stateQuad (H i) (u i) (a i) (codebook (c_new i)))
      ≤ sweepTotal s (fun i => stateQuad (H i) (u i) (a i) (codebook (c_old i))) :=
  sweep_step_le s
    (fun i => stateQuad (H i) (u i) (a i) (codebook (c_old i)))
    (fun i => stateQuad (H i) (u i) (a i) (codebook (c_new i)))
    (fun i hi => state_step_descends (H i) (hH i hi) (u i) (a i) (c_old i) (c_new i) (h i hi))

/-- **A sweep with one strictly improving block strictly lowers the summed state objective.** -/
theorem sweep_state_descent_lt {ι : Type} (s : Finset ι) (H : ι → Fin 8 → Fin 8 → ℝ)
    (hH : ∀ i ∈ s, ∀ p q, H i p q = H i q p) (u : ι → Fin 8 → ℝ) (a : ι → ℝ)
    (c_old c_new : ι → CodeIndex) (h : ∀ i ∈ s, IcmStepSpec (H i) (u i) (a i) (c_new i))
    (i₀ : ι) (hi₀ : i₀ ∈ s)
    (hlt : stateQuad (H i₀) (u i₀) (a i₀) (codebook (c_new i₀))
      < stateQuad (H i₀) (u i₀) (a i₀) (codebook (c_old i₀))) :
    sweepTotal s (fun i => stateQuad (H i) (u i) (a i) (codebook (c_new i)))
      < sweepTotal s (fun i => stateQuad (H i) (u i) (a i) (codebook (c_old i))) :=
  sweep_step_lt s
    (fun i => stateQuad (H i) (u i) (a i) (codebook (c_old i)))
    (fun i => stateQuad (H i) (u i) (a i) (codebook (c_new i)))
    (fun i hi => state_step_descends (H i) (hH i hi) (u i) (a i) (c_old i) (c_new i) (h i hi))
    i₀ hi₀ hlt

/-- **The model's block cost is the word cost at the sweep's effective data.** With `u` the data
whose metric image is the sweep's `f_blk` (`H u = f_blk`, `f_blk = H x_m + f_out` in
`icm.rs::sweep_rows`), `blockReduce` is exactly `a²·q_c − 2a·dot_c` with `q_c = C_cᵀHC_c` and
`dot_c = C_cᵀ f_blk`. -/
theorem blockReduce_eq_word_cost (H : Fin 8 → Fin 8 → ℝ) (f u : Fin 8 → ℝ) (a : ℝ)
    (c : CodeIndex) (hu : ∀ p, ∑ q, H p q * u q = f p) :
    blockReduce H u a c
      = a ^ 2 * (∑ p, ∑ q, H p q * codebook c p * codebook c q)
        - 2 * a * (∑ p, codebook c p * f p) := by
  rw [blockReduce_eq_word]
  have hlin : ∑ p, codebook c p * (∑ q, H p q * u q) = ∑ p, codebook c p * f p :=
    Finset.sum_congr rfl fun p _ => by rw [hu p]
  rw [hlin]

/-- **The word-path descent bound.** If the Q32.32 `q`/`dot` differ from their exact values by at
most `εq`/`εd`, a minimiser of the word cost `aa·q̂ − two_a·d̂` has exact cost at most the old
code's plus `2·(|aa|εq + |two_a|εd)`. The exact non-increase is the `εq = εd = 0` case. The
end-to-end `εq`/`εd` are `Learn.Limb.q_all_end2end` and `Learn.Limb.dots_end2end`, instantiated by
`word_argmin_cost_bound_end2end` below; `dots`' `f_blk` state error remains a hypothesis. -/
theorem word_argmin_cost_bound (aa two_a : ℝ) (q d qh dh : CodeIndex → ℝ) (εq εd : ℝ)
    (hq : ∀ c, |qh c - q c| ≤ εq) (hd : ∀ c, |dh c - d c| ≤ εd)
    (cstar cold : CodeIndex)
    (hmin : ∀ c, aa * qh cstar - two_a * dh cstar ≤ aa * qh c - two_a * dh c) :
    aa * q cstar - two_a * d cstar
      ≤ aa * q cold - two_a * d cold + 2 * (|aa| * εq + |two_a| * εd) := by
  set ε : ℝ := |aa| * εq + |two_a| * εd with hεdef
  have hε : ∀ c, |(aa * qh c - two_a * dh c) - (aa * q c - two_a * d c)| ≤ ε := by
    intro c
    have hsplit : aa * qh c - two_a * dh c - (aa * q c - two_a * d c)
        = aa * (qh c - q c) - two_a * (dh c - d c) := by ring
    rw [hsplit]
    calc |aa * (qh c - q c) - two_a * (dh c - d c)|
        ≤ |aa * (qh c - q c)| + |two_a * (dh c - d c)| :=
          abs_sub (aa * (qh c - q c)) (two_a * (dh c - d c))
      _ = |aa| * |qh c - q c| + |two_a| * |dh c - d c| := by rw [abs_mul, abs_mul]
      _ ≤ |aa| * εq + |two_a| * εd :=
          add_le_add (mul_le_mul_of_nonneg_left (hq c) (abs_nonneg _))
            (mul_le_mul_of_nonneg_left (hd c) (abs_nonneg _))
  have h1 : aa * q cstar - two_a * d cstar
      ≤ aa * qh cstar - two_a * dh cstar + ε := by
    have := (abs_le.mp (hε cstar)).1
    linarith
  have h2 : aa * qh cold - two_a * dh cold
      ≤ aa * q cold - two_a * d cold + ε := by
    have := (abs_le.mp (hε cold)).2
    linarith
  have h3 := hmin cold
  linarith

/-- The `Down`-peak kernel bounds, as one exact real: `2·(1·εq + 2·εd) = 102492/2^32`. -/
theorem word_argmin_cost_bound_down_const :
    2 * ((1 : ℝ) * (1562 / 2 ^ 32) + 2 * (24842 / 2 ^ 32)) = 102492 / 2 ^ 32 := by
  norm_num

/-- **The `Down` peaks, instantiated.** `εq = 1562/2^32`, `εd = 24842/2^32` (Q32.32 words) are the
conservative kernel bounds at the measured `Down` finite-difference peaks: `εq` at 12 words, `εd` at 182
words, where the measured argmin margin is `1.75e-7`. The theorem carries the conservative kernel bound,
not the measurement: `2·(1·εq + 2·εd) = 102492/2^32 ≈ 2.386e-5`. `aa = 1`, `two_a = 2` are the shipped
block-cost coefficients (`block_cost aa two_a q dot = a²q − 2·(2a)·dot` at `a = 1`).

**Per-call, not end-to-end.** `1562` bounds one `nt_q` call and `24842` bounds one call on an operand
at `(5.30e11, 1.21e10)`; the shipped `q` is `block_quads_q`'s two calls, and `dots` runs on an
`f_blk` carrying the `s·H` call's error. Use `word_argmin_cost_bound_end2end`. -/
theorem word_argmin_cost_bound_down (q d qh dh : CodeIndex → ℝ)
    (hq : ∀ c, |qh c - q c| ≤ 1562 / 2 ^ 32)
    (hd : ∀ c, |dh c - d c| ≤ 24842 / 2 ^ 32)
    (cstar cold : CodeIndex)
    (hmin : ∀ c, qh cstar - 2 * dh cstar ≤ qh c - 2 * dh c) :
    q cstar - 2 * d cstar
      ≤ q cold - 2 * d cold + 2 * ((1 : ℝ) * (1562 / 2 ^ 32) + 2 * (24842 / 2 ^ 32)) := by
  have h := word_argmin_cost_bound (1 : ℝ) 2 q d qh dh (1562 / 2 ^ 32) (24842 / 2 ^ 32) hq hd
    cstar cold (fun c => by simpa using hmin c)
  simpa using h

/-- **The end-to-end word-path bound.** `εq` is `Learn.Limb.q_all_end2end`'s constant: the second
`nt_q` call at its own peak (99,338 words) plus the first call's 3,114-word error through the codebook
column. `εd` is `Learn.Limb.dots_end2end`'s: the call's 24,842 words plus the `f_blk` word error `εf`
through the same column. `εf` stays a hypothesis: `Learn.Limb.fblk_error_le` gives it as
`εres + 3114/2^32`, and `εres` — the sweep's residual-state invariant — is the named missing model
lemma. `aa = 1`, `two_a = 2` are the shipped coefficients. -/
theorem word_argmin_cost_bound_end2end (q d qh dh : CodeIndex → ℝ) (εf : ℝ)
    (hq : ∀ c, |qh c - q c| ≤ 99338 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32))
    (hd : ∀ c, |dh c - d c| ≤ 24842 / 2 ^ 32 + 8 * εf * (12100000000 / 2 ^ 32))
    (cstar cold : CodeIndex)
    (hmin : ∀ c, qh cstar - 2 * dh cstar ≤ qh c - 2 * dh c) :
    q cstar - 2 * d cstar
      ≤ q cold - 2 * d cold
        + 2 * ((1 : ℝ) * (99338 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32))
            + 2 * (24842 / 2 ^ 32 + 8 * εf * (12100000000 / 2 ^ 32))) := by
  have h := word_argmin_cost_bound (1 : ℝ) 2 q d qh dh
    (99338 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32))
    (24842 / 2 ^ 32 + 8 * εf * (12100000000 / 2 ^ 32)) hq hd cstar cold
    (fun c => by simpa using hmin c)
  simpa using h

/-- The end-to-end margin at an exact residual state (`εf = 3114/2^32`):
`2·(εq + 2·εd) = 3088700432769024/2^64 ≈ 1.675e-4`. -/
theorem word_argmin_cost_bound_end2end_const :
    2 * ((1 : ℝ) * (99338 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32))
        + 2 * (24842 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32)))
      = 3088700432769024 / 2 ^ 64 := by
  norm_num

/-- **The end-to-end bound, exact residual state.** The shipped path's constants with `εres = 0`, so
`εf = 3114/2^32`: the exact margin is `3088700432769024/2^64 ≈ 1.675e-4`. The measured first-sweep
argmin margin is `1.75e-7`; the kernel bound is conservative at the peaks by ~10^3. -/
theorem word_argmin_cost_bound_end2end_local (q d qh dh : CodeIndex → ℝ)
    (hq : ∀ c, |qh c - q c| ≤ 99338 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32))
    (hd : ∀ c, |dh c - d c| ≤ 24842 / 2 ^ 32 + 8 * (3114 / 2 ^ 32) * (12100000000 / 2 ^ 32))
    (cstar cold : CodeIndex)
    (hmin : ∀ c, qh cstar - 2 * dh cstar ≤ qh c - 2 * dh c) :
    q cstar - 2 * d cstar ≤ q cold - 2 * d cold + 3088700432769024 / 2 ^ 64 := by
  have h := word_argmin_cost_bound_end2end q d qh dh (3114 / 2 ^ 32) hq hd cstar cold hmin
  rw [word_argmin_cost_bound_end2end_const] at h
  exact h


variable {ι : Type} [Fintype ι]

/-- The coordinates of a row: `ι` blocks of `BLOCK = 8` coordinates. -/
abbrev Coord (ι : Type) : Type := ι × Fin 8

/-- Reindex a sum over `ι × Fin 8` as a double sum. -/
theorem sum_prod_coord {M : Type} [AddCommMonoid M] (f : Coord ι → M) :
    (∑ p : Coord ι, f p) = ∑ m, ∑ k, f (m, k) := by
  have h := Finset.sum_product (s := (Finset.univ : Finset ι)) (t := (Finset.univ : Finset (Fin 8)))
    (f := fun p : Coord ι => f p)
  simpa using h

/-- `Σ_p Σ_q K_pq u_p d_q = Σ_q d_q (K u)_q` for a symmetric kernel. -/
theorem sum_symm {α : Type} [Fintype α] (K : α → α → ℝ) (hK : ∀ p q, K p q = K q p)
    (u d : α → ℝ) :
    (∑ p, ∑ q, K p q * u p * d q) = ∑ q, d q * (∑ p, K q p * u p) := by
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun q _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [hK p q]
  ring

/-- **The Taylor identity** for a symmetric bilinear form: moving `r` by `δ` changes the quadratic
form by `2·(H·r)·δ + δᵀHδ`. -/
theorem quad_delta {α : Type} [Fintype α] (H : α → α → ℝ) (hH : ∀ p q, H p q = H q p)
    (r δ : α → ℝ) :
    (∑ p, ∑ q, H p q * (r p + δ p) * (r q + δ q))
      = (∑ p, ∑ q, H p q * r p * r q)
        + 2 * (∑ q, δ q * (∑ p, H q p * r p))
        + ∑ p, ∑ q, H p q * δ p * δ q := by
  have hpt : ∀ p q, H p q * (r p + δ p) * (r q + δ q)
      = H p q * r p * r q + (H p q * r p * δ q + H p q * δ p * r q)
        + H p q * δ p * δ q := by
    intro p q
    ring
  have hB : (∑ p, ∑ q, H p q * r p * δ q) = ∑ q, δ q * (∑ p, H q p * r p) :=
    sum_symm H hH r δ
  have hC : (∑ p, ∑ q, H p q * δ p * r q) = ∑ q, δ q * (∑ p, H q p * r p) := by
    have hcomm : (∑ p, ∑ q, H p q * δ p * r q) = ∑ p, δ p * (∑ q, H p q * r q) := by
      refine Finset.sum_congr rfl fun p _ => ?_
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun q _ => by ring
    rw [hcomm]
  calc (∑ p, ∑ q, H p q * (r p + δ p) * (r q + δ q))
      = ∑ p, ∑ q, (H p q * r p * r q + (H p q * r p * δ q + H p q * δ p * r q)
          + H p q * δ p * δ q) :=
        Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => hpt p q
    _ = (∑ p, ∑ q, H p q * r p * r q) + 2 * (∑ q, δ q * (∑ p, H q p * r p))
          + ∑ p, ∑ q, H p q * δ p * δ q := by
        simp only [Finset.sum_add_distrib, hB, hC]
        ring

variable [DecidableEq ι]

/-- A sum of a fiber-supported function collapses to the fiber. -/
theorem sum_fiber (F : Coord ι → ℝ) (m : ι) (g : Fin 8 → ℝ) :
    (∑ p : Coord ι, (if p.1 = m then g p.2 else 0) * F p) = ∑ k, g k * F (m, k) := by
  rw [sum_prod_coord (f := fun p : Coord ι => (if p.1 = m then g p.2 else 0) * F p)]
  rw [Finset.sum_eq_single m]
  · simp
  · intro m' _ hm'
    simp [hm']
  · intro hm
    exact absurd (Finset.mem_univ m) hm

/-- The same collapse for a bilinear kernel. -/
theorem sum_fiber₂ (F : Coord ι → Coord ι → ℝ) (m : ι) (g : Fin 8 → ℝ) :
    (∑ p : Coord ι, ∑ q : Coord ι,
        (if p.1 = m then g p.2 else 0) * (if q.1 = m then g q.2 else 0) * F p q)
      = ∑ k, ∑ l, g k * g l * F (m, k) (m, l) := by
  have hinner : ∀ p : Coord ι,
      (∑ q : Coord ι, (if q.1 = m then g q.2 else 0) * F p q) = ∑ l, g l * F p (m, l) :=
    fun p => sum_fiber (fun q => F p q) m g
  have hstep : ∀ p : Coord ι,
      (∑ q : Coord ι,
          (if p.1 = m then g p.2 else 0) * (if q.1 = m then g q.2 else 0) * F p q)
        = (if p.1 = m then g p.2 else 0)
          * (∑ q, (if q.1 = m then g q.2 else 0) * F p q) := by
    intro p
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun q _ => by ring
  calc (∑ p : Coord ι, ∑ q : Coord ι,
        (if p.1 = m then g p.2 else 0) * (if q.1 = m then g q.2 else 0) * F p q)
      = ∑ p, (if p.1 = m then g p.2 else 0)
          * (∑ q, (if q.1 = m then g q.2 else 0) * F p q) :=
        Finset.sum_congr rfl fun p _ => hstep p
    _ = ∑ p, (if p.1 = m then g p.2 else 0) * (∑ l, g l * F p (m, l)) :=
        Finset.sum_congr rfl fun p _ => by rw [hinner p]
    _ = ∑ k, g k * (∑ l, g l * F (m, k) (m, l)) :=
        sum_fiber (fun p => ∑ l, g l * F p (m, l)) m g
    _ = ∑ k, ∑ l, g k * g l * F (m, k) (m, l) := by
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun l _ => by ring

/-! ## The row state and the shipped block step

`sweep_rows` carries `s_cur` (the codes) and `f_res` (the residual gradient); a block decision sets
`s_cur[b0 + k] = cb[best][k]` and then `f_res -= a·(H·Δ)` with `Δ_k = cb[best][k] − s_old[k]`. The
state below is exactly `(s_cur, f_res)`, and `blockStep` is one iteration of that body. -/

/-- The row state: codes `s` and the residual gradient `f` (`f_res`). -/
structure IcmState (ι : Type) : Type where
  /-- the codes, one row of `BLOCK` per block -/
  s : ι → Fin 8 → ℝ
  /-- the residual gradient `H·(x − a·s)` the sweep carries in `f_res` -/
  f : ι → Fin 8 → ℝ

/-- The row objective `J(s) = Σ_p Σ_q H_pq r_p r_q` at `r = x − a·s`, over `Coord ι = ι × Fin 8`. -/
def rowQuadS (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (x s : ι → Fin 8 → ℝ) : ℝ :=
  ∑ p : Coord ι, ∑ q : Coord ι,
    H p.1 q.1 p.2 q.2 * (a p.1 * s p.1 p.2 - x p.1 p.2) * (a q.1 * s q.1 q.2 - x q.1 q.2)

/-- The residual gradient `f_{m,k} = Σ_{m',l} H_{m m' k l}·(x_{m'l} − a_{m'} s_{m'l})`. -/
def blockGradRow (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (x s : ι → Fin 8 → ℝ) :
    ι → Fin 8 → ℝ :=
  fun m k => ∑ q : Coord ι, H m q.1 k q.2 * (x q.1 q.2 - a q.1 * s q.1 q.2)

/-- The invariant `f` is the residual gradient at `s`. -/
def GradState (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (x : ι → Fin 8 → ℝ)
    (st : IcmState ι) : Prop :=
  st.f = blockGradRow H a x st.s

/-- The invariant that every block of `s` is a codebook row (the shipped `s_cur = cb[idx]`). -/
def CodeState (st : IcmState ι) : Prop := ∀ m, ∃ c : CodeIndex, st.s m = codebook c

/-- The sweep's `f_blk`: the gradient at the block plus the block's own metric contribution,
`f_blk_k = f_{m,k} + a_m·Σ_l H_{m m k l} s_{m,l}` (`f_res + a·s_blk·H_block`). -/
def fblk (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (st : IcmState ι) (m : ι) :
    Fin 8 → ℝ :=
  fun k => st.f m k + a m * ∑ l, H m m k l * st.s m l

/-- The block cost at a candidate vector `v`: `a²·vᵀH_{mm}v − 2a·vᵀ f_blk`, the quantity
`fit::icm::block_cost` minimises (`aa·q − two_a·dot`). -/
def phi (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (st : IcmState ι) (m : ι)
    (v : Fin 8 → ℝ) : ℝ :=
  a m ^ 2 * (∑ k, ∑ l, H m m k l * v k * v l) - 2 * a m * (∑ k, v k * fblk H a st m k)

/-- The block's argmin code: first index on a tie, as `best_code_q` returns. -/
noncomputable def blockArgmin (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (st : IcmState ι)
    (m : ι) : CodeIndex :=
  (Finset.exists_min_image (Finset.univ : Finset CodeIndex) (fun c => phi H a st m (codebook c))
    ⟨0, Finset.mem_univ 0⟩).choose

omit [Fintype ι] [DecidableEq ι] in
/-- The block decision attains the argmin. -/
theorem blockArgmin_spec (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (st : IcmState ι)
    (m : ι) (c : CodeIndex) :
    phi H a st m (codebook (blockArgmin H a st m)) ≤ phi H a st m (codebook c) :=
  (Finset.exists_min_image (Finset.univ : Finset CodeIndex) (fun c => phi H a st m (codebook c))
    ⟨0, Finset.mem_univ 0⟩).choose_spec.2 c (Finset.mem_univ c)

/-- One iteration of the shipped block body: choose the argmin code at the current effective data,
set the block of `s`, and update `f` by `f -= a·(H·Δ)`. -/
noncomputable def blockStep (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (st : IcmState ι)
    (m : ι) : IcmState ι :=
  { s := Function.update st.s m (codebook (blockArgmin H a st m))
    f := fun m' k =>
      st.f m' k - a m
        * ∑ l, H m' m k l * (codebook (blockArgmin H a st m) l - st.s m l) }

/-! ## The update identity and the descent -/

omit [Fintype ι] [DecidableEq ι] in
/-- The block-vector `phi` difference: moving a block from `s_m` to `s_m + d` changes the block cost
by `a²dᵀH_{mm}d − 2a·dᵀf`, the gradient term.

This is the expansion identity used below; `phi` is written with `f_blk = f + a·H_{mm}s_m`, and the
two `a·H_{mm}s_m` terms cancel. -/
theorem phi_delta (H : ι → ι → Fin 8 → Fin 8 → ℝ)
    (hH : ∀ m m' k l, H m m' k l = H m' m l k) (a : ι → ℝ) (st : IcmState ι) (m : ι)
    (d : Fin 8 → ℝ) :
    phi H a st m (fun k => st.s m k + d k) - phi H a st m (st.s m)
      = a m ^ 2 * (∑ k, ∑ l, H m m k l * d k * d l)
        - 2 * a m * (∑ k, d k * st.f m k) := by
  have hK : ∀ k l, H m m k l = H m m l k := fun k l => hH m m k l
  have hq := quad_delta (fun k l => H m m k l) hK (st.s m) d
  have hq' : (∑ k, ∑ l, H m m k l * (st.s m k + d k) * (st.s m l + d l))
      = (∑ k, ∑ l, H m m k l * (st.s m) k * (st.s m) l)
        + 2 * (∑ l, d l * (∑ k, H m m l k * (st.s m) k))
        + ∑ k, ∑ l, H m m k l * d k * d l := by
    simpa [Pi.add_apply] using hq
  have hsmd : (∑ k, ∑ l, H m m k l * (st.s m) k * d l)
      = ∑ l, d l * (∑ k, H m m l k * (st.s m) k) := sum_symm _ hK (st.s m) d
  have hlin : (∑ k, (st.s m k + d k) * fblk H a st m k)
      = (∑ k, (st.s m) k * fblk H a st m k) + ∑ k, d k * fblk H a st m k := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun k _ => ?_
    ring
  have hfbd : (∑ k, d k * fblk H a st m k)
      = (∑ k, d k * st.f m k)
        + a m * (∑ l, d l * (∑ k, H m m l k * (st.s m) k)) := by
    simp only [fblk]
    have hsplit : ∀ k : Fin 8,
        d k * (st.f m k + a m * ∑ l, H m m k l * st.s m l)
          = d k * st.f m k + a m * (d k * ∑ l, H m m k l * st.s m l) := by
      intro k
      ring
    simp only [hsplit, Finset.sum_add_distrib]
    rw [← Finset.mul_sum]
  simp only [phi]
  rw [hq', hlin, hfbd]
  ring



/-! ## The row update identity -/

/-- **The row delta identity.** Moving one block to `v` changes the row objective by exactly the
block cost difference `phi v − phi s_m`, given that `f` is the residual gradient.

This is what makes the sequential sweep provable: `phi` carries the gradient `f_blk` of the *current*
state, so the fold's `f` (updated `f -= a·H·Δ`) is the effective data the next block reads. -/
theorem rowQuad_update_delta (H : ι → ι → Fin 8 → Fin 8 → ℝ)
    (hH : ∀ m m' k l, H m m' k l = H m' m l k) (a : ι → ℝ) (x : ι → Fin 8 → ℝ)
    (st : IcmState ι) (m : ι) (v : Fin 8 → ℝ) (hgrad : GradState H a x st) :
    rowQuadS H a x (Function.update st.s m v) - rowQuadS H a x st.s
      = phi H a st m v - phi H a st m (st.s m) := by
  let r : Coord ι → ℝ := fun p => x p.1 p.2 - a p.1 * st.s p.1 p.2
  let δ : Coord ι → ℝ := fun p =>
    if p.1 = m then -(a m) * (v p.2 - st.s m p.2) else 0
  have hHflat : ∀ p q : Coord ι, H p.1 q.1 p.2 q.2 = H q.1 p.1 q.2 p.2 :=
    fun p q => hH p.1 q.1 p.2 q.2
  have hupd : ∀ q : Coord ι,
      (Function.update st.s m v) q.1 q.2
        = st.s q.1 q.2 + (if q.1 = m then v q.2 - st.s m q.2 else 0) := by
    intro q
    rw [Function.update_apply]
    by_cases hq : q.1 = m
    · subst hq; simp
    · simp [hq]
  have hr : ∀ p : Coord ι,
      x p.1 p.2 - a p.1 * (Function.update st.s m v) p.1 p.2 = r p + δ p := by
    intro p
    by_cases hp : p.1 = m
    · have h1 : (Function.update st.s m v) p.1 p.2 = v p.2 := by
        simp [Function.update, hp]
      have h2 : st.s m p.2 = st.s p.1 p.2 := by rw [hp]
      have h3 : a m = a p.1 := by rw [hp]
      rw [h1]
      simp only [δ, if_pos hp, r]
      rw [h2, h3]
      ring
    · have h1 : (Function.update st.s m v) p.1 p.2 = st.s p.1 p.2 := by
        simp [Function.update, hp]
      rw [h1]
      simp only [δ, if_neg hp, r]
      ring
  have hqr : rowQuadS H a x st.s
      = ∑ p : Coord ι, ∑ q : Coord ι, H p.1 q.1 p.2 q.2 * r p * r q := by
    simp only [rowQuadS, r]
    refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
    ring
  have hqr' : rowQuadS H a x (Function.update st.s m v)
      = ∑ p : Coord ι, ∑ q : Coord ι,
          H p.1 q.1 p.2 q.2 * (r p + δ p) * (r q + δ q) := by
    simp only [rowQuadS]
    refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
    have hp := hr p
    have hq := hr q
    have hp' : a p.1 * (Function.update st.s m v) p.1 p.2 - x p.1 p.2
        = -(r p + δ p) := by linarith
    have hq' : a q.1 * (Function.update st.s m v) q.1 q.2 - x q.1 q.2
        = -(r q + δ q) := by linarith
    calc H p.1 q.1 p.2 q.2
          * (a p.1 * (Function.update st.s m v) p.1 p.2 - x p.1 p.2)
          * (a q.1 * (Function.update st.s m v) q.1 q.2 - x q.1 q.2)
        = H p.1 q.1 p.2 q.2 * (-(r p + δ p)) * (-(r q + δ q)) := by rw [hp', hq']
      _ = H p.1 q.1 p.2 q.2 * (r p + δ p) * (r q + δ q) := by ring
  have hdelta := quad_delta (fun p q : Coord ι => H p.1 q.1 p.2 q.2) hHflat r δ
  have hF : ∀ k : Fin 8, (∑ p : Coord ι, H m p.1 k p.2 * r p) = st.f m k := by
    intro k
    have h := congrFun (congrFun hgrad m) k
    simp only [blockGradRow] at h
    rw [← h]
  have hlin : (∑ q : Coord ι, δ q * (∑ p : Coord ι, H q.1 p.1 q.2 p.2 * r p))
      = -(a m) * ∑ k, (v k - st.s m k) * st.f m k := by
    have h := sum_fiber (F := fun q : Coord ι => ∑ p : Coord ι, H q.1 p.1 q.2 p.2 * r p) m
      (fun k => -(a m) * (v k - st.s m k))
    simp only [δ] at h ⊢
    rw [h, Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [hF k]
    ring
  have hquad : (∑ p : Coord ι, ∑ q : Coord ι, H p.1 q.1 p.2 q.2 * δ p * δ q)
      = a m ^ 2 * (∑ k, ∑ l, H m m k l * (v k - st.s m k) * (v l - st.s m l)) := by
    have h := sum_fiber₂ (F := fun p q : Coord ι => H p.1 q.1 p.2 q.2) m
      (fun k => -(a m) * (v k - st.s m k))
    have hδ : (∑ p : Coord ι, ∑ q : Coord ι, H p.1 q.1 p.2 q.2 * δ p * δ q)
        = ∑ p : Coord ι, ∑ q : Coord ι,
            (if p.1 = m then -(a m) * (v p.2 - st.s m p.2) else 0)
              * (if q.1 = m then -(a m) * (v q.2 - st.s m q.2) else 0)
              * H p.1 q.1 p.2 q.2 := by
      refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
      simp only [δ]
      ring
    rw [hδ, h]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun l _ => ?_
    ring
  have hphi := phi_delta H hH a st m (fun k => v k - st.s m k)
  have hv : (fun k => st.s m k + (v k - st.s m k)) = v := by
    funext k; ring
  rw [hv] at hphi
  calc rowQuadS H a x (Function.update st.s m v) - rowQuadS H a x st.s
      = (∑ p : Coord ι, ∑ q : Coord ι, H p.1 q.1 p.2 q.2 * (r p + δ p) * (r q + δ q))
        - (∑ p : Coord ι, ∑ q : Coord ι, H p.1 q.1 p.2 q.2 * r p * r q) := by
        rw [hqr, hqr']
    _ = 2 * (∑ q : Coord ι, δ q * (∑ p : Coord ι, H q.1 p.1 q.2 p.2 * r p))
          + ∑ p : Coord ι, ∑ q : Coord ι, H p.1 q.1 p.2 q.2 * δ p * δ q := by
        linarith [hdelta]
    _ = 2 * (-(a m) * ∑ k, (v k - st.s m k) * st.f m k)
          + a m ^ 2 * (∑ k, ∑ l, H m m k l * (v k - st.s m k) * (v l - st.s m l)) := by
        rw [hlin, hquad]
    _ = phi H a st m v - phi H a st m (st.s m) := by
        rw [hphi]; ring

/-- The per-block descent: the argmin step never increases the row objective. -/
theorem rowQuad_step_le (H : ι → ι → Fin 8 → Fin 8 → ℝ)
    (hH : ∀ m m' k l, H m m' k l = H m' m l k) (a : ι → ℝ) (x : ι → Fin 8 → ℝ)
    (st : IcmState ι) (m : ι) (hgrad : GradState H a x st) (hcode : CodeState st) :
    rowQuadS H a x (blockStep H a st m).s ≤ rowQuadS H a x st.s := by
  obtain ⟨c₀, hc₀⟩ := hcode m
  have hbs : (blockStep H a st m).s = Function.update st.s m (codebook (blockArgmin H a st m)) :=
    rfl
  have hdelta := rowQuad_update_delta H hH a x st m (codebook (blockArgmin H a st m)) hgrad
  have hspec := blockArgmin_spec H a st m c₀
  have hphi_eq : phi H a st m (st.s m) = phi H a st m (codebook c₀) := by rw [hc₀]
  rw [hbs]
  linarith

/-- The block step preserves gradient truth. -/
theorem gradState_step (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (x : ι → Fin 8 → ℝ)
    (st : IcmState ι) (m : ι) (hgrad : GradState H a x st) :
    GradState H a x (blockStep H a st m) := by
  unfold GradState at *
  funext m' k
  have hupd : ∀ q : Coord ι,
      (blockStep H a st m).s q.1 q.2
        = st.s q.1 q.2
          + (if q.1 = m
              then codebook (blockArgmin H a st m) q.2 - st.s m q.2 else 0) := by
    intro q
    rw [show (blockStep H a st m).s
        = Function.update st.s m (codebook (blockArgmin H a st m)) from rfl]
    rw [Function.update_apply]
    by_cases hq : q.1 = m
    · subst hq; simp
    · simp [hq]
  have hL : (blockStep H a st m).f m' k
      = st.f m' k
        - a m * ∑ l, H m' m k l * (codebook (blockArgmin H a st m) l - st.s m l) := rfl
  rw [hL, congrFun (congrFun hgrad m') k]
  simp only [blockGradRow]
  have hR : (∑ q : Coord ι,
        H m' q.1 k q.2 * (x q.1 q.2 - a q.1 * (blockStep H a st m).s q.1 q.2))
      = (∑ q : Coord ι, H m' q.1 k q.2 * (x q.1 q.2 - a q.1 * st.s q.1 q.2))
        + ∑ q : Coord ι, H m' q.1 k q.2
            * (if q.1 = m
                then -(a m) * (codebook (blockArgmin H a st m) q.2 - st.s m q.2)
                else 0) := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun q _ => ?_
    rw [hupd q]
    by_cases hq : q.1 = m
    · subst hq; simp; ring
    · simp [hq]
  rw [hR]
  have hfib := sum_fiber (F := fun q : Coord ι => H m' q.1 k q.2) m
    (fun l => -(a m) * (codebook (blockArgmin H a st m) l - st.s m l))
  have hswap : (∑ q : Coord ι,
        H m' q.1 k q.2
          * (if q.1 = m
              then -(a m) * (codebook (blockArgmin H a st m) q.2 - st.s m q.2)
              else 0))
      = ∑ q : Coord ι,
          (if q.1 = m
              then -(a m) * (codebook (blockArgmin H a st m) q.2 - st.s m q.2)
              else 0) * H m' q.1 k q.2 := by
    refine Finset.sum_congr rfl fun q _ => ?_
    ring
  rw [hswap, hfib]
  have hsum : (∑ l : Fin 8,
        (-(a m) * (codebook (blockArgmin H a st m) l - st.s m l)) * H m' m k l)
      = -(a m) * ∑ l, H m' m k l * (codebook (blockArgmin H a st m) l - st.s m l) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun l _ => ?_
    ring
  rw [hsum]
  ring

omit [Fintype ι] in
/-- The block step keeps every block on a codebook row. -/
theorem codeState_step (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (st : IcmState ι) (m : ι)
    (h : CodeState st) : CodeState (blockStep H a st m) := by
  intro m'
  by_cases hm' : m' = m
  · rw [hm']
    exact ⟨blockArgmin H a st m, by simp [blockStep]⟩
  · obtain ⟨c, hc⟩ := h m'
    exact ⟨c, by simpa [blockStep, Function.update, hm'] using hc⟩

/-! ## The invariant and the ordered fold -/

/-- The pass invariant: the state's `f` is the residual gradient and every block holds a codebook row.
`sweep_rows` maintains both: `f_res` is updated by exactly `−a·H·Δ`, and `s_cur` only ever receives
`cb[best]`. -/
def IcmInv (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (x : ι → Fin 8 → ℝ)
    (st : IcmState ι) : Prop :=
  GradState H a x st ∧ CodeState st

/-- The block step preserves the pass invariant. -/
theorem icmInv_step (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (x : ι → Fin 8 → ℝ)
    (st : IcmState ι) (m : ι) (h : IcmInv H a x st) : IcmInv H a x (blockStep H a st m) :=
  ⟨gradState_step H a x st m h.1, codeState_step H a st m h.2⟩

/-- A fold whose steps preserve `Inv` and do not increase `g`. -/
theorem foldl_inv_le {β α : Type} (f : β → α → β) (g : β → ℝ) (Inv : β → Prop)
    (hstep : ∀ b a, Inv b → Inv (f b a) ∧ g (f b a) ≤ g b) (l : List α) :
    ∀ b : β, Inv b → Inv (l.foldl f b) ∧ g (l.foldl f b) ≤ g b := by
  induction l with
  | nil => intro b hb; exact ⟨hb, le_rfl⟩
  | cons a t ih =>
      intro b hb
      obtain ⟨h1, h2⟩ := hstep b a hb
      obtain ⟨h3, h4⟩ := ih (f b a) h1
      exact ⟨h3, le_trans h4 h2⟩

/-- The pass: `sweep_rows`' block loop, folding `blockStep` over the block order. -/
noncomputable def icmPass (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ)
    (order : List ι) (st : IcmState ι) : IcmState ι :=
  order.foldl (fun st m => blockStep H a st m) st

/-- A pass preserves the invariant and does not increase the row objective. -/
theorem icmPass_inv_le (H : ι → ι → Fin 8 → Fin 8 → ℝ)
    (hH : ∀ m m' k l, H m m' k l = H m' m l k) (a : ι → ℝ) (x : ι → Fin 8 → ℝ)
    (order : List ι) (st : IcmState ι) (h : IcmInv H a x st) :
    IcmInv H a x (icmPass H a order st) ∧
      rowQuadS H a x (icmPass H a order st).s ≤ rowQuadS H a x st.s := by
  have hstep : ∀ b (m : ι), IcmInv H a x b →
      IcmInv H a x (blockStep H a b m) ∧
        rowQuadS H a x (blockStep H a b m).s ≤ rowQuadS H a x b.s :=
    fun b m hb =>
      ⟨icmInv_step H a x b m hb, rowQuad_step_le H hH a x b m hb.1 hb.2⟩
  simpa only [icmPass] using foldl_inv_le (fun st m => blockStep H a st m)
    (fun st => rowQuadS H a x st.s) (IcmInv H a x) hstep order st h

/-! ## The `sweep_rows` block order and the pass composition -/

/-- `sweep_rows`' block order at pass `sw`: even passes walk `nb−1, …, 0`, odd passes `0, …, nb−1`
(`m = if sw % 2 == 0 { nb - 1 - step } else { step }`). -/
def blockOrder (nb sw : ℕ) : List (Fin nb) :=
  if sw % 2 = 0 then (List.finRange nb).reverse else List.finRange nb

/-- An even pass runs the blocks in reverse. -/
theorem blockOrder_even (nb sw : ℕ) (h : sw % 2 = 0) :
    blockOrder nb sw = (List.finRange nb).reverse := by
  simp [blockOrder, h]

/-- An odd pass runs the blocks forward. -/
theorem blockOrder_odd (nb sw : ℕ) (h : sw % 2 ≠ 0) :
    blockOrder nb sw = List.finRange nb := by
  simp [blockOrder, h]

/-- The order visits every block exactly once. -/
theorem blockOrder_length (nb sw : ℕ) : (blockOrder nb sw).length = nb := by
  unfold blockOrder
  split <;> simp

/-- The order visits every block exactly once (no repeats). -/
theorem blockOrder_nodup (nb sw : ℕ) : (blockOrder nb sw).Nodup := by
  unfold blockOrder
  split
  · exact List.nodup_reverse.mpr (List.nodup_finRange nb)
  · exact List.nodup_finRange nb

/-- The pass composition: `k` sweeps, pass `j` at parity `j % 2`. -/
noncomputable def icmPasses (nb : ℕ) (H : Fin nb → Fin nb → Fin 8 → Fin 8 → ℝ)
    (a : Fin nb → ℝ) (st : IcmState (Fin nb)) : ℕ → IcmState (Fin nb)
  | 0 => st
  | j + 1 => icmPass H a (blockOrder nb j) (icmPasses nb H a st j)

/-- Every pass keeps the invariant. -/
theorem icmPasses_inv (nb : ℕ) (H : Fin nb → Fin nb → Fin 8 → Fin 8 → ℝ)
    (hH : ∀ m m' k l, H m m' k l = H m' m l k) (a : Fin nb → ℝ) (x : Fin nb → Fin 8 → ℝ)
    (st : IcmState (Fin nb)) (h : IcmInv H a x st) (k : ℕ) :
    IcmInv H a x (icmPasses nb H a st k) := by
  induction k with
  | zero => exact h
  | succ j ih => exact (icmPass_inv_le H hH a x (blockOrder nb j) _ ih).1

/-- The composition: `k` passes never increase the row objective. -/
theorem icmPasses_quad_le (nb : ℕ) (H : Fin nb → Fin nb → Fin 8 → Fin 8 → ℝ)
    (hH : ∀ m m' k l, H m m' k l = H m' m l k) (a : Fin nb → ℝ) (x : Fin nb → Fin 8 → ℝ)
    (st : IcmState (Fin nb)) (h : IcmInv H a x st) (k : ℕ) :
    rowQuadS H a x (icmPasses nb H a st k).s ≤ rowQuadS H a x st.s := by
  induction k with
  | zero => exact le_rfl
  | succ j ih =>
      have hinv : IcmInv H a x (icmPasses nb H a st j) :=
        icmPasses_inv nb H hH a x st h j
      exact le_trans (icmPass_inv_le H hH a x (blockOrder nb j) _ hinv).2 ih

/-! ## The `f_blk` / `f_out` split -/

/-- The off-block residual contribution to block `m`'s effective data (`f_out` in `sweep_rows`). -/
def fout (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (x s : ι → Fin 8 → ℝ) (m : ι)
    (k : Fin 8) : ℝ :=
  ∑ m' ∈ (Finset.univ.erase m), ∑ l, H m m' k l * (x m' l - a m' * s m' l)

/-- **The `f_blk`/`f_out` correspondence**: under the gradient invariant the sweep's `f_blk` is the
block's data image plus the other blocks' residual pull, `f_blk_k = Σ_l H_{m m k l} x_{m,l} + f_out_k`.

The fold's `f` therefore represents the sequential `f_blk` with no additional hypothesis: the
hypothesis is `GradState`, which `gradState_step` preserves. -/
theorem fblk_eq_data_add_fout (H : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (x : ι → Fin 8 → ℝ)
    (st : IcmState ι) (m : ι) (hgrad : GradState H a x st) (k : Fin 8) :
    fblk H a st m k = (∑ l, H m m k l * x m l) + fout H a x st.s m k := by
  have hf : st.f m k = ∑ m', ∑ l, H m m' k l * (x m' l - a m' * st.s m' l) := by
    have h := congrFun (congrFun hgrad m) k
    simp only [blockGradRow] at h
    rw [h]
    exact sum_prod_coord (f := fun p : Coord ι =>
      H m p.1 k p.2 * (x p.1 p.2 - a p.1 * st.s p.1 p.2))
  simp only [fblk, fout]
  rw [hf]
  rw [← Finset.add_sum_erase (s := (Finset.univ : Finset ι))
    (f := fun m' => ∑ l, H m m' k l * (x m' l - a m' * st.s m' l)) (Finset.mem_univ m)]
  have hmm : (∑ l, H m m k l * (x m l - a m * st.s m l))
        + a m * (∑ l, H m m k l * st.s m l)
      = ∑ l, H m m k l * x m l := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun l _ => ?_
    ring
  linarith [hmm]

/-! ## The joint pair decision (`joint_band`, `best_pair_q`)

`joint_band` (`src/learn/fold/fit/icm.rs:1265`) replaces both planes' codes of one block by the
`CODE_LEN²`-way argmin of `pair_cost` (`:235`):

`pairPhi c₁ c₂ = 2·a₁a₂·cross c₁ c₂ + phi₁ c₁ + phi₂ c₂`,

with `cross c₁ c₂ = C_{c₁}ᵀ H₁₂ C_{c₂}` (`x_all`, `:836`) and `phiᵢ` the block cost at
`fᵢ_blk = fᵢ + aᵢ·Hᵢᵢsᵢ + a_other·Hᵢ_other·s_other` (`:1355`). The block below carries the two-plane
state, both planes' effective data, the pair cost, the flat-index argmin with the shipped tie rule,
the joint gradient invariant, the joint delta identity and the descent over the same `blockOrder`
pass.

The ridge is not a separate term of the pair cost: it is inside `H₁₁`/`H₂₂` (the banded ridge solve
feeds the blocks the decision reads), so `pairPhi_codebook_eq` is the whole cost.

`H₁₂` is not symmetric. It enters the objective only as `2·r₁ᵀH₁₂r₂`, so the stacked metric
`[[H₁₁, H₁₂], [H₁₂ᵀ, H₂₂]]` is symmetric by construction and the identity needs only `H₁₁` and
`H₂₂` symmetric; `H₂₁ := H₁₂ᵀ` is a definition, not a hypothesis.
-/

/-! ## The single-plane block cost part -/


/-- One plane's block cost: `a²·vᵀKv − 2a·vᵀ f_blk` (`block_cost aa two_a q dot` with
`aa = a²`, `two_a = 2a`). -/
def planePhi (K : Fin 8 → Fin 8 → ℝ) (a : ℝ) (fb : Fin 8 → ℝ) (v : Fin 8 → ℝ) : ℝ :=
  a ^ 2 * (∑ k, ∑ l, K k l * v k * v l) - 2 * a * (∑ k, v k * fb k)

/-- The single-plane `phi` difference: moving from `u` to `u + d` changes the block cost by
`a²dᵀKd − 2a·dᵀ(f_blk − a·Ku)`. `K` is the plane's metric block, so it must be symmetric. -/
theorem planePhi_delta (K : Fin 8 → Fin 8 → ℝ) (hK : ∀ k l, K k l = K l k) (a : ℝ)
    (fb : Fin 8 → ℝ) (u d : Fin 8 → ℝ) :
    planePhi K a fb (fun k => u k + d k) - planePhi K a fb u
      = a ^ 2 * (∑ k, ∑ l, K k l * d k * d l)
        - 2 * a * (∑ k, d k * (fb k - a * (∑ l, K k l * u l))) := by
  have hq := quad_delta K hK u d
  have hq' : (∑ k, ∑ l, K k l * (u k + d k) * (u l + d l))
      = (∑ k, ∑ l, K k l * u k * u l)
        + 2 * (∑ l, d l * (∑ k, K l k * u k))
        + ∑ k, ∑ l, K k l * d k * d l := by
    simpa [Pi.add_apply] using hq
  have hlin : (∑ k, (u k + d k) * fb k)
      = (∑ k, u k * fb k) + ∑ k, d k * fb k := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun k _ => ?_
    ring
  have hku : (∑ k, d k * fb k) - a * (∑ l, d l * (∑ k, K l k * u k))
      = ∑ k, d k * (fb k - a * (∑ l, K k l * u l)) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun k _ => ?_
    ring
  simp only [planePhi]
  rw [hq', hlin, ← hku]
  ring


/-! ## The cross block -/



/-- `H₂₁` is `H₁₂` transposed: `(H₂₁)[m,m']_{k,l} = (H₁₂)[m',m]_{l,k}`. `H₁₂` is not symmetric
(`icm.rs:1425`), so the second plane's gradient contracts the transposed window. -/
def H21 (H12 : ι → ι → Fin 8 → Fin 8 → ℝ) (m m' : ι) (k l : Fin 8) : ℝ := H12 m' m l k

/-- The cross block form `uᵀ(H₁₂[m])v` at block `m`. -/
def crossForm (H12 : ι → ι → Fin 8 → Fin 8 → ℝ) (m : ι) (u v : Fin 8 → ℝ) : ℝ :=
  ∑ k, ∑ l, H12 m m k l * u k * v l

omit [Fintype ι] [DecidableEq ι] in
/-- The transpose of the cross block is the cross block of the transposed kernel. -/
theorem crossForm_H21 (H12 : ι → ι → Fin 8 → Fin 8 → ℝ) (m : ι) (u v : Fin 8 → ℝ) :
    crossForm (H21 H12) m u v = crossForm H12 m v u := by
  simp only [crossForm, H21]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => ?_
  ring




omit [DecidableEq ι] in
/-- **Bilinearity of the cross quadratic.** The cross term `2·r₁ᵀH₁₂r₂` changes by
`2(r₁ᵀH₁₂δ₂ + δ₁ᵀH₁₂r₂ + δ₁ᵀH₁₂δ₂)`; no symmetry of `H₁₂` is used or needed, because the stacked
metric's `(2,1)` block is `H₁₂ᵀ` by construction. -/
theorem cross_delta (H12 : ι → ι → Fin 8 → Fin 8 → ℝ)
    (r1 r2 δ1 δ2 : Coord ι → ℝ) :
    2 * (∑ p : Coord ι, ∑ q : Coord ι,
          H12 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r2 q + δ2 q))
      - 2 * (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * r1 p * r2 q)
      = 2 * (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * r1 p * δ2 q)
        + 2 * (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * δ1 p * r2 q)
        + 2 * (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * δ1 p * δ2 q) := by
  have hpt : ∀ p q : Coord ι, H12 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r2 q + δ2 q)
      = H12 p.1 q.1 p.2 q.2 * r1 p * r2 q
        + H12 p.1 q.1 p.2 q.2 * r1 p * δ2 q
        + H12 p.1 q.1 p.2 q.2 * δ1 p * r2 q
        + H12 p.1 q.1 p.2 q.2 * δ1 p * δ2 q := by
    intro p q
    ring
  simp only [hpt, Finset.sum_add_distrib]
  ring


/-! ## The two-plane state and the pair cost -/



/-- The two-plane row state: codes and residual gradients per plane (`joint_band`'s `s₁,s₂,f₁,f₂`). -/
structure TwoState (ι : Type) : Type where
  /-- plane 1's codes -/
  s1 : ι → Fin 8 → ℝ
  /-- plane 2's codes -/
  s2 : ι → Fin 8 → ℝ
  /-- plane 1's residual gradient `f₁` -/
  f1 : ι → Fin 8 → ℝ
  /-- plane 2's residual gradient `f₂` -/
  f2 : ι → Fin 8 → ℝ

/-- The cross quadratic `r₁ᵀH₁₂r₂` of a row. -/
def crossQuad (H12 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (x1 x2 : ι → Fin 8 → ℝ) (s1 s2 : ι → Fin 8 → ℝ) : ℝ :=
  ∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2
    * (a1 p.1 * s1 p.1 p.2 - x1 p.1 p.2) * (a2 q.1 * s2 q.1 q.2 - x2 q.1 q.2)

/-- The joint row objective of the two planes: `J = r₁ᵀH₁₁r₁ + 2·r₁ᵀH₁₂r₂ + r₂ᵀH₂₂r₂` at
`rᵢ = aᵢsᵢ − xᵢ`.

The cross block carries the factor `2`, so the stacked metric `[[H₁₁, H₁₂], [H₁₂ᵀ, H₂₂]]` is
symmetric by construction: only `H₁₁` and `H₂₂` need their block symmetry, and `H₁₂` needs none. -/
def jointQuad (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (x1 x2 : ι → Fin 8 → ℝ) (s1 s2 : ι → Fin 8 → ℝ) : ℝ :=
  rowQuadS H11 a1 x1 s1 + 2 * crossQuad H12 a1 a2 x1 x2 s1 s2 + rowQuadS H22 a2 x2 s2

/-- The plane-1 joint gradient `f₁ = H₁₁r₁ + H₁₂r₂`. -/
def jointGrad1 (H11 H12 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (x1 x2 : ι → Fin 8 → ℝ) (s1 s2 : ι → Fin 8 → ℝ) : ι → Fin 8 → ℝ :=
  fun m k => (∑ q : Coord ι, H11 m q.1 k q.2 * (x1 q.1 q.2 - a1 q.1 * s1 q.1 q.2))
    + ∑ q : Coord ι, H12 m q.1 k q.2 * (x2 q.1 q.2 - a2 q.1 * s2 q.1 q.2)

/-- The plane-2 joint gradient `f₂ = H₂₁r₁ + H₂₂r₂` at `H₂₁ = H₁₂ᵀ`. -/
def jointGrad2 (H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (x1 x2 : ι → Fin 8 → ℝ) (s1 s2 : ι → Fin 8 → ℝ) : ι → Fin 8 → ℝ :=
  fun m k => (∑ q : Coord ι, H21 H12 m q.1 k q.2 * (x1 q.1 q.2 - a1 q.1 * s1 q.1 q.2))
    + ∑ q : Coord ι, H22 m q.1 k q.2 * (x2 q.1 q.2 - a2 q.1 * s2 q.1 q.2)

/-- The joint gradient invariant: both `fᵢ` are the plane's residual gradients. -/
def JointGradState (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (x1 x2 : ι → Fin 8 → ℝ) (st : TwoState ι) : Prop :=
  st.f1 = jointGrad1 H11 H12 a1 a2 x1 x2 st.s1 st.s2 ∧
  st.f2 = jointGrad2 H12 H22 a1 a2 x1 x2 st.s1 st.s2

/-- Every block of both planes holds a codebook row. -/
def TwoCodeState (st : TwoState ι) : Prop :=
  (∀ m, ∃ c : CodeIndex, st.s1 m = codebook c) ∧
    ∀ m, ∃ c : CodeIndex, st.s2 m = codebook c

/-- The sweep's `f_blk` for plane 1: `f₁ + a₁·H₁₁s₁ + a₂·H₁₂s₂` (`:1355`, `f1b`). -/
def pairFblk1 (H11 H12 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ) (st : TwoState ι)
    (m : ι) : Fin 8 → ℝ :=
  fun k => st.f1 m k + a1 m * (∑ l, H11 m m k l * st.s1 m l)
    + a2 m * (∑ l, H12 m m k l * st.s2 m l)

/-- The sweep's `f_blk` for plane 2: `f₂ + a₂·H₂₂s₂ + a₁·H₂₁s₁`, `H₂₁ = H₁₂ᵀ` (`f2b`). -/
def pairFblk2 (H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ) (st : TwoState ι)
    (m : ι) : Fin 8 → ℝ :=
  fun k => st.f2 m k + a2 m * (∑ l, H22 m m k l * st.s2 m l)
    + a1 m * (∑ l, H21 H12 m m k l * st.s1 m l)

/-- The pair cost: `pairPhi = (a₁²q₁ − 2a₁d₁) + 2a₁a₂·X + (a₂²q₂ − 2a₂d₂)`, exactly
`pair_cost two_a12 X av bv = av + bv + 2a₁a₂X` (`icm.rs:235`) with `av`, `bv` the single-plane
`block_cost`s at `f1b`, `f2b` and `X = C_{c₁}ᵀH₁₂C_{c₂}` (the precomputed `x_all`). -/
def pairPhi (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ) (st : TwoState ι)
    (m : ι) (v1 v2 : Fin 8 → ℝ) : ℝ :=
  planePhi (H11 m m) (a1 m) (pairFblk1 H11 H12 a1 a2 st m) v1
    + 2 * (a1 m * a2 m) * crossForm H12 m v1 v2
    + planePhi (H22 m m) (a2 m) (pairFblk2 H12 H22 a1 a2 st m) v2

omit [Fintype ι] [DecidableEq ι] in
/-- The cross block is bilinear. -/
theorem crossForm_add (H12 : ι → ι → Fin 8 → Fin 8 → ℝ) (m : ι) (u du v dv : Fin 8 → ℝ) :
    crossForm H12 m (fun k => u k + du k) (fun l => v l + dv l)
      = crossForm H12 m u v + crossForm H12 m u dv
        + crossForm H12 m du v + crossForm H12 m du dv := by
  simp only [crossForm]
  have hpt : ∀ k l : Fin 8, H12 m m k l * (u k + du k) * (v l + dv l)
      = H12 m m k l * u k * v l + H12 m m k l * u k * dv l
        + H12 m m k l * du k * v l + H12 m m k l * du k * dv l := by
    intro k l
    ring
  simp only [hpt, Finset.sum_add_distrib]

omit [Fintype ι] [DecidableEq ι] in
/-- The plane-1 effective data minus its own metric contribution is `f₁ + a₂·H₁₂s₂`. -/
theorem pairFblk1_sub (H11 H12 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (st : TwoState ι) (m : ι) (d1 : Fin 8 → ℝ) :
    (∑ k, d1 k * (pairFblk1 H11 H12 a1 a2 st m k
        - a1 m * (∑ l, H11 m m k l * st.s1 m l)))
      = (∑ k, d1 k * st.f1 m k) + a2 m * crossForm H12 m d1 (st.s2 m) := by
  have hsum : (∑ k, a2 m * (d1 k * (∑ l, H12 m m k l * st.s2 m l)))
      = a2 m * crossForm H12 m d1 (st.s2 m) := by
    simp only [crossForm, Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => by ring
  rw [← hsum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [pairFblk1]
  ring

omit [Fintype ι] [DecidableEq ι] in
/-- The plane-2 effective data minus its own metric contribution is `f₂ + a₁·H₂₁s₁`. -/
theorem pairFblk2_sub (H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (st : TwoState ι) (m : ι) (d2 : Fin 8 → ℝ) :
    (∑ k, d2 k * (pairFblk2 H12 H22 a1 a2 st m k
        - a2 m * (∑ l, H22 m m k l * st.s2 m l)))
      = (∑ k, d2 k * st.f2 m k) + a1 m * crossForm H12 m (st.s1 m) d2 := by
  have hsum : (∑ k, a1 m * (d2 k * (∑ l, H21 H12 m m k l * st.s1 m l)))
      = a1 m * crossForm H12 m (st.s1 m) d2 := by
    rw [← crossForm_H21 H12 m d2 (st.s1 m)]
    simp only [crossForm, Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => by ring
  rw [← hsum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [pairFblk2]
  ring

omit [Fintype ι] [DecidableEq ι] in
/-- **The pair block delta.** Moving both planes of a block from `(s₁ₘ, s₂ₘ)` to
`(s₁ₘ + d₁, s₂ₘ + d₂)` changes the pair cost by
`a₁²d₁ᵀH₁₁d₁ + 2a₁a₂d₁ᵀH₁₂d₂ + a₂²d₂ᵀH₂₂d₂ − 2a₁d₁ᵀf₁ − 2a₂d₂ᵀf₂`.

The two `aᵢ·Hᵢᵢsᵢ` terms of the effective data cancel against the corresponding `Hᵢᵢ` expansions, and
the cross terms cancel exactly because `H₂₁ = H₁₂ᵀ`; no symmetry of `H₁₂` is used. -/
theorem pairPhi_delta (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ)
    (h11 : ∀ m k l, H11 m m k l = H11 m m l k)
    (h22 : ∀ m k l, H22 m m k l = H22 m m l k) (a1 a2 : ι → ℝ) (st : TwoState ι) (m : ι)
    (d1 d2 : Fin 8 → ℝ) :
    pairPhi H11 H12 H22 a1 a2 st m (fun k => st.s1 m k + d1 k) (fun k => st.s2 m k + d2 k)
      - pairPhi H11 H12 H22 a1 a2 st m (st.s1 m) (st.s2 m)
      = a1 m ^ 2 * (∑ k, ∑ l, H11 m m k l * d1 k * d1 l)
        + 2 * (a1 m * a2 m) * (∑ k, ∑ l, H12 m m k l * d1 k * d2 l)
        + a2 m ^ 2 * (∑ k, ∑ l, H22 m m k l * d2 k * d2 l)
        - 2 * a1 m * (∑ k, d1 k * st.f1 m k)
        - 2 * a2 m * (∑ k, d2 k * st.f2 m k) := by
  have h1 := planePhi_delta (H11 m m) (h11 m) (a1 m) (pairFblk1 H11 H12 a1 a2 st m) (st.s1 m) d1
  have h2 := planePhi_delta (H22 m m) (h22 m) (a2 m) (pairFblk2 H12 H22 a1 a2 st m) (st.s2 m) d2
  have hb := crossForm_add H12 m (st.s1 m) d1 (st.s2 m) d2
  have e1 := pairFblk1_sub H11 H12 a1 a2 st m d1
  have e2 := pairFblk2_sub H12 H22 a1 a2 st m d2
  simp only [pairPhi]
  have hexp : planePhi (H11 m m) (a1 m) (pairFblk1 H11 H12 a1 a2 st m) (fun k => st.s1 m k + d1 k)
        + 2 * (a1 m * a2 m) * crossForm H12 m (fun k => st.s1 m k + d1 k)
            (fun k => st.s2 m k + d2 k)
        + planePhi (H22 m m) (a2 m) (pairFblk2 H12 H22 a1 a2 st m) (fun k => st.s2 m k + d2 k)
        - (planePhi (H11 m m) (a1 m) (pairFblk1 H11 H12 a1 a2 st m) (st.s1 m)
          + 2 * (a1 m * a2 m) * crossForm H12 m (st.s1 m) (st.s2 m)
          + planePhi (H22 m m) (a2 m) (pairFblk2 H12 H22 a1 a2 st m) (st.s2 m))
      = (planePhi (H11 m m) (a1 m) (pairFblk1 H11 H12 a1 a2 st m) (fun k => st.s1 m k + d1 k)
          - planePhi (H11 m m) (a1 m) (pairFblk1 H11 H12 a1 a2 st m) (st.s1 m))
        + 2 * (a1 m * a2 m) * (crossForm H12 m (fun k => st.s1 m k + d1 k)
            (fun k => st.s2 m k + d2 k) - crossForm H12 m (st.s1 m) (st.s2 m))
        + (planePhi (H22 m m) (a2 m) (pairFblk2 H12 H22 a1 a2 st m) (fun k => st.s2 m k + d2 k)
          - planePhi (H22 m m) (a2 m) (pairFblk2 H12 H22 a1 a2 st m) (st.s2 m)) := by
    ring
  rw [hexp, h1, h2, hb, e1, e2]
  simp only [crossForm]
  ring




/-- **The joint row delta.** Changing both planes of block `m` by `d₁, d₂` changes the joint
objective by `a₁²d₁ᵀH₁₁d₁ + 2a₁a₂d₁ᵀH₁₂d₂ + a₂²d₂ᵀH₂₂d₂ − 2a₁d₁ᵀf₁ − 2a₂d₂ᵀf₂`, under the joint
gradient invariant.

`H₁₁` and `H₂₂` are metric blocks and are symmetric; `H₁₂` is not, and no symmetry of it is used
(`H₂₁ = H₁₂ᵀ` is its definition). -/
theorem jointQuad_delta_form (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ)
    (h11 : ∀ m m' k l, H11 m m' k l = H11 m' m l k)
    (h22 : ∀ m m' k l, H22 m m' k l = H22 m' m l k) (a1 a2 : ι → ℝ) (x1 x2 : ι → Fin 8 → ℝ)
    (st : TwoState ι) (m : ι) (d1 d2 : Fin 8 → ℝ)
    (hgrad : JointGradState H11 H12 H22 a1 a2 x1 x2 st) :
    jointQuad H11 H12 H22 a1 a2 x1 x2 (Function.update st.s1 m (fun k => st.s1 m k + d1 k))
        (Function.update st.s2 m (fun k => st.s2 m k + d2 k))
      - jointQuad H11 H12 H22 a1 a2 x1 x2 st.s1 st.s2
      = a1 m ^ 2 * (∑ k, ∑ l, H11 m m k l * d1 k * d1 l)
        + 2 * (a1 m * a2 m) * (∑ k, ∑ l, H12 m m k l * d1 k * d2 l)
        + a2 m ^ 2 * (∑ k, ∑ l, H22 m m k l * d2 k * d2 l)
        - 2 * a1 m * (∑ k, d1 k * st.f1 m k)
        - 2 * a2 m * (∑ k, d2 k * st.f2 m k) := by
  let r1 : Coord ι → ℝ := fun p => x1 p.1 p.2 - a1 p.1 * st.s1 p.1 p.2
  let r2 : Coord ι → ℝ := fun p => x2 p.1 p.2 - a2 p.1 * st.s2 p.1 p.2
  let δ1 : Coord ι → ℝ := fun p => if p.1 = m then -(a1 m) * d1 p.2 else 0
  let δ2 : Coord ι → ℝ := fun p => if p.1 = m then -(a2 m) * d2 p.2 else 0
  have hH11flat : ∀ p q : Coord ι, H11 p.1 q.1 p.2 q.2 = H11 q.1 p.1 q.2 p.2 :=
    fun p q => h11 p.1 q.1 p.2 q.2
  have hH22flat : ∀ p q : Coord ι, H22 p.1 q.1 p.2 q.2 = H22 q.1 p.1 q.2 p.2 :=
    fun p q => h22 p.1 q.1 p.2 q.2
  have hr1 : ∀ p : Coord ι,
      x1 p.1 p.2 - a1 p.1 * (Function.update st.s1 m (fun k => st.s1 m k + d1 k)) p.1 p.2
        = r1 p + δ1 p := by
    intro p
    by_cases hp : p.1 = m
    · have h1 : (Function.update st.s1 m (fun k => st.s1 m k + d1 k)) p.1 p.2
          = st.s1 m p.2 + d1 p.2 := by simp [Function.update, hp]
      have h2 : st.s1 m p.2 = st.s1 p.1 p.2 := by rw [hp]
      have h3 : a1 m = a1 p.1 := by rw [hp]
      rw [h1]
      simp only [δ1, if_pos hp, r1, h2, h3]
      ring
    · have h1 : (Function.update st.s1 m (fun k => st.s1 m k + d1 k)) p.1 p.2
          = st.s1 p.1 p.2 := by simp [Function.update, hp]
      rw [h1]
      simp only [δ1, if_neg hp, r1]
      ring
  have hr2 : ∀ p : Coord ι,
      x2 p.1 p.2 - a2 p.1 * (Function.update st.s2 m (fun k => st.s2 m k + d2 k)) p.1 p.2
        = r2 p + δ2 p := by
    intro p
    by_cases hp : p.1 = m
    · have h1 : (Function.update st.s2 m (fun k => st.s2 m k + d2 k)) p.1 p.2
          = st.s2 m p.2 + d2 p.2 := by simp [Function.update, hp]
      have h2 : st.s2 m p.2 = st.s2 p.1 p.2 := by rw [hp]
      have h3 : a2 m = a2 p.1 := by rw [hp]
      rw [h1]
      simp only [δ2, if_pos hp, r2, h2, h3]
      ring
    · have h1 : (Function.update st.s2 m (fun k => st.s2 m k + d2 k)) p.1 p.2
          = st.s2 p.1 p.2 := by simp [Function.update, hp]
      rw [h1]
      simp only [δ2, if_neg hp, r2]
      ring
  have hqr1 : rowQuadS H11 a1 x1 st.s1
      = ∑ p : Coord ι, ∑ q : Coord ι, H11 p.1 q.1 p.2 q.2 * r1 p * r1 q := by
    simp only [rowQuadS, r1]
    refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
    ring
  have hqr1' : rowQuadS H11 a1 x1 (Function.update st.s1 m (fun k => st.s1 m k + d1 k))
      = ∑ p : Coord ι, ∑ q : Coord ι,
          H11 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r1 q + δ1 q) := by
    simp only [rowQuadS]
    refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
    have hp := hr1 p
    have hq := hr1 q
    have hp' : a1 p.1 * (Function.update st.s1 m (fun k => st.s1 m k + d1 k)) p.1 p.2
          - x1 p.1 p.2 = -(r1 p + δ1 p) := by linarith
    have hq' : a1 q.1 * (Function.update st.s1 m (fun k => st.s1 m k + d1 k)) q.1 q.2
          - x1 q.1 q.2 = -(r1 q + δ1 q) := by linarith
    calc H11 p.1 q.1 p.2 q.2
          * (a1 p.1 * (Function.update st.s1 m (fun k => st.s1 m k + d1 k)) p.1 p.2 - x1 p.1 p.2)
          * (a1 q.1 * (Function.update st.s1 m (fun k => st.s1 m k + d1 k)) q.1 q.2 - x1 q.1 q.2)
        = H11 p.1 q.1 p.2 q.2 * (-(r1 p + δ1 p)) * (-(r1 q + δ1 q)) := by rw [hp', hq']
      _ = H11 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r1 q + δ1 q) := by ring
  have hqr2 : rowQuadS H22 a2 x2 st.s2
      = ∑ p : Coord ι, ∑ q : Coord ι, H22 p.1 q.1 p.2 q.2 * r2 p * r2 q := by
    simp only [rowQuadS, r2]
    refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
    ring
  have hqr2' : rowQuadS H22 a2 x2 (Function.update st.s2 m (fun k => st.s2 m k + d2 k))
      = ∑ p : Coord ι, ∑ q : Coord ι,
          H22 p.1 q.1 p.2 q.2 * (r2 p + δ2 p) * (r2 q + δ2 q) := by
    simp only [rowQuadS]
    refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
    have hp := hr2 p
    have hq := hr2 q
    have hp' : a2 p.1 * (Function.update st.s2 m (fun k => st.s2 m k + d2 k)) p.1 p.2
          - x2 p.1 p.2 = -(r2 p + δ2 p) := by linarith
    have hq' : a2 q.1 * (Function.update st.s2 m (fun k => st.s2 m k + d2 k)) q.1 q.2
          - x2 q.1 q.2 = -(r2 q + δ2 q) := by linarith
    calc H22 p.1 q.1 p.2 q.2
          * (a2 p.1 * (Function.update st.s2 m (fun k => st.s2 m k + d2 k)) p.1 p.2 - x2 p.1 p.2)
          * (a2 q.1 * (Function.update st.s2 m (fun k => st.s2 m k + d2 k)) q.1 q.2 - x2 q.1 q.2)
        = H22 p.1 q.1 p.2 q.2 * (-(r2 p + δ2 p)) * (-(r2 q + δ2 q)) := by rw [hp', hq']
      _ = H22 p.1 q.1 p.2 q.2 * (r2 p + δ2 p) * (r2 q + δ2 q) := by ring
  have hcq : crossQuad H12 a1 a2 x1 x2 st.s1 st.s2
      = ∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * r1 p * r2 q := by
    simp only [crossQuad, r1, r2]
    refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
    ring
  have hcq' : crossQuad H12 a1 a2 x1 x2
        (Function.update st.s1 m (fun k => st.s1 m k + d1 k))
        (Function.update st.s2 m (fun k => st.s2 m k + d2 k))
      = ∑ p : Coord ι, ∑ q : Coord ι,
          H12 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r2 q + δ2 q) := by
    simp only [crossQuad]
    refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
    have hp := hr1 p
    have hq := hr2 q
    have hp' : a1 p.1 * (Function.update st.s1 m (fun k => st.s1 m k + d1 k)) p.1 p.2
          - x1 p.1 p.2 = -(r1 p + δ1 p) := by linarith
    have hq' : a2 q.1 * (Function.update st.s2 m (fun k => st.s2 m k + d2 k)) q.1 q.2
          - x2 q.1 q.2 = -(r2 q + δ2 q) := by linarith
    calc H12 p.1 q.1 p.2 q.2
          * (a1 p.1 * (Function.update st.s1 m (fun k => st.s1 m k + d1 k)) p.1 p.2 - x1 p.1 p.2)
          * (a2 q.1 * (Function.update st.s2 m (fun k => st.s2 m k + d2 k)) q.1 q.2 - x2 q.1 q.2)
        = H12 p.1 q.1 p.2 q.2 * (-(r1 p + δ1 p)) * (-(r2 q + δ2 q)) := by rw [hp', hq']
      _ = H12 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r2 q + δ2 q) := by ring
  have hq1 : jointQuad H11 H12 H22 a1 a2 x1 x2 st.s1 st.s2
      = (∑ p : Coord ι, ∑ q : Coord ι, H11 p.1 q.1 p.2 q.2 * r1 p * r1 q)
        + 2 * (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * r1 p * r2 q)
        + ∑ p : Coord ι, ∑ q : Coord ι, H22 p.1 q.1 p.2 q.2 * r2 p * r2 q := by
    simp only [jointQuad, hqr1, hcq, hqr2]
  have hq2 : jointQuad H11 H12 H22 a1 a2 x1 x2
        (Function.update st.s1 m (fun k => st.s1 m k + d1 k))
        (Function.update st.s2 m (fun k => st.s2 m k + d2 k))
      = (∑ p : Coord ι, ∑ q : Coord ι,
            H11 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r1 q + δ1 q))
        + 2 * (∑ p : Coord ι, ∑ q : Coord ι,
            H12 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r2 q + δ2 q))
        + ∑ p : Coord ι, ∑ q : Coord ι,
            H22 p.1 q.1 p.2 q.2 * (r2 p + δ2 p) * (r2 q + δ2 q) := by
    simp only [jointQuad, hqr1', hcq', hqr2']
  have hG1 : ∀ k : Fin 8, (∑ p : Coord ι, H11 m p.1 k p.2 * r1 p)
      + (∑ p : Coord ι, H12 m p.1 k p.2 * r2 p) = st.f1 m k := by
    intro k
    have h := congrFun (congrFun hgrad.1 m) k
    simp only [jointGrad1] at h
    rw [← h]
  have hG2 : ∀ k : Fin 8, (∑ p : Coord ι, H12 p.1 m p.2 k * r1 p)
      + (∑ p : Coord ι, H22 m p.1 k p.2 * r2 p) = st.f2 m k := by
    intro k
    have h := congrFun (congrFun hgrad.2 m) k
    simp only [jointGrad2, H21] at h
    rw [← h]
  have hL1 : (∑ q : Coord ι, δ1 q * (∑ p : Coord ι, H11 q.1 p.1 q.2 p.2 * r1 p))
      = -(a1 m) * ∑ k, d1 k * (∑ p : Coord ι, H11 m p.1 k p.2 * r1 p) := by
    have h := sum_fiber (F := fun q : Coord ι => ∑ p : Coord ι, H11 q.1 p.1 q.2 p.2 * r1 p) m
      (fun k => -(a1 m) * d1 k)
    simp only [δ1] at h ⊢
    rw [h, Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    ring
  have hL2 : (∑ q : Coord ι, δ2 q * (∑ p : Coord ι, H22 q.1 p.1 q.2 p.2 * r2 p))
      = -(a2 m) * ∑ k, d2 k * (∑ p : Coord ι, H22 m p.1 k p.2 * r2 p) := by
    have h := sum_fiber (F := fun q : Coord ι => ∑ p : Coord ι, H22 q.1 p.1 q.2 p.2 * r2 p) m
      (fun k => -(a2 m) * d2 k)
    simp only [δ2] at h ⊢
    rw [h, Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    ring
  have hC1 : (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * r1 p * δ2 q)
      = -(a2 m) * ∑ k, d2 k * (∑ p : Coord ι, H12 p.1 m p.2 k * r1 p) := by
    have hstep : (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * r1 p * δ2 q)
        = ∑ q : Coord ι, δ2 q * (∑ p : Coord ι, H12 p.1 q.1 p.2 q.2 * r1 p) := by
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun q _ => ?_
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun p _ => by ring
    rw [hstep]
    have h := sum_fiber (F := fun q : Coord ι => ∑ p : Coord ι, H12 p.1 q.1 p.2 q.2 * r1 p) m
      (fun k => -(a2 m) * d2 k)
    simp only [δ2] at h ⊢
    rw [h, Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    ring
  have hC2 : (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * δ1 p * r2 q)
      = -(a1 m) * ∑ k, d1 k * (∑ q : Coord ι, H12 m q.1 k q.2 * r2 q) := by
    have hstep : (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * δ1 p * r2 q)
        = ∑ p : Coord ι, δ1 p * (∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * r2 q) := by
      refine Finset.sum_congr rfl fun p _ => ?_
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun q _ => by ring
    rw [hstep]
    have h := sum_fiber (F := fun p : Coord ι => ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * r2 q) m
      (fun k => -(a1 m) * d1 k)
    simp only [δ1] at h ⊢
    rw [h, Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    ring
  have hC12 : (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * δ1 p * δ2 q)
      = (a1 m * a2 m) * ∑ k, ∑ l, H12 m m k l * d1 k * d2 l := by
    have hstep : (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * δ1 p * δ2 q)
        = ∑ p : Coord ι, δ1 p * (∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * δ2 q) := by
      refine Finset.sum_congr rfl fun p _ => ?_
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun q _ => by ring
    have hq : ∀ p : Coord ι, (∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * δ2 q)
        = -(a2 m) * ∑ l, d2 l * H12 p.1 m p.2 l := by
      intro p
      have hcomm : (∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * δ2 q)
          = ∑ q : Coord ι, δ2 q * H12 p.1 q.1 p.2 q.2 := by
        refine Finset.sum_congr rfl fun q _ => ?_
        ring
      rw [hcomm]
      have h := sum_fiber (F := fun q : Coord ι => H12 p.1 q.1 p.2 q.2) m
        (fun k => -(a2 m) * d2 k)
      simp only [δ2] at h ⊢
      rw [h, Finset.mul_sum]
      refine Finset.sum_congr rfl fun l _ => ?_
      ring
    rw [hstep]
    simp only [hq]
    have h := sum_fiber (F := fun p : Coord ι => -(a2 m) * ∑ l, d2 l * H12 p.1 m p.2 l) m
      (fun k => -(a1 m) * d1 k)
    simp only [δ1] at h ⊢
    rw [h]
    simp only [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => ?_
    ring
  have hQ1 : (∑ p : Coord ι, ∑ q : Coord ι, H11 p.1 q.1 p.2 q.2 * δ1 p * δ1 q)
      = a1 m ^ 2 * ∑ k, ∑ l, H11 m m k l * d1 k * d1 l := by
    have h := sum_fiber₂ (F := fun p q : Coord ι => H11 p.1 q.1 p.2 q.2) m
      (fun k => -(a1 m) * d1 k)
    have hδ : (∑ p : Coord ι, ∑ q : Coord ι, H11 p.1 q.1 p.2 q.2 * δ1 p * δ1 q)
        = ∑ p : Coord ι, ∑ q : Coord ι,
            (if p.1 = m then -(a1 m) * d1 p.2 else 0)
              * (if q.1 = m then -(a1 m) * d1 q.2 else 0) * H11 p.1 q.1 p.2 q.2 := by
      refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
      simp only [δ1]
      ring
    rw [hδ, h]
    simp only [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => ?_
    ring
  have hQ2 : (∑ p : Coord ι, ∑ q : Coord ι, H22 p.1 q.1 p.2 q.2 * δ2 p * δ2 q)
      = a2 m ^ 2 * ∑ k, ∑ l, H22 m m k l * d2 k * d2 l := by
    have h := sum_fiber₂ (F := fun p q : Coord ι => H22 p.1 q.1 p.2 q.2) m
      (fun k => -(a2 m) * d2 k)
    have hδ : (∑ p : Coord ι, ∑ q : Coord ι, H22 p.1 q.1 p.2 q.2 * δ2 p * δ2 q)
        = ∑ p : Coord ι, ∑ q : Coord ι,
            (if p.1 = m then -(a2 m) * d2 p.2 else 0)
              * (if q.1 = m then -(a2 m) * d2 q.2 else 0) * H22 p.1 q.1 p.2 q.2 := by
      refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
      simp only [δ2]
      ring
    rw [hδ, h]
    simp only [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => ?_
    ring
  have hsum1 : (∑ k, d1 k * (∑ p : Coord ι, H11 m p.1 k p.2 * r1 p))
      + (∑ k, d1 k * (∑ q : Coord ι, H12 m q.1 k q.2 * r2 q))
      = ∑ k, d1 k * st.f1 m k := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [← hG1 k]
    ring
  have hsum2 : (∑ k, d2 k * (∑ p : Coord ι, H12 p.1 m p.2 k * r1 p))
      + (∑ k, d2 k * (∑ q : Coord ι, H22 m q.1 k q.2 * r2 q))
      = ∑ k, d2 k * st.f2 m k := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [← hG2 k]
    ring
  have hd11 := quad_delta (fun p q : Coord ι => H11 p.1 q.1 p.2 q.2) hH11flat r1 δ1
  have hd22 := quad_delta (fun p q : Coord ι => H22 p.1 q.1 p.2 q.2) hH22flat r2 δ2
  have hd12 := cross_delta H12 r1 r2 δ1 δ2
  have e11 : (∑ p : Coord ι, ∑ q : Coord ι,
        H11 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r1 q + δ1 q))
      - (∑ p : Coord ι, ∑ q : Coord ι, H11 p.1 q.1 p.2 q.2 * r1 p * r1 q)
      = 2 * (∑ q : Coord ι, δ1 q * (∑ p : Coord ι, H11 q.1 p.1 q.2 p.2 * r1 p))
        + ∑ p : Coord ι, ∑ q : Coord ι, H11 p.1 q.1 p.2 q.2 * δ1 p * δ1 q := by
    linarith [hd11]
  have e22 : (∑ p : Coord ι, ∑ q : Coord ι,
        H22 p.1 q.1 p.2 q.2 * (r2 p + δ2 p) * (r2 q + δ2 q))
      - (∑ p : Coord ι, ∑ q : Coord ι, H22 p.1 q.1 p.2 q.2 * r2 p * r2 q)
      = 2 * (∑ q : Coord ι, δ2 q * (∑ p : Coord ι, H22 q.1 p.1 q.2 p.2 * r2 p))
        + ∑ p : Coord ι, ∑ q : Coord ι, H22 p.1 q.1 p.2 q.2 * δ2 p * δ2 q := by
    linarith [hd22]
  have e12 : 2 * (∑ p : Coord ι, ∑ q : Coord ι,
        H12 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r2 q + δ2 q))
      - 2 * (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * r1 p * r2 q)
      = 2 * (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * r1 p * δ2 q)
        + 2 * (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * δ1 p * r2 q)
        + 2 * (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * δ1 p * δ2 q) := hd12
  rw [hq2, hq1]
  have hexp : ((∑ p : Coord ι, ∑ q : Coord ι,
          H11 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r1 q + δ1 q))
        + 2 * (∑ p : Coord ι, ∑ q : Coord ι,
          H12 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r2 q + δ2 q))
        + ∑ p : Coord ι, ∑ q : Coord ι,
          H22 p.1 q.1 p.2 q.2 * (r2 p + δ2 p) * (r2 q + δ2 q))
      - ((∑ p : Coord ι, ∑ q : Coord ι, H11 p.1 q.1 p.2 q.2 * r1 p * r1 q)
        + 2 * (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * r1 p * r2 q)
        + ∑ p : Coord ι, ∑ q : Coord ι, H22 p.1 q.1 p.2 q.2 * r2 p * r2 q)
      = ((∑ p : Coord ι, ∑ q : Coord ι,
          H11 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r1 q + δ1 q))
          - (∑ p : Coord ι, ∑ q : Coord ι, H11 p.1 q.1 p.2 q.2 * r1 p * r1 q))
        + (2 * (∑ p : Coord ι, ∑ q : Coord ι,
          H12 p.1 q.1 p.2 q.2 * (r1 p + δ1 p) * (r2 q + δ2 q))
          - 2 * (∑ p : Coord ι, ∑ q : Coord ι, H12 p.1 q.1 p.2 q.2 * r1 p * r2 q))
        + ((∑ p : Coord ι, ∑ q : Coord ι,
          H22 p.1 q.1 p.2 q.2 * (r2 p + δ2 p) * (r2 q + δ2 q))
          - (∑ p : Coord ι, ∑ q : Coord ι, H22 p.1 q.1 p.2 q.2 * r2 p * r2 q)) := by
    ring
  rw [hexp, e11, e12, e22, hL1, hL2, hC1, hC2, hC12, hQ1, hQ2, ← hsum1, ← hsum2]
  ring

/-- **The joint update identity.** Moving both planes of block `m` to `(v₁, v₂)` changes the joint
objective by exactly the pair cost difference `pairPhi v₁ v₂ − pairPhi s₁ₘ s₂ₘ`. -/
theorem jointQuad_update_delta (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ)
    (h11 : ∀ m m' k l, H11 m m' k l = H11 m' m l k)
    (h22 : ∀ m m' k l, H22 m m' k l = H22 m' m l k) (a1 a2 : ι → ℝ) (x1 x2 : ι → Fin 8 → ℝ)
    (st : TwoState ι) (m : ι) (v1 v2 : Fin 8 → ℝ)
    (hgrad : JointGradState H11 H12 H22 a1 a2 x1 x2 st) :
    jointQuad H11 H12 H22 a1 a2 x1 x2 (Function.update st.s1 m v1) (Function.update st.s2 m v2)
      - jointQuad H11 H12 H22 a1 a2 x1 x2 st.s1 st.s2
      = pairPhi H11 H12 H22 a1 a2 st m v1 v2
        - pairPhi H11 H12 H22 a1 a2 st m (st.s1 m) (st.s2 m) := by
  have hd := jointQuad_delta_form H11 H12 H22 h11 h22 a1 a2 x1 x2 st m
    (fun k => v1 k - st.s1 m k) (fun k => v2 k - st.s2 m k) hgrad
  have hp := pairPhi_delta H11 H12 H22 (fun m k l => h11 m m k l) (fun m k l => h22 m m k l)
    a1 a2 st m (fun k => v1 k - st.s1 m k) (fun k => v2 k - st.s2 m k)
  have hv1 : (fun k => st.s1 m k + (v1 k - st.s1 m k)) = v1 := by funext k; ring
  have hv2 : (fun k => st.s2 m k + (v2 k - st.s2 m k)) = v2 := by funext k; ring
  rw [hv1, hv2] at hd hp
  rw [hd, hp]




/-- The flat pair index `i1v * CODE_LEN + i2v`, `CODE_LEN² = 65536`. -/
abbrev PairIndex : Type := Fin (256 * 256)

/-- The plane-1 code of a flat pair index: `i1v = flat / CODE_LEN`. -/
def pairFst (i : PairIndex) : CodeIndex :=
  ⟨i.val / 256, by have := i.isLt; omega⟩

/-- The plane-2 code of a flat pair index: `i2v = flat % CODE_LEN`. -/
def pairSnd (i : PairIndex) : CodeIndex :=
  ⟨i.val % 256, by have := i.isLt; omega⟩

/-- The flat index of a code pair (`i1v * CODE_LEN + i2v`). -/
def pairFlat (c1 c2 : CodeIndex) : PairIndex :=
  ⟨c1.val * 256 + c2.val, by have := c1.isLt; have := c2.isLt; omega⟩

/-- `pairFst` inverts `pairFlat`. -/
theorem pairFst_pairFlat (c1 c2 : CodeIndex) : pairFst (pairFlat c1 c2) = c1 := by
  ext
  simp only [pairFst, pairFlat]
  have := c2.isLt
  omega

/-- `pairSnd` inverts `pairFlat`. -/
theorem pairSnd_pairFlat (c1 c2 : CodeIndex) : pairSnd (pairFlat c1 c2) = c2 := by
  ext
  simp only [pairSnd, pairFlat]
  have := c2.isLt
  omega

/-- `pairFlat` inverts the flat pair. -/
theorem pairFlat_pairFst (i : PairIndex) : pairFlat (pairFst i) (pairSnd i) = i := by
  ext
  simp only [pairSnd, pairFst, pairFlat]
  omega

/-- The `pair_cost` value at a flat index: `pairPhi` at the pair `(i/256, i%256)`. -/
def pairCostFlat (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ) (st : TwoState ι)
    (m : ι) (i : PairIndex) : ℝ :=
  pairPhi H11 H12 H22 a1 a2 st m (codebook (pairFst i)) (codebook (pairSnd i))

/-- The pair argmin: the least flat index attaining the minimum of the pair cost.

`best_pair_q` scans `i1v` outer, `i2v` inner and replaces only on a strict `<`, so it returns the
least `i1v * 256 + i2v` among the minimisers; `pairArgmin_le` states exactly that. -/
noncomputable def pairArgmin (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (st : TwoState ι) (m : ι) : PairIndex :=
  (Finset.univ.filter (fun i : PairIndex =>
      ∀ j : PairIndex,
        pairCostFlat H11 H12 H22 a1 a2 st m i ≤ pairCostFlat H11 H12 H22 a1 a2 st m j)).min'
    (by
      obtain ⟨i, _, hi⟩ := Finset.exists_min_image (Finset.univ : Finset PairIndex)
        (fun i => pairCostFlat H11 H12 H22 a1 a2 st m i) ⟨0, Finset.mem_univ 0⟩
      exact ⟨i, Finset.mem_filter.mpr
        ⟨Finset.mem_univ i, fun j => hi j (Finset.mem_univ j)⟩⟩)

omit [Fintype ι] [DecidableEq ι] in
/-- The chosen flat pair attains the minimum. -/
theorem pairArgmin_spec (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (st : TwoState ι) (m : ι) (j : PairIndex) :
    pairCostFlat H11 H12 H22 a1 a2 st m (pairArgmin H11 H12 H22 a1 a2 st m)
      ≤ pairCostFlat H11 H12 H22 a1 a2 st m j := by
  have hnon : (Finset.univ.filter (fun i : PairIndex => ∀ j : PairIndex,
      pairCostFlat H11 H12 H22 a1 a2 st m i ≤ pairCostFlat H11 H12 H22 a1 a2 st m j)).Nonempty := by
    obtain ⟨i, _, hi⟩ := Finset.exists_min_image (Finset.univ : Finset PairIndex)
      (fun i => pairCostFlat H11 H12 H22 a1 a2 st m i) ⟨0, Finset.mem_univ 0⟩
    exact ⟨i, Finset.mem_filter.mpr ⟨Finset.mem_univ i, fun j => hi j (Finset.mem_univ j)⟩⟩
  exact (Finset.mem_filter.mp (Finset.min'_mem _ hnon)).2 j

omit [Fintype ι] [DecidableEq ι] in
/-- **The tie rule.** No earlier flat index attains the minimum: the chosen pair is the first
minimal one, which is what the strict-`<` scan with `flat = i1v * CODE_LEN + i2v` returns. -/
theorem pairArgmin_le (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (st : TwoState ι) (m : ι) (j : PairIndex)
    (hj : pairCostFlat H11 H12 H22 a1 a2 st m j
      ≤ pairCostFlat H11 H12 H22 a1 a2 st m (pairArgmin H11 H12 H22 a1 a2 st m)) :
    pairArgmin H11 H12 H22 a1 a2 st m ≤ j := by
  refine Finset.min'_le _ _ (Finset.mem_filter.mpr ⟨Finset.mem_univ j, ?_⟩)
  intro j'
  exact le_trans hj (pairArgmin_spec H11 H12 H22 a1 a2 st m j')

omit [Fintype ι] [DecidableEq ι] in
/-- **The pair cost at codebook vectors is `pair_cost`.** With `qᵢ = codeᵢᵀHᵢᵢcodeᵢ`,
`dᵢ = codeᵢᵀ fᵢ_blk` and `X = code₁ᵀH₁₂code₂`, `pairPhi = (a₁²q₁ − 2a₁d₁) + 2a₁a₂X + (a₂²q₂ − 2a₂d₂)`
— the `av + 2·a₁a₂·x_all + bv` of `pair_cost` (`icm.rs:235`), with `x_all[i1v][i2v]` the precomputed
cross Gram (`block_cross_q`, `:836`). -/
theorem pairPhi_codebook_eq (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (st : TwoState ι) (m : ι) (c1 c2 : CodeIndex) :
    pairPhi H11 H12 H22 a1 a2 st m (codebook c1) (codebook c2)
      = (a1 m ^ 2 * (∑ k, ∑ l, H11 m m k l * codebook c1 k * codebook c1 l)
          - 2 * a1 m * (∑ k, codebook c1 k * pairFblk1 H11 H12 a1 a2 st m k))
        + 2 * (a1 m * a2 m) * (∑ k, ∑ l, H12 m m k l * codebook c1 k * codebook c2 l)
        + (a2 m ^ 2 * (∑ k, ∑ l, H22 m m k l * codebook c2 k * codebook c2 l)
          - 2 * a2 m * (∑ k, codebook c2 k * pairFblk2 H12 H22 a1 a2 st m k)) := by
  simp only [pairPhi, planePhi, crossForm]

/-- One iteration of the shipped joint block body (`joint_band`, `:1425`–`:1450`): both planes take
their pair-argmin codes, and each residual contracts **both** planes' deltas:
`f₁ -= a₁·(H₁₁Δ₁) + a₂·(H₁₂Δ₂)`, `f₂ -= a₂·(H₂₂Δ₂) + a₁·(H₂₁Δ₁)` with `H₂₁ = H₁₂ᵀ`. -/
noncomputable def pairBlockStep (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (st : TwoState ι) (m : ι) : TwoState ι :=
  { s1 := Function.update st.s1 m (codebook (pairFst (pairArgmin H11 H12 H22 a1 a2 st m)))
    s2 := Function.update st.s2 m (codebook (pairSnd (pairArgmin H11 H12 H22 a1 a2 st m)))
    f1 := fun m' k =>
      st.f1 m' k
        - a1 m * (∑ l, H11 m' m k l
            * (codebook (pairFst (pairArgmin H11 H12 H22 a1 a2 st m)) l - st.s1 m l))
        - a2 m * (∑ l, H12 m' m k l
            * (codebook (pairSnd (pairArgmin H11 H12 H22 a1 a2 st m)) l - st.s2 m l))
    f2 := fun m' k =>
      st.f2 m' k
        - a2 m * (∑ l, H22 m' m k l
            * (codebook (pairSnd (pairArgmin H11 H12 H22 a1 a2 st m)) l - st.s2 m l))
        - a1 m * (∑ l, H21 H12 m' m k l
            * (codebook (pairFst (pairArgmin H11 H12 H22 a1 a2 st m)) l - st.s1 m l)) }

/-- **The pair step never raises the joint objective.** -/
theorem jointQuad_step_le (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ)
    (h11 : ∀ m m' k l, H11 m m' k l = H11 m' m l k)
    (h22 : ∀ m m' k l, H22 m m' k l = H22 m' m l k) (a1 a2 : ι → ℝ) (x1 x2 : ι → Fin 8 → ℝ)
    (st : TwoState ι) (m : ι) (hgrad : JointGradState H11 H12 H22 a1 a2 x1 x2 st)
    (hcode : TwoCodeState st) :
    jointQuad H11 H12 H22 a1 a2 x1 x2 (pairBlockStep H11 H12 H22 a1 a2 st m).s1
        (pairBlockStep H11 H12 H22 a1 a2 st m).s2
      ≤ jointQuad H11 H12 H22 a1 a2 x1 x2 st.s1 st.s2 := by
  obtain ⟨c1, hc1⟩ := hcode.1 m
  obtain ⟨c2, hc2⟩ := hcode.2 m
  have hd := jointQuad_update_delta H11 H12 H22 h11 h22 a1 a2 x1 x2 st m
    (codebook (pairFst (pairArgmin H11 H12 H22 a1 a2 st m)))
    (codebook (pairSnd (pairArgmin H11 H12 H22 a1 a2 st m))) hgrad
  have hspec := pairArgmin_spec H11 H12 H22 a1 a2 st m (pairFlat c1 c2)
  simp only [pairCostFlat, pairFst_pairFlat, pairSnd_pairFlat] at hspec
  have hle : pairPhi H11 H12 H22 a1 a2 st m
      (codebook (pairFst (pairArgmin H11 H12 H22 a1 a2 st m)))
      (codebook (pairSnd (pairArgmin H11 H12 H22 a1 a2 st m)))
      ≤ pairPhi H11 H12 H22 a1 a2 st m (codebook c1) (codebook c2) := hspec
  have hphi : pairPhi H11 H12 H22 a1 a2 st m (st.s1 m) (st.s2 m)
      = pairPhi H11 H12 H22 a1 a2 st m (codebook c1) (codebook c2) := by
    rw [hc1, hc2]
  have hgoal : jointQuad H11 H12 H22 a1 a2 x1 x2 (pairBlockStep H11 H12 H22 a1 a2 st m).s1
      (pairBlockStep H11 H12 H22 a1 a2 st m).s2
      = jointQuad H11 H12 H22 a1 a2 x1 x2
          (Function.update st.s1 m (codebook (pairFst (pairArgmin H11 H12 H22 a1 a2 st m))))
          (Function.update st.s2 m (codebook (pairSnd (pairArgmin H11 H12 H22 a1 a2 st m)))) := rfl
  rw [hgoal]
  linarith [hd, hle, hphi]

/-- The cross-plane gradient step: a block update of `s` changes `K·(x − a·s)` by
`−a_m·K_{m'm}·Δ`. This is the single step both planes' residuals perform, with `K = H₁₁`, `H₁₂`,
`H₂₂`, `H₂₁`. -/
theorem grad_step_generic (K : ι → ι → Fin 8 → Fin 8 → ℝ) (a : ι → ℝ) (x s : ι → Fin 8 → ℝ)
    (m : ι) (v : Fin 8 → ℝ) (m' : ι) (k : Fin 8) :
    (∑ q : Coord ι, K m' q.1 k q.2
        * (x q.1 q.2 - a q.1 * (Function.update s m v) q.1 q.2))
      = (∑ q : Coord ι, K m' q.1 k q.2 * (x q.1 q.2 - a q.1 * s q.1 q.2))
        - a m * ∑ l, K m' m k l * (v l - s m l) := by
  have hupd : ∀ q : Coord ι,
      (Function.update s m v) q.1 q.2
        = s q.1 q.2 + (if q.1 = m then v q.2 - s m q.2 else 0) := by
    intro q
    rw [Function.update_apply]
    by_cases hq : q.1 = m
    · subst hq; simp
    · simp [hq]
  have hR : (∑ q : Coord ι, K m' q.1 k q.2 * (x q.1 q.2 - a q.1 * (Function.update s m v) q.1 q.2))
      = (∑ q : Coord ι, K m' q.1 k q.2 * (x q.1 q.2 - a q.1 * s q.1 q.2))
        + ∑ q : Coord ι, K m' q.1 k q.2
            * (if q.1 = m then -(a m) * (v q.2 - s m q.2) else 0) := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun q _ => ?_
    rw [hupd q]
    by_cases hq : q.1 = m
    · subst hq; simp; ring
    · simp [hq]
  rw [hR]
  have hfib := sum_fiber (F := fun q : Coord ι => K m' q.1 k q.2) m
    (fun l => -(a m) * (v l - s m l))
  have hswap : (∑ q : Coord ι, K m' q.1 k q.2
        * (if q.1 = m then -(a m) * (v q.2 - s m q.2) else 0))
      = ∑ q : Coord ι,
          (if q.1 = m then -(a m) * (v q.2 - s m q.2) else 0) * K m' q.1 k q.2 := by
    refine Finset.sum_congr rfl fun q _ => ?_
    ring
  rw [hswap, hfib]
  have hsum : (∑ l : Fin 8, (-(a m) * (v l - s m l)) * K m' m k l)
      = -(a m) * ∑ l, K m' m k l * (v l - s m l) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun l _ => ?_
    ring
  rw [hsum]
  ring

/-- The pair step preserves the joint gradient invariant. -/
theorem jointGradState_step (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (x1 x2 : ι → Fin 8 → ℝ) (st : TwoState ι) (m : ι)
    (hgrad : JointGradState H11 H12 H22 a1 a2 x1 x2 st) :
    JointGradState H11 H12 H22 a1 a2 x1 x2 (pairBlockStep H11 H12 H22 a1 a2 st m) := by
  obtain ⟨hg1, hg2⟩ := hgrad
  constructor
  · funext m' k
    have h1 := grad_step_generic H11 a1 x1 st.s1 m
      (codebook (pairFst (pairArgmin H11 H12 H22 a1 a2 st m))) m' k
    have h2 := grad_step_generic H12 a2 x2 st.s2 m
      (codebook (pairSnd (pairArgmin H11 H12 H22 a1 a2 st m))) m' k
    have h3 := congrFun (congrFun hg1 m') k
    simp only [jointGrad1] at h3
    simp only [pairBlockStep, jointGrad1]
    rw [h1, h2, h3]
    ring
  · funext m' k
    have h1 := grad_step_generic (H21 H12) a1 x1 st.s1 m
      (codebook (pairFst (pairArgmin H11 H12 H22 a1 a2 st m))) m' k
    have h2 := grad_step_generic H22 a2 x2 st.s2 m
      (codebook (pairSnd (pairArgmin H11 H12 H22 a1 a2 st m))) m' k
    have h3 := congrFun (congrFun hg2 m') k
    simp only [jointGrad2] at h3
    simp only [pairBlockStep, jointGrad2]
    rw [h1, h2, h3]
    ring

omit [Fintype ι] in
/-- The pair step keeps every block of both planes on a codebook row. -/
theorem twoCodeState_step (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (st : TwoState ι) (m : ι) (h : TwoCodeState st) :
    TwoCodeState (pairBlockStep H11 H12 H22 a1 a2 st m) := by
  obtain ⟨h1, h2⟩ := h
  constructor
  · intro m'
    by_cases hm' : m' = m
    · rw [hm']
      exact ⟨pairFst (pairArgmin H11 H12 H22 a1 a2 st m), by simp [pairBlockStep]⟩
    · obtain ⟨c, hc⟩ := h1 m'
      exact ⟨c, by simpa [pairBlockStep, Function.update, hm'] using hc⟩
  · intro m'
    by_cases hm' : m' = m
    · rw [hm']
      exact ⟨pairSnd (pairArgmin H11 H12 H22 a1 a2 st m), by simp [pairBlockStep]⟩
    · obtain ⟨c, hc⟩ := h2 m'
      exact ⟨c, by simpa [pairBlockStep, Function.update, hm'] using hc⟩

/-- The pair pass invariant: the joint gradients are true and every block of both planes is a
codebook row. -/
def TwoInv (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (x1 x2 : ι → Fin 8 → ℝ) (st : TwoState ι) : Prop :=
  JointGradState H11 H12 H22 a1 a2 x1 x2 st ∧ TwoCodeState st

/-- The pair step preserves the pass invariant. -/
theorem twoInv_step (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (x1 x2 : ι → Fin 8 → ℝ) (st : TwoState ι) (m : ι) (h : TwoInv H11 H12 H22 a1 a2 x1 x2 st) :
    TwoInv H11 H12 H22 a1 a2 x1 x2 (pairBlockStep H11 H12 H22 a1 a2 st m) :=
  ⟨jointGradState_step H11 H12 H22 a1 a2 x1 x2 st m h.1,
    twoCodeState_step H11 H12 H22 a1 a2 st m h.2⟩

/-- The joint pass: `joint_band`'s block loop, folding `pairBlockStep` over the block order. -/
noncomputable def pairPass (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ) (a1 a2 : ι → ℝ)
    (order : List ι) (st : TwoState ι) : TwoState ι :=
  order.foldl (fun st m => pairBlockStep H11 H12 H22 a1 a2 st m) st

/-- A joint pass preserves the invariant and does not increase the joint objective. -/
theorem pairPass_inv_le (H11 H12 H22 : ι → ι → Fin 8 → Fin 8 → ℝ)
    (h11 : ∀ m m' k l, H11 m m' k l = H11 m' m l k)
    (h22 : ∀ m m' k l, H22 m m' k l = H22 m' m l k) (a1 a2 : ι → ℝ) (x1 x2 : ι → Fin 8 → ℝ)
    (order : List ι) (st : TwoState ι) (h : TwoInv H11 H12 H22 a1 a2 x1 x2 st) :
    TwoInv H11 H12 H22 a1 a2 x1 x2 (pairPass H11 H12 H22 a1 a2 order st) ∧
      jointQuad H11 H12 H22 a1 a2 x1 x2 (pairPass H11 H12 H22 a1 a2 order st).s1
          (pairPass H11 H12 H22 a1 a2 order st).s2
        ≤ jointQuad H11 H12 H22 a1 a2 x1 x2 st.s1 st.s2 := by
  have hstep : ∀ b (m : ι), TwoInv H11 H12 H22 a1 a2 x1 x2 b →
      TwoInv H11 H12 H22 a1 a2 x1 x2 (pairBlockStep H11 H12 H22 a1 a2 b m) ∧
        jointQuad H11 H12 H22 a1 a2 x1 x2 (pairBlockStep H11 H12 H22 a1 a2 b m).s1
            (pairBlockStep H11 H12 H22 a1 a2 b m).s2
          ≤ jointQuad H11 H12 H22 a1 a2 x1 x2 b.s1 b.s2 :=
    fun b m hb =>
      ⟨twoInv_step H11 H12 H22 a1 a2 x1 x2 b m hb,
        jointQuad_step_le H11 H12 H22 h11 h22 a1 a2 x1 x2 b m hb.1 hb.2⟩
  simpa only [pairPass] using foldl_inv_le (fun st m => pairBlockStep H11 H12 H22 a1 a2 st m)
    (fun st => jointQuad H11 H12 H22 a1 a2 x1 x2 st.s1 st.s2)
    (TwoInv H11 H12 H22 a1 a2 x1 x2) hstep order st h

/-- The joint pass composition: `k` sweeps, pass `j` at parity `j % 2`, the same order
`sweep_rows` uses (`blockOrder`). -/
noncomputable def pairPasses (nb : ℕ) (H11 H12 H22 : Fin nb → Fin nb → Fin 8 → Fin 8 → ℝ)
    (a1 a2 : Fin nb → ℝ) (st : TwoState (Fin nb)) : ℕ → TwoState (Fin nb)
  | 0 => st
  | j + 1 => pairPass H11 H12 H22 a1 a2 (blockOrder nb j)
      (pairPasses nb H11 H12 H22 a1 a2 st j)

/-- Every joint pass keeps the invariant. -/
theorem pairPasses_inv (nb : ℕ) (H11 H12 H22 : Fin nb → Fin nb → Fin 8 → Fin 8 → ℝ)
    (h11 : ∀ m m' k l, H11 m m' k l = H11 m' m l k)
    (h22 : ∀ m m' k l, H22 m m' k l = H22 m' m l k) (a1 a2 : Fin nb → ℝ)
    (x1 x2 : Fin nb → Fin 8 → ℝ) (st : TwoState (Fin nb))
    (h : TwoInv H11 H12 H22 a1 a2 x1 x2 st) (k : ℕ) :
    TwoInv H11 H12 H22 a1 a2 x1 x2 (pairPasses nb H11 H12 H22 a1 a2 st k) := by
  induction k with
  | zero => exact h
  | succ j ih => exact (pairPass_inv_le H11 H12 H22 h11 h22 a1 a2 x1 x2 (blockOrder nb j) _ ih).1

/-- The joint composition: `k` passes never increase the joint objective. -/
theorem pairPasses_quad_le (nb : ℕ) (H11 H12 H22 : Fin nb → Fin nb → Fin 8 → Fin 8 → ℝ)
    (h11 : ∀ m m' k l, H11 m m' k l = H11 m' m l k)
    (h22 : ∀ m m' k l, H22 m m' k l = H22 m' m l k) (a1 a2 : Fin nb → ℝ)
    (x1 x2 : Fin nb → Fin 8 → ℝ) (st : TwoState (Fin nb))
    (h : TwoInv H11 H12 H22 a1 a2 x1 x2 st) (k : ℕ) :
    jointQuad H11 H12 H22 a1 a2 x1 x2 (pairPasses nb H11 H12 H22 a1 a2 st k).s1
        (pairPasses nb H11 H12 H22 a1 a2 st k).s2
      ≤ jointQuad H11 H12 H22 a1 a2 x1 x2 st.s1 st.s2 := by
  induction k with
  | zero => exact le_rfl
  | succ j ih =>
      have hinv : TwoInv H11 H12 H22 a1 a2 x1 x2 (pairPasses nb H11 H12 H22 a1 a2 st j) :=
        pairPasses_inv nb H11 H12 H22 h11 h22 a1 a2 x1 x2 st h j
      exact le_trans (pairPass_inv_le H11 H12 H22 h11 h22 a1 a2 x1 x2 (blockOrder nb j) _ hinv).2 ih

open Sanguine.CountLMFloor (klDiv)

/-! ### The finite-temperature pair law

`best_pair_q_beta` (`src/learn/fold/fit/icm.rs:352`) draws one joint pair from the
`CODE_LEN² = 65536` cost table with `gibbs_choice`. This section states the law that draw
targets, proves the Gibbs variational principle for it, and states the correspondence to the
integer table with its error terms. -/

/-- The Boltzmann weight of a pair at inverse temperature `β`: `w_β(i) = exp(−β·c(i))`. -/
noncomputable def pairWeight (β : ℝ) (c : PairIndex → ℝ) (i : PairIndex) : ℝ :=
  Real.exp (-(β * c i))

/-- The partition function `Z_β = Σ_i w_β(i)`. -/
noncomputable def pairZ (β : ℝ) (c : PairIndex → ℝ) : ℝ := ∑ i, pairWeight β c i

/-- The Gibbs law `q_β(i) = w_β(i) / Z_β`. -/
noncomputable def pairLaw (β : ℝ) (c : PairIndex → ℝ) (i : PairIndex) : ℝ :=
  pairWeight β c i / pairZ β c

/-- `H(q) = Σ_i q_i log q_i`, the **negative** Shannon entropy: the sign convention in which the
variational principle reads `E_q[c] + (1/β)·H(q) ≥ −(1/β)·log Z`, with equality at `q = q_β`.
In the Shannon convention `H_S(q) = −Σ q log q` the same statement is
`E_q[c] − (1/β)·H_S(q) ≥ −(1/β)·log Z` (Gibbs 1902: the canonical ensemble minimises `E − TS` at
`T = 1/β`). -/
noncomputable def pairLogEntropy (q : PairIndex → ℝ) : ℝ := ∑ i, q i * Real.log (q i)

/-- The free energy at inverse temperature `β`: `F_β(q) = E_q[c] + (1/β)·H(q)`. -/
noncomputable def pairFree (β : ℝ) (c q : PairIndex → ℝ) : ℝ :=
  (∑ i, q i * c i) + (1 / β) * pairLogEntropy q

theorem pairWeight_pos (β : ℝ) (c : PairIndex → ℝ) (i : PairIndex) : 0 < pairWeight β c i :=
  Real.exp_pos _

theorem pairZ_pos (β : ℝ) (c : PairIndex → ℝ) : 0 < pairZ β c :=
  Finset.sum_pos (fun i _ => pairWeight_pos β c i) ⟨pairFlat 0 0, Finset.mem_univ _⟩

theorem pairLaw_pos (β : ℝ) (c : PairIndex → ℝ) (i : PairIndex) : 0 < pairLaw β c i :=
  div_pos (pairWeight_pos β c i) (pairZ_pos β c)

theorem pairLaw_sum_one (β : ℝ) (c : PairIndex → ℝ) : ∑ i, pairLaw β c i = 1 := by
  simp only [pairLaw]
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (pairZ_pos β c))

/-- The pointwise Gibbs step: for `q ≥ 0` and `p > 0`, `q − p ≤ q·log(q/p)`. At `q > 0` this is
`log(p/q) ≤ p/q − 1` multiplied by `q`; at `q = 0` it is `−p ≤ 0`. A vanishing `p` is excluded:
there the pointwise bound fails while the summed KL is still nonnegative, which is why `p` is the
strictly positive Boltzmann law throughout. -/
theorem klTerm_ge {q p : ℝ} (hq : 0 ≤ q) (hp : 0 < p) : q - p ≤ q * Real.log (q / p) := by
  rcases eq_or_lt_of_le hq with h | h
  · subst h
    simp only [zero_mul, sub_nonpos]
    exact le_of_lt hp
  · have hlog : Real.log (q / p) = -Real.log (p / q) := by
      rw [Real.log_div (ne_of_gt h) (ne_of_gt hp), Real.log_div (ne_of_gt hp) (ne_of_gt h)]
      ring
    have h1 : Real.log (p / q) ≤ p / q - 1 := Real.log_le_sub_one_of_pos (div_pos hp h)
    have h3 : q * Real.log (p / q) ≤ q * (p / q - 1) :=
      mul_le_mul_of_nonneg_left h1 (le_of_lt h)
    have h4 : q * (p / q - 1) = p - q := by field_simp
    have h2 : q * Real.log (q / p) = -(q * Real.log (p / q)) := by rw [hlog]; ring
    rw [h2]
    linarith

/-- **KL ≥ 0 on the pair simplex, with a vanishing proposal allowed.** `0 ≤ Σ q log(q/p)` for
`q ≥ 0`, `Σ q = 1`, `p > 0`, `Σ p = 1`. This strictly generalises
`Sanguine.CountLMFloor.kl_nonneg` (which asks `0 < q` everywhere); see
`klDiv_nonneg_of_pos` for the direct link. -/
theorem klDiv_nonneg_of_nonneg (q p : PairIndex → ℝ) (hp : ∀ i, 0 < p i)
    (hq : ∀ i, 0 ≤ q i) (hq1 : ∑ i, q i = 1) (hp1 : ∑ i, p i = 1) : 0 ≤ klDiv q p := by
  have h : ∑ i, (q i - p i) ≤ ∑ i, q i * Real.log (q i / p i) :=
    Finset.sum_le_sum (fun i _ => klTerm_ge (hq i) (hp i))
  have hs : ∑ i, (q i - p i) = 0 := by
    rw [Finset.sum_sub_distrib, hq1, hp1]; ring
  unfold klDiv
  linarith

/-- The corpus's Gibbs inequality, instantiated at the pair law: the `0 < q` case of
`klDiv_nonneg_of_nonneg`, proved by `Sanguine.CountLMFloor.kl_nonneg`. -/
theorem klDiv_nonneg_of_pos (q p : PairIndex → ℝ) (hp : ∀ i, 0 < p i)
    (hq : ∀ i, 0 < q i) (hq1 : ∑ i, q i = 1) (hp1 : ∑ i, p i = 1) : 0 ≤ klDiv q p :=
  Sanguine.CountLMFloor.kl_nonneg q p hq hp hq1 hp1

/-- `log q_β(i) = −β·c(i) − log Z_β`: the log of a Gibbs weight. -/
theorem log_pairLaw (β : ℝ) (c : PairIndex → ℝ) (i : PairIndex) :
    Real.log (pairLaw β c i) = -(β * c i) - Real.log (pairZ β c) := by
  rw [pairLaw, Real.log_div (ne_of_gt (pairWeight_pos β c i)) (ne_of_gt (pairZ_pos β c)),
    pairWeight, Real.log_exp]

/-- **The KL against the Gibbs law.** `KL(q‖q_β) = Σ q log q + β·Σ q c + log Z_β` at `Σ q = 1`. -/
theorem klDiv_pairLaw (β : ℝ) (c q : PairIndex → ℝ) (hq1 : ∑ i, q i = 1) :
    klDiv q (pairLaw β c)
      = (∑ i, q i * Real.log (q i)) + β * (∑ i, q i * c i) + Real.log (pairZ β c) := by
  have hterm : ∀ i : PairIndex,
      q i * Real.log (q i / pairLaw β c i)
        = q i * Real.log (q i) - q i * Real.log (pairLaw β c i) := by
    intro i
    have hp := pairLaw_pos β c i
    rcases eq_or_ne (q i) 0 with hq | hq
    · simp [hq]
    · rw [Real.log_div hq (ne_of_gt hp)]
      ring
  have hlog : ∀ i : PairIndex,
      q i * Real.log (pairLaw β c i) = -(q i * (β * c i)) - q i * Real.log (pairZ β c) := by
    intro i
    rw [log_pairLaw]
    ring
  unfold klDiv
  rw [Finset.sum_congr rfl (fun i _ => hterm i)]
  rw [Finset.sum_sub_distrib]
  rw [Finset.sum_congr rfl (fun i _ => hlog i)]
  rw [Finset.sum_sub_distrib, Finset.sum_neg_distrib, ← Finset.sum_mul, hq1, one_mul]
  have hsplit : ∑ i, q i * (β * c i) = β * ∑ i, q i * c i := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun i _ => by ring)
  rw [hsplit]
  ring

/-- **The Gibbs identity in scaled form.** `β·F_β(q) = KL(q‖q_β) − log Z_β`. -/
theorem pairFree_mul_beta (β : ℝ) (c q : PairIndex → ℝ) (hβ : β ≠ 0) (hq1 : ∑ i, q i = 1) :
    β * pairFree β c q = klDiv q (pairLaw β c) - Real.log (pairZ β c) := by
  have hmul : β * pairFree β c q = β * (∑ i, q i * c i) + pairLogEntropy q := by
    unfold pairFree
    rw [mul_add, ← mul_assoc, one_div, mul_inv_cancel₀ hβ, one_mul]
  rw [hmul, klDiv_pairLaw β c q hq1]
  unfold pairLogEntropy
  ring

/-- `KL(q_β‖q_β) = 0`: the pointwise term is `q·log 1`. -/
theorem klDiv_pairLaw_self (β : ℝ) (c : PairIndex → ℝ) : klDiv (pairLaw β c) (pairLaw β c) = 0 := by
  unfold klDiv
  rw [Finset.sum_eq_zero]
  intro i _
  rw [div_self (ne_of_gt (pairLaw_pos β c i)), Real.log_one, mul_zero]

/-- **The free energy of the Gibbs law.** `β·F_β(q_β) = −log Z_β`. -/
theorem pairFree_law_mul_beta (β : ℝ) (c : PairIndex → ℝ) (hβ : β ≠ 0) :
    β * pairFree β c (pairLaw β c) = -Real.log (pairZ β c) := by
  rw [pairFree_mul_beta β c (pairLaw β c) hβ (pairLaw_sum_one β c),
    klDiv_pairLaw_self, zero_sub]

/-- **The Gibbs identity.** `F_β(q) − F_β(q_β) = (1/β)·KL(q‖q_β)` for any law `q`. -/
theorem pairFree_sub_law (β : ℝ) (c q : PairIndex → ℝ) (hβ : 0 < β) (hβ0 : β ≠ 0)
    (hq1 : ∑ i, q i = 1) :
    pairFree β c q - pairFree β c (pairLaw β c) = (1 / β) * klDiv q (pairLaw β c) := by
  have hq := pairFree_mul_beta β c q hβ0 hq1
  have hl := pairFree_law_mul_beta β c hβ0
  have h1 : β * (pairFree β c q - pairFree β c (pairLaw β c)) = klDiv q (pairLaw β c) := by
    rw [mul_sub, hq, hl]; ring
  have h2 : pairFree β c q - pairFree β c (pairLaw β c)
      = klDiv q (pairLaw β c) / β := by
    rw [eq_div_iff (ne_of_gt hβ), mul_comm, h1]
  rw [h2, one_div, inv_mul_eq_div]

/-- **Equality at the Gibbs law**: the variational bound is attained by `q_β`. -/
theorem pairFree_law_eq (β : ℝ) (c : PairIndex → ℝ) (hβ : 0 < β) :
    pairFree β c (pairLaw β c) = -(1 / β) * Real.log (pairZ β c) := by
  have h := pairFree_law_mul_beta β c (ne_of_gt hβ)
  field_simp at h ⊢
  linarith

/-- **The Gibbs variational principle.** For `β > 0` and any law `q` on the `65536` pairs
(`q ≥ 0`, `Σ q = 1`), `E_q[c] + (1/β)·H(q) ≥ −(1/β)·log Z_β`, with `H(q) = Σ q log q`.
Equality holds at the Gibbs law (`pairFree_law_eq`). -/
theorem pairFree_ge (β : ℝ) (c q : PairIndex → ℝ) (hβ : 0 < β) (hq0 : ∀ i, 0 ≤ q i)
    (hq1 : ∑ i, q i = 1) :
    -(1 / β) * Real.log (pairZ β c) ≤ pairFree β c q := by
  have h0 := pairFree_sub_law β c q hβ (ne_of_gt hβ) hq1
  have hkl : 0 ≤ klDiv q (pairLaw β c) :=
    klDiv_nonneg_of_nonneg q (pairLaw β c) (pairLaw_pos β c) hq0 hq1 (pairLaw_sum_one β c)
  have hnn : 0 ≤ (1 / β) * klDiv q (pairLaw β c) :=
    mul_nonneg (le_of_lt (one_div_pos.mpr hβ)) hkl
  rw [← pairFree_law_eq β c hβ]
  linarith

/-! ### The integer table's correspondence

`gibbs_choice` (`icm.rs:454`) builds a Q32.32 weight per pair from
`x = max(−β·(cost − min), −16)`, sums them into `total`, draws `state % total` from a xorshift
state, and walks the cumulative weights. That is the law `w_i / total` with
`w_i = fixed_exp(x_i)`. The model law is `q_β`. This section states the error chain from one to
the other; every term is a hypothesis, none is assumed away. -/

/-- Scaling the inverse temperature scales the cost: `q_1(β·c) = q_β(c)`. -/
theorem pairLaw_scale (β : ℝ) (c : PairIndex → ℝ) (i : PairIndex) :
    pairLaw 1 (fun j => β * c j) i = pairLaw β c i := by
  simp only [pairLaw, pairZ, pairWeight, one_mul]

/-- Shifting the cost shifts the weight by a constant factor. -/
theorem pairWeight_shift (β a : ℝ) (c : PairIndex → ℝ) (i : PairIndex) :
    pairWeight β (fun j => c j + a) i = Real.exp (-(β * a)) * pairWeight β c i := by
  simp only [pairWeight]
  rw [show -(β * (c i + a)) = -(β * a) + -(β * c i) by ring, Real.exp_add]

/-- Adding a constant to the cost leaves `Z` shifted by that factor. -/
theorem pairZ_shift (β a : ℝ) (c : PairIndex → ℝ) :
    pairZ β (fun i => c i + a) = Real.exp (-(β * a)) * pairZ β c := by
  simp only [pairZ]
  rw [Finset.sum_congr rfl (fun i _ => pairWeight_shift β a c i), Finset.mul_sum]

/-- Adding a constant to the cost leaves the law unchanged (the factor cancels in `Z`). -/
theorem pairLaw_shift (β a : ℝ) (c : PairIndex → ℝ) (i : PairIndex) :
    pairLaw β (fun j => c j + a) i = pairLaw β c i := by
  have hk : Real.exp (-(β * a)) ≠ 0 := ne_of_gt (Real.exp_pos _)
  simp only [pairLaw, pairZ_shift, pairWeight_shift]
  exact mul_div_mul_left _ _ hk

/-- The normalised integer weight vector: `intLaw(i) = w_i / Σ_j w_j`. -/
noncomputable def intLaw (w : PairIndex → ℝ) (i : PairIndex) : ℝ := w i / ∑ j, w j

theorem intLaw_sum_one (w : PairIndex → ℝ) (hW : ∑ j, w j ≠ 0) : ∑ i, intLaw w i = 1 := by
  simp only [intLaw]
  rw [← Finset.sum_div]
  exact div_self hW

/-- **The integer weights are positive** under a relative error `ε < 1`. -/
theorem weight_pos_of_rel (β : ℝ) (c w : PairIndex → ℝ) (ε : ℝ) (_hε0 : 0 ≤ ε) (hε1 : ε < 1)
    (hrel : ∀ i, |w i - pairWeight β c i| ≤ ε * pairWeight β c i) (i : PairIndex) : 0 < w i := by
  have hW := pairWeight_pos β c i
  have h := (abs_le.mp (hrel i)).1
  nlinarith [hε1]

/-- **The normaliser's relative error.** `(1−ε)·Z_β ≤ Σ w ≤ (1+ε)·Z_β`: the integer total is the
model partition function to relative error `ε`. -/
theorem weight_sum_ge (β : ℝ) (c w : PairIndex → ℝ) (ε : ℝ) (_hε0 : 0 ≤ ε) (_hε1 : ε < 1)
    (hrel : ∀ i, |w i - pairWeight β c i| ≤ ε * pairWeight β c i) :
    (1 - ε) * pairZ β c ≤ ∑ j, w j := by
  have hper : ∀ j ∈ (Finset.univ : Finset PairIndex), (1 - ε) * pairWeight β c j ≤ w j := by
    intro j _
    have hW := pairWeight_pos β c j
    have h := (abs_le.mp (hrel j)).1
    nlinarith
  calc (1 - ε) * pairZ β c = ∑ j, (1 - ε) * pairWeight β c j := by
        rw [pairZ, Finset.mul_sum]
    _ ≤ ∑ j, w j := Finset.sum_le_sum hper

theorem weight_sum_pos (β : ℝ) (c w : PairIndex → ℝ) (ε : ℝ) (hε0 : 0 ≤ ε) (hε1 : ε < 1)
    (hrel : ∀ i, |w i - pairWeight β c i| ≤ ε * pairWeight β c i) : 0 < ∑ j, w j := by
  have hP := pairZ_pos β c
  have h1 : 0 < 1 - ε := by linarith
  have h2 := weight_sum_ge β c w ε hε0 hε1 hrel
  nlinarith

/-- Numerator bound for the law's relative error: `|w·P − W·S| ≤ 2εWP` from per-weight and
per-total relative errors. -/
theorem abs_mul_sub_mul_le {w W S P ε : ℝ} (hW : 0 < W) (hP : 0 < P)
    (h1 : |w - W| ≤ ε * W) (h2 : |S - P| ≤ ε * P) :
    |w * P - W * S| ≤ 2 * ε * W * P := by
  have hdecomp : w * P - W * S = (w - W) * P - W * (S - P) := by ring
  rw [hdecomp]
  have h3 : |(w - W) * P| ≤ ε * W * P := by
    rw [abs_mul, abs_of_pos hP]
    exact mul_le_mul_of_nonneg_right h1 (le_of_lt hP)
  have h4 : |W * (S - P)| ≤ ε * W * P := by
    rw [abs_mul, abs_of_pos hW]
    calc W * |S - P| ≤ W * (ε * P) := mul_le_mul_of_nonneg_left h2 (le_of_lt hW)
      _ = ε * W * P := by ring
  rw [abs_sub_le_iff]
  have h3' := abs_le.mp h3
  have h4' := abs_le.mp h4
  constructor <;> linarith [h3'.1, h3'.2, h4'.1, h4'.2]

/-- The law-level relative error from the two numerator bounds. -/
theorem abs_div_sub_div_le {w W S P ε : ℝ} (hW : 0 < W) (hP : 0 < P) (hS : 0 < S)
    (h1 : |w - W| ≤ ε * W) (h2 : |S - P| ≤ ε * P) :
    |w / S - W / P| ≤ (2 * ε * W * P) / (S * P) := by
  have hnum := abs_mul_sub_mul_le hW hP h1 h2
  have hSP : 0 < S * P := mul_pos hS hP
  have heq : w / S - W / P = (w * P - W * S) / (S * P) := by field_simp
  rw [heq, abs_div, abs_of_pos hSP]
  exact div_le_div_of_nonneg_right hnum (le_of_lt hSP)

/-- `a ≤ b` with `0 ≤ a` gives `a² ≤ b²` (scalar). -/
theorem sq_le_sq_aux {a b : ℝ} (ha : 0 ≤ a) (h : a ≤ b) : a ^ 2 ≤ b ^ 2 := by
  rw [sq, sq]
  exact mul_self_le_mul_self ha h

/-- The last normalisation step, on scalars: `2εaP/(SP) ≤ (2ε/(1−ε))·(a/P)`. -/
theorem div_le_rel_aux {a S P ε : ℝ} (ha : 0 ≤ a) (hε0 : 0 ≤ ε) (hS : 0 < S) (hP : 0 < P)
    (hε1 : ε < 1) (hSP : (1 - ε) * P ≤ S) :
    (2 * ε * a * P) / (S * P) ≤ (2 * ε / (1 - ε)) * (a / P) := by
  have h1e : 0 < 1 - ε := by linarith
  have hmass : (2 * ε * a * P) / (S * P) = 2 * ε * a / S := by
    rw [div_eq_div_iff (mul_ne_zero hS.ne' hP.ne') hS.ne']
    ring
  rw [hmass]
  have h2 : 2 * ε * a / S ≤ 2 * ε * a / ((1 - ε) * P) :=
    div_le_div_of_nonneg_left (by positivity) (by positivity) hSP
  have h3 : 2 * ε * a / ((1 - ε) * P) = (2 * ε / (1 - ε)) * (a / P) := by
    rw [div_mul_div_comm]
  linarith

/-- The pointwise KL step, on scalars: `a²/b ≤ (1+η)²·b` when `0 ≤ a ≤ (1+η)·b`. -/
theorem div_sq_le_aux {a b η : ℝ} (ha : 0 ≤ a) (hb : 0 < b) (h : a ≤ (1 + η) * b) :
    a ^ 2 / b ≤ (1 + η) ^ 2 * b := by
  have hM : 0 ≤ (1 + η) * b := le_trans ha h
  have hsq : a ^ 2 ≤ ((1 + η) * b) ^ 2 := sq_le_sq_aux ha h
  rw [div_le_iff₀ hb]
  calc a ^ 2 ≤ ((1 + η) * b) ^ 2 := hsq
    _ = (1 + η) ^ 2 * b * b := by ring

/-- **The normalised law's relative error.** Per-weight relative error `ε` gives per-law relative
error `2ε/(1−ε)`: the weight error and the normaliser error add. -/
theorem intLaw_rel (β : ℝ) (c w : PairIndex → ℝ) (ε : ℝ) (hε0 : 0 ≤ ε) (hε1 : ε < 1)
    (hrel : ∀ i, |w i - pairWeight β c i| ≤ ε * pairWeight β c i) (i : PairIndex) :
    |intLaw w i - pairLaw β c i| ≤ (2 * ε / (1 - ε)) * pairLaw β c i := by
  have hsum_abs0 : |∑ j, w j - pairZ β c| ≤ ε * pairZ β c := by
    have hsub : ∑ j, w j - pairZ β c = ∑ j, (w j - pairWeight β c j) := by
      rw [pairZ, Finset.sum_sub_distrib]
    rw [hsub]
    calc |∑ j, (w j - pairWeight β c j)| ≤ ∑ j, |w j - pairWeight β c j| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j, ε * pairWeight β c j := Finset.sum_le_sum (fun j _ => hrel j)
      _ = ε * pairZ β c := by rw [pairZ, Finset.mul_sum]
  obtain ⟨S, hS⟩ : ∃ S : ℝ, S = ∑ j, w j := ⟨_, rfl⟩
  obtain ⟨P, hP⟩ : ∃ P : ℝ, P = pairZ β c := ⟨_, rfl⟩
  have hPpos : 0 < P := by rw [hP]; exact pairZ_pos β c
  have hSpos : 0 < S := by rw [hS]; exact weight_sum_pos β c w ε hε0 hε1 hrel
  have hSge : (1 - ε) * P ≤ S := by rw [hS, hP]; exact weight_sum_ge β c w ε hε0 hε1 hrel
  have hsum_abs : |S - P| ≤ ε * P := by rw [hS, hP]; exact hsum_abs0
  have hW : 0 < pairWeight β c i := pairWeight_pos β c i
  have hmain : |w i / S - pairWeight β c i / P| ≤ (2 * ε * pairWeight β c i * P) / (S * P) :=
    abs_div_sub_div_le hW hPpos hSpos (hrel i) hsum_abs
  have hgoal : intLaw w i - pairLaw β c i = w i / S - pairWeight β c i / P := by
    rw [intLaw, pairLaw, hS, hP]
  rw [hgoal]
  have hbound := div_le_rel_aux (le_of_lt hW) hε0 hSpos hPpos hε1 hSge
  have hpl : pairLaw β c i = pairWeight β c i / P := by rw [pairLaw, hP]
  rw [hpl]
  linarith

/-- **The KL bound from a relative law error.** If `|q − p| ≤ η·p` pointwise, `p > 0`,
`Σ q = Σ p = 1`, then `KL(q‖p) ≤ 2η + η²` — the `log x ≤ x − 1` bound, summed. -/
theorem klDiv_le_of_rel (p q : PairIndex → ℝ) (η : ℝ) (_hη : 0 ≤ η) (hp : ∀ i, 0 < p i)
    (hq0 : ∀ i, 0 ≤ q i) (hq1 : ∑ i, q i = 1) (hp1 : ∑ i, p i = 1)
    (hrel : ∀ i, |q i - p i| ≤ η * p i) : klDiv q p ≤ 2 * η + η ^ 2 := by
  have hterm : ∀ i ∈ (Finset.univ : Finset PairIndex),
      q i * Real.log (q i / p i) ≤ q i * (q i / p i - 1) := by
    intro i _
    rcases eq_or_ne (q i) 0 with hq | hq
    · simp [hq]
    · have hqpos : 0 < q i := lt_of_le_of_ne (hq0 i) (Ne.symm hq)
      exact mul_le_mul_of_nonneg_left (Real.log_le_sub_one_of_pos (div_pos hqpos (hp i)))
        (le_of_lt hqpos)
  have hsum1 : klDiv q p ≤ ∑ i, q i * (q i / p i - 1) := by
    unfold klDiv
    exact Finset.sum_le_sum hterm
  have hsplit : ∑ i, q i * (q i / p i - 1) = (∑ i, q i ^ 2 / p i) - 1 := by
    have hterm2 : ∀ i : PairIndex, q i * (q i / p i - 1) = q i ^ 2 / p i - q i := by
      intro i
      field_simp
    rw [Finset.sum_congr rfl (fun i _ => hterm2 i), Finset.sum_sub_distrib, hq1]
  have hb : ∑ i, q i ^ 2 / p i ≤ (1 + η) ^ 2 := by
    have hpt : ∀ i ∈ (Finset.univ : Finset PairIndex),
        q i ^ 2 / p i ≤ (1 + η) ^ 2 * p i := by
      intro i _
      have hp' := hp i
      have hle : q i ≤ (1 + η) * p i := by
        have h := (abs_le.mp (hrel i)).2
        linarith
      exact div_sq_le_aux (hq0 i) hp' hle
    calc ∑ i, q i ^ 2 / p i ≤ ∑ i, (1 + η) ^ 2 * p i := Finset.sum_le_sum hpt
      _ = (1 + η) ^ 2 * ∑ i, p i := by rw [Finset.mul_sum]
      _ = (1 + η) ^ 2 := by rw [hp1, mul_one]
  have hfin : (1 + η) ^ 2 = 2 * η + η ^ 2 + 1 := by ring
  linarith

/-- **The free-energy gap of the integer table.** With a per-weight relative error `ε < 1` and the
model cost `c`, the table's normalised law is at most `(2η + η²)/β` above the Gibbs optimum, where
`η = 2ε/(1−ε)`. This is the error the correspondence carries; it is not zero. -/
theorem pairFree_gap_le (β : ℝ) (c w : PairIndex → ℝ) (ε : ℝ) (hβ : 0 < β) (hε0 : 0 ≤ ε)
    (hε1 : ε < 1) (hrel : ∀ i, |w i - pairWeight β c i| ≤ ε * pairWeight β c i) :
    pairFree β c (intLaw w) - pairFree β c (pairLaw β c)
      ≤ (2 * (2 * ε / (1 - ε)) + (2 * ε / (1 - ε)) ^ 2) / β := by
  set η : ℝ := 2 * ε / (1 - ε) with hηdef
  have h1e : 0 < 1 - ε := by linarith
  have hη0 : 0 ≤ η := by
    rw [hηdef]
    positivity
  have hS : 0 < ∑ j, w j := weight_sum_pos β c w ε hε0 hε1 hrel
  have hw0 : ∀ i, 0 ≤ intLaw w i := fun i =>
    le_of_lt (div_pos (weight_pos_of_rel β c w ε hε0 hε1 hrel i) hS)
  have hrel' : ∀ i, |intLaw w i - pairLaw β c i| ≤ η * pairLaw β c i := by
    intro i
    have h := intLaw_rel β c w ε hε0 hε1 hrel i
    rwa [hηdef]
  have hkl := klDiv_le_of_rel (pairLaw β c) (intLaw w) η hη0 (pairLaw_pos β c) hw0
    (intLaw_sum_one w (ne_of_gt hS)) (pairLaw_sum_one β c) hrel'
  have hid := pairFree_sub_law β c (intLaw w) hβ (ne_of_gt hβ) (intLaw_sum_one w (ne_of_gt hS))
  have h1 : 0 ≤ 1 / β := le_of_lt (one_div_pos.mpr hβ)
  have h2 : (1 / β) * klDiv (intLaw w) (pairLaw β c) ≤ (1 / β) * (2 * η + η ^ 2) :=
    mul_le_mul_of_nonneg_left hkl h1
  have h3 : (1 / β) * (2 * η + η ^ 2) = (2 * η + η ^ 2) / β := by
    rw [one_div, inv_mul_eq_div]
  rw [hηdef] at *
  linarith

/-- **The fixed-point `exp` relative error, propagated to a weight.** If two Q32.32 exponents agree
to `δ`, the weights they produce agree to relative `e^δ − 1`. -/
theorem exp_abs_sub_le {x y δ : ℝ} (h : |x - y| ≤ δ) :
    |Real.exp x - Real.exp y| ≤ (Real.exp δ - 1) * Real.exp y := by
  have hδ : 0 ≤ δ := le_trans (abs_nonneg _) h
  have hx : Real.exp x = Real.exp y * Real.exp (x - y) := by
    rw [← Real.exp_add]
    ring_nf
  have hfac : Real.exp y * Real.exp (x - y) - Real.exp y
      = Real.exp y * (Real.exp (x - y) - 1) := by ring
  rw [hx, hfac, abs_mul, abs_of_pos (Real.exp_pos y)]
  have hkey : |Real.exp (x - y) - 1| ≤ Real.exp δ - 1 := by
    have hle : x - y ≤ δ := le_trans (le_abs_self _) h
    have hge : -δ ≤ x - y := neg_le_of_abs_le h
    rcases le_or_gt 0 (x - y) with h0 | h0
    · have he : 1 ≤ Real.exp (x - y) := Real.one_le_exp h0
      rw [abs_of_nonneg (by linarith)]
      exact sub_le_sub_right (Real.exp_le_exp.mpr hle) 1
    · have he : Real.exp (x - y) ≤ 1 := Real.exp_le_one_iff.mpr (le_of_lt h0)
      have hgm : 2 ≤ Real.exp (x - y) + Real.exp (-(x - y)) := by
        have hsq : (Real.exp ((x - y) / 2) - Real.exp (-((x - y) / 2))) ^ 2 ≥ 0 := sq_nonneg _
        have hprod : Real.exp ((x - y) / 2) * Real.exp (-((x - y) / 2)) = 1 := by
          rw [← Real.exp_add]
          ring_nf
          simp
        have ha : Real.exp ((x - y) / 2) ^ 2 = Real.exp (x - y) := by
          rw [sq, ← Real.exp_add]
          ring_nf
        have hb : Real.exp (-((x - y) / 2)) ^ 2 = Real.exp (-(x - y)) := by
          rw [sq, ← Real.exp_add]
          ring_nf
        nlinarith
      have hmono : Real.exp (-(x - y)) ≤ Real.exp δ := Real.exp_le_exp.mpr (by linarith)
      rw [abs_of_nonpos (by linarith), neg_sub]
      linarith
  nlinarith [Real.exp_pos y, hkey, hδ]

/-- The cost the integer table exponentiates: `β·(c − min)` clipped at `16` nats, so the largest
possible gap is `e^{−16}` and every weight stays positive. -/
noncomputable def tableCost (β cmin : ℝ) (c : PairIndex → ℝ) (i : PairIndex) : ℝ :=
  min (β * (c i - cmin)) 16

/-- The table's normalised law: the Gibbs law of the clipped cost at unit inverse temperature. -/
noncomputable def tableLaw (β cmin : ℝ) (c : PairIndex → ℝ) : PairIndex → ℝ :=
  pairLaw 1 (tableCost β cmin c)

theorem tableLaw_pos (β cmin : ℝ) (c : PairIndex → ℝ) (i : PairIndex) :
    0 < tableLaw β cmin c i :=
  pairLaw_pos 1 (tableCost β cmin c) i

theorem tableLaw_sum_one (β cmin : ℝ) (c : PairIndex → ℝ) :
    ∑ i, tableLaw β cmin c i = 1 :=
  pairLaw_sum_one 1 (tableCost β cmin c)

/-- **The table's own variational principle.** Whatever the clip does, the table's law is the Gibbs
law of the clipped cost, so `E_q[clipped cost] + H(q) ≥ −log Z` for every law `q`, with equality at
the table law. -/
theorem tableLaw_free_ge (β cmin : ℝ) (c q : PairIndex → ℝ) (hq0 : ∀ i, 0 ≤ q i)
    (hq1 : ∑ i, q i = 1) :
    -Real.log (pairZ 1 (tableCost β cmin c)) ≤ pairFree 1 (tableCost β cmin c) q := by
  have h := pairFree_ge 1 (tableCost β cmin c) q (by norm_num) hq0 hq1
  simpa using h

/-- **Unclipped, the table is exactly the model law.** When every gap is below `16/β` the clip is
inactive and `tableLaw = pairLaw β c`: the truncation costs nothing and the Gibbs law is realised
exactly (up to the fixed-point terms stated separately). -/
theorem tableLaw_eq_pairLaw_of_unclamped (β cmin : ℝ) (c : PairIndex → ℝ)
    (hun : ∀ i, β * (c i - cmin) ≤ 16) (i : PairIndex) :
    tableLaw β cmin c i = pairLaw β c i := by
  have hcost : tableCost β cmin c = (fun j => β * c j + -(β * cmin)) := by
    funext j
    simp only [tableCost]
    rw [min_eq_left (hun j)]
    ring
  rw [tableLaw, hcost, pairLaw_shift, pairLaw_scale]

/-- **The sampling premise, stated rather than assumed away.** If the word path's draw `P` is the
table's normalised law (`hP`), then its free energy is within `(2η + η²)/β` of the Gibbs optimum
with `η = 2ε/(1−ε)`.

The premise is not proved for the word RNG and no equality with it is claimed: `gibbs_choice`
computes `state % total` from a xorshift state, so `P = intLaw w` holds only if that state is
uniform on `[0, 2^64)`, and the modulo reduction deviates from uniformity by at most one count per
residue class. Both gaps are outside this module. -/
theorem draw_gap_le (β : ℝ) (c w P : PairIndex → ℝ) (ε : ℝ) (hβ : 0 < β) (hε0 : 0 ≤ ε)
    (hε1 : ε < 1) (hrel : ∀ i, |w i - pairWeight β c i| ≤ ε * pairWeight β c i)
    (hP : ∀ i, P i = intLaw w i) :
    pairFree β c P - pairFree β c (pairLaw β c)
      ≤ (2 * (2 * ε / (1 - ε)) + (2 * ε / (1 - ε)) ^ 2) / β := by
  have hfun : P = intLaw w := funext hP
  rw [hfun]
  exact pairFree_gap_le β c w ε hβ hε0 hε1 hrel

end Learn.FitIcm
