/-
# Dimension-changing stage composition and a pre-norm transformer block error bound

`Nous.Model.ErrorComposition` composes per-operation errors through `Layer E`
(`ErrorComposition.lean:71-83`), an endomorphism `E → E` on one normed space: a dimension-changing
projection (`n_in → n_out`) cannot enter its fold, and its only instantiations are single-layer
scalar folds (`siluLayer`, `rmsLayer`, `ErrorComposition.lean:522,601`). This module adds a
dimension-changing stage contract, `Stage E F`, with an explicit admissible domain and a
domain-propagation obligation carried through composition (not the `Layer.ofKernel` device of
setting `exec := ref` off-domain, `ErrorComposition.lean:352-359`), and assembles a pre-norm
transformer block from it.

## The stage contract and its combinators

`Stage E F` pairs `ref exec : E → F` with an explicit domain `dom : Set E`; `defect` is stated only
on `dom`. `expansion` (the reference map's Lipschitz bound) is global, matching `Layer.expansion`.

* `Stage.comp` — sequential composition. The composed domain is exactly the propagation obligation
  `x ∈ dom₁ ∧ exec₁ x ∈ dom₂`; the composed bound is `delta₂ + L₂ * delta₁`.
* `Stage.prod` — fan-out to a pair of parallel stages from the same source, sup norm
  (`Prod.norm_mk`); `delta`/`L` are the pairwise `max`.
* `Stage.residual` — a residual (skip-connection) update `z ↦ z + s.exec z`; `delta` is unchanged,
  `L` becomes `1 + s.L`.
* `Stage.pi` — coordinatewise lifting of a same-space stage to a finite-index Pi type under the sup
  norm (`Pi.seminormedGroup`); `delta`/`L` are unchanged.
* `Stage.ofLayer` — every `Layer E` (`ErrorComposition.lean:71`) is a `Stage E E` on `dom = Set.univ`.
* `Stage.matvec` — a real matrix-vector product as a `Stage`; the expansion factor is the
  `ℓ∞ → ℓ∞` operator norm (the max row `ℓ1` norm), proved once from `abs_sum_le_sum_abs`.
* `Stage.chain` — an `N`-stage list fold by repeated `Stage.comp`, the list version of the
  composition (the analogue of `Layer.budget`'s fold, `ErrorComposition.lean:99-104`).

## Trust boundary

Every combinator above is proved at the abstract real-algebra level: kernel-checked Lean theorems,
no `sorry`/`axiom`/`native_decide`. Two classes of per-stage input are Rust-kernel-grounded, reused
unchanged from `Nous.Model.ErrorComposition` (read, not restated):

* `siluLayer : Layer ℝ` (`ErrorComposition.lean:522`), whose defect traces to
  `silu_window_refines` (`Numeric/Engine/Silu.lean:755`);
* `rmsLayer a ha B hB : Layer ℝ` (`ErrorComposition.lean:601`), whose defect traces to
  `rmsElement_bound`/`rms_norm_refines_proved` (`Numeric/Engine/Norm.lean:685,726`). `a`
  (the fixed mean-square-plus-epsilon radicand) is an **external hypothesis** of `rmsLayer` itself:
  the mean-square computation ("RMSNorm's mean") is not derived from the block's input vector by
  `rmsLayer`, and is not re-derived here either — it is exactly the hypothesis this file inherits.

`Stage.matvec`'s per-coordinate defect `matvecEps = 1/(2·2^32)` is an **abstract-algebra model** of
`matvec_refines_proved`'s conclusion (`Numeric/Engine/Matvec.lean:96-108`: the extracted
`word_matvec_fixed`'s output is within `1/(2·q32)` of `rowDotReal (deq r) x` per coordinate); the
correspondence between the abstract linear map used here (`matvecRef`, real-valued, `Fin m → Fin n →
ℝ`) and the extracted body over `Slice I64`/`Nat`-indexed rows is a reading (`model` grade), not a
Lean term linking the two — bridging `Slice I64` matvec to a `Fin m → Fin n → ℝ` matrix is a separate,
larger obligation.

`attn` (RoPE, the softmax reweighting and the value sum, end-to-end) and `mul` (the FFN's elementwise
gate/up product) are genuine hypotheses of `BlockResidual`, carrying no kernel refinement wired in at
this revision. `rotate_head_bound` (`Numeric/Rope/HeadContent.lean:403`) proves a per-rotary-index
RoPE bound, and `attn_stage_compose_closed`/`softmaxPerturbBound`
(`Numeric/Attn/SoftmaxPerturb.lean:581,572`) prove an attention-stage perturbation bound; neither is
wired into `attn`'s `delta`/`L` here, matching `ErrorComposition.lean:335-336`'s own note that
`rope_pair_spec_proved` is proved but "not yet wired into this file". `attn.ref` is left fully open
(no formula pinned), as `ErrorComposition.lean`'s `UnrefinedOp.attentionSoftmax` leaves it unpinned
in its residual list (`ErrorComposition.lean:618-627`). `mul.ref` is pinned to the real elementwise
product **on `mul.dom`** (`mul_ref_on`), since that operation's mathematical form is not itself in
question — only its kernel bound is unrefined. The agreement is restricted to the domain because the
product has no global Lipschitz constant while `Stage.expansion` is global: pinning the product on
the whole space makes `BlockResidual p h` empty for every `1 ≤ h`
(`Nous.Model.BlockGroundedWitness1.pinnedMul_stage_impossible`).

No claim is made about rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V, the `unsafe` container, or the
dequant-to-ideal-weight reading (`Numeric/Engine/Matvec.lean`'s own class-(b) `tested`
boundary, not re-examined here).
-/
import Mathlib.Analysis.Normed.Group.Constructions
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic
import Nous.Model.ErrorComposition

namespace Nous.Model

/-! ## The dimension-changing stage contract -/

/-- A stage from `E` to `F`: an executing map `exec`, a reference map `ref`, an explicit admissible
domain `dom`, an absolute defect budget `delta` on `dom`, and a global expansion factor `L` for
`ref`. Unlike `Layer.ofKernel` (`ErrorComposition.lean:352-393`), `exec` is not forced to equal `ref`
off `dom`: composition (`Stage.comp`) carries its own domain-propagation obligation instead of a
default value there. -/
structure Stage (E F : Type*) [NormedAddCommGroup E] [NormedAddCommGroup F] where
  /-- The reference map. -/
  ref : E → F
  /-- The executing map. -/
  exec : E → F
  /-- The admissible domain: `defect` binds only here. -/
  dom : Set E
  /-- The absolute defect budget on `dom`. -/
  delta : ℝ
  /-- The global expansion (Lipschitz) factor of `ref`. -/
  L : ℝ
  delta_nonneg : 0 ≤ delta
  L_nonneg : 0 ≤ L
  defect : ∀ z ∈ dom, ‖exec z - ref z‖ ≤ delta
  expansion : ∀ z w, ‖ref z - ref w‖ ≤ L * ‖z - w‖

namespace Stage

variable {E F G : Type*} [NormedAddCommGroup E] [NormedAddCommGroup F] [NormedAddCommGroup G]

/-- Sequential composition. The composed domain is exactly the domain-propagation obligation:
`x ∈ dom₁` and `exec₁ x ∈ dom₂`. The composed bound is `delta₂ + L₂ * delta₁`
(`ErrorComposition.lean`'s `Layer.budget` one-step recurrence, generalized to `E ≠ F`). -/
def comp (s1 : Stage E F) (s2 : Stage F G) : Stage E G where
  ref := s2.ref ∘ s1.ref
  exec := s2.exec ∘ s1.exec
  dom := {x | x ∈ s1.dom ∧ s1.exec x ∈ s2.dom}
  delta := s2.delta + s2.L * s1.delta
  L := s2.L * s1.L
  delta_nonneg := add_nonneg s2.delta_nonneg (mul_nonneg s2.L_nonneg s1.delta_nonneg)
  L_nonneg := mul_nonneg s2.L_nonneg s1.L_nonneg
  defect := by
    intro x hx
    obtain ⟨hx1, hx2⟩ := hx
    have hsplit :
        s2.exec (s1.exec x) - s2.ref (s1.ref x)
          = (s2.exec (s1.exec x) - s2.ref (s1.exec x))
            + (s2.ref (s1.exec x) - s2.ref (s1.ref x)) := by abel
    show ‖s2.exec (s1.exec x) - s2.ref (s1.ref x)‖ ≤ s2.delta + s2.L * s1.delta
    calc ‖s2.exec (s1.exec x) - s2.ref (s1.ref x)‖
        ≤ ‖s2.exec (s1.exec x) - s2.ref (s1.exec x)‖
            + ‖s2.ref (s1.exec x) - s2.ref (s1.ref x)‖ := by
          rw [hsplit]; exact norm_add_le _ _
      _ ≤ s2.delta + s2.L * ‖s1.exec x - s1.ref x‖ :=
          add_le_add (s2.defect _ hx2) (s2.expansion _ _)
      _ ≤ s2.delta + s2.L * s1.delta :=
          add_le_add le_rfl (mul_le_mul_of_nonneg_left (s1.defect x hx1) s2.L_nonneg)
  expansion := by
    intro z w
    show ‖s2.ref (s1.ref z) - s2.ref (s1.ref w)‖ ≤ s2.L * s1.L * ‖z - w‖
    calc ‖s2.ref (s1.ref z) - s2.ref (s1.ref w)‖
        ≤ s2.L * ‖s1.ref z - s1.ref w‖ := s2.expansion _ _
      _ ≤ s2.L * (s1.L * ‖z - w‖) :=
          mul_le_mul_of_nonneg_left (s1.expansion z w) s2.L_nonneg
      _ = s2.L * s1.L * ‖z - w‖ := by ring

/-- The literal domain-propagation statement of `comp`: on `x ∈ dom₁` and `exec₁ x ∈ dom₂` the
composed executing map differs from the composed reference map by at most `delta₂ + L₂ * delta₁`. -/
theorem comp_bound (s1 : Stage E F) (s2 : Stage F G) (x : E)
    (hx1 : x ∈ s1.dom) (hx2 : s1.exec x ∈ s2.dom) :
    ‖s2.exec (s1.exec x) - s2.ref (s1.ref x)‖ ≤ s2.delta + s2.L * s1.delta :=
  (s1.comp s2).defect x ⟨hx1, hx2⟩

/-- Fan-out: two stages from the same source, composed into one stage targeting the product under
the sup norm (`Prod.norm_mk`). `delta`/`L` are the pairwise `max`. -/
def prod (s1 : Stage E F) (s2 : Stage E G) : Stage E (F × G) where
  ref := fun x => (s1.ref x, s2.ref x)
  exec := fun x => (s1.exec x, s2.exec x)
  dom := s1.dom ∩ s2.dom
  delta := max s1.delta s2.delta
  L := max s1.L s2.L
  delta_nonneg := le_trans s1.delta_nonneg (le_max_left _ _)
  L_nonneg := le_trans s1.L_nonneg (le_max_left _ _)
  defect := by
    intro x hx
    obtain ⟨hx1, hx2⟩ := hx
    show ‖(s1.exec x - s1.ref x, s2.exec x - s2.ref x)‖ ≤ max s1.delta s2.delta
    rw [Prod.norm_mk]
    exact max_le_max (s1.defect x hx1) (s2.defect x hx2)
  expansion := by
    intro z w
    show ‖(s1.ref z - s1.ref w, s2.ref z - s2.ref w)‖ ≤ max s1.L s2.L * ‖z - w‖
    rw [Prod.norm_mk]
    have h1 : ‖s1.ref z - s1.ref w‖ ≤ max s1.L s2.L * ‖z - w‖ :=
      le_trans (s1.expansion z w)
        (mul_le_mul_of_nonneg_right (le_max_left _ _) (norm_nonneg _))
    have h2 : ‖s2.ref z - s2.ref w‖ ≤ max s1.L s2.L * ‖z - w‖ :=
      le_trans (s2.expansion z w)
        (mul_le_mul_of_nonneg_right (le_max_right _ _) (norm_nonneg _))
    exact max_le h1 h2

/-- A residual (skip-connection) update `z ↦ z + s.exec z` against the reference `z ↦ z + s.ref z`.
The defect is unchanged (the shared `z` cancels); the expansion factor becomes `1 + s.L`. -/
def residual (s : Stage E E) : Stage E E where
  ref := fun z => z + s.ref z
  exec := fun z => z + s.exec z
  dom := s.dom
  delta := s.delta
  L := 1 + s.L
  delta_nonneg := s.delta_nonneg
  L_nonneg := by linarith [s.L_nonneg]
  defect := by
    intro z hz
    show ‖(z + s.exec z) - (z + s.ref z)‖ ≤ s.delta
    have : (z + s.exec z) - (z + s.ref z) = s.exec z - s.ref z := by abel
    rw [this]
    exact s.defect z hz
  expansion := by
    intro z w
    show ‖(z + s.ref z) - (w + s.ref w)‖ ≤ (1 + s.L) * ‖z - w‖
    have hsplit : (z + s.ref z) - (w + s.ref w) = (z - w) + (s.ref z - s.ref w) := by abel
    calc ‖(z + s.ref z) - (w + s.ref w)‖
        = ‖(z - w) + (s.ref z - s.ref w)‖ := by rw [hsplit]
      _ ≤ ‖z - w‖ + ‖s.ref z - s.ref w‖ := norm_add_le _ _
      _ ≤ ‖z - w‖ + s.L * ‖z - w‖ := by linarith [s.expansion z w]
      _ = (1 + s.L) * ‖z - w‖ := by ring

/-- Every `Layer E` (`ErrorComposition.lean:71-83`) is a `Stage E E` on the full domain. -/
def ofLayer (g : Layer E) : Stage E E where
  ref := g.ref
  exec := g.exec
  dom := Set.univ
  delta := g.delta
  L := g.L
  delta_nonneg := g.delta_nonneg
  L_nonneg := g.L_nonneg
  defect := fun z _ => g.defect z
  expansion := g.expansion

/-- Coordinatewise lifting of a same-space stage to a finite-index Pi type, under the sup norm
(`Pi.seminormedGroup`). `delta`/`L` are unchanged; the domain is coordinatewise membership. -/
def pi {ι : Type*} [Fintype ι] (s : Stage E E) : Stage (ι → E) (ι → E) where
  ref := fun x i => s.ref (x i)
  exec := fun x i => s.exec (x i)
  dom := {x | ∀ i, x i ∈ s.dom}
  delta := s.delta
  L := s.L
  delta_nonneg := s.delta_nonneg
  L_nonneg := s.L_nonneg
  defect := by
    intro x hx
    apply (pi_norm_le_iff_of_nonneg s.delta_nonneg).2
    intro i
    show ‖s.exec (x i) - s.ref (x i)‖ ≤ s.delta
    exact s.defect (x i) (hx i)
  expansion := by
    intro z w
    apply (pi_norm_le_iff_of_nonneg (mul_nonneg s.L_nonneg (norm_nonneg _))).2
    intro i
    show ‖s.ref (z i) - s.ref (w i)‖ ≤ s.L * ‖z - w‖
    have hzw : ‖z i - w i‖ ≤ ‖z - w‖ := by
      have := norm_le_pi_norm (z - w) i
      rwa [Pi.sub_apply] at this
    exact le_trans (s.expansion (z i) (w i)) (mul_le_mul_of_nonneg_left hzw s.L_nonneg)

/-! ## Real matrix-vector product as a stage -/

/-- The real matrix-vector product `y i = Σ_j W i j * x j`, the reference form of
`matvec_refines_proved`'s conclusion (`Numeric/Engine/Matvec.lean:99-107`), abstracted from
`Slice I64`/`rowDotReal` to a real `Fin m → Fin n → ℝ` matrix. -/
def matvecRef {n m : ℕ} (W : Fin m → Fin n → ℝ) (x : Fin n → ℝ) : Fin m → ℝ :=
  fun i => ∑ j, W i j * x j

/-- The `ℓ∞ → ℓ∞` operator-norm bound: `matvecRef W` is Lipschitz with constant the max row `ℓ1`
norm of `W`. Proved from the triangle inequality (`abs_sum_le_sum_abs`) and the sup-norm coordinate
bound (`norm_le_pi_norm`). -/
theorem matvecRef_lipschitz {n m : ℕ} (W : Fin m → Fin n → ℝ) (rowBound : ℝ)
    (hrb : 0 ≤ rowBound) (hrow : ∀ i, ∑ j, |W i j| ≤ rowBound) (z w : Fin n → ℝ) :
    ‖matvecRef W z - matvecRef W w‖ ≤ rowBound * ‖z - w‖ := by
  apply (pi_norm_le_iff_of_nonneg (mul_nonneg hrb (norm_nonneg _))).2
  intro i
  show ‖matvecRef W z i - matvecRef W w i‖ ≤ rowBound * ‖z - w‖
  have heq : matvecRef W z i - matvecRef W w i = ∑ j, W i j * (z j - w j) := by
    unfold matvecRef
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun j _ => by ring)
  rw [Real.norm_eq_abs, heq]
  have hcoord : ∀ j : Fin n, ‖z j - w j‖ ≤ ‖z - w‖ := by
    intro j
    have := norm_le_pi_norm (z - w) j
    rwa [Pi.sub_apply] at this
  calc |∑ j, W i j * (z j - w j)|
      ≤ ∑ j, |W i j * (z j - w j)| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ j, |W i j| * |z j - w j| := Finset.sum_congr rfl (fun j _ => abs_mul _ _)
    _ ≤ ∑ j, |W i j| * ‖z - w‖ := by
        apply Finset.sum_le_sum
        intro j _
        exact mul_le_mul_of_nonneg_left
          (by rw [← Real.norm_eq_abs]; exact hcoord j) (abs_nonneg _)
    _ = (∑ j, |W i j|) * ‖z - w‖ := (Finset.sum_mul _ _ _).symm
    _ ≤ rowBound * ‖z - w‖ := mul_le_mul_of_nonneg_right (hrow i) (norm_nonneg _)

/-- A real matrix-vector product packaged as a `Stage`: `exec` is the caller-supplied model of the
extracted kernel, `delta` its per-coordinate defect on `dom`, and `L` the max row `ℓ1` norm of `W`
(`matvecRef_lipschitz`). -/
def matvec {n m : ℕ} (W : Fin m → Fin n → ℝ) (exec : (Fin n → ℝ) → (Fin m → ℝ))
    (dom : Set (Fin n → ℝ)) (eps rowBound : ℝ) (heps : 0 ≤ eps) (hrb : 0 ≤ rowBound)
    (hrow : ∀ i, ∑ j, |W i j| ≤ rowBound)
    (hdefect : ∀ x ∈ dom, ∀ i, |exec x i - matvecRef W x i| ≤ eps) :
    Stage (Fin n → ℝ) (Fin m → ℝ) where
  ref := matvecRef W
  exec := exec
  dom := dom
  delta := eps
  L := rowBound
  delta_nonneg := heps
  L_nonneg := hrb
  defect := by
    intro x hx
    apply (pi_norm_le_iff_of_nonneg heps).2
    intro i
    show ‖exec x i - matvecRef W x i‖ ≤ eps
    rw [Real.norm_eq_abs]
    exact hdefect x hx i
  expansion := matvecRef_lipschitz W rowBound hrb hrow

/-! ## The `N`-stage list version -/

/-- The identity stage: zero defect, expansion factor `1`. -/
def id (E : Type*) [NormedAddCommGroup E] : Stage E E where
  ref := _root_.id
  exec := _root_.id
  dom := Set.univ
  delta := 0
  L := 1
  delta_nonneg := le_refl 0
  L_nonneg := zero_le_one
  defect := by intro z _; simp
  expansion := by intro z w; simp

/-- The `N`-stage fold of a list of same-space stages, applied head-first, by repeated `Stage.comp`:
the list version of `Layer.budget`'s accumulation (`ErrorComposition.lean:99-104`), generalized to
carry each stage's own domain-propagation obligation instead of `Layer`'s unconditional `defect`. -/
def chain {E : Type*} [NormedAddCommGroup E] : List (Stage E E) → Stage E E
  | [] => Stage.id E
  | s :: rest => s.comp (chain rest)

/-- The one-step unfold of `chain`, the list analogue of `Layer.budget_cons_cons`
(`ErrorComposition.lean:114-118`): the head stage's defect is scaled by the tail's accumulated
expansion factor and added to the tail's own accumulated defect. -/
theorem chain_cons {E : Type*} [NormedAddCommGroup E] (s : Stage E E) (rest : List (Stage E E)) :
    (chain (s :: rest)).delta = (chain rest).delta + (chain rest).L * s.delta ∧
    (chain (s :: rest)).L = (chain rest).L * s.L := by
  constructor <;> rfl

end Stage

/-! ## A pre-norm transformer block

`V n := Fin n → ℝ`, sup norm. `d` is the residual-stream width, `p` the (combined) Q/K/V projection
width, `h` the FFN hidden width. -/

/-- The abstract-algebra model of `matvec_refines_proved`'s per-coordinate bound
(`Numeric/Engine/Matvec.lean:107`): `1/(2·q32)`, `q32 = 2^32`
(`Numeric.Scale.q32`). Restated as a real literal so `Stage.matvec` does not depend on the Aeneas
`I64`/`Slice` types it is a model of. -/
noncomputable def matvecEps : ℝ := 1 / (2 * (2 : ℝ) ^ (32 : ℕ))

theorem matvecEps_nonneg : 0 ≤ matvecEps := by unfold matvecEps; norm_num

/-- One real matvec projection packaged with its row-bound hypotheses, so the block-assembly
definitions below take one argument per projection instead of five. `MatvecSpec.stage` builds the
`Stage.matvec` instance with the fixed defect `matvecEps`. -/
structure MatvecSpec (n m : ℕ) where
  W : Fin m → Fin n → ℝ
  exec : (Fin n → ℝ) → (Fin m → ℝ)
  dom : Set (Fin n → ℝ)
  rowBound : ℝ
  rowBound_nonneg : 0 ≤ rowBound
  hrow : ∀ i, ∑ j, |W i j| ≤ rowBound
  hdefect : ∀ x ∈ dom, ∀ i, |exec x i - Stage.matvecRef W x i| ≤ matvecEps

/-- The `Stage` of a `MatvecSpec`. -/
noncomputable def MatvecSpec.stage {n m : ℕ} (spec : MatvecSpec n m) :
    Stage (Fin n → ℝ) (Fin m → ℝ) :=
  Stage.matvec spec.W spec.exec spec.dom matvecEps spec.rowBound matvecEps_nonneg
    spec.rowBound_nonneg spec.hrow spec.hdefect

@[simp] theorem MatvecSpec.stage_delta {n m : ℕ} (spec : MatvecSpec n m) :
    spec.stage.delta = matvecEps := rfl

@[simp] theorem MatvecSpec.stage_L {n m : ℕ} (spec : MatvecSpec n m) :
    spec.stage.L = spec.rowBound := rfl

/-- The operations of the block with no landed kernel refinement at this revision: attention
end-to-end (RoPE, the softmax reweighting and the value sum) and the FFN's elementwise gate/up
product. See the file docstring for the exact trust boundary each carries. -/
structure BlockResidual (p h : ℕ) where
  /-- The attention sub-block from the fanned-out `(Q, K, V)` projections to the pre-output-matvec
  attention result. Its reference map is left fully open: no formula is pinned. -/
  attn : Stage ((Fin p → ℝ) × (Fin p → ℝ) × (Fin p → ℝ)) (Fin p → ℝ)
  /-- The FFN elementwise gate/up product. -/
  mul : Stage ((Fin h → ℝ) × (Fin h → ℝ)) (Fin h → ℝ)
  /-- `mul`'s reference map is pinned to the real elementwise product on `mul.dom`: the kernel bound
  is unrefined, not the mathematical operation. The agreement is restricted to the domain because
  the product has no global Lipschitz constant while `Stage.expansion` is global (`∀ z w, ‖ref z -
  ref w‖ ≤ L * ‖z - w‖`, no domain): off a bounded box its expansion factor is unbounded. -/
  mul_ref_on : ∀ z ∈ mul.dom, mul.ref z = fun i => z.1 i * z.2 i

variable {d p h : ℕ}

/-- The `(Q, K, V)` fan-out, `Stage.prod` twice. -/
noncomputable def qkvFanout (qs ks vs : MatvecSpec d p) :
    Stage (Fin d → ℝ) ((Fin p → ℝ) × (Fin p → ℝ) × (Fin p → ℝ)) :=
  qs.stage.prod (ks.stage.prod vs.stage)

/-- The attention sub-block: fan out to `(Q, K, V)`, apply the unrefined `attn`, project back to the
residual width with the output matvec (`os : MatvecSpec p d`, width `p` to width `d`). Maps
`Fin d → ℝ` to itself. -/
noncomputable def attnSubBlock (qs ks vs : MatvecSpec d p) (os : MatvecSpec p d)
    (br : BlockResidual p h) :
    Stage (Fin d → ℝ) (Fin d → ℝ) :=
  (qkvFanout qs ks vs).comp (br.attn.comp os.stage)

/-- The FFN sub-block: fan out to the SiLU-gated branch and the up branch, apply the unrefined
elementwise product, project back to the residual width with the down matvec. Maps `Fin d → ℝ` to
itself. `silu` is applied coordinatewise on the hidden width `h` (`Stage.pi`). -/
noncomputable def ffnSubBlock (gs us : MatvecSpec d h) (dws : MatvecSpec h d)
    (br : BlockResidual p h) :
    Stage (Fin d → ℝ) (Fin d → ℝ) :=
  ((gs.stage.comp (Stage.pi (Stage.ofLayer siluLayer))).prod us.stage).comp
    (br.mul.comp dws.stage)

open apeiron_word Aeneas.Std in
/-- The pre-norm transformer block: `RMSNorm → attnSubBlock → residual add → RMSNorm →
ffnSubBlock → residual add`. `a1`/`a2` are the two RMSNorm mean-square-plus-epsilon radicands,
external hypotheses standing in for the unrefined mean computation (see the file docstring). -/
noncomputable def fullBlock (qs ks vs : MatvecSpec d p) (os : MatvecSpec p d)
    (gs us : MatvecSpec d h) (dws : MatvecSpec h d) (br : BlockResidual p h)
    (a1 : I64) (ha1 : rsqrt_window a1) (B1 : ℝ) (hB1 : 0 ≤ B1)
    (a2 : I64) (ha2 : rsqrt_window a2) (B2 : ℝ) (hB2 : 0 ≤ B2) :
    Stage (Fin d → ℝ) (Fin d → ℝ) :=
  (Stage.residual ((Stage.pi (Stage.ofLayer (rmsLayer a1 ha1 B1 hB1))).comp
      (attnSubBlock qs ks vs os br))).comp
    (Stage.residual ((Stage.pi (Stage.ofLayer (rmsLayer a2 ha2 B2 hB2))).comp
      (ffnSubBlock gs us dws br)))

open apeiron_word Aeneas.Std in
/-- **Block error bound.** On `fullBlock`'s domain (the full domain-propagation obligation of every
internal `Stage.comp`/`Stage.prod`/`Stage.residual`/`Stage.pi`), the executing block differs from
the reference block by at most `fullBlock`'s `delta`, which one unfold step below
(`fullBlock_delta_eq`) exposes as `block2.delta + block2.L * block1.delta` in the two residual
sub-blocks, and which `attnSubBlock`/`ffnSubBlock`'s own definitions further expose down to the
per-stage `matvecEps`, `rowBound`, `siluDelta`/`rmsDelta`, and `br.attn`/`br.mul`'s `delta`/`L`. -/
theorem block_error_bound (qs ks vs : MatvecSpec d p) (os : MatvecSpec p d)
    (gs us : MatvecSpec d h) (dws : MatvecSpec h d) (br : BlockResidual p h)
    (a1 : I64) (ha1 : rsqrt_window a1) (B1 : ℝ) (hB1 : 0 ≤ B1)
    (a2 : I64) (ha2 : rsqrt_window a2) (B2 : ℝ) (hB2 : 0 ≤ B2)
    (x : Fin d → ℝ)
    (hx : x ∈ (fullBlock qs ks vs os gs us dws br a1 ha1 B1 hB1 a2 ha2 B2 hB2).dom) :
    ‖(fullBlock qs ks vs os gs us dws br a1 ha1 B1 hB1 a2 ha2 B2 hB2).exec x
        - (fullBlock qs ks vs os gs us dws br a1 ha1 B1 hB1 a2 ha2 B2 hB2).ref x‖
      ≤ (fullBlock qs ks vs os gs us dws br a1 ha1 B1 hB1 a2 ha2 B2 hB2).delta :=
  (fullBlock qs ks vs os gs us dws br a1 ha1 B1 hB1 a2 ha2 B2 hB2).defect x hx

open apeiron_word Aeneas.Std in
/-- One-step unfold of `fullBlock.delta`, in terms of the two residual sub-blocks:
`block2.delta + block2.L * block1.delta`, the two-stage `Stage.comp` recurrence
(`ErrorComposition.lean`'s `Layer.budget_cons_cons` shape) applied at the top level. -/
theorem fullBlock_delta_eq (qs ks vs : MatvecSpec d p) (os : MatvecSpec p d)
    (gs us : MatvecSpec d h) (dws : MatvecSpec h d) (br : BlockResidual p h)
    (a1 : I64) (ha1 : rsqrt_window a1) (B1 : ℝ) (hB1 : 0 ≤ B1)
    (a2 : I64) (ha2 : rsqrt_window a2) (B2 : ℝ) (hB2 : 0 ≤ B2) :
    (fullBlock qs ks vs os gs us dws br a1 ha1 B1 hB1 a2 ha2 B2 hB2).delta =
      (Stage.residual ((Stage.pi (Stage.ofLayer (rmsLayer a2 ha2 B2 hB2))).comp
          (ffnSubBlock gs us dws br))).delta
        + (Stage.residual ((Stage.pi (Stage.ofLayer (rmsLayer a2 ha2 B2 hB2))).comp
              (ffnSubBlock gs us dws br))).L
          * (Stage.residual ((Stage.pi (Stage.ofLayer (rmsLayer a1 ha1 B1 hB1))).comp
              (attnSubBlock qs ks vs os br))).delta :=
  rfl

/-- The `(SiLU(gate), up)` fan-out's own `delta`: the pairwise `max` of the SiLU-composed gate
branch's defect (`Stage.comp`'s formula gives `siluLayer.delta + siluLayer.L * matvecEps`, since the
gate matvec's own defect is `matvecEps`) and the up branch's defect (`matvecEps`). -/
theorem ffn_fanout_delta_eq (gs us : MatvecSpec d h) :
    ((gs.stage.comp (Stage.pi (Stage.ofLayer siluLayer))).prod us.stage).delta
      = max (siluLayer.delta + siluLayer.L * matvecEps) matvecEps := rfl

/-- One-step unfold of the attention sub-block's `delta`, fully closed in the leaf quantities:
the output matvec's own defect (`matvecEps`) plus its expansion factor (`os.rowBound`) times the
`attn` hypothesis's defect, plus `os.rowBound * br.attn.L` times the `(Q, K, V)` fan-out's own
defect — which collapses to `matvecEps` since every one of the three projections shares the same
kernel bound (`max_self`, twice). -/
theorem attnSubBlock_delta_eq (qs ks vs : MatvecSpec d p) (os : MatvecSpec p d)
    (br : BlockResidual p h) :
    (attnSubBlock qs ks vs os br).delta =
      matvecEps + os.rowBound * br.attn.delta + os.rowBound * br.attn.L * matvecEps := by
  have hstruct : (attnSubBlock qs ks vs os br).delta =
      (matvecEps + os.rowBound * br.attn.delta)
        + (os.rowBound * br.attn.L) * max matvecEps (max matvecEps matvecEps) := rfl
  rw [hstruct, max_self, max_self]

/-- One-step unfold of the FFN sub-block's `delta`, fully closed in the leaf quantities: the down
matvec's own defect (`matvecEps`) plus its expansion factor (`dws.rowBound`) times the `mul`
hypothesis's defect, plus `dws.rowBound * br.mul.L` times the `(SiLU(gate), up)` fan-out's own
defect (`ffn_fanout_delta_eq`). -/
theorem ffnSubBlock_delta_eq (gs us : MatvecSpec d h) (dws : MatvecSpec h d)
    (br : BlockResidual p h) :
    (ffnSubBlock gs us dws br).delta =
      (matvecEps + dws.rowBound * br.mul.delta)
        + (dws.rowBound * br.mul.L) * max (siluLayer.delta + siluLayer.L * matvecEps) matvecEps :=
  rfl

/-! ## Concrete non-vacuity witness

A one-dimensional toy composite exercising every combinator (`matvec`, `comp`, `pi`, `residual`,
`prod`) with explicit weights, a non-trivial domain-propagation obligation, and a `norm_num`-checked
numeric `delta`. It uses the real `Layer.affineLayer` (`ErrorComposition.lean:186-199`) in place of
the Rust-grounded `siluLayer`/`rmsLayer`, so the witness needs no Aeneas `I64` literal: the point is
the composed real-algebra bound, already exercised at Rust-kernel grade by `MatvecSpec`/`siluLayer`
above. -/

/-- The toy weight: a single `2 × 2`-free scalar row `[2]` on `Fin 1 → Fin 1 → ℝ`. -/
def toyW : Fin 1 → Fin 1 → ℝ := fun _ _ => 2

theorem toyW_rowBound : ∀ i : Fin 1, ∑ j, |toyW i j| ≤ 2 := by
  intro i; simp [toyW]

/-- The toy matvec stage: zero kernel defect (`exec = ref`), restricted to the non-trivial domain
`x 0 ≤ 5`, expansion factor `2` (the row bound). -/
noncomputable def toyMatvec : Stage (Fin 1 → ℝ) (Fin 1 → ℝ) :=
  Stage.matvec toyW (Stage.matvecRef toyW) {x | x 0 ≤ 5} 0 2 le_rfl (by norm_num) toyW_rowBound
    (by intro x _ i; simp)

/-- The toy affine stage lifted coordinatewise: defect `1/10`, expansion factor `1`. -/
noncomputable def toyAffine : Stage (Fin 1 → ℝ) (Fin 1 → ℝ) :=
  Stage.pi (Stage.ofLayer (Layer.affineLayer (1 / 10 : ℝ)))

/-- The toy residual composite: `z ↦ z + affine(matvec z)`. -/
noncomputable def toyBlock : Stage (Fin 1 → ℝ) (Fin 1 → ℝ) :=
  Stage.residual (toyMatvec.comp toyAffine)

/-- The witness point: `x = (1)`, satisfying `toyMatvec`'s domain (`1 ≤ 5`) and propagating into
`toyAffine`'s domain (`Set.univ`, via `Stage.pi`'s coordinatewise membership, trivial). -/
theorem toyBlock_dom_witness : (fun _ : Fin 1 => (1 : ℝ)) ∈ toyBlock.dom := by
  refine ⟨?_, ?_⟩
  · show (fun _ : Fin 1 => (1 : ℝ)) 0 ≤ 5
    norm_num
  · intro i
    exact Set.mem_univ _

/-- **Non-vacuity witness.** The toy composite's accumulated defect is exactly `1/10`
(`toyAffine.delta + toyAffine.L * toyMatvec.delta = 1/10 + 1 * 0`), proved by `norm_num` after
unfolding every combinator: `Stage.comp`, `Stage.residual`, `Stage.pi`, `Stage.ofLayer`,
`Stage.matvec`, and `Layer.affineLayer`. -/
theorem toyBlock_delta_eq : toyBlock.delta = 1 / 10 := by
  unfold toyBlock toyMatvec toyAffine Stage.comp Stage.residual Stage.pi Stage.ofLayer
    Stage.matvec Layer.affineLayer
  norm_num

/-- The toy composite's expansion factor is `1 + 1 * 2 = 3`. -/
theorem toyBlock_L_eq : toyBlock.L = 3 := by
  unfold toyBlock toyMatvec toyAffine Stage.comp Stage.residual Stage.pi Stage.ofLayer
    Stage.matvec Layer.affineLayer
  norm_num

/-- The witness point's executing/reference images differ by at most `1/10`, discharged from
`toyBlock_delta_eq` and `toyBlock_dom_witness` alone: `Stage.defect` needs no further computation. -/
theorem toyBlock_bound :
    ‖toyBlock.exec (fun _ : Fin 1 => (1 : ℝ)) - toyBlock.ref (fun _ : Fin 1 => (1 : ℝ))‖ ≤ 1 / 10 := by
  rw [← toyBlock_delta_eq]
  exact toyBlock.defect _ toyBlock_dom_witness

end Nous.Model
