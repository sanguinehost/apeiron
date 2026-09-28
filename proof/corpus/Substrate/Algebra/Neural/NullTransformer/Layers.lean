import Substrate.Algebra.NullCell.NullCell
import Substrate.Algebra.Neural.AttentionSpinor
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Analysis.Calculus.FDeriv.Pi
import Mathlib.Tactic
import Substrate.Algebra.Neural.NullTransformer.Core

/-!
# NullTransformer.Layers — circuits, shape/KV, loss, training, and inference

Split from historical `Substrate.Algebra.Neural.NullTransformer` (lines 850–1648).
The nonlinearity derivation, ops and graph definitions are in
`Substrate.Algebra.Neural.NullTransformer.Core`.
-/

noncomputable section

open Finset Filter Topology

namespace Sanguine
namespace NullTransformer

open Sanguine.CrossUniverse

variable {P H dp dff V T : ℕ}

-- ═══════════════════════════════════════════════════════════════════
-- §3b. QKVO CONDENSES — four matrices, TWO circuits, one instruction
-- ═══════════════════════════════════════════════════════════════════
-- The block spends seven contractions, four of them in attention. Three of those four read the
-- SAME vector (LAW 2 ⇒ one instruction, one pass: `qkv_is_one_contraction`). And the four
-- weights are not four semantic objects: `W_Q` and `W_K` only ever meet inside a score, `W_V`
-- and `W_O` only ever meet inside the residual write, so by LAW 1 each pair collapses to ONE
-- `d×d` matrix of rank ≤ d_head — the QK circuit and the OV circuit. RoPE does not obstruct the
-- first collapse; it puts the rotor INSIDE it at the relative offset, which also closes the
-- "Next" `RopeRelative` leaves open in its own header (it proves one rotation-pair and says the
-- step to a whole head "is stated in the @engineer line and nowhere proved").

/-- One `rope` lane, as the Being-cell rotor's first component. -/
theorem rope_lane_false (bl : Block P H dp dff) (m : ℕ) (y : Act P (Dim H dp)) (h : Fin H)
    (p : Fin dp) (θ : Fin P → ℝ) :
    rope bl m y (h, p, false) θ
      = (AttentionSpinor.rotor ((m : ℝ) * bl.freq p) (y (h, p, false) θ)
          (y (h, p, true) θ)).1 := by
  simp [rope, AttentionSpinor.rotor]

/-- The other lane. -/
theorem rope_lane_true (bl : Block P H dp dff) (m : ℕ) (y : Act P (Dim H dp)) (h : Fin H)
    (p : Fin dp) (θ : Fin P → ℝ) :
    rope bl m y (h, p, true) θ
      = (AttentionSpinor.rotor ((m : ℝ) * bl.freq p) (y (h, p, false) θ)
          (y (h, p, true) θ)).2 := by
  simp [rope, AttentionSpinor.rotor]

/-- **The WHOLE HEAD's score is relative.** Not one rotation-pair — the head: summing the rotated
    query against the rotated key over all `dp` cells and both lanes equals reading the
    UN-rotated query against the key rotated by the position DIFFERENCE alone. The absolute
    positions cancel cell by cell because `rope` is block-diagonal in the cells. This is the step
    `RopeRelative` names as open and does not take. -/
theorem head_score_is_relative (bl : Block P H dp dff) (y z : Act P (Dim H dp))
    (h : Fin H) (a b : ℕ) (θ : Fin P → ℝ) :
    (∑ pl : HeadDim dp, rope bl a y (h, pl) θ * rope bl b z (h, pl) θ)
      = ∑ p : Fin dp, RopeRelative.rdot (y (h, p, false) θ, y (h, p, true) θ)
          (AttentionSpinor.rotor (((b : ℝ) - (a : ℝ)) * bl.freq p) (z (h, p, false) θ)
            (z (h, p, true) θ)) := by
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [Fintype.sum_bool, rope_lane_true, rope_lane_true, rope_lane_false, rope_lane_false]
  have hrel := RopeRelative.rope_score_is_relative ((a : ℝ) * bl.freq p) ((b : ℝ) * bl.freq p)
    (y (h, p, false) θ, y (h, p, true) θ) (z (h, p, false) θ, z (h, p, true) θ)
  rw [show ((b : ℝ) * bl.freq p - (a : ℝ) * bl.freq p) = ((b : ℝ) - (a : ℝ)) * bl.freq p from
    by ring] at hrel
  simp only [RopeRelative.rdot] at hrel ⊢
  linarith [hrel]

/-- The rotor is linear, so it commutes with a contraction: rotating the PROJECTED vector is
    contracting with the ROTATED weight. This is what lets the position rotor be absorbed into
    the circuit instead of applied to activations. -/
theorem rotor_of_contraction {n : Type} [Fintype n] (φ : ℝ) (w0 w1 c : n → ℝ) :
    (AttentionSpinor.rotor φ (∑ j, w0 j * c j) (∑ j, w1 j * c j)).1
        = ∑ j, (AttentionSpinor.rotor φ (w0 j) (w1 j)).1 * c j
      ∧ (AttentionSpinor.rotor φ (∑ j, w0 j * c j) (∑ j, w1 j * c j)).2
        = ∑ j, (AttentionSpinor.rotor φ (w0 j) (w1 j)).2 * c j := by
  constructor <;>
  · simp only [AttentionSpinor.rotor, Finset.mul_sum, ← Finset.sum_sub_distrib,
      ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring

/-- The key projection with the relative rotor already folded into the WEIGHT: `R(Δ)·W_K`. -/
def rotKey (bl : Block P H dp dff) (h : Fin H) (Δ : ℝ) (θ : Fin P → ℝ) :
    HeadDim dp → Dim H dp → ℝ := fun pl j =>
  if pl.2 then
    (AttentionSpinor.rotor (Δ * bl.freq pl.1) (θ (bl.wK (h, pl.1, false) j))
      (θ (bl.wK (h, pl.1, true) j))).2
  else
    (AttentionSpinor.rotor (Δ * bl.freq pl.1) (θ (bl.wK (h, pl.1, false) j))
      (θ (bl.wK (h, pl.1, true) j))).1

@[simp] theorem rotKey_false (bl : Block P H dp dff) (h : Fin H) (Δ : ℝ) (θ : Fin P → ℝ)
    (p : Fin dp) (j : Dim H dp) :
    rotKey bl h Δ θ (p, false) j
      = (AttentionSpinor.rotor (Δ * bl.freq p) (θ (bl.wK (h, p, false) j))
          (θ (bl.wK (h, p, true) j))).1 := rfl

@[simp] theorem rotKey_true (bl : Block P H dp dff) (h : Fin H) (Δ : ℝ) (θ : Fin P → ℝ)
    (p : Fin dp) (j : Dim H dp) :
    rotKey bl h Δ θ (p, true) j
      = (AttentionSpinor.rotor (Δ * bl.freq p) (θ (bl.wK (h, p, false) j))
          (θ (bl.wK (h, p, true) j))).2 := rfl

/-- **The QK circuit** — ONE `d×d` matrix per head per relative offset, `W_Qᵀ·R(Δ)·W_K`, of rank
    at most the head dimension. Two of the block's four attention weights, condensed. -/
def qkCircuit (bl : Block P H dp dff) (h : Fin H) (Δ : ℝ) (θ : Fin P → ℝ) :
    Dim H dp → Dim H dp → ℝ :=
  fun i j => ∑ pl : HeadDim dp, θ (bl.wQ (h, pl) i) * rotKey bl h Δ θ pl j

/-- **The OV circuit** — ONE `d×d` matrix per head, `W_O·W_V`, of rank at most the head
    dimension. The other two, condensed. -/
def ovCircuit (bl : Block P H dp dff) (h : Fin H) (θ : Fin P → ℝ) :
    Dim H dp → Dim H dp → ℝ :=
  fun m j => ∑ pl : HeadDim dp, θ (bl.wO m (h, pl)) * θ (bl.wV (h, pl) j)

/-- **Q, K and V are ONE contraction.** They read the same post-norm residual, so LAW 2 makes the
    three projections the components of a single contraction against the stacked weight: one pass
    over the input, one instruction, three output blocks. -/
theorem qkv_is_one_contraction (bl : Block P H dp dff) (n1 : Act P (Dim H dp))
    (θ : Fin P → ℝ) :
    mv (stackRows (stackRows (fun kk i => θ (bl.wQ kk i)) (fun kk i => θ (bl.wK kk i)))
        (fun kk i => θ (bl.wV kk i))) (fun i => n1 i θ)
      = Sum.elim (Sum.elim (mv (fun kk i => θ (bl.wQ kk i)) (fun i => n1 i θ))
          (mv (fun kk i => θ (bl.wK kk i)) (fun i => n1 i θ)))
        (mv (fun kk i => θ (bl.wV kk i)) (fun i => n1 i θ)) := by
  rw [mv_stack, mv_stack]

/-- **The score is ONE bilinear form (headline).** The head's whole rotated query–key sum equals
    the double contraction of the POST-NORM residuals against a single `d×d` matrix
    `W_Qᵀ·R(u−t)·W_K`. `W_Q` and `W_K` never appear apart: two matrices, one circuit, and the
    position enters only as the relative rotor inside it. -/
theorem qk_is_one_circuit (bl : Block P H dp dff) (x : Fin T → Act P (Dim H dp))
    (h : Fin H) (t u : Fin T) (θ : Fin P → ℝ) :
    (∑ pl : HeadDim dp,
        rope bl t.val (lin (wt bl.wQ) (x t)) (h, pl) θ
          * rope bl u.val (lin (wt bl.wK) (x u)) (h, pl) θ)
      = ∑ i, ∑ j, qkCircuit bl h ((u.val : ℝ) - (t.val : ℝ)) θ i j * (x t i θ * x u j θ) := by
  classical
  set Δ : ℝ := (u.val : ℝ) - (t.val : ℝ) with hΔ
  have step1 := head_score_is_relative bl (lin (wt bl.wQ) (x t)) (lin (wt bl.wK) (x u)) h
    t.val u.val θ
  rw [← hΔ] at step1
  rw [step1]
  have step2 : ∀ p : Fin dp,
      RopeRelative.rdot (lin (wt bl.wQ) (x t) (h, p, false) θ,
          lin (wt bl.wQ) (x t) (h, p, true) θ)
        (AttentionSpinor.rotor (Δ * bl.freq p) (lin (wt bl.wK) (x u) (h, p, false) θ)
          (lin (wt bl.wK) (x u) (h, p, true) θ))
      = (∑ i, θ (bl.wQ (h, p, false) i) * x t i θ)
          * (∑ j, rotKey bl h Δ θ (p, false) j * x u j θ)
        + (∑ i, θ (bl.wQ (h, p, true) i) * x t i θ)
          * (∑ j, rotKey bl h Δ θ (p, true) j * x u j θ) := by
    intro p
    obtain ⟨h1, h2⟩ := rotor_of_contraction (Δ * bl.freq p)
      (fun j => θ (bl.wK (h, p, false) j)) (fun j => θ (bl.wK (h, p, true) j))
      (fun j => x u j θ)
    simp only [RopeRelative.rdot, lin, wt, prm, rotKey_false, rotKey_true]
    rw [h1, h2]
  simp only [step2]
  have step3 : (∑ p : Fin dp,
      ((∑ i, θ (bl.wQ (h, p, false) i) * x t i θ)
          * (∑ j, rotKey bl h Δ θ (p, false) j * x u j θ)
        + (∑ i, θ (bl.wQ (h, p, true) i) * x t i θ)
          * (∑ j, rotKey bl h Δ θ (p, true) j * x u j θ)))
      = ∑ pl : HeadDim dp, (∑ i, θ (bl.wQ (h, pl) i) * x t i θ)
          * (∑ j, rotKey bl h Δ θ pl j * x u j θ) := by
    rw [Fintype.sum_prod_type]
    exact Finset.sum_congr rfl fun p _ => by rw [Fintype.sum_bool]; ring
  rw [step3, bilinear_expand]
  rfl

/-- **Any refactorization preserving the circuit is free.** Two blocks whose `W_Q`, `W_K` and
    RoPE frequencies differ arbitrarily produce IDENTICAL scores as soon as their QK circuits
    agree at the offset in question. The four-matrix parametrization carries gauge; only the
    circuit is observable. (The same argument on `ovCircuit` gives the OV gauge.) -/
theorem qk_gauge_freedom (bl bl' : Block P H dp dff) (x : Fin T → Act P (Dim H dp))
    (h : Fin H) (t u : Fin T) (θ : Fin P → ℝ)
    (hcirc : qkCircuit bl h ((u.val : ℝ) - (t.val : ℝ)) θ
      = qkCircuit bl' h ((u.val : ℝ) - (t.val : ℝ)) θ) :
    (∑ pl : HeadDim dp, rope bl t.val (lin (wt bl.wQ) (x t)) (h, pl) θ
        * rope bl u.val (lin (wt bl.wK) (x u)) (h, pl) θ)
      = ∑ pl : HeadDim dp, rope bl' t.val (lin (wt bl'.wQ) (x t)) (h, pl) θ
        * rope bl' u.val (lin (wt bl'.wK) (x u)) (h, pl) θ := by
  rw [qk_is_one_circuit, qk_is_one_circuit, hcirc]

/-- **The value path is ONE circuit per head (headline).** The attention sublayer's write into the
    residual stream — value projection, the attention average, and the output projection — equals
    the attention-weighted sum of `W_O·W_V` applied to the post-norm residuals, head by head.
    `W_V` and `W_O` never appear apart either: two matrices, one circuit. So the block's four
    attention weights are TWO `d×d` objects, each of rank ≤ d_head, and the whole sublayer is
    `Σ_h Σ_u w_u(h)·(OV_h·x_u)` — one instruction under one weighted sum. -/
theorem ov_is_one_circuit (bl : Block P H dp dff) (q k : Fin T → Act P (Dim H dp))
    (x : Fin T → Act P (Dim H dp)) (t : Fin T) (m : Dim H dp) (θ : Fin P → ℝ) :
    lin (wt bl.wO) (attnOut dp q k (fun u => lin (wt bl.wV) (x u)) t) m θ
      = ∑ h : Fin H, ∑ u ∈ pre t,
          (Real.exp (score dp q k h t u θ) / ∑ u' ∈ pre t, Real.exp (score dp q k h t u' θ))
            * mv (ovCircuit bl h θ) (fun j => x u j θ) m := by
  classical
  show (∑ n : Dim H dp, θ (bl.wO m n) * attnOut dp q k (fun u => lin (wt bl.wV) (x u)) t n θ) = _
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun h _ => ?_
  have hw : ∀ pl : HeadDim dp,
      attnOut dp q k (fun u => lin (wt bl.wV) (x u)) t (h, pl) θ
        = ∑ u ∈ pre t,
            (Real.exp (score dp q k h t u θ) / ∑ u' ∈ pre t, Real.exp (score dp q k h t u' θ))
              * (∑ j, θ (bl.wV (h, pl) j) * x u j θ) := by
    intro pl
    exact attnOut_eq_softmax_weighted dp q k (fun u => lin (wt bl.wV) (x u)) t (h, pl) θ
  simp only [hw, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun u _ => ?_
  simp only [mv, ovCircuit, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun pl _ => by ring

/-- **QKVO is two circuits and one instruction (headline).** For every head of every block:
    (1) Q, K and V are a single contraction of a single input (`qkv_is_one_contraction`);
    (2) the score is one bilinear form against `W_Qᵀ·R(u−t)·W_K` (`qk_is_one_circuit`), the whole
        head, with the position present only as the relative rotor;
    (3) the residual write is one contraction against `W_O·W_V` under the attention weights
        (`ov_is_one_circuit`);
    (4) so any reparametrization of the four matrices that preserves those two circuits changes
        nothing (`qk_gauge_freedom`).
    Four `d×d_head` matrices per head are a FACTORIZATION of two `d×d` rank-`d_head` circuits.
    The factorization is a compute choice; the function is the circuits. -/
theorem attention_is_two_circuits (bl : Block P H dp dff) (x : Fin T → Act P (Dim H dp))
    (q k : Fin T → Act P (Dim H dp)) (h : Fin H) (t u : Fin T) (m : Dim H dp)
    (θ : Fin P → ℝ) :
    (∑ pl : HeadDim dp,
        rope bl t.val (lin (wt bl.wQ) (x t)) (h, pl) θ
          * rope bl u.val (lin (wt bl.wK) (x u)) (h, pl) θ
      = ∑ i, ∑ j, qkCircuit bl h ((u.val : ℝ) - (t.val : ℝ)) θ i j * (x t i θ * x u j θ))
    ∧ (lin (wt bl.wO) (attnOut dp q k (fun w => lin (wt bl.wV) (x w)) t) m θ
      = ∑ h' : Fin H, ∑ w ∈ pre t,
          (Real.exp (score dp q k h' t w θ) / ∑ u' ∈ pre t, Real.exp (score dp q k h' t u' θ))
            * mv (ovCircuit bl h' θ) (fun j => x w j θ) m) :=
  ⟨qk_is_one_circuit bl x h t u θ, ov_is_one_circuit bl q k x t m θ⟩

-- ═══════════════════════════════════════════════════════════════════
-- §3c. THE SHAPE — position is a point on a torus, and that IS the KV resolution
-- ═══════════════════════════════════════════════════════════════════
-- What §3b left standing: the score is one bilinear form with a rotor inside it, at the
-- relative offset. That rotor is the whole of position, and it is COMPACT. Read the shape off:
--   · the residual stream is `H·dp` Being cells — `H·dp` 2-planes (`Dim H dp = Fin H × Fin dp × Bool`);
--   · a head's `dp` frequencies send an offset `Δ` to ONE point of the `dp`-torus,
--     `(cos Δω_p, sin Δω_p)_p` — a winding line in `T^{dp}`;
--   · the score is a trigonometric polynomial in `Δ` with exactly those `dp` frequencies.
-- So the two cells split the two jobs, and this is the whole architecture in one line: the BEING
-- cell (the rotor, a circle — compact) carries POSITION, and the NULL cell (the boost, a
-- hyperbola — non-compact) carries WEIGHT (§1). Compact ⇒ the position channel costs a bounded
-- number of reals per key however long the context is; non-compact ⇒ the weight channel is what
-- has to be budgeted, which is exactly what §7 bounds.
--
-- The KV consequence is sharp and is the theorem below: a head sees position ONLY through its
-- torus point, and sees ALL of it. Two offsets are indistinguishable — for EVERY content, for
-- EVERY weight matrix — exactly when they land on the same point. Choosing RoPE frequencies is
-- therefore choosing the injectivity radius of the KV cache, not tuning a hyperparameter.

/-- A head's relative score, as a function of the offset and the PRE-ROTATION content:
    `∑_p ⟪a_p, R(Δ·ω_p)·b_p⟫`. `head_score_is_relative` proves this is what the head computes. -/
def headScore {dp : ℕ} (ω : Fin dp → ℝ) (Δ : ℝ) (a b : Fin dp → ℝ × ℝ) : ℝ :=
  ∑ p, RopeRelative.rdot (a p) (AttentionSpinor.rotor (Δ * ω p) (b p).1 (b p).2)

/-- **The head's position channel**: the point of the `dp`-torus the offset lands on, in the
    `(cos, sin)` coordinates of each cell. `2·dp` reals per head — bounded, whatever `Δ` is. -/
def torusPoint {dp : ℕ} (ω : Fin dp → ℝ) (Δ : ℝ) : Fin dp → ℝ × ℝ :=
  fun p => (Real.cos (Δ * ω p), Real.sin (Δ * ω p))

/-- The model's head sum IS `headScore` at the relative offset — `head_score_is_relative`, renamed
    onto the object §3c reasons about. -/
theorem head_sum_eq_headScore (bl : Block P H dp dff) (y z : Act P (Dim H dp))
    (h : Fin H) (a b : ℕ) (θ : Fin P → ℝ) :
    (∑ pl : HeadDim dp, rope bl a y (h, pl) θ * rope bl b z (h, pl) θ)
      = headScore bl.freq ((b : ℝ) - (a : ℝ))
          (fun p => (y (h, p, false) θ, y (h, p, true) θ))
          (fun p => (z (h, p, false) θ, z (h, p, true) θ)) :=
  head_score_is_relative bl y z h a b θ

/-- Probing one cell: with the content supported on cell `p` alone, the head's score is that
    cell's rotated inner product and nothing else. The cells do not mix. -/
theorem headScore_single {dp : ℕ} (ω : Fin dp → ℝ) (Δ : ℝ) (p : Fin dp) (u v : ℝ × ℝ) :
    headScore ω Δ (fun q => if q = p then u else (0, 0))
        (fun q => if q = p then v else (0, 0))
      = RopeRelative.rdot u (AttentionSpinor.rotor (Δ * ω p) v.1 v.2) := by
  classical
  unfold headScore
  rw [Finset.sum_eq_single p]
  · simp
  · intro q _ hq; simp [hq, RopeRelative.rdot]
  · intro hp; exact absurd (Finset.mem_univ p) hp

/-- **Position enters only through the torus point.** Two offsets landing on the same point of the
    torus give the identical score for every content — the score cannot see `Δ` itself, only where
    `Δ` lands. -/
theorem headScore_factors_through_torus {dp : ℕ} (ω : Fin dp → ℝ) (Δ Δ' : ℝ)
    (h : torusPoint ω Δ = torusPoint ω Δ') (a b : Fin dp → ℝ × ℝ) :
    headScore ω Δ a b = headScore ω Δ' a b := by
  unfold headScore
  refine Finset.sum_congr rfl fun p _ => ?_
  have hc : Real.cos (Δ * ω p) = Real.cos (Δ' * ω p) := congrArg Prod.fst (congrFun h p)
  have hs : Real.sin (Δ * ω p) = Real.sin (Δ' * ω p) := congrArg Prod.snd (congrFun h p)
  simp only [RopeRelative.rdot, AttentionSpinor.rotor, hc, hs]

/-- **And ALL of the torus point is recovered.** If two offsets score identically on every content
    then their torus points coincide: probing cell `p` with `(1,0)` against `(1,0)` reads `cos Δω_p`
    and against `(0,1)` reads `−sin Δω_p`. Nothing about position is hidden from the score, and
    nothing beyond the torus point is exposed to it. -/
theorem torus_point_is_recovered {dp : ℕ} (ω : Fin dp → ℝ) (Δ Δ' : ℝ)
    (hEq : ∀ a b, headScore ω Δ a b = headScore ω Δ' a b) :
    torusPoint ω Δ = torusPoint ω Δ' := by
  classical
  funext p
  have hc := hEq (fun q => if q = p then (1, 0) else (0, 0))
    (fun q => if q = p then (1, 0) else (0, 0))
  have hs := hEq (fun q => if q = p then (1, 0) else (0, 0))
    (fun q => if q = p then (0, 1) else (0, 0))
  rw [headScore_single, headScore_single] at hc
  rw [headScore_single, headScore_single] at hs
  simp only [RopeRelative.rdot, AttentionSpinor.rotor] at hc hs
  refine Prod.ext ?_ ?_
  · simpa [torusPoint] using hc
  · have : -Real.sin (Δ * ω p) = -Real.sin (Δ' * ω p) := by simpa using hs
    simpa [torusPoint] using neg_injective this

/-- **THE POSITIONAL RESOLUTION LAW (headline).** A head separates two relative offsets **iff**
    they land on different points of its `dp`-torus:

      `(∀ content, score Δ = score Δ') ↔ torusPoint ω Δ = torusPoint ω Δ'`.

    Both directions matter and they say different things. `←` is the ceiling: coincident torus
    points are indistinguishable **for every content and every weight matrix**, so no training and
    no capacity can separate them — it is not a limitation of the model, it is the shape of the
    channel. `→` is the floor: the torus point is fully recovered, so a head's positional
    information is *exactly* `2·dp` reals, bounded, no matter how long the context is.

    Read on the KV cache: the stored key is `content ⊗ torus point` and the value carries no
    position at all (`ovCircuit` has no offset in it — the V half is a plain content vector). The
    position half is COMPACT, which is why it costs a fixed number of reals per key at any context
    length; the weight half is the null cell's boost, non-compact, and is what §7 has to budget. -/
theorem head_separates_offsets_iff_torus {dp : ℕ} (ω : Fin dp → ℝ) (Δ Δ' : ℝ) :
    (∀ a b, headScore ω Δ a b = headScore ω Δ' a b) ↔ torusPoint ω Δ = torusPoint ω Δ' :=
  ⟨torus_point_is_recovered ω Δ Δ',
   fun h a b => headScore_factors_through_torus ω Δ Δ' h a b⟩

/-- The squared separation of two offsets on the head's torus — how far apart the position
    channel actually places them. -/
def torusSep {dp : ℕ} (ω : Fin dp → ℝ) (Δ Δ' : ℝ) : ℝ :=
  ∑ p, ((Real.cos (Δ * ω p) - Real.cos (Δ' * ω p)) ^ 2
      + (Real.sin (Δ * ω p) - Real.sin (Δ' * ω p)) ^ 2)

/-- **The discriminability curve.** The separation collapses to `∑_p 2·(1 − cos((Δ−Δ')·ω_p))` — a
    sum of per-cell cosine bowls that depends only on the DIFFERENCE of the offsets. This is the
    design curve for a frequency set: it is what has to stay away from zero across the context you
    intend to serve, and it is computable from `ω` alone, before any weight exists. -/
theorem torusSep_eq {dp : ℕ} (ω : Fin dp → ℝ) (Δ Δ' : ℝ) :
    torusSep ω Δ Δ' = ∑ p, 2 * (1 - Real.cos ((Δ - Δ') * ω p)) := by
  unfold torusSep
  refine Finset.sum_congr rfl fun p _ => ?_
  have hsub : (Δ - Δ') * ω p = Δ * ω p - Δ' * ω p := by ring
  rw [hsub, Real.cos_sub]
  have h1 : Real.sin (Δ * ω p) ^ 2 + Real.cos (Δ * ω p) ^ 2 = 1 := Real.sin_sq_add_cos_sq _
  have h2 : Real.sin (Δ' * ω p) ^ 2 + Real.cos (Δ' * ω p) ^ 2 = 1 := Real.sin_sq_add_cos_sq _
  nlinarith [h1, h2]

/-- **Zero separation IS indistinguishability.** The curve vanishes exactly at the offsets a head
    cannot tell apart, so `torusSep` is a faithful measure of the resolution law, not a proxy for
    it: `head_separates_offsets_iff_torus` says the score separates `Δ` from `Δ'` iff this is
    nonzero. -/
theorem torusSep_zero_iff {dp : ℕ} (ω : Fin dp → ℝ) (Δ Δ' : ℝ) :
    torusSep ω Δ Δ' = 0 ↔ torusPoint ω Δ = torusPoint ω Δ' := by
  constructor
  · intro h
    funext p
    have hnn : ∀ q ∈ (Finset.univ : Finset (Fin dp)),
        0 ≤ (Real.cos (Δ * ω q) - Real.cos (Δ' * ω q)) ^ 2
          + (Real.sin (Δ * ω q) - Real.sin (Δ' * ω q)) ^ 2 := fun q _ => by positivity
    have hp := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp h p (Finset.mem_univ p)
    have hcsq : (Real.cos (Δ * ω p) - Real.cos (Δ' * ω p)) ^ 2 = 0 := by
      nlinarith [sq_nonneg (Real.cos (Δ * ω p) - Real.cos (Δ' * ω p)),
        sq_nonneg (Real.sin (Δ * ω p) - Real.sin (Δ' * ω p))]
    have hssq : (Real.sin (Δ * ω p) - Real.sin (Δ' * ω p)) ^ 2 = 0 := by
      nlinarith [sq_nonneg (Real.cos (Δ * ω p) - Real.cos (Δ' * ω p)),
        sq_nonneg (Real.sin (Δ * ω p) - Real.sin (Δ' * ω p))]
    have hc : Real.cos (Δ * ω p) = Real.cos (Δ' * ω p) :=
      sub_eq_zero.mp ((pow_eq_zero_iff (by norm_num)).mp hcsq)
    have hs : Real.sin (Δ * ω p) = Real.sin (Δ' * ω p) :=
      sub_eq_zero.mp ((pow_eq_zero_iff (by norm_num)).mp hssq)
    simp only [torusPoint, hc, hs]
  · intro h
    unfold torusSep
    refine Finset.sum_eq_zero fun p _ => ?_
    have hc : Real.cos (Δ * ω p) = Real.cos (Δ' * ω p) := congrArg Prod.fst (congrFun h p)
    have hs : Real.sin (Δ * ω p) = Real.sin (Δ' * ω p) := congrArg Prod.snd (congrFun h p)
    rw [hc, hs]; ring

/-- **Aliasing: a common period is a hard ceiling on the cache (headline).** If every frequency of
    a head completes a whole number of turns in `L` — `L·ω_p ∈ 2π·ℤ` for all `p` — then offsets `L`
    apart score identically for every content. The head cannot tell position `u` from position
    `u + L`, and no choice of `W_Q`, `W_K` or training changes that: the torus point is the same
    point.

    So the RoPE frequency set fixes the KV cache's **injectivity radius**. Rationally-related
    frequencies close the winding line and cap the distinguishable context at the common period;
    keeping the line from closing is a property of the frequencies, chosen before any weight is
    trained. This is what `AttentionSpinor.rotor_full_turn` says for ONE rotor, at a whole head. -/
theorem headScore_aliases_at_common_period {dp : ℕ} (ω : Fin dp → ℝ) (L : ℝ)
    (hL : ∀ p, ∃ k : ℤ, L * ω p = (k : ℝ) * (2 * Real.pi)) (Δ : ℝ) (a b : Fin dp → ℝ × ℝ) :
    headScore ω (Δ + L) a b = headScore ω Δ a b := by
  refine headScore_factors_through_torus ω (Δ + L) Δ ?_ a b
  funext p
  obtain ⟨k, hk⟩ := hL p
  have hsplit : (Δ + L) * ω p = Δ * ω p + (k : ℝ) * (2 * Real.pi) := by
    rw [add_mul, hk]
  simp only [torusPoint, hsplit, Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]

/-- **The model's head aliases too** — the ceiling, transported onto the network's own attention by
    `head_score_is_relative`. Positions whose offsets differ by a common period of the block's
    frequencies produce the identical head score, for any content and any `θ`. -/
theorem model_head_aliases_at_common_period (bl : Block P H dp dff) (y z : Act P (Dim H dp))
    (h : Fin H) (a b : ℕ) (L : ℝ) (θ : Fin P → ℝ)
    (hL : ∀ p, ∃ k : ℤ, L * bl.freq p = (k : ℝ) * (2 * Real.pi)) :
    headScore bl.freq (((b : ℝ) - (a : ℝ)) + L)
        (fun p => (y (h, p, false) θ, y (h, p, true) θ))
        (fun p => (z (h, p, false) θ, z (h, p, true) θ))
      = ∑ pl : HeadDim dp, rope bl a y (h, pl) θ * rope bl b z (h, pl) θ := by
  rw [head_sum_eq_headScore]
  exact headScore_aliases_at_common_period bl.freq L hL _ _ _

-- ═══════════════════════════════════════════════════════════════════
-- §4. THE LOSS — mean next-token cross-entropy, and its log-sum-exp face
-- ═══════════════════════════════════════════════════════════════════

/-- The per-position cross-entropy of the logits against the target token: `−log softmax_z(y)`. -/
def ceAt {ι : Type} [Fintype ι] (z : ι → ℝ) (y : ι) : ℝ := -Real.log (softmax z y)

/-- **Cross-entropy IS log-sum-exp minus the target logit.** The identity every trainer uses to
    avoid the division — and it is exact, the drag's log subtracted once instead of divided in.
    (`B/U` under `log` is `log B − log U`: the bulk ratio linearized.) -/
theorem crossEntropy_eq_logSumExp {ι : Type} [Fintype ι] [Nonempty ι] (z : ι → ℝ) (y : ι) :
    ceAt z y = Real.log (∑ j, Real.exp (z j)) - z y := by
  have hpos : 0 < ∑ j, Real.exp (z j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty
  unfold ceAt softmax
  rw [Real.log_div (Real.exp_ne_zero _) (ne_of_gt hpos), Real.log_exp]
  ring

/-- **The training objective** — mean next-token cross-entropy over the sequence, in the
    log-sum-exp form (so no division is taken and none has to be proved nonzero). -/
def netLoss (ε : ℝ) (net : Net P H dp dff V) (toks : Fin T → Fin V) (tgt : Fin T → Fin V) :
    Node P :=
  fun θ => (∑ t, (Real.log (∑ w, Real.exp (netLogits ε net toks t w θ))
    - netLogits ε net toks t (tgt t) θ)) / (T : ℝ)

/-- **The objective is the mean cross-entropy** — `netLoss` really is `(1/T)·Σₜ CE(zₜ, yₜ)`,
    by `crossEntropy_eq_logSumExp` at every position. -/
theorem netLoss_is_mean_cross_entropy {ε : ℝ} [NeZero V] (net : Net P H dp dff V)
    (toks tgt : Fin T → Fin V) (θ : Fin P → ℝ) :
    netLoss ε net toks tgt θ
      = (∑ t, ceAt (fun w => netLogits ε net toks t w θ) (tgt t)) / (T : ℝ) := by
  have : Nonempty (Fin V) := ⟨⟨0, Nat.pos_of_ne_zero (NeZero.ne V)⟩⟩
  unfold netLoss
  congr 1
  exact Finset.sum_congr rfl fun t _ =>
    (crossEntropy_eq_logSumExp (fun w => netLogits ε net toks t w θ) (tgt t)).symm

-- ═══════════════════════════════════════════════════════════════════
-- §5. THE HEADLINE — THE WHOLE PIPELINE IS DIFFERENTIABLE IN θ
-- ═══════════════════════════════════════════════════════════════════

/-- **The end-to-end loss is differentiable in the parameters (headline).** For any width `H·dp`,
    any MLP width `dff`, any vocabulary `V`, any sequence length `T`, any depth (any block list),
    any token stream and any targets, with `ε > 0`, the map
    `θ ↦ mean next-token cross-entropy` is `Differentiable ℝ` at **every** θ — embedding, RMSNorm,
    RoPE, causal multi-head attention, SwiGLU, both residuals, the final norm, the unembedding and
    the log-sum-exp, composed.

    So the gradient exists everywhere: there is no point of the parameter space at which this
    model's training signal is undefined, and backprop (the chain rule, `fderiv_comp`, run
    right-to-left over exactly these combinators) computes it. The two places a transformer could
    have gone singular are both closed upstream: `√` never meets `0` because `ε > 0`
    (`rms_denom_pos`), and the softmax drag is never `0` because `exp > 0` on a nonempty causal
    prefix (`attn_denom_pos`) — the null cell's own condition, kept off the cone. -/
theorem netLoss_differentiable {ε : ℝ} (hε : 0 < ε) [NeZero V] (net : Net P H dp dff V)
    (toks tgt : Fin T → Fin V) :
    Differentiable ℝ (netLoss ε net toks tgt) := by
  have : Nonempty (Fin V) := ⟨⟨0, Nat.pos_of_ne_zero (NeZero.ne V)⟩⟩
  unfold netLoss
  refine differentiable_div_const ?_ _
  refine Differentiable.fun_sum fun t _ => Differentiable.sub ?_ ?_
  · refine Differentiable.log ?_ ?_
    · exact Differentiable.fun_sum fun w _ =>
        Differentiable.exp (netLogits_differentiable hε net toks t w)
    · intro θ
      exact ne_of_gt (Finset.sum_pos (fun w _ => Real.exp_pos _) Finset.univ_nonempty)
  · exact netLogits_differentiable hε net toks t (tgt t)

/-- **Not vacuous — a concrete instantiation.** A 64-slot parameter vector, 2 heads × 2 cells
    (`d_model = 8`), MLP width 8, vocabulary 4, three positions, one block: the hypotheses of
    `netLoss_differentiable` are satisfiable and the theorem applies to a real shape. -/
example :
    Differentiable ℝ (netLoss (P := 64) (H := 2) (dp := 2) (dff := 8) (V := 4) (T := 3) (1 / 2)
      { emb := fun _ _ => 0,
        blocks := [{ gAttn := fun _ => 0, wQ := fun _ _ => 0, wK := fun _ _ => 0,
                     wV := fun _ _ => 0, wO := fun _ _ => 0, gMlp := fun _ => 0,
                     wGate := fun _ _ => 0, wUp := fun _ _ => 0, wDown := fun _ _ => 0,
                     freq := fun _ => 1 }],
        gFinal := fun _ => 0, unemb := fun _ _ => 0 }
      (fun _ => 0) (fun _ => 0)) :=
  netLoss_differentiable (by norm_num) _ _ _

-- ═══════════════════════════════════════════════════════════════════
-- §6. THE STEP — the gradient, and that descending it lowers the loss
-- ═══════════════════════════════════════════════════════════════════

/-- **The gradient** — the derivative against each basis slot, i.e. the vector of partial
    derivatives. This is exactly what a reverse-mode pass returns. -/
def grad {P : ℕ} (f : Node P) (θ : Fin P → ℝ) : Fin P → ℝ :=
  fun i => fderiv ℝ f θ (Pi.single i 1)

/-- **The derivative IS the gradient's inner product.** The directional derivative along any `v`
    is `Σᵢ vᵢ·(∇f)ᵢ` — the reverse-mode sum. Backprop returns `grad`, and this says `grad`
    determines the whole derivative. No differentiability hypothesis is needed: `fderiv` is a
    linear map by construction, so the decomposition holds of it whatever it is; what
    differentiability buys is that this linear map is the actual derivative
    (`netLoss_differentiable`). -/
theorem fderiv_eq_sum_grad {P : ℕ} {f : Node P} (θ v : Fin P → ℝ) :
    fderiv ℝ f θ v = ∑ i, v i * grad f θ i := by
  classical
  have hv : v = ∑ i, (v i) • (Pi.single (M := fun _ : Fin P => ℝ) i 1) := by
    funext j
    rw [Finset.sum_apply]
    rw [Finset.sum_eq_single j (fun i _ hij => by simp [Ne.symm hij])
      (fun h => absurd (Finset.mem_univ j) h)]
    simp
  calc fderiv ℝ f θ v
      = fderiv ℝ f θ (∑ i, (v i) • (Pi.single (M := fun _ : Fin P => ℝ) i 1)) := by rw [← hv]
    _ = ∑ i, v i * grad f θ i := by
        rw [map_sum]
        exact Finset.sum_congr rfl fun i _ => by rw [map_smul]; simp [grad, smul_eq_mul]

/-- **One gradient-descent step** — `θ ↦ θ − η·∇f(θ)`. The whole update rule; a preconditioner
    (Adam's diagonal, Muon's whitening) multiplies `grad` and changes nothing below. -/
def sgdStep {P : ℕ} (f : Node P) (η : ℝ) (θ : Fin P → ℝ) : Fin P → ℝ := θ - η • grad f θ

/-- The step's directional derivative at `η = 0` is `−‖∇f‖²`. -/
theorem sgd_deriv_at_zero {P : ℕ} {f : Node P} (hf : Differentiable ℝ f) (θ : Fin P → ℝ) :
    HasDerivAt (fun η : ℝ => f (sgdStep f η θ)) (-∑ i, grad f θ i * grad f θ i) 0 := by
  have hline : HasDerivAt (fun η : ℝ => sgdStep f η θ) (-grad f θ) 0 := by
    have : (fun η : ℝ => sgdStep f η θ) = fun η : ℝ => θ + η • (-grad f θ) := by
      funext η; simp [sgdStep, sub_eq_add_neg, smul_neg]
    rw [this]
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const (-grad f θ)).const_add θ
  have hcomp := (hf (sgdStep f 0 θ)).hasFDerivAt.comp_hasDerivAt 0 hline
  have hval : fderiv ℝ f (sgdStep f 0 θ) (-grad f θ) = -∑ i, grad f θ i * grad f θ i := by
    have h0 : sgdStep f 0 θ = θ := by simp [sgdStep]
    rw [h0, fderiv_eq_sum_grad]
    simp only [Pi.neg_apply, neg_mul, Finset.sum_neg_distrib]
  rw [hval] at hcomp
  exact hcomp

/-- **A small enough gradient step strictly decreases the loss (headline).** If the gradient at
    `θ` is nonzero, then for every sufficiently small positive learning rate the step lands
    strictly below: `f(θ − η∇f(θ)) < f(θ)`. Training is possible — descent is not merely defined,
    it descends. (Nothing here claims a rate, a minimum, or that any fixed `η` works: the
    quantifier is `∀ᶠ η in 𝓝[>] 0`, which is exactly what the first derivative licenses.) -/
theorem sgd_step_decreases {P : ℕ} {f : Node P} (hf : Differentiable ℝ f) (θ : Fin P → ℝ)
    (hg : grad f θ ≠ 0) :
    ∀ᶠ η in 𝓝[>] (0 : ℝ), f (sgdStep f η θ) < f θ := by
  classical
  set g := grad f θ with hgdef
  have hnormpos : 0 < ∑ i, g i * g i := by
    rcases Function.ne_iff.mp hg with ⟨i₀, hi₀⟩
    refine Finset.sum_pos' (fun i _ => mul_self_nonneg (g i)) ⟨i₀, Finset.mem_univ i₀, ?_⟩
    exact mul_self_pos.mpr (by simpa using hi₀)
  have hd := sgd_deriv_at_zero hf θ
  have hneg : (-∑ i, g i * g i) < 0 := by simpa using hnormpos
  have hslope := hasDerivAt_iff_tendsto_slope.mp hd
  have hev : ∀ᶠ y in 𝓝[≠] (0 : ℝ),
      slope (fun η : ℝ => f (sgdStep f η θ)) 0 y < 0 :=
    hslope.eventually_lt_const hneg
  have hle : 𝓝[>] (0 : ℝ) ≤ 𝓝[≠] (0 : ℝ) := nhdsWithin_mono 0 (fun x hx => ne_of_gt hx)
  have h0 : sgdStep f 0 θ = θ := by simp [sgdStep]
  filter_upwards [hev.filter_mono hle, self_mem_nhdsWithin] with η hslp hpos
  rw [slope_def_field, h0, sub_zero] at hslp
  have hpos' : (0 : ℝ) < η := hpos
  have hnum : f (sgdStep f η θ) - f θ < 0 := by
    by_contra hc
    push Not at hc
    exact absurd hslp (not_lt.mpr (div_nonneg hc (le_of_lt hpos')))
  linarith

-- ═══════════════════════════════════════════════════════════════════
-- §6b. THE PIPELINE — a batch, a run of steps, and the loss going down
-- ═══════════════════════════════════════════════════════════════════

/-- **The batch objective** — the mean loss over `B` sequences. The dataset enters here and
    nowhere else; nothing about the model changed. -/
def batchLoss (ε : ℝ) (net : Net P H dp dff V) {B : ℕ}
    (toks tgt : Fin B → Fin T → Fin V) : Node P :=
  fun θ => (∑ b, netLoss ε net (toks b) (tgt b) θ) / (B : ℝ)

/-- **The batch objective is differentiable too** — a mean of differentiable losses. -/
theorem batchLoss_differentiable {ε : ℝ} (hε : 0 < ε) [NeZero V] (net : Net P H dp dff V)
    {B : ℕ} (toks tgt : Fin B → Fin T → Fin V) :
    Differentiable ℝ (batchLoss ε net toks tgt) :=
  differentiable_div_const
    (Differentiable.fun_sum fun b _ => netLoss_differentiable hε net (toks b) (tgt b)) _

/-- **A decreasing step always exists.** Wherever the gradient is nonzero there is a strictly
    positive learning rate that strictly lowers the objective — the witness extracted from
    `sgd_step_decreases` (the filter `𝓝[>] 0` is nontrivial). Training never gets stuck for want
    of a rate; it stops only where the gradient vanishes. -/
theorem exists_decreasing_step {P : ℕ} {f : Node P} (hf : Differentiable ℝ f) (θ : Fin P → ℝ)
    (hg : grad f θ ≠ 0) : ∃ η : ℝ, 0 < η ∧ f (sgdStep f η θ) < f θ := by
  obtain ⟨η, hlt, hpos⟩ :=
    ((sgd_step_decreases hf θ hg).and self_mem_nhdsWithin).exists
  exact ⟨η, hpos, hlt⟩

/-- **The training run** — `n` gradient steps with the schedule `η`, from `θ`. The whole loop:
    data (`batchLoss`) → forward (`netLogits`) → loss → gradient (`grad`) → step (`sgdStep`) →
    repeat. -/
def trainRun {P : ℕ} (f : Node P) (η : ℕ → ℝ) : ℕ → (Fin P → ℝ) → (Fin P → ℝ)
  | 0, θ => θ
  | n + 1, θ => sgdStep f (η n) (trainRun f η n θ)

/-- **The run never goes up.** If every step of the schedule is admissible at the state it is
    taken from (which `exists_decreasing_step` says is always achievable off a stationary point),
    the objective after `n` steps is at most the objective before the first. Descent composes. -/
theorem trainRun_nonincreasing {P : ℕ} (f : Node P) (η : ℕ → ℝ) (θ : Fin P → ℝ) :
    ∀ n : ℕ, (∀ k < n, f (sgdStep f (η k) (trainRun f η k θ)) ≤ f (trainRun f η k θ)) →
      f (trainRun f η n θ) ≤ f θ
  | 0, _ => le_refl _
  | n + 1, h => by
      have hstep : f (trainRun f η (n + 1) θ) ≤ f (trainRun f η n θ) := h n (Nat.lt_succ_self n)
      exact le_trans hstep
        (trainRun_nonincreasing f η θ n (fun k hk => h k (Nat.lt_succ_of_lt hk)))

-- ═══════════════════════════════════════════════════════════════════
-- §7. INFERENCE — what replaces the T² softmax read, and what licenses it
-- ═══════════════════════════════════════════════════════════════════
-- Training needs the exact `attnOut` above. Serving does not need to READ every key. Three
-- statements, in the order they matter:
--   (a) the read is ONE bulk ratio — numerator and drag, not `T` divisions (`attnOut_is_bulk_ratio`);
--   (b) that ratio is computed EXACTLY by an O(1)-state streaming scan — no `T²` table, no
--       approximation (`attnOut_streams_in_O1_state`, riding `softmax_attention_is_online_spinor`);
--   (c) restricting the read to a BUDGETED support and renormalizing on it deviates by at most
--       `2M·(dropped mass / total mass)` — a bound with NO `T` in it
--       (`budget_attention_gap_is_context_free`, riding `bounded_support_gap`). Zero dropped mass
--       is exact; a dropped mass under the emission margin leaves the token unchanged
--       (`SparseSelection.sparse_emits_same_token`).
-- So the replacement is not a different function: it is the SAME softmax, read off a bounded
-- support with an exact renormalizer. The two cheaper "replacements" are refuted upstream and are
-- named here so they are not re-tried: a fixed additive carry (`A = I`) is ORDER-BLIND
-- (`AttentionSpinor.spinor_scan_is_order_sensitive`), and a random-feature map is not an isometry
-- (`AttentionSpinor.rotor_isometry` is what a working carry needs).

/-- The causal prefix of position `t`, head-lane `i`, as the `(score, value)` STREAM the engine
    actually feeds the online-softmax state — one pair per key, in prefix order. -/
def attnStream (dp : ℕ) (q k v : Fin T → Act P (Dim H dp)) (t : Fin T) (i : Dim H dp)
    (θ : Fin P → ℝ) : List (ℝ × ℝ) :=
  (pre t).toList.map (fun u => (score dp q k i.1 t u θ, v u i θ))

/-- **The read streams in O(1) state, exactly — unconditionally.** Feed this position's own
    `(score, value)` stream through the online-softmax carry `(m, den, num)` — three numbers,
    whatever the context length — and `num/den` IS `attnOut`, bit for bit. This is
    `AttentionSpinor.softmax_attention_is_online_spinor` instantiated at this file's attention:
    no `T×T` score matrix is ever formed, the state is context-free, and nothing is approximated.
    The `T²` in a transformer's attention is a *schedule*, not the operator. -/
theorem attnOut_streams_in_O1_state (dp : ℕ) (q k v : Fin T → Act P (Dim H dp))
    (t : Fin T) (i : Dim H dp) (θ : Fin P → ℝ) :
    AttentionSpinor.oaNum (AttentionSpinor.oaScan (attnStream dp q k v t i θ))
        / AttentionSpinor.oaDen (AttentionSpinor.oaScan (attnStream dp q k v t i θ))
      = attnOut dp q k v t i θ := by
  obtain ⟨hnum, hden, _⟩ :=
    AttentionSpinor.softmax_attention_is_online_spinor (attnStream dp q k v t i θ)
  rw [hnum, hden, attnOut_is_bulk_ratio]
  congr 1 <;> · unfold attnStream; simp [List.map_map, Function.comp_def]

/-- This position's attention scores, extended by zero off the causal prefix — the same read,
    written over the whole sequence so a budgeted support can be compared against it. -/
def maskedScore (dp : ℕ) (q k : Fin T → Act P (Dim H dp)) (t : Fin T) (h : Fin H)
    (θ : Fin P → ℝ) : Fin T → ℝ :=
  fun u => if u ∈ pre t then Real.exp (score dp q k h t u θ) else 0

theorem maskedScore_nonneg (dp : ℕ) (q k : Fin T → Act P (Dim H dp)) (t : Fin T) (h : Fin H)
    (θ : Fin P → ℝ) (u : Fin T) : 0 ≤ maskedScore dp q k t h θ u := by
  unfold maskedScore; split
  · exact le_of_lt (Real.exp_pos _)
  · exact le_refl 0

theorem sum_maskedScore_subset (dp : ℕ) (q k : Fin T → Act P (Dim H dp)) (t : Fin T) (h : Fin H)
    (θ : Fin P → ℝ) (g : Fin T → ℝ) {A : Finset (Fin T)} (hA : A ⊆ pre t) :
    ∑ u ∈ A, maskedScore dp q k t h θ u * g u
      = ∑ u ∈ A, Real.exp (score dp q k h t u θ) * g u :=
  Finset.sum_congr rfl fun u hu => by simp [maskedScore, hA hu]

theorem sum_maskedScore_univ (dp : ℕ) (q k : Fin T → Act P (Dim H dp)) (t : Fin T) (h : Fin H)
    (θ : Fin P → ℝ) (g : Fin T → ℝ) :
    ∑ u, maskedScore dp q k t h θ u * g u
      = ∑ u ∈ pre t, Real.exp (score dp q k h t u θ) * g u := by
  classical
  simp only [maskedScore, ite_mul, zero_mul, Finset.sum_ite_mem, Finset.univ_inter]

/-- **The masked read IS the causal read** — extending the scores by zero off the prefix changes
    nothing, so the budget bound below is a statement about this transformer's own attention. -/
theorem att_maskedScore (dp : ℕ) (q k v : Fin T → Act P (Dim H dp)) (t : Fin T) (i : Dim H dp)
    (θ : Fin P → ℝ) :
    AttentionSpinor.att (maskedScore dp q k t i.1 θ) (fun u => v u i θ)
      = attnOut dp q k v t i θ := by
  unfold AttentionSpinor.att
  rw [attnOut_is_bulk_ratio, sum_maskedScore_univ]
  congr 1
  simpa using sum_maskedScore_univ dp q k t i.1 θ (fun _ => 1)

/-- **The budgeted read of THIS transformer, bounded by the mass it drops (headline).** Serve
    position `t` from any retained key budget `S ⊆ pre t` — renormalizing on `S`, which is what a
    block-bound gather over the heavy keys computes — and the answer differs from the full causal
    attention by at most `2M·(dropped mass / total mass)`, `M` bounding the values. **No `T`
    appears in the bound.** The cost of serving is set by how much score-mass the budget omits,
    not by how long the context is: keep a fixed budget of keys hot, the rest cold, and the read
    is exact to its tail. Zero dropped mass is exact
    (`AttentionSpinor.sparse_exact_on_support`); a tail under the emission margin leaves the
    emitted token identical (`SparseSelection.sparse_emits_same_token`). This is the licence for
    flat-VRAM inference, and it is the replacement for the `T²` read — the same softmax, over a
    bounded support, with an exact renormalizer. -/
theorem attnOut_budget_gap (dp : ℕ) (q k v : Fin T → Act P (Dim H dp)) (t : Fin T)
    (i : Dim H dp) (θ : Fin P → ℝ) (S : Finset (Fin T)) (hS : S ⊆ pre t) {M : ℝ}
    (hv : ∀ u, |v u i θ| ≤ M) (hSpos : 0 < ∑ u ∈ S, Real.exp (score dp q k i.1 t u θ)) :
    |attnOut dp q k v t i θ
        - (∑ u ∈ S, Real.exp (score dp q k i.1 t u θ) * v u i θ)
          / (∑ u ∈ S, Real.exp (score dp q k i.1 t u θ))|
      ≤ 2 * M * ((∑ u ∈ pre t, Real.exp (score dp q k i.1 t u θ))
          - ∑ u ∈ S, Real.exp (score dp q k i.1 t u θ))
        / (∑ u ∈ pre t, Real.exp (score dp q k i.1 t u θ)) := by
  classical
  have hone : ∀ (A : Finset (Fin T)), A ⊆ pre t →
      ∑ u ∈ A, maskedScore dp q k t i.1 θ u = ∑ u ∈ A, Real.exp (score dp q k i.1 t u θ) := by
    intro A hA
    simpa using sum_maskedScore_subset dp q k t i.1 θ (fun _ => 1) hA
  have hUniv : ∑ u, maskedScore dp q k t i.1 θ u
      = ∑ u ∈ pre t, Real.exp (score dp q k i.1 t u θ) := by
    simpa using sum_maskedScore_univ dp q k t i.1 θ (fun _ => 1)
  have hSsum := hone S hS
  have key := AttentionSpinor.bounded_support_gap (maskedScore dp q k t i.1 θ)
    (fun u => v u i θ) S (maskedScore_nonneg dp q k t i.1 θ) hv (by rw [hSsum]; exact hSpos)
  rw [att_maskedScore, sum_maskedScore_subset dp q k t i.1 θ (fun u => v u i θ) hS, hSsum,
    hUniv] at key
  exact key

/-- **The budgeted read's error carries no context length (headline).** Restrict the attention to
    any retained support `S` (the heavy keys — a FIXED budget) and renormalize on `S`: the result
    differs from the full read by at most `2M·((total mass − retained mass)/total mass)`, where
    `M` bounds the values. The bound is the DROPPED MASS FRACTION and nothing else — no `T`, no
    prefix length. Zero dropped mass is exact (`AttentionSpinor.sparse_exact_on_support`), and a
    dropped mass under the emission margin leaves the emitted token unchanged
    (`SparseSelection.sparse_emits_same_token`). This is the licence for serving at flat VRAM:
    keep a budget of keys hot, the rest cold, and the softmax over the budget is the softmax. -/
theorem budget_attention_gap_is_context_free {n : ℕ} (sc v : Fin n → ℝ) (S : Finset (Fin n))
    {M : ℝ} (hsc : ∀ s, 0 ≤ sc s) (hv : ∀ s, |v s| ≤ M) (hS : 0 < ∑ s ∈ S, sc s) :
    |AttentionSpinor.att sc v - (∑ s ∈ S, sc s * v s) / (∑ s ∈ S, sc s)|
      ≤ 2 * M * ((∑ s, sc s) - ∑ s ∈ S, sc s) / (∑ s, sc s) :=
  AttentionSpinor.bounded_support_gap sc v S hsc hv hS

-- ═══════════════════════════════════════════════════════════════════
-- §8. THE HEADLINE BUNDLE
-- ═══════════════════════════════════════════════════════════════════

/-- **The transformer, from the null cell, end to end (headline).** For any shape and any depth:

    1. **the nonlinearity is the cell** — softmax is `B/U` on the null-cell boost's `+`-lightcone
       channel (`softmax_is_boost_channel`), and the SwiGLU gate is that same ratio inside one
       cell (`sigmoid_is_cell_ratio`);
    2. **the objective is cross-entropy** — `netLoss` is the mean next-token cross-entropy of the
       model's own logits (`netLoss_is_mean_cross_entropy`);
    3. **the whole pipeline is differentiable in θ** — embedding · RMSNorm · RoPE · causal MHA ·
       SwiGLU · residuals · final norm · unembedding · log-sum-exp, at every θ
       (`netLoss_differentiable`), so the gradient exists everywhere and backprop is its chain
       rule;
    4. **the step descends** — with a nonzero gradient, every sufficiently small positive learning
       rate strictly lowers the loss (`sgd_step_decreases`), a decreasing rate always exists
       (`exists_decreasing_step`), and a run of such steps never goes up
       (`trainRun_nonincreasing`);
    5. **its four attention matrices are two circuits** — the score is one bilinear form against
       `W_Qᵀ·R(u−t)·W_K` and the residual write one contraction against `W_O·W_V`
       (`attention_is_two_circuits`), so the factorization is a compute choice and the function is
       the circuits;
    6. **and it serves without the `T²`** — the read is one bulk ratio computed exactly by an
       O(1)-state stream (`attnOut_streams_in_O1_state`), and a budgeted support deviates by at
       most the dropped-mass fraction, with no context length in the bound
       (`attnOut_budget_gap`).

    A modern transformer, its training objective, its gradient, its update and its serving read,
    standing on `{0, 1, /}` and one two-dimensional cell. -/
theorem transformer_from_the_null_cell {ε : ℝ} (hε : 0 < ε) [NeZero V]
    (net : Net P H dp dff V) (toks tgt : Fin T → Fin V) (s : ℝ) {ι : Type} [Fintype ι]
    (z : ι → ℝ) (y : ι) :
    (softmax z y = lPlus (nExp (z y)) / ∑ j, lPlus (nExp (z j)))
      ∧ (sigmoid s = lPlus (nExp (s / 2)) / (lPlus (nExp (s / 2)) + lMinus (nExp (s / 2))))
      ∧ Differentiable ℝ (netLoss ε net toks tgt)
      ∧ (∀ θ : Fin P → ℝ, grad (netLoss ε net toks tgt) θ ≠ 0 →
          ∃ η : ℝ, 0 < η ∧
            netLoss ε net toks tgt (sgdStep (netLoss ε net toks tgt) η θ)
              < netLoss ε net toks tgt θ)
      ∧ (∀ (dp' : ℕ) (q k v : Fin T → Act P (Dim H dp')) (t : Fin T) (i : Dim H dp')
          (θ : Fin P → ℝ),
          AttentionSpinor.oaNum (AttentionSpinor.oaScan (attnStream dp' q k v t i θ))
              / AttentionSpinor.oaDen (AttentionSpinor.oaScan (attnStream dp' q k v t i θ))
            = attnOut dp' q k v t i θ) :=
  ⟨softmax_is_boost_channel z y,
   sigmoid_is_cell_ratio s,
   netLoss_differentiable hε net toks tgt,
   fun θ hgz => exists_decreasing_step (netLoss_differentiable hε net toks tgt) θ hgz,
   fun dp' q k v t i θ => attnOut_streams_in_O1_state dp' q k v t i θ⟩


end NullTransformer
end Sanguine

end
