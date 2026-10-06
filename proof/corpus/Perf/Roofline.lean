import Mathlib.Tactic

/-!
# The roofline lower bound, the move theorem, and fusion on the resource carrier

A kernel is a denotation `f` plus a schedule. Its time is bounded below by the demand on the
resource that binds: `timeLB m r = max over fields of ceil (r.demand f / m.cap f)`, plus the
`depth * latency` term of a serial chain. A *move* is a denotation-preserving rewrite `f -> g`; its
speedup is at most the demand ratio on the binding resource.

This module restates the minimal carrier from `proof/resource/CostAlgebra.lean` (the seven-field
`Cost`, componentwise `<=`, `overlap`, `scalar_not_sufficient`) as the field-indexed `Res` here,
because that package is a separate Lean build and is not importable from this corpus. It ports the
fusion results of `Closures/Engineering/FusionCost.lean` (sanguine `e830d3fb`) and of
`Closures/Engineering/FusionHidesLatency.lean` (sanguine `e830d3fb`) onto `Res`, and carries the
machine's capacities as declared parameters.

## What is proved here and what is not

* **Proved** under the closed axiom cone `{propext, Classical.choice, Quot.sound}`:
  `timeLB_mono` (a componentwise demand increase raises the bound), `move_speedup_le` (the bound
  before a move is at most the demand ratio on the binding resource times the bound after),
  `bindingRatio_bounds` and `move_speedup_le_binding` (the demand-ratio corollary with its side
  condition), `scalar_not_sufficient` (a move that lowers the instruction demand and raises the
  cache-line demand raises the bound, the map-face control), `fusion_reduces_latency`,
  `fusion_removes` and `fanout_forces_cost` (fusion is free only on a use-count-1 edge), and the
  denotation invariance of segmentation.
* **Not proved.** No hardware speed follows. `Machine.cap` and `Machine.latency` are declared
  parameters, not measurements, and no theorem here produces a byte, a cycle or a microsecond.
  rustc/LLVM codegen, the CPU and GPU drivers, clock and power behaviour, and cache replacement are
  outside every statement. The field counts in the closing example are `model` inputs of this
  module.
-/

namespace Perf

/-! ## 1 · The resource carrier -/

/-- The field index of a counted resource. `Fin 9` so a maximum over the fields is finite. -/
abbrev ResField := Fin 9

/-- Field 0: bytes of global-memory traffic, in bytes. -/
def fGlobal : ResField := ⟨0, by decide⟩
/-- Field 1: bytes of shared-memory traffic, in bytes. -/
def fShared : ResField := ⟨1, by decide⟩
/-- Field 2: distinct 64-byte cache lines touched, in lines. -/
def fLine : ResField := ⟨2, by decide⟩
/-- Field 3: register-file bytes moved, in bytes. -/
def fReg : ResField := ⟨3, by decide⟩
/-- Field 4: fused-multiply-add instructions, in instructions. -/
def fFma : ResField := ⟨4, by decide⟩
/-- Field 5: four-way dot-product-add instructions, in instructions. -/
def fDp4a : ResField := ⟨5, by decide⟩
/-- Field 6: matrix-unit multiply-accumulate instructions, in instructions. -/
def fMma : ResField := ⟨6, by decide⟩
/-- Field 7: scalar arithmetic-logic instructions, in instructions. -/
def fAlu : ResField := ⟨7, by decide⟩
/-- Field 8: kernel launches, in launches. -/
def fLaunch : ResField := ⟨8, by decide⟩

/-- The counted demand of one kernel together with the depth of its serial chain. `demand` is the
per-field count; `depth` is the longest dependent chain of stages, kept separate because a chain's
latency does not amortize. -/
structure Res where
  /-- The per-field counted demand. -/
  demand : ResField → Nat
  /-- The serial chain depth, the longest dependent stage chain. -/
  depth : Nat

/-- A machine's capacity per field, as measured parameters, plus the per-step latency of a serial
chain. Each capacity is positive, so `ceil (demand / capacity)` is the number of steps the demand
needs on that resource. -/
structure Machine where
  /-- The capacity of each field: the resource units serviced per step. -/
  cap : ResField → Nat
  /-- Every capacity is positive, so a demand of zero is zero steps. -/
  capPos : ∀ f, 0 < cap f
  /-- The latency of one step of the serial chain, in steps. -/
  latency : Nat

/-- The componentwise dominance order: a demand record is below another when no field and no chain
depth grows. A single weighted sum would hide a regression in one field, which is the point of
`scalar_not_sufficient` below. -/
instance : LE Res where
  le r r' := (∀ f, r.demand f ≤ r'.demand f) ∧ r.depth ≤ r'.depth

/-- `ceil (n / m)`. -/
def ceilDiv (n m : Nat) : Nat := (n + m - 1) / m

/-! ## 2 · The ceiling arithmetic -/

/-- `ceilDiv` is monotone in its dividend. -/
theorem ceilDiv_mono {a b c : Nat} (h : a ≤ b) : ceilDiv a c ≤ ceilDiv b c := by
  unfold ceilDiv
  exact Nat.div_le_div_right (Nat.sub_le_sub_right (Nat.add_le_add_right h c) 1)

/-- A positive divisor's ceiling covers its dividend: `a <= ceil (a / b) * b`. -/
theorem le_ceilDiv_mul {a b : Nat} (hb : 0 < b) : a ≤ ceilDiv a b * b := by
  unfold ceilDiv
  have h := Nat.div_add_mod' (a + b - 1) b
  have hmod := Nat.mod_lt (a + b - 1) hb
  omega

/-- The ceiling splits over a product by a constant: `ceil (K * a / c) <= K * ceil (a / c)`. -/
theorem ceilDiv_mul_le {K a c : Nat} (hc : 0 < c) : ceilDiv (K * a) c ≤ K * ceilDiv a c := by
  have ha : a ≤ ceilDiv a c * c := le_ceilDiv_mul hc
  have hnum : K * a + c - 1 ≤ K * (ceilDiv a c * c) + c - 1 :=
    Nat.sub_le_sub_right (Nat.add_le_add_right (Nat.mul_le_mul_left K ha) c) 1
  calc ceilDiv (K * a) c
      = (K * a + c - 1) / c := rfl
    _ ≤ (K * (ceilDiv a c * c) + c - 1) / c := Nat.div_le_div_right hnum
    _ = K * ceilDiv a c := by
        rw [show K * (ceilDiv a c * c) = (K * ceilDiv a c) * c by ring]
        exact Nat.div_eq_of_lt_le (by omega) (by rw [Nat.add_mul, Nat.one_mul]; omega)

/-- `ceil (1 / c) = 1` for a positive `c`. -/
theorem ceilDiv_one {c : Nat} (hc : 0 < c) : ceilDiv 1 c = 1 := by
  unfold ceilDiv
  rw [show 1 + c - 1 = c by omega]
  exact Nat.div_self hc

/-- The componentwise maximum is monotone in each argument. -/
theorem max_le_max_nat {a b c d : Nat} (h1 : a ≤ b) (h2 : c ≤ d) : max a c ≤ max b d :=
  Nat.max_le_of_le_of_le (Nat.le_trans h1 (Nat.le_max_left b d))
    (Nat.le_trans h2 (Nat.le_max_right b d))

/-! ## 3 · The maximum over fields

`maxVec k v` is the maximum of a `k`-vector of `Nat`, by structural recursion on `k`. It is
kernel-computable, so the concrete examples below close by `decide`. -/

/-- The maximum of a `Fin k -> Nat` demand vector. -/
def maxVec : (k : Nat) → (Fin k → Nat) → Nat
  | 0, _ => 0
  | k + 1, v => max (v 0) (maxVec k fun i => v i.succ)

/-- Every entry of a vector is below its maximum. -/
theorem le_maxVec : ∀ (k : Nat) (v : Fin k → Nat) (i : Fin k), v i ≤ maxVec k v
  | 0, v, i => i.elim0
  | k + 1, v, i => by
      refine Fin.cases ?_ ?_ i
      · show v 0 ≤ max (v 0) (maxVec k fun i => v i.succ)
        exact Nat.le_max_left _ _
      · intro j
        show v j.succ ≤ max (v 0) (maxVec k fun i => v i.succ)
        exact Nat.le_trans (le_maxVec k (fun i => v i.succ) j) (Nat.le_max_right _ _)

/-- The maximum is monotone in every entry. -/
theorem maxVec_mono :
    ∀ (k : Nat) (u v : Fin k → Nat), (∀ i, u i ≤ v i) → maxVec k u ≤ maxVec k v
  | 0, u, v, h => Nat.zero_le 0
  | k + 1, u, v, h => by
      show max (u 0) (maxVec k fun i => u i.succ) ≤ max (v 0) (maxVec k fun i => v i.succ)
      exact max_le_max_nat (h 0)
        (maxVec_mono k (fun i => u i.succ) (fun i => v i.succ) (fun i => h i.succ))

/-- Scaling every entry by a constant scales the maximum by that constant. -/
theorem maxVec_scale : ∀ (k : Nat) (K : Nat) (v : Fin k → Nat),
    maxVec k (fun i => K * v i) = K * maxVec k v
  | 0, K, v => by simp [maxVec]
  | k + 1, K, v => by
      show max (K * v 0) (maxVec k (fun i => K * v i.succ))
        = K * max (v 0) (maxVec k fun i => v i.succ)
      rw [maxVec_scale k K (fun i => v i.succ)]
      exact Nat.mul_max_mul_left K (v 0) (maxVec k fun i => v i.succ)

/-! ## 4 · The roofline lower bound -/

/-- The lower bound contributed by one field: `ceil (demand / capacity)`. -/
def fieldLB (m : Machine) (r : Res) (f : ResField) : Nat := ceilDiv (r.demand f) (m.cap f)

/-- The maximum field lower bound, the resource term of the bound. -/
def fieldMax (m : Machine) (r : Res) : Nat := maxVec 9 (fun f => fieldLB m r f)

/-- **The roofline lower bound.** A kernel's time is at least the demand on each resource over that
resource's capacity, maximized over the fields, plus the serial chain depth times the per-step
latency, which does not amortize. -/
def timeLB (m : Machine) (r : Res) : Nat := max (fieldMax m r) (r.depth * m.latency)

/-- The field maximum is monotone in every field demand. -/
theorem fieldMax_mono (m : Machine) {r r' : Res} (h : ∀ f, r.demand f ≤ r'.demand f) :
    fieldMax m r ≤ fieldMax m r' := by
  show maxVec 9 (fun f => ceilDiv (r.demand f) (m.cap f))
    ≤ maxVec 9 (fun f => ceilDiv (r'.demand f) (m.cap f))
  exact maxVec_mono 9 _ _ (fun f => ceilDiv_mono (h f))

/-- **Monotonicity.** Componentwise dominance of resource demand raises the bound, which is the
`CostAlgebra.costLe` sufficient case read on `Res`. -/
theorem timeLB_mono (m : Machine) {r r' : Res} (h : r ≤ r') :
    timeLB m r ≤ timeLB m r' := by
  change (∀ f, r.demand f ≤ r'.demand f) ∧ r.depth ≤ r'.depth at h
  obtain ⟨hd, hdep⟩ := h
  show max (fieldMax m r) (r.depth * m.latency) ≤ max (fieldMax m r') (r'.depth * m.latency)
  exact max_le_max_nat (fieldMax_mono m hd) (Nat.mul_le_mul_right m.latency hdep)

/-- Every field's lower bound is at most the whole bound, so a move may be compared on the field
that binds. -/
theorem fieldLB_le_timeLB (m : Machine) (r : Res) (f : ResField) :
    ceilDiv (r.demand f) (m.cap f) ≤ timeLB m r :=
  Nat.le_trans (le_maxVec 9 (fun g => ceilDiv (r.demand g) (m.cap g)) f)
    (Nat.le_max_left _ _)

/-! ## 5 · The move theorem -/

/-- A demand-ratio bound of `K`: the baseline demand of every field and of the chain depth is at
most `K` times the after demand. This is the load-bearing hypothesis of `move_speedup_le`. -/
def DemandRatioBounded (f g : Res) (K : Nat) : Prop :=
  (∀ i, f.demand i ≤ K * g.demand i) ∧ f.depth ≤ K * g.depth

/-- The demand ratio of a move: the maximum over the fields of `ceil (f.demand / g.demand)`, with
the chain-depth ratio folded in. -/
def bindingRatio (f g : Res) : Nat :=
  max (maxVec 9 (fun i => ceilDiv (f.demand i) (g.demand i))) (ceilDiv f.depth g.depth)

/-- **The move theorem.** If the after schedule's demand is at most `1/K` of the baseline's on every
field and on the chain depth, then the baseline's lower bound is at most `K` times the after
schedule's. The speedup of a denotation-preserving rewrite is bounded by the demand ratio on the
binding resource. -/
theorem move_speedup_le (m : Machine) {f g : Res} {K : Nat}
    (h : DemandRatioBounded f g K) : timeLB m f ≤ K * timeLB m g := by
  obtain ⟨hfield, hdepth⟩ := h
  have hpoint : ∀ i, ceilDiv (f.demand i) (m.cap i) ≤ K * ceilDiv (g.demand i) (m.cap i) :=
    fun i => Nat.le_trans (ceilDiv_mono (hfield i)) (ceilDiv_mul_le (m.capPos i))
  have hA : fieldMax m f ≤ K * fieldMax m g := by
    show maxVec 9 (fun i => ceilDiv (f.demand i) (m.cap i))
      ≤ K * maxVec 9 (fun i => ceilDiv (g.demand i) (m.cap i))
    calc maxVec 9 (fun i => ceilDiv (f.demand i) (m.cap i))
        ≤ maxVec 9 (fun i => K * ceilDiv (g.demand i) (m.cap i)) := maxVec_mono 9 _ _ hpoint
      _ = K * maxVec 9 (fun i => ceilDiv (g.demand i) (m.cap i)) := maxVec_scale 9 K _
  have hB : f.depth * m.latency ≤ K * (g.depth * m.latency) :=
    Nat.le_trans (Nat.mul_le_mul_right m.latency hdepth)
      (le_of_eq (Nat.mul_assoc K g.depth m.latency))
  exact Nat.max_le_of_le_of_le
    (Nat.le_trans hA (Nat.mul_le_mul_left K (Nat.le_max_left _ _)))
    (Nat.le_trans hB (Nat.mul_le_mul_left K (Nat.le_max_right _ _)))

/-- **The side condition on the demand ratio.** The move introduces no demand on a field or a chain
depth where the baseline had none, so no ratio is infinite. -/
def NoNewField (f g : Res) : Prop :=
  (∀ i, 0 < f.demand i → 0 < g.demand i) ∧ (0 < f.depth → 0 < g.depth)

/-- `bindingRatio` is a demand-ratio bound under `NoNewField`. -/
theorem bindingRatio_bounds {f g : Res} (h : NoNewField f g) :
    DemandRatioBounded f g (bindingRatio f g) := by
  obtain ⟨hfield, hdepth⟩ := h
  refine ⟨?_, ?_⟩
  · intro i
    by_cases hg : g.demand i = 0
    · have hf : f.demand i = 0 := by
        by_contra hf0
        have hpos : 0 < g.demand i := hfield i (Nat.pos_of_ne_zero hf0)
        rw [hg] at hpos
        exact Nat.lt_irrefl 0 hpos
      simp [hf, hg]
    · have hgpos : 0 < g.demand i := Nat.pos_of_ne_zero hg
      have hle : ceilDiv (f.demand i) (g.demand i) ≤ bindingRatio f g :=
        Nat.le_trans (le_maxVec 9 (fun j => ceilDiv (f.demand j) (g.demand j)) i)
          (Nat.le_max_left _ _)
      exact Nat.le_trans (le_ceilDiv_mul hgpos) (Nat.mul_le_mul_right (g.demand i) hle)
  · by_cases hg : g.depth = 0
    · have hf : f.depth = 0 := by
        by_contra hf0
        have hpos : 0 < g.depth := hdepth (Nat.pos_of_ne_zero hf0)
        rw [hg] at hpos
        exact Nat.lt_irrefl 0 hpos
      simp [hf, hg]
    · have hgpos : 0 < g.depth := Nat.pos_of_ne_zero hg
      have hle : ceilDiv f.depth g.depth ≤ bindingRatio f g := Nat.le_max_right _ _
      exact Nat.le_trans (le_ceilDiv_mul hgpos) (Nat.mul_le_mul_right g.depth hle)

/-- **The move theorem at the demand ratio.** The bound before a move is at most `bindingRatio`
times the bound after, when the move introduces no demand on a field the baseline left nonzero. -/
theorem move_speedup_le_binding (m : Machine) {f g : Res} (h : NoNewField f g) :
    timeLB m f ≤ bindingRatio f g * timeLB m g :=
  move_speedup_le m (bindingRatio_bounds h)

/-! ## 6 · Denotation invariance of segmentation -/

section Denotation

variable {A : Type}

/-- The composition of a chain of maps, right to left. -/
def compose (fs : List (A → A)) : A → A := fs.foldr (· ∘ ·) id

@[simp] theorem compose_nil : compose ([] : List (A → A)) = id := rfl

@[simp] theorem compose_cons (f : A → A) (fs : List (A → A)) :
    compose (f :: fs) = f ∘ compose fs := rfl

/-- Composition is the list-concatenation monoid homomorphism. -/
theorem compose_append (a b : List (A → A)) :
    compose (a ++ b) = compose a ∘ compose b := by
  induction a with
  | nil => simp [compose]
  | cons f a ih =>
    show compose (f :: (a ++ b)) = compose (f :: a) ∘ compose b
    rw [compose_cons, compose_cons, ih]
    exact (Function.comp_assoc f (compose a) (compose b)).symm

/-- A segmentation: run each inner list as one fused kernel, then compose the kernels. -/
def runSegs (segs : List (List (A → A))) : A → A := compose (segs.map compose)

/-- **The denotation is invariant to segmentation.** Cutting a chain into kernels never changes the
computed function. Port of `FusionCost.seg_denotation_invariant`. -/
theorem seg_denotation_invariant (segs : List (List (A → A))) :
    runSegs segs = compose segs.flatten := by
  induction segs with
  | nil => rfl
  | cons s ss ih =>
    show compose (compose s :: ss.map compose) = compose (s ++ ss.flatten)
    rw [compose_cons, compose_append,
      show compose (ss.map compose) = compose ss.flatten from ih]

/-- **A fusion preserves the denotation.** Two segmentations of one chain compute the same function.
Port of `FusionCost.fusion_preserves_denotation`. -/
theorem fusion_preserves_denotation (fine coarse : List (List (A → A)))
    (hsame : fine.flatten = coarse.flatten) :
    runSegs fine = runSegs coarse := by
  rw [seg_denotation_invariant, seg_denotation_invariant, hsame]

end Denotation

/-! ## 7 · The move structure -/

/-- A denotation-preserving rewrite: a baseline demand record and an after demand record, together
with the equality of their denotations that makes the rewrite legal. The cost theorem below reads
only the demand pair; the denotation equality is the legality witness. -/
structure Move (D : Type) where
  /-- The baseline demand record. -/
  before : Res
  /-- The after demand record. -/
  after : Res
  /-- The baseline denotation. -/
  beforeDenote : D
  /-- The after denotation. -/
  afterDenote : D
  /-- The denotations agree, so the rewrite preserves the meaning. -/
  preserves : beforeDenote = afterDenote

/-- The move theorem on a `Move`: the baseline bound is at most the demand ratio times the after
bound. The denotation equality is carried but not used: the bound is a statement about demand. -/
theorem Move.speedup_le {D : Type} (m : Machine) (K : Nat) (mv : Move D)
    (h : DemandRatioBounded mv.before mv.after K) :
    timeLB m mv.before ≤ K * timeLB m mv.after :=
  move_speedup_le m h

/-! ## 8 · Fusion on a use-count-1 edge -/

/-- A chain cut into `n` kernels: `n` launches, `(n - 1) * inter` global bytes for the materialised
intermediates, and serial depth `n`. -/
def segCost (inter n : Nat) : Res where
  demand := fun i => match i with
    | ⟨0, _⟩ => (n - 1) * inter
    | ⟨8, _⟩ => n
    | _ => 0
  depth := n

/-- Fewer kernels means no larger demand on any field of a segmentation. -/
theorem segCost_mono (inter n n' : Nat) (h : n ≤ n') : segCost inter n ≤ segCost inter n' := by
  change (∀ i, (segCost inter n).demand i ≤ (segCost inter n').demand i) ∧
    (segCost inter n).depth ≤ (segCost inter n').depth
  refine ⟨?_, ?_⟩
  · intro i
    fin_cases i <;>
      dsimp only [segCost] <;>
      first
        | exact Nat.zero_le _
        | exact Nat.mul_le_mul_right inter (Nat.sub_le_sub_right h 1)
        | exact h
  · dsimp only [segCost]
    exact h

/-- **Fusion reduces the launch latency.** Two segmentations of one chain (`coarse <= fine` kernels)
compute the same function (`fusion_preserves_denotation`); the coarser one has no larger time lower
bound, because it pays fewer launches and materialises fewer intermediates. Port of
`FusionCost.fusion_reduces_latency` (sanguine `e830d3fb`). -/
theorem fusion_reduces_latency (m : Machine) (inter : Nat) {coarse fine : Nat}
    (hfewer : coarse ≤ fine) :
    timeLB m (segCost inter coarse) ≤ timeLB m (segCost inter fine) :=
  timeLB_mono m (segCost_mono inter coarse fine hfewer)

/-- Fusing one interior edge of a chain removes exactly one launch and the intermediate's `inter`
global bytes. -/
theorem fusion_removes (inter n : Nat) :
    (segCost inter (n + 2)).demand fLaunch = (segCost inter (n + 1)).demand fLaunch + 1 ∧
    (segCost inter (n + 2)).demand fGlobal = (segCost inter (n + 1)).demand fGlobal + inter := by
  constructor
  · show n + 2 = (n + 1) + 1
    omega
  · show ((n + 2) - 1) * inter = ((n + 1) - 1) * inter + inter
    rw [show (n + 2) - 1 = (n + 1) - 1 + 1 by omega, Nat.add_mul, Nat.one_mul]

/-- A chain whose interior value has `k` consumers: dropping the producing launch recomputes the
producer `k` times, so the launch demand is `k`, not one. -/
def fanoutChain (k : Nat) : Res where
  demand := fun i => match i with
    | ⟨8, _⟩ => k
    | _ => 0
  depth := 0

/-- The launch demand of a fanout chain is its use count. -/
theorem fanoutChain_demand (k : Nat) : (fanoutChain k).demand fLaunch = k := rfl

/-- The launch-only lower bound of a demand record. -/
def launchLB (m : Machine) (r : Res) : Nat := ceilDiv (r.demand fLaunch) (m.cap fLaunch)

/-- **A fanout forces the cost.** With `k >= 2` consumers and a launch capacity of one, the
fused one-launch demand is strictly below the `k`-launch demand the fanout forces. The use-count-1
side condition (`2 <= k`) is load-bearing: at `k = 1` the statement is false. Port of
`FusionCost.fanout_forces_cost` (sanguine `e830d3fb`). -/
theorem fanout_forces_cost (m : Machine) {k : Nat} (hcap : m.cap fLaunch = 1) (hk : 2 ≤ k) :
    launchLB m (fanoutChain 1) < launchLB m (fanoutChain k) := by
  have h1 : launchLB m (fanoutChain 1) = 1 := by
    show ceilDiv ((fanoutChain 1).demand fLaunch) (m.cap fLaunch) = 1
    rw [fanoutChain_demand 1, hcap]
    exact ceilDiv_one (by omega)
  have h2 : launchLB m (fanoutChain k) = k := by
    show ceilDiv ((fanoutChain k).demand fLaunch) (m.cap fLaunch) = k
    rw [fanoutChain_demand k, hcap]
    unfold ceilDiv
    rw [Nat.add_sub_cancel, Nat.div_one]
  omega

/-! ## 9 · The map face: instruction count is not sufficient

The counts below are declared `model` inputs of this module, not measurements. `mapAp` executes
fewer instructions than `mapHb` and touches more cache lines; on the unit machine its time lower
bound is strictly the larger, so a model that counts instructions only predicts the wrong sign. -/

/-- A demand record from nine field counts and a depth. -/
def mkRes (global shared line reg fma dp4a mma alu launch depth : Nat) : Res where
  demand := fun i => match i with
    | ⟨0, _⟩ => global
    | ⟨1, _⟩ => shared
    | ⟨2, _⟩ => line
    | ⟨3, _⟩ => reg
    | ⟨4, _⟩ => fma
    | ⟨5, _⟩ => dp4a
    | ⟨6, _⟩ => mma
    | ⟨7, _⟩ => alu
    | _ => launch
  depth := depth

/-- The total instruction demand across the four instruction classes. -/
def totalInstr (r : Res) : Nat :=
  r.demand fFma + r.demand fDp4a + r.demand fMma + r.demand fAlu

/-- A machine at one unit of every resource per step: every demand is exactly its own bound. The
capacities are declared parameters, not measurements. -/
def unitMachine : Machine where
  cap := fun _ => 1
  capPos := fun _ => by omega
  latency := 0

/-- The map face's `ap` layout (`model`): 832 lines touched, 336 instructions. -/
def mapAp : Res := mkRes 0 0 832 0 16 0 0 320 1 1

/-- The map face's baseline layout (`model`): 203 lines touched, 640 instructions. -/
def mapHb : Res := mkRes 0 0 203 0 64 0 0 576 1 1

/-- **Instruction count is not sufficient.** `mapAp` executes fewer instructions than `mapHb`
(`336 < 640`) yet its time lower bound is strictly larger (`832 > 576`), because its cache-line
demand binds. The counts are declared model inputs. -/
theorem scalar_not_sufficient :
    totalInstr mapAp < totalInstr mapHb ∧ timeLB unitMachine mapHb < timeLB unitMachine mapAp := by
  decide

-- The bound is tight at capacity one: the 832-line demand is exactly 832 steps.
#guard timeLB unitMachine mapAp = 832

-- The ceiling is slack by one: 7 lines at 3 lines per step is 3 steps, not 7 / 3.
#guard ceilDiv 7 3 = 3

-- Concrete non-vacuity of the fusion theorems.
example : timeLB unitMachine (segCost 0 1) ≤ timeLB unitMachine (segCost 0 3) :=
  fusion_reduces_latency unitMachine 0 (by omega)

example : launchLB unitMachine (fanoutChain 1) < launchLB unitMachine (fanoutChain 2) :=
  fanout_forces_cost unitMachine rfl (by omega)

-- Concrete non-vacuity of the move theorem: `mapAp` is within ratio 5 of `mapHb`.
example : timeLB unitMachine mapAp ≤ 5 * timeLB unitMachine mapHb :=
  move_speedup_le unitMachine (K := 5)
    ⟨fun i => by fin_cases i <;> decide, by decide⟩

end Perf
