import Perf.Chain
import Perf.Instances

/-!
# The two-phase overlap model: shared fields add, disjoint fields hide

`Perf.timeLB` bounds one kernel; `Perf.Concurrency` amortizes one serial chain over `conc`
independent chains. Neither carries two *phases* of a schedule whose second is issued inside the
first. P9 (`docs/evidence/perf-predictions.md`) issued the MMAs of block `kb+1` before the epilogues
of block `kb` and measured the node no faster (node ratio `0.947`); the model the formalisation audit
records as missing is a pair of phases whose second is admitted into the first's interval exactly
when its resource field is idle.

This module adds the two-phase carrier. `Res.add` sums the per-field demands and takes the **maximum**
of the two chain depths: two overlapped phases contend for every field they share, so their demands
add, but their dependent chains run beside each other, so the longest chain of the pair is the longer
of the two, not their sum. `overlapLB m A B = timeLB m (A.add B)` is the bound of the overlapped
schedule; `seqLB m A B = timeLB m A + timeLB m B` is the bound of the sequential one.

* **Overlap never loses** (`overlap_le_seq`): the overlapped bound is at most the sequential one.
* **Overlap cannot beat the larger phase** (`overlap_ge_max`): it is at least `max` of the two.
* **A shared binding field gains nothing** (`overlap_no_gain_of_shared_binding`): when one field binds
  both phases and each phase's demand on it is a whole number of capacity steps, the overlapped bound
  equals the sequential one — the two phases queue for the same resource.
* **Disjoint fields overlap fully** (`overlap_full_of_disjoint`): when the two phases' non-zero fields
  are disjoint, the overlapped bound is exactly `max` of the two.

## The P8/P9 instance

P8 (outcome, arms E1/E2/E3) decomposes the `gemm_9728_896_Q8_cm` node into an MMA term on `fMma`, an
epilogue term on `fFma`/`fAlu`, a staging term, and a residual `385.0 us` above their sum. P9
overlapped the MMA phase of block `kb+1` with the epilogue of block `kb`; the MMA and epilogue fields
are disjoint, so `overlap_full_of_disjoint` predicts the model bound `max 680.7 469.8 = 680.7 us`
against the sequential `1150.5 us` — a gain the measurement did not show (`node ratio 0.947`).
`residual_not_in_fields` states the measured node as the fully overlapped two-phase term plus the
sequential excess, the staging term and the `385.0 us` residual: the residual is a separate `Nat`
term, not any field's demand, so field overlap cannot recover it.

## What is proved here and what is not

* **Proved** under the closed axiom cone `{propext, Classical.choice, Quot.sound}` (to be confirmed
  by the lead's `#audit` lines): `overlap_le_seq`, `overlap_ge_max`,
  `overlap_no_gain_of_shared_binding`, `overlap_full_of_disjoint`, `ceilDiv_add_of_dvd`,
  `fieldMax_add` and the P8 instance.
* **Not proved.** No hardware speed follows. `Machine.cap` and `Machine.latency` are declared
  parameters; the P8 numbers (`680.6`, `469.8`, `325.1`, `385.0` us and the arm means) are `model`
  inputs read from `docs/evidence/perf-predictions.md` P8/P9, not measurements. Nothing here models
  the register pressure, the per-fragment MMA-to-epilogue dependency or the issue contention the P9
  outcome names as candidate causes of the residual; those are outside the field carrier. rustc/LLVM
  codegen, the GPU driver and the real pipeline's issue scheduling are outside every statement.
-/

namespace Perf

/-! ## 1 · The two-phase carrier -/

/-- The field-wise sum of two phases' demands, with the chain depth the **maximum** of the two. Two
overlapped phases contend for every field they share, so their demands add; their dependent chains
run beside each other, so the pair's longest chain is the longer of the two, not their sum. -/
def Res.add (A B : Res) : Res where
  demand := fun f => A.demand f + B.demand f
  depth := max A.depth B.depth

/-- The bound of a schedule in which `B` is admitted into `A`'s interval: the phases contend for
every shared field, so the overlapped bound is the roofline of the summed demand. -/
def overlapLB (m : Machine) (A B : Res) : Nat := timeLB m (A.add B)

/-- The bound of the two phases run one after the other: their bounds add. -/
def seqLB (m : Machine) (A B : Res) : Nat := timeLB m A + timeLB m B

/-- The two phases carry disjoint non-zero fields: no field is loaded by both. -/
def FieldsDisjoint (A B : Res) : Prop := ∀ f, A.demand f = 0 ∨ B.demand f = 0

/-! ## 2 · Field arithmetic -/

/-- `maxVec` is subadditive: the maximum of a pointwise sum is at most the sum of the maxima. -/
theorem maxVec_add_le (k : Nat) (u v : Fin k → Nat) :
    maxVec k (fun i => u i + v i) ≤ maxVec k u + maxVec k v :=
  maxVec_le (fun i => Nat.add_le_add (le_maxVec k u i) (le_maxVec k v i))

/-- The field maximum is subadditive in the demand: `fieldMax (A.add B) ≤ fieldMax A + fieldMax B`.
The demands on a shared field add, and `ceilDiv` is subadditive. -/
theorem fieldMax_add (m : Machine) (A B : Res) :
    fieldMax m (A.add B) ≤ fieldMax m A + fieldMax m B := by
  have h1 : fieldMax m (A.add B)
      ≤ maxVec 9 (fun i => ceilDiv (A.demand i) (m.cap i) + ceilDiv (B.demand i) (m.cap i)) := by
    unfold fieldMax fieldLB
    exact maxVec_mono 9 _ _ (fun i => by
      simp only [Res.add]
      exact Chain.ceilDiv_add_le (m.capPos i))
  have h2 : maxVec 9 (fun i => ceilDiv (A.demand i) (m.cap i) + ceilDiv (B.demand i) (m.cap i))
      ≤ fieldMax m A + fieldMax m B := by
    unfold fieldMax
    exact maxVec_add_le 9 _ _
  exact le_trans h1 h2

/-- `ceilDiv` is additive on a whole number of capacity steps: if the capacity divides both demands,
their sum needs exactly the sum of the steps, with no rounding. -/
theorem ceilDiv_add_of_dvd {dA dB c : Nat} (hc : 0 < c) (hA : c ∣ dA) (hB : c ∣ dB) :
    ceilDiv (dA + dB) c = ceilDiv dA c + ceilDiv dB c := by
  obtain ⟨a, rfl⟩ := hA
  obtain ⟨b, rfl⟩ := hB
  rw [show c * a + c * b = c * (a + b) by ring]
  rw [Chain.ceilDiv_mul_self hc, Chain.ceilDiv_mul_self hc, Chain.ceilDiv_mul_self hc]

/-! ## 3 · The four overlap theorems -/

/-- **Overlap never loses.** Admitting `B` into `A`'s interval loads every shared field with the sum
of the two demands and keeps the longer chain, so the overlapped bound is at most the sequential one;
`ceilDiv` subadditivity supplies the field arithmetic and the depth term is dominated termwise. -/
theorem overlap_le_seq (m : Machine) (A B : Res) : overlapLB m A B ≤ seqLB m A B := by
  have hf : fieldMax m (A.add B) ≤ fieldMax m A + fieldMax m B := fieldMax_add m A B
  have hd : max A.depth B.depth * m.latency ≤ A.depth * m.latency + B.depth * m.latency := by
    rcases Nat.le_total A.depth B.depth with h | h
    · rw [Nat.max_eq_right h]; omega
    · rw [Nat.max_eq_left h]; omega
  unfold overlapLB seqLB timeLB
  simp only [Res.add]
  apply Nat.max_le.mpr
  constructor
  · calc fieldMax m (A.add B) ≤ fieldMax m A + fieldMax m B := hf
      _ ≤ max (fieldMax m A) (A.depth * m.latency) + max (fieldMax m B) (B.depth * m.latency) :=
          Nat.add_le_add (Nat.le_max_left _ _) (Nat.le_max_left _ _)
  · calc max A.depth B.depth * m.latency ≤ A.depth * m.latency + B.depth * m.latency := hd
      _ ≤ max (fieldMax m A) (A.depth * m.latency) + max (fieldMax m B) (B.depth * m.latency) :=
          Nat.add_le_add (Nat.le_max_right _ _) (Nat.le_max_right _ _)

/-- Dominance: `A` is below `A.add B` on every field and on the chain depth. -/
theorem res_le_add_left (A B : Res) : A ≤ A.add B :=
  ⟨fun f => by simp only [Res.add]; omega,
   by simp only [Res.add]; exact Nat.le_max_left A.depth B.depth⟩

/-- Dominance: `B` is below `A.add B` on every field and on the chain depth. -/
theorem res_le_add_right (A B : Res) : B ≤ A.add B :=
  ⟨fun f => by simp only [Res.add]; omega,
   by simp only [Res.add]; exact Nat.le_max_right A.depth B.depth⟩

/-- **Overlap cannot beat the larger phase.** Overlapped phases still fund both phase demands, so the
overlapped bound is at least the sequential maximum; monotonicity of `timeLB` under `add` gives it. -/
theorem overlap_ge_max (m : Machine) (A B : Res) :
    max (timeLB m A) (timeLB m B) ≤ overlapLB m A B := by
  unfold overlapLB
  exact max_le (timeLB_mono m (res_le_add_left A B)) (timeLB_mono m (res_le_add_right A B))

/-- **A shared binding field gains nothing.** When one field `f` binds both phases
(`fieldLB m A f = fieldMax m A`, and the same for `B`) and each phase's demand on `f` is a whole
number of capacity steps, the overlapped bound equals the sequential one: the two phases queue for
the same resource, and the only arithmetic freedom — the rounding of a partial step — is absent. -/
theorem overlap_no_gain_of_shared_binding (m : Machine) (A B : Res) (f : ResField)
    (hbindA : fieldLB m A f = fieldMax m A) (hbindB : fieldLB m B f = fieldMax m B)
    (hdepthA : A.depth * m.latency ≤ fieldLB m A f)
    (hdepthB : B.depth * m.latency ≤ fieldLB m B f)
    (hdivA : m.cap f ∣ A.demand f) (hdivB : m.cap f ∣ B.demand f) :
    overlapLB m A B = seqLB m A B := by
  have hadd : fieldLB m (A.add B) f = fieldLB m A f + fieldLB m B f := by
    simp only [fieldLB, Res.add]
    exact ceilDiv_add_of_dvd (m.capPos f) hdivA hdivB
  have hsum : fieldMax m (A.add B) = fieldLB m A f + fieldLB m B f := by
    apply le_antisymm
    · unfold fieldMax
      refine maxVec_le (fun i => ?_)
      calc fieldLB m (A.add B) i ≤ fieldLB m A i + fieldLB m B i := by
            simp only [fieldLB, Res.add]; exact Chain.ceilDiv_add_le (m.capPos i)
        _ ≤ fieldMax m A + fieldMax m B :=
            Nat.add_le_add (le_maxVec 9 _ i) (le_maxVec 9 _ i)
        _ = fieldLB m A f + fieldLB m B f := by rw [← hbindA, ← hbindB]
    · rw [← hadd]
      unfold fieldMax
      exact le_maxVec 9 (fun g => fieldLB m (A.add B) g) f
  have hdepth : max A.depth B.depth * m.latency ≤ fieldLB m A f + fieldLB m B f := by
    rcases Nat.le_total A.depth B.depth with h | h
    · rw [Nat.max_eq_right h]; omega
    · rw [Nat.max_eq_left h]; omega
  unfold overlapLB seqLB timeLB
  rw [hsum, ← hbindA, ← hbindB]
  simp only [Res.add]
  rw [max_eq_left hdepth, max_eq_left hdepthA, max_eq_left hdepthB]

/-- **Disjoint fields overlap fully.** When no field carries demand in both phases, every field is
loaded by at most one phase, so the overlapped field maximum is the maximum of the two phase maxima,
and the pair's chain depth is the maximum of the two depth terms; the overlapped bound is therefore
exactly `max` of the two phase bounds. The depth needs no hypothesis: `add` takes the maximum. -/
theorem overlap_full_of_disjoint (m : Machine) (A B : Res) (h : FieldsDisjoint A B) :
    overlapLB m A B = max (timeLB m A) (timeLB m B) := by
  have hfmax : fieldMax m (A.add B) = max (fieldMax m A) (fieldMax m B) := by
    apply le_antisymm
    · unfold fieldMax
      refine maxVec_le (fun i => ?_)
      rcases h i with h0 | h0
      · calc fieldLB m (A.add B) i = fieldLB m B i := by
              unfold fieldLB
              rw [show (A.add B).demand i = B.demand i by simp [Res.add, h0]]
          _ ≤ fieldMax m B := le_maxVec 9 _ i
          _ ≤ max (fieldMax m A) (fieldMax m B) := Nat.le_max_right _ _
      · calc fieldLB m (A.add B) i = fieldLB m A i := by
              unfold fieldLB
              rw [show (A.add B).demand i = A.demand i by simp [Res.add, h0]]
          _ ≤ fieldMax m A := le_maxVec 9 _ i
          _ ≤ max (fieldMax m A) (fieldMax m B) := Nat.le_max_left _ _
    · exact Nat.max_le.mpr
        ⟨fieldMax_mono m (fun g => by simp only [Res.add]; omega),
         fieldMax_mono m (fun g => by simp only [Res.add]; omega)⟩
  unfold overlapLB timeLB
  rw [hfmax]
  simp only [Res.add]
  apply le_antisymm
  · apply Nat.max_le.mpr
    constructor
    · exact Nat.max_le.mpr
        ⟨Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_left _ _),
         Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_right _ _)⟩
    · rcases Nat.le_total A.depth B.depth with hle | hle
      · rw [Nat.max_eq_right hle]
        exact Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_right _ _)
      · rw [Nat.max_eq_left hle]
        exact Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_left _ _)
  · apply Nat.max_le.mpr
    constructor
    · apply Nat.max_le.mpr
      exact ⟨Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_left _ _),
        Nat.le_trans (Nat.mul_le_mul_right _ (Nat.le_max_left _ _)) (Nat.le_max_right _ _)⟩
    · apply Nat.max_le.mpr
      exact ⟨Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_left _ _),
        Nat.le_trans (Nat.mul_le_mul_right _ (Nat.le_max_right _ _)) (Nat.le_max_right _ _)⟩

/-! ## 4 · The P8/P9 instance

The numbers are `model` inputs read from `docs/evidence/perf-predictions.md` P8 outcome (the arm
means, in units of `0.1 us`) and P9 outcome; they are not measurements here. -/

/-- P8 arm E1 (no epilogue) node mean, `1005.8 us` in units of `0.1 us`. `model`. -/
def p8E1 : Nat := 10058

/-- P8 arm E2 (no MMA) node mean, `794.9 us` in units of `0.1 us`. `model`. -/
def p8E2 : Nat := 7949

/-- P8 arm E3 (staging only) node mean, `325.1 us` in units of `0.1 us`. `model`. -/
def p8E3 : Nat := 3251

/-- P8 default node mean, `1860.6 us` over six interleaved rounds. `model`. -/
def p8Default : Nat := 18606

/-- The MMA term `E1 - E3`, `680.7 us` (the row rounds it to `680.6`). `model`. -/
def p8MmaTerm : Nat := p8E1 - p8E3

/-- The epilogue term `E2 - E3`, `469.8 us`. `model`. -/
def p8EpiTerm : Nat := p8E2 - p8E3

/-- The staging term `E3`, `325.1 us`. `model`. -/
def p8Staging : Nat := p8E3

/-- The residual: the default node above the sum of the three terms, `385.0 us`. `model`. -/
def p8Residual : Nat := p8Default - (p8E1 + p8E2 - p8E3)

/-- The P8 MMA phase on `fMma`, depth zero. `model`. -/
def p8Mma : Res := singleField fMma p8MmaTerm

/-- The P8 epilogue: `fFma` carries the `469.8 us` term (two FMAs per element) and `fAlu` the magic
add, at the epilogue's two-FMA/one-add ratio. `model`. -/
def p8Epi : Res := twoField fFma fAlu p8EpiTerm 2349

/-- **The P8 phases carry disjoint fields.** The MMA phase loads only `fMma`; the epilogue loads only
`fFma` and `fAlu`. The instance therefore falls under `overlap_full_of_disjoint`, and the model
predicts the overlapped bound `max 680.7 469.8 = 680.7 us`. -/
theorem p8_disjoint : FieldsDisjoint p8Mma p8Epi := by
  intro f
  fin_cases f <;> decide

/-- **The P8 model predicts a full overlap.** `overlap_full_of_disjoint` on the P8 fields gives the
overlapped bound `680.7 us`; the sequential bound is `1150.5 us`, so the model predicts a `469.8 us`
gain. -/
theorem p8_overlap_full : overlapLB unitMachine p8Mma p8Epi = 6807 := by
  rw [overlap_full_of_disjoint unitMachine p8Mma p8Epi p8_disjoint]
  simp only [p8Mma, p8Epi, p8MmaTerm, p8EpiTerm, p8E1, p8E3]
  rw [timeLB_unit_single fMma 6807,
      timeLB_unit_two (f1 := fFma) (f2 := fAlu) (by decide)]
  decide

/-- The sequential P8 bound: `680.7 + 469.8 = 1150.5 us`. -/
theorem p8_seq : seqLB unitMachine p8Mma p8Epi = 11505 := by
  simp only [seqLB, p8Mma, p8Epi, p8MmaTerm, p8EpiTerm, p8E1, p8E3]
  rw [timeLB_unit_single fMma 6807,
      timeLB_unit_two (f1 := fFma) (f2 := fAlu) (by decide)]
  decide

/-- **P9 measured no gain.** The model's full-overlap prediction (`680.7 us`) is below the sequential
bound (`1150.5 us`), yet the measured node ratio is `0.947` (P9 outcome): the gain the model predicts
was not present. -/
theorem p8_overlap_predicted_gain :
    overlapLB unitMachine p8Mma p8Epi < seqLB unitMachine p8Mma p8Epi := by
  rw [p8_overlap_full, p8_seq]
  decide

/-- **The residual is not a field demand.** The measured default node is the fully overlapped
two-phase term `max 680.7 469.8` plus the sequential excess `min 680.7 469.8`, the staging term and
the residual: `680.7 + 469.8 + 325.1 + 385.0 = 1860.6 us`. The residual is a separate `Nat` term — no
field's demand — so field overlap can remove at most the `469.8 us` sequential excess and cannot
touch the `385.0 us`. The identity is arithmetic over the arm means; it attributes the residual to no field and does not explain it. -/
theorem residual_not_in_fields :
    p8Default = max p8MmaTerm p8EpiTerm + min p8MmaTerm p8EpiTerm + p8Staging + p8Residual ∧
    0 < p8Residual := by
  refine ⟨?_, ?_⟩
  · simp only [p8Default, p8MmaTerm, p8EpiTerm, p8Staging, p8Residual, p8E1, p8E2, p8E3]
    decide
  · simp only [p8Residual, p8Default, p8E1, p8E2, p8E3]
    decide

-- Non-vacuity: the abstract theorems instantiated at the P8 records.
#guard overlapLB unitMachine p8Mma p8Epi = 6807
#guard seqLB unitMachine p8Mma p8Epi = 11505
#guard overlapLB unitMachine p8Mma p8Epi ≤ seqLB unitMachine p8Mma p8Epi
#guard max (timeLB unitMachine p8Mma) (timeLB unitMachine p8Epi)
    ≤ overlapLB unitMachine p8Mma p8Epi

example : overlapLB unitMachine p8Mma p8Epi ≤ seqLB unitMachine p8Mma p8Epi :=
  overlap_le_seq unitMachine p8Mma p8Epi

example : max (timeLB unitMachine p8Mma) (timeLB unitMachine p8Epi)
    ≤ overlapLB unitMachine p8Mma p8Epi :=
  overlap_ge_max unitMachine p8Mma p8Epi

end Perf
