import Perf.Chain

/-!
# The shared issue-port cut, the slot move gain and the lever ceiling

`Perf.Roofline` bounds a kernel by its demand on each resource separately:
`timeLB m r = max (fieldMax m r) (r.depth * m.latency)`. That is the family of **singleton cuts** —
one cut per field, load `r.demand f`, capacity `m.cap f`. P8 refuted that family on
`gemm_9728_896_Q8_cm` (the default is 1.85x the max of the two leading terms); the missing cut is the
**shared issue port**, one port fed by several pipes, whose load is the *sum* of their slot counts,
not a max.

P13 counted the cooperative-matrix K-block loop at 1104 issue slots (32 `IMMA`, 640 float, 432
other) with an issue floor of 0.538 of the node; P14 removed 73 slots (6.6% of 1104) and the node
moved 6.8% (x1.073, 3 runs, noise floor 2.6%). The unit whose capacity matches both is issue slots on
one shared port, not independent pipes.

`slots` is the load of the shared cut over a list of pipes; `issueLB` divides it by the port width
`p`; `timeLBShared` takes the max of the field roofline and the shared cut.

## What is proved here and what is not

* **Proved** under the closed axiom cone `{propext, Classical.choice, Quot.sound}`:
  `shared_ge_timeLB` (the shared bound is at least the field bound), `shared_strict` (two distinct
  pipes whose slot sum exceeds `p * timeLB` make the shared bound strictly larger), `slot_move_gain`
  (a move that removes `s` slots and adds none lowers the issue term by at most `ceilDiv s p`), and
  `lever_ceiling` (relieving a subset of cuts cannot push the total below the max of the remaining
  cuts).
* **Not proved.** No hardware speed follows. `p` and the slot counts are declared `model` inputs;
  the P13/P14 numbers below are `model` inputs, not measurements, and no theorem here produces a
  cycle or a microsecond. The bit-identical epilogue's identity floor and the unattributed 385-us
  residual of the P8 decomposition are open.
  rustc/LLVM codegen, the GPU/NPU drivers and the real issue-port topology are outside every
  statement.
-/

namespace Perf

/-! ## 1 · The shared issue-port cut -/

/-- The load of the shared issue-port cut: the sum of the per-pipe slot counts over a list of pipes.
`m` is the per-field slot count (here `r.demand`). The temporal cost is linear in `fs.length`; each
entry is one addition. -/
def slots (m : ResField → Nat) (fs : List ResField) : Nat := (fs.map m).sum

/-- **The issue cut.** A kernel is a flow through ports; its time is at least the load of any cut
over its capacity. This is the shared issue-port cut: `slots r.demand fs` slots over a port width
`p`. -/
def issueLB (p : Nat) (r : Res) (fs : List ResField) : Nat := ceilDiv (slots r.demand fs) p

/-- **The shared issue bound.** The larger of the field roofline `timeLB m r` (the singleton cuts)
and the shared issue-port cut `issueLB p r fs`. -/
def timeLBShared (m : Machine) (p : Nat) (fs : List ResField) (r : Res) : Nat :=
  max (timeLB m r) (issueLB p r fs)

/-! ## 2 · The ceiling arithmetic of the shared cut -/

/-- Every entry of a slot list is at most the total. -/
theorem le_slots_of_mem {v : ResField → Nat} {fs : List ResField} {a : ResField}
    (h : a ∈ fs) : v a ≤ slots v fs := by
  induction fs with
  | nil => exact absurd h (by simp)
  | cons c t ih =>
      simp only [slots, List.map_cons, List.sum_cons]
      rw [List.mem_cons] at h
      rcases h with rfl | h
      · exact Nat.le_add_right _ _
      · have ih' : v a ≤ (t.map v).sum := ih h
        exact Nat.le_trans ih' (Nat.le_add_left _ _)

/-- The sum over a list is at least the sum of two distinct entries. -/
theorem two_le_slots {v : ResField → Nat} {fs : List ResField} {a b : ResField}
    (ha : a ∈ fs) (hb : b ∈ fs) (hab : a ≠ b) : v a + v b ≤ slots v fs := by
  induction fs with
  | nil => exact absurd ha (by simp)
  | cons c t ih =>
      simp only [slots, List.map_cons, List.sum_cons]
      rw [List.mem_cons] at ha hb
      rcases ha with rfl | ha
      · rcases hb with rfl | hb
        · exact absurd rfl hab
        · have hb' : v b ≤ (t.map v).sum := le_slots_of_mem hb
          omega
      · rcases hb with rfl | hb
        · have ha' : v a ≤ (t.map v).sum := le_slots_of_mem ha
          omega
        · have ih' : v a + v b ≤ (t.map v).sum := ih ha hb
          omega

/-- A maximum over a vector is at most `B` when every entry is. The companion to `le_maxVec`; the
kernel-computable `maxVec` is not exported with one. -/
theorem maxVec_le_of_forall : ∀ (k : Nat) (v : Fin k → Nat) (B : Nat),
    (∀ i, v i ≤ B) → maxVec k v ≤ B
  | 0, _, B, _ => Nat.zero_le B
  | k + 1, v, B, h => by
      show max (v 0) (maxVec k fun i => v i.succ) ≤ B
      exact Nat.max_le.mpr
        ⟨h 0, maxVec_le_of_forall k (fun i => v i.succ) B (fun i => h i.succ)⟩

/-! ## 3 · The shared bound dominates, and where it is strict -/

/-- **The shared bound is at least the field bound.** -/
theorem shared_ge_timeLB (m : Machine) (p : Nat) (fs : List ResField) (r : Res) :
    timeLB m r ≤ timeLBShared m p fs r :=
  Nat.le_max_left _ _

/-- **The shared cut is strict.** If two distinct pipes `a`, `b` of the port carry slot counts whose
sum exceeds `p * timeLB m r`, the shared bound is strictly above the field bound. This hypothesis is
what makes the shared port bind; it is not derivable from the field bound alone. -/
theorem shared_strict (m : Machine) (p : Nat) (fs : List ResField) (r : Res)
    (hp : 0 < p) {a b : ResField} (ha : a ∈ fs) (hb : b ∈ fs) (hab : a ≠ b)
    (h : p * timeLB m r < r.demand a + r.demand b) :
    timeLB m r < timeLBShared m p fs r := by
  have hsum : r.demand a + r.demand b ≤ slots r.demand fs := two_le_slots ha hb hab
  have hceil : ceilDiv (r.demand a + r.demand b) p ≤ ceilDiv (slots r.demand fs) p :=
    ceilDiv_mono hsum
  have hbig : timeLB m r < ceilDiv (r.demand a + r.demand b) p := by
    rw [Nat.mul_comm] at h
    have hk : timeLB m r + 1 ≤ ceilDiv (r.demand a + r.demand b) p :=
      Chain.le_ceilDiv_of_lt_mul hp (by simpa using h)
    omega
  have h1 : timeLB m r < issueLB p r fs := lt_of_lt_of_le hbig hceil
  show timeLB m r < max (timeLB m r) (issueLB p r fs)
  exact lt_of_lt_of_le h1 (Nat.le_max_right _ _)

/-! ## 4 · The slot move gain -/

/-- **The slot move gain.** A move that lowers every pipe's slot count (`r' ≤ r` on the pipes) with
total slot difference `s` lowers the issue term by at most `ceilDiv s p`: dropping `s` slots from a
port of width `p` saves at most `ceilDiv s p` steps. The monotonicity premise is kept for the stated
direction of the move but is unused by the arithmetic, which needs only the total difference `hs`;
it is bound as `_hmono`. -/
theorem slot_move_gain (p s : Nat) (r r' : Res) (fs : List ResField) (hp : 0 < p)
    (_hmono : ∀ f ∈ fs, r'.demand f ≤ r.demand f)
    (hs : slots r.demand fs = slots r'.demand fs + s) :
    issueLB p r fs ≤ issueLB p r' fs + ceilDiv s p := by
  unfold issueLB
  rw [hs]
  exact Chain.ceilDiv_add_le hp

/-- **The slot move gain, as a difference.** With `slots r' + s = slots r` the issue term falls by at
most `ceilDiv s p`. -/
theorem slot_move_gain_sub (p s : Nat) (r r' : Res) (fs : List ResField) (hp : 0 < p)
    (hmono : ∀ f ∈ fs, r'.demand f ≤ r.demand f)
    (hs : slots r'.demand fs + s = slots r.demand fs) :
    issueLB p r fs - issueLB p r' fs ≤ ceilDiv s p := by
  rw [Nat.sub_le_iff_le_add']
  exact slot_move_gain p s r r' fs hp hmono hs.symm

/-! ## 5 · The P13/P14 instance

The slot counts are `model` inputs: P13 counted the cooperative-matrix K-block loop at 32 `IMMA`
(`fMma`), 640 float (`fFma`) and 432 other (`fAlu`) issue slots. `p = 1` slot per unit. The field
bound alone is `max 640 432 = 640`; the shared port is `1104`. P14 removed 73 of the `fAlu` slots,
leaving 359 (total 1031). -/

/-- A demand record with `n1` slots of field `f1`, `n2` of the distinct field `f2`, `n3` of the
distinct field `f3`, and zero elsewhere, zero depth. -/
def pipeRes (f1 f2 f3 : ResField) (n1 n2 n3 : Nat) : Res where
  demand := fun i => if i = f1 then n1 else if i = f2 then n2 else if i = f3 then n3 else 0
  depth := 0

@[simp] theorem pipeRes_f1 (f1 f2 f3 : ResField) (n1 n2 n3 : Nat) :
    (pipeRes f1 f2 f3 n1 n2 n3).demand f1 = n1 := by
  show (if f1 = f1 then n1 else if f1 = f2 then n2 else if f1 = f3 then n3 else 0) = n1
  rw [if_pos rfl]

theorem pipeRes_f2 {f1 f2 f3 : ResField} (h21 : f2 ≠ f1) (n1 n2 n3 : Nat) :
    (pipeRes f1 f2 f3 n1 n2 n3).demand f2 = n2 := by
  show (if f2 = f1 then n1 else if f2 = f2 then n2 else if f2 = f3 then n3 else 0) = n2
  rw [if_neg h21, if_pos rfl]

theorem pipeRes_f3 {f1 f2 f3 : ResField} (h31 : f3 ≠ f1) (h32 : f3 ≠ f2) (n1 n2 n3 : Nat) :
    (pipeRes f1 f2 f3 n1 n2 n3).demand f3 = n3 := by
  show (if f3 = f1 then n1 else if f3 = f2 then n2 else if f3 = f3 then n3 else 0) = n3
  rw [if_neg h31, if_neg h32, if_pos rfl]

theorem pipeRes_other {f1 f2 f3 i : ResField} (h1 : i ≠ f1) (h2 : i ≠ f2) (h3 : i ≠ f3)
    (n1 n2 n3 : Nat) : (pipeRes f1 f2 f3 n1 n2 n3).demand i = 0 := by
  show (if i = f1 then n1 else if i = f2 then n2 else if i = f3 then n3 else 0) = 0
  rw [if_neg h1, if_neg h2, if_neg h3]

@[simp] theorem pipeRes_depth (f1 f2 f3 : ResField) (n1 n2 n3 : Nat) :
    (pipeRes f1 f2 f3 n1 n2 n3).depth = 0 := rfl

/-- `ceilDiv n 1 = n`. -/
@[simp] theorem ceilDiv_one_slot (n : Nat) : ceilDiv n 1 = n := by
  unfold ceilDiv; rw [Nat.add_sub_cancel, Nat.div_one]

/-- On the unit machine the lower bound of a three-pipe record is the maximum of the three slot
counts. -/
theorem timeLB_unit_pipe {f1 f2 f3 : ResField} (h12 : f1 ≠ f2) (h13 : f1 ≠ f3) (h23 : f2 ≠ f3)
    (n1 n2 n3 : Nat) :
    timeLB unitMachine (pipeRes f1 f2 f3 n1 n2 n3) = max n1 (max n2 n3) := by
  have hle : fieldMax unitMachine (pipeRes f1 f2 f3 n1 n2 n3) ≤ max n1 (max n2 n3) := by
    unfold fieldMax
    refine maxVec_le_of_forall 9
      (fun f => fieldLB unitMachine (pipeRes f1 f2 f3 n1 n2 n3) f) (max n1 (max n2 n3))
      (fun i => ?_)
    rw [fieldLB, show unitMachine.cap i = 1 from rfl, ceilDiv_one_slot]
    by_cases h1 : i = f1
    · subst h1; rw [pipeRes_f1]; exact Nat.le_max_left _ _
    · by_cases h2 : i = f2
      · rw [h2]
        rw [pipeRes_f2 (show f2 ≠ f1 from h12.symm)]
        exact Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_right _ _)
      · by_cases h3 : i = f3
        · rw [h3]
          rw [pipeRes_f3 (show f3 ≠ f1 from h13.symm) (show f3 ≠ f2 from h23.symm)]
          exact Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_right _ _)
        · rw [pipeRes_other h1 h2 h3]; exact Nat.zero_le _
  have hge : max n1 (max n2 n3) ≤ fieldMax unitMachine (pipeRes f1 f2 f3 n1 n2 n3) := by
    refine Nat.max_le.mpr ⟨?_, Nat.max_le.mpr ⟨?_, ?_⟩⟩
    · have h := le_maxVec 9 (fun i => fieldLB unitMachine (pipeRes f1 f2 f3 n1 n2 n3) i) f1
      simpa [fieldMax, fieldLB, show unitMachine.cap f1 = 1 from rfl, ceilDiv_one_slot, pipeRes_f1]
        using h
    · have h := le_maxVec 9 (fun i => fieldLB unitMachine (pipeRes f1 f2 f3 n1 n2 n3) i) f2
      simpa [fieldMax, fieldLB, show unitMachine.cap f2 = 1 from rfl, ceilDiv_one_slot,
        pipeRes_f2 (show f2 ≠ f1 from h12.symm)] using h
    · have h := le_maxVec 9 (fun i => fieldLB unitMachine (pipeRes f1 f2 f3 n1 n2 n3) i) f3
      simpa [fieldMax, fieldLB, show unitMachine.cap f3 = 1 from rfl, ceilDiv_one_slot,
        pipeRes_f3 (show f3 ≠ f1 from h13.symm) (show f3 ≠ f2 from h23.symm)] using h
  have hmax : timeLB unitMachine (pipeRes f1 f2 f3 n1 n2 n3) =
      max (fieldMax unitMachine (pipeRes f1 f2 f3 n1 n2 n3)) 0 := by
    unfold timeLB; rw [pipeRes_depth, Nat.zero_mul]
  rw [hmax]
  exact le_antisymm (Nat.max_le.mpr ⟨hle, Nat.zero_le _⟩)
    (Nat.le_trans hge (Nat.le_max_left _ _))

/-- The pipes sharing the issue port: `IMMA`, float and other. -/
def issuePipes : List ResField := [fFma, fMma, fAlu]

/-- P13: the cooperative-matrix K-block loop, 32 `IMMA`, 640 float, 432 other slots. -/
def loopRes : Res := pipeRes fFma fMma fAlu 640 32 432

/-- P14: the same loop with 73 `other` slots removed (432 -> 359). -/
def loopResP14 : Res := pipeRes fFma fMma fAlu 640 32 359

theorem loopRes_fma : loopRes.demand fFma = 640 := by
  unfold loopRes; rw [pipeRes_f1]

theorem loopRes_mma : loopRes.demand fMma = 32 := by
  unfold loopRes; rw [pipeRes_f2 (by decide)]

theorem loopRes_alu : loopRes.demand fAlu = 432 := by
  unfold loopRes; rw [pipeRes_f3 (by decide) (by decide)]

theorem loopResP14_fma : loopResP14.demand fFma = 640 := by
  unfold loopResP14; rw [pipeRes_f1]

theorem loopResP14_mma : loopResP14.demand fMma = 32 := by
  unfold loopResP14; rw [pipeRes_f2 (by decide)]

theorem loopResP14_alu : loopResP14.demand fAlu = 359 := by
  unfold loopResP14; rw [pipeRes_f3 (by decide) (by decide)]

/-- P13: 640 + 32 + 432 = 1104 issue slots on the shared port. -/
theorem loopRes_slots : slots loopRes.demand issuePipes = 1104 := by
  unfold slots issuePipes
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  rw [loopRes_fma, loopRes_mma, loopRes_alu]
  norm_num

/-- P14: 640 + 32 + 359 = 1031 issue slots on the shared port. -/
theorem loopResP14_slots : slots loopResP14.demand issuePipes = 1031 := by
  unfold slots issuePipes
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  rw [loopResP14_fma, loopResP14_mma, loopResP14_alu]
  norm_num

/-- The field bound alone is 640, the max of the three slot counts. -/
theorem loopRes_timeLB : timeLB unitMachine loopRes = 640 := by
  unfold loopRes
  rw [timeLB_unit_pipe (by decide) (by decide) (by decide)]
  norm_num

/-- The shared bound is 1104, strictly above the field bound. -/
theorem loopRes_sharedLB : timeLBShared unitMachine 1 issuePipes loopRes = 1104 := by
  unfold timeLBShared issueLB
  rw [loopRes_timeLB, loopRes_slots]
  rw [show ceilDiv 1104 1 = 1104 by unfold ceilDiv; norm_num]
  exact Nat.max_eq_right (by norm_num)

/-- The shared bound exceeds the field bound by 464 slots: `1104 - 640`. -/
theorem loopRes_shared_gap :
    timeLBShared unitMachine 1 issuePipes loopRes - timeLB unitMachine loopRes = 464 := by
  rw [loopRes_sharedLB, loopRes_timeLB]

/-- **`shared_strict` applies to P13.** The two pipes `fFma` (640) and `fMma` (32) have slot sum
672 > `1 * 640`, so the shared bound is strictly above the field bound. -/
theorem loopRes_issue_strict :
    timeLB unitMachine loopRes < timeLBShared unitMachine 1 issuePipes loopRes := by
  have h : (1 : Nat) * timeLB unitMachine loopRes < loopRes.demand fFma + loopRes.demand fMma := by
    rw [loopRes_timeLB, loopRes_fma, loopRes_mma]
    norm_num
  exact shared_strict (m := unitMachine) (p := 1) (fs := issuePipes) (r := loopRes)
    (hp := by norm_num) (a := fFma) (b := fMma)
    (ha := by simp [issuePipes]) (hb := by simp [issuePipes]) (hab := by decide) h

/-- The P14 issue term is 1031 of the P13 1104: the ratio lies in `[93/100, 94/100)`, so the issue
term drops by strictly more than 6% and at most 7%. -/
theorem p14_issue_ratio : 93 * 1104 ≤ 100 * 1031 ∧ 100 * 1031 < 94 * 1104 := by
  norm_num

/-- The P14 1031-of-1104 ratio is below 94/100 (`1031/1104 < 940/1000`). -/
theorem p14_below_94_pct : 1031 * 1000 < 1104 * 940 := by
  norm_num

/-- P14 removes exactly 73 slots (432 -> 359). -/
theorem p14_drop_slots : (1104 : Nat) - 1031 = 73 := by norm_num

-- Non-vacuity of the shared cut and the P13/P14 instance.
#guard slots loopRes.demand issuePipes = 1104
#guard slots loopResP14.demand issuePipes = 1031
#guard timeLB unitMachine loopRes = 640
#guard timeLBShared unitMachine 1 issuePipes loopRes = 1104

example : timeLB unitMachine loopRes ≤ timeLBShared unitMachine 1 issuePipes loopRes :=
  shared_ge_timeLB unitMachine 1 issuePipes loopRes

/-! ## 6 · The lever ceiling -/

/-- The maximum of a finite list of cut bounds (`0` for the empty list). -/
def maxOf : List Nat → Nat
  | [] => 0
  | a :: t => max a (maxOf t)

/-- The maximum of a list is at least its entries. -/
theorem maxOf_le_mem {l : List Nat} {a : Nat} (h : a ∈ l) : a ≤ maxOf l := by
  induction l with
  | nil => exact absurd h (by simp)
  | cons c t ih =>
      rw [List.mem_cons] at h
      rcases h with rfl | h
      · exact Nat.le_max_left _ _
      · exact Nat.le_trans (ih h) (Nat.le_max_right _ _)

/-- The maximum of a list is at most `B` when every entry is. -/
theorem maxOf_le_of_forall {l : List Nat} {B : Nat} (h : ∀ a ∈ l, a ≤ B) : maxOf l ≤ B := by
  induction l with
  | nil => exact Nat.zero_le B
  | cons c t ih =>
      show max c (maxOf t) ≤ B
      exact Nat.max_le.mpr ⟨h c (by simp), ih (fun a ha => h a (by simp [ha]))⟩

/-- **The lever ceiling.** A move relieves a subset `S` of the cut family `cuts`; `keep` is the
family of unrelieved cuts (`cuts` without `S`) and `cuts'` the new family, which agrees with `cuts`
off `S` — so every unrelieved cut bound is still present in `cuts'`. Relieving a subset therefore
cannot lower the total below `maxOf keep`, the maximum of the remaining cuts: the ceiling of a move
is the max of the cuts it does not relieve. -/
theorem lever_ceiling (keep cuts' : List Nat) (h : ∀ a ∈ keep, a ∈ cuts') :
    maxOf keep ≤ maxOf cuts' :=
  maxOf_le_of_forall (fun a ha => maxOf_le_mem (h a ha))

-- Non-vacuity: a family that keeps its three unrelieved cuts cannot fall below their max.
example : maxOf [(1 : Nat), 2, 5] ≤ maxOf [1, 2, 5, 3] :=
  lever_ceiling [1, 2, 5] [1, 2, 5, 3] (by decide)

/-- `layer_cfg` (`src/gpu/decode/upload.rs`) reads two `F64` carriers (`rms_eps`, `rope_theta`) per layer
at upload, so a load of `nLayer` layers performs `2 * nLayer` conversions and a decode of any token
count performs none: the conversions are paid once at load, outside the per-token cost.
Cost model only; no hardware timing is claimed. -/
def uploadConvs (nLayer : Nat) : Nat := 2 * nLayer

theorem upload_convs_token_free (nLayer tokens : Nat) :
    uploadConvs nLayer + 0 * tokens = uploadConvs nLayer := by
  simp

-- Non-vacuity: a 24-layer upload performs 48 conversions.
example : uploadConvs 24 = 48 := by decide

end Perf
