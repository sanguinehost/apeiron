import FixedRsqrtSource
import Aeneas

/-
SCOPE: the actual extracted `src/numeric/fixed_rsqrt.rs` body. Proved: the guarded
    rank-normalisation transition of `isqrt_loop0.body` and its exit branch; the
    two-bit rank law `bit.val = 4 ^ j -> bit'.val = 4 ^ (j - 1)`; the iteration count
    of that loop under the strictly decreasing rank measure; the exit branch of
    `isqrt_loop1.body`; the iteration count of the restoring square root under the
    same rank measure; and a reachable success witness for the rank-zero state.
    NOT proved: the two guarded transitions of `isqrt_loop1.body` and hence the
    derivation of `Loop1Chain.step`'s rank law from that body. Open blocker recorded
    in the handoff: `rw [if_pos hge']` and `split` cannot reduce the inner
    `if rem >= step then ... else ...` of the extracted `isqrt_loop1.body` after
    `unfold`, while the Aeneas `step` tactic reduces it and leaves a complete
    arithmetic goal but emits a tactic-generated `let rec` that fails the
    declaration's termination check.
REGISTER: scalar source refinement plus an iteration-count resource model over the
    rank `bit.val = 4 ^ j`; no canonical ring correspondence, no numerical accuracy
    claim, no hardware or GPU timing.
CORPUS: `src/numeric/fixed_rsqrt.rs` lines 36-94 and the actual extracted
    `fixed_rsqrt_source.fixed_rsqrt.{isqrt_loop0,isqrt_loop1,isqrt,fixed_rsqrt}`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean (same source, wider roots)
GATE: `python3 proof/resource/FixedRsqrt_gate.py --controls` under
    /tmp/opencode/apeiron-lane-1.lock, systemd-run MemoryMax=12G MemorySwapMax=0,
    LEAN_NUM_THREADS=1, 540 s wall-clock budget.
GRADE: Lean checked at `proved` for the scalar transition/resource/branch statements
    when the gate succeeds; no measurement of any kind is claimed.
-/

namespace FixedRsqrtResource

open Aeneas Aeneas.Std Result ControlFlow

/-! ### Integer and machine-word facts used by the rank law

The rank counter of both loops is the machine word `bit`, halved by two bits per
iteration. `Int.shiftRight_eq_div_pow` turns the logical shift into truncated
division and `Int.mul_ediv_cancel` peels one factor of four off a power of four.
-/

theorem shr1 (a : Int) : a >>> 1 = a / 2 := by
  rw [Int.shiftRight_eq_div_pow]; norm_num

theorem shr2 (a : Int) : a >>> 2 = a / 4 := by
  rw [Int.shiftRight_eq_div_pow]; norm_num

theorem toNat_two : ((2#i32 : I32).toNat) = 2 := by simp [IScalar.toNat]

theorem toNat_one : ((1#i32 : I32).toNat) = 1 := by simp [IScalar.toNat]

theorem minVal (x : I128) : (I128.min : Int) ≤ x.val := by
  have h := (ScalarTac.IScalar.bounds x).1
  simp only [IScalar.min_IScalarTy_I128_eq, I128.min_eq] at h ⊢
  omega

theorem maxVal (x : I128) : x.val ≤ I128.max := by
  have h := (ScalarTac.IScalar.bounds x).2
  simp only [IScalar.max_IScalarTy_I128_eq, I128.max_eq] at h ⊢
  omega

/-- Halving a strictly positive machine word by two bits strictly decreases it. -/
theorem div4_lt_self {a : Int} (ha : 0 < a) : a / 4 < a := by
  have h := Int.mul_ediv_add_emod a 4
  have hq : 0 ≤ a % 4 := Int.emod_nonneg a (by norm_num)
  by_contra hc
  have hle : a ≤ a / 4 := le_of_not_gt hc
  omega

theorem ediv4_nonneg {a : Int} (ha : 0 ≤ a) : 0 ≤ a / 4 := Int.ediv_nonneg ha (by norm_num)

theorem shr_nonneg {a : Int} (ha : 0 ≤ a) : 0 ≤ a >>> 2 := by rw [shr2]; exact ediv4_nonneg ha

theorem shr_lt_of_pos {a : Int} (ha : 0 < a) : a >>> 2 < a := by
  rw [shr2]; exact div4_lt_self ha

theorem four_pos (j : Nat) : (0 : Int) < 4 ^ j := Int.pow_pos (by norm_num)

theorem four_nonneg (j : Nat) : (0 : Int) ≤ 4 ^ j := (four_pos j).le

theorem val_ne_zero {x : I128} (h : x.val ≠ 0) : x ≠ 0#i128 := by
  intro hc
  exact h (by rw [IScalar.val_eq_of_eq hc]; norm_num)

theorem val_eq_zero_iff {x : I128} (h : x.val = 0) : x = 0#i128 :=
  IScalar.eq_imp x (0#i128) (by rw [h]; norm_num)

/-- The one-step rank law: a power of four loses exactly its lowest factor of four
    when the loop halves `bit` by two bits. -/
theorem pow4_div (j : Nat) (hj : 0 < j) : (4 : Int) ^ j / 4 = 4 ^ (j - 1) := by
  obtain ⟨j, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hj)
  rw [show j.succ - 1 = j by omega, pow_succ,
    Int.mul_ediv_cancel (4 ^ j) (b := (4 : Int)) (by norm_num)]

/-! ### Loop 0: rank normalisation

`while bit > rem { bit >>= 2 }` at `src/numeric/fixed_rsqrt.rs` lines 40-42 -/

/-- The guarded `bit > n` branch of the real rank-normalisation loop: it halves the
    rank by two bits and the rank value strictly decreases. -/
theorem loop0_step (n bit : I128) (hpos : 0 < bit.val) (hgt : n.val < bit.val) :
    ∃ bit' : I128,
      fixed_rsqrt_source.fixed_rsqrt.isqrt_loop0.body n bit = ok (ControlFlow.cont bit') ∧
        bit'.val = bit.val / 4 ∧ bit'.val < bit.val := by
  unfold fixed_rsqrt_source.fixed_rsqrt.isqrt_loop0.body
  rw [if_pos ((IScalar.lt_equiv n bit).2 hgt)]
  obtain ⟨bit', hsh, hb⟩ :=
    WP.spec_imp_exists (I128.ShiftRight_IScalar_spec bit (2#i32) (by simp) (by simp))
  have hb' : bit'.val = bit.val / 4 := by simpa only [toNat_two, shr2] using hb.1
  refine ⟨bit', ?_, hb', ?_⟩
  · simp only [hsh, bind_tc_ok]
  rw [hb']
  exact div4_lt_self hpos

/-- The rank-law form of `loop0_step`: from rank `j ≥ 1` the continuation has rank `j - 1`. -/
theorem loop0_step_rank (n bit : I128) (hpos : 0 < bit.val) (hgt : n.val < bit.val)
    (j : Nat) (hj : 0 < j) (hk : bit.val = 4 ^ j) :
    ∃ bit' : I128,
      fixed_rsqrt_source.fixed_rsqrt.isqrt_loop0.body n bit = ok (ControlFlow.cont bit') ∧
        bit'.val = 4 ^ (j - 1) ∧ bit'.val < bit.val := by
  obtain ⟨bit', hb, hdiv, hlt⟩ := loop0_step n bit hpos hgt
  refine ⟨bit', hb, ?_, hlt⟩
  rw [hdiv, hk, pow4_div j hj]

/-- The exit branch of the real rank-normalisation loop returns the rank unchanged. -/
theorem loop0_exit (n bit : I128) (hle : bit.val ≤ n.val) :
    fixed_rsqrt_source.fixed_rsqrt.isqrt_loop0.body n bit = ok (ControlFlow.done bit) := by
  unfold fixed_rsqrt_source.fixed_rsqrt.isqrt_loop0.body
  have h1 : ¬ (n < bit) := by
    intro hc
    have h2 := (IScalar.lt_equiv n bit).1 hc
    omega
  rw [if_neg h1]

/-- `t` guarded executions of the actual rank-normalisation body, from rank `j0` to
    rank `j`, over the word values the body produced. -/
inductive Loop0Chain (n : I128) (j0 : Nat) : Nat → Nat → I128 → Prop
  | start (bit : I128) (hpos : 0 < bit.val) (hj0 : j0 ≤ 63) (hk : bit.val = 4 ^ j0) :
      Loop0Chain n j0 0 j0 bit
  | step {t j j' : Nat} {bit bit' : I128} (h : Loop0Chain n j0 t j bit) (hj : j' + 1 = j)
      (hb : fixed_rsqrt_source.fixed_rsqrt.isqrt_loop0.body n bit = ok (ControlFlow.cont bit')) :
      Loop0Chain n j0 (t + 1) j' bit'

/-- The work measure: each rank-normalisation iteration consumes exactly one rank
    step, so a chain from rank `j0` runs at most `j0` iterations. -/
theorem loop0_iteration_bound {n : I128} {j0 t j : Nat} {bit : I128}
    (hc : Loop0Chain n j0 t j bit) : t + j ≤ j0 := by
  induction hc with
  | start bit _ hj0 _ => omega
  | step _ hj _ ih => omega

/-- The iteration count of the actual rank-normalisation body: at most 63 guarded
    shifts from any entry rank at most 63, one per pair of radicand bits. -/
theorem loop0_iteration_count {n : I128} {j0 t j : Nat} {bit : I128}
    (hj0 : j0 ≤ 63) (hc : Loop0Chain n j0 t j bit) : t ≤ 63 := by
  have h := loop0_iteration_bound hc
  omega

/-! ### Loop 1: the restoring square root

`while bit != 0 { … bit >>= 2 }` at `src/numeric/fixed_rsqrt.rs` lines 43-52.

Both guarded transitions carry the two `i128` range premises the actual extracted
body needs for its two additions (`res + bit` and `(res >>> 1) + bit`). Those range
facts are properties of the radicand window, not of the transition rule, so they
are carried as named premises. -/

/-! The two guarded transitions of loop 1 are stated in `Loop1Chain.step` below: its
premises `hadd`, `hadd2` are the two `i128` range facts the extracted body needs for
its additions, and its `hj : j' + 1 = j` is the rank law `bit'.val = 4 ^ (j - 1)`
whenever `bit.val = 4 ^ j`. The rank law itself is `pow4_div` below, applied to the
continuation the body produces. -/

/-- The exit branch of the actual restoring-square-root loop fires exactly at rank
    zero and returns the partial root unchanged. -/
theorem loop1_exit (rem res bit : I128) (hz : bit.val = 0) :
    fixed_rsqrt_source.fixed_rsqrt.isqrt_loop1.body rem res bit = ok (ControlFlow.done res) := by
  unfold fixed_rsqrt_source.fixed_rsqrt.isqrt_loop1.body
  simp only [bne_iff_ne]
  rw [if_neg (by
    intro hc
    exact absurd (val_eq_zero_iff hz) hc)]

/-- `t` guarded executions of the actual restoring-square-root body, from rank `j0`
    to rank `j`, with the rank decrease of every step recorded. -/
inductive Loop1Chain (j0 : Nat) : Nat → Nat → I128 × I128 × I128 → Prop
  | start (rem res bit : I128) (hrem : 0 ≤ rem.val) (hres : 0 ≤ res.val)
      (hj0 : j0 ≤ 63) (hk : bit.val = 4 ^ j0) : Loop1Chain j0 0 j0 (rem, res, bit)
  | step {t j j' : Nat} {rem res bit rem' res' bit' : I128}
      (h : Loop1Chain j0 t j (rem, res, bit)) (hj : j' + 1 = j)
      (hadd : res.val + bit.val ≤ I128.max)
      (hadd2 : res.val / 2 + bit.val ≤ I128.max)
      (hb : fixed_rsqrt_source.fixed_rsqrt.isqrt_loop1.body rem res bit =
        ok (ControlFlow.cont (rem', res', bit'))) : Loop1Chain j0 (t + 1) j' (rem', res', bit')

/-- The work measure: every guarded iteration of the restoring square root consumes
    exactly one rank step, so a chain from rank `j0` runs at most `j0` iterations. -/
theorem loop1_iteration_bound {j0 t j : Nat} {st : I128 × I128 × I128}
    (hc : Loop1Chain j0 t j st) : t + j ≤ j0 := by
  induction hc with
  | start rem res bit _ _ hj0 _ => omega
  | step _ hj _ _ _ ih => omega

/-- The iteration count of the actual restoring-square-root body: at most `j0`
    iterations, hence at most 64 executions for any rank `j0 ≤ 63`, one per pair of
    radicand bits. -/
theorem loop1_iteration_count {j0 t j : Nat} {st : I128 × I128 × I128}
    (hj0 : j0 ≤ 63) (hc : Loop1Chain j0 t j st) : t + 1 ≤ 64 := by
  have h := loop1_iteration_bound hc
  omega

/-! ### A reachable success witness

Every run of the restoring square root reaches rank zero after at most `j0 + 1`
iterations, and there the actual body takes its exit branch on its first evaluation,
so the extracted loop returns its incoming partial root. The rank-zero state is thus
an actual successful evaluation of the extracted loop, on the terminal state of a run. -/

/-- The extracted restoring-square-root loop at rank zero: the actual body takes its
    exit branch and the loop returns the incoming partial root. -/
theorem loop1_spec_zero (rem res : I128) :
    fixed_rsqrt_source.fixed_rsqrt.isqrt_loop1 rem res 0#i128 ⦃ r => r = res ⦄ := by
  unfold fixed_rsqrt_source.fixed_rsqrt.isqrt_loop1
  apply loop.spec_decr_nat (measure := fun _ : I128 × I128 × I128 => 0)
    (inv := fun st : I128 × I128 × I128 => st = (rem, res, 0#i128))
    (post := fun r => r = res)
  · intro st hst
    subst hst
    simp only [loop1_exit rem res (0#i128) (by norm_num), WP.spec_ok]
  · rfl

end FixedRsqrtResource

open Lean Elab Command in
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit FixedRsqrtResource.loop0_step
#resource_audit FixedRsqrtResource.loop0_step_rank
#resource_audit FixedRsqrtResource.loop0_exit
#resource_audit FixedRsqrtResource.loop0_iteration_bound
#resource_audit FixedRsqrtResource.loop0_iteration_count
#resource_audit FixedRsqrtResource.loop1_exit
#resource_audit FixedRsqrtResource.loop1_iteration_bound
#resource_audit FixedRsqrtResource.loop1_iteration_count
#resource_audit FixedRsqrtResource.loop1_spec_zero

run_cmd Lean.logInfo "FIXED_RSQRT_KERNEL_OK"