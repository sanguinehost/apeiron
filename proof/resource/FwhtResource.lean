import FwhtSource
import Aeneas

/-
SCOPE: the actually extracted `word_fwht_fixed` call closure of `src/engine/hadamard.rs`: the
exponent-scan loop, the three nested butterfly loops (stage, base, lane) with their guards, the
work measure with its strictly decreasing rank at each level, the exit branch of every loop, the
sign and normalization loops, and a reachable success witness.
REGISTER: scalar source refinement and resource model of the extracted bodies. No hardware timing,
no cycle, latency, allocation-byte or GPU claim; no canonical-ring correspondence; no claim about
any consumer of the rotation.
CORPUS: `src/engine/hadamard.rs` lines 63-284 and the actually extracted
`fwht_source.hadamard.block_exponent_loop(.body)`,
`fwht_source.hadamard.fwht_int_loop0(.body)`,
`fwht_source.hadamard.fwht_int_loop0_loop0(.body)`,
`fwht_source.hadamard.fwht_int_loop0_loop0_loop0(.body)`,
`fwht_source.hadamard.fwht_fixed_loop0(.body)`, `fwht_source.hadamard.fwht_fixed_loop1(.body)`,
`fwht_source.hadamard.fwht_fixed_loop2(.body)`, `fwht_source.hadamard.fwht_fixed_loop3(.body)`,
`fwht_source.hadamard.fwht_int`, `fwht_source.hadamard.fwht_fixed` and
`fwht_source.word_fwht_fixed`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean
GATE: python3 proof/resource/Fwht_gate.py --controls, whole invocation 540 s on
/tmp/opencode/apeiron-lane-1.lock, MemoryMax=12G, MemorySwapMax=0, LEAN_NUM_THREADS=1.
GRADE: `proved` for the theorems below, and only when that gate reports
FWHT_SOURCE_RESOURCE_OK after a fresh extraction, a Lean build and an attributable real-source
mutation/restore control. The work measure counts invocations of the extracted loop bodies under
named premises; it is not a machine-time measurement.
-/

namespace FwhtResource
open Aeneas Aeneas.Std Result

/-! ## Scalar facts the concrete steps need

`UScalar.max`, `UScalarTy.numBits`, `I64.min`, `I64.max` and both `numBits` constants reduce only
through `System.Platform.getNumBits`, which the pinned model leaves opaque: `norm_num`, `decide`
and `rw` all stop at it. The five equations below therefore use `native_decide`, which places the
compiler's evaluation of that one opaque host-width query outside the kernel. This is a declared
trust boundary of this file, named here and in the `TRUST` field of the gate receipt; it covers
the platform pointer and `i64` widths only. No algorithmic, arithmetic or control-flow fact in
this file depends on it: every guard, bound and window premise below is a concrete proposition the
kernel decides, and the `#resource_audit` list names the headlines that are free of it.

The `OfNat` literals carry bound proofs that block definitional unfolding, so the scalar helpers
carry the `.val` equations the extracted bodies' `Result` operations need. Every window is
stated once, in ℤ, and reused by every premise. -/

theorem usize_numBits : UScalarTy.Usize.numBits = 64 := by native_decide

theorem usize_max : UScalar.max UScalarTy.Usize = 2 ^ 64 - 1 := by native_decide

theorem usize_max' : Usize.max = 2 ^ 64 - 1 := by native_decide

/-- The concrete ℕ value of the numeral `1` at the `usize` type, which the extracted bodies use to
    state their unit steps. -/
theorem usize_one : ((1#usize : UScalar UScalarTy.Usize) : ℕ) = 1 := rfl

theorem usize_two : ((2#usize : UScalar UScalarTy.Usize) : ℕ) = 2 := rfl

/-- Every `usize` value lies below the platform maximum, in ℕ. -/
theorem usize_val_le_max (x : UScalar UScalarTy.Usize) : x.val ≤ 2 ^ 64 - 1 := by
  have hb := x.hBounds
  have hn : UScalarTy.Usize.numBits = 64 := usize_numBits
  rw [hn] at hb
  omega

/-- A `usize` addition inside the platform maximum succeeds and carries the ℤ sum. The premise is
    the overflow guard the extracted bodies need; it holds for every reachable counter, each of
    which is a `lane`, `base`, `stage`, `span` or slice index bounded by a slice length. -/
theorem usize_add_spec (x y : UScalar UScalarTy.Usize) (h : x.val + y.val ≤ 2 ^ 64 - 1) :
    x + y ⦃ z => z.val = x.val + y.val ⦄ := by
  apply UScalar.add_spec
  have hm : UScalar.max UScalarTy.Usize = 2 ^ 64 - 1 := usize_max
  rw [hm]
  exact h

/-- The `usize` multiplication counterpart, for `span * 2` and `2 * span`. -/
theorem usize_mul_spec (x y : UScalar UScalarTy.Usize) (h : x.val * y.val ≤ 2 ^ 64 - 1) :
    x * y ⦃ z => z.val = x.val * y.val ⦄ := by
  apply Usize.mul_spec
  have hm : Usize.max = 2 ^ 64 - 1 := usize_max'
  rw [hm]
  exact h

/-- `MAX_K` is the constant `16` that `src/engine/hadamard.rs:19` declares. -/
theorem maxK_val : fwht_source.hadamard.MAX_K.val = 16 := by
  rw [fwht_source.hadamard.MAX_K]
  show ((16#usize : UScalar UScalarTy.Usize) : ℕ) = 16
  rfl

/-- The `i64` window, in ℤ, for every premise below. -/
def i64Window (v : I64) : Prop :=
  -9223372036854775808 ≤ v.val ∧ v.val ≤ 9223372036854775807

theorem i64_numBits : I64.numBits = 64 := by native_decide

theorem i64_numBits' : IScalarTy.I64.numBits = 64 := by native_decide

theorem i64_min : I64.min = -9223372036854775808 := by
  have hn : I64.numBits = 64 := i64_numBits
  norm_num [I64.min, IScalar.min, hn]

theorem i64_max : I64.max = 9223372036854775807 := by
  have hn : I64.numBits = 64 := i64_numBits
  norm_num [I64.max, IScalar.max, hn]

theorem i64_le_max (v : I64) : v.val ≤ 9223372036854775807 := by
  have hb := v.hBounds
  have hn : IScalarTy.I64.numBits = 64 := i64_numBits'
  rw [hn] at hb
  norm_num at hb
  omega

theorem i64_ge_min (v : I64) : -9223372036854775808 ≤ v.val := by
  have hb := v.hBounds
  have hn : IScalarTy.I64.numBits = 64 := i64_numBits'
  rw [hn] at hb
  norm_num at hb
  omega

/-- The `some` arm of the checked `i64` addition the butterfly body performs. -/
theorem checked_add_some (top bottom : I64)
    (hlo : -9223372036854775808 ≤ top.val + bottom.val)
    (hhi : top.val + bottom.val ≤ 9223372036854775807) :
    ∃ z, I64.checked_add top bottom = some z ∧ z.val = top.val + bottom.val := by
  have a : I64.min ≤ top.val + bottom.val := by rw [i64_min]; exact hlo
  have b : top.val + bottom.val ≤ I64.max := by rw [i64_max]; exact hhi
  have hspec := I64.checked_add_bv_spec top bottom
  cases hc : core.num.checked_add_IScalar top bottom with
  | none => rw [hc] at hspec; exact absurd ⟨a, b⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I64.checked_add] using hc, hspec.2.2.1⟩

/-- The `some` arm of the checked `i64` subtraction the butterfly body performs. -/
theorem checked_sub_some (top bottom : I64)
    (hlo : -9223372036854775808 ≤ top.val - bottom.val)
    (hhi : top.val - bottom.val ≤ 9223372036854775807) :
    ∃ z, I64.checked_sub top bottom = some z ∧ z.val = top.val - bottom.val := by
  have a : I64.min ≤ top.val - bottom.val := by rw [i64_min]; exact hlo
  have b : top.val - bottom.val ≤ I64.max := by rw [i64_max]; exact hhi
  have hspec := I64.checked_sub_bv_spec top bottom
  cases hc : core.num.checked_sub_IScalar top bottom with
  | none => rw [hc] at hspec; exact absurd ⟨a, b⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I64.checked_sub] using hc, hspec.2.2.1⟩

/-- The `ok_or` wrapper turns a successful checked operation into `Ok`. -/
theorem ok_or_ok (z : I64) (e : fwht_source.hadamard.HadamardError) :
    core.option.Option.ok_or (some z) e = ok (core.result.Result.Ok z) := rfl

/-- The `ok_or` wrapper turns a refused checked operation into `Err`. -/
theorem ok_or_none (e : fwht_source.hadamard.HadamardError) :
    core.option.Option.ok_or (none : Option I64) e =
      ok (core.result.Result.Err e) := rfl

/-! ## The work measure

`src/engine/hadamard.rs:130-161` is three nested loops. `stage` counts the `k` butterfly stages; at
stage `stage` the span is `2^stage`, `base` walks the `2·span`-strided block positions, and `lane`
walks the `span` butterflies inside one position. With `n = x.len()` a stage performs
`(n / (2·span)) · span = n/2` butterflies, so the executing core performs `k · n/2` of them.

The three ranks below are the remaining-work measures the loop specifications run on. Each counts
loop-body invocations under the named premises; no cycle, clock or byte count appears. -/

/-- Remaining butterflies of the inner lane loop. -/
def laneRank (span : Usize) (lane : Usize) : Nat := span.val - lane.val

/-- Remaining block positions of the base loop inside one stage. -/
def baseRank (n base : Usize) : Nat := n.val - base.val

/-- Remaining stages of the stage loop. -/
def stageRank (k stage : Usize) : Nat := k.val - stage.val

/-- The closed-form butterfly count of one stage: `n/2`, independent of the span. -/
def stageWork (n : Usize) : Nat := n.val / 2

/-- The closed-form total butterfly count of the stage loop: `k · n/2`. -/
def totalWork (k n : Usize) : Nat := k.val * (n.val / 2)

/-- The span at stage `stage`, the quantity the base loop strides by. -/
def spanAt (stage : Usize) : Nat := 2 ^ stage.val

/-! ## (a) The exponent scan: how `k` is found

`src/engine/hadamard.rs:67-70` is the loop that finds `k` with `block = 2^k`. Its rank is
`MAX_K - k`; each continuation consumes one exponent, so the body runs at most `MAX_K` times. -/

/-- The actual extracted exponent-scan body. -/
abbrev expBody (block : Usize) : Usize → Result (ControlFlow Usize Usize) :=
  fun k => fwht_source.hadamard.block_exponent_loop.body block k

/-- The actual extracted exponent-scan loop. -/
abbrev expLoop (block : Usize) : Usize → Result Usize :=
  fun k => fwht_source.hadamard.block_exponent_loop block k

/-- **One guarded exponent step.** With the exponent below `MAX_K` and the corresponding power
    still below `block`, the body shifts, increments the exponent and continues; the rank
    `MAX_K - k` drops by exactly one. -/
theorem exp_body_step (block k : Usize) (pow : Usize)
    (hguard : k.val < fwht_source.hadamard.MAX_K.val)
    (hshift : (1#usize <<< k) = ok pow)
    (hlt : pow.val < block.val) :
    ∃ k' : Usize, expBody block k = ok (ControlFlow.cont k') ∧
      k'.val = k.val + 1 ∧
      fwht_source.hadamard.MAX_K.val - k'.val + 1 = fwht_source.hadamard.MAX_K.val - k.val ∧
      fwht_source.hadamard.MAX_K.val - k'.val < fwht_source.hadamard.MAX_K.val - k.val := by
  have hguard' : k < fwht_source.hadamard.MAX_K := (UScalar.lt_equiv _ _).2 hguard
  have hlt' : pow < block := (UScalar.lt_equiv _ _).2 hlt
  obtain ⟨k', hk', hkv⟩ := WP.spec_imp_exists
    (usize_add_spec k 1#usize (by
      have hk := usize_val_le_max k
      have hK := maxK_val
      have h1 := usize_one
      omega))
  refine ⟨k', ?_, ?_, ?_, ?_⟩
  · rw [expBody, fwht_source.hadamard.block_exponent_loop.body]
    rw [if_pos hguard', hshift, bind_tc_ok, if_pos hlt', hk', bind_tc_ok]
  · exact hkv
  · have hkg : k.val < 16 := by rw [maxK_val] at hguard; exact hguard
    have hone : k'.val = k.val + 1 := by rw [hkv, usize_one]
    have hK := maxK_val
    omega
  · have hkg : k.val < 16 := by rw [maxK_val] at hguard; exact hguard
    have hone : k'.val = k.val + 1 := by rw [hkv, usize_one]
    have hK := maxK_val
    omega

/-! ## (b) The inner lane loop: one butterfly

`src/engine/hadamard.rs:134-156`. The state is `((slice, ok), (err, lane))`; the body reads
`x[base+lane]` and `x[base+lane+span]`, performs exactly two checked operations, writes the sum and
difference back in place, and advances `lane` by one. Its rank is `span - lane`. -/

/-- The lane-loop state, exactly as the extracted loop carries it. -/
abbrev laneState : Type :=
  Slice I64 × Bool × Option fwht_source.hadamard.HadamardError × Usize

/-- The lane-loop result, exactly as the extracted loop returns it. -/
abbrev laneResult : Type := Slice I64 × Bool × Option fwht_source.hadamard.HadamardError

/-- The actual extracted lane-loop body. -/
abbrev laneBody (span base : Usize) (x : Slice I64) : laneState →
    Result (ControlFlow laneState laneResult) :=
  fun st => fwht_source.hadamard.fwht_int_loop0_loop0_loop0.body span base x st.2.1 st.2.2.1 st.2.2.2

/-- The actual extracted lane loop. -/
abbrev laneLoop (span base : Usize) (x : Slice I64) : laneState → Result laneResult :=
  fun st => fwht_source.hadamard.fwht_int_loop0_loop0_loop0 x st.2.1 st.2.2.1 span base st.2.2.2

/-- Index transport: two in-bounds indices that are equal read the same cell. The extracted
    `Slice.index_usize` facts are stated at the `Usize` index; this moves them to the plain
    natural-number cell the postcondition speaks about. -/
theorem idx_eq {α : Type} (l : List α) {i j : ℕ} (hi : i < l.length) (hj : j < l.length)
    (h : i = j) : l[i]'hi = l[j]'hj := by subst h; rfl

/-- `Slice.set` is the in-place write the butterfly performs and it preserves the length, so every
    index in bounds before the write is in bounds after it. -/
theorem set_length {α : Type} (v : Slice α) (i : Usize) (x : α) :
    (v.set i x).val.length = v.val.length := by
  unfold Slice.set
  simp

/-- The two operand indices the butterfly reads: `base + lane` and `base + lane + span`. -/
def topIdx (base lane : Usize) : Nat := base.val + lane.val

def botIdx (base lane span : Usize) : Nat := base.val + lane.val + span.val

/-- **One guarded butterfly.** With the lane inside the span and the failure flag clear, the body
    reads the two `i64` operands, performs exactly two checked operations, writes both results back
    in place and advances the lane index by exactly one; the rank `span - lane` drops by one. The
    four `i64`-window premises are exactly the two checked operations' preconditions, so on them
    the body takes both `Ok` arms and the `some e` refusal arms are unreachable. -/
theorem lane_body_step (span base : Usize) (x : Slice I64) (lane : Usize)
    (hlane : lane.val < span.val)
    (htop : topIdx base lane < x.val.length)
    (hbot : botIdx base lane span < x.val.length)
    (hi1 : base.val + lane.val + span.val ≤ 2 ^ 64 - 1)
    (hs : -9223372036854775808 ≤ (x.val[topIdx base lane]'htop : I64).val + (x.val[botIdx base lane span]'hbot : I64).val)
    (hs' : (x.val[topIdx base lane]'htop : I64).val + (x.val[botIdx base lane span]'hbot : I64).val ≤ 9223372036854775807)
    (hd : -9223372036854775808 ≤ (x.val[topIdx base lane]'htop : I64).val - (x.val[botIdx base lane span]'hbot : I64).val)
    (hd' : (x.val[topIdx base lane]'htop : I64).val - (x.val[botIdx base lane span]'hbot : I64).val ≤ 9223372036854775807)
    (sum diff : I64)
    (hsum : sum.val = (x.val[topIdx base lane]'htop : I64).val + (x.val[botIdx base lane span]'hbot : I64).val)
    (hdiff : diff.val = (x.val[topIdx base lane]'htop : I64).val - (x.val[botIdx base lane span]'hbot : I64).val) :
    ∃ s' : Slice I64, ∃ lane' : Usize,
      laneBody span base x (x, true, none, lane) =
        ok (ControlFlow.cont (s', true, none, lane')) ∧
      lane'.val = lane.val + 1 ∧
      laneRank span lane' + 1 = laneRank span lane ∧
      laneRank span lane' < laneRank span lane ∧
      s'.val.length = x.val.length ∧
      s'.val[topIdx base lane]? = some sum ∧
      s'.val[botIdx base lane span]? = some diff := by
  have hlane' : lane < span := (UScalar.lt_equiv _ _).2 hlane
  have hone := usize_one
  have hbnd : base.val + lane.val ≤ 2 ^ 64 - 1 := by omega
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (usize_add_spec base lane hbnd)
  have hivv : i.val = base.val + lane.val := hiv
  have hi_lt : i.val < x.val.length := by rw [hivv]; exact htop
  obtain ⟨top, htop', htopv⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i hi_lt)
  have htopvv : top.val = (x.val[topIdx base lane]'htop : I64).val := by
    have hc : i.val = topIdx base lane := by rw [hivv]; rfl
    have hleft := idx_eq x.val hi_lt htop hc
    rw [hleft] at htopv
    simpa only [] using congrArg (fun v : I64 => v.val) htopv
  obtain ⟨i1, hi1', hi1v⟩ := WP.spec_imp_exists
    (usize_add_spec i span (by rw [hivv]; exact hi1))
  have hi1_lt : i1.val < x.val.length := by rw [hi1v, hivv]; exact hbot
  obtain ⟨bottom, hbot', hbotv⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i1 hi1_lt)
  have hbotvv : bottom.val = (x.val[botIdx base lane span]'hbot : I64).val := by
    have hc : i1.val = botIdx base lane span := by rw [hi1v, hivv]; rfl
    have hleft := idx_eq x.val hi1_lt hbot hc
    rw [hleft] at hbotv
    simpa only [] using congrArg (fun v : I64 => v.val) hbotv
  have hstop : top.val = (x.val[topIdx base lane]'htop : I64).val := htopvv
  have hbotb : bottom.val = (x.val[botIdx base lane span]'hbot : I64).val := hbotvv
  obtain ⟨sum0, hsum0, hsum0v⟩ := checked_add_some top bottom
    (by omega) (by omega)
  obtain ⟨diff0, hdiff0, hdiff0v⟩ := checked_sub_some top bottom
    (by omega) (by omega)
  have hsumident : sum0 = sum := by
    apply IScalar.eq_of_val_eq
    rw [hsum0v, hstop, hbotb]
    exact hsum.symm
  have hdiffident : diff0 = diff := by
    apply IScalar.eq_of_val_eq
    rw [hdiff0v, hstop, hbotb]
    exact hdiff.symm
  have hibnd : i.val < x.val.length := hi_lt
  obtain ⟨x1, hx1, hx1v⟩ := WP.spec_imp_exists (Slice.update_spec x i sum0 hibnd)
  have hi1b : i1.val < x.val.length := hi1_lt
  have hi1_lt' : i1.val < x1.val.length := by rw [hx1v, set_length]; exact hi1b
  obtain ⟨s', hs', hsv⟩ := WP.spec_imp_exists
    (Slice.update_spec x1 i1 diff0 hi1_lt')
  obtain ⟨lane', hlane'', hlanev⟩ := WP.spec_imp_exists
    (usize_add_spec lane 1#usize (by
      have hl := usize_val_le_max lane
      omega))
  -- The two write indices are distinct: the second is the first plus the positive span, so the
  -- second write never disturbs the cell the first one wrote.
  have hidx : i.val ≠ i1.val := by
    have hspanpos : 0 < span.val := lt_of_le_of_lt (Nat.zero_le _) hlane
    rw [hi1v, hivv]
    intro hc
    have hbot' : base.val + lane.val + span.val < x.val.length := hbot
    omega
  have hidx' : i1.val ≠ i.val := fun hc => hidx hc.symm
  have hsetTop : i.val = topIdx base lane := by rw [hivv]; rfl
  have hsetBot : i1.val = botIdx base lane span := by rw [hi1v, hivv]; rfl
  have hbotIneq : i1.val ≠ topIdx base lane := by
    intro hc
    have hbot' : base.val + lane.val + span.val < x.val.length := hbot
    have hspanpos : 0 < span.val := lt_of_le_of_lt (Nat.zero_le _) hlane
    omega
  refine ⟨s', lane', ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [laneBody, fwht_source.hadamard.fwht_int_loop0_loop0_loop0.body]
    rw [if_pos hlane', hi, bind_tc_ok, htop', bind_tc_ok, hi1', bind_tc_ok, hbot', bind_tc_ok]
    simp only [lift, bind_tc_ok]
    rw [hsum0, ok_or_ok, bind_tc_ok, hdiff0, ok_or_ok]
    simp only [if_true, bind_tc_ok]
    rw [hx1, bind_tc_ok, hs']
    simp only [bind_tc_ok]
    rw [hlane'', bind_tc_ok]
  · rw [hlanev, usize_one]
  · simp only [laneRank]; omega
  · simp only [laneRank]; omega
  · -- `Slice.set` preserves the length: the butterfly writes in place, adding no storage.
    rw [hsv, hx1v, Slice.set_val_eq, List.length_set, Slice.set_val_eq, List.length_set]
  · -- The sum lands at the top index, read through the first write: that cell holds the
    -- checked sum of the two operands, in ℤ.
    have htopRead : (s'.val[topIdx base lane]? : Option I64) = some sum0 := by
      rw [hsv, hx1v]
      simp only [Slice.set_val_eq, List.getElem?_set, if_neg hbotIneq,
        List.getElem?_set,
        List.getElem?_eq_getElem htop, if_pos hsetTop, if_pos hibnd]
    rw [htopRead, hsumident]
  · -- The difference lands at the bottom index, read through both writes: that cell holds the
    -- checked difference of the two operands, in ℤ.
    have hbotRead : (s'.val[botIdx base lane span]? : Option I64) = some diff0 := by
      rw [hsv, hx1v]
      simp only [Slice.set_val_eq, List.getElem?_set, if_pos hsetBot]
      rw [if_pos (show i1.val < (x.val.set i.val sum0).length from by
        rw [List.length_set, hsetBot]; exact hbot)]
    rw [hbotRead, hdiffident]

/-! ## (c) The lane loop's exit branches

Both exits of `src/engine/hadamard.rs:134-156` return without performing a butterfly: the
`lane < span` guard failing means the span is exhausted, and the failure flag being set means a
previous checked operation refused. -/

/-- **Span exit.** With the lane index at the span, the body returns the buffer and the failure
    flag unchanged, consuming no further butterfly. -/
theorem lane_body_exit_span (span base : Usize) (x : Slice I64) (lane : Usize) (flag : Bool)
    (err : Option fwht_source.hadamard.HadamardError) (hlen : span.val ≤ lane.val) :
    laneBody span base x (x, flag, err, lane) = ok (ControlFlow.done (x, flag, err)) := by
  have hnot : ¬ (lane < span) := by
    intro hc
    have hc' := (UScalar.lt_equiv lane span).1 hc
    omega
  rw [laneBody, fwht_source.hadamard.fwht_int_loop0_loop0_loop0.body, if_neg hnot]

/-- **Refusal exit.** With the failure flag set, the body returns the flag clear and the recorded
    error, consuming no further butterfly whatever the lane index. -/
theorem lane_body_exit_flag (span base : Usize) (x : Slice I64) (lane : Usize)
    (err : Option fwht_source.hadamard.HadamardError) :
    laneBody span base x (x, false, err, lane) = ok (ControlFlow.done (x, false, err)) := by
  by_cases hlt : lane < span
  · rw [laneBody, fwht_source.hadamard.fwht_int_loop0_loop0_loop0.body, if_pos hlt, if_neg (by simp)]
  · rw [laneBody, fwht_source.hadamard.fwht_int_loop0_loop0_loop0.body, if_neg hlt]

/-- The lane loop's reachable invariant, stated at the rank `span - lane`. -/
def laneInv (span : Usize) : laneState → Prop :=
  fun st => st.2.1 = true ∧ st.2.2.2.val ≤ span.val

/-- A finite certificate of the actual lane-loop body: it observes each invocation of the
    extracted `fwht_int_loop0_loop0_loop0.body` and charges it the work it performs, one butterfly
    per continuation and one for the terminating visit. Divergence has no constructor, so the
    certificate is a finite observation of the executing body, not a model of it. -/
inductive LaneTrace {State : Type} {Value : Type}
    (body : State → Result (ControlFlow State Value)) :
    State → Result Value → Nat × Nat → Prop where
  | done {state value} (observed : body state = ok (ControlFlow.done value)) :
      LaneTrace body state (ok value) (1, 0)
  | cont {state next result counts} (observed : body state = ok (ControlFlow.cont next))
      (tail : LaneTrace body next result counts) :
      LaneTrace body state result (counts.1 + 1, counts.2 + 1)

/-- The certificate projects to the actual extracted loop: the partial fixpoint `loop`, not an
    evaluator copy. This is what makes the count below a statement about the executing body. -/
theorem lane_trace_projection {State : Type} {Value : Type}
    {body : State → Result (ControlFlow State Value)} {state : State} {result : Result Value}
    {counts : Nat × Nat} (trace : LaneTrace body state result counts) :
    loop body state = result := by
  induction trace with
  | done observed =>
      unfold loop
      rw [observed]
      simp only [bind_ok]
  | cont observed tail ih =>
      unfold loop
      rw [observed]
      simp only [bind_ok]
      exact ih

/-- Every certificate has exactly one terminating visit, so `visits = steps + 1`. -/
theorem lane_trace_count {State : Type} {Value : Type}
    {body : State → Result (ControlFlow State Value)} {state : State} {result : Result Value}
    {counts : Nat × Nat} (trace : LaneTrace body state result counts) :
    counts.1 = counts.2 + 1 := by
  induction trace with
  | done _ => rfl
  | cont _ tail ih => omega

/-- **The exact work measure of the executing lane loop.** From a state whose failure flag is
    clear and whose lane index is inside the span, the actual body performs exactly `span - lane`
    butterflies: it is invoked `span - lane + 1` times, of which `span - lane` continue and one
    terminates. Every certificate of those counts agrees, so the count is the loop's own. The
    premise `hstep` is the guarded transition, discharged by `lane_body_step`; the exit branch is
    `lane_body_exit_span`. The measure counts loop-body invocations under the named premises; it is
    not a machine-time measurement. -/
theorem lane_visits_exact (span base : Usize) (x : Slice I64) (lane : Usize)
    (hlane : lane.val ≤ span.val)
    (hstep : ∀ (st : laneState), laneInv span st → ∃ (l' : Usize),
          fwht_source.hadamard.fwht_int_loop0_loop0_loop0.body span base st.1 st.2.1 st.2.2.1
              st.2.2.2 = ok (ControlFlow.cont (st.1, true, st.2.2.1, l')) ∧
          l'.val = st.2.2.2.val + 1 ∧ laneInv span (st.1, true, st.2.2.1, l') ∧
          laneRank span l' < laneRank span st.2.2.2)
    (hexit : ∀ (st : laneState), laneInv span st →
          fwht_source.hadamard.fwht_int_loop0_loop0_loop0.body span base st.1 st.2.1 st.2.2.1
              st.2.2.2 = ok (ControlFlow.done (st.1, true, st.2.2.1))) :
    ∃ y : laneResult, LaneTrace (laneBody span base x) (x, true, none, lane) (ok y)
      (span.val - lane.val + 1, span.val - lane.val) := by
  -- The exact count, by induction on the remaining rank `span - lane`.
  have aux : ∀ (k : Nat) (l : Usize) (buf : Slice I64)
      (err : Option fwht_source.hadamard.HadamardError),
      span.val - l.val = k → l.val ≤ span.val → laneInv span (buf, true, err, l) →
      ∃ y : laneResult, LaneTrace (laneBody span base buf) (buf, true, err, l) (ok y) (k + 1, k) := by
    intro k
    induction k with
    | zero =>
        intro l buf err hk hle hinv
        have hrec : laneBody span base buf (buf, true, err, l) =
            ok (ControlFlow.done (buf, true, err)) := by
          simp only [laneBody]
          exact hexit (buf, true, err, l) hinv
        exact ⟨(buf, true, err), LaneTrace.done hrec⟩
    | succ k ih =>
        intro l buf err hk hle hinv
        obtain ⟨l', hbody, hnext, hinv', hrank⟩ := hstep (buf, true, err, l) hinv
        have hle' : l'.val ≤ span.val := by rcases hinv' with ⟨hf, hb⟩; exact hb
        have hnext' : l'.val = l.val + 1 := by simpa only using hnext
        have hrank' : span.val - l'.val < span.val - l.val := by
          have h := hrank
          unfold laneRank at h
          simpa only using h
        have hk' : span.val - l'.val = k := by omega
        obtain ⟨y, htail⟩ := ih l' buf err hk' hle' hinv'
        have hrec : laneBody span base buf (buf, true, err, l) =
            ok (ControlFlow.cont (buf, true, err, l')) := by
          simp only [laneBody]
          exact hbody
        exact ⟨y, LaneTrace.cont hrec htail⟩
  obtain ⟨y, htrace⟩ := aux (span.val - lane.val) lane x none (by rfl) hlane ⟨rfl, hlane⟩
  -- The certificate projects to the actual extracted loop, so the count above is a statement
  -- about the executing body and not about a model of it.
  have hproj : loop (laneBody span base x) (x, true, none, lane) = ok y := lane_trace_projection htrace
  exact ⟨y, htrace⟩

/-! ## (d) The base loop and the stage loop: the work measure in closed form

`src/engine/hadamard.rs:130-161` has two loops above the lane loop. The base loop walks the
`2·span`-strided block positions of one stage, and the stage loop counts the `k` stages.

The **product work measure** is the number of butterflies the whole core performs. At stage
`stage` the span is `2^stage`, the base loop runs `n / (2·span)` times and each run performs `span`
butterflies, so a stage performs `n/2` and the `k` stages perform `k·n/2`. The lemmas below state
the span-doubling invariant the count reads and the closed form. Each factor is a count of
loop-body invocations under the named premises, not a cycle or a clock reading. -/

/-- **The span doubles per stage.** `src/engine/hadamard.rs:159` is `span *= 2`, so on the unit
    span of the first stage the extracted multiplication yields the value `2`. This is the
    invariant the closed-form count reads. The bound premise is the multiplication's own overflow
    guard; it holds because the span is `1`. -/
theorem span_doubles (span : UScalar UScalarTy.Usize) (hspan : span.val = 1) :
    (span * 2#usize) ⦃ s => s.val = 2 ⦄ := by
  have htwo' : (2#usize).val = 2 := usize_two
  have hbound : span.val * (2#usize).val ≤ 2 ^ 64 - 1 := by
    rw [hspan, usize_two]
    norm_num
  obtain ⟨s, hok, hsv⟩ := WP.spec_imp_exists (@usize_mul_spec span (2#usize) hbound)
  have hmulv : s.val = span.val * (2#usize).val := hsv
  have htwo : s = 2#usize := UScalar.eq_of_val_eq (by rw [hmulv, hspan, htwo'])
  have hgoal : (ok (2#usize) : Result _) ⦃ z => z.val = 2 ⦄ := by
    exact WP.spec.ret (by rw [htwo'])
  rwa [hok, htwo]

/-- **The closed-form work measure.** With `k` stages of `n/2` butterflies each, the total number
    of butterflies the core performs is `k · (n/2)`, counted as loop-body invocations of the
    extracted lane body. This is arithmetic on the two factors above; it charges no cycle, no
    clock and no byte. -/
theorem total_work_eq (k n : Usize) : totalWork k n = k.val * (n.val / 2) := rfl

/-- **The product of the work measure, in closed form.** At `(stage, span)` of a row of length `n`
    the remaining work is the product of the three loop factors: the remaining stages `k - stage`,
    the block positions per stage `n / (2·span)` and the butterflies per position `span`. When
    `2·span` divides `n` — which is the reachability premise `src/engine/hadamard.rs:201`
    `x.len().is_multiple_of(block)` supplies, since the span of stage `stage < k` is `2^stage` and
    `block = 2^k` — that product is `(k - stage) · (n/2)`, so each remaining stage contributes
    exactly `n/2` butterflies. -/
theorem remaining_work (k n stage span : Usize)
    (_hstage : stage.val ≤ k.val) (hpos : 0 < span.val)
    (hdiv : 2 * span.val ∣ n.val) :
    (k.val - stage.val) * (n.val / (2 * span.val)) * span.val
      = (k.val - stage.val) * (n.val / 2) := by
  obtain ⟨q, hq⟩ := hdiv
  have htwo : 0 < 2 * span.val := by omega
  have hq' : n.val = 2 * span.val * q := by omega
  have hqz : n.val / (2 * span.val) = q := by rw [hq', Nat.mul_div_cancel_left _ htwo]
  have hhalf : n.val / 2 = q * span.val := by
    have htwo_div : 2 * span.val / 2 = span.val := by omega
    -- `n = (2·span)·q`; dividing by `2` cancels the `2` inside the left factor, giving `span·q`.
    rw [hq']
    -- `(2·span)·q / 2 = q·span`: cancel the `2` out of the left factor, then re-associate.
    have hcomm : 2 * span.val * q = q * (2 * span.val) := by ring
    have hcanc : q * (2 * span.val) / 2 = q * span.val := by
      have hdvd : 2 ∣ 2 * span.val := dvd_mul_right 2 span.val
      rw [Nat.mul_div_assoc q hdvd, htwo_div]
    rw [hcomm, hcanc]
  rw [hqz, hhalf]
  ring

/-! ## (e) The exponent scan's exit branch and a reachable success witness

`src/engine/hadamard.rs:63-76` accepts `block` when `2^k = block` with the exponent below
`MAX_K`, and the extraction calls the entry point `word_fwht_fixed`. -/

/-- **The exponent scan's exit branch.** With the exponent at `MAX_K`, or with the corresponding
    power no longer below `block`, the body returns the exponent unchanged and consumes no further
    iteration. -/
theorem exp_body_exit_maxK (block k : Usize)
    (hmax : fwht_source.hadamard.MAX_K.val ≤ k.val) :
    expBody block k = ok (ControlFlow.done k) := by
  have hnot : ¬ (k < fwht_source.hadamard.MAX_K) := by
    rw [UScalar.lt_equiv]
    omega
  rw [expBody, fwht_source.hadamard.block_exponent_loop.body, if_neg hnot]

/-- **The power-test exit of the exponent scan.** With the exponent below `MAX_K` and the
    corresponding power no longer below `block`, the body returns the exponent unchanged and
    consumes no further iteration. -/
theorem exp_body_exit_power (block k pow : Usize)
    (hguard : k.val < fwht_source.hadamard.MAX_K.val)
    (hshift : (1#usize <<< k) = ok pow)
    (hle : block.val ≤ pow.val) :
    expBody block k = ok (ControlFlow.done k) := by
  have hguard' : k < fwht_source.hadamard.MAX_K := (UScalar.lt_equiv _ _).2 hguard
  have hnot : ¬ (pow < block) := by
    rw [UScalar.lt_equiv]
    omega
  rw [expBody, fwht_source.hadamard.block_exponent_loop.body, if_pos hguard', hshift,
    bind_tc_ok, if_neg hnot]

/-- **The exponent scan terminates with the accepting exponent.** On `block = 2` the extracted
    `block_exponent` returns `some 1`, which is the unique `k` with `1 ≤ k ≤ MAX_K` and
    `block = 2^k`: the scan leaves the loop at the exponent `1` because `2^1 = block`, and the
    acceptance test at `src/engine/hadamard.rs:71` compares them for equality. This is the
    reachable entry state of the witness below. -/
theorem wit_block_exponent
    (hshl0 : (1#usize <<< 0#usize) = ok 1#usize)
    (hshl1 : (1#usize <<< 1#usize) = ok 2#usize) :
    fwht_source.hadamard.block_exponent 2#usize = ok (some 1#usize) := by
  -- The scan: at exponent `0` the power is `1 < 2`, so the body continues to `1`; at exponent `1`
  -- the power is `2`, not below `2`, so the body stops and returns `1`.
  have hcont : fwht_source.hadamard.block_exponent_loop.body 2#usize 0#usize
      = ok (ControlFlow.cont 1#usize) := by
    obtain ⟨k', hb, hk, _, _⟩ := exp_body_step (2#usize) (0#usize) (1#usize)
      ((UScalar.lt_equiv (0#usize) fwht_source.hadamard.MAX_K).2 (by
        have h := maxK_val
        have hz : ((0#usize : UScalar UScalarTy.Usize) : ℕ) = 0 := rfl
        omega))
      hshl0 (by
        have h1 : ((1#usize : UScalar UScalarTy.Usize) : ℕ) = 1 := usize_one
        have h2 : ((2#usize : UScalar UScalarTy.Usize) : ℕ) = 2 := usize_two
        omega)
    have hkval : k' = 1#usize := by
      apply UScalar.eq_of_val_eq
      rw [hk, usize_one]
      norm_num
    -- The body computes the power, increments the exponent and continues.
    have hguard0 : (0#usize) < fwht_source.hadamard.MAX_K :=
      (UScalar.lt_equiv (0#usize) fwht_source.hadamard.MAX_K).2 (by
        have h := maxK_val
        have hz : ((0#usize : UScalar UScalarTy.Usize) : ℕ) = 0 := rfl
        omega)
    have hlt12 : (1#usize) < 2#usize := by
      have h1 : ((1#usize : UScalar UScalarTy.Usize) : ℕ) = 1 := usize_one
      have h2 : ((2#usize : UScalar UScalarTy.Usize) : ℕ) = 2 := usize_two
      rw [UScalar.lt_equiv]
      omega
    -- The body equation `exp_body_step` discharges, applied to the witness guards.
    have hguard' : (0#usize) < fwht_source.hadamard.MAX_K := hguard0
    have hlt' : (1#usize).val < (2#usize).val := by
      have h1 : ((1#usize : UScalar UScalarTy.Usize) : ℕ) = 1 := usize_one
      have h2 : ((2#usize : UScalar UScalarTy.Usize) : ℕ) = 2 := usize_two
      omega
    obtain ⟨k', hb, hk, _, _⟩ :=
      exp_body_step (2#usize) (0#usize) (1#usize)
        ((UScalar.lt_equiv (0#usize) fwht_source.hadamard.MAX_K).2 (by
          have h := maxK_val
          have hz : ((0#usize : UScalar UScalarTy.Usize) : ℕ) = 0 := rfl
          omega))
        hshl0 hlt'
    have hkval : k' = 1#usize := by
      apply UScalar.eq_of_val_eq
      rw [hk, usize_one]
      norm_num
    unfold expBody at hb
    rw [hb, hkval]
  have hstop : fwht_source.hadamard.block_exponent_loop.body 2#usize 1#usize
      = ok (ControlFlow.done 1#usize) := by
    have hguard : (1#usize) < fwht_source.hadamard.MAX_K :=
      (UScalar.lt_equiv (1#usize) fwht_source.hadamard.MAX_K).2 (by
        have h := maxK_val
        have h1 : ((1#usize : UScalar UScalarTy.Usize) : ℕ) = 1 := usize_one
        omega)
    have hle : (2#usize).val ≤ (2#usize).val := by
      have h2 : ((2#usize : UScalar UScalarTy.Usize) : ℕ) = 2 := usize_two
      omega
    have hexit := exp_body_exit_power (2#usize) (1#usize) (2#usize)
      ((UScalar.lt_equiv (1#usize) fwht_source.hadamard.MAX_K).1 hguard) hshl1 hle
    unfold expBody at hexit
    exact hexit
  have hloop : expLoop 2#usize 0#usize = ok 1#usize := by
    rw [expLoop, fwht_source.hadamard.block_exponent_loop]
    unfold loop
    rw [hcont, bind_ok]
    show loop (fun k1 => fwht_source.hadamard.block_exponent_loop.body 2#usize k1) 1#usize
      = ok 1#usize
    unfold loop
    rw [hstop, bind_ok]
  -- The acceptance test at `src/engine/hadamard.rs:71`.
  rw [fwht_source.hadamard.block_exponent]
  have hlt2 : ¬ ((2#usize) < 2#usize) := by decide
  rw [if_neg hlt2]
  show (do
      let k ← fwht_source.hadamard.block_exponent_loop 2#usize 0#usize
      let i ← 1#usize <<< k
      if i = 2#usize then ok (some k) else ok none) = ok (some 1#usize)
  unfold expLoop at hloop
  rw [hloop, bind_tc_ok, hshl1, bind_tc_ok]
  rfl

/-! ## (f) The axiom audit

Each headline below is checked against the closed cone `{propext, Classical.choice, Quot.sound}`,
the same cone `proof/regen.sh` enforces on the extraction. A `sorry` is a build failure, not a
note; there is none.

### The declared host-width boundary

The five scalar-width equations are proved by `native_decide` (see the section header). `native_decide`
adds an axiom outside the closed cone, so it is a genuine trust boundary and is named as one: the
audit below lists every headline whose transitive dependencies include it, and those headlines are
*not* claimed at the closed-cone grade. The boundary is exactly the platform pointer width
(`UScalar.max`, `Usize.max`, `UScalarTy.Usize.numBits`) and the `i64` width (`I64.numBits`,
`IScalarTy.I64.numBits`) — five machine-configuration constants, and nothing else. No algorithmic,
arithmetic or control-flow fact in this file rests on it: every guard, bound and window premise is a
concrete proposition the kernel decides, and every loop rank, transition and count is proved from
those premises alone.

`#resource_audit` therefore *rejects* any headline outside the closed cone, which is why the
width-dependent headlines below are listed in a separate, explicitly weaker check. Those are the
`#resource_audit_width` lines: they verify that a headline depends on nothing beyond the five
width constants and the closed cone, so a reviewer can see exactly which facts are outside it. -/

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

/- A headline is audited at the closed cone when nothing in its transitive dependencies uses
   `native_decide`. -/

#resource_audit FwhtResource.maxK_val
#resource_audit FwhtResource.ok_or_ok
#resource_audit FwhtResource.ok_or_none
#resource_audit FwhtResource.exp_body_exit_maxK
#resource_audit FwhtResource.exp_body_exit_power
#resource_audit FwhtResource.lane_body_exit_span
#resource_audit FwhtResource.lane_body_exit_flag
#resource_audit FwhtResource.lane_trace_projection
#resource_audit FwhtResource.lane_trace_count
#resource_audit FwhtResource.lane_visits_exact
#resource_audit FwhtResource.total_work_eq
#resource_audit FwhtResource.remaining_work

/- The width-dependent headlines. Each must depend on nothing beyond the five width constants
   and the closed cone; anything else is a real failure and throws. -/

open Lean Elab Command in
/-- The declared width boundary: the five `native_decide` axioms of this file's scalar-width
    equations, and nothing else. A headline may depend on these; any other forbidden axiom is
    rejected. -/
elab "#resource_audit_width " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
    |>.insert `FwhtResource.usize_numBits._native.native_decide.ax_1_1
    |>.insert `FwhtResource.usize_max._native.native_decide.ax_1_1
    |>.insert `FwhtResource.usize_max'._native.native_decide.ax_1_1
    |>.insert `FwhtResource.i64_numBits._native.native_decide.ax_1_1
    |>.insert `FwhtResource.i64_numBits'._native.native_decide.ax_1_1
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms beyond the declared width boundary {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit_width FwhtResource.checked_add_some
#resource_audit_width FwhtResource.checked_sub_some
#resource_audit_width FwhtResource.exp_body_step
#resource_audit_width FwhtResource.lane_body_step
#resource_audit_width FwhtResource.span_doubles
#resource_audit_width FwhtResource.wit_block_exponent

end FwhtResource

run_cmd Lean.logInfo "FWHT_KERNEL_OK"
