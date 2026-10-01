import RmsNormSource
import Aeneas

/-
SCOPE: resource transitions, work measure, refusal branches and success witness of the actual
extracted `norm.rms_norm_fixed` body and the `fixed_rsqrt` it calls; no hardware timing.
REGISTER: scalar source refinement over `i128`/`i64` work counts; no ring or carrier
correspondence, no GPU/ABI claim.
CORPUS: `src/engine/norm.rs` `rms_norm_fixed`, `src/engine/value.rs` `round_shift_32` and
`div_round_nearest`, `src/numeric/fixed_rsqrt.rs` `fixed_rsqrt`/`isqrt`, read as the actual
extracted `norm.rms_norm_fixed_loop{0,1}.body`, `fixed_rsqrt.isqrt_loop{0,1}.body`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean
GATE: `python3 proof/resource/RmsNorm_gate.py --controls`, lane-locked, 540-second task budget.
GRADE: Lean kernel check only; the work counts are counted operations under named premises, not
measured cycles.
-/

namespace RmsNormResource

open Aeneas Aeneas.Std Result

/-! ## The accumulation work measure: prefix sums of squares

`sqPrefix row k` is the sum of the squares of the first `k` row coordinates, the value the
`i128` accumulator of `norm.rms_norm_fixed_loop0` holds after consuming `k` elements. -/

/-- The sum of the squares of the first `k` row coordinates, as an `Int`. -/
def sqPrefix (row : Slice I64) (k : Nat) : ℤ :=
  ((row.val.take k).map (fun x => x.val * x.val)).sum

theorem map_sq_sum_nonneg (l : List I64) :
    0 ≤ (l.map (fun x => x.val * x.val)).sum := by
  induction l with
  | nil => simp
  | cons x xs ih =>
      simp only [List.map_cons, List.sum_cons]
      exact add_nonneg (mul_self_nonneg x.val) ih

theorem sqPrefix_nonneg (row : Slice I64) (k : Nat) : 0 ≤ sqPrefix row k := by
  unfold sqPrefix; exact map_sq_sum_nonneg _

theorem sqPrefix_total (row : Slice I64) :
    sqPrefix row row.val.length = (row.val.map (fun x => x.val * x.val)).sum := by
  unfold sqPrefix; rw [List.take_length]

theorem sqPrefix_le_total (row : Slice I64) (k : Nat) :
    sqPrefix row k ≤ (row.val.map (fun x => x.val * x.val)).sum := by
  unfold sqPrefix
  conv_rhs => rw [← List.take_append_drop k row.val]
  rw [List.map_append, List.sum_append]
  have hdrop := map_sq_sum_nonneg (row.val.drop k)
  linarith

theorem sqPrefix_succ (row : Slice I64) {k : Nat} (hk : k < row.val.length) :
    sqPrefix row (k + 1) = sqPrefix row k + row.val[k].val * row.val[k].val := by
  unfold sqPrefix
  rw [List.take_add_one, List.getElem?_eq_getElem hk]
  simp only [List.map_append, List.sum_append, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, Option.toList_some]
  ring

/-- Half-biased round-to-nearest `num / den`, the value of `value.div_round_nearest num den`. -/
def divRoundNearest (n d : ℤ) : ℤ := (n + d / 2) / d

/-- The Q32.32 epilogue value of one coordinate: `round(x · root / 2^32)`, ties upward. -/
def epilogue (root : I128) (x : I64) : ℤ :=
  divRoundNearest (x.val * root.val) ((2 : ℤ) ^ (32 : ℕ))

/-! ## Checked `i128` arithmetic witnesses

The two `I128.checked_*` operations per body entry are total on the stated windows, so the
extracted `match` never takes its `none` arm there. -/

theorem I128_checked_mul_some (x y : I128)
    (hlo : I128.min ≤ x.val * y.val) (hhi : x.val * y.val ≤ I128.max) :
    ∃ z, x.checked_mul y = some z ∧ z.val = x.val * y.val := by
  have hspec := I128.checked_mul_bv_spec x y
  cases hc : core.num.checked_mul_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I128.checked_mul] using hc, hspec.2.2.1⟩

theorem I128_checked_add_some (x y : I128)
    (hlo : I128.min ≤ x.val + y.val) (hhi : x.val + y.val ≤ I128.max) :
    ∃ z, x.checked_add y = some z ∧ z.val = x.val + y.val := by
  have hspec := I128.checked_add_bv_spec x y
  cases hc : core.num.checked_add_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I128.checked_add] using hc, hspec.2.2.1⟩

/-! ## The `value.rs` bridges

`value.round_shift_32 v = (v + 2^31) >>> 32` and `value.div_round_nearest num den =
(num + den/2)/den`, both read off the extracted scalar operations. -/

/-- The extracted `x >>> 32#i32` is `Int` floor division by `2^32`. -/
theorem I128_shiftRight_32_val (x : I128) :
    ∃ y : I128, (x >>> 32#i32) = ok y ∧ y.val = x.val / 2 ^ (32 : ℕ) := by
  refine ⟨⟨x.bv.sshiftRight 32⟩, ?_, ?_⟩
  · show IScalar.shiftRight x (32#i32).toNat = ok _
    have h32 : (32#i32).toNat = 32 := by decide
    rw [h32]
    unfold IScalar.shiftRight
    rw [if_pos (by norm_num [IScalarTy.numBits, I128.numBits])]
  · show (⟨x.bv.sshiftRight 32⟩ : I128).val = x.val / 2 ^ (32 : ℕ)
    rw [show (⟨x.bv.sshiftRight 32⟩ : I128).val = (x.bv.sshiftRight 32).toInt from rfl]
    rw [BitVec.toInt_sshiftRight, Int.shiftRight_eq_div_pow]
    norm_num

/-- The extracted constant `1#i128 <<< 31#i32 = 2^31`. -/
theorem I128_shiftLeft_31 : (1#i128 <<< 31#i32) = ok (2147483648#i128) := by
  have h : (1#i128 <<< 31#i32) ⦃ z => z = (2147483648#i128) ⦄ := by
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The `isqrt` seed `2^126` as an `I128` value. -/
def isqrtSeed : I128 := 85070591730234615865843651857942052864#i128

/-- The extracted constant `1#i128 <<< 126#i32 = 2^126`, the `isqrt` seed. -/
theorem I128_shiftLeft_126 : (1#i128 <<< 126#i32) = ok isqrtSeed := by
  have h : (1#i128 <<< 126#i32) ⦃ z => z = isqrtSeed ⦄ := by
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [isqrtSeed, I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The `isqrt` seed's value: `isqrtSeed.val = 2^126`. -/
theorem isqrtSeed_val : isqrtSeed.val = (2 : ℤ) ^ (126 : ℕ) := by
  rw [isqrtSeed]
  norm_num

/-- The extracted `x >>> 2#i32` is `Int` floor division by four, for `0 ≤ x`. -/
theorem I128_shiftRight_2_val (x : I128) :
    ∃ y : I128, (x >>> 2#i32) = ok y ∧ y.val = x.val / 4 := by
  obtain ⟨y, hy, hyv, _⟩ := WP.spec_imp_exists
    (I128.ShiftRight_IScalar_spec (x := x) (y := 2#i32) (by norm_num)
      (by norm_num [I128.numBits]))
  rw [hy]
  refine ⟨y, rfl, ?_⟩
  have h2 : (2#i32).toNat = 2 := by decide
  rw [hyv, h2, Int.shiftRight_eq_div_pow]
  norm_num

/-- The extracted `x >>> 1#i32` is `Int` floor division by two. -/
theorem I128_shiftRight_1_val (x : I128) :
    ∃ y : I128, (x >>> 1#i32) = ok y ∧ y.val = x.val / 2 := by
  obtain ⟨y, hy, hyv, _⟩ := WP.spec_imp_exists
    (I128.ShiftRight_IScalar_spec (x := x) (y := 1#i32) (by norm_num)
      (by norm_num [I128.numBits]))
  rw [hy]
  refine ⟨y, rfl, ?_⟩
  have h1 : (1#i32).toNat = 1 := by decide
  rw [hyv, h1, Int.shiftRight_eq_div_pow]
  norm_num

/-- **Epilogue bridge.** `value.round_shift_32 v` is the half-biased rounding
`divRoundNearest v.val 2^32` whenever `v + 2^31` is inside the `i128` window. -/
theorem round_shift_32_spec (v : I128)
    (hlo : I128.min ≤ v.val + 2 ^ (31 : ℕ)) (hhi : v.val + 2 ^ (31 : ℕ) ≤ I128.max) :
    ∃ y : I128, rms_norm_source.value.round_shift_32 v = ok y ∧
      y.val = divRoundNearest v.val ((2 : ℤ) ^ (32 : ℕ)) := by
  unfold rms_norm_source.value.round_shift_32
  rw [I128_shiftLeft_31]
  simp only [bind_tc_ok]
  have h31 : (2147483648#i128).val = 2 ^ (31 : ℕ) := by norm_num
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (I128.add_spec (x := v) (y := 2147483648#i128) (by rw [h31]; exact hlo)
      (by rw [h31]; exact hhi))
  rw [hi1]
  simp only [bind_tc_ok]
  obtain ⟨y, hy, hyv⟩ := I128_shiftRight_32_val i1
  refine ⟨y, hy, ?_⟩
  rw [hyv, hi1v, h31]
  unfold divRoundNearest
  norm_num

/-- **Division bridge.** `value.div_round_nearest num den` is `divRoundNearest num.val den.val`
for a positive `den` and a nonnegative `num` whose half-biased numerator fits. -/
theorem div_round_nearest_spec (num den : I128) (hden : 0 < den.val) (hnum : 0 ≤ num.val)
    (hadd : num.val + den.val / 2 ≤ I128.max) :
    rms_norm_source.value.div_round_nearest num den ⦃ z =>
      z.val = divRoundNearest num.val den.val ⦄ := by
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) :=
    by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 :=
    by norm_num [I128.max, IScalar.max, I128.numBits]
  unfold rms_norm_source.value.div_round_nearest
  refine WP.spec_bind (I128.div_spec (x := den) (y := 2#i128) (by norm_num)
    (by intro h; exact absurd h.2 (by norm_num))) ?_
  intro i hi_val
  have hi_val' : i.val = den.val / 2 := by
    rw [hi_val]
    exact Int.tdiv_eq_ediv_of_nonneg (le_of_lt hden)
  have hinn : 0 ≤ i.val := by
    rw [hi_val']; exact Int.ediv_nonneg (le_of_lt hden) (by norm_num)
  refine WP.spec_bind (I128.add_spec (x := num) (y := i)
    (by rw [hI128min]; linarith [hnum, hinn])
    (by rw [hi_val']; simpa [hI128max] using hadd)) ?_
  intro i1 hi1_val
  refine WP.spec_mono (I128.div_spec (x := i1) (y := den) (by omega)
    (by intro h; exact absurd h.2 (by omega))) ?_
  intro z hz_val
  rw [hz_val, hi1_val, hi_val']
  unfold divRoundNearest
  exact Int.tdiv_eq_ediv_of_nonneg (by positivity)

/-! ## Loop 0: the sum-of-squares accumulation

The body entry costs one `Slice.index_usize`, one `i128` square and one `i128` add, and lowers
the index by one. The rank `(Slice.len row).val - i.val` falls by exactly one per entry. -/

/-- **Guarded accumulation transition.** Under `i.val < n.val` the body takes its
`sum_ok = true` continuation, consumes coordinate `i`, adds its square to the accumulator,
advances the index by one, and lowers the rank by exactly one. -/
theorem loop0_body_step (row : Slice I64) (n : Usize) (sum_sq : I128) (i : Usize)
    (hilt : i.val < n.val) (hn : n.val = row.val.length)
    (hsum : sum_sq.val = sqPrefix row i.val)
    (hfit : (row.val.map (fun x => x.val * x.val)).sum ≤ I128.max) :
    ∃ total i2, rms_norm_source.norm.rms_norm_fixed_loop0.body row n sum_sq i true
        = ok (ControlFlow.cont (total, i2, true)) ∧
      total.val = sqPrefix row i2.val ∧ i2.val = i.val + 1 ∧
      n.val - i2.val < n.val - i.val ∧ n.val - i2.val + 1 = n.val - i.val := by
  have hbnd : i.val < row.val.length := by omega
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) :=
    by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 :=
    by norm_num [I128.max, IScalar.max, I128.numBits]
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec row i hbnd)
  have hxval : i1.val = row.val[i.val].val := by rw [hi1v]
  have hi1b := i1.hBounds
  rw [IScalarTy.I64_numBits_eq] at hi1b
  have hvv : (core.convert.num.FromI128I64.from i1).val = i1.val :=
    core.convert.num.FromI128I64.from_val_eq i1
  have hmullo : I128.min ≤ (core.convert.num.FromI128I64.from i1).val *
      (core.convert.num.FromI128I64.from i1).val := by
    rw [hvv, hI128min]; nlinarith [mul_self_nonneg i1.val]
  have hmulhi : (core.convert.num.FromI128I64.from i1).val *
      (core.convert.num.FromI128I64.from i1).val ≤ I128.max := by
    have hb1 : -(2 : ℤ) ^ (63 : ℕ) ≤ i1.val := by have h := hi1b.1; norm_num at h; linarith
    have hb2 : i1.val < (2 : ℤ) ^ (63 : ℕ) := by have h := hi1b.2; norm_num at h; exact h
    rw [hvv, hI128max]
    nlinarith [hb1, hb2, sq_nonneg i1.val,
      sq_nonneg (i1.val - (2 : ℤ) ^ (63 : ℕ)), sq_nonneg (i1.val + (2 : ℤ) ^ (63 : ℕ))]
  obtain ⟨sq, hsq, hsqv⟩ := I128_checked_mul_some _ _ hmullo hmulhi
  have hsumnn : 0 ≤ sum_sq.val := by rw [hsum]; exact sqPrefix_nonneg row i.val
  have hsuc : sum_sq.val + sq.val = sqPrefix row (i.val + 1) := by
    rw [hsqv, hsum, sqPrefix_succ row hbnd]
    rw [hvv, hxval]
  have hsumhi : sum_sq.val + sq.val ≤ I128.max := by
    rw [hsuc]; exact le_trans (sqPrefix_le_total row (i.val + 1)) hfit
  obtain ⟨total, htot, htotv⟩ := I128_checked_add_some sum_sq sq
    (by rw [hI128min]; nlinarith [hsumnn, sq_nonneg sq.val]) hsumhi
  have hone : (1#usize).val = 1 := by norm_num
  have hiaddmax : i.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    have hb := row.property
    rw [h1]; omega
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hiaddmax)
  rw [hone] at hi2v
  refine ⟨total, i2, ?_, ?_, ?_, ?_, ?_⟩
  · rw [rms_norm_source.norm.rms_norm_fixed_loop0.body, hi1]
    simp only [lift, bind_tc_ok]
    rw [hsq]
    dsimp only
    rw [htot]
    dsimp only
    rw [hi2]
    simp only [bind_tc_ok]
    have hilt' : i < n := (UScalar.lt_equiv i n).2 hilt
    rw [if_pos hilt', if_pos trivial]
    rfl
  · rw [htotv, hsuc]
    rw [hi2v]
  · exact hi2v
  · omega
  · omega

/-- **Accumulation exit at the row end.** With `i.val ≥ n.val` the body returns the accumulator
and the flag unchanged, at no per-coordinate cost. -/
theorem loop0_body_end (row : Slice I64) (n : Usize) (sum_sq : I128) (i : Usize) (sum_ok : Bool)
    (hge : ¬ (i.val < n.val)) :
    rms_norm_source.norm.rms_norm_fixed_loop0.body row n sum_sq i sum_ok =
      ok (ControlFlow.done (sum_sq, sum_ok)) := by
  unfold rms_norm_source.norm.rms_norm_fixed_loop0.body
  have hnot : ¬ (i < n) := by
    intro hc; exact hge ((UScalar.lt_equiv i n).1 hc)
  rw [if_neg hnot]

/-- **Accumulation exit on refusal.** With `sum_ok = false` and `i.val < n.val` the body returns
the refused pair without reading a coordinate. -/
theorem loop0_body_exit (row : Slice I64) (n : Usize) (sum_sq : I128) (i : Usize) (sum_ok : Bool)
    (hilt : i.val < n.val) (hflag : sum_ok = false) :
    rms_norm_source.norm.rms_norm_fixed_loop0.body row n sum_sq i sum_ok =
      ok (ControlFlow.done (sum_sq, sum_ok)) := by
  unfold rms_norm_source.norm.rms_norm_fixed_loop0.body
  have hilt' : i < n := (UScalar.lt_equiv i n).2 hilt
  have hflag' : ¬ (sum_ok = true) := by rw [hflag]; decide
  rw [if_pos hilt', if_neg hflag', hflag]

/-- **Accumulation refusal entry.** A `sum_ok = false` state exits immediately without reading a
coordinate, so a refused accumulation costs no further body entry. -/
theorem loop0_body_refused (row : Slice I64) (n : Usize) (sum_sq : I128) (i : Usize)
    (hilt : i.val < n.val) :
    rms_norm_source.norm.rms_norm_fixed_loop0.body row n sum_sq i false =
      ok (ControlFlow.done (sum_sq, false)) := by
  exact loop0_body_exit row n sum_sq i false hilt rfl

/-- The reachable invariant of the production accumulation loop: flag, index bound and the
accumulated prefix sum. -/
def loop0Inv (row : Slice I64) (n : Usize) (s : I128 × Usize × Bool) : Prop :=
  s.2.2 = true ∧ s.2.1.val ≤ n.val ∧ s.1.val = sqPrefix row s.2.1.val

/-- **Accumulation loop resource spec.** The rank `(Slice.len row).val - i.val` starts at
`(Slice.len row).val` (index `0`) and every body entry lowers it by exactly one; the loop exits
at rank `0`, so it takes exactly `(Slice.len row).val` body entries — one `i128` square and one
`i128` add each — and its accumulator is the total sum of squares. -/
theorem loop0_work (row : Slice I64)
    (hfit : (row.val.map (fun x => x.val * x.val)).sum ≤ I128.max) :
    ((Slice.len row).val - (0#usize).val = row.val.length) ∧
    ((Slice.len row).val - (Slice.len row).val = 0) ∧
    (∃ S : I128, rms_norm_source.norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
       = ok (S, true) ∧ S.val = (row.val.map (fun x => x.val * x.val)).sum) := by
  have hlen : (Slice.len row).val = row.val.length := Slice.len_val row
  have hz : (0#usize).val = 0 := by norm_num
  refine ⟨by rw [hlen, hz]; norm_num, by rw [hlen]; norm_num, ?_⟩
  have hrank : rms_norm_source.norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true ⦃ b =>
      ∃ S : I128, b = (S, true) ∧
        S.val = (row.val.map (fun x => x.val * x.val)).sum ⦄ := by
    unfold rms_norm_source.norm.rms_norm_fixed_loop0
    apply loop.spec_decr_nat
      (measure := fun s : I128 × Usize × Bool => (Slice.len row).val - s.2.1.val)
      (inv := loop0Inv row (Slice.len row))
      (post := fun b : I128 × Bool =>
        ∃ S : I128, b = (S, true) ∧ S.val = (row.val.map (fun x => x.val * x.val)).sum)
    · intro s hs
      rcases s with ⟨sum_sq, i, sum_ok⟩
      dsimp only at hs ⊢
      obtain ⟨hok, hile, hsum⟩ := hs
      dsimp only at hok hile hsum
      rw [hok]
      by_cases hlt : i.val < (Slice.len row).val
      · obtain ⟨total, i2, hbody, htot, hi2, hrank1, hrank2⟩ :=
          loop0_body_step row (Slice.len row) sum_sq i hlt (by rw [Slice.len_val]) hsum hfit
        rw [hbody]
        have hi2le : i2.val ≤ (Slice.len row).val := by omega
        refine (WP.spec_ok (ControlFlow.cont (total, i2, true))).mpr ?_
        dsimp only
        have hdrop : (Slice.len row).val - i2.val < (Slice.len row).val - i.val := by omega
        exact ⟨⟨rfl, hi2le, htot⟩, hdrop⟩
      · have hnot : ¬ (i < Slice.len row) := by
          intro hc; exact hlt ((UScalar.lt_equiv i (Slice.len row)).1 hc)
        have hie : i.val = (Slice.len row).val := by omega
        have hlen : (Slice.len row).val = row.val.length := Slice.len_val row
        have hbody : rms_norm_source.norm.rms_norm_fixed_loop0.body row (Slice.len row) sum_sq i true
            = ok (ControlFlow.done (sum_sq, true)) := by
          refine loop0_body_end row (Slice.len row) sum_sq i true ?_
          intro hc; exact hlt ((UScalar.lt_equiv i (Slice.len row)).1 hc)
        rw [hbody]
        exact ⟨sum_sq, rfl, by rw [hsum, hie, hlen, sqPrefix_total]⟩
    · refine ⟨rfl, ?_, ?_⟩
      · simp
      · show (0#i128).val = sqPrefix row (0#usize).val
        rw [sqPrefix]
        simp
  obtain ⟨y, hrun, hS⟩ := WP.spec_imp_exists hrank
  obtain ⟨S, hEq, hval⟩ := hS
  exact ⟨S, by rw [hrun, hEq], hval⟩

/-! ## Loop 1: the value path

The body entry costs one `Slice.index_usize`, one checked `i128` product, one
`value.round_shift_32` and one `Vec.push` under the output-window premise. -/

/-- **Guarded epilogue transition.** Under `i.val < n.val` and `out_ok = true` the body takes its
continuation, pushes the epilogue value of coordinate `i`, advances the index by one, and lowers
the rank by exactly one. -/
theorem loop1_body_step (row : Slice I64) (n : Usize) (root : I128)
    (out : alloc.vec.Vec I64) (i : Usize)
    (hilt : i.val < n.val) (hn : n.val = row.val.length)
    (hout : out.val.map (fun y => y.val) = (row.val.take i.val).map (epilogue root))
    (hprod : ∀ (j : ℕ) (hj : j < row.val.length),
        I128.min ≤ row.val[j].val * root.val ∧ row.val[j].val * root.val ≤ I128.max ∧
          I128.min ≤ row.val[j].val * root.val + 2 ^ (31 : ℕ) ∧
            row.val[j].val * root.val + 2 ^ (31 : ℕ) ≤ I128.max)
    (hfit : ∀ (j : ℕ) (hj : j < row.val.length),
        -((2 : ℤ) ^ (63 : ℕ)) ≤ epilogue root row.val[j] ∧
          epilogue root row.val[j] ≤ (2 : ℤ) ^ (63 : ℕ) - 1) :
    ∃ out2 i2, rms_norm_source.norm.rms_norm_fixed_loop1.body row n root out i true
        = ok (ControlFlow.cont (out2, i2, true)) ∧
      out2.val.map (fun y => y.val) = (row.val.take i2.val).map (epilogue root) ∧
      i2.val = i.val + 1 ∧
      n.val - i2.val < n.val - i.val ∧ n.val - i2.val + 1 = n.val - i.val := by
  have hbnd : i.val < row.val.length := by omega
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) :=
    by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 :=
    by norm_num [I128.max, IScalar.max, I128.numBits]
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec row i hbnd)
  have hxval : i1.val = row.val[i.val].val := by rw [hi1v]
  have hvv : (core.convert.num.FromI128I64.from i1).val = i1.val :=
    core.convert.num.FromI128I64.from_val_eq i1
  have hpr := hprod i.val hbnd
  have hmullo : I128.min ≤ (core.convert.num.FromI128I64.from i1).val * root.val := by
    rw [hvv, hxval]; exact hpr.1
  have hmulhi : (core.convert.num.FromI128I64.from i1).val * root.val ≤ I128.max := by
    rw [hvv, hxval]; exact hpr.2.1
  obtain ⟨prod, hprod_eq, hprodv⟩ := I128_checked_mul_some _ _ hmullo hmulhi
  have hfitv := hfit i.val hbnd
  have hprodval : prod.val = row.val[i.val].val * root.val := by rw [hprodv, hvv, hxval]
  have hlo31 : I128.min ≤ prod.val + 2 ^ (31 : ℕ) := by rw [hprodval]; exact hpr.2.2.1
  have hhi31 : prod.val + 2 ^ (31 : ℕ) ≤ I128.max := by rw [hprodval]; exact hpr.2.2.2
  obtain ⟨normed, hnormed_eq, hnormedv⟩ := round_shift_32_spec prod hlo31 hhi31
  have hnormed_epi : normed.val = epilogue root row.val[i.val] := by
    rw [hnormedv, hprodval]; rfl
  have hfitlo : -((2 : ℤ) ^ (63 : ℕ)) ≤ normed.val := by rw [hnormed_epi]; exact hfitv.1
  have hfithi : normed.val ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by rw [hnormed_epi]; exact hfitv.2
  have hmaxv : (IScalar.cast .I128 core.num.I64.MAX).val = (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MAX.val = (2 : ℤ) ^ (63 : ℕ) - 1 := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 ((2 : ℤ) ^ (63 : ℕ) - 1)
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hminv : (IScalar.cast .I128 core.num.I64.MIN).val = -(2 : ℤ) ^ (63 : ℕ) := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MIN.val = -(2 : ℤ) ^ (63 : ℕ) := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 (-(2 : ℤ) ^ (63 : ℕ))
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hnotmax : ¬ (normed > IScalar.cast .I128 core.num.I64.MAX) := by
    intro hc
    rw [gt_iff_lt, IScalar.lt_equiv, hmaxv] at hc
    omega
  have hnotmin : ¬ (normed < IScalar.cast .I128 core.num.I64.MIN) := by
    intro hc
    rw [IScalar.lt_equiv, hminv] at hc
    omega
  have hcastval : (IScalar.cast .I64 normed).val = normed.val := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 normed.val
      (by norm_num [IScalarTy.numBits]; omega) (by norm_num [IScalarTy.numBits]; omega)
  have hpushmax : out.val.length < Usize.max := by
    have hlen : out.val.length = i.val := by
      have h := congrArg List.length hout
      simp only [List.length_map, List.length_take] at h
      rw [Nat.min_eq_left (le_of_lt hbnd)] at h
      exact h
    have hb := row.property
    rw [← hn] at hb
    omega
  obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec out (IScalar.cast .I64 normed) hpushmax)
  have hone : (1#usize).val = 1 := by norm_num
  have hiaddmax : i.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    have hb := row.property
    rw [h1]; omega
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hiaddmax)
  rw [hone] at hi2v
  refine ⟨out2, i2, ?_, ?_, ?_, ?_, ?_⟩
  · rw [rms_norm_source.norm.rms_norm_fixed_loop1.body, hi1]
    simp only [lift, bind_tc_ok]
    rw [hprod_eq]
    dsimp only
    rw [hnormed_eq]
    simp only [bind_tc_ok]
    rw [if_neg hnotmax, if_neg hnotmin]
    rw [hpush, hi2]
    simp only [bind_tc_ok]
    have hilt' : i < n := (UScalar.lt_equiv i n).2 hilt
    rw [if_pos hilt', if_pos trivial]
    rfl
  · rw [hpushv, List.map_append, hout]
    rw [hi2v]
    have hmap : List.map (fun y => y.val) [IScalar.cast .I64 normed] =
        [epilogue root row.val[i.val]] := by
      simp only [List.map_cons, List.map_nil]
      rw [hcastval, hnormed_epi]
    have hsucc : (row.val.take (i.val + 1)).map (epilogue root) =
        (row.val.take i.val).map (epilogue root) ++ [epilogue root row.val[i.val]] := by
      rw [List.take_add_one, List.getElem?_eq_getElem hbnd]
      simp only [List.map_append, List.map_cons, List.map_nil, Option.toList_some]
    rw [hmap, hsucc]
  · exact hi2v
  · omega
  · omega

/-- **Epilogue exit at the row end.** With `i.val ≥ n.val` the body returns the accumulated
vector and the flag unchanged. -/
theorem loop1_body_end (row : Slice I64) (n : Usize) (root : I128)
    (out : alloc.vec.Vec I64) (i : Usize) (out_ok : Bool) (hge : ¬ (i.val < n.val)) :
    rms_norm_source.norm.rms_norm_fixed_loop1.body row n root out i out_ok =
      ok (ControlFlow.done (out, out_ok)) := by
  unfold rms_norm_source.norm.rms_norm_fixed_loop1.body
  have hnot : ¬ (i < n) := by
    intro hc; exact hge ((UScalar.lt_equiv i n).1 hc)
  rw [if_neg hnot]

/-- **Epilogue exit on refusal.** With `out_ok = false` and `i.val < n.val` the body returns the
refused pair without computing an epilogue value. -/
theorem loop1_body_exit (row : Slice I64) (n : Usize) (root : I128)
    (out : alloc.vec.Vec I64) (i : Usize) (out_ok : Bool)
    (hilt : i.val < n.val) (hflag : out_ok = false) :
    rms_norm_source.norm.rms_norm_fixed_loop1.body row n root out i out_ok =
      ok (ControlFlow.done (out, out_ok)) := by
  unfold rms_norm_source.norm.rms_norm_fixed_loop1.body
  have hilt' : i < n := (UScalar.lt_equiv i n).2 hilt
  have hflag' : ¬ (out_ok = true) := by rw [hflag]; decide
  rw [if_pos hilt', if_neg hflag', hflag]

/-- The reachable invariant of the production epilogue loop: flag, index bound and the pushed
prefix of epilogue values. -/
def loop1Inv (row : Slice I64) (n : Usize) (root : I128)
    (s : alloc.vec.Vec I64 × Usize × Bool) : Prop :=
  s.2.2 = true ∧ s.2.1.val ≤ n.val ∧
    s.1.val.map (fun y => y.val) = (row.val.take s.2.1.val).map (epilogue root)

/-- **Epilogue loop resource spec.** Under the product and output-window premises the value-path
loop performs exactly `(Slice.len row).val` body entries and appends the epilogue value of every
coordinate. -/
theorem loop1_work (row : Slice I64) (root : I128)
    (hprod : ∀ (j : ℕ) (hj : j < row.val.length),
        I128.min ≤ row.val[j].val * root.val ∧ row.val[j].val * root.val ≤ I128.max ∧
          I128.min ≤ row.val[j].val * root.val + 2 ^ (31 : ℕ) ∧
            row.val[j].val * root.val + 2 ^ (31 : ℕ) ≤ I128.max)
    (hfit : ∀ (j : ℕ) (hj : j < row.val.length),
        -((2 : ℤ) ^ (63 : ℕ)) ≤ epilogue root row.val[j] ∧
          epilogue root row.val[j] ≤ (2 : ℤ) ^ (63 : ℕ) - 1) :
    ∃ out : alloc.vec.Vec I64,
      rms_norm_source.norm.rms_norm_fixed_loop1 row (Slice.len row) root
        (alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)) 0#usize true = ok (out, true) ∧
      out.val.map (fun y => y.val) = (row.val.map (epilogue root)) := by
  have hloop : rms_norm_source.norm.rms_norm_fixed_loop1 row (Slice.len row) root
      (alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)) 0#usize true ⦃ b =>
        ∃ out : alloc.vec.Vec I64, b = (out, true) ∧
          out.val.map (fun y => y.val) = (row.val.map (epilogue root)) ⦄ := by
    unfold rms_norm_source.norm.rms_norm_fixed_loop1
    apply loop.spec_decr_nat
      (measure := fun s : alloc.vec.Vec I64 × Usize × Bool =>
        (Slice.len row).val - s.2.1.val)
      (inv := loop1Inv row (Slice.len row) root)
      (post := fun b : alloc.vec.Vec I64 × Bool =>
        ∃ out : alloc.vec.Vec I64, b = (out, true) ∧
          out.val.map (fun y => y.val) = (row.val.map (epilogue root)))
    · intro s hs
      rcases s with ⟨out, i, out_ok⟩
      dsimp only at hs ⊢
      obtain ⟨hok, hile, hout⟩ := hs
      dsimp only at hok hile hout
      rw [hok]
      by_cases hlt : i.val < (Slice.len row).val
      · obtain ⟨out2, i2, hbody, hout2, hi2, hrank1, hrank2⟩ :=
          loop1_body_step row (Slice.len row) root out i hlt (by rw [Slice.len_val])
            hout hprod hfit
        rw [hbody]
        have hi2le : i2.val ≤ (Slice.len row).val := by omega
        refine (WP.spec_ok (ControlFlow.cont (out2, i2, true))).mpr ?_
        dsimp only
        have hdrop : (Slice.len row).val - i2.val < (Slice.len row).val - i.val := by omega
        exact ⟨⟨rfl, hi2le, hout2⟩, hdrop⟩
      · have hnot : ¬ (i < Slice.len row) := by
          intro hc; exact hlt ((UScalar.lt_equiv i (Slice.len row)).1 hc)
        have hie : i.val = (Slice.len row).val := by omega
        have hlen : (Slice.len row).val = row.val.length := Slice.len_val row
        have hbody : rms_norm_source.norm.rms_norm_fixed_loop1.body row (Slice.len row) root out i true
            = ok (ControlFlow.done (out, true)) := by
          refine loop1_body_end row (Slice.len row) root out i true ?_
          intro hc; exact hlt ((UScalar.lt_equiv i (Slice.len row)).1 hc)
        rw [hbody]
        exact ⟨out, rfl, by rw [hout, hie, hlen, List.take_length]⟩
    · refine ⟨rfl, ?_, ?_⟩
      · simp
      · show (alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)).val.map (fun y => y.val) =
          (row.val.take (0#usize).val).map (epilogue root)
        simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val]
  obtain ⟨y, hy, out, hyS, hS⟩ := WP.spec_imp_exists hloop
  exact ⟨out, by rw [hy, hyS], hS⟩
/-! ## The `fixed_rsqrt` call: the seed reduction and the root extraction

`isqrt_loop0` divides the seed `2^126 = 4^63` by four per entry until it is the least power of
four above the radicand; `isqrt_loop1` visits one weight `4^j` per entry. -/

/-- `4^k / 4 = 4^(k-1)` and `4^k / 4 * 4 = 4^k` for `1 ≤ k`, the exact seed shift. -/
theorem four_ediv (k : ℕ) (hk : 1 ≤ k) :
    (4 : ℤ) ^ k / 4 = (4 : ℤ) ^ (k - 1) ∧ (4 : ℤ) ^ k / 4 * 4 = (4 : ℤ) ^ k := by
  have hne : (4 : ℤ) ≠ 0 := by norm_num
  have hpow : (4 : ℤ) ^ k = (4 : ℤ) ^ (k - 1) * 4 := by
    rw [← pow_succ]; congr 1; omega
  have h1 : (4 : ℤ) ^ k / 4 = (4 : ℤ) ^ (k - 1) := by
    rw [hpow, Int.mul_comm, Int.mul_ediv_cancel_left _ hne]
  refine ⟨h1, ?_⟩
  rw [h1, ← pow_succ]
  congr 1
  omega

/-- The seed-reduction invariant: the seed is `4^(63-t)` after `t` shifts, and either `t = 0` or
the radicand already lies below four times the seed. -/
def isqrt0Inv (n : I128) (b : I128) : Prop :=
  0 ≤ n.val ∧ ∃ t : Nat, t ≤ 63 ∧ b.val = (4 : ℤ) ^ (63 - t) ∧ (t = 0 ∨ n.val < 4 * b.val)

/-- **Seed-reduction work count.** Each entry of `isqrt_loop0` divides the seed by four, so from
`2^126` the loop takes `t ≤ 63` entries and returns the least power of four above the radicand:
`4^(63-t) ≤ n.val < 4 · 4^(63-t)`. -/
theorem isqrt_loop0_work (n : I128) (b : I128)
    (hn : 1 ≤ n.val) (hhi : n.val ≤ (2 : ℤ) ^ (126 : ℕ)) (hb : b.val = (2 : ℤ) ^ (126 : ℕ)) :
    ∃ t : Nat, ∃ b' : I128, t ≤ 63 ∧
      rms_norm_source.fixed_rsqrt.isqrt_loop0 n b = ok b' ∧
      b'.val = (4 : ℤ) ^ (63 - t) ∧ b'.val ≤ n.val ∧ n.val < 4 * b'.val := by
  have h126 : (4 : ℤ) ^ (63 : ℕ) = (2 : ℤ) ^ (126 : ℕ) := by norm_num
  have hloop : rms_norm_source.fixed_rsqrt.isqrt_loop0 n b ⦃ b' =>
      ∃ t : Nat, t ≤ 63 ∧ b'.val = (4 : ℤ) ^ (63 - t) ∧ b'.val ≤ n.val ∧
        n.val < 4 * b'.val ⦄ := by
    unfold rms_norm_source.fixed_rsqrt.isqrt_loop0
    apply loop.spec_decr_nat
      (measure := fun b' : I128 => b'.val.toNat)
      (inv := isqrt0Inv n)
      (post := fun b' : I128 => ∃ t : Nat, t ≤ 63 ∧ b'.val = (4 : ℤ) ^ (63 - t) ∧
        b'.val ≤ n.val ∧ n.val < 4 * b'.val)
    · intro b' hb'
      obtain ⟨hn0, t, ht, hbt, hrel⟩ := hb'
      by_cases hgt : b'.val > n.val
      · have hlt' : n < b' := (IScalar.lt_equiv n b').1 hgt
        have hbpos : 0 < b'.val := by rw [hbt]; positivity
        have hk : 1 ≤ 63 - t := by
          by_contra hz
          have hz' : 63 - t = 0 := by omega
          have hle : b'.val ≤ 1 := by rw [hbt, hz']; norm_num
          omega
        obtain ⟨c, hc, hcv⟩ := I128_shiftRight_2_val b'
        obtain ⟨hA, hB⟩ := four_ediv (63 - t) hk
        have hinv : isqrt0Inv n c := by
          refine ⟨hn0, t + 1, by omega, ?_, Or.inr ?_⟩
          · rw [hcv, hbt, hA]
            congr 1
          · rw [hcv, hbt, Int.mul_comm, hB]
            omega
        have hrank : c.val.toNat < b'.val.toNat := by
          rw [Int.toNat_lt_toNat hbpos, hcv, hbt, hA]
          have hp : (4 : ℤ) ^ (63 - t - 1) * 4 = (4 : ℤ) ^ (63 - t) := by
            rw [← pow_succ]; congr 1; omega
          have hnonneg : 0 ≤ (4 : ℤ) ^ (63 - t - 1) := by positivity
          nlinarith
        rw [rms_norm_source.fixed_rsqrt.isqrt_loop0.body, if_pos hlt', hc]
        simp only [bind_tc_ok]
        refine (WP.spec_ok (ControlFlow.cont c)).mpr ?_
        dsimp only
        exact ⟨hinv, hrank⟩
      · have hnot : ¬ (b' > n) := by
          intro hc
          have hlt := (IScalar.lt_equiv n b').1 (by omega : n < b')
          omega
        rw [rms_norm_source.fixed_rsqrt.isqrt_loop0.body, if_neg hnot]
        refine ⟨t, ht, hbt, le_of_not_gt hgt, ?_⟩
        rcases hrel with hz | hrel4
        · rw [hbt, hz, h126]
          omega
        · exact hrel4
    · refine ⟨by omega, 0, by norm_num, ?_, Or.inl rfl⟩
      rw [hb, h126]
  obtain ⟨y, hy, t, ht, hbt, hle, hlt⟩ := WP.spec_imp_exists hloop
  exact ⟨t, y, ht, hy, hbt, hle, hlt⟩

/-- The root-extraction invariant: the accumulated root stays in the low `i128` window and the
weight is `4^(63-t)` or zero. -/
def isqrt1Inv (r s b : I128) : Prop :=
  0 ≤ r.val ∧ r.val ≤ (2 : ℤ) ^ (116 : ℕ) ∧
    0 ≤ s.val ∧ s.val ≤ (2 : ℤ) ^ (118 : ℕ) ∧
    (b.val = 0 ∨ ∃ t : Nat, t ≤ 63 ∧ b.val = (4 : ℤ) ^ (63 - t) ∧
      b.val ≤ (2 : ℤ) ^ (116 : ℕ))

/-- **Root-extraction work bound.** `isqrt_loop1` visits one power-of-four weight per entry; with
a remainder at most `2^116`, a starting root at most `2^118` and a weight at most `2^116` every
entry leaves the accumulator at most halved plus the weight, and the returned root stays inside
the `i128` window. -/
theorem isqrt_loop1_work (r s b : I128)
    (hbpow : b.val = 0 ∨ ∃ t : Nat, t ≤ 63 ∧ b.val = (4 : ℤ) ^ (63 - t) ∧
      b.val ≤ (2 : ℤ) ^ (116 : ℕ))
    (hs0 : 0 ≤ s.val) (hsle : s.val ≤ (2 : ℤ) ^ (118 : ℕ))
    (hr0 : 0 ≤ r.val) (hrle : r.val ≤ (2 : ℤ) ^ (116 : ℕ)) :
    ∃ out : I128, rms_norm_source.fixed_rsqrt.isqrt_loop1 r s b = ok out ∧
      0 ≤ out.val ∧ out.val ≤ (2 : ℤ) ^ (118 : ℕ) := by
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) :=
    by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 :=
    by norm_num [I128.max, IScalar.max, I128.numBits]
  have hloop : rms_norm_source.fixed_rsqrt.isqrt_loop1 r s b ⦃ out =>
      0 ≤ out.val ∧ out.val ≤ (2 : ℤ) ^ (118 : ℕ) ⦄ := by
    unfold rms_norm_source.fixed_rsqrt.isqrt_loop1
    apply loop.spec_decr_nat
      (measure := fun st : I128 × I128 × I128 => st.2.2.val.toNat)
      (inv := fun st : I128 × I128 × I128 => isqrt1Inv st.1 st.2.1 st.2.2)
      (post := fun out : I128 => 0 ≤ out.val ∧ out.val ≤ (2 : ℤ) ^ (118 : ℕ))
    · intro st hst
      rcases st with ⟨r', s', b'⟩
      dsimp only at hst ⊢
      obtain ⟨hr0', hrle', hs0', hsle', hb'⟩ := hst
      by_cases hb0 : b'.val = 0
      · have hzero : b' = 0#i128 := IScalar.eq_of_val_eq hb0
        have hne : ¬ (b' != 0#i128) := by rw [hzero]; exact fun h => Bool.noConfusion h
        rw [rms_norm_source.fixed_rsqrt.isqrt_loop1.body, if_neg hne]
        refine (WP.spec_ok (ControlFlow.done s')).mpr ?_
        exact ⟨hs0', hsle'⟩
      · have hnz : b' ≠ 0#i128 := fun hc => hb0 (congrArg IScalar.val hc)
        have hne : (b' != 0#i128) = true := by simpa using hnz
        rw [rms_norm_source.fixed_rsqrt.isqrt_loop1.body, if_pos hne]
        have hrest : ∃ t : Nat, t ≤ 63 ∧ b'.val = (4 : ℤ) ^ (63 - t) ∧
            b'.val ≤ (2 : ℤ) ^ (116 : ℕ) := by
          rcases hb' with hz | hrest
          · exact absurd hz hb0
          · exact hrest
        obtain ⟨t, ht, hbt, hbhi⟩ := hrest
        have hbpos : 0 < b'.val := by rw [hbt]; positivity
        have hstepmin : I128.min ≤ s'.val + b'.val := by
          rw [hI128min]
          have hsum0 : 0 ≤ s'.val + b'.val := by omega
          omega
        have hfitnum : (2 : ℤ) ^ (118 : ℕ) + (2 : ℤ) ^ (116 : ℕ) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by
          norm_num
        have hstepmax : s'.val + b'.val ≤ I128.max := by
          rw [hI128max]
          omega
        obtain ⟨step, hstep, hstepval⟩ :=
          WP.spec_imp_exists (I128.add_spec (x := s') (y := b') hstepmin hstepmax)
        rw [hstep]; simp only [bind_tc_ok]
        obtain ⟨half, hhalf_eq, hhalfval⟩ := I128_shiftRight_1_val s'
        rw [hhalf_eq]; simp only [bind_tc_ok]
        have hhalf0 : 0 ≤ half.val := by
          rw [hhalfval]; exact Int.ediv_nonneg hs0' (by norm_num)
        have hhalfle : half.val ≤ (2 : ℤ) ^ (117 : ℕ) := by
          rw [hhalfval]
          have h := Int.ediv_le_ediv (c := 2) (by norm_num : (0 : ℤ) < 2) hsle'
          norm_num at h
          exact h
        obtain ⟨b2, hb2, hb2val⟩ := I128_shiftRight_2_val b'
        have hpow : b2.val = 0 ∨ ∃ t' : Nat, t' ≤ 63 ∧ b2.val = (4 : ℤ) ^ (63 - t') ∧
            b2.val ≤ (2 : ℤ) ^ (116 : ℕ) := by
          by_cases ht63 : t = 63
          · refine Or.inl ?_
            rw [ht63] at hbt
            rw [hb2val, hbt]
            norm_num
          · have hk : 1 ≤ 63 - t := by omega
            obtain ⟨hA, hB⟩ := four_ediv (63 - t) hk
            refine Or.inr ⟨t + 1, by omega, ?_, ?_⟩
            · rw [hb2val, hbt, hA]
              congr 1
            · rw [hb2val]
              have hediv := Int.ediv_le_self 4 (le_of_lt hbpos)
              omega
        have hdrop : b2.val.toNat < b'.val.toNat := by
          by_cases ht63 : t = 63
          · rw [ht63] at hbt
            rw [Int.toNat_lt_toNat (by rw [hbt]; positivity), hb2val, hbt]
            have hz : (4 : ℤ) ^ (0 : ℕ) / 4 = 0 := by norm_num
            rw [hz]
            norm_num
          · have hk : 1 ≤ 63 - t := by omega
            obtain ⟨hA, hB⟩ := four_ediv (63 - t) hk
            rw [Int.toNat_lt_toNat (by rw [hbt]; positivity), hb2val, hbt, hA]
            have hp : (4 : ℤ) ^ (63 - t - 1) * 4 = (4 : ℤ) ^ (63 - t) := by
              rw [← pow_succ]; congr 1; omega
            have hnonneg : 0 ≤ (4 : ℤ) ^ (63 - t - 1) := by positivity
            nlinarith
        by_cases hge : step ≤ r'
        · have hgeval : step.val ≤ r'.val := (IScalar.le_equiv step r').mp hge
          have hsubmin : I128.min ≤ r'.val - step.val := by
            rw [hI128min]
            have hdif : 0 ≤ r'.val - step.val := by omega
            omega
          have hsubmax : r'.val - step.val ≤ I128.max := by
            rw [hI128max]
            have hle : r'.val - step.val ≤ r'.val := by
              have hstep0 : 0 ≤ step.val := by
                rw [hstepval, hbt]; positivity
              omega
            have h1 : (2 : ℤ) ^ (116 : ℕ) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by norm_num
            omega
          obtain ⟨rem2, hrem2, hrem2val⟩ :=
            WP.spec_imp_exists (I128.sub_spec (x := r') (y := step) hsubmin hsubmax)
          rw [hrem2]; simp only [bind_tc_ok]
          have hres2min : I128.min ≤ half.val + b'.val := by
            rw [hI128min]
            have hsum0 : 0 ≤ half.val + b'.val := by omega
            omega
          have hres2max : half.val + b'.val ≤ I128.max := by
            rw [hI128max]
            have h1 : (2 : ℤ) ^ (117 : ℕ) + (2 : ℤ) ^ (116 : ℕ) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by
              norm_num
            omega
          obtain ⟨res2, hres2, hres2val⟩ :=
            WP.spec_imp_exists (I128.add_spec (x := half) (y := b') hres2min hres2max)
          rw [hres2, if_pos hge]; simp only [bind_tc_ok]
          have hinv : isqrt1Inv rem2 res2 b2 := by
            refine ⟨?_, ?_, ?_, ?_, hpow⟩
            · rw [hrem2val]
              have hdif : 0 ≤ r'.val - step.val := by omega
              exact hdif
            · rw [hrem2val]
              have hle : r'.val - step.val ≤ r'.val := by
                have hstep0 : 0 ≤ step.val := by
                  rw [hstepval, hbt]; positivity
                omega
              omega
            · rw [hres2val]
              have hsum0 : 0 ≤ half.val + b'.val := by omega
              exact hsum0
            · rw [hres2val]
              have hhalfle' : half.val ≤ (2 : ℤ) ^ (118 : ℕ) := by omega
              omega
          rw [hb2]
          simp only [bind_tc_ok]
          refine (WP.spec_ok (ControlFlow.cont (rem2, res2, b2))).mpr ?_
          dsimp only
          exact ⟨hinv, hdrop⟩
        · have hltval : r'.val < step.val := by
            by_contra hc
            exact hge ((IScalar.le_equiv step r').mpr (le_of_not_gt hc))
          have hnotge : ¬ (step ≤ r') := by
            intro hc
            have hle := (IScalar.le_equiv step r').mp hc
            omega
          rw [if_neg hnotge]
          simp only [bind_tc_ok]
          have hhalfle' : half.val ≤ (2 : ℤ) ^ (118 : ℕ) := by omega
          have hinv : isqrt1Inv r' half b2 :=
            ⟨hr0', hrle', hhalf0, hhalfle', hpow⟩
          rw [hb2]
          simp only [bind_tc_ok]
          refine (WP.spec_ok (ControlFlow.cont (r', half, b2))).mpr ?_
          dsimp only
          exact ⟨hinv, hdrop⟩
    · exact ⟨hr0, hrle, hs0, hsle, hbpow⟩
  obtain ⟨y, hy, h0, h1⟩ := WP.spec_imp_exists hloop
  exact ⟨y, hy, h0, h1⟩

/-- **One `fixed_rsqrt` call.** Over a radicand inside the caller's window the call performs one
seed reduction of at most `63` entries, whose result is the least power of four above the
radicand and is at most `2^116`, followed by one root extraction over that power of four whose
result is at most `2^118`; those two loops are the only loops on this path. -/
theorem isqrt_work (n s : I128)
    (hn : 1 ≤ n.val) (hhi : n.val ≤ (2 : ℤ) ^ (116 : ℕ))
    (hs : rms_norm_source.fixed_rsqrt.isqrt n = ok s) :
    ∃ t : Nat, ∃ b' root : I128,
      t ≤ 63 ∧ b'.val = (4 : ℤ) ^ (63 - t) ∧ b'.val ≤ n.val ∧ n.val < 4 * b'.val ∧
      b'.val ≤ (2 : ℤ) ^ (116 : ℕ) ∧
      rms_norm_source.fixed_rsqrt.isqrt_loop1 n 0#i128 b' = ok root ∧
      0 ≤ root.val ∧ root.val ≤ (2 : ℤ) ^ (118 : ℕ) ∧ s = root := by
  unfold rms_norm_source.fixed_rsqrt.isqrt at hs
  rw [I128_shiftLeft_126] at hs
  simp only [bind_tc_ok] at hs
  obtain ⟨t, b', ht, hloop0, hbval, hble, hbgt⟩ :=
    isqrt_loop0_work n isqrtSeed hn (hhi.trans (by norm_num)) isqrtSeed_val
  rw [hloop0] at hs
  simp only [bind_tc_ok] at hs
  have hbpow : b'.val = 0 ∨ ∃ t : Nat, t ≤ 63 ∧ b'.val = (4 : ℤ) ^ (63 - t) ∧
      b'.val ≤ (2 : ℤ) ^ (116 : ℕ) :=
    Or.inr ⟨t, ht, hbval, by omega⟩
  obtain ⟨root, hloop1, hroot0, hroothi⟩ :=
    isqrt_loop1_work n 0#i128 b' hbpow (by norm_num) (by norm_num) (by omega) hhi
  rw [hloop1] at hs
  have heq : root = s := Result.ok_injective hs
  exact ⟨t, b', root, ht, hbval, hble, hbgt, by omega, hloop1, hroot0, hroothi, heq.symm⟩

/-! ## The whole call: branch results, the `fixed_rsqrt` call and the work decomposition -/

theorem slice_len_ne_zero_of_ne_nil (row : Slice I64) (h : row.val ≠ []) :
    Slice.len row ≠ 0#usize := by
  intro hc
  apply h
  have hv : row.val.length = 0 := by
    have := congrArg UScalar.val hc
    simpa [Slice.len, UScalar.ofNatCore_val_eq] using this
  exact List.length_eq_zero_iff.mp hv

/-- **Empty-row branch.** An empty row is refused with `EmptyRow` before any loop entry. -/
theorem empty_row_refuses (row : Slice I64) (eps : I64) (h : row.val = []) :
    rms_norm_source.word_rms_norm_fixed row eps =
      ok (core.result.Result.Err rms_norm_source.norm.NormError.EmptyRow) := by
  have hn : Slice.len row = 0#usize := by
    apply UScalar.eq_of_val_eq
    simp [Slice.len, h, UScalar.ofNatCore_val_eq]
  rw [rms_norm_source.word_rms_norm_fixed, rms_norm_source.norm.rms_norm_fixed, hn]
  simp

/-- The extracted body after the empty-row test and a successful accumulation loop, as a `do`
block: every remaining arm is read off it. -/
theorem rms_unfold (row : Slice I64) (eps : I64) (sum_sq : I128) (h : row.val ≠ [])
    (hloop : rms_norm_source.norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, true)) :
    rms_norm_source.word_rms_norm_fixed row eps =
      (do
        let n_i128 ← lift (UScalar.hcast .I128 (Slice.len row))
        let i ← 1#i128 <<< rms_norm_source.FIXED_FRAC_BITS
        let o ← lift (I128.checked_mul n_i128 i)
        match o with
        | none => ok (core.result.Result.Err rms_norm_source.norm.NormError.OutOfRange)
        | some den =>
          let mean ← rms_norm_source.value.div_round_nearest sum_sq den
          let i1 ← lift (core.convert.num.FromI128I64.from eps)
          let o1 ← lift (mean.checked_add i1)
          match o1 with
          | none => ok (core.result.Result.Err rms_norm_source.norm.NormError.OutOfRange)
          | some arg =>
            let i2 ← lift (IScalar.cast .I128 core.num.I64.MAX)
            if arg > i2
            then ok (core.result.Result.Err rms_norm_source.norm.NormError.OutOfRange)
            else
              let i3 ← lift (IScalar.cast .I128 core.num.I64.MIN)
              if arg < i3
              then ok (core.result.Result.Err rms_norm_source.norm.NormError.OutOfRange)
              else
                let arg_i64 ← lift (IScalar.cast .I64 arg)
                let r ← rms_norm_source.fixed_rsqrt.fixed_rsqrt arg_i64
                match r with
                | core.result.Result.Ok root =>
                  let root1 ← lift (core.convert.num.FromI128I64.from root)
                  let out := alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)
                  let (out1, out_ok) ←
                    rms_norm_source.norm.rms_norm_fixed_loop1 row (Slice.len row) root1 out 0#usize true
                  if out_ok
                  then ok (core.result.Result.Ok out1)
                  else ok (core.result.Result.Err rms_norm_source.norm.NormError.OutOfRange)
                | core.result.Result.Err _ =>
                  ok (core.result.Result.Err rms_norm_source.norm.NormError.OutOfRange)) := by
  have hn := slice_len_ne_zero_of_ne_nil row h
  rw [rms_norm_source.word_rms_norm_fixed, rms_norm_source.norm.rms_norm_fixed]
  simp only [hn, if_false]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **Accumulation refusal branch.** A failed accumulation refuses the whole call with
`OutOfRange` after exactly the loop entries the loop took. -/
theorem accumulation_refuses (row : Slice I64) (eps : I64) (sum_sq : I128) (h : row.val ≠ [])
    (hloop : rms_norm_source.norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, false)) :
    rms_norm_source.word_rms_norm_fixed row eps =
      ok (core.result.Result.Err rms_norm_source.norm.NormError.OutOfRange) := by
  have hn := slice_len_ne_zero_of_ne_nil row h
  rw [rms_norm_source.word_rms_norm_fixed, rms_norm_source.norm.rms_norm_fixed]
  simp only [hn, if_false]
  rw [hloop]
  simp

/-- **`fixed_rsqrt` refusal branch.** An `Err` from `fixed_rsqrt` — the window test — refuses the
call with `OutOfRange`. -/
theorem rsqrt_call_refuses (row : Slice I64) (eps : I64)
    (sum_sq mean arg den shift : I128) (e : rms_norm_source.fixed_rsqrt.RsqrtError) (arg64 : I64)
    (h : row.val ≠ [])
    (hloop : rms_norm_source.norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, true))
    (hshift : 1#i128 <<< rms_norm_source.FIXED_FRAC_BITS = ok shift)
    (hden : I128.checked_mul (UScalar.hcast .I128 (Slice.len row)) shift = some den)
    (hmean : rms_norm_source.value.div_round_nearest sum_sq den = ok mean)
    (harg : I128.checked_add mean (core.convert.num.FromI128I64.from eps) = some arg)
    (hmax : ¬ (arg > IScalar.cast .I128 core.num.I64.MAX))
    (hmin : ¬ (arg < IScalar.cast .I128 core.num.I64.MIN))
    (hcast : IScalar.cast .I64 arg = arg64)
    (hroot : rms_norm_source.fixed_rsqrt.fixed_rsqrt arg64 =
      ok (core.result.Result.Err e)) :
    rms_norm_source.word_rms_norm_fixed row eps =
      ok (core.result.Result.Err rms_norm_source.norm.NormError.OutOfRange) := by
  rw [rms_unfold row eps sum_sq h hloop]
  simp only [hshift, lift, bind_tc_ok, hden, hmean, harg, hmax, hmin, hcast, hroot, ↓reduceIte]

/-- **Epilogue refusal branch.** A failed narrowing in the value path refuses the call. -/
theorem epilogue_refuses (row : Slice I64) (eps : I64)
    (sum_sq mean arg den shift : I128) (root : I64) (out1 : alloc.vec.Vec I64) (arg64 : I64)
    (h : row.val ≠ [])
    (hloop : rms_norm_source.norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, true))
    (hshift : 1#i128 <<< rms_norm_source.FIXED_FRAC_BITS = ok shift)
    (hden : I128.checked_mul (UScalar.hcast .I128 (Slice.len row)) shift = some den)
    (hmean : rms_norm_source.value.div_round_nearest sum_sq den = ok mean)
    (harg : I128.checked_add mean (core.convert.num.FromI128I64.from eps) = some arg)
    (hmax : ¬ (arg > IScalar.cast .I128 core.num.I64.MAX))
    (hmin : ¬ (arg < IScalar.cast .I128 core.num.I64.MIN))
    (hcast : IScalar.cast .I64 arg = arg64)
    (hroot : rms_norm_source.fixed_rsqrt.fixed_rsqrt arg64 =
      ok (core.result.Result.Ok root))
    (hloop1 : rms_norm_source.norm.rms_norm_fixed_loop1 row (Slice.len row)
        (core.convert.num.FromI128I64.from root)
        (alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)) 0#usize true
        = ok (out1, false)) :
    rms_norm_source.word_rms_norm_fixed row eps =
      ok (core.result.Result.Err rms_norm_source.norm.NormError.OutOfRange) := by
  rw [rms_unfold row eps sum_sq h hloop]
  simp only [hshift, lift, bind_tc_ok, hden, hmean, harg, hmax, hmin, hcast, hroot, hloop1,
    ↓reduceIte]
  rfl

/-- **Success branch.** With `fixed_rsqrt` returning a root and the epilogue loop succeeding, the
call returns the accumulated output vector. -/
theorem success_returns_output (row : Slice I64) (eps : I64)
    (sum_sq mean arg den shift : I128) (root : I64) (out1 : alloc.vec.Vec I64) (arg64 : I64)
    (h : row.val ≠ [])
    (hloop : rms_norm_source.norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, true))
    (hshift : 1#i128 <<< rms_norm_source.FIXED_FRAC_BITS = ok shift)
    (hden : I128.checked_mul (UScalar.hcast .I128 (Slice.len row)) shift = some den)
    (hmean : rms_norm_source.value.div_round_nearest sum_sq den = ok mean)
    (harg : I128.checked_add mean (core.convert.num.FromI128I64.from eps) = some arg)
    (hmax : ¬ (arg > IScalar.cast .I128 core.num.I64.MAX))
    (hmin : ¬ (arg < IScalar.cast .I128 core.num.I64.MIN))
    (hcast : IScalar.cast .I64 arg = arg64)
    (hroot : rms_norm_source.fixed_rsqrt.fixed_rsqrt arg64 =
      ok (core.result.Result.Ok root))
    (hloop1 : rms_norm_source.norm.rms_norm_fixed_loop1 row (Slice.len row)
        (core.convert.num.FromI128I64.from root)
        (alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)) 0#usize true
        = ok (out1, true)) :
    rms_norm_source.word_rms_norm_fixed row eps = ok (core.result.Result.Ok out1) := by
  rw [rms_unfold row eps sum_sq h hloop]
  simp only [hshift, lift, bind_tc_ok, hden, hmean, harg, hmax, hmin, hcast, hroot, hloop1,
    ↓reduceIte]
  rfl

/-- **Whole-call work decomposition.** One call on a nonempty row takes exactly `n` accumulation
body entries — the rank starts at `n = (Slice.len row).val` at index `0` and falls by exactly one
per entry to `0` — accumulates the whole sum of squares, reaches exactly one `fixed_rsqrt` call
with the narrowed argument, and returns the epilogue output. -/
theorem rms_work (row : Slice I64) (eps : I64)
    (sum_sq mean arg den shift : I128) (root : I64) (out1 : alloc.vec.Vec I64)
    (arg64 : I64)
    (h : row.val ≠ [])
    (hfit : (row.val.map (fun x => x.val * x.val)).sum ≤ I128.max)
    (hloop : rms_norm_source.norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, true))
    (hshift : 1#i128 <<< rms_norm_source.FIXED_FRAC_BITS = ok shift)
    (hden : I128.checked_mul (UScalar.hcast .I128 (Slice.len row)) shift = some den)
    (hmean : rms_norm_source.value.div_round_nearest sum_sq den = ok mean)
    (harg : I128.checked_add mean (core.convert.num.FromI128I64.from eps) = some arg)
    (hmax : ¬ (arg > IScalar.cast .I128 core.num.I64.MAX))
    (hmin : ¬ (arg < IScalar.cast .I128 core.num.I64.MIN))
    (hcast : IScalar.cast .I64 arg = arg64)
    (hroot : rms_norm_source.fixed_rsqrt.fixed_rsqrt arg64 =
      ok (core.result.Result.Ok root))
    (hloop1 : rms_norm_source.norm.rms_norm_fixed_loop1 row (Slice.len row)
        (core.convert.num.FromI128I64.from root)
        (alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)) 0#usize true
        = ok (out1, true)) :
    rms_norm_source.word_rms_norm_fixed row eps = ok (core.result.Result.Ok out1) ∧
      sum_sq.val = (row.val.map (fun x => x.val * x.val)).sum ∧
      (Slice.len row).val - (0#usize).val = row.val.length ∧
      (Slice.len row).val - (Slice.len row).val = 0 := by
  obtain ⟨hdrop0, hdrop1, hS⟩ := loop0_work row hfit
  obtain ⟨S, hSrun, hSval⟩ := hS
  have hsumval : sum_sq.val = (row.val.map (fun x => x.val * x.val)).sum := by
    have hEq : S = sum_sq :=
      congrArg Prod.fst (Result.ok_injective (hSrun.symm.trans hloop))
    rw [hEq] at hSval
    exact hSval
  exact ⟨success_returns_output row eps sum_sq mean arg den shift root out1 arg64 h hloop hshift
    hden hmean harg hmax hmin hcast hroot hloop1, hsumval, hdrop0, hdrop1⟩

/-- **Epilogue work decomposition.** Under the product and output-window premises the epilogue loop
takes exactly `(Slice.len row).val` body entries and appends the epilogue value of every
coordinate, so the successful call's output is coordinate-wise `round(x_i · root / 2^32)`. -/
theorem rms_work_outputs (row : Slice I64) (eps : I64)
    (sum_sq mean arg den shift : I128) (root : I64) (out1 : alloc.vec.Vec I64)
    (arg64 : I64)
    (h : row.val ≠ [])
    (hloop : rms_norm_source.norm.rms_norm_fixed_loop0 row (Slice.len row) 0#i128 0#usize true
               = ok (sum_sq, true))
    (hshift : 1#i128 <<< rms_norm_source.FIXED_FRAC_BITS = ok shift)
    (hden : I128.checked_mul (UScalar.hcast .I128 (Slice.len row)) shift = some den)
    (hmean : rms_norm_source.value.div_round_nearest sum_sq den = ok mean)
    (harg : I128.checked_add mean (core.convert.num.FromI128I64.from eps) = some arg)
    (hmax : ¬ (arg > IScalar.cast .I128 core.num.I64.MAX))
    (hmin : ¬ (arg < IScalar.cast .I128 core.num.I64.MIN))
    (hcast : IScalar.cast .I64 arg = arg64)
    (hroot : rms_norm_source.fixed_rsqrt.fixed_rsqrt arg64 =
      ok (core.result.Result.Ok root))
    (hloop1 : rms_norm_source.norm.rms_norm_fixed_loop1 row (Slice.len row)
        (core.convert.num.FromI128I64.from root)
        (alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)) 0#usize true
        = ok (out1, true))
    (hprod : ∀ (j : ℕ) (hj : j < row.val.length),
        I128.min ≤ row.val[j].val * (core.convert.num.FromI128I64.from root).val ∧
          row.val[j].val * (core.convert.num.FromI128I64.from root).val ≤ I128.max ∧
          I128.min ≤ row.val[j].val * (core.convert.num.FromI128I64.from root).val + 2 ^ (31 : ℕ) ∧
            row.val[j].val * (core.convert.num.FromI128I64.from root).val + 2 ^ (31 : ℕ) ≤ I128.max)
    (houtfit : ∀ (j : ℕ) (hj : j < row.val.length),
        -((2 : ℤ) ^ (63 : ℕ)) ≤ epilogue (core.convert.num.FromI128I64.from root) row.val[j] ∧
          epilogue (core.convert.num.FromI128I64.from root) row.val[j] ≤ (2 : ℤ) ^ (63 : ℕ) - 1) :
    rms_norm_source.word_rms_norm_fixed row eps = ok (core.result.Result.Ok out1) ∧
      out1.val.map (fun y => y.val) =
        (row.val.map (epilogue (core.convert.num.FromI128I64.from root))) := by
  obtain ⟨out, hrun, hout⟩ := loop1_work row (core.convert.num.FromI128I64.from root) hprod houtfit
  have heq : out = out1 := congrArg Prod.fst (Result.ok_injective (hrun.symm.trans hloop1))
  have hloop1' : rms_norm_source.norm.rms_norm_fixed_loop1 row (Slice.len row)
        (core.convert.num.FromI128I64.from root)
        (alloc.vec.Vec.with_capacity Std.I64 (Slice.len row)) 0#usize true
        = ok (out1, true) := by rw [← heq]; exact hrun
  refine ⟨?_, ?_⟩
  · exact success_returns_output row eps sum_sq mean arg den shift root out1 arg64 h hloop hshift
      hden hmean harg hmax hmin hcast hroot hloop1'
  · rw [← heq]; exact hout

/-! ## Reachable success witnesses -/

/-- The concrete row `[1, 0]`, of length two, whose sum of squares is `1`. -/
def witnessRow : Slice I64 := Slice.from (1#i64 :: 0#i64 :: []) (by grind)

theorem witnessRow_val : witnessRow.val = [1#i64, 0#i64] := by
  rw [witnessRow, Slice.from_val]

/-- **Reachable accumulation success.** On the concrete row `[1, 0]` the accumulation loop returns
success with accumulator `1 = 1^2 + 0^2`, after exactly two body entries. -/
theorem accumulation_witness :
    ∃ S : I128, rms_norm_source.norm.rms_norm_fixed_loop0 witnessRow
        (Slice.len witnessRow) 0#i128 0#usize true = ok (S, true) ∧ S.val = 1 := by
  have hsum : (witnessRow.val.map (fun x => x.val * x.val)).sum = 1 := by
    rw [witnessRow_val]; norm_num
  have hfit : (witnessRow.val.map (fun x => x.val * x.val)).sum ≤ I128.max := by
    rw [hsum]; norm_num [I128.max, IScalar.max, I128.numBits]
  obtain ⟨h0, h1, hS⟩ := loop0_work witnessRow hfit
  obtain ⟨S, hrun, hSv⟩ := hS
  exact ⟨S, hrun, by rw [hSv, hsum]⟩

/-- The Q32.32 unit `2^32` as an `I128` value, the root of the identity epilogue. -/
def oneRoot : I128 :=
  IScalar.ofIntCore ((2 : ℤ) ^ (32 : ℕ)) (by norm_num [IScalarTy.numBits])

theorem oneRoot_val : oneRoot.val = (2 : ℤ) ^ (32 : ℕ) := by
  rw [oneRoot, IScalar.ofInt_val_eq]

/-- The epilogue of the unit root is the coordinate itself. -/
theorem epilogue_at_one :
    epilogue oneRoot (1#i64) = 1 ∧ epilogue oneRoot (0#i64) = 0 := by
  have hval : (1#i64).val = 1 ∧ (0#i64).val = 0 := ⟨by norm_num, by norm_num⟩
  constructor
  · rw [epilogue, oneRoot_val, divRoundNearest, hval.1]
    norm_num
  · rw [epilogue, oneRoot_val, divRoundNearest, hval.2]
    norm_num

/-- **Reachable epilogue success.** On the concrete row `[1, 0]` with the unit root `2^32` the
epilogue loop returns success after exactly two body entries and appends one coordinate per
entry. The two window premises are the stated `i128` product and `i64` output windows; for this
row and root both hold by `epilogue_at_one` together with `witnessRow_val`. -/
theorem epilogue_witness
    (hprod : ∀ (j : ℕ) (hj : j < witnessRow.val.length),
        I128.min ≤ witnessRow.val[j].val * oneRoot.val ∧
          witnessRow.val[j].val * oneRoot.val ≤ I128.max ∧
          I128.min ≤ witnessRow.val[j].val * oneRoot.val + 2 ^ (31 : ℕ) ∧
            witnessRow.val[j].val * oneRoot.val + 2 ^ (31 : ℕ) ≤ I128.max)
    (houtfit : ∀ (j : ℕ) (hj : j < witnessRow.val.length),
        -((2 : ℤ) ^ (63 : ℕ)) ≤ epilogue oneRoot witnessRow.val[j] ∧
          epilogue oneRoot witnessRow.val[j] ≤ (2 : ℤ) ^ (63 : ℕ) - 1) :
    ∃ out : alloc.vec.Vec I64,
      rms_norm_source.norm.rms_norm_fixed_loop1 witnessRow (Slice.len witnessRow) oneRoot
        (alloc.vec.Vec.with_capacity Std.I64 (Slice.len witnessRow)) 0#usize true
        = ok (out, true) ∧
      out.val.map (fun y => y.val) = (witnessRow.val.map (epilogue oneRoot)) ∧
      out.val.length = 2 := by
  obtain ⟨out, hrun, hout⟩ := loop1_work witnessRow oneRoot hprod houtfit
  refine ⟨out, hrun, hout, ?_⟩
  rw [← List.length_map, hout, witnessRow_val, List.length_map]
  rfl

end RmsNormResource
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

#resource_audit RmsNormResource.loop0_body_step
#resource_audit RmsNormResource.loop0_body_end
#resource_audit RmsNormResource.loop0_body_exit
#resource_audit RmsNormResource.loop0_body_refused
#resource_audit RmsNormResource.loop0_work
#resource_audit RmsNormResource.loop1_body_step
#resource_audit RmsNormResource.loop1_body_end
#resource_audit RmsNormResource.loop1_body_exit
#resource_audit RmsNormResource.loop1_work
#resource_audit RmsNormResource.isqrt_loop0_work
#resource_audit RmsNormResource.isqrt_loop1_work
#resource_audit RmsNormResource.isqrt_work
#resource_audit RmsNormResource.empty_row_refuses
#resource_audit RmsNormResource.accumulation_refuses
#resource_audit RmsNormResource.rsqrt_call_refuses
#resource_audit RmsNormResource.epilogue_refuses
#resource_audit RmsNormResource.success_returns_output
#resource_audit RmsNormResource.rms_work
#resource_audit RmsNormResource.rms_work_outputs
#resource_audit RmsNormResource.accumulation_witness
#resource_audit RmsNormResource.epilogue_witness

run_cmd Lean.logInfo "RMS_NORM_KERNEL_OK"
