/-
Integer, index and layout refinement of the extracted `kern::gemm` block loop
(`pack_group_loop0_loop1`), the sixth group of the `kern::gemm` refinement.
`Serve/Kern/R4/Gemm5.lean` refined the four-code quad packer
`pack_group_loop0_loop1_loop0`; this module refines its enclosing block loop.

Serving path: `Engine::next_token_state` -> `step::cols` -> `kern::gemm::pack_group` ->
`pack_group_loop0` (column loop) -> `pack_group_loop0_loop1` (block loop) ->
`pack_group_loop0_loop1_loop0` (quad packer, `Gemm5`).

For each block `b < nb` the body reads `s = scales[b]`, computes
`sh = if s = 0 then 0 else s.trailing_zeros() - lo` (a checked `u32` subtraction), writes the
widening `(i64) sh` at `shift[b*8 + lane]`, folds `group_spread = max group_spread sh`, then runs
the eight-quad packer of `Gemm5` on block `b` into `words`.

The reference is plain Lean over `Nat`/`List`: a `List.set` fold of writes (`gemm6_setShifts`), a
`Nat` maximum fold (`gemm6_spread`) and the block packer fold (`gemm6_packBlocks`). It states WHAT
the loop writes. The loop refuses only through `Slice.index_usize` / `Slice.update` out of bounds,
the checked `usize` `*`/`+` and the checked `u32` `-`; the premise bundle excludes all of them, and
`hlo` names the `u32` underflow condition the caller (`pack_group_loop0`) owes.

NOT refined here, stated rather than skirted:

* `chart.q8act_fast_bound_ok`: not a plain total body -- it calls `chart.q8act_fast_bound_ok_loop`,
  whose value over `core.num.I128.abs` / `saturating_mul` is a separate refinement. Its Goal-1 value
  lemma is open, not guessed.
* `kern.gemm.fast_columns_ok` (Goal 3): its body calls `chart.QuantAct.len`,
  `chart.QuantAct.impl.scales`, `chart.QuantAct.magnitude` and `chart.q8act_fast_bound_ok`. The
  accessors are refined here (Goal 1); `chart.QuantAct.magnitude` additionally needs the value of
  `chart.QuantAct.magnitude_loop` (`I64.checked_mul` / `I128.checked_add` overflow arms and
  `chart.ActMagnitude.{Exact,NoQ32,TotalOverflow}`), and `fast_columns_ok` has its own bodies. Open.

Trust boundary: rustc/LLVM codegen, the `F32`/AVX2 layers, the `axiom` pool driver and the
`R[x]/(x^2 - sigma)` word kernels are not covered here. No `sorry`/`admit`/`axiom`/`native_decide`/
`bv_decide`; `decide` appears only on literal lists and literal integer bounds.
-/
import Serve.Kern.R4.Gemm5

set_option autoImplicit false
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## `chart.QuantAct` accessors (Goal 1)

The three accessors are total: each is a single `ok` of a `Vec` projection, so the value lemma is
`rfl` and the `step` form is the equation plus `WP.spec_ok`. -/

/-- `chart.QuantAct.impl.scales` returns the scale slice. -/
theorem gemm6_quantact_impl_scales_eq (q : chart.QuantAct) :
    chart.QuantAct.impl.scales q = ok (alloc.vec.Vec.deref q.scales) := rfl

/-- `chart.QuantAct.impl.codes` returns the code slice. -/
theorem gemm6_quantact_impl_codes_eq (q : chart.QuantAct) :
    chart.QuantAct.impl.codes q = ok (alloc.vec.Vec.deref q.codes) := rfl

/-- `chart.QuantAct.len` returns the code-count. -/
theorem gemm6_quantact_len_eq (q : chart.QuantAct) :
    chart.QuantAct.len q = ok (alloc.vec.Vec.len q.codes) := rfl

/-- `chart.QuantAct.magnitude` is the loop `magnitude_loop` from `(0, 0)`. -/
theorem gemm6_quantact_magnitude_eq (q : chart.QuantAct) :
    chart.QuantAct.magnitude q = chart.QuantAct.magnitude_loop q 0#i128 0#usize := rfl

/-- Step rule for `chart.QuantAct.impl.scales`, for the `pack_group_loop0` walk. -/
@[step]
theorem gemm6_quantact_impl_scales_spec (q : chart.QuantAct) :
    chart.QuantAct.impl.scales q ⦃ s => s = alloc.vec.Vec.deref q.scales ⦄ := by
  rw [gemm6_quantact_impl_scales_eq]; simp only [WP.spec_ok]

/-- Step rule for `chart.QuantAct.impl.codes`, for the `pack_group_loop0` walk. -/
@[step]
theorem gemm6_quantact_impl_codes_spec (q : chart.QuantAct) :
    chart.QuantAct.impl.codes q ⦃ s => s = alloc.vec.Vec.deref q.codes ⦄ := by
  rw [gemm6_quantact_impl_codes_eq]; simp only [WP.spec_ok]

/-- Step rule for `chart.QuantAct.len`. -/
@[step]
theorem gemm6_quantact_len_spec (q : chart.QuantAct) :
    chart.QuantAct.len q ⦃ n => n = alloc.vec.Vec.len q.codes ⦄ := by
  rw [gemm6_quantact_len_eq]; simp only [WP.spec_ok]

/-! ## Local value lemmas for the model groups the body calls -/

/-- `numGap_tzAux` never advances past `acc + fuel`. -/
theorem gemm6_tzAux_le {w : Nat} (x : BitVec w) (fuel acc : Nat) :
    numGap_trailingZeros.numGap_tzAux x fuel acc ≤ acc + fuel := by
  induction fuel generalizing acc with
  | zero => simp only [numGap_trailingZeros.numGap_tzAux]; omega
  | succ f ih =>
    simp only [numGap_trailingZeros.numGap_tzAux]
    split
    · omega
    · exact Nat.le_trans (ih (acc + 1)) (by omega)

/-- `i64::trailing_zeros` (the model) is at most the word width. -/
theorem gemm6_trailingZeros_le {w : Nat} (x : BitVec w) : numGap_trailingZeros x ≤ w := by
  rw [numGap_trailingZeros]
  exact Nat.le_trans (gemm6_tzAux_le x w 0) (by omega)

/-- The model `trailing_zeros` of an `i64` fits a `u32`. -/
theorem gemm6_trailingZeros_lt32 (x : Std.I64) : numGap_trailingZeros x.bv < 2 ^ 32 := by
  have h := gemm6_trailingZeros_le x.bv
  omega

/-- The `u32` literal `⟨BitVec.ofNat _ n⟩` has value `n` below `2^32`. -/
theorem gemm6_u32_val_ofNat {n : Nat} (h : n < 2 ^ 32) :
    (⟨BitVec.ofNat _ n⟩ : Std.U32).val = n := by
  show (BitVec.ofNat _ n).toNat = n
  rw [BitVec.toNat_ofNat, UScalarTy.U32_numBits_eq]
  exact Nat.mod_eq_of_lt h

/-- Value of the model `core.num.I64.trailing_zeros` on the `u32` it returns. -/
theorem gemm6_i64_trailing_zeros_spec (x : Std.I64) :
    core.num.I64.trailing_zeros x ⦃ r => r.val = numGap_trailingZeros x.bv ⦄ := by
  rw [core.num.I64.trailing_zeros_val, WP.spec_ok]
  exact gemm6_u32_val_ofNat (gemm6_trailingZeros_lt32 x)

/-- Value of the `u32 -> i64` widening (the `isize`-range check is vacuous on a `u32`). -/
theorem gemm6_fromU32I64_spec (x : Std.U32) :
    I64.Insts.CoreConvertFromU32.from x ⦃ y => y.val = (x.val : Int) ⦄ :=
  UScalar.hcast_inBounds_spec .I64 x (by scalar_tac)

/-- `core.cmp.Ord.max.default core.cmp.OrdU32` returns the larger word. -/
theorem gemm6_ordU32_max_eq (x y : Std.U32) :
    core.cmp.Ord.max.default core.cmp.OrdU32 x y
      = ok (if y.val < x.val then x else y) := by
  unfold core.cmp.Ord.max.default core.cmp.Ord.max_body
  simp [core.cmp.impls.PartialOrdU32.lt, liftFun2]
  split <;> rfl

/-- `max.default OrdU32` returns `max x.val y.val` as a natural. -/
theorem gemm6_ordU32_max_spec (x y : Std.U32) :
    ∃ z : Std.U32, core.cmp.Ord.max.default core.cmp.OrdU32 x y = ok z ∧
      z.val = Max.max x.val y.val := by
  refine ⟨if y.val < x.val then x else y, gemm6_ordU32_max_eq x y, ?_⟩
  by_cases h : y.val < x.val
  · rw [if_pos h]; exact (Nat.max_eq_left (le_of_lt h)).symm
  · rw [if_neg h]; exact (Nat.max_eq_right (le_of_not_gt h)).symm

/-- `kern.gemm.LANES` is 8, the lane stride. -/
theorem gemm6_LANES_val : (kern.gemm.LANES : Std.Usize).val = 8 := by
  unfold kern.gemm.LANES; rfl

/-- In-bounds `List.getD` of an `i64` list is its `getElem`. -/
theorem gemm6_getD_getElem (l : List Std.I64) (i : Nat) (h : i < l.length) :
    l.getD i 0#i64 = l[i]'h := by
  rw [List.getD, List.getElem?_eq_getElem h]
  rfl

/-! ## References (plain Lean) -/

/-- The `sh` the block loop writes for one block: `0` on a zero scale,
`s.trailing_zeros() - lo` otherwise (the extracted checked `u32` subtraction, on `Nat`). -/
def gemm6_shRef (s : Std.I64) (lo : Std.U32) : Nat :=
  if s.val = 0 then 0 else numGap_trailingZeros s.bv - lo.val

/-- The `sh` reference is below `2^63`, so its `i64` literal does not truncate. -/
theorem gemm6_shRef_lt (s : Std.I64) (lo : Std.U32) : gemm6_shRef s lo < 2 ^ 63 := by
  unfold gemm6_shRef
  split
  · norm_num
  · have := gemm6_trailingZeros_le s.bv
    omega

/-- The `i64` literal of a natural. -/
def gemm6_i64Lit (n : Nat) : Std.I64 := ⟨BitVec.ofNat 64 n⟩

/-- The literal's value, below `2^63`. -/
theorem gemm6_i64Lit_val {n : Nat} (h : n < 2 ^ 63) : (gemm6_i64Lit n).val = (n : Int) := by
  show (BitVec.ofNat 64 n).toInt = (n : Int)
  rw [BitVec.toInt_ofNat']
  refine Int.bmod_eq_of_le ?_ ?_
  · rw [show (↑(2 ^ 64 : Nat) : Int) / 2 = (2 : Int) ^ 63 from by norm_num]; omega
  · rw [show ((↑(2 ^ 64 : Nat) : Int) + 1) / 2 = (2 : Int) ^ 63 from by norm_num]; omega

/-- `64 ≤ Usize.max`, the largest index the `Slice` constructors admit. -/
theorem gemm6_usize_max_ge_64 : (64 : Nat) ≤ Std.Usize.max := by
  have h := Slice.length_ineq (Slice.from (List.replicate 64 0#u32)
    (by simp only [List.length_replicate]; scalar_tac))
  simpa only [Slice.from_val, List.length_replicate] using h

/-- The words after the block loop has packed blocks `0 .. n-1`: the `n`-fold composition of
`Gemm5`'s eight-quad packer. -/
def gemm6_packBlocks (w0 : List Std.U32) (lane : Nat) (codes : List Std.I8) :
    Nat → List Std.U32
  | 0 => w0
  | n + 1 => gemm5_packQuads (gemm6_packBlocks w0 lane codes n) lane codes n 8

/-- The shifts after blocks `0 .. n-1`, each an `i64` widening of `gemm6_shRef`. -/
def gemm6_setShifts (sh0 : List Std.I64) (lane : Nat) (scales : List Std.I64) (lo : Std.U32) :
    Nat → List Std.I64
  | 0 => sh0
  | n + 1 => (gemm6_setShifts sh0 lane scales lo n).set (n * 8 + lane)
      (gemm6_i64Lit (gemm6_shRef (scales.getD n 0#i64) lo))

/-- The running maximum spread after blocks `0 .. n-1`. -/
def gemm6_spread (g0 : Nat) (scales : List Std.I64) (lo : Std.U32) : Nat → Nat
  | 0 => g0
  | n + 1 => Max.max (gemm6_spread g0 scales lo n) (gemm6_shRef (scales.getD n 0#i64) lo)

/-- One unfolding of `gemm6_packBlocks`, for the loop-body successor index. -/
theorem gemm6_packBlocks_succ (w0 : List Std.U32) (lane : Nat) (codes : List Std.I8) (n : Nat) :
    gemm6_packBlocks w0 lane codes (n + 1) =
      gemm5_packQuads (gemm6_packBlocks w0 lane codes n) lane codes n 8 := rfl

/-- One unfolding of `gemm6_setShifts`, for the loop-body successor index. -/
theorem gemm6_setShifts_succ (sh0 : List Std.I64) (lane : Nat) (scales : List Std.I64)
    (lo : Std.U32) (n : Nat) :
    gemm6_setShifts sh0 lane scales lo (n + 1) =
      (gemm6_setShifts sh0 lane scales lo n).set (n * 8 + lane)
        (gemm6_i64Lit (gemm6_shRef (scales.getD n 0#i64) lo)) := rfl

/-- One unfolding of `gemm6_spread`, for the loop-body successor index. -/
theorem gemm6_spread_succ (g0 : Nat) (scales : List Std.I64) (lo : Std.U32) (n : Nat) :
    gemm6_spread g0 scales lo (n + 1) =
      Max.max (gemm6_spread g0 scales lo n) (gemm6_shRef (scales.getD n 0#i64) lo) := rfl

/-- The `Slice.from` constructor's length, in the constructor's own list. -/
theorem gemm6_slice_from_length {α : Type} (l : List α) (h : l.length ≤ Std.Usize.max) :
    (Slice.from l h).length = l.length := by
  show (Slice.from l h).val.length = l.length
  rw [Slice.from_val]

/-- A fold of `List.set` writes preserves the list length. -/
theorem gemm6_foldl_set_length {α : Type} (l : List Nat) (w : List α) (idx : Nat → Nat)
    (f : Nat → α) :
    (l.foldl (fun w k => w.set (idx k) (f k)) w).length = w.length := by
  induction l generalizing w with
  | nil => rfl
  | cons k ks ih =>
    simp only [List.foldl_cons]
    rw [ih, List.length_set]

/-- `gemm5_packQuads` preserves the word-list length. -/
theorem gemm6_packQuads_length (w0 : List Std.U32) (lane : Nat) (codes : List Std.I8)
    (b q : Nat) : (gemm5_packQuads w0 lane codes b q).length = w0.length := by
  unfold gemm5_packQuads
  exact gemm6_foldl_set_length (List.range q) w0 _ _

/-- `gemm6_packBlocks` preserves the word-list length. -/
theorem gemm6_packBlocks_length (w0 : List Std.U32) (lane : Nat) (codes : List Std.I8) (n : Nat) :
    (gemm6_packBlocks w0 lane codes n).length = w0.length := by
  induction n with
  | zero => rfl
  | succ n ih => rw [gemm6_packBlocks, gemm6_packQuads_length, ih]

/-- The block-loop `sh` computation as a spec: the `if s = 0` branch and the checked `u32`
subtraction. -/
theorem gemm6_sh_calc (s : Std.I64) (lo : Std.U32)
    (hlo : s.val ≠ 0 → lo.val ≤ numGap_trailingZeros s.bv) :
    (if s = 0#i64 then ok 0#u32
      else (do let i ← core.num.I64.trailing_zeros s; i - lo)) ⦃ sh =>
        sh.val = gemm6_shRef s lo ⦄ := by
  by_cases hz : s = 0#i64
  · rw [if_pos hz, WP.spec_ok]
    have hv : s.val = 0 := by rw [IScalar.val_eq_of_eq hz]; rfl
    unfold gemm6_shRef
    rw [if_pos hv]
    rfl
  · rw [if_neg hz]
    have hv : s.val ≠ 0 := fun h => hz (IScalar.eq_of_val_eq (by rw [h]; rfl))
    obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (gemm6_i64_trailing_zeros_spec s)
    rw [hi]; simp only [bind_tc_ok]
    obtain ⟨i1, hi1, hi1v⟩ :=
      WP.spec_imp_exists (U32.sub_spec (x := i) (y := lo) (by rw [hiv]; exact hlo hv))
    rw [hi1, WP.spec_ok]
    unfold gemm6_shRef
    rw [if_neg hv, hi1v.1, hiv]

/-! ## The block-loop body: one step -/

/-- One body step of `pack_group_loop0_loop1`: on `b < nb` it returns the `cont` state with the
quad packer applied to block `b`, the `shift` write and the `group_spread` maximum, and `b` advanced
by one. -/
theorem gemm6_body_cont
    (scales : Slice Std.I64) (lo : Std.U32) (lane : Std.Usize) (codes : Slice Std.I8)
    (nb : Std.Usize) (words : Slice Std.U32) (shift : Slice Std.I64)
    (group_spread : Std.U32) (b : Std.Usize)
    (hb : b.val < nb.val)
    (hsb : b.val < scales.length)
    (hlo : (scales.val.getD b.val 0#i64).val ≠ 0 →
      lo.val ≤ numGap_trailingZeros (scales.val.getD b.val 0#i64).bv)
    (hupd : b.val * 8 + lane.val < shift.length)
    (hno : (b.val * 8 + 8) * 8 + lane.val ≤ Std.Usize.max)
    (hcode : ∀ q, q < 8 → b.val * 32 + 4 * q + 3 < codes.length)
    (hbound : ∀ q, q < 8 → (b.val * 8 + q) * 8 + lane.val < words.length) :
    ∃ (words1 : Slice Std.U32) (shift1 : Slice Std.I64) (gs1 : Std.U32) (b1 : Std.Usize),
      kern.gemm.pack_group_loop0_loop1.body scales lo lane codes nb words shift group_spread b
          = ok (ControlFlow.cont (words1, shift1, gs1, b1)) ∧
      b1.val = b.val + 1 ∧
      words1.val = gemm5_packQuads words.val lane.val codes.val b.val 8 ∧
      shift1.length = shift.length ∧
      shift1.val = shift.val.set (b.val * 8 + lane.val)
        (gemm6_i64Lit (gemm6_shRef (scales.val.getD b.val 0#i64) lo)) ∧
      gs1.val = Max.max group_spread.val (gemm6_shRef (scales.val.getD b.val 0#i64) lo) := by
  have hmul : b.val * 8 ≤ Std.Usize.max := by omega
  have haddu : b.val * 8 + lane.val ≤ Std.Usize.max := by omega
  unfold kern.gemm.pack_group_loop0_loop1.body
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hb)]
  obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists (Slice.index_usize_spec scales b hsb)
  rw [hs]; simp only [bind_tc_ok]
  have hsval : s = scales.val.getD b.val 0#i64 := by
    rw [hsv, gemm6_getD_getElem scales.val b.val hsb]
  have hlo' : s.val ≠ 0 → lo.val ≤ numGap_trailingZeros s.bv := by
    intro h
    rw [hsval] at h
    have h' := hlo h
    rwa [← hsval] at h'
  obtain ⟨sh, hsh, hshv⟩ := WP.spec_imp_exists (gemm6_sh_calc s lo hlo')
  rw [hsh]; simp only [bind_tc_ok]
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (gemm6_fromU32I64_spec sh)
  rw [hi]; simp only [bind_tc_ok]
  obtain ⟨i1, hi1, hi1v⟩ :=
    WP.spec_imp_exists (Usize.mul_spec (x := b) (y := kern.gemm.LANES) (by
      rw [gemm6_LANES_val]; exact hmul))
  rw [hi1]; simp only [bind_tc_ok]
  obtain ⟨i2, hi2, hi2v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := i1) (y := lane) (by
      rw [hi1v, gemm6_LANES_val]; exact haddu))
  rw [hi2]; simp only [bind_tc_ok]
  obtain ⟨s1, hs1, hs1v⟩ :=
    WP.spec_imp_exists (Slice.update_spec shift i2 i (by
      rw [hi2v, hi1v, gemm6_LANES_val]; exact hupd))
  rw [hs1]; simp only [bind_tc_ok]
  obtain ⟨gs1, hgs, hgsv⟩ := gemm6_ordU32_max_spec group_spread sh
  rw [hgs]; simp only [bind_tc_ok]
  obtain ⟨w1, hw1, hw1v⟩ :=
    WP.spec_imp_exists (word_pack_group_loop0_loop1_loop0_ok lane codes b words hno hcode hbound)
  rw [hw1]; simp only [bind_tc_ok]
  obtain ⟨b1, hb1, hb1v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := b) (y := 1#usize) (by scalar_tac))
  rw [hb1]; simp only [bind_tc_ok]
  have hb1v' : b1.val = b.val + 1 := by
    have hone : ((1#usize : Std.Usize) : Nat) = 1 := rfl
    rw [hb1v, hone]
  have hieq : i = gemm6_i64Lit (gemm6_shRef (scales.val.getD b.val 0#i64) lo) := by
    apply IScalar.eq_of_val_eq
    rw [hiv, hshv, ← hsval]
    exact (gemm6_i64Lit_val (gemm6_shRef_lt s lo)).symm
  refine ⟨w1, s1, gs1, b1, rfl, hb1v', hw1v, ?_, ?_, ?_⟩
  · show s1.val.length = shift.val.length
    rw [hs1v, Slice.set_val_eq, List.length_set]
  · rw [hs1v, Slice.set_val_eq, hi2v, hi1v, gemm6_LANES_val, hieq]
  · rw [hgsv, hshv, ← hsval]

/-- The body step on `¬ b < nb`: it returns `done` with the state unchanged. -/
theorem gemm6_body_done
    (scales : Slice Std.I64) (lo : Std.U32) (lane : Std.Usize) (codes : Slice Std.I8)
    (nb : Std.Usize) (words : Slice Std.U32) (shift : Slice Std.I64)
    (group_spread : Std.U32) (b : Std.Usize) (hb : ¬ b < nb) :
    kern.gemm.pack_group_loop0_loop1.body scales lo lane codes nb words shift group_spread b
      = ok (ControlFlow.done (words, shift, group_spread)) := by
  unfold kern.gemm.pack_group_loop0_loop1.body
  rw [if_neg hb]

/-! ## The block loop -/

/-- The block loop writes, for every block `b < nb`, the eight packed words of block `b`, the
widened `sh` at `shift[b*8 + lane]`, and the running maximum `group_spread`.

The loop's measure is `nb - b`; the invariant is the running `gemm6_packBlocks` /
`gemm6_setShifts` / `gemm6_spread` folds plus the two length preservations that keep the `Slice`
indices in bounds. -/
@[step]
theorem word_pack_group_loop0_loop1_ok
    (words : Slice Std.U32) (shift : Slice Std.I64) (group_spread : Std.U32)
    (scales : Slice Std.I64) (lo : Std.U32) (lane : Std.Usize) (codes : Slice Std.I8)
    (nb : Std.Usize)
    (hlo : ∀ k, k < nb.val → (scales.val.getD k 0#i64).val ≠ 0 →
      lo.val ≤ numGap_trailingZeros (scales.val.getD k 0#i64).bv)
    (hsc : ∀ k, k < nb.val → k < scales.length)
    (hupd : ∀ k, k < nb.val → k * 8 + lane.val < shift.length)
    (hno : ∀ k, k < nb.val → (k * 8 + 8) * 8 + lane.val ≤ Std.Usize.max)
    (hcode : ∀ k, k < nb.val → ∀ q, q < 8 → k * 32 + 4 * q + 3 < codes.length)
    (hbound : ∀ k, k < nb.val → ∀ q, q < 8 → (k * 8 + q) * 8 + lane.val < words.length) :
    kern.gemm.pack_group_loop0_loop1 words shift group_spread scales lo lane codes nb 0#usize ⦃ r =>
      r.1.val = gemm6_packBlocks words.val lane.val codes.val nb.val ∧
      r.2.1.length = shift.length ∧
      r.2.1.val = gemm6_setShifts shift.val lane.val scales.val lo nb.val ∧
      r.2.2.val = gemm6_spread group_spread.val scales.val lo nb.val ⦄ := by
  unfold kern.gemm.pack_group_loop0_loop1
  apply loop.spec_decr_nat
    (measure := fun st : (Slice Std.U32 × Slice Std.I64 × Std.U32 × Std.Usize) =>
      nb.val - st.2.2.2.val)
    (inv := fun st : (Slice Std.U32 × Slice Std.I64 × Std.U32 × Std.Usize) =>
      st.2.2.2.val ≤ nb.val ∧
      st.1.length = words.length ∧
      st.2.1.length = shift.length ∧
      st.1.val = gemm6_packBlocks words.val lane.val codes.val st.2.2.2.val ∧
      st.2.1.val = gemm6_setShifts shift.val lane.val scales.val lo st.2.2.2.val ∧
      st.2.2.1.val = gemm6_spread group_spread.val scales.val lo st.2.2.2.val)
    (post := fun r : (Slice Std.U32 × Slice Std.I64 × Std.U32) =>
      r.1.val = gemm6_packBlocks words.val lane.val codes.val nb.val ∧
      r.2.1.length = shift.length ∧
      r.2.1.val = gemm6_setShifts shift.val lane.val scales.val lo nb.val ∧
      r.2.2.val = gemm6_spread group_spread.val scales.val lo nb.val)
  · rintro ⟨w, sh, gs, b⟩ ⟨hble, hwlen, hshlen, hwv, hshv, hgsv⟩
    dsimp only at *
    by_cases hlt : b.val < nb.val
    · obtain ⟨w1, sh1, gs1, b1, hbody, hb1v, hw1v, hsh1len, hsh1v, hgs1v⟩ :=
        gemm6_body_cont scales lo lane codes nb w sh gs b hlt (hsc b.val hlt) (hlo b.val hlt)
          (by rw [hshlen]; exact hupd b.val hlt) (hno b.val hlt) (hcode b.val hlt)
          (by rw [hwlen]; exact hbound b.val hlt)
      rw [hbody, WP.spec_ok]
      dsimp only
      refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
      · omega
      · show w1.val.length = words.val.length
        rw [hw1v, gemm6_packQuads_length]
        exact hwlen
      · rw [hsh1len, hshlen]
      · rw [hw1v, hwv, hb1v, gemm6_packBlocks_succ]
      · rw [hsh1v, hshv, hb1v, gemm6_setShifts_succ]
      · rw [hgs1v, hgsv, hb1v, gemm6_spread_succ]
      · omega
    · have hbge : nb.val ≤ b.val := by omega
      have hbeq : b.val = nb.val := le_antisymm hble hbge
      rw [gemm6_body_done scales lo lane codes nb w sh gs b (by rw [UScalar.lt_equiv]; omega),
        WP.spec_ok]
      dsimp only
      exact ⟨by rw [hwv, hbeq], hshlen, by rw [hshv, hbeq], by rw [hgsv, hbeq]⟩
  · refine ⟨?_, rfl, rfl, ?_, ?_, ?_⟩
    · exact Nat.zero_le _
    · rfl
    · rfl
    · rfl

/-! ## Non-vacuity -/

/-- Non-vacuity of the `sh` reference: `8` has three trailing zeros, minus `lo = 2`. -/
example : gemm6_shRef (8#i64) 2#u32 = 1 := by decide

/-- Non-vacuity of the `sh` reference: a zero scale maps to `0`, whatever `lo` is. -/
example : gemm6_shRef (0#i64) 5#u32 = 0 := by decide

/-- Non-vacuity of the block reference: one block packs its eight quads into the word list. -/
example : gemm6_packBlocks ([0#u32, 0#u32] : List Std.U32) 1 ([0#i8, 1#i8] : List Std.I8) 1
    = gemm5_packQuads ([0#u32, 0#u32] : List Std.U32) 1 ([0#i8, 1#i8] : List Std.I8) 0 8 :=
  rfl

/-- Non-vacuity of the hypothesis bundle: `scales = [1]`, `codes` length 32, `words`/`shift` length
64, `lo = 0`, `lane = 0`, `nb = 1` satisfy `hlo`, `hsc`, `hupd`, `hno`, `hcode` and `hbound`
jointly. -/
example : ∃ (words : Slice Std.U32) (shift : Slice Std.I64) (codes : Slice Std.I8)
    (scales : Slice Std.I64),
    (∀ k, k < (1#usize).val → (scales.val.getD k 0#i64).val ≠ 0 →
      (0#u32).val ≤ numGap_trailingZeros (scales.val.getD k 0#i64).bv) ∧
    (∀ k, k < (1#usize).val → k < scales.length) ∧
    (∀ k, k < (1#usize).val → k * 8 + (0#usize).val < shift.length) ∧
    (∀ k, k < (1#usize).val → (k * 8 + 8) * 8 + (0#usize).val ≤ Std.Usize.max) ∧
    (∀ k, k < (1#usize).val → ∀ q, q < 8 → k * 32 + 4 * q + 3 < codes.length) ∧
    (∀ k, k < (1#usize).val → ∀ q, q < 8 → (k * 8 + q) * 8 + (0#usize).val < words.length) := by
  refine ⟨Slice.from (List.replicate 64 0#u32) (by simp only [List.length_replicate]; scalar_tac),
    Slice.from (List.replicate 64 0#i64) (by simp only [List.length_replicate]; scalar_tac),
    Slice.from (List.replicate 32 1#i8) (by simp only [List.length_replicate]; scalar_tac),
    Slice.from [1#i64] (by simp only [List.length_cons, List.length_nil]; scalar_tac),
    ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro k hk h
    exact Nat.zero_le _
  · intro k hk
    rw [gemm6_slice_from_length, List.length_cons, List.length_nil]
    have hk1 : k < 1 := hk
    omega
  · intro k hk
    rw [gemm6_slice_from_length, List.length_replicate]
    have h0 : ((0#usize : Std.Usize) : Nat) = 0 := rfl
    rw [h0]
    have hk1 : k < 1 := hk
    omega
  · intro k hk
    have hk1 : k < 1 := hk
    have hk0 : k = 0 := by omega
    subst hk0
    have h0 : ((0#usize : Std.Usize) : Nat) = 0 := rfl
    rw [h0]
    rw [show (0 * 8 + 8) * 8 + 0 = (64 : Nat) from by norm_num]
    exact gemm6_usize_max_ge_64
  · intro k hk q hq
    rw [gemm6_slice_from_length, List.length_replicate]
    have hk1 : k < 1 := hk
    omega
  · intro k hk q hq
    rw [gemm6_slice_from_length, List.length_replicate]
    have h0 : ((0#usize : Std.Usize) : Nat) = 0 := rfl
    rw [h0]
    have hk1 : k < 1 := hk
    omega

end apeiron_word
