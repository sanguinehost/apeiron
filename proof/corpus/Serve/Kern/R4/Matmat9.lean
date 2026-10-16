import Serve.Kern.R4.Matmat8

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow

/-!
# Serving kernel group `kern.matmat` (part 9): the outer `row_accs` block loop

`src/engine/kern/matmat.rs:494-523` is `row_accs_loop0`, the loop that walks the output blocks
`blk ∈ [0, geom.blocks)`. For each block it reads `d ← dequant.f16_to_fixed` of the two `f16`
header bytes, reads `dots ← q8_0_block_kdots`, accumulates into the `i128` totals with the inner
`row_accs_loop0_loop0`, and advances to the next block; a block whose inner loop bails stops the
whole row with `some RowOutcome.Bail`.

This file states WHAT that loop computes, not how it loops:

* §0 the block references — the per-slot contribution of a block over the inner cursor range, the
  running sum over the blocks `[0, n)`, and the two accumulator references.
* §1 the composition steps: one clean block extends the running sum by its contribution, and one
  bailing block extends the completed sum by its partial accumulation.
* §2 the inner-loop laws of `Matmat8` instanced at the outer cursor `0`.
* §3 the outer-loop body step (`Matmat6`/`Matmat8` body idioms).
* §4 **`word_matmat9_row_accs_loop0_ok`**: the outer loop law over the whole block range.
* §5 non-vacuity: the hypothesis bundle is satisfiable at concrete inputs.

Trust boundaries: rustc/LLVM codegen; the `F32`/AVX2 layers; the `dequant.f16_to_fixed` and
`q8_0_block_kdots` drivers, whose outputs `d` and `dots` enter as named hypotheses. A
`dequant.f16_to_fixed` returning `Err` (`NonFinite`) is outside the stated domain — the hypotheses
require `Ok` for every block.

No `sorry`/`admit`/`axiom`/`native_decide`/`bv_decide`; `decide` is only used on literal
`isize`/`usize` values, never on an extracted body.
-/

/-! ## §0 · The block references -/

/-- The `i128` contribution `d·(scale·dot)` of block `b`, with the block read as a natural index
against the totalised `matmat7_scale`/`matmat7_dotv` (the Natural-indexed twin of `matmat7_term`). -/
def matmat9_termN (acts : Slice chart.QuantAct) (b : Nat) (d : I64)
    (dots : Array I32 16#usize) (j : Nat) : ℤ :=
  d.val * (((acts.val.getD j qaDefault).scales.val.getD b (0#i64)).val *
    (dots.val.getD j (0#i32)).val)

/-- The per-column contribution of block `b`: `matmat9_termN` on a `Live` column, `0` otherwise. -/
def matmat9_deltaN (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (b : Nat) (d : I64) (dots : Array I32 16#usize) (j : Nat) : ℤ :=
  match state.val.getD j kern.matmat.ColumnState.Live with
  | .Live => matmat9_termN acts b d dots j
  | _ => 0

/-- The inner loop's contribution of block `b` into slot `j`: `matmat9_deltaN` inside the cursor
range `[0, k)` (the inner loop is entered at cursor `0`). -/
def matmat9_deltaB (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (k : Std.Usize) (b : Nat) (d : I64) (dots : Array I32 16#usize) (j : Nat) : ℤ :=
  if j < k.val then matmat9_deltaN acts state b d dots j else 0

/-- The accumulated contribution of the blocks `[0, n)`: block `0` first, block `n-1` last. -/
def matmat9_sum (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (k : Std.Usize) (ds : Nat → I64) (dotsF : Nat → Array I32 16#usize) (j : Nat) : Nat → ℤ
  | 0 => 0
  | n + 1 => matmat9_sum acts state k ds dotsF j n +
      matmat9_deltaB acts state k n (ds n) (dotsF n) j

/-- The totals after `n` blocks: each slot is its initial value plus `matmat9_sum`. -/
def matmat9_acc (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (k : Std.Usize) (ds : Nat → I64) (dotsF : Nat → Array I32 16#usize)
    (T0 : Array I128 16#usize) (n : Nat) (T : Array I128 16#usize) : Prop :=
  T.val.length = T0.val.length ∧
  ∀ j, j < T0.val.length →
    (T.val.getD j (0#i128)).val =
      (T0.val.getD j (0#i128)).val + matmat9_sum acts state k ds dotsF j n

/-- The totals when block `b0` bails at column `j0`: the sum over the completed blocks `[0, b0)`
plus block `b0`'s contributions on the columns `[0, j0)` already passed. -/
def matmat9_bailref (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (k : Std.Usize) (ds : Nat → I64) (dotsF : Nat → Array I32 16#usize)
    (T0 : Array I128 16#usize) (b0 j0 : Nat) (T : Array I128 16#usize) : Prop :=
  T.val.length = T0.val.length ∧
  ∀ j, j < T0.val.length →
    (T.val.getD j (0#i128)).val =
      (T0.val.getD j (0#i128)).val + matmat9_sum acts state k ds dotsF j b0 +
        (if j < j0 then matmat9_deltaN acts state b0 (ds b0) (dotsF b0) j else 0)

/-- The Natural-indexed block reference is the `Usize`-indexed `matmat7_delta` at the block's value. -/
theorem matmat9_deltaN_eq (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize) (j : Nat) :
    matmat9_deltaN acts state blk.val d dots j = matmat7_delta acts state blk d dots j := by
  unfold matmat9_deltaN matmat9_termN matmat7_delta matmat7_term matmat7_scale matmat7_dotv
  rfl

/-- The inner cursor guard `0 ≤ j ∧ j < k` is `j < k` for a natural `j`. -/
theorem matmat9_guard_eq (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (k : Std.Usize) (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize) (j : Nat) :
    (if 0 ≤ j ∧ j < k.val then matmat7_delta acts state blk d dots j else 0) =
      matmat9_deltaB acts state k blk.val d dots j := by
  unfold matmat9_deltaB
  rw [matmat9_deltaN_eq]
  by_cases h : j < k.val
  · rw [if_pos (by omega), if_pos h]
  · rw [if_neg (by omega), if_neg h]

/-- The bail guard `0 ≤ j ∧ j < j0` is `j < j0` for a natural `j`. -/
theorem matmat9_bailguard_eq (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize) (j0 j : Nat) :
    (if 0 ≤ j ∧ j < j0 then matmat7_delta acts state blk d dots j else 0) =
      (if j < j0 then matmat9_deltaN acts state blk.val d dots j else 0) := by
  rw [matmat9_deltaN_eq]
  by_cases h : j < j0
  · rw [if_pos (by omega), if_pos h]
  · rw [if_neg (by omega), if_neg h]

/-- One clean block composes: the accumulator after `b + 1` blocks is the accumulator after `b` plus
block `b`'s contributions. -/
theorem matmat9_acc_step (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (k : Std.Usize) (ds : Nat → I64) (dotsF : Nat → Array I32 16#usize)
    (T0 T T' : Array I128 16#usize) (blk : Std.Usize)
    (hacc : matmat9_acc acts state k ds dotsF T0 blk.val T)
    (hstep : matmat8_acc acts state blk (ds blk.val) (dotsF blk.val) T 0 k.val T') :
    matmat9_acc acts state k ds dotsF T0 (blk.val + 1) T' := by
  obtain ⟨hlen, hpt⟩ := hacc
  obtain ⟨hlen', hpt'⟩ := hstep
  refine ⟨by rw [hlen', hlen], ?_⟩
  intro j hj
  have hjT : j < T.val.length := by rw [hlen]; exact hj
  rw [hpt' j hjT, matmat9_guard_eq acts state k blk (ds blk.val) (dotsF blk.val) j, hpt j hj]
  show (T0.val.getD j (0#i128)).val + matmat9_sum acts state k ds dotsF j blk.val +
      matmat9_deltaB acts state k blk.val (ds blk.val) (dotsF blk.val) j =
    (T0.val.getD j (0#i128)).val + matmat9_sum acts state k ds dotsF j (blk.val + 1)
  rw [show matmat9_sum acts state k ds dotsF j (blk.val + 1) =
      matmat9_sum acts state k ds dotsF j blk.val +
        matmat9_deltaB acts state k blk.val (ds blk.val) (dotsF blk.val) j from rfl]
  ring

/-- A bailing block composes: block `blk`'s partial accumulation on `[0, j0)` extends the sum over
the completed blocks `[0, blk.val)`. -/
theorem matmat9_bailref_step (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (k : Std.Usize) (ds : Nat → I64) (dotsF : Nat → Array I32 16#usize)
    (T0 T T' : Array I128 16#usize) (blk : Std.Usize) (j0 : Nat)
    (hacc : matmat9_acc acts state k ds dotsF T0 blk.val T)
    (hstep : matmat8_acc acts state blk (ds blk.val) (dotsF blk.val) T 0 j0 T') :
    matmat9_bailref acts state k ds dotsF T0 blk.val j0 T' := by
  obtain ⟨hlen, hpt⟩ := hacc
  obtain ⟨hlen', hpt'⟩ := hstep
  refine ⟨by rw [hlen', hlen], ?_⟩
  intro j hj
  have hjT : j < T.val.length := by rw [hlen]; exact hj
  rw [hpt' j hjT,
    matmat9_bailguard_eq acts state blk (ds blk.val) (dotsF blk.val) j0 j, hpt j hj]

/-! ## §1 · The per-block domain bundles -/

/-- The columns of block `blk` over the inner cursor range `[0, k)`: each `Live` column has an
in-range scale index and in-range checked operations against the accumulator `T`. This is the
hypothesis bundle of `word_matmat8_rowaccs_total_ok` at `c = 0`. -/
def matmat9_clean (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (k : Std.Usize) (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize)
    (T : Array I128 16#usize) : Prop :=
  ∀ j (hj : j < state.val.length) (hjA : j < acts.val.length),
    j < k.val → state.val[j] = kern.matmat.ColumnState.Live →
    blk.val < (acts.val[j]).scales.val.length ∧
    matmat8_colDom acts blk d dots (T.val.getD j (0#i128)) j

/-- Block `blk` bails at column `j0` against the accumulator `T`: the columns `[0, j0)` are `Live`
with in-range checked operations, column `j0` satisfies the bail predicate, and `j0` is inside every
slice domain. This is the hypothesis bundle of `word_matmat8_rowaccs_bail_ok` at `c = 0`; the
`j0`-indexed slice reads are stated through `List.getD` so the predicate is a total `Prop`. -/
def matmat9_bailpre (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (k : Std.Usize) (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize)
    (T : Array I128 16#usize) (j0 : Nat) : Prop :=
  j0 < k.val ∧ k.val ≤ state.val.length ∧ j0 < state.val.length ∧ j0 < acts.val.length ∧
  j0 < dots.val.length ∧ j0 < T.val.length ∧
  (∀ j (hj : j < acts.val.length), j < j0 → blk.val < (acts.val[j]).scales.val.length) ∧
  (∀ j (hj : j < state.val.length), j < j0 → state.val[j] = kern.matmat.ColumnState.Live) ∧
  blk.val < (acts.val.getD j0 qaDefault).scales.val.length ∧
  state.val.getD j0 kern.matmat.ColumnState.Live = kern.matmat.ColumnState.Live ∧
  (∀ j, j < j0 →
    I64.min ≤ (matmat7_scale acts blk j).val * (matmat7_dotv dots j).val ∧
    (matmat7_scale acts blk j).val * (matmat7_dotv dots j).val ≤ I64.max ∧
    I128.min ≤ d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ∧
    d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ≤ I128.max ∧
    I128.min ≤ (T.val.getD j (0#i128)).val +
      d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ∧
    (T.val.getD j (0#i128)).val +
      d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ≤ I128.max) ∧
  matmat8_bails acts blk d dots (T.val.getD j0 (0#i128)) j0

/-! ## §2 · The inner-loop laws at `c = 0` -/

/-- The `Matmat8` total law instanced at the outer loop's cursor `0`: a clean block returns
`(T'', none)` with `T'' = T + Σ_{j < k} delta`, stated as `matmat8_acc`. -/
theorem matmat9_inner_total (acts : Slice chart.QuantAct) (k : Std.Usize)
    (state : Slice kern.matmat.ColumnState) (blk : Std.Usize) (d : I64)
    (dots : Array I32 16#usize) (T : Array I128 16#usize)
    (hk16 : k.val ≤ (16#usize).val) (hk : k.val ≤ state.val.length)
    (hacts : k.val ≤ acts.val.length) (hclean : matmat9_clean acts state k blk d dots T) :
    ∃ T' : Array I128 16#usize,
      kern.matmat.row_accs_loop0_loop0 acts k state T none blk d dots 0#usize = ok (T', none) ∧
      matmat8_acc acts state blk d dots T 0 k.val T' := by
  obtain ⟨v, hv, T', hvT, hpt⟩ := WP.spec_imp_exists
    (word_matmat8_rowaccs_total_ok acts k state blk d dots T 0#usize (by simp) hk hacts
      (by rw [Array.length_eq]; exact hk16) (by rw [Array.length_eq]; exact hk16)
      (fun j hj hjA _ hlt hl => hclean j hj hjA hlt hl))
  rw [hv]
  refine ⟨T', by rw [hvT], ⟨by simp only [Array.length_eq], hpt⟩⟩

/-- The `Matmat8` bail law instanced at the outer loop's cursor `0`: a bailing block returns
`(T'', some Bail)` with `T'' = T + Σ_{j < j0} delta`, stated as `matmat8_acc`. -/
theorem matmat9_inner_bail (acts : Slice chart.QuantAct) (k : Std.Usize)
    (state : Slice kern.matmat.ColumnState) (blk : Std.Usize) (d : I64)
    (dots : Array I32 16#usize) (T : Array I128 16#usize) (j0 : Nat)
    (hpre : matmat9_bailpre acts state k blk d dots T j0) :
    ∃ T' : Array I128 16#usize,
      kern.matmat.row_accs_loop0_loop0 acts k state T none blk d dots 0#usize =
        ok (T', some kern.matmat.RowOutcome.Bail) ∧
      matmat8_acc acts state blk d dots T 0 j0 T' := by
  obtain ⟨hj0k, hk, hj0state, hj0acts, hj0dots, hj0T, hblk, hlive, hj0blkG, hj0liveG, hdom,
    hbailG⟩ := hpre
  have hj0blk : blk.val < (acts.val[j0]).scales.val.length := by
    rw [List.getD_eq_getElem acts.val qaDefault hj0acts] at hj0blkG
    exact hj0blkG
  have hj0live : state.val[j0] = kern.matmat.ColumnState.Live := by
    rw [List.getD_eq_getElem state.val kern.matmat.ColumnState.Live hj0state] at hj0liveG
    exact hj0liveG
  have hbail : matmat8_bails acts blk d dots (T.val[j0]'hj0T) j0 := by
    rw [List.getD_eq_getElem T.val (0#i128) hj0T] at hbailG
    exact hbailG
  obtain ⟨v, hv, T', hvT, hpt⟩ := WP.spec_imp_exists
    (word_matmat8_rowaccs_bail_ok acts k state blk d dots T 0#usize j0 (by simp) hj0k hk
      hj0state hj0acts hj0dots hj0T
      (fun j hj _ hlt => hblk j hj hlt) (fun j hj _ hlt => hlive j hj hlt) hj0blk hj0live
      (fun j _ hlt => hdom j hlt) hbail)
  rw [hv]
  exact ⟨T', by rw [hvT], hpt⟩

/-! ## §3 · The outer-loop body -/

/-- A clean block with a succeeding header read and a successful inner loop step: the body advances
to `(T', early', b1)`. -/
theorem matmat9_body_cont
    (row : Slice Std.U8) (geom : kern.matmat.Geom) (acts : Slice chart.QuantAct)
    (k : Std.Usize) (state : Slice kern.matmat.ColumnState) (avx : Bool)
    (T : Array I128 16#usize) (blk : Std.Usize) (offF off1F i4F b1 : Std.Usize)
    (d : I64) (dots : Array I32 16#usize) (T' : Array I128 16#usize)
    (early' : Option kern.matmat.RowOutcome)
    (hlt : blk.val < geom.blocks.val)
    (hoff : blk * geom.block_bytes = ok offF)
    (hoff1 : offF + 1#usize = ok off1F)
    (hoff2 : offF + 2#usize = ok i4F)
    (ho0 : offF.val < row.val.length) (ho1 : off1F.val < row.val.length)
    (hdeq : dequant.f16_to_fixed (core.num.U16.from_le_bytes (Array.make 2#usize
        [row.val[offF.val]'ho0, row.val[off1F.val]'ho1])) = ok (core.result.Result.Ok d))
    (hdots : kern.matmat.q8_0_block_kdots avx row i4F acts blk k = ok dots)
    (hinner : kern.matmat.row_accs_loop0_loop0 acts k state T none blk d dots 0#usize
      = ok (T', early'))
    (hb1 : blk + 1#usize = ok b1) :
    kern.matmat.row_accs_loop0.body kern.matmat.Family.Q8_0 row geom acts k state avx T none blk
      = ok (ControlFlow.cont (T', early', b1)) := by
  unfold kern.matmat.row_accs_loop0.body
  rw [if_pos ((UScalar.lt_equiv blk geom.blocks).2 hlt)]
  have h0 := slice_index_usize_ok row offF ho0
  have h1 := slice_index_usize_ok row off1F ho1
  simp only [core.option.Option.is_none, Option.isNone_none, if_true, bind_tc_ok]
  rw [hoff]
  simp only [bind_tc_ok]
  rw [h0]
  simp only [bind_tc_ok]
  rw [hoff1]
  simp only [bind_tc_ok]
  rw [h1]
  simp only [bind_tc_ok, lift]
  rw [hdeq]
  simp only [bind_tc_ok]
  rw [hoff2]
  simp only [bind_tc_ok]
  rw [hdots]
  simp only [bind_tc_ok]
  rw [hinner]
  simp only [bind_tc_ok]
  rw [hb1]
  simp only [bind_tc_ok]
  rfl

/-- The outer body past the block range returns the final accumulator. -/
theorem matmat9_body_done (family : kern.matmat.Family) (row : Slice Std.U8)
    (geom : kern.matmat.Geom) (acts : Slice chart.QuantAct) (k : Std.Usize)
    (state : Slice kern.matmat.ColumnState) (avx : Bool) (T : Array I128 16#usize)
    (blk : Std.Usize) (hlt : ¬ blk.val < geom.blocks.val) :
    kern.matmat.row_accs_loop0.body family row geom acts k state avx T none blk
      = ok (ControlFlow.done (T, none)) := by
  unfold kern.matmat.row_accs_loop0.body
  rw [if_neg (by rw [UScalar.lt_equiv]; exact hlt)]

/-- The outer body with `early = some x` returns immediately: an earlier block already stopped the
row. -/
theorem matmat9_body_early (family : kern.matmat.Family) (row : Slice Std.U8)
    (geom : kern.matmat.Geom) (acts : Slice chart.QuantAct) (k : Std.Usize)
    (state : Slice kern.matmat.ColumnState) (avx : Bool) (T : Array I128 16#usize)
    (x : kern.matmat.RowOutcome) (blk : Std.Usize) :
    kern.matmat.row_accs_loop0.body family row geom acts k state avx T (some x) blk
      = ok (ControlFlow.done (T, some x)) := by
  unfold kern.matmat.row_accs_loop0.body
  by_cases hlt : blk.val < geom.blocks.val
  · rw [if_pos ((UScalar.lt_equiv blk geom.blocks).2 hlt)]
    simp only [core.option.Option.is_none, Option.isNone_some, Bool.false_eq_true, if_false]
  · rw [if_neg (by rw [UScalar.lt_equiv]; exact hlt)]

/-! ## §4 · The outer loop law -/

/-- The invariant of `row_accs_loop0`: either the loop is still running at a cursor `≤ b0` with
`early = none` and the reference accumulator, or block `b0 < nb` has bailed and the next body call
stops the loop with `(totals, some Bail)`. -/
def matmat9_outInv (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (k : Std.Usize) (ds : Nat → I64) (dotsF : Nat → Array I32 16#usize)
    (T0 : Array I128 16#usize) (nb b0 : Nat)
    (st : (Array I128 16#usize) × Option kern.matmat.RowOutcome × Std.Usize) : Prop :=
  (st.2.1 = none ∧ st.2.2.val ≤ b0 ∧ matmat9_acc acts state k ds dotsF T0 st.2.2.val st.1) ∨
  (b0 < nb ∧ ∃ j0, st.2.1 = some kern.matmat.RowOutcome.Bail ∧ st.2.2.val = b0 + 1 ∧
    matmat9_bailref acts state k ds dotsF T0 b0 j0 st.1)

/-- **`row_accs` outer loop law.** `row_accs_loop0` over the output blocks `blk ∈ [0, geom.blocks)`,
with `nb = geom.blocks`, `b0` the first block at which the loop stops, `d_b` the
`dequant.f16_to_fixed` output and `dots_b` the `q8_0_block_kdots` output of block `b` (named by
their extracted equations). Every block `blk < b0` is clean and block `b0` bails at column `j0` when
`b0 < nb`. The loop then returns either `(totals', none)` with `totals'` the accumulation of all `nb`
blocks (`b0 = nb` in that case), or `(totals', some Bail)` with `totals'` the accumulation over the
completed blocks `[0, b0)` plus block `b0`'s contributions on the columns `[0, j0)`. -/
theorem word_matmat9_row_accs_loop0_ok
    (row : Slice Std.U8) (geom : kern.matmat.Geom) (acts : Slice chart.QuantAct)
    (k : Std.Usize) (state : Slice kern.matmat.ColumnState) (avx : Bool)
    (totals : Array I128 16#usize) (nb b0 : Nat)
    (ds : Nat → I64) (dotsF : Nat → Array I32 16#usize)
    (offF off1F i4F : Nat → Std.Usize)
    (hnb : nb = geom.blocks.val) (hb0 : b0 ≤ nb)
    (hk16 : k.val ≤ (16#usize).val)
    (hk : k.val ≤ state.val.length) (hacts : k.val ≤ acts.val.length)
    (hoff : ∀ blk : Std.Usize, blk.val < nb → blk * geom.block_bytes = ok (offF blk.val))
    (hoff1 : ∀ blk : Std.Usize, blk.val < nb → offF blk.val + 1#usize = ok (off1F blk.val))
    (hoff2 : ∀ blk : Std.Usize, blk.val < nb → offF blk.val + 2#usize = ok (i4F blk.val))
    (ho0 : ∀ blk : Std.Usize, blk.val < nb → (offF blk.val).val < row.val.length)
    (ho1 : ∀ blk : Std.Usize, blk.val < nb → (off1F blk.val).val < row.val.length)
    (hdeq : ∀ blk : Std.Usize, (hb : blk.val < nb) →
      dequant.f16_to_fixed (core.num.U16.from_le_bytes (Array.make 2#usize
        [row.val[(offF blk.val).val]'(ho0 blk hb), row.val[(off1F blk.val).val]'(ho1 blk hb)]))
        = ok (core.result.Result.Ok (ds blk.val)))
    (hdots : ∀ blk : Std.Usize, blk.val < nb →
      kern.matmat.q8_0_block_kdots avx row (i4F blk.val) acts blk k = ok (dotsF blk.val))
    (hclean : ∀ blk : Std.Usize, blk.val < b0 → ∀ T,
      matmat9_acc acts state k ds dotsF totals blk.val T →
        matmat9_clean acts state k blk (ds blk.val) (dotsF blk.val) T)
    (hbail : b0 < nb → ∀ blk : Std.Usize, blk.val = b0 → ∀ T,
      matmat9_acc acts state k ds dotsF totals b0 T →
        ∃ j0, matmat9_bailpre acts state k blk (ds blk.val) (dotsF blk.val) T j0) :
    kern.matmat.row_accs_loop0 kern.matmat.Family.Q8_0 row geom acts k state avx totals none
        0#usize ⦃ r =>
      (b0 = nb ∧ ∃ T', r = (T', none) ∧ matmat9_acc acts state k ds dotsF totals nb T') ∨
      (∃ T' j0, r = (T', some kern.matmat.RowOutcome.Bail) ∧
        matmat9_bailref acts state k ds dotsF totals b0 j0 T') ⦄ := by
  unfold kern.matmat.row_accs_loop0
  apply loop.spec_decr_nat
    (measure := fun st : (Array I128 16#usize) × Option kern.matmat.RowOutcome × Std.Usize =>
      nb - st.2.2.val)
    (inv := matmat9_outInv acts state k ds dotsF totals nb b0)
    (post := fun y : (Array I128 16#usize) × Option kern.matmat.RowOutcome =>
      (b0 = nb ∧ ∃ T', y = (T', none) ∧ matmat9_acc acts state k ds dotsF totals nb T') ∨
      (∃ T' j0, y = (T', some kern.matmat.RowOutcome.Bail) ∧
        matmat9_bailref acts state k ds dotsF totals b0 j0 T'))
  · rintro ⟨totals', early', blk'⟩ hinv
    rcases hinv with ⟨he, hle, hacc⟩ | ⟨hb0lt, j0, he, hcur, hbref⟩
    · dsimp only at he hle hacc
      subst he
      simp only []
      by_cases hlt : blk'.val < nb
      · by_cases hltb : blk'.val < b0
        · -- a clean block: accumulate and advance
          have hb' : blk'.val < geom.blocks.val := by rw [← hnb]; exact hlt
          obtain ⟨T'', hinner, hacc'⟩ := matmat9_inner_total acts k state blk' (ds blk'.val)
            (dotsF blk'.val) totals' hk16 hk hacts (hclean blk' hltb totals' hacc)
          obtain ⟨b1, hb1, hb1v⟩ := WP.spec_imp_exists
            (UScalar.add_spec (x := blk') (y := 1#usize) (by scalar_tac))
          rw [matmat9_body_cont row geom acts k state avx totals' blk' (offF blk'.val)
            (off1F blk'.val) (i4F blk'.val) b1 (ds blk'.val) (dotsF blk'.val) T'' none hb'
            (hoff blk' hlt) (hoff1 blk' hlt) (hoff2 blk' hlt) (ho0 blk' hlt)
            (ho1 blk' hlt) (hdeq blk' hlt) (hdots blk' hlt) hinner hb1]
          simp only [WP.spec_ok]
          have hone : (1#usize).val = 1 := by decide
          have hb1v' : b1.val = blk'.val + 1 := by rw [hb1v, hone]
          have hle1 : b1.val ≤ b0 := by omega
          have hmeas : nb - b1.val < nb - blk'.val := by omega
          refine ⟨Or.inl ⟨rfl, hle1, ?_⟩, hmeas⟩
          rw [hb1v']
          exact matmat9_acc_step acts state k ds dotsF totals totals' T'' blk' hacc hacc'
        · -- the stopping block with `b0 < nb`: block `b0` bails at `j0`
          have hbeq : blk'.val = b0 := le_antisymm hle (by omega)
          have hb0lt' : b0 < nb := by omega
          have hacc0 : matmat9_acc acts state k ds dotsF totals b0 totals' := by
            rw [← hbeq]; exact hacc
          obtain ⟨j0', hpre⟩ := hbail hb0lt' blk' hbeq totals' hacc0
          obtain ⟨T'', hinner, hacc'⟩ := matmat9_inner_bail acts k state blk' (ds blk'.val)
            (dotsF blk'.val) totals' j0' hpre
          have hb' : blk'.val < geom.blocks.val := by rw [← hnb]; exact hlt
          obtain ⟨b1, hb1, hb1v⟩ := WP.spec_imp_exists
            (UScalar.add_spec (x := blk') (y := 1#usize) (by scalar_tac))
          rw [matmat9_body_cont row geom acts k state avx totals' blk' (offF blk'.val)
            (off1F blk'.val) (i4F blk'.val) b1 (ds blk'.val) (dotsF blk'.val) T''
            (some kern.matmat.RowOutcome.Bail) hb' (hoff blk' hlt) (hoff1 blk' hlt)
            (hoff2 blk' hlt) (ho0 blk' hlt) (ho1 blk' hlt) (hdeq blk' hlt)
            (hdots blk' hlt) hinner hb1]
          simp only [WP.spec_ok]
          have hone : (1#usize).val = 1 := by decide
          have hb1v' : b1.val = blk'.val + 1 := by rw [hb1v, hone]
          have hle1 : b1.val = b0 + 1 := by rw [hb1v', hbeq]
          have hmeas : nb - b1.val < nb - blk'.val := by omega
          refine ⟨Or.inr ⟨hb0lt', j0', rfl, hle1, ?_⟩, hmeas⟩
          rw [← hbeq]
          exact matmat9_bailref_step acts state k ds dotsF totals totals' T'' blk' j0'
            hacc hacc'
      · -- the cursor reached `nb`: the loop is finished
        have hcke : blk'.val = nb := le_antisymm (by omega) (by omega)
        have hb' : ¬ blk'.val < geom.blocks.val := by rw [← hnb]; exact hlt
        rw [matmat9_body_done kern.matmat.Family.Q8_0 row geom acts k state avx totals' blk' hb']
        simp only [WP.spec_ok]
        refine Or.inl ⟨by omega, totals', rfl, ?_⟩
        rw [← hcke]
        exact hacc
    · -- an earlier block already stopped the row
      dsimp only at he hcur hbref
      simp only []
      rw [he]
      rw [matmat9_body_early kern.matmat.Family.Q8_0 row geom acts k state avx totals'
        kern.matmat.RowOutcome.Bail blk']
      simp only [WP.spec_ok]
      exact Or.inr ⟨totals', j0, rfl, hbref⟩
  · have h0v : (0#usize).val = 0 := by decide
    refine Or.inl ⟨rfl, by rw [h0v]; exact Nat.zero_le b0, ?_⟩
    rw [h0v]
    refine ⟨rfl, ?_⟩
    intro j hj
    simp only [matmat9_sum, add_zero]

/-! ## §5 · Non-vacuity -/

/-- The hypothesis bundle of `word_matmat9_row_accs_loop0_ok` is satisfiable: with a zero-block
geometry (`geom.blocks = 0`, so `nb = 0`), empty `row`/`acts`/`state` slices and `k = 0`, every
per-block hypothesis (`∀ blk, blk.val < 0 → …`) is vacuous and the length bounds are `0 ≤ _`. -/
example : ∃ (row : Slice Std.U8) (geom : kern.matmat.Geom) (acts : Slice chart.QuantAct)
    (k : Std.Usize) (state : Slice kern.matmat.ColumnState) (avx : Bool)
    (totals : Array I128 16#usize) (ds : Nat → I64) (dotsF : Nat → Array I32 16#usize)
    (offF off1F i4F : Nat → Std.Usize)
    (ho0 : ∀ blk : Std.Usize, blk.val < (0 : Nat) → (offF blk.val).val < row.val.length)
    (ho1 : ∀ blk : Std.Usize, blk.val < (0 : Nat) → (off1F blk.val).val < row.val.length),
    geom.blocks.val = 0 ∧
    (∀ blk : Std.Usize, blk.val < (0 : Nat) → blk * geom.block_bytes = ok (offF blk.val)) ∧
    (∀ blk : Std.Usize, blk.val < (0 : Nat) → offF blk.val + 1#usize = ok (off1F blk.val)) ∧
    (∀ blk : Std.Usize, blk.val < (0 : Nat) → offF blk.val + 2#usize = ok (i4F blk.val)) ∧
    (∀ blk : Std.Usize, blk.val < (0 : Nat) → (offF blk.val).val < row.val.length) ∧
    (∀ blk : Std.Usize, blk.val < (0 : Nat) → (off1F blk.val).val < row.val.length) ∧
    (∀ blk : Std.Usize, (hb : blk.val < (0 : Nat)) →
      dequant.f16_to_fixed (core.num.U16.from_le_bytes (Array.make 2#usize
        [row.val[(offF blk.val).val]'(ho0 blk hb), row.val[(off1F blk.val).val]'(ho1 blk hb)]))
        = ok (core.result.Result.Ok (ds blk.val))) ∧
    (∀ blk : Std.Usize, blk.val < (0 : Nat) →
      kern.matmat.q8_0_block_kdots avx row (i4F blk.val) acts blk k = ok (dotsF blk.val)) ∧
    (∀ blk : Std.Usize, blk.val < (0 : Nat) → ∀ T,
      matmat9_acc acts state k ds dotsF totals blk.val T →
        matmat9_clean acts state k blk (ds blk.val) (dotsF blk.val) T) ∧
    ((0 : Nat) < 0 → ∀ blk : Std.Usize, blk.val = (0 : Nat) → ∀ T,
      matmat9_acc acts state k ds dotsF totals (0 : Nat) T →
        ∃ j0, matmat9_bailpre acts state k blk (ds blk.val) (dotsF blk.val) T j0) ∧
    k.val ≤ (16#usize).val ∧ k.val ≤ state.val.length ∧ k.val ≤ acts.val.length := by
  refine ⟨Slice.from ([] : List Std.U8) (by scalar_tac),
    { blocks := 0#usize, block_bytes := 0#usize, row_bytes := 0#usize },
    Slice.from ([] : List chart.QuantAct) (by scalar_tac), 0#usize,
    Slice.from ([] : List kern.matmat.ColumnState) (by scalar_tac), false,
    Array.repeat 16#usize 0#i128, (fun _ => 0#i64), (fun _ => Array.repeat 16#usize 0#i32),
    (fun _ => 0#usize), (fun _ => 0#usize), (fun _ => 0#usize),
    (fun blk hb => absurd hb (Nat.not_lt_zero blk.val)),
    (fun blk hb => absurd hb (Nat.not_lt_zero blk.val)),
    by decide, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, by decide, by simp, by simp⟩
  · intro blk hb; exact absurd hb (Nat.not_lt_zero blk.val)
  · intro blk hb; exact absurd hb (Nat.not_lt_zero blk.val)
  · intro blk hb; exact absurd hb (Nat.not_lt_zero blk.val)
  · intro blk hb; exact absurd hb (Nat.not_lt_zero blk.val)
  · intro blk hb; exact absurd hb (Nat.not_lt_zero blk.val)
  · intro blk hb; exact absurd hb (Nat.not_lt_zero blk.val)
  · intro blk hb; exact absurd hb (Nat.not_lt_zero blk.val)
  · intro blk hb; exact absurd hb (Nat.not_lt_zero blk.val)
  · intro h; exact absurd h (Nat.lt_irrefl 0)

end apeiron_word
