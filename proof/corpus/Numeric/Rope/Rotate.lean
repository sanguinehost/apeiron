/-
The outer `rope.RopeTable.rotate_loop` over heads, its refusal arms, and a two-head success witness
.

`Numeric.Rope.HeadContent` carries the written values through one head (`rotate_head_content`,
`rotPrefix`, `RotHeadFits`). This module covers the outer loop of `RopeTable::rotate`
(`src/engine/rope.rs` lines 149-159), which walks `heads = q.len / head_dim` windows of width
`head_dim`, rotates the `q` head and then the `k` head at each window, and stops at the first
arithmetic refusal:

* `headWin`, `rotHeadsState` — the closed-form reference: the head window and the accumulated list
  after `h` heads, each of the `h` windows replaced by its per-head `rotPrefix` content;
* `rotate_loop_spec` — on the success domain where every head window satisfies the
  `rotate_head_content` hypotheses, `rotate_loop` returns the flag `true` and the two slices denote
  the concatenation of the per-head `rotPrefix` contents;
* `rotate_refuses` — the exact `VectorLength` and `OutOfRange` refusal equalities of `rotate` and the
  `OutOfRange` refusal arm of `rotate_head`, with the hypotheses forcing each;
* `rope_two_head_witness` — a two-head table (`head_dim = 2`) reusing `RopeHeadWitness`, exhibiting
  the success domain at once.

Scope: integer content and refusal shape of the outer loop. rustc/LLVM codegen, the kernel ABI and
the `unsafe` container are outside this module.
-/
import Numeric.Rope.HeadWitness

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Closed-form reference for the outer loop

The outer loop replaces the head window `[h·hd, (h+1)·hd)` of `q` by the `rotPrefix` content of its
original entries. `rotHeadsState` accumulates that replacement `setSlice!`-wise, so the update step
of the loop is its definitional unfolding. -/

/-- The `h`-th head window of a flat list: `hd` entries starting at `h·hd`. -/
def headWin (x : List I64) (hd h : ℕ) : List I64 := (x.drop (h * hd)).take hd

/-- The content of a flat head list after `h` of its `hd`-wide head windows have been replaced by
their `rotPrefix` content. The stride is `hd`; the loop walks it. -/
def rotHeadsState (x : List I64) (hd d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) : ℕ → List ℤ
  | 0 => x.map (fun a => a.val)
  | (h + 1) =>
      (rotHeadsState x hd d2 scAt h).setSlice! (h * hd)
        (rotPrefix (headWin x hd h) d2 scAt d2)

/-- `rotHeadsState` preserves length. -/
theorem rotHeadsState_length (x : List I64) (hd d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (h : ℕ) :
    (rotHeadsState x hd d2 scAt h).length = x.length := by
  induction h with
  | zero => simp [rotHeadsState]
  | succ h ih => rw [rotHeadsState, List.length_setSlice!, ih]

/-- Dropping the `i + |s'|` prefix of a `setSlice!` is dropping it of the original. -/
theorem drop_setSlice!_eq {α : Type} (s s' : List α) (i : ℕ) :
    (s.setSlice! i s').drop (i + s'.length) = s.drop (i + s'.length) := by
  apply List.ext_getElem?
  intro j
  rw [List.getElem?_drop, List.getElem?_drop]
  rw [List.setSlice!_getElem?_suffix _ _ _ _ (by omega)]

/-- `headWin` has length `hd` while the window stays inside the list. -/
theorem headWin_length (x : List I64) (hd h : ℕ) (hle : (h + 1) * hd ≤ x.length) :
    (headWin x hd h).length = hd := by
  unfold headWin
  rw [List.length_take, List.length_drop]
  apply Nat.min_eq_left
  apply Nat.le_sub_of_add_le
  have e : (h + 1) * hd = hd + h * hd := by ring
  rw [e] at hle
  exact hle

/-- `RotHeadFits` read at a list rather than a `Slice`. -/
def RotHeadFitsL (x : List I64) (d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) : Prop :=
  ∀ j, j < d2 →
    I64.min ≤ (ropeHeadPair x d2 scAt j).1 ∧ (ropeHeadPair x d2 scAt j).1 ≤ I64.max ∧
    I64.min ≤ (ropeHeadPair x d2 scAt j).2 ∧ (ropeHeadPair x d2 scAt j).2 ≤ I64.max

/-- `RotHeadFitsL` is `RotHeadFits` on the slice's values. -/
theorem RotHeadFitsL_slice (s : Slice I64) (d2 : Usize) (scAt : ℕ → fixed_trig.SinCos) :
    RotHeadFitsL s.val d2.val scAt ↔ RotHeadFits s d2 scAt := Iff.rfl

/-! ## Refusal arms

`RopeTable::rotate` has three refusal exits: the two static `VectorLength` guards and the propagated
`OutOfRange` of either head rotation. `rotate_head` has one: its inner loop's first arithmetic
refusal returns the flag `false`, which `rotate_head` maps to `OutOfRange`. -/

/-- Unequal `q`/`k` lengths: the first static guard, both slices unchanged. -/
theorem rotate_refuses_vector_length_ne (table : rope.RopeTable) (q k : Slice I64) (pos : U32)
    (h : Slice.len q ≠ Slice.len k) :
    rope.RopeTable.rotate table q k pos =
      ok (core.result.Result.Err rope.RopeError.VectorLength, q, k) :=
  rope_rotate_fixed_vector_length q k pos table h

/-- Length not a multiple of `head_dim`: the second static guard, both slices unchanged. -/
theorem rotate_refuses_vector_length_mod (table : rope.RopeTable) (q k : Slice I64) (pos : U32)
    (hlen : Slice.len q = Slice.len k) (h : ¬ (Slice.len q).val % table.head_dim.val = 0) :
    rope.RopeTable.rotate table q k pos =
      ok (core.result.Result.Err rope.RopeError.VectorLength, q, k) :=
  rope_rotate_fixed_head_dim q k pos table hlen h

/-- A `false` loop flag: the propagated `OutOfRange`, with the mutated slices. -/
theorem rotate_refuses_out_of_range (table : rope.RopeTable) (q k : Slice I64) (pos : U32)
    (heads : Usize) (q' k' : Slice I64)
    (hlen : Slice.len q = Slice.len k)
    (hmult : core.num.Usize.is_multiple_of (Slice.len q) table.head_dim = ok true)
    (hdiv : Slice.len q / table.head_dim = ok heads)
    (hloop : rope.RopeTable.rotate_loop table table.head_dim table.turns q k pos heads 0#usize
      true = ok (q', k', false)) :
    rope.RopeTable.rotate table q k pos =
      ok (core.result.Result.Err rope.RopeError.OutOfRange, q', k') := by
  unfold rope.RopeTable.rotate
  dsimp only
  rw [if_neg (by rw [bne_iff_ne]; exact fun hc => hc hlen)]
  rw [hmult]
  simp only [bind_tc_ok]
  rw [if_pos trivial]
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  change (if (false = true) then
      ok (core.result.Result.Ok (), q', k')
    else ok (core.result.Result.Err rope.RopeError.OutOfRange, q', k')) =
    ok (core.result.Result.Err rope.RopeError.OutOfRange, q', k')
  rw [if_neg (by decide : ¬ (false = true))]

/-- The `rotate_head` refusal arm: a `false` loop flag is `OutOfRange` with the mutated slice. -/
theorem rotate_head_refuses (self : rope.RopeTable) (x : Slice I64) (pos : U32) (d2 : Usize)
    (x' : Slice I64) (hd2 : self.head_dim / 2#usize = ok d2)
    (h : rope.RopeTable.rotate_head_loop self.turns x pos d2 0#usize true = ok (x', false)) :
    rope.RopeTable.rotate_head self x pos =
      ok (core.result.Result.Err rope.RopeError.OutOfRange, x') :=
  rotate_head_refusal_eq self x pos d2 x' hd2 h

/-- **Refusals.** The three `rotate` exits and the `rotate_head` exit as equalities, each with the
hypotheses that force it. -/
theorem rotate_refuses :
    (∀ (table : rope.RopeTable) (q k : Slice I64) (pos : U32),
      Slice.len q ≠ Slice.len k →
      rope.RopeTable.rotate table q k pos =
        ok (core.result.Result.Err rope.RopeError.VectorLength, q, k)) ∧
    (∀ (table : rope.RopeTable) (q k : Slice I64) (pos : U32),
      Slice.len q = Slice.len k → ¬ (Slice.len q).val % table.head_dim.val = 0 →
      rope.RopeTable.rotate table q k pos =
        ok (core.result.Result.Err rope.RopeError.VectorLength, q, k)) ∧
    (∀ (table : rope.RopeTable) (q k : Slice I64) (pos : U32) (heads : Usize)
      (q' k' : Slice I64),
      Slice.len q = Slice.len k →
      core.num.Usize.is_multiple_of (Slice.len q) table.head_dim = ok true →
      Slice.len q / table.head_dim = ok heads →
      rope.RopeTable.rotate_loop table table.head_dim table.turns q k pos heads 0#usize true =
        ok (q', k', false) →
      rope.RopeTable.rotate table q k pos =
        ok (core.result.Result.Err rope.RopeError.OutOfRange, q', k')) ∧
    (∀ (table : rope.RopeTable) (x : Slice I64) (pos : U32) (d2 : Usize) (x' : Slice I64),
      table.head_dim / 2#usize = ok d2 →
      rope.RopeTable.rotate_head_loop table.turns x pos d2 0#usize true = ok (x', false) →
      rope.RopeTable.rotate_head table x pos =
        ok (core.result.Result.Err rope.RopeError.OutOfRange, x')) :=
  ⟨fun _ _ _ _ h => rotate_refuses_vector_length_ne _ _ _ _ h,
   fun _ _ _ _ hlen h => rotate_refuses_vector_length_mod _ _ _ _ hlen h,
   fun _ _ _ _ _ _ _ hlen hmult hdiv hloop =>
     rotate_refuses_out_of_range _ _ _ _ _ _ _ hlen hmult hdiv hloop,
   fun _ _ _ _ _ hd2 hhead => rotate_head_refuses _ _ _ _ _ hd2 hhead⟩

/-! ## Outer loop

The loop state is `(q1, k1, i, ok)`. The invariant carries, at head `i`: the flag is still `true`,
the accumulated `ℤ` content of `q1`/`k1` is `rotHeadsState` at `i`, and — because `setSlice!` only
rewrites the processed prefix — the unprocessed suffix of each slice is still the original `I64`
list, which is what makes the current window literally `headWin`. -/

/-- Loop invariant at state `(q1, k1, i, ok)`. -/
def RotateLoopInv (hd d2 : Usize) (scAt : ℕ → fixed_trig.SinCos)
    (q k : Slice I64) (heads : Usize) (st : Slice I64 × Slice I64 × Usize × Bool) : Prop :=
  st.2.2.2 = true ∧
  st.2.2.1.val ≤ heads.val ∧
  st.1.val.map (fun a => a.val) = rotHeadsState q.val hd.val d2.val scAt st.2.2.1.val ∧
  st.2.1.val.map (fun a => a.val) = rotHeadsState k.val hd.val d2.val scAt st.2.2.1.val ∧
  st.1.val.drop (st.2.2.1.val * hd.val) = q.val.drop (st.2.2.1.val * hd.val) ∧
  st.2.1.val.drop (st.2.2.1.val * hd.val) = k.val.drop (st.2.2.1.val * hd.val)

/-- `List.slice` of a head window is `headWin`. -/
theorem slice_eq_headWin (x : List I64) (hd h : ℕ) :
    List.slice (h * hd) (h * hd + hd) x = headWin x hd h := by
  unfold List.slice headWin
  rw [show h * hd + hd - h * hd = hd by omega]

/-- Checked `usize` multiplication with the exact product. -/
theorem rr_usize_mul (x y : Usize) (h : x.val * y.val ≤ Usize.max) :
    ∃ z : Usize, x * y = ok z ∧ z.val = x.val * y.val := by
  obtain ⟨z, hz, hzv, -⟩ := Aeneas.Std.WP.spec_imp_exists
    (Usize.mul_bv_spec (x := x) (y := y) h)
  exact ⟨z, hz, hzv⟩

/-- Checked `usize` addition with the exact sum. -/
theorem rr_usize_add (x y : Usize) (h : x.val + y.val ≤ Usize.max) :
    ∃ z : Usize, x + y = ok z ∧ z.val = x.val + y.val := by
  obtain ⟨z, hz, hzv, -⟩ := Aeneas.Std.WP.spec_imp_exists
    (Usize.add_bv_spec (x := x) (y := y) h)
  exact ⟨z, hz, hzv⟩

/-- **Body step.** Under the success-domain hypotheses every arithmetic branch succeeds, so one
iteration replaces the head windows of `q1` and `k1` by their `rotPrefix` content and advances the
index. -/
private theorem rotate_loop_body_spec (self : rope.RopeTable) (d2 : Usize) (pos : U32)
    (heads : Usize) (q k : Slice I64) (scAt : ℕ → fixed_trig.SinCos)
    (hhd : 0 < self.head_dim.val)
    (hlenq : q.val.length = self.head_dim.val * heads.val)
    (hlenk : k.val.length = self.head_dim.val * heads.val)
    (hd2 : self.head_dim / 2#usize = ok d2)
    (htrig : ∀ i, i < d2.val → ∃ t, RopeTrigAt self.turns pos i t (scAt i))
    (hqfit : ∀ hh, hh < heads.val → RotHeadFitsL (headWin q.val self.head_dim.val hh) d2.val scAt)
    (hkfit : ∀ hh, hh < heads.val → RotHeadFitsL (headWin k.val self.head_dim.val hh) d2.val scAt)
    (st : Slice I64 × Slice I64 × Usize × Bool)
    (hst : RotateLoopInv self.head_dim d2 scAt q k heads st) :
    rope.RopeTable.rotate_loop.body self self.head_dim self.turns pos heads
        st.1 st.2.1 st.2.2.1 st.2.2.2 ⦃ r =>
      match r with
      | .done y => y.2.2 = true ∧
          y.1.val.map (fun a => a.val) =
            rotHeadsState q.val self.head_dim.val d2.val scAt heads.val ∧
          y.2.1.val.map (fun a => a.val) =
            rotHeadsState k.val self.head_dim.val d2.val scAt heads.val
      | .cont st' => RotateLoopInv self.head_dim d2 scAt q k heads st' ∧
          heads.val - st'.2.2.1.val < heads.val - st.2.2.1.val ⦄ := by
  rcases st with ⟨q1, k1, hi, oki⟩
  obtain ⟨hok, hle, hmapq, hmapk, hdropq, hdropk⟩ := hst
  dsimp only at hok hle hmapq hmapk hdropq hdropk
  have hd2v : d2.val = self.head_dim.val / 2 := by
    obtain ⟨z, hz, hzv⟩ := UScalar.div_spec self.head_dim (y := 2#usize) (by decide)
    have hz_eq : z = d2 := Result.ok_injective (hz.symm.trans hd2)
    rw [← hz_eq]; exact hzv
  have hq1len : q1.val.length = q.val.length := by
    have := congrArg List.length hmapq
    rw [List.length_map, rotHeadsState_length] at this
    exact this
  have hk1len : k1.val.length = k.val.length := by
    have := congrArg List.length hmapk
    rw [List.length_map, rotHeadsState_length] at this
    exact this
  by_cases hlt : hi < heads
  · have hltv : hi.val < heads.val := (UScalar.lt_equiv hi heads).mp hlt
    -- window arithmetic
    have hlenq' : (Slice.len q).val = self.head_dim.val * heads.val := by
      rw [Slice.len_val]; exact hlenq
    have hdivq : self.head_dim.val ∣ (Slice.len q).val := ⟨heads.val, hlenq'⟩
    have hheadsq : heads.val = (Slice.len q).val / self.head_dim.val := by
      rw [hlenq', Nat.mul_div_cancel_left _ hhd]
    obtain ⟨hwb1, hwb2, hwb3⟩ :=
      rotate_window_arith q self.head_dim hi heads hhd hdivq hheadsq hlt
    have hlenk' : (Slice.len k).val = self.head_dim.val * heads.val := by
      rw [Slice.len_val]; exact hlenk
    have hdivk : self.head_dim.val ∣ (Slice.len k).val := ⟨heads.val, hlenk'⟩
    have hheadsk : heads.val = (Slice.len k).val / self.head_dim.val := by
      rw [hlenk', Nat.mul_div_cancel_left _ hhd]
    obtain ⟨_, _, hwbk3⟩ :=
      rotate_window_arith k self.head_dim hi heads hhd hdivk hheadsk hlt
    -- base, i
    obtain ⟨base, hbase, hbasev⟩ := rr_usize_mul hi self.head_dim hwb1
    have hbadd : base.val + self.head_dim.val ≤ Usize.max := by rw [hbasev]; exact hwb2
    obtain ⟨i, hiadd, hiv⟩ := rr_usize_add base self.head_dim hbadd
    have hivle : i.val ≤ (Slice.len q).val := by rw [hiv, hbasev]; exact hwb3
    have hivlek : i.val ≤ (Slice.len k).val := by rw [hiv, hbasev]; exact hwbk3
    have hivle1 : i.val ≤ q1.val.length := by rw [hq1len]; exact hivle
    have hivlek1 : i.val ≤ k1.val.length := by rw [hk1len]; exact hivlek
    have hile : base.val ≤ i.val := by rw [hiv]; omega
    -- index_mut q1
    obtain ⟨pq, hq_eq0, hq_spec⟩ := Aeneas.Std.WP.spec_imp_exists
      (core.slice.index.SliceIndexRangeUsizeSlice.index_mut.step_spec
        ({ start := base, «end» := i } : core.ops.range.Range Usize) q1
        ((UScalar.le_equiv base i).2 hile)
        (by simpa using hivle1))
    rcases pq with ⟨s, back⟩
    have hq_eq : core.slice.index.Slice.index_mut
        (core.slice.index.SliceIndexRangeUsizeSlice I64) q1
        { start := base, «end» := i } = ok (s, back) := by
      rw [Slice.index_mut_SliceIndexRangeUsizeSliceInst]; exact hq_eq0
    obtain ⟨hs_val, _, hs_back⟩ := hq_spec
    have hs_hw1 : s.val = headWin q1.val self.head_dim.val hi.val := by
      rw [hs_val, hiv, hbasev, slice_eq_headWin]
    have hwinq : headWin q1.val self.head_dim.val hi.val =
        headWin q.val self.head_dim.val hi.val := by
      unfold headWin; rw [hdropq]
    have hs_hw : s.val = headWin q.val self.head_dim.val hi.val := hs_hw1.trans hwinq
    have hfit_q : RotHeadFits s d2 scAt :=
      (RotHeadFitsL_slice s d2 scAt).mp (by rw [hs_hw]; exact hqfit hi.val hltv)
    have hsideq : (hi.val + 1) * self.head_dim.val ≤ q.val.length := by
      rw [hlenq, Nat.mul_comm self.head_dim.val heads.val]
      exact Nat.mul_le_mul_right self.head_dim.val (Nat.succ_le_of_lt hltv)
    have hslen_hd : s.val.length = self.head_dim.val := by
      rw [hs_hw, headWin_length q.val self.head_dim.val hi.val hsideq]
    have hx_q : 2 * d2.val ≤ s.length := by
      rw [Slice.length, hslen_hd, hd2v]
      rw [Nat.mul_comm]
      exact Nat.div_mul_le_self _ _
    obtain ⟨s1, hrot_q0, hcont_q⟩ :=
      rotate_head_content { head_dim := self.head_dim, turns := self.turns } s pos d2
        hd2 scAt htrig hfit_q hx_q
    have hrot_q : (rope.RopeTable.mk self.head_dim self.turns).rotate_head s pos =
        ok (core.result.Result.Ok (), s1) := hrot_q0
    have hs1len : s1.val.length = self.head_dim.val := by
      have := congrArg List.length hcont_q
      rw [List.length_map, rotPrefix_length, hslen_hd] at this
      exact this
    -- index_mut k1
    obtain ⟨pk, hk_eq0, hk_spec⟩ := Aeneas.Std.WP.spec_imp_exists
      (core.slice.index.SliceIndexRangeUsizeSlice.index_mut.step_spec
        ({ start := base, «end» := i } : core.ops.range.Range Usize) k1
        ((UScalar.le_equiv base i).2 hile)
        (by simpa using hivlek1))
    rcases pk with ⟨s3, back1⟩
    have hk_eq : core.slice.index.Slice.index_mut
        (core.slice.index.SliceIndexRangeUsizeSlice I64) k1
        { start := base, «end» := i } = ok (s3, back1) := by
      rw [Slice.index_mut_SliceIndexRangeUsizeSliceInst]; exact hk_eq0
    obtain ⟨hs3_val, _, hs3_back⟩ := hk_spec
    have hs3_hw1 : s3.val = headWin k1.val self.head_dim.val hi.val := by
      rw [hs3_val, hiv, hbasev, slice_eq_headWin]
    have hwink : headWin k1.val self.head_dim.val hi.val =
        headWin k.val self.head_dim.val hi.val := by
      unfold headWin; rw [hdropk]
    have hs3_hw : s3.val = headWin k.val self.head_dim.val hi.val := hs3_hw1.trans hwink
    have hfit_k : RotHeadFits s3 d2 scAt :=
      (RotHeadFitsL_slice s3 d2 scAt).mp (by rw [hs3_hw]; exact hkfit hi.val hltv)
    have hsidek : (hi.val + 1) * self.head_dim.val ≤ k.val.length := by
      rw [hlenk, Nat.mul_comm self.head_dim.val heads.val]
      exact Nat.mul_le_mul_right self.head_dim.val (Nat.succ_le_of_lt hltv)
    have hs3len_hd : s3.val.length = self.head_dim.val := by
      rw [hs3_hw, headWin_length k.val self.head_dim.val hi.val hsidek]
    have hx_k : 2 * d2.val ≤ s3.length := by
      rw [Slice.length, hs3len_hd, hd2v]
      rw [Nat.mul_comm]
      exact Nat.div_mul_le_self _ _
    obtain ⟨s4, hrot_k0, hcont_k⟩ :=
      rotate_head_content { head_dim := self.head_dim, turns := self.turns } s3 pos d2
        hd2 scAt htrig hfit_k hx_k
    have hrot_k : (rope.RopeTable.mk self.head_dim self.turns).rotate_head s3 pos =
        ok (core.result.Result.Ok (), s4) := hrot_k0
    have hs4len : s4.val.length = self.head_dim.val := by
      have := congrArg List.length hcont_k
      rw [List.length_map, rotPrefix_length, hs3len_hd] at this
      exact this
    -- the advance
    have hinc : hi.val + 1 ≤ Usize.max := by
      have h1 : hi.val + 1 ≤ heads.val := Nat.succ_le_of_lt hltv
      have h2 : heads.val ≤ Usize.max := by
        calc heads.val ≤ self.head_dim.val * heads.val := Nat.le_mul_of_pos_left _ hhd
          _ = (Slice.len q).val := hlenq'.symm
          _ ≤ Usize.max := Slice.length_ineq q
      omega
    obtain ⟨h1u, hinc_eq, h1uv⟩ := rr_usize_add hi 1#usize (by simpa using hinc)
    have h1v : h1u.val = hi.val + 1 := by simpa using h1uv
    -- body equality
    have hbody : rope.RopeTable.rotate_loop.body self self.head_dim self.turns pos heads
        q1 k1 hi true =
        ok (ControlFlow.cont
          (q1.setSlice! base.val s1, k1.setSlice! base.val s4, h1u, true)) := by
      unfold rope.RopeTable.rotate_loop.body
      dsimp only
      rw [if_pos hlt, if_pos rfl]
      rw [hbase]
      simp only [bind_tc_ok]
      rw [hiadd]
      simp only [bind_tc_ok]
      simp only [hq_eq, bind_tc_ok, uncurry_apply_pair]
      rw [hrot_q]
      simp only [bind_tc_ok, uncurry_apply_pair]
      simp only [hk_eq, bind_tc_ok, uncurry_apply_pair]
      rw [hrot_k]
      simp only [bind_tc_ok, uncurry_apply_pair]
      simp only [hinc_eq, bind_tc_ok]
      rw [hs_back s1, hs3_back s4]
    rw [hok, hbody]
    simp only [Aeneas.Std.WP.spec_ok]
    -- content of q1'
    have hq1c : (q1.setSlice! base.val s1).val.map (fun a => a.val) =
        (rotHeadsState q.val self.head_dim.val d2.val scAt hi.val).setSlice!
          (hi.val * self.head_dim.val)
          (rotPrefix (headWin q.val self.head_dim.val hi.val) d2.val scAt d2.val) := by
      rw [Slice.setSlice!_val, List.map_setSlice!, hmapq, hcont_q, hs_hw, hbasev]
    have hq1d : (q1.setSlice! base.val s1).val.drop (h1u.val * self.head_dim.val) =
        q.val.drop (h1u.val * self.head_dim.val) := by
      rw [Slice.setSlice!_val]
      have hidx : h1u.val * self.head_dim.val = base.val + s1.val.length := by
        rw [h1v, hbasev, hs1len]; ring
      rw [hidx, drop_setSlice!_eq]
      rw [show base.val + s1.val.length = (hi.val + 1) * self.head_dim.val by
        rw [hbasev, hs1len]; ring]
      rw [show (hi.val + 1) * self.head_dim.val =
        hi.val * self.head_dim.val + self.head_dim.val by ring]
      rw [← List.drop_drop, hdropq, List.drop_drop]
    have hk1c : (k1.setSlice! base.val s4).val.map (fun a => a.val) =
        (rotHeadsState k.val self.head_dim.val d2.val scAt hi.val).setSlice!
          (hi.val * self.head_dim.val)
          (rotPrefix (headWin k.val self.head_dim.val hi.val) d2.val scAt d2.val) := by
      rw [Slice.setSlice!_val, List.map_setSlice!, hmapk, hcont_k, hs3_hw, hbasev]
    have hk1d : (k1.setSlice! base.val s4).val.drop (h1u.val * self.head_dim.val) =
        k.val.drop (h1u.val * self.head_dim.val) := by
      rw [Slice.setSlice!_val]
      have hidx : h1u.val * self.head_dim.val = base.val + s4.val.length := by
        rw [h1v, hbasev, hs4len]; ring
      rw [hidx, drop_setSlice!_eq]
      rw [show base.val + s4.val.length = (hi.val + 1) * self.head_dim.val by
        rw [hbasev, hs4len]; ring]
      rw [show (hi.val + 1) * self.head_dim.val =
        hi.val * self.head_dim.val + self.head_dim.val by ring]
      rw [← List.drop_drop, hdropk, List.drop_drop]
    refine ⟨⟨rfl, by rw [h1v]; omega, ?_, ?_, ?_, ?_⟩, by rw [h1v]; omega⟩
    · rw [hq1c, h1v, rotHeadsState]
    · rw [hk1c, h1v, rotHeadsState]
    · rw [hq1d, h1v]
    · rw [hk1d, h1v]
  · have hhev : hi.val = heads.val := by
      have hnlt : ¬ hi.val < heads.val := fun h => hlt ((UScalar.lt_equiv hi heads).mpr h)
      omega
    have hbody : rope.RopeTable.rotate_loop.body self self.head_dim self.turns pos heads
        q1 k1 hi true = ok (ControlFlow.done (q1, k1, true)) := by
      unfold rope.RopeTable.rotate_loop.body
      dsimp only
      rw [if_neg hlt]
    rw [hok, hbody]
    simp only [Aeneas.Std.WP.spec_ok]
    exact ⟨trivial, by rw [← hhev]; exact hmapq, by rw [← hhev]; exact hmapk⟩

/-- **Outer loop.** On the success domain — `head_dim ∣ q.len = k.len`, `d2 = head_dim/2`, every
head carries its `RopeTrigAt` trig pair and `RotHeadFits` bounds — `rotate_loop` returns the flag
`true` and the two slices denote the concatenation of the per-head `rotPrefix` contents. -/
theorem rotate_loop_spec (self : rope.RopeTable) (pos : U32) (heads : Usize)
    (q k : Slice I64) (d2 : Usize) (scAt : ℕ → fixed_trig.SinCos)
    (hhd : 0 < self.head_dim.val)
    (hlenq : q.val.length = self.head_dim.val * heads.val)
    (hlenk : k.val.length = self.head_dim.val * heads.val)
    (hd2 : self.head_dim / 2#usize = ok d2)
    (htrig : ∀ i, i < d2.val → ∃ t, RopeTrigAt self.turns pos i t (scAt i))
    (hqfit : ∀ hh, hh < heads.val → RotHeadFitsL (headWin q.val self.head_dim.val hh) d2.val scAt)
    (hkfit : ∀ hh, hh < heads.val → RotHeadFitsL (headWin k.val self.head_dim.val hh) d2.val scAt) :
    ∃ q' k' : Slice I64,
      rope.RopeTable.rotate_loop self self.head_dim self.turns q k pos heads 0#usize true =
        ok (q', k', true) ∧
      q'.val.map (fun a => a.val) =
        rotHeadsState q.val self.head_dim.val d2.val scAt heads.val ∧
      k'.val.map (fun a => a.val) =
        rotHeadsState k.val self.head_dim.val d2.val scAt heads.val := by
  have hspec : loop
      (fun (s : Slice I64 × Slice I64 × Usize × Bool) =>
        rope.RopeTable.rotate_loop.body self self.head_dim self.turns pos heads
          s.1 s.2.1 s.2.2.1 s.2.2.2)
      (q, k, 0#usize, true) ⦃ r =>
        r.2.2 = true ∧
        r.1.val.map (fun a => a.val) =
          rotHeadsState q.val self.head_dim.val d2.val scAt heads.val ∧
        r.2.1.val.map (fun a => a.val) =
          rotHeadsState k.val self.head_dim.val d2.val scAt heads.val ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun st : Slice I64 × Slice I64 × Usize × Bool =>
        heads.val - st.2.2.1.val)
      (inv := RotateLoopInv self.head_dim d2 scAt q k heads)
      (post := fun r : Slice I64 × Slice I64 × Bool =>
        r.2.2 = true ∧
        r.1.val.map (fun a => a.val) =
          rotHeadsState q.val self.head_dim.val d2.val scAt heads.val ∧
        r.2.1.val.map (fun a => a.val) =
          rotHeadsState k.val self.head_dim.val d2.val scAt heads.val)
    · intro st hst
      convert rotate_loop_body_spec self d2 pos heads q k scAt hhd hlenq hlenk hd2 htrig
        hqfit hkfit st hst using 1
      funext r
      cases r with
      | done y => rfl
      | cont x' => rfl
    · refine ⟨rfl, by simp, ?_, ?_, ?_, ?_⟩
      · show q.val.map (fun a => a.val) = rotHeadsState q.val self.head_dim.val d2.val scAt
          (0#usize).val
        rfl
      · show k.val.map (fun a => a.val) = rotHeadsState k.val self.head_dim.val d2.val scAt
          (0#usize).val
        rfl
      · show q.val.drop ((0#usize).val * self.head_dim.val) =
          q.val.drop ((0#usize).val * self.head_dim.val)
        rfl
      · show k.val.drop ((0#usize).val * self.head_dim.val) =
          k.val.drop ((0#usize).val * self.head_dim.val)
        rfl
  obtain ⟨r, hr, hr2, hr1, hrk⟩ := Aeneas.Std.WP.spec_imp_exists hspec
  obtain ⟨a, b, c⟩ := r
  have hc : c = true := hr2
  refine ⟨a, b, ?_, hr1, hrk⟩
  rw [rope.RopeTable.rotate_loop, hr, hc]

/-- **Two-head witness.** A `head_dim = 2` table whose two heads are both the `RopeHeadWitness` head
`[2^32, 0]`: `rotate_loop_spec`'s success domain holds jointly, so the outer loop returns the flag
`true` and the concatenation of the two per-head `rotPrefix` contents. -/
def ropeWit2Head : Slice I64 :=
  Slice.from [4294967296#i64, 0#i64, 4294967296#i64, 0#i64] (by simp; scalar_tac)

/-- **Two-head success witness** reusing `RopeHeadWitness`. -/
theorem rope_two_head_witness :
    ∃ q' k' : Slice I64,
      rope.RopeTable.rotate_loop ropeWitTable ropeWitTable.head_dim ropeWitTable.turns
        ropeWit2Head ropeWit2Head ropeWitPos 2#usize 0#usize true = ok (q', k', true) ∧
      q'.val.map (fun a => a.val) =
        rotHeadsState ropeWit2Head.val ropeWitTable.head_dim.val ropeWitD2.val ropeWitScAt 2 ∧
      k'.val.map (fun a => a.val) =
        rotHeadsState ropeWit2Head.val ropeWitTable.head_dim.val ropeWitD2.val ropeWitScAt 2 := by
  have hd2 : ropeWitTable.head_dim / 2#usize = ok ropeWitD2 := rope_head_domain_witness.1
  have htrig : ∀ i, i < ropeWitD2.val →
      ∃ t, RopeTrigAt ropeWitTable.turns ropeWitPos i t (ropeWitScAt i) :=
    rope_head_domain_witness.2.1
  have hlen : ropeWit2Head.val.length = ropeWitTable.head_dim.val * 2 := by
    simp [ropeWit2Head, ropeWitTable, Slice.from_val]
  have hfit : ∀ hh, hh < 2 → RotHeadFitsL (headWin ropeWit2Head.val ropeWitTable.head_dim.val hh)
      ropeWitD2.val ropeWitScAt := by
    intro hh hh2
    have hwin : headWin ropeWit2Head.val ropeWitTable.head_dim.val hh = ropeWitHead.val := by
      have hh01 : hh = 0 ∨ hh = 1 := by omega
      rcases hh01 with rfl | rfl <;>
        simp [headWin, ropeWit2Head, ropeWitTable, ropeWitHead, Slice.from_val]
    rw [hwin]
    exact (RotHeadFitsL_slice ropeWitHead ropeWitD2 ropeWitScAt).mpr ropeWitTable_hfit
  have hhd : 0 < ropeWitTable.head_dim.val := by simp [ropeWitTable]
  exact rotate_loop_spec ropeWitTable ropeWitPos 2#usize ropeWit2Head ropeWit2Head ropeWitD2
    ropeWitScAt hhd hlen hlen hd2 htrig hfit hfit

end

end apeiron_word


