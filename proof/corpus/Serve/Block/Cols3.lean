/-
  `Serve/Cols3.lean` — refinement statements for the column loops of the projection drivers
  `step.cols.Engine.ffn_cols` (`src/engine/step/cols.rs:907`) and `step.cols.Engine.group_cols`
  (`src/engine/step/cols.rs:381`, `:423`), extracted into `proof/ApeironWord.lean`
  (module `ApeironWord`, namespace `apeiron_word`). Sibling `Serve/Cols.lean` and `Serve/Cols2.lean`
  carry the `step.cols.add_cols` kernel and its `cols_ivals`/`addPrefix` references, which are reused.

  ## §1 `step.cols.Engine.ffn_cols_loop2` — the FFN column buffer

  Lines 907:22–908:45 of `cols.rs`:

      while c < xs.len() {
          let v = vec![0i64; n_ff];
          hs.push(v);
          c += 1;
      }

  `n_ff` columns of `xs` each receive a fresh `n_ff`-long zero column. The loop has exactly one
  partial operation: `Vec::push`, which refuses with `Error.maximumSizeExceeded` when the accumulator
  length has reached `Usize::max` (`alloc.vec.Vec.push`, `Vec.lean:243`). References:

  * `cols3_zeroCol n` — the zero column of width `n`;
  * `cols3_colvals hs` — the column values of a vector of columns (one `List ℤ` per column);
  * `cols3_ffnCols n k` — `k` zero columns of width `n`.

  ## §2 `step.cols.Engine.group_cols_loop2` — payloads to `GemmPart`

  Lines 423:16–431:17: each payload `(ggml_type, at, n_out)` of the payload vector is pushed as one
  `kern.gemm.GemmPart`. Reference: `cols3_takeParts payloads i k` — the parts at indices
  `i, …, i+k-1`.

  ## Boundary

  The loop bodies proved here call only `Vec::push`, `Vec::from_elem`, `Vec::index`,
  `Slice::index_usize` and `Usize::add`, whose bodies are in the model. The §1 headlines quantify
  over `self : loader.Engine`, whose `pool`/`Arc`/`OnceLock` fields make them mention the extraction
  boundary axioms `alloc.sync.Arc`, `chart.pool.Excl`, `std.sync.once_lock.OnceLock`
  (`proof/boundary-axioms.txt`); the proof never calls them, only the type does. The §2 headlines are
  free of them.

  The driver loops whose callees are the opaque boundaries (`step.cols.Engine.ffn_single_sched`,
  `step.Engine.ffn_q8act_at`, `step.spec.Engine.embed_push`, `step.cols.Engine.layer_cols`,
  `step.LayerIndex.at`, `step.Engine.payload_at`) are stated conditionally, with each boundary call's
  result as a hypothesis; they are not reached by the headlines below.

  Trust boundary: nothing here covers rustc/LLVM codegen, the `unsafe` pointer writes, the
  `F32`/`F64` carriers, the AVX2 lanes, `pool`/`Arc`/`Excl`, or the opaque boundaries named above.
-/

import Serve.Block.Cols2

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow

/-! ## §0 · Reference definitions and scalar value lemmas -/

/-- The zero column of width `n`: the reference for `Vec::from_elem 0 n`. -/
def cols3_zeroCol (n : ℕ) : List ℤ := List.replicate n 0

/-- The column values of a vector of columns: one `List ℤ` per column. -/
def cols3_colvals (hs : alloc.vec.Vec (alloc.vec.Vec Std.I64)) : List (List ℤ) :=
  hs.val.map (fun v => cols_ivals v.slice)

theorem cols3_colvals_length (hs : alloc.vec.Vec (alloc.vec.Vec Std.I64)) :
    (cols3_colvals hs).length = hs.val.length := by
  simp [cols3_colvals]

/-- `k` zero columns of width `n`: the reference the FFN column loop appends to a buffer. -/
def cols3_ffnCols (n : ℕ) (k : ℕ) : List (List ℤ) := List.replicate k (cols3_zeroCol n)

theorem cols3_ffnCols_length (n k : ℕ) : (cols3_ffnCols n k).length = k := by
  simp [cols3_ffnCols]

/-- One more column: the reference of one `Vec::push` turn. -/
theorem cols3_ffnCols_succ (n k : ℕ) :
    cols3_ffnCols n (k + 1) = cols3_ffnCols n k ++ [cols3_zeroCol n] := by
  simp [cols3_ffnCols, List.replicate_succ']

/-- The reference of one `Vec::push` turn on the column buffer. -/
theorem cols3_colvals_push (hs1 hs : alloc.vec.Vec (alloc.vec.Vec Std.I64))
    (v : alloc.vec.Vec Std.I64) (h : hs1.val = hs.val ++ [v]) :
    cols3_colvals hs1 = cols3_colvals hs ++ [cols_ivals v.slice] := by
  simp only [cols3_colvals, h, List.map_append, List.map_cons, List.map_nil]

theorem cols3_usize_zero_val : ((0#usize : Std.Usize)).val = 0 := by rfl

theorem cols3_usize_one_val : ((1#usize : Std.Usize)).val = 1 := by rfl

theorem cols3_i64_zero_val : ((0#i64 : Std.I64)).val = 0 := by simp

/-- The capacity value of `Vec::with_capacity`. -/
theorem cols3_with_capacity_val (T : Type) (n : Std.Usize) :
    (alloc.vec.Vec.with_capacity T n).val = ([] : List T) := by
  simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val]

/-- `Vec::from_elem 0 n` is the zero column of width `n`. -/
theorem cols3_from_elem_zeros (n : Std.Usize) :
    ∃ v : alloc.vec.Vec Std.I64,
      alloc.vec.from_elem core.clone.CloneI64 0#i64 n = ok v ∧
      cols_ivals v.slice = cols3_zeroCol n.val := by
  obtain ⟨v, hv, hval, _⟩ := WP.spec_imp_exists
    (alloc.vec.from_elem_spec (T := Std.I64) core.clone.CloneI64 0#i64 n (by simp))
  refine ⟨v, hv, ?_⟩
  rw [show cols_ivals v.slice = (v.val).map (fun z : Std.I64 => z.val) from rfl, hval,
    List.map_replicate, cols3_zeroCol, cols3_i64_zero_val]

/-! ## §1 · `step.cols.Engine.ffn_cols_loop2` -/

/-- **One `ffn_cols_loop2` body turn, continue.** `c` is below the column count and the accumulator
is below the maximum length: the turn pushes one `n_ff`-long zero column. -/
theorem cols3_ffn_cols_loop2_body_cont (self : loader.Engine) (xs : Slice (alloc.vec.Vec Std.I64))
    (hs : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (hc : c.val < xs.val.length) (hcap : hs.val.length < Std.Usize.max) :
    ∃ hs1 : alloc.vec.Vec (alloc.vec.Vec Std.I64), ∃ c1 : Std.Usize,
      step.cols.Engine.ffn_cols_loop2.body self xs hs c = ok (ControlFlow.cont (hs1, c1)) ∧
      c1.val = c.val + 1 ∧
      cols3_colvals hs1 = cols3_colvals hs ++ [cols3_zeroCol self.config.n_ff.val] := by
  have hmax : c.val + (1#usize).val ≤ Std.Usize.max := by
    have := Slice.length_ineq xs
    rw [cols3_usize_one_val]
    omega
  unfold step.cols.Engine.ffn_cols_loop2.body
  simp only []
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hc)]
  obtain ⟨v, hv, hvals⟩ := cols3_from_elem_zeros self.config.n_ff
  rw [hv]
  simp only [bind_tc_ok]
  obtain ⟨hs1, hpush, hpushval⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec hs v hcap)
  rw [hpush]
  simp only [bind_tc_ok]
  obtain ⟨c1, hc1, hc1val⟩ := WP.spec_imp_exists (Usize.add_spec (x := c) (y := 1#usize) hmax)
  rw [hc1]
  simp only [bind_tc_ok]
  refine ⟨hs1, c1, rfl, ?_, ?_⟩
  · rw [hc1val, cols3_usize_one_val]
  · rw [cols3_colvals_push hs1 hs v hpushval, hvals]

/-- **One `ffn_cols_loop2` body turn, past the end.** The cursor has reached the column count: the
turn returns the accumulator unchanged. -/
theorem cols3_ffn_cols_loop2_body_done (self : loader.Engine) (xs : Slice (alloc.vec.Vec Std.I64))
    (hs : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (hc : ¬ c.val < xs.val.length) :
    step.cols.Engine.ffn_cols_loop2.body self xs hs c = ok (ControlFlow.done hs) := by
  unfold step.cols.Engine.ffn_cols_loop2.body
  simp only []
  rw [if_neg (by rw [UScalar.lt_equiv, Slice.len_val]; exact hc)]

/-- **The capacity check of `Vec::push`, refused.** An accumulator at the maximum length fails the
`Vec::push` width test, so the body (whose only partial operation it is) returns
`Error.maximumSizeExceeded`. -/
theorem cols3_ffn_cols_loop2_body_refuses (self : loader.Engine) (xs : Slice (alloc.vec.Vec Std.I64))
    (hs : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (hc : c.val < xs.val.length) (hmax : hs.val.length = Std.Usize.max) :
    step.cols.Engine.ffn_cols_loop2.body self xs hs c = Result.fail Error.maximumSizeExceeded := by
  have hpush_fail : ∀ v : alloc.vec.Vec Std.I64,
      hs.val.length = Std.Usize.max → alloc.vec.Vec.push hs v = Result.fail Error.maximumSizeExceeded := by
    intro v hv
    have hno : ¬ ((hs.val.length + 1) ≤ U32.max || (hs.val.length + 1) ≤ Usize.max) := by
      rw [hv]
      have h1 : ¬ (Usize.max + 1 ≤ U32.max) := by
        rcases Usize.bounds_eq with hb | hb
        · rw [hb, U32.max_eq]; norm_num
        · rw [hb, U32.max_eq, U64.max_eq]; norm_num
      have h2 : ¬ (Usize.max + 1 ≤ Usize.max) := by omega
      intro hc2
      simp only [Bool.or_eq_true, decide_eq_true_eq] at hc2
      exact hc2.elim h1 h2
    unfold alloc.vec.Vec.push
    simp only []
    split
    · rename_i hb
      exact absurd hb hno
    · rfl
  unfold step.cols.Engine.ffn_cols_loop2.body
  simp only []
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hc)]
  obtain ⟨v, hv, _⟩ := cols3_from_elem_zeros self.config.n_ff
  rw [hv]
  simp only [bind_tc_ok]
  rw [hpush_fail v hmax]
  simp only [bind_tc_fail]

/-- **The `ffn_cols_loop2` loop, from any cursor at or below the column count.** Every appended
column is a zero column of width `n_ff`; the accumulator's columns are the input's followed by
`xs.len - c` zero columns. -/
theorem cols3_ffn_cols_loop2_spec (self : loader.Engine) (xs : Slice (alloc.vec.Vec Std.I64))
    (hs : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (hc : c.val ≤ xs.val.length)
    (hcap : hs.val.length + (xs.val.length - c.val) ≤ Std.Usize.max) :
    step.cols.Engine.ffn_cols_loop2 self xs hs c ⦃ o =>
      cols3_colvals o =
        cols3_colvals hs ++ cols3_ffnCols self.config.n_ff.val (xs.val.length - c.val) ⦄ := by
  unfold step.cols.Engine.ffn_cols_loop2
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec (alloc.vec.Vec Std.I64) × Std.Usize =>
      xs.val.length - st.2.val)
    (inv := fun st => c.val ≤ st.2.val ∧ st.2.val ≤ xs.val.length ∧
      cols3_colvals st.1 = cols3_colvals hs ++
        cols3_ffnCols self.config.n_ff.val (st.2.val - c.val))
    (post := fun o : alloc.vec.Vec (alloc.vec.Vec Std.I64) =>
      cols3_colvals o = cols3_colvals hs ++
        cols3_ffnCols self.config.n_ff.val (xs.val.length - c.val))
  · rintro ⟨hs1, c1⟩ ⟨hle1, hle2, hinv⟩
    dsimp only at hle1 hle2 hinv
    by_cases hlt : c1.val < xs.val.length
    · have hcap1 : hs1.val.length < Std.Usize.max := by
        have hlen : hs1.val.length = hs.val.length + (c1.val - c.val) := by
          have h := congrArg List.length hinv
          rw [cols3_colvals_length, List.length_append, cols3_colvals_length,
            cols3_ffnCols_length] at h
          exact h
        rw [hlen]
        omega
      obtain ⟨hs2, c2, hbody, hc2val, hvals2⟩ :=
        cols3_ffn_cols_loop2_body_cont self xs hs1 c1 hlt hcap1
      rw [WP.spec_equiv_exists]
      refine ⟨_, hbody, ?_⟩
      dsimp only
      refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
      · rw [hc2val]; omega
      · rw [hc2val]; omega
      · have hidx : c2.val - c.val = (c1.val - c.val) + 1 := by rw [hc2val]; omega
        rw [hvals2, hinv, hidx, cols3_ffnCols_succ, List.append_assoc]
      · rw [hc2val]; omega
    · have hge : c1.val = xs.val.length := le_antisymm hle2 (Nat.le_of_not_lt hlt)
      have hbody := cols3_ffn_cols_loop2_body_done self xs hs1 c1 hlt
      rw [WP.spec_equiv_exists]
      refine ⟨_, hbody, ?_⟩
      dsimp only
      rw [hinv, hge]
  · refine ⟨le_refl _, hc, ?_⟩
    simp [cols3_ffnCols]

/-- **Success law.** From a `Slice.len xs`-capacity buffer at cursor zero, the extracted
`step.cols.Engine.ffn_cols_loop2` returns one zero column of width `n_ff` per input column. -/
theorem word_ffn_cols_loop2_ok (self : loader.Engine) (xs : Slice (alloc.vec.Vec Std.I64)) :
    ∃ hs : alloc.vec.Vec (alloc.vec.Vec Std.I64),
      step.cols.Engine.ffn_cols_loop2 self xs
        (alloc.vec.Vec.with_capacity (alloc.vec.Vec Std.I64) (Slice.len xs)) 0#usize = ok hs ∧
      cols3_colvals hs = cols3_ffnCols self.config.n_ff.val xs.val.length := by
  have hcap0 : (alloc.vec.Vec.with_capacity (alloc.vec.Vec Std.I64) (Slice.len xs)).val.length +
      (xs.val.length - (0#usize).val) ≤ Std.Usize.max := by
    have hp := Slice.length_ineq xs
    rw [cols3_usize_zero_val, cols3_with_capacity_val, List.length_nil]
    omega
  obtain ⟨hs, ho, hpost⟩ := WP.spec_imp_exists
    (cols3_ffn_cols_loop2_spec self xs
      (alloc.vec.Vec.with_capacity (alloc.vec.Vec Std.I64) (Slice.len xs)) 0#usize
      (by rw [cols3_usize_zero_val]; exact Nat.zero_le _) hcap0)
  refine ⟨hs, ho, ?_⟩
  rw [hpost, cols3_usize_zero_val, Nat.sub_zero]
  simp only [cols3_colvals, cols3_with_capacity_val, List.map_nil, List.nil_append]

/-- **Refusal law.** At a cursor below the column count and an accumulator already at
`Usize.max` length, the extracted `step.cols.Engine.ffn_cols_loop2` body fails with
`Error.maximumSizeExceeded` and appends nothing. -/
theorem word_ffn_cols_loop2_refuses (self : loader.Engine) (xs : Slice (alloc.vec.Vec Std.I64))
    (hs : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (hc : c.val < xs.val.length) (hmax : hs.val.length = Std.Usize.max) :
    step.cols.Engine.ffn_cols_loop2.body self xs hs c = Result.fail Error.maximumSizeExceeded :=
  cols3_ffn_cols_loop2_body_refuses self xs hs c hc hmax

/-! ## §2 · `step.cols.Engine.group_cols_loop2` -/

/-- The dummy payload used past the end of the payload vector. -/
def cols3_zeroRef : loader.PayloadRef := { off := 0#usize, len := 0#usize }

/-- The zero `GemmPart`, the `List.getD` default of the parts list. -/
def cols3_zeroPart : kern.gemm.GemmPart :=
  { ggml_type := 0#u32, «at» := cols3_zeroRef, n_out := 0#usize }

/-- The `GemmPart` a payload triple denotes. -/
def cols3_partOf (p : Std.U32 × loader.PayloadRef × Std.Usize) : kern.gemm.GemmPart :=
  { ggml_type := p.1, «at» := p.2.1, n_out := p.2.2 }

/-- The payload vector's parts, in order. -/
def cols3_gemmParts (payloads : alloc.vec.Vec (Std.U32 × loader.PayloadRef × Std.Usize)) :
    List kern.gemm.GemmPart :=
  payloads.val.map cols3_partOf

/-- The part at index `i`, with the dummy part past the end. -/
def cols3_partAt (payloads : alloc.vec.Vec (Std.U32 × loader.PayloadRef × Std.Usize))
    (i : ℕ) : kern.gemm.GemmPart :=
  cols3_partOf (payloads.val.getD i (0#u32, (cols3_zeroRef, 0#usize)))

/-- The parts at indices `i, …, i+k-1`: the reference of the payload loop's output. -/
def cols3_takeParts (payloads : alloc.vec.Vec (Std.U32 × loader.PayloadRef × Std.Usize))
    (i k : ℕ) : List kern.gemm.GemmPart :=
  (List.range k).map (fun j => cols3_partAt payloads (i + j))

theorem cols3_takeParts_length (payloads : alloc.vec.Vec (Std.U32 × loader.PayloadRef × Std.Usize))
    (i k : ℕ) : (cols3_takeParts payloads i k).length = k := by
  simp [cols3_takeParts, List.length_map, List.length_range]

/-- One more part: the reference of one `Vec::push` turn of the payload loop. -/
theorem cols3_takeParts_succ
    (payloads : alloc.vec.Vec (Std.U32 × loader.PayloadRef × Std.Usize)) (i k : ℕ) :
    cols3_takeParts payloads i (k + 1) =
      cols3_takeParts payloads i k ++ [cols3_partAt payloads (i + k)] := by
  simp only [cols3_takeParts, List.range_succ, List.map_append, List.map_singleton]

/-- **One `group_cols_loop2` body turn, continue.** The cursor is below the payload count: the turn
reads payload `gi` and pushes its `GemmPart`. -/
theorem cols3_group_cols_loop2_body_cont
    (payloads : alloc.vec.Vec (Std.U32 × loader.PayloadRef × Std.Usize))
    (group : alloc.vec.Vec kern.gemm.GemmPart) (gi : Std.Usize)
    (hgi : gi.val < payloads.val.length) (hcap : group.val.length < Std.Usize.max) :
    ∃ group1 : alloc.vec.Vec kern.gemm.GemmPart, ∃ gi1 : Std.Usize,
      step.cols.Engine.group_cols_loop2.body payloads group gi
        = ok (ControlFlow.cont (group1, gi1)) ∧
      gi1.val = gi.val + 1 ∧
      group1.val = group.val ++ [cols3_partAt payloads gi.val] := by
  have hmax : gi.val + (1#usize).val ≤ Std.Usize.max := by
    have := alloc.vec.Vec.len_ineq payloads
    rw [cols3_usize_one_val]
    omega
  unfold step.cols.Engine.group_cols_loop2.body
  simp only []
  rw [if_pos (by rw [UScalar.lt_equiv, alloc.vec.Vec.len_val]; exact hgi)]
  simp only [alloc.vec.Vec.index, core.slice.index.SliceIndexUsizeSlice,
    core.slice.index.Usize.index]
  obtain ⟨w, hw, hww⟩ := WP.spec_imp_exists (Slice.index_usize_spec payloads.slice gi hgi)
  obtain ⟨ggml_type, at1, n_out⟩ := w
  rw [hw]
  simp only [bind_tc_ok]
  -- `bind_tc_ok` leaves the monadic tuple destructuring as a `match` on the returned triple;
  -- `change` discharges it by the kernel's ι-reduction.
  change ∃ group1 : alloc.vec.Vec kern.gemm.GemmPart, ∃ gi1 : Std.Usize,
      (do
        let g0 ← alloc.vec.Vec.push group
          ({ ggml_type := ggml_type, «at» := at1, n_out := n_out } : kern.gemm.GemmPart)
        let i0 ← gi + 1#usize
        ok (ControlFlow.cont (g0, i0))) =
        ok (ControlFlow.cont (group1, gi1)) ∧
      gi1.val = gi.val + 1 ∧
      group1.val = group.val ++ [cols3_partAt payloads gi.val]
  obtain ⟨group1, hpush, hpushval⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec group
      ({ ggml_type := ggml_type, «at» := at1, n_out := n_out } : kern.gemm.GemmPart) hcap)
  rw [hpush]
  simp only [bind_tc_ok]
  obtain ⟨gi1, hgi1, hgi1val⟩ := WP.spec_imp_exists (Usize.add_spec (x := gi) (y := 1#usize) hmax)
  rw [hgi1]
  simp only [bind_tc_ok]
  refine ⟨group1, gi1, rfl, ?_, ?_⟩
  · rw [hgi1val, cols3_usize_one_val]
  · have hpart : cols3_partAt payloads gi.val =
        ({ ggml_type := ggml_type, «at» := at1, n_out := n_out } : kern.gemm.GemmPart) := by
      rw [cols3_partAt, List.getD_eq_getElem payloads.val (0#u32, (cols3_zeroRef, 0#usize)) hgi]
      have hg : payloads.val[gi.val] = (ggml_type, at1, n_out) := hww.symm
      rw [hg]
      rfl
    rw [hpushval, hpart]

/-- **One `group_cols_loop2` body turn, past the end.** The cursor has reached the payload count:
the turn returns the group unchanged. -/
theorem cols3_group_cols_loop2_body_done
    (payloads : alloc.vec.Vec (Std.U32 × loader.PayloadRef × Std.Usize))
    (group : alloc.vec.Vec kern.gemm.GemmPart) (gi : Std.Usize)
    (hgi : ¬ gi.val < payloads.val.length) :
    step.cols.Engine.group_cols_loop2.body payloads group gi = ok (ControlFlow.done group) := by
  unfold step.cols.Engine.group_cols_loop2.body
  simp only []
  rw [if_neg (by rw [UScalar.lt_equiv, alloc.vec.Vec.len_val]; exact hgi)]

/-- **The `group_cols_loop2` loop, from any cursor at or below the payload count.** The group is the
input group followed by the parts of the payloads from the cursor on. -/
theorem cols3_group_cols_loop2_spec
    (payloads : alloc.vec.Vec (Std.U32 × loader.PayloadRef × Std.Usize))
    (group : alloc.vec.Vec kern.gemm.GemmPart) (gi : Std.Usize)
    (hgi : gi.val ≤ payloads.val.length)
    (hcap : group.val.length + (payloads.val.length - gi.val) ≤ Std.Usize.max) :
    step.cols.Engine.group_cols_loop2 payloads group gi ⦃ o =>
      o.val = group.val ++ cols3_takeParts payloads gi.val (payloads.val.length - gi.val) ⦄ := by
  unfold step.cols.Engine.group_cols_loop2
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec kern.gemm.GemmPart × Std.Usize =>
      payloads.val.length - st.2.val)
    (inv := fun st => gi.val ≤ st.2.val ∧ st.2.val ≤ payloads.val.length ∧
      st.1.val = group.val ++ cols3_takeParts payloads gi.val (st.2.val - gi.val))
    (post := fun o : alloc.vec.Vec kern.gemm.GemmPart =>
      o.val = group.val ++ cols3_takeParts payloads gi.val (payloads.val.length - gi.val))
  · rintro ⟨g, ga⟩ ⟨hle1, hle2, hinv⟩
    dsimp only at hle1 hle2 hinv
    by_cases hlt : ga.val < payloads.val.length
    · have hcap1 : g.val.length < Std.Usize.max := by
        have hlen := cols3_takeParts_length payloads gi.val (ga.val - gi.val)
        rw [hinv, List.length_append, hlen]
        omega
      obtain ⟨g1, ga1, hbody, hga1, hg1val⟩ :=
        cols3_group_cols_loop2_body_cont payloads g ga hlt hcap1
      rw [WP.spec_equiv_exists]
      refine ⟨_, hbody, ?_⟩
      dsimp only
      refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
      · rw [hga1]; omega
      · rw [hga1]; omega
      · have hidx : ga1.val - gi.val = (ga.val - gi.val) + 1 := by rw [hga1]; omega
        have hidx2 : gi.val + (ga.val - gi.val) = ga.val := by omega
        rw [hg1val, hinv, hidx, cols3_takeParts_succ, List.append_assoc, hidx2]
      · rw [hga1]; omega
    · have hge : ga.val = payloads.val.length := le_antisymm hle2 (Nat.le_of_not_lt hlt)
      have hbody := cols3_group_cols_loop2_body_done payloads g ga hlt
      rw [WP.spec_equiv_exists]
      refine ⟨_, hbody, ?_⟩
      dsimp only
      rw [hinv, hge]
  · refine ⟨le_refl _, hgi, ?_⟩
    simp [cols3_takeParts]

/-- **Success law.** From an empty group at cursor zero, the extracted
`step.cols.Engine.group_cols_loop2` returns the parts of every payload, in order. -/
theorem word_group_cols_loop2_ok
    (payloads : alloc.vec.Vec (Std.U32 × loader.PayloadRef × Std.Usize)) :
    ∃ group : alloc.vec.Vec kern.gemm.GemmPart,
      step.cols.Engine.group_cols_loop2 payloads
        (alloc.vec.Vec.with_capacity kern.gemm.GemmPart (alloc.vec.Vec.len payloads)) 0#usize
        = ok group ∧
      group.val = cols3_takeParts payloads 0 payloads.val.length := by
  have hcap0 : (alloc.vec.Vec.with_capacity kern.gemm.GemmPart (alloc.vec.Vec.len payloads)).val.length +
      (payloads.val.length - (0#usize).val) ≤ Std.Usize.max := by
    have hp := alloc.vec.Vec.len_ineq payloads
    rw [cols3_usize_zero_val, cols3_with_capacity_val, List.length_nil]
    omega
  obtain ⟨g, ho, hpost⟩ := WP.spec_imp_exists
    (cols3_group_cols_loop2_spec payloads
      (alloc.vec.Vec.with_capacity kern.gemm.GemmPart (alloc.vec.Vec.len payloads)) 0#usize
      (by rw [cols3_usize_zero_val]; exact Nat.zero_le _) hcap0)
  refine ⟨g, ho, ?_⟩
  rw [hpost, cols3_usize_zero_val, Nat.sub_zero]
  simp only [cols3_with_capacity_val, List.nil_append]

/-! ### Non-vacuity -/

/-- The FFN reference at a concrete input: two columns of width three. -/
example : cols3_ffnCols 3 2 = [[0, 0, 0], [0, 0, 0]] := by decide

/-- The zero-column reference is a column of zeros. -/
example : cols3_zeroCol 2 = [0, 0] := by decide

/-- The loop-spec capacity hypothesis is satisfiable at the caller's instantiation. -/
example (xs : Slice (alloc.vec.Vec Std.I64)) (c : Std.Usize) (hc : c.val ≤ xs.val.length) :
    (alloc.vec.Vec.with_capacity (alloc.vec.Vec Std.I64) (Slice.len xs)).val.length +
      (xs.val.length - c.val) ≤ Std.Usize.max := by
  have hp := Slice.length_ineq xs
  rw [cols3_with_capacity_val, List.length_nil]
  omega

/-! ## Axiom cone -/


end apeiron_word
