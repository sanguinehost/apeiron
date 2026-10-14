/-
Row-chunked matvec equals the whole.

The Rust body is `src/engine/par.rs`; `matvec_rows_chunked` splits the `n_out` rows of a projection
into contiguous row chunks, calls the extracted `matvec_fixed` on each chunk's row-aligned byte
sub-range and concatenates the outputs in row order. The chunk split is exactly a byte split at a
row boundary: `row_bytes` is `div_ceil n_in qk · bb`, rows are contiguous with that stride, and
chunk `c` covering rows `[lo, hi)` reads `raw[lo·row_bytes .. hi·row_bytes)`.

This module proves the law `par.rs` relies on, on the `matvec_domain` success domain:

* `matvec_fixed_rows`: the output of `word_matvec_fixed` is `(List.range n_out).map (rowOut raw r)`
  for the pure `rowOut`, which reads only row `r`'s byte window and `x`;
* `matvec_fixed_chunk`: for a row-aligned split `raw = raw₁ ++ raw₂` with `n_out = a + b`, the
  output on `raw` is the output on `raw₁` (a rows) concatenated with the output on `raw₂` (b rows);
* `matvec_fixed_chunks`: the same for any list of consecutive chunks, by induction.

The dequant-to-row-value reading stays the class-(b) `tested` boundary of
`Numeric/Engine/Matvec.lean`; the per-row value here is *defined* by the extracted
dequantizer and the extracted row dot on the canonical empty scratch buffer, so no dequant bound or
ideal-weight identification is re-proved. Trust boundary: rustc/LLVM codegen, the kernel ABI and
the `unsafe` container are outside; the parallel spawn/join in `par.rs` is not modelled.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. -/
import Numeric.Dequant.Refinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The row window and the pure row reference -/

/-- A canonical empty scratch buffer for the dequantizer. Any empty `Vec` gives the same row values
(the domain's dequant conjunct is universal over empty buffers), so the reference can fix one. -/
def matvecScratch : alloc.vec.Vec Std.I64 :=
  { slice := Slice.new Std.I64, cap := 0#usize }

@[simp, simp_lists_safe]
theorem matvecScratch_val : matvecScratch.val = [] := rfl

/-- The `row_bytes`-byte window of row `r` of `raw`: bytes `[r·row_bytes, (r+1)·row_bytes)`. -/
def rowWindow (raw : Slice U8) (row_bytes : Usize) (r : Nat) : Slice U8 :=
  Slice.from (raw.val.slice (r * row_bytes.val) ((r + 1) * row_bytes.val))
    (le_trans (List.slice_length_le _ _ _) (Slice.property raw))

@[simp]
theorem rowWindow_val (raw : Slice U8) (row_bytes : Usize) (r : Nat) :
    (rowWindow raw row_bytes r).val =
      raw.val.slice (r * row_bytes.val) ((r + 1) * row_bytes.val) :=
  Slice.from_val _ _

/-- The pure per-row value: dequantize the row window into the canonical scratch and reduce it
against `x`. This reads only the window bytes and `x`. -/
def rowOut (ggml_type : U32) (row : Slice U8) (n_in : Usize) (x : Slice I64) : I64 :=
  match (dequant.dequant_row_fixed ggml_type row n_in matvecScratch).match with
  | .ok (core.result.Result.Ok _, rb') =>
      match (matvec.row_dot_fixed (alloc.vec.Vec.deref rb') x).match with
      | .ok (core.result.Result.Ok v) => v
      | _ => 0#i64
  | _ => 0#i64

@[simp]
theorem rowOut_def (ggml_type : U32) (row : Slice U8) (n_in : Usize) (x : Slice I64)
    (rb' : alloc.vec.Vec I64) (v : I64)
    (hdeq : dequant.dequant_row_fixed ggml_type row n_in matvecScratch
      = ok (core.result.Result.Ok (), rb'))
    (hdot : matvec.row_dot_fixed (alloc.vec.Vec.deref rb') x
      = ok (core.result.Result.Ok v)) :
    rowOut ggml_type row n_in x = v := by
  unfold rowOut
  rw [hdeq, Result.match.ok]
  simp only [hdot, Result.match.ok]

/-! ## Slicing a concatenation -/

/-- A slice of a concatenation that stays inside the left part is the left slice. -/
theorem list_slice_append_left {α} (l₁ l₂ : List α) {i j : Nat} (hj : j ≤ l₁.length) :
    (l₁ ++ l₂).slice i j = l₁.slice i j := by
  unfold List.slice
  by_cases hij : i ≤ j
  · have hi : i ≤ l₁.length := le_trans hij hj
    rw [List.drop_append_of_le_length hi]
    have hlt : j - i ≤ (l₁.drop i).length := by rw [List.length_drop]; omega
    rw [List.take_append_of_le_length hlt]
  · have h0 : j - i = 0 := by omega
    rw [h0]; simp

/-- A slice of a concatenation at offset `l₁.length` is the corresponding slice of the right
part. -/
theorem list_slice_append_right {α} (l₁ l₂ : List α) (i j : Nat) :
    (l₁ ++ l₂).slice (l₁.length + i) (l₁.length + j) = l₂.slice i j := by
  unfold List.slice
  rw [List.drop_length_add_append]
  have hsub : l₁.length + j - (l₁.length + i) = j - i := by omega
  rw [hsub]

/-- Row `r` of the left chunk is the same window as row `r` of the whole. -/
theorem rowWindow_left (raw raw₁ raw₂ : Slice U8) (row_bytes : Usize) (a r : Nat)
    (hsplit : raw.val = raw₁.val ++ raw₂.val)
    (hlen : raw₁.val.length = a * row_bytes.val) (hr : r < a) :
    rowWindow raw row_bytes r = rowWindow raw₁ row_bytes r := by
  apply Slice.ext
  rw [rowWindow_val, rowWindow_val, hsplit]
  apply list_slice_append_left
  rw [hlen]
  have : (r + 1) * row_bytes.val ≤ a * row_bytes.val :=
    Nat.mul_le_mul_right _ (by omega)
  exact this

/-- Row `a + r` of the whole is row `r` of the right chunk. -/
theorem rowWindow_right (raw raw₁ raw₂ : Slice U8) (row_bytes : Usize) (a r : Nat)
    (hsplit : raw.val = raw₁.val ++ raw₂.val)
    (hlen : raw₁.val.length = a * row_bytes.val) :
    rowWindow raw row_bytes (a + r) = rowWindow raw₂ row_bytes r := by
  apply Slice.ext
  rw [rowWindow_val, rowWindow_val, hsplit]
  have e1 : (a + r) * row_bytes.val = raw₁.val.length + r * row_bytes.val := by
    rw [hlen]; ring
  have e2 : (a + r + 1) * row_bytes.val = raw₁.val.length + (r + 1) * row_bytes.val := by
    rw [hlen]; ring
  rw [e1, e2]
  apply list_slice_append_right

/-! ## The row window from the loop's own `Slice.get` -/

/-- The domain's `Slice.get` window at `o·row_bytes` is exactly `rowWindow`. -/
theorem rowWindow_of_get (raw : Slice U8) (row_bytes : Usize) (o : Nat)
    (start i : Usize) (row : Slice U8)
    (hstart : start.val = o * row_bytes.val)
    (hi : i.val = (o + 1) * row_bytes.val)
    (hget : core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice Std.U8) raw
      { start := start, «end» := i } = ok (some row)) :
    row = rowWindow raw row_bytes o := by
  change core.slice.index.SliceIndexRangeUsizeSlice.get { start := start, «end» := i } raw
    = ok (some row) at hget
  unfold core.slice.index.SliceIndexRangeUsizeSlice.get at hget
  split at hget
  · simp only [Result.ok.injEq, Option.some.injEq] at hget
    rw [← hget]
    apply Slice.ext
    rw [rowWindow_val, Slice.from_val, hstart, hi]
  · simp at hget

/-! ## One exact row-loop step -/

/-- One row-loop body step on `matvec_domain`, carrying the exact appended value: the body reads
row `o`'s window, dequantizes it and appends `rowOut` of that window. -/
theorem matvec_chunk_body_step
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64)
    (out row_buf : alloc.vec.Vec I64) (o : Usize)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq)
    (ho : o.val < n_out.val) (houtlen : out.val.length < Usize.max) :
    WP.spec (matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need
        out row_buf o true none)
      (fun cf =>
        ∃ (out2 : alloc.vec.Vec Std.I64) (row_buf3 : alloc.vec.Vec Std.I64) (o2 : Usize),
          cf = ControlFlow.cont (out2, row_buf3, o2, true, none) ∧
          o2.val = o.val + 1 ∧
          out2.val = out.val ++ [rowOut ggml_type (rowWindow raw row_bytes o.val) n_in x]) := by
  obtain ⟨_hgeom, _hxlen, _hdiv, hrowb, hneed, _hsuff, hdeqlen, hdotdom, hslice, hdeq⟩ := hdom
  obtain ⟨start, i, row, hstart, hadd, hget⟩ := hslice o ho
  have hstartv : start.val = o.val * row_bytes.val := by
    have hb := Usize.checked_mul_bv_spec o row_bytes
    rw [hstart] at hb; simp only at hb; exact hb.2.1
  have hneedv : need.val = n_out.val * row_bytes.val := by
    have hb := Usize.checked_mul_bv_spec n_out row_bytes
    rw [hneed] at hb; simp only at hb; exact hb.2.1
  have hmax : start.val + row_bytes.val ≤ Usize.max := by
    have hb := Usize.checked_mul_bv_spec n_out row_bytes
    rw [hneed] at hb; simp only at hb
    have hmul : (o.val + 1) * row_bytes.val ≤ n_out.val * row_bytes.val :=
      Nat.mul_le_mul_right _ (by omega)
    have hexp : (o.val + 1) * row_bytes.val = o.val * row_bytes.val + row_bytes.val := by ring
    rw [hexp] at hmul
    rw [hstartv]
    omega
  have hiv : i.val = (o.val + 1) * row_bytes.val := by
    obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists
      (Usize.add_bv_spec (x := start) (y := row_bytes) hmax)
    have hiz : i = z := by
      have h : ok i = ok z := by rw [← hadd, hz]
      simpa only [Result.ok.injEq] using h
    rw [hiz, hzv.1, hstartv]
    ring
  have hrowEq : row = rowWindow raw row_bytes o.val :=
    rowWindow_of_get raw row_bytes o.val start i row hstartv hiv hget
  have hclear : (alloc.vec.Vec.clear Global row_buf
      : Result (alloc.vec.Vec Std.I64)) = ok { row_buf with slice := Slice.new Std.I64 } := rfl
  have hcleared : ({ row_buf with slice := Slice.new Std.I64 } : alloc.vec.Vec Std.I64).val = [] := by
    simp [alloc.vec.Vec.val, Slice.new_val]
  obtain ⟨rb', hdeq_eq, hdeq_val⟩ :=
    hdeq o start i row { row_buf with slice := Slice.new Std.I64 } ho hstart hadd hget hcleared
  obtain ⟨rb'', hdeq_eq'', hdeq_val''⟩ :=
    hdeq o start i row matvecScratch ho hstart hadd hget matvecScratch_val
  have hval_eq : rb'.val = rb''.val := by rw [hdeq_val, hdeq_val'']
  have hderef : alloc.vec.Vec.deref rb' = alloc.vec.Vec.deref rb'' := by
    apply Slice.ext
    simp only [alloc.vec.Vec.deref, Slice.from_val]
    exact hval_eq
  have hsval : (alloc.vec.Vec.deref rb').val = rb'.val := by
    simp [alloc.vec.Vec.deref]
  have hseq : alloc.vec.Vec.deref rb' = deq o.val := by
    apply Slice.ext
    rw [hsval, hdeq_val]
  have hdotdom' : row_dot_domain (alloc.vec.Vec.deref rb') x := by
    rw [hseq]
    exact hdotdom o.val ho
  obtain ⟨y, hy, _hboundy⟩ := row_dot_refines_proved (alloc.vec.Vec.deref rb') x hdotdom'
  have hy' : matvec.row_dot_fixed (alloc.vec.Vec.deref rb') x = ok (core.result.Result.Ok y) := hy
  have hdot'' : matvec.row_dot_fixed (alloc.vec.Vec.deref rb'') x = ok (core.result.Result.Ok y) := by
    rw [← hderef]; exact hy'
  have hrowOut : rowOut ggml_type row n_in x = y :=
    rowOut_def ggml_type row n_in x rb'' y hdeq_eq'' hdot''
  have hrowOut' : rowOut ggml_type (rowWindow raw row_bytes o.val) n_in x = y := by
    rw [← hrowEq]; exact hrowOut
  have hlenrow : alloc.vec.Vec.len rb' = n_in := by
    apply UScalar.eq_of_val_eq
    rw [alloc.vec.Vec.len_val]
    show rb'.val.length = n_in.val
    rw [hdeq_val]
    exact hdeqlen o.val ho
  obtain ⟨out2, hpush, hpushv⟩ :=
    WP.spec_imp_exists (alloc.vec.Vec.push_spec out y houtlen)
  obtain ⟨o2, hadd1, ho2⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := o) (y := 1#usize) (o_succ_le_max n_out o ho))
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (out2, rb', o2, true, none), ?_, ?_⟩
  · rw [matvec.matvec_fixed_loop.body]
    simp only [if_pos ((UScalar.lt_equiv o n_out).2 ho),
      hstart, hadd, hget, hclear, hdeq_eq, if_pos hlenrow, hy', hpush, hadd1,
      lift, bind_tc_ok, uncurry_apply_pair]
    rfl
  · refine ⟨out2, rb', o2, rfl, ho2, ?_⟩
    rw [hpushv, hrowOut']

/-! ## The row loop computes the map -/

/-- The row-loop invariant tracking the exact output list: the flag is clear, the error is `none`,
the index is at most `n_out`, and the output is the map of `rowOut` over the windows visited. -/
def chunkLoopInv (ggml_type : U32) (raw : Slice U8) (row_bytes : Usize) (n_in : Usize)
    (x : Slice I64) (n_out : Usize) (out _row_buf : alloc.vec.Vec I64) (o : Usize)
    (ok : Bool) (err : Option matvec.MatvecError) : Prop :=
  ok = true ∧ err = none ∧ o.val ≤ n_out.val ∧
  out.val = (List.range o.val).map
    (fun r => rowOut ggml_type (rowWindow raw row_bytes r) n_in x)

/-- On `matvec_domain` the row loop returns the exact map of `rowOut` over its windows. -/
theorem matvec_fixed_loop_rows
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq) :
    WP.spec (matvec.matvec_fixed_loop ggml_type raw n_out n_in x row_bytes need
        (alloc.vec.Vec.with_capacity Std.I64 n_out)
        (alloc.vec.Vec.with_capacity Std.I64 n_in) 0#usize true none)
      (fun b =>
        ∃ out : alloc.vec.Vec I64, b = (out, none) ∧
          out.val = (List.range n_out.val).map
            (fun r => rowOut ggml_type (rowWindow raw row_bytes r) n_in x)) := by
  have hinv0 : chunkLoopInv ggml_type raw row_bytes n_in x n_out
      (alloc.vec.Vec.with_capacity Std.I64 n_out)
      (alloc.vec.Vec.with_capacity Std.I64 n_in) 0#usize true none := by
    refine ⟨rfl, rfl, ?_, ?_⟩
    · show (0#usize).val ≤ n_out.val
      have h1 : (0#usize).val = 0 := by norm_num
      omega
    · show (alloc.vec.Vec.with_capacity Std.I64 n_out).val =
        (List.range (0#usize).val).map
          (fun r => rowOut ggml_type (rowWindow raw row_bytes r) n_in x)
      have h1 : (0#usize).val = 0 := by norm_num
      simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val, h1]
  unfold matvec.matvec_fixed_loop
  apply loop.spec_decr_nat
    (measure := fun s : (alloc.vec.Vec Std.I64) × (alloc.vec.Vec Std.I64) ×
        (Std.Usize × Bool × (Option matvec.MatvecError)) => n_out.val - s.2.2.1.val)
    (inv := fun s =>
      chunkLoopInv ggml_type raw row_bytes n_in x n_out s.1 s.2.1 s.2.2.1 s.2.2.2.1 s.2.2.2.2)
    (post := fun b : (alloc.vec.Vec Std.I64) × (Option matvec.MatvecError) =>
      ∃ out : alloc.vec.Vec I64, b = (out, none) ∧
        out.val = (List.range n_out.val).map
          (fun r => rowOut ggml_type (rowWindow raw row_bytes r) n_in x))
  · rintro ⟨out, row_buf, o, ok1, err⟩ hs
    dsimp only at hs ⊢
    obtain ⟨hok, herr, ho, hlen⟩ := hs
    rw [hok, herr]
    have hnmax := n_out_le_max n_out
    by_cases hlt : o.val < n_out.val
    · obtain ⟨cf, hcf, hpost⟩ := WP.spec_imp_exists
        (matvec_chunk_body_step ggml_type raw n_out n_in x qk bb blocks row_bytes need deq
          out row_buf o hdom hlt (by rw [hlen]; simp only [List.length_map, List.length_range]; omega))
      obtain ⟨out2, row_buf2, o2, hcfg, ho2, hval⟩ := hpost
      rw [hcfg] at hcf
      rw [hcf]
      rw [Aeneas.Std.WP.spec_ok]
      dsimp only
      refine ⟨⟨rfl, rfl, by omega, ?_⟩, ?_⟩
      · rw [hval, hlen, ho2]
        rw [List.range_succ]
        simp
      · show n_out.val - o2.val < n_out.val - o.val
        omega
    · have hnot : ¬ (o < n_out) := by
        intro hc
        exact hlt ((UScalar.lt_equiv o n_out).1 hc)
      have hbody : matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need
          out row_buf o true none = ok (ControlFlow.done (out, none)) := by
        rw [matvec.matvec_fixed_loop.body, if_neg hnot]
      simp only [hbody]
      refine ⟨out, rfl, ?_⟩
      have hoeq : o.val = n_out.val := by omega
      rw [hlen, hoeq]
  · exact hinv0

/-! ## Statement 1: the output is the map of the per-row value -/

/-- **Rows.** On `matvec_domain` the output of `word_matvec_fixed` is
`(List.range n_out).map (rowOut raw r)`: row `r` of the output depends only on row `r`'s byte window
and `x`. -/
theorem matvec_fixed_rows
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq) :
    ∃ out : alloc.vec.Vec I64,
      word_matvec_fixed ggml_type raw n_out n_in x = ok (core.result.Result.Ok out) ∧
      out.val = (List.range n_out.val).map
        (fun r => rowOut ggml_type (rowWindow raw row_bytes r) n_in x) := by
  have hdom' := hdom
  obtain ⟨b, hb, out, hbout, hval⟩ := WP.spec_imp_exists
    (matvec_fixed_loop_rows ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom')
  obtain ⟨hgeom, hxlen, hdiv, hrowb, hneed, hsuff, _hdeqlen, _hdotdom, _hslice, _hdeq⟩ := hdom
  rw [hbout] at hb
  refine ⟨out, ?_, hval⟩
  rw [word_matvec_fixed_unfold ggml_type raw n_out n_in x qk bb blocks row_bytes need
    hgeom hxlen hdiv hrowb hneed hsuff]
  rw [hb]
  simp only [bind_tc_ok, uncurry_apply_pair]

/-! ## Statement 2: a row-aligned two-way split -/

/-- **Chunk.** For a row-aligned byte split `raw = raw₁ ++ raw₂` with `n_out = a + b` and
`|raw₁| = a·row_bytes`, the output on `raw` is the output on `raw₁` (a rows) concatenated with the
output on `raw₂` (b rows). -/
theorem matvec_fixed_chunk
    (ggml_type : U32) (raw raw₁ raw₂ : Slice U8) (n_out n_in a b : Usize) (x : Slice I64)
    (qk bb blocks blocks₁ blocks₂ row_bytes need need₁ need₂ : Usize)
    (deq deq₁ deq₂ : Nat → Slice I64)
    (hsplit : raw.val = raw₁.val ++ raw₂.val)
    (hlen1 : raw₁.val.length = a.val * row_bytes.val)
    (hab : n_out.val = a.val + b.val)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq)
    (hdom1 : matvec_domain ggml_type raw₁ a n_in x qk bb blocks₁ row_bytes need₁ deq₁)
    (hdom2 : matvec_domain ggml_type raw₂ b n_in x qk bb blocks₂ row_bytes need₂ deq₂) :
    ∃ out out₁ out₂ : alloc.vec.Vec I64,
      word_matvec_fixed ggml_type raw n_out n_in x = ok (core.result.Result.Ok out) ∧
      word_matvec_fixed ggml_type raw₁ a n_in x = ok (core.result.Result.Ok out₁) ∧
      word_matvec_fixed ggml_type raw₂ b n_in x = ok (core.result.Result.Ok out₂) ∧
      out.val = out₁.val ++ out₂.val := by
  obtain ⟨out, hout, hval⟩ :=
    matvec_fixed_rows ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom
  obtain ⟨out₁, hout₁, hval₁⟩ :=
    matvec_fixed_rows ggml_type raw₁ a n_in x qk bb blocks₁ row_bytes need₁ deq₁ hdom1
  obtain ⟨out₂, hout₂, hval₂⟩ :=
    matvec_fixed_rows ggml_type raw₂ b n_in x qk bb blocks₂ row_bytes need₂ deq₂ hdom2
  refine ⟨out, out₁, out₂, hout, hout₁, hout₂, ?_⟩
  rw [hval, hval₁, hval₂, hab, List.range_add, List.map_append]
  congr 1
  · apply List.map_congr_left
    intro r hr
    have hr' : r < a.val := by simpa using hr
    rw [rowWindow_left raw raw₁ raw₂ row_bytes a.val r hsplit hlen1 hr']
  · rw [List.map_map]
    apply List.map_congr_left
    intro r _hr
    dsimp only [Function.comp_apply]
    rw [rowWindow_right raw raw₁ raw₂ row_bytes a.val r hsplit hlen1]

/-! ## Statement 3: any list of consecutive chunks -/

/-- The per-row reference over a concatenated list of consecutive row chunks is the join of the
per-chunk references. This is the induction `par.rs`'s assembly relies on, at the window level. -/
theorem rowOut_join (ggml_type : U32) (n_in : Usize) (x : Slice I64) (row_bytes : Usize) :
    ∀ (chunks : List (Slice U8 × Nat)) (raw : Slice U8) (n_out : Nat),
      (chunks.map (fun c => c.2)).sum = n_out →
      (chunks.map (fun c => c.1.val.length)) = chunks.map (fun c => c.2 * row_bytes.val) →
      raw.val = (chunks.map (fun c => c.1.val)).flatten →
      (List.range n_out).map (fun r => rowOut ggml_type (rowWindow raw row_bytes r) n_in x)
        = (chunks.map (fun c => (List.range c.2).map
            (fun r => rowOut ggml_type (rowWindow c.1 row_bytes r) n_in x))).flatten := by
  intro chunks
  induction chunks with
  | nil =>
      intro raw n_out hsum _hlen _hcat
      rw [← hsum]
      simp
  | cons c cs ih =>
      intro raw n_out hsum hlen hcat
      obtain ⟨r₀, k⟩ := c
      simp only [List.map_cons, List.sum_cons] at hsum
      obtain ⟨hlen0, hlenrest⟩ := by
        simpa only [List.map_cons, List.cons.injEq] using hlen
      simp only [List.map_cons, List.flatten_cons] at hcat
      let L : List U8 := (cs.map (fun c => c.1.val)).flatten
      have hLle : L.length ≤ Usize.max := by
        have hLraw : L.length ≤ raw.val.length := by
          rw [hcat]
          show L.length ≤ (r₀.val ++ L).length
          simp only [List.length_append]
          omega
        exact le_trans hLraw (Slice.property raw)
      let rest : Slice U8 := Slice.from L hLle
      have hrest : rest.val = L := by simp [rest]
      have hsplit : raw.val = r₀.val ++ rest.val := by
        rw [hrest]
        exact hcat
      have ih' := ih rest (cs.map (fun c => c.2)).sum rfl hlenrest (by rw [hrest])
      rw [show n_out = k + (cs.map (fun c => c.2)).sum from hsum.symm]
      rw [List.range_add, List.map_append, List.map_cons, List.flatten_cons]
      have hA : (List.range k).map
            (fun r => rowOut ggml_type (rowWindow raw row_bytes r) n_in x)
          = (List.range k).map
            (fun r => rowOut ggml_type (rowWindow r₀ row_bytes r) n_in x) := by
        apply List.map_congr_left
        intro r hr
        have hr' : r < k := by simpa using hr
        rw [rowWindow_left raw r₀ rest row_bytes k r hsplit hlen0 hr']
      have hB : ((List.range (cs.map (fun c => c.2)).sum).map (fun r => k + r)).map
            (fun r => rowOut ggml_type (rowWindow raw row_bytes r) n_in x)
          = (cs.map (fun c => (List.range c.2).map
              (fun r => rowOut ggml_type (rowWindow c.1 row_bytes r) n_in x))).flatten := by
        rw [List.map_map, ← ih']
        apply List.map_congr_left
        intro r _hr
        dsimp only [Function.comp_apply]
        rw [rowWindow_right raw r₀ rest row_bytes k r hsplit hlen0]
      rw [hA, hB]

/-- **Chunks.** For any list of consecutive row chunks whose bytes concatenate to `raw` and whose
row counts sum to `n_out`, if the outputs of the per-chunk `word_matvec_fixed` calls are the `outs`
list (whose entries are the per-chunk maps by `matvec_fixed_rows`), then the whole output is the
join of the chunk outputs in order. -/
theorem matvec_fixed_chunks
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64)
    (chunks : List (Slice U8 × Usize)) (outs : List (alloc.vec.Vec I64))
    (hchunks : chunks.length = outs.length)
    (hsum : (chunks.map (fun c => c.2.val)).sum = n_out.val)
    (hlen : chunks.map (fun c => c.1.val.length) = chunks.map (fun c => c.2.val * row_bytes.val))
    (hcat : raw.val = (chunks.map (fun c => c.1.val)).flatten)
    (houtmaps : ∀ (i : Nat) (hi : i < chunks.length),
      (outs[i]'(Nat.lt_of_lt_of_eq hi hchunks)).val
        = (List.range (chunks[i]'hi).2.val).map
            (fun r => rowOut ggml_type (rowWindow (chunks[i]'hi).1 row_bytes r) n_in x))
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq) :
    ∀ out : alloc.vec.Vec I64,
      word_matvec_fixed ggml_type raw n_out n_in x = ok (core.result.Result.Ok out) →
      out.val = (outs.map (fun v => v.val)).flatten := by
  intro out hout
  obtain ⟨out', hout', hval⟩ :=
    matvec_fixed_rows ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom
  have hout_eq : out = out' := by
    have h : (ok (core.result.Result.Ok out)
        : Result (core.result.Result (alloc.vec.Vec Std.I64) matvec.MatvecError))
        = ok (core.result.Result.Ok out') := by
      rw [← hout, hout']
    simpa only [Result.ok.injEq, core.result.Result.Ok.injEq] using h
  have hval' : out.val = (List.range n_out.val).map
      (fun r => rowOut ggml_type (rowWindow raw row_bytes r) n_in x) := by
    rw [hout_eq]; exact hval
  rw [hval']
  have houts_eq : outs.map (fun v => v.val)
      = chunks.map (fun c => (List.range c.2.val).map
          (fun r => rowOut ggml_type (rowWindow c.1 row_bytes r) n_in x)) := by
    apply List.ext_getElem
    · simp [hchunks]
    · intro i _h1 h2
      rw [List.getElem_map, List.getElem_map]
      exact houtmaps i (by simpa using h2)
  rw [houts_eq]
  have hjoin := rowOut_join ggml_type n_in x row_bytes
    (chunks.map (fun c => (c.1, c.2.val))) raw n_out.val
    (by simpa [List.map_map, Function.comp_def] using hsum)
    (by simpa [List.map_map, Function.comp_def] using hlen)
    (by simpa [List.map_map, Function.comp_def] using hcat)
  simpa [List.map_map, Function.comp_def] using hjoin

end

end apeiron_word
