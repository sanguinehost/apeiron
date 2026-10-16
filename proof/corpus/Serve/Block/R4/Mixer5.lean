/-
  `Serve/Block/R4/Mixer5.lean` — the extraction-level refinement of `mixer.norm_qk_q32`
  (`src/engine/mixer.rs:4111-4155`), the Q32.32 Q·K normalisation on the serving path.  It continues
  `Serve/Block/R4/Mixer4.lean` (the `l2_normalize` refinement, its `mixer4_*` references and the
  refusal of a non-positive scale); the column loop, its per-row scaling loop and the driver are
  closed here.

  What the Rust body computes (`norm_qk_q32_loop0` → `norm_qk_q32_loop0_loop0` → `l2_normalize`):

  * `kr0 = kh * hk`; then for every column `c < fa.k` the flattened conv buffer is read twice,
    `q = fa.conv[c·cdim + kr0 … + hk)` and `k = fa.conv[c·cdim + key_dim + kr0 … + hk)`, and each
    window is L2-normalised in Q32.32 by `mixer.l2_normalize` (`Mixer4`);
  * the destination window `dst[2c·hk … 2(c+1)·hk)` receives the scaled query row
    `d[i] = mul_q32(q_hat[i], q_scale)` for `i < hk` (`Overflow` when a product does not fit
    `i64`) followed by `k_hat` copied into `d[hk…]`;
  * the loop stops at the first column whose normalisation or product overflows, and the driver
    returns `Err e` for that `e`, else `Ok ()`.

  The references below are plain `List ℤ/Nat` definitions of that *meaning*: `mixer5_hatRef` is the
  normalised row (`Mixer4`'s `mixer4_expected` at the reference denominator), `mixer5_windowRef` the
  scaled-query-then-key window, and `mixer5_dstRef` the destination with the windows spliced in by
  `List.setSlice!` at offset `2c·hk` — never a restatement of the Rust loops.  The loop invariants
  are `loop.spec_decr_nat` on the remaining column index, in the style of `Mixer4` and
  `Icarus/Extraction/Store.lean`.

  Domain, stated rather than papered over (each is a bound the Rust checks or its caller
  guarantees):

  * `mixer5_domain eps l` — the `l2_normalize` window `Mixer4` requires: `1 ≤ Σᵢ lᵢ² ≤ 2¹¹⁶`, every
    Q32.32 quotient `lᵢ·2³²/den` inside `i64`; `hk < Usize.max` is the slice-width cap;
  * `mixer5_scaleFit q_scale eps l` — every product `mul_q32(q̂ᵢ, q_scale)` inside `i64`; a product
    outside it is the `Overflow` refusal, not a silent wrap;
  * the column bounds `mixer5_colOk` — the two conv windows lie inside `fa.conv` and the destination
    window `2(c+1)·hk ≤ dst.len` is in range, so no index arithmetic overflows `usize`;
  * the destination keeps its length (`List.setSlice!` writes in place), so one bound per column
    covers every iteration.

  Not covered, with the reason: rustc/LLVM codegen; the `F32`/`F64` carriers and the AVX2/`unsafe`
  bodies (no `Slice` model — `proof/regen.sh` opaque list, listed in §Functions not refined below);
  the `i64`-narrowing and message-conversion boundaries of `Mixer4` §Boundary, which every theorem
  mentioning `mixer.l2_normalize` carries in its cone.
-/

import Serve.Block.R4.Mixer4

set_option autoImplicit false
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## §1 · References — what the kernel computes

`mixer5_mulW` is the Q32.32 product narrowed to `i64`, `mixer5_scaleL` its map over a row,
`mixer5_hatRef` the L2-normalised row (`Mixer4`'s `mixer4_expected` at the reference denominator),
`mixer5_windowRef` one destination window, and `mixer5_dstRef` the whole destination with the
windows spliced in.  `mixer5_domain` / `mixer5_scaleFit` are the stated admissible domain. -/

/-- The reference product `a·b·2⁻³²` truncated toward zero, as an `i64` word.  This is the value
`value.mul_q32` returns when the product fits (`mixer2_mul_q32_val`). -/
def mixer5_mulW (q_scale : ℤ) (v : Std.I64) : Std.I64 :=
  ⟨BitVec.ofInt 64 (mixer2_mulQ32Ref v.val q_scale)⟩

/-- The word carries the reference product when that product fits `i64`. -/
theorem mixer5_mulW_val (q_scale : ℤ) (v : Std.I64)
    (hlo : I64.min ≤ mixer2_mulQ32Ref v.val q_scale)
    (hhi : mixer2_mulQ32Ref v.val q_scale ≤ I64.max) :
    (mixer5_mulW q_scale v).val = mixer2_mulQ32Ref v.val q_scale := by
  have hlo' : -(2 : ℤ) ^ (63 : ℕ) ≤ mixer2_mulQ32Ref v.val q_scale := by
    have h := hlo; rw [I64.min_eq] at h; linarith
  have hhi' : mixer2_mulQ32Ref v.val q_scale < (2 : ℤ) ^ (63 : ℕ) := by
    have h := hhi; rw [I64.max_eq] at h
    have h2 : (2 : ℤ) ^ (63 : ℕ) = 9223372036854775808 := by norm_num
    linarith
  show (BitVec.ofInt 64 (mixer2_mulQ32Ref v.val q_scale)).toInt = mixer2_mulQ32Ref v.val q_scale
  exact BitVec.toInt_ofInt_eq_self (by norm_num) hlo' hhi'

/-- `q_scale` applied elementwise to a row, as words. -/
def mixer5_scaleL (q_scale : ℤ) (l : List Std.I64) : List Std.I64 := l.map (mixer5_mulW q_scale)

/-- The L2-normalised row at the reference denominator: `Mixer4`'s `mixer4_expected` at
`mixer4_denRef (Σᵢ lᵢ²) eps`. -/
def mixer5_hatRef (eps : ℤ) (l : List Std.I64) : List Std.I64 :=
  mixer4_expected (mixer4_denRef (mixer4_sumSq l) eps) l

/-- One destination window: the scaled query row followed by the key row. -/
def mixer5_windowRef (eps q_scale : ℤ) (qs ks : List Std.I64) : List Std.I64 :=
  mixer5_scaleL q_scale (mixer5_hatRef eps qs) ++ mixer5_hatRef eps ks

/-- The destination after the first `c` columns have been written: `dst` with the window of column
`j` spliced in at offset `2j·hk` for every `j < c`.  Structural on the column count, so the loop
invariant is a single equation; `List.setSlice!` writes in place and preserves the length.  -/
def mixer5_dstRef (win : ℕ → List Std.I64) (hk c : ℕ) (dst : List Std.I64) : List Std.I64 :=
  Nat.rec dst (fun n ih => ih.setSlice! ((n * 2) * hk) (win n)) c

/-- No column has been written yet. -/
theorem mixer5_dstRef_zero (win : ℕ → List Std.I64) (hk : ℕ) (dst : List Std.I64) :
    mixer5_dstRef win hk 0 dst = dst := rfl

/-- One more column: the previous destination with this column's window spliced in. -/
theorem mixer5_dstRef_succ (win : ℕ → List Std.I64) (hk c : ℕ) (dst : List Std.I64) :
    mixer5_dstRef win hk (c + 1) dst
      = (mixer5_dstRef win hk c dst).setSlice! ((c * 2) * hk) (win c) := rfl

/-- The query window of column `c` of the flattened conv buffer. -/
def mixer5_qwin (conv : List Std.I64) (kr0 cdim hk c : ℕ) : List Std.I64 :=
  conv.slice (c * cdim + kr0) (c * cdim + kr0 + hk)

/-- The key window of column `c` of the flattened conv buffer. -/
def mixer5_kwin (conv : List Std.I64) (key_dim kr0 cdim hk c : ℕ) : List Std.I64 :=
  conv.slice (c * cdim + key_dim + kr0) (c * cdim + key_dim + kr0 + hk)

/-- The window of column `c`, read out of the flattened conv buffer. -/
def mixer5_winOf (conv : List Std.I64) (eps q_scale : ℤ) (key_dim cdim hk kr0 c : ℕ) :
    List Std.I64 :=
  mixer5_windowRef eps q_scale (mixer5_qwin conv kr0 cdim hk c)
    (mixer5_kwin conv key_dim kr0 cdim hk c)

/-- `mixer5_winOf` unfolded to its two window reads: the form the loop body's splice produces. -/
theorem mixer5_winOf_eq (conv : List Std.I64) (eps q_scale : ℤ) (key_dim cdim hk kr0 c : ℕ) :
    mixer5_winOf conv eps q_scale key_dim cdim hk kr0 c
      = mixer5_windowRef eps q_scale (mixer5_qwin conv kr0 cdim hk c)
        (mixer5_kwin conv key_dim kr0 cdim hk c) := rfl

/-- **The whole output of `norm_qk_q32`**: `dst` with, for every column `c < k`, the window
`2c·hk … 2(c+1)·hk` replaced by the scaled normalised query row followed by the normalised key
row. -/
def mixer5_outRef (conv dst : List Std.I64) (eps q_scale : ℤ) (key_dim cdim hk kr0 k : ℕ) :
    List Std.I64 :=
  mixer5_dstRef (mixer5_winOf conv eps q_scale key_dim cdim hk kr0) hk k dst

/-- The `l2_normalize` domain on a window: `Mixer4`'s hypotheses, stated on the list itself.  The
`1 ≤` / `≤ 2¹¹⁶` pair is the `isqrt` window of `Numeric/Engine/Rsqrt.lean` (`isqrt_spec`), the
quotient clause the `i64` fit of every output word. -/
def mixer5_domain (eps : ℤ) (l : List Std.I64) : Prop :=
  1 ≤ mixer4_sumSq l ∧ mixer4_sumSq l ≤ (2 : ℤ) ^ (116 : ℕ) ∧
  ∀ v ∈ l, I64.min ≤ mixer4_divRef v.val (mixer4_denRef (mixer4_sumSq l) eps) ∧
    mixer4_divRef v.val (mixer4_denRef (mixer4_sumSq l) eps) ≤ I64.max

/-- The product domain on a row: every `mul_q32(v, q_scale)` fits `i64`.  A product outside it is
the `Overflow` refusal of `norm_qk_q32_loop0_loop0`, not a wrap. -/
def mixer5_fit (q_scale : ℤ) (l : List Std.I64) : Prop :=
  ∀ v ∈ l, I64.min ≤ mixer2_mulQ32Ref v.val q_scale ∧
    mixer2_mulQ32Ref v.val q_scale ≤ I64.max

/-- The product domain on the *raw* row, phrased on the normalised output words: this is the
predicate the loop can discharge on `q̂` from a bound on the unnormalised window. -/
def mixer5_scaleFit (q_scale : ℤ) (eps : ℤ) (l : List Std.I64) : Prop :=
  ∀ v ∈ l,
    I64.min ≤ mixer2_mulQ32Ref (mixer4_divRef v.val (mixer4_denRef (mixer4_sumSq l) eps)) q_scale ∧
    mixer2_mulQ32Ref (mixer4_divRef v.val (mixer4_denRef (mixer4_sumSq l) eps)) q_scale ≤ I64.max

/-- The per-column hypotheses of the loop: the two conv windows inside the buffer, the two
`l2_normalize` domains, the product domain of the query row, and the destination window in range
(also bounding the `usize` index arithmetic of one turn). -/
def mixer5_colOk (fa : mixer.FlatAffine) (dst : Slice Std.I64) (key_dim cdim hk : Std.Usize)
    (kr0 : ℕ) (eps q_scale : Std.I64) (c : ℕ) : Prop :=
  c * cdim.val + kr0 + hk.val ≤ fa.conv.val.length ∧
  c * cdim.val + key_dim.val + kr0 + hk.val ≤ fa.conv.val.length ∧
  mixer5_domain eps.val (mixer5_qwin fa.conv.val kr0 cdim.val hk.val c) ∧
  mixer5_domain eps.val (mixer5_kwin fa.conv.val key_dim.val kr0 cdim.val hk.val c) ∧
  mixer5_scaleFit q_scale.val eps.val (mixer5_qwin fa.conv.val kr0 cdim.val hk.val c) ∧
  (c + 1) * 2 * hk.val ≤ dst.val.length ∧
  (c + 1) * 2 ≤ Usize.max

/-! ## §2 · List and arithmetic helpers -/

/-- `List.setSlice!` at an in-range source is the splice `take i ++ s' ++ drop (i + |s'|)`. -/
theorem mixer5_setSlice_eq {α : Type} (l s' : List α) (i : ℕ) (h : s'.length ≤ l.length - i) :
    l.setSlice! i s' = l.take i ++ s' ++ l.drop (i + s'.length) := by
  simp only [List.setSlice!]
  rw [min_eq_left h, List.take_of_length_le (le_refl s'.length)]

/-- Replacing the first `i` entries of `A ++ B` when `A` has length `i`. -/
theorem mixer5_list_set_append {α : Type} (A B : List α) (i : ℕ) (x : α)
    (hA : A.length = i) (hi : i < (A ++ B).length) :
    (A ++ B).set i x = A ++ x :: B.drop 1 := by
  rw [List.set_eq_take_append_cons_drop, if_pos hi]
  have h1 : List.take i (A ++ B) = A := by
    rw [List.take_append_of_le_length (le_of_eq hA.symm), List.take_of_length_le (le_of_eq hA)]
  have h2 : List.drop (i + 1) (A ++ B) = B.drop 1 := by
    rw [List.drop_append]
    have h3 : List.drop (i + 1) A = [] := List.drop_eq_nil_of_le (by rw [hA]; omega)
    have h4 : i + 1 - A.length = 1 := by rw [hA]; omega
    rw [h3, h4, List.nil_append]
  rw [h1, h2]

/-- The tail splice: overwriting `[hk, 2hk)` of `u ++ D.drop hk` with `k'` (both of length `hk`)
leaves `u ++ k'`. -/
theorem mixer5_tail_splice {α : Type} (D u k' : List α) (hk : ℕ)
    (hu : u.length = hk) (hk' : k'.length = hk) (hD : D.length = 2 * hk) :
    (u ++ D.drop hk).setSlice! hk k' = u ++ k' := by
  have hlen : k'.length ≤ (u ++ D.drop hk).length - hk := by
    rw [hk', List.length_append, hu, List.length_drop]
    omega
  rw [mixer5_setSlice_eq _ _ _ hlen]
  have h1 : List.take hk (u ++ D.drop hk) = u := by
    rw [List.take_append_of_le_length (le_of_eq hu.symm), List.take_of_length_le (le_of_eq hu)]
  have h2 : List.drop (hk + k'.length) (u ++ D.drop hk) = [] := by
    rw [hk', List.drop_append]
    have h3 : List.drop (hk + hk) u = [] := List.drop_eq_nil_of_le (by rw [hu]; omega)
    have h4 : hk + hk - u.length = hk := by rw [hu]; omega
    rw [h3, h4, List.drop_drop, show hk + hk = 2 * hk from by omega,
      List.drop_eq_nil_of_le (by rw [hD])]
    rfl
  rw [h1, h2, List.append_nil]

/-- The column splice the loop body performs: the inner write of `u` over `[0, hk)` and the key
copy over `[hk, 2hk)` of the window `D`, then the window written back at `a`. -/
theorem mixer5_col_splice {α : Type} (l D u k' : List α) (a hk : ℕ)
    (hu : u.length = hk) (hk' : k'.length = hk) (hD : D.length = 2 * hk) :
    l.setSlice! a ((u ++ D.drop hk).setSlice! hk k') = l.setSlice! a (u ++ k') := by
  rw [mixer5_tail_splice D u k' hk hu hk' hD]

/-- `mixer5_scaleL` commutes with `take`. -/
theorem mixer5_scaleL_take (q_scale : ℤ) (l : List Std.I64) (n : ℕ) :
    mixer5_scaleL q_scale (l.take n) = (mixer5_scaleL q_scale l).take n := by
  simp only [mixer5_scaleL, List.map_take]

/-- One more entry: `take (i+1)` peels off the `i`-th scaled word. -/
theorem mixer5_scaleL_take_succ (q_scale : ℤ) (l : List Std.I64) (i : ℕ) (h : i < l.length) :
    mixer5_scaleL q_scale (l.take (i + 1))
      = mixer5_scaleL q_scale (l.take i) ++ [mixer5_mulW q_scale l[i]] := by
  rw [List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some]
  simp only [mixer5_scaleL, List.map_append, List.map_cons, List.map_nil]

/-- A slice taken at `[a, a + hk)` has length `hk`. -/
theorem mixer5_slice_len (l : List Std.I64) (a hk : ℕ) (h : a + hk ≤ l.length) :
    (l.slice a (a + hk)).length = hk := by
  rw [List.slice, List.length_take, List.length_drop]
  omega

/-- The normalised row is in the product domain whenever the raw row is. -/
theorem mixer5_fit_hatRef (q_scale eps : ℤ) (l : List Std.I64)
    (h : ∀ v ∈ l, I64.min ≤ mixer4_divRef v.val (mixer4_denRef (mixer4_sumSq l) eps) ∧
      mixer4_divRef v.val (mixer4_denRef (mixer4_sumSq l) eps) ≤ I64.max)
    (hs : mixer5_scaleFit q_scale eps l) :
    mixer5_fit q_scale (mixer5_hatRef eps l) := by
  intro v hv
  rw [mixer5_hatRef, mixer4_expected, List.mem_map] at hv
  obtain ⟨u, hu, rfl⟩ := hv
  have h1 := h u hu
  have h2 : (mixer4_outRef (mixer4_denRef (mixer4_sumSq l) eps) u).val
      = mixer4_divRef u.val (mixer4_denRef (mixer4_sumSq l) eps) :=
    mixer4_outRef_val _ u h1.1 h1.2
  rw [h2]
  exact hs u hu

/-- The two destination offsets of one column differ by `2·hk`: the `i64` word count the body
writes. -/
theorem mixer5_off_sub (c hk : ℕ) : (c + 1) * 2 * hk - c * 2 * hk = 2 * hk := by
  have h : (c + 1) * 2 * hk = c * 2 * hk + 2 * hk := by ring
  omega

/-- `(c·2)·hk` grows with `c`, so the destination bound of column `c + 1` bounds column `c`. -/
theorem mixer5_off_mono (c hk : ℕ) : c * 2 * hk ≤ (c + 1) * 2 * hk :=
  Nat.mul_le_mul_right _ (by omega)

/-! ## §3 · `mixer.norm_qk_q32_loop0_loop0` — the per-row scaling loop

The inner loop maps `q_hat[i] ← mul_q32(q_hat[i], q_scale)` over the first `hk` entries of the
destination sub-slice, stopping at the first product that does not fit `i64`.  Its reference is the
prefix update: after `i` turns the destination is the scaled prefix followed by the untouched
suffix. -/

/-- **One inner turn.**  For a cursor inside `hk`, a running destination equal to the scaled prefix
of `i` words followed by the untouched suffix, and a product that fits `i64`, the body writes that
product, keeps `err = none`, and advances the cursor. -/
theorem mixer5_inner_body_step (d : Slice Std.I64) (q_hat : alloc.vec.Vec Std.I64)
    (q_scale : Std.I64) (hk i : Std.Usize) (dcur : Slice Std.I64)
    (hilt : i.val < hk.val)
    (hqlen : hk.val ≤ q_hat.val.length)
    (hlen : hk.val ≤ d.val.length)
    (hfit : mixer5_fit q_scale.val q_hat.val)
    (hinv : dcur.val = mixer5_scaleL q_scale.val (q_hat.val.take i.val) ++ d.val.drop i.val) :
    ∃ (d' : Slice Std.I64) (i2 : Std.Usize),
      mixer.norm_qk_q32_loop0_loop0.body hk q_scale q_hat none dcur i
          = ok (ControlFlow.cont (none, d', i2)) ∧
      i2.val = i.val + 1 ∧
      d'.val = mixer5_scaleL q_scale.val (q_hat.val.take (i.val + 1)) ++ d.val.drop (i.val + 1) := by
  have hik : i.val < q_hat.val.length := by omega
  have htake : (q_hat.val.take i.val).length = i.val := by
    rw [List.length_take]; omega
  have hdcurlen : dcur.val.length = d.val.length := by
    rw [hinv, List.length_append, mixer5_scaleL, List.length_map, htake, List.length_drop]
    omega
  have hidlen : i.val < dcur.val.length := by rw [hdcurlen]; omega
  unfold mixer.norm_qk_q32_loop0_loop0.body
  dsimp only
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hilt)]
  rw [if_pos (by rfl : core.option.Option.is_none (none : Option step.StepError) = true)]
  rw [alloc.vec.Vec.index_slice_index]
  obtain ⟨x, hx, hxv⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec q_hat i hik)
  rw [hx]
  simp only [bind_tc_ok]
  have hxval : x.val = (q_hat.val[i.val]'hik).val := by rw [hxv]
  have hfitx : I64.min ≤ mixer2_mulQ32Ref x.val q_scale.val ∧
      mixer2_mulQ32Ref x.val q_scale.val ≤ I64.max :=
    hfit x (by rw [hxv]; exact List.getElem_mem hik)
  obtain ⟨w, hw, hwv⟩ := mixer2_mul_q32_val x q_scale hfitx
  rw [hw]
  simp only [bind_tc_ok]
  obtain ⟨d1, hd1, hd1v⟩ := WP.spec_imp_exists (Slice.update_spec dcur i w hidlen)
  rw [hd1]
  simp only [bind_tc_ok]
  have h1v : ((1#usize : Std.Usize)).val = 1 := by decide
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := 1#usize) (by rw [h1v]; have := Slice.length_ineq d; omega))
  rw [hi2]
  simp only [bind_tc_ok]
  have hfitv : I64.min ≤ mixer2_mulQ32Ref (q_hat.val[i.val]'hik).val q_scale.val ∧
      mixer2_mulQ32Ref (q_hat.val[i.val]'hik).val q_scale.val ≤ I64.max := by
    rw [← hxval]; exact hfitx
  have hwref : w = mixer5_mulW q_scale.val (q_hat.val[i.val]'hik) := by
    apply IScalar.eq_of_val_eq
    rw [hwv, hxval]
    exact (mixer5_mulW_val q_scale.val (q_hat.val[i.val]'hik) hfitv.1 hfitv.2).symm
  refine ⟨d1, i2, rfl, by rw [hi2v, h1v], ?_⟩
  rw [hd1v, Slice.set_val_eq, hinv]
  have hA : (mixer5_scaleL q_scale.val (q_hat.val.take i.val)).length = i.val := by
    rw [mixer5_scaleL, List.length_map, htake]
  rw [mixer5_list_set_append _ _ i.val w hA
    (by rw [List.length_append, hA, List.length_drop]; omega)]
  rw [mixer5_scaleL_take_succ q_scale.val q_hat.val i.val hik, List.drop_drop, hwref]
  simp only [List.append_assoc, List.singleton_append]

/-- **`norm_qk_q32_loop0_loop0` scales the first `hk` destination words.**  From cursor zero, with
the products of the query row inside `i64` and `hk` inside both the row and the destination, the
loop returns `err = none` and the destination with its first `hk` words scaled by `q_scale`.
`proved`, relative to the boundary closure of the loop (it contains none of its own). -/
theorem mixer5_inner_spec (d : Slice Std.I64) (q_hat : alloc.vec.Vec Std.I64) (q_scale : Std.I64)
    (hk : Std.Usize)
    (hqlen : hk.val ≤ q_hat.val.length)
    (hlen : hk.val ≤ d.val.length)
    (hfit : mixer5_fit q_scale.val q_hat.val) :
    ∃ d1 : Slice Std.I64,
      mixer.norm_qk_q32_loop0_loop0 hk q_scale none q_hat d 0#usize = ok (none, d1) ∧
      d1.val = mixer5_scaleL q_scale.val (q_hat.val.take hk.val) ++ d.val.drop hk.val := by
  have htriple : mixer.norm_qk_q32_loop0_loop0 hk q_scale none q_hat d 0#usize ⦃ r =>
      r.1 = none ∧ ∃ d1 : Slice Std.I64, r.2 = d1 ∧
        d1.val = mixer5_scaleL q_scale.val (q_hat.val.take hk.val) ++ d.val.drop hk.val ⦄ := by
    unfold mixer.norm_qk_q32_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun s : (Option step.StepError) × Slice Std.I64 × Std.Usize =>
        hk.val - s.2.2.val)
      (inv := fun s : (Option step.StepError) × Slice Std.I64 × Std.Usize =>
        s.1 = none ∧ s.2.2.val ≤ hk.val ∧
        s.2.1.val = mixer5_scaleL q_scale.val (q_hat.val.take s.2.2.val) ++ d.val.drop s.2.2.val)
      (post := fun y : (Option step.StepError) × Slice Std.I64 =>
        y.1 = none ∧ ∃ d1 : Slice Std.I64, y.2 = d1 ∧
          d1.val = mixer5_scaleL q_scale.val (q_hat.val.take hk.val) ++ d.val.drop hk.val)
    · rintro ⟨err, dcur, i⟩ ⟨herr, hile, hinv⟩
      dsimp only at herr hile hinv
      cases herr
      by_cases hlt : i < hk
      · have hilt : i.val < hk.val := by rw [UScalar.lt_equiv] at hlt; exact hlt
        obtain ⟨d', i2, hbody, hi2v, hd'v⟩ :=
          mixer5_inner_body_step d q_hat q_scale hk i dcur hilt hqlen hlen hfit hinv
        dsimp only
        rw [hbody]
        simp only [WP.spec_ok]
        refine ⟨⟨trivial, by omega, by rw [hd'v, hi2v]⟩, by omega⟩
      · have hge : hk.val ≤ i.val :=
          le_of_not_gt (by simpa only [UScalar.lt_equiv] using hlt)
        have hieq : i.val = hk.val := by omega
        dsimp only
        unfold mixer.norm_qk_q32_loop0_loop0.body
        dsimp only
        rw [if_neg hlt]
        simp only [WP.spec_ok]
        exact ⟨trivial, dcur, rfl, by rw [hinv, hieq]⟩
    · refine ⟨rfl, by rw [show (0#usize).val = 0 from by decide]; omega, ?_⟩
      dsimp only
      rw [show (0#usize : Std.Usize).val = 0 from by decide]
      simp only [mixer5_scaleL, List.take_zero, List.map_nil, List.nil_append, List.drop_zero]
  obtain ⟨y, hy, hyerr, d1, hd1, hd1v⟩ := WP.spec_imp_exists htriple
  refine ⟨d1, ?_, hd1v⟩
  have hyeq : y = (none, d1) := Prod.ext hyerr hd1
  rw [hy, hyeq]

/-! ## §4 · `mixer.norm_qk_q32_loop0` — one body turn

The outer loop's turn reads the two conv windows of column `c`, normalises both, and writes the
destination window. -/

/-- **One outer turn.**  For a column `c` inside `fa.k`, with both conv windows inside the buffer,
both `l2_normalize` domains and the query product domain satisfied, and the destination window in
range, the body returns the destination with that column's window spliced in at `2c·hk`, keeps
`err = none`, and advances the column cursor. -/
theorem mixer5_loop0_body_step
    (fa : mixer.FlatAffine) (key_dim cdim hk : Std.Usize) (kr0 : Std.Usize) (eps q_scale : Std.I64)
    (acc : Slice Std.I64) (c : Std.Usize) (qs ks : List Std.I64)
    (hclt : c.val < fa.k.val)
    (hqsl : qs.length = hk.val) (hksl : ks.length = hk.val)
    (hhk : hk.val < Usize.max)
    (hqslice : fa.conv.val.slice (c.val * cdim.val + kr0.val) (c.val * cdim.val + kr0.val + hk.val)
      = qs)
    (hkslice : fa.conv.val.slice (c.val * cdim.val + key_dim.val + kr0.val)
      (c.val * cdim.val + key_dim.val + kr0.val + hk.val) = ks)
    (hconvq : c.val * cdim.val + kr0.val + hk.val ≤ fa.conv.val.length)
    (hconvk : c.val * cdim.val + key_dim.val + kr0.val + hk.val ≤ fa.conv.val.length)
    (hdomq : mixer5_domain eps.val qs) (hdomk : mixer5_domain eps.val ks)
    (hsfitq : mixer5_scaleFit q_scale.val eps.val qs)
    (hdst : (c.val + 1) * 2 * hk.val ≤ acc.val.length)
    (hi9 : (c.val + 1) * 2 ≤ Usize.max) :
    ∃ (acc1 : Slice Std.I64) (c1 : Std.Usize),
      mixer.norm_qk_q32_loop0.body fa key_dim cdim hk eps q_scale kr0 acc none c
          = ok (ControlFlow.cont (acc1, none, c1)) ∧
      c1.val = c.val + 1 ∧
      acc1.val = acc.val.setSlice! ((c.val * 2) * hk.val)
        (mixer5_windowRef eps.val q_scale.val qs ks) := by
  have hconvmax : fa.conv.val.length ≤ Usize.max := alloc.vec.Vec.property fa.conv
  have haccmax : acc.val.length ≤ Usize.max := Slice.length_ineq acc
  have h1v : ((1#usize : Std.Usize)).val = 1 := by decide
  have h2v : ((2#usize : Std.Usize)).val = 2 := by decide
  -- the two conv windows lie inside the buffer
  have hi1le : c.val * cdim.val + kr0.val + hk.val ≤ fa.conv.val.length := hconvq
  unfold mixer.norm_qk_q32_loop0.body
  dsimp only
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hclt)]
  rw [if_pos (by rfl : core.option.Option.is_none (none : Option step.StepError) = true)]
  -- base = c * cdim
  obtain ⟨base, hbase, hbasev⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := c) (y := cdim) (by omega))
  rw [hbase]
  simp only [bind_tc_ok]
  -- i = base + kr0
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := base) (y := kr0) (by omega))
  rw [hi]
  simp only [bind_tc_ok]
  -- i1 = i + hk
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := hk) (by omega))
  rw [hi1]
  simp only [bind_tc_ok]
  have hiv' : (i.val : ℕ) = c.val * cdim.val + kr0.val := by rw [hiv, hbasev]
  have hi1v' : (i1.val : ℕ) = c.val * cdim.val + kr0.val + hk.val := by rw [hi1v, hiv']
  -- s = conv[i … i1]
  obtain ⟨s, hs, hsval, hslen⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_Range_spec fa.conv { start := i, «end» := i1 }
      (by rw [UScalar.le_equiv, hiv, hi1v]; omega)
      (by rw [hi1v, hiv, hbasev]; omega))
  rw [hs]
  simp only [bind_tc_ok]
  have hsval' : s.val = qs := by rw [hsval, hiv', hi1v', hqslice]
  have hdomq_s : mixer5_domain eps.val s.val := by rw [hsval']; exact hdomq
  -- q_hat = l2_normalize s eps
  obtain ⟨q_hat, hq, hqv⟩ := word_l2_normalize_ok s eps
    (by rw [hsval', mixer2_i128_max_eq]
        have h := hdomq.2.1
        have h2 : (2 : ℤ) ^ (116 : ℕ) ≤ mixer2_i128Max := by norm_num [mixer2_i128Max]
        linarith)
    (by rw [hsval']; exact hdomq.1)
    (by rw [hsval']; exact hdomq.2.1)
    (by rw [hsval', hqsl]; exact hhk)
    (by intro j hj; exact hdomq_s.2.2 _ (List.getElem_mem hj))
  rw [hq]
  simp only [bind_tc_ok]
  have hqhatval : q_hat.val = mixer5_hatRef eps.val qs := by rw [hqv, hsval', mixer5_hatRef]
  -- i2 = base + key_dim
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := base) (y := key_dim) (by omega))
  rw [hi2]
  simp only [bind_tc_ok]
  -- i3 = i2 + kr0
  obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i2) (y := kr0) (by omega))
  rw [hi3]
  simp only [bind_tc_ok]
  have hi3v' : (i3.val : ℕ) = c.val * cdim.val + key_dim.val + kr0.val := by rw [hi3v, hi2v, hbasev]
  -- i5 = i3 + hk (the extraction's second `i2 + kr0` collapses to `i3`)
  obtain ⟨i5, hi5, hi5v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i3) (y := hk) (by omega))
  rw [hi5]
  simp only [bind_tc_ok]
  have hi5v' : (i5.val : ℕ) = c.val * cdim.val + key_dim.val + kr0.val + hk.val := by
    rw [hi5v, hi3v']
  -- s1 = conv[i3 … i5]
  obtain ⟨s1, hs1, hs1val, hs1len⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_Range_spec fa.conv { start := i3, «end» := i5 }
      (by rw [UScalar.le_equiv, hi3v, hi5v]; omega)
      (by rw [hi5v, hi3v, hi2v, hbasev]; omega))
  rw [hs1]
  simp only [bind_tc_ok]
  have hs1val' : s1.val = ks := by rw [hs1val, hi3v', hi5v', hkslice]
  have hdomk_s : mixer5_domain eps.val s1.val := by rw [hs1val']; exact hdomk
  -- k_hat = l2_normalize s1 eps
  obtain ⟨k_hat, hk_, hk_v⟩ := word_l2_normalize_ok s1 eps
    (by rw [hs1val', mixer2_i128_max_eq]
        have h := hdomk.2.1
        have h2 : (2 : ℤ) ^ (116 : ℕ) ≤ mixer2_i128Max := by norm_num [mixer2_i128Max]
        linarith)
    (by rw [hs1val']; exact hdomk.1)
    (by rw [hs1val']; exact hdomk.2.1)
    (by rw [hs1val', hksl]; exact hhk)
    (by intro j hj; exact hdomk_s.2.2 _ (List.getElem_mem hj))
  rw [hk_]
  simp only [bind_tc_ok]
  have hkhatval : k_hat.val = mixer5_hatRef eps.val ks := by rw [hk_v, hs1val', mixer5_hatRef]
  -- the destination window offsets
  obtain ⟨i6, hi6, hi6v⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := c) (y := 2#usize) (by rw [h2v]; omega))
  rw [hi6]
  simp only [bind_tc_ok]
  obtain ⟨i7, hi7, hi7v⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := i6) (y := hk)
      (by rw [hi6v, h2v]
          exact le_trans (mixer5_off_mono c.val hk.val) (le_trans hdst haccmax)))
  rw [hi7]
  simp only [bind_tc_ok]
  obtain ⟨i8, hi8, hi8v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := c) (y := 1#usize) (by omega))
  rw [hi8]
  simp only [bind_tc_ok]
  obtain ⟨i9v, hi9e, hi9v⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := i8) (y := 2#usize) (by rw [hi8v, h1v, h2v]; omega))
  rw [hi9e]
  simp only [bind_tc_ok]
  obtain ⟨i10, hi10, hi10v⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := i9v) (y := hk) (by rw [hi9v, hi8v, h1v, h2v]; omega))
  rw [hi10]
  simp only [bind_tc_ok]
  have hi7v' : i7.val = c.val * 2 * hk.val := by rw [hi7v, hi6v, h2v]
  have hi8v' : i8.val = c.val + 1 := by rw [hi8v, h1v]
  have hi9v' : i9v.val = (c.val + 1) * 2 := by rw [hi9v, hi8v', h2v]
  have hi10v' : i10.val = (c.val + 1) * 2 * hk.val := by rw [hi10v, hi9v']
  have hi7i10 : i7.val ≤ i10.val := by rw [hi7v', hi10v']; exact mixer5_off_mono c.val hk.val
  have hi10len : i10.val ≤ acc.val.length := by rw [hi10v']; exact hdst
  -- the mutable destination window
  rw [Slice.index_mut_SliceIndexRangeUsizeSliceInst]
  obtain ⟨p, hd, hdval, hdlen, hdback⟩ := WP.spec_imp_exists
    (core.slice.index.SliceIndexRangeUsizeSlice.index_mut.step_spec (α := Std.I64)
      { start := i7, «end» := i10 } acc
      (by rw [UScalar.le_equiv]; exact hi7i10)
      (by rw [hi10v']; exact hdst))
  rcases p with ⟨d, imb⟩
  dsimp only at hdval hdlen hdback
  rw [hd]
  simp only [bind_tc_ok]
  have hDlen : d.val.length = 2 * hk.val := by
    change d.length = 2 * hk.val
    rw [hdlen, hi10v', hi7v', mixer5_off_sub]
  -- the inner scaling loop
  obtain ⟨d1, hd1eq, hd1v⟩ := mixer5_inner_spec d q_hat q_scale hk
    (by rw [hqhatval, mixer5_hatRef, mixer4_expected, List.length_map, hqsl])
    (by rw [hDlen]; omega)
    (by rw [hqhatval]
        exact mixer5_fit_hatRef q_scale.val eps.val qs hdomq.2.2 hsfitq)
  change ∃ acc1 c1,
    (do
      let (err1, d1) ← mixer.norm_qk_q32_loop0_loop0 hk q_scale none q_hat d 0#usize
      let dst1 ←
        if core.option.Option.is_none err1 = true then
          (do
            let (s2, index_mut_back1) ←
              core.slice.index.Slice.index_mut (core.slice.index.SliceIndexRangeFromUsizeSlice Std.I64) d1
                { start := hk }
            let s4 ← core.slice.Slice.copy_from_slice core.marker.CopyI64 s2 k_hat.deref
            ok (imb (index_mut_back1 s4)))
        else ok (imb d1)
      ok (ControlFlow.cont (dst1, err1, i8))) =
        ok (ControlFlow.cont (acc1, none, c1)) ∧
      c1.val = c.val + 1 ∧
      acc1.val = acc.val.setSlice! (c.val * 2 * hk.val) (mixer5_windowRef (↑eps) (↑q_scale) qs ks)
  rw [hd1eq]
  simp only [bind_tc_ok]
  change ∃ acc1 c1,
    (do
      let dst1 ←
        if core.option.Option.is_none (none : Option step.StepError) = true then
          (do
            let (s2, index_mut_back1) ←
              core.slice.index.Slice.index_mut (core.slice.index.SliceIndexRangeFromUsizeSlice Std.I64) d1
                { start := hk }
            let s4 ← core.slice.Slice.copy_from_slice core.marker.CopyI64 s2 k_hat.deref
            ok (imb (index_mut_back1 s4)))
        else ok (imb d1)
      ok (ControlFlow.cont (dst1, none, i8))) =
        ok (ControlFlow.cont (acc1, none, c1)) ∧
      c1.val = c.val + 1 ∧
      acc1.val = acc.val.setSlice! (c.val * 2 * hk.val) (mixer5_windowRef (↑eps) (↑q_scale) qs ks)
  -- the error flag is still none, so the key row is copied in
  rw [if_pos (by rfl : core.option.Option.is_none (none : Option step.StepError) = true)]
  have hqhatlen : q_hat.val.length = hk.val := by
    rw [hqhatval, mixer5_hatRef, mixer4_expected, List.length_map, hqsl]
  have hd1len : d1.val.length = 2 * hk.val := by
    rw [hd1v, List.length_append, List.length_drop, hDlen, mixer5_scaleL, List.length_map,
      List.length_take, hqhatlen, Nat.min_self]
    omega
  rw [Slice.index_mut_SliceIndexRangeFromUsizeSliceInst]
  obtain ⟨p2, hs2, hs2val, hs2len, hs2back⟩ := WP.spec_imp_exists
    (core.slice.index.SliceIndexRangeFromUsizeSlice.index_mut.step_spec (α := Std.I64)
      { start := hk } d1 (by change hk.val ≤ d1.val.length; rw [hd1len]; omega))
  rcases p2 with ⟨s2, imb1⟩
  dsimp only at hs2val hs2len hs2back
  rw [hs2]
  simp only [bind_tc_ok]
  change ∃ acc1 c1,
    (do
      let dst1 ←
        (do
          let s4 ← core.slice.Slice.copy_from_slice core.marker.CopyI64 s2 k_hat.deref
          ok (imb (imb1 s4)))
      ok (ControlFlow.cont (dst1, none, i8))) =
        ok (ControlFlow.cont (acc1, none, c1)) ∧
      c1.val = c.val + 1 ∧
      acc1.val = acc.val.setSlice! (c.val * 2 * hk.val) (mixer5_windowRef (↑eps) (↑q_scale) qs ks)
  have hs2len' : s2.length = hk.val := by
    simp only [Slice.length] at hs2len ⊢
    rw [hs2len, hd1len]; omega
  have hklv : (alloc.vec.Vec.deref k_hat).length = hk.val := by
    simp only [Slice.length, alloc.vec.Vec.deref, Slice.from_val]
    rw [hkhatval, mixer5_hatRef, mixer4_expected, List.length_map, hksl]
  have hk'len : k_hat.val.length = hk.val := by
    rw [hkhatval, mixer5_hatRef, mixer4_expected, List.length_map, hksl]
  obtain ⟨s4, hs4, hs4v⟩ := WP.spec_imp_exists
    (core.slice.Slice.copy_from_slice.step_spec core.marker.CopyI64 s2 (alloc.vec.Vec.deref k_hat)
      (by rw [hs2len', hklv]))
  rw [hs4]
  simp only [bind_tc_ok]
  refine ⟨imb (imb1 s4), i8, rfl, hi8v', ?_⟩
  -- the value of the written destination
  have hs4val : s4.val = k_hat.val := by rw [hs4v]; simp only [alloc.vec.Vec.deref, Slice.from_val]
  have himb1 : (imb1 s4).val = d1.val.setSlice! hk.val k_hat.val := by
    rw [hs2back s4, hs4val]
  have himb : (imb (imb1 s4)).val = acc.val.setSlice! i7.val (d1.val.setSlice! hk.val k_hat.val) := by
    rw [show (imb (imb1 s4)).val = (acc.setSlice! i7.val (imb1 s4).val).val from
        congrArg Slice.val (hdback (imb1 s4)),
      Slice.setSlice!_val, himb1]
  rw [himb, hi7v', hd1v]
  rw [List.take_of_length_le (le_of_eq hqhatlen), hqhatval]
  rw [mixer5_col_splice acc.val d.val (mixer5_scaleL q_scale.val (mixer5_hatRef eps.val qs))
      k_hat.val ((c.val * 2) * hk.val) hk.val
      (by rw [mixer5_scaleL, List.length_map, mixer5_hatRef, mixer4_expected, List.length_map,
        hqsl])
      hk'len
      hDlen]
  rw [hkhatval, mixer5_windowRef]

/-! ## §5 · `mixer.norm_qk_q32_loop0` — the column loop

The invariant is the reference: after `c` turns the destination is `mixer5_dstRef` at `c`. -/

/-- **`norm_qk_q32_loop0` writes every column window.**  Under the per-column hypotheses the loop
returns `err = none` and the reference output.  `proved`, relative to the boundary closure of
`mixer.norm_qk_q32_loop0` (which carries the `l2_normalize` and `i64`-narrowing boundaries). -/
theorem mixer5_loop0_spec (fa : mixer.FlatAffine) (key_dim cdim hk kr0 : Std.Usize)
    (eps q_scale : Std.I64) (dst : Slice Std.I64)
    (hhk : hk.val < Usize.max)
    (hcol : ∀ c, c < fa.k.val → mixer5_colOk fa dst key_dim cdim hk kr0.val eps q_scale c) :
    ∃ dst1 : Slice Std.I64,
      mixer.norm_qk_q32_loop0 fa key_dim cdim hk eps q_scale dst kr0 none 0#usize
          = ok (dst1, none) ∧
      dst1.val = mixer5_outRef fa.conv.val dst.val eps.val q_scale.val key_dim.val cdim.val hk.val
        kr0.val fa.k.val := by
  have htriple : mixer.norm_qk_q32_loop0 fa key_dim cdim hk eps q_scale dst kr0 none 0#usize ⦃ r =>
      ∃ dst1 : Slice Std.I64, r = (dst1, none) ∧
        dst1.val = mixer5_outRef fa.conv.val dst.val eps.val q_scale.val key_dim.val cdim.val hk.val
          kr0.val fa.k.val ⦄ := by
    unfold mixer.norm_qk_q32_loop0
    apply loop.spec_decr_nat
      (measure := fun s : (Slice Std.I64) × (Option step.StepError) × Std.Usize =>
        fa.k.val - s.2.2.val)
      (inv := fun s : (Slice Std.I64) × (Option step.StepError) × Std.Usize =>
        s.2.1 = none ∧ s.2.2.val ≤ fa.k.val ∧ s.1.val.length = dst.val.length ∧
        s.1.val = mixer5_dstRef (mixer5_winOf fa.conv.val eps.val q_scale.val key_dim.val cdim.val
          hk.val kr0.val) hk.val s.2.2.val dst.val)
      (post := fun y : (Slice Std.I64) × (Option step.StepError) =>
        ∃ dst1 : Slice Std.I64, y = (dst1, none) ∧
          dst1.val = mixer5_outRef fa.conv.val dst.val eps.val q_scale.val key_dim.val cdim.val
            hk.val kr0.val fa.k.val)
    · rintro ⟨acc, err, c⟩ ⟨herr, hcle, hacclen, hinv⟩
      dsimp only at herr hcle hacclen hinv
      cases herr
      by_cases hlt : c < fa.k
      · have hclt : c.val < fa.k.val := by rw [UScalar.lt_equiv] at hlt; exact hlt
        obtain ⟨hconvq, hconvk, hdomq, hdomk, hsfitq, hdst, hi9⟩ := hcol c.val hclt
        have hqsl : (mixer5_qwin fa.conv.val kr0.val cdim.val hk.val c.val).length = hk.val :=
          mixer5_slice_len fa.conv.val _ _ hconvq
        have hksl : (mixer5_kwin fa.conv.val key_dim.val kr0.val cdim.val hk.val c.val).length
            = hk.val := mixer5_slice_len fa.conv.val _ _ hconvk
        have hdst' : (c.val + 1) * 2 * hk.val ≤ acc.val.length := by
          rw [hacclen]; exact hdst
        obtain ⟨acc1, c1, hbody, hc1v, hacc1v⟩ :=
          mixer5_loop0_body_step fa key_dim cdim hk kr0 eps q_scale acc c
            (mixer5_qwin fa.conv.val kr0.val cdim.val hk.val c.val)
            (mixer5_kwin fa.conv.val key_dim.val kr0.val cdim.val hk.val c.val)
            hclt hqsl hksl hhk rfl rfl hconvq hconvk hdomq hdomk hsfitq hdst' hi9
        dsimp only
        rw [hbody]
        simp only [WP.spec_ok]
        refine ⟨⟨trivial, by omega, ?_, ?_⟩, by omega⟩
        · rw [hacc1v, List.length_setSlice!]; exact hacclen
        · rw [hacc1v, hinv, hc1v, mixer5_dstRef_succ, mixer5_winOf_eq]
      · have hge : fa.k.val ≤ c.val :=
          le_of_not_gt (by simpa only [UScalar.lt_equiv] using hlt)
        have hceq : c.val = fa.k.val := by omega
        dsimp only
        unfold mixer.norm_qk_q32_loop0.body
        dsimp only
        rw [if_neg hlt]
        simp only [WP.spec_ok]
        exact ⟨acc, rfl, by rw [hinv, hceq, mixer5_outRef]⟩
    · refine ⟨rfl, by rw [show (0#usize : Std.Usize).val = 0 from by decide]; omega, rfl, ?_⟩
      rw [show (0#usize : Std.Usize).val = 0 from by decide, mixer5_dstRef_zero]
  obtain ⟨y, hy, dst1, hdst1, hdst1v⟩ := WP.spec_imp_exists htriple
  refine ⟨dst1, ?_, hdst1v⟩
  rw [hy, hdst1]

/-! ## §6 · `mixer.norm_qk_q32` — the driver

`kr0 = kh * hk`, then the column loop, then the error mapping. -/

/-- **`norm_qk_q32` normalises and scales every column in Q32.32.**  Under the stated domain — the
per-column bounds `mixer5_colOk`, the width cap `hk < Usize.max` and the `usize` product `kh·hk` —
the extracted body returns `Ok ()` with the reference output.  `proved`, relative to the boundary
closure of `mixer.norm_qk_q32`. -/
theorem word_norm_qk_q32_ok (fa : mixer.FlatAffine) (key_dim cdim hk kh : Std.Usize)
    (eps q_scale : Std.I64) (dst : Slice Std.I64)
    (hkr0 : kh.val * hk.val ≤ Usize.max)
    (hhk : hk.val < Usize.max)
    (hcol : ∀ c, c < fa.k.val →
      mixer5_colOk fa dst key_dim cdim hk (kh.val * hk.val) eps q_scale c) :
    ∃ dst1 : Slice Std.I64,
      mixer.norm_qk_q32 fa key_dim cdim hk eps q_scale kh dst
          = ok (core.result.Result.Ok (), dst1) ∧
      dst1.val = mixer5_outRef fa.conv.val dst.val eps.val q_scale.val key_dim.val cdim.val hk.val
        (kh.val * hk.val) fa.k.val := by
  unfold mixer.norm_qk_q32
  obtain ⟨kr0, hkr0eq, hkr0v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := kh) (y := hk) hkr0)
  rw [hkr0eq]
  simp only [bind_tc_ok]
  obtain ⟨dst1, hloop, hloopv⟩ :=
    mixer5_loop0_spec fa key_dim cdim hk kr0 eps q_scale dst hhk
      (fun c hc => by rw [hkr0v]; exact hcol c hc)
  rw [hloop]
  simp only [bind_tc_ok]
  have hloopv' : dst1.val = mixer5_outRef fa.conv.val dst.val eps.val q_scale.val key_dim.val
      cdim.val hk.val (kh.val * hk.val) fa.k.val := by rw [hloopv, hkr0v]
  exact ⟨dst1, rfl, hloopv'⟩

/-! ## §7 · The refusal

`norm_qk_q32_loop0` stops at the first column whose turn sets `err`; from that point the state no
longer changes, and the driver maps the carried `StepError` into `Err`. -/

/-- **The body turn is `done` once the error is set.**  With `err ≠ none` the body's second guard
fails, so it returns its input unchanged, whatever the column cursor. -/
theorem mixer5_loop0_body_done (fa : mixer.FlatAffine) (key_dim cdim hk : Std.Usize)
    (eps q_scale : Std.I64) (kr0 : Std.Usize) (dst : Slice Std.I64) (err : Option step.StepError)
    (c : Std.Usize) (herr : err ≠ none) :
    mixer.norm_qk_q32_loop0.body fa key_dim cdim hk eps q_scale kr0 dst err c
      = ok (ControlFlow.done (dst, err)) := by
  unfold mixer.norm_qk_q32_loop0.body
  dsimp only
  by_cases hc : c < fa.k
  · rw [if_pos hc]
    have hb : core.option.Option.is_none err = false := by
      cases err with
      | none => exact absurd rfl herr
      | some e => rfl
    rw [if_neg (by rw [hb]; exact Bool.false_ne_true)]
  · rw [if_neg hc]

/-- **An error already in the state is carried out unchanged.**  Once `err = some e` the loop's
body takes its `done` branch on every turn, so the loop returns the destination it was given. -/
theorem mixer5_loop0_err_absorbs (fa : mixer.FlatAffine) (key_dim cdim hk kr0 : Std.Usize)
    (eps q_scale : Std.I64) (dst : Slice Std.I64) (e : step.StepError) (c : Std.Usize)
    (hc : c.val ≤ fa.k.val) :
    mixer.norm_qk_q32_loop0 fa key_dim cdim hk eps q_scale dst kr0 (some e) c
      = ok (dst, some e) := by
  have htriple : mixer.norm_qk_q32_loop0 fa key_dim cdim hk eps q_scale dst kr0 (some e) c
      ⦃ r => r = (dst, some e) ⦄ := by
    unfold mixer.norm_qk_q32_loop0
    apply loop.spec_decr_nat
      (measure := fun s : (Slice Std.I64) × (Option step.StepError) × Std.Usize =>
        fa.k.val - s.2.2.val)
      (inv := fun s : (Slice Std.I64) × (Option step.StepError) × Std.Usize =>
        s.1 = dst ∧ s.2.1 = some e ∧ s.2.2.val ≤ fa.k.val)
      (post := fun y : (Slice Std.I64) × (Option step.StepError) => y = (dst, some e))
    · rintro ⟨acc, err, c'⟩ ⟨hacc, herr, hcle⟩
      dsimp only at hacc herr hcle ⊢
      rw [hacc, herr]
      rw [mixer5_loop0_body_done fa key_dim cdim hk eps q_scale kr0 dst (some e) c' (by simp)]
      simp only [WP.spec_ok]
    · exact ⟨rfl, rfl, hc⟩
  obtain ⟨y, hy, hyeq⟩ := WP.spec_imp_exists htriple
  rw [hy, hyeq]

/-- **The first turn.**  If the `err = none` state at cursor zero takes its `cont` branch and sets
`err = some e`, the loop returns `ok (dst, some e)`: the successor state absorbs and the loop ends. -/
theorem mixer5_loop0_first_turn (fa : mixer.FlatAffine) (key_dim cdim hk kr0 : Std.Usize)
    (eps q_scale : Std.I64) (dst : Slice Std.I64) (e : step.StepError)
    (hbody : mixer.norm_qk_q32_loop0.body fa key_dim cdim hk eps q_scale kr0 dst none 0#usize
      = ok (ControlFlow.cont (dst, some e, 1#usize))) :
    mixer.norm_qk_q32_loop0 fa key_dim cdim hk eps q_scale dst kr0 none 0#usize
      = ok (dst, some e) := by
  have htriple : mixer.norm_qk_q32_loop0 fa key_dim cdim hk eps q_scale dst kr0 none 0#usize
      ⦃ r => r = (dst, some e) ⦄ := by
    unfold mixer.norm_qk_q32_loop0
    apply loop.spec_decr_nat
      (measure := fun s : (Slice Std.I64) × (Option step.StepError) × Std.Usize => 2 - s.2.2.val)
      (inv := fun s : (Slice Std.I64) × (Option step.StepError) × Std.Usize =>
        s.2.2.val ≤ 1 ∧
        (s.2.2.val = 0 → s.1 = dst ∧ s.2.1 = none) ∧
        (1 ≤ s.2.2.val → s.1 = dst ∧ s.2.1 = some e))
      (post := fun y : (Slice Std.I64) × (Option step.StepError) => y = (dst, some e))
    · rintro ⟨acc, err, c'⟩ ⟨hle, h0, h1⟩
      dsimp only at hle h0 h1
      by_cases hz : c'.val = 0
      · obtain ⟨hacc, herr⟩ := h0 hz
        have hc0 : c' = 0#usize := UScalar.eq_of_val_eq hz
        dsimp only
        rw [hacc, herr, hc0, hbody]
        simp only [WP.spec_ok]
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · have h1v : (1#usize : Std.Usize).val = 1 := by decide
          omega
        · intro h
          have h1v : (1#usize : Std.Usize).val = 1 := by decide
          omega
        · intro _; exact ⟨trivial, trivial⟩
        · have h1v : (1#usize : Std.Usize).val = 1 := by decide
          have h0v : (0#usize : Std.Usize).val = 0 := by decide
          omega
      · have hge1 : 1 ≤ c'.val := by omega
        obtain ⟨hacc, herr⟩ := h1 hge1
        dsimp only
        rw [hacc, herr]
        rw [mixer5_loop0_body_done fa key_dim cdim hk eps q_scale kr0 dst (some e) c' (by simp)]
        simp only [WP.spec_ok]
    · dsimp only
      refine ⟨?_, ?_, ?_⟩
      · rw [show (0#usize : Std.Usize).val = 0 from by decide]; omega
      · intro _; exact ⟨rfl, rfl⟩
      · intro h
        rw [show (0#usize : Std.Usize).val = 0 from by decide] at h; omega
  obtain ⟨y, hy, hyeq⟩ := WP.spec_imp_exists htriple
  rw [hy, hyeq]

/-- **`norm_qk_q32` refuses when a column turn fails.**  If the first turn sets
`err = some e` — the exact condition is carried by the turn: a non-positive scale in either
`l2_normalize` (which `Mixer4` proves is exactly `Σ = 0` with `eps ≤ 0`) or a product outside
`i64` (`Overflow`) — the loop returns the destination with `some e` and the driver returns
`Err e`. -/
theorem word_norm_qk_q32_refuses_of_turn (fa : mixer.FlatAffine) (key_dim cdim hk kh : Std.Usize)
    (eps q_scale : Std.I64) (dst : Slice Std.I64) (e : step.StepError) (kr0 : Std.Usize)
    (hkr0 : kh.val * hk.val ≤ Usize.max)
    (hkr0eq : kh * hk = ok kr0)
    (hbody : mixer.norm_qk_q32_loop0.body fa key_dim cdim hk eps q_scale kr0 dst none 0#usize
      = ok (ControlFlow.cont (dst, some e, 1#usize))) :
    ∃ dst2 : Slice Std.I64,
      mixer.norm_qk_q32 fa key_dim cdim hk eps q_scale kh dst
        = ok (core.result.Result.Err e, dst2) := by
  unfold mixer.norm_qk_q32
  rw [hkr0eq]
  simp only [bind_tc_ok]
  rw [mixer5_loop0_first_turn fa key_dim cdim hk kr0 eps q_scale dst e hbody]
  simp only [bind_tc_ok]
  exact ⟨dst, rfl⟩

/-! ## §8 · Non-vacuity

The domain predicates are inhabited jointly, and the refusal's condition is the `Mixer4` witness:
an empty window with `eps ≤ 0`. -/

/-- `[1]` at `eps = 1` is a joint witness of the `l2_normalize` domain and the product domain with
`q_scale = 2³²`: the sum of squares is `1`, `den = 1`, the normalised word is `2³²` and the product
`1` is the scaled word. -/
example : mixer5_domain 1 [1#i64] ∧ mixer5_fit 4294967296 [4294967296#i64] := by
  constructor
  · refine ⟨?_, ?_, ?_⟩
    · rw [mixer4_sumSq_singleton]
      have h1v : (1#i64).val = 1 := by decide
      rw [h1v]; norm_num
    · rw [mixer4_sumSq_singleton]
      have h1v : (1#i64).val = 1 := by decide
      rw [h1v]; norm_num
    · intro v hv
      rw [List.mem_singleton] at hv
      subst hv
      have h1v : (1#i64).val = 1 := by decide
      have hsq : mixer4_sumSq [1#i64] = 1 := by rw [mixer4_sumSq_singleton, h1v]; norm_num
      have hisq : mixer4_isqrtRef 1 = 1 :=
        (mixer4_isqrt_unique (by norm_num) (by norm_num) (by norm_num)).symm
      have hnorm : mixer4_normRef (mixer4_sumSq [1#i64]) = 1 := by
        rw [mixer4_normRef, hsq, mixer4_roundRef, hisq]; norm_num
      have hden : mixer4_denRef (mixer4_sumSq [1#i64]) 1 = 1 := by
        rw [mixer4_denRef, hnorm]; norm_num
      have hdiv : mixer4_divRef (1#i64).val (mixer4_denRef (mixer4_sumSq [1#i64]) 1)
          = 4294967296 := by
        rw [hden, h1v, mixer4_divRef, mixer2_divQ32Ref, if_pos (by norm_num)]; norm_num
      rw [hdiv, I64.min_eq, I64.max_eq]; norm_num
  · intro v hv
    rw [List.mem_singleton] at hv
    subst hv
    have h1v : (4294967296#i64).val = 4294967296 := by decide
    rw [h1v, mixer2_mulQ32Ref, mixer2_roundShift32Ref, I64.min_eq, I64.max_eq]
    decide

/-- The refusal condition is inhabited: the empty window with `eps = 0` gives the `Mixer4`
`BadConfig` message. -/
example : ∃ msg : String,
    mixer.l2_normalize (Slice.new Std.I64) 0#i64 = ok (core.result.Result.Err (step.StepError.BadConfig msg)) :=
  ⟨_, word_l2_normalize_refuses_badconfig 0#i64 (by decide) _ rfl⟩

end apeiron_word
