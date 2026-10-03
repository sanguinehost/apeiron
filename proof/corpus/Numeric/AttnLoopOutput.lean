/-
Content of the value and output loops of the extracted single-token attention kernel
.

`Numeric.EngineAttnRefinement` proves the per-call identities `weight_at_spec` (the weight
division) and `attn_head_loop3_body_epilogue` (the output epilogue), leaving the four
`attn_head_loop0..3` loop specifications open (residual item 1 there and in
`Numeric.AttnSoftmaxPerturb`). This module discharges that residual for the value loop
(`attn_head_loop2`, the key-outer pass), for its nested per-component accumulation
(`attn_head_loop2_loop0`, the `head_dim`-inner pass over one key's row) and for the output loop
(`attn_head_loop3`):

* `attn_head_loop2_loop0_spec`: on the value row window, the per-component accumulation returns,
  clear, each component `i < head_dim` equal to the accumulator's entry at `i` plus `w · v_{s,i}`;
* `attn_head_loop2_spec`: on `AttnWeightWindow` and the value row window, the value loop returns,
  clear, each component `i < head_dim` equal to `Σ_{j≤pos} attnWeightAt exps total j · v_{j,i}`;
* `attn_head_loop3_spec`: on the epilogue narrowing at every component, the output loop returns,
  clear, every component `i < head_dim` equal to `divRoundNearest (acc[i]) 2^32`;
* `value_component_spec`: the per-component composition of the two, restated for the reshaped
  kernel: the accumulator content at component `i` is the exact sum `Σ_{s≤pos} W_s·V_{s,i}` with
  `V` the `valueList` entries, and the output epilogue reduces it by the single half-biased
  division.

## The reshaped extraction this module reads

The staged artifact moved the weight division into the value pass: the separate `weights` vector of
the predecessor is gone, and `attn.value_component`, `attn.value_component_loop`,
`attn.value_term` and `attn.plane_at` are no longer extracted. The weight at key `s` is resolved
by `attn_head_loop2.body` immediately before that key's `head_dim` row is read once and folded
into the per-component `i128` accumulator by the nested `attn_head_loop2_loop0`. The three loop
specifications above are therefore stated against the three loops that exist; the arithmetic each
one claims is unchanged.

## Windows not supplied by `attn_domain_v2`

`attn_domain_v2` (`Numeric.AttnSoftmaxPerturb`) states the value window (`|v_{j,i}| ≤ 2^62`) and
`pos + 1 ≤ 2^30`, and its own header already documents an implicit weight-magnitude assumption
(`"|W_j·v_{j,i}| ≤ 2^95"`) it never states as a conjunct. Concretely:

* `AttnWeightWindow`'s numerator sign (`0 ≤ E_j`) and total positivity (`0 < T`) are guard facts
  the extracted `word_attn_head` establishes upstream (`total ≤ 0` refuses `ZeroSum` before the
  value loop runs; `fixed_exp`'s codomain is nonnegative) — no `attn_domain_v2` conjunct;
* `AttnWeightWindow`'s per-index `weight_at_spec` headroom (`e·2^32 + T/2 ≤ I128.max`) and `i64`
  fit are consequences of `0 ≤ E_j ≤ T` (a weight is a rounded probability), not stated by
  `attn_domain_v2`; its magnitude bound `0 ≤ W_j ≤ 2^32` is the conjunct the folded pass needs for
  the `i128` product, and `attn_weight_bounds` (`Numeric.AttnHeadWindowsValue.lean`) supplies it;
* `AttnValueWindow`'s row range-index identity (the `head_dim`-wide block
  `vplane[(first + s·stride) .. (first + s·stride + head_dim)]` exists, is `head_dim` long, and its
  `i`-th entry is the plane's `(first + s·stride + i)`-th entry) is the "range-index identity
  tying `planeRow` to the plane contents" `Numeric.EngineAttnRefinement`'s header still lists as
  residual; `attn_domain_v2` gives the plane *lengths* but not this checked arithmetic, which is a
  consequence of those lengths and `s ≤ pos`, `i < head_dim` not derived here;
* `AttnValueWindow`'s checked `usize` block arithmetic (`s·stride`, `first + ·` and `+ head_dim`
  all inside `usize`) is the same checked-arithmetic gap, supplied as its first conjunct at its
  largest value, `stride·pos + first + head_dim`; that is the executed worst case under the loop's
  own `s ≤ pos` guard, and it is not a consequence of `attn_domain_v2` because the plane-length
  bound does not reach it (see `Numeric.AttnHeadWindowsValue`'s header for the counterexample to
  the wider `stride·(pos+1)` form this conjunct previously carried);
* the output epilogue's narrowing (`hlo`, `hhi`, `hfit`) is passed through unchanged (reused, not
  restated) from `attn_head_loop3_body_epilogue`; it is the weighted-mean bound
  `|out_i| ≤ max_j |v_{j,i}|` (well inside `i64` given the value window), not derived here.

## Trust boundary

Reaches the extracted `attn.KvState.attn_head_loop2`, `attn.KvState.attn_head_loop2_loop0` and
`attn.KvState.attn_head_loop3` of `proof/ApeironWord.lean` at the digest
`Numeric.EngineAttnRefinement` binds to. Does not cover rustc/LLVM codegen of
`src/engine/attn.rs`, the kernel ABI, GPU/SPIR-V, or the `unsafe` container.
-/
import Numeric.AttnSoftmaxPerturb

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## List-index helpers -/

/-- `getD` at an in-range index is the checked `getElem`. -/
private theorem getD_eq_get' {α : Type} (d : α) {l : List α} {n : ℕ} (h : n < l.length) :
    l.getD n d = l[n]'h := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]

/-- `getD` of `List.set` at the updated index. -/
private theorem getD_set_same {α : Type} (l : List α) (i : ℕ) (x d : α) (hi : i < l.length) :
    (l.set i x).getD i d = x := by
  rw [List.getD_eq_getElem?_getD (l := l.set i x), List.getElem?_set, if_pos rfl, if_pos hi,
    Option.getD_some]

/-- `getD` of `List.set` away from the updated index. -/
private theorem getD_set_ne {α : Type} (l : List α) (j i : ℕ) (x d : α) (hne : i ≠ j) :
    (l.set j x).getD i d = l.getD i d := by
  rw [List.getD_eq_getElem?_getD (l := l.set j x), List.getElem?_set,
    if_neg (fun h => hne h.symm), List.getD_eq_getElem?_getD]

/-- The list behind `Vec.set` is the `List.set` of the original. -/
private theorem vecSet_val {α : Type} (v : alloc.vec.Vec α) (i : Usize) (x : α) :
    (v.set i x).val = v.val.set i.val x := by
  rw [alloc.vec.Vec.val, alloc.vec.Vec.set, Slice.from_val]

/-- `getD` at an in-range index of a `List.slice` is the same entry of the original list. -/
private theorem getD_slice_eq {α : Type} (l : List α) (s e i : ℕ) (hi : i < e - s) (d : α) :
    (l.slice s e).getD i d = l.getD (s + i) d := by
  rw [List.getD_eq_getElem?_getD (l := l.slice s e), List.getD_eq_getElem?_getD (l := l)]
  rw [List.slice, List.getElem?_take, if_pos hi, List.getElem?_drop]

/-- The value of the extracted `saturating_sub` on `usize`: the clamped difference. -/
private theorem usize_saturating_sub_val (x y : Usize) :
    (core.num.Usize.saturating_sub x y).val = max 0 (x.val - y.val) := by
  rw [core.num.Usize.saturating_sub, UScalar.saturating_sub, UScalar.val, BitVec.toNat_ofNat]
  have hle : max 0 (x.val - y.val) ≤ x.val :=
    max_le (Nat.zero_le _) (Nat.sub_le x.val y.val)
  rw [Nat.mod_eq_of_lt (lt_of_le_of_lt hle x.hBounds)]

/-- `(2:ℕ)^30 ≤ Usize.max` on either platform word size. -/
theorem usize_max_ge_2_pow_30 : (2 : ℕ) ^ 30 ≤ Usize.max := by
  have h : (Usize.max : Nat) = 2 ^ System.Platform.numBits - 1 := by
    simp [Usize.max, Usize.numBits]
  rcases System.Platform.numBits_eq with h' | h' <;> simp [h, h']

/-! ## Weight and value stages: the exact integer content -/

/-- The exact weight of causal index `j`: `divRoundNearest (E_j·2^32) T`
(`attn.weight_at`, `src/engine/attn.rs:143-152`). -/
def attnWeightAt (exps : List I64) (total : I128) (j : ℕ) : ℤ :=
  divRoundNearest ((exps.getD j 0#i64).val * (2 : ℤ) ^ (32 : ℕ)) total.val

/-- The weight-stage window: nonnegative numerators, positive total, room for the loop to read the
numerator vector, the per-index `weight_at_spec` premises, and the magnitude of each rounded
weight. See the module header for which conjuncts `attn_domain_v2` does and does not supply. -/
def AttnWeightWindow (exps : alloc.vec.Vec I64) (total : I128) (pos : Usize) : Prop :=
  exps.val.length = pos.val + 1 ∧
  pos.val + 1 ≤ Usize.max ∧
  0 < total.val ∧
  (∀ j, j ≤ pos.val →
    0 ≤ (exps.val.getD j 0#i64).val ∧
    (exps.val.getD j 0#i64).val * (2 : ℤ) ^ (32 : ℕ) + total.val / 2 ≤ I128.max ∧
    -((2 : ℤ) ^ (63 : ℕ)) ≤ attnWeightAt exps.val total j ∧
    attnWeightAt exps.val total j ≤ (2 : ℤ) ^ (63 : ℕ) - 1) ∧
  (∀ j, j ≤ pos.val →
    0 ≤ attnWeightAt exps.val total j ∧ attnWeightAt exps.val total j ≤ (2 : ℤ) ^ (32 : ℕ))

/-- **Per-key weight contract of the value pass.** Whichever reciprocal arm the executed loop
carries — `attn.weight_at` on `none`, `attn.weight_at_recip` on `some r` — the resolved weight at
every causal index returns `Ok w` with `w` the canonical rounded weight `attnWeightAt`. This is the
hypothesis that generalizes the value-pass specifications from the hardcoded `none` path to either
arm; on the `none` arm it is `weight_at_spec`, on the `some r` arm it is the reciprocal value
identity. -/
def AttnWeightResolve (exps : alloc.vec.Vec I64) (total : I128) (pos : Usize)
    (recip : Option U64) : Prop :=
  ∀ j, j ≤ pos.val → ∃ w : I64,
    (match recip with
      | none => attn.weight_at (exps.val.getD j 0#i64) total
      | some r => attn.weight_at_recip (exps.val.getD j 0#i64) total r) =
      ok (core.result.Result.Ok w) ∧
    w.val = attnWeightAt exps.val total j

/-- **The `recip = none` arm of `AttnWeightResolve`, discharged.** On the fallback arm the
executed loop resolves the weight at causal index `j` with `attn.weight_at`, and
`weight_at_spec` (`Numeric.EngineAttnRefinement`) identifies that call's result as `Ok w` with
`w.val = divRoundNearest (e·2^32) T` for `e = (exps.val.getD j 0#i64).val` — which is
`attnWeightAt exps.val total j` by definition. The three premises that identity needs
(`0 ≤ e`, `0 < T` and the per-index headroom and `i64` fit) are the `AttnWeightWindow` guard facts
the module header names: they are established upstream by the extracted `word_attn_head`, not by
`attn_domain_v2`.

The `some r` arm — `attn.weight_at_recip`, the `u128` reciprocal-multiply path — has no
specification anywhere in the corpus (`Numeric.EngineAttnRefinement`'s residual item 4), and
`recip_u64` returns `Some` for every `total` in `2 ..= u64::MAX/2`, so that arm is on the executed
path. This theorem therefore proves the fallback arm only; `AttnWeightResolve` at `some r` and
every consumer stating it at an unquantified `recip` remain open. -/
theorem attnWeightResolve_none (exps : alloc.vec.Vec I64) (total : I128) (pos : Usize)
    (hwin : AttnWeightWindow exps total pos) :
    AttnWeightResolve exps total pos none := by
  intro j hj
  obtain ⟨w, hw, hwv⟩ := weight_at_spec (exps.val.getD j 0#i64) total
    (hwin.2.2.2.1 j hj).1 hwin.2.2.1 (hwin.2.2.2.1 j hj).2.1
    ⟨(hwin.2.2.2.1 j hj).2.2.1, (hwin.2.2.2.1 j hj).2.2.2⟩
  exact ⟨w, hw, hwv⟩

/-- The exact value entry at component `i` of causal key `s`, `valueList`'s range-index form. -/
def attnValueAt (vplane : List I64) (stride first s i : ℕ) : ℤ :=
  (vplane.getD (s * stride + first + i) 0#i64).val

/-- The value-loop window at one causal key `s ≤ pos`: the `head_dim`-wide block
`vplane[(first + s·stride) .. (first + s·stride + head_dim)]` exists, is `head_dim` long and its
`i`-th entry is the plane's `(first + s·stride + i)`-th entry (the range-index identity), and the
value magnitude is the `attn_domain_v2` window `|v_{s,i}| ≤ 2^62`; `pos+1 ≤ 2^30` (also
`attn_domain_v2`) keeps the checked `i128` products and the index increments inside range. The
checked `usize` block arithmetic the row read performs is the first conjunct, stated at the largest
key the loop reads: `src/engine/attn.rs:1183-1187` evaluates `first + s·stride` and `+ head_dim`
under `s ≤ pos` (`src/engine/attn.rs:1176`), so its worst case is `stride·pos + first + head_dim`
and no key one step past `pos` is ever addressed. -/
def AttnValueWindow (hd stride first pos : Usize) (vplane : alloc.vec.Vec I64) : Prop :=
  stride.val * pos.val + first.val + hd.val ≤ Usize.max ∧
  pos.val + 1 ≤ (2 : ℕ) ^ 30 ∧
  (∀ s : Usize, s.val ≤ pos.val →
    ∃ (sat base end_ : Usize) (row : Slice I64),
      sat.val = max 0 (vplane.val.length - hd.val) ∧
      base.val = first.val + s.val * stride.val ∧
      end_.val = base.val + hd.val ∧
      end_.val ≤ vplane.val.length ∧
      ¬ (base > sat) ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexRangeUsizeSlice Std.I64) vplane
          { start := base, «end» := end_ } = ok row ∧
      row.val.length = hd.val ∧
      (∀ i, i < hd.val →
        row.val.getD i 0#i64 = vplane.val.getD (s.val * stride.val + first.val + i) 0#i64)) ∧
  (∀ s, s ≤ pos.val → ∀ i, i < hd.val →
    |attnValueAt vplane.val stride.val first.val s i| ≤ (2 : ℤ) ^ (62 : ℕ))


/-! ## The range-index identities -/

/-- `planeRow` at an in-range column is the plane's flat get: the checked-arithmetic index of the
extracted row read (`(s·kv + kvh)·hd + i`) matches `planeRow`'s definition literally. -/
theorem planeRow_getD (plane : List I64) (kv hd kvh s i : ℕ) (hi : i < hd) :
    (planeRow plane kv hd kvh s).getD i 0#i64 = plane.getD ((s * kv + kvh) * hd + i) 0#i64 := by
  have h1 : (planeRow plane kv hd kvh s)[i]? = some (plane.getD ((s * kv + kvh) * hd + i) 0#i64) := by
    simp only [planeRow, List.getElem?_map, List.getElem?_range hi, Option.map_some]
  rw [List.getD_eq_getElem?_getD, h1, Option.getD_some]

/-- **Value-list range-index identity.** `valueList`'s `s`-th entry is the plane's
`((s·kv + kvh)·hd + i)`-th entry whenever `i < head_dim` and `s ≤ pos` — the range-index identity
the module header names. -/
theorem valueList_getD (plane : List I64) (kv hd kvh pos i s : ℕ) (hi : i < hd) (hs : s ≤ pos) :
    (valueList plane kv hd kvh pos i).getD s 0 =
      (plane.getD ((s * kv + kvh) * hd + i) 0#i64).val := by
  have hlt : s < pos + 1 := by omega
  have h1 : (valueList plane kv hd kvh pos i)[s]? =
      some ((plane.getD ((s * kv + kvh) * hd + i) 0#i64).val) := by
    simp only [valueList, List.getElem?_map, List.getElem?_range hlt, Option.map_some]
    rw [planeRow_getD plane kv hd kvh s i hi]
  rw [List.getD_eq_getElem?_getD, h1, Option.getD_some]

/-- The exact value entry of the value-loop invariant, in the `stride`/`first` addressing the
extracted row read uses: `attnValueAt plane stride first s i` is `valueList`'s entry at
`(kv, kvh)` with `stride = kv · head_dim` and `first = kvh · head_dim`. -/
theorem attnValueAt_eq (plane : List I64) (kv hd kvh pos i s : ℕ) (hi : i < hd) (hs : s ≤ pos) :
    attnValueAt plane (kv * hd) (kvh * hd) s i = (valueList plane kv hd kvh pos i).getD s 0 := by
  rw [attnValueAt]
  have hidx : s * (kv * hd) + kvh * hd + i = (s * kv + kvh) * hd + i := by ring
  rw [hidx]
  rw [List.getD_eq_getElem?_getD (l := valueList plane kv hd kvh pos i)]
  have hlt : s < pos + 1 := by omega
  have h1 : (valueList plane kv hd kvh pos i)[s]? =
      some ((plane.getD ((s * kv + kvh) * hd + i) 0#i64).val) := by
    simp only [valueList, List.getElem?_map, List.getElem?_range hlt, Option.map_some]
    rw [planeRow_getD plane kv hd kvh s i hi]
  rw [h1, Option.getD_some, List.getD_eq_getElem?_getD (l := plane)]

/-- The value-accumulator content after `s` causal keys: the failure flag is clear, `s` is within
one of `pos`, `acc` is `head_dim` long, and every component `i < head_dim` holds its entry in the
initial accumulator raised by the sum over the first `s` keys of `attnWeightAt · attnValueAt`. The
`acc0` term is what makes the invariant hold for a caller's arbitrary starting accumulator; on the
kernel's own accumulator, which is `vec![0i128; head_dim]`, `hacc0zero` in the two specifications
below reduces it to the bare sum. -/
def AttnAccInv (hd pos : Usize) (exps : alloc.vec.Vec I64) (total : I128)
    (vplane : alloc.vec.Vec I64) (stride first : Usize) (acc0 acc : alloc.vec.Vec I128)
    (s : ℕ) (ok1 : Bool) (err : Option attn.AttnError) : Prop :=
  ok1 = true ∧ err = none ∧ s ≤ pos.val + 1 ∧ acc.val.length = hd.val ∧
  (∀ i, i < hd.val →
    (acc.val.getD i 0#i128).val = (acc0.val.getD i 0#i128).val +
      ∑ j ∈ Finset.range s, attnWeightAt exps.val total j *
        attnValueAt vplane.val stride.val first.val j i)

/-! ## The nested per-component accumulation -/

/-- Row-accumulation invariant after `i` components: the failure flag is clear, `i` is within the
row, `acc` differs from `acc0` exactly at the components below `i`, each raised by the one term
`w·v_j`, and its length is unchanged. -/
def AttnRowAccInv (row : Slice I64) (w : I128) (acc0 acc : alloc.vec.Vec I128) (i : ℕ)
    (ok1 : Bool) (err : Option attn.AttnError) : Prop :=
  ok1 = true ∧ err = none ∧ i ≤ row.val.length ∧
  acc.val.length = acc0.val.length ∧
  (∀ j, j < i →
    (acc.val.getD j 0#i128).val =
      (acc0.val.getD j 0#i128).val + w.val * (row.val.getD j 0#i64).val) ∧
  (∀ j, i ≤ j → (acc.val.getD j 0#i128).val = (acc0.val.getD j 0#i128).val)

/-- **One component step.** With `i < head_dim` and the row invariant holding, the body reads
`row[i]`, multiplies the weight by it, adds the product to `acc[i]` (both checked at `i128`) and
continues at `i + 1` with the invariant restored. The two checked operations are admitted from the
magnitude bounds: `|w| ≤ 2^32` and `|row[j]| ≤ 2^62` give `|w·row[i]| ≤ 2^94`, and
`|acc0[i]| ≤ 2^124` gives `|acc0[i] + w·row[i]| ≤ 2^124 + 2^94 < 2^127`. -/
theorem attn_head_loop2_loop0_body_step (hd : Usize) (row : Slice I64) (w : I128)
    (acc0 acc : alloc.vec.Vec I128) (i : Usize)
    (hrow : row.val.length = hd.val)
    (hacl0 : acc0.val.length = hd.val)
    (hwb : 0 ≤ w.val ∧ w.val ≤ (2 : ℤ) ^ (32 : ℕ))
    (hvb : ∀ j, j < hd.val → |(row.val.getD j 0#i64).val| ≤ (2 : ℤ) ^ (62 : ℕ))
    (hacc0b : ∀ j, j < hd.val → |(acc0.val.getD j 0#i128).val| ≤ (2 : ℤ) ^ (124 : ℕ))
    (hinv : AttnRowAccInv row w acc0 acc i.val true none)
    (hi : i.val < hd.val) :
    WP.spec (attn.KvState.attn_head_loop2_loop0.body hd row w acc true none i)
      (fun cf =>
        ∃ (acc2 : alloc.vec.Vec I128) (i2 : Usize),
          cf = ControlFlow.cont (acc2, true, none, i2) ∧
          AttnRowAccInv row w acc0 acc2 i2 true none ∧
          i2.val = i.val + 1) := by
  obtain ⟨_hok, _herr, hile, hlen, hlow, hhigh⟩ := hinv
  have hacc0b_i : |(acc0.val.getD i.val 0#i128).val| ≤ (2 : ℤ) ^ (124 : ℕ) := hacc0b i.val hi
  have hvb_i : |(row.val.getD i.val 0#i64).val| ≤ (2 : ℤ) ^ (62 : ℕ) := hvb i.val hi
  have hrowlt : i.val < row.val.length := by rw [hrow]; exact hi
  obtain ⟨e, he_eq, hev⟩ := WP.spec_imp_exists (Slice.index_usize_spec row i hrowlt)
  have he_eq' : row.val.getD i.val 0#i64 = e := by rw [getD_eq_get' _ hrowlt, hev]
  have hacl : acc.val.length = hd.val := by rw [hlen, hacl0]
  have hilt : i.val < acc.val.length := by rw [hacl]; exact hi
  obtain ⟨a, hacc_eq, hav⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec acc i hilt)
  have ha_eq : acc.val.getD i.val 0#i128 = a := by rw [getD_eq_get' _ hilt, hav]
  have ha0 : a.val = (acc0.val.getD i.val 0#i128).val := by
    calc a.val = (acc.val.getD i.val 0#i128).val := by rw [hav, getD_eq_get' _ hilt]
      _ = (acc0.val.getD i.val 0#i128).val := hhigh i.val (le_refl _)
  have he2 : (core.convert.num.FromI128I64.from e).val = e.val :=
    core.convert.num.FromI128I64.from_val_eq e
  have hI128min : I128.min = -(2 : ℤ) ^ 127 := by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ 127 - 1 := by norm_num [I128.max, IScalar.max, I128.numBits]
  have hprodfit : |w.val * e.val| ≤ (2 : ℤ) ^ 94 := by
    calc |w.val * e.val| = |w.val| * |e.val| := abs_mul _ _
      _ ≤ (2 : ℤ) ^ 32 * (2 : ℤ) ^ 62 := by
          rw [he_eq'] at hvb_i
          have hwabs : |w.val| ≤ (2 : ℤ) ^ (32 : ℕ) := by
            rw [abs_of_nonneg hwb.1]; exact hwb.2
          exact mul_le_mul hwabs hvb_i (abs_nonneg _) (by norm_num)
      _ = (2 : ℤ) ^ 94 := by norm_num
  have hp94lo : -(2 : ℤ) ^ 127 ≤ -(2 : ℤ) ^ 94 := by norm_num
  have hp94hi : (2 : ℤ) ^ 94 ≤ (2 : ℤ) ^ 127 - 1 := by norm_num
  have hprodlo : -(2 : ℤ) ^ 94 ≤ w.val * e.val := by simpa using (abs_le.mp hprodfit).1
  have hprodhi : w.val * e.val ≤ (2 : ℤ) ^ 94 := by simpa using (abs_le.mp hprodfit).2
  obtain ⟨term, hterm_eq, htermv⟩ := I128_checked_mul_some w
    (core.convert.num.FromI128I64.from e)
    (by rw [hI128min, he2]; exact le_trans hp94lo hprodlo)
    (by rw [hI128max, he2]; exact le_trans hprodhi hp94hi)
  have htermv' : term.val = w.val * e.val := by rw [htermv, he2]
  have haddfit : |a.val + w.val * e.val| ≤ (2 : ℤ) ^ 124 + (2 : ℤ) ^ 94 := by
    have h3 : |a.val| ≤ (2 : ℤ) ^ 124 := by rw [ha0]; exact hacc0b_i
    calc |a.val + w.val * e.val| ≤ |a.val| + |w.val * e.val| := abs_add_le _ _
      _ ≤ (2 : ℤ) ^ 124 + (2 : ℤ) ^ 94 := add_le_add h3 hprodfit
  obtain ⟨haddlo, haddhi⟩ := abs_le.mp haddfit
  have hpSumLo : -(2 : ℤ) ^ 127 ≤ -((2 : ℤ) ^ 124 + (2 : ℤ) ^ 94) := by norm_num
  have hpSumHi : (2 : ℤ) ^ 124 + (2 : ℤ) ^ 94 ≤ (2 : ℤ) ^ 127 - 1 := by norm_num
  have hfitlo : I128.min ≤ a.val + term.val := by
    rw [hI128min, htermv']
    exact le_trans hpSumLo haddlo
  have hfithi : a.val + term.val ≤ I128.max := by
    rw [hI128max, htermv']
    exact le_trans haddhi hpSumHi
  obtain ⟨next, hnext_eq, hnextv⟩ := I128_checked_add_some a term hfitlo hfithi
  have hidxmut : alloc.vec.Vec.index_mut
      (core.slice.index.SliceIndexUsizeSlice Std.I128) acc i = ok (a, acc.set i) := by
    rw [alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize]
    simp only [hacc_eq, bind_tc_ok]
  have hsmax : i.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]
    have hbw := hd.hBounds
    rw [UScalarTy.Usize_numBits_eq] at hbw
    have hm : (Usize.max : ℕ) = 2 ^ System.Platform.numBits - 1 := by
      simp [Usize.max, Usize.numBits]
    have h3 : hd.val ≤ 2 ^ System.Platform.numBits - 1 := Nat.le_sub_one_of_lt hbw
    have h4 : i.val + 1 ≤ hd.val := by omega
    rw [hm]
    exact le_trans h4 h3
  obtain ⟨i2, hadd1, hi2⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hsmax)
  have hi2' : i2.val = i.val + 1 := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hi2, h1]
  have hi2le : i2.val ≤ row.val.length := by rw [hi2', hrow]; omega
  have hlen2 : (acc.set i next).val.length = acc0.val.length := by
    rw [vecSet_val, List.length_set]; exact hlen
  have hlow2 : ∀ j, j < i2.val →
      ((acc.set i next).val.getD j 0#i128).val =
        (acc0.val.getD j 0#i128).val + w.val * (row.val.getD j 0#i64).val := by
    intro j hj
    rw [vecSet_val]
    by_cases hji : j = i.val
    · subst hji
      have hset : (acc.val.set i.val next).getD i.val 0#i128 = next :=
        getD_set_same acc.val i.val next (0#i128) hilt
      rw [hset, hnextv, htermv', ha0, he_eq']
    · rw [getD_set_ne acc.val i.val j next (0#i128) hji]
      exact hlow j (by omega)
  have hhigh2 : ∀ j, i2.val ≤ j → ((acc.set i next).val.getD j 0#i128).val
      = (acc0.val.getD j 0#i128).val := by
    intro j hj
    rw [vecSet_val, getD_set_ne acc.val i.val j next (0#i128) (by omega)]
    exact hhigh j (by omega)
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (acc.set i next, true, none, i2), ?_,
    acc.set i next, i2, rfl, ⟨rfl, rfl, hi2le, hlen2, hlow2, hhigh2⟩, hi2'⟩
  rw [attn.KvState.attn_head_loop2_loop0.body, if_pos ((UScalar.lt_equiv i hd).2 hi),
    if_pos rfl]
  try simp only []
  rw [he_eq]
  simp only [lift, bind_tc_ok]
  rw [hterm_eq]
  try simp only []
  rw [alloc.vec.Vec.index_slice_index, hacc_eq]
  simp only [bind_tc_ok]
  rw [hnext_eq]
  try simp only []
  rw [hidxmut]
  simp only [bind_tc_ok]
  rw [hadd1]
  simp only [bind_tc_ok]
  simp

/-- **Per-component accumulation.** On the value row window and the accumulator magnitude bound,
`attn_head_loop2_loop0` returns, clear, with every component `i < head_dim` raised by exactly the
one term `w · v_i` of that row. -/
theorem attn_head_loop2_loop0_spec (hd : Usize) (row : Slice I64) (w : I128)
    (acc0 : alloc.vec.Vec I128)
    (hrow : row.val.length = hd.val)
    (hacl0 : acc0.val.length = hd.val)
    (hwb : 0 ≤ w.val ∧ w.val ≤ (2 : ℤ) ^ (32 : ℕ))
    (hvb : ∀ j, j < hd.val → |(row.val.getD j 0#i64).val| ≤ (2 : ℤ) ^ (62 : ℕ))
    (hacc0b : ∀ j, j < hd.val → |(acc0.val.getD j 0#i128).val| ≤ (2 : ℤ) ^ (124 : ℕ)) :
    ∃ acc : alloc.vec.Vec I128,
      attn.KvState.attn_head_loop2_loop0 hd acc0 true none row w 0#usize
        = ok (acc, true, none) ∧
      AttnRowAccInv row w acc0 acc hd.val true none := by
  have h0 : (0#usize).val = 0 := by norm_num
  have hinv0 : AttnRowAccInv row w acc0 acc0 0 true none := by
    refine ⟨rfl, rfl, ?_, rfl, ?_, ?_⟩
    · omega
    · intro j hj; omega
    · intro j hj; omega
  have hspec : WP.spec
      (loop (fun (st : alloc.vec.Vec Std.I128 × Bool
          × (Option attn.AttnError) × Std.Usize) =>
          attn.KvState.attn_head_loop2_loop0.body hd row w st.1 st.2.1 st.2.2.1 st.2.2.2)
        (acc0, true, (none : Option attn.AttnError), (0#usize : Std.Usize)))
      (fun b : alloc.vec.Vec Std.I128 × Bool × (Option attn.AttnError) =>
        ∃ acc : alloc.vec.Vec I128, b = (acc, true, none) ∧
          AttnRowAccInv row w acc0 acc hd.val true none) := by
    apply loop.spec_decr_nat
      (measure := fun st : alloc.vec.Vec Std.I128 × Bool
          × (Option attn.AttnError) × Std.Usize => hd.val - st.2.2.2.val)
      (inv := fun st => AttnRowAccInv row w acc0 st.1 st.2.2.2.val st.2.1 st.2.2.1)
      (post := fun b : alloc.vec.Vec Std.I128 × Bool × (Option attn.AttnError) =>
        ∃ acc : alloc.vec.Vec I128, b = (acc, true, none) ∧
          AttnRowAccInv row w acc0 acc hd.val true none)
    · intro st hst
      rcases st with ⟨acc, ok1, err, i1⟩
      obtain ⟨hok, herr, hile, hlen, hlow, hhigh⟩ := hst
      dsimp only at hok herr hile hlen hlow hhigh
      subst hok herr
      by_cases hlt : i1.val < hd.val
      · obtain ⟨cf, hcf, hpost⟩ := WP.spec_imp_exists
          (attn_head_loop2_loop0_body_step hd row w acc0 acc i1 hrow hacl0 hwb hvb hacc0b
            ⟨rfl, rfl, hile, hlen, hlow, hhigh⟩ hlt)
        obtain ⟨acc2, i2, hcfg, hinv2, hi2⟩ := hpost
        rw [hcfg] at hcf
        simp only [hcf]
        refine ⟨hinv2, ?_⟩
        show hd.val - i2.val < hd.val - i1.val
        omega
      · have hnot : ¬ (i1 < hd) := by
          intro hc
          exact hlt ((UScalar.lt_equiv i1 hd).1 hc)
        have hbody : attn.KvState.attn_head_loop2_loop0.body hd row w acc true none i1
            = ok (ControlFlow.done (acc, true, none)) := by
          rw [attn.KvState.attn_head_loop2_loop0.body, if_neg hnot]
        simp only [hbody]
        have hi1eq : i1.val = hd.val := by omega
        refine ⟨acc, rfl, ?_⟩
        rw [hi1eq] at hlow hhigh
        exact ⟨rfl, rfl, by rw [hrow], hlen, hlow, hhigh⟩
    · exact hinv0
  obtain ⟨b, hb, acc, hbeq, hinv⟩ := WP.spec_imp_exists hspec
  rw [hbeq] at hb
  refine ⟨acc, ?_, hinv⟩
  unfold attn.KvState.attn_head_loop2_loop0
  exact hb

/-! ## Value stage: the key-outer loop -/

/-- **One key step.** With `s ≤ pos` and the accumulator invariant holding, the body resolves the
weight at key `s` (`weight_at_spec`, reused not restated), reads the `head_dim`-wide row at
`first + s·stride` (the window's range-index identity), folds it into `acc` through
`attn_head_loop2_loop0` and continues at `s + 1` with the invariant restored. -/
theorem attn_head_loop2_body_step (hd pos : Usize) (vplane : alloc.vec.Vec I64)
    (stride first : Usize) (exps : alloc.vec.Vec I64) (total : I128) (recip : Option U64)
    (acc0 acc : alloc.vec.Vec I128) (s : Usize)
    (hwin : AttnWeightWindow exps total pos)
    (hvwin : AttnValueWindow hd stride first pos vplane)
    (hwres : AttnWeightResolve exps total pos recip)
    (hacc0zero : ∀ i, i < hd.val → (acc0.val.getD i 0#i128).val = 0)
    (hinv : AttnAccInv hd pos exps total vplane stride first acc0 acc s.val true none)
    (hs : s.val ≤ pos.val) :
    WP.spec
      (attn.KvState.attn_head_loop2.body hd pos vplane stride first exps total recip acc s
        true none)
      (fun cf =>
        ∃ (acc2 : alloc.vec.Vec I128) (s2 : Usize),
          cf = ControlFlow.cont (acc2, s2, true, none) ∧
          AttnAccInv hd pos exps total vplane stride first acc0 acc2 s2 true none ∧
          s2.val = s.val + 1) := by
  obtain ⟨hlen, hcap, htot, -, hwm⟩ := hwin
  obtain ⟨harith, hnpos, hroww, hvb⟩ := hvwin
  obtain ⟨_, _, hle, haclen, hcontent⟩ := hinv
  obtain ⟨sat, base, end_, row, hsat, hbasev, hendv, hendin, hbound, hrowidx, hrowlen,
    hrowget⟩ := hroww s hs
  have hspos : s ≤ pos := (UScalar.le_equiv s pos).2 hs
  have hslt : s.val < exps.val.length := by rw [hlen]; omega
  obtain ⟨e, he_eq, hev⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec exps s hslt)
  have he_eq' : exps.val.getD s.val 0#i64 = e := by
    rw [getD_eq_get' _ hslt, hev]
  have hWs : 0 ≤ attnWeightAt exps.val total s.val ∧
      attnWeightAt exps.val total s.val ≤ (2 : ℤ) ^ (32 : ℕ) := hwm s.val hs
  obtain ⟨hW0, hW32⟩ := hWs
  unfold AttnWeightResolve at hwres
  obtain ⟨w, hw_eq, hwv⟩ := hwres s.val hs
  rw [he_eq'] at hw_eq
  -- the window's first conjunct is stated at the largest key the loop reads, `s = pos`, and
  -- `s ≤ pos` moves it down to this key
  have harith1 : stride.val * pos.val + first.val ≤ Usize.max := by omega
  have harith0 : stride.val * pos.val ≤ Usize.max := by omega
  have hmulle : s.val * stride.val ≤ stride.val * pos.val := by
    calc s.val * stride.val = stride.val * s.val := by ring
      _ ≤ stride.val * pos.val := mul_le_mul_of_nonneg_left hs (Nat.zero_le _)
  have hsmul : s.val * stride.val ≤ Usize.max := le_trans hmulle harith0
  obtain ⟨si, hmul_eq, hmulv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := s) (y := stride) hsmul)
  have hsadd : first.val + si.val ≤ Usize.max := by
    rw [hmulv]
    have h1 : first.val + s.val * stride.val ≤ stride.val * pos.val + first.val := by omega
    exact le_trans h1 harith1
  obtain ⟨bs, hadd_eq, haddv⟩ := WP.spec_imp_exists (Usize.add_spec (x := first) (y := si) hsadd)
  have hii2 : (core.num.Usize.saturating_sub (alloc.vec.Vec.len vplane) hd).val = sat.val := by
    rw [usize_saturating_sub_val, alloc.vec.Vec.len_val]
    exact hsat.symm
  have hii : core.num.Usize.saturating_sub (alloc.vec.Vec.len vplane) hd = sat :=
    UScalar.val_eq_imp _ _ hii2
  have hbs : bs = base := UScalar.val_eq_imp _ _ (by rw [haddv, hmulv, hbasev])
  have hnotgt : ¬ (bs > sat) := by rw [hbs]; exact hbound
  have hbsval : bs.val = base.val := by rw [haddv, hmulv, hbasev]
  have hbsv : bs.val + hd.val = end_.val := by rw [hbsval, hendv]
  have hendle : first.val + s.val * stride.val + hd.val ≤ Usize.max := by
    have h1 : first.val + s.val * stride.val ≤ stride.val * pos.val + first.val := by
      omega
    omega
  have hendle' : bs.val + hd.val ≤ Usize.max := by rw [hbsval, hbasev]; exact hendle
  obtain ⟨en, hend_eq, henval⟩ := WP.spec_imp_exists (Usize.add_spec (x := bs) (y := hd) hendle')
  have henle : en.val ≤ vplane.val.length := by rw [henval, hbsv]; exact hendin
  have hbsle : bs ≤ en := (UScalar.le_equiv bs en).2 (by omega)
  obtain ⟨row', hrow_eq, hrowval, hrowlen'⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_Range_spec vplane { start := bs, «end» := en } hbsle henle)
  have hen : en = end_ := UScalar.val_eq_imp _ _ (by rw [henval, haddv, hmulv, hendv, ← hbasev])
  have hrowEq : row' = row := by
    rw [hbs, hen] at hrow_eq
    exact ok.injEq.mp (hrow_eq.symm.trans hrowidx)
  have hw1v : (core.convert.num.FromI128I64.from w).val = w.val :=
    core.convert.num.FromI128I64.from_val_eq w
  have hrowgetW : ∀ i, i < hd.val →
      row.val.getD i 0#i64 = vplane.val.getD (s.val * stride.val + first.val + i) 0#i64 :=
    hrowget
  rw [← hrowEq] at hrowgetW hrowlen
  have hrowget' : ∀ i, i < hd.val →
      row'.val.getD i 0#i64 = vplane.val.getD (s.val * stride.val + first.val + i) 0#i64 :=
    hrowgetW
  have hatt : ∀ i, i < hd.val →
      attnValueAt vplane.val stride.val first.val s.val i = (row'.val.getD i 0#i64).val := by
    intro i hi
    rw [attnValueAt, hrowget' i hi]
  have hvb' : ∀ i, i < hd.val → |(row'.val.getD i 0#i64).val| ≤ (2 : ℤ) ^ (62 : ℕ) := by
    intro i hi
    rw [← hatt i hi]
    exact hvb s.val hs i hi
  have hcontent' : ∀ i, i < hd.val →
      |(acc.val.getD i 0#i128).val| ≤ (2 : ℤ) ^ (124 : ℕ) := by
    intro i hi
    rw [hcontent i hi, hacc0zero i hi, zero_add]
    have hsumabs : |(∑ j ∈ Finset.range s.val, attnWeightAt exps.val total j *
          attnValueAt vplane.val stride.val first.val j i : ℤ)| ≤
        ∑ j ∈ Finset.range s.val, |(attnWeightAt exps.val total j *
          attnValueAt vplane.val stride.val first.val j i : ℤ)| :=
      Finset.abs_sum_le_sum_abs _ _
    have hterm : ∀ j ∈ Finset.range s.val, |(attnWeightAt exps.val total j *
          attnValueAt vplane.val stride.val first.val j i : ℤ)| ≤
        (2 : ℤ) ^ (32 : ℕ) * (2 : ℤ) ^ (62 : ℕ) := by
      intro j hj
      simp only [Finset.mem_range] at hj
      have hjle : j ≤ pos.val := by omega
      rw [abs_mul, abs_of_nonneg (hwm j hjle).1]
      exact mul_le_mul (hwm j hjle).2 (hvb j hjle i hi) (by norm_num) (by norm_num)

    have hp : s.val ≤ 2 ^ 30 := by omega
    have hp' : (s.val : ℤ) ≤ (2 : ℤ) ^ 30 := by exact_mod_cast hp
    have hcard : ∑ _j ∈ Finset.range s.val,
          ((2 : ℤ) ^ (32 : ℕ) * (2 : ℤ) ^ (62 : ℕ)) = s.val * (2 : ℤ) ^ (94 : ℕ) := by
      rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      calc (s.val : ℤ) * ((2 : ℤ) ^ (32 : ℕ) * (2 : ℤ) ^ (62 : ℕ))
          = (s.val : ℤ) * (2 : ℤ) ^ (94 : ℕ) := by norm_num
        _ = s.val * (2 : ℤ) ^ (94 : ℕ) := rfl
    have hlt94 : s.val * (2 : ℤ) ^ (94 : ℕ) ≤ (2 : ℤ) ^ (124 : ℕ) := by
      calc (s.val : ℤ) * (2 : ℤ) ^ (94 : ℕ) ≤ (2 : ℤ) ^ 30 * (2 : ℤ) ^ (94 : ℕ) :=
            mul_le_mul hp' (le_refl _) (by norm_num) (by positivity)
        _ = (2 : ℤ) ^ 124 := by norm_num
    exact le_trans (le_trans hsumabs ((Finset.sum_le_sum hterm).trans_eq hcard)) hlt94
  obtain ⟨acc1, hi1, hinner, hinnerinv⟩ :=
    attn_head_loop2_loop0_spec hd row' (core.convert.num.FromI128I64.from w)
      acc hrowlen haclen (by
        rw [hw1v, hwv]
        exact ⟨hW0, hW32⟩) hvb' hcontent'
  have hsmax : s.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]
    have h2 : s.val + 1 ≤ pos.val + 1 := by omega
    exact le_trans h2 hcap
  obtain ⟨s2, hadd2, hs2⟩ := WP.spec_imp_exists (Usize.add_spec (x := s) (y := 1#usize) hsmax)
  have hs2' : s2.val = s.val + 1 := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hs2, h1]
  have hs2le : s2.val ≤ pos.val + 1 := by rw [hs2']; omega
  have haclen2 : acc1.val.length = hd.val := by
    have hz := hinnerinv.2.2.1
    rw [haclen] at hz
    exact hz
  have hcontent2 : ∀ i, i < hd.val →
      (acc1.val.getD i 0#i128).val = (acc0.val.getD i 0#i128).val +
        ∑ j ∈ Finset.range s2.val, attnWeightAt exps.val total j *
          attnValueAt vplane.val stride.val first.val j i := by
    intro i hi
    have hrowv : (row'.val.getD i 0#i64).val =
        attnValueAt vplane.val stride.val first.val s.val i :=
      (hatt i hi).symm
    have hcore := hinnerinv.2.2.2.1 i hi
    rw [hrowv, hw1v, hwv] at hcore
    calc (acc1.val.getD i 0#i128).val
        = (acc.val.getD i 0#i128).val + attnWeightAt exps.val total s.val *
            attnValueAt vplane.val stride.val first.val s.val i := hcore
      _ = (acc0.val.getD i 0#i128).val +
            (∑ j ∈ Finset.range s.val, attnWeightAt exps.val total j *
              attnValueAt vplane.val stride.val first.val j i) +
            attnWeightAt exps.val total s.val *
              attnValueAt vplane.val stride.val first.val s.val i := by
              rw [hcontent i hi]
      _ = (acc0.val.getD i 0#i128).val + (∑ j ∈ Finset.range (s.val + 1),
            attnWeightAt exps.val total j *
              attnValueAt vplane.val stride.val first.val j i) := by
              rw [Finset.sum_range_succ]
              ring
      _ = (acc0.val.getD i 0#i128).val + (∑ j ∈ Finset.range s2.val,
            attnWeightAt exps.val total j *
              attnValueAt vplane.val stride.val first.val j i) := by rw [hs2']
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (acc1, s2, true, none), ?_, acc1, s2, rfl,
    ⟨rfl, rfl, hs2le, haclen2, hcontent2⟩, hs2'⟩
  cases recip with
  | none =>
    rw [attn.KvState.attn_head_loop2.body.eq_def, if_pos hspos, if_pos rfl]
    simp only [alloc.vec.Vec.index_slice_index, he_eq, bind_tc_ok]
    dsimp only at hw_eq
    rw [hw_eq]
    try simp only [bind_tc_ok]
    rw [hmul_eq]
    simp only [lift, bind_tc_ok]
    rw [hadd_eq]
    simp only [bind_tc_ok]
    rw [hii]
    try simp only [lift, bind_tc_ok]
    try simp only [bind_tc_ok]
    rw [if_neg hnotgt]
    try simp only [bind_tc_ok]
    rw [hend_eq]
    simp only [bind_tc_ok]
    rw [hrow_eq]
    try simp only [bind_tc_ok]
    rw [hi1]
    try simp only [bind_tc_ok]
    rw [hadd2]
    try simp only []
    try simp only [bind_tc_ok]
    rfl
  | some r =>
    rw [attn.KvState.attn_head_loop2.body.eq_def, if_pos hspos, if_pos rfl]
    simp only [alloc.vec.Vec.index_slice_index, he_eq, bind_tc_ok]
    dsimp only at hw_eq
    rw [hw_eq]
    try simp only [bind_tc_ok]
    rw [hmul_eq]
    simp only [lift, bind_tc_ok]
    rw [hadd_eq]
    simp only [bind_tc_ok]
    rw [hii]
    try simp only [lift, bind_tc_ok]
    try simp only [bind_tc_ok]
    rw [if_neg hnotgt]
    try simp only [bind_tc_ok]
    rw [hend_eq]
    simp only [bind_tc_ok]
    rw [hrow_eq]
    try simp only [bind_tc_ok]
    rw [hi1]
    try simp only [bind_tc_ok]
    rw [hadd2]
    try simp only []
    try simp only [bind_tc_ok]
    rfl

/-- **Value-loop content.** On `AttnWeightWindow` and the value row window, `attn_head_loop2`
returns, clear, each component `i < head_dim` equal to `Σ_{j≤pos} attnWeightAt exps total j ·
attnValueAt vplane stride first j i`. -/
theorem attn_head_loop2_spec (hd pos : Usize) (vplane : alloc.vec.Vec I64) (stride first : Usize)
    (exps : alloc.vec.Vec I64) (total : I128) (recip : Option U64) (acc0 : alloc.vec.Vec I128)
    (hwin : AttnWeightWindow exps total pos)
    (hvwin : AttnValueWindow hd stride first pos vplane)
    (hwres : AttnWeightResolve exps total pos recip)
    (hacl0 : acc0.val.length = hd.val)
    (hacc0zero : ∀ i, i < hd.val → (acc0.val.getD i 0#i128).val = 0) :
    ∃ acc : alloc.vec.Vec I128,
      attn.KvState.attn_head_loop2 hd pos vplane stride first exps total recip acc0 0#usize
          true none = ok (acc, none) ∧
      AttnAccInv hd pos exps total vplane stride first acc0 acc (pos.val + 1) true none := by
  have h0 : (0#usize).val = 0 := by norm_num
  have hinv0 : AttnAccInv hd pos exps total vplane stride first acc0 acc0 0 true none := by
    refine ⟨rfl, rfl, ?_, hacl0, ?_⟩
    · omega
    · intro i hi
      rw [Finset.sum_range_zero, add_zero]
  have hspec : WP.spec
      (loop (fun (st : alloc.vec.Vec Std.I128 × Std.Usize × Bool
          × (Option attn.AttnError)) =>
          attn.KvState.attn_head_loop2.body hd pos vplane stride first exps total recip st.1 st.2.1
            st.2.2.1 st.2.2.2)
        (acc0, (0#usize : Std.Usize), true, (none : Option attn.AttnError)))
      (fun b : alloc.vec.Vec Std.I128 × (Option attn.AttnError) =>
        ∃ acc : alloc.vec.Vec I128, b = (acc, none) ∧
          AttnAccInv hd pos exps total vplane stride first acc0 acc (pos.val + 1) true none) := by
    apply loop.spec_decr_nat
      (measure := fun st : alloc.vec.Vec Std.I128 × Std.Usize × Bool
          × (Option attn.AttnError) => pos.val + 1 - st.2.1.val)
      (inv := fun st => AttnAccInv hd pos exps total vplane stride first acc0 st.1 st.2.1.val st.2.2.1
        st.2.2.2)
      (post := fun b : alloc.vec.Vec Std.I128 × (Option attn.AttnError) =>
        ∃ acc : alloc.vec.Vec I128, b = (acc, none) ∧
          AttnAccInv hd pos exps total vplane stride first acc0 acc (pos.val + 1) true none)
    · intro st hst
      rcases st with ⟨acc, s1, ok1, err⟩
      obtain ⟨hok, herr, hle, hlen, hcontent⟩ := hst
      dsimp only at hok herr hle hlen hcontent
      subst hok herr
      by_cases hlt : s1.val ≤ pos.val
      · obtain ⟨cf, hcf, hpost⟩ := WP.spec_imp_exists
          (attn_head_loop2_body_step hd pos vplane stride first exps total recip acc0 acc s1 hwin
            hvwin hwres hacc0zero ⟨rfl, rfl, hle, hlen, hcontent⟩ hlt)
        obtain ⟨acc2, s2, hcfg, hinv2, hs2⟩ := hpost
        rw [hcfg] at hcf
        simp only [hcf]
        refine ⟨hinv2, ?_⟩
        show pos.val + 1 - s2.val < pos.val + 1 - s1.val
        omega
      · have hnot : ¬ (s1 ≤ pos) := by
          intro hc
          exact hlt ((UScalar.le_equiv s1 pos).1 hc)
        have hbody : attn.KvState.attn_head_loop2.body hd pos vplane stride first exps total recip
            acc s1 true none = ok (ControlFlow.done (acc, none)) := by
          rw [attn.KvState.attn_head_loop2.body.eq_def, if_neg hnot]
        simp only [hbody]
        refine ⟨acc, rfl, ?_⟩
        have h1 : s1.val = pos.val + 1 := by omega
        rw [h1] at hcontent
        exact ⟨rfl, rfl, le_of_eq rfl, hlen, hcontent⟩
    · exact hinv0
  obtain ⟨b, hb, acc, hbeq, hinv⟩ := WP.spec_imp_exists hspec
  rw [hbeq] at hb
  refine ⟨acc, ?_, hinv⟩
  unfold attn.KvState.attn_head_loop2
  exact hb

/-! ## Output stage: the write-through loop -/

/-- The exact output component of `attn_head_loop3` at index `j`: the single half-biased reduction
of the accumulator entry at `j` by `2^32`. -/
def attnOutAt (acc : alloc.vec.Vec I128) (j : ℕ) : ℤ :=
  divRoundNearest (acc.val.getD j 0#i128).val ((2 : ℤ) ^ (32 : ℕ))

/-- The output slice after `j` written components: entry `idx < j` is `V idx`, the rest unchanged
(the closed-form reference, in the style of `Numeric.RopeHeadContent`'s `rotPrefix`). -/
def outPrefix (out0 : List I64) (V : ℕ → I64) (j : ℕ) : List I64 :=
  out0.mapIdx fun idx a => if idx < j then V idx else a

theorem outPrefix_length (out0 : List I64) (V : ℕ → I64) (j : ℕ) :
    (outPrefix out0 V j).length = out0.length := by simp [outPrefix]

theorem outPrefix_zero (out0 : List I64) (V : ℕ → I64) :
    outPrefix out0 V 0 = out0 := by
  apply List.ext_getElem
  · simp [outPrefix]
  · intro n h1 _
    simp only [outPrefix, List.getElem_mapIdx]
    rw [if_neg (by omega)]

/-- **One output-loop iteration.** For `j < out0.length`, iteration `j` writes `V j` at `j`. -/
theorem outPrefix_succ (out0 : List I64) (V : ℕ → I64) (j : ℕ) (_hj : j < out0.length) :
    outPrefix out0 V (j + 1) = (outPrefix out0 V j).set j (V j) := by
  apply List.ext_getElem
  · simp [outPrefix]
  · intro n h1 _
    simp only [outPrefix, List.getElem_mapIdx, List.getElem_set]
    by_cases hjn : j = n
    · subst hjn; rw [if_pos rfl, if_pos (by omega)]
    · rw [if_neg hjn]
      by_cases hlt2 : n < j
      · rw [if_pos (by omega), if_pos hlt2]
      · rw [if_neg (by omega), if_neg hlt2]

/-- Output-loop invariant: clear flag/error, `j` within one of `head_dim`, `out` denoting
`outPrefix out0 V j`. -/
def AttnOutLoopInv (out0 : Slice I64) (V : ℕ → I64) (hd : Usize)
    (out : Slice I64) (j : Usize) (ok1 : Bool) (err : Option attn.AttnError) : Prop :=
  ok1 = true ∧ err = none ∧ j.val ≤ hd.val ∧
  out.val.length = out0.val.length ∧
  out.val = outPrefix out0.val V j.val

/-- **One output-loop step.** With `j < head_dim` and the invariant holding, the body applies the
single `round_shift_32` epilogue to the accumulator entry at `j`, narrows it to `i64` and writes
it at `j` (`round_shift_32_spec` and `IScalar.cast`, the same steps
`attn_head_loop3_body_epilogue` discharges, applied here at a single index so the loop invariant
can be carried), continuing at `j + 1` with the invariant restored. -/
theorem attn_head_loop3_body_step (hd : Usize) (acc : alloc.vec.Vec I128) (out0 : Slice I64)
    (V : ℕ → I64) (hout0len : out0.val.length = hd.val) (hacl : acc.val.length = hd.val)
    (hV : ∀ j, j < hd.val → (V j).val = attnOutAt acc j)
    (hnarrow : ∀ j, j < hd.val →
      I128.min ≤ (acc.val.getD j 0#i128).val + 2 ^ (31 : ℕ) ∧
      (acc.val.getD j 0#i128).val + 2 ^ (31 : ℕ) ≤ I128.max ∧
      -((2 : ℤ) ^ (63 : ℕ)) ≤ attnOutAt acc j ∧ attnOutAt acc j ≤ (2 : ℤ) ^ (63 : ℕ) - 1)
    (out : Slice I64) (j : Usize) (hj : j.val < hd.val)
    (hinv : AttnOutLoopInv out0 V hd out j true none) :
    WP.spec (attn.KvState.attn_head_loop3.body hd acc out j true none)
      (fun cf =>
        ∃ (out2 : Slice I64) (j2 : Usize),
          cf = ControlFlow.cont (out2, j2, true, none) ∧
          AttnOutLoopInv out0 V hd out2 j2 true none ∧
          j2.val = j.val + 1) := by
  obtain ⟨_, _, hjle, hlen, hmap⟩ := hinv
  have hjlt : j < hd := (UScalar.lt_equiv j hd).2 hj
  have hjout : j.val < out.val.length := by rw [hlen, hout0len]; exact hj
  have hjacc : j.val < acc.val.length := by rw [hacl]; exact hj
  obtain ⟨a, hacc_eq, hav⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec acc j hjacc)
  have ha_get : acc.val.getD j.val 0#i128 = a := by rw [getD_eq_get' _ hjacc, hav]
  obtain ⟨hlo, hhi, hfl, hfh⟩ := hnarrow j.val hj
  have ha0 : a.val = (acc.val.getD j.val 0#i128).val := by
    have h2 : acc.val.getD j.val 0#i128 = acc.val[j.val] := by
      rw [getD_eq_get' _ hjacc]
    rw [h2, hav]
  have hlo' : I128.min ≤ a.val + 2 ^ (31 : ℕ) := by rw [ha0]; exact hlo
  have hhi' : a.val + 2 ^ (31 : ℕ) ≤ I128.max := by rw [ha0]; exact hhi
  obtain ⟨rounded, hround_eq, hroundv⟩ := round_shift_32_spec a hlo' hhi'
  have hmaxv : (core.convert.num.FromI128I64.from core.num.I64.MAX).val
      = (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    have h : core.num.I64.MAX.val = (2 : ℤ) ^ (63 : ℕ) - 1 := by
      norm_num [core.num.I64.MAX, I64.rMax, I64.rMin, IScalar.max, I64.numBits, I64.size]
    rw [h]
  have hminv : (core.convert.num.FromI128I64.from core.num.I64.MIN).val
      = -((2 : ℤ) ^ (63 : ℕ)) := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    have h : core.num.I64.MIN.val = -((2 : ℤ) ^ (63 : ℕ)) := by
      norm_num [core.num.I64.MIN, I64.rMax, I64.rMin, IScalar.min, I64.numBits, I64.size]
    rw [h]
  have hnb : (2 : ℤ) ^ (IScalarTy.I64.numBits - 1) = (2 : ℤ) ^ 63 := by
    rw [IScalarTy.I64_numBits_eq]
  have hfl0 : divRoundNearest a.val ((2 : ℤ) ^ (32 : ℕ)) = attnOutAt acc j.val := by
    rw [ha0]
    rfl
  have hfl' : -((2 : ℤ) ^ (IScalarTy.I64.numBits - 1)) ≤ divRoundNearest a.val
      ((2 : ℤ) ^ (32 : ℕ)) := by
    rw [hnb, hfl0]
    exact hfl
  have hfh' : divRoundNearest a.val ((2 : ℤ) ^ (32 : ℕ)) < (2 : ℤ) ^ (IScalarTy.I64.numBits - 1) := by
    rw [hnb, hfl0]
    have := hfh
    omega
  have hmax : ¬ (rounded > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    intro hc
    rw [gt_iff_lt, IScalar.lt_equiv, hmaxv, hroundv] at hc
    omega
  have hmin : ¬ (rounded < core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    intro hc
    rw [IScalar.lt_equiv, hminv, hroundv] at hc
    omega
  have hVn : (V j.val).val = rounded.val := by
    rw [hV j.val hj, attnOutAt, ← ha0, hroundv]
  have hcastv : (IScalar.cast IScalarTy.I64 rounded).val = rounded.val := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 rounded.val
      (by rw [hroundv]; exact hfl') (by rw [hroundv]; exact hfh')
  have hVcast : V j.val = IScalar.cast .I64 rounded :=
    IScalar.val_eq_imp _ _ (hVn.trans hcastv.symm)
  obtain ⟨out2, hupd, hupdv⟩ := WP.spec_imp_exists (Slice.update_spec out j (V j.val) hjout)
  have hsmax : j.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [h1]
    have hb := hd.hBounds
    rw [UScalarTy.Usize_numBits_eq] at hb
    have h3 : hd.val ≤ 2 ^ System.Platform.numBits - 1 := Nat.le_sub_one_of_lt hb
    have h4 : j.val + 1 ≤ 2 ^ System.Platform.numBits - 1 := by omega
    rw [show Usize.max = 2 ^ System.Platform.numBits - 1 by
      simp [Usize.max, Usize.numBits]]
    exact h4
  obtain ⟨j2, hadd1, hj2⟩ := WP.spec_imp_exists (Usize.add_spec (x := j) (y := 1#usize) hsmax)
  have hj2' : j2.val = j.val + 1 := by
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hj2, h1]
  have hj2le : j2.val ≤ hd.val := by rw [hj2']; have := hd.hBounds; omega
  have hout0 : j.val < out0.val.length := by rw [hout0len]; exact hj
  have hlen2 : out2.val.length = out0.val.length := by
    rw [hupdv, Slice.set_val_eq, List.length_set, hlen]
  have hmap2 : out2.val = outPrefix out0.val V j2.val := by
    rw [hupdv, Slice.set_val_eq, hmap, hj2', outPrefix_succ out0.val V j.val hout0]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (out2, j2, true, none), ?_, out2, j2, rfl,
    ⟨rfl, rfl, hj2le, hlen2, hmap2⟩, hj2'⟩
  rw [attn.KvState.attn_head_loop3.body, if_pos hjlt, if_pos rfl]
  try simp only [bind_tc_ok]
  rw [alloc.vec.Vec.index_slice_index, hacc_eq]
  simp only [lift, bind_tc_ok]
  rw [hround_eq]
  try simp only [bind_tc_ok]
  rw [if_neg hmax]
  try simp only [bind_tc_ok]
  rw [if_neg hmin]
  try simp only [bind_tc_ok]
  rw [← hVcast, hupd]
  try simp only [bind_tc_ok]
  rw [hadd1]
  simp only [bind_tc_ok]
  rfl

/-- **Output-loop content.** On the epilogue narrowing at every component, `attn_head_loop3`
returns, clear, every component `i < head_dim` equal to `attnOutAt acc i`, the single half-biased
reduction of the accumulator entry at `i`. -/
theorem attn_head_loop3_spec (hd : Usize) (acc : alloc.vec.Vec I128) (out0 : Slice I64)
    (V : ℕ → I64) (hout0len : out0.val.length = hd.val) (hacl : acc.val.length = hd.val)
    (hV : ∀ j, j < hd.val → (V j).val = attnOutAt acc j)
    (hnarrow : ∀ j, j < hd.val →
      I128.min ≤ (acc.val.getD j 0#i128).val + 2 ^ (31 : ℕ) ∧
      (acc.val.getD j 0#i128).val + 2 ^ (31 : ℕ) ≤ I128.max ∧
      -((2 : ℤ) ^ (63 : ℕ)) ≤ attnOutAt acc j ∧ attnOutAt acc j ≤ (2 : ℤ) ^ (63 : ℕ) - 1) :
    ∃ out' : Slice I64,
      attn.KvState.attn_head_loop3 hd out0 acc 0#usize true none = ok (out', none) ∧
      out'.val.length = out0.val.length ∧
      ∀ j, j < hd.val → (out'.val.getD j 0#i64).val = attnOutAt acc j := by
  have hinv0 : AttnOutLoopInv out0 V hd out0 0#usize true none := by
    refine ⟨rfl, rfl, ?_, rfl, ?_⟩
    · have h0 : (0#usize).val = 0 := by norm_num
      omega
    · have h0 : (0#usize).val = 0 := by norm_num
      rw [h0]
      exact (outPrefix_zero out0.val V).symm
  have hspec : WP.spec
      (loop (fun (st : Slice Std.I64 × Std.Usize × Bool × (Option attn.AttnError)) =>
          attn.KvState.attn_head_loop3.body hd acc st.1 st.2.1 st.2.2.1 st.2.2.2)
        (out0, (0#usize : Std.Usize), true, (none : Option attn.AttnError)))
      (fun b : Slice Std.I64 × (Option attn.AttnError) =>
        ∃ out' : Slice I64, b = (out', none) ∧
          out'.val.length = out0.val.length ∧
          ∀ j, j < hd.val → (out'.val.getD j 0#i64).val = attnOutAt acc j) := by
    apply loop.spec_decr_nat
      (measure := fun st : Slice Std.I64 × Std.Usize × Bool
          × (Option attn.AttnError) => hd.val - st.2.1.val)
      (inv := fun st => AttnOutLoopInv out0 V hd st.1 st.2.1 st.2.2.1 st.2.2.2)
      (post := fun b : Slice Std.I64 × (Option attn.AttnError) =>
        ∃ out' : Slice I64, b = (out', none) ∧
          out'.val.length = out0.val.length ∧
          ∀ j, j < hd.val → (out'.val.getD j 0#i64).val = attnOutAt acc j)
    · intro st hst
      rcases st with ⟨out, j1, ok1, err⟩
      obtain ⟨hok, herr, hjle, hlen, hmap⟩ := hst
      dsimp only at hok herr hjle hlen hmap
      subst hok herr
      by_cases hlt : j1.val < hd.val
      · obtain ⟨cf, hcf, hpost⟩ := WP.spec_imp_exists
          (attn_head_loop3_body_step hd acc out0 V hout0len hacl hV hnarrow out j1 hlt
            ⟨rfl, rfl, hjle, hlen, hmap⟩)
        obtain ⟨out2, j2, hcfg, hinv2, hj2⟩ := hpost
        rw [hcfg] at hcf
        simp only [hcf]
        refine ⟨hinv2, ?_⟩
        show hd.val - j2.val < hd.val - j1.val
        omega
      · have hnot : ¬ (j1 < hd) := by
          intro hc
          exact hlt ((UScalar.lt_equiv j1 hd).1 hc)
        have hbody : attn.KvState.attn_head_loop3.body hd acc out j1 true none
            = ok (ControlFlow.done (out, none)) := by
          rw [attn.KvState.attn_head_loop3.body, if_neg hnot]
        simp only [hbody]
        refine ⟨out, rfl, hlen, ?_⟩
        have hj1eq : j1.val = hd.val := by omega
        rw [hj1eq] at hmap
        intro j hj
        have hjlt2 : j < out0.val.length := by rw [hout0len]; exact hj
        rw [hmap]
        simp only [outPrefix, List.getD_eq_getElem?_getD, List.getElem?_mapIdx,
          List.getElem?_eq_getElem hjlt2, Option.map_some, Option.getD_some]
        rw [if_pos hj, hV j hj]
    · exact hinv0
  obtain ⟨b, hb, out', hbeq, hlen, hval'⟩ := WP.spec_imp_exists hspec
  rw [hbeq] at hb
  refine ⟨out', ?_, hlen, hval'⟩
  unfold attn.KvState.attn_head_loop3
  exact hb

/-! ## The per-component composition -/

/-- **Per-component content of the value loop.** Each component `i < head_dim` of the accumulator
`attn_head_loop2` returns is the exact sum `Σ_{s≤pos} attnWeightAt exps total s · V_{s,i}` with `V`
the `valueList` entries — the value-column content the predecessor stated as
`value_component_spec`'s first half, re-derived for the folded pass. -/
theorem attn_head_loop2_component (hd pos : Usize) (vplane : alloc.vec.Vec I64)
    (exps : alloc.vec.Vec I64) (total : I128) (acc : alloc.vec.Vec I128)
    (kv kvh : ℕ)
    (hcontent : ∀ i, i < hd.val →
      (acc.val.getD i 0#i128).val =
        ∑ j ∈ Finset.range (pos.val + 1), attnWeightAt exps.val total j *
          attnValueAt vplane.val (kv * hd.val) (kvh * hd.val) j i) :
    ∀ i, i < hd.val →
      (acc.val.getD i 0#i128).val =
        ∑ s ∈ Finset.range (pos.val + 1), attnWeightAt exps.val total s *
          (valueList vplane.val kv hd.val kvh pos.val i).getD s 0 := by
  intro i hi
  rw [hcontent i hi]
  apply Finset.sum_congr rfl
  intro j hj
  simp only [Finset.mem_range] at hj
  rw [attnValueAt_eq vplane.val kv hd.val kvh pos.val i j hi (by omega)]

/-- **Per-component composition.** On `AttnWeightWindow` and the value row window, the value loop
followed by the output loop returns, clear, every component `i < head_dim` equal to the exact
half-biased reduction `divRoundNearest (Σ_{s≤pos} W_s·V_{s,i}) 2^32` of the `valueList` entries —
the claim of the predecessor's `value_component_spec`, now on the executed path of the reshaped
kernel. The narrowing is the output epilogue's own (`hnarrow`), passed through unchanged.

The output length `out'.val.length = out0.val.length` is exported as a third conjunct, after the two
loop equations and before the per-component content. It is `attn_head_loop3_spec`'s second component
(`Numeric/AttnLoopOutput.lean:880`), which this theorem previously destructured as `houtlen` and
dropped; the end-to-end target `attn_head_refines_v2`
(`Numeric/AttnSoftmaxPerturb.lean:739`) requires it. The export is additive: no existing conjunct is
changed, and the per-component conclusion is the same statement. -/
theorem value_component_spec (hd pos : Usize) (vplane : alloc.vec.Vec I64)
    (exps : alloc.vec.Vec I64) (total : I128) (recip : Option U64) (acc0 : alloc.vec.Vec I128)
    (out0 : Slice I64) (stride first : Usize) (kv kvh : ℕ)
    (hstride : stride.val = kv * hd.val) (hfirst : first.val = kvh * hd.val)
    (hwin : AttnWeightWindow exps total pos)
    (hvwin : AttnValueWindow hd stride first pos vplane)
    (hwres : AttnWeightResolve exps total pos recip)
    (hacl0 : acc0.val.length = hd.val)
    (hacc0zero : ∀ i, i < hd.val → (acc0.val.getD i 0#i128).val = 0)
    (hout0len : out0.val.length = hd.val)
    (V : ℕ → I64)
    (hV : ∀ (acc : alloc.vec.Vec I128),
      attn.KvState.attn_head_loop2 hd pos vplane stride first exps total recip acc0 0#usize
          true none = ok (acc, none) →
      ∀ j, j < hd.val → (V j).val = attnOutAt acc j)
    (hnarrow : ∀ (acc : alloc.vec.Vec I128),
      attn.KvState.attn_head_loop2 hd pos vplane stride first exps total recip acc0 0#usize
          true none = ok (acc, none) →
      ∀ j, j < hd.val →
        I128.min ≤ (acc.val.getD j 0#i128).val + 2 ^ (31 : ℕ) ∧
        (acc.val.getD j 0#i128).val + 2 ^ (31 : ℕ) ≤ I128.max ∧
        -((2 : ℤ) ^ (63 : ℕ)) ≤ attnOutAt acc j ∧ attnOutAt acc j ≤ (2 : ℤ) ^ (63 : ℕ) - 1) :
    ∃ (acc : alloc.vec.Vec I128) (out' : Slice I64),
      attn.KvState.attn_head_loop2 hd pos vplane stride first exps total recip
          acc0 0#usize true none = ok (acc, none) ∧
      attn.KvState.attn_head_loop3 hd out0 acc 0#usize true none = ok (out', none) ∧
      out'.val.length = out0.val.length ∧
      ∀ i, i < hd.val →
        (out'.val.getD i 0#i64).val =
          divRoundNearest
            (∑ s ∈ Finset.range (pos.val + 1), attnWeightAt exps.val total s *
              (valueList vplane.val kv hd.val kvh pos.val i).getD s 0) ((2 : ℤ) ^ (32 : ℕ)) := by
  obtain ⟨acc, haccloop, haccinv⟩ :=
    attn_head_loop2_spec hd pos vplane stride first exps total recip acc0 hwin hvwin hwres hacl0
      hacc0zero
  obtain ⟨out', houtloop, houtlen, houtval⟩ :=
    attn_head_loop3_spec hd acc out0 V hout0len haccinv.2.2.2.1 (hV acc haccloop)
      (hnarrow acc haccloop)
  refine ⟨acc, out', haccloop, houtloop, houtlen, ?_⟩
  intro i hi
  have hcc : ∀ i, i < hd.val →
      (acc.val.getD i 0#i128).val =
        ∑ j ∈ Finset.range (pos.val + 1), attnWeightAt exps.val total j *
          attnValueAt vplane.val (kv * hd.val) (kvh * hd.val) j i := by
    intro i hi
    have hz := haccinv.2.2.2.2 i hi
    rw [hstride, hfirst] at hz
    rw [hz, hacc0zero i hi, zero_add]
  have hcontent := attn_head_loop2_component hd pos vplane exps total acc kv kvh hcc
  have hvl : (acc.val.getD i 0#i128).val =
      ∑ s ∈ Finset.range (pos.val + 1), attnWeightAt exps.val total s *
        (valueList vplane.val kv hd.val kvh pos.val i).getD s 0 := hcontent i hi
  rw [houtval i hi, ← hV acc haccloop i hi, ← hvl]
  exact hV acc haccloop i hi

end

end apeiron_word
