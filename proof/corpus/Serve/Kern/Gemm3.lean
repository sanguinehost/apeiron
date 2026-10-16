/-
Integer, index and layout refinement of the extracted `kern::gemm` bodies, third group.

Serving path: `Engine::next_token_state` -> `step::cols` -> `kern::gemm::gemm_q8_0_group`. This
module refines the portable (non-AVX2, non-`unsafe`) bodies:

* the three loops of `gemm_q8_0_group`: `..._loop0` (the part contract scan), `..._loop1` (the `k`
  empty flat columns), `..._loop2` (the split of the flat columns into per-part outputs);
* the packer loops of `pack_group`: `..._loop0_loop0` (the scale scan), `..._loop0_loop1_loop0` (the
  quad words), `..._loop0_loop1` (the block loop), and the pad loops `..._loop1_loop0_loop0`,
  `..._loop1_loop0`, `..._loop1`;
* `fast_columns_ok` / `..._loop0` (the acceptance predicate) and `column_ptrs` / `..._loop0` (the
  per-column spare-capacity pointers).

Each reference is plain Lean over `Nat`/`List`: a contract predicate, an index arithmetic, a `Bool`
scan. It states WHAT the body computes, not how the loop runs. `word_<fn>_ok` is the refinement;
`word_<fn>_refuses` is the exact refusal condition.

NOT refined here, with the reason stated rather than skirted:

* the tile value itself. Every output word is written by `kern.gemm.x86.rows`, emitted by the
  extraction as `axiom`; `gemm_q8_0_group` also calls the `std_detect` `axiom`s and the `axiom` pool
  driver `chart.pool.chunks.for_pbands`. No `wp` spec is derivable for an `axiom : Result _`, so
  neither the dot-product equality nor `ok`-totality of the composed body is provable at this layer.
* `group_band` / `group_band_loop`: the loop body calls `kern.gemm.x86.rows` (`axiom`) and
  `loader.PayloadRef.bytes`; the same obstruction, and its band argument is a pool `PBand` whose
  `rows` is an `axiom`.
* `pack` / `pack_group` composed: they reach `chart.pool.chunks.for_chunks4` (`axiom`); only the
  inner loops, whose bodies are pure, are refined.
* `column_ptrs`: the returned pointers are `MutRawPtr`, whose values no reference describes; only
  the layout (one pointer per column, the input slice unchanged) is stated.
-/
import Serve.Kern.Gemm
import Serve.Kern.Gemm2

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## References (plain Lean) -/

/-- The four refusal disjuncts of `partRefusedRef`, as a `Bool` scan. -/
def gemm3_refusedPart (file : Slice Std.U8) (rowBytes : Std.Usize)
    (gp : kern.gemm.GemmPart) : Bool :=
  decide (gp.ggml_type.val ≠ 8) || decide (gp.n_out.val = 0) ||
    decide ((Slice.len file).val < satAddRef Usize.max gp.at.off.val gp.at.len.val) ||
    decide (gp.at.len.val < satMulRef Usize.max gp.n_out.val rowBytes.val)

/-- The refusal scan over the first `n` parts: `true` exactly when one of them is refused. -/
def gemm3_refuseAny (file : Slice Std.U8) (parts : Slice kern.gemm.GemmPart)
    (rowBytes : Std.Usize) (n : Nat) : Bool :=
  (parts.val.take n).any (gemm3_refusedPart file rowBytes)

/-- The two refusal predicates agree. -/
private lemma gemm3_refusedPart_iff (file : Slice Std.U8) (rowBytes : Std.Usize)
    (gp : kern.gemm.GemmPart) :
    gemm3_refusedPart file rowBytes gp = true ↔
      partRefusedRef Usize.max (Slice.len file).val gp.ggml_type.val gp.n_out.val
        gp.at.off.val gp.at.len.val rowBytes.val := by
  simp only [gemm3_refusedPart, partRefusedRef, Bool.or_eq_true, decide_eq_true_eq]
  tauto

/-- Flatten two levels of a vector of vectors of vectors. -/
def gemm3_vals3 (v : alloc.vec.Vec (alloc.vec.Vec (alloc.vec.Vec Std.I64))) :
    List (List (List Std.I64)) :=
  v.val.map (fun col => vecVals col)

/-- The split reference of `gemm_q8_0_group_loop2`: part `q`'s columns, each split at its own row
start. `zipWith` truncates at the shorter list, so the definition is total for any `rstarts`. -/
def gemm3_splitRef (flat : alloc.vec.Vec (alloc.vec.Vec Std.I64))
    (parts : Slice kern.gemm.GemmPart) (rstarts : alloc.vec.Vec Std.Usize) (n : Nat) :
    List (List (List Std.I64)) :=
  List.zipWith (fun gp s => flat.val.map (splitPartRef s.val gp.n_out.val))
    (parts.val.take n) (rstarts.val.take n)

private lemma gemm3_splitRef_succ (flat : alloc.vec.Vec (alloc.vec.Vec Std.I64))
    (parts : Slice kern.gemm.GemmPart) (rstarts : alloc.vec.Vec Std.Usize) (n : Nat)
    (hn : n < parts.val.length) (hnr : parts.val.length ≤ rstarts.val.length) :
    gemm3_splitRef flat parts rstarts (n + 1) =
      gemm3_splitRef flat parts rstarts n ++
        [flat.val.map (splitPartRef (rstarts.val[n]).val (parts.val[n]).n_out.val)] := by
  have hn' : n < rstarts.val.length := Nat.lt_of_lt_of_le hn hnr
  have hlen : (parts.val.take n).length = (rstarts.val.take n).length := by
    simp [List.length_take, Nat.min_eq_left (Nat.le_of_lt hn), Nat.min_eq_left (Nat.le_of_lt hn')]
  unfold gemm3_splitRef
  rw [List.take_succ_eq_append_getElem hn, List.take_succ_eq_append_getElem hn',
    List.zipWith_append hlen, List.zipWith_cons_cons, List.zipWith_nil_left]

private lemma gemm3_replicate_succ_append (n : Nat) (a : List Std.I64) :
    List.replicate (n + 1) a = List.replicate n a ++ [a] := by
  simpa using (List.replicate_succ' (n := n) (a := a))

private lemma gemm3_usize_max_pos : (0 : Nat) < Usize.max := by
  have h := usize_le_max (1#usize)
  have h1 : (1#usize : Std.Usize).val = 1 := by scalar_tac
  omega

/-- `Vec.deref` does not change the value list. -/
private lemma gemm3_deref_val (T : Type) (v : alloc.vec.Vec T) :
    (↑(alloc.vec.Vec.deref v) : List T) = v.val := by
  simp [alloc.vec.Vec.deref]

/-! ## `part_refused` as a step spec -/

/-- `part_refused` returns the reference predicate word. -/
@[step]
theorem gemm3_part_refused_spec (file : Slice Std.U8) (part : kern.gemm.GemmPart)
    (row_bytes : Std.Usize) :
    kern.gemm.part_refused file part row_bytes ⦃ b =>
      b = gemm3_refusedPart file row_bytes part ⦄ := by
  refine WP.spec_mono (word_part_refused_ok file part row_bytes) ?_
  intro b hb
  have hiff := gemm3_refusedPart_iff file row_bytes part
  cases b <;> cases hgr : gemm3_refusedPart file row_bytes part <;> simp_all

/-! ## `gemm_q8_0_group` loop 0: the part contract scan -/

private lemma gemm3_refuseAny_succ (file : Slice Std.U8) (parts : Slice kern.gemm.GemmPart)
    (rowBytes : Std.Usize) (n : Nat) (h : n < parts.val.length) :
    gemm3_refuseAny file parts rowBytes (n + 1) =
      (gemm3_refuseAny file parts rowBytes n ||
        gemm3_refusedPart file rowBytes (parts.val[n]'h)) := by
  unfold gemm3_refuseAny
  rw [List.take_succ_eq_append_getElem h, List.any_append, List.any_cons, List.any_nil,
    Bool.or_false]

theorem gemm3_gemm_q8_0_group_loop0_spec (file : Slice Std.U8)
    (parts : Slice kern.gemm.GemmPart) (rowBytes : Std.Usize) (refuse0 : Bool) :
    kern.gemm.gemm_q8_0_group_loop0 file parts rowBytes refuse0 0#usize ⦃ r =>
      r = (refuse0 || gemm3_refuseAny file parts rowBytes parts.val.length) ⦄ := by
  unfold kern.gemm.gemm_q8_0_group_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Bool × Std.Usize => parts.val.length - st.2.val)
    (inv := fun st : Bool × Std.Usize =>
      st.2.val ≤ parts.val.length ∧
      st.1 = (refuse0 || gemm3_refuseAny file parts rowBytes st.2.val))
    (post := fun r =>
      r = (refuse0 || gemm3_refuseAny file parts rowBytes parts.val.length))
  · rintro ⟨refuse, pi⟩ ⟨hpi', hinv'⟩
    dsimp only at *
    unfold kern.gemm.gemm_q8_0_group_loop0.body
    by_cases hlt : pi < Slice.len parts
    · rw [if_pos hlt]
      have hpiLt : pi.val < parts.val.length := by
        simpa [Slice.len_val, UScalar.lt_equiv] using hlt
      by_cases hr : refuse = true
      · rw [if_pos hr]
        step*
        refine ⟨?_, ?_, ?_⟩
        · rw [pi1_post]; omega
        · rw [pi1_post, gemm3_refuseAny_succ file parts rowBytes pi.val hpiLt]
          have h1 : (refuse0 || gemm3_refuseAny file parts rowBytes pi.val) = true := by
            rw [← hinv']; exact hr
          rw [← Bool.or_assoc, h1, Bool.true_or]
        · rw [pi1_post]; omega
      · rw [if_neg hr]
        have hrf : refuse = false := by cases refuse <;> simp_all
        step*
        have hor : (refuse0 || gemm3_refuseAny file parts rowBytes pi.val) = false := by
          rw [← hinv']; exact hrf
        have h0 : refuse0 = false := (Bool.or_eq_false_iff.mp hor).1
        have hany : gemm3_refuseAny file parts rowBytes pi.val = false :=
          (Bool.or_eq_false_iff.mp hor).2
        refine ⟨?_, ?_, ?_⟩
        · rw [pi1_post]; omega
        · rw [pi1_post, gemm3_refuseAny_succ file parts rowBytes pi.val hpiLt]
          rw [refuse1_post, x_post, h0, hany]
          simp only [Bool.false_or]
        · rw [pi1_post]; omega
    · rw [if_neg hlt]
      have hge : parts.val.length ≤ pi.val := by
        have h : ¬ (pi.val < parts.val.length) := by
          simpa [Slice.len_val, UScalar.lt_equiv] using hlt
        omega
      have hpk : pi.val = parts.val.length := le_antisymm hpi' hge
      simp only [WP.spec_ok]
      rw [← hpk]
      exact hinv'
  · refine ⟨by simp, ?_⟩
    simp [gemm3_refuseAny]

/-- Non-vacuity: a zero-row part is refused. -/
example : gemm3_refusedPart (Slice.from ([] : List Std.U8) (by simp))
    (32#usize) { ggml_type := 8#u32, «at» := ⟨0#usize, 0#usize⟩, n_out := 0#usize } = true := by
  simp [gemm3_refusedPart]

/-- Non-vacuity: the scan over no parts is `false`, over one refused part is `true`. -/
example : gemm3_refuseAny (Slice.from ([] : List Std.U8) (by simp))
    (Slice.from ([{ ggml_type := 8#u32, «at» := ⟨0#usize, 0#usize⟩, n_out := 0#usize }]
      : List kern.gemm.GemmPart)
      (by rw [List.length_cons, List.length_nil]; exact gemm3_usize_max_pos))
    (32#usize) 0 = false ∧
    gemm3_refuseAny (Slice.from ([] : List Std.U8) (by simp))
    (Slice.from ([{ ggml_type := 8#u32, «at» := ⟨0#usize, 0#usize⟩, n_out := 0#usize }]
      : List kern.gemm.GemmPart)
      (by rw [List.length_cons, List.length_nil]; exact gemm3_usize_max_pos))
    (32#usize) 1 = true := by
  constructor <;> simp [gemm3_refuseAny, gemm3_refusedPart]

/-! ## `gemm_q8_0_group` loop 1: the `k` empty flat columns -/

/-- The value list of a `Vec.new` is empty. -/
private lemma gemm3_vecNew_val (T : Type) : (alloc.vec.Vec.new T).val = [] := rfl

theorem gemm3_gemm_q8_0_group_loop1_spec (k : Std.Usize)
    (flat0 : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c0 : Std.Usize)
    (hc0 : c0.val ≤ k.val)
    (hcap0 : flat0.val.length + (k.val - c0.val) ≤ Usize.max) :
    kern.gemm.gemm_q8_0_group_loop1 k flat0 c0 ⦃ r =>
      vecVals r = vecVals flat0 ++ List.replicate (k.val - c0.val) [] ⦄ := by
  unfold kern.gemm.gemm_q8_0_group_loop1
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec (alloc.vec.Vec Std.I64) × Std.Usize =>
      k.val - st.2.val)
    (inv := fun st : alloc.vec.Vec (alloc.vec.Vec Std.I64) × Std.Usize =>
      c0.val ≤ st.2.val ∧ st.2.val ≤ k.val ∧
      st.1.val.length + (k.val - st.2.val) ≤ Usize.max ∧
      vecVals st.1 = vecVals flat0 ++ List.replicate (st.2.val - c0.val) [])
    (post := fun r => vecVals r = vecVals flat0 ++ List.replicate (k.val - c0.val) [])
  · rintro ⟨flat, c⟩ ⟨hc0', hc', hcap, hinv'⟩
    dsimp only at *
    unfold kern.gemm.gemm_q8_0_group_loop1.body
    by_cases hlt : c < k
    · rw [if_pos hlt]
      have hck : c.val < k.val := by simpa [UScalar.lt_equiv] using hlt
      step*
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · rw [c1_post]; omega
      · rw [c1_post]; omega
      · simp only [flat1_post, List.length_append, List.length_cons, List.length_nil, c1_post]
        omega
      · simp only [vecVals] at hinv' ⊢
        simp only [flat1_post, c1_post, List.map_append, List.map_cons, List.map_nil,
          gemm3_vecNew_val, hinv']
        have hidx : c.val + 1 - c0.val = (c.val - c0.val) + 1 := by omega
        rw [hidx, gemm3_replicate_succ_append, List.append_assoc]
      · rw [c1_post]; omega
    · rw [if_neg hlt]
      have hge : k.val ≤ c.val := by
        have h : ¬ (c.val < k.val) := by simpa [UScalar.lt_equiv] using hlt
        omega
      have hck : c.val = k.val := le_antisymm hc' hge
      simp only [WP.spec_ok]
      rw [← hck]
      exact hinv'
  · exact ⟨le_refl _, hc0, hcap0, by simp⟩

/-- Non-vacuity: three flat columns are three empty value lists. -/
example : List.replicate 3 [] = ([[], [], []] : List (List Std.I64)) := by decide

/-- Non-vacuity of the joint preconditions at `k = c0 = 0` and an empty start. -/
example : ∃ (flat0 : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (k c0 : Std.Usize),
    c0.val ≤ k.val ∧ flat0.val.length + (k.val - c0.val) ≤ Usize.max :=
  ⟨alloc.vec.Vec.new (alloc.vec.Vec Std.I64), 0#usize, 0#usize, by scalar_tac,
    by rw [gemm3_vecNew_val, List.length_nil, Nat.sub_self, Nat.add_zero]
       exact Nat.zero_le _⟩

/-! ## `gemm_q8_0_group` loop 2: split the flat columns into per-part outputs -/

/-- `split_part` returns the reference split, as a step spec. -/
@[step]
theorem gemm3_split_part_spec (flat : Slice (alloc.vec.Vec Std.I64)) (start n_out : Std.Usize)
    (hno : start.val + n_out.val ≤ Usize.max) (hcap : flat.val.length < Usize.max)
    (hlen : ∀ (i : Nat) (h : i < flat.val.length),
      start.val + n_out.val ≤ (flat.val[i]'h).val.length) :
    kern.gemm.split_part flat start n_out ⦃ r =>
      vecVals r = flat.val.map (splitPartRef start.val n_out.val) ⦄ :=
  word_split_part_ok flat start n_out hno hcap hlen

theorem gemm3_gemm_q8_0_group_loop2_spec
    (parts : Slice kern.gemm.GemmPart) (rstarts : alloc.vec.Vec Std.Usize)
    (flat : alloc.vec.Vec (alloc.vec.Vec Std.I64))
    (out0 : alloc.vec.Vec (alloc.vec.Vec (alloc.vec.Vec Std.I64)))
    (hpartsR : parts.val.length ≤ rstarts.val.length)
    (hflat : flat.val.length < Usize.max)
    (hno : ∀ (q : Nat) (hq : q < parts.val.length),
      (rstarts.val[q]).val + (parts.val[q]).n_out.val ≤ Usize.max)
    (hlen : ∀ (q : Nat) (hq : q < parts.val.length) (i : Nat) (hi : i < flat.val.length),
      (rstarts.val[q]).val + (parts.val[q]).n_out.val ≤ (flat.val[i]'hi).val.length)
    (hout : out0.val.length + parts.val.length ≤ Usize.max) :
    kern.gemm.gemm_q8_0_group_loop2 parts rstarts flat out0 0#usize ⦃ r =>
      gemm3_vals3 r = gemm3_vals3 out0 ++ gemm3_splitRef flat parts rstarts parts.val.length ⦄ := by
  unfold kern.gemm.gemm_q8_0_group_loop2
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec (alloc.vec.Vec (alloc.vec.Vec Std.I64)) × Std.Usize =>
      parts.val.length - st.2.val)
    (inv := fun st : alloc.vec.Vec (alloc.vec.Vec (alloc.vec.Vec Std.I64)) × Std.Usize =>
      st.2.val ≤ parts.val.length ∧
      st.1.val.length + (parts.val.length - st.2.val) ≤ Usize.max ∧
      gemm3_vals3 st.1 = gemm3_vals3 out0 ++ gemm3_splitRef flat parts rstarts st.2.val)
    (post := fun r =>
      gemm3_vals3 r = gemm3_vals3 out0 ++ gemm3_splitRef flat parts rstarts parts.val.length)
  · rintro ⟨out, p⟩ ⟨hp, hcap', hinv⟩
    dsimp only at *
    unfold kern.gemm.gemm_q8_0_group_loop2.body
    by_cases hlt : p < Slice.len parts
    · rw [if_pos hlt]
      have hpLt : p.val < parts.val.length := by
        simpa [Slice.len_val, UScalar.lt_equiv] using hlt
      step*
      · rw [gemm3_deref_val]; exact hflat
      · intro i hi
        rw [i1_post, gp_post]
        simp only [gemm3_deref_val] at hi ⊢
        exact hlen p.val hpLt i hi
      · refine ⟨?_, ?_, ?_, ?_⟩
        · rw [p1_post]; omega
        · simp only [out1_post, p1_post, List.length_append, List.length_cons, List.length_nil]
          omega
        · have hv : vecVals v
              = flat.val.map (splitPartRef (rstarts.val[p.val]).val (parts.val[p.val]).n_out.val) := by
            simp only [v_post, i1_post, gp_post, gemm3_deref_val]
          have h1 : gemm3_vals3 out1 = gemm3_vals3 out ++ [vecVals v] := by
            simp only [gemm3_vals3]
            rw [out1_post, List.map_append, List.map_cons, List.map_nil]
          rw [h1, hinv, hv, p1_post,
            gemm3_splitRef_succ flat parts rstarts p.val hpLt hpartsR, List.append_assoc]
        · rw [p1_post]; omega
    · rw [if_neg hlt]
      have hge : parts.val.length ≤ p.val := by
        have h : ¬ (p.val < parts.val.length) := by
          simpa [Slice.len_val, UScalar.lt_equiv] using hlt
        omega
      have hpk : p.val = parts.val.length := le_antisymm hp hge
      simp only [WP.spec_ok]
      rw [← hpk]
      exact hinv
  · refine ⟨by simp, ?_, ?_⟩
    · simpa using hout
    · simp [gemm3_splitRef]

/-! ## `pack_group` loop 6: the pad quads -/

/-- The pad reference of `pack_group_loop1_loop0_loop0`: the first `q` quads of block `b` and pad
lane `pad_lane` set to the byte mask `0x80808080`, starting from `w0`. -/
def gemm3_padWords (w0 : List Std.U32) (base q : Nat) : List Std.U32 :=
  (List.range q).foldl (fun w k => w.set (base + k * 8) 2155905152#u32) w0

private lemma gemm3_padWords_succ (w0 : List Std.U32) (base q : Nat) :
    gemm3_padWords w0 base (q + 1) =
      (gemm3_padWords w0 base q).set (base + q * 8) 2155905152#u32 := by
  unfold gemm3_padWords
  rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

/-- `Slice.length` is the length of the coerced list, so the two forms are interchangeable in a
rewrite. -/
private lemma gemm3_len_eq {α : Type} (s : Slice α) : s.length = s.val.length := rfl

/-- The literal `LANES` value, which the loop's index arithmetic uses. -/
private lemma gemm3_LANES_val : (kern.gemm.LANES : Std.Usize).val = 8 := by
  unfold kern.gemm.LANES
  rfl

theorem gemm3_pack_group_loop1_loop0_loop0_spec (pad_lane b : Std.Usize)
    (words0 : Slice Std.U32)
    (hno : (b.val * 8 + 8) * 8 + pad_lane.val ≤ Usize.max)
    (hbound : ∀ (q : Nat) (_ : q < 8), (b.val * 8 + q) * 8 + pad_lane.val < words0.val.length) :
    kern.gemm.pack_group_loop1_loop0_loop0 words0 pad_lane b 0#usize ⦃ r =>
      r.val = gemm3_padWords words0.val (b.val * 8 * 8 + pad_lane.val) 8 ⦄ := by
  unfold kern.gemm.pack_group_loop1_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Slice Std.U32 × Std.Usize => 8 - st.2.val)
    (inv := fun st : Slice Std.U32 × Std.Usize =>
      st.2.val ≤ 8 ∧ st.1.val.length = words0.val.length ∧
      st.1.val = gemm3_padWords words0.val (b.val * 8 * 8 + pad_lane.val) st.2.val)
    (post := fun r : Slice Std.U32 =>
      r.val = gemm3_padWords words0.val (b.val * 8 * 8 + pad_lane.val) 8)
  · rintro ⟨words, q⟩ ⟨hq, hlen, hinv⟩
    dsimp only at *
    unfold kern.gemm.pack_group_loop1_loop0_loop0.body
    by_cases hlt : q < 8#usize
    · rw [if_pos hlt]
      have hq8 : q.val < 8 := by simpa [UScalar.lt_equiv] using hlt
      step*
      · rw [i1_post, i_post, gemm3_LANES_val]; omega
      · rw [i2_post, i1_post, i_post, gemm3_LANES_val]; omega
      · rw [i3_post, i2_post, i1_post, i_post, gemm3_LANES_val, gemm3_len_eq, hlen]
        exact hbound q.val hq8
      · refine ⟨?_, ?_, ?_, ?_⟩
        · rw [quad1_post]; omega
        · simp only [s_post, Slice.set, Slice.setAtNat, Slice.from_val, List.length_set, hlen]
        · have hi3 : (↑i3 : Nat)
              = (b.val * 8 * 8 + pad_lane.val) + q.val * 8 := by
            rw [i3_post, i2_post, i1_post, i_post, gemm3_LANES_val]; omega
          simp only [s_post, Slice.set, Slice.setAtNat, Slice.from_val]
          rw [hi3, quad1_post, gemm3_padWords_succ, ← hinv]
        · rw [quad1_post]; omega
    · rw [if_neg hlt]
      have hge : 8 ≤ q.val := by
        have h : ¬ (q.val < 8) := by simpa [UScalar.lt_equiv] using hlt
        omega
      have hqk : q.val = 8 := le_antisymm hq hge
      simp only [WP.spec_ok]
      exact hqk ▸ hinv
  · refine ⟨by simp, by simp, ?_⟩
    simp [gemm3_padWords]

/-- Non-vacuity of the pad reference: one quad pads only its own word, at `base = 0`. -/
example : gemm3_padWords ([0#u32, 0#u32, 0#u32] : List Std.U32) 0 1
    = [2155905152#u32, 0#u32, 0#u32] := by decide

/-! ## Axiom cone -/
#print axioms apeiron_word.gemm3_part_refused_spec
#print axioms apeiron_word.gemm3_gemm_q8_0_group_loop0_spec
#print axioms apeiron_word.gemm3_gemm_q8_0_group_loop1_spec
#print axioms apeiron_word.gemm3_split_part_spec
#print axioms apeiron_word.gemm3_gemm_q8_0_group_loop2_spec
#print axioms apeiron_word.gemm3_pack_group_loop1_loop0_loop0_spec

end apeiron_word
