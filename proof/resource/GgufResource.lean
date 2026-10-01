import GgufSource
import Aeneas

/-
SCOPE: resource theorems for the actual extracted `src/format/gguf.rs` header-parse body reached by
  the `word_gguf_parse_header` extraction root: the byte cursor's per-read transition, the
  length-prefixed string scan loop and its total byte count, the metadata, tensor-dimension and
  tensor-directory loops with their rank decrease and refusal exits, the `general.alignment` scan and
  its key-comparison loop, the allocation-driving count guards, and the reachable zero-count states.
REGISTER: local source resource/termination obligation - byte and iteration work measures under
  named guards, refusal branches, reachable witnesses. Excluded: the canonical ring/table algebra,
  hardware or wall-clock timing, and the metadata value interpreter `Cur::value` beyond its role as
  the `parse_kv` callee.
CORPUS: src/format/gguf.rs lines 180-233, 262-275, 462-608 and the actual extracted
  gguf.Cur.{u8,string,string_loop,array_header}, gguf.parse_kv, gguf.parse_tensor,
  gguf.parse_tensor_loop and gguf.parse_header_loop{,0,1,1_loop0,2} of
  proof/resource/GgufSource.lean.
REFERENCE: NONE - a resource claim over the extracted body; no canonical-corpus theorem is refined.
GATE: python3 proof/resource/Gguf_gate.py --controls, 540 s task budget, lane 2.
GRADE: Lean checked by that gate under the {propext, Classical.choice, Quot.sound} cone. The work
  measures are byte counts and iteration counts under named premises, not cycles and not GPU time.
OPEN: the whole-loop iteration bound for the metadata, tensor-dimension and tensor-directory loops
  (`loop.spec_decr_nat` over their bodies) is not established here: those bodies call `parse_kv` /
  `parse_tensor`, which may return a `Result.fail` in the model, and `fail e ⦃ P ⦄` is unprovable in
  Aeneas (`WP.spec` has only the `ret` constructor), so the loop's step obligation cannot be
  discharged for those callees without a totality premise about them. Their per-iteration
  transitions and decreasing ranks are established (`kv_loop_step`, `dims_loop_step`,
  `tensor_loop_step`), and the byte-work measure with a full loop specification is established for
  the string scan (`string_loop_bytes`).
-/

namespace GgufResource
open Aeneas Aeneas.Std Result
open gguf_source

/-! ## Scalar and `Option` guard helpers -/

lemma uscalar_lt_of {ty : UScalarTy} {x y : UScalar ty} (h : x.val < y.val) : x < y :=
  (UScalar.lt_equiv x y).2 h

lemma uscalar_lt_neg {ty : UScalarTy} {x y : UScalar ty} (h : y.val ≤ x.val) : ¬ (x < y) := by
  intro hc
  have h1 := (UScalar.lt_equiv x y).1 hc
  omega

lemma uscalar_gt_of {ty : UScalarTy} {x y : UScalar ty} (h : x.val > y.val) : x > y :=
  (UScalar.lt_equiv y x).2 h

/-- `Usize.max` is positive: the machine's index width is nonzero. -/
lemma usize_max_pos : (0 : Nat) < Usize.max := by
  have h2 : (1 : Nat) < 2 ^ UScalarTy.Usize.numBits :=
    Nat.one_lt_two_pow (UScalarTy.numBits_nonzero UScalarTy.Usize)
  have hb := Usize.cMax_bound
  have hconv : 2 ^ UScalarTy.Usize.numBits = 2 ^ System.Platform.numBits := by congr 1
  omega

lemma is_none_none_eq_true {α : Type} :
    core.option.Option.is_none (none : Option α) = true := rfl

lemma is_none_some_eq_false {α : Type} (e : α) :
    core.option.Option.is_none (some e) = false := rfl

/-- The `Option` branch the extracted `?` pattern produces. -/
lemma option_branch_some {α : Type} (v : α) :
    core.option.Option.Insts.CoreOpsTry_traitTry.branch (some v) =
      ok (core.ops.control_flow.ControlFlow.Continue v) := rfl

/-! ## The byte cursor: one read is one byte, and a missing byte consumes nothing

`gguf.Cur.u8` is the only read in the parse that advances the cursor, so its transition is the
per-byte work measure of the whole header parse. -/

/-- A successful byte read returns that byte, keeps the buffer, and advances the cursor by exactly
    one byte. -/
theorem cur_u8_step (s : gguf.Cur) (v : Std.U8) (h : s.b.val[s.i.val]? = some v)
    (hmax : s.i.val + 1 ≤ Usize.max) :
    gguf.Cur.u8 s ⦃ r => r.1 = some v ∧ r.2.b = s.b ∧ r.2.i.val = s.i.val + 1 ⦄ := by
  have h1 : core.slice.Slice.get (core.slice.index.SliceIndexUsizeSlice Std.U8) s.b s.i =
      ok (some v) := by
    rw [show core.slice.Slice.get (core.slice.index.SliceIndexUsizeSlice Std.U8) s.b s.i =
      ok (s.b.val[s.i.val]?) from rfl, h]
  obtain ⟨i1, hadd, hiv⟩ := WP.spec_imp_exists (Usize.add_spec (x := s.i) (y := 1#usize) hmax)
  have hiv' : i1.val = s.i.val + 1 := by simpa using hiv
  unfold gguf.Cur.u8
  rw [h1]
  simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch]
  rw [hadd]
  simp only [bind_tc_ok]
  exact ⟨rfl, rfl, hiv'⟩

/-- A read whose byte is absent leaves the cursor where it was and yields no value: the parse never
    consumes a byte it did not read. -/
theorem cur_u8_refuse (s : gguf.Cur) (h : s.b.val[s.i.val]? = none) :
    gguf.Cur.u8 s ⦃ r => r.1 = none ∧ r.2 = s ⦄ := by
  have h1 : core.slice.Slice.get (core.slice.index.SliceIndexUsizeSlice Std.U8) s.b s.i =
      ok none := by
    rw [show core.slice.Slice.get (core.slice.index.SliceIndexUsizeSlice Std.U8) s.b s.i =
      ok (s.b.val[s.i.val]?) from rfl, h]
  unfold gguf.Cur.u8
  rw [h1]
  simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch, bind_tc_ok]
  simp [core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual]

/-! ## The length-prefixed string scan: bytes consumed, and a decreasing rank -/

/-- One guarded scan iteration copies exactly the byte at the cursor, advances the cursor by one
    byte, and strictly decreases the remaining rank `end - k`. -/
theorem string_loop_step (s : Slice Std.U8) («end» k : Std.Usize) (out : alloc.vec.Vec Std.U8)
    (hk : k.val < «end».val) (hindex : k.val < s.val.length)
    (hpush : out.val.length < Usize.max) (hkadd : k.val + 1 ≤ Usize.max) :
    ∃ out' : alloc.vec.Vec Std.U8, ∃ k' : Std.Usize,
      gguf.Cur.string_loop.body s «end» out k = ok (ControlFlow.cont (out', k')) ∧
      k'.val = k.val + 1 ∧ out'.val = out.val ++ [s.val[k.val]] ∧
      out'.val.length = out.val.length + 1 ∧
      «end».val - k'.val < «end».val - k.val := by
  have hguard : k < «end» := uscalar_lt_of hk
  obtain ⟨x, hx, hxv⟩ := WP.spec_imp_exists (Slice.index_usize_spec s k hindex)
  obtain ⟨out', hpush', hout⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out x hpush)
  obtain ⟨k', hk', hkv⟩ := WP.spec_imp_exists (Usize.add_spec (x := k) (y := 1#usize) hkadd)
  have hkv' : k'.val = k.val + 1 := by simpa using hkv
  refine ⟨out', k', ?_, hkv', ?_, ?_, ?_⟩
  · unfold gguf.Cur.string_loop.body
    simp [hguard, hx, hpush', hk']
  · rw [hout, hxv]
  · rw [hout]; simp
  · rw [hkv']; omega

/-- The scan leaves only once the cursor has reached the declared end: it reads no byte past it. -/
theorem string_loop_exit (s : Slice Std.U8) («end» k : Std.Usize) (out : alloc.vec.Vec Std.U8)
    (hk : «end».val ≤ k.val) :
    gguf.Cur.string_loop.body s «end» out k = ok (ControlFlow.done out) := by
  unfold gguf.Cur.string_loop.body
  rw [if_neg (uscalar_lt_neg hk)]

/-- The scan's state invariant: the cursor lies between the start and the declared end inside the
    buffer, the produced vector holds exactly the bytes copied so far, and the remaining work fits
    the vector bound. -/
def stringInv (s : Slice Std.U8) («end» : Std.Usize) (out0 : alloc.vec.Vec Std.U8)
    (k0 : Std.Usize) (st : alloc.vec.Vec Std.U8 × Std.Usize) : Prop :=
  k0.val ≤ st.2.val ∧ st.2.val ≤ «end».val ∧ «end».val ≤ s.val.length ∧
    st.1.val.length = out0.val.length + (st.2.val - k0.val) ∧
    st.1.val.length + («end».val - st.2.val) ≤ Usize.max

/-- The whole scan copies exactly `end - k` bytes: the work measure is the byte distance from the
    cursor to the declared end, and each iteration lowers it by one. -/
theorem string_loop_bytes (s : Slice Std.U8) («end» k0 : Std.Usize)
    (out0 : alloc.vec.Vec Std.U8) (hk0 : k0.val ≤ «end».val) (hbounded : «end».val ≤ s.val.length)
    (hcap : out0.val.length + («end».val - k0.val) ≤ Usize.max) :
    gguf.Cur.string_loop s «end» out0 k0 ⦃ out' =>
      out'.val.length = out0.val.length + («end».val - k0.val) ⦄ := by
  unfold gguf.Cur.string_loop
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec Std.U8 × Std.Usize => «end».val - st.2.val)
    (inv := fun st : alloc.vec.Vec Std.U8 × Std.Usize => stringInv s «end» out0 k0 st)
    (post := fun (out' : alloc.vec.Vec Std.U8) =>
      out'.val.length = out0.val.length + («end».val - k0.val))
  · intro st hst
    rcases st with ⟨out, k⟩
    dsimp only at hst ⊢
    simp only [stringInv] at hst
    obtain ⟨hk0le, hkle, hbound, hlen, hcap0⟩ := hst
    by_cases hlt : k.val < «end».val
    · have hklt : k.val < s.val.length := by omega
      have hkadd : k.val + 1 ≤ Usize.max := by
        have hle := s.property
        omega
      have hpush : out.val.length < Usize.max := by
        rw [hlen]; omega
      refine Exists.elim (string_loop_step s «end» k out hlt hklt hpush hkadd) fun out' hrest =>
        Exists.elim hrest fun k' hrest => ?_
      rcases hrest with ⟨hbody, hk', hout', hlen', hrank⟩
      have hnext : k0.val ≤ k'.val ∧ k'.val ≤ «end».val ∧ «end».val ≤ s.val.length ∧
          out'.val.length = out0.val.length + (k'.val - k0.val) ∧
          out'.val.length + («end».val - k'.val) ≤ Usize.max := by
        refine ⟨by omega, by omega, hbound, ?_, ?_⟩
        · rw [hlen', hlen]; omega
        · rw [hlen', hlen]; omega
      rw [hbody]
      simp only [stringInv]
      exact ⟨hnext, hrank⟩
    · rw [string_loop_exit s «end» k out (by omega)]
      refine WP.spec.ret ?_
      show out.val.length = out0.val.length + («end».val - k0.val)
      rw [hlen]
      omega
  · simp only [stringInv]
    exact ⟨le_rfl, hk0, hbounded, by omega, hcap⟩

/-- A concrete one-byte payload reaches the scan's continuation: one iteration, one byte copied,
    and that byte is the buffer's byte at the cursor. -/
def oneByte : Slice Std.U8 := Slice.from [7#u8] (by have := usize_max_pos; simp; omega)

theorem string_loop_step_witness :
    ∃ out' : alloc.vec.Vec Std.U8, ∃ k' : Std.Usize,
      gguf.Cur.string_loop.body oneByte 1#usize (alloc.vec.Vec.new Std.U8) 0#usize =
        ok (ControlFlow.cont (out', k')) ∧
      out'.val = [7#u8] ∧ k'.val = 1 := by
  have hval : (oneByte : Slice Std.U8).val = [7#u8] := by
    rw [oneByte]; exact Slice.from_val _ _
  have hzero : ((0#usize : Std.Usize).val = 0) := rfl
  have hone : ((1#usize : Std.Usize).val = 1) := rfl
  have hindex : (0 : Nat) < (oneByte : Slice Std.U8).val.length := by
    rw [hval]; simp
  have hpush : (alloc.vec.Vec.new Std.U8 : alloc.vec.Vec Std.U8).val.length < Usize.max := by
    rw [alloc.vec.Vec.new]; simp; have hz := usize_max_pos; omega
  have hguard : ((0#usize : Std.Usize).val < (1#usize : Std.Usize).val) := by omega
  have hkadd : ((0#usize : Std.Usize).val + 1 ≤ Usize.max) := by
    have hz := usize_max_pos; omega
  obtain ⟨out', k', hbody, hk', hout, hlen', hrank⟩ :=
    string_loop_step oneByte 1#usize 0#usize (alloc.vec.Vec.new Std.U8)
      hguard hindex hpush hkadd
  have hbyte : ((oneByte : Slice Std.U8).val[(0#usize : Std.Usize).val] : Std.U8) = 7#u8 := by
    show ([7#u8] : List Std.U8)[(0#usize : Std.Usize).val] = 7#u8
    simp
  refine ⟨out', k', hbody, ?_, ?_⟩
  · rw [hout]
    simp [alloc.vec.Vec.new, hval]
  · rw [hk']; omega

/-! ## Refusal branches: the length guards that bound the work -/

/-- A length prefix that does not fit `usize` is refused before any byte is copied. -/
theorem string_refuses_oversized_length (self : gguf.Cur) (val : U64) (self1 : gguf.Cur)
    (hu : gguf.Cur.u64 self = ok (some val, self1))
    (hbig : val > UScalar.cast .U64 core.num.Usize.MAX) :
    gguf.Cur.string self = ok (none, self1) := by
  unfold gguf.Cur.string
  rw [hu]
  simp only [lift, bind_tc_ok, uncurry_apply_pair, option_branch_some, bind_tc_ok]
  rw [if_pos hbig]

/-- A length prefix whose end runs past the buffer is refused: no byte past the buffer is read. -/
theorem string_refuses_past_end (self : gguf.Cur) (val : U64) (self1 : gguf.Cur)
    (n «end» : Std.Usize) (hu : gguf.Cur.u64 self = ok (some val, self1))
    (hcast : UScalar.cast .Usize val = n)
    (hadd : Usize.checked_add self1.i n = some «end»)
    (hnotbig : ¬ (val > UScalar.cast .U64 core.num.Usize.MAX))
    (hpast : «end».val > self1.b.val.length) :
    gguf.Cur.string self = ok (none, self1) := by
  unfold gguf.Cur.string
  rw [hu]
  simp only [lift, bind_tc_ok, uncurry_apply_pair, option_branch_some, bind_tc_ok,
    if_neg hnotbig, hcast, hadd, option_branch_some, bind_tc_ok]
  rw [if_pos (uscalar_gt_of hpast)]

/-- The `ARRAY` header's count guard: a count larger than the bytes the buffer still holds is
    refused, so a bogus count cannot drive the element allocation. -/
theorem array_header_refuses_oversized_count (self : gguf.Cur) (et : U32) (val1 : U64)
    (self1 self2 : gguf.Cur) (n i : Std.Usize)
    (hu32 : gguf.Cur.u32 self = ok (some et, self1))
    (hu64 : gguf.Cur.u64 self1 = ok (some val1, self2))
    (hcast : UScalar.cast .Usize val1 = n)
    (hnotbig : ¬ (val1 > UScalar.cast .U64 core.num.Usize.MAX))
    (hsat : core.num.Usize.saturating_sub (Slice.len self2.b) self2.i = i)
    (hbig : n > i) :
    gguf.Cur.array_header self = ok (none, self2) := by
  unfold gguf.Cur.array_header
  rw [hu32]
  simp only [lift, bind_tc_ok, uncurry_apply_pair, option_branch_some, bind_tc_ok, hu64,
    lift, bind_tc_ok, uncurry_apply_pair, option_branch_some, bind_tc_ok,
    if_neg hnotbig, hcast, bind_tc_ok, hsat, bind_tc_ok]
  rw [if_pos (uscalar_lt_of hbig)]

/-! ## The metadata loop (`parse_header`'s first loop) -/

/-- One guarded metadata iteration appends exactly one entry, advances the entry counter by one,
    and strictly decreases the remaining rank `n_kv - ki`. -/
theorem kv_loop_step (n_kv : U64) (c : gguf.Cur)
    (kv : alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal)) (ki : U64)
    (entry : alloc.vec.Vec Std.U8 × gguf.GVal) (c' : gguf.Cur)
    (hguard : ki.val < n_kv.val)
    (hkvg : gguf.parse_kv c = ok (core.result.Result.Ok entry, c'))
    (hpush : kv.val.length < Usize.max) (hkadd : ki.val + 1 ≤ U64.max) :
    ∃ kv' : alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal), ∃ ki' : U64,
      gguf.parse_header_loop0.body n_kv c kv none ki = ok (ControlFlow.cont (c', kv', none, ki')) ∧
      ki'.val = ki.val + 1 ∧ kv'.val = kv.val ++ [entry] ∧
      kv'.val.length = kv.val.length + 1 ∧
      n_kv.val - ki'.val < n_kv.val - ki.val := by
  obtain ⟨kv', hpush', hkv⟩ :=
    WP.spec_imp_exists (alloc.vec.Vec.push_spec kv entry hpush)
  obtain ⟨ki', hkadd', hki⟩ := WP.spec_imp_exists (U64.add_spec (x := ki) (y := 1#u64) hkadd)
  have hki' : ki'.val = ki.val + 1 := by simpa using hki
  refine ⟨kv', ki', ?_, hki', ?_, ?_, ?_⟩
  · unfold gguf.parse_header_loop0.body
    simp [uscalar_lt_of hguard, hkvg, hpush', hkadd']
  · rw [hkv]
  · rw [hkv]; simp
  · rw [hki']; omega

/-- A failing metadata read is recorded in the error slot and no entry is appended; the next
    iteration leaves the loop at once, so no further entry is parsed. -/
theorem kv_loop_refuses_entry (n_kv : U64) (c : gguf.Cur)
    (kv : alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal)) (e : gguf.GgufError) (c' : gguf.Cur)
    (ki : U64) (hguard : ki.val < n_kv.val)
    (hkvge : gguf.parse_kv c = ok (core.result.Result.Err e, c')) :
    gguf.parse_header_loop0.body n_kv c kv none ki =
      ok (ControlFlow.cont (c', kv, some e, ki)) := by
  unfold gguf.parse_header_loop0.body
  simp [uscalar_lt_of hguard, hkvge]

/-- A recorded refusal leaves the metadata loop at once, with the table unchanged. -/
theorem kv_loop_error_exit (n_kv : U64) (c : gguf.Cur)
    (kv : alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal)) (e : gguf.GgufError) (ki : U64) :
    gguf.parse_header_loop0.body n_kv c kv (some e) ki =
      ok (ControlFlow.done (c.b, c.i, kv, some e)) := by
  unfold gguf.parse_header_loop0.body
  simp

/-- The metadata loop stops when the declared entry count is reached. -/
theorem kv_loop_count_exit (n_kv : U64) (c : gguf.Cur)
    (kv : alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal)) (ki : U64)
    (hki : n_kv.val ≤ ki.val) :
    gguf.parse_header_loop0.body n_kv c kv none ki = ok (ControlFlow.done (c.b, c.i, kv, none)) := by
  unfold gguf.parse_header_loop0.body
  rw [if_neg (uscalar_lt_neg hki)]

/-- A header declaring no metadata entries reaches the loop body's exit at once, with the table
    unchanged and no pending refusal: the zero-work path of the metadata loop. -/
theorem kv_loop_zero_count (c : gguf.Cur)
    (kv : alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal)) :
    gguf.parse_header_loop0.body 0#u64 c kv none 0#u64 =
      ok (ControlFlow.done (c.b, c.i, kv, none)) :=
  kv_loop_count_exit 0#u64 c kv 0#u64 le_rfl

/-- The metadata loop's state invariant: the entry counter, the produced table and the pending
    refusal agree, the remaining capacity fits, and a recorded refusal was taken below the declared
    count. -/
def kvInv (n_kv : U64) (kv0 : alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal)) (ki0 : U64)
    (st : gguf.Cur × (alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal) ×
      (Option gguf.GgufError) × U64)) : Prop :=
  (∃ j, st.2.1.val.length = kv0.val.length + j ∧ st.2.2.2.val = ki0.val + j ∧
    j ≤ n_kv.val - ki0.val) ∧
  st.2.1.val.length + (n_kv.val - st.2.2.2.val) ≤ Usize.max ∧
  (st.2.2.1 ≠ none → st.2.2.2.val < n_kv.val)

/-- The work measure of the metadata loop: two units per unparsed entry plus one unit while no
    refusal is pending. Every iteration lowers it. -/
def kvMeasure (n_kv : U64)
    (st : gguf.Cur × (alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal) ×
      (Option gguf.GgufError) × U64)) : Nat :=
  2 * (n_kv.val - st.2.2.2.val) + (if st.2.2.1 = none then 1 else 0)

/-! ## The tensor-dimension loop (`parse_tensor`) -/

/-- One guarded dimension iteration appends exactly one dimension, advances the counter by one, and
    strictly decreases the remaining rank `nd - k`. -/
theorem dims_loop_step (nd : U32) (c : gguf.Cur) (dims : alloc.vec.Vec U64) (k : U32)
    (d : U64) (c' : gguf.Cur)
    (hguard : k.val < nd.val) (hu : gguf.Cur.u64 c = ok (some d, c'))
    (hpush : dims.val.length < Usize.max) (hkadd : k.val + 1 ≤ U32.max) :
    ∃ dims' : alloc.vec.Vec U64, ∃ k' : U32,
      gguf.parse_tensor_loop.body nd c dims none k = ok (ControlFlow.cont (c', dims', none, k')) ∧
      k'.val = k.val + 1 ∧ dims'.val = dims.val ++ [d] ∧
      dims'.val.length = dims.val.length + 1 ∧
      nd.val - k'.val < nd.val - k.val := by
  obtain ⟨dims', hpush', hdims⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec dims d hpush)
  obtain ⟨k', hkadd', hk⟩ := WP.spec_imp_exists (U32.add_spec (x := k) (y := 1#u32) hkadd)
  have hk' : k'.val = k.val + 1 := by simpa using hk
  refine ⟨dims', k', ?_, hk', ?_, ?_, ?_⟩
  · unfold gguf.parse_tensor_loop.body
    simp [uscalar_lt_of hguard, hu, hpush', hkadd']
  · rw [hdims]
  · rw [hdims]; simp
  · rw [hk']; omega

/-- A dimension read that runs past the buffer is recorded as `Truncated "tensor dims"`, and the
    loop then leaves: no further dimension is read. -/
theorem dims_loop_refuses_truncated (nd : U32) (c : gguf.Cur) (dims : alloc.vec.Vec U64)
    (k : U32) (c' : gguf.Cur) (hguard : k.val < nd.val)
    (hu : gguf.Cur.u64 c = ok (none, c')) :
    gguf.parse_tensor_loop.body nd c dims none k =
      ok (ControlFlow.cont (c', dims, some (gguf.GgufError.Truncated (toStr "tensor dims")), k)) := by
  unfold gguf.parse_tensor_loop.body
  simp [uscalar_lt_of hguard, hu]

/-- The recorded `Truncated "tensor dims"` leaves the dimension loop at once. -/
theorem dims_loop_error_exit (nd : U32) (c : gguf.Cur) (dims : alloc.vec.Vec U64) (k : U32) :
    gguf.parse_tensor_loop.body nd c dims
        (some (gguf.GgufError.Truncated (toStr "tensor dims"))) k =
      ok (ControlFlow.done (c.b, c.i, dims, some (gguf.GgufError.Truncated (toStr "tensor dims")))) := by
  unfold gguf.parse_tensor_loop.body
  simp

/-- The dimension loop stops when the declared rank is reached. -/
theorem dims_loop_count_exit (nd : U32) (c : gguf.Cur) (dims : alloc.vec.Vec U64) (k : U32)
    (hki : nd.val ≤ k.val) :
    gguf.parse_tensor_loop.body nd c dims none k = ok (ControlFlow.done (c.b, c.i, dims, none)) := by
  unfold gguf.parse_tensor_loop.body
  rw [if_neg (uscalar_lt_neg hki)]

/-! ## The tensor-directory loop (`parse_header`'s third loop) -/

/-- One guarded directory iteration appends exactly one descriptor, advances the counter by one, and
    strictly decreases the remaining rank `n_tensors - ti`. -/
theorem tensor_loop_step (n_tensors : U64) (c : gguf.Cur) (tensors : alloc.vec.Vec gguf.TensorDesc)
    (ti : U64) (desc : gguf.TensorDesc) (c' : gguf.Cur)
    (hguard : ti.val < n_tensors.val)
    (ht : gguf.parse_tensor c = ok (core.result.Result.Ok desc, c'))
    (hpush : tensors.val.length < Usize.max) (hkadd : ti.val + 1 ≤ U64.max) :
    ∃ tensors' : alloc.vec.Vec gguf.TensorDesc, ∃ ti' : U64,
      gguf.parse_header_loop2.body n_tensors c tensors none ti =
        ok (ControlFlow.cont (c', tensors', none, ti')) ∧
      ti'.val = ti.val + 1 ∧ tensors'.val = tensors.val ++ [desc] ∧
      tensors'.val.length = tensors.val.length + 1 ∧
      n_tensors.val - ti'.val < n_tensors.val - ti.val := by
  obtain ⟨tensors', hpush', htensors⟩ :=
    WP.spec_imp_exists (alloc.vec.Vec.push_spec tensors desc hpush)
  obtain ⟨ti', hkadd', hti⟩ := WP.spec_imp_exists (U64.add_spec (x := ti) (y := 1#u64) hkadd)
  have hti' : ti'.val = ti.val + 1 := by simpa using hti
  refine ⟨tensors', ti', ?_, hti', ?_, ?_, ?_⟩
  · unfold gguf.parse_header_loop2.body
    simp [uscalar_lt_of hguard, ht, hpush', hkadd']
  · rw [htensors]
  · rw [htensors]; simp
  · rw [hti']; omega

/-- A failing tensor-directory read stops the loop with the recorded refusal. -/
theorem tensor_loop_refuses_entry (n_tensors : U64) (c : gguf.Cur)
    (tensors : alloc.vec.Vec gguf.TensorDesc) (e : gguf.GgufError) (ti : U64) (c' : gguf.Cur)
    (hguard : ti.val < n_tensors.val)
    (ht : gguf.parse_tensor c = ok (core.result.Result.Err e, c')) :
    gguf.parse_header_loop2.body n_tensors c tensors none ti =
      ok (ControlFlow.cont (c', tensors, some e, ti)) := by
  unfold gguf.parse_header_loop2.body
  simp [uscalar_lt_of hguard, ht]

/-- A recorded refusal leaves the tensor-directory loop at once, with the table unchanged. -/
theorem tensor_loop_error_exit (n_tensors : U64) (c : gguf.Cur)
    (tensors : alloc.vec.Vec gguf.TensorDesc) (e : gguf.GgufError) (ti : U64) :
    gguf.parse_header_loop2.body n_tensors c tensors (some e) ti =
      ok (ControlFlow.done (c.i, tensors, some e)) := by
  unfold gguf.parse_header_loop2.body
  simp

/-- The directory loop stops when the declared descriptor count is reached. -/
theorem tensor_loop_count_exit (n_tensors : U64) (c : gguf.Cur)
    (tensors : alloc.vec.Vec gguf.TensorDesc) (ti : U64) (hti : n_tensors.val ≤ ti.val) :
    gguf.parse_header_loop2.body n_tensors c tensors none ti =
      ok (ControlFlow.done (c.i, tensors, none)) := by
  unfold gguf.parse_header_loop2.body
  rw [if_neg (uscalar_lt_neg hti)]

/-- A header declaring no tensors reaches the directory loop body's exit at once, with the table
    unchanged and no pending refusal: the zero-work path of the directory loop. -/
theorem tensor_loop_zero_count (c : gguf.Cur) (tensors : alloc.vec.Vec gguf.TensorDesc) :
    gguf.parse_header_loop2.body 0#u64 c tensors none 0#u64 = ok (ControlFlow.done (c.i, tensors, none)) :=
  tensor_loop_count_exit 0#u64 c tensors 0#u64 le_rfl

/-! ## The `general.alignment` key scan -/

/-- The key-length mismatch branch: the key is not the alignment key, the alignment is left
    unchanged, and the entry index advances by one. -/
theorem align_loop_step_len_ne (kv : alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal))
    (alignment : U64) (ai : Std.Usize) (v : alloc.vec.Vec Std.U8) (g : gguf.GVal)
    (hguard : ai < alloc.vec.Vec.len kv)
    (hidx : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice
      (alloc.vec.Vec Std.U8 × gguf.GVal)) kv ai = ok (v, g))
    (hne : v.val.length ≠ gguf.ALIGN_LEN) (hkadd : ai.val + 1 ≤ Usize.max) :
    ∃ ai' : Std.Usize,
      gguf.parse_header_loop1.body kv alignment ai = ok (ControlFlow.cont (alignment, ai')) ∧
      ai'.val = ai.val + 1 ∧
      kv.val.length - ai'.val < kv.val.length - ai.val := by
  obtain ⟨ai', hkadd', hk⟩ := WP.spec_imp_exists (Usize.add_spec (x := ai) (y := 1#usize) hkadd)
  have hk' : ai'.val = ai.val + 1 := by simpa using hk
  refine ⟨ai', ?_, hk', ?_⟩
  · unfold gguf.parse_header_loop1.body
    simp [uscalar_lt_of hguard, hidx, hne, hkadd']
  · rw [hk']
    have hlt' : ai.val < kv.val.length := by simpa using hguard
    omega

/-- The alignment scan leaves once every metadata entry has been examined. -/
theorem align_loop_exit (kv : alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal))
    (alignment : U64) (ai : Std.Usize) (hguard : ¬ (ai < alloc.vec.Vec.len kv)) :
    gguf.parse_header_loop1.body kv alignment ai = ok (ControlFlow.done alignment) := by
  unfold gguf.parse_header_loop1.body
  rw [if_neg hguard]

/-- An empty metadata table reaches the alignment scan body's exit with the default alignment: the
    reachable zero-work path of the scan. -/
theorem align_loop_empty :
    gguf.parse_header_loop1.body
      (alloc.vec.Vec.with_capacity (alloc.vec.Vec Std.U8 × gguf.GVal) 0#usize)
      gguf.DEFAULT_ALIGNMENT 0#usize = ok (ControlFlow.done gguf.DEFAULT_ALIGNMENT) := by
  refine align_loop_exit _ _ _ ?_
  have hlen : (alloc.vec.Vec.len (alloc.vec.Vec.with_capacity
      (alloc.vec.Vec Std.U8 × gguf.GVal) 0#usize) : Std.Usize).val = 0 := rfl
  show ¬ ((0#usize : Std.Usize).val < _)
  simp only [hlen]
  omega

/-- One key-byte comparison that differs forces the flag to `false` and advances the compare cursor
    by one under the strictly decreasing rank `ALIGN_LEN - j`. -/
theorem align_key_step_ne (kv : alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal))
    (ai j : Std.Usize) (same : Bool) (v : alloc.vec.Vec Std.U8) (g : gguf.GVal) (x y : Std.U8)
    (hj : j.val < gguf.ALIGN_LEN)
    (hidx : kv.index_usize ai = ok (v, g))
    (hidx2 : v.index_usize j = ok x)
    (harr : gguf.ALIGN_BYTES.index_usize j = ok y) (hne : ¬ (x = y))
    (hadd : j.val + 1 ≤ Usize.max) :
    ∃ j' : Std.Usize,
      gguf.parse_header_loop1_loop0.body kv ai j same = ok (ControlFlow.cont (j', false)) ∧
      j'.val = j.val + 1 ∧ gguf.ALIGN_LEN - j'.val < gguf.ALIGN_LEN - j.val := by
  obtain ⟨j', hkadd, hk⟩ := WP.spec_imp_exists (Usize.add_spec (x := j) (y := 1#usize) hadd)
  have hk' : j'.val = j.val + 1 := by simpa using hk
  refine ⟨j', ?_, hk', ?_⟩
  · unfold gguf.parse_header_loop1_loop0.body
    simp only [alloc.vec.Vec.index_slice_index] at hidx hidx2 ⊢
    rw [if_pos (uscalar_lt_of hj), hidx]
    simp only [bind_tc_ok, uncurry_apply_pair]
    rw [hidx2, harr]
    simp only [bind_tc_ok]
    rw [if_pos (by simp [hne]), hkadd]
    simp only [bind_tc_ok]
  · rw [hk', ← Nat.sub_sub]
    exact Nat.sub_lt (by omega) (by omega)

/-- One key-byte comparison that agrees preserves the flag and advances the compare cursor by one
    under the strictly decreasing rank `ALIGN_LEN - j`. -/
theorem align_key_step_eq (kv : alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal))
    (ai j : Std.Usize) (same : Bool) (v : alloc.vec.Vec Std.U8) (g : gguf.GVal) (x y : Std.U8)
    (hj : j.val < gguf.ALIGN_LEN)
    (hidx : kv.index_usize ai = ok (v, g))
    (hidx2 : v.index_usize j = ok x)
    (harr : gguf.ALIGN_BYTES.index_usize j = ok y) (hne : x = y)
    (hadd : j.val + 1 ≤ Usize.max) :
    ∃ j' : Std.Usize,
      gguf.parse_header_loop1_loop0.body kv ai j same = ok (ControlFlow.cont (j', same)) ∧
      j'.val = j.val + 1 ∧ gguf.ALIGN_LEN - j'.val < gguf.ALIGN_LEN - j.val := by
  obtain ⟨j', hkadd, hk⟩ := WP.spec_imp_exists (Usize.add_spec (x := j) (y := 1#usize) hadd)
  have hk' : j'.val = j.val + 1 := by simpa using hk
  refine ⟨j', ?_, hk', ?_⟩
  · unfold gguf.parse_header_loop1_loop0.body
    simp only [alloc.vec.Vec.index_slice_index] at hidx hidx2 ⊢
    rw [if_pos (uscalar_lt_of hj), hidx]
    simp only [bind_tc_ok, uncurry_apply_pair]
    rw [hidx2, harr]
    simp only [bind_tc_ok]
    rw [if_neg (by simp [hne]), hkadd]
    simp only [bind_tc_ok]
  · rw [hk', ← Nat.sub_sub]
    exact Nat.sub_lt (by omega) (by omega)

/-- The key comparison stops at the key length: it reads no byte past `ALIGN_LEN`. -/
theorem align_key_exit (kv : alloc.vec.Vec (alloc.vec.Vec Std.U8 × gguf.GVal)) (ai j : Std.Usize)
    (same : Bool) (hguard : ¬ (j.val < gguf.ALIGN_LEN)) :
    gguf.parse_header_loop1_loop0.body kv ai j same = ok (ControlFlow.done same) := by
  unfold gguf.parse_header_loop1_loop0.body
  rw [if_neg (uscalar_lt_neg (by omega))]

end GgufResource

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

#resource_audit GgufResource.cur_u8_step
#resource_audit GgufResource.cur_u8_refuse
#resource_audit GgufResource.string_loop_step
#resource_audit GgufResource.string_loop_exit
#resource_audit GgufResource.string_loop_bytes
#resource_audit GgufResource.string_loop_step_witness
#resource_audit GgufResource.string_refuses_oversized_length
#resource_audit GgufResource.string_refuses_past_end
#resource_audit GgufResource.array_header_refuses_oversized_count
#resource_audit GgufResource.kv_loop_step
#resource_audit GgufResource.kv_loop_refuses_entry
#resource_audit GgufResource.kv_loop_error_exit
#resource_audit GgufResource.kv_loop_count_exit
#resource_audit GgufResource.kv_loop_zero_count
#resource_audit GgufResource.dims_loop_step
#resource_audit GgufResource.dims_loop_refuses_truncated
#resource_audit GgufResource.dims_loop_error_exit
#resource_audit GgufResource.dims_loop_count_exit
#resource_audit GgufResource.tensor_loop_step
#resource_audit GgufResource.tensor_loop_refuses_entry
#resource_audit GgufResource.tensor_loop_error_exit
#resource_audit GgufResource.tensor_loop_count_exit
#resource_audit GgufResource.tensor_loop_zero_count
#resource_audit GgufResource.align_loop_step_len_ne
#resource_audit GgufResource.align_loop_exit
#resource_audit GgufResource.align_loop_empty
#resource_audit GgufResource.align_key_step_ne
#resource_audit GgufResource.align_key_step_eq
#resource_audit GgufResource.align_key_exit

run_cmd Lean.logInfo "GGUF_KERNEL_OK"