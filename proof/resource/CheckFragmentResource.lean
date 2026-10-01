import CheckFragmentSource
import Aeneas

/- SCOPE: the real `src/spirv/check.rs` `check_fragment` header guards, the instruction-scan
   transitions with their D1 framing guards, the exit and refusal branches, the word-rank work
   measure and its strictly decreasing step, and a reachable acceptance.
   REGISTER: resource and source refinement of the checker body only; no GPU, hardware timing or
   canonical-corpus claim.
   CORPUS: `src/spirv/check.rs` lines 171-208 (`check_fragment`), 191-206 (the `while` scan loop),
   read with 360-624 (`shape_of`) and 627-675 (`apply_shape`); the extracted bodies under test are
   `check_fragment_source.spirv.check.check_fragment` and
   `check_fragment_source.spirv.check.check_fragment_loop`.
   REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean, the committed extraction of the
   same source, read for the shape only; no theorem is imported from it.
   GATE: `python3 proof/resource/CheckFragment_gate.py --controls`, 540 s task budget.
   GRADE: Lean kernel check of the extracted body; no hardware measurement. -/

namespace CheckFragmentResource
open Aeneas Aeneas.Std Result
open check_fragment_source.spirv.check ControlFlow

/-! ## Platform bounds

The model's `Usize.max` is `2 ^ System.Platform.numBits - 1` with `numBits` either 32 or 64, so
`Usize.max` is `U32.max` or `U64.max`. Both cases are discharged here; the concrete witnesses below
are built only from the resulting bounds. -/

theorem usize_max_u32 (h32 : System.Platform.numBits = 32) : Usize.max = U32.max := by
  rw [Usize.max, Usize.numBits, UScalarTy.Usize_numBits_eq, h32, U32.max_eq]
  norm_num

theorem usize_max_u64 (h64 : System.Platform.numBits = 64) : Usize.max = U64.max := by
  rw [Usize.max, Usize.numBits, UScalarTy.Usize_numBits_eq, h64, U64.max_eq]
  norm_num

theorem usize_ge_five : 5 ≤ Usize.max := by
  rcases System.Platform.numBits_eq with h32 | h64
  · rw [usize_max_u32 h32, U32.max_eq]
    norm_num
  · rw [usize_max_u64 h64, U64.max_eq]
    norm_num

theorem usize_ge_zero : 0 ≤ Usize.max := by
  rcases System.Platform.numBits_eq with h32 | h64
  · rw [usize_max_u32 h32, U32.max_eq]
    norm_num
  · rw [usize_max_u64 h64, U64.max_eq]
    norm_num

/-! ## In-bounds reads and decoded fields

The scan reads `words[index]` only under `index < words.length`; the extracted body needs that as a
premise to eliminate `Slice.index_usize`. -/

theorem index_ok {α : Type} (s : Slice α) (i : Usize) (h : i.val < s.length) :
    ∃ x : α, s.index_usize i = ok x := by
  obtain ⟨x, hx, _⟩ := WP.spec_imp_exists (Slice.index_usize_spec s i h)
  exact ⟨x, hx⟩

/-- The operand window `words[start..end)` of one instruction, in bounds. The body reads it only
    after the D1 framing guard has bounded `end`. -/
theorem operand_window (words : Slice U32) (start «end» : Usize)
    (hlo : start.val ≤ «end».val) (hhi : «end».val ≤ words.length) :
    core.slice.index.Slice.index
        (core.slice.index.SliceIndexRangeUsizeSlice Std.U32) words
        { start := start, «end» := «end» }
      = ok (Slice.from (words.val.slice start.val «end».val) (by scalar_tac)) := by
  have hcond : start ≤ «end» ∧ «end».val ≤ words.length := by
    constructor
    · show start.val ≤ «end».val
      omega
    · show «end».val ≤ words.length
      omega
  simp only [core.slice.index.Slice.index, core.slice.index.SliceIndexRangeUsizeSlice.index,
    if_pos hcond]

/-- `UScalar.cast` preserves the value, so a decoded `u32` `WordCount` reaches the `usize` frame
    arithmetic unchanged. -/
theorem cast_val (x : U32) : (UScalar.cast .Usize x).val = x.val := by simp

theorem lift_ok {α : Type} (x : α) : lift x = ok x := rfl

/-- A false `!=` comparison is not `true`, the form the extracted body tests. -/
theorem bne_false {x y : U32} (h : (x != y) = false) : ¬ ((x != y) = true) := by
  rw [h]; decide

/-- The frame end the real body computes: `index + WordCount`, at its exact value. -/
theorem frame_end (words : Slice U32) (index : Usize) (wc : U32)
    (hframe : index.val + wc.val ≤ words.length) :
    ∃ e : Usize, (index + UScalar.cast .Usize wc) = ok e ∧ e.val = index.val + wc.val := by
  have hprop : words.length ≤ Usize.max := Slice.property words
  have hbound : index.val + (UScalar.cast .Usize wc).val ≤ Usize.max := by
    rw [cast_val wc]; omega
  obtain ⟨e, he, hev⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := index) (y := UScalar.cast .Usize wc) hbound)
  refine ⟨e, he, ?_⟩
  rw [hev, cast_val wc]

/-- The frame end, under only the `usize` addition bound. -/
theorem frame_end_bound (index : Usize) (wc : U32) (hbound : index.val + wc.val ≤ Usize.max) :
    ∃ e : Usize, (index + UScalar.cast .Usize wc) = ok e ∧ e.val = index.val + wc.val := by
  have hcast : (UScalar.cast .Usize wc).val = wc.val := cast_val wc
  obtain ⟨e, he, hev⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := index) (y := UScalar.cast .Usize wc) (by omega))
  refine ⟨e, he, ?_⟩
  rw [hev, hcast]

/-! ## The D1 word split

`word_count` is the decoded `WordCount` field (`first >>> 16`) and `opcode` the decoded opcode
(`first &&& 0xffff`). -/

/-- The shift is total at the fixed amount 16 and yields the high-half `WordCount` field. -/
theorem word_count_field (first : U32) :
    ∃ wc : U32, first >>> (16#i32) = ok wc ∧ wc.val = first.val / 65536 := by
  obtain ⟨wc, hwv, hwpost⟩ := WP.spec_imp_exists
    (U32.ShiftRight_IScalar_spec first (16#i32) (by norm_num) (by norm_num))
  refine ⟨wc, hwv, ?_⟩
  have h16 : ((16#i32 : IScalar .I32).toNat) = 16 := by rfl
  rw [h16] at hwpost
  omega

/-- The opcode mask yields exactly the low 16 bits of the same instruction header word. -/
theorem opcode_field (first : U32) : (first &&& 65535#u32).val = first.val &&& 65535 := rfl

/-! ## The work measure

The rank of a scan cursor is the number of stream words not yet framed. It is a `Nat`; a scanned
instruction advances the cursor by its decoded `WordCount`, at least one word, so each continuation
lowers the rank by at least one, and the rank is zero exactly at the declared end of the stream. -/

def scanRank (words : Slice U32) (index : Usize) : Nat := words.length - index.val

theorem scanRank_eq (words : Slice U32) (index : Usize) :
    scanRank words index = words.length - index.val := rfl

/-- The rank is exhausted exactly at the declared end of the stream. -/
theorem scanRank_zero_iff (words : Slice U32) (index : Usize) :
    scanRank words index = 0 ↔ words.length ≤ index.val := by
  rw [scanRank_eq, Nat.sub_eq_zero_iff_le]

/-! ## The five header validation transitions

Each is the actual `check_fragment` guard on the actual body, with the slice reads discharged by
the caller's premises and the `!=` tests in the boolean form the body compares. -/

/-- D0: a stream shorter than the five-word header is refused with its own length. -/
theorem header_short_refuses (words : Slice U32) (h : words.length < HEADER_WORDS.val) :
    check_fragment words =
      ok (core.result.Result.Err (CheckError.HeaderTooShort (Slice.len words))) := by
  unfold check_fragment
  rw [if_pos ((UScalar.lt_equiv (Slice.len words) HEADER_WORDS).mpr
    (by simpa [Slice.len] using h))]

/-- `magic`: a first header word whose boolean comparison against the SPIR-V magic number is true is
    refused with that word. -/
theorem bad_magic_refuses (words : Slice U32) (hlen : HEADER_WORDS.val ≤ words.length)
    (w0 : U32) (hw0 : words.index_usize 0#usize = ok w0) (hne : (w0 != MAGIC) = true) :
    check_fragment words = ok (core.result.Result.Err (CheckError.BadMagic w0)) := by
  unfold check_fragment
  rw [if_neg ((UScalar.lt_equiv (Slice.len words) HEADER_WORDS).not.mpr
    (by simpa [Slice.len] using hlen))]
  rw [hw0, bind_tc_ok, if_pos hne]

/-- `version`: a second header word whose boolean comparison against the SPIR-V 1.3 version word is
    true is refused with that word. -/
theorem bad_version_refuses (words : Slice U32) (hlen : HEADER_WORDS.val ≤ words.length)
    (w0 w1 : U32) (hw0 : words.index_usize 0#usize = ok w0)
    (hw1 : words.index_usize 1#usize = ok w1) (hm : (w0 != MAGIC) = false)
    (hne : (w1 != VERSION_1_3) = true) :
    check_fragment words = ok (core.result.Result.Err (CheckError.BadVersion w1)) := by
  unfold check_fragment
  rw [if_neg ((UScalar.lt_equiv (Slice.len words) HEADER_WORDS).not.mpr
    (by simpa [Slice.len] using hlen))]
  rw [hw0, bind_tc_ok, if_neg (bne_false hm), hw1, bind_tc_ok, if_pos hne]

/-- `bound`: a zero id bound is refused. -/
theorem zero_bound_refuses (words : Slice U32) (hlen : HEADER_WORDS.val ≤ words.length)
    (w0 w1 w3 : U32) (hw0 : words.index_usize 0#usize = ok w0)
    (hw1 : words.index_usize 1#usize = ok w1) (hw3 : words.index_usize 3#usize = ok w3)
    (hm : (w0 != MAGIC) = false) (hv : (w1 != VERSION_1_3) = false) (hz : w3 = 0#u32) :
    check_fragment words = ok (core.result.Result.Err CheckError.ZeroBound) := by
  unfold check_fragment
  rw [if_neg ((UScalar.lt_equiv (Slice.len words) HEADER_WORDS).not.mpr
    (by simpa [Slice.len] using hlen))]
  rw [hw0, bind_tc_ok, if_neg (bne_false hm), hw1, bind_tc_ok, if_neg (bne_false hv),
    hw3, bind_tc_ok, if_pos hz]

/-- `schema`: a nonzero schema word is refused with that word. -/
theorem bad_schema_refuses (words : Slice U32) (hlen : HEADER_WORDS.val ≤ words.length)
    (w0 w1 w3 w4 : U32) (hw0 : words.index_usize 0#usize = ok w0)
    (hw1 : words.index_usize 1#usize = ok w1) (hw3 : words.index_usize 3#usize = ok w3)
    (hw4 : words.index_usize 4#usize = ok w4)
    (hm : (w0 != MAGIC) = false) (hv : (w1 != VERSION_1_3) = false) (hb : w3 ≠ 0#u32)
    (hne : (w4 != 0#u32) = true) :
    check_fragment words = ok (core.result.Result.Err (CheckError.BadSchema w4)) := by
  unfold check_fragment
  rw [if_neg ((UScalar.lt_equiv (Slice.len words) HEADER_WORDS).not.mpr
    (by simpa [Slice.len] using hlen))]
  rw [hw0, bind_tc_ok, if_neg (bne_false hm), hw1, bind_tc_ok, if_neg (bne_false hv),
    hw3, bind_tc_ok, if_neg hb, hw4, bind_tc_ok, if_pos hne]

/-- All five header guards passing is the transition into the instruction scan, at the empty
    definition and type tables the real body starts from and at the header cursor. -/
theorem header_ok_enters_scan (words : Slice U32) (hlen : HEADER_WORDS.val ≤ words.length)
    (w0 w1 w3 w4 : U32) (hw0 : words.index_usize 0#usize = ok w0)
    (hw1 : words.index_usize 1#usize = ok w1) (hw3 : words.index_usize 3#usize = ok w3)
    (hw4 : words.index_usize 4#usize = ok w4)
    (hm : (w0 != MAGIC) = false) (hv : (w1 != VERSION_1_3) = false) (hb : w3 ≠ 0#u32)
    (hs : (w4 != 0#u32) = false) :
    check_fragment words =
      check_fragment_loop words (alloc.vec.Vec.new U32)
        (alloc.vec.Vec.new (U32 × TyTag)) HEADER_WORDS := by
  unfold check_fragment
  rw [if_neg ((UScalar.lt_equiv (Slice.len words) HEADER_WORDS).not.mpr
    (by simpa [Slice.len] using hlen))]
  rw [hw0, bind_tc_ok, if_neg (bne_false hm), hw1, bind_tc_ok, if_neg (bne_false hv),
    hw3, bind_tc_ok, if_neg hb, hw4, bind_tc_ok, if_neg (bne_false hs)]

/-! ## The instruction-scan transitions

`check_fragment_loop.body` is the real `while index < words.len()` body. Its exits are the D1
zero-`WordCount` refusal, the D1 truncation refusal, and the loop exit; its continuation advances
the cursor to the instruction's own frame end. -/

/-- The scan's exit branch: a cursor at or past the declared end accepts the stream. This is the
    only transition of the scan that produces `Ok ()`. -/
theorem scan_exit (words : Slice U32) (defs : alloc.vec.Vec U32)
    (tys : alloc.vec.Vec (U32 × TyTag)) (index : Usize)
    (h : words.length ≤ index.val) :
    check_fragment_loop.body words defs tys index = ok (done (core.result.Result.Ok ())) := by
  unfold check_fragment_loop.body
  rw [if_neg ((UScalar.lt_equiv index (Slice.len words)).not.mpr
    (by simpa [Slice.len] using h))]

/-- The scan accepts only at word rank zero: its exit branch is taken exactly when the rank is
    exhausted. -/
theorem scan_exit_rank_zero (words : Slice U32) (defs : alloc.vec.Vec U32)
    (tys : alloc.vec.Vec (U32 × TyTag)) (index : Usize)
    (h : words.length ≤ index.val) :
    check_fragment_loop.body words defs tys index = ok (done (core.result.Result.Ok ())) ∧
      scanRank words index = 0 :=
  ⟨scan_exit words defs tys index h, (scanRank_zero_iff words index).mpr h⟩

/-- D1: a zero `WordCount` leaves the instruction boundary undefined and is refused at its own word
    index, which lies strictly inside the stream. -/
theorem zero_word_count_refuses (words : Slice U32) (defs : alloc.vec.Vec U32)
    (tys : alloc.vec.Vec (U32 × TyTag)) (index : Usize)
    (hlt : index.val < words.length) (first : U32)
    (hw : words.index_usize index = ok first) (hz : first.val / 65536 = 0) :
    check_fragment_loop.body words defs tys index =
      ok (done (core.result.Result.Err (CheckError.ZeroWordCount index))) := by
  unfold check_fragment_loop.body
  rw [if_pos ((UScalar.lt_equiv index (Slice.len words)).mpr
    (by simpa [Slice.len] using hlt))]
  rw [hw, bind_tc_ok]
  obtain ⟨wc, hwc, hwcv⟩ := word_count_field first
  have hzero : wc = 0#u32 := UScalar.eq_of_val_eq (by rw [hwcv]; exact hz)
  simp (config := { zeta := true }) only [hwc, bind_tc_ok, hzero, reduceIte, lift_ok, bind_tc_ok]

/-- D1: a `WordCount` that overruns the remaining words is refused at its own word index, carrying
    the decoded count. -/
theorem truncated_refuses (words : Slice U32) (defs : alloc.vec.Vec U32)
    (tys : alloc.vec.Vec (U32 × TyTag)) (index : Usize)
    (hlt : index.val < words.length) (first : U32) (hw : words.index_usize index = ok first)
    (hadd : index.val + first.val / 65536 ≤ Usize.max)
    (hz : 0 < first.val / 65536)
    (htrunc : words.length < index.val + first.val / 65536) :
    ∃ wc : U32, wc.val = first.val / 65536 ∧ 0 < wc.val ∧
      check_fragment_loop.body words defs tys index =
        ok (done (core.result.Result.Err (CheckError.TruncatedInstruction index wc))) := by
  obtain ⟨wc, hwc, hwcv⟩ := word_count_field first
  refine ⟨wc, hwcv, (by rw [hwcv]; omega), ?_⟩
  have hzero : wc ≠ 0#u32 := by
    intro hc
    have hv : wc.val = 0 := by rw [hc]; rfl
    rw [hwcv] at hv
    omega
  obtain ⟨e, he, hev⟩ := frame_end_bound index wc (by rw [hwcv]; omega)
  have hbody :
      check_fragment_loop.body words defs tys index =
        ok (done (core.result.Result.Err (CheckError.TruncatedInstruction index wc))) := by
    unfold check_fragment_loop.body
    rw [if_pos ((UScalar.lt_equiv index (Slice.len words)).mpr
      (by simpa [Slice.len] using hlt))]
    simp (config := { zeta := true }) only [hw, bind_tc_ok, hwc, bind_tc_ok, hzero, reduceIte,
      lift, bind_tc_ok, he]
    have hgt : e > (Slice.len words) := by
      show words.length < e.val
      rw [hev, hwcv]
      simpa [Slice.len] using htrunc
    rw [if_pos hgt]
  exact hbody

/-- **The scan's continuation transition and its strictly decreasing rank.** Under the D1 framing
    guards — the cursor is in bounds, the decoded `WordCount` is nonzero and its frame end stays
    inside the stream — and with one decoded instruction accepted by the D2-D4 obligations, the
    real body continues at exactly that instruction's frame end, the new cursor stays inside the
    stream, and the word rank strictly decreases by at least one. -/
theorem scan_step (words : Slice U32) (defs : alloc.vec.Vec U32)
    (tys : alloc.vec.Vec (U32 × TyTag)) (index : Usize) (first : U32) (wc : U32)
    («end» i3 : Usize) (ops : Slice U32) (shp : Shape)
    (dd : alloc.vec.Vec U32) (tt : alloc.vec.Vec (U32 × TyTag))
    (hlt : index.val < words.length) (hw : words.index_usize index = ok first)
    (hwc : first >>> (16#i32) = ok wc) (hwcv : wc.val = first.val / 65536)
    (hwc0 : 0 < wc.val) (he : (index + UScalar.cast .Usize wc) = ok «end»)
    (hev : «end».val = index.val + wc.val) (hbnd : «end».val ≤ words.length)
    (hine : ¬ («end» > (Slice.len words)))
    (hi3 : (index + 1#usize) = ok i3)
    (hops : core.slice.index.Slice.index
        (core.slice.index.SliceIndexRangeUsizeSlice Std.U32) words
        { start := i3, «end» := «end» } = ok ops)
    (hs : shape_of index (first &&& 65535#u32) ops = ok (core.result.Result.Ok shp))
    (hap : apply_shape index shp defs tys = ok (core.result.Result.Ok (), dd, tt)) :
    check_fragment_loop.body words defs tys index = ok (cont (dd, tt, «end»)) ∧
      «end».val = index.val + first.val / 65536 ∧
      «end».val ≤ words.length ∧
      scanRank words «end» + 1 ≤ scanRank words index := by
  have hzero : wc ≠ 0#u32 := by
    intro hc
    have hv : wc.val = 0 := by rw [hc]; rfl
    rw [hwcv] at hv
    omega
  have hbody : check_fragment_loop.body words defs tys index = ok (cont (dd, tt, «end»)) := by
    unfold check_fragment_loop.body
    rw [if_pos ((UScalar.lt_equiv index (Slice.len words)).mpr
      (by simpa [Slice.len] using hlt))]
    simp (config := { zeta := true }) only [hw, hwc, bind_tc_ok, lift_ok, bind_tc_ok, hzero, he,
      if_neg hine, reduceIte, bind_tc_ok, hi3, hops, hs, bind_tc_ok,
      core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok, hap, bind_tc_ok,
      core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok, reduceIte]
    simp
  refine ⟨hbody, ?_, ?_, ?_⟩
  · rw [hev, hwcv]
  · exact hbnd
  · rw [scanRank_eq, scanRank_eq, hev]; omega

/-- A loop whose body leaves immediately at `x` returns that value. -/
theorem loop_done {α : Type} {β : Type} (body : α → Result (ControlFlow α β)) (x : α) (y : β)
    (h : body x = ok (ControlFlow.done y)) : loop body x = ok y := by
  rw [loop.eq_1, h]
  show Std.bind (ok (ControlFlow.done y)) (fun r : ControlFlow α β =>
    match r with
    | ControlFlow.cont x => loop body x
    | ControlFlow.done x => ok x) = ok y
  simp

/-! ## Acceptance of a stream with no instruction to scan

With the header read and the cursor already at the declared end, the real loop takes its exit
branch on its first application and the checker accepts. -/

/-- A header-passing stream whose length is exactly the header length is accepted: the scan starts
    at the declared end, so it exits at word rank zero without framing anything. -/
theorem header_only_accepted (words : Slice U32) (hlen : HEADER_WORDS.val ≤ words.length)
    (weq : words.length = HEADER_WORDS.val)
    (w0 w1 w3 w4 : U32) (hw0 : words.index_usize 0#usize = ok w0)
    (hw1 : words.index_usize 1#usize = ok w1) (hw3 : words.index_usize 3#usize = ok w3)
    (hw4 : words.index_usize 4#usize = ok w4)
    (hm : (w0 != MAGIC) = false) (hv : (w1 != VERSION_1_3) = false) (hb : w3 ≠ 0#u32)
    (hs : (w4 != 0#u32) = false) :
    check_fragment words = ok (core.result.Result.Ok ()) := by
  rw [header_ok_enters_scan words hlen w0 w1 w3 w4 hw0 hw1 hw3 hw4 hm hv hb hs]
  exact loop_done _ _ _
    (scan_exit words (alloc.vec.Vec.new U32) (alloc.vec.Vec.new (U32 × TyTag)) HEADER_WORDS
      (by simp [HEADER_WORDS, weq]))

/-- The extracted root forwards to the real body without adding work or changing the result. -/
theorem root_forwards (words : Slice U32)
    (r : core.result.Result Unit CheckError) (h : check_fragment words = ok r) :
    check_fragment_source.word_check_fragment words = ok r := by
  unfold check_fragment_source.word_check_fragment
  rw [h]

/-! ## Reachable witnesses

Both witnesses are concrete word streams fed to the real body, with every header read discharged at
its exact value. -/

/-- The five-word SPIR-V header with no instruction body: `magic`, the 1.3 version word, a generator
    word, a nonzero id bound and a zero schema word. -/
theorem header_words_bound :
    ([119734787#u32, 66304#u32, 0#u32, 1#u32, 0#u32] : List U32).length ≤ Usize.max := by
  have h : ([119734787#u32, 66304#u32, 0#u32, 1#u32, 0#u32] : List U32).length = 5 := by simp
  rw [h]
  exact usize_ge_five

/-- A concrete five-word SPIR-V header stream, built by `Slice.from` under a checked bound. -/
def headerWords : Slice U32 :=
  Slice.from [119734787#u32, 66304#u32, 0#u32, 1#u32, 0#u32] header_words_bound

theorem headerWords_length : headerWords.length = 5 := by simp [headerWords]

theorem headerWords_length_eq : headerWords.length = HEADER_WORDS.val := by
  simp [headerWords_length, HEADER_WORDS]

theorem headerWords_val :
    headerWords.val = [119734787#u32, 66304#u32, 0#u32, 1#u32, 0#u32] := by
  simp [headerWords]

/-- **Reachable success witness.** The concrete five-word header reaches the extracted root and is
    accepted. -/
theorem header_witness : check_fragment_source.word_check_fragment headerWords = ok (core.result.Result.Ok ()) := by
  have hlen : HEADER_WORDS.val ≤ headerWords.length := by
    simp [HEADER_WORDS, headerWords_length]
  have hw0 : headerWords.index_usize 0#usize = ok 119734787#u32 := by
    simp [Slice.index_usize, headerWords]
  have hw1 : headerWords.index_usize 1#usize = ok 66304#u32 := by
    simp [Slice.index_usize, headerWords]
  have hw3 : headerWords.index_usize 3#usize = ok 1#u32 := by
    simp [Slice.index_usize, headerWords]
  have hw4 : headerWords.index_usize 4#usize = ok 0#u32 := by
    simp [Slice.index_usize, headerWords]
  exact root_forwards headerWords (core.result.Result.Ok ())
    (header_only_accepted headerWords hlen headerWords_length_eq _ _ _ _ hw0 hw1 hw3 hw4
      (by simp [MAGIC]) (by simp [VERSION_1_3]) (by simp) (by simp))

/-- **Reachable refusal witness.** The empty word stream is refused by the D0 length guard with its
    own length, through the extracted root. -/
theorem empty_words_bound : ([] : List U32).length ≤ Usize.max := by
  have h : ([] : List U32).length = 0 := by simp
  rw [h]
  exact usize_ge_zero

/-- The empty word stream. -/
def emptyWords : Slice U32 := Slice.from [] empty_words_bound

theorem emptyWords_length : emptyWords.length = 0 := by simp [emptyWords]

theorem empty_refusal_witness :
    check_fragment_source.word_check_fragment emptyWords =
      ok (core.result.Result.Err (CheckError.HeaderTooShort (Slice.len emptyWords))) := by
  have hlt : emptyWords.length < HEADER_WORDS.val := by
    simp [emptyWords_length, HEADER_WORDS]
  rw [root_forwards _ _ (header_short_refuses _ hlt)]

end CheckFragmentResource

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

#resource_audit CheckFragmentResource.header_short_refuses
#resource_audit CheckFragmentResource.bad_magic_refuses
#resource_audit CheckFragmentResource.bad_version_refuses
#resource_audit CheckFragmentResource.zero_bound_refuses
#resource_audit CheckFragmentResource.bad_schema_refuses
#resource_audit CheckFragmentResource.header_ok_enters_scan
#resource_audit CheckFragmentResource.scan_exit
#resource_audit CheckFragmentResource.scan_exit_rank_zero
#resource_audit CheckFragmentResource.zero_word_count_refuses
#resource_audit CheckFragmentResource.truncated_refuses
#resource_audit CheckFragmentResource.scan_step
#resource_audit CheckFragmentResource.header_only_accepted
#resource_audit CheckFragmentResource.root_forwards
#resource_audit CheckFragmentResource.header_witness
#resource_audit CheckFragmentResource.empty_refusal_witness
#resource_audit CheckFragmentResource.usize_ge_five
#resource_audit CheckFragmentResource.scanRank_zero_iff
#resource_audit CheckFragmentResource.word_count_field
#resource_audit CheckFragmentResource.frame_end

run_cmd Lean.logInfo "CHECK_FRAGMENT_KERNEL_OK"