import Spirv.Fragment
import Spirv.Decode
import Spirv.Semantics
import ApeironWord

/-!
# Checker soundness bridge: the extracted `check_fragment` against `Spirv.Decode`

The plan (`docs/epics/gpu-spv.md`, W3-T1) states

`checkFragment_sound : checkFragment ws = true → IsDp4aFragment (decode ws).get`.

`Option.get` needs `decode ws ≠ none`, so the well-typed exact equivalent over the extracted
`Result` is the statement this module targets:

`word_check_fragment words = ok (core.result.Result.Ok ()) →
   ∃ m, decode (wordsOfSlice words) = some m ∧ IsDp4aFragment m`

(`word_check_fragment`, `proof/ApeironWord.lean`, the Aeneas extraction of
`src/spirv/check.rs::check_fragment`, `Result (core.result.Result Unit CheckError)`; `decode` is
`Spirv.Decode.decode` over `List U32Word`). The checker hypothesis is load-bearing in that
statement: it is what makes `decode` succeed (`decode_sound` shows every decoded module is
`IsDp4aFragment`, since `inst_opcode_mem_served` holds of every `Inst`, so the checker's burden in
the plan's statement is entirely decoder completeness).

## What is proved here

The header leg of that bridge, and the constant identities it rests on:

* `check_fragment_accepts_header`: acceptance implies the five header obligations the checker
  enforces — length ≥ 5, `words[0] = MAGIC`, `words[1] = VERSION_1_3`, `words[3] ≠ 0` (bound),
  `words[4] = 0` (schema) — read through the extracted `Slice` body itself (`HeaderTooShort`,
  `BadMagic`, `BadVersion`, `ZeroBound`, `BadSchema` refusals);
* `magic_eq`, `version_eq`, `headerWords_val`: the extracted header constants are the constants
  `Spirv.Decode.parseHeader` compares against;
* `parseHeader_shape`: the pure `Spirv.Decode` reading of those obligations;
* `checkFragment_header`: composition — acceptance implies
  `parseHeader (wordsOfSlice words) = some ((wordsOfSlice words).drop 5)`.

`checkFragment_header` is the first component of the exact equivalent above: with the header parsed,
the residual is `parseInsts (wordsOfSlice words).drop 5 ≠ none` and
`wfCheckB WFState.init code = true`. The negative control (drop the `BadMagic` refusal from the
extracted `check_fragment` body in a throwaway copy) turns `check_fragment_accepts_header` red.

## Residual obligation (not proved here)

The exact equivalent needs three more components over the extracted `Vec`/`Slice` scans:

1. **framing + shapes (`shape_of` ↔ `decodeInst`)**: `check_fragment_loop` (fuel-free `loop`,
   measure `words.len - index`) walks the same `WordCount` boundaries as `parseInstsAux`, and for
   each of the 42 admitted opcodes `shape_of`'s `require_exact`/`require_min` arm accepts exactly
   the operand shapes `decodeInst`'s pattern matches accept (`OpEntryPoint` literal terminator ↔
   `takeLiteral`, `OpSUDot` `fmt = 0`, `OpVariable` 3/4 operands);
2. **scan state (`apply_shape` ↔ `stepInst`/`wfCheckB`)**: the `defs : Vec U32` and
   `tys : Vec (U32 × TyTag)` scans maintain `defs.toList = st.defs.reverse` and a `tys` list whose
   `ty_get`/`resolve_tag` lookups agree with `lookupTy`/`resultTag` under D2 id-uniqueness
   (`ty_get` scans oldest-first, `lookupTy` newest-first; unique keys make them coincide), and each
   `apply_shape` refusal (`UndefinedId`, `DuplicateResultId`, `TypeMismatch`, `BadStorageClass`,
   `BadDecoration`) matches the corresponding `WFInst` Boolean conjunct;
3. **composition**: `loop.spec_decr_nat` over `check_fragment_loop` with that invariant, then
   `decode ws = some ⟨code⟩` from `parseHeader` + `parseInsts` + `wfCheckB`, with `IsDp4aFragment`
   discharged by `Spirv.Decode.decode_sound`.

Until (1)–(3) land, `checkFragment_sound` is not proved; nothing here weakens its statement or
adds a hypothesis the checker does not carry.
-/

namespace Spirv

open Aeneas Aeneas.Std Result
open apeiron_word

/-- The pure word stream (`List U32Word`) carried by an extracted `Slice Std.U32`: one `Nat` per
little-endian word, the input type of `Spirv.Decode.decode`.

`→` [takes the left and gives the right]
`↔` [exactly when, true in both directions]
`∧` [both are true]
`¬` [this is not true]
`≠` [is not equal to]
`≤` [is smaller than or equal to]
`∃` [there is at least one]
`List` [a finite sequence of things, in order] -/
def wordsOfSlice (w : Slice Std.U32) : List U32Word := w.val.map (·.val)

/-- In-bounds slice indexing by a `Usize` index reads the underlying list.

`→` [takes the left and gives the right]
`↔` [exactly when, true in both directions]
`∧` [both are true]
`¬` [this is not true]
`≠` [is not equal to]
`≤` [is smaller than or equal to]
`∃` [there is at least one]
`List` [a finite sequence of things, in order] -/
theorem index_usize_iff {α : Type} (v : Slice α) (i : Usize) (x : α) :
    v.index_usize i = ok x ↔ v.val[i.val]? = some x := by
  cases hv : v.val[i.val]? with
  | none =>
      simp [Slice.index_usize, Slice.getElem?_Usize_eq, hv]
  | some y =>
      simp [Slice.index_usize, Slice.getElem?_Usize_eq, hv]

/-- The extracted magic constant is `Spirv.Decode.spirvMagic` (`0x07230203`).

`→` [takes the left and gives the right]
`↔` [exactly when, true in both directions]
`∧` [both are true]
`¬` [this is not true]
`≠` [is not equal to]
`≤` [is smaller than or equal to]
`∃` [there is at least one]
`List` [a finite sequence of things, in order] -/
theorem magic_eq : spirv.check.MAGIC.val = spirvMagic := by
  simp [spirv.check.MAGIC, spirvMagic]

/-- The extracted version constant is `Spirv.Decode.spirvVersion13` (`0x00010300`).

`→` [takes the left and gives the right]
`↔` [exactly when, true in both directions]
`∧` [both are true]
`¬` [this is not true]
`≠` [is not equal to]
`≤` [is smaller than or equal to]
`∃` [there is at least one]
`List` [a finite sequence of things, in order] -/
theorem version_eq : spirv.check.VERSION_1_3.val = spirvVersion13 := by
  simp [spirv.check.VERSION_1_3, spirvVersion13]

/-- The extracted header length is the five-word header `Spirv.Decode.parseHeader` matches.

`→` [takes the left and gives the right]
`↔` [exactly when, true in both directions]
`∧` [both are true]
`¬` [this is not true]
`≠` [is not equal to]
`≤` [is smaller than or equal to]
`∃` [there is at least one]
`List` [a finite sequence of things, in order] -/
theorem headerWords_val : spirv.check.HEADER_WORDS.val = 5 := by
  simp [spirv.check.HEADER_WORDS]

/-- Acceptance of the extracted checker implies its five header obligations, read from the real
extracted body (`check_fragment`'s `HeaderTooShort`/`BadMagic`/`BadVersion`/`ZeroBound`/`BadSchema`
refusals).

`→` [takes the left and gives the right]
`↔` [exactly when, true in both directions]
`∧` [both are true]
`¬` [this is not true]
`≠` [is not equal to]
`≤` [is smaller than or equal to]
`∃` [there is at least one]
`List` [a finite sequence of things, in order] -/
theorem check_fragment_accepts_header {words : Slice Std.U32}
    (h : spirv.check.check_fragment words = ok (core.result.Result.Ok ())) :
    5 ≤ words.val.length ∧
    words.val[0]? = some spirv.check.MAGIC ∧
    words.val[1]? = some spirv.check.VERSION_1_3 ∧
    words.val[3]? ≠ some (0#u32) ∧
    words.val[4]? = some (0#u32) := by
  unfold spirv.check.check_fragment at h
  by_cases hlen : words.val.length < 5
  · rw [if_pos (by simpa [Slice.len_val, spirv.check.HEADER_WORDS] using hlen)] at h
    simp at h
  · rw [if_neg (by simpa [Slice.len_val, spirv.check.HEADER_WORDS] using hlen)] at h
    cases hc0 : words.index_usize 0#usize with
    | ret x0 =>
        rw [hc0] at h
        simp only [bind_tc_ok] at h
        by_cases hm0 : (x0 != spirv.check.MAGIC) = true
        · rw [if_pos hm0] at h
          simp at h
        · rw [if_neg hm0] at h
          have e0 : words.val[0]? = some spirv.check.MAGIC := by
            have hget := (index_usize_iff words 0#usize x0).mp hc0
            have hx : x0 = spirv.check.MAGIC := by
              by_contra hh
              exact hm0 (by simpa [bne_iff_ne] using hh)
            rw [hx] at hget
            exact hget
          cases hc1 : words.index_usize 1#usize with
          | ret x1 =>
              rw [hc1] at h
              simp only [bind_tc_ok] at h
              by_cases hm1 : (x1 != spirv.check.VERSION_1_3) = true
              · rw [if_pos hm1] at h
                simp at h
              · rw [if_neg hm1] at h
                have e1 : words.val[1]? = some spirv.check.VERSION_1_3 := by
                  have hget := (index_usize_iff words 1#usize x1).mp hc1
                  have hx : x1 = spirv.check.VERSION_1_3 := by
                    by_contra hh
                    exact hm1 (by simpa [bne_iff_ne] using hh)
                  rw [hx] at hget
                  exact hget
                cases hc3 : words.index_usize 3#usize with
                | ret x3 =>
                    rw [hc3] at h
                    simp only [bind_tc_ok] at h
                    by_cases hz3 : x3 = 0#u32
                    · rw [if_pos hz3] at h
                      simp at h
                    · rw [if_neg hz3] at h
                      have e3 : words.val[3]? ≠ some (0#u32) := by
                        have hget := (index_usize_iff words 3#usize x3).mp hc3
                        intro hh
                        have key : some x3 = some (0#u32) := by
                          rw [← hget]
                          exact hh
                        exact hz3 (Option.some.inj key)
                      cases hc4 : words.index_usize 4#usize with
                      | ret x4 =>
                          rw [hc4] at h
                          simp only [bind_tc_ok] at h
                          by_cases hs4 : (x4 != 0#u32) = true
                          · rw [if_pos hs4] at h
                            simp at h
                          · rw [if_neg hs4] at h
                            have e4 : words.val[4]? = some (0#u32) := by
                              have hget := (index_usize_iff words 4#usize x4).mp hc4
                              have hx : x4 = 0#u32 := by
                                by_contra hh
                                exact hs4 (by simpa [bne_iff_ne] using hh)
                              rw [hx] at hget
                              exact hget
                            exact ⟨Nat.not_lt.mp hlen, e0, e1, e3, e4⟩
                        | div => rw [hc4] at h; simp at h
                        | vis _ _ => rw [hc4] at h; simp at h
                  | div => rw [hc3] at h; simp at h
                  | vis _ _ => rw [hc3] at h; simp at h
            | div => rw [hc1] at h; simp at h
            | vis _ _ => rw [hc1] at h; simp at h
      | div => rw [hc0] at h; simp at h
      | vis _ _ => rw [hc0] at h; simp at h

/-- The pure `Spirv.Decode` reading of the five header obligations.

`→` [takes the left and gives the right]
`↔` [exactly when, true in both directions]
`∧` [both are true]
`¬` [this is not true]
`≠` [is not equal to]
`≤` [is smaller than or equal to]
`∃` [there is at least one]
`List` [a finite sequence of things, in order] -/
theorem parseHeader_shape {ws : List U32Word}
    (hlen : 5 ≤ ws.length) (h0 : ws[0]? = some spirvMagic) (h1 : ws[1]? = some spirvVersion13)
    (h3 : ws[3]? ≠ some 0) (h4 : ws[4]? = some 0) :
    parseHeader ws = some (ws.drop 5) := by
  rcases ws with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, t⟩⟩⟩⟩⟩
  · simp at hlen
  · simp at hlen
  · simp at hlen
  · simp at hlen
  · simp at hlen
  · have ha : a = spirvMagic := by simpa using h0
    have hb : b = spirvVersion13 := by simpa using h1
    have hd : d ≠ 0 := by
      intro hh
      exact h3 (by simp [hh])
    have he : e = 0 := by simpa using h4
    subst ha; subst hb; subst he
    simp [parseHeader, hd]

/-- **Header leg of `checkFragment_sound`.** Acceptance of the extracted checker implies
`Spirv.Decode.parseHeader` accepts the same word stream and returns the instruction words: the
checker's five header refusals are exactly the decoder's header conditions.

`→` [takes the left and gives the right]
`↔` [exactly when, true in both directions]
`∧` [both are true]
`¬` [this is not true]
`≠` [is not equal to]
`≤` [is smaller than or equal to]
`∃` [there is at least one]
`List` [a finite sequence of things, in order] -/
theorem checkFragment_header {words : Slice Std.U32}
    (h : word_check_fragment words = ok (core.result.Result.Ok ())) :
    ∃ rest, parseHeader (wordsOfSlice words) = some rest ∧
      rest = (wordsOfSlice words).drop 5 := by
  have hc : spirv.check.check_fragment words = ok (core.result.Result.Ok ()) := by
    simpa [word_check_fragment] using h
  obtain ⟨hlen, h0, h1, h3, h4⟩ := check_fragment_accepts_header hc
  refine ⟨(wordsOfSlice words).drop 5, ?_, rfl⟩
  have wlen : 5 ≤ (wordsOfSlice words).length := by
    simpa [wordsOfSlice] using hlen
  have w0 : (wordsOfSlice words)[0]? = some spirvMagic := by
    simp [wordsOfSlice, h0, magic_eq]
  have w1 : (wordsOfSlice words)[1]? = some spirvVersion13 := by
    simp [wordsOfSlice, h1, version_eq]
  have w3 : (wordsOfSlice words)[3]? ≠ some 0 := by
    intro hh
    have hget : (wordsOfSlice words)[3]? = (words.val[3]?).map (·.val) := by
      simp [wordsOfSlice]
    rw [hget] at hh
    cases hv : words.val[3]? with
    | none => simp [hv] at hh
    | some y =>
        simp [hv] at hh
        have hy : y = (0#u32 : U32) := by
          apply UScalar.eq_of_val_eq
          have h0v : (0#u32 : U32).val = 0 := by rfl
          rw [h0v]
          exact hh
        exact h3 (by simp [hv, hy])
  have w4 : (wordsOfSlice words)[4]? = some 0 := by
    simp [wordsOfSlice, h4]
  exact parseHeader_shape wlen w0 w1 w3 w4

end Spirv
