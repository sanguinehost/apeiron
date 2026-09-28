/-
Refinement of the extracted fixed-point LM-head projection.

The Rust body is `src/engine/lm_head.rs`; `i64` Q32.32. The public entry point is
`apeiron_word.word_lm_head_fixed`, the extracted wrapper over `lm_head.lm_head_fixed`
(`proof/ApeironWord.lean:19054`, `:7024`). The `lm_head` leaf is a guard-and-delegate wrapper:

* `Slice.len x != n_embd` returns `Err (ShapeMismatch (Slice.len x) n_embd)` exactly;
* with the widths equal, `quant.block_geom ggml_type = ok none` returns
  `Err (UnsupportedQuant ggml_type)` exactly;
* otherwise it calls `matvec.matvec_fixed ggml_type raw vocab n_embd x` and maps a `MatvecError`
  through `LmHeadError::Matvec`, i.e. the whole projection is the extracted `matvec` with the head
  error wrapper.

This module lands the two guard refusal equalities (`word_lm_head_fixed_shape_mismatch`,
`word_lm_head_fixed_unsupported_quant`), the delegation identity
(`word_lm_head_fixed_unfold`) and its success/refusal projections
(`word_lm_head_fixed_of_matvec_ok`, `word_lm_head_fixed_of_matvec_err`).

## Logit bound

The `vocab` per-token logits are the `matvec` output at `n_out = vocab`, `n_in = n_embd`; the
coordinate bound is the `matvec` refinement statement. `Numeric.EngineMatvecRefinement` states
`matvec_refines` and records `matvec_refines_proved` as an open residual, so the logit bound is
stated here conditionally on `matvec_refines` (`lm_head_refines_of_matvec`); when the `matvec` proof
lands this composes without change. That the Q32.32 output is a winning logit (the tie/argmax
reading) is stated in `Numeric.EngineSamplerRefinement`, not here.

No `sorry`/`admit`/`set_option`/`native_decide`/`maxHeartbeats 0`; the conditional statement claims
nothing for the open `matvec_refines_proved` row.
-/
import Numeric.EngineMatvecRefinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The width guard

The extracted body starts `let i := Slice.len x; if i != n_embd then Err (ShapeMismatch i n_embd)`.
The refusal is the exact `ShapeMismatch` payload with `x`'s length. -/

/-- **Width refusal.** An activation length other than `n_embd` returns
`Err (ShapeMismatch (Slice.len x) n_embd)` exactly (`src/engine/lm_head.rs:120-122`). -/
theorem word_lm_head_fixed_shape_mismatch
    (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64)
    (h : x.val.length ≠ n_embd.val) :
    word_lm_head_fixed ggml_type raw vocab n_embd x =
      ok (core.result.Result.Err
        (lm_head.LmHeadError.ShapeMismatch (Slice.len x) n_embd)) := by
  have hlen : Slice.len x ≠ n_embd := by
    intro hc
    have hv : (Slice.len x).val = n_embd.val := congrArg (fun u : Usize => u.val) hc
    rw [Slice.len_val x] at hv
    exact h hv
  have hcond : (Slice.len x != n_embd) = true := by simpa using hlen
  rw [word_lm_head_fixed, lm_head.lm_head_fixed]
  dsimp only
  rw [if_pos hcond]

/-! ## The quant guard

With the widths equal the body reads `o ← quant.block_geom ggml_type`, tests
`Option.is_none o`, and on `none` returns `Err (UnsupportedQuant ggml_type)`. -/

/-- **Quant refusal.** With `x.len = n_embd` and no fixed-point reader for the family
(`quant.block_geom ggml_type = ok none`), the body returns `Err (UnsupportedQuant ggml_type)`
exactly (`src/engine/lm_head.rs:123-125`). -/
theorem word_lm_head_fixed_unsupported_quant
    (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64)
    (hlen : x.val.length = n_embd.val)
    (h : quant.block_geom ggml_type = ok none) :
    word_lm_head_fixed ggml_type raw vocab n_embd x =
      ok (core.result.Result.Err (lm_head.LmHeadError.UnsupportedQuant ggml_type)) := by
  have hlen' : Slice.len x = n_embd := by
    apply UScalar.eq_of_val_eq
    rw [Slice.len_val x]
    exact hlen
  have hcond : (Slice.len x != n_embd) = false := by simp [hlen']
  rw [word_lm_head_fixed, lm_head.lm_head_fixed]
  dsimp only
  rw [hcond]
  simp only [Bool.false_eq_true, if_false]
  rw [h]
  simp only [bind_tc_ok]
  simp [core.option.Option.is_none]

/-! ## The delegation identity

With both guards passed the head is the extracted `matvec` behind the `LmHeadError::Matvec`
wrapper. -/

/-- **Delegation.** With `x.len = n_embd` and `quant.block_geom ggml_type = ok (some (qk, bb))` the
head body is `word_matvec_fixed` composed with `Result.map_err` through
`LmHeadError::Matvec.constructor`, the extracted delegation `src/engine/lm_head.rs:126`. -/
theorem word_lm_head_fixed_unfold
    (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64) (qk bb : Usize)
    (hlen : x.val.length = n_embd.val)
    (h : quant.block_geom ggml_type = ok (some (qk, bb))) :
    word_lm_head_fixed ggml_type raw vocab n_embd x =
      (do
        let r ← word_matvec_fixed ggml_type raw vocab n_embd x
        core.result.Result.map_err
          P.Insts.CoreOpsFunctionFnOnceTupleMatvecErrorLmHeadError r
          (lm_head.LmHeadError.Matvec.constructor)) := by
  have hlen' : Slice.len x = n_embd := by
    apply UScalar.eq_of_val_eq
    rw [Slice.len_val x]
    exact hlen
  have hcond : (Slice.len x != n_embd) = false := by simp [hlen']
  rw [word_lm_head_fixed, lm_head.lm_head_fixed]
  dsimp only
  rw [hcond]
  simp only [Bool.false_eq_true, if_false]
  rw [h]
  simp only [bind_tc_ok]
  simp only [core.option.Option.is_none, Option.isNone_some, Bool.false_eq_true, if_false]
  rfl

/-- **Success projection.** An `Ok` from the extracted `matvec` is an `Ok` from the head unchanged
(the head error wrapper touches only `Err`). -/
theorem word_lm_head_fixed_of_matvec_ok
    (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64) (qk bb : Usize)
    (out : alloc.vec.Vec I64)
    (hlen : x.val.length = n_embd.val)
    (hgeom : quant.block_geom ggml_type = ok (some (qk, bb)))
    (h : word_matvec_fixed ggml_type raw vocab n_embd x = ok (core.result.Result.Ok out)) :
    word_lm_head_fixed ggml_type raw vocab n_embd x = ok (core.result.Result.Ok out) := by
  rw [word_lm_head_fixed_unfold ggml_type raw vocab n_embd x qk bb hlen hgeom]
  rw [h]
  simp only [bind_tc_ok, core.result.Result.map_err_ok]

/-- **Refusal projection.** A `MatvecError` from the extracted `matvec` is wrapped in
`LmHeadError::Matvec` unchanged. -/
theorem word_lm_head_fixed_of_matvec_err
    (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64) (qk bb : Usize)
    (e : matvec.MatvecError)
    (hlen : x.val.length = n_embd.val)
    (hgeom : quant.block_geom ggml_type = ok (some (qk, bb)))
    (h : word_matvec_fixed ggml_type raw vocab n_embd x = ok (core.result.Result.Err e)) :
    word_lm_head_fixed ggml_type raw vocab n_embd x =
      ok (core.result.Result.Err (lm_head.LmHeadError.Matvec e)) := by
  rw [word_lm_head_fixed_unfold ggml_type raw vocab n_embd x qk bb hlen hgeom]
  rw [h]
  simp only [bind_tc_ok, core.result.Result.map_err_err]
  rw [P.Insts.CoreOpsFunctionFnOnceTupleMatvecErrorLmHeadError.call_once]
  rw [lm_head.LmHeadError.Matvec.constructor]
  simp only [bind_tc_ok]

/-! ## The logit bound, conditional on the matvec refinement

`lm_head_fixed` delegates to `matvec_fixed` over `vocab` rows of `n_embd`. The head's stated domain
is the `matvec_domain` at `n_out = vocab`, `n_in = n_embd`; the coordinate bound is
`matvec_refines`. -/

/-- The target LM-head refinement statement: on the `matvec` domain at `n_out = vocab`,
`n_in = n_embd`, the extracted head returns a length-`vocab` vector whose `r`-th Q32.32 logit is
within `1/(2·q32)` of the real dot of the dequantized row `deq r` with `x`. -/
def lm_head_refines : Prop :=
  ∀ (ggml_type : U32) (raw : Slice U8) (vocab n_embd : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64),
    matvec_domain ggml_type raw vocab n_embd x qk bb blocks row_bytes need deq →
      ∃ out : alloc.vec.Vec I64,
        word_lm_head_fixed ggml_type raw vocab n_embd x = ok (core.result.Result.Ok out) ∧
        out.val.length = vocab.val ∧
        ∀ (r : Nat) (hr : r < out.val.length),
          |toReal (out.val[r]'hr) - rowDotReal (deq r) x| ≤ 1 / (2 * q32)

/-- **LM-head logit bound.** Given the `matvec` refinement (`Numeric.EngineMatvecRefinement`,
`matvec_refines`; the pending `matvec_refines_proved`), the head carries its per-coordinate bound
through the delegation: the width and quant guards are the `matvec_domain` conjuncts, and the head
error wrapper does not touch the `Ok` arm. -/
theorem lm_head_refines_of_matvec (hmat : matvec_refines) : lm_head_refines := by
  intro ggml_type raw vocab n_embd x qk bb blocks row_bytes need deq hdom
  have hgeom : quant.block_geom ggml_type = ok (some (qk, bb)) := hdom.1
  have hxlen : x.val.length = n_embd.val := hdom.2.1
  obtain ⟨out, hout, hlen_out, hbound⟩ :=
    hmat ggml_type raw vocab n_embd x qk bb blocks row_bytes need deq hdom
  refine ⟨out, ?_, hlen_out, hbound⟩
  rw [word_lm_head_fixed_unfold ggml_type raw vocab n_embd x qk bb hxlen hgeom]
  rw [hout]
  simp only [bind_tc_ok, core.result.Result.map_err_ok]


/-- **`lm_head` refinement (proved).** `lm_head_refines_of_matvec` instantiated at the proved row
refinement `matvec_refines_proved`. -/
theorem lm_head_refines_proved : lm_head_refines :=
  lm_head_refines_of_matvec matvec_refines_proved

end

end apeiron_word
