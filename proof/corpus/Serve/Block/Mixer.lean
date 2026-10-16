/-
  `Serve/Mixer.lean` — reference semantics and refinement theorems for the Q32.32 integer
  kernels of `src/engine/mixer.rs`, extracted into `ApeironWord` as group `mixer`.

  Q32.32 is `i64 / 2^32`. The group's value kernels are `value.round_shift_32`,
  `value.mul_q32`, `mixer.div_q32`, `mixer.hist_q32` (the causal window reader),
  `mixer.delta_head_row`, `mixer.l2_normalize`, `mixer.norm_qk_q32`,
  `mixer.head_gates_q32`, `mixer.gate_norm_q32`, `mixer.conv_channel_q32`, and the route
  selector `mixer.chunked_route`.

  ## Boundary — read this before quoting a cone

  Five of the named kernels narrow an `i128` accumulator to `i64` through
  `I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from`, which `ApeironWord.lean:291`
  declares as an **`axiom`** (listed in `proof/boundary-axioms.txt` under "std-model gap
  (shrink-only)": the pinned Aeneas Lean model has no `I128 → I64` `TryFrom`). Measured with
  `#print axioms`, the extracted defs depend on it:

    mixer.delta_head_row     → try_from, alloc.string.String.Insts.CoreConvertFromShared0Str.from
    mixer.div_q32            → try_from, alloc.string.String.Insts.CoreConvertFromShared0Str.from
    mixer.l2_normalize       → try_from, alloc.string.String.Insts.CoreConvertFromShared0Str.from
    mixer.norm_qk_q32        → try_from, alloc.string.String.Insts.CoreConvertFromShared0Str.from
    mixer.gate_norm_q32      → try_from
    mixer.head_gates_q32     → try_from
    mixer.conv_channel_q32   → try_from, core.slice.Slice.copy_within, Range.{start,end}_bound
    mixer.chunked_route      → mixer.exact_scan
    mixer.hist_q32           → (clean)

  Every theorem below whose statement mentions one of those defs therefore carries the same
  boundary in its own cone: `#print axioms` on it lists the axiom, and the cone cannot be
  closed to `{propext, Classical.choice, Quot.sound}` without one of the two discharges
  `boundary-axioms.txt` names — model the operation in the patched Lean backend, or rewrite
  the call out of the extracted closure. Both are outside this file's scope (`src/`,
  `proof/ApeironWord.lean` and the Aeneas model are not owned here). The theorems are
  therefore stated *relative to* the boundary, which is what `boundary-axioms.txt` requires,
  and are labelled accordingly rather than reported as cone-clean.

  Because the axiom is opaque (no `def`, no spec anywhere in the corpus), the bodies that
  call it cannot be shown to return `ok` at all: nothing is known about the narrowing's value.
  `word_delta_head_row_refuses` below takes the one narrowing result it needs as an explicit
  hypothesis for exactly that reason.

  Trust boundary: nothing here covers rustc/LLVM codegen, the `unsafe` container, the AVX2
  twin `mixer.delta_head_step_fast`, the F32 carrier, or the `exact_scan` environment read.
  The statements are about the Aeneas-extracted bodies over the model `Slice`, `Vec` and
  scalar word bounds.
-/

import ApeironWord

set_option autoImplicit false
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow

/-! ## §1 · References

The plain-Lean meaning of each kernel, independent of the extraction. `Int`-level arithmetic is
floor division where the Rust body shifts, and the two index references read a `List` with
`getD`, so the off-the-end case is the identity rather than a partiality. -/

/-- Q32.32 rounding to the nearest multiple of `2^32` with ties up: `round_shift_32 v` is
`(v + 2^31) div 2^32` (the Rust body shifts arithmetically, i.e. floors). -/
def roundShift32Ref (v : Int) : Int := Int.fdiv (v + 2147483648) 4294967296

/-- Q32.32 multiply: narrow the product of the two `i64` words. Total as a mapping; the
narrowing's refusal is the extracted body's, not the reference's. -/
def mulQ32Ref (a b : Int) : Int := roundShift32Ref (a * b)

/-- Q32.32 divide on the admissible domain `0 < den`: `num·2^32 / den` with a half divisor
bias and the integer quotient rounding toward zero, negated for a negative numerator. -/
def divQ32Ref (num den : Int) : Int :=
  if 0 ≤ num then Int.ediv (num * 4294967296 + Int.ediv den 2) den
  else -Int.ediv ((-num) * 4294967296 + Int.ediv den 2) den

/-- `hist_q32`'s reference: row `r` of the `cols`-wide causal window when `r < cols`, otherwise
row `r - cols` of the state window, at column `c`, strided by `ch`. -/
def histRef (window xs : List Std.I64) (cols ch r c : Nat) : Std.I64 :=
  if r < cols then window.getD (r * ch + c) default
  else xs.getD ((r - cols) * ch + c) default

/-- `chunked_route`'s reference, at the `Result` level so the two `ite` trees line up: the route
is taken only for the Q32 carrier, only when the environment scan `mixer.exact_scan` is set, and
only with more than one column; then it is `no_snapshot`, else `false`. -/
def chunkedRouteRef (carrierIsQ32 scan : Bool) (n_cols : Nat) (no_snapshot : Bool) :
    Result Bool :=
  if carrierIsQ32 then
    (if scan then (if 1 < n_cols then ok no_snapshot else ok false) else ok false)
  else ok false

/-! ## §2 · `mixer.hist_q32` — the causal window reader (clean cone)

`hist_q32` reads one word: the row `r` of the `(cols + state)`-row window, at column `c`,
stride `ch`. It has no boundary operation in its body, so its headline's cone closes to the
three standard axioms. Its partiality is index arithmetic: the checked `Usize` multiply and add
must not overflow and the selected index must be inside the chosen slice — the domain is stated
in the hypotheses, exactly as the Rust caller's window geometry supplies it. -/

/-- `n > 1#usize` is the `Nat` comparison of the values. -/
theorem usize_gt_one_iff (n : Std.Usize) : (n > 1#usize) ↔ 1 < n.val := by
  change 1#usize < n ↔ 1 < n.val
  rw [UScalar.lt_equiv]
  norm_num

/-- **`hist_q32` is total on its admitted domain and denotes the reference.** For every window,
state slice, column count, stride and position with the checked index arithmetic in range, the
extracted body returns `ok out` with `out` the reference read. `proved`. -/
theorem word_hist_q32_ok (window xs : Slice Std.I64) (cols ch r c : Std.Usize)
    (hw_mul : r.val * ch.val ≤ Usize.max) (hw_add : r.val * ch.val + c.val ≤ Usize.max)
    (hw_idx : r.val * ch.val + c.val < window.val.length)
    (hx_mul : (r.val - cols.val) * ch.val ≤ Usize.max)
    (hx_add : (r.val - cols.val) * ch.val + c.val ≤ Usize.max)
    (hx_idx : (r.val - cols.val) * ch.val + c.val < xs.val.length) :
    ∃ out : Std.I64, mixer.hist_q32 window xs cols ch r c = ok out ∧
      out = histRef window.val xs.val cols.val ch.val r.val c.val := by
  unfold mixer.hist_q32
  by_cases hrc : r.val < cols.val
  · rw [if_pos (by rw [UScalar.lt_equiv]; exact hrc)]
    obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := r) (y := ch) hw_mul)
    simp only [hi, bind_tc_ok]
    obtain ⟨i1, hi1, hi1v⟩ :=
      WP.spec_imp_exists (Usize.add_spec (x := i) (y := c) (by rw [hiv]; exact hw_add))
    simp only [hi1, bind_tc_ok]
    obtain ⟨out, hout, houtv⟩ :=
      WP.spec_imp_exists (Slice.index_usize_spec window i1 (by rw [hi1v, hiv]; exact hw_idx))
    rw [hout]
    refine ⟨out, rfl, ?_⟩
    have h1 : i1.val < window.val.length := by rw [hi1v, hiv]; exact hw_idx
    have h2 : i1.val = r.val * ch.val + c.val := by rw [hi1v, hiv]
    have houtD : out = window.val.getD i1.val default := by
      rw [houtv, List.getD_eq_getElem (l := window.val) (d := default) (n := i1.val) h1]
    rw [histRef, if_pos hrc, houtD, h2]
  · have hcl : cols.val ≤ r.val := by omega
    rw [if_neg (by intro h; exact hrc ((UScalar.lt_equiv r cols).1 h))]
    obtain ⟨i, hi, hiv, -⟩ :=
      WP.spec_imp_exists (Usize.sub_spec (x := r) (y := cols) hcl)
    simp only [hi, bind_tc_ok]
    obtain ⟨i1, hi1, hi1v⟩ :=
      WP.spec_imp_exists (Usize.mul_spec (x := i) (y := ch) (by rw [hiv]; exact hx_mul))
    simp only [hi1, bind_tc_ok]
    obtain ⟨i2, hi2, hi2v⟩ :=
      WP.spec_imp_exists (Usize.add_spec (x := i1) (y := c) (by rw [hi1v, hiv]; exact hx_add))
    simp only [hi2, bind_tc_ok]
    obtain ⟨out, hout, houtv⟩ :=
      WP.spec_imp_exists
        (Slice.index_usize_spec xs i2 (by rw [hi2v, hi1v, hiv]; exact hx_idx))
    rw [hout]
    refine ⟨out, rfl, ?_⟩
    have h1 : i2.val < xs.val.length := by rw [hi2v, hi1v, hiv]; exact hx_idx
    have h2 : i2.val = (r.val - cols.val) * ch.val + c.val := by rw [hi2v, hi1v, hiv]
    have houtD : out = xs.val.getD i2.val default := by
      rw [houtv, List.getD_eq_getElem (l := xs.val) (d := default) (n := i2.val) h1]
    rw [histRef, if_neg hrc, houtD, h2]

/-! ## §3 · `mixer.div_q32` — the refusing divisor

`div_q32` refuses before it narrows: `den ≤ 0` returns `Err (BadConfig "affine decay factor
c <= 0")`. That branch reaches no boundary operation, so the refusal is provable directly; the
success branch narrows and carries the boundary. -/

/-- **`div_q32` refuses a non-positive divisor.** For every numerator and every `den ≤ 0`, the
extracted body returns `ok (Err (BadConfig msg))` with `msg` the message the boundary `String`
conversion produced. The message itself is the `Str → String` boundary of `boundary-axioms.txt`;
the branch is otherwise unconditional. `proved`. -/
theorem word_div_q32_refuses (num den : Std.I64) (hden : den.val ≤ 0) (msg : String)
    (hmsg : alloc.string.String.Insts.CoreConvertFromShared0Str.from
      (toStr "affine decay factor c <= 0") = ok msg) :
    mixer.div_q32 num den = ok (core.result.Result.Err (step.StepError.BadConfig msg)) := by
  unfold mixer.div_q32
  rw [if_pos (by rw [IScalar.le_equiv]; exact hden)]
  rw [hmsg]
  simp only [bind_tc_ok]

/-! ## §4 · `mixer.chunked_route` — the route selector

The selector reads the carrier equality and the environment scan `mixer.exact_scan` (an `axiom`,
`ApeironWord.lean:47086`, "host boundary" in `boundary-axioms.txt`) and the column count. Both
the carrier test and the scan are taken as their results here: the carrier equality is a
hypothesis, so the theorem holds for every carrier without reducing `read_discriminant`, and the
scan is the `hscan` hypothesis. -/

/-- **`chunked_route` selects per the reference.** For every carrier whose equality against
`Q32` returns `ok carrierIsQ32` and every scan result, the extracted selector returns the
reference Boolean at the `Result` level: the route is on exactly for `Q32`, a set scan and more
than one column. `proved`, relative to `mixer.exact_scan`. -/
theorem word_chunked_route_ok (carrier : mixer.Carrier) (n_cols : Std.Usize)
    (no_snapshot : Bool) (scan carrierIsQ32 : Bool)
    (heq : mixer.Carrier.Insts.CoreCmpPartialEqCarrier.eq carrier mixer.Carrier.Q32
      = ok carrierIsQ32)
    (hscan : mixer.exact_scan = ok scan) :
    mixer.chunked_route carrier n_cols no_snapshot
      = chunkedRouteRef carrierIsQ32 scan n_cols.val no_snapshot := by
  unfold mixer.chunked_route chunkedRouteRef
  rw [heq, hscan]
  simp only [bind_tc_ok]
  by_cases hc : n_cols > 1#usize
  · have hc' : 1 < n_cols.val := (usize_gt_one_iff n_cols).1 hc
    simp only [hc, hc', if_true]
  · have hc' : ¬ (1 < n_cols.val) := fun h => hc ((usize_gt_one_iff n_cols).2 h)
    simp only [hc, hc', if_false]

/-! ## §5 · `mixer.delta_head_row` — the `i128 → i64` refusal

`delta_head_row` narrows two `i128` accumulators through the boundary axiom. Because the axiom is
opaque, the narrowing's value is unknowable; the theorem below *supposes* the first narrowing
refuses and shows the body short-circuits on the `Try` residual. The residual conversion is
`mixer.delta_head_row.closure`, which maps every `TryFromIntError` to `StepError.Overflow`, so the
refusal is the named `Overflow` error and no further arithmetic runs. -/

/-- **`delta_head_row` propagates a refused `i128 → i64` narrowing.** For every input on which the
first narrowing (`value.round_shift_32` then the boundary `try_from`) returns `Err e`, the body
returns `ok (Err StepError.Overflow)`. `proved`, relative to the `try_from` boundary — the two
hypotheses are the narrowing results, which the axiom leaves undetermined. -/
theorem word_delta_head_row_refuses (c_val beta v_j : Std.I64) (acc_m acc_o : Std.I128)
    (k_dot_q : Std.I64) (x : Std.I128) (e : core.num.error.TryFromIntError)
    (hx : value.round_shift_32 acc_m = ok x)
    (hn : I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from x
      = ok (core.result.Result.Err e)) :
    mixer.delta_head_row c_val beta v_j acc_m acc_o k_dot_q
      = ok (core.result.Result.Err step.StepError.Overflow) := by
  unfold mixer.delta_head_row
  rw [hx]
  simp only [bind_tc_ok]
  rw [hn]
  simp only [core.result.Result.map_err_err,
    mixer.delta_head_row.closure.Insts.CoreOpsFunctionFnOnceTupleTryFromIntErrorStepError.call_once,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame, core.convert.FromSame.from, bind_tc_ok]

/-! ## §6 · `mixer.l2_normalize` — the loop-refusal propagation

`l2_normalize` runs a reduction loop, then refuses on a non-positive scale and otherwise runs the
scaling loop. Every refusal path is a `match` on the loop's `Option StepError`; the first is reached
from the loop's `some e` result, before the boundary narrowing, so it is provable from that loop
result alone. -/

/-- **`l2_normalize` propagates a reduction-loop refusal.** If the reduction loop stops with an
error, the body returns that error unchanged: the `some` branch of the `err` match. `proved`,
relative to `try_from` (the def's closure reaches `div_q32`, whose success branch narrows). -/
theorem word_l2_normalize_refuses (x : Slice Std.I64) (eps : Std.I64)
    (e : step.StepError) (sum : Std.I128)
    (hloop : mixer.l2_normalize_loop0 x (none : Option step.StepError) 0#i128 0#usize
      = ok (some e, sum)) :
    mixer.l2_normalize x eps = ok (core.result.Result.Err e) := by
  unfold mixer.l2_normalize
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-! ## §7 · `mixer.norm_qk_q32` — the loop-refusal propagation

`norm_qk_q32` computes `kr0 = kh * hk` (a checked `Usize` multiply) and runs one loop. The error
branch returns the loop's error with the unchanged destination slice. -/

/-- **`norm_qk_q32` propagates a loop refusal.** Given the `Usize` multiply does not overflow and
the loop stops on `some e`, the body returns `ok (Err e, dst1)`. `proved`, relative to `try_from`. -/
theorem word_norm_qk_q32_refuses (fa : mixer.FlatAffine) (key_dim cdim hk : Std.Usize)
    (eps q_scale : Std.I64) (kh : Std.Usize) (dst : Slice Std.I64) (kr0 : Std.Usize)
    (hkr : kh * hk = ok kr0) (dst1 : Slice Std.I64) (e : step.StepError)
    (hloop : mixer.norm_qk_q32_loop0 fa key_dim cdim hk eps q_scale dst kr0
      (none : Option step.StepError) 0#usize = ok (dst1, some e)) :
    mixer.norm_qk_q32 fa key_dim cdim hk eps q_scale kh dst
      = ok (core.result.Result.Err e, dst1) := by
  unfold mixer.norm_qk_q32
  rw [hkr]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-! ## §8 · `mixer.head_gates_q32` — the loop-refusal propagation

`head_gates_q32` runs one loop and returns its error with the unchanged destination slice. -/

/-- **`head_gates_q32` propagates a loop refusal.** Given the loop stops on `some e`, the body
returns `ok (Err e, dst1)`. `proved`, relative to `try_from`. -/
theorem word_head_gates_q32_refuses (fa : mixer.FlatAffine) (p : loader.AffineVectors)
    (nv h : Std.Usize) (dst : Slice Std.I64) (dst1 : Slice Std.I64) (e : step.StepError)
    (hloop : mixer.head_gates_q32_loop fa p nv h dst (none : Option step.StepError) 0#usize
      = ok (dst1, some e)) :
    mixer.head_gates_q32 fa p nv h dst = ok (core.result.Result.Err e, dst1) := by
  unfold mixer.head_gates_q32
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-! ## §9 · `mixer.gate_norm_q32` — the loop-refusal propagation

`gate_norm_q32` divides `head_v` by `slices`, allocates a zero vector, then runs one loop; the
error branch returns the loop's error with the unchanged destination slice. -/

/-- **`gate_norm_q32` propagates a loop refusal.** Given the `Usize` divide and the zero-vector
allocation succeed and the loop stops on `some e`, the body returns `ok (Err e, dst1)`. `proved`,
relative to `try_from` (the def's closure reaches `norm.rms_norm_fixed`). -/
theorem word_gate_norm_q32_refuses (fa : mixer.FlatAffine) (p : loader.AffineVectors)
    (head_v val_dim slices : Std.Usize) (raw : Slice Std.I64) (eps : Std.I64)
    (h : Std.Usize) (dst : Slice Std.I64) (w : Std.Usize) (hw : head_v / slices = ok w)
    (o : alloc.vec.Vec Std.I64)
    (ho : alloc.vec.from_elem core.clone.CloneI64 0#i64 head_v = ok o)
    (dst1 : Slice Std.I64) (e : step.StepError)
    (hloop : mixer.gate_norm_q32_loop0 fa fa.z p head_v val_dim slices raw eps h dst w o
      (none : Option step.StepError) 0#usize = ok (dst1, some e)) :
    mixer.gate_norm_q32 fa p head_v val_dim slices raw eps h dst
      = ok (core.result.Result.Err e, dst1) := by
  unfold mixer.gate_norm_q32
  rw [hw]
  simp only [bind_tc_ok]
  rw [ho]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-! ## §10 · Open — `mixer.conv_channel_q32`

The refusal theorem for `conv_channel_q32` is not proved here, and no green gate is claimed for it.
Its shape is the same loop-refusal pattern as §6–§9: `base = c * kernel` (checked `Usize` multiply);
`conv_channel_q32_loop0` returns `(taps1, hist1)`; `conv_channel_q32_loop1 window xs … taps1 hist1 ys`
returns `(hist2, ys1, err)`; a `match err` returns `ok (Err e, taps1, hist2, ys1)`. The obstacle is
the second rewrite: the extracted body binds the first loop's pair with the names `taps1, hist1`,
which shadow the theorem's hypothesis variables of the same name, so after `rw [hl0]` and
`simp only [bind_tc_ok]` the residual `let (taps1, hist1) := (taps1, hist1)` exposes no occurrence
`rw`/`simp` can match against the loop-1 hypothesis, and it survives `dsimp only` and `beta_reduce`
unchanged. The unrun next step is a case analysis (`cases` on the loop result, `cases` on the pair,
`subst`) that replaces the bound pair by the hypothesis components before `rw [hl1]`. -/

#print axioms word_hist_q32_ok
#print axioms word_div_q32_refuses
#print axioms word_chunked_route_ok
#print axioms word_delta_head_row_refuses
#print axioms word_l2_normalize_refuses
#print axioms word_norm_qk_q32_refuses
#print axioms word_head_gates_q32_refuses
#print axioms word_gate_norm_q32_refuses

end apeiron_word
