/-
Loop specifications for the attention score and softmax-numerator stages
.

`Numeric.EngineAttnRefinement` states the exact stage specifications (`attnScoreInt`, `scoreList`,
`planeRow`, `row_dot_domain`) and the structural unfold of `word_attn_head`
(`src/engine/attn.rs:1019-1213`); its residual list names the four `attn_head_loop0..3` loop
specifications as unproved. `Numeric.AttnSoftmaxPerturb` proves the real stage composition
(`attn_stage_compose_closed`) against hypotheses shaped exactly as `attnNumeratorCase` and
`∃ j, y j = mx`; this module supplies those hypotheses from the extracted `attn_head_loop0` (score
loop, `src/engine/attn.rs:1085-1120`) and `attn_head_loop1` (numerator loop, `src/engine/attn.rs:
1127-1150`), the latter two inlined into their loop bodies.

## Landed here

* `row_dot_fixed_exact`: the bit-exact value of `matvec.row_dot_fixed`
  (`Numeric.EngineDotRefinement`) on `row_dot_domain` — `divRoundNearest (dotPrefix …) 2^32`, not
  merely the `1/(2·2^32)`-bounded reading `row_dot_refines_proved` exposes. Derived, not restated,
  from the same reused `row_dot_fixed_loop_eq`/`round_shift_32_spec`; the score loop body's own
  epilogue composes two such reductions exactly, so `attnScoreInt` needs the bit-exact value.
* `score_at_spec`: on a plane-row window (`row_dot_domain` of the row `krow` read at
  `first + s·stride = (s·kv+kvh)·hd`, the plane-range guard, and the `attnScoreInt` result inside
  `[-2^63, 2^63-1]`), `attn.KvState.attn_head_loop0.body` returns the continuation carrying `y`
  with `y.val` exactly `attnScoreInt (dot of krow and q) scale.val` — the `s`-th entry of
  `scoreList`.
* `attn_head_loop0_spec`: the score loop's equational invariant (`scores` pointwise equal to
  `scoreList`'s prefix, `mx` the pure running maximum seeded at `I64.MIN`) and, at loop exit,
  `scores` the whole `scoreList`, `mx` its maximum, and `∃ j, scores[j] = mx` — the `hmax` hypothesis
  `attn_stage_compose`/`attn_stage_compose_closed` need.
* `attn_head_loop1_spec`: the numerator loop's invariant (`exps[j]` in the `exp_at` form
  `attnNumeratorCase (scores[j] - mx) exps[j]` for every completed `j`, `total` their checked `i128`
  sum) and, at loop exit, the full `attnNumeratorCase` family, `0 ≤ total` and `total = Σ exps[j]`.
  Entries are read with `getD j 0`, the `ℕ → ℤ` shape `attn_stage_compose_closed` takes.

## Windows *not* implied by `attn_domain_v2` (findings)

`attn_domain_v2` (`Numeric.AttnSoftmaxPerturb`) states `row_dot_domain krow q` only for a `krow`
*already known* to equal `planeRow …`; it does not itself supply:

1. the hoisted `usize` address arithmetic `first + s'·stride` succeeding inside the plane
   (`base ≤ kplane.len() − hd`), or the range index into the plane succeeding with that `krow`
   (`attnHeadLoop0Domain`'s per-position existential) — the "range-index identity tying `planeRow`
   to the plane contents" `EngineAttnRefinement`'s header names as open;
2. the `i64` checked-subtraction `score - mx` succeeding with a nonpositive result at each key
   (`attnHeadLoop1Domain`) — implied by the `attn_domain_v2` score window `|score_j| ≤ 2^62`
   together with `mx` itself being one such score, but not stated as a checked-arithmetic fact;
3. the checked `i128` total staying inside `[I128.min, I128.max]` — not carried by `attn_domain_v2`
   as a checked-arithmetic witness, but derived here: `attnNumeratorCase_le` bounds each numerator
   by `0 ≤ E ≤ 4·2^32` (`sh ≤ 0`, `epsExp < 1`), and `pos + 1 ≤ 2^30` (part of
   `attnHeadLoop1Domain`) keeps the total at most `2^64`.

Windows 1 and 2 are taken as explicit hypotheses (`attnHeadLoop0Domain`, `attnHeadLoop1Domain`),
rather than re-derived from `attn_domain_v2`'s magnitude windows.

## Trust boundary

Reaches the extracted `attn.KvState.attn_head_loop0`, `attn.KvState.attn_head_loop1` and
`matvec.row_dot_fixed` of `proof/ApeironWord.lean`, and the Q32.32 denotation in `Numeric.Scale`.
Does not cover rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V, or the `unsafe` container.
-/
import Numeric.AttnSoftmaxPerturb

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## List-index helper -/

/-- `getD` at an in-range index is the checked `getElem`. -/
private theorem getD_eq_get {l : List I64} {n : ℕ} (h : n < l.length) :
    l.getD n 0#i64 = l[n]'h := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]


/-! ## A pure running maximum over `ℤ`, seeded at a fixed value -/

/-- The pure running maximum `attn_head_loop0` computes (`src/engine/attn.rs:1104-1107`): a left
fold of `max` seeded at `I64.MIN`'s denotation. -/
def intListMax (seed : ℤ) (l : List ℤ) : ℤ := l.foldl max seed

@[simp] theorem intListMax_nil (seed : ℤ) : intListMax seed [] = seed := rfl

theorem intListMax_cons (seed a : ℤ) (l : List ℤ) :
    intListMax seed (a :: l) = intListMax (max seed a) l := rfl

/-- The running maximum never drops below the seed. -/
theorem intListMax_ge_seed (seed : ℤ) (l : List ℤ) : seed ≤ intListMax seed l := by
  induction l generalizing seed with
  | nil => exact le_refl seed
  | cons a l ih =>
    rw [intListMax_cons]
    exact le_trans (le_max_left seed a) (ih (max seed a))

/-- The running maximum dominates every member of the list. -/
theorem intListMax_ge_mem (seed : ℤ) (l : List ℤ) (x : ℤ) (hx : x ∈ l) :
    x ≤ intListMax seed l := by
  induction l generalizing seed with
  | nil => cases hx
  | cons a l ih =>
    rw [intListMax_cons]
    simp only [List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact le_trans (le_max_right seed x) (intListMax_ge_seed (max seed x) l)
    · exact ih (max seed a) hx

/-- The running maximum is either the seed or one of the list's own members. -/
theorem intListMax_mem_or_eq_seed (seed : ℤ) (l : List ℤ) :
    intListMax seed l = seed ∨ intListMax seed l ∈ l := by
  induction l generalizing seed with
  | nil => exact Or.inl rfl
  | cons a l ih =>
    rw [intListMax_cons]
    rcases le_total a seed with hle | hle
    · have hm : max seed a = seed := max_eq_left hle
      rw [hm]
      rcases ih seed with h | h
      · exact Or.inl h
      · exact Or.inr (List.mem_cons_of_mem a h)
    · have hm : max seed a = a := max_eq_right hle
      rw [hm]
      rcases ih a with h | h
      · exact Or.inr (by rw [h]; exact List.mem_cons_self)
      · exact Or.inr (List.mem_cons_of_mem a h)

/-- `intListMax` over an appended singleton: the fold step. -/
theorem intListMax_append_singleton (seed : ℤ) (l : List ℤ) (x : ℤ) :
    intListMax seed (l ++ [x]) = max (intListMax seed l) x := by
  unfold intListMax
  rw [List.foldl_append]
  simp only [List.foldl_cons, List.foldl_nil]

/-! ## Exact row-dot value on `row_dot_domain` -/

/-- **Exact row-dot value.** On `row_dot_domain`, `matvec.row_dot_fixed`
(`src/engine/matvec.rs`, refined by `Numeric.EngineDotRefinement`) returns the exact half-biased
value `divRoundNearest (dotPrefix w_row x len) 2^32`, narrowed to `i64` without change (the
narrowing is exact because `row_dot_domain`'s own output window keeps it inside `i64`). This is the
exact form behind `row_dot_refines_proved`'s `1/(2·2^32)` bound; it is derived here — not restated —
from the same reused `row_dot_fixed_loop_eq`/`round_shift_32_spec`, since `score_at_spec` composes
the bit-exact value into `attnScoreInt`, not the bound. -/
theorem row_dot_fixed_exact (w_row x : Slice I64) (hdom : row_dot_domain w_row x) :
    ∃ y : I64, matvec.row_dot_fixed w_row x = ok (core.result.Result.Ok y) ∧
      y.val = divRoundNearest (dotPrefix w_row x w_row.val.length) ((2 : ℤ) ^ (32 : ℕ)) := by
  obtain ⟨hlen, hfit, houtlo, houthi⟩ := hdom
  have hxlen : x.val.length = w_row.val.length := hlen.symm
  obtain ⟨S, hSval, hloop⟩ := row_dot_fixed_loop_eq w_row x hxlen hfit
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by
    norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
    norm_num [I128.max, IScalar.max, I128.numBits]
  set yv : ℤ := divRoundNearest S.val ((2 : ℤ) ^ (32 : ℕ)) with hyvdef
  have hyvb : -((2 : ℤ) ^ (63 : ℕ)) ≤ yv ∧ yv ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [hyvdef, hSval]; exact ⟨houtlo, houthi⟩
  have hZ := div_round_nearest_abs_bound S.val ((2 : ℤ) ^ (32 : ℕ)) (by norm_num)
  rw [show (2 : ℤ) ^ (32 : ℕ) / 2 = (2 : ℤ) ^ (31 : ℕ) from by norm_num] at hZ
  have hZ' : |yv * (2 : ℤ) ^ (32 : ℕ) - S.val| ≤ (2 : ℤ) ^ (31 : ℕ) := by
    rw [hyvdef]; exact hZ
  have hSbound : |S.val| ≤ (2 : ℤ) ^ (95 : ℕ) + (2 : ℤ) ^ (31 : ℕ) := by
    have hyabs : |yv| ≤ (2 : ℤ) ^ (63 : ℕ) := by rw [abs_le]; omega
    have h1 : |yv * (2 : ℤ) ^ (32 : ℕ)| ≤ (2 : ℤ) ^ (63 : ℕ) * (2 : ℤ) ^ (32 : ℕ) := by
      rw [abs_mul, abs_of_pos (by norm_num : (0 : ℤ) < (2 : ℤ) ^ (32 : ℕ))]
      exact mul_le_mul_of_nonneg_right hyabs (by norm_num)
    have h2 : S.val = yv * (2 : ℤ) ^ (32 : ℕ) -
        (yv * (2 : ℤ) ^ (32 : ℕ) - S.val) := by ring
    rw [h2]
    calc |yv * (2 : ℤ) ^ (32 : ℕ) - (yv * (2 : ℤ) ^ (32 : ℕ) - S.val)|
        ≤ |yv * (2 : ℤ) ^ (32 : ℕ)| + |yv * (2 : ℤ) ^ (32 : ℕ) - S.val| := abs_sub _ _
      _ ≤ (2 : ℤ) ^ (63 : ℕ) * (2 : ℤ) ^ (32 : ℕ) + (2 : ℤ) ^ (31 : ℕ) := add_le_add h1 hZ'
      _ = (2 : ℤ) ^ (95 : ℕ) + (2 : ℤ) ^ (31 : ℕ) := by norm_num
  have hlo31 : I128.min ≤ S.val + 2 ^ (31 : ℕ) := by
    rw [hI128min]; have h := (abs_le.mp hSbound).1; linarith
  have hhi31 : S.val + 2 ^ (31 : ℕ) ≤ I128.max := by
    rw [hI128max]; have h := (abs_le.mp hSbound).2; linarith
  obtain ⟨rounded, hround_eq, hroundval⟩ := round_shift_32_spec S hlo31 hhi31
  have hroundb : -((2 : ℤ) ^ (63 : ℕ)) ≤ rounded.val ∧ rounded.val ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [hroundval, hSval]; exact ⟨houtlo, houthi⟩
  have hmaxv : (IScalar.cast .I128 core.num.I64.MAX).val = (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MAX.val = (2 : ℤ) ^ (63 : ℕ) - 1 := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 ((2 : ℤ) ^ (63 : ℕ) - 1)
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hminv : (IScalar.cast .I128 core.num.I64.MIN).val = -(2 : ℤ) ^ (63 : ℕ) := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MIN.val = -(2 : ℤ) ^ (63 : ℕ) := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 (-(2 : ℤ) ^ (63 : ℕ))
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hmax : ¬ (rounded > IScalar.cast .I128 core.num.I64.MAX) := by
    intro hc; rw [gt_iff_lt, IScalar.lt_equiv, hmaxv] at hc; omega
  have hmin : ¬ (rounded < IScalar.cast .I128 core.num.I64.MIN) := by
    intro hc; rw [IScalar.lt_equiv, hminv] at hc; omega
  have hcastval : (IScalar.cast .I64 rounded).val = rounded.val := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 rounded.val
      (by norm_num [IScalarTy.numBits]; omega) (by norm_num [IScalarTy.numBits]; omega)
  refine ⟨IScalar.cast .I64 rounded, ?_, ?_⟩
  · rw [show matvec.row_dot_fixed w_row x = word_row_dot_fixed w_row x from rfl]
    rw [word_row_dot_fixed_unfold w_row x S hlen hloop]
    rw [hround_eq]
    simp only [lift, bind_tc_ok]
    rw [if_neg hmax]
    rw [if_neg hmin]
  · rw [hcastval, hroundval, hSval]

/-! ## The score loop body: exact evaluation on the plane-row window -/

set_option linter.unusedSimpArgs false in
/-- **Exact single-key score.** On the plane-row window — the hoisted address arithmetic
`first + s'·stride` succeeding inside the plane (`base ≤ kplane.len() − hd`, the one guard the body
keeps per key, `src/engine/attn.rs:1091-1092`), the range index into `kplane` succeeding with
`krow`, `row_dot_domain krow q`, the `push` capacity, and the `attnScoreInt` result inside
`[-2^63, 2^63-1]` (the `i64` narrowing window) — the extracted score-loop body
`attn.KvState.attn_head_loop0.body` (`src/engine/attn.rs:1090-1120`) returns
`cont (scores ++ [y], mx', s + 1, true, none)`, where `mx'` is `y` or the incoming `mx`, with
`y.val` exactly `attnScoreInt` of the real dot of `krow` and `q` against `scale` — the `s`-th entry
of `scoreList` once `krow.val = planeRow …`. The pre-hoist extraction reached the same value
through the separate `attn.score_at` helper (`src/engine/attn.rs:1071-1088` in the re-derived
source, `101-123` before); the loop body now inlines it, so the window is stated against the body.
-/
theorem score_at_spec (hd : Usize) (scale : I64) (stride first pos : Usize)
    (kplane : alloc.vec.Vec I64) (q : Slice I64) (scores : alloc.vec.Vec I64) (mx : I64)
    (s : Usize) (hlen : scores.val.length < Usize.max)
    (hsincr : s.val + 1 ≤ Usize.max) (hs : s.val ≤ pos.val)
    (i1 base val3 : Usize)
    (hsmul : s * stride = ok i1)
    (haddfirst : first + i1 = ok base)
    (hnotgt : ¬ (base > core.num.Usize.saturating_sub (alloc.vec.Vec.len kplane) hd))
    (haddhd : base + hd = ok val3)
    (krow : Slice I64)
    (hindex : alloc.vec.Vec.index (core.slice.index.SliceIndexRangeUsizeSlice Std.I64) kplane
      { start := base, «end» := val3 } = ok krow)
    (hdom : row_dot_domain krow q)
    (houtlo : -(2 : ℤ) ^ (63 : ℕ) ≤ attnScoreInt (dotPrefix krow q krow.val.length) scale.val)
    (houthi : attnScoreInt (dotPrefix krow q krow.val.length) scale.val
        ≤ (2 : ℤ) ^ (63 : ℕ) - 1) :
    ∃ (y : I64) (scores2 : alloc.vec.Vec I64) (s2 : Usize),
      scores2.val = scores.val ++ [y] ∧
      s2.val = s.val + 1 ∧
      y.val = attnScoreInt (dotPrefix krow q krow.val.length) scale.val ∧
      (y > mx →
        attn.KvState.attn_head_loop0.body hd scale q pos kplane stride first scores mx s true none
          = ok (ControlFlow.cont (scores2, y, s2, true, none))) ∧
      (¬ y > mx →
        attn.KvState.attn_head_loop0.body hd scale q pos kplane stride first scores mx s true none
          = ok (ControlFlow.cont (scores2, mx, s2, true, none))) := by
  obtain ⟨raw, hraw_eq, hraw_val⟩ := row_dot_fixed_exact krow q hdom
  have hi : (core.convert.num.FromI128I64.from raw).val = raw.val :=
    core.convert.num.FromI128I64.from_val_eq raw
  have hi1 : (core.convert.num.FromI128I64.from scale).val = scale.val :=
    core.convert.num.FromI128I64.from_val_eq scale
  obtain ⟨hfitlo, hfithi⟩ := score_epilogue_mul_fits raw scale
  obtain ⟨num, hnum_eq, hnumv⟩ := I128_checked_mul_some
    (core.convert.num.FromI128I64.from raw) (core.convert.num.FromI128I64.from scale)
    hfitlo hfithi
  have hnumv' : num.val = raw.val * scale.val := by rw [hnumv, hi, hi1]
  have hrawB := raw.hBounds
  have hscaleB := scale.hBounds
  rw [IScalarTy.I64_numBits_eq] at hrawB hscaleB
  norm_num at hrawB hscaleB
  have habs : |raw.val * scale.val| ≤ (2 : ℤ) ^ (126 : ℕ) := by
    rw [abs_mul]
    have h1 : |raw.val| ≤ (2 : ℤ) ^ (63 : ℕ) :=
      abs_le.mpr ⟨by linarith [hrawB.1], by linarith [hrawB.2]⟩
    have h2 : |scale.val| ≤ (2 : ℤ) ^ (63 : ℕ) :=
      abs_le.mpr ⟨by linarith [hscaleB.1], by linarith [hscaleB.2]⟩
    calc |raw.val| * |scale.val| ≤ (2 : ℤ) ^ (63 : ℕ) * (2 : ℤ) ^ (63 : ℕ) :=
          mul_le_mul h1 h2 (abs_nonneg _) (by positivity)
      _ = (2 : ℤ) ^ (126 : ℕ) := by norm_num
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by
    norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
    norm_num [I128.max, IScalar.max, I128.numBits]
  have hlo31 : I128.min ≤ num.val + 2 ^ (31 : ℕ) := by
    rw [hI128min, hnumv']; have h := (abs_le.mp habs).1; linarith
  have hhi31 : num.val + 2 ^ (31 : ℕ) ≤ I128.max := by
    rw [hI128max, hnumv']; have h := (abs_le.mp habs).2; linarith
  obtain ⟨rounded, hround_eq, hroundval⟩ := round_shift_32_spec num hlo31 hhi31
  have hroundval' : rounded.val = attnScoreInt (dotPrefix krow q krow.val.length) scale.val := by
    rw [hroundval, hnumv', hraw_val]
    rfl
  have hmaxv : (core.convert.num.FromI128I64.from core.num.I64.MAX).val
      = (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    have h : core.num.I64.MAX.val = (2 : ℤ) ^ (63 : ℕ) - 1 := by decide
    rw [h]
  have hminv : (core.convert.num.FromI128I64.from core.num.I64.MIN).val
      = -((2 : ℤ) ^ (63 : ℕ)) := by
    rw [core.convert.num.FromI128I64.from_val_eq]
    have h : core.num.I64.MIN.val = -((2 : ℤ) ^ (63 : ℕ)) := by decide
    rw [h]
  have hmax : ¬ (rounded > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    intro hc; rw [gt_iff_lt, IScalar.lt_equiv, hmaxv, hroundval'] at hc; linarith [houthi]
  have hmin : ¬ (rounded < core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    intro hc; rw [IScalar.lt_equiv, hminv, hroundval'] at hc; linarith [houtlo]
  have hcastval : (IScalar.cast .I64 rounded).val = rounded.val := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 rounded.val
      (by norm_num [IScalarTy.numBits]; rw [hroundval']; omega)
      (by norm_num [IScalarTy.numBits]; rw [hroundval']; omega)
  obtain ⟨scores2, hpush, hpushv⟩ :=
    WP.spec_imp_exists (alloc.vec.Vec.push_spec scores (IScalar.cast .I64 rounded) hlen)
  have hsle : s ≤ pos := (UScalar.le_equiv s pos).2 hs
  have h1 : (1#usize).val = 1 := by norm_num
  have hlemax : s.val + (1#usize).val ≤ Usize.max := by rw [h1]; exact hsincr
  obtain ⟨s2, hadd1, hs2⟩ := WP.spec_imp_exists (Usize.add_spec (x := s) (y := 1#usize) hlemax)
  have hs2' : s2.val = s.val + 1 := by rw [hs2, h1]
  refine ⟨(IScalar.cast .I64 rounded : I64), scores2, s2, hpushv, hs2',
    hcastval.trans hroundval', ⟨?_, ?_⟩⟩
  · intro hgt
    rw [attn.KvState.attn_head_loop0.body, if_pos hsle]
    try dsimp only
    rw [hsmul]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [haddfirst]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [if_neg hnotgt]
    rw [haddhd]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hindex]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hraw_eq]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hnum_eq]
    try dsimp only
    rw [hround_eq]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [if_neg hmax]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [if_neg hmin]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [if_pos hgt]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hpush]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hadd1]
    try simp only [lift, bind_tc_ok]
    try simp only [lift, bind_tc_ok]
    dsimp
    rfl
  · intro hgt
    rw [attn.KvState.attn_head_loop0.body, if_pos hsle]
    try dsimp only
    rw [hsmul]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [haddfirst]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [if_neg hnotgt]
    rw [haddhd]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hindex]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hraw_eq]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hnum_eq]
    try dsimp only
    rw [hround_eq]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [if_neg hmax]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [if_neg hmin]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [if_neg hgt]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hpush]
    try simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hadd1]
    try simp only [lift, bind_tc_ok]
    try simp only [lift, bind_tc_ok]
    dsimp
    rfl

/-! ## Loop 0: scores and the running maximum -/

/-- **`scoreList` entry, via a plane-row window.** The `s`-th entry of `scoreList` is exactly
`attnScoreInt` of the real dot of any `krow` equal to `planeRow` at that row against `q`, once the
two rows agree in length (the equal-length component of `row_dot_domain`). Bridges
`attnHeadLoop0Domain`'s per-position window (stated via `dotPrefix` on a `Slice`) to `scoreList`
(stated over plain `List I64`s). -/
theorem scoreList_getD_eq (plane : List I64) (q : Slice I64) (kv hd kvh : ℕ) (scale : ℤ)
    (pos s : ℕ) (hs : s ≤ pos) (krow : Slice I64)
    (hkrow : krow.val = planeRow plane kv hd kvh s) (hlen : krow.val.length = q.val.length) :
    (scoreList plane q.val kv hd kvh scale pos).getD s 0 =
      attnScoreInt (dotPrefix krow q krow.val.length) scale := by
  have hidx : s < (scoreList plane q.val kv hd kvh scale pos).length := by
    unfold scoreList; simp only [List.length_map, List.length_range]; omega
  rw [List.getD_eq_getElem _ _ hidx]
  unfold scoreList
  simp only [List.getElem_map, List.getElem_range]
  rw [dotPrefix_full krow q hlen, hkrow]

/-- The per-position window `attn.KvState.attn_head_loop0` needs at row `s'`: the hoisted address
arithmetic `first + s'·stride` succeeds inside the plane, the range index into `kplane` succeeds
with a `krow` equal to `planeRow`, that row is on `row_dot_domain` against `q`, and the resulting
`attnScoreInt` fits `i64`. Not implied by `attn_domain_v2` as a checked-arithmetic fact (finding 1
above); supplied explicitly here. `stride = kv·hd` and `first = kvh·hd` are the hoisted
`src/engine/attn.rs:1079-1080` factors, so `first + s'·stride = (s'·kv + kvh)·hd` and the window
is the same `planeRow` the pre-hoist per-key chain read. -/
def attnHeadLoop0Domain (hd : Usize) (scale : I64) (kv kvh stride first pos : Usize)
    (kplane : alloc.vec.Vec I64) (q : Slice I64) : Prop :=
  stride.val = kv.val * hd.val ∧ first.val = kvh.val * hd.val ∧
  ∀ s' : Usize, s'.val ≤ pos.val →
    ∃ (i1 base val3 : Usize) (krow : Slice I64),
      s' * stride = ok i1 ∧
      first + i1 = ok base ∧
      ¬ (base > core.num.Usize.saturating_sub (alloc.vec.Vec.len kplane) hd) ∧
      base + hd = ok val3 ∧
      alloc.vec.Vec.index (core.slice.index.SliceIndexRangeUsizeSlice Std.I64) kplane
        { start := base, «end» := val3 } = ok krow ∧
      krow.val = planeRow kplane.val kv.val hd.val kvh.val s'.val ∧
      row_dot_domain krow q ∧
      -(2 : ℤ) ^ (62 : ℕ) ≤ attnScoreInt (dotPrefix krow q krow.val.length) scale.val ∧
      attnScoreInt (dotPrefix krow q krow.val.length) scale.val ≤ (2 : ℤ) ^ (62 : ℕ)

/-- The score-loop invariant after `s` completed rows: the failure flag is clear, `scores` holds
exactly `s` entries pointwise equal to `scoreList`'s corresponding entries, and `mx` is the pure
running maximum of the executed scores seeded at `I64.MIN`. -/
def attnHeadLoop0Inv (hd : Usize) (scale : I64) (kv kvh pos : Usize)
    (kplane : alloc.vec.Vec I64) (q : Slice I64)
    (scores : alloc.vec.Vec I64) (mx : I64) (s : Usize) (ok1 : Bool)
    (err : Option attn.AttnError) : Prop :=
  ok1 = true ∧ err = none ∧ s.val ≤ pos.val + 1 ∧ scores.val.length = s.val ∧
  (∀ (j : ℕ) (_hj : j < scores.val.length),
    (scores.val.getD j 0#i64).val =
      (scoreList kplane.val q.val kv.val hd.val kvh.val scale.val pos.val).getD j 0) ∧
  mx.val = intListMax core.num.I64.MIN.val (scores.val.map (fun x => x.val))

/-- One score-loop body step: on the domain and the invariant, with `s.val ≤ pos.val` (the loop's
own guard `s ≤ pos` still true), the body reads the plane row at `first + s·stride`, evaluates the
score epilogue, updates the running maximum, pushes the score, and continues at `s + 1` with the
invariant restored. -/
theorem attn_head_loop0_body_step (hd : Usize) (scale : I64) (kv kvh stride first pos : Usize)
    (kplane : alloc.vec.Vec I64) (q : Slice I64)
    (hslots : pos.val + 1 ≤ Usize.max)
    (scores : alloc.vec.Vec I64) (mx : I64) (s : Usize)
    (hdom : attnHeadLoop0Domain hd scale kv kvh stride first pos kplane q)
    (hs : s.val ≤ pos.val)
    (hinv : attnHeadLoop0Inv hd scale kv kvh pos kplane q scores mx s true none) :
    WP.spec
      (attn.KvState.attn_head_loop0.body hd scale q pos kplane stride first scores mx s true none)
      (fun cf => ∃ scores2 mx2 s2,
        cf = ControlFlow.cont (scores2, mx2, s2, true, none) ∧
        attnHeadLoop0Inv hd scale kv kvh pos kplane q scores2 mx2 s2 true none ∧
        s2.val = s.val + 1) := by
  obtain ⟨_hok, _herr, _hle, hlen, hpt, hmxval⟩ := hinv
  obtain ⟨hstride, hfirst, hper⟩ := hdom
  obtain ⟨i1, base, val3, krow, hsmul, haddfirst, hnotgt, haddhd, hindex, hkrow, hdotdom,
    houtlo, houthi⟩ := hper s hs
  have hkl : krow.val.length = q.val.length := hdotdom.1
  have houtlo' : -(2 : ℤ) ^ (63 : ℕ) ≤ attnScoreInt (dotPrefix krow q krow.val.length) scale.val := by
    have hpow : (2 : ℤ) ^ (62 : ℕ) ≤ (2 : ℤ) ^ (63 : ℕ) := by norm_num
    linarith
  have houthi' : attnScoreInt (dotPrefix krow q krow.val.length) scale.val
      ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by
    have hpow : (2 : ℤ) ^ (62 : ℕ) ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by norm_num
    linarith
  have hpushmax : scores.val.length < Usize.max := by rw [hlen]; omega
  obtain ⟨y, scores2, s2, hpushv, hs2', hy_val, hbody⟩ :=
    score_at_spec hd scale stride first pos kplane q scores mx s hpushmax (by omega) hs
      i1 base val3 hsmul haddfirst hnotgt haddhd krow hindex hdotdom houtlo' houthi'
  have hyS : y.val =
      (scoreList kplane.val q.val kv.val hd.val kvh.val scale.val pos.val).getD s.val 0 := by
    rw [hy_val]
    exact (scoreList_getD_eq kplane.val q kv.val hd.val kvh.val scale.val pos.val s.val hs
      krow hkrow hkl).symm
  have hlen2 : scores2.val.length = s2.val := by rw [hpushv, hs2']; simp [hlen]
  have hpt2 : ∀ (j : ℕ) (_hj : j < scores2.val.length),
      (scores2.val.getD j 0#i64).val =
        (scoreList kplane.val q.val kv.val hd.val kvh.val scale.val pos.val).getD j 0 := by
    intro j hj
    rw [hpushv] at hj ⊢
    by_cases hjlt : j < scores.val.length
    · rw [List.getD_append scores.val [y] 0#i64 j hjlt]; exact hpt j hjlt
    · have hjeq : j = scores.val.length := by
        simp only [List.length_append, List.length_singleton] at hj
        omega
      subst hjeq
      rw [List.getD_append_right scores.val [y] 0#i64 scores.val.length (le_refl _)]
      simp only [Nat.sub_self, List.getD_cons_zero]
      rw [hyS, hlen]
  have hmap2 : scores2.val.map (fun x => x.val) = scores.val.map (fun x => x.val) ++ [y.val] := by
    rw [hpushv]; simp
  rw [WP.spec_equiv_exists]
  by_cases hgt : y > mx
  · have hgtv : mx.val < y.val := (IScalar.lt_equiv mx y).1 hgt
    refine ⟨ControlFlow.cont (scores2, y, s2, true, none), hbody.1 hgt,
      scores2, y, s2, rfl,
      ⟨rfl, rfl, by omega, hlen2, hpt2, ?_⟩, hs2'⟩
    rw [hmap2, intListMax_append_singleton, ← hmxval]
    exact (max_eq_right hgtv.le).symm
  · have hgev : y.val ≤ mx.val := by
      by_contra hc
      exact hgt ((IScalar.lt_equiv mx y).2 (by omega))
    refine ⟨ControlFlow.cont (scores2, mx, s2, true, none), hbody.2 hgt,
      scores2, mx, s2, rfl,
      ⟨rfl, rfl, by omega, hlen2, hpt2, ?_⟩, hs2'⟩
    rw [hmap2, intListMax_append_singleton, ← hmxval]
    exact (max_eq_left hgev).symm

/-- **Score-loop specification.** On `attnHeadLoop0Domain`, the extracted `attn_head_loop0`
starting from an empty `scores` and `mx = I64.MIN` runs to completion (`err = none`) with `scores`
the whole `scoreList` of the causal prefix, `mx` its exact maximum, and that maximum attained by
some entry of `scores` — the `hmax` hypothesis `attn_stage_compose`/`attn_stage_compose_closed`
need. -/
theorem attn_head_loop0_spec (hd : Usize) (scale : I64) (kv kvh stride first pos : Usize)
    (kplane : alloc.vec.Vec I64) (q : Slice I64)
    (hslots : pos.val + 1 ≤ Usize.max)
    (hdom : attnHeadLoop0Domain hd scale kv kvh stride first pos kplane q)
    (scores0 : alloc.vec.Vec I64) (hscores0 : scores0.val = []) :
    WP.spec (attn.KvState.attn_head_loop0 hd scale q pos kplane stride first scores0
        core.num.I64.MIN 0#usize true none)
      (fun b => ∃ scores mx, b = (scores, mx, none) ∧
        scores.val.length = pos.val + 1 ∧
        (∀ (j : ℕ) (_hj : j < scores.val.length),
          (scores.val.getD j 0#i64).val =
            (scoreList kplane.val q.val kv.val hd.val kvh.val scale.val pos.val).getD j 0) ∧
        mx.val = intListMax core.num.I64.MIN.val (scores.val.map (fun x => x.val)) ∧
        ∃ (j : ℕ) (_hj : j < scores.val.length), scores.val.getD j 0#i64 = mx) := by
  have h0z : (0#usize : Usize).val = 0 := by norm_num
  have hinv0 : attnHeadLoop0Inv hd scale kv kvh pos kplane q scores0 core.num.I64.MIN
      0#usize true none := by
    refine ⟨rfl, rfl, by omega, ?_, ?_, ?_⟩
    · rw [hscores0, h0z]; rfl
    · intro j hj; rw [hscores0] at hj; simp at hj
    · rw [hscores0]; rfl
  have hspec : WP.spec
      (loop (fun (st : (alloc.vec.Vec Std.I64) × Std.I64 × Std.Usize × Bool ×
            (Option attn.AttnError)) =>
          attn.KvState.attn_head_loop0.body hd scale q pos kplane stride first st.1 st.2.1
            st.2.2.1 st.2.2.2.1 st.2.2.2.2)
        (scores0, core.num.I64.MIN, (0#usize : Std.Usize), true, (none : Option attn.AttnError)))
      (fun b : (alloc.vec.Vec Std.I64) × Std.I64 × (Option attn.AttnError) =>
        ∃ scores mx, b = (scores, mx, none) ∧
          scores.val.length = pos.val + 1 ∧
          (∀ (j : ℕ) (_hj : j < scores.val.length),
            (scores.val.getD j 0#i64).val =
              (scoreList kplane.val q.val kv.val hd.val kvh.val scale.val pos.val).getD j 0) ∧
          mx.val = intListMax core.num.I64.MIN.val (scores.val.map (fun x => x.val))) := by
    apply loop.spec_decr_nat
      (measure := fun st :
          (alloc.vec.Vec Std.I64) × Std.I64 × Std.Usize × Bool × (Option attn.AttnError) =>
        pos.val + 1 - st.2.2.1.val)
      (inv := fun st => attnHeadLoop0Inv hd scale kv kvh pos kplane q
        st.1 st.2.1 st.2.2.1 st.2.2.2.1 st.2.2.2.2)
      (post := fun b : (alloc.vec.Vec Std.I64) × Std.I64 × (Option attn.AttnError) =>
        ∃ scores mx, b = (scores, mx, none) ∧
          scores.val.length = pos.val + 1 ∧
          (∀ (j : ℕ) (_hj : j < scores.val.length),
            (scores.val.getD j 0#i64).val =
              (scoreList kplane.val q.val kv.val hd.val kvh.val scale.val pos.val).getD j 0) ∧
          mx.val = intListMax core.num.I64.MIN.val (scores.val.map (fun x => x.val)))
    · intro st hst
      rcases st with ⟨scores, mx, s, ok1, err⟩
      obtain ⟨hok, herr, hle, hlen, hpt, hmxval⟩ := hst
      dsimp only at hok herr hle hlen hpt hmxval
      subst hok herr
      by_cases hlt : s.val ≤ pos.val
      · obtain ⟨cf, hcf, hpost⟩ := WP.spec_imp_exists
          (attn_head_loop0_body_step hd scale kv kvh stride first pos kplane q hslots scores mx s
            hdom hlt
            ⟨rfl, rfl, hle, hlen, hpt, hmxval⟩)
        obtain ⟨scores2, mx2, s2, hcfg, hinv2, hs2⟩ := hpost
        rw [hcfg] at hcf
        simp only [hcf]
        refine ⟨hinv2, ?_⟩
        show pos.val + 1 - s2.val < pos.val + 1 - s.val
        omega
      · have hnot : ¬ (s ≤ pos) := by
          intro hc
          exact hlt ((UScalar.le_equiv s pos).1 hc)
        have hbody : attn.KvState.attn_head_loop0.body hd scale q pos kplane stride first scores
            mx s true none = ok (ControlFlow.done (scores, mx, none)) := by
          rw [attn.KvState.attn_head_loop0.body, if_neg hnot]
        simp only [hbody]
        exact ⟨scores, mx, rfl, by omega, hpt, hmxval⟩
    · exact hinv0
  obtain ⟨b, hb, scores, mx, hbeq, hlen, hpt, hmxval⟩ := WP.spec_imp_exists hspec
  rw [WP.spec_equiv_exists]
  refine ⟨b, by unfold attn.KvState.attn_head_loop0; exact hb, scores, mx, hbeq, hlen, hpt,
    hmxval, ?_⟩
  have h0 : 0 < scores.val.length := by omega
  have hmem0 : (scores.val.getD 0 0#i64).val ∈ scores.val.map (fun x => x.val) := by
    rw [getD_eq_get h0]
    exact List.mem_map.2 ⟨_, List.getElem_mem h0, rfl⟩
  have hx0 : core.num.I64.MIN.val < (scores.val.getD 0 0#i64).val := by
    rw [hpt 0 h0]
    obtain ⟨_i10, _b0, _v30, krow0, _hsm0, _haf0, _hng0, _hah0, _hi0, hkrow0, hdotdom0,
      houtlo0, _houthi0⟩ := hdom.2.2 0#usize (by rw [h0z]; omega)
    rw [h0z] at hkrow0
    have hkl0 : krow0.val.length = q.val.length := hdotdom0.1
    rw [scoreList_getD_eq kplane.val q kv.val hd.val kvh.val scale.val pos.val 0
      (Nat.zero_le _) krow0 hkrow0 hkl0]
    have hMIN : core.num.I64.MIN.val = -((2 : ℤ) ^ (63 : ℕ)) := by decide
    rw [hMIN]
    have hpow : (2 : ℤ) ^ (62 : ℕ) < (2 : ℤ) ^ (63 : ℕ) := by norm_num
    linarith
  rcases intListMax_mem_or_eq_seed core.num.I64.MIN.val (scores.val.map (fun x => x.val))
    with h | h
  · exfalso
    have hge := intListMax_ge_mem core.num.I64.MIN.val (scores.val.map (fun x => x.val)) _ hmem0
    rw [h] at hge
    exact absurd hx0 (not_lt.2 hge)
  · rw [← hmxval] at h
    obtain ⟨x, hxmem, hxval⟩ := List.mem_map.1 h
    obtain ⟨j, hj, hjx⟩ := List.getElem_of_mem hxmem
    refine ⟨j, hj, ?_⟩
    rw [getD_eq_get hj, hjx]
    exact IScalar.eq_of_val_eq hxval

/-! ## Loop 1: softmax numerators and the checked total -/

/-- **Numerator magnitude bound.** On the `exp_at` shape `attnNumeratorCase sh E` at a nonpositive
shift `sh`, the numerator is nonnegative and at most `4·2^32`: `attn_numerator_facts` bounds
`E/2^32` by `exp(sh/2^32)·(1 + epsExp) + 4/2^32 + exp(-16)`, and each of the four terms is at most
`1` once `sh ≤ 0` and `epsExp < 1`. Keeps the loop's checked `i128` total inside range without a
separate domain hypothesis. -/
theorem attnNumeratorCase_le (sh E : ℤ) (hsh : sh ≤ 0) (h : attnNumeratorCase sh E) :
    0 ≤ E ∧ E ≤ 4 * (2 : ℤ) ^ (32 : ℕ) := by
  obtain ⟨hb, hnn⟩ := attn_numerator_facts sh E h
  have hq : q32 = (2 : ℝ) ^ (32 : ℕ) := rfl
  have hEq : (E : ℝ) = (E : ℝ) / q32 * q32 := (div_mul_cancel₀ _ q32_pos.ne').symm
  have hnnZ : 0 ≤ E := by
    have h0 : (0 : ℝ) ≤ (E : ℝ) := by rw [hEq]; exact mul_nonneg hnn q32_pos.le
    exact_mod_cast h0
  refine ⟨hnnZ, ?_⟩
  have hshR : (sh : ℝ) / q32 ≤ 0 := by
    have h1 : (sh : ℝ) ≤ 0 := by exact_mod_cast hsh
    exact div_nonpos_of_nonpos_of_nonneg h1 q32_pos.le
  have hexple : Real.exp ((sh : ℝ) / q32) ≤ 1 := by
    calc Real.exp ((sh : ℝ) / q32) ≤ Real.exp 0 := Real.exp_le_exp.2 hshR
      _ = 1 := Real.exp_zero
  have hexp16 : Real.exp (-16 : ℝ) ≤ 1 := by
    calc Real.exp (-16 : ℝ) ≤ Real.exp 0 := Real.exp_le_exp.2 (by norm_num)
      _ = 1 := Real.exp_zero
  have hup := (abs_le.1 hb).2
  have h4q32 : (4 : ℝ) / q32 ≤ 1 := by rw [hq]; norm_num
  have hepsexp : epsExp * Real.exp ((sh : ℝ) / q32) ≤ 1 := by
    calc epsExp * Real.exp ((sh : ℝ) / q32) ≤ 1 * 1 :=
          mul_le_mul epsExp_lt_one.le hexple (Real.exp_pos _).le zero_le_one
      _ = 1 := one_mul 1
  have hfin : (E : ℝ) / q32 ≤ 4 := by linarith
  have hfin2 : (E : ℝ) ≤ 4 * (2 : ℝ) ^ (32 : ℕ) := by
    rw [hEq, ← hq]
    nlinarith [q32_pos]
  exact_mod_cast hfin2

/-- `I64.checked_sub` returning `some` carries the exact difference. -/
theorem I64_checked_sub_val {x y z : I64} (h : I64.checked_sub x y = some z) :
    z.val = x.val - y.val := by
  have hspec := I64.checked_sub_bv_spec x y
  have hc : core.num.checked_sub_IScalar x y = some z := h
  rw [hc] at hspec
  exact hspec.2.2.1

/-- The per-position window `attn.KvState.attn_head_loop1` needs at row `j`: the checked `i64`
subtraction `score_j - mx` succeeds with a nonpositive result (so `exp_at_numerator_case` applies).
Not implied by `attn_domain_v2` as a checked-arithmetic fact (finding 2 above); supplied explicitly
here. `pos.val + 1 ≤ 2^30` is `attn_domain_v2`'s own occupancy-independent value window; combined
with `attnNumeratorCase_le` it keeps the checked `i128` total inside range without a further
hypothesis (closing finding 3 above). -/
def attnHeadLoop1Domain (pos : Usize) (scores : alloc.vec.Vec I64) (mx : I64) : Prop :=
  pos.val + 1 ≤ 2 ^ (30 : ℕ) ∧
  ∀ j : ℕ, j ≤ pos.val →
    ∃ shifted : I64, I64.checked_sub (scores.val.getD j 0#i64) mx = some shifted ∧
      shifted.val ≤ 0

/-- The numerator-loop invariant after `s` completed rows: the failure flag is clear, `exps` holds
exactly `s` entries in the `exp_at` shape against `scores`/`mx`, and `total` is their exact checked
`i128` sum, kept inside `[0, s·4·2^32]`. -/
def attnHeadLoop1Inv (pos : Usize) (scores : alloc.vec.Vec I64) (mx : I64)
    (exps : alloc.vec.Vec I64) (total : I128) (s : Usize) (ok1 : Bool)
    (err : Option attn.AttnError) : Prop :=
  ok1 = true ∧ err = none ∧ s.val ≤ pos.val + 1 ∧ exps.val.length = s.val ∧
  (∀ j, j < exps.val.length →
    attnNumeratorCase ((scores.val.getD j 0#i64).val - mx.val) (exps.val.getD j 0#i64).val) ∧
  0 ≤ total.val ∧ total.val ≤ (s.val : ℤ) * (4 * (2 : ℤ) ^ (32 : ℕ)) ∧
  total.val = (exps.val.map (fun x => x.val)).sum

/-- One numerator-loop body step: on the domain and the invariant, with `s.val ≤ pos.val` (the
loop's own guard `s ≤ pos` still true) and `scores.val.length = pos.val + 1`, the body reads
`scores[s]`, evaluates `exp_at` against `mx`, accumulates the checked `i128` total, pushes the
numerator, and continues at `s + 1` with the invariant restored. -/
theorem attn_head_loop1_body_step (pos : Usize) (scores : alloc.vec.Vec I64) (mx : I64)
    (hslen : scores.val.length = pos.val + 1) (hslots : pos.val + 1 ≤ Usize.max)
    (exps : alloc.vec.Vec I64) (total : I128) (s : Usize)
    (hdom : attnHeadLoop1Domain pos scores mx)
    (hs : s.val ≤ pos.val)
    (hinv : attnHeadLoop1Inv pos scores mx exps total s true none) :
    WP.spec
      (attn.KvState.attn_head_loop1.body pos scores mx exps total s true none)
      (fun cf => ∃ exps2 total2 s2,
        cf = ControlFlow.cont (exps2, total2, s2, true, none) ∧
        attnHeadLoop1Inv pos scores mx exps2 total2 s2 true none ∧
        s2.val = s.val + 1) := by
  obtain ⟨_hok, _herr, _hle, hlen, hpt, htotlo, htothi, htoteq⟩ := hinv
  obtain ⟨hcap, hshdom⟩ := hdom
  have hsb : s.val < scores.val.length := by rw [hslen]; omega
  obtain ⟨shifted, hsub, hshle⟩ := hshdom s.val hs
  have hshv : shifted.val = (scores.val.getD s.val 0#i64).val - mx.val := I64_checked_sub_val hsub
  rw [getD_eq_get hsb] at hsub
  obtain ⟨e, he_eq, hcase⟩ := exp_at_numerator_case (scores.val[s.val]'hsb) mx shifted hsub hshle
  have hebound := attnNumeratorCase_le _ _ hshle hcase
  have hsle : s ≤ pos := (UScalar.le_equiv s pos).2 hs
  have hidx : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.I64) scores s
      = ok (scores.val[s.val]'hsb) := by
    rw [alloc.vec.Vec.index_slice_index]
    obtain ⟨x, hx1, hx2⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec scores s hsb)
    rw [hx1, hx2]
  have hei : (core.convert.num.FromI128I64.from e).val = e.val :=
    core.convert.num.FromI128I64.from_val_eq e
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by
    norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
    norm_num [I128.max, IScalar.max, I128.numBits]
  have hs30 : (s.val : ℤ) + 1 ≤ (2 : ℤ) ^ (30 : ℕ) := by
    have : s.val + 1 ≤ 2 ^ (30 : ℕ) := by omega
    exact_mod_cast this
  obtain ⟨t, hadd_eq, htv⟩ := I128_checked_add_some total (core.convert.num.FromI128I64.from e)
    (by rw [hei, hI128min]; have : (0 : ℤ) ≤ (2 : ℤ) ^ (127 : ℕ) := by positivity
        linarith [hebound.1])
    (by
      rw [hei, hI128max]
      have hC : (0 : ℤ) ≤ 4 * (2 : ℤ) ^ (32 : ℕ) := by positivity
      have hstep : (s.val : ℤ) * (4 * (2 : ℤ) ^ (32 : ℕ)) + 4 * (2 : ℤ) ^ (32 : ℕ)
          ≤ (2 : ℤ) ^ (30 : ℕ) * (4 * (2 : ℤ) ^ (32 : ℕ)) := by
        have := mul_le_mul_of_nonneg_right hs30 hC
        linarith
      have hnum : (2 : ℤ) ^ (30 : ℕ) * (4 * (2 : ℤ) ^ (32 : ℕ)) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by
        norm_num
      linarith [hebound.2])
  have h1 : (1#usize).val = 1 := by norm_num
  have hlemax : s.val + (1#usize).val ≤ Usize.max := by rw [h1]; omega
  obtain ⟨s2, hadd1, hs2⟩ := WP.spec_imp_exists (Usize.add_spec (x := s) (y := 1#usize) hlemax)
  have hs2' : s2.val = s.val + 1 := by rw [hs2, h1]
  have hpushmax : exps.val.length < Usize.max := by rw [hlen]; omega
  obtain ⟨exps2, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec exps e hpushmax)
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (exps2, t, s2, true, none), ?_, exps2, t, s2, rfl,
    ⟨rfl, rfl, by omega, ?_, ?_, ?_, ?_, ?_⟩, hs2'⟩
  · rw [attn.KvState.attn_head_loop1.body, if_pos hsle]
    try dsimp only
    rw [hidx]
    simp only [bind_tc_ok]
    try dsimp only
    rw [he_eq]
    simp only [bind_tc_ok]
    try dsimp only
    simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hadd_eq]
    try dsimp only
    rw [hpush]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hadd1]
    simp only [bind_tc_ok]
    rfl
  · rw [hpushv, hs2']; simp [hlen]
  · intro j hj
    rw [hpushv] at hj ⊢
    by_cases hjlt : j < exps.val.length
    · rw [List.getD_append exps.val [e] 0#i64 j hjlt]; exact hpt j hjlt
    · have hjeq : j = exps.val.length := by
        simp only [List.length_append, List.length_singleton] at hj
        omega
      subst hjeq
      rw [List.getD_append_right exps.val [e] 0#i64 exps.val.length (le_refl _)]
      simp only [Nat.sub_self, List.getD_cons_zero]
      rw [hlen, ← hshv]
      exact hcase
  · rw [htv, hei]; linarith [hebound.1]
  · rw [htv, hei, hs2']
    push_cast
    linarith [hebound.2]
  · rw [hpushv, htv, hei, htoteq]
    simp [List.map_append, List.sum_append]

/-- **Numerator-loop specification.** On `attnHeadLoop1Domain` and `scores.val.length = pos.val+1`
(loop 0's own conclusion), the extracted `attn_head_loop1` starting from an empty `exps` and
`total = 0` runs to completion (`err = none`) with `exps` in the `exp_at` shape
`attnNumeratorCase (scores[j] - mx) exps[j]` against every completed key, and `total` their exact
nonnegative sum — the hypotheses `attn_stage_compose`/`attn_stage_compose_closed`
(`Numeric.AttnSoftmaxPerturb`) need beyond `hmax`. -/
theorem attn_head_loop1_spec (pos : Usize) (scores : alloc.vec.Vec I64) (mx : I64)
    (hslen : scores.val.length = pos.val + 1) (hslots : pos.val + 1 ≤ Usize.max)
    (hdom : attnHeadLoop1Domain pos scores mx)
    (exps0 : alloc.vec.Vec I64) (hexps0 : exps0.val = []) :
    WP.spec (attn.KvState.attn_head_loop1 pos scores mx exps0 0#i128 0#usize true none)
      (fun b => ∃ exps total, b = (exps, total, none) ∧
        exps.val.length = pos.val + 1 ∧
        (∀ j, j < exps.val.length →
          attnNumeratorCase ((scores.val.getD j 0#i64).val - mx.val) (exps.val.getD j 0#i64).val) ∧
        0 ≤ total.val ∧
        total.val = (exps.val.map (fun x => x.val)).sum) := by
  have h0z : (0#usize : Usize).val = 0 := by norm_num
  have h0t : (0#i128 : I128).val = 0 := by norm_num
  have hinv0 : attnHeadLoop1Inv pos scores mx exps0 0#i128 0#usize true none := by
    refine ⟨rfl, rfl, by omega, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hexps0, h0z]; rfl
    · intro j hj; rw [hexps0] at hj; simp at hj
    · rw [h0t]
    · rw [h0t, h0z]; norm_num
    · rw [hexps0, h0t]; simp
  have hspec : WP.spec
      (loop (fun (st : (alloc.vec.Vec Std.I64) × Std.I128 × Std.Usize × Bool ×
            (Option attn.AttnError)) =>
          attn.KvState.attn_head_loop1.body pos scores mx st.1 st.2.1 st.2.2.1 st.2.2.2.1
            st.2.2.2.2)
        (exps0, (0#i128 : Std.I128), (0#usize : Std.Usize), true, (none : Option attn.AttnError)))
      (fun b : (alloc.vec.Vec Std.I64) × Std.I128 × (Option attn.AttnError) =>
        ∃ exps total, b = (exps, total, none) ∧
          exps.val.length = pos.val + 1 ∧
          (∀ j, j < exps.val.length →
            attnNumeratorCase ((scores.val.getD j 0#i64).val - mx.val)
              (exps.val.getD j 0#i64).val) ∧
          0 ≤ total.val ∧
          total.val = (exps.val.map (fun x => x.val)).sum) := by
    apply loop.spec_decr_nat
      (measure := fun st :
          (alloc.vec.Vec Std.I64) × Std.I128 × Std.Usize × Bool × (Option attn.AttnError) =>
        pos.val + 1 - st.2.2.1.val)
      (inv := fun st => attnHeadLoop1Inv pos scores mx st.1 st.2.1 st.2.2.1 st.2.2.2.1 st.2.2.2.2)
      (post := fun b : (alloc.vec.Vec Std.I64) × Std.I128 × (Option attn.AttnError) =>
        ∃ exps total, b = (exps, total, none) ∧
          exps.val.length = pos.val + 1 ∧
          (∀ j, j < exps.val.length →
            attnNumeratorCase ((scores.val.getD j 0#i64).val - mx.val)
              (exps.val.getD j 0#i64).val) ∧
          0 ≤ total.val ∧
          total.val = (exps.val.map (fun x => x.val)).sum)
    · intro st hst
      rcases st with ⟨exps, total, s, ok1, err⟩
      obtain ⟨hok, herr, hle, hlen, hpt, htotlo, htothi, htoteq⟩ := hst
      dsimp only at hok herr hle hlen hpt htotlo htothi htoteq
      subst hok herr
      by_cases hlt : s.val ≤ pos.val
      · obtain ⟨cf, hcf, hpost⟩ := WP.spec_imp_exists
          (attn_head_loop1_body_step pos scores mx hslen hslots exps total s
            hdom hlt ⟨rfl, rfl, hle, hlen, hpt, htotlo, htothi, htoteq⟩)
        obtain ⟨exps2, total2, s2, hcfg, hinv2, hs2⟩ := hpost
        rw [hcfg] at hcf
        simp only [hcf]
        refine ⟨hinv2, ?_⟩
        show pos.val + 1 - s2.val < pos.val + 1 - s.val
        omega
      · have hnot : ¬ (s ≤ pos) := by
          intro hc
          exact hlt ((UScalar.le_equiv s pos).1 hc)
        have hbody : attn.KvState.attn_head_loop1.body pos scores mx exps total s true none
            = ok (ControlFlow.done (exps, total, none)) := by
          rw [attn.KvState.attn_head_loop1.body, if_neg hnot]
        simp only [hbody]
        exact ⟨exps, total, rfl, by omega, hpt, htotlo, htoteq⟩
    · exact hinv0
  unfold attn.KvState.attn_head_loop1
  exact hspec

end

end apeiron_word
