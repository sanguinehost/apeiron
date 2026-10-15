import ApeironWord
import Learn.Grad.Head
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The extracted vocabulary head: the loop refinements

`src/learn/grad/head.rs` holds the head of the ported one-bit transfer. `Learn/Grad/Head.lean`
refines its non-loop bodies against the `ℤ` specifications; this module refines the thirteen
extracted `_loop` bodies and the public entry points that compose them.

Each loop theorem is `loop.spec_decr_nat` over the extracted `_loop` definition itself, with an
explicit invariant that carries the accumulator's value as the exact `ℤ` prefix sum the Rust
computes (`klAccAux`, `dzVals`, `tvAbsAux`/`tvSignedAux`, `dotAux`, `dwVals`, …) and the explicit
no-overflow domain that makes every checked operation admit its result. The loop accumulator models
are the ones stated in `Learn/Grad/Head.lean`; they are not restated here.

Trust boundary: rustc/LLVM codegen, the kernel ABI and the container layout are not covered; only
the extracted integer body. The refusal arms of the loops carry the invariant's `fail` component:
outside the stated domain the body sets `fail` and leaves the accumulator alone.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_head

open apeiron_word
open Aeneas Aeneas.Std Result

/-! ## 1. Value lists and shared helpers -/

/-- The value list of an `i64` slice. -/
def hvals (x : Slice I64) : List ℤ := x.val.map (fun z => z.val)

/-- The value list of an `i128` slice. -/
def svals (a : Slice I128) : List ℤ := a.val.map (fun z => z.val)

/-- The value list of an `i64` vector. -/
def vvals (v : alloc.vec.Vec I64) : List ℤ := v.val.map (fun z => z.val)

theorem hvals_length (x : Slice I64) : (hvals x).length = x.val.length := by simp [hvals]

theorem svals_length (a : Slice I128) : (svals a).length = a.val.length := by simp [svals]

theorem hvals_getD (x : Slice I64) {k : ℕ} (h : k < x.val.length) :
    (hvals x).getD k 0 = (x.val.getD k default).val := by
  rw [hvals, List.getD_eq_getElem _ _ (by rwa [List.length_map]), List.getElem_map,
    List.getD_eq_getElem _ _ h]

theorem svals_getD (a : Slice I128) {k : ℕ} (h : k < a.val.length) :
    (svals a).getD k 0 = (a.val.getD k default).val := by
  rw [svals, List.getD_eq_getElem _ _ (by rwa [List.length_map]), List.getElem_map,
    List.getD_eq_getElem _ _ h]

/-- The `i64` value list of a `Slice.set`: the mapped list is set at the index. -/
theorem hvals_set (x : Slice I64) (i : Usize) (v : I64) :
    hvals (x.set i v) = (hvals x).set i.val v.val := by
  simp only [hvals, Slice.set_val_eq, List.map_set]

theorem vvals_set (v : alloc.vec.Vec I64) (k : Usize) (x : I64) :
    vvals (v.set k x) = (vvals v).set k.val x.val := by
  simp only [vvals, alloc.vec.Vec.set_val_eq, List.map_set]

theorem svals_set (x : Slice I128) (i : Usize) (v : I128) :
    svals (x.set i v) = (svals x).set i.val v.val := by
  simp only [svals, Slice.set_val_eq, List.map_set]

/-- A `usize` successor that fits: the extracted `i + 1#usize`. -/
theorem usize_succ_fits {i lim : Usize} (h : i.val < lim.val) (hlim : lim.val ≤ Usize.max) :
    ∃ i', i + 1#usize = ok i' ∧ i'.val = i.val + 1 := by
  obtain ⟨i', hi', hi'v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) (by
    show i.val + (1#usize).val ≤ Usize.max
    rw [show (1#usize).val = 1 from rfl]
    omega))
  exact ⟨i', hi', by rw [hi'v, show (1#usize).val = 1 from rfl]⟩

/-! ## 2. `row_max`: the maximum of a row

Specification (`Learn/Grad/Head.lean`): `maxInit l m` folds `l` with `max` from `m`. The loop
carries the prefix maximum `maxInit ((hvals z).take i) m0` and returns the whole-row maximum
`maxInit ((hvals z).take v) m0`. No checked arithmetic: the only refusal is out of the domain
`v ≤ z.len`, which would index out of bounds. -/

theorem maxInit_append_single (l : List ℤ) (m x : ℤ) :
    maxInit (l ++ [x]) m = (if x > maxInit l m then x else maxInit l m) := by
  simp only [maxInit, List.foldl_append, List.foldl_cons, List.foldl_nil]

/-- One fold step of `maxInit` on a prefix, at an index inside the list. -/
theorem maxInit_take_succ (l : List ℤ) (m : ℤ) (n : ℕ) (h : n < l.length) :
    maxInit (l.take (n + 1)) m =
      (if l.getD n 0 > maxInit (l.take n) m then l.getD n 0 else maxInit (l.take n) m) := by
  rw [← List.take_append_getElem h, maxInit_append_single, List.getD_eq_getElem _ _ h]

/-- The `row_max` loop invariant: `i` scanned, `m` the prefix maximum from `m0`. -/
def rowMaxInv (z : Slice I64) (v : Usize) (m0 : ℤ) : I64 × Usize → Prop :=
  fun s => s.2.val ≤ v.val ∧ s.1.val = maxInit ((hvals z).take s.2.val) m0

/-- The extracted `row_max` body: one scan step advances the prefix maximum. -/
theorem row_max_loop_body_spec (z : Slice I64) (v : Usize) (m0 : ℤ)
    (hvl : v.val ≤ (hvals z).length)
    (st : I64 × Usize) (hinv : rowMaxInv z v m0 st) :
    learn_port.head.row_max_loop.body z v st.1 st.2 ⦃ r =>
      match r with
      | ControlFlow.done y => y.val = maxInit ((hvals z).take v.val) m0
      | ControlFlow.cont st' => rowMaxInv z v m0 st' ∧
          v.val - st'.2.val < v.val - st.2.val ⦄ := by
  obtain ⟨m, i⟩ := st
  simp only [rowMaxInv] at hinv
  obtain ⟨hiv, hmv⟩ := hinv
  have hzmax : z.val.length ≤ Usize.max := Slice.property z
  have hvl' : v.val ≤ z.val.length := by rw [hvals_length z] at hvl; exact hvl
  by_cases hlt : i < v
  · have hltv : i.val < v.val := (UScalar.lt_equiv i v).mp hlt
    have hlen : i.val < z.val.length := by omega
    have hidx := slice_index_getD z i hlen
    obtain ⟨i2, hi2, hi2v⟩ := usize_succ_fits hltv (by omega)
    have hi2_val : i2.val = i.val + 1 := hi2v
    by_cases hgt : z.val.getD i.val default > m
    · have hgtv : (z.val.getD i.val default).val > m.val := (IScalar.lt_equiv m _).mp hgt
      have hbody : learn_port.head.row_max_loop.body z v m i =
          ok (ControlFlow.cont (z.val.getD i.val default, i2)) := by
        rw [learn_port.head.row_max_loop.body, if_pos hlt, hidx]
        simp only [bind_tc_ok, if_pos hgt, hi2]
      rw [hbody]
      simp only [WP.spec_ok, rowMaxInv]
      have hgetD : (hvals z).getD i.val 0 = (z.val.getD i.val default).val :=
        hvals_getD z hlen
      refine ⟨⟨by omega, ?_⟩, by omega⟩
      rw [show i2.val = i.val + 1 from hi2v,
        maxInit_take_succ (hvals z) m0 i.val (by rw [hvals_length]; exact hlen),
        hgetD, ← hmv, if_pos hgtv]
    · have hgtv : ¬ ((z.val.getD i.val default).val > m.val) := by
        intro hc
        exact hgt ((IScalar.lt_equiv m _).mpr hc)
      have hbody : learn_port.head.row_max_loop.body z v m i =
          ok (ControlFlow.cont (m, i2)) := by
        rw [learn_port.head.row_max_loop.body, if_pos hlt, hidx]
        simp only [bind_tc_ok, if_neg hgt, hi2]
      rw [hbody]
      simp only [WP.spec_ok, rowMaxInv]
      have hgetD : (hvals z).getD i.val 0 = (z.val.getD i.val default).val :=
        hvals_getD z hlen
      refine ⟨⟨by omega, ?_⟩, by omega⟩
      rw [show i2.val = i.val + 1 from hi2v,
        maxInit_take_succ (hvals z) m0 i.val (by rw [hvals_length]; exact hlen),
        hgetD, ← hmv, if_neg hgtv]
  · have hbody : learn_port.head.row_max_loop.body z v m i = ok (ControlFlow.done m) := by
      rw [learn_port.head.row_max_loop.body, if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hieq : i.val = v.val := by
      have hvle : v.val ≤ i.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv i v).mpr (by omega))
      omega
    rw [← hieq, hmv]

/-- **The `row_max` loop.** Over the whole row the loop returns `maxInit ((hvals z).take v) m0`. -/
theorem row_max_loop_spec (z : Slice I64) (v : Usize) (m : I64) (i : Usize) (m0 : ℤ)
    (hvl : v.val ≤ (hvals z).length)
    (hinv0 : rowMaxInv z v m0 (m, i)) :
    WP.spec (learn_port.head.row_max_loop z v m i)
      (fun m' : I64 => m'.val = maxInit ((hvals z).take v.val) m0) := by
  unfold learn_port.head.row_max_loop
  apply loop.spec_decr_nat
    (measure := fun s : I64 × Usize => v.val - s.2.val)
    (inv := rowMaxInv z v m0)
    (post := fun m' : I64 => m'.val = maxInit ((hvals z).take v.val) m0)
  · intro st hst
    convert row_max_loop_body_spec z v m0 hvl st hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · exact hinv0

/-! ## 3. Shared checked-arithmetic helpers -/

/-- The `i64` difference fits `i128`. -/
theorem i64_sub_fits (a b : I64) : I128.min ≤ a.val - b.val ∧ a.val - b.val ≤ I128.max := by
  obtain ⟨ha1, ha2⟩ := i64_bounds a
  obtain ⟨hb1, hb2⟩ := i64_bounds b
  simp only [i64_min_eq, i64_max_eq] at ha1 ha2 hb1 hb2
  simp only [i128_min_eq, i128_max_eq]
  norm_num at ha1 ha2 hb1 hb2 ⊢
  omega

/-- The `i128` widening of an `i64` difference fits `i128`: the hypothesis `I128.sub_bv_spec`
needs, with the `FromI128I64.from` values named rather than left as inference holes. -/
theorem from_i64_sub_fits (a b : I64) :
    I128.min ≤ (core.convert.num.FromI128I64.from a).val -
        (core.convert.num.FromI128I64.from b).val ∧
      (core.convert.num.FromI128I64.from a).val - (core.convert.num.FromI128I64.from b).val ≤
        I128.max := by
  simpa only [core.convert.num.FromI128I64.from_val_eq] using i64_sub_fits a b

/-- The `round_div` domain at one call: the two rounded quotients fit `i128`. -/
def rndDivDom (n d : ℤ) : Prop :=
  (I128.min ≤ Int.tdiv n d + 1 ∧ Int.tdiv n d + 1 ≤ I128.max) ∧
  (I128.min ≤ Int.tdiv n d - 1 ∧ Int.tdiv n d - 1 ≤ I128.max)

/-- `round_div` on its stated domain returns the `ℤ` rounding `rndDiv`. -/
theorem round_div_dom_spec (n d : I128) (hd : 0 < d.val)
    (h2 : 2 * absZ (Int.tmod n.val d.val) ≤ I128.max) (h : rndDivDom n.val d.val) :
    ∃ v, learn_port.head.round_div n d = ok v ∧ v.val = rndDiv n.val d.val :=
  round_div_spec n d hd h2 h.1 h.2

/-! ## 4. `kl_acc`: the KL accumulator

Specification (`Learn/Grad/Head.lean`): `klAccAux p lp lp_s a n` is `a` plus the `n` terms
`p[t]·(lp[t] − lp_s[t])`, the exact `i128` sum the Rust accumulates. The loop starts at the zero
accumulator with no failure; the stated domain is that every per-index product and every prefix sum
fits `i128`, so no checked operation refuses and the accumulator is the whole sum. -/

/-- The `kl_acc` term at index `t`: `p_t[t]·(lp_t[t] − lp_s[t])`. -/
def klTerm (p lp lp_s : List ℤ) (t : ℕ) : ℤ := p.getD t 0 * (lp.getD t 0 - lp_s.getD t 0)

/-- `klTerm` at its unfolded body, as a rewrite target. -/
theorem klTerm_eq (p lp lp_s : List ℤ) (t : ℕ) :
    klTerm p lp lp_s t = p.getD t 0 * (lp.getD t 0 - lp_s.getD t 0) := rfl

/-- `klAccAux` is additive in its starting accumulator. -/
theorem klAccAux_add (p lp lp_s : List ℤ) (a : ℤ) (n : ℕ) :
    klAccAux p lp lp_s a n = a + klAccAux p lp lp_s 0 n := by
  induction n generalizing a with
  | zero => simp only [klAccAux, add_zero]
  | succ n ih =>
    simp only [klAccAux]
    rw [ih (a + p.getD n 0 * (lp.getD n 0 - lp_s.getD n 0))]
    rw [ih (0 + p.getD n 0 * (lp.getD n 0 - lp_s.getD n 0))]
    ring

/-- One `kl_acc` step: the term at index `n` is added. -/
theorem klAccAux_succ (p lp lp_s : List ℤ) (a : ℤ) (n : ℕ) :
    klAccAux p lp lp_s a (n + 1) = klAccAux p lp lp_s a n + klTerm p lp lp_s n := by
  have hunfold : ∀ b : ℤ, klAccAux p lp lp_s b (n + 1) =
      klAccAux p lp lp_s (b + klTerm p lp lp_s n) n := by
    intro b
    simp only [klTerm, klAccAux]
  rw [hunfold a, klAccAux_add p lp lp_s (a + klTerm p lp lp_s n) n,
    klAccAux_add p lp lp_s a n]
  ring

/-- The `kl_acc` loop invariant. -/
def klAccInv (p_t lp_t lp_s : Slice I64) (v : Usize)
    (s : I128 × Option learn_port.head.HeadError × Usize) : Prop :=
  s.2.1 = none ∧ s.2.2.val ≤ v.val ∧
    s.1.val = klAccAux (hvals p_t) (hvals lp_t) (hvals lp_s) 0 s.2.2.val

/-- The extracted `kl_acc` body: one term is accumulated. On the stated no-overflow domain the
multiply and add admit their results, so the invariant is preserved with the prefix sum extended. -/
theorem kl_acc_loop_body_spec (p_t lp_t lp_s : Slice I64) (v : Usize)
    (htl : v.val ≤ lp_t.val.length) (hsl : v.val ≤ lp_s.val.length) (hpl : v.val ≤ p_t.val.length)
    (hprod : ∀ t < v.val, I128.min ≤ klTerm (hvals p_t) (hvals lp_t) (hvals lp_s) t ∧
        klTerm (hvals p_t) (hvals lp_t) (hvals lp_s) t ≤ I128.max)
    (hsum : ∀ t ≤ v.val, I128.min ≤ klAccAux (hvals p_t) (hvals lp_t) (hvals lp_s) 0 t ∧
        klAccAux (hvals p_t) (hvals lp_t) (hvals lp_s) 0 t ≤ I128.max)
    (st : I128 × Option learn_port.head.HeadError × Usize)
    (hinv : klAccInv p_t lp_t lp_s v st) :
    learn_port.head.kl_acc_loop.body p_t lp_t lp_s v st.1 st.2.1 st.2.2 ⦃ r =>
      match r with
      | ControlFlow.done y => y.1.val = klAccAux (hvals p_t) (hvals lp_t) (hvals lp_s) 0 v.val ∧
          y.2 = none
      | ControlFlow.cont st' => klAccInv p_t lp_t lp_s v st' ∧
          v.val - st'.2.2.val < v.val - st.2.2.val ⦄ := by
  obtain ⟨acc, fail, i⟩ := st
  simp only [klAccInv] at hinv
  obtain ⟨hfail, hle, hacc⟩ := hinv
  by_cases hlt : i < v
  · have hltv : i.val < v.val := (UScalar.lt_equiv i v).mp hlt
    have hlim_t : lp_t.val.length ≤ Usize.max := Slice.property lp_t
    have hlim_s : lp_s.val.length ≤ Usize.max := Slice.property lp_s
    have hlim_p : p_t.val.length ≤ Usize.max := Slice.property p_t
    have hlen_t : i.val < lp_t.val.length := by omega
    have hlen_s : i.val < lp_s.val.length := by omega
    have hlen_p : i.val < p_t.val.length := by omega
    have hidx_t := slice_index_getD lp_t i hlen_t
    have hidx_s := slice_index_getD lp_s i hlen_s
    have hidx_p := slice_index_getD p_t i hlen_p
    obtain ⟨delta, hdelta, hdeltav, _⟩ := WP.spec_imp_exists (I128.sub_bv_spec
      (x := core.convert.num.FromI128I64.from (lp_t.val.getD i.val default))
      (y := core.convert.num.FromI128I64.from (lp_s.val.getD i.val default))
      (from_i64_sub_fits (lp_t.val.getD i.val default) (lp_s.val.getD i.val default)).1
      (from_i64_sub_fits (lp_t.val.getD i.val default) (lp_s.val.getD i.val default)).2)
    have hdv : delta.val = (lp_t.val.getD i.val default).val - (lp_s.val.getD i.val default).val := by
      rw [hdeltav]
      simp only [core.convert.num.FromI128I64.from_val_eq]
    have htermfit : I128.min ≤
          (core.convert.num.FromI128I64.from (p_t.val.getD i.val default)).val * delta.val ∧
        (core.convert.num.FromI128I64.from (p_t.val.getD i.val default)).val * delta.val ≤
          I128.max := by
      rw [core.convert.num.FromI128I64.from_val_eq, hdv, ← hvals_getD p_t hlen_p,
        ← hvals_getD lp_t hlen_t, ← hvals_getD lp_s hlen_s]
      exact hprod i.val hltv
    obtain ⟨term, hmul, htermv⟩ :=
      mul128_ok (core.convert.num.FromI128I64.from (p_t.val.getD i.val default)) delta htermfit
    have heq : acc.val + term.val =
        klAccAux (hvals p_t) (hvals lp_t) (hvals lp_s) 0 (i.val + 1) := by
      rw [hacc, htermv, core.convert.num.FromI128I64.from_val_eq, hdv,
        ← hvals_getD p_t hlen_p, ← hvals_getD lp_t hlen_t, ← hvals_getD lp_s hlen_s,
        ← klTerm_eq, ← klAccAux_succ]
    obtain ⟨s, hadd, hsv⟩ := add128_ok acc term (by rw [heq]; exact hsum (i.val + 1) (by omega))
    obtain ⟨i7, hi7, hi7v⟩ := usize_succ_fits hltv (by omega)
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.kl_acc_loop.body p_t lp_t lp_s v acc fail i =
        ok (ControlFlow.cont (s, none, i7)) := by
      rw [learn_port.head.kl_acc_loop.body, if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, lift, hidx_t, hidx_s, hidx_p, hdelta, hmul,
        hadd, hi7]
      rw [hfail]
      rfl
    rw [hbody]
    simp only [WP.spec_ok, klAccInv]
    refine ⟨⟨trivial, by omega, ?_⟩, by omega⟩
    rw [hsv, heq, hi7v]
  · have hbody : learn_port.head.kl_acc_loop.body p_t lp_t lp_s v acc fail i =
        ok (ControlFlow.done (acc, fail)) := by
      rw [learn_port.head.kl_acc_loop.body, if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hieq : i.val = v.val := by
      have hvle : v.val ≤ i.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv i v).mpr (by omega))
      omega
    exact ⟨by rw [← hieq, hacc], hfail⟩

/-- **The `kl_acc` loop.** From the zero accumulator the loop returns the whole sum. -/
theorem kl_acc_loop_spec (p_t lp_t lp_s : Slice I64) (v : Usize) (acc : I128)
    (fail : Option learn_port.head.HeadError) (i : Usize)
    (htl : v.val ≤ lp_t.val.length) (hsl : v.val ≤ lp_s.val.length) (hpl : v.val ≤ p_t.val.length)
    (hprod : ∀ t < v.val, I128.min ≤ klTerm (hvals p_t) (hvals lp_t) (hvals lp_s) t ∧
        klTerm (hvals p_t) (hvals lp_t) (hvals lp_s) t ≤ I128.max)
    (hsum : ∀ t ≤ v.val, I128.min ≤ klAccAux (hvals p_t) (hvals lp_t) (hvals lp_s) 0 t ∧
        klAccAux (hvals p_t) (hvals lp_t) (hvals lp_s) 0 t ≤ I128.max)
    (hinv0 : klAccInv p_t lp_t lp_s v (acc, fail, i)) :
    WP.spec (learn_port.head.kl_acc_loop p_t lp_t lp_s v acc fail i)
      (fun y : I128 × Option learn_port.head.HeadError =>
        y.1.val = klAccAux (hvals p_t) (hvals lp_t) (hvals lp_s) 0 v.val ∧ y.2 = none) := by
  unfold learn_port.head.kl_acc_loop
  apply loop.spec_decr_nat
    (measure := fun s : I128 × Option learn_port.head.HeadError × Usize => v.val - s.2.2.val)
    (inv := klAccInv p_t lp_t lp_s v)
    (post := fun y : I128 × Option learn_port.head.HeadError =>
      y.1.val = klAccAux (hvals p_t) (hvals lp_t) (hvals lp_s) 0 v.val ∧ y.2 = none)
  · intro st hst
    convert kl_acc_loop_body_spec p_t lp_t lp_s v htl hsl hpl hprod hsum st hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · exact hinv0

/-! ## 5. `dz_diff`: the logit gradient

Specification (`Learn/Grad/Head.lean`): `dzVals p_s p_t dz n` is `dz` with entry `t < n` set to
`p_s[t] − p_t[t]`. The domain is that the differences fit `i64` (so `to64` admits) and `dz` is long
enough. -/

theorem dzVals_succ (p_s p_t dz : List ℤ) (n : ℕ) :
    dzVals p_s p_t dz (n + 1) = (dzVals p_s p_t dz n).set n (p_s.getD n 0 - p_t.getD n 0) := rfl

theorem dzVals_length (p_s p_t dz : List ℤ) (n : ℕ) :
    (dzVals p_s p_t dz n).length = dz.length := by
  induction n with
  | zero => rfl
  | succ k ih => rw [dzVals_succ, List.length_set, ih]

/-- The `dz_diff` loop invariant. -/
def dzDiffInv (p_s p_t : Slice I64) (dz0 : List ℤ) (v : Usize)
    (s : Slice I64 × Option learn_port.head.HeadError × Usize) : Prop :=
  s.2.1 = none ∧ s.2.2.val ≤ v.val ∧
    hvals s.1 = dzVals (hvals p_s) (hvals p_t) dz0 s.2.2.val

/-- The extracted `dz_diff` body: one gradient entry is written. -/
theorem dz_diff_loop_body_spec (p_s p_t : Slice I64) (dz0 : List ℤ) (v : Usize)
    (hsl : v.val ≤ p_s.val.length) (htl : v.val ≤ p_t.val.length)
    (hdl : v.val ≤ dz0.length)
    (hdiff : ∀ t < v.val, I64.min ≤ (hvals p_s).getD t 0 - (hvals p_t).getD t 0 ∧
        (hvals p_s).getD t 0 - (hvals p_t).getD t 0 ≤ I64.max)
    (st : Slice I64 × Option learn_port.head.HeadError × Usize)
    (hinv : dzDiffInv p_s p_t dz0 v st) :
    learn_port.head.dz_diff_loop.body p_s p_t v st.1 st.2.1 st.2.2 ⦃ r =>
      match r with
      | ControlFlow.done y => hvals y.2 = dzVals (hvals p_s) (hvals p_t) dz0 v.val ∧ y.1 = none
      | ControlFlow.cont st' => dzDiffInv p_s p_t dz0 v st' ∧
          v.val - st'.2.2.val < v.val - st.2.2.val ⦄ := by
  obtain ⟨dz, fail, i⟩ := st
  simp only [dzDiffInv] at hinv
  obtain ⟨hfail, hle, hdz⟩ := hinv
  by_cases hlt : i < v
  · have hltv : i.val < v.val := (UScalar.lt_equiv i v).mp hlt
    have hlim_s : p_s.val.length ≤ Usize.max := Slice.property p_s
    have hlim_t : p_t.val.length ≤ Usize.max := Slice.property p_t
    have hlen_s : i.val < p_s.val.length := by omega
    have hlen_t : i.val < p_t.val.length := by omega
    have hlen_dz : i.val < dz.val.length := by
      rw [← hvals_length dz]; rw [hdz, dzVals_length]; omega
    have hidx_s := slice_index_getD p_s i hlen_s
    have hidx_t := slice_index_getD p_t i hlen_t
    obtain ⟨d, hdsub, hdv, _⟩ := WP.spec_imp_exists (I128.sub_bv_spec
      (x := core.convert.num.FromI128I64.from (p_s.val.getD i.val default))
      (y := core.convert.num.FromI128I64.from (p_t.val.getD i.val default))
      (from_i64_sub_fits (p_s.val.getD i.val default) (p_t.val.getD i.val default)).1
      (from_i64_sub_fits (p_s.val.getD i.val default) (p_t.val.getD i.val default)).2)
    have hdv' : d.val = (p_s.val.getD i.val default).val - (p_t.val.getD i.val default).val := by
      rw [hdv]
      simp only [core.convert.num.FromI128I64.from_val_eq]
    have hgs : (hvals p_s).getD i.val 0 = (p_s.val.getD i.val default).val :=
      hvals_getD p_s hlen_s
    have hgt : (hvals p_t).getD i.val 0 = (p_t.val.getD i.val default).val :=
      hvals_getD p_t hlen_t
    obtain ⟨g, hg, hgv⟩ := to64_ok d (by
      rw [hdv', ← hgs, ← hgt]
      exact hdiff i.val hltv)
    obtain ⟨dz1, hupd, hdz1⟩ := WP.spec_imp_exists (Slice.update_spec dz i g hlen_dz)
    obtain ⟨i6, hi6, hi6v⟩ := usize_succ_fits hltv (by omega)
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.dz_diff_loop.body p_s p_t v dz fail i =
        ok (ControlFlow.cont (dz1, none, i6)) := by
      unfold learn_port.head.dz_diff_loop.body
      rw [if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, lift, hidx_s, hidx_t, hdsub, hg, hupd, hi6]
      rw [hfail]
      rfl
    rw [hbody]
    simp only [WP.spec_ok, dzDiffInv]
    refine ⟨⟨trivial, by omega, ?_⟩, by omega⟩
    rw [hdz1, hvals_set, hdz, hgv, hdv', ← hgs, ← hgt, hi6v, dzVals_succ]
  · have hbody : learn_port.head.dz_diff_loop.body p_s p_t v dz fail i =
        ok (ControlFlow.done (fail, dz)) := by
      unfold learn_port.head.dz_diff_loop.body
      rw [if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hieq : i.val = v.val := by
      have hvle : v.val ≤ i.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv i v).mpr (by omega))
      omega
    exact ⟨by rw [← hieq, hdz], hfail⟩

/-- **The `dz_diff` loop.** Over the whole row the loop returns `p_s − p_t` entrywise. -/
theorem dz_diff_loop_spec (p_s p_t : Slice I64) (dz : Slice I64) (v : Usize)
    (fail : Option learn_port.head.HeadError) (i : Usize)
    (hsl : v.val ≤ p_s.val.length) (htl : v.val ≤ p_t.val.length)
    (hdl : v.val ≤ dz.val.length)
    (hdiff : ∀ t < v.val, I64.min ≤ (hvals p_s).getD t 0 - (hvals p_t).getD t 0 ∧
        (hvals p_s).getD t 0 - (hvals p_t).getD t 0 ≤ I64.max)
    (hinv0 : dzDiffInv p_s p_t (hvals dz) v (dz, fail, i)) :
    WP.spec (learn_port.head.dz_diff_loop p_s p_t dz v fail i)
      (fun y : Option learn_port.head.HeadError × Slice I64 =>
        hvals y.2 = dzVals (hvals p_s) (hvals p_t) (hvals dz) v.val ∧ y.1 = none) := by
  unfold learn_port.head.dz_diff_loop
  apply loop.spec_decr_nat
    (measure := fun s : Slice I64 × Option learn_port.head.HeadError × Usize => v.val - s.2.2.val)
    (inv := dzDiffInv p_s p_t (hvals dz) v)
    (post := fun y : Option learn_port.head.HeadError × Slice I64 =>
      hvals y.2 = dzVals (hvals p_s) (hvals p_t) (hvals dz) v.val ∧ y.1 = none)
  · intro st hst
    convert dz_diff_loop_body_spec p_s p_t (hvals dz) v hsl htl (by rw [hvals_length]; exact hdl)
      hdiff st hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · exact hinv0

/-! ## 6. `dw_out_fill`: the rounded `W_head` gradient

Specification (`Learn/Grad/Head.lean`): `dwOutVals dw nscale out n` is `out` with entry `t < n` set
to `round(dw[t] / nscale)`. The domain carries the `round_div` and `to64` fits. -/

theorem dwOutVals_succ (dw : List ℤ) (nscale : ℤ) (out : List ℤ) (n : ℕ) :
    dwOutVals dw nscale out (n + 1) =
      (dwOutVals dw nscale out n).set n (rndDiv (dw.getD n 0) nscale) := rfl

theorem dwOutVals_length (dw : List ℤ) (nscale : ℤ) (out : List ℤ) (n : ℕ) :
    (dwOutVals dw nscale out n).length = out.length := by
  induction n with
  | zero => rfl
  | succ k ih => rw [dwOutVals_succ, List.length_set, ih]

/-- The `dw_out_fill` loop invariant. -/
def dwOutInv (dw : Slice I128) (nscale : ℤ) (out0 : List ℤ) (len : Usize)
    (s : Slice I64 × Option learn_port.head.HeadError × Usize) : Prop :=
  s.2.1 = none ∧ s.2.2.val ≤ len.val ∧
    hvals s.1 = dwOutVals (svals dw) nscale out0 s.2.2.val

/-- The extracted `dw_out_fill` body: one rounded gradient entry is written. -/
theorem dw_out_fill_loop_body_spec (dw : Slice I128) (nscale : I128) (out0 : List ℤ) (len : Usize)
    (hdl : len.val ≤ dw.val.length) (hol : len.val ≤ out0.length) (hpos : 0 < nscale.val)
    (hrnd : ∀ t < len.val, rndDivDom ((svals dw).getD t 0) nscale.val ∧
        2 * absZ (Int.tmod ((svals dw).getD t 0) nscale.val) ≤ I128.max)
    (hfit : ∀ t < len.val, I64.min ≤ rndDiv ((svals dw).getD t 0) nscale.val ∧
        rndDiv ((svals dw).getD t 0) nscale.val ≤ I64.max)
    (st : Slice I64 × Option learn_port.head.HeadError × Usize)
    (hinv : dwOutInv dw nscale out0 len st) :
    learn_port.head.dw_out_fill_loop.body dw nscale len st.1 st.2.1 st.2.2 ⦃ r =>
      match r with
      | ControlFlow.done y =>
          hvals y.2 = dwOutVals (svals dw) nscale out0 len.val ∧ y.1 = none
      | ControlFlow.cont st' => dwOutInv dw nscale out0 len st' ∧
          len.val - st'.2.2.val < len.val - st.2.2.val ⦄ := by
  obtain ⟨out, fail, j⟩ := st
  simp only [dwOutInv] at hinv
  obtain ⟨hfail, hle, hout⟩ := hinv
  by_cases hlt : j < len
  · have hltv : j.val < len.val := (UScalar.lt_equiv j len).mp hlt
    have hlim_dw : dw.val.length ≤ Usize.max := Slice.property dw
    have hlen_dw : j.val < dw.val.length := by omega
    have hlen_out : j.val < out.val.length := by
      rw [← hvals_length out]; rw [hout, dwOutVals_length]; omega
    have hidx := slice_index_getD dw j hlen_dw
    have hidx' : (svals dw).getD j.val 0 = (dw.val.getD j.val default).val := svals_getD dw hlen_dw
    obtain ⟨v, hrv, hrvv⟩ := round_div_dom_spec (dw.val.getD j.val default) nscale hpos
      (by rw [← hidx']; exact (hrnd j.val hltv).2)
      (by rw [← hidx']; exact (hrnd j.val hltv).1)
    obtain ⟨x, hx, hxv⟩ := to64_ok v (by rw [hrvv, ← hidx']; exact hfit j.val hltv)
    obtain ⟨out1, hupd, hout1⟩ := WP.spec_imp_exists (Slice.update_spec out j x hlen_out)
    obtain ⟨j1, hj1, hj1v⟩ := usize_succ_fits hltv (by omega)
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.dw_out_fill_loop.body dw nscale len out fail j =
        ok (ControlFlow.cont (out1, none, j1)) := by
      unfold learn_port.head.dw_out_fill_loop.body
      rw [if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, hidx, hrv, hx, hupd, hj1]
      rw [hfail]
      rfl
    rw [hbody]
    simp only [WP.spec_ok, dwOutInv]
    refine ⟨⟨trivial, by omega, ?_⟩, by omega⟩
    rw [hout1, hvals_set, hout, hxv, hrvv, ← hidx', hj1v, dwOutVals_succ]
  · have hbody : learn_port.head.dw_out_fill_loop.body dw nscale len out fail j =
        ok (ControlFlow.done (fail, out)) := by
      unfold learn_port.head.dw_out_fill_loop.body
      rw [if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hjeq : j.val = len.val := by
      have hvle : len.val ≤ j.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv j len).mpr (by omega))
      omega
    exact ⟨by rw [← hjeq, hout], hfail⟩

/-- **The `dw_out_fill` loop.** Over the whole gradient the loop rounds every entry. -/
theorem dw_out_fill_loop_spec (dw : Slice I128) (nscale : I128) (out : Slice I64) (len : Usize)
    (fail : Option learn_port.head.HeadError) (j : Usize)
    (hdl : len.val ≤ dw.val.length) (hol : len.val ≤ out.val.length) (hpos : 0 < nscale.val)
    (hrnd : ∀ t < len.val, rndDivDom ((svals dw).getD t 0) nscale.val ∧
        2 * absZ (Int.tmod ((svals dw).getD t 0) nscale.val) ≤ I128.max)
    (hfit : ∀ t < len.val, I64.min ≤ rndDiv ((svals dw).getD t 0) nscale.val ∧
        rndDiv ((svals dw).getD t 0) nscale.val ≤ I64.max)
    (hinv0 : dwOutInv dw nscale (hvals out) len (out, fail, j)) :
    WP.spec (learn_port.head.dw_out_fill_loop dw nscale out len fail j)
      (fun y : Option learn_port.head.HeadError × Slice I64 =>
        hvals y.2 = dwOutVals (svals dw) nscale (hvals out) len.val ∧ y.1 = none) := by
  unfold learn_port.head.dw_out_fill_loop
  apply loop.spec_decr_nat
    (measure := fun s : Slice I64 × Option learn_port.head.HeadError × Usize => len.val - s.2.2.val)
    (inv := dwOutInv dw nscale (hvals out) len)
    (post := fun y : Option learn_port.head.HeadError × Slice I64 =>
      hvals y.2 = dwOutVals (svals dw) nscale (hvals out) len.val ∧ y.1 = none)
  · intro st hst
    convert dw_out_fill_loop_body_spec dw nscale (hvals out) len hdl
      (by rw [hvals_length]; exact hol) hpos hrnd hfit st hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · exact hinv0

/-! ## 7. `dot_round`: the rounded dot product

Specification (`Learn/Grad/Head.lean`): `dotAux a ab b bb acc n = acc + Σ_{k<n} a[ab+k]·b[bb+k]`. The
loop walks `j < d` accumulating `a[abase+j]·b[bbase+j]`; the public `dot_round` then divides by
`2^32` (`engine.ONE`). The domain carries the `usize`-sum fits, the two slice reaches, the `mul128`
product fit at every index and the running-sum fit. -/

theorem dotAux_add (a : List ℤ) (ab : ℕ) (b : List ℤ) (bb : ℕ) (n : ℕ) (acc x : ℤ) :
    dotAux a ab b bb (acc + x) n = dotAux a ab b bb acc n + x := by
  induction n generalizing acc with
  | zero => rfl
  | succ k ih =>
    change dotAux a ab b bb (acc + x + a.getD (ab + k) 0 * b.getD (bb + k) 0) k =
      dotAux a ab b bb (acc + a.getD (ab + k) 0 * b.getD (bb + k) 0) k + x
    rw [show acc + x + a.getD (ab + k) 0 * b.getD (bb + k) 0 =
        (acc + a.getD (ab + k) 0 * b.getD (bb + k) 0) + x by ring]
    exact ih (acc + a.getD (ab + k) 0 * b.getD (bb + k) 0)

theorem dotAux_succ (a : List ℤ) (ab : ℕ) (b : List ℤ) (bb : ℕ) (acc : ℤ) (n : ℕ) :
    dotAux a ab b bb acc (n + 1) =
      dotAux a ab b bb acc n + a.getD (ab + n) 0 * b.getD (bb + n) 0 := by
  change dotAux a ab b bb (acc + a.getD (ab + n) 0 * b.getD (bb + n) 0) n = _
  rw [dotAux_add]

/-- The `dot_round` loop invariant. -/
def dotRoundInv (a : Slice I64) (abase : Usize) (b : Slice I64) (bbase : Usize)
    (acc0 : ℤ) (d : Usize) (s : I128 × Option learn_port.head.HeadError × Usize) : Prop :=
  s.2.1 = none ∧ s.2.2.val ≤ d.val ∧
    s.1.val = dotAux (hvals a) abase.val (hvals b) bbase.val acc0 s.2.2.val

/-- The extracted `dot_round` body: one product is accumulated. -/
theorem dot_round_loop_body_spec (a : Slice I64) (abase : Usize) (b : Slice I64)
    (bbase : Usize) (d : Usize) (acc0 : ℤ)
    (hab : ∀ t < d.val, abase.val + t ≤ Usize.max)
    (hbb : ∀ t < d.val, bbase.val + t ≤ Usize.max)
    (hla : ∀ t < d.val, abase.val + t < a.val.length)
    (hlb : ∀ t < d.val, bbase.val + t < b.val.length)
    (hprod : ∀ t < d.val,
      I128.min ≤ (hvals a).getD (abase.val + t) 0 * (hvals b).getD (bbase.val + t) 0 ∧
      (hvals a).getD (abase.val + t) 0 * (hvals b).getD (bbase.val + t) 0 ≤ I128.max)
    (hsum : ∀ t ≤ d.val, I128.min ≤ dotAux (hvals a) abase.val (hvals b) bbase.val acc0 t ∧
      dotAux (hvals a) abase.val (hvals b) bbase.val acc0 t ≤ I128.max)
    (st : I128 × Option learn_port.head.HeadError × Usize)
    (hinv : dotRoundInv a abase b bbase acc0 d st) :
    learn_port.head.dot_round_loop.body a abase b bbase d st.1 st.2.1 st.2.2 ⦃ r =>
      match r with
      | ControlFlow.done y => y.1.val = dotAux (hvals a) abase.val (hvals b) bbase.val acc0 d.val ∧
          y.2 = none
      | ControlFlow.cont st' => dotRoundInv a abase b bbase acc0 d st' ∧
          d.val - st'.2.2.val < d.val - st.2.2.val ⦄ := by
  obtain ⟨acc, fail, i⟩ := st
  simp only [dotRoundInv] at hinv
  obtain ⟨hfail, hle, hacc⟩ := hinv
  by_cases hlt : i < d
  · have hltv : i.val < d.val := (UScalar.lt_equiv i d).mp hlt
    obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := abase) (y := i) (by
      have := hab i.val hltv; omega))
    have hlai : i1.val < a.val.length := by rw [hi1v]; exact hla i.val hltv
    have hidxa := slice_index_getD a i1 hlai
    have hgs : (hvals a).getD i1.val 0 = (a.val.getD i1.val default).val := hvals_getD a hlai
    obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists (Usize.add_spec (x := bbase) (y := i) (by
      have := hbb i.val hltv; omega))
    have hlbi : i3.val < b.val.length := by rw [hi3v]; exact hlb i.val hltv
    have hidxb := slice_index_getD b i3 hlbi
    have hgt : (hvals b).getD i3.val 0 = (b.val.getD i3.val default).val := hvals_getD b hlbi
    have hfit : I128.min ≤ (core.convert.num.FromI128I64.from (a.val.getD i1.val default)).val *
          (core.convert.num.FromI128I64.from (b.val.getD i3.val default)).val ∧
        (core.convert.num.FromI128I64.from (a.val.getD i1.val default)).val *
          (core.convert.num.FromI128I64.from (b.val.getD i3.val default)).val ≤ I128.max := by
      rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
        ← hgs, ← hgt, hi1v, hi3v]
      exact hprod i.val hltv
    obtain ⟨term, hmul, htermv⟩ :=
      mul128_ok (core.convert.num.FromI128I64.from (a.val.getD i1.val default))
        (core.convert.num.FromI128I64.from (b.val.getD i3.val default)) hfit
    have heq : acc.val + term.val =
        dotAux (hvals a) abase.val (hvals b) bbase.val acc0 (i.val + 1) := by
      rw [hacc, htermv, core.convert.num.FromI128I64.from_val_eq,
        core.convert.num.FromI128I64.from_val_eq, ← hgs, ← hgt, hi1v, hi3v, dotAux_succ]
    obtain ⟨s, hadd, hsv⟩ := add128_ok acc term (by rw [heq]; exact hsum (i.val + 1) (by omega))
    obtain ⟨i7, hi7, hi7v⟩ := usize_succ_fits hltv (by scalar_tac)
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.dot_round_loop.body a abase b bbase d acc fail i =
        ok (ControlFlow.cont (s, none, i7)) := by
      rw [learn_port.head.dot_round_loop.body, if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, lift, hi1, hi3, hidxa, hidxb, hmul, hadd, hi7]
      rw [hfail]
      rfl
    rw [hbody]
    simp only [WP.spec_ok, dotRoundInv]
    refine ⟨⟨trivial, by omega, ?_⟩, by omega⟩
    rw [hsv, heq, hi7v]
  · have hbody : learn_port.head.dot_round_loop.body a abase b bbase d acc fail i =
        ok (ControlFlow.done (acc, fail)) := by
      rw [learn_port.head.dot_round_loop.body, if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hieq : i.val = d.val := by
      have hvle : d.val ≤ i.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv i d).mpr (by omega))
      omega
    exact ⟨by rw [← hieq, hacc], hfail⟩

/-- **The `dot_round` loop.** From the zero accumulator the loop returns the whole sum. -/
theorem dot_round_loop_spec (a : Slice I64) (abase : Usize) (b : Slice I64) (bbase : Usize)
    (d : Usize) (acc : I128) (fail : Option learn_port.head.HeadError) (i : Usize)
    (hab : ∀ t < d.val, abase.val + t ≤ Usize.max)
    (hbb : ∀ t < d.val, bbase.val + t ≤ Usize.max)
    (hla : ∀ t < d.val, abase.val + t < a.val.length)
    (hlb : ∀ t < d.val, bbase.val + t < b.val.length)
    (hprod : ∀ t < d.val,
      I128.min ≤ (hvals a).getD (abase.val + t) 0 * (hvals b).getD (bbase.val + t) 0 ∧
      (hvals a).getD (abase.val + t) 0 * (hvals b).getD (bbase.val + t) 0 ≤ I128.max)
    (hsum : ∀ t ≤ d.val, I128.min ≤ dotAux (hvals a) abase.val (hvals b) bbase.val acc.val t ∧
      dotAux (hvals a) abase.val (hvals b) bbase.val acc.val t ≤ I128.max)
    (hinv0 : dotRoundInv a abase b bbase acc.val d (acc, fail, i)) :
    WP.spec (learn_port.head.dot_round_loop a abase b bbase d acc fail i)
      (fun y : I128 × Option learn_port.head.HeadError =>
        y.1.val = dotAux (hvals a) abase.val (hvals b) bbase.val acc.val d.val ∧ y.2 = none) := by
  unfold learn_port.head.dot_round_loop
  apply loop.spec_decr_nat
    (measure := fun s : I128 × Option learn_port.head.HeadError × Usize => d.val - s.2.2.val)
    (inv := dotRoundInv a abase b bbase acc.val d)
    (post := fun y : I128 × Option learn_port.head.HeadError =>
      y.1.val = dotAux (hvals a) abase.val (hvals b) bbase.val acc.val d.val ∧ y.2 = none)
  · intro st hst
    convert dot_round_loop_body_spec a abase b bbase d acc.val hab hbb hla hlb hprod hsum st hst
      using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · exact hinv0

/-! ## 8. `dw_fill`: the `W_head` gradient

Specification (`Learn/Grad/Head.lean`): `dwVals dz hidden hbase d dw n` sets entry `n` to
`dw[n] + dz[n/d]·hidden[hbase + n%d]`. The loop walks `t < total` (with `total = d·len(dz)`),
reading `dz[t/d]`, `hidden[hbase + t%d]`, multiplying and adding into `dw[t]`. The domain carries
`d ≠ 0`, the reaches, the per-index product fit and the running-entry fit. -/

theorem dwVals_succ (dz hidden : List ℤ) (hbase d : ℕ) (dw : List ℤ) (n : ℕ) :
    dwVals dz hidden hbase d dw (n + 1) =
      (dwVals dz hidden hbase d dw n).set n
        ((dwVals dz hidden hbase d dw n).getD n 0 + dz.getD (n / d) 0 * hidden.getD (hbase + n % d) 0) :=
  rfl

theorem dwVals_length (dz hidden : List ℤ) (hbase d : ℕ) (dw : List ℤ) (n : ℕ) :
    (dwVals dz hidden hbase d dw n).length = dw.length := by
  induction n with
  | zero => rfl
  | succ k ih => rw [dwVals_succ, List.length_set, ih]

/-- The `dw_fill` loop invariant. -/
def dwFillInv (dz hidden : Slice I64) (hbase d : Usize) (dw0 : List ℤ) (total : Usize)
    (s : Slice I128 × Option learn_port.head.HeadError × Usize) : Prop :=
  s.2.1 = none ∧ s.2.2.val ≤ total.val ∧
    svals s.1 = dwVals (hvals dz) (hvals hidden) hbase.val d.val dw0 s.2.2.val

/-- The extracted `dw_fill` body: one entry of `d_w_head` is accumulated. -/
theorem dw_fill_loop_body_spec (dz hidden : Slice I64) (hbase d : Usize) (dw0 : List ℤ)
    (total : Usize) (st : Slice I128 × Option learn_port.head.HeadError × Usize)
    (hd : d.val ≠ 0) (hdw0len : total.val ≤ dw0.length)
    (hdz : ∀ u < total.val, u / d.val < dz.val.length)
    (hhb : ∀ u < total.val, hbase.val + u % d.val ≤ Usize.max)
    (hhid : ∀ u < total.val, hbase.val + u % d.val < hidden.val.length)
    (hprod : ∀ u < total.val, I128.min ≤
        (hvals dz).getD (u / d.val) 0 * (hvals hidden).getD (hbase.val + u % d.val) 0 ∧
      (hvals dz).getD (u / d.val) 0 * (hvals hidden).getD (hbase.val + u % d.val) 0 ≤ I128.max)
    (hadd : ∀ u < total.val, I128.min ≤
        (dwVals (hvals dz) (hvals hidden) hbase.val d.val dw0 u).getD u 0 +
          (hvals dz).getD (u / d.val) 0 * (hvals hidden).getD (hbase.val + u % d.val) 0 ∧
        (dwVals (hvals dz) (hvals hidden) hbase.val d.val dw0 u).getD u 0 +
          (hvals dz).getD (u / d.val) 0 * (hvals hidden).getD (hbase.val + u % d.val) 0 ≤ I128.max)
    (hinv : dwFillInv dz hidden hbase d dw0 total st) :
    learn_port.head.dw_fill_loop.body dz hidden hbase d total st.1 st.2.1 st.2.2 ⦃ r =>
      match r with
      | ControlFlow.done y =>
          svals y.2 = dwVals (hvals dz) (hvals hidden) hbase.val d.val dw0 total.val ∧ y.1 = none
      | ControlFlow.cont st' => dwFillInv dz hidden hbase d dw0 total st' ∧
          total.val - st'.2.2.val < total.val - st.2.2.val ⦄ := by
  obtain ⟨dw, fail, t⟩ := st
  simp only [dwFillInv] at hinv
  obtain ⟨hfail, hle, hdw0⟩ := hinv
  by_cases hlt : t < total
  · have hltv : t.val < total.val := (UScalar.lt_equiv t total).mp hlt
    obtain ⟨k, hk, hkv⟩ := WP.spec_imp_exists (Usize.div_spec (x := t) (y := d) hd)
    obtain ⟨j, hj, hjv⟩ := WP.spec_imp_exists (Usize.rem_spec (x := t) (y := d) hd)
    have hklen : k.val < dz.val.length := by rw [hkv]; exact hdz t.val hltv
    have hidxz := slice_index_getD dz k hklen
    have hgdz : (hvals dz).getD k.val 0 = (dz.val.getD k.val default).val := hvals_getD dz hklen
    obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := hbase) (y := j) (by
      have := hhb t.val hltv; rw [hjv]; omega))
    have hi2len : i2.val < hidden.val.length := by rw [hi2v, hjv]; exact hhid t.val hltv
    have hidxh := slice_index_getD hidden i2 hi2len
    have hgh : (hvals hidden).getD i2.val 0 = (hidden.val.getD i2.val default).val :=
      hvals_getD hidden hi2len
    have hprodeq : (hvals dz).getD (t.val / d.val) 0 * (hvals hidden).getD (hbase.val + t.val % d.val) 0 =
        (core.convert.num.FromI128I64.from (dz.val.getD k.val default)).val *
          (core.convert.num.FromI128I64.from (hidden.val.getD i2.val default)).val := by
      rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
        ← hgdz, ← hgh, hkv, hi2v, hjv]
    have hfit : I128.min ≤ (core.convert.num.FromI128I64.from (dz.val.getD k.val default)).val *
          (core.convert.num.FromI128I64.from (hidden.val.getD i2.val default)).val ∧
        (core.convert.num.FromI128I64.from (dz.val.getD k.val default)).val *
          (core.convert.num.FromI128I64.from (hidden.val.getD i2.val default)).val ≤ I128.max := by
      rw [← hprodeq]; exact hprod t.val hltv
    obtain ⟨term, hmul, htermv⟩ :=
      mul128_ok (core.convert.num.FromI128I64.from (dz.val.getD k.val default))
        (core.convert.num.FromI128I64.from (hidden.val.getD i2.val default)) hfit
    have hterm_eq : term.val =
        (hvals dz).getD (t.val / d.val) 0 * (hvals hidden).getD (hbase.val + t.val % d.val) 0 := by
      rw [htermv, ← hprodeq]
    have htlen : t.val < dw.val.length := by
      rw [← svals_length, hdw0, dwVals_length]; omega
    have hidxdw := slice_index_getD dw t htlen
    have hgdw : (svals dw).getD t.val 0 = (dw.val.getD t.val default).val := svals_getD dw htlen
    have hsv_eq : (svals dw).getD t.val 0 =
        (dwVals (hvals dz) (hvals hidden) hbase.val d.val dw0 t.val).getD t.val 0 := by rw [hdw0]
    have hadd_i : I128.min ≤ (dw.val.getD t.val default).val + term.val ∧
        (dw.val.getD t.val default).val + term.val ≤ I128.max := by
      rw [← hgdw, hsv_eq, hterm_eq]; exact hadd t.val hltv
    obtain ⟨s, hadd_ok, hsv⟩ := add128_ok (dw.val.getD t.val default) term hadd_i
    obtain ⟨dwx, hupd, hdwx⟩ := WP.spec_imp_exists (Slice.update_spec dw t s htlen)
    obtain ⟨t1, ht1, ht1v⟩ := usize_succ_fits hltv (by scalar_tac)
    have hsv' : s.val =
        (dwVals (hvals dz) (hvals hidden) hbase.val d.val dw0 t.val).getD t.val 0 +
          (hvals dz).getD (t.val / d.val) 0 * (hvals hidden).getD (hbase.val + t.val % d.val) 0 := by
      rw [hsv, ← hgdw, hsv_eq, hterm_eq]
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.dw_fill_loop.body dz hidden hbase d total dw fail t =
        ok (ControlFlow.cont (dwx, none, t1)) := by
      rw [learn_port.head.dw_fill_loop.body, if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, lift, hk, hj, hidxz, hi2, hidxh, hmul, hidxdw,
        hadd_ok, hupd, ht1]
      rw [hfail]
    rw [hbody]
    simp only [WP.spec_ok, dwFillInv]
    refine ⟨⟨trivial, by omega, ?_⟩, by omega⟩
    rw [hdwx, svals_set, hdw0, ht1v, dwVals_succ, hsv']
  · have hbody : learn_port.head.dw_fill_loop.body dz hidden hbase d total dw fail t =
        ok (ControlFlow.done (fail, dw)) := by
      rw [learn_port.head.dw_fill_loop.body, if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hteq : t.val = total.val := by
      have hvle : total.val ≤ t.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv t total).mpr (by omega))
      omega
    exact ⟨by rw [← hteq, hdw0], hfail⟩

/-- **The `dw_fill` loop.** From the accumulated gradient the loop returns `dw + dz ⊗ hidden`. -/
theorem dw_fill_loop_spec (dz hidden : Slice I64) (hbase d : Usize) (dw : Slice I128)
    (total : Usize) (fail : Option learn_port.head.HeadError) (t : Usize)
    (hd : d.val ≠ 0) (htdw : total.val ≤ dw.val.length)
    (hdz : ∀ u < total.val, u / d.val < dz.val.length)
    (hhb : ∀ u < total.val, hbase.val + u % d.val ≤ Usize.max)
    (hhid : ∀ u < total.val, hbase.val + u % d.val < hidden.val.length)
    (hprod : ∀ u < total.val, I128.min ≤
        (hvals dz).getD (u / d.val) 0 * (hvals hidden).getD (hbase.val + u % d.val) 0 ∧
      (hvals dz).getD (u / d.val) 0 * (hvals hidden).getD (hbase.val + u % d.val) 0 ≤ I128.max)
    (hadd : ∀ u < total.val, I128.min ≤
        (dwVals (hvals dz) (hvals hidden) hbase.val d.val (svals dw) u).getD u 0 +
          (hvals dz).getD (u / d.val) 0 * (hvals hidden).getD (hbase.val + u % d.val) 0 ∧
        (dwVals (hvals dz) (hvals hidden) hbase.val d.val (svals dw) u).getD u 0 +
          (hvals dz).getD (u / d.val) 0 * (hvals hidden).getD (hbase.val + u % d.val) 0 ≤ I128.max)
    (hinv0 : dwFillInv dz hidden hbase d (svals dw) total (dw, fail, t)) :
    WP.spec (learn_port.head.dw_fill_loop dz hidden hbase d dw total fail t)
      (fun y : Option learn_port.head.HeadError × Slice I128 =>
        svals y.2 = dwVals (hvals dz) (hvals hidden) hbase.val d.val (svals dw) total.val ∧
          y.1 = none) := by
  unfold learn_port.head.dw_fill_loop
  apply loop.spec_decr_nat
    (measure := fun s : Slice I128 × Option learn_port.head.HeadError × Usize =>
      total.val - s.2.2.val)
    (inv := dwFillInv dz hidden hbase d (svals dw) total)
    (post := fun y : Option learn_port.head.HeadError × Slice I128 =>
      svals y.2 = dwVals (hvals dz) (hvals hidden) hbase.val d.val (svals dw) total.val ∧
        y.1 = none)
  · intro st hst
    convert dw_fill_loop_body_spec dz hidden hbase d (svals dw) total st hd
      (by rw [svals_length]; exact htdw) hdz hhb hhid hprod hadd hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · exact hinv0

/-! ## 9. `tv_scan`: the sign vector and the two total-variation accumulators

Specification (`Learn/Grad/Head.lean`): `sgVals p_s p_t sg n` sets `sg[i] = sgn(p_s[i] − p_t[i])`,
`tvAbsAux p_s p_t a n = a + Σ_{i<n} |p_s[i] − p_t[i]|` and
`tvSignedAux p_s p_t s n = s + Σ_{i<n} p_s[i]·sgn(p_s[i] − p_t[i])`. The loop walks `i < v` writing
the sign vector in place and accumulating both sums. -/

theorem tvAbsAux_add (p_s p_t : List ℤ) (n : ℕ) (a x : ℤ) :
    tvAbsAux p_s p_t (a + x) n = tvAbsAux p_s p_t a n + x := by
  induction n generalizing a with
  | zero => rfl
  | succ k ih =>
    change tvAbsAux p_s p_t (a + x + absZ (p_s.getD k 0 - p_t.getD k 0)) k =
      tvAbsAux p_s p_t (a + absZ (p_s.getD k 0 - p_t.getD k 0)) k + x
    rw [show a + x + absZ (p_s.getD k 0 - p_t.getD k 0) =
        (a + absZ (p_s.getD k 0 - p_t.getD k 0)) + x by ring]
    exact ih (a + absZ (p_s.getD k 0 - p_t.getD k 0))

theorem tvAbsAux_succ (p_s p_t : List ℤ) (a : ℤ) (n : ℕ) :
    tvAbsAux p_s p_t a (n + 1) = tvAbsAux p_s p_t a n + absZ (p_s.getD n 0 - p_t.getD n 0) := by
  change tvAbsAux p_s p_t (a + absZ (p_s.getD n 0 - p_t.getD n 0)) n = _
  rw [tvAbsAux_add]

theorem tvSignedAux_add (p_s p_t : List ℤ) (n : ℕ) (s x : ℤ) :
    tvSignedAux p_s p_t (s + x) n = tvSignedAux p_s p_t s n + x := by
  induction n generalizing s with
  | zero => rfl
  | succ k ih =>
    change tvSignedAux p_s p_t (s + x + p_s.getD k 0 * sgnZ (p_s.getD k 0 - p_t.getD k 0)) k =
      tvSignedAux p_s p_t (s + p_s.getD k 0 * sgnZ (p_s.getD k 0 - p_t.getD k 0)) k + x
    rw [show s + x + p_s.getD k 0 * sgnZ (p_s.getD k 0 - p_t.getD k 0) =
        (s + p_s.getD k 0 * sgnZ (p_s.getD k 0 - p_t.getD k 0)) + x by ring]
    exact ih (s + p_s.getD k 0 * sgnZ (p_s.getD k 0 - p_t.getD k 0))

theorem tvSignedAux_succ (p_s p_t : List ℤ) (s : ℤ) (n : ℕ) :
    tvSignedAux p_s p_t s (n + 1) =
      tvSignedAux p_s p_t s n + p_s.getD n 0 * sgnZ (p_s.getD n 0 - p_t.getD n 0) := by
  change tvSignedAux p_s p_t (s + p_s.getD n 0 * sgnZ (p_s.getD n 0 - p_t.getD n 0)) n = _
  rw [tvSignedAux_add]

theorem sgVals_succ (p_s p_t : List ℤ) (sg : List ℤ) (n : ℕ) :
    sgVals p_s p_t sg (n + 1) =
      (sgVals p_s p_t sg n).set n (sgnZ (p_s.getD n 0 - p_t.getD n 0)) := rfl

theorem sgVals_length (p_s p_t : List ℤ) (sg : List ℤ) (n : ℕ) :
    (sgVals p_s p_t sg n).length = sg.length := by
  induction n with
  | zero => rfl
  | succ k ih => rw [sgVals_succ, List.length_set, ih]

/-- The `tv_scan` loop invariant. -/
def tvScanInv (p_s p_t : Slice I64) (sg0 : List ℤ) (abs0 signed0 : ℤ) (v : Usize)
    (s : Slice I128 × I128 × I128 × Option learn_port.head.HeadError × Usize) : Prop :=
  s.2.2.2.1 = none ∧ s.2.2.2.2.val ≤ v.val ∧
    svals s.1 = sgVals (hvals p_s) (hvals p_t) sg0 s.2.2.2.2.val ∧
    s.2.1.val = tvAbsAux (hvals p_s) (hvals p_t) abs0 s.2.2.2.2.val ∧
    s.2.2.1.val = tvSignedAux (hvals p_s) (hvals p_t) signed0 s.2.2.2.2.val

/-- The extracted `tv_scan` body: one sign and both partial sums are updated. -/
theorem tv_scan_loop_body_spec (p_s p_t : Slice I64) (sg0 : List ℤ) (abs0 signed0 : ℤ)
    (v : Usize) (st : Slice I128 × I128 × I128 × Option learn_port.head.HeadError × Usize)
    (hsl : v.val ≤ p_s.val.length) (htl : v.val ≤ p_t.val.length) (hsg0len : v.val ≤ sg0.length)
    (habsfit : ∀ t ≤ v.val, I128.min ≤ tvAbsAux (hvals p_s) (hvals p_t) abs0 t ∧
        tvAbsAux (hvals p_s) (hvals p_t) abs0 t ≤ I128.max)
    (hsignedfit : ∀ t ≤ v.val, I128.min ≤ tvSignedAux (hvals p_s) (hvals p_t) signed0 t ∧
        tvSignedAux (hvals p_s) (hvals p_t) signed0 t ≤ I128.max)
    (hprod : ∀ t < v.val, I128.min ≤
        (hvals p_s).getD t 0 * sgnZ ((hvals p_s).getD t 0 - (hvals p_t).getD t 0) ∧
      (hvals p_s).getD t 0 * sgnZ ((hvals p_s).getD t 0 - (hvals p_t).getD t 0) ≤ I128.max)
    (hinv : tvScanInv p_s p_t sg0 abs0 signed0 v st) :
    learn_port.head.tv_scan_loop.body p_s p_t v st.1 st.2.1 st.2.2.1 st.2.2.2.1 st.2.2.2.2 ⦃ r =>
      match r with
      | ControlFlow.done y =>
          svals y.1 = sgVals (hvals p_s) (hvals p_t) sg0 v.val ∧
          y.2.1.val = tvAbsAux (hvals p_s) (hvals p_t) abs0 v.val ∧
          y.2.2.1.val = tvSignedAux (hvals p_s) (hvals p_t) signed0 v.val ∧
          y.2.2.2 = none
      | ControlFlow.cont st' => tvScanInv p_s p_t sg0 abs0 signed0 v st' ∧
          v.val - st'.2.2.2.2.val < v.val - st.2.2.2.2.val ⦄ := by
  obtain ⟨sg, abs_sum, signed, fail, i⟩ := st
  simp only [tvScanInv] at hinv
  obtain ⟨hfail, hle, hsg, habs, hsigned⟩ := hinv
  by_cases hlt : i < v
  · have hltv : i.val < v.val := (UScalar.lt_equiv i v).mp hlt
    have hlen_s : i.val < p_s.val.length := by omega
    have hlen_t : i.val < p_t.val.length := by omega
    have hidxs := slice_index_getD p_s i hlen_s
    have hidxt := slice_index_getD p_t i hlen_t
    have hgs : (hvals p_s).getD i.val 0 = (p_s.val.getD i.val default).val :=
      hvals_getD p_s hlen_s
    have hgt : (hvals p_t).getD i.val 0 = (p_t.val.getD i.val default).val :=
      hvals_getD p_t hlen_t
    obtain ⟨diff, hsub, hdv, _⟩ := WP.spec_imp_exists (I128.sub_bv_spec
      (x := core.convert.num.FromI128I64.from (p_s.val.getD i.val default))
      (y := core.convert.num.FromI128I64.from (p_t.val.getD i.val default))
      (from_i64_sub_fits (p_s.val.getD i.val default) (p_t.val.getD i.val default)).1
      (from_i64_sub_fits (p_s.val.getD i.val default) (p_t.val.getD i.val default)).2)
    have hdvv : diff.val = (p_s.val.getD i.val default).val - (p_t.val.getD i.val default).val := by
      rw [hdv]; simp only [core.convert.num.FromI128I64.from_val_eq]
    obtain ⟨i5, hsgn, hi5v⟩ := sgn_spec diff
    have hi5eq : i5.val = sgnZ ((hvals p_s).getD i.val 0 - (hvals p_t).getD i.val 0) := by
      rw [hi5v, hdvv, ← hgs, ← hgt]
    have hsglen : i.val < sg.val.length := by
      rw [← svals_length, hsg, sgVals_length]; omega
    obtain ⟨sg1, hupd, hsg1⟩ := WP.spec_imp_exists (Slice.update_spec sg i i5 hsglen)
    have hmin : I128.min < diff.val := by
      rw [hdvv, i128_min_eq]
      have h1 := i64_bounds (p_s.val.getD i.val default)
      have h2 := i64_bounds (p_t.val.getD i.val default)
      rw [i64_min_eq, i64_max_eq] at h1 h2
      omega
    obtain ⟨i6, habsok, hi6v⟩ := abs128_spec diff hmin
    have haux_eq : abs_sum.val + i6.val =
        tvAbsAux (hvals p_s) (hvals p_t) abs0 (i.val + 1) := by
      rw [habs, hi6v, hdvv, ← hgs, ← hgt, tvAbsAux_succ]
    obtain ⟨abs1, haddok, habs1⟩ := add128_ok abs_sum i6 (by
      rw [haux_eq]; exact habsfit (i.val + 1) (by omega))
    have hi7eq : (core.convert.num.FromI128I64.from (p_s.val.getD i.val default)).val =
        (hvals p_s).getD i.val 0 := by
      rw [core.convert.num.FromI128I64.from_val_eq, ← hgs]
    have hlen8 : i.val < sg1.val.length := by
      rw [slice_update_length sg sg1 i i5 hupd]; exact hsglen
    have hidx8 := slice_index_getD sg1 i hlen8
    have h8v : (sg1.val.getD i.val default).val = i5.val := by
      rw [slice_update_val sg sg1 i i5 hupd,
        hl_getD_set sg.val i.val i.val i5 hsglen default, if_pos rfl]
    have hprodfit : I128.min ≤
          (core.convert.num.FromI128I64.from (p_s.val.getD i.val default)).val *
            (sg1.val.getD i.val default).val ∧
        (core.convert.num.FromI128I64.from (p_s.val.getD i.val default)).val *
            (sg1.val.getD i.val default).val ≤ I128.max := by
      rw [hi7eq, h8v, hi5eq]; exact hprod i.val hltv
    obtain ⟨term, hmulok, htermv⟩ :=
      mul128_ok (core.convert.num.FromI128I64.from (p_s.val.getD i.val default))
        (sg1.val.getD i.val default) hprodfit
    have hsign_eq : signed.val + term.val =
        tvSignedAux (hvals p_s) (hvals p_t) signed0 (i.val + 1) := by
      rw [hsigned, htermv, hi7eq, h8v, hi5eq, tvSignedAux_succ]
    obtain ⟨signed1, haddok1, hsigned1⟩ := add128_ok signed term (by
      rw [hsign_eq]; exact hsignedfit (i.val + 1) (by omega))
    obtain ⟨i9, hi9, hi9v⟩ := usize_succ_fits hltv (by scalar_tac)
    have hb : core.option.Option.is_none fail = true := by rw [hfail]; rfl
    have hbody : learn_port.head.tv_scan_loop.body p_s p_t v sg abs_sum signed fail i =
        ok (ControlFlow.cont (sg1, abs1, signed1, none, i9)) := by
      rw [learn_port.head.tv_scan_loop.body, if_pos hlt]
      simp only [hb, ↓reduceIte, bind_tc_ok, lift, hidxs, hidxt, hsub, hsgn, hupd, habsok,
        haddok, hidx8, hmulok, haddok1, hi9]
      rw [hfail]
      rfl
    rw [hbody]
    simp only [WP.spec_ok, tvScanInv]
    refine ⟨⟨trivial, by omega, ?_, ?_, ?_⟩, by omega⟩
    · rw [hsg1, svals_set, hsg, hi9v, sgVals_succ, hi5eq]
    · rw [habs1, haux_eq, hi9v]
    · rw [hsigned1, hsign_eq, hi9v]
  · have hbody : learn_port.head.tv_scan_loop.body p_s p_t v sg abs_sum signed fail i =
        ok (ControlFlow.done (sg, abs_sum, signed, fail)) := by
      rw [learn_port.head.tv_scan_loop.body, if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    have hieq : i.val = v.val := by
      have hvle : v.val ≤ i.val := by
        by_contra hc
        exact hlt ((UScalar.lt_equiv i v).mpr (by omega))
      omega
    exact ⟨by rw [← hieq, hsg], by rw [← hieq, habs], by rw [← hieq, hsigned], hfail⟩

/-- **The `tv_scan` loop.** From the zero accumulators the loop returns both whole-row sums. -/
theorem tv_scan_loop_spec (p_s p_t : Slice I64) (sg : Slice I128) (v : Usize)
    (abs_sum signed : I128) (fail : Option learn_port.head.HeadError) (i : Usize)
    (hsl : v.val ≤ p_s.val.length) (htl : v.val ≤ p_t.val.length) (hsgl : v.val ≤ sg.val.length)
    (habsfit : ∀ t ≤ v.val,
      I128.min ≤ tvAbsAux (hvals p_s) (hvals p_t) 0 t ∧
      tvAbsAux (hvals p_s) (hvals p_t) 0 t ≤ I128.max)
    (hsignedfit : ∀ t ≤ v.val,
      I128.min ≤ tvSignedAux (hvals p_s) (hvals p_t) 0 t ∧
      tvSignedAux (hvals p_s) (hvals p_t) 0 t ≤ I128.max)
    (hprod : ∀ t < v.val, I128.min ≤
        (hvals p_s).getD t 0 * sgnZ ((hvals p_s).getD t 0 - (hvals p_t).getD t 0) ∧
      (hvals p_s).getD t 0 * sgnZ ((hvals p_s).getD t 0 - (hvals p_t).getD t 0) ≤ I128.max)
    (hinv0 : tvScanInv p_s p_t (svals sg) 0 0 v (sg, abs_sum, signed, fail, i)) :
    WP.spec (learn_port.head.tv_scan_loop p_s p_t sg v abs_sum signed fail i)
      (fun y : Slice I128 × I128 × I128 × Option learn_port.head.HeadError =>
        svals y.1 = sgVals (hvals p_s) (hvals p_t) (svals sg) v.val ∧
        y.2.1.val = tvAbsAux (hvals p_s) (hvals p_t) 0 v.val ∧
        y.2.2.1.val = tvSignedAux (hvals p_s) (hvals p_t) 0 v.val ∧ y.2.2.2 = none) := by
  unfold learn_port.head.tv_scan_loop
  apply loop.spec_decr_nat
    (measure := fun s : Slice I128 × I128 × I128 × Option learn_port.head.HeadError × Usize =>
      v.val - s.2.2.2.2.val)
    (inv := tvScanInv p_s p_t (svals sg) 0 0 v)
    (post := fun y : Slice I128 × I128 × I128 × Option learn_port.head.HeadError =>
      svals y.1 = sgVals (hvals p_s) (hvals p_t) (svals sg) v.val ∧
      y.2.1.val = tvAbsAux (hvals p_s) (hvals p_t) 0 v.val ∧
      y.2.2.1.val = tvSignedAux (hvals p_s) (hvals p_t) 0 v.val ∧ y.2.2.2 = none)
  · intro st hst
    convert tv_scan_loop_body_spec p_s p_t (svals sg) 0 0 v st hsl htl
      (by rw [svals_length]; exact hsgl) habsfit hsignedfit hprod hst using 1
    funext r
    cases r with
    | done y => rfl
    | cont st' => rfl
  · exact hinv0

end apeiron_head
