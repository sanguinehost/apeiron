import Learn.Grad.DenseNorm

/-!
# Refinement of `src/learn/grad/dense.rs::rmsnorm_bwd` loop 1

`DenseNorm.lean` closes `silu_deriv`, `swiglu_bwd`, `rmsnorm_fwd`, `inv_rms` and the `rmsnorm_bwd`
loop-0 accumulator. This module carries the `rmsnorm_bwd` loop-1 refinement (the `r`/`S` output
loop) and its `ℤ` specification link, over the same extracted body (`ApeironWord.lean`).
-/

namespace apeiron_grad

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result
open apeiron_norm
open apeiron_word
open learn_port.dense

/-- The extracted constant `1#i128 <<< 63#i32 = 2^63`. -/
theorem norm_I128_shiftLeft_63 : (1#i128 <<< 63#i32) = ok (9223372036854775808#i128) := by
  have h : (1#i128 <<< 63#i32) ⦃ z => z = (9223372036854775808#i128) ⦄ := by
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted `x >>> 64#i32` is the `BitVec` arithmetic shift right by 64. -/
theorem norm_I128_shiftRight_64_eq (x : I128) :
    (x >>> 64#i32) = ok (⟨x.bv.sshiftRight 64⟩ : I128) := by
  show IScalar.shiftRight x (64#i32).toNat = ok _
  have h64 : (64#i32).toNat = 64 := by decide
  rw [h64]
  unfold IScalar.shiftRight
  rw [if_pos (by norm_num [IScalarTy.numBits, I128.numBits])]

/-- The extracted `x >>> 64#i32` is `Int` floor division by `2^64`. -/
theorem norm_I128_shiftRight_64_val (x : I128) :
    ∃ y : I128, (x >>> 64#i32) = ok y ∧ y.val = x.val / 2^64 := by
  refine ⟨⟨x.bv.sshiftRight 64⟩, norm_I128_shiftRight_64_eq x, ?_⟩
  show (⟨x.bv.sshiftRight 64⟩ : I128).val = x.val / 2^64
  rw [show (⟨x.bv.sshiftRight 64⟩ : I128).val = (x.bv.sshiftRight 64).toInt from rfl]
  rw [BitVec.toInt_sshiftRight, Int.shiftRight_eq_div_pow]
  norm_num

/-- The `dx` prefix of `rmsnorm_bwd` loop 1: `dx_j = bwdDx r a_j (bwdT x_j r r2) s nd`, reading the
`a_j` back from the loop-0 vector. -/
def bwdDxVals (xs : List I64) (a : alloc.vec.Vec I64) (r r2 s nd : ℤ) : ℕ → List ℤ
  | 0 => []
  | (j + 1) => bwdDxVals xs a r r2 s nd j ++
      [bwdDx r (a.val.getD j default).val (bwdT (xs.getD j default).val r r2) s nd]

theorem bwdDxVals_step (xs : List I64) (a : alloc.vec.Vec I64) (r r2 s nd : ℤ) (j : ℕ) :
    bwdDxVals xs a r r2 s nd (j + 1) = bwdDxVals xs a r r2 s nd j ++
      [bwdDx r (a.val.getD j default).val (bwdT (xs.getD j default).val r r2) s nd] := rfl

theorem bwdDxVals_length (xs : List I64) (a : alloc.vec.Vec I64) (r r2 s nd : ℤ) (j : ℕ) :
    (bwdDxVals xs a r r2 s nd j).length = j := by
  induction j with
  | zero => rfl
  | succ k ih => rw [bwdDxVals_step, List.length_append, List.length_singleton, ih]

/-- The `dg` prefix of `rmsnorm_bwd` loop 1: `dg_j = bwdDg dy_j x_j r`. -/
def bwdDgVals (dys xs : List I64) (r : ℤ) : ℕ → List ℤ
  | 0 => []
  | (j + 1) => bwdDgVals dys xs r j ++
      [bwdDg (dys.getD j default).val (xs.getD j default).val r]

theorem bwdDgVals_step (dys xs : List I64) (r : ℤ) (j : ℕ) :
    bwdDgVals dys xs r (j + 1) = bwdDgVals dys xs r j ++
      [bwdDg (dys.getD j default).val (xs.getD j default).val r] := rfl

theorem bwdDgVals_length (dys xs : List I64) (r : ℤ) (j : ℕ) :
    (bwdDgVals dys xs r j).length = j := by
  induction j with
  | zero => rfl
  | succ k ih => rw [bwdDgVals_step, List.length_append, List.length_singleton, ih]

/-- The value-list `dx` prefix of `rmsnorm_bwd` loop 1, for the `a` vector carried only through its
`ℤ` values: `dx_j = bwdDx r a_j (bwdT x_j r r2) s nd`, with `a_j` read from `as`. The vector form is
`bwdDxVals`; the two agree when `as` is the vector's value list. -/
def bwdDxValsL (xs : List I64) (vs : List ℤ) (r r2 s nd : ℤ) : ℕ → List ℤ
  | 0 => []
  | (j + 1) => bwdDxValsL xs vs r r2 s nd j ++
      [bwdDx r (vs.getD j (default : I64).val) (bwdT (xs.getD j default).val r r2) s nd]

theorem bwdDxValsL_step (xs : List I64) (vs : List ℤ) (r r2 s nd : ℤ) (j : ℕ) :
    bwdDxValsL xs vs r r2 s nd (j + 1) = bwdDxValsL xs vs r r2 s nd j ++
      [bwdDx r (vs.getD j (default : I64).val) (bwdT (xs.getD j default).val r r2) s nd] := rfl

theorem bwdDxValsL_length (xs : List I64) (vs : List ℤ) (r r2 s nd : ℤ) (j : ℕ) :
    (bwdDxValsL xs vs r r2 s nd j).length = j := by
  induction j with
  | zero => rfl
  | succ k ih => rw [bwdDxValsL_step, List.length_append, List.length_singleton, ih]

/-- The `a`-vector prefix equals its value-list prefix. -/
theorem bwdDxVals_eq_L (xs : List I64) (a : alloc.vec.Vec I64) (r r2 s nd : ℤ) (j : ℕ) :
    bwdDxVals xs a r r2 s nd j =
      bwdDxValsL xs (a.val.map (fun y => y.val)) r r2 s nd j := by
  induction j with
  | zero => rfl
  | succ k ih =>
    have hb : (a.val.map (fun y : I64 => y.val)).getD k (default : I64).val =
        (a.val.getD k default).val :=
      List.getD_map a.val default (n := k) (fun y : I64 => y.val)
    rw [bwdDxVals_step, bwdDxValsL_step, ih, hb]

/-- One `rmsnorm_bwd_loop1` iteration with `ok1 = true`, on the domain where every callee of the
iteration returns: the `a` read, the `mul_round`/contract chain forming `t`, `q`, `d`, `g` and the
two pushes all succeed, and their values are the `ℤ` specification values. -/
theorem rmsnorm_bwd_loop1_body_step
    (x dy : Slice I64) (n : Usize) (nd r r2 : I64) (a : alloc.vec.Vec I64) (s : I64)
    (dx dg : alloc.vec.Vec I64) (k : Usize)
    (hx : x.val.length = n.val) (hdy : dy.val.length = n.val)
    (halen : a.val.length = n.val)
    (hk : k.val < n.val) (hnumax : k.val + 1 ≤ Usize.max) (hkbound : k.val + 1 ≤ U32.max)
    (hdxlen : dx.val.length = k.val) (hdglen : dg.val.length = k.val)
    (hdx : dx.val.map (fun y => y.val) = bwdDxVals x.val a r.val r2.val s.val nd.val k.val)
    (hdg : dg.val.map (fun y => y.val) = bwdDgVals dy.val x.val r.val k.val)
    (pv tv ts qv dv v1 gv : I64) (xp xpr sum : I128)
    (hp : learn_port.dense.mul_round r (a.val.getD k.val default) = ok (some pv))
    (hpv : pv.val = bwdP r.val (a.val.getD k.val default).val)
    (hxm : I128.checked_mul (core.convert.num.FromI128I64.from (x.val.getD k.val default))
        (core.convert.num.FromI128I64.from r2) = some xp)
    (hxmr : xp.checked_mul (core.convert.num.FromI128I64.from r) = some xpr)
    (hadd : xpr.checked_add (9223372036854775808#i128) = some sum)
    (hnarrow : solve_port.q32.narrow (⟨sum.bv.sshiftRight 64⟩ : I128) = ok (some tv))
    (htv : tv.val = bwdT (x.val.getD k.val default).val r.val r2.val)
    (hqm : learn_port.dense.mul_round tv s = ok (some ts))
    (hdiv : learn_port.dense.div_round ts nd = ok (some qv))
    (hqv : qv.val = bwdQ tv.val s.val nd.val)
    (hsub : pv.checked_sub qv = some dv)
    (hdv : dv.val = pv.val - qv.val)
    (hg1 : learn_port.dense.mul_round (dy.val.getD k.val default) (x.val.getD k.val default)
        = ok (some v1))
    (hg2 : learn_port.dense.mul_round v1 r = ok (some gv))
    (hgv : gv.val = bwdDg (dy.val.getD k.val default).val (x.val.getD k.val default).val r.val) :
    ∃ dx2 dg2 k2, learn_port.dense.rmsnorm_bwd_loop1.body x dy n nd r r2 a s k true dx dg
        = ok (ControlFlow.cont (k2, true, dx2, dg2)) ∧
      k2.val = k.val + 1 ∧
      dx2.val.map (fun y => y.val) = bwdDxVals x.val a r.val r2.val s.val nd.val k2.val ∧
      dg2.val.map (fun y => y.val) = bwdDgVals dy.val x.val r.val k2.val := by
  have hkg : k.val < x.val.length := by omega
  have hkd : k.val < dy.val.length := by omega
  have hka : k.val < a.val.length := by omega
  have hix : Slice.index_usize x k = ok (x.val.getD k.val default) :=
    apeiron_gram.slice_index_getD x k hkg
  have hiy : Slice.index_usize dy k = ok (dy.val.getD k.val default) :=
    apeiron_gram.slice_index_getD dy k hkd
  have haind : alloc.vec.Vec.index_usize a k = ok (a.val.getD k.val default) := by
    obtain ⟨v, hv, hvv⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec a k hka)
    rw [hv, hvv, List.getD_eq_getElem a.val default hka]
  have hdxb : dx.val.length + 1 ≤ U32.max := by rw [hdxlen]; exact hkbound
  have hdgb : dg.val.length + 1 ≤ U32.max := by rw [hdglen]; exact hkbound
  obtain ⟨dx2, hpushdx, hdx2v⟩ := alloc.vec.Vec.push_val dx dv hdxb
  obtain ⟨dg2, hpushdg, hdg2v⟩ := alloc.vec.Vec.push_val dg gv hdgb
  obtain ⟨k2, hk2, hk2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := k) (y := 1#usize) hnumax)
  have hkn : k < n := (UScalar.lt_equiv k n).2 hk
  have hdv' : dv.val = bwdDx r.val (a.val.getD k.val default).val
      (bwdT (x.val.getD k.val default).val r.val r2.val) s.val nd.val := by
    rw [hdv, hpv, hqv, htv]; rfl
  refine ⟨dx2, dg2, k2, ?_, hk2v, ?_, ?_⟩
  · rw [learn_port.dense.rmsnorm_bwd_loop1.body, if_pos hkn, if_pos rfl]
    rw [alloc.vec.Vec.index_slice_index, haind, hix, hiy]
    simp only [norm_I128_shiftLeft_63, norm_I128_shiftRight_64_eq, hp, hxm, hxmr, hadd, hnarrow,
      hqm, hdiv, hsub, hg1, hg2, hpushdx, hpushdg, hk2, bind_tc_ok, lift]
    rfl
  · rw [hdx2v]
    simp only [List.map_append, List.map_cons, List.map_nil]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hk2v, h1, bwdDxVals_step, ← hdx, hdv']
  · rw [hdg2v]
    simp only [List.map_append, List.map_cons, List.map_nil]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hk2v, h1, bwdDgVals_step, ← hdg, hgv]

/-- Specification of `rmsnorm_bwd_loop1` on the all-succeed domain: every per-index callee chain
returns its `ℤ` specification value, and the two output prefixes are the specification prefixes. -/
theorem rmsnorm_bwd_loop1_spec
    (x dy : Slice I64) (n : Usize) (nd r r2 : I64) (a : alloc.vec.Vec I64) (s : I64)
    (dx0 dg0 : alloc.vec.Vec I64)
    (hx : x.val.length = n.val) (hdy : dy.val.length = n.val)
    (halen : a.val.length = n.val)
    (hmax : n.val ≤ Usize.max) (hnmax : n.val + 1 ≤ U32.max)
    (hstep : ∀ k, k < n.val →
        ∃ (pv tv ts qv dv v1 gv : I64) (xp xpr sum : I128),
          learn_port.dense.mul_round r (a.val.getD k default) = ok (some pv) ∧
          pv.val = bwdP r.val (a.val.getD k default).val ∧
          I128.checked_mul (core.convert.num.FromI128I64.from (x.val.getD k default))
            (core.convert.num.FromI128I64.from r2) = some xp ∧
          xp.checked_mul (core.convert.num.FromI128I64.from r) = some xpr ∧
          xpr.checked_add (9223372036854775808#i128) = some sum ∧
          solve_port.q32.narrow (⟨sum.bv.sshiftRight 64⟩ : I128) = ok (some tv) ∧
          tv.val = bwdT (x.val.getD k default).val r.val r2.val ∧
          learn_port.dense.mul_round tv s = ok (some ts) ∧
          learn_port.dense.div_round ts nd = ok (some qv) ∧
          qv.val = bwdQ tv.val s.val nd.val ∧
          pv.checked_sub qv = some dv ∧
          dv.val = pv.val - qv.val ∧
          learn_port.dense.mul_round (dy.val.getD k default) (x.val.getD k default)
            = ok (some v1) ∧
          learn_port.dense.mul_round v1 r = ok (some gv) ∧
          gv.val = bwdDg (dy.val.getD k default).val (x.val.getD k default).val r.val)
    (hdx0 : dx0.val.map (fun y => y.val) = bwdDxVals x.val a r.val r2.val s.val nd.val 0)
    (hdg0 : dg0.val.map (fun y => y.val) = bwdDgVals dy.val x.val r.val 0) :
    WP.spec (learn_port.dense.rmsnorm_bwd_loop1 x dy n nd r r2 a 0#usize true s dx0 dg0)
      (fun b => b.1 = true ∧
        b.2.1.val.map (fun y => y.val) = bwdDxVals x.val a r.val r2.val s.val nd.val n.val ∧
        b.2.2.val.map (fun y => y.val) = bwdDgVals dy.val x.val r.val n.val) := by
  unfold learn_port.dense.rmsnorm_bwd_loop1
  apply loop.spec_decr_nat
    (measure := fun st : Usize × Bool × alloc.vec.Vec I64 × alloc.vec.Vec I64 =>
      n.val - st.1.val)
    (inv := fun st => st.1.val ≤ n.val ∧ st.2.1 = true ∧
      st.2.2.1.val.map (fun y => y.val) = bwdDxVals x.val a r.val r2.val s.val nd.val st.1.val ∧
      st.2.2.2.val.map (fun y => y.val) = bwdDgVals dy.val x.val r.val st.1.val)
    (post := fun b : Bool × alloc.vec.Vec I64 × alloc.vec.Vec I64 =>
      b.1 = true ∧
      b.2.1.val.map (fun y => y.val) = bwdDxVals x.val a r.val r2.val s.val nd.val n.val ∧
      b.2.2.val.map (fun y => y.val) = bwdDgVals dy.val x.val r.val n.val)
  · rintro ⟨k', ok', dx', dg'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hkle, hok, hdx, hdg⟩ := hinv
    rw [hok]
    by_cases hklt : k'.val < n.val
    · have hnumax : k'.val + 1 ≤ Usize.max := by omega
      have hkbound : k'.val + 1 ≤ U32.max := by omega
      have hdxlen : dx'.val.length = k'.val := by
        have hh := congrArg List.length hdx
        simp only [List.length_map, bwdDxVals_length] at hh
        exact hh
      have hdglen : dg'.val.length = k'.val := by
        have hh := congrArg List.length hdg
        simp only [List.length_map, bwdDgVals_length] at hh
        exact hh
      obtain ⟨pv, tv, ts, qv, dv, v1, gv, xp, xpr, sum, hp, hpv, hxm, hxmr, hadd, hnarrow,
        htv, hqm, hdiv, hqv, hsub, hdv, hg1, hg2, hgv⟩ := hstep k'.val hklt
      obtain ⟨dx2, dg2, k2, hexec, hk2v, hdx2, hdg2⟩ :=
        rmsnorm_bwd_loop1_body_step x dy n nd r r2 a s dx' dg' k' hx hdy halen hklt hnumax hkbound
          hdxlen hdglen hdx hdg pv tv ts qv dv v1 gv xp xpr sum hp hpv hxm hxmr hadd hnarrow htv
          hqm hdiv hqv hsub hdv hg1 hg2 hgv
      rw [hexec]
      refine ⟨⟨by rw [hk2v]; omega, rfl, hdx2, hdg2⟩, ?_⟩
      · dsimp only; rw [hk2v]; omega
    · have hnot : ¬ (k' < n) := fun hc => hklt ((UScalar.lt_equiv k' n).1 hc)
      have hbody : learn_port.dense.rmsnorm_bwd_loop1.body x dy n nd r r2 a s k' true dx' dg'
          = ok (ControlFlow.done (true, dx', dg')) := by
        rw [learn_port.dense.rmsnorm_bwd_loop1.body, if_neg hnot]
      simp only [hbody]
      have hk'n : k'.val = n.val := by omega
      exact ⟨rfl, by rw [← hk'n]; exact hdx, by rw [← hk'n]; exact hdg⟩
  · exact ⟨by show (0#usize).val ≤ n.val; rw [show (0#usize).val = 0 from rfl]; omega,
      rfl, hdx0, hdg0⟩

/-! ## 3. `rmsnorm_bwd`

The backward pass entry point. The shape guards and the `usize → u64 / i64` casts are the explicit
domain; the `inv_rms` root `r`, its square `r2 = mul_round r r`, the `i128` accumulator's single
`round32` to the scalar `S` and the per-index `dx`/`dgamma` chain of loop 1 are the callee
contracts. The two output vectors are the `ℤ` specification of `DenseNormSpec.lean`. -/

/-- Specification of `rmsnorm_bwd` on the all-succeed domain. `r` is the `inv_rms` root and `r2` its
`mul_round` square, both pinned by their extracted callees; the loop-1 `dx` model reads the internal
`a` vector through `bwdAVals`, so the conclusion mentions only `x`, `gamma`, `dy`, `r`. -/
theorem rmsnorm_bwd_spec (x gamma dy : Slice I64) (eps r r2 : I64)
    (hne : x.val.length ≠ 0)
    (hga : gamma.val.length = x.val.length)
    (hdy : dy.val.length = x.val.length)
    (hguard : x.val.length ≤ 9223372036854775807)
    (hnmax : x.val.length + 1 ≤ U32.max)
    (hinv : learn_port.dense.inv_rms x eps = ok (core.result.Result.Ok r))
    (hr2 : learn_port.dense.mul_round r r = ok (some r2))
    (hr2v : r2.val = apeiron_norm.mulRound r.val r.val)
    (hfitA : ∀ k, k < x.val.length →
      -9223372036854775808 ≤ bwdA (gamma.val.getD k default).val (dy.val.getD k default).val ∧
        bwdA (gamma.val.getD k default).val (dy.val.getD k default).val ≤ 9223372036854775807)
    (hfitmul : ∀ k, k < x.val.length →
      I128.min ≤ bwdA (gamma.val.getD k default).val (dy.val.getD k default).val *
          (x.val.getD k default).val ∧
        bwdA (gamma.val.getD k default).val (dy.val.getD k default).val *
          (x.val.getD k default).val ≤ I128.max)
    (hfitadd : ∀ k, k < x.val.length →
      I128.min ≤ bwdASum x.val gamma.val dy.val k +
          bwdA (gamma.val.getD k default).val (dy.val.getD k default).val *
            (x.val.getD k default).val ∧
        bwdASum x.val gamma.val dy.val k +
          bwdA (gamma.val.getD k default).val (dy.val.getD k default).val *
            (x.val.getD k default).val ≤ I128.max)
    (hroundS : I128.min ≤ bwdASum x.val gamma.val dy.val x.val.length + (2 : ℤ) ^ (31 : ℕ) ∧
        bwdASum x.val gamma.val dy.val x.val.length + (2 : ℤ) ^ (31 : ℕ) ≤ I128.max)
    (hnarrowS : -9223372036854775808 ≤
        apeiron_norm.round32 (bwdASum x.val gamma.val dy.val x.val.length) ∧
        apeiron_norm.round32 (bwdASum x.val gamma.val dy.val x.val.length) ≤ 9223372036854775807)
    (hstep : ∀ (r2a sa : I64) (a : alloc.vec.Vec I64) (k : ℕ), k < x.val.length →
        a.val.map (fun y => y.val) = bwdAVals gamma.val dy.val x.val.length →
        r2a.val = apeiron_norm.mulRound r.val r.val →
        sa.val = apeiron_norm.round32 (bwdASum x.val gamma.val dy.val x.val.length) →
        ∃ (pv tv ts qv dv v1 gv : I64) (xp xpr sum : I128),
          learn_port.dense.mul_round r (a.val.getD k default) = ok (some pv) ∧
          pv.val = apeiron_norm.bwdP r.val (a.val.getD k default).val ∧
          I128.checked_mul (core.convert.num.FromI128I64.from (x.val.getD k default))
            (core.convert.num.FromI128I64.from r2a) = some xp ∧
          xp.checked_mul (core.convert.num.FromI128I64.from r) = some xpr ∧
          xpr.checked_add (9223372036854775808#i128) = some sum ∧
          solve_port.q32.narrow (⟨sum.bv.sshiftRight 64⟩ : I128) = ok (some tv) ∧
          tv.val = apeiron_norm.bwdT (x.val.getD k default).val r.val r2a.val ∧
          learn_port.dense.mul_round tv sa = ok (some ts) ∧
          learn_port.dense.div_round ts (UScalar.hcast IScalarTy.I64 (Slice.len x)) = ok (some qv) ∧
          qv.val = apeiron_norm.bwdQ tv.val sa.val (x.val.length : ℤ) ∧
          pv.checked_sub qv = some dv ∧
          dv.val = pv.val - qv.val ∧
          learn_port.dense.mul_round (dy.val.getD k default) (x.val.getD k default) = ok (some v1) ∧
          learn_port.dense.mul_round v1 r = ok (some gv) ∧
          gv.val = apeiron_norm.bwdDg (dy.val.getD k default).val (x.val.getD k default).val r.val) :
    ∃ dx dg : alloc.vec.Vec I64,
      learn_port.dense.rmsnorm_bwd x gamma eps dy = ok (core.result.Result.Ok (dx, dg)) ∧
      dx.val.map (fun y => y.val) =
        bwdDxValsL x.val (bwdAVals gamma.val dy.val x.val.length) r.val
          (apeiron_norm.mulRound r.val r.val)
          (apeiron_norm.round32 (bwdASum x.val gamma.val dy.val x.val.length))
          (x.val.length : ℤ) x.val.length ∧
      dg.val.map (fun y => y.val) = bwdDgVals dy.val x.val r.val x.val.length := by
  have hn : (Slice.len x).val = x.val.length := Slice.len_val x
  have hnz : ¬ (Slice.len x = 0#usize) := by
    intro h
    apply hne
    have hv := congrArg (fun u : Usize => u.val) h
    rw [hn, show (0#usize).val = 0 from rfl] at hv
    exact hv
  have hgl : (Slice.len gamma).val = gamma.val.length := Slice.len_val gamma
  have hshape : ¬ (Slice.len gamma != Slice.len x) := by
    rw [bne_iff_ne]
    intro h
    exact h (UScalar.eq_of_val_eq (by rw [hgl, hn, hga]))
  have hdl : (Slice.len dy).val = dy.val.length := Slice.len_val dy
  have hshape2 : ¬ (Slice.len dy != Slice.len x) := by
    rw [bne_iff_ne]
    intro h
    exact h (UScalar.eq_of_val_eq (by rw [hdl, hn, hdy]))
  have hguardC : ¬ (UScalar.cast UScalarTy.U64 (Slice.len x) >
      IScalar.hcast UScalarTy.U64 core.num.I64.MAX) := by
    rw [norm_guard_iff, hn]
    exact not_lt.2 hguard
  have hcap : (alloc.vec.Vec.with_capacity I64 (Slice.len x)).val = [] := rfl
  have hloop0 := rmsnorm_bwd_loop0_spec x gamma dy (Slice.len x)
    (alloc.vec.Vec.with_capacity I64 (Slice.len x))
    (by rw [hn]; exact hga) (by rw [hn]; exact hdy) (by rw [hn])
    (by rw [hn]; exact Slice.property x) (by rw [hn]; exact hnmax)
    (by intro k hk; rw [hn] at hk; exact hfitA k hk)
    (by intro k hk; rw [hn] at hk; exact hfitmul k hk)
    (by intro k hk; rw [hn] at hk; exact hfitadd k hk)
    (by rw [hcap]; rfl)
    (by show (0#i128).val = bwdASum x.val gamma.val dy.val 0
        rw [show (0#i128).val = 0 from rfl]; rfl)
  obtain ⟨b0, hbeq0, hbpost0⟩ := WP.spec_imp_exists hloop0
  rcases b0 with ⟨a1, acc, ok1⟩
  obtain ⟨hok1, ha1v, haccv⟩ := hbpost0
  have ha1v' : a1.val.map (fun y => y.val) = bwdAVals gamma.val dy.val x.val.length := by
    rw [ha1v, hn]
  have haccv' : acc.val = bwdASum x.val gamma.val dy.val x.val.length := by rw [haccv, hn]
  obtain ⟨sy, hsy, hsyv⟩ := dense_round32_val acc
    (by rw [haccv']; exact hroundS.1) (by rw [haccv']; exact hroundS.2)
  have hsyv' : sy.val =
      apeiron_norm.round32 (bwdASum x.val gamma.val dy.val x.val.length) := by rw [hsyv, haccv']
  have hnarrow_sy : solve_port.q32.narrow sy =
      ok (some (IScalar.cast IScalarTy.I64 sy)) :=
    dense_narrow_some sy (by rw [hsyv']; exact not_lt.2 hnarrowS.2)
      (by rw [hsyv']; exact not_lt.2 hnarrowS.1)
  have hsvv : (IScalar.cast IScalarTy.I64 sy).val =
      apeiron_norm.round32 (bwdASum x.val gamma.val dy.val x.val.length) := by
    rw [cast_I64_val sy (by rw [hsyv']; exact hnarrowS.1) (by rw [hsyv']; exact hnarrowS.2)]
    exact hsyv'
  have hndv : (UScalar.hcast IScalarTy.I64 (Slice.len x)).val = (x.val.length : ℤ) := by
    rw [norm_hcast_I64_val (Slice.len x) (by rw [hn]; exact hguard), hn]
  have halen : a1.val.length = (Slice.len x).val := by
    have hh := congrArg List.length ha1v'
    simp only [List.length_map, bwdAVals_length] at hh
    rw [hh, hn]
  have hloop1 := rmsnorm_bwd_loop1_spec x dy (Slice.len x)
    (UScalar.hcast IScalarTy.I64 (Slice.len x)) r r2 a1 (IScalar.cast IScalarTy.I64 sy)
    (alloc.vec.Vec.with_capacity I64 (Slice.len x)) (alloc.vec.Vec.with_capacity I64 (Slice.len x))
    (by rw [hn]) (by rw [hn]; exact hdy) (by rw [halen, hn])
    (by rw [hn]; exact Slice.property x) (by rw [hn]; exact hnmax)
    (by
      intro k hk
      rw [hn] at hk
      obtain ⟨pv, tv, ts, qv, dv, v1, gv, xp, xpr, sum, hc⟩ :=
        hstep r2 (IScalar.cast IScalarTy.I64 sy) a1 k hk ha1v' hr2v hsvv
      refine ⟨pv, tv, ts, qv, dv, v1, gv, xp, xpr, sum, ?_⟩
      rw [hndv]
      exact hc)
    (by rw [hcap]; rfl) (by rw [hcap]; rfl)
  obtain ⟨c1, hbeq1, hbpost1⟩ := WP.spec_imp_exists hloop1
  rcases c1 with ⟨ok2, dx1, dg1⟩
  obtain ⟨hok2, hdxm, hdgm⟩ := hbpost1
  have hdxm' : dx1.val.map (fun y => y.val) =
      bwdDxValsL x.val (bwdAVals gamma.val dy.val x.val.length) r.val
        (apeiron_norm.mulRound r.val r.val)
        (apeiron_norm.round32 (bwdASum x.val gamma.val dy.val x.val.length))
        (x.val.length : ℤ) x.val.length := by
    rw [hdxm, bwdDxVals_eq_L x.val a1 r.val r2.val (IScalar.cast IScalarTy.I64 sy).val
      (UScalar.hcast IScalarTy.I64 (Slice.len x)).val (Slice.len x).val, ha1v', hn, hsvv, hndv, hr2v]
  have hdgm' : dg1.val.map (fun y => y.val) = bwdDgVals dy.val x.val r.val x.val.length := by
    rw [hdgm, hn]
  refine ⟨dx1, dg1, ?_, hdxm', hdgm'⟩
  rw [learn_port.dense.rmsnorm_bwd, if_neg hnz, if_neg hshape, if_neg hshape2]
  dsimp (zeta := true) only
  simp only [lift, bind_tc_ok]
  rw [if_neg hguardC]
  rw [hinv]
  simp only [core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok]
  rw [hr2]
  simp only [dense_ov_some, bind_tc_ok]
  rw [hbeq0]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [if_pos hok1]
  rw [hsy]
  simp only [hnarrow_sy, dense_ov_some, bind_tc_ok]
  rw [hbeq1]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [if_pos hok2]
end apeiron_grad
