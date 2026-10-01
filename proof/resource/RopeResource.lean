import RopeSource
import Aeneas

/-
SCOPE: the actual extracted rotary-pair loop `rope_source.rope.RopeTable.rotate_head_loop` (from
`src/engine/rope.rs` lines 217-260, reached from `crate::word_rope_rotate_fixed` in
`proof/extract/lib.rs:738` through the typed forwarding wrapper), the two exit branches of its
body, its work measure (the strictly decreasing rank `head_dim/2 - j`, one `fixed_sin_cos`
evaluation per guarded iteration), the outer head loop's exit branch, and a reachable success
witness for the extracted root. Completion: `proof/resource/Rope_gate.py` prints
`ROPE_SOURCE_RESOURCE_OK` with the mutation control red and restored.
REGISTER: scalar source refinement and a visit-count resource claim on the extracted body.
Excluded: the arithmetic value of a rotated component (no `fixed_sin_cos` evaluation is proved for
any concrete phase), the outer head loop's slice write-back, the number of *loop iterations* as an
observable of the extracted result type (the result carries no counter), and any hardware timing.
CORPUS: `src/engine/rope.rs`, `src/engine/value.rs`, `src/numeric/fixed_trig.rs`; extracted as
`proof/resource/RopeSource.lean`; theorem names `RopeResource.{head_body_step, head_body_refused,
head_body_exit, head_body_stopped, rotate_body_exit, rotate_loop_empty, empty_rotate_witness}`.
REFERENCE: NONE - the resource claim is about the extracted body, not a canonical definition.
GATE: `python3 proof/resource/Rope_gate.py --controls`, 540 s wall clock, lane
`/tmp/opencode/apeiron-lane-2.lock`, `systemd-run -p MemoryMax=12G -p MemorySwapMax=0`,
`LEAN_NUM_THREADS=1`; mutation control `j += 1;` -> `j += 2;` in `src/engine/rope.rs:254`.
GRADE: `proved` for every `#resource_audit`ed theorem (Lean kernel, the closed foundation
{propext, Classical.choice, Quot.sound} checked by the audit command below); `tested` for the wrapper crate's 4 `src/numeric/
fixed_trig.rs` unit tests, which the gate compiles and runs. Work is counted in loop iterations
and `fixed_sin_cos` evaluations under the named premises of each theorem; no cycle, wall-clock or
GPU figure is claimed and no benchmark was run.
-/

namespace RopeResource
open Aeneas Aeneas.Std Result
open rope_source

/-- The value of a `Result` computation known to succeed, read without case analysis on the
computation; the equation it inverts is stated separately as `res_eq`. -/
noncomputable def res {α : Type} (e : Result α) (h : ∃ y, e = ok y) : α := Classical.choose h

/-- The equation that `res` inverts. -/
theorem res_eq {α : Type} (e : Result α) (h : ∃ y, e = ok y) : e = ok (res e h) :=
  Classical.choose_spec h

/-- The product of two Q32.32 components in `i128`: the value-path product of
`src/engine/rope.rs:227-230`. -/
def rot_prod (a b : I64) : Result (Std.IScalar .I128) :=
  core.convert.num.FromI128I64.from a * core.convert.num.FromI128I64.from b

set_option maxHeartbeats 400000 in
/-- One guarded iteration of the actual extracted rotary-pair loop. Under the guard
`j < head_dim/2` and `ok1 = true`, and the named sub-computations of the phase and of the
epilogue succeeding, the body evaluates `fixed_sin_cos` once on the phase of index `j`
(`htrig`, the premise that fixes the result the body then branches on), advances the pair index
to `j + 1`, keeps the vector length, and lowers the rank `d2.val - j.val` by one.

The work measure is the rank `d2.val - j.val`, where `d2` is `head_dim / 2`
(`src/engine/rope.rs:218`) and `j` is the pair index: `head_body_refused` and `head_body_step`
give a decrease of exactly one per guarded iteration, `head_body_exit` shows the loop stops at
`j = d2` (rank zero) and `head_body_stopped` shows an earlier refusal stops it. The entry point
starts the loop at `j = 0` (`RopeSource.lean:400`), so a head that rotates without refusal
performs `d2.val = head_dim/2` guarded iterations, one `fixed_sin_cos` evaluation each, and hence
`head_dim/2` trig calls and `head_dim` in-place component writes. The premises that are not
proved here are exactly the `Result`-valued sub-computations of the loop body, named one by one:
the table read, the `i128` products `hmul`, the round-to-nearest epilogue `hrs`, the `usize` index
sums `hj1`/`hjd`, the exact-width phase product `hp3`/`hφ`, the radian lift `hθ` and the trig call
`htrig`. -/
theorem head_body_step (v : alloc.vec.Vec U64) (pos : U32) (d2 j : Usize) (x : Slice I64)
    (ok1 : Bool) (x0 : Slice I64) (hjg : j < d2) (hj : j.val < d2.val) (hok : ok1 = true)
    (hb : j.val < v.length) (hx : d2.val + d2.val ≤ x.length) (hlen : x.length = x0.length)
    (hj1 : j.val + (UScalar.one : Usize).val ≤ Usize.max) (hjd : j.val + d2.val ≤ Usize.max)
    (hs : ∀ i : Usize, i.val < x.length → ∃ c : I64, x.index_usize i = ok c)
    (hmul : ∀ a b : I64, ∃ p : Std.IScalar .I128, rot_prod a b = ok p)
    (hrs : ∀ a : Std.IScalar .I128, ∃ o : Std.IScalar .I128, value.round_shift_32 a = ok o)
    (p3 : Std.UScalar .U128)
    (hp3 : (UScalar.cast .U128 (v.val[j.val]'hb) * UScalar.cast .U128 pos) =
      (ok p3 : Result (Std.UScalar .U128)))
    (φ : U64) (hφ : (UScalar.cast .U64 (p3 &&& UScalar.cast .U128 core.num.U64.MAX)) = φ)
    (θ : I64) (hθ : fixed_trig.turn_to_radians φ = ok θ)
    (htrig : ∃ sc : fixed_trig.SinCos, fixed_trig.fixed_sin_cos θ = ok (core.result.Result.Ok sc)) :
    ∃ x' : Slice I64, ∃ jn ok' : _, j + (UScalar.one : Usize) = ok jn ∧
      jn.val = j.val + 1 ∧
      rope.RopeTable.rotate_head_loop.body v pos d2 x j ok1 =
        ok (ControlFlow.cont (x', jn, ok')) ∧
      x'.length = x0.length ∧ d2.val - jn.val < d2.val - j.val := by
  have hone : (1#usize : Usize) = (UScalar.one : Usize) := rfl
  have hone1 : (UScalar.one : Usize).val = 1 := by rfl
  obtain ⟨turn, hidx, hturn⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec v j hb)
  obtain ⟨jn, hadd1, hjn⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := j) (y := (UScalar.one : Usize)) hj1)
  obtain ⟨jd, hadd2, hjd'⟩ := WP.spec_imp_exists (Usize.add_spec (x := j) (y := d2) hjd)
  have hlj : j.val < x.length := by omega
  have hld : jd.val < x.length := by omega
  let c0 : I64 := res (x.index_usize j) (hs j hlj)
  have hs0 : x.index_usize j = ok c0 := res_eq _ _
  let c1 : I64 := res (x.index_usize jd) (hs jd hld)
  have hs1 : x.index_usize jd = ok c1 := res_eq _ _
  unfold rope.RopeTable.rotate_head_loop.body
  rw [if_pos hjg, if_pos hok]
  simp only [alloc.vec.Vec.index_slice_index]
  rw [hone, hidx, hadd1, hadd2]
  simp only [Std.lift, bind_tc_ok]
  rw [hs0, hs1]
  simp only [bind_tc_ok]
  rw [← hturn] at hp3
  rw [hp3]
  simp only [bind_tc_ok]
  rw [hφ, hθ]
  simp only [bind_tc_ok]
  obtain ⟨sc0, hr0⟩ := htrig
  rw [hr0]
  simp only [bind_tc_ok]
  let p00 : Std.IScalar .I128 := res (rot_prod c0 sc0.cos) (hmul c0 sc0.cos)
  have hp00 : rot_prod c0 sc0.cos = ok p00 := res_eq _ _
  let p01 : Std.IScalar .I128 := res (rot_prod c1 sc0.sin) (hmul c1 sc0.sin)
  have hp01 : rot_prod c1 sc0.sin = ok p01 := res_eq _ _
  let p10 : Std.IScalar .I128 := res (rot_prod c1 sc0.cos) (hmul c1 sc0.cos)
  have hp10 : rot_prod c1 sc0.cos = ok p10 := res_eq _ _
  let p11 : Std.IScalar .I128 := res (rot_prod c0 sc0.sin) (hmul c0 sc0.sin)
  have hp11 : rot_prod c0 sc0.sin = ok p11 := res_eq _ _
  simp only [rot_prod] at hp00 hp01 hp10 hp11
  simp only [hp00, hp01, hp10, hp11, bind_tc_ok]
  cases ho : I128.checked_sub p00 p01 with
  | none => exact ⟨x, jn, false, rfl, by omega, rfl, hlen, by omega⟩
  | some re =>
    cases ho1 : I128.checked_add p10 p11 with
    | none => exact ⟨x, jn, false, rfl, by omega, rfl, hlen, by omega⟩
    | some im =>
      let out0 : Std.IScalar .I128 := res (value.round_shift_32 re) (hrs re)
      have hrs0 : value.round_shift_32 re = ok out0 := res_eq _ _
      let out1 : Std.IScalar .I128 := res (value.round_shift_32 im) (hrs im)
      have hrs1 : value.round_shift_32 im = ok out1 := res_eq _ _
      simp only [hrs0, hrs1, bind_tc_ok]
      split
      · exact ⟨x, jn, false, rfl, by omega, rfl, hlen, by omega⟩
      · split
        · exact ⟨x, jn, false, rfl, by omega, rfl, hlen, by omega⟩
        · split
          · exact ⟨x, jn, false, rfl, by omega, rfl, hlen, by omega⟩
          · split
            · exact ⟨x, jn, false, rfl, by omega, rfl, hlen, by omega⟩
            · have hspec2 := Slice.update_spec x j (IScalar.cast .I64 out0) hlj
              have hex2 : ∃ nv : Slice I64,
                  Slice.update x j (IScalar.cast .I64 out0) = ok nv ∧
                    nv = x.set j (IScalar.cast .I64 out0) :=
                (WP.spec_equiv_exists _ _).mp hspec2
              let x2 : Slice I64 := Classical.choose hex2
              have hupd2 : Slice.update x j (IScalar.cast .I64 out0) = ok x2 :=
                (Classical.choose_spec hex2).1
              simp only [hupd2, bind_tc_ok]
              have hx2len : x2.length = x.length := by
                show (Classical.choose hex2).length = x.length
                rw [(Classical.choose_spec hex2).2, Slice.set_length]
              have hspec3 := Slice.update_spec x2 jd (IScalar.cast .I64 out1) (by
                rw [hx2len]; exact hld)
              have hex3 : ∃ nv : Slice I64,
                  Slice.update x2 jd (IScalar.cast .I64 out1) = ok nv ∧
                    nv = x2.set jd (IScalar.cast .I64 out1) :=
                (WP.spec_equiv_exists _ _).mp hspec3
              let s : Slice I64 := Classical.choose hex3
              have hupd3 : Slice.update x2 jd (IScalar.cast .I64 out1) = ok s :=
                (Classical.choose_spec hex3).1
              rw [hupd3]
              simp only [bind_tc_ok]
              refine ⟨s, jn, true, rfl, ?_, rfl, ?_, ?_⟩
              · omega
              · show (Classical.choose hex3).length = x0.length
                rw [(Classical.choose_spec hex3).2, Slice.set_length, hx2len, hlen]
              · omega


set_option maxHeartbeats 400000 in
/-- The trig refusal arm: when `fixed_sin_cos` returns `Err` the body advances the pair index by
one, clears the acceptance flag, and leaves the vector untouched. -/
theorem head_body_refused (v : alloc.vec.Vec U64) (pos : U32) (d2 j : Usize) (x : Slice I64)
    (ok1 : Bool) (x0 : Slice I64) (hjg : j < d2) (hj : j.val < d2.val) (hok : ok1 = true)
    (hb : j.val < v.length) (hlen : x.length = x0.length)
    (hj1 : j.val + (UScalar.one : Usize).val ≤ Usize.max)
    (p3 : Std.UScalar .U128)
    (hp3 : (UScalar.cast .U128 (v.val[j.val]'hb) * UScalar.cast .U128 pos) =
      (ok p3 : Result (Std.UScalar .U128)))
    (φ : U64) (hφ : (UScalar.cast .U64 (p3 &&& UScalar.cast .U128 core.num.U64.MAX)) = φ)
    (θ : I64) (hθ : fixed_trig.turn_to_radians φ = ok θ)
    (e : fixed_trig.TrigError)
    (hr : fixed_trig.fixed_sin_cos θ = ok (core.result.Result.Err e)) :
    ∃ x' : Slice I64, ∃ jn ok' : _, j + (UScalar.one : Usize) = ok jn ∧
      jn.val = j.val + 1 ∧
      rope.RopeTable.rotate_head_loop.body v pos d2 x j ok1 =
        ok (ControlFlow.cont (x', jn, ok')) ∧
      x'.length = x0.length ∧ d2.val - jn.val < d2.val - j.val := by
  have hone : (1#usize : Usize) = (UScalar.one : Usize) := rfl
  have hone1 : (UScalar.one : Usize).val = 1 := by rfl
  obtain ⟨turn, hidx, hturn⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_usize_spec v j hb)
  obtain ⟨jn, hadd1, hjn⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := j) (y := (UScalar.one : Usize)) hj1)
  unfold rope.RopeTable.rotate_head_loop.body
  rw [if_pos hjg, if_pos hok]
  simp only [alloc.vec.Vec.index_slice_index]
  rw [hone, hidx, hadd1]
  simp only [Std.lift, bind_tc_ok]
  rw [← hturn] at hp3
  rw [hp3]
  simp only [bind_tc_ok]
  rw [hφ, hθ]
  simp only [bind_tc_ok]
  rw [hr]
  simp only [bind_tc_ok]
  exact ⟨x, jn, false, rfl, by omega, rfl, hlen, by omega⟩

/-- The first exit branch: once the pair index reaches `d2 = head_dim / 2` the body stops and
reports the accumulated flag unchanged. -/
theorem head_body_exit (v : alloc.vec.Vec U64) (pos : U32) (d2 j : Usize) (x : Slice I64)
    (ok1 : Bool) (hj : ¬ (j < d2)) :
    rope.RopeTable.rotate_head_loop.body v pos d2 x j ok1 = ok (ControlFlow.done (x, ok1)) := by
  unfold rope.RopeTable.rotate_head_loop.body
  rw [if_neg hj]

/-- The second exit branch: a refusal recorded by an earlier pair stops the loop at once. -/
theorem head_body_stopped (v : alloc.vec.Vec U64) (pos : U32) (d2 j : Usize) (x : Slice I64)
    (ok1 : Bool) (hjg : j < d2) (hok : ¬ (ok1 = true)) :
    rope.RopeTable.rotate_head_loop.body v pos d2 x j ok1 = ok (ControlFlow.done (x, false)) := by
  unfold rope.RopeTable.rotate_head_loop.body
  rw [if_pos hjg, if_neg hok]

/-- The head width two, a valid `RopeTable::from_turns` head (`head_dim >= 2`, even). -/
def two : Usize := UScalar.ofNat 2 (by decide)

/-- A valid load-time turn table for `head_dim = 2`, carrying the two rotary phases. -/
def tbl0 : rope.RopeTable :=
  { head_dim := two, turns := alloc.vec.Vec.from [(UScalar.zero : U64)] (by scalar_tac) }

/-- An empty `i64` vector, a legal `head_dim`-multiple operand. -/
def q0 : Slice I64 := Slice.from [] (by scalar_tac)

/-- The outer loop's exit branch: once the head index reaches `heads = q.len() / head_dim` the
body stops and reports the accumulated flag unchanged. -/
theorem rotate_body_exit (self : rope.RopeTable) (hd : Usize) (v : alloc.vec.Vec U64)
    (pos : U32) (heads h : Usize) (q k : Slice I64) (ok1 : Bool) (hh : ¬ (h < heads)) :
    rope.RopeTable.rotate_loop.body self hd v pos heads q k h ok1 =
      ok (ControlFlow.done (q, k, ok1)) := by
  unfold rope.RopeTable.rotate_loop.body
  rw [if_neg hh]

/-- A zero-head rotation performs no rotary-pair iteration: the outer loop exits on its first
guard and returns both vectors unchanged with the acceptance flag set. This is the work measure
of the reachable `q.len() = 0` case: zero `rotate_head` calls, hence zero `fixed_sin_cos`
evaluations. -/
theorem rotate_loop_empty (self : rope.RopeTable) (hd : Usize) (v : alloc.vec.Vec U64)
    (pos : U32) (q k : Slice I64) :
    rope.RopeTable.rotate_loop self hd v q k pos (UScalar.zero : Usize) (UScalar.zero : Usize) true
      ⦃ r => r = (q, k, true) ⦄ := by
  have hzero : (0#usize : Usize) = (UScalar.zero : Usize) := rfl
  unfold rope.RopeTable.rotate_loop
  apply loop.spec_decr_nat
    (measure := fun _ : Slice I64 × Slice I64 × Usize × Bool => 0)
    (inv := fun st : Slice I64 × Slice I64 × Usize × Bool =>
      st = (q, k, (UScalar.zero : Usize), true))
    (post := fun r : Slice I64 × Slice I64 × Bool => r = (q, k, true))
    (body := fun st : Slice I64 × Slice I64 × Usize × Bool =>
      rope.RopeTable.rotate_loop.body self hd v pos (UScalar.zero : Usize) st.1 st.2.1 st.2.2.1
        st.2.2.2)
    (x := (q, k, (UScalar.zero : Usize), true))
  · intro st hst
    subst st
    rw [rotate_body_exit self hd v pos (UScalar.zero : Usize) (UScalar.zero : Usize) q k true
      (by simp)]
    exact ⟨rfl⟩
  · show (q, k, (UScalar.zero : Usize), true) = (q, k, (UScalar.zero : Usize), true)
    rfl

/-- The reachable success witness for the extracted root: a load-time table and two empty
`head_dim`-multiple vectors at position zero return `Ok (())` with both vectors unchanged, and
the pair loop is never entered. -/
theorem empty_rotate_witness :
    rope_rotate_forward q0 q0 (UScalar.zero : U32) tbl0 =
      ok (core.result.Result.Ok (), q0, q0) := by
  have hzero : (0#usize : Usize) = (UScalar.zero : Usize) := rfl
  have hone : (1#usize : Usize) = (UScalar.one : Usize) := rfl
  have hlen0 : (Slice.len q0).val = 0 := by simp [q0, Slice.len, Slice.from_val]
  have htwo : two.val = 2 := by rfl
  have hhd : tbl0.head_dim.val = 2 := by rfl
  have hzval : (UScalar.zero : Usize).val = 0 := by rfl
  have hmul : core.num.Usize.is_multiple_of (Slice.len q0) tbl0.head_dim = ok true := by
    simp only [core.num.Usize.is_multiple_of, UScalar.is_multiple_of]
    norm_num [hlen0, hhd]
  have hdiv : Slice.len q0 / tbl0.head_dim = ok (UScalar.zero : Usize) := by
    obtain ⟨z, hz, hzv⟩ :=
      WP.spec_imp_exists (Usize.div_spec (x := Slice.len q0) (y := tbl0.head_dim) (by omega))
    have hz0 : z = (UScalar.zero : Usize) := by
      refine UScalar.eq_imp z (UScalar.zero : Usize) ?_
      rw [hzv]
      norm_num [hlen0, hhd, hzval]
    rw [← hz0]
    exact hz
  obtain ⟨r, hr, hr'⟩ := (WP.spec_equiv_exists _ _).mp
    (rotate_loop_empty tbl0 tbl0.head_dim tbl0.turns (UScalar.zero : U32) q0 q0)
  unfold rope_rotate_forward
  unfold rope.rope_rotate_fixed
  unfold rope.RopeTable.rotate
  have hne : ∀ a b : Usize, a = b → (a != b) = false := by
    intro a b h
    subst h
    simp
  simp (config := { zetaDelta := true }) only [Slice.len]
  rw [hne _ _ rfl, hmul, hdiv]
  simp only [bind_tc_ok, hzero]
  rw [hr]
  simp only [bind_tc_ok, hr']
  rfl

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

#resource_audit RopeResource.head_body_step
#resource_audit RopeResource.head_body_refused
#resource_audit RopeResource.head_body_exit
#resource_audit RopeResource.head_body_stopped
#resource_audit RopeResource.rotate_body_exit
#resource_audit RopeResource.rotate_loop_empty
#resource_audit RopeResource.empty_rotate_witness

end RopeResource

run_cmd Lean.logInfo "ROPE_KERNEL_OK"
