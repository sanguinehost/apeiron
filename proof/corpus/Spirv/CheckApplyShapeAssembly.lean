import Spirv.CheckApplyShape

/-!
# `apply_shape` assembly

The pure assembly of the extracted `src/spirv/check.rs::apply_shape`, over the extracted body in
`proof/ApeironWord.lean` (line 14060) and the per-component specs of `Spirv.CheckScanState`
(`storage_class_admitted_spec`, `decoration_admitted_spec`), `Spirv.CheckScan`
(`defs_contains_spec`, `ty_get_spec`) and `Spirv.CheckApplyShape` (the eight used-id loop specs
`apply_shape_loopN_spec`, `resolve_tag_spec`, `ty_insert_spec`).

* `apply_shape_spec` — `spirv.check.apply_shape index shape defs tys = ok (applyShapePure …)`
  under `hdefs : defs.val.length < Usize.max` and `htys : tys.val.length < Usize.max`. The
  `storage_class`/`decoration` admission, the `sudot` `TypeMismatch` checks, the used-id scan
  (`firstUndef` ⇒ `UndefinedId`), the duplicate-result-id refusal, the `defs` push and the
  `ty_insert` update are all read through their extracted specs.
* `applyShapePure` — the pure image: a `let next` holding the used-id scan and the post-scan push
  (undefined-id refusal, then duplicate-result-id refusal, then `Vec.from` push and `tyInsertVec`),
  followed by the nested `storage_class`/`decoration`/`sudot` dispatch whose `sudot` arm is
  `applyShapeSudotPure`. Its refusal order is the order of `src/spirv/check.rs::apply_shape`.
* `apply_shape_sudot_checks` — the extracted `result_ok`/`operands_ok` block equals the
  `sudotResultOk`/`sudotOperandsOk` dispatch; `apply_shape_spec` applies it by `Eq.trans`, since the
  extracted `32#uscalar` literal-pattern matcher and this module's are distinct auxiliary
  declarations, equal only up to definitional unfolding.

## Residual obligation (components 2–3 of `Spirv/Check.lean`, not proved here)

* **Capacity bounds.** `hdefs`/`htys` are hypotheses of `apply_shape_spec`. `check_fragment_loop`
  must discharge them: `defs` grows by at most one per iteration and each iteration advances
  `index` by `word_count ≥ 1`, so `defs.val.length ≤ index.val ≤ words.len ≤ Usize.max`; `tys`
  likewise grows only on a `defs` push. Neither bound is stated in the corpus yet.
* **`check_fragment_loop`.** A `loop.spec_decr_nat` spec (measure `words.len - index`) whose step
  composes `shape_of` (framing and shape admission, component 1, open) with `apply_shape_spec`
  and the `CoreOpsTry.branch` propagation of its refusal, carrying the invariant
  `defs.val = st.defs.reverse` (`WFState.defs`) and a `tys` invariant (`tys.val.map (τ ×) =
  st.tys.reverse` up to key `.val`), so that `firstUndef`/`∈ defs.val` equal the `WFInst`
  used-id and duplicate-id conjuncts and `sudotResultOk`/`sudotOperandsOk` equal its `OpSUDot`
  type conjunct. Nothing about `check_fragment_loop` is proved.
* **`τ` translation.** A total `τ : spirv.check.TyTag → Spirv.TyTag` does not exist in the corpus;
  `ty_get_eq_lookupTy_of_nodup` (CheckScan) takes it as a parameter. Still open: define `τ`, prove
  `resolveTagPure shape.tag tys.val` maps under `τ` to `resultTag st i` for the instruction `i` that
  `shape_of` produced `shape` from, and prove key-uniqueness (`Nodup` of `tys.val.map Prod.fst`) is
  preserved by `tyInsertVec`, which the scan-order lemma `firstLookup_reverse_lookupTy` needs.
* The composition to `decode ws = some m` and `checkFragment_sound` stays open until all of the
  above land.

## Controls

The mutation control swaps the order of the duplicate-result-id refusal and the undefined-id scan in
`applyShapePure`'s `next` (throwaway copy `Mutant.lean`): the proof of
`apply_shape_spec` then fails, since the extracted body scans before it refuses a duplicate.
-/

namespace Spirv

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

open Aeneas Aeneas.Std Result
open apeiron_word

/-- The pure reading of the extracted `sudot` operand check on one id: `true` exactly when the id is
recorded as `TyTag.Int 32`. -/
def sudotIntOk (tys : List (Std.U32 × spirv.check.TyTag)) (id : Std.U32) : Bool :=
  match firstLookup tys id with
  | none => false
  | some tt =>
      match tt with
      | spirv.check.TyTag.Int i =>
          match i with
          | 32#uscalar => true
          | _ => false
      | _ => false

/-- The pure reading of the extracted `result_ok` check: `true` exactly when the result id has a
`TyTag.Int width` with `width ≥ 32`. -/
def sudotResultOk (tys : List (Std.U32 × spirv.check.TyTag)) (ty : Std.U32) : Bool :=
  match firstLookup tys ty with
  | none => false
  | some tt =>
      match tt with
      | spirv.check.TyTag.Int width => width >= 32#u32
      | _ => false

/-- The pure reading of the extracted `operands_ok` check: the first operand is `TyTag.Int 32` and,
only then, the second operand is `TyTag.Int 32`. -/
def sudotOperandsOk (tys : List (Std.U32 × spirv.check.TyTag)) (a b : Std.U32) : Bool :=
  if sudotIntOk tys a then sudotIntOk tys b else false

/-- The pure `sudot` dispatch: a present `sudot` triple runs the result/operand checks and either
continues with `next` or refuses with `TypeMismatch`; an absent one continues. -/
def applyShapeSudotPure (index : Std.Usize) (shape : spirv.check.Shape)
    (defs : alloc.vec.Vec Std.U32) (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag))
    (next : (core.result.Result Unit spirv.check.CheckError) × (alloc.vec.Vec Std.U32) ×
      (alloc.vec.Vec (Std.U32 × spirv.check.TyTag))) :
    (core.result.Result Unit spirv.check.CheckError) × (alloc.vec.Vec Std.U32) ×
      (alloc.vec.Vec (Std.U32 × spirv.check.TyTag)) :=
  match shape.sudot with
  | none => next
  | some (ty, a, b) =>
      if sudotResultOk tys.val ty then
        if sudotOperandsOk tys.val a b then next
        else (core.result.Result.Err (spirv.check.CheckError.TypeMismatch index), defs, tys)
      else (core.result.Result.Err (spirv.check.CheckError.TypeMismatch index), defs, tys)

/-- The pure image of the extracted `apply_shape`: the used-id scan and post-scan push (`next`,
mirroring `apply_shape_loopN`'s `applyShapeExit`), then the nested
`storage_class`/`decoration` admission and the `sudot` `TypeMismatch` dispatch. -/
def applyShapePure (index : Std.Usize) (shape : spirv.check.Shape)
    (defs : alloc.vec.Vec Std.U32) (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag))
    (hdefs : defs.val.length < Usize.max) (htys : tys.val.length < Usize.max) :
    (core.result.Result Unit spirv.check.CheckError) × (alloc.vec.Vec Std.U32) ×
      (alloc.vec.Vec (Std.U32 × spirv.check.TyTag)) :=
  let next :=
    match firstUndef shape.uses.val defs.val with
    | some used =>
        (core.result.Result.Err (spirv.check.CheckError.UndefinedId index used), defs, tys)
    | none =>
        match shape.result with
        | none => (core.result.Result.Ok (), defs, tys)
        | some result =>
            if result ∈ defs.val then
              (core.result.Result.Err (spirv.check.CheckError.DuplicateResultId index result),
                defs, tys)
            else
              let defs1 := alloc.vec.Vec.from (defs.val ++ [result])
                (by rw [List.length_append, List.length_singleton]; omega)
              match resolveTagPure shape.tag tys.val with
              | none => (core.result.Result.Ok (), defs1, tys)
              | some tag => (core.result.Result.Ok (), defs1, tyInsertVec tys result tag htys)
  match shape.storage_class with
  | some storage_class =>
      if decide (storage_class.val ∈ validStorageClassesList) then
        match shape.decoration with
        | some decoration =>
            if decide (decoration.val ∈ validDecorationsList) then
              applyShapeSudotPure index shape defs tys next
            else
              (core.result.Result.Err
                (spirv.check.CheckError.BadDecoration index decoration), defs, tys)
        | none => applyShapeSudotPure index shape defs tys next
      else
        (core.result.Result.Err
          (spirv.check.CheckError.BadStorageClass index storage_class), defs, tys)
  | none =>
      match shape.decoration with
      | some decoration =>
          if decide (decoration.val ∈ validDecorationsList) then
            applyShapeSudotPure index shape defs tys next
          else
            (core.result.Result.Err
              (spirv.check.CheckError.BadDecoration index decoration), defs, tys)
      | none => applyShapeSudotPure index shape defs tys next


theorem ty_get_deref_spec (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag)) (id : Std.U32) :
    spirv.check.ty_get (alloc.vec.Vec.deref tys) id = ok (firstLookup tys.val id) := by
  rw [ty_get_spec]
  simp [alloc.vec.Vec.deref, Slice.from_val]

theorem sudotResultOk_match (tys : List (Std.U32 × spirv.check.TyTag)) (ty : Std.U32) :
    (match firstLookup tys ty with
     | none => ok false
     | some tt =>
       match tt with
       | spirv.check.TyTag.Int width => ok (width >= 32#u32)
       | spirv.check.TyTag.Bool => ok false
       | spirv.check.TyTag.Float _ => ok false
       | spirv.check.TyTag.Vec => ok false
       | spirv.check.TyTag.Pointer => ok false
       | spirv.check.TyTag.RuntimeArray => ok false
       | spirv.check.TyTag.Function => ok false
       | spirv.check.TyTag.Struct => ok false
       | spirv.check.TyTag.Void => ok false)
    = ok (sudotResultOk tys ty) := by
  unfold sudotResultOk
  cases h : firstLookup tys ty with
  | none => rfl
  | some tt => cases tt <;> rfl


theorem sudotIntOk_match (tys : List (Std.U32 × spirv.check.TyTag)) (id : Std.U32) :
    (match firstLookup tys id with
     | none => ok false
     | some tt =>
       match tt with
       | spirv.check.TyTag.Int i =>
         match i with
         | 32#uscalar => ok true
         | _ => ok false
       | spirv.check.TyTag.Bool => ok false
       | spirv.check.TyTag.Float _ => ok false
       | spirv.check.TyTag.Vec => ok false
       | spirv.check.TyTag.Pointer => ok false
       | spirv.check.TyTag.RuntimeArray => ok false
       | spirv.check.TyTag.Function => ok false
       | spirv.check.TyTag.Struct => ok false
       | spirv.check.TyTag.Void => ok false)
    = ok (sudotIntOk tys id) := by
  unfold sudotIntOk
  cases h : firstLookup tys id with
  | none => rfl
  | some tt =>
      cases hh : tt with
      | Int i => simp only [hh]; split <;> simp_all
      | Bool => simp only [hh]
      | Float _ => simp only [hh]
      | Vec => simp only [hh]
      | Pointer => simp only [hh]
      | RuntimeArray => simp only [hh]
      | Function => simp only [hh]
      | Struct => simp only [hh]
      | Void => simp only [hh]

theorem apply_shape_sudot_checks (index : Std.Usize)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag)) (defs : alloc.vec.Vec Std.U32)
    (ty a b : Std.U32)
    (k : Result ((core.result.Result Unit spirv.check.CheckError) × (alloc.vec.Vec Std.U32) ×
      (alloc.vec.Vec (Std.U32 × spirv.check.TyTag)))) :
    (do
      let result_ok ←
        match firstLookup tys.val ty with
        | none => ok false
        | some tt =>
          match tt with
          | spirv.check.TyTag.Int width => ok (width >= 32#u32)
          | spirv.check.TyTag.Bool => ok false
          | spirv.check.TyTag.Float _ => ok false
          | spirv.check.TyTag.Vec => ok false
          | spirv.check.TyTag.Pointer => ok false
          | spirv.check.TyTag.RuntimeArray => ok false
          | spirv.check.TyTag.Function => ok false
          | spirv.check.TyTag.Struct => ok false
          | spirv.check.TyTag.Void => ok false
      let b1 ←
        match firstLookup tys.val a with
        | none => ok false
        | some tt =>
          match tt with
          | spirv.check.TyTag.Int i =>
            match i with
            | 32#uscalar => ok true
            | _ => ok false
          | spirv.check.TyTag.Bool => ok false
          | spirv.check.TyTag.Float _ => ok false
          | spirv.check.TyTag.Vec => ok false
          | spirv.check.TyTag.Pointer => ok false
          | spirv.check.TyTag.RuntimeArray => ok false
          | spirv.check.TyTag.Function => ok false
          | spirv.check.TyTag.Struct => ok false
          | spirv.check.TyTag.Void => ok false
      let operands_ok ←
        if b1 = true then
          match firstLookup tys.val b with
          | none => ok false
          | some tt =>
            match tt with
            | spirv.check.TyTag.Int i =>
              match i with
              | 32#uscalar => ok true
              | _ => ok false
            | spirv.check.TyTag.Bool => ok false
            | spirv.check.TyTag.Float _ => ok false
            | spirv.check.TyTag.Vec => ok false
            | spirv.check.TyTag.Pointer => ok false
            | spirv.check.TyTag.RuntimeArray => ok false
            | spirv.check.TyTag.Function => ok false
            | spirv.check.TyTag.Struct => ok false
            | spirv.check.TyTag.Void => ok false
        else ok false
      if result_ok = true then
        if operands_ok = true then k
        else ok (core.result.Result.Err (spirv.check.CheckError.TypeMismatch index), defs, tys)
      else ok (core.result.Result.Err (spirv.check.CheckError.TypeMismatch index), defs, tys))
    = (if sudotResultOk tys.val ty = true then
        (if sudotOperandsOk tys.val a b = true then k
         else ok (core.result.Result.Err (spirv.check.CheckError.TypeMismatch index), defs, tys))
       else ok (core.result.Result.Err (spirv.check.CheckError.TypeMismatch index), defs, tys)) := by
  rw [sudotResultOk_match (tys := tys.val), sudotIntOk_match (tys := tys.val) (id := a),
    sudotIntOk_match (tys := tys.val) (id := b)]
  simp only [bind_tc_ok]
  by_cases hA : sudotIntOk tys.val a = true <;>
  by_cases hB : sudotIntOk tys.val b = true <;>
  by_cases hR : sudotResultOk tys.val ty = true <;>
  simp_all [sudotOperandsOk, if_true, if_false]

/-- The same block with `t` destructured by the `let (ty, a, b) := t` the extractor emits. -/
theorem apply_shape_sudot (index : Std.Usize)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag)) (defs : alloc.vec.Vec Std.U32)
    (t : Std.U32 × Std.U32 × Std.U32)
    (k : Result ((core.result.Result Unit spirv.check.CheckError) × (alloc.vec.Vec Std.U32) ×
      (alloc.vec.Vec (Std.U32 × spirv.check.TyTag)))) :
    (do
      let (ty, a, b) := t
      let result_ok ←
        match firstLookup tys.val ty with
        | none => ok false
        | some tt =>
          match tt with
          | spirv.check.TyTag.Int width => ok (width >= 32#u32)
          | spirv.check.TyTag.Bool => ok false
          | spirv.check.TyTag.Float _ => ok false
          | spirv.check.TyTag.Vec => ok false
          | spirv.check.TyTag.Pointer => ok false
          | spirv.check.TyTag.RuntimeArray => ok false
          | spirv.check.TyTag.Function => ok false
          | spirv.check.TyTag.Struct => ok false
          | spirv.check.TyTag.Void => ok false
      let b1 ←
        match firstLookup tys.val a with
        | none => ok false
        | some tt =>
          match tt with
          | spirv.check.TyTag.Int i =>
            match i with
            | 32#uscalar => ok true
            | _ => ok false
          | spirv.check.TyTag.Bool => ok false
          | spirv.check.TyTag.Float _ => ok false
          | spirv.check.TyTag.Vec => ok false
          | spirv.check.TyTag.Pointer => ok false
          | spirv.check.TyTag.RuntimeArray => ok false
          | spirv.check.TyTag.Function => ok false
          | spirv.check.TyTag.Struct => ok false
          | spirv.check.TyTag.Void => ok false
      let operands_ok ←
        if b1 = true then
          match firstLookup tys.val b with
          | none => ok false
          | some tt =>
            match tt with
            | spirv.check.TyTag.Int i =>
              match i with
              | 32#uscalar => ok true
              | _ => ok false
            | spirv.check.TyTag.Bool => ok false
            | spirv.check.TyTag.Float _ => ok false
            | spirv.check.TyTag.Vec => ok false
            | spirv.check.TyTag.Pointer => ok false
            | spirv.check.TyTag.RuntimeArray => ok false
            | spirv.check.TyTag.Function => ok false
            | spirv.check.TyTag.Struct => ok false
            | spirv.check.TyTag.Void => ok false
        else ok false
      if result_ok = true then
        if operands_ok = true then k
        else ok (core.result.Result.Err (spirv.check.CheckError.TypeMismatch index), defs, tys)
      else ok (core.result.Result.Err (spirv.check.CheckError.TypeMismatch index), defs, tys))
    = (do
      let (ty, a, b) := t
      if sudotResultOk tys.val ty = true then
        (if sudotOperandsOk tys.val a b = true then k
         else ok (core.result.Result.Err (spirv.check.CheckError.TypeMismatch index), defs, tys))
      else ok (core.result.Result.Err (spirv.check.CheckError.TypeMismatch index), defs, tys)) := by
  rcases t with ⟨ty, a, b⟩
  exact apply_shape_sudot_checks index tys defs ty a b k


/-- **`apply_shape` spec.** The extracted `spirv.check.apply_shape` returns exactly
`applyShapePure`: `BadStorageClass`, then `BadDecoration`, then the `sudot` `TypeMismatch` checks,
then the used-id scan (`UndefinedId` at the first absent used id), then the duplicate-result-id
refusal, the `defs` push and the `ty_insert` update.

Proof: case on `storage_class`/`decoration`/`sudot` (twelve branches of the extracted body); the
admission scans read through `storage_class_admitted_spec`/`decoration_admitted_spec`, the three
`ty_get` calls through `ty_get_deref_spec`, the `sudot` block through `apply_shape_sudot_checks`
(applied by `Eq.trans`, which unifies the extracted literal-pattern matcher with this module's up to
definitional unfolding, where `rw` requires syntactic equality), and the eight scan loops through
`apply_shape_loopN_spec`; the remaining `if`s split on identical conditions on both sides. -/
theorem apply_shape_spec (index : Std.Usize) (shape : spirv.check.Shape)
    (defs : alloc.vec.Vec Std.U32) (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag))
    (hdefs : defs.val.length < Usize.max) (htys : tys.val.length < Usize.max) :
    spirv.check.apply_shape index shape defs tys =
      ok (applyShapePure index shape defs tys hdefs htys) := by
  unfold spirv.check.apply_shape applyShapePure applyShapeSudotPure
  rcases hsc : shape.storage_class with _ | sc <;>
  rcases hdc : shape.decoration with _ | dc <;>
  rcases hsd : shape.sudot with _ | ⟨ty, a, b⟩ <;>
  simp only [storage_class_admitted_spec, decoration_admitted_spec, bind_tc_ok]
  all_goals (
    try simp only [ty_get_deref_spec, bind_tc_ok]
    try split_ifs
    all_goals (
      try dsimp only
      try refine (apply_shape_sudot_checks _ _ _ _ _ _ _).trans ?_
      try first
      | rw [apply_shape_loop0_spec _ _ _ _ _ _ hdefs htys]
      | rw [apply_shape_loop1_spec _ _ _ _ _ _ hdefs htys]
      | rw [apply_shape_loop2_spec _ _ _ _ _ _ hdefs htys]
      | rw [apply_shape_loop3_spec _ _ _ _ _ _ hdefs htys]
      | rw [apply_shape_loop4_spec _ _ _ _ _ _ hdefs htys]
      | rw [apply_shape_loop5_spec _ _ _ _ _ _ hdefs htys]
      | rw [apply_shape_loop6_spec _ _ _ _ _ _ hdefs htys]
      | rw [apply_shape_loop7_spec _ _ _ _ _ _ hdefs htys]
      all_goals first | rfl | (split_ifs; all_goals rfl)))

end Spirv
