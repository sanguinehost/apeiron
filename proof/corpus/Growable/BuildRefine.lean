import ApeironWord
import Apeiron.Table.Basic
import Growable.InsertRefine
import Growable.Schedule

/-!
# The ported sequential `build` fold: stream independence and the canonical link

This is obligation 3 of `ht-port receipt` §4: the ported `build` fold is
independent of the port's `home` and sentinel.  The generated bodies are the real sequential build
of `proof/ApeironWord.lean`:

| generated declaration | `proof/ApeironWord.lean` | Rust source |
|---|---|---|
| `table.Table.build_at_sequential_loop.body` | 23355-23386 | `src/table.rs:1401-1407` |
| `table.Table.build_at_sequential_loop` | 23391-23406 | `src/table.rs:1401-1407` |
| `table.Table.build_at_sequential` | 23410-23446 | `src/table.rs:1387-1413` |
| `word_stored_table_build_at` | 24163-24173 | `lib.rs:434-439` |
| `word_regen_table_build_at` | 24193-24206 | `lib.rs:468-473` |

The content below is the *stream* obligation, stated and proved unconditionally over the generated
loop: the fold hands the input slice's own `(face, key, value)` to the generated step
`insert_with_reserved`, at every index, in slice order, and never substitutes a constant.  The
generated loop body (`23355-23386`) reads only `kvs`, its own `failure`/`i` scalars, and the table
`t`; it does not read `t.home` and it does not read `table.COORD_EMPTY`.  The route value and the
vacant sentinel therefore enter the build *only* inside `insert_with_reserved`, i.e. only as the
placement of an accepted pair, which is exactly the licence §4 obligation 3 names.

The canonical link is `Apeiron.Table.build` / `Apeiron.Table.build_eq_of_key_streams_eq`
(`../sanguine/proof/Apeiron/Table/Basic.lean:121,169`, corpus revision
`880aa57e47e782ed7a9d9ee1974cfc87360b446b`): the canonical fold depends on the stream only through
each key's value subsequences.  §5 (below) composes the generated fold with a named per-step
denotation predicate; that predicate is the whole-table `RepR` insert-preservation (read-map row 4,
`Growable/Read.lean:52-58`), whose unconditional form is not produced here and is named as the
residual.

`is_intact` is not an input: no statement below mentions it.
-/

namespace Apeiron.GrowableBuildRefine

open Aeneas Aeneas.Std Result
open apeiron_word
open Apeiron.GrowableSchedule

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

/-- The generated sequential step, as a `List` fold.  Each entry `((face, k), v)` is handed to the
generated `table.Table.insert_with_reserved` exactly as the loop body hands `kvs[i]`; the first
`Err` short-circuits, matching the loop body's `failure` thread.  This is the fold the generated
`build_at_sequential_loop` computes (proved in `build_at_sequential_loop_eq_foldList`). -/
def foldList {K V H R F Faces : Type} {N : Std.Usize}
    (corecloneCloneInst : core.clone.Clone K) (corecloneCloneInst1 : core.clone.Clone V)
    (HomeInst : table.Home H K Faces) (keyrepKeyRepInst : table.keyrep.KeyRep R K H)
    (MergeInst : table.Merge F V) (merge : F)
    (entries : List ((Faces × K) × V)) (t : table.Table K V H R N)
    (failure : Option table.StorageError) :
    Result (table.Table K V H R N × Option table.StorageError) :=
  match failure with
  | some _ => ok (t, failure)
  | none =>
    match entries with
    | [] => ok (t, failure)
    | ((face, k), v) :: rest =>
      do
        let t1 ← corecloneCloneInst.clone k
        let t2 ← corecloneCloneInst1.clone v
        let (r, t3) ←
          table.Table.insert_with_reserved corecloneCloneInst corecloneCloneInst1 HomeInst
            keyrepKeyRepInst MergeInst t merge face t1 t2
        let failure1 ←
          match r with
          | core.result.Result.Ok _ => ok failure
          | core.result.Result.Err error => ok (some error)
        foldList corecloneCloneInst corecloneCloneInst1 HomeInst keyrepKeyRepInst MergeInst merge
          rest t3 failure1

/-- `Slice.len` is the `List` length, as a `Usize`. -/
@[simp] theorem slice_len_val {α : Type} (kvs : Slice α) : (Slice.len kvs).val = kvs.val.length := by
  simp [Slice.len, Slice.length]

/-- `i` past the slice has an empty residual. -/
theorem drop_eq_nil_of_le {α : Type} {kvs : Slice α} {i : Usize}
    (h : kvs.val.length ≤ i.val) : kvs.val.drop i.val = [] :=
  List.drop_eq_nil_of_le h

/-- A residual list, with the index bound, fixes its head and its successor residual. -/
theorem getElem_and_drop_of_drop_cons {α : Type} {l : List α} {i : Usize} {e : α} {es : List α}
    (h : l.drop i.val = e :: es) (hlt : i.val < l.length) :
    l[i.val]'hlt = e ∧ l.drop (i.val + 1) = es := by
  have hd : l.drop i.val = l[i.val]'hlt :: l.drop (i.val + 1) :=
    List.drop_eq_getElem_cons (i := i.val) (l := l) hlt
  rw [hd] at h
  exact (List.cons.injEq ..).mp h

/-! ## The generated loop body: `done` and `cont` inversions

`build_at_sequential_loop.body` (`proof/ApeironWord.lean:23355-23386`) is a two-branch `if`: past
the slice or with a recorded `failure` it returns `done (t, failure)`; below the slice with no
failure it indexes the slice, clones the pair, hands it to `table.Table.insert_with_reserved`, and
returns `cont` of the step's table and failure flag.  The two lemmas below read that body. -/

/-- A `done` outcome of the generated loop body below the slice is the fold's stop state: the fold
over the residual entry list returns exactly the `done` pair. -/
theorem build_at_sequential_loop_body_done {K V H R F Faces : Type} {N : Std.Usize}
    (corecloneCloneInst : core.clone.Clone K) (corecloneCloneInst1 : core.clone.Clone V)
    (HomeInst : table.Home H K Faces) (keyrepKeyRepInst : table.keyrep.KeyRep R K H)
    (MergeInst : table.Merge F V) (merge : F)
    (kvs : Slice ((Faces × K) × V)) (t : table.Table K V H R N)
    (failure : Option table.StorageError) (i : Std.Usize)
    (t' : table.Table K V H R N) (failure' : Option table.StorageError)
    (hbody : table.Table.build_at_sequential_loop.body corecloneCloneInst corecloneCloneInst1
        HomeInst keyrepKeyRepInst MergeInst merge kvs t failure i
      = ok (ControlFlow.done (t', failure'))) :
    foldList corecloneCloneInst corecloneCloneInst1 HomeInst keyrepKeyRepInst
        MergeInst merge (kvs.val.drop i.val) t failure = ok (t', failure') := by
  unfold table.Table.build_at_sequential_loop.body at hbody
  dsimp only at hbody
  by_cases hlt : i < Slice.len kvs
  · rw [if_pos hlt] at hbody
    by_cases hb : core.option.Option.is_none failure = true
    · rw [if_pos hb] at hbody
      simp only [Bind.bind] at hbody
      obtain ⟨p, _, hbody⟩ := bind_eq_ok (Slice.index_usize kvs i) _ _ hbody
      obtain ⟨t1, _, hbody⟩ := bind_eq_ok (corecloneCloneInst.clone p.1.2) _ _ hbody
      obtain ⟨t2, _, hbody⟩ := bind_eq_ok (corecloneCloneInst1.clone p.2) _ _ hbody
      obtain ⟨⟨r, t3⟩, _, hbody⟩ :=
        bind_eq_ok (table.Table.insert_with_reserved corecloneCloneInst corecloneCloneInst1
          HomeInst keyrepKeyRepInst MergeInst t merge p.1.1 t1 t2) _ _ hbody
      obtain ⟨failure1, _, hbody⟩ := bind_eq_ok (match r with
        | core.result.Result.Ok _ => ok failure
        | core.result.Result.Err error => ok (some error)) _ _ hbody
      obtain ⟨i2, _, hbody⟩ := bind_eq_ok (i + 1#usize) _ _ hbody
      exact absurd (Result.ok.injEq.mp hbody) (by simp)
    · rw [if_neg hb] at hbody
      have hx := ControlFlow.done.inj (Result.ok.injEq.mp hbody)
      rw [Prod.mk.injEq] at hx
      rw [← hx.1, ← hx.2]
      cases hf : failure with
      | none => simp [core.option.Option.is_none, hf] at hb
      | some e => simp [foldList]
  · rw [if_neg hlt] at hbody
    have hx := ControlFlow.done.inj (Result.ok.injEq.mp hbody)
    rw [Prod.mk.injEq] at hx
    rw [← hx.1, ← hx.2]
    have hge : kvs.val.length ≤ i.val := by
      have hle : Slice.len kvs ≤ i := not_lt.mp hlt
      simpa using hle
    rw [List.drop_eq_nil_of_le hge]
    cases failure <;> rfl

/-- A `cont` outcome of the generated loop body below the slice exposes the missing failure, the
indexed pair, the two clones, the generated `insert_with_reserved` call and its table/error flag, and
the successor index.  No `home` value and no sentinel is substituted: the pair delivered is the
slice's own `kvs[i]`. -/
theorem build_at_sequential_loop_body_cont {K V H R F Faces : Type} {N : Std.Usize}
    (corecloneCloneInst : core.clone.Clone K) (corecloneCloneInst1 : core.clone.Clone V)
    (HomeInst : table.Home H K Faces) (keyrepKeyRepInst : table.keyrep.KeyRep R K H)
    (MergeInst : table.Merge F V) (merge : F)
    (kvs : Slice ((Faces × K) × V)) (t : table.Table K V H R N)
    (failure : Option table.StorageError) (i : Std.Usize)
    (t' : table.Table K V H R N) (failure' : Option table.StorageError) (i' : Std.Usize)
    (hbody : table.Table.build_at_sequential_loop.body corecloneCloneInst corecloneCloneInst1
        HomeInst keyrepKeyRepInst MergeInst merge kvs t failure i
      = ok (ControlFlow.cont (t', failure', i'))) :
    failure = none ∧ i < Slice.len kvs ∧
    ∃ (p : (Faces × K) × V) (t1 : K) (t2 : V)
      (r : core.result.Result Unit table.StorageError) (failure1 : Option table.StorageError),
      Slice.index_usize kvs i = ok p ∧
      corecloneCloneInst.clone p.1.2 = ok t1 ∧
      corecloneCloneInst1.clone p.2 = ok t2 ∧
      table.Table.insert_with_reserved corecloneCloneInst corecloneCloneInst1 HomeInst
        keyrepKeyRepInst MergeInst t merge p.1.1 t1 t2 = ok (r, t') ∧
      (match r with
        | core.result.Result.Ok _ => ok failure
        | core.result.Result.Err error => ok (some error)) = ok failure1 ∧
      failure' = failure1 ∧
      i + 1#usize = ok i' := by
  unfold table.Table.build_at_sequential_loop.body at hbody
  dsimp only at hbody
  by_cases hlt : i < Slice.len kvs
  · rw [if_pos hlt] at hbody
    by_cases hb : core.option.Option.is_none failure = true
    · rw [if_pos hb] at hbody
      have hnone : failure = none := by
        cases failure with
        | none => rfl
        | some e => simp [core.option.Option.is_none] at hb
      simp only [Bind.bind] at hbody
      obtain ⟨p, hp, hbody⟩ := bind_eq_ok (Slice.index_usize kvs i) _ _ hbody
      obtain ⟨t1, ht1, hbody⟩ := bind_eq_ok (corecloneCloneInst.clone p.1.2) _ _ hbody
      obtain ⟨t2, ht2, hbody⟩ := bind_eq_ok (corecloneCloneInst1.clone p.2) _ _ hbody
      obtain ⟨⟨r, t3⟩, hins, hbody⟩ :=
        bind_eq_ok (table.Table.insert_with_reserved corecloneCloneInst corecloneCloneInst1
          HomeInst keyrepKeyRepInst MergeInst t merge p.1.1 t1 t2) _ _ hbody
      obtain ⟨failure1, hf1, hbody⟩ := bind_eq_ok (match r with
        | core.result.Result.Ok _ => ok failure
        | core.result.Result.Err error => ok (some error)) _ _ hbody
      obtain ⟨i2, hi2, hbody⟩ := bind_eq_ok (i + 1#usize) _ _ hbody
      have hx : (t3, failure1, i2) = (t', failure', i') :=
        ControlFlow.cont.inj (Result.ok.injEq.mp hbody)
      simp only [Prod.mk.injEq, and_assoc] at hx
      obtain ⟨rfl, rfl, rfl⟩ := hx
      exact ⟨hnone, hlt, p, t1, t2, r, failure1, hp, ht1, ht2, hins, hf1, rfl, hi2⟩
    · rw [if_neg hb] at hbody
      exact absurd (Result.ok.injEq.mp hbody) (by simp)
  · rw [if_neg hlt] at hbody
    exact absurd (Result.ok.injEq.mp hbody) (by simp)

/-! ## The fold equation

`build_at_sequential_loop` is the ordered left fold of the generated step
`table.Table.insert_with_reserved` over the slice's own entry list, short-circuiting at the first
`Err`.  Stated as the loop specification: when the loop succeeds, its result is exactly that fold.
The `home` value and the sentinel do not occur on the right: they enter only through
`insert_with_reserved`'s placement of an accepted pair. -/

/-- The residual fold at a `done` or end-of-slice state is the state itself. -/
theorem foldList_done_self {K V H R F Faces : Type} {N : Std.Usize}
    (corecloneCloneInst : core.clone.Clone K) (corecloneCloneInst1 : core.clone.Clone V)
    (HomeInst : table.Home H K Faces) (keyrepKeyRepInst : table.keyrep.KeyRep R K H)
    (MergeInst : table.Merge F V) (merge : F)
    (entries : List ((Faces × K) × V)) (t : table.Table K V H R N)
    (failure : Option table.StorageError)
    (h : entries = [] ∨ ∃ e, failure = some e) :
    foldList corecloneCloneInst corecloneCloneInst1 HomeInst keyrepKeyRepInst
        MergeInst merge entries t failure = ok (t, failure) := by
  rcases h with rfl | ⟨e, rfl⟩
  · cases failure <;> rfl
  · simp [foldList]

/-- **The ported sequential build fold is the stream fold.**  A successful
`table.Table.build_at_sequential_loop` returns the ordered left fold of the generated
`table.Table.insert_with_reserved` over the slice's own entries, with the first `Err` short-circuiting.
Generated body: `proof/ApeironWord.lean:23355-23386` (body), `23391-23406` (loop). -/
theorem build_at_sequential_loop_spec {K V H R F Faces : Type} {N : Std.Usize}
    (corecloneCloneInst : core.clone.Clone K) (corecloneCloneInst1 : core.clone.Clone V)
    (HomeInst : table.Home H K Faces) (keyrepKeyRepInst : table.keyrep.KeyRep R K H)
    (MergeInst : table.Merge F V) (merge : F)
    (kvs : Slice ((Faces × K) × V)) (t : table.Table K V H R N)
    (failure : Option table.StorageError) (i : Std.Usize) :
    ∀ y, table.Table.build_at_sequential_loop corecloneCloneInst corecloneCloneInst1 HomeInst
        keyrepKeyRepInst MergeInst merge kvs t failure i = ok y →
      foldList corecloneCloneInst corecloneCloneInst1 HomeInst keyrepKeyRepInst MergeInst
        merge (kvs.val.drop i.val) t failure = ok y := by
  unfold table.Table.build_at_sequential_loop
  refine loop_ok_spec
    (measure := fun x : table.Table K V H R N × Option table.StorageError × Std.Usize =>
      (kvs.val.drop x.2.2.val).length)
    (inv := fun x =>
      foldList corecloneCloneInst corecloneCloneInst1 HomeInst keyrepKeyRepInst MergeInst merge
          (kvs.val.drop x.2.2.val) x.1 x.2.1
        = foldList corecloneCloneInst corecloneCloneInst1 HomeInst keyrepKeyRepInst MergeInst merge
            (kvs.val.drop i.val) t failure)
    (post := fun y : table.Table K V H R N × Option table.StorageError =>
      foldList corecloneCloneInst corecloneCloneInst1 HomeInst keyrepKeyRepInst MergeInst merge
            (kvs.val.drop i.val) t failure = ok y)
    (body := fun x => table.Table.build_at_sequential_loop.body corecloneCloneInst
      corecloneCloneInst1 HomeInst keyrepKeyRepInst MergeInst merge kvs x.1 x.2.1 x.2.2)
    ?_ (t, failure, i) rfl
  rintro ⟨t₀, failure₀, i₀⟩ hinv
  constructor
  · rintro ⟨t', failure'⟩ hbody
    dsimp only at hinv ⊢
    have hdone := build_at_sequential_loop_body_done corecloneCloneInst corecloneCloneInst1
      HomeInst keyrepKeyRepInst MergeInst merge kvs t₀ failure₀ i₀ t' failure' hbody
    rw [← hinv]
    exact hdone
  · rintro ⟨t', failure', i'⟩ hbody
    obtain ⟨hnone, hlt, p, t1, t2, r, failure1, hp, ht1, ht2, hins, hf1, hf', hi'⟩ :=
      build_at_sequential_loop_body_cont corecloneCloneInst corecloneCloneInst1 HomeInst
        keyrepKeyRepInst MergeInst merge kvs t₀ failure₀ i₀ t' failure' i' hbody
    dsimp only at hinv ⊢
    have hltv : i₀.val < kvs.val.length := by simpa using hlt
    have hi'v : i'.val = i₀.val + 1 := add_val_of_ok hi'
    have hdrop : kvs.val.drop i₀.val = p :: kvs.val.drop (i₀.val + 1) := by
      have hget := slice_index_ok kvs i₀ hltv
      have hpe : kvs.val[i₀.val]'hltv = p := by
        rw [hget] at hp
        exact Result.ok.injEq.mp hp
      rw [List.drop_eq_getElem_cons (i := i₀.val) (l := kvs.val) hltv, hpe]
    rw [hnone] at hf1
    constructor
    · -- invariant preservation: the step consumed the head of the residual
      rw [← hinv]
      rw [show kvs.val.drop i'.val = kvs.val.drop (i₀.val + 1) by rw [hi'v],
        hdrop, hnone, hf']
      cases r with
      | Ok a =>
          obtain rfl : failure1 = none := by simpa using (Result.ok.injEq.mp hf1).symm
          simp [foldList, ht1, ht2, hins, bind_ok, bind_tc_ok, Bind.bind]
      | Err e =>
          obtain rfl : failure1 = some e := by simpa using (Result.ok.injEq.mp hf1).symm
          simp [foldList, ht1, ht2, hins, bind_ok, bind_tc_ok, Bind.bind]
    · -- the residual strictly shrinks
      have hdecr : (kvs.val.drop i'.val).length < (kvs.val.drop i₀.val).length := by
        rw [hi'v, List.length_drop, List.length_drop]
        omega
      exact hdecr

/-! ## The canonical link: `Apeiron.Table.build`

The generated fold above consumes the slice's own entries through `insert_with_reserved`.  The
canonical object is `Apeiron.Table.build` (`../sanguine/proof/Apeiron/Table/Basic.lean:121`),
the `List.foldl` of the write `Apeiron.Table.insertWith` (`:80`) from the constant default.  Below:
the canonical fold, its exact equality with the generated fold's entry list, and the composition
that carries the generated fold's per-step denotation to the canonical built map. -/

/-- The canonical ordered build, read as the `List.foldl` of `Apeiron.Table.insertWith`. -/
def buildFold {Faces K V : Type} [DecidableEq (Faces × K)] (mergeFn : V → V → V)
    (entries : List ((Faces × K) × V)) (m : Apeiron.Table.HMap (Faces × K) V) :
    Apeiron.Table.HMap (Faces × K) V :=
  entries.foldl (fun m e => Apeiron.Table.insertWith mergeFn e.1 e.2 m) m

/-- `Apeiron.Table.build` is exactly the canonical fold from the constant default:
`Apeiron.Table.build mergeFn default entries = buildFold mergeFn entries (fun _ => default)`.
Canonical source: `Apeiron/Table/Basic.lean:121-122`. -/
theorem canonical_build_eq_buildFold {Faces K V : Type} [DecidableEq (Faces × K)]
    (mergeFn : V → V → V) (default : V) (entries : List ((Faces × K) × V)) :
    Apeiron.Table.build mergeFn default entries = buildFold mergeFn entries (fun _ => default) :=
  rfl

/-- **The built map is a function of the key-value input alone.**  Two entry streams with equal
per-key value sequences build the same canonical map, under every merge —
`Apeiron.Table.build_eq_of_key_streams_eq` (`Apeiron/Table/Basic.lean:169`) at the word entry type.
The generated `build_at_sequential_loop` hands exactly such an entry stream to its step
(`build_at_sequential_loop_spec`), so the route and the sentinel can only reorder the fold's
placement, never the map it denotes. -/
theorem canonical_build_eq_of_key_streams_eq {Faces K V : Type} [DecidableEq (Faces × K)]
    (mergeFn : V → V → V) (default : V) {entries₁ entries₂ : List ((Faces × K) × V)}
    (h : ∀ x : Faces × K, (entries₁.filter (fun e => e.1 = x)).map Prod.snd
        = (entries₂.filter (fun e => e.1 = x)).map Prod.snd) :
    Apeiron.Table.build mergeFn default entries₁ = Apeiron.Table.build mergeFn default entries₂ :=
  Apeiron.Table.build_eq_of_key_streams_eq mergeFn default h

/-- **The generated fold refines the canonical build, modulo the per-step denotation.**  Let
`Denote` be any predicate relating a generated table to a canonical map, and suppose the generated
step `table.Table.insert_with_reserved` preserves it as `Apeiron.Table.insertWith` (this is the
whole-table `RepR` insert-preservation, read-map row 4 of `Growable/Read.lean:52-58`; it is carried,
never assumed).  Then a successful `foldList` over `entries` yields a table whose `Denote`d map is
`buildFold mergeFn entries logical`, and so — from the default — `Apeiron.Table.build`.  No `home`
value, no `COORD_EMPTY`, and no placement appears in the conclusion. -/
theorem foldList_logical {K V H R F Faces : Type} {N : Std.Usize} [DecidableEq (Faces × K)]
    (corecloneCloneInst : core.clone.Clone K) (corecloneCloneInst1 : core.clone.Clone V)
    (HomeInst : table.Home H K Faces) (keyrepKeyRepInst : table.keyrep.KeyRep R K H)
    (MergeInst : table.Merge F V) (merge : F) (mergeFn : V → V → V)
    (Denote : table.Table K V H R N → Apeiron.Table.HMap (Faces × K) V → Prop)
    (hstep : ∀ (t : table.Table K V H R N) (logical : Apeiron.Table.HMap (Faces × K) V)
        (face : Faces) (k : K) (v : V) (t1 : K) (t2 : V)
        (r : core.result.Result Unit table.StorageError) (t' : table.Table K V H R N),
      Denote t logical →
      corecloneCloneInst.clone k = ok t1 →
      corecloneCloneInst1.clone v = ok t2 →
      table.Table.insert_with_reserved corecloneCloneInst corecloneCloneInst1 HomeInst
        keyrepKeyRepInst MergeInst t merge face t1 t2 = ok (r, t') →
      ∃ logical', Denote t' logical' ∧
        logical' = (match r with
          | core.result.Result.Ok _ => Apeiron.Table.insertWith mergeFn (face, k) v logical
          | core.result.Result.Err _ => logical))
    (entries : List ((Faces × K) × V)) (t : table.Table K V H R N)
    (failure : Option table.StorageError) (logical : Apeiron.Table.HMap (Faces × K) V)
    (hinit : Denote t logical) :
    ∀ y, foldList corecloneCloneInst corecloneCloneInst1 HomeInst keyrepKeyRepInst MergeInst merge
        entries t failure = ok y →
      ∃ logical', Denote y.1 logical' ∧
        (y.2 = none → logical' = buildFold mergeFn entries logical) := by
  induction entries generalizing t failure logical with
  | nil =>
      intro y h
      cases failure with
      | some e =>
          have hy : y = (t, some e) := by
            have hh : ok (t, some e) = ok y := by simpa [foldList] using h
            exact (Result.ok.injEq.mp hh).symm
          refine ⟨logical, ?_, ?_⟩
          · rw [hy]; exact hinit
          · intro hc; rw [hy] at hc; exact absurd hc (by simp)
      | none =>
          have hy : y = (t, none) := by
            have hh : ok (t, none) = ok y := by simpa [foldList] using h
            exact (Result.ok.injEq.mp hh).symm
          refine ⟨logical, ?_, ?_⟩
          · rw [hy]; exact hinit
          · intro _; rfl
  | cons e es ih =>
      intro y h
      cases failure with
      | some e' =>
          have hy : y = (t, some e') := by
            have hh : ok (t, some e') = ok y := by simpa [foldList] using h
            exact (Result.ok.injEq.mp hh).symm
          refine ⟨logical, ?_, ?_⟩
          · rw [hy]; exact hinit
          · intro hc; rw [hy] at hc; exact absurd hc (by simp)
      | none =>
          simp only [foldList] at h
          obtain ⟨t1, ht1, h⟩ := bind_eq_ok (corecloneCloneInst.clone e.1.2) _ _ h
          obtain ⟨t2, ht2, h⟩ := bind_eq_ok (corecloneCloneInst1.clone e.2) _ _ h
          obtain ⟨⟨r, t3⟩, hins, h⟩ :=
            bind_eq_ok (table.Table.insert_with_reserved corecloneCloneInst corecloneCloneInst1
              HomeInst keyrepKeyRepInst MergeInst t merge e.1.1 t1 t2) _ _ h
          obtain ⟨failure1, hf1, h⟩ := bind_eq_ok (match r with
            | core.result.Result.Ok _ => ok failure
            | core.result.Result.Err error => ok (some error)) _ _ h
          obtain ⟨logical3, hden3, hlog3⟩ :=
            hstep t logical e.1.1 e.1.2 e.2 t1 t2 r t3 hinit ht1 ht2 hins
          obtain ⟨logical', hden', hlog'⟩ := ih t3 failure1 logical3 hden3 y h
          refine ⟨logical', hden', ?_⟩
          intro hy2
          rw [hlog' hy2]
          cases r with
          | Ok a =>
              rw [hlog3]
              simp [buildFold, List.foldl_cons]
          | Err err =>
              exfalso
              have hok : (ok (some err) : Result (Option table.StorageError)) = ok failure1 := by
                simpa using hf1
              have hf1' : failure1 = some err := (Result.ok.injEq.mp hok).symm
              have hy : y = (t3, some err) := by
                subst hf1'
                have hh : foldList corecloneCloneInst corecloneCloneInst1 HomeInst keyrepKeyRepInst
                    MergeInst merge es t3 (some err) = ok (t3, some err) := by simp [foldList]
                exact Result.ok.injEq.mp (h.symm.trans hh)
              rw [hy] at hy2
              exact absurd hy2 (by simp)

end Apeiron.GrowableBuildRefine

/-! ## Cone placement

Each headline's transitive axiom set must be the permitted closed cone
`{propext, Classical.choice, Quot.sound}`.  `AxiomAudit/Core.lean` is another lane's file; the check
is stated here directly, as `Growable/InsertRefine.lean` does. -/
#print axioms Apeiron.GrowableBuildRefine.build_at_sequential_loop_body_done
#print axioms Apeiron.GrowableBuildRefine.build_at_sequential_loop_body_cont
#print axioms Apeiron.GrowableBuildRefine.build_at_sequential_loop_spec
#print axioms Apeiron.GrowableBuildRefine.foldList_done_self
#print axioms Apeiron.GrowableBuildRefine.canonical_build_eq_buildFold
#print axioms Apeiron.GrowableBuildRefine.canonical_build_eq_of_key_streams_eq
#print axioms Apeiron.GrowableBuildRefine.foldList_logical
