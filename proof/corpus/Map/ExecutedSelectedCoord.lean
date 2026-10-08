-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe` (APEIRON-HT-READ): the `Regen<u64>` face's
-- selected read, no `ok` premise.
import Map.ExecutedSelected
import Map.ExecutedReadCoordTotal

/-!
# `SelectedFace` on the executed `Regen<u64>` body

`Map/ExecutedSelected.lean` gives the `Stored<u64>` face (`word_hashtrinity_get`) the pair
`selectedReadRep_total` / `executed_get_refines_canonical_get_total` (lines 159–194): the executed
read refines the canonical last-write `build` read on every `execBuild` state, with no `ok` premise.
This module states the exact `Regen<u64>` twins over `coordExecBuild` states:

* `coordExecutedFace enc` is `SelectedFace` instantiated at the extracted state
  `hashtrinity.Table U64 I64 (Regen U64)` with the extracted bodies: `read` is `word_coordmap_get`,
  `home` is `hashtrinity.Table.home` at `RKS`, `slots` is the slot count, and `insert` is
  `word_coordmap_insert`.
* `coordSelectedReadRep_total` and `coord_executed_get_refines_canonical_get_total` are the twins of
  the `Stored` pair, over `coord_build_refines`' representation `CoordMapRep` with the read's
  totality (`coord_get_ok_of_coordRep`) discharging the partial-correctness premise.
* `coord_build_get_total` is the read-side headline of `coord_build_get_refines`
  (`Map/ExecutedBuild.lean:259`) with its `ok` premise discharged by the same totality lemma.

No `sorry`/`axiom`/`admit`; the cone is `{propext, Classical.choice, Quot.sound}`, `#audit`-ed below.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open Aeneas Aeneas.Std Result
open apeiron_word Apeiron.MapFace

open Classical in
/-- `SelectedFace` at the extracted `Regen<u64>` state, through a key encoding `enc`.  The exact twin
of `executedFace`, with `word_coordmap_get` / `word_coordmap_insert` in place of the `Stored` bodies
and `RKS` in place of `SKS`. -/
noncomputable def coordExecutedFace (enc : MapKey → U64) :
    SelectedFace (Tbl (hashtrinity.algebra.Regen U64)) where
  read s key := if h : ∃ r, word_coordmap_get s (enc key) = ok r then Classical.choose h else none
  home s key :=
    if h : ∃ i, hashtrinity.Table.home core.cmp.EqU64 CI RKS s (enc key) = ok i then
      (Classical.choose h).val else 0
  slots s := s.coords.length
  insert s key v _ :=
    if h : ∃ s', word_coordmap_insert s (enc key) v = ok s' then Classical.choose h else s

theorem coordExecutedFace_read (enc : MapKey → U64) (s : Tbl (hashtrinity.algebra.Regen U64))
    (key : MapKey) (r : Option I64) (h : word_coordmap_get s (enc key) = ok r) :
    (coordExecutedFace enc).read s key = r := by
  have hex : ∃ r, word_coordmap_get s (enc key) = ok r := ⟨r, h⟩
  unfold coordExecutedFace
  dsimp only
  rw [dif_pos hex]
  have := Classical.choose_spec hex
  exact (Result.ok_injective (this.symm.trans h))

open Classical in
/-- **`SelectedReadRep` from the executed `Regen<u64>` read, with no bridge premise.**  Under
`CoordRep s m live`, the executed face `coordExecutedFace enc` satisfies the read contract of
`Map/Selected.lean` at live set `live ∘ enc` and canonical map `key ↦ if live (enc key) then
m (enc key) else default`.  The read's totality (`coord_get_refines_total`) enters directly, so no
`ok` premise remains. -/
theorem coord_selectedReadRep_of_coordRep
    (enc : MapKey → U64) (s : Tbl (hashtrinity.algebra.Regen U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) (hR : CoordRep s m live)
    (default : I64) :
    SelectedReadRep (coordExecutedFace enc) s (fun key => live (enc key))
      (fun key => if live (enc key) then m (enc key) else default) default := by
  have hread : ∀ key, ∃ r, (coordExecutedFace enc).read s key = r ∧
      (∀ v, r = some v ↔ live (enc key) ∧ v = Apeiron.Table.get m (enc key)) ∧
      (¬ live (enc key) → r = none) := by
    intro key
    obtain ⟨r, hr, hspec, hnone⟩ := coord_get_refines_total s m live hR (enc key)
    exact ⟨r, coordExecutedFace_read enc s key r hr, hspec, hnone⟩
  refine ⟨?_, ?_, ?_⟩
  · intro key
    obtain ⟨r, hrd, hsome, _⟩ := hread key
    rw [hrd]
    constructor
    · intro hl
      exact Option.isSome_iff_exists.mpr ⟨_, (hsome _).mpr ⟨hl, rfl⟩⟩
    · intro hs
      obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp hs
      exact ((hsome v).mp hv).1
  · intro key hl
    obtain ⟨r, hrd, hsome, _⟩ := hread key
    rw [hrd, (hsome _).mpr ⟨hl, rfl⟩]
    simp [Apeiron.Table.get, hl]
  · intro key hnl
    obtain ⟨r, hrd, _, hnone⟩ := hread key
    refine ⟨by rw [hrd]; exact hnone hnl, ?_⟩
    simp [Apeiron.Table.get, hnl]

open Classical in
/-- **`SelectedReadRep` on every reachable `coordExecBuild` state.**  From `coordExecBuild n xs = ok s`,
the executed `Regen<u64>` face satisfies the read contract at live set `enc key ∈ xs.map Prod.fst` and
canonical map `Apeiron.Table.get (Apeiron.Table.build (fun new _ => new) 0#i64 xs)`.  The `ok` premise
is discharged by `coord_get_ok_of_coordRep` through `coord_build_refines`.  The exact twin of
`selectedReadRep_total`. -/
theorem coordSelectedReadRep_total
    (enc : MapKey → U64) (n : Usize) (xs : List (U64 × I64)) (hlen : xs.length < nilN)
    (s : Tbl (hashtrinity.algebra.Regen U64)) (h : coordExecBuild n xs = ok s)
    (default : I64) :
    SelectedReadRep (coordExecutedFace enc) s (fun key => enc key ∈ xs.map Prod.fst)
      (fun key => if enc key ∈ xs.map Prod.fst then
        Apeiron.Table.get (Apeiron.Table.build (fun new _ => new) 0#i64 xs) (enc key)
        else default) default := by
  obtain ⟨hR, _⟩ := coord_build_refines n xs hlen s h
  have h := coord_selectedReadRep_of_coordRep enc s
    (Apeiron.Table.build (fun new _ => new) 0#i64 xs) (fun c => c ∈ xs.map Prod.fst) hR.1 default
  refine ⟨fun key => h.liveSet key, ?_, ?_⟩
  · intro key hlive
    rw [h.get_live key hlive]
    congr 1
    simp only [Apeiron.Table.get]
    rw [if_pos hlive, if_pos hlive]
  · intro key hnot
    obtain ⟨h1, h2⟩ := h.get_absent key hnot
    exact ⟨h1, by simp only [Apeiron.Table.get]; exact if_neg hnot⟩

open Classical in
/-- **The executed `Regen<u64>` read refines the canonical build read on every reachable state, no
`ok` premise.**  The exact twin of `executed_get_refines_canonical_get_total`: reading
`coordExecutedFace enc` at a state `coordExecBuild n xs` reaches answers `some value` exactly when
`enc key` is a key of the stream and the canonical last-write build reads `value` there (the default
off the key set). -/
theorem coord_executed_get_refines_canonical_get_total
    (enc : MapKey → U64) (n : Usize) (xs : List (U64 × I64)) (hlen : xs.length < nilN)
    (s : Tbl (hashtrinity.algebra.Regen U64)) (h : coordExecBuild n xs = ok s)
    (default : I64) (key : MapKey) (value : I64) :
    (coordExecutedFace enc).read s key = some value ↔
      (enc key ∈ xs.map Prod.fst) ∧
      (if enc key ∈ xs.map Prod.fst then
        Apeiron.Table.get (Apeiron.Table.build (fun new _ => new) 0#i64 xs) (enc key)
        else default) = value := by
  have hrep := coordSelectedReadRep_total enc n xs hlen s h default
  exact selected_get_refines_canonical_get hrep key value

open Classical in
/-- **`coord_build_get_refines` with its `ok` premise discharged.**  For every key the executed
`Regen<u64>` read of the built table returns `ok r`, with `r = some v` exactly when `k` is a key of
the stream and `v` is the canonical last-write build read there, and `none` off the key set. -/
theorem coord_build_get_total
    (n : Usize) (xs : List (U64 × I64)) (hlen : xs.length < nilN)
    (s : Tbl (hashtrinity.algebra.Regen U64)) (h : coordExecBuild n xs = ok s) (k : U64) :
    ∃ r, word_coordmap_get s k = ok r ∧
      (∀ v, r = some v ↔
          k ∈ xs.map Prod.fst ∧
          v = Apeiron.Table.get (Apeiron.Table.build (fun new _ => new) 0#i64 xs) k) ∧
      (k ∉ xs.map Prod.fst → r = none) := by
  obtain ⟨hR, _⟩ := coord_build_refines n xs hlen s h
  obtain ⟨r, hr⟩ := coord_get_ok_of_coordRep s _ _ hR.1 k
  exact ⟨r, hr, coord_build_get_refines n xs hlen s h k r hr⟩

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.coordExecutedFace
#audit Apeiron.MapExecuted.coordExecutedFace_read
#audit Apeiron.MapExecuted.coord_selectedReadRep_of_coordRep
#audit Apeiron.MapExecuted.coordSelectedReadRep_total
#audit Apeiron.MapExecuted.coord_executed_get_refines_canonical_get_total
#audit Apeiron.MapExecuted.coord_build_get_total
