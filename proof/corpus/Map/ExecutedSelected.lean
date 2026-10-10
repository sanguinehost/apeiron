-- `SelectedFace` rebased onto the executed body.
import Map.ExecutedRead
import Map.ExecutedBuild
import Map.ExecutedReadUncond
import Map.ExecutedReadTotal
import Map.Selected

/-!
# `SelectedFace` on the executed body

`Map/Selected.lean` states the executed map face's read contract `SelectedReadRep F s live map default`
over an abstract `F : SelectedFace State`, and derives it (`selectedReadRep_of_mapRep`) from the
growable table's `RepR` plus the premise `hbridge`, because the executed body had no extraction.

`executedFace enc` is `SelectedFace` instantiated at the extracted state
`hashtrinity.Table U64 I64 (Stored U64)` with the extracted bodies: `read` is `word_hashtrinity_get`,
`home` is `hashtrinity.Table.home`, `slots` is the slot count, and `insert` is
`word_hashtrinity_insert` (the last-write face; the merge argument is not read, as in the executed
wrapper).  Map-face keys reach the `u64` face through an arbitrary encoding `enc : MapKey → U64`.  A
body that does not return `ok` reads as `none`/the unchanged state; the theorem below assumes the
read returns `ok` (`hok`), because this tree proves partial correctness only.

`selectedReadRep_of_tableRep` is the one lemma that replaces `hbridge`: under `TableRep s m live` the
executed face satisfies `SelectedReadRep` with live set `live ∘ enc`, the canonical map that reads
`m ∘ enc` on that set and `default` off it, and the given default.  Its only premises are the
representation, `hok`, and the AVX2-branch contract `avx2_probe = true → Avx2Spec`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open Aeneas Aeneas.Std Result
open apeiron_word Apeiron.MapFace

open Classical in
/-- `SelectedFace` at the extracted `Stored<u64>` state, through a key encoding `enc`. -/
noncomputable def executedFace (enc : MapKey → U64) :
    SelectedFace (Tbl (hashtrinity.algebra.Stored U64)) where
  read s key := if h : ∃ r, word_hashtrinity_get s (enc key) = ok r then Classical.choose h else none
  home s key :=
    if h : ∃ i, hashtrinity.Table.home core.cmp.EqU64 CI SKS s (enc key) = ok i then
      (Classical.choose h).val else 0
  slots s := s.coords.length
  insert s key v _ :=
    if h : ∃ s', word_hashtrinity_insert s (enc key) v = ok s' then Classical.choose h else s

theorem executedFace_read (enc : MapKey → U64) (s : Tbl (hashtrinity.algebra.Stored U64))
    (key : MapKey) (r : Option I64) (h : word_hashtrinity_get s (enc key) = ok r) :
    (executedFace enc).read s key = r := by
  have hex : ∃ r, word_hashtrinity_get s (enc key) = ok r := ⟨r, h⟩
  unfold executedFace
  dsimp only
  rw [dif_pos hex]
  have := Classical.choose_spec hex
  exact (Result.ok_injective (this.symm.trans h))

open Classical in
/-- **`SelectedReadRep` from the executed read, with no bridge premise.**  Under `TableRep s m live`,
the executed face `executedFace enc` satisfies the read contract of `Map/Selected.lean` at live set
`live ∘ enc` and canonical map `key ↦ if live (enc key) then m (enc key) else default`. -/
theorem selectedReadRep_of_tableRep
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (enc : MapKey → U64) (s : Tbl (hashtrinity.algebra.Stored U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) (hR : TableRep s m live)
    (default : I64) (hok : ∀ key, ∃ r, word_hashtrinity_get s (enc key) = ok r) :
    SelectedReadRep (executedFace enc) s (fun key => live (enc key))
      (fun key => if live (enc key) then m (enc key) else default) default := by
  have hread : ∀ key, ∃ r, (executedFace enc).read s key = r ∧
      (∀ v, r = some v ↔ live (enc key) ∧ v = Apeiron.Table.get m (enc key)) ∧
      (¬ live (enc key) → r = none) := by
    intro key
    obtain ⟨r, hr⟩ := hok key
    exact ⟨r, executedFace_read enc s key r hr, get_refines hA s m live hR (enc key) r hr⟩
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

/-! ## The read contract on reachable build states

`selectedReadRep_of_tableRep` still takes `hok`, the premise that the executed read returns `ok`,
because this tree proves partial correctness only.  On the states the executed build actually
reaches, the representation is `build_refines`' `MapRep s (Apeiron.Table.build (fun new _ => new)
0#i64 xs) (· ∈ xs.map Prod.fst)`, so `live` is the key set of the stream, the canonical map is the
canonical last-write build, and the AVX2 premise is discharged by `avx2_spec_holds`. -/

open Classical in
/-- **`SelectedReadRep` on every reachable build state.**  From `execBuild n xs = ok s`, the
executed face `executedFace enc` satisfies the read contract at live set `enc key ∈ xs.map Prod.fst`
and canonical map `Apeiron.Table.get (Apeiron.Table.build (fun new _ => new) 0#i64 xs)`, with the
AVX2 premise discharged by `avx2_spec_holds`. -/
theorem selectedReadRep_reachable
    (enc : MapKey → U64) (n : Usize) (xs : List (U64 × I64)) (hlen : xs.length < nilN)
    (s : Tbl (hashtrinity.algebra.Stored U64)) (h : execBuild n xs = ok s)
    (hok : ∀ key, ∃ r, word_hashtrinity_get s (enc key) = ok r)
    (default : I64) :
    SelectedReadRep (executedFace enc) s (fun key => enc key ∈ xs.map Prod.fst)
      (fun key => if enc key ∈ xs.map Prod.fst then
        Apeiron.Table.get (Apeiron.Table.build (fun new _ => new) 0#i64 xs) (enc key)
        else default) default := by
  obtain ⟨hR, _⟩ := build_refines n xs hlen s h
  have h := selectedReadRep_of_tableRep (fun _ => avx2_spec_holds) enc s
    (Apeiron.Table.build (fun new _ => new) 0#i64 xs) (fun c => c ∈ xs.map Prod.fst) hR.1
    default hok
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
/-- **The executed read refines the canonical build read on every reachable state, with no `hA`,
no `hbridge` and no `TableRep` premise.**  Reading `executedFace enc` at `s` (a state
`execBuild n xs` reaches) answers `some value` exactly when `enc key` is a key of the stream and the
canonical last-write build reads `value` there (the default off the key set).  The AVX2 premise is
`avx2_spec_holds`; the residual premise is `hok`, the partial-correctness boundary of the extracted
body. -/
theorem executed_get_refines_canonical_get_reachable
    (enc : MapKey → U64) (n : Usize) (xs : List (U64 × I64)) (hlen : xs.length < nilN)
    (s : Tbl (hashtrinity.algebra.Stored U64)) (h : execBuild n xs = ok s)
    (hok : ∀ key, ∃ r, word_hashtrinity_get s (enc key) = ok r)
    (default : I64) (key : MapKey) (value : I64) :
    (executedFace enc).read s key = some value ↔
      (enc key ∈ xs.map Prod.fst) ∧
      (if enc key ∈ xs.map Prod.fst then
        Apeiron.Table.get (Apeiron.Table.build (fun new _ => new) 0#i64 xs) (enc key)
        else default) = value := by
  have hrep := selectedReadRep_reachable enc n xs hlen s h hok default
  exact selected_get_refines_canonical_get hrep key value

/-! ## The read contract with the totality premise discharged

`get_ok_of_mapRep` (`Map/ExecutedReadTotal.lean`) proves the executed read returns `ok` on every
`MapRep` state, so on a reachable build state `hok` follows from `build_refines` and no partial
premise remains. -/

open Classical in
/-- **`SelectedReadRep` on every reachable build state, no `hok`.**  The executed read's totality
(`get_ok_of_mapRep`) discharges the partial-correctness premise of `selectedReadRep_reachable`. -/
theorem selectedReadRep_total
    (enc : MapKey → U64) (n : Usize) (xs : List (U64 × I64)) (hlen : xs.length < nilN)
    (s : Tbl (hashtrinity.algebra.Stored U64)) (h : execBuild n xs = ok s)
    (default : I64) :
    SelectedReadRep (executedFace enc) s (fun key => enc key ∈ xs.map Prod.fst)
      (fun key => if enc key ∈ xs.map Prod.fst then
        Apeiron.Table.get (Apeiron.Table.build (fun new _ => new) 0#i64 xs) (enc key)
        else default) default := by
  obtain ⟨hR, _⟩ := build_refines n xs hlen s h
  exact selectedReadRep_reachable enc n xs hlen s h
    (fun key => get_ok_of_mapRep s (Apeiron.Table.build (fun new _ => new) 0#i64 xs)
      (fun c => c ∈ xs.map Prod.fst) hR (enc key)) default

open Classical in
/-- **The executed read refines the canonical build read on every reachable state, no `hok`.**
`executed_get_refines_canonical_get_reachable` with the read's totality discharged. -/
theorem executed_get_refines_canonical_get_total
    (enc : MapKey → U64) (n : Usize) (xs : List (U64 × I64)) (hlen : xs.length < nilN)
    (s : Tbl (hashtrinity.algebra.Stored U64)) (h : execBuild n xs = ok s)
    (default : I64) (key : MapKey) (value : I64) :
    (executedFace enc).read s key = some value ↔
      (enc key ∈ xs.map Prod.fst) ∧
      (if enc key ∈ xs.map Prod.fst then
        Apeiron.Table.get (Apeiron.Table.build (fun new _ => new) 0#i64 xs) (enc key)
        else default) = value := by
  have hrep := selectedReadRep_total enc n xs hlen s h default
  exact selected_get_refines_canonical_get hrep key value

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.selectedReadRep_of_tableRep
#audit Apeiron.MapExecuted.selectedReadRep_reachable
#audit Apeiron.MapExecuted.executed_get_refines_canonical_get_reachable
#audit Apeiron.MapExecuted.get_ok_of_mapRep
#audit Apeiron.MapExecuted.selectedReadRep_total
#audit Apeiron.MapExecuted.executed_get_refines_canonical_get_total
