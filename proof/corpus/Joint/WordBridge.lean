import Denotation
import Slang.Word
import Apeiron.Table.Basic

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Joint

open Aeneas Aeneas.Std

/-- The Rust and Song words carry the same unsigned 64-bit pattern. -/
def WordI64Rel (rust : I64) (song : Slang.Word) : Prop :=
  rust.bv.toNat = song.val

/-- A Song bit pattern identifies at most one extracted Rust word. -/
theorem wordI64Rel_rust_unique {left right : I64} {song : Slang.Word}
    (hleft : WordI64Rel left song) (hright : WordI64Rel right song) : left = right := by
  cases left with
  | mk leftBits =>
    cases right with
    | mk rightBits =>
      congr 1
      apply BitVec.eq_of_toNat_eq
      exact hleft.trans hright.symm

/-- An extracted Rust bit pattern identifies at most one Song word. -/
theorem wordI64Rel_song_unique {rust : I64} {left right : Slang.Word}
    (hleft : WordI64Rel rust left) (hright : WordI64Rel rust right) : left = right := by
  apply Fin.ext
  exact hleft.symm.trans hright

/-- Related word pairs reflect equality on either representation.

    leftRust = rightRust                                 [equality of extracted Rust words]
    ↔                                                    [exactly when, true in both directions]
    leftSong = rightSong                                 [equality of related Song words] -/
theorem wordI64Rel_eq_iff {leftRust rightRust : I64} {leftSong rightSong : Slang.Word}
    (hleft : WordI64Rel leftRust leftSong) (hright : WordI64Rel rightRust rightSong) :
    leftRust = rightRust ↔ leftSong = rightSong := by
  constructor
  · intro hrust
    subst rightRust
    exact wordI64Rel_song_unique hleft hright
  · intro hsong
    subst rightSong
    exact wordI64Rel_rust_unique hleft hright

/-- Pointwise relation between the two canonical total maps. -/
def HMapRel (rustMap : Apeiron.Table.HMap I64 I64)
    (songMap : Apeiron.Table.HMap Slang.Word Slang.Word) : Prop :=
  ∀ rustKey songKey, WordI64Rel rustKey songKey →
    WordI64Rel (Apeiron.Table.get rustMap rustKey) (Apeiron.Table.get songMap songKey)

/-- New-first replacement preserves the pointwise relation across key representations. -/
theorem hMapRel_insertWith_replacement
    {rustMap : Apeiron.Table.HMap I64 I64}
    {songMap : Apeiron.Table.HMap Slang.Word Slang.Word}
    {rustKey rustValue : I64} {songKey songValue : Slang.Word}
    (hmaps : HMapRel rustMap songMap)
    (hkey : WordI64Rel rustKey songKey)
    (hvalue : WordI64Rel rustValue songValue) :
    HMapRel
      (Apeiron.Table.insertWith (fun new _old => new) rustKey rustValue rustMap)
      (Apeiron.Table.insertWith (fun new _old => new) songKey songValue songMap) := by
  intro observedRust observedSong hobserved
  have heq : observedRust = rustKey ↔ observedSong = songKey :=
    wordI64Rel_eq_iff hobserved hkey
  by_cases hrust : observedRust = rustKey
  · have hsong : observedSong = songKey := heq.mp hrust
    simp [Apeiron.Table.get, Apeiron.Table.insertWith, hrust, hsong, hvalue]
  · have hsong : observedSong ≠ songKey := fun equal => hrust (heq.mpr equal)
    simpa [Apeiron.Table.get, Apeiron.Table.insertWith, hrust, hsong] using
      hmaps observedRust observedSong hobserved

end Apeiron.Joint
