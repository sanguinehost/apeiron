import Init.Data.ByteArray.Lemmas
import Init.Data.UInt.Lemmas

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source

/-- A source byte has exactly 256 values. -/
abbrev SourceByte := Fin 256

/-- Source input before conversion to Lean's packed byte array. -/
abbrev SourceBytes := List SourceByte

/-- Pack source bytes without changing their values or order. -/
def encodeBytes (bytes : SourceBytes) : ByteArray :=
  (bytes.map UInt8.ofFin).toByteArray

/-- The packed byte array has exactly the source byte values in order. -/
def ByteRep (source : SourceBytes) (packed : ByteArray) : Prop :=
  packed.data.toList.map UInt8.toFin = source

namespace ByteRep

private theorem concat_eq_append (bytes : List UInt8) (byte : UInt8) :
    bytes.concat byte = bytes ++ [byte] := by
  induction bytes with
  | nil => rfl
  | cons head tail ih =>
      change head :: tail.concat byte = head :: (tail ++ [byte])
      exact congrArg (List.cons head) ih

private theorem append_nil (bytes : List UInt8) : bytes ++ [] = bytes := by
  induction bytes with
  | nil => rfl
  | cons head tail ih =>
      change head :: (tail ++ []) = head :: tail
      exact congrArg (List.cons head) ih

private theorem append_assoc (left middle right : List UInt8) :
    (left ++ middle) ++ right = left ++ (middle ++ right) := by
  induction left with
  | nil => rfl
  | cons head tail ih =>
      change head :: ((tail ++ middle) ++ right) = head :: (tail ++ (middle ++ right))
      exact congrArg (List.cons head) ih

private theorem toByteArray_loop_view (remaining : List UInt8) (packed : ByteArray) :
    (List.toByteArray.loop remaining packed).data.toList =
      packed.data.toList ++ remaining := by
  induction remaining generalizing packed with
  | nil =>
      change packed.data.toList = packed.data.toList ++ []
      exact (append_nil packed.data.toList).symm
  | cons head tail ih =>
      change (List.toByteArray.loop tail (packed.push head)).data.toList =
        packed.data.toList ++ head :: tail
      calc
        (List.toByteArray.loop tail (packed.push head)).data.toList =
            (packed.push head).data.toList ++ tail := ih (packed := packed.push head)
        _ = (packed.data.toList ++ [head]) ++ tail := by
          cases packed with
          | mk data =>
              cases data with
              | mk bytes =>
                  change (bytes.concat head) ++ tail = (bytes ++ [head]) ++ tail
                  exact congrArg (fun front => front ++ tail) (concat_eq_append bytes head)
        _ = packed.data.toList ++ ([head] ++ tail) :=
          append_assoc packed.data.toList [head] tail
        _ = packed.data.toList ++ head :: tail := rfl

private theorem toByteArray_view (bytes : List UInt8) :
    bytes.toByteArray.data.toList = bytes := by
  unfold List.toByteArray
  exact toByteArray_loop_view bytes ByteArray.empty

private theorem length_map_toFin (bytes : List UInt8) :
    (bytes.map UInt8.toFin).length = bytes.length := by
  induction bytes with
  | nil => rfl
  | cons _ tail ih =>
      change Nat.succ (tail.map UInt8.toFin).length = Nat.succ tail.length
      exact congrArg Nat.succ ih

/-- Every source byte list is represented by its encoding. -/
theorem encodeBytes (source : SourceBytes) : ByteRep source (Slang.Source.encodeBytes source) := by
  unfold ByteRep Slang.Source.encodeBytes
  calc
    (source.map UInt8.ofFin).toByteArray.data.toList.map UInt8.toFin =
        (source.map UInt8.ofFin).map UInt8.toFin :=
      congrArg (List.map UInt8.toFin) (toByteArray_view (source.map UInt8.ofFin))
    _ = source := by
      induction source with
      | nil => rfl
      | cons head tail ih =>
          change head :: (tail.map UInt8.ofFin).map UInt8.toFin = head :: tail
          exact congrArg (List.cons head) ih

/-- One packed byte array represents at most one source byte list. -/
theorem eq {left right : SourceBytes} {packed : ByteArray}
    (hleft : ByteRep left packed) (hright : ByteRep right packed) : left = right := by
  exact hleft.symm.trans hright

/-- Representation by an encoding determines the encoded source bytes. -/
theorem eq_of_encodeBytes {left right : SourceBytes}
    (h : ByteRep left (Slang.Source.encodeBytes right)) : left = right :=
  h.eq (encodeBytes right)

/-- A representation agrees with its canonical encoding. -/
theorem encodeBytes_eq {source : SourceBytes} {packed : ByteArray}
    (h : ByteRep source packed) : packed = Slang.Source.encodeBytes source := by
  have inverse : packed.data.toList = source.map UInt8.ofFin := by
    calc
      packed.data.toList =
          (packed.data.toList.map UInt8.toFin).map UInt8.ofFin := by
            induction packed.data.toList with
            | nil => rfl
            | cons head tail ih =>
                change head :: tail = UInt8.ofFin (UInt8.toFin head) ::
                  (tail.map UInt8.toFin).map UInt8.ofFin
                change head :: tail = head :: (tail.map UInt8.toFin).map UInt8.ofFin
                exact congrArg (List.cons head) ih
      _ = source.map UInt8.ofFin := congrArg (List.map UInt8.ofFin) h
  have dataEq : packed.data = (Slang.Source.encodeBytes source).data := by
    calc
      packed.data = packed.data.toList.toArray := Array.toArray_toList.symm
      _ = (source.map UInt8.ofFin).toArray := congrArg List.toArray inverse
      _ = (Slang.Source.encodeBytes source).data := by
        unfold Slang.Source.encodeBytes
        exact congrArg List.toArray (toByteArray_view (source.map UInt8.ofFin)).symm
  exact congrArg ByteArray.mk dataEq

/-- Representation is equivalent to equality with the canonical encoding. -/
theorem iff_eq_encodeBytes {source : SourceBytes} {packed : ByteArray} :
    ByteRep source packed ↔ packed = Slang.Source.encodeBytes source := by
  constructor
  · exact encodeBytes_eq
  · intro h
    cases h
    exact encodeBytes source

/-- Packed and source views have equal lengths. -/
theorem size_eq_length {source : SourceBytes} {packed : ByteArray}
    (h : ByteRep source packed) : packed.size = source.length := by
  calc
    packed.size = packed.data.toList.length := rfl
    _ = (packed.data.toList.map UInt8.toFin).length :=
      (length_map_toFin packed.data.toList).symm
    _ = source.length := congrArg List.length h

/-- The packed view of the canonical encoding is the source mapped to unsigned bytes. -/
theorem encodeBytes_data (source : SourceBytes) :
    (Slang.Source.encodeBytes source).data.toList = source.map UInt8.ofFin := by
  have representative := encodeBytes source
  calc
    (Slang.Source.encodeBytes source).data.toList =
        ((Slang.Source.encodeBytes source).data.toList.map UInt8.toFin).map UInt8.ofFin := by
          induction (Slang.Source.encodeBytes source).data.toList with
          | nil => rfl
          | cons head tail ih =>
              change head :: tail = UInt8.ofFin (UInt8.toFin head) ::
                (tail.map UInt8.toFin).map UInt8.ofFin
              change head :: tail = head :: (tail.map UInt8.toFin).map UInt8.ofFin
              exact congrArg (List.cons head) ih
    _ = source.map UInt8.ofFin := congrArg (List.map UInt8.ofFin) representative

end ByteRep

/-- Encoding preserves and reflects equality of source byte lists. -/
theorem encodeBytes_injective : Function.Injective encodeBytes := by
  intro left right h
  have views := congrArg
    (fun packed : ByteArray => packed.data.toList.map UInt8.toFin) h
  exact (ByteRep.encodeBytes left).symm.trans
    (views.trans (ByteRep.encodeBytes right))

end Slang.Source
