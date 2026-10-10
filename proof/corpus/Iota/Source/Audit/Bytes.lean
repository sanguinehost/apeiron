import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Source.BytesAudit

open Iota.Source

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

private def sourceByte (value : Nat) : SourceByte :=
  ⟨value % 256, Nat.mod_lt value (Nat.zero_lt_succ 255)⟩

private def secondValue (bytes : SourceBytes) : Nat :=
  Option.elim bytes.tail.head? 0 Fin.val

private theorem nat_ne_succ (value : Nat) : value ≠ Nat.succ value := by
  induction value with
  | zero =>
      intro h
      exact Nat.noConfusion h
  | succ value ih =>
      intro h
      exact ih (Nat.noConfusion h (fun equal => equal))

/-- A small literal source-byte fixture. -/
def sample : SourceBytes := [sourceByte 0, sourceByte 127, sourceByte 255]

/-- The rejecting-control mutation changes the middle byte. -/
def mutated : SourceBytes := [sourceByte 0, sourceByte 128, sourceByte 255]

/-- The fixture is represented by its own encoding. -/
theorem sample_rep : ByteRep sample (encodeBytes sample) :=
  ByteRep.encodeBytes sample

/-- Any representation of an encoding fixes its source list. -/
theorem represented_encoding_eq {bytes bytes' : SourceBytes}
    (h : ByteRep bytes (encodeBytes bytes')) : bytes = bytes' :=
  ByteRep.eq_of_encodeBytes h

/-- The changed fixture cannot represent the original fixture. -/
theorem mutated_rejected : ¬ByteRep sample (encodeBytes mutated) := by
  intro h
  have heq : sample = mutated := represented_encoding_eq h
  have values := congrArg secondValue heq
  change 127 = Nat.succ 127 at values
  exact nat_ne_succ 127 values

#audit_empty Iota.Source.SourceByte : Type
#audit_empty Iota.Source.SourceBytes : Type
#audit_empty Iota.Source.encodeBytes : Iota.Source.SourceBytes → ByteArray
#audit_empty Iota.Source.ByteRep : Iota.Source.SourceBytes → ByteArray → Prop
#audit_empty Iota.Source.ByteRep.encodeBytes : ∀ (source : Iota.Source.SourceBytes), Iota.Source.ByteRep source (Iota.Source.encodeBytes source)
#audit_empty Iota.Source.ByteRep.eq : ∀ {left right : Iota.Source.SourceBytes} {packed : ByteArray}, Iota.Source.ByteRep left packed → Iota.Source.ByteRep right packed → left = right
#audit_empty Iota.Source.ByteRep.eq_of_encodeBytes : ∀ {left right : Iota.Source.SourceBytes}, Iota.Source.ByteRep left (Iota.Source.encodeBytes right) → left = right
#audit_empty Iota.Source.ByteRep.encodeBytes_eq : ∀ {source : Iota.Source.SourceBytes} {packed : ByteArray}, Iota.Source.ByteRep source packed → packed = Iota.Source.encodeBytes source
#audit_empty Iota.Source.ByteRep.iff_eq_encodeBytes : ∀ {source : Iota.Source.SourceBytes} {packed : ByteArray}, Iota.Source.ByteRep source packed ↔ packed = Iota.Source.encodeBytes source
#audit_empty Iota.Source.ByteRep.size_eq_length : ∀ {source : Iota.Source.SourceBytes} {packed : ByteArray}, Iota.Source.ByteRep source packed → packed.size = source.length
#audit_empty Iota.Source.encodeBytes_injective : Function.Injective Iota.Source.encodeBytes
#audit_empty Iota.Source.BytesAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Source.BytesAudit.sample : Iota.Source.SourceBytes
#audit_empty Iota.Source.BytesAudit.mutated : Iota.Source.SourceBytes
#audit_empty Iota.Source.BytesAudit.sample_rep : Iota.Source.ByteRep Iota.Source.BytesAudit.sample (Iota.Source.encodeBytes Iota.Source.BytesAudit.sample)
#audit_empty Iota.Source.BytesAudit.represented_encoding_eq : ∀ {bytes bytes' : Iota.Source.SourceBytes}, Iota.Source.ByteRep bytes (Iota.Source.encodeBytes bytes') → bytes = bytes'
#audit_empty Iota.Source.BytesAudit.mutated_rejected : ¬Iota.Source.ByteRep Iota.Source.BytesAudit.sample (Iota.Source.encodeBytes Iota.Source.BytesAudit.mutated)

end Iota.Source.BytesAudit
