import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source.BytesAudit

open Slang.Source

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

#audit_empty Slang.Source.SourceByte : Type
#audit_empty Slang.Source.SourceBytes : Type
#audit_empty Slang.Source.encodeBytes : Slang.Source.SourceBytes → ByteArray
#audit_empty Slang.Source.ByteRep : Slang.Source.SourceBytes → ByteArray → Prop
#audit_empty Slang.Source.ByteRep.encodeBytes : ∀ (source : Slang.Source.SourceBytes), Slang.Source.ByteRep source (Slang.Source.encodeBytes source)
#audit_empty Slang.Source.ByteRep.eq : ∀ {left right : Slang.Source.SourceBytes} {packed : ByteArray}, Slang.Source.ByteRep left packed → Slang.Source.ByteRep right packed → left = right
#audit_empty Slang.Source.ByteRep.eq_of_encodeBytes : ∀ {left right : Slang.Source.SourceBytes}, Slang.Source.ByteRep left (Slang.Source.encodeBytes right) → left = right
#audit_empty Slang.Source.ByteRep.encodeBytes_eq : ∀ {source : Slang.Source.SourceBytes} {packed : ByteArray}, Slang.Source.ByteRep source packed → packed = Slang.Source.encodeBytes source
#audit_empty Slang.Source.ByteRep.iff_eq_encodeBytes : ∀ {source : Slang.Source.SourceBytes} {packed : ByteArray}, Slang.Source.ByteRep source packed ↔ packed = Slang.Source.encodeBytes source
#audit_empty Slang.Source.ByteRep.size_eq_length : ∀ {source : Slang.Source.SourceBytes} {packed : ByteArray}, Slang.Source.ByteRep source packed → packed.size = source.length
#audit_empty Slang.Source.encodeBytes_injective : Function.Injective Slang.Source.encodeBytes
#audit_empty Slang.Source.BytesAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Source.BytesAudit.sample : Slang.Source.SourceBytes
#audit_empty Slang.Source.BytesAudit.mutated : Slang.Source.SourceBytes
#audit_empty Slang.Source.BytesAudit.sample_rep : Slang.Source.ByteRep Slang.Source.BytesAudit.sample (Slang.Source.encodeBytes Slang.Source.BytesAudit.sample)
#audit_empty Slang.Source.BytesAudit.represented_encoding_eq : ∀ {bytes bytes' : Slang.Source.SourceBytes}, Slang.Source.ByteRep bytes (Slang.Source.encodeBytes bytes') → bytes = bytes'
#audit_empty Slang.Source.BytesAudit.mutated_rejected : ¬Slang.Source.ByteRep Slang.Source.BytesAudit.sample (Slang.Source.encodeBytes Slang.Source.BytesAudit.mutated)

end Slang.Source.BytesAudit
