import Iota.Refinement.WordDenotation
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Refinement.WordDenotationAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Refinement.WordDenotationAudit

#audit_empty Iota.Refinement.WordDenotationAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Refinement.wordDenote : Iota.Word → Iota.Word
#audit_empty Iota.Refinement.wordDenote_eq_val : ∀ (word : Iota.Word), (Iota.Refinement.wordDenote word).val = word.val
#audit_empty Iota.Refinement.wordDenote_eq : ∀ (word : Iota.Word), Iota.Refinement.wordDenote word = word
#audit_empty Iota.Refinement.wordDenote_ofNat : ∀ (value : Nat), (Iota.Refinement.wordDenote (Iota.Word.ofNat value)).val = value % Iota.wordModulus
#audit_empty Iota.Refinement.wordDenote_add_rep : ∀ (left right : Iota.Word), (Iota.Refinement.wordDenote (Iota.Word.add left right)).val = (left.val + right.val) % Iota.wordModulus
#audit_empty Iota.Refinement.wordDenote_sub_rep : ∀ (left right : Iota.Word), (Iota.Refinement.wordDenote (Iota.Word.sub left right)).val = (left.val + Iota.wordModulus - right.val) % Iota.wordModulus
#audit_empty Iota.Refinement.wordDenote_mul_rep : ∀ (left right : Iota.Word), (Iota.Refinement.wordDenote (Iota.Word.mul left right)).val = (left.val * right.val) % Iota.wordModulus
#audit_empty Iota.Refinement.wordDenote_add : ∀ (left right : Iota.Word), Iota.Refinement.wordDenote (Iota.Word.add left right) = Iota.Refinement.wordRing.add (Iota.Refinement.wordDenote left) (Iota.Refinement.wordDenote right)
#audit_empty Iota.Refinement.wordDenote_sub : ∀ (left right : Iota.Word), Iota.Refinement.wordDenote (Iota.Word.sub left right) = Iota.Refinement.wordRing.sub (Iota.Refinement.wordDenote left) (Iota.Refinement.wordDenote right)
#audit_empty Iota.Refinement.wordDenote_mul : ∀ (left right : Iota.Word), Iota.Refinement.wordDenote (Iota.Word.mul left right) = Iota.Refinement.wordRing.mul (Iota.Refinement.wordDenote left) (Iota.Refinement.wordDenote right)
#audit_empty Iota.Refinement.wordDenote_bitAnd : ∀ (left right : Iota.Word), Iota.Refinement.wordDenote (Iota.Word.bitAnd left right) = Iota.Refinement.wordRing.ofNat (Iota.Word.bitwise Bool.and Iota.wordBits left.val right.val)
#audit_empty Iota.Refinement.wordDenote_tableSpread : ∀ (key : Iota.Word), Iota.Refinement.wordDenote (Iota.Word.tableSpread key) = Iota.Refinement.wordRing.ofNat (Iota.Word.bitwise Bool.and Iota.wordBits (Iota.Word.tableSpreadProduct key).val Iota.Word.spreadMask.val)
#audit_empty Iota.Refinement.cellDenote : Iota.Word × Iota.Word → Apeiron.Trichotomy.AnyRing.RingCell Iota.Word
#audit_empty Iota.Refinement.cellMul : Iota.Word → Iota.Word × Iota.Word → Iota.Word × Iota.Word → Iota.Word × Iota.Word
#audit_empty Iota.Refinement.cellNormForm : Iota.Word → Iota.Word × Iota.Word → Iota.Word
#audit_empty Iota.Refinement.cellDenote_mul : ∀ (sigma : Iota.Word) (left right : Iota.Word × Iota.Word), Iota.Refinement.cellDenote (Iota.Refinement.cellMul sigma left right) = Apeiron.Trichotomy.AnyRing.ringMul (Iota.Refinement.wordDenote sigma) (Iota.Refinement.cellDenote left) (Iota.Refinement.cellDenote right)
#audit_empty Iota.Refinement.cellNormForm_denote : ∀ (sigma : Iota.Word) (cell : Iota.Word × Iota.Word), Iota.Refinement.wordDenote (Iota.Refinement.cellNormForm sigma cell) = Apeiron.Trichotomy.AnyRing.ringNormForm (Iota.Refinement.wordDenote sigma) (Iota.Refinement.cellDenote cell)
#audit_empty Iota.Refinement.wordMul_comm_denote : ∀ (left right : Iota.Word), Iota.Refinement.wordDenote (Iota.Word.mul left right) = Iota.Refinement.wordDenote (Iota.Word.mul right left)
#audit_empty Iota.Refinement.wordMul_assoc_denote : ∀ (left middle right : Iota.Word), Iota.Refinement.wordDenote (Iota.Word.mul (Iota.Word.mul left middle) right) = Iota.Refinement.wordDenote (Iota.Word.mul left (Iota.Word.mul middle right))
#audit_empty Iota.Refinement.cellMul_comm_denote : ∀ (sigma : Iota.Word) (left right : Iota.Word × Iota.Word), Iota.Refinement.cellDenote (Iota.Refinement.cellMul sigma left right) = Iota.Refinement.cellDenote (Iota.Refinement.cellMul sigma right left)
#audit_empty Iota.Refinement.cellMul_assoc_denote : ∀ (sigma : Iota.Word) (left middle right : Iota.Word × Iota.Word), Iota.Refinement.cellDenote (Iota.Refinement.cellMul sigma (Iota.Refinement.cellMul sigma left middle) right) = Iota.Refinement.cellDenote (Iota.Refinement.cellMul sigma left (Iota.Refinement.cellMul sigma middle right))
#audit_empty Iota.Refinement.cellNormForm_mul_denote : ∀ (sigma : Iota.Word) (left right : Iota.Word × Iota.Word), Iota.Refinement.wordDenote (Iota.Refinement.cellNormForm sigma (Iota.Refinement.cellMul sigma left right)) = Iota.Refinement.wordRing.mul (Iota.Refinement.wordDenote (Iota.Refinement.cellNormForm sigma left)) (Iota.Refinement.wordDenote (Iota.Refinement.cellNormForm sigma right))
