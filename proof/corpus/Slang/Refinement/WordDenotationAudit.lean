import Slang.Refinement.WordDenotation
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Refinement.WordDenotationAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Refinement.WordDenotationAudit

#audit_empty Slang.Refinement.WordDenotationAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Refinement.wordDenote : Slang.Word → Slang.Word
#audit_empty Slang.Refinement.wordDenote_eq_val : ∀ (word : Slang.Word), (Slang.Refinement.wordDenote word).val = word.val
#audit_empty Slang.Refinement.wordDenote_eq : ∀ (word : Slang.Word), Slang.Refinement.wordDenote word = word
#audit_empty Slang.Refinement.wordDenote_ofNat : ∀ (value : Nat), (Slang.Refinement.wordDenote (Slang.Word.ofNat value)).val = value % Slang.wordModulus
#audit_empty Slang.Refinement.wordDenote_add_rep : ∀ (left right : Slang.Word), (Slang.Refinement.wordDenote (Slang.Word.add left right)).val = (left.val + right.val) % Slang.wordModulus
#audit_empty Slang.Refinement.wordDenote_sub_rep : ∀ (left right : Slang.Word), (Slang.Refinement.wordDenote (Slang.Word.sub left right)).val = (left.val + Slang.wordModulus - right.val) % Slang.wordModulus
#audit_empty Slang.Refinement.wordDenote_mul_rep : ∀ (left right : Slang.Word), (Slang.Refinement.wordDenote (Slang.Word.mul left right)).val = (left.val * right.val) % Slang.wordModulus
#audit_empty Slang.Refinement.wordDenote_add : ∀ (left right : Slang.Word), Slang.Refinement.wordDenote (Slang.Word.add left right) = Slang.Refinement.wordRing.add (Slang.Refinement.wordDenote left) (Slang.Refinement.wordDenote right)
#audit_empty Slang.Refinement.wordDenote_sub : ∀ (left right : Slang.Word), Slang.Refinement.wordDenote (Slang.Word.sub left right) = Slang.Refinement.wordRing.sub (Slang.Refinement.wordDenote left) (Slang.Refinement.wordDenote right)
#audit_empty Slang.Refinement.wordDenote_mul : ∀ (left right : Slang.Word), Slang.Refinement.wordDenote (Slang.Word.mul left right) = Slang.Refinement.wordRing.mul (Slang.Refinement.wordDenote left) (Slang.Refinement.wordDenote right)
#audit_empty Slang.Refinement.wordDenote_bitAnd : ∀ (left right : Slang.Word), Slang.Refinement.wordDenote (Slang.Word.bitAnd left right) = Slang.Refinement.wordRing.ofNat (Slang.Word.bitwise Bool.and Slang.wordBits left.val right.val)
#audit_empty Slang.Refinement.wordDenote_tableSpread : ∀ (key : Slang.Word), Slang.Refinement.wordDenote (Slang.Word.tableSpread key) = Slang.Refinement.wordRing.ofNat (Slang.Word.bitwise Bool.and Slang.wordBits (Slang.Word.tableSpreadProduct key).val Slang.Word.spreadMask.val)
#audit_empty Slang.Refinement.cellDenote : Slang.Word × Slang.Word → Apeiron.Trichotomy.AnyRing.RingCell Slang.Word
#audit_empty Slang.Refinement.cellMul : Slang.Word → Slang.Word × Slang.Word → Slang.Word × Slang.Word → Slang.Word × Slang.Word
#audit_empty Slang.Refinement.cellNormForm : Slang.Word → Slang.Word × Slang.Word → Slang.Word
#audit_empty Slang.Refinement.cellDenote_mul : ∀ (sigma : Slang.Word) (left right : Slang.Word × Slang.Word), Slang.Refinement.cellDenote (Slang.Refinement.cellMul sigma left right) = Apeiron.Trichotomy.AnyRing.ringMul (Slang.Refinement.wordDenote sigma) (Slang.Refinement.cellDenote left) (Slang.Refinement.cellDenote right)
#audit_empty Slang.Refinement.cellNormForm_denote : ∀ (sigma : Slang.Word) (cell : Slang.Word × Slang.Word), Slang.Refinement.wordDenote (Slang.Refinement.cellNormForm sigma cell) = Apeiron.Trichotomy.AnyRing.ringNormForm (Slang.Refinement.wordDenote sigma) (Slang.Refinement.cellDenote cell)
#audit_empty Slang.Refinement.wordMul_comm_denote : ∀ (left right : Slang.Word), Slang.Refinement.wordDenote (Slang.Word.mul left right) = Slang.Refinement.wordDenote (Slang.Word.mul right left)
#audit_empty Slang.Refinement.wordMul_assoc_denote : ∀ (left middle right : Slang.Word), Slang.Refinement.wordDenote (Slang.Word.mul (Slang.Word.mul left middle) right) = Slang.Refinement.wordDenote (Slang.Word.mul left (Slang.Word.mul middle right))
#audit_empty Slang.Refinement.cellMul_comm_denote : ∀ (sigma : Slang.Word) (left right : Slang.Word × Slang.Word), Slang.Refinement.cellDenote (Slang.Refinement.cellMul sigma left right) = Slang.Refinement.cellDenote (Slang.Refinement.cellMul sigma right left)
#audit_empty Slang.Refinement.cellMul_assoc_denote : ∀ (sigma : Slang.Word) (left middle right : Slang.Word × Slang.Word), Slang.Refinement.cellDenote (Slang.Refinement.cellMul sigma (Slang.Refinement.cellMul sigma left middle) right) = Slang.Refinement.cellDenote (Slang.Refinement.cellMul sigma left (Slang.Refinement.cellMul sigma middle right))
#audit_empty Slang.Refinement.cellNormForm_mul_denote : ∀ (sigma : Slang.Word) (left right : Slang.Word × Slang.Word), Slang.Refinement.wordDenote (Slang.Refinement.cellNormForm sigma (Slang.Refinement.cellMul sigma left right)) = Slang.Refinement.wordRing.mul (Slang.Refinement.wordDenote (Slang.Refinement.cellNormForm sigma left)) (Slang.Refinement.wordDenote (Slang.Refinement.cellNormForm sigma right))
