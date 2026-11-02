import ApeironWord
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# Aeneas index reads on the extracted slices and vectors

`Slice.index_usize` and `Vec.index` at an in-range index read the backing list. The dense backward
refinements (`Learn/Grad/DenseNorm.lean`) rewrite every element read through these two lemmas. They
were stated in the removed `Learn/GramExact.lean` and move here unchanged.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_index

open apeiron_word
open Aeneas Aeneas.Std Result

/-- `Slice.index_usize` reads the backing list at the in-range index. -/
theorem slice_index_getD {α : Type} [Inhabited α] (x : Slice α) (k : Usize)
    (h : k.val < x.val.length) :
    Slice.index_usize x k = ok (x.val.getD k.val default) := by
  unfold Slice.index_usize
  rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem h, List.getD_eq_getElem _ _ h]

/-- `Vec.index` reads the backing list at the in-range index. -/
theorem vec_index_getD (g : alloc.vec.Vec I64) (k : Usize) (h : k.val < g.val.length) :
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice I64) g k
      = ok (g.val.getD k.val 0#i64) := by
  rw [alloc.vec.Vec.index_slice_index]
  unfold alloc.vec.Vec.index_usize
  rw [alloc.vec.Vec.getElem?_Nat_eq, List.getElem?_eq_getElem h, List.getD_eq_getElem _ _ h]

end apeiron_index
