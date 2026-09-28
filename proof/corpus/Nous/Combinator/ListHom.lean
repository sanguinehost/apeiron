import Mathlib
import Substrate.Algebra.Neural.MLPrimitivesBridge
import Numeric.ColumnCover

set_option autoImplicit false
set_option warningAsError true

open Sanguine.MLPrimitivesBridge

namespace Nous.Combinator.ListHom

/-!
# List Homomorphisms and Combinator Fold Links

Spec: `docs/epics/combinator-links.md` (task `APEIRON-CL-W1-T2`).
Namespace `Nous.Combinator.ListHom`.

Fold links for three atlas objects:
1. Generic list homomorphisms (`IsListHom`, `IsAddListHom`), flatten equivalence, and merge trees (`MTree`).
2. Instance (a): Rectangle / column cover (`colSum`) as an additive list homomorphism.
3. Instance (b): Quantisation lattice (`quantChunks`, `quant_flatten`, and aligned merge tree).
4. Instance (c): Gibbs measure (online-softmax log-sum-exp fusion `osFuse`, denominator and numerator trees).
-/

/-! ## 1. Generic List Homomorphisms and Merge Trees -/

/-- Multiplicative list homomorphism structure: `h` preserves empty list and concatenation. -/
structure IsListHom {α M} [Monoid M] (h : List α → M) : Prop where
  nil : h [] = 1
  append : ∀ l m, h (l ++ m) = h l * h m

/-- Additive list homomorphism structure: `h` preserves empty list and concatenation. -/
structure IsAddListHom {α M} [AddMonoid M] (h : List α → M) : Prop where
  nil : h [] = 0
  append : ∀ l m, h (l ++ m) = h l + h m

/-- Every multiplicative list homomorphism decomposes as the product of its singleton images. -/
theorem hom_eq_prod {α M} [Monoid M] (h : List α → M) (H : IsListHom h) (l : List α) :
    h l = (l.map fun a => h [a]).prod := by
  induction l with
  | nil =>
    rw [H.nil, List.map_nil, List.prod_nil]
  | cons a as ih =>
    have : a :: as = [a] ++ as := rfl
    rw [this, H.append, ih]
    rfl

/-- Every additive list homomorphism decomposes as the sum of its singleton images. -/
theorem add_hom_eq_sum {α M} [AddMonoid M] (h : List α → M) (H : IsAddListHom h) (l : List α) :
    h l = (l.map fun a => h [a]).sum := by
  induction l with
  | nil =>
    rw [H.nil, List.map_nil, List.sum_nil]
  | cons a as ih =>
    have : a :: as = [a] ++ as := rfl
    rw [this, H.append, ih]
    rfl

/-- Every consecutive partition evaluated under a list homomorphism gives the product of the parts. -/
theorem hom_flatten {α M} [Monoid M] (h : List α → M) (H : IsListHom h) (L : List (List α)) :
    h L.flatten = (L.map h).prod := by
  induction L with
  | nil =>
    rw [List.flatten_nil, H.nil, List.map_nil, List.prod_nil]
  | cons l Ls ih =>
    rw [List.flatten_cons, H.append, ih, List.map_cons, List.prod_cons]

/-- Additive version: summing over any consecutive partition gives the sum of the parts. -/
theorem add_hom_flatten {α M} [AddMonoid M] (h : List α → M) (H : IsAddListHom h) (L : List (List α)) :
    h L.flatten = (L.map h).sum := by
  induction L with
  | nil =>
    rw [List.flatten_nil, H.nil, List.map_nil, List.sum_nil]
  | cons l Ls ih =>
    rw [List.flatten_cons, H.append, ih, List.map_cons, List.sum_cons]

/-- Binary merge tree over lists of leaves. -/
inductive MTree (α : Type*) where
  | leaf : List α → MTree α
  | node : MTree α → MTree α → MTree α

/-- Flattening a merge tree to its underlying list of elements. -/
def flat {α : Type*} : MTree α → List α
  | MTree.leaf l => l
  | MTree.node t1 t2 => flat t1 ++ flat t2

/-- Evaluating a merge tree under a multiplicative homomorphism `h`. -/
def eval {α M} [Monoid M] (h : List α → M) : MTree α → M
  | MTree.leaf l => h l
  | MTree.node t1 t2 => eval h t1 * eval h t2

/-- Evaluating a merge tree under an additive homomorphism `h`. -/
def evalAdd {α M} [AddMonoid M] (h : List α → M) : MTree α → M
  | MTree.leaf l => h l
  | MTree.node t1 t2 => evalAdd h t1 + evalAdd h t2

/-- Any merge tree evaluated under `h` agrees with `h` on the flattened leaves. -/
theorem hom_tree {α M} [Monoid M] (h : List α → M) (H : IsListHom h) (t : MTree α) :
    eval h t = h (flat t) := by
  induction t with
  | leaf l => rfl
  | node t1 t2 ih1 ih2 =>
    dsimp [eval, flat]
    rw [H.append, ih1, ih2]

/-- Any merge tree evaluated under an additive `h` agrees with `h` on the flattened leaves. -/
theorem add_hom_tree {α M} [AddMonoid M] (h : List α → M) (H : IsAddListHom h) (t : MTree α) :
    evalAdd h t = h (flat t) := by
  induction t with
  | leaf l => rfl
  | node t1 t2 ih1 ih2 =>
    dsimp [evalAdd, flat]
    rw [H.append, ih1, ih2]


/-! ## 2. Instance (a): Rectangle / Column Cover -/

/-- Sum of column weights over a list of column features / entries. -/
def colSum {α : Type*} (f : α → ℤ) (l : List α) : ℤ :=
  (l.map f).sum

/-- `colSum f` is an additive list homomorphism. -/
theorem colSum_isAddListHom {α : Type*} (f : α → ℤ) :
    IsAddListHom (colSum f) where
  nil := by simp [colSum]
  append := by
    intro l m
    simp [colSum, List.map_append, List.sum_append]

/-- Summing any partition of the columns gives the whole sum. -/
theorem colSum_flatten {α : Type*} (f : α → ℤ) (L : List (List α)) :
    colSum f L.flatten = (L.map (colSum f)).sum :=
  add_hom_flatten (colSum f) (colSum_isAddListHom f) L

/-- Summing columns via any merge tree equals the sum on the flattened list. -/
theorem colSum_tree {α : Type*} (f : α → ℤ) (t : MTree α) :
    evalAdd (colSum f) t = colSum f (flat t) :=
  add_hom_tree (colSum f) (colSum_isAddListHom f) t

/-!
### Bridge to `Numeric.ColumnCover`

`Numeric.ColumnCover.column_cover_round` is formulated over indexed families `Fin n → ℤ` partitioned
via fibers of an index map `part : Fin n → Fin k`, with a rounding operator `r : ℤ → ℤ`:
```
theorem column_cover_round {n k : ℕ} (part : Fin n → Fin k) (acc : Fin n → ℤ) (r : ℤ → ℤ) :
    r (∑ b : Fin n, acc b)
      = r (∑ i : Fin k, ∑ b ∈ (Finset.univ.filter (fun b : Fin n => part b = i)), acc b)
```
In contrast, `colSum_flatten` is the free list-homomorphism counterpart:
`colSum f L.flatten = (L.map (colSum f)).sum`.
When `r = id`, `column_cover_round` expresses commutativity and associativity of summation across
fiber sets (unordered/indexed blocks), while `colSum_flatten` expresses the same property for
ordered consecutive partitions in a list. When consecutive slices are represented as a list of lists `L`,
`colSum` on each slice sums to the total `colSum` on `L.flatten`.
-/


/-! ## 3. Instance (b): Quantisation Lattice -/

/-- Helper lemma: the concatenation of 32-aligned lists has length a multiple of 32. -/
theorem length_flatten_mod32 (L : List (List ℤ)) (hL : ∀ l ∈ L, l.length % 32 = 0) :
    L.flatten.length % 32 = 0 := by
  induction L with
  | nil => rfl
  | cons l Ls ih =>
    have hl : l.length % 32 = 0 := hL l (List.Mem.head Ls)
    have hLs : ∀ m ∈ Ls, m.length % 32 = 0 := fun m hm => hL m (List.Mem.tail l hm)
    rw [List.flatten_cons, List.length_append]
    have ihLs := ih hLs
    obtain ⟨nl, hnl⟩ : ∃ nl, l.length = 32 * nl := ⟨l.length / 32, by omega⟩
    obtain ⟨ns, hns⟩ : ∃ ns, Ls.flatten.length = 32 * ns := ⟨Ls.flatten.length / 32, by omega⟩
    rw [hnl, hns]
    omega

/-- Quantizing any list of 32-aligned chunks equals flattening the separately quantized chunks. -/
theorem quant_flatten (L : List (List ℤ)) (hL : ∀ l ∈ L, l.length % 32 = 0) :
    ColumnCover.quantChunks L.flatten = (L.map ColumnCover.quantChunks).flatten := by
  induction L with
  | nil => rfl
  | cons l Ls ih =>
    have hl : l.length % 32 = 0 := hL l (List.Mem.head Ls)
    have hLs : ∀ m ∈ Ls, m.length % 32 = 0 := fun m hm => hL m (List.Mem.tail l hm)
    rw [List.flatten_cons, List.map_cons, List.flatten_cons]
    have hflat_len : Ls.flatten.length % 32 = 0 := length_flatten_mod32 Ls hLs
    rw [ColumnCover.quant_locality l Ls.flatten hl hflat_len]
    rw [ih hLs]

/-- Predicate that all leaves of a merge tree have length divisible by 32. -/
def MTree.AllLeavesAligned (t : MTree ℤ) : Prop :=
  match t with
  | MTree.leaf l => l.length % 32 = 0
  | MTree.node t1 t2 => MTree.AllLeavesAligned t1 ∧ MTree.AllLeavesAligned t2

/-- If all leaves in a merge tree are 32-aligned, the flattened tree is 32-aligned. -/
theorem flat_length_mod32 (t : MTree ℤ) (ht : t.AllLeavesAligned) :
    (flat t).length % 32 = 0 := by
  induction t with
  | leaf l => exact ht
  | node t1 t2 ih1 ih2 =>
    dsimp [flat]
    rw [List.length_append]
    have h1 := ih1 ht.1
    have h2 := ih2 ht.2
    obtain ⟨n1, hn1⟩ : ∃ n1, (flat t1).length = 32 * n1 := ⟨(flat t1).length / 32, by omega⟩
    obtain ⟨n2, hn2⟩ : ∃ n2, (flat t2).length = 32 * n2 := ⟨(flat t2).length / 32, by omega⟩
    rw [hn1, hn2]
    omega

/-- Evaluation of an aligned quantisation tree by blockwise quantization on leaves and concatenation. -/
def evalQuantTree : MTree ℤ → List ℤ
  | MTree.leaf l => ColumnCover.quantChunks l
  | MTree.node t1 t2 => evalQuantTree t1 ++ evalQuantTree t2

/-- Any merge tree with 32-aligned leaves computes the same quantization as the flat list. -/
theorem quant_tree (t : MTree ℤ) (ht : t.AllLeavesAligned) :
    evalQuantTree t = ColumnCover.quantChunks (flat t) := by
  induction t with
  | leaf l => rfl
  | node t1 t2 ih1 ih2 =>
    dsimp [evalQuantTree, flat]
    rw [ih1 ht.1, ih2 ht.2]
    have h1 := flat_length_mod32 t1 ht.1
    have h2 := flat_length_mod32 t2 ht.2
    exact (ColumnCover.quant_locality (flat t1) (flat t2) h1 h2).symm


/-! ## 4. Instance (c): Gibbs Measure (Online-Softmax) -/

/-- Leaf of the online-softmax denominator: `(s, 1)` denoting `1 · e^s = e^s`. -/
noncomputable def osLeaf (s : ℝ) : ℝ × ℝ := (s, 1)

/-- Safe fusion of two optional online-softmax states (treating `none` as neutral). -/
noncomputable def osFuseOpt : Option (ℝ × ℝ) → Option (ℝ × ℝ) → Option (ℝ × ℝ)
  | none, y => y
  | x, none => x
  | some x, some y => some (osFuse x y)

/-- Fold `osFuse` over a list of scores, returning `none` on the empty list. -/
noncomputable def osFold : List ℝ → Option (ℝ × ℝ)
  | [] => none
  | s :: ss => osFuseOpt (some (osLeaf s)) (osFold ss)

/-- Interpretation of an optional online-softmax state, mapping `none` to 0. -/
noncomputable def optVal : Option (ℝ × ℝ) → ℝ
  | none => 0
  | some a => osVal a

theorem osVal_osLeaf (s : ℝ) : osVal (osLeaf s) = Real.exp s := by
  dsimp [osVal, osLeaf]
  ring

theorem optVal_fuse (x y : Option (ℝ × ℝ)) :
    optVal (osFuseOpt x y) = optVal x + optVal y := by
  cases x <;> cases y <;> simp [osFuseOpt, optVal, osFuse_is_add]

theorem osFold_append (l m : List ℝ) :
    optVal (osFold (l ++ m)) = optVal (osFold l) + optVal (osFold m) := by
  induction l with
  | nil => simp [osFold, optVal]
  | cons s ss ih =>
    dsimp [osFold]
    rw [optVal_fuse, optVal_fuse, ih]
    ring

/-- **Online-softmax denominator.** The decoded value of `osFold l` is `∑ e^s`. -/
theorem os_denominator (l : List ℝ) : optVal (osFold l) = (l.map Real.exp).sum := by
  induction l with
  | nil => rfl
  | cons s ss ih =>
    dsimp [osFold]
    rw [optVal_fuse]
    have : optVal (some (osLeaf s)) = Real.exp s := osVal_osLeaf s
    rw [this, ih]

/-- Tree evaluation for the online-softmax denominator. -/
noncomputable def evalOsTree : MTree ℝ → Option (ℝ × ℝ)
  | MTree.leaf l => osFold l
  | MTree.node t1 t2 => osFuseOpt (evalOsTree t1) (evalOsTree t2)

/-- Any merge tree over any split of positions computes the same online-softmax denominator. -/
theorem os_tree (t : MTree ℝ) :
    optVal (evalOsTree t) = optVal (osFold (flat t)) := by
  induction t with
  | leaf l => rfl
  | node t1 t2 ih1 ih2 =>
    dsimp [evalOsTree, flat]
    rw [optVal_fuse, ih1, ih2, osFold_append]

/-- Triple state `(m, s, n)` denoting `(s · e^m, n · e^m)` for denominator and weighted numerator. -/
noncomputable def osFuseNum (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let m1 := a.1
  let s1 := a.2.1
  let n1 := a.2.2
  let m2 := b.1
  let s2 := b.2.1
  let n2 := b.2.2
  let M := max m1 m2
  (M,
   s1 * Real.exp (m1 - M) + s2 * Real.exp (m2 - M),
   n1 * Real.exp (m1 - M) + n2 * Real.exp (m2 - M))

/-- Interpretation of the triple state: `(s · e^m, n · e^m)`. -/
noncomputable def osValNum (a : ℝ × ℝ × ℝ) : ℝ × ℝ :=
  (a.2.1 * Real.exp a.1, a.2.2 * Real.exp a.1)

/-- Leaf for a score-value pair `(s, v)`: `(s, 1, v)` denoting `(e^s, v · e^s)`. -/
noncomputable def osLeafNum (sv : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (sv.1, 1, sv.2)

theorem osFuseNum_fst (a b : ℝ × ℝ × ℝ) :
    (osValNum (osFuseNum a b)).1 = (osValNum a).1 + (osValNum b).1 := by
  dsimp [osValNum, osFuseNum]
  simp only [add_mul, mul_assoc, ← Real.exp_add]
  ring_nf

theorem osFuseNum_snd (a b : ℝ × ℝ × ℝ) :
    (osValNum (osFuseNum a b)).2 = (osValNum a).2 + (osValNum b).2 := by
  dsimp [osValNum, osFuseNum]
  simp only [add_mul, mul_assoc, ← Real.exp_add]
  ring_nf

/-- The numerator merge denotes exact vector addition of `(s · e^m, n · e^m)`. -/
theorem osFuseNum_is_add (a b : ℝ × ℝ × ℝ) :
    osValNum (osFuseNum a b) = ((osValNum a).1 + (osValNum b).1, (osValNum a).2 + (osValNum b).2) := by
  ext
  · exact osFuseNum_fst a b
  · exact osFuseNum_snd a b

/-- Safe fusion of two optional numerator states. -/
noncomputable def osFuseNumOpt : Option (ℝ × ℝ × ℝ) → Option (ℝ × ℝ × ℝ) → Option (ℝ × ℝ × ℝ)
  | none, y => y
  | x, none => x
  | some x, some y => some (osFuseNum x y)

/-- Fold `osFuseNum` over a list of score-value pairs. -/
noncomputable def osFoldNum : List (ℝ × ℝ) → Option (ℝ × ℝ × ℝ)
  | [] => none
  | sv :: rest => osFuseNumOpt (some (osLeafNum sv)) (osFoldNum rest)

/-- Interpretation of an optional numerator state, mapping `none` to `(0, 0)`. -/
noncomputable def optValNum : Option (ℝ × ℝ × ℝ) → ℝ × ℝ
  | none => (0, 0)
  | some a => osValNum a

theorem optValNum_fuse (x y : Option (ℝ × ℝ × ℝ)) :
    optValNum (osFuseNumOpt x y) = ((optValNum x).1 + (optValNum y).1, (optValNum x).2 + (optValNum y).2) := by
  cases x <;> cases y <;> simp [osFuseNumOpt, optValNum, osFuseNum_is_add]

theorem osValNum_osLeafNum (sv : ℝ × ℝ) :
    osValNum (osLeafNum sv) = (Real.exp sv.1, sv.2 * Real.exp sv.1) := by
  dsimp [osValNum, osLeafNum]
  ring_nf

theorem osFoldNum_append (l m : List (ℝ × ℝ)) :
    optValNum (osFoldNum (l ++ m)) =
      ((optValNum (osFoldNum l)).1 + (optValNum (osFoldNum m)).1,
       (optValNum (osFoldNum l)).2 + (optValNum (osFoldNum m)).2) := by
  induction l with
  | nil => simp [osFoldNum, optValNum]
  | cons sv rest ih =>
    dsimp [osFoldNum]
    rw [optValNum_fuse, optValNum_fuse, ih]
    ring_nf

/-- **Online-softmax numerator.** The decoded numerator of `osFoldNum l` is `∑ v_i · e^{s_i}`. -/
theorem os_numerator (l : List (ℝ × ℝ)) :
    (optValNum (osFoldNum l)).2 = (l.map fun sv => sv.2 * Real.exp sv.1).sum := by
  induction l with
  | nil => rfl
  | cons sv rest ih =>
    dsimp [osFoldNum]
    rw [optValNum_fuse]
    have hleaf : (optValNum (some (osLeafNum sv))).2 = sv.2 * Real.exp sv.1 := by
      dsimp [optValNum]
      rw [osValNum_osLeafNum]
    rw [hleaf, ih]

/-- Tree evaluation for the online-softmax numerator. -/
noncomputable def evalOsNumTree : MTree (ℝ × ℝ) → Option (ℝ × ℝ × ℝ)
  | MTree.leaf l => osFoldNum l
  | MTree.node t1 t2 => osFuseNumOpt (evalOsNumTree t1) (evalOsNumTree t2)

/-- Any merge tree over any split of positions computes the same online-softmax numerator. -/
theorem os_numerator_tree (t : MTree (ℝ × ℝ)) :
    (optValNum (evalOsNumTree t)).2 = (optValNum (osFoldNum (flat t))).2 := by
  induction t with
  | leaf l => rfl
  | node t1 t2 ih1 ih2 =>
    dsimp [evalOsNumTree, flat]
    rw [optValNum_fuse, osFoldNum_append]
    have ih1' : (optValNum (evalOsNumTree t1)).2 = (optValNum (osFoldNum (flat t1))).2 := ih1
    have ih2' : (optValNum (evalOsNumTree t2)).2 = (optValNum (osFoldNum (flat t2))).2 := ih2
    dsimp
    rw [ih1', ih2']


/-! ## 5. Witnesses -/

namespace Witness

-- (a) Small ℤ list witness for column sum
def l1 : List ℤ := [1, 2, 3]
def l2 : List ℤ := [4, 5]

#guard colSum id (l1 ++ l2) == colSum id l1 + colSum id l2
#guard colSum id (l1 ++ l2) == 15

-- (b) Two 32-long aligned lists of small literals
def q1 : List ℤ := List.replicate 32 1
def q2 : List ℤ := List.replicate 32 2

#guard (ColumnCover.quantChunks (q1 ++ q2)).length == 64
#guard ColumnCover.quantChunks (q1 ++ q2) == ColumnCover.quantChunks q1 ++ ColumnCover.quantChunks q2

-- (c) Example instance for online softmax
noncomputable example : ℝ :=
  let scores : List ℝ := [0, 1]
  optVal (osFold scores)

example (s v : ℝ) :
    osValNum (osLeafNum (s, v)) = (Real.exp s, v * Real.exp s) :=
  osValNum_osLeafNum (s, v)

end Witness

end Nous.Combinator.ListHom
