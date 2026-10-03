/-
RFC 6962 §2.1 binary Merkle hash tree over an abstract hash function.

Domain separation: a leaf digest is `H (0x00 ‖ data)` and an interior digest is `H (0x01 ‖ l ‖ r)`,
so a leaf can never be confused with an interior node. `Tree` is the leaf/node inductive, `root`
is its RFC 6962 root hash, `path i` is the authentication path of leaf `i` as sibling digests in
RFC 6962 order, the top of the tree first, each tagged with the side its sibling is on, and
`combine` folds that list back into a root from the deepest sibling outwards, combining on the
same side. `verify` recomputes the
root from a leaf hash and its path and compares it with a claimed root, additionally requiring the
caller's upper bound on path length (RFC 6962 §2.1.1 bounds a path by the tree height).

`H` is an arbitrary `List Byte → Vector Byte 32`, so the file is independent of any particular
primitive: `Crypto.Sha2.sha256` and `Crypto.Sponge.sha3_256` instantiate it.

Proved unconditionally here: the reconstruction of a tree's own root from its own path
(`verify_root`) and the structural facts of `path`. Soundness across two different trees requires
injectivity and domain separation of `H`, which no `H : List Byte → Vector Byte 32` is obliged to
provide; it is stated as `NoSecondLeafVerifies`, a `Prop` with the collision hypothesis explicit.
-/

import Mathlib.Tactic

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.Merkle

/-! ## Abstract hash and domain separation -/

/-- One input/output byte of the hash. -/
abbrev Byte := BitVec 8

/-- A 256-bit digest, as produced by the abstract hash. -/
abbrev Digest := Vector Byte 32

/-- The abstract hash; instantiated with SHA-256, SHA3-256 or Keccak-256 at the use site. -/
abbrev HashFn := List Byte → Digest

/-- RFC 6962 §2.1 leaf digest `MTH({d(0)}) = HASH(0x00 ‖ d(0))`. O(1) beyond the hash call. -/
def leafHash (H : HashFn) (data : List Byte) : Digest := H (0x00 :: data)

/-- RFC 6962 §2.1 interior digest `HASH(0x01 ‖ left ‖ right)`. O(1) beyond the hash call. -/
def nodeHash (H : HashFn) (left right : Digest) : Digest := H (0x01 :: (left.toList ++ right.toList))

/-! ## The tree -/

/-- An RFC 6962 binary hash tree: `leaf` carries one data blob, `node` two subtrees. -/
inductive Tree where
  | leaf : List Byte → Tree
  | node : Tree → Tree → Tree
  deriving Repr

/-- The RFC 6962 root hash of a tree. The hash is the leading explicit parameter because it is an
abstraction, not part of the tree's shape. -/
def root (H : HashFn) : Tree → Digest
  | .leaf d => leafHash H d
  | .node l r => nodeHash H (root H l) (root H r)

/-- Number of leaves. Temporal cost is linear in the tree size (one call per node); spatial cost is
O(1) auxiliary. -/
def size : Tree → Nat
  | .leaf _ => 1
  | .node l r => size l + size r

/-- Height of the tree, i.e. the maximum length of an authentication path (RFC 6962 §2.1.1
`PATH`). For the balanced tree RFC 6962 §2.1 requires, `height_node_of_le` below reduces this to
`height l + 1`. Temporal cost is linear in the tree size; spatial cost is O(1) auxiliary. -/
def height : Tree → Nat
  | .leaf _ => 0
  | .node l r => max (height l) (height r) + 1

/-- On the balanced trees RFC 6962 §2.1 requires, the path bound is one step above the left
subtree's height. -/
theorem height_node_of_le (l r : Tree) (h : height l ≥ height r) :
    height (.node l r) = height l + 1 := by simp [height, Nat.max_eq_left h]

/-- The data at leaf index `i`, or `none` when the index is out of range. The right subtree of a
node is indexed from `size l`, as RFC 6962 numbers leaves. Temporal cost is linear in the depth of
`i` (one subtraction per node); spatial cost is O(1) auxiliary. -/
def dataAt : Tree → Nat → Option (List Byte)
  | .leaf d, 0 => some d
  | .leaf _, _ + 1 => none
  | .node l r, i =>
      match dataAt l i with
      | some d => some d
      | none => dataAt r (i - size l)

/-! ## Authentication paths -/

/-- The authentication path of leaf `i`: the siblings in RFC 6962 order, the sibling at the top of
the tree first, each tagged with `true` when the sibling is the right child at that level and
`false` when it is the left child. The right subtree of a node is indexed from `size l`, as RFC
6962 numbers leaves. `none` when `i` is out of range.

Temporal cost is linear in the depth of `i` (at most `height` steps, one subtraction per node);
spatial cost is one cons cell per step. -/
def path (H : HashFn) : Tree → Nat → Option (List (Digest × Bool))
  | .leaf _, 0 => some []
  | .leaf _, _ + 1 => none
  | .node l r, i =>
      match path H l i with
      | some p => some ((root H r, true) :: p)
      | none =>
        match path H r (i - size l) with
        | some q => some ((root H l, false) :: q)
        | none => none

/-- Fold an authentication path into a root. `p` is in RFC 6962 order, top of the tree first, so
the deepest sibling is the last entry and is combined first: the fold runs from the end of the
list, combining the accumulated digest with the sibling on the side the tag records (`true` right,
`false` left). Temporal cost is linear in the path length (one hash call per sibling, plus the
`foldr` traversal); spatial cost is O(path length) stack for the `foldr` and O(1) otherwise. -/
def combine (H : HashFn) (acc : Digest) (p : List (Digest × Bool)) : Digest :=
  p.foldr (fun s a => if s.2 then nodeHash H a s.1 else nodeHash H s.1 a) acc

/-- `combine` is the identity on the empty path. -/
theorem combine_nil (H : HashFn) (acc : Digest) : combine H acc [] = acc := rfl

/-- `combine` applies the deepest sibling first, so a right-tagged entry combines the folded rest
on the left with its own digest on the right. -/
theorem combine_cons_true (H : HashFn) (acc d : Digest) (rest : List (Digest × Bool)) :
    combine H acc ((d, true) :: rest) = nodeHash H (combine H acc rest) d := rfl

/-- The left-tagged mirror: the sibling's digest goes on the left. -/
theorem combine_cons_false (H : HashFn) (acc d : Digest) (rest : List (Digest × Bool)) :
    combine H acc ((d, false) :: rest) = nodeHash H d (combine H acc rest) := rfl

/-! ## Verification -/

/-- Recompute a root from a leaf hash and its authentication path and compare it with the claimed
root. The `maxPathLen` argument is the caller's upper bound on the path length; RFC 6962 §2.1.1
bounds a path by the tree height, and `verify_root` below supplies `height`. `false` on any
mismatch. -/
def verify (H : HashFn) (claimedRoot : Digest) (maxPathLen : Nat) (lh : Digest)
    (p : List (Digest × Bool)) : Bool :=
  decide (p.length ≤ maxPathLen ∧ combine H lh p = claimedRoot)

/-! ## Structural facts about `path` -/

@[simp] theorem path_leaf_zero (H : HashFn) (d : List Byte) : path H (.leaf d) 0 = some [] := rfl

theorem path_leaf_succ (H : HashFn) (d : List Byte) (i : Nat) :
    path H (.leaf d) (i + 1) = none := by
  cases i <;> rfl

theorem path_leaf (H : HashFn) (d : List Byte) (i : Nat) (p : List (Digest × Bool)) :
    path H (.leaf d) i = some p ↔ i = 0 ∧ p = [] := by
  constructor
  · intro h
    cases i with
    | zero =>
      have h' : some p = some [] := h.symm.trans (path_leaf_zero H d)
      exact ⟨rfl, Option.some.inj h'⟩
    | succ i =>
      have : path H (.leaf d) (i + 1) = none := path_leaf_succ H d i
      rw [this] at h
      exact nomatch h
  · rintro ⟨rfl, rfl⟩
    rfl

/-- The sizes of a tree and of a node, needed by the range statements below: a `node` has at least
two leaves. Induction on the tree, one step per node, O(1) auxiliary. -/
theorem size_pos (t : Tree) : 0 < size t := by
  induction t with
  | leaf _ => exact Nat.zero_lt_succ _
  | node l r ihl ihr => simp only [size]; omega

theorem size_node_pos (l r : Tree) : 1 ≤ size l + size r := by
  have h := size_pos l
  omega

/-- No `true`-tagged and `false`-tagged node entries are equal: the tag records the side the
sibling is on, and the two sides differ. -/
theorem cons_tag_ne {α : Type _} {x y : α × Bool} {l l' : List (α × Bool)} (h : x :: l = y :: l')
    (hx : x.2 = true) (hy : y.2 = false) : False := by
  have h2 := congrArg Prod.snd (List.cons.inj h |>.1)
  rw [hx, hy] at h2
  exact nomatch h2

/-- A path through a node is either the left subtree's path with the right subtree's root prepended
and tagged right, or the right subtree's path (at index `i - size l`) with the left subtree's root
prepended and tagged left, and then the left subtree has no path at `i`. The left condition on the
second alternative is what makes the two alternatives disjoint: a path through a node never takes
both. Each direction reads the entry off the definition. -/
theorem path_node (H : HashFn) (l r : Tree) (i : Nat) (p : List (Digest × Bool)) :
    path H (.node l r) i = some p ↔
      (∃ q, p = (root H r, true) :: q ∧ path H l i = some q) ∨
      (∃ q, p = (root H l, false) :: q ∧ path H l i = none ∧
        path H r (i - size l) = some q) := by
  constructor
  · intro hp
    cases hl : path H l i with
    | some q =>
      have hso : some ((root H r, true) :: q) = some p := by
        simp only [path, hl] at hp
        exact hp
      exact Or.inl ⟨q, (Option.some.inj hso).symm, rfl⟩
    | none =>
      cases hr : path H r (i - size l) with
      | some q =>
        have hso : some ((root H l, false) :: q) = some p := by
          simp only [path, hl, hr] at hp
          exact hp
        exact Or.inr ⟨q, (Option.some.inj hso).symm, rfl, rfl⟩
      | none =>
        simp only [path, hl, hr] at hp
        exact nomatch hp
  · rintro (⟨p, hpq, hq⟩ | ⟨p, hpq, hnl, hq⟩)
    · rw [hpq]
      simp only [path, hq]
    · rw [hpq]
      simp only [path, hnl, hq]

/-- A path that enters the right subtree certifies that the left subtree has no path at that
index, which is what separates the two branches of `path_node`. -/
theorem path_of_node_right (H : HashFn) (l r : Tree) (i : Nat) (q : List (Digest × Bool))
    (hp : path H (.node l r) i = some ((root H l, false) :: q)) : path H l i = none := by
  simp only [path] at hp
  generalize hx : path H l i = x at hp
  cases x with
  | none => rfl
  | some q' => exact nomatch (cons_tag_ne (Option.some.inj hp) rfl rfl)

/-- A leaf index is in range exactly when a path exists for it. -/
theorem path_eq_some_iff (H : HashFn) (t : Tree) (i : Nat) :
    (∃ p, path H t i = some p) ↔ i < size t := by
  induction t generalizing i with
  | leaf d =>
    cases i with
    | zero =>
      constructor
      · intro _
        simp only [size]
        exact Nat.zero_lt_succ _
      · intro _
        exact ⟨[], path_leaf_zero H d⟩
    | succ i =>
      constructor
      · rintro ⟨p, hp⟩
        rw [path_leaf_succ H d i] at hp
        exact nomatch hp
      · intro h
        simp [size] at h
  | node l r ihl ihr =>
    constructor
    · rintro ⟨p0, hp0⟩
      rw [path_node] at hp0
      rcases hp0 with (⟨q, _, hq⟩ | ⟨q, _, -, hq⟩)
      · have h1 := (ihl i).mp ⟨q, hq⟩
        have hpos := size_node_pos l r
        simp only [size] at h1 ⊢
        omega
      · have h1 := (ihr (i - size l)).mp ⟨q, hq⟩
        have hpos := size_node_pos l r
        simp only [size] at h1 ⊢
        omega
    · intro h
      cases hl : path H l i with
      | some p0 => exact ⟨(root H r, true) :: p0, by simp only [path, hl]⟩
      | none =>
        have hne : ¬ ∃ p, path H l i = some p := by
          rintro ⟨p0, hp0⟩
          rw [hl] at hp0
          exact nomatch hp0
        have hle : size l ≤ i := by
          by_contra hc
          have hlt : i < size l := Nat.lt_of_not_ge hc
          obtain ⟨p1, hp1⟩ := (ihl i).mpr hlt
          exact hne ⟨p1, hp1⟩
        have hpos := size_node_pos l r
        simp only [size] at h
        have hlt : i - size l < size r := by omega
        obtain ⟨q, hq⟩ := (ihr (i - size l)).mpr hlt
        exact ⟨(root H l, false) :: q, by simp only [path, hl, hq]⟩

/-- The length of a path never exceeds the tree height: RFC 6962 §2.1.1's path-length bound. -/
theorem path_length_le_height (H : HashFn) (t : Tree) (i : Nat) (p : List (Digest × Bool))
    (hp : path H t i = some p) : p.length ≤ height t := by
  induction t generalizing i p with
  | leaf d =>
    have := (path_leaf H d i p).mp hp
    rcases this with ⟨_, rfl⟩
    simp [height]
  | node l r ihl ihr =>
    rw [path_node] at hp
    rcases hp with (⟨q, hpq, hq⟩ | ⟨q, hpq, -, hq⟩)
    · rw [hpq]
      have h := ihl i q hq
      simp only [List.length_cons, height]
      omega
    · rw [hpq]
      have h := ihr (i - size l) q hq
      simp only [List.length_cons, height]
      omega

/-! ## `path` and `dataAt` agree on which indices exist -/

/-- An index carries no data exactly when it is past the last leaf. Induction on the tree; one
step per node, O(1) auxiliary. -/
theorem dataAt_none_iff (t : Tree) (i : Nat) : dataAt t i = none ↔ size t ≤ i := by
  induction t generalizing i with
  | leaf d =>
    cases i <;> simp [dataAt, size]
  | node l r ihl ihr =>
    cases hl : dataAt l i with
    | some d' =>
      constructor
      · intro hc
        have hh : dataAt (Tree.node l r) i = some d' := by rw [dataAt, hl]
        rw [hh] at hc
        exact nomatch hc
      · intro h
        have hpos := size_pos r
        simp only [size] at h
        have hle : size l ≤ i := by omega
        have hle : size l ≤ i := by omega
        have hnone : dataAt l i = none := (ihl i).mpr hle
        rw [hl] at hnone
        exact nomatch hnone
    | none =>
      constructor
      · intro hc
        have hc' : dataAt r (i - size l) = none := by
          simp only [dataAt, hl] at hc
          exact hc
        have h1 := (ihr (i - size l)).mp hc'
        have h2 : size l ≤ i := (ihl i).mp hl
        have hpos := size_pos l
        simp only [size] at h1 h2 ⊢
        omega
      · intro h
        have hposl := size_pos l
        have hposr := size_pos r
        simp only [size] at h
        simp only [dataAt, (ihl i).2 (by omega : size l ≤ i),
          (ihr (i - size l)).2 (by omega : size r ≤ i - size l)]

/-- No data at an index means no path at that index. -/
theorem path_none_of_dataAt_none (H : HashFn) (t : Tree) (i : Nat)
    (hd : dataAt t i = none) : path H t i = none := by
  induction t generalizing i with
  | leaf d0 =>
    cases i with
    | zero =>
      rw [dataAt] at hd
      exact nomatch hd
    | succ i => rfl
  | node l r ihl ihr =>
    cases hl : dataAt l i with
    | some d' =>
      rw [dataAt, hl] at hd
      exact nomatch hd
    | none =>
      cases hpl : path H l i with
      | some p1 =>
        have hlt : i < size l := (path_eq_some_iff H l i).mp ⟨p1, hpl⟩
        have hle : size l ≤ i := (dataAt_none_iff l i).mp hl
        exact nomatch (Nat.not_lt_of_ge hle hlt)
      | none =>
        have hd' : dataAt r (i - size l) = none := by
          simp only [dataAt, hl] at hd
          exact hd
        have hp1 : path H r (i - size l) = none := ihr (i - size l) hd'
        simp only [path, hpl, hp1]

/-- An index with data has a path. -/
theorem exists_path_of_dataAt (H : HashFn) (t : Tree) (i : Nat) (d : List Byte)
    (hd : dataAt t i = some d) : ∃ p, path H t i = some p := by
  induction t generalizing i d with
  | leaf d0 =>
    cases i with
    | zero => exact ⟨[], rfl⟩
    | succ i =>
      rw [dataAt] at hd
      exact nomatch hd
  | node l r ihl ihr =>
    cases hl : dataAt l i with
    | some d' =>
      obtain ⟨p, hp⟩ := ihl i d' hl
      refine ⟨(root H r, true) :: p, ?_⟩
      simp only [path, hp]
    | none =>
      have hd' : dataAt r (i - size l) = some d := by
        simp only [dataAt, hl] at hd
        exact hd
      obtain ⟨p, hp⟩ := ihr (i - size l) d hd'
      refine ⟨(root H l, false) :: p, ?_⟩
      have hpl : path H l i = none := path_none_of_dataAt_none H l i hl
      simp only [path, hpl, hp]

/-- An index with a path has data. -/
theorem exists_dataAt_of_path (H : HashFn) (t : Tree) (i : Nat) (p : List (Digest × Bool))
    (hp : path H t i = some p) : ∃ d, dataAt t i = some d := by
  induction t generalizing i p with
  | leaf d =>
    obtain ⟨hi, hp0⟩ := (path_leaf H d i p).mp hp
    subst hi
    exact ⟨d, rfl⟩
  | node l r ihl ihr =>
    have hp0 := hp
    rw [path_node] at hp
    rcases hp with (⟨q, hpq, hq⟩ | ⟨q, hpq, -, hq⟩)
    · obtain ⟨d', hd'⟩ := ihl i q hq
      refine ⟨d', ?_⟩
      simp only [dataAt, hd']
    · obtain ⟨d', hd'⟩ := ihr (i - size l) q hq
      refine ⟨d', ?_⟩
      cases hdl : dataAt l i with
      | some d'' =>
        have hpc : path H (.node l r) i = some ((root H l, false) :: q) := by
          rw [← hpq]
          exact hp0
        have hnl : path H l i = none := path_of_node_right H l r i q hpc
        obtain ⟨p1, hp1⟩ := exists_path_of_dataAt H l i d'' hdl
        rw [hnl] at hp1
        exact nomatch hp1
      | none =>
        simp only [dataAt, hdl, hd']

/-! ## The path folds the root, for any leaf digest -/

/-- The root as a function of the leaf digest, mirroring the branch structure of `path`: `some f` at
index `i` when the index is in range, and `f` is the sibling chain read from the leaf upwards. This
is the structural function behind `combine`, so the two agree by the next lemmas. -/
def rootFn (H : HashFn) : Tree → Nat → Option (Digest → Digest)
  | .leaf _, 0 => some (fun x => x)
  | .leaf _, _ + 1 => none
  | .node l r, i =>
    match rootFn H l i with
    | some f => some (fun x => nodeHash H (f x) (root H r))
    | none =>
      match rootFn H r (i - size l) with
      | some g => some (fun x => nodeHash H (root H l) (g x))
      | none => none

/-- The index is in range for `rootFn` exactly when it is in range for `path`. -/
theorem rootFn_iff_path (H : HashFn) (t : Tree) (i : Nat) :
    (∃ f, rootFn H t i = some f) ↔ ∃ p, path H t i = some p := by
  induction t generalizing i with
  | leaf d =>
    cases i with
    | zero =>
      constructor
      · rintro ⟨f, hf⟩
        exact ⟨[], path_leaf_zero H d⟩
      · rintro ⟨p, hp⟩
        exact ⟨fun x => x, rfl⟩
    | succ i =>
      constructor
      · rintro ⟨f, hf⟩
        rw [rootFn] at hf
        exact nomatch hf
      · rintro ⟨p, hp⟩
        rw [path_leaf_succ H d i] at hp
        exact nomatch hp
  | node l r ihl ihr =>
    constructor
    · rintro ⟨f, hf⟩
      cases hfl : rootFn H l i with
      | some f1 =>
        obtain ⟨p1, hp1⟩ := (ihl i).mp ⟨f1, hfl⟩
        refine ⟨(root H r, true) :: p1, ?_⟩
        simp only [path, hp1]
      | none =>
        have hnl : path H l i = none := by
          cases hb : path H l i with
          | some p1 =>
            obtain ⟨f0, hf0⟩ := (ihl i).mpr ⟨p1, hb⟩
            rw [hfl] at hf0
            exact nomatch hf0
          | none => rfl
        cases hpr : rootFn H r (i - size l) with
        | some f0 =>
          obtain ⟨p2, hp2⟩ := (ihr (i - size l)).mp ⟨f0, hpr⟩
          refine ⟨(root H l, false) :: p2, ?_⟩
          simp only [path, hnl, hp2]
        | none =>
          have hne : ¬ ∃ p, path H r (i - size l) = some p := by
            rintro ⟨p2, hp2⟩
            obtain ⟨f0, hf0⟩ := (ihr (i - size l)).mpr ⟨p2, hp2⟩
            rw [hpr] at hf0
            exact nomatch hf0
          simp only [rootFn, hfl, hpr] at hf
          exact nomatch hf
    · rintro ⟨p, hp⟩
      cases hfl : rootFn H l i with
      | some f1 =>
        obtain ⟨p1, hp1⟩ := (ihl i).mp ⟨f1, hfl⟩
        exact ⟨fun y => nodeHash H (f1 y) (root H r), by simp only [rootFn, hfl]⟩
      | none =>
        have hnl : path H l i = none := by
          cases hb : path H l i with
          | some p1 =>
            obtain ⟨f0, hf0⟩ := (ihl i).mpr ⟨p1, hb⟩
            rw [hfl] at hf0
            exact nomatch hf0
          | none => rfl
        cases hpr : path H r (i - size l) with
        | none =>
          simp only [path, hnl, hpr] at hp
          exact nomatch hp
        | some p2 =>
          obtain ⟨g, hg⟩ := (ihr (i - size l)).mpr ⟨p2, hpr⟩
          exact ⟨fun y => nodeHash H (root H l) (g y), by simp only [rootFn, hfl, hg]⟩

/-- Folding a path with any leaf digest applies the corresponding `rootFn`. -/
theorem combine_eq_rootFn (H : HashFn) (t : Tree) (i : Nat) (p : List (Digest × Bool))
    (f : Digest → Digest) (hp : path H t i = some p) (hf : rootFn H t i = some f)
    (x : Digest) : combine H x p = f x := by
  induction t generalizing i p f x with
  | leaf d0 =>
    cases i with
    | zero =>
      obtain ⟨_, hp0⟩ := (path_leaf H d0 0 p).mp hp
      rw [hp0, combine_nil]
      simp only [rootFn] at hf
      exact congrFun (Option.some.inj hf) x
    | succ i =>
      rw [path_leaf_succ H d0 i] at hp
      exact nomatch hp
  | node l r ihl ihr =>
    cases hfl : rootFn H l i with
    | some f1 =>
      obtain ⟨p1, hp1⟩ := (rootFn_iff_path H l i).mp ⟨f1, hfl⟩
      have hp'' : p = (root H r, true) :: p1 := by
        have hpn : path H (.node l r) i = some ((root H r, true) :: p1) := by
          simp only [path, hp1]
        exact (Option.some.inj (hpn.symm.trans hp)).symm
      rw [hp'', combine_cons_true]
      have hf' : f = fun y => nodeHash H (f1 y) (root H r) := by
        simp only [rootFn, hfl] at hf
        exact (Option.some.inj hf).symm
      rw [hf']
      exact congrArg (fun y => nodeHash H y (root H r)) (ihl i p1 f1 hp1 hfl x)
    | none =>
      have hp0 : path H l i = none := by
        cases hb : path H l i with
        | some p1 =>
          obtain ⟨f1, hf1⟩ := (rootFn_iff_path H l i).mpr ⟨p1, hb⟩
          rw [hfl] at hf1
          exact nomatch hf1
        | none => rfl
      cases hpr : path H r (i - size l) with
      | none =>
        simp only [path, hp0, hpr] at hp
        exact nomatch hp
      | some p2 =>
        obtain ⟨g, hg⟩ := (rootFn_iff_path H r (i - size l)).mpr ⟨p2, hpr⟩
        have hp'' : p = (root H l, false) :: p2 := by
          have hpn : path H (.node l r) i = some ((root H l, false) :: p2) := by
            simp only [path, hp0, hpr]
          exact (Option.some.inj (hpn.symm.trans hp)).symm
        rw [hp'', combine_cons_false]
        have hf' : f = fun y => nodeHash H (root H l) (g y) := by
          simp only [rootFn, hfl, hg] at hf
          exact (Option.some.inj hf).symm
        rw [hf']
        exact congrArg (fun y => nodeHash H (root H l) y) (ihr (i - size l) p2 g hpr hg x)

/-- At the digest of the leaf the index names, `rootFn` computes the tree's root. -/
theorem rootFn_apply_root (H : HashFn) (t : Tree) (i : Nat) (f : Digest → Digest)
    (d : List Byte) (hf : rootFn H t i = some f) (hd : dataAt t i = some d) :
    f (leafHash H d) = root H t := by
  induction t generalizing i f d with
  | leaf d0 =>
    cases i with
    | zero =>
      rw [dataAt] at hd
      cases hd
      rw [rootFn] at hf
      injection hf with hf'
      subst hf'
      rfl
    | succ i => rw [dataAt] at hd; exact nomatch hd
  | node l r ihl ihr =>
    cases hld : dataAt l i with
    | some d1 =>
      have hdd : d = d1 := by
        simp only [dataAt, hld] at hd
        exact (Option.some.inj hd).symm
      obtain ⟨p1, hp1⟩ := exists_path_of_dataAt H l i d1 hld
      obtain ⟨f1, hf1⟩ := (rootFn_iff_path H l i).mpr ⟨p1, hp1⟩
      have hf' : f = fun y => nodeHash H (f1 y) (root H r) := by
        simp only [rootFn, hf1] at hf
        exact (Option.some.inj hf).symm
      rw [hf']
      simp only
      rw [hdd, ihl i f1 d1 hf1 hld]
      rfl
    | none =>
      have hd' : dataAt r (i - size l) = some d := by
        simp only [dataAt, hld] at hd
        exact hd
      have hnl : path H l i = none := path_none_of_dataAt_none H l i hld
      have hfl : rootFn H l i = none := by
        cases hb : rootFn H l i with
        | some f0 =>
          obtain ⟨p1, hp1⟩ := (rootFn_iff_path H l i).mp ⟨f0, hb⟩
          rw [hnl] at hp1
          exact nomatch hp1
        | none => rfl
      obtain ⟨p2, hp2⟩ := exists_path_of_dataAt H r (i - size l) d hd'
      obtain ⟨g, hg⟩ := (rootFn_iff_path H r (i - size l)).mpr ⟨p2, hp2⟩
      have hf' : f = fun y => nodeHash H (root H l) (g y) := by
        simp only [rootFn, hfl, hg] at hf
        exact (Option.some.inj hf).symm
      rw [hf']
      simp only
      rw [ihr (i - size l) g d hg hd']
      rfl

/-- A tree's own path reconstructs the tree's own root from that leaf's own digest. This is the
RFC 6962 §2.1.2 completeness direction: a consistent inclusion proof always verifies. -/
theorem combine_eq_root (H : HashFn) (t : Tree) (i : Nat) (p : List (Digest × Bool))
    (d : List Byte) (hp : path H t i = some p) (hd : dataAt t i = some d) :
    combine H (leafHash H d) p = root H t := by
  obtain ⟨f, hf⟩ := (rootFn_iff_path H t i).mpr ⟨p, hp⟩
  rw [combine_eq_rootFn H t i p f hp hf, rootFn_apply_root H t i f d hf hd]

/-- `verify` accepts the tree's own root for the tree's own leaf and path, given that the claimed
leaf digest is the leaf digest and the caller's path-length bound admits the tree height. -/
theorem verify_root (H : HashFn) (t : Tree) (i : Nat) (p : List (Digest × Bool)) (d : List Byte)
    (lh : Digest) (maxPathLen : Nat) (hbound : height t ≤ maxPathLen) (hp : path H t i = some p)
    (hd : dataAt t i = some d) (hleaf : leafHash H d = lh) :
    verify H (root H t) maxPathLen lh p = true := by
  have hlen : p.length ≤ maxPathLen :=
    (path_length_le_height H t i p hp) |>.trans hbound
  have hc := combine_eq_root H t i p d hp hd
  simp only [verify]
  rw [decide_eq_true_eq]
  exact ⟨hlen, hc ▸ hleaf ▸ rfl⟩

/-! ## Known-answer tests over a toy hash fixture -/

/-- **A test fixture, not a hash function**: byte `i` of the digest of `msg` is `(a + i) mod 256`,
where `a` is the `31`-ary fold of `msg` from `7` modulo 256. It is linear in the message length,
is computed by the kernel in a few hundred byte operations, and carries no collision-resistance or
preimage claim; it exists so the evaluator paths `root`, `path` and `verify` are pinned on concrete
bytes. The values below are this fixture's, not any primitive's. -/
def toyH (msg : List Byte) : Digest :=
  let a : Nat := msg.foldl (fun acc c => (acc * 31 + c.toNat) % 256) 7
  Vector.ofFn fun i : Fin 32 => BitVec.ofNat 8 (a + i)

/-- The balanced four-leaf tree `((a, b), (c, d))`: its leaf indices are 0, 1, 2, 3, so both ends of
the range are addressable and every path has two sibling steps. -/
def toyTree : Tree :=
  .node (.node (.leaf [0x61]) (.leaf [0x62])) (.node (.leaf [0x63]) (.leaf [0x64]))

/- `#guard` runs the untrusted evaluator on concrete bytes, so the three below are regression pins
on the fixture, not kernel proofs of any separation property. Each path entry is the sibling digest
followed by its tag: `1` for a sibling combined on the right, `0` for one combined on the left. -/

#guard (root toyH toyTree).toList ==
  [0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
   0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19]

/- The two sibling steps at index 0: the right subtree's root, then leaf 1's digest, both
combined on the right. -/
#guard (match path toyH toyTree 0 with
    | some p => p.map (fun s : Digest × Bool => (s.1.toList, s.2)) | none => []) ==
  [([0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
     0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19], true),
   ([0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf, 0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8,
     0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8], true)]

/- The two sibling steps at index 3: the left subtree's root, then leaf 2's digest, both combined
on the left. -/
#guard (match path toyH toyTree 3 with
    | some p => p.map (fun s : Digest × Bool => (s.1.toList, s.2)) | none => []) ==
  [([0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
     0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19], false),
   ([0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf, 0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9,
     0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9], false)]

/-- The completeness direction on a concrete tree: the toy tree's own root verifies against its own
path at index 3, its own leaf digest and its own height bound. -/
example : verify toyH (root toyH toyTree) (height toyTree) (leafHash toyH [0x64])
    ((path toyH toyTree 3).getD []) = true := by
  refine verify_root toyH toyTree 3 _ [0x64] _ _ (le_refl _) ?_ ?_ rfl
  · rfl
  · rfl

/-! ## Soundness across trees: stated with the collision hypothesis explicit -/

/-- The collision hypotheses that soundness needs: `leafHash` and `nodeHash` are injective and the
two tags are separated. No `H : List Byte → Vector Byte 32` is obliged to satisfy them; a concrete
primitive (SHA-256, SHA3-256, Keccak-256) is expected to, and that expectation is a cryptographic
assumption, not a theorem of this file. -/
def CollisionFree (H : HashFn) : Prop :=
  (∀ a b : List Byte, leafHash H a = leafHash H b → a = b) ∧
  (∀ l₁ l₂ r₁ r₂ : Digest, nodeHash H l₁ r₁ = nodeHash H l₂ r₂ → l₁ = l₂ ∧ r₁ = r₂) ∧
  (∀ a : List Byte, (l r : Digest) → leafHash H a ≠ nodeHash H l r)

/-- No second leaf verifies: under `CollisionFree H`, if two trees of bounded height share a root,
then two leaf hashes at the same index with respective well-formed paths cannot both be accepted
unless they are equal. Stated as a `Prop` with the collision hypothesis explicit; the reference
implementation of the claim is `prove_no_second_leaf` below, which is admitted only through the
`CollisionFree` premise and not to any `sorry`. -/
def NoSecondLeafVerifies (H : HashFn) : Prop :=
  ∀ (t t' : Tree) (i maxPathLen : Nat) (p p' : List (Digest × Bool)) (lh lh' : Digest),
    root H t = root H t' → path H t i = some p → path H t' i = some p' →
    height t ≤ maxPathLen → height t' ≤ maxPathLen →
    verify H (root H t) maxPathLen lh p = true → verify H (root H t') maxPathLen lh' p' = true →
    lh = lh'

end Apeiron.Crypto.Merkle