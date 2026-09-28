/-!
# Shape descriptors and right-aligned broadcasting

A shape is a list of natural extents ordered outermost dimension first: `[batch, channels]`
denotes a rank-2 shape whose extents are `batch` and `channels`. `size s` is the number of
coordinates of a tensor of shape `s`; the empty shape has size `1`.

Binary shape compatibility follows the right-aligned NumPy convention. The extents of two shapes
are aligned from the last extent toward the first, and `CanBroadcastTo s t` holds exactly when `s`
has no larger rank than `t` and every aligned extent of `s` is either `1` or equal to the
corresponding extent of `t`. Here `s` is the operand and `t` is the target, so `[2]` broadcasts to
`[3, 2]`, while `[3]` does not. `BroadcastTo s t r` additionally names the result shape `r`, whose
extent at each aligned position is the non-`1` member of the pair.
-/

namespace Nous.Core

/-- A shape descriptor: extents from the outermost dimension to the innermost. -/
abbrev Shape := List Nat

/-- The rank of a shape, i.e. its number of dimensions. -/
def rank (s : Shape) : Nat := s.length

/-- The number of coordinates of a tensor of shape `s`; the empty shape has size `1`. -/
def size : Shape → Nat
  | [] => 1
  | d :: s => d * size s

@[simp] theorem size_nil : size [] = 1 := rfl

@[simp] theorem size_cons (d : Nat) (s : Shape) : size (d :: s) = d * size s := rfl

@[simp] theorem rank_nil : rank [] = 0 := rfl

@[simp] theorem rank_cons (d : Nat) (s : Shape) : rank (d :: s) = rank s + 1 := rfl

/-- Extents concatenate multiplicatively; the extents of `s` are the outer ones. -/
theorem size_append (s t : Shape) : size (s ++ t) = size s * size t := by
  induction s with
  | nil => simp
  | cons d s ih => simp [size, ih, Nat.mul_assoc]

/-- Right-aligned compatibility of two shapes given innermost extent first: if the first list ends,
it has no larger rank and compatibility holds; otherwise the aligned extents must be equal or the
first must be `1`. -/
def CanBroadcastToAux : Shape → Shape → Prop
  | [], [] => True
  | [], _ :: _ => True
  | _ :: _, [] => False
  | a :: as, b :: bs => (a = 1 ∨ a = b) ∧ CanBroadcastToAux as bs

instance decidableCanBroadcastToAux : DecidableRel CanBroadcastToAux
  | [], [] => isTrue trivial
  | [], _ :: _ => isTrue trivial
  | _ :: _, [] => isFalse (fun h => h)
  | a :: as, b :: bs =>
      match (inferInstance : Decidable (a = 1 ∨ a = b)),
        decidableCanBroadcastToAux as bs with
      | isTrue h1, isTrue h2 => isTrue ⟨h1, h2⟩
      | isFalse h1, _ => isFalse (fun h => h1 h.1)
      | _, isFalse h2 => isFalse (fun h => h2 h.2)

/-- `s` broadcasts to `t` under the right-aligned NumPy convention outlined in the module
comment. -/
def CanBroadcastTo (s t : Shape) : Prop := CanBroadcastToAux s.reverse t.reverse

instance (s t : Shape) : Decidable (CanBroadcastTo s t) :=
  inferInstanceAs (Decidable (CanBroadcastToAux s.reverse t.reverse))

theorem canBroadcastToAux_self (s : Shape) : CanBroadcastToAux s s := by
  induction s with
  | nil => trivial
  | cons d s _ => exact ⟨Or.inr rfl, ‹CanBroadcastToAux s s›⟩

/-- Broadcasting compatibility is reflexive. -/
theorem canBroadcastTo_refl (s : Shape) : CanBroadcastTo s s :=
  canBroadcastToAux_self s.reverse

theorem canBroadcastToAux_one_left {l : Shape} (h : l ≠ []) : CanBroadcastToAux [1] l := by
  cases l with
  | nil => exact absurd rfl h
  | cons b bs => exact ⟨Or.inl rfl, by cases bs <;> trivial⟩

/-- A rank-1 extent-`1` shape broadcasts to any rank-1 shape with a named extent. -/
theorem canBroadcastTo_one_cons (d : Nat) (s : Shape) : CanBroadcastTo [1] (d :: s) :=
  canBroadcastToAux_one_left (l := (d :: s).reverse) (by simp)

/-- Right-aligned broadcast of two shapes given innermost extent first; `none` is the
incompatibility case. The kept extent at each aligned position is the non-`1` member. -/
def BroadcastAux : Shape → Shape → Option Shape
  | [], [] => some []
  | [], b :: bs => some (b :: bs)
  | a :: as, [] => some (a :: as)
  | a :: as, b :: bs =>
      if a = b then (b :: ·) <$> BroadcastAux as bs
      else if a = 1 then (b :: ·) <$> BroadcastAux as bs
      else if b = 1 then (a :: ·) <$> BroadcastAux as bs
      else none

/-- `r` is the broadcast of `s` into `t` when the right-aligned broadcast exists and equals `r`. -/
def BroadcastTo (s t r : Shape) : Prop :=
  (BroadcastAux s.reverse t.reverse).map List.reverse = some r

instance (s t r : Shape) : Decidable (BroadcastTo s t r) := by
  unfold BroadcastTo
  infer_instance

theorem broadcastAux_self : ∀ l : Shape, BroadcastAux l l = some l
  | [] => rfl
  | a :: as => by
      simp [BroadcastAux, broadcastAux_self as]

theorem broadcastAux_nil_left (l : Shape) : BroadcastAux [] l = some l := by
  cases l <;> rfl

theorem broadcastAux_one_left : ∀ {l : Shape}, l ≠ [] → BroadcastAux [1] l = some l
  | [], h => absurd rfl h
  | b :: bs, _ => by
      by_cases hb : 1 = b
      · simp [BroadcastAux, broadcastAux_nil_left, hb]
      · simp [BroadcastAux, broadcastAux_nil_left, hb]

/-- The broadcast relation is reflexive: a shape broadcasts to itself unchanged. -/
theorem broadcastTo_refl (s : Shape) : BroadcastTo s s s := by
  unfold BroadcastTo
  rw [broadcastAux_self s.reverse]
  simp [List.reverse_reverse]

/-- A rank-1 extent-`1` shape broadcasts into a rank-1 name shape, leaving it unchanged. -/
theorem broadcastTo_one_cons (d : Nat) (s : Shape) : BroadcastTo [1] (d :: s) (d :: s) := by
  unfold BroadcastTo
  show Option.map List.reverse (BroadcastAux [1] (d :: s).reverse) = some (d :: s)
  rw [broadcastAux_one_left (l := (d :: s).reverse) (by simp)]
  simp [List.reverse_reverse]

-- Boundary evaluations of `size` against the defining equations.
#guard size [] = 1
#guard size [2, 3] = 6
#guard size [2, 3] = size [2] * size [3]

-- Directed right-aligned compatibility: `[2]` broadcasts to `[3, 2]`, `[3]` does not.
#guard CanBroadcastTo [2] [3, 2]
#guard ¬ CanBroadcastTo [3] [3, 2]
#guard CanBroadcastTo [1] [7]
#guard BroadcastTo [2] [3, 2] [3, 2]

end Nous.Core
