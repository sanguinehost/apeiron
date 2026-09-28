import Nous.Core.Shape
import Mathlib.Algebra.Ring.Defs

/-!
# Shape-indexed tensors over a commutative ring

A tensor of shape `s` over `R` is a function from the `size s` coordinates `Fin (size s)` to `R`.
Every operation is pointwise, so the type is total: a shape with a zero extent has no coordinate,
and no operation discards or fails to produce a value. The definitions here are ring-generic; they
introduce no broadcasting, coordinate or reshaping map, which the statement `CanBroadcastTo` in
`Nous.Core.Shape` describes separately.
-/

namespace Nous.Core

/-- A tensor of shape `s` over `R`: the coordinates `Fin (size s)` mapped to ring elements. -/
abbrev Tensor (R : Type _) [CommRing R] (s : Shape) : Type _ := Fin (size s) → R

variable {R : Type _} [CommRing R] {s : Shape}

/-- The pointwise zero tensor. -/
def Tensor.zero : Tensor R s := fun _ => 0

/-- Pointwise addition of tensors. -/
def Tensor.add (f g : Tensor R s) : Tensor R s := fun x => f x + g x

/-- Pointwise subtraction of tensors. -/
def Tensor.sub (f g : Tensor R s) : Tensor R s := fun x => f x - g x

/-- Pointwise multiplication of tensors. -/
def Tensor.mul (f g : Tensor R s) : Tensor R s := fun x => f x * g x

/-- Pointwise scaling of a tensor by a ring element. -/
def Tensor.scale (c : R) (f : Tensor R s) : Tensor R s := fun x => c * f x

/-- Two tensors are equal when they agree at every coordinate. -/
@[ext] theorem Tensor.ext {f g : Tensor R s} (h : ∀ x, f x = g x) : f = g :=
  funext h

/-- Pointwise addition is commutative. -/
theorem Tensor.add_comm (f g : Tensor R s) : Tensor.add f g = Tensor.add g f := by
  funext x
  exact _root_.add_comm (f x) (g x)

/-- Pointwise addition is associative. -/
theorem Tensor.add_assoc (f g h : Tensor R s) :
    Tensor.add (Tensor.add f g) h = Tensor.add f (Tensor.add g h) := by
  funext x
  exact _root_.add_assoc (f x) (g x) (h x)

/-- The pointwise zero tensor is a left identity for pointwise addition. -/
@[simp] theorem Tensor.zero_add (f : Tensor R s) : Tensor.add Tensor.zero f = f := by
  funext x
  exact _root_.zero_add (f x)

/-- The pointwise zero tensor is a right identity for pointwise addition. -/
@[simp] theorem Tensor.add_zero (f : Tensor R s) : Tensor.add f Tensor.zero = f := by
  funext x
  exact _root_.add_zero (f x)

/-- Scaling distributes over pointwise addition. -/
theorem Tensor.scale_add (c : R) (f g : Tensor R s) :
    Tensor.scale c (Tensor.add f g) = Tensor.add (Tensor.scale c f) (Tensor.scale c g) := by
  funext x
  exact _root_.mul_add c (f x) (g x)

/-- Scalar addition distributes over pointwise scaling. -/
theorem Tensor.add_scale (a b : R) (f : Tensor R s) :
    Tensor.scale (a + b) f = Tensor.add (Tensor.scale a f) (Tensor.scale b f) := by
  funext x
  exact _root_.add_mul a b (f x)

end Nous.Core
