import Nous.Core.Tensor

/-!
# Reverse-mode vector-Jacobian product operator contracts

This module formalizes ring-generic operator contracts for reverse-mode automatic
differentiation (vector-Jacobian products).

For a commutative ring `R` and shapes `σ, τ`:
- `OpSpec R σ τ` packages a forward evaluation map `Tensor R σ → Tensor R τ` and a
  backward map `Tensor R σ → Tensor R τ → Tensor R σ` computing the vector-Jacobian
  product (VJP) with respect to an upstream cotangent.
- `OpSpec.id` defines the identity operator.
- `OpSpec.compose` (`>>>`) defines sequential operator composition, threading
  intermediate forward outputs and composing backward VJPs in reverse order.

The reverse-mode chain rule `opSpec_compose_vjp` holds definitionally (`rfl`) by
construction of `OpSpec.compose`. Operator composition satisfies two-sided identity
laws and definitional associativity. Cotangent linearity (cotangent additivity and
scalar homogeneity) is preserved under identity and operator composition.
-/

namespace Nous.Core

/-- Atomic operator specification: forward evaluation map and backward vector-Jacobian
product map over a commutative ring `R`. -/
@[ext]
structure OpSpec (R : Type*) [CommRing R] (σ τ : Shape) where
  /-- Forward evaluation mapping input tensors of shape `σ` to output tensors of shape `τ`. -/
  forward : Tensor R σ → Tensor R τ
  /-- Backward vector-Jacobian product: given input tensor `x` and upstream cotangent `dL`,
  returns the cotangent with respect to `x`. -/
  backward : Tensor R σ → Tensor R τ → Tensor R σ

variable {R : Type*} [CommRing R]

namespace OpSpec

/-- The identity operator on tensors of shape `σ`. -/
def id (σ : Shape) : OpSpec R σ σ where
  forward x := x
  backward _ dL := dL

/-- Sequential composition of two operators with the reverse-mode chain rule.
Given `op1 : σ → τ` and `op2 : τ → υ`, forward evaluates `op2 (op1 x)` and backward
propagates cotangents in reverse: `op1.backward x (op2.backward (op1.forward x) dL)`. -/
def compose {σ τ υ : Shape}
    (op1 : OpSpec R σ τ) (op2 : OpSpec R τ υ) : OpSpec R σ υ where
  forward x := op2.forward (op1.forward x)
  backward x dL := op1.backward x (op2.backward (op1.forward x) dL)

end OpSpec

/-- Infix notation for operator composition. -/
infixr:80 " >>> " => OpSpec.compose

namespace OpSpec

/-- Forward evaluation of composed operators.
Holds definitionally by construction of `OpSpec.compose`. -/
theorem compose_forward {σ τ υ : Shape}
    (op1 : OpSpec R σ τ) (op2 : OpSpec R τ υ)
    (x : Tensor R σ) :
    (op1 >>> op2).forward x = op2.forward (op1.forward x) :=
  rfl

/-- Reverse-mode vector-Jacobian product chain rule for composed operators.
Holds definitionally (`rfl`) by construction of `OpSpec.compose`. -/
theorem compose_backward {σ τ υ : Shape}
    (op1 : OpSpec R σ τ) (op2 : OpSpec R τ υ)
    (x : Tensor R σ) (dL : Tensor R υ) :
    (op1 >>> op2).backward x dL = op1.backward x (op2.backward (op1.forward x) dL) :=
  rfl

/-- The identity operator is a left identity for operator composition. -/
theorem id_compose {σ τ : Shape} (op : OpSpec R σ τ) :
    (id σ >>> op) = op := by
  ext x <;> rfl

/-- The identity operator is a right identity for operator composition. -/
theorem compose_id {σ τ : Shape} (op : OpSpec R σ τ) :
    (op >>> id τ) = op := by
  ext x <;> rfl

/-- Operator composition is associative.
Holds definitionally (`rfl`) by construction of `OpSpec.compose`. -/
theorem compose_assoc {σ τ υ φ : Shape}
    (op1 : OpSpec R σ τ) (op2 : OpSpec R τ υ) (op3 : OpSpec R υ φ) :
    ((op1 >>> op2) >>> op3) = (op1 >>> (op2 >>> op3)) :=
  rfl

/-- Additivity of the backward vector-Jacobian product in the upstream cotangent argument. -/
def CotangentAdditive {σ τ : Shape} (op : OpSpec R σ τ) : Prop :=
  ∀ (x : Tensor R σ) (dL1 dL2 : Tensor R τ),
    op.backward x (Tensor.add dL1 dL2) = Tensor.add (op.backward x dL1) (op.backward x dL2)

/-- Scalar homogeneity of the backward vector-Jacobian product in the upstream cotangent argument. -/
def CotangentHomogeneous {σ τ : Shape} (op : OpSpec R σ τ) : Prop :=
  ∀ (x : Tensor R σ) (c : R) (dL : Tensor R τ),
    op.backward x (Tensor.scale c dL) = Tensor.scale c (op.backward x dL)

/-- Cotangent linearity: joint additivity and scalar homogeneity of the backward map. -/
def CotangentLinear {σ τ : Shape} (op : OpSpec R σ τ) : Prop :=
  CotangentAdditive op ∧ CotangentHomogeneous op

/-- The identity operator has an additive backward map. -/
theorem id_cotangentAdditive (σ : Shape) : (id (R := R) σ).CotangentAdditive :=
  fun _ _ _ => rfl

/-- The identity operator has a homogeneous backward map. -/
theorem id_cotangentHomogeneous (σ : Shape) : (id (R := R) σ).CotangentHomogeneous :=
  fun _ _ _ => rfl

/-- The identity operator has a cotangent-linear backward map. -/
theorem id_cotangentLinear (σ : Shape) : (id (R := R) σ).CotangentLinear :=
  ⟨id_cotangentAdditive σ, id_cotangentHomogeneous σ⟩

/-- Operator composition preserves cotangent additivity. -/
theorem compose_cotangentAdditive {σ τ υ : Shape}
    {op1 : OpSpec R σ τ} {op2 : OpSpec R τ υ}
    (h1 : op1.CotangentAdditive) (h2 : op2.CotangentAdditive) :
    (op1 >>> op2).CotangentAdditive := by
  intro x dL1 dL2
  simp only [compose_backward]
  rw [h2 (op1.forward x) dL1 dL2]
  exact h1 x (op2.backward (op1.forward x) dL1) (op2.backward (op1.forward x) dL2)

/-- Operator composition preserves cotangent scalar homogeneity. -/
theorem compose_cotangentHomogeneous {σ τ υ : Shape}
    {op1 : OpSpec R σ τ} {op2 : OpSpec R τ υ}
    (h1 : op1.CotangentHomogeneous) (h2 : op2.CotangentHomogeneous) :
    (op1 >>> op2).CotangentHomogeneous := by
  intro x c dL
  simp only [compose_backward]
  rw [h2 (op1.forward x) c dL]
  exact h1 x c (op2.backward (op1.forward x) dL)

/-- Operator composition preserves cotangent linearity. -/
theorem compose_cotangentLinear {σ τ υ : Shape}
    {op1 : OpSpec R σ τ} {op2 : OpSpec R τ υ}
    (h1 : op1.CotangentLinear) (h2 : op2.CotangentLinear) :
    (op1 >>> op2).CotangentLinear :=
  ⟨compose_cotangentAdditive h1.1 h2.1, compose_cotangentHomogeneous h1.2 h2.2⟩

end OpSpec

/-- Spec contract `opSpec_compose_vjp`: reverse-mode VJP chain rule for operator composition.
This equation holds definitionally (`rfl`) by construction of `OpSpec.compose`. -/
theorem opSpec_compose_vjp {σ τ υ : Shape}
    (op1 : OpSpec R σ τ) (op2 : OpSpec R τ υ)
    (x : Tensor R σ) (dL : Tensor R υ) :
    (op1 >>> op2).backward x dL = op1.backward x (op2.backward (op1.forward x) dL) :=
  rfl

end Nous.Core
