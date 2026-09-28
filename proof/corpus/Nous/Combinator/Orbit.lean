/-
# RoPE Orbits and Church Numerals on Closed Rings

The connection between untyped combinatory logic (Church numerals), rotary position embeddings
(RoPE), and finite rings:
- A first-order reading `readFO` of application-only terms interprets `iterApp (var 0) n (var 1)`
  as `g^[n] x0`.
- The RoPE rotor orbit under `rhoMul (rho θ)` is denoted by Church numerals:
  `readFO (rhoMul (rho θ)) x0 (iterApp (var 0) n (var 1)) = some (rhoMul (rho (n · θ)) x0)`.
- On any closed ring where `g^[N] = id`, numerals `n` and `n + N` evaluate identically
  (`church_ring_mod`), whereas freely they remain distinct combinator terms (`church_ring_distinct`).
- The turn ring instance `g_f(φ) = (φ + f) % 2^64` satisfies `g_f^[p] 0 = Numeric.RopeTurns.phase f p`
  and has period `2^64` on reduced phases, establishing `church_turn_mod`.
  This is the combinator statement of L9: the RoPE phase at position p is the Church numeral p
  read on the closed ring Z/2^64.

## Trust boundary

The statements are about algebraic definitions, pure combinator reduction, and integer arithmetic.
They do not reach `rustc`/LLVM, the kernel ABI, or GPU codegen.
-/

import Apeiron.Combinator.Basic
import Apeiron.Combinator.Confluence
import Nous.Combinator.Church
import Nous.Model.RopeOrbit
import Numeric.RopeTurns

set_option linter.apeiron.notation false

namespace Nous
namespace Combinator
namespace Orbit

open Apeiron.Combinator
open Apeiron.Combinator.Comb
open Nous.Combinator
open Nous.Model

variable {X : Type*}

/-- A first-order reading of application-only terms: interpreting `var 0` as the function
symbol and `var 1` as the base point, `app (var 0) t` ↦ g applied to the reading of t;
anything else ↦ none. -/
def readFO (g : X → X) (x0 : X) : Comb → Option X
  | var 1 => some x0
  | app (var 0) t => (readFO g x0 t).map g
  | _ => none

@[simp]
theorem readFO_base (g : X → X) (x0 : X) : readFO g x0 (var 1) = some x0 := rfl

@[simp]
theorem readFO_step (g : X → X) (x0 : X) (t : Comb) :
    readFO g x0 (app (var 0) t) = (readFO g x0 t).map g := rfl

/-- Evaluating `iterApp (var 0) n (var 1)` under `readFO` yields `g^[n] x0`. -/
theorem readFO_iterApp (g : X → X) (x0 : X) (n : ℕ) :
    readFO g x0 (iterApp (var 0) n (var 1)) = some (g^[n] x0) := by
  induction n with
  | zero =>
    simp only [iterApp_zero, readFO_base, Function.iterate_zero, id_eq]
  | succ n ih =>
    simp only [iterApp_succ, readFO_step, ih, Option.map_some, Function.iterate_succ_apply']

/-- Commutativity of `rhoMul`. -/
theorem rhoMul_comm (a b : ℝ × ℝ) : rhoMul a b = rhoMul b a := by
  apply Prod.ext <;> simp only [rhoMul] <;> ring

/-- Iterating `rhoMul (rho θ)` n times corresponds to multiplication by `rhoPow n (rho θ)`. -/
theorem rhoMul_pow_iterate (θ : ℝ) (x0 : ℝ × ℝ) (n : ℕ) :
    ((rhoMul (rho θ))^[n] x0) = rhoMul (rhoPow n (rho θ)) x0 := by
  induction n generalizing x0 with
  | zero =>
    simp only [Function.iterate_zero, id_eq, rhoPow]
    simp only [rhoMul]
    ring_nf
  | succ n ih =>
    rw [Function.iterate_succ_apply', ih, rhoPow_succ]
    rw [rhoMul_comm (rhoPow n (rho θ)) (rho θ)]
    exact (rhoMul_assoc (rho θ) (rhoPow n (rho θ)) x0).symm

/-- Iterating `rhoMul (rho θ)` n times corresponds to rotation by `n · θ`. -/
theorem rhoMul_iterate (θ : ℝ) (x0 : ℝ × ℝ) (n : ℕ) :
    ((rhoMul (rho θ))^[n] x0) = rhoMul (rho ((n : ℝ) * θ)) x0 := by
  rw [rho_nat, rhoMul_pow_iterate]

/-- Headline 2: The unique normal form of `church n (var 0) (var 1)` reads under
`g = rhoMul (rho θ)` as `rho (n · θ)` applied to `x0`, combined with the Church reduction. -/
theorem church_reads_orbit (n : ℕ) (θ : ℝ) (x0 : ℝ × ℝ) :
    readFO (rhoMul (rho θ)) x0 (iterApp (var 0) n (var 1)) =
      some (rhoMul (rho ((n : ℝ) * θ)) x0) ∧
    Reduces (app (app (church n) (var 0)) (var 1)) (iterApp (var 0) n (var 1)) := by
  constructor
  · rw [readFO_iterApp, rhoMul_iterate]
  · exact church_spec n (var 0) (var 1)

/-- Closed ring: `g^[N] = id` implies `g^[n + N] = g^[n]`. -/
theorem ring_period (g : X → X) (N : ℕ) (hN : g^[N] = id) (n : ℕ) :
    g^[n + N] = g^[n] := by
  ext x
  rw [Function.iterate_add_apply, hN, id_eq]

/-- Multiple periods collapse to identity. -/
theorem iterate_mul_period (g : X → X) (N : ℕ) (hN : g^[N] = id) (k : ℕ) :
    g^[k * N] = id := by
  induction k with
  | zero =>
    simp only [Nat.zero_mul, Function.iterate_zero]
  | succ k ih =>
    have heq : (k + 1) * N = k * N + N := by ring
    rw [heq, Function.iterate_add, ih, hN]
    rfl

/-- Iterates of a periodic endomorphism collapse modulo the period `N`. -/
theorem iterate_mod_period (g : X → X) (N : ℕ) (hN : g^[N] = id) (n : ℕ) :
    g^[n] = g^[n % N] := by
  by_cases hN0 : N = 0
  · subst hN0
    rw [Nat.mod_zero]
  · nth_rw 1 [← Nat.div_add_mod n N]
    ext x
    rw [Function.iterate_add_apply]
    have hmul : g^[N * (n / N)] = id := by
      rw [mul_comm]
      exact iterate_mul_period g N hN (n / N)
    rw [hmul, id_eq]

/-- Headline 3: Numerals are read mod N on a closed ring. -/
theorem church_ring_mod (g : X → X) (x0 : X) (N : ℕ) (hN : g^[N] = id) (n : ℕ) :
    readFO g x0 (iterApp (var 0) n (var 1)) =
      readFO g x0 (iterApp (var 0) (n % N) (var 1)) := by
  rw [readFO_iterApp, readFO_iterApp, iterate_mod_period g N hN n]

/-- Headline 3 (distinctness): On the free reading (via `church_separated`), Church numerals `n`
and `n + N` are distinct combinator terms whenever `N > 0`; only the ring reading identifies them. -/
theorem church_ring_distinct (n N : ℕ) (hN : 0 < N) :
    (¬ ∃ d, Reduces (app (app (church n) (var 0)) (var 1)) d ∧
            Reduces (app (app (church (n + N)) (var 0)) (var 1)) d) ∧
    church n ≠ church (n + N) := by
  have hne : n ≠ n + N := by omega
  refine ⟨church_separated n (n + N) hne, ?_⟩
  intro h_eq
  have hconfl : ∃ d, Reduces (app (app (church n) (var 0)) (var 1)) d ∧
                     Reduces (app (app (church (n + N)) (var 0)) (var 1)) d := by
    rw [h_eq]
    exact ⟨app (app (church (n + N)) (var 0)) (var 1), Reduces.refl _, Reduces.refl _⟩
  exact (church_separated n (n + N) hne) hconfl

/-- The step map of the turn ring `Z/2^64`: `g_f(φ) = (φ + f) % 2^64`. -/
def g_f (f : ℕ) (φ : ℕ) : ℕ := (φ + f) % 2 ^ 64

/-- Headline 4: Iterating `g_f` from 0 produces `Numeric.RopeTurns.phase f p`. -/
theorem turn_iter (f p : ℕ) : (g_f f)^[p] 0 = Numeric.RopeTurns.phase f p := by
  induction p with
  | zero =>
    simp only [Function.iterate_zero, id_eq, Numeric.RopeTurns.phase, Nat.mul_zero, Nat.zero_mod]
  | succ p ih =>
    rw [Function.iterate_succ_apply', ih]
    unfold g_f
    have hphase1 : f % 2 ^ 64 = Numeric.RopeTurns.phase f 1 := by
      unfold Numeric.RopeTurns.phase
      rw [Nat.mul_one]
    rw [show (Numeric.RopeTurns.phase f p + f) % 2 ^ 64 =
             (Numeric.RopeTurns.phase f p + f % 2 ^ 64) % 2 ^ 64 by omega]
    rw [hphase1]
    rw [← Numeric.RopeTurns.phase_add]

/-- General iteration of `g_f` from an initial phase. -/
theorem turn_iter_general (f : ℕ) (φ : ℕ) (p : ℕ) :
    (g_f f)^[p] (φ % 2 ^ 64) = (φ + Numeric.RopeTurns.phase f p) % 2 ^ 64 := by
  induction p with
  | zero =>
    simp only [Function.iterate_zero, id_eq, Numeric.RopeTurns.phase, Nat.mul_zero]
    omega
  | succ p ih =>
    rw [Function.iterate_succ_apply', ih]
    unfold g_f
    have hphase_step := Numeric.RopeTurns.phase_add f p 1
    unfold Numeric.RopeTurns.phase at hphase_step ⊢
    omega

/-- Headline 4: `g_f` has period `2^64` on reduced phases (`φ < 2^64`). -/
theorem turn_period (f : ℕ) (φ : ℕ) (hφ : φ < 2 ^ 64) :
    (g_f f)^[2 ^ 64] φ = φ := by
  have hφ_mod : φ = φ % 2 ^ 64 := (Nat.mod_eq_of_lt hφ).symm
  nth_rw 1 [hφ_mod]
  rw [turn_iter_general]
  unfold Numeric.RopeTurns.phase
  have hzero : (f * 2 ^ 64) % 2 ^ 64 = 0 := by
    rw [mul_comm]
    exact Nat.mul_mod_right (2 ^ 64) f
  rw [hzero, add_zero, Nat.mod_eq_of_lt hφ]

/-- Phase is periodic with period `2^64` in the position argument `p`. -/
theorem phase_mod (f p : ℕ) :
    Numeric.RopeTurns.phase f p = Numeric.RopeTurns.phase f (p % 2 ^ 64) := by
  unfold Numeric.RopeTurns.phase
  have hdiv := Nat.div_add_mod p (2 ^ 64)
  conv_lhs => rw [← hdiv]
  rw [Nat.mul_add]
  have hzero : (f * (2 ^ 64 * (p / 2 ^ 64))) % 2 ^ 64 = 0 := by
    rw [show f * (2 ^ 64 * (p / 2 ^ 64)) = 2 ^ 64 * (f * (p / 2 ^ 64)) by ring]
    exact Nat.mul_mod_right (2 ^ 64) _
  rw [Nat.add_mod, hzero, Nat.zero_add, Nat.mod_mod]

/-- Iterating `g_f` on reduced phases is invariant modulo `2^64`. -/
theorem turn_iter_mod (f : ℕ) (φ : ℕ) (p : ℕ) :
    (g_f f)^[p] (φ % 2 ^ 64) = (g_f f)^[p % 2 ^ 64] (φ % 2 ^ 64) := by
  rw [turn_iter_general, turn_iter_general, phase_mod]

/-- Headline 4: `church_turn_mod`: Church numerals evaluate modulo `2^64` on the turn ring.
This is the combinator statement of L9: the RoPE phase at position p is the Church numeral p
read on the closed ring Z/2^64. -/
theorem church_turn_mod (f : ℕ) (φ : ℕ) (p : ℕ) (hφ : φ < 2 ^ 64) :
    readFO (g_f f) φ (iterApp (var 0) p (var 1)) =
      readFO (g_f f) φ (iterApp (var 0) (p % 2 ^ 64) (var 1)) := by
  rw [readFO_iterApp, readFO_iterApp]
  have hφ_mod : φ = φ % 2 ^ 64 := (Nat.mod_eq_of_lt hφ).symm
  nth_rw 1 [hφ_mod]
  nth_rw 2 [hφ_mod]
  rw [turn_iter_mod]

section Witness

/-- A ring of period 5 with generator `g(x) = (x + 2) % 5`.
Readings of Church numerals 3 and 8 agree on this ring. -/
def g_mod5 (x : ℕ) : ℕ := (x + 2) % 5

#guard (readFO g_mod5 0 (iterApp (var 0) 3 (var 1)) == some 1)
#guard (readFO g_mod5 0 (iterApp (var 0) 8 (var 1)) == some 1)
#guard (readFO g_mod5 0 (iterApp (var 0) 3 (var 1)) == readFO g_mod5 0 (iterApp (var 0) 8 (var 1)))

example : readFO g_mod5 0 (iterApp (var 0) 3 (var 1)) =
          readFO g_mod5 0 (iterApp (var 0) 8 (var 1)) := by
  decide

example : readFO (g_f 7) 0 (iterApp (var 0) 3 (var 1)) = some 21 := by
  decide

end Witness

end Orbit
end Combinator
end Nous
