import Apeiron.Combinator.Basic
import Apeiron.Combinator.Confluence

/-!
# Church Numerals and Combinator Primitives

Combinatory logic representations of the composition combinator `B`,
the self-application/duplication combinator `W`, and Church numerals
defined via `church 0 = K I` and `church (n+1) = S B (church n)`.

Proves:
- Reduction specifications for `B` and `W`
- `church_spec`: `church n f x ↠ fⁿ x`
- `church_add`: addition as composition
- `church_mul`: multiplication as composition
- `church_double`: doubling step `2n = n + n`
- `church_separated`: non-vacuity via confluence and normal forms
-/

set_option linter.apeiron.notation false

namespace Nous
namespace Combinator

open Apeiron.Combinator
open Apeiron.Combinator.Comb

/-- The composition combinator `B := S (K S) K`. -/
def B : Comb :=
  app (app S (app K S)) K

/-- Specification of `B`: `B f g x ↠ f (g x)`. -/
theorem B_spec (f g x : Comb) :
    Reduces (app (app (app B f) g) x) (app f (app g x)) := by
  have step1 : Reduces (app (app (app B f) g) x)
                       (app (app (app (app (app K S) f) (app K f)) g) x) := by
    apply Reduces.left x
    apply Reduces.left g
    exact Reduces.single (Step.s (app K S) K f)
  have step2 : Reduces (app (app (app (app (app K S) f) (app K f)) g) x)
                       (app (app (app S (app K f)) g) x) := by
    apply Reduces.left x
    apply Reduces.left g
    apply Reduces.left (app K f)
    exact Reduces.single (Step.k S f)
  have step3 : Reduces (app (app (app S (app K f)) g) x)
                       (app (app (app K f) x) (app g x)) := by
    exact Reduces.single (Step.s (app K f) g x)
  have step4 : Reduces (app (app (app K f) x) (app g x))
                       (app f (app g x)) := by
    apply Reduces.left (app g x)
    exact Reduces.single (Step.k f x)
  exact step1.trans (step2.trans (step3.trans step4))

/-- The duplication combinator `W := S S (K I)`. -/
def W : Comb :=
  app (app S S) (app K I)

/-- Specification of `W`: `W x y ↠ x y y`. -/
theorem W_spec (x y : Comb) :
    Reduces (app (app W x) y) (app (app x y) y) := by
  have step1 : Reduces (app (app W x) y)
                       (app (app (app S x) (app (app K I) x)) y) := by
    apply Reduces.left y
    exact Reduces.single (Step.s S (app K I) x)
  have step2 : Reduces (app (app (app S x) (app (app K I) x)) y)
                       (app (app (app S x) I) y) := by
    apply Reduces.left y
    apply Reduces.right (app S x)
    exact Reduces.single (Step.k I x)
  have step3 : Reduces (app (app (app S x) I) y)
                       (app (app x y) (app I y)) := by
    exact Reduces.single (Step.s x I y)
  have step4 : Reduces (app (app x y) (app I y))
                       (app (app x y) y) := by
    apply Reduces.right (app x y)
    exact I_app y
  exact step1.trans (step2.trans (step3.trans step4))

/-- Iterated application `fⁿ x`: `iterApp f n x = f (... (f x) ...)`. -/
def iterApp (f : Comb) : ℕ → Comb → Comb
  | 0, x => x
  | n + 1, x => app f (iterApp f n x)

@[simp]
theorem iterApp_zero (f x : Comb) : iterApp f 0 x = x := rfl

@[simp]
theorem iterApp_succ (f x : Comb) (n : ℕ) :
    iterApp f (n + 1) x = app f (iterApp f n x) := rfl

/-- Iterated application splits over addition: `iterApp f (m + n) x = iterApp f m (iterApp f n x)`. -/
theorem iterApp_add (f : Comb) (m n : ℕ) (x : Comb) :
    iterApp f (m + n) x = iterApp f m (iterApp f n x) := by
  induction m with
  | zero =>
    rw [Nat.zero_add, iterApp_zero]
  | succ m ih =>
    rw [Nat.succ_add, iterApp_succ, iterApp_succ, ih]

/-- Church numerals: `church 0 = K I`, `church (n+1) = S B (church n)`. -/
def church : ℕ → Comb
  | 0 => app K I
  | n + 1 => app (app S B) (church n)

@[simp]
theorem church_zero : church 0 = app K I := rfl

@[simp]
theorem church_succ (n : ℕ) : church (n + 1) = app (app S B) (church n) := rfl

/-- Headline 1: `church_spec (n) (f x : Comb) : Reduces (app (app (church n) f) x) (iterApp f n x)`. -/
theorem church_spec (n : ℕ) (f x : Comb) :
    Reduces (app (app (church n) f) x) (iterApp f n x) := by
  induction n generalizing x with
  | zero =>
    simp only [church_zero, iterApp_zero]
    have h1 : Reduces (app (app (app K I) f) x) (app I x) := by
      apply Reduces.left x
      exact Reduces.single (Step.k I f)
    exact h1.trans (I_app x)
  | succ n ih =>
    simp only [church_succ, iterApp_succ]
    have h1 : Reduces (app (app (app (app S B) (church n)) f) x)
                      (app (app (app B f) (app (church n) f)) x) := by
      apply Reduces.left x
      exact Reduces.single (Step.s B (church n) f)
    have h2 : Reduces (app (app (app B f) (app (church n) f)) x)
                      (app f (app (app (church n) f) x)) :=
      B_spec f (app (church n) f) x
    have h3 : Reduces (app f (app (app (church n) f) x))
                      (app f (iterApp f n x)) := by
      exact Reduces.right f (ih x)
    exact h1.trans (h2.trans h3)

/-- Congruence for `iterApp` along reduction of its base argument `x`. -/
theorem iterApp_reduces_congr (f : Comb) (n : ℕ) {x y : Comb} (h : Reduces x y) :
    Reduces (iterApp f n x) (iterApp f n y) := by
  induction n with
  | zero =>
    simp only [iterApp_zero]
    exact h
  | succ n ih =>
    simp only [iterApp_succ]
    exact Reduces.right f ih

/-- Headline 2: `church_add (m n f x)`: both `app (app (church (m+n)) f) x`
and `app (app (church m) f) (app (app (church n) f) x)` reduce to `iterApp f (m+n) x`. -/
theorem church_add (m n : ℕ) (f x : Comb) :
    Reduces (app (app (church (m + n)) f) x) (iterApp f (m + n) x) ∧
    Reduces (app (app (church m) f) (app (app (church n) f) x)) (iterApp f (m + n) x) := by
  constructor
  · exact church_spec (m + n) f x
  · rw [iterApp_add f m n x]
    have h1 : Reduces (app (app (church m) f) (app (app (church n) f) x))
                      (iterApp f m (app (app (church n) f) x)) :=
      church_spec m f (app (app (church n) f) x)
    have h2 : Reduces (iterApp f m (app (app (church n) f) x))
                      (iterApp f m (iterApp f n x)) :=
      iterApp_reduces_congr f m (church_spec n f x)
    exact h1.trans h2

/-- Iterated application splits over multiplication: `iterApp (church n f) m x` reduces to `iterApp f (m * n) x`. -/
theorem iterApp_church_mul (m n : ℕ) (f x : Comb) :
    Reduces (iterApp (app (church n) f) m x) (iterApp f (m * n) x) := by
  induction m generalizing x with
  | zero =>
    simp only [Nat.zero_mul, iterApp_zero]
    exact Reduces.refl _
  | succ m ih =>
    simp only [iterApp_succ]
    have h1 : Reduces (app (app (church n) f) (iterApp (app (church n) f) m x))
                      (iterApp f n (iterApp (app (church n) f) m x)) :=
      church_spec n f (iterApp (app (church n) f) m x)
    have h2 : Reduces (iterApp f n (iterApp (app (church n) f) m x))
                      (iterApp f n (iterApp f (m * n) x)) :=
      iterApp_reduces_congr f n (ih x)
    have h3 : Reduces (iterApp f n (iterApp f (m * n) x)) (iterApp f ((m + 1) * n) x) := by
      rw [← iterApp_add f n (m * n) x]
      have heq : n + m * n = (m + 1) * n := by
        rw [Nat.succ_mul, Nat.add_comm]
      rw [heq]
      exact Reduces.refl _
    exact h1.trans (h2.trans h3)

/-- Headline 3: `church_mul (m n f x)`: both `app (app (church (m*n)) f) x`
and `app (app (church m) (app (church n) f)) x` reduce to `iterApp f (m*n) x`. -/
theorem church_mul (m n : ℕ) (f x : Comb) :
    Reduces (app (app (church (m * n)) f) x) (iterApp f (m * n) x) ∧
    Reduces (app (app (church m) (app (church n) f)) x) (iterApp f (m * n) x) := by
  constructor
  · exact church_spec (m * n) f x
  · have h1 : Reduces (app (app (church m) (app (church n) f)) x)
                      (iterApp (app (church n) f) m x) :=
      church_spec m (app (church n) f) x
    have h2 : Reduces (iterApp (app (church n) f) m x)
                      (iterApp f (m * n) x) :=
      iterApp_church_mul m n f x
    exact h1.trans h2

/-- Headline 4: `church_double (n f x)`: the special case 2n = n + n. -/
theorem church_double (n : ℕ) (f x : Comb) :
    Reduces (app (app (church (n + n)) f) x) (iterApp f (n + n) x) ∧
    Reduces (app (app (church n) f) (app (app (church n) f) x)) (iterApp f (n + n) x) :=
  church_add n n f x

-- Normal form lemmas for variables and iterApp (var 0) k (var 1)

/-- Step out of a variable is impossible. -/
theorem no_step_var (i : ℕ) (b : Comb) : ¬ Step (var i) b := by
  intro h; cases h

/-- Step out of an application `var i t` implies a step in `t`. -/
theorem step_var_app_inv {i : ℕ} {t b : Comb} (h : Step (app (var i) t) b) :
    ∃ t', b = app (var i) t' ∧ Step t t' := by
  cases h with
  | left _ hleft => cases hleft
  | right _ hright => exact ⟨_, rfl, hright⟩

/-- `iterApp (var 0) k (var 1)` has no Step. -/
theorem normalForm_iterApp_var (k : ℕ) :
    NormalForm (iterApp (var 0) k (var 1)) := by
  induction k with
  | zero =>
    intro ⟨b, hb⟩
    exact no_step_var 1 b hb
  | succ k ih =>
    intro ⟨b, hb⟩
    simp only [iterApp_succ] at hb
    obtain ⟨t', _, ht'⟩ := step_var_app_inv hb
    exact ih ⟨t', ht'⟩

/-- Injection on `iterApp (var 0)`: different `k` give different terms. -/
theorem iterApp_var_inj {m n : ℕ}
    (h : iterApp (var 0) m (var 1) = iterApp (var 0) n (var 1)) : m = n := by
  induction m generalizing n with
  | zero =>
    cases n with
    | zero => rfl
    | succ n =>
      simp only [iterApp_zero, iterApp_succ] at h
      nomatch h
  | succ m ihm =>
    cases n with
    | zero =>
      simp only [iterApp_succ, iterApp_zero] at h
      nomatch h
    | succ n =>
      simp only [iterApp_succ] at h
      injection h with h1 h2
      rw [ihm h2]

/-- Headline 5: Non-vacuity `church_separated (m n) (h : m ≠ n)`:
with f = var 0 and x = var 1, there is no term both
`app (app (church m) f) x` and `app (app (church n) f) x` reduce to. -/
theorem church_separated (m n : ℕ) (h : m ≠ n) :
    ¬ ∃ d, Reduces (app (app (church m) (var 0)) (var 1)) d ∧
           Reduces (app (app (church n) (var 0)) (var 1)) d := by
  intro ⟨d, hm, hn⟩
  have hm_spec : Reduces (app (app (church m) (var 0)) (var 1)) (iterApp (var 0) m (var 1)) :=
    church_spec m (var 0) (var 1)
  have hn_spec : Reduces (app (app (church n) (var 0)) (var 1)) (iterApp (var 0) n (var 1)) :=
    church_spec n (var 0) (var 1)
  obtain ⟨dm, hmd, hddm⟩ := confluence hm_spec hm
  have hdm_eq : iterApp (var 0) m (var 1) = dm :=
    eq_of_reduces_normalForm (normalForm_iterApp_var m) hmd
  obtain ⟨dn, hnd, hddn⟩ := confluence hn_spec hn
  have hdn_eq : iterApp (var 0) n (var 1) = dn :=
    eq_of_reduces_normalForm (normalForm_iterApp_var n) hnd
  obtain ⟨final, hfinal_m, hfinal_n⟩ := confluence hddm hddn
  rw [← hdm_eq] at hfinal_m
  rw [← hdn_eq] at hfinal_n
  have hf_m : iterApp (var 0) m (var 1) = final :=
    eq_of_reduces_normalForm (normalForm_iterApp_var m) hfinal_m
  have hf_n : iterApp (var 0) n (var 1) = final :=
    eq_of_reduces_normalForm (normalForm_iterApp_var n) hfinal_n
  have h_eq : iterApp (var 0) m (var 1) = iterApp (var 0) n (var 1) := by
    rw [hf_m, ← hf_n]
  exact h (iterApp_var_inj h_eq)

namespace Witness

/-- Small reduction example: `church 2` applied to `var 0` and `var 1` reduces
to `app (var 0) (app (var 0) (var 1))`. -/
example : Reduces (app (app (church 2) (var 0)) (var 1))
                  (app (var 0) (app (var 0) (var 1))) :=
  church_spec 2 (var 0) (var 1)

#guard (iterApp (var 0) 2 (var 1) == app (var 0) (app (var 0) (var 1)))

end Witness

end Combinator
end Nous
