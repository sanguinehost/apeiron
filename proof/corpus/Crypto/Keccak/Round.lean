/-
Keccak-f[1600] step functions and the 24-round permutation over `Vector (BitVec 64) 25`.

FIPS PUB 202, §3.2: the state is `A[x, y]` with lane index `x + 5 * y`, `x, y : Fin 5`; the round
is `ι ∘ χ ∘ π ∘ ρ ∘ θ`. Rotation offsets and round constants live in `Crypto.Keccak.Constants`.
-/

import Crypto.Keccak.Constants

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.Keccak

/-- One lane of the Keccak-f[1600] state. -/
private abbrev Lane := BitVec 64

/-- The Keccak-f[1600] state, 25 lanes indexed `x + 5 * y`. -/
private abbrev State := Vector Lane 25

/-- Lane at index `k` (reduced mod 25), for FIPS index arithmetic in `Nat`. -/
private def lane (s : State) (k : Nat) : Lane :=
  s.get ⟨k % 25, Nat.mod_lt k (by decide)⟩

/-- The θ column parity `C[x] = A[x,0] ⊕ A[x,1] ⊕ A[x,2] ⊕ A[x,3] ⊕ A[x,4]`. -/
private def column (s : State) (x : Nat) : Lane :=
  lane s x ^^^ lane s (x + 5) ^^^ lane s (x + 10) ^^^ lane s (x + 15) ^^^ lane s (x + 20)

/-- FIPS-202 θ: `A′[x,y] = A[x,y] ⊕ D[x]`,
`D[x] = C[(x+4) mod 5] ⊕ rotl(C[(x+1) mod 5], 1)`. -/
def theta (s : State) : State :=
  Vector.ofFn fun i =>
    let x := i.val % 5
    s.get i ^^^ (column s ((x + 4) % 5) ^^^ (column s ((x + 1) % 5)).rotateLeft 1)

/-- FIPS-202 ρ: `A′[x,y] = rotl(A[x,y], ROT[x + 5y])`. -/
def rho (s : State) : State :=
  Vector.ofFn fun i => (s.get i).rotateLeft (ROT.get i)

/-- FIPS-202 π: `A′[x,y] = A[(x + 3y) mod 5, x]`. -/
def pi (s : State) : State :=
  Vector.ofFn fun i =>
    let x := i.val % 5
    let y := i.val / 5
    lane s (((x + 3 * y) % 5) + 5 * x)

/-- FIPS-202 χ: `A′[x,y] = A[x,y] ⊕ (¬A[(x+1) mod 5, y] ∧ A[(x+2) mod 5, y])`. -/
def chi (s : State) : State :=
  Vector.ofFn fun i =>
    let x := i.val % 5
    let y := i.val / 5
    s.get i ^^^
      (~~~ (lane s (((x + 1) % 5) + 5 * y)) &&& lane s (((x + 2) % 5) + 5 * y))

/-- FIPS-202 ι for round `n`: lane 0 is XORed with `RC[n]`. -/
def iota (n : Fin 24) (s : State) : State :=
  s.set 0 (s.get ⟨0, by decide⟩ ^^^ RC.get n)

/-- The Keccak-f[1600] round `n`: `ι[n] ∘ χ ∘ π ∘ ρ ∘ θ`. -/
def round (n : Fin 24) (s : State) : State :=
  iota n (chi (pi (rho (theta s))))

/-- Keccak-f[1600]: the 24 rounds `0, …, 23` folded left over the state. -/
def permute (s : State) : State :=
  (List.finRange 24).foldl (fun acc n => round n acc) s

/-- `permute` is the left fold of `round` over `0..23` (definitional). -/
theorem permute_eq_foldl_round (s : State) :
    permute s = (List.finRange 24).foldl (fun acc n => round n acc) s := rfl

end Apeiron.Crypto.Keccak
