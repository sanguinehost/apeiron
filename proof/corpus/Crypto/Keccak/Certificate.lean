/-
  Crypto/Keccak/Certificate.lean

  Refinement certificate for FIPS-202 Keccak-f[1600]: the extracted
  `apeiron_word.word_keccakf` computes `Apeiron.Crypto.Keccak.permute` on the
  canonical 25-lane state.

  `proof/ApeironWord.lean` carries the extraction (`crypto.keccak_f1600`,
  `word_keccakf`); `Crypto/Keccak/Round.lean` carries the canonical permutation
  over `Vector (BitVec 64) 25`. The two objects live at different types, so the
  certificate is stated through the one-way state conversion `view` below:
  `view (final state) = permute (view (initial state))`. No injectivity of
  `view` is needed for that statement; `Array.ext` is reserved for the internal
  loop invariants.

  Contents (this file), each against the canonical step:
  * the state conversion `view` and its read/write lemmas;
  * the extracted constant-table bridges `crypto.ROT` and `crypto.RC`;
  * the theta column loop: `laneW`/`colW`/`colGet`, the per-iteration
    `column_body_step` and the loop invariant `columns_spec` over
    `loop.spec_decr_nat`;
  * the theta apply loop (`thetaInv`, `thetaRowInv`, `thetaRow_body_step`,
    `thetaRow_spec`, `thetaApply_body_step`, `thetaApply_spec`);
  * the rho-pi sweep (`rhoPi_body_step`, `rhoPi_row_spec`, `rhoPi_spec`) and the
    chi sweep (`chi_body_step`, `chi_row_spec`, `chi_spec`);
  * the iota read-back `iota_write`;
  * one outer iteration against `round` (`RoundStepDenotes`), the outer loop
    against the prefix fold `foldRounds`/`permute` (`keccak_f1600_loop0_spec`),
    and the refinement headline `WordKeccakfDenotes`.

  `maxHeartbeats` is a finite bound (not `0`) for the extracted outer-loop body
  reductions; the `#audit` registration is in `Crypto/Keccak/Audit.lean`.
-/

import Crypto.Keccak.Round
import ApeironWord

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option maxHeartbeats 1500000

namespace apeiron_word

open Aeneas Aeneas.Std Result
open Apeiron.Crypto.Keccak (RC ROT theta rho pi chi iota round permute lane)

/-! ## State conversion -/

/-- The extracted 25-lane array read as the canonical state: lane `i` is the
    bit vector behind the `u64` word. One bridge direction; the reverse map is
    never needed for the refinement statement. -/
def view (a : Array Std.U64 25#usize) : Vector (BitVec 64) 25 :=
  Vector.ofFn fun i : Fin 25 => (a[i.val]).bv

/-- Reading a lane through `view` is reading the word's bit vector. -/
theorem view_get (a : Array Std.U64 25#usize) (i : Fin 25) :
    (view a).get i = (a[i.val]).bv := by
  show (view a)[i.val] = (a[i.val]).bv
  simp only [view, Vector.getElem_ofFn]

/-- Reading `view` after a write: the written lane is the written word's bit
    vector, every other lane is unchanged. -/
theorem view_set_get (a : Array Std.U64 25#usize) (i : Std.Usize) (v : Std.U64)
    (hi : i.val < 25) (j : Fin 25) :
    (view (a.set i v)).get j = if j.val = i.val then v.bv else (view a).get j := by
  show (view (a.set i v))[j.val] = if j.val = i.val then v.bv else (view a)[j.val]
  simp only [view, Vector.getElem_ofFn]
  have hb : j.val < (a.set i v).val.length := by
    rw [Array.set_val_eq, List.length_set]
    have ha : a.val.length = 25 := by simpa using a.property
    omega
  by_cases h : j.val = i.val
  · rw [if_pos h]
    exact congrArg Std.U64.bv
      (Array.getElem_Nat_set_eq a i j.val v hb h.symm)
  · rw [if_neg h]
    exact congrArg Std.U64.bv
      (Array.getElem_Nat_set_ne a i j.val v hb (by omega))

/-- The write lemma for `view`, as a single vector update. -/
theorem view_set (a : Array Std.U64 25#usize) (i : Std.Usize) (v : Std.U64)
    (hi : i.val < 25) :
    view (a.set i v) = (view a).set i.val v.bv hi := by
  apply Vector.ext
  intro j hj
  rw [Vector.getElem_set,
    show (view (a.set i v))[j] = (view (a.set i v)).get ⟨j, hj⟩ from rfl,
    view_set_get a i v hi ⟨j, hj⟩]
  by_cases h : j = i.val
  · rw [if_pos h, if_pos h.symm]
  · rw [if_neg h, if_neg (by omega : ¬ i.val = j)]
    rfl

/-! ## Canonical view bridges

`theta` is stated on `Vector (BitVec 64) 25` with its own lane and column
helpers. These mirror the private helpers of `Crypto/Keccak/Round.lean` so the
extracted theta loop can be read through `view`. -/

/-- Lane `k mod 25` of the canonical state, as `theta` reads it. -/
private def laneV (s : Vector (BitVec 64) 25) (k : Nat) : BitVec 64 :=
  s.get ⟨k % 25, Nat.mod_lt k (by decide)⟩

/-- The canonical theta column parity `C[k]`. -/
private def columnV (s : Vector (BitVec 64) 25) (k : Nat) : BitVec 64 :=
  laneV s k ^^^ laneV s (k + 5) ^^^ laneV s (k + 10) ^^^ laneV s (k + 15) ^^^
    laneV s (k + 20)

/-- The lane `theta` writes: `A[i] ⊕ D[i mod 5]`. -/
theorem theta_get (s : Vector (BitVec 64) 25) (i : Fin 25) :
    (theta s).get i =
      s.get i ^^^
        (columnV s ((i.val % 5 + 4) % 5) ^^^
          (columnV s ((i.val % 5 + 1) % 5)).rotateLeft 1) := by
  show (theta s)[i.val] =
    s.get i ^^^
      (columnV s ((i.val % 5 + 4) % 5) ^^^
        (columnV s ((i.val % 5 + 1) % 5)).rotateLeft 1)
  unfold theta
  simp only [Vector.getElem_ofFn]
  rfl

/-! ## Constant-table bridges -/

/-- `crypto.ROT` read at index `i` is the canonical rho offset `ROT[i]`. -/
theorem ROT_get (i : Fin 25) : (crypto.ROT[i.val]).val = ROT.get i := by
  fin_cases i <;> (simp only [crypto.ROT, Array.make_val]; decide)

/-- `crypto.RC` read at index `n` is the canonical round constant `RC[n]`. -/
theorem RC_get (n : Fin 24) : (crypto.RC[n.val]).bv = RC.get n := by
  fin_cases n <;> (simp only [crypto.RC, Array.make_val]; decide)

/-! ## Theta column loop

`src/crypto.rs:77-80` accumulates `c[x] = state[x] ^ state[x+5] ^ state[x+10] ^
state[x+15] ^ state[x+20]` for `x = 0..4` into the 5-word scratch array `c`.
`column_body_step` is the per-iteration step, `columns_spec` the loop invariant
that the finished `c` holds the five theta column parities. -/

/-- In-bounds `Array.index_usize` reads the underlying list. -/
theorem array_index_usize_eq {α : Type} {n : Std.Usize} (a : Array α n) (i : Std.Usize)
    (h : i.val < a.val.length) : Array.index_usize a i = ok (a.val[i.val]'h) := by
  obtain ⟨x, hx, hxv⟩ := WP.spec_imp_exists (Array.index_usize_spec a i h)
  rw [hxv] at hx
  exact hx

/-- Lane `k mod 25` of an extracted state as a word. -/
def laneW (state : Array Std.U64 25#usize) (k : Nat) : Std.U64 :=
  state[k % 25]'(by
    have h : state.val.length = 25 := by simpa using state.property
    omega)

/-- The theta column parity word `C[j]` of FIPS-202 §3.2.1 over the extracted state. -/
def colW (state : Array Std.U64 25#usize) (j : Nat) : Std.U64 :=
  laneW state j ^^^ laneW state (j + 5) ^^^ laneW state (j + 10) ^^^
    laneW state (j + 15) ^^^ laneW state (j + 20)

/-- The theta correction `D[x]` computed from the extracted column-parity words. -/
def thetaD (state : Array Std.U64 25#usize) (x : Nat) : BitVec 64 :=
  (colW state ((x + 4) % 5)).bv ^^^ ((colW state ((x + 1) % 5)).bv).rotateLeft 1

/-- The extracted column-parity word carries the canonical column parity. -/
theorem colW_bv (state : Array Std.U64 25#usize) (k : Nat) :
    (colW state k).bv = columnV (view state) k := by
  unfold colW laneW columnV laneV
  simp only [UScalar.bv_xor, view_get]

/-- Entry `j` of the extracted 5-word column scratch array. -/
def colGet (c : Array Std.U64 5#usize) (j : Fin 5) : Std.U64 :=
  c[j.val]'(by
    have h : c.val.length = 5 := by simpa using c.property
    omega)

/-- Loop invariant of the theta column accumulation: the first `x` entries of `c`
    hold the column parities, the rest are still zero. -/
def colInv (state : Array Std.U64 25#usize) (c : Array Std.U64 5#usize)
    (x : Std.Usize) : Prop :=
  x.val ≤ 5 ∧ ∀ j : Fin 5,
    (colGet c j).bv = if j.val < x.val then (colW state j.val).bv else 0

/-- One theta column iteration: for `x < 5`, the body stores `C[x]` at `c[x]`,
    increments `x`, and leaves every other entry alone. -/
theorem column_body_step (state : Array Std.U64 25#usize) (c : Array Std.U64 5#usize)
    (x : Std.Usize) (hx : x.val < 5)
    (hc : ∀ j : Fin 5,
      (colGet c j).bv = if j.val < x.val then (colW state j.val).bv else 0) :
    ∃ c' x', crypto.keccak_f1600_loop0_loop0.body state c x
        = ok (ControlFlow.cont (c', x')) ∧
      x'.val = x.val + 1 ∧ c' = c.set x (colW state x.val) := by
  have hs : state.val.length = 25 := by simpa using state.property
  have hc5 : c.val.length = 5 := by simpa using c.property
  have hx25 : x.val < state.val.length := by omega
  have h5 : x.val + 5 < state.val.length := by omega
  have h10 : x.val + 10 < state.val.length := by omega
  have h15 : x.val + 15 < state.val.length := by omega
  have h20 : x.val + 20 < state.val.length := by omega
  have hlt : x < 5#usize := (UScalar.lt_equiv x 5#usize).mpr hx
  have hmax5 : x.val + (5#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨x5, hx5, hx5v⟩ := WP.spec_imp_exists (Usize.add_spec (x := x) (y := 5#usize) hmax5)
  have hx5val : x5.val = x.val + 5 := by rw [hx5v]; rfl
  have h5' : x5.val < state.val.length := by omega
  have hmax10 : x.val + (10#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨x10, hx10, hx10v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := x) (y := 10#usize) hmax10)
  have hx10val : x10.val = x.val + 10 := by rw [hx10v]; rfl
  have h10' : x10.val < state.val.length := by omega
  have hmax15 : x.val + (15#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨x15, hx15, hx15v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := x) (y := 15#usize) hmax15)
  have hx15val : x15.val = x.val + 15 := by rw [hx15v]; rfl
  have h15' : x15.val < state.val.length := by omega
  have hmax20 : x.val + (20#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨x20, hx20, hx20v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := x) (y := 20#usize) hmax20)
  have hx20val : x20.val = x.val + 20 := by rw [hx20v]; rfl
  have h20' : x20.val < state.val.length := by omega
  have hidx0 : Array.index_usize state x = ok (state.val[x.val]'hx25) :=
    array_index_usize_eq state x hx25
  have hidx5 : Array.index_usize state x5 = ok (state.val[x5.val]'h5') :=
    array_index_usize_eq state x5 h5'
  have hidx10 : Array.index_usize state x10 = ok (state.val[x10.val]'h10') :=
    array_index_usize_eq state x10 h10'
  have hidx15 : Array.index_usize state x15 = ok (state.val[x15.val]'h15') :=
    array_index_usize_eq state x15 h15'
  have hidx20 : Array.index_usize state x20 = ok (state.val[x20.val]'h20') :=
    array_index_usize_eq state x20 h20'
  have hmax1 : x.val + (1#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨x1, hx1, hx1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := x) (y := 1#usize) hmax1)
  have hx1val : x1.val = x.val + 1 := by rw [hx1v]; rfl
  have hxc : x.val < c.val.length := by omega
  obtain ⟨c1, hc1, hc1eq⟩ := WP.spec_imp_exists (Array.update_spec c x
    (state.val[x.val]'hx25 ^^^ state.val[x5.val]'h5' ^^^ state.val[x10.val]'h10' ^^^
      state.val[x15.val]'h15' ^^^ state.val[x20.val]'h20') hxc)
  have hc1v : c1 = c.set x
      (state.val[x.val]'hx25 ^^^ state.val[x5.val]'h5' ^^^ state.val[x10.val]'h10' ^^^
        state.val[x15.val]'h15' ^^^ state.val[x20.val]'h20') := hc1eq
  refine ⟨c1, x1, ?_, hx1val, ?_⟩
  · unfold crypto.keccak_f1600_loop0_loop0.body
    simp only [lift, bind_tc_ok]
    rw [if_pos hlt, hidx0]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hx5]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hidx5]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hx10]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hidx10]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hx15]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hidx15]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hx20]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hidx20]
    simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hc1]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hx1]
    simp only [bind_tc_ok]
    try dsimp only
  · rw [hc1v]
    congr 1
    unfold colW laneW
    simp only [Nat.mod_eq_of_lt (by omega : x.val < 25),
      Nat.mod_eq_of_lt (by omega : x.val + 5 < 25),
      Nat.mod_eq_of_lt (by omega : x.val + 10 < 25),
      Nat.mod_eq_of_lt (by omega : x.val + 15 < 25),
      Nat.mod_eq_of_lt (by omega : x.val + 20 < 25),
      hx5val, hx10val, hx15val, hx20val]
    rfl

/-- Reading the column scratch through a write. -/
theorem colGet_set (c : Array Std.U64 5#usize) (i : Std.Usize) (v : Std.U64)
    (hi : i.val < 5) (j : Fin 5) :
    colGet (c.set i v) j = if j.val = i.val then v else colGet c j := by
  have hc5 : c.val.length = 5 := by simpa using c.property
  have hb : j.val < (c.set i v).val.length := by
    rw [Array.set_val_eq, List.length_set, hc5]
    exact j.isLt
  unfold colGet
  by_cases h : j.val = i.val
  · rw [if_pos h]
    exact Array.getElem_Nat_set_eq c i j.val v (by simpa using hb) h.symm
  · rw [if_neg h]
    exact Array.getElem_Nat_set_ne c i j.val v (by simpa using hb) (by omega)

/-- The theta column loop of FIPS-202 §3.2.1: starting from `x ≤ 5` with the
    first `x` columns accumulated, the extracted loop finishes all five. -/
theorem columns_spec (state : Array Std.U64 25#usize) (c : Array Std.U64 5#usize)
    (x : Std.Usize) (hx : x.val ≤ 5)
    (hc : ∀ j : Fin 5,
      (colGet c j).bv = if j.val < x.val then (colW state j.val).bv else 0) :
    crypto.keccak_f1600_loop0_loop0 state c x ⦃ c' =>
      ∀ j : Fin 5, (colGet c' j).bv = (colW state j.val).bv ⦄ := by
  unfold crypto.keccak_f1600_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array Std.U64 5#usize × Std.Usize => 5 - s.2.val)
    (inv := fun s => colInv state s.1 s.2)
    (post := fun c' => ∀ j : Fin 5, (colGet c' j).bv = (colW state j.val).bv)
  · intro s hs
    rcases s with ⟨c1, x1⟩
    dsimp only at hs ⊢
    obtain ⟨hx1, hc1⟩ := hs
    by_cases hlt : x1.val < 5
    · obtain ⟨c2, x2, hbody, hx2v, hc2eq⟩ := column_body_step state c1 x1 hlt hc1
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      refine ⟨⟨by omega, ?_⟩, by omega⟩
      intro j
      rw [hc2eq, colGet_set c1 x1 (colW state x1.val) hlt j]
      by_cases hje : j.val = x1.val
      · rw [if_pos hje, if_pos (by omega : j.val < x2.val), hje]
      · rw [if_neg hje, hc1 j]
        by_cases hltj : j.val < x1.val
        · rw [if_pos hltj, if_pos (by omega : j.val < x2.val)]
        · rw [if_neg hltj, if_neg (by omega : ¬ j.val < x2.val)]
    · have hx5 : x1.val = 5 := by omega
      have hnot : ¬ (x1 < 5#usize) := by
        intro hc'
        exact hlt ((UScalar.lt_equiv x1 5#usize).mp hc')
      have hbody : crypto.keccak_f1600_loop0_loop0.body state c1 x1 =
          ok (ControlFlow.done c1) := by
        unfold crypto.keccak_f1600_loop0_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      intro j
      rw [hc1 j, if_pos (by omega : j.val < x1.val)]
  · exact ⟨hx, hc⟩

/-! ## Rho-pi and chi lanes

`src/crypto.rs:93-113` runs two sweeps over the 5x5 state. The rho-pi sweep
writes, for every pair `(x, y)` with `x, y < 5`,
`b[y + 5 * ((2 * x + 3 * y) % 5)] = state[x + 5 * y].rotate_left(ROT[x + 5 * y])`;
the chi sweep writes `state[x + 5 * y] = b[x + 5 * y] ^ ((!b[(x+1) mod 5 + 5y]) &
b[(x+2) mod 5 + 5y])`, reading only the rho-pi output.

`rhoPiDone` / `chiDone` record which lanes a partial sweep has written,
`rhoPiInv` / `chiInv` are the loop invariants over `view`, and the four `*_spec`
theorems close the two sweeps against `pi (rho s)` and `chi s`. -/

/-- The destination lane, in the state's own indexing, of the rho-pi pair
    `(x, y)`. -/
def rhoPiWrite (x y : Nat) : Nat :=
  y + 5 * ((2 * x + 3 * y) % 5)

/-- Lanes the rho-pi sweep has written after `y` complete rows and `x` lanes of
    row `y`. -/
def rhoPiDone (y x j : Nat) : Prop :=
  j % 5 < y ∨ ∃ x' : Nat, x' < x ∧ rhoPiWrite x' y = j

/-- Lanes the chi sweep has written after `y` complete rows and `x` lanes of
    row `y`; chi writes lane `x + 5y` in place, so no permutation is involved. -/
def chiDone (y x j : Nat) : Prop :=
  j / 5 < y ∨ (j / 5 = y ∧ j % 5 < x)

/-- Loop invariant of the rho-pi sweep, read through `view`: written lanes hold
    the canonical `pi (rho s)` value, the rest are still the zero the scratch
    array was created with. -/
def rhoPiInv (s : Vector (BitVec 64) 25) (b : Array Std.U64 25#usize) (y x : Nat) : Prop :=
  ∀ j : Fin 25,
    (rhoPiDone y x j.val → (view b).get j = (pi (rho s)).get j) ∧
      (¬ rhoPiDone y x j.val → (view b).get j = 0)

/-- Loop invariant of the chi sweep, read through `view`: written lanes hold the
    canonical `chi s` value, the rest are the state the sweep started from. -/
def chiInv (s : Vector (BitVec 64) 25) (s0 b : Array Std.U64 25#usize) (y x : Nat) : Prop :=
  ∀ j : Fin 25,
    (chiDone y x j.val → (view b).get j = (chi s).get j) ∧
      (¬ chiDone y x j.val → (view b).get j = (view s0).get j)

/-- One more lane of rho-pi row `y` changes the written set of exactly one
    lane. -/
theorem rhoPiDone_step (y x j : Nat) :
    rhoPiDone y (x + 1) j ↔ rhoPiDone y x j ∨ rhoPiWrite x y = j := by
  simp only [rhoPiDone]
  constructor
  · intro h
    rcases h with h | ⟨x', hx', heq⟩
    · exact Or.inl (Or.inl h)
    · have hsplit : x' < x ∨ x' = x := by omega
      rcases hsplit with hsplit | rfl
      · exact Or.inl (Or.inr ⟨x', hsplit, heq⟩)
      · exact Or.inr heq
  · intro h
    rcases h with h | heq
    · rcases h with h | ⟨x', hx', heq⟩
      · exact Or.inl h
      · exact Or.inr ⟨x', by omega, heq⟩
    · exact Or.inr ⟨x, by omega, heq⟩

/-- One more lane of chi row `y` changes the written set of exactly one lane. -/
theorem chiDone_step (y x j : Nat) :
    chiDone y (x + 1) j ↔ chiDone y x j ∨ (j / 5 = y ∧ j % 5 = x) := by
  simp only [chiDone]
  constructor
  · intro h
    rcases h with h | ⟨hdiv, hmod⟩
    · exact Or.inl (Or.inl h)
    · have hsplit : j % 5 < x ∨ j % 5 = x := by omega
      rcases hsplit with hsplit | rfl
      · exact Or.inl (Or.inr ⟨hdiv, hsplit⟩)
      · exact Or.inr ⟨hdiv, rfl⟩
  · intro h
    rcases h with h | ⟨hdiv, hmod⟩
    · rcases h with h | ⟨hdiv, hmod⟩
      · exact Or.inl h
      · exact Or.inr ⟨hdiv, by omega⟩
    · exact Or.inr ⟨hdiv, by omega⟩

/-! ### Lane arithmetic -/

theorem rhoPiWrite_lt (x y : Nat) (hx : x < 5) (hy : y < 5) : rhoPiWrite x y < 25 := by
  unfold rhoPiWrite
  have hk : (2 * x + 3 * y) % 5 < 5 := Nat.mod_lt _ (by decide)
  omega

theorem rhoPiWrite_mod (x y : Nat) (hy : y < 5) : rhoPiWrite x y % 5 = y := by
  unfold rhoPiWrite
  have hk : (2 * x + 3 * y) % 5 < 5 := Nat.mod_lt _ (by decide)
  omega

theorem rhoPiWrite_div (x y : Nat) (hy : y < 5) :
    rhoPiWrite x y / 5 = (2 * x + 3 * y) % 5 := by
  unfold rhoPiWrite
  have hk : (2 * x + 3 * y) % 5 < 5 := Nat.mod_lt _ (by decide)
  omega

/-- For `x, y < 5`, π reads lane `x + 5 * y` of its argument at the destination
    `rhoPiWrite x y`: this is the FIPS-202 ρπ index identity. -/
theorem pi_rho_index (x y : Nat) (hx : x < 5) (hy : y < 5) :
    (rhoPiWrite x y % 5 + 3 * (rhoPiWrite x y / 5)) % 5 + 5 * (rhoPiWrite x y % 5)
      = x + 5 * y := by
  have hk : (2 * x + 3 * y) % 5 < 5 := Nat.mod_lt _ (by decide)
  have hmod : rhoPiWrite x y % 5 = y := rhoPiWrite_mod x y hy
  have hdiv : rhoPiWrite x y / 5 = (2 * x + 3 * y) % 5 := rhoPiWrite_div x y hy
  have h1 : (2 * x + 3 * y) % 5 + 5 * ((2 * x + 3 * y) / 5) = 2 * x + 3 * y :=
    Nat.mod_add_div _ _
  have : (y + 3 * ((2 * x + 3 * y) % 5)) % 5 = x := by omega
  rw [hmod, hdiv, this]

/-- The map `x ↦ (2 * x + 3 * y) mod 5` covers every residue for `y < 5`, so
    every lane of column `y` is reached by some rho-pi pair. -/
theorem rhoPi_dest_surj (y k : Nat) (hy : y < 5) (hk : k < 5) :
    ∃ x : Nat, x < 5 ∧ (2 * x + 3 * y) % 5 = k := by
  refine ⟨(3 * k + y) % 5, Nat.mod_lt _ (by decide), ?_⟩
  have hk2 : (3 * k + y) % 5 < 5 := Nat.mod_lt _ (by decide)
  have h1 : (3 * k + y) % 5 + 5 * ((3 * k + y) / 5) = 3 * k + y := Nat.mod_add_div _ _
  have : (2 * ((3 * k + y) % 5) + 3 * y) % 5 = k % 5 := by omega
  rw [this, Nat.mod_eq_of_lt hk]

/-! ### Canonical lane values -/

/-- FIPS-202 ρ at lane `j`, `j < 25`. -/
theorem rho_get (s : Vector (BitVec 64) 25) (j : Nat) (hj : j < 25) :
    (rho s).get ⟨j, hj⟩ = (s.get ⟨j, hj⟩).rotateLeft (ROT.get ⟨j, hj⟩) := by
  show (rho s)[j] = _
  simp only [rho, Vector.getElem_ofFn]

/-- FIPS-202 χ at lane `x + 5 * y`, `x, y < 5`. -/
theorem chi_get (s : Vector (BitVec 64) 25) (x y : Nat) (hx : x < 5) (hy : y < 5) :
    (chi s).get ⟨x + 5 * y, by omega⟩ =
      s.get ⟨x + 5 * y, by omega⟩ ^^^
        ((~~~s.get ⟨(x + 1) % 5 + 5 * y, by omega⟩) &&&
          s.get ⟨(x + 2) % 5 + 5 * y, by omega⟩) := by
  have hmod : (x + 5 * y) % 5 = x := by omega
  have hdiv : (x + 5 * y) / 5 = y := by omega
  show (chi s)[x + 5 * y] = _
  simp only [chi, lane, Vector.getElem_ofFn]
  rw [hmod, hdiv]
  have h1 : (x + 1) % 5 + 5 * y < 25 := by
    have : (x + 1) % 5 < 5 := Nat.mod_lt _ (by decide)
    omega
  have h2 : (x + 2) % 5 + 5 * y < 25 := by
    have : (x + 2) % 5 < 5 := Nat.mod_lt _ (by decide)
    omega
  simp only [Nat.mod_eq_of_lt h1, Nat.mod_eq_of_lt h2]

/-- FIPS-202 π at lane `j < 25`, in the form the FIPS index arithmetic produces. -/
theorem pi_get (s : Vector (BitVec 64) 25) (j : Nat) (hj : j < 25) :
    (pi s).get ⟨j, hj⟩ =
      s.get ⟨((j % 5 + 3 * (j / 5)) % 5 + 5 * (j % 5)) % 25, by omega⟩ := by
  show (pi s)[j] = _
  simp only [pi, lane, Vector.getElem_ofFn]

/-- FIPS-202 π∘ρ at the destination `rhoPiWrite x y` of the pair `(x, y)`: the
    rotated lane `x + 5 * y` of `s`. -/
theorem pi_rho_get (s : Vector (BitVec 64) 25) (x y : Nat) (hx : x < 5) (hy : y < 5) :
    (pi (rho s)).get ⟨rhoPiWrite x y, rhoPiWrite_lt x y hx hy⟩ =
      (s.get ⟨x + 5 * y, by omega⟩).rotateLeft (ROT.get ⟨x + 5 * y, by omega⟩) := by
  have hidx : (rhoPiWrite x y % 5 + 3 * (rhoPiWrite x y / 5)) % 5
      + 5 * (rhoPiWrite x y % 5) = x + 5 * y := pi_rho_index x y hx hy
  have hsrc : x + 5 * y < 25 := by omega
  have h25 : ((rhoPiWrite x y % 5 + 3 * (rhoPiWrite x y / 5)) % 5
      + 5 * (rhoPiWrite x y % 5)) % 25 = x + 5 * y := by
    rw [hidx]; exact Nat.mod_eq_of_lt hsrc
  rw [pi_get (rho s) (rhoPiWrite x y) (rhoPiWrite_lt x y hx hy)]
  simp only [rho_get, h25]

/-- No lane of row `y` has been written at `x = 0`. -/
theorem rhoPiDone_zero (y j : Nat) : rhoPiDone y 0 j ↔ j % 5 < y := by
  simp only [rhoPiDone]
  constructor
  · rintro (h | ⟨x', hx', _⟩)
    · exact h
    · omega
  · intro h
    exact Or.inl h

/-- Once row `y` is complete, every lane with `j % 5 ≤ y` has been written. -/
theorem rhoPiDone_five (y j : Nat) (hy : y < 5) (hj : j < 25) :
    rhoPiDone y 5 j ↔ j % 5 < y + 1 := by
  constructor
  · intro hd
    rcases hd with h | ⟨x', hx', heq⟩
    · omega
    · have hmod := rhoPiWrite_mod x' y hy
      rw [heq] at hmod
      omega
  · intro h
    by_cases hlt : j % 5 < y
    · exact Or.inl hlt
    · have hmod : j % 5 = y := by omega
      have hk : j / 5 < 5 := Nat.div_lt_of_lt_mul (by omega : j < 5 * 5)
      obtain ⟨x', hx5, hxw⟩ := rhoPi_dest_surj y (j / 5) hy hk
      refine Or.inr ⟨x', hx5, ?_⟩
      rw [rhoPiWrite, hxw]
      have hjm : j % 5 + 5 * (j / 5) = j := Nat.mod_add_div j 5
      omega

/-- No lane of row `y` has been written at `x = 0`. -/
theorem chiDone_zero (y j : Nat) : chiDone y 0 j ↔ j / 5 < y := by
  simp only [chiDone]
  constructor
  · rintro (h | ⟨hdiv, hmod⟩)
    · exact h
    · omega
  · intro h
    exact Or.inl h

/-- Once row `y` is complete, every lane with `j / 5 ≤ y` has been written. -/
theorem chiDone_five (y j : Nat) : chiDone y 5 j ↔ j / 5 < y + 1 := by
  constructor
  · intro hd
    rcases hd with h | ⟨hdiv, hmod⟩ <;> omega
  · intro h
    by_cases hlt : j / 5 < y
    · exact Or.inl hlt
    · exact Or.inr ⟨by omega, Nat.mod_lt j (by decide)⟩

/-! ### Rho-pi body step -/

/-- Lane `x + 5 * y` of the state as a word. -/
def laneW64 (state : Array Std.U64 25#usize) (j : Nat) (hj : j < 25) : Std.U64 :=
  state[j]'(by
    have h : state.val.length = 25 := by simpa using state.property
    omega)

/-- The rho offset table read at a `Nat` index. -/
def rotW (j : Nat) (hj : j < 25) : Std.U32 :=
  crypto.ROT[j]'(by
    have h : crypto.ROT.val.length = 25 := by
      simp only [crypto.ROT, Array.make_val]
      rfl
    omega)

/-- Lane `x + 5 * y` of `state`, rotated by its rho offset: the word the rho-pi
    body writes. -/
def rhoPiRot (state : Array Std.U64 25#usize) (x y : Nat) (hx : x + 5 * y < 25) : Std.U64 :=
  UScalar.rotate_left (laneW64 state (x + 5 * y) hx) (rotW (x + 5 * y) hx)

/-- One rho-pi iteration at `(x, y)`: for `x, y < 5` the body stores
    `state[x + 5y].rotate_left(ROT[x + 5y])` at the destination lane
    `rhoPiWrite x y`, increments `x`, and leaves every other lane alone. -/
theorem rhoPi_body_step (state b : Array Std.U64 25#usize) (y x : Std.Usize)
    (hy : y.val < 5) (hx : x.val < 5) :
    ∃ (w : Std.Usize) (b' : Array Std.U64 25#usize) (x' : Std.Usize),
      crypto.keccak_f1600_loop0_loop2_loop0.body state y b x
          = ok (ControlFlow.cont (b', x')) ∧
        x'.val = x.val + 1 ∧
        w.val = rhoPiWrite x.val y.val ∧
        b' = b.set w (rhoPiRot state x.val y.val (by omega)) := by
  have hs : state.val.length = 25 := by simpa using state.property
  have hb : b.val.length = 25 := by simpa using b.property
  have hrotlen : crypto.ROT.val.length = 25 := by
    simp only [crypto.ROT, Array.make_val]
    rfl
  have h2v : (2#usize).val = 2 := rfl
  have h3v : (3#usize).val = 3 := rfl
  have h5v : (5#usize).val = 5 := rfl
  have hlt : x < 5#usize := (UScalar.lt_equiv x 5#usize).mpr hx
  have hxy : x.val + 5 * y.val < state.val.length := by omega
  -- i = 5 * y
  have hmax5 : (5#usize).val * y.val ≤ Usize.max := by scalar_tac
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := 5#usize) (y := y) hmax5)
  -- i1 = i3 = x + i
  have hmax1 : x.val + i.val ≤ Usize.max := by scalar_tac
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := x) (y := i) hmax1)
  have hi1val : i1.val = x.val + 5 * y.val := by rw [hi1v, hiv, h5v]
  have hidx : Array.index_usize state i1 = ok (state.val[x.val + 5 * y.val]'hxy) := by
    simpa only [hi1val] using array_index_usize_eq state i1 (by omega)
  -- i4 = ROT[x + 5y]
  have hrotIdx : Array.index_usize crypto.ROT i1 =
      ok (crypto.ROT.val[x.val + 5 * y.val]'(by rw [hrotlen]; omega)) := by
    simpa only [hi1val] using array_index_usize_eq crypto.ROT i1 (by omega)
  -- i6 = 2 * x, i7 = 3 * y
  have hmax2 : (2#usize).val * x.val ≤ Usize.max := by scalar_tac
  obtain ⟨i6, hi6, hi6v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := 2#usize) (y := x) hmax2)
  have hmax3 : (3#usize).val * y.val ≤ Usize.max := by scalar_tac
  obtain ⟨i7, hi7, hi7v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := 3#usize) (y := y) hmax3)
  -- i8 = 2x + 3y
  have hmax8 : i6.val + i7.val ≤ Usize.max := by scalar_tac
  obtain ⟨i8, hi8, hi8v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i6) (y := i7) hmax8)
  have hi8val : i8.val = 2 * x.val + 3 * y.val := by rw [hi8v, hi6v, hi7v, h2v, h3v]
  -- i9 = (2x + 3y) % 5
  obtain ⟨i9, hi9v, hi9⟩ := WP.spec_imp_exists (Usize.rem_spec (x := i8) (y := 5#usize) (by decide))
  have hi9val : i9.val = (2 * x.val + 3 * y.val) % 5 := by rw [hi9, hi8val, h5v]
  -- i10 = 5 * i9
  have hmax10 : (5#usize).val * i9.val ≤ Usize.max := by
    have : (2 * x.val + 3 * y.val) % 5 < 5 := Nat.mod_lt _ (by decide)
    rw [hi9val, h5v]; scalar_tac
  obtain ⟨i10, hi10, hi10v⟩ :=
    WP.spec_imp_exists (Usize.mul_spec (x := 5#usize) (y := i9) hmax10)
  have hi10val : i10.val = 5 * ((2 * x.val + 3 * y.val) % 5) := by rw [hi10v, hi9val, h5v]
  -- i11 = y + i10
  have hmax11 : y.val + i10.val ≤ Usize.max := by
    have : (2 * x.val + 3 * y.val) % 5 < 5 := Nat.mod_lt _ (by decide)
    rw [hi10val]; scalar_tac
  obtain ⟨i11, hi11, hi11v⟩ := WP.spec_imp_exists (Usize.add_spec (x := y) (y := i10) hmax11)
  have hi11val : i11.val = rhoPiWrite x.val y.val := by
    rw [hi11v, hi10val]
    unfold rhoPiWrite
    rfl
  have hwb : i11.val < b.val.length := by
    rw [hi11val, hb]
    exact rhoPiWrite_lt x.val y.val hx hy
  -- x1 = x + 1
  have hmaxx1 : x.val + (1#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨x1, hx1, hx1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := x) (y := 1#usize) hmaxx1)
  -- b1 = b.set i11 (rotate_left state[x+5y] (ROT[x+5y]))
  obtain ⟨b1, hb1, hb1eq⟩ := WP.spec_imp_exists (Array.update_spec b i11
    (core.num.U64.rotate_left (state.val[x.val + 5 * y.val]'hxy)
      (crypto.ROT.val[x.val + 5 * y.val]'(by rw [hrotlen]; omega))) hwb)
  have hrotEq : core.num.U64.rotate_left (state.val[x.val + 5 * y.val]'hxy)
        (crypto.ROT.val[x.val + 5 * y.val]'(by rw [hrotlen]; omega))
      = rhoPiRot state x.val y.val (by omega) := by
    unfold rhoPiRot laneW64 rotW
    rfl
  have hb1v : b1 = b.set i11 (rhoPiRot state x.val y.val (by omega)) := by
    rw [← hrotEq]
    exact hb1eq
  refine ⟨i11, b1, x1, ?_, hx1v, hi11val, hb1v⟩
  · unfold crypto.keccak_f1600_loop0_loop2_loop0.body
    simp only [lift, bind_tc_ok]
    rw [if_pos hlt]
    try dsimp only
    rw [hi]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hi1]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hidx]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hrotIdx]
    simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hi6]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hi7]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hi8]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hi9v]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hi10]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hi11]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hb1]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hx1]
    simp only [bind_tc_ok]
    try dsimp only

/-! ### Rho-pi loops -/

/-- The `x` loop of the rho-pi sweep: from `x` lanes of row `y` written, the
    extracted loop finishes row `y`. -/
theorem rhoPi_row_spec (state b : Array Std.U64 25#usize) (y x : Std.Usize)
    (hy : y.val < 5) (hx : x.val ≤ 5)
    (hinv : rhoPiInv (view state) b y.val x.val) :
    crypto.keccak_f1600_loop0_loop2_loop0 state b y x ⦃ b' =>
      rhoPiInv (view state) b' y.val 5 ⦄ := by
  unfold crypto.keccak_f1600_loop0_loop2_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array Std.U64 25#usize × Std.Usize => 5 - s.2.val)
    (inv := fun s => s.2.val ≤ 5 ∧ rhoPiInv (view state) s.1 y.val s.2.val)
    (post := fun b' => rhoPiInv (view state) b' y.val 5)
  · intro s hs
    rcases s with ⟨b1, x1⟩
    dsimp only at hs ⊢
    have hxle : x1.val ≤ 5 := hs.1
    have hs' : rhoPiInv (view state) b1 y.val x1.val := hs.2
    by_cases hlt : x1.val < 5
    · obtain ⟨w, b2, x2, hbody, hx2v, hw, hb2eq⟩ :=
        rhoPi_body_step state b1 y x1 hy hlt
      have hwb : w.val < 25 := by rw [hw]; exact rhoPiWrite_lt _ _ hlt hy
      have hlen : state.val.length = 25 := by simpa using state.property
      have hrotlen : crypto.ROT.val.length = 25 := by
        simp only [crypto.ROT, Array.make_val]
        rfl
      have hidx1 : x1.val + 5 * y.val < state.val.length := by omega
      have hidx2 : x1.val + 5 * y.val < crypto.ROT.val.length := by omega
      have hval : (rhoPiRot state x1.val y.val (by omega)).bv =
          (pi (rho (view state))).get ⟨rhoPiWrite x1.val y.val, by omega⟩ := by
        show (state.val[x1.val + 5 * y.val]'hidx1).bv.rotateLeft
            (crypto.ROT.val[x1.val + 5 * y.val]'hidx2).val = _
        rw [show (state.val[x1.val + 5 * y.val]'hidx1).bv
              = ((view state).get ⟨x1.val + 5 * y.val, by omega⟩) from
            (view_get state ⟨x1.val + 5 * y.val, by omega⟩).symm]
        rw [show (crypto.ROT.val[x1.val + 5 * y.val]'hidx2).val
              = ROT.get ⟨x1.val + 5 * y.val, by omega⟩ from
            ROT_get ⟨x1.val + 5 * y.val, by omega⟩]
        exact (pi_rho_get (view state) x1.val y.val hlt hy).symm
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      refine ⟨⟨by omega, ?_⟩, by omega⟩
      intro j
      rw [hb2eq, view_set_get b1 w _ hwb j]
      have hstep : rhoPiDone y.val x2.val j.val ↔
          rhoPiDone y.val x1.val j.val ∨ w.val = j.val := by
        rw [hx2v]
        exact (rhoPiDone_step y.val x1.val j.val).trans (by rw [hw])
      have hdone := (hs' j).1
      have hfresh := (hs' j).2
      constructor
      · intro hd
        by_cases hje : j.val = w.val
        · rw [if_pos hje, hval]
          congr 1
          apply Fin.ext
          show rhoPiWrite x1.val y.val = j.val
          rw [← hw]
          exact hje.symm
        · rw [if_neg hje]
          exact hdone ((hstep.mp hd).resolve_right (fun h => hje h.symm))
      · intro hn
        by_cases hje : j.val = w.val
        · exact absurd ((hstep.mpr) (Or.inr hje.symm)) hn
        · rw [if_neg hje]
          exact hfresh (fun hd1 => hn ((hstep.mpr) (Or.inl hd1)))
    · have hx5 : x1.val = 5 := by omega
      have hnot : ¬(x1 < 5#usize) := fun hc' => hlt ((UScalar.lt_equiv x1 5#usize).mp hc')
      have hbody : crypto.keccak_f1600_loop0_loop2_loop0.body state y b1 x1 =
          ok (ControlFlow.done b1) := by
        unfold crypto.keccak_f1600_loop0_loop2_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      rw [hx5] at hs'
      exact fun j => hs' j
  · exact ⟨hx, hinv⟩

/-- The `y` loop of the rho-pi sweep: with no lane of row `y` written, the
    extracted loop writes every lane of the scratch array with the canonical
    `pi (rho s)` value. -/
theorem rhoPi_spec (state b : Array Std.U64 25#usize) (y : Std.Usize)
    (hy : y.val ≤ 5) (hinv : rhoPiInv (view state) b y.val 0) :
    crypto.keccak_f1600_loop0_loop2 state b y ⦃ b' =>
      ∀ j : Fin 25, (view b').get j = (pi (rho (view state))).get j ⦄ := by
  unfold crypto.keccak_f1600_loop0_loop2
  apply loop.spec_decr_nat
    (measure := fun s : Array Std.U64 25#usize × Std.Usize => 5 - s.2.val)
    (inv := fun s => s.2.val ≤ 5 ∧ rhoPiInv (view state) s.1 s.2.val 0)
    (post := fun b' => ∀ j : Fin 25, (view b').get j = (pi (rho (view state))).get j)
  · intro s hs
    rcases s with ⟨b1, y1⟩
    dsimp only at hs ⊢
    have hyle : y1.val ≤ 5 := hs.1
    have hs' : rhoPiInv (view state) b1 y1.val 0 := hs.2
    by_cases hlt : y1.val < 5
    · obtain ⟨brow, hbrow, hfull⟩ :=
        WP.spec_imp_exists (rhoPi_row_spec state b1 y1 (0#usize) (by omega) (by decide) hs')
      have hstep : ∀ j : Fin 25, rhoPiDone y1.val 5 j.val ↔ rhoPiDone (y1.val + 1) 0 j.val := by
        intro j
        simp only [rhoPiDone_zero (y1.val + 1) j.val, rhoPiDone_five y1.val j.val hlt j.isLt]
      have hmax1 : y1.val + (1#usize).val ≤ Usize.max := by scalar_tac
      obtain ⟨y2, hy2v, hy2⟩ := WP.spec_imp_exists (Usize.add_spec (x := y1) (y := 1#usize) hmax1)
      have hy2 : y2.val = y1.val + 1 := by
        have h1v : (1#usize).val = 1 := rfl
        omega
      have hbody : crypto.keccak_f1600_loop0_loop2.body state b1 y1
          = ok (ControlFlow.cont (brow, y2)) := by
        unfold crypto.keccak_f1600_loop0_loop2.body
        rw [if_pos ((UScalar.lt_equiv y1 5#usize).mpr hlt)]
        rw [hbrow]
        simp only [bind_tc_ok]
        rw [hy2v]
        simp only [bind_tc_ok]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      refine ⟨⟨by omega, ?_⟩, by omega⟩
      have hnext : rhoPiInv (view state) brow (y1.val + 1) 0 := by
        intro k
        constructor
        · intro hd
          exact (hfull k).1 ((hstep k).mpr hd)
        · intro hn
          exact (hfull k).2 (fun h => hn ((hstep k).mp h))
      rw [hy2]
      exact hnext
    · have hy5 : y1.val = 5 := by omega
      have hnot : ¬(y1 < 5#usize) := fun hc' => hlt ((UScalar.lt_equiv y1 5#usize).mp hc')
      have hbody : crypto.keccak_f1600_loop0_loop2.body state b1 y1
          = ok (ControlFlow.done b1) := by
        unfold crypto.keccak_f1600_loop0_loop2.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      intro j
      have hdone : rhoPiDone y1.val 0 j.val := Or.inl (by rw [hy5]; omega)
      exact (hs' j).1 hdone
  · exact ⟨hy, hinv⟩

/-- The fresh rho-pi scratch array satisfies the invariant at row `0`. -/
theorem rhoPi_inv_fresh (state b : Array Std.U64 25#usize) (hz : ∀ j : Fin 25, (view b).get j = 0)
    (y : Std.Usize) (hy : y.val = 0) : rhoPiInv (view state) b y.val 0 := by
  intro j
  constructor
  · intro hd
    rcases hd with h | ⟨x', hx', heq⟩ <;> omega
  · intro _
    exact hz j

/-- The 25-lane zero scratch the round body allocates. -/
theorem view_repeat_zero (j : Fin 25) : (view (Array.repeat 25#usize 0#u64)).get j = 0 := by
  show (view (Array.repeat 25#usize 0#u64))[j.val] = 0
  simp only [view, Vector.getElem_ofFn]
  have h : (Array.repeat 25#usize 0#u64)[j.val] = 0#u64 := by
    rw [Array.getElem_Nat_eq (Array.repeat 25#usize 0#u64) j.val
      (by rw [Array.repeat_val]; simp)]
    simp only [Array.repeat_val, List.getElem_replicate]
  rw [h]
  rfl

/-- The rho-pi sweep of one round: from the zero scratch, the extracted loop
    `crypto.keccak_f1600_loop0_loop2` returns the canonical `pi (rho (view state))`. -/
theorem rhoPi_zero_spec (state : Array Std.U64 25#usize) :
    crypto.keccak_f1600_loop0_loop2 state (Array.repeat 25#usize 0#u64) 0#usize ⦃ b' =>
      ∀ j : Fin 25, (view b').get j = (pi (rho (view state))).get j ⦄ :=
  rhoPi_spec state (Array.repeat 25#usize 0#u64) 0#usize (by decide)
    (rhoPi_inv_fresh state (Array.repeat 25#usize 0#u64) view_repeat_zero 0#usize rfl)

/-! ### Chi body step -/

/-- Lane `j` of the rho-pi output array, as a word. -/
def bW (b : Array Std.U64 25#usize) (j : Nat) (hj : j < 25) : Std.U64 :=
  b[j]'(by
    have h : b.val.length = 25 := by simpa using b.property
    omega)

/-- One chi iteration at `(x, y)`: for `x, y < 5` the body stores
    `b[x + 5y] ^ ((!b[((x+1) mod 5) + 5y]) & b[((x+2) mod 5) + 5y])` at lane
    `x + 5 * y` of the state, increments `x`, and leaves every other lane alone. -/
theorem chi_body_step (b state : Array Std.U64 25#usize) (y x : Std.Usize)
    (hy : y.val < 5) (hx : x.val < 5) :
    ∃ (w : Std.Usize) (a : Array Std.U64 25#usize) (x' : Std.Usize),
      crypto.keccak_f1600_loop0_loop3_loop0.body b y state x
          = ok (ControlFlow.cont (a, x')) ∧
        x'.val = x.val + 1 ∧
        w.val = x.val + 5 * y.val ∧
        a = state.set w (UScalar.xor (bW b (x.val + 5 * y.val) (by omega))
            (UScalar.and (UScalar.not (bW b ((x.val + 1) % 5 + 5 * y.val)
                (by have : (x.val + 1) % 5 < 5 := Nat.mod_lt _ (by decide); omega)))
              (bW b ((x.val + 2) % 5 + 5 * y.val)
                (by have : (x.val + 2) % 5 < 5 := Nat.mod_lt _ (by decide); omega)))) := by
  have hb : b.val.length = 25 := by simpa using b.property
  have hs : state.val.length = 25 := by simpa using state.property
  have h1v : (1#usize).val = 1 := rfl
  have h2v : (2#usize).val = 2 := rfl
  have h5v : (5#usize).val = 5 := rfl
  have hlt : x < 5#usize := (UScalar.lt_equiv x 5#usize).mpr hx
  -- i = 5 * y
  have hmax5 : (5#usize).val * y.val ≤ Usize.max := by scalar_tac
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := 5#usize) (y := y) hmax5)
  -- i1 = i13 = x + i
  have hmax1 : x.val + i.val ≤ Usize.max := by scalar_tac
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := x) (y := i) hmax1)
  have hi1val : i1.val = x.val + 5 * y.val := by rw [hi1v, hiv, h5v]
  have hidx1 : Array.index_usize b i1 = ok (b.val[x.val + 5 * y.val]) := by
    simpa only [hi1val] using array_index_usize_eq b i1 (by omega)
  -- i3 = x + 1
  have hmax3 : x.val + (1#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists (Usize.add_spec (x := x) (y := 1#usize) hmax3)
  have hi3val : i3.val = x.val + 1 := by rw [hi3v, h1v]
  -- i4 = (x + 1) % 5
  obtain ⟨i4, hi4, hi4v⟩ := WP.spec_imp_exists (Usize.rem_spec (x := i3) (y := 5#usize) (by decide))
  have hi4val : i4.val = (x.val + 1) % 5 := by rw [hi4v, hi3val, h5v]
  -- i5 = ((x + 1) % 5) + 5y
  have hmax5' : i4.val + i.val ≤ Usize.max := by
    have : (x.val + 1) % 5 < 5 := Nat.mod_lt _ (by decide)
    rw [hi4val, hiv, h5v]; scalar_tac
  obtain ⟨i5, hi5, hi5v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i4) (y := i) hmax5')
  have hi5val : i5.val = (x.val + 1) % 5 + 5 * y.val := by rw [hi5v, hi4val, hiv, h5v]
  have hidx5 : Array.index_usize b i5 = ok (b.val[(x.val + 1) % 5 + 5 * y.val]) := by
    simpa only [hi5val] using array_index_usize_eq b i5 (by
      have : (x.val + 1) % 5 < 5 := Nat.mod_lt _ (by decide)
      omega)
  -- i8 = x + 2
  have hmax8 : x.val + (2#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨i8, hi8, hi8v⟩ := WP.spec_imp_exists (Usize.add_spec (x := x) (y := 2#usize) hmax8)
  have hi8val : i8.val = x.val + 2 := by rw [hi8v, h2v]
  -- i9 = (x + 2) % 5
  obtain ⟨i9, hi9, hi9v⟩ := WP.spec_imp_exists (Usize.rem_spec (x := i8) (y := 5#usize) (by decide))
  have hi9val : i9.val = (x.val + 2) % 5 := by rw [hi9v, hi8val, h5v]
  -- i10 = ((x + 2) % 5) + 5y
  have hmax10 : i9.val + i.val ≤ Usize.max := by
    have : (x.val + 2) % 5 < 5 := Nat.mod_lt _ (by decide)
    rw [hi9val, hiv, h5v]; scalar_tac
  obtain ⟨i10, hi10, hi10v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i9) (y := i) hmax10)
  have hi10val : i10.val = (x.val + 2) % 5 + 5 * y.val := by rw [hi10v, hi9val, hiv, h5v]
  have hidx10 : Array.index_usize b i10 = ok (b.val[(x.val + 2) % 5 + 5 * y.val]) := by
    simpa only [hi10val] using array_index_usize_eq b i10 (by
      have : (x.val + 2) % 5 < 5 := Nat.mod_lt _ (by decide)
      omega)
  have hbx : x.val + 5 * y.val < b.val.length := by rw [hb]; omega
  have hb1' : (x.val + 1) % 5 + 5 * y.val < b.val.length := by
    have : (x.val + 1) % 5 < 5 := Nat.mod_lt _ (by decide)
    rw [hb]; omega
  have hb2' : (x.val + 2) % 5 + 5 * y.val < b.val.length := by
    have : (x.val + 2) % 5 < 5 := Nat.mod_lt _ (by decide)
    rw [hb]; omega
  obtain ⟨a, ha, haeq⟩ := WP.spec_imp_exists (Array.update_spec state i1
    (b.val[x.val + 5 * y.val]'hbx ^^^
      ((~~~ (b.val[(x.val + 1) % 5 + 5 * y.val]'hb1')) &&&
        (b.val[(x.val + 2) % 5 + 5 * y.val]'hb2')))
    (by have hlen : state.length = 25 := state.property; omega))
  have haeq' : a = state.set i1
      (b.val[x.val + 5 * y.val]'hbx ^^^
        ((~~~ (b.val[(x.val + 1) % 5 + 5 * y.val]'hb1')) &&&
          (b.val[(x.val + 2) % 5 + 5 * y.val]'hb2'))) := haeq
  refine ⟨i1, a, i3, ?_, hi3val, hi1val, ?_⟩
  · unfold crypto.keccak_f1600_loop0_loop3_loop0.body
    simp only [lift, bind_tc_ok]
    rw [if_pos hlt]
    try dsimp only
    rw [hi]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hi1]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hidx1]
    simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hi3]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hi4]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hi5]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hidx5]
    simp only [lift, bind_tc_ok]
    try dsimp only
    rw [hi8]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hi9]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hi10]
    simp only [bind_tc_ok]
    try dsimp only
    rw [hidx10]
    simp only [lift, bind_tc_ok]
    try dsimp only
    rw [ha]
    simp only [bind_tc_ok]
    try dsimp only
  · exact haeq'

/-! ### Chi loops -/

/-- The `x` loop of the chi sweep: from `x` lanes of row `y` written, the
    extracted loop finishes row `y`. -/
theorem chi_row_spec (b s0 state : Array Std.U64 25#usize) (y x : Std.Usize)
    (hy : y.val < 5) (hx : x.val ≤ 5)
    (hinv : chiInv (view b) s0 state y.val x.val) :
    crypto.keccak_f1600_loop0_loop3_loop0 state b y x ⦃ a =>
      chiInv (view b) s0 a y.val 5 ⦄ := by
  unfold crypto.keccak_f1600_loop0_loop3_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array Std.U64 25#usize × Std.Usize => 5 - s.2.val)
    (inv := fun s => s.2.val ≤ 5 ∧ chiInv (view b) s0 s.1 y.val s.2.val)
    (post := fun a => chiInv (view b) s0 a y.val 5)
  · intro s hs
    rcases s with ⟨st, x1⟩
    dsimp only at hs ⊢
    have hxle : x1.val ≤ 5 := hs.1
    have hs' : chiInv (view b) s0 st y.val x1.val := hs.2
    by_cases hlt : x1.val < 5
    · obtain ⟨w, a, x2, hbody, hx2v, hw, hval⟩ := chi_body_step b st y x1 hy hlt
      have hwb : w.val < 25 := by rw [hw]; omega
      have hstep : ∀ j : Nat, chiDone y.val x2.val j ↔
          chiDone y.val x1.val j ∨ w.val = j := by
        intro j
        rw [hx2v, chiDone_step y.val x1.val j, hw]
        constructor
        · intro h
          rcases h with h | ⟨hdiv, hmod⟩
          · exact Or.inl h
          · refine Or.inr ?_
            have hm := Nat.mod_add_div j 5
            omega
        · intro h
          rcases h with h | heq
          · exact Or.inl h
          · refine Or.inr ⟨?_, ?_⟩
            · have hm := Nat.mod_add_div j 5; omega
            · have hm := Nat.mod_add_div j 5; omega
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      have hvb : U64.bv (UScalar.xor (bW b (x1.val + 5 * y.val) (by omega))
            (UScalar.and (UScalar.not (bW b ((x1.val + 1) % 5 + 5 * y.val)
                (by have : (x1.val + 1) % 5 < 5 := Nat.mod_lt _ (by decide); omega)))
              (bW b ((x1.val + 2) % 5 + 5 * y.val)
                (by have : (x1.val + 2) % 5 < 5 := Nat.mod_lt _ (by decide); omega))))
          = (chi (view b)).get ⟨x1.val + 5 * y.val, by omega⟩ := by
        show U64.bv (bW b (x1.val + 5 * y.val) (by omega)) ^^^
            (~~~ U64.bv (bW b ((x1.val + 1) % 5 + 5 * y.val)
                (by have : (x1.val + 1) % 5 < 5 := Nat.mod_lt _ (by decide); omega)) &&&
              U64.bv (bW b ((x1.val + 2) % 5 + 5 * y.val)
                (by have : (x1.val + 2) % 5 < 5 := Nat.mod_lt _ (by decide); omega))) = _
        have e1 : U64.bv (bW b (x1.val + 5 * y.val) (by omega))
            = (view b).get ⟨x1.val + 5 * y.val, by omega⟩ :=
          (view_get b ⟨x1.val + 5 * y.val, by omega⟩).symm
        have e2 : U64.bv (bW b ((x1.val + 1) % 5 + 5 * y.val)
              (by have : (x1.val + 1) % 5 < 5 := Nat.mod_lt _ (by decide); omega))
            = (view b).get ⟨(x1.val + 1) % 5 + 5 * y.val, by omega⟩ :=
          (view_get b ⟨(x1.val + 1) % 5 + 5 * y.val, by omega⟩).symm
        have e3 : U64.bv (bW b ((x1.val + 2) % 5 + 5 * y.val)
              (by have : (x1.val + 2) % 5 < 5 := Nat.mod_lt _ (by decide); omega))
            = (view b).get ⟨(x1.val + 2) % 5 + 5 * y.val, by omega⟩ :=
          (view_get b ⟨(x1.val + 2) % 5 + 5 * y.val, by omega⟩).symm
        rw [e1, e2, e3]
        exact (chi_get (view b) x1.val y.val hlt hy).symm
      refine ⟨⟨by omega, ?_⟩, by omega⟩
      intro j
      rw [hval, view_set_get st w _ hwb j]
      constructor
      · intro hd
        by_cases hje : j.val = w.val
        · rw [if_pos hje, hvb]
          congr 1
          apply Fin.ext
          show x1.val + 5 * y.val = j.val
          rw [← hw]
          exact hje.symm
        · rw [if_neg hje]
          exact (hs' j).1 (((hstep j.val).mp hd).resolve_right (fun h => hje h.symm))
      · intro hn
        by_cases hje : j.val = w.val
        · exact absurd ((hstep j.val).mpr (Or.inr hje.symm)) hn
        · rw [if_neg hje]
          exact (hs' j).2 (fun hd1 => hn ((hstep j.val).mpr (Or.inl hd1)))
    · have hx5 : x1.val = 5 := by omega
      have hnot : ¬(x1 < 5#usize) := fun hc' => hlt ((UScalar.lt_equiv x1 5#usize).mp hc')
      have hbody : crypto.keccak_f1600_loop0_loop3_loop0.body b y st x1
          = ok (ControlFlow.done st) := by
        unfold crypto.keccak_f1600_loop0_loop3_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      rw [hx5] at hs'
      exact fun j => hs' j
  · exact ⟨hx, hinv⟩

/-- The `y` loop of the chi sweep: with no lane of row `y` written, the
    extracted loop writes the canonical `chi (view b)` at every lane. -/
theorem chi_spec (b s0 state : Array Std.U64 25#usize) (y : Std.Usize)
    (hy : y.val ≤ 5) (hinv : chiInv (view b) s0 state y.val 0) :
    crypto.keccak_f1600_loop0_loop3 state b y ⦃ a =>
      ∀ j : Fin 25, (view a).get j = (chi (view b)).get j ⦄ := by
  unfold crypto.keccak_f1600_loop0_loop3
  apply loop.spec_decr_nat
    (measure := fun s : Array Std.U64 25#usize × Std.Usize => 5 - s.2.val)
    (inv := fun s => s.2.val ≤ 5 ∧ chiInv (view b) s0 s.1 s.2.val 0)
    (post := fun a => ∀ j : Fin 25, (view a).get j = (chi (view b)).get j)
  · intro s hs
    rcases s with ⟨st, y1⟩
    dsimp only at hs ⊢
    have hyle : y1.val ≤ 5 := hs.1
    have hs' : chiInv (view b) s0 st y1.val 0 := hs.2
    by_cases hlt : y1.val < 5
    · obtain ⟨strow, hstrow, hfull⟩ :=
        WP.spec_imp_exists (chi_row_spec b s0 st y1 (0#usize) (by omega) (by decide) hs')
      have hstep : ∀ j : Fin 25, chiDone y1.val 5 j.val ↔ chiDone (y1.val + 1) 0 j.val := by
        intro j
        simp only [chiDone_zero (y1.val + 1) j.val, chiDone_five y1.val j.val]
      have hmax1 : y1.val + (1#usize).val ≤ Usize.max := by scalar_tac
      obtain ⟨y2, hy2v, hy2⟩ := WP.spec_imp_exists (Usize.add_spec (x := y1) (y := 1#usize) hmax1)
      have hy2 : y2.val = y1.val + 1 := by
        have h1v : (1#usize).val = 1 := rfl
        omega
      have hbody : crypto.keccak_f1600_loop0_loop3.body b st y1
          = ok (ControlFlow.cont (strow, y2)) := by
        unfold crypto.keccak_f1600_loop0_loop3.body
        rw [if_pos ((UScalar.lt_equiv y1 5#usize).mpr hlt)]
        rw [hstrow]
        simp only [bind_tc_ok]
        rw [hy2v]
        simp only [bind_tc_ok]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      refine ⟨⟨by omega, ?_⟩, by omega⟩
      have hnext : chiInv (view b) s0 strow (y1.val + 1) 0 := by
        intro k
        constructor
        · intro hd
          exact (hfull k).1 ((hstep k).mpr hd)
        · intro hn
          exact (hfull k).2 (fun h => hn ((hstep k).mp h))
      rw [hy2]
      exact hnext
    · have hy5 : y1.val = 5 := by omega
      have hnot : ¬(y1 < 5#usize) := fun hc' => hlt ((UScalar.lt_equiv y1 5#usize).mp hc')
      have hbody : crypto.keccak_f1600_loop0_loop3.body b st y1
          = ok (ControlFlow.done st) := by
        unfold crypto.keccak_f1600_loop0_loop3.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      intro j
      exact (hs' j).1 (Or.inl (by rw [hy5]; omega))
  · exact ⟨hy, hinv⟩

/-- The chi sweep of one round: from the theta output `s0`, the extracted loop
    `crypto.keccak_f1600_loop0_loop3` returns the canonical `chi (view b)`. -/
theorem chi_of_spec (b s0 : Array Std.U64 25#usize) :
    crypto.keccak_f1600_loop0_loop3 s0 b 0#usize ⦃ a =>
      ∀ j : Fin 25, (view a).get j = (chi (view b)).get j ⦄ :=
  chi_spec b s0 s0 0#usize (by decide) (by
    intro j
    constructor
    · intro hd
      exact absurd ((chiDone_zero 0 j).mp hd) (by omega)
    · intro _
      rfl)

/-! ### Iota read-back -/

/-- The iota write of one round: XORing lane 0 with the round constant read
    from `crypto.RC` is the canonical `iota` through `view`. -/
theorem iota_write (a : Array Std.U64 25#usize) (n : Fin 24) (rc : Std.U64)
    (hrc : rc.bv = RC.get n) :
    view (a.set 0#usize (a[0] ^^^ rc)) = iota n (view a) := by
  have h0 : (0#usize).val < 25 := by decide
  have hxor : U64.bv (a[0] ^^^ rc) = U64.bv (a[0]) ^^^ U64.bv rc := UScalar.bv_xor _ _
  rw [view_set a 0#usize _ h0, iota, hxor, ← hrc, view_get a ⟨0, by decide⟩]
  rfl

/-! ## Theta apply loop

`src/crypto.rs:82-90` runs the outer `x` loop that, for each column `x`,
computes `D[x]` from the column-parity scratch `c` and XORs it into the five
lanes `state[x + 5y]`, `y = 0..4`. `thetaInv` is the `x`-loop invariant over
`view`; `thetaRowInv` is the inner `y`-loop invariant; `thetaApply_spec` joins
them against the canonical `theta`. -/

/-- The `x`-loop invariant of θ: the lanes whose column index is below `x`
    already hold the canonical `theta` value, the rest still hold the input. -/
def thetaInv (state0 state : Array Std.U64 25#usize) (x : Nat) : Prop :=
  x ≤ 5 ∧ ∀ j : Fin 25, (view state).get j =
    if j.val % 5 < x then (theta (view state0)).get j else (view state0).get j

/-- The `y`-loop invariant of the θ apply loop: the lanes of column `x` whose
    row index is below `y` hold the start state XORed with the correction `d`,
    every other lane is unchanged. -/
def thetaRowInv (start : Array Std.U64 25#usize) (d : Std.U64) (x y : Nat)
    (state : Array Std.U64 25#usize) : Prop :=
  y ≤ 5 ∧ ∀ j : Fin 25, (view state).get j =
    if j.val % 5 = x ∧ j.val / 5 < y then (view start).get j ^^^ d.bv
    else (view start).get j

/-- The zeroed 5-word scratch read through `colGet` is bit-vector zero. -/
theorem colGet_repeat_zero (j : Fin 5) :
    (colGet (Array.repeat 5#usize 0#u64) j).bv = 0 := by
  unfold colGet
  rw [Array.getElem_Nat_eq (Array.repeat 5#usize 0#u64) j.val
    (by rw [Array.repeat_val, List.length_replicate]; exact j.isLt)]
  simp only [Array.repeat_val, List.getElem_replicate]
  rfl

/-- One θ apply row iteration at `(x, y)`: the body reads `state[x + 5y]`,
    writes it XORed with `d` back in place, and increments `y`. -/
theorem thetaRow_body_step (state : Array Std.U64 25#usize) (d : Std.U64)
    (x y : Std.Usize) (hx : x.val < 5) (hy : y.val < 5) :
    ∃ (w : Std.Usize) (state' : Array Std.U64 25#usize) (y' : Std.Usize),
      crypto.keccak_f1600_loop0_loop1_loop0.body x d state y
          = ok (ControlFlow.cont (state', y')) ∧
        y'.val = y.val + 1 ∧
        w.val = x.val + 5 * y.val ∧
        ∀ j : Fin 25, (view state').get j =
          if j.val = w.val then (view state).get j ^^^ d.bv else (view state).get j := by
  have h1v : (1#usize).val = 1 := rfl
  have h5v : (5#usize).val = 5 := rfl
  have hs : state.val.length = 25 := by simpa using state.property
  have hlt : y < 5#usize := (UScalar.lt_equiv y 5#usize).mpr hy
  have hmax5 : (5#usize).val * y.val ≤ Usize.max := by scalar_tac
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := 5#usize) (y := y) hmax5)
  have hival : i.val = 5 * y.val := by rw [hiv, h5v]
  have hmax1 : x.val + i.val ≤ Usize.max := by scalar_tac
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := x) (y := i) hmax1)
  have hi1val : i1.val = x.val + 5 * y.val := by rw [hi1v, hival]
  have hxy : i1.val < state.val.length := by rw [hi1val, hs]; omega
  have hxy25 : i1.val < 25 := by rw [hi1val]; omega
  have hidx : Array.index_usize state i1 = ok (state.val[i1.val]'hxy) :=
    array_index_usize_eq state i1 hxy
  have hmaxy1 : y.val + (1#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨y1, hy1, hy1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := y) (y := 1#usize) hmaxy1)
  have hy1val : y1.val = y.val + 1 := by rw [hy1v, h1v]
  obtain ⟨a, ha, haeq⟩ := WP.spec_imp_exists (Array.update_spec state i1
    (state.val[i1.val]'hxy ^^^ d) hxy)
  refine ⟨i1, a, y1, ?_, hy1val, hi1val, ?_⟩
  · unfold crypto.keccak_f1600_loop0_loop1_loop0.body
    rw [if_pos hlt]
    simp only [lift, bind_tc_ok, hi, hi1, hidx, ha, hy1]
  · intro j
    rw [haeq, view_set_get state i1 (state.val[i1.val]'hxy ^^^ d) hxy25 j]
    by_cases h : j.val = i1.val
    · simp only [if_pos h]
      have hv : U64.bv (state.val[i1.val]'hxy ^^^ d) = (view state).get j ^^^ d.bv := by
        show U64.bv (state.val[i1.val]'hxy) ^^^ U64.bv d = (view state).get j ^^^ d.bv
        have hj : j = ⟨i1.val, hxy25⟩ := Fin.ext h
        rw [hj]
        rw [view_get]
        rfl
      exact hv
    · simp only [if_neg h]

/-- The `y` loop of the θ apply loop: from `y ≤ 5` it XORs `d` into every lane
    of column `x` at or above row `y`. -/
theorem thetaRow_spec (start state : Array Std.U64 25#usize) (d : Std.U64)
    (x : Std.Usize) (hx : x.val < 5) (y : Std.Usize) (hy : y.val ≤ 5)
    (hinv : thetaRowInv start d x.val y.val state) :
    crypto.keccak_f1600_loop0_loop1_loop0 state x d y ⦃ state' =>
      ∀ j : Fin 25, (view state').get j =
        if j.val % 5 = x.val then (view start).get j ^^^ d.bv
        else (view start).get j ⦄ := by
  unfold crypto.keccak_f1600_loop0_loop1_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array Std.U64 25#usize × Std.Usize => 5 - s.2.val)
    (inv := fun s => thetaRowInv start d x.val s.2.val s.1)
    (post := fun state' => ∀ j : Fin 25, (view state').get j =
      if j.val % 5 = x.val then (view start).get j ^^^ d.bv else (view start).get j)
  · intro s hs
    rcases s with ⟨st, y1⟩
    dsimp only at hs ⊢
    obtain ⟨hyle, hs'⟩ := hs
    by_cases hlt : y1.val < 5
    · obtain ⟨w, st2, y2, hbody, hy2, hw, hst2⟩ := thetaRow_body_step st d x y1 hx hlt
      have hwx : w.val % 5 = x.val := by
        rw [hw, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hx]
      have hwd : w.val / 5 = y1.val := by
        rw [hw, Nat.add_mul_div_left _ _ (by decide : 0 < 5), Nat.div_eq_of_lt hx, Nat.zero_add]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      refine ⟨⟨by omega, ?_⟩, by omega⟩
      intro j
      rw [hy2]
      rw [hst2 j]
      by_cases hcol : j.val % 5 = x.val
      · by_cases hrow : j.val / 5 < y1.val
        · have hneq : ¬ (j.val = w.val) := by
            intro hje
            rw [hje, hwd] at hrow
            omega
          rw [if_neg hneq, hs' j, if_pos ⟨hcol, hrow⟩, if_pos ⟨hcol, by omega⟩]
        · by_cases heq : j.val = w.val
          · have hrow2 : j.val / 5 < y1.val + 1 := by rw [heq, hwd]; omega
            have hncond : ¬ (j.val % 5 = x.val ∧ j.val / 5 < y1.val) := by
              intro hc; rw [heq, hwd] at hc; omega
            rw [if_pos heq, hs' j, if_neg hncond, if_pos ⟨hcol, hrow2⟩]
          · have hncond : ¬ (j.val % 5 = x.val ∧ j.val / 5 < y1.val) := by
              intro hc; exact hrow hc.2
            have hncond2 : ¬ (j.val % 5 = x.val ∧ j.val / 5 < y1.val + 1) := by
              rintro ⟨h1, h2⟩
              have hd : j.val / 5 = y1.val := by omega
              have hmad := Nat.mod_add_div j.val 5
              rw [h1, hd, Nat.mul_comm] at hmad
              have hjw : j.val = w.val := by rw [hw]; omega
              exact heq hjw
            rw [if_neg heq, hs' j, if_neg hncond, if_neg hncond2]
      · have hneq : ¬ (j.val = w.val) := by
          intro hje
          exact hcol (by rw [hje, hwx])
        rw [if_neg hneq, hs' j,
          if_neg (by intro hc; exact hcol hc.1), if_neg (by intro hc; exact hcol hc.1)]
    · have hy5 : y1.val = 5 := by omega
      have hnot : ¬ (y1 < 5#usize) := fun hc' => hlt ((UScalar.lt_equiv y1 5#usize).mp hc')
      have hbody : crypto.keccak_f1600_loop0_loop1_loop0.body x d st y1
          = ok (ControlFlow.done st) := by
        unfold crypto.keccak_f1600_loop0_loop1_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      intro j
      rw [hs' j]
      by_cases hcol : j.val % 5 = x.val
      · have hj5 : j.val / 5 < y1.val := by
          rw [hy5]
          exact Nat.div_lt_of_lt_mul (by have := j.isLt; omega)
        rw [if_pos ⟨hcol, hj5⟩, if_pos hcol]
      · rw [if_neg (by intro hc; exact hcol hc.1), if_neg hcol]
  · show thetaRowInv start d x.val y.val state
    exact hinv

/-- One θ apply iteration at `x < 5`: computes `D[x]` from the scratch column
    parities and applies the row loop, advancing the invariant to `x + 1`. -/
theorem thetaApply_body_step (state0 : Array Std.U64 25#usize) (c : Array Std.U64 5#usize)
    (hc : ∀ j : Fin 5, (colGet c j).bv = (colW state0 j.val).bv)
    (state : Array Std.U64 25#usize) (x : Std.Usize) (hx : x.val < 5)
    (hinv : thetaInv state0 state x.val) :
    ∃ (state' : Array Std.U64 25#usize) (x' : Std.Usize),
      crypto.keccak_f1600_loop0_loop1.body c state x
          = ok (ControlFlow.cont (state', x')) ∧
        x'.val = x.val + 1 ∧
        thetaInv state0 state' x'.val := by
  obtain ⟨hxle, hstate⟩ := hinv
  have h1v : (1#usize).val = 1 := rfl
  have h4v : (4#usize).val = 4 := rfl
  have h5v : (5#usize).val = 5 := rfl
  have hlt : x < 5#usize := (UScalar.lt_equiv x 5#usize).mpr hx
  have hmax4 : x.val + (4#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.add_spec (x := x) (y := 4#usize) hmax4)
  have hival : i.val = x.val + 4 := by rw [hiv, h4v]
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.rem_spec (x := i) (y := 5#usize) (by decide))
  have hi1val : i1.val = (x.val + 4) % 5 := by rw [hi1v, hival, h5v]
  have hi1lt : i1.val < 5 := by rw [hi1val]; exact Nat.mod_lt _ (by decide)
  have hi1c : i1.val < c.val.length := by
    have : c.val.length = 5 := by simpa using c.property
    omega
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Array.index_usize_spec c i1 hi1c)
  have hi2col : i2 = colGet c ⟨i1.val, hi1lt⟩ := by rw [hi2v]; rfl
  have hmaxx1 : x.val + (1#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists (Usize.add_spec (x := x) (y := 1#usize) hmaxx1)
  have hi3val : i3.val = x.val + 1 := by rw [hi3v, h1v]
  obtain ⟨i4, hi4, hi4v⟩ := WP.spec_imp_exists (Usize.rem_spec (x := i3) (y := 5#usize) (by decide))
  have hi4val : i4.val = (x.val + 1) % 5 := by rw [hi4v, hi3val, h5v]
  have hi4lt : i4.val < 5 := by rw [hi4val]; exact Nat.mod_lt _ (by decide)
  have hi4c : i4.val < c.val.length := by
    have : c.val.length = 5 := by simpa using c.property
    omega
  obtain ⟨i5, hi5, hi5v⟩ := WP.spec_imp_exists (Array.index_usize_spec c i4 hi4c)
  have hi5col : i5 = colGet c ⟨i4.val, hi4lt⟩ := by rw [hi5v]; rfl
  have h2bv : i2.bv = (colW state0 ((x.val + 4) % 5)).bv := by
    have h := hc ⟨(x.val + 4) % 5, Nat.mod_lt _ (by decide)⟩
    have hc' : (colGet c ⟨(x.val + 4) % 5, Nat.mod_lt _ (by decide)⟩).bv
        = (colGet c ⟨i1.val, hi1lt⟩).bv :=
      congrArg (fun k : Fin 5 => (colGet c k).bv) (Fin.ext hi1val.symm)
    rw [hi2col, ← hc']
    exact h
  have h5bv : i5.bv = (colW state0 ((x.val + 1) % 5)).bv := by
    have h := hc ⟨(x.val + 1) % 5, Nat.mod_lt _ (by decide)⟩
    have hc' : (colGet c ⟨(x.val + 1) % 5, Nat.mod_lt _ (by decide)⟩).bv
        = (colGet c ⟨i4.val, hi4lt⟩).bv :=
      congrArg (fun k : Fin 5 => (colGet c k).bv) (Fin.ext hi4val.symm)
    rw [hi5col, ← hc']
    exact h
  have hD : (i2 ^^^ core.num.U64.rotate_left i5 1#u32).bv = thetaD state0 x.val := by
    show i2.bv ^^^ (core.num.U64.rotate_left i5 1#u32).bv = thetaD state0 x.val
    have hr : (core.num.U64.rotate_left i5 1#u32).bv = (i5.bv).rotateLeft 1 := rfl
    simp only [hr, h2bv, h5bv]
    unfold thetaD
    rfl
  have hrow0 : thetaRowInv state (i2 ^^^ core.num.U64.rotate_left i5 1#u32) x.val 0 state := by
    refine ⟨by omega, ?_⟩
    intro j
    rw [if_neg (by omega : ¬ (j.val % 5 = x.val ∧ j.val / 5 < 0))]
  obtain ⟨state1, hrow, hrowpost⟩ := WP.spec_imp_exists
    (thetaRow_spec state state (i2 ^^^ core.num.U64.rotate_left i5 1#u32) x hx 0#usize
      (by decide) hrow0)
  refine ⟨state1, i3, ?_, hi3val, ?_⟩
  · unfold crypto.keccak_f1600_loop0_loop1.body
    rw [if_pos hlt]
    simp only [lift, bind_tc_ok, hi, hi1, hi2, hi3, hi4, hi5, hrow]
  · refine ⟨by omega, ?_⟩
    intro j
    rw [hrowpost j, hi3val]
    by_cases hcol : j.val % 5 = x.val
    · have hnotlt : ¬ (j.val % 5 < x.val) := by omega
      rw [if_pos hcol, hstate j, if_neg hnotlt,
        if_pos (by omega : j.val % 5 < x.val + 1)]
      rw [theta_get (view state0) j]
      simp only [hD]
      unfold thetaD
      rw [hcol, colW_bv, colW_bv]
    · by_cases hltx : j.val % 5 < x.val
      · rw [if_neg hcol, hstate j, if_pos hltx,
          if_pos (by omega : j.val % 5 < x.val + 1)]
      · rw [if_neg hcol, hstate j, if_neg hltx,
          if_neg (by omega : ¬ (j.val % 5 < x.val + 1))]

/-- The θ apply loop of FIPS-202 §3.2.1: from the scratch parity array `c` the
    extracted `x` loop returns the canonical `theta` of the input. -/
theorem thetaApply_spec (state0 : Array Std.U64 25#usize) (c : Array Std.U64 5#usize)
    (hc : ∀ j : Fin 5, (colGet c j).bv = (colW state0 j.val).bv) :
    crypto.keccak_f1600_loop0_loop1 state0 c 0#usize ⦃ state' =>
      ∀ j : Fin 25, (view state').get j = (theta (view state0)).get j ⦄ := by
  unfold crypto.keccak_f1600_loop0_loop1
  apply loop.spec_decr_nat
    (measure := fun s : Array Std.U64 25#usize × Std.Usize => 5 - s.2.val)
    (inv := fun s => thetaInv state0 s.1 s.2.val)
    (post := fun state' => ∀ j : Fin 25, (view state').get j = (theta (view state0)).get j)
  · intro s hs
    rcases s with ⟨st, x1⟩
    dsimp only at hs ⊢
    by_cases hlt : x1.val < 5
    · obtain ⟨st1, x2, hbody, hx2, hinv2⟩ := thetaApply_body_step state0 c hc st x1 hlt hs
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      exact ⟨hinv2, by omega⟩
    · have hx5 : x1.val = 5 := by obtain ⟨hxle, _⟩ := hs; omega
      have hnot : ¬ (x1 < 5#usize) := fun hc' => hlt ((UScalar.lt_equiv x1 5#usize).mp hc')
      have hbody : crypto.keccak_f1600_loop0_loop1.body c st x1
          = ok (ControlFlow.done st) := by
        unfold crypto.keccak_f1600_loop0_loop1.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      intro j
      rw [hs.2 j, hx5, if_pos (by have := j.isLt; omega)]
  · show (0#usize).val ≤ 5 ∧ ∀ j : Fin 25, (view state0).get j =
        if j.val % 5 < (0#usize).val then (theta (view state0)).get j else (view state0).get j
    refine ⟨by decide, ?_⟩
    intro j
    rw [if_neg (by show ¬ (j.val % 5 < 0); omega)]

/-! ## The outer 24-round fold

`src/crypto.rs:71-119` runs one `round < 24` loop over the four step sweeps and
the iota XOR. `RoundStepDenotes` is one iteration against the canonical `round`;
`foldRounds` is the prefix fold the loop invariant carries; `RoundStepDenotes`
and `WordKeccakfDenotes` are the refinement stages, proved below. -/

/-- The round `n` of the canonical permutation if `n < 24`, and the identity
    otherwise. Total so it can appear in a `Fin.foldl`. -/
private def roundAt (m : Nat) : Vector (BitVec 64) 25 → Vector (BitVec 64) 25 :=
  fun s => if h : m < 24 then round ⟨m, h⟩ s else s

/-- The canonical permutation prefix: rounds `0, …, m-1` folded over `s`. -/
private def foldRounds (s : Vector (BitVec 64) 25) (m : Nat) : Vector (BitVec 64) 25 :=
  Fin.foldl m (fun acc (j : Fin m) => roundAt j.val acc) s

private theorem foldRounds_zero (s : Vector (BitVec 64) 25) : foldRounds s 0 = s := by
  unfold foldRounds
  exact Fin.foldl_zero _ _

private theorem foldRounds_succ (s : Vector (BitVec 64) 25) (m : Nat) (hm : m < 24) :
    foldRounds s (m + 1) = round ⟨m, hm⟩ (foldRounds s m) := by
  unfold foldRounds
  rw [Fin.foldl_succ_last]
  have hfold : Fin.foldl m (fun acc (i : Fin m) => roundAt (Fin.castSucc i).val acc) s
      = Fin.foldl m (fun acc (i : Fin m) => roundAt i.val acc) s := by
    apply congrArg (fun g => Fin.foldl m g s)
    funext acc i
    rfl
  rw [hfold, show (Fin.last m).val = m from rfl]
  unfold roundAt
  rw [dif_pos hm]

private theorem foldRounds_permute (s : Vector (BitVec 64) 25) :
    foldRounds s 24 = permute s := by
  unfold foldRounds permute
  rw [Fin.foldl_eq_finRange_foldl]
  apply congrArg (fun g => (List.finRange 24).foldl g s)
  funext acc j
  show roundAt j.val acc = round j acc
  unfold roundAt
  rw [dif_pos j.isLt]

theorem usize_add_one_val {n n' : Std.Usize} (h : n + 1#usize = ok n') :
    n'.val = n.val + 1 := by
  have ha := UScalar.add_equiv n (1#usize : Std.Usize)
  have he : n + 1#usize = UScalar.add n (1#usize : Std.Usize) := rfl
  rw [he] at h
  rw [he] at ha
  rw [h] at ha
  simp only [Result.match.ok] at ha
  have hv := ha.2.1
  rw [show (1#usize : Std.Usize).val = 1 from rfl] at hv
  exact hv

/-- The outer-loop body equation, isolated so the do-block reduction gets its own
    heartbeat budget: from the intermediate results and their equations, the body
    returns `cont (a, n')`. -/
private theorem keccak_f1600_loop0_body_eq (state : Array Std.U64 25#usize) (n : Std.Usize)
    (rc : Std.U64) (c1 : Array Std.U64 5#usize) (state1 : Array Std.U64 25#usize)
    (b1 state2 a : Array Std.U64 25#usize) (i0 : Std.U64) (n' : Std.Usize)
    (hlt : n < 24#usize)
    (hrcEq : Array.index_usize crypto.RC n = ok rc)
    (hc1Eq : crypto.keccak_f1600_loop0_loop0 state (Array.repeat 5#usize 0#u64) 0#usize = ok c1)
    (hstate1Eq : crypto.keccak_f1600_loop0_loop1 state c1 0#usize = ok state1)
    (hb1Eq : crypto.keccak_f1600_loop0_loop2 state1 (Array.repeat 25#usize 0#u64) 0#usize = ok b1)
    (hstate2Eq : crypto.keccak_f1600_loop0_loop3 state1 b1 0#usize = ok state2)
    (hi0Eq : Array.index_usize state2 0#usize = ok i0)
    (haEq : Array.update state2 0#usize (i0 ^^^ rc) = ok a)
    (hn'Eq : n + 1#usize = ok n') :
    crypto.keccak_f1600_loop0.body state n = ok (ControlFlow.cont (a, n')) := by
  unfold crypto.keccak_f1600_loop0.body
  rw [if_pos hlt]
  simp only [lift, bind_tc_ok]
  rw [hrcEq]; simp only [bind_tc_ok]
  rw [hc1Eq]; simp only [bind_tc_ok]
  rw [hstate1Eq]; simp only [bind_tc_ok]
  rw [hb1Eq]; simp only [bind_tc_ok]
  rw [hstate2Eq]; simp only [bind_tc_ok]
  rw [hi0Eq]; simp only [lift, bind_tc_ok]
  rw [haEq]; simp only [bind_tc_ok]
  rw [hn'Eq]; simp only [bind_tc_ok]

/-- The iota step of one outer iteration: the in-place XOR of lane 0 is the
    canonical `iota` through `view`. -/
private theorem roundBody_iota (state2 a : Array Std.U64 25#usize) (n : Fin 24)
    (rc i0 : Std.U64) (haSet : a = state2.set 0#usize (i0 ^^^ rc))
    (hi0V : i0 = state2.val[(0#usize).val]) (hrc : rc.bv = RC.get n) :
    view a = iota n (view state2) := by
  rw [haSet]
  show view (state2.set 0#usize (i0 ^^^ rc)) = iota n (view state2)
  rw [hi0V]
  exact iota_write state2 n rc hrc

/-- The view chain of one outer iteration: the four sweep posts and the iota
    step compose to `round`. -/
private theorem roundBody_view (state : Array Std.U64 25#usize) (n : Fin 24)
    (a b1 state1 state2 : Array Std.U64 25#usize)
    (hstate1post : ∀ j : Fin 25, (view state1).get j = (theta (view state)).get j)
    (hb1post : ∀ j : Fin 25, (view b1).get j = (pi (rho (view state1))).get j)
    (hstate2post : ∀ j : Fin 25, (view state2).get j = (chi (view b1)).get j)
    (hav : view a = iota n (view state2)) :
    view a = round n (view state) := by
  rw [hav]
  have e2 : view state2 = chi (view b1) := by
    apply Vector.ext; intro j hj; exact hstate2post ⟨j, hj⟩
  have e1 : view b1 = pi (rho (view state1)) := by
    apply Vector.ext; intro j hj; exact hb1post ⟨j, hj⟩
  have e0 : view state1 = theta (view state) := by
    apply Vector.ext; intro j hj; exact hstate1post ⟨j, hj⟩
  rw [e2, e1, e0]
  rfl

/-- One iteration of the extracted outer loop: for `n < 24` the body runs the
    four sweeps and the iota write, returning `view state' = round n (view state)`
    and advancing the round counter. -/
theorem RoundStepDenotes (state : Array Std.U64 25#usize) (n : Std.Usize) (hn : n.val < 24) :
    ∃ (state' : Array Std.U64 25#usize) (n' : Std.Usize),
      crypto.keccak_f1600_loop0.body state n = ok (ControlFlow.cont (state', n')) ∧
      n + 1#usize = ok n' ∧
      view state' = round ⟨n.val, hn⟩ (view state) := by
  have hs : state.val.length = 25 := by simpa using state.property
  have hRClen : crypto.RC.length = 24 := by
    show crypto.RC.val.length = 24
    rw [Array.property crypto.RC]
    rfl
  have hlt : n < 24#usize := (UScalar.lt_equiv n 24#usize).mpr hn
  obtain ⟨rc, hrcEq, hrcV⟩ := WP.spec_imp_exists
    (Array.index_usize_spec crypto.RC n (by rw [hRClen]; omega))
  obtain ⟨c1, hc1Eq, hc1post⟩ := WP.spec_imp_exists
    (columns_spec state (Array.repeat 5#usize 0#u64) 0#usize (by decide)
      (fun j => by
        rw [if_neg (by show ¬ (j.val < 0); omega)]
        exact colGet_repeat_zero j))
  obtain ⟨state1, hstate1Eq, hstate1post⟩ := WP.spec_imp_exists (thetaApply_spec state c1 hc1post)
  obtain ⟨b1, hb1Eq, hb1post⟩ := WP.spec_imp_exists (rhoPi_zero_spec state1)
  obtain ⟨state2, hstate2Eq, hstate2post⟩ := WP.spec_imp_exists (chi_of_spec b1 state1)
  have hs2 : state2.val.length = 25 := by simpa using state2.property
  have h0b : (0#usize).val < state2.val.length := by rw [hs2]; decide
  obtain ⟨i0, hi0Eq, hi0V⟩ := WP.spec_imp_exists (Array.index_usize_spec state2 0#usize h0b)
  have hmaxx1 : n.val + (1#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨n', hn'Eq, hn'V⟩ := WP.spec_imp_exists (Usize.add_spec (x := n) (y := 1#usize) hmaxx1)
  have hn'val : n'.val = n.val + 1 := by rw [hn'V, show (1#usize).val = 1 from rfl]
  obtain ⟨a, haEq, haSet⟩ := WP.spec_imp_exists
    (Array.update_spec state2 0#usize (i0 ^^^ rc)
      (by show (0#usize).val < state2.val.length; rw [hs2]; decide))
  have hrc : rc.bv = RC.get ⟨n.val, hn⟩ := by
    rw [hrcV]
    exact RC_get ⟨n.val, hn⟩
  have hav : view a = iota ⟨n.val, hn⟩ (view state2) :=
    roundBody_iota state2 a ⟨n.val, hn⟩ rc i0 haSet hi0V hrc
  refine ⟨a, n', ?_, hn'Eq, ?_⟩
  · exact keccak_f1600_loop0_body_eq state n rc c1 state1 b1 state2 a i0 n' hlt
      hrcEq hc1Eq hstate1Eq hb1Eq hstate2Eq hi0Eq haEq hn'Eq
  · exact roundBody_view state ⟨n.val, hn⟩ a b1 state1 state2
      hstate1post hb1post hstate2post hav

/-- The extracted outer 24-round loop computes the canonical `permute`. -/
theorem keccak_f1600_loop0_spec (state0 : Array Std.U64 25#usize) :
    crypto.keccak_f1600_loop0 state0 0#usize ⦃ state' =>
      view state' = permute (view state0) ⦄ := by
  unfold crypto.keccak_f1600_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array Std.U64 25#usize × Std.Usize => 24 - s.2.val)
    (inv := fun s => s.2.val ≤ 24 ∧ view s.1 = foldRounds (view state0) s.2.val)
    (post := fun state' => view state' = permute (view state0))
  · intro s hs
    rcases s with ⟨st, m⟩
    dsimp only at hs ⊢
    obtain ⟨hmle, hsv⟩ := hs
    by_cases hlt : m.val < 24
    · obtain ⟨st1, m1, hbody, hm1eq, hview⟩ := RoundStepDenotes st m hlt
      have hm1val : m1.val = m.val + 1 := usize_add_one_val hm1eq
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      refine ⟨⟨by omega, ?_⟩, by omega⟩
      rw [hview, hsv, hm1val]
      exact (foldRounds_succ (view state0) m.val hlt).symm
    · have hm24 : m.val = 24 := by omega
      have hnot : ¬ (m < 24#usize) := fun hc' => hlt ((UScalar.lt_equiv m 24#usize).mp hc')
      have hbody : crypto.keccak_f1600_loop0.body st m = ok (ControlFlow.done st) := by
        unfold crypto.keccak_f1600_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      rw [hsv, hm24, foldRounds_permute]
  · show (0#usize).val ≤ 24 ∧ view state0 = foldRounds (view state0) (0#usize).val
    refine ⟨by decide, ?_⟩
    simpa using (foldRounds_zero (view state0)).symm

/-- The refinement headline: on every extracted state, `word_keccakf` returns
    `ok b` with `view b` the canonical 24-round permutation of `view a`. -/
theorem WordKeccakfDenotes :
    ∀ a : Array Std.U64 25#usize,
      ∃ b : Array Std.U64 25#usize,
        word_keccakf a = ok b ∧ view b = permute (view a) := by
  intro a
  obtain ⟨b, hb, hview⟩ := WP.spec_imp_exists (keccak_f1600_loop0_spec a)
  refine ⟨b, ?_, hview⟩
  show crypto.keccak_f1600_loop0 a 0#usize = ok b
  exact hb

end apeiron_word
