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

  Staged contents (this file):
  * the state conversion `view` and its read lemma;
  * the extracted constant-table bridges `crypto.ROT` and `crypto.RC`;
  * the theta column loop: `laneW`/`colW`/`colGet`, the per-iteration
    `column_body_step` and the loop invariant `columns_spec` over
    `loop.spec_decr_nat`;
  * the exact headline target (`WordKeccakfDenotes`) and the per-round target
    (`RoundStepDenotes`), carried as named propositions until their loop
    refinements land.

  Open obligations, next stages: the theta apply loop
  (`crypto.keccak_f1600_loop0_loop1`), rho-pi and chi loops, iota, the outer
  24-round fold (`crypto.keccak_f1600_loop0`) against `List.finRange 24`, the
  join `colW ↔ column` and then the headline. The iota step additionally needs
  a `view (a.set i x)` write lemma; a first `simp`-based attempt at it hit the
  default heartbeat limit and was deferred rather than run with an unbounded
  budget.
-/

import Crypto.Keccak.Round
import ApeironWord

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result
open Apeiron.Crypto.Keccak (RC ROT theta rho pi chi iota round permute)

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
  have hlt : x < 5#usize := (UScalar.lt_equiv x 5#usize).2 hx
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
        exact hlt ((UScalar.lt_equiv x1 5#usize).1 hc')
      have hbody : crypto.keccak_f1600_loop0_loop0.body state c1 x1 =
          ok (ControlFlow.done c1) := by
        unfold crypto.keccak_f1600_loop0_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      intro j
      rw [hc1 j, if_pos (by omega : j.val < x1.val)]
  · exact ⟨hx, hc⟩

/-! ## Targets

The headline and the per-round step, as named propositions. They are not yet
asserted: the loop refinements below them are the following stages, and a
`theorem` with an unfinished proof is not admissible in this tree. -/

/-- The refinement target: on every extracted state, `word_keccakf` returns
    `ok b` with `view b` the canonical 24-round permutation of `view a`. -/
def WordKeccakfDenotes : Prop :=
  ∀ a : Array Std.U64 25#usize,
    ∃ b : Array Std.U64 25#usize,
      word_keccakf a = ok b ∧ view b = permute (view a)

/-- The per-round target: one iteration of the extracted outer loop, from
    `view state = s` at round `n`, returns `view state' = round n s`. -/
def RoundStepDenotes : Prop :=
  ∀ (state : Array Std.U64 25#usize) (n : Std.Usize) (hn : n.val < 24),
    ∃ (state' : Array Std.U64 25#usize) (n' : Std.Usize),
      crypto.keccak_f1600_loop0.body state n = ok (ControlFlow.cont (state', n')) ∧
      n + 1#usize = ok n' ∧
      view state' = round ⟨n.val, hn⟩ (view state)

end apeiron_word
