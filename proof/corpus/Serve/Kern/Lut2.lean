/-
Integer refinement of the outer `PQ2_0` lookup-table kernels (`src/engine/kern/lut.rs`), group
`kern.lut`, continuing `Serve/Lut.lean` (which proves `split_nibbles` and the two inner loops of
`build_tables_range`).

`build_tables_range` writes, per 4-activation group `g`, the 16-entry table of the subset sums of
the four codes in `codes[4g..4g+4]`: entry `n` is `Σ_{k<4} bit_k(n)·code_k`. Each entry is stored
as two byte planes (`P + 508 = lo + 256·hi`) plus the group sum `Σ_k code_k` in `xs`. The reference
is `subsetSum`; the refinement is `word_build_tables_range_loop0_ok` (the general range) and its
corollary `word_build_tables_range_ok` (the whole input).

Boundaries stated, not papered over:

* `block_inners_avx2` / `tile_accs_avx2` are extracted `axiom`s (no body to refine);
* `core.num.I128.saturating_mul` and `core.num.I128.abs` are `axiom`s with no Aeneas spec, so the
  `*_bound_ok` predicates have no concrete value from the extraction;
* the `F32`/`F64` carriers and the `unsafe`/AVX2 paths are out of scope;
* the `+508` byte encode assumes the `PQ2_0` weights satisfy `|xq| ≤ 127` (Rust documents this but
  performs no check); it is carried as the explicit hypothesis `hcode : codeRange codes …`.

Grade: `proved` for each named theorem, `tested` for `src/engine/kern/lut_tests.rs`.
-/
import Serve.Kern.Lut

set_option autoImplicit false
set_option maxHeartbeats 200000

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## References for `build_tables_range`

`subsetSum c n` is the `n`-th subset sum of the four weights `c0..c3`, stated bitwise over the
index `n` rather than as the doubling loop the Rust runs. -/

/-- The `n`-th subset sum of four weights: `Σ_{k<4} bit_k(n)·c_k`. -/
def subsetSum (c0 c1 c2 c3 : Int) (n : Nat) : Int :=
  (if n.testBit 0 then c0 else 0) + (if n.testBit 1 then c1 else 0) +
  (if n.testBit 2 then c2 else 0) + (if n.testBit 3 then c3 else 0)

@[simp] lemma subsetSum_0 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 0 = 0 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_1 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 1 = c0 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_2 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 2 = c1 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_3 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 3 = c0 + c1 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_4 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 4 = c2 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_5 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 5 = c0 + c2 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_6 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 6 = c1 + c2 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_7 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 7 = c0 + c1 + c2 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_8 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 8 = c3 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_9 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 9 = c0 + c3 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_10 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 10 = c1 + c3 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_11 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 11 = c0 + c1 + c3 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_12 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 12 = c2 + c3 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_13 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 13 = c0 + c2 + c3 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_14 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 14 = c1 + c2 + c3 := by
  simp [subsetSum, Nat.testBit]
@[simp] lemma subsetSum_15 (c0 c1 c2 c3 : Int) : subsetSum c0 c1 c2 c3 15 = c0 + c1 + c2 + c3 := by
  simp [subsetSum, Nat.testBit]

lemma lt4_cases {n : Nat} (h : n < 4) : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 := by omega

lemma lt8_cases {n : Nat} (h : n < 8) :
    n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 := by omega

lemma lt16_cases {n : Nat} (h : n < 16) :
    n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨
    n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11 ∨ n = 12 ∨ n = 13 ∨ n = 14 ∨ n = 15 := by omega

/-- The high half of the doubling table is the low half plus `c3` (bit 3 selector). -/
lemma subsetSum_add_eight (c0 c1 c2 c3 : Int) (k : Nat) (h : k < 8) :
    subsetSum c0 c1 c2 c3 (8 + k) = subsetSum c0 c1 c2 c3 k + c3 := by
  rcases lt8_cases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [subsetSum, Nat.testBit]

/-- Code `j` of the slice, as an `Int` (the Rust reads it through `i8`; `getD` covers the padding
past the input only through the caller's length hypothesis). -/
def cAt (codes : Slice Std.I8) (j : Nat) : Int := (codes.val.getD j 0#i8).val

/-- The four weights of group `g` supply the table. -/
def grpLo (codes : Slice Std.I8) (g n : Nat) : Nat :=
  encLo (subsetSum (cAt codes (4 * g)) (cAt codes (4 * g + 1)) (cAt codes (4 * g + 2))
    (cAt codes (4 * g + 3)) n)

def grpHi (codes : Slice Std.I8) (g n : Nat) : Nat :=
  encHi (subsetSum (cAt codes (4 * g)) (cAt codes (4 * g + 1)) (cAt codes (4 * g + 2))
    (cAt codes (4 * g + 3)) n)

/-- The group sum `Σ_k code_k`, the last table entry. -/
def grpX (codes : Slice Std.I8) (g : Nat) : Int :=
  subsetSum (cAt codes (4 * g)) (cAt codes (4 * g + 1)) (cAt codes (4 * g + 2))
    (cAt codes (4 * g + 3)) 15

/-! ## WP form of the two inner loops (`Serve/Lut.lean`) -/

@[step]
lemma loop0_spec (p : Array Std.I32 16#usize) (c3 : Std.I32)
    (h : ∀ k, k < 8 → -2147483648 ≤ (p.val.getD k 0#i32).val + c3.val ∧
      (p.val.getD k 0#i32).val + c3.val ≤ 2147483647) :
    kern.lut.build_tables_range_loop0_loop0 p c3 0#usize ⦃ r =>
      (∀ k, k < 8 → (r.val.getD (8 + k) 0#i32).val = (p.val.getD k 0#i32).val + c3.val) ∧
      (∀ j, j < 8 → (r.val.getD j 0#i32).val = (p.val.getD j 0#i32).val) ⦄ := by
  obtain ⟨p', hp', h1, h2⟩ := word_build_tables_range_loop0_loop0_ok p c3 h
  rw [hp']
  simp only [WP.spec_ok]
  exact ⟨h1, h2⟩

@[step]
lemma loop1_spec (p : Array Std.I32 16#usize) (lo16 hi16 : Array Std.U8 16#usize)
    (hp : ∀ n, n < 16 → 0 ≤ (p.val.getD n 0#i32).val + 508 ∧
      (p.val.getD n 0#i32).val + 508 ≤ 65535) :
    kern.lut.build_tables_range_loop0_loop1 p lo16 hi16 0#usize ⦃ r =>
      (∀ n, n < 16 → (r.1.val.getD n 0#u8).val = encLo (p.val.getD n 0#i32).val) ∧
      (∀ n, n < 16 → (r.2.val.getD n 0#u8).val = encHi (p.val.getD n 0#i32).val) ⦄ := by
  obtain ⟨lo', hi', hp', h1, h2⟩ := word_build_tables_range_loop0_loop1_ok p lo16 hi16 hp
  rw [hp']
  simp only [WP.spec_ok]
  exact ⟨h1, h2⟩

/-! ## `getD` ↔ `getElem!` and the 8-entry staging block -/

/-- `getD` at an index below the length is the indexed element. -/
lemma getD_eq_getElem {α : Type} (l : List α) (j : Nat) (d : α) (h : j < l.length) :
    l.getD j d = l[j] := by
  simp [List.getD, List.getElem?_eq_getElem h]

/-- `getD` is `getElem?` followed by the default. -/
lemma getD_eq_getElem? {α : Type} (l : List α) (j : Nat) (d : α) :
    l.getD j d = (l[j]?).getD d := rfl

/-- `List.set` at an in-range index, read back with `getD`. -/
lemma getD_set_lt {α : Type} (l : List α) (j i : Nat) (x d : α) (hi : i < l.length) :
    (l.set j x).getD i d = if i = j then x else l.getD i d := by
  rw [getD_set]
  by_cases hij : i = j
  · rw [if_pos ⟨hij, by omega⟩, if_pos hij]
  · rw [if_neg (fun h => hij h.1), if_neg hij]

/-- `getD` with the scalar zero default is `getElem!` (the defaults coincide). -/
lemma getD_u8 (l : List Std.U8) (i : Nat) : l.getD i 0#u8 = l[i]! := by
  simp only [List.getD, List.getElem!_eq_getElem?_getD]
  rw [show (0#u8 : Std.U8) = default from UScalar.eq_of_val_eq (by simp)]

lemma getD_i16 (l : List Std.I16) (i : Nat) : l.getD i 0#i16 = l[i]! := by
  simp only [List.getD, List.getElem!_eq_getElem?_getD]
  rw [show (0#i16 : Std.I16) = default from
    IScalar.eq_of_val_eq (by simp [default, IScalar.ofInt_val_eq])]

lemma getD_i32 (l : List Std.I32) (i : Nat) : l.getD i 0#i32 = l[i]! := by
  simp only [List.getD, List.getElem!_eq_getElem?_getD]
  rw [show (0#i32 : Std.I32) = default from
    IScalar.eq_of_val_eq (by simp [default, IScalar.ofInt_val_eq])]

/-- The unrolled first three doubling steps write `[0, c0, c1, c0+c1, c2, c0+c2, c1+c2,
c0+c1+c2]` into `p[0..8)`. -/
lemma arr8_getD (p : Array Std.I32 16#usize) (v0 v1 v2 v3 v4 v5 v6 v7 : Std.I32)
    (j : Nat) (hj : j < 8) :
    (p.set 0#usize v0 |>.set 1#usize v1 |>.set 2#usize v2 |>.set 3#usize v3 |>.set 4#usize v4
      |>.set 5#usize v5 |>.set 6#usize v6 |>.set 7#usize v7).val.getD j 0#i32
    = ([v0, v1, v2, v3, v4, v5, v6, v7] : List Std.I32).getD j 0#i32 := by
  simp only [Array.set_val_eq]
  rw [getD_i32]
  rcases lt8_cases hj with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [Array.property]

/-- The staged low half carries the reference subset sums. -/
lemma a8_getD (p : Array Std.I32 16#usize) (c0 c1 c2 c3 x3 x5 x6 x7 : Std.I32)
    (hx3 : x3.val = c0.val + c1.val) (hx5 : x5.val = c0.val + c2.val)
    (hx6 : x6.val = c1.val + c2.val) (hx7 : x7.val = c0.val + c1.val + c2.val)
    (k : Nat) (hk : k < 8) :
    ((p.set 0#usize 0#i32 |>.set 1#usize c0 |>.set 2#usize c1 |>.set 3#usize x3
      |>.set 4#usize c2 |>.set 5#usize x5 |>.set 6#usize x6 |>.set 7#usize x7).val.getD
        k 0#i32).val
    = subsetSum c0.val c1.val c2.val c3.val k := by
  rw [arr8_getD p 0#i32 c0 c1 x3 c2 x5 x6 x7 k hk]
  rcases lt8_cases hk with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [subsetSum, Nat.testBit, hx3, hx5, hx6, hx7]

/-- The full `p` table of one group, from the two halves. -/
lemma p_of_loop (c0 c1 c2 c3 : Std.I32) (a p8 : Array Std.I32 16#usize)
    (ha : ∀ k, k < 8 → (a.val.getD k 0#i32).val = subsetSum c0.val c1.val c2.val c3.val k)
    (h8 : ∀ k, k < 8 → (p8.val.getD (8 + k) 0#i32).val = (a.val.getD k 0#i32).val + c3.val)
    (hlo : ∀ j, j < 8 → (p8.val.getD j 0#i32).val = (a.val.getD j 0#i32).val)
    (n : Nat) (hn : n < 16) :
    (p8.val.getD n 0#i32).val = subsetSum c0.val c1.val c2.val c3.val n := by
  by_cases hn8 : n < 8
  · rw [hlo n hn8]; exact ha n hn8
  · have hn_eq : n = 8 + (n - 8) := by omega
    rw [hn_eq, h8 (n - 8) (by omega), ha (n - 8) (by omega),
      subsetSum_add_eight c0.val c1.val c2.val c3.val (n - 8) (by omega)]

/-- `(x : i32) as i16` preserves the value when `x` fits `i16`. -/
lemma castI16_val (x : Std.I32) (h : -32768 ≤ x.val ∧ x.val ≤ 32767) :
    (IScalar.cast IScalarTy.I16 x).val = x.val := by
  rw [IScalar.cast_val_eq]
  simp only [IScalarTy.I16_numBits_eq, IScalarTy.I32_numBits_eq]
  rw [Nat.min_eq_left (by norm_num)]
  refine bmod_pow_numBits_eq_of_lt IScalarTy.I16 x.val ?_ ?_ <;>
    (simp only [IScalarTy.I16_numBits_eq]; omega)

/-- Every table entry of a group of `i8` codes lies in `[-512, 508]`. -/
lemma subsetSum_bound (c0 c1 c2 c3 : Int) (n : Nat)
    (h0 : -128 ≤ c0 ∧ c0 ≤ 127) (h1 : -128 ≤ c1 ∧ c1 ≤ 127)
    (h2 : -128 ≤ c2 ∧ c2 ≤ 127) (h3 : -128 ≤ c3 ∧ c3 ≤ 127) :
    -512 ≤ subsetSum c0 c1 c2 c3 n ∧ subsetSum c0 c1 c2 c3 n ≤ 508 := by
  unfold subsetSum
  split <;> split <;> split <;> split <;> omega

/-- The `i8` code `j` of the slice is `cAt` once `j` is in range. -/
lemma cAt_eq (codes : Slice Std.I8) (j : Nat) (h : j < codes.length) :
    cAt codes j = (codes.val[j]).val := by
  unfold cAt
  rw [getD_eq_getElem codes.val j 0#i8 h]

/-- Every `i8` value lies in `[-128, 127]`. -/
lemma i8_bounds (x : Std.I8) : -128 ≤ x.val ∧ x.val ≤ 127 := by
  have h := x.hBounds
  rw [IScalarTy.I8_numBits_eq] at h
  norm_num at h
  omega

/-- `i32::from(i8)` (the sign-extending `FromI32I8.from`) preserves the `i8` range. -/
lemma from_i8_bounds (x : Std.I8) :
    -128 ≤ (core.convert.num.FromI32I8.from x).val ∧
    (core.convert.num.FromI32I8.from x).val ≤ 127 := by
  rw [core.convert.num.FromI32I8.from_val_eq]
  exact i8_bounds x

/-- `List.getD` read from the `getElem?` form: the `getElem` proof never enters the goal, so no
dependent-index rewrite is needed. -/
lemma getD_of_getElem? {α : Type} (l : List α) (j : Nat) (d x : α) (h : l[j]? = some x) :
    l.getD j d = x := by
  rw [getD_eq_getElem?, h]
  rfl

/-- A code of the slice, read through `cAt`, from its `getElem?` form. -/
lemma cAt_of_getElem? (codes : Slice Std.I8) (j : Nat) (x : Std.I8)
    (h : codes.val[j]? = some x) : cAt codes j = x.val := by
  show (codes.val.getD j 0#i8).val = x.val
  rw [getD_of_getElem? codes.val j 0#i8 x h]

/-- The codes in `[lo, hi)` fit the `|x| ≤ 127` range the `+508` byte encoder needs. The Rust
documentation asserts this of the `PQ2_0` weights but performs no check, so it is a hypothesis. -/
def codeRange (codes : Slice Std.I8) (lo hi : Nat) : Prop :=
  ∀ j, lo ≤ j → j < hi → -127 ≤ cAt codes j ∧ cAt codes j ≤ 127

/-- With `|c_k| ≤ 127` every subset sum lies in `[-508, 508]`, so `P + 508` is a `u16` byte
pair. -/
lemma subsetSum_bound127 (c0 c1 c2 c3 : Int) (n : Nat)
    (h0 : -127 ≤ c0 ∧ c0 ≤ 127) (h1 : -127 ≤ c1 ∧ c1 ≤ 127)
    (h2 : -127 ≤ c2 ∧ c2 ≤ 127) (h3 : -127 ≤ c3 ∧ c3 ≤ 127) :
    -508 ≤ subsetSum c0 c1 c2 c3 n ∧ subsetSum c0 c1 c2 c3 n ≤ 508 := by
  unfold subsetSum
  split <;> split <;> split <;> split <;> omega

lemma getD_setSlice!_u8_mid (s s' : List Std.U8) (i j : Nat)
    (h : i ≤ j ∧ j - i < s'.length ∧ j < s.length) :
    (s.setSlice! i s').getD j 0#u8 = s'[j - i]! := by
  rw [getD_u8, List.getElem!_setSlice!_middle s s' i j h]

lemma getD_setSlice!_u8_pre (s s' : List Std.U8) (i j : Nat) (h : j < i) :
    (s.setSlice! i s').getD j 0#u8 = s.getD j 0#u8 := by
  rw [getD_u8, List.getElem!_setSlice!_prefix s s' i j h, ← getD_u8]

/-! ## The outer loop of `build_tables_range`

`build_tables_range_loop0` walks `g = g0 .. g1`, writing group `g`'s table into
`lo[16(g-g0) .. 16(g-g0)+16)`, `hi` likewise, and `xs[g-g0]`. The invariant tracks the written
prefix entrywise. -/

/-- The extracted outer loop's state tuple. -/
abbrev BuildState := Slice Std.U8 × Slice Std.U8 × Slice Std.I16 × Array Std.I32 16#usize ×
  Array Std.U8 16#usize × Array Std.U8 16#usize × Std.Usize

/-- The outer-loop invariant: every group `g'` with `g0 ≤ g' < g` has been written into `lo`, `hi`
(16 bytes at offset `16·(g'-g0)`) and `xs` (one `i16` at offset `g'-g0`); the three slices hold
room for the whole range `[g0, g1)`. -/
def buildInv (codes : Slice Std.I8) (g0 g1 : Std.Usize) (lo hi : Slice Std.U8)
    (xs : Slice Std.I16) (g : Std.Usize) : Prop :=
  g0.val ≤ g.val ∧ g.val ≤ g1.val ∧
  16 * (g1.val - g0.val) ≤ lo.length ∧
  16 * (g1.val - g0.val) ≤ hi.length ∧
  g1.val - g0.val ≤ xs.length ∧
  (∀ g', g0.val ≤ g' → g' < g.val → ∀ n, n < 16 →
    (lo.val.getD (16 * (g' - g0.val) + n) 0#u8).val = grpLo codes g' n) ∧
  (∀ g', g0.val ≤ g' → g' < g.val → ∀ n, n < 16 →
    (hi.val.getD (16 * (g' - g0.val) + n) 0#u8).val = grpHi codes g' n) ∧
  (∀ g', g0.val ≤ g' → g' < g.val →
    (xs.val.getD (g' - g0.val) 0#i16).val = grpX codes g')

/-- The postcondition of the outer loop over `[g0, g1)`. -/
def buildPost (codes : Slice Std.I8) (g0 g1 : Std.Usize)
    (r : Slice Std.U8 × Slice Std.U8 × Slice Std.I16) : Prop :=
  (∀ g', g0.val ≤ g' → g' < g1.val → ∀ n, n < 16 →
    (r.1.val.getD (16 * (g' - g0.val) + n) 0#u8).val = grpLo codes g' n) ∧
  (∀ g', g0.val ≤ g' → g' < g1.val → ∀ n, n < 16 →
    (r.2.1.val.getD (16 * (g' - g0.val) + n) 0#u8).val = grpHi codes g' n) ∧
  (∀ g', g0.val ≤ g' → g' < g1.val →
    (r.2.2.val.getD (g' - g0.val) 0#i16).val = grpX codes g')

/-- One step of the outer loop preserves the invariant. The three `group g` facts are what the
`p`-fill and the two byte-plane encodes produce for the group just processed. -/
lemma buildInv_step {codes : Slice Std.I8} {g0 g1 : Std.Usize}
    {lo hi : Slice Std.U8} {xs : Slice Std.I16} {g : Std.Usize}
    (lo161 hi161 : Array Std.U8 16#usize) (i14 : Std.I16) (out i13 g2 : Std.Usize)
    (hinv : buildInv codes g0 g1 lo hi xs g)
    (hg : g.val < g1.val)
    (hout : out.val = 16 * (g.val - g0.val))
    (hi13 : i13.val = g.val - g0.val)
    (hg2 : g2.val = g.val + 1)
    (hlo16 : ∀ n, n < 16 → (lo161.val.getD n 0#u8).val = grpLo codes g.val n)
    (hhi16 : ∀ n, n < 16 → (hi161.val.getD n 0#u8).val = grpHi codes g.val n)
    (hx14 : i14.val = grpX codes g.val) :
    buildInv codes g0 g1 (lo.setSlice! out.val lo161.val) (hi.setSlice! out.val hi161.val)
      (xs.set i13 i14) g2 := by
  rw [buildInv] at hinv ⊢
  obtain ⟨hg0le, hgle, hloRoom, hhiRoom, hxsRoom, hloI, hhiI, hxsI⟩ := hinv
  have hlen16 : lo161.val.length = 16 := by rw [Array.property lo161]; norm_num
  have hlen16' : hi161.val.length = 16 := by rw [Array.property hi161]; norm_num
  refine ⟨by omega, by omega, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [Slice.setSlice!_length]; exact hloRoom
  · simp only [Slice.setSlice!_length]; exact hhiRoom
  · simp only [Slice.set_length]; exact hxsRoom
  · intro g' hg0g' hg'lt n hn
    simp only [Slice.setSlice!_val]
    by_cases hlt : g' < g.val
    · have hJ : 16 * (g' - g0.val) + n < out.val := by rw [hout]; omega
      rw [getD_setSlice!_u8_pre lo.val lo161.val out.val (16 * (g' - g0.val) + n) hJ]
      exact hloI g' hg0g' hlt n hn
    · have heq : g' = g.val := by omega
      have hJ : 16 * (g' - g0.val) + n = out.val + n := by rw [heq, hout]
      have hmid : out.val + n < lo.val.length := by
        have h : 16 * (g1.val - g0.val) ≤ lo.val.length := hloRoom
        rw [hout]; omega
      rw [hJ, getD_setSlice!_u8_mid lo.val lo161.val out.val (out.val + n)
        ⟨by omega, by rw [hlen16]; omega, hmid⟩]
      rw [show out.val + n - out.val = n from by omega, ← getD_u8, heq, hlo16 n hn]
  · intro g' hg0g' hg'lt n hn
    simp only [Slice.setSlice!_val]
    by_cases hlt : g' < g.val
    · have hJ : 16 * (g' - g0.val) + n < out.val := by rw [hout]; omega
      rw [getD_setSlice!_u8_pre hi.val hi161.val out.val (16 * (g' - g0.val) + n) hJ]
      exact hhiI g' hg0g' hlt n hn
    · have heq : g' = g.val := by omega
      have hJ : 16 * (g' - g0.val) + n = out.val + n := by rw [heq, hout]
      have hmid : out.val + n < hi.val.length := by
        have h : 16 * (g1.val - g0.val) ≤ hi.val.length := hhiRoom
        rw [hout]; omega
      rw [hJ, getD_setSlice!_u8_mid hi.val hi161.val out.val (out.val + n)
        ⟨by omega, by rw [hlen16']; omega, hmid⟩]
      rw [show out.val + n - out.val = n from by omega, ← getD_u8, heq, hhi16 n hn]
  · intro g' hg0g' hg'lt
    simp only [Slice.set_val_eq]
    by_cases hlt : g' < g.val
    · have hne : ¬((g' - g0.val) = i13.val ∧ i13.val < xs.val.length) := by
        rintro ⟨h1, _⟩
        rw [hi13] at h1
        omega
      rw [getD_set, if_neg hne]
      exact hxsI g' hg0g' hlt
    · have heq : g' = g.val := by omega
      subst heq
      have hidx : g.val - g0.val = i13.val := by rw [hi13]
      have hlen : i13.val < xs.val.length := by
        rw [hi13]
        have h : g1.val - g0.val ≤ xs.val.length := hxsRoom
        omega
      rw [getD_set, if_pos ⟨hidx, hlen⟩]
      exact hx14

/-- `build_tables_range_loop0 codes g0 g1 lo hi xs p lo16 hi16 g0` writes the tables of groups
`[g0, g1)` into the matching ranges of `lo`, `hi` and `xs`. The length hypotheses are the caller's
(the Rust indexes `codes[4g..4g+4)`, `lo[16(g-g0)..)`, `hi` likewise and `xs[g-g0]`); `hcode`
carries the `|xq| ≤ 127` weight range the `+508` byte encode needs. -/
theorem word_build_tables_range_loop0_ok (codes : Slice Std.I8) (g0 g1 : Std.Usize)
    (lo hi : Slice Std.U8) (xs : Slice Std.I16)
    (p : Array Std.I32 16#usize) (lo16 hi16 : Array Std.U8 16#usize)
    (hcodes : 4 * g1.val ≤ codes.length)
    (hcode : codeRange codes (4 * g0.val) (4 * g1.val))
    (hlo : 16 * (g1.val - g0.val) ≤ lo.length)
    (hhi : 16 * (g1.val - g0.val) ≤ hi.length)
    (hxs : g1.val - g0.val ≤ xs.length)
    (hgg : g0.val ≤ g1.val) :
    ∃ lo' hi' xs',
      kern.lut.build_tables_range_loop0 codes g0 g1 lo hi xs p lo16 hi16 g0 = ok (lo', hi', xs') ∧
      buildPost codes g0 g1 (lo', hi', xs') := by
  have hspec : kern.lut.build_tables_range_loop0 codes g0 g1 lo hi xs p lo16 hi16 g0
      ⦃ r => buildPost codes g0 g1 r ⦄ := by
    unfold kern.lut.build_tables_range_loop0
    apply loop.spec_decr_nat
      (measure := fun st : BuildState => g1.val - st.2.2.2.2.2.2.val)
      (inv := fun st => buildInv codes g0 g1 st.1 st.2.1 st.2.2.1 st.2.2.2.2.2.2)
      (post := fun r : Slice Std.U8 × Slice Std.U8 × Slice Std.I16 => buildPost codes g0 g1 r)
    · rintro ⟨lo1, hi1, xs1, p0, lo160, hi160, g⟩
        ⟨hg0le, hgle, hloRoom, hhiRoom, hxsRoom, hloI, hhiI, hxsI⟩
      dsimp only at hg0le hgle hloRoom hhiRoom hxsRoom hloI hhiI hxsI
      dsimp only
      unfold kern.lut.build_tables_range_loop0.body
      by_cases hg : g < g1
      · rw [if_pos hg]
        simp only [lift, bind_tc_ok]
        step*
        · -- 1: the `p`-fill's `i32` domain
          have ha : ∀ k, k < 8 → (a.val.getD k 0#i32).val
              = subsetSum (core.convert.num.FromI32I8.from i1).val
                  (core.convert.num.FromI32I8.from i3).val
                  (core.convert.num.FromI32I8.from i5).val
                  (core.convert.num.FromI32I8.from i7).val k := by
            intro k hk
            rw [a_post, p7_post, p6_post, p5_post, p4_post, p3_post, p2_post, p1_post]
            exact a8_getD p0 (core.convert.num.FromI32I8.from i1)
              (core.convert.num.FromI32I8.from i3) (core.convert.num.FromI32I8.from i5)
              (core.convert.num.FromI32I8.from i7) i8 i9 i10 i11 (by rw [i8_post]) (by rw [i9_post])
              (by rw [i10_post]) (by rw [i11_post, i8_post]) k hk
          intro k hk
          rw [ha k hk]
          have hb := subsetSum_bound (core.convert.num.FromI32I8.from i1).val
            (core.convert.num.FromI32I8.from i3).val (core.convert.num.FromI32I8.from i5).val
            (core.convert.num.FromI32I8.from i7).val k
            (from_i8_bounds i1) (from_i8_bounds i3) (from_i8_bounds i5) (from_i8_bounds i7)
          have h7 := from_i8_bounds i7
          omega
        · -- 2: the `+508` encoder's domain
          have ha : ∀ k, k < 8 → (a.val.getD k 0#i32).val
              = subsetSum (core.convert.num.FromI32I8.from i1).val
                  (core.convert.num.FromI32I8.from i3).val
                  (core.convert.num.FromI32I8.from i5).val
                  (core.convert.num.FromI32I8.from i7).val k := by
            intro k hk
            rw [a_post, p7_post, p6_post, p5_post, p4_post, p3_post, p2_post, p1_post]
            exact a8_getD p0 (core.convert.num.FromI32I8.from i1)
              (core.convert.num.FromI32I8.from i3) (core.convert.num.FromI32I8.from i5)
              (core.convert.num.FromI32I8.from i7) i8 i9 i10 i11 (by rw [i8_post]) (by rw [i9_post])
              (by rw [i10_post]) (by rw [i11_post, i8_post]) k hk
          have hp8 : ∀ n, n < 16 → (p8.val.getD n 0#i32).val
              = subsetSum (core.convert.num.FromI32I8.from i1).val
                  (core.convert.num.FromI32I8.from i3).val
                  (core.convert.num.FromI32I8.from i5).val
                  (core.convert.num.FromI32I8.from i7).val n :=
            fun n hn => p_of_loop _ _ _ _ a p8 ha p8_post p8_post1 n hn
          have hglt : g.val < g1.val := by scalar_tac
          have hc : 4 * g1.val ≤ codes.val.length := hcodes
          have hbase : base.val = 4 * g.val := base_post
          have hq0 : cAt codes (4 * g.val) = (core.convert.num.FromI32I8.from i1).val := by
            rw [core.convert.num.FromI32I8.from_val_eq]
            refine cAt_of_getElem? codes (4 * g.val) i1 ?_
            rw [← hbase,
              List.getElem?_eq_getElem (by rw [hbase]; omega : base.val < codes.val.length),
              ← i1_post]
          have hi2 : i2.val = 4 * g.val + 1 := by rw [i2_post, hbase]
          have hq1 : cAt codes (4 * g.val + 1) = (core.convert.num.FromI32I8.from i3).val := by
            rw [core.convert.num.FromI32I8.from_val_eq]
            refine cAt_of_getElem? codes (4 * g.val + 1) i3 ?_
            rw [← hi2, List.getElem?_eq_getElem (by omega : i2.val < codes.val.length), ← i3_post]
          have hi4 : i4.val = 4 * g.val + 2 := by rw [i4_post, hbase]
          have hq2 : cAt codes (4 * g.val + 2) = (core.convert.num.FromI32I8.from i5).val := by
            rw [core.convert.num.FromI32I8.from_val_eq]
            refine cAt_of_getElem? codes (4 * g.val + 2) i5 ?_
            rw [← hi4, List.getElem?_eq_getElem (by omega : i4.val < codes.val.length), ← i5_post]
          have hi6 : i6.val = 4 * g.val + 3 := by rw [i6_post, hbase]
          have hq3 : cAt codes (4 * g.val + 3) = (core.convert.num.FromI32I8.from i7).val := by
            rw [core.convert.num.FromI32I8.from_val_eq]
            refine cAt_of_getElem? codes (4 * g.val + 3) i7 ?_
            rw [← hi6, List.getElem?_eq_getElem (by omega : i6.val < codes.val.length), ← i7_post]
          intro n hn
          rw [hp8 n hn]
          have hcb0 : -127 ≤ (core.convert.num.FromI32I8.from i1).val ∧
              (core.convert.num.FromI32I8.from i1).val ≤ 127 := by
            have h := hcode (4 * g.val) (by omega) (by omega)
            rw [hq0] at h
            exact h
          have hcb1 : -127 ≤ (core.convert.num.FromI32I8.from i3).val ∧
              (core.convert.num.FromI32I8.from i3).val ≤ 127 := by
            have h := hcode (4 * g.val + 1) (by omega) (by omega)
            rw [hq1] at h
            exact h
          have hcb2 : -127 ≤ (core.convert.num.FromI32I8.from i5).val ∧
              (core.convert.num.FromI32I8.from i5).val ≤ 127 := by
            have h := hcode (4 * g.val + 2) (by omega) (by omega)
            rw [hq2] at h
            exact h
          have hcb3 : -127 ≤ (core.convert.num.FromI32I8.from i7).val ∧
              (core.convert.num.FromI32I8.from i7).val ≤ 127 := by
            have h := hcode (4 * g.val + 3) (by omega) (by omega)
            rw [hq3] at h
            exact h
          have hs := subsetSum_bound127 (core.convert.num.FromI32I8.from i1).val
            (core.convert.num.FromI32I8.from i3).val (core.convert.num.FromI32I8.from i5).val
            (core.convert.num.FromI32I8.from i7).val n hcb0 hcb1 hcb2 hcb3
          omega
        · -- 3: the invariant is preserved and the measure decreases
          have ha : ∀ k, k < 8 → (a.val.getD k 0#i32).val
              = subsetSum (core.convert.num.FromI32I8.from i1).val
                  (core.convert.num.FromI32I8.from i3).val
                  (core.convert.num.FromI32I8.from i5).val
                  (core.convert.num.FromI32I8.from i7).val k := by
            intro k hk
            rw [a_post, p7_post, p6_post, p5_post, p4_post, p3_post, p2_post, p1_post]
            exact a8_getD p0 (core.convert.num.FromI32I8.from i1)
              (core.convert.num.FromI32I8.from i3) (core.convert.num.FromI32I8.from i5)
              (core.convert.num.FromI32I8.from i7) i8 i9 i10 i11 (by rw [i8_post]) (by rw [i9_post])
              (by rw [i10_post]) (by rw [i11_post, i8_post]) k hk
          have hp8 : ∀ n, n < 16 → (p8.val.getD n 0#i32).val
              = subsetSum (core.convert.num.FromI32I8.from i1).val
                  (core.convert.num.FromI32I8.from i3).val
                  (core.convert.num.FromI32I8.from i5).val
                  (core.convert.num.FromI32I8.from i7).val n :=
            fun n hn => p_of_loop _ _ _ _ a p8 ha p8_post p8_post1 n hn
          have hglt : g.val < g1.val := by scalar_tac
          have hc : 4 * g1.val ≤ codes.val.length := hcodes
          have hbase : base.val = 4 * g.val := base_post
          have hq0 : cAt codes (4 * g.val) = (core.convert.num.FromI32I8.from i1).val := by
            rw [core.convert.num.FromI32I8.from_val_eq]
            refine cAt_of_getElem? codes (4 * g.val) i1 ?_
            rw [← hbase,
              List.getElem?_eq_getElem (by rw [hbase]; omega : base.val < codes.val.length),
              ← i1_post]
          have hi2 : i2.val = 4 * g.val + 1 := by rw [i2_post, hbase]
          have hq1 : cAt codes (4 * g.val + 1) = (core.convert.num.FromI32I8.from i3).val := by
            rw [core.convert.num.FromI32I8.from_val_eq]
            refine cAt_of_getElem? codes (4 * g.val + 1) i3 ?_
            rw [← hi2, List.getElem?_eq_getElem (by omega : i2.val < codes.val.length), ← i3_post]
          have hi4 : i4.val = 4 * g.val + 2 := by rw [i4_post, hbase]
          have hq2 : cAt codes (4 * g.val + 2) = (core.convert.num.FromI32I8.from i5).val := by
            rw [core.convert.num.FromI32I8.from_val_eq]
            refine cAt_of_getElem? codes (4 * g.val + 2) i5 ?_
            rw [← hi4, List.getElem?_eq_getElem (by omega : i4.val < codes.val.length), ← i5_post]
          have hi6 : i6.val = 4 * g.val + 3 := by rw [i6_post, hbase]
          have hq3 : cAt codes (4 * g.val + 3) = (core.convert.num.FromI32I8.from i7).val := by
            rw [core.convert.num.FromI32I8.from_val_eq]
            refine cAt_of_getElem? codes (4 * g.val + 3) i7 ?_
            rw [← hi6, List.getElem?_eq_getElem (by omega : i6.val < codes.val.length), ← i7_post]
          have hgrp : ∀ n, grpLo codes g.val n = encLo (subsetSum
              (core.convert.num.FromI32I8.from i1).val
              (core.convert.num.FromI32I8.from i3).val
              (core.convert.num.FromI32I8.from i5).val
              (core.convert.num.FromI32I8.from i7).val n) := by
            intro n; unfold grpLo; rw [hq0, hq1, hq2, hq3]
          have hgrph : ∀ n, grpHi codes g.val n = encHi (subsetSum
              (core.convert.num.FromI32I8.from i1).val
              (core.convert.num.FromI32I8.from i3).val
              (core.convert.num.FromI32I8.from i5).val
              (core.convert.num.FromI32I8.from i7).val n) := by
            intro n; unfold grpHi; rw [hq0, hq1, hq2, hq3]
          have hlo16 : ∀ n, n < 16 →
              (lo161.val.getD n 0#u8).val = grpLo codes g.val n := by
            intro n hn
            rw [lo161_post n hn, hp8 n hn, ← hgrp n]
          have hhi16 : ∀ n, n < 16 →
              (hi161.val.getD n 0#u8).val = grpHi codes g.val n := by
            intro n hn
            rw [lo161_post1 n hn, hp8 n hn, ← hgrph n]
          have hi12 : i12.val = subsetSum (core.convert.num.FromI32I8.from i1).val
              (core.convert.num.FromI32I8.from i3).val
              (core.convert.num.FromI32I8.from i5).val
              (core.convert.num.FromI32I8.from i7).val 15 := by
            have h12 : i12.val = (p8.val.getD 15 0#i32).val := by
              rw [i12_post]
              exact congrArg IScalar.val (getD_eq_getElem p8.val 15 0#i32
                (by rw [Array.property p8]; norm_num)).symm
            rw [h12, hp8 15 (by norm_num)]
          have hbound : -32768 ≤ i12.val ∧ i12.val ≤ 32767 := by
            rw [hi12]
            have hb := subsetSum_bound (core.convert.num.FromI32I8.from i1).val
              (core.convert.num.FromI32I8.from i3).val
              (core.convert.num.FromI32I8.from i5).val
              (core.convert.num.FromI32I8.from i7).val 15
              (from_i8_bounds i1) (from_i8_bounds i3) (from_i8_bounds i5) (from_i8_bounds i7)
            omega
          have hx14 : (IScalar.cast IScalarTy.I16 i12).val = grpX codes g.val := by
            have hgrpx : grpX codes g.val = subsetSum (core.convert.num.FromI32I8.from i1).val
                (core.convert.num.FromI32I8.from i3).val
                (core.convert.num.FromI32I8.from i5).val
                (core.convert.num.FromI32I8.from i7).val 15 := by
              unfold grpX; rw [hq0, hq1, hq2, hq3]
            rw [castI16_val i12 hbound, hi12, hgrpx]
          have hout : out.val = 16 * (g.val - g0.val) := by
            rw [out_post, i_post, Nat.mul_comm]
          rw [s1_post2 s3, s3_post, Array.val_to_slice, s4_post2 s6, s6_post,
            Array.val_to_slice, s_post]
          refine ⟨buildInv_step lo161 hi161 (IScalar.cast IScalarTy.I16 i12) out i13 g2
            ⟨hg0le, hgle, hloRoom, hhiRoom, hxsRoom, hloI, hhiI, hxsI⟩ hglt hout i13_post
            g2_post hlo16 hhi16 hx14, ?_⟩
          rw [g2_post]; omega
      · rw [if_neg hg]
        simp only [WP.spec_ok]
        have hgv : g1.val ≤ g.val := by scalar_tac
        refine ⟨?_, ?_, ?_⟩
        · intro g' h1 h2 n hn; exact hloI g' h1 (Nat.lt_of_lt_of_le h2 hgv) n hn
        · intro g' h1 h2 n hn; exact hhiI g' h1 (Nat.lt_of_lt_of_le h2 hgv) n hn
        · intro g' h1 h2; exact hxsI g' h1 (Nat.lt_of_lt_of_le h2 hgv)
    · refine ⟨le_refl _, hgg, hlo, hhi, hxs, ?_, ?_, ?_⟩
      · intro g' h1 h2 n hn; exact absurd h2 (Nat.not_lt.mpr h1)
      · intro g' h1 h2 n hn; exact absurd h2 (Nat.not_lt.mpr h1)
      · intro g' h1 h2; exact absurd h2 (Nat.not_lt.mpr h1)
  rcases WP.spec_imp_exists hspec with ⟨r, hr, h1, h2, h3⟩
  exact ⟨r.1, r.2.1, r.2.2, hr, h1, h2, h3⟩

/-! ## Non-vacuity

Both ends of the `[-508, 508]` subset-sum bound are attained, `buildInv` holds at the initial
state, and the `codeRange` hypothesis is inhabited by a concrete group. -/

/-- The `-508` floor is attained by four `-127` codes. -/
example : subsetSum (-127) (-127) (-127) (-127) 15 = -508 := by
  simp [subsetSum_15]

/-- The `508` ceiling is attained by four `127` codes. -/
example : subsetSum 127 127 127 127 15 = 508 := by
  simp [subsetSum_15]

/-- Four `127` codes satisfy the `+508` precondition. -/
example : codeRange (Slice.from [127#i8, 127#i8, 127#i8, 127#i8] (by scalar_tac)) 0 4 := by
  intro j _ hj4
  rcases lt4_cases hj4 with rfl | rfl | rfl | rfl <;> norm_num [cAt, Slice.from_val]

/-- `buildInv` is inhabited at the initial state (`g = g0`), where the written-prefix clauses are
vacuous. -/
example (codes : Slice Std.I8) (lo hi : Slice Std.U8) (xs : Slice Std.I16) :
    buildInv codes 0#usize 0#usize lo hi xs 0#usize := by
  refine ⟨by simp, by simp, by simp, by simp, by simp, ?_, ?_, ?_⟩ <;>
    (intro g' h1 h2; exact absurd h2 (Nat.not_lt.mpr h1))

end apeiron_word

#print axioms apeiron_word.word_build_tables_range_loop0_ok
