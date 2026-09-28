/-
Exact lookup-table decomposition of the `PQ2_0` row dot product.

The `PQ2_0` row weight is a two-bit code over a `Q32.32` scale: `pq2_0Code` reads the four two-bit
fields of each `qs` byte and `pq2_0Weight` is `(code - 1)·d`. This module states that dot product
exactly as a table lookup over 4-activation groups — the T-MAC bit-serial decomposition (Wei et al.,
arXiv `2407.00088` §3.1) — and proves the two T-MAC consolidation identities used by that scheme:

- `block_dot_lut` / `row_dot_lut`: the row dot regroups into one `pq2_0ScaleRef raw b` times a sum of
  four-entry `lutEntry` tables, one per `qs` byte (`j < 32`).
- `dequant_dot_lut`: the same identity read off the extracted `dequant_pq2_0_row` values.
- `lutEntry_bitplane`: each four-entry table is two two-entry bit-plane sums minus the group sum, its
  `2^g = 16`-entry form split into the even/odd bit planes.
- `lutEntry_mirror`: on ternary bytes `0xAA - b` is the byte mirror and negates the table.
- `planeSum_bound` / `lutEntry_bound`: the plane and table values fit `i16` (`|v| ≤ 508`) and hence the
  byte split `v = (v % 256) + 256·(v / 256)` with `0 ≤ v + 508 < 1024`.

All sums are exact integer arithmetic over `ℤ`; nothing here refines a Rust body except
`dequant_dot_lut`, which composes the extracted `dequant_pq2_0_row` row law with `row_dot_lut`.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` and `omega` act only on closed
Nat literals and linear Nat arithmetic, never on an extracted body or an unbounded kernel reduction.
-/
import Numeric.DequantQ1Q2
import Mathlib

namespace apeiron_word

namespace LutPQ2

open Aeneas Aeneas.Std Result

open scoped BigOperators

noncomputable section

/-! ## Definitions -/

/-- The `k`-th two-bit field of a byte, `k < 4`: `(b >>> 2k) &&& 3`. -/
def field (b k : Nat) : Nat := (b >>> (2 * k)) &&& 3

/-- Bit `i` of `b`: `(b >>> i) &&& 1`. -/
def bit (b i : Nat) : Nat := (b >>> i) &&& 1

/-- The low (even-index) bit plane of `b` packed into a nibble: `Σ_k bit(b,2k)·2^k`. -/
def loNib (b : Nat) : Nat := ∑ k : Fin 4, bit b (2 * (k : Nat)) * 2 ^ (k : Nat)

/-- The high (odd-index) bit plane of `b` packed into a nibble: `Σ_k bit(b,2k+1)·2^k`. -/
def hiNib (b : Nat) : Nat := ∑ k : Fin 4, bit b (2 * (k : Nat) + 1) * 2 ^ (k : Nat)

/-- The four-activation bit-plane partial sum: `Σ_k bit(n,k)·x k`. -/
def planeSum (x : Fin 4 → ℤ) (n : Nat) : ℤ :=
  ∑ k : Fin 4, (((n >>> (k : Nat)) &&& 1 : Nat) : ℤ) * x k

/-- The four-entry table row of byte `b`: `Σ_k ((field b k) - 1)·x k`. -/
def lutEntry (x : Fin 4 → ℤ) (b : Nat) : ℤ :=
  ∑ k : Fin 4, ((field b (k : Nat) : ℤ) - 1) * x k

/-- A byte whose every two-bit field is a ternary code in `{0,1,2}`. -/
def ternaryByte (b : Nat) : Prop := b < 256 ∧ ∀ k : Fin 4, field b (k : Nat) ≤ 2

/-! ## The `PQ2_0` weight group -/

/-- **Item 1.** Weight `128·b + 4·j + k` is the `k`-th field of `qs` byte `j` of block `b`, times the
block scale: `((field (pq2_0CodeByte raw b j) k) - 1)·pq2_0ScaleRef raw b`. -/
lemma pq2_0Weight_group (raw : Slice U8) (b j k : Nat) (hj : j < 32) (hk : k < 4) :
    pq2_0Weight raw (128 * b + 4 * j + k) =
      ((field (pq2_0CodeByte raw b j) k : ℤ) - 1) * pq2_0ScaleRef raw b := by
  have hcode : pq2_0Code raw b (4 * j + k) = field (pq2_0CodeByte raw b j) k := by
    unfold pq2_0Code field
    rw [show (4 * j + k) / 4 = j by omega, show (4 * j + k) % 4 = k by omega]
  unfold pq2_0Weight
  rw [show (128 * b + 4 * j + k) / 128 = b by omega,
    show (128 * b + 4 * j + k) % 128 = 4 * j + k by omega, hcode]

/-! ## Regrouping helpers -/

/-- A `range` sum over `m·n` regroups into `m` blocks of `n`. -/
lemma sum_range_mul (m n : Nat) (g : Nat → ℤ) :
    (∑ l ∈ Finset.range (m * n), g l) =
      ∑ j ∈ Finset.range m, ∑ k ∈ Finset.range n, g (n * j + k) := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [Nat.succ_mul, Finset.sum_range_add, ih, Finset.sum_range_succ]
    congr 1
    apply Finset.sum_congr rfl
    intro k _
    rw [Nat.mul_comm m n]

/-! ## Block and row dot decompositions -/

/-- **Item 2.** One block: the 128-weight dot with `x` is `pq2_0ScaleRef raw b` times the sum of the
thirty-two four-entry `lutEntry` tables, one per `qs` byte. -/
theorem block_dot_lut (raw : Slice U8) (b : Nat) (x : Nat → ℤ) :
    (∑ l ∈ Finset.range 128, pq2_0Weight raw (128 * b + l) * x (128 * b + l)) =
      pq2_0ScaleRef raw b *
        ∑ j ∈ Finset.range 32,
          lutEntry (fun k => x (128 * b + 4 * j + (k : Nat))) (pq2_0CodeByte raw b j) := by
  rw [show (128 : Nat) = 32 * 4 by norm_num, sum_range_mul]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  have hj32 : j < 32 := Finset.mem_range.mp hj
  have hinner :
      (∑ k ∈ Finset.range 4,
          pq2_0Weight raw (128 * b + (4 * j + k)) * x (128 * b + (4 * j + k))) =
        pq2_0ScaleRef raw b *
          ∑ k ∈ Finset.range 4,
            ((field (pq2_0CodeByte raw b j) k : ℤ) - 1) * x (128 * b + 4 * j + k) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k hk
    have hk4 : k < 4 := Finset.mem_range.mp hk
    rw [show 128 * b + (4 * j + k) = 128 * b + 4 * j + k by ring,
      pq2_0Weight_group raw b j k hj32 hk4]
    ring
  rw [hinner]
  congr 1

/-- **Item 3.** `B` whole blocks: the `128·B`-weight dot is the sum over blocks of item 2. -/
theorem row_dot_lut (raw : Slice U8) (B : Nat) (x : Nat → ℤ) :
    (∑ i ∈ Finset.range (128 * B), pq2_0Weight raw i * x i) =
      ∑ b ∈ Finset.range B, pq2_0ScaleRef raw b *
        ∑ j ∈ Finset.range 32,
          lutEntry (fun k => x (128 * b + 4 * j + (k : Nat))) (pq2_0CodeByte raw b j) := by
  rw [show 128 * B = B * 128 by ring, sum_range_mul B 128]
  apply Finset.sum_congr rfl
  intro b _
  rw [block_dot_lut]

/-- The dot of a decoded value list with `x`: `Σ_{i < l.length} l[i]·x i` (`0` off the end). -/
def listDot (l : List ℤ) (x : Nat → ℤ) : ℤ :=
  ∑ i ∈ Finset.range l.length, l.getD i 0 * x i

/-- The row-value list's dot is the raw-weight dot. -/
lemma listDot_pq2_0RowValues (raw : Slice U8) (n : Nat) (x : Nat → ℤ) :
    listDot (pq2_0RowValues raw n) x = ∑ i ∈ Finset.range n, pq2_0Weight raw i * x i := by
  unfold listDot pq2_0RowValues
  rw [List.length_map, List.length_range]
  apply Finset.sum_congr rfl
  intro i hi
  have hi' : i < n := Finset.mem_range.mp hi
  rw [List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_range (by omega)]
  rfl

/-- **Item 3 bridge.** Under the `dequant_pq2_0_row_spec` hypotheses with empty output and
`n = 128·B`, the extracted `dequant_pq2_0_row` returns `out1` whose decoded values, dotted with `x`,
equal the block sum of `pq2_0ScaleRef raw b` times the `lutEntry` tables. -/
theorem dequant_dot_lut (raw : Slice U8) (B : Nat) (n : Usize) (hn : n.val = 128 * B)
    (x : Nat → ℤ)
    (hdom : ∀ b : Usize, b.val < (n.val + 127) / 128 → pq2_0BlockDomain raw b)
    (hcap : (alloc.vec.Vec.new I64).val.length + n.val ≤ Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.lowbit.dequant_pq2_0_row raw n (alloc.vec.Vec.new I64) = ok (.Ok (), out1) ∧
      listDot (dequantVals out1.val) x =
        ∑ b ∈ Finset.range B, pq2_0ScaleRef raw b *
          ∑ j ∈ Finset.range 32,
            lutEntry (fun k => x (128 * b + 4 * j + (k : Nat))) (pq2_0CodeByte raw b j) := by
  obtain ⟨out1, hrow, hvals⟩ :=
    dequant_pq2_0_row_spec raw n (alloc.vec.Vec.new I64) hdom hcap
  have hvals' : dequantVals out1.val = pq2_0RowValues raw n.val := by
    rw [hvals, q8_0_vec_new_val_i64]
    simp [dequantVals]
  refine ⟨out1, hrow, ?_⟩
  rw [hvals', hn, listDot_pq2_0RowValues, row_dot_lut]

/-! ## Bit-plane and mirror identities -/

/-- `field b k` is the base-4 digit `(b / 4^k) % 4`. -/
lemma field_eq_mod (b k : Nat) : field b k = (b / 4 ^ k) % 4 := by
  unfold field
  rw [Nat.shiftRight_eq_div_pow]
  have h2 : (2 : Nat) ^ (2 * k) = 4 ^ k := by
    rw [show (4 : Nat) = 2 ^ 2 by norm_num, ← pow_mul]
  rw [h2, show (3 : Nat) = 2 ^ 2 - 1 by norm_num, Nat.and_two_pow_sub_one_eq_mod]

/-- A single bit is `0` or `1`. -/
lemma bit_le_one (b i : Nat) : bit b i ≤ 1 := by
  unfold bit
  have := Nat.and_lt_two_pow (b >>> i) (y := 1) (n := 1) (by norm_num)
  omega

/-- A closed four-bit binary reconstruction: bit `j` of `Σ_k c k·2^k` is `c j` for `c k ≤ 1`. -/
lemma bit_sum_four_fin : ∀ c : Fin 4 → Fin 2, ∀ j : Fin 4,
    ((∑ k : Fin 4, (c k : Nat) * 2 ^ (k : Nat)) >>> (j : Nat)) &&& 1 = (c j : Nat) := by
  decide

/-- Bit `j` of a four-term binary sum with coefficients `≤ 1`. -/
lemma bit_sum_four (c : Fin 4 → Nat) (hc : ∀ k, c k ≤ 1) (j : Fin 4) :
    ((∑ k : Fin 4, c k * 2 ^ (k : Nat)) >>> (j : Nat)) &&& 1 = c j := by
  have := bit_sum_four_fin (fun k => (⟨c k, by have := hc k; omega⟩ : Fin 2)) j
  simpa using this

/-- Bit `j` of `hiNib b` is `bit b (2j+1)`. -/
lemma hiNib_testBit (b : Nat) (j : Fin 4) :
    (hiNib b >>> (j : Nat)) &&& 1 = bit b (2 * (j : Nat) + 1) := by
  unfold hiNib
  exact bit_sum_four (fun k => bit b (2 * (k : Nat) + 1)) (fun k => bit_le_one b _) j

/-- Bit `j` of `loNib b` is `bit b (2j)`. -/
lemma loNib_testBit (b : Nat) (j : Fin 4) :
    (loNib b >>> (j : Nat)) &&& 1 = bit b (2 * (j : Nat)) := by
  unfold loNib
  exact bit_sum_four (fun k => bit b (2 * (k : Nat))) (fun k => bit_le_one b _) j

/-- The high-plane sum over `hiNib b` unfolds to the odd-bit weighted sum. -/
lemma planeSum_hiNib (x : Fin 4 → ℤ) (b : Nat) :
    planeSum x (hiNib b) = ∑ k : Fin 4, ((bit b (2 * (k : Nat) + 1) : Nat) : ℤ) * x k := by
  unfold planeSum
  apply Finset.sum_congr rfl
  intro k _
  rw [hiNib_testBit b k]

/-- The low-plane sum over `loNib b` unfolds to the even-bit weighted sum. -/
lemma planeSum_loNib (x : Fin 4 → ℤ) (b : Nat) :
    planeSum x (loNib b) = ∑ k : Fin 4, ((bit b (2 * (k : Nat)) : Nat) : ℤ) * x k := by
  unfold planeSum
  apply Finset.sum_congr rfl
  intro k _
  rw [loNib_testBit b k]

/-- A field is its two bits: `field b k = 2·bit(b,2k+1) + bit(b,2k)`. -/
lemma field_eq_two_bit (b k : Nat) : field b k = 2 * bit b (2 * k + 1) + bit b (2 * k) := by
  rw [field_eq_mod]
  unfold bit
  rw [Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow, Nat.and_one_is_mod,
    Nat.and_one_is_mod]
  have hpow : (2 : Nat) ^ (2 * k) = 4 ^ k := by
    rw [show (4 : Nat) = 2 ^ 2 by norm_num, ← pow_mul]
  have hpow' : (2 : Nat) ^ (2 * k + 1) = 4 ^ k * 2 := by
    rw [pow_succ, hpow]
  rw [hpow, hpow']
  have hdiv : b / (4 ^ k * 2) = b / 4 ^ k / 2 := (Nat.div_div_eq_div_mul b (4 ^ k) 2).symm
  rw [hdiv]
  omega

/-- **Item 4.** The table row is the two bit-plane sums minus the group sum:
`lutEntry x b = 2·planeSum x (hiNib b) + planeSum x (loNib b) - Σ_k x k`. -/
theorem lutEntry_bitplane (x : Fin 4 → ℤ) (b : Nat) (_ : b < 256) :
    lutEntry x b = 2 * planeSum x (hiNib b) + planeSum x (loNib b) - ∑ k : Fin 4, x k := by
  rw [planeSum_hiNib, planeSum_loNib]
  unfold lutEntry
  rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro k _
  rw [field_eq_two_bit]
  push_cast
  ring

/-- The base-4 reconstruction of a byte: `b = Σ_k field b k · 4^k` for `b < 256`. -/
lemma base4_repr (b : Nat) (hb : b < 256) :
    b = ∑ k : Fin 4, field b k * 4 ^ (k : Nat) := by
  rw [Fin.sum_univ_four]
  simp only [field_eq_mod]
  norm_num
  omega

/-- Base-4 digit extraction at position `0`. -/
lemma field_base4_0 (g : Fin 4 → Nat) (hg : ∀ k, g k ≤ 3) :
    field (∑ k : Fin 4, g k * 4 ^ (k : Nat)) 0 = g 0 := by
  have h0 := hg 0; have h1 := hg 1; have h2 := hg 2; have h3 := hg 3
  rw [field_eq_mod, Fin.sum_univ_four]; norm_num; omega

/-- Base-4 digit extraction at position `1`. -/
lemma field_base4_1 (g : Fin 4 → Nat) (hg : ∀ k, g k ≤ 3) :
    field (∑ k : Fin 4, g k * 4 ^ (k : Nat)) 1 = g 1 := by
  have h0 := hg 0; have h1 := hg 1; have h2 := hg 2; have h3 := hg 3
  rw [field_eq_mod, Fin.sum_univ_four]; norm_num; omega

/-- Base-4 digit extraction at position `2`. -/
lemma field_base4_2 (g : Fin 4 → Nat) (hg : ∀ k, g k ≤ 3) :
    field (∑ k : Fin 4, g k * 4 ^ (k : Nat)) 2 = g 2 := by
  have h0 := hg 0; have h1 := hg 1; have h2 := hg 2; have h3 := hg 3
  rw [field_eq_mod, Fin.sum_univ_four]; norm_num; omega

/-- Base-4 digit extraction at position `3`. -/
lemma field_base4_3 (g : Fin 4 → Nat) (hg : ∀ k, g k ≤ 3) :
    field (∑ k : Fin 4, g k * 4 ^ (k : Nat)) 3 = g 3 := by
  have h0 := hg 0; have h1 := hg 1; have h2 := hg 2; have h3 := hg 3
  rw [field_eq_mod, Fin.sum_univ_four]; norm_num; omega

/-- Base-4 digit extraction: the `j`-th field of `Σ_k g k · 4^k` is `g j` for `g k ≤ 3`. -/
lemma field_base4 (g : Fin 4 → Nat) (hg : ∀ k, g k ≤ 3) (j : Fin 4) :
    field (∑ k : Fin 4, g k * 4 ^ (k : Nat)) (j : Nat) = g j := by
  fin_cases j
  · exact field_base4_0 g hg
  · exact field_base4_1 g hg
  · exact field_base4_2 g hg
  · exact field_base4_3 g hg

/-- Mirror: on a ternary byte `b`, `field (0xAA - b) k = 2 - field b k` with no borrow. -/
lemma field_mirror (b : Nat) (hb : ternaryByte b) (k : Fin 4) :
    field (0xAA - b) (k : Nat) = 2 - field b (k : Nat) := by
  have hb4 : b =
      field b 0 * 4 ^ 0 + field b 1 * 4 ^ 1 + field b 2 * 4 ^ 2 + field b 3 * 4 ^ 3 := by
    conv_lhs => rw [base4_repr b hb.1]
    rw [Fin.sum_univ_four]
    simp only [show ((0 : Fin 4) : Nat) = 0 from rfl, show ((1 : Fin 4) : Nat) = 1 from rfl,
      show ((2 : Fin 4) : Nat) = 2 from rfl, show ((3 : Fin 4) : Nat) = 3 from rfl]
  have hsub : 0xAA - b = ∑ j : Fin 4, (2 - field b (j : Nat)) * 4 ^ (j : Nat) := by
    conv_lhs => rw [hb4]
    rw [Fin.sum_univ_four]
    simp only [show ((0 : Fin 4) : Nat) = 0 from rfl, show ((1 : Fin 4) : Nat) = 1 from rfl,
      show ((2 : Fin 4) : Nat) = 2 from rfl, show ((3 : Fin 4) : Nat) = 3 from rfl]
    have h0 : field b 0 ≤ 2 := by simpa using hb.2 0
    have h1 : field b 1 ≤ 2 := by simpa using hb.2 1
    have h2 : field b 2 ≤ 2 := by simpa using hb.2 2
    have h3 : field b 3 ≤ 2 := by simpa using hb.2 3
    norm_num
    omega
  rw [hsub]
  exact field_base4 (fun j => 2 - field b (j : Nat)) (fun j => by have := hb.2 j; omega) k

/-- **Item 5.** A ternary byte mirrors: `0xAA - b` is ternary and `lutEntry x (0xAA - b) =
-lutEntry x b`. -/
theorem lutEntry_mirror (x : Fin 4 → ℤ) (b : Nat) (hb : ternaryByte b) :
    ternaryByte (0xAA - b) ∧ lutEntry x (0xAA - b) = - lutEntry x b := by
  refine ⟨?_, ?_⟩
  · refine ⟨by omega, ?_⟩
    intro k
    rw [field_mirror b hb k]
    have := hb.2 k
    omega
  · unfold lutEntry
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro k _
    rw [field_mirror b hb k]
    have hf : field b (k : Nat) ≤ 2 := hb.2 k
    rw [Nat.cast_sub (by omega : field b (k : Nat) ≤ 2)]
    ring

/-! ## Bounds -/

/-- **Item 6.** A bit-plane sum of `|x k| ≤ 127` values is bounded by `508`. -/
theorem planeSum_bound (x : Fin 4 → ℤ) (n : Nat) (hx : ∀ k, |x k| ≤ 127) :
    |planeSum x n| ≤ 508 := by
  unfold planeSum
  calc |∑ k : Fin 4, (((n >>> (k : Nat)) &&& 1 : Nat) : ℤ) * x k|
      ≤ ∑ k : Fin 4, |(((n >>> (k : Nat)) &&& 1 : Nat) : ℤ) * x k| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _k : Fin 4, (127 : ℤ) := by
        apply Finset.sum_le_sum
        intro k _
        rw [abs_mul]
        have hbitle : |(((n >>> (k : Nat)) &&& 1 : Nat) : ℤ)| ≤ 1 := by
          rw [abs_of_nonneg (by positivity)]
          have hlt : ((n >>> (k : Nat)) &&& 1) < 2 := by
            have := Nat.and_lt_two_pow (n >>> (k : Nat)) (y := 1) (n := 1) (by norm_num)
            simpa using this
          have : ((n >>> (k : Nat)) &&& 1 : Nat) ≤ 1 := by omega
          exact_mod_cast this
        calc |(((n >>> (k : Nat)) &&& 1 : Nat) : ℤ)| * |x k|
            ≤ 1 * 127 := mul_le_mul hbitle (hx k) (abs_nonneg _) (by norm_num)
          _ = 127 := by ring
    _ = 508 := by simp

/-- **Item 6.** A ternary table row of `|x k| ≤ 127` values is bounded by `508` (fits `i16`). -/
theorem lutEntry_bound (x : Fin 4 → ℤ) (b : Nat) (hb : ternaryByte b)
    (hx : ∀ k, |x k| ≤ 127) : |lutEntry x b| ≤ 508 := by
  unfold lutEntry
  calc |∑ k : Fin 4, ((field b (k : Nat) : ℤ) - 1) * x k|
      ≤ ∑ k : Fin 4, |((field b (k : Nat) : ℤ) - 1) * x k| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _k : Fin 4, (127 : ℤ) := by
        apply Finset.sum_le_sum
        intro k _
        rw [abs_mul]
        have hcoef : |((field b (k : Nat) : ℤ) - 1)| ≤ 1 := by
          have hle : (field b (k : Nat) : ℤ) ≤ 2 := by exact_mod_cast hb.2 k
          rw [abs_le]
          constructor <;> omega
        calc |((field b (k : Nat) : ℤ) - 1)| * |x k|
            ≤ 1 * 127 := mul_le_mul hcoef (hx k) (abs_nonneg _) (by norm_num)
          _ = 127 := by ring
    _ = 508 := by simp

/-- **Item 6.** The shifted plane sum lies in `[0, 1024)` for the byte split
`v = (v % 256) + 256·(v / 256)`. -/
theorem planeSum_shift_bound (x : Fin 4 → ℤ) (n : Nat) (hx : ∀ k, |x k| ≤ 127) :
    0 ≤ planeSum x n + 508 ∧ planeSum x n + 508 < 1024 := by
  have h := planeSum_bound x n hx
  rw [abs_le] at h
  constructor <;> omega

end

end LutPQ2

end apeiron_word
