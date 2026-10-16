/-
Integer refinement of the extracted `PQ2_0` lookup-table kernels (`src/engine/kern/lut.rs`),
group `kern.lut`.

The Rust is the serving kernel of the chart `N`-projection: `split_nibbles` unpacks a byte into its
even- and odd-indexed bit planes, `build_tables_range` builds the per-group activation table
(`p` coefficients, then the `P + 508` byte planes `lo`/`hi`), `block_inners_portable` and
`tile_accs_portable` accumulate the row dot, and the `*_bound_ok` predicates choose the accumulator
width.

Each theorem states what the extracted body computes against a plain-`Nat`/`Int` reference that does
not restate the Rust loop. Grade: `proved` for the named theorem, `tested` for the `cargo test` in
`src/engine/kern/lut_tests.rs`.

Boundaries stated, not papered over:

* `block_inners_avx2` and `tile_accs_avx2` are extracted `axiom`s (there is no body to refine);
* `lut_bound_ok` / `lut_i64_bound_ok` / `lut_row_i64_bound_ok` call the unmodeled externals
  `core.num.I128.saturating_mul` and `core.num.I128.abs` (`axiom` in the extraction, with no spec in
  the Aeneas model), so no concrete value for them follows from the extraction;
* the `F32`/`F64` carriers and the `unsafe`/AVX2 paths are out of scope.
-/
import ApeironWord

set_option autoImplicit false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## `kern.lut.split_nibbles`

`split_nibbles byte` gathers the four even-indexed bits of `byte` into `lo` and the four odd-indexed
bits into `hi`: bit `c` of `lo` is bit `2c` of `byte`, bit `c` of `hi` is bit `2c+1`. The reference
below is that statement in closed form, as a sum of disjoint bit contributions. -/

/-- Bit `2c+parity` of `b` (the `c`-th output bit of the interleave). -/
def splitBit (b : Nat) (parity c : Nat) : Nat := (b >>> (2 * c + parity)) % 2

/-- Partial interleave: the first `c` output bits as a value, `Σ_{c'<c} splitBit b parity c'·2^{c'}`. -/
def splitPart (b : Nat) (parity : Nat) : Nat → Nat
  | 0 => 0
  | c + 1 => splitPart b parity c + splitBit b parity c * 2 ^ c

/-- The low nibble: bit `c` of the result is bit `2c` of `b`. -/
def splitLo (b : Nat) : Nat := splitPart b 0 4

/-- The high nibble: bit `c` of the result is bit `2c+1` of `b`. -/
def splitHi (b : Nat) : Nat := splitPart b 1 4

lemma splitPart_zero_right (b parity : Nat) : splitPart b parity 0 = 0 := rfl

/-- The bit values are binary. -/
lemma splitBit_lt_two (b parity c : Nat) : splitBit b parity c < 2 :=
  Nat.mod_lt _ (by norm_num)

lemma splitPart_lt (b parity c : Nat) : splitPart b parity c < 2 ^ c := by
  induction c with
  | zero => simp [splitPart_zero_right]
  | succ c ih =>
    have h2 : splitBit b parity c * 2 ^ c ≤ 2 ^ c := by
      have h1 : splitBit b parity c ≤ 1 := Nat.le_of_lt_succ (splitBit_lt_two b parity c)
      calc splitBit b parity c * 2 ^ c ≤ 1 * 2 ^ c := Nat.mul_le_mul_right _ h1
        _ = 2 ^ c := one_mul _
    have hsum : splitPart b parity c + splitBit b parity c * 2 ^ c < 2 ^ c + 2 ^ c :=
      Nat.add_lt_add_of_lt_of_le ih h2
    rw [splitPart, pow_succ]
    calc splitPart b parity c + splitBit b parity c * 2 ^ c < 2 ^ c + 2 ^ c := hsum
      _ = 2 ^ c * 2 := by ring

/-- Or-ing one new high bit onto a value that fits below it is addition. -/
lemma split_or_add (p x c : Nat) (hp : p < 2 ^ c) :
    p ||| (x <<< c) = p + x <<< c := by
  rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt hp x, Nat.add_comm]

/-- One loop iteration adds the `c`-th output bit at position `c`. -/
lemma splitPart_succ (b parity c : Nat) :
    splitPart b parity (c + 1) = splitPart b parity c + splitBit b parity c * 2 ^ c := rfl

/-- `split_nibbles byte = ok (lo, hi)` with `lo = splitLo byte.val`, `hi = splitHi byte.val`
(the Rust loop `split_nibbles_loop` with its `step*`-discharged `i + 1`/shift side conditions). -/
theorem word_split_nibbles_ok (byte : Std.U8) :
    ∃ lo hi : Std.U8,
      kern.lut.split_nibbles byte = ok (lo, hi) ∧
      lo.val = splitLo byte.val ∧ hi.val = splitHi byte.val := by
  have hspec : kern.lut.split_nibbles byte ⦃ r =>
      r.1.val = splitLo byte.val ∧ r.2.val = splitHi byte.val ⦄ := by
    unfold kern.lut.split_nibbles kern.lut.split_nibbles_loop
    apply loop.spec_decr_nat
      (measure := fun st : Std.U8 × Std.U8 × Std.Usize => 4 - st.2.2.val)
      (inv := fun st => st.1.val = splitPart byte.val 0 st.2.2.val ∧
        st.2.2.val ≤ 4 ∧ st.2.1.val = splitPart byte.val 1 st.2.2.val)
      (post := fun r : Std.U8 × Std.U8 => r.1.val = splitLo byte.val ∧ r.2.val = splitHi byte.val)
    · rintro ⟨lo, hi, c⟩ ⟨hlo, hc4, hhi⟩
      dsimp only
      unfold kern.lut.split_nibbles_loop.body
      by_cases hc : c < 4#usize
      · rw [if_pos hc]
        simp only [lift, bind_tc_ok]
        step*
        have hcv : c.val < 4 := by scalar_tac
        have hone : (1#u8 : Std.U8).val = 1 := by decide
        have hbit0 : (i1 &&& 1#u8).val = splitBit byte.val 0 c.val := by
          rw [UScalar.val_and, hone, i1_post, i_post, Nat.and_one_is_mod, splitBit]
          norm_num
        have hbit1 : (i5 &&& 1#u8).val = splitBit byte.val 1 c.val := by
          rw [UScalar.val_and, hone, i5_post, i4_post, i_post, Nat.and_one_is_mod, splitBit]
        have hsz : U8.size = 256 := by simp [U8.size, U8.numBits]
        have hpow : ∀ k : Nat, splitBit byte.val k c.val * 2 ^ c.val ≤ 2 ^ c.val := by
          intro k
          have hb : splitBit byte.val k c.val ≤ 1 :=
            Nat.le_of_lt_succ (splitBit_lt_two byte.val k c.val)
          calc splitBit byte.val k c.val * 2 ^ c.val ≤ 1 * 2 ^ c.val := Nat.mul_le_mul_right _ hb
            _ = 2 ^ c.val := one_mul _
        have hlt256 : 2 ^ c.val < 256 := by
          calc 2 ^ c.val ≤ 2 ^ 3 := Nat.pow_le_pow_right (by norm_num) (by omega)
            _ = 8 := by norm_num
            _ < 256 := by norm_num
        have hi3 : i3.val = splitBit byte.val 0 c.val <<< c.val := by
          rw [i3_post, hbit0, hsz, Nat.shiftLeft_eq, Nat.mod_eq_of_lt]
          exact lt_of_le_of_lt (hpow 0) hlt256
        have hi7 : i7.val = splitBit byte.val 1 c.val <<< c.val := by
          rw [i7_post, hbit1, hsz, Nat.shiftLeft_eq, Nat.mod_eq_of_lt]
          exact lt_of_le_of_lt (hpow 1) hlt256
        refine ⟨?_, by rw [c1_post]; omega, ?_, by rw [c1_post]; omega⟩
        · rw [UScalar.val_or, hlo, hi3, split_or_add _ _ _ (splitPart_lt byte.val 0 c.val),
              Nat.shiftLeft_eq, c1_post, ← splitPart_succ]
        · rw [UScalar.val_or, hhi, hi7, split_or_add _ _ _ (splitPart_lt byte.val 1 c.val),
              Nat.shiftLeft_eq, c1_post, ← splitPart_succ]
      · rw [if_neg hc]
        have hc_eq : c.val = 4 := by scalar_tac
        rw [hc_eq] at hlo hhi
        exact ⟨by simpa [splitLo] using hlo, by simpa [splitHi] using hhi⟩
    · exact ⟨rfl, by norm_num, rfl⟩
  rcases WP.spec_imp_exists hspec with ⟨r, hr, h1, h2⟩
  exact ⟨r.1, r.2, hr, h1, h2⟩

/-! ## `kern.lut.build_tables_range` loop 2 (the `P + 508` byte-plane encoder)

`build_tables_range_loop0_loop1 p lo16 hi16 0` walks `n = 0..15`, writes into `lo16[n]` the low byte
and into `hi16[n]` the high byte of `p[n] + 508` (little-endian 16-bit split). The reference is
`encLo`/`encHi`. -/

/-- Low plane byte of the entry value plus the `508` offset. -/
def encLo (v : Int) : Nat := (v + 508).toNat % 256

/-- High plane byte of the entry value plus the `508` offset. -/
def encHi (v : Int) : Nat := (v + 508).toNat / 256

lemma getD_set {α : Type} (l : List α) (j i : Nat) (x d : α) :
    (l.set j x).getD i d = if i = j ∧ j < l.length then x else l.getD i d := by
  rw [List.getD, List.getElem?_set]
  by_cases hij : i = j
  · subst i
    rw [if_pos rfl]
    by_cases hj : j < l.length
    · rw [if_pos hj, if_pos ⟨rfl, hj⟩]
      rfl
    · rw [if_neg hj, if_neg (fun h => hj h.2)]
      rw [List.getD, List.getElem?_eq_none (by omega)]
  · rw [if_neg (fun h => hij h.symm), if_neg (fun h => hij h.1)]
    rfl

/-- Loop invariant for the `P + 508` byte-plane encoder: the first `n` entries are written. -/
def encInv (p : Array Std.I32 16#usize) (lo16 hi16 : Array Std.U8 16#usize) (n : Nat) : Prop :=
  n ≤ 16 ∧
  (∀ k, k < n → (lo16.val.getD k 0#u8).val = encLo (p.val.getD k 0#i32).val) ∧
  (∀ k, k < n → (hi16.val.getD k 0#u8).val = encHi (p.val.getD k 0#i32).val)

lemma encInv_zero (p : Array Std.I32 16#usize) (lo16 hi16 : Array Std.U8 16#usize) :
    encInv p lo16 hi16 0 := ⟨by norm_num, by omega, by omega⟩

lemma getD_i32_lt (v : Array Std.I32 16#usize) (k : Nat) (hk : k < 16) :
    (v.val.getD k 0#i32).val = (v.val[k]'(by rw [Array.property v]; exact hk)).val := by
  rw [List.getD, List.getElem?_eq_getElem (by rw [Array.property v]; exact hk), Option.getD_some]

lemma hcast_u32_val (x : Std.I32) (h0 : 0 ≤ x.val) :
    (IScalar.hcast UScalarTy.U32 x).val = x.val.toNat := by
  have hle : x.val ≤ (UScalar.max UScalarTy.U32 : Nat) := by scalar_tac
  have hs := IScalar.hcast_inBounds_spec (tgt_ty := UScalarTy.U32) x ⟨h0, hle⟩
  rcases WP.spec_imp_exists hs with ⟨y, hy, hyv⟩
  simp only [lift, Result.ok.injEq] at hy
  rw [hy]
  omega

lemma cast_u8_val (x : Std.U32) (h : x.val ≤ 255) :
    (UScalar.cast UScalarTy.U8 x).val = x.val := by
  have hs := UScalar.cast_inBounds_spec (tgt_ty := UScalarTy.U8) x (by scalar_tac)
  rcases WP.spec_imp_exists hs with ⟨y, hy, hyv⟩
  simp only [lift, Result.ok.injEq] at hy
  rw [hy]
  exact hyv

lemma and_255 (x : Std.U32) : (x &&& 255#u32).val = x.val % 256 := by
  rw [UScalar.val_and]
  norm_num
  exact Nat.and_two_pow_sub_one_eq_mod x.val 8

/-- `build_tables_range_loop0_loop1 p lo16 hi16 0 = ok (lo', hi')`: `lo'[n]` and `hi'[n]` are the
low and high bytes of `p[n] + 508` for every `n < 16`. The hypothesis is the `I32`-overflow-free
domain the Rust caller guarantees (`p[n] + 508 ∈ [0, 65535]`). -/
theorem word_build_tables_range_loop0_loop1_ok (p : Array Std.I32 16#usize)
    (lo16 hi16 : Array Std.U8 16#usize)
    (hp : ∀ n, n < 16 → 0 ≤ (p.val.getD n 0#i32).val + 508 ∧
      (p.val.getD n 0#i32).val + 508 ≤ 65535) :
    ∃ lo' hi' : Array Std.U8 16#usize,
      kern.lut.build_tables_range_loop0_loop1 p lo16 hi16 0#usize = ok (lo', hi') ∧
      (∀ n, n < 16 → (lo'.val.getD n 0#u8).val = encLo (p.val.getD n 0#i32).val) ∧
      (∀ n, n < 16 → (hi'.val.getD n 0#u8).val = encHi (p.val.getD n 0#i32).val) := by
  have hspec : kern.lut.build_tables_range_loop0_loop1 p lo16 hi16 0#usize ⦃ r =>
      (∀ n, n < 16 → (r.1.val.getD n 0#u8).val = encLo (p.val.getD n 0#i32).val) ∧
      (∀ n, n < 16 → (r.2.val.getD n 0#u8).val = encHi (p.val.getD n 0#i32).val) ⦄ := by
    unfold kern.lut.build_tables_range_loop0_loop1
    apply loop.spec_decr_nat
      (measure := fun st : Array Std.U8 16#usize × Array Std.U8 16#usize × Std.Usize =>
        16 - st.2.2.val)
      (inv := fun st => encInv p st.1 st.2.1 st.2.2.val)
      (post := fun r : Array Std.U8 16#usize × Array Std.U8 16#usize =>
        (∀ n, n < 16 → (r.1.val.getD n 0#u8).val = encLo (p.val.getD n 0#i32).val) ∧
        (∀ n, n < 16 → (r.2.val.getD n 0#u8).val = encHi (p.val.getD n 0#i32).val))
    · rintro ⟨l, h, n⟩ ⟨hn16, hl, hh⟩
      dsimp only at hn16 hl hh
      dsimp only
      unfold kern.lut.build_tables_range_loop0_loop1.body
      by_cases hn : n < 16#usize
      · rw [if_pos hn]
        simp only [lift, bind_tc_ok]
        step*
        · -- `p[n] + 508` does not overflow `I32`
          have hn16' : n.val < 16 := by scalar_tac
          have hiv : i.val = (p.val.getD n.val 0#i32).val := by
            rw [i_post]
            exact (getD_i32_lt p n.val hn16').symm
          rw [hiv, I32.max_eq]
          have h508 : (↑(508#i32) : Int) = 508 := rfl
          rw [h508]
          have hb : (p.val.getD n.val 0#i32).val + 508 ≤ 65535 := (hp n.val hn16').2
          omega
        · -- invariant preservation and measure decrease
          have hn16' : n.val < 16 := by scalar_tac
          have hiv : i.val = (p.val.getD n.val 0#i32).val := by
            rw [i_post]
            exact (getD_i32_lt p n.val hn16').symm
          have hi1_lo : 0 ≤ i1.val := by
            rw [i1_post, hiv]; exact (hp n.val hn16').1
          have h1 : (IScalar.hcast UScalarTy.U32 i1).val = i1.val.toNat := hcast_u32_val i1 hi1_lo
          have h2 : ((IScalar.hcast UScalarTy.U32 i1) &&& 255#u32).val = i1.val.toNat % 256 := by
            rw [and_255, h1]
          have h3 : ((IScalar.hcast UScalarTy.U32 i1) &&& 255#u32).val ≤ 255 := by
            rw [h2]
            have := Nat.mod_lt i1.val.toNat (show 0 < 256 by norm_num)
            omega
          have hlo_val : (UScalar.cast UScalarTy.U8 (IScalar.hcast UScalarTy.U32 i1 &&& 255#u32)).val
              = encLo i.val := by
            rw [encLo, cast_u8_val _ h3, h2, congrArg Int.toNat i1_post]
          have hhi_val : (UScalar.cast UScalarTy.U8 i4).val = encHi i.val := by
            rw [encHi]
            have h4 : i4.val = i1.val.toNat / 256 := by
              rw [i4_post, h1, Nat.shiftRight_eq_div_pow]
            have h5 : i4.val ≤ 255 := by
              rw [h4]
              have h6 : i1.val.toNat ≤ 65535 := by
                have hh6 : i1.val ≤ 65535 := by rw [i1_post, hiv]; exact (hp n.val hn16').2
                omega
              have h7 : i1.val.toNat / 256 ≤ 65535 / 256 := Nat.div_le_div_right h6
              norm_num at h7
              exact h7
            rw [cast_u8_val i4 h5, h4, congrArg Int.toNat i1_post]
          rw [n1_post]
          refine ⟨⟨by omega, ?_, ?_⟩, by omega⟩
          · intro k hk
            by_cases hkn : k = n.val
            · subst hkn
              have hlen : n.val < l.val.length := by rw [Array.property l]; exact hn16'
              rw [a_post, Array.set_val_eq, getD_set, if_pos ⟨rfl, hlen⟩]
              rw [hlo_val, hiv]
            · rw [a_post, Array.set_val_eq, getD_set, if_neg (fun hh => hkn hh.1)]
              exact hl k (by omega)
          · intro k hk
            by_cases hkn : k = n.val
            · subst hkn
              have hlen : n.val < h.val.length := by rw [Array.property h]; exact hn16'
              rw [a1_post, Array.set_val_eq, getD_set, if_pos ⟨rfl, hlen⟩]
              rw [hhi_val, hiv]
            · rw [a1_post, Array.set_val_eq, getD_set, if_neg (fun hh => hkn hh.1)]
              exact hh k (by omega)
      · rw [if_neg hn]
        have hn16'' : n.val = 16 := by scalar_tac
        rw [hn16''] at hl hh
        exact ⟨hl, hh⟩
    · exact encInv_zero p lo16 hi16
  rcases WP.spec_imp_exists hspec with ⟨r, hr, h1, h2⟩
  exact ⟨r.1, r.2, hr, h1, h2⟩

/-! ## `kern.lut.build_tables_range` loop 1 (the doubling p-fill)

`build_tables_range_loop0_loop0 p c3 0` walks `n = 0..7`, writes `l[8+n] = l[n] + c3`. Because the
low half `p[0..8]` is fixed before the loop starts, the reference is `p'[8+k] = p[k] + c3` for
`k < 8`, with the low half unchanged. -/

/-- Loop invariant for the doubling p-fill: `l[8+k] = p0[k] + c3` for `k < n`, and `l` agrees with
`p0` below 8. -/
def addInv (p0 : Array Std.I32 16#usize) (c3 : Std.I32)
    (l : Array Std.I32 16#usize) (n : Nat) : Prop :=
  n ≤ 8 ∧
  (∀ k, k < n → (l.val.getD (8 + k) 0#i32).val = (p0.val.getD k 0#i32).val + c3.val) ∧
  (∀ j, j < 8 → (l.val.getD j 0#i32).val = (p0.val.getD j 0#i32).val)

lemma addInv_zero (p0 : Array Std.I32 16#usize) (c3 : Std.I32) :
    addInv p0 c3 p0 0 := ⟨by norm_num, by omega, by omega⟩

/-- `build_tables_range_loop0_loop0 p c3 0 = ok p'`: `p'[8+k] = p[k] + c3` for `k < 8` and `p'[j] =
p[j]` for `j < 8`. The hypothesis is the `I32`-overflow-free domain (`p[k] + c3 ∈ I32`). -/
theorem word_build_tables_range_loop0_loop0_ok (p : Array Std.I32 16#usize) (c3 : Std.I32)
    (h : ∀ k, k < 8 → -2147483648 ≤ (p.val.getD k 0#i32).val + c3.val ∧
      (p.val.getD k 0#i32).val + c3.val ≤ 2147483647) :
    ∃ p' : Array Std.I32 16#usize,
      kern.lut.build_tables_range_loop0_loop0 p c3 0#usize = ok p' ∧
      (∀ k, k < 8 → (p'.val.getD (8 + k) 0#i32).val = (p.val.getD k 0#i32).val + c3.val) ∧
      (∀ j, j < 8 → (p'.val.getD j 0#i32).val = (p.val.getD j 0#i32).val) := by
  have hspec : kern.lut.build_tables_range_loop0_loop0 p c3 0#usize ⦃ r =>
      (∀ k, k < 8 → (r.val.getD (8 + k) 0#i32).val = (p.val.getD k 0#i32).val + c3.val) ∧
      (∀ j, j < 8 → (r.val.getD j 0#i32).val = (p.val.getD j 0#i32).val) ⦄ := by
    unfold kern.lut.build_tables_range_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun st : Array Std.I32 16#usize × Std.Usize => 8 - st.2.val)
      (inv := fun st => addInv p c3 st.1 st.2.val)
      (post := fun r : Array Std.I32 16#usize =>
        (∀ k, k < 8 → (r.val.getD (8 + k) 0#i32).val = (p.val.getD k 0#i32).val + c3.val) ∧
        (∀ j, j < 8 → (r.val.getD j 0#i32).val = (p.val.getD j 0#i32).val))
    · rintro ⟨l, n⟩ ⟨hn8, hl, hl8⟩
      dsimp only at hn8 hl hl8
      dsimp only
      unfold kern.lut.build_tables_range_loop0_loop0.body
      by_cases hn : n < 8#usize
      · rw [if_pos hn]
        step*
        · -- `p[n] + c3` does not underflow `I32`
          have hn8'' : n.val < 8 := by scalar_tac
          have hiv : i.val = (p.val.getD n.val 0#i32).val := by
            calc i.val = (l.val.getD n.val 0#i32).val := by
                  rw [i_post]; exact (getD_i32_lt l n.val (by omega)).symm
              _ = (p.val.getD n.val 0#i32).val := hl8 n.val hn8''
          rw [hiv, I32.min_eq]
          have hb := (h n.val hn8'').1
          omega
        · -- `p[n] + c3` does not overflow `I32`
          have hn8'' : n.val < 8 := by scalar_tac
          have hiv : i.val = (p.val.getD n.val 0#i32).val := by
            calc i.val = (l.val.getD n.val 0#i32).val := by
                  rw [i_post]; exact (getD_i32_lt l n.val (by omega)).symm
              _ = (p.val.getD n.val 0#i32).val := hl8 n.val hn8''
          rw [hiv, I32.max_eq]
          have hb := (h n.val hn8'').2
          omega
        · -- invariant preservation and measure decrease
          have hn8'' : n.val < 8 := by scalar_tac
          have hiv : i.val = (p.val.getD n.val 0#i32).val := by
            calc i.val = (l.val.getD n.val 0#i32).val := by
                  rw [i_post]; exact (getD_i32_lt l n.val (by omega)).symm
              _ = (p.val.getD n.val 0#i32).val := hl8 n.val hn8''
          rw [n1_post]
          refine ⟨⟨by omega, ?_, ?_⟩, by omega⟩
          · intro k hk
            by_cases hkn : k = n.val
            · subst hkn
              have hlen : (8 + n.val) < l.val.length := by
                have hl16 : l.val.length = 16 := Array.property l
                omega
              rw [a_post, Array.set_val_eq, i2_post, getD_set, if_pos ⟨rfl, hlen⟩]
              rw [i1_post, hiv]
            · rw [a_post, Array.set_val_eq, i2_post, getD_set, if_neg (fun hh => by omega)]
              exact hl k (by omega)
          · intro j hj
            rw [a_post, Array.set_val_eq, i2_post, getD_set, if_neg (fun hh => by omega)]
            exact hl8 j hj
      · rw [if_neg hn]
        have hn8' : n.val = 8 := by scalar_tac
        refine ⟨?_, hl8⟩
        intro k hk
        exact hl k (by omega)
    · exact addInv_zero p c3
  rcases WP.spec_imp_exists hspec with ⟨r, hr, h1, h2⟩
  exact ⟨r, hr, h1, h2⟩

/-! ## Non-vacuity -/

/-- `split_nibbles` on `0xAB` (bits `1010_1011`): even bits `0001`, odd bits `1111`. -/
example : splitLo 0xAB = 1 ∧ splitHi 0xAB = 15 := by decide

lemma repeat_getD (x : Std.I32) (n : Nat) (h : n < 16) :
    ((Array.repeat 16#usize x).val.getD n 0#i32).val = x.val := by
  have h' : (Array.repeat 16#usize x).val.getD n 0#i32 = x := by
    rw [Array.repeat_val]
    exact List.getD_replicate x h
  rw [h']

/-- The encoder hypothesis holds jointly for the zero block (`p[n] = 0`, `0 + 508 ∈ [0, 65535]`). -/
example : ∀ n, n < 16 →
    0 ≤ ((Array.repeat 16#usize (0#i32)).val.getD n 0#i32).val + 508 ∧
    ((Array.repeat 16#usize (0#i32)).val.getD n 0#i32).val + 508 ≤ 65535 := by
  intro n h
  rw [repeat_getD (0#i32) n h]
  norm_num

/-- The p-fill hypothesis holds jointly for the zero block and `c3 = 0`. -/
example : ∀ k, k < 8 →
    -2147483648 ≤ ((Array.repeat 16#usize (0#i32)).val.getD k 0#i32).val + (0#i32).val ∧
    ((Array.repeat 16#usize (0#i32)).val.getD k 0#i32).val + (0#i32).val ≤ 2147483647 := by
  intro k hk
  rw [repeat_getD (0#i32) k (by omega)]
  norm_num

end apeiron_word

#print axioms apeiron_word.word_split_nibbles_ok
#print axioms apeiron_word.word_build_tables_range_loop0_loop1_ok
#print axioms apeiron_word.word_build_tables_range_loop0_loop0_ok
