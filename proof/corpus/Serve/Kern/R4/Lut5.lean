/-
Refinement of `block_inners_portable` (`src/engine/kern/lut.rs` lines 357:0-390:1), group `kern.lut`,
continuing `Serve/Kern/R4/Lut4.lean` (the `tile_accs_portable` chain) and `Serve/Kern/R4/Lut3.lean`
(`word_tile_accs_inner`, the innermost group loop).

`block_inners_portable` walks the `(lane, sub-block, group)` nest of one block `b` and writes lane
`lane`'s block value `Σ_{sb<4} S_{4b+sb} · (2·SH + SL − SX)` into its output row. It reads the code
bytes from `lut.codes` at `(tile·lut.n_groups + g)·32 + lane` and allocates in `i64`, not the `i128`
row accumulator of `tile_accs_portable`; its three loops are therefore the same nest over its own
value and its own overflow hypotheses.

Reference: `lut5_blk lut q tables tile lane b = Σ_{sb<4} s(4b+sb) ·
(2·SH + SL − SX)(b,sb)`, exactly `Lut4`'s `lut4_blk` read on `lut.codes` / `lut.n_groups`.

Proved here (grade `proved`, axiom cone `{propext, Classical.choice, Quot.sound}` under the stated
hypotheses):

* `word_block_inners_group_ok` — `_loop0_loop0_loop0`, the group loop, on `Lut3`'s `lut3_innerInv`;
* `word_block_inners_sb_ok` — `_loop0_loop0`, the sub-block loop, in `i64`;
* `word_block_inners_lane_ok` — `_loop0`, the lane loop;
* `word_block_inners_portable_ok` — the top level;
* `word_block_inners_portable_refuses` — the out-of-bounds refusal;
* the non-vacuity witness `lut5_witLut` / `lut5_witQuant` and its `example`.

Boundaries stated, not papered over:

* the `i16::from(u8)` model gap is inherited from `Lut3` as the hypothesis `lut3_fromU8I16_ok`;
* the byte planes are read as `getD` under the caller's length hypotheses and `hhb` keeps the high
  plane `≤ 127`;
* the `i64` value hypothesis `hterm` is a caller hypothesis: the sub-block product
  `S·inner` is stated as not overflowing `i64`, matching the caller's `lut_i64_bound_ok` check, not
  derived from the scale magnitudes here;
* `block_inners_avx2` is an extracted `axiom` (no body to refine); the `F32`/`F64` carriers and the
  `unsafe`/AVX2 paths are out of scope.
-/
import Serve.Kern.R4.Lut4

set_option autoImplicit false
set_option maxHeartbeats 800000

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## Reference -/

/-- The `block_inners_portable` reference: `Lut4`'s block value `lut4_blk` read on the code plane
`lut.codes` and the group count `lut.n_groups`. -/
def lut5_blk (lut : kern.lut.Pq2Lut) (q : chart.QuantAct) (tables : kern.lut.LutTables)
    (tile lane b : Nat) : Int :=
  lut4_blk q lut.codes tables lut.n_groups.val tile lane b

/-- The sub-block product of sub-block `s` of block `b`: `S_{4b+s} · inner_{b,s}`. -/
def lut5_term (lut : kern.lut.Pq2Lut) (q : chart.QuantAct) (tables : kern.lut.LutTables)
    (tile lane b s : Nat) : Int :=
  lut3_sval q b s * lut4_iv lut.codes tables lut.n_groups.val tile lane b s

/-- `Σ_{k≤n}` of a zero sequence is zero. -/
lemma lut5_isum_zero_const (n : Nat) : lut3_isum n (fun _ : Nat => (0 : Int)) = 0 := by
  induction n with
  | zero => rfl
  | succ k ih => rw [lut3_isum_succ, ih]; rfl

/-- Four `i64`-bounded products sum inside `i64`. -/
lemma lut5_isum4_i64 (f : Nat → Int) (k : Nat) (hk : k ≤ 4)
    (h : ∀ j, j < k → lut4_absb (f j) 1152921504606846976) :
    lut4_absb (lut3_isum k f) 4611686018427387904 := by
  refine lut4_absb_mono (lut4_isum_absb k f 1152921504606846976 h) ?_
  have hk' : (k : Int) ≤ 4 := by exact_mod_cast hk
  nlinarith

/-- `lut3_innerPost` at bound `8` over `lut.codes`: its four conjuncts. -/
lemma lut5_ipost (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables)
    (n_groups tile lane gbase : Nat) (tables1 : kern.lut.LutTables)
    (sum_hi sum_lo sum_x : Std.I64)
    (h : lut3_innerPost lut.codes tables n_groups tile lane gbase 8
      (tables1, sum_hi, sum_lo, sum_x)) :
    tables1 = tables ∧
    (↑sum_hi : Int) = lut3_isum 8 (fun j => lut3_pval tables (gbase + j)
      (lut3_hi lut.codes n_groups tile (gbase + j) lane)) ∧
    (↑sum_lo : Int) = lut3_isum 8 (fun j => lut3_pval tables (gbase + j)
      (lut3_lo lut.codes n_groups tile (gbase + j) lane)) ∧
    (↑sum_x : Int) = lut3_isum 8 (fun j => lut3_xval tables (gbase + j)) :=
  lut4_innerPost_eq lut.codes tables n_groups tile lane gbase tables1 sum_hi sum_lo sum_x h

/-- The sub-block product `s1 · i8` of the body equals the reference `lut5_term`. -/
lemma lut5_prod_term (q : chart.QuantAct) (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables)
    (tile lane b sb : Nat) (s1 i8 : Std.I64)
    (hs1 : (↑s1 : Int) = lut3_sval q b sb)
    (hiv : (↑i8 : Int) = lut4_iv lut.codes tables lut.n_groups.val tile lane b sb) :
    (↑s1 : Int) * (↑i8 : Int) = lut5_term lut q tables tile lane b sb := by
  rw [hs1, hiv]
  unfold lut5_term
  rfl

/-! ## The group loop

`block_inners_portable_loop0_loop0_loop0` is `Lut3`'s innermost group loop `tile_accs_portable_…_
loop0` read on `lut.codes` / `lut.n_groups` and returning the `Pq2Lut` alongside; its invariant and
postcondition are `Lut3`'s `lut3_innerInv` / `lut3_innerPost` at `v := lut.codes`,
`n_groups := lut.n_groups.val`. -/

/-- The group loop of `block_inners_portable` refines `Lut3`'s three reference partial sums when the
code byte and the table entries are in range, the high plane stays `≤ 127`, and every partial sum
keeps `i64` headroom. -/
theorem word_block_inners_group_ok (hconv : lut3_fromU8I16_ok)
    (i : Std.Usize) (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables)
    (tile lane gbase : Std.Usize) (sh sl sx : Std.I64) (g8 : Std.Usize)
    (hinv : lut3_innerInv lut.codes tables lut.n_groups.val tile.val lane.val gbase.val i.val
      sh sl sx g8.val)
    (hidx : (tile.val * lut.n_groups.val + gbase.val + i.val) * 32 + lane.val ≤ Usize.max)
    (hv : ∀ j, j < i.val →
      lut3_cidx lut.n_groups.val tile.val (gbase.val + j) lane.val < lut.codes.val.length)
    (hloR : ∀ j, j < i.val → 16 * (gbase.val + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ j, j < i.val → 16 * (gbase.val + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ j, j < i.val → gbase.val + j < tables.xs.val.length)
    (hrng : ∀ k, k ≤ i.val →
      lut3_rng (lut3_isum k (fun j => lut3_pval tables (gbase.val + j)
        (lut3_hi lut.codes lut.n_groups.val tile.val (gbase.val + j) lane.val))) ∧
      lut3_rng (lut3_isum k (fun j => lut3_pval tables (gbase.val + j)
        (lut3_lo lut.codes lut.n_groups.val tile.val (gbase.val + j) lane.val))) ∧
      lut3_rng (lut3_isum k (fun j => lut3_xval tables (gbase.val + j)))) :
    ∃ r, kern.lut.block_inners_portable_loop0_loop0_loop0 i lut tables tile lane gbase sh sl sx g8
        = ok r ∧
      r.1 = lut ∧ lut3_innerPost lut.codes tables lut.n_groups.val tile.val lane.val gbase.val i.val
        (r.2.1, r.2.2.1, r.2.2.2.1, r.2.2.2.2) := by
  have hspec : kern.lut.block_inners_portable_loop0_loop0_loop0 i lut tables tile lane gbase sh sl sx
      g8 ⦃ r => r.1 = lut ∧ lut3_innerPost lut.codes tables lut.n_groups.val tile.val lane.val
        gbase.val i.val (r.2.1, r.2.2.1, r.2.2.2.1, r.2.2.2.2) ⦄ := by
    unfold kern.lut.block_inners_portable_loop0_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun st : Std.I64 × Std.I64 × Std.I64 × Std.Usize => i.val - st.2.2.2.val)
      (inv := fun st => lut3_innerInv lut.codes tables lut.n_groups.val tile.val lane.val gbase.val
        i.val st.1 st.2.1 st.2.2.1 st.2.2.2.val)
      (post := fun r : kern.lut.Pq2Lut × kern.lut.LutTables × Std.I64 × Std.I64 × Std.I64 =>
        r.1 = lut ∧ lut3_innerPost lut.codes tables lut.n_groups.val tile.val lane.val gbase.val
          i.val (r.2.1, r.2.2.1, r.2.2.2.1, r.2.2.2.2))
    · rintro ⟨sh1, sl1, sx1, g8p⟩ hinv1
      obtain ⟨hle, hsh, hsl, hsx⟩ := hinv1
      dsimp only at hle hsh hsl hsx
      dsimp only
      unfold kern.lut.block_inners_portable_loop0_loop0_loop0.body
      by_cases hg : g8p < i
      · rw [if_pos hg]
        simp only [lift, bind_tc_ok]
        have hglt : g8p.val < i.val := by scalar_tac
        have hTR : (↑kern.lut.TILE_ROWS : Nat) = 32 := by
          simp only [kern.lut.TILE_ROWS]
          rfl
        have hconv8 : ∀ x : Std.U8, (↑(core.convert.num.FromUsizeU8.from x) : Nat) = ↑x :=
          fun x => lut3_fromU8 x
        have hconv16 : ∀ x : Std.I16, (↑(core.convert.num.FromI64I16.from x) : Int) = ↑x :=
          fun x => lut3_i16ToI64 x
        have hloAt : ∀ n : Nat, n ≤ 15 → 16 * (gbase.val + g8p.val) + n < tables.lo.val.length := by
          intro n hn
          have h := hloR g8p.val hglt
          omega
        have hhiAt : ∀ n : Nat, n ≤ 15 → 16 * (gbase.val + g8p.val) + n < tables.hi.val.length := by
          intro n hn
          have h := hhiR g8p.val hglt
          omega
        have hhinAt : ∀ n : Nat, n ≤ 15 →
            (tables.hi.val.getD (16 * (gbase.val + g8p.val) + n) 0#u8).val ≤ 127 := by
          intro n hn
          exact hhb _ (by have h := hhiR g8p.val hglt; omega)
        step*
        · -- 1: the code byte is in range
          rw [idx_post, i3_post, i2_post, i1_post, g_post, hTR]
          exact hv g8p.val hglt
        · -- 2: the high-plane index is in `lo`
          rw [g_post, hconv8, i5_post]
          exact hloAt _ (lut3_hi_nib i4)
        · -- 3: the high-plane index is in `hi`
          rw [g_post, hconv8, i5_post]
          exact hhiAt _ (lut3_hi_nib i4)
        · -- 4: the high plane stays `≤ 127`
          rw [g_post, hconv8, i5_post]
          exact hhinAt _ (lut3_hi_nib i4)
        · -- 5: `sh1 + p` does not underflow `i64`
          rw [hconv16, i7_post, g_post, hconv8, i5_post, hsh]
          have h1 := (hrng g8p.val hle).1
          simp only [lut3_rng] at h1
          obtain ⟨h1a, h1b⟩ := h1
          have h2 := lut3_pval_ge tables (↑gbase + ↑g8p) (↑i4 >>> 4)
          rw [I64.min_eq] at h1a ⊢
          omega
        · -- 6: `sh1 + p` does not overflow `i64`
          rw [hconv16, i7_post, g_post, hconv8, i5_post, hsh]
          have h1 := (hrng g8p.val hle).1
          simp only [lut3_rng] at h1
          obtain ⟨h1a, h1b⟩ := h1
          have h2 := lut3_pval_le tables (↑gbase + ↑g8p) (↑i4 >>> 4)
            (hhinAt _ (lut3_hi_nib i4))
          rw [I64.max_eq] at h1b ⊢
          omega
        · -- 7: the low-plane index is in `lo`
          rw [g_post, hconv8]
          exact hloAt _ (lut3_lo_nib i4)
        · -- 8: the low-plane index is in `hi`
          rw [g_post, hconv8]
          exact hhiAt _ (lut3_lo_nib i4)
        · -- 9: the low plane stays `≤ 127`
          rw [g_post, hconv8]
          exact hhinAt _ (lut3_lo_nib i4)
        · -- 10: `sl1 + p` does not underflow `i64`
          rw [hconv16, i11_post, g_post, hconv8, hsl]
          have h1 := (hrng g8p.val hle).2.1
          simp only [lut3_rng] at h1
          obtain ⟨h1a, h1b⟩ := h1
          have h2 := lut3_pval_ge tables (↑gbase + ↑g8p) (↑(i4 &&& 15#u8))
          rw [I64.min_eq] at h1a ⊢
          omega
        · -- 11: `sl1 + p` does not overflow `i64`
          rw [hconv16, i11_post, g_post, hconv8, hsl]
          have h1 := (hrng g8p.val hle).2.1
          simp only [lut3_rng] at h1
          obtain ⟨h1a, h1b⟩ := h1
          have h2 := lut3_pval_le tables (↑gbase + ↑g8p) (↑(i4 &&& 15#u8))
            (hhinAt _ (lut3_lo_nib i4))
          rw [I64.max_eq] at h1b ⊢
          omega
        · -- 12: `sx1 + xs[g]` does not underflow `i64`
          have h13 : (↑i13 : Int) = lut3_xval tables (↑gbase + ↑g8p) := by
            have hglt2 : (↑g : Nat) < tables.xs.val.length := by
              rw [g_post]; exact hxR g8p.val hglt
            have h1 : (↑i13 : Int) = ↑(tables.xs.val.getD (↑g) 0#i16) := by
              rw [i13_post]
              exact congrArg (fun y : Std.I16 => (↑y : Int))
                (getD_eq_getElem tables.xs.val (↑g) 0#i16 hglt2).symm
            have h2 : (↑i13 : Int) = ↑(tables.xs.val.getD (↑gbase + ↑g8p) 0#i16) := by
              rw [h1, g_post]
            rw [h2]
            rfl
          rw [hconv16, hsx, h13]
          have h1 := (hrng g8p.val hle).2.2
          simp only [lut3_rng] at h1
          obtain ⟨h1a, h1b⟩ := h1
          have h3 := lut3_xval_bounds tables (↑gbase + ↑g8p)
          rw [I64.min_eq] at h1a ⊢
          omega
        · -- 13: `sx1 + xs[g]` does not overflow `i64`
          have h13 : (↑i13 : Int) = lut3_xval tables (↑gbase + ↑g8p) := by
            have hglt2 : (↑g : Nat) < tables.xs.val.length := by
              rw [g_post]; exact hxR g8p.val hglt
            have h1 : (↑i13 : Int) = ↑(tables.xs.val.getD (↑g) 0#i16) := by
              rw [i13_post]
              exact congrArg (fun y : Std.I16 => (↑y : Int))
                (getD_eq_getElem tables.xs.val (↑g) 0#i16 hglt2).symm
            have h2 : (↑i13 : Int) = ↑(tables.xs.val.getD (↑gbase + ↑g8p) 0#i16) := by
              rw [h1, g_post]
            rw [h2]
            rfl
          rw [hconv16, hsx, h13]
          have h1 := (hrng g8p.val hle).2.2
          simp only [lut3_rng] at h1
          obtain ⟨h1a, h1b⟩ := h1
          have h3 := lut3_xval_bounds tables (↑gbase + ↑g8p)
          rw [I64.max_eq] at h1b ⊢
          omega
        · -- 14: the invariant is preserved and the measure decreases
          refine ⟨?_, ?_⟩
          · unfold lut3_innerInv
            refine ⟨?_, ?_, ?_, ?_⟩
            · rw [g81_post]; omega
            · have hidxeq : (↑idx : Nat) = lut3_cidx (↑lut.n_groups) (↑tile) (↑gbase + ↑g8p)
                  (↑lane) := by
                rw [idx_post, i3_post, i2_post, i1_post, g_post, hTR]
                rfl
              have himeq : (↑(core.convert.num.FromUsizeU8.from i5) : Nat)
                  = lut3_hi lut.codes (↑lut.n_groups) (↑tile) (↑gbase + ↑g8p) (↑lane) := by
                have hidxlt : (↑idx : Nat) < lut.codes.val.length := by
                  rw [hidxeq]; exact hv g8p.val hglt
                have h4d : (↑i4 : Nat) = ↑(lut.codes.val.getD (↑idx) 0#u8) := by
                  rw [i4_post]
                  exact congrArg (fun y : Std.U8 => (↑y : Nat))
                    (getD_eq_getElem lut.codes.val (↑idx) 0#u8 hidxlt).symm
                have h4e : (↑i4 : Nat) = ↑(lut.codes.val.getD
                    (lut3_cidx (↑lut.n_groups) (↑tile) (↑gbase + ↑g8p) (↑lane)) 0#u8) := by
                  rw [h4d, hidxeq]
                rw [hconv8, i5_post, h4e]
                rfl
              have h1 : (↑sum_hi1 : Int)
                  = lut3_isum (↑g8p)
                      (fun j => lut3_pval tables (↑gbase + j)
                        (lut3_hi lut.codes (↑lut.n_groups) (↑tile) (↑gbase + j) (↑lane)))
                    + lut3_pval tables (↑gbase + ↑g8p)
                        (↑(core.convert.num.FromUsizeU8.from i5)) := by
                rw [sum_hi1_post, hconv16, i7_post, g_post, hsh]
              have h2 : lut3_pval tables (↑gbase + ↑g8p)
                    (↑(core.convert.num.FromUsizeU8.from i5))
                  = (fun j => lut3_pval tables (↑gbase + j)
                      (lut3_hi lut.codes (↑lut.n_groups) (↑tile) (↑gbase + j) (↑lane))) (↑g8p) := by
                rw [himeq]
              rw [h1, h2, g81_post, lut3_isum_succ]
            · have hidxeq : (↑idx : Nat) = lut3_cidx (↑lut.n_groups) (↑tile) (↑gbase + ↑g8p)
                  (↑lane) := by
                rw [idx_post, i3_post, i2_post, i1_post, g_post, hTR]
                rfl
              have hlomeq : (↑(core.convert.num.FromUsizeU8.from (i4 &&& 15#u8)) : Nat)
                  = lut3_lo lut.codes (↑lut.n_groups) (↑tile) (↑gbase + ↑g8p) (↑lane) := by
                have hidxlt : (↑idx : Nat) < lut.codes.val.length := by
                  rw [hidxeq]; exact hv g8p.val hglt
                have h4d : (↑i4 : Nat) = ↑(lut.codes.val.getD (↑idx) 0#u8) := by
                  rw [i4_post]
                  exact congrArg (fun y : Std.U8 => (↑y : Nat))
                    (getD_eq_getElem lut.codes.val (↑idx) 0#u8 hidxlt).symm
                have h4e : (↑i4 : Nat) = ↑(lut.codes.val.getD
                    (lut3_cidx (↑lut.n_groups) (↑tile) (↑gbase + ↑g8p) (↑lane)) 0#u8) := by
                  rw [h4d, hidxeq]
                rw [hconv8, show (↑(i4 &&& 15#u8) : Nat) = (↑i4 : Nat) % 16 from lut3_u8_lo i4,
                  h4e]
                rfl
              have h1 : (↑sum_lo1 : Int)
                  = lut3_isum (↑g8p)
                      (fun j => lut3_pval tables (↑gbase + j)
                        (lut3_lo lut.codes (↑lut.n_groups) (↑tile) (↑gbase + j) (↑lane)))
                    + lut3_pval tables (↑gbase + ↑g8p)
                        (↑(core.convert.num.FromUsizeU8.from (i4 &&& 15#u8))) := by
                rw [sum_lo1_post, hconv16, i11_post, g_post, hsl]
              have h2 : lut3_pval tables (↑gbase + ↑g8p)
                    (↑(core.convert.num.FromUsizeU8.from (i4 &&& 15#u8)))
                  = (fun j => lut3_pval tables (↑gbase + j)
                      (lut3_lo lut.codes (↑lut.n_groups) (↑tile) (↑gbase + j) (↑lane))) (↑g8p) := by
                rw [hlomeq]
              rw [h1, h2, g81_post, lut3_isum_succ]
            · have hx13 : (↑(core.convert.num.FromI64I16.from i13) : Int)
                  = lut3_xval tables (↑gbase + ↑g8p) := by
                have hglt2 : (↑g : Nat) < tables.xs.val.length := by
                  rw [g_post]; exact hxR g8p.val hglt
                have h1 : (↑i13 : Int) = ↑(tables.xs.val.getD (↑g) 0#i16) := by
                  rw [i13_post]
                  exact congrArg (fun y : Std.I16 => (↑y : Int))
                    (getD_eq_getElem tables.xs.val (↑g) 0#i16 hglt2).symm
                have h2 : (↑i13 : Int) = ↑(tables.xs.val.getD (↑gbase + ↑g8p) 0#i16) := by
                  rw [h1, g_post]
                rw [hconv16, h2]
                rfl
              have h1 : (↑sum_x1 : Int)
                  = lut3_isum (↑g8p) (fun j => lut3_xval tables (↑gbase + j))
                    + (↑(core.convert.num.FromI64I16.from i13) : Int) := by
                rw [sum_x1_post, hsx]
              have h2 : (↑(core.convert.num.FromI64I16.from i13) : Int)
                  = (fun j => lut3_xval tables (↑gbase + j)) (↑g8p) := by
                rw [hx13]
              rw [h1, h2, g81_post, lut3_isum_succ]
          · rw [g81_post]; omega
      · rw [if_neg hg]
        simp only [WP.spec_ok]
        have hgv : i.val ≤ g8p.val := by scalar_tac
        have hveq : i.val = g8p.val := le_antisymm hgv hle
        refine ⟨?_, ?_⟩
        · first | rfl | trivial
        · unfold lut3_innerPost
          exact ⟨rfl, by rw [hsh, hveq], by rw [hsl, hveq], by rw [hsx, hveq]⟩
    · exact hinv
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  exact ⟨r, hr, hpost⟩

/-- `word_block_inners_group_ok` in weakest-precondition form at bound `8`, for `step*`. -/
@[step]
lemma lut5_group8_step (hconv : lut3_fromU8I16_ok)
    (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables) (tile lane gbase : Std.Usize)
    (hidx : (tile.val * lut.n_groups.val + gbase.val + 8) * 32 + lane.val ≤ Usize.max)
    (hv : ∀ j, j < 8 →
      lut3_cidx lut.n_groups.val tile.val (gbase.val + j) lane.val < lut.codes.val.length)
    (hloR : ∀ j, j < 8 → 16 * (gbase.val + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ j, j < 8 → 16 * (gbase.val + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ j, j < 8 → gbase.val + j < tables.xs.val.length) :
    kern.lut.block_inners_portable_loop0_loop0_loop0 8#usize lut tables tile lane gbase
      0#i64 0#i64 0#i64 0#usize
      ⦃ r => r.1 = lut ∧ lut3_innerPost lut.codes tables lut.n_groups.val tile.val lane.val
        gbase.val 8 (r.2.1, r.2.2.1, r.2.2.2.1, r.2.2.2.2) ⦄ := by
  have h8 : (8#usize : Std.Usize).val = 8 := by simp
  have hile := lut4_hile_of tables hhb
  have hmain := word_block_inners_group_ok hconv 8#usize lut tables tile lane gbase
    0#i64 0#i64 0#i64 0#usize
    (by unfold lut3_innerInv
        refine ⟨by simp, ?_, ?_, ?_⟩ <;> simp [lut3_isum_zero])
    (by rw [h8]; exact hidx)
    (by intro j hj; rw [h8] at hj; exact hv j hj)
    (by intro j hj; rw [h8] at hj; exact hloR j hj)
    (by intro j hj; rw [h8] at hj; exact hhiR j hj)
    (by exact hhb)
    (by intro j hj; rw [h8] at hj; exact hxR j hj)
    (by intro k hk
        rw [h8] at hk
        exact ⟨lut4_rng_hi tables hile lut.codes lut.n_groups.val tile.val lane.val gbase.val k hk,
          lut4_rng_lo tables hile lut.codes lut.n_groups.val tile.val lane.val gbase.val k hk,
          lut4_rng_x tables gbase.val k hk⟩)
  obtain ⟨r, hr, hp⟩ := hmain
  rw [hr]
  simp only [WP.spec_ok]
  exact hp

/-! ## The sub-block loop

`block_inners_portable_loop0_loop0` walks the four sub-blocks of block `b` in `i64`:
`acc += S_{4b+sb} · (2·sum_hi + sum_lo − sum_x)`. It is `Lut4`'s `tile_accs_portable_loop0_loop0` over
the same nest, on `lut.codes` and in `i64` instead of `i128`. -/

/-- The sub-block loop of `block_inners_portable` refines the block value `lut5_blk` when the code
bytes, the table entries and the activation scales are in range, the high plane stays `≤ 127`, the
sub-block products and their partial sums fit `i64`. -/
theorem word_block_inners_sb_ok (hconv : lut3_fromU8I16_ok)
    (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables) (q : chart.QuantAct)
    (tile b lane : Std.Usize)
    (hb2 : ∀ s, s < 4 → 4 * b.val + s ≤ Usize.max)
    (hg : ∀ s, s < 4 → (4 * b.val + s) * 8 ≤ Usize.max)
    (hsc : ∀ s, s < 4 → 4 * b.val + s < q.scales.val.length)
    (hidx : ∀ s, s < 4 →
      (tile.val * lut.n_groups.val + (4 * b.val + s) * 8 + 8) * 32 + lane.val ≤ Usize.max)
    (hv : ∀ s, s < 4 → ∀ j, j < 8 →
      lut3_cidx lut.n_groups.val tile.val ((4 * b.val + s) * 8 + j) lane.val
        < lut.codes.val.length)
    (hloR : ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b.val + s) * 8 + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b.val + s) * 8 + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ s, s < 4 → ∀ j, j < 8 → (4 * b.val + s) * 8 + j < tables.xs.val.length)
    (hterm : ∀ s, s < 4 → lut4_absb (lut5_term lut q tables tile.val lane.val b.val s)
      1152921504606846976) :
    ∃ r, kern.lut.block_inners_portable_loop0_loop0 lut tables q tile b lane 0#i64 0#usize
        = ok r ∧
      r.1 = lut ∧ r.2.1 = tables ∧ (r.2.2).val = lut5_blk lut q tables tile.val lane.val b.val := by
  have hspec : kern.lut.block_inners_portable_loop0_loop0 lut tables q tile b lane 0#i64 0#usize
      ⦃ r => r.1 = lut ∧ r.2.1 = tables ∧
        (r.2.2).val = lut5_blk lut q tables tile.val lane.val b.val ⦄ := by
    unfold kern.lut.block_inners_portable_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun st : kern.lut.Pq2Lut × kern.lut.LutTables × Std.I64 × Std.Usize =>
        4 - st.2.2.2.val)
      (inv := fun st : kern.lut.Pq2Lut × kern.lut.LutTables × Std.I64 × Std.Usize =>
        st.1 = lut ∧ st.2.1 = tables ∧ st.2.2.2.val ≤ 4 ∧
        (st.2.2.1).val = lut3_isum st.2.2.2.val (fun s => lut5_term lut q tables tile.val lane.val
          b.val s))
      (post := fun r : kern.lut.Pq2Lut × kern.lut.LutTables × Std.I64 =>
        r.1 = lut ∧ r.2.1 = tables ∧
        (r.2.2).val = lut5_blk lut q tables tile.val lane.val b.val)
    · rintro ⟨lut0, tbl0, acc0, sb0⟩ ⟨h1, h2, hsb4, hacc⟩
      dsimp only at h1 h2 hsb4 hacc ⊢
      unfold kern.lut.block_inners_portable_loop0_loop0.body
      rw [lut3_subs_per_block]
      simp only [bind_tc_ok]
      by_cases hc : sb0 < (4#usize : Std.Usize)
      · rw [if_pos hc]
        unfold chart.QuantAct.impl.scales
        simp only [bind_tc_ok]
        rw [lut3_groups_per_sub]
        simp only [bind_tc_ok]
        have hsbv : sb0.val < 4 := by scalar_tac
        have hile0 : lut4_hile tables := lut4_hile_of tables hhb
        have hile1 : lut4_hile tbl0 := by rw [h2]; exact hile0
        step*
        · -- 1: the scale index is in range
          rw [i2_post, i1_post]
          simpa only [Slice.length, lut4_deref_val] using hsc (↑sb0) hsbv
        · -- 2: `4·b` does not overflow
          have h := hb2 (↑sb0) hsbv; scalar_tac
        · -- 3: `4·b + sb` does not overflow
          rw [i3_post]; have h := hb2 (↑sb0) hsbv; scalar_tac
        · -- 4: `(4·b + sb)·8` does not overflow
          rw [i4_post, i3_post]; have h := hg (↑sb0) hsbv; scalar_tac
        · -- 5: the lane index is in range
          rw [h1, gbase_post, i4_post, i3_post]
          exact hidx (↑sb0) hsbv
        · -- 6: the code byte is in range
          intro j hj
          rw [h1, gbase_post, i4_post, i3_post]
          exact hv (↑sb0) hsbv j hj
        · -- 7: the high-plane index is in `lo`
          intro j hj
          rw [h2, gbase_post, i4_post, i3_post]
          exact hloR (↑sb0) hsbv j hj
        · -- 8: the high-plane index is in `hi`
          intro j hj
          rw [h2, gbase_post, i4_post, i3_post]
          exact hhiR (↑sb0) hsbv j hj
        · -- 9: the `xs` index is in range
          intro j hj
          rw [h2, gbase_post, i4_post, i3_post]
          exact hxR (↑sb0) hsbv j hj
        · -- 10: `2·sum_hi` does not underflow `i64`
          obtain ⟨-, hsh, -, -⟩ := lut5_ipost lut0 tbl0 (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x lut1_post1
          rw [lut4_i2, hsh]
          have h := lut4_sum_hi_absb tbl0 hile1 lut0.codes (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
          unfold lut4_absb at h
          rw [I64.min_eq]; omega
        · -- 11: `2·sum_hi` does not overflow `i64`
          obtain ⟨-, hsh, -, -⟩ := lut5_ipost lut0 tbl0 (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x lut1_post1
          rw [lut4_i2, hsh]
          have h := lut4_sum_hi_absb tbl0 hile1 lut0.codes (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
          unfold lut4_absb at h
          rw [I64.max_eq]; omega
        · -- 12: `2·sum_hi + sum_lo` does not underflow `i64`
          obtain ⟨-, hsh, hsl, -⟩ := lut5_ipost lut0 tbl0 (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x lut1_post1
          rw [i6_post, hsh, hsl]
          have h1 := lut4_sum_hi_absb tbl0 hile1 lut0.codes (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
          have h2 := lut4_sum_lo_absb tbl0 hile1 lut0.codes (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
          unfold lut4_absb at h1 h2
          rw [I64.min_eq]; omega
        · -- 13: `2·sum_hi + sum_lo` does not overflow `i64`
          obtain ⟨-, hsh, hsl, -⟩ := lut5_ipost lut0 tbl0 (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x lut1_post1
          rw [i6_post, hsh, hsl]
          have h1 := lut4_sum_hi_absb tbl0 hile1 lut0.codes (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
          have h2 := lut4_sum_lo_absb tbl0 hile1 lut0.codes (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
          unfold lut4_absb at h1 h2
          rw [I64.max_eq]; omega
        · -- 14: `2·sum_hi + sum_lo − sum_x` does not underflow `i64`
          obtain ⟨-, hsh, hsl, hsx⟩ := lut5_ipost lut0 tbl0 (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x lut1_post1
          rw [i7_post, i6_post, hsh, hsl, hsx]
          have h1 := lut4_sum_hi_absb tbl0 hile1 lut0.codes (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
          have h2 := lut4_sum_lo_absb tbl0 hile1 lut0.codes (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
          have h3 := lut4_sum_x_absb tbl0 (↑gbase)
          unfold lut4_absb at h1 h2 h3
          rw [I64.min_eq]; omega
        · -- 15: `2·sum_hi + sum_lo − sum_x` does not overflow `i64`
          obtain ⟨-, hsh, hsl, hsx⟩ := lut5_ipost lut0 tbl0 (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x lut1_post1
          rw [i7_post, i6_post, hsh, hsl, hsx]
          have h1 := lut4_sum_hi_absb tbl0 hile1 lut0.codes (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
          have h2 := lut4_sum_lo_absb tbl0 hile1 lut0.codes (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
          have h3 := lut4_sum_x_absb tbl0 (↑gbase)
          unfold lut4_absb at h1 h2 h3
          rw [I64.max_eq]; omega
        · -- 16: the sub-block product does not underflow `i64`
          have hsi : (↑s1 : Int) = lut3_sval q (↑b) (↑sb0) := by
            have hlt : (↑i2 : Nat) < (↑q.scales.deref : List Std.I64).length := by
              rw [i2_post, i1_post]
              simpa only [Slice.length, lut4_deref_val] using hsc (↑sb0) hsbv
            rw [lut4_sval_read q.scales i2 s1 hlt s1_post, lut4_deref_val, i2_post, i1_post]
            unfold lut3_sval
            rfl
          have hiv : (↑i8 : Int) = lut4_iv lut.codes tables (↑lut.n_groups) (↑tile) (↑lane)
              (↑b) (↑sb0) := by
            obtain ⟨-, hsh, hsl, hsx⟩ := lut5_ipost lut0 tbl0 (↑lut0.n_groups) (↑tile) (↑lane)
              (↑gbase) tables1 sum_hi sum_lo sum_x lut1_post1
            rw [i8_post, i7_post, i6_post, hsh, hsl, hsx, h1, h2, gbase_post, i4_post, i3_post]
            unfold lut4_iv lut4_sh lut4_sl lut4_sx lut4_gidx
            rfl
          have hb := hterm (↑sb0) hsbv
          rw [lut5_prod_term q lut tables (↑tile) (↑lane) (↑b) (↑sb0) s1 i8 hsi hiv]
          unfold lut4_absb at hb
          rw [I64.min_eq]; omega
        · -- 17: the sub-block product does not overflow `i64`
          have hsi : (↑s1 : Int) = lut3_sval q (↑b) (↑sb0) := by
            have hlt : (↑i2 : Nat) < (↑q.scales.deref : List Std.I64).length := by
              rw [i2_post, i1_post]
              simpa only [Slice.length, lut4_deref_val] using hsc (↑sb0) hsbv
            rw [lut4_sval_read q.scales i2 s1 hlt s1_post, lut4_deref_val, i2_post, i1_post]
            unfold lut3_sval
            rfl
          have hiv : (↑i8 : Int) = lut4_iv lut.codes tables (↑lut.n_groups) (↑tile) (↑lane)
              (↑b) (↑sb0) := by
            obtain ⟨-, hsh, hsl, hsx⟩ := lut5_ipost lut0 tbl0 (↑lut0.n_groups) (↑tile) (↑lane)
              (↑gbase) tables1 sum_hi sum_lo sum_x lut1_post1
            rw [i8_post, i7_post, i6_post, hsh, hsl, hsx, h1, h2, gbase_post, i4_post, i3_post]
            unfold lut4_iv lut4_sh lut4_sl lut4_sx lut4_gidx
            rfl
          have hb := hterm (↑sb0) hsbv
          rw [lut5_prod_term q lut tables (↑tile) (↑lane) (↑b) (↑sb0) s1 i8 hsi hiv]
          unfold lut4_absb at hb
          rw [I64.max_eq]; omega
        · -- 18: the running accumulator does not underflow `i64`
          have hsi : (↑s1 : Int) = lut3_sval q (↑b) (↑sb0) := by
            have hlt : (↑i2 : Nat) < (↑q.scales.deref : List Std.I64).length := by
              rw [i2_post, i1_post]
              simpa only [Slice.length, lut4_deref_val] using hsc (↑sb0) hsbv
            rw [lut4_sval_read q.scales i2 s1 hlt s1_post, lut4_deref_val, i2_post, i1_post]
            unfold lut3_sval
            rfl
          have hiv : (↑i8 : Int) = lut4_iv lut.codes tables (↑lut.n_groups) (↑tile) (↑lane)
              (↑b) (↑sb0) := by
            obtain ⟨-, hsh, hsl, hsx⟩ := lut5_ipost lut0 tbl0 (↑lut0.n_groups) (↑tile) (↑lane)
              (↑gbase) tables1 sum_hi sum_lo sum_x lut1_post1
            rw [i8_post, i7_post, i6_post, hsh, hsl, hsx, h1, h2, gbase_post, i4_post, i3_post]
            unfold lut4_iv lut4_sh lut4_sl lut4_sx lut4_gidx
            rfl
          have hi9 : (↑i9 : Int) = lut5_term lut q tables (↑tile) (↑lane) (↑b) (↑sb0) := by
            rw [i9_post]
            exact lut5_prod_term q lut tables (↑tile) (↑lane) (↑b) (↑sb0) s1 i8 hsi hiv
          have haccb := lut5_isum4_i64 (fun s => lut5_term lut q tables (↑tile) (↑lane) (↑b) s)
            (↑sb0) hsb4 (fun j hj => hterm j (by omega))
          have hb := hterm (↑sb0) hsbv
          unfold lut4_absb at haccb hb
          rw [hacc, hi9, I64.min_eq]; omega
        · -- 19: the running accumulator does not overflow `i64`
          have hsi : (↑s1 : Int) = lut3_sval q (↑b) (↑sb0) := by
            have hlt : (↑i2 : Nat) < (↑q.scales.deref : List Std.I64).length := by
              rw [i2_post, i1_post]
              simpa only [Slice.length, lut4_deref_val] using hsc (↑sb0) hsbv
            rw [lut4_sval_read q.scales i2 s1 hlt s1_post, lut4_deref_val, i2_post, i1_post]
            unfold lut3_sval
            rfl
          have hiv : (↑i8 : Int) = lut4_iv lut.codes tables (↑lut.n_groups) (↑tile) (↑lane)
              (↑b) (↑sb0) := by
            obtain ⟨-, hsh, hsl, hsx⟩ := lut5_ipost lut0 tbl0 (↑lut0.n_groups) (↑tile) (↑lane)
              (↑gbase) tables1 sum_hi sum_lo sum_x lut1_post1
            rw [i8_post, i7_post, i6_post, hsh, hsl, hsx, h1, h2, gbase_post, i4_post, i3_post]
            unfold lut4_iv lut4_sh lut4_sl lut4_sx lut4_gidx
            rfl
          have hi9 : (↑i9 : Int) = lut5_term lut q tables (↑tile) (↑lane) (↑b) (↑sb0) := by
            rw [i9_post]
            exact lut5_prod_term q lut tables (↑tile) (↑lane) (↑b) (↑sb0) s1 i8 hsi hiv
          have haccb := lut5_isum4_i64 (fun s => lut5_term lut q tables (↑tile) (↑lane) (↑b) s)
            (↑sb0) hsb4 (fun j hj => hterm j (by omega))
          have hb := hterm (↑sb0) hsbv
          unfold lut4_absb at haccb hb
          rw [hacc, hi9, I64.max_eq]; omega
        · -- 20: the invariant is preserved and the measure decreases
          have hsi : (↑s1 : Int) = lut3_sval q (↑b) (↑sb0) := by
            have hlt : (↑i2 : Nat) < (↑q.scales.deref : List Std.I64).length := by
              rw [i2_post, i1_post]
              simpa only [Slice.length, lut4_deref_val] using hsc (↑sb0) hsbv
            rw [lut4_sval_read q.scales i2 s1 hlt s1_post, lut4_deref_val, i2_post, i1_post]
            unfold lut3_sval
            rfl
          have hiv : (↑i8 : Int) = lut4_iv lut.codes tables (↑lut.n_groups) (↑tile) (↑lane)
              (↑b) (↑sb0) := by
            obtain ⟨-, hsh, hsl, hsx⟩ := lut5_ipost lut0 tbl0 (↑lut0.n_groups) (↑tile) (↑lane)
              (↑gbase) tables1 sum_hi sum_lo sum_x lut1_post1
            rw [i8_post, i7_post, i6_post, hsh, hsl, hsx, h1, h2, gbase_post, i4_post, i3_post]
            unfold lut4_iv lut4_sh lut4_sl lut4_sx lut4_gidx
            rfl
          have hi9 : (↑i9 : Int) = lut5_term lut q tables (↑tile) (↑lane) (↑b) (↑sb0) := by
            rw [i9_post]
            exact lut5_prod_term q lut tables (↑tile) (↑lane) (↑b) (↑sb0) s1 i8 hsi hiv
          refine ⟨?_, ?_, ?_, ?_, ?_⟩
          · rw [lut1_post, h1]
          · obtain ⟨htab1, -⟩ := lut5_ipost lut0 tbl0 (↑lut0.n_groups) (↑tile) (↑lane) (↑gbase)
              tables1 sum_hi sum_lo sum_x lut1_post1
            rw [htab1, h2]
          · rw [sb1_post]; omega
          · rw [acc1_post, hacc, hi9, sb1_post, lut3_isum_succ]
          · rw [sb1_post]; omega
      · rw [if_neg hc]
        simp only [WP.spec_ok]
        have h4 : sb0.val = 4 := by scalar_tac
        refine ⟨?_, ?_, ?_⟩
        · first | rfl | trivial
        · first | rfl | trivial
        · rw [hacc, h4]
          rfl
    · refine ⟨rfl, rfl, by norm_num, ?_⟩
      simp [lut3_isum_zero]
  obtain ⟨r, hr, hp⟩ := WP.spec_imp_exists hspec
  exact ⟨r, hr, hp⟩

/-- `word_block_inners_sb_ok` in weakest-precondition form at the fixed bound `4`, for `step*`. -/
@[step]
lemma lut5_sb_step (hconv : lut3_fromU8I16_ok)
    (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables) (q : chart.QuantAct)
    (tile b lane : Std.Usize)
    (hb2 : ∀ s, s < 4 → 4 * b.val + s ≤ Usize.max)
    (hg : ∀ s, s < 4 → (4 * b.val + s) * 8 ≤ Usize.max)
    (hsc : ∀ s, s < 4 → 4 * b.val + s < q.scales.val.length)
    (hidx : ∀ s, s < 4 →
      (tile.val * lut.n_groups.val + ((4 * b.val + s) * 8) + 8) * 32 + lane.val ≤ Usize.max)
    (hv : ∀ s, s < 4 → ∀ j, j < 8 →
      lut3_cidx lut.n_groups.val tile.val ((4 * b.val + s) * 8 + j) lane.val < lut.codes.val.length)
    (hloR : ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b.val + s) * 8 + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b.val + s) * 8 + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ s, s < 4 → ∀ j, j < 8 → (4 * b.val + s) * 8 + j < tables.xs.val.length)
    (hterm : ∀ s, s < 4 →
      lut4_absb (lut5_term lut q tables tile.val lane.val b.val s) 1152921504606846976) :
    kern.lut.block_inners_portable_loop0_loop0 lut tables q tile b lane 0#i64 0#usize
      ⦃ r => r.1 = lut ∧ r.2.1 = tables ∧
        (r.2.2).val = lut5_blk lut q tables tile.val lane.val b.val ⦄ := by
  obtain ⟨r, hr, hp⟩ := word_block_inners_sb_ok hconv lut tables q tile b lane
    hb2 hg hsc hidx hv hloR hhiR hhb hxR hterm
  rw [hr]
  simp only [WP.spec_ok]
  exact hp

/-! ## The lane loop

`block_inners_portable_loop0` walks the 32 lanes of one tile, each lane's block value written into
its output row by `Array::update`. -/

/-- The lane-loop invariant: lanes below `lane` hold their block value, lanes above keep the input. -/
def lut5_laneInv (lut : kern.lut.Pq2Lut) (q : chart.QuantAct) (tables : kern.lut.LutTables)
    (tile b : Nat) (accs cur : Array Std.I64 32#usize) (lane : Nat) : Prop :=
  (∀ k, k < lane → (cur.val.getD k default).val = lut5_blk lut q tables tile k b) ∧
  (∀ k, lane ≤ k → k < 32 → cur.val.getD k default = accs.val.getD k default)

/-- One lane step: writing lane `lane`'s block value keeps the lane invariant. -/
lemma lut5_laneInv_set (lut : kern.lut.Pq2Lut) (q : chart.QuantAct) (tables : kern.lut.LutTables)
    (tile b lane : Nat) (i : Std.Usize)
    (accs cur : Array Std.I64 32#usize) (v : Std.I64)
    (hiv : i.val = lane) (hn : lane < 32)
    (h : lut5_laneInv lut q tables tile b accs cur lane)
    (hv : v.val = lut5_blk lut q tables tile lane b) :
    lut5_laneInv lut q tables tile b accs (cur.set i v) (lane + 1) := by
  obtain ⟨h1, h2⟩ := h
  have hlen : lane < cur.val.length := by simpa using hn
  refine ⟨?_, ?_⟩
  · intro k hk
    rw [Array.set_val_eq, hiv]
    rcases Nat.lt_or_eq_of_le (Nat.lt_succ_iff.mp hk) with hlt | heq
    · rw [lut4_getD_set_ne cur.val lane k default v (Ne.symm (Nat.ne_of_lt hlt))]
      exact h1 k hlt
    · rw [heq, lut4_getD_set_self cur.val lane default v hlen]
      exact hv
  · intro k hle hk32
    rw [Array.set_val_eq, hiv]
    rw [lut4_getD_set_ne cur.val lane k default v (by omega)]
    exact h2 k (by omega) hk32

/-- The lane loop of `block_inners_portable` refines the block-value row vector. -/
theorem word_block_inners_lane_ok (hconv : lut3_fromU8I16_ok)
    (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables) (q : chart.QuantAct)
    (tile b : Std.Usize) (accs : Array Std.I64 32#usize)
    (hb2 : ∀ s, s < 4 → 4 * b.val + s ≤ Usize.max)
    (hg : ∀ s, s < 4 → (4 * b.val + s) * 8 ≤ Usize.max)
    (hsc : ∀ s, s < 4 → 4 * b.val + s < q.scales.val.length)
    (hidx : ∀ lane, lane < 32 → ∀ s, s < 4 →
      (tile.val * lut.n_groups.val + ((4 * b.val + s) * 8) + 8) * 32 + lane ≤ Usize.max)
    (hv : ∀ lane, lane < 32 → ∀ s, s < 4 → ∀ j, j < 8 →
      lut3_cidx lut.n_groups.val tile.val ((4 * b.val + s) * 8 + j) lane < lut.codes.val.length)
    (hloR : ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b.val + s) * 8 + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b.val + s) * 8 + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ s, s < 4 → ∀ j, j < 8 → (4 * b.val + s) * 8 + j < tables.xs.val.length)
    (hterm : ∀ lane, lane < 32 → ∀ s, s < 4 →
      lut4_absb (lut5_term lut q tables tile.val lane b.val s) 1152921504606846976) :
    ∃ r, kern.lut.block_inners_portable_loop0 lut tables q tile b accs 0#usize = ok r ∧
      ∀ k, k < 32 → (r.val.getD k default).val = lut5_blk lut q tables tile.val k b.val := by
  have hspec : kern.lut.block_inners_portable_loop0 lut tables q tile b accs 0#usize
      ⦃ r => ∀ k, k < 32 → (r.val.getD k default).val = lut5_blk lut q tables tile.val k b.val ⦄ := by
    unfold kern.lut.block_inners_portable_loop0
    apply loop.spec_decr_nat
      (measure := fun st : kern.lut.Pq2Lut × kern.lut.LutTables × Array Std.I64 32#usize × Std.Usize =>
        32 - st.2.2.2.val)
      (inv := fun st => st.1 = lut ∧ st.2.1 = tables ∧
        lut5_laneInv lut q tables tile.val b.val accs st.2.2.1 st.2.2.2.val)
    · rintro ⟨lut0, tbl0, accs0, lane0⟩ ⟨h1, h2, h3⟩
      dsimp only at h1 h2 h3 ⊢
      unfold kern.lut.block_inners_portable_loop0.body
      by_cases hl : lane0 < kern.lut.TILE_ROWS
      · rw [if_pos hl]
        have hl32 : lane0.val < 32 := by
          have h := hl
          simp only [kern.lut.TILE_ROWS] at h
          scalar_tac
        step*
        refine ⟨?_, ?_, ?_, ?_⟩
        · rw [lut1_post, h1]
        · rw [tables1_post, h2]
        · rw [lane1_post, a_post]
          exact lut5_laneInv_set lut q tables (↑tile) (↑b) (↑lane0) lane0 accs accs0 acc rfl hl32 h3
            (by rw [lut1_post1, h1, h2])
        · rw [lane1_post]; omega
      · rw [if_neg hl]
        simp only [WP.spec_ok]
        have hl32 : 32 ≤ lane0.val := by
          have h : ¬ ((↑lane0 : Nat) < (↑kern.lut.TILE_ROWS : Nat)) := by simpa using hl
          have hTR : (↑kern.lut.TILE_ROWS : Nat) = 32 := by simp only [kern.lut.TILE_ROWS]; rfl
          rw [hTR] at h
          omega
        intro k hk32
        exact h3.1 k (by omega)
    · refine ⟨rfl, rfl, ?_⟩
      simp [lut5_laneInv]
  obtain ⟨r, hr, hp⟩ := WP.spec_imp_exists hspec
  exact ⟨r, hr, hp⟩

/-! ## The top level

`block_inners_portable` is the lane loop started at lane `0`. -/

/-- `block_inners_portable` computes the block-value row vector. -/
theorem word_block_inners_portable_ok (hconv : lut3_fromU8I16_ok)
    (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables) (q : chart.QuantAct)
    (tile b : Std.Usize) (v : Array Std.I64 32#usize)
    (hb2 : ∀ s, s < 4 → 4 * b.val + s ≤ Usize.max)
    (hg : ∀ s, s < 4 → (4 * b.val + s) * 8 ≤ Usize.max)
    (hsc : ∀ s, s < 4 → 4 * b.val + s < q.scales.val.length)
    (hidx : ∀ lane, lane < 32 → ∀ s, s < 4 →
      (tile.val * lut.n_groups.val + ((4 * b.val + s) * 8) + 8) * 32 + lane ≤ Usize.max)
    (hv : ∀ lane, lane < 32 → ∀ s, s < 4 → ∀ j, j < 8 →
      lut3_cidx lut.n_groups.val tile.val ((4 * b.val + s) * 8 + j) lane < lut.codes.val.length)
    (hloR : ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b.val + s) * 8 + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b.val + s) * 8 + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ s, s < 4 → ∀ j, j < 8 → (4 * b.val + s) * 8 + j < tables.xs.val.length)
    (hterm : ∀ lane, lane < 32 → ∀ s, s < 4 →
      lut4_absb (lut5_term lut q tables tile.val lane b.val s) 1152921504606846976) :
    ∃ r, kern.lut.block_inners_portable lut tables q tile b v = ok r ∧
      ∀ k, k < 32 → (r.val.getD k default).val = lut5_blk lut q tables tile.val k b.val := by
  simpa only [kern.lut.block_inners_portable] using
    word_block_inners_lane_ok hconv lut tables q tile b v hb2 hg hsc hidx hv hloR hhiR hhb hxR
      hterm

/-! ## Refusal

`block_inners_portable` reads `lut.codes[(tile·n_groups + g)·32 + lane]` through `Vec::index`; on an
empty code plane the first group-loop read is out of range and the whole call returns
`arrayOutOfBounds`. -/

/-- The notation `x + y` on `Usize` is `UScalar.add`. -/
lemma lut5_hadd {x y : Std.Usize} : (x + y : Result Std.Usize) = UScalar.add x y := rfl

/-- The notation `x * y` on `Usize` is `UScalar.mul`. -/
lemma lut5_hmul {x y : Std.Usize} : (x * y : Result Std.Usize) = UScalar.mul x y := rfl

/-- `block_inners_portable`'s group loop refuses at the first group when the code plane is empty. -/
theorem word_block_inners_portable_refuses :
    kern.lut.block_inners_portable_loop0_loop0_loop0 1#usize lut4_witLut lut3_witTables
        0#usize 0#usize 0#usize 0#i64 0#i64 0#i64 0#usize
      = .fail .arrayOutOfBounds := by
  rw [kern.lut.block_inners_portable_loop0_loop0_loop0, loop.eq_def]
  simp only [kern.lut.block_inners_portable_loop0_loop0_loop0.body]
  simp only [lut5_hadd, lut5_hmul]
  simp [UScalar.add, UScalar.mul, UScalar.tryMk, UScalar.tryMkOpt, Result.ofOption,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
    alloc.vec.Vec.from_val, lut4_witLut]

/-! ## Non-vacuity

A witness with `n_groups = 8` and zero scales: `lut3_sval` is zero at every `(b, sb)`, so the
reference value `lut5_blk` is zero at every lane and the top level writes the zero row vector on
block `0`. The length hypotheses are satisfied by zero-entry planes (1024 code bytes, 512 entries in
each nibble plane, 32 group sums); `lut4_iv`'s concrete value does not enter because its scale factor
is zero. -/

/-- Witness `Pq2Lut`: 1024 zero code bytes, 8 groups, no scales. -/
def lut5_witLut : kern.lut.Pq2Lut :=
  { n_out := 0#usize, n_in := 0#usize, n_blocks := 0#usize, n_groups := 8#usize,
    n_tiles := 0#usize, scales := alloc.vec.Vec.from [] (by scalar_tac), d_max := 0#i128,
    codes := alloc.vec.Vec.from (List.replicate 1024 0#u8) (by scalar_tac),
    raw := alloc.vec.Vec.from [] (by scalar_tac) }

/-- Witness `QuantAct`: four zero scales and an empty code plane. -/
def lut5_witQuant : chart.QuantAct :=
  { scales := alloc.vec.Vec.from (List.replicate 4 0#i64) (by scalar_tac),
    codes := alloc.vec.Vec.from [] (by scalar_tac) }

/-- Witness table: 512 zero bytes in each nibble plane and 32 zero group sums. -/
def lut5_witTables : kern.lut.LutTables :=
  ⟨alloc.vec.Vec.from (List.replicate 512 0#u8) (by scalar_tac),
   alloc.vec.Vec.from (List.replicate 512 0#u8) (by scalar_tac),
   alloc.vec.Vec.from (List.replicate 32 0#i16) (by scalar_tac)⟩

/-- The witness has `8` groups. -/
lemma lut5_witLut_n_groups : lut5_witLut.n_groups.val = 8 := by
  simp only [lut5_witLut, Usize.ofNat, UScalar.ofNat, UScalar.ofNatCore_val_eq]

/-- The witness code plane has `1024` bytes. -/
lemma lut5_witLut_codes_len : lut5_witLut.codes.val.length = 1024 := by
  simp only [lut5_witLut, alloc.vec.Vec.from_val, List.length_replicate]

/-- The witness scale plane has four entries. -/
lemma lut5_witQuant_scales_len : lut5_witQuant.scales.val.length = 4 := by
  simp only [lut5_witQuant, alloc.vec.Vec.from_val, List.length_replicate]

/-- The witness low plane has `512` entries. -/
lemma lut5_witTables_lo_len : lut5_witTables.lo.val.length = 512 := by
  simp only [lut5_witTables, alloc.vec.Vec.from_val, List.length_replicate]

/-- The witness high plane has `512` entries. -/
lemma lut5_witTables_hi_len : lut5_witTables.hi.val.length = 512 := by
  simp only [lut5_witTables, alloc.vec.Vec.from_val, List.length_replicate]

/-- The witness group-sum plane has `32` entries. -/
lemma lut5_witTables_xs_len : lut5_witTables.xs.val.length = 32 := by
  simp only [lut5_witTables, alloc.vec.Vec.from_val, List.length_replicate]

/-- `getD` of the witness's zero-filled high plane is the zero byte at every index. -/
lemma lut5_witTables_hi_getD (m : Nat) : lut5_witTables.hi.val.getD m 0#u8 = 0#u8 := by
  have h : lut5_witTables.hi.val = List.replicate 512 0#u8 := by
    simp only [lut5_witTables, alloc.vec.Vec.from_val]
  rw [h]
  by_cases hm : m < 512
  · exact List.getD_replicate 0#u8 hm
  · exact List.getD_eq_default _ _ (by simp only [List.length_replicate]; omega)

/-- Every witness activation scale is zero. -/
lemma lut5_witQuant_sval (b s : Nat) : lut3_sval lut5_witQuant b s = 0 := by
  unfold lut3_sval
  have h : lut5_witQuant.scales.val = List.replicate 4 0#i64 := by
    simp only [lut5_witQuant, alloc.vec.Vec.from_val]
  rw [h]
  by_cases hm : 4 * b + s < 4
  · rw [List.getD_replicate 0#i64 hm]; simp
  · rw [List.getD_eq_default _ _ (by simp only [List.length_replicate]; omega)]; simp

/-- On the witness the reference block value is zero at every lane. -/
lemma lut5_wit_blk (k : Nat) :
    lut5_blk lut5_witLut lut5_witQuant lut5_witTables (↑(0#usize)) k (↑(0#usize)) = 0 := by
  unfold lut5_blk lut4_blk
  simp [lut5_witQuant_sval]

/-- The top level runs on the witness and writes the zero row vector: every hypothesis of
`word_block_inners_portable_ok` holds jointly at `(tile, b) = (0, 0)`. -/
example (hconv : lut3_fromU8I16_ok) :
    ∃ r, kern.lut.block_inners_portable lut5_witLut lut5_witTables lut5_witQuant 0#usize 0#usize
        (Array.repeat 32#usize 0#i64) = ok r ∧
      ∀ k, k < 32 → (r.val.getD k default).val = 0 := by
  have hmain := word_block_inners_portable_ok hconv lut5_witLut lut5_witTables lut5_witQuant
    0#usize 0#usize (Array.repeat 32#usize 0#i64)
    (by intro s hs; scalar_tac)
    (by intro s hs; scalar_tac)
    (by intro s hs; simp [lut5_witQuant_scales_len]; omega)
    (by intro lane hl s hs; simp [lut5_witLut_n_groups]; scalar_tac)
    (by intro lane hl s hs j hj; simp [lut3_cidx, lut5_witLut_n_groups, lut5_witLut_codes_len];
        omega)
    (by intro s hs j hj; simp [lut5_witTables_lo_len]; omega)
    (by intro s hs j hj; simp [lut5_witTables_hi_len]; omega)
    (by intro m _; rw [lut5_witTables_hi_getD m]; simp)
    (by intro s hs j hj; simp [lut5_witTables_xs_len]; omega)
    (by intro lane hl s hs; unfold lut5_term; rw [lut5_witQuant_sval]; simp [lut4_absb])
  obtain ⟨r, hr, hp⟩ := hmain
  exact ⟨r, hr, fun k hk => (hp k hk).trans (lut5_wit_blk k)⟩

end apeiron_word
