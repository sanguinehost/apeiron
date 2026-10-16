/-
Refinement of the outer loops of `tile_accs_portable` (`src/engine/kern/lut.rs`), group `kern.lut`,
continuing `Serve/Kern/R4/Lut3.lean` (which proves `word_tile_accs_inner`, the innermost group loop
`tile_accs_portable_loop0_loop0_loop0_loop0`).

`tile_accs_portable` walks the `(lane, block, sub-block, group)` nest. `Lut3` refines the group loop
to the three partial sums `lut3_isum`; this file refines the sub-block loop (`_loop0_loop0_loop0`),
the block loop (`_loop0_loop0`), the lane loop (`_loop0`) and the top level against the closed-form
row value `lut4_row` — a finite sum over `(block, sub-block, group)`.

Reference: `lut4_row lut q v tables tile lane = Σ_{b<n_blocks} w(tile,lane,b) ·
Σ_{sb<4} s(4b+sb) · (2·SH + SL − SX)(b,sb)`, where `SH`/`SL`/`SX` are the group sums of the two
`LutTables::p` planes and the `xs` plane. Stated as sums, not as the Rust loop.

Proved here (grade `proved`, axiom cone `{propext, Classical.choice, Quot.sound}`, under the stated
hypotheses):

* `word_tile_accs_sb_ok` — `_loop0_loop0_loop0`, the sub-block loop;
* `word_tile_accs_b_ok` — `_loop0_loop0`, the block loop;
* `word_tile_accs_lane_ok` — `_loop0`, the lane loop;
* `word_tile_accs_portable_ok` — the top level;
* `lut4_laneInv_set`, the one-lane step of the lane invariant;
* the non-vacuity witness `lut4_witLut` / `lut4_witQuant` and their `example`.

Open in this file, not proved:

* the sibling `block_inners_portable` chain (`_loop0` lane, `_loop0_loop0` sub-block,
  `_loop0_loop0_loop0` group) against `lut4_blk` read on `lut.codes` / `lut.n_groups`. Its group loop
  is `word_tile_accs_inner`'s body read on `lut.codes`; its sub-block and lane loops accumulate in
  `i64` (`acc + s1 * i8`) rather than `i128`, so they need their own `i64` value and no-overflow
  hypotheses. The chain is not written here and does not fit this file under the 1,500-line cap.
* the `_refuses` headlines: no violated-precondition (out-of-bounds) theorem is stated here.

Boundaries stated, not papered over:

* the `i16::from(u8)` model gap is inherited from `Lut3` as the hypothesis `lut3_fromU8I16_ok`; in
  the current model `I16.Insts.CoreConvertFromU8.from` is a `def`, so it is not an axiom and no
  headline depends on one;
* the byte planes are read as `getD` under the caller's length hypotheses and `hhb` keeps the high
  plane `≤ 127`;
* the reference bounds (`hdblk`, `hpsum`) are caller hypotheses: the `i128` row accumulation is
  stated as not overflowing, not derived from the scale magnitudes;
* `tile_accs_avx2` / `block_inners_avx2` and the `F32`/`F64` carriers are out of scope.
-/
import Serve.Kern.R4.Lut3

set_option autoImplicit false
set_option maxHeartbeats 800000

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## Reference: the closed-form row value -/

/-- `−C ≤ x ≤ C`, the bounded-value form used by the no-overflow side conditions. -/
def lut4_absb (x C : Int) : Prop := -C ≤ x ∧ x ≤ C

/-- The `i16` high plane stays `≤ 127` at every index (`getD … 0#u8` is `0` off the end, so no
length hypothesis is needed). -/
def lut4_hile (tables : kern.lut.LutTables) : Prop :=
  ∀ m, (tables.hi.val.getD m 0#u8).val ≤ 127

/-- The base group index of sub-block `sb` of block `b`: the Rust `(SUBS_PER_BLOCK·b + sb) ·
GROUPS_PER_SUB` with the closed values `4` and `8`. -/
def lut4_gidx (b sb : Nat) : Nat := (4 * b + sb) * 8

/-- The high-plane group sum of one sub-block. -/
def lut4_sh (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (n_groups tile lane b sb : Nat) : Int :=
  lut3_isum 8 (fun g => lut3_pval tables (lut4_gidx b sb + g)
    (lut3_hi v n_groups tile (lut4_gidx b sb + g) lane))

/-- The low-plane group sum of one sub-block. -/
def lut4_sl (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (n_groups tile lane b sb : Nat) : Int :=
  lut3_isum 8 (fun g => lut3_pval tables (lut4_gidx b sb + g)
    (lut3_lo v n_groups tile (lut4_gidx b sb + g) lane))

/-- The `xs` group sum of one sub-block. -/
def lut4_sx (_v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (_n_groups _tile _lane : Nat) (b sb : Nat) : Int :=
  lut3_isum 8 (fun g => lut3_xval tables (lut4_gidx b sb + g))

/-- The sub-block inner value `2·sum_hi + sum_lo − sum_x`. -/
def lut4_iv (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (n_groups tile lane b sb : Nat) : Int :=
  2 * lut4_sh v tables n_groups tile lane b sb + lut4_sl v tables n_groups tile lane b sb
    - lut4_sx v tables n_groups tile lane b sb

/-- The block value `Σ_{sb<4} scales[4b+sb] · inner`. -/
def lut4_blk (q : chart.QuantAct) (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (n_groups tile lane b : Nat) : Int :=
  lut3_isum 4 (fun sb => lut3_sval q b sb * lut4_iv v tables n_groups tile lane b sb)

/-- The row value of `(tile, lane)`: `Σ_{b<n_blocks} w(tile,lane,b) · blk b`. -/
def lut4_row (lut : kern.lut.Pq2Lut) (q : chart.QuantAct) (v : alloc.vec.Vec Std.U8)
    (tables : kern.lut.LutTables) (tile lane : Nat) : Int :=
  lut3_isum lut.n_blocks.val (fun b => lut3_dval lut tile lane b
    * lut4_blk q v tables lut.n_groups.val tile lane b)

/-! ## Finite-sum bounds -/

/-- `Σ_{j<n} f j ≤ n·C` when every term is `≤ C`. -/
lemma lut4_isum_le (n : Nat) (f : Nat → Int) (C : Int) (h : ∀ j, j < n → f j ≤ C) :
    lut3_isum n f ≤ (n : Int) * C := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [lut3_isum_succ]
    have hk := ih (fun j hj => h j (Nat.lt_trans hj (Nat.lt_succ_self k)))
    have hc := h k (Nat.lt_succ_self k)
    push_cast
    linarith

/-- `n·C ≤ Σ_{j<n} f j` when every term is `≥ C`. -/
lemma lut4_isum_ge (n : Nat) (f : Nat → Int) (C : Int) (h : ∀ j, j < n → C ≤ f j) :
    (n : Int) * C ≤ lut3_isum n f := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [lut3_isum_succ]
    have hk := ih (fun j hj => h j (Nat.lt_trans hj (Nat.lt_succ_self k)))
    have hc := h k (Nat.lt_succ_self k)
    push_cast
    linarith

/-- `|Σ_{j<n} f j| ≤ n·C` when every term is bounded by `C`. -/
lemma lut4_isum_absb (n : Nat) (f : Nat → Int) (C : Int)
    (h : ∀ j, j < n → lut4_absb (f j) C) : lut4_absb (lut3_isum n f) ((n : Int) * C) := by
  refine ⟨?_, lut4_isum_le n f C (fun j hj => (h j hj).2)⟩
  have h1 : (n : Int) * (-C) ≤ lut3_isum n f := lut4_isum_ge n f (-C) (fun j hj => (h j hj).1)
  rw [mul_neg] at h1
  linarith

/-! ## Value bounds on the reference -/

/-- `LutTables::p`'s value is at most `32259` under the high-plane hypothesis. -/
lemma lut4_pval_le (tables : kern.lut.LutTables) (h : lut4_hile tables) (g n : Nat) :
    lut3_pval tables g n ≤ 32259 :=
  lut3_pval_le tables g n (h (16 * g + n))

/-- Values, scalars and their widenings. -/
lemma lut4_u4 : (4#usize : Std.Usize).val = 4 := by simp
lemma lut4_u8 : (8#usize : Std.Usize).val = 8 := by simp
lemma lut4_u32 : (32#usize : Std.Usize).val = 32 := by simp
lemma lut4_u1 : (1#usize : Std.Usize).val = 1 := by simp
lemma lut4_i2 : (2#i64 : Std.I64).val = 2 := by simp
lemma lut4_i0 : (0#i64 : Std.I64).val = 0 := by simp

/-- The `i64 → i128` widening is the identity on values. -/
lemma lut4_fromI128I64_val (x : Std.I64) :
    (core.convert.num.FromI128I64.from x).val = x.val :=
  core.convert.num.FromI128I64.from_val_eq x

/-- `Vec::deref` is the identity on the underlying list. -/
lemma lut4_deref_val (v : alloc.vec.Vec Std.I64) :
    (↑v.deref : List Std.I64) = v.val := by
  unfold alloc.vec.Vec.deref
  exact Slice.from_val v.val v.property

/-- The scale read `i4 = scales[i3]` is the `getD` that `lut3_sval`/`lut3_dval` use. -/
lemma lut4_sval_read (v : alloc.vec.Vec Std.I64) (i3 : Std.Usize) (i4 : Std.I64)
    (hlt : i3.val < (↑v.deref : List Std.I64).length)
    (h4 : i4 = (↑v.deref)[↑i3]) :
    (↑i4 : Int) = ((↑v.deref : List Std.I64).getD i3.val 0#i64).val := by
  rw [h4]
  exact congrArg (fun y : Std.I64 => (↑y : Int))
    (getD_eq_getElem (↑v.deref) i3.val 0#i64 hlt).symm

/-- The `Vec → List` (not `deref`) scale read `i4 = scales[i3]` as the `getD` that `lut3_dval` uses:
the index of `getD` carries no proof, so the arithmetic reindexing is safe. -/
lemma lut4_sval_readv (v : alloc.vec.Vec Std.I64) (i3 : Std.Usize) (i4 : Std.I64)
    (hlt : i3.val < (↑v : List Std.I64).length)
    (h4 : i4 = (↑v : List Std.I64)[↑i3]) :
    (↑i4 : Int) = ((↑v : List Std.I64).getD i3.val 0#i64).val := by
  rw [h4]
  exact congrArg (fun y : Std.I64 => (↑y : Int))
    (getD_eq_getElem (↑v) i3.val 0#i64 hlt).symm

/-- An `i64` value is in `[−2^63, 2^63)`. -/
lemma lut4_i64_absb (x : Std.I64) : lut4_absb x.val 9223372036854775808 := by
  unfold lut4_absb
  have h := x.hBounds
  rw [IScalarTy.I64_numBits_eq] at h
  norm_num at h
  omega

/-- An `i128` value is in `[−2^127, 2^127)`. -/
lemma lut4_i128_absb (x : Std.I128) : lut4_absb x.val 170141183460469231731687303715884105728 := by
  unfold lut4_absb
  have h := x.hBounds
  rw [IScalarTy.I128_numBits_eq] at h
  norm_num at h
  omega

/-- The three group sums of a sub-block are bounded. -/
lemma lut4_sh_absb (tables : kern.lut.LutTables) (h : lut4_hile tables) (v : alloc.vec.Vec Std.U8)
    (n_groups tile lane b sb : Nat) :
    lut4_absb (lut4_sh v tables n_groups tile lane b sb) 258072 := by
  unfold lut4_absb lut4_sh
  constructor
  · have h1 := lut4_isum_ge 8 (fun g => lut3_pval tables (lut4_gidx b sb + g)
      (lut3_hi v n_groups tile (lut4_gidx b sb + g) lane)) (-508)
      (fun j _ => lut3_pval_ge tables (lut4_gidx b sb + j) (lut3_hi v n_groups tile (lut4_gidx b sb + j) lane))
    have h2 : ((8 : Nat) : Int) * (-508) = -4064 := by norm_num
    linarith
  · have h1 := lut4_isum_le 8 (fun g => lut3_pval tables (lut4_gidx b sb + g)
      (lut3_hi v n_groups tile (lut4_gidx b sb + g) lane)) 32259
      (fun j _ => lut4_pval_le tables h (lut4_gidx b sb + j) _)
    have h2 : ((8 : Nat) : Int) * 32259 = 258072 := by norm_num
    linarith

lemma lut4_sl_absb (tables : kern.lut.LutTables) (h : lut4_hile tables) (v : alloc.vec.Vec Std.U8)
    (n_groups tile lane b sb : Nat) :
    lut4_absb (lut4_sl v tables n_groups tile lane b sb) 258072 := by
  unfold lut4_absb lut4_sl
  constructor
  · have h1 := lut4_isum_ge 8 (fun g => lut3_pval tables (lut4_gidx b sb + g)
      (lut3_lo v n_groups tile (lut4_gidx b sb + g) lane)) (-508)
      (fun j _ => lut3_pval_ge tables (lut4_gidx b sb + j) (lut3_lo v n_groups tile (lut4_gidx b sb + j) lane))
    have h2 : ((8 : Nat) : Int) * (-508) = -4064 := by norm_num
    linarith
  · have h1 := lut4_isum_le 8 (fun g => lut3_pval tables (lut4_gidx b sb + g)
      (lut3_lo v n_groups tile (lut4_gidx b sb + g) lane)) 32259
      (fun j _ => lut4_pval_le tables h (lut4_gidx b sb + j) _)
    have h2 : ((8 : Nat) : Int) * 32259 = 258072 := by norm_num
    linarith

lemma lut4_sx_absb (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (n_groups tile lane b sb : Nat) :
    lut4_absb (lut4_sx v tables n_groups tile lane b sb) 262144 := by
  unfold lut4_absb lut4_sx
  constructor
  · have h1 := lut4_isum_ge 8 (fun g => lut3_xval tables (lut4_gidx b sb + g)) (-32768)
      (fun j _ => (lut3_xval_bounds tables (lut4_gidx b sb + j)).1)
    have h2 : ((8 : Nat) : Int) * (-32768) = -262144 := by norm_num
    linarith
  · have h1 := lut4_isum_le 8 (fun g => lut3_xval tables (lut4_gidx b sb + g)) 32767
      (fun j _ => (lut3_xval_bounds tables (lut4_gidx b sb + j)).2)
    have h2 : ((8 : Nat) : Int) * 32767 = 262136 := by norm_num
    linarith

/-- The sub-block inner value is bounded by `2^20`. -/
lemma lut4_iv_absb (tables : kern.lut.LutTables) (h : lut4_hile tables) (v : alloc.vec.Vec Std.U8)
    (n_groups tile lane b sb : Nat) :
    lut4_absb (lut4_iv v tables n_groups tile lane b sb) 1048576 := by
  unfold lut4_absb lut4_iv
  have h1 := lut4_sh_absb tables h v n_groups tile lane b sb
  have h2 := lut4_sl_absb tables h v n_groups tile lane b sb
  have h3 := lut4_sx_absb v tables n_groups tile lane b sb
  unfold lut4_absb at h1 h2 h3
  constructor <;> linarith

/-- A scale times a sub-block inner value is bounded by `2^83`. -/
lemma lut4_sval_iv_absb (tables : kern.lut.LutTables) (h : lut4_hile tables) (q : chart.QuantAct)
    (v : alloc.vec.Vec Std.U8) (n_groups tile lane b sb : Nat) :
    lut4_absb (lut3_sval q b sb * lut4_iv v tables n_groups tile lane b sb) 9671406556917033397649408 := by
  have h1 := lut4_i64_absb (q.scales.val.getD (4 * b + sb) 0#i64)
  have h2 := lut4_iv_absb tables h v n_groups tile lane b sb
  unfold lut4_absb at h1 h2
  unfold lut4_absb
  rw [show lut3_sval q b sb = (q.scales.val.getD (4 * b + sb) 0#i64).val from rfl]
  constructor <;> nlinarith

/-- The block value is bounded by `2^85`. -/
lemma lut4_blk_absb (tables : kern.lut.LutTables) (h : lut4_hile tables) (q : chart.QuantAct)
    (v : alloc.vec.Vec Std.U8) (n_groups tile lane b : Nat) :
    lut4_absb (lut4_blk q v tables n_groups tile lane b) 38685626227668133590597632 := by
  have h0 := lut4_isum_absb 4 (fun sb => lut3_sval q b sb * lut4_iv v tables n_groups tile lane b sb)
    9671406556917033397649408 (fun j _ => lut4_sval_iv_absb tables h q v n_groups tile lane b j)
  have h2 : ((4 : Nat) : Int) * 9671406556917033397649408 = 38685626227668133590597632 := by norm_num
  rw [h2] at h0
  exact h0

/-! ## Bounds on the reference sums, and the inner-loop step -/

/-- Monotonicity of the bounded-value form. -/
lemma lut4_absb_mono {x A B : Int} (h : lut4_absb x A) (hAB : A ≤ B) : lut4_absb x B := by
  unfold lut4_absb at h ⊢
  constructor <;> linarith

/-- `lut4_hile` from the length-bounded form the inner loop carries. -/
lemma lut4_hile_of (tables : kern.lut.LutTables)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127) :
    lut4_hile tables := by
  intro m
  by_cases h : m < tables.hi.val.length
  · exact hhb m h
  · rw [List.getD_eq_default _ _ (by simpa using h)]
    simp

/-- `LutTables::p`'s value is bounded by `32259` in absolute value. -/
lemma lut4_pval_absb (tables : kern.lut.LutTables) (hile : lut4_hile tables) (g n : Nat) :
    lut4_absb (lut3_pval tables g n) 32259 := by
  unfold lut4_absb
  exact ⟨by have h := lut3_pval_ge tables g n; linarith,
    lut3_pval_le tables g n (hile (16 * g + n))⟩

/-- `xs[g]` is bounded by `32768` in absolute value. -/
lemma lut4_xval_absb (tables : kern.lut.LutTables) (g : Nat) :
    lut4_absb (lut3_xval tables g) 32768 := by
  unfold lut4_absb
  exact ⟨by have h := (lut3_xval_bounds tables g).1; linarith,
    by have h := (lut3_xval_bounds tables g).2; linarith⟩

/-- A sum of at most eight `p`-values is bounded by `258072`. -/
lemma lut4_absb8 (k : Nat) (f : Nat → Int) (hk : k ≤ 8)
    (h : ∀ j, j < k → lut4_absb (f j) 32259) : lut4_absb (lut3_isum k f) 258072 := by
  refine lut4_absb_mono (lut4_isum_absb k f 32259 h) ?_
  have hk' : (k : Int) ≤ 8 := by exact_mod_cast hk
  nlinarith

/-- The `i64` headroom form `lut3_rng` holds for any value bounded by `C ≤ I64.max − 33000`. -/
lemma lut4_rng_of_absb {x C : Int} (h : lut4_absb x C) (hC : C ≤ 9223372036821772807) :
    lut3_rng x := by
  unfold lut4_absb at h
  unfold lut3_rng
  rw [I64.min_eq, I64.max_eq]
  constructor <;> omega

lemma lut4_rng_hi (tables : kern.lut.LutTables) (hile : lut4_hile tables)
    (v : alloc.vec.Vec Std.U8) (i1 tile lane gbase k : Nat) (hk : k ≤ 8) :
    lut3_rng (lut3_isum k (fun j => lut3_pval tables (gbase + j)
      (lut3_hi v i1 tile (gbase + j) lane))) :=
  lut4_rng_of_absb (lut4_absb8 k _ hk (fun j _ =>
    lut4_pval_absb tables hile (gbase + j) (lut3_hi v i1 tile (gbase + j) lane))) (by norm_num)

lemma lut4_rng_lo (tables : kern.lut.LutTables) (hile : lut4_hile tables)
    (v : alloc.vec.Vec Std.U8) (i1 tile lane gbase k : Nat) (hk : k ≤ 8) :
    lut3_rng (lut3_isum k (fun j => lut3_pval tables (gbase + j)
      (lut3_lo v i1 tile (gbase + j) lane))) :=
  lut4_rng_of_absb (lut4_absb8 k _ hk (fun j _ =>
    lut4_pval_absb tables hile (gbase + j) (lut3_lo v i1 tile (gbase + j) lane))) (by norm_num)

lemma lut4_rng_x (tables : kern.lut.LutTables) (gbase k : Nat) (hk : k ≤ 8) :
    lut3_rng (lut3_isum k (fun j => lut3_xval tables (gbase + j))) := by
  refine lut4_rng_of_absb (C := 262144) ?_ (by norm_num)
  refine lut4_absb_mono
    (lut4_isum_absb k _ 32768 (fun j _ => lut4_xval_absb tables (gbase + j))) ?_
  have hk' : (k : Int) ≤ 8 := by exact_mod_cast hk
  nlinarith

/-- `word_tile_accs_inner` in weakest-precondition form, for `step*`. -/
@[step]
lemma lut4_inner_step (hconv : lut3_fromU8I16_ok)
    (i i1 : Std.Usize) (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (tile lane gbase : Std.Usize) (sh sl sx : Std.I64) (g8 : Std.Usize)
    (hinv : lut3_innerInv v tables i1.val tile.val lane.val gbase.val i.val sh sl sx g8.val)
    (hidx : (tile.val * i1.val + gbase.val + i.val) * 32 + lane.val ≤ Usize.max)
    (hv : ∀ j, j < i.val → lut3_cidx i1.val tile.val (gbase.val + j) lane.val < v.val.length)
    (hloR : ∀ j, j < i.val → 16 * (gbase.val + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ j, j < i.val → 16 * (gbase.val + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ j, j < i.val → gbase.val + j < tables.xs.val.length)
    (hrng : ∀ k, k ≤ i.val →
      lut3_rng (lut3_isum k (fun j => lut3_pval tables (gbase.val + j)
        (lut3_hi v i1.val tile.val (gbase.val + j) lane.val))) ∧
      lut3_rng (lut3_isum k (fun j => lut3_pval tables (gbase.val + j)
        (lut3_lo v i1.val tile.val (gbase.val + j) lane.val))) ∧
      lut3_rng (lut3_isum k (fun j => lut3_xval tables (gbase.val + j)))) :
    kern.lut.tile_accs_portable_loop0_loop0_loop0_loop0 i i1 v tables tile lane sh sl sx gbase g8
      ⦃ r => lut3_innerPost v tables i1.val tile.val lane.val gbase.val i.val r ⦄ := by
  obtain ⟨r, hr, hp⟩ := word_tile_accs_inner hconv i i1 v tables tile lane gbase sh sl sx g8
    hinv hidx hv hloR hhiR hhb hxR hrng
  rw [hr]
  simp only [WP.spec_ok]
  exact hp

/-- The fixed-width inner call of the sub-block loop: bound `8`, zero accumulators, with the
inner-loop side hypotheses supplied. -/
@[step]
lemma lut4_inner8_step (hconv : lut3_fromU8I16_ok)
    (i1 : Std.Usize) (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (tile lane gbase : Std.Usize)
    (hidx : (tile.val * i1.val + gbase.val + 8) * 32 + lane.val ≤ Usize.max)
    (hv : ∀ j, j < 8 → lut3_cidx i1.val tile.val (gbase.val + j) lane.val < v.val.length)
    (hloR : ∀ j, j < 8 → 16 * (gbase.val + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ j, j < 8 → 16 * (gbase.val + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ j, j < 8 → gbase.val + j < tables.xs.val.length) :
    kern.lut.tile_accs_portable_loop0_loop0_loop0_loop0 8#usize i1 v tables tile lane
      0#i64 0#i64 0#i64 gbase 0#usize
      ⦃ r => lut3_innerPost v tables i1.val tile.val lane.val gbase.val 8 r ⦄ := by
  have h8 : ((8#usize) : Std.Usize).val = 8 := by simp
  have hile := lut4_hile_of tables hhb
  refine lut4_inner_step hconv 8#usize i1 v tables tile lane gbase 0#i64 0#i64 0#i64 0#usize
    ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · unfold lut3_innerInv
    refine ⟨by simp, ?_, ?_, ?_⟩ <;> simp [lut3_isum_zero]
  · rw [h8]; exact hidx
  · intro j hj; rw [h8] at hj; exact hv j hj
  · intro j hj; rw [h8] at hj; exact hloR j hj
  · intro j hj; rw [h8] at hj; exact hhiR j hj
  · exact hhb
  · intro j hj; rw [h8] at hj; exact hxR j hj
  · intro k hk
    rw [h8] at hk
    exact ⟨lut4_rng_hi tables hile v i1.val tile.val lane.val gbase.val k hk,
           lut4_rng_lo tables hile v i1.val tile.val lane.val gbase.val k hk,
           lut4_rng_x tables gbase.val k hk⟩

/-! ## Bounds on the inner-loop post-state -/

/-- Unfolding of `lut3_innerPost` at bound `8` into its four conjuncts. -/
lemma lut4_innerPost_eq (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (n_groups tile lane gbase : Nat) (tables1 : kern.lut.LutTables)
    (sum_hi sum_lo sum_x : Std.I64)
    (h : lut3_innerPost v tables n_groups tile lane gbase 8 (tables1, sum_hi, sum_lo, sum_x)) :
    tables1 = tables ∧
    (↑sum_hi : Int) = lut3_isum 8 (fun j => lut3_pval tables (gbase + j)
      (lut3_hi v n_groups tile (gbase + j) lane)) ∧
    (↑sum_lo : Int) = lut3_isum 8 (fun j => lut3_pval tables (gbase + j)
      (lut3_lo v n_groups tile (gbase + j) lane)) ∧
    (↑sum_x : Int) = lut3_isum 8 (fun j => lut3_xval tables (gbase + j)) := by
  unfold lut3_innerPost at h
  exact h

/-- The high-plane sub-block sum is bounded by `258072`. -/
lemma lut4_sum_hi_absb (tables : kern.lut.LutTables) (hile : lut4_hile tables)
    (v : alloc.vec.Vec Std.U8) (n_groups tile lane gbase : Nat) :
    lut4_absb (lut3_isum 8 (fun j => lut3_pval tables (gbase + j)
      (lut3_hi v n_groups tile (gbase + j) lane))) 258072 :=
  lut4_absb8 8 _ (le_refl 8) (fun j _ =>
    lut4_pval_absb tables hile (gbase + j) (lut3_hi v n_groups tile (gbase + j) lane))

/-- The low-plane sub-block sum is bounded by `258072`. -/
lemma lut4_sum_lo_absb (tables : kern.lut.LutTables) (hile : lut4_hile tables)
    (v : alloc.vec.Vec Std.U8) (n_groups tile lane gbase : Nat) :
    lut4_absb (lut3_isum 8 (fun j => lut3_pval tables (gbase + j)
      (lut3_lo v n_groups tile (gbase + j) lane))) 258072 :=
  lut4_absb8 8 _ (le_refl 8) (fun j _ =>
    lut4_pval_absb tables hile (gbase + j) (lut3_lo v n_groups tile (gbase + j) lane))

/-- The `xs` sub-block sum is bounded by `262144`. -/
lemma lut4_sum_x_absb (tables : kern.lut.LutTables) (gbase : Nat) :
    lut4_absb (lut3_isum 8 (fun j => lut3_xval tables (gbase + j))) 262144 :=
  lut4_absb_mono (lut4_isum_absb 8 _ 32768 (fun j _ => lut4_xval_absb tables (gbase + j)))
    (by norm_num)

/-- A sum of at most four scale-inner products is bounded by `2^85`. -/
lemma lut4_isum4_absb (f : Nat → Int) (k : Nat) (hk : k ≤ 4)
    (h : ∀ j, j < k → lut4_absb (f j) 9671406556917033397649408) :
    lut4_absb (lut3_isum k f) 38685626227668133590597632 := by
  refine lut4_absb_mono (lut4_isum_absb k f 9671406556917033397649408 h) ?_
  have hk' : (k : Int) ≤ 4 := by exact_mod_cast hk
  nlinarith

/-- Any `i128`-bounded value fits `i128`. -/
lemma lut4_i128_range {x C : Int} (h : lut4_absb x C)
    (hC : C ≤ 170141183460469231731687303715884105727) :
    I128.min ≤ x ∧ x ≤ I128.max := by
  unfold lut4_absb at h
  rw [I128.min_eq, I128.max_eq]
  constructor <;> omega

/-- The product `scale · inner` of a sub-block, as the body computes it. -/
lemma lut4_i11_eq (q : chart.QuantAct) (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (i1 tile lane b sb gbase : Nat) (i4 inner : Std.I64)
    (hbase : gbase = lut4_gidx b sb) (hi4 : (↑i4 : Int) = lut3_sval q b sb)
    (hinner : (↑inner : Int) = 2 * (lut3_isum 8 (fun j => lut3_pval tables (gbase + j)
        (lut3_hi v i1 tile (gbase + j) lane)))
      + (lut3_isum 8 (fun j => lut3_pval tables (gbase + j)
        (lut3_lo v i1 tile (gbase + j) lane)))
      - (lut3_isum 8 (fun j => lut3_xval tables (gbase + j)))) :
    (↑i4 : Int) * (↑inner : Int) = lut3_sval q b sb * lut4_iv v tables i1 tile lane b sb := by
  rw [hinner, hi4, hbase]
  unfold lut4_iv lut4_sh lut4_sl lut4_sx
  rfl

/-! ## The sub-block loop -/

theorem word_tile_accs_sb_ok (hconv : lut3_fromU8I16_ok)
    (i : Std.Usize) (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (q : chart.QuantAct) (tile lane b : Std.Usize)
    (hb2 : ∀ s, s < 4 → 4 * b.val + s ≤ Usize.max)
    (hg : ∀ s, s < 4 → (4 * b.val + s) * 8 ≤ Usize.max)
    (hsc : ∀ s, s < 4 → 4 * b.val + s < q.scales.val.length)
    (hidx : ∀ s, s < 4 → (tile.val * i.val + ((4 * b.val + s) * 8) + 8) * 32 + lane.val ≤ Usize.max)
    (hv : ∀ s, s < 4 → ∀ j, j < 8 →
      lut3_cidx i.val tile.val ((4 * b.val + s) * 8 + j) lane.val < v.val.length)
    (hloR : ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b.val + s) * 8 + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b.val + s) * 8 + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ s, s < 4 → ∀ j, j < 8 → (4 * b.val + s) * 8 + j < tables.xs.val.length) :
    ∃ r, kern.lut.tile_accs_portable_loop0_loop0_loop0 i v tables q tile lane b 0#i128 0#usize
        = ok r ∧
      r.1 = tables ∧ r.2.val = lut4_blk q v tables i.val tile.val lane.val b.val := by
  have hspec : kern.lut.tile_accs_portable_loop0_loop0_loop0 i v tables q tile lane b 0#i128
      0#usize ⦃ r => r.1 = tables ∧ r.2.val = lut4_blk q v tables i.val tile.val lane.val b.val ⦄ := by
    unfold kern.lut.tile_accs_portable_loop0_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun st : kern.lut.LutTables × Std.I128 × Std.Usize => 4 - st.2.2.val)
      (inv := fun st : kern.lut.LutTables × Std.I128 × Std.Usize =>
        st.1 = tables ∧ st.2.2.val ≤ 4 ∧
        st.2.1.val = lut3_isum st.2.2.val
          (fun sb => lut3_sval q b.val sb * lut4_iv v tables i.val tile.val lane.val b.val sb))
      (post := fun r : kern.lut.LutTables × Std.I128 =>
        r.1 = tables ∧ r.2.val = lut4_blk q v tables i.val tile.val lane.val b.val)
    · rintro ⟨tbl0, ib1, sb0⟩ ⟨htab, hsb4, hib⟩
      dsimp only at htab hsb4 hib ⊢
      unfold kern.lut.tile_accs_portable_loop0_loop0_loop0.body
      rw [lut3_subs_per_block]
      simp only [bind_tc_ok]
      by_cases hc : sb0 < (4#usize : Std.Usize)
      · rw [if_pos hc]
        simp only [lift, bind_tc_ok]
        unfold chart.QuantAct.impl.scales
        simp only [bind_tc_ok]
        rw [lut3_groups_per_sub]
        simp only [bind_tc_ok]
        have hsbv : sb0.val < 4 := by scalar_tac
        have hile0 : lut4_hile tables := lut4_hile_of tables hhb
        have hile1 : lut4_hile tbl0 := by rw [htab]; exact hile0
        step*
        · rw [i3_post, i2_post]
          simpa only [Slice.length, lut4_deref_val] using hsc (↑sb0) hsbv
        · have h := hb2 (↑sb0) hsbv; scalar_tac
        · rw [i5_post]; have h := hb2 (↑sb0) hsbv; scalar_tac
        · rw [i6_post, i5_post]; have h := hg (↑sb0) hsbv; scalar_tac
        · rw [gbase_post, i6_post, i5_post]; exact hidx (↑sb0) hsbv
        · intro j hj; rw [gbase_post, i6_post, i5_post]; exact hv (↑sb0) hsbv j hj
        · intro j hj; rw [htab, gbase_post, i6_post, i5_post]
          exact hloR (↑sb0) hsbv j hj
        · intro j hj; rw [htab, gbase_post, i6_post, i5_post]
          exact hhiR (↑sb0) hsbv j hj
        · intro j hj; rw [htab, gbase_post, i6_post, i5_post]
          exact hxR (↑sb0) hsbv j hj
        · obtain ⟨-, hsh, hsl, hsx⟩ := lut4_innerPost_eq v tbl0 (↑i) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x tables1_post
          rw [lut4_i2, hsh]
          have h := lut4_sum_hi_absb tbl0 hile1 v (↑i) (↑tile) (↑lane) (↑gbase)
          unfold lut4_absb at h
          rw [I64.min_eq]; omega
        · obtain ⟨-, hsh, hsl, hsx⟩ := lut4_innerPost_eq v tbl0 (↑i) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x tables1_post
          rw [lut4_i2, hsh]
          have h := lut4_sum_hi_absb tbl0 hile1 v (↑i) (↑tile) (↑lane) (↑gbase)
          unfold lut4_absb at h
          rw [I64.max_eq]; omega
        · obtain ⟨-, hsh, hsl, hsx⟩ := lut4_innerPost_eq v tbl0 (↑i) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x tables1_post
          rw [i8_post, hsh, hsl]
          have h1 := lut4_sum_hi_absb tbl0 hile1 v (↑i) (↑tile) (↑lane) (↑gbase)
          have h2 := lut4_sum_lo_absb tbl0 hile1 v (↑i) (↑tile) (↑lane) (↑gbase)
          unfold lut4_absb at h1 h2
          rw [I64.min_eq]; omega
        · obtain ⟨-, hsh, hsl, hsx⟩ := lut4_innerPost_eq v tbl0 (↑i) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x tables1_post
          rw [i8_post, hsh, hsl]
          have h1 := lut4_sum_hi_absb tbl0 hile1 v (↑i) (↑tile) (↑lane) (↑gbase)
          have h2 := lut4_sum_lo_absb tbl0 hile1 v (↑i) (↑tile) (↑lane) (↑gbase)
          unfold lut4_absb at h1 h2
          rw [I64.max_eq]; omega
        · obtain ⟨-, hsh, hsl, hsx⟩ := lut4_innerPost_eq v tbl0 (↑i) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x tables1_post
          rw [i9_post, i8_post, hsh, hsl, hsx]
          have h1 := lut4_sum_hi_absb tbl0 hile1 v (↑i) (↑tile) (↑lane) (↑gbase)
          have h2 := lut4_sum_lo_absb tbl0 hile1 v (↑i) (↑tile) (↑lane) (↑gbase)
          have h3 := lut4_sum_x_absb tbl0 (↑gbase)
          unfold lut4_absb at h1 h2 h3
          rw [I64.min_eq]; omega
        · obtain ⟨-, hsh, hsl, hsx⟩ := lut4_innerPost_eq v tbl0 (↑i) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x tables1_post
          rw [i9_post, i8_post, hsh, hsl, hsx]
          have h1 := lut4_sum_hi_absb tbl0 hile1 v (↑i) (↑tile) (↑lane) (↑gbase)
          have h2 := lut4_sum_lo_absb tbl0 hile1 v (↑i) (↑tile) (↑lane) (↑gbase)
          have h3 := lut4_sum_x_absb tbl0 (↑gbase)
          unfold lut4_absb at h1 h2 h3
          rw [I64.max_eq]; omega
        · obtain ⟨-, hsh, hsl, hsx⟩ := lut4_innerPost_eq v tbl0 (↑i) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x tables1_post
          have hbase : (↑gbase : Nat) = lut4_gidx (↑b) (↑sb0) := by
            rw [gbase_post, i6_post, i5_post]; rfl
          have hi4 : (↑i4 : Int) = lut3_sval q (↑b) (↑sb0) := by
            have hlt : (↑i3 : Nat) < (↑q.scales.deref : List Std.I64).length := by
              rw [i3_post, i2_post]
              simpa only [lut4_deref_val] using hsc (↑sb0) hsbv
            rw [lut4_sval_read q.scales i3 i4 hlt i4_post, lut4_deref_val, i3_post, i2_post]
            unfold lut3_sval
            rfl
          have hinner : (↑inner : Int) = 2 * (lut3_isum 8 (fun j => lut3_pval tbl0 (↑gbase + j)
              (lut3_hi v (↑i) (↑tile) (↑gbase + j) (↑lane))))
            + (lut3_isum 8 (fun j => lut3_pval tbl0 (↑gbase + j)
              (lut3_lo v (↑i) (↑tile) (↑gbase + j) (↑lane))))
            - (lut3_isum 8 (fun j => lut3_xval tbl0 (↑gbase + j))) := by
            rw [inner_post, i9_post, i8_post, hsh, hsl, hsx]
          have hiv := lut4_i11_eq q v tbl0 (↑i) (↑tile) (↑lane) (↑b) (↑sb0) (↑gbase)
            i4 inner hbase hi4 hinner
          rw [htab] at hiv
          rw [lut4_fromI128I64_val, lut4_fromI128I64_val, hiv]
          exact (lut4_i128_range (lut4_sval_iv_absb tables hile0 q v (↑i) (↑tile) (↑lane)
            (↑b) (↑sb0)) (by norm_num)).1
        · obtain ⟨-, hsh, hsl, hsx⟩ := lut4_innerPost_eq v tbl0 (↑i) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x tables1_post
          have hbase : (↑gbase : Nat) = lut4_gidx (↑b) (↑sb0) := by
            rw [gbase_post, i6_post, i5_post]; rfl
          have hi4 : (↑i4 : Int) = lut3_sval q (↑b) (↑sb0) := by
            have hlt : (↑i3 : Nat) < (↑q.scales.deref : List Std.I64).length := by
              rw [i3_post, i2_post]
              simpa only [lut4_deref_val] using hsc (↑sb0) hsbv
            rw [lut4_sval_read q.scales i3 i4 hlt i4_post, lut4_deref_val, i3_post, i2_post]
            unfold lut3_sval
            rfl
          have hinner : (↑inner : Int) = 2 * (lut3_isum 8 (fun j => lut3_pval tbl0 (↑gbase + j)
              (lut3_hi v (↑i) (↑tile) (↑gbase + j) (↑lane))))
            + (lut3_isum 8 (fun j => lut3_pval tbl0 (↑gbase + j)
              (lut3_lo v (↑i) (↑tile) (↑gbase + j) (↑lane))))
            - (lut3_isum 8 (fun j => lut3_xval tbl0 (↑gbase + j))) := by
            rw [inner_post, i9_post, i8_post, hsh, hsl, hsx]
          have hiv := lut4_i11_eq q v tbl0 (↑i) (↑tile) (↑lane) (↑b) (↑sb0) (↑gbase)
            i4 inner hbase hi4 hinner
          rw [htab] at hiv
          rw [lut4_fromI128I64_val, lut4_fromI128I64_val, hiv]
          exact (lut4_i128_range (lut4_sval_iv_absb tables hile0 q v (↑i) (↑tile) (↑lane)
            (↑b) (↑sb0)) (by norm_num)).2
        · obtain ⟨-, hsh, hsl, hsx⟩ := lut4_innerPost_eq v tbl0 (↑i) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x tables1_post
          have hi11 : (↑i11 : Int) = lut3_sval q (↑b) (↑sb0)
              * lut4_iv v tables (↑i) (↑tile) (↑lane) (↑b) (↑sb0) := by
            have hbase : (↑gbase : Nat) = lut4_gidx (↑b) (↑sb0) := by
              rw [gbase_post, i6_post, i5_post]; rfl
            have hi4 : (↑i4 : Int) = lut3_sval q (↑b) (↑sb0) := by
              have hlt : (↑i3 : Nat) < (↑q.scales.deref : List Std.I64).length := by
                rw [i3_post, i2_post]
                simpa only [lut4_deref_val] using hsc (↑sb0) hsbv
              rw [lut4_sval_read q.scales i3 i4 hlt i4_post, lut4_deref_val, i3_post, i2_post]
              unfold lut3_sval
              rfl
            have hinner : (↑inner : Int) = 2 * (lut3_isum 8 (fun j => lut3_pval tbl0 (↑gbase + j)
                (lut3_hi v (↑i) (↑tile) (↑gbase + j) (↑lane))))
              + (lut3_isum 8 (fun j => lut3_pval tbl0 (↑gbase + j)
                (lut3_lo v (↑i) (↑tile) (↑gbase + j) (↑lane))))
              - (lut3_isum 8 (fun j => lut3_xval tbl0 (↑gbase + j))) := by
              rw [inner_post, i9_post, i8_post, hsh, hsl, hsx]
            have h0 := lut4_i11_eq q v tbl0 (↑i) (↑tile) (↑lane) (↑b) (↑sb0) (↑gbase)
              i4 inner hbase hi4 hinner
            rw [htab] at h0
            rw [i11_post, lut4_fromI128I64_val, lut4_fromI128I64_val]
            exact h0
          rw [hi11, hib]
          have h1 := lut4_isum4_absb
            (fun sb => lut3_sval q (↑b) sb * lut4_iv v tables (↑i) (↑tile) (↑lane) (↑b) sb)
            (↑sb0) hsb4 (fun j _ => lut4_sval_iv_absb tables hile0 q v (↑i) (↑tile) (↑lane) (↑b) j)
          have h2 := lut4_sval_iv_absb tables hile0 q v (↑i) (↑tile) (↑lane) (↑b) (↑sb0)
          unfold lut4_absb at h1 h2
          rw [I128.min_eq]; omega
        · obtain ⟨-, hsh, hsl, hsx⟩ := lut4_innerPost_eq v tbl0 (↑i) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x tables1_post
          have hi11 : (↑i11 : Int) = lut3_sval q (↑b) (↑sb0)
              * lut4_iv v tables (↑i) (↑tile) (↑lane) (↑b) (↑sb0) := by
            have hbase : (↑gbase : Nat) = lut4_gidx (↑b) (↑sb0) := by
              rw [gbase_post, i6_post, i5_post]; rfl
            have hi4 : (↑i4 : Int) = lut3_sval q (↑b) (↑sb0) := by
              have hlt : (↑i3 : Nat) < (↑q.scales.deref : List Std.I64).length := by
                rw [i3_post, i2_post]
                simpa only [lut4_deref_val] using hsc (↑sb0) hsbv
              rw [lut4_sval_read q.scales i3 i4 hlt i4_post, lut4_deref_val, i3_post, i2_post]
              unfold lut3_sval
              rfl
            have hinner : (↑inner : Int) = 2 * (lut3_isum 8 (fun j => lut3_pval tbl0 (↑gbase + j)
                (lut3_hi v (↑i) (↑tile) (↑gbase + j) (↑lane))))
              + (lut3_isum 8 (fun j => lut3_pval tbl0 (↑gbase + j)
                (lut3_lo v (↑i) (↑tile) (↑gbase + j) (↑lane))))
              - (lut3_isum 8 (fun j => lut3_xval tbl0 (↑gbase + j))) := by
              rw [inner_post, i9_post, i8_post, hsh, hsl, hsx]
            have h0 := lut4_i11_eq q v tbl0 (↑i) (↑tile) (↑lane) (↑b) (↑sb0) (↑gbase)
              i4 inner hbase hi4 hinner
            rw [htab] at h0
            rw [i11_post, lut4_fromI128I64_val, lut4_fromI128I64_val]
            exact h0
          rw [hi11, hib]
          have h1 := lut4_isum4_absb
            (fun sb => lut3_sval q (↑b) sb * lut4_iv v tables (↑i) (↑tile) (↑lane) (↑b) sb)
            (↑sb0) hsb4 (fun j _ => lut4_sval_iv_absb tables hile0 q v (↑i) (↑tile) (↑lane) (↑b) j)
          have h2 := lut4_sval_iv_absb tables hile0 q v (↑i) (↑tile) (↑lane) (↑b) (↑sb0)
          unfold lut4_absb at h1 h2
          rw [I128.max_eq]; omega
        · obtain ⟨htab1, hsh, hsl, hsx⟩ := lut4_innerPost_eq v tbl0 (↑i) (↑tile) (↑lane) (↑gbase)
            tables1 sum_hi sum_lo sum_x tables1_post
          have hi11 : (↑i11 : Int) = lut3_sval q (↑b) (↑sb0)
              * lut4_iv v tables (↑i) (↑tile) (↑lane) (↑b) (↑sb0) := by
            have hbase : (↑gbase : Nat) = lut4_gidx (↑b) (↑sb0) := by
              rw [gbase_post, i6_post, i5_post]; rfl
            have hi4 : (↑i4 : Int) = lut3_sval q (↑b) (↑sb0) := by
              have hlt : (↑i3 : Nat) < (↑q.scales.deref : List Std.I64).length := by
                rw [i3_post, i2_post]
                simpa only [lut4_deref_val] using hsc (↑sb0) hsbv
              rw [lut4_sval_read q.scales i3 i4 hlt i4_post, lut4_deref_val, i3_post, i2_post]
              unfold lut3_sval
              rfl
            have hinner : (↑inner : Int) = 2 * (lut3_isum 8 (fun j => lut3_pval tbl0 (↑gbase + j)
                (lut3_hi v (↑i) (↑tile) (↑gbase + j) (↑lane))))
              + (lut3_isum 8 (fun j => lut3_pval tbl0 (↑gbase + j)
                (lut3_lo v (↑i) (↑tile) (↑gbase + j) (↑lane))))
              - (lut3_isum 8 (fun j => lut3_xval tbl0 (↑gbase + j))) := by
              rw [inner_post, i9_post, i8_post, hsh, hsl, hsx]
            have h0 := lut4_i11_eq q v tbl0 (↑i) (↑tile) (↑lane) (↑b) (↑sb0) (↑gbase)
              i4 inner hbase hi4 hinner
            rw [htab] at h0
            rw [i11_post, lut4_fromI128I64_val, lut4_fromI128I64_val]
            exact h0
          refine ⟨?_, ?_, ?_, ?_⟩
          · rw [htab1]; exact htab
          · rw [sb1_post]; omega
          · rw [inner_block1_post, hib, sb1_post, lut3_isum_succ, hi11]
          · rw [sb1_post]; omega
      · rw [if_neg hc]
        simp only [WP.spec_ok]
        have h4 : sb0.val = 4 := by scalar_tac
        refine ⟨htab, ?_⟩
        rw [hib, h4]
        rfl
    · refine ⟨rfl, by norm_num, ?_⟩
      simp [lut3_isum_zero]
  obtain ⟨r, hr, hp⟩ := WP.spec_imp_exists hspec
  exact ⟨r, hr, hp⟩

/-! ## The block loop -/

/-- `word_tile_accs_sb_ok` in weakest-precondition form, for `step*`. -/
@[step]
lemma lut4_sb_step (hconv : lut3_fromU8I16_ok)
    (i : Std.Usize) (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (q : chart.QuantAct) (tile lane b : Std.Usize)
    (hb2 : ∀ s, s < 4 → 4 * b.val + s ≤ Usize.max)
    (hg : ∀ s, s < 4 → (4 * b.val + s) * 8 ≤ Usize.max)
    (hsc : ∀ s, s < 4 → 4 * b.val + s < q.scales.val.length)
    (hidx : ∀ s, s < 4 → (tile.val * i.val + ((4 * b.val + s) * 8) + 8) * 32 + lane.val ≤ Usize.max)
    (hv : ∀ s, s < 4 → ∀ j, j < 8 →
      lut3_cidx i.val tile.val ((4 * b.val + s) * 8 + j) lane.val < v.val.length)
    (hloR : ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b.val + s) * 8 + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b.val + s) * 8 + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ s, s < 4 → ∀ j, j < 8 → (4 * b.val + s) * 8 + j < tables.xs.val.length) :
    kern.lut.tile_accs_portable_loop0_loop0_loop0 i v tables q tile lane b 0#i128 0#usize
      ⦃ r => r.1 = tables ∧ r.2.val = lut4_blk q v tables i.val tile.val lane.val b.val ⦄ := by
  obtain ⟨r, hr, hp⟩ := word_tile_accs_sb_ok hconv i v tables q tile lane b
    hb2 hg hsc hidx hv hloR hhiR hhb hxR
  rw [hr]
  simp only [WP.spec_ok]
  exact hp

/-- The block loop of `tile_accs_portable` refines the row value `lut4_row`. -/
theorem word_tile_accs_b_ok (hconv : lut3_fromU8I16_ok)
    (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables) (q : chart.QuantAct)
    (tile lane : Std.Usize)
    (hi : tile.val * 32 ≤ Usize.max)
    (hi1 : tile.val * 32 + lane.val ≤ Usize.max)
    (hi2 : (tile.val * 32 + lane.val) * lut.n_blocks.val ≤ Usize.max)
    (hi3 : ∀ b', b' < lut.n_blocks.val →
      (tile.val * 32 + lane.val) * lut.n_blocks.val + b' ≤ Usize.max)
    (hsc3 : ∀ b', b' < lut.n_blocks.val →
      (tile.val * 32 + lane.val) * lut.n_blocks.val + b' < lut.scales.val.length)
    (hb2 : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → 4 * b' + s ≤ Usize.max)
    (hg : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → (4 * b' + s) * 8 ≤ Usize.max)
    (hsc : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → 4 * b' + s < q.scales.val.length)
    (hidx : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 →
      (tile.val * lut.n_groups.val + ((4 * b' + s) * 8) + 8) * 32 + lane.val ≤ Usize.max)
    (hv : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      lut3_cidx lut.n_groups.val tile.val ((4 * b' + s) * 8 + j) lane.val < lut.codes.val.length)
    (hloR : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b' + s) * 8 + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b' + s) * 8 + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      (4 * b' + s) * 8 + j < tables.xs.val.length)
    (hdblk : ∀ b', b' < lut.n_blocks.val →
      lut4_absb (lut3_dval lut tile.val lane.val b'
        * lut4_blk q lut.codes tables lut.n_groups.val tile.val lane.val b')
        85070591730234615865843651857942052864)
    (hpsum : ∀ k, k ≤ lut.n_blocks.val →
      lut4_absb (lut3_isum k (fun b => lut3_dval lut tile.val lane.val b
        * lut4_blk q lut.codes tables lut.n_groups.val tile.val lane.val b))
        170141183460469231731687303715884105727) :
    ∃ r, kern.lut.tile_accs_portable_loop0_loop0 lut tables q tile lane 0#i128 0#usize = ok r ∧
      r.#0 = lut.n_out ∧ r.#1 = lut.n_in ∧ r.#2 = lut.n_blocks ∧ r.#3 = lut.n_groups ∧
      r.#4 = lut.n_tiles ∧ r.#5 = lut.scales ∧ r.#6 = lut.d_max ∧ r.#7 = lut.codes ∧
      r.#8 = lut.raw ∧ r.#9 = tables ∧ (r.#10).val = lut4_row lut q lut.codes tables tile.val lane.val := by
  have hspec : kern.lut.tile_accs_portable_loop0_loop0 lut tables q tile lane 0#i128 0#usize
      ⦃ r => r.#0 = lut.n_out ∧ r.#1 = lut.n_in ∧ r.#2 = lut.n_blocks ∧ r.#3 = lut.n_groups ∧
        r.#4 = lut.n_tiles ∧ r.#5 = lut.scales ∧ r.#6 = lut.d_max ∧ r.#7 = lut.codes ∧
        r.#8 = lut.raw ∧ r.#9 = tables ∧ (r.#10).val = lut4_row lut q lut.codes tables tile.val lane.val ⦄ := by
    unfold kern.lut.tile_accs_portable_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun st : kern.lut.LutTables × Std.I128 × Std.Usize =>
        lut.n_blocks.val - st.2.2.val)
      (inv := fun st : kern.lut.LutTables × Std.I128 × Std.Usize =>
        st.1 = tables ∧ st.2.2.val ≤ lut.n_blocks.val ∧
        st.2.1.val = lut3_isum st.2.2.val (fun b => lut3_dval lut tile.val lane.val b
          * lut4_blk q lut.codes tables lut.n_groups.val tile.val lane.val b))
    · rintro ⟨tbl0, acc0, b0⟩ ⟨htab, hbnd, hacc⟩
      dsimp only at htab hbnd hacc ⊢
      unfold kern.lut.tile_accs_portable_loop0_loop0.body
      by_cases hb : b0 < lut.n_blocks
      · rw [if_pos hb]
        have hblt : b0.val < lut.n_blocks.val := by scalar_tac
        have hTR : (↑kern.lut.TILE_ROWS : Nat) = 32 := by
          simp only [kern.lut.TILE_ROWS]; rfl
        simp only [lift, bind_tc_ok]
        step*
        · have hi4 : (↑i4 : Int) = lut3_dval lut tile.val lane.val b0.val := by
            have hi3eq : (↑i3 : Nat) = (↑tile * 32 + ↑lane) * ↑lut.n_blocks + ↑b0 := by
              rw [i3_post, i2_post, i1_post, i_post, hTR]
            have hltv : (↑i3 : Nat) < (↑lut.scales : List Std.I64).length := by
              rw [hi3eq]; exact hsc3 (↑b0) hblt
            rw [lut4_sval_readv lut.scales i3 i4 hltv i4_post]
            unfold lut3_dval
            rw [hi3eq]
          have hblk : (↑inner_block : Int)
              = lut4_blk q lut.codes tables lut.n_groups.val tile.val lane.val b0.val := by
            rw [tables1_post1, htab]
          rw [lut4_fromI128I64_val, hi4, hblk]
          exact (lut4_i128_range (hdblk (↑b0) hblt) (by norm_num)).1
        · have hi4 : (↑i4 : Int) = lut3_dval lut tile.val lane.val b0.val := by
            have hi3eq : (↑i3 : Nat) = (↑tile * 32 + ↑lane) * ↑lut.n_blocks + ↑b0 := by
              rw [i3_post, i2_post, i1_post, i_post, hTR]
            have hltv : (↑i3 : Nat) < (↑lut.scales : List Std.I64).length := by
              rw [hi3eq]; exact hsc3 (↑b0) hblt
            rw [lut4_sval_readv lut.scales i3 i4 hltv i4_post]
            unfold lut3_dval
            rw [hi3eq]
          have hblk : (↑inner_block : Int)
              = lut4_blk q lut.codes tables lut.n_groups.val tile.val lane.val b0.val := by
            rw [tables1_post1, htab]
          rw [lut4_fromI128I64_val, hi4, hblk]
          exact (lut4_i128_range (hdblk (↑b0) hblt) (by norm_num)).2
        · have hsum := hpsum (b0.val + 1) (by omega)
          rw [lut3_isum_succ] at hsum
          unfold lut4_absb at hsum
          have hi4 : (↑i4 : Int) = lut3_dval lut tile.val lane.val b0.val := by
            have hi3eq : (↑i3 : Nat) = (↑tile * 32 + ↑lane) * ↑lut.n_blocks + ↑b0 := by
              rw [i3_post, i2_post, i1_post, i_post, hTR]
            have hltv : (↑i3 : Nat) < (↑lut.scales : List Std.I64).length := by
              rw [hi3eq]; exact hsc3 (↑b0) hblt
            rw [lut4_sval_readv lut.scales i3 i4 hltv i4_post]
            unfold lut3_dval
            rw [hi3eq]
          have hblk : (↑inner_block : Int)
              = lut4_blk q lut.codes tables lut.n_groups.val tile.val lane.val b0.val := by
            rw [tables1_post1, htab]
          rw [i5_post, lut4_fromI128I64_val, hi4, hblk, hacc]
          rw [I128.min_eq]
          omega
        · have hsum := hpsum (b0.val + 1) (by omega)
          rw [lut3_isum_succ] at hsum
          unfold lut4_absb at hsum
          have hi4 : (↑i4 : Int) = lut3_dval lut tile.val lane.val b0.val := by
            have hi3eq : (↑i3 : Nat) = (↑tile * 32 + ↑lane) * ↑lut.n_blocks + ↑b0 := by
              rw [i3_post, i2_post, i1_post, i_post, hTR]
            have hltv : (↑i3 : Nat) < (↑lut.scales : List Std.I64).length := by
              rw [hi3eq]; exact hsc3 (↑b0) hblt
            rw [lut4_sval_readv lut.scales i3 i4 hltv i4_post]
            unfold lut3_dval
            rw [hi3eq]
          have hblk : (↑inner_block : Int)
              = lut4_blk q lut.codes tables lut.n_groups.val tile.val lane.val b0.val := by
            rw [tables1_post1, htab]
          rw [i5_post, lut4_fromI128I64_val, hi4, hblk, hacc]
          rw [I128.max_eq]
          omega
        · refine ⟨?_, ?_, ?_, ?_⟩
          · rw [tables1_post, htab]
          · rw [b1_post]; omega
          · have hi4 : (↑i4 : Int) = lut3_dval lut tile.val lane.val b0.val := by
              have hi3eq : (↑i3 : Nat) = (↑tile * 32 + ↑lane) * ↑lut.n_blocks + ↑b0 := by
                rw [i3_post, i2_post, i1_post, i_post, hTR]
              have hltv : (↑i3 : Nat) < (↑lut.scales : List Std.I64).length := by
                rw [hi3eq]; exact hsc3 (↑b0) hblt
              rw [lut4_sval_readv lut.scales i3 i4 hltv i4_post]
              unfold lut3_dval
              rw [hi3eq]
            have hblk : (↑inner_block : Int)
                = lut4_blk q lut.codes tables lut.n_groups.val tile.val lane.val b0.val := by
              rw [tables1_post1, htab]
            have hi5 : (↑i5 : Int) = lut3_dval lut (↑tile) (↑lane) (↑b0)
                * lut4_blk q lut.codes tables (↑lut.n_groups) (↑tile) (↑lane) (↑b0) := by
              rw [i5_post, lut4_fromI128I64_val, hi4, hblk]
            rw [acc1_post, b1_post, lut3_isum_succ, hacc, hi5]
          · rw [b1_post]; omega
      · rw [if_neg hb]
        simp only [WP.spec_ok]
        have hb4 : b0.val = lut.n_blocks.val := by scalar_tac
        simp only [true_and]
        refine ⟨htab, ?_⟩
        rw [hacc, hb4]
        rfl
    · refine ⟨rfl, Nat.zero_le _, ?_⟩
      simp [lut3_isum_zero]
  obtain ⟨r, hr, hp⟩ := WP.spec_imp_exists hspec
  exact ⟨r, hr, hp⟩

/-! ## The lane loop -/

/-- `getD` after a `set` at a different index is unchanged. -/
lemma lut4_getD_set_ne {α : Type} (l : List α) (i j : Nat) (d a : α) (h : i ≠ j) :
    (l.set i a).getD j d = l.getD j d := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_set_ne h]

/-- `getD` after a `set` at the same index below the length is the written value. -/
lemma lut4_getD_set_self {α : Type} (l : List α) (i : Nat) (d a : α) (h : i < l.length) :
    (l.set i a).getD i d = a := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_set_self h]
  rfl

/-- The lane-loop invariant: lanes below `lane` hold their row value, lanes above keep the input. -/
def lut4_laneInv (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables) (q : chart.QuantAct)
    (tile : Nat) (accs cur : Array Std.I128 32#usize) (lane : Nat) : Prop :=
  (∀ k, k < lane → (cur.val.getD k default).val = lut4_row lut q lut.codes tables tile k) ∧
  (∀ k, lane ≤ k → k < 32 → cur.val.getD k default = accs.val.getD k default)

/-- One lane step: writing lane `lane`'s row value keeps the lane invariant. -/
lemma lut4_laneInv_set (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables)
    (q : chart.QuantAct) (tile lane : Nat) (i : Std.Usize)
    (accs cur : Array Std.I128 32#usize) (v : Std.I128)
    (hiv : i.val = lane) (hn : lane < 32)
    (h : lut4_laneInv lut tables q tile accs cur lane)
    (hv : v.val = lut4_row lut q lut.codes tables tile lane) :
    lut4_laneInv lut tables q tile accs (cur.set i v) (lane + 1) := by
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

/-- `word_tile_accs_b_ok` in weakest-precondition form, for `step*`. -/
@[step]
lemma lut4_b_step (hconv : lut3_fromU8I16_ok)
    (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables) (q : chart.QuantAct)
    (tile lane : Std.Usize)
    (hi : tile.val * 32 ≤ Usize.max)
    (hi1 : tile.val * 32 + lane.val ≤ Usize.max)
    (hi2 : (tile.val * 32 + lane.val) * lut.n_blocks.val ≤ Usize.max)
    (hi3 : ∀ b', b' < lut.n_blocks.val →
      (tile.val * 32 + lane.val) * lut.n_blocks.val + b' ≤ Usize.max)
    (hsc3 : ∀ b', b' < lut.n_blocks.val →
      (tile.val * 32 + lane.val) * lut.n_blocks.val + b' < lut.scales.val.length)
    (hb2 : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → 4 * b' + s ≤ Usize.max)
    (hg : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → (4 * b' + s) * 8 ≤ Usize.max)
    (hsc : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → 4 * b' + s < q.scales.val.length)
    (hidx : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 →
      (tile.val * lut.n_groups.val + ((4 * b' + s) * 8) + 8) * 32 + lane.val ≤ Usize.max)
    (hv : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      lut3_cidx lut.n_groups.val tile.val ((4 * b' + s) * 8 + j) lane.val < lut.codes.val.length)
    (hloR : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b' + s) * 8 + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b' + s) * 8 + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      (4 * b' + s) * 8 + j < tables.xs.val.length)
    (hdblk : ∀ b', b' < lut.n_blocks.val →
      lut4_absb (lut3_dval lut tile.val lane.val b'
        * lut4_blk q lut.codes tables lut.n_groups.val tile.val lane.val b')
        85070591730234615865843651857942052864)
    (hpsum : ∀ k, k ≤ lut.n_blocks.val →
      lut4_absb (lut3_isum k (fun b => lut3_dval lut tile.val lane.val b
        * lut4_blk q lut.codes tables lut.n_groups.val tile.val lane.val b))
        170141183460469231731687303715884105727) :
    kern.lut.tile_accs_portable_loop0_loop0 lut tables q tile lane 0#i128 0#usize
      ⦃ r => r.#0 = lut.n_out ∧ r.#1 = lut.n_in ∧ r.#2 = lut.n_blocks ∧ r.#3 = lut.n_groups ∧
        r.#4 = lut.n_tiles ∧ r.#5 = lut.scales ∧ r.#6 = lut.d_max ∧ r.#7 = lut.codes ∧
        r.#8 = lut.raw ∧ r.#9 = tables ∧ (r.#10).val = lut4_row lut q lut.codes tables tile.val lane.val ⦄ := by
  obtain ⟨r, hr, hp⟩ := word_tile_accs_b_ok hconv lut tables q tile lane
    hi hi1 hi2 hi3 hsc3 hb2 hg hsc hidx hv hloR hhiR hhb hxR hdblk hpsum
  rw [hr]
  simp only [WP.spec_ok]
  exact hp

/-- The lane loop of `tile_accs_portable` refines the row vector. -/
theorem word_tile_accs_lane_ok (hconv : lut3_fromU8I16_ok)
    (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables) (q : chart.QuantAct)
    (tile : Std.Usize) (accs : Array Std.I128 32#usize)
    (hi : tile.val * 32 ≤ Usize.max)
    (hi1 : ∀ lane, lane < 32 → tile.val * 32 + lane ≤ Usize.max)
    (hi2 : ∀ lane, lane < 32 → (tile.val * 32 + lane) * lut.n_blocks.val ≤ Usize.max)
    (hi3 : ∀ lane, lane < 32 → ∀ b', b' < lut.n_blocks.val →
      (tile.val * 32 + lane) * lut.n_blocks.val + b' ≤ Usize.max)
    (hsc3 : ∀ lane, lane < 32 → ∀ b', b' < lut.n_blocks.val →
      (tile.val * 32 + lane) * lut.n_blocks.val + b' < lut.scales.val.length)
    (hb2 : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → 4 * b' + s ≤ Usize.max)
    (hg : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → (4 * b' + s) * 8 ≤ Usize.max)
    (hsc : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → 4 * b' + s < q.scales.val.length)
    (hidx : ∀ lane, lane < 32 → ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 →
      (tile.val * lut.n_groups.val + ((4 * b' + s) * 8) + 8) * 32 + lane ≤ Usize.max)
    (hv : ∀ lane, lane < 32 → ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      lut3_cidx lut.n_groups.val tile.val ((4 * b' + s) * 8 + j) lane < lut.codes.val.length)
    (hloR : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b' + s) * 8 + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b' + s) * 8 + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      (4 * b' + s) * 8 + j < tables.xs.val.length)
    (hdblk : ∀ lane, lane < 32 → ∀ b', b' < lut.n_blocks.val →
      lut4_absb (lut3_dval lut tile.val lane b'
        * lut4_blk q lut.codes tables lut.n_groups.val tile.val lane b')
        85070591730234615865843651857942052864)
    (hpsum : ∀ lane, lane < 32 → ∀ k, k ≤ lut.n_blocks.val →
      lut4_absb (lut3_isum k (fun b => lut3_dval lut tile.val lane b
        * lut4_blk q lut.codes tables lut.n_groups.val tile.val lane b))
        170141183460469231731687303715884105727) :
    ∃ r, kern.lut.tile_accs_portable_loop0 lut tables q tile accs 0#usize = ok r ∧
      ∀ k, k < 32 → (r.val.getD k default).val = lut4_row lut q lut.codes tables tile.val k := by
  have hspec : kern.lut.tile_accs_portable_loop0 lut tables q tile accs 0#usize
      ⦃ r => ∀ k, k < 32 → (r.val.getD k default).val = lut4_row lut q lut.codes tables tile.val k ⦄ := by
    unfold kern.lut.tile_accs_portable_loop0
    apply loop.spec_decr_nat
      (measure := fun st : kern.lut.Pq2Lut × kern.lut.LutTables × Array Std.I128 32#usize × Std.Usize =>
        32 - st.2.2.2.val)
      (inv := fun st => st.1 = lut ∧ st.2.1 = tables ∧
        lut4_laneInv lut tables q tile.val accs st.2.2.1 st.2.2.2.val)
    · rintro ⟨lut0, tbl0, accs0, lane0⟩ ⟨h1, h2, h3⟩
      dsimp only at h1 h2 h3 ⊢
      unfold kern.lut.tile_accs_portable_loop0.body
      by_cases hl : lane0 < kern.lut.TILE_ROWS
      · rw [if_pos hl]
        have hl32 : lane0.val < 32 := by
          have h := hl
          simp only [kern.lut.TILE_ROWS] at h
          scalar_tac
        step*
        · intro k hk
          rw [h1] at hk
          rw [h1, h2]
          exact hpsum lane0.val hl32 k hk
        · refine ⟨?_, ?_, ?_, ?_⟩
          · rw [i_post, i1_post, i2_post, i3_post, i4_post, v_post, i5_post, v1_post, v2_post, h1]
          · rw [tables1_post, h2]
          · rw [lane1_post, a_post]
            exact lut4_laneInv_set lut tables q (↑tile) (↑lane0) lane0 accs accs0 acc rfl hl32 h3
              (by rw [i_post1, h1, h2])
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
      simp [lut4_laneInv]
  obtain ⟨r, hr, hp⟩ := WP.spec_imp_exists hspec
  exact ⟨r, hr, hp⟩

/-! ## The top level -/

/-- `tile_accs_portable` computes the row vector `lut4_row` (the lane loop started at lane `0`). -/
theorem word_tile_accs_portable_ok (hconv : lut3_fromU8I16_ok)
    (lut : kern.lut.Pq2Lut) (tables : kern.lut.LutTables) (q : chart.QuantAct)
    (tile : Std.Usize) (accs : Array Std.I128 32#usize)
    (hi : tile.val * 32 ≤ Usize.max)
    (hi1 : ∀ lane, lane < 32 → tile.val * 32 + lane ≤ Usize.max)
    (hi2 : ∀ lane, lane < 32 → (tile.val * 32 + lane) * lut.n_blocks.val ≤ Usize.max)
    (hi3 : ∀ lane, lane < 32 → ∀ b', b' < lut.n_blocks.val →
      (tile.val * 32 + lane) * lut.n_blocks.val + b' ≤ Usize.max)
    (hsc3 : ∀ lane, lane < 32 → ∀ b', b' < lut.n_blocks.val →
      (tile.val * 32 + lane) * lut.n_blocks.val + b' < lut.scales.val.length)
    (hb2 : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → 4 * b' + s ≤ Usize.max)
    (hg : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → (4 * b' + s) * 8 ≤ Usize.max)
    (hsc : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → 4 * b' + s < q.scales.val.length)
    (hidx : ∀ lane, lane < 32 → ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 →
      (tile.val * lut.n_groups.val + ((4 * b' + s) * 8) + 8) * 32 + lane ≤ Usize.max)
    (hv : ∀ lane, lane < 32 → ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      lut3_cidx lut.n_groups.val tile.val ((4 * b' + s) * 8 + j) lane < lut.codes.val.length)
    (hloR : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b' + s) * 8 + j) + 15 < tables.lo.val.length)
    (hhiR : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      16 * ((4 * b' + s) * 8 + j) + 15 < tables.hi.val.length)
    (hhb : ∀ m, m < tables.hi.val.length → (tables.hi.val.getD m 0#u8).val ≤ 127)
    (hxR : ∀ b', b' < lut.n_blocks.val → ∀ s, s < 4 → ∀ j, j < 8 →
      (4 * b' + s) * 8 + j < tables.xs.val.length)
    (hdblk : ∀ lane, lane < 32 → ∀ b', b' < lut.n_blocks.val →
      lut4_absb (lut3_dval lut tile.val lane b'
        * lut4_blk q lut.codes tables lut.n_groups.val tile.val lane b')
        85070591730234615865843651857942052864)
    (hpsum : ∀ lane, lane < 32 → ∀ k, k ≤ lut.n_blocks.val →
      lut4_absb (lut3_isum k (fun b => lut3_dval lut tile.val lane b
        * lut4_blk q lut.codes tables lut.n_groups.val tile.val lane b))
        170141183460469231731687303715884105727) :
    ∃ r, kern.lut.tile_accs_portable lut tables q tile accs = ok r ∧
      ∀ k, k < 32 → (r.val.getD k default).val = lut4_row lut q lut.codes tables tile.val k := by
  simpa only [kern.lut.tile_accs_portable] using
    word_tile_accs_lane_ok hconv lut tables q tile accs hi hi1 hi2 hi3 hsc3 hb2 hg hsc hidx hv
      hloR hhiR hhb hxR hdblk hpsum

/-! ## Non-vacuity

A witness with `n_blocks = 0`: the block loop runs zero iterations, so every lane's row value is
`lut4_row … = 0` and the top level writes the zero vector over its input. Every per-block hypothesis
is vacuous and every length hypothesis is satisfied by the zero-entry tables of `lut3_witTables`. -/

/-- Witness `Pq2Lut`: zero extents, empty planes. -/
def lut4_witLut : kern.lut.Pq2Lut :=
  { n_out := 0#usize, n_in := 0#usize, n_blocks := 0#usize, n_groups := 0#usize,
    n_tiles := 0#usize, scales := alloc.vec.Vec.from [] (by scalar_tac), d_max := 0#i128,
    codes := alloc.vec.Vec.from [] (by scalar_tac), raw := alloc.vec.Vec.from [] (by scalar_tac) }

/-- Witness `QuantAct`: empty planes. -/
def lut4_witQuant : chart.QuantAct :=
  { scales := alloc.vec.Vec.from [] (by scalar_tac), codes := alloc.vec.Vec.from [] (by scalar_tac) }

/-- With `n_blocks = 0` the reference row value is `0`. -/
lemma lut4_witLut_row (k : Nat) :
    lut4_row lut4_witLut lut4_witQuant lut4_witLut.codes lut3_witTables (↑(0#usize)) k = 0 := by
  unfold lut4_row
  simp [lut4_witLut]

/-- `lut3_witTables`'s high plane satisfies the `≤ 127` hypothesis. -/
lemma lut4_witLut_hile : lut4_hile lut3_witTables := by
  intro m
  rw [show lut3_witTables.hi.val.getD m 0#u8 = 0#u8 from by
    rw [show lut3_witTables.hi.val = List.replicate 16 0#u8 from by simp [lut3_witTables]]
    exact lut3_zero_getD m]
  simp

/-- The top level runs on the zero-block witness and writes the zero row vector: every hypothesis
of `word_tile_accs_portable_ok` holds jointly. -/
example (hconv : lut3_fromU8I16_ok) :
    ∃ r, kern.lut.tile_accs_portable lut4_witLut lut3_witTables lut4_witQuant 0#usize
        (Array.repeat 32#usize 0#i128) = ok r ∧
      ∀ k, k < 32 → (r.val.getD k default).val = 0 := by
  obtain ⟨r, hr, hp⟩ := word_tile_accs_portable_ok hconv lut4_witLut lut3_witTables lut4_witQuant
    0#usize (Array.repeat 32#usize 0#i128)
    (by scalar_tac)
    (by intro lane hl; scalar_tac)
    (by intro lane hl; simp [lut4_witLut])
    (by intro lane hl b' hb; simp [lut4_witLut] at hb)
    (by intro lane hl b' hb; simp [lut4_witLut] at hb)
    (by intro b' hb; simp [lut4_witLut] at hb)
    (by intro b' hb; simp [lut4_witLut] at hb)
    (by intro b' hb; simp [lut4_witLut] at hb)
    (by intro lane hl b' hb; simp [lut4_witLut] at hb)
    (by intro lane hl b' hb; simp [lut4_witLut] at hb)
    (by intro b' hb; simp [lut4_witLut] at hb)
    (by intro b' hb; simp [lut4_witLut] at hb)
    (by intro m hm; exact lut4_witLut_hile m)
    (by intro b' hb; simp [lut4_witLut] at hb)
    (by intro lane hl b' hb; simp [lut4_witLut] at hb)
    (by intro lane hl k hk
        have hk0 : k = 0 := by
          have h : (lut4_witLut.n_blocks.val : Nat) = 0 := by simp [lut4_witLut]
          omega
        subst hk0
        simp only [lut3_isum_zero, lut4_absb]
        norm_num)
  exact ⟨r, hr, fun k hk => (hp k hk).trans (lut4_witLut_row k)⟩

end apeiron_word
