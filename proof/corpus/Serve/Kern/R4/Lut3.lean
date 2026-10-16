/-
Integer refinement of the innermost group-accumulation loop of the `PQ2_0` lookup-table kernels
(`src/engine/kern/lut.rs`), group `kern.lut`, continuing `Serve/Lut.lean` and `Serve/Lut2.lean`
(which prove `split_nibbles`, `build_tables_range`, and the subset-sum table).

`tile_accs_portable` walks the (lane, block, sub-block, group) nest, looks the two nibble planes of
each group's reconstructed table entry up through `LutTables::p` and accumulates them into an exact
`i128` row value. The reference below is the closed-form partial sum `lut3_*` over the same index
space, stated as a finite sum, not as the Rust loop.

Proved here:

* `lut3_act_qk`, `lut3_subs_per_block`, `lut3_groups_per_sub` — the closed values `32`, `ok 4`,
  `ok 8` of the three `@[irreducible]` extent constants;
* `lut3_p_ok` — `LutTables::p` returns the closed-form `lut3_pval` on its stated domain;
* `word_tile_accs_inner` — the innermost loop `tile_accs_portable_loop0_loop0_loop0_loop0`
  (groups `0..GROUPS_PER_SUB`) refines the three reference partial sums `lut3_isum` at its caller's
  base;
* the non-vacuity witnesses (`lut3_witTables`, `lut3_witCodes`, three `example`s).

Open in this file, not yet proved:

* the outer loops of `tile_accs_portable` (`_loop0_loop0_loop0` sb-loop, `_loop0_loop0` b-loop,
  `_loop0` lane-loop, top level). `kern.lut.SUBS_PER_BLOCK` / `GROUPS_PER_SUB` are `@[irreducible]`,
  so `rfl` alone does not reduce them; `lut3_subs_per_block` / `lut3_groups_per_sub` below supply
  their closed values and are a prerequisite, not the remaining work. The sb-loop then needs the
  `I64 → I128` widening value lemma for `FromI128I64` and the `2·sum_hi + sum_lo − sum_x` no-overflow
  bound, and the b-lane loop additionally needs a stated scale-value hypothesis;
* `block_inners_portable` (its inner loop differs from `word_tile_accs_inner` only in reading
  `lut.codes`/`lut.n_groups` and returning `(lut, tables, sums)`, so it is a mechanical adaptation,
  not attempted here);
* `Pq2Lut::from_pq2_0` (success and the three refusal branches).

Boundaries stated, not papered over:

* `tables.p` reconstructs an entry through `i16::from(u8)` (the infallible `From<u8> for i16`). The
  Aeneas model defines `From<u8>` for `i64`/`i32`/`i128` but **not** for `i16`, so the extraction
  emits `I16.Insts.CoreConvertFromU8.from` as an `axiom` (`proof/ApeironWord.lean:217`). The value of
  every table lookup therefore cannot be derived from the extraction alone: each exact-value theorem
  below carries `lut3_fromU8I16_ok`, the statement that this conversion is the zero-extending cast
  the Rust semantics gives it. That hypothesis is a modelling gap in the extraction; it is stated,
  not assumed silently, and it is what the regeneration must close.
* the byte planes are read as `getD` under the caller's length hypotheses, and the high plane must
  stay `≤ 127` so that `hi * 256` and `lo + hi * 256` fit `i16` (the Rust encoder writes
  `P + 508 ≤ 1016`, so this holds on any table `build_tables_range` produced);
* `block_inners_avx2` / `tile_accs_avx2` are extracted `axiom`s (no body to refine);
* the `F32`/`F64` carriers and the `unsafe`/AVX2 paths are out of scope.

Grade: `proved` for `lut3_p_ok` and `word_tile_accs_inner` under their stated hypotheses; every
other refinement named above is open, not `proved`. The `#print axioms` of those two lists the
closed cone `{propext, Classical.choice, Quot.sound}` plus the disclosed extraction axiom
`apeiron_word.I16.Insts.CoreConvertFromU8.from`; the three extent-constant lemmas are axiom-free.
-/
import Serve.Kern.Lut2

set_option autoImplicit false
set_option maxHeartbeats 200000

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## Extraction constants

`kern.lut.ACT_QK`, `SUBS_PER_BLOCK` and `GROUPS_PER_SUB` are `@[irreducible]` in the extraction, so
they do not unfold under the default transparency and `rfl` alone cannot reduce them. `simp only`
naming the declaration unfolds an irreducible constant regardless of the attribute, and the
`UScalar` division then reduces by `rfl`, so each constant has a closed value. These three are what
a caller stepping the outer loops needs before `step*` can resolve the `let` bindings that read
them. -/

lemma lut3_act_qk : kern.lut.ACT_QK = 32#usize := by
  simp only [kern.lut.ACT_QK]

lemma lut3_subs_per_block : kern.lut.SUBS_PER_BLOCK = ok 4#usize := by
  simp only [kern.lut.SUBS_PER_BLOCK, kern.lut.ACT_QK, quant.PQ2_0_QK]
  rfl

lemma lut3_groups_per_sub : kern.lut.GROUPS_PER_SUB = ok 8#usize := by
  simp only [kern.lut.GROUPS_PER_SUB, kern.lut.ACT_QK]
  rfl

/-! ## The reference

A finite sum `Σ_{k<n} f k` over the index space, and the two lookup planes. `lut3_pval` is the value
the Rust reconstruction `lo + 256·hi − 508` names; `lut3_hi`/`lut3_lo` are the two nibble indices
`LutTables::p` is called with. -/

/-- `Σ_{k<n} f k`. -/
def lut3_isum (n : Nat) (f : Nat → Int) : Int :=
  match n with
  | 0 => 0
  | k + 1 => lut3_isum k f + f k

@[simp] lemma lut3_isum_zero (f : Nat → Int) : lut3_isum 0 f = 0 := rfl

@[simp] lemma lut3_isum_succ (k : Nat) (f : Nat → Int) :
    lut3_isum (k + 1) f = lut3_isum k f + f k := rfl

/-- The `(tile, lane)` byte index of group `g` in the code plane: the Rust `(tile*n_groups + g)*32
+ lane`. -/
def lut3_cidx (n_groups tile g lane : Nat) : Nat := (tile * n_groups + g) * 32 + lane

/-- The high-plane nibble of the packed code byte of `(tile, g, lane)`, the first index
`LutTables::p` is called with. -/
def lut3_hi (v : alloc.vec.Vec Std.U8) (n_groups tile g lane : Nat) : Nat :=
  (v.val.getD (lut3_cidx n_groups tile g lane) 0#u8).val >>> 4

/-- The low-plane nibble of that byte, the second index. -/
def lut3_lo (v : alloc.vec.Vec Std.U8) (n_groups tile g lane : Nat) : Nat :=
  (v.val.getD (lut3_cidx n_groups tile g lane) 0#u8).val % 16

/-- The reconstructed table entry `lo(g·16+n) + 256·hi(g·16+n) − 508`: the value of
`LutTables::p tables g n`. -/
def lut3_pval (tables : kern.lut.LutTables) (g n : Nat) : Int :=
  ((tables.lo.val.getD (16 * g + n) 0#u8).val : Int)
    + ((tables.hi.val.getD (16 * g + n) 0#u8).val : Int) * 256 - 508

/-- The `xs` plane entry `Σ_{k<4} xq_{4g+k}` of group `g`, read as an `Int`. -/
def lut3_xval (tables : kern.lut.LutTables) (g : Nat) : Int :=
  (tables.xs.val.getD g 0#i16).val

/-- The activation scale of sub-block `sb` of block `b`, read as an `Int`. -/
def lut3_sval (q : chart.QuantAct) (b sb : Nat) : Int :=
  (q.scales.val.getD (4 * b + sb) 0#i64).val

/-- The weight scale of `(tile, lane)` in block `b`, read as an `Int`. -/
def lut3_dval (lut : kern.lut.Pq2Lut) (tile lane b : Nat) : Int :=
  (lut.scales.val.getD ((tile * 32 + lane) * lut.n_blocks.val + b) 0#i64).val

/-! ## The `i16::from(u8)` model gap

The Aeneas model has no definition for `From<u8> for i16`, so the extraction leaves the conversion
an `axiom`. `lut3_fromU8I16_ok` is the statement the missing model entry must have: the conversion
is the zero-extending cast, so the `I16`'s value is the byte's. -/

/-- The `i16::from(u8)` conversion returns a value equal to its argument's value. -/
def lut3_fromU8I16_ok : Prop :=
  ∀ u : Std.U8, ∃ v : Std.I16,
    I16.Insts.CoreConvertFromU8.from u = ok v ∧ v.val = (u.val : Int)

lemma lut3_fromU8I16_val (hconv : lut3_fromU8I16_ok) (u : Std.U8) : ∃ v : Std.I16,
    I16.Insts.CoreConvertFromU8.from u = ok v ∧ v.val = (u.val : Int) := hconv u

@[step]
lemma lut3_fromU8I16_spec (hconv : lut3_fromU8I16_ok) (u : Std.U8) :
    I16.Insts.CoreConvertFromU8.from u ⦃ r => r.val = (u.val : Int) ⦄ := by
  obtain ⟨v, hv, hvv⟩ := hconv u
  rw [hv]
  simp only [WP.spec_ok]
  exact hvv

/-! ## `LutTables::p`

The reconstruction, in closed form. The bounds are the caller's: the two plances hold `16·g + n`,
the multiplication `g · 16` does not wrap `usize`, and the high byte stays `≤ 127` so that
`hi · 256` and `lo + hi · 256` fit `i16`. -/

/-- Reindexing: a list's `getElem` at `i` is its `getD` at any equal `j`. The index proof never
enters the value, so this replaces a dependent-index rewrite. -/
lemma lut3_getElem_eq_getD {α : Type} [Inhabited α] (l : List α) (i j : Nat) (h : i = j)
    (hj : j < l.length) : l[i] = l.getD j default := by
  subst h
  rw [getD_eq_getElem l i default hj]

/-- The reconstruction `LutTables::p` is the closed-form value `lut3_pval`. The bounds are the
caller's: the two planes hold `16·g + n`, the multiplication `g · 16` does not wrap `usize`, and the
high byte stays `≤ 127` so that `hi · 256` and `lo + hi · 256` fit `i16`. -/
@[step]
theorem lut3_p_ok (hconv : lut3_fromU8I16_ok) (tables : kern.lut.LutTables)
    (g n : Std.Usize)
    (hlo : 16 * g.val + n.val < tables.lo.length)
    (hhi : 16 * g.val + n.val < tables.hi.length)
    (hhin : (tables.hi.val.getD (16 * g.val + n.val) 0#u8).val ≤ 127) :
    kern.lut.LutTables.p tables g n ⦃ r => r.val = lut3_pval tables g.val n.val ⦄ := by
  unfold kern.lut.LutTables.p
  step*
  · rw [i6_post]
    have h5l : i5 = tables.hi.val.getD (16 * g.val + n.val) 0#u8 := by
      rw [i5_post]
      exact lut3_getElem_eq_getD tables.hi.val i4.val (16 * g.val + n.val) (by omega) hhi
    rw [h5l]
    have hb := hhin
    scalar_tac
  · rw [r_post, i8_post, i7_post, i3_post, i6_post]
    unfold lut3_pval
    have h2l : i2 = tables.lo.val.getD (16 * g.val + n.val) 0#u8 := by
      rw [i2_post]
      exact lut3_getElem_eq_getD tables.lo.val i1.val (16 * g.val + n.val) (by omega) hlo
    have h5l : i5 = tables.hi.val.getD (16 * g.val + n.val) 0#u8 := by
      rw [i5_post]
      exact lut3_getElem_eq_getD tables.hi.val i4.val (16 * g.val + n.val) (by omega) hhi
    rw [h2l, h5l]

/-! ## Bounds on the table values -/

/-- `LutTables::p`'s value is at least `−508`: `lo ≥ 0`, `hi ≥ 0`, `lo + 256·hi − 508`. -/
lemma lut3_pval_ge (tables : kern.lut.LutTables) (g n : Nat) : -508 ≤ lut3_pval tables g n := by
  unfold lut3_pval
  omega

/-- `LutTables::p`'s value is at most `32259`: `lo ≤ 255`, `hi ≤ 127`, `lo + 256·hi − 508`. -/
lemma lut3_pval_le (tables : kern.lut.LutTables) (g n : Nat)
    (hhin : (tables.hi.val.getD (16 * g + n) 0#u8).val ≤ 127) :
    lut3_pval tables g n ≤ 32259 := by
  unfold lut3_pval
  have h := (tables.lo.val.getD (16 * g + n) 0#u8).hBounds
  simp only [UScalarTy.numBits] at h
  omega

/-- The low nibble `x &&& 15` is `x % 16`. -/
lemma lut3_u8_lo (x : Std.U8) : (x &&& 15#u8).val = x.val % 16 := by
  rw [UScalar.val_and, show (15#u8).val = 15 from rfl]
  have := Nat.and_two_pow_sub_one_eq_mod x.val 4
  simpa using this

/-- The body's high-nibble value is the reference `lut3_hi`. -/
lemma lut3_hi_of (v : alloc.vec.Vec Std.U8) (n_groups tile g lane : Nat)
    (h : lut3_cidx n_groups tile g lane < v.val.length) :
    (v.val[lut3_cidx n_groups tile g lane]).val >>> 4 = lut3_hi v n_groups tile g lane := by
  unfold lut3_hi
  rw [getD_eq_getElem v.val (lut3_cidx n_groups tile g lane) 0#u8 h]

/-- The packer's `FromUsizeU8` is the identity on values. -/
lemma lut3_fromU8_conv (x : Std.U8) :
    (core.convert.num.FromUsizeU8.from x).val = x.val :=
  core.convert.num.FromUsizeU8.from_val_eq x

/-- The packer's `FromI64I16` is the identity on values. -/
lemma lut3_i16ToI64_conv (x : Std.I16) :
    (core.convert.num.FromI64I16.from x).val = x.val :=
  core.convert.num.FromI64I16.from_val_eq x

/-- The group sum `xs[g]` lies in `i16`'s range. -/
lemma lut3_xval_bounds (tables : kern.lut.LutTables) (g : Nat) :
    -32768 ≤ lut3_xval tables g ∧ lut3_xval tables g ≤ 32767 := by
  show -32768 ≤ (tables.xs.val.getD g 0#i16).val ∧ (tables.xs.val.getD g 0#i16).val ≤ 32767
  have h := (tables.xs.val.getD g 0#i16).hBounds
  rw [IScalarTy.I16_numBits_eq] at h
  omega

/-- The high nibble `x >>> 4` of a byte is at most `15`. -/
lemma lut3_hi_nib (x : Std.U8) : (↑x : Nat) >>> 4 ≤ 15 := by
  rw [Nat.shiftRight_eq_div_pow, show (2 : Nat) ^ 4 = 16 from by norm_num]
  have h : (↑x : Nat) < 256 := by
    have hb := x.hBounds
    simp only [UScalarTy.numBits] at hb
    omega
  omega

/-- The low nibble `x &&& 15` of a byte is at most `15`. -/
lemma lut3_lo_nib (x : Std.U8) : (↑(x &&& 15#u8) : Nat) ≤ 15 := by
  have h : (↑(x &&& 15#u8) : Nat) = (↑x : Nat) % 16 := lut3_u8_lo x
  rw [h]
  have := Nat.mod_lt (↑x : Nat) (by norm_num : 0 < 16)
  omega

/-- The packer's `FromUsizeU8` is the identity on values, as the goals read it. -/
lemma lut3_fromU8 (x : Std.U8) :
    (↑(core.convert.num.FromUsizeU8.from x) : Nat) = ↑x :=
  core.convert.num.FromUsizeU8.from_val_eq x

/-- The packer's `FromI64I16` is the identity on values, as the goals read it. -/
lemma lut3_i16ToI64 (x : Std.I16) :
    (↑(core.convert.num.FromI64I16.from x) : Int) = ↑x :=
  core.convert.num.FromI64I16.from_val_eq x

/-- A vector element read by index is the reference `lut3_xval` at the same index. -/
lemma lut3_xval_get (tables : kern.lut.LutTables) (i : Nat)
    (h : i < tables.xs.val.length) :
    (↑(tables.xs.val[i]'h : Std.I16) : Int) = lut3_xval tables i := by
  have h1 : (tables.xs.val[i]'h : Std.I16) = tables.xs.val.getD i 0#i16 :=
    (getD_eq_getElem tables.xs.val i 0#i16 h).symm
  rw [h1]
  rfl

/-- The body's low-nibble value is the reference `lut3_lo`. -/
lemma lut3_lo_of (v : alloc.vec.Vec Std.U8) (n_groups tile g lane : Nat)
    (h : lut3_cidx n_groups tile g lane < v.val.length) :
    (v.val[lut3_cidx n_groups tile g lane]).val % 16 = lut3_lo v n_groups tile g lane := by
  unfold lut3_lo
  rw [getD_eq_getElem v.val (lut3_cidx n_groups tile g lane) 0#u8 h]

/-! ## The inner group loop

`tile_accs_portable_loop0_loop0_loop0_loop0` and `block_inners_portable_loop0_loop0_loop0` walk the
groups `[gbase, gbase + bound)` of one `(tile, lane)`, accumulating the two `LutTables::p` nibble
lookups (high plane, then low plane) and the group sum `xs[g]` into three exact accumulators. The
reference is the three partial sums `lut3_isum k`. -/

/-- An accumulator with headroom for one more table value: every `|value| ≤ 32768`. -/
def lut3_rng (x : Int) : Prop := I64.min + 33000 ≤ x ∧ x ≤ I64.max - 33000

/-- The inner-accumulation invariant after `k` groups: the three accumulator values are the three
reference partial sums over `gbase .. gbase + k`. -/
def lut3_innerInv (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (n_groups tile lane gbase bound : Nat) (sh sl sx : Std.I64) (k : Nat) : Prop :=
  k ≤ bound ∧
  sh.val = lut3_isum k (fun j => lut3_pval tables (gbase + j)
    (lut3_hi v n_groups tile (gbase + j) lane)) ∧
  sl.val = lut3_isum k (fun j => lut3_pval tables (gbase + j)
    (lut3_lo v n_groups tile (gbase + j) lane)) ∧
  sx.val = lut3_isum k (fun j => lut3_xval tables (gbase + j))

/-- The postcondition of the inner loop: the three sums are the full reference sums. -/
def lut3_innerPost (v : alloc.vec.Vec Std.U8) (tables : kern.lut.LutTables)
    (n_groups tile lane gbase bound : Nat)
    (r : kern.lut.LutTables × Std.I64 × Std.I64 × Std.I64) : Prop :=
  r.1 = tables ∧
  r.2.1.val = lut3_isum bound (fun j => lut3_pval tables (gbase + j)
    (lut3_hi v n_groups tile (gbase + j) lane)) ∧
  r.2.2.1.val = lut3_isum bound (fun j => lut3_pval tables (gbase + j)
    (lut3_lo v n_groups tile (gbase + j) lane)) ∧
  r.2.2.2.val = lut3_isum bound (fun j => lut3_xval tables (gbase + j))

/-- The inner accumulation loop of `tile_accs_portable` refines the three reference partial sums
when the code byte and the table entries are in range, the high plane stays `≤ 127`, and every
partial sum keeps `i64` headroom. -/
theorem word_tile_accs_inner (hconv : lut3_fromU8I16_ok)
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
    ∃ r, kern.lut.tile_accs_portable_loop0_loop0_loop0_loop0 i i1 v tables tile lane sh sl sx gbase
        g8 = ok r ∧ lut3_innerPost v tables i1.val tile.val lane.val gbase.val i.val r := by
  have hspec : kern.lut.tile_accs_portable_loop0_loop0_loop0_loop0 i i1 v tables tile lane sh sl sx
      gbase g8 ⦃ r => lut3_innerPost v tables i1.val tile.val lane.val gbase.val i.val r ⦄ := by
    unfold kern.lut.tile_accs_portable_loop0_loop0_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun st : Std.I64 × Std.I64 × Std.I64 × Std.Usize => i.val - st.2.2.2.val)
      (inv := fun st => lut3_innerInv v tables i1.val tile.val lane.val gbase.val i.val
        st.1 st.2.1 st.2.2.1 st.2.2.2.val)
      (post := fun r : kern.lut.LutTables × Std.I64 × Std.I64 × Std.I64 =>
        lut3_innerPost v tables i1.val tile.val lane.val gbase.val i.val r)
    · rintro ⟨sh1, sl1, sx1, g8p⟩ hinv1
      obtain ⟨hle, hsh, hsl, hsx⟩ := hinv1
      dsimp only at hle hsh hsl hsx
      dsimp only
      unfold kern.lut.tile_accs_portable_loop0_loop0_loop0_loop0.body
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
          have hidxhi : 16 * (gbase.val + g8p.val) + n < tables.hi.val.length := by
            have h := hhiR g8p.val hglt
            omega
          exact hhb _ hidxhi
        step*
        · -- the code byte is in range
          rw [idx_post, i4_post, i3_post, i2_post, g_post, hTR]
          exact hv g8p.val hglt
        · -- the high-plane index is in `lo`
          rw [g_post, hconv8, i6_post]
          exact hloAt _ (lut3_hi_nib i5)
        · -- 5: the high-plane index is in `hi`
          rw [g_post, hconv8, i6_post]
          exact hhiAt _ (lut3_hi_nib i5)
        · -- 6: the high plane stays `≤ 127`
          rw [g_post, hconv8, i6_post]
          exact hhinAt _ (lut3_hi_nib i5)
        · -- 7: `sh1 + p` does not underflow `i64`
          rw [hconv16, i8_post, g_post, hconv8, i6_post, hsh]
          have h1 := (hrng g8p.val hle).1
          simp only [lut3_rng] at h1
          obtain ⟨h1a, h1b⟩ := h1
          have h2 := lut3_pval_ge tables (↑gbase + ↑g8p) (↑i5 >>> 4)
          rw [I64.min_eq] at h1a ⊢
          omega
        · -- 8: `sh1 + p` does not overflow `i64`
          rw [hconv16, i8_post, g_post, hconv8, i6_post, hsh]
          have h1 := (hrng g8p.val hle).1
          simp only [lut3_rng] at h1
          obtain ⟨h1a, h1b⟩ := h1
          have h2 := lut3_pval_le tables (↑gbase + ↑g8p) (↑i5 >>> 4)
            (hhinAt _ (lut3_hi_nib i5))
          rw [I64.max_eq] at h1b ⊢
          omega
        · -- 9: the low-plane index is in `lo`
          rw [g_post, hconv8]
          exact hloAt _ (lut3_lo_nib i5)
        · -- 10: the low-plane index is in `hi`
          rw [g_post, hconv8]
          exact hhiAt _ (lut3_lo_nib i5)
        · -- 11: the low plane stays `≤ 127`
          rw [g_post, hconv8]
          exact hhinAt _ (lut3_lo_nib i5)
        · -- 12: `sl1 + p` does not underflow `i64`
          rw [hconv16, i12_post, g_post, hconv8, hsl]
          have h1 := (hrng g8p.val hle).2.1
          simp only [lut3_rng] at h1
          obtain ⟨h1a, h1b⟩ := h1
          have h2 := lut3_pval_ge tables (↑gbase + ↑g8p) (↑(i5 &&& 15#u8))
          rw [I64.min_eq] at h1a ⊢
          omega
        · -- 13: `sl1 + p` does not overflow `i64`
          rw [hconv16, i12_post, g_post, hconv8, hsl]
          have h1 := (hrng g8p.val hle).2.1
          simp only [lut3_rng] at h1
          obtain ⟨h1a, h1b⟩ := h1
          have h2 := lut3_pval_le tables (↑gbase + ↑g8p) (↑(i5 &&& 15#u8))
            (hhinAt _ (lut3_lo_nib i5))
          rw [I64.max_eq] at h1b ⊢
          omega
        · -- 14: `sx1 + xs[g]` does not underflow `i64`
          have h14 : (↑i14 : Int) = lut3_xval tables (↑gbase + ↑g8p) := by
            have hglt2 : (↑g : Nat) < tables.xs.val.length := by
              rw [g_post]; exact hxR g8p.val hglt
            have h1 : (↑i14 : Int) = ↑(tables.xs.val.getD (↑g) 0#i16) := by
              rw [i14_post]
              exact congrArg (fun y : Std.I16 => (↑y : Int))
                (getD_eq_getElem tables.xs.val (↑g) 0#i16 hglt2).symm
            have h2 : (↑i14 : Int) = ↑(tables.xs.val.getD (↑gbase + ↑g8p) 0#i16) := by
              rw [h1, g_post]
            rw [h2]
            rfl
          rw [hconv16, hsx, h14]
          have h1 := (hrng g8p.val hle).2.2
          simp only [lut3_rng] at h1
          obtain ⟨h1a, h1b⟩ := h1
          have h3 := lut3_xval_bounds tables (↑gbase + ↑g8p)
          rw [I64.min_eq] at h1a ⊢
          omega
        · -- 15: `sx1 + xs[g]` does not overflow `i64`
          have h14 : (↑i14 : Int) = lut3_xval tables (↑gbase + ↑g8p) := by
            have hglt2 : (↑g : Nat) < tables.xs.val.length := by
              rw [g_post]; exact hxR g8p.val hglt
            have h1 : (↑i14 : Int) = ↑(tables.xs.val.getD (↑g) 0#i16) := by
              rw [i14_post]
              exact congrArg (fun y : Std.I16 => (↑y : Int))
                (getD_eq_getElem tables.xs.val (↑g) 0#i16 hglt2).symm
            have h2 : (↑i14 : Int) = ↑(tables.xs.val.getD (↑gbase + ↑g8p) 0#i16) := by
              rw [h1, g_post]
            rw [h2]
            rfl
          rw [hconv16, hsx, h14]
          have h1 := (hrng g8p.val hle).2.2
          simp only [lut3_rng] at h1
          obtain ⟨h1a, h1b⟩ := h1
          have h3 := lut3_xval_bounds tables (↑gbase + ↑g8p)
          rw [I64.max_eq] at h1b ⊢
          omega
        · -- 16: the invariant is preserved and the measure decreases
          refine ⟨?_, ?_⟩
          · unfold lut3_innerInv
            refine ⟨?_, ?_, ?_, ?_⟩
            · rw [g81_post]; omega
            · have hidxeq : (↑idx : Nat) = lut3_cidx (↑i1) (↑tile) (↑gbase + ↑g8p) (↑lane) := by
                rw [idx_post, i4_post, i3_post, i2_post, g_post, hTR]
                rfl
              have himeq : (↑(core.convert.num.FromUsizeU8.from i6) : Nat)
                  = lut3_hi v (↑i1) (↑tile) (↑gbase + ↑g8p) (↑lane) := by
                have hidxlt : (↑idx : Nat) < v.val.length := by
                  rw [hidxeq]; exact hv g8p.val hglt
                have h5d : (↑i5 : Nat) = ↑(v.val.getD (↑idx) 0#u8) := by
                  rw [i5_post]
                  exact congrArg (fun y : Std.U8 => (↑y : Nat))
                    (getD_eq_getElem v.val (↑idx) 0#u8 hidxlt).symm
                have h5e : (↑i5 : Nat)
                    = ↑(v.val.getD (lut3_cidx (↑i1) (↑tile) (↑gbase + ↑g8p) (↑lane)) 0#u8) := by
                  rw [h5d, hidxeq]
                rw [hconv8, i6_post, h5e]
                rfl
              have h1 : (↑sum_hi1 : Int)
                  = lut3_isum (↑g8p)
                      (fun j => lut3_pval tables (↑gbase + j)
                        (lut3_hi v (↑i1) (↑tile) (↑gbase + j) (↑lane)))
                    + lut3_pval tables (↑gbase + ↑g8p) (↑(core.convert.num.FromUsizeU8.from i6)) := by
                rw [sum_hi1_post, hconv16, i8_post, g_post, hsh]
              have h2 : lut3_pval tables (↑gbase + ↑g8p) (↑(core.convert.num.FromUsizeU8.from i6))
                  = (fun j => lut3_pval tables (↑gbase + j)
                      (lut3_hi v (↑i1) (↑tile) (↑gbase + j) (↑lane))) (↑g8p) := by
                rw [himeq]
              rw [h1, h2, g81_post, lut3_isum_succ]
            · have hidxeq : (↑idx : Nat) = lut3_cidx (↑i1) (↑tile) (↑gbase + ↑g8p) (↑lane) := by
                rw [idx_post, i4_post, i3_post, i2_post, g_post, hTR]
                rfl
              have hlomeq : (↑(core.convert.num.FromUsizeU8.from (i5 &&& 15#u8)) : Nat)
                  = lut3_lo v (↑i1) (↑tile) (↑gbase + ↑g8p) (↑lane) := by
                have hidxlt : (↑idx : Nat) < v.val.length := by
                  rw [hidxeq]; exact hv g8p.val hglt
                have h5d : (↑i5 : Nat) = ↑(v.val.getD (↑idx) 0#u8) := by
                  rw [i5_post]
                  exact congrArg (fun y : Std.U8 => (↑y : Nat))
                    (getD_eq_getElem v.val (↑idx) 0#u8 hidxlt).symm
                have h5e : (↑i5 : Nat)
                    = ↑(v.val.getD (lut3_cidx (↑i1) (↑tile) (↑gbase + ↑g8p) (↑lane)) 0#u8) := by
                  rw [h5d, hidxeq]
                rw [hconv8, show (↑(i5 &&& 15#u8) : Nat) = (↑i5 : Nat) % 16 from lut3_u8_lo i5, h5e]
                rfl
              have h1 : (↑sum_lo1 : Int)
                  = lut3_isum (↑g8p)
                      (fun j => lut3_pval tables (↑gbase + j)
                        (lut3_lo v (↑i1) (↑tile) (↑gbase + j) (↑lane)))
                    + lut3_pval tables (↑gbase + ↑g8p)
                        (↑(core.convert.num.FromUsizeU8.from (i5 &&& 15#u8))) := by
                rw [sum_lo1_post, hconv16, i12_post, g_post, hsl]
              have h2 : lut3_pval tables (↑gbase + ↑g8p)
                    (↑(core.convert.num.FromUsizeU8.from (i5 &&& 15#u8)))
                  = (fun j => lut3_pval tables (↑gbase + j)
                      (lut3_lo v (↑i1) (↑tile) (↑gbase + j) (↑lane))) (↑g8p) := by
                rw [hlomeq]
              rw [h1, h2, g81_post, lut3_isum_succ]
            · have hx14 : (↑(core.convert.num.FromI64I16.from i14) : Int)
                  = lut3_xval tables (↑gbase + ↑g8p) := by
                have hglt2 : (↑g : Nat) < tables.xs.val.length := by
                  rw [g_post]; exact hxR g8p.val hglt
                have h1 : (↑i14 : Int) = ↑(tables.xs.val.getD (↑g) 0#i16) := by
                  rw [i14_post]
                  exact congrArg (fun y : Std.I16 => (↑y : Int))
                    (getD_eq_getElem tables.xs.val (↑g) 0#i16 hglt2).symm
                have h2 : (↑i14 : Int) = ↑(tables.xs.val.getD (↑gbase + ↑g8p) 0#i16) := by
                  rw [h1, g_post]
                rw [hconv16, h2]
                rfl
              have h1 : (↑sum_x1 : Int)
                  = lut3_isum (↑g8p) (fun j => lut3_xval tables (↑gbase + j))
                    + (↑(core.convert.num.FromI64I16.from i14) : Int) := by
                rw [sum_x1_post, hsx]
              have h2 : (↑(core.convert.num.FromI64I16.from i14) : Int)
                  = (fun j => lut3_xval tables (↑gbase + j)) (↑g8p) := by
                rw [hx14]
              rw [h1, h2, g81_post, lut3_isum_succ]
          · rw [g81_post]; omega
      · rw [if_neg hg]
        simp only [WP.spec_ok]
        have hgv : i.val ≤ g8p.val := by scalar_tac
        have hveq : i.val = g8p.val := le_antisymm hgv hle
        exact ⟨rfl, by rw [hsh, hveq], by rw [hsl, hveq], by rw [hsx, hveq]⟩
    · exact hinv
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  exact ⟨r, hr, hpost⟩

/-! ## Non-vacuity

A concrete witness: a `LutTables` whose two byte planes each hold 16 zero bytes and whose group
sum is zero, with a one-entry code vector. The bounds of `lut3_p_ok` hold at its first entry and
its closed value is `−508`; the bound-zero instance of `word_tile_accs_inner` has all its
per-iteration hypotheses vacuous and its range hypothesis satisfied at `0`. -/

/-- Witness table: 16 zero bytes in each plane, one zero group sum. -/
def lut3_witTables : kern.lut.LutTables :=
  ⟨alloc.vec.Vec.from (List.replicate 16 0#u8) (by scalar_tac),
   alloc.vec.Vec.from (List.replicate 16 0#u8) (by scalar_tac),
   alloc.vec.Vec.from [0#i16] (by scalar_tac)⟩

/-- Witness code vector: one byte. -/
def lut3_witCodes : alloc.vec.Vec Std.U8 :=
  alloc.vec.Vec.from [0#u8] (by scalar_tac)

/-- `getD` of the zero-filled byte plane is the zero byte at every index. -/
lemma lut3_zero_getD (m : Nat) : (List.replicate 16 0#u8).getD m 0#u8 = 0#u8 := by
  by_cases h : m < 16
  · exact List.getD_replicate 0#u8 h
  · exact List.getD_eq_default _ _ (by simpa using h)

/-- The witness table's first entry reconstructs to `−508`. -/
example : lut3_pval lut3_witTables 0 0 = -508 := by
  unfold lut3_pval
  simp [lut3_witTables]

/-- The witness range holds at `0`. -/
example : lut3_rng 0 := by
  unfold lut3_rng
  rw [I64.min_eq, I64.max_eq]
  norm_num

/-- `lut3_p_ok`'s hypotheses hold jointly on the witness table at `(0, 0)`; the function returns
`−508`, the closed value. -/
example (hconv : lut3_fromU8I16_ok) :
    ∃ r, kern.lut.LutTables.p lut3_witTables 0#usize 0#usize = ok r ∧ r.val = -508 := by
  have hp := lut3_p_ok hconv lut3_witTables 0#usize 0#usize
    (by simp [lut3_witTables]) (by simp [lut3_witTables]) (by simp [lut3_witTables])
  have hv : lut3_pval lut3_witTables (↑(0#usize)) (↑(0#usize)) = -508 := by
    unfold lut3_pval
    simp [lut3_witTables]
  rw [hv] at hp
  exact WP.spec_imp_exists hp

/-- `word_tile_accs_inner`'s hypotheses hold jointly at bound `0` (the per-group hypotheses are
vacuous, the invariant holds at `0`, and the range holds at `0`). -/
example (hconv : lut3_fromU8I16_ok) :
    ∃ r, kern.lut.tile_accs_portable_loop0_loop0_loop0_loop0 0#usize 0#usize lut3_witCodes
        lut3_witTables 0#usize 0#usize 0#i64 0#i64 0#i64 0#usize 0#usize = ok r ∧
      lut3_innerPost lut3_witCodes lut3_witTables 0 0 0 0 0 r := by
  have h0 : (↑(0#usize) : Nat) = 0 := rfl
  have h0i : (↑(0#i64) : Int) = 0 := rfl
  refine word_tile_accs_inner hconv 0#usize 0#usize lut3_witCodes lut3_witTables 0#usize
    0#usize 0#usize 0#i64 0#i64 0#i64 0#usize ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · unfold lut3_innerInv
    simp only [h0, h0i, lut3_isum_zero]
    norm_num
  · exact Nat.zero_le _
  · intro j hj; rw [h0] at hj; omega
  · intro j hj; rw [h0] at hj; omega
  · intro j hj; rw [h0] at hj; omega
  · intro m hm
    rw [show lut3_witTables.hi.val.getD m 0#u8 = 0#u8 from by
      rw [show lut3_witTables.hi.val = List.replicate 16 0#u8 from by simp [lut3_witTables]]
      exact lut3_zero_getD m]
    simp
  · intro j hj; rw [h0] at hj; omega
  · intro k hk
    rw [h0] at hk
    have hk0 : k = 0 := by omega
    subst hk0
    refine ⟨?_, ?_, ?_⟩ <;>
      (rw [lut3_isum_zero]; unfold lut3_rng; rw [I64.min_eq, I64.max_eq]; norm_num)

end apeiron_word

