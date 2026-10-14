/-
Register stage refinement over `R[x]/(x² − σ)` (`QREG-W2-T4B`, epic `docs/epics/quantum-register.md`).

The Rust body is `src/numeric/register.rs`; the extraction root is `apeiron_word.word_stage_mul`
(`proof/ApeironWord.lean`, source `lib.rs` lines 867:0-874:1), which calls the real
`register.stage_mul` body over a `&mut [i64]` slice. The canonical target is
`apeiron_word.Register.cellStage` (`Numeric/Quantum/Register.lean`): a stage at span `2^k` multiplies every
pair `(x i, x (i + 2^k))` by the cell `w` in `R[x]/(x² − σ)`.

## Landed here

* `checked_mul_spec`: on its no-overflow domain the extracted `cell.CellI64.checked_mul` returns
  `some c` whose two slots are the exact integer products-and-sums of `w·p`, with
  `denoteCell c = ringMul (denoteSigma s) (denoteCell w) (denoteCell p)`.
* `checked_mul_none`, `checked_mul_none_of_a`: an out-of-range product or first-slot sum returns
  `none`, so the checked multiply is partial and not silently wrapping.
* the three validation refusals of `register.stage_mul` — `NotPowerOfTwo` (not a power of two, and
  `span ≥ 2^16` separately) and `LengthMismatch` — each leaving the input slice unchanged.
* `stage_block_spec`: `register.stage_block` returns `none` with every pair of its block replaced by
  the two slots of its checked product (`blockRef`), the pairs read from the list the block started
  from, so a written position is never read again.
* `stage_mul_loop_spec`, `stageRef_denote`, `stage_mul_denote`: the outer `register.stage_mul_loop`
  walks the register in blocks of `2·span` and the extracted `register.stage_mul` returns `Ok ()`
  with entry `i < x.len` denoting `Register.cellStage (denoteSigma s) (denoteCell w) k (denoteArr x)`
  at `span = 2^k` on its success path.

## Not claimed

Nothing here is a claim about floating point or about the `f32` training path: the body is exact
integer arithmetic on `i64` and every statement is over `ZMod (2^64)` or over the `i64` values
themselves. No claim is made past the stated domain: a `span` of `2^16` or more, a length that is
not a multiple of `2·span`, and a pair whose product or sum leaves `i64` are outside every
success-path statement here. The kernel ABI, rustc/LLVM codegen of `src/numeric/register.rs`, and
the growable-table consumer of the stage are out of scope.

## Complexity

A stage over `x.len` entries costs `x.len / 2` cell multiplies — the pair count of
`Numeric.Quantum.Register.register_stage_pairs` — and each cell multiply is 4 `i64` products and 2 checked
sums (`Cell::checked_mul`), so `2 · x.len` products and `x.len` checked sums, `O(1)` extra space.
`pairStage` evaluates the pair map once per output entry, so the Lean definition does `x.len`
evaluations; the cost model counts each pair once.

The axiom cone is `{propext, Classical.choice, Quot.sound}`; no `sorry`, `axiom`, `native_decide`,
`decide` over an extracted body, `maxHeartbeats 0` or raised `maxRecDepth` appears.
-/
import ApeironWord
import WordRefinement
import Numeric.Quantum.Register

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result
open Classical

noncomputable section

namespace RegisterRefinement

/-! ## A · the checked cell multiply

`cell.CellI64.checked_mul` is the extracted `Cell<i64>::checked_mul` (`src/cell.rs:279`). It is the
`?`-chain the generator emits for the four products and two sums of
`(a₁a₂ + σ b₁b₂, a₁b₂ + b₁a₂)`, each narrowed through `Option::branch`. -/

/-- The five intermediates of `cell.CellI64.checked_mul w s p` all fit `i64`: the three products
`w.a·p.a`, `σ·w.b` and `(σ·w.b)·p.b`, and the two sums that form the result's slots. This is the
no-overflow domain; outside it the body returns `none` (`checked_mul_none`). -/
def MulFits (w : cell.Cell I64) (s : cell.Sigma) (p : cell.Cell I64) : Prop :=
  I64.min ≤ w.a.val * p.a.val ∧ w.a.val * p.a.val ≤ I64.max ∧
  I64.min ≤ (sigmaI64 s).val * w.b.val ∧ (sigmaI64 s).val * w.b.val ≤ I64.max ∧
  I64.min ≤ ((sigmaI64 s).val * w.b.val) * p.b.val ∧
    ((sigmaI64 s).val * w.b.val) * p.b.val ≤ I64.max ∧
  I64.min ≤ w.a.val * p.a.val + ((sigmaI64 s).val * w.b.val) * p.b.val ∧
    w.a.val * p.a.val + ((sigmaI64 s).val * w.b.val) * p.b.val ≤ I64.max ∧
  I64.min ≤ w.a.val * p.b.val ∧ w.a.val * p.b.val ≤ I64.max ∧
  I64.min ≤ w.b.val * p.a.val ∧ w.b.val * p.a.val ≤ I64.max ∧
  I64.min ≤ w.a.val * p.b.val + w.b.val * p.a.val ∧
    w.a.val * p.b.val + w.b.val * p.a.val ≤ I64.max

/-- `I64.checked_mul` succeeds inside range and returns the exact product. -/
theorem i64_mul_some (x y : I64) (hlo : I64.min ≤ x.val * y.val)
    (hhi : x.val * y.val ≤ I64.max) :
    ∃ z, x.checked_mul y = some z ∧ z.val = x.val * y.val := by
  have hspec := I64.checked_mul_bv_spec x y
  cases hc : core.num.checked_mul_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I64.checked_mul] using hc, hspec.2.2.1⟩

/-- `I64.checked_add` succeeds inside range and returns the exact sum. -/
theorem i64_add_some (x y : I64) (hlo : I64.min ≤ x.val + y.val)
    (hhi : x.val + y.val ≤ I64.max) :
    ∃ z, x.checked_add y = some z ∧ z.val = x.val + y.val := by
  have hspec := I64.checked_add_bv_spec x y
  cases hc : core.num.checked_add_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I64.checked_add] using hc, hspec.2.2.1⟩

/-- A successful checked product is ring multiplication under `denote`. -/
theorem denote_checked_mul {x y z : I64} (h : I64.checked_mul x y = some z) :
    denote z = denote x * denote y := by
  have hspec := I64.checked_mul_bv_spec x y
  have hb : core.num.checked_mul_IScalar x y = some z := by simpa only [I64.checked_mul] using h
  rw [hb] at hspec
  obtain ⟨_, _, hval, _⟩ := hspec
  rw [denote_eq_intCast z, denote_eq_intCast x, denote_eq_intCast y, hval]
  push_cast
  ring

/-- A successful checked sum is ring addition under `denote`. -/
theorem denote_checked_add {x y z : I64} (h : I64.checked_add x y = some z) :
    denote z = denote x + denote y := by
  have hspec := I64.checked_add_bv_spec x y
  have hb : core.num.checked_add_IScalar x y = some z := by simpa only [I64.checked_add] using h
  rw [hb] at hspec
  obtain ⟨_, _, hval, _⟩ := hspec
  rw [denote_eq_intCast z, denote_eq_intCast x, denote_eq_intCast y, hval]
  push_cast
  ring

/-- **Item A · the checked cell multiply succeeds on its no-overflow domain.** Every intermediate of
`cell.CellI64.checked_mul w s p` fits `i64`, the body returns `some c`, `c`'s two slots are the
exact integer products-and-sums of `w·p`, and `denoteCell c` is the canonical product
`ringMul (denoteSigma s) (denoteCell w) (denoteCell p)`. -/
theorem checked_mul_spec (w : cell.Cell I64) (s : cell.Sigma) (p : cell.Cell I64)
    (h : MulFits w s p) :
    ∃ c : cell.Cell I64, cell.CellI64.checked_mul w s p = ok (some c) ∧
      c.a.val = w.a.val * p.a.val + ((sigmaI64 s).val * w.b.val) * p.b.val ∧
      c.b.val = w.a.val * p.b.val + w.b.val * p.a.val ∧
      denoteCell c =
        Apeiron.Trichotomy.AnyRing.ringMul (denoteSigma s) (denoteCell w) (denoteCell p) := by
  obtain ⟨ha, ha', hb, hb', hc, hc', hd, hd', he, he', hf, hf', hg, hg'⟩ := h
  obtain ⟨qa, hqa, hqav⟩ := i64_mul_some w.a p.a ha ha'
  obtain ⟨sb, hsb, hsBv⟩ := i64_mul_some (sigmaI64 s) w.b hb hb'
  obtain ⟨sbp, hsBp, hsBpv⟩ :=
    i64_mul_some sb p.b (by rw [hsBv]; exact hc) (by rw [hsBv]; exact hc')
  have hsBpv' : sbp.val = ((sigmaI64 s).val * w.b.val) * p.b.val := by rw [hsBpv, hsBv]
  obtain ⟨q1, hq1, hq1v⟩ :=
    i64_add_some qa sbp (by rw [hqav, hsBpv']; exact hd) (by rw [hqav, hsBpv']; exact hd')
  have hq1v' : q1.val = w.a.val * p.a.val + ((sigmaI64 s).val * w.b.val) * p.b.val := by
    rw [hq1v, hqav, hsBpv']
  obtain ⟨ab, hab, habv⟩ := i64_mul_some w.a p.b he he'
  obtain ⟨ba, hba, hbav⟩ := i64_mul_some w.b p.a hf hf'
  obtain ⟨q2, hq2, hq2v⟩ :=
    i64_add_some ab ba (by rw [habv, hbav]; exact hg) (by rw [habv, hbav]; exact hg')
  have hq2v' : q2.val = w.a.val * p.b.val + w.b.val * p.a.val := by rw [hq2v, habv, hbav]
  refine ⟨⟨q1, q2⟩, ?_, hq1v', hq2v', ?_⟩
  · rw [cell.CellI64.checked_mul, sigma_value_eq]
    simp only [lift, bind_tc_ok, hqa, hsb, hsBp, hq1, hab, hba, hq2,
      core.option.Option.Insts.CoreOpsTry_traitTry.branch,
      core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual]
  · simp only [denoteCell, denoteSigma, denote_eq_intCast, Apeiron.Trichotomy.AnyRing.ringMul,
      Prod.mk.injEq, hq1v', hq2v']
    constructor <;> push_cast <;> ring

/-- **Item A · the checked cell multiply refuses an out-of-range product.** With the three products
in range and the first slot's exact sum outside `i64`, the body returns `none`. -/
theorem checked_mul_none (w : cell.Cell I64) (s : cell.Sigma) (p : cell.Cell I64)
    (ha : I64.min ≤ w.a.val * p.a.val) (ha' : w.a.val * p.a.val ≤ I64.max)
    (hb : I64.min ≤ (sigmaI64 s).val * w.b.val) (hb' : (sigmaI64 s).val * w.b.val ≤ I64.max)
    (hc : I64.min ≤ ((sigmaI64 s).val * w.b.val) * p.b.val)
    (hc' : ((sigmaI64 s).val * w.b.val) * p.b.val ≤ I64.max)
    (hd : ¬ (I64.min ≤ w.a.val * p.a.val + ((sigmaI64 s).val * w.b.val) * p.b.val ∧
      w.a.val * p.a.val + ((sigmaI64 s).val * w.b.val) * p.b.val ≤ I64.max)) :
    cell.CellI64.checked_mul w s p = ok none := by
  obtain ⟨qa, hqa, hqav⟩ := i64_mul_some w.a p.a ha ha'
  obtain ⟨sb, hsb, hsBv⟩ := i64_mul_some (sigmaI64 s) w.b hb hb'
  obtain ⟨sbp, hsBp, hsBpv⟩ :=
    i64_mul_some sb p.b (by rw [hsBv]; exact hc) (by rw [hsBv]; exact hc')
  have hsBpv' : sbp.val = ((sigmaI64 s).val * w.b.val) * p.b.val := by rw [hsBpv, hsBv]
  have hadd : I64.checked_add qa sbp = none := by
    have hspec := I64.checked_add_bv_spec qa sbp
    cases hm : core.num.checked_add_IScalar qa sbp with
    | none => simpa only [I64.checked_add] using hm
    | some z =>
        rw [hm] at hspec
        rw [hqav, hsBpv'] at hspec
        exact absurd ⟨hspec.1, hspec.2.1⟩ hd
  rw [cell.CellI64.checked_mul, sigma_value_eq]
  simp only [lift, bind_tc_ok, hqa, hsb, hsBp, hadd,
    core.option.Option.Insts.CoreOpsTry_traitTry.branch,
    core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual]

/-- **Item A · the first product leaves `i64`.** Out of range, the very first checked product is
`none` and so is the result. -/
theorem checked_mul_none_of_a (w : cell.Cell I64) (s : cell.Sigma) (p : cell.Cell I64)
    (h : ¬ (I64.min ≤ w.a.val * p.a.val ∧ w.a.val * p.a.val ≤ I64.max)) :
    cell.CellI64.checked_mul w s p = ok none := by
  have hc : I64.checked_mul w.a p.a = none := by
    have hspec := I64.checked_mul_bv_spec w.a p.a
    cases hm : core.num.checked_mul_IScalar w.a p.a with
    | none => simpa only [I64.checked_mul] using hm
    | some z =>
        rw [hm] at hspec
        exact absurd ⟨hspec.1, hspec.2.1⟩ h
  rw [cell.CellI64.checked_mul, sigma_value_eq]
  simp only [lift, bind_tc_ok, hc,
    core.option.Option.Insts.CoreOpsTry_traitTry.branch,
    core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual]

/-! ## B · the validation refusals of `register.stage_mul`

`register.stage_mul` guards the span before it writes: the `is_power_of_two` test, the `< 2^16`
bound and the `x.len % (2·span) = 0` divisibility test. Each failing guard returns
`Ok (Err e, x)` with `x` the input slice, so a refused call leaves the caller's buffer untouched. -/

/-- `Usize` carries at least 32 bits, so `2 ^ 17 ≤ Usize.max`. -/
theorem two_pow_17_le_usize_max : 2 ^ 17 ≤ Usize.max := by
  have hnb : 32 ≤ System.Platform.numBits := by
    rcases System.Platform.numBits_eq with h | h <;> omega
  rw [Usize.max_def, Usize.numBits_def, UScalarTy.Usize_numBits_eq]
  have h : 2 ^ 17 < 2 ^ System.Platform.numBits :=
    Nat.pow_lt_pow_right (by norm_num) (by omega)
  omega

/-- `1#usize <<< 16` is `2^16`, the span ceiling. -/
theorem shift16_eq : (1#usize : Usize) <<< 16#i32 = ok 65536#usize := by
  obtain ⟨i, hi, hiv⟩ :=
    WP.spec_imp_exists (Usize.ShiftLeft_IScalar_spec (1#usize) (16#i32) (by norm_num)
      (by have h16 : (16#i32).val = 16 := by norm_num
          have hnb : 32 ≤ System.Platform.numBits := by
            rcases System.Platform.numBits_eq with h | h <;> omega
          omega))
  have hiv' : i.val = 65536 := by
    obtain ⟨h1, _⟩ := hiv
    rw [h1]
    have h1v : (1#usize : Usize).val = 1 := by norm_num
    have h16t : (16#i32).toNat = 16 := by decide
    rw [h1v, h16t, Nat.shiftLeft_eq, one_mul]
    rw [show (2 : Nat) ^ 16 = 65536 by norm_num]
    apply Nat.mod_eq_of_lt
    rw [Usize.size_def, Usize.numBits_def, UScalarTy.Usize_numBits_eq]
    have hnb : 32 ≤ System.Platform.numBits := by
      rcases System.Platform.numBits_eq with h | h <;> omega
    have h : 2 ^ 17 ≤ 2 ^ System.Platform.numBits :=
      Nat.pow_le_pow_right (by norm_num) (by omega)
    omega
  rw [hi]
  exact congrArg Result.ok (UScalar.eq_of_val_eq (by rw [hiv']; rfl))

/-- **Not a power of two.** When `span.is_power_of_two()` is false (which includes `span = 0`),
the body returns `Err NotPowerOfTwo` with `x` untouched. -/
theorem stage_mul_not_pow (x : Slice I64) (span : Usize) (w : cell.Cell I64)
    (sigma : cell.Sigma) (h : span.val.isPowerOfTwo = false) :
    word_stage_mul x span w sigma =
      ok (core.result.Result.Err register.RegisterError.NotPowerOfTwo, x) := by
  have hb : core.num.Usize.is_power_of_two span = ok false := by
    simp [core.num.Usize.is_power_of_two, h]
  rw [word_stage_mul, register.stage_mul.eq_def, hb]
  simp

/-- **Span at least `2^16`.** With a power-of-two span but `span ≥ 2^16`, the body returns
`Err NotPowerOfTwo` with `x` untouched. -/
theorem stage_mul_too_large (x : Slice I64) (span : Usize) (w : cell.Cell I64)
    (sigma : cell.Sigma) (hpow : span.val.isPowerOfTwo = true) (hge : 2 ^ 16 ≤ span.val) :
    word_stage_mul x span w sigma =
      ok (core.result.Result.Err register.RegisterError.NotPowerOfTwo, x) := by
  have hb : core.num.Usize.is_power_of_two span = ok true := by
    simp [core.num.Usize.is_power_of_two, hpow]
  have hle : span ≥ 65536#usize := by
    have h16 : (65536#usize).val = 65536 := by norm_num
    rw [ge_iff_le, UScalar.le_equiv, h16]
    norm_num at hge
    exact hge
  rw [word_stage_mul, register.stage_mul.eq_def, hb, shift16_eq]
  simp only [bind_tc_ok, if_true, if_pos hle]

/-- **Length mismatch.** With a span in `[1, 2^16)` and `x.len` not a multiple of `2·span`, the
body returns `Err LengthMismatch` with `x` untouched. -/
theorem stage_mul_length_mismatch (x : Slice I64) (span : Usize) (w : cell.Cell I64)
    (sigma : cell.Sigma) (hpow : span.val.isPowerOfTwo = true) (hlt : ¬ 2 ^ 16 ≤ span.val)
    (hdiv : x.val.length % (2 * span.val) ≠ 0) :
    word_stage_mul x span w sigma =
      ok (core.result.Result.Err register.RegisterError.LengthMismatch, x) := by
  have hb : core.num.Usize.is_power_of_two span = ok true := by
    simp [core.num.Usize.is_power_of_two, hpow]
  have hspos : 0 < span.val := by
    by_contra h0
    have h0' : span.val = 0 := by omega
    rw [h0'] at hpow
    simp [Nat.not_isPowerOfTwo_zero] at hpow
  have hmax : (2#usize).val * span.val ≤ Usize.max := by
    have h2 : (2#usize).val = 2 := by norm_num
    rw [h2]
    have hsp : span.val < 65536 := by
      have hp : (2 : Nat) ^ 16 = 65536 := by norm_num
      omega
    have h217 : 2 * span.val ≤ 2 ^ 17 := by
      have hp : (2 : Nat) ^ 17 = 131072 := by norm_num
      omega
    exact le_trans h217 two_pow_17_le_usize_max
  obtain ⟨block, hblock, hblockv⟩ :=
    WP.spec_imp_exists (Usize.mul_spec (x := 2#usize) (y := span) hmax)
  have hblockv' : block.val = 2 * span.val := by rw [hblockv]; norm_num
  obtain ⟨r, hr, hrv⟩ :=
    WP.spec_imp_exists (Usize.rem_spec (Slice.len x) (y := block) (by rw [hblockv']; omega))
  have hrv' : r.val = x.val.length % (2 * span.val) := by
    rw [hrv, Slice.len_val, hblockv']
  have hrne : r ≠ 0#usize := by
    intro h0
    have h1 : r.val = 0 := by rw [h0]; norm_num
    exact hdiv (by rw [← hrv', h1])
  have hlt' : ¬ span ≥ 65536#usize := by
    have h16 : (65536#usize).val = 65536 := by norm_num
    rw [ge_iff_le, UScalar.le_equiv, h16]
    norm_num at hlt
    omega
  rw [word_stage_mul, register.stage_mul.eq_def, hb, shift16_eq]
  simp only [bind_tc_ok, hblock, hr, if_true, if_neg hlt', if_pos (bne_iff_ne.mpr hrne)]

/-! ## C · the block loop

`register.stage_block` multiplies the `span` pairs of the block at `base`; pair `j` is
`(x[base+j], x[base+j+span])` and every pair is read from the list the loop started from, so a
written position is never read again. `blockRef` folds exactly those updates. -/

/-- The output cell of a successful `checked_mul`, named without an existential: on `MulFits` it is
the `checked_mul_spec` witness (`mulCell_checked`). -/
noncomputable def mulCell (w : cell.Cell I64) (s : cell.Sigma) (p : cell.Cell I64) : cell.Cell I64 :=
  if h : MulFits w s p then Classical.choose (checked_mul_spec w s p h) else ⟨0#i64, 0#i64⟩

theorem mulCell_checked (w : cell.Cell I64) (s : cell.Sigma) (p : cell.Cell I64)
    (h : MulFits w s p) :
    cell.CellI64.checked_mul w s p = ok (some (mulCell w s p)) := by
  unfold mulCell
  rw [dif_pos h]
  exact (Classical.choose_spec (checked_mul_spec w s p h)).1

/-- `acc` with the pairs `0, …, lane − 1` of the block at `base` replaced by their checked products;
pair `j` is read from `zx`, the list the loop started from. -/
def blockRef (w : cell.Cell I64) (s : cell.Sigma) (base span : Nat) (zx : List I64)
    (acc : List I64) : Nat → List I64
  | 0 => acc
  | lane + 1 =>
      let c := mulCell w s ⟨zx[base + lane]!, zx[base + lane + span]!⟩
      ((blockRef w s base span zx acc lane).set (base + lane) c.a).set (base + lane + span) c.b

@[simp] theorem blockRef_zero (w : cell.Cell I64) (s : cell.Sigma) (base span : Nat)
    (zx acc : List I64) : blockRef w s base span zx acc 0 = acc := rfl

@[simp] theorem blockRef_succ (w : cell.Cell I64) (s : cell.Sigma) (base span : Nat)
    (zx acc : List I64) (lane : Nat) :
    blockRef w s base span zx acc (lane + 1) =
      ((blockRef w s base span zx acc lane).set (base + lane)
          (mulCell w s ⟨zx[base + lane]!, zx[base + lane + span]!⟩).a).set (base + lane + span)
        (mulCell w s ⟨zx[base + lane]!, zx[base + lane + span]!⟩).b := rfl

theorem blockRef_length (w : cell.Cell I64) (s : cell.Sigma) (base span : Nat) (zx acc : List I64)
    (lane : Nat) : (blockRef w s base span zx acc lane).length = acc.length := by
  induction lane with
  | zero => simp
  | succ n ih => simp only [blockRef_succ, List.length_set]; exact ih

/-- A position not among the first `lane` pair-updates is unchanged by them. -/
theorem blockRef_getElem!_untouched (w : cell.Cell I64) (s : cell.Sigma) (base span : Nat)
    (zx acc : List I64) (lane j : Nat)
    (h : ∀ i < lane, j ≠ base + i ∧ j ≠ base + i + span) :
    (blockRef w s base span zx acc lane)[j]! = acc[j]! := by
  induction lane with
  | zero => rfl
  | succ n ih =>
      obtain ⟨h1, h2⟩ := h n (Nat.lt_succ_self n)
      have hn1 : Nat.not_eq (base + n) j := Or.inl (by omega)
      have hn2 : Nat.not_eq (base + n + span) j := Or.inl (by omega)
      rw [blockRef_succ,
        List.set_getElem!_ne (i := base + n + span) (j := j) (h := hn2),
        List.set_getElem!_ne (i := base + n) (j := j) (h := hn1)]
      exact ih fun i hi => h i (Nat.lt_trans hi (Nat.lt_succ_self n))

/-- The low slot of the pair at label `j` of the block, once `j` has been updated. -/
theorem blockRef_getElem!_low (w : cell.Cell I64) (s : cell.Sigma) (base span : Nat)
    (zx acc : List I64) (hlen : base + 2 * span ≤ acc.length) (hacc : zx.length = acc.length)
    (lane j : Nat) (hle : lane ≤ span) (hj : j < lane) :
    (blockRef w s base span zx acc lane)[base + j]! =
      (mulCell w s ⟨zx[base + j]!, zx[base + j + span]!⟩).a := by
  induction lane with
  | zero => omega
  | succ n ih =>
      rw [blockRef_succ]
      rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hj | hjn
      · have h1 : Nat.not_eq (base + n) (base + j) := Or.inl (by omega)
        have h2 : Nat.not_eq (base + n + span) (base + j) := Or.inl (by omega)
        rw [List.set_getElem!_ne (i := base + n + span) (j := base + j) (h := h2),
          List.set_getElem!_ne (i := base + n) (j := base + j) (h := h1)]
        exact ih (by omega) hj
      · rw [← hjn]
        have hlen' : base + j < (blockRef w s base span zx acc j).length := by
          rw [blockRef_length]; omega
        rw [List.set_getElem!_ne (i := base + j + span) (j := base + j) (h := Or.inl (by omega)),
          List.set_getElem!_eq (i := base + j) (i' := base + j) (h := ⟨hlen', rfl⟩)]

/-- The high slot of the pair at label `j` of the block, once `j` has been updated. -/
theorem blockRef_getElem!_high (w : cell.Cell I64) (s : cell.Sigma) (base span : Nat)
    (zx acc : List I64) (hlen : base + 2 * span ≤ acc.length) (hacc : zx.length = acc.length)
    (lane j : Nat) (hle : lane ≤ span) (hj : j < lane) :
    (blockRef w s base span zx acc lane)[base + j + span]! =
      (mulCell w s ⟨zx[base + j]!, zx[base + j + span]!⟩).b := by
  induction lane with
  | zero => omega
  | succ n ih =>
      rw [blockRef_succ]
      rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hj | hjn
      · have h1 : Nat.not_eq (base + n) (base + j + span) := Or.inl (by omega)
        have h2 : Nat.not_eq (base + n + span) (base + j + span) := Or.inl (by omega)
        rw [List.set_getElem!_ne (i := base + n + span) (j := base + j + span) (h := h2),
          List.set_getElem!_ne (i := base + n) (j := base + j + span) (h := h1)]
        exact ih (by omega) hj
      · rw [← hjn]
        have hlen' : base + j + span < ((blockRef w s base span zx acc j).set
            (base + j) (mulCell w s ⟨zx[base + j]!, zx[base + j + span]!⟩).a).length := by
          rw [List.length_set, blockRef_length]; omega
        rw [List.set_getElem!_eq (i := base + j + span) (i' := base + j + span)
          (h := ⟨hlen', rfl⟩)]

/-- **Item C · the block loop.** With the block in range and every pair of the block inside
`MulFits`, `register.stage_block` returns `none` and the input with every pair of the block replaced
by the two slots of its checked product. -/
theorem stage_block_spec (x : Slice I64) (base span : Usize) (w : cell.Cell I64)
    (sigma : cell.Sigma) (hlen : base.val + 2 * span.val ≤ x.val.length)
    (hfit : ∀ j, j < span.val →
      MulFits w sigma ⟨x.val[base.val + j]!, x.val[base.val + j + span.val]!⟩) :
    ∃ x' : Slice I64, register.stage_block x base span w sigma = ok (none, x') ∧
      x'.val = blockRef w sigma base.val span.val x.val x.val span.val := by
  have hspec : register.stage_block_loop x base span w sigma 0#usize none ⦃ r =>
      r.1 = none ∧ r.2.val = blockRef w sigma base.val span.val x.val x.val span.val ⦄ := by
    unfold register.stage_block_loop
    apply loop.spec_decr_nat
      (measure := fun st : Slice I64 × Usize × Option register.RegisterError =>
        span.val - st.2.1.val)
      (inv := fun st => st.2.1.val ≤ span.val ∧ st.2.2 = none ∧
        st.1.val = blockRef w sigma base.val span.val x.val x.val st.2.1.val)
    · intro st hst
      obtain ⟨x1, lane, err⟩ := st
      obtain ⟨hle, herr, hmap⟩ := hst
      simp only at hle herr hmap
      subst herr
      by_cases hlt : lane < span
      · have hltv : lane.val < span.val := (UScalar.lt_equiv lane span).mp hlt
        unfold register.stage_block_loop.body
        dsimp only
        rw [if_pos hlt, if_pos (show core.option.Option.is_none
          (none : Option register.RegisterError) = true from rfl)]
        refine WP.spec_bind (Usize.add_spec (x := base) (y := lane) (by
          have := Slice.length_ineq x; omega)) ?_
        intro lo hlo
        have hlov : lo.val = base.val + lane.val := hlo
        refine WP.spec_bind (Usize.add_spec (x := lo) (y := span) (by
          have := Slice.length_ineq x; omega)) ?_
        intro hi hhi
        have hhiv : hi.val = base.val + lane.val + span.val := by rw [hhi, hlov]
        have hx1len : x1.val.length = x.val.length := by rw [hmap, blockRef_length]
        have hlo_lt : lo.val < x1.val.length := by rw [hx1len]; omega
        have hhi_lt : hi.val < x1.val.length := by rw [hx1len]; omega
        refine WP.spec_bind (Slice.index_usize_spec x1 lo hlo_lt) ?_
        intro iv hiv
        refine WP.spec_bind (Slice.index_usize_spec x1 hi hhi_lt) ?_
        intro iv1 hiv1
        have hiv_eq : iv = x.val[base.val + lane.val]! := by
          rw [hiv, List.Inhabited_getElem_eq_getElem! x1.val lo.val hlo_lt, hmap, hlov]
          exact blockRef_getElem!_untouched w sigma base.val span.val x.val x.val lane.val
            (base.val + lane.val) (fun i hi' => by omega)
        have hiv1_eq : iv1 = x.val[base.val + lane.val + span.val]! := by
          rw [hiv1, List.Inhabited_getElem_eq_getElem! x1.val hi.val hhi_lt, hmap, hhiv]
          exact blockRef_getElem!_untouched w sigma base.val span.val x.val x.val lane.val
            (base.val + lane.val + span.val) (fun i hi' => by omega)
        have hmul : cell.CellI64.checked_mul w sigma { a := iv, b := iv1 } =
            ok (some (mulCell w sigma { a := iv, b := iv1 })) := by
          refine mulCell_checked w sigma _ ?_
          rw [hiv_eq, hiv1_eq]
          exact hfit lane.val hltv
        rw [hmul]
        simp only [bind_tc_ok]
        refine WP.spec_bind (Slice.update_spec x1 lo
          (mulCell w sigma { a := iv, b := iv1 }).a hlo_lt) ?_
        intro x2 hx2
        refine WP.spec_bind (Slice.update_spec x2 hi (mulCell w sigma { a := iv, b := iv1 }).b
          (by rw [hx2, Slice.set_length]; exact hhi_lt)) ?_
        intro x3 hx3
        refine WP.spec_bind (Usize.add_spec (x := lane) (y := 1#usize) (by
          have h1 : (1#usize).val = 1 := by simp
          rw [h1]
          have hli : x.val.length ≤ Usize.max := Slice.length_ineq x
          omega)) ?_
        intro lane1 hlane1
        have hlane1v : lane1.val = lane.val + 1 := by rw [hlane1]; simp
        rw [WP.spec_ok]
        dsimp only
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · rw [hlane1v]; omega
        · rfl
        · rw [hlane1v, hx3, hx2, Slice.set_val_eq, Slice.set_val_eq, hmap, blockRef_succ,
            hlov, hhiv, hiv_eq, hiv1_eq]
        · rw [hlane1v]; omega
      · unfold register.stage_block_loop.body
        dsimp only
        rw [if_neg hlt, WP.spec_ok]
        have hge : span.val ≤ lane.val := by
          have h' : ¬ lane.val < span.val := fun hc => hlt ((UScalar.lt_equiv lane span).mpr hc)
          omega
        refine ⟨rfl, ?_⟩
        rw [hmap, show lane.val = span.val by omega]
    · refine ⟨by simp, rfl, by simp⟩
  obtain ⟨⟨res, x'⟩, hloop, hpost⟩ := WP.spec_imp_exists hspec
  obtain ⟨hres, hx'⟩ := hpost
  refine ⟨x', ?_, hx'⟩
  rw [register.stage_block.eq_def, hloop, show res = none from hres]

/-! ## D · the stage loop and its denotation

`register.stage_mul` walks the register in blocks of `2·span` entries (`stage_mul_loop`) and each
block multiplies its `span` pairs (`stage_block`). `stageRef` is that walk as a list fold: block `b`
replaces the pairs `(l (b·2·span + j), l (b·2·span + j + span))`, `j < span`, read from the list the
block started from, so a written position is never read again. `stageRef_denote` identifies the fold
with `Register.cellStage` at `k` with `span = 2^k`, and `stage_mul_denote` carries that through the
extracted loop on the success path. -/

/-- The denotation of a slice as a register `ℕ → ZMod (2^64)`; a position past the end reads the
`i64` default `0`. -/
def denoteArr (l : List I64) : ℕ → ZMod (2 ^ 64) := fun i => denote l[i]!

/-- The list after blocks `0, …, m − 1` of a stage have multiplied their pairs. -/
def stageRef (w : cell.Cell I64) (s : cell.Sigma) (span : Nat) : Nat → List I64 → List I64
  | 0, l => l
  | m + 1, l =>
      blockRef w s (m * (2 * span)) span (stageRef w s span m l) (stageRef w s span m l) span

@[simp] theorem stageRef_zero (w : cell.Cell I64) (s : cell.Sigma) (span : Nat) (l : List I64) :
    stageRef w s span 0 l = l := rfl

@[simp] theorem stageRef_succ (w : cell.Cell I64) (s : cell.Sigma) (span m : Nat) (l : List I64) :
    stageRef w s span (m + 1) l =
      blockRef w s (m * (2 * span)) span (stageRef w s span m l) (stageRef w s span m l) span :=
  rfl

theorem stageRef_length (w : cell.Cell I64) (s : cell.Sigma) (span m : Nat) (l : List I64) :
    (stageRef w s span m l).length = l.length := by
  induction m with
  | zero => rfl
  | succ n ih => rw [stageRef_succ, blockRef_length, ih]

/-- A position at or past the end of block `m − 1` is unchanged by the first `m` blocks. -/
theorem stageRef_untouched (w : cell.Cell I64) (s : cell.Sigma) (span m : Nat) (l : List I64)
    (j : Nat) (hspan : 0 < span) (h : m * (2 * span) ≤ j) :
    (stageRef w s span m l)[j]! = l[j]! := by
  induction m with
  | zero => rfl
  | succ n ih =>
      have h2n : (n + 1) * (2 * span) = n * (2 * span) + 2 * span := by ring
      have hnt : ∀ t < span, j ≠ n * (2 * span) + t ∧ j ≠ n * (2 * span) + t + span :=
        fun t ht => ⟨by omega, by omega⟩
      rw [stageRef_succ, blockRef_getElem!_untouched (h := hnt)]
      exact ih (by omega)

/-- The value at a position inside block `b` after `m` blocks: the low slot of its pair when the
position is in the block's first half, the high slot when it is in the second. -/
theorem stageRef_getElem!_block (w : cell.Cell I64) (s : cell.Sigma) (span m : Nat) (l : List I64)
    (hspan : 0 < span) (hlen : m * (2 * span) ≤ l.length) (b j : Nat) (hb : b < m)
    (hj : j < 2 * span) :
    (stageRef w s span m l)[b * (2 * span) + j]! =
      if j < span then
        (mulCell w s ⟨l[b * (2 * span) + j]!, l[b * (2 * span) + j + span]!⟩).a
      else (mulCell w s ⟨l[b * (2 * span) + (j - span)]!, l[b * (2 * span) + j]!⟩).b := by
  induction m with
  | zero => omega
  | succ n ih =>
      rw [stageRef_succ]
      have h2n : (n + 1) * (2 * span) = n * (2 * span) + 2 * span := by ring
      have hlen' : n * (2 * span) ≤ l.length := by omega
      rcases Nat.lt_succ_iff_lt_or_eq.mp hb with hbn | hbn
      · have hbmul : (b + 1) * (2 * span) ≤ n * (2 * span) :=
          Nat.mul_le_mul_right _ (by omega)
        have hb1 : (b + 1) * (2 * span) = b * (2 * span) + 2 * span := by ring
        have hnt : ∀ t < span,
            b * (2 * span) + j ≠ n * (2 * span) + t ∧
              b * (2 * span) + j ≠ n * (2 * span) + t + span :=
          fun t ht => ⟨by omega, by omega⟩
        rw [blockRef_getElem!_untouched (h := hnt)]
        exact ih hlen' hbn
      · rw [hbn]
        have hpos : n * (2 * span) + 2 * span ≤ (stageRef w s span n l).length := by
          rw [stageRef_length, ← h2n]; exact hlen
        by_cases hjs : j < span
        · have e1 : (stageRef w s span n l)[n * (2 * span) + j]! =
              l[n * (2 * span) + j]! :=
            stageRef_untouched w s span n l (n * (2 * span) + j) hspan (by omega)
          have e2 : (stageRef w s span n l)[n * (2 * span) + j + span]! =
              l[n * (2 * span) + j + span]! :=
            stageRef_untouched w s span n l (n * (2 * span) + j + span) hspan (by omega)
          rw [blockRef_getElem!_low w s (n * (2 * span)) span (stageRef w s span n l)
              (stageRef w s span n l) hpos rfl span j le_rfl hjs, e1, e2, if_pos hjs]
        · have hj' : j - span < span := by omega
          have hidx : n * (2 * span) + j = n * (2 * span) + (j - span) + span := by omega
          have e1 : (stageRef w s span n l)[n * (2 * span) + (j - span)]! =
              l[n * (2 * span) + (j - span)]! :=
            stageRef_untouched w s span n l (n * (2 * span) + (j - span)) hspan (by omega)
          have e2 : (stageRef w s span n l)[n * (2 * span) + (j - span) + span]! =
              l[n * (2 * span) + (j - span) + span]! :=
            stageRef_untouched w s span n l (n * (2 * span) + (j - span) + span) hspan
              (by omega)
          rw [hidx, blockRef_getElem!_high w s (n * (2 * span)) span (stageRef w s span n l)
              (stageRef w s span n l) hpos rfl span (j - span) le_rfl hj', e1, e2, if_neg hjs]

/-! ### The checked multiply as a ring product -/

/-- On its no-overflow domain the named `mulCell` output has the canonical ring product as its
denotation. -/
theorem mulCell_denote (w : cell.Cell I64) (s : cell.Sigma) (p : cell.Cell I64)
    (h : MulFits w s p) :
    denoteCell (mulCell w s p) =
      Apeiron.Trichotomy.AnyRing.ringMul (denoteSigma s) (denoteCell w) (denoteCell p) := by
  unfold mulCell
  rw [dif_pos h]
  exact (Classical.choose_spec (checked_mul_spec w s p h)).2.2.2

theorem denote_mulCell_fst (w : cell.Cell I64) (s : cell.Sigma) (p : cell.Cell I64)
    (h : MulFits w s p) :
    denote ((mulCell w s p).a) =
      (Apeiron.Trichotomy.AnyRing.ringMul (denoteSigma s) (denoteCell p) (denoteCell w)).1 := by
  have h1 : (denoteCell (mulCell w s p)).1 =
      (Apeiron.Trichotomy.AnyRing.ringMul (denoteSigma s) (denoteCell w) (denoteCell p)).1 :=
    congrArg Prod.fst (mulCell_denote w s p h)
  rw [Apeiron.Trichotomy.AnyRing.ringMul_comm] at h1
  simpa only [denoteCell] using h1

theorem denote_mulCell_snd (w : cell.Cell I64) (s : cell.Sigma) (p : cell.Cell I64)
    (h : MulFits w s p) :
    denote ((mulCell w s p).b) =
      (Apeiron.Trichotomy.AnyRing.ringMul (denoteSigma s) (denoteCell p) (denoteCell w)).2 := by
  have h1 : (denoteCell (mulCell w s p)).2 =
      (Apeiron.Trichotomy.AnyRing.ringMul (denoteSigma s) (denoteCell w) (denoteCell p)).2 :=
    congrArg Prod.snd (mulCell_denote w s p h)
  rw [Apeiron.Trichotomy.AnyRing.ringMul_comm] at h1
  simpa only [denoteCell] using h1

/-- **Item D · the block fold denotes the canonical cell stage.** With `span = 2^k` and every pair of
every block inside `MulFits`, entry `i < l.length` of the folded list denotes `cellStage` at `k`. -/
theorem stageRef_denote (w : cell.Cell I64) (s : cell.Sigma) (span m k : Nat) (l : List I64)
    (hspan : 0 < span) (hk : span = 2 ^ k) (hmlen : l.length = m * (2 * span))
    (hfit : ∀ b j, b < m → j < span →
      MulFits w s ⟨l[b * (2 * span) + j]!, l[b * (2 * span) + j + span]!⟩) :
    ∀ i, i < l.length →
      denote ((stageRef w s span m l)[i]!) =
        Register.cellStage (denoteSigma s) (denoteCell w) k (denoteArr l) i := by
  intro i hi
  have h2span : 2 * span = 2 ^ (k + 1) := by rw [hk, pow_succ]; ring
  set b := i / (2 * span) with hb
  set j := i % (2 * span) with hj
  have hdecomp : b * (2 * span) + j = i := by
    have h := Nat.div_add_mod i (2 * span)
    rw [hb, hj, Nat.mul_comm (i / (2 * span)) (2 * span)]
    exact h
  have hjm : j < 2 * span := by rw [hj]; exact Nat.mod_lt _ (by omega)
  have hbm : b < m := by
    rw [hb, Nat.div_lt_iff_lt_mul (by omega)]
    rw [← hmlen]
    exact hi
  rw [← hdecomp, stageRef_getElem!_block w s span m l hspan (by rw [hmlen]) b j hbm hjm]
  by_cases hjs : j < span
  · rw [if_pos hjs]
    have hif : (b * (2 * span) + j) % 2 ^ (k + 1) < 2 ^ k := by
      rw [hdecomp, ← h2span, ← hj, ← hk]; exact hjs
    have hf : MulFits w s ⟨l[b * (2 * span) + j]!, l[b * (2 * span) + j + span]!⟩ :=
      hfit b j hbm hjs
    rw [denote_mulCell_fst w s ⟨l[b * (2 * span) + j]!, l[b * (2 * span) + j + span]!⟩ hf]
    unfold Register.cellStage
    rw [Register.pairStage_low _ _ hif]
    simp only [denoteArr, denoteCell]
    rw [← hk]
  · rw [if_neg hjs]
    have hif : ¬ (b * (2 * span) + j) % 2 ^ (k + 1) < 2 ^ k := by
      rw [hdecomp, ← h2span, ← hj, ← hk]; exact hjs
    have hf : MulFits w s ⟨l[b * (2 * span) + (j - span)]!, l[b * (2 * span) + j]!⟩ := by
      have h := hfit b (j - span) hbm (by omega)
      rwa [show b * (2 * span) + (j - span) + span = b * (2 * span) + j from by omega] at h
    rw [denote_mulCell_snd w s ⟨l[b * (2 * span) + (j - span)]!, l[b * (2 * span) + j]!⟩ hf]
    unfold Register.cellStage
    rw [Register.pairStage_high _ _ hif]
    simp only [denoteArr, denoteCell]
    have hidx3 : (b * (2 * span) + j) - 2 ^ k = b * (2 * span) + (j - span) := by
      have h2kj : 2 ^ k ≤ j := by rw [← hk]; omega
      rw [hk]; omega
    rw [hidx3]

/-! ### The stage-mul loop -/

/-- **Item D · the outer stage loop.** With the length an exact multiple of the block `2·span`, every
block pair inside `MulFits` and `block = 2·span`, the extracted loop returns `none` and the block
fold. -/
theorem stage_mul_loop_spec (x : Slice I64) (span block : Usize) (w : cell.Cell I64)
    (sigma : cell.Sigma) (m : Nat) (hspan : 0 < span.val) (hblock : block.val = 2 * span.val)
    (hmlen : x.val.length = m * (2 * span.val))
    (hfit : ∀ b j, b < m → j < span.val →
      MulFits w sigma ⟨x.val[b * (2 * span.val) + j]!, x.val[b * (2 * span.val) + j + span.val]!⟩) :
    ∃ x', register.stage_mul_loop x span w sigma block 0#usize none = ok (x', none) ∧
      x'.val.length = x.val.length ∧ x'.val = stageRef w sigma span.val m x.val := by
  have hspec : register.stage_mul_loop x span w sigma block 0#usize none ⦃ r =>
      r.2 = none ∧ r.1.val = stageRef w sigma span.val m x.val ⦄ := by
    unfold register.stage_mul_loop
    apply loop.spec_decr_nat
      (measure := fun st : Slice I64 × Usize × Option register.RegisterError =>
        x.val.length - st.2.1.val)
      (inv := fun st => st.2.2 = none ∧ ∃ b, b ≤ m ∧ st.2.1.val = b * (2 * span.val) ∧
        st.1.val = stageRef w sigma span.val b x.val)
    · intro st hst
      obtain ⟨x1, base, err⟩ := st
      obtain ⟨herr, b, hbm, hbase, hx1⟩ := hst
      simp only at herr hbm hbase hx1
      subst herr
      by_cases hlt : base < Slice.len x1
      · have hltv : base.val < x.val.length := by
          have h := (UScalar.lt_equiv base (Slice.len x1)).mp hlt
          have h2 : x1.val.length = x.val.length := by rw [hx1, stageRef_length]
          rw [Slice.len_val] at h
          simp only [Slice.length] at h
          omega
        have hbm' : b < m := by
          rw [hbase, hmlen] at hltv
          exact Nat.lt_of_mul_lt_mul_right hltv
        have hfit' : ∀ j, j < span.val → MulFits w sigma
            ⟨x1.val[base.val + j]!, x1.val[base.val + j + span.val]!⟩ := by
          intro j hj
          have h1 : base.val + j = b * (2 * span.val) + j := by rw [hbase]
          have e1 : (stageRef w sigma span.val b x.val)[b * (2 * span.val) + j]! =
              x.val[b * (2 * span.val) + j]! :=
            stageRef_untouched w sigma span.val b x.val (b * (2 * span.val) + j) hspan (by omega)
          have e2 : (stageRef w sigma span.val b x.val)[b * (2 * span.val) + j + span.val]! =
              x.val[b * (2 * span.val) + j + span.val]! :=
            stageRef_untouched w sigma span.val b x.val
              (b * (2 * span.val) + j + span.val) hspan (by omega)
          rw [h1, hx1, e1, e2]
          exact hfit b j hbm' hj
        have hlen : base.val + 2 * span.val ≤ x1.val.length := by
          rw [hx1, stageRef_length, hbase]
          have h2s : (b + 1) * (2 * span.val) = b * (2 * span.val) + 2 * span.val := by ring
          have hmul : (b + 1) * (2 * span.val) ≤ m * (2 * span.val) :=
            Nat.mul_le_mul_right _ (by omega)
          rw [h2s, ← hmlen] at hmul
          exact hmul
        unfold register.stage_mul_loop.body
        dsimp only
        rw [if_pos hlt, if_pos (show core.option.Option.is_none
          (none : Option register.RegisterError) = true from rfl)]
        obtain ⟨x2, hblk, hx2⟩ := stage_block_spec x1 base span w sigma hlen hfit'
        rw [hblk]
        simp only [bind_tc_ok]
        refine WP.spec_bind (Usize.add_spec (x := base) (y := block) (by
          rw [hbase, hblock]
          have hle : x1.val.length ≤ Usize.max := Slice.length_ineq x1
          omega)) ?_
        intro base1 hbase1
        rw [WP.spec_ok]
        refine ⟨⟨rfl, ⟨b + 1, by omega, ?_, ?_⟩⟩, ?_⟩
        · rw [hbase1, hbase, hblock]; ring
        · rw [hx2, hx1, hbase, stageRef_succ]
        · rw [hbase1, hblock]
          omega
      · have hge : x.val.length ≤ base.val := by
          have h := (UScalar.le_equiv (Slice.len x1) base).mp (not_lt.mp hlt)
          have h2 : x1.val.length = x.val.length := by rw [hx1, stageRef_length]
          rw [Slice.len_val] at h
          simp only [Slice.length] at h
          omega
        have hbm' : b = m := by
          rw [hmlen, hbase] at hge
          have : m ≤ b := Nat.le_of_mul_le_mul_right hge (by omega)
          omega
        unfold register.stage_mul_loop.body
        dsimp only
        rw [if_neg hlt, WP.spec_ok]
        refine ⟨rfl, ?_⟩
        rw [hx1, hbm']
    · exact ⟨rfl, ⟨0, by omega, by simp, by simp⟩⟩
  obtain ⟨⟨x', err⟩, hloop, hpost⟩ := WP.spec_imp_exists hspec
  obtain ⟨herr, hx'⟩ := hpost
  refine ⟨x', ?_, ?_, hx'⟩
  · rw [hloop, show err = none from herr]
  · rw [hx', stageRef_length]

/-- **Item D · the extracted stage denotes the canonical cell stage.** On the success path — a
power-of-two span `2^k` below `2^16`, a length an exact multiple of the block `2·span`, and every
block pair inside `MulFits` — `register.stage_mul` returns `Ok ()` with the input slice modified so
that entry `i` denotes `Register.cellStage` at `k` on the register's denotation. -/
theorem stage_mul_denote (x : Slice I64) (span : Usize) (w : cell.Cell I64) (sigma : cell.Sigma)
    (k m : Nat) (hs : span.val = 2 ^ k) (hlt : span.val < 2 ^ 16)
    (hmlen : x.val.length = m * (2 * span.val))
    (hfit : ∀ b j, b < m → j < span.val →
      MulFits w sigma ⟨x.val[b * (2 * span.val) + j]!, x.val[b * (2 * span.val) + j + span.val]!⟩) :
    ∃ x', register.stage_mul x span w sigma = ok (core.result.Result.Ok (), x') ∧
      x'.val.length = x.val.length ∧
      ∀ i, i < x.val.length →
        denote (x'.val[i]!) =
          Register.cellStage (denoteSigma sigma) (denoteCell w) k (denoteArr x.val) i := by
  have hspan : 0 < span.val := by rw [hs]; positivity
  have hpow : span.val.isPowerOfTwo = true := by
    rw [hs]
    simp [Nat.isPowerOfTwo]
  have hb : core.num.Usize.is_power_of_two span = ok true := by
    simp [core.num.Usize.is_power_of_two, hpow]
  have hge : ¬ (span ≥ 65536#usize) := by
    have h16 : (65536#usize).val = 65536 := by norm_num
    rw [ge_iff_le, UScalar.le_equiv, h16]
    have h65536 : (2 : Nat) ^ 16 = 65536 := by norm_num
    omega
  have hmax : (2#usize).val * span.val ≤ Usize.max := by
    have h2 : (2#usize).val = 2 := by norm_num
    rw [h2]
    have h217 : 2 * span.val ≤ 2 ^ 17 := by
      have hp : (2 : Nat) ^ 17 = 131072 := by norm_num
      omega
    exact le_trans h217 two_pow_17_le_usize_max
  obtain ⟨block, hblock, hblockv⟩ :=
    WP.spec_imp_exists (Usize.mul_spec (x := 2#usize) (y := span) hmax)
  have hblockv' : block.val = 2 * span.val := by rw [hblockv]; norm_num
  obtain ⟨r, hr, hrv⟩ :=
    WP.spec_imp_exists (Usize.rem_spec (Slice.len x) (y := block) (by rw [hblockv']; omega))
  have hrv' : r.val = x.val.length % (2 * span.val) := by rw [hrv, Slice.len_val, hblockv']
  have hr0 : r = 0#usize := by
    apply UScalar.eq_of_val_eq
    rw [hrv', hmlen, Nat.mul_comm]
    exact Nat.mul_mod_right (2 * span.val) m
  have hne0 : ¬ ((0#usize != 0#usize) = true) := by
    rw [bne_iff_ne]; exact fun h => h rfl
  rw [register.stage_mul.eq_def, hb, shift16_eq]
  simp only [bind_tc_ok, if_true, if_neg hge, hblock, hr, hr0, if_neg hne0]
  obtain ⟨x1, hloop, hx1len, hx1val⟩ :=
    stage_mul_loop_spec x span block w sigma m hspan hblockv' hmlen hfit
  rw [hloop]
  simp only [bind_tc_ok]
  refine ⟨x1, rfl, hx1len, ?_⟩
  intro i hi
  rw [hx1val]
  exact stageRef_denote w sigma span.val m k x.val hspan hs hmlen hfit i hi

/-- The two-entry register `[3, 5]` of the joint witness below. -/
def witnessReg : Slice I64 := Slice.from [3#i64, 5#i64] (by scalar_tac)

/-- **Joint witness for `stage_mul_denote`.** Its success-path hypotheses hold together on the
register `[3, 5]` at span `1 = 2^0`, cell `w = (2, 1)` and the hyperbolic dial, so the headline is
not vacuous: the extracted body returns `Ok ()` there and its output denotes the canonical stage. -/
theorem stage_mul_denote_witness :
    ∃ x', register.stage_mul witnessReg 1#usize ⟨2#i64, 1#i64⟩ cell.Sigma.Hyperbolic =
        ok (core.result.Result.Ok (), x') ∧
      x'.val.length = witnessReg.val.length ∧
      ∀ i, i < witnessReg.val.length →
        denote (x'.val[i]!) =
          Register.cellStage (denoteSigma cell.Sigma.Hyperbolic) (denoteCell ⟨2#i64, 1#i64⟩) 0
            (denoteArr witnessReg.val) i := by
  refine stage_mul_denote witnessReg 1#usize ⟨2#i64, 1#i64⟩ cell.Sigma.Hyperbolic 0 1
    (by simp) (by simp) (by simp [witnessReg]) ?_
  intro b j hb hj
  have hj1 : j < 1 := by simpa using hj
  obtain rfl : b = 0 := by omega
  obtain rfl : j = 0 := by omega
  simp [MulFits, witnessReg, sigmaI64, I64.min, I64.max]
  norm_num [I64.numBits]

end RegisterRefinement

end

end apeiron_word
