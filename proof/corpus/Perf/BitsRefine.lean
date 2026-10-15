import ApeironWord
import Mathlib.Tactic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Perf.Receipts.Onebit
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The executed bit-budget bodies refine their closed forms

`onebit_bits.*` (`src/learn/onebit_bits.rs`) is the checked-`u64` bit budget of the one-bit
transformer arm `...-sink-fnorm-grow-res-qmlp-nb`: `bits_sign`, `bits_r`, `bits_c`, `bits_d`,
`bits_corr`, `total_bits`, `params`, `index_bits`. Every add and multiply is checked; an overflow, or
a scale group that does not divide a tensor width, returns `none`. Nothing wraps and nothing fails.

Landed here, for all configurations `c : onebit_bits.Cfg` (no reduction over the configuration
space):

* `sign_layer_spec`, `bits_sign_spec`, `params_spec`, `bits_r_spec`, `bits_c_spec`, `bits_d_spec`,
  `bits_corr_spec`, `total_bits_spec`, `index_bits_spec`: if the explicit `ℕ` predicate `Fits…` holds,
  `f c = ok (some x)` with `x.val` equal to the closed form (`signLayerN`, `bitsSignN`, `paramsN`,
  `bitsRN`, `bitsCN`, `bitsDN`, `bitsCorrN`, `totalN`, `indexBitsN`); otherwise `f c = ok none`. The
  call is total (`ok`, never `fail`);
* `qwen_*`: at `qwenCfg` (the field values of `Cfg::qwen25_05b_nb()`) the executed bodies return
  the generated receipt constants `Receipts.t3_nb_3seed.{bitsSign, bitsR, bitsC, bitsD, bitsCorr,
  bits, params}`; `qwen_total_is_sum` states that the receipt total is the sum of its five
  components, and `qwen_fits` gives a joint `Fits` witness for every function at that cfg;
* `choose_4864_416_le`, `lt_choose_4864_416`: `2 ^ 2043 < C(4864, 416) ≤ 2 ^ 2044`, so
  `index_bits_per_set = 2044` is the minimal sufficient integer; `index_bits_qwen`: the row-index side
  information is `24 · 2 · 2044 = 98112` bits, which covers the `98105.99` bits of the `f64` estimate
  (`row_index_cover`); `headline_bpw_bounds`: `(421541376 + 98112) / 357826560` lies strictly between
  `1.1783` and `1.1784`, and the counted value without the index is `1.178061` (the receipt
  `bpwLinears` rounds to `1.17803`).

Grades. Refinement of the extracted body to the closed form: proved (Lean kernel, closed axiom
cone). Closed form to the Python estimator (`fold_transfer.py`): tested via the receipt only (the
estimator is not extracted). No packed format or decoder exists, so this counts a format that is
not implemented.

Trust boundary: rustc/LLVM codegen and the kernel ABI are not covered; the Python estimator is not
extracted; the `f64` value `98105.99` is a number read from the estimator, not computed here.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

namespace apeiron_word

open Aeneas Aeneas.Std Result

namespace BitsRefine

/-! ## Checked arithmetic on `ℕ` -/

/-- Checked `a * b` at 64 bits. -/
def nmul (a b : ℕ) : Option ℕ := if a * b < 2 ^ 64 then some (a * b) else none

/-- Checked `a + b` at 64 bits. -/
def nadd (a b : ℕ) : Option ℕ := if a + b < 2 ^ 64 then some (a + b) else none

theorem nmul_some {a b : ℕ} (h : a * b < 2 ^ 64) : nmul a b = some (a * b) := by
  unfold nmul; rw [if_pos h]

theorem nmul_none {a b : ℕ} (h : ¬ a * b < 2 ^ 64) : nmul a b = none := by
  unfold nmul; rw [if_neg h]

theorem nadd_some {a b : ℕ} (h : a + b < 2 ^ 64) : nadd a b = some (a + b) := by
  unfold nadd; rw [if_pos h]

theorem nadd_none {a b : ℕ} (h : ¬ a + b < 2 ^ 64) : nadd a b = none := by
  unfold nadd; rw [if_neg h]

/-- `r` is a total `ok` whose payload, read through `val`, is `o`. -/
def S (r : Result (Option Std.U64)) (o : Option ℕ) : Prop :=
  ∃ x : Option Std.U64, r = ok x ∧ x.map (fun z => z.val) = o

theorem mul_S (a b : Std.U64) : S (onebit_bits.mul a b) (nmul a.val b.val) := by
  refine ⟨a.checked_mul b, rfl, ?_⟩
  have h := U64.checked_mul_bv_spec a b
  rw [U64.max_eq] at h
  rcases hx : a.checked_mul b with _ | z
  · rw [hx] at h
    simp only [Option.map_none]
    exact (nmul_none (by omega)).symm
  · rw [hx] at h
    obtain ⟨h1, h2, -⟩ := h
    simp only [Option.map_some]
    rw [nmul_some (by omega), h2]

theorem add_S (a b : Std.U64) : S (onebit_bits.add a b) (nadd a.val b.val) := by
  refine ⟨a.checked_add b, rfl, ?_⟩
  have h := U64.checked_add_bv_spec a b
  rw [U64.max_eq] at h
  rcases hx : a.checked_add b with _ | z
  · rw [hx] at h
    simp only [Option.map_none]
    exact (nadd_none (by omega)).symm
  · rw [hx] at h
    obtain ⟨h1, h2, -⟩ := h
    simp only [Option.map_some]
    rw [nadd_some (by omega), h2]

/-- The extracted `?` pattern: run `m`, branch on the `Option`, continue with `k` or propagate
`none`. -/
def chain (m : Result (Option Std.U64)) (k : Std.U64 → Result (Option Std.U64)) :
    Result (Option Std.U64) := do
  let o ← m
  let cf ← core.option.Option.Insts.CoreOpsTry_traitTry.branch o
  match cf with
  | core.ops.control_flow.ControlFlow.Continue v => k v
  | core.ops.control_flow.ControlFlow.Break residual =>
    core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual
      Std.U64 residual

theorem chain_S {m : Result (Option Std.U64)} {o : Option ℕ}
    {k : Std.U64 → Result (Option Std.U64)} {g : ℕ → Option ℕ}
    (hm : S m o) (hk : ∀ z : Std.U64, S (k z) (g z.val)) : S (chain m k) (o.bind g) := by
  obtain ⟨x, hx, hxo⟩ := hm
  subst hx
  cases x with
  | none =>
    simp only [Option.map_none] at hxo
    subst hxo
    refine ⟨none, ?_, rfl⟩
    simp [chain, core.option.Option.Insts.CoreOpsTry_traitTry.branch,
      core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual]
  | some z =>
    simp only [Option.map_some] at hxo
    subst hxo
    have := hk z
    simpa [chain, core.option.Option.Insts.CoreOpsTry_traitTry.branch] using this

/-! ## Natural-number configuration -/

/-- The configuration read as natural numbers. -/
structure NCfg where
  d_model : ℕ
  kv : ℕ
  ff : ℕ
  layers : ℕ
  k_rows : ℕ
  group : ℕ
  header_bits : ℕ
  r_level_bits : ℕ
  r_mu_bits : ℕ
  r_sp_bits : ℕ
  c_bits : ℕ
  d_bits : ℕ
  corr_bits : ℕ
  rank_q : ℕ
  rank_down : ℕ
  index_sets : ℕ
  index_bits_per_set : ℕ

/-- Read an extracted configuration through `val`. -/
def toN (c : onebit_bits.Cfg) : NCfg where
  d_model := c.d_model.val
  kv := c.kv.val
  ff := c.ff.val
  layers := c.layers.val
  k_rows := c.k_rows.val
  group := c.group.val
  header_bits := c.header_bits.val
  r_level_bits := c.r_level_bits.val
  r_mu_bits := c.r_mu_bits.val
  r_sp_bits := c.r_sp_bits.val
  c_bits := c.c_bits.val
  d_bits := c.d_bits.val
  corr_bits := c.corr_bits.val
  rank_q := c.rank_q.val
  rank_down := c.rank_down.val
  index_sets := c.index_sets.val
  index_bits_per_set := c.index_bits_per_set.val

/-! ## Closed forms (`src/learn/onebit_bits.rs`) -/

/-- Sign bits of one layer: `3 d² + 4 kv d + 3 ff d + 2 k_rows d`. -/
def signLayerN (n : NCfg) : ℕ :=
  3 * (n.d_model * n.d_model) + 4 * (n.kv * n.d_model) + 3 * (n.ff * n.d_model)
    + 2 * (n.k_rows * n.d_model)

def bitsSignN (n : NCfg) : ℕ := n.layers * signLayerN n

def paramsN (n : NCfg) : ℕ :=
  n.layers * (2 * (n.d_model * n.d_model) + 2 * (n.kv * n.d_model) + 3 * (n.ff * n.d_model))

/-- `quant_r_hier` bits of an `(o, m)` call: `r_level o (m / group) + (r_mu + r_sp) o + 2 header`. -/
def callRN (n : NCfg) (o m : ℕ) : ℕ :=
  n.r_level_bits * o * (m / n.group) + (n.r_mu_bits + n.r_sp_bits) * o + 2 * n.header_bits

def bitsRN (n : NCfg) : ℕ :=
  n.layers * (3 * callRN n n.d_model n.d_model + 4 * callRN n n.kv n.d_model
    + 2 * callRN n n.ff n.d_model + callRN n n.d_model n.ff + 2 * callRN n n.k_rows n.d_model)

/-- `quant_c_u8(bits)` bits for a vector of length `m`: `bits m + header`. -/
def callVecN (n : NCfg) (bits m : ℕ) : ℕ := bits * m + n.header_bits

def vecLayerN (n : NCfg) (bits : ℕ) : ℕ :=
  11 * callVecN n bits n.d_model + callVecN n bits n.ff

def bitsCN (n : NCfg) : ℕ := n.layers * vecLayerN n n.c_bits

def bitsDN (n : NCfg) : ℕ := n.layers * vecLayerN n n.d_bits

/-- Rank-`rank` corrector on an `(o, m)` tensor: `rank (corr (o + m) + 2 header)`. -/
def corrCallN (n : NCfg) (rank o m : ℕ) : ℕ :=
  rank * (n.corr_bits * (o + m) + 2 * n.header_bits)

def bitsCorrN (n : NCfg) : ℕ :=
  n.layers * (corrCallN n n.rank_q n.d_model n.d_model + corrCallN n n.rank_down n.d_model n.ff)

def totalN (n : NCfg) : ℕ := bitsSignN n + bitsRN n + bitsCN n + bitsDN n + bitsCorrN n

def indexBitsN (n : NCfg) : ℕ := n.layers * n.index_sets * n.index_bits_per_set

/-! ## Reference computations: the closed form with every operation checked -/

def refSignLayer (n : NCfg) : Option ℕ :=
  (nmul n.d_model n.d_model).bind fun dd =>
  (nmul n.kv n.d_model).bind fun kvd =>
  (nmul n.ff n.d_model).bind fun ffd =>
  (nmul n.k_rows n.d_model).bind fun krd =>
  (nmul 3 dd).bind fun a =>
  (nmul 4 kvd).bind fun b =>
  (nadd a b).bind fun t =>
  (nmul 3 ffd).bind fun e =>
  (nadd t e).bind fun t2 =>
  (nmul 2 krd).bind fun f =>
  nadd t2 f

def refBitsSign (n : NCfg) : Option ℕ := (refSignLayer n).bind fun s => nmul n.layers s

def refParams (n : NCfg) : Option ℕ :=
  (nmul n.d_model n.d_model).bind fun dd =>
  (nmul n.kv n.d_model).bind fun kvd =>
  (nmul n.ff n.d_model).bind fun ffd =>
  (nmul 2 dd).bind fun a =>
  (nmul 2 kvd).bind fun b =>
  (nadd a b).bind fun t =>
  (nmul 3 ffd).bind fun e =>
  (nadd t e).bind fun t2 =>
  nmul n.layers t2

def refCallTail (n : NCfg) (o groups : ℕ) : Option ℕ :=
  (nmul n.r_level_bits o).bind fun a =>
  (nmul a groups).bind fun levels =>
  (nadd n.r_mu_bits n.r_sp_bits).bind fun s =>
  (nmul s o).bind fun rows =>
  (nadd levels rows).bind fun t =>
  (nmul 2 n.header_bits).bind fun h2 =>
  nadd t h2

/-- `quant_r_hier` reference: refuses (`none`) on a zero group or a group that does not divide `m`. -/
def refCallR (n : NCfg) (o m : ℕ) : Option ℕ :=
  if n.group ≠ 0 then (if m % n.group = 0 then refCallTail n o (m / n.group) else none) else none

def refBitsR (n : NCfg) : Option ℕ :=
  (refCallR n n.d_model n.d_model).bind fun a =>
  (nmul 3 a).bind fun a3 =>
  (refCallR n n.kv n.d_model).bind fun b =>
  (nmul 4 b).bind fun b4 =>
  (nadd a3 b4).bind fun t1 =>
  (refCallR n n.ff n.d_model).bind fun cc =>
  (nmul 2 cc).bind fun c2 =>
  (nadd t1 c2).bind fun t2 =>
  (refCallR n n.d_model n.ff).bind fun dd =>
  (nadd t2 dd).bind fun t3 =>
  (refCallR n n.k_rows n.d_model).bind fun ee =>
  (nmul 2 ee).bind fun e2 =>
  (nadd t3 e2).bind fun t4 =>
  nmul n.layers t4

def refCallVec (n : NCfg) (bits m : ℕ) : Option ℕ :=
  (nmul bits m).bind fun a => nadd a n.header_bits

def refVecLayer (n : NCfg) (bits : ℕ) : Option ℕ :=
  (refCallVec n bits n.d_model).bind fun a =>
  (nmul 11 a).bind fun a11 =>
  (refCallVec n bits n.ff).bind fun b =>
  nadd a11 b

def refBitsC (n : NCfg) : Option ℕ := (refVecLayer n n.c_bits).bind fun v => nmul n.layers v

def refBitsD (n : NCfg) : Option ℕ := (refVecLayer n n.d_bits).bind fun v => nmul n.layers v

def refCorrCall (n : NCfg) (rank o m : ℕ) : Option ℕ :=
  (nadd o m).bind fun s =>
  (nmul n.corr_bits s).bind fun w =>
  (nmul 2 n.header_bits).bind fun h2 =>
  (nadd w h2).bind fun t =>
  nmul rank t

def refBitsCorr (n : NCfg) : Option ℕ :=
  (refCorrCall n n.rank_q n.d_model n.d_model).bind fun q =>
  (refCorrCall n n.rank_down n.d_model n.ff).bind fun dn =>
  (nadd q dn).bind fun t =>
  nmul n.layers t

def refTotal (n : NCfg) : Option ℕ :=
  (refBitsSign n).bind fun s =>
  (refBitsR n).bind fun r =>
  (nadd s r).bind fun t1 =>
  (refBitsC n).bind fun cc =>
  (nadd t1 cc).bind fun t2 =>
  (refBitsD n).bind fun dd =>
  (nadd t2 dd).bind fun t3 =>
  (refBitsCorr n).bind fun e =>
  nadd t3 e

def refIndexBits (n : NCfg) : Option ℕ :=
  (nmul n.layers n.index_sets).bind fun a => nmul a n.index_bits_per_set

/-! ## Explicit fit predicates

`FitsX n` lists, in evaluation order, every checked intermediate that must be `< 2 ^ 64`. The
`ref…_eq` lemmas show the reference computation is the closed form exactly when the predicate holds,
and `none` otherwise. -/

theorem nmul_bind (a b : ℕ) (f : ℕ → Option ℕ) :
    (nmul a b).bind f = if a * b < 2 ^ 64 then f (a * b) else none := by
  unfold nmul; split_ifs <;> rfl

theorem nadd_bind (a b : ℕ) (f : ℕ → Option ℕ) :
    (nadd a b).bind f = if a + b < 2 ^ 64 then f (a + b) else none := by
  unfold nadd; split_ifs <;> rfl

theorem ite_bind_some (p : Prop) [Decidable p] (x : ℕ) (f : ℕ → Option ℕ) :
    (if p then some x else none).bind f = if p then f x else none := by
  split_ifs <;> rfl

theorem ite_none_and (p q : Prop) [Decidable p] [Decidable q] (x : ℕ) :
    (if p then (if q then some x else none) else none) = if p ∧ q then some x else none := by
  by_cases hp : p <;> by_cases hq : q <;> simp [hp, hq]

def FitsSignLayer (n : NCfg) : Prop :=
  n.d_model * n.d_model < 2 ^ 64 ∧ n.kv * n.d_model < 2 ^ 64 ∧ n.ff * n.d_model < 2 ^ 64 ∧
  n.k_rows * n.d_model < 2 ^ 64 ∧ 3 * (n.d_model * n.d_model) < 2 ^ 64 ∧
  4 * (n.kv * n.d_model) < 2 ^ 64 ∧
  3 * (n.d_model * n.d_model) + 4 * (n.kv * n.d_model) < 2 ^ 64 ∧
  3 * (n.ff * n.d_model) < 2 ^ 64 ∧
  3 * (n.d_model * n.d_model) + 4 * (n.kv * n.d_model) + 3 * (n.ff * n.d_model) < 2 ^ 64 ∧
  2 * (n.k_rows * n.d_model) < 2 ^ 64 ∧ signLayerN n < 2 ^ 64

instance (n : NCfg) : Decidable (FitsSignLayer n) := by unfold FitsSignLayer; infer_instance

theorem refSignLayer_eq (n : NCfg) :
    refSignLayer n = if FitsSignLayer n then some (signLayerN n) else none := by
  simp only [refSignLayer, nmul_bind, nadd_bind]
  simp only [nmul, nadd, ite_none_and, FitsSignLayer, signLayerN]

def FitsBitsSign (n : NCfg) : Prop := FitsSignLayer n ∧ n.layers * signLayerN n < 2 ^ 64

instance (n : NCfg) : Decidable (FitsBitsSign n) := by unfold FitsBitsSign; infer_instance

theorem refBitsSign_eq (n : NCfg) :
    refBitsSign n = if FitsBitsSign n then some (bitsSignN n) else none := by
  simp only [refBitsSign, refSignLayer_eq, ite_bind_some, nmul_bind]
  simp only [nmul, nadd, ite_none_and, FitsBitsSign, bitsSignN]

def FitsParams (n : NCfg) : Prop :=
  n.d_model * n.d_model < 2 ^ 64 ∧ n.kv * n.d_model < 2 ^ 64 ∧ n.ff * n.d_model < 2 ^ 64 ∧
  2 * (n.d_model * n.d_model) < 2 ^ 64 ∧ 2 * (n.kv * n.d_model) < 2 ^ 64 ∧
  2 * (n.d_model * n.d_model) + 2 * (n.kv * n.d_model) < 2 ^ 64 ∧
  3 * (n.ff * n.d_model) < 2 ^ 64 ∧
  2 * (n.d_model * n.d_model) + 2 * (n.kv * n.d_model) + 3 * (n.ff * n.d_model) < 2 ^ 64 ∧
  paramsN n < 2 ^ 64

instance (n : NCfg) : Decidable (FitsParams n) := by unfold FitsParams; infer_instance

theorem refParams_eq (n : NCfg) :
    refParams n = if FitsParams n then some (paramsN n) else none := by
  simp only [refParams, nmul_bind, nadd_bind]
  simp only [nmul, nadd, ite_none_and, FitsParams, paramsN]

def FitsCallR (n : NCfg) (o m : ℕ) : Prop :=
  n.group ≠ 0 ∧ m % n.group = 0 ∧ n.r_level_bits * o < 2 ^ 64 ∧
  n.r_level_bits * o * (m / n.group) < 2 ^ 64 ∧ n.r_mu_bits + n.r_sp_bits < 2 ^ 64 ∧
  (n.r_mu_bits + n.r_sp_bits) * o < 2 ^ 64 ∧
  n.r_level_bits * o * (m / n.group) + (n.r_mu_bits + n.r_sp_bits) * o < 2 ^ 64 ∧
  2 * n.header_bits < 2 ^ 64 ∧ callRN n o m < 2 ^ 64

instance (n : NCfg) (o m : ℕ) : Decidable (FitsCallR n o m) := by
  unfold FitsCallR; infer_instance

theorem refCallR_eq (n : NCfg) (o m : ℕ) :
    refCallR n o m = if FitsCallR n o m then some (callRN n o m) else none := by
  simp only [refCallR, refCallTail, nmul_bind, nadd_bind]
  simp only [nmul, nadd, ite_none_and, FitsCallR, callRN]

def FitsBitsR (n : NCfg) : Prop :=
  FitsCallR n n.d_model n.d_model ∧ 3 * callRN n n.d_model n.d_model < 2 ^ 64 ∧
  FitsCallR n n.kv n.d_model ∧ 4 * callRN n n.kv n.d_model < 2 ^ 64 ∧
  3 * callRN n n.d_model n.d_model + 4 * callRN n n.kv n.d_model < 2 ^ 64 ∧
  FitsCallR n n.ff n.d_model ∧ 2 * callRN n n.ff n.d_model < 2 ^ 64 ∧
  3 * callRN n n.d_model n.d_model + 4 * callRN n n.kv n.d_model
    + 2 * callRN n n.ff n.d_model < 2 ^ 64 ∧
  FitsCallR n n.d_model n.ff ∧
  3 * callRN n n.d_model n.d_model + 4 * callRN n n.kv n.d_model
    + 2 * callRN n n.ff n.d_model + callRN n n.d_model n.ff < 2 ^ 64 ∧
  FitsCallR n n.k_rows n.d_model ∧ 2 * callRN n n.k_rows n.d_model < 2 ^ 64 ∧
  3 * callRN n n.d_model n.d_model + 4 * callRN n n.kv n.d_model
    + 2 * callRN n n.ff n.d_model + callRN n n.d_model n.ff
    + 2 * callRN n n.k_rows n.d_model < 2 ^ 64 ∧ bitsRN n < 2 ^ 64

instance (n : NCfg) : Decidable (FitsBitsR n) := by unfold FitsBitsR; infer_instance

theorem refBitsR_eq (n : NCfg) :
    refBitsR n = if FitsBitsR n then some (bitsRN n) else none := by
  simp only [refBitsR, refCallR_eq, ite_bind_some, nmul_bind, nadd_bind]
  simp only [nmul, nadd, ite_none_and, FitsBitsR, bitsRN]

def FitsCallVec (n : NCfg) (bits m : ℕ) : Prop :=
  bits * m < 2 ^ 64 ∧ callVecN n bits m < 2 ^ 64

instance (n : NCfg) (bits m : ℕ) : Decidable (FitsCallVec n bits m) := by
  unfold FitsCallVec; infer_instance

theorem refCallVec_eq (n : NCfg) (bits m : ℕ) :
    refCallVec n bits m = if FitsCallVec n bits m then some (callVecN n bits m) else none := by
  simp only [refCallVec, nmul_bind]
  simp only [nmul, nadd, ite_none_and, FitsCallVec, callVecN]

def FitsVecLayer (n : NCfg) (bits : ℕ) : Prop :=
  FitsCallVec n bits n.d_model ∧ 11 * callVecN n bits n.d_model < 2 ^ 64 ∧
  FitsCallVec n bits n.ff ∧ vecLayerN n bits < 2 ^ 64

instance (n : NCfg) (bits : ℕ) : Decidable (FitsVecLayer n bits) := by
  unfold FitsVecLayer; infer_instance

theorem refVecLayer_eq (n : NCfg) (bits : ℕ) :
    refVecLayer n bits = if FitsVecLayer n bits then some (vecLayerN n bits) else none := by
  simp only [refVecLayer, refCallVec_eq, ite_bind_some, nmul_bind]
  simp only [nmul, nadd, ite_none_and, FitsVecLayer, vecLayerN]

def FitsBitsC (n : NCfg) : Prop :=
  FitsVecLayer n n.c_bits ∧ n.layers * vecLayerN n n.c_bits < 2 ^ 64

instance (n : NCfg) : Decidable (FitsBitsC n) := by unfold FitsBitsC; infer_instance

theorem refBitsC_eq (n : NCfg) :
    refBitsC n = if FitsBitsC n then some (bitsCN n) else none := by
  simp only [refBitsC, refVecLayer_eq, ite_bind_some, nmul_bind]
  simp only [nmul, nadd, ite_none_and, FitsBitsC, bitsCN]

def FitsBitsD (n : NCfg) : Prop :=
  FitsVecLayer n n.d_bits ∧ n.layers * vecLayerN n n.d_bits < 2 ^ 64

instance (n : NCfg) : Decidable (FitsBitsD n) := by unfold FitsBitsD; infer_instance

theorem refBitsD_eq (n : NCfg) :
    refBitsD n = if FitsBitsD n then some (bitsDN n) else none := by
  simp only [refBitsD, refVecLayer_eq, ite_bind_some, nmul_bind]
  simp only [nmul, nadd, ite_none_and, FitsBitsD, bitsDN]

def FitsCorrCall (n : NCfg) (rank o m : ℕ) : Prop :=
  o + m < 2 ^ 64 ∧ n.corr_bits * (o + m) < 2 ^ 64 ∧ 2 * n.header_bits < 2 ^ 64 ∧
  n.corr_bits * (o + m) + 2 * n.header_bits < 2 ^ 64 ∧ corrCallN n rank o m < 2 ^ 64

instance (n : NCfg) (rank o m : ℕ) : Decidable (FitsCorrCall n rank o m) := by
  unfold FitsCorrCall; infer_instance

theorem refCorrCall_eq (n : NCfg) (rank o m : ℕ) :
    refCorrCall n rank o m = if FitsCorrCall n rank o m then some (corrCallN n rank o m) else none := by
  simp only [refCorrCall, nmul_bind, nadd_bind]
  simp only [nmul, nadd, ite_none_and, FitsCorrCall, corrCallN]

def FitsBitsCorr (n : NCfg) : Prop :=
  FitsCorrCall n n.rank_q n.d_model n.d_model ∧ FitsCorrCall n n.rank_down n.d_model n.ff ∧
  corrCallN n n.rank_q n.d_model n.d_model + corrCallN n n.rank_down n.d_model n.ff < 2 ^ 64 ∧
  bitsCorrN n < 2 ^ 64

instance (n : NCfg) : Decidable (FitsBitsCorr n) := by unfold FitsBitsCorr; infer_instance

theorem refBitsCorr_eq (n : NCfg) :
    refBitsCorr n = if FitsBitsCorr n then some (bitsCorrN n) else none := by
  simp only [refBitsCorr, refCorrCall_eq, ite_bind_some, nmul_bind, nadd_bind]
  simp only [nmul, nadd, ite_none_and, FitsBitsCorr, bitsCorrN]

def FitsTotal (n : NCfg) : Prop :=
  FitsBitsSign n ∧ FitsBitsR n ∧ bitsSignN n + bitsRN n < 2 ^ 64 ∧ FitsBitsC n ∧
  bitsSignN n + bitsRN n + bitsCN n < 2 ^ 64 ∧ FitsBitsD n ∧
  bitsSignN n + bitsRN n + bitsCN n + bitsDN n < 2 ^ 64 ∧ FitsBitsCorr n ∧ totalN n < 2 ^ 64

instance (n : NCfg) : Decidable (FitsTotal n) := by unfold FitsTotal; infer_instance

theorem refTotal_eq (n : NCfg) :
    refTotal n = if FitsTotal n then some (totalN n) else none := by
  simp only [refTotal, refBitsSign_eq, refBitsR_eq, refBitsC_eq, refBitsD_eq, refBitsCorr_eq,
    ite_bind_some, nadd_bind]
  simp only [nmul, nadd, ite_none_and, FitsTotal, totalN]

def FitsIndexBits (n : NCfg) : Prop :=
  n.layers * n.index_sets < 2 ^ 64 ∧ indexBitsN n < 2 ^ 64

instance (n : NCfg) : Decidable (FitsIndexBits n) := by unfold FitsIndexBits; infer_instance

theorem refIndexBits_eq (n : NCfg) :
    refIndexBits n = if FitsIndexBits n then some (indexBitsN n) else none := by
  simp only [refIndexBits, nmul_bind]
  simp only [nmul, nadd, ite_none_and, FitsIndexBits, indexBitsN]

/-! ## Executed bodies refine the reference computations -/

section
open core.option.Option.Insts

theorem sign_layer_S (c : onebit_bits.Cfg) : S (onebit_bits.sign_layer c) (refSignLayer (toN c)) := by
  have e : onebit_bits.sign_layer c =
      chain (onebit_bits.mul c.d_model c.d_model) fun dd =>
      chain (onebit_bits.mul c.kv c.d_model) fun kvd =>
      chain (onebit_bits.mul c.ff c.d_model) fun ffd =>
      chain (onebit_bits.mul c.k_rows c.d_model) fun krd =>
      chain (onebit_bits.mul 3#u64 dd) fun a =>
      chain (onebit_bits.mul 4#u64 kvd) fun b =>
      chain (onebit_bits.add a b) fun t =>
      chain (onebit_bits.mul 3#u64 ffd) fun e =>
      chain (onebit_bits.add t e) fun t2 =>
      chain (onebit_bits.mul 2#u64 krd) fun f =>
      onebit_bits.add t2 f := rfl
  rw [e]
  simp only [refSignLayer, toN]
  refine chain_S (mul_S _ _) fun dd => ?_
  refine chain_S (mul_S _ _) fun kvd => ?_
  refine chain_S (mul_S _ _) fun ffd => ?_
  refine chain_S (mul_S _ _) fun krd => ?_
  refine chain_S (mul_S _ _) fun a => ?_
  refine chain_S (mul_S _ _) fun b => ?_
  refine chain_S (add_S _ _) fun t => ?_
  refine chain_S (mul_S _ _) fun e => ?_
  refine chain_S (add_S _ _) fun t2 => ?_
  refine chain_S (mul_S _ _) fun f => ?_
  exact add_S _ _

theorem bits_sign_S (c : onebit_bits.Cfg) : S (onebit_bits.bits_sign c) (refBitsSign (toN c)) := by
  have e : onebit_bits.bits_sign c =
      chain (onebit_bits.sign_layer c) fun s => onebit_bits.mul c.layers s := rfl
  rw [e]
  exact chain_S (sign_layer_S c) fun s => mul_S _ _

theorem params_S (c : onebit_bits.Cfg) : S (onebit_bits.params c) (refParams (toN c)) := by
  have e : onebit_bits.params c =
      chain (onebit_bits.mul c.d_model c.d_model) fun dd =>
      chain (onebit_bits.mul c.kv c.d_model) fun kvd =>
      chain (onebit_bits.mul c.ff c.d_model) fun ffd =>
      chain (onebit_bits.mul 2#u64 dd) fun a =>
      chain (onebit_bits.mul 2#u64 kvd) fun b =>
      chain (onebit_bits.add a b) fun t =>
      chain (onebit_bits.mul 3#u64 ffd) fun e =>
      chain (onebit_bits.add t e) fun t2 =>
      onebit_bits.mul c.layers t2 := rfl
  rw [e]
  simp only [refParams, toN]
  refine chain_S (mul_S _ _) fun dd => ?_
  refine chain_S (mul_S _ _) fun kvd => ?_
  refine chain_S (mul_S _ _) fun ffd => ?_
  refine chain_S (mul_S _ _) fun a => ?_
  refine chain_S (mul_S _ _) fun b => ?_
  refine chain_S (add_S _ _) fun t => ?_
  refine chain_S (mul_S _ _) fun e => ?_
  refine chain_S (add_S _ _) fun t2 => ?_
  exact mul_S _ _

/-- The `call_r` body after the group checks. -/
def callRTail (c : onebit_bits.Cfg) (o groups : Std.U64) : Result (Option Std.U64) :=
  chain (onebit_bits.mul c.r_level_bits o) fun val =>
  chain (onebit_bits.mul val groups) fun val1 =>
  chain (onebit_bits.add c.r_mu_bits c.r_sp_bits) fun val2 =>
  chain (onebit_bits.mul val2 o) fun val3 =>
  chain (onebit_bits.add val1 val3) fun val4 =>
  chain (onebit_bits.mul 2#u64 c.header_bits) fun val5 =>
  onebit_bits.add val4 val5

theorem callRTail_S (c : onebit_bits.Cfg) (o groups : Std.U64) :
    S (callRTail c o groups) (refCallTail (toN c) o.val groups.val) := by
  unfold callRTail
  simp only [refCallTail, toN]
  refine chain_S (mul_S _ _) fun val => ?_
  refine chain_S (mul_S _ _) fun val1 => ?_
  refine chain_S (add_S _ _) fun val2 => ?_
  refine chain_S (mul_S _ _) fun val3 => ?_
  refine chain_S (add_S _ _) fun val4 => ?_
  refine chain_S (mul_S _ _) fun val5 => ?_
  exact add_S _ _

theorem call_r_S (o n : Std.U64) (c : onebit_bits.Cfg) :
    S (onebit_bits.call_r o n c) (refCallR (toN c) o.val n.val) := by
  unfold onebit_bits.call_r
  by_cases hg : c.group = 0#u64
  · rw [if_pos hg]
    have hgv : c.group.val = 0 := by rw [hg]; rfl
    refine ⟨none, rfl, ?_⟩
    simp [refCallR, toN, hgv]
  · rw [if_neg hg]
    have hgv : c.group.val ≠ 0 := fun h => hg (by rw [UScalar.eq_equiv]; simpa using h)
    obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (UScalar.rem_spec n hgv)
    rw [hi]
    simp only [bind_tc_ok]
    by_cases hi0 : i = 0#u64
    · have hb : (i != 0#u64) = false := by simp [hi0]
      have hmod : n.val % c.group.val = 0 := by
        rw [← hiv]; rw [hi0]; rfl
      rw [hb]
      simp only [Bool.false_eq_true, ↓reduceIte]
      obtain ⟨gr, hgr, hgrv⟩ := UScalar.div_spec n hgv
      rw [hgr]
      simp only [bind_tc_ok]
      have key : refCallR (toN c) o.val n.val = refCallTail (toN c) o.val gr.val := by
        simp only [refCallR, toN]
        rw [if_pos hgv, if_pos hmod, hgrv]
      rw [key]
      exact callRTail_S c o gr
    · have hb : (i != 0#u64) = true := by simpa using hi0
      have hmod : ¬ n.val % c.group.val = 0 := by
        intro h
        apply hi0
        rw [UScalar.eq_equiv, hiv]
        simpa using h
      rw [hb]
      simp only [↓reduceIte]
      refine ⟨none, rfl, ?_⟩
      simp only [refCallR, toN, Option.map_none]
      rw [if_pos hgv, if_neg hmod]

theorem call_vec_S (bits n : Std.U64) (c : onebit_bits.Cfg) :
    S (onebit_bits.call_vec bits n c) (refCallVec (toN c) bits.val n.val) := by
  have e : onebit_bits.call_vec bits n c =
      chain (onebit_bits.mul bits n) fun a => onebit_bits.add a c.header_bits := rfl
  rw [e]
  simp only [refCallVec, toN]
  exact chain_S (mul_S _ _) fun a => add_S _ _

theorem vec_layer_S (bits : Std.U64) (c : onebit_bits.Cfg) :
    S (onebit_bits.vec_layer bits c) (refVecLayer (toN c) bits.val) := by
  have e : onebit_bits.vec_layer bits c =
      chain (onebit_bits.call_vec bits c.d_model c) fun a =>
      chain (onebit_bits.mul 11#u64 a) fun a11 =>
      chain (onebit_bits.call_vec bits c.ff c) fun b =>
      onebit_bits.add a11 b := rfl
  rw [e]
  simp only [refVecLayer]
  refine chain_S (call_vec_S _ _ _) fun a => ?_
  refine chain_S (mul_S _ _) fun a11 => ?_
  refine chain_S (call_vec_S _ _ _) fun b => ?_
  exact add_S _ _

theorem bits_c_S (c : onebit_bits.Cfg) : S (onebit_bits.bits_c c) (refBitsC (toN c)) := by
  have e : onebit_bits.bits_c c =
      chain (onebit_bits.vec_layer c.c_bits c) fun v => onebit_bits.mul c.layers v := rfl
  rw [e]
  exact chain_S (vec_layer_S _ _) fun v => mul_S _ _

theorem bits_d_S (c : onebit_bits.Cfg) : S (onebit_bits.bits_d c) (refBitsD (toN c)) := by
  have e : onebit_bits.bits_d c =
      chain (onebit_bits.vec_layer c.d_bits c) fun v => onebit_bits.mul c.layers v := rfl
  rw [e]
  exact chain_S (vec_layer_S _ _) fun v => mul_S _ _

theorem corr_call_S (rank o n : Std.U64) (c : onebit_bits.Cfg) :
    S (onebit_bits.corr_call rank o n c) (refCorrCall (toN c) rank.val o.val n.val) := by
  have e : onebit_bits.corr_call rank o n c =
      chain (onebit_bits.add o n) fun s =>
      chain (onebit_bits.mul c.corr_bits s) fun w =>
      chain (onebit_bits.mul 2#u64 c.header_bits) fun h2 =>
      chain (onebit_bits.add w h2) fun t =>
      onebit_bits.mul rank t := rfl
  rw [e]
  simp only [refCorrCall, toN]
  refine chain_S (add_S _ _) fun s => ?_
  refine chain_S (mul_S _ _) fun w => ?_
  refine chain_S (mul_S _ _) fun h2 => ?_
  refine chain_S (add_S _ _) fun t => ?_
  exact mul_S _ _

theorem bits_corr_S (c : onebit_bits.Cfg) : S (onebit_bits.bits_corr c) (refBitsCorr (toN c)) := by
  have e : onebit_bits.bits_corr c =
      chain (onebit_bits.corr_call c.rank_q c.d_model c.d_model c) fun q =>
      chain (onebit_bits.corr_call c.rank_down c.d_model c.ff c) fun dn =>
      chain (onebit_bits.add q dn) fun t =>
      onebit_bits.mul c.layers t := rfl
  rw [e]
  simp only [refBitsCorr]
  refine chain_S (corr_call_S _ _ _ _) fun q => ?_
  refine chain_S (corr_call_S _ _ _ _) fun dn => ?_
  refine chain_S (add_S _ _) fun t => ?_
  exact mul_S _ _

theorem bits_r_S (c : onebit_bits.Cfg) : S (onebit_bits.bits_r c) (refBitsR (toN c)) := by
  have e : onebit_bits.bits_r c =
      chain (onebit_bits.call_r c.d_model c.d_model c) fun a =>
      chain (onebit_bits.mul 3#u64 a) fun a3 =>
      chain (onebit_bits.call_r c.kv c.d_model c) fun b =>
      chain (onebit_bits.mul 4#u64 b) fun b4 =>
      chain (onebit_bits.add a3 b4) fun t1 =>
      chain (onebit_bits.call_r c.ff c.d_model c) fun cc =>
      chain (onebit_bits.mul 2#u64 cc) fun c2 =>
      chain (onebit_bits.add t1 c2) fun t2 =>
      chain (onebit_bits.call_r c.d_model c.ff c) fun dd =>
      chain (onebit_bits.add t2 dd) fun t3 =>
      chain (onebit_bits.call_r c.k_rows c.d_model c) fun ee =>
      chain (onebit_bits.mul 2#u64 ee) fun e2 =>
      chain (onebit_bits.add t3 e2) fun t4 =>
      onebit_bits.mul c.layers t4 := rfl
  rw [e]
  simp only [refBitsR]
  refine chain_S (call_r_S _ _ _) fun a => ?_
  refine chain_S (mul_S _ _) fun a3 => ?_
  refine chain_S (call_r_S _ _ _) fun b => ?_
  refine chain_S (mul_S _ _) fun b4 => ?_
  refine chain_S (add_S _ _) fun t1 => ?_
  refine chain_S (call_r_S _ _ _) fun cc => ?_
  refine chain_S (mul_S _ _) fun c2 => ?_
  refine chain_S (add_S _ _) fun t2 => ?_
  refine chain_S (call_r_S _ _ _) fun dd => ?_
  refine chain_S (add_S _ _) fun t3 => ?_
  refine chain_S (call_r_S _ _ _) fun ee => ?_
  refine chain_S (mul_S _ _) fun e2 => ?_
  refine chain_S (add_S _ _) fun t4 => ?_
  exact mul_S _ _

theorem total_bits_S (c : onebit_bits.Cfg) : S (onebit_bits.total_bits c) (refTotal (toN c)) := by
  have e : onebit_bits.total_bits c =
      chain (onebit_bits.bits_sign c) fun s =>
      chain (onebit_bits.bits_r c) fun r =>
      chain (onebit_bits.add s r) fun t1 =>
      chain (onebit_bits.bits_c c) fun cc =>
      chain (onebit_bits.add t1 cc) fun t2 =>
      chain (onebit_bits.bits_d c) fun dd =>
      chain (onebit_bits.add t2 dd) fun t3 =>
      chain (onebit_bits.bits_corr c) fun ee =>
      onebit_bits.add t3 ee := rfl
  rw [e]
  simp only [refTotal]
  refine chain_S (bits_sign_S _) fun s => ?_
  refine chain_S (bits_r_S _) fun r => ?_
  refine chain_S (add_S _ _) fun t1 => ?_
  refine chain_S (bits_c_S _) fun cc => ?_
  refine chain_S (add_S _ _) fun t2 => ?_
  refine chain_S (bits_d_S _) fun dd => ?_
  refine chain_S (add_S _ _) fun t3 => ?_
  refine chain_S (bits_corr_S _) fun ee => ?_
  exact add_S _ _

theorem index_bits_S (c : onebit_bits.Cfg) :
    S (onebit_bits.index_bits c) (refIndexBits (toN c)) := by
  have e : onebit_bits.index_bits c =
      chain (onebit_bits.mul c.layers c.index_sets) fun a =>
      onebit_bits.mul a c.index_bits_per_set := rfl
  rw [e]
  simp only [refIndexBits, toN]
  exact chain_S (mul_S _ _) fun a => mul_S _ _

end

/-! ## Total refinement theorems, for all configurations -/

theorem spec_of_S {r : Result (Option Std.U64)} {F : Prop} [Decidable F] {v : ℕ}
    (h : S r (if F then some v else none)) :
    (F → ∃ x : Std.U64, r = ok (some x) ∧ x.val = v) ∧ (¬ F → r = ok none) := by
  obtain ⟨x, hr, hx⟩ := h
  by_cases hF : F
  · rw [if_pos hF] at hx
    cases x with
    | none => simp at hx
    | some z =>
      simp only [Option.map_some, Option.some.injEq] at hx
      exact ⟨fun _ => ⟨z, hr, hx⟩, fun h => absurd hF h⟩
  · rw [if_neg hF] at hx
    cases x with
    | none => exact ⟨fun h => absurd h hF, fun _ => hr⟩
    | some z => simp at hx

/-- `sign_layer` is total; it returns `signLayerN` exactly when `FitsSignLayer` holds. -/
theorem sign_layer_spec (c : onebit_bits.Cfg) :
    (FitsSignLayer (toN c) → ∃ x : Std.U64, onebit_bits.sign_layer c = ok (some x) ∧
      x.val = signLayerN (toN c)) ∧
    (¬ FitsSignLayer (toN c) → onebit_bits.sign_layer c = ok none) := by
  have h := sign_layer_S c
  rw [refSignLayer_eq] at h
  exact spec_of_S h

theorem bits_sign_spec (c : onebit_bits.Cfg) :
    (FitsBitsSign (toN c) → ∃ x : Std.U64, onebit_bits.bits_sign c = ok (some x) ∧
      x.val = bitsSignN (toN c)) ∧
    (¬ FitsBitsSign (toN c) → onebit_bits.bits_sign c = ok none) := by
  have h := bits_sign_S c
  rw [refBitsSign_eq] at h
  exact spec_of_S h

theorem params_spec (c : onebit_bits.Cfg) :
    (FitsParams (toN c) → ∃ x : Std.U64, onebit_bits.params c = ok (some x) ∧
      x.val = paramsN (toN c)) ∧
    (¬ FitsParams (toN c) → onebit_bits.params c = ok none) := by
  have h := params_S c
  rw [refParams_eq] at h
  exact spec_of_S h

theorem bits_r_spec (c : onebit_bits.Cfg) :
    (FitsBitsR (toN c) → ∃ x : Std.U64, onebit_bits.bits_r c = ok (some x) ∧
      x.val = bitsRN (toN c)) ∧
    (¬ FitsBitsR (toN c) → onebit_bits.bits_r c = ok none) := by
  have h := bits_r_S c
  rw [refBitsR_eq] at h
  exact spec_of_S h

theorem bits_c_spec (c : onebit_bits.Cfg) :
    (FitsBitsC (toN c) → ∃ x : Std.U64, onebit_bits.bits_c c = ok (some x) ∧
      x.val = bitsCN (toN c)) ∧
    (¬ FitsBitsC (toN c) → onebit_bits.bits_c c = ok none) := by
  have h := bits_c_S c
  rw [refBitsC_eq] at h
  exact spec_of_S h

theorem bits_d_spec (c : onebit_bits.Cfg) :
    (FitsBitsD (toN c) → ∃ x : Std.U64, onebit_bits.bits_d c = ok (some x) ∧
      x.val = bitsDN (toN c)) ∧
    (¬ FitsBitsD (toN c) → onebit_bits.bits_d c = ok none) := by
  have h := bits_d_S c
  rw [refBitsD_eq] at h
  exact spec_of_S h

theorem bits_corr_spec (c : onebit_bits.Cfg) :
    (FitsBitsCorr (toN c) → ∃ x : Std.U64, onebit_bits.bits_corr c = ok (some x) ∧
      x.val = bitsCorrN (toN c)) ∧
    (¬ FitsBitsCorr (toN c) → onebit_bits.bits_corr c = ok none) := by
  have h := bits_corr_S c
  rw [refBitsCorr_eq] at h
  exact spec_of_S h

theorem total_bits_spec (c : onebit_bits.Cfg) :
    (FitsTotal (toN c) → ∃ x : Std.U64, onebit_bits.total_bits c = ok (some x) ∧
      x.val = totalN (toN c)) ∧
    (¬ FitsTotal (toN c) → onebit_bits.total_bits c = ok none) := by
  have h := total_bits_S c
  rw [refTotal_eq] at h
  exact spec_of_S h

theorem index_bits_spec (c : onebit_bits.Cfg) :
    (FitsIndexBits (toN c) → ∃ x : Std.U64, onebit_bits.index_bits c = ok (some x) ∧
      x.val = indexBitsN (toN c)) ∧
    (¬ FitsIndexBits (toN c) → onebit_bits.index_bits c = ok none) := by
  have h := index_bits_S c
  rw [refIndexBits_eq] at h
  exact spec_of_S h

/-! ## The Qwen2.5-0.5B configuration -/

/-- The field values of `Cfg::qwen25_05b_nb()` in `src/learn/onebit_bits.rs`. -/
def qwenCfg : onebit_bits.Cfg where
  d_model := 896#u64
  kv := 128#u64
  ff := 4864#u64
  layers := 24#u64
  k_rows := 416#u64
  group := 128#u64
  header_bits := 32#u64
  r_level_bits := 4#u64
  r_mu_bits := 6#u64
  r_sp_bits := 2#u64
  c_bits := 6#u64
  d_bits := 5#u64
  corr_bits := 8#u64
  rank_q := 2#u64
  rank_down := 1#u64
  index_sets := 2#u64
  index_bits_per_set := 2044#u64

/-- Joint `Fits` witness at the Qwen cfg: every checked intermediate of every function is below
`2 ^ 64` (closed `ℕ` evaluation at one configuration, no reduction over the configuration space). -/
theorem qwen_fits :
    FitsTotal (toN qwenCfg) ∧ FitsParams (toN qwenCfg) ∧ FitsIndexBits (toN qwenCfg) := by
  decide +kernel

theorem qwen_bits_sign :
    ∃ x : Std.U64, onebit_bits.bits_sign qwenCfg = ok (some x) ∧
      x.val = Receipts.t3_nb_3seed.bitsSign := by
  obtain ⟨x, hx, hv⟩ := (bits_sign_spec qwenCfg).1 qwen_fits.1.1
  refine ⟨x, hx, ?_⟩
  rw [hv]
  decide +kernel

theorem qwen_bits_r :
    ∃ x : Std.U64, onebit_bits.bits_r qwenCfg = ok (some x) ∧
      x.val = Receipts.t3_nb_3seed.bitsR := by
  obtain ⟨x, hx, hv⟩ := (bits_r_spec qwenCfg).1 qwen_fits.1.2.1
  refine ⟨x, hx, ?_⟩
  rw [hv]
  decide +kernel

theorem qwen_bits_c :
    ∃ x : Std.U64, onebit_bits.bits_c qwenCfg = ok (some x) ∧
      x.val = Receipts.t3_nb_3seed.bitsC := by
  obtain ⟨x, hx, hv⟩ := (bits_c_spec qwenCfg).1 qwen_fits.1.2.2.2.1
  refine ⟨x, hx, ?_⟩
  rw [hv]
  decide +kernel

theorem qwen_bits_d :
    ∃ x : Std.U64, onebit_bits.bits_d qwenCfg = ok (some x) ∧
      x.val = Receipts.t3_nb_3seed.bitsD := by
  obtain ⟨x, hx, hv⟩ := (bits_d_spec qwenCfg).1 qwen_fits.1.2.2.2.2.2.1
  refine ⟨x, hx, ?_⟩
  rw [hv]
  decide +kernel

theorem qwen_bits_corr :
    ∃ x : Std.U64, onebit_bits.bits_corr qwenCfg = ok (some x) ∧
      x.val = Receipts.t3_nb_3seed.bitsCorr := by
  obtain ⟨x, hx, hv⟩ := (bits_corr_spec qwenCfg).1 qwen_fits.1.2.2.2.2.2.2.2.1
  refine ⟨x, hx, ?_⟩
  rw [hv]
  decide +kernel

/-- The receipt total is the sum of its five components. -/
theorem qwen_total_is_sum :
    Receipts.t3_nb_3seed.bits = Receipts.t3_nb_3seed.bitsSign + Receipts.t3_nb_3seed.bitsR +
      Receipts.t3_nb_3seed.bitsC + Receipts.t3_nb_3seed.bitsD + Receipts.t3_nb_3seed.bitsCorr :=
  Receipts.t3_nb_3seed.bitsComponents_eq

theorem qwen_total_bits : onebit_bits.total_bits qwenCfg = ok (some 421541376#u64) := by
  obtain ⟨x, hx, hv⟩ := (total_bits_spec qwenCfg).1 qwen_fits.1
  have h : x = 421541376#u64 := by
    rw [UScalar.eq_equiv, hv]
    decide +kernel
  rw [hx, h]

theorem qwen_total_bits_receipt :
    ∃ x : Std.U64, onebit_bits.total_bits qwenCfg = ok (some x) ∧
      x.val = Receipts.t3_nb_3seed.bits := by
  refine ⟨421541376#u64, qwen_total_bits, ?_⟩
  decide +kernel

theorem qwen_params : onebit_bits.params qwenCfg = ok (some 357826560#u64) := by
  obtain ⟨x, hx, hv⟩ := (params_spec qwenCfg).1 qwen_fits.2.1
  have h : x = 357826560#u64 := by
    rw [UScalar.eq_equiv, hv]
    decide +kernel
  rw [hx, h]

theorem qwen_params_receipt :
    ∃ x : Std.U64, onebit_bits.params qwenCfg = ok (some x) ∧
      x.val = Receipts.t3_nb_3seed.params := by
  refine ⟨357826560#u64, qwen_params, ?_⟩
  decide +kernel

/-! ## The row-index term -/

/-- `⌈log₂ C(4864, 416)⌉`-side bound: `C(4864, 416) ≤ 2 ^ 2044`, through
`416! · C(4864, 416) = 4864^{\underline{416}}` and kernel `ℕ` arithmetic on the falling factorial
(not on `Nat.choose`, whose Pascal recursion does not terminate in budget). -/
theorem choose_4864_416_le : Nat.choose 4864 416 ≤ 2 ^ 2044 := by
  have h : Nat.descFactorial 4864 416 ≤ 2 ^ 2044 * Nat.factorial 416 := by decide +kernel
  rw [Nat.descFactorial_eq_factorial_mul_choose, mul_comm (2 ^ 2044)] at h
  exact Nat.le_of_mul_le_mul_left h (Nat.factorial_pos 416)

theorem lt_choose_4864_416 : 2 ^ 2043 < Nat.choose 4864 416 := by
  have h : 2 ^ 2043 * Nat.factorial 416 < Nat.descFactorial 4864 416 := by decide +kernel
  rw [Nat.descFactorial_eq_factorial_mul_choose, mul_comm (2 ^ 2043)] at h
  exact Nat.lt_of_mul_lt_mul_left h

/-- `index_bits_per_set = 2044` is the minimal sufficient integer for one row set. -/
theorem index_bits_per_set_minimal :
    Nat.choose 4864 416 ≤ 2 ^ 2044 ∧ 2 ^ 2043 < Nat.choose 4864 416 :=
  ⟨choose_4864_416_le, lt_choose_4864_416⟩

theorem index_bits_qwen : onebit_bits.index_bits qwenCfg = ok (some 98112#u64) := by
  obtain ⟨x, hx, hv⟩ := (index_bits_spec qwenCfg).1 qwen_fits.2.2
  have h : x = 98112#u64 := by
    rw [UScalar.eq_equiv, hv]
    decide +kernel
  rw [hx, h]

/-- The `98112` row-index bits cover the `f64` estimate `98105.99` (read from the estimator). -/
theorem row_index_cover : (98105.99 : ℚ) ≤ 48 * 2044 := by norm_num

/-- Corrected headline accounting: counted bits plus the row index, over the parameter count. -/
theorem headline_bpw_bounds :
    (11783 : ℚ) / 10000 < ((421541376 + 98112 : ℕ) : ℚ) / 357826560 ∧
    ((421541376 + 98112 : ℕ) : ℚ) / 357826560 < (11784 : ℚ) / 10000 := by
  constructor <;> norm_num

/-- Without the index term the counted value is `1.178061` bits per weight. -/
theorem counted_bpw_bounds :
    (1178 : ℚ) / 1000 < (421541376 : ℚ) / 357826560 ∧
    (421541376 : ℚ) / 357826560 < (11781 : ℚ) / 10000 := by
  constructor <;> norm_num

/-- Receipt-level statement of the same accounting through the generated constants. -/
theorem headline_bpw_receipt :
    (11783 : ℚ) / 10000 <
      ((Receipts.t3_nb_3seed.bits + 98112 : ℕ) : ℚ) / Receipts.t3_nb_3seed.params ∧
    ((Receipts.t3_nb_3seed.bits + 98112 : ℕ) : ℚ) / Receipts.t3_nb_3seed.params <
      (11784 : ℚ) / 10000 := by
  unfold Receipts.t3_nb_3seed.bits Receipts.t3_nb_3seed.params
  constructor <;> norm_num


end BitsRefine

end apeiron_word
