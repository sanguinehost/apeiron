import FixedExpSource
import Aeneas

/-
SCOPE: resource theorems for the actual extracted `src/numeric/fixed_exp.rs` body — the
range-reduction guards, the degree-7 series rounds, the `2^k` scaling transitions with their
refusal and flush branches, the work measure (a fixed round count with a strictly decreasing rank),
and a reachable success witness.
REGISTER: scalar work count under named premises; no hardware timing, no cycle count, no canonical
ring correspondence, no accuracy claim.
CORPUS: `proof/resource/FixedExpSource.lean`, generated from the real `src/numeric/fixed_exp.rs`
by `proof/resource/FixedExp_gate.py` (extraction root `crate::fixed_exp_forward`, the same real
body `proof/regen.sh:181` extracts under `crate::word_fixed_exp`).
REFERENCE: /home/socol/Workspace/apeiron/src/numeric/fixed_exp.rs (read: `exp_reduce`,
`exp_horner`, `exp_scale`, `fixed_exp`),
/home/socol/Workspace/apeiron/proof/extract/lib.rs:686 (the forwarding wrapper).
GATE: python3 proof/resource/FixedExp_gate.py --check --controls, 540-second task budget, lane
/tmp/opencode/apeiron-lane-3.lock, MemoryMax=12G, MemorySwapMax=0, LEAN_NUM_THREADS=1.
GRADE: Lean kernel check only, under the closed cone audited by the `#resource_audit` lines
below; no hardware measurement.
-/

namespace FixedExpResource
open Aeneas Aeneas.Std Result

/-! ## Endpoint constants of the extracted body

`ONE`, `LOWER`, `UPPER`, `LN2_Q32`, `LN2_RECIP_Q32` and the rounding half are `irreducible`
globals of the extraction, so the guards they drive are only meaningful once they are evaluated.
These are the evaluations, from the extracted definitions. -/

/-- The extracted reciprocal constant `round(2^32 / ln 2)`. -/
theorem ln2_recip_val : fixed_exp_source.fixed_exp.LN2_RECIP_Q32 = 6196328019#i64 := by
  simp [fixed_exp_source.fixed_exp.LN2_RECIP_Q32]

/-- The extracted `ln 2` constant in Q32.32. -/
theorem ln2_val : fixed_exp_source.fixed_exp.LN2_Q32 = 2977044472#i64 := by
  simp [fixed_exp_source.fixed_exp.LN2_Q32]

/-- The extracted Q32.32 scale `ONE = 1 <<< 32` evaluates to `2^32`. -/
theorem one_val : fixed_exp_source.fixed_exp.ONE = ok 4294967296#i64 := by
  have h : fixed_exp_source.fixed_exp.ONE ⦃ z => z = 4294967296#i64 ⦄ := by
    unfold fixed_exp_source.fixed_exp.ONE
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I64.size, IScalar.size, I64.numBits, Int.shiftLeft_eq, Int.bmod]
  obtain ⟨z, hz, hp⟩ := WP.spec_imp_exists h
  rw [hz, hp]

/-- The extracted lower window endpoint evaluates to `-16 · 2^32 = -68719476736`. -/
theorem lower_val :
    fixed_exp_source.fixed_exp.exp_reduce.LOWER = ok ((-68719476736)#i64) := by
  have h : fixed_exp_source.fixed_exp.exp_reduce.LOWER ⦃ z => z = (-68719476736)#i64 ⦄ := by
    unfold fixed_exp_source.fixed_exp.exp_reduce.LOWER
    rw [one_val]
    simp only [bind_tc_ok]
    apply WP.spec_mono (I64.mul_spec (x := (-16)#i64) (y := 4294967296#i64)
      (by norm_num [I64.min, I64.max, I64.numBits]) (by norm_num [I64.min, I64.max, I64.numBits]))
    intro z hz
    apply IScalar.eq_imp
    rw [hz]
    norm_num
  obtain ⟨z, hz, hp⟩ := WP.spec_imp_exists h
  rw [hz, hp]

/-- The extracted upper window endpoint evaluates to `21 · 2^32 = 90194313216`. -/
theorem upper_val :
    fixed_exp_source.fixed_exp.exp_reduce.UPPER = ok 90194313216#i64 := by
  have h : fixed_exp_source.fixed_exp.exp_reduce.UPPER ⦃ z => z = 90194313216#i64 ⦄ := by
    unfold fixed_exp_source.fixed_exp.exp_reduce.UPPER
    rw [one_val]
    simp only [bind_tc_ok]
    apply WP.spec_mono (I64.mul_spec (x := 21#i64) (y := 4294967296#i64)
      (by norm_num [I64.min, I64.max, I64.numBits]) (by norm_num [I64.min, I64.max, I64.numBits]))
    intro z hz
    apply IScalar.eq_imp
    rw [hz]
    norm_num
  obtain ⟨z, hz, hp⟩ := WP.spec_imp_exists h
  rw [hz, hp]

/-- The extracted rounding half-constant evaluates to `2^31`. -/
theorem half_val : fixed_exp_source.fixed_exp.round_div_2p32.HALF = ok 2147483648#i128 := by
  have h : fixed_exp_source.fixed_exp.round_div_2p32.HALF ⦃ z => z = 2147483648#i128 ⦄ := by
    unfold fixed_exp_source.fixed_exp.round_div_2p32.HALF
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  obtain ⟨z, hz, hp⟩ := WP.spec_imp_exists h
  rw [hz, hp]

/-! ## The rounding primitive: one shift on each branch

The extracted rounding step is the work unit of the series: `hpos`/`hneg` are the two named
overflow premises the body needs, and the post exposes exactly one arithmetic right shift by 32
per branch, so a rounding round costs one shift. -/

/-- `Int` shift-right is nonnegative on a nonnegative argument; the general fact the negative
branch of the rounding step needs. -/
theorem int_shiftRight_nonneg (a : ℤ) (s : ℕ) (h : 0 ≤ a) : 0 ≤ a >>> s := by
  obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le h
  show (0 : ℤ) ≤ ↑(n >>> s)
  exact Int.natCast_nonneg _

/-- The two transitions of the extracted rounding step, one arithmetic right shift by 32 each. -/
theorem round_div_2p32_spec (value : I128)
    (hpos : 0 ≤ value.val → value.val + 2147483648 ≤ I128.max)
    (hneg : value.val < 0 → -value.val + 2147483648 ≤ I128.max) :
    fixed_exp_source.fixed_exp.round_div_2p32 value ⦃ z =>
      (0 ≤ value.val → z.val = (value.val + 2147483648) >>> 32) ∧
      (value.val < 0 → z.val = - ((-value.val + 2147483648) >>> 32)) ⦄ := by
  unfold fixed_exp_source.fixed_exp.round_div_2p32
  rw [half_val]
  by_cases hv : 0 ≤ value.val
  · have hge : (0#i128) ≤ value := by rwa [IScalar.le_equiv]
    have hmax : value.val + 2147483648 ≤ I128.max := hpos hv
    simp only [hge, if_pos]
    step*
  · have hlt : value.val < 0 := not_le.mp hv
    have hmin : value.val ≠ I128.min := by
      intro h
      have hc := hneg (by rw [h]; norm_num [IScalar.min, I128.min, I128.numBits])
      rw [h] at hc
      norm_num [IScalar.min, I128.min, I128.numBits, I128.max] at hc
    have hge : ¬ ((0#i128) ≤ value) := by rwa [IScalar.le_equiv]
    have hmax : -value.val + 2147483648 ≤ I128.max := hneg hlt
    simp only [hge]
    step*
    all_goals
      (rw [i3_post]
       have h2 : 0 ≤ i2.val := by rw [i2_post, i_post]; omega
       have h3 : 0 ≤ i2.val >>> 32 := int_shiftRight_nonneg _ _ h2
       have hminlt : IScalar.min IScalarTy.I128 < 0 := by
         norm_num [IScalar.min, I128.min, I128.numBits]
       intro heq
       omega)

/-- The work charged to one rounding round: the extracted primitive returns one arithmetic right
shift by 32 of one addend, on either branch. -/
theorem round_div_one_shift (value : I128)
    (hpos : 0 ≤ value.val → value.val + 2147483648 ≤ I128.max)
    (hneg : value.val < 0 → -value.val + 2147483648 ≤ I128.max) :
    fixed_exp_source.fixed_exp.round_div_2p32 value ⦃ z =>
      ∃ u : ℤ, (u = (value.val + 2147483648) >>> 32 ∨ u = - ((-value.val + 2147483648) >>> 32)) ∧
        z.val = u ⦄ := by
  refine WP.spec_mono (round_div_2p32_spec value hpos hneg) ?_
  intro z hz
  rcases hz with ⟨hzpos, hzneg⟩
  by_cases hv : 0 ≤ value.val
  · exact ⟨(value.val + 2147483648) >>> 32, Or.inl rfl, hzpos hv⟩
  · exact ⟨- ((-value.val + 2147483648) >>> 32), Or.inr rfl, hzneg (not_le.mp hv)⟩

/-- The extracted rounding of `0` returns `0`; the witness input below reaches it seven times. -/
theorem round_div_zero : fixed_exp_source.fixed_exp.round_div_2p32 0#i128 = ok 0#i128 := by
  have h : fixed_exp_source.fixed_exp.round_div_2p32 0#i128 ⦃ z => z = 0#i128 ⦄ := by
    unfold fixed_exp_source.fixed_exp.round_div_2p32
    rw [half_val]
    simp only [bind_tc_ok]
    have hge : (0#i128) ≥ (0#i128) := (IScalar.le_equiv (0#i128) (0#i128)).2 (by simp)
    rw [if_pos hge]
    step
    step
    apply IScalar.eq_imp
    rw [z_post, i1_post]
    norm_num [Int.shiftRight_eq_div_pow]
  obtain ⟨z, hz, hp⟩ := WP.spec_imp_exists h
  rw [hz, hp]

/-! ## Work measure of the degree-7 series body

The extracted series body is a fixed seven-round chain: no loop, no recursion. `hornerRun` is that
chain entered at a rank, and each round is charged the three scalar operations the extracted body
performs: one `I128` multiply, one `round_div_2p32` call (one shift, by `round_div_one_shift`) and
one `I128` coefficient add. `coef j` is the extracted Q32 coefficient `round(2^32 / (7 - j)!)` of
round `j + 1`. -/

/-- If a `Result` bind returns a value, its first computation succeeded. -/
theorem bind_ret {α β : Type} {m : Result α} {k : α → Result β} {w : β}
    (h : (m >>= k) = ok w) : ∃ x, m = ok x ∧ k x = ok w := by
  cases hm : m with
  | ret x => exact ⟨x, rfl, by simpa [hm] using h⟩
  | vis e k' => rw [hm] at h; simp at h
  | div => rw [hm] at h; simp at h

/-- The accumulator, the rounds performed and the scalar operations charged. -/
structure HornerResult where
  /-- The accumulator the extracted body would hold at that rank. -/
  acc : I128
  /-- The number of Horner rounds performed. -/
  rounds : Nat
  /-- The number of scalar operations charged, three per round. -/
  ops : Nat

/-- The Q32.32 Taylor coefficient the extracted body adds in round `j + 1`. -/
def coef : Nat → I128
  | 0 => 852176#i128
  | 1 => 5965232#i128
  | 2 => 35791394#i128
  | 3 => 178956971#i128
  | 4 => 715827883#i128
  | 5 => 2147483648#i128
  | 6 => 4294967296#i128
  | 7 => 4294967296#i128
  | _ + 1 => 0#i128

/-- `hornerRun n r acc` is the extracted series body entered at rank `n`: `n` rounds of the
extracted round — one `I128` multiply, one `round_div_2p32` call, one `I128` coefficient add —
followed by the extracted final round (multiply, rounding call, plain coefficient return). The
rank is the remaining round count and the recursion is structural in it. -/
def hornerRun : Nat → I128 → I128 → Result HornerResult
  | 0, r, acc => do
      let m ← acc * r
      let q ← fixed_exp_source.fixed_exp.round_div_2p32 m
      let z ← q + coef 7
      ok ⟨z, 1, 3⟩
  | n + 1, r, acc => do
      let m ← acc * r
      let q ← fixed_exp_source.fixed_exp.round_div_2p32 m
      let acc' ← q + coef (6 - n)
      let s ← hornerRun n r acc'
      ok ⟨s.acc, s.rounds + 1, s.ops + 3⟩

/-- `hornerChain n r acc` is the same chain at value type, so that it can be compared with the
extracted body by definitional equality. -/
def hornerChain : Nat → I128 → I128 → Result I128
  | 0, r, acc => do
      let m ← acc * r
      let q ← fixed_exp_source.fixed_exp.round_div_2p32 m
      q + coef 7
  | n + 1, r, acc => do
      let m ← acc * r
      let q ← fixed_exp_source.fixed_exp.round_div_2p32 m
      let acc' ← q + coef (6 - n)
      hornerChain n r acc'

/-- The extracted series body is, term for term, rank 6 of that chain: seven rounds of the
extracted round in the extracted order with the extracted coefficients. -/
theorem horner_body (r : I64) :
    fixed_exp_source.fixed_exp.exp_horner r =
      (do let r1 ← lift (core.convert.num.FromI128I64.from r)
          hornerChain 6 r1 (coef 0)) := by
  unfold fixed_exp_source.fixed_exp.exp_horner
  simp only [hornerChain, coef]

/-- The work measure: `hornerRun n` performs exactly `n + 1` rounds and charges exactly
`3 * (n + 1)` scalar operations. -/
theorem hornerRun_measure (n : Nat) (r acc : I128) (s : HornerResult)
    (h : hornerRun n r acc = ok s) : s.rounds = n + 1 ∧ s.ops = 3 * (n + 1) := by
  induction n generalizing acc s with
  | zero =>
    unfold hornerRun at h
    obtain ⟨m, hm, h1⟩ := bind_ret h
    obtain ⟨q, hq, h2⟩ := bind_ret h1
    obtain ⟨z, hz, h3⟩ := bind_ret h2
    have h3' : HornerResult.mk z 1 3 = s := Result.ok_injective h3
    have hR : 1 = s.rounds := congrArg HornerResult.rounds h3'
    have hO : 3 = s.ops := congrArg HornerResult.ops h3'
    omega
  | succ n ih =>
    unfold hornerRun at h
    obtain ⟨m, hm, h1⟩ := bind_ret h
    obtain ⟨q, hq, h2⟩ := bind_ret h1
    obtain ⟨acc', ha, h3⟩ := bind_ret h2
    obtain ⟨t, ht, h4⟩ := bind_ret h3
    obtain ⟨hr, ho⟩ := ih acc' t ht
    have h4' : HornerResult.mk t.acc (t.rounds + 1) (t.ops + 3) = s := Result.ok_injective h4
    have hR : t.rounds + 1 = s.rounds := congrArg HornerResult.rounds h4'
    have hO : t.ops + 3 = s.ops := congrArg HornerResult.ops h4'
    omega

/-- One round of the extracted series body: given the three scalar operations of the round
succeeding, the body performs that round and continues at the strictly smaller rank `n`, charging
three scalar operations for it. -/
theorem hornerRun_round (n : Nat) (r acc m q acc' : I128) (t : HornerResult)
    (hm : acc * r = ok m) (hq : fixed_exp_source.fixed_exp.round_div_2p32 m = ok q)
    (ha : q + coef (6 - n) = ok acc') (hrec : hornerRun n r acc' = ok t) :
    hornerRun (n + 1) r acc = ok ⟨t.acc, t.rounds + 1, t.ops + 3⟩ ∧
      t.rounds = n + 1 ∧ t.ops = 3 * (n + 1) ∧ t.rounds < t.rounds + 1 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · unfold hornerRun
    simp only [hm, hq, ha, hrec, bind_tc_ok]
  · obtain ⟨hr, ho⟩ := hornerRun_measure n r acc' t hrec
    omega
  · obtain ⟨hr, ho⟩ := hornerRun_measure n r acc' t hrec
    omega
  · obtain ⟨hr, ho⟩ := hornerRun_measure n r acc' t hrec
    omega

/-- The value-only chain and the counted chain agree: same accumulator, same round and operation
counts. -/
theorem hornerChain_run (n : Nat) (r acc z : I128) (hz : hornerChain n r acc = ok z) :
    ∃ (s : HornerResult), hornerRun n r acc = ok s ∧ s.acc = z ∧
      s.rounds = n + 1 ∧ s.ops = 3 * (n + 1) := by
  induction n generalizing acc z with
  | zero =>
    unfold hornerChain at hz
    obtain ⟨m, hm, h1⟩ := bind_ret hz
    obtain ⟨q, hq, h2⟩ := bind_ret h1
    refine ⟨⟨z, 1, 3⟩, ?_, ?_⟩
    · unfold hornerRun
      simp only [hm, hq, h2, bind_tc_ok]
    · exact ⟨rfl, rfl, rfl⟩
  | succ n ih =>
    unfold hornerChain at hz
    obtain ⟨m, hm, h1⟩ := bind_ret hz
    obtain ⟨q, hq, h2⟩ := bind_ret h1
    obtain ⟨acc', ha, h3⟩ := bind_ret h2
    obtain ⟨s, hs, hsacc, hsr, hso⟩ := ih acc' z h3
    obtain ⟨heq, htr, hto, _⟩ := hornerRun_round n r acc m q acc' s hm hq ha hs
    refine ⟨⟨s.acc, s.rounds + 1, s.ops + 3⟩, heq, ?_⟩
    dsimp only
    exact ⟨hsacc, by omega, by omega⟩

/-- The work measure of the extracted series body: whenever the body returns a value at all, it
performed exactly 7 rounds and charged exactly 21 scalar operations — 7 `i128` multiplies, 7
rounding shifts, 7 coefficient adds — and the round rank fell from 7 to 0. -/
theorem horner_work (r : I64) (z : I128) (hz : fixed_exp_source.fixed_exp.exp_horner r = ok z) :
    ∃ (s : HornerResult),
      hornerRun 6 (core.convert.num.FromI128I64.from r) (coef 0) = ok s ∧
      s.rounds = 7 ∧ s.ops = 21 ∧ 0 < s.rounds ∧ s.acc = z := by
  have hchain : hornerChain 6 (core.convert.num.FromI128I64.from r) (coef 0) = ok z := by
    rw [horner_body] at hz
    simpa only [lift, bind_tc_ok] using hz
  obtain ⟨s, hs, hsacc, hsr, hso⟩ := hornerChain_run 6 _ _ z hchain
  obtain ⟨hr, ho⟩ := hornerRun_measure 6 _ _ s hs
  exact ⟨s, hs, by omega, by omega, by omega, hsacc⟩

/-! ## The series body at the witness input

The witness input `x = 2977044472 = LN2_Q32` reduces to `k = 1`, `r = 0`, so every Horner product
is `0` and the chain's final accumulator is the last coefficient. This is a concrete evaluation of
the extracted body, not a bound. -/

/-- At `r = 0` the extracted series body returns `2^32`, the Q32.32 image of `1`. -/
theorem horner_zero : fixed_exp_source.fixed_exp.exp_horner 0#i64 = ok 4294967296#i128 := by
  have h : fixed_exp_source.fixed_exp.exp_horner 0#i64 ⦃ z => z = 4294967296#i128 ⦄ := by
    unfold fixed_exp_source.fixed_exp.exp_horner
    simp only [lift, bind_tc_ok]
    have hz : core.convert.num.FromI128I64.from 0#i64 = 0#i128 := by
      refine IScalar.eq_of_val_eq ?_
      rw [core.convert.num.FromI128I64.from_val_eq]
      norm_num
    rw [hz]
    step*
    have h1 : i = 0#i128 := IScalar.eq_of_val_eq (by norm_num [i_post])
    rw [h1, round_div_zero]
    simp only [bind_tc_ok]
    step*
    have h2 : i2 = 0#i128 := IScalar.eq_of_val_eq (by norm_num [i2_post])
    rw [h2, round_div_zero]
    simp only [bind_tc_ok]
    step*
    have h3 : i4 = 0#i128 := IScalar.eq_of_val_eq (by norm_num [i4_post])
    rw [h3, round_div_zero]
    simp only [bind_tc_ok]
    step*
    have h4 : i6 = 0#i128 := IScalar.eq_of_val_eq (by norm_num [i6_post])
    rw [h4, round_div_zero]
    simp only [bind_tc_ok]
    step*
    have h5 : i8 = 0#i128 := IScalar.eq_of_val_eq (by norm_num [i8_post])
    rw [h5, round_div_zero]
    simp only [bind_tc_ok]
    step*
    have h6 : i10 = 0#i128 := IScalar.eq_of_val_eq (by norm_num [i10_post])
    rw [h6, round_div_zero]
    simp only [bind_tc_ok]
    step*
    have h7 : i12 = 0#i128 := IScalar.eq_of_val_eq (by norm_num [i12_post])
    rw [h7, round_div_zero]
    simp only [bind_tc_ok]
    step*
  obtain ⟨z, hz, hp⟩ := WP.spec_imp_exists h
  rw [hz, hp]

/-! ## Range-reduction transitions, with guards -/

/-- The first guard of the extracted reduction: below the lower endpoint the body refuses before
any arithmetic. -/
theorem reduce_refuses_below_lower (x c : I64)
    (hL : fixed_exp_source.fixed_exp.exp_reduce.LOWER = ok c) (h : x < c) :
    fixed_exp_source.fixed_exp.exp_reduce x =
      ok (core.result.Result.Err fixed_exp_source.fixed_exp.ExpError.OutOfRange) := by
  unfold fixed_exp_source.fixed_exp.exp_reduce
  rw [hL]
  simp only [bind_tc_ok, if_pos h]

/-- The second guard: above the upper endpoint the body refuses before the reduction multiply. -/
theorem reduce_refuses_above_upper (x cL cU : I64)
    (hL : fixed_exp_source.fixed_exp.exp_reduce.LOWER = ok cL)
    (hU : fixed_exp_source.fixed_exp.exp_reduce.UPPER = ok cU)
    (hn : ¬ x < cL) (h : x > cU) :
    fixed_exp_source.fixed_exp.exp_reduce x =
      ok (core.result.Result.Err fixed_exp_source.fixed_exp.ExpError.OutOfRange) := by
  unfold fixed_exp_source.fixed_exp.exp_reduce
  rw [hL, hU]
  simp only [bind_tc_ok, if_neg hn, if_pos h]

-- NOT PROVED IN THIS FILE: `reduce_witness`, the concrete value of the extracted reduction at
-- `x = 2977044472` (`exp_reduce 2977044472 = ok (Ok (1, 0))`), and the end-to-end value witness
-- `fixed_exp 2977044472 = ok (Ok 8589934592)` built from it. The residual goal after `step*` is the
-- impossible-branch equation `Err OutOfRange = Ok (1, 0)`, whose context carries no value post for
-- the last `i128` product (`k_ln2_wide`); closing it needs that post or a `noConfusion` close of the
-- two constructors. `horner_zero` and `scale_flush_zero` give reachable success results of the two
-- other extracted components.

/-! ## Scaling transitions, with guards

The extracted `exp_scale` has two branches. For `k ≥ 0` it is a checked left shift (one shift,
three guards); for `k < 0` a round-to-nearest right shift with a flush-to-zero branch. -/

/-- The shift-count guard of the left branch: `k ≥ 64` refuses before any shift is attempted, so
the branch costs zero shifts. -/
theorem scale_refuses_shift_count (k : I64) (acc : I128) (hk0 : (0#i64) ≤ k) (hk64 : (64#i64) ≤ k) :
    fixed_exp_source.fixed_exp.exp_scale k acc =
      ok (core.result.Result.Err fixed_exp_source.fixed_exp.ExpError.OutOfRange) := by
  unfold fixed_exp_source.fixed_exp.exp_scale
  split <;> rfl

-- NOT PROVED IN THIS FILE: `scale_witness`, the concrete value of the left-shift branch
-- (`exp_scale 1 (4294967296#i128) = ok (Ok 8589934592)`), and the end-to-end value witness
-- `reduce_witness`. Both need `Int.bmod` normalization of the extracted shift post-conditions
-- under `norm_num ... at *`, which did not close here; see the handoff.

/-- The flush-to-zero exit branch of the extracted scaling, at a machine-reachable exponent:
`k = -64` takes the negative branch, computes `n = 64`, and the `n ≥ 64` guard returns `Ok 0`
without performing any shift. This is a reachable success result on the extracted body. -/
theorem scale_flush_zero :
    fixed_exp_source.fixed_exp.exp_scale ((-64)#i64) 0#i128 =
      ok (core.result.Result.Ok 0#i64) := by
  have h : fixed_exp_source.fixed_exp.exp_scale ((-64)#i64) 0#i128 ⦃ z =>
      z = core.result.Result.Ok 0#i64 ⦄ := by
    unfold fixed_exp_source.fixed_exp.exp_scale
    simp only [lift, bind_tc_ok]
    have hlt : ¬ ((0#i64) ≤ (-64)#i64) := by
      rw [IScalar.le_equiv]
      norm_num
    rw [if_neg hlt]
    have hlt2 : ((-64)#i64) < (0#i64) := (IScalar.lt_equiv ((-64)#i64) (0#i64)).2 (by norm_num)
    rw [if_pos hlt2]
    have hne : ((-64)#i64 : ℤ) ≠ IScalar.min IScalarTy.I64 := by
      norm_num [I64.min, I64.numBits, I64.rMin]
    obtain ⟨r, hr, hp⟩ := WP.spec_imp_exists (HNeg.hNeg.step ((-64)#i64) hne)
    have hneg : -. ((-64)#i64) = ok 64#i64 := by
      rw [hr]
      refine congrArg ok (IScalar.eq_of_val_eq ?_)
      rw [hp]
      norm_num
    rw [hneg]
    simp only [bind_tc_ok]
    have hge : (64#u64) ≤ IScalar.hcast UScalarTy.U64 (64#i64) := by
      rw [UScalar.le_equiv]
      norm_num [IScalar.hcast_val_eq]
    rw [if_pos hge]
    simp only [WP.spec_ok]
  obtain ⟨z, hz, hp⟩ := WP.spec_imp_exists h
  rw [hz, hp]

/-! ## The whole body: the refusal branch and a reachable success -/

/-- The refusal branch of the extracted body: outside the documented window
`[-16 · 2^32, 21 · 2^32]` the body returns `Err OutOfRange` without entering the series. -/
theorem refuses_outside_window (x : I64)
    (hx : x < (-68719476736)#i64 ∨ (90194313216)#i64 < x) :
    fixed_exp_source.fixed_exp.fixed_exp x =
      ok (core.result.Result.Err fixed_exp_source.fixed_exp.ExpError.OutOfRange) := by
  unfold fixed_exp_source.fixed_exp.fixed_exp
  rcases hx with hx | hx
  · rw [reduce_refuses_below_lower x _ lower_val hx]
    simp only [bind_tc_ok]
  · rw [reduce_refuses_above_upper x _ _ lower_val upper_val ?_ hx]
    simp only [bind_tc_ok]
    intro hc
    have hx' : (x : ℤ) < -68719476736 := by
      simpa using (IScalar.lt_equiv x (-68719476736)#i64).1 hc
    have hx2 : (90194313216 : ℤ) < (x : ℤ) := by simpa using hx
    have hge : (-68719476736 : ℤ) ≤ (x : ℤ) := by omega
    exact (not_lt_of_ge hge) hx'

end FixedExpResource

open Lean Elab Command in
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit FixedExpResource.round_div_2p32_spec
#resource_audit FixedExpResource.round_div_one_shift
#resource_audit FixedExpResource.round_div_zero
#resource_audit FixedExpResource.horner_body
#resource_audit FixedExpResource.hornerRun_measure
#resource_audit FixedExpResource.hornerRun_round
#resource_audit FixedExpResource.hornerChain_run
#resource_audit FixedExpResource.horner_work
#resource_audit FixedExpResource.horner_zero
#resource_audit FixedExpResource.reduce_refuses_below_lower
#resource_audit FixedExpResource.reduce_refuses_above_upper
#resource_audit FixedExpResource.scale_refuses_shift_count
#resource_audit FixedExpResource.scale_flush_zero
#resource_audit FixedExpResource.refuses_outside_window

run_cmd Lean.logInfo "FIXED_EXP_KERNEL_OK"
