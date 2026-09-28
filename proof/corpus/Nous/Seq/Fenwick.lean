import Mathlib

/-!
# The Fenwick (binary-indexed) partition of a prefix, and log-linear attention

Log-linear attention (Guo et al., *Log-Linear Attention*, arXiv 2506.04761) replaces the
semiseparable attention mask by a hierarchical one and, at decoding time, partitions the
prefix `[0, t)` into power-of-two *buckets* indexed by a level `ℓ`, following Fenwick's
binary-indexed tree (Fenwick, *A new data structure for cumulative frequency tables*, 1994;
Ryabko, *Adaptive coding of a source with unknown statistics*, 1992).

The paper text retrieved at `~/lit/2506.04761_log_linear_attention.txt`
describes the partition in §3.1 (lines 352-366): "prefix into buckets of exponentially
increasing size via a Fenwick-tree decomposition", "at most `L = O(log T)` disjoint buckets
indexed by level ℓ", guided by `lssb(t) = max {ℓ ∈ ℕ | 2^ℓ divides t}` (line 366), and §3.2
(lines 552-576, 646) gives the incremental update: "At each step, the immediate term
`v_t k_tᵀ` enters the finest level; buckets up to `lssb(t)` merge and promote one level
coarser." The output in Eq. (3) (line 393) is `o_t = Σ_ℓ λ_t^{(ℓ)} q_tᵀ S_t^{(ℓ)}` and the
parallel form in Eq. (4) (line 423) is `O = (QKᵀ ⊙ M) V` with `M_{ts} = λ_t^{ℓ(t,s)}` for
`s ≤ t`, where `ℓ(t, s)` is the bucket level of `s` relative to `t`.

This module formalises the partition and its incremental update. The convention here is the
standard Fenwick one stated in the task: **each set bit `ℓ` of `t` contributes one interval
of length `2^ℓ`**, so that for `t = 13 = 1101₂` the three buckets have sizes `8, 4, 1` (the
paper's prose indexes the sentinel bucket as level `0` and writes sizes `2^{ℓ-1}` for levels
`ℓ ≥ 1`; that is the same partition in a shifted indexing, and this module's `level` returns
the exponent of the bucket size).

The partition of `[0, t)` is equivalently the canonical dyadic decomposition of the interval:
the level of `s < t` is the highest bit position at which `s` and `t` differ, which is
`(t ^^^ s).log2`. Every `s < t` lies in exactly one bucket; the number of nonempty buckets is
the popcount of `t`; and the step `t → t+1` merges exactly the trailing-ones levels of `t`
into the next level, so that the total number of merges over `T` steps is at most `T`.

The final section states the equivalence, over a commutative ring `R`, between the output
computed from the per-level accumulated states and the masked quadratic form of Eq. (4).

## References

* Fenwick, *Software: Practice and Experience* **24** (1994) 327-336.
* Guo et al., arXiv:2506.04761, §3.1-§3.2, Eq. (3), Eq. (4).
-/

open scoped BigOperators

namespace Nous.Seq.Fenwick

/-! ## The partition -/

/-- `level t s` is the Fenwick level of position `s` in the prefix `[0, t)` for `s < t`: the
exponent `ℓ` such that the bucket containing `s` has size `2^ℓ`. It is the highest bit position
at which `s` and `t` differ, i.e. the binary logarithm of their bitwise exclusive or. For
`s ≥ t` (outside the admitted use) the value is meaningless but the function is total. -/
def level (t s : ℕ) : ℕ := (t ^^^ s).log2

/-- `bucket t ℓ` is the set of positions `s < t` whose Fenwick level relative to `t` is `ℓ`. -/
def bucket (t ℓ : ℕ) : Finset ℕ := (Finset.range t).filter (fun s => level t s = ℓ)

/-- `levelSet t` is the set of levels occupied by the partition of `[0, t)`, i.e. the image of
`level t` on the prefix. -/
def levelSet (t : ℕ) : Finset ℕ := (Finset.range t).image (level t)

/-- `setBits t` is the set of positions of the set bits in the binary expansion of `t`. Any set
bit of `t` is below `t`, so filtering the range `[0, t)` by `testBit` gives exactly the set
bits. This is the shape in which the popcount is computed. -/
def setBits (t : ℕ) : Finset ℕ := (Finset.range t).filter (fun ℓ => t.testBit ℓ)

/-- `popcount t` is the number of set bits in the binary expansion of `t`, i.e. the number of
nonempty Fenwick buckets of `[0, t)`. -/
def popcount (t : ℕ) : ℕ := (setBits t).card

/-- `trailingOnes t` is the number of consecutive `1` bits at the bottom of the binary
expansion of `t`; this is the number of Fenwick levels merged by the step `t → t + 1`. -/
def trailingOnes (t : ℕ) : ℕ := (List.takeWhile id t.bits).length

/-! ## Membership and the partition properties -/

@[simp] lemma mem_bucket {t ℓ s : ℕ} : s ∈ bucket t ℓ ↔ s < t ∧ level t s = ℓ := by
  simp [bucket]

lemma mem_levelSet {t ℓ : ℕ} : ℓ ∈ levelSet t ↔ ∃ s, s < t ∧ level t s = ℓ := by
  simp [levelSet, Finset.mem_image, eq_comm]

/-- The buckets of a fixed prefix are pairwise disjoint. -/
theorem buckets_disjoint (t ℓ ℓ' : ℕ) (h : ℓ ≠ ℓ') : Disjoint (bucket t ℓ) (bucket t ℓ') := by
  rw [Finset.disjoint_left]
  intro s hs hs'
  rw [mem_bucket] at hs hs'
  exact h (hs.2.symm.trans hs'.2)

/-- Every `s < t` lies in exactly one bucket of the partition of `[0, t)`. -/
theorem buckets_cover (t s : ℕ) :
    s ∈ Finset.range t ↔ ∃! ℓ, s ∈ bucket t ℓ := by
  constructor
  · intro hs
    refine ⟨level t s, ?_, ?_⟩
    · change s ∈ bucket t (level t s)
      rw [mem_bucket]; exact ⟨Finset.mem_range.mp hs, rfl⟩
    · intro ℓ hℓ
      rw [mem_bucket] at hℓ
      exact hℓ.2.symm
  · rintro ⟨ℓ, hℓ, _⟩
    rw [mem_bucket] at hℓ
    exact Finset.mem_range.mpr hℓ.1

/-- The buckets indexed by the occupied levels cover the prefix. -/
lemma biUnion_buckets (t : ℕ) : (levelSet t).biUnion (bucket t) = Finset.range t := by
  apply Finset.Subset.antisymm
  · intro s hs
    rw [Finset.mem_biUnion] at hs
    obtain ⟨ℓ, -, hsℓ⟩ := hs
    rw [mem_bucket] at hsℓ
    exact Finset.mem_range.mpr hsℓ.1
  · intro s hs
    rw [Finset.mem_biUnion]
    refine ⟨level t s, ?_, ?_⟩
    · rw [mem_levelSet]; exact ⟨s, Finset.mem_range.mp hs, rfl⟩
    · rw [mem_bucket]; exact ⟨Finset.mem_range.mp hs, rfl⟩

/-! ## Elementary bit facts -/

lemma testBit_two_mul_zero (n : ℕ) : (2 * n).testBit 0 = false := by
  rw [Nat.testBit_zero, Nat.mul_mod_right]
  rfl

lemma testBit_two_mul_add_one_zero (n : ℕ) : (2 * n + 1).testBit 0 = true := by
  rw [Nat.testBit_zero]
  have h : (2 * n + 1) % 2 = 1 := by omega
  rw [h]
  rfl

lemma testBit_two_mul_succ (n j : ℕ) : (2 * n).testBit (j + 1) = n.testBit j := by
  rw [Nat.testBit_succ, Nat.mul_div_right n (by norm_num : 0 < 2)]

lemma testBit_two_mul_add_one_succ (n j : ℕ) :
    (2 * n + 1).testBit (j + 1) = n.testBit j := by
  rw [Nat.testBit_succ]
  have h : (2 * n + 1) / 2 = n := by
    rw [Nat.mul_add_div (by norm_num : 0 < 2) n 1]
    simp
  rw [h]

lemma testBit_eq_false_of_le {n j : ℕ} (h : n ≤ j) : n.testBit j = false :=
  Nat.testBit_eq_false_of_lt (lt_of_le_of_lt h Nat.lt_two_pow_self)

lemma t_lt_two_pow_log2_succ (t : ℕ) : t < 2 ^ (t.log2 + 1) := by
  simpa [Nat.log2_eq_log_two] using Nat.lt_pow_succ_log_self (by norm_num : 1 < 2) t

lemma log2_lt_self (t : ℕ) (h : 0 < t) : t.log2 < t := by
  rw [Nat.log2_lt h.ne']
  exact Nat.lt_two_pow_self

lemma testBit_of_isHighestBit {n ℓ : ℕ} (hlo : 2 ^ ℓ ≤ n) (hhi : n < 2 ^ (ℓ + 1)) :
    n.testBit ℓ = true := by
  rw [Nat.testBit_eq_decide_div_mod_eq, decide_eq_true_eq]
  have hdiv : n / 2 ^ ℓ = 1 := by
    apply Nat.div_eq_of_lt_le
    · simpa using hlo
    · simpa [Nat.pow_succ, Nat.mul_comm] using hhi
  rw [hdiv]

lemma testBit_eq_of_xor_testBit_false {a b j : ℕ} (h : (a ^^^ b).testBit j = false) :
    a.testBit j = b.testBit j := by
  rw [Nat.testBit_xor] at h
  cases ha : a.testBit j <;> cases hb : b.testBit j <;> simp_all

lemma testBit_ne_of_xor_testBit_true {a b j : ℕ} (h : (a ^^^ b).testBit j = true) :
    a.testBit j ≠ b.testBit j := by
  rw [Nat.testBit_xor] at h
  intro heq
  rw [heq, Bool.xor_self] at h
  exact Bool.noConfusion h

/-! ## The levels are exactly the set bits -/

/-- A level of the partition is a set bit of `t`, and conversely: the occupied levels are the
positions of the `1` bits in the binary expansion of `t`. -/
theorem levelSet_eq_setBits (t : ℕ) : levelSet t = setBits t := by
  apply Finset.Subset.antisymm
  · -- occupied levels are set bits
    intro ℓ hℓ
    rw [mem_levelSet] at hℓ
    obtain ⟨s, hst, hsl⟩ := hℓ
    have hne : t ^^^ s ≠ 0 := by
      intro h0
      rw [Nat.xor_eq_zero_iff] at h0
      omega
    obtain ⟨hlo, hhi⟩ := (Nat.log2_eq_iff hne).mp hsl
    have hagree : ∀ j, ℓ < j → t.testBit j = s.testBit j := by
      intro j hj
      refine testBit_eq_of_xor_testBit_false ?_
      exact Nat.testBit_eq_false_of_lt
        (lt_of_lt_of_le hhi (Nat.pow_le_pow_right (by norm_num) (by omega)))
    have hbit : t.testBit ℓ = true := by
      have hnbit : (t ^^^ s).testBit ℓ = true := testBit_of_isHighestBit hlo hhi
      have hneq : t.testBit ℓ ≠ s.testBit ℓ := testBit_ne_of_xor_testBit_true hnbit
      by_contra hf
      have hf' : t.testBit ℓ = false := by cases hv : t.testBit ℓ <;> simp_all
      have hs' : s.testBit ℓ = true := by
        cases hv : t.testBit ℓ <;> cases hw : s.testBit ℓ <;> simp_all
      have := Nat.lt_of_testBit ℓ hf' hs' hagree
      omega
    rw [setBits, Finset.mem_filter, Finset.mem_range]
    refine ⟨?_, hbit⟩
    have ht_lt : t < 2 ^ (t.log2 + 1) := t_lt_two_pow_log2_succ t
    have hn_lt : t ^^^ s < 2 ^ (t.log2 + 1) :=
      Nat.xor_lt_two_pow ht_lt (lt_trans hst ht_lt)
    have hℓlt : ℓ < t.log2 + 1 := by
      have := (Nat.log2_lt hne).mpr hn_lt
      rw [← hsl]
      exact this
    have hlog : t.log2 < t := log2_lt_self t (by omega)
    omega
  · -- set bits are occupied levels
    intro ℓ hℓ
    rw [setBits, Finset.mem_filter, Finset.mem_range] at hℓ
    obtain ⟨-, hbit⟩ := hℓ
    rw [mem_levelSet]
    refine ⟨t ^^^ 2 ^ ℓ, ?_, ?_⟩
    · apply Nat.lt_of_testBit ℓ
      · rw [Nat.testBit_xor, hbit, Nat.testBit_two_pow]; simp
      · exact hbit
      · intro j hj
        rw [Nat.testBit_xor, Nat.testBit_two_pow]
        have : decide (ℓ = j) = false := by simp [ne_of_lt hj]
        rw [this, Bool.xor_false]
    · show (t ^^^ (t ^^^ 2 ^ ℓ)).log2 = ℓ
      rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor, Nat.log2_two_pow]

lemma level_le_log2 {t s : ℕ} (h : s < t) : level t s ≤ t.log2 := by
  have hne : t ^^^ s ≠ 0 := by
    intro h0
    rw [Nat.xor_eq_zero_iff] at h0
    omega
  have ht : t < 2 ^ (t.log2 + 1) := t_lt_two_pow_log2_succ t
  have hn : t ^^^ s < 2 ^ (t.log2 + 1) := Nat.xor_lt_two_pow ht (lt_trans h ht)
  have := (Nat.log2_lt hne).mpr hn
  change (t ^^^ s).log2 ≤ t.log2
  omega

/-- The number of nonempty buckets of the prefix `[0, t)` is the popcount of `t`, and it is at
most `log₂ t + 1`. -/
theorem bucket_count (t : ℕ) :
    (levelSet t).card = popcount t ∧ popcount t ≤ t.log2 + 1 := by
  have hset : levelSet t = setBits t := levelSet_eq_setBits t
  constructor
  · rw [hset, popcount]
  · rw [popcount, ← hset]
    calc (levelSet t).card ≤ (Finset.range (t.log2 + 1)).card := by
          apply Finset.card_le_card
          intro ℓ hℓ
          rw [mem_levelSet] at hℓ
          obtain ⟨s, hst, _⟩ := hℓ
          rw [Finset.mem_range]
          have := level_le_log2 hst
          omega
      _ = t.log2 + 1 := Finset.card_range _

/-! ## The binary carry: popcount and the merge count -/

lemma setBits_two_mul (n : ℕ) : setBits (2 * n) = (setBits n).image (· + 1) := by
  ext ℓ
  rw [Finset.mem_image]
  constructor
  · intro h
    rw [setBits, Finset.mem_filter, Finset.mem_range] at h
    obtain ⟨-, hb⟩ := h
    cases ℓ with
    | zero => rw [testBit_two_mul_zero] at hb; exact absurd hb (by simp)
    | succ j =>
      refine ⟨j, ?_, rfl⟩
      rw [setBits, Finset.mem_filter, Finset.mem_range]
      have hjn : j < n := by
        by_contra hj
        rw [testBit_two_mul_succ, testBit_eq_false_of_le (not_lt.mp hj)] at hb
        exact absurd hb (by simp)
      exact ⟨hjn, by rwa [testBit_two_mul_succ] at hb⟩
  · rintro ⟨j, hj, rfl⟩
    rw [setBits, Finset.mem_filter, Finset.mem_range] at hj
    obtain ⟨hjn, hb⟩ := hj
    rw [setBits, Finset.mem_filter, Finset.mem_range]
    exact ⟨by omega, by rwa [testBit_two_mul_succ]⟩

lemma setBits_two_mul_add_one (n : ℕ) :
    setBits (2 * n + 1) = insert 0 ((setBits n).image (· + 1)) := by
  ext ℓ
  rw [Finset.mem_insert, Finset.mem_image]
  constructor
  · intro h
    rw [setBits, Finset.mem_filter, Finset.mem_range] at h
    obtain ⟨-, hb⟩ := h
    cases ℓ with
    | zero => exact Or.inl rfl
    | succ j =>
      refine Or.inr ⟨j, ?_, rfl⟩
      rw [setBits, Finset.mem_filter, Finset.mem_range]
      have hjn : j < n := by
        by_contra hj
        rw [testBit_two_mul_add_one_succ, testBit_eq_false_of_le (not_lt.mp hj)] at hb
        exact absurd hb (by simp)
      exact ⟨hjn, by rwa [testBit_two_mul_add_one_succ] at hb⟩
  · rintro (rfl | ⟨j, hj, rfl⟩)
    · rw [setBits, Finset.mem_filter, Finset.mem_range]
      exact ⟨by omega, testBit_two_mul_add_one_zero n⟩
    · rw [setBits, Finset.mem_filter, Finset.mem_range] at hj
      obtain ⟨hjn, hb⟩ := hj
      rw [setBits, Finset.mem_filter, Finset.mem_range]
      exact ⟨by omega, by rwa [testBit_two_mul_add_one_succ]⟩

lemma popcount_zero : popcount 0 = 0 := by simp [popcount, setBits]

lemma popcount_two_mul (n : ℕ) : popcount (2 * n) = popcount n := by
  rw [popcount, setBits_two_mul, Finset.card_image_of_injective _ fun a b h => Nat.succ.inj h,
    popcount]

lemma popcount_two_mul_add_one (n : ℕ) : popcount (2 * n + 1) = popcount n + 1 := by
  have h0 : 0 ∉ (setBits n).image (· + 1) := by
    intro h
    obtain ⟨j, -, hj⟩ := Finset.mem_image.mp h
    omega
  rw [popcount, setBits_two_mul_add_one, Finset.card_insert_of_notMem h0,
    Finset.card_image_of_injective _ fun a b h => Nat.succ.inj h, ← popcount]

lemma trailingOnes_two_mul (n : ℕ) : trailingOnes (2 * n) = 0 := by
  rcases eq_or_ne n 0 with rfl | hn
  · simp [trailingOnes]
  · rw [trailingOnes, Nat.bit0_bits n hn]
    simp

lemma trailingOnes_two_mul_add_one (n : ℕ) :
    trailingOnes (2 * n + 1) = trailingOnes n + 1 := by
  rw [trailingOnes, Nat.bit1_bits n]
  simp [trailingOnes]

/-- The popcount recurrence across a step: increasing `T` by one adds a set bit unless it
carries through `trailingOnes T` ones, which are cleared. -/
lemma popcount_succ_add (T : ℕ) : popcount (T + 1) + trailingOnes T = popcount T + 1 := by
  induction T using Nat.strong_induction_on with
  | h T ih =>
    rcases Nat.even_or_odd T with he | ho
    · obtain ⟨m, rfl⟩ : ∃ m, T = 2 * m := by
        obtain ⟨m, hm⟩ := he
        exact ⟨m, by omega⟩
      rw [popcount_two_mul_add_one, trailingOnes_two_mul, popcount_two_mul]
    · obtain ⟨m, rfl⟩ : ∃ m, T = 2 * m + 1 := by
        obtain ⟨m, hm⟩ := ho
        exact ⟨m, hm⟩
      have h2 : 2 * m + 1 + 1 = 2 * (m + 1) := by ring
      rw [h2, popcount_two_mul, popcount_two_mul_add_one, trailingOnes_two_mul_add_one]
      have := ih m (by omega)
      omega

/-- Over `T` steps the number of level merges plus the popcount telescopes: the total number of
merges in steps `0, …, T-1` is `T - popcount T`. -/
theorem sum_trailingOnes_add_popcount (T : ℕ) :
    (∑ t ∈ Finset.range T, trailingOnes t) + popcount T = T := by
  induction T with
  | zero => simp [popcount_zero]
  | succ T ih =>
    rw [Finset.sum_range_succ]
    have hpc := popcount_succ_add T
    omega

/-- The total number of level merges over the steps `0, …, T-1` is at most `T`: amortised
`O(1)` merges per token, even though a single step `t → t+1` may carry through up to
`trailingOnes t ≤ log₂ t + 1` levels. -/
theorem merges_total_le (T : ℕ) :
    (∑ t ∈ Finset.range T, trailingOnes t) ≤ T := by
  have h := sum_trailingOnes_add_popcount T
  omega

/-! ## Output equivalence: per-level states versus the masked quadratic form -/

section Ring

variable {R : Type*} [CommRing R]

/-- The accumulated state of bucket `ℓ` at time `t`: the sum of the outer products
`k_s ⊗ v_s` over the positions `s` of the bucket. This is the `S_t^{(ℓ)}` of Eq. (3) for the
ungated (linear attention) case. -/
def levelState {d e : ℕ} (t ℓ : ℕ) (k : ℕ → Fin d → R) (v : ℕ → Fin e → R) :
    Matrix (Fin d) (Fin e) R :=
  ∑ s ∈ bucket t ℓ, Matrix.vecMulVec (k s) (v s)

/-- **Log-linear output equivalence.** Summing `λ_t^{(ℓ)} q_t` against the per-level states
`S_t^{(ℓ)}` (Eq. (3), paper line 393) equals the masked quadratic form
`Σ_{s<t} λ_t^{ℓ(t,s)} (q_t · k_s) v_s` (Eq. (4), paper line 423). -/
theorem loglinear_eq_masked (d e : ℕ) (q k : ℕ → Fin d → R) (v : ℕ → Fin e → R)
    (lam : ℕ → ℕ → R) (t : ℕ) :
    (∑ ℓ ∈ levelSet t, lam t ℓ • Matrix.vecMul (q t) (levelState t ℓ k v))
      = ∑ s ∈ Finset.range t, lam t (level t s) • ((dotProduct (q t) (k s)) • v s) := by
  have h1 : ∀ ℓ : ℕ, Matrix.vecMul (q t) (levelState t ℓ k v)
      = ∑ s ∈ bucket t ℓ, (dotProduct (q t) (k s)) • v s := by
    intro ℓ
    rw [levelState, Matrix.vecMul_sum]
    refine Finset.sum_congr rfl fun s _ => ?_
    rw [Matrix.vecMul_vecMulVec]
  simp_rw [h1, Finset.smul_sum]
  have h2 : ∀ ℓ : ℕ,
      (∑ s ∈ bucket t ℓ, lam t ℓ • ((dotProduct (q t) (k s)) • v s))
        = ∑ s ∈ bucket t ℓ, lam t (level t s) • ((dotProduct (q t) (k s)) • v s) := by
    intro ℓ
    refine Finset.sum_congr rfl fun s hs => ?_
    rw [mem_bucket] at hs
    rw [hs.2]
  simp_rw [h2]
  have hdisj : ((levelSet t : Set ℕ)).PairwiseDisjoint (bucket t) := by
    rw [Finset.pairwiseDisjoint_iff]
    intro i _ j _ hij
    by_contra hne
    obtain ⟨s, hs⟩ := hij
    rw [Finset.mem_inter] at hs
    exact Finset.disjoint_left.mp (buckets_disjoint t i j hne) hs.1 hs.2
  rw [← Finset.sum_biUnion (f := fun s => lam t (level t s) •
        ((dotProduct (q t) (k s)) • v s)) hdisj, biUnion_buckets]

end Ring

/-! ## Witnesses

Concrete closed instances discharge the non-vacuity of the definitions and of the headline
statements. `t = 13 = 1101₂` has set bits at levels `0, 2, 3`, hence three buckets of sizes
`2^3 = 8`, `2^2 = 4` and `2^0 = 1`, matching the task witness and Fig. 2 of the paper. -/

namespace Witness

/-- `13 = 1101₂` has set bits at levels `0, 2, 3`. -/
example : setBits 13 = {0, 2, 3} := by decide

/-- The three nonempty buckets of `[0, 13)`. -/
example : levelSet 13 = {0, 2, 3} := by decide

/-- Bucket sizes are `8`, `4`, `1`. -/
example : (bucket 13 3).card = 8 := by decide
example : (bucket 13 2).card = 4 := by decide
example : (bucket 13 0).card = 1 := by decide

/-- The count equals the popcount, here `3 ≤ log₂ 13 + 1 = 4`. -/
example : popcount 13 = 3 := by decide
example : popcount 13 ≤ (13).log2 + 1 := by decide

/-- `13 = 1101₂` has one trailing one; `7 = 111₂` has three. -/
example : trailingOnes 13 = 1 := by decide
example : trailingOnes 7 = 3 := by decide

/-- The telescoping bound on a concrete range: `Σ_{t<8} trailingOnes t = 7 ≤ 8`. -/
example : (∑ t ∈ Finset.range 8, trailingOnes t) = 7 := by decide
example : (∑ t ∈ Finset.range 8, trailingOnes t) ≤ 8 := by decide

/-- Numeric check of `loglinear_eq_masked` over `ℤ` with `d = e = 1`, `t = 3 = 11₂`:
`k_s = s`, `v_s = 2s+1`, `q_3 = 1`, `λ_3^{ℓ} = 3 + ℓ`. Both sides evaluate to `42`. -/
example :
    (∑ ℓ ∈ levelSet 3, (3 + ℓ : ℤ) •
        Matrix.vecMul (fun _ : Fin 1 => (1 : ℤ))
          (levelState 3 ℓ (fun s _ => (s : ℤ)) (fun s _ => (2 * (s : ℤ) + 1))))
      = (fun _ : Fin 1 => (42 : ℤ)) := by decide

end Witness

end Nous.Seq.Fenwick
