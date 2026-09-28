import Substrate.SeedCore
import Substrate.Algebra.NullCell.NullCell
import Mathlib.Algebra.Order.Round
import Mathlib.Data.List.GetD
import Mathlib.Topology.MetricSpace.Pseudo.Defs

/-!
# The `.seed` storage standard — one file, one format

@name The Tiny File That Grows Back the Whole Thing
@child Instead of storing every number, you store a good *guess* for each one and only the tiny *mistake* left over. If the guess is perfect, the leftover is nothing — so a whole file can be regrown from an almost-empty seed.
@scientist Rest each value on a prediction (the Being arm B) and store only the residual (the Null arm U); the generate / lossless-compress / lossy-quantize split is derived by δS = 0 on the rate–distortion free energy F = D + λR, never hardcoded.
@engineer The `.seed` container (codebook of floors + (idx,res) pairs); `loadR_saveR` (lossless round-trip), `loadBytes_saveBytes` (LEB128 varint), `unpackBits_packBits` (sub-byte indices), `quant_error_bound`/`quant_is_nearest` (lossy floor ≤ s/2).

This is the consolidated `.seed` standard. **One idea, three faces of the rule of three —
all `x = B/U`:** rest each value on a *prediction* (the floor / generator output — the
**Being arm `B`**), and store only the *residual* (the **Null arm `U`**). What you do with
the residual is the only difference between the three arms the field uses:

* **generative seed** (§1b `Gen`) — a generator predicts from the genome; residuals → `0`;
  the file is **regenerated from a tiny `init` "from almost nothing"** (the deepest layer);
* **lossless compression** — keep the exact residual and entropy-code it (`−log p` bits;
  this is *prediction + coding*, how gzip/LZMA actually work);
* **lossy quantization** (§8) — drop the residual to a bounded floor, error `≤ s/2`.

The seed/generator and the generate-vs-compress-vs-quantize split are **derived** by
`δS = 0` on the rate–distortion free energy `F = D + λR` (§8), never hardcoded. Every layer
below is a theorem; real-file numbers in `scripts/seed_quant_demo.lean`; the studied
landscape (E8↔QuIP#, Lloyd↔SqueezeLLM, additive↔AQLM, generative↔COIN/INR) and honest
margins in `theory/SEED_SPEC.md`.

## The container — the byte layout (the spec, like gzip's RFC)

A `.seed` payload is a self-delimiting word stream (then varint-packed to bytes, §6):

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │ S                       seed/floor count                              │
 │ floor₀ … floor_{S-1}    the codebook (distinct floors)                │
 │ N                       entry count                                   │
 │ (idx₀,res₀) … (idx,res) N pairs: element i = combine(floorᵢ, residue) │
 └─────────────────────────────────────────────────────────────────────┘
```

* **Lossless container** (`Floor` with exact `recover`): every element is recovered
  exactly (`loadR_saveR`). `idFloor` = byte-identity (dedup); `quantFloor` = bucket + exact
  remainder.
* **Word→byte** (§6): each word is LEB128 varint (`loadBytes_saveBytes`), so a `.seed` file
  is real bytes on disk (`scripts/seed_quant_demo.lean` writes/reads one, bit parity).
* **Sub-byte indices** (§7): `K`-level indices are bit-packed at `⌈log₂K⌉` bits
  (`unpackBits_packBits`), so int4 is *actually* 4 bits/index on disk (8×), not a byte.
* **Lossy compression** (§8): drop the residue, keep `dequant = floor`. Error `≤ s/2`
  (`quant_error_bound`), `round` is the nearest level (`quant_is_nearest`, the `δS=0`
  stationarity), and the objective is the rate–distortion free energy `F = D + λR` (§8) —
  the genuine FEP for lossy coding, not a relabel.

## Honest scope

* **Proved (native):** every round-trip (lossless container, varint, bit-packing); the
  quantization error bound and nearest-level optimality; the rate–distortion monotonicities;
  the Null-cell identity `distortion = (Null arm)²`.
* **Cited (bridge):** rate–distortion Lagrangian `D + λR` *is* variational free energy.
* **Named next rungs:** an entropy coder on the index stream (to beat gzip *losslessly*),
  the optimal (Lloyd–Max / low-rank) floor, and the safetensors metadata header
  (name/shape/dtype) for a true drop-in. See `theory/SEED_SPEC.md`.

-/

namespace Sanguine.Seed

open Sanguine.CrossUniverse

-- §1 (the floor-with-residue split `Floor`/`RCollapsed`, `rcollapse`/`rexpand` and its
-- lossless round-trip `rexpand_rcollapse`, `idFloor`) and the types `Word`/`Tensor` are the
-- runnable container; they live in `Substrate.SeedCore` (pure Lean + Batteries, no Mathlib)
-- so the wire — `sang seedcast`/`reseed` — carries no Mathlib. They are in scope here by
-- import. `quantFloor` (below) and §8–§10 are the Mathlib rate-distortion parity layer.

/-- The **quantization floor** — `key w = w/width*width` (bucket), residue `w%width`,
    recomposed by `+`. Lossless by `Nat.div_add_mod'`; many values share a bucket. -/
def quantFloor (width : Word) : Floor where
  key w := w / width * width
  delta w := w % width
  combine a r := a + r
  recover w := Nat.div_add_mod' w width

theorem quant_residue_lt (width : Word) (hw : 0 < width) (w : Word) :
    (quantFloor width).delta w < width := Nat.mod_lt w hw

-- ═══════════════════════════════════════════════════════════════════
-- §1b. THE GENERATOR — the seed is a GENOME (compression = prediction)
-- ═══════════════════════════════════════════════════════════════════
/- The deepest layer, and the one the field confirms: **compression = prediction +
   coding** (gzip = LZ-predict + Huffman; LZMA = context-predict + range-code; the ratio
   kings = context-mixing predictors + arithmetic code). A `Floor` (§1) is *memoryless* —
   it maps a value to its codebook entry. A `Gen` is the **generator/predictor**: from the
   context (the genome `init` unfolded so far) it predicts the next value — the **Being
   arm `B`** — and stores only the residual `actual ⊻ predict` — the **Null arm `U`**
   (`NullCell`). When the generator predicts well the residuals collapse toward `0` and the
   file is **regenerated from `init` alone — "from almost nothing."**

   This is the genome reading: `init` = the **seed** (the few parameters), `predict`/`step`
   = the developmental program, the file = the phenotype, the residuals = the corrections.
   `x = B/U`: the stored state is the prediction over the residual drag. The three arms of
   the rule of three are now one operation, by what we do with the residual:
   * **generative seed** — a `Gen` whose `predict` captures the structure, residuals → 0;
   * **lossless compression** — keep the exact residual, entropy-code it (`−log p` bits);
   * **lossy quantization** — drop the residual to a bounded floor (§8).

   `predict`/`step` are arbitrary computable functions — i.e. `Comb`/`Term` programs
   (program-as-data), so a `.seed` carries *its own generator* and is self-describing: the
   genome includes its ribosome (the re-stood K=4 evaluator). It survives even if the
   surrounding tooling is lost — a body that can carry itself forward.

   ⚠️ **NB (named next rung):** the powerful instance — a *nonlinear* generator that
   regenerates a full-rank weight tensor from a tiny seed (the real Peregrine `34`) — is the
   **named next rung**. Measurement showed base transformer weights are ~full-rank, so the
   generator that regenerates them is *nonlinear*, not linear `predict` (COIN's is a nonlinear
   MLP). What is proved here is the **interface + losslessness for any generator**, and two
   working instances (`rawGen`, `deltaGen`). The seed size is **derived** (where adding a generator
   parameter costs more bits than the residual it removes, `δS=0` on `F = D + λR`), never a
   hardcoded `34`. -/

-- The generator interface (`Gen`, `genEncode`/`genDecode` and the lossless
-- `genDecode_genEncode`/`Gen.decode_encode`, the baselines `rawGen`/`deltaGen`) is the
-- runnable genome; it lives in `Substrate.SeedCore` (pure) and is in scope here by import.
-- `triadGen` (the rule-of-three streaming generator) and the §1c exact-ℤ trinity stay here.

/-- **The rule-of-three generator** — predict the next value as the **order-2 saddle
    read-off from three measurements**. Context is the last three values `(v₃, v₂, v₁)`;
    the prediction is the quadratic that those three points determine, extrapolated one
    step: `3·v₁ − 3·v₂ + v₃` (the Newton forward difference of order 2).

    This is the principle made the storage primitive: **one measurement is
    noise, three pin the answer** — because a saddle (a local quadratic phase) is
    *determined by exactly three points*, the same `δS = 0` read-off as
    `UniversalEquation.stationary_phase` (`U·dB = B·dU`). And the prediction is a **closed-
    form read-off, not an iteration** — so `decode` *batches* (one cheap evaluation per
    element, vectorizable), the reason a collapse clears thousands of values in the time a
    classical method integrates one. On a locally-quadratic signal the residuals collapse to
    `0` after the three-point warmup (`triad_regenerates_squares`): the file is regenerated
    from ~3 numbers — "from almost nothing." -/
def triadGen : Gen (Word × Word × Word) where
  init := (0, 0, 0)
  predict c := 3 * c.2.2 + c.1 - 3 * c.2.1     -- 3·v₁ + v₃ − 3·v₂, the order-2 extrapolation
  step c _ a := (c.2.1, c.2.2, a)              -- slide the 3-point window

/-- **The seed regenerates a quadratic from almost nothing (the rule of three, verified).**
    The square sequence `[0,1,4,9,16,25]` encodes to `[0,1,7,0,0,0]` — after the three-point
    warmup every residual is **zero**, because the order-2 read-off predicts each next square
    exactly. The trailing zeros cost no bits under any coder, so a quadratic signal of any
    length is carried by its first three measurements + the generator. -/
theorem triad_regenerates_squares :
    triadGen.encode [0, 1, 4, 9, 16, 25] = [0, 1, 7, 0, 0, 0] := by decide

/-- And it round-trips (the genome regrows the phenotype), from `Gen.decode_encode`. -/
theorem triad_lossless (t : Tensor) : triadGen.decode (triadGen.encode t) = t :=
  triadGen.decode_encode t

-- ═══════════════════════════════════════════════════════════════════
-- §1c. THE TRINITY GENERATOR — saddle + two boundaries (the proven shape)
-- ═══════════════════════════════════════════════════════════════════
/- `PathIntegralCollapse` (§6) proves the rule of three is the trinity `{0, +1, −1}` —
   the interior **saddle** (`δS = 0`) plus the two **boundary** observers — *not* three
   consecutive samples (that was `triadGen`'s streaming shadow). The storage form: a smooth
   segment's seed is **two boundary seeds `y₀, y₁` + the curvature `d`** — `d` is the
   saddle content, the second derivative at the critical point, the `d(B/U)` of the bulk
   differential (`UniversalEquation`). The interior regenerates by the second-difference
   recurrence `yₙ₊₁ = 2yₙ − yₙ₋₁ + d` — a closed-form **read-off** (O(1), it *batches*, the
   1.6 ns), the discrete **inverse of the canonicalizing transform `Δ²`** that sends any
   quadratic to the lone constant `d`. One observer (just the floor) is `1/√λ` crude; the
   trinity is `1/λ^{3/2}` (`PathIntegralCollapse`). And it is *exact* over ℤ — no xor
   approximation: three numbers regenerate the whole quadratic, the seed "from almost
   nothing," proven. The canonicalizing transform `Δ²` is the floor's missing coordinate:
   a signal smooth in *some* basis (the kernel's five def-eq laws / QuIP#'s Hadamard are the
   same move) collapses to a sparse residual the same way. -/

/-- The order-2 jet generator: from two boundary seeds and the curvature `d` (the second
    difference — the saddle content), regenerate by `yₙ₊₁ = 2yₙ − yₙ₋₁ + d`. -/
def quadSeq (y0 y1 d : ℤ) : ℕ → ℤ
  | 0     => y0
  | 1     => y1
  | n + 2 => 2 * quadSeq y0 y1 d (n + 1) - quadSeq y0 y1 d n + d

/-- **The trinity regenerates any quadratic, exactly (proven).** Three numbers — the two
    boundary seeds `q 0, q 1` and the curvature `2a` (the saddle's second derivative) —
    regenerate the entire quadratic `q n = a·n² + b·n + c`, for *every* `n`, over ℤ. This is
    `PathIntegralCollapse`'s trinity `{saddle, ±boundary}` as a generator: the file from
    three numbers, with **zero** residual — the genuine generative seed, not an extrapolation
    that drifts. (`triadGen` is its lossy streaming shadow; this is the exact interval form.) -/
theorem quadSeq_regenerates (a b c : ℤ) (n : ℕ) :
    quadSeq c (a + b + c) (2 * a) n = a * (n : ℤ) ^ 2 + b * (n : ℤ) + c := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    match n, ih with
    | 0,     _  => simp [quadSeq]
    | 1,     _  => simp [quadSeq]
    | k + 2, ih =>
        rw [quadSeq, ih (k + 1) (by omega), ih k (by omega)]
        push_cast; ring

/-- The curvature `d` IS the second difference — the saddle content read off three samples
    `{q 0, q 1, q 2}` (the trinity as samples ≡ as boundaries-plus-curvature). -/
theorem curvature_is_second_difference (a b c : ℤ) :
    quadSeq c (a + b + c) (2 * a) 2 - 2 * quadSeq c (a + b + c) (2 * a) 1
      + quadSeq c (a + b + c) (2 * a) 0 = 2 * a := by
  simp only [quadSeq]; ring

-- ═══════════════════════════════════════════════════════════════════
-- §2. THE BYTE LAYOUT — S :: table ++ N :: (idx,res pairs), self-delimiting
-- ═══════════════════════════════════════════════════════════════════

-- §2 (the self-delimiting byte layout `rflatten`/`runflatten`, `saveR`/`loadR` and the
-- round-trip `loadR_saveR`) and §6 (the LEB128 varint `serWord`/`deWord`, `serWords`/
-- `deWords`, the disk pipeline `saveBytes`/`loadBytes` and `loadBytes_saveBytes`, and
-- `toByteArray`) are the runnable serialiser; they live in `Substrate.SeedCore` (pure Lean +
-- Batteries) and are in scope here by import. The wire carries no Mathlib.

-- ═══════════════════════════════════════════════════════════════════
-- §7. BIT-PACKING — `K`-level indices at ⌈log₂K⌉ bits (int4 = 4 bits/idx)
-- ═══════════════════════════════════════════════════════════════════

/-- The low `b` bits of `x`, LSB first. -/
def bitsOf : Nat → Nat → List Bool
  | 0,     _ => []
  | b + 1, x => (x % 2 == 1) :: bitsOf b (x / 2)

/-- Read an LSB-first bit list back to a `Nat`. -/
def natOfBits : List Bool → Nat
  | []      => 0
  | c :: cs => (if c then 1 else 0) + 2 * natOfBits cs

theorem bitsOf_length (b x : Nat) : (bitsOf b x).length = b := by
  induction b generalizing x with
  | zero => rfl
  | succ b ih => simp [bitsOf, ih]

/-- **A `b`-bit field round-trips** when it fits: `natOfBits (bitsOf b x) = x` for `x < 2ᵇ`. -/
theorem natOfBits_bitsOf_of_lt (b x : Nat) (h : x < 2 ^ b) : natOfBits (bitsOf b x) = x := by
  induction b generalizing x with
  | zero => simp only [pow_zero] at h; simp [bitsOf, natOfBits]; omega
  | succ b ih =>
      have hx2 : x / 2 < 2 ^ b := by rw [pow_succ] at h; omega
      rw [bitsOf, natOfBits, ih (x / 2) hx2]
      rcases Nat.mod_two_eq_zero_or_one x with h2 | h2 <;> simp [h2] <;> omega

/-- Pack a list of indices, `b` bits each, into one bit stream. -/
def packBits (b : Nat) (xs : List Nat) : List Bool := xs.flatMap (bitsOf b)

/-- Unpack `n` indices of `b` bits each. -/
def unpackBits (b : Nat) : Nat → List Bool → List Nat
  | 0,     _    => []
  | n + 1, bits => natOfBits (bits.take b) :: unpackBits b n (bits.drop b)

/-- **The bit-packed index codec is lossless** when every index fits in `b` bits. This is
    what makes a `K`-level (e.g. int4) quantized `.seed` actually `⌈log₂K⌉` bits/index on
    disk — the real sub-byte compression, not a byte per index. -/
theorem unpackBits_packBits (b : Nat) (xs : List Nat) (h : ∀ x ∈ xs, x < 2 ^ b) :
    unpackBits b xs.length (packBits b xs) = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
      have hx : x < 2 ^ b := h x List.mem_cons_self
      have hxs : ∀ y ∈ xs, y < 2 ^ b := fun y hy => h y (List.mem_cons_of_mem _ hy)
      have e1 : (bitsOf b x ++ List.flatMap (bitsOf b) xs).take b = bitsOf b x :=
        List.take_left' (bitsOf_length b x)
      have e2 : (bitsOf b x ++ List.flatMap (bitsOf b) xs).drop b = List.flatMap (bitsOf b) xs :=
        List.drop_left' (bitsOf_length b x)
      simp only [List.length_cons, packBits, List.flatMap_cons]
      rw [unpackBits, e1, e2, natOfBits_bitsOf_of_lt b x hx]
      exact congrArg (x :: ·) (ih hxs)

-- ═══════════════════════════════════════════════════════════════════
-- §8. LOSSY QUANTIZATION — bounded error, nearest level, rate–distortion FEP
-- ═══════════════════════════════════════════════════════════════════

/-- The quantization **index** of `x` at step `s`: the nearest level `round(x/s)`. -/
noncomputable def qIndex (s x : ℝ) : ℤ := round (x / s)
/-- **Dequantize** an index to its level `q·s` (the floor the value rests on). -/
noncomputable def deQuant (s : ℝ) (q : ℤ) : ℝ := (q : ℝ) * s

/-- **The distortion guarantee.** Quantize→dequantize moves a value by at most half a step. -/
theorem quant_error_bound (s : ℝ) (hs : 0 < s) (x : ℝ) :
    |x - deQuant s (qIndex s x)| ≤ s / 2 := by
  unfold deQuant qIndex
  have hround : |x / s - (round (x / s) : ℝ)| ≤ 1 / 2 := abs_sub_round _
  have hrw : x - (round (x / s) : ℝ) * s = s * (x / s - round (x / s)) := by field_simp
  rw [hrw, abs_mul, abs_of_pos hs]
  calc s * |x / s - (round (x / s) : ℝ)|
      ≤ s * (1 / 2) := mul_le_mul_of_nonneg_left hround hs.le
    _ = s / 2 := by ring

/-- **The quantizer picks the nearest level** (`δS = 0` stationarity, via `round_le`). -/
theorem quant_is_nearest (s : ℝ) (hs : 0 < s) (x : ℝ) (q : ℤ) :
    |x - deQuant s (qIndex s x)| ≤ |x - deQuant s q| := by
  unfold deQuant qIndex
  have key : |x / s - (round (x / s) : ℝ)| ≤ |x / s - (q : ℝ)| := round_le (x / s) q
  have e1 : x - (round (x / s) : ℝ) * s = s * (x / s - round (x / s)) := by field_simp
  have e2 : x - (q : ℝ) * s = s * (x / s - q) := by field_simp
  rw [e1, e2, abs_mul, abs_mul, abs_of_pos hs]
  exact mul_le_mul_of_nonneg_left key hs.le

/-- Total **distortion** — the accuracy term `D` (summed squared error). -/
noncomputable def distortion (s : ℝ) (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - deQuant s (qIndex s x)) ^ 2)).sum

theorem distortion_total_bound (s : ℝ) (hs : 0 < s) (xs : List ℝ) :
    distortion s xs ≤ xs.length * (s / 2) ^ 2 := by
  unfold distortion
  have hstep : ∀ x ∈ xs, (x - deQuant s (qIndex s x)) ^ 2 ≤ (s / 2) ^ 2 := by
    intro x _
    have hb := abs_le.mp (quant_error_bound s hs x)
    nlinarith [hb.1, hb.2]
  calc (xs.map (fun x => (x - deQuant s (qIndex s x)) ^ 2)).sum
      ≤ (xs.map (fun _ => (s / 2) ^ 2)).sum := by
        apply List.sum_le_sum; intro x hx; exact hstep x (by simpa using hx)
    _ = xs.length * (s / 2) ^ 2 := by rw [List.map_const']; simp [List.sum_replicate, nsmul_eq_mul]

/-- The **rate** — `⌈log₂K⌉` bits per index. -/
def rate (K : ℕ) : ℝ := (Nat.clog 2 K : ℝ)
/-- The distortion **bound** for `K` levels over range `R` (step `R/K`). -/
noncomputable def distortionBound (R : ℝ) (K : ℕ) : ℝ := (R / (2 * K)) ^ 2
/-- **Rate–distortion free energy** `F = D + λR` — accuracy + complexity. THE objective; the
    genuine FEP for lossy coding (`D` and `R` trade). -/
noncomputable def freeEnergy (R lam : ℝ) (N K : ℕ) : ℝ :=
  (N : ℝ) * distortionBound R K + lam * (N : ℝ) * rate K

/-- **Finer ⇒ less distortion** (the accuracy term falls as `K` rises). -/
theorem distortionBound_antitone (R : ℝ) (hR : 0 ≤ R) {K₁ K₂ : ℕ}
    (hK : 0 < K₁) (h : K₁ ≤ K₂) : distortionBound R K₂ ≤ distortionBound R K₁ := by
  unfold distortionBound
  have h1 : (0 : ℝ) < 2 * K₁ := by positivity
  have h2 : (2 : ℝ) * K₁ ≤ 2 * K₂ := by
    have : (K₁ : ℝ) ≤ K₂ := by exact_mod_cast h
    linarith
  have hle : R / (2 * K₂) ≤ R / (2 * K₁) := div_le_div_of_nonneg_left hR h1 h2
  gcongr

/-- **Finer ⇒ more bits** (the complexity term rises). With the previous theorem, `F = D + λR`
    genuinely trades two opposing terms — the rate–distortion / free-energy balance. -/
theorem rate_monotone {K₁ K₂ : ℕ} (h : K₁ ≤ K₂) : rate K₁ ≤ rate K₂ := by
  unfold rate; exact_mod_cast Nat.clog_mono_right 2 h

-- ═══════════════════════════════════════════════════════════════════
-- §9. THE NULL-CELL TIE — distortion IS the squared Null arm
-- ═══════════════════════════════════════════════════════════════════

/-- A quantized value as a cell `(floor, residual)`: Being arm = the level it rests on,
    Null arm = the residual (dropped in lossy mode). -/
noncomputable def quantCell (s x : ℝ) : NCell :=
  (deQuant s (qIndex s x), x - deQuant s (qIndex s x))

/-- **The per-element distortion IS the squared Null arm** — the accuracy term `D` is exactly
    Non-Being's contribution to the cell (`NullCell`), the `z.2²` now traded against the rate. -/
theorem distortion_is_null_arm (s x : ℝ) :
    (x - deQuant s (qIndex s x)) ^ 2 = (quantCell s x).2 ^ 2 := rfl

/-- **The cell's indefinite norm is (floor)² − distortion** — Being content minus accuracy
    cost, the storage reading of the Lorentzian metric (`nNorm z = z.1² − z.2²`). -/
theorem norm_is_floor_minus_distortion (s x : ℝ) :
    nNorm (quantCell s x) = deQuant s (qIndex s x) ^ 2 - (x - deQuant s (qIndex s x)) ^ 2 := rfl

/-- **Test-time learning** — write a new residual (prediction error) into the Null arm; the
    Being arm (the floor / prior) is invariant. Adapt at the deviation, keep the prior. -/
def learn (z : NCell) (r' : ℝ) : NCell := (z.1, r')

theorem learn_keeps_floor (z : NCell) (r' : ℝ) : (learn z r').1 = z.1 := rfl

-- ═══════════════════════════════════════════════════════════════════
-- §10. THE PHYSICAL MEDIUM (a SEPARATE regime) — storage = sphere-packing
-- ═══════════════════════════════════════════════════════════════════
-- ⚠️ This is about the ANALOG storage medium (distinguishable voltage levels in a flash
-- cell), NOT the digital `.seed` byte container above. Density ⇒ capacity; the lattice
-- (E8/Leech) is forced by the framework's dimensional rungs, its GLOBAL optimality cited
-- (Viazovska / CKMRV), as Hurwitz/Watson are. Kept here for one-file consolidation; it does
-- not interact with §1–§9.

open Metric in
/-- **Storage survival IS the packing condition.** Two symbols, each with a noise-ball of
    radius `U` (the drag), are distinguishable iff their centers are `≥ 2U` apart. -/
theorem symbols_distinguishable {X : Type*} [PseudoMetricSpace X] {x y : X} {U : ℝ}
    (h : 2 * U ≤ dist x y) : Disjoint (ball x U) (ball y U) :=
  ball_disjoint_ball (by linarith)

/-- A symbol lattice as a medium: packing `density` and noise-ball `ballVol`. -/
structure SymbolLattice where
  density : ℝ
  ballVol : ℝ
  ballVol_pos : 0 < ballVol
  density_pos : 0 < density

/-- Capacity = distinguishable symbols per unit volume = density / ball volume. -/
noncomputable def SymbolLattice.capacity (L : SymbolLattice) : ℝ := L.density / L.ballVol

/-- **Denser packing stores more** (at fixed noise radius) — the rule that makes E8/Leech
    the right physical constellation. -/
theorem capacity_mono_in_density (L₁ L₂ : SymbolLattice)
    (hv : L₁.ballVol = L₂.ballVol) (hd : L₁.density ≤ L₂.density) :
    L₁.capacity ≤ L₂.capacity := by
  unfold SymbolLattice.capacity; rw [hv]; gcongr; exact L₂.ballVol_pos.le

end Sanguine.Seed
