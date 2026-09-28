import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Nat.Bitwise
import Mathlib.Data.ZMod.Basic
import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Tactic.Common
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# The Radix Descent — two keys always separate, and that is NOT a probe-length bound
@name Why Two Keys Never Share Every Digit
@child When two keys want the same box, you don't pile them up — you look at the next digit of their address, and the next, until the digits disagree. They always do within a fixed number of digits, so no two keys can chase each other forever. What that does NOT stop: a bully who parks one letter on each place your letter would look. He pays one letter per place and it works, so "the pile can never get tall" was the wrong reading of this file — corrected 2026-08-06. What he does NOT get is a discount: every different place your letter has to look, he pays for with one more letter parked in one more hole — 11 letters bought 11 places, measured. And only 256 letters in the whole world can share your letter's entire route. Your letter also only ever has a fixed number of places to look — it runs out of digits — so if any of them is free it finds one within that many looks. The bully's win is never a long walk; it is taking every place at once — and the real table beats even that, by walking the holes one after another from where it stopped, never the same hole twice, so one empty hole anywhere in the room is enough. This file also answers the calmer question: if nobody is attacking you, how full does the table get on average?
@scientist The probe chain of the `PHI`-fold table is the radix-`2^b` digit sequence of `k·PHI`. The fold is injective (PHI a unit ⇒ no zero divisors with a unit factor — the Being-cell `NullCell.phase_no_zero_divisors`), so distinct keys' products differ in some digit and SEPARATE within `⌈W/b⌉` of them — worst-case, over every key set including an adversarial one. Separation is a statement about a PAIR; a probe length is a function of the OCCUPIED SET, and the two do not join — `AggregationBound.descent_can_be_wholly_blocked` exhibits `D` keys, none of them `k`, whose home slots cover every address `k`'s descent will try, with `≤ D` of the `2^b` slots occupied. `home_balanced` (the same `PHI`-unit root) shows the fold is *exactly* uniform, which promotes the classical average-case numbers from plausible to provable on the same fold — collision probability, expected load, occupancy, and `probe_set_fully_occupied_prob_pow`: the blocked-descent mass is `≤ α^S.card`, `α = n/2^b`, over i.i.d.-uniform key sets — a union bound over the witness maps `S → Fin n`, sharpening the single-slot `probe_set_fully_occupied_prob`'s `≤ α`. Worst-case the LENGTH is bounded too, but by digit exhaustion rather than separation: `plen_addr_le_radix`, `plen ≤ D` for every key and every occupied set (`plen_seqLin_le_radix`, `G+D−1`, for the shipped `seqLin`), because past `W ≤ b·j` every address is slot `0` (`addr_eq_zero_of_exhausted`). It bounds a walk that TERMINATES, and termination is exactly what `descent_can_be_wholly_blocked` removes. What else survives worst-case is the EXCHANGE RATE and the pile: `plen_image_card_le` — the distinct slots a forced probe crosses are slots the occupier already owns, so blocking is bought one occupied slot per distinct address and never amplified, and `plen_le_card` makes it one per STEP where the walk does not revisit (measured 1:1 — 11 aimed keys at 0.067% load, worst probe 12) — and `window_pile_card`, `2^(W−b·F)` keys per whole window set, 256 at the shipped `W = 64, b = 14`. None of it is keyed — `PHI` is the public constant `0x9E3779B97F4A7C15`. The shipped table's terminal sweep past the `D` windows is `GroupProbe.seqSweep`: `GroupProbe.seqSweep_terminates` proves the walk stops on any table below full with no `∃ j ∉ O` hypothesis at all, at `plen ≤ (G + F − 1) + O.card`, so what the crafted key set takes is bought back and only the `Θ(n²)` ceiling remains.
@engineer `fold_injective` (mul by a `ZMod`-unit is injective); `digits_separate` (distinct bounded words differ in some digit < D); `chain_separates_within_radix` (worst-case ALIASING, `D = W/b`); `linear_prefix_chain_separates` (the shipped probe with its linear prefix, `D+G−1`); `seqLin_lt` (that probe's codomain); `homeSlot_eq_addr_top` (the shipped `home` IS digit `D−1`); `home_balanced`/`plen_mono` (the load lever); `plen_image_card_le`/`plen_le_card` (the no-amplification bound — a forced probe costs one occupied slot per distinct address, and `addr_not_injOn` is the guard on the one-per-step form); `addr_eq_zero_of_exhausted`/`plen_addr_le_radix`/`plen_seqLin_le_radix`/`addr_not_injOn_of_exhausted` (the LENGTH bound with no `Set.InjOn`: the digits run out at `D`, so a terminating walk stops by step `D`, `G+D−1` for `seqLin`); `windows_determine_prefix`/`consumed_prefix_lt`/`window_pile_card`/`shipped_pile_card` (the pile at key granularity: `2^(W−b·F)` keys share a whole window set, 256 at `W = 64, b = 14, F = 4`); `ProbabilisticBounds.expected_load`/`expected_occupied_slots` (the average-case envelope); `ProbabilisticBounds.probe_set_fully_occupied_prob`/`ProbabilisticBounds.probe_set_fully_occupied_prob_pow` (a whole probe-address set occupied: mass `≤ n/2^b`, sharpened to `≤ (n/2^b)^S.card` by the union bound over witness maps, `ProbabilisticBounds.covering_needs_as_many_keys` making its `S.card ≤ n` free); `ProbabilisticBounds.per_pair_double_collision_prob` (the concrete two-flaw spinor pair `SpinorReadout` reads).

This file is the bridge `HashTrinityCore.lean` does not carry. There the hashmap is the abstract
total `K → V` — no array, no probe, no collision. A concrete table realizes that total function in a
finite power-of-two array, and on a coordinate clash it `descend`s: it re-addresses by a WIDER window
of `k·PHI`. The question `RadixDescent` closes: can two distinct keys alias at every window the
descent reads? **No** — and the reason is injectivity, not hardness. The question the SEPARATION
bound does not close, and the header claimed until 2026-08-06 that it did: how long a probe runs.
That is a function of the OCCUPIED SET, and `AggregationBound.descent_can_be_wholly_blocked` refutes
the universal reading — `D` crafted keys, at `≤ D` of `2^b` slots, occupy every address a chosen
key's descent will try. The length is nonetheless bounded, by a different fact: the digits RUN OUT
(`addr_eq_zero_of_exhausted`), so `plen_addr_le_radix` gives `plen ≤ D` for every key and every
occupied set. What the crafted set takes away is not a short walk — it is the free slot, hence the
walk's termination, hence any `plen` at all.

`coord ↦ coord·PHI` over `ZMod (2^W)` is injective **because `PHI` is a unit** (odd). A full collision
`k₁·PHI = k₂·PHI` is `(k₁ − k₂)·PHI = 0` with `k₁ − k₂ ≠ 0` and `PHI` a unit — a *zero divisor with a
unit factor*, which cannot exist. That is the Being-cell face of the trinity: where `/` is total there
are no zero divisors (`Substrate/Algebra/NullCell/NullCell.lean : phase_no_zero_divisors`,
`phase_invertible`). The fold lives in the Being cell; the collision the adversary needs lives only on
the null cone, which the fold never touches.

Given injectivity, the separation is a counting fact. The probe reads the radix-`2^b` digits of
`p = k·PHI` (`digit b j`; the model's digit `0` is the LOW window and the shipped `home` is the TOP
one, `digit b (D−1)` by `homeSlot_eq_addr_top` — every theorem here is `∃ j < D`, so which end the
walk starts from is immaterial). Two distinct products differ in some bit, hence in some digit
`j < D = ⌈W/b⌉`; the descent reaches that digit and the keys land in different slots. So **any** two
keys separate within `D` steps — the radix length, a constant fixed by the word, *independent of how
many keys exist*. No pile: no two keys chase each other forever. NOT no flood: separation says
nothing about how many slots are already taken. The empirical 1.2× "blowup" is this `D`; the 133×
we once saw was a linear-probe bug (a real bucket), not the descent.

What the blocked descent costs is bounded even though its length is not: `plen_image_card_le` says
the distinct slots a probe crosses are slots the occupier already owns, so a forced probe is bought
one occupied slot per distinct address and never amplified. Measured against the shipped table
(`fold/hashtrinity/tests/hashdos.rs`, c94526c): 11 aimed keys — 11 of 16384 slots, 0.067% load —
force worst probe 12, which is 11 occupied crossings then the free slot; 11 unaimed keys leave worst
probe 1. That is the theorem's rate, exactly. And the pile a flood can put on one window set is
`2^(W−b·F)` keys (`window_pile_card`), 256 at the shipped `W = 64, b = 14` — set by the word, not by
`n`, which is why 12,000 keys aimed at one window measure worst probe < 128 rather than 12,000.

`namespace ProbabilisticBounds` (below `RadixDescent`) trades the worst-case's adversary-freedom for
the **uniform-hashing assumption** (keys placed i.i.d. uniformly — CLRS §11.2's textbook idealization)
and gets the sharper average-case numbers on the *same* fold: `home_balanced`'s exactness, not an
assumed distribution, is what makes them provable outright. It also supplies the concrete two-flaw
address (`digit1`, the window beneath `homeSlot`) that `SpinorReadout.concrete_spinor_depth_two`
discharges its depth-2 hypothesis with.

## What is proved here, over which key set, and where a machine enters

* **Proved (native):** the separation bound, for EVERY key set including an adversarially chosen
  one — two distinct keys separate within `D = ⌈W/b⌉` descent steps
  (`chain_separates_within_radix`), and within `D + G − 1` for the shipped probe's linear prefix
  (`linear_prefix_chain_separates`). Quantifying over all key sets is strictly stronger than an
  average-case statement, and it rests on injectivity (`fold_injective`, `bits_separate`,
  `digits_separate`), never on hardness, secrecy, or a seed. It bounds ALIASING — how long two keys
  can share addresses — and nothing else.
* **Retired:** *"no craftable input walks a long chain"* AS A READING OF THE SEPARATION BOUND,
  offered here until 2026-08-06 as what makes this the HashDOS law. The derivation is refuted:
  `AggregationBound.separation_does_not_bound_probe` gives a distinct blocking key at every single
  step, and `AggregationBound.descent_can_be_wholly_blocked` gives `D` keys — none of them the
  victim, `≤ D` of the `2^b` slots occupied — whose home slots cover the victim's ENTIRE probe
  address set (`shipped_probe_can_be_wholly_blocked` is the same for `seqLin`, the probe as
  shipped). A probe length is a function of the occupied set; separation constrains a pair. The
  sentence itself is not refuted and the correction is recorded below: `plen_addr_le_radix` proves
  `plen ≤ D` from digit exhaustion, not from separation, and what the craft buys is a walk with no
  free slot at all — no `plen` — never a long one.
* **Proved (native):** the fold is EXACTLY uniform on the home window — every slot's fibre has the
  same cardinality (`trunc_fiber_card`, `home_balanced`) — and probe length is monotone in the
  occupied set (`plen_mono`).
* **Proved (native):** the no-amplification bound, for EVERY occupied set including an adversarial
  one — the DISTINCT addresses a probe crosses before it stops number at most `O.card`
  (`plen_image_card_le`), with no hypothesis at all; and `plen ≤ O.card` outright when the probe
  does not revisit an address (`plen_le_card`, hypothesis `Set.InjOn`). The contrapositive is the
  security property: forcing a probe of length `L` costs the adversary `L` occupied slots
  (`AggregationBound.probe_length_forces_occupied_slots`). The `Set.InjOn` hypothesis is
  UNDISCHARGED here and is not dischargeable for this probe — `addr_not_injOn`: the key `0` folds to
  `0`, whose every digit is `0`, so its probe sits on slot `0` at every step, and
  `addr_not_injOn_of_exhausted` shows every OTHER key's walk repeats too once the digits run out.
  A repeat is a discount for the attacker, so `plen_image_card_le` is the form that carries the
  reading — and the repeats are also what `plen_addr_le_radix` turns into the length bound.
* **Proved (native):** the pile at key granularity — exactly `2^(W − b·F)` keys agree with a given
  key on all `F` windows the descent consumes (`window_pile_card`, hypothesis `b·F ≤ W`, riding
  `home_balanced` at width `b·F` through `windows_determine_prefix`), and `2^(64−56) = 256` at the
  shipped `W = 64, b = 14, F = 4` (`shipped_pile_card`). `chain_separates_within_radix`'s `W = b·D`
  would force the pile to `2^0 = 1`; the shipped table has `14 ∤ 64`, so the live case is
  `GroupProbe.group_descent_dichotomy`'s `b·F ≤ W` and the live pile is 256 — bounded by the word,
  never by `n`.
* **Proved (native):** the average-case envelope in `ProbabilisticBounds`, over the stated sample
  space of `n` keys drawn i.i.d. uniformly (`Keys`) — per-pair collision probability
  (`per_pair_collision_prob`), tag false-positive rate (`tag_false_positive_rate`), the birthday
  bound (`birthday_bound`), expected load (`expected_load`), expected occupied slots
  (`expected_occupied_slots`), the displacement tail (`displacement_tail_bound`), the mass of key
  sets that occupy a WHOLE probe-address set (`probe_set_fully_occupied_count` /
  `probe_set_fully_occupied_prob`, `≤ α = n/2^b`), and the depth-2 pair `homeSlot`/`digit1` whose
  double-collision mass is exact (`wide_window_split`,
  `pair_windows_eq_wide_window`, `per_pair_double_collision_prob`). `home_balanced` is what makes
  these outright rather than assumed OF THE HASH; the i.i.d.-uniform draw of the KEYS is still a
  modelling hypothesis, and it is exactly the hypothesis the separation bound above does not need.
* **Proved (Mathlib):** the arithmetic is `ZMod (2^W)` unit theory plus `Fintype`/`Finset`
  counting; `home_resident_ceiling_at_two_fifths` is `norm_num` on `homeResidentCeiling 1024 410`,
  so the "≈ 0.82" home-hit ceiling is computed here, not quoted from a run.
* **Proved (riding):** that a unit has no zero divisor — `NullCell.phase_no_zero_divisors` and
  `NullCell.phase_invertible`, the Being-cell face the injectivity is read off.
* **Proved (native):** the probe LENGTH bound, with no `Set.InjOn`, no distribution and no load
  hypothesis — `plen_addr_le_radix`: for `W ≤ b·D`, every key and every occupied set,
  `plen ≤ D`; and `plen_seqLin_le_radix`: `plen ≤ G + D − 1` for the shipped `seqLin`. What closes
  it is that the digits RUN OUT — `addr_eq_zero_of_exhausted`, past `W ≤ b·j` every key's address
  is slot `0` — so the walk has at most `D` distinct addresses and is constant after them, and a
  walk still running at step `D + 1` would need slot `0` both occupied (step `D` was blocked) and
  free (it stopped there). The repeats that defeated `plen_le_card` are what proves it, and they
  are universal rather than the key `0`'s alone (`addr_not_injOn_of_exhausted`). It bounds a walk
  GIVEN it terminates and nothing else: `AggregationBound.descent_can_be_wholly_blocked` refutes
  `plen < D`, and when its `O` also holds slot `0` the hypothesis `∃ j, addr b j PHI k ∉ O` fails
  and there is no `plen` to bound. The located defect is TERMINATION, not length.
* **Proved (native):** the sharper average-case coverage bound —
  `probe_set_fully_occupied_prob_pow`: `#{κ : ∀ s ∈ S, 0 < load b PHI κ s} / (2^W)^n ≤ α^S.card`,
  `α = n/2^b`, hypothesis `S.card ≤ n`, and that hypothesis is free — `covering_needs_as_many_keys`
  proves `n` keys cover at most `n` slots, so past it the event is empty. A union bound over the
  witness maps `S → Fin n` (`probe_set_fully_occupied_count_pow`), each term counted by
  `fiber_lift_subset`; a covering placement's witness map is injective because one key has one
  home, and a non-injective map pins one key to two slots and counts `0`. Covering all of `S` is
  now counted as harder: `probe_set_fully_occupied_prob`'s `≤ α` was the single-slot Markov bound
  read through one member of `S`. At the operating `α ≈ 0.4` over the shipped table's `D = 5`
  windows (`W = 64, b = 14`) that is `0.4^5 = 0.01024` where the single-slot read permitted `0.4`. Still a MASS over i.i.d.-uniform
  key sets, never a bound over every key set.
* **Proved (riding):** a length bound for the SHIPPED probe past its `D` windows — the `Open:`
  recorded here until 2026-08-06. `plen_addr_le_radix` bounds the model's walk, whose addresses are
  exhausted at `D`; the shipped `Table::descend` (`hashtrinity/src/lib.rs`) falls back to a terminal
  sweep there, and `GroupProbe.seqSweep` is that probe with all three branches.
  `GroupProbe.plen_seqSweep_le_of_not_full` gives `plen ≤ (G + F − 1) + O.card` with the table's
  not-fullness as the only hypothesis — no `∃ j ∉ O`, because `GroupProbe.seqSweep_terminates`
  proves it from `GroupProbe.sweep_period_no_repeat`'s injectivity. The honest worst-case build
  ceiling is still `AggregationBound.build_bounded_by_n_squared`'s `n·n`
  (`AggregationBound.build_bounded_by_sweep` restates it at `n·(G + F − 1 + n)` with the
  termination hypothesis discharged), the same `Θ(n²)` a flooded `hash & mask` pays. What the fold
  moves is the constant and the average case; what the sweep moves is termination.
* **Anchor:** that `homeSlot` and `digit1` ARE the shipped table's two probe windows,
  that the measured 1.2× probe blowup IS this `D`, and that the 133× once seen was a linear-probe
  bug rather than a descent. Those are readings of runs onto the model; no theorem here measures a
  table, a cache line, or a probe's wall time.
-/

namespace Sanguine
namespace RadixDescent

-- ═══════════════════════════════════════════════════════════════════
-- THE FOLD IS INJECTIVE — the Being-cell face (no zero divisor with a unit)
-- ═══════════════════════════════════════════════════════════════════

/-- **The fold is injective.** Multiplication by a unit is injective even in a ring *with* zero
    divisors (`ZMod (2^W)` has plenty): a clash `a·u = b·u` cancels by `u⁻¹`. A full coordinate
    collision would need `(a−b)·u = 0` with `a−b ≠ 0` and `u` a unit — a zero divisor with a unit
    factor, which does not exist. This is the Being-cell `NullCell.phase_no_zero_divisors` /
    `phase_invertible` read in `ZMod`: where `/` by `PHI` is total, the fold separates keys. -/
theorem fold_injective {N : ℕ} (u : (ZMod N)ˣ) :
    Function.Injective (fun k : ZMod N => k * (u : ZMod N)) := by
  intro a b h
  have := congrArg (· * (↑u⁻¹ : ZMod N)) h
  simpa [mul_assoc] using this

-- ═══════════════════════════════════════════════════════════════════
-- THE RADIX DIGITS — the probe addresses
-- ═══════════════════════════════════════════════════════════════════

/-- The `j`-th base-`2^b` digit of `p`: the window `[b·j, b·j + b)` of its bits. `descend j` reads
    digit `j`. Digit `0` is the LOW window; the shipped `home` reads the TOP one, `digit b (D−1)`
    (`homeSlot_eq_addr_top`) — every theorem here is `∃ j < D`, so the walk's direction is immaterial. -/
def digit (b j p : ℕ) : ℕ := p / 2 ^ (b * j) % 2 ^ b

/-- **A digit's bits ARE the word's bits.** Bit `r < b` of digit `j` is bit `b·j + r` of `p`. So a
    digit equality is a window of bit equalities — the link between "same digit" and "same bits". -/
theorem digit_testBit (b j p r : ℕ) (hr : r < b) :
    (digit b j p).testBit r = p.testBit (b * j + r) := by
  unfold digit
  rw [Nat.testBit_mod_two_pow, decide_eq_true hr, Bool.true_and,
      ← Nat.shiftRight_eq_div_pow, Nat.testBit_shiftRight, Nat.add_comm]

-- ═══════════════════════════════════════════════════════════════════
-- SEPARATION — distinct bounded words differ in some digit < D
-- ═══════════════════════════════════════════════════════════════════

/-- **Distinct bounded words differ in some bit below the width.** `p ≠ q` both `< 2^W` ⇒ they
    disagree at some bit `i < W` (bits `≥ W` are `0` for both, so the difference is below `W`). -/
theorem bits_separate {W p q : ℕ} (hpq : p ≠ q) (hp : p < 2 ^ W) (hq : q < 2 ^ W) :
    ∃ i < W, p.testBit i ≠ q.testBit i := by
  by_contra h
  push Not at h
  apply hpq
  apply Nat.eq_of_testBit_eq
  intro i
  rcases lt_or_ge i W with hi | hi
  · exact h i hi
  · have hpow : (2 : ℕ) ^ W ≤ 2 ^ i := Nat.pow_le_pow_right (by norm_num) hi
    rw [Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hp hpow),
        Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hq hpow)]

/-- **Distinct words separate within `D` digits.** With width `W = b·D`, two distinct words
    `< 2^(b·D)` differ in some digit `j < D`. The differing bit `i` lives in digit `i / b`, and
    that digit must differ (else its bit `i % b` would agree). The descent reaches digit `j` in `j`
    steps, so the chain that separates them has length `< D` — the radix length. -/
theorem digits_separate {b D p q : ℕ} (hb : 0 < b) (hpq : p ≠ q)
    (hp : p < 2 ^ (b * D)) (hq : q < 2 ^ (b * D)) :
    ∃ j < D, digit b j p ≠ digit b j q := by
  obtain ⟨i, hiW, hbit⟩ := bits_separate hpq hp hq
  refine ⟨i / b, Nat.div_lt_of_lt_mul hiW, ?_⟩
  intro hdig
  apply hbit
  have hr : i % b < b := Nat.mod_lt _ hb
  have key : (digit b (i / b) p).testBit (i % b) = (digit b (i / b) q).testBit (i % b) := by
    rw [hdig]
  rw [digit_testBit b (i / b) p (i % b) hr, digit_testBit b (i / b) q (i % b) hr,
      Nat.div_add_mod] at key
  exact key

-- ═══════════════════════════════════════════════════════════════════
-- THE HEADLINE — any two keys separate within the radix length, not n
-- ═══════════════════════════════════════════════════════════════════

/-- A digit is a residue below the radix — the codomain bound the linear-prefix walk works in. -/
theorem digit_lt (b j p : ℕ) (_hb : 0 < b) : digit b j p < 2 ^ b :=
  Nat.mod_lt _ (by positivity)

/-- The probe address of `k` at step `j`: the `j`-th radix-`2^b` digit of the fold `k·PHI`. -/
def addr (b j : ℕ) {W : ℕ} (PHI : (ZMod (2 ^ W))ˣ) (k : ZMod (2 ^ W)) : ℕ :=
  digit b j ((k * (PHI : ZMod (2 ^ W))).val)

/-- **The radix-descent bound (closure of `descend`).** For any width `W = b·D` and any unit fold
    `PHI`, two *distinct* keys have probe addresses that differ at some step `j < D`. So the descent
    separates them within `D = ⌈W/b⌉` steps — the radix length, a **constant fixed by the word**,
    not the number of keys: there is no bucket to pile into, and a full collision (a key with the
    same fold) cannot exist — it would be a zero divisor with a unit factor. What it does NOT say,
    and what this docstring said until 2026-08-06: that no adversary can build a chain longer than
    `D`. A chain's length is a function of the OCCUPIED SET —
    `AggregationBound.descent_can_be_wholly_blocked` fills every one of a key's `D` addresses with
    `D` crafted keys, and this separation bound holds of every pair among them. -/
theorem chain_separates_within_radix {b D W : ℕ} (hb : 0 < b) (hW : W = b * D)
    (PHI : (ZMod (2 ^ W))ˣ) {k₁ k₂ : ZMod (2 ^ W)} (hk : k₁ ≠ k₂) :
    ∃ j < D, addr b j PHI k₁ ≠ addr b j PHI k₂ := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  have hfold : k₁ * (PHI : ZMod (2 ^ W)) ≠ k₂ * (PHI : ZMod (2 ^ W)) :=
    fun h => hk (fold_injective PHI h)
  have hval : (k₁ * (PHI : ZMod (2 ^ W))).val ≠ (k₂ * (PHI : ZMod (2 ^ W))).val :=
    fun h => hfold (ZMod.val_injective _ h)
  have b1 : (k₁ * (PHI : ZMod (2 ^ W))).val < 2 ^ (b * D) := hW ▸ ZMod.val_lt _
  have b2 : (k₂ * (PHI : ZMod (2 ^ W))).val < 2 ^ (b * D) := hW ▸ ZMod.val_lt _
  exact digits_separate hb hval b1 b2

-- ═══════════════════════════════════════════════════════════════════
-- THE SHIPPED PROBE — the linear prefix, legitimized (the audit's unwritten lemma)
-- ═══════════════════════════════════════════════════════════════════
-- `chain_separates_within_radix` covers the PURE digit walk. The shipped `descend` interposes a
-- cache-locality prefix: after home it scans the next `G−1` slots of the home line before taking
-- fresh digits. That probe was, until here, covered by tests but not by a theorem. It separates
-- too — one step later per prefix slot: the prefix is a translation (injective mod 2^b), and the
-- digits after it are the proven walk, re-indexed.

/-- The shipped probe sequence: step `0` is the home digit; steps `1..G−1` are the linear prefix
    (home + j, mod the table); steps `≥ G` are the fresh radix digits, re-indexed. `G = 1` is the
    pure descent. -/
def seqLin (b G p j : ℕ) : ℕ :=
  if j < G then (digit b 0 p + j) % 2 ^ b
  else digit b (j - G + 1) p

/-- The shipped probe's codomain: every step addresses a slot below the radix. Both branches land
    in `[0, 2^b)` — the linear prefix is a residue, the fresh digits are `digit_lt`. -/
theorem seqLin_lt {b G p : ℕ} (hb : 0 < b) (j : ℕ) : seqLin b G p j < 2 ^ b := by
  unfold seqLin
  split
  · exact Nat.mod_lt _ (by positivity)
  · exact digit_lt b _ p hb

/-- **The shipped probe separates within `G + D − 1` (closure of the deployed `descend`).** Distinct
    keys' `seqLin` addresses differ at some step `j < G + D − 1`: if the home digit differs, step 0
    reads it (translation by `j` is injective mod `2^b`, so every prefix step differs too); if the
    home digit agrees, some later digit `j* ∈ [1, D)` differs (`digits_separate`) and step
    `j* + G − 1` reads it. The linear prefix costs at most `G − 1` extra steps over the pure
    descent's `D` — still a constant fixed by the word and the group width, never `n`. Flood
    immunity of the probe AS SHIPPED, not only as idealized. -/
theorem linear_prefix_chain_separates {b D G W : ℕ} (hb : 0 < b) (hG : 1 ≤ G) (hW : W = b * D)
    (PHI : (ZMod (2 ^ W))ˣ) {k₁ k₂ : ZMod (2 ^ W)} (hk : k₁ ≠ k₂) :
    ∃ j < G + D - 1,
      seqLin b G ((k₁ * (PHI : ZMod (2 ^ W))).val) j
        ≠ seqLin b G ((k₂ * (PHI : ZMod (2 ^ W))).val) j := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  have hfold : k₁ * (PHI : ZMod (2 ^ W)) ≠ k₂ * (PHI : ZMod (2 ^ W)) :=
    fun h => hk (fold_injective PHI h)
  have hval : (k₁ * (PHI : ZMod (2 ^ W))).val ≠ (k₂ * (PHI : ZMod (2 ^ W))).val :=
    fun h => hfold (ZMod.val_injective _ h)
  set p₁ := (k₁ * (PHI : ZMod (2 ^ W))).val with hp₁
  set p₂ := (k₂ * (PHI : ZMod (2 ^ W))).val with hp₂
  have b1 : p₁ < 2 ^ (b * D) := hW ▸ ZMod.val_lt _
  have b2 : p₂ < 2 ^ (b * D) := hW ▸ ZMod.val_lt _
  obtain ⟨js, hjs, hne⟩ := digits_separate hb hval b1 b2
  by_cases h0 : digit b 0 p₁ = digit b 0 p₂
  · -- home digits agree ⇒ the separating digit is js ≥ 1; the walk reads it at js + G − 1
    have hjs0 : js ≠ 0 := fun h => hne (h ▸ h0)
    have hD : 1 ≤ D := by omega
    refine ⟨js + G - 1, by omega, ?_⟩
    have hnotlt : ¬ (js + G - 1 < G) := by omega
    unfold seqLin
    rw [if_neg hnotlt, if_neg hnotlt]
    have hidx : js + G - 1 - G + 1 = js := by omega
    rw [hidx]
    exact hne
  · -- home digits differ ⇒ step 0 reads it (the +0 translation is the identity below 2^b)
    refine ⟨0, by omega, ?_⟩
    unfold seqLin
    simp only [if_pos (show 0 < G by omega), Nat.add_zero,
      Nat.mod_eq_of_lt (digit_lt b 0 p₁ hb), Nat.mod_eq_of_lt (digit_lt b 0 p₂ hb)]
    exact h0

-- ═══════════════════════════════════════════════════════════════════
-- THE LOAD FACE — why lowering the load shortens the probe (the lever)
-- ═══════════════════════════════════════════════════════════════════

/- The bound above is worst-case (flood-immunity). The PROFILE says the LIVE cost is the *average*
   probe length at the working load: at load 0.76 only 62% of lookups hit home (avg 1.99 steps); at
   load 0.48, 76% hit home (avg 1.39). So the optimization lever is the load factor. This section
   grounds it by construction: (1) probe length is monotone in occupancy — fewer keys in the way can
   only shorten a probe; (2) the home map is perfectly balanced — `PHI` a unit ⇒ every slot is the
   home of exactly the same number of coordinates, so there is no hot slot and the load is the ONLY
   knob. Lowering the grow threshold is therefore a sound, adversary-proof win, not a guess. -/

open Finset

/-- The probe visits slots `seq 0, seq 1, …` (`home`, then `descend 1, 2, …`). An unsuccessful
    search / insertion stops at the first slot not in the occupied set `O` — its probe length. -/
def plen {α : Type} [DecidableEq α] (seq : ℕ → α) (O : Finset α) (h : ∃ j, seq j ∉ O) : ℕ :=
  Nat.find h

/-- **The load lever (by construction).** Enlarging the occupied set can only lengthen a probe:
    `O ⊆ O' ⇒ plen O ≤ plen O'`. Lowering the load (a smaller occupied set) gives weakly shorter
    probes — the measured 62%→76% home-hit swing as load drops 0.76→0.48 is exactly this. No
    adversary, no randomness: more keys in the way ⇒ the first free slot comes later. -/
theorem plen_mono {α : Type} [DecidableEq α] (seq : ℕ → α) {O O' : Finset α} (hsub : O ⊆ O')
    (h : ∃ j, seq j ∉ O) (h' : ∃ j, seq j ∉ O') : plen seq O h ≤ plen seq O' h' := by
  apply Nat.find_mono
  intro j hj
  exact fun hO => hj (hsub hO)

/-- **No amplification — the distinct slots a probe crosses are slots the occupier owns.** The
    probe's addresses over `[0, plen)` all lie in `O` by `Nat.find_min`, so their image is a subset
    of `O`: whoever forced the probe to run that long already owns that many distinct occupied
    slots. Unconditional — no injectivity, no distribution, every occupied set including an
    adversarial one. The contrapositive is the security reading
    (`AggregationBound.probe_prefix_forces_distinct_occupied_slots`). -/
theorem plen_image_card_le {α : Type} [DecidableEq α] (seq : ℕ → α) (O : Finset α)
    (h : ∃ j, seq j ∉ O) :
    ((range (plen seq O h)).image seq).card ≤ O.card := by
  apply Finset.card_le_card
  intro s hs
  obtain ⟨j, hj, rfl⟩ := Finset.mem_image.mp hs
  have := Nat.find_min h (m := j) (Finset.mem_range.mp hj)
  simpa using this

/-- **The exchange rate is 1:1 when the probe does not revisit (headline).** If the probe's
    addresses are pairwise distinct over its own prefix, `j ↦ seq j` injects `range (plen)` into
    `O`, so `plen ≤ O.card`: a probe of length `L` costs the occupier `L` slots, one per step. The
    `Set.InjOn` hypothesis is a real hypothesis and it is NOT discharged for this probe —
    `addr_not_injOn` below is the counterexample. Repeats are the adversary's discount, and
    `plen_image_card_le` is the bound that survives them. -/
theorem plen_le_card {α : Type} [DecidableEq α] (seq : ℕ → α) (O : Finset α)
    (h : ∃ j, seq j ∉ O) (hinj : Set.InjOn seq (range (plen seq O h))) :
    plen seq O h ≤ O.card := by
  have hmaps : ∀ j ∈ range (plen seq O h), seq j ∈ O := by
    intro j hj
    have := Nat.find_min h (m := j) (Finset.mem_range.mp hj)
    simpa using this
  calc plen seq O h = (range (plen seq O h)).card := (Finset.card_range _).symm
    _ ≤ O.card := Finset.card_le_card_of_injOn seq hmaps hinj

/-- The key `0` folds to `0`, whose every digit is `0` — its probe sits on slot `0` at every step. -/
theorem addr_zero_constant {b W : ℕ} (PHI : (ZMod (2 ^ W))ˣ) (j : ℕ) :
    addr b j PHI 0 = 0 := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  simp [addr, digit]

/-- **The honesty guard on `plen_le_card`: this probe is not injective in the step.** `addr` at the
    key `0` is constant, so `Set.InjOn` fails already on two steps. A digit walk repeats an address
    whenever two windows of the fold agree, and nothing forbids that — so `plen_le_card`'s
    hypothesis is undischarged here BY CONSTRUCTION, not by an unproved lemma.
    `addr_not_injOn_of_exhausted` below is the general fact this is one instance of, and
    `plen_addr_le_radix` is the `plen` bound the repeats give instead. -/
theorem addr_not_injOn {b W : ℕ} (PHI : (ZMod (2 ^ W))ˣ) :
    ¬ Set.InjOn (fun j => addr b j PHI 0) (range 2) := by
  intro hinj
  have h01 : (0 : ℕ) = 1 := hinj (by simp) (by simp) (by simp [addr_zero_constant])
  exact absurd h01 (by omega)

-- ═══════════════════════════════════════════════════════════════════
-- THE DIGITS RUN OUT — `plen ≤ D` with no `Set.InjOn`, and what it does NOT say
-- ═══════════════════════════════════════════════════════════════════
-- `plen_le_card` wanted the walk to be injective in the step and it is not. The repeats are the
-- bound: past the word there are no digits left, every address is slot `0`, and a walk that has
-- not stopped by then never will. So the length is bounded outright — and the defect the crafted
-- key set exhibits is not length but TERMINATION.

/-- **A digit past the word is `0`.** `p < 2^W ≤ 2^(b·j)` empties the window `[b·j, b·j+b)`. -/
theorem digit_eq_zero_of_exhausted {b j p W : ℕ} (hp : p < 2 ^ W) (hW : W ≤ b * j) :
    digit b j p = 0 := by
  have : p < 2 ^ (b * j) := lt_of_lt_of_le hp (Nat.pow_le_pow_right (by norm_num) hW)
  simp [digit, Nat.div_eq_of_lt this]

/-- **The descent parks on slot `0` once the digits run out.** For `W ≤ b·j` every key's step-`j`
    address is `0`: the walk has at most `D = ⌈W/b⌉` distinct addresses and is constant after them.
    This is the MODEL's window schedule — the shipped `Table::descend` sweeps there instead, which
    is `GroupProbe.seqSweep` and terminates by `GroupProbe.seqSweep_terminates`. -/
theorem addr_eq_zero_of_exhausted {b j W : ℕ} (hW : W ≤ b * j) (PHI : (ZMod (2 ^ W))ˣ)
    (k : ZMod (2 ^ W)) : addr b j PHI k = 0 := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  exact digit_eq_zero_of_exhausted (ZMod.val_lt _) hW

/-- **`plen ≤ D` — the length bound `plen_le_card` could not have (headline).** No `Set.InjOn`, no
    distribution, no load hypothesis, every key and every occupied set: if some step of the descent
    lands outside `O` at all, the first one does so by step `D`. A walk still running at step `D+1`
    would have had step `D` blocked and stopped at a later step, and both addresses are slot `0`
    (`addr_eq_zero_of_exhausted`) — occupied and free at once.
    What it does NOT say is that the probe terminates. That is the real hypothesis, and it is what
    `AggregationBound.descent_can_be_wholly_blocked` takes: that `O` refutes `plen < D`, and once it
    also holds slot `0` there is no `∃ j, addr b j PHI k ∉ O` and no `plen` to bound. Length was
    never the defect. -/
theorem plen_addr_le_radix {b D W : ℕ} (hW : W ≤ b * D) (PHI : (ZMod (2 ^ W))ˣ)
    (k : ZMod (2 ^ W)) (O : Finset ℕ) (h : ∃ j, addr b j PHI k ∉ O) :
    plen (fun j => addr b j PHI k) O h ≤ D := by
  by_contra hgtn
  have hgt : D < plen (fun j => addr b j PHI k) O h := Nat.lt_of_not_le hgtn
  have hD : addr b D PHI k ∈ O := by
    have := Nat.find_min h (m := D) hgt
    simpa using this
  have hspec : addr b (Nat.find h) PHI k ∉ O := Nat.find_spec h
  have hge : W ≤ b * Nat.find h :=
    le_trans hW (Nat.mul_le_mul_left b (le_of_lt hgt))
  rw [addr_eq_zero_of_exhausted hge PHI k] at hspec
  rw [addr_eq_zero_of_exhausted hW PHI k] at hD
  exact hspec hD

/-- The shipped probe exhausts too: past `G + D − 1` the linear prefix is over and the fresh digit
    it asks for is beyond the word. -/
theorem seqLin_eq_zero_of_exhausted {b G D W p j : ℕ} (hp : p < 2 ^ W) (hW : W ≤ b * D)
    (hD : 1 ≤ D) (hj : G + D - 1 ≤ j) : seqLin b G p j = 0 := by
  have hjG : ¬ j < G := by omega
  unfold seqLin
  rw [if_neg hjG]
  exact digit_eq_zero_of_exhausted hp (le_trans hW (Nat.mul_le_mul_left b (by omega)))

/-- **`plen ≤ G + D − 1` for the probe AS SHIPPED.** `plen_addr_le_radix` at `seqLin`: the linear
    prefix costs its `G − 1` steps and then the digits run out the same way. -/
theorem plen_seqLin_le_radix {b G D W : ℕ} (hW : W ≤ b * D) (hD : 1 ≤ D)
    (PHI : (ZMod (2 ^ W))ˣ) (k : ZMod (2 ^ W)) (O : Finset ℕ)
    (h : ∃ j, seqLin b G ((k * (PHI : ZMod (2 ^ W))).val) j ∉ O) :
    plen (fun j => seqLin b G ((k * (PHI : ZMod (2 ^ W))).val) j) O h ≤ G + D - 1 := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  by_contra hgtn
  have hgt : G + D - 1 < Nat.find h := Nat.lt_of_not_le hgtn
  have hmem : seqLin b G ((k * (PHI : ZMod (2 ^ W))).val) (G + D - 1) ∈ O := by
    have := Nat.find_min h (m := G + D - 1) hgt
    simpa using this
  have hspec : seqLin b G ((k * (PHI : ZMod (2 ^ W))).val) (Nat.find h) ∉ O := Nat.find_spec h
  rw [seqLin_eq_zero_of_exhausted (ZMod.val_lt _) hW hD (le_of_lt hgt)] at hspec
  rw [seqLin_eq_zero_of_exhausted (ZMod.val_lt _) hW hD (le_refl _)] at hmem
  exact hspec hmem

/-- **Every key's walk repeats, not only `0`'s.** Steps `D` and `D+1` both read the exhausted `0`,
    so `Set.InjOn` fails on `range (D+2)` for EVERY key — `addr_not_injOn` is one instance of this.
    The one-slot-per-step route was closed for the whole probe, not for a corner case. -/
theorem addr_not_injOn_of_exhausted {b D W : ℕ} (hW : W ≤ b * D) (PHI : (ZMod (2 ^ W))ˣ)
    (k : ZMod (2 ^ W)) : ¬ Set.InjOn (fun j => addr b j PHI k) (range (D + 2)) := by
  intro hinj
  have hD1 : W ≤ b * (D + 1) := le_trans hW (Nat.mul_le_mul_left b (by omega))
  have : D = D + 1 := hinj (by simp) (by simp) (by
    simp [addr_eq_zero_of_exhausted hW PHI k, addr_eq_zero_of_exhausted hD1 PHI k])
  omega

/-- The home slot of a coordinate: the top `b` bits of `c·PHI` — `home` in the sparse branch
    (`(c·PHI) >> (W − b)`). -/
def homeSlot {W : ℕ} (b : ℕ) (PHI : (ZMod (2 ^ W))ˣ) (c : ZMod (2 ^ W)) : ℕ :=
  (c * (PHI : ZMod (2 ^ W))).val / 2 ^ (W - b)

/-- **The shipped `home` IS one of the walk's digits — the top one.** `homeSlot` reads bits
    `[W−b, W)`; `digit b j` reads `[b·j, b·j+b)`; with `W = b·D` they coincide at `j = D−1`. So the
    worst-case half of this file (`addr`) and the counting half (`homeSlot`, `load`) address the
    same slot space, and the average-case bounds below are about the walk, not a second model. -/
theorem homeSlot_eq_addr_top {b D W : ℕ} (hD : 0 < D) (hW : W = b * D)
    (PHI : (ZMod (2 ^ W))ˣ) (c : ZMod (2 ^ W)) :
    homeSlot b PHI c = addr b (D - 1) PHI c := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  have hbW : b ≤ W := by
    rw [hW]
    calc b = b * 1 := by ring
      _ ≤ b * D := Nat.mul_le_mul_left b hD
  have hbd : b * (D - 1) = W - b := by
    rw [hW]
    cases D with
    | zero => omega
    | succ m => simp [Nat.mul_succ]
  have hpow : (2 : ℕ) ^ b * 2 ^ (W - b) = 2 ^ W := by
    rw [← pow_add]; congr 1; omega
  have hlt : (c * (PHI : ZMod (2 ^ W))).val / 2 ^ (W - b) < 2 ^ b := by
    rw [Nat.div_lt_iff_lt_mul (by positivity), mul_comm, hpow]
    exact ZMod.val_lt _
  unfold homeSlot addr digit
  rw [hbd, Nat.mod_eq_of_lt hlt]

/-- The truncation `v ↦ v / 2^(W−b)` is exactly `2^(W−b)`-to-one on `[0, 2^W)`: each slot `s < 2^b`
    is the image of the contiguous block `[s·2^(W−b), (s+1)·2^(W−b))`. -/
theorem trunc_fiber_card {W b s : ℕ} (hb : b ≤ W) (hs : s < 2 ^ b) :
    ((range (2 ^ W)).filter (fun v => v / 2 ^ (W - b) = s)).card = 2 ^ (W - b) := by
  have hD : 0 < 2 ^ (W - b) := pow_pos (by norm_num) _
  have hpow : 2 ^ b * 2 ^ (W - b) = 2 ^ W := by rw [← pow_add]; congr 1; omega
  have hset : (range (2 ^ W)).filter (fun v => v / 2 ^ (W - b) = s)
      = Ico (s * 2 ^ (W - b)) ((s + 1) * 2 ^ (W - b)) := by
    ext v
    simp only [mem_filter, mem_range, mem_Ico]
    constructor
    · rintro ⟨_, hq⟩
      refine ⟨(Nat.le_div_iff_mul_le hD).mp hq.ge, (Nat.div_lt_iff_lt_mul hD).mp ?_⟩
      omega
    · rintro ⟨hlo, hhi⟩
      have hv : v / 2 ^ (W - b) = s := Nat.div_eq_of_lt_le hlo hhi
      refine ⟨?_, hv⟩
      calc v < (s + 1) * 2 ^ (W - b) := hhi
        _ ≤ 2 ^ b * 2 ^ (W - b) := Nat.mul_le_mul_right _ hs
        _ = 2 ^ W := hpow
  rw [hset, Nat.card_Ico, Nat.succ_mul]
  omega

/-- **No hot slot (by construction).** Because `PHI` is a unit, `home` is *perfectly balanced*: every
    slot `s < 2^b` is the home of exactly `2^(W−b)` coordinates. The fold spreads the coordinate
    space uniformly across the slots — collisions come only from how MANY keys are present (the load),
    never from a structurally favored slot an adversary could pile onto. This is the average-case
    companion to `chain_separates_within_radix`'s worst-case bound, the same `PHI`-unit root. -/
theorem home_balanced {W b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) {s : ℕ} (hs : s < 2 ^ b) :
    (univ.filter (fun c => homeSlot b PHI c = s)).card = 2 ^ (W - b) := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  rw [← trunc_fiber_card hb hs]
  apply Finset.card_bij (fun c _ => (c * (PHI : ZMod (2 ^ W))).val)
  · intro c hc
    rw [mem_filter] at hc ⊢
    exact ⟨mem_range.mpr (ZMod.val_lt _), hc.2⟩
  · intro c₁ _ c₂ _ heq
    exact fold_injective PHI (ZMod.val_injective _ heq)
  · intro v hv
    rw [mem_filter, mem_range] at hv
    refine ⟨(v : ZMod (2 ^ W)) * (↑PHI⁻¹ : ZMod (2 ^ W)), ?_, ?_⟩
    · rw [mem_filter]
      refine ⟨mem_univ _, ?_⟩
      show ((v : ZMod (2 ^ W)) * (↑PHI⁻¹) * (PHI : ZMod (2 ^ W))).val / 2 ^ (W - b) = s
      rw [mul_assoc, ← Units.val_mul, inv_mul_cancel, Units.val_one, mul_one,
          ZMod.val_natCast_of_lt hv.1]
      exact hv.2
    · show ((v : ZMod (2 ^ W)) * (↑PHI⁻¹) * (PHI : ZMod (2 ^ W))).val = v
      rw [mul_assoc, ← Units.val_mul, inv_mul_cancel, Units.val_one, mul_one,
          ZMod.val_natCast_of_lt hv.1]

-- ═══════════════════════════════════════════════════════════════════
-- THE PILE AT SLOT GRANULARITY — how many keys can share a whole window set
-- ═══════════════════════════════════════════════════════════════════
-- `chain_separates_within_radix` assumes `W = b·D` EXACTLY, which forces the pile to `2^0 = 1`.
-- The shipped table has `14 ∤ 64`, so the live case is `b·F ≤ W` with a residual `r = W − b·F > 0`
-- — `GroupProbe.group_descent_dichotomy`'s hypothesis, whose pile `GroupProbe.pile_card` counts at
-- fold granularity over `range (2^W)`. Here it is at KEY granularity over `ZMod (2^W)`, and it
-- needs no new counting: `home_balanced` at width `b·F` IS that count, the fold bijection already
-- taken. (`GroupProbe.pile_card` and `trunc_fiber_card` are the same statement at `r = W − b`;
-- welded 2026-08-06 — the second 25-line proof is gone and `pile_card` now rides this one.)

/-- Agreement on every window of the consumed prefix IS equality of the prefix: two prefixes below
    `2^(b·F)` that share all `F` digits are equal (`digits_separate`, contrapositive). -/
theorem windows_determine_prefix {b F q₁ q₂ : ℕ} (hb : 0 < b)
    (h1 : q₁ < 2 ^ (b * F)) (h2 : q₂ < 2 ^ (b * F))
    (hagree : ∀ i < F, digit b i q₁ = digit b i q₂) : q₁ = q₂ := by
  by_contra hne
  obtain ⟨i, hi, hdi⟩ := digits_separate hb hne h1 h2
  exact hdi (hagree i hi)

/-- The consumed prefix of a fold is a `b·F`-bit word — the codomain the pile is counted in. -/
theorem consumed_prefix_lt {b F W : ℕ} (hbF : b * F ≤ W) (PHI : (ZMod (2 ^ W))ˣ)
    (c : ZMod (2 ^ W)) :
    (c * (PHI : ZMod (2 ^ W))).val / 2 ^ (W - b * F) < 2 ^ (b * F) := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  rw [Nat.div_lt_iff_lt_mul (by positivity)]
  have hpow : (2 : ℕ) ^ (b * F) * 2 ^ (W - b * F) = 2 ^ W := by
    rw [← pow_add]; congr 1; omega
  rw [hpow]
  exact ZMod.val_lt _

/-- **The pile is `2^(W − b·F)` keys, exactly (headline).** The keys agreeing with `k` on EVERY one
    of the `F` windows the descent consumes number exactly `2^(W − b·F)` — the residual the walk
    never reads, bounded by the word and not by `n`. `digit b i (p / 2^(W−b·F))` for `i < F` is
    `GroupProbe.topDigit b F W p (F−1−i)`, the same `F` windows re-indexed, so this is
    `GroupProbe.pile_card` brought to key granularity; the proof is `home_balanced` at width `b·F`
    (`homeSlot (b·F)` IS the consumed prefix) plus `windows_determine_prefix`. This is the ceiling
    on how many keys can ride a victim's whole window set — at the shipped numbers, 256. -/
theorem window_pile_card {b F W : ℕ} (hb : 0 < b) (hbF : b * F ≤ W)
    (PHI : (ZMod (2 ^ W))ˣ) (k : ZMod (2 ^ W)) :
    (univ.filter (fun c : ZMod (2 ^ W) => ∀ i < F,
        digit b i ((c * (PHI : ZMod (2 ^ W))).val / 2 ^ (W - b * F))
          = digit b i ((k * (PHI : ZMod (2 ^ W))).val / 2 ^ (W - b * F)))).card
      = 2 ^ (W - b * F) := by
  classical
  have hset : (univ.filter (fun c : ZMod (2 ^ W) => ∀ i < F,
      digit b i ((c * (PHI : ZMod (2 ^ W))).val / 2 ^ (W - b * F))
        = digit b i ((k * (PHI : ZMod (2 ^ W))).val / 2 ^ (W - b * F))))
      = univ.filter (fun c : ZMod (2 ^ W) =>
          homeSlot (b * F) PHI c = homeSlot (b * F) PHI k) := by
    ext c
    simp only [mem_filter, mem_univ, true_and]
    constructor
    · intro hagree
      exact windows_determine_prefix hb (consumed_prefix_lt hbF PHI c)
        (consumed_prefix_lt hbF PHI k) hagree
    · intro heq i _
      show digit b i (homeSlot (b * F) PHI c) = digit b i (homeSlot (b * F) PHI k)
      rw [heq]
  rw [hset]
  exact home_balanced hbF PHI (consumed_prefix_lt hbF PHI k)

/-- **The shipped pile is 256.** `W = 64, b = 14`: the descent windows consume product bits
    `[8, 64)` — used `14, 28, 42, 56`; the next, `70`, exceeds the word — so `F = 4`, `b·F = 56`,
    and exactly `2^(64−56) = 256` keys share a victim's entire window set. Stated on the consumed
    prefix `homeSlot 56` — the same set `window_pile_card` filters — and proved by `home_balanced` at
    width `56`, because the bounded-∀ form does not elaborate at these numerals (`DecidablePred` for
    `∀ i < 4, …` over `ZMod (2^64)` fails to synthesize). Why a 12,000-key flood aimed at one window
    cannot put 12,000 keys on one descent: 256 is the ceiling, and the word sets it. -/
theorem shipped_pile_card (PHI : (ZMod (2 ^ 64))ˣ) (k : ZMod (2 ^ 64)) :
    (univ.filter (fun c : ZMod (2 ^ 64) =>
        homeSlot 56 PHI c = homeSlot 56 PHI k)).card = 256 := by
  have hs : homeSlot 56 PHI k < 2 ^ 56 := by
    have := consumed_prefix_lt (b := 56) (F := 1) (W := 64) (by norm_num) PHI k
    norm_num at this ⊢
    exact this
  have h := home_balanced (W := 64) (b := 56) (by norm_num) PHI hs
  norm_num at h
  exact h

end RadixDescent

-- ═══════════════════════════════════════════════════════════════════
-- namespace ProbabilisticBounds — the average-case companion (uniform-hashing assumption)
-- ═══════════════════════════════════════════════════════════════════

namespace ProbabilisticBounds

open Finset RadixDescent

variable {W : ℕ}

-- ═══════════════════════════════════════════════════════════════════
-- §0. THE SAMPLE SPACE — `n` keys drawn i.i.d. uniformly from the real key universe
-- ═══════════════════════════════════════════════════════════════════

/-- `n` keys drawn i.i.d. and uniformly from the real key universe `ZMod (2^W)` — the sample
    space every result below is a statement about. -/
abbrev Keys (n W : ℕ) : Type := Fin n → ZMod (2 ^ W)

variable {n : ℕ}

theorem card_keys : Fintype.card (Keys n W) = (2 ^ W) ^ n := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  simp [Keys, ZMod.card]

/-- **`home_balanced`, lifted from one key to `n` keys at once.** Fixing an arbitrary target slot
    `t i` for *every* key `i` simultaneously still leaves exactly `(2^(W−b))^n` assignments — the
    same constant, independent of which target pattern `t` was asked for. This is the pushforward
    fact that makes `homeSlot ∘ κ` behave as `n` i.i.d.-uniform slot draws: every pattern is
    equally likely, because `home_balanced` says every single slot is equally likely and the `n`
    keys are drawn independently. Proved by lifting `home_balanced`'s one-key fiber count through
    `Equiv.subtypePiEquivPi` (a tuple satisfying `n` independent pointwise constraints is a tuple
    of independent per-slot choices) and multiplying with `Fintype.card_pi`. -/
theorem fiber_lift {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) (t : Fin n → ℕ)
    (ht : ∀ i, t i < 2 ^ b) :
    Fintype.card {κ : Keys n W // ∀ i, homeSlot b PHI (κ i) = t i} = (2 ^ (W - b)) ^ n := by
  have step : {κ : Keys n W // ∀ i, homeSlot b PHI (κ i) = t i}
      ≃ ∀ i : Fin n, {c : ZMod (2 ^ W) // homeSlot b PHI c = t i} :=
    Equiv.subtypePiEquivPi (β := fun _ : Fin n => ZMod (2 ^ W))
      (p := fun i c => homeSlot b PHI c = t i)
  rw [Fintype.card_congr step, Fintype.card_pi]
  have : ∀ i, Fintype.card {c : ZMod (2 ^ W) // homeSlot b PHI c = t i} = 2 ^ (W - b) := by
    intro i
    rw [Fintype.card_subtype]
    exact home_balanced hb PHI (ht i)
  simp [this]

/-- **`fiber_lift`, constrained on only a subset `S` of the `n` keys** — the other keys are
    free. The same pushforward fact, but only asking `homeSlot` to hit a target on `S`: the count
    factors as `(2^(W−b))^{|S|}` (the constrained keys, `home_balanced`) times `(2^W)^{n−|S|}`
    (the free keys, unconstrained). Specializing `S = univ` recovers `fiber_lift`; `S = {i}` gives
    the single-key marginal `expected_load` needs. -/
theorem fiber_lift_subset {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ)
    (S : Finset (Fin n)) (t : Fin n → ℕ) (ht : ∀ i ∈ S, t i < 2 ^ b) :
    Fintype.card {κ : Keys n W // ∀ i ∈ S, homeSlot b PHI (κ i) = t i}
      = (2 ^ (W - b)) ^ S.card * (2 ^ W) ^ (n - S.card) := by
  classical
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  have step : {κ : Keys n W // ∀ i, i ∈ S → homeSlot b PHI (κ i) = t i}
      ≃ ∀ i : Fin n, {c : ZMod (2 ^ W) // i ∈ S → homeSlot b PHI c = t i} :=
    Equiv.subtypePiEquivPi (β := fun _ : Fin n => ZMod (2 ^ W))
      (p := fun i c => i ∈ S → homeSlot b PHI c = t i)
  rw [Fintype.card_congr step, Fintype.card_pi]
  have hcard : ∀ i, Fintype.card {c : ZMod (2 ^ W) // i ∈ S → homeSlot b PHI c = t i}
      = if i ∈ S then 2 ^ (W - b) else 2 ^ W := by
    intro i
    by_cases hi : i ∈ S
    · rw [if_pos hi]
      rw [Fintype.card_congr (Equiv.subtypeEquivRight
        (p := fun c => i ∈ S → homeSlot b PHI c = t i) (q := fun c => homeSlot b PHI c = t i)
        (fun c => ⟨fun h => h hi, fun h _ => h⟩))]
      rw [Fintype.card_subtype]
      exact home_balanced hb PHI (ht i hi)
    · rw [if_neg hi]
      rw [Fintype.card_congr (Equiv.subtypeUnivEquiv (fun c => fun hc => absurd hc hi))]
      exact ZMod.card (2 ^ W)
  rw [Finset.prod_congr rfl (fun i _ => hcard i)]
  rw [Finset.prod_ite]
  have hf : (Finset.univ.filter (fun i : Fin n => i ∈ S)) = S := by ext i; simp
  have hg : (Finset.univ.filter (fun i : Fin n => ¬ i ∈ S)) = Sᶜ := by ext i; simp
  rw [hf, hg, Finset.prod_const, Finset.prod_const, Finset.card_compl, Fintype.card_fin]

/-- `homeSlot` always lands below `2^b` — the codomain bound the sums below range over. -/
theorem homeSlot_lt {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) (c : ZMod (2 ^ W)) :
    homeSlot b PHI c < 2 ^ b := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  unfold homeSlot
  rw [Nat.div_lt_iff_lt_mul (by positivity)]
  have hpow : (2 : ℕ) ^ b * 2 ^ (W - b) = 2 ^ W := by rw [← pow_add]; congr 1; omega
  rw [hpow]
  exact ZMod.val_lt _

-- ═══════════════════════════════════════════════════════════════════
-- §1. PER-PAIR COLLISION PROBABILITY (item 1) / TAG FALSE-POSITIVE RATE (item 4)
-- ═══════════════════════════════════════════════════════════════════

/-- **Per-pair collision count, on the real fold.** Among `n` i.i.d.-uniform keys, exactly two
    fixed distinct keys `i ≠ j` land on the same `homeSlot` for exactly this many of the `(2^W)^n`
    total placements: partition by *which* slot they collide on (`Finset.card_eq_sum_card_fiberwise`)
    and read each slot's count off `fiber_lift_subset` at `S = {i,j}`. -/
theorem pair_collision_count {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) {i j : Fin n}
    (hij : i ≠ j) :
    (Finset.univ.filter
      (fun κ : Keys n W => homeSlot b PHI (κ i) = homeSlot b PHI (κ j))).card
      = 2 ^ b * (2 ^ (W - b)) ^ 2 * (2 ^ W) ^ (n - 2) := by
  classical
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  have hmaps : (↑(Finset.univ.filter
      (fun κ : Keys n W => homeSlot b PHI (κ i) = homeSlot b PHI (κ j))) : Set (Keys n W)).MapsTo
      (fun κ => homeSlot b PHI (κ i)) (Finset.range (2 ^ b)) := by
    intro κ _
    exact Finset.mem_range.mpr (homeSlot_lt hb PHI _)
  rw [Finset.card_eq_sum_card_fiberwise hmaps]
  have hS : ({i, j} : Finset (Fin n)).card = 2 := by
    rw [Finset.card_insert_of_notMem (by simp [hij]), Finset.card_singleton]
  have heach : ∀ s ∈ Finset.range (2 ^ b),
      ((Finset.univ.filter
        (fun κ : Keys n W => homeSlot b PHI (κ i) = homeSlot b PHI (κ j))).filter
        (fun κ => homeSlot b PHI (κ i) = s)).card
        = (2 ^ (W - b)) ^ 2 * (2 ^ W) ^ (n - 2) := by
    intro s hs
    have hlt : s < 2 ^ b := Finset.mem_range.mp hs
    have heq : (Finset.univ.filter
        (fun κ : Keys n W => homeSlot b PHI (κ i) = homeSlot b PHI (κ j))).filter
        (fun κ => homeSlot b PHI (κ i) = s)
        = Finset.univ.filter
          (fun κ : Keys n W => ∀ k ∈ ({i, j} : Finset (Fin n)), homeSlot b PHI (κ k) = s) := by
      ext κ
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert,
        Finset.mem_singleton]
      constructor
      · rintro ⟨heqij, heqis⟩ k hk
        rcases hk with hk | hk
        · subst hk; exact heqis
        · subst hk; rw [← heqij]; exact heqis
      · intro h
        exact ⟨(h i (by simp)).trans (h j (by simp)).symm, h i (by simp)⟩
    rw [heq, ← Fintype.card_subtype]
    rw [fiber_lift_subset hb PHI ({i, j} : Finset (Fin n)) (fun _ => s) (fun k _ => hlt), hS]
  rw [Finset.sum_congr rfl heach, Finset.sum_const, Finset.card_range]
  ring

/-- **Per-pair collision probability (headline, item 1).** Two distinct keys among `n`
    independently-uniform keys land on the same slot with probability exactly `1/2^b` — the
    classical `2^-W` at full width `b = W`, or the tag width `b` for a lossy `b`-bit fingerprint
    (`tag_false_positive_rate` below is this theorem, read at `b` = tag width). Proved on the real
    `homeSlot`/`PHI`, not an abstract space: `pair_collision_count` divided by `card_keys`. -/
theorem per_pair_collision_prob {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) {i j : Fin n}
    (hij : i ≠ j) (hn : 2 ≤ n) :
    ((Finset.univ.filter
        (fun κ : Keys n W => homeSlot b PHI (κ i) = homeSlot b PHI (κ j))).card : ℚ)
      / (Fintype.card (Keys n W) : ℚ) = 1 / 2 ^ b := by
  rw [pair_collision_count hb PHI hij, card_keys]
  have hnat : 2 ^ b * (2 ^ (W - b)) ^ 2 * (2 ^ W) ^ (n - 2) * 2 ^ b = (2 ^ W) ^ n := by
    have hpb : (2 : ℕ) ^ b * 2 ^ (W - b) = 2 ^ W := by rw [← pow_add]; congr 1; omega
    have hn2 : (2 ^ W : ℕ) ^ n = (2 ^ W) ^ 2 * (2 ^ W) ^ (n - 2) := by
      rw [← pow_add]; congr 1; omega
    rw [hn2, ← hpb]; ring
  have hcast := congrArg (Nat.cast (R := ℚ)) hnat
  push_cast at hcast
  rw [div_eq_div_iff (by positivity) (by positivity)]
  push_cast
  linear_combination hcast

/-- **Tag false-positive rate (item 4), the same theorem at tag width.** A `b`-bit lossy tag
    (SwissTable-style) reports a false positive on two distinct keys exactly when they collide on
    `homeSlot` at width `b` — `per_pair_collision_prob`, unchanged, read with `b` = tag width
    instead of the full table width. No new theorem needed: the tag *is* a truncated `homeSlot`.
    (Historical: the framework no longer ships a tag line — `hashtrinity` THE CUT, 2026-07-02 —
    this reading stays as the fact it was always about the `homeSlot` truncation, not the tag.) -/
theorem tag_false_positive_rate {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) {i j : Fin n}
    (hij : i ≠ j) (hn : 2 ≤ n) :
    ((Finset.univ.filter
        (fun κ : Keys n W => homeSlot b PHI (κ i) = homeSlot b PHI (κ j))).card : ℚ)
      / (Fintype.card (Keys n W) : ℚ) = 1 / 2 ^ b :=
  per_pair_collision_prob hb PHI hij hn

-- ═══════════════════════════════════════════════════════════════════
-- §2. BIRTHDAY BOUND (item 2) — aggregating the per-pair count via a union bound
-- ═══════════════════════════════════════════════════════════════════

/-- **Birthday-bound style pigeonhole counting (headline, item 2).** Among `n` i.i.d.-uniform
    keys, the count of placements with *some* two keys sharing a `homeSlot` is at most
    `n·(n−1)` times a single pair's collision count — a union bound over the `n·(n−1)` ordered
    distinct-index pairs (`Finset.offDiag`), each contributing exactly `pair_collision_count`.
    Dividing by `card_keys` gives the probability bound `P[collision] ≤ n(n−1)/2^b` — off by a
    constant factor `2` from the tightest textbook form `C(n,2)/2^b` (this counts each unordered
    pair twice, via ordered pairs, to avoid `Sym2` bookkeeping); still the genuine union-bound
    birthday estimate, `O(n²/2^b)`. -/
theorem birthday_bound {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) :
    (Finset.univ.filter (fun κ : Keys n W =>
        ∃ i j : Fin n, i ≠ j ∧ homeSlot b PHI (κ i) = homeSlot b PHI (κ j))).card
      ≤ n * (n - 1) * (2 ^ b * (2 ^ (W - b)) ^ 2 * (2 ^ W) ^ (n - 2)) := by
  classical
  have hsub : Finset.univ.filter (fun κ : Keys n W =>
      ∃ i j : Fin n, i ≠ j ∧ homeSlot b PHI (κ i) = homeSlot b PHI (κ j))
      ⊆ (Finset.univ : Finset (Fin n)).offDiag.biUnion
        (fun p => Finset.univ.filter
          (fun κ : Keys n W => homeSlot b PHI (κ p.1) = homeSlot b PHI (κ p.2))) := by
    intro κ hκ
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hκ
    obtain ⟨i, j, hij, heq⟩ := hκ
    refine Finset.mem_biUnion.mpr ⟨(i, j), Finset.mem_offDiag.mpr ⟨Finset.mem_univ _,
      Finset.mem_univ _, hij⟩, ?_⟩
    simp [heq]
  calc (Finset.univ.filter (fun κ : Keys n W =>
        ∃ i j : Fin n, i ≠ j ∧ homeSlot b PHI (κ i) = homeSlot b PHI (κ j))).card
      ≤ ((Finset.univ : Finset (Fin n)).offDiag.biUnion
          (fun p => Finset.univ.filter
            (fun κ : Keys n W => homeSlot b PHI (κ p.1) = homeSlot b PHI (κ p.2)))).card :=
        Finset.card_le_card hsub
    _ ≤ ∑ p ∈ (Finset.univ : Finset (Fin n)).offDiag,
        (Finset.univ.filter
          (fun κ : Keys n W => homeSlot b PHI (κ p.1) = homeSlot b PHI (κ p.2))).card :=
        Finset.card_biUnion_le
    _ = ∑ _p ∈ (Finset.univ : Finset (Fin n)).offDiag,
        (2 ^ b * (2 ^ (W - b)) ^ 2 * (2 ^ W) ^ (n - 2)) := by
        refine Finset.sum_congr rfl (fun p hp => ?_)
        exact pair_collision_count hb PHI (Finset.mem_offDiag.mp hp).2.2
    _ = n * (n - 1) * (2 ^ b * (2 ^ (W - b)) ^ 2 * (2 ^ W) ^ (n - 2)) := by
        rw [Finset.sum_const, Finset.offDiag_card, Finset.card_univ, Fintype.card_fin,
          smul_eq_mul, Nat.mul_sub_left_distrib, mul_one]

-- ═══════════════════════════════════════════════════════════════════
-- §3. EXPECTED LOAD AS A FUNCTION OF α = n/2^b (item 3)
-- ═══════════════════════════════════════════════════════════════════

/-- The occupancy of slot `s` under placement `κ` — how many of the `n` keys land there. -/
def load (b : ℕ) (PHI : (ZMod (2 ^ W))ˣ) (κ : Keys n W) (s : ℕ) : ℕ :=
  (Finset.univ.filter (fun i : Fin n => homeSlot b PHI (κ i) = s)).card

/-- **The total occupancy of slot `s`, summed over every placement.** Swap the two sums (keys
    vs. placements) and read each key's single-coordinate marginal off `fiber_lift_subset` at the
    singleton `S = {i}`: every key contributes the same `2^(W−b)·(2^W)^{n−1}`, and there are `n`
    of them. -/
theorem sum_load {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) {s : ℕ} (hs : s < 2 ^ b) :
    ∑ κ : Keys n W, load b PHI κ s = n * (2 ^ (W - b) * (2 ^ W) ^ (n - 1)) := by
  classical
  have hswap : ∑ κ : Keys n W, load b PHI κ s
      = ∑ i : Fin n, (Finset.univ.filter (fun κ : Keys n W => homeSlot b PHI (κ i) = s)).card := by
    unfold load
    simp_rw [Finset.card_filter, Finset.sum_comm (s := (Finset.univ : Finset (Keys n W)))
      (t := (Finset.univ : Finset (Fin n)))]
  rw [hswap]
  have heach : ∀ i : Fin n,
      (Finset.univ.filter (fun κ : Keys n W => homeSlot b PHI (κ i) = s)).card
        = 2 ^ (W - b) * (2 ^ W) ^ (n - 1) := by
    intro i
    rw [← Fintype.card_subtype]
    have hSi : ({i} : Finset (Fin n)).card = 1 := Finset.card_singleton i
    have hfl := fiber_lift_subset hb PHI ({i} : Finset (Fin n)) (fun _ => s)
      (fun k hk => by rw [Finset.mem_singleton] at hk; subst hk; exact hs)
    rw [hSi, pow_one] at hfl
    rw [← hfl]
    exact Fintype.card_congr (Equiv.subtypeEquivRight (fun κ => by simp))
  rw [Finset.sum_congr rfl (fun i _ => heach i), Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, smul_eq_mul]

/-- **Expected load = `n / 2^b` (headline, item 3) — the load factor `α`.** The average occupancy
    of any slot, over the `(2^W)^n` i.i.d.-uniform placements, is exactly the classical `α = n/m`
    (`m = 2^b` slots): `sum_load` divided by the total placement count. Matches the textbook
    "expected bucket size" formula for uniform hashing, now proved on the real fold. -/
theorem expected_load {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) {s : ℕ} (hs : s < 2 ^ b)
    (hn : 1 ≤ n) :
    ((∑ κ : Keys n W, load b PHI κ s : ℕ) : ℚ) / (Fintype.card (Keys n W) : ℚ) = n / 2 ^ b := by
  rw [sum_load hb PHI hs, card_keys]
  have hnat : n * (2 ^ (W - b) * (2 ^ W) ^ (n - 1)) * 2 ^ b = n * (2 ^ W) ^ n := by
    have hpb : (2 : ℕ) ^ b * 2 ^ (W - b) = 2 ^ W := by rw [← pow_add]; congr 1; omega
    have hn1 : (2 ^ W : ℕ) ^ n = 2 ^ W * (2 ^ W) ^ (n - 1) := by
      conv_lhs => rw [show n = 1 + (n - 1) by omega]
      rw [pow_add, pow_one]
    rw [hn1, ← hpb]; ring
  have hqcast := congrArg (Nat.cast (R := ℚ)) hnat
  push_cast at hqcast
  rw [div_eq_div_iff (by positivity) (by positivity)]
  push_cast
  linear_combination hqcast

-- ═══════════════════════════════════════════════════════════════════
-- §4. MARKOV DISPLACEMENT TAIL BOUND (item 5)
-- ═══════════════════════════════════════════════════════════════════

/-- **Markov displacement tail bound (headline, item 5) — `O(f(α))` at load `α = n/2^b`.** The
    count of placements where slot `s` is occupied by at least `t` keys, times `t`, is at most the
    total occupancy summed over every placement (each such placement contributes ≥ `t`, and no
    placement contributes negatively) — Markov's inequality read off `sum_load` directly. Dividing
    by `card_keys` and by `t > 0` gives `P[load ≥ t] ≤ (n/2^b)/t`: a genuine tail bound at load `α`,
    tightening as `t` grows past the mean `α`. (A sharper, trie-depth-style `Θ(log n / b)` bound is
    a separate, harder result not attempted here.) -/
theorem displacement_tail_bound {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) {s : ℕ} (hs : s < 2 ^ b)
    (t : ℕ) :
    t * (Finset.univ.filter (fun κ : Keys n W => t ≤ load b PHI κ s)).card
      ≤ n * (2 ^ (W - b) * (2 ^ W) ^ (n - 1)) := by
  classical
  rw [← sum_load hb PHI hs]
  calc t * (Finset.univ.filter (fun κ : Keys n W => t ≤ load b PHI κ s)).card
      = ∑ _κ ∈ Finset.univ.filter (fun κ : Keys n W => t ≤ load b PHI κ s), t := by
        rw [Finset.sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ κ ∈ Finset.univ.filter (fun κ : Keys n W => t ≤ load b PHI κ s), load b PHI κ s :=
        Finset.sum_le_sum (fun κ hκ => (Finset.mem_filter.mp hκ).2)
    _ ≤ ∑ κ : Keys n W, load b PHI κ s :=
        Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
          (fun κ _ _ => Nat.zero_le _)

/-- **A whole probe-address set occupied — the count.** If every slot of `S` holds at least one of
    the `n` keys then a fixed member `s₀ ∈ S` does, so these placements are at most
    `displacement_tail_bound` at `t = 1`. `S` is any slot set; in the reading it is the addresses a
    key's descent will try, which makes this the average-case face of
    `AggregationBound.descent_can_be_wholly_blocked` — the same event, CERTAIN for a crafted key
    set. The bound is the single-slot one: covering all of `S` is not counted as harder here. -/
theorem probe_set_fully_occupied_count {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ)
    (S : Finset ℕ) (hS : ∀ s ∈ S, s < 2 ^ b) (hne : S.Nonempty) :
    (Finset.univ.filter (fun κ : Keys n W => ∀ s ∈ S, 0 < load b PHI κ s)).card
      ≤ n * (2 ^ (W - b) * (2 ^ W) ^ (n - 1)) := by
  classical
  obtain ⟨s₀, hs₀⟩ := hne
  have hsub : Finset.univ.filter (fun κ : Keys n W => ∀ s ∈ S, 0 < load b PHI κ s)
      ⊆ Finset.univ.filter (fun κ : Keys n W => 1 ≤ load b PHI κ s₀) := by
    intro κ hκ
    rw [Finset.mem_filter] at hκ ⊢
    exact ⟨hκ.1, hκ.2 s₀ hs₀⟩
  have h := displacement_tail_bound (n := n) hb PHI (hS s₀ hs₀) 1
  rw [one_mul] at h
  exact le_trans (Finset.card_le_card hsub) h

/-- **A whole probe-address set is occupied for at most an `α = n/2^b` fraction of key sets.**
    `probe_set_fully_occupied_count` over `card_keys`, read through `expected_load`. This is the
    honest strength of "the fold path resists a flood": a MASS over i.i.d.-uniform key sets, never
    a bound over every key set — worst-case that event is not rare but certain
    (`AggregationBound.descent_can_be_wholly_blocked`), and at the operating `α ≈ 0.4` this
    permits two key sets in five. Sharpened to `α^S.card` by `probe_set_fully_occupied_prob_pow`
    below: this form reads the event through ONE member of `S`, so covering the rest is free. -/
theorem probe_set_fully_occupied_prob {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ)
    (S : Finset ℕ) (hS : ∀ s ∈ S, s < 2 ^ b) (hne : S.Nonempty) (hn : 1 ≤ n) :
    ((Finset.univ.filter (fun κ : Keys n W => ∀ s ∈ S, 0 < load b PHI κ s)).card : ℚ)
      / (Fintype.card (Keys n W) : ℚ) ≤ n / 2 ^ b := by
  obtain ⟨s₀, hs₀⟩ := hne
  have hcount : ((Finset.univ.filter (fun κ : Keys n W => ∀ s ∈ S, 0 < load b PHI κ s)).card : ℚ)
      ≤ ((∑ κ : Keys n W, load b PHI κ s₀ : ℕ) : ℚ) := by
    have h := probe_set_fully_occupied_count (n := n) hb PHI S hS ⟨s₀, hs₀⟩
    rw [sum_load hb PHI (hS s₀ hs₀)]
    exact_mod_cast h
  have hpos : (0 : ℚ) < (Fintype.card (Keys n W) : ℚ) := by
    rw [card_keys]; positivity
  calc ((Finset.univ.filter (fun κ : Keys n W => ∀ s ∈ S, 0 < load b PHI κ s)).card : ℚ)
        / (Fintype.card (Keys n W) : ℚ)
      ≤ ((∑ κ : Keys n W, load b PHI κ s₀ : ℕ) : ℚ) / (Fintype.card (Keys n W) : ℚ) :=
        div_le_div_of_nonneg_right hcount hpos.le
    _ = n / 2 ^ b := expected_load hb PHI (hS s₀ hs₀) hn

-- ═══════════════════════════════════════════════════════════════════
-- §4b. THE SHARPER COVERAGE BOUND — `α^S.card`, by a union bound over the witness maps
-- ═══════════════════════════════════════════════════════════════════
-- `probe_set_fully_occupied_prob` reads the covering event through ONE member of `S` and gets the
-- single-slot Markov `α`. Covering all of `S` is harder, and the count says by how much: a covering
-- placement names a key per slot, that naming is injective (one key, one home), there are at most
-- `n^|S|` namings, and `fiber_lift_subset` counts each one.

/-- **`n` keys cover at most `n` slots.** A placement occupying every slot of `S` sends `S` into
    the image of its own `homeSlot`, which has at most `n` elements. So `S.card ≤ n` below is not a
    restriction — past it the covering event is empty. -/
theorem covering_needs_as_many_keys {b : ℕ} (PHI : (ZMod (2 ^ W))ˣ) (S : Finset ℕ)
    (κ : Keys n W) (hcov : ∀ s ∈ S, 0 < load b PHI κ s) : S.card ≤ n := by
  classical
  have hsub : S ⊆ Finset.univ.image (fun i : Fin n => homeSlot b PHI (κ i)) := by
    intro s hs
    obtain ⟨i, hi⟩ := Finset.card_pos.mp (hcov s hs)
    exact Finset.mem_image.mpr ⟨i, Finset.mem_univ i, (Finset.mem_filter.mp hi).2⟩
  calc S.card ≤ (Finset.univ.image (fun i : Fin n => homeSlot b PHI (κ i))).card :=
        Finset.card_le_card hsub
    _ ≤ (Finset.univ : Finset (Fin n)).card := Finset.card_image_le
    _ = n := by simp

/-- **The covering count, per WITNESS map — the union bound.** A placement that occupies every slot
    of `S` carries a map `S → Fin n` picking a key that homes at each `s`, and that map is injective
    because one key has one home. There are `n^S.card` such maps (`Finset.pi`); each pins `S.card`
    distinct keys to `S.card` fixed slots, which `fiber_lift_subset` counts at exactly
    `(2^(W−b))^S.card · (2^W)^(n−S.card)`; a non-injective map pins one key to two slots and counts
    `0`. Product: `n^S.card` times the single-slot fiber count raised to `S.card` — the factor
    `probe_set_fully_occupied_count` throws away. -/
theorem probe_set_fully_occupied_count_pow {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ)
    (S : Finset ℕ) (hS : ∀ s ∈ S, s < 2 ^ b) :
    (Finset.univ.filter (fun κ : Keys n W => ∀ s ∈ S, 0 < load b PHI κ s)).card
      ≤ n ^ S.card * ((2 ^ (W - b)) ^ S.card * (2 ^ W) ^ (n - S.card)) := by
  classical
  set F := S.pi (fun _ => (Finset.univ : Finset (Fin n))) with hF
  set B : (∀ a ∈ S, Fin n) → Finset (Keys n W) := fun f =>
    Finset.univ.filter (fun κ : Keys n W => ∀ s, ∀ hs : s ∈ S, homeSlot b PHI (κ (f s hs)) = s)
    with hB
  have hsub : (Finset.univ.filter (fun κ : Keys n W => ∀ s ∈ S, 0 < load b PHI κ s))
      ⊆ F.biUnion B := by
    intro κ hκ
    rw [Finset.mem_filter] at hκ
    have hex : ∀ s, ∀ hs : s ∈ S, ∃ i : Fin n, homeSlot b PHI (κ i) = s := by
      intro s hs
      have h0 := hκ.2 s hs
      unfold load at h0
      obtain ⟨i, hi⟩ := Finset.card_pos.mp h0
      exact ⟨i, (Finset.mem_filter.mp hi).2⟩
    choose f hf using hex
    refine Finset.mem_biUnion.mpr ⟨f, ?_, ?_⟩
    · exact Finset.mem_pi.mpr (fun a ha => Finset.mem_univ _)
    · exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, hf⟩
  have hterm : ∀ f : (∀ a ∈ S, Fin n),
      (B f).card ≤ (2 ^ (W - b)) ^ S.card * (2 ^ W) ^ (n - S.card) := by
    intro f
    by_cases hinj : ∀ s (hs : s ∈ S) s' (hs' : s' ∈ S), f s hs = f s' hs' → s = s'
    · set g : {x // x ∈ S} → Fin n := fun p => f p.1 p.2 with hg
      set T : Finset (Fin n) := S.attach.image g with hT
      set t : Fin n → ℕ := fun i => if h : ∃ p : {x // x ∈ S}, g p = i then (h.choose).1 else 0
        with ht
      have htval : ∀ p : {x // x ∈ S}, t (g p) = p.1 := by
        intro p
        have hex : ∃ q : {x // x ∈ S}, g q = g p := ⟨p, rfl⟩
        rw [ht]
        simp only [dif_pos hex]
        exact hinj _ hex.choose.2 _ p.2 hex.choose_spec
      have hTcard : T.card = S.card := by
        rw [hT, Finset.card_image_of_injOn, S.card_attach]
        intro p _ q _ hpq
        exact Subtype.ext (hinj _ p.2 _ q.2 hpq)
      have hTlt : ∀ i ∈ T, t i < 2 ^ b := by
        intro i hi
        obtain ⟨p, _, rfl⟩ := Finset.mem_image.mp hi
        rw [htval p]; exact hS p.1 p.2
      have hBsub : B f ⊆ Finset.univ.filter
          (fun κ : Keys n W => ∀ i ∈ T, homeSlot b PHI (κ i) = t i) := by
        intro κ hκ
        rw [hB, Finset.mem_filter] at hκ
        refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩
        intro i hi
        obtain ⟨p, _, rfl⟩ := Finset.mem_image.mp hi
        rw [htval p]
        exact hκ.2 p.1 p.2
      have hcount := fiber_lift_subset (n := n) hb PHI T t hTlt
      rw [hTcard] at hcount
      calc (B f).card ≤ (Finset.univ.filter
              (fun κ : Keys n W => ∀ i ∈ T, homeSlot b PHI (κ i) = t i)).card :=
            Finset.card_le_card hBsub
        _ = Fintype.card {κ : Keys n W // ∀ i ∈ T, homeSlot b PHI (κ i) = t i} :=
            (Fintype.card_subtype _).symm
        _ = (2 ^ (W - b)) ^ S.card * (2 ^ W) ^ (n - S.card) := hcount
    · have : B f = ∅ := by
        rw [Finset.eq_empty_iff_forall_notMem]
        intro κ hκ
        apply hinj
        intro s hs s' hs' heq
        rw [hB, Finset.mem_filter] at hκ
        have h1 := hκ.2 s hs
        have h2 := hκ.2 s' hs'
        rw [heq] at h1
        rw [← h1, h2]
      rw [this]
      simp
  calc (Finset.univ.filter (fun κ : Keys n W => ∀ s ∈ S, 0 < load b PHI κ s)).card
      ≤ (F.biUnion B).card := Finset.card_le_card hsub
    _ ≤ ∑ f ∈ F, (B f).card := Finset.card_biUnion_le
    _ ≤ ∑ _f ∈ F, ((2 ^ (W - b)) ^ S.card * (2 ^ W) ^ (n - S.card)) :=
        Finset.sum_le_sum (fun f _ => hterm f)
    _ = F.card * ((2 ^ (W - b)) ^ S.card * (2 ^ W) ^ (n - S.card)) := by
        rw [Finset.sum_const, smul_eq_mul]
    _ = n ^ S.card * ((2 ^ (W - b)) ^ S.card * (2 ^ W) ^ (n - S.card)) := by
        rw [hF, Finset.card_pi]
        simp

/-- **A whole probe-address set is occupied for at most an `α^S.card` fraction of key sets
    (headline).** `probe_set_fully_occupied_count_pow` over `card_keys`, `α = n/2^b`. One factor of
    `α` per slot covered — covering all of `S` IS counted as harder, which is exactly what
    `probe_set_fully_occupied_prob`'s single-slot Markov read threw away. At the operating `α ≈ 0.4`
    over the shipped table's `D = ⌈64/14⌉ = 5` windows that is `0.4^5 = 0.01024`, where the
    single-slot form permitted `0.4`. The hypothesis `S.card ≤ n` is free
    (`covering_needs_as_many_keys`). Still a MASS over i.i.d.-uniform key sets and nothing more:
    worst-case the event is certain (`AggregationBound.descent_can_be_wholly_blocked`), and `PHI` is
    public, so this is not keyed immunity. -/
theorem probe_set_fully_occupied_prob_pow {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ)
    (S : Finset ℕ) (hS : ∀ s ∈ S, s < 2 ^ b) (hSn : S.card ≤ n) :
    ((Finset.univ.filter (fun κ : Keys n W => ∀ s ∈ S, 0 < load b PHI κ s)).card : ℚ)
      / (Fintype.card (Keys n W) : ℚ) ≤ ((n : ℚ) / 2 ^ b) ^ S.card := by
  have hpos : (0 : ℚ) < (Fintype.card (Keys n W) : ℚ) := by rw [card_keys]; positivity
  have hle : (((Finset.univ.filter (fun κ : Keys n W => ∀ s ∈ S, 0 < load b PHI κ s)).card : ℚ))
      ≤ ((n ^ S.card * ((2 ^ (W - b)) ^ S.card * (2 ^ W) ^ (n - S.card)) : ℕ) : ℚ) := by
    exact_mod_cast probe_set_fully_occupied_count_pow (n := n) hb PHI S hS
  refine le_trans (div_le_div_of_nonneg_right hle hpos.le) (le_of_eq ?_)
  rw [card_keys]
  push_cast
  rw [div_pow, div_eq_div_iff (by positivity) (by positivity)]
  have e1 : ((2 : ℚ) ^ W) ^ n = ((2 : ℚ) ^ W) ^ S.card * ((2 : ℚ) ^ W) ^ (n - S.card) := by
    rw [← pow_add]; congr 1; omega
  have e2 : ((2 : ℚ) ^ (W - b)) ^ S.card * ((2 : ℚ) ^ b) ^ S.card = ((2 : ℚ) ^ W) ^ S.card := by
    rw [← mul_pow, ← pow_add]; congr 2; omega
  rw [e1]
  linear_combination ((n : ℚ) ^ S.card * ((2 : ℚ) ^ W) ^ (n - S.card)) * e2

-- ═══════════════════════════════════════════════════════════════════
-- §5. EXPECTED NUMBER OF OCCUPIED HOME SLOTS (item 6) — the home-hit envelope
-- ═══════════════════════════════════════════════════════════════════

/-- **Complement of `home_balanced`.** The coordinates that *miss* a fixed slot `s` number exactly
    `2^W − 2^(W−b)` — the whole universe minus the `2^(W−b)` that hit it. The average-case dual of
    `home_balanced`, and the per-key factor the empty-slot count is built from. -/
theorem miss_fiber_card {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) {s : ℕ} (hs : s < 2 ^ b) :
    (Finset.univ.filter (fun c : ZMod (2 ^ W) => homeSlot b PHI c ≠ s)).card
      = 2 ^ W - 2 ^ (W - b) := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  classical
  have h := Finset.card_filter_add_card_filter_not
    (s := (Finset.univ : Finset (ZMod (2 ^ W)))) (p := fun c => homeSlot b PHI c = s)
  rw [home_balanced hb PHI hs, Finset.card_univ, ZMod.card] at h
  simp only [ne_eq]
  omega

/-- **`miss_fiber_card`, lifted from one key to `n` keys — the empty-slot placement count.** The
    placements under which slot `s` receives *no* key are exactly those where every one of the `n`
    keys independently misses `s`: `(2^W − 2^(W−b))^n`. Same `Equiv.subtypePiEquivPi` pushforward as
    `fiber_lift`, run on the complement predicate. This is `#{κ : load(s) = 0}`. -/
theorem miss_fiber_lift {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) {s : ℕ} (hs : s < 2 ^ b) :
    Fintype.card {κ : Keys n W // ∀ i, homeSlot b PHI (κ i) ≠ s} = (2 ^ W - 2 ^ (W - b)) ^ n := by
  classical
  have step : {κ : Keys n W // ∀ i, homeSlot b PHI (κ i) ≠ s}
      ≃ ∀ i : Fin n, {c : ZMod (2 ^ W) // homeSlot b PHI c ≠ s} :=
    Equiv.subtypePiEquivPi (β := fun _ : Fin n => ZMod (2 ^ W))
      (p := fun _ c => homeSlot b PHI c ≠ s)
  have hval : Fintype.card {c : ZMod (2 ^ W) // homeSlot b PHI c ≠ s} = 2 ^ W - 2 ^ (W - b) := by
    rw [Fintype.card_subtype]; exact miss_fiber_card hb PHI hs
  rw [Fintype.card_congr step, Fintype.card_pi, Finset.prod_const, Finset.card_univ,
      Fintype.card_fin, hval]

/-- The number of *occupied* home slots under placement `κ` — slots that received at least one of
    the `n` keys. Each occupied slot stores exactly one value, so this is an upper bound on the
    number of keys resident at their own home slot (two home-resident keys cannot share a slot);
    `n − occupied` is therefore a floor on the displaced count. -/
def occupied (b : ℕ) (PHI : (ZMod (2 ^ W))ˣ) (κ : Keys n W) : ℕ :=
  ((Finset.range (2 ^ b)).filter (fun s => 0 < load b PHI κ s)).card

/-- **Total occupied-slot count, summed over every placement.** Swap the sums (slots vs.
    placements) and read each slot's occupancy off the complement: a slot is empty under
    `(2^W − 2^(W−b))^n` placements (`miss_fiber_lift`), so it is occupied under the remaining
    `(2^W)^n − (2^W − 2^(W−b))^n`; there are `2^b` slots. -/
theorem sum_occupied {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) :
    ∑ κ : Keys n W, occupied b PHI κ
      = 2 ^ b * ((2 ^ W) ^ n - (2 ^ W - 2 ^ (W - b)) ^ n) := by
  classical
  have hswap : ∑ κ : Keys n W, occupied b PHI κ
      = ∑ s ∈ Finset.range (2 ^ b),
          (Finset.univ.filter (fun κ : Keys n W => 0 < load b PHI κ s)).card := by
    unfold occupied
    simp_rw [Finset.card_filter]
    rw [Finset.sum_comm]
  have heach : ∀ s ∈ Finset.range (2 ^ b),
      (Finset.univ.filter (fun κ : Keys n W => 0 < load b PHI κ s)).card
        = (2 ^ W) ^ n - (2 ^ W - 2 ^ (W - b)) ^ n := by
    intro s hs'
    have hs : s < 2 ^ b := Finset.mem_range.mp hs'
    have h := Finset.card_filter_add_card_filter_not
      (s := (Finset.univ : Finset (Keys n W))) (p := fun κ => ∀ i, homeSlot b PHI (κ i) ≠ s)
    rw [Finset.card_univ, card_keys] at h
    have hmiss : (Finset.univ.filter (fun κ : Keys n W => ∀ i, homeSlot b PHI (κ i) ≠ s)).card
        = (2 ^ W - 2 ^ (W - b)) ^ n := by
      rw [← Fintype.card_subtype]; exact miss_fiber_lift hb PHI hs
    rw [hmiss] at h
    have hpe : Finset.univ.filter (fun κ : Keys n W => 0 < load b PHI κ s)
        = Finset.univ.filter (fun κ : Keys n W => ¬ (∀ i, homeSlot b PHI (κ i) ≠ s)) := by
      ext κ
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, load, Finset.card_pos,
        Finset.filter_nonempty_iff, ne_eq, not_forall, not_not]
    rw [hpe]
    omega
  rw [hswap, Finset.sum_congr rfl heach, Finset.sum_const, Finset.card_range, smul_eq_mul]

/-- **Expected occupied home slots = `2^b · (1 − (1 − 1/2^b)^n)` (headline, item 6).** The
    average number of distinct home slots hit by `n` i.i.d.-uniform keys — `sum_occupied` divided by
    `card_keys` — is the classical occupancy formula `m(1 − (1 − 1/m)^n)`, `m = 2^b` slots, proved on
    the real fold. Dividing by `n` gives `(1 − (1 − 1/2^b)^n)/α`, the exact average-case **ceiling on
    the home-hit fraction** (each occupied slot hosts at most one home-resident key). At the operating
    load `α = n/2^b ≈ 0.4` this ceiling is `≈ 0.82`; the average-case companion the audit
    flagged as missing: the home-hit number is no longer only measured, it has a proven envelope on
    the same `homeSlot`/`PHI`/`home_balanced` fold. -/
theorem expected_occupied_slots {b : ℕ} (hb : b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) :
    ((∑ κ : Keys n W, occupied b PHI κ : ℕ) : ℚ) / (Fintype.card (Keys n W) : ℚ)
      = 2 ^ b * (1 - (1 - 1 / 2 ^ b) ^ n) := by
  rw [sum_occupied hb PHI, card_keys]
  have hbase : (2 : ℕ) ^ (W - b) ≤ 2 ^ W := Nat.pow_le_pow_right (by norm_num) (by omega)
  have hle : ((2 ^ W - 2 ^ (W - b)) ^ n : ℕ) ≤ (2 ^ W) ^ n := Nat.pow_le_pow_left (Nat.sub_le _ _) n
  have hx : (2 : ℚ) ^ (W - b) = (2 ^ W) / (2 ^ b) := by
    rw [eq_div_iff (by positivity), ← pow_add]; congr 1; omega
  have hbaseq : (2 ^ W : ℚ) * (1 - 1 / 2 ^ b) = 2 ^ W - 2 ^ (W - b) := by rw [hx]; ring
  have hfac : ((2 ^ W : ℚ) - 2 ^ (W - b)) ^ n = ((2 ^ W : ℚ)) ^ n * (1 - 1 / 2 ^ b) ^ n := by
    rw [← hbaseq, mul_pow]
  push_cast [Nat.cast_sub hle, Nat.cast_sub hbase]
  rw [div_eq_iff (by positivity), hfac]
  ring

/-- Per-key form of `expected_occupied_slots`: the average-case ceiling on the home-hit
    fraction at load `α = n/m`. -/
def homeResidentCeiling (m n : ℕ) : ℚ := (m : ℚ) * (1 - (1 - 1 / (m : ℚ)) ^ n) / n

set_option exponentiation.threshold 512 in
/-- The "≈ 0.82" above, checked: at `m = 1024`, `n = 410` (`α ≈ 0.4`) the ceiling lies in
    `[0.82, 0.83]`. -/
theorem home_resident_ceiling_at_two_fifths :
    82 / 100 ≤ homeResidentCeiling 1024 410 ∧ homeResidentCeiling 1024 410 ≤ 83 / 100 := by
  unfold homeResidentCeiling
  norm_num

-- ═══════════════════════════════════════════════════════════════════
-- §6. THE SECOND WINDOW — the concrete spinor pair (flaw one, flaw two)
-- ═══════════════════════════════════════════════════════════════════
-- `SpinorReadout.spinor_depth_two` takes pair-injectivity as a hypothesis. This section supplies
-- the CONCRETE pair for the real fold: flaw one = `homeSlot` (the top `b`-bit window of `c·PHI`),
-- flaw two = `digit1` (the window directly beneath it). The pair is not injective on the whole
-- key space (pigeonhole; `SpinorFold.opaque_home_collides`) — the honest theorem is exact
-- counting: the two windows together ARE the top `2b`-bit window, so every pair theorem reduces
-- to the already-proven `home_balanced`/`pair_collision_count` at width `2b`. Double collisions
-- have exact mass `2^{-2b}`; the depth-2 readout confirms them exactly
-- (`SpinorReadout.resolve₂`). NOT provable for two DIFFERENT multipliers — see the guard below.

/-- Base-`p` positional uniqueness: `a·p + d` with `d < p` determines and is determined by
    `(a, d)`. The mod/div uniqueness both spinor-pair directions ride on. -/
private theorem mul_add_eq_iff {p a₁ d₁ a₂ d₂ : ℕ} (hp : 0 < p) (hd₁ : d₁ < p) (hd₂ : d₂ < p) :
    a₁ * p + d₁ = a₂ * p + d₂ ↔ a₁ = a₂ ∧ d₁ = d₂ := by
  constructor
  · intro h
    have hd : d₁ = d₂ := by
      have h1 := congrArg (· % p) h
      simpa [Nat.mul_add_mod, Nat.mod_eq_of_lt hd₁, Nat.mod_eq_of_lt hd₂] using h1
    subst hd
    exact ⟨Nat.eq_of_mul_eq_mul_right hp (Nat.add_right_cancel h), rfl⟩
  · rintro ⟨rfl, rfl⟩; rfl

/-- **The second flaw.** The next-from-top `b`-bit window of the fold — bits `[W−2b, W−b)` of
    `c·PHI`. `homeSlot` reads the top window (flaw one); this is the window directly beneath it
    (flaw two). Together they are the top `2b` bits — the concrete spinor pair of
    `SpinorReadout`. -/
def digit1 (b : ℕ) (PHI : (ZMod (2 ^ W))ˣ) (c : ZMod (2 ^ W)) : ℕ :=
  (c * (PHI : ZMod (2 ^ W))).val / 2 ^ (W - 2 * b) % 2 ^ b

theorem digit1_lt (b : ℕ) (PHI : (ZMod (2 ^ W))ˣ) (c : ZMod (2 ^ W)) :
    digit1 b PHI c < 2 ^ b :=
  Nat.mod_lt _ (by positivity)

/-- **The pair of windows IS the wide window.** `homeSlot b` and `digit1 b` are the quotient and
    remainder of `homeSlot (2b)` by `2^b`: the two flaws together read exactly the top `2b` bits.
    This is the whole reduction — every pair statement becomes a width-`2b` statement. -/
theorem wide_window_split {b : ℕ} (h2b : 2 * b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) (c : ZMod (2 ^ W)) :
    homeSlot (2 * b) PHI c = homeSlot b PHI c * 2 ^ b + digit1 b PHI c := by
  unfold homeSlot digit1
  have hsplit : (2 : ℕ) ^ (W - b) = 2 ^ (W - 2 * b) * 2 ^ b := by
    rw [← pow_add]; congr 1; omega
  rw [hsplit, ← Nat.div_div_eq_div_mul]
  exact (Nat.div_add_mod' _ _).symm

/-- **Pair collision ⟺ wide-window collision.** Two coordinates agree on BOTH flaws exactly when
    they agree on the top `2b`-bit window — the spinor pair's collision structure is the single
    wide window's, nothing extra. -/
theorem pair_windows_eq_wide_window {b : ℕ} (h2b : 2 * b ≤ W) (PHI : (ZMod (2 ^ W))ˣ)
    (c c' : ZMod (2 ^ W)) :
    (homeSlot b PHI c = homeSlot b PHI c' ∧ digit1 b PHI c = digit1 b PHI c')
      ↔ homeSlot (2 * b) PHI c = homeSlot (2 * b) PHI c' := by
  rw [wide_window_split h2b PHI c, wide_window_split h2b PHI c',
      mul_add_eq_iff (by positivity) (digit1_lt b PHI c) (digit1_lt b PHI c')]

/-- **Exact joint balance of the spinor pair.** Every `(flaw₁, flaw₂)` target is hit by exactly
    `2^(W−2b)` coordinates — `home_balanced` at width `2b`, read through the pair. The two flaws
    of ONE unit product are exactly jointly balanced; no independence assumption, an equation. -/
theorem top_pair_fiber_card {b : ℕ} (h2b : 2 * b ≤ W) (PHI : (ZMod (2 ^ W))ˣ)
    {s₁ s₂ : ℕ} (hs₁ : s₁ < 2 ^ b) (hs₂ : s₂ < 2 ^ b) :
    (Finset.univ.filter (fun c : ZMod (2 ^ W) =>
        homeSlot b PHI c = s₁ ∧ digit1 b PHI c = s₂)).card = 2 ^ (W - 2 * b) := by
  have h2 : (2 : ℕ) ^ (2 * b) = 2 ^ b * 2 ^ b := by rw [← pow_add]; congr 1; omega
  have hwide : s₁ * 2 ^ b + s₂ < 2 ^ (2 * b) := by
    calc s₁ * 2 ^ b + s₂ < s₁ * 2 ^ b + 2 ^ b := by omega
      _ = (s₁ + 1) * 2 ^ b := by ring
      _ ≤ 2 ^ b * 2 ^ b := Nat.mul_le_mul_right _ hs₁
      _ = 2 ^ (2 * b) := h2.symm
  have hset : Finset.univ.filter (fun c : ZMod (2 ^ W) =>
      homeSlot b PHI c = s₁ ∧ digit1 b PHI c = s₂)
      = Finset.univ.filter (fun c : ZMod (2 ^ W) =>
          homeSlot (2 * b) PHI c = s₁ * 2 ^ b + s₂) := by
    ext c
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    rw [wide_window_split h2b PHI c,
        mul_add_eq_iff (by positivity) (digit1_lt b PHI c) hs₂]
  rw [hset]
  exact home_balanced h2b PHI hwide

/-- **Double-collision count.** Placements where two fixed distinct keys collide on BOTH flaws —
    `pair_collision_count` at width `2b`, via the pair⟺wide-window reduction. -/
theorem double_collision_count {b : ℕ} (h2b : 2 * b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) {i j : Fin n}
    (hij : i ≠ j) :
    (Finset.univ.filter (fun κ : Keys n W =>
        homeSlot b PHI (κ i) = homeSlot b PHI (κ j)
          ∧ digit1 b PHI (κ i) = digit1 b PHI (κ j))).card
      = 2 ^ (2 * b) * (2 ^ (W - 2 * b)) ^ 2 * (2 ^ W) ^ (n - 2) := by
  have hpred : Finset.univ.filter (fun κ : Keys n W =>
      homeSlot b PHI (κ i) = homeSlot b PHI (κ j) ∧ digit1 b PHI (κ i) = digit1 b PHI (κ j))
      = Finset.univ.filter (fun κ : Keys n W =>
          homeSlot (2 * b) PHI (κ i) = homeSlot (2 * b) PHI (κ j)) := by
    ext κ
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact pair_windows_eq_wide_window h2b PHI _ _
  rw [hpred]
  exact pair_collision_count h2b PHI hij

/-- **Double-collision probability is exactly `2^{-2b}` (headline).** Two distinct i.i.d.-uniform
    keys collide on BOTH spinor flaws with probability exactly `1/2^(2b)` — the square of the
    single-flaw rate, not by an independence assumption but by the exact width-`2b` reduction.
    This is the mass on which the depth-2 readout (`SpinorReadout.resolve₂`) pays its one stored-
    identity confirm; everywhere else it resolves in ≤ 2 coordinate reads
    (`SpinorReadout.concrete_spinor_depth_two`). -/
theorem per_pair_double_collision_prob {b : ℕ} (h2b : 2 * b ≤ W) (PHI : (ZMod (2 ^ W))ˣ)
    {i j : Fin n} (hij : i ≠ j) (hn : 2 ≤ n) :
    ((Finset.univ.filter (fun κ : Keys n W =>
        homeSlot b PHI (κ i) = homeSlot b PHI (κ j)
          ∧ digit1 b PHI (κ i) = digit1 b PHI (κ j))).card : ℚ)
      / (Fintype.card (Keys n W) : ℚ) = 1 / 2 ^ (2 * b) := by
  have hpred : Finset.univ.filter (fun κ : Keys n W =>
      homeSlot b PHI (κ i) = homeSlot b PHI (κ j) ∧ digit1 b PHI (κ i) = digit1 b PHI (κ j))
      = Finset.univ.filter (fun κ : Keys n W =>
          homeSlot (2 * b) PHI (κ i) = homeSlot (2 * b) PHI (κ j)) := by
    ext κ
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact pair_windows_eq_wide_window h2b PHI _ _
  rw [hpred]
  exact per_pair_collision_prob h2b PHI hij hn

-- THE HONESTY GUARD — two DIFFERENT unit multipliers are NOT jointly balanced. At `W = 4, b = 1`,
-- multipliers `1` and `3` (both odd, both units): the fiber of `(0,0)` under
-- `c ↦ (top-bit(c·1), top-bit(c·3))` has 5 elements, not `2^(4−2) = 4`. The exact joint balance
-- above holds for two windows of ONE unit product; it FAILS for two products (a discrepancy
-- object, not an equation). Do not "generalize" the spinor pair to a second multiplier.
#guard (Finset.univ.filter (fun c : ZMod (2 ^ 4) =>
    (c * 1).val / 2 ^ 3 = 0 ∧ (c * 3).val / 2 ^ 3 = 0)).card = 5

end ProbabilisticBounds
end Sanguine
