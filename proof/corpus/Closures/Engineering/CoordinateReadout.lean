import Substrate.Algebra.Security.EncryptionProjection
import Closures.Engineering.RadixDescent
import Substrate.Ski.StateVectorMemoryModel
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.List.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.Logic.Function.Basic
import Mathlib.Tactic.Common
import Mathlib.Tactic.Positivity

/-!
# The Readout — the address confirms without the key, resolves in depth ≤ 2, sizes to `W−b` bits
@name The Address Confirms Without Reading the Key
@child To check you found the right box you can either read the label inside (slow, you open the box)
  or check the box's own address on the outside. If every address is unique, the address alone tells
  you — you never open the box. Two labels always pin it down exactly, never three. And once you know
  which box, you don't need to keep the whole address either — just the part the box's position on
  the shelf doesn't already tell you.
@scientist Three theorems, one readout. (1) A lookup confirms a hit either by comparing the full stored
  key or the stored address; under an injective address the two coincide, so the readout confirms with
  zero key accesses (`namespace CoordinateReadout`). (2) A two-flaw coordinate address separates any two
  keys within depth ≤ 2 — the spinor double cover — and the concrete flaw pair is exactly counted, not
  assumed independent (`namespace SpinorReadout`). (3) The slot (top `b` bits) is free as the address, so
  a home-resident key needs only the low `W−b` bits stored — the seed (`namespace SeedSizing`).
@engineer `addrConfirm_correct`/`exact_read_is_read` (zero/no-compare); `resolve₂`/`concrete_spinor_depth_two`
  (depth-2, exact confirm on the `2^{-2b}` mass); `residual`/`decode_homeSlot_residual` (the lossless
  `W−b`-bit split). All three ride the same injective `PHI`-fold as `RadixDescent`.

This is the closure of `hashtrinity`'s coordinate `get`. A hash table with a partial control tag
(SwissTable's 7-bit `h2`) can never confirm a hit from the tag alone — the tag is non-injective, so a
match may be a false positive — and must therefore **deref and compare the full key** on every hit (a
second memory access, the payload/key bytes). The substrate stores the **full coordinate** as the
slot's identity; the coordinate map is injective (`RadixDescent.fold_injective`: multiplication by the
unit `PHI` is injective), so a coordinate match *forces* key equality (`addrConfirm_correct`, proved
below straight out of `Injective`) and the value is returned with **no key access at all**. Fewer
accesses is fewer, unconditionally: the readout cannot be slower than the
compare on the confirm step, and is strictly cheaper on it. There is no construction under which the
extra key access the tag table pays makes it faster — that is what "by construction" means here.

The three sections below are one closure read three ways: `CoordinateReadout` is the confirm step (0
vs ≥1 key accesses), `SpinorReadout` is the recursion depth of getting there (≤ 2, never a growing
radix chain), `SeedSizing` is the storage cost once you're there (`W−b` bits, not `W`). All three cite
`RadixDescent`'s injective fold as their common root.

## What is proved here vs what is read onto a machine

* **Proved (native):** the confirm step — `addrConfirm_correct`, `partial_tag_false_positive`,
  `readout_strictly_fewer_key_accesses`, bundled by `readout_beats_compare_by_construction`; the
  footprint arithmetic — `footprint_mono`, `readout_footprint_halves`,
  `footprint_prescribes_the_split`, and the write-side `blockTransitions_le_length` /
  `sorted_blockTransitions_le`; the depth — `separate_within_depth_two`, `one_flaw_blind`,
  `spinor_depth_two`, `resolve₂_correct`, `resolve₂_touch_iff_double_collision`,
  `concrete_spinor_depth_two`, `two_band_recall`; the sizing — `residual_lt`,
  `decode_homeSlot_residual`, `seed_width`.
* **Proved (riding):** injectivity of the coordinate fold is `RadixDescent.fold_injective`
  (multiplication by the unit `PHI`); the cells the footprint counts are
  `StateVectorMemoryModel`'s. Both are used, neither is reproved here.
* **Anchor:** that a `B`-cell block **is** a cache line, that a slot width in cells
  **is** a real layout, and that fewer key accesses **is** less time. `n`, `w` and `B` are parameters
  a caller supplies; the theorems fix only the block count `⌈n·w/B⌉` and the `0 < 1` confirm-access
  gap, and predict no time. The `100k` mid-cache cost named in `footprint_prescribes_the_split` is a
  measurement, not a consequence of anything below.
* **Anchor:** that the two flaws of the concrete address **are** the two null
  directions of one spinor. That a coordinate match **is** key equality is not a bridge and is not
  read onto anything — `addrConfirm_correct` proves it above, out of `Injective addr`'s definition.
* **Open:** the three-distance theorem — the golden rotation's orbit gaps take at most three values,
  the deterministic equidistribution of `c ↦ c + φ` on the circle. It would replace
  `ProbabilisticBounds`' i.i.d.-uniform assumption with a deterministic guarantee for structured
  (e.g. sequential) key streams. Named again at the write face below; unproved in this corpus.
-/

namespace Sanguine
namespace CoordinateReadout

open Function

variable {K C : Type} [DecidableEq K] [DecidableEq C]

-- ═══════════════════════════════════════════════════════════════════
-- THE CONFIRM STEP — compare the stored fingerprint to the query's
-- ═══════════════════════════════════════════════════════════════════

/-- The confirm a lookup runs at the candidate slot: compare the **stored fingerprint** `addr stored`
    to the query's `addr q`. For the coordinate readout `addr` is the full `u64` coordinate — a word
    already loaded with the slot; the key is never touched here. -/
def addrConfirm (addr : K → C) (stored q : K) : Bool := decide (addr stored = addr q)

-- ═══════════════════════════════════════════════════════════════════
-- THE INJECTIVE ADDRESS CONFIRMS EXACTLY — WITH ZERO KEY ACCESSES
-- ═══════════════════════════════════════════════════════════════════

/-- **The address confirm IS the membership test, when the address is injective.** For an injective
    `addr`, `addrConfirm addr stored q` equals the exact test `stored = q` — reading only the two
    address words, never the keys. So the readout confirms a hit with **zero key accesses**: the
    address is the key, which is `Injective addr` read at the confirm step and nothing more. In
    `hashtrinity` `addr` is `k ↦ k·PHI` restricted, injective by `RadixDescent.fold_injective`. -/
theorem addrConfirm_correct (addr : K → C) (hinj : Injective addr) (stored q : K) :
    addrConfirm addr stored q = decide (stored = q) := by
  unfold addrConfirm
  by_cases h : stored = q
  · subst h; simp
  · have hne : addr stored ≠ addr q := fun he => h (hinj he)
    rw [decide_eq_false_iff_not.mpr hne, decide_eq_false_iff_not.mpr h]

/-- The readout's confirm touches the key **zero** times — it reads only the address words. -/
def readoutKeyAccesses : ℕ := 0

-- ═══════════════════════════════════════════════════════════════════
-- A LOSSY TAG IS NON-INJECTIVE — FALSE POSITIVES FORCE THE KEY COMPARE
-- ═══════════════════════════════════════════════════════════════════

omit [DecidableEq K] in
/-- **A non-injective fingerprint has a false positive.** If `tag` is not injective (a 7-bit control
    byte over a large key space cannot be), there exist **distinct** keys `a ≠ b` with
    `addrConfirm tag a b = true`: the tag match does not imply the keys are equal. So the tag alone is
    *unsound* as a membership test — a correct lookup on it must additionally compare the full keys. -/
theorem partial_tag_false_positive (tag : K → C) (hnon : ¬ Injective tag) :
    ∃ a b : K, addrConfirm tag a b = true ∧ a ≠ b := by
  obtain ⟨a, b, hab, hne⟩ := Function.not_injective_iff.mp hnon
  exact ⟨a, b, by simp [addrConfirm, hab], hne⟩

/-- To be correct on a non-injective tag, the confirm must fall back to the full key compare: at least
    **one** key access. -/
def compareKeyAccesses : ℕ := 1

/-- **Fewer is fewer.** The coordinate readout's key accesses on a confirm (`0`) are strictly below the
    compare's (`1`). Combined with `addrConfirm_correct` (the readout is *correct* at `0`) and
    `partial_tag_false_positive` (the tag is *incorrect* at `0`, forcing the `1`), the readout does a
    correct lookup at strictly lower confirm cost than any lossy-tag table — no assumption needed. -/
theorem readout_strictly_fewer_key_accesses : readoutKeyAccesses < compareKeyAccesses := by decide

-- ═══════════════════════════════════════════════════════════════════
-- THE HEADLINE — THE READOUT BEATS THE COMPARE BY CONSTRUCTION
-- ═══════════════════════════════════════════════════════════════════

/-- **The coordinate readout beats the compare, by construction (headline).** For an injective address
    (the full coordinate) and a non-injective tag (a lossy 7-bit fingerprint):

    1. **the readout is correct at zero key accesses** — the address confirm equals the exact
       membership test, reading only the address word (`addrConfirm_correct`);
    2. **the tag is incorrect at zero key accesses** — it admits a false positive `a ≠ b` with
       `confirm = true` (`partial_tag_false_positive`), so soundness *forces* a key compare;
    3. **the confirm cost is strictly lower** — `0 < 1` (`readout_strictly_fewer_key_accesses`).

    So the coordinate readout resolves a hit with strictly fewer key/payload accesses than any table
    that filters on a lossy tag — every SwissTable-family map, hashbrown included. This is not a tuning
    claim; it is the count. The address is the key (`EncryptionProjection`); reading it is the whole
    lookup, and there is nothing left to compare. -/
theorem readout_beats_compare_by_construction
    (addr tag : K → C) (hinj : Injective addr) (hnon : ¬ Injective tag) :
    (∀ stored q : K, addrConfirm addr stored q = decide (stored = q))
      ∧ (∃ a b : K, addrConfirm tag a b = true ∧ a ≠ b)
      ∧ readoutKeyAccesses < compareKeyAccesses :=
  ⟨addrConfirm_correct addr hinj,
   partial_tag_false_positive tag hnon,
   readout_strictly_fewer_key_accesses⟩

-- ═══════════════════════════════════════════════════════════════════
-- WHAT "SLOT FOOTPRINT" IS — cells of the state vector, counted in blocks
-- ═══════════════════════════════════════════════════════════════════
-- Not an appeal to physical cache. In the substrate, memory IS the state vector
-- (`StateVectorMemoryModel`: a `List` read by `load` at a depth; `alloca_length`: length is depth;
-- `gepOffset` composes depths additively). A slot of **width `w`** is `w` consecutive cells; `n` slots
-- are `n·w` cells; a memory **block** (the unit fetched together — a cache line, but as a substrate
-- parameter `B`, not a machine constant) holds `B` cells. So the table's "footprint" is the number of
-- blocks its cells span — `⌈n·w / B⌉`. A number. The hardware is an `Isa` value (`Song.Metal`); the
-- memory model is ours, so this number is ours to minimise, by construction.

/-- The **footprint** of `n` slots of width `w` under block size `B`: the count of state-vector blocks
    they span, `⌈n·w / B⌉`. This is the exact content of "slot footprint" — the cells of
    `StateVectorMemoryModel` a scan/pipeline of the table must fetch, in blocks. -/
def footprint (n w B : Nat) : Nat := (n * w + (B - 1)) / B

/-- **Footprint is monotone in the slot width.** A narrower slot (`w' ≤ w`) spans no more blocks:
    `footprint n w' B ≤ footprint n w B`. So moving a field out of the slot cannot increase the
    footprint — and (below) strictly shrinks it when it clears a block. -/
theorem footprint_mono {n w' w B : Nat} (h : w' ≤ w) : footprint n w' B ≤ footprint n w B := by
  unfold footprint
  exact Nat.div_le_div_right (by have := Nat.mul_le_mul_left n h; omega)

/-- **The key out of the readout slot strictly shrinks the footprint.** A readout of `n` slots wide
    enough to fetch the narrow `(coord,value)` layout in one block but not the wide `(coord,key,value)`
    layout: narrow width `2`, wide width `4` (the `&str` key is two words), block `B = 2·n` — one block
    vs two. The key doubles the readout's footprint; storing it in a separate state vector removes it. -/
theorem readout_footprint_halves (n : Nat) (hn : 1 ≤ n) :
    footprint n 2 (2 * n) < footprint n 4 (2 * n) := by
  have hnw : n * 2 + (2 * n - 1) = 4 * n - 1 := by omega
  have hww : n * 4 + (2 * n - 1) = 6 * n - 1 := by omega
  unfold footprint
  rw [hnw, hww]
  have h1 : (4 * n - 1) / (2 * n) = 1 := by
    rw [Nat.div_eq_of_lt_le] <;> omega
  have h2 : (6 * n - 1) / (2 * n) = 2 := by
    rw [Nat.div_eq_of_lt_le] <;> omega
  omega

-- concrete: 1M slots, 8-cell (64-byte) block; narrow (coord,value)=2 cells halves the wide=4 count.
#guard footprint 1000000 2 8 < footprint 1000000 4 8   -- 250000 < 500000

/-- **The substrate prescribes the split (headline).** "Slot footprint" is `⌈n·w/B⌉` blocks of the
    state vector (`footprint`, on `StateVectorMemoryModel`) — a number, not a machine I hide behind.
    Two facts fix the layout with no reference to any particular hardware (which is only an `Isa`
    value, `Song.Metal`):

    1. **zero key access** — the coordinate readout confirms a hit reading only the address word
       (`addrConfirm_correct`); the key is never on the hot path;
    2. **footprint monotone in width** — a narrower readout slot spans no more blocks
       (`footprint_mono`), strictly fewer when the key would overflow a block
       (`readout_footprint_halves`).

    So the memory model itself says: store the readout `(coord, value)` narrow, the key wide and apart.
    The 100k mid-cache cost was the key riding in the readout slot — a footprint the substrate never
    required. Fixing it is construction, not tuning. -/
theorem footprint_prescribes_the_split (addr : K → C) (hinj : Injective addr) (n : Nat) (hn : 1 ≤ n) :
    (∀ stored q : K, addrConfirm addr stored q = decide (stored = q))
      ∧ (∀ {w' w B : Nat}, w' ≤ w → footprint n w' B ≤ footprint n w B)
      ∧ footprint n 2 (2 * n) < footprint n 4 (2 * n) :=
  ⟨addrConfirm_correct addr hinj, fun h => footprint_mono h, readout_footprint_halves n hn⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE WRITE FACE — the build's footprint dual: scattered writes pay per WRITE,
-- monotone writes pay per BLOCK
-- ═══════════════════════════════════════════════════════════════════
-- `footprint` above counts the blocks a READ must fetch. The build faces the dual: a write
-- SEQUENCE pays each time it crosses a block boundary (a fresh line fetched, an old one
-- evicted). Arrival-order insertion scatters placements around the table — the golden rotation
-- (`c·PHI`, Fibonacci hashing = the irrational rotation on SO(2), the Being cell's rotor face)
-- equidistributes homes (`RadixDescent.home_balanced`), so consecutive arrivals land in
-- unrelated blocks and the build pays ~one crossing PER WRITE. The monotone gauge — placements
-- in home order, licensed meaning-free by `HashTrinityCore.build_gauge_invariance` — walks the
-- circle once and pays ~one crossing PER BLOCK: a `B/w`-fold reduction, the exact size of the
-- opaque-build gap. The sort key is not foreign machinery: it IS the coordinate's radix digits
-- (`RadixDescent.digit`), and partitioning to depth 2 (flaw one, then flaw two — the spinor pair,
-- `SpinorReadout.concrete_spinor_depth_two`) already brings each partition under cache size.
--
-- NAMED OPEN SEAM (deep face, not proven here): the three-distance theorem — the golden
-- rotation's orbit gaps take at most three values, the deterministic equidistribution of
-- `c ↦ c + φ` on the circle. It would upgrade `ProbabilisticBounds`' i.i.d.-uniform assumption
-- to a deterministic guarantee for structured (e.g. sequential) key streams. Heavy Mathlib
-- work; named, not blocked on.

/-- **Block-boundary crossings of a write sequence** — the streaming cost: every adjacent pair
    of writes landing in different `B`-blocks is a fresh line fetch. The write-side dual of
    `footprint`. -/
def blockTransitions (B : Nat) : List Nat → Nat
  | [] => 0
  | [_] => 0
  | a :: b :: rest => (if a / B = b / B then 0 else 1) + blockTransitions B (b :: rest)

/-- Crossings never exceed the write count: each adjacent pair contributes at most one. -/
theorem blockTransitions_le_length (B : Nat) :
    ∀ as : List Nat, blockTransitions B as ≤ as.length
  | [] => Nat.le_refl 0
  | [_] => Nat.zero_le 1
  | a :: b :: rest => by
      have ih := blockTransitions_le_length B (b :: rest)
      unfold blockTransitions
      split <;> simp_all <;> omega

/-- The strengthened bound the induction carries: a sorted write sequence with all writes in
    `[a₀, L)` crosses at most `L/B − a₀/B` block boundaries — each crossing strictly raises the
    (monotone) block index, which starts at ≥ `a₀/B` and cannot reach `L/B`. -/
theorem sorted_blockTransitions_le_from (B : Nat) :
    ∀ (as : List Nat), as.Pairwise (· ≤ ·) → ∀ L a₀, (∀ a ∈ as, a < L) → (∀ a ∈ as, a₀ ≤ a) →
      blockTransitions B as ≤ L / B - a₀ / B
  | [], _, _, _, _, _ => by simp [blockTransitions]
  | [_], _, _, _, _, _ => by simp [blockTransitions]
  | a :: b :: rest, hsort, L, a₀, hL, ha₀ => by
      obtain ⟨hle, hsort'⟩ := List.pairwise_cons.mp hsort
      have hab : a ≤ b := hle b (List.mem_cons_self ..)
      have hb : ∀ x ∈ b :: rest, b ≤ x := by
        intro x hx
        rcases List.mem_cons.mp hx with h | h
        · exact h ▸ le_refl b
        · exact (List.pairwise_cons.mp hsort').1 x h
      have hL' : ∀ x ∈ b :: rest, x < L := fun x hx => hL x (List.mem_cons_of_mem a hx)
      have ih := sorted_blockTransitions_le_from B (b :: rest) hsort' L b hL' hb
      have hd₀ : a₀ / B ≤ a / B := Nat.div_le_div_right (ha₀ a (List.mem_cons_self ..))
      have hdab : a / B ≤ b / B := Nat.div_le_div_right hab
      have hbL : b / B ≤ L / B := Nat.div_le_div_right (le_of_lt (hL b (by simp)))
      unfold blockTransitions
      split
      · omega
      · omega

/-- **The monotone write gauge pays per block, not per write (headline).** A sorted write
    sequence below `L` crosses at most `L/B` block boundaries — the footprint floor — where a
    scattered sequence pays up to one crossing per write (`blockTransitions_le_length`, attained
    by the #guard witness below: the SAME six writes cost 5 crossings scattered, 1 sorted).
    Sorting the placements — licensed meaning-free by `HashTrinityCore.build_gauge_invariance` —
    converts the build from per-write to per-block memory traffic: the `B/w`-fold write-side
    saving that is the opaque build's gap. -/
theorem sorted_blockTransitions_le (B : Nat) (as : List Nat) (hsort : as.Pairwise (· ≤ ·))
    (L : Nat) (hL : ∀ a ∈ as, a < L) :
    blockTransitions B as ≤ L / B := by
  have h := sorted_blockTransitions_le_from B as hsort L 0 hL (fun a _ => Nat.zero_le a)
  simpa [Nat.zero_div] using h

-- The same six writes (B = 64): arrival order ping-pongs between two blocks and pays 5
-- crossings; home order walks each block once and pays 1. The write face, concretely.
#guard blockTransitions 64 [0, 64, 1, 65, 2, 66] = 5
#guard blockTransitions 64 [0, 1, 2, 64, 65, 66] = 1

-- ═══════════════════════════════════════════════════════════════════
-- THE COORDINV LICENSE — a round-trip decoder makes the exact read free
-- ═══════════════════════════════════════════════════════════════════

-- The Rust `CoordInv` contract is a decoder: `from_coord (coord k) = k`. That single equation is
-- all the injectivity `addrConfirm_correct` needs — so `get_exact` on a CoordInv key IS `get`:
-- the address confirm is already the bit-exact membership test, and the key compare it performs
-- today is redundant work. One deliberate exclusion in the Rust: `coord_of` nudges the reserved
-- `u64::MAX` to `EMPTY − 1`, so injectivity has a single defect point; the compare must stay for
-- exactly that coordinate (and can go everywhere else).

omit [DecidableEq K] [DecidableEq C] in
/-- **A round-tripping decoder is an injectivity proof.** `from_coord (coord k) = k` for every `k`
    forces `coord` injective — the `CoordInv` contract, `EncryptionProjection.injective_iff_decryptor_exists`
    read left-to-right as a one-liner. -/
theorem roundtrip_injective (coord : K → C) (from_coord : C → K)
    (hrt : ∀ k, from_coord (coord k) = k) : Injective coord :=
  fun a b h => by rw [← hrt a, ← hrt b, h]

/-- **The exact read is the read (the no-compare license).** For a key type with a round-tripping
    decoder, the address confirm equals the exact membership test — `get_exact`'s extra key
    compare on the `CoordInv` path confirms nothing the coordinate hasn't already decided. -/
theorem exact_read_is_read (coord : K → C) (from_coord : C → K)
    (hrt : ∀ k, from_coord (coord k) = k) (stored q : K) :
    addrConfirm coord stored q = decide (stored = q) :=
  addrConfirm_correct coord (roundtrip_injective coord from_coord hrt) stored q

-- ═══════════════════════════════════════════════════════════════════
-- APPLICABILITY BOUNDARY — where the zero-access confirm does and does NOT apply
-- ═══════════════════════════════════════════════════════════════════
-- The count above (`readout_beats_compare_by_construction`: 0 < 1 key accesses on the
-- CONFIRM) eliminates the key compare when the VALUE the lookup returns is determined by
-- the coordinate alone — the map's case: confirm the address, return the payload, the key
-- bytes are never needed. It does NOT license skipping the source read in an LZ MATCH
-- FINDER, and misreading it there cost a measured regression (hashtrinity turbo,
-- 2026-07-10: ratio 1.728→1.347, enc 538→293 MB/s). The finder's "value" is the match
-- LENGTH — a function of the SOURCE BYTES at the candidate position — so the extension
-- scan (`lce`) must run on every ACCEPT no matter how the candidate was confirmed. What
-- the confirm mechanism decides is only the REJECT path: a lossy b-bit tag rejects with
-- zero source access at rate 1 − 2⁻ᵇ (`partial_tag_false_positive` bounds what leaks
-- through; `GroupProbe.tag_width_is_maximal` fixes b), and the 2⁻ᵇ false-accept mass pays
-- one wasted extension — which the accept path was paying anyway. Storing the full k-gram
-- coordinate in the cell buys nothing on accepts, widens the cell (`footprint_mono`
-- charges for it), and narrows the effective hash. The theorem's precondition, stated
-- once: **the readout wins where reading the address ends the transaction; where the
-- transaction's value lives in the pointed-to bytes, the address only gates the pointer.**
--
-- SECOND BOUNDARY — THE CONFIRM IS NOT THE SWEEP, AND NOT THE GROW. Read
-- `readout_beats_compare_by_construction`'s statement: under `(hinj : Injective addr)` and
-- `(hnon : ¬ Injective tag)` it concludes `addrConfirm addr stored q = decide (stored = q)`,
-- a false positive for `tag`, and `readoutKeyAccesses < compareKeyAccesses` — `0 < 1`. Its
-- quantifiers are `stored q : K`, at an ALREADY-LOCATED slot. There is no capacity in it, no
-- slot set, no live/empty, and no write. So it does NOT license refusing a per-slot
-- OCCUPANCY witness, and misreading it there cost two measured regressions (hashtrinity,
-- 2026-07-27): full iteration 0.44–0.81× vs `std` at every shape and both key species — at
-- n = 65536, ~2 MB streamed against hashbrown's ~1.15 MB, and the measured 0.45–0.53× is that
-- byte ratio — and the un-presized build 0.76–1.09× from n ≥ 1024, with `with_capacity(4)`
-- × 100k at 0.38× where the same shape via `new()` is 6.03× (`vec![EMPTY; n]` memsets 8 B/slot
-- against hashbrown's 1, and `EMPTY = u64::MAX ≠ 0` forecloses `alloc_zeroed`). A control byte
-- or a live bitmap read to answer "which slots are live" is not confirming a key against a
-- query — it answers a question this theorem does not range over, and (the crux) it carries no
-- bits of the key, so it can neither skip a compare nor weaken `addrConfirm`'s injectivity
-- conclusion. What DOES engage this theorem is a witness carrying HASH bits (SwissTable's
-- `h2`): non-injective on the key, so `partial_tag_false_positive` applies and the key access
-- returns. The three faces, their three cost measures on the one width parameter, and the
-- `EMPTY = 0` question (decided: the reserved sentinel triple is gauge — same defect, same
-- one-compare liveness test, same confirm) are `Closures.Engineering.OccupancySweep`. The
-- precondition, stated once more: **this theorem prices IDENTITY per slot, not OCCUPANCY per
-- slot; the second is a different question with the opposite width preference, and the two are
-- independent.**

end CoordinateReadout

-- ═══════════════════════════════════════════════════════════════════
-- namespace SpinorReadout — the readout resolves in depth ≤ 2, the double cover
-- ═══════════════════════════════════════════════════════════════════
-- @scientist A two-flaw coordinate address (the two null cones of `NullReadout`, `nNull`/`nNull'`)
-- has an injective PAIR but non-injective components; so any two distinct keys differ in flaw one
-- or flaw two, and are separated in depth ≤ 2 — the spinor double cover (`SpinorDoubleCover`), a
-- constant, where the single-flaw radix descent separates only within `⌈W/b⌉`, which grows with
-- the word. `RadixDescent.chain_separates_within_radix` bounds the single-projection probe by the
-- radix length `D`. But the readout is a spinor: `NullReadout` proves the state is recovered from
-- two boundary flaws in closed form, and one flaw alone is blind. Two, by construction — not `D`.

namespace SpinorReadout

open Function

variable {K C : Type}

/-- **Two flaws separate distinct keys (the spinor cap).** With a two-flaw address `(f₁, f₂)` whose
    PAIR is injective — the two cones together recover the key, exactly `NullReadout.readout_injective`
    (the two boundary pairings recover the whole state) — any two distinct keys differ in `f₁` or in
    `f₂`. So the readout distinguishes them by reading at most the two flaws: **depth ≤ 2**. There is
    no third flaw; the double cover closes at two sheets. -/
theorem separate_within_depth_two {f₁ f₂ : K → C}
    (hpair : Injective (fun k => (f₁ k, f₂ k))) {k k' : K} (hne : k ≠ k') :
    f₁ k ≠ f₁ k' ∨ f₂ k ≠ f₂ k' := by
  by_contra h
  push Not at h
  apply hne
  apply hpair
  simp only [h.1, h.2]

/-- **One flaw is blind (the cover needs both sheets).** If a single flaw `f₁` is not injective —
    which a bounded projection of a large key space must not be — some distinct keys share it, and
    depth 1 cannot separate them: the second flaw is required (`NullReadout.one_flaw_is_blind`, the
    single-cone reading has a nonzero kernel). So the depth-2 cap is not slack — depth 1 genuinely
    fails, depth 2 genuinely suffices. -/
theorem one_flaw_blind {f₁ : K → C} (hnon : ¬ Injective f₁) :
    ∃ k k' : K, f₁ k = f₁ k' ∧ k ≠ k' :=
  Function.not_injective_iff.mp hnon

/-- **The spinor depth-two readout (headline).** For a two-flaw address whose pair is injective (the
    two cones recover the key) but whose first flaw alone is not (blind on its own):

    1. **two flaws suffice** — distinct keys differ in `f₁` or `f₂` (`separate_within_depth_two`), so a
       lookup resolves any key by reading at most two coordinate projections and combining them
       (`NullReadout.decodeNull`) — recursion depth **≤ 2**;
    2. **one flaw fails** — depth 1 cannot separate the `f₁`-collisions (`one_flaw_blind`);
    3. so the readout's recursion depth is **exactly 2** — the spinor double cover
       (`SpinorDoubleCover`: two sheets, `4π`). A constant, independent of the word width.

    Contrast `RadixDescent.chain_separates_within_radix`: the single-projection probe separates keys
    only within `D = ⌈W/b⌉` steps, a walk that *grows with the word*. The two-flaw readout caps the
    collision recursion at the double cover — two reads, closed form, no chain. -/
theorem spinor_depth_two {f₁ f₂ : K → C}
    (hpair : Injective (fun k => (f₁ k, f₂ k))) (hblind : ¬ Injective f₁) :
    (∀ k k' : K, k ≠ k' → f₁ k ≠ f₁ k' ∨ f₂ k ≠ f₂ k')
      ∧ (∃ k k' : K, f₁ k = f₁ k' ∧ k ≠ k') :=
  ⟨fun _ _ hne => separate_within_depth_two hpair hne, one_flaw_blind hblind⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE CONCRETE READOUT — resolve₂: exact for ALL keys, one confirm only on double collision
-- ═══════════════════════════════════════════════════════════════════
-- `spinor_depth_two` above takes pair-injectivity as a hypothesis, which pigeonhole denies for
-- bounded flaws over a large key space (`SpinorFold.opaque_home_collides`). `resolve₂` is the
-- readout that needs NO such hypothesis: read flaw one, read flaw two, and touch the stored
-- identity only when both collide — which happens on an exactly-counted `2^{-2b}` mass
-- (`ProbabilisticBounds.per_pair_double_collision_prob`), where the confirm is exact. Depth 2 is
-- structural; the residual is measure, not design.

/-- **The depth-2 readout with exact confirm.** Compare flaw one (the word already loaded); if it
    separates, done at depth 1. Compare flaw two; if it separates, done at depth 2. Only when both
    collide, pay the ONE stored-identity access — the second component counts it. -/
def resolve₂ [DecidableEq K] [DecidableEq C] (f₁ f₂ : K → C) (stored q : K) : Bool × ℕ :=
  if f₁ stored ≠ f₁ q then (false, 0)
  else if f₂ stored ≠ f₂ q then (false, 0)
  else (decide (stored = q), 1)

/-- **`resolve₂` is exact — for every key pair, no hypothesis.** Its verdict equals the exact
    membership test: a flaw mismatch already implies the keys differ, and on a double collision
    the confirm IS the exact test. Correctness never leans on injectivity. -/
theorem resolve₂_correct [DecidableEq K] [DecidableEq C] (f₁ f₂ : K → C) (stored q : K) :
    (resolve₂ f₁ f₂ stored q).1 = decide (stored = q) := by
  unfold resolve₂
  split_ifs with h1 h2
  · have hne : stored ≠ q := fun he => h1 (congrArg f₁ he)
    simp [hne]
  · have hne : stored ≠ q := fun he => h2 (congrArg f₂ he)
    simp [hne]
  · rfl

/-- **Separated pairs are free.** When either flaw separates the two keys, `resolve₂` touches the
    stored identity zero times — the depth-≤2 coordinate read is the whole lookup. -/
theorem resolve₂_no_touch_of_separated [DecidableEq K] [DecidableEq C] {f₁ f₂ : K → C}
    {stored q : K} (h : f₁ stored ≠ f₁ q ∨ f₂ stored ≠ f₂ q) :
    (resolve₂ f₁ f₂ stored q).2 = 0 := by
  unfold resolve₂
  split_ifs with h1 h2
  · rfl
  · rfl
  · tauto

/-- **The confirm fires exactly on the double collision.** One stored-identity access iff both
    flaws collide — the exactly-`2^{-2b}` mass, nowhere else. -/
theorem resolve₂_touch_iff_double_collision [DecidableEq K] [DecidableEq C] (f₁ f₂ : K → C)
    (stored q : K) :
    (resolve₂ f₁ f₂ stored q).2 = 1 ↔ (f₁ stored = f₁ q ∧ f₂ stored = f₂ q) := by
  unfold resolve₂
  split_ifs with h1 h2 <;> simp_all

open Finset RadixDescent ProbabilisticBounds in
/-- **The spinor depth-two readout, discharged for the concrete fold (headline).** Flaw one =
    `homeSlot b` (the top `b`-bit window of `c·PHI`), flaw two = `digit1 b` (the window beneath
    it). Then, with no undischarged hypothesis:

    1. **exact for all keys** — `resolve₂`'s verdict is the exact membership test
       (`resolve₂_correct`);
    2. **the confirm fires exactly on the wide-window collision** — one stored-identity access iff
       the top `2b` bits collide (`resolve₂_touch_iff_double_collision` +
       `ProbabilisticBounds.pair_windows_eq_wide_window`);
    3. **that mass is exactly counted** — every `(flaw₁, flaw₂)` fiber has exactly `2^(W−2b)`
       coordinates (`ProbabilisticBounds.top_pair_fiber_card`); per i.i.d. pair the double
       collision has probability exactly `2^{-2b}`
       (`ProbabilisticBounds.per_pair_double_collision_prob`).

    So the lookup resolves in **two coordinate reads** on all but an exactly-`2^{-2b}` mass, and
    is exact there too at one confirm. This is `spinor_depth_two` with the hypothesis replaced by
    counting — the two flaws are the first two steps of the proven radix descent
    (`RadixDescent.chain_separates_within_radix`), not a new mechanism. The two flaws must be two
    windows of ONE unit product: two different multipliers are NOT jointly balanced
    (the `#guard` counterexample in `ProbabilisticBounds`). -/
theorem concrete_spinor_depth_two {W b : ℕ} (h2b : 2 * b ≤ W) (PHI : (ZMod (2 ^ W))ˣ) :
    (∀ stored q : ZMod (2 ^ W),
        (resolve₂ (homeSlot b PHI) (digit1 b PHI) stored q).1 = decide (stored = q))
    ∧ (∀ stored q : ZMod (2 ^ W),
        (resolve₂ (homeSlot b PHI) (digit1 b PHI) stored q).2 = 1
          ↔ homeSlot (2 * b) PHI stored = homeSlot (2 * b) PHI q)
    ∧ (∀ {s₁ s₂ : ℕ}, s₁ < 2 ^ b → s₂ < 2 ^ b →
        (univ.filter (fun c : ZMod (2 ^ W) =>
            homeSlot b PHI c = s₁ ∧ digit1 b PHI c = s₂)).card = 2 ^ (W - 2 * b)) :=
  ⟨fun stored q => resolve₂_correct _ _ stored q,
   fun stored q => (resolve₂_touch_iff_double_collision _ _ stored q).trans
     (pair_windows_eq_wide_window h2b PHI stored q),
   fun hs₁ hs₂ => top_pair_fiber_card h2b PHI hs₁ hs₂⟩

open Finset RadixDescent ProbabilisticBounds in
/-- **The two-band OR-gather recall (the displaced-key fallback, proven).** Activate `k` for `q`
    when EITHER window matches. Then:
    1. a key past its home window is still gathered by the second — a miss requires BOTH windows
       to differ (De Morgan; one band alone is `one_flaw_blind`);
    2. no third face hides in "both": joint agreement IS the wide `2b`-window
       (`ProbabilisticBounds.pair_windows_eq_wide_window`);
    3. for distinct i.i.d. keys the both-collide mass is exactly `2^{-2b}`
       (`ProbabilisticBounds.per_pair_double_collision_prob`). -/
theorem two_band_recall {W b n : ℕ} (h2b : 2 * b ≤ W) (PHI : (ZMod (2 ^ W))ˣ)
    {i j : Fin n} (hij : i ≠ j) (hn : 2 ≤ n) :
    (∀ c c' : ZMod (2 ^ W),
        ¬ (homeSlot b PHI c = homeSlot b PHI c' ∨ digit1 b PHI c = digit1 b PHI c')
          ↔ (homeSlot b PHI c ≠ homeSlot b PHI c' ∧ digit1 b PHI c ≠ digit1 b PHI c'))
    ∧ (∀ c c' : ZMod (2 ^ W),
        (homeSlot b PHI c = homeSlot b PHI c' ∧ digit1 b PHI c = digit1 b PHI c')
          ↔ homeSlot (2 * b) PHI c = homeSlot (2 * b) PHI c')
    ∧ ((univ.filter (fun κ : Keys n W =>
          homeSlot b PHI (κ i) = homeSlot b PHI (κ j)
            ∧ digit1 b PHI (κ i) = digit1 b PHI (κ j))).card : ℚ)
        / (Fintype.card (Keys n W) : ℚ) = 1 / 2 ^ (2 * b) :=
  ⟨fun _ _ => not_or,
   fun c c' => pair_windows_eq_wide_window h2b PHI c c',
   per_pair_double_collision_prob h2b PHI hij hn⟩

end SpinorReadout

-- ═══════════════════════════════════════════════════════════════════
-- namespace SeedSizing — the seed is the low `W−b` bits of the fold, and it is lossless
-- ═══════════════════════════════════════════════════════════════════
-- @scientist `homeSlot`/`residual` are the high/low split of the invertible fold `c·PHI` at bit
-- `W−b`. The slot (top `b` bits) is the address — free. The residual (low `W−b` bits) is the
-- seed. `decode_homeSlot_residual` proves `decode (slot) (seed) = c`: the split is lossless
-- because `PHI` is a unit, so the fold is a bijection and the two halves reassemble it. This is
-- `CoordInv.from_coord` as a theorem, on the real `homeSlot`/`PHI` fold.

namespace SeedSizing

open RadixDescent

variable {W : ℕ}

/-- **The seed / residual.** The low `W−b` bits of the fold `c·PHI` — everything the slot index
    (`homeSlot`, the top `b` bits) does not already pin down. This is what a home-resident key
    actually stores; the slot is its address. -/
def residual (b : ℕ) (PHI : (ZMod (2 ^ W))ˣ) (c : ZMod (2 ^ W)) : ℕ :=
  (c * (PHI : ZMod (2 ^ W))).val % 2 ^ (W - b)

/-- **The seed fits in `W−b` bits** — the stored width for a home-resident key. The `b` bits of
    the slot index are free (the address), so a key costs `W−b` bits, saving `b = log₂ m`. -/
theorem residual_lt {b : ℕ} (PHI : (ZMod (2 ^ W))ˣ) (c : ZMod (2 ^ W)) :
    residual b PHI c < 2 ^ (W - b) :=
  Nat.mod_lt _ (by positivity)

/-- **Decode a `(slot, seed)` pair back to a coordinate.** Reassemble the fold value
    `s·2^(W−b) + r` and undo the (unit) fold with `PHI⁻¹`. This is `CoordInv.from_coord`: the
    slot supplies the top `b` bits, the seed the low `W−b`, and `PHI⁻¹` turns the fold back into
    the coordinate. -/
def decode (b : ℕ) (PHI : (ZMod (2 ^ W))ˣ) (s r : ℕ) : ZMod (2 ^ W) :=
  ((s * 2 ^ (W - b) + r : ℕ) : ZMod (2 ^ W)) * (↑PHI⁻¹ : ZMod (2 ^ W))

/-- **The seed is lossless (headline): slot + seed regrow the coordinate exactly.**
    `decode (homeSlot c) (residual c) = c`. The `b`-bit slot index (the address) plus the
    `(W−b)`-bit residual reconstruct the full `W`-bit coordinate — because `homeSlot` and
    `residual` are the high/low split of the *invertible* fold `c·PHI` (`PHI` a unit, so the
    fold is a bijection and `Nat.div_add_mod'` reassembles its `val`). Storing only the residual
    (the slot is where the key lives) costs `W−b` bits per home-resident key, with no loss. This
    is `CoordInv` proved on the real fold. -/
theorem decode_homeSlot_residual {b : ℕ} (PHI : (ZMod (2 ^ W))ˣ) (c : ZMod (2 ^ W)) :
    decode b PHI (homeSlot b PHI c) (residual b PHI c) = c := by
  haveI : NeZero (2 ^ W) := ⟨by positivity⟩
  unfold decode homeSlot residual
  rw [Nat.div_add_mod', ZMod.natCast_val, ZMod.cast_id, mul_assoc, ← Units.val_mul,
      mul_inv_cancel, Units.val_one, mul_one]

/-- **Seed sizing, bundled.** For every coordinate `c`: the seed is `< 2^(W−b)` (fits in `W−b`
    bits) and `decode` regrows `c` from `(slot, seed)`. So a home-resident key — one whose slot
    *is* its `homeSlot`, the degree-1 / Being case (`SpinorFold.one_half_decides`) — is stored
    losslessly in `W−b` bits. The average-case count of such keys is the home-hit envelope
    `ProbabilisticBounds.expected_occupied_slots` = `2^b·(1 − (1−1/2^b)^n)`: at load `α = n/2^b`
    the home-hit fraction is `≤ (1 − (1−1/2^b)^n)/α`, and only the (envelope-bounded) displaced
    remainder falls back to the full coordinate. The sizing law `seedfold` needed. -/
theorem seed_width {b : ℕ} (PHI : (ZMod (2 ^ W))ˣ) (c : ZMod (2 ^ W)) :
    residual b PHI c < 2 ^ (W - b)
      ∧ decode b PHI (homeSlot b PHI c) (residual b PHI c) = c :=
  ⟨residual_lt PHI c, decode_homeSlot_residual PHI c⟩

end SeedSizing

end Sanguine
