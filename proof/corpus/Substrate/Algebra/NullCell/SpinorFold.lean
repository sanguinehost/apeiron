import Substrate.Algebra.NullCell.NullCell

/-!
# The Spinor Fold — the null basis diagonalizes the cell, and that is the whole 2×
@name Two Wires Instead of Four Multiplies
@child The cell's one number that matters is a difference of two squares. Switch to writing a pair as its sum and its difference, and that number becomes a plain product of two halves that never talk to each other again. So a reduction that looked like one tangled calculation is really two running products side by side: half the multiplies, one pass, and the order you do them in cannot change the answer.
@scientist The null basis `(u, v) = (z₁+z₂, z₁−z₂)` — the two boost halves of `NullCell.cell_is_two_boost_halves` — diagonalizes the cell multiply: each half is a homomorphism (`upper_nmul`, `lower_nmul`), so the vector-basis multiply's FOUR real products become `nmulSpinor`'s TWO (`nmul_is_diagonal`). The degree-2 observable is the product of two degree-1 halves (`norm_is_product_of_halves`), so a reduce is two abelian scalar channels closed by one multiply (`norm_of_reduction`) and schedule-freedom is DERIVED from the diagonalization, not assumed (`spinorFold_schedule_free`). The 2 is read off the two DEFINITIONS — `nmul` is written with four real products, `nmulSpinor` with two, and `nmul_is_diagonal` equates the maps that licenses the substitution; no theorem here counts a multiply, and no `Spin(2)`, `SO(2)` or covering degree occurs in this file or in its one import. The Being/Null split is the whole read/write story: off the cone a decryptor exists and ONE half decides identity (`one_half_decides`), on the cone none does (`NullCell.null_not_invertible`) and both halves must be carried (`opaque_needs_lower`, `fold_collision_on_cone`). And it is not two halves but three faces — the monad is the third (`monad_is_the_third`, `nulls_are_the_two_single_halves`).
@engineer `upper` / `lower` / `ofHalves` — the change of basis, inverted by `ofHalves_halves`. `spinorFold`: the one-pass two-wire reduce, equal to the two channel products (`spinorFold_eq`) and unchanged under any reordering (`spinorFold_schedule_free`) — no barrier. `norm_of_reduction`: `N(∏ zᵢ) = (∏ uᵢ)·(∏ vᵢ)` via `upper_foldr` / `lower_foldr` — one closing multiply, never a per-element square. `rotor_folds_into_weight` (from `nmul_assoc`): pre-compose the rotor into the weight, so the runtime FWHT costs nothing; `rotor_isometry` keeps the observable exact at unit norm, unlike skipping it. `one_half_decides`: given a decryptor, agreement on the `upper` channel alone decides identity — the stored key is redundant (hashtrinity's `get`). `coord_home_collisionFree` / `opaque_home_collides`: the coordinate home never collides at ANY slot count, the truncating fold does — so the low-load slack is dead footprint for coordinate keys. `opaque_needs_lower` / `fold_collision_on_cone`: with no decryptor both halves are carried, and a fold collision IS a null-cone point.

Built off `NullCell.lean` ALONE (its `nmul`, `nNorm`, `nOne`, and `nNorm_mul`). No other corpus.

The Null cell is `ℝ²` with one number that matters, the norm `N(z) = z₁² − z₂²`. There are two
bases on the same state:

* the **vector** basis `(z₁, z₂)` — where `N` is a *difference of two squares* and the multiply
  `nmul` MIXES both components (four real products);
* the **null / spinor** basis `(u, v) = (z₁+z₂, z₁−z₂)` — the two boost halves
  (`cell_is_two_boost_halves`), where `N = u·v` (one product) and the multiply is **diagonal**:
  each half is an independent scalar channel (`upper_nmul`, `lower_nmul`).

That diagonalization is the entire content. A degree-2 observable (`N`, a softmax denominator, a
dot product, an identity test) is the PRODUCT of two degree-1 halves. Form the bilinear in the
vector basis and you pay for both halves — four multiplies, two passes, a materialized square, a
barrier. Carry the halves in the null basis and it is two independent scalar folds — one pass, no
square, schedule-free. The 2 is a count of the two definitions — four real products in `nmul`,
two in `nmulSpinor`, the same map by `nmul_is_diagonal`. That this 2 IS the degree of the double
cover `Spin(2) → SO(2)`, a *vector = spinor ⊗ spinor*, hence a topological invariant carrying
unchanged to a CPU, a GPU and a hashmap, is asserted here and not proved: no covering degree
appears in this file or in its one import.

The two existence proofs, at the exact line:

* **hashbrown** `src/control/group/sse2.rs:73` `match_tag` —
  `_mm_movemask_epi8(_mm_cmpeq_epi8(ctrl, set1(tag)))`. It matches 16 control bytes (the *amplitude*
  / one half) in one SIMD op and touches a key (the second half) only on a hit. Single-sided.
* **llama.cpp** `ggml/src/ggml-cpu/ops.cpp:8490-8535` (online softmax) — carries `M` (running max =
  the boost rapidity) and `S` (running sum = the amplitude): `Mold=M; ms=expf(Mold-M); S = S*ms+vs`.
  `S = S*ms + vs` IS the boost `e^{jΔφ}` acting on the two null coordinates. One streaming pass, no
  materialized `exp` surface, no barrier.

This file proves the structure both embody, from the cell up. `hashtrinity`'s coordinate readout is
`one_half_decides`; the FWHT/rotor cost is removed by `rotor_folds_into_weight`.

## What is proved here, and what is read onto hardware

* **Proved (native):** the diagonalization and everything standing on it. `upper_nmul` and `lower_nmul`
  (each half is a multiplicative homomorphism, so the two channels never talk); `nmul_is_diagonal` (the
  cell multiply in the null basis IS the componentwise `nmulSpinor`); `norm_is_product_of_halves`
  (`nNorm z = upper z * lower z`); `spinorFold_eq`, `upper_foldr`, `lower_foldr` and
  `norm_of_reduction`; `spinorFold_schedule_free`, DERIVED from the diagonalization by permutation of a
  commutative product rather than assumed; `nmul_assoc` and `rotor_folds_into_weight`; `rotor_isometry`
  (from `NullCell.nNorm_mul`); `one_half_decides` and its dual `opaque_needs_lower`;
  `fold_collision_on_cone`; `coord_home_collisionFree` and `opaque_home_collides`;
  `monad_is_the_third` and `nulls_are_the_two_single_halves`.
* **Next:** the operation count. `nmul` is *written* with four real products and
  `nmulSpinor` with two, and no theorem here counts a multiply. `nmul_is_diagonal` states the equality
  of the two maps, which is what licenses the substitution; the 2 is read off the two definitions.
* **Next:** that the 2 IS the degree of the double cover `Spin(2) → SO(2)`, hence a
  topological invariant. Neither `Spin(2)`, nor `SO(2)`, nor a covering degree occurs in this file or
  in its one import; the identification of the null basis with a spinor basis is unformalised here.
* **Anchor:** "the same 2× on a CPU, a GPU and a hashmap". Nothing here is measured on
  any of the three, and the invariance argument that would carry it is the bullet above. The two
  existence proofs are external source read as instances of the structure: hashbrown
  `src/control/group/sse2.rs:73` `match_tag`, single-sided on the control byte, and llama.cpp
  `ggml/src/ggml-cpu/ops.cpp:8490-8535`, whose online softmax carries `M` and `S` as
  `S = S*ms + vs`. Quoted by line; neither is run or proved here.
* **Anchor:** the hashtrinity reading — `upper` as the coordinate read, `lower` as the key
  that stopped being stored, `dec` as `from_coord`, `home` as the slot map, and the `bits_for` `+1`
  doubling as dead footprint for a coordinate key. `one_half_decides` and `coord_home_collisionFree` are
  the algebra; that they are `get` and a real table's probe is the reading.
-/

namespace Sanguine
namespace Spinor

open Sanguine.CrossUniverse

noncomputable section

/-- The `+1` boost half (light-cone `u`): scales by `e^{+φ}` under a boost. `cell_is_two_boost_halves`. -/
def upper (z : NCell) : ℝ := z.1 + z.2
/-- The `−1` boost half (light-cone `v`): scales by `e^{-φ}`. The mirror down the cone. -/
def lower (z : NCell) : ℝ := z.1 - z.2

/-- Reconstruct a cell from its two halves — the inverse change of basis. -/
def ofHalves (u v : ℝ) : NCell := ((u + v) / 2, (u - v) / 2)

@[simp] theorem upper_ofHalves (u v : ℝ) : upper (ofHalves u v) = u := by
  simp only [upper, ofHalves]; ring
@[simp] theorem lower_ofHalves (u v : ℝ) : lower (ofHalves u v) = v := by
  simp only [lower, ofHalves]; ring
@[simp] theorem ofHalves_halves (z : NCell) : ofHalves (upper z) (lower z) = z := by
  simp only [ofHalves, upper, lower, Prod.ext_iff]; constructor <;> ring

/- ── THE DIAGONALIZATION: the null basis turns the cell multiply into two scalar channels ── -/

/-- **Upper half is a homomorphism.** `u(z·w) = u(z)·u(w)` — the `+1` channel is independent. -/
theorem upper_nmul (z w : NCell) : upper (nmul z w) = upper z * upper w := by
  simp only [upper, nmul]; ring
/-- **Lower half is a homomorphism.** `v(z·w) = v(z)·v(w)` — the `−1` channel is independent. -/
theorem lower_nmul (z w : NCell) : lower (nmul z w) = lower z * lower w := by
  simp only [lower, nmul]; ring

/-- **The one number is the product of the two halves.** `N(z) = u·v`. A degree-2 invariant as the
    product of two degree-1 coordinates — NOT a difference of two squares to be formed. -/
theorem norm_is_product_of_halves (z : NCell) : nNorm z = upper z * lower z := by
  simp only [nNorm, upper, lower]; ring

/-- The spinor (null-basis) multiply: TWO scalar products. -/
def nmulSpinor (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 * b.1, a.2 * b.2)

/-- **The diagonalization; the operation count is read off the two definitions.** `nmul` in the
    vector basis mixes both components — FOUR real products written out (`z₁w₁, z₂w₂, z₁w₂, z₂w₁`).
    In the null basis it is `nmulSpinor` — TWO (`u₁u₂, v₁v₂`). What is PROVED is that they are the
    same map, which is what licenses the substitution; the count is definitional, and naming that 2
    the degree of `Spin(2) → SO(2)` is prose. -/
theorem nmul_is_diagonal (z w : NCell) :
    (upper (nmul z w), lower (nmul z w)) = nmulSpinor (upper z, lower z) (upper w, lower w) := by
  simp only [nmulSpinor, upper_nmul, lower_nmul]

/- ── THE SPINOR FOLD: the reduction as two independent scalar channels ── -/

/-- The two channels of a reduction: the product of the upper halves, the product of the lowers. -/
def upperChan (zs : List NCell) : ℝ := (zs.map upper).prod
def lowerChan (zs : List NCell) : ℝ := (zs.map lower).prod

/-- The spinor fold: reduce a stream of cells in the null basis — one pass, two scalar wires,
    accumulator `(u_acc, v_acc)`. No per-element square; no materialized bilinear. -/
def spinorFold : List NCell → (ℝ × ℝ)
  | [] => (1, 1)
  | z :: zs => let p := spinorFold zs; (upper z * p.1, lower z * p.2)

/-- The fold is exactly the two channel-products — a single two-wire pass. -/
theorem spinorFold_eq (zs : List NCell) : spinorFold zs = (upperChan zs, lowerChan zs) := by
  induction zs with
  | nil => rfl
  | cons z zs ih =>
    simp only [spinorFold, ih, upperChan, lowerChan, List.map_cons, List.prod_cons]

/-- **SCHEDULE-FREE, PROVEN FROM THE DIAGONALIZATION.** Reorder the stream any way: the fold is
    unchanged, because each channel is a commutative scalar product. A diagonal (abelian) fold has
    no schedule — hence no barrier. This is `preduce_schedule_free` derived from the null basis, not
    assumed. -/
theorem spinorFold_schedule_free {zs ws : List NCell} (h : zs.Perm ws) :
    spinorFold zs = spinorFold ws := by
  rw [spinorFold_eq, spinorFold_eq]
  have hu : upperChan zs = upperChan ws := (h.map upper).prod_eq
  have hl : lowerChan zs = lowerChan ws := (h.map lower).prod_eq
  rw [hu, hl]

/-- `upper` is a monoid homomorphism through the whole fold: `u(∏ zᵢ) = ∏ u(zᵢ)`. -/
theorem upper_foldr (zs : List NCell) : upper (zs.foldr nmul nOne) = upperChan zs := by
  induction zs with
  | nil => simp [upperChan, upper, nOne]
  | cons z zs ih => simp [List.foldr_cons, upper_nmul, ih, upperChan, List.map_cons, List.prod_cons]

/-- `lower` is a monoid homomorphism through the whole fold: `v(∏ zᵢ) = ∏ v(zᵢ)`. -/
theorem lower_foldr (zs : List NCell) : lower (zs.foldr nmul nOne) = lowerChan zs := by
  induction zs with
  | nil => simp [lowerChan, lower, nOne]
  | cons z zs ih => simp [List.foldr_cons, lower_nmul, ih, lowerChan, List.map_cons, List.prod_cons]

/-- **The reduction's norm is the two channels, combined by ONE final multiply.**
    `N(∏ zᵢ) = (∏ uᵢ)·(∏ vᵢ)`. The whole reduce is two scalar product-streams and a single closing
    multiply — never a per-element square, never a barrier. Just the product-of-halves through the
    fold (`upper_foldr`/`lower_foldr`). -/
theorem norm_of_reduction (zs : List NCell) :
    nNorm (zs.foldr nmul nOne) = upperChan zs * lowerChan zs := by
  rw [norm_is_product_of_halves, upper_foldr, lower_foldr]

/- ── THE ROTOR FOLDS INTO THE WEIGHT: associativity of the cell multiply ── -/

/-- The cell multiply is associative (it is a commutative ring). -/
theorem nmul_assoc (a b c : NCell) : nmul (nmul a b) c = nmul a (nmul b c) := by
  simp only [nmul, Prod.ext_iff]; constructor <;> ring

/-- **THE ROTOR FOLDS INTO THE WEIGHT.** A rotor `R` applied to the activation `x`, then the weight
    `w`, equals the pre-composed weight `(w·R)` applied to `x`: `W(Rx) = (WR)x`. So the runtime rotor
    (the FWHT) costs nothing — it lives in the seal, computed once, offline. Pure associativity. -/
theorem rotor_folds_into_weight (w R x : NCell) :
    nmul w (nmul R x) = nmul (nmul w R) x := (nmul_assoc w R x).symm

/-- **The rotor is a coherence-preserving fold, not a lossy skip.** A unit-norm rotor (`N(R) = 1`)
    leaves the observable `N` invariant: `N(Rx) = N(x)`. So folding it into the weight keeps the
    answer exact — unlike "skip the rotor" (the CPU twin of NOBAR), which changes `N`. Fast AND
    coherent, because an isometry commutes with the observable. -/
theorem rotor_isometry (R x : NCell) (hR : nNorm R = 1) : nNorm (nmul R x) = nNorm x := by
  rw [nNorm_mul, hR, one_mul]

/- ── THE HASHTRINITY READOUT: identity from ONE half ── -/

/-- **The coordinate readout is one half.** When the second half is *determined* by the first —
    `lower = dec ∘ upper`, an injective coordinate with a decryptor (`CoordInv` /
    `EncryptionProjection`) — agreement on the `upper` channel ALONE decides identity. The stored key
    (the `lower` channel) is redundant: degree-1, not degree-2.

    This IS `hashtrinity`'s `get`: `upper` = the coordinate we read, `lower` = the key we stopped
    storing (the `MaybeUninit`/no-key-compare win), `dec` = `from_coord`. The spinor is the single
    coordinate channel; the vector was coordinate + key. -/
theorem one_half_decides {z w : NCell} (dec : ℝ → ℝ)
    (hz : lower z = dec (upper z)) (hw : lower w = dec (upper w))
    (h : upper z = upper w) : z = w := by
  have hl : lower z = lower w := by rw [hz, hw, h]
  calc z = ofHalves (upper z) (lower z) := (ofHalves_halves z).symm
    _ = ofHalves (upper w) (lower w) := by rw [h, hl]
    _ = w := ofHalves_halves w

/- ── THE WRITE DUAL: the probe is an affine loop; for a coordinate key it never runs ──
   `one_half_decides` is the READ: don't *store* the redundant half. This is the WRITE/scan side:
   don't *provision* for a probe loop that has already affine-collapsed. The probe walks
   `home → descend → …` ONLY on a collision. A coordinate key's `home c = c` is the affine identity
   (`AffineLoopCollapse`: the recurrence's fixed point reached at step 0 — "the loop that never
   runs"), injective, so distinct keys never share a home. The low-load slack (hashtrinity's
   `bits_for` `+1` doubling) is footprint bought against a collision that, for a coordinate key,
   cannot occur — insurance on a loop with zero iterations. Crucially, `collides` takes NO array-size
   argument: collision-freedom is a property of `home` alone, hence independent of the load. -/

/-- Two keys collide: distinct coordinates land in the same home slot. The only thing the probe loop
    resolves, and the only thing lowering the load shortens. -/
def collides (home : ℕ → ℕ) (a b : ℕ) : Prop := a ≠ b ∧ home a = home b

/-- **Coordinate home is collision-free — at every size.** `home c = c` (the affine identity) is
    injective, so distinct coordinate keys never share a home: the probe is one read, zero descents,
    for ANY slot count. No load reduction can shorten a probe already at its floor — the `+1`
    doubling is dead footprint for coordinate keys. This is `AffineLoopCollapse` on the write side. -/
theorem coord_home_collisionFree (a b : ℕ) : ¬ collides id a b := by
  rintro ⟨hne, heq⟩; exact hne heq

/-- **Opaque home can collide.** The truncating fold (`p >>> s`, the non-affine lossy Null species)
    is not injective: distinct coordinates share a home, so the probe loop genuinely runs and the
    low-load slack is justified insurance. The Being/Null split, made precise: affine (loop-collapsed,
    coordinate) vs non-affine (loop-runs, opaque). -/
theorem opaque_home_collides : ∃ a b, collides (· >>> 1) a b :=
  ⟨0, 1, by decide, by decide⟩

/- ── THE OPAQUE / NULL-CONE CASE: no decryptor, so BOTH halves must be carried ──
   `one_half_decides` needed a decryptor `lower = dec ∘ upper` — the coordinate (Being) key, OFF the
   cone (`N ≠ 0`, `/` total). The OPAQUE key is the cone itself: its byte-fold is lossy (non-injective
   by pigeonhole — more keys than coordinates), so it lands where `N = 0` and `/` is singular
   (`null_not_invertible`) — NO decryptor exists. Neither half is recoverable from the other, so both
   are carried: the VECTOR basis, degree 2. The two opaque costs ARE the two halves — `upper` computed
   on READ (the fold; a coordinate key's `upper` IS the key, free) and `lower` kept on WRITE (the
   stored key; a coordinate key regenerates it). -/

/-- **The dual of `one_half_decides`.** If two distinct states share their `upper` half — a fold
    collision, the non-injective/opaque case — their `lower` halves MUST differ: `upper` alone cannot
    tell them apart, so `lower` (the stored key) is not redundant and must be carried. Coordinate key
    ⇒ `dec` exists ⇒ one half; opaque key ⇒ no `dec` ⇒ both halves. -/
theorem opaque_needs_lower {z w : NCell} (hu : upper z = upper w) (hne : z ≠ w) :
    lower z ≠ lower w := by
  intro hl
  apply hne
  calc z = ofHalves (upper z) (lower z) := (ofHalves_halves z).symm
    _ = ofHalves (upper w) (lower w) := by rw [hu, hl]
    _ = w := ofHalves_halves w

/-- **A fold collision IS a null-cone point.** Two keys the fold sends to the same `upper` differ by a
    vector with `upper = 0` — a point on the null cone (`N = upper·lower = 0`), the ray where the norm
    vanishes and `/` dies. The opaque fold collapses distinct keys onto one cone ray; that degeneracy
    is exactly the non-invertibility (`null_not_invertible`) that forces the stored key. -/
theorem fold_collision_on_cone {z w : NCell} (hu : upper z = upper w) :
    nNorm ((z.1 - w.1, z.2 - w.2) : NCell) = 0 := by
  have hup : upper ((z.1 - w.1, z.2 - w.2) : NCell) = 0 := by
    simp only [upper] at hu ⊢; linarith
  rw [norm_is_product_of_halves, hup, zero_mul]

/- ── NOT TWO HALVES — THE TRINITY (the monad is the third) ──
   `cell_is_two_boost_halves` gives a 2D basis, but the operational structure is the trinity
   {NB, B, B/U} (`TrinityOperations`): the monad `nOne` is "on neither side" (the `NullCell` headline)
   — a THIRD face the two-halves reading omits. The opaque lookup pays across ALL THREE faces, not two
   halves:
     • **B (the monad = the address)** — computing `home = fold(k)`. The One, the direct hit. A
       coordinate key gets this free (`home(c)=c`, the monad IS the key) — the in-cache `get` gap.
     • **B/U (the ratio = the probe + the stored key)** — resolving a collision *by division* (descend)
       and the key kept to confirm identity on the cone (`opaque_needs_lower`). The build's key array.
     • **NB (the cone = absence)** — the empty slots, the low density insurance against the golden
       wave's aliasing. The build/iterate footprint. A coordinate key pays none of this (commensurate).
   Three faces, not two halves. -/

/-- **The monad is the third.** `nOne` is off the cone (`N = 1 ≠ 0`) and balanced (`upper = lower`) —
    neither pure null direction; the unit both cells share, on neither side. -/
theorem monad_is_the_third : nNorm nOne = 1 ∧ upper nOne = lower nOne := by
  refine ⟨?_, ?_⟩ <;> norm_num [nNorm, upper, lower, nOne]

/-- **The two null directions are the two single-half faces** — each on the cone, the other half zero:
    `nNull` is upper-only (NB), `nNull'` is lower-only (B/U). With the monad `nOne` (B), that is three
    — the trinity, not two halves. -/
theorem nulls_are_the_two_single_halves :
    (lower nNull = 0 ∧ upper nNull ≠ 0) ∧ (upper nNull' = 0 ∧ lower nNull' ≠ 0) := by
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩ <;> norm_num [upper, lower, nNull, nNull']

/- ── Witnesses (proved, since ℝ-equality is noncomputable) ── -/

-- the diagonalization on a concrete pair: nmul (2,1)(3,1) = (7,5), halves (u,v) = (12,2) = (3·4, 1·2).
example : (upper (nmul (2, 1) (3, 1)), lower (nmul (2, 1) (3, 1))) = ((3 : ℝ) * 4, (1 : ℝ) * 2) := by
  simp only [nmul, upper, lower, Prod.ext_iff]; constructor <;> norm_num
-- the norm is the product of halves: N((2,1)) = 4 − 1 = 3 = 3·1 = u·v.
example : nNorm ((2, 1) : NCell) = upper (2, 1) * lower (2, 1) := norm_is_product_of_halves _

end

end Spinor
end Sanguine
