import Apeiron.Combinator.Confluence
import Apeiron.Trichotomy.Derived.Gates
import Apeiron.Trichotomy.Implicative

/-!
# What a point of the algebra can say about a term

@name Invariant
@cite peirce-1870
@cite boole-1847
@cite palmieri-2026
@cite curry-1930
@cite artin-schreier-1927
@grade proved

`theory/DEAD_ENDS.md` leaves three shapes open, and the second is the algebra classifying
something about the calculus rather than carrying it. This file settles that shape exactly: it
says what every algebra-valued invariant of a term can see, exhibits the one the two channels
carry, and shows reduction moving the point up the channel order.

## Root

| from | here |
|---|---|
| `two_channels_iff_disc_pos`, `channel_iff_of_disc_pos` | `two_reduction_invariants_iff_disc_pos` |
| `upper_mul`, `lower_mul`, `fold_perm` | `mulInvariant_perm`, `mulInvariant_sees_only_leaves` |
| `upper_ofHalves`, `lower_ofHalves`, `halves_ext` | `freeCell_mulInvariant` |
| `upper_le_iff_exists_sq` through `Precedes` | `freeCell_precedes_of_reduces` |

The face is forced. `two_reduction_invariants_iff_disc_pos` says a point carrying two
independent reduction invariants, each read off by a channel of the algebra, exists exactly when
σ is above zero: at or below, `upper` and `lower` are the same read, so one point carries
one invariant rather than two. That is `two_channels_iff_disc_pos` with the calculus supplying
the two things to be read. Which point of that face is not fixed here, and does not need to be:
`positive_face_is_one_face` says every σ > 0 gives the same algebra with the same two
channels, so the invariant transports across the whole face.

## The exact strength, first

@cite palmieri-2026

`mulInvariant_sees_only_leaves` is the ceiling. Call a map from terms to points multiplicative
when it sends applying one term to another to multiplying their points. Every such map factors
through the multiset of the term's leaves: two terms with the same leaves in any order have the
same point, `mulInvariant_perm`. So a multiplicative invariant cannot see the shape of a term at
all, only its bag of constants and variables. `mulInvariant_agrees_on_swap` is the corollary at
the case `theory/DEAD_ENDS.md` opens with: every multiplicative map sends `S K` and `K S` to one
point, which is that entry's refutation stated for all such maps rather than for the identity.

The reason is `mul_comm` and `mul_assoc`, so the ceiling is a fact about commutative associative
multiplication rather than about this algebra. What the algebra adds is how much fits under the
ceiling, and the answer is two independent readings on the split face and one elsewhere.

## The invariant the channels carry

`dupFree` says the term holds no duplicating constant and `discardFree` says it holds no
discarding one. Each is preserved by rewriting: a term free of `S` has no duplicating redex to
fire and the discarding rule only deletes, while a term free of `K` has no discarding redex and
the duplicating rule copies what is already free of it. Neither is preserved by the other's
rule, so they are two separate facts about a term rather than one.

`freeCell` packs them into one point, one on each channel. `freeCell_mulInvariant` says the pack
is multiplicative, so `Gates.lean`'s pattern repeats: one multiplication, two wires.
`freeCell_precedes_of_reduces` says rewriting moves that point up the channel order, which is
the order `upper_le_iff_exists_sq` proves is the algebra's own. So reduction is sent to an
order, which is the first of the three open shapes, at the strength the ceiling allows.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy

/-- The leaves of a term, left to right.

    Comb → List Comb          [a term gives the finite sequence of its constants and variables]

    So: an application concatenates its two halves, and everything else is one leaf. -/
def leafList : Comb → List Comb
  | Comb.app f g => leafList f ++ leafList g
  | e => [e]

/-- A map from terms to points that turns applying into multiplying.

    (σ : ℝ)                                              [the parameter]
    (Φ : Comb → Cell)                                    [a map from terms to points]
    ∀ a b, Φ (Comb.app a b) = mul σ (Φ a) (Φ b)          [applying goes to multiplying]

    So: a homomorphism from the term algebra to the algebra. Nothing is asked of it beyond this
    one equation, and the theorems below say that one equation already fixes what it can see. -/
def MulInvariant (σ : ℝ) (Φ : Comb → Cell) : Prop :=
  ∀ a b : Comb, Φ (Comb.app a b) = mul σ (Φ a) (Φ b)

/-- The first channel of a multiplicative map is a product over the leaves.

    {σ : ℝ} (hσ : 0 ≤ σ)                                        [σ, at zero or above]
    {Φ : Comb → Cell} (hΦ : MulInvariant σ Φ)                   [a multiplicative map]
    (t : Comb)                                                  [any term]
    upper σ (Φ t)                                               [its first reading]
    = ((leafList t).map (fun e => upper σ (Φ e))).prod          [is one factor per leaf] -/
theorem mulInvariant_upper_prod {σ : ℝ} (hσ : 0 ≤ σ) {Φ : Comb → Cell}
    (hΦ : MulInvariant σ Φ) (t : Comb) :
    upper σ (Φ t) = ((leafList t).map (fun e => upper σ (Φ e))).prod := by
  induction t with
  | S => simp [leafList]
  | K => simp [leafList]
  | var i => simp [leafList]
  | app f g ihf ihg =>
    rw [hΦ, upper_mul hσ, ihf, ihg, leafList, List.map_append, List.prod_append]

/-- The second channel of a multiplicative map is a product over the leaves.

    {σ : ℝ} (hσ : 0 ≤ σ)                                        [σ, at zero or above]
    {Φ : Comb → Cell} (hΦ : MulInvariant σ Φ)                   [a multiplicative map]
    (t : Comb)                                                  [any term]
    lower σ (Φ t)                                               [its second reading]
    = ((leafList t).map (fun e => lower σ (Φ e))).prod          [is one factor per leaf] -/
theorem mulInvariant_lower_prod {σ : ℝ} (hσ : 0 ≤ σ) {Φ : Comb → Cell}
    (hΦ : MulInvariant σ Φ) (t : Comb) :
    lower σ (Φ t) = ((leafList t).map (fun e => lower σ (Φ e))).prod := by
  induction t with
  | S => simp [leafList]
  | K => simp [leafList]
  | var i => simp [leafList]
  | app f g ihf ihg =>
    rw [hΦ, lower_mul hσ, ihf, ihg, leafList, List.map_append, List.prod_append]

/-- Rearranging the leaves cannot change the point.

    {σ : ℝ} (hσ : 0 < σ)                                   [the parameter, above zero]
    {Φ : Comb → Cell} (hΦ : MulInvariant σ Φ)              [a multiplicative map]
    {t u : Comb}                                           [two terms]
    (h : (leafList t).Perm (leafList u))                   [same leaves, in some order]
    Φ t = Φ u                                              [they go to the same point] -/
theorem mulInvariant_perm {σ : ℝ} (hσ : 0 < σ) {Φ : Comb → Cell}
    (hΦ : MulInvariant σ Φ) {t u : Comb} (h : (leafList t).Perm (leafList u)) :
    Φ t = Φ u :=
  halves_ext hσ
    (by rw [mulInvariant_upper_prod hσ.le hΦ, mulInvariant_upper_prod hσ.le hΦ,
      (h.map _).prod_eq])
    (by rw [mulInvariant_lower_prod hσ.le hΦ, mulInvariant_lower_prod hσ.le hΦ,
      (h.map _).prod_eq])

/-- A multiplicative map sees the bag of leaves and no more.

@cite palmieri-2026

    {σ : ℝ} (hσ : 0 < σ)                                   [the parameter, above zero]
    {Φ : Comb → Cell} (hΦ : MulInvariant σ Φ)              [a multiplicative map]
    ∀ t u : Comb, (leafList t).Perm (leafList u) → Φ t = Φ u   [same leaves, same point]
    Φ (Comb.app Comb.S Comb.K) = Φ (Comb.app Comb.K Comb.S)    [so these two go together]
    ∧                                                          [both together]

    So: the ceiling on this whole route. A multiplicative map cannot see the shape of a term,
    because `mul_comm` and `mul_assoc` collapse the tree into a product over its leaves. The
    second half is `theory/DEAD_ENDS.md`'s opening refutation stated for every multiplicative
    map rather than for one candidate. -/
theorem mulInvariant_sees_only_leaves {σ : ℝ} (hσ : 0 < σ) {Φ : Comb → Cell}
    (hΦ : MulInvariant σ Φ) :
    (∀ t u : Comb, (leafList t).Perm (leafList u) → Φ t = Φ u) ∧
      Φ (Comb.app Comb.S Comb.K) = Φ (Comb.app Comb.K Comb.S) := by
  refine ⟨fun t u h => mulInvariant_perm hσ hΦ h, mulInvariant_perm hσ hΦ ?_⟩
  show ([Comb.S, Comb.K] : List Comb).Perm [Comb.K, Comb.S]
  exact List.Perm.swap Comb.K Comb.S []

-- ═══════════════════════════════════════════════════════════════════
-- TWO PROPERTIES REWRITING KEEPS
-- ═══════════════════════════════════════════════════════════════════

/-- Does the term hold no duplicating constant?

    Comb → Bool          [a term gives true exactly when no S occurs anywhere in it] -/
def dupFree : Comb → Bool
  | Comb.S => false
  | Comb.K => true
  | Comb.var _ => true
  | Comb.app f g => dupFree f && dupFree g

/-- Does the term hold no discarding constant?

    Comb → Bool          [a term gives true exactly when no K occurs anywhere in it] -/
def discardFree : Comb → Bool
  | Comb.S => true
  | Comb.K => false
  | Comb.var _ => true
  | Comb.app f g => discardFree f && discardFree g

/-- Rewriting keeps a term free of the duplicating constant.

    {t u : Comb}                        [two terms]
    (h : Comb.Step t u)                 [the first rewrites to the second in one step]
    dupFree t = true                    [the first holds no duplicating constant]
    →                                   [then]
    dupFree u = true                    [neither does the second]

    So: the duplicating rule cannot fire in such a term, and the discarding rule only deletes.
    -/
theorem dupFree_step {t u : Comb} (h : Comb.Step t u) :
    dupFree t = true → dupFree u = true := by
  induction h with
  | k x y => intro hx; simp [dupFree] at hx; simp [hx]
  | s f g x => intro hx; simp [dupFree] at hx
  | left b _ ih => intro hx; simp [dupFree] at hx ⊢; exact ⟨ih hx.1, hx.2⟩
  | right a _ ih => intro hx; simp [dupFree] at hx ⊢; exact ⟨hx.1, ih hx.2⟩

/-- Rewriting keeps a term free of the discarding constant.

    {t u : Comb}                        [two terms]
    (h : Comb.Step t u)                 [the first rewrites to the second in one step]
    discardFree t = true                [the first holds no discarding constant]
    →                                   [then]
    discardFree u = true                [neither does the second]

    So: the discarding rule cannot fire in such a term, and the duplicating rule copies a term
    that is already free of it. -/
theorem discardFree_step {t u : Comb} (h : Comb.Step t u) :
    discardFree t = true → discardFree u = true := by
  induction h with
  | k x y => intro hx; simp [discardFree] at hx
  | s f g x => intro hx; simp [discardFree] at hx ⊢; tauto
  | left b _ ih => intro hx; simp [discardFree] at hx ⊢; exact ⟨ih hx.1, hx.2⟩
  | right a _ ih => intro hx; simp [discardFree] at hx ⊢; exact ⟨hx.1, ih hx.2⟩

/-- Reaching keeps both properties.

    {t u : Comb}                                       [two terms]
    (h : Comb.Reduces t u)                             [the first reaches the second]
    dupFree t = true → dupFree u = true                [holding no S survives]
    discardFree t = true → discardFree u = true        [and so does holding no K]
    ∧                                                  [both together] -/
theorem free_reduces {t u : Comb} (h : Comb.Reduces t u) :
    (dupFree t = true → dupFree u = true) ∧
      (discardFree t = true → discardFree u = true) := by
  induction h with
  | refl => exact ⟨id, id⟩
  | tail _ hstep ih =>
    exact ⟨fun ht => dupFree_step hstep (ih.1 ht),
      fun ht => discardFree_step hstep (ih.2 ht)⟩

-- ═══════════════════════════════════════════════════════════════════
-- ONE POINT, TWO WIRES
-- ═══════════════════════════════════════════════════════════════════

/-- A term as a point: one property on each channel.

    (σ : ℝ)                                                       [the parameter]
    (t : Comb)                                                    [a term]
    ofHalves σ (boolVal (dupFree t)) (boolVal (discardFree t))    [one point, two wires]
    : Cell                                                        [a point of the algebra] -/
def freeCell (σ : ℝ) (t : Comb) : Cell :=
  ofHalves σ (boolVal (dupFree t)) (boolVal (discardFree t))

/-- The first channel reads freedom from the duplicating constant.

    {σ : ℝ} (hσ : 0 < σ)                                [the parameter, above zero]
    (t : Comb)                                          [a term]
    upper σ (freeCell σ t) = boolVal (dupFree t)        [the first reading is that property] -/
theorem upper_freeCell {σ : ℝ} (hσ : 0 < σ) (t : Comb) :
    upper σ (freeCell σ t) = boolVal (dupFree t) :=
  upper_ofHalves hσ _ _

/-- The second channel reads freedom from the discarding constant.

    {σ : ℝ} (hσ : 0 < σ)                                    [the parameter, above zero]
    (t : Comb)                                              [a term]
    lower σ (freeCell σ t) = boolVal (discardFree t)        [the second reading is the other] -/
theorem lower_freeCell {σ : ℝ} (hσ : 0 < σ) (t : Comb) :
    lower σ (freeCell σ t) = boolVal (discardFree t) :=
  lower_ofHalves hσ _ _

/-- The pack turns applying into multiplying.

@cite boole-1847

    {σ : ℝ} (hσ : 0 < σ)                    [the parameter, above zero]
    MulInvariant σ (freeCell σ)             [packing the two properties is multiplicative]

    So: `Gates.lean`'s pattern with terms in place of truth values. Both properties are AND
    over the two halves of an application, `boolVal_mul` says AND is multiplication of readings,
    and `upper_mul` says the two channels multiply separately. One multiplication, two wires. -/
theorem freeCell_mulInvariant {σ : ℝ} (hσ : 0 < σ) : MulInvariant σ (freeCell σ) := by
  intro a b
  refine halves_ext hσ ?_ ?_
  · rw [upper_freeCell hσ, upper_mul hσ.le, upper_freeCell hσ, upper_freeCell hσ, boolVal_mul]
    rfl
  · rw [lower_freeCell hσ, lower_mul hσ.le, lower_freeCell hσ, lower_freeCell hσ, boolVal_mul]
    rfl

/-- A truth value that only ever gains is a reading that only ever rises.

    {b c : Bool}                        [two truth values]
    (h : b = true → c = true)           [the first being true forces the second]
    boolVal b ≤ boolVal c               [so the first reads at or below the second] -/
theorem boolVal_mono {b c : Bool} (h : b = true → c = true) : boolVal b ≤ boolVal c := by
  cases b <;> cases c <;> simp_all [boolVal]

/-- Rewriting moves the point up the channel order.

@cite artin-schreier-1927

    {σ : ℝ} (hσ : 0 < σ)                                  [the parameter, above zero]
    {t u : Comb}                                          [two terms]
    (h : Comb.Reduces t u)                                [the first reaches the second]
    Precedes σ (freeCell σ t) (freeCell σ u)              [the first point sits below]

    So: reduction is sent to an order rather than to an operation, which is the first of the
    three shapes `theory/DEAD_ENDS.md` leaves open. The order is not supplied from outside:
    `precedes_iff_exists_sq` says each comparison is an existence statement about squares in the
    algebra itself. -/
theorem freeCell_precedes_of_reduces {σ : ℝ} (hσ : 0 < σ) {t u : Comb}
    (h : Comb.Reduces t u) : Precedes σ (freeCell σ t) (freeCell σ u) := by
  refine ⟨?_, ?_⟩
  · rw [upper_freeCell hσ, upper_freeCell hσ]
    exact boolVal_mono (free_reduces h).1
  · rw [lower_freeCell hσ, lower_freeCell hσ]
    exact boolVal_mono (free_reduces h).2

-- ═══════════════════════════════════════════════════════════════════
-- THE HEADLINE: TWO INVARIANTS IN ONE POINT NEEDS THE SPLIT FACE
-- ═══════════════════════════════════════════════════════════════════

/-- One point carries two independent reduction invariants exactly above zero.

    (σ : ℝ)                                                     [the parameter]
    ∃ Φ : Comb → Cell                                           [there is a map terms to points]
    ∃ φ ψ : Cell → ℝ                                            [and two reads of the algebra]
    Channel σ φ ∧ Channel σ ψ                                   [each respecting the operations]
    ∀ t, φ (Φ t) = boolVal (dupFree t)                          [one reading off freedom from S]
    ∀ t, ψ (Φ t) = boolVal (discardFree t)                      [the other freedom from K]
    ↔                                                           [exactly when]
    0 < σ                                                       [σ is above zero]

    So: the duplicating constant is not free of itself and is free of the discarding one, so the
    two reads disagree at that one term, and `two_channels_iff_disc_pos` says two disagreeing
    reads exist on the split face alone. At or below zero the two readings of every point
    coincide, so one point carries one invariant. This is the whole σ-dependence of the route:
    how many independent facts about a term a single point can hold. -/
theorem two_reduction_invariants_iff_disc_pos (σ : ℝ) :
    (∃ (Φ : Comb → Cell) (φ ψ : Cell → ℝ),
        Channel σ φ ∧ Channel σ ψ ∧
        (∀ t : Comb, φ (Φ t) = boolVal (dupFree t)) ∧
        (∀ t : Comb, ψ (Φ t) = boolVal (discardFree t))) ↔ 0 < σ := by
  constructor
  · rintro ⟨Φ, φ, ψ, hφ, hψ, hdup, hdis⟩
    refine (two_channels_iff_disc_pos σ).mp ⟨φ, ψ, hφ, hψ, fun h => ?_⟩
    have h1 : φ (Φ Comb.S) = 0 := by rw [hdup]; simp [dupFree, boolVal]
    have h2 : ψ (Φ Comb.S) = 1 := by rw [hdis]; simp [discardFree, boolVal]
    rw [h] at h1
    rw [h1] at h2
    norm_num at h2
  · intro hσ
    exact ⟨freeCell σ, upper σ, lower σ, channel_upper hσ.le, channel_lower hσ.le,
      upper_freeCell hσ, lower_freeCell hσ⟩

end Combinator
end Apeiron

end
