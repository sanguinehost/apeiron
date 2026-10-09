import Apeiron.Trichotomy.Basic

/-!
# Use-once resources

@name Use Once
@cite cockle-1849
@grade proved

A use-once resource is `live a` or `spent`. Consuming a live one delivers its value and leaves
it spent; consuming a spent one delivers the residue and leaves it spent. So reaching twice for
the same value does not fail a check, it finds nothing: `no_double_use`.

Aliasing is where that stops being decidable at the binding. Two keys naming one resource are
two doors on one room, and tracking "spent" on the key rather than on the room lets the second
door deliver what the first already took. `consumeLocal_double_delivery_under_aliasing` exhibits
it. Tracking on the identity closes it across every alias with nothing enumerated
(`consumeAt_second_alias_is_nothing`), and the dividing line is injectivity of the key map, not any
property of use-once itself (`unaliased_no_cross_talk`).

## Reuse, which closes on the identity too

`consumeAt_second_alias_is_nothing` is stated on the store one consume later, so nothing happens
in between. An allocator that hands a released slot back breaks that. `../song`'s
`src/kernel/mem/ring.slang` does exactly this: `ringRelease` pushes the carrier onto a free stack
and `ringReuse` pops it and returns it **unchanged**, same base and same size, so a stale carrier
is bit-identical to the fresh one. The slot goes live again and a key held from before finds the
next holder's value rather than a residue. `reuse_readmits_delivery` exhibits it, and it needs no
hypothesis about what the slot held before, because reclaiming overwrites.

`generation_closes_reuse` and `generation_closes_alias` close it: carry the generation on the key,
bump it on reclaim, and refuse a reach whose generation does not match. The check is one equality
on a counting number, so it is decidable and costs one comparison. Nothing is enumerated and no
analysis over programs is run, which is the same shape as tracking on the identity rather than the
key, one level up.

## What the algebra supplies, and what it does not

A residue has to be a point that is not the unit, that absorbs, and that the unit cannot be
recovered from. The algebra has one exactly on the non-negative face:
`use_once_needs_a_residue`. Below zero the size is a sum of two squares, so nothing but the
origin has size zero and there is nothing to fall onto.

**It does not say `no_double_use` is non-invertibility.** Those are facts about different
objects: one is about a two-constructor inductive, the other about `Cell σ`, and no map between
them is built here. The predecessor's `use_once_is_the_cone` conjoined four such facts and read
the conjunction as an identification, which is the shape `Emergence.lean` was deleted for. What
is true and is stated below is that the algebra supplies a residue exactly when `0 ≤ σ`, so a
carrier with a use-once discipline needs that face and cannot be built on the elliptic one.
-/

universe u

namespace Apeiron
namespace Resource

open Trichotomy

/-- A use-once resource: a value not yet spent, or the spent state.

    Linear α                 [a use-once resource carrying values of type α]
    live (a : α)             [holding an unspent value]
    spent                    [already consumed] -/
inductive Linear (α : Type u) where
  /-- Holding an unspent value. -/
  | live (a : α)
  /-- Already consumed. -/
  | spent

/-- What a consume hands back: the value, or the residue.

    Residue α                [what consuming delivers]
    delivered (a : α)        [the value, on its first and only delivery]
    nothing                  [the residue, on every later reach] -/
inductive Residue (α : Type u) where
  /-- The value, on its first and only delivery. -/
  | delivered (a : α)
  /-- The residue, on every later reach. -/
  | nothing
  deriving DecidableEq

/-- Consume a resource: deliver what is there, and leave it spent.

    {α : Type u}                       [a value type]
    Linear α → Residue α × Linear α    [takes a resource, gives what came out and what is left] -/
def consume {α : Type u} : Linear α → Residue α × Linear α
  | .live a => (Residue.delivered a, .spent)
  | .spent => (Residue.nothing, .spent)

/-- The first reach delivers the value and spends the resource.

    {α : Type u} (a : α)                                             [a value]
    consume (Linear.live a) = (Residue.delivered a, Linear.spent)    [delivered, and now spent] -/
theorem consume_live {α : Type u} (a : α) :
    consume (Linear.live a) = (Residue.delivered a, Linear.spent) := rfl

/-- Spent absorbs.

    {α : Type u}                                                      [a value type]
    consume (Linear.spent : Linear α)                   [consuming a spent one]
      = (Residue.nothing, Linear.spent)                [gives nothing, and stays spent] -/
theorem consume_spent {α : Type u} :
    consume (Linear.spent : Linear α) = (Residue.nothing, Linear.spent) := rfl

/-- The second reach finds the residue.

    {α : Type u} (a : α)                                       [a value]
    (consume (consume (Linear.live a)).2).1 = Residue.nothing  [reaching twice finds nothing] -/
theorem reuse_is_nothing {α : Type u} (a : α) :
    (consume (consume (Linear.live a)).2).1 = Residue.nothing := rfl

/-- A value cannot be delivered twice.

    {α : Type u} (a : α)                                             [a value]
    (consume (consume (Linear.live a)).2).1 ≠ Residue.delivered a    [the second reach is not it]

    So: use-once is not a rule that catches a violation, it is that the violation has nothing to
    return. -/
theorem no_double_use {α : Type u} (a : α) :
    (consume (consume (Linear.live a)).2).1 ≠ Residue.delivered a := by
  rw [reuse_is_nothing]
  intro h
  nomatch h

-- ═══════════════════════════════════════════════════════════════════
-- WHAT A RESIDUE HAS TO BE, AND WHERE THE ALGEBRA HAS ONE
-- ═══════════════════════════════════════════════════════════════════

/-- A point of size zero absorbs: every product with it has size zero too.

    (σ : ℝ) (n z : Cell)              [the parameter, a point of size zero, and any point]
    (h : normForm σ n = 0)                [the first has size zero]
    normForm σ (mul σ z n) = 0            [so does anything multiplied by it]

    So: once on the cone, always on the cone. The absorbing arm of a consume is that closure. -/
theorem spend_stays_on_cone (σ : ℝ) (n z : Cell) (h : normForm σ n = 0) :
    normForm σ (mul σ z n) = 0 := by
  rw [normForm_mul, h, mul_zero]

/-- The algebra supplies a residue exactly on the non-negative face.

    (σ : ℝ)                                              [the parameter]
    ∃ n : Cell, n ≠ 0 ∧ normForm σ n = 0                     [a point other than zero, of size zero]
      ∧ ¬ ∃ w : Cell, mul σ n w = one                    [which nothing inverts]
    ↔                                                    [exactly when]
    0 ≤ σ                                                [σ is at zero or above]

    So: a residue must be a point that is not the unit, absorbs (`spend_stays_on_cone`), and
    cannot be inverted back to the unit (`normForm_zero_not_invertible`). The algebra has one when
    σ is at zero or above and has none below, where the size is a sum of two squares.

    A use-once discipline carried by this algebra therefore needs the non-negative face. That is
    a statement about which `σ` supplies a residue. It is not a claim that `no_double_use` and
    non-invertibility are the same theorem: those are facts about different objects and no map
    between them is built here. -/
theorem use_once_needs_a_residue (σ : ℝ) :
    (∃ n : Cell, n ≠ 0 ∧ normForm σ n = 0 ∧ ¬ ∃ w : Cell, mul σ n w = one) ↔ 0 ≤ σ := by
  constructor
  · rintro ⟨n, hn0, hnn, _⟩
    exact (exists_normForm_zero_iff_disc_nonneg σ).mp ⟨n, hn0, hnn⟩
  · intro hσ
    obtain ⟨n, hn0, hnn⟩ := (exists_normForm_zero_iff_disc_nonneg σ).mpr hσ
    exact ⟨n, hn0, hnn, normForm_zero_not_invertible σ n hnn⟩

-- ═══════════════════════════════════════════════════════════════════
-- ALIASING: WHERE TRACKING ON THE KEY STOPS WORKING
-- ═══════════════════════════════════════════════════════════════════

/-- A resource's identity: which room, not which door. -/
abbrev Ident := Nat

/-- A name a program holds: which door, not which room. -/
abbrev Key := Nat

/-- What the store holds at each identity. -/
abbrev Store (α : Type u) := Ident → Linear α

/-- Which identity each key names. Aliasing is this failing to be injective. -/
abbrev KeyMap := Key → Ident

/-- Write one identity's slot.

    {α : Type u} (st : Store α)        [a store]
    (i : Ident) (v : Linear α)         [an identity and what to put there]
    : Store α                          [the store after] -/
def storeUpdate {α : Type u} (st : Store α) (i : Ident) (v : Linear α) : Store α :=
  fun j => if j = i then v else st j

/-- Consume through a key, marking the identity spent rather than the key.

    {α : Type u} (st : Store α)        [a store]
    (km : KeyMap) (k : Key)            [which identity each key names, and the key used]
    : Residue α × Store α              [what was delivered, and the store after] -/
def consumeAt {α : Type u} (st : Store α) (km : KeyMap) (k : Key) : Residue α × Store α :=
  let r := consume (st (km k))
  (r.1, storeUpdate st (km k) r.2)

/-- Consuming a live identity delivers its value and spends the identity.

    {α : Type u} (st : Store α) (km : KeyMap)      [a store and a key map]
    (k : Key) (a : α)                              [a key and a value]
    (h : st (km k) = Linear.live a)                               [the identity is live]
    (consumeAt st km k).1 = Residue.delivered a                   [the value comes out]
    (consumeAt st km k).2 (km k) = Linear.spent                   [and the identity is spent]
    ∧                                                             [both together] -/
theorem consumeAt_delivers {α : Type u} (st : Store α) (km : KeyMap) (k : Key) (a : α)
    (h : st (km k) = Linear.live a) :
    (consumeAt st km k).1 = Residue.delivered a ∧
      (consumeAt st km k).2 (km k) = Linear.spent := by
  constructor
  · show (consume (st (km k))).1 = Residue.delivered a
    rw [h]
    rfl
  · show storeUpdate st (km k) (consume (st (km k))).2 (km k) = Linear.spent
    simp only [storeUpdate, h]
    rfl

/-- Tracking on the identity closes every alias at once.

    {α : Type u} (st : Store α) (km : KeyMap)         [a store and a key map]
    (k₁ k₂ : Key) (a : α)                             [two keys and a value]
    (halias : km k₁ = km k₂)                          [the two keys name one identity]
    (hlive : st (km k₁) = Linear.live a)              [which is live]
    (consumeAt (consumeAt st km k₁).2 km k₂).1 = Residue.nothing   [the second finds nothing]

    So: no enumeration of aliases is needed, because the mark lives on the identity that every
    alias reads through. -/
theorem consumeAt_second_alias_is_nothing {α : Type u} (st : Store α) (km : KeyMap)
    (k₁ k₂ : Key) (a : α) (halias : km k₁ = km k₂) (hlive : st (km k₁) = Linear.live a) :
    (consumeAt (consumeAt st km k₁).2 km k₂).1 = Residue.nothing := by
  have hspent : (consumeAt st km k₁).2 (km k₂) = Linear.spent := by
    rw [← halias]
    exact (consumeAt_delivers st km k₁ a hlive).2
  show (consume ((consumeAt st km k₁).2 (km k₂))).1 = Residue.nothing
  rw [hspent]
  rfl

/-- Reading a key's own snapshot, which is what a per-binding check can decide.

    {α : Type u} (snapshot : Key → Linear α)    [what each key believes it holds]
    (k : Key)                                   [a key]
    : Residue α                                 [what consuming through it delivers] -/
def consumeLocal {α : Type u} (snapshot : Key → Linear α) (k : Key) : Residue α :=
  (consume (snapshot k)).1

/-- Tracking on the key double-delivers under aliasing.

    {α : Type u} (st : Store α) (km : KeyMap)              [a store and a key map]
    (k₁ k₂ : Key) (a : α)                                  [two keys and a value]
    (halias : km k₁ = km k₂)                               [naming one identity]
    (hlive : st (km k₁) = Linear.live a)                   [which is live]
    consumeLocal (st ∘ km) k₁ = Residue.delivered a        [the first key gets the value]
    consumeLocal (st ∘ km) k₂ = Residue.delivered a        [and so does the second]
    ∧                                                      [both together]

    So: `no_double_use` violated, exhibited rather than predicted. The mark lived on the key and
    the resource lived on the identity, and aliasing is those two coming apart. -/
theorem consumeLocal_double_delivery_under_aliasing {α : Type u}
    (st : Store α) (km : KeyMap) (k₁ k₂ : Key) (a : α)
    (halias : km k₁ = km k₂) (hlive : st (km k₁) = Linear.live a) :
    consumeLocal (st ∘ km) k₁ = Residue.delivered a ∧
      consumeLocal (st ∘ km) k₂ = Residue.delivered a := by
  have hlive2 : st (km k₂) = Linear.live a := by rw [← halias]; exact hlive
  constructor
  · show (consume (st (km k₁))).1 = Residue.delivered a
    rw [hlive]; rfl
  · show (consume (st (km k₂))).1 = Residue.delivered a
    rw [hlive2]; rfl

/-- Injectivity of the key map is the dividing line.

    {km : KeyMap} (hinj : Function.Injective km)     [distinct keys name distinct identities]
    {k₁ k₂ : Key} (hne : k₁ ≠ k₂)                    [two different keys]
    km k₁ ≠ km k₂                                    [name different identities]

    So: on the unaliased fragment there is nothing for a per-key scheme to get wrong, which is
    why a static per-binding check is adequate exactly there and nowhere wider. Which analysis
    could decide injectivity for a real program is not settled here. -/
theorem unaliased_no_cross_talk {km : KeyMap} (hinj : Function.Injective km)
    {k₁ k₂ : Key} (hne : k₁ ≠ k₂) : km k₁ ≠ km k₂ :=
  fun h => hne (hinj h)

-- ═══════════════════════════════════════════════════════════════════
-- REUSE: WHERE TRACKING ON THE IDENTITY STOPS WORKING TOO
-- ═══════════════════════════════════════════════════════════════════

/-- Put a fresh value back at an identity, so the slot is live again.

    {α : Type u} (st : Store α)        [a store]
    (i : Ident) (a : α)                [an identity and the value going back there]
    : Store α                          [the store after]

    So: an allocator handing a released slot to its next holder. `../song`'s `ringReuse` pops the
    released carrier and returns it unchanged, so this is that step. -/
def reclaimAt {α : Type u} (st : Store α) (i : Ident) (a : α) : Store α :=
  storeUpdate st i (Linear.live a)

/-- Reclaiming a slot hands the next holder's value to a key held from before.

    {α : Type u} (st : Store α) (km : KeyMap)                 [a store and a key map]
    (k₁ k₂ : Key) (b : α)                                     [two keys and the next value]
    (halias : km k₁ = km k₂)                                  [both name one identity]
    (consumeAt (reclaimAt (consumeAt st km k₁).2 (km k₁) b) km k₂).1
      = Residue.delivered b                                   [the stale key delivers it]

    So: tracking on the identity closes the alias only while the identity stays spent. Once the
    slot is reclaimed the guard is gone, and the second reach does not find a residue, it finds
    the next holder's live value. That is use after free into reallocated storage, which is the
    variant that matters, and `consumeAt_second_alias_is_nothing` cannot see it because that
    theorem is stated on the store one consume later with nothing in between.

    No hypothesis about what the slot held before is needed, and that is the point: reclaiming
    overwrites, so the failure does not depend on the earlier state at all. -/
theorem reuse_readmits_delivery {α : Type u} (st : Store α) (km : KeyMap)
    (k₁ k₂ : Key) (b : α) (halias : km k₁ = km k₂) :
    (consumeAt (reclaimAt (consumeAt st km k₁).2 (km k₁) b) km k₂).1
      = Residue.delivered b := by
  have h : reclaimAt (consumeAt st km k₁).2 (km k₁) b (km k₂) = Linear.live b := by
    rw [← halias]
    simp [reclaimAt, storeUpdate]
  show (consume (reclaimAt (consumeAt st km k₁).2 (km k₁) b (km k₂))).1 = Residue.delivered b
  rw [h]
  rfl

/-- A key carrying the generation it was minted at.

    GenKey                     [a key that knows which reclaim it belongs to]
    ident : Ident              [which identity it names]
    gen : Nat                  [which generation of that identity]

    So: `../song`'s carrier packs a base and a size and leaves its high bits unused, which is
    where a field like this would go. Whether to spend them is song's decision. -/
structure GenKey where
  /-- Which identity the key names. -/
  ident : Ident
  /-- Which generation of that identity the key was minted at. -/
  gen : Nat
  deriving DecidableEq

/-- A store that records which generation is current at each identity.

    GenStore (α : Type u)      [a store with a generation counter per identity]
    cell : Ident → Linear α    [what is held at each identity]
    gen : Ident → Nat          [which generation each identity is on]
    →                          [takes the left and gives the right] -/
structure GenStore (α : Type u) where
  /-- What is held at each identity. -/
  cell : Ident → Linear α
  /-- Which generation each identity is on. -/
  gen : Ident → Nat

/-- Consume through a generation-tagged key, refusing a reach whose generation is stale.

    {α : Type u} (st : GenStore α) (k : GenKey)   [a store and a tagged key]
    : Residue α × GenStore α                      [what was delivered, and the store after]

    So: one equality on a counting number decides it. A stale key takes the second arm and gets
    the residue without the slot being read, so nothing is enumerated and no analysis over
    programs is run. -/
def genConsume {α : Type u} (st : GenStore α) (k : GenKey) : Residue α × GenStore α :=
  if st.gen k.ident = k.gen then
    let r := consume (st.cell k.ident)
    (r.1, { st with cell := storeUpdate st.cell k.ident r.2 })
  else
    (Residue.nothing, st)

/-- Reclaim a slot and move it to the next generation.

    {α : Type u} (st : GenStore α)     [a store]
    (i : Ident) (a : α)                [an identity and the value going back there]
    : GenStore α                       [the store after]

    So: `reclaimAt` with the counter bumped. The bump is what every key minted before this point
    now fails against. -/
def genReclaim {α : Type u} (st : GenStore α) (i : Ident) (a : α) : GenStore α :=
  { cell := storeUpdate st.cell i (Linear.live a),
    gen := fun j => if j = i then st.gen j + 1 else st.gen j }

/-- A reach whose generation does not match delivers the residue without reading the slot.

    {α : Type u} (st : GenStore α) (k : GenKey)          [a store and a tagged key]
    (h : st.gen k.ident ≠ k.gen)                         [the key is a generation behind]
    (genConsume st k).1 = Residue.nothing                [it gets the residue]
    ≠                                                    [is not equal to]

    So: the whole guard, in one line. Everything below is an argument that the generation does
    not match, and this turns that into the verdict. -/
theorem genConsume_stale {α : Type u} (st : GenStore α) (k : GenKey)
    (h : st.gen k.ident ≠ k.gen) : (genConsume st k).1 = Residue.nothing := by
  simp [genConsume, h]

/-- A key held across a reclaim finds the residue.

    {α : Type u} (st : GenStore α) (k : GenKey) (b : α)      [a store, a tagged key, a value]
    (hgen : st.gen k.ident = k.gen)                          [the key was current to begin with]
    (genConsume (genReclaim (genConsume st k).2 k.ident b) k).1 = Residue.nothing
                                                             [after the reclaim it delivers nothing]

    So: this is what `reuse_readmits_delivery` says is impossible without the tag. Consuming does
    not move the counter, so the key is still current when the reclaim runs; the reclaim moves it,
    and the key is one behind from then on however many times the slot turns over. -/
theorem generation_closes_reuse {α : Type u} (st : GenStore α) (k : GenKey) (b : α)
    (hgen : st.gen k.ident = k.gen) :
    (genConsume (genReclaim (genConsume st k).2 k.ident b) k).1 = Residue.nothing := by
  refine genConsume_stale _ k ?_
  have hstep : (genConsume st k).2.gen = st.gen := by simp [genConsume, hgen]
  have hafter : (genReclaim (genConsume st k).2 k.ident b).gen k.ident = k.gen + 1 := by
    simp [genReclaim, hstep, hgen]
  rw [hafter]
  omega

/-- Two keys on one identity both find the residue after a reclaim.

    {α : Type u} (st : GenStore α) (k₁ k₂ : GenKey) (b : α)  [a store, two tagged keys, a value]
    (halias : k₁.ident = k₂.ident)                           [both name one identity]
    (hg₁ : st.gen k₁.ident = k₁.gen)                         [the first was current]
    (hg₂ : st.gen k₂.ident = k₂.gen)                         [so was the second]
    (genConsume (genReclaim (genConsume st k₁).2 k₁.ident b) k₂).1 = Residue.nothing
                                                             [the alias delivers nothing]

    So: the direct answer to `reuse_readmits_delivery`. Aliasing is not treated separately, because
    the counter lives on the identity and every key naming it goes stale at the same moment.
    `unaliased_no_cross_talk` said injectivity of the key map is the dividing line for the
    untagged scheme; with the tag there is no dividing line to be on the wrong side of. -/
theorem generation_closes_alias {α : Type u} (st : GenStore α) (k₁ k₂ : GenKey) (b : α)
    (halias : k₁.ident = k₂.ident)
    (hg₁ : st.gen k₁.ident = k₁.gen) (hg₂ : st.gen k₂.ident = k₂.gen) :
    (genConsume (genReclaim (genConsume st k₁).2 k₁.ident b) k₂).1 = Residue.nothing := by
  refine genConsume_stale _ k₂ ?_
  have hstep : (genConsume st k₁).2.gen = st.gen := by simp [genConsume, hg₁]
  have hafter : (genReclaim (genConsume st k₁).2 k₁.ident b).gen k₂.ident = k₂.gen + 1 := by
    rw [← halias]
    simp [genReclaim, hstep, hg₁]
    rw [← hg₁, halias, hg₂]
  rw [hafter]
  omega

end Resource
end Apeiron
