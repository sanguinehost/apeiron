import Apeiron.Combinator.Linear.Exponential

/-!
# Functoriality: an application carried under the exponential

@name Functorial
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@cite joyal-street-verity-1996
@grade proved

The last of the three equations about the exponential. Abramsky, Haghverdi and Scott write it
`F !x !y = !(xy)`: a reusable copy of a route applied to a reusable copy of an argument is a
reusable copy of the application.

## Root

| from | here |
|---|---|
| `one_mul`, `mul_comm` through `mul_one_right` | `bangArg_copyPos`, `plug_copyPos` |

The relabelling is a bijection of the counting numbers, so the same routing serves every setting
and `theory/NOVELTY.md`'s A1 spectator test applies as it has throughout. The parameter weights a
path.

## What this is not

`Combinator/Duplication.lean`'s `bang_rid` and `bang_rcomp` say the exponential keeps identities
and carries composition, which is functoriality for the monoid `rcomp` makes. That is not this.
Application here is the trace, not the composition, so what is needed is that the exponential
commutes with the **trace**, which is the traced functor condition in Abramsky, Haghverdi and
Scott's Geometry of Interaction situation and does not follow from carrying composition. It is
proved below at the level of paths, and it needs a new tool.

## The tool

`Combinator/Tracks.lean`'s `runs_embed` transports a path relation along an embedding, but it
asks that the wide loop be exactly the image of the narrow one. Here it cannot be: the loop is
every odd position, the embedding reaches the odd positions of one copy, and the other copies are
loop positions the path never visits. `runs_embed_wide` is `runs_embed` with the wide loop
allowed to be larger, provided the surplus is off the image. The proof is the same pair of
inductions with one extra case, where a path lands on a position which is in the image and so
cannot be in the surplus.

## The layout

A position of `bang x` is a tag and a position of `x`. A position of the wire on which `bang x`
is applied is an even or odd position whose payload is a tag and a position of `x`. `copySplit`
is the bijection between those two readings, `bangShift x` is `bang x` read the second way, and
`copyPos t` embeds the whole of `x`'s wire into copy `t`. Every path of the applied machine stays
in the copy it starts in, which is the content of the equation, and `plug_copyPos` is that
statement for a single step.

## The eight equations, and what is and is not claimed

The carrier is the one `Combinator/Interaction.lean` builds: a route is a weighted partial
injection of the counting numbers, `Nat → Option (Nat times Cell)`, composition multiplies the
weight along a path the parameter's way, and application is the trace,
`rapp sigma a b = traceRoute sigma (plug sigma a b)`, with the exponential `bang` running a
route inside every one of the countably many copies Cantor's pairing supplies. At that carrier
all eight equations of Abramsky, Haghverdi and Scott's linear combinatory algebra are proved:

```
   I x = x                 rapp_iRoute          a constant route, iRoute
   K x !y = x              rapp_kRoute_bang     a constant route, kRoute
   D !x = x                rapp_dRoute          a constant route, dRoute
   digging !x = !!x        rapp_diggingRoute    a constant route, diggingRoute
   B x y z = x (y z)       bracket              a construction, bComb
   C x y z = x z y         commutator           a construction, cComb
   W x y = x y y           contraction          a relabelling, conjRoute wIdx x
   F !x !y = !(x y)        functoriality        a relabelling, bangShift x
```

and `S x y z = x z (y z)` besides, as `substitution`, a construction `sComb`.

**Two caveats, and they are not decoration.** First, the contraction and the substitution take
their duplicated argument as its two copies, `rtensor y y` and `rtensor z z`, which is the tensor
form linearity gives. Abramsky, Haghverdi and Scott write the contraction with the exponential,
`W x !y = x !y !y`; passing from the form proved here to that one needs the comultiplication
carrying `!y` to `!y` tensor `!y`, which is not proved here. Second, four of the nine are stated
with a constant route and five with a construction or a relabelling of a stored route. There is
no constant route `bRoute` here with `rapp sigma (rapp sigma bRoute x) y = bComb sigma x y`, and
none is proved for the others either. So what holds is the eight equations at this carrier, and
the statement that these routes form a linear combinatory algebra as a structure, with the
combinators as elements of it, is **not** claimed and would need exactly those two things.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy
open RouteAnyRing

-- ═══════════════════════════════════════════════════════════════════
-- A PATH RELATION TRANSPORTED INTO ONE COPY OF A LARGER LOOP
-- ═══════════════════════════════════════════════════════════════════

/-- A wide path from an embedded start is a narrow one, when the surplus is off the image.

    {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)      [the parameter and an embedding]
    {f g : Route}                                        [a wide route and a narrow one]
    (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2))   [agreeing on the image]
    {S T : Set ℕ} (hT : ∀ a, ι a ∉ T)                    [a loop and a surplus off the image]
    ∀ {n a b : ℕ} {w : Cell}                             [a length, two positions, a weight]
    RunsN σ f (ι '' S ∪ T) n a b w → ∀ i, a = ι i →      [a wide path from an embedded start]
      ∃ j, b = ι j ∧ RunsN σ g S n i j w                 [is a narrow one, ending embedded] -/
theorem runsN_embed_wide_fwd {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)
    {f g : Route} (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2)) {S T : Set ℕ}
    (hT : ∀ a, ι a ∉ T) :
    ∀ {n a b : ℕ} {w : Cell}, RunsN σ f (ι '' S ∪ T) n a b w → ∀ i, a = ι i →
      ∃ j, b = ι j ∧ RunsN σ g S n i j w := by
  intro n a b w h
  induction h with
  | @exit a b w hf hns =>
    intro i hai
    subst hai
    rw [hfg i] at hf
    obtain ⟨⟨c1, c2⟩, hg, hq⟩ := Option.map_eq_some_iff.mp hf
    injection hq with h1 h2
    subst h1
    subst h2
    exact ⟨c1, rfl, RunsN.exit hg fun hc => hns (Or.inl ⟨c1, hc, rfl⟩)⟩
  | @step n a k b w v hf hs hr ih =>
    intro i hai
    subst hai
    rw [hfg i] at hf
    obtain ⟨⟨c1, c2⟩, hg, hq⟩ := Option.map_eq_some_iff.mp hf
    injection hq with h1 h2
    subst h1
    subst h2
    rcases hs with ⟨t, htS, hti⟩ | hbad
    · have htc : t = c1 := hι hti
      subst htc
      obtain ⟨j, hbj, hrun⟩ := ih t rfl
      exact ⟨j, hbj, RunsN.step hg htS hrun⟩
    · exact absurd hbad (hT c1)

/-- A narrow path is a wide one, when the surplus is off the image.

    {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)      [the parameter and an embedding]
    {f g : Route}                                        [a wide route and a narrow one]
    (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2))   [agreeing on the image]
    {S T : Set ℕ} (hT : ∀ a, ι a ∉ T)                    [a loop and a surplus off the image]
    ∀ {n i j : ℕ} {w : Cell}, RunsN σ g S n i j w →      [a narrow path]
      RunsN σ f (ι '' S ∪ T) n (ι i) (ι j) w             [is the same path, embedded] -/
theorem runsN_embed_wide_bwd {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)
    {f g : Route} (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2)) {S T : Set ℕ}
    (hT : ∀ a, ι a ∉ T) :
    ∀ {n i j : ℕ} {w : Cell}, RunsN σ g S n i j w → RunsN σ f (ι '' S ∪ T) n (ι i) (ι j) w := by
  intro n i j w h
  induction h with
  | @exit i j w hg hns =>
    refine RunsN.exit ?_ ?_
    · rw [hfg i, hg]; rfl
    · rintro (⟨t, htS, htj⟩ | hbad)
      · exact hns (hι htj ▸ htS)
      · exact hT j hbad
  | @step n i k j w v hg hs hr ih =>
    refine RunsN.step ?_ (Or.inl ⟨k, hs, rfl⟩) ih
    rw [hfg i, hg]; rfl

/-- **A path relation transports into one copy of a larger loop.**

@cite girard-1989

    {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)      [the parameter and an embedding]
    {f g : Route}                                        [a wide route and a narrow one]
    (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2))   [agreeing on the image]
    {S T : Set ℕ} (hT : ∀ a, ι a ∉ T)                    [a loop and a surplus off the image]
    (i j : ℕ) (w : Cell)                                 [two positions and a weight]
    Runs σ f (ι '' S ∪ T) (ι i) (ι j) w                  [a wide path between embedded points]
    ↔                                                    [exactly when]
    Runs σ g S i j w                                     [a narrow path between the points]

    So: `runs_embed` asks that the wide loop be exactly the image. The exponential breaks that,
    because it puts countably many copies of the loop on the wire and a path visits one of them,
    and this is the version which allows the rest. -/
theorem runs_embed_wide {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)
    {f g : Route} (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2)) {S T : Set ℕ}
    (hT : ∀ a, ι a ∉ T) (i j : ℕ) (w : Cell) :
    Runs σ f (ι '' S ∪ T) (ι i) (ι j) w ↔ Runs σ g S i j w := by
  constructor
  · rintro ⟨n, h⟩
    obtain ⟨j', hj', hrun⟩ := runsN_embed_wide_fwd hι hfg hT h i rfl
    have hjj : j = j' := hι hj'
    subst hjj
    exact ⟨n, hrun⟩
  · rintro ⟨n, h⟩
    exact ⟨n, runsN_embed_wide_bwd hι hfg hT h⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE TAG OUTSIDE OR INSIDE
-- ═══════════════════════════════════════════════════════════════════

/-- The second half of a pair is determined by the pair.

    {t p p' : ℕ} (h : Nat.pair t p = Nat.pair t p')     [two pairs with the same tag]
    p = p'                                              [have the same payload] -/
theorem pair_inj_right {t p p' : ℕ} (h : Nat.pair t p = Nat.pair t p') : p = p' := by
  simpa [Nat.unpair_pair] using congrArg (fun n => (Nat.unpair n).2) h

/-- Read a position of the wire, whose payload is tagged, as a tagged position of the wire.

    (n : ℕ)                                        [a position whose payload carries a tag]
    ℕ                                              [the tag, carrying a position] -/
def copySplitFun (n : ℕ) : ℕ :=
  Sum.elim
    (fun c => Nat.pair (Nat.unpair c).1 (extPos (Nat.unpair c).2))
    (fun c => Nat.pair (Nat.unpair c).1 (argPos (Nat.unpair c).2))
    (Equiv.natSumNatEquivNat.symm n)

/-- Read a tagged position of the wire as a position whose payload is tagged.

    (m : ℕ)                                        [a tag, carrying a position]
    ℕ                                              [a position whose payload carries the tag] -/
def copySplitInv (m : ℕ) : ℕ :=
  Sum.elim
    (fun p => extPos (Nat.pair (Nat.unpair m).1 p))
    (fun u => argPos (Nat.pair (Nat.unpair m).1 u))
    (Equiv.natSumNatEquivNat.symm (Nat.unpair m).2)

/-- The reading on an even position.

    (t p : ℕ)                                                          [a tag and a position]
    copySplitFun (extPos (Nat.pair t p)) = Nat.pair t (extPos p)       [the tag comes out] -/
theorem copySplitFun_ext (t p : ℕ) :
    copySplitFun (extPos (Nat.pair t p)) = Nat.pair t (extPos p) := by
  simp only [copySplitFun, extPos, Equiv.symm_apply_apply, Sum.elim_inl, Nat.unpair_pair]

/-- The reading on an odd position.

    (t u : ℕ)                                                          [a tag and a position]
    copySplitFun (argPos (Nat.pair t u)) = Nat.pair t (argPos u)       [the tag comes out] -/
theorem copySplitFun_arg (t u : ℕ) :
    copySplitFun (argPos (Nat.pair t u)) = Nat.pair t (argPos u) := by
  simp only [copySplitFun, argPos, Equiv.symm_apply_apply, Sum.elim_inr, Nat.unpair_pair]

/-- The reading back on an even position.

    (t p : ℕ)                                                          [a tag and a position]
    copySplitInv (Nat.pair t (extPos p)) = extPos (Nat.pair t p)       [the tag goes in] -/
theorem copySplitInv_ext (t p : ℕ) :
    copySplitInv (Nat.pair t (extPos p)) = extPos (Nat.pair t p) := by
  simp only [copySplitInv, extPos, Nat.unpair_pair, Equiv.symm_apply_apply, Sum.elim_inl]

/-- The reading back on an odd position.

    (t u : ℕ)                                                          [a tag and a position]
    copySplitInv (Nat.pair t (argPos u)) = argPos (Nat.pair t u)       [the tag goes in] -/
theorem copySplitInv_arg (t u : ℕ) :
    copySplitInv (Nat.pair t (argPos u)) = argPos (Nat.pair t u) := by
  simp only [copySplitInv, argPos, Nat.unpair_pair, Equiv.symm_apply_apply, Sum.elim_inr]

/-- The two readings are a bijection.

    ℕ ≃ ℕ                                          [the tag outside or inside] -/
def copySplit : ℕ ≃ ℕ where
  toFun := copySplitFun
  invFun := copySplitInv
  left_inv := by
    intro n
    rcases pos_split2 n with ⟨c, rfl⟩ | ⟨c, rfl⟩ <;>
      obtain ⟨t, p, rfl⟩ : ∃ t p, c = Nat.pair t p := ⟨_, _, (Nat.pair_unpair c).symm⟩
    · rw [copySplitFun_ext, copySplitInv_ext]
    · rw [copySplitFun_arg, copySplitInv_arg]
  right_inv := by
    intro m
    obtain ⟨t, r, rfl⟩ : ∃ t r, m = Nat.pair t r := ⟨_, _, (Nat.pair_unpair m).symm⟩
    rcases pos_split2 r with ⟨p, rfl⟩ | ⟨u, rfl⟩
    · rw [copySplitInv_ext, copySplitFun_ext]
    · rw [copySplitInv_arg, copySplitFun_arg]

/-- The bijection is the map.

    (n : ℕ)                                        [a position]
    copySplit n = copySplitFun n                   [the same reading] -/
theorem copySplit_apply (n : ℕ) : copySplit n = copySplitFun n := rfl

/-- The inverse bijection is the inverse map.

    (n : ℕ)                                        [a position]
    copySplit.symm n = copySplitInv n              [the same reading back] -/
theorem copySplit_symm_apply (n : ℕ) : copySplit.symm n = copySplitInv n := rfl

/-- The whole of a wire, embedded in one copy.

    (t r : ℕ)                                      [which copy, and a position of the wire]
    copySplit.symm (Nat.pair t r)                  [that position, inside that copy]
    : ℕ                                            [a position of the wide wire] -/
def copyPos (t r : ℕ) : ℕ := copySplit.symm (Nat.pair t r)

/-- An even position embedded in a copy stays even.

    (t p : ℕ)                                      [which copy, and a position]
    copyPos t (extPos p) = extPos (Nat.pair t p)   [the tag rides in the payload] -/
theorem copyPos_ext (t p : ℕ) : copyPos t (extPos p) = extPos (Nat.pair t p) :=
  copySplitInv_ext t p

/-- An odd position embedded in a copy stays odd.

    (t u : ℕ)                                      [which copy, and a position]
    copyPos t (argPos u) = argPos (Nat.pair t u)   [the tag rides in the payload] -/
theorem copyPos_arg (t u : ℕ) : copyPos t (argPos u) = argPos (Nat.pair t u) :=
  copySplitInv_arg t u

/-- Reading a copy's position back gives the tag and the position.

    (t r : ℕ)                                      [which copy, and a position]
    copySplit (copyPos t r) = Nat.pair t r         [the round trip] -/
theorem copySplit_copyPos (t r : ℕ) : copySplit (copyPos t r) = Nat.pair t r :=
  copySplit.apply_symm_apply _

/-- Distinct positions stay distinct inside a copy.

    (t : ℕ)                                        [which copy]
    Function.Injective (copyPos t)                 [the copy is a faithful image of the wire] -/
theorem copyPos_injective (t : ℕ) : Function.Injective (copyPos t) := by
  intro a b h
  rcases pos_split2 a with ⟨p, rfl⟩ | ⟨u, rfl⟩ <;> rcases pos_split2 b with ⟨p', rfl⟩ | ⟨u', rfl⟩
  · rw [copyPos_ext, copyPos_ext] at h
    rw [pair_inj_right (extPos_injective h)]
  · rw [copyPos_ext, copyPos_arg] at h
    exact absurd h (extPos_ne_argPos _ _)
  · rw [copyPos_arg, copyPos_ext] at h
    exact absurd h.symm (extPos_ne_argPos _ _)
  · rw [copyPos_arg, copyPos_arg] at h
    rw [pair_inj_right (argPos_injective h)]

/-- The loop positions outside one copy.

    (t : ℕ)                                        [which copy]
    Set ℕ                                          [the loop positions of the other copies] -/
def copyOff (t : ℕ) : Set ℕ := oddPos \ (copyPos t '' oddPos)

/-- A copy's loop positions are loop positions.

    (t : ℕ)                                        [which copy]
    copyPos t '' oddPos ⊆ oddPos                   [embedding keeps a position odd] -/
theorem copyPos_image_subset (t : ℕ) : copyPos t '' oddPos ⊆ oddPos := by
  rintro q ⟨r, ⟨u, rfl⟩, rfl⟩
  exact ⟨Nat.pair t u, copyPos_arg t u⟩

/-- One copy and the rest are the whole loop.

    (t : ℕ)                                                  [which copy]
    copyPos t '' oddPos ∪ copyOff t = oddPos                 [every loop position is somewhere] -/
theorem copyPos_union (t : ℕ) : copyPos t '' oddPos ∪ copyOff t = oddPos := by
  ext q
  constructor
  · rintro (h | ⟨h, _⟩)
    · exact copyPos_image_subset t h
    · exact h
  · intro h
    by_cases hc : q ∈ copyPos t '' oddPos
    · exact Or.inl hc
    · exact Or.inr ⟨h, hc⟩

/-- Nothing in a copy is outside it.

    (t a : ℕ)                                      [which copy, and a position]
    copyPos t a ∉ copyOff t                        [the surplus is off the image] -/
theorem copyPos_not_off (t a : ℕ) : copyPos t a ∉ copyOff t := by
  rintro ⟨hmem, hnot⟩
  rcases pos_split2 a with ⟨p, rfl⟩ | ⟨u, rfl⟩
  · rw [copyPos_ext] at hmem
    exact not_mem_oddPos (Nat.pair t p) hmem
  · exact hnot ⟨argPos u, ⟨u, rfl⟩, rfl⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE EXPONENTIAL READ ON THE APPLICATION'S OWN WIRE
-- ═══════════════════════════════════════════════════════════════════

/-- The exponential of a route, read on the wire the application uses.

@cite abramsky-haghverdi-scott-2002

    (x : Route)                                    [a weighted wire map]
    conjRoute copySplit (bang x)                   [the copies with the tag read inside]
    : Route                                        [one route on the wide wire]

    So: `bang x` tags a position of `x`; this reads the tag inside, so that a position of the
    combined wire is even or odd with a tagged payload. That is the reading on which the argument
    channel of copy `t` faces copy `t` of the argument, which is what the equation needs. -/
def bangShift (x : Route) : Route := conjRoute copySplit (bang x)

/-- The shifted exponential inside one copy is the route itself.

    (x : Route) (t a : ℕ)                          [a route, which copy, a position]
    bangShift x (copyPos t a)                                    [a step inside copy t]
    = (x a).map fun q => (copyPos t q.1, q.2)                    [is the route, in that copy] -/
theorem bangShift_copyPos (x : Route) (t a : ℕ) :
    bangShift x (copyPos t a) = (x a).map fun q => (copyPos t q.1, q.2) := by
  rw [bangShift, conjRoute, copySplit_copyPos, bang_tag, Option.map_map]
  cases hx : x a with
  | none => rfl
  | some q => rfl

/-- The exponential of an argument inside one copy is the argument itself.

    (y : Route) (t c : ℕ)                          [a route, which copy, a position]
    rtensor rid (bang y) (copyPos t c)                           [the argument layer in copy t]
    = ((rtensor rid y) c).map fun q => (copyPos t q.1, q.2)      [is its own argument layer] -/
theorem bangArg_copyPos (y : Route) (t c : ℕ) :
    rtensor rid (bang y) (copyPos t c)
      = ((rtensor rid y) c).map fun q => (copyPos t q.1, q.2) := by
  rcases pos_split2 c with ⟨p, rfl⟩ | ⟨u, rfl⟩
  · rw [copyPos_ext, rtensor_ext_id, rtensor_ext_id, Option.map_some, copyPos_ext]
  · rw [copyPos_arg, rtensor_arg, rtensor_arg, bang_tag]
    cases hy : y u with
    | none => rfl
    | some q =>
      rw [Option.map_some, Option.map_some, Option.map_some, Option.map_some, copyPos_arg]

/-- **One step of the applied machine stays in the copy it started in.**

@cite girard-1989

    (σ : ℝ) (x y : Route) (t a : ℕ)                [the parameter, two routes, a copy, a position]
    plug σ (bangShift x) (bang y) (copyPos t a)                  [a step of the applied machine]
    = (plug σ x y a).map fun q => (copyPos t q.1, q.2)           [is a step of the plain one]

    So: the copies do not talk to each other. That is the whole geometric content of
    functoriality, and every path statement below is this one lemma run along a path. -/
theorem plug_copyPos (σ : ℝ) (x y : Route) (t a : ℕ) :
    plug σ (bangShift x) (bang y) (copyPos t a)
      = (plug σ x y a).map fun q => (copyPos t q.1, q.2) := by
  rw [plug, rcomp_embed (bangShift_copyPos x t) (bangArg_copyPos y t) a, plug]

/-- **Functoriality**, `!x` applied to `!y` is `!(x y)`, as an equation between routes.

@cite abramsky-haghverdi-scott-2002
@cite girard-1989
@cite joyal-street-verity-1996

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    rapp σ (bangShift x) (bang y) = bang (rapp σ x y)  [applying the copies copies the applying]

    So: the exponential commutes with the trace, not only with composition. `bang_rcomp` gives
    the second and is not enough, because application here is the trace. What makes it true is
    that a path of the applied machine never leaves the copy it starts in, so the trace taken
    over all the copies at once splits into one trace per copy. -/
theorem functoriality (σ : ℝ) (x y : Route) :
    rapp σ (bangShift x) (bang y) = bang (rapp σ x y) := by
  refine route_ext fun i j w => ?_
  obtain ⟨t, p, rfl⟩ : ∃ t p, i = Nat.pair t p := ⟨_, _, (Nat.pair_unpair i).symm⟩
  rw [rapp, ← tracesTo_traceRoute σ (plug σ (bangShift x) (bang y)) (Nat.pair t p) j w,
    traces_iff_runs_ext, bang_tag]
  constructor
  · intro h
    rw [← copyPos_union t, ← copyPos_ext] at h
    obtain ⟨n, hn⟩ := h
    obtain ⟨jj, hjj, hrun⟩ := runsN_embed_wide_fwd (copyPos_injective t)
      (plug_copyPos σ x y t) (copyPos_not_off t) hn (extPos p) rfl
    obtain ⟨p', rfl⟩ := not_oddPos_ext (runs_end_not_mem ⟨n, hrun⟩)
    rw [copyPos_ext] at hjj
    have hj : j = Nat.pair t p' := extPos_injective hjj
    subst hj
    have hxy : rapp σ x y p = some (p', w) :=
      (tracesTo_traceRoute σ (plug σ x y) p p' w).mp
        ((traces_iff_runs_ext σ (plug σ x y) p p' w).mpr ⟨n, hrun⟩)
    rw [hxy, Option.map_some]
  · intro h
    cases hxy : rapp σ x y p with
    | none => rw [hxy] at h; simp at h
    | some q =>
      rw [hxy, Option.map_some] at h
      injection h with h1
      injection h1 with hj hw
      subst hj
      subst hw
      have hr : Runs σ (plug σ x y) oddPos (extPos p) (extPos q.1) q.2 :=
        (traces_iff_runs_ext σ (plug σ x y) p q.1 q.2).mp
          ((tracesTo_traceRoute σ (plug σ x y) p q.1 q.2).mpr hxy)
      have hw := (runs_embed_wide (copyPos_injective t) (plug_copyPos σ x y t)
        (copyPos_not_off t) (extPos p) (extPos q.1) q.2).mpr hr
      rw [copyPos_union, copyPos_ext, copyPos_ext] at hw
      exact hw

end Combinator
end Apeiron

end
