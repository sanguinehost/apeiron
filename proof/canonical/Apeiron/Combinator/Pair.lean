import Apeiron.Combinator.Basic

/-!
# The pair, and the one constant as a pair

@name Pair
@cite barker-2001
@cite goldberg-2004
@cite schonfinkel-1924
@cite curry-1930
@grade proved

`pack a b` is the term that stores two terms and hands them, in order, to whatever it is
applied to:

```
   pack a b m     becomes    m a b
```

Its construction is `S (S I (K a)) (K b)`, and `pack_app` proves the rule above by running
the reductions. Storing is one half; reading back is the other. Applying a packed pair to
`K` returns the first stored term, and applying it to `K I` returns the second (`pack_fst`,
`pack_snd`). The reader chooses a slot, and the discarding constant does the choosing.

`iota` is not a new shape. Unfolding both definitions gives the same term:

```
   iota = pack S K
```

`iota_eq_pack` is that identity, and it holds by definition, with no reduction taken. The
one constant that generates the calculus is the pair holding the calculus's own two
constants. Its single rule, `iota M` reduces to `M S K`, is `pack_app` at `a = S` and
`b = K`. Reading its two slots returns the constants themselves: `iota K` reaches `S`, and
`iota (K I)` reaches `K` (`iota_fst`, `iota_snd`). `iota_is_pair` bundles the three.
-/

namespace Apeiron
namespace Combinator
namespace Comb

/-- The term that stores two terms, in order.

    (a b : Comb)      [the two terms being stored]
    : Comb            [the term holding them]

    So: built as `S (S I (K a)) (K b)`. The one rule it obeys is `pack_app` below. -/
def pack (a b : Comb) : Comb := app (app S (app (app S I) (app K a))) (app K b)

/-- A packed pair hands its argument the two stored terms, in order.

    (a b m : Comb)                                [two stored terms, and any argument]
    Reduces (app (pack a b) m) (app (app m a) b)  [pack a b applied to m reaches m a b]

    So: `S` hands `m` to both halves, the identity surrenders `m` itself, and the two `K`s
    surrender the stored terms. -/
theorem pack_app (a b m : Comb) : Reduces (app (pack a b) m) (app (app m a) b) := by
  refine Reduces.trans (Reduces.single (Step.s _ _ m)) ?_
  refine Reduces.trans (Reduces.left _ (Reduces.single (Step.s I (app K a) m))) ?_
  refine Reduces.trans (Reduces.left _ (Reduces.left _ (I_app m))) ?_
  refine Reduces.trans (Reduces.left _ (Reduces.right _ (Reduces.single (Step.k a m)))) ?_
  exact Reduces.right _ (Reduces.single (Step.k b m))

/-- Reading the first slot: apply the pair to the discarding constant.

    (a b : Comb)                       [the two stored terms]
    Reduces (app (pack a b) K) a       [the pair applied to K reaches the first term]

    So: the pair hands `K` both terms, and `K` keeps the first. -/
theorem pack_fst (a b : Comb) : Reduces (app (pack a b) K) a :=
  Reduces.trans (pack_app a b K) (Reduces.single (Step.k a b))

/-- Reading the second slot: apply the pair to `K I`.

    (a b : Comb)                               [the two stored terms]
    Reduces (app (pack a b) (app K I)) b       [the pair applied to K I reaches the second]

    So: the pair hands `K I` both terms, `K` throws the first away and leaves the
    identity, and the identity surrenders the second. -/
theorem pack_snd (a b : Comb) : Reduces (app (pack a b) (app K I)) b := by
  refine Reduces.trans (pack_app a b (app K I)) ?_
  refine Reduces.trans (Reduces.left b (Reduces.single (Step.k I a))) ?_
  exact I_app b

/-- The one constant is the pair holding the two constants.

@cite barker-2001

    iota = pack S K        [the two terms are identical, symbol for symbol]

    So: unfolding `iota` and unfolding `pack S K` give the same tree. No reduction is
    taken; the identity is definitional. -/
theorem iota_eq_pack : iota = pack S K := rfl

/-- Reading the one constant's first slot returns the duplicating constant.

    Reduces (app iota K) S        [iota applied to K reaches S] -/
theorem iota_fst : Reduces (app iota K) S := by
  rw [iota_eq_pack]
  exact pack_fst S K

/-- Reading the one constant's second slot returns the discarding constant.

    Reduces (app iota (app K I)) K        [iota applied to K I reaches K] -/
theorem iota_snd : Reduces (app iota (app K I)) K := by
  rw [iota_eq_pack]
  exact pack_snd S K

/-- The one constant is a two-slot pair, and its slots hold the calculus.

    iota = pack S K                       [it is the pair of S and K, definitionally]
    Reduces (app iota K) S                [its first slot reads back as S]
    Reduces (app iota (app K I)) K        [its second slot reads back as K]
    ∧                                     [all three together]

    So: `iota_generates` showed one constant regrows `I`, `K`, and `S` by self-application.
    This says what that constant is: a container with two ordered slots, read by discard. -/
theorem iota_is_pair :
    iota = pack S K ∧ Reduces (app iota K) S ∧ Reduces (app iota (app K I)) K :=
  ⟨iota_eq_pack, iota_fst, iota_snd⟩

end Comb
end Combinator
end Apeiron
