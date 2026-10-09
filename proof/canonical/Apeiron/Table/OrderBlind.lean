import Apeiron.Table.Basic

/-!
# Order-blind folds

@name Order-blind folds
@cite shapiro-preguica-baquero-zawirski-2011
@grade proved

`OrderBlind f` says that folding the merge `f`, with the new value first as in
`Table.insertWith`, is unchanged by every permutation for every initial accumulator. The exact
condition is not a value of the trichotomy parameter. It is that the left translations of `f`
commute. Associativity and a right identity reduce that condition to ordinary commutativity.

`build_orderBlind` transfers the list theorem to `Table.build` one key at a time through
`get_build_acc`. `build_gauge_invariance_mul` already shows that this applies to `Cell.mul sigma`
for every `sigma`; `Table.append_not_gauge` supplies the concrete failure for list append.
-/

namespace Apeiron
namespace Table

/-- A merge fold is unchanged by every permutation, from every initial accumulator.

    (f : V -> V -> V)                                  [new value first, old value second]
    forall d xs ys, xs.Perm ys                         [every start and rearranged value stream]
    xs.foldl (fun acc v => f v acc) d                  [the first fold]
      = ys.foldl (fun acc v => f v acc) d              [equals the rearranged fold]

    Quantifying over the initial accumulator is necessary for the exact local characterization
    below. A fixed default can hide noncommutativity that is unreachable from that default. -/
def OrderBlind {V : Type} (f : V -> V -> V) : Prop :=
  forall (d : V) {xs ys : List V}, xs.Perm ys ->
    xs.foldl (fun acc v => f v acc) d = ys.foldl (fun acc v => f v acc) d

/-- A merge fold is order-blind exactly when its left translations commute.

    ↔                                                   [exactly when, true in both directions]
    OrderBlind f                                        [every permuted fold agrees]
    iff                                                 [exactly when]
    forall a b z, f b (f a z) = f a (f b z)           [adjacent updates commute at every state]

    The forward direction uses the two-element swap. The reverse direction is permutation
    induction through `List.Perm.foldl_eq'`. No associativity or identity is hidden here. -/
theorem orderBlind_iff_leftTranslations_commute {V : Type} (f : V -> V -> V) :
    OrderBlind f ↔ forall a b z, f b (f a z) = f a (f b z) := by
  constructor
  · intro h a b z
    simpa [OrderBlind] using h z (List.Perm.swap b a [])
  · intro h d xs ys hperm
    exact hperm.foldl_eq' (fun a _ b _ z => h a b z) d

/-- With associativity and a right identity, order-blindness is ordinary commutativity.

    ↔                                                   [exactly when, true in both directions]
    (hassoc : forall a b c, f (f a b) c = f a (f b c)) [associativity]
    (hright : forall a, f a e = a)                      [right identity]
    OrderBlind f <-> Function.Commutative f              [schedule freedom iff commutativity]

    The right identity exposes a binary application from the three-input translation law.
    Associativity and commutativity give that law in the reverse direction. -/
theorem orderBlind_iff_commutative {V : Type} (f : V -> V -> V) (e : V)
    (hassoc : forall a b c, f (f a b) c = f a (f b c))
    (hright : forall a, f a e = a) :
    OrderBlind f ↔ (forall a b, f a b = f b a) := by
  rw [orderBlind_iff_leftTranslations_commute]
  constructor
  · intro h a b
    have hab := h a b e
    rw [hright a, hright b] at hab
    exact hab.symm
  · intro hcomm a b z
    rw [<- hassoc, hcomm b a, hassoc]

/-- An order-blind value fold makes the table build invariant under stream permutation.

    List                                                [a finite sequence, in order]
    ×                                                   [a key paired with a value]
    (hf : OrderBlind f)                                  [the merge fold ignores order]
    (hperm : kvs1.Perm kvs2)                             [the streams are rearrangements]
    build f d kvs1 = build f d kvs2                      [the resulting tables agree]

    Filtering a permutation at a key and projecting its values preserves permutation. The
    premise then applies directly to the per-key fold exposed by `get_build_acc`. -/
theorem build_orderBlind {K V : Type} [DecidableEq K] {f : V -> V -> V} (hf : OrderBlind f)
    (d : V) {kvs1 kvs2 : List (K × V)} (hperm : kvs1.Perm kvs2) :
    build f d kvs1 = build f d kvs2 := by
  funext x
  show get (build f d kvs1) x = get (build f d kvs2) x
  unfold build
  rw [get_build_acc f x kvs1 (fun _ => d), get_build_acc f x kvs2 (fun _ => d)]
  exact hf d ((hperm.filter _).map Prod.snd)

end Table
end Apeiron
