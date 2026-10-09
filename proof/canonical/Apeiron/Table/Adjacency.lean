import Apeiron.Table.Basic

/-!
# The graph face of the table

@name Adjacency
@grade proved

Give the table list values and merge by `++`, and each key holds an adjacency list: inserting
`(v, [w])` records an edge from `v` to `w`, and reading `v` returns its neighbours. This is the
fourth of the four merges the predecessor names for the one table: presence by `∨`, replace,
count by `+`, and append.

Sort those four by the hypotheses of `build_gauge_invariance` and they split two and two. The
set's `∨` and the counter's `+` are commutative and associative, so for them insertion order is
a schedule, and an implementation may sort the stream or shard it across cores. Replace and
`++` are associative and not commutative, and for them that licence does not exist: order is
part of the meaning. A map remembers which write came last, and an adjacency list remembers the
order the edges arrived in.

Three theorems make that a fact rather than a remark:

- `get_build_append` is what associativity alone gives: the adjacency list at a key is the
  concatenation of the lists inserted at that key, latest insertion first. The stream
  determines it in full, order included.
- `build_append_perm_pointwise` is what survives of gauge invariance: permuting the stream
  permutes each adjacency list. Which edges a key holds is schedule-free; where they sit in
  the list is not.
- `append_not_gauge` is the failure itself: swapping two insertions builds a different table,
  whenever the edge type has two distinct elements. So the commutativity hypothesis of
  `build_gauge_invariance` cannot be dropped.

The distinct-elements condition is not decoration. Over a one-element edge type, a permutation
preserves each key's total edge count, and a list of copies of one thing is determined by its
length, so order washes out and the permutation licence still holds there. Two distinct
elements is the least that can break it, and it does.

What remains for an implementation is `build_eq_of_key_streams_eq` in `Table/Basic.lean`: under
every merge, any reordering that keeps each key's entries in their relative order builds the
same table. Sharding by key stays legal for adjacency; sorting within a key does not.

## The map face sits inside this one

Replace, `fun new _ => new`, is the merge of the predecessor's map face, and last write wins is
an order dependence by name. `get_build_replace` states it: the value at a key is the last
value the stream wrote there. `replace_is_head_of_append` locates it inside the graph face:
reading a replace-built table is reading the head of the append-built adjacency list, key by
key. The map face keeps the head of what the graph face keeps whole.
-/

namespace Apeiron
namespace Table

variable {K : Type} [DecidableEq K]

/-- Folding lists onto an accumulator, newest in front, concatenates them back to front.

    ∀ (l : List (List A)) (acc : List A)     [for every stack of lists and every starting list]
    l.foldl (fun acc v => v ++ acc) acc      [fold each onto the front of the accumulator]
    = l.reverse.flatten ++ acc               [last list first, joined, in front of the start]

    So: this is where associativity of `++` is spent, once, in the induction step. -/
theorem foldl_flip_append {A : Type} :
    ∀ (l : List (List A)) (acc : List A),
      l.foldl (fun acc v => v ++ acc) acc = l.reverse.flatten ++ acc := by
  intro l
  induction l with
  | nil => intro acc; simp
  | cons v rest ih =>
    intro acc
    rw [List.foldl_cons, ih (v ++ acc), List.reverse_cons, List.flatten_append]
    simp [List.append_assoc]

/-- The adjacency list at a key is that key's inserted lists, latest first, concatenated.

    (kvs : List (K × List A)) (x : K)           [a stream of key and edge-list pairs, and a key]
    get (build (· ++ ·) [] kvs) x               [the adjacency list the build leaves at that key]
    = ((kvs.filter (fun kv => kv.1 = x))        [take the stream's entries at that key]
        .map Prod.snd)                          [keep the second of each pair, the edge list]
        .reverse.flatten                        [latest first, joined into one list]

    So: the stream determines the list in full, order included. Nothing at other keys enters,
    and nothing about the order is forgotten. -/
theorem get_build_append {A : Type} (kvs : List (K × List A)) (x : K) :
    get (build (· ++ ·) ([] : List A) kvs) x
      = ((kvs.filter (fun kv => kv.1 = x)).map Prod.snd).reverse.flatten := by
  unfold build
  rw [get_build_acc (· ++ ·) x kvs (fun _ => [])]
  rw [foldl_flip_append]
  simp [get]

/-- Permuting the stream permutes each adjacency list.

    {kvs₁ kvs₂ : List (K × List A)}             [two streams of key and edge-list pairs]
    (hperm : kvs₁.Perm kvs₂) (x : K)            [one a rearrangement of the other, and a key]
    (get (build (· ++ ·) [] kvs₁) x).Perm       [the first build's adjacency list at that key]
      (get (build (· ++ ·) [] kvs₂) x)          [is a rearrangement of the second build's]

    So: this is the part of gauge invariance that survives losing commutativity. Which edges a
    key holds is schedule-free; where they sit in the list is not. -/
theorem build_append_perm_pointwise {A : Type} {kvs₁ kvs₂ : List (K × List A)}
    (hperm : kvs₁.Perm kvs₂) (x : K) :
    (get (build (· ++ ·) ([] : List A) kvs₁) x).Perm
      (get (build (· ++ ·) ([] : List A) kvs₂) x) := by
  rw [get_build_append, get_build_append]
  exact ((List.reverse_perm _).trans
    (((hperm.filter _).map Prod.snd).trans (List.reverse_perm _).symm)).flatten

/-- Swapping two insertions at one key builds a different table.

    (k : K) {a b : A} (hab : a ≠ b)             [a key, and two distinct edges]
    build (· ++ ·) ([] : List A)                [merging by append, from the empty list]
      [(k, [a]), (k, [b])]                      [insert a then b at that key]
    ≠ build (· ++ ·) ([] : List A)              [is not the same build]
      [(k, [b]), (k, [a])]                      [from insert b then a]

    So: the two streams are permutations of each other and the tables differ at `k`, holding
    `[b, a]` and `[a, b]`. -/
theorem build_append_swap_ne {A : Type} (k : K) {a b : A} (hab : a ≠ b) :
    build (· ++ ·) ([] : List A) [(k, [a]), (k, [b])]
      ≠ build (· ++ ·) ([] : List A) [(k, [b]), (k, [a])] := by
  intro h
  have h2 := congrFun h k
  simp [build, insertWith] at h2
  exact hab h2.2

/-- Insertion order is not gauge for append: no permutation licence exists.

    (k : K) {a b : A} (hab : a ≠ b)             [a key, and two distinct edges]
    ¬                                           [it is not true that]
    ∀ (kvs₁ kvs₂ : List (K × List A))           [for all pairs of edge streams]
    kvs₁.Perm kvs₂                              [one a rearrangement of the other]
    →                                           [if that holds then]
    build (· ++ ·) [] kvs₁                      [the first build]
      = build (· ++ ·) [] kvs₂                  [equals the second]

    So: the conclusion of `build_gauge_invariance` is false for `++`, by the two-insertion swap
    of `build_append_swap_ne`. An implementation may not sort or shard an adjacency stream
    without fixing an order; what it keeps is `build_eq_of_key_streams_eq`. -/
theorem append_not_gauge {A : Type} (k : K) {a b : A} (hab : a ≠ b) :
    ¬ ∀ (kvs₁ kvs₂ : List (K × List A)), kvs₁.Perm kvs₂ →
        build (· ++ ·) ([] : List A) kvs₁ = build (· ++ ·) ([] : List A) kvs₂ :=
  fun h => build_append_swap_ne k hab
    (h [(k, [a]), (k, [b])] [(k, [b]), (k, [a])] (List.Perm.swap (k, [b]) (k, [a]) []))

-- ═══════════════════════════════════════════════════════════════════
-- THE MAP FACE IS THE HEAD OF THIS ONE
-- ═══════════════════════════════════════════════════════════════════

/-- The map merge: keep the new value, discard the old.

    {V : Type}               [a value type]
    V → V → V                [takes the new value and the old, gives the new] -/
def replace {V : Type} : V → V → V := fun new _ => new

/-- Folding replace over a list lands on its last element.

    ∀ (l : List V) (d : V)                      [for every list of values and every default]
    l.foldl (fun acc v => replace v acc) d      [fold, each value displacing the accumulator]
    = l.getLastD d                              [is the last value, or the default if none] -/
theorem foldl_replace {V : Type} :
    ∀ (l : List V) (d : V), l.foldl (fun acc v => replace v acc) d = l.getLastD d := by
  intro l
  induction l with
  | nil => intro d; rfl
  | cons v rest ih =>
    intro d
    rw [List.getLastD_cons, List.foldl_cons]
    simpa [replace] using ih v

/-- Under replace, the value at a key is the last value the stream wrote there.

    (d : V) (kvs : List (K × V)) (x : K)        [a default, a stream of pairs, and a key]
    get (build replace d kvs) x                 [the value the build leaves at that key]
    = ((kvs.filter (fun kv => kv.1 = x))        [take the stream's entries at that key]
        .map Prod.snd)                          [keep the second of each pair, the value]
        .getLastD d                             [the last one, or the default if none]

    So: last write wins is order dependence stated as a policy. Replace is associative and not
    commutative, so `build_gauge_invariance` does not apply to the map face either. -/
theorem get_build_replace {V : Type} (d : V) (kvs : List (K × V)) (x : K) :
    get (build replace d kvs) x
      = ((kvs.filter (fun kv => kv.1 = x)).map Prod.snd).getLastD d := by
  unfold build
  rw [get_build_acc replace x kvs (fun _ => d)]
  rw [foldl_replace]
  rfl

/-- Swapping two writes at one key builds a different table under replace.

    (k : K) (d : V) {a b : V} (hab : a ≠ b)     [a key, a default, and two distinct values]
    build replace d [(k, a), (k, b)]            [write a then b at that key]
    ≠ build replace d [(k, b), (k, a)]          [is not write b then a]

    So: the map face fails the permutation licence by the same two-element swap as the graph
    face. -/
theorem build_replace_swap_ne {V : Type} (k : K) (d : V) {a b : V} (hab : a ≠ b) :
    build replace d [(k, a), (k, b)] ≠ build replace d [(k, b), (k, a)] := by
  intro h
  have h2 := congrFun h k
  simp [build, insertWith, replace] at h2
  exact hab h2.symm

/-- Wrapping each element as a one-element list and joining gives the list back.

    (f : B → V)                                 [a read applied to each element]
    ∀ (l : List B)                              [for every list]
    (l.map (fun b => [f b])).flatten            [wrap each read as a one-element list, join]
    = l.map f                                   [is the reads themselves] -/
theorem flatten_map_singleton {B V : Type} (f : B → V) :
    ∀ l : List B, (l.map (fun b => [f b])).flatten = l.map f := by
  intro l
  induction l with
  | nil => rfl
  | cons b rest ih => simp [ih]

/-- The same, through a reversal.

    (f : B → V) (l : List B)                    [a read applied to each element, and a list]
    (l.map (fun b => [f b])).reverse.flatten    [wrap each read, last first, joined]
    = (l.map f).reverse                         [is the reads, reversed] -/
theorem reverse_flatten_map_singleton {B V : Type} (f : B → V) (l : List B) :
    (l.map (fun b => [f b])).reverse.flatten = (l.map f).reverse := by
  rw [← List.map_reverse, flatten_map_singleton, List.map_reverse]

/-- Reading a replace-built table is reading the head of the append-built adjacency list.

    (d : V) (kvs : List (K × V)) (x : K)        [a default, a stream of pairs, and a key]
    get (build replace d kvs) x                 [what the map face holds at that key]
    = (get (build (· ++ ·) []                   [what the graph face holds at that key,]
        (kvs.map (fun kv => (kv.1, [kv.2]))))   [built from the same stream, values wrapped]
        x).headD d                              [read at its first entry, default if none]

    So: the map face is the graph face truncated to its most recent entry. The two faces that
    refuse the permutation licence refuse it for the one reason: `++` keeps the stream's order
    and replace keeps the newest end of it. -/
theorem replace_is_head_of_append {V : Type} (d : V) (kvs : List (K × V)) (x : K) :
    get (build replace d kvs) x
      = (get (build (· ++ ·) ([] : List V)
          (kvs.map (fun kv => (kv.1, [kv.2])))) x).headD d := by
  rw [get_build_replace, get_build_append, List.filter_map]
  simp only [Function.comp_def, List.map_map]
  rw [reverse_flatten_map_singleton]
  simp

end Table
end Apeiron

