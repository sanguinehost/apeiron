import Apeiron.Trichotomy.Basic
import Mathlib.Data.List.Perm.Basic

/-!
# The table

@name The Table
@cite stone-1936
@grade proved

A table is a total function `K → V`. Absence is the default, so there is no partiality to carry
and no separate "not found". One operation writes to it: `insertWith f k v` puts `v` at `k`,
merging with what was there by `f`. The merge is the only thing that changes between the jobs a
table does.

`build` folds that operation over a stream of pairs. Three theorems say what a built table is:

- `get_build_acc`: reading at `x` is a fold of the merge over exactly the stream's values at
  `x`, in stream order. Everything at other keys is invisible.
- `build_eq_of_key_streams_eq`: under every merge, the built table depends on the stream only
  through each key's values in their stream order. Reordering across keys is free for any
  merge at all; reordering within a key is what the next theorem's hypotheses buy.
- `build_gauge_invariance`: when the merge is commutative and associative, any permutation of
  the stream builds the same table. Insertion order is then a choice of schedule, not part of
  the meaning, so an implementation may sort by home address, shard across cores, or take
  arrival order, and get the same function.

A merge that is associative but not commutative refuses that last licence, and the table then
remembers its schedule. `Table/Adjacency.lean` proves both halves for `++` and for replace.

## Where σ enters, and where it does not

`mul σ` is commutative (`mul_comm`) and associative (`mul_assoc`) at every setting, so the
algebra's own multiplication is a legal merge and `build_gauge_invariance_mul` follows by
instantiation rather than by assertion. That is the whole of what the algebra says here, and it
is worth being exact about, because more was claimed once and was wrong.

**The discriminant does not classify merges.** A merge is a binary operation on an arbitrary
`V`, a two-dimensional real algebra is a different kind of object, and there is no map between
them for a discriminant to be preserved by. `theory/DEAD_ENDS.md` records the refutation. What
survives is `idempotent_classification`: σ's sign fixes how many idempotents the algebra has,
so how many independent Boolean channels it supplies as a merge, not how merges in general are
sorted.

## What is not here

The predecessor's four face lemmas are worked instances at one key against a constant map, as
its own header says: `count_two` is `1 + (1 + 0) = 2`. They characterise nothing and are not
re-derived. Its `hashmap_trinity` conjoins them and adds nothing beyond the conjunction, which
is the shape `Emergence.lean` was deleted for. Its estimation arc is a separate concern and its
headline readings are declared mappings rather than theorems, by its own grading.
-/

open Apeiron.Trichotomy

namespace Apeiron
namespace Table

/-- A table: a total function from keys to values, with absence carried by the default.

    HMap K V              [a table from keys of type K to values of type V]
    K → V                 [which is just a function] -/
abbrev HMap (K V : Type) : Type := K → V

/-- Read a key.

    {K V : Type}          [a key type and a value type]
    (m : HMap K V)        [a table]
    (k : K)               [a key]
    : V                   [the value there] -/
def get {K V : Type} (m : HMap K V) (k : K) : V := m k

/-- The one write. Put a value at a key, merging with what was there.

    {K V : Type} [DecidableEq K]      [a key type whose keys can be compared]
    (f : V → V → V)                   [the merge, taking the new value first]
    (k : K) (v : V)                   [where to write, and what]
    (m : HMap K V)                    [the table before]
    : HMap K V                        [the table after] -/
def insertWith {K V : Type} [DecidableEq K] (f : V → V → V) (k : K) (v : V) (m : HMap K V) :
    HMap K V := fun x => if x = k then f v (m x) else m x

variable {K V : Type} [DecidableEq K]

/-- Reading where you wrote gives the merge of the new value with the old.

    (f : V → V → V) (k : K) (v : V) (m : HMap K V)     [a merge, a key, a value, a table]
    get (insertWith f k v m) k = f v (m k)             [reading back gives the merge] -/
theorem get_insertWith_self (f : V → V → V) (k : K) (v : V) (m : HMap K V) :
    get (insertWith f k v m) k = f v (m k) := by simp [get, insertWith]

/-- Reading anywhere else is untouched.

    (f : V → V → V) (k : K) (v : V) (m : HMap K V)     [a merge, a key, a value, a table]
    {x : K} (h : x ≠ k)                                [some other key]
    get (insertWith f k v m) x = get m x               [reading there is unchanged] -/
theorem get_insertWith_ne (f : V → V → V) (k : K) (v : V) (m : HMap K V) {x : K} (h : x ≠ k) :
    get (insertWith f k v m) x = get m x := by simp [get, insertWith, h]

/-- The merge is the only freedom the write has.

    {f g : V → V → V}                                  [two merges]
    (h : ∀ a b, f a b = g a b)                         [agreeing at every pair]
    (k : K) (v : V) (m : HMap K V)                     [a key, a value, a table]
    insertWith f k v m = insertWith g k v m            [write the same table]

    So: `insertWith` depends on the merge only through its values, never through how it is
    written. It does not say the merge is the only degree of freedom a table has: the key type,
    the default and the structure are fixed by the definitions above and are not shown
    minimal. -/
theorem merge_is_the_only_freedom {f g : V → V → V} (h : ∀ a b, f a b = g a b)
    (k : K) (v : V) (m : HMap K V) : insertWith f k v m = insertWith g k v m := by
  funext x
  by_cases hx : x = k <;> simp [insertWith, hx, h]

/-- Build a table by folding the write over a stream of pairs, from the constant default.

    (f : V → V → V) (d : V)              [the merge and the default]
    (kvs : List (K × V))                 [a stream of key and value pairs]
    : HMap K V                           [the table it builds] -/
def build (f : V → V → V) (d : V) (kvs : List (K × V)) : HMap K V :=
  kvs.foldl (fun m kv => insertWith f kv.1 kv.2 m) (fun _ => d)

/-- Reading a built table at one key is a fold over exactly that key's values.

    (f : V → V → V) (x : K)                     [a merge, and the key being read]
    ∀ (kvs : List (K × V)) (m : HMap K V)       [for every single stream and starting table]
    get (folding insertWith over kvs) x         [reading the built table at that key]
      = folding f over the values at x          [is a fold of the merge over those values]
    where the values are Prod.snd of the pairs  [the second of each pair, the value]

    So: a build is two-dimensional and a read is one-dimensional. Everything written at other
    keys is invisible here, which is `get_insertWith_ne` applied down the whole stream. -/
theorem get_build_acc (f : V → V → V) (x : K) :
    ∀ (kvs : List (K × V)) (m : HMap K V),
      get (kvs.foldl (fun m kv => insertWith f kv.1 kv.2 m) m) x
        = ((kvs.filter (fun kv => kv.1 = x)).map Prod.snd).foldl (fun acc v => f v acc)
            (get m x) := by
  intro kvs
  induction kvs with
  | nil => intro m; rfl
  | cons kv rest ih =>
    intro m
    rw [List.foldl_cons, ih (insertWith f kv.1 kv.2 m)]
    by_cases h : kv.1 = x
    · rw [List.filter_cons_of_pos (by simpa using h), List.map_cons, List.foldl_cons]
      congr 1
      subst h
      exact get_insertWith_self f kv.1 kv.2 m
    · rw [List.filter_cons_of_neg (by simpa using h)]
      congr 1
      exact get_insertWith_ne f kv.1 kv.2 m (fun hxk => h hxk.symm)

/-- Streams with the same per-key value sequences build the same table, under every merge.

    (f : V → V → V) (d : V)                     [any merge at all, and the default]
    {kvs₁ kvs₂ : List (K × V)}                  [two streams of key and value pairs]
    ∀ x : K                                     [such that at every single key]
    (kvs₁.filter (fun kv => kv.1 = x))          [the first stream's entries at that key]
      .map Prod.snd                             [read off their values, the second of each pair]
    = (kvs₂.filter (fun kv => kv.1 = x))        [equal the second stream's entries there]
      .map Prod.snd                             [read off the same way]
    build f d kvs₁ = build f d kvs₂             [then the two builds are the same table]

    So: a build's meaning is its per-key value streams and nothing about how they interleave.
    This is the schedule freedom every merge has, commutative or not: reorder across keys at
    will, keep each key's order. Sharding a stream by key is licensed here for any merge;
    reordering within a key needs `build_gauge_invariance` and its hypotheses. -/
theorem build_eq_of_key_streams_eq (f : V → V → V) (d : V) {kvs₁ kvs₂ : List (K × V)}
    (h : ∀ x : K, (kvs₁.filter (fun kv => kv.1 = x)).map Prod.snd
        = (kvs₂.filter (fun kv => kv.1 = x)).map Prod.snd) :
    build f d kvs₁ = build f d kvs₂ := by
  funext x
  show get (build f d kvs₁) x = get (build f d kvs₂) x
  unfold build
  rw [get_build_acc f x kvs₁ (fun _ => d), get_build_acc f x kvs₂ (fun _ => d), h x]

/-- Insertion order is a schedule, not part of the meaning.

    {f : V → V → V}                                     [a merge]
    (hcomm : ∀ a b, f a b = f b a)                      [order of its two arguments does not matter]
    (hassoc : ∀ a b c, f (f a b) c = f a (f b c))       [nor does bracketing]
    (d : V)                                             [the default]
    {kvs₁ kvs₂ : List (K × V)} (hperm : kvs₁.Perm kvs₂)   [two streams, one a rearrangement]
    build f d kvs₁ = build f d kvs₂                     [build the same table]

    So: an implementation may sort the stream by home address, shard it across cores, or take
    arrival order, and the function it builds is the same one. Per key the build is a fold over
    that key's values (`get_build_acc`), a rearrangement of the stream rearranges each key's
    list, and a fold of a commutative associative merge does not see the order. -/
theorem build_gauge_invariance {f : V → V → V}
    (hcomm : ∀ a b, f a b = f b a) (hassoc : ∀ a b c, f (f a b) c = f a (f b c))
    (d : V) {kvs₁ kvs₂ : List (K × V)} (hperm : kvs₁.Perm kvs₂) :
    build f d kvs₁ = build f d kvs₂ := by
  funext x
  show get (build f d kvs₁) x = get (build f d kvs₂) x
  unfold build
  rw [get_build_acc f x kvs₁ (fun _ => d), get_build_acc f x kvs₂ (fun _ => d)]
  exact ((hperm.filter _).map Prod.snd).foldl_eq'
    (fun a _ b _ z => by
      show f b (f a z) = f a (f b z)
      rw [← hassoc, hcomm b a, hassoc])
    _

-- ═══════════════════════════════════════════════════════════════════
-- THE ALGEBRA SUPPLIES A MERGE
-- ═══════════════════════════════════════════════════════════════════

/-- The algebra's own multiplication is a gauge-invariant merge, at every σ.

    (σ : ℝ)                                             [the parameter, anywhere]
    (d : Cell)                                          [the default point]
    {kvs₁ kvs₂ : List (K × Cell)}                       [two streams of key and point pairs]
    (hperm : kvs₁.Perm kvs₂)                            [one a rearrangement of the other]
    build (mul σ) d kvs₁ = build (mul σ) d kvs₂         [build the same table]

    So: this follows by instantiation, from `mul_comm` and `mul_assoc`, and is the whole of what
    the algebra says about tables. It is not a classification of merges by σ. That was
    claimed once and refuted, and `theory/DEAD_ENDS.md` carries the reason: a merge is a binary
    operation on an arbitrary type and a two-dimensional real algebra is a different kind of
    object. What σ's sign does fix is how many idempotents the algebra has
    (`idempotent_classification`), so how many independent Boolean channels this merge carries,
    which is a statement about this merge and not about merges. -/
theorem build_gauge_invariance_mul (σ : ℝ) (d : Trichotomy.Cell)
    {kvs₁ kvs₂ : List (K × Trichotomy.Cell)} (hperm : kvs₁.Perm kvs₂) :
    build (Trichotomy.mul σ) d kvs₁ = build (Trichotomy.mul σ) d kvs₂ :=
  build_gauge_invariance (Trichotomy.mul_comm σ) (Trichotomy.mul_assoc σ) d hperm

/-- **Reading a table built from affine steps at one key is that key's affine fold.**

    {K R : Type}                                 [a key type and a ring]
    [DecidableEq K] [CommRing R]                 [keys are comparable; the ring is commutative]
    (sigma : R)                                  [the parameter the steps compose under]
    (kvs : List (K × AnyRing.RingAffineStep R))  [a stream of key and affine-step pairs]
    (key : K)                                    [the key being read]
    get (build (ringAffineCompose sigma)         [reading the table built by composing steps]
      (ringAffineId R) kvs) key                  [starting from the identity step]
      = ringAffineFold sigma                     [is the fold of the composed steps]
        ((kvs.filter (fun kv => kv.1 = key))     [taken over exactly the entries]
          .map Prod.snd)                         [with the step, the second of each pair]
    `List`                                       [a finite sequence of things, in order]
    `×`                                          [paired with, so K × V means a key and a value]
    `build`                                      [fold the write over a stream of pairs]
    `Prod.snd`                                   [the second of each pair]
    `ringAffineCompose`                          [compose two affine steps: inner then outer]
    `ringAffineId`                               [the identity affine step]
    `ringAffineFold`                             [the left fold of a list of steps into one step]

    So: `get_build_acc` reads the built table as a fold of the merge over that key's values, and
    `ringAffineFold` is that same left fold started from `ringAffineId`, so the two sides are one
    list fold. -/
theorem get_build_ringAffineCompose {K R : Type} [DecidableEq K] [CommRing R]
    (sigma : R) (kvs : List (K × AnyRing.RingAffineStep R)) (key : K) :
    get (build (AnyRing.ringAffineCompose sigma) (AnyRing.ringAffineId R) kvs) key =
      AnyRing.ringAffineFold sigma
        ((kvs.filter (fun kv => kv.1 = key)).map Prod.snd) := by
  unfold build
  rw [get_build_acc (AnyRing.ringAffineCompose sigma) key kvs
    (fun _ => AnyRing.ringAffineId R)]
  rfl

end Table
end Apeiron
