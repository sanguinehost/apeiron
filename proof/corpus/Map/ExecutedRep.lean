-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe` (APEIRON-HT-READ): the executed map face's representation.
import Map.ExecutedOverflow
open Aeneas Aeneas.Std Result

/-!
# The executed map face's representation

`TableRep s m live` (key store `Stored<u64>`, the `word_hashtrinity_*` face) and `CoordRep s m live`
(key store `Regen<u64>`, the `word_coordmap_*` face) say that the extracted state
`s : hashtrinity.Table U64 I64 KS` represents the canonical map `m : Apeiron.Table.HMap U64 I64`
(`Apeiron/Table/Basic.lean:57`) on the live set `live`.  Both are `CoreRep` at their key store;
`TableRep` adds the stored key column.  A `u64` key is its own coordinate.

`CoreRep`'s clauses, field by field:

* geometry: `bits > 0`, `coords.len() = 2^bits`, `vals.len() = coords.len()`;
* main slots: a slot whose coordinate `c` is not reserved holds `some (m c)` and `c` is live; a
  non-reserved coordinate occupies at most one slot; and it sits at probe step `j < limN bits` of its
  own probe sequence with every earlier step's slot non-`EMPTY` (the capped placement);
* side store: its coordinates are reserved, live, distinct, each with value `some (m c)` and live
  flag `true`; the three vectors have one length;
* overflow (`OvInv`): absent, or a search tree (`Emb`, `BST`) with distinct arena indices, free
  list disjoint from the tree, and each node holding the single entry `(c, some (m c))` for a
  non-reserved coordinate `c` whose whole capped prefix holds neither `c` nor `EMPTY` (`Pass`) —
  the placement invariant of `src/hashtrinity/overflow.rs`;
* `len` is the number of non-reserved main coordinates;
* `live c` iff `c` is in the side store, a main slot, or the overflow tree.

`Pass`/`Desc` are stated against the extracted `descend`, so the probe sequence in the invariant is
the executed one.  The `3/4` load bound is not a clause: it is not an invariant of the executed body
(a resize re-places overflow entries into the primary array without a load check; see the report),
and no clause of the read depends on it.

* `with_bits_rep`, `new_rep`, `with_capacity_rep`, `coordinate_with_capacity_rep`: every constructor
  returns a representation of the constant-`0` map on the empty live set.
* `side_find_spec`: `SideStore::find` returns an entry with the query coordinate, or `none` when the
  store has none.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-- The `Stored<u64>` key store. -/
noncomputable abbrev SKS := hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore CI
  core.cmp.EqU64
/-- The `Regen<u64>` key store. -/
abbrev RKS := hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore
  U64.Insts.Apeiron_wordHashtrinityAlgebraCoordInv core.cmp.EqU64

theorem SKS_matches : ∀ keys i c k, (SKS).«matches» keys i c k = ok true :=
  fun keys i c k => stored_matches_ok keys i c k

theorem RKS_matches : ∀ keys i c k, (RKS).«matches» keys i c k = ok true :=
  fun keys i c k => regen_matches_ok keys i c k

section
variable {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)

/-- `c` is a side-store coordinate. -/
def InSide (s : Tbl KS) (c : U64) : Prop := ∃ i : Nat, s.side.coords.val[i]? = some c

/-- `c` occupies a main slot (and is not reserved). -/
def InMain (s : Tbl KS) (c : U64) : Prop := ¬ IsSent c ∧ ∃ i : Nat, s.coords.val[i]? = some c

/-- `c` is the coordinate of a node of the overflow tree. -/
def InOv (s : Tbl KS) (c : U64) : Prop :=
  ∃ o, s.overflow = some o ∧ ∃ T, Emb o.nodes.val o.root T ∧
    ∃ i ∈ idxs T, ∃ nd, o.nodes.val[i]? = some nd ∧ nd.key = c

/-- The overflow tree's invariant. -/
def OvInv (s : Tbl KS) (m : Apeiron.Table.HMap U64 I64) : Prop :=
  match s.overflow with
  | none => True
  | some o => ∃ T, Emb o.nodes.val o.root T ∧ BST o.nodes.val T ∧ (idxs T).Nodup ∧
      (∀ f ∈ o.free.val, f.val ∉ idxs T) ∧
      ∀ i ∈ idxs T, ∃ nd, o.nodes.val[i]? = some nd ∧
        nd.entries.val = [(nd.key, some (m nd.key))] ∧ ¬ IsSent nd.key ∧
        ∀ j : U32, j.val < limN s.bits.val → Pass ks s nd.key j

/-- Main-slot placement: `c` at slot `i` sits at a probe step below the cap whose predecessors are
all non-`EMPTY`. -/
def Placed (s : Tbl KS) (c : U64) (i : Nat) : Prop :=
  ∃ j : U32, j.val < limN s.bits.val ∧ ∃ d : Usize, Desc ks s c j d ∧ d.val = i ∧
    ∀ j' : U32, j'.val < j.val → ∃ d', Desc ks s c j' d' ∧
      ∃ x, s.coords.val[d'.val]? = some x ∧ x.val ≠ EMPTYN

/-- The number of non-reserved coordinates of the main array. -/
def mainCount (s : Tbl KS) : Nat := s.coords.val.countP (fun c => decide (¬ IsSent c))

/-- **The executed table represents `m` on `live`**, at key store `ks`. -/
structure CoreRep (s : Tbl KS) (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) : Prop where
  bits_pos : 0 < s.bits.val
  len_pow : s.coords.length = 2 ^ s.bits.val
  vals_len : s.vals.length = s.coords.length
  main_val : ∀ (i : Nat) (c : U64), s.coords.val[i]? = some c → ¬ IsSent c →
    live c ∧ s.vals.val[i]? = some (some (m c))
  main_uniq : ∀ (i j : Nat) (c : U64), s.coords.val[i]? = some c → s.coords.val[j]? = some c →
    ¬ IsSent c → i = j
  main_placed : ∀ (i : Nat) (c : U64), s.coords.val[i]? = some c → ¬ IsSent c → Placed ks s c i
  len_count : s.len.val = mainCount s
  side_len_vals : s.side.vals.length = s.side.coords.length
  side_val : ∀ (i : Nat) (c : U64), s.side.coords.val[i]? = some c →
    IsSent c ∧ live c ∧ s.side.vals.val[i]? = some (some (m c))
  side_uniq : ∀ (i j : Nat) (c : U64), s.side.coords.val[i]? = some c →
    s.side.coords.val[j]? = some c → i = j
  ov : OvInv ks s m
  live_iff : ∀ c, live c ↔ InSide s c ∨ InMain s c ∨ InOv s c

/-- `SideStore::find`: an entry with the query coordinate, or `none` when there is none. -/
theorem side_find_spec (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (ss : hashtrinity.side.SideStore U64 I64 KS) (c k : U64) (r : Option Usize)
    (h : hashtrinity.side.SideStore.find CI ks ss c k = ok r) :
    (∀ i, r = some i → ss.coords.val[i.val]? = some c) ∧
      (r = none → ∀ i : Nat, ss.coords.val[i]? ≠ some c) := by
  unfold hashtrinity.side.SideStore.find hashtrinity.side.SideStore.find_loop at h
  refine loop_pc (fun (i : Usize) => ∀ i' : Nat, i' < i.val → ss.coords.val[i']? ≠ some c)
    (fun r => (∀ i, r = some i → ss.coords.val[i.val]? = some c) ∧
      (r = none → ∀ i : Nat, ss.coords.val[i]? ≠ some c)) _ ?_ 0#usize
    (fun i' hi' => absurd hi' (by simp)) r h
  intro i hi cf hcf
  unfold hashtrinity.side.SideStore.find_loop.body at hcf
  by_cases hl : i < alloc.vec.Vec.len ss.coords
  · rw [if_pos hl] at hcf
    obtain ⟨x, hx, hcf⟩ := bind_inv hcf
    have hx' := vec_index_ok _ _ _ hx
    have hcont : ∀ i3 : Usize, i + 1#usize = ok i3 → x ≠ c →
        ∀ i' : Nat, i' < i3.val → ss.coords.val[i']? ≠ some c := by
      intro i3 hi3 hxc i' hi'
      have := (uadd_ok hi3).1
      have h1 : (1#usize).val = 1 := rfl
      rw [h1] at this
      by_cases hlt : i' < i.val
      · exact hi i' hlt
      · have : i' = i.val := by omega
        rw [this, hx']; intro hh; exact hxc (Option.some.inj hh)
    by_cases hxc : x = c
    · rw [if_pos hxc] at hcf
      obtain ⟨b, hb, hcf⟩ := bind_inv hcf
      rw [hm] at hb
      have hb' : b = true := (Result.ok_injective hb).symm
      subst hb'
      rw [if_pos rfl] at hcf
      have := (Result.ok_injective hcf).symm
      subst this
      refine ⟨fun i' hi' => ?_, fun hh => absurd hh (by simp)⟩
      have := Option.some.inj hi'
      subst this
      rw [hx', hxc]
    · rw [if_neg hxc] at hcf
      obtain ⟨i3, hi3, hcf⟩ := bind_inv hcf
      have := (Result.ok_injective hcf).symm
      subst this
      exact hcont i3 hi3 hxc
  · rw [if_neg hl] at hcf
    have := (Result.ok_injective hcf).symm
    subst this
    refine ⟨fun i' hi' => absurd hi' (by simp), fun _ i' => ?_⟩
    by_cases hlt : i' < i.val
    · exact hi i' hlt
    · have hl' : ¬ i.val < ss.coords.val.length := by
        intro hh; apply hl; rw [UScalar.lt_equiv]; simpa using hh
      rw [List.getElem?_eq_none (by omega)]
      simp

end

/-! ## Constructors -/

theorem alloc_uninit_val (T : Type) (n : Usize) (s : Slice (Option T))
    (h : hashtrinity.algebra.alloc_uninit_boxed T n = ok s) :
    s.val = List.replicate n.val none := by
  unfold hashtrinity.algebra.alloc_uninit_boxed at h
  obtain ⟨v1, hv1, h⟩ := bind_inv h
  unfold alloc.vec.Vec.into_boxed_slice at h
  have hs := (Result.ok_injective h).symm
  subst hs
  unfold alloc.vec.Vec.set_len at hv1
  have hcap : n.val ≤ (alloc.vec.Vec.with_capacity (core.mem.maybe_uninit.MaybeUninit T) n).cap.val := by
    simp [alloc.vec.Vec.with_capacity]
  rw [dif_pos hcap] at hv1
  have := (Result.ok_injective hv1).symm
  subst this
  have hu : (alloc.vec.UninitElem.uninit : Option T) = none := rfl
  simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, hu, Slice.new_val]

theorem from_elem_val (x : U64) (n : Usize) (v : alloc.vec.Vec U64)
    (h : alloc.vec.from_elem core.clone.CloneU64 x n = ok v) :
    v.val = List.replicate n.val x := by
  obtain ⟨v', h1, hP, _⟩ := WP.spec_imp_exists (alloc.vec.from_elem_spec core.clone.CloneU64 x n rfl)
  rw [h1] at h
  have := Result.ok_injective h
  subst this
  exact hP

theorem countP_replicate_empty (n : Nat) :
    (List.replicate n hashtrinity.EMPTY).countP (fun c => decide (¬ IsSent c)) = 0 := by
  rw [List.countP_eq_zero]
  intro a ha
  rw [List.mem_replicate] at ha
  rw [ha.2]
  simp [IsSent, EMPTY_val]

section
variable {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)

/-- The fresh table of `with_bits b` (`b > 0`) represents the constant-`0` map on the empty live
set. -/
theorem with_bits_rep (b : U32) (hb : 0 < b.val) (s : Tbl KS)
    (h : hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI ks b = ok s) :
    CoreRep ks s (fun _ => 0#i64) (fun _ => False) := by
  unfold hashtrinity.Table.with_bits at h
  obtain ⟨n, hn, h⟩ := bind_inv h
  obtain ⟨v, hv, h⟩ := bind_inv h
  obtain ⟨sl, hsl, h⟩ := bind_inv h
  obtain ⟨s1, hs1, h⟩ := bind_inv h
  obtain ⟨t, ht, h⟩ := bind_inv h
  obtain ⟨ss, hss, h⟩ := bind_inv h
  have := (Result.ok_injective h).symm
  subst this
  have hnv := (ushl_ok hn).2
  have h1 : (1#usize).val = 1 := rfl
  rw [h1, one_mul] at hnv
  have hvv := from_elem_val _ _ _ hv
  unfold alloc.vec.Vec.into_boxed_slice at hsl
  have hsl' := (Result.ok_injective hsl).symm
  subst hsl'
  have hs1v := alloc_uninit_val I64 n s1 hs1
  unfold hashtrinity.side.SideStore.empty at hss
  obtain ⟨t0, ht0, hss⟩ := bind_inv hss
  have := (Result.ok_injective hss).symm
  subst this
  have hcoords : (alloc.vec.Vec.val v) = List.replicate n.val hashtrinity.EMPTY := hvv
  have hlenn : n.val = 2 ^ b.val := by
    rw [hnv]
    apply Nat.mod_eq_of_lt
    have := (ushl_ok hn).1
    exact Nat.pow_lt_pow_right (by norm_num) this
  have hcv : ∀ (i : Nat) (c : U64), v.slice.val[i]? = some c → c = hashtrinity.EMPTY := by
    intro i c hc
    have : v.val[i]? = some c := hc
    rw [hcoords] at this
    exact (List.mem_replicate.mp (List.mem_of_getElem? this)).2
  have hcs : ∀ (i : Nat) (c : U64), v.slice.val[i]? = some c → IsSent c := by
    intro i c hc
    rw [hcv i c hc]
    left; exact EMPTY_val
  refine
    { bits_pos := hb
      len_pow := ?_
      vals_len := ?_
      main_val := fun i c hc hns => absurd (hcs i c hc) hns
      main_uniq := fun i _ c hc _ hns => absurd (hcs i c hc) hns
      main_placed := fun i c hc hns => absurd (hcs i c hc) hns
      len_count := ?_
      side_len_vals := rfl
      side_val := fun i c hc => by simp [alloc.vec.Vec.new] at hc
      side_uniq := fun i _ c hc => by simp [alloc.vec.Vec.new] at hc
      ov := trivial
      live_iff := ?_ }
  · show v.val.length = 2 ^ b.val
    rw [hcoords, List.length_replicate, hlenn]
  · show s1.val.length = v.val.length
    rw [hs1v, hcoords]; simp
  · show (0#usize).val = (v.val).countP _
    rw [hcoords, countP_replicate_empty]; rfl
  · intro c
    simp only [false_iff]
    rintro (⟨i, hi⟩ | ⟨hns, i, hi⟩ | ⟨o, ho, _⟩)
    · simp [alloc.vec.Vec.new] at hi
    · exact hns (hcs i c hi)
    · simp at ho

end

/-- **`TableRep`**: the `Stored<u64>` face's representation — `CoreRep` plus the key column
(`keys[i] = some c` at every non-reserved main slot `c`). -/
def TableRep (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) : Prop :=
  CoreRep SKS s m live ∧ s.keys.length = s.coords.length ∧
    ∀ (i : Nat) (c : U64), s.coords.val[i]? = some c → ¬ IsSent c → s.keys.val[i]? = some (some c)

/-- **`CoordRep`**: the `Regen<u64>` face's representation (no key column). -/
def CoordRep (s : Tbl (hashtrinity.algebra.Regen U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) : Prop :=
  CoreRep RKS s m live

theorem stored_with_bits_rep (b : U32) (hb : 0 < b.val) (s : Tbl (hashtrinity.algebra.Stored U64))
    (h : hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI SKS b = ok s) :
    TableRep s (fun _ => 0#i64) (fun _ => False) := by
  have hcore := with_bits_rep SKS b hb s h
  refine ⟨hcore, ?_, ?_⟩
  · unfold hashtrinity.Table.with_bits at h
    obtain ⟨n, hn, h⟩ := bind_inv h
    obtain ⟨v, hv, h⟩ := bind_inv h
    obtain ⟨sl, hsl, h⟩ := bind_inv h
    obtain ⟨s1, hs1, h⟩ := bind_inv h
    obtain ⟨t, ht, h⟩ := bind_inv h
    obtain ⟨ss, hss, h⟩ := bind_inv h
    have := (Result.ok_injective h).symm
    subst this
    have hvv := from_elem_val _ _ _ hv
    unfold alloc.vec.Vec.into_boxed_slice at hsl
    have hsl' := (Result.ok_injective hsl).symm
    subst hsl'
    change hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.alloc CI
      core.cmp.EqU64 n = ok t at ht
    unfold hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.alloc at ht
    obtain ⟨t1, ht1, ht⟩ := bind_inv ht
    have := (Result.ok_injective ht).symm
    subst this
    have := alloc_uninit_val U64 n t ht1
    show t.val.length = v.val.length
    rw [this, hvv]; simp
  · intro i c hc hns
    exact absurd hc (fun hh => hns (by
      unfold hashtrinity.Table.with_bits at h
      obtain ⟨n, hn, h⟩ := bind_inv h
      obtain ⟨v, hv, h⟩ := bind_inv h
      obtain ⟨sl, hsl, h⟩ := bind_inv h
      obtain ⟨s1, hs1, h⟩ := bind_inv h
      obtain ⟨t, ht, h⟩ := bind_inv h
      obtain ⟨ss, hss, h⟩ := bind_inv h
      have := (Result.ok_injective h).symm
      subst this
      have hvv := from_elem_val _ _ _ hv
      unfold alloc.vec.Vec.into_boxed_slice at hsl
      have hsl' := (Result.ok_injective hsl).symm
      subst hsl'
      have : v.val[i]? = some c := hh
      rw [hvv] at this
      rw [(List.mem_replicate.mp (List.mem_of_getElem? this)).2]
      left; exact EMPTY_val))

theorem bits_for_ge (cap : Usize) (b : U32) (h : hashtrinity.bits_for cap = ok b) : 3 ≤ b.val := by
  unfold hashtrinity.bits_for at h
  obtain ⟨_, _, h⟩ := bind_inv h
  obtain ⟨_, _, h⟩ := bind_inv h
  obtain ⟨_, _, h⟩ := bind_inv h
  obtain ⟨_, _, h⟩ := bind_inv h
  obtain ⟨_, _, h⟩ := bind_inv h
  obtain ⟨_, _, h⟩ := bind_inv h
  obtain ⟨_, _, h⟩ := bind_inv h
  obtain ⟨_, _, h⟩ := bind_inv h
  unfold core.cmp.Ord.max.default core.cmp.Ord.max_body at h
  simp [core.cmp.impls.PartialOrdU32.lt, liftFun2] at h
  split at h
  · have := Result.ok_injective h; subst this; omega
  · have := Result.ok_injective h; subst this; rfl

/-- `word_hashtrinity_new` returns a representation of the constant-`0` map on the empty live set. -/
theorem new_rep (s : Tbl (hashtrinity.algebra.Stored U64)) (h : word_hashtrinity_new = ok s) :
    TableRep s (fun _ => 0#i64) (fun _ => False) := by
  unfold word_hashtrinity_new hashtrinity.TableKVStored.new at h
  exact stored_with_bits_rep 3#u32 (by decide) s h

/-- `word_hashtrinity_with_capacity cap` returns a representation of the constant-`0` map on the
empty live set. -/
theorem with_capacity_rep (cap : Usize) (s : Tbl (hashtrinity.algebra.Stored U64))
    (h : word_hashtrinity_with_capacity cap = ok s) :
    TableRep s (fun _ => 0#i64) (fun _ => False) := by
  unfold word_hashtrinity_with_capacity hashtrinity.TableKVStored.with_capacity at h
  obtain ⟨b, hb, h⟩ := bind_inv h
  exact stored_with_bits_rep b (by have := bits_for_ge cap b hb; omega) s h

/-- `word_coordmap_coordinate_with_capacity cap` returns a representation of the constant-`0` map on
the empty live set. -/
theorem coordinate_with_capacity_rep (cap : Usize) (s : Tbl (hashtrinity.algebra.Regen U64))
    (h : word_coordmap_coordinate_with_capacity cap = ok s) :
    CoordRep s (fun _ => 0#i64) (fun _ => False) := by
  unfold word_coordmap_coordinate_with_capacity hashtrinity.TableKVRegen.coordinate_with_capacity
    at h
  obtain ⟨b, hb, h⟩ := bind_inv h
  exact with_bits_rep RKS b (by have := bits_for_ge cap b hb; omega) s h

end Apeiron.MapExecuted
