-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe` (APEIRON-HT-READ): insert on a reserved coordinate.
import Map.ExecutedRead
open Aeneas Aeneas.Std Result

/-!
# Insert on a reserved coordinate

`rep_side_update`: replacing only the side store of a represented table, with the map changed only at
reserved coordinates, preserves `CoreRep`.  The side-store branch of `insert_merge_last_write`
(`push` for a new coordinate, a value overwrite for an existing one) reduces to it.  The only fact
about `SideStore::find` used is `side_find_spec`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

section
variable {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)

theorem not_inOv_of_sent {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} (hov : OvInv ks s m)
    {c : U64} (hc : IsSent c) : ¬ InOv s c := by
  rintro ⟨o, ho, T, hT, i, hi, nd, hnd, hk⟩
  unfold OvInv at hov
  rw [ho] at hov
  obtain ⟨T', hT', -, -, -, hall⟩ := hov
  have := emb_unique hT hT'
  subst this
  obtain ⟨nd', hnd', -, hns, -⟩ := hall i hi
  rw [hnd] at hnd'
  have := Option.some.inj hnd'
  subst this
  exact hns (hk ▸ hc)

theorem rep_side_update {s : Tbl KS} {m m' : Apeiron.Table.HMap U64 I64} {live live' : U64 → Prop}
    (hR : CoreRep ks s m live) (ss : hashtrinity.side.SideStore U64 I64 KS)
    (hlv : ss.vals.length = ss.coords.length) (hll : ss.live.length = ss.coords.length)
    (hsv : ∀ (i : Nat) (c : U64), ss.coords.val[i]? = some c →
      IsSent c ∧ live' c ∧ ss.vals.val[i]? = some (some (m' c)) ∧ ss.live.val[i]? = some true)
    (hsu : ∀ (i j : Nat) (c : U64), ss.coords.val[i]? = some c → ss.coords.val[j]? = some c → i = j)
    (hlS : ∀ c, IsSent c → (live' c ↔ ∃ i : Nat, ss.coords.val[i]? = some c))
    (hm : ∀ c, ¬ IsSent c → m' c = m c) (hl : ∀ c, ¬ IsSent c → (live' c ↔ live c)) :
    CoreRep ks { s with side := ss } m' live' := by
  have hpl : ∀ (c : U64) (i : Nat), Placed ks s c i → Placed ks { s with side := ss } c i := by
    rintro c i ⟨j, hj, d, hd, hdi, hall⟩
    refine ⟨j, hj, d, desc_frame ks s _ c j d rfl rfl hd, hdi, fun j' hj' => ?_⟩
    obtain ⟨d', hd', x, hx, hxe⟩ := hall j' hj'
    exact ⟨d', desc_frame ks s _ c j' d' rfl rfl hd', x, hx, hxe⟩
  refine
    { bits_pos := hR.bits_pos, len_pow := hR.len_pow, vals_len := hR.vals_len,
      main_val := fun i c hi hns => ?_, main_uniq := hR.main_uniq,
      main_placed := fun i c hi hns => hpl c i (hR.main_placed i c hi hns),
      len_count := hR.len_count, side_len_vals := hlv, side_len_live := hll,
      side_val := hsv, side_uniq := hsu, ov := ?_, live_iff := fun c => ?_ }
  · obtain ⟨h1, h2⟩ := hR.main_val i c hi hns
    exact ⟨(hl c hns).mpr h1, by rw [h2, hm c hns]⟩
  · have hov := hR.ov
    unfold OvInv at hov ⊢
    show match s.overflow with | none => True | some _ => _
    revert hov
    cases s.overflow with
    | none => intro _; trivial
    | some o =>
      rintro ⟨T, hT, hB, hN, hF, hall⟩
      refine ⟨T, hT, hB, hN, hF, fun i hi => ?_⟩
      obtain ⟨nd, hnd, hent, hns, hP⟩ := hall i hi
      refine ⟨nd, hnd, by rw [hent, hm _ hns], hns, fun j hj => ?_⟩
      obtain ⟨i', hd, x, hx, hxc, hxe⟩ := hP j hj
      exact ⟨i', desc_frame ks s _ _ j i' rfl rfl hd, x, hx, hxc, hxe⟩
  · by_cases hc : IsSent c
    · rw [hlS c hc]
      constructor
      · intro h; exact Or.inl h
      · rintro (h | ⟨hh, _⟩ | h)
        · exact h
        · exact absurd hc hh
        · exact absurd h (not_inOv_of_sent ks hR.ov hc)
    · rw [hl c hc, hR.live_iff c]
      have hside : ¬ InSide s c := fun ⟨i, hi⟩ => hc (hR.side_val i c hi).1
      have hside' : ¬ InSide { s with side := ss } c := fun ⟨i, hi⟩ => hc (hsv i c hi).1
      constructor
      · rintro (h | h | h)
        · exact absurd h hside
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr h)
      · rintro (h | h | h)
        · exact absurd h hside'
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr h)

/-- `SideStore::push` appends the coordinate, the value and a live flag. -/
theorem push_spec (ss ss' : hashtrinity.side.SideStore U64 I64 KS) (c k : U64) (v : I64)
    (h : hashtrinity.side.SideStore.push CI ks ss c k v = ok ss') :
    ss'.coords.val = ss.coords.val ++ [c] ∧ ss'.vals.val = ss.vals.val ++ [some v] ∧
      ss'.live.val = ss.live.val ++ [true] := by
  unfold hashtrinity.side.SideStore.push at h
  dsimp only at h
  obtain ⟨need, -, h⟩ := bind_inv h
  by_cases hc : need > ss.cap
  · rw [if_pos hc] at h
    obtain ⟨i1, -, h⟩ := bind_inv h
    obtain ⟨i2, -, h⟩ := bind_inv h
    obtain ⟨nc, -, h⟩ := bind_inv h
    obtain ⟨g, -, h⟩ := bind_inv h
    obtain ⟨g1, -, h⟩ := bind_inv h
    obtain ⟨g2, -, h⟩ := bind_inv h
    obtain ⟨v1, hv1, h⟩ := bind_inv h
    obtain ⟨mu, hmu, h⟩ := bind_inv h
    obtain ⟨v2, hv2, h⟩ := bind_inv h
    obtain ⟨v3, hv3, h⟩ := bind_inv h
    have := (Result.ok_injective h).symm
    subst this
    have hmu' : mu = some v := by
      have := Result.ok_injective hmu; exact this.symm
    subst hmu'
    exact ⟨vec_push_ok _ _ _ hv1, vec_push_ok _ _ _ hv2, vec_push_ok _ _ _ hv3⟩
  · rw [if_neg hc] at h
    obtain ⟨g, -, h⟩ := bind_inv h
    obtain ⟨v1, hv1, h⟩ := bind_inv h
    obtain ⟨mu, hmu, h⟩ := bind_inv h
    obtain ⟨v2, hv2, h⟩ := bind_inv h
    obtain ⟨v3, hv3, h⟩ := bind_inv h
    have := (Result.ok_injective h).symm
    subst this
    have hmu' : mu = some v := by
      have := Result.ok_injective hmu; exact this.symm
    subst hmu'
    exact ⟨vec_push_ok _ _ _ hv1, vec_push_ok _ _ _ hv2, vec_push_ok _ _ _ hv3⟩

theorem append_single_get {α : Type} (l : List α) (x c : α) (i : Nat) (h : (l ++ [x])[i]? = some c) :
    (i < l.length ∧ l[i]? = some c) ∨ (i = l.length ∧ c = x) := by
  rw [List.getElem?_append] at h
  by_cases hi : i < l.length
  · rw [if_pos hi] at h; exact Or.inl ⟨hi, h⟩
  · rw [if_neg hi] at h
    right
    have : i - l.length = 0 := by
      by_contra hne
      have : 0 < i - l.length := Nat.pos_of_ne_zero hne
      obtain ⟨j, hj⟩ : ∃ j, i - l.length = j + 1 := ⟨i - l.length - 1, by omega⟩
      rw [hj] at h; simp at h
    rw [this] at h
    exact ⟨by omega, (Option.some.inj h).symm⟩

theorem vec_index_mut_ok {α : Type} (v : alloc.vec.Vec α) (i : Usize) (x : α) (back : α → alloc.vec.Vec α)
    (h : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice α) v i = ok (x, back)) :
    v.val[i.val]? = some x ∧ back = v.set i := by
  rw [alloc.vec.Vec.index_mut_slice_index] at h
  unfold alloc.vec.Vec.index_mut_usize at h
  obtain ⟨y, hy, h⟩ := bind_inv h
  have := Result.ok_injective h
  simp only [Prod.mk.injEq] at this
  obtain ⟨h1, h2⟩ := this
  subst h1
  refine ⟨?_, h2.symm⟩
  rw [← alloc.vec.Vec.index_slice_index] at hy
  exact vec_index_ok _ _ _ hy

/-- The side-store branch of `insert_merge_last_write`: a reserved key leaves everything but the side
store unchanged.  `SideStore::find` enters only through `side_find_spec`. -/
theorem insert_sent_rep (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) (k : U64) (v : I64) (hk : IsSent k) (s' : Tbl KS)
    (h : hashtrinity.Table.insert_merge_last_write core.cmp.EqU64 CI ks s k v = ok s') :
    ∃ ss, s' = { s with side := ss } ∧
      CoreRep ks s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) := by
  unfold hashtrinity.Table.insert_merge_last_write at h
  obtain ⟨⟨o, s1⟩, hic, h⟩ := bind_inv h
  unfold hashtrinity.Table.insert_core at hic
  obtain ⟨coord, hco, hic⟩ := bind_inv hic
  rw [coord_of_ok] at hco
  have hco' := Result.ok_injective hco
  subst hco'
  obtain ⟨b, hb, hic⟩ := bind_inv hic
  rw [is_sentinel_ok] at hb
  have hb' := (Result.ok_injective hb).symm
  subst hb'
  rw [if_pos (by simpa using hk)] at hic
  obtain ⟨f, hf, hic⟩ := bind_inv hic
  have hfs := side_find_spec ks hm s.side k k f hf
  rcases f with _ | i
  · have hnone := hfs.2 rfl
    simp only at hic
    obtain ⟨ss, hss, hic⟩ := bind_inv hic
    have hic' := Result.ok_injective hic
    simp only [Prod.mk.injEq] at hic'
    obtain ⟨rfl, rfl⟩ := hic'
    simp only at h
    have hs' := (Result.ok_injective h).symm
    subst hs'
    obtain ⟨pc, pv, pl⟩ := push_spec ks s.side ss k k v hss
    have hlv0 : s.side.vals.val.length = s.side.coords.val.length := hR.side_len_vals
    have hll0 : s.side.live.val.length = s.side.coords.val.length := hR.side_len_live
    refine ⟨ss, rfl, rep_side_update ks hR ss ?_ ?_ ?_ ?_ ?_ ?_ ?_⟩
    · show ss.vals.val.length = ss.coords.val.length
      rw [pv, pc]; simp [hlv0]
    · show ss.live.val.length = ss.coords.val.length
      rw [pl, pc]; simp [hll0]
    · intro i c hi
      rw [pc] at hi
      rcases append_single_get _ _ _ _ hi with ⟨hlt, hi0⟩ | ⟨hie, hck⟩
      · obtain ⟨a, b, c', d⟩ := hR.side_val i c hi0
        have hne : c ≠ k := fun hh => hnone i (hh ▸ hi0)
        refine ⟨a, Or.inl b, ?_, ?_⟩
        · rw [pv, List.getElem?_append_left (by omega), c']
          simp [Apeiron.Table.insertWith, hne]
        · rw [pl, List.getElem?_append_left (by omega), d]
      · rw [hck]
        refine ⟨hk, Or.inr rfl, ?_, ?_⟩
        · rw [pv, hie, ← hlv0]
          simp [Apeiron.Table.insertWith]
        · rw [pl, hie, ← hll0]; simp
    · intro i j c hi hj
      rw [pc] at hi hj
      rcases append_single_get _ _ _ _ hi with ⟨hlt, hi⟩ | ⟨hie, hck⟩
      · rcases append_single_get _ _ _ _ hj with ⟨hlt', hj⟩ | ⟨hje, hck'⟩
        · exact hR.side_uniq i j c hi hj
        · exact absurd (hck' ▸ hi) (hnone i)
      · rcases append_single_get _ _ _ _ hj with ⟨hlt', hj⟩ | ⟨hje, hck'⟩
        · exact absurd (hck ▸ hj) (hnone j)
        · omega
    · intro c hc
      rw [pc]
      constructor
      · rintro (hl | hck)
        · obtain ⟨i, hi⟩ := ((hR.live_iff c).mp hl).elim id (fun h => h.elim
            (fun ⟨hh, _⟩ => absurd hc hh) (fun h => absurd h (not_inOv_of_sent ks hR.ov hc)))
          exact ⟨i, by
            rw [List.getElem?_append_left ((List.getElem?_eq_some_iff.mp hi).1)]; exact hi⟩
        · rw [hck]; exact ⟨s.side.coords.val.length, by simp⟩
      · rintro ⟨i, hi⟩
        rcases append_single_get _ _ _ _ hi with ⟨hlt, hi⟩ | ⟨hie, hck⟩
        · exact Or.inl ((hR.live_iff c).mpr (Or.inl ⟨i, hi⟩))
        · exact Or.inr hck
    · intro c hc
      have : c ≠ k := fun h => hc (h ▸ hk)
      simp [Apeiron.Table.insertWith, this]
    · intro c hc
      have : c ≠ k := fun h => hc (h ▸ hk)
      simp [this]
  · have hik := hfs.1 i rfl
    simp only at hic
    obtain ⟨⟨sl, dback⟩, hl, hic⟩ := bind_inv hic
    have hl' := (Result.ok_injective hl).symm
    simp only [alloc.vec.Vec.deref_mut, Prod.mk.injEq] at hl'
    obtain ⟨rfl, rfl⟩ := hl'
    obtain ⟨⟨mu, gback⟩, hg, hic⟩ := bind_inv hic
    obtain ⟨hg1, hg2⟩ := get_unchecked_mut_ok _ _ _ _ hg
    obtain ⟨old, hold, hic⟩ := bind_inv hic
    have hmu := assume_init_read_ok _ _ hold
    obtain ⟨⟨b0, ibk⟩, hib, hic⟩ := bind_inv hic
    obtain ⟨hib1, hib2⟩ := vec_index_mut_ok _ _ _ _ hib
    have hic' := Result.ok_injective hic
    simp only [Prod.mk.injEq] at hic'
    obtain ⟨rfl, rfl⟩ := hic'
    obtain ⟨-, -, h⟩ := bind_inv h
    dsimp only at h
    obtain ⟨mu2, hmu2, h⟩ := bind_inv h
    obtain ⟨⟨sl2, dback2⟩, hl2, h⟩ := bind_inv h
    obtain ⟨⟨mu3, gback2⟩, hg3, h⟩ := bind_inv h
    obtain ⟨⟨b3, ibk2⟩, hib3, h⟩ := bind_inv h
    have hs' := (Result.ok_injective h).symm
    have hmu2' : mu2 = some v := (Result.ok_injective hmu2).symm
    simp only [lift, alloc.vec.Vec.deref_mut] at hl2
    have hl2' := (Result.ok_injective hl2)
    simp only [Prod.mk.injEq] at hl2'
    obtain ⟨rfl, rfl⟩ := hl2'
    obtain ⟨hg31, hg32⟩ := get_unchecked_mut_ok _ _ _ _ hg3
    obtain ⟨-, hib32⟩ := vec_index_mut_ok _ _ _ _ hib3
    subst hs'
    refine ⟨{ s.side with vals := alloc.vec.Vec.ofSlice (gback2 mu2), live := ibk2 true }, rfl, ?_⟩
    have hv : (alloc.vec.Vec.ofSlice (gback2 mu2)).val = s.side.vals.val.set i.val (some v) := by
      rw [hg32, hmu2', hg2]
      show ((s.side.vals.slice.set i mu).set i (some v)).val = _
      rw [Slice.set_val_eq, Slice.set_val_eq, List.set_set]
      rfl
    have hlv : (ibk2 true).val = s.side.live.val.set i.val true := by
      rw [hib32, hib2]
      simp only [alloc.vec.Vec.set_val_eq, List.set_set]
    have hlv0 : s.side.vals.val.length = s.side.coords.val.length := hR.side_len_vals
    have hll0 : s.side.live.val.length = s.side.coords.val.length := hR.side_len_live
    have hilt : i.val < s.side.coords.val.length := (List.getElem?_eq_some_iff.mp hik).1
    refine rep_side_update ks hR _ ?_ ?_ ?_ ?_ ?_ ?_ ?_
    · show (alloc.vec.Vec.ofSlice (gback2 mu2)).val.length = s.side.coords.val.length
      rw [hv]; simp [hlv0]
    · show (ibk2 true).val.length = s.side.coords.val.length
      rw [hlv]; simp [hll0]
    · intro j c hj
      have hjc : s.side.coords.val[j]? = some c := hj
      obtain ⟨a, b, c', d⟩ := hR.side_val j c hjc
      by_cases hji : j = i.val
      · subst hji
        have hck : c = k := by rw [hik] at hjc; exact (Option.some.inj hjc).symm
        subst hck
        refine ⟨a, Or.inr rfl, ?_, ?_⟩
        · show (alloc.vec.Vec.ofSlice (gback2 mu2)).val[i.val]? = _
          rw [hv]; simp [Apeiron.Table.insertWith, hlv0, hilt]
        · show (ibk2 true).val[i.val]? = _
          rw [hlv]; simp [hll0, hilt]
      · have hne : c ≠ k := by
          rintro rfl
          exact hji (hR.side_uniq j i.val c hjc hik)
        refine ⟨a, Or.inl b, ?_, ?_⟩
        · show (alloc.vec.Vec.ofSlice (gback2 mu2)).val[j]? = _
          rw [hv, List.getElem?_set_ne (Ne.symm hji), c']
          simp [Apeiron.Table.insertWith, hne]
        · show (ibk2 true).val[j]? = _
          rw [hlv, List.getElem?_set_ne (Ne.symm hji), d]
    · exact hR.side_uniq
    · intro c hc
      constructor
      · rintro (hl | hck)
        · exact ((hR.live_iff c).mp hl).elim id (fun h => h.elim
            (fun ⟨hh, _⟩ => absurd hc hh) (fun h => absurd h (not_inOv_of_sent ks hR.ov hc)))
        · exact ⟨i.val, by rw [hck]; exact hik⟩
      · intro hex
        exact Or.inl ((hR.live_iff c).mpr (Or.inl hex))
    · intro c hc
      have : c ≠ k := fun h => hc (h ▸ hk)
      simp [Apeiron.Table.insertWith, this]
    · intro c hc
      have : c ≠ k := fun h => hc (h ▸ hk)
      simp [this]

end

/-- **`insert_side_refines` (`Stored<u64>`).**  `word_hashtrinity_insert` on a reserved key (`EMPTY` or
`TOMB`) preserves `TableRep` and updates the canonical map by `insertWith (fun new _ => new)`. -/
theorem insert_side_refines (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) (hR : TableRep s m live) (k : U64) (v : I64) (hk : IsSent k)
    (s' : Tbl (hashtrinity.algebra.Stored U64)) (h : word_hashtrinity_insert s k v = ok s') :
    TableRep s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) := by
  obtain ⟨ss, hs, hc⟩ := insert_sent_rep SKS SKS_matches hR.1 k v hk s' h
  subst hs
  exact ⟨hc, hR.2.1, hR.2.2⟩

/-- **`coord_insert_side_refines` (`Regen<u64>`).**  The same for `word_coordmap_insert`. -/
theorem coord_insert_side_refines (s : Tbl (hashtrinity.algebra.Regen U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) (hR : CoordRep s m live) (k : U64) (v : I64) (hk : IsSent k)
    (s' : Tbl (hashtrinity.algebra.Regen U64)) (h : word_coordmap_insert s k v = ok s') :
    CoordRep s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) := by
  obtain ⟨ss, hs, hc⟩ := insert_sent_rep RKS RKS_matches hR k v hk s' h
  subst hs
  exact hc

end Apeiron.MapExecuted
