-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe`: `resize` preserves the representation.
import Map.ExecutedInsert
import Map.ExecutedResizeOrder
open Aeneas Aeneas.Std Result

/-!
# `resize` preserves the representation

`Table::resize(new_bits)` (`src/hashtrinity/mod.rs`) builds a fresh table (`with_bits`), re-places
every live main slot through `place_capped` (`resize_loop0`), moves the side store over, and re-places
every overflow entry (`into_entries`, then `resize_loop1`).

* `place_rep`: `place_capped` of a non-reserved coordinate `c` that is not live, with key `c` and
  value `m c`, represents `m` on `live ∪ {c}`; the arena plus the main count grows by at most one.
* `resize_rep`: at a key store whose key column `KC` is set by `KeyStore::set`, read by
  `KeyStore::take_key` and established by `with_bits`, `resize` to a positive bit count preserves
  `RepW` and `KC`, and does not raise the arena plus the main count (`ResizeOK`).
* `resizeOK_stored`, `resizeOK_regen`: `ResizeOK` at `Stored<u64>` and `Regen<u64>`.

No load bound is used: `place_capped` falls through to the overflow whenever the capped prefix is
full.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

theorem slice_update_ok {α : Type} (sl sl' : Slice α) (i : Usize) (x : α)
    (h : Slice.update sl i x = ok sl') : sl'.val = sl.val.set i.val x := by
  unfold Slice.update at h
  split at h
  · simp at h
  · have := (Result.ok_injective h).symm
    subst this
    simp

theorem upd_same (c : U64) (m : Apeiron.Table.HMap U64 I64) : upd c (m c) m = m := by
  funext x
  by_cases hx : x = c
  · subst hx; exact upd_self _ _ _
  · exact upd_ne _ _ hx

section
variable {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)

/-- `CoreRep` reads the map only at live coordinates. -/
theorem rep_map_congr {s : Tbl KS} {m m' : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) (hm : ∀ c, live c → m' c = m c) : CoreRep ks s m' live := by
  refine
    { bits_pos := hR.bits_pos, len_pow := hR.len_pow, vals_len := hR.vals_len,
      main_val := fun i c hc hcns => ?_, main_uniq := hR.main_uniq,
      main_placed := hR.main_placed, len_count := hR.len_count,
      side_len_vals := hR.side_len_vals, side_val := fun i c hc => ?_,
      side_uniq := hR.side_uniq,
      ov := ovInv_transfer ks hR.ov rfl rfl rfl (fun _ _ h _ => h)
        (fun c hc => hm c ((hR.live_iff c).mpr (Or.inr (Or.inr hc)))),
      live_iff := hR.live_iff }
  · obtain ⟨hl, hv⟩ := hR.main_val i c hc hcns
    exact ⟨hl, by rw [hv, hm c hl]⟩
  · obtain ⟨a, hl, hv⟩ := hR.side_val i c hc
    exact ⟨a, hl, by rw [hv, hm c hl]⟩

theorem with_bits_overflow (b : U32) (t : Tbl KS)
    (h : hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI ks b = ok t) : t.overflow = none := by
  unfold hashtrinity.Table.with_bits at h
  obtain ⟨_, _, h⟩ := bind_inv h
  obtain ⟨_, _, h⟩ := bind_inv h
  obtain ⟨_, _, h⟩ := bind_inv h
  obtain ⟨_, _, h⟩ := bind_inv h
  obtain ⟨_, _, h⟩ := bind_inv h
  obtain ⟨_, _, h⟩ := bind_inv h
  have := (Result.ok_injective h).symm
  subst this
  rfl

/-- **`place_capped` of a fresh coordinate.** -/
theorem place_rep (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (KC : Slice U64 → KS → Prop)
    (hKC : ∀ (coords coords' : Slice U64) (keys keys' : KS) (i : Usize) (k : U64),
      KC coords keys → coords'.val = coords.val.set i.val k → ks.set keys i k = ok keys' →
      KC coords' keys')
    {t : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks t m live) (hK : KC t.coords t.keys) (hcap : arenaLen t + mainCount t < nilN)
    {c : U64} (hns : ¬ IsSent c) (hnl : ¬ live c) {t' : Tbl KS}
    (h : hashtrinity.Table.place_capped core.cmp.EqU64 CI ks t c c (m c) = ok t') :
    RepW ks t' m (fun x => live x ∨ x = c) ∧ KC t'.coords t'.keys ∧
      arenaLen t' + mainCount t' ≤ arenaLen t + mainCount t + 1 := by
  unfold hashtrinity.Table.place_capped at h
  obtain ⟨r, hr, h⟩ := bind_inv h
  have hcok := pck_spec ks hm (fun _ => avx2_spec_holds) t c c hR.1.bits_pos hR.1.len_pow r hr
  have hupd := upd_same c m
  rcases r with i | i | _
  · exact absurd (hR.1.main_val i.val c hcok hns).1 hnl
  · obtain ⟨⟨x1, back⟩, hg1, h⟩ := bind_inv h
    obtain ⟨-, hback⟩ := get_unchecked_mut_ok _ _ _ _ hg1
    obtain ⟨mu, hmu, h⟩ := bind_inv h
    obtain ⟨⟨x2, back1⟩, hg2, h⟩ := bind_inv h
    obtain ⟨-, hback1⟩ := get_unchecked_mut_ok _ _ _ _ hg2
    obtain ⟨keys', hset, h⟩ := bind_inv h
    obtain ⟨len', hlen', h⟩ := bind_inv h
    have := (Result.ok_injective h).symm
    subst this
    have hmu' : mu = some (m c) := (Result.ok_injective hmu).symm
    have hco : (back c).val = t.coords.val.set i.val c := by rw [hback, Slice.set_val_eq]
    have hv : (back1 mu).val = t.vals.val.set i.val (some (m c)) := by
      rw [hback1, hmu', Slice.set_val_eq]
    have hl : len'.val = t.len.val + 1 := by
      have := (uadd_ok hlen').1
      rw [this]; rfl
    have hC := rep_fill ks hR.1 hns i hcok (m c) _ _ keys' len' hco hv hl
    rw [hupd] at hC
    refine ⟨⟨hC, hR.2⟩, hKC t.coords (back c) t.keys keys' i c hK hco hset, ?_⟩
    have h1 : len'.val = mainCount _ := hC.len_count
    have h0 : t.len.val = mainCount t := hR.1.len_count
    show arenaLen t + _ ≤ _
    omega
  · obtain ⟨oo, hoo, h⟩ := bind_inv h
    obtain ⟨⟨o1, asb⟩, has, h⟩ := bind_inv h
    unfold core.option.Option.as_mut at has
    have has' := Result.ok_injective has
    simp only [Prod.mk.injEq] at has'
    obtain ⟨rfl, rfl⟩ := has'
    have hov : ∃ ov T, oo = some ov ∧ OvFull ks t m ov T ∧ (∀ c, InT ov T c ↔ InOv t c) ∧
        ov.nodes.val.length ≤ arenaLen t := by
      cases hto : t.overflow with
      | none =>
        rw [if_pos (by simp [hto])] at hoo
        obtain ⟨o2, ho2, hoo⟩ := bind_inv hoo
        have := (Result.ok_injective hoo).symm
        subst this
        obtain ⟨hF, hnone, hl0⟩ := ov_full_new ks t m o2 ho2
        refine ⟨o2, .leaf, rfl, hF, fun c => ⟨fun h => absurd h (hnone c), ?_⟩, by omega⟩
        rintro ⟨o', ho', _⟩
        rw [hto] at ho'
        exact absurd ho' (by simp)
      | some ov =>
        rw [if_neg (by simp [hto])] at hoo
        have := (Result.ok_injective hoo).symm
        rw [hto] at this
        subst this
        obtain ⟨T, hF, hiff⟩ := ov_full_of_some ks hR hto
        refine ⟨ov, T, rfl, hF, hiff, ?_⟩
        unfold arenaLen; rw [hto]
    obtain ⟨ov, T, rfl, hF, hiff, hlen⟩ := hov
    obtain ⟨⟨nd, e⟩, ov1, hins, h⟩ : ∃ p ov1, hashtrinity.overflow.Overflow.insert_new_at CI ov c c
        (m c) = ok (p, ov1) ∧ ok { t with overflow := some ov1 } = ok t' := by
      obtain ⟨⟨p, ov1⟩, hins, h⟩ := bind_inv h
      exact ⟨p, ov1, hins, h⟩
    have := (Result.ok_injective h).symm
    subst this
    have habs : ¬ Present ov.nodes.val T c c := by
      rintro ⟨j, hj, nd', hnd', hk, _⟩
      exact hnl ((hR.1.live_iff c).mpr (Or.inr (Or.inr ((hiff c).mp ⟨j, hj, nd', hnd', hk⟩))))
    obtain ⟨T', hF', hT', _, hlenA⟩ :=
      blocked_new_full_abs ks ov T hF hns hcok (by omega) habs (m c) nd e ov1 hins
    rw [hupd] at hF'
    have hnm := blocked_not_main ks hR.1 hcok
    refine ⟨rep_set_ov ks hR.1 ov1 T' hF' (fun _ _ => rfl) (fun _ _ => rfl) (fun x => ?_), hK, ?_⟩
    · rw [hT', hiff, hR.1.live_iff x]
      constructor
      · rintro ((h | h | h) | h)
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr (Or.inl h))
        · exact Or.inr (Or.inr (Or.inr h))
      · rintro (h | h | h | h)
        · exact Or.inl (Or.inl h)
        · exact Or.inl (Or.inr (Or.inl h))
        · exact Or.inl (Or.inr (Or.inr h))
        · exact Or.inr h
    · show ov1.nodes.val.length + mainCount t ≤ _
      omega

theorem list_set_self {α : Type} (l : List α) (i : Nat) (a : α) (h : l[i]? = some a) :
    l.set i a = l := by
  apply List.ext_getElem?
  intro j
  by_cases hj : j = i
  · subst hj; rw [h, List.getElem?_set_self (List.getElem?_eq_some_iff.mp h).1]
  · rw [List.getElem?_set_ne (Ne.symm hj)]

theorem countP_take_succ {α : Type} (p : α → Bool) (l : List α) (i : Nat) (a : α)
    (h : l[i]? = some a) :
    (l.take (i + 1)).countP p = (l.take i).countP p + (if p a then 1 else 0) := by
  rw [List.take_add_one, List.countP_append, h]
  simp [List.countP_cons]

theorem countP_take_mono {α : Type} (p : α → Bool) (l : List α) (i : Nat) :
    (l.take i).countP p ≤ (l.take (i + 1)).countP p :=
  (List.take_sublist_take_left (by omega)).countP_le

theorem repW_live_congr {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {L L' : U64 → Prop}
    (h : RepW ks s m L) (hL : ∀ c, L c ↔ L' c) : RepW ks s m L' := by
  have : L = L' := funext fun c => propext (hL c)
  subst this; exact h

/-- The non-reserved main coordinates of `s` below slot `n`. -/
def LowMain (s : Tbl KS) (n : Nat) (c : U64) : Prop :=
  ¬ IsSent c ∧ ∃ d < n, s.coords.val[d]? = some c

theorem mainCount_le (s : Tbl KS) (n : Nat) :
    (s.coords.val.take n).countP (fun c => decide (¬ IsSent c)) ≤ mainCount s :=
  (List.take_sublist n _).countP_le

/-- `resize_loop0`: every live main slot of `s`, in slot order, re-placed into the fresh table. -/
theorem resize_loop0_spec (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (KC : Slice U64 → KS → Prop)
    (hKC : ∀ (coords coords' : Slice U64) (keys keys' : KS) (i : Usize) (k : U64),
      KC coords keys → coords'.val = coords.val.set i.val k → ks.set keys i k = ok keys' →
      KC coords' keys')
    (hTK : ∀ (coords coords' : Slice U64) (keys : KS) (i : Usize) (c key : U64) (keys' : KS),
      KC coords keys → coords.val[i.val]? = some c → ¬ IsSent c →
      ks.take_key keys i c = ok (key, keys') → coords'.val = coords.val.set i.val hashtrinity.EMPTY →
      key = c ∧ KC coords' keys')
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys) (hsmall : arenaLen s + mainCount s < nilN)
    (b0 : Tbl KS) (hb0 : RepW ks b0 m (LowMain s 0)) (hKb0 : KC b0.coords b0.keys)
    (hcb0 : arenaLen b0 + mainCount b0 = 0)
    (r : hashtrinity.side.SideStore U64 I64 KS × Option Ov × Tbl KS)
    (h : hashtrinity.Table.resize_loop0 core.cmp.EqU64 CI ks s b0 0#usize = ok r) :
    r.1 = s.side ∧ r.2.1 = s.overflow ∧ RepW ks r.2.2 m (InMain s) ∧
      KC r.2.2.coords r.2.2.keys ∧ arenaLen r.2.2 + mainCount r.2.2 ≤ mainCount s := by
  unfold hashtrinity.Table.resize_loop0 at h
  refine loop_pc (fun (p : Tbl KS × Tbl KS × Usize) =>
      p.1.coords.val.length = s.coords.val.length ∧
      (∀ d, p.2.2.val ≤ d → p.1.coords.val[d]? = s.coords.val[d]?) ∧
      p.1.vals.val = s.vals.val ∧ KC p.1.coords p.1.keys ∧
      p.1.side = s.side ∧ p.1.overflow = s.overflow ∧
      RepW ks p.2.1 m (LowMain s p.2.2.val) ∧ KC p.2.1.coords p.2.1.keys ∧
      arenaLen p.2.1 + mainCount p.2.1 ≤
        (s.coords.val.take p.2.2.val).countP (fun c => decide (¬ IsSent c)))
    (fun r => r.1 = s.side ∧ r.2.1 = s.overflow ∧ RepW ks r.2.2 m (InMain s) ∧
      KC r.2.2.coords r.2.2.keys ∧ arenaLen r.2.2 + mainCount r.2.2 ≤ mainCount s) _ ?_ _
    ⟨rfl, fun _ _ => rfl, rfl, hK, rfl, rfl, hb0, hKb0, by
      show arenaLen b0 + mainCount b0 ≤ (s.coords.val.take 0).countP _
      rw [hcb0]; exact Nat.zero_le _⟩ r h
  rintro ⟨t, b, i⟩ ⟨hA, hB, hC, hD, hE1, hE2, hF, hG, hH⟩ cf hcf
  dsimp only at hcf hA hB hC hD hE1 hE2 hF hG hH
  unfold hashtrinity.Table.resize_loop0.body at hcf
  have hLlt : (Slice.len t.coords).val = s.coords.val.length := by rw [Slice.len_val]; exact hA
  by_cases hlt : i < Slice.len t.coords
  · rw [if_pos hlt] at hcf
    have hilt : i.val < s.coords.val.length := by rw [UScalar.lt_equiv] at hlt; omega
    obtain ⟨c, hc, hcf⟩ := bind_inv hcf
    have hc' : s.coords.val[i.val]? = some c := by
      rw [← hB i.val le_rfl]; exact slice_index_ok _ _ _ hc
    obtain ⟨bl, hbl, hcf⟩ := bind_inv hcf
    rw [is_live_ok] at hbl
    have hbl' := (Result.ok_injective hbl).symm
    subst hbl'
    by_cases hns : IsSent c
    · rw [if_neg (by simpa using hns)] at hcf
      obtain ⟨i2, hi2, hcf⟩ := bind_inv hcf
      have := (Result.ok_injective hcf).symm
      subst this
      dsimp only
      have hi2v : i2.val = i.val + 1 := by have := (uadd_ok hi2).1; rw [this]; rfl
      refine ⟨hA, fun d hd => hB d (by omega), hC, hD, hE1, hE2, ?_, hG, ?_⟩
      · refine repW_live_congr ks hF (fun x => ⟨fun ⟨hx, d, hd, hdx⟩ => ⟨hx, d, by omega, hdx⟩,
          fun ⟨hx, d, hd, hdx⟩ => ⟨hx, d, ?_, hdx⟩⟩)
        by_contra hge
        have hdi : d = i.val := by omega
        subst hdi
        rw [hc'] at hdx
        exact hx ((Option.some.inj hdx) ▸ hns)
      · rw [hi2v]; exact le_trans hH (countP_take_mono _ _ _)
    · rw [if_pos (by simpa using hns)] at hcf
      obtain ⟨⟨key, keys'⟩, htk, hcf⟩ := bind_inv hcf
      obtain ⟨⟨mu, back⟩, hg, hcf⟩ := bind_inv hcf
      obtain ⟨hmu, hback⟩ := get_unchecked_mut_ok _ _ _ _ hg
      obtain ⟨v, hv, hcf⟩ := bind_inv hcf
      obtain ⟨cs, hcs, hcf⟩ := bind_inv hcf
      have hcsv := slice_update_ok _ _ _ _ hcs
      obtain ⟨b1, hpl, hcf⟩ := bind_inv hcf
      obtain ⟨i2, hi2, hcf⟩ := bind_inv hcf
      have := (Result.ok_injective hcf).symm
      subst this
      dsimp only
      have hi2v : i2.val = i.val + 1 := by have := (uadd_ok hi2).1; rw [this]; rfl
      have htc : t.coords.val[i.val]? = some c := by rw [hB i.val le_rfl]; exact hc'
      obtain ⟨rfl, hKt⟩ := hTK t.coords cs t.keys i c key keys' hD htc hns htk hcsv
      obtain ⟨_, hval⟩ := hR.1.main_val i.val key hc' hns
      have hmu' : mu = some (m key) := by
        have : t.vals.val[i.val]? = some mu := hmu
        rw [hC, hval] at this
        exact (Option.some.inj this).symm
      have hvv : v = m key := by
        rw [hmu'] at hv
        have := assume_init_read_ok _ _ hv
        exact (Option.some.inj this).symm
      subst hvv
      have hnl : ¬ LowMain s i.val key := by
        rintro ⟨_, d, hd, hdx⟩
        have := hR.1.main_uniq d i.val key hdx hc' hns
        omega
      have hcnt : (s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)) + 1 ≤
          mainCount s := by
        have := countP_take_succ (fun c => decide (¬ IsSent c)) s.coords.val i.val key hc'
        have h2 := mainCount_le s (i.val + 1)
        simp only [hns, decide_true, if_true, not_false_eq_true] at this
        omega
      obtain ⟨hRb1, hKb1, hcb1⟩ := place_rep ks hm KC hKC hF hG (by omega) hns hnl hpl
      refine ⟨?_, fun d hd => ?_, ?_, hKt, hE1, hE2, ?_, hKb1, ?_⟩
      · show cs.val.length = _
        rw [hcsv, List.length_set]; exact hA
      · show cs.val[d]? = _
        rw [hcsv, List.getElem?_set_ne (by omega)]
        exact hB d (by omega)
      · show (back mu).val = _
        rw [hback, Slice.set_val_eq, list_set_self _ _ _ hmu]; exact hC
      · rw [hi2v]
        refine repW_live_congr ks hRb1 (fun x => ⟨?_, ?_⟩)
        · rintro (⟨hx, d, hd, hdx⟩ | rfl)
          · exact ⟨hx, d, by omega, hdx⟩
          · exact ⟨hns, i.val, by omega, hc'⟩
        · rintro ⟨hx, d, hd, hdx⟩
          by_cases hdi : d = i.val
          · subst hdi; rw [hc'] at hdx; exact Or.inr (Option.some.inj hdx).symm
          · exact Or.inl ⟨hx, d, by omega, hdx⟩
      · rw [hi2v, countP_take_succ _ _ _ _ hc']
        simp only [hns, decide_true, if_true, not_false_eq_true]
        omega
  · rw [if_neg hlt] at hcf
    have := (Result.ok_injective hcf).symm
    subst this
    dsimp only
    have hge : s.coords.val.length ≤ i.val := by rw [UScalar.lt_equiv] at hlt; omega
    refine ⟨hE1, hE2, repW_live_congr ks hF (fun x => ⟨fun ⟨hx, d, _, hdx⟩ => ⟨hx, d, hdx⟩,
      fun ⟨hx, d, hdx⟩ => ⟨hx, d, ?_, hdx⟩⟩), hG, ?_⟩
    · exact lt_of_lt_of_le (List.getElem?_eq_some_iff.mp hdx).1 hge
    · rw [List.take_of_length_le hge] at hH; exact hH

/-- `resize_loop1`: the overflow entries, re-placed in order. -/
theorem resize_loop1_spec (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (KC : Slice U64 → KS → Prop)
    (hKC : ∀ (coords coords' : Slice U64) (keys keys' : KS) (i : Usize) (k : U64),
      KC coords keys → coords'.val = coords.val.set i.val k → ks.set keys i k = ok keys' →
      KC coords' keys')
    {m : Apeiron.Table.HMap U64 I64} (es : List (U64 × U64 × I64))
    (hes : ∀ x ∈ es, x.2.1 = x.1 ∧ x.2.2 = m x.1 ∧ ¬ IsSent x.1)
    (hnd : (es.map Prod.fst).Nodup) {L0 : U64 → Prop} (t0 : Tbl KS) (hR0 : RepW ks t0 m L0)
    (hK0 : KC t0.coords t0.keys) (hdisj : ∀ x ∈ es, ¬ L0 x.1)
    (hc0 : arenaLen t0 + mainCount t0 + es.length < nilN)
    (iter : alloc.vec.into_iter.IntoIter (U64 × U64 × I64)) (hiter : iter.val = es) (r : Tbl KS)
    (h : hashtrinity.Table.resize_loop1 core.cmp.EqU64 CI ks iter t0 = ok r) :
    RepW ks r m (fun c => L0 c ∨ c ∈ es.map Prod.fst) ∧ KC r.coords r.keys ∧
      arenaLen r + mainCount r ≤ arenaLen t0 + mainCount t0 + es.length := by
  unfold hashtrinity.Table.resize_loop1 at h
  refine loop_pc (fun (p : alloc.vec.into_iter.IntoIter (U64 × U64 × I64) × Tbl KS) =>
      ∃ dn, dn ++ p.1.val = es ∧ RepW ks p.2 m (fun c => L0 c ∨ c ∈ dn.map Prod.fst) ∧
        KC p.2.coords p.2.keys ∧ arenaLen p.2 + mainCount p.2 ≤ arenaLen t0 + mainCount t0 + dn.length)
    (fun r => RepW ks r m (fun c => L0 c ∨ c ∈ es.map Prod.fst) ∧ KC r.coords r.keys ∧
      arenaLen r + mainCount r ≤ arenaLen t0 + mainCount t0 + es.length) _ ?_ _
    ⟨[], by simpa using hiter, repW_live_congr ks hR0 (fun c => by simp), hK0, by simp⟩ r h
  rintro ⟨it, t⟩ ⟨dn, hdn, hR, hK, hc⟩ cf hcf
  dsimp only at hcf hdn hR hK hc
  unfold hashtrinity.Table.resize_loop1.body at hcf
  obtain ⟨⟨x, it1⟩, hn, hcf⟩ := bind_inv hcf
  rcases next_spec _ _ _ hn with ⟨hnil, rfl⟩ | ⟨y, hcons, rfl⟩
  · have := (Result.ok_injective hcf).symm
    subst this
    rw [hnil, List.append_nil] at hdn
    subst hdn
    exact ⟨hR, hK, hc⟩
  · obtain ⟨c, k, v⟩ := y
    obtain ⟨t1, hpl, hcf⟩ := bind_inv hcf
    have := (Result.ok_injective hcf).symm
    subst this
    dsimp only
    rw [hcons] at hdn
    have hmem : (c, k, v) ∈ es := by rw [← hdn]; simp
    obtain ⟨hk, hv, hns⟩ := hes _ hmem
    dsimp only at hk hv hns
    subst k v
    have hnin : c ∉ dn.map Prod.fst := by
      intro hc'
      rw [← hdn, List.map_append, List.nodup_append] at hnd
      exact hnd.2.2 c hc' c (by simp) rfl
    have hlen : dn.length + 1 ≤ es.length := by rw [← hdn]; simp
    obtain ⟨hR1, hK1, hc1⟩ := place_rep ks hm KC hKC hR hK (by omega) hns
      (fun h => h.elim (hdisj _ hmem) hnin) hpl
    refine ⟨dn ++ [(c, c, m c)], by rw [← hdn]; simp, repW_live_congr ks hR1 (fun x => ?_), hK1, ?_⟩
    · simp only [List.map_append, List.map_cons, List.map_nil, List.mem_append, List.mem_singleton]
      tauto
    · simp only [List.length_append, List.length_cons, List.length_nil]; omega

theorem mainCount_zero {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64}
    (hR : CoreRep ks s m (fun _ => False)) : mainCount s = 0 := by
  unfold mainCount
  rw [List.countP_eq_zero]
  intro a ha hna
  obtain ⟨i, hi⟩ := List.mem_iff_getElem?.mp ha
  exact (hR.main_val i a hi (by simpa using hna)).1

theorem idxs_length_le {ns : List ONode} {T : OTree} (hnd : (idxs T).Nodup)
    (hlt : ∀ j ∈ idxs T, j < ns.length) : (idxs T).length ≤ ns.length := by
  have hsub : idxs T ⊆ List.range ns.length := fun j hj => List.mem_range.mpr (hlt j hj)
  have := (hnd.subperm hsub).length_le
  rwa [List.length_range] at this

/-- Moving the side store of `s` onto a table that represents `m` on the main coordinates of `s`. -/
theorem attach_side {s b : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) (hb : RepW ks b m (InMain s)) :
    RepW ks { b with side := s.side } m (fun c => InMain s c ∨ InSide s c) := by
  refine ⟨rep_side_update ks hb.1 s.side hR.side_len_vals (fun i c hc => ?_) hR.side_uniq
    (fun c hc => ?_) (fun _ _ => rfl) (fun c hc => ?_), hb.2⟩
  · obtain ⟨a, _, d⟩ := hR.side_val i c hc
    exact ⟨a, Or.inr ⟨i, hc⟩, d⟩
  · exact ⟨fun h => h.elim (fun ⟨hh, _⟩ => absurd hc hh) id, fun h => Or.inr h⟩
  · exact ⟨fun h => h.elim id (fun ⟨i, hi⟩ => absurd (hR.side_val i c hi).1 hc), Or.inl⟩

/-- **`resize_rep`.**  `resize` to a positive bit count preserves `RepW` and the key column, and does
not raise the arena plus the main count. -/
theorem resize_rep (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (KC : Slice U64 → KS → Prop)
    (hKC : ∀ (coords coords' : Slice U64) (keys keys' : KS) (i : Usize) (k : U64),
      KC coords keys → coords'.val = coords.val.set i.val k → ks.set keys i k = ok keys' →
      KC coords' keys')
    (hTK : ∀ (coords coords' : Slice U64) (keys : KS) (i : Usize) (c key : U64) (keys' : KS),
      KC coords keys → coords.val[i.val]? = some c → ¬ IsSent c →
      ks.take_key keys i c = ok (key, keys') → coords'.val = coords.val.set i.val hashtrinity.EMPTY →
      key = c ∧ KC coords' keys')
    (hKW : ∀ (b : U32) (t : Tbl KS), 0 < b.val →
      hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI ks b = ok t → KC t.coords t.keys) :
    ResizeOK ks KC := by
  intro s s' nb m live hR hK hsmall hpos h
  unfold hashtrinity.Table.resize at h
  obtain ⟨b0, hwb, h⟩ := bind_inv h
  have hb0C := rep_map_congr ks (m' := m) (with_bits_rep ks nb hpos b0 hwb) (fun _ h => h.elim)
  have hb0o := with_bits_overflow ks nb b0 hwb
  have hb0 : RepW ks b0 m (LowMain s 0) := by
    refine repW_live_congr ks ⟨hb0C, by unfold OvWF; rw [hb0o]; trivial⟩ (fun c => ?_)
    exact ⟨False.elim, fun ⟨_, d, hd, _⟩ => absurd hd (Nat.not_lt_zero d)⟩
  have hcb0 : arenaLen b0 + mainCount b0 = 0 := by
    have h1 : arenaLen b0 = 0 := by unfold arenaLen; rw [hb0o]
    rw [h1, mainCount_zero ks hb0C]
  obtain ⟨⟨ss, o, b1⟩, hl0, h⟩ := bind_inv h
  obtain ⟨hss, hoo, hRb1, hKb1, hcb1⟩ :=
    resize_loop0_spec ks hm KC hKC hTK hR hK hsmall b0 hb0 (hKW nb b0 hpos hwb) hcb0 _ hl0
  dsimp only at hss hoo hRb1 hKb1 hcb1
  subst hss hoo
  obtain ⟨ss1, -, h⟩ := bind_inv h
  have hRs := attach_side ks hR.1 hRb1
  simp only [core.mem.replace, core.option.Option.take] at h
  cases hso : s.overflow with
  | none =>
    rw [hso] at h
    have := (Result.ok_injective h).symm
    subst this
    refine ⟨repW_live_congr ks hRs (fun c => ?_), hKb1, ?_⟩
    · rw [hR.1.live_iff c]
      have : ¬ InOv s c := fun ⟨o', ho', _⟩ => by rw [hso] at ho'; exact absurd ho' (by simp)
      tauto
    · show arenaLen b1 + mainCount b1 ≤ arenaLen s + mainCount s
      omega
  | some ov =>
    rw [hso] at h
    obtain ⟨es, hes, h1⟩ := bind_inv h
    obtain ⟨iter, hiter, h2⟩ := bind_inv h1
    unfold alloc.vec.IntoIteratorVec.into_iter at hiter
    have hie : iter = es := (Result.ok_injective hiter).symm
    subst hie
    obtain ⟨T, ⟨hI, hall⟩, hiff⟩ := ov_full_of_some ks hR hso
    have hesv := into_entries_spec m ov T hI.1 hI.2.2.1 (fun i hi => by
      obtain ⟨nd, hnd, ha, he, _⟩ := hall i hi; exact ⟨nd, hnd, ha, he⟩) iter hes
    have hTnd := nd_nodup T hI.2.2.1
    have hmemE : ∀ x ∈ iter.val, ∃ j ∈ idxs T, ∃ nd, ov.nodes.val[j]? = some nd ∧
        x = (nd.key, nd.key, m nd.key) ∧ ¬ IsSent nd.key := by
      intro x hx
      rw [hesv, List.mem_map] at hx
      obtain ⟨j, hj, rfl⟩ := hx
      obtain ⟨nd, hnd, _, _, hns, _⟩ := hall j hj
      exact ⟨j, hj, nd, hnd, by simp [entOf, hnd], hns⟩
    have hnodup : (iter.val.map Prod.fst).Nodup := by
      rw [hesv, List.map_map]
      refine List.Nodup.map_on (fun a ha b hb hab => ?_) hTnd
      by_contra hne
      obtain ⟨nda, hnda, _⟩ := hall a ha
      obtain ⟨ndb, hndb, _⟩ := hall b hb
      apply bst_key_inj hI.2.1 ha hb hne
      simp only [Function.comp, entOf, hnda, hndb] at hab
      rw [keyAt_get hnda, keyAt_get hndb, hab]
    have hlenT : (idxs T).length ≤ ov.nodes.val.length :=
      idxs_length_le hTnd (fun j hj => by
        obtain ⟨nd, hnd, _⟩ := hall j hj
        exact (List.getElem?_eq_some_iff.mp hnd).1)
    have hlenE : iter.val.length = (idxs T).length := by rw [hesv, List.length_map]
    have harena : arenaLen s = ov.nodes.val.length := by unfold arenaLen; rw [hso]
    obtain ⟨hRr, hKr, hcr⟩ := resize_loop1_spec ks hm KC hKC iter.val
      (fun x hx => by
        obtain ⟨j, _, nd, _, rfl, hns⟩ := hmemE x hx
        exact ⟨rfl, rfl, hns⟩)
      hnodup _ hRs hKb1
      (fun x hx => by
        obtain ⟨j, hj, nd, hnd, rfl, hns⟩ := hmemE x hx
        rintro (⟨_, d, hd⟩ | ⟨d, hd⟩)
        · exact not_inOv_of_main ks hR.1 hns hd ((hiff _).mp ⟨j, hj, nd, hnd, rfl⟩)
        · exact hns (hR.1.side_val d _ hd).1)
      (by show arenaLen b1 + mainCount b1 + _ < nilN; omega) iter rfl s' h2
    refine ⟨repW_live_congr ks hRr (fun c => ?_), hKr, by
      have : arenaLen ({ b1 with side := s.side } : Tbl KS) + mainCount ({ b1 with side := s.side } : Tbl KS) =
        arenaLen b1 + mainCount b1 := rfl
      omega⟩
    rw [hR.1.live_iff c]
    have hE : c ∈ iter.val.map Prod.fst ↔ InOv s c := by
      rw [← hiff c]
      constructor
      · intro hc
        obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hc
        obtain ⟨j, hj, nd, hnd, rfl, _⟩ := hmemE x hx
        exact ⟨j, hj, nd, hnd, rfl⟩
      · rintro ⟨j, hj, nd, hnd, rfl⟩
        rw [hesv, List.map_map]
        exact List.mem_map.mpr ⟨j, hj, by simp [entOf, hnd]⟩
    rw [hE]
    tauto

end

/-! ## The two key stores -/

theorem stored_take_key (coords coords' : Slice U64) (keys : hashtrinity.algebra.Stored U64)
    (i : Usize) (c key : U64) (keys' : hashtrinity.algebra.Stored U64) (hK : SKC coords keys)
    (hc : coords.val[i.val]? = some c) (hns : ¬ IsSent c)
    (h : (SKS).take_key keys i c = ok (key, keys'))
    (hco : coords'.val = coords.val.set i.val hashtrinity.EMPTY) :
    key = c ∧ SKC coords' keys' := by
  change hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key CI
    core.cmp.EqU64 keys i c = ok (key, keys') at h
  unfold hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key at h
  obtain ⟨⟨mu, back⟩, hg, h⟩ := bind_inv h
  obtain ⟨hmu, hback⟩ := get_unchecked_mut_ok _ _ _ _ hg
  obtain ⟨t, ht, h⟩ := bind_inv h
  have h' := Result.ok_injective h
  simp only [Prod.mk.injEq] at h'
  obtain ⟨rfl, rfl⟩ := h'
  obtain ⟨hl, hall⟩ := hK
  have hki := hall i.val c hc hns
  have hmu' : mu = some c := by
    have : keys.val[i.val]? = some mu := hmu
    rw [hki] at this
    exact (Option.some.inj this).symm
  have hkv : (back mu).val = keys.val := by rw [hback, Slice.set_val_eq, list_set_self _ _ _ hmu]
  subst hmu'
  refine ⟨(Option.some.inj (assume_init_read_ok _ _ ht)).symm, ?_, fun d c' hd hcns => ?_⟩
  · show (back (some c)).val.length = coords'.val.length
    rw [hkv, hco, List.length_set]; exact hl
  · show (back (some c)).val[d]? = _
    rw [hkv]
    rw [hco] at hd
    by_cases hdi : d = i.val
    · subst hdi
      rw [List.getElem?_set_self (List.getElem?_eq_some_iff.mp hc).1] at hd
      exact absurd (Option.some.inj hd ▸ Or.inl EMPTY_val) hcns
    · rw [List.getElem?_set_ne (Ne.symm hdi)] at hd
      exact hall d c' hd hcns

theorem regen_take_key (coords coords' : Slice U64) (keys : hashtrinity.algebra.Regen U64)
    (i : Usize) (c key : U64) (keys' : hashtrinity.algebra.Regen U64) (_hK : True)
    (_hc : coords.val[i.val]? = some c) (_hns : ¬ IsSent c)
    (h : (RKS).take_key keys i c = ok (key, keys'))
    (_hco : coords'.val = coords.val.set i.val hashtrinity.EMPTY) :
    key = c ∧ True := by
  change hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
    U64.Insts.Apeiron_wordHashtrinityAlgebraCoordInv core.cmp.EqU64 keys i c = ok (key, keys') at h
  unfold hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key at h
  obtain ⟨t, ht, h⟩ := bind_inv h
  have h1 : t = key := congrArg Prod.fst (Result.ok_injective h)
  have h2 : c = t := Result.ok_injective ht
  exact ⟨by rw [← h1, ← h2], trivial⟩

/-- **`resizeOK_stored`.**  `ResizeOK` at `Stored<u64>` with its key column. -/
theorem resizeOK_stored : ResizeOK SKS SKC :=
  resize_rep SKS SKS_matches SKC skc_set stored_take_key
    (fun b t hb h => (stored_with_bits_rep b hb t h).2)

/-- **`resizeOK_regen`.**  `ResizeOK` at `Regen<u64>`. -/
theorem resizeOK_regen : ResizeOK RKS (fun _ _ => True) :=
  resize_rep RKS RKS_matches (fun _ _ => True) (fun _ _ _ _ _ _ _ _ _ => trivial) regen_take_key
    (fun _ _ _ _ => trivial)

/-! ## The headlines -/

/-- **`insert_preserves` (`Stored<u64>`).**  `word_hashtrinity_insert` on a non-reserved key preserves
`TableRep ∧ OvWF` and updates the canonical map by `insertWith (fun new _ => new)`, while the arena
plus the main count is below `NIL`. -/
theorem insert_preserves (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) (hR : TableRep s m live) (hW : OvWF s)
    (hsmall : arenaLen s + mainCount s < nilN) (k : U64) (hns : ¬ IsSent k) (v : I64)
    (s' : Tbl (hashtrinity.algebra.Stored U64)) (h : word_hashtrinity_insert s k v = ok s') :
    TableRep s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) ∧
      OvWF s' :=
  insert_preserves_of_resize resizeOK_stored s m live hR hW hsmall k hns v s' h

/-- **`coord_insert_preserves` (`Regen<u64>`).** -/
theorem coord_insert_preserves (s : Tbl (hashtrinity.algebra.Regen U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) (hR : CoordRep s m live) (hW : OvWF s)
    (hsmall : arenaLen s + mainCount s < nilN) (k : U64) (hns : ¬ IsSent k) (v : I64)
    (s' : Tbl (hashtrinity.algebra.Regen U64)) (h : word_coordmap_insert s k v = ok s') :
    CoordRep s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) ∧
      OvWF s' :=
  coord_insert_preserves_of_resize resizeOK_regen s m live hR hW hsmall k hns v s' h

/-- **`insert_refines` (`Stored<u64>`, every key).** -/
theorem insert_refines (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) (hR : TableRep s m live) (hW : OvWF s)
    (hsmall : arenaLen s + mainCount s < nilN) (k : U64) (v : I64)
    (s' : Tbl (hashtrinity.algebra.Stored U64)) (h : word_hashtrinity_insert s k v = ok s') :
    TableRep s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) ∧
      OvWF s' :=
  insert_refines_of_resize resizeOK_stored s m live hR hW hsmall k v s' h

/-- **`coord_insert_refines` (`Regen<u64>`, every key).** -/
theorem coord_insert_refines (s : Tbl (hashtrinity.algebra.Regen U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) (hR : CoordRep s m live) (hW : OvWF s)
    (hsmall : arenaLen s + mainCount s < nilN) (k : U64) (v : I64)
    (s' : Tbl (hashtrinity.algebra.Regen U64)) (h : word_coordmap_insert s k v = ok s') :
    CoordRep s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) ∧
      OvWF s' :=
  coord_insert_refines_of_resize resizeOK_regen s m live hR hW hsmall k v s' h

/-- The constructors establish `OvWF` and an empty arena and main array. -/
theorem with_bits_wf {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64) (b : U32) (hb : 0 < b.val)
    (t : Tbl KS) (h : hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI ks b = ok t) :
    OvWF t ∧ arenaLen t + mainCount t = 0 := by
  have ho := with_bits_overflow ks b t h
  refine ⟨by unfold OvWF; rw [ho]; trivial, ?_⟩
  have h1 : arenaLen t = 0 := by unfold arenaLen; rw [ho]
  rw [h1, mainCount_zero ks (with_bits_rep ks b hb t h)]

theorem new_wf (s : Tbl (hashtrinity.algebra.Stored U64)) (h : word_hashtrinity_new = ok s) :
    OvWF s ∧ arenaLen s + mainCount s = 0 := by
  unfold word_hashtrinity_new hashtrinity.TableKVStored.new at h
  exact with_bits_wf SKS 3#u32 (by decide) s h

theorem with_capacity_wf (cap : Usize) (s : Tbl (hashtrinity.algebra.Stored U64))
    (h : word_hashtrinity_with_capacity cap = ok s) : OvWF s ∧ arenaLen s + mainCount s = 0 := by
  unfold word_hashtrinity_with_capacity hashtrinity.TableKVStored.with_capacity at h
  obtain ⟨b, hb, h⟩ := bind_inv h
  exact with_bits_wf SKS b (by have := bits_for_ge cap b hb; omega) s h

theorem coordinate_with_capacity_wf (cap : Usize) (s : Tbl (hashtrinity.algebra.Regen U64))
    (h : word_coordmap_coordinate_with_capacity cap = ok s) :
    OvWF s ∧ arenaLen s + mainCount s = 0 := by
  unfold word_coordmap_coordinate_with_capacity hashtrinity.TableKVRegen.coordinate_with_capacity
    at h
  obtain ⟨b, hb, h⟩ := bind_inv h
  exact with_bits_wf RKS b (by have := bits_for_ge cap b hb; omega) s h

end Apeiron.MapExecuted
