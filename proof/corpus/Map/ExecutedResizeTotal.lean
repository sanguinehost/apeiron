-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (APEIRON-HT-INSERT, HT-INSERT-W17): totality of
-- the executed resize.
import Map.ExecutedInsertTotal
open Aeneas Aeneas.Std Result

/-!
# Totality of the executed resize

`resize_loop0_spec` / `resize_loop1_spec` (`Map/ExecutedResize.lean`) are partial correctness: they
assume the run `resize_loopN ... = ok r`.  This module proves the loops and `resize` itself return
`ok`, by the decreasing measure `loop_tot_inv` (`Map/ExecutedReadTotal.lean:30`).

The `place_capped` call the loops make is total under the counter-form overflow domain `Hov` and the
initialised-value domain `BigInit`; `place_capped_domain` supplies the existence and advances `Hov`
by one, `place_capped_frame` the shape, `place_capped_bigInit` the initialised-value domain.  Every
step's domain premise is a the body reads; the loop index `i` never reaches `Usize.max` because
`i.val < s.coords.val.length ≤ Usize.max`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

theorem slice_update_ex {α : Type} (s : Slice α) (i : Usize) (x : α) (h : i.val < s.val.length) :
    ∃ s', Slice.update s i x = ok s' := by
  obtain ⟨s', hs', _⟩ := WP.spec_imp_exists (Slice.update_spec s i x h)
  exact ⟨s', hs'⟩

/-- `resize_loop0`'s step is total, from the invariant in the dispatch (`RepW`'s geometry plus
`BigInit`, `Hov`, the counter bound and `take_key` totality). -/
theorem resize_loop0_ok {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), ∃ keys', ks.set keys i k = ok keys')
    (hTK : ∀ (keys : KS) (i : Usize) (c : U64), ¬ IsSent c →
      ∃ keys', ks.take_key keys i c = ok (c, keys'))
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live)
    (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (b0 : Tbl KS) (hb0 : RepW ks b0 m (LowMain s 0))
    (hcb0 : arenaLen b0 + mainCount b0 = 0) (hb0o : b0.overflow = none) :
    ∃ r, hashtrinity.Table.resize_loop0 core.cmp.EqU64 CI ks s b0 0#usize = ok r := by
  unfold hashtrinity.Table.resize_loop0
  refine loop_tot_inv
    (fun p : Tbl KS × Tbl KS × Usize =>
      hashtrinity.Table.resize_loop0.body core.cmp.EqU64 CI ks p.1 p.2.1 p.2.2)
    (fun p : Tbl KS × Tbl KS × Usize =>
      p.1.coords.val.length = s.coords.val.length ∧
      (∀ d, p.2.2.val ≤ d → p.1.coords.val[d]? = s.coords.val[d]?) ∧
      p.1.vals.val = s.vals.val ∧
      p.1.side = s.side ∧ p.1.overflow = s.overflow ∧
      0 < p.2.1.bits.val ∧
      p.2.1.coords.val.length = 2 ^ p.2.1.bits.val ∧
      p.2.1.vals.val.length = p.2.1.coords.val.length ∧
      BigInit p.2.1 ∧
      p.2.1.len.val ≤ (s.coords.val.take p.2.2.val).countP (fun c => decide (¬ IsSent c)) ∧
      Hov p.2.1 ((s.coords.val.take p.2.2.val).countP (fun c => decide (¬ IsSent c))) ∧
      p.2.2.val ≤ s.coords.val.length)
    (fun p => s.coords.val.length - p.2.2.val)
    ?_ ?_ (s, b0, 0#usize) ?_
  · rintro ⟨t, b, i⟩ ⟨hA1, hB, hC, hE1, hE2, hbitsb, hlenb, hslb, hBig, hlenbnd, hHov, hib⟩
    dsimp only at *
    unfold hashtrinity.Table.resize_loop0.body
    by_cases hlt : i < Slice.len t.coords
    · rw [if_pos hlt]
      have hltv : i.val < t.coords.val.length := by
        have h := (UScalar.lt_equiv i (Slice.len t.coords)).mp hlt
        rwa [Slice.len_val] at h
      have hit : i.val < s.coords.val.length := by rw [← hA1]; exact hltv
      obtain ⟨c, hc⟩ := slice_index_ex t.coords i hltv
      rw [hc]; simp only [bind_tc_ok]
      rw [is_live_ok]; simp only [bind_tc_ok]
      have hiadd : i.val + (1#usize).val < 2 ^ UScalarTy.Usize.numBits := by
        rw [val_1_usize, show UScalarTy.Usize.numBits = System.Platform.numBits from rfl]
        have h : s.coords.val.length < 2 ^ System.Platform.numBits := slice_len_lt s.coords
        omega
      by_cases hsent : IsSent c
      · rw [if_neg (by simpa using hsent)]
        obtain ⟨i2, hi2, -⟩ := uadd_tot i 1#usize hiadd
        rw [hi2]; simp only [bind_tc_ok]
        exact ⟨_, rfl⟩
      · rw [if_pos (by simpa using hsent)]
        obtain ⟨kt, htk⟩ := hTK t.keys i c hsent
        rw [htk]
        simp only [bind_tc_ok]
        change ∃ cf, (do
            let (mu, get_unchecked_mut_back) ← core.slice.Slice.get_unchecked_mut
              (core.slice.index.SliceIndexUsizeSlice (core.mem.maybe_uninit.MaybeUninit I64)) t.vals i
            let v ← core.mem.maybe_uninit.MaybeUninit.assume_init_read mu
            let s ← Slice.update t.coords i hashtrinity.EMPTY
            let bigger1 ← hashtrinity.Table.place_capped core.cmp.EqU64 CI ks b c c v
            let i2 ← i + 1#usize
            ok (ControlFlow.cont
              ({ t with coords := s, vals := get_unchecked_mut_back mu, keys := kt }, bigger1, i2)))
          = ok cf
        have hltv' : i.val < t.vals.val.length := by
          have h1 : t.vals.val.length = s.coords.val.length := by rw [hC]; exact hR.1.vals_len
          rw [h1]; exact hit
        obtain ⟨mu, back, hg⟩ := gum_ex t.vals i hltv'
        have htc : t.coords.val[i.val]? = some c := slice_index_ok t.coords i c hc
        have hsc : s.coords.val[i.val]? = some c := by rw [← hB i.val le_rfl]; exact htc
        have hsrc : s.vals.val[i.val]? = some (some (m c)) := (hR.1.main_val i.val c hsc hsent).2
        have hmu_arr : t.vals.val[i.val]? = some mu := (get_unchecked_mut_ok t.vals i mu back hg).1
        have hmu : mu = some (m c) := by
          have h2 : t.vals.val[i.val]? = some (some (m c)) := by rw [hC]; exact hsrc
          rw [h2] at hmu_arr
          exact (Option.some.inj hmu_arr).symm
        rw [hg]
        rw [show (mu, back) = (some (m c), back) from by rw [hmu]]
        simp only [bind_tc_ok]
        change ∃ cf, (match (some (m c), back) with
          | (mu, get_unchecked_mut_back) => (do
              let v ← core.mem.maybe_uninit.MaybeUninit.assume_init_read mu
              let s ← Slice.update t.coords i hashtrinity.EMPTY
              let bigger1 ← hashtrinity.Table.place_capped core.cmp.EqU64 CI ks b c c v
              let i2 ← i + 1#usize
              ok (ControlFlow.cont
                ({ t with coords := s, vals := get_unchecked_mut_back mu, keys := kt }, bigger1, i2))))
          = ok cf
        have hred : (match (some (m c), back) with
          | (mu, get_unchecked_mut_back) => (do
              let v ← core.mem.maybe_uninit.MaybeUninit.assume_init_read mu
              let s ← Slice.update t.coords i hashtrinity.EMPTY
              let bigger1 ← hashtrinity.Table.place_capped core.cmp.EqU64 CI ks b c c v
              let i2 ← i + 1#usize
              ok (ControlFlow.cont
                ({ t with coords := s, vals := get_unchecked_mut_back mu, keys := kt }, bigger1, i2))))
          = ((do
              let v ← core.mem.maybe_uninit.MaybeUninit.assume_init_read (some (m c))
              let s ← Slice.update t.coords i hashtrinity.EMPTY
              let bigger1 ← hashtrinity.Table.place_capped core.cmp.EqU64 CI ks b c c v
              let i2 ← i + 1#usize
              ok (ControlFlow.cont
                ({ t with coords := s, vals := back (some (m c)), keys := kt }, bigger1, i2)))
            : Result (ControlFlow (Tbl KS × Tbl KS × Usize)
                (hashtrinity.side.SideStore U64 I64 KS ×
                  Option (hashtrinity.overflow.Overflow U64 I64) × Tbl KS))) := rfl
        rw [hred]
        rw [show core.mem.maybe_uninit.MaybeUninit.assume_init_read (some (m c)) = ok (m c) from rfl]
        simp only [bind_tc_ok]
        obtain ⟨cs, hupd⟩ := slice_update_ex t.coords i hashtrinity.EMPTY hltv
        rw [hupd]; simp only [bind_tc_ok]
        have hK : (s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)) + 4 ≤
            2 ^ 32 - 1 := by
          have hn := mainCount_le s i.val
          have h := hcap; unfold nilN at h; omega
        have hlenmax : b.len.val < Usize.max := by
          have h := nilN_le_usize_max; unfold nilN at h; omega
        obtain ⟨big1, hpc, -⟩ := place_capped_domain ks hm hA hset b c (m c)
          ((s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)))
          hbitsb hlenb hslb (fun j hcj => hBig j c hcj hsent) hlenmax hHov hK
        rw [hpc]; simp only [bind_tc_ok]
        obtain ⟨i2, hi2, -⟩ := uadd_tot i 1#usize hiadd
        rw [hi2]; simp only [bind_tc_ok]
        exact ⟨_, rfl⟩
    · rw [if_neg hlt]
      exact ⟨_, rfl⟩
  · rintro ⟨t, b, i⟩ ⟨t', b', i'⟩ ⟨hA1, hB, hC, hE1, hE2, hbitsb, hlenb, hslb, hBig, hlenbnd, hHov, hib⟩ hcf
    dsimp only at *
    unfold hashtrinity.Table.resize_loop0.body at hcf
    by_cases hlt : i < Slice.len t.coords
    · rw [if_pos hlt] at hcf
      have hltv : i.val < t.coords.val.length := by
        have h := (UScalar.lt_equiv i (Slice.len t.coords)).mp hlt
        rwa [Slice.len_val] at h
      have hit : i.val < s.coords.val.length := by rw [← hA1]; exact hltv
      obtain ⟨c, hc, hcf⟩ := bind_inv hcf
      have htc : t.coords.val[i.val]? = some c := slice_index_ok t.coords i c hc
      have hsc : s.coords.val[i.val]? = some c := by rw [← hB i.val le_rfl]; exact htc
      rw [is_live_ok] at hcf; simp only [bind_tc_ok] at hcf
      by_cases hsent : IsSent c
      · rw [if_neg (by simpa using hsent)] at hcf
        obtain ⟨i2, hi2, hcf⟩ := bind_inv hcf
        have hi2v : i2.val = i.val + 1 := by
          have := (uadd_ok hi2).1; rw [val_1_usize] at this; exact this
        have hh := Result.ok_injective hcf
        simp only [ControlFlow.cont.injEq, Prod.mk.injEq] at hh
        obtain ⟨rfl, rfl, rfl⟩ := hh
        dsimp only at *
        have hcnt : (s.coords.val.take i2.val).countP (fun c => decide (¬ IsSent c)) =
            (s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)) := by
          rw [hi2v, countP_take_succ (fun c => decide (¬ IsSent c)) s.coords.val i.val c hsc]
          simp [hsent]
        refine ⟨⟨hA1, (fun d hd => hB d (by omega)), hC, hE1, hE2, hbitsb, hlenb, hslb, hBig,
          ?_, ?_, ?_⟩, ?_⟩
        · rw [hcnt]; exact hlenbnd
        · rw [hcnt]; exact hHov
        · omega
        · omega
      · rw [if_pos (by simpa using hsent)] at hcf
        obtain ⟨⟨key, keys'⟩, htk, hcf⟩ := bind_inv hcf
        obtain ⟨kt, htk2⟩ := hTK t.keys i c hsent
        have hkey : key = c := congrArg Prod.fst (Result.ok_injective (htk.symm.trans htk2))
        obtain ⟨⟨mu, back⟩, hg, hcf⟩ := bind_inv hcf
        obtain ⟨hmu_arr, hback⟩ := get_unchecked_mut_ok t.vals i mu back hg
        obtain ⟨v, hread, hcf⟩ := bind_inv hcf
        obtain ⟨cs, hupd, hcf⟩ := bind_inv hcf
        obtain ⟨big1, hpc, hcf⟩ := bind_inv hcf
        rw [hkey] at hpc
        obtain ⟨i2, hi2, hcf⟩ := bind_inv hcf
        have hi2v : i2.val = i.val + 1 := by
          have := (uadd_ok hi2).1; rw [val_1_usize] at this; exact this
        have hh := Result.ok_injective hcf
        simp only [ControlFlow.cont.injEq, Prod.mk.injEq] at hh
        obtain ⟨rfl, rfl, rfl⟩ := hh
        dsimp only at *
        have hcnt : (s.coords.val.take i2.val).countP (fun c => decide (¬ IsSent c)) =
            (s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)) + 1 := by
          rw [hi2v, countP_take_succ (fun c => decide (¬ IsSent c)) s.coords.val i.val c hsc]
          simp [hsent]
        obtain ⟨hbitsf, hlenf, hslf, hlenf'⟩ := place_capped_frame ks b big1 c v hpc
        have hlenf2 : big1.coords.val.length = b.coords.val.length := hlenf
        have hslf2 : big1.vals.val.length = b.vals.val.length := hslf
        have hK : (s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)) + 4 ≤
            2 ^ 32 - 1 := by
          have hn := mainCount_le s i.val
          have h := hcap; unfold nilN at h; omega
        have hlenmax : b.len.val < Usize.max := by
          have h := nilN_le_usize_max; unfold nilN at h; omega
        have hHov' : Hov big1 ((s.coords.val.take i2.val).countP (fun c => decide (¬ IsSent c))) := by
          rw [hcnt]
          exact place_capped_hov ks hm hA hset b big1 c v
            ((s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)))
            hbitsb hlenb hslb (fun j hcj => hBig j c hcj hsent) hlenmax hHov hK hpc
        have hupd' : cs.val = t.coords.val.set i.val hashtrinity.EMPTY :=
          slice_update_ok t.coords cs i hashtrinity.EMPTY hupd
        have hvals : (back mu).val = s.vals.val := by
          rw [hback, Slice.set_val_eq, list_set_self _ _ _ hmu_arr, hC]
        refine ⟨⟨?_, ?_, hvals, hE1, hE2, ?_, ?_, ?_,
          place_capped_bigInit ks b big1 c v hpc hBig, ?_, hHov', ?_⟩, ?_⟩
        · rw [hupd', List.length_set]; exact hA1
        · intro d hd
          rw [hupd', List.getElem?_set_ne (by omega)]
          exact hB d (by omega)
        · rw [hbitsf]; exact hbitsb
        · rw [hlenf2, hbitsf]; exact hlenb
        · rw [hslf2, hlenf2]; exact hslb
        · rw [hcnt]; omega
        · omega
        · omega
    · rw [if_neg hlt] at hcf
      exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
  · refine ⟨rfl, (fun d _ => rfl), rfl, rfl, rfl, hb0.1.bits_pos, hb0.1.len_pow, hb0.1.vals_len,
      ?_, ?_, ?_, Nat.zero_le _⟩
    · exact fun j c' hc hns => ⟨m c', (hb0.1.main_val j c' hc hns).2⟩
    · have hb0len : b0.len.val = 0 := by rw [hb0.1.len_count]; omega
      rw [hb0len]; exact Nat.zero_le _
    · intro ov h
      rw [hb0o] at h
      exact absurd h (by simp)

/-- `IntoIter::next` is total (`none` at the end, the head otherwise). -/
theorem next_ex {α : Type} (it : alloc.vec.into_iter.IntoIter α) :
    ∃ o it1, alloc.vec.into_iter.IteratorIntoIter.next it = ok (o, it1) := by
  unfold alloc.vec.into_iter.IteratorIntoIter.next
  split
  · exact ⟨none, it, rfl⟩
  · rename_i hd tl heq
    exact ⟨some hd, _, rfl⟩

/-- `resize_loop1`'s step is total, from the invariant of the dispatch: the counter-form overflow
domain `Hov`, the shape and `BigInit`, and the entry list's `k = coord` and non-sentinel facts. -/
theorem resize_loop1_ok {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), ∃ keys', ks.set keys i k = ok keys')
    (es : List (U64 × U64 × I64))
    (hes : ∀ x ∈ es, x.2.1 = x.1 ∧ ¬ IsSent x.1)
    {t0 : Tbl KS} (n0 : Nat)
    (hbits0 : 0 < t0.bits.val) (hlen0 : t0.coords.val.length = 2 ^ t0.bits.val)
    (hsl0 : t0.vals.val.length = t0.coords.val.length)
    (hBig0 : BigInit t0) (hlenbnd0 : t0.len.val ≤ n0) (hHov0 : Hov t0 n0)
    (hcap : n0 + es.length + 4 ≤ nilN)
    (iter : alloc.vec.into_iter.IntoIter (U64 × U64 × I64)) (hiter : iter.val = es) :
    ∃ r, hashtrinity.Table.resize_loop1 core.cmp.EqU64 CI ks iter t0 = ok r := by
  unfold hashtrinity.Table.resize_loop1
  refine loop_tot_inv
    (fun p : alloc.vec.into_iter.IntoIter (U64 × U64 × I64) × Tbl KS =>
      hashtrinity.Table.resize_loop1.body core.cmp.EqU64 CI ks p.1 p.2)
    (fun p : alloc.vec.into_iter.IntoIter (U64 × U64 × I64) × Tbl KS =>
      ∃ dn, dn ++ p.1.val = es ∧
        0 < p.2.bits.val ∧ p.2.coords.val.length = 2 ^ p.2.bits.val ∧
        p.2.vals.val.length = p.2.coords.val.length ∧ BigInit p.2 ∧
        p.2.len.val ≤ n0 + dn.length ∧ Hov p.2 (n0 + dn.length))
    (fun p => p.1.val.length) ?_ ?_ (iter, t0) ?_
  · rintro ⟨it, tb⟩ ⟨dn, hdn, hbits, hlen, hsl, hBig, hlenbnd, hHov⟩
    dsimp only at *
    unfold hashtrinity.Table.resize_loop1.body
    obtain ⟨o, itr, hn⟩ := next_ex it
    rw [hn]; simp only [bind_tc_ok]
    rcases next_spec _ _ _ hn with ⟨hnil, rfl⟩ | ⟨e, hcons, rfl⟩
    · exact ⟨_, rfl⟩
    · obtain ⟨c0, k0, v0⟩ := e
      have hmem : (c0, k0, v0) ∈ es := by rw [← hdn]; simp [hcons]
      obtain ⟨hk2, hns2⟩ := hes _ hmem
      have hk2' : k0 = c0 := hk2
      subst k0
      change ∃ cf, (match (c0, c0, v0) with
        | (c, k, v) => (do
            let bigger1 ← hashtrinity.Table.place_capped core.cmp.EqU64 CI ks tb c c v
            ok (ControlFlow.cont (itr, bigger1))))
        = ok cf
      have hred : (match (c0, c0, v0) with
        | (c, k, v) => (do
            let bigger1 ← hashtrinity.Table.place_capped core.cmp.EqU64 CI ks tb c c v
            ok (ControlFlow.cont (itr, bigger1))))
        = ((do
            let bigger1 ← hashtrinity.Table.place_capped core.cmp.EqU64 CI ks tb c0 c0 v0
            ok (ControlFlow.cont (itr, bigger1)))
          : Result (ControlFlow
              (alloc.vec.into_iter.IntoIter (U64 × U64 × I64) × Tbl KS) (Tbl KS))) := rfl
      rw [hred]
      have hK : n0 + dn.length + 4 ≤ 2 ^ 32 - 1 := by
        have hdle : dn.length ≤ es.length := by
          have h1 : (dn ++ it.val).length = es.length := congrArg List.length hdn
          simp at h1; omega
        have h := hcap; unfold nilN at h; omega
      have hlenmax : tb.len.val < Usize.max := by
        have h := nilN_le_usize_max; unfold nilN at h; omega
      obtain ⟨big1, hpc, -⟩ := place_capped_domain ks hm hA hset tb c0 v0 (n0 + dn.length)
        hbits hlen hsl (fun j hcj => hBig j c0 hcj hns2) hlenmax hHov hK
      rw [hpc]; simp only [bind_tc_ok]
      exact ⟨_, rfl⟩
  · rintro ⟨it, tb⟩ ⟨it', tb'⟩ ⟨dn, hdn, hbits, hlen, hsl, hBig, hlenbnd, hHov⟩ hcf
    dsimp only at *
    unfold hashtrinity.Table.resize_loop1.body at hcf
    obtain ⟨⟨o, itr⟩, hn, hcf⟩ := bind_inv hcf
    rcases next_spec _ _ _ hn with ⟨hnil, rfl⟩ | ⟨e, hcons, rfl⟩
    · have hh := Result.ok_injective hcf
      exact absurd hh (by intro hh'; cases hh')
    · obtain ⟨c0, k0, v0⟩ := e
      have hmem : (c0, k0, v0) ∈ es := by rw [← hdn]; simp [hcons]
      obtain ⟨hk2, hns2⟩ := hes _ hmem
      have hk2' : k0 = c0 := hk2
      subst k0
      obtain ⟨tb2, hpc, hcf⟩ := bind_inv hcf
      have hh := Result.ok_injective hcf
      simp only [ControlFlow.cont.injEq, Prod.mk.injEq] at hh
      obtain ⟨rfl, rfl⟩ := hh
      dsimp only at *
      have hdn' : dn ++ [(c0, c0, v0)] ++ itr.val = es := by
        rw [← hdn, hcons]; simp
      have hcnt : (dn ++ [(c0, c0, v0)]).length = dn.length + 1 := by simp
      have hK : n0 + dn.length + 4 ≤ 2 ^ 32 - 1 := by
        have hdle : dn.length ≤ es.length := by
          have h1 : (dn ++ it.val).length = es.length := congrArg List.length hdn
          simp at h1; omega
        have h := hcap; unfold nilN at h; omega
      have hlenmax : tb.len.val < Usize.max := by
        have h := nilN_le_usize_max; unfold nilN at h; omega
      obtain ⟨hbf, hlf, hslf, hlenf'⟩ := place_capped_frame ks tb tb2 c0 v0 hpc
      have hlf2 : tb2.coords.val.length = tb.coords.val.length := hlf
      have hslf2 : tb2.vals.val.length = tb.vals.val.length := hslf
      have hHov' : Hov tb2 (n0 + (dn ++ [(c0, c0, v0)]).length) := by
        rw [hcnt]
        exact place_capped_hov ks hm hA hset tb tb2 c0 v0 (n0 + dn.length)
          hbits hlen hsl (fun j hcj => hBig j c0 hcj hns2) hlenmax hHov hK hpc
      refine ⟨⟨dn ++ [(c0, c0, v0)], hdn', ?_, ?_, ?_, ?_, ?_, hHov'⟩, ?_⟩
      · rw [hbf]; exact hbits
      · rw [hlf2, hbf]; exact hlen
      · rw [hslf2, hlf2]; exact hsl
      · exact place_capped_bigInit ks tb tb2 c0 v0 hpc hBig
      · rw [hcnt]; omega
      · rw [hcons]; simp
  · refine ⟨[], by simpa using hiter, hbits0, hlen0, hsl0, hBig0, ?_, ?_⟩
    · simpa using hlenbnd0
    · simpa using hHov0

/-- **Joint witness.**  Every hypothesis of `resize_loop0_ok` holds at the `Regen<u64>` key store and
the fresh `with_bits 1` table (`s = b0`): `BigInit`/geometry/`Hov`/the counter bound are the fresh
table's, the `take_key`/`matches`/`set` premises are `Regen`'s (all total), and the loop returns
`ok`.  The premises are not individually satisfiable-only: one state satisfies them jointly. -/
theorem resize_loop0_witness :
    ∃ (s b0 : Tbl (hashtrinity.algebra.Regen U64))
      (r : hashtrinity.side.SideStore U64 I64 (hashtrinity.algebra.Regen U64) × Option Ov ×
        Tbl (hashtrinity.algebra.Regen U64)),
      hashtrinity.Table.resize_loop0 core.cmp.EqU64 CI RKS s b0 0#usize = ok r := by
  obtain ⟨t, hwb, -⟩ := with_bits1_RKS_ex
  have hcR : CoreRep RKS t (fun _ => 0#i64) (fun _ => False) :=
    with_bits_rep RKS 1#u32 (by decide) t hwb
  have hovt : t.overflow = none := with_bits_overflow RKS 1#u32 t hwb
  have hsum : arenaLen t + mainCount t = 0 := (with_bits_wf RKS 1#u32 (by decide) t hwb).2
  have hR : RepW RKS t (fun _ => 0#i64) (fun _ => False) :=
    ⟨hcR, by unfold OvWF; rw [hovt]; trivial⟩
  have hb0 : RepW RKS t (fun _ => 0#i64) (LowMain t 0) :=
    repW_live_congr RKS hR
      (fun c => ⟨False.elim, fun ⟨_, d, hd, _⟩ => absurd hd (Nat.not_lt_zero d)⟩)
  have hcap : arenaLen t + mainCount t + 4 ≤ nilN := by unfold nilN; omega
  obtain ⟨r, hr⟩ := resize_loop0_ok RKS
    (fun keys i c k => ⟨true, RKS_matches keys i c k⟩)
    (fun _ => avx2_spec_holds)
    (fun keys i k => ⟨keys, rfl⟩)
    (fun keys i c _ => ⟨keys, by
      change hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
        U64.Insts.Apeiron_wordHashtrinityAlgebraCoordInv core.cmp.EqU64 keys i c = ok (c, keys)
      unfold hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
      change (do let t ← (ok c : Result U64); ok (t, keys)) = ok (c, keys)
      simp only [bind_tc_ok]⟩)
    hR hcap t hb0 hsum hovt
  exact ⟨t, t, r, hr⟩

/-! ## `with_bits` totality

`Table::with_bits` builds the fresh destination table of a resize.  Its shift `1#usize <<< bits`
fails for `bits.val ≥ Usize.numBits`; below that it is `ok (2 ^ bits.val)`, and every other step
(`from_elem`, `into_boxed_slice`, `alloc_uninit_boxed`, the key store's `alloc`, `SideStore.empty`)
is total once the key store's `alloc` is.  The fresh table's shape facts and the initialised-value
domain `BigInit` and empty-overflow `Hov t 0` are its own construction. -/

/-- `1#usize <<< bits` succeeds and equals `2 ^ bits.val` below the word width. -/
theorem shl_1_ex (bits : U32) (h : bits.val < UScalarTy.Usize.numBits) :
    ∃ n : Usize, (1#usize <<< bits = ok n) ∧ n.val = 2 ^ bits.val := by
  refine ⟨⟨(1#usize).bv.shiftLeft bits.val⟩, ?_, ?_⟩
  · show UScalar.shiftLeft_UScalar 1#usize bits = ok _
    unfold UScalar.shiftLeft_UScalar UScalar.shiftLeft
    simp only [h, if_true]
  · show ((1#usize).bv.shiftLeft bits.val).toNat = 2 ^ bits.val
    rw [BitVec.shiftLeft_eq, BitVec.toNat_shiftLeft]
    have h1 : (1#usize).bv.toNat = 1 := rfl
    rw [h1, Nat.shiftLeft_eq, one_mul]
    apply Nat.mod_eq_of_lt
    exact Nat.pow_lt_pow_right (by norm_num) h

/-- **`with_bits_ex`.**  `Table::with_bits` is total below the word width when the key store's
`alloc` is, and the fresh table carries its shape facts, the initialised-value domain `BigInit` and
the empty overflow domain `Hov · 0`. -/
theorem with_bits_ex {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks')
    (bits : U32) (hbits : bits.val < UScalarTy.Usize.numBits) :
    ∃ t : Tbl KS,
      hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI ks bits = ok t ∧
      t.bits = bits ∧ t.coords.val.length = 2 ^ bits.val ∧
      t.vals.val.length = t.coords.val.length ∧
      t.len.val = 0 ∧ t.overflow = none ∧ BigInit t ∧ Hov t 0 := by
  unfold hashtrinity.Table.with_bits
  obtain ⟨n, hn, hnv⟩ := shl_1_ex bits hbits
  rw [hn]; simp only [bind_tc_ok]
  obtain ⟨v, hv⟩ := from_elem_ok n
  rw [hv]; simp only [bind_tc_ok]
  unfold alloc.vec.Vec.into_boxed_slice
  simp only [bind_tc_ok]
  obtain ⟨s1, hs1⟩ := alloc_uninit_ok I64 n
  rw [hs1]; simp only [bind_tc_ok]
  obtain ⟨g, hg⟩ := halloc n
  rw [hg]; simp only [bind_tc_ok]
  unfold hashtrinity.side.SideStore.empty
  obtain ⟨g0, hg0⟩ := halloc 0#usize
  rw [hg0]; simp only [bind_tc_ok]
  refine ⟨_, rfl, rfl, ?_, ?_, rfl, rfl, ?_, ?_⟩
  · show v.val.length = 2 ^ bits.val
    rw [from_elem_val hashtrinity.EMPTY n v hv, List.length_replicate, hnv]
  · show s1.val.length = v.val.length
    rw [alloc_uninit_val I64 n s1 hs1, from_elem_val hashtrinity.EMPTY n v hv]
    simp
  · intro i c hc hns
    have hvv := from_elem_val hashtrinity.EMPTY n v hv
    have hci : v.val[i]? = some c := hc
    rw [hvv] at hci
    have hc' : c = hashtrinity.EMPTY := (List.mem_replicate.mp (List.mem_of_getElem? hci)).2
    exact absurd (by rw [hc']; exact Or.inl EMPTY_val) hns
  · intro ov h
    simp at h

/-! ## `Overflow::in_order` totality

The in-order walk's inner loop (`in_order_loop0_loop0`/`loop1`) pushes the left spine of the
current subtree onto the stack; it descends into a strictly smaller subtree at each step, so it
terminates by structural recursion on the subtree shape.  The stack-length bound
`stack.val.length + |idxs T| < Usize.max` keeps `Vec.push` below its size ceiling. -/

theorem in_order_loop0_loop0_total (v : alloc.vec.Vec ONode) :
    ∀ (T0 : OTree) (cur : U32) (stack : alloc.vec.Vec U32),
      Emb v.val cur T0 →
      stack.val.length + (idxs T0).length < Usize.max →
      ∃ r, hashtrinity.overflow.Overflow.in_order_loop0_loop0 v stack cur = ok r := by
  intro T0
  induction T0 with
  | leaf =>
    intro cur stack hE hb
    refine ⟨stack, ?_⟩
    unfold hashtrinity.overflow.Overflow.in_order_loop0_loop0
    rw [loop.eq_def]
    unfold hashtrinity.overflow.Overflow.in_order_loop0_loop0.body
    dsimp only
    have hcur : cur = hashtrinity.overflow.NIL := (NIL_eq_iff cur).mpr hE
    rw [if_neg (by simp [hcur])]
    simp
  | node l i r ihl ihr =>
    intro cur stack hE hb
    obtain ⟨hci, hine, nd, hnd, hl, hr⟩ := hE
    have hciN : cur ≠ hashtrinity.overflow.NIL := fun hh =>
      hine (by rw [← hci]; exact (NIL_eq_iff cur).mp hh)
    obtain ⟨stack1, hpush⟩ := vec_push_ex stack cur (by
      have h1 : 0 < (idxs (OTree.node l i r)).length := by
        simp only [idxs, List.length_append, List.length_cons]; omega
      omega)
    obtain ⟨n, hidx⟩ := vec_index_ex v (UScalar.cast .Usize cur) (by
      rw [cast_usize_u32, hci]
      exact (List.getElem?_eq_some_iff.mp hnd).1)
    have hn : n = nd := by
      have := vec_index_ok v (UScalar.cast .Usize cur) n hidx
      rw [cast_usize_u32, hci, hnd] at this
      exact (Option.some.inj this).symm
    rw [hn] at hidx
    have hcond : (cur != hashtrinity.overflow.NIL) = true := by simpa using hciN
    have hbody : hashtrinity.overflow.Overflow.in_order_loop0_loop0.body v stack cur
        = ok (ControlFlow.cont (stack1, nd.left)) := by
      unfold hashtrinity.overflow.Overflow.in_order_loop0_loop0.body
      rw [if_pos hcond, hpush]
      simp only [lift, bind_tc_ok]
      rw [hidx]; simp only [bind_tc_ok]
    obtain ⟨r, hr⟩ := ihl nd.left stack1 hl (by
      rw [vec_push_ok stack stack1 cur hpush, List.length_append, List.length_singleton]
      have h2 : (idxs (OTree.node l i r)).length = (idxs l).length + 1 + (idxs r).length := by
        simp only [idxs, List.length_append, List.length_cons]; omega
      omega)
    refine ⟨r, ?_⟩
    unfold hashtrinity.overflow.Overflow.in_order_loop0_loop0
    rw [loop.eq_def]
    dsimp only
    rw [hbody]
    simp only [bind_ok]
    show hashtrinity.overflow.Overflow.in_order_loop0_loop0 v stack1 nd.left = ok r
    exact hr

/-! ## `Overflow::in_order` outer-loop totality -/

/-- `pendG` has at least one element per stack frame. -/
theorem pendG_length_le : ∀ (fr : List (Nat × OTree)), fr.length ≤ (pendG fr).length
  | [] => by simp [pendG]
  | (x, R) :: fr => by
    have ih := pendG_length_le fr
    simp only [pendG, List.length_cons, List.length_append]
    omega

/-- **`in_order_descend`.**  The left-spine push moves the subtree `idxs T0` onto the frame list:
from `Emb v.val cur T0` and `StackOK v.val stack.val fr`, its result `stack1` has frames `fr1` with
`StackOK v.val stack1.val fr1` and `pendG fr1 = idxs T0 ++ pendG fr`. -/
theorem in_order_descend (v : alloc.vec.Vec ONode) (T0 : OTree) (fr : List (Nat × OTree))
    (stack : alloc.vec.Vec U32) (cur : U32)
    (hE : Emb v.val cur T0) (hS : StackOK v.val stack.val fr)
    (stack1 : alloc.vec.Vec U32)
    (h : hashtrinity.overflow.Overflow.in_order_loop0_loop0 v stack cur = ok stack1) :
    ∃ fr1, StackOK v.val stack1.val fr1 ∧ pendG fr1 = idxs T0 ++ pendG fr := by
  unfold hashtrinity.overflow.Overflow.in_order_loop0_loop0 at h
  refine loop_pc (fun (p : alloc.vec.Vec U32 × U32) => ∃ Tc G, Emb v.val p.2 Tc ∧
      StackOK v.val p.1.val G ∧ idxs Tc ++ pendG G = idxs T0 ++ pendG fr)
    (fun r => ∃ G, StackOK v.val r.val G ∧ pendG G = idxs T0 ++ pendG fr) _ ?_
    (stack, cur) ⟨T0, fr, hE, hS, rfl⟩ stack1 h
  rintro ⟨st, c⟩ ⟨Tc, G, hEc, ⟨hst, hfr⟩, hp⟩ cf hcf
  dsimp only at hcf hEc hst
  unfold hashtrinity.overflow.Overflow.in_order_loop0_loop0.body at hcf
  by_cases hc : c = hashtrinity.overflow.NIL
  · rw [if_neg (by simp [hc])] at hcf
    have hcf' := (Result.ok_injective hcf).symm
    subst hcf'
    rw [hc] at hEc
    have hleaf := emb_nil hEc
    subst hleaf
    exact ⟨G, ⟨hst, hfr⟩, by simpa [idxs] using hp⟩
  · rw [if_pos (by simp [hc])] at hcf
    obtain ⟨st1, hst1, hcf⟩ := bind_inv hcf
    simp only [lift, bind_tc_ok] at hcf
    obtain ⟨n, hn, hcf⟩ := bind_inv hcf
    have hcf' := (Result.ok_injective hcf).symm
    subst hcf'
    obtain ⟨l, i, r', rfl, hci, nd, hnd, hl, hr⟩ := emb_ne_nil hEc hc
    have hn' := idx_ok hn
    rw [hci, hnd] at hn'
    have hnn := Option.some.inj hn'
    subst hnn
    refine ⟨l, (i, r') :: G, hl, ⟨?_, ?_⟩, ?_⟩
    · show st1.val.map _ = _
      rw [vec_push_ok _ _ _ hst1, List.map_append, hst]; simp [hci]
    · intro p hp'
      rcases List.mem_cons.mp hp' with rfl | hp'
      · exact ⟨nd, hnd, hr⟩
      · exact hfr p hp'
    · rw [← hp]; simp [idxs, pendG]

/-- A nonempty vector pops a `some` element. -/
theorem vec_pop_some_ex {α : Type} (v : alloc.vec.Vec α) (h : v.val.length ≠ 0) :
    ∃ y v', alloc.vec.Vec.pop Global v = ok (some y, v') := by
  obtain ⟨o, v', hpop⟩ := vec_pop_ex v
  rcases pop_spec v v' o hpop with ⟨h0, ho, _⟩ | ⟨y, hy, ho⟩
  · rw [h0] at h; exact absurd rfl h
  · exact ⟨y, v', by rw [ho] at hpop; exact hpop⟩

/-- **`in_order_pop_step`.**  The pop step of `in_order`'s outer loop: from a nonempty frame list it
decodes the popped node `nd` and its right subtree `R`, the shortened stack's frames `fr2`, and shows
the pending list loses exactly the popped node. -/
theorem in_order_pop_step (v : alloc.vec.Vec ONode) (stack1 : alloc.vec.Vec U32)
    (fr1 : List (Nat × OTree)) (T0 : OTree) (fr : List (Nat × OTree))
    (hS1 : StackOK v.val stack1.val fr1) (hp1 : pendG fr1 = idxs T0 ++ pendG fr)
    (hne : stack1.val.length ≠ 0) :
    ∃ (y : U32) (stack2 : alloc.vec.Vec U32) (R : OTree) (fr2 : List (Nat × OTree))
      (nd : ONode),
      alloc.vec.Vec.pop Global stack1 = ok (some y, stack2) ∧
      v.val[y.val]? = some nd ∧ Emb v.val nd.right R ∧ StackOK v.val stack2.val fr2 ∧
      pendG fr1 = y.val :: idxs R ++ pendG fr2 ∧
      (idxs T0).length + (pendG fr).length = 1 + ((idxs R).length + (pendG fr2).length) := by
  obtain ⟨y, stack2, hpop⟩ := vec_pop_some_ex stack1 hne
  have hsplit : stack1.val = stack2.val ++ [y] := by
    rcases pop_spec stack1 stack2 (some y) hpop with ⟨h1, h2, _⟩ | ⟨y0, h1, h2⟩
    · rw [h1] at hne; exact absurd rfl hne
    · have hy : y = y0 := Option.some.inj h2
      rw [← hy] at h1; exact h1
  have hrev : (fr1.map Prod.fst).reverse = stack2.val.map (fun x => x.val) ++ [y.val] := by
    rw [← hS1.1, hsplit, List.map_append]; rfl
  obtain ⟨R, fr2, hfr1, hfr2rev⟩ := stack_pop_frame hrev
  have hframe := hS1.2 (y.val, R) (by rw [hfr1]; exact List.mem_cons_self ..)
  obtain ⟨nd, hnd, hR⟩ := hframe
  have hlen : (idxs T0).length + (pendG fr).length =
      1 + ((idxs R).length + (pendG fr2).length) := by
    have hh := congrArg List.length hp1
    rw [hfr1] at hh
    simp only [pendG, List.length_cons, List.length_append] at hh
    omega
  refine ⟨y, stack2, R, fr2, nd, hpop, hnd, hR, ?_, ?_, hlen⟩
  · exact ⟨hfr2rev.symm, fun p hp => hS1.2 p (by rw [hfr1]; exact List.mem_cons_of_mem _ hp)⟩
  · rw [hfr1]; simp only [pendG]

/-- The `Vec.pop` + `Option.expect` prefix of `in_order_loop0.body` reduces to the expected node `y`
and the shortened stack `stack2`. -/
theorem in_order_body_reduce (v : alloc.vec.Vec ONode) (out stack2 : alloc.vec.Vec U32) (y : U32) :
    ((let (o, stack2') := (some y, stack2);
      (do
        let n ← core.option.Option.expect o (toStr "nonempty stack")
        let n1 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v
          (UScalar.cast .Usize n)
        let out1 ← if n1.alive then alloc.vec.Vec.push out n else ok out
        let n2 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v
          (UScalar.cast .Usize n)
        ok (ControlFlow.cont (out1, stack2', n2.right))))
      : Result (ControlFlow ((alloc.vec.Vec U32) × (alloc.vec.Vec U32) × U32) (alloc.vec.Vec U32)))
    = ((do
        let n1 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v
          (UScalar.cast .Usize y)
        let out1 ← if n1.alive then alloc.vec.Vec.push out y else ok out
        let n2 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v
          (UScalar.cast .Usize y)
        ok (ControlFlow.cont (out1, stack2, n2.right)))
      : Result (ControlFlow ((alloc.vec.Vec U32) × (alloc.vec.Vec U32) × U32) (alloc.vec.Vec U32))) :=
  by simp only [core.option.Option.expect, Result.ofOption, bind_tc_ok]

/-- **`in_order_loop0_total`.**  `Overflow::in_order`'s outer loop returns `ok`.  The pending work
`idxs T0 ++ pendG fr` drops by exactly one node at each `cont`; strong induction on the bound `W`
carries it. -/
theorem in_order_loop0_total (v : alloc.vec.Vec ONode) :
    ∀ (W : Nat) (out stack : alloc.vec.Vec U32) (cur : U32) (T0 : OTree)
      (fr : List (Nat × OTree)),
      Emb v.val cur T0 → StackOK v.val stack.val fr →
      (idxs T0).length + (pendG fr).length ≤ W →
      out.val.length + (idxs T0).length + (pendG fr).length < Usize.max →
      ∃ r, hashtrinity.overflow.Overflow.in_order_loop0 v out stack cur = ok r := by
  intro W
  induction W using Nat.strongRecOn with
  | ind N ih =>
    intro out stack cur T0 fr hE hS hb hlen
    have hslen : stack.val.length = fr.length := by
      have h1 := congrArg List.length hS.1
      rw [List.length_map, List.length_reverse, List.length_map] at h1
      exact h1
    have hpend : fr.length ≤ (pendG fr).length := pendG_length_le fr
    have hbdd : stack.val.length + (idxs T0).length < Usize.max := by
      have h1 : stack.val.length ≤ (pendG fr).length := by rw [hslen]; exact hpend
      omega
    have houtmax : out.val.length < Usize.max := by omega
    unfold hashtrinity.overflow.Overflow.in_order_loop0
    rw [loop.eq_def]
    dsimp only
    by_cases hcN : cur = hashtrinity.overflow.NIL
    · subst hcN
      by_cases he : stack.val.length = 0
      · have hbody : hashtrinity.overflow.Overflow.in_order_loop0.body v out stack
            hashtrinity.overflow.NIL = ok (ControlFlow.done out) := by
          unfold hashtrinity.overflow.Overflow.in_order_loop0.body
          rw [if_neg (by simp)]
          simp only [alloc.vec.Vec.is_empty, bind_tc_ok]
          rw [if_pos (by simp [he])]
        rw [hbody]; simp only [bind_ok]
        exact ⟨out, rfl⟩
      · have hfrne : fr ≠ [] := by
          intro hh; rw [hh] at hslen; simp only [List.length_nil] at hslen; exact he hslen
        have hpendne : pendG fr ≠ [] := by
          cases fr with
          | nil => exact absurd rfl hfrne
          | cons a fr' => simp [pendG]
        obtain ⟨stack1, hstack1₀⟩ := in_order_loop0_loop0_total v T0
          hashtrinity.overflow.NIL stack hE hbdd
        have hstack1 : hashtrinity.overflow.Overflow.in_order_loop0_loop1 v stack
            hashtrinity.overflow.NIL = ok stack1 := by
          rw [loop1_eq]; exact hstack1₀
        obtain ⟨fr1, hS1, hp1⟩ := in_order_descend v T0 fr stack hashtrinity.overflow.NIL
          hE hS stack1 hstack1₀
        have hfr1ne : fr1 ≠ [] := by
          intro hh
          have h2 : (idxs T0 ++ pendG fr).length = 0 := by rw [← hp1, hh]; rfl
          have h3 : 0 < (pendG fr).length := List.length_pos_of_ne_nil hpendne
          rw [List.length_append] at h2
          omega
        have hne : stack1.val.length ≠ 0 := by
          have h1 : stack1.val.length = fr1.length := by
            have h2 := congrArg List.length hS1.1
            rw [List.length_map, List.length_reverse, List.length_map] at h2
            exact h2
          rw [h1]; exact fun hh => hfr1ne (List.eq_nil_of_length_eq_zero hh)
        obtain ⟨y, stack2, R, fr2, nd, hpop, hnd, hR, hS2, hpend2, hpendlen⟩ :=
          in_order_pop_step v stack1 fr1 T0 fr hS1 hp1 hne
        have hyt : y.val < v.val.length := (List.getElem?_eq_some_iff.mp hnd).1
        have hndidx : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v
            (UScalar.cast .Usize y) = ok nd := by
          obtain ⟨m, hm⟩ := vec_index_ex v (UScalar.cast .Usize y)
            (by rw [cast_usize_u32]; exact hyt)
          have hmv := vec_index_ok v (UScalar.cast .Usize y) m hm
          rw [cast_usize_u32, hnd] at hmv
          rw [(Option.some.inj hmv).symm] at hm
          exact hm
        have hnewlt : (idxs R).length + (pendG fr2).length < N := by omega
        by_cases hal : nd.alive = true
        · obtain ⟨out1, hpush⟩ := vec_push_ex out y houtmax
          have hbody : hashtrinity.overflow.Overflow.in_order_loop0.body v out stack
              hashtrinity.overflow.NIL = ok (ControlFlow.cont (out1, stack2, nd.right)) := by
            unfold hashtrinity.overflow.Overflow.in_order_loop0.body
            rw [if_neg (by simp)]
            simp only [alloc.vec.Vec.is_empty, bind_tc_ok]
            rw [if_neg (by simp [he]), hstack1]
            simp only [bind_tc_ok]
            rw [hpop]
            simp only [lift, bind_tc_ok]
            change (match (some y, stack2) with
              | (o, stack2') => (do
                  let n ← core.option.Option.expect o (toStr "nonempty stack")
                  let n1 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v (UScalar.cast .Usize n)
                  let o1 ← if n1.alive then alloc.vec.Vec.push out n else ok out
                  let n2 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v (UScalar.cast .Usize n)
                  ok (ControlFlow.cont (o1, stack2', n2.right)))) =
              ok (ControlFlow.cont (out1, stack2, nd.right))
            rw [in_order_body_reduce]
            rw [hndidx]
            simp only [bind_tc_ok]
            rw [if_pos hal, hpush]
            simp only [bind_tc_ok]
          have hlen' : out1.val.length + (idxs R).length + (pendG fr2).length < Usize.max := by
            have h1 : out1.val.length = out.val.length + 1 := by
              rw [vec_push_ok out out1 y hpush]; simp
            omega
          have hr := ih ((idxs R).length + (pendG fr2).length) hnewlt out1 stack2 nd.right
            R fr2 hR hS2 le_rfl hlen'
          rw [hbody]; simp only [bind_ok]
          simpa only [hashtrinity.overflow.Overflow.in_order_loop0] using hr
        · have hbody : hashtrinity.overflow.Overflow.in_order_loop0.body v out stack
              hashtrinity.overflow.NIL = ok (ControlFlow.cont (out, stack2, nd.right)) := by
            unfold hashtrinity.overflow.Overflow.in_order_loop0.body
            rw [if_neg (by simp)]
            simp only [alloc.vec.Vec.is_empty, bind_tc_ok]
            rw [if_neg (by simp [he]), hstack1]
            simp only [bind_tc_ok]
            rw [hpop]
            simp only [lift, bind_tc_ok]
            change (match (some y, stack2) with
              | (o, stack2') => (do
                  let n ← core.option.Option.expect o (toStr "nonempty stack")
                  let n1 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v (UScalar.cast .Usize n)
                  let o1 ← if n1.alive then alloc.vec.Vec.push out n else ok out
                  let n2 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v (UScalar.cast .Usize n)
                  ok (ControlFlow.cont (o1, stack2', n2.right)))) =
              ok (ControlFlow.cont (out, stack2, nd.right))
            rw [in_order_body_reduce]
            rw [hndidx]
            simp only [bind_tc_ok]
            rw [if_neg hal]
            simp only [bind_tc_ok]
          have hlen' : out.val.length + (idxs R).length + (pendG fr2).length < Usize.max := by
            omega
          have hr := ih ((idxs R).length + (pendG fr2).length) hnewlt out stack2 nd.right
            R fr2 hR hS2 le_rfl hlen'
          rw [hbody]; simp only [bind_ok]
          simpa only [hashtrinity.overflow.Overflow.in_order_loop0] using hr
    · have hcond : (cur != hashtrinity.overflow.NIL) = true := by simpa using hcN
      obtain ⟨stack1, hstack1₀⟩ := in_order_loop0_loop0_total v T0 cur stack hE hbdd
      obtain ⟨fr1, hS1, hp1⟩ := in_order_descend v T0 fr stack cur hE hS stack1 hstack1₀
      have hidxT0 : idxs T0 ≠ [] := by
        cases T0 with
        | leaf => intro hh; exact hcN ((NIL_eq_iff cur).mpr hE)
        | node l i r =>
          intro hh
          have h2 : 0 < (idxs (OTree.node l i r)).length := by
            simp only [idxs, List.length_append, List.length_cons]; omega
          rw [hh] at h2; simp at h2
      have hpendne : pendG fr1 ≠ [] := by
        intro hh
        have h2 : (idxs T0 ++ pendG fr).length = 0 := by rw [← hp1, hh]; rfl
        have h3 : 0 < (idxs T0).length := List.length_pos_of_ne_nil hidxT0
        rw [List.length_append] at h2
        omega
      have hfr1ne : fr1 ≠ [] := by
        intro hh; rw [hh] at hpendne; simp [pendG] at hpendne
      have hne : stack1.val.length ≠ 0 := by
        have h1 : stack1.val.length = fr1.length := by
          have h2 := congrArg List.length hS1.1
          rw [List.length_map, List.length_reverse, List.length_map] at h2
          exact h2
        rw [h1]; exact fun hh => hfr1ne (List.eq_nil_of_length_eq_zero hh)
      obtain ⟨y, stack2, R, fr2, nd, hpop, hnd, hR, hS2, hpend2, hpendlen⟩ :=
        in_order_pop_step v stack1 fr1 T0 fr hS1 hp1 hne
      have hyt : y.val < v.val.length := (List.getElem?_eq_some_iff.mp hnd).1
      have hndidx : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v
          (UScalar.cast .Usize y) = ok nd := by
        obtain ⟨m, hm⟩ := vec_index_ex v (UScalar.cast .Usize y)
          (by rw [cast_usize_u32]; exact hyt)
        have hmv := vec_index_ok v (UScalar.cast .Usize y) m hm
        rw [cast_usize_u32, hnd] at hmv
        rw [(Option.some.inj hmv).symm] at hm
        exact hm
      have hnewlt : (idxs R).length + (pendG fr2).length < N := by omega
      by_cases hal : nd.alive = true
      · obtain ⟨out1, hpush⟩ := vec_push_ex out y houtmax
        have hbody : hashtrinity.overflow.Overflow.in_order_loop0.body v out stack cur =
            ok (ControlFlow.cont (out1, stack2, nd.right)) := by
          unfold hashtrinity.overflow.Overflow.in_order_loop0.body
          rw [if_pos hcond, hstack1₀]
          simp only [bind_tc_ok]
          rw [hpop]
          simp only [lift, bind_tc_ok]
          change (match (some y, stack2) with
            | (o, stack2') => (do
                let n ← core.option.Option.expect o (toStr "nonempty stack")
                let n1 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v (UScalar.cast .Usize n)
                let o1 ← if n1.alive then alloc.vec.Vec.push out n else ok out
                let n2 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v (UScalar.cast .Usize n)
                ok (ControlFlow.cont (o1, stack2', n2.right)))) =
            ok (ControlFlow.cont (out1, stack2, nd.right))
          rw [in_order_body_reduce]
          rw [hndidx]
          simp only [bind_tc_ok]
          rw [if_pos hal, hpush]
          simp only [bind_tc_ok]
        have hlen' : out1.val.length + (idxs R).length + (pendG fr2).length < Usize.max := by
          have h1 : out1.val.length = out.val.length + 1 := by
            rw [vec_push_ok out out1 y hpush]; simp
          omega
        have hr := ih ((idxs R).length + (pendG fr2).length) hnewlt out1 stack2 nd.right
          R fr2 hR hS2 le_rfl hlen'
        rw [hbody]; simp only [bind_ok]
        simpa only [hashtrinity.overflow.Overflow.in_order_loop0] using hr
      · have hbody : hashtrinity.overflow.Overflow.in_order_loop0.body v out stack cur =
            ok (ControlFlow.cont (out, stack2, nd.right)) := by
          unfold hashtrinity.overflow.Overflow.in_order_loop0.body
          rw [if_pos hcond, hstack1₀]
          simp only [bind_tc_ok]
          rw [hpop]
          simp only [lift, bind_tc_ok]
          change (match (some y, stack2) with
            | (o, stack2') => (do
                let n ← core.option.Option.expect o (toStr "nonempty stack")
                let n1 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v (UScalar.cast .Usize n)
                let o1 ← if n1.alive then alloc.vec.Vec.push out n else ok out
                let n2 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v (UScalar.cast .Usize n)
                ok (ControlFlow.cont (o1, stack2', n2.right)))) =
            ok (ControlFlow.cont (out, stack2, nd.right))
          rw [in_order_body_reduce]
          rw [hndidx]
          simp only [bind_tc_ok]
          rw [if_neg hal]
          simp only [bind_tc_ok]
        have hlen' : out.val.length + (idxs R).length + (pendG fr2).length < Usize.max := by
          omega
        have hr := ih ((idxs R).length + (pendG fr2).length) hnewlt out stack2 nd.right
          R fr2 hR hS2 le_rfl hlen'
        rw [hbody]; simp only [bind_ok]
        simpa only [hashtrinity.overflow.Overflow.in_order_loop0] using hr

/-- **`in_order_total`.**  With every node of the embedded tree alive, `Overflow::in_order` from the
empty output returns `ok` and emits exactly `idxs T`. -/
theorem in_order_total (o : Ov) (T : OTree)
    (hE : Emb o.nodes.val o.root T)
    (hal : ∀ i ∈ idxs T, ∃ nd, o.nodes.val[i]? = some nd ∧ nd.alive = true)
    (hb : (idxs T).length < Usize.max) :
    ∃ r, hashtrinity.overflow.Overflow.in_order CI o (alloc.vec.Vec.new U32) = ok r ∧
      r.val.map (fun x => x.val) = idxs T := by
  have hS : StackOK o.nodes.val (alloc.vec.Vec.new U32).val [] :=
    ⟨by simp [alloc.vec.Vec.new], by intro p hp; simp at hp⟩
  obtain ⟨r, hr⟩ := in_order_loop0_total o.nodes (idxs T).length (alloc.vec.Vec.new U32)
    (alloc.vec.Vec.new U32) o.root T [] hE hS (by simp [pendG]) (by simpa [pendG] using hb)
  refine ⟨r, ?_, ?_⟩
  · unfold hashtrinity.overflow.Overflow.in_order; exact hr
  · exact in_order_spec o T hE hal r (by unfold hashtrinity.overflow.Overflow.in_order; exact hr)

/-- **`into_entries_loop0_loop0_total`.**  The per-node entry scan returns `ok`, by the decreasing
measure `iter.val.length`. -/
theorem into_entries_loop0_loop0_total {K V : Type}
    (iter : alloc.vec.into_iter.IntoIter (K × Option V)) (out : alloc.vec.Vec (U64 × K × V))
    (key : U64) (hb : out.val.length + iter.val.length < Usize.max) :
    ∃ r, hashtrinity.overflow.Overflow.into_entries_loop0_loop0 iter out key = ok r := by
  unfold hashtrinity.overflow.Overflow.into_entries_loop0_loop0
  refine loop_tot_inv
    (fun p : alloc.vec.into_iter.IntoIter (K × Option V) × alloc.vec.Vec (U64 × K × V) =>
      hashtrinity.overflow.Overflow.into_entries_loop0_loop0.body key p.1 p.2)
    (fun p => p.2.val.length + p.1.val.length < Usize.max)
    (fun p => p.1.val.length) ?_ ?_ (iter, out) hb
  · rintro ⟨it, o⟩ hinv
    dsimp only at hinv
    obtain ⟨o1, it1, hn⟩ := next_ex it
    unfold hashtrinity.overflow.Overflow.into_entries_loop0_loop0.body
    rw [hn]; simp only [bind_tc_ok]
    rcases next_spec _ _ _ hn with ⟨hnil, rfl⟩ | ⟨e, hcons, rfl⟩
    · exact ⟨_, rfl⟩
    · obtain ⟨k, v⟩ := e
      cases v with
      | none => exact ⟨_, rfl⟩
      | some v1 =>
        have hlt : o.val.length < Usize.max := by
          rw [hcons] at hinv; simp only [List.length_cons] at hinv; omega
        obtain ⟨o1', hpush⟩ := vec_push_ex o (key, k, v1) hlt
        exact ⟨ControlFlow.cont (it1, o1'), by
          show (do
            let out1 ← alloc.vec.Vec.push o (key, k, v1)
            ok (ControlFlow.cont (it1, out1))) = ok (ControlFlow.cont (it1, o1'))
          rw [hpush]; simp only [bind_tc_ok]⟩
  · rintro ⟨it, o⟩ ⟨it', o'⟩ hinv hcf
    dsimp only at hinv
    unfold hashtrinity.overflow.Overflow.into_entries_loop0_loop0.body at hcf
    obtain ⟨⟨o1, it1⟩, hn, hcf⟩ := bind_inv hcf
    rcases next_spec _ _ _ hn with ⟨hnil, rfl⟩ | ⟨e, hcons, rfl⟩
    · exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
    · obtain ⟨k, v⟩ := e
      cases v with
      | none =>
        have hh := Result.ok_injective hcf
        simp only [ControlFlow.cont.injEq, Prod.mk.injEq] at hh
        refine ⟨?_, ?_⟩
        · rw [hh.2.symm, hh.1.symm]
          show o.val.length + it1.val.length < Usize.max
          rw [hcons] at hinv; simp only [List.length_cons] at hinv; omega
        · rw [hh.1.symm]; rw [hcons]; simp only [List.length_cons]; omega
      | some v1 =>
        obtain ⟨o1', hpush, hcf⟩ := bind_inv hcf
        have hh := Result.ok_injective hcf
        simp only [ControlFlow.cont.injEq, Prod.mk.injEq] at hh
        refine ⟨?_, ?_⟩
        · rw [hh.2.symm, hh.1.symm, vec_push_ok o o1' (key, k, v1) hpush]
          rw [hcons] at hinv
          simp only [List.length_append, List.length_cons, List.length_nil] at hinv ⊢
          omega
        · rw [hh.1.symm]; rw [hcons]; simp only [List.length_cons]; omega

/-! ## `Overflow::into_entries` totality

`into_entries_loop0` walks the node indices `in_order` emits; for each it moves that node's `entries`
out (`take` leaves the default empty vector), scans them (`into_entries_loop0_loop0`), and continues
at the next index.  The decrease is the iterator length; the bound is the remaining entry count
`remEntries`, which the scan drains.  Because `take` empties the visited node, `remEntries` is
computed on the updated vector `v2`; the step only ever lowers it, so no duplicate-free side
condition on `iter` is used. -/

/-- The entry count of arena node `j`, or `0` off the arena. -/
def remLen (v : alloc.vec.Vec ONode) (j : Nat) : Nat :=
  (v.val[j]?.map (fun nd => nd.entries.val.length)).getD 0

/-- The remaining entry count: the sum of `remLen` over the node indices `iter` has not yet
visited. -/
def remEntries (v : alloc.vec.Vec ONode) (iter : alloc.vec.into_iter.IntoIter U32) : Nat :=
  (iter.val.map (fun x => remLen v x.val)).sum

/-- `remEntries` step: with the head node present, the sum splits into its entry count and the
tail's. -/
theorem remEntries_cons (v : alloc.vec.Vec ONode) (n : U32) (node : ONode)
    (it1 it : alloc.vec.into_iter.IntoIter U32) (hnode : v.val[n.val]? = some node)
    (h : it.val = n :: it1.val) :
    remEntries v it = node.entries.val.length + remEntries v it1 := by
  have hf : remLen v n.val = node.entries.val.length := by simp [remLen, hnode]
  unfold remEntries
  rw [h, List.map_cons, List.sum_cons, hf]

/-- `remEntries` is monotone in the per-index entry counts. -/
theorem remEntries_mono (v v' : alloc.vec.Vec ONode) (it : alloc.vec.into_iter.IntoIter U32)
    (h : ∀ j, remLen v' j ≤ remLen v j) : remEntries v' it ≤ remEntries v it := by
  unfold remEntries
  generalize it.val = l
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [List.map_cons, List.sum_cons]
    exact Nat.add_le_add (h x.val) ih

/-- The `take`-emptied node only lowers `remEntries`: replacing the node at `n` by `node'`, of no
more entries, cannot raise the remaining sum. -/
theorem remEntries_set_le (v v' : alloc.vec.Vec ONode) (n : Nat) (node' : ONode)
    (it : alloc.vec.into_iter.IntoIter U32) (hval : v'.val = v.val.set n node')
    (hlen : node'.entries.val.length ≤ remLen v n) :
    remEntries v' it ≤ remEntries v it := by
  apply remEntries_mono
  intro j
  simp only [remLen]
  rw [hval]
  by_cases hj : j = n
  · rw [hj]
    by_cases hlt : n < v.val.length
    · rw [List.getElem?_set_self hlt]
      exact hlen
    · rw [List.getElem?_set_self', List.getElem?_eq_none_iff.mpr (by omega)]
      simp
  · rw [List.getElem?_set_ne (Ne.symm hj)]

/-- The entry scan of `into_entries_loop0` grows the output by at most its input length. -/
theorem into_entries_loop0_loop0_len {K V : Type}
    (iter : alloc.vec.into_iter.IntoIter (K × Option V)) (out : alloc.vec.Vec (U64 × K × V))
    (key : U64) (r : alloc.vec.Vec (U64 × K × V))
    (h : hashtrinity.overflow.Overflow.into_entries_loop0_loop0 iter out key = ok r) :
    r.val.length ≤ out.val.length + iter.val.length := by
  unfold hashtrinity.overflow.Overflow.into_entries_loop0_loop0 at h
  refine loop_pc
    (fun p : alloc.vec.into_iter.IntoIter (K × Option V) × alloc.vec.Vec (U64 × K × V) =>
      p.2.val.length + p.1.val.length ≤ out.val.length + iter.val.length)
    (fun r => r.val.length ≤ out.val.length + iter.val.length) _ ?_ (iter, out) (by simp) r h
  rintro ⟨it, o⟩ hinv cf hcf
  try dsimp only at hcf hinv
  unfold hashtrinity.overflow.Overflow.into_entries_loop0_loop0.body at hcf
  obtain ⟨⟨x, it1⟩, hn, hcf⟩ := bind_inv hcf
  rcases next_spec _ _ _ hn with ⟨hnil, rfl⟩ | ⟨e, hcons, rfl⟩
  · have hcfe : ControlFlow.done o = cf := Result.ok_injective hcf
    rw [← hcfe]
    show o.val.length ≤ out.val.length + iter.val.length
    rw [hnil] at hinv
    simpa using hinv
  · obtain ⟨k, v⟩ := e
    cases v with
    | none =>
      have hcfe : ControlFlow.cont (it1, o) = cf := Result.ok_injective hcf
      rw [← hcfe]
      show o.val.length + it1.val.length ≤ out.val.length + iter.val.length
      rw [hcons] at hinv
      simp only [List.length_cons] at hinv ⊢
      omega
    | some v1 =>
      obtain ⟨o1, hpush, hcf⟩ := bind_inv hcf
      have hcfe : ControlFlow.cont (it1, o1) = cf := Result.ok_injective hcf
      rw [← hcfe]
      show o1.val.length + it1.val.length ≤ out.val.length + iter.val.length
      rw [vec_push_ok o o1 (key, k, v1) hpush]
      rw [hcons] at hinv
      simp only [List.length_append, List.length_cons, List.length_nil] at hinv ⊢
      omega

/-- `mem::take` of a `Vec` returns the original vector and the empty default. -/
theorem mem_take_vec_ok {T : Type} (x : alloc.vec.Vec T) :
    core.mem.take (alloc.vec.Vec.Insts.CoreDefaultDefault T) x = ok (x, alloc.vec.Vec.new T) ∧
      (alloc.vec.Vec.new T).val = [] := by
  refine ⟨?_, by simp [alloc.vec.Vec.new]⟩
  simp only [core.mem.take, alloc.vec.Vec.Insts.CoreDefaultDefault.default, bind_tc_ok]

/-- **`into_entries_loop0_total`.**  The outer entry-collection loop returns `ok`: it visits each
node index once, draining the bound `remEntries`, so no `Vec::push` and no `index_mut` fails.  The
decreasing measure is the iterator length (`loop_tot_inv`). -/
theorem into_entries_loop0_total (iter : alloc.vec.into_iter.IntoIter U32)
    (v : alloc.vec.Vec ONode) (out : alloc.vec.Vec (U64 × U64 × I64))
    (hvalid : ∀ j ∈ iter.val, j.val < v.val.length)
    (hbound : out.val.length + remEntries v iter < Usize.max) :
    ∃ r, hashtrinity.overflow.Overflow.into_entries_loop0 iter v out = ok r := by
  unfold hashtrinity.overflow.Overflow.into_entries_loop0
  refine loop_tot_inv
    (fun p : alloc.vec.into_iter.IntoIter U32 × alloc.vec.Vec ONode ×
        alloc.vec.Vec (U64 × U64 × I64) =>
      hashtrinity.overflow.Overflow.into_entries_loop0.body p.1 p.2.1 p.2.2)
    (fun p => (∀ j ∈ p.1.val, j.val < p.2.1.val.length) ∧
      p.2.2.val.length + remEntries p.2.1 p.1 < Usize.max)
    (fun p => p.1.val.length) ?_ ?_ (iter, v, out) ⟨hvalid, hbound⟩
  · rintro ⟨it, vv, oo⟩ ⟨hval, hbdd⟩
    dsimp only at *
    obtain ⟨o, it1, hn⟩ := next_ex it
    unfold hashtrinity.overflow.Overflow.into_entries_loop0.body
    rw [hn]; simp only [bind_tc_ok]
    rcases next_spec _ _ _ hn with ⟨hnil, rfl⟩ | ⟨n, hcons, rfl⟩
    · exact ⟨_, rfl⟩
    · change ∃ cf, (do
        let i ← lift (UScalar.cast .Usize n)
        let (node', index_mut_back) ← alloc.vec.Vec.index_mut
          (core.slice.index.SliceIndexUsizeSlice (hashtrinity.overflow.Node U64 I64)) vv i
        let (entries, v1) ← core.mem.take
          (alloc.vec.Vec.Insts.CoreDefaultDefault (U64 × Option I64)) node'.entries
        let iter2 ← alloc.vec.IntoIteratorVec.into_iter entries
        let out1' ← hashtrinity.overflow.Overflow.into_entries_loop0_loop0 iter2 oo node'.key
        let v2 := index_mut_back { node' with entries := v1 }
        ok (ControlFlow.cont (it1, v2, out1'))) = ok cf
      have hnv : n.val < vv.val.length := hval n (by rw [hcons]; exact List.mem_cons_self ..)
      obtain ⟨node, back, hidx⟩ := vec_index_mut_ex vv (UScalar.cast .Usize n)
        (by rw [cast_usize_u32]; exact hnv)
      obtain ⟨hnode, hback⟩ := idxm_ok hidx
      have htake : core.mem.take (alloc.vec.Vec.Insts.CoreDefaultDefault (U64 × Option I64))
            node.entries = ok (node.entries, alloc.vec.Vec.new (U64 × Option I64)) :=
        (mem_take_vec_ok node.entries).1
      have hiter2 : alloc.vec.IntoIteratorVec.into_iter node.entries
          = (ok node.entries : Result (alloc.vec.into_iter.IntoIter (U64 × Option I64))) := rfl
      have hrec : remEntries vv it = node.entries.val.length + remEntries vv it1 :=
        remEntries_cons vv n node it1 it hnode hcons
      have hb2 : oo.val.length + node.entries.val.length < Usize.max := by
        have := hbdd; rw [hrec] at this; omega
      obtain ⟨out1, hout1⟩ := into_entries_loop0_loop0_total node.entries oo node.key hb2
      refine ⟨ControlFlow.cont (it1,
        back { node with entries := alloc.vec.Vec.new (U64 × Option I64) }, out1), ?_⟩
      simp only [lift, bind_tc_ok]
      rw [hidx, bind_pair_ok]
      rw [htake, bind_pair_ok]
      rw [hiter2, bind_tc_ok]
      rw [hout1, bind_tc_ok]
  · rintro ⟨it, vv, oo⟩ ⟨it', vv', oo'⟩ ⟨hval, hbdd⟩ hcf
    dsimp only at *
    unfold hashtrinity.overflow.Overflow.into_entries_loop0.body at hcf
    obtain ⟨o, it1, hn⟩ := next_ex it
    rw [hn] at hcf
    rw [bind_pair_ok] at hcf
    rcases next_spec _ _ _ hn with ⟨hnil, rfl⟩ | ⟨n, hcons, rfl⟩
    · dsimp only at hcf
      exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
    · dsimp only at hcf
      simp only [lift, bind_tc_ok] at hcf
      obtain ⟨node, back, hidx⟩ := vec_index_mut_ex vv (UScalar.cast .Usize n)
        (by rw [cast_usize_u32]; exact hval n (by rw [hcons]; exact List.mem_cons_self ..))
      obtain ⟨hnode, hback⟩ := idxm_ok hidx
      rw [hidx, bind_pair_ok] at hcf
      rw [(mem_take_vec_ok node.entries).1, bind_pair_ok] at hcf
      have hiter2 : alloc.vec.IntoIteratorVec.into_iter node.entries
          = (ok node.entries : Result (alloc.vec.into_iter.IntoIter (U64 × Option I64))) := rfl
      rw [hiter2, bind_tc_ok] at hcf
      obtain ⟨out1, hout1, hcf⟩ := bind_inv hcf
      try dsimp only at hcf
      have hh := Result.ok_injective hcf
      simp only [ControlFlow.cont.injEq, Prod.mk.injEq] at hh
      obtain ⟨rfl, rfl, rfl⟩ := hh
      try dsimp only at *
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · intro j hj
        have hj' : j ∈ it.val := by rw [hcons]; exact List.mem_cons_of_mem n hj
        rw [hback _, List.length_set]
        exact hval j hj'
      · have hv1 : (alloc.vec.Vec.new (U64 × Option I64)).val = [] :=
          (mem_take_vec_ok node.entries).2
        have hrec : remEntries vv it = node.entries.val.length + remEntries vv it1 :=
          remEntries_cons vv n node it1 it hnode hcons
        have hout1len : out1.val.length ≤ oo.val.length + node.entries.val.length :=
          into_entries_loop0_loop0_len node.entries oo node.key out1 hout1
        have hlen1 : ({ node with entries := alloc.vec.Vec.new (U64 × Option I64) } : ONode).entries.val.length
            ≤ remLen vv n.val := by
          have h0 : ({ node with entries := alloc.vec.Vec.new (U64 × Option I64) } : ONode).entries.val.length
              = 0 := by simp [hv1]
          rw [h0]; exact Nat.zero_le _
        have hset : remEntries (back { node with entries := alloc.vec.Vec.new (U64 × Option I64) }) it1
            ≤ remEntries vv it1 :=
          remEntries_set_le vv (back { node with entries := alloc.vec.Vec.new (U64 × Option I64) })
            n.val _ it1 (hback _) hlen1
        have := hbdd; rw [hrec] at this; omega
      · have : it.val.length = it1.val.length + 1 := by rw [hcons]; simp
        omega

/-- The outer loop's output is at most the initial output plus the remaining entry count; it also
preserves node validity. -/
theorem into_entries_loop0_len (iter : alloc.vec.into_iter.IntoIter U32)
    (v : alloc.vec.Vec ONode) (out : alloc.vec.Vec (U64 × U64 × I64))
    (hvalid : ∀ j ∈ iter.val, j.val < v.val.length)
    (r : alloc.vec.Vec (U64 × U64 × I64))
    (h : hashtrinity.overflow.Overflow.into_entries_loop0 iter v out = ok r) :
    r.val.length ≤ out.val.length + remEntries v iter := by
  unfold hashtrinity.overflow.Overflow.into_entries_loop0 at h
  refine loop_pc
    (fun p : alloc.vec.into_iter.IntoIter U32 × alloc.vec.Vec ONode ×
        alloc.vec.Vec (U64 × U64 × I64) =>
      (∀ j ∈ p.1.val, j.val < p.2.1.val.length) ∧
        p.2.2.val.length + remEntries p.2.1 p.1 ≤ out.val.length + remEntries v iter)
    (fun r => r.val.length ≤ out.val.length + remEntries v iter) _ ?_
    (iter, v, out) ⟨hvalid, by simp⟩ r h
  rintro ⟨it, vv, oo⟩ ⟨hval, hinv⟩ cf hcf
  try dsimp only at hval hinv
  try dsimp only at hcf
  unfold hashtrinity.overflow.Overflow.into_entries_loop0.body at hcf
  obtain ⟨o, it1, hn⟩ := next_ex it
  rw [hn] at hcf
  rw [bind_pair_ok] at hcf
  rcases next_spec _ _ _ hn with ⟨hnil, rfl⟩ | ⟨n, hcons, rfl⟩
  · dsimp only at hcf
    have hcfe : ControlFlow.done oo = cf := Result.ok_injective hcf
    rw [← hcfe]
    show oo.val.length ≤ out.val.length + remEntries v iter
    have hz : remEntries vv it = 0 := by unfold remEntries; rw [hnil]; simp
    rw [hz] at hinv; omega
  · dsimp only at hcf
    simp only [lift, bind_tc_ok] at hcf
    obtain ⟨node, back, hidx⟩ := vec_index_mut_ex vv (UScalar.cast .Usize n)
      (by rw [cast_usize_u32]; exact hval n (by rw [hcons]; exact List.mem_cons_self ..))
    obtain ⟨hnode, hback⟩ := idxm_ok hidx
    rw [hidx, bind_pair_ok] at hcf
    rw [(mem_take_vec_ok node.entries).1, bind_pair_ok] at hcf
    have hiter2 : alloc.vec.IntoIteratorVec.into_iter node.entries
          = (ok node.entries : Result (alloc.vec.into_iter.IntoIter (U64 × Option I64))) := rfl
    rw [hiter2, bind_tc_ok] at hcf
    obtain ⟨out1, hout1, hcf⟩ := bind_inv hcf
    try dsimp only at hcf
    have hcfe : ControlFlow.cont (it1,
        back { node with entries := alloc.vec.Vec.new (U64 × Option I64) }, out1) = cf :=
      Result.ok_injective hcf
    rw [← hcfe]
    refine ⟨?_, ?_⟩
    · intro j hj
      have hj' : j ∈ it.val := by rw [hcons]; exact List.mem_cons_of_mem n hj
      rw [hback _, List.length_set]
      exact hval j hj'
    · show out1.val.length + remEntries (back { node with entries := alloc.vec.Vec.new (U64 × Option I64) }) it1
          ≤ out.val.length + remEntries v iter
      have hv1 : (alloc.vec.Vec.new (U64 × Option I64)).val = [] := (mem_take_vec_ok node.entries).2
      have hrec : remEntries vv it = node.entries.val.length + remEntries vv it1 :=
        remEntries_cons vv n node it1 it hnode hcons
      have hout1len : out1.val.length ≤ oo.val.length + node.entries.val.length :=
        into_entries_loop0_loop0_len node.entries oo node.key out1 hout1
      have hlen1 : ({ node with entries := alloc.vec.Vec.new (U64 × Option I64) } : ONode).entries.val.length
          ≤ remLen vv n.val := by
        have h0 : ({ node with entries := alloc.vec.Vec.new (U64 × Option I64) } : ONode).entries.val.length = 0 := by
          simp [hv1]
        rw [h0]; exact Nat.zero_le _
      have hset : remEntries (back { node with entries := alloc.vec.Vec.new (U64 × Option I64) }) it1
          ≤ remEntries vv it1 :=
        remEntries_set_le vv (back { node with entries := alloc.vec.Vec.new (U64 × Option I64) })
          n.val _ it1 (hback _) hlen1
      rw [hrec] at hinv
      omega

/-- **`into_entries_total`.**  On an embedded tree whose nodes are alive, `Overflow::into_entries`
returns `ok`, with the output no longer than the tree's total entry count. -/
theorem into_entries_total (o : Ov) (T : OTree)
    (hE : Emb o.nodes.val o.root T)
    (hal : ∀ i ∈ idxs T, ∃ nd, o.nodes.val[i]? = some nd ∧ nd.alive = true)
    (hb : (idxs T).length + ((idxs T).map (remLen o.nodes)).sum < Usize.max) :
    ∃ es, hashtrinity.overflow.Overflow.into_entries CI o = ok es ∧
      es.val.length ≤ ((idxs T).map (remLen o.nodes)).sum := by
  have hb_len : (idxs T).length < Usize.max := by omega
  obtain ⟨order, horder, hordv⟩ := in_order_total o T hE hal hb_len
  have hvalid : ∀ j ∈ order.val, j.val < o.nodes.val.length := by
    intro j hj
    have hjT : j.val ∈ idxs T := by
      rw [← hordv]; exact List.mem_map_of_mem hj
    obtain ⟨nd, hnd, _⟩ := emb_mem hE j.val hjT
    exact (List.getElem?_eq_some_iff.mp hnd).1
  have hsum : remEntries o.nodes order = ((idxs T).map (remLen o.nodes)).sum := by
    unfold remEntries
    rw [← hordv]
    apply congrArg List.sum
    rw [List.map_map]
    exact List.map_congr_left (fun x _ => rfl)
  have hout0 : (alloc.vec.Vec.with_capacity (U64 × U64 × I64) o.len).val.length = 0 := by
    simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val]
  have hout : (alloc.vec.Vec.with_capacity (U64 × U64 × I64) o.len).val.length
      + remEntries o.nodes order < Usize.max := by
    rw [hout0, hsum]; omega
  obtain ⟨es, hes⟩ := into_entries_loop0_total order o.nodes
    (alloc.vec.Vec.with_capacity (U64 × U64 × I64) o.len) hvalid hout
  have heslen : es.val.length ≤ ((idxs T).map (remLen o.nodes)).sum := by
    have := into_entries_loop0_len order o.nodes
      (alloc.vec.Vec.with_capacity (U64 × U64 × I64) o.len) hvalid es hes
    rw [hout0, hsum] at this
    simpa using this
  refine ⟨es, ?_, heslen⟩
  · unfold hashtrinity.overflow.Overflow.into_entries
    rw [horder]; simp only [bind_tc_ok]
    unfold alloc.vec.IntoIteratorVec.into_iter
    simp only [bind_tc_ok]
    exact hes

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.resize_loop0_ok
#audit Apeiron.MapExecuted.resize_loop1_ok
#audit Apeiron.MapExecuted.resize_loop0_witness
#audit Apeiron.MapExecuted.shl_1_ex
#audit Apeiron.MapExecuted.with_bits_ex
#audit Apeiron.MapExecuted.in_order_loop0_loop0_total
#audit Apeiron.MapExecuted.pendG_length_le
#audit Apeiron.MapExecuted.in_order_descend
#audit Apeiron.MapExecuted.vec_pop_some_ex
#audit Apeiron.MapExecuted.in_order_pop_step
#audit Apeiron.MapExecuted.in_order_loop0_total
#audit Apeiron.MapExecuted.in_order_total
#audit Apeiron.MapExecuted.into_entries_loop0_loop0_total
#audit Apeiron.MapExecuted.remLen
#audit Apeiron.MapExecuted.remEntries
#audit Apeiron.MapExecuted.remEntries_cons
#audit Apeiron.MapExecuted.remEntries_mono
#audit Apeiron.MapExecuted.remEntries_set_le
#audit Apeiron.MapExecuted.into_entries_loop0_loop0_len
#audit Apeiron.MapExecuted.mem_take_vec_ok
#audit Apeiron.MapExecuted.into_entries_loop0_total
#audit Apeiron.MapExecuted.into_entries_loop0_len
#audit Apeiron.MapExecuted.into_entries_total
