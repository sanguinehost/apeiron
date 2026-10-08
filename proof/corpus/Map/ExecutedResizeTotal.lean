-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (HT-INSERT-W17): totality of
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

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.resize_loop0_ok
#audit Apeiron.MapExecuted.resize_loop1_ok
#audit Apeiron.MapExecuted.resize_loop0_witness
#audit Apeiron.MapExecuted.shl_1_ex
#audit Apeiron.MapExecuted.with_bits_ex
#audit Apeiron.MapExecuted.in_order_loop0_loop0_total
