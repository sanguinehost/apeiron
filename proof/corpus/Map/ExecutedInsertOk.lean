-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (HT-INSERT-W23): Stage 3,
-- the executed insert is total.
import Map.ExecutedGrowTotal
open Aeneas Aeneas.Std Result

/-!
# Totality of the executed insert

`insert_refines` / `coord_insert_refines` (`Map/ExecutedResize.lean`) take the premise
`word_hashtrinity_insert s k v = ok s'`.  This module discharges it: under the representation and the
overflow domain the resize carries, the executed bodies return `ok`.

* `grow_post`: the post-state of one `grow_for_one` call — the representation, the shape, the arena
  bound and the counter-form overflow domain `Hov`.
* `insert_ok` / `coord_insert_ok`: `∃ s', word_hashtrinity_insert s k v = ok s'`, the `Stored<u64>`
  and `Regen<u64>` faces.
* `insert_witness`: a reachable joint witness for the `Regen<u64>` face.

The pivot is `Hov` (`Map/ExecutedInsertTotal.lean:2560`), the counter-form overflow domain.  A
`resize` writes its destination's overflow by `resize_loop1`, a fold of `place_capped`;
`place_capped_hov` advances the counter by one per call, so the resized table satisfies
`Hov · (mainCount s + es.length)` (`resize_loop1_ok_hov`), whence `grow_post` gives the counter the
first insert reads.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-- `Hov` is monotone in its counter. -/
theorem Hov_mono {KS : Type} {t : Tbl KS} {n m : Nat} (h : Hov t n) (hnm : n ≤ m) : Hov t m := by
  intro ov hsome
  obtain ⟨T, hI, hn, hl, he⟩ := h ov hsome
  exact ⟨T, hI, by omega, by omega, fun j hj nd hnd => by have := he j hj nd hnd; omega⟩

/-- **`resize_loop1_ok_hov`.**  `resize_loop1` returns `ok` and its destination carries the
counter-form overflow domain with the counter advanced by one per re-placed entry.  This exports the
`done`-case `Hov` of `resize_loop1_ok`'s invariant (`ExecutedResizeTotal.lean`), which that theorem
discards; the exit domain is read off `loop_pc` on the same invariant. -/
theorem resize_loop1_ok_hov {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
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
    ∃ r, hashtrinity.Table.resize_loop1 core.cmp.EqU64 CI ks iter t0 = ok r ∧
      Hov r (n0 + es.length) := by
  obtain ⟨r, hr⟩ :=
    resize_loop1_ok ks hm hA hset es hes n0 hbits0 hlen0 hsl0 hBig0 hlenbnd0 hHov0 hcap iter hiter
  refine ⟨r, hr, ?_⟩
  unfold hashtrinity.Table.resize_loop1 at hr
  refine loop_pc
    (fun p : alloc.vec.into_iter.IntoIter (U64 × U64 × I64) × Tbl KS =>
      ∃ dn, dn ++ p.1.val = es ∧
        0 < p.2.bits.val ∧ p.2.coords.val.length = 2 ^ p.2.bits.val ∧
        p.2.vals.val.length = p.2.coords.val.length ∧ BigInit p.2 ∧
        p.2.len.val ≤ n0 + dn.length ∧ Hov p.2 (n0 + dn.length))
    (fun r => Hov r (n0 + es.length))
    (fun p => hashtrinity.Table.resize_loop1.body core.cmp.EqU64 CI ks p.1 p.2)
    ?_ (iter, t0) ?_ r hr
  · rintro ⟨it, tb⟩ ⟨dn, hdn, hbits, hlen, hsl, hBig, hlenbnd, hHov⟩ cf hcf
    dsimp only at *
    unfold hashtrinity.Table.resize_loop1.body at hcf
    obtain ⟨⟨o, itr⟩, hn, hcf⟩ := bind_inv hcf
    rcases next_spec it itr o hn with ⟨hnil, rfl⟩ | ⟨e, hcons, rfl⟩
    · -- `next` is exhausted: the body returns `done tb` unchanged.
      have hcfd : cf = ControlFlow.done tb := by
        change ok (ControlFlow.done tb) = ok cf at hcf
        exact (Result.ok_injective hcf).symm
      rw [hcfd]
      have hdne : dn = es := by rw [hnil] at hdn; simpa using hdn
      rw [hdne] at hHov
      exact hHov
    · -- one more entry: `place_capped` advances the counter by one.
      obtain ⟨c0, k0, v0⟩ := e
      have hmem : (c0, k0, v0) ∈ es := by rw [← hdn]; simp [hcons]
      obtain ⟨hk2, hns2⟩ := hes _ hmem
      have hk2' : k0 = c0 := hk2
      subst k0
      obtain ⟨tb2, hpc, hcf⟩ := bind_inv hcf
      have hcfv : cf = ControlFlow.cont (itr, tb2) := (Result.ok_injective hcf).symm
      rw [hcfv]
      dsimp only at *
      have hdn' : dn ++ [(c0, c0, v0)] ++ itr.val = es := by rw [← hdn, hcons]; simp
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
      refine ⟨dn ++ [(c0, c0, v0)], hdn', ?_, ?_, ?_, ?_, ?_, hHov'⟩
      · rw [hbf]; exact hbits
      · rw [hlf2, hbf]; exact hlen
      · rw [hslf2, hlf2]; exact hsl
      · exact place_capped_bigInit ks tb tb2 c0 v0 hpc hBig
      · rw [hcnt]; omega
  · refine ⟨[], by simpa using hiter, hbits0, hlen0, hsl0, hBig0, ?_, ?_⟩
    · simpa using hlenbnd0
    · simpa using hHov0

/-- **`resize_hov`.**  `resize` preserves the counter-form overflow domain, raising the counter to at
most `mainCount s + (entry count)`; with `Hov s n`, `mainCount s ≤ n` and `n*n + n + 4 ≤ nilN`, the
destination satisfies `Hov · (n*n + n)`.  `resize_loop0_ok_hov_strong` gives the destination's main
half `Hov · (mainCount s)`; `resize_loop1_ok_hov` the overflow half `Hov · (mainCount s + es.length)`;
`into_entries_total` and `Hov s n` bound `es.length` by `n*n`. -/
theorem resize_hov {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), ∃ keys', ks.set keys i k = ok keys')
    (hTK : ∀ (keys : KS) (i : Usize) (c : U64), ¬ IsSent c →
      ∃ keys', ks.take_key keys i c = ok (c, keys'))
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (new_bits : U32) (hbits : new_bits.val < UScalarTy.Usize.numBits) (hpos : 0 < new_bits.val)
    (n : Nat) (hHov : Hov s n) (hmain : mainCount s ≤ n) (hn : n * n + n + 4 ≤ nilN)
    {s1 : Tbl KS} (hres : hashtrinity.Table.resize core.cmp.EqU64 CI ks s new_bits = ok s1) :
    Hov s1 (n * n + n) := by
  have key : ∀ ov T, s.overflow = some ov → OInv ov T →
      (idxs T).length ≤ n ∧ ((idxs T).map (remLen ov.nodes)).sum ≤ n * n := by
    intro ov T hsome hI
    obtain ⟨T', hI', hnodes', -, hent'⟩ := hHov ov hsome
    have hTT : T' = T := emb_unique hI'.1 hI.1
    subst T'
    have hlenT : (idxs T).length ≤ ov.nodes.val.length :=
      idxs_length_le (nd_nodup T hI.2.2.1) (fun j hj => by
        obtain ⟨nd, hnd, -⟩ := emb_mem hI.1 j hj
        exact (List.getElem?_eq_some_iff.mp hnd).1)
    refine ⟨le_trans hlenT hnodes', ?_⟩
    have hrem : ∀ j ∈ idxs T, remLen ov.nodes j ≤ n := by
      intro j hj
      unfold remLen
      cases hnd : ov.nodes.val[j]? with
      | none => simp
      | some nd => simpa [hnd] using hent' j hj nd hnd
    exact le_trans (sum_map_le_mul _ _ n hrem) (Nat.mul_le_mul_right n (le_trans hlenT hnodes'))
  unfold hashtrinity.Table.resize at hres
  obtain ⟨bigger, hwb, hbitsb, hlenbb, hslbb, hlenb0, hovnone, hBigb, hHov0b⟩ :=
    with_bits_ex ks halloc new_bits hbits
  rw [hwb] at hres; simp only [bind_tc_ok] at hres
  have hb0C : CoreRep ks bigger m (fun _ => False) :=
    rep_map_congr ks (m' := m) (with_bits_rep ks new_bits hpos bigger hwb) (fun _ h => h.elim)
  have hb0 : RepW ks bigger m (LowMain s 0) := by
    refine repW_live_congr ks ⟨hb0C, by unfold OvWF; rw [hovnone]; trivial⟩ (fun c => ?_)
    exact ⟨False.elim, fun ⟨_, d, hd, _⟩ => absurd hd (Nat.not_lt_zero d)⟩
  have hcb0 : arenaLen bigger + mainCount bigger = 0 := by
    have h1 : arenaLen bigger = 0 := by unfold arenaLen; rw [hovnone]
    rw [h1, mainCount_zero ks hb0C]
  obtain ⟨r, hr, hbr, hcr, hvr, hBigr, hlenr, hHovr, hside, hovf⟩ :=
    resize_loop0_ok_hov_strong ks hm hA hset hTK hR hcap bigger hb0 hcb0 hovnone
  rw [hr] at hres; simp only [bind_tc_ok] at hres
  obtain ⟨ss1, hss1, hres⟩ := bind_inv hres
  rcases r with ⟨ss0, o, b1⟩
  simp only [core.mem.replace, core.option.Option.take] at hres
  cases o with
  | none =>
    have hs1 : Hov s1 (mainCount s) := by
      rw [(Result.ok_injective hres).symm]
      exact hHovr
    exact Hov_mono hs1 (by omega)
  | some ov =>
    have hso : s.overflow = some ov := hovf.symm
    have hov : ∀ ov T, s.overflow = some ov → OInv ov T →
        (idxs T).length + ((idxs T).map (remLen ov.nodes)).sum < Usize.max := by
      intro ov T hs hI
      obtain ⟨h1, h2⟩ := key ov T hs hI
      have hsum : (idxs T).length + ((idxs T).map (remLen ov.nodes)).sum ≤ n + n * n := by omega
      have hlt : n + n * n < Usize.max := by
        have h := nilN_le_usize_max
        unfold nilN at h
        have hn' : n * n + n + 4 ≤ 2 ^ 32 - 1 := hn
        omega
      omega
    have hvcap : ∀ ov T, s.overflow = some ov → OInv ov T →
        mainCount s + ((idxs T).map (remLen ov.nodes)).sum + 4 ≤ nilN := by
      intro ov T hs hI
      obtain ⟨h1, h2⟩ := key ov T hs hI
      omega
    obtain ⟨T, hF, -⟩ := ov_full_of_some ks hR hso
    obtain ⟨hI, hall⟩ := hF
    have hEmb : Emb ov.nodes.val ov.root T := hI.1
    have hND : ND T := hI.2.2.1
    have hal : ∀ i ∈ idxs T, ∃ nd, ov.nodes.val[i]? = some nd ∧ nd.alive = true :=
      fun i hi => by obtain ⟨nd, hnd, ha, -, -, -⟩ := hall i hi; exact ⟨nd, hnd, ha⟩
    have hspec : ∀ i ∈ idxs T, ∃ nd, ov.nodes.val[i]? = some nd ∧ nd.alive = true ∧
        nd.entries.val = [(nd.key, some (m nd.key))] :=
      fun i hi => by obtain ⟨nd, hnd, ha, he, -, -⟩ := hall i hi; exact ⟨nd, hnd, ha, he⟩
    obtain ⟨es, hesIn, heslen⟩ := into_entries_total ov T hEmb hal (hov ov T hso hI)
    have hesv := into_entries_spec m ov T hEmb hND hspec es hesIn
    have hes : ∀ x ∈ es.val, x.2.1 = x.1 ∧ ¬ IsSent x.1 := by
      intro x hx
      have hx' : x ∈ (idxs T).map (entOf ov.nodes.val m) := by rw [← hesv]; exact hx
      obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hx'
      obtain ⟨nd, hnd, -, hent, hns, -⟩ := hall j hj
      exact ⟨by simp [entOf, hnd], by simpa [entOf, hnd] using hns⟩
    change (do
      let v ← hashtrinity.overflow.Overflow.into_entries CI ov
      let iter ← alloc.vec.IntoIteratorVec.into_iter v
      hashtrinity.Table.resize_loop1 core.cmp.EqU64 CI ks iter
        { b1 with side := ss0 }) = ok s1 at hres
    rw [hesIn] at hres; simp only [bind_tc_ok] at hres
    obtain ⟨iter, hiter, hres⟩ := bind_inv hres
    unfold alloc.vec.IntoIteratorVec.into_iter at hiter
    have hies : iter = es := (Result.ok_injective hiter).symm
    have hcap1 : mainCount s + es.val.length + 4 ≤ nilN := by
      have := hvcap ov T hso hI
      omega
    obtain ⟨r2, hr2, hHov2⟩ :=
      resize_loop1_ok_hov ks hm hA hset es.val hes (t0 := { b1 with side := ss0 }) (mainCount s)
        hbr hcr hvr hBigr hlenr hHovr hcap1 iter (by rw [hies])
    have hs1eq : s1 = r2 := Result.ok_injective (hres.symm.trans hr2)
    subst hs1eq
    have hlen2 : es.val.length ≤ n * n := le_trans heslen (key ov T hso hI).2
    exact Hov_mono hHov2 (by omega)

/-- **`grow_post`.**  The post-state of one `grow_for_one` call: the representation (via `grow_rep`,
under `ResizeOK`), the shape, the arena bound and the counter-form overflow domain `Hov · (n*n + n)`.
The counter is the input counter raised by the widest amount one `resize` can add — its destination's
main half plus the drained entries (`resize_hov`).  The `else` branch leaves the table unchanged. -/
theorem grow_post {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64) (KC : Slice U64 → KS → Prop)
    (hRes : ResizeOK ks KC)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), ∃ keys', ks.set keys i k = ok keys')
    (hTK : ∀ (keys : KS) (i : Usize) (c : U64), ¬ IsSent c →
      ∃ keys', ks.take_key keys i c = ok (c, keys'))
    {s s1 : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (n : Nat) (hHov : Hov s n) (hmain : mainCount s ≤ n) (hn : n * n + n + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length)
    (hg : hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok s1) :
    RepW ks s1 m live ∧ 0 < s1.bits.val ∧ s1.coords.length = 2 ^ s1.bits.val ∧
      s1.vals.length = s1.coords.length ∧
      arenaLen s1 + mainCount s1 + 4 ≤ nilN ∧ Hov s1 (n * n + n) := by
  have hsmall : arenaLen s + mainCount s < nilN := by omega
  obtain ⟨hR1, -, hle1⟩ := grow_rep ks KC hRes hR hK hsmall hg
  refine ⟨hR1, hR1.1.bits_pos, hR1.1.len_pow, hR1.1.vals_len, by omega, ?_⟩
  unfold hashtrinity.Table.grow_for_one at hg
  obtain ⟨i, -, hg⟩ := bind_inv hg
  obtain ⟨i1, -, hg⟩ := bind_inv hg
  obtain ⟨i2, -, hg⟩ := bind_inv hg
  obtain ⟨i3, -, hg⟩ := bind_inv hg
  by_cases hge : i2 >= i3
  · rw [if_pos hge] at hg
    obtain ⟨i4, -, hg⟩ := bind_inv hg
    obtain ⟨nb, hnb, hg⟩ := bind_inv hg
    have hbits_nb : nb.val < UScalarTy.Usize.numBits := by
      by_cases hlt : s.len < i4
      · rw [if_pos hlt] at hnb
        have := (Result.ok_injective hnb).symm
        subst this
        omega
      · rw [if_neg hlt] at hnb
        have hnbv := (uadd_ok hnb).1
        have h1 : (1#u32).val = 1 := rfl
        omega
    have hpos : 0 < nb.val := by
      by_cases hlt : s.len < i4
      · rw [if_pos hlt] at hnb
        have := (Result.ok_injective hnb).symm
        subst this
        exact hR.1.bits_pos
      · rw [if_neg hlt] at hnb
        have hnbv := (uadd_ok hnb).1
        have h1 : (1#u32).val = 1 := rfl
        omega
    exact resize_hov ks halloc hm hA hset hTK hR hcap nb hbits_nb hpos n hHov hmain hn hg
  · rw [if_neg hge] at hg
    have hs1 : s1 = s := (Result.ok_injective hg).symm
    subst hs1
    exact Hov_mono hHov (by omega)

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.Hov_mono
#audit Apeiron.MapExecuted.resize_loop1_ok_hov
#audit Apeiron.MapExecuted.resize_hov
#audit Apeiron.MapExecuted.grow_post
