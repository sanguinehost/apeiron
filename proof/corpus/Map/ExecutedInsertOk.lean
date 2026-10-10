-- Stage 3,
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
    (sz : KS → Nat)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (es : List (U64 × U64 × I64))
    (hes : ∀ x ∈ es, x.2.1 = x.1 ∧ ¬ IsSent x.1)
    {t0 : Tbl KS} (n0 e0 : Nat)
    (hbits0 : 0 < t0.bits.val) (hlen0 : t0.coords.val.length = 2 ^ t0.bits.val)
    (hsl0 : t0.vals.val.length = t0.coords.val.length)
    (hBig0 : BigInit t0) (hlenbnd0 : t0.len.val ≤ n0) (hHov0 : Hov t0 n0)
    (hent0 : Hent t0 e0)
    (hcap : n0 + es.length + 4 ≤ nilN) (hsz0 : t0.coords.val.length ≤ sz t0.keys)
    (iter : alloc.vec.into_iter.IntoIter (U64 × U64 × I64)) (hiter : iter.val = es) :
    ∃ r, hashtrinity.Table.resize_loop1 core.cmp.EqU64 CI ks iter t0 = ok r ∧
      Hov r (n0 + es.length) ∧ Hent r (e0 + es.length) ∧ r.coords.val.length ≤ sz r.keys := by
  obtain ⟨r, hr⟩ :=
    resize_loop1_ok ks sz hm hA hset es hes n0 hbits0 hlen0 hsl0 hBig0 hlenbnd0 hHov0 hcap hsz0
      iter hiter
  refine ⟨r, hr, ?_⟩
  unfold hashtrinity.Table.resize_loop1 at hr
  refine loop_pc
    (fun p : alloc.vec.into_iter.IntoIter (U64 × U64 × I64) × Tbl KS =>
      ∃ dn, dn ++ p.1.val = es ∧
        0 < p.2.bits.val ∧ p.2.coords.val.length = 2 ^ p.2.bits.val ∧
        p.2.vals.val.length = p.2.coords.val.length ∧ BigInit p.2 ∧
        p.2.len.val ≤ n0 + dn.length ∧ Hov p.2 (n0 + dn.length) ∧
        Hent p.2 (e0 + dn.length) ∧
        p.2.coords.val.length ≤ sz p.2.keys)
    (fun r => Hov r (n0 + es.length) ∧ Hent r (e0 + es.length) ∧
      r.coords.val.length ≤ sz r.keys)
    (fun p => hashtrinity.Table.resize_loop1.body core.cmp.EqU64 CI ks p.1 p.2)
    ?_ (iter, t0) ?_ r hr
  · rintro ⟨it, tb⟩ ⟨dn, hdn, hbits, hlen, hsl, hBig, hlenbnd, hHov, hHent, hsz⟩ cf hcf
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
      rw [hdne] at hHov hHent
      exact ⟨hHov, hHent, hsz⟩
    · -- one more entry: `place_capped` advances both counters by one.
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
      obtain ⟨hHov0', hHent0', hszb'⟩ := place_capped_hent ks sz hm hA hset tb tb2 c0 v0
        (n0 + dn.length) (e0 + dn.length) hbits hlen hsl
        (fun j hcj => hBig j c0 hcj hns2) hlenmax hHov hHent hK hsz hpc
      have hHov' : Hov tb2 (n0 + (dn ++ [(c0, c0, v0)]).length) := by
        rw [hcnt]; exact hHov0'
      have hHent' : Hent tb2 (e0 + (dn ++ [(c0, c0, v0)]).length) := by
        rw [hcnt]; exact hHent0'
      refine ⟨dn ++ [(c0, c0, v0)], hdn', ?_, ?_, ?_, ?_, ?_, hHov', hHent', ?_⟩
      · rw [hbf]; exact hbits
      · rw [hlf2, hbf]; exact hlen
      · rw [hslf2, hlf2]; exact hsl
      · exact place_capped_bigInit ks tb tb2 c0 v0 hpc hBig
      · rw [hcnt]; omega
      · rw [hlf2, hszb']; exact hsz
  · refine ⟨[], by simpa using hiter, hbits0, hlen0, hsl0, hBig0, ?_, ?_, ?_, ?_⟩
    · simpa using hlenbnd0
    · simpa using hHov0
    · simpa using hent0
    · simpa using hsz0

/-- **`resize_hov`.**  `resize` preserves the counter-form overflow domain, raising the counter to at
most `mainCount s + (entry count)`; with `Hov s n`, `mainCount s ≤ n` and `n*n + n + 4 ≤ nilN`, the
destination satisfies `Hov · (n*n + n)`.  `resize_loop0_ok_hov_strong` gives the destination's main
half `Hov · (mainCount s)`; `resize_loop1_ok_hov` the overflow half `Hov · (mainCount s + es.length)`;
`into_entries_total` and `Hov s n` bound `es.length` by `n*n`. -/
theorem resize_hov {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (sz : KS → Nat)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks' ∧ n.val ≤ sz ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (KC : Slice U64 → KS → Prop) (hTK : TakeKeyOK ks sz KC)
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys)
    (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (new_bits : U32) (hbits : new_bits.val < UScalarTy.Usize.numBits) (hpos : 0 < new_bits.val)
    (n : Nat) (hHov : Hov s n) (hmain : mainCount s ≤ n) (hn : n * n + n + 4 ≤ nilN)
    (hsz_s : s.coords.val.length ≤ sz s.keys)
    {s1 : Tbl KS} (hres : hashtrinity.Table.resize core.cmp.EqU64 CI ks s new_bits = ok s1) :
    Hov s1 (n * n + n) ∧ s1.coords.val.length ≤ sz s1.keys := by
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
  obtain ⟨bigger, hwb, hbitsb, hlenbb, hslbb, hlenb0, hovnone, hBigb, hHov0b, hszbb⟩ :=
    with_bits_ex ks sz halloc new_bits hbits
  rw [hwb] at hres; simp only [bind_tc_ok] at hres
  have hb0C : CoreRep ks bigger m (fun _ => False) :=
    rep_map_congr ks (m' := m) (with_bits_rep ks new_bits hpos bigger hwb) (fun _ h => h.elim)
  have hb0 : RepW ks bigger m (LowMain s 0) := by
    refine repW_live_congr ks ⟨hb0C, by unfold OvWF; rw [hovnone]; trivial⟩ (fun c => ?_)
    exact ⟨False.elim, fun ⟨_, d, hd, _⟩ => absurd hd (Nat.not_lt_zero d)⟩
  have hcb0 : arenaLen bigger + mainCount bigger = 0 := by
    have h1 : arenaLen bigger = 0 := by unfold arenaLen; rw [hovnone]
    rw [h1, mainCount_zero ks hb0C]
  obtain ⟨r, hr, hbr, hcr, hvr, hBigr, hlenr, hHovr, hHentr, hside, hovf, hszr⟩ :=
    resize_loop0_ok_hov_strong ks sz hm hA hset KC hTK hR hK hcap bigger hb0 hcb0 hovnone 0
      (Hent_none hovnone) hsz_s hszbb
  rw [hr] at hres; simp only [bind_tc_ok] at hres
  obtain ⟨ss1, hss1, hres⟩ := bind_inv hres
  rcases r with ⟨ss0, o, b1⟩
  simp only [core.mem.replace, core.option.Option.take] at hres
  cases o with
  | none =>
    have hs1 : Hov s1 (mainCount s) := by
      rw [(Result.ok_injective hres).symm]
      exact hHovr
    exact ⟨Hov_mono hs1 (by omega), by rw [(Result.ok_injective hres).symm]; exact hszr⟩
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
    obtain ⟨r2, hr2, hHov2, -, hsz2⟩ :=
      resize_loop1_ok_hov ks sz hm hA hset es.val hes (t0 := { b1 with side := ss0 })
        (mainCount s) (0 + mainCount s)
        hbr hcr hvr hBigr hlenr hHovr
        (show Hent { b1 with side := ss0 } (0 + mainCount s) from hHentr)
        hcap1 (by simpa using hszr) iter (by rw [hies])
    have hs1eq : s1 = r2 := Result.ok_injective (hres.symm.trans hr2)
    subst hs1eq
    have hlen2 : es.val.length ≤ n * n := le_trans heslen (key ov T hso hI).2
    exact ⟨Hov_mono hHov2 (by omega), hsz2⟩

/-- **`resize_hov_lin`.**  `resize` preserves the counter-form overflow domain and the entry counter
linearly: from `Hov s n` and `Hent s e`, the destination satisfies `Hov · (n + e)` and
`Hent · (n + e)`.  The `n · n` of `resize_hov` becomes `e`, because `resize` re-places exactly the
live overflow entry count, bounded by `Hent s e` (`es.length ≤ e`, `into_entries_total`). -/
theorem resize_hov_lin {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (sz : KS → Nat)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks' ∧ n.val ≤ sz ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (KC : Slice U64 → KS → Prop) (hTK : TakeKeyOK ks sz KC)
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys)
    (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (new_bits : U32) (hbits : new_bits.val < UScalarTy.Usize.numBits) (hpos : 0 < new_bits.val)
    (n e : Nat) (hHov : Hov s n) (hent : Hent s e) (hmain : mainCount s ≤ n)
    (hne : n + e + 4 ≤ nilN)
    (hsz_s : s.coords.val.length ≤ sz s.keys)
    {s1 : Tbl KS} (hres : hashtrinity.Table.resize core.cmp.EqU64 CI ks s new_bits = ok s1) :
    Hov s1 (n + e) ∧ Hent s1 (n + e) ∧ s1.coords.val.length ≤ sz s1.keys := by
  have hlen : ∀ ov T, s.overflow = some ov → OInv ov T → (idxs T).length ≤ n := by
    intro ov T hsome hI
    obtain ⟨T', hI', hnodes', -, -⟩ := hHov ov hsome
    have hTT : T' = T := emb_unique hI'.1 hI.1
    subst T'
    exact le_trans (idxs_length_le (nd_nodup T hI.2.2.1) (fun j hj => by
      obtain ⟨nd, hnd, -⟩ := emb_mem hI.1 j hj
      exact (List.getElem?_eq_some_iff.mp hnd).1)) hnodes'
  unfold hashtrinity.Table.resize at hres
  obtain ⟨bigger, hwb, hbitsb, hlenbb, hslbb, hlenb0, hovnone, hBigb, hHov0b, hszbb⟩ :=
    with_bits_ex ks sz halloc new_bits hbits
  rw [hwb] at hres; simp only [bind_tc_ok] at hres
  have hb0C : CoreRep ks bigger m (fun _ => False) :=
    rep_map_congr ks (m' := m) (with_bits_rep ks new_bits hpos bigger hwb) (fun _ h => h.elim)
  have hb0 : RepW ks bigger m (LowMain s 0) := by
    refine repW_live_congr ks ⟨hb0C, by unfold OvWF; rw [hovnone]; trivial⟩ (fun c => ?_)
    exact ⟨False.elim, fun ⟨_, d, hd, _⟩ => absurd hd (Nat.not_lt_zero d)⟩
  have hcb0 : arenaLen bigger + mainCount bigger = 0 := by
    have h1 : arenaLen bigger = 0 := by unfold arenaLen; rw [hovnone]
    rw [h1, mainCount_zero ks hb0C]
  obtain ⟨r, hr, hbr, hcr, hvr, hBigr, hlenr, hHovr, hHentr, hside, hovf, hszr⟩ :=
    resize_loop0_ok_hov_strong ks sz hm hA hset KC hTK hR hK hcap bigger hb0 hcb0 hovnone 0
      (Hent_none hovnone) hsz_s hszbb
  rw [hr] at hres; simp only [bind_tc_ok] at hres
  obtain ⟨ss1, hss1, hres⟩ := bind_inv hres
  rcases r with ⟨ss0, o, b1⟩
  simp only [core.mem.replace, core.option.Option.take] at hres
  cases o with
  | none =>
    have hs1 : Hov s1 (mainCount s) := by
      rw [(Result.ok_injective hres).symm]
      exact hHovr
    have he1 : Hent s1 (0 + mainCount s) := by
      rw [(Result.ok_injective hres).symm]
      exact hHentr
    exact ⟨Hov_mono hs1 (by omega), Hent_mono he1 (by omega),
      by rw [(Result.ok_injective hres).symm]; exact hszr⟩
  | some ov =>
    have hso : s.overflow = some ov := hovf.symm
    have hov : ∀ ov T, s.overflow = some ov → OInv ov T →
        (idxs T).length + ((idxs T).map (remLen ov.nodes)).sum < Usize.max := by
      intro ov T hs hI
      have h1 := hlen ov T hs hI
      have h2 := hent ov T hs hI
      have h3 : n + e < Usize.max := by
        have h := nilN_le_usize_max
        unfold nilN at h
        have hne' : n + e + 4 ≤ 2 ^ 32 - 1 := hne
        omega
      omega
    have hvcap : ∀ ov T, s.overflow = some ov → OInv ov T →
        mainCount s + ((idxs T).map (remLen ov.nodes)).sum + 4 ≤ nilN := by
      intro ov T hs hI
      have h2 := hent ov T hs hI
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
    obtain ⟨r2, hr2, hHov2, hHent2, hsz2⟩ :=
      resize_loop1_ok_hov ks sz hm hA hset es.val hes (t0 := { b1 with side := ss0 })
        (mainCount s) (0 + mainCount s)
        hbr hcr hvr hBigr hlenr hHovr
        (show Hent { b1 with side := ss0 } (0 + mainCount s) from hHentr)
        hcap1 (by simpa using hszr) iter (by rw [hies])
    have hs1eq : s1 = r2 := Result.ok_injective (hres.symm.trans hr2)
    subst hs1eq
    have hlen2 : es.val.length ≤ e := le_trans heslen (hent ov T hso hI)
    exact ⟨Hov_mono hHov2 (by omega), Hent_mono hHent2 (by omega), hsz2⟩

/-- **`grow_post`.**  The post-state of one `grow_for_one` call: the representation (via `grow_rep`,
under `ResizeOK`), the shape, the arena bound and the counter-form overflow domain `Hov · (n*n + n)`.
The counter is the input counter raised by the widest amount one `resize` can add — its destination's
main half plus the drained entries (`resize_hov`).  The `else` branch leaves the table unchanged. -/
theorem grow_post {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64) (KC : Slice U64 → KS → Prop)
    (hRes : ResizeOK ks KC)
    (sz : KS → Nat)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks' ∧ n.val ≤ sz ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (hTK : TakeKeyOK ks sz KC)
    {s s1 : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (n : Nat) (hHov : Hov s n) (hmain : mainCount s ≤ n) (hn : n * n + n + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length)
    (hsz_s : s.coords.val.length ≤ sz s.keys)
    (hg : hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok s1) :
    RepW ks s1 m live ∧ 0 < s1.bits.val ∧ s1.coords.length = 2 ^ s1.bits.val ∧
      s1.vals.length = s1.coords.length ∧
      arenaLen s1 + mainCount s1 + 4 ≤ nilN ∧ Hov s1 (n * n + n) ∧
      s1.coords.val.length ≤ sz s1.keys ∧ KC s1.coords s1.keys ∧
      mainCount s1 ≤ 2 * n := by
  have hsmall : arenaLen s + mainCount s < nilN := by omega
  obtain ⟨hR1, hK1, hle1⟩ := grow_rep ks KC hRes hR hK hsmall hg
  have har : arenaLen s ≤ n := by
    unfold arenaLen
    cases hso : s.overflow with
    | none => simp
    | some ov => obtain ⟨T, _, hnodes, _, _⟩ := hHov ov hso; exact hnodes
  have hmc1 : mainCount s1 ≤ 2 * n := by omega
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
    obtain ⟨hHov1, hsz1⟩ := resize_hov ks sz halloc hm hA hset KC hTK hR hK hcap nb hbits_nb hpos n
      hHov hmain hn hsz_s hg
    exact ⟨hHov1, hsz1, hK1, hmc1⟩
  · rw [if_neg hge] at hg
    have hs1 : s1 = s := (Result.ok_injective hg).symm
    subst hs1
    exact ⟨Hov_mono hHov (by omega), hsz_s, hK, hmc1⟩

/-- **`grow_post_lin`.**  `grow_post` with the entry counter: from `Hov s n` and `Hent s e`, one
`grow_for_one` call returns a table with `Hov · (n + e)`, `Hent · (n + e)` (the resize branch uses
`resize_hov_lin`; the no-resize branch is the identity). -/
theorem grow_post_lin {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64) (KC : Slice U64 → KS → Prop)
    (hRes : ResizeOK ks KC)
    (sz : KS → Nat)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks' ∧ n.val ≤ sz ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (hTK : TakeKeyOK ks sz KC)
    {s s1 : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (n e : Nat) (hHov : Hov s n) (hent : Hent s e) (hmain : mainCount s ≤ n)
    (hne : n + e + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length)
    (hsz_s : s.coords.val.length ≤ sz s.keys)
    (hg : hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok s1) :
    RepW ks s1 m live ∧ 0 < s1.bits.val ∧ s1.coords.length = 2 ^ s1.bits.val ∧
      s1.vals.length = s1.coords.length ∧
      arenaLen s1 + mainCount s1 + 4 ≤ nilN ∧ Hov s1 (n + e) ∧ Hent s1 (n + e) ∧
      s1.coords.val.length ≤ sz s1.keys ∧ KC s1.coords s1.keys ∧
      mainCount s1 ≤ 2 * n := by
  have hsmall : arenaLen s + mainCount s < nilN := by omega
  obtain ⟨hR1, hK1, hle1⟩ := grow_rep ks KC hRes hR hK hsmall hg
  have har : arenaLen s ≤ n := by
    unfold arenaLen
    cases hso : s.overflow with
    | none => simp
    | some ov => obtain ⟨T, _, hnodes, _, _⟩ := hHov ov hso; exact hnodes
  have hmc1 : mainCount s1 ≤ 2 * n := by omega
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
    obtain ⟨hHov1, hHent1, hsz1⟩ := resize_hov_lin ks sz halloc hm hA hset KC hTK hR hK hcap nb
      hbits_nb hpos n e hHov hent hmain hne hsz_s hg
    exact ⟨hHov1, hHent1, hsz1, hK1, hmc1⟩
  · rw [if_neg hge] at hg
    have hs1 : s1 = s := (Result.ok_injective hg).symm
    subst hs1
    exact ⟨Hov_mono hHov (by omega), Hent_mono hent (by omega), hsz_s, hK, hmc1⟩

/-! ## Frames: `Hov`/`Hent` under the shape-preserving operations, and the `bits`/`tombs` frames

The probe write's `Hit`/`Vacant` verdicts and the overflow `take_value`/`write_back` pair change
neither the overflow's node count, its `len`, nor any entry-vector length, so they preserve the
counter invariants `Hov`/`Hent`.  `bits` and `tombs` are untouched by `place_capped`, by `resize`'s
loops, and by a `resize` they are `nb` and `0`. -/

/-- `Hov` reads only the table's overflow store. -/
theorem Hov_of_overflow {KS : Type} {s s' : Tbl KS} {N : Nat} (h : s'.overflow = s.overflow) :
    Hov s' N ↔ Hov s N := by
  unfold Hov; rw [h]

/-- `Hent` reads only the table's overflow store. -/
theorem Hent_of_overflow {KS : Type} {s s' : Tbl KS} {e : Nat} (h : s'.overflow = s.overflow) :
    Hent s' e ↔ Hent s e := by
  unfold Hent; rw [h]

/-- `mainCount` reads only the table's coordinate vector. -/
theorem mainCount_of_coords {KS : Type} {s s' : Tbl KS} (h : s'.coords = s.coords) :
    mainCount s' = mainCount s := by
  unfold mainCount; rw [h]

/-- `Overflow::take_value` leaves the store's `len` unchanged. -/
theorem take_value_len (o : Ov) (n e : U32) (old : I64) (o' : Ov)
    (h : hashtrinity.overflow.Overflow.take_value CI o n e = ok (old, o')) :
    o'.len = o.len := by
  unfold hashtrinity.overflow.Overflow.take_value at h
  simp only [lift, bind_tc_ok] at h
  obtain ⟨⟨n1, back⟩, _, h⟩ := bind_inv h
  obtain ⟨⟨⟨t, oo⟩, back1⟩, _, h⟩ := bind_inv h
  obtain ⟨t1, _, h⟩ := bind_inv h
  have hh := Result.ok_injective h
  simp only [Prod.mk.injEq] at hh
  have h2 : o' = { o with nodes := back { n1 with entries := back1 (t, none) } } := hh.2.symm
  rw [h2]

/-- `Overflow::write_back` leaves the store's `len` unchanged. -/
theorem write_back_len (o : Ov) (n e : U32) (v : I64) (o' : Ov)
    (h : hashtrinity.overflow.Overflow.write_back CI o n e v = ok o') :
    o'.len = o.len := by
  unfold hashtrinity.overflow.Overflow.write_back at h
  simp only [lift, bind_tc_ok] at h
  obtain ⟨⟨n1, back⟩, _, h⟩ := bind_inv h
  obtain ⟨⟨⟨t, oo⟩, back1⟩, _, h⟩ := bind_inv h
  have h2 : o' = { o with nodes := back { n1 with entries := back1 (t, some v) } } :=
    (Result.ok_injective h).symm
  rw [h2]

/-- `Table::place_capped` never writes `bits` or `tombs`. -/
theorem place_capped_bits_tombs {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (t t' : Tbl KS) (c k : U64) (v : I64)
    (h : hashtrinity.Table.place_capped core.cmp.EqU64 CI ks t c k v = ok t') :
    t'.bits = t.bits ∧ t'.tombs = t.tombs := by
  unfold hashtrinity.Table.place_capped at h
  obtain ⟨r, hr, h⟩ := bind_inv h
  cases r with
  | Blocked =>
    obtain ⟨bo, -, h⟩ := bind_inv h
    unfold core.option.Option.as_mut at h
    obtain ⟨⟨o1, backo⟩, -, h⟩ := bind_inv h
    cases o1 with
    | none => exact absurd h (by intro hh; exact ok_not_fail hh.symm)
    | some ov =>
      change (do
        let (_, ov1) ← hashtrinity.overflow.Overflow.insert_new_at CI ov c k v
        ok ({ t with overflow := backo (some ov1) } : Tbl KS)) = ok t' at h
      obtain ⟨a, hins, h⟩ := bind_inv h
      rcases a with ⟨_, ov1⟩
      have ht'' : t' = {t with overflow := backo (some ov1)} := (Result.ok_injective h).symm
      subst ht''
      exact ⟨rfl, rfl⟩
  | Hit i =>
    obtain ⟨⟨mu, back⟩, hg, h⟩ := bind_inv h
    obtain ⟨old, -, h⟩ := bind_inv h
    obtain ⟨-, -, h⟩ := bind_inv h
    obtain ⟨mu1, hnew, h⟩ := bind_inv h
    obtain ⟨⟨x2, back1⟩, hg1, h⟩ := bind_inv h
    have ht' : t' = {t with vals := back1 mu1} := (Result.ok_injective h).symm
    subst ht'
    exact ⟨rfl, rfl⟩
  | Vacant i =>
    obtain ⟨⟨x0, backc⟩, hgc, h⟩ := bind_inv h
    obtain ⟨mu, hnew, h⟩ := bind_inv h
    obtain ⟨⟨y, backv⟩, hgv, h⟩ := bind_inv h
    obtain ⟨tk, hset, h⟩ := bind_inv h
    obtain ⟨i1, hi1, h⟩ := bind_inv h
    have ht' : t' = {t with coords := backc c, vals := backv mu, keys := tk, len := i1} :=
      (Result.ok_injective h).symm
    subst ht'
    exact ⟨rfl, rfl⟩

/-- `Table::resize_loop0` never writes its destination's `bits` or `tombs`. -/
theorem resize_loop0_bits_tombs {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (s b0 : Tbl KS)
    {r : hashtrinity.side.SideStore U64 I64 KS × Option Ov × Tbl KS}
    (h : hashtrinity.Table.resize_loop0 core.cmp.EqU64 CI ks s b0 0#usize = ok r) :
    r.2.2.bits = b0.bits ∧ r.2.2.tombs = b0.tombs := by
  unfold hashtrinity.Table.resize_loop0 at h
  refine loop_pc (fun (p : Tbl KS × Tbl KS × Usize) =>
      p.2.1.bits = b0.bits ∧ p.2.1.tombs = b0.tombs)
    (fun r => r.2.2.bits = b0.bits ∧ r.2.2.tombs = b0.tombs)
    (fun p => hashtrinity.Table.resize_loop0.body core.cmp.EqU64 CI ks p.1 p.2.1 p.2.2)
    ?_ (s, b0, 0#usize) ⟨rfl, rfl⟩ r h
  rintro ⟨t, b, i⟩ ⟨hb, ht⟩ cf hcf
  dsimp only at hcf
  unfold hashtrinity.Table.resize_loop0.body at hcf
  by_cases hlt : i < Slice.len t.coords
  · rw [if_pos hlt] at hcf
    obtain ⟨c, hc, hcf⟩ := bind_inv hcf
    obtain ⟨bl, hbl, hcf⟩ := bind_inv hcf
    rw [is_live_ok] at hbl
    have hbl' := (Result.ok_injective hbl).symm
    subst hbl'
    by_cases hns : IsSent c
    · rw [if_neg (by simpa using hns)] at hcf
      obtain ⟨i2, hi2, hcf⟩ := bind_inv hcf
      have hcont : cf = ControlFlow.cont (t, b, i2) := (Result.ok_injective hcf).symm
      subst hcont
      exact ⟨hb, ht⟩
    · rw [if_pos (by simpa using hns)] at hcf
      obtain ⟨⟨key, tk⟩, htk, hcf⟩ := bind_inv hcf
      obtain ⟨⟨mu, back⟩, hg, hcf⟩ := bind_inv hcf
      obtain ⟨vv, hread, hcf⟩ := bind_inv hcf
      obtain ⟨cs, hupd, hcf⟩ := bind_inv hcf
      obtain ⟨big1, hpc, hcf⟩ := bind_inv hcf
      obtain ⟨i2, hi2, hcf⟩ := bind_inv hcf
      have hcont : cf =
          ControlFlow.cont ({ t with coords := cs, vals := back mu, keys := tk }, big1, i2) :=
        (Result.ok_injective hcf).symm
      subst hcont
      obtain ⟨hb1, ht1⟩ := place_capped_bits_tombs ks b big1 c key vv hpc
      exact ⟨hb1.trans hb, ht1.trans ht⟩
  · rw [if_neg hlt] at hcf
    have hdone : cf = ControlFlow.done (t.side, t.overflow, b) := (Result.ok_injective hcf).symm
    subst hdone
    exact ⟨hb, ht⟩

/-- `Table::resize_loop1` never writes `bits` or `tombs`. -/
theorem resize_loop1_bits_tombs {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (iter : alloc.vec.into_iter.IntoIter (U64 × U64 × I64)) (t0 : Tbl KS) {r : Tbl KS}
    (h : hashtrinity.Table.resize_loop1 core.cmp.EqU64 CI ks iter t0 = ok r) :
    r.bits = t0.bits ∧ r.tombs = t0.tombs := by
  unfold hashtrinity.Table.resize_loop1 at h
  refine loop_pc
    (fun (p : alloc.vec.into_iter.IntoIter (U64 × U64 × I64) × Tbl KS) =>
      p.2.bits = t0.bits ∧ p.2.tombs = t0.tombs)
    (fun r => r.bits = t0.bits ∧ r.tombs = t0.tombs)
    (fun p => hashtrinity.Table.resize_loop1.body core.cmp.EqU64 CI ks p.1 p.2)
    ?_ (iter, t0) ⟨rfl, rfl⟩ r h
  rintro ⟨it, t⟩ ⟨hb, ht⟩ cf hcf
  dsimp only at hcf
  unfold hashtrinity.Table.resize_loop1.body at hcf
  obtain ⟨⟨o, it1⟩, hn, hcf⟩ := bind_inv hcf
  rcases next_spec it it1 o hn with ⟨hnil, rfl⟩ | ⟨e, hcons, rfl⟩
  · have hdone : cf = ControlFlow.done t := (Result.ok_injective hcf).symm
    subst hdone
    exact ⟨hb, ht⟩
  · obtain ⟨c, k, v⟩ := e
    obtain ⟨t1, hpl, hcf⟩ := bind_inv hcf
    have hcont : cf = ControlFlow.cont (it1, t1) := (Result.ok_injective hcf).symm
    subst hcont
    obtain ⟨hb1, ht1⟩ := place_capped_bits_tombs ks t t1 c k v hpl
    exact ⟨hb1.trans hb, ht1.trans ht⟩

/-- `Table::with_bits` returns a table whose `bits` is `bits` and whose `tombs` is `0`. -/
theorem with_bits_bits_tombs {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64) {bits : U32}
    {t : Tbl KS} (h : hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI ks bits = ok t) :
    t.bits = bits ∧ t.tombs.val = 0 := by
  unfold hashtrinity.Table.with_bits at h
  obtain ⟨n, hn, h⟩ := bind_inv h
  obtain ⟨v, hv, h⟩ := bind_inv h
  unfold alloc.vec.Vec.into_boxed_slice at h
  simp only [bind_tc_ok] at h
  obtain ⟨s1, hs1, h⟩ := bind_inv h
  obtain ⟨keys, hkeys, h⟩ := bind_inv h
  unfold hashtrinity.side.SideStore.empty at h
  obtain ⟨ss, hss, h⟩ := bind_inv h
  have ht := (Result.ok_injective h).symm
  subst ht
  exact ⟨rfl, rfl⟩

/-- `Table::resize` to `nb` returns a table whose `bits` is `nb` and whose `tombs` is `0`. -/
theorem resize_bits_tombs {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64) {s s1 : Tbl KS}
    {nb : U32} (h : hashtrinity.Table.resize core.cmp.EqU64 CI ks s nb = ok s1) :
    s1.bits = nb ∧ s1.tombs.val = 0 := by
  unfold hashtrinity.Table.resize at h
  obtain ⟨bigger, hwb, h⟩ := bind_inv h
  obtain ⟨⟨ss, o, b1⟩, hl0, h⟩ := bind_inv h
  obtain ⟨hbb, hbt⟩ := with_bits_bits_tombs ks hwb
  obtain ⟨hb1, ht1⟩ := resize_loop0_bits_tombs ks s bigger hl0
  obtain ⟨ss1, hss1, h⟩ := bind_inv h
  simp only [core.mem.replace, core.option.Option.take] at h
  cases o with
  | none =>
    have hs1 : s1 = { b1 with side := ss } := (Result.ok_injective h).symm
    rw [hs1]
    refine ⟨hb1.trans hbb, ?_⟩
    show b1.tombs.val = 0
    rw [ht1]; exact hbt
  | some ov =>
    change (do
      let v ← hashtrinity.overflow.Overflow.into_entries CI ov
      let iter ← alloc.vec.IntoIteratorVec.into_iter v
      hashtrinity.Table.resize_loop1 core.cmp.EqU64 CI ks iter
        ({ b1 with side := ss } : Tbl KS)) = ok s1 at h
    obtain ⟨ev, hev, h⟩ := bind_inv h
    obtain ⟨iter, hiter, h⟩ := bind_inv h
    obtain ⟨hb2, ht2⟩ := resize_loop1_bits_tombs ks iter ({ b1 with side := ss } : Tbl KS) h
    refine ⟨hb2.trans (hb1.trans hbb), ?_⟩
    rw [ht2]
    show b1.tombs.val = 0
    rw [ht1]; exact hbt

/-- `Table::grow_for_one` raises `bits` by at most one and does not raise `tombs`. -/
theorem grow_for_one_bits_tombs {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    {s s1 : Tbl KS}
    (hg : hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok s1) :
    s1.bits.val ≤ s.bits.val + 1 ∧ s1.tombs.val ≤ s.tombs.val := by
  unfold hashtrinity.Table.grow_for_one at hg
  obtain ⟨i, _, hg⟩ := bind_inv hg
  obtain ⟨i1, _, hg⟩ := bind_inv hg
  obtain ⟨i2, _, hg⟩ := bind_inv hg
  obtain ⟨i3, _, hg⟩ := bind_inv hg
  by_cases hge : i2 >= i3
  · rw [if_pos hge] at hg
    obtain ⟨i4, _, hg⟩ := bind_inv hg
    obtain ⟨nb, hnb, hg⟩ := bind_inv hg
    have hnb_le : nb.val ≤ s.bits.val + 1 := by
      by_cases hlt : s.len < i4
      · rw [if_pos hlt] at hnb
        have := (Result.ok_injective hnb).symm
        subst this
        omega
      · rw [if_neg hlt] at hnb
        have := (uadd_ok hnb).1
        have h1 : (1#u32).val = 1 := rfl
        omega
    obtain ⟨hb, ht⟩ := resize_bits_tombs ks hg
    exact ⟨by rw [hb]; exact hnb_le, by rw [ht]; exact Nat.zero_le _⟩
  · rw [if_neg hge] at hg
    have hs1 : s1 = s := (Result.ok_injective hg).symm
    subst hs1
    exact ⟨Nat.le_succ _, le_refl _⟩

/-! ## Forward totality of the overflow write path

`insert_core`'s `Blocked` verdict on a present key calls `Overflow::take_value` then
`Overflow::write_back`; elsewhere they are partial-correctness.  Their common domain is `LiveEntry`:
the node index is in the arena and its entry holds a value. -/

/-- Node `n` of the arena exists and its entry `e` holds a value. -/
def LiveEntry (o : Ov) (n e : U32) : Prop :=
  ∃ nd, o.nodes.val[n.val]? = some nd ∧ ∃ t v, nd.entries.val[e.val]? = some (t, some v)

/-- **`take_value_ex`.**  `Overflow::take_value` returns `ok` on a live entry, and the node index
`n` is left in range with its entry vector unchanged in length (so a following `write_back` reads
`(n, e)` in range). -/
theorem take_value_ex (o : Ov) (n e : U32) (h : LiveEntry o n e) :
    ∃ old o', hashtrinity.overflow.Overflow.take_value CI o n e = ok (old, o') ∧
      (∃ nd', o'.nodes.val[n.val]? = some nd' ∧ e.val < nd'.entries.val.length) := by
  obtain ⟨nd, hnd, t, v, hent⟩ := h
  have hnlt : n.val < o.nodes.val.length := (List.getElem?_eq_some_iff.mp hnd).1
  unfold hashtrinity.overflow.Overflow.take_value
  simp only [lift, bind_tc_ok]
  obtain ⟨n1, back, hmut⟩ := vec_index_mut_ex o.nodes (UScalar.cast .Usize n)
    (by rw [cast_usize_u32]; exact hnlt)
  rw [hmut]; rw [bind_pair_ok]
  have hn1 : n1 = nd := by
    have h1 := (vec_index_mut_ok _ _ _ _ hmut).1
    rw [cast_usize_u32, hnd] at h1
    exact (Option.some.inj h1).symm
  rw [hn1]
  have helt : (UScalar.cast .Usize e).val < nd.entries.val.length := by
    rw [cast_usize_u32]; exact (List.getElem?_eq_some_iff.mp hent).1
  obtain ⟨p, back1, hmut1⟩ := vec_index_mut_ex nd.entries (UScalar.cast .Usize e) helt
  rw [hmut1]; rw [bind_pair_ok]
  have hp : p = (t, some v) := by
    have h1 := (vec_index_mut_ok _ _ _ _ hmut1).1
    rw [cast_usize_u32, hent] at h1
    exact (Option.some.inj h1).symm
  rw [hp]
  refine ⟨v, { o with nodes := back { nd with entries := back1 (t, none) } }, ?_, ?_⟩
  · simp [core.option.Option.take, core.option.Option.expect, Result.ofOption, bind_tc_ok]
  · refine ⟨{ nd with entries := back1 (t, none) }, ?_, ?_⟩
    · have hb : back { nd with entries := back1 (t, none) } =
          o.nodes.set (UScalar.cast .Usize n) { nd with entries := back1 (t, none) } :=
        congrFun (vec_index_mut_ok _ _ _ _ hmut).2 _
      rw [hb, vec_set_val, cast_usize_u32, List.getElem?_set_self hnlt]
    · show e.val < (back1 (t, none)).val.length
      have hb1 : back1 (t, none) =
          nd.entries.set (UScalar.cast .Usize e) (t, none) :=
        congrFun (vec_index_mut_ok _ _ _ _ hmut1).2 (t, none)
      rw [hb1, vec_set_val, cast_usize_u32, List.length_set]
      exact (List.getElem?_eq_some_iff.mp hent).1

/-- **`write_back_ex`.**  `Overflow::write_back` returns `ok` on an in-range entry. -/
theorem write_back_ex (o : Ov) (n e : U32) (v : I64)
    (hn : n.val < o.nodes.val.length)
    (he : ∃ nd, o.nodes.val[n.val]? = some nd ∧ e.val < nd.entries.val.length) :
    ∃ o', hashtrinity.overflow.Overflow.write_back CI o n e v = ok o' := by
  obtain ⟨nd, hnd, helt⟩ := he
  unfold hashtrinity.overflow.Overflow.write_back
  simp only [lift, bind_tc_ok]
  obtain ⟨n1, back, hmut⟩ := vec_index_mut_ex o.nodes (UScalar.cast .Usize n)
    (by rw [cast_usize_u32]; exact hn)
  rw [hmut]; rw [bind_pair_ok]
  have hn1 : n1 = nd := by
    have h1 := (vec_index_mut_ok _ _ _ _ hmut).1
    rw [cast_usize_u32, hnd] at h1
    exact (Option.some.inj h1).symm
  rw [hn1]
  have helt' : (UScalar.cast .Usize e).val < nd.entries.val.length := by
    rw [cast_usize_u32]; exact helt
  obtain ⟨p, back1, hmut1⟩ := vec_index_mut_ex nd.entries (UScalar.cast .Usize e) helt'
  rw [hmut1]; rw [bind_pair_ok]
  exact ⟨_, rfl⟩

/-! ## Forward totality of the `Blocked` verdict

`insert_core`'s `Blocked` verdict runs `find_entry` and branches: an absent key takes
`insert_new_at`; a present key takes `take_value` then `write_back`. -/

/-! ## Totality of `insert_core` on the non-reserved path -/

/-- **`insert_core_ns_ok`.**  On a non-reserved key, `insert_core k v true` returns `(none, s')`:
the `grow_for_one` prefix is total (`grow_for_one_ok`), the capped probe is total
(`probe_capped_keyed_ok`), and each verdict's write is total — the slot overwrite/fill of `Hit`/
`Vacant` (`slice_get_unchecked_mut_ex`, `set`) and the overflow write of `Blocked`
(`find_entry_total`, `ov_insert_new_at_ok`, `take_value_ex`, `write_back_ex`). -/
theorem insert_core_ns_ok {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (KC : Slice U64 → KS → Prop) (hRes : ResizeOK ks KC)
    (sz : KS → Nat)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks' ∧ n.val ≤ sz ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (hTK : TakeKeyOK ks sz KC)
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (n : Nat) (hHov : Hov s n) (hmain : mainCount s ≤ n) (hn : n * n + n + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length)
    (hsz_s : s.coords.val.length ≤ sz s.keys)
    (hKCset : ∀ (coords coords' : Slice U64) (keys keys' : KS) (i : Usize) (k : U64),
      KC coords keys → coords'.val = coords.val.set i.val k → ks.set keys i k = ok keys' →
      KC coords' keys')
    (KCS : alloc.vec.Vec U64 → KS → Prop)
    (hSide : ∀ t, hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok t →
      KCS t.side.coords t.side.keys ∧ t.side.cap.val ≤ sz t.side.keys)
    {k : U64} (hns : ¬ IsSent k) (v : I64) :
    ∃ s', hashtrinity.Table.insert_core core.cmp.EqU64 CI ks s k v true = ok (none, s') ∧
      KC s'.coords s'.keys ∧ KCS s'.side.coords s'.side.keys ∧
      s'.side.cap.val ≤ sz s'.side.keys := by
  unfold hashtrinity.Table.insert_core
  rw [coord_of_ok]
  simp only [bind_tc_ok]
  rw [is_sentinel_ok]
  simp only [bind_tc_ok]
  rw [if_neg (by simpa using hns)]
  obtain ⟨s1, hg⟩ := grow_for_one_ok ks sz halloc hm hA hset KC hTK hR hK hcap n hHov hmain hn hbits
    hocc hsz_s
  rw [hg]
  simp only [bind_tc_ok]
  obtain ⟨hR1, hbits1, hlen1, hsl1, hcap1, hHov1, hsz1, hK1, hmc1⟩ :=
    grow_post ks KC hRes sz halloc hm hA hset hTK hR hK hcap n hHov hmain hn hbits hocc hsz_s hg
  have hSide1 := hSide s1 hg
  obtain ⟨c, hc⟩ := probe_capped_keyed_ok ks hm hA s1 k k hbits1 hlen1
  rw [hc]
  simp only [bind_tc_ok]
  have hslot := probe_capped_keyed_slot_weak ks s1 k k c hc
  cases c with
  | Hit i =>
    have hcx : s1.coords.val[i.val]? = some k := hslot
    have hltc : i.val < s1.coords.val.length := (List.getElem?_eq_some_iff.mp hcx).1
    have hltv : i.val < s1.vals.length := by rw [hsl1]; exact hltc
    obtain ⟨mu, back, hget⟩ := slice_get_unchecked_mut_ex s1.vals i hltv
    obtain ⟨hmu, hback⟩ := get_unchecked_mut_ok s1.vals i mu back hget
    have hmu' : mu = some (m k) := by
      have h1 := (hR1.1.main_val i.val k hcx hns).2
      rw [hmu] at h1
      exact Option.some.inj h1
    have hread : core.mem.maybe_uninit.MaybeUninit.assume_init_read (some (m k)) = ok (m k) := rfl
    have hltv2 : i.val < (s1.vals.set i (some (m k))).length := by rw [slice_set_length]; exact hltv
    obtain ⟨_, back1, hget2⟩ := slice_get_unchecked_mut_ex (s1.vals.set i (some (m k))) i hltv2
    have hnew : core.mem.maybe_uninit.MaybeUninit.new v = ok (some v) := rfl
    refine ⟨{ s1 with vals := back1 (some v) }, ?_, hK1, hSide1.1, hSide1.2⟩
    simp only [hget, bind_tc_ok, ite_true]
    rw [hmu', hback]
    simp only [uncurry_apply_pair, hread, hget2, core.mem.drop,
      core.mem.maybe_uninit.MaybeUninit.new, bind_tc_ok]
  | Vacant i =>
    obtain ⟨x0, hcx, hx0e⟩ := hslot
    have hltc : i.val < s1.coords.val.length := (List.getElem?_eq_some_iff.mp hcx).1
    have hltv : i.val < s1.vals.length := by rw [hsl1]; exact hltc
    obtain ⟨xc, backc, hgetc⟩ := slice_get_unchecked_mut_ex s1.coords i hltc
    obtain ⟨-, hbackc⟩ := get_unchecked_mut_ok s1.coords i xc backc hgetc
    have hco : (backc k).val = s1.coords.val.set i.val k := by rw [hbackc, Slice.set_val_eq]
    obtain ⟨_, backv, hgetv⟩ := slice_get_unchecked_mut_ex s1.vals i hltv
    obtain ⟨keys', hset', -⟩ := hset s1.keys i k (by omega)
    have hlenmax1 : s1.len.val < Usize.max := by
      have hmc := hR1.1.len_count
      have hc := hcap1
      unfold nilN at hc
      have h := nilN_le_usize_max
      unfold nilN at h
      omega
    have hltmax : s1.len.val + (1#usize).val < 2 ^ UScalarTy.Usize.numBits := by
      rw [val_1_usize]
      have hl := hlenmax1
      rw [Usize.max_def, Usize.numBits_def] at hl
      have hp : 0 < 2 ^ UScalarTy.Usize.numBits := Nat.two_pow_pos _
      omega
    obtain ⟨i1, hi1, -⟩ := uadd_tot s1.len 1#usize hltmax
    refine ⟨{ s1 with coords := backc k, vals := backv (some v), keys := keys', len := i1 }, ?_,
      hKCset s1.coords (backc k) s1.keys keys' i k hK1 hco hset', hSide1.1, hSide1.2⟩
    simp only [hgetc, hgetv, hset', hi1, core.mem.maybe_uninit.MaybeUninit.new, bind_tc_ok]
    rfl
  | Blocked =>
    obtain ⟨ov, T, hX, hF, hcap, hlen⟩ : ∃ (ov : Ov) (T : OTree),
        (if core.option.Option.is_none s1.overflow = true then
            (do let o1 ← hashtrinity.overflow.Overflow.new I64 CI; ok (some o1))
          else ok s1.overflow) = ok (some ov) ∧
        OvFull ks s1 m ov T ∧ ov.nodes.val.length + 4 ≤ 2 ^ 32 - 1 ∧ ov.len.val < Usize.max := by
      cases hto : s1.overflow with
      | none =>
        let OV : Ov := { nodes := alloc.vec.Vec.new (hashtrinity.overflow.Node U64 I64),
                         free := alloc.vec.Vec.new U32,
                         root := hashtrinity.overflow.NIL,
                         len := 0#usize }
        have hnew : hashtrinity.overflow.Overflow.new I64 CI = ok OV := rfl
        obtain ⟨hF, -, hl0⟩ := ov_full_new ks s1 m OV hnew
        refine ⟨OV, .leaf, ?_, hF, by rw [hl0]; norm_num, ?_⟩
        · simp only [core.option.Option.is_none, Option.isNone, hnew, bind_tc_ok, ite_true]
        · have h0 : OV.len.val = 0 := rfl
          rw [h0]
          exact lt_of_lt_of_le (by unfold nilN; norm_num) nilN_le_usize_max
      | some ov0 =>
        obtain ⟨T, hF, -⟩ := ov_full_of_some ks hR1 hto
        obtain ⟨T', hI', hnn, hln, -⟩ := hHov1 ov0 hto
        refine ⟨ov0, T, ?_, hF, ?_, ?_⟩
        · simp only [core.option.Option.is_none, Option.isNone]
          rfl
        · have hc := hn
          unfold nilN at hc
          omega
        · have h := nilN_le_usize_max
          unfold nilN at h
          have hc := hn
          unfold nilN at hc
          omega
    simp only [hX, bind_tc_ok, core.option.Option.as_mut, uncurry_apply_pair]
    have hI : OInv ov T := hF.1
    have hall := hF.2
    have hsingle : ∀ j ∈ idxs T, ∃ nd, ov.nodes.val[j]? = some nd ∧
        ∃ w, nd.entries.val = [(nd.key, some w)] :=
      fun j hj => by obtain ⟨nd, hnd, -, hent, -, -⟩ := hall j hj; exact ⟨nd, hnd, m nd.key, hent⟩
    have hentries : ∀ j ∈ idxs T, ∀ nd, ov.nodes.val[j]? = some nd →
        nd.entries.val.length < Usize.max := by
      intro j hj nd hnd
      obtain ⟨nd', hnd', w, hent⟩ := hsingle j hj
      rw [hnd] at hnd'
      have hEq : nd = nd' := Option.some.inj hnd'
      rw [hEq, hent]
      exact lt_of_lt_of_le (by unfold nilN; norm_num) nilN_le_usize_max
    have hentries32 : ∀ j ∈ idxs T, ∀ nd, ov.nodes.val[j]? = some nd →
        nd.entries.val.length < 2 ^ 32 := by
      intro j hj nd hnd
      obtain ⟨nd', hnd', w, hent⟩ := hsingle j hj
      rw [hnd] at hnd'
      have hEq : nd = nd' := Option.some.inj hnd'
      rw [hEq, hent]; norm_num
    have hS : SingleEntries ov.nodes.val T := fun i hi => by
      obtain ⟨nd, hnd, w, hent⟩ := hsingle i hi
      exact ⟨nd, hnd, w, hent⟩
    obtain ⟨r, hfe, -⟩ := find_entry_total ov T hI.1 hI.2.1 hS k
    rw [hfe]
    simp only [bind_tc_ok]
    rcases r with _ | ⟨node, entry⟩
    · obtain ⟨nd, e, ov1, hins⟩ := ov_insert_new_at_ok ov T k k v hI hlen hentries hcap
      rw [hins]
      simp only [bind_tc_ok, uncurry_apply_pair]
      exact ⟨{ s1 with overflow := _ }, rfl, hK1, hSide1.1, hSide1.2⟩
    · have hspec := (find_entry_spec ov T hI.1 hI.2.1 hentries32 k k (.some (node, entry)) hfe).2
        node entry
      obtain ⟨hnmem, nd, hnd, hkey, hfirst⟩ := hspec.mp rfl
      obtain ⟨nd1, hnd1, -, hent1, hns1, -⟩ := hall node.val hnmem
      have hndeq : nd = nd1 := Option.some.inj (hnd.symm.trans hnd1)
      rw [hndeq] at hkey hfirst
      have hentk : nd1.entries.val = [(k, some (m k))] := by rw [hent1, hkey]
      have hlive : LiveEntry ov node entry := by
        obtain ⟨x, hx, -⟩ := hfirst.1
        rw [hentk] at hx
        have hlt1 : entry.val < 1 := by
          have := (List.getElem?_eq_some_iff.mp hx).1; simpa using this
        have he0 : entry.val = 0 := by omega
        exact ⟨nd1, hnd1, k, m k, by rw [hentk, he0]; rfl⟩
      obtain ⟨old, ov1, htv, ndv, hndv, hlenv⟩ := take_value_ex ov node entry hlive
      have hn1 : node.val < ov1.nodes.val.length := (List.getElem?_eq_some_iff.mp hndv).1
      obtain ⟨ov2, hwb⟩ := write_back_ex ov1 node entry v hn1 ⟨ndv, hndv, hlenv⟩
      simp only [core.mem.drop, bind_tc_ok, uncurry_apply_pair]
      rw [htv]
      simp only [bind_tc_ok, uncurry_apply_pair, ite_true]
      rw [hwb]
      simp only [bind_tc_ok]
      exact ⟨{ s1 with overflow := _ }, rfl, hK1, hSide1.1, hSide1.2⟩

/-- **`insert_core_ns_ok_lin_gen`.**  The probe half of `insert_core_ns_ok_lin`, parameterised by the
post-`grow_for_one` state.  From `hg : grow_for_one … s = ok s1`, an overflow counter `A` bounding
`Hov s1 A`/`Hent s1 A`, a witness `W` bounding `mainCount s1`, and the arena capacity `A + 4 ≤ nilN`,
the executed `insert_core` returns `ok (none, s')` with `Hov s' N`, `Hent s' N` and
`mainCount s' ≤ N` for the single counter `N = max (A+1) (W+1)`.  `A` is the tight counter the
`grow_post` branch supplies (`Hov s1 (n+e)` on a resize, `Hov s1 n` on the identity branch); `W`
carries the possibly larger `mainCount` bound so the two can differ. -/
theorem insert_core_ns_ok_lin_gen {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (KC : Slice U64 → KS → Prop)
    (sz : KS → Nat)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (KCS : alloc.vec.Vec U64 → KS → Prop)
    {s s1 : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR1 : RepW ks s1 m live) (hK1 : KC s1.coords s1.keys)
    (hSide1 : KCS s1.side.coords s1.side.keys ∧ s1.side.cap.val ≤ sz s1.side.keys)
    (hcap1 : arenaLen s1 + mainCount s1 + 4 ≤ nilN)
    (hbits1 : 0 < s1.bits.val) (hlen1 : s1.coords.length = 2 ^ s1.bits.val)
    (hsl1 : s1.vals.length = s1.coords.length) (hsz1 : s1.coords.val.length ≤ sz s1.keys)
    (A W B : Nat) (hHov1 : Hov s1 A) (hHent1 : Hent s1 A) (hmc1 : mainCount s1 ≤ W)
    (hAw : A + 4 ≤ nilN) (hgbits : s1.bits.val ≤ B) (htombs1 : s1.tombs.val = 0)
    (hKCset : ∀ (coords coords' : Slice U64) (keys keys' : KS) (i : Usize) (k : U64),
      KC coords keys → coords'.val = coords.val.set i.val k → ks.set keys i k = ok keys' →
      KC coords' keys')
    {k : U64} (hns : ¬ IsSent k) (v : I64)
    (hg : hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok s1) :
    ∃ (s' : Tbl KS),
      hashtrinity.Table.insert_core core.cmp.EqU64 CI ks s k v true = ok (none, s') ∧
      KC s'.coords s'.keys ∧ KCS s'.side.coords s'.side.keys ∧
      s'.side.cap.val ≤ sz s'.side.keys ∧
      Hov s' (max (A + 1) (W + 1)) ∧ Hent s' (max (A + 1) (W + 1)) ∧
      mainCount s' ≤ max (A + 1) (W + 1) ∧
      s'.tombs.val = 0 ∧ s'.bits.val ≤ B := by
  unfold hashtrinity.Table.insert_core
  rw [coord_of_ok]
  simp only [bind_tc_ok]
  rw [is_sentinel_ok]
  simp only [bind_tc_ok]
  rw [if_neg (by simpa using hns)]
  rw [hg]
  simp only [bind_tc_ok]
  obtain ⟨c, hc⟩ := probe_capped_keyed_ok ks hm hA s1 k k hbits1 hlen1
  rw [hc]
  simp only [bind_tc_ok]
  have hslot := probe_capped_keyed_slot_weak ks s1 k k c hc
  cases c with
  | Hit i =>
    have hcx : s1.coords.val[i.val]? = some k := hslot
    have hltc : i.val < s1.coords.val.length := (List.getElem?_eq_some_iff.mp hcx).1
    have hltv : i.val < s1.vals.length := by rw [hsl1]; exact hltc
    obtain ⟨mu, back, hget⟩ := slice_get_unchecked_mut_ex s1.vals i hltv
    obtain ⟨hmu, hback⟩ := get_unchecked_mut_ok s1.vals i mu back hget
    have hmu' : mu = some (m k) := by
      have h1 := (hR1.1.main_val i.val k hcx hns).2
      rw [hmu] at h1
      exact Option.some.inj h1
    have hread : core.mem.maybe_uninit.MaybeUninit.assume_init_read (some (m k)) = ok (m k) := rfl
    have hltv2 : i.val < (s1.vals.set i (some (m k))).length := by rw [slice_set_length]; exact hltv
    obtain ⟨_, back1, hget2⟩ := slice_get_unchecked_mut_ex (s1.vals.set i (some (m k))) i hltv2
    have hnew : core.mem.maybe_uninit.MaybeUninit.new v = ok (some v) := rfl
    refine ⟨{ s1 with vals := back1 (some v) }, ?_, hK1, hSide1.1, hSide1.2,
      ?_, ?_, ?_, ?_, ?_⟩
    · simp only [hget, bind_tc_ok, ite_true]
      rw [hmu', hback]
      simp only [uncurry_apply_pair, hread, hget2, core.mem.drop,
        core.mem.maybe_uninit.MaybeUninit.new, bind_tc_ok]
    · exact (Hov_of_overflow (s := s1) rfl).mpr (Hov_mono hHov1 (by omega))
    · exact (Hent_of_overflow (s := s1) rfl).mpr (Hent_mono hHent1 (by omega))
    · show mainCount s1 ≤ max (A + 1) (W + 1)
      omega
    · show s1.tombs.val = 0
      omega
    · show s1.bits.val ≤ B
      exact hgbits
  | Vacant i =>
    obtain ⟨x0, hcx, hx0e⟩ := hslot
    have hltc : i.val < s1.coords.val.length := (List.getElem?_eq_some_iff.mp hcx).1
    have hltv : i.val < s1.vals.length := by rw [hsl1]; exact hltc
    obtain ⟨xc, backc, hgetc⟩ := slice_get_unchecked_mut_ex s1.coords i hltc
    obtain ⟨-, hbackc⟩ := get_unchecked_mut_ok s1.coords i xc backc hgetc
    have hco : (backc k).val = s1.coords.val.set i.val k := by rw [hbackc, Slice.set_val_eq]
    obtain ⟨_, backv, hgetv⟩ := slice_get_unchecked_mut_ex s1.vals i hltv
    obtain ⟨keys', hset', -⟩ := hset s1.keys i k (by omega)
    have hlenmax1 : s1.len.val < Usize.max := by
      have hmc := hR1.1.len_count
      have hc := hcap1
      unfold nilN at hc
      have h := nilN_le_usize_max
      unfold nilN at h
      omega
    have hltmax : s1.len.val + (1#usize).val < 2 ^ UScalarTy.Usize.numBits := by
      rw [val_1_usize]
      have hl := hlenmax1
      rw [Usize.max_def, Usize.numBits_def] at hl
      have hp : 0 < 2 ^ UScalarTy.Usize.numBits := Nat.two_pow_pos _
      omega
    obtain ⟨i1, hi1, -⟩ := uadd_tot s1.len 1#usize hltmax
    refine ⟨{ s1 with coords := backc k, vals := backv (some v), keys := keys', len := i1 },
      ?_, hKCset s1.coords (backc k) s1.keys keys' i k hK1 hco hset', hSide1.1,
      hSide1.2, ?_, ?_, ?_, ?_, ?_⟩
    · simp only [hgetc, hgetv, hset', hi1, core.mem.maybe_uninit.MaybeUninit.new, bind_tc_ok]
      rfl
    · exact (Hov_of_overflow (s := s1) rfl).mpr (Hov_mono hHov1 (by omega))
    · exact (Hent_of_overflow (s := s1) rfl).mpr (Hent_mono hHent1 (by omega))
    · show (backc k).val.countP (fun c => decide (¬ IsSent c)) ≤ max (A + 1) (W + 1)
      rw [hco]
      have h1 := countP_set_le (fun c => decide (¬ IsSent c)) s1.coords.val i.val k
      unfold mainCount at hmc1
      omega
    · show s1.tombs.val = 0
      omega
    · show s1.bits.val ≤ B
      exact hgbits
  | Blocked =>
    obtain ⟨ov, T, hX, hF, hcap_b, hlen_b, hnodes_b, hlenov_b, hentries_b, hsum_b⟩ :
        ∃ (ov : Ov) (T : OTree),
          (if core.option.Option.is_none s1.overflow = true then
              (do let o1 ← hashtrinity.overflow.Overflow.new I64 CI; ok (some o1))
            else ok s1.overflow) = ok (some ov) ∧
          OvFull ks s1 m ov T ∧ ov.nodes.val.length + 4 ≤ 2 ^ 32 - 1 ∧ ov.len.val < Usize.max ∧
          ov.nodes.val.length ≤ A ∧ ov.len.val ≤ A ∧
          (∀ j ∈ idxs T, ∀ nd, ov.nodes.val[j]? = some nd → nd.entries.val.length ≤ A) ∧
          ((idxs T).map (remLen ov.nodes)).sum ≤ A := by
      cases hto : s1.overflow with
      | none =>
        let OV : Ov := { nodes := alloc.vec.Vec.new (hashtrinity.overflow.Node U64 I64),
                         free := alloc.vec.Vec.new U32,
                         root := hashtrinity.overflow.NIL,
                         len := 0#usize }
        have hnew : hashtrinity.overflow.Overflow.new I64 CI = ok OV := rfl
        obtain ⟨hF, -, hl0⟩ := ov_full_new ks s1 m OV hnew
        refine ⟨OV, .leaf, ?_, hF, by rw [hl0]; norm_num, ?_, ?_, ?_, ?_, ?_⟩
        · simp only [core.option.Option.is_none, Option.isNone, hnew, bind_tc_ok, ite_true]
        · have h0 : OV.len.val = 0 := rfl
          rw [h0]
          exact lt_of_lt_of_le (by unfold nilN; norm_num) nilN_le_usize_max
        · rw [hl0]; omega
        · have h0 : OV.len.val = 0 := rfl
          rw [h0]; omega
        · intro j hj; simp [idxs] at hj
        · simp [idxs]
      | some ov0 =>
        obtain ⟨T, hF, -⟩ := ov_full_of_some ks hR1 hto
        obtain ⟨T', hI', hnn, hln, hent'⟩ := hHov1 ov0 hto
        have hTT : T' = T := emb_unique hI'.1 hF.1.1
        have hent'' : ∀ j ∈ idxs T, ∀ nd, ov0.nodes.val[j]? = some nd →
            nd.entries.val.length ≤ A :=
          fun j hj => hent' j (hTT.symm ▸ hj)
        have hsum := hHent1 ov0 T hto hF.1
        refine ⟨ov0, T, ?_, hF, ?_, ?_, hnn, hln, hent'', hsum⟩
        · simp only [core.option.Option.is_none, Option.isNone]
          rfl
        · have hc := hAw
          unfold nilN at hc
          omega
        · have h := nilN_le_usize_max
          unfold nilN at h
          have hc := hAw
          unfold nilN at hc
          omega
    simp only [hX, bind_tc_ok, core.option.Option.as_mut, uncurry_apply_pair]
    have hI : OInv ov T := hF.1
    have hall := hF.2
    have hsingle : ∀ j ∈ idxs T, ∃ nd, ov.nodes.val[j]? = some nd ∧
        ∃ w, nd.entries.val = [(nd.key, some w)] :=
      fun j hj => by obtain ⟨nd, hnd, -, hent, -, -⟩ := hall j hj; exact ⟨nd, hnd, m nd.key, hent⟩
    have hentries : ∀ j ∈ idxs T, ∀ nd, ov.nodes.val[j]? = some nd →
        nd.entries.val.length < Usize.max := by
      intro j hj nd hnd
      obtain ⟨nd', hnd', w, hent⟩ := hsingle j hj
      rw [hnd] at hnd'
      have hEq : nd = nd' := Option.some.inj hnd'
      rw [hEq, hent]
      exact lt_of_lt_of_le (by unfold nilN; norm_num) nilN_le_usize_max
    have hentries32 : ∀ j ∈ idxs T, ∀ nd, ov.nodes.val[j]? = some nd →
        nd.entries.val.length < 2 ^ 32 := by
      intro j hj nd hnd
      obtain ⟨nd', hnd', w, hent⟩ := hsingle j hj
      rw [hnd] at hnd'
      have hEq : nd = nd' := Option.some.inj hnd'
      rw [hEq, hent]; norm_num
    have hS : SingleEntries ov.nodes.val T := fun i hi => by
      obtain ⟨nd, hnd, w, hent⟩ := hsingle i hi
      exact ⟨nd, hnd, w, hent⟩
    obtain ⟨r, hfe, -⟩ := find_entry_total ov T hI.1 hI.2.1 hS k
    rw [hfe]
    simp only [bind_tc_ok]
    rcases r with _ | ⟨node, entry⟩
    · obtain ⟨nd, e1, ov1, hins⟩ := ov_insert_new_at_ok ov T k k v hI hlen_b hentries hcap_b
      rw [hins]
      simp only [bind_tc_ok, uncurry_apply_pair]
      obtain ⟨T', hI', hlen', hnodes', hent', hsum'⟩ :=
        ov_insert_new_at_domain ov ov1 T k k v nd e1 A hI (by unfold nilN; omega)
          hentries_b hins
      refine ⟨{ s1 with overflow := some ov1 }, rfl, hK1, hSide1.1, hSide1.2,
        ?_, ?_, ?_, ?_, ?_⟩
      · intro ov' hsome
        have hov' : ov' = ov1 := (Option.some.inj hsome).symm
        subst hov'
        refine ⟨T', hI', by omega, by omega, fun j hj nd' hnd' => ?_⟩
        have := hent' j hj nd' hnd'
        omega
      · intro ov' Tg hsome hI2
        have hov' : ov' = ov1 := (Option.some.inj hsome).symm
        subst hov'
        have hTT : Tg = T' := emb_unique hI2.1 hI'.1
        subst hTT
        omega
      · show mainCount s1 ≤ max (A + 1) (W + 1)
        omega
      · show s1.tombs.val = 0
        omega
      · show s1.bits.val ≤ B
        exact hgbits
    · have hspec := (find_entry_spec ov T hI.1 hI.2.1 hentries32 k k (.some (node, entry)) hfe).2
        node entry
      obtain ⟨hnmem, nd, hnd, hkey, hfirst⟩ := hspec.mp rfl
      obtain ⟨nd1, hnd1, -, hent1, hns1, -⟩ := hall node.val hnmem
      have hndeq : nd = nd1 := Option.some.inj (hnd.symm.trans hnd1)
      rw [hndeq] at hkey hfirst
      have hentk : nd1.entries.val = [(k, some (m k))] := by rw [hent1, hkey]
      have hlive : LiveEntry ov node entry := by
        obtain ⟨x, hx, -⟩ := hfirst.1
        rw [hentk] at hx
        have hlt1 : entry.val < 1 := by
          have := (List.getElem?_eq_some_iff.mp hx).1; simpa using this
        have he0 : entry.val = 0 := by omega
        exact ⟨nd1, hnd1, k, m k, by rw [hentk, he0]; rfl⟩
      obtain ⟨old, ov1, htv, ndv, hndv, hlenv⟩ := take_value_ex ov node entry hlive
      have hn1 : node.val < ov1.nodes.val.length := (List.getElem?_eq_some_iff.mp hndv).1
      obtain ⟨ov2, hwb⟩ := write_back_ex ov1 node entry v hn1 ⟨ndv, hndv, hlenv⟩
      simp only [core.mem.drop, bind_tc_ok, uncurry_apply_pair]
      rw [htv]
      simp only [bind_tc_ok, uncurry_apply_pair, ite_true]
      rw [hwb]
      simp only [bind_tc_ok]
      obtain ⟨hF2, hT2, hTk, hl2⟩ := blocked_hit_full ks ov T hF node entry hfe old ov1 htv v ov2 hwb
      have hlen2 : ov2.len.val ≤ A := by
        have h1 : ov1.len.val = ov.len.val := congrArg (fun x : Usize => x.val)
          (take_value_len ov node entry old ov1 htv)
        have h2 : ov2.len.val = ov1.len.val := congrArg (fun x : Usize => x.val)
          (write_back_len ov1 node entry v ov2 hwb)
        omega
      refine ⟨{ s1 with overflow := some ov2 }, rfl, hK1, hSide1.1, hSide1.2,
        ?_, ?_, ?_, ?_, ?_⟩
      · intro ov' hsome
        have hov' : ov' = ov2 := (Option.some.inj hsome).symm
        subst hov'
        refine ⟨T, hF2.1, by omega, by omega, fun j hj nd' hnd' => ?_⟩
        obtain ⟨a, ha, _, henteq, -⟩ := hF2.2 j hj
        rw [hnd'] at ha
        have := Option.some.inj ha
        subst this
        rw [henteq]; simp
      · intro ov' Tg hsome hI2
        have hov' : ov' = ov2 := (Option.some.inj hsome).symm
        rw [hov'] at hI2 ⊢
        have hTT : Tg = T := emb_unique hI2.1 hF2.1.1
        rw [hTT]
        have hle1 : ∀ j ∈ idxs T, remLen ov2.nodes j ≤ 1 := by
          intro j hj
          obtain ⟨a, ha, _, henteq, -⟩ := hF2.2 j hj
          rw [remLen_eq ov2.nodes j a ha, henteq]; simp
        have hsum := sum_map_le_mul (idxs T) (remLen ov2.nodes) 1 hle1
        have hlenT : (idxs T).length ≤ ov2.nodes.val.length :=
          idxs_length_le (nd_nodup T hF2.1.2.2.1) (fun j hj => by
            obtain ⟨a, ha, -⟩ := hF2.2 j hj
            exact (List.getElem?_eq_some_iff.mp ha).1)
        omega
      · show mainCount s1 ≤ max (A + 1) (W + 1)
        omega
      · show s1.tombs.val = 0
        omega
      · show s1.bits.val ≤ B
        exact hgbits

/-- **`insert_core_ns_ok_lin`.**  `insert_core_ns_ok` with the counter frames.  From `Hov s n`,
`Hent s e` and `mainCount s ≤ n`, the post-state `s'` satisfies `Hov s' N`, `Hent s' N` and
`mainCount s' ≤ N` for a single counter `N ≤ 2·(n+e)+1` (one `grow_for_one` raises both counters to
`n+e` and the main count to at most `2·n`; the `Hit`/`Vacant` verdicts and the overflow write of
`Blocked` raise them by at most one more).  The `bits` headroom frame is the growth of `bits` by at
most one per call; `tombs` stays `0` when it starts `0`. -/
theorem insert_core_ns_ok_lin {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (KC : Slice U64 → KS → Prop) (hRes : ResizeOK ks KC)
    (sz : KS → Nat)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks' ∧ n.val ≤ sz ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (hTK : TakeKeyOK ks sz KC)
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (n e : Nat) (hHov : Hov s n) (hent : Hent s e) (hmain : mainCount s ≤ n)
    (hn : n * n + n + 4 ≤ nilN) (hne : n + e + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length)
    (htombs : s.tombs.val = 0)
    (hsz_s : s.coords.val.length ≤ sz s.keys)
    (hKCset : ∀ (coords coords' : Slice U64) (keys keys' : KS) (i : Usize) (k : U64),
      KC coords keys → coords'.val = coords.val.set i.val k → ks.set keys i k = ok keys' →
      KC coords' keys')
    (KCS : alloc.vec.Vec U64 → KS → Prop)
    (hSide : ∀ t, hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok t →
      KCS t.side.coords t.side.keys ∧ t.side.cap.val ≤ sz t.side.keys)
    {k : U64} (hns : ¬ IsSent k) (v : I64) :
    ∃ (s' : Tbl KS) (N : Nat),
      hashtrinity.Table.insert_core core.cmp.EqU64 CI ks s k v true = ok (none, s') ∧
      KC s'.coords s'.keys ∧ KCS s'.side.coords s'.side.keys ∧
      s'.side.cap.val ≤ sz s'.side.keys ∧
      Hov s' N ∧ Hent s' N ∧ mainCount s' ≤ N ∧ N ≤ 2 * (n + e) + 1 ∧
      s'.tombs.val = 0 ∧ s'.bits.val ≤ s.bits.val + 1 := by
  obtain ⟨s1, hg⟩ := grow_for_one_ok ks sz halloc hm hA hset KC hTK hR hK hcap n hHov hmain hn hbits
    hocc hsz_s
  obtain ⟨hR1, hbits1, hlen1, hsl1, hcap1, hHov1, hHent1, hsz1, hK1, hmc1⟩ :=
    grow_post_lin ks KC hRes sz halloc hm hA hset hTK hR hK hcap n e hHov hent hmain hne hbits
      hocc hsz_s hg
  obtain ⟨hgbits, hgtombs⟩ := grow_for_one_bits_tombs ks hg
  have hSide1 := hSide s1 hg
  have htombs1 : s1.tombs.val = 0 := by omega
  obtain ⟨s', h, hK', hKS', hcap', hHov', hHent', hmain', htombs', hbits'⟩ :=
    insert_core_ns_ok_lin_gen ks KC sz hm hA hset KCS hR1 hK1 hSide1 hcap1 hbits1 hlen1 hsl1 hsz1
      (n + e) (2 * (n + e)) (s.bits.val + 1) hHov1 hHent1 (by omega) (by omega) hgbits htombs1
      hKCset hns v hg
  exact ⟨s', 2 * (n + e) + 1, h, hK', hKS', hcap', Hov_mono hHov' (by omega),
    Hent_mono hHent' (by omega), by omega, by omega, htombs', hbits'⟩

/-- **`insert_core_ns_ok_nr`.**  The identity branch of `insert_core_ns_ok_lin_gen`: when
`grow_for_one` does not resize (`s1 = s`, i.e. the arithmetic check `(len+tombs+1)·LOAD_DEN ≥
len(coords)·LOAD_NUM` fails), `Hov s n`/`Hent s n` are carried unchanged into the probe, so the whole
insert returns `ok (none, s')` with `Hov s' (n+1)`, `Hent s' (n+1)` and `mainCount s' ≤ n+1`.  This is
the linear half of the split step; the resize half is bounded by `2·(n+e)+1` in
`insert_core_ns_ok_lin` (no resize: `grow_for_one` returns the table and the counter moves by at most
one). -/
theorem insert_core_ns_ok_nr {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (KC : Slice U64 → KS → Prop)
    (sz : KS → Nat)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys)
    (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (n : Nat) (hHov : Hov s n) (hHent : Hent s n) (hmain : mainCount s ≤ n)
    (hn : n + 4 ≤ nilN) (htombs : s.tombs.val = 0)
    (hsz_s : s.coords.val.length ≤ sz s.keys)
    (hKCset : ∀ (coords coords' : Slice U64) (keys keys' : KS) (i : Usize) (k : U64),
      KC coords keys → coords'.val = coords.val.set i.val k → ks.set keys i k = ok keys' →
      KC coords' keys')
    (KCS : alloc.vec.Vec U64 → KS → Prop)
    (hSide : ∀ t, hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok t →
      KCS t.side.coords t.side.keys ∧ t.side.cap.val ≤ sz t.side.keys)
    {k : U64} (hns : ¬ IsSent k) (v : I64)
    (hg : hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok s) :
    ∃ s', hashtrinity.Table.insert_core core.cmp.EqU64 CI ks s k v true = ok (none, s') ∧
      KC s'.coords s'.keys ∧ KCS s'.side.coords s'.side.keys ∧
      s'.side.cap.val ≤ sz s'.side.keys ∧
      Hov s' (n + 1) ∧ Hent s' (n + 1) ∧ mainCount s' ≤ n + 1 ∧
      s'.tombs.val = 0 ∧ s'.bits.val ≤ s.bits.val + 1 := by
  have hSide1 := hSide s hg
  obtain ⟨s', h, hK', hKS', hcap', hHov', hHent', hmain', htombs', hbits'⟩ :=
    insert_core_ns_ok_lin_gen ks KC sz hm hA hset KCS hR hK hSide1 hcap
      hR.1.bits_pos hR.1.len_pow hR.1.vals_len hsz_s n n (s.bits.val + 1)
      hHov hHent hmain (by omega) (Nat.le_succ _) htombs hKCset hns v hg
  exact ⟨s', h, hK', hKS', hcap', Hov_mono hHov' (by omega),
    Hent_mono hHent' (by omega), by omega, htombs', hbits'⟩

/-! ## The `Regen<u64>` face -/

/-- **`coord_insert_ok`.**  `word_coordmap_insert` (the `Regen<u64>` face) returns `ok` on a
non-reserved key: `insert_core_ns_ok` at `RKS`, whose `alloc`, `matches`, `set` and `take_key` are
total, then the `none` arm of `insert_merge_last_write`.  This discharges the premise
`word_coordmap_insert s k v = ok s'` of `coord_insert_refines` on that domain. -/
theorem coord_insert_ok {s : Tbl (hashtrinity.algebra.Regen U64)} {m : Apeiron.Table.HMap U64 I64}
    {live : U64 → Prop} (hR : RepW RKS s m live) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (n : Nat) (hHov : Hov s n) (hmain : mainCount s ≤ n) (hn : n * n + n + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length)
    {k : U64} (hns : ¬ IsSent k) (v : I64) :
    ∃ s', word_coordmap_insert s k v = ok s' := by
  obtain ⟨s', h, -, -, -⟩ := insert_core_ns_ok RKS (fun _ _ => True) resizeOK_regen regenSz
    (fun n => ⟨(), rfl, usize_val_le_two_pow64 n⟩)
    (fun keys i c k => ⟨true, RKS_matches keys i c k⟩)
    (fun _ => avx2_spec_holds)
    (fun keys i k _ => ⟨keys, rfl, rfl⟩)
    regen_takeKeyOK
    hR trivial hcap n hHov hmain hn hbits hocc (regen_sz_inv s)
    (fun _ _ _ _ _ _ _ _ _ => trivial)
    (fun _ _ => True)
    (fun t _ => ⟨trivial, usize_val_le_two_pow64 t.side.cap⟩)
    hns v
  refine ⟨s', ?_⟩
  unfold word_coordmap_insert hashtrinity.Table.insert_merge_last_write
  rw [h]
  simp only [bind_tc_ok, uncurry_apply_pair]

/-- **`coord_insert_witness`.**  The premises of `coord_insert_ok` hold jointly at a reachable
state: the table `with_bits 1` builds (`grow_witness`), with the non-reserved key `1`. -/
theorem coord_insert_witness :
    ∃ (s : Tbl (hashtrinity.algebra.Regen U64)) (m : Apeiron.Table.HMap U64 I64)
      (live : U64 → Prop) (n : Nat) (s' : Tbl (hashtrinity.algebra.Regen U64)),
      RepW RKS s m live ∧
      arenaLen s + mainCount s + 4 ≤ nilN ∧
      Hov s n ∧ mainCount s ≤ n ∧ n * n + n + 4 ≤ nilN ∧
      s.bits.val + 3 ≤ UScalarTy.Usize.numBits ∧
      s.len.val + s.tombs.val ≤ s.coords.length ∧
      ¬ IsSent 1#u64 ∧
      word_coordmap_insert s 1#u64 0#i64 = ok s' := by
  obtain ⟨s, m, live, n, -, hR, hcap, hHov, hmain, hn, hbits, hocc, -⟩ := grow_witness
  have hns : ¬ IsSent 1#u64 := by
    unfold IsSent EMPTYN TOMBN
    decide
  obtain ⟨s', h⟩ := coord_insert_ok hR hcap n hHov hmain hn hbits hocc hns 0#i64
  exact ⟨s, m, live, n, s', hR, hcap, hHov, hmain, hn, hbits, hocc, hns, h⟩

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.Hov_mono
#audit Apeiron.MapExecuted.resize_loop1_ok_hov
#audit Apeiron.MapExecuted.resize_hov
#audit Apeiron.MapExecuted.grow_post
#audit Apeiron.MapExecuted.take_value_ex
#audit Apeiron.MapExecuted.write_back_ex
#audit Apeiron.MapExecuted.insert_core_ns_ok
#audit Apeiron.MapExecuted.coord_insert_ok
#audit Apeiron.MapExecuted.coord_insert_witness
