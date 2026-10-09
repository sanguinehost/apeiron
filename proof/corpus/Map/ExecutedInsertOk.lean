-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (APEIRON-HT-INSERT, HT-INSERT-W23): Stage 3,
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
    {t0 : Tbl KS} (n0 : Nat)
    (hbits0 : 0 < t0.bits.val) (hlen0 : t0.coords.val.length = 2 ^ t0.bits.val)
    (hsl0 : t0.vals.val.length = t0.coords.val.length)
    (hBig0 : BigInit t0) (hlenbnd0 : t0.len.val ≤ n0) (hHov0 : Hov t0 n0)
    (hcap : n0 + es.length + 4 ≤ nilN) (hsz0 : t0.coords.val.length ≤ sz t0.keys)
    (iter : alloc.vec.into_iter.IntoIter (U64 × U64 × I64)) (hiter : iter.val = es) :
    ∃ r, hashtrinity.Table.resize_loop1 core.cmp.EqU64 CI ks iter t0 = ok r ∧
      Hov r (n0 + es.length) ∧ r.coords.val.length ≤ sz r.keys := by
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
        p.2.coords.val.length ≤ sz p.2.keys)
    (fun r => Hov r (n0 + es.length) ∧ r.coords.val.length ≤ sz r.keys)
    (fun p => hashtrinity.Table.resize_loop1.body core.cmp.EqU64 CI ks p.1 p.2)
    ?_ (iter, t0) ?_ r hr
  · rintro ⟨it, tb⟩ ⟨dn, hdn, hbits, hlen, hsl, hBig, hlenbnd, hHov, hsz⟩ cf hcf
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
      exact ⟨hHov, hsz⟩
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
      obtain ⟨hHov0', hszb'⟩ := place_capped_hov ks sz hm hA hset tb tb2 c0 v0 (n0 + dn.length)
        hbits hlen hsl (fun j hcj => hBig j c0 hcj hns2) hlenmax hHov hK hsz hpc
      have hHov' : Hov tb2 (n0 + (dn ++ [(c0, c0, v0)]).length) := by
        rw [hcnt]; exact hHov0'
      refine ⟨dn ++ [(c0, c0, v0)], hdn', ?_, ?_, ?_, ?_, ?_, hHov', ?_⟩
      · rw [hbf]; exact hbits
      · rw [hlf2, hbf]; exact hlen
      · rw [hslf2, hlf2]; exact hsl
      · exact place_capped_bigInit ks tb tb2 c0 v0 hpc hBig
      · rw [hcnt]; omega
      · rw [hlf2, hszb']; exact hsz
  · refine ⟨[], by simpa using hiter, hbits0, hlen0, hsl0, hBig0, ?_, ?_, ?_⟩
    · simpa using hlenbnd0
    · simpa using hHov0
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
  obtain ⟨r, hr, hbr, hcr, hvr, hBigr, hlenr, hHovr, hside, hovf, hszr⟩ :=
    resize_loop0_ok_hov_strong ks sz hm hA hset KC hTK hR hK hcap bigger hb0 hcb0 hovnone hsz_s hszbb
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
    obtain ⟨r2, hr2, hHov2, hsz2⟩ :=
      resize_loop1_ok_hov ks sz hm hA hset es.val hes (t0 := { b1 with side := ss0 }) (mainCount s)
        hbr hcr hvr hBigr hlenr hHovr hcap1 (by simpa using hszr) iter (by rw [hies])
    have hs1eq : s1 = r2 := Result.ok_injective (hres.symm.trans hr2)
    subst hs1eq
    have hlen2 : es.val.length ≤ n * n := le_trans heslen (key ov T hso hI).2
    exact ⟨Hov_mono hHov2 (by omega), hsz2⟩

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
    obtain ⟨hI, hall⟩ := hF
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
