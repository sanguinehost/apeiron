-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (APEIRON-HT-INSERT, HT-INSERT-W22): totality
-- of the executed `grow_for_one`, on the resize totality split out of `Map.ExecutedResizeTotal`.
import Map.ExecutedResizeTotal
open Aeneas Aeneas.Std Result

/-!
# Totality of the executed `grow_for_one`

`grow_for_one` (`ApeironWord.lean`, `src/hashtrinity/mod.rs`) resizes to `bits` or `bits + 1` when
`(len + tombs + 1)·LOAD_DEN` reaches `cap·LOAD_NUM`, else returns the table unchanged.  Its
partial-correctness twin is `grow_rep` (`Map/ExecutedInsert.lean`).

The resize half is `resize_ok` / `resize_witness`, split here from `Map.ExecutedResizeTotal` so that
module stays under the source-size cap.  `resize_ok_hov` restates `resize_ok`'s two caller-supplied
entry bounds as the counter invariant `Hov` plus a numeric cap; `grow_for_one_ok` closes
`grow_for_one`'s checked-arithmetic branches and calls `resize_ok_hov`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-- `SideStore::empty` is total when the key store's `alloc` is. -/
theorem side_empty_ex {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (sz : KS → Nat)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks' ∧ n.val ≤ sz ks') :
    ∃ ss1, hashtrinity.side.SideStore.empty I64 CI ks = ok ss1 := by
  unfold hashtrinity.side.SideStore.empty
  obtain ⟨g0, hg0, -⟩ := halloc 0#usize
  rw [hg0]; simp only [bind_tc_ok]
  exact ⟨_, rfl⟩

/-- **`resize_ok`.**  `Table::resize` is total at a bit count below the word width, at a key store
whose `alloc`/`set` are total and whose `matches` returns.  The four failure-prone steps of the body
are composed in bind order: `with_bits_ex` (the fresh destination), `resize_loop0_ok_hov_strong` (the
live-slot loop; the destination's shape, `BigInit`, `len` bound and `Hov` it supplies are exactly
`resize_loop1_ok`'s premises), `into_entries_total` (the drained overflow) and `resize_loop1_ok`
(re-placing the entries).  `RepW` supplies the drained tree's `Emb`/alive facts through `OvWF`, but
carries no numeric bound on its entry count: `hov` bounds that count below `Usize.max` (the scan's
`Vec::push` ceiling) and `hvcap` below `nilN − mainCount s − 4` (loop 1's counter cap); both are
scoped to `s.overflow = some ov` so an `s` with no overflow satisfies them vacuously. -/
theorem resize_ok {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (sz : KS → Nat)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks' ∧ n.val ≤ sz ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (hTK : ∀ (keys : KS) (i : Usize) (c : U64), i.val < sz keys → ¬ IsSent c →
      ∃ keys', ks.take_key keys i c = ok (c, keys') ∧ sz keys' = sz keys)
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (new_bits : U32) (hbits : new_bits.val < UScalarTy.Usize.numBits)
    (hpos : 0 < new_bits.val)
    (hov : ∀ ov T, s.overflow = some ov → OInv ov T →
      (idxs T).length + ((idxs T).map (remLen ov.nodes)).sum < Usize.max)
    (hvcap : ∀ ov T, s.overflow = some ov → OInv ov T →
      mainCount s + ((idxs T).map (remLen ov.nodes)).sum + 4 ≤ nilN)
    (hsz_s : s.coords.val.length ≤ sz s.keys) :
    ∃ t, hashtrinity.Table.resize core.cmp.EqU64 CI ks s new_bits = ok t := by
  unfold hashtrinity.Table.resize
  obtain ⟨bigger, hwb, hbitsb, hlenbb, hslbb, hlenb0, hovnone, hBigb, hHov0b, hszbb⟩ :=
    with_bits_ex ks sz halloc new_bits hbits
  rw [hwb]; simp only [bind_tc_ok]
  have hb0C : CoreRep ks bigger m (fun _ => False) :=
    rep_map_congr ks (m' := m) (with_bits_rep ks new_bits hpos bigger hwb) (fun _ h => h.elim)
  have hb0 : RepW ks bigger m (LowMain s 0) := by
    refine repW_live_congr ks ⟨hb0C, by unfold OvWF; rw [hovnone]; trivial⟩ (fun c => ?_)
    exact ⟨False.elim, fun ⟨_, d, hd, _⟩ => absurd hd (Nat.not_lt_zero d)⟩
  have hcb0 : arenaLen bigger + mainCount bigger = 0 := by
    have h1 : arenaLen bigger = 0 := by unfold arenaLen; rw [hovnone]
    rw [h1, mainCount_zero ks hb0C]
  obtain ⟨r, hr, hbr, hcr, hvr, hBigr, hlenr, hHovr, hside, hovf, hszr⟩ :=
    resize_loop0_ok_hov_strong ks sz hm hA hset hTK hR hcap bigger hb0 hcb0 hovnone hsz_s hszbb
  rw [hr]; simp only [bind_tc_ok]
  obtain ⟨ss1, hss1⟩ := side_empty_ex ks sz halloc
  rw [hss1]; simp only [bind_tc_ok]
  rcases r with ⟨ss0, o, b1⟩
  simp only [core.mem.replace, core.option.Option.take]
  cases o with
  | none => exact ⟨_, rfl⟩
  | some ov =>
    change ∃ t, (do
      let v ← hashtrinity.overflow.Overflow.into_entries CI ov
      let iter ← alloc.vec.IntoIteratorVec.into_iter v
      hashtrinity.Table.resize_loop1 core.cmp.EqU64 CI ks iter
        { b1 with side := ss0 }) = ok t
    have hso : s.overflow = some ov := hovf.symm
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
    rw [hesIn]; simp only [bind_tc_ok]
    unfold alloc.vec.IntoIteratorVec.into_iter; simp only [bind_tc_ok]
    have hcap1 : mainCount s + es.val.length + 4 ≤ nilN := by
      have := hvcap ov T hso hI; omega
    exact resize_loop1_ok ks sz hm hA hset es.val hes (t0 := { b1 with side := ss0 }) (mainCount s)
      hbr hcr hvr hBigr hlenr hHovr hcap1 (by simpa using hszr) es rfl

/-- A list's mapped sum is at most its length times a uniform per-element bound. -/
theorem sum_map_le_mul {α : Type} (l : List α) (f : α → Nat) (n : Nat)
    (h : ∀ x ∈ l, f x ≤ n) : (l.map f).sum ≤ l.length * n := by
  induction l with
  | nil => simp
  | cons a l ih =>
    have ha : f a ≤ n := h a (by simp)
    have hl : ∀ x ∈ l, f x ≤ n := fun x hx => h x (by simp [hx])
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    have := ih hl
    nlinarith

/-- **`resize_ok_hov`.**  `resize_ok` with its two caller-supplied overflow-entry bounds replaced by
the counter invariant `Hov s n` and a numeric cap.  `Hov` bounds a present overflow's node count,
`len` and per-node entry count by `n`; the entry sum `Σ remLen` over the tree is therefore at most
`n · n` (at most `n` tree indices, each carrying at most `n` entries), and the two `resize_ok`
premises follow from `mainCount s ≤ n` and `n·n + n + 4 ≤ nilN`.  The sharper identity
`Σ remLen = ov.len` is not used: `OInv` does not constrain `ov.len`, so that identity would need a
new entries-`len` invariant carried through every overflow write. -/
theorem resize_ok_hov {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (sz : KS → Nat)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks' ∧ n.val ≤ sz ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (hTK : ∀ (keys : KS) (i : Usize) (c : U64), i.val < sz keys → ¬ IsSent c →
      ∃ keys', ks.take_key keys i c = ok (c, keys') ∧ sz keys' = sz keys)
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (new_bits : U32) (hbits : new_bits.val < UScalarTy.Usize.numBits)
    (hpos : 0 < new_bits.val)
    (n : Nat) (hHov : Hov s n) (hmain : mainCount s ≤ n) (hn : n * n + n + 4 ≤ nilN)
    (hsz_s : s.coords.val.length ≤ sz s.keys) :
    ∃ t, hashtrinity.Table.resize core.cmp.EqU64 CI ks s new_bits = ok t := by
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
  refine resize_ok ks sz halloc hm hA hset hTK hR hcap new_bits hbits hpos ?_ ?_ hsz_s
  · intro ov T hsome hI
    obtain ⟨h1, h2⟩ := key ov T hsome hI
    have h4 := nilN_le_usize_max
    have h5 : n * n + n + 4 ≤ nilN := hn
    unfold nilN at h4 h5
    omega
  · intro ov T hsome hI
    obtain ⟨h1, h2⟩ := key ov T hsome hI
    omega

/-- **`resize_witness`.**  A reachable joint witness for `resize_ok`'s state hypotheses at a table
that has run one placement step: `place_capped` at the (`Regen<u64>`) fresh `with_bits 1` table
puts the non-reserved coordinate `0` in a live slot (`t0 → s`).  `s`'s overflow stays `none`-or-
small: `place_capped_domain` gives `Hov s 1`, so any `OInv ov T` of `s`'s overflow has
`|idxs T| ≤ 1` and a total entry count `≤ 1`, satisfying both bounds, and `resize` itself returns
`ok r`.  Unlike `resize_loop0_witness`'s empty table, `s` has a live main slot. -/
theorem resize_witness :
    ∃ (t0 s : Tbl (hashtrinity.algebra.Regen U64)) (m : Apeiron.Table.HMap U64 I64)
      (live : U64 → Prop) (r : Tbl (hashtrinity.algebra.Regen U64)),
      hashtrinity.Table.place_capped core.cmp.EqU64 CI RKS t0 0#u64 0#u64 0#i64 = ok s ∧
      RepW RKS s m live ∧
      arenaLen s + mainCount s + 4 ≤ nilN ∧
      (∀ ov T, s.overflow = some ov → OInv ov T →
        (idxs T).length + ((idxs T).map (remLen ov.nodes)).sum < Usize.max) ∧
      (∀ ov T, s.overflow = some ov → OInv ov T →
        mainCount s + ((idxs T).map (remLen ov.nodes)).sum + 4 ≤ nilN) ∧
      hashtrinity.Table.resize core.cmp.EqU64 CI RKS s 1#u32 = ok r := by
  obtain ⟨t0, hwb, -⟩ := with_bits1_RKS_ex
  have hR0 : RepW RKS t0 (fun _ => 0#i64) (fun _ => False) :=
    ⟨with_bits_rep RKS 1#u32 (by decide) t0 hwb, by
      unfold OvWF; rw [with_bits_overflow RKS 1#u32 t0 hwb]; trivial⟩
  have hsum0 : arenaLen t0 + mainCount t0 = 0 := (with_bits_wf RKS 1#u32 (by decide) t0 hwb).2
  have hmc0 : mainCount t0 = 0 := by omega
  have hns0 : ¬ IsSent (0#u64) := by
    intro h; rcases h with h | h
    · have hv : (0#u64).val = 0 := rfl
      rw [hv] at h; unfold EMPTYN at h; omega
    · have hv : (0#u64).val = 0 := rfl
      rw [hv] at h; unfold TOMBN at h; omega
  have hlenmax : t0.len.val < Usize.max := by
    rw [hR0.1.len_count, hmc0]
    have h := nilN_le_usize_max; unfold nilN at h; omega
  obtain ⟨t1, hpc, hHov1, -⟩ := place_capped_domain RKS regenSz
    (fun keys i c k => ⟨true, RKS_matches keys i c k⟩)
    (fun _ => avx2_spec_holds)
    (fun keys i k _ => ⟨keys, rfl, rfl⟩)
    t0 0#u64 0#i64 0
    hR0.1.bits_pos hR0.1.len_pow hR0.1.vals_len
    (fun i hc => ⟨0#i64, (hR0.1.main_val i 0#u64 hc hns0).2⟩)
    hlenmax
    (by intro ov hsome; rw [with_bits_overflow RKS 1#u32 t0 hwb] at hsome
        exact absurd hsome (by simp))
    (by norm_num)
    (regen_sz_inv t0)
  obtain ⟨hR1, -, hbnd1⟩ := place_rep RKS RKS_matches (fun _ _ => True)
    (fun _ _ _ _ _ _ _ _ _ => trivial) hR0 trivial
    (by rw [hsum0]; unfold nilN; norm_num) hns0 (fun h => h) hpc
  have hcap1 : arenaLen t1 + mainCount t1 + 4 ≤ nilN := by have := hbnd1; unfold nilN; omega
  have hbounds : ∀ ov T, t1.overflow = some ov → OInv ov T →
      (idxs T).length ≤ 1 ∧ ((idxs T).map (remLen ov.nodes)).sum ≤ 1 := by
    intro ov T hso hI
    obtain ⟨T', hI', hl1, -, hent⟩ := hHov1 ov hso
    have hTT : T' = T := emb_unique hI'.1 hI.1
    subst T'
    have hnd : (idxs T).Nodup := nd_nodup T hI.2.2.1
    have hlen : (idxs T).length ≤ ov.nodes.val.length :=
      idxs_length_le hnd (fun j hj => by
        obtain ⟨nd, hndj, -⟩ := emb_mem hI.1 j hj
        exact (List.getElem?_eq_some_iff.mp hndj).1)
    have hrem : ∀ j ∈ idxs T, remLen ov.nodes j ≤ 1 := by
      intro j hj
      obtain ⟨nd, hndj, -⟩ := emb_mem hI.1 j hj
      have := hent j hj nd hndj
      simpa [remLen, hndj] using this
    have hsum : ((idxs T).map (remLen ov.nodes)).sum ≤ (idxs T).length * 1 :=
      sum_map_le_mul _ _ 1 hrem
    have hlen2 : (idxs T).length ≤ 1 := le_trans hlen hl1
    exact ⟨hlen2, le_trans hsum (by rw [Nat.mul_one]; exact hlen2)⟩
  have hov1 : ∀ ov T, t1.overflow = some ov → OInv ov T →
      (idxs T).length + ((idxs T).map (remLen ov.nodes)).sum < Usize.max := by
    intro ov T hso hI
    obtain ⟨h1, h2⟩ := hbounds ov T hso hI
    have h := nilN_le_usize_max; unfold nilN at h; omega
  have hvcap1 : ∀ ov T, t1.overflow = some ov → OInv ov T →
      mainCount t1 + ((idxs T).map (remLen ov.nodes)).sum + 4 ≤ nilN := by
    intro ov T hso hI
    obtain ⟨h1, h2⟩ := hbounds ov T hso hI
    have hmain : mainCount t1 ≤ 1 := by have := hbnd1; omega
    unfold nilN; omega
  have hbits1 : (1#u32).val < UScalarTy.Usize.numBits := by
    show 1 < System.Platform.numBits
    rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> decide
  obtain ⟨r, hr⟩ := resize_ok RKS regenSz
    (fun n => ⟨(), rfl, usize_val_le_two_pow64 n⟩)
    (fun keys i c k => ⟨true, RKS_matches keys i c k⟩)
    (fun _ => avx2_spec_holds)
    (fun keys i k _ => ⟨keys, rfl, rfl⟩)
    (fun keys i c _ _ => ⟨keys, (by
      change hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
        U64.Insts.Apeiron_wordHashtrinityAlgebraCoordInv core.cmp.EqU64 keys i c = ok (c, keys)
      unfold hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
      change (do let t ← (ok c : Result U64); ok (t, keys)) = ok (c, keys)
      simp only [bind_tc_ok]), rfl⟩)
    hR1 hcap1 1#u32 hbits1 (by decide) hov1 hvcap1 (regen_sz_inv t1)
  exact ⟨t0, t1, (fun _ => 0#i64), (fun x => False ∨ x = 0#u64), r,
    hpc, hR1, hcap1, hov1, hvcap1, hr⟩

/-- The power arithmetic of `grow_for_one`'s checked multiplies: at three bits of headroom below the
word width, `(2^b + 1)·4` and `2^b·3` both fit the `usize` bound.  This is the exact bound the
`len + tombs + 1` and `cap` products need; it implies `b + 1 < 64` on the 64-bit platform. -/
theorem grow_pow_bound (b : Nat) (hb : b + 3 ≤ UScalarTy.Usize.numBits) :
    (2^b + 1) * 4 ≤ Usize.max ∧ 2^b * 3 ≤ Usize.max := by
  have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by simp [Usize.max, Usize.numBits]
  have h2b : 2^b ≤ 2 ^ (UScalarTy.Usize.numBits - 3) :=
    Nat.pow_le_pow_right (by norm_num) (by omega)
  have hnb : UScalarTy.Usize.numBits = System.Platform.numBits := rfl
  rw [hmax]
  rw [hnb] at h2b
  rcases System.Platform.numBits_eq with h | h
  · rw [h] at h2b ⊢; norm_num at h2b ⊢; constructor <;> nlinarith
  · rw [h] at h2b ⊢; norm_num at h2b ⊢; constructor <;> nlinarith

/-- `with_bits 1` at `Regen<u64>` returns a table whose `bits` is `1`, `len` and `tombs` are `0`. -/
theorem with_bits1_RKS_ex' :
    ∃ t : Tbl (hashtrinity.algebra.Regen U64),
      hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI RKS 1#u32 = ok t ∧
      t.bits = 1#u32 ∧ t.len.val = 0 ∧ t.tombs.val = 0 := by
  unfold hashtrinity.Table.with_bits
  rw [shl_1_1]
  simp only [bind_tc_ok]
  obtain ⟨v, hv⟩ := from_elem_ok 2#usize
  rw [hv]; simp only [bind_tc_ok]
  unfold alloc.vec.Vec.into_boxed_slice
  simp only [bind_tc_ok]
  obtain ⟨s1, hs1⟩ := alloc_uninit_ok I64 2#usize
  rw [hs1]; simp only [bind_tc_ok]
  simp only [hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.alloc,
    bind_tc_ok]
  unfold hashtrinity.side.SideStore.empty
  simp only [hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.alloc,
    bind_tc_ok]
  exact ⟨_, rfl, rfl, rfl, rfl⟩

/-- **`grow_for_one_ok`.**  `Table::grow_for_one` returns `ok`.  Its body reads the occupancy
`len + tombs` and compares `(len + tombs + 1)·LOAD_DEN` with `cap·LOAD_NUM`; with three bits of
headroom (`bits + 3 ≤ numBits`) and occupancy at most the slot count (`len + tombs ≤ cap`,
`cap = 2^bits`), every checked `+`/`*`/`/` on the `usize` path returns `ok`, and the `then` branch's
`bits` / `bits + 1` both stay below the word width; that resize step is `resize_ok_hov`.  The `else`
branch returns the table unchanged. -/
theorem grow_for_one_ok {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (sz : KS → Nat)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks' ∧ n.val ≤ sz ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (hTK : ∀ (keys : KS) (i : Usize) (c : U64), i.val < sz keys → ¬ IsSent c →
      ∃ keys', ks.take_key keys i c = ok (c, keys') ∧ sz keys' = sz keys)
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (n : Nat) (hHov : Hov s n) (hmain : mainCount s ≤ n) (hn : n * n + n + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length)
    (hsz_s : s.coords.val.length ≤ sz s.keys) :
    ∃ s1, hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok s1 := by
  have hlenpow : s.coords.length = 2 ^ s.bits.val := hR.1.len_pow
  have hbig := grow_pow_bound s.bits.val hbits
  have h_occ : s.len.val + s.tombs.val ≤ 2 ^ s.bits.val := by rw [← hlenpow]; exact hocc
  have h_lt : 2 ^ s.bits.val < 2 ^ UScalarTy.Usize.numBits :=
    (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mpr (by omega)
  have h_i : s.len.val + s.tombs.val < 2 ^ UScalarTy.Usize.numBits := lt_of_le_of_lt h_occ h_lt
  have h_i1 : s.len.val + s.tombs.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
    have h6 : s.len.val + s.tombs.val + 1 ≤ 2 ^ s.bits.val + 1 := by omega
    have h7 : 2 ^ s.bits.val + 1 ≤ Usize.max := by omega
    have hmax : Usize.max = 2 ^ UScalarTy.Usize.numBits - 1 := by
      simp [Usize.max, Usize.numBits]
    rw [hmax] at h7
    omega
  have h_mul : (s.len.val + s.tombs.val + 1) * 4 ≤ UScalar.max UScalarTy.Usize := by
    rw [UScalar.max_USize_eq]
    have h6 : s.len.val + s.tombs.val + 1 ≤ 2 ^ s.bits.val + 1 := by omega
    have := hbig.1
    omega
  have h_mul3 : s.coords.length * 3 ≤ UScalar.max UScalarTy.Usize := by
    rw [UScalar.max_USize_eq, hlenpow]; exact hbig.2
  have hnb64 : UScalarTy.Usize.numBits ≤ 64 := by
    rw [show UScalarTy.Usize.numBits = System.Platform.numBits from rfl]
    rcases System.Platform.numBits_eq with h | h <;> omega
  have h_bits32 : s.bits.val + 1 < 2 ^ UScalarTy.U32.numBits := by
    have hu : UScalarTy.U32.numBits = 32 := rfl
    rw [hu]
    omega
  unfold hashtrinity.Table.grow_for_one
  obtain ⟨i, hi, hiv⟩ := uadd_tot s.len s.tombs h_i
  rw [hi]; simp only [bind_tc_ok]
  obtain ⟨i1, hi1, hi1v⟩ := uadd_tot i 1#usize (by rw [hiv, val_1_usize]; exact h_i1)
  rw [hi1]; simp only [bind_tc_ok]
  obtain ⟨i2, hi2, hi2v⟩ := umul_tot i1 hashtrinity.LOAD_DEN (by
    have hd : hashtrinity.LOAD_DEN.val = 4 := by unfold hashtrinity.LOAD_DEN; rfl
    rw [hi1v, val_1_usize, hiv, hd]; exact h_mul)
  rw [hi2]; simp only [bind_tc_ok]
  obtain ⟨i3, hi3, hi3v⟩ := umul_tot (Slice.len s.coords) hashtrinity.LOAD_NUM (by
    have hn' : hashtrinity.LOAD_NUM.val = 3 := by unfold hashtrinity.LOAD_NUM; rfl
    rw [Slice.len_val, hn']; exact h_mul3)
  rw [hi3]; simp only [bind_tc_ok]
  by_cases hge : i2 >= i3
  · rw [if_pos hge]
    obtain ⟨i4, hi4, -⟩ := udiv_tot (Slice.len s.coords) 2#usize (by decide)
    rw [hi4]; simp only [bind_tc_ok]
    by_cases hlt : s.len < i4
    · rw [if_pos hlt]; simp only [bind_tc_ok]
      exact resize_ok_hov ks sz halloc hm hA hset hTK hR hcap s.bits (by omega) hR.1.bits_pos
        n hHov hmain hn hsz_s
    · rw [if_neg hlt]
      obtain ⟨nb, hnb, hnbv⟩ := uadd_tot s.bits 1#u32 h_bits32
      rw [hnb]; simp only [bind_tc_ok]
      refine resize_ok_hov ks sz halloc hm hA hset hTK hR hcap nb ?_ ?_ n hHov hmain hn hsz_s
      · rw [hnbv, val_1_u32]; omega
      · rw [hnbv, val_1_u32]; omega
  · rw [if_neg hge]; exact ⟨s, rfl⟩

/-- **`grow_witness`.**  A joint witness for `grow_for_one_ok`'s premises at the `Regen<u64>` instance,
built from the actual constructor: the fresh `with_bits 1` table (`overflow = none`).  Since the
overflow branch is absent (`overflow = none`), `Hov t0 0` is witnessed **vacuously**; no reachable
state with `overflow = some _` is exhibited here.  `grow_for_one` on this table computes
`(0 + 0 + 1)·4 = 4 < 2·3 = cap·LOAD_NUM`, so it takes the `else` branch and returns the table. -/
theorem grow_witness :
    ∃ (s : Tbl (hashtrinity.algebra.Regen U64)) (m : Apeiron.Table.HMap U64 I64)
      (live : U64 → Prop) (n : Nat) (s1 : Tbl (hashtrinity.algebra.Regen U64)),
      RepW RKS s m live ∧
      arenaLen s + mainCount s + 4 ≤ nilN ∧
      Hov s n ∧ mainCount s ≤ n ∧ n * n + n + 4 ≤ nilN ∧
      s.bits.val + 3 ≤ UScalarTy.Usize.numBits ∧
      s.len.val + s.tombs.val ≤ s.coords.length ∧
      hashtrinity.Table.grow_for_one core.cmp.EqU64 CI RKS s = ok s1 := by
  obtain ⟨t0, hwb, hbits0, hlen0, htombs0⟩ := with_bits1_RKS_ex'
  have hR0 : RepW RKS t0 (fun _ => 0#i64) (fun _ => False) :=
    ⟨with_bits_rep RKS 1#u32 (by decide) t0 hwb, by
      unfold OvWF; rw [with_bits_overflow RKS 1#u32 t0 hwb]; trivial⟩
  have hsum0 : arenaLen t0 + mainCount t0 = 0 := (with_bits_wf RKS 1#u32 (by decide) t0 hwb).2
  have hmc0 : mainCount t0 = 0 := by omega
  have hcap0 : arenaLen t0 + mainCount t0 + 4 ≤ nilN := by unfold nilN; omega
  have hHov0 : Hov t0 0 := by
    intro ov hsome
    rw [with_bits_overflow RKS 1#u32 t0 hwb] at hsome
    exact absurd hsome (by simp)
  have hbits3 : t0.bits.val + 3 ≤ UScalarTy.Usize.numBits := by
    rw [hbits0]
    show (1 : Nat) + 3 ≤ System.Platform.numBits
    rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> decide
  have hocc0 : t0.len.val + t0.tombs.val ≤ t0.coords.length := by
    have hcl : t0.coords.length = 2 := by
      rw [(with_bits_rep RKS 1#u32 (by decide) t0 hwb).len_pow, hbits0]
      decide
    rw [hlen0, htombs0, hcl]
    omega
  obtain ⟨s1, hs1⟩ := grow_for_one_ok RKS regenSz
    (fun n => ⟨(), rfl, usize_val_le_two_pow64 n⟩)
    (fun keys i c k => ⟨true, RKS_matches keys i c k⟩)
    (fun _ => avx2_spec_holds)
    (fun keys i k _ => ⟨keys, rfl, rfl⟩)
    (fun keys i c _ _ => ⟨keys, (by
      change hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
        U64.Insts.Apeiron_wordHashtrinityAlgebraCoordInv core.cmp.EqU64 keys i c = ok (c, keys)
      unfold hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
      change (do let t ← (ok c : Result U64); ok (t, keys)) = ok (c, keys)
      simp only [bind_tc_ok]), rfl⟩)
    hR0 hcap0 0 hHov0 (by omega) (by omega) hbits3 hocc0 (regen_sz_inv t0)
  exact ⟨t0, (fun _ => 0#i64), (fun _ => False), 0, s1, hR0, hcap0, hHov0,
    by omega, by omega, hbits3, hocc0, hs1⟩

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.side_empty_ex
#audit Apeiron.MapExecuted.resize_ok
#audit Apeiron.MapExecuted.sum_map_le_mul
#audit Apeiron.MapExecuted.resize_ok_hov
#audit Apeiron.MapExecuted.resize_witness
#audit Apeiron.MapExecuted.grow_pow_bound
#audit Apeiron.MapExecuted.with_bits1_RKS_ex'
#audit Apeiron.MapExecuted.grow_for_one_ok
#audit Apeiron.MapExecuted.grow_witness
