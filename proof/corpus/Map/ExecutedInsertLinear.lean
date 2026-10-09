-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (APEIRON-HT-INSERT, W36): the live-entry counter.
--
-- W35 split the per-insert step into a no-resize branch (`N' ≤ N+1`) and a resize branch
-- (`N' ≤ 2·(N+N)+1`).  The resize branch is exponential because one merged counter bounds `Hov`,
-- `Hent` and `mainCount`, and `resize_hov_lin` charges the destination `n + e = 2N`, even though a
-- `resize` re-places only the *live* entries.  This module separates the live-entry count:
--
-- * `Hlive t C` is `mainCount t` plus the present overflow's live entry count (`Σ remLen` over the
--   `OInv` tree) at most `C`; with no overflow store it is `mainCount t ≤ C`.
-- * `place_capped_live`: one `place_capped` call raises `Hlive` by at most one -- jointly, so the
--   verdict either fills a main slot (`+1` to `mainCount`, overflow intact) or inserts one overflow
--   entry (`+1` to the overflow sum, `mainCount` intact).  This is the step the resize loops need to
--   carry the live count `mainCount s + es.length` through, and it is what the merged counter missed.
--
-- Header line: APEIRON-HT-INSERT W36.
import Map.ExecutedInsertOk
open Aeneas Aeneas.Std Result

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-- The live-entry counter: the number of live entries of `t` -- its non-reserved main coordinates
plus the present overflow tree's live entry count -- is at most `C`; with no overflow store it is
`mainCount t ≤ C`.  This is the quantity a `resize` preserves (it re-places each live entry once). -/
def Hlive {KS : Type} (t : Tbl KS) (C : Nat) : Prop :=
  mainCount t ≤ C ∧
  (∀ ov T, t.overflow = some ov → OInv ov T →
    mainCount t + ((idxs T).map (remLen ov.nodes)).sum ≤ C)

/-- `Hlive` is monotone in its counter. -/
theorem Hlive_mono {KS : Type} {t : Tbl KS} {C C' : Nat} (h : Hlive t C) (hCC : C ≤ C') :
    Hlive t C' :=
  ⟨le_trans h.1 hCC, fun ov T hs hI => le_trans (h.2 ov T hs hI) hCC⟩

/-- `Hlive` implies the overflow entry bound `Hent`: the overflow's live entry count is at most the
joint bound. -/
theorem Hent_of_Hlive {KS : Type} {t : Tbl KS} {C : Nat} (h : Hlive t C) : Hent t C :=
  fun ov T hs hI => by have := h.2 ov T hs hI; omega

/-- **`place_capped_live`.**  One `Table::place_capped` call raises the live-entry counter by at most
one: `Hlive t C` becomes `Hlive t' (C+1)`.  The `Hit` verdict overwrites a value in place (both
`mainCount` and the overflow intact); the `Vacant` verdict fills one main slot (`mainCount` rises by
at most one, overflow intact); the `Blocked` verdict runs one `insert_new_at` (the overflow's live
entry count rises by one, `mainCount` intact).  `place_capped_domain_hent` bounds the two measures
separately; this lemma takes their *joint* step, which is what the resize loops need.  `hK`
(`n + 4 ≤ 2^32 - 1`) supplies the arena capacity the `Blocked` verdict reads. -/
theorem place_capped_live {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (sz : KS → Nat)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (t : Tbl KS) (c : U64) (v : I64) (n : Nat) (C : Nat)
    (hbits : 0 < t.bits.val) (hlen : t.coords.length = 2 ^ t.bits.val)
    (hsl : t.vals.length = t.coords.length)
    (hinit : ∀ (i : Nat), t.coords.val[i]? = some c → ∃ z, t.vals.val[i]? = some (some z))
    (hlenmax : t.len.val < Usize.max)
    (hov : Hov t n) (hlive : Hlive t C) (hK : n + 4 ≤ 2 ^ 32 - 1)
    (hsz : t.coords.val.length ≤ sz t.keys) :
    ∃ t', hashtrinity.Table.place_capped core.cmp.EqU64 CI ks t c c v = ok t' ∧
      Hov t' (n + 1) ∧ Hlive t' (C + 1) := by
  obtain ⟨r, hr⟩ := probe_capped_keyed_ok ks hm hA t c c hbits hlen
  have hslot := probe_capped_keyed_slot_weak ks t c c r hr
  unfold hashtrinity.Table.place_capped
  rw [hr]
  simp only [bind_tc_ok]
  have hl1 : mainCount t ≤ C := hlive.1
  cases r with
  | Hit i =>
    have hcx : t.coords.val[i.val]? = some c := hslot
    have hltc : i.val < t.coords.val.length := (List.getElem?_eq_some_iff.mp hcx).1
    have hltv : i.val < t.vals.length := by rw [hsl]; exact hltc
    obtain ⟨mu, back, hget⟩ := slice_get_unchecked_mut_ex t.vals i hltv
    obtain ⟨hmu, hback⟩ := get_unchecked_mut_ok t.vals i mu back hget
    obtain ⟨z, hz⟩ := hinit i.val hcx
    have hmuz : mu = some z := Option.some.inj (hmu.symm.trans hz)
    have hread : core.mem.maybe_uninit.MaybeUninit.assume_init_read (some z) = ok z := rfl
    have hltv2 : i.val < (t.vals.set i (some z)).length := by rw [slice_set_length]; exact hltv
    obtain ⟨_, back1, hget2⟩ := slice_get_unchecked_mut_ex (t.vals.set i (some z)) i hltv2
    refine ⟨{t with vals := back1 (some v)}, ?_, ?_, ?_⟩
    · simp only [hget, hmuz, hback, bind_tc_ok]
      change (do
        let old ← core.mem.maybe_uninit.MaybeUninit.assume_init_read (some z)
        core.mem.drop old
        let mu1 ← core.mem.maybe_uninit.MaybeUninit.new v
        let back1' ← core.slice.Slice.get_unchecked_mut
          (core.slice.index.SliceIndexUsizeSlice (core.mem.maybe_uninit.MaybeUninit I64))
          (t.vals.set i (some z)) i
        ok { t with vals := back1'.2 mu1 }) = ok { t with vals := back1 (some v) }
      simp only [hread, hget2, core.mem.drop, core.mem.maybe_uninit.MaybeUninit.new, bind_tc_ok]
    · intro ov hsome
      have hsome' : t.overflow = some ov := hsome
      obtain ⟨T, hI0, hnode0, hlenov, hentries0⟩ := hov ov hsome'
      exact ⟨T, hI0, by omega, by omega, fun j hj nd hnd => by have := hentries0 j hj nd hnd; omega⟩
    · refine ⟨?_, ?_⟩
      · have hc : mainCount ({t with vals := back1 (some v)} : Tbl KS) = mainCount t :=
          mainCount_of_coords rfl
        omega
      · intro ov T hsome hI
        have hsum := hlive.2 ov T (show t.overflow = some ov from hsome) hI
        have hc : mainCount ({t with vals := back1 (some v)} : Tbl KS) = mainCount t :=
          mainCount_of_coords rfl
        omega
  | Vacant i =>
    obtain ⟨_, hcx, _⟩ := hslot
    have hltc : i.val < t.coords.val.length := (List.getElem?_eq_some_iff.mp hcx).1
    have hltv : i.val < t.vals.length := by rw [hsl]; exact hltc
    obtain ⟨x0, backc, hgetc⟩ := slice_get_unchecked_mut_ex t.coords i hltc
    obtain ⟨-, hbackc⟩ := get_unchecked_mut_ok t.coords i x0 backc hgetc
    obtain ⟨_, backv, hgetv⟩ := slice_get_unchecked_mut_ex t.vals i hltv
    obtain ⟨keys', hset', -⟩ := hset t.keys i c (by omega)
    have hltmax : t.len.val + (1#usize).val < 2 ^ UScalarTy.Usize.numBits := by
      rw [val_1_usize]
      have hl := hlenmax
      rw [Usize.max_def, Usize.numBits_def] at hl
      have hp : 0 < 2 ^ UScalarTy.Usize.numBits := Nat.two_pow_pos _
      omega
    obtain ⟨i1, hi1, -⟩ := uadd_tot t.len 1#usize hltmax
    have hmc : mainCount ({t with coords := backc c, vals := backv (some v), keys := keys', len := i1} : Tbl KS) ≤ mainCount t + 1 := by
      show (backc c).val.countP (fun c => decide (¬ IsSent c)) ≤
        t.coords.val.countP (fun c => decide (¬ IsSent c)) + 1
      rw [hbackc, Slice.set_val_eq]
      exact countP_set_le _ t.coords.val i.val c
    refine ⟨{t with coords := backc c, vals := backv (some v), keys := keys', len := i1}, ?_, ?_, ?_⟩
    · simp only [hgetc, hgetv, hset', hi1, core.mem.maybe_uninit.MaybeUninit.new, bind_tc_ok]
      rfl
    · intro ov hsome
      have hsome' : t.overflow = some ov := hsome
      obtain ⟨T, hI0, hnode0, hlenov, hentries0⟩ := hov ov hsome'
      exact ⟨T, hI0, by omega, by omega, fun j hj nd hnd => by have := hentries0 j hj nd hnd; omega⟩
    · refine ⟨by omega, ?_⟩
      intro ov T hsome hI
      have hsum := hlive.2 ov T (show t.overflow = some ov from hsome) hI
      omega
  | Blocked =>
    cases hto : t.overflow with
    | none =>
      let OV : Ov := { nodes := alloc.vec.Vec.new (hashtrinity.overflow.Node U64 I64),
                       free := alloc.vec.Vec.new U32,
                       root := hashtrinity.overflow.NIL,
                       len := 0#usize }
      have hnew : hashtrinity.overflow.Overflow.new I64 CI = ok OV := rfl
      obtain ⟨hF, -, hl0⟩ := ov_full_new ks t (fun _ => 0#i64) OV hnew
      have hlen0 : OV.len.val < Usize.max := by
        have h0 : OV.len.val = 0 := rfl
        rw [h0]
        have hge := nilN_le_usize_max
        unfold nilN at hge
        omega
      have hent0 : ∀ j ∈ idxs OTree.leaf, ∀ nd, OV.nodes.val[j]? = some nd →
          nd.entries.val.length < Usize.max := by
        intro j hj; simp [idxs] at hj
      have hcap0 : OV.nodes.val.length + 4 ≤ 2 ^ 32 - 1 := by rw [hl0]; norm_num
      obtain ⟨nd, e, ov1, hins⟩ := ov_insert_new_at_ok OV .leaf c c v hF.1 hlen0 hent0 hcap0
      refine ⟨{t with overflow := some ov1}, ?_, ?_, ?_⟩
      · simp only [core.option.Option.is_none, Option.isNone, hnew,
          core.option.Option.as_mut, bind_tc_ok]
        rw [if_pos trivial]
        simp only [bind_tc_ok]
        change (do
          let (_, ov1') ← hashtrinity.overflow.Overflow.insert_new_at CI OV c c v
          ok { t with overflow := some ov1' }) = ok { t with overflow := some ov1 }
        rw [hins]
        simp only [bind_tc_ok]
        rfl
      · intro ov hsome
        have heq : ov = ov1 := (Option.some.inj hsome).symm
        rw [heq]
        obtain ⟨T', hI', hlen', hnodes', hent', -⟩ :=
          ov_insert_new_at_domain OV ov1 OTree.leaf c c v nd e 0 hF.1
            (by rw [hl0]; unfold nilN; norm_num) (by intro j hj; simp [idxs] at hj) hins
        have hl0len : OV.len.val = 0 := rfl
        rw [hl0] at hnodes'
        rw [hl0len] at hlen'
        exact ⟨T', hI', by omega, by omega, fun j hj nd hnd => by have := hent' j hj nd hnd; omega⟩
      · refine ⟨?_, ?_⟩
        · have hc : mainCount ({t with overflow := some ov1} : Tbl KS) = mainCount t :=
            mainCount_of_coords rfl
          omega
        · intro ov T hsome hI
          have heq : ov = ov1 := (Option.some.inj hsome).symm
          rw [heq] at hI ⊢
          obtain ⟨T', hI', hlen', hnodes', hent', hsum'⟩ :=
            ov_insert_new_at_domain OV ov1 OTree.leaf c c v nd e 0 hF.1
              (by rw [hl0]; unfold nilN; norm_num) (by intro j hj; simp [idxs] at hj) hins
          have hIuniq : T' = T := emb_unique hI'.1 hI.1
          subst T'
          have hleaf : ((idxs OTree.leaf).map (remLen OV.nodes)).sum = 0 := by simp [idxs]
          rw [hleaf, Nat.add_zero] at hsum'
          have hc : mainCount ({t with overflow := some ov1} : Tbl KS) = mainCount t :=
            mainCount_of_coords rfl
          omega
    | some ov0 =>
      obtain ⟨T, hI0, hnode0, hlenov, hentries0⟩ := hov ov0 hto
      have hUsize : n < Usize.max := by
        have h := nilN_le_usize_max; unfold nilN at h; omega
      have hcap : ov0.nodes.val.length + 4 ≤ 2 ^ 32 - 1 := by omega
      obtain ⟨nd, e, ov1, hins⟩ := ov_insert_new_at_ok ov0 T c c v hI0
        (by omega) (fun j hj nd hnd => by have := hentries0 j hj nd hnd; omega) hcap
      refine ⟨{t with overflow := some ov1}, ?_, ?_, ?_⟩
      · simp only [core.option.Option.is_none, Option.isNone,
          core.option.Option.as_mut, bind_tc_ok]
        rw [if_neg (by simp)]
        simp only [bind_tc_ok]
        change (do
          let (_, ov1') ← hashtrinity.overflow.Overflow.insert_new_at CI ov0 c c v
          ok { t with overflow := some ov1' }) = ok { t with overflow := some ov1 }
        rw [hins]
        simp only [bind_tc_ok]
        rfl
      · intro ov hsome
        have heq : ov = ov1 := (Option.some.inj hsome).symm
        rw [heq]
        obtain ⟨T', hI', hlen', hnodes', hent', -⟩ :=
          ov_insert_new_at_domain ov0 ov1 T c c v nd e n hI0 (by unfold nilN; omega) hentries0 hins
        refine ⟨T', hI', by omega, by omega, fun j hj nd hnd => by have := hent' j hj nd hnd; omega⟩
      · refine ⟨?_, ?_⟩
        · have hc : mainCount ({t with overflow := some ov1} : Tbl KS) = mainCount t :=
            mainCount_of_coords rfl
          omega
        · intro ov Tg hsome hI
          have heq : ov = ov1 := (Option.some.inj hsome).symm
          rw [heq] at hI ⊢
          obtain ⟨T'', hI'', hlen', hnodes', hent', hsum'⟩ :=
            ov_insert_new_at_domain ov0 ov1 T c c v nd e n hI0 (by unfold nilN; omega) hentries0 hins
          have hIuniq : T'' = Tg := emb_unique hI''.1 hI.1
          subst T''
          have he : mainCount t + ((idxs T).map (remLen ov0.nodes)).sum ≤ C := hlive.2 ov0 T hto hI0
          have hc : mainCount ({t with overflow := some ov1} : Tbl KS) = mainCount t :=
            mainCount_of_coords rfl
          omega

/-- `Hlive` reads only a table's `coords` and `overflow`; a side-store update leaves it unchanged. -/
theorem Hlive_side {KS : Type} (t : Tbl KS) (ss : hashtrinity.side.SideStore U64 I64 KS) (C : Nat) :
    Hlive ({t with side := ss} : Tbl KS) C ↔ Hlive t C :=
  Iff.rfl

/-- A table with no overflow store satisfies `Hlive` exactly when `mainCount` is at most `C`. -/
theorem Hlive_of_overflow_none {KS : Type} {t : Tbl KS} {C : Nat}
    (ho : t.overflow = none) (hmc : mainCount t ≤ C) : Hlive t C :=
  ⟨hmc, fun ov T hs _ => by rw [ho] at hs; exact absurd hs (by simp)⟩

/-- **`resize_loop0_live`.**  `resize_loop0` re-places each live main slot of `s` once with
`place_capped`, so its destination's live-entry count is at most `mainCount s`: the loop invariant
tracks `Hlive · (live coords seen so far)` and `place_capped_live` raises it by one per placement.
This is the linear replacement for `resize_loop0_ok_hov_strong`'s `Hov · (mainCount s)` half as far
as the live count is concerned; the shape, `Hov` and `Hent` halves are re-derived alongside so the
step has the shape facts `place_capped_live` reads. -/
theorem resize_loop0_live {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (sz : KS → Nat)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (KC : Slice U64 → KS → Prop) (hTK : TakeKeyOK ks sz KC)
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (b0 : Tbl KS) (hb0 : RepW ks b0 m (LowMain s 0))
    (hcb0 : arenaLen b0 + mainCount b0 = 0) (hb0o : b0.overflow = none)
    (hb0live : Hlive b0 0)
    (hsz_s : s.coords.val.length ≤ sz s.keys) (hsz_b0 : b0.coords.val.length ≤ sz b0.keys) :
    ∃ r, hashtrinity.Table.resize_loop0 core.cmp.EqU64 CI ks s b0 0#usize = ok r ∧
      Hlive r.2.2 (mainCount s) := by
  obtain ⟨r, hr⟩ := resize_loop0_ok ks sz hm hA hset KC hTK hR hK hcap b0 hb0 hcb0 hb0o hsz_s hsz_b0
  refine ⟨r, hr, ?_⟩
  unfold hashtrinity.Table.resize_loop0 at hr
  refine loop_pc
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
      Hent p.2.1 (0 + (s.coords.val.take p.2.2.val).countP (fun c => decide (¬ IsSent c))) ∧
      p.2.2.val ≤ s.coords.val.length ∧
      p.1.coords.val.length ≤ sz p.1.keys ∧
      p.2.1.coords.val.length ≤ sz p.2.1.keys ∧
      KC p.1.coords p.1.keys ∧
      Hlive p.2.1 ((s.coords.val.take p.2.2.val).countP (fun c => decide (¬ IsSent c))))
    (fun r => Hlive r.2.2 (mainCount s))
    (fun p : Tbl KS × Tbl KS × Usize =>
      hashtrinity.Table.resize_loop0.body core.cmp.EqU64 CI ks p.1 p.2.1 p.2.2)
    ?_ (s, b0, 0#usize) ?_ r hr
  · rintro ⟨t, b, i⟩ ⟨hA1, hB, hC, hE1, hE2, hbitsb, hlenb, hslb, hBig, hlenbnd, hHov,
      hHent, hib, hszs, hszb, hKt, hHLive⟩ cf hcf
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
        have hh : cf = ControlFlow.cont (t, b, i2) := (Result.ok_injective hcf).symm
        rw [hh]; dsimp only
        have hi2v : i2.val = i.val + 1 := by
          have := (uadd_ok hi2).1; rw [val_1_usize] at this; exact this
        have hcnt : (s.coords.val.take i2.val).countP (fun c => decide (¬ IsSent c)) =
            (s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)) := by
          rw [hi2v, countP_take_succ (fun c => decide (¬ IsSent c)) s.coords.val i.val c hsc]
          simp [hsent]
        refine ⟨hA1, (fun d hd => hB d (by omega)), hC, hE1, hE2, hbitsb, hlenb, hslb, hBig,
          ?_, ?_, ?_, ?_, hszs, hszb, hKt, ?_⟩
        · rw [hcnt]; exact hlenbnd
        · rw [hcnt]; exact hHov
        · rw [hcnt]; exact hHent
        · omega
        · rw [hcnt]; exact hHLive
      · rw [if_pos (by simpa using hsent)] at hcf
        obtain ⟨⟨key, keys'⟩, htk, hcf⟩ := bind_inv hcf
        obtain ⟨kt, htk2, hszkt, hKpres⟩ := hTK t.coords t.keys i c hKt htc hsent
        have hkey : key = c := congrArg Prod.fst (Result.ok_injective (htk.symm.trans htk2))
        obtain ⟨⟨mu, back⟩, hg, hcf⟩ := bind_inv hcf
        obtain ⟨hmu, hback⟩ := get_unchecked_mut_ok t.vals i mu back hg
        obtain ⟨v, hread, hcf⟩ := bind_inv hcf
        obtain ⟨cs, hupd, hcf⟩ := bind_inv hcf
        obtain ⟨big1, hpc, hcf⟩ := bind_inv hcf
        rw [hkey] at hpc
        obtain ⟨i2, hi2, hcf⟩ := bind_inv hcf
        have hh : cf = ControlFlow.cont
            ({ t with coords := cs, vals := back mu, keys := keys' }, big1, i2) :=
          (Result.ok_injective hcf).symm
        rw [hh]; dsimp only
        have hi2v : i2.val = i.val + 1 := by
          have := (uadd_ok hi2).1; rw [val_1_usize] at this; exact this
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
        obtain ⟨hHov0, hHent0, hszb'⟩ := place_capped_hent ks sz hm hA hset b big1 c v
          ((s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)))
          (0 + (s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)))
          hbitsb hlenb hslb (fun j hcj => hBig j c hcj hsent) hlenmax hHov hHent hK hszb hpc
        obtain ⟨big1L, hpcL, -, hHLiveL⟩ := place_capped_live ks sz hm hA hset b c v
          ((s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)))
          ((s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)))
          hbitsb hlenb hslb (fun j hcj => hBig j c hcj hsent) hlenmax hHov hHLive hK hszb
        have hbig1 : big1L = big1 := Result.ok_injective (hpcL.symm.trans hpc)
        have hHLiveL' : Hlive big1
            ((s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)) + 1) :=
          hbig1 ▸ hHLiveL
        have hHov' : Hov big1 ((s.coords.val.take i2.val).countP (fun c => decide (¬ IsSent c))) := by
          rw [hcnt]; exact hHov0
        have hHent' : Hent big1
            (0 + (s.coords.val.take i2.val).countP (fun c => decide (¬ IsSent c))) := by
          rw [hcnt, ← Nat.add_assoc]; exact hHent0
        have hHLive' : Hlive big1
            ((s.coords.val.take i2.val).countP (fun c => decide (¬ IsSent c))) := by
          rw [hcnt]; exact hHLiveL'
        have hupd' : cs.val = t.coords.val.set i.val hashtrinity.EMPTY :=
          slice_update_ok t.coords cs i hashtrinity.EMPTY hupd
        have hvals : (back mu).val = s.vals.val := by
          rw [hback, Slice.set_val_eq, list_set_self _ _ _ hmu, hC]
        refine ⟨?_, ?_, hvals, hE1, hE2, ?_, ?_, ?_,
          place_capped_bigInit ks b big1 c v hpc hBig, ?_, hHov', hHent', ?_, ?_, ?_, ?_, ?_⟩
        · rw [hupd', List.length_set]; exact hA1
        · intro d hd
          rw [hupd', List.getElem?_set_ne (by omega)]
          exact hB d (by omega)
        · rw [hbitsf]; exact hbitsb
        · rw [hlenf2, hbitsf]; exact hlenb
        · rw [hslf2, hlenf2]; exact hslb
        · rw [hcnt]; omega
        · omega
        · have hkeys' : keys' = kt := congrArg Prod.snd (Result.ok_injective (htk.symm.trans htk2))
          rw [hupd', List.length_set, hkeys', hszkt]
          exact hszs
        · rw [hlenf2, hszb']; exact hszb
        · have hkeys' : keys' = kt := congrArg Prod.snd (Result.ok_injective (htk.symm.trans htk2))
          rw [hkeys']
          exact hKpres cs hupd'
        · exact hHLive'
    · rw [if_neg hlt] at hcf
      have hh : cf = ControlFlow.done (t.side, t.overflow, b) := (Result.ok_injective hcf).symm
      rw [hh]; dsimp only
      have hge : s.coords.val.length ≤ i.val := by
        have h1 : ¬ i.val < t.coords.val.length := by
          intro h
          exact hlt ((UScalar.lt_equiv i (Slice.len t.coords)).mpr (by
            rw [Slice.len_val]; simp only [Slice.length]; exact h))
        rw [hA1] at h1
        omega
      have hmain : (s.coords.val.take i.val).countP (fun c => decide (¬ IsSent c)) = mainCount s := by
        rw [List.take_of_length_le hge]; rfl
      rw [← hmain]; exact hHLive
  · refine ⟨rfl, (fun d _ => rfl), rfl, rfl, rfl, hb0.1.bits_pos, hb0.1.len_pow, hb0.1.vals_len,
      ?_, ?_, ?_, ?_, Nat.zero_le _, hsz_s, hsz_b0, hK, ?_⟩
    · exact fun j c' hc hns => ⟨m c', (hb0.1.main_val j c' hc hns).2⟩
    · have hb0len : b0.len.val = 0 := by rw [hb0.1.len_count]; omega
      rw [hb0len]; exact Nat.zero_le _
    · intro ov h
      rw [hb0o] at h
      exact absurd h (by simp)
    · simpa using Hent_of_Hlive hb0live
    · simpa using hb0live

/-- **`resize_loop1_live`.**  `resize_loop1` re-places each drained overflow entry once with
`place_capped`, so its destination's live-entry count is at most `n0 + es.length` where `n0` bounds its
input's: the loop invariant tracks `Hlive · (n0 + entries placed so far)` and `place_capped_live`
raises it by one per placement. -/
theorem resize_loop1_live {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
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
    (hent0 : Hent t0 e0) (hlive0 : Hlive t0 n0)
    (hcap : n0 + es.length + 4 ≤ nilN) (hsz0 : t0.coords.val.length ≤ sz t0.keys)
    (iter : alloc.vec.into_iter.IntoIter (U64 × U64 × I64)) (hiter : iter.val = es) :
    ∃ r, hashtrinity.Table.resize_loop1 core.cmp.EqU64 CI ks iter t0 = ok r ∧
      Hov r (n0 + es.length) ∧ Hent r (e0 + es.length) ∧ Hlive r (n0 + es.length) := by
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
        Hent p.2 (e0 + dn.length) ∧ Hlive p.2 (n0 + dn.length) ∧
        p.2.coords.val.length ≤ sz p.2.keys)
    (fun r => Hov r (n0 + es.length) ∧ Hent r (e0 + es.length) ∧ Hlive r (n0 + es.length))
    (fun p => hashtrinity.Table.resize_loop1.body core.cmp.EqU64 CI ks p.1 p.2)
    ?_ (iter, t0) ?_ r hr
  · rintro ⟨it, tb⟩ ⟨dn, hdn, hbits, hlen, hsl, hBig, hlenbnd, hHov, hHent, hHLive, hsz⟩ cf hcf
    dsimp only at *
    unfold hashtrinity.Table.resize_loop1.body at hcf
    obtain ⟨⟨o, itr⟩, hn, hcf⟩ := bind_inv hcf
    rcases next_spec it itr o hn with ⟨hnil, rfl⟩ | ⟨e, hcons, rfl⟩
    · have hcfd : cf = ControlFlow.done tb := by
        change ok (ControlFlow.done tb) = ok cf at hcf
        exact (Result.ok_injective hcf).symm
      rw [hcfd]
      have hdne : dn = es := by rw [hnil] at hdn; simpa using hdn
      rw [hdne] at hHov hHent hHLive
      exact ⟨hHov, hHent, hHLive⟩
    · obtain ⟨c0, k0, v0⟩ := e
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
      obtain ⟨tb2L, hpcL, -, hHLive0'⟩ := place_capped_live ks sz hm hA hset tb c0 v0
        (n0 + dn.length) (n0 + dn.length) hbits hlen hsl
        (fun j hcj => hBig j c0 hcj hns2) hlenmax hHov hHLive hK hsz
      have hbig : tb2L = tb2 := Result.ok_injective (hpcL.symm.trans hpc)
      have hHLive0'' : Hlive tb2 (n0 + dn.length + 1) := hbig ▸ hHLive0'
      have hHov' : Hov tb2 (n0 + (dn ++ [(c0, c0, v0)]).length) := by
        rw [hcnt]; exact hHov0'
      have hHent' : Hent tb2 (e0 + (dn ++ [(c0, c0, v0)]).length) := by
        rw [hcnt]; exact hHent0'
      have hHLive' : Hlive tb2 (n0 + (dn ++ [(c0, c0, v0)]).length) := by
        rw [hcnt, ← Nat.add_assoc]; exact hHLive0''
      refine ⟨dn ++ [(c0, c0, v0)], hdn', ?_, ?_, ?_, ?_, ?_, hHov', hHent', hHLive', ?_⟩
      · rw [hbf]; exact hbits
      · rw [hlf2, hbf]; exact hlen
      · rw [hslf2, hlf2]; exact hsl
      · exact place_capped_bigInit ks tb tb2 c0 v0 hpc hBig
      · rw [hcnt]; omega
      · rw [hlf2, hszb']; exact hsz
  · refine ⟨[], by simpa using hiter, hbits0, hlen0, hsl0, hBig0, ?_, ?_, ?_, ?_, ?_⟩
    · simpa using hlenbnd0
    · simpa using hHov0
    · simpa using hent0
    · simpa using hlive0
    · simpa using hsz0

/-- **`resize_hov_live`.**  `resize` preserves the live-entry counter: from `Hov s n` and
`Hlive s C`, the destination satisfies `Hov · C`, `Hent · C` and `Hlive · C`.  The `n + e` of
`resize_hov_lin` becomes `C`, because the drained entry list has length `es.length = Σ remLen`
(`into_entries_total`) so `mainCount s + es.length ≤ C`; the resize loops' live counts
(`resize_loop0_live`, `resize_loop1_live`) give `Hlive s1 C`.  This is the linear replacement for
`resize_hov_lin`'s `mainCount s + e` bound. -/
theorem resize_hov_live {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
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
    (n C : Nat) (hHov : Hov s n) (hlive : Hlive s C) (hnC : n + C + 4 ≤ nilN)
    (hsz_s : s.coords.val.length ≤ sz s.keys)
    {s1 : Tbl KS} (hres : hashtrinity.Table.resize core.cmp.EqU64 CI ks s new_bits = ok s1) :
    Hov s1 C ∧ Hent s1 C ∧ Hlive s1 C ∧ mainCount s1 ≤ C := by
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
  have hb0live : Hlive bigger 0 := Hlive_of_overflow_none hovnone (by omega)
  obtain ⟨r, hr, hbr, hcr, hvr, hBigr, hlenr, hHovr, hHentr, hside, hovf, hszr⟩ :=
    resize_loop0_ok_hov_strong ks sz hm hA hset KC hTK hR hK hcap bigger hb0 hcb0 hovnone 0
      (Hent_none hovnone) hsz_s hszbb
  obtain ⟨rL, hrL, hHLive0⟩ :=
    resize_loop0_live ks sz hm hA hset KC hTK hR hK hcap bigger hb0 hcb0 hovnone hb0live hsz_s hszbb
  have hrr : rL = r := Result.ok_injective (hrL.symm.trans hr)
  have hHL0 : Hlive r.2.2 (mainCount s) := hrr ▸ hHLive0
  rw [hr] at hres; simp only [bind_tc_ok] at hres
  obtain ⟨ss1, hss1, hres⟩ := bind_inv hres
  rcases r with ⟨ss0, o, b1⟩
  simp only [core.mem.replace, core.option.Option.take] at hres
  cases o with
  | none =>
    have hle : mainCount s ≤ C := hlive.1
    have hs1 : Hov s1 (mainCount s) := by
      rw [(Result.ok_injective hres).symm]
      exact hHovr
    have he1 : Hent s1 (0 + mainCount s) := by
      rw [(Result.ok_injective hres).symm]
      exact hHentr
    have hl1 : Hlive s1 (mainCount s) := by
      rw [(Result.ok_injective hres).symm]
      rw [Hlive_side]
      exact hHL0
    exact ⟨Hov_mono hs1 hle, Hent_mono he1 (by omega), Hlive_mono hl1 hle,
      (Hlive_mono hl1 hle).1⟩
  | some ov =>
    have hso : s.overflow = some ov := hovf.symm
    have hov : ∀ ov T, s.overflow = some ov → OInv ov T →
        (idxs T).length + ((idxs T).map (remLen ov.nodes)).sum < Usize.max := by
      intro ov T hs hI
      have h1 := hlen ov T hs hI
      have hsumC : ((idxs T).map (remLen ov.nodes)).sum ≤ C := by
        have := hlive.2 ov T hs hI; omega
      have h3 : n + C < Usize.max := by
        have h := nilN_le_usize_max
        unfold nilN at h
        have hnC' : n + C + 4 ≤ 2 ^ 32 - 1 := hnC
        omega
      omega
    have hvcap : ∀ ov T, s.overflow = some ov → OInv ov T →
        mainCount s + ((idxs T).map (remLen ov.nodes)).sum + 4 ≤ nilN := by
      intro ov T hs hI
      have h2 := hlive.2 ov T hs hI
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
    have hHlive_t0 : Hlive ({ b1 with side := ss0 } : Tbl KS) (mainCount s) := by
      rw [Hlive_side]; exact hHL0
    obtain ⟨r2, hr2, hHov2, hHent2, hHLive2⟩ :=
      resize_loop1_live ks sz hm hA hset es.val hes (t0 := { b1 with side := ss0 })
        (mainCount s) (0 + mainCount s)
        hbr hcr hvr hBigr hlenr hHovr
        (show Hent { b1 with side := ss0 } (0 + mainCount s) from hHentr)
        hHlive_t0 hcap1 (by simpa using hszr) iter (by rw [hies])
    have hs1eq : s1 = r2 := Result.ok_injective (hres.symm.trans hr2)
    subst hs1eq
    have hle : mainCount s + es.val.length ≤ C := by
      have h2 := hlive.2 ov T hso hI
      omega
    exact ⟨Hov_mono hHov2 hle, Hent_mono hHent2 (by omega),
      Hlive_mono hHLive2 hle, (Hlive_mono hHLive2 hle).1⟩

end Apeiron.MapExecuted
