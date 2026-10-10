-- The live-entry counter.
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
-- Header line: APEIRON-HT-INSERT W38.  W38 adds the logarithmic `bits` frame `Hbits`, its
-- `grow_for_one` branch lemma `grow_for_one_bits_split`/`grow_for_one_hbits`, and the `Hbits`
-- conjunct of `insert_core_ns_ok_live`.
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


/-! ## The insert step's live-entry frame

`insert_core_ns_ok_lin_gen`'s probe raises the separate counters `Hov`, `Hent` and `mainCount` to
`max (A+1) (W+1)`; the live-entry counter needs the *joint* step.  `insert_core_ns_ok_lin_gen_live`
re-runs that probe body with `place_capped_live`'s joint accounting: `Hit` leaves both measures,
`Vacant` raises `mainCount` by at most one, `Blocked` with an absent key inserts one overflow entry
and `Blocked` with a present key overwrites an entry in place; the joint bound rises by at most one.
`grow_post_live` carries `Hlive` through `grow_for_one` (`resize_hov_live` on the resize branch), and
`insert_core_ns_ok_live` composes them into the executed `insert_core` frame. -/

/-- `Hlive` reads only `coords` and `overflow`. -/
theorem Hlive_coords_overflow {KS : Type} {t t' : Tbl KS} {C : Nat}
    (hc : t'.coords = t.coords) (ho : t'.overflow = t.overflow) : Hlive t' C ↔ Hlive t C := by
  unfold Hlive mainCount
  rw [hc, ho]

/-- One `Vacant`-style step: the overflow is untouched and `mainCount` rises by at most one, so the
live-entry counter rises by at most one. -/
theorem Hlive_vacant_step {KS : Type} {t t' : Tbl KS} {C : Nat}
    (ho : t'.overflow = t.overflow) (hmc : mainCount t' ≤ mainCount t + 1) (h : Hlive t C) :
    Hlive t' (C + 1) := by
  obtain ⟨h1, h2⟩ := h
  refine ⟨by omega, fun ov T hs hI => ?_⟩
  rw [ho] at hs
  have hsum := h2 ov T hs hI
  omega

/-- An `OvFull` overflow has a single entry at every node, so its live-entry sum is its node count. -/
theorem OvFull_remLen_sum {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {o : Ov} {T : OTree}
    (hF : OvFull ks s m o T) :
    ((idxs T).map (remLen o.nodes)).sum = (idxs T).length := by
  have h : (idxs T).map (remLen o.nodes) = (idxs T).map (fun _ => 1) := by
    apply List.map_congr_left
    intro j hj
    obtain ⟨nd, hnd, -, hent, -, -⟩ := hF.2 j hj
    rw [remLen_eq o.nodes j nd hnd, hent]; rfl
  rw [h]; simp

/-- **`insert_core_ns_ok_lin_gen_live`.**  `insert_core_ns_ok_lin_gen`'s probe body with the joint
live-entry counter: from `Hlive s1 C`, the executed `insert_core` returns `ok (none, s')` with the
separate frames of `insert_core_ns_ok_lin_gen` (at `max (A+1) (W+1)`) *and* `Hlive s' (C + 1)`. -/
theorem insert_core_ns_ok_lin_gen_live {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
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
    (A W B C : Nat) (hHov1 : Hov s1 A) (hHent1 : Hent s1 A) (hHL1 : Hlive s1 C)
    (hmc1 : mainCount s1 ≤ W)
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
      Hlive s' (C + 1) ∧
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
      ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp only [hget, bind_tc_ok, ite_true]
      rw [hmu', hback]
      simp only [uncurry_apply_pair, hread, hget2, core.mem.drop,
        core.mem.maybe_uninit.MaybeUninit.new, bind_tc_ok]
    · exact (Hov_of_overflow (s := s1) rfl).mpr (Hov_mono hHov1 (by omega))
    · exact (Hent_of_overflow (s := s1) rfl).mpr (Hent_mono hHent1 (by omega))
    · exact (Hlive_coords_overflow (t := s1) (t' := { s1 with vals := back1 (some v) }) rfl rfl).mpr
        (Hlive_mono hHL1 (by omega))
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
      hSide1.2, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp only [hgetc, hgetv, hset', hi1, core.mem.maybe_uninit.MaybeUninit.new, bind_tc_ok]
      rfl
    · exact (Hov_of_overflow (s := s1) rfl).mpr (Hov_mono hHov1 (by omega))
    · exact (Hent_of_overflow (s := s1) rfl).mpr (Hent_mono hHent1 (by omega))
    · exact Hlive_vacant_step (t := s1)
        (t' := { s1 with coords := backc k, vals := backv (some v), keys := keys', len := i1 })
        rfl
        (by
          show (backc k).val.countP (fun c => decide (¬ IsSent c)) ≤ mainCount s1 + 1
          rw [hco]
          exact countP_set_le (fun c => decide (¬ IsSent c)) s1.coords.val i.val k)
        hHL1
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
    obtain ⟨ov, T, hX, hF, hcap_b, hlen_b, hnodes_b, hlenov_b, hentries_b, hsum_b, hsumC⟩ :
        ∃ (ov : Ov) (T : OTree),
          (if core.option.Option.is_none s1.overflow = true then
              (do let o1 ← hashtrinity.overflow.Overflow.new I64 CI; ok (some o1))
            else ok s1.overflow) = ok (some ov) ∧
          OvFull ks s1 m ov T ∧ ov.nodes.val.length + 4 ≤ 2 ^ 32 - 1 ∧ ov.len.val < Usize.max ∧
          ov.nodes.val.length ≤ A ∧ ov.len.val ≤ A ∧
          (∀ j ∈ idxs T, ∀ nd, ov.nodes.val[j]? = some nd → nd.entries.val.length ≤ A) ∧
          ((idxs T).map (remLen ov.nodes)).sum ≤ A ∧
          mainCount s1 + ((idxs T).map (remLen ov.nodes)).sum ≤ C := by
      cases hto : s1.overflow with
      | none =>
        let OV : Ov := { nodes := alloc.vec.Vec.new (hashtrinity.overflow.Node U64 I64),
                         free := alloc.vec.Vec.new U32,
                         root := hashtrinity.overflow.NIL,
                         len := 0#usize }
        have hnew : hashtrinity.overflow.Overflow.new I64 CI = ok OV := rfl
        obtain ⟨hF, -, hl0⟩ := ov_full_new ks s1 m OV hnew
        refine ⟨OV, .leaf, ?_, hF, by rw [hl0]; norm_num, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · simp only [core.option.Option.is_none, Option.isNone, hnew, bind_tc_ok, ite_true]
        · have h0 : OV.len.val = 0 := rfl
          rw [h0]
          exact lt_of_lt_of_le (by unfold nilN; norm_num) nilN_le_usize_max
        · rw [hl0]; omega
        · have h0 : OV.len.val = 0 := rfl
          rw [h0]; omega
        · intro j hj; simp [idxs] at hj
        · simp [idxs]
        · have h1 := hHL1.1
          simp [idxs]
          omega
      | some ov0 =>
        obtain ⟨T, hF, -⟩ := ov_full_of_some ks hR1 hto
        obtain ⟨T', hI', hnn, hln, hent'⟩ := hHov1 ov0 hto
        have hTT : T' = T := emb_unique hI'.1 hF.1.1
        have hent'' : ∀ j ∈ idxs T, ∀ nd, ov0.nodes.val[j]? = some nd →
            nd.entries.val.length ≤ A :=
          fun j hj => hent' j (hTT.symm ▸ hj)
        have hsum := hHent1 ov0 T hto hF.1
        refine ⟨ov0, T, ?_, hF, ?_, ?_, hnn, hln, hent'', hsum, ?_⟩
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
        · exact hHL1.2 ov0 T hto hF.1
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
        ?_, ?_, ?_, ?_, ?_, ?_⟩
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
      · refine ⟨?_, ?_⟩
        · show mainCount s1 ≤ C + 1
          omega
        · intro ov' Tg hsome hI2
          have hov' : ov' = ov1 := (Option.some.inj hsome).symm
          rw [hov'] at hI2 ⊢
          have hTT : Tg = T' := emb_unique hI2.1 hI'.1
          rw [hTT]
          have hc : mainCount ({ s1 with overflow := some ov1 } : Tbl KS) = mainCount s1 :=
            mainCount_of_coords rfl
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
        ?_, ?_, ?_, ?_, ?_, ?_⟩
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
      · refine ⟨?_, ?_⟩
        · show mainCount s1 ≤ C + 1
          omega
        · intro ov' Tg hsome hI2
          have hov' : ov' = ov2 := (Option.some.inj hsome).symm
          rw [hov'] at hI2 ⊢
          have hTT : Tg = T := emb_unique hI2.1 hF2.1.1
          rw [hTT]
          have hc : mainCount ({ s1 with overflow := some ov2 } : Tbl KS) = mainCount s1 :=
            mainCount_of_coords rfl
          have hsum2 := OvFull_remLen_sum ks hF2
          have hsumo := OvFull_remLen_sum ks hF
          rw [hsum2] at *
          rw [hsumo] at hsumC
          omega
      · show mainCount s1 ≤ max (A + 1) (W + 1)
        omega
      · show s1.tombs.val = 0
        omega
      · show s1.bits.val ≤ B
        exact hgbits

/-- **`Hbits`.**  The `bits`-frame that a per-insert `+1` counter cannot carry: the table size
`2 ^ s.bits.val` is at most `max (2 ^ B₀) (4 · N)`, where `N` is the `Hlive` counter.  The `2 ^ B₀`
disjunct covers the initial state; the `4 · N` disjunct is what one `grow_for_one` doubling
preserves.  `grow_for_one` doubles only when `len ≥ cap / 2`, so
`2 ^ (bits + 1) = 2 · cap ≤ 4 · len ≤ 4 · N`; because `N` rises by one per insert, the frame is
`O(log L)` over a length-`L` fold, not the `O(L)` of a per-call `bits + 1`. -/
def Hbits {KS : Type} (s : Tbl KS) (B₀ N : Nat) : Prop :=
  2 ^ s.bits.val ≤ max (2 ^ B₀) (4 * N)

/-- `Hbits` is monotone in its `Hlive` counter. -/
theorem Hbits_mono {KS : Type} {s : Tbl KS} {B₀ N N' : Nat} (h : Hbits s B₀ N) (hNN : N ≤ N') :
    Hbits s B₀ N' := by
  unfold Hbits at h ⊢
  exact le_trans h (max_le_max (le_refl _) (by omega))

/-- **`grow_for_one_bits_split`.**  `grow_for_one` changes `bits` only on its `resize` branch: the
identity branch returns the table unchanged and a same-size rebuild sets `bits := nb = s.bits`, so
`bits` is fixed; the doubling branch sets `bits := s.bits + 1` and is taken only when
`¬ (s.len < cap / 2)`, i.e. `cap / 2 ≤ s.len`, so `2 ^ s.bits.val = cap ≤ 2 · s.len.val`.  (`cap =
s.coords.length = 2 ^ s.bits.val` is the `RepW` shape clause `hlen`.) -/
theorem grow_for_one_bits_split {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    {s s1 : Tbl KS} (hbits_pos : 0 < s.bits.val) (hlen : s.coords.length = 2 ^ s.bits.val)
    (hg : hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok s1) :
    s1.bits = s.bits ∨ (s1.bits.val = s.bits.val + 1 ∧ 2 ^ s.bits.val ≤ 2 * s.len.val) := by
  unfold hashtrinity.Table.grow_for_one at hg
  obtain ⟨i, -, hg⟩ := bind_inv hg
  obtain ⟨i1, -, hg⟩ := bind_inv hg
  obtain ⟨i2, -, hg⟩ := bind_inv hg
  obtain ⟨i3, -, hg⟩ := bind_inv hg
  by_cases hge : i2 >= i3
  · rw [if_pos hge] at hg
    obtain ⟨i4, hi4, hg⟩ := bind_inv hg
    obtain ⟨nb, hnb, hg⟩ := bind_inv hg
    by_cases hlt : s.len < i4
    · rw [if_pos hlt] at hnb
      left
      obtain ⟨hb, -⟩ := resize_bits_tombs ks hg
      rw [hb, (Result.ok_injective hnb).symm]
    · rw [if_neg hlt] at hnb
      right
      obtain ⟨hb, -⟩ := resize_bits_tombs ks hg
      have hnbv : nb.val = s.bits.val + 1 := by
        have h := (uadd_ok hnb).1
        have h1 : (1#u32).val = 1 := rfl
        omega
      refine ⟨by rw [hb, hnbv], ?_⟩
      have hle : i4.val ≤ s.len.val := by
        simp only [UScalar.lt_equiv] at hlt
        omega
      have hi4v : i4.val = s.coords.length / 2 := by
        have h := (udiv_ok hi4).2
        rw [Slice.len_val] at h
        have h2 : (2#usize).val = 2 := rfl
        rw [h2] at h
        exact h
      have hcap_le : s.coords.length ≤ 2 * s.len.val := by
        have hdiv : s.coords.length / 2 ≤ s.len.val := by rw [← hi4v]; exact hle
        have heven : 2 ∣ s.coords.length := by
          rw [hlen]
          obtain ⟨k, hk⟩ : ∃ k, s.bits.val = k + 1 := ⟨s.bits.val - 1, by omega⟩
          refine ⟨2 ^ k, ?_⟩
          rw [hk, pow_succ, Nat.mul_comm]
        calc s.coords.length = 2 * (s.coords.length / 2) := by
              rw [Nat.mul_comm]
              exact (Nat.div_mul_cancel heven).symm
          _ ≤ 2 * s.len.val := Nat.mul_le_mul_left 2 hdiv
      rw [hlen] at hcap_le
      exact hcap_le
  · rw [if_neg hge] at hg
    left
    rw [(Result.ok_injective hg).symm]

/-- **`grow_for_one_hbits`.**  `grow_for_one` preserves `Hbits s B₀ N` when `s.len.val ≤ N`: the
non-doubling branches leave `bits` fixed, and the doubling branch's fresh bound `2 ^ (bits + 1) ≤
4 · len ≤ 4 · N` dominates the `4 · N` disjunct.  `hlenN` is `mainCount s ≤ N` (`s.len.val =
mainCount s`); `bits_pos` and `len_pow` are the `RepW` shape clauses. -/
theorem grow_for_one_hbits {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    {s s1 : Tbl KS} {B₀ N : Nat}
    (hbits_pos : 0 < s.bits.val) (hlen : s.coords.length = 2 ^ s.bits.val)
    (hlenN : s.len.val ≤ N) (hHbits : Hbits s B₀ N)
    (hg : hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok s1) :
    Hbits s1 B₀ N := by
  rcases grow_for_one_bits_split ks hbits_pos hlen hg with h | ⟨hbv, hle⟩
  · unfold Hbits at hHbits ⊢
    rw [h]; exact hHbits
  · unfold Hbits at ⊢
    rw [hbv, pow_succ]
    have h2 : 2 ^ s.bits.val * 2 ≤ 4 * N := by
      have h3 := Nat.mul_le_mul_right 2 hle
      omega
    exact le_trans h2 (le_max_right _ _)

/-- **`grow_hov_live`.**  `grow_for_one` preserves the live-entry counter: from `Hov s C` and
`Hlive s C`, the output satisfies `Hov · C`, `Hent · C` and `Hlive · C` (the resize branch is
`resize_hov_live`; the identity branch is `s1 = s`). -/
theorem grow_hov_live {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (sz : KS → Nat)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks' ∧ n.val ≤ sz ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (KC : Slice U64 → KS → Prop) (hTK : TakeKeyOK ks sz KC)
    {s s1 : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (C : Nat) (hHov : Hov s C) (hlive : Hlive s C) (hCC : C + C + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length)
    (hsz_s : s.coords.val.length ≤ sz s.keys)
    (hg : hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok s1) :
    Hov s1 C ∧ Hent s1 C ∧ Hlive s1 C ∧ mainCount s1 ≤ C := by
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
    exact resize_hov_live ks sz halloc hm hA hset KC hTK hR hK hcap nb hbits_nb hpos C C
      hHov hlive hCC hsz_s hg
  · rw [if_neg hge] at hg
    have hs1 : s1 = s := (Result.ok_injective hg).symm
    subst hs1
    exact ⟨hHov, Hent_of_Hlive hlive, hlive, hlive.1⟩

/-- **`grow_post_live`.**  `grow_post_lin` with the live-entry counter: from `Hov s C` and
`Hlive s C`, one `grow_for_one` call returns a table with `Hov · C`, `Hent · C`, `Hlive · C` and
`mainCount ≤ C` (the shape, `RepW`, `bits` and `sz` frames are those of `grow_post_lin`). -/
theorem grow_post_live {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (KC : Slice U64 → KS → Prop) (hRes : ResizeOK ks KC)
    (sz : KS → Nat)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks' ∧ n.val ≤ sz ks')
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (hTK : TakeKeyOK ks sz KC)
    {s s1 : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (C : Nat) (hHov : Hov s C) (hlive : Hlive s C) (hCC : C + C + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length)
    (hsz_s : s.coords.val.length ≤ sz s.keys)
    (hg : hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok s1) :
    RepW ks s1 m live ∧ 0 < s1.bits.val ∧ s1.coords.length = 2 ^ s1.bits.val ∧
      s1.vals.length = s1.coords.length ∧
      arenaLen s1 + mainCount s1 + 4 ≤ nilN ∧ Hov s1 C ∧ Hent s1 C ∧ Hlive s1 C ∧
      s1.coords.val.length ≤ sz s1.keys ∧ KC s1.coords s1.keys ∧ mainCount s1 ≤ C := by
  obtain ⟨hR1, hbits1, hlen1, hsl1, hcap1, -, -, hsz1, hK1, -⟩ :=
    grow_post_lin ks KC hRes sz halloc hm hA hset hTK hR hK hcap C C hHov
      (Hent_of_Hlive hlive) hlive.1 hCC hbits hocc hsz_s hg
  obtain ⟨hHov1, hHent1, hHLive1, hmc1⟩ :=
    grow_hov_live ks sz halloc hm hA hset KC hTK hR hK hcap C hHov hlive hCC hbits hocc hsz_s hg
  exact ⟨hR1, hbits1, hlen1, hsl1, hcap1, hHov1, hHent1, hHLive1, hsz1, hK1, hmc1⟩

/-- **`insert_core_ns_ok_live`.**  The `Hlive` frame of `insert_core` on a non-reserved key: from
`Hov s C` and `Hlive s C` and `C · C + C + 4 ≤ nilN`, the executed `insert_core` returns
`ok (none, s')` with `Hov s' (C + 1)`, `Hent s' (C + 1)`, `Hlive s' (C + 1)` and
`mainCount s' ≤ C + 1`. -/
theorem insert_core_ns_ok_live {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
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
    (C : Nat) (hHov : Hov s C) (hlive : Hlive s C) (hC : C * C + C + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (B₀ : Nat) (hHbits : Hbits s B₀ C)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length) (htombs : s.tombs.val = 0)
    (hsz_s : s.coords.val.length ≤ sz s.keys)
    (hKCset : ∀ (coords coords' : Slice U64) (keys keys' : KS) (i : Usize) (k : U64),
      KC coords keys → coords'.val = coords.val.set i.val k → ks.set keys i k = ok keys' →
      KC coords' keys')
    (KCS : alloc.vec.Vec U64 → KS → Prop)
    (hSide : ∀ t, hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok t →
      KCS t.side.coords t.side.keys ∧ t.side.cap.val ≤ sz t.side.keys)
    {k : U64} (hns : ¬ IsSent k) (v : I64) :
    ∃ (s' : Tbl KS),
      hashtrinity.Table.insert_core core.cmp.EqU64 CI ks s k v true = ok (none, s') ∧
      KC s'.coords s'.keys ∧ KCS s'.side.coords s'.side.keys ∧
      s'.side.cap.val ≤ sz s'.side.keys ∧
      Hov s' (C + 1) ∧ Hent s' (C + 1) ∧ Hlive s' (C + 1) ∧ mainCount s' ≤ C + 1 ∧
      s'.tombs.val = 0 ∧ s'.bits.val ≤ s.bits.val + 1 ∧ Hbits s' B₀ (C + 1) := by
  have hmain : mainCount s ≤ C := hlive.1
  have hCC : C + C + 4 ≤ nilN := by have hmm := Nat.le_mul_self C; omega
  obtain ⟨s1, hg⟩ := grow_for_one_ok ks sz halloc hm hA hset KC hTK hR hK hcap C hHov hmain hC hbits
    hocc hsz_s
  obtain ⟨hR1, hbits1, hlen1, hsl1, hcap1, hHov1, hHent1, hHLive1, hsz1, hK1, hmc1⟩ :=
    grow_post_live ks KC hRes sz halloc hm hA hset hTK hR hK hcap C hHov hlive hCC hbits hocc hsz_s hg
  obtain ⟨hgbits, hgtombs⟩ := grow_for_one_bits_tombs ks hg
  have hSide1 := hSide s1 hg
  have htombs1 : s1.tombs.val = 0 := by omega
  have hs1f : Hbits s1 B₀ C :=
    grow_for_one_hbits ks hR.1.bits_pos hR.1.len_pow (by rw [hR.1.len_count]; exact hmain) hHbits hg
  obtain ⟨s', h, hK', hKS', hcap', hHov', hHent', hHLiv', hmain', htombs', hbits'⟩ :=
    insert_core_ns_ok_lin_gen_live ks KC sz hm hA hset KCS hR1 hK1 hSide1 hcap1 hbits1 hlen1 hsl1 hsz1
      C C s1.bits.val C hHov1 hHent1 hHLive1 hmc1 (by omega) (le_refl _) htombs1 hKCset hns v hg
  have hbits_le : s'.bits.val ≤ s.bits.val + 1 := by omega
  have hframe : Hbits s' B₀ (C + 1) := by
    unfold Hbits at hs1f ⊢
    have h2 : 2 ^ s'.bits.val ≤ max (2 ^ B₀) (4 * C) :=
      le_trans ((Nat.pow_le_pow_iff_right (by norm_num : 1 < 2)).mpr hbits') hs1f
    exact le_trans h2 (max_le_max (le_refl _) (by omega))
  exact ⟨s', h, hK', hKS', hcap', Hov_mono hHov' (by omega), Hent_mono hHent' (by omega),
    hHLiv', by omega, htombs', hbits_le, hframe⟩

end Apeiron.MapExecuted
