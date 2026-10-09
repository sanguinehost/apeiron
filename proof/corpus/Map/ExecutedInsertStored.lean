-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (APEIRON-HT-INSERT, HT-INSERT-W30, W31, W32): the
-- `Stored<u64>` size carrier and its key-store totality lemmas.
--
-- STATUS.  Kernel-checked: `stored_insert_total`, `stored_insert_witness`, `stored_insert_keycols` and
-- `stored_insert_step` (the structural half of the per-insert step; the `Hov` counter is not linear
-- across a `resize`, which sends `n` to `n * n + n`).
--
-- The size-only `take_key` premise (`i.val < sz keys → take_key keys i c = ok (c, _)`, and its
-- arbitrary-key side-path twin) is unsatisfiable at `SKS`: `Stored::take_key` reads slot `i` as a
-- `MaybeUninit U64` (= `Option U64`), fails on `none` (`Stored::alloc` writes `none`) and returns the
-- stored key, not the query (`stored_take_key_is_some`, `sks_take_key_total_false`).
--
-- The totality chain therefore carries a key-column relation `KC : Slice U64 → KS → Prop`
-- (`TakeKeyOK`, `Map/ExecutedResizeTotal.lean`; `resize_loop0`'s invariant keeps `KC t.coords t.keys`)
-- and, for the side path, `KCS : Vec U64 → KS → Prop` (`TakeKeySideOK`,
-- `Map/ExecutedInsertSentOk.lean`).  At `SKS`, `KC := SKC` (`stored_takeKeyOK`) and `KCS := SKCs`
-- (`stored_takeKeySideOK`); at `RKS` both are `fun _ _ => True`.
import Map.ExecutedInsertSentOk
import Map.ExecutedInsert
import Map.ExecutedResize
import Map.ExecutedBuildWitness
open Aeneas Aeneas.Std Result

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-- The `Stored<u64>` size carrier: the key column's length. -/
noncomputable def storedSz (g : hashtrinity.algebra.Stored U64) : Nat := g.val.length

/-- **`sks_alloc_total`.**  `SKS.alloc n` returns a key column of length `n`, so the size carrier
`storedSz` dominates `n`. -/
theorem sks_alloc_total (n : Usize) :
    ∃ ks', SKS.alloc n = ok ks' ∧ n.val ≤ storedSz ks' := by
  obtain ⟨g, hg, hgl⟩ := stored_alloc_ex n
  refine ⟨g, hg, ?_⟩
  show n.val ≤ g.val.length
  omega

/-- **`sks_set_total`.**  `SKS.set` writes one initialised slot; the column length is unchanged. -/
theorem sks_set_total (keys : hashtrinity.algebra.Stored U64) (i : Usize) (k : U64)
    (h : i.val < storedSz keys) :
    ∃ keys', SKS.set keys i k = ok keys' ∧ storedSz keys' = storedSz keys := by
  obtain ⟨g2, hg2, hgl⟩ := stored_set_ex keys i k (by simpa [storedSz] using h)
  refine ⟨g2, hg2, ?_⟩
  show g2.val.length = keys.val.length
  exact hgl

/-- **`stored_take_key_is_some`.**  A successful `SKS.take_key` forces its slot to be initialised
(`some`) and returns the stored key.  This is the kernel fact that refutes the W29 `hTK` premise at
`SKS`: `Stored::alloc` writes `none`, so the premise's `i.val < sz keys` is insufficient. -/
theorem stored_take_key_is_some (keys : hashtrinity.algebra.Stored U64) (i : Usize) (c : U64)
    (k' : U64) (keys' : hashtrinity.algebra.Stored U64)
    (h : SKS.take_key keys i c = ok (k', keys')) :
    keys.val[i.val]? = some (some k') := by
  change hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key CI
    core.cmp.EqU64 keys i c = ok (k', keys') at h
  unfold hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key at h
  obtain ⟨⟨mu, back⟩, hg, h⟩ := bind_inv h
  have hmu : keys.val[i.val]? = some mu := (get_unchecked_mut_ok keys i mu back hg).1
  obtain ⟨t, ht, h⟩ := bind_inv h
  have ht' : mu = some t := assume_init_read_ok mu t ht
  have hk : k' = t := by
    have h2 := Result.ok_injective h
    simp only [Prod.mk.injEq] at h2
    exact h2.1.symm
  subst hk
  rw [hmu, ht']

/-- **`stored_take_key_live`.**  On an initialised slot whose stored key is `c` — the key-column
invariant `SKC` at a live slot — `SKS.take_key` returns `(c, keys')`, preserves the size carrier and
leaves the column's list unchanged (it rewrites the slot with the value it read). -/
theorem stored_take_key_live (keys : hashtrinity.algebra.Stored U64) (i : Usize) (c : U64)
    (hc : keys.val[i.val]? = some (some c)) :
    ∃ keys', SKS.take_key keys i c = ok (c, keys') ∧ storedSz keys' = storedSz keys ∧
      keys'.val = keys.val := by
  have hlt : i.val < keys.val.length := (List.getElem?_eq_some_iff.mp hc).1
  change ∃ keys', hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
    CI core.cmp.EqU64 keys i c = ok (c, keys') ∧ storedSz keys' = storedSz keys ∧
      keys'.val = keys.val
  unfold hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
  obtain ⟨x, back, hx⟩ := gum_ex keys i hlt
  have hx1 : keys.val[i.val]? = some x := (get_unchecked_mut_ok keys i x back hx).1
  have hxb : back = keys.set i := (get_unchecked_mut_ok keys i x back hx).2
  have hxc : x = some c := by rw [hc] at hx1; exact (Option.some.inj hx1).symm
  subst hxc
  have hval : (back (some c)).val = keys.val := by
    rw [hxb, Slice.set_val_eq]
    apply List.ext_getElem?
    intro j
    by_cases hj : j = i.val
    · subst hj
      rw [List.getElem?_set_self hlt, hc]
    · rw [List.getElem?_set_ne (Ne.symm hj)]
  refine ⟨back (some c), ?_, ?_, hval⟩
  · rw [hx]
    rw [bind_pair_ok]
    simp only [core.mem.maybe_uninit.MaybeUninit.assume_init_read, bind_tc_ok]
  · show (back (some c)).val.length = keys.val.length
    rw [hval]

/-- **`sks_take_key_total_false`.**  The W29 `hTK` premise — in *either* its arbitrary-key
(side-path) or `c`-returning (resize-path) form — is unsatisfiable at `SKS` under the size carrier
`storedSz`.  `Stored::alloc n` builds a key column of `n` `none`s, so `i.val < storedSz keys` does
not make `SKS.take_key keys i c` return; this refutes the size-carrier form of the premise and hence
the length-only interface at `Stored<u64>`. -/
theorem sks_take_key_total_false :
    ¬ (∀ (keys : hashtrinity.algebra.Stored U64) (i : Usize) (c : U64),
        i.val < storedSz keys →
        ∃ (k' : U64) (keys' : hashtrinity.algebra.Stored U64),
          SKS.take_key keys i c = ok (k', keys')) := by
  intro H
  obtain ⟨g, hg0⟩ := alloc_uninit_ok U64 1#usize
  have hgv : g.val = [none] := by
    have h := alloc_uninit_val U64 1#usize g hg0
    simpa using h
  have h0 : (0#usize).val < storedSz g := by simp [storedSz, hgv]
  obtain ⟨k', keys', hk⟩ := H g 0#usize 0#u64 h0
  have hsome := stored_take_key_is_some g 0#usize 0#u64 k' keys' hk
  rw [hgv] at hsome
  simp at hsome

/-- `hashtrinity.EMPTY` is reserved. -/
theorem isSent_EMPTY : IsSent hashtrinity.EMPTY := by
  exact Or.inl EMPTY_val

/-- **`stored_takeKeyOK`.**  The `SKC`-parametrised `take_key` premise at `SKS`: at a live slot the
key column holds the coordinate (`SKC`), so `take_key` returns it, keeps the size carrier, and `SKC`
survives the loop's overwrite of that coordinate with `EMPTY` (a reserved value, which `SKC`
does not constrain). -/
theorem stored_takeKeyOK : TakeKeyOK SKS storedSz SKC := by
  intro coords keys i c hK hco hns
  obtain ⟨hl, hall⟩ := hK
  obtain ⟨keys', htk, hsz, hval⟩ := stored_take_key_live keys i c (hall i.val c hco hns)
  refine ⟨keys', htk, hsz, fun coords' hco' => ⟨?_, fun d c' hd hcns => ?_⟩⟩
  · show keys'.val.length = coords'.val.length
    have hl' : keys.val.length = coords.val.length := hl
    rw [hval, hco', List.length_set, hl']
  · rw [hco'] at hd
    by_cases hdi : d = i.val
    · subst hdi
      have hlt : i.val < coords.val.length := (List.getElem?_eq_some_iff.mp hco).1
      rw [List.getElem?_set_self hlt] at hd
      exact absurd (by rw [← Option.some.inj hd]; exact isSent_EMPTY) hcns
    · rw [List.getElem?_set_ne (Ne.symm hdi)] at hd
      rw [hval]
      exact hall d c' hd hcns

/-- The `Stored<u64>` side key column: `keys[i] = some c` at every side coordinate `c`.  The column
may be longer than the coordinate vector (spare capacity), unlike `SKC`. -/
def SKCs (coords : alloc.vec.Vec U64) (keys : hashtrinity.algebra.Stored U64) : Prop :=
  ∀ (i : Nat) (c : U64), coords.val[i]? = some c → keys.val[i]? = some (some c)

/-- **`stored_takeKeySideOK`.**  The side-path `take_key` premise at `SKS` under `SKCs`: every side
coordinate has an initialised key, so `take_key` succeeds (returning that key) and leaves the column
unchanged. -/
theorem stored_takeKeySideOK : TakeKeySideOK SKS storedSz SKCs := by
  intro coords keys i c hK hco
  obtain ⟨keys', htk, hsz, hval⟩ := stored_take_key_live keys i c (hK i.val c hco)
  exact ⟨c, keys', htk, hsz, fun d c' hd => by rw [hval]; exact hK d c' hd⟩

/-! ## The side key column survives `push` and `resize` -/

/-- **`sks_set_val`.**  A successful `SKS.set` rewrites one slot of the column's list with `some k`. -/
theorem sks_set_val (g : hashtrinity.algebra.Stored U64) (i : Usize) (k : U64)
    {g' : hashtrinity.algebra.Stored U64} (h : SKS.set g i k = ok g') :
    g'.val = g.val.set i.val (some k) := by
  change hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.set CI
    core.cmp.EqU64 g i k = ok g' at h
  unfold hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.set at h
  obtain ⟨mu, hmu, h⟩ := bind_inv h
  obtain ⟨⟨x, back⟩, hg, h⟩ := bind_inv h
  obtain ⟨-, hback⟩ := get_unchecked_mut_ok g i x back hg
  have := (Result.ok_injective h).symm
  subst this
  have hmu' : mu = some k := (Result.ok_injective hmu).symm
  rw [hback, hmu', Slice.set_val_eq]

/-- **`sks_set_lt`.**  A successful `SKS.set` only succeeds on an in-range index. -/
theorem sks_set_lt (g : hashtrinity.algebra.Stored U64) (i : Usize) (k : U64)
    {g' : hashtrinity.algebra.Stored U64} (h : SKS.set g i k = ok g') :
    i.val < g.val.length := by
  change hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.set CI
    core.cmp.EqU64 g i k = ok g' at h
  unfold hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.set at h
  obtain ⟨mu, hmu, h⟩ := bind_inv h
  obtain ⟨⟨x, back⟩, hg, h⟩ := bind_inv h
  obtain ⟨hmu2, -⟩ := get_unchecked_mut_ok g i x back hg
  exact (List.getElem?_eq_some_iff.mp hmu2).1

/-- **`stored_push_loop_kcs`.**  `SideStore::push_loop` copies `v`'s coordinates into the key column
the loop writes: on a source column satisfying `SKCs v`, the result again satisfies `SKCs v`.  The
loop reads each slot with `take_key` (which returns the stored coordinate) and writes it with `set`;
the partial-output invariant below tracks the already-copied prefix. -/
theorem stored_push_loop_kcs (v : alloc.vec.Vec U64)
    (grown grown1 : hashtrinity.algebra.Stored U64) (hKCS : SKCs v grown)
    {r : hashtrinity.algebra.Stored U64}
    (h : hashtrinity.side.SideStore.push_loop SKS v grown grown1 0#usize = ok r) :
    SKCs v r := by
  unfold hashtrinity.side.SideStore.push_loop at h
  refine loop_pc
    (fun (g : hashtrinity.algebra.Stored U64 × hashtrinity.algebra.Stored U64 × Usize) =>
      SKCs v g.1 ∧
      (∀ (j : Nat) (d : U64), j < g.2.2.val → v.val[j]? = some d → g.2.1.val[j]? = some (some d)))
    (fun r => SKCs v r)
    (fun g => hashtrinity.side.SideStore.push_loop.body SKS v g.1 g.2.1 g.2.2)
    ?_ (grown, grown1, 0#usize)
      ⟨hKCS, by intro j d hj; simp at hj⟩ r h
  · rintro ⟨g2, g3, i⟩ ⟨hKg, hPg⟩ cf hcf
    dsimp only at hKg hPg
    unfold hashtrinity.side.SideStore.push_loop.body at hcf
    dsimp only at hcf
    by_cases hl : i < alloc.vec.Vec.len v
    · rw [if_pos hl] at hcf
      obtain ⟨c, hc, hcf⟩ := bind_inv hcf
      obtain ⟨⟨key, g2'⟩, hkey, hcf⟩ := bind_inv hcf
      obtain ⟨g3', hset', hcf⟩ := bind_inv hcf
      obtain ⟨i2, hi2, hcf⟩ := bind_inv hcf
      have hcfv : cf = ControlFlow.cont (g2', g3', i2) := (Result.ok_injective hcf).symm
      subst hcfv
      dsimp only
      have hi : v.val[i.val]? = some c := vec_index_ok v i c hc
      obtain ⟨keys', htk, -, hvalk⟩ := stored_take_key_live g2 i c (hKg i.val c hi)
      have hkeyc : key = c := congrArg Prod.fst (Result.ok_injective (hkey.symm.trans htk))
      have hg2' : g2' = keys' := congrArg Prod.snd (Result.ok_injective (hkey.symm.trans htk))
      rw [hkeyc] at hset'
      rw [hg2']
      have hi2v : i2.val = i.val + 1 := by
        have := (uadd_ok hi2).1; rw [val_1_usize] at this; exact this
      have hsetv : g3'.val = g3.val.set i.val (some c) := sks_set_val g3 i c hset'
      have hsetlt : i.val < g3.val.length := sks_set_lt g3 i c hset'
      refine ⟨?_, ?_⟩
      · intro j d hd
        rw [hvalk]
        exact hKg j d hd
      · intro j d hj hd
        rw [hsetv]
        by_cases hji : j = i.val
        · subst hji
          have hdc : d = c := by rw [hi] at hd; exact (Option.some.inj hd).symm
          subst hdc
          rw [List.getElem?_set_self hsetlt]
        · rw [List.getElem?_set_ne (Ne.symm hji)]
          exact hPg j d (by omega) hd
    · rw [if_neg hl] at hcf
      have hcfv : cf = ControlFlow.done g3 := (Result.ok_injective hcf).symm
      subst hcfv
      dsimp only
      have hge : v.val.length ≤ i.val := by
        have h1 : ¬ i.val < (alloc.vec.Vec.len v).val := fun hh => hl ((UScalar.lt_equiv _ _).mpr hh)
        have h2 : (alloc.vec.Vec.len v).val ≤ i.val := Nat.le_of_not_lt h1
        rwa [alloc.vec.Vec.len_val] at h2
      intro j d hd
      exact hPg j d (by have := (List.getElem?_eq_some_iff.mp hd).1; omega) hd

/-- **`stored_side_push_pc`.**  `SideStore::push` of the reserved coordinate `c` (key `c`) preserves
both `SKCs` and the capacity bound.  The growth branch copies the old keys through `push_loop`
(`stored_push_loop_kcs`), writes the appended coordinate with `set`, and its fresh capacity is the
allocation size; the no-growth branch writes in place under the capacity bound. -/
theorem stored_side_push_pc (ss : hashtrinity.side.SideStore U64 I64 (hashtrinity.algebra.Stored U64))
    (c : U64) (v : I64) (hKCS : SKCs ss.coords ss.keys)
    (hcapsz : ss.cap.val ≤ storedSz ss.keys)
    {ss' : hashtrinity.side.SideStore U64 I64 (hashtrinity.algebra.Stored U64)}
    (h : hashtrinity.side.SideStore.push CI SKS ss c c v = ok ss') :
    SKCs ss'.coords ss'.keys ∧ ss'.cap.val ≤ storedSz ss'.keys := by
  unfold hashtrinity.side.SideStore.push at h
  dsimp only at h
  obtain ⟨need, hneed, h⟩ := bind_inv h
  have hneedv : need.val = ss.coords.val.length + 1 := by
    have := (uadd_ok hneed).1
    rw [alloc.vec.Vec.len_val, val_1_usize] at this
    exact this
  by_cases hg : need > ss.cap
  · rw [if_pos hg] at h
    obtain ⟨i1, hi1, h⟩ := bind_inv h
    obtain ⟨i2, hi2, h⟩ := bind_inv h
    rw [max_usize] at hi2
    have hi2f : i2 = if (4#usize).val < i1.val then i1 else 4#usize :=
      (Result.ok_injective hi2).symm
    obtain ⟨nc, hnc, h⟩ := bind_inv h
    rw [max_usize] at hnc
    have hncf : nc = if need.val < i2.val then i2 else need := (Result.ok_injective hnc).symm
    obtain ⟨grown, hgrown, h⟩ := bind_inv h
    have hgrownv : storedSz grown = nc.val := by
      obtain ⟨g, hg0, hgl0⟩ := stored_alloc_ex nc
      have : g = grown := Result.ok_injective (hg0.symm.trans hgrown)
      subst this
      simpa [storedSz] using hgl0
    have hneed_nc : need.val ≤ nc.val := by rw [hncf]; split <;> omega
    have hlenlt : ss.coords.val.length < storedSz grown := by rw [hgrownv]; omega
    obtain ⟨grown1, hgl, h⟩ := bind_inv h
    obtain ⟨grown2, hset2, h⟩ := bind_inv h
    obtain ⟨v1, hv1, h⟩ := bind_inv h
    obtain ⟨mu, hmu, h⟩ := bind_inv h
    have hmu' : mu = some v := (Result.ok_injective hmu).symm
    subst hmu'
    obtain ⟨v2, hv2, h⟩ := bind_inv h
    have hss' : ss' = { ss with coords := v1, vals := v2, keys := grown2, cap := nc } :=
      (Result.ok_injective h).symm
    subst hss'
    have hKCS1 : SKCs ss.coords grown1 := stored_push_loop_kcs ss.coords ss.keys grown hKCS hgl
    have hgl_sz : storedSz grown1 = storedSz grown :=
      side_push_loop_sz SKS storedSz sks_set_total SKCs stored_takeKeySideOK ss.coords ss.keys grown
        0#usize (le_of_lt hlenlt) hKCS hgl
    have hgl_len : grown1.val.length = storedSz grown := by simpa [storedSz] using hgl_sz
    have hsetv : grown2.val = grown1.val.set ss.coords.val.length (some c) := by
      rw [sks_set_val grown1 (alloc.vec.Vec.len ss.coords) c hset2, alloc.vec.Vec.len_val]
    have hset_sz : storedSz grown2 = storedSz grown1 := by
      obtain ⟨g2', hset2', hsz2⟩ := sks_set_total grown1 (alloc.vec.Vec.len ss.coords) c
        (by rw [hgl_sz, alloc.vec.Vec.len_val]; omega)
      have : g2' = grown2 := Result.ok_injective (hset2'.symm.trans hset2)
      subst this
      exact hsz2
    have hv1v : v1.val = ss.coords.val ++ [c] := vec_push_ok _ _ _ hv1
    refine ⟨?_, ?_⟩
    · intro j d hd
      rw [hv1v] at hd
      rw [hsetv]
      rcases append_single_get ss.coords.val c d j hd with ⟨hlt, hd'⟩ | ⟨hje, hdc⟩
      · rw [List.getElem?_set_ne (by omega)]
        exact hKCS1 j d hd'
      · subst hdc
        rw [hje, List.getElem?_set_self (by rw [hgl_len]; exact hlenlt)]
    · show nc.val ≤ storedSz grown2
      rw [hset_sz, hgl_sz]
      exact le_of_eq hgrownv.symm
  · rw [if_neg hg] at h
    obtain ⟨grown, hset, h⟩ := bind_inv h
    obtain ⟨v1, hv1, h⟩ := bind_inv h
    obtain ⟨mu, hmu, h⟩ := bind_inv h
    have hmu' : mu = some v := (Result.ok_injective hmu).symm
    subst hmu'
    obtain ⟨v2, hv2, h⟩ := bind_inv h
    have hss' : ss' = { ss with coords := v1, vals := v2, keys := grown } :=
      (Result.ok_injective h).symm
    subst hss'
    have hng : need.val ≤ ss.cap.val := by
      have := not_lt.mp (fun hh => hg ((UScalar.lt_equiv _ _).mpr hh)); exact this
    have hlenlt : ss.coords.val.length < storedSz ss.keys := by omega
    have hsetv : grown.val = ss.keys.val.set ss.coords.val.length (some c) := by
      rw [sks_set_val ss.keys (alloc.vec.Vec.len ss.coords) c hset, alloc.vec.Vec.len_val]
    have hset_sz : storedSz grown = storedSz ss.keys := by
      obtain ⟨g', hset', hsz'⟩ := sks_set_total ss.keys (alloc.vec.Vec.len ss.coords) c
        (by rw [alloc.vec.Vec.len_val]; exact hlenlt)
      have : g' = grown := Result.ok_injective (hset'.symm.trans hset)
      subst this
      exact hsz'
    have hv1v : v1.val = ss.coords.val ++ [c] := vec_push_ok _ _ _ hv1
    refine ⟨?_, ?_⟩
    · intro j d hd
      rw [hv1v] at hd
      rw [hsetv]
      rcases append_single_get ss.coords.val c d j hd with ⟨hlt, hd'⟩ | ⟨hje, hdc⟩
      · rw [List.getElem?_set_ne (by omega)]
        exact hKCS j d hd'
      · subst hdc
        rw [hje, List.getElem?_set_self (by simpa [storedSz] using hlenlt)]
    · show ss.cap.val ≤ storedSz grown
      rw [hset_sz]; exact hcapsz

/-- **`place_capped_side_frame`.**  `place_capped` never writes the side store: every verdict updates
only `vals`, `coords`/`keys`/`len` or `overflow`. -/
theorem place_capped_side_frame {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (t t' : Tbl KS) (c k : U64) (v : I64)
    (h : hashtrinity.Table.place_capped core.cmp.EqU64 CI ks t c k v = ok t') :
    t'.side = t.side := by
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
      rcases a with ⟨⟨nd, e⟩, ov1⟩
      have ht'' : t' = {t with overflow := backo (some ov1)} := (Result.ok_injective h).symm
      subst ht''
      rfl
  | Hit i =>
    obtain ⟨⟨mu, back⟩, hg, h⟩ := bind_inv h
    obtain ⟨old, -, h⟩ := bind_inv h
    obtain ⟨-, -, h⟩ := bind_inv h
    obtain ⟨mu1, hnew, h⟩ := bind_inv h
    obtain ⟨⟨x2, back1⟩, hg1, h⟩ := bind_inv h
    have ht' : t' = {t with vals := back1 mu1} := (Result.ok_injective h).symm
    subst ht'
    rfl
  | Vacant i =>
    obtain ⟨⟨x0, backc⟩, hgc, h⟩ := bind_inv h
    obtain ⟨mu, hnew, h⟩ := bind_inv h
    obtain ⟨⟨y, backv⟩, hgv, h⟩ := bind_inv h
    obtain ⟨tk, hset, h⟩ := bind_inv h
    obtain ⟨i1, hi1, h⟩ := bind_inv h
    have ht' : t' = {t with coords := backc c, vals := backv mu, keys := tk, len := i1} :=
      (Result.ok_injective h).symm
    subst ht'
    rfl

/-- **`resize_loop1_side_frame`.**  `resize_loop1` re-places overflow entries with `place_capped`, which
never writes the side store, so the loop preserves it. -/
theorem resize_loop1_side_frame {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (iter : alloc.vec.into_iter.IntoIter (U64 × U64 × I64)) (t0 : Tbl KS) {r : Tbl KS}
    (h : hashtrinity.Table.resize_loop1 core.cmp.EqU64 CI ks iter t0 = ok r) :
    r.side = t0.side := by
  unfold hashtrinity.Table.resize_loop1 at h
  refine loop_pc
    (fun (p : alloc.vec.into_iter.IntoIter (U64 × U64 × I64) × Tbl KS) => p.2.side = t0.side)
    (fun r => r.side = t0.side)
    (fun p => hashtrinity.Table.resize_loop1.body core.cmp.EqU64 CI ks p.1 p.2)
    ?_ (iter, t0) rfl r h
  rintro ⟨it, t⟩ hside cf hcf
  dsimp only at hcf
  unfold hashtrinity.Table.resize_loop1.body at hcf
  obtain ⟨⟨o, it1⟩, hn, hcf⟩ := bind_inv hcf
  rcases next_spec it it1 o hn with ⟨hnil, rfl⟩ | ⟨e, hcons, rfl⟩
  · have hdone : cf = ControlFlow.done t := (Result.ok_injective hcf).symm
    subst hdone
    exact hside
  · obtain ⟨c, k, v⟩ := e
    obtain ⟨t1, hpl, hcf⟩ := bind_inv hcf
    have hcont : cf = ControlFlow.cont (it1, t1) := (Result.ok_injective hcf).symm
    subst hcont
    dsimp only
    have hsf := place_capped_side_frame ks t t1 c k v hpl
    rw [hsf]; exact hside

/-- **`resize_loop0_side_frame`.**  `resize_loop0` only rewrites `coords`/`vals`/`keys` of its source,
so the side it hands back is the original table's side. -/
theorem resize_loop0_side_frame {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (s b0 : Tbl KS) {r : hashtrinity.side.SideStore U64 I64 KS × Option Ov × Tbl KS}
    (h : hashtrinity.Table.resize_loop0 core.cmp.EqU64 CI ks s b0 0#usize = ok r) :
    r.1 = s.side := by
  unfold hashtrinity.Table.resize_loop0 at h
  refine loop_pc (fun (p : Tbl KS × Tbl KS × Usize) => p.1.side = s.side)
    (fun r => r.1 = s.side)
    (fun p => hashtrinity.Table.resize_loop0.body core.cmp.EqU64 CI ks p.1 p.2.1 p.2.2)
    ?_ (s, b0, 0#usize) rfl r h
  rintro ⟨t, b, i⟩ hside cf hcf
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
      exact hside
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
      exact hside
  · rw [if_neg hlt] at hcf
    have hdone : cf = ControlFlow.done (t.side, t.overflow, b) := (Result.ok_injective hcf).symm
    subst hdone
    exact hside

/-- **`stored_resize_side`.**  `Table::resize` keeps the source side store: `resize_loop0` hands back
the source's side and `resize_loop1` (through `place_capped`) never writes it. -/
theorem stored_resize_side {s s' : Tbl (hashtrinity.algebra.Stored U64)} {nb : U32}
    (h : hashtrinity.Table.resize core.cmp.EqU64 CI SKS s nb = ok s') :
    s'.side = s.side := by
  unfold hashtrinity.Table.resize at h
  obtain ⟨bigger, hwb, h⟩ := bind_inv h
  obtain ⟨⟨ss, o, b1⟩, hl0, h⟩ := bind_inv h
  have hss : ss = s.side := resize_loop0_side_frame SKS s bigger hl0
  obtain ⟨ss1, hs1, h⟩ := bind_inv h
  simp only [core.mem.replace, core.option.Option.take] at h
  cases o with
  | none =>
    have hs' : s' = { b1 with side := ss } := (Result.ok_injective h).symm
    subst hs'
    exact hss
  | some ov =>
    obtain ⟨ev, hev, h⟩ := bind_inv h
    obtain ⟨iter, hiter, h⟩ := bind_inv h
    have hsf := resize_loop1_side_frame SKS iter ({ b1 with side := ss } :
      Tbl (hashtrinity.algebra.Stored U64)) h
    rw [hsf]
    exact hss

/-- **`stored_grow_side`.**  `grow_for_one` either returns the table (side unchanged) or resizes
(side reset to empty), so a `SKCs`/capacity-carrying side store survives it. -/
theorem stored_grow_side (s s1 : Tbl (hashtrinity.algebra.Stored U64))
    (hKS : SKCs s.side.coords s.side.keys) (hcapsz : s.side.cap.val ≤ storedSz s.side.keys)
    (hg : hashtrinity.Table.grow_for_one core.cmp.EqU64 CI SKS s = ok s1) :
    SKCs s1.side.coords s1.side.keys ∧ s1.side.cap.val ≤ storedSz s1.side.keys := by
  unfold hashtrinity.Table.grow_for_one at hg
  obtain ⟨i, -, hg⟩ := bind_inv hg
  obtain ⟨i1, -, hg⟩ := bind_inv hg
  obtain ⟨i2, -, hg⟩ := bind_inv hg
  obtain ⟨i3, -, hg⟩ := bind_inv hg
  by_cases hge : i2 >= i3
  · rw [if_pos hge] at hg
    obtain ⟨i4, -, hg⟩ := bind_inv hg
    obtain ⟨nb, -, hg⟩ := bind_inv hg
    have hside := stored_resize_side hg
    rw [hside]
    exact ⟨hKS, hcapsz⟩
  · rw [if_neg hge] at hg
    have hs1 : s1 = s := (Result.ok_injective hg).symm
    subst hs1
    exact ⟨hKS, hcapsz⟩


/-- **`stored_insert_total`.**  `word_hashtrinity_insert` (the `Stored<u64>` face) returns `ok` for
every key.  A reserved key goes to the side store (`insert_core_sent_ok`, `KCS := SKCs`); any other
key through `grow_for_one` and the probe (`insert_core_ns_ok`, `KC := SKC`).  The premises are the
`Regen<u64>` ones plus the two key-column facts `SKC`, `SKCs` and the side capacity bound
`side.cap ≤ storedSz side.keys`; none of them is a size-only `take_key` premise
(`sks_take_key_total_false`). -/
theorem stored_insert_total {s : Tbl (hashtrinity.algebra.Stored U64)}
    {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop} (hR : RepW SKS s m live)
    (hK : SKC s.coords s.keys) (hKS : SKCs s.side.coords s.side.keys)
    (hcapsz : s.side.cap.val ≤ storedSz s.side.keys)
    (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (n : Nat) (hHov : Hov s n) (hmain : mainCount s ≤ n) (hn : n * n + n + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length)
    (k : U64) (v : I64) :
    ∃ s', word_hashtrinity_insert s k v = ok s' := by
  by_cases hk : IsSent k
  · obtain ⟨s', h, -, -, -⟩ := insert_core_sent_ok SKS SKS_matches storedSz sks_alloc_total sks_set_total
      SKCs stored_takeKeySideOK SKC hR hK hcapsz hKS hk v
      (fun ss' hp => stored_side_push_pc s.side k v hKS hcapsz hp)
    refine ⟨s', ?_⟩
    unfold word_hashtrinity_insert hashtrinity.Table.insert_merge_last_write
    rw [h]
    simp only [bind_tc_ok, uncurry_apply_pair]
  · obtain ⟨s', h, -, -, -⟩ := insert_core_ns_ok SKS SKC resizeOK_stored storedSz sks_alloc_total
      (fun keys i c k => ⟨true, SKS_matches keys i c k⟩) (fun _ => avx2_spec_holds)
      sks_set_total stored_takeKeyOK hR hK hcap n hHov hmain hn hbits hocc
      (by have h : s.keys.val.length = s.coords.val.length := hK.1
          show s.coords.val.length ≤ s.keys.val.length
          omega)
      skc_set SKCs (fun t ht => stored_grow_side s t hKS hcapsz ht) hk v
    refine ⟨s', ?_⟩
    unfold word_hashtrinity_insert hashtrinity.Table.insert_merge_last_write
    rw [h]
    simp only [bind_tc_ok, uncurry_apply_pair]

/-- **`stored_insert_keycols`.**  Under exactly the `stored_insert_total` premises, a successful
insert returns a state whose key columns still track their coordinates: the main column satisfies
`SKC` and the side column `SKCs` with `side.cap ≤ storedSz side.keys`.  The reserved branch carries
the side facts through `SideStore::push` (`stored_side_push_pc`); the non-reserved branch through
`grow_for_one` (`stored_grow_side`) and the probe, whose `Vacant` write is `skc_set`. -/
theorem stored_insert_keycols {s : Tbl (hashtrinity.algebra.Stored U64)}
    {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop} (hR : RepW SKS s m live)
    (hK : SKC s.coords s.keys) (hKS : SKCs s.side.coords s.side.keys)
    (hcapsz : s.side.cap.val ≤ storedSz s.side.keys)
    (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (n : Nat) (hHov : Hov s n) (hmain : mainCount s ≤ n) (hn : n * n + n + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length)
    (k : U64) (v : I64) :
    ∃ s', word_hashtrinity_insert s k v = ok s' ∧
      SKC s'.coords s'.keys ∧ SKCs s'.side.coords s'.side.keys ∧
      s'.side.cap.val ≤ storedSz s'.side.keys := by
  by_cases hk : IsSent k
  · obtain ⟨s', h, hK', hKS', hcap'⟩ := insert_core_sent_ok SKS SKS_matches storedSz sks_alloc_total
      sks_set_total SKCs stored_takeKeySideOK SKC hR hK hcapsz hKS hk v
      (fun ss' hp => stored_side_push_pc s.side k v hKS hcapsz hp)
    refine ⟨s', ?_, hK', hKS', hcap'⟩
    unfold word_hashtrinity_insert hashtrinity.Table.insert_merge_last_write
    rw [h]
    simp only [bind_tc_ok, uncurry_apply_pair]
  · obtain ⟨s', h, hK', hKS', hcap'⟩ := insert_core_ns_ok SKS SKC resizeOK_stored storedSz
      sks_alloc_total (fun keys i c k => ⟨true, SKS_matches keys i c k⟩) (fun _ => avx2_spec_holds)
      sks_set_total stored_takeKeyOK hR hK hcap n hHov hmain hn hbits hocc
      (by have h : s.keys.val.length = s.coords.val.length := hK.1
          show s.coords.val.length ≤ s.keys.val.length
          omega)
      skc_set SKCs (fun t ht => stored_grow_side s t hKS hcapsz ht) hk v
    refine ⟨s', ?_, hK', hKS', hcap'⟩
    unfold word_hashtrinity_insert hashtrinity.Table.insert_merge_last_write
    rw [h]
    simp only [bind_tc_ok, uncurry_apply_pair]

/-- **`StoredPre`.**  The state invariants carried between `Stored<u64>` inserts: every
`stored_insert_total` premise except the two budget inequalities `arenaLen s + mainCount s + 4 ≤ nilN`
and `n * n + n + 4 ≤ nilN`, which are carried separately (their size grows per insert). -/
def StoredPre (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) (n : Nat) : Prop :=
  RepW SKS s m live ∧ SKC s.coords s.keys ∧ SKCs s.side.coords s.side.keys ∧
  s.side.cap.val ≤ storedSz s.side.keys ∧ Hov s n ∧ mainCount s ≤ n ∧
  s.bits.val + 3 ≤ UScalarTy.Usize.numBits ∧ s.len.val + s.tombs.val ≤ s.coords.length

/-- **`stored_insert_step` (structural half).**  From a `StoredPre` state and the two budget
inequalities, `word_hashtrinity_insert` returns `ok s'` and the *preserved* part of `StoredPre`
holds at `s'`: the representation (`upd k v m`, `live ∪ {k}`) and the key columns, side capacity
(`stored_insert_keycols`), together with the arena/main-count frame `+ 1` of `insert_ns_rep`
(a reserved key runs no `resize` and leaves the main table, so its frame is `0`).

The `Hov` counter, the `mainCount ≤ n` bound, the `bits + 3 ≤ numBits` headroom and the occupancy
bound are **not** established on `s'` here. Residual obligations: `insert_core_ns_ok` and
`insert_core_sent_ok` export no `Hov`/`mainCount` post-state frame on the probe write; a `resize`
raises `bits` with no `bits = nb` frame; the post-state occupancy needs `resize ⇒ tombs = 0`; and the
counter is quadratic across a `resize` (`resize_hov`: `n ↦ n * n + n`), so no linear `n' ≤ n + c`
holds. -/
theorem stored_insert_step {s : Tbl (hashtrinity.algebra.Stored U64)}
    {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop} {n : Nat} (hP : StoredPre s m live n)
    (hcap : arenaLen s + mainCount s + 4 ≤ nilN) (hn : n * n + n + 4 ≤ nilN)
    (k : U64) (v : I64) :
    ∃ s', word_hashtrinity_insert s k v = ok s' ∧
      RepW SKS s' (upd k v m) (fun c => live c ∨ c = k) ∧
      SKC s'.coords s'.keys ∧ SKCs s'.side.coords s'.side.keys ∧
      s'.side.cap.val ≤ storedSz s'.side.keys ∧
      arenaLen s' + mainCount s' ≤ arenaLen s + mainCount s + 1 := by
  obtain ⟨hR, hK, hKS, hcapsz, hHov, hmain, hbits, hocc⟩ := hP
  obtain ⟨s', h, hK', hKS', hcap'⟩ :=
    stored_insert_keycols hR hK hKS hcapsz hcap n hHov hmain hn hbits hocc k v
  have hsmall : arenaLen s + mainCount s < nilN := by omega
  have hmw : hashtrinity.Table.insert_merge_last_write core.cmp.EqU64 CI SKS s k v = ok s' := by
    unfold word_hashtrinity_insert at h; exact h
  refine ⟨s', h, ?_, hK', hKS', hcap', ?_⟩
  · by_cases hk : IsSent k
    · obtain ⟨ss, hs', hc⟩ := insert_sent_rep SKS SKS_matches hR.1 k v hk s' hmw
      subst hs'
      exact ⟨hc, hR.2⟩
    · exact (insert_ns_rep SKS SKS_matches SKC skc_set resizeOK_stored hR hK hsmall hk v hmw).1
  · by_cases hk : IsSent k
    · obtain ⟨ss, hs', -⟩ := insert_sent_rep SKS SKS_matches hR.1 k v hk s' hmw
      subst hs'
      show arenaLen s + mainCount s ≤ arenaLen s + mainCount s + 1
      omega
    · exact (insert_ns_rep SKS SKS_matches SKC skc_set resizeOK_stored hR hK hsmall hk v hmw).2.2

/-- `with_bits 1` at `Stored<u64>`: `bits = 1`, empty occupancy, an empty side store of capacity 0. -/
theorem with_bits1_SKS_ex :
    ∃ t : Tbl (hashtrinity.algebra.Stored U64),
      hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI SKS 1#u32 = ok t ∧
      t.bits = 1#u32 ∧ t.len.val = 0 ∧ t.tombs.val = 0 ∧
      t.side.coords.val = [] ∧ t.side.cap = 0#usize := by
  unfold hashtrinity.Table.with_bits
  rw [shl_1_1]
  simp only [bind_tc_ok]
  obtain ⟨v, hv⟩ := from_elem_ok 2#usize
  rw [hv]
  simp only [bind_tc_ok]
  unfold alloc.vec.Vec.into_boxed_slice
  simp only [bind_tc_ok]
  obtain ⟨s1, hs1⟩ := alloc_uninit_ok I64 2#usize
  rw [hs1]
  simp only [bind_tc_ok]
  obtain ⟨t, ht, -⟩ := stored_alloc_ex 2#usize
  rw [ht]
  simp only [bind_tc_ok]
  unfold hashtrinity.side.SideStore.empty
  obtain ⟨t0, ht0, -⟩ := stored_alloc_ex 0#usize
  have ht0' : SKS.alloc 0#usize = ok t0 := ht0
  rw [ht0']
  simp only [bind_tc_ok]
  exact ⟨_, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- **`stored_insert_witness`.**  The premises of `stored_insert_total` hold jointly at a reachable
state, the table `with_bits 1` builds at `Stored<u64>`, and the reserved key `EMPTY = 2^64 - 1`
inserts: `word_hashtrinity_insert` returns `ok`. -/
theorem stored_insert_witness :
    ∃ (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64)
      (live : U64 → Prop) (n : Nat) (s' : Tbl (hashtrinity.algebra.Stored U64)),
      RepW SKS s m live ∧ SKC s.coords s.keys ∧ SKCs s.side.coords s.side.keys ∧
      s.side.cap.val ≤ storedSz s.side.keys ∧
      arenaLen s + mainCount s + 4 ≤ nilN ∧
      Hov s n ∧ mainCount s ≤ n ∧ n * n + n + 4 ≤ nilN ∧
      s.bits.val + 3 ≤ UScalarTy.Usize.numBits ∧
      s.len.val + s.tombs.val ≤ s.coords.length ∧
      IsSent 18446744073709551615#u64 ∧
      word_hashtrinity_insert s 18446744073709551615#u64 0#i64 = ok s' := by
  obtain ⟨t0, hwb, hbits0, hlen0, htombs0, hside, hcap0⟩ := with_bits1_SKS_ex
  have hT := stored_with_bits_rep 1#u32 (by decide) t0 hwb
  have hR0 : RepW SKS t0 (fun _ => 0#i64) (fun _ => False) :=
    ⟨hT.1, by
      unfold OvWF; rw [with_bits_overflow SKS 1#u32 t0 hwb]; trivial⟩
  have hsum0 : arenaLen t0 + mainCount t0 = 0 := (with_bits_wf SKS 1#u32 (by decide) t0 hwb).2
  have hmc0 : mainCount t0 = 0 := by omega
  have hcap : arenaLen t0 + mainCount t0 + 4 ≤ nilN := by unfold nilN; omega
  have hHov0 : Hov t0 0 := by
    intro ov hsome
    rw [with_bits_overflow SKS 1#u32 t0 hwb] at hsome
    exact absurd hsome (by simp)
  have hbits3 : t0.bits.val + 3 ≤ UScalarTy.Usize.numBits := by
    rw [hbits0]
    show (1 : Nat) + 3 ≤ System.Platform.numBits
    rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> decide
  have hocc0 : t0.len.val + t0.tombs.val ≤ t0.coords.length := by
    have hcl : t0.coords.length = 2 := by
      rw [hT.1.len_pow, hbits0]
      decide
    rw [hlen0, htombs0, hcl]
    omega
  have hk : IsSent 18446744073709551615#u64 := by
    unfold IsSent EMPTYN TOMBN
    left
    decide
  have hKS : SKCs t0.side.coords t0.side.keys := by
    intro i c hi
    rw [hside] at hi
    simp at hi
  have hcs : t0.side.cap.val ≤ storedSz t0.side.keys := by
    rw [hcap0]; exact Nat.zero_le _
  obtain ⟨s', h⟩ := stored_insert_total hR0 ⟨hT.2.1, hT.2.2⟩ hKS hcs hcap 0 hHov0
    (by omega) (by unfold nilN; omega) hbits3 hocc0 18446744073709551615#u64 0#i64
  exact ⟨t0, (fun _ => 0#i64), (fun _ => False), 0, s', hR0, ⟨hT.2.1, hT.2.2⟩, hKS, hcs, hcap,
    hHov0, by omega, by unfold nilN; omega, hbits3, hocc0, hk, h⟩

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.sks_alloc_total
#audit Apeiron.MapExecuted.sks_set_total
#audit Apeiron.MapExecuted.stored_take_key_is_some
#audit Apeiron.MapExecuted.stored_take_key_live
#audit Apeiron.MapExecuted.sks_take_key_total_false
#audit Apeiron.MapExecuted.isSent_EMPTY
#audit Apeiron.MapExecuted.stored_takeKeyOK
#audit Apeiron.MapExecuted.stored_takeKeySideOK
#audit Apeiron.MapExecuted.sks_set_val
#audit Apeiron.MapExecuted.sks_set_lt
#audit Apeiron.MapExecuted.stored_push_loop_kcs
#audit Apeiron.MapExecuted.stored_side_push_pc
#audit Apeiron.MapExecuted.place_capped_side_frame
#audit Apeiron.MapExecuted.resize_loop1_side_frame
#audit Apeiron.MapExecuted.resize_loop0_side_frame
#audit Apeiron.MapExecuted.stored_resize_side
#audit Apeiron.MapExecuted.stored_grow_side
#audit Apeiron.MapExecuted.stored_insert_total
#audit Apeiron.MapExecuted.stored_insert_keycols
#audit Apeiron.MapExecuted.stored_insert_step
#audit Apeiron.MapExecuted.with_bits1_SKS_ex
#audit Apeiron.MapExecuted.stored_insert_witness
