-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (APEIRON-HT-INSERT, HT-INSERT-W30, W31, W32, W33, W34):
-- the `Stored<u64>` size carrier and its key-store totality lemmas.  W34 adds the post-state counter
-- frames of `insert_core` (`insert_core_ns_ok_lin`, `insert_core_sent_ok_lin`), the two-counter
-- `StoredPre` at `(N, B)` with `stored_insert_step`, and the fold `stored_insert_seq`.  Header
-- line: APEIRON-HT-INSERT W38.  W38 replaces the per-insert `bits ≤ B + 1` frame with the
-- logarithmic `Hbits s B N` (`2 ^ bits ≤ max (2 ^ B) (4 · N)`) and instantiates `stored_insert_seq`
-- non-vacuously at `L = 1000`.
--
-- STATUS.  Kernel-checked: `stored_insert_total`, `stored_insert_witness`, `stored_insert_keycols`,
-- `stored_insert_step` (the per-insert step, now carrying the counters `N ↦ N' ≤ 2·(N+N)+1` and the
-- `bits` budget `B ↦ B+1`), and `stored_insert_seq` (the fold over a key/value list, under an
-- exponential-in-`ks.length` budget — see `seqBudget`).  A `resize` no longer sends the counter to
-- `n * n + n`; one insert grows it by at most `2·(N+N)+1`.
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
import Map.ExecutedInsertLinear
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

/-- **`StoredPre`.**  The state invariants carried between `Stored<u64>` inserts.  `N` bounds the
counter-form overflow domain `Hov` and the joint live-entry counter `Hlive` (whose `.1` gives
`mainCount ≤ N` and which `Hent_of_Hlive` turns into the entry counter `Hent`); `B` is the `Hbits`
base exponent.  `Hbits s B N` replaces the per-insert `bits ≤ B + 1` frame: `bits` rises only on a
`grow_for_one` doubling and is `O(log L)` over a length-`L` fold, so `B` is fixed across the fold.
`tombs = 0` makes the occupancy bound `len + tombs ≤ coords.length` follow from
`len = mainCount ≤ coords.length`. -/
def StoredPre (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) (N B : Nat) : Prop :=
  RepW SKS s m live ∧ SKC s.coords s.keys ∧ SKCs s.side.coords s.side.keys ∧
  s.side.cap.val ≤ storedSz s.side.keys ∧ Hov s N ∧ Hlive s N ∧
  Hbits s B N ∧ s.tombs.val = 0

/-- A `StoredPre` state satisfies the occupancy premise: `len = mainCount` and `tombs = 0`. -/
theorem StoredPre.hocc {s : Tbl (hashtrinity.algebra.Stored U64)}
    {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop} {N B : Nat} (hP : StoredPre s m live N B) :
    s.len.val + s.tombs.val ≤ s.coords.length := by
  obtain ⟨hR, -, -, -, -, -, -, htombs⟩ := hP
  rw [htombs, hR.1.len_count, Nat.add_zero]
  show mainCount s ≤ s.coords.val.length
  unfold mainCount
  exact List.countP_le_length

/-- A `StoredPre` state satisfies the `bits` headroom premise `s.bits.val + 3 ≤ numBits` when
`max (2 ^ B) (4 · N) ≤ 2 ^ (numBits - 4)`: `Hbits` gives `2 ^ s.bits.val ≤ 2 ^ (numBits - 4)`, so
`s.bits.val ≤ numBits - 4`. -/
theorem StoredPre.hbits {s : Tbl (hashtrinity.algebra.Stored U64)}
    {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop} {N B : Nat} (hP : StoredPre s m live N B)
    (hh : max (2 ^ B) (4 * N) ≤ 2 ^ (UScalarTy.Usize.numBits - 4)) :
    s.bits.val + 3 ≤ UScalarTy.Usize.numBits := by
  obtain ⟨-, -, -, -, -, -, hbits, -⟩ := hP
  unfold Hbits at hbits
  have hnum : 4 ≤ UScalarTy.Usize.numBits := by
    rw [show UScalarTy.Usize.numBits = System.Platform.numBits from rfl]
    rcases System.Platform.numBits_eq with h | h <;> omega
  have hle : s.bits.val ≤ UScalarTy.Usize.numBits - 4 :=
    (Nat.pow_le_pow_iff_right (by norm_num : 1 < 2)).mp (le_trans hbits hh)
  omega

/-- A `StoredPre` state satisfies the key-column size bound. -/
theorem StoredPre.hsz {s : Tbl (hashtrinity.algebra.Stored U64)}
    {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop} {N B : Nat} (hP : StoredPre s m live N B) :
    s.coords.val.length ≤ storedSz s.keys := by
  obtain ⟨-, hK, -, -, -, -, -, -⟩ := hP
  unfold storedSz
  exact le_of_eq hK.1.symm

/-- A `StoredPre` state satisfies the arena bound when `N · N + N + 4 ≤ nilN`. -/
theorem StoredPre.hcap {s : Tbl (hashtrinity.algebra.Stored U64)}
    {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop} {N B : Nat} (hP : StoredPre s m live N B)
    (hN : N * N + N + 4 ≤ nilN) : arenaLen s + mainCount s + 4 ≤ nilN := by
  obtain ⟨-, -, -, -, hHov, hlive, -, -⟩ := hP
  have hmain : mainCount s ≤ N := hlive.1
  have har : arenaLen s ≤ N := by
    unfold arenaLen
    cases hso : s.overflow with
    | none => simp
    | some ov => obtain ⟨T, _, hnodes, _, _⟩ := hHov ov hso; exact hnodes
  have hmm : N ≤ N * N := Nat.le_mul_self N
  omega

/-- **`stored_insert_step_live`.**  From a `StoredPre` state at counter `N`, base `B` and the
budget inequality `N · N + N + 4 ≤ nilN`, `word_hashtrinity_insert` returns `ok s'` at a state
satisfying `StoredPre` at `N + 1` and the same base `B`: the representation (`upd k v m`, `live ∪ {k}`)
via `insert_ns_rep`/`insert_sent_rep`, and the joint live-entry frame.  A reserved key runs no
`grow_for_one` and writes only the side store, so `bits` and `Hlive` are unchanged
(`insert_core_sent_ok_lin`, `Hlive_coords_overflow`); a non-reserved key runs the probe with the joint
`Hlive` step (`insert_core_ns_ok_live`, one entry) and the `Hbits` frame, which the doubling branch of
`grow_for_one` preserves without raising `B`; `N + 1` bounds both `Hov` and `Hlive`, so the fold is
linear in `N` with a `B` fixed across the fold. -/
theorem stored_insert_step_live {s : Tbl (hashtrinity.algebra.Stored U64)}
    {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop} {N B : Nat}
    (hP : StoredPre s m live N B) (hN : N * N + N + 4 ≤ nilN)
    (hB : max (2 ^ B) (4 * N) ≤ 2 ^ (UScalarTy.Usize.numBits - 4)) (k : U64) (v : I64) :
    ∃ (s' : Tbl (hashtrinity.algebra.Stored U64)),
      word_hashtrinity_insert s k v = ok s' ∧
      StoredPre s' (upd k v m) (fun c => live c ∨ c = k) (N + 1) B := by
  have hocc : s.len.val + s.tombs.val ≤ s.coords.length := StoredPre.hocc hP
  have hcap : arenaLen s + mainCount s + 4 ≤ nilN := StoredPre.hcap hP hN
  have hsz : s.coords.val.length ≤ storedSz s.keys := StoredPre.hsz hP
  have hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits := StoredPre.hbits hP hB
  obtain ⟨hR, hK, hKS, hcapsz, hHov, hlive, hHbits, htombs⟩ := hP
  have hHent : Hent s N := Hent_of_Hlive hlive
  have hmain : mainCount s ≤ N := hlive.1
  by_cases hk : IsSent k
  · obtain ⟨s', h, hK', hKS', hcap', hHov', -, -, htombs', hbits'⟩ :=
      insert_core_sent_ok_lin SKS SKS_matches storedSz sks_alloc_total sks_set_total SKCs
        stored_takeKeySideOK SKC hR hK hcapsz hKS hk v
        (fun ss' hp => stored_side_push_pc s.side k v hKS hcapsz hp) N N hHov hHent hmain htombs
    have hmw : hashtrinity.Table.insert_merge_last_write core.cmp.EqU64 CI SKS s k v = ok s' := by
      unfold hashtrinity.Table.insert_merge_last_write
      rw [h]; simp only [bind_tc_ok, uncurry_apply_pair]
    have hword : word_hashtrinity_insert s k v = ok s' := by
      unfold word_hashtrinity_insert hashtrinity.Table.insert_merge_last_write
      rw [h]; simp only [bind_tc_ok, uncurry_apply_pair]
    refine ⟨s', hword, ?_⟩
    obtain ⟨ss, hs', hc⟩ := insert_sent_rep SKS SKS_matches hR.1 k v hk s' hmw
    have hHbits' : Hbits s' B (N + 1) := by
      unfold Hbits at hHbits ⊢
      rw [hbits']
      exact le_trans hHbits (max_le_max (le_refl _) (by omega))
    subst hs'
    refine ⟨⟨hc, hR.2⟩, hK', hKS', hcap', Hov_mono hHov' (by omega),
      Hlive_mono ((Hlive_coords_overflow (t := s) rfl rfl).mpr hlive) (by omega),
      hHbits', htombs'⟩
  · obtain ⟨s', h, hK', hKS', hcap', hHov', -, hHLiv', -, htombs', -, hframe⟩ :=
      insert_core_ns_ok_live SKS SKC resizeOK_stored storedSz sks_alloc_total
        (fun keys i c k => ⟨true, SKS_matches keys i c k⟩) (fun _ => avx2_spec_holds)
        sks_set_total stored_takeKeyOK hR hK hcap N hHov hlive hN hbits B hHbits hocc htombs hsz
        skc_set SKCs (fun t ht => stored_grow_side s t hKS hcapsz ht) hk v
    have hmw : hashtrinity.Table.insert_merge_last_write core.cmp.EqU64 CI SKS s k v = ok s' := by
      unfold hashtrinity.Table.insert_merge_last_write
      rw [h]; simp only [bind_tc_ok, uncurry_apply_pair]
    have hsmall : arenaLen s + mainCount s < nilN := by omega
    have hrep := (insert_ns_rep SKS SKS_matches SKC skc_set resizeOK_stored hR hK hsmall hk v hmw).1
    have hword : word_hashtrinity_insert s k v = ok s' := by
      unfold word_hashtrinity_insert hashtrinity.Table.insert_merge_last_write
      rw [h]; simp only [bind_tc_ok, uncurry_apply_pair]
    exact ⟨s', hword, ⟨hrep, hK', hKS', hcap', hHov', hHLiv', hframe, htombs'⟩⟩

/-- **`stored_insert_fold`.**  `stored_insert_step_live` iterated over a key/value list: from a
`StoredPre` state at counter `N`, base `B`, the linear budget `(N + L) · (N + L) + (N + L) + 4 ≤
nilN` (`L = ks.length`) and the single monotone `bits` premise `max (2 ^ B) (4 · (N + L)) ≤ 2 ^
(numBits - 4)`, the whole fold returns `ok` at a `StoredPre` state.  `stored_insert_step_live` raises
the live counter by exactly one and leaves `B` fixed, so the budget is linear in the fold length and
the `bits` premise is checked once, not per step. -/
theorem stored_insert_fold (ks : List (U64 × I64)) :
    ∀ (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop)
      (N B : Nat),
      StoredPre s m live N B →
      (N + ks.length) * (N + ks.length) + (N + ks.length) + 4 ≤ nilN →
      max (2 ^ B) (4 * (N + ks.length)) ≤ 2 ^ (UScalarTy.Usize.numBits - 4) →
      ∃ (s' : Tbl (hashtrinity.algebra.Stored U64)) (m' : Apeiron.Table.HMap U64 I64)
        (live' : U64 → Prop) (N' B' : Nat),
        List.foldlM (fun s kv => word_hashtrinity_insert s kv.1 kv.2) s ks = ok s' ∧
        StoredPre s' m' live' N' B' := by
  induction ks with
  | nil =>
    intro s m live N B hP _ _
    exact ⟨s, m, live, N, B, rfl, hP⟩
  | cons kv rest ih =>
    intro s m live N B hP hbud hpre
    simp only [List.length_cons] at hbud hpre
    have hM : N + (rest.length + 1) = (N + 1) + rest.length := by omega
    have hstepN : N * N + N + 4 ≤ nilN := by
      have hNle : N ≤ N + (rest.length + 1) := by omega
      have h1 : N * N ≤ (N + (rest.length + 1)) * (N + (rest.length + 1)) :=
        Nat.mul_le_mul hNle hNle
      omega
    have hstep : max (2 ^ B) (4 * N) ≤ 2 ^ (UScalarTy.Usize.numBits - 4) :=
      le_trans (max_le_max (le_refl _) (by omega)) hpre
    obtain ⟨s1, hword, hP1⟩ := stored_insert_step_live hP hstepN hstep kv.1 kv.2
    have htailN : (N + 1 + rest.length) * (N + 1 + rest.length) +
        (N + 1 + rest.length) + 4 ≤ nilN := by
      rw [← hM]
      exact hbud
    have htail : max (2 ^ B) (4 * ((N + 1) + rest.length)) ≤ 2 ^ (UScalarTy.Usize.numBits - 4) := by
      rw [← hM]
      exact hpre
    obtain ⟨s', m', live', N', B', hfold, hP'⟩ :=
      ih s1 (upd kv.1 kv.2 m) (fun c => live c ∨ c = kv.1) (N + 1) B hP1 htailN htail
    refine ⟨s', m', live', N', B', ?_, hP'⟩
    rw [List.foldlM_cons, hword]
    simp only [bind_tc_ok]
    exact hfold

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

/-- **`stored_insert_seq`.**  Over a key/value list `ks` from the state `with_bits 1` builds, the fold
`List.foldlM word_hashtrinity_insert` returns `ok s'` and `RepW` of the folded map holds, under the
linear budget `ks.length · ks.length + ks.length + 4 ≤ nilN` (the `(N₀ + L)·(N₀ + L) + (N₀ + L) + 4`
of `stored_insert_fold` at `N₀ = 0`, `B₀ = 1`) and the single `bits` premise
`max (2 ^ 1) (4 · ks.length) ≤ 2 ^ (numBits - 4)`, which the `Hbits` frame checks once. -/
theorem stored_insert_seq (ks : List (U64 × I64))
    (hbud : ks.length * ks.length + ks.length + 4 ≤ nilN)
    (hpre : max (2 ^ 1) (4 * (0 + ks.length)) ≤ 2 ^ (UScalarTy.Usize.numBits - 4)) :
    ∃ (t0 : Tbl (hashtrinity.algebra.Stored U64)) (s' : Tbl (hashtrinity.algebra.Stored U64))
      (m' : Apeiron.Table.HMap U64 I64) (live' : U64 → Prop),
      hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI SKS 1#u32 = ok t0 ∧
      List.foldlM (fun s kv => word_hashtrinity_insert s kv.1 kv.2) t0 ks = ok s' ∧
      RepW SKS s' m' live' := by
  obtain ⟨t0, hwb, hbits0, hlen0, htombs0, hside, hcap0⟩ := with_bits1_SKS_ex
  have hT := stored_with_bits_rep 1#u32 (by decide) t0 hwb
  have hR0 : RepW SKS t0 (fun _ => 0#i64) (fun _ => False) :=
    ⟨hT.1, by
      unfold OvWF; rw [with_bits_overflow SKS 1#u32 t0 hwb]; trivial⟩
  have hmc0 : mainCount t0 = 0 := by
    have hsum0 : arenaLen t0 + mainCount t0 = 0 := (with_bits_wf SKS 1#u32 (by decide) t0 hwb).2
    omega
  have hHov0 : Hov t0 0 := by
    intro ov hsome
    rw [with_bits_overflow SKS 1#u32 t0 hwb] at hsome
    exact absurd hsome (by simp)
  have hHLive0 : Hlive t0 0 :=
    Hlive_of_overflow_none (with_bits_overflow SKS 1#u32 t0 hwb) (by omega)
  have hbits3 : t0.bits.val + 3 ≤ UScalarTy.Usize.numBits := by
    rw [hbits0]
    show (1 : Nat) + 3 ≤ System.Platform.numBits
    rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> decide
  have hKS : SKCs t0.side.coords t0.side.keys := by
    intro i c hi
    rw [hside] at hi
    simp at hi
  have hcs : t0.side.cap.val ≤ storedSz t0.side.keys := by
    rw [hcap0]; exact Nat.zero_le _
  have hHbits0 : Hbits t0 1 0 := by
    unfold Hbits
    rw [hbits0]
    norm_num
  have hP0 : StoredPre t0 (fun _ => 0#i64) (fun _ => False) 0 1 :=
    ⟨hR0, ⟨hT.2.1, hT.2.2⟩, hKS, hcs, hHov0, hHLive0, hHbits0, htombs0⟩
  obtain ⟨s', m', live', N', B', hfold, hP'⟩ :=
    stored_insert_fold ks t0 (fun _ => 0#i64) (fun _ => False) 0 1 hP0
      (by simpa using hbud) (by simpa using hpre)
  exact ⟨t0, s', m', live', hwb, hfold, hP'.1⟩

/-- **`stored_insert_live_witness`.**  The linear fold runs (non-vacuously) at `L = 1000`: from the
state `with_bits 1` builds, a length-1000 key/value list folds to a `RepW` state, under the linear
nilN budget `1000 · 1000 + 1000 + 4 ≤ nilN` and the single `bits` premise
`max 2 (4 · 1000) ≤ 2 ^ (numBits - 4)` (both hold for either platform width). -/
theorem stored_insert_live_witness :
    ∃ (t0 : Tbl (hashtrinity.algebra.Stored U64)) (s' : Tbl (hashtrinity.algebra.Stored U64))
      (m' : Apeiron.Table.HMap U64 I64) (live' : U64 → Prop),
      hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI SKS 1#u32 = ok t0 ∧
      List.foldlM (fun s kv => word_hashtrinity_insert s kv.1 kv.2) t0
        (List.replicate 1000 (0#u64, 0#i64)) = ok s' ∧
      RepW SKS s' m' live' :=
  stored_insert_seq (List.replicate 1000 (0#u64, 0#i64))
    (by
      rw [List.length_replicate]
      unfold nilN; norm_num)
    (by
      rw [show UScalarTy.Usize.numBits = System.Platform.numBits from rfl]
      rw [List.length_replicate]
      rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> norm_num)

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
#audit Apeiron.MapExecuted.stored_insert_step_live
#audit Apeiron.MapExecuted.with_bits1_SKS_ex
#audit Apeiron.MapExecuted.stored_insert_witness
-- W33: the linear overflow entry counter `Hent` and the linear `resize` bound `resize_hov_lin`.
#audit Apeiron.MapExecuted.remLen_eq
#audit Apeiron.MapExecuted.sum_map_le_of_le
#audit Apeiron.MapExecuted.sum_map_le_add_one
#audit Apeiron.MapExecuted.Hent
#audit Apeiron.MapExecuted.Hent_none
#audit Apeiron.MapExecuted.Hent_of_Hov
#audit Apeiron.MapExecuted.Hent_mono
#audit Apeiron.MapExecuted.place_capped_domain_hent
#audit Apeiron.MapExecuted.place_capped_hent
#audit Apeiron.MapExecuted.resize_hov_lin
#audit Apeiron.MapExecuted.grow_post_lin
-- W36: the live-entry counter `Hlive` and the per-`place_capped` joint bound.
#audit Apeiron.MapExecuted.Hlive
#audit Apeiron.MapExecuted.Hlive_mono
#audit Apeiron.MapExecuted.Hent_of_Hlive
#audit Apeiron.MapExecuted.place_capped_live
#audit Apeiron.MapExecuted.Hlive_side
#audit Apeiron.MapExecuted.Hlive_of_overflow_none
#audit Apeiron.MapExecuted.resize_loop0_live
#audit Apeiron.MapExecuted.resize_loop1_live
#audit Apeiron.MapExecuted.resize_hov_live
-- W35: the split step.  `insert_core_ns_ok_lin_gen` is the post-`grow_for_one` probe bound
-- parameterised by `(A, W)`; `insert_core_ns_ok_nr` instantiates its identity branch (no resize,
-- `N' ≤ N+1`).
#audit Apeiron.MapExecuted.insert_core_ns_ok_lin_gen
#audit Apeiron.MapExecuted.insert_core_ns_ok_nr
-- W34: the post-state counter frames, `StoredPre` at `(N, B)`, and the fold.
#audit Apeiron.MapExecuted.Hov_of_overflow
#audit Apeiron.MapExecuted.Hent_of_overflow
#audit Apeiron.MapExecuted.mainCount_of_coords
#audit Apeiron.MapExecuted.take_value_len
#audit Apeiron.MapExecuted.write_back_len
#audit Apeiron.MapExecuted.place_capped_bits_tombs
#audit Apeiron.MapExecuted.resize_loop0_bits_tombs
#audit Apeiron.MapExecuted.resize_loop1_bits_tombs
#audit Apeiron.MapExecuted.with_bits_bits_tombs
#audit Apeiron.MapExecuted.resize_bits_tombs
#audit Apeiron.MapExecuted.grow_for_one_bits_tombs
#audit Apeiron.MapExecuted.insert_core_ns_ok_lin
#audit Apeiron.MapExecuted.insert_core_sent_ok_lin
#audit Apeiron.MapExecuted.StoredPre.hocc
#audit Apeiron.MapExecuted.StoredPre.hbits
#audit Apeiron.MapExecuted.StoredPre.hsz
#audit Apeiron.MapExecuted.StoredPre.hcap
#audit Apeiron.MapExecuted.stored_insert_fold
#audit Apeiron.MapExecuted.stored_insert_seq
-- W37: the joint live-entry frame of the insert step (`Hlive` through `insert_core`) and the linear
-- fold.  `insert_core_ns_ok_lin_gen_live` re-runs `insert_core_ns_ok_lin_gen`'s probe body with the
-- joint counter; `grow_hov_live`/`grow_post_live` carry it through `grow_for_one`;
-- `insert_core_ns_ok_live` composes them; `stored_insert_step_live` gives `StoredPre`'s step at
-- `N + 1`; `stored_insert_fold`/`stored_insert_seq` use the linear `(N₀ + L)` budget.
#audit Apeiron.MapExecuted.Hlive_coords_overflow
#audit Apeiron.MapExecuted.Hlive_vacant_step
#audit Apeiron.MapExecuted.OvFull_remLen_sum
#audit Apeiron.MapExecuted.insert_core_ns_ok_lin_gen_live
#audit Apeiron.MapExecuted.grow_hov_live
#audit Apeiron.MapExecuted.grow_post_live
#audit Apeiron.MapExecuted.insert_core_ns_ok_live
#audit Apeiron.MapExecuted.stored_insert_step_live
#audit Apeiron.MapExecuted.stored_insert_live_witness
-- W38: the logarithmic `bits` frame.  `Hbits s B₀ N` (`2 ^ bits ≤ max (2 ^ B₀) (4 · N)`) replaces the
-- per-insert `bits ≤ B + 1`; `grow_for_one_bits_split` is the load-bearing branch lemma (`bits` fixed
-- or doubled, and a doubling gives `2 ^ bits ≤ 2 · len`), `grow_for_one_hbits` preserves the frame,
-- `insert_core_ns_ok_live` carries it through the step, `StoredPre` holds it (with `B` fixed across the
-- fold), and `stored_insert_live_witness` instantiates `stored_insert_seq` non-vacuously at `L = 1000`.
#audit Apeiron.MapExecuted.Hbits
#audit Apeiron.MapExecuted.Hbits_mono
#audit Apeiron.MapExecuted.grow_for_one_bits_split
#audit Apeiron.MapExecuted.grow_for_one_hbits
