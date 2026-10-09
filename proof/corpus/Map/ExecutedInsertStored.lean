-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (HT-INSERT-W30): the
-- `Stored<u64>` size carrier and its key-store totality lemmas.
--
-- STATUS.  Kernel-checked: `stored_insert_total` and `stored_insert_witness`.
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
  · obtain ⟨s', h⟩ := insert_core_sent_ok SKS SKS_matches storedSz sks_alloc_total sks_set_total
      SKCs stored_takeKeySideOK hR hcapsz hKS hk v
    refine ⟨s', ?_⟩
    unfold word_hashtrinity_insert hashtrinity.Table.insert_merge_last_write
    rw [h]
    simp only [bind_tc_ok, uncurry_apply_pair]
  · obtain ⟨s', h⟩ := insert_core_ns_ok SKS SKC resizeOK_stored storedSz sks_alloc_total
      (fun keys i c k => ⟨true, SKS_matches keys i c k⟩) (fun _ => avx2_spec_holds)
      sks_set_total stored_takeKeyOK hR hK hcap n hHov hmain hn hbits hocc
      (by have h : s.keys.val.length = s.coords.val.length := hK.1
          show s.coords.val.length ≤ s.keys.val.length
          omega) hk v
    refine ⟨s', ?_⟩
    unfold word_hashtrinity_insert hashtrinity.Table.insert_merge_last_write
    rw [h]
    simp only [bind_tc_ok, uncurry_apply_pair]

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
#audit Apeiron.MapExecuted.stored_insert_total
#audit Apeiron.MapExecuted.with_bits1_SKS_ex
#audit Apeiron.MapExecuted.stored_insert_witness
