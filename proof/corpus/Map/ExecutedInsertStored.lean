-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (HT-INSERT-W30): the
-- `Stored<u64>` size carrier and its key-store totality lemmas.
--
-- STATUS.  The lemmas below are kernel-checked.  `stored_insert_total` /
-- `stored_insert_witness` are **not** stated: the W29 `hTK` premise
--   `∀ keys i c, i.val < sz keys → ¬ IsSent c → ks.take_key keys i c = ok (c, keys')`
-- (and its arbitrary-key side-path twin `∃ k' keys', ks.take_key keys i c = ok (k', keys')`)
-- is unsatisfiable at `SKS`.
--
-- `SKS.take_key` is the extracted `Stored::take_key`: it reads slot `i` as a
-- `MaybeUninit U64` (= `Option U64`) and `assume_init_read`s it.  It therefore
--   (a) fails (`fail .undef`) on an uninitialised slot (`none`, which `Stored::alloc`
--       writes for every element), and
--   (b) returns the *stored* key, not the query coordinate `c`.
-- `stored_take_key_is_some` records (a)+(b) at the kernel: any successful `take_key`
-- forces `keys.val[i]? = some (some k')`.  `Stored::alloc` builds a `none` column
-- (`Map/ExecutedRep.lean:173` `alloc_uninit_val`), so the premise fails on the slots
-- the resize loop never visits and, for (b), on the slots it does.
--
-- The resize loop visits exactly the live slots (`resize_loop0`: `¬ IsSent c`), where the
-- key-column invariant `SKC` (`Map/ExecutedInsert.lean:262`) gives `keys[i] = some c`.  The
-- fix is to carry `SKC` (a `Slice U64 → KS → Prop`) through `resize_loop0`'s invariant and to
-- replace `hTK` by an `SKC`-parametrised premise; `stored_take_key_live` below is exactly the
-- instance that premise needs.  That touches `Map/ExecutedResizeTotal.lean`,
-- `Map/ExecutedGrowTotal.lean` and `Map/ExecutedInsertOk.lean`, outside this task's SCOPE,
-- and the side path (`side_push_loop`) needs the analogous side-key-column fact.  See the
-- W30 result for the exact refactor and the resolved goal.
--
-- `sks_take_key_total_false` is the kernel-checked refutation of the premise in its
-- arbitrary-key (side-path) form: `Stored::alloc 1` gives a column `[none]`, on which
-- `take_key` does not return.  The same `keys` refutes the `c`-returning (resize) form.
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
invariant `SKC` at a live slot — `SKS.take_key` returns `(c, keys)` and preserves the size carrier.
This is the instance the `SKC`-parametrised resize premise needs. -/
theorem stored_take_key_live (keys : hashtrinity.algebra.Stored U64) (i : Usize) (c : U64)
    (hc : keys.val[i.val]? = some (some c)) :
    ∃ keys', SKS.take_key keys i c = ok (c, keys') ∧ storedSz keys' = storedSz keys := by
  have hlt : i.val < keys.val.length := (List.getElem?_eq_some_iff.mp hc).1
  change ∃ keys', hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
    CI core.cmp.EqU64 keys i c = ok (c, keys') ∧ storedSz keys' = storedSz keys
  unfold hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
  obtain ⟨x, back, hx⟩ := gum_ex keys i hlt
  have hx1 : keys.val[i.val]? = some x := (get_unchecked_mut_ok keys i x back hx).1
  have hxb : back = keys.set i := (get_unchecked_mut_ok keys i x back hx).2
  have hxc : x = some c := by rw [hc] at hx1; exact (Option.some.inj hx1).symm
  subst hxc
  refine ⟨back (some c), ?_, ?_⟩
  · rw [hx]
    rw [bind_pair_ok]
    simp only [core.mem.maybe_uninit.MaybeUninit.assume_init_read, bind_tc_ok]
  · show (back (some c)).val.length = keys.val.length
    rw [hxb]
    exact slice_set_length keys i (some c)

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

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.sks_alloc_total
#audit Apeiron.MapExecuted.sks_set_total
#audit Apeiron.MapExecuted.stored_take_key_is_some
#audit Apeiron.MapExecuted.stored_take_key_live
#audit Apeiron.MapExecuted.sks_take_key_total_false
