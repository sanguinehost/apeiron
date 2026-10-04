import Growable.Refinement
import Growable.Search

/-!
# Publication refinement for the ported `Table::insertWith` spill commit

The flat spill port replaced `Region.tags` and the occupancy summary by one contiguous `u64`
coordinate word per slot with the single vacant sentinel `table.COORD_EMPTY`, and moved the fresh
spill publication into `table.Region.publish_live`.  `proof/ApeironWord.lean` extracts that body
verbatim from `src/table/state.rs`:

| generated body | `proof/ApeironWord.lean` | `src/table/state.rs` |
|---|---|---|
| `table.Region.publish_live` | 20687-20719 | 194:4-205:5 |
| `table.Table.apply_insert` (vacant spill branch) | 22992-23067, branch 23050-23061 | 1109:4-1155:5 |
| `word_stored_apply_insert` | 24197-24223 | lib.rs 494:0-506:1 |
| `word_regen_apply_insert` | 24235-24261 | lib.rs 514:0-526:1 |

This is obligation 2 of `/home/socol/oc-tmp/ht-port/receipt.md` §4: the merged value lands at the
located slot, and no search-visible state carries a live coordinate over an unpublished payload.
The publication point in the port is the coordinate store: `publish_live` writes the payload
(`sigma`, `held`, `value`), then the occupancy `tag`, and only then `coords[index] = coord_of place`.
`coord_of` never returns the vacant sentinel, so the published coordinate is a live, search-visible
word, and it is written last at the one index the located decision named.

`publish_live_publish_ordered` states the executed-body fact: the located slot's payload is the
published triple, its coordinate word is `Region.coord_of address.place` and is not
`COORD_EMPTY`, and every other coordinate and slot is unchanged.  `publish_live_readable` is the
corollary that discharges the `Apeiron.GrowableSearch.PublishReadable` premise the vacant-spill
apply theorems currently assume.  `word_*_apply_insert_vacant_spill_coordinate` lifts the fact to
the two generated table roots: the returned shard region is the published one, at the located
`index` it carries the merged value and the live coordinate, and no other shard is touched.

`is_intact` is not an input: every statement is a consequence of the executed body alone.  The
publication fact is all-`σ`: it reads the word-level `coords`/`slots` arrays and the `coord_of`/
`tag` maps, not the ring identity, so it is stated for an arbitrary `cell.Sigma` and carries no
face boundary.  What is not covered here is named in the receipt: the read-side composition (that a
search actually reaches `index` for the placed key) is obligation 1, and the `insert_with_reserved`
wrapper's locator and grow branches are outside this file.
-/

namespace Apeiron.GrowableInsertRefine

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-! ## Generic readers of generated `do` code -/

/-- Generic bind inversion on the generated `do` form: a successful bind ran its head to `ok`. -/
private theorem bind_ret_inv {α β : Type} (x : Result α) (f : α → Result β) (r : β)
    (hrun : (do let y ← x; f y) = ok r) : ∃ y, x = ok y ∧ f y = ok r := by
  cases hx : x with
  | vis eff k =>
      rw [hx, bind_tc_vis] at hrun
      exact (vis_not_ok hrun).elim
  | div =>
      rw [hx, bind_tc_div] at hrun
      exact (div_not_ok hrun).elim
  | ret y =>
      rw [hx] at hrun
      rw [bind_tc_ok] at hrun
      exact ⟨y, rfl, hrun⟩

/-- One generated `Vec.index_mut_usize` step at an in-bounds index: the read element is the element
under the index, the closure is the generated `Vec.set`, so the closure writes the new element at
that index and leaves every other lane alone. -/
private theorem getElem?_set_eq_of_ne {α : Type} (xs : List α) (written read : Nat)
    (value : α) (hne : read ≠ written) : (xs.set written value)[read]? = xs[read]? := by
  simp [hne]

private theorem vec_index_mut_step {α : Type} (v : alloc.vec.Vec α) (i : Usize) (old : α)
    (back : α → alloc.vec.Vec α) (y : α) (hbound : i.val < v.val.length)
    (hstep : alloc.vec.Vec.index_mut_usize v i = ok (old, back))
    (hold : old = v.val[i.val]) (hback : back = v.set i) :
    (back y).val[i.val]? = some y ∧
      ∀ j : Nat, j ≠ i.val → (back y).val[j]? = v.val[j]? := by
  subst old back
  rw [alloc.vec.Vec.set_val_eq]
  exact ⟨List.getElem?_set_self hbound, fun _ hne => getElem?_set_eq_of_ne _ _ _ _ hne⟩

/-! ## The vacant sentinel is never a published coordinate -/

/-- `COORD_EMPTY - 1` is not `COORD_EMPTY`: the sentinel's value is `2 ^ 64 - 1`. -/
private theorem coord_empty_sub_one_ok_ne (c : U64)
    (h : table.COORD_EMPTY - 1#u64 = ok c) : c ≠ table.COORD_EMPTY := by
  have hmax : table.COORD_EMPTY.val = 2 ^ 64 - 1 := by
    unfold table.COORD_EMPTY
    decide
  have h1 : (1#u64 : U64).val = 1 := by decide
  have hspec : (table.COORD_EMPTY - 1#u64) ⦃ z => z.val = table.COORD_EMPTY.val - 1 ⦄ :=
    WP.spec_mono (UScalar.sub_spec (ty := .U64) (x := table.COORD_EMPTY) (y := 1#u64)
      (by rw [h1, hmax]; omega)) (fun z hz => hz.1)
  rw [h, WP.spec_ok] at hspec
  intro hc
  rw [hc] at hspec
  omega

/-- `Region.coord_of` never returns the vacant sentinel: the nudge maps the one colliding value to
`COORD_EMPTY - 1` and the untouched value is unequal by the branch condition. -/
private theorem coord_of_result_ne_empty {R V : Type} (place : U64) (c : U64)
    (h : table.Region.coord_of R V place = ok c) : c ≠ table.COORD_EMPTY := by
  unfold table.Region.coord_of at h
  obtain ⟨raw, hraw, hrest⟩ := bind_ret_inv (place >>> table.SHARD_BITS)
    (fun raw => if raw = table.COORD_EMPTY then table.COORD_EMPTY - 1#u64 else ok raw) c h
  by_cases hraweq : raw = table.COORD_EMPTY
  · rw [if_pos hraweq] at hrest
    exact coord_empty_sub_one_ok_ne c hrest
  · rw [if_neg hraweq] at hrest
    have hcr : c = raw := Result.ok_injective hrest.symm
    rw [hcr]
    exact hraweq

/-! ## The publication ordering -/

/-- The executed publication fact for one `table.Region.publish_live`.  After the call at `index`
with payload `(sigma, held, value)` and place `place`:
* `coordOf` — the located slot's coordinate word is `Region.coord_of place` and is live, i.e. not
  the vacant sentinel `COORD_EMPTY`;
* `payloadPublished` — the located slot carries the published triple, so a search that reaches
  `index` cannot read a live coordinate over an unpublished payload;
* `coordsFrame` / `slotsFrame` — no other index is made search-visible: every other coordinate word
  and every other slot is unchanged.

The coordinate store is the publication point, and it is the only lane the call touches. -/
structure PublishOrdered {R V : Type} (before : table.Region R V) (index : Usize)
    (payload : cell.Sigma × R × V) (place : U64) (after : table.Region R V) : Prop where
  coordOf : ∃ c, table.Region.coord_of R V place = ok c ∧
    after.coords.val[index.val]? = some c ∧ c ≠ table.COORD_EMPTY
  payloadPublished : ∃ slot, after.slots.val[index.val]? = some slot ∧
    slot.sigma = payload.1 ∧ slot.held = payload.2.1 ∧ slot.value = payload.2.2
  coordsFrame : ∀ j : Nat, j ≠ index.val → after.coords.val[j]? = before.coords.val[j]?
  slotsFrame : ∀ j : Nat, j ≠ index.val → after.slots.val[j]? = before.slots.val[j]?

/-- **The ported `Region.publish_live` publishes its coordinate last.**  On the generated body
(`proof/ApeironWord.lean:20687-20719`) the payload store, the occupancy `tag` and the coordinate
store are one executed call: the payload triple lands at `index`, the coordinate word
`Region.coord_of place` lands at `index` and is live, and every other lane is untouched.  No
insert-path invariant is assumed. -/
theorem publish_live_publish_ordered {R V : Type} (before after : table.Region R V) (index : Usize)
    (sigma : cell.Sigma) (held : R) (value : V) (place : U64)
    (hpub : table.Region.publish_live before index (sigma, held, value) place = ok after) :
    PublishOrdered before index (sigma, held, value) place after := by
  unfold table.Region.publish_live at hpub
  rw [alloc.vec.Vec.index_mut_slice_index] at hpub
  cases h1 : alloc.vec.Vec.index_mut_usize before.slots index with
  | vis eff k => simp [h1] at hpub
  | div => simp [h1] at hpub
  | ret r1 =>
      rcases r1 with ⟨ss, back1⟩
      simp only [h1, bind_tc_ok] at hpub
      simp at hpub
      have hbound0 : index.val < before.slots.val.length := by
        cases hget : before.slots.val[index.val]? with
        | none => simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize, hget] at h1
        | some old => exact (List.getElem?_eq_some_iff.mp hget).1
      have h1spec := alloc.vec.Vec.index_mut_usize_spec before.slots index hbound0
      rw [h1, WP.spec_ok] at h1spec
      cases h2 : alloc.vec.Vec.index_mut_usize (back1 { ss with sigma := sigma }) index with
      | vis eff k => simp [h2] at hpub
      | div => simp [h2] at hpub
      | ret r2 =>
          rcases r2 with ⟨ss1, back2⟩
          simp only [h2, bind_tc_ok] at hpub
          simp at hpub
          have hbound1 : index.val < (back1 { ss with sigma := sigma }).val.length := by
            rw [h1spec.2, alloc.vec.Vec.set_val_eq]
            simpa using hbound0
          have h2spec := alloc.vec.Vec.index_mut_usize_spec
            (back1 { ss with sigma := sigma }) index hbound1
          rw [h2, WP.spec_ok] at h2spec
          cases h3 : alloc.vec.Vec.index_mut_usize (back2 { ss1 with held := held }) index with
          | vis eff k => simp [h3] at hpub
          | div => simp [h3] at hpub
          | ret r3 =>
              rcases r3 with ⟨ss2, back3⟩
              simp only [h3, bind_tc_ok] at hpub
              simp at hpub
              have hbound2 : index.val < (back2 { ss1 with held := held }).val.length := by
                rw [h2spec.2, alloc.vec.Vec.set_val_eq]
                simpa using hbound1
              have h3spec := alloc.vec.Vec.index_mut_usize_spec
                (back2 { ss1 with held := held }) index hbound2
              rw [h3, WP.spec_ok] at h3spec
              generalize htag : table.tag place = tagResult at hpub
              cases tagResult with
              | vis eff k => simp at hpub
              | div => simp at hpub
              | ret tag =>
                  cases h4 : alloc.vec.Vec.index_mut_usize (back3 { ss2 with value := value }) index with
                  | vis eff k => simp [h4] at hpub
                  | div => simp [h4] at hpub
                  | ret r4 =>
                      rcases r4 with ⟨ss3, back4⟩
                      simp only [h4, bind_tc_ok] at hpub
                      simp at hpub
                      have hbound3 :
                          index.val < (back3 { ss2 with value := value }).val.length := by
                        rw [h3spec.2, alloc.vec.Vec.set_val_eq]
                        simpa using hbound2
                      have h4spec := alloc.vec.Vec.index_mut_usize_spec
                        (back3 { ss2 with value := value }) index hbound3
                      rw [h4, WP.spec_ok] at h4spec
                      cases hcoord : table.Region.coord_of R V place with
                      | vis eff k => simp [hcoord] at hpub
                      | div => simp [hcoord] at hpub
                      | ret c =>
                          simp only [hcoord, bind_tc_ok] at hpub
                          cases h5 : alloc.vec.Vec.index_mut_usize before.coords index with
                          | vis eff k => simp [h5] at hpub
                          | div => simp [h5] at hpub
                          | ret r5 =>
                              rcases r5 with ⟨ss4, back5⟩
                              simp only [h5, bind_tc_ok] at hpub
                              simp at hpub
                              have hbound4 : index.val < before.coords.val.length := by
                                cases hget : before.coords.val[index.val]? with
                                | none =>
                                    simp [alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.index_usize,
                                      hget] at h5
                                | some old => exact (List.getElem?_eq_some_iff.mp hget).1
                              have h5spec := alloc.vec.Vec.index_mut_usize_spec before.coords index
                                hbound4
                              rw [h5, WP.spec_ok] at h5spec
                              generalize hadd : before.live + 1#usize = addResult at hpub
                              cases addResult with
                              | vis eff k => simp at hpub
                              | div => simp at hpub
                              | ret live1 =>
                                  simp at hpub
                                  subst after
                                  have hread0 := vec_index_mut_step before.slots index ss back1
                                    { ss with sigma := sigma } hbound0 h1 h1spec.1 h1spec.2
                                  have hread1 := vec_index_mut_step (back1 { ss with sigma := sigma })
                                    index ss1 back2 { ss1 with held := held } hbound1 h2 h2spec.1 h2spec.2
                                  have hread2 := vec_index_mut_step (back2 { ss1 with held := held })
                                    index ss2 back3 { ss2 with value := value } hbound2 h3 h3spec.1
                                    h3spec.2
                                  have hread3 := vec_index_mut_step (back3 { ss2 with value := value })
                                    index ss3 back4 { ss3 with tag := tag } hbound3 h4 h4spec.1 h4spec.2
                                  have hread4 := vec_index_mut_step before.coords index ss4 back5 c
                                    hbound4 h5 h5spec.1 h5spec.2
                                  have hval1 : (back1 { ss with sigma := sigma }).val[index.val] =
                                      { ss with sigma := sigma } :=
                                    (List.getElem?_eq_some_iff.mp hread0.1).2
                                  have hval2 : (back2 { ss1 with held := held }).val[index.val] =
                                      { ss1 with held := held } :=
                                    (List.getElem?_eq_some_iff.mp hread1.1).2
                                  have hval3 : (back3 { ss2 with value := value }).val[index.val] =
                                      { ss2 with value := value } :=
                                    (List.getElem?_eq_some_iff.mp hread2.1).2
                                  have hss1 : ss1 = { ss with sigma := sigma } := h2spec.1.trans hval1
                                  have hss2 : ss2 = { ss1 with held := held } := h3spec.1.trans hval2
                                  have hss3 : ss3 = { ss2 with value := value } := h4spec.1.trans hval3
                                  have hframeSlots : ∀ j : Nat, j ≠ index.val →
                                      (back4 { ss3 with tag := tag }).val[j]? =
                                        before.slots.val[j]? := by
                                    intro j hj
                                    rw [hread3.2 j hj, hread2.2 j hj, hread1.2 j hj, hread0.2 j hj]
                                  have hne_c : c ≠ table.COORD_EMPTY :=
                                    coord_of_result_ne_empty (R := R) (V := V) place c hcoord
                                  subst ss1 ss2 ss3
                                  exact
                                    ⟨⟨c, hcoord, hread4.1, hne_c⟩,
                                      ⟨{ tag := tag, sigma := sigma, held := held, value := value },
                                        hread3.1, rfl, rfl, rfl⟩,
                                      hread4.2, hframeSlots⟩

/-- The publication readback as the predicate the vacant-spill apply theorems assume.  This
discharges `Apeiron.GrowableSearch.PublishReadable`. -/
theorem publish_live_readable {R : Type} (before after : table.Region R I64) (index : Usize)
    (sigma : cell.Sigma) (held : R) (value : I64) (place : U64)
    (hpub : table.Region.publish_live before index (sigma, held, value) place = ok after) :
    Apeiron.GrowableSearch.PublishReadable before index (sigma, held, value) place := by
  obtain ⟨hcoord, hpayload, -, -⟩ :=
    publish_live_publish_ordered (V := I64) before after index sigma held value place hpub
  obtain ⟨slot, hslot, hsigma, hheld, hvalue⟩ := hpayload
  exact ⟨after, slot, hpub, hslot, hsigma, hheld, hvalue⟩

/-! ## Lifting the publication to the generated vacant-spill commit

The `table.Table.apply_insert` vacant spill branch (`proof/ApeironWord.lean:23050-23061`) computes
the merged value, holds the key into the carried carrier, calls `table.Region.publish_live` at the
located `index`, and writes the returned region back into the shard.  `word_stored_apply_insert`
(`proof/ApeironWord.lean:24197`) wraps that commit.  The theorem below inverts the executed body's
success equation and exposes the published region: the located slot carries the merged value and a
live coordinate word. -/

/-- The generated stored vacant-spill commit exposes the region `publish_live` returned at the
located `index`, with the merged value in its slot and its live coordinate word. -/
theorem word_stored_apply_insert_vacant_spill_coordinate
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (shard index : Usize)
    (region : table.Region (table.keyrep.Stored I64) I64)
    (oldSlot : table.SpillSlot (table.keyrep.Stored I64) I64)
    (next : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (hregion : state.shards.val[shard.val]? = some region)
    (_hslot : region.slots.val[index.val]? = some oldSlot)
    (_hlen : state.len.val + 1 < Usize.max)
    (h : word_stored_apply_insert state { «class» := .Vacant, location := .Spill shard index }
        sigma key value address = ok (next, none)) :
    ∃ publishedRegion slot c,
      next.shards.val[shard.val]? = some publishedRegion ∧
      publishedRegion.slots.val[index.val]? = some slot ∧
      slot.sigma = sigma ∧
      slot.value = core.num.I64.wrapping_add value state.default ∧
      publishedRegion.coords.val[index.val]? = some c ∧
      table.Region.coord_of (table.keyrep.Stored I64) I64 address.place = ok c ∧
        c ≠ table.COORD_EMPTY := by
  have hshard : shard.val < state.shards.val.length :=
    (List.getElem?_eq_some_iff.mp hregion).1
  unfold word_stored_apply_insert at h
  obtain ⟨⟨ao, state1⟩, hcommit, hrest⟩ := bind_ret_inv _ _ _ h
  cases ao with
  | NeedsGrowth k v =>
      simp at hrest
  | Committed =>
      simp at hrest
      subst state1
      simp only [table.Table.apply_insert, alloc.vec.Vec.index_mut_slice_index] at hcommit
      obtain ⟨t, hclone, hc1⟩ := bind_ret_inv _ _ _ hcommit
      obtain ⟨merged, hmerge, hc2⟩ := bind_ret_inv _ _ _ hc1
      obtain ⟨⟨r, back⟩, him, hc3⟩ := bind_ret_inv _ _ _ hc2
      obtain ⟨t1, hhold, hc4⟩ := bind_ret_inv _ _ _ hc3
      obtain ⟨publishedRegion, hpub, hc5⟩ := bind_ret_inv _ _ _ hc4
      obtain ⟨i, hadd, hc6⟩ := bind_ret_inv _ _ _ hc5
      have hnextEq : { state with shards := back publishedRegion, len := i } = next :=
        (Prod.mk.inj (Result.ok_injective hc6)).2
      have himspec := alloc.vec.Vec.index_mut_usize_spec state.shards shard hshard
      rw [him, WP.spec_ok] at himspec
      simp only [table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.hold,
        table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge] at hhold hmerge
      obtain ⟨horderedCoord, hpayload, -, -⟩ :=
        publish_live_publish_ordered r publishedRegion index sigma t1 merged address.place hpub
      obtain ⟨c, hcoord, hcoordLive, hne_c⟩ := horderedCoord
      obtain ⟨slot, hslotPub, hsigma, hheld, hvalue⟩ := hpayload
      have hnextShard : next.shards.val[shard.val]? = some publishedRegion := by
        rw [← hnextEq, himspec.2, alloc.vec.Vec.set_val_eq]
        exact List.getElem?_set_self hshard
      have hvalueMerged : slot.value = core.num.I64.wrapping_add value state.default := by
        simp_all
      exact ⟨publishedRegion, slot, c, hnextShard, hslotPub, hsigma, hvalueMerged, hcoordLive,
        hcoord, hne_c⟩

/-- The generated regenerating vacant-spill commit exposes the region `publish_live` returned at the
located `index`, with the merged value in its slot and its live coordinate word. -/
theorem word_regen_apply_insert_vacant_spill_coordinate
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (sigma : cell.Sigma) (key value : I64) (address : table.TableAddress) (shard index : Usize)
    (region : table.Region (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (oldSlot : table.SpillSlot (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (next : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (hregion : state.shards.val[shard.val]? = some region)
    (_hslot : region.slots.val[index.val]? = some oldSlot)
    (_hlen : state.len.val + 1 < Usize.max)
    (h : word_regen_apply_insert state { «class» := .Vacant, location := .Spill shard index }
        sigma key value address = ok (next, none)) :
    ∃ publishedRegion slot c,
      next.shards.val[shard.val]? = some publishedRegion ∧
      publishedRegion.slots.val[index.val]? = some slot ∧
      slot.sigma = sigma ∧
      slot.value = core.num.I64.wrapping_add value state.default ∧
      publishedRegion.coords.val[index.val]? = some c ∧
      table.Region.coord_of (table.keyrep.Regen table.keyrep.I64WordCodec) I64 address.place =
        ok c ∧ c ≠ table.COORD_EMPTY := by
  have hshard : shard.val < state.shards.val.length :=
    (List.getElem?_eq_some_iff.mp hregion).1
  unfold word_regen_apply_insert at h
  obtain ⟨⟨ao, state1⟩, hcommit, hrest⟩ := bind_ret_inv _ _ _ h
  cases ao with
  | NeedsGrowth k v =>
      simp at hrest
  | Committed =>
      simp at hrest
      subst state1
      simp only [table.Table.apply_insert, alloc.vec.Vec.index_mut_slice_index] at hcommit
      obtain ⟨t, hclone, hc1⟩ := bind_ret_inv _ _ _ hcommit
      obtain ⟨merged, hmerge, hc2⟩ := bind_ret_inv _ _ _ hc1
      obtain ⟨⟨r, back⟩, him, hc3⟩ := bind_ret_inv _ _ _ hc2
      obtain ⟨t1, hhold, hc4⟩ := bind_ret_inv _ _ _ hc3
      obtain ⟨publishedRegion, hpub, hc5⟩ := bind_ret_inv _ _ _ hc4
      obtain ⟨i, hadd, hc6⟩ := bind_ret_inv _ _ _ hc5
      have hnextEq : { state with shards := back publishedRegion, len := i } = next :=
        (Prod.mk.inj (Result.ok_injective hc6)).2
      have himspec := alloc.vec.Vec.index_mut_usize_spec state.shards shard hshard
      rw [him, WP.spec_ok] at himspec
      simp only [table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.hold,
        table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge] at hhold hmerge
      obtain ⟨horderedCoord, hpayload, -, -⟩ :=
        publish_live_publish_ordered r publishedRegion index sigma t1 merged address.place hpub
      obtain ⟨c, hcoord, hcoordLive, hne_c⟩ := horderedCoord
      obtain ⟨slot, hslotPub, hsigma, hheld, hvalue⟩ := hpayload
      have hnextShard : next.shards.val[shard.val]? = some publishedRegion := by
        rw [← hnextEq, himspec.2, alloc.vec.Vec.set_val_eq]
        exact List.getElem?_set_self hshard
      have hvalueMerged : slot.value = core.num.I64.wrapping_add value state.default := by
        simp_all
      exact ⟨publishedRegion, slot, c, hnextShard, hslotPub, hsigma, hvalueMerged, hcoordLive,
        hcoord, hne_c⟩

end Apeiron.GrowableInsertRefine

/-! ## Cone placement

The repository's `#audit` root (`AxiomAudit/Core.lean`) imports the stale `Growable.Scan`/
`Growable.Representation` pair and does not build against the ported extraction, so the closed-cone
check is stated here directly: each headline's transitive axiom set is the permitted
`{propext, Classical.choice, Quot.sound}`.  Register the four names in `AxiomAudit/Core.lean` when
that root is rebuilt. -/
#print axioms Apeiron.GrowableInsertRefine.publish_live_publish_ordered
#print axioms Apeiron.GrowableInsertRefine.publish_live_readable
#print axioms Apeiron.GrowableInsertRefine.word_stored_apply_insert_vacant_spill_coordinate
#print axioms Apeiron.GrowableInsertRefine.word_regen_apply_insert_vacant_spill_coordinate
