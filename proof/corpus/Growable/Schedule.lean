import ApeironWord

/-!
# The `Parts::of` constructor: setup for the per-shard run equation

`src/table/partition.rs` builds the growable constructor's index schedule in three extracted
passes (`partition_chunk`, `proof/ApeironWord.lean:19305-19328`): count the routed loads per shard,
prefix-sum them into `ends` with a per-shard `cursor`, then walk the input again, writing each index
into `buf[cursor[shard]]` and bumping that cursor.  `Parts::of` (`proof/ApeironWord.lean:19330`) is
that one window with `first = 0` and `last = input.len()`, and is the index schedule
`Table.build_at` validates (`proof/ApeironWord.lean:23533`).

The intended constructor theorem is: **for every shard, the run `buf[ends[s]..ends[s+1]]` is exactly
the increasing sequence of input indexes the producer's own routing expression selects for that
shard.**  Nothing about the output is assumed and `is_intact` is not an input.  The four obligations
`Parts::is_intact` checks at runtime — tiling endpoints, in-range local order, complete coverage,
fresh routing — are intended to be *derived* from the constructor.

This file contains the shared setup — the routing expression `producerShard`, its characteristic
function `routedTo`, the increasing list `routedRun` the theorem is stated with, and `usizeOf` with
its `Usize.max` bridge — the count-pass prerequisites (`slice_index_ok`, `producerShard_eq_ok`,
`producerShard_det`, `producerShard_routedTo_iff`, and the generated shard arithmetic
`SHARDS_spec`/`SHARD_MASK_spec`/`shard_lt_256`/`shard_spec`), the conditional loop rule
`loop_ok_spec`, **the count pass** `loop0_spec` (a successful `partition_chunk_loop0` from the zero
array stores, per shard, the number of input indexes of `[first, last)` the producer's routing
selects for it), and **the prefix pass** `loop1_spec` (a successful `partition_chunk_loop1` stores
the prefix sums of `counts` in both `ends` and `cursor`, under `Σ counts ≤ Usize.max`).

The scatter pass is closed: `routedMid` (the routed indexes of `[first, i)`),
`routedMid_self`/`routedMid_step`, `endsVal` (the base offset `ends[s]`), `scatterInv` (the
per-shard buffer-region invariant), the body inversions `loop2_body_cont`/`loop2_body_done`,
`take_set_succ` (the buffer-append step), the four load/monotonicity/prefix/disjointness facts
(`endsVal_step`, `endsVal_mono`, `routedMid_isPrefix`, `scatter_write_disjoint`) and the invariant
step `scatterInv_step` prove **`loop2_spec`**: a successful `partition_chunk_loop2` from the entry
cursor `cursor[s] = ends[s]` writes, for every shard, the routed indexes of `[first, last)` into
`buf[ends[s] .. ends[s] + |routedMid|)`.  Composing the count pass, `loop0_spec`, the prefix bound
`prefixSum_le_max_of_loop1` (read off the successful `partition_chunk_loop1` run, no premise),
`loop1_spec` and `loop2_spec` through `partition_chunk`/`Parts.of` gives **`parts_of_run_eq`**: for
a successful `Parts::of`, shard `s`'s buffer region is exactly the routed run
`routedRun input.len s`.  The coverage count `prefixSum counts 256 = input.len` (every input index
increments exactly one count) is `loop0_spec_sum`, from the array-sum-under-`set` lemma
`prefixSum_set`; it closes the endpoint tiling.  Together these give **`parts_of_shard_run`**, the
four `Parts::is_intact` obligations: tiling endpoints, in-range local order, complete coverage, and
fresh routing.

`loop.spec_decr_nat` could not state the count pass because `partition_chunk_loop0.body` calls
`HomeInst.address home s key`, an opaque partial structure field, so the body can fail and the
`body x ⦃ · ⦄` hypothesis (`False` on `fail`) is undischargeable.  `loop_ok_spec` supplies the
missing conditional rule (`loop body x = ok y → post y`), proved from `loop.eq_def` by well-founded
recursion and the `bind_eq_ok` inversion; the count pass is stated as an implication on the
successful result.  The run equation and the four `is_intact` obligations are proved here; the
`is_intact` function's own evaluation to `true`, the general window and the parallel executor are
not.

## The routing is the producer's, not a parameter

`producerShard` is the generated routing expression of the third pass written out once: read
`input[i]`, take that element's `sigma face`, hand `(sigma, key)` to `table.Home.address`, and reduce
the generated address with `table.shard`.  `routedTo` is the characteristic function of "this index
routes to this shard" *by that expression*.  It is `Bool`-valued through `Classical.propDecidable`,
which is a mathematical device, not an assumption: every existential fact behind it is read out of
the generated `Result` chain, because a failing address makes `Parts::of` fail rather than produce
a value.

## Not present here

Not present, and not replaced by an assumption: the `is_intact` function itself is not shown to
evaluate to `true` (the four conjuncts of `parts_of_shard_run` are its obligations, proved of
`Parts::of`'s output directly).  Also open: the
general window `first ≠ 0` that the parallel `ScatterPlan` uses (`src/table/partition.rs:245-330`);
the parallel `ScatterScratch` executor of `src/table/parallel.rs`, `std::thread::scope`, allocator
and panic realisation; and the `GrowableSchedule.partition_preserves_key_streams` composition with
the canonical `Apeiron.Table.build_eq_of_key_streams_eq` (`docs/epics/parallel-stable-scatter.md`,
"Representation relation").  `Parts::is_intact` is neither removed nor weakened.
-/

namespace Apeiron.GrowableSchedule

open Aeneas Aeneas.Std Result
open apeiron_word
open Classical

set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

/-- `x ≤ Usize.max` is `x < 2^numBits`, the bound `UScalar.ofNatCore` needs. -/
theorem usize_lt_pow_numBits (x : Nat) (h : x ≤ Usize.max) :
    x < 2 ^ UScalarTy.Usize.numBits := by
  have hmax : (Usize.max : Nat) = 2 ^ System.Platform.numBits - 1 := by
    simp [Usize.max, Usize.numBits]
  rw [UScalarTy.Usize_numBits_eq]
  rw [hmax] at h
  have hp : (0 : Nat) < 2 ^ System.Platform.numBits := pow_pos (by norm_num) _
  omega

/-! ## The producer's routing expression -/

/-- The `Usize` carrying `i`, saturating above the machine word.  Every index the constructor reads
    is `< input.len() ≤ Usize.max`, so `usizeOf i = i` there (`usizeOf_eq`). -/
noncomputable def usizeOf (i : Nat) : Usize :=
  if hi : i ≤ Usize.max then UScalar.ofNatCore i (usize_lt_pow_numBits i hi) else 0#usize

theorem usizeOf_val (i : Nat) (hi : i ≤ Usize.max) : (usizeOf i).val = i := by
  simp only [usizeOf, dif_pos hi, UScalar.ofNatCore_val_eq]

/-- The machine word holds at least `256`, one past the shard range. -/
theorem le_usize_max_256 : (256 : Nat) ≤ Usize.max := by
  have hmax : (Usize.max : Nat) = 2 ^ System.Platform.numBits - 1 := by
    simp [Usize.max, Usize.numBits]
  rw [hmax]
  rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> norm_num

theorem usizeOf_val_of_le_256 {n : Nat} (hn : n ≤ 256) : (usizeOf n).val = n :=
  usizeOf_val n (by have := le_usize_max_256; omega)

theorem usize_val_le_max (x : Usize) : x.val ≤ Usize.max := by
  have h := UScalar.hBounds x
  have hmax : (Usize.max : Nat) = 2 ^ UScalarTy.Usize.numBits - 1 := by
    simp [Usize.max, Usize.numBits]
  omega

/-- A successful `Usize` addition is in range. -/
theorem add_le_of_ok {x y z : Usize} (h : x + y = ok z) : x.val + y.val ≤ Usize.max := by
  have hh := UScalar.add_equiv (ty := UScalarTy.Usize) x y
  rw [h, Result.match.ok] at hh
  have hmax : (Usize.max : Nat) = 2 ^ UScalarTy.Usize.numBits - 1 := by
    simp [Usize.max, Usize.numBits]
  omega

/-- A successful `Usize` addition carries the exact sum. -/
theorem add_val_of_ok {x y z : Usize} (h : x + y = ok z) : z.val = x.val + y.val := by
  have hh := UScalar.add_equiv (ty := UScalarTy.Usize) x y
  rw [h, Result.match.ok] at hh
  exact hh.2.1

/-- A successful `Usize` subtraction carries the exact difference. -/
theorem sub_val_of_ok {x y z : Usize} (h : x - y = ok z) : z.val = x.val - y.val := by
  have hh := UScalar.sub_equiv (ty := UScalarTy.Usize) x y
  rw [h, Result.match.ok] at hh
  obtain ⟨hle, hsum, _⟩ := hh
  omega

theorem usizeOf_eq (i : Usize) (hi : i.val ≤ Usize.max) : usizeOf i.val = i :=
  UScalar.eq_of_val_eq (usizeOf_val i.val hi)

/-- The generated shard decision for one input index: the four statements of the generated routing
    expression (`proof/ApeironWord.lean:19268-19271`, `19275-19276`), written out once. -/
def producerShard {K V H Faces : Type} (HomeInst : table.Home H K Faces) (home : H)
    (input : Slice ((Faces × K) × V)) (i : Usize) : Result Usize := do
  let ((face, key), _) ← Slice.index_usize input i
  let s ← HomeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceDomainInst.sigma face
  let address ← HomeInst.address home s key
  table.shard address

/-- Input index `j` is routed to shard `s` by the producer's own routing expression. -/
noncomputable def routedTo {K V H Faces : Type} (HomeInst : table.Home H K Faces) (home : H)
    (input : Slice ((Faces × K) × V)) (s : Usize) : Nat → Bool :=
  fun j => decide (∃ r : Usize, producerShard HomeInst home input (usizeOf j) = ok r ∧ r.val = s.val)

theorem routedTo_eq_true_iff {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {s : Usize} {j : Nat} :
    routedTo HomeInst home input s j = true ↔
      ∃ r : Usize, producerShard HomeInst home input (usizeOf j) = ok r ∧ r.val = s.val := by
  unfold routedTo
  exact decide_eq_true_iff

/-- The increasing list of input indexes of `[0, i)` that the producer routes to shard `s`. -/
noncomputable def routedRun {K V H Faces : Type} (HomeInst : table.Home H K Faces) (home : H)
    (input : Slice ((Faces × K) × V)) (i : Usize) (s : Usize) : List Nat :=
  (List.range i.val).filter (routedTo HomeInst home input s)

theorem routedRun_nil {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} (s : Usize) :
    routedRun HomeInst home input 0#usize s = [] := by
  simp [routedRun]

/-- One input index extends exactly the run of the shard its own routing expression selects. -/
theorem routedRun_succ {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {i s : Usize} {r : Usize}
    (hi : i.val + 1 ≤ Usize.max) (hi' : usizeOf i.val = i)
    (h : producerShard HomeInst home input i = ok r) (hr : r.val = s.val) :
    routedRun HomeInst home input (usizeOf (i.val + 1)) s
      = routedRun HomeInst home input i s ++ [i.val] := by
  have hpos : routedTo HomeInst home input s i.val = true :=
    (routedTo_eq_true_iff).2 ⟨r, by rw [hi']; exact h, hr⟩
  rw [routedRun, routedRun, usizeOf_val (i.val + 1) hi, List.range_succ,
    List.filter_append, List.filter_cons_of_pos hpos, List.filter_nil]

/-- `List.range` is monotone for the prefix order. -/
theorem range_isPrefix {m n : Nat} (h : m ≤ n) : List.range m <+: List.range n := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [List.range_add]
  exact ⟨_, rfl⟩

/-- `routedRun` is strictly increasing: it is a sublist of `List.range`. -/
theorem routedRun_pairwise {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {i s : Usize} :
    (routedRun HomeInst home input i s).Pairwise (· < ·) := by
  unfold routedRun
  exact List.Pairwise.filter _ List.pairwise_lt_range

theorem routedRun_getElem?_lt {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {i s : Usize} {j j' aj bj : Nat} (hj : j < j')
    (ha : (routedRun HomeInst home input i s)[j]? = some aj)
    (hb : (routedRun HomeInst home input i s)[j']? = some bj) :
    aj < bj := by
  obtain ⟨hjlen, haeq⟩ := List.getElem?_eq_some_iff.mp ha
  obtain ⟨hj'len, hbeq⟩ := List.getElem?_eq_some_iff.mp hb
  have hlt := (List.pairwise_iff_getElem.mp (routedRun_pairwise (i := i) (s := s)))
    j j' hjlen hj'len hj
  rwa [haeq, hbeq] at hlt

/-- The run over a prefix is no longer than the run over the whole window. -/
theorem routedRun_length_mono {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {i j : Usize} {s : Usize} (h : i.val ≤ j.val) :
    (routedRun HomeInst home input i s).length ≤ (routedRun HomeInst home input j s).length := by
  rw [routedRun, routedRun]
  exact List.IsPrefix.length_le (List.IsPrefix.filter _ (range_isPrefix h))

/-! ## Routing facts read out of the generated `Result` chain

These are the pieces of the counting pass the run equation consumes: the generated `Slice::index`
read, the reconstruction of the producer's four-statement routing expression from the extracted
`Result` postconditions, its determinism, and the `routedTo` characterization at the producer's own
shard. -/

/-- The generated `Slice::index_usize` returns the in-bounds element. -/
theorem slice_index_ok {α : Type} (v : Slice α) (i : Usize) (h : i.val < v.val.length) :
    Slice.index_usize v i = ok v.val[i.val] := by
  obtain ⟨x, hx, hval⟩ := WP.spec_imp_exists (Slice.index_usize_spec v i h)
  rw [hx, hval]

/-- The producer's routing expression equals `ok routed` when its four extracted statements return
    the tuple component `(face, key)`, that face's `sigma`, its address, and the shard. -/
theorem producerShard_eq_ok {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {i : Usize} {face : Faces} {key : K} {v : V}
    {s : cell.Sigma} {address : table.TableAddress} {routed : Usize}
    (hbound : i.val < input.val.length)
    (hface : ((face, key), v) = input.val[i.val])
    (hs : HomeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceDomainInst.sigma face = ok s)
    (ha : HomeInst.address home s key = ok address)
    (hr : table.shard address = ok routed) :
    producerShard HomeInst home input i = ok routed := by
  unfold producerShard
  rw [slice_index_ok input i hbound, ← hface]
  simp only [Bind.bind]
  rw [bind_ok]
  change Std.bind (HomeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceDomainInst.sigma face)
      (fun s => Std.bind (HomeInst.address home s key) (fun address => table.shard address))
    = ok routed
  rw [hs, bind_ok, ha, bind_ok, hr]

/-- The producer's routing expression is a function: one input index routes to at most one shard. -/
theorem producerShard_det {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {i r r' : Usize}
    (h : producerShard HomeInst home input i = ok r)
    (h' : producerShard HomeInst home input i = ok r') : r = r' := by
  rw [h] at h'
  exact Result.ok_injective h'

/-- An index whose routing expression returns shard `r` is selected for shard `r` by `routedTo`. -/
theorem producerShard_routedTo_iff {K V H Faces : Type} {HomeInst : table.Home H K Faces}
    {home : H} {input : Slice ((Faces × K) × V)} {i s r : Usize}
    (hi : usizeOf i.val = i) (h : producerShard HomeInst home input i = ok r) :
    routedTo HomeInst home input s i.val = true ↔ s.val = r.val := by
  rw [routedTo_eq_true_iff]
  constructor
  · rintro ⟨r', hr', hv⟩
    rw [hi, h] at hr'
    rw [Result.ok.injEq] at hr'
    rw [← hr'] at hv
    exact hv.symm
  · intro hv
    exact ⟨r, by rw [hi]; exact h, hv.symm⟩

/-! ### The generated shard arithmetic

The count pass reads and writes `counts[routed]` where `routed = table.shard address`.  The generated
shard is `(cast word) &&& SHARD_MASK` with `SHARD_MASK = SHARDS - 1`, so it is in `[0, 256)`.  These
facts are read out of the generated `SHARDS`/`SHARD_MASK` bodies, not assumed. -/

/-- The generated shard count constant is `256`. -/
theorem SHARDS_spec : table.partition.SHARDS ⦃ r => r = 256#usize ⦄ := by
  unfold table.partition.SHARDS table.partition.SHARD_BITS
  step
  all_goals (cases System.Platform.numBits_eq <;> simp_all)
  all_goals (apply UScalar.eq_of_val_eq; rw [r_post]; simp_all [Usize.size, Usize.numBits])

/-- The generated shard mask is `255`. -/
theorem SHARD_MASK_spec : table.partition.SHARD_MASK ⦃ r => r = 255#usize ⦄ := by
  unfold table.partition.SHARD_MASK
  rw [show table.partition.SHARDS = ok 256#usize from by
    obtain ⟨r, hr, hval⟩ := WP.spec_imp_exists SHARDS_spec
    rw [hr, hval]]
  step
  all_goals (cases System.Platform.numBits_eq <;> simp_all)
  all_goals (apply UScalar.eq_of_val_eq; rw [i_post]; simp_all)

/-- The generated shard is the low byte of the address word, hence `< 256`. -/
theorem shard_lt_256 (address : table.TableAddress) :
    table.shard address ⦃ r => r.val < 256 ⦄ := by
  unfold table.shard table.SHARD_MASK
  rw [show table.partition.SHARD_MASK = ok 255#usize from by
    obtain ⟨r, hr, hval⟩ := WP.spec_imp_exists SHARD_MASK_spec
    rw [hr, hval]]
  split <;> rename_i hw <;>
    · step
      rw [UScalar.val_and]
      simp only [UScalar.ofNatCore_val_eq]
      exact Nat.lt_of_le_of_lt (Nat.and_le_right) (by norm_num)

/-- The generated shard carries its exact value and its bound: the step spec the count loop needs. -/
theorem shard_spec (address : table.TableAddress) :
    table.shard address ⦃ r => r.val < 256 ∧ table.shard address = ok r ⦄ := by
  obtain ⟨r, hr, hb⟩ := WP.spec_imp_exists (shard_lt_256 address)
  exact (WP.spec_equiv_exists _ _).2 ⟨r, hr, hb, hr⟩

/-! ## Conditional loop rule and count pass -/

/-- `bind` inversion for the successful case: a successful `bind` has a successful left operand. -/
theorem bind_eq_ok {α β : Type} (m : Result α) (k : α → Result β) (y : β)
    (h : Std.bind m k = ok y) : ∃ a, m = ok a ∧ k a = ok y := by
  revert h
  refine Result.cases m ?_ ?_ ?_
  · intro a h
    rw [bind_ok] at h
    exact ⟨a, rfl, h⟩
  · intro i k' h
    rw [bind_vis] at h
    exact absurd h.symm (by simp)
  · intro h
    rw [bind_div] at h
    exact absurd h.symm (by simp)

/-- Conditional loop rule: if every `inv` state steps to `inv`/`post` **whenever the body
    succeeds**, then a successful loop run ends in `post`.  No totality of the body is assumed —
    this is the rule the count pass needs, because `Home::address` is partial. -/
theorem loop_ok_spec {α β : Type} (measure : α → Nat) (inv : α → Prop) (post : β → Prop)
    (body : α → Result (ControlFlow α β))
    (hstep : ∀ x, inv x →
      (∀ y, body x = ok (ControlFlow.done y) → post y) ∧
      (∀ x', body x = ok (ControlFlow.cont x') → inv x' ∧ measure x' < measure x)) :
    ∀ x, inv x → ∀ y, loop body x = ok y → post y := by
  intro x
  suffices H : ∀ n, ∀ x, measure x = n → inv x → ∀ y, loop body x = ok y → post y by
    exact fun hx => H (measure x) x rfl hx
  intro n
  refine Nat.strong_induction_on n ?_
  intro n ih x hxn hx y hloop
  rw [loop.eq_def] at hloop
  obtain ⟨r, hbody, hk⟩ :=
    bind_eq_ok (body x) (fun r => match r with
      | ControlFlow.cont x' => loop body x'
      | ControlFlow.done z => ok z) y hloop
  obtain ⟨hdone, hcont⟩ := hstep x hx
  cases r with
  | done z =>
    simp only at hk
    have hzy : z = y := Result.ok.injEq.mp hk
    rw [hzy] at hbody
    exact hdone y hbody
  | cont x' =>
    simp only at hk
    have hlt : measure x' < n := by rw [← hxn]; exact (hcont x' hbody).2
    exact ih (measure x') hlt x' rfl (hcont x' hbody).1 y hk

/-- One index step of `routedRun`: the route at `i` either appends `i` to its shard's run or leaves
    it unchanged. -/
theorem routedRun_step {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {i s : Usize} (hi : i.val + 1 ≤ Usize.max) :
    routedRun HomeInst home input (usizeOf (i.val + 1)) s =
      (if routedTo HomeInst home input s i.val then
        routedRun HomeInst home input i s ++ [i.val]
      else routedRun HomeInst home input i s) := by
  rw [routedRun, routedRun, usizeOf_val (i.val + 1) hi, List.range_succ, List.filter_append,
    List.filter_cons]
  cases routedTo HomeInst home input s i.val <;> simp [List.filter_nil]

/-- The count loop's invariant: the running counts are the routed prefix lengths. -/
def countInv {K V H Faces : Type} (HomeInst : table.Home H K Faces) (home : H)
    (input : Slice ((Faces × K) × V)) (last first : Usize)
    (counts : Array Usize 256#usize) (i : Usize) : Prop :=
  first.val ≤ i.val ∧ i.val ≤ last.val ∧
  ∀ s : Usize, s.val < 256 →
    counts.val[s.val]? = some (usizeOf ((routedRun HomeInst home input i s).length
      - (routedRun HomeInst home input first s).length))

/-- The invariant holds at the count loop entry (`counts = 0`, `i = first`). -/
theorem countInv_init {K V H Faces : Type} (HomeInst : table.Home H K Faces) (home : H)
    (input : Slice ((Faces × K) × V)) (last first : Usize) (hf : first.val ≤ last.val) :
    countInv HomeInst home input last first (Array.repeat 256#usize 0#usize) first := by
  refine ⟨le_refl _, hf, ?_⟩
  intro s hs
  have h0 : usizeOf 0 = 0#usize := UScalar.eq_of_val_eq (by rw [usizeOf_val 0 (by simp)]; rfl)
  have hlen : s.val < (Array.repeat 256#usize 0#usize).val.length := by
    rw [Array.repeat_val, List.length_replicate]
    exact hs
  have hget : (Array.repeat 256#usize 0#usize).val[s.val]? = some 0#usize := by
    rw [List.getElem?_eq_getElem hlen]
    simp only [Array.repeat_val, List.getElem_replicate]
  rw [hget, Nat.sub_self, h0]

/-- Invert a successful `cont` result of the count body into its generated `Result` chain. -/
theorem loop0_body_cont {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {last : Usize} {counts : Array Usize 256#usize}
    {i : Usize} {a : Array Usize 256#usize} {i3 : Usize} (hlt : i < last)
    (hbody : table.partition.partition_chunk_loop0.body HomeInst home input last counts i
      = ok (ControlFlow.cont (a, i3))) :
    ∃ (p : (Faces × K) × V) (s : cell.Sigma) (address : table.TableAddress)
      (routed i1 i2 : Usize),
      Slice.index_usize input i = ok p ∧
      HomeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceDomainInst.sigma p.1.1 = ok s ∧
      HomeInst.address home s p.1.2 = ok address ∧
      table.shard address = ok routed ∧
      counts.index_usize routed = ok i1 ∧
      i1 + 1#usize = ok i2 ∧
      counts.update routed i2 = ok a ∧
      i + 1#usize = ok i3 := by
  unfold table.partition.partition_chunk_loop0.body at hbody
  rw [if_pos hlt] at hbody
  simp only [Bind.bind] at hbody
  obtain ⟨p, h1, hbody⟩ := bind_eq_ok (Slice.index_usize input i) _ _ hbody
  obtain ⟨⟨face, key⟩, v⟩ := p
  obtain ⟨s, h2, hbody⟩ :=
    bind_eq_ok (HomeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceDomainInst.sigma face) _ _ hbody
  obtain ⟨address, h3, hbody⟩ := bind_eq_ok (HomeInst.address home s key) _ _ hbody
  obtain ⟨routed, h4, hbody⟩ := bind_eq_ok (table.shard address) _ _ hbody
  obtain ⟨i1, h5, hbody⟩ := bind_eq_ok (counts.index_usize routed) _ _ hbody
  obtain ⟨i2, h6, hbody⟩ := bind_eq_ok (i1 + 1#usize) _ _ hbody
  obtain ⟨a', h7, hbody⟩ := bind_eq_ok (counts.update routed i2) _ _ hbody
  obtain ⟨i3', h8, hdone⟩ := bind_eq_ok (i + 1#usize) _ _ hbody
  have hpair : (a', i3') = (a, i3) := ControlFlow.cont.inj (Result.ok.injEq.mp hdone)
  have hae : a' = a := congrArg Prod.fst hpair
  have hie : i3' = i3 := congrArg Prod.snd hpair
  rw [hae] at h7
  rw [hie] at h8
  exact ⟨((face, key), v), s, address, routed, i1, i2,
    h1, h2, h3, h4, h5, h6, h7, h8⟩

/-- `x + 1 = ok y` yields `y = usizeOf (x.val + 1)`. -/
theorem add_one_eq {x y : Usize} (hb : x.val + 1 ≤ Usize.max) (h : x + 1#usize = ok y) :
    y = usizeOf (x.val + 1) := by
  obtain ⟨z, hz, hzv, _⟩ := WP.spec_imp_exists
    (UScalar.add_bv_spec (x := x) (y := 1#usize) (by scalar_tac))
  rw [h] at hz
  have hyz : y = z := Result.ok.injEq.mp hz
  subst hyz
  apply UScalar.eq_of_val_eq
  rw [usizeOf_val (x.val + 1) hb]
  simpa using hzv

/-- `x + 1 = ok y` yields `y.val = x.val + 1`. -/
theorem add_one_val {x y : Usize} (hb : x.val + 1 ≤ Usize.max) (h : x + 1#usize = ok y) :
    y.val = x.val + 1 := by
  rw [add_one_eq hb h, usizeOf_val (x.val + 1) hb]

/-- `x + y = ok z` yields `z.val = x.val + y.val`. -/
theorem add_val {x y z : Usize} (hb : x.val + y.val ≤ Usize.max) (h : x + y = ok z) :
    z.val = x.val + y.val := by
  have hb' : x.val + y.val ≤ (UScalar.max UScalarTy.Usize : Nat) := by scalar_tac
  obtain ⟨w, hw, hwv, _⟩ := WP.spec_imp_exists
    (UScalar.add_bv_spec (x := x) (y := y) hb')
  rw [h] at hw
  have hzw : z = w := Result.ok.injEq.mp hw
  rw [hzw]
  simpa using hwv

/-- `Array.index_usize` returns the `getElem?` at the index. -/
theorem index_usize_eq {α : Type} {n : Usize} (v : Array α n) (i : Usize) (y : α)
    (h : v.index_usize i = ok y) : v.val[i.val]? = some y := by
  cases hx : v.val[i.val]? with
  | none =>
    unfold Array.index_usize at h
    rw [Array.getElem?_Usize_eq, hx] at h
    simp at h
  | some x =>
    unfold Array.index_usize at h
    rw [Array.getElem?_Usize_eq, hx] at h
    simp only at h
    have hxy : x = y := Result.ok.injEq.mp h
    exact congrArg some hxy

/-- A successful `Vec.index_mut` proves the index in bounds: the generated model fails
    `arrayOutOfBounds` otherwise. -/
theorem vec_index_mut_bound {α : Type} (v : alloc.vec.Vec α) (i : Usize)
    (q : α × (α → alloc.vec.Vec α))
    (h : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice α) v i = ok q) :
    i.val < v.val.length := by
  rw [alloc.vec.Vec.index_mut_slice_index] at h
  unfold alloc.vec.Vec.index_mut_usize at h
  simp only [Bind.bind] at h
  obtain ⟨x, hx, _⟩ := bind_eq_ok (alloc.vec.Vec.index_usize v i) _ _ h
  by_contra hb
  have hnone : v.val[i.val]? = none := List.getElem?_eq_none_iff.mpr (Nat.le_of_not_lt hb)
  unfold alloc.vec.Vec.index_usize at hx
  simp only [alloc.vec.Vec.getElem?_Nat_eq, hnone] at hx
  exact absurd hx (by simp)

/-- Invert a successful `done` result of the count body. -/
theorem loop0_body_done {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {last : Usize} {counts : Array Usize 256#usize}
    {i : Usize} {y : Array Usize 256#usize} (hle : last.val ≤ i.val)
    (hbody : table.partition.partition_chunk_loop0.body HomeInst home input last counts i
      = ok (ControlFlow.done y)) : y = counts := by
  unfold table.partition.partition_chunk_loop0.body at hbody
  rw [if_neg (by intro h; have := (UScalar.lt_equiv i last).mp h; omega)] at hbody
  exact (ControlFlow.done.inj (Result.ok.injEq.mp hbody)).symm

/-- The count invariant is preserved by one successful `cont` step. -/
theorem countInv_step {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {last first : Usize} {counts : Array Usize 256#usize}
    {i : Usize} {a : Array Usize 256#usize} {i3 : Usize} (hlt : i < last)
    (hbound : i.val < input.val.length) (hinv : countInv HomeInst home input last first counts i)
    (hbody : table.partition.partition_chunk_loop0.body HomeInst home input last counts i
      = ok (ControlFlow.cont (a, i3))) :
    countInv HomeInst home input last first a i3 := by
  obtain ⟨⟨⟨face, key⟩, v⟩, s, address, routed, i1, i2, h1, h2, h3, h4, h5, h6, h7, h8⟩ :=
    loop0_body_cont hlt hbody
  obtain ⟨hfi, hile, hcnt⟩ := hinv
  -- value facts
  have himax : i.val ≤ Usize.max := by
    have := input.property; omega
  have hltmax : i.val + 1 ≤ Usize.max := by
    have := input.property; omega
  have hface : ((face, key), v) = input.val[i.val] := by
    rw [slice_index_ok input i hbound] at h1
    exact (Result.ok.injEq.mp h1).symm
  have hps : producerShard HomeInst home input i = ok routed :=
    producerShard_eq_ok hbound hface h2 h3 h4
  have hrt : ∀ s' : Usize, routedTo HomeInst home input s' i.val = true ↔ s'.val = routed.val :=
    fun s' => producerShard_routedTo_iff (usizeOf_eq i himax) hps
  have hrouted : routed.val < 256 := by
    obtain ⟨r, hr, hb⟩ := WP.spec_imp_exists (shard_lt_256 address)
    rw [h4] at hr
    have hrr : r = routed := (Result.ok.injEq.mp hr).symm
    rwa [hrr] at hb
  have hbnd : routed.val < counts.length := by simpa [Array.length_eq] using hrouted
  have hi1some : counts.val[routed.val]? = some i1 := index_usize_eq counts routed i1 h5
  have hi3 : i3 = usizeOf (i.val + 1) := add_one_eq hltmax h8
  have hi3v : i3.val = i.val + 1 := add_one_val hltmax h8
  have ha_set : a = counts.set routed i2 := by
    obtain ⟨nv, hnv, hset⟩ := WP.spec_imp_exists
      (Array.update_spec counts routed i2 (by simpa using hbnd))
    rw [h7] at hnv
    have hane : a = nv := Result.ok.injEq.mp hnv
    rw [hane, hset]
  refine ⟨by rw [hi3v]; omega,
    by rw [hi3v]; have := (UScalar.lt_equiv i last).mp hlt; omega, ?_⟩
  intro s hs
  rw [ha_set, Array.set_val_eq]
  have hlenI : (routedRun HomeInst home input i routed).length ≤ i.val := by
    rw [routedRun]
    exact (List.length_filter_le _ _).trans (by simp)
  have hlenImax : (routedRun HomeInst home input i routed).length ≤ Usize.max :=
    hlenI.trans himax
  have hFN : (routedRun HomeInst home input first routed).length
      ≤ (routedRun HomeInst home input i routed).length :=
    routedRun_length_mono hfi
  by_cases hsr : s.val = routed.val
  · have hse : s = routed := UScalar.eq_of_val_eq hsr
    rw [hse]
    rw [List.getElem?_set_self (by rw [Array.length_eq]; exact hrouted)]
    have hrtrue : routedTo HomeInst home input routed i.val = true := (hrt routed).2 rfl
    have hrun : routedRun HomeInst home input i3 routed
        = routedRun HomeInst home input i routed ++ [i.val] := by
      rw [hi3, routedRun_step hltmax, if_pos hrtrue]
    have hi1eq : i1 = usizeOf ((routedRun HomeInst home input i routed).length
        - (routedRun HomeInst home input first routed).length) := by
      have hci := hcnt routed hrouted
      rw [hi1some] at hci
      exact Option.some.inj hci
    have hi1v : i1.val = (routedRun HomeInst home input i routed).length
        - (routedRun HomeInst home input first routed).length := by
      rw [hi1eq, usizeOf_val _ (by omega)]
    rw [Option.some.injEq]
    apply UScalar.eq_of_val_eq
    have hb3 : (routedRun HomeInst home input i3 routed).length
        - (routedRun HomeInst home input first routed).length ≤ Usize.max := by
      rw [hrun, List.length_append, List.length_singleton]
      omega
    rw [usizeOf_val _ hb3, add_one_val (by rw [hi1v]; omega) h6, hi1v,
      hrun, List.length_append, List.length_singleton]
    omega
  · have hrfalse : routedTo HomeInst home input s i.val = false := by
      cases h : routedTo HomeInst home input s i.val with
      | true => exact absurd ((hrt s).1 h) hsr
      | false => rfl
    rw [List.getElem?_set_ne (Ne.symm hsr), hi3,
      routedRun_step hltmax, if_neg (by simpa using hrfalse)]
    exact hcnt s hs

/-- A successful count body with `i < last` always returns a `cont`. -/
theorem loop0_body_out {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {last : Usize} {counts : Array Usize 256#usize}
    {i : Usize} {out : ControlFlow (Array Usize 256#usize × Usize) (Array Usize 256#usize)}
    (hlt : i < last)
    (hbody : table.partition.partition_chunk_loop0.body HomeInst home input last counts i = ok out) :
    ∃ (a : Array Usize 256#usize) (i3 : Usize), out = ControlFlow.cont (a, i3) := by
  unfold table.partition.partition_chunk_loop0.body at hbody
  rw [if_pos hlt] at hbody
  simp only [Bind.bind] at hbody
  obtain ⟨p, _, hbody⟩ := bind_eq_ok (Slice.index_usize input i) _ _ hbody
  obtain ⟨⟨face, key⟩, v⟩ := p
  obtain ⟨s, _, hbody⟩ :=
    bind_eq_ok (HomeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceDomainInst.sigma face) _ _ hbody
  obtain ⟨address, _, hbody⟩ := bind_eq_ok (HomeInst.address home s key) _ _ hbody
  obtain ⟨routed, _, hbody⟩ := bind_eq_ok (table.shard address) _ _ hbody
  obtain ⟨i1, _, hbody⟩ := bind_eq_ok (counts.index_usize routed) _ _ hbody
  obtain ⟨i2, _, hbody⟩ := bind_eq_ok (i1 + 1#usize) _ _ hbody
  obtain ⟨a', _, hbody⟩ := bind_eq_ok (counts.update routed i2) _ _ hbody
  obtain ⟨i3', _, hdone⟩ := bind_eq_ok (i + 1#usize) _ _ hbody
  exact ⟨a', i3', (Result.ok.injEq.mp hdone).symm⟩

/-- **The count pass.** A successful `partition_chunk_loop0` from the zero array stores, per shard,
    the number of input indexes of `[first, last)` the producer's routing selects for it.  Stated as
    an implication on the successful result: the loop may fail at `Home::address`. -/
theorem loop0_spec {K V H Faces : Type} (HomeInst : table.Home H K Faces) (home : H)
    (input : Slice ((Faces × K) × V)) (last first : Usize)
    (hlen : last.val ≤ input.val.length) (hf : first.val ≤ last.val) :
    ∀ counts, table.partition.partition_chunk_loop0 HomeInst home input last
        (Array.repeat 256#usize 0#usize) first = ok counts →
      ∀ s : Usize, s.val < 256 →
        counts.val[s.val]? = some (usizeOf ((routedRun HomeInst home input last s).length
          - (routedRun HomeInst home input first s).length)) := by
  unfold table.partition.partition_chunk_loop0
  refine loop_ok_spec
    (measure := fun p : Array Usize 256#usize × Usize => last.val - p.2.val)
    (inv := fun p => countInv HomeInst home input last first p.1 p.2)
    (post := fun counts => ∀ s : Usize, s.val < 256 →
      counts.val[s.val]? = some (usizeOf ((routedRun HomeInst home input last s).length
        - (routedRun HomeInst home input first s).length)))
    (body := fun p => table.partition.partition_chunk_loop0.body HomeInst home input last p.1 p.2)
    ?_ (Array.repeat 256#usize 0#usize, first) (countInv_init HomeInst home input last first hf)
  rintro ⟨counts, i⟩ hx
  dsimp only at hx ⊢
  obtain ⟨hfi, hile, hcnt⟩ := hx
  by_cases hlt : i < last
  · have hltmax : i.val + 1 ≤ Usize.max := by
      have hl := (UScalar.lt_equiv i last).mp hlt
      have := input.property
      omega
    refine ⟨?_, ?_⟩
    · intro y hbody
      obtain ⟨a, i3, hout⟩ := loop0_body_out hlt hbody
      exact absurd hout (by intro h; cases h)
    · intro x' hbody
      obtain ⟨a, i3⟩ := x'
      dsimp only at hbody ⊢
      obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, h8⟩ := loop0_body_cont hlt hbody
      have hi3v : i3.val = i.val + 1 := add_one_val hltmax h8
      refine ⟨countInv_step hlt ?_ ⟨hfi, hile, hcnt⟩ hbody, ?_⟩
      · have := (UScalar.lt_equiv i last).mp hlt; omega
      · rw [hi3v]; have := (UScalar.lt_equiv i last).mp hlt; omega
  · refine ⟨?_, ?_⟩
    · intro y hbody
      have hle : last.val ≤ i.val := by
        have := (UScalar.lt_equiv i last).not.mp hlt; omega
      have hy := loop0_body_done hle hbody
      have hieq : i.val = last.val := le_antisymm hile hle
      rw [hy]
      intro s hs
      have hruni : routedRun HomeInst home input i s = routedRun HomeInst home input last s := by
        rw [routedRun, routedRun, hieq]
      have := hcnt s hs
      rw [hruni] at this
      exact this
    · intro x' hbody
      obtain ⟨a, i3⟩ := x'
      unfold table.partition.partition_chunk_loop0.body at hbody
      rw [if_neg hlt] at hbody
      exact absurd (Result.ok.injEq.mp hbody) (by intro h; cases h)

/-! ## The prefix pass -/

/-- The prefix sum of the first `n` counts. -/
def prefixSum (counts : Array Usize 256#usize) (n : Nat) : Nat :=
  (List.range n).foldl (fun acc u => acc + ((counts.val[u]?).getD 0#usize).val) 0

theorem prefixSum_zero (counts : Array Usize 256#usize) : prefixSum counts 0 = 0 := rfl

theorem prefixSum_succ (counts : Array Usize 256#usize) (n : Nat) :
    prefixSum counts (n + 1) = prefixSum counts n + ((counts.val[n]?).getD 0#usize).val := by
  unfold prefixSum
  rw [List.range_succ, List.foldl_append]
  rfl

theorem prefixSum_mono (counts : Array Usize 256#usize) {m n : Nat} (h : m ≤ n) :
    prefixSum counts m ≤ prefixSum counts n := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le h
  induction k with
  | zero => simp
  | succ k ih => rw [Nat.add_succ, prefixSum_succ]; omega

/-- `prefixSum` as a `Finset.range` sum. -/
theorem prefixSum_eq_finset (counts : Array Usize 256#usize) (n : Nat) :
    prefixSum counts n
      = ∑ u ∈ Finset.range n, ((counts.val[u]?).getD 0#usize).val := by
  induction n with
  | zero => simp [prefixSum]
  | succ n ih => rw [prefixSum_succ, ih, Finset.sum_range_succ]

/-- Reading a `set` at one index: `getD` and the pointwise `if`. -/
theorem getD_set (counts : Array Usize 256#usize) (k v : Usize) (u : Nat) (hk : k.val < 256) :
    ((counts.set k v).val[u]?).getD 0#usize
      = if u = k.val then v else (counts.val[u]?).getD 0#usize := by
  rw [Array.set_val_eq]
  by_cases h : u = k.val
  · subst h
    rw [List.getElem?_set_self (by rw [Array.length_eq]; exact hk)]
    simp
  · rw [List.getElem?_set_ne (Ne.symm h), if_neg h]

/-- **Array-sum under `set`.** Changing one entry of a 256-entry `counts` changes `prefixSum` by the
    difference of the values; subtraction needs the new value at least the old. -/
theorem prefixSum_set (counts : Array Usize 256#usize) (k v : Usize) (hk : k.val < 256)
    (hv : ((counts.val[k.val]?).getD 0#usize).val ≤ v.val) :
    prefixSum (counts.set k v) 256 + ((counts.val[k.val]?).getD 0#usize).val
      = prefixSum counts 256 + v.val := by
  rw [prefixSum_eq_finset, prefixSum_eq_finset]
  have hfun : ∀ u ∈ (Finset.range 256 : Finset Nat),
      (((counts.set k v).val[u]?).getD 0#usize).val
        = (if u = k.val then v.val else (((counts.val[u]?).getD 0#usize)).val) := by
    intro u _
    rw [getD_set counts k v u hk]
    by_cases h : u = k.val <;> simp [h]
  rw [Finset.sum_congr rfl hfun]
  have hkmem : k.val ∈ (Finset.range 256 : Finset Nat) := Finset.mem_range.mpr hk
  have hpt : ∀ u, (if u = k.val then v.val else ((counts.val[u]?).getD 0#usize).val)
      = ((counts.val[u]?).getD 0#usize).val
        + (if u = k.val then (v.val - ((counts.val[k.val]?).getD 0#usize).val) else 0) := by
    intro u
    by_cases h : u = k.val
    · subst h
      rw [if_pos rfl, if_pos rfl]
      omega
    · rw [if_neg h, if_neg h, Nat.add_zero]
  have hsum : (∑ u ∈ (Finset.range 256 : Finset Nat),
        (if u = k.val then v.val else ((counts.val[u]?).getD 0#usize).val))
      = (∑ u ∈ (Finset.range 256 : Finset Nat), ((counts.val[u]?).getD 0#usize).val)
        + (v.val - ((counts.val[k.val]?).getD 0#usize).val) := by
    rw [Finset.sum_congr rfl (fun u _ => hpt u), Finset.sum_add_distrib,
      Finset.sum_ite_eq', if_pos hkmem]
  rw [hsum]
  omega

theorem prefixSum_repeat_zero : prefixSum (Array.repeat 256#usize 0#usize) 256 = 0 := by
  rw [prefixSum_eq_finset]
  apply Finset.sum_eq_zero
  intro u _
  have h : (((Array.repeat 256#usize 0#usize).val[u]?).getD 0#usize).val = 0 := by
    rw [Array.repeat_val, List.getElem?_replicate]
    by_cases hu : u < 256 <;> simp [hu]
  exact h

/-- **The count pass, summed.** A successful `partition_chunk_loop0` from the zero array has
    `prefixSum counts 256 = last - first`: every index of `[first, last)` increments exactly one
    count, so the total counts the window length. -/
theorem loop0_spec_sum {K V H Faces : Type} (HomeInst : table.Home H K Faces) (home : H)
    (input : Slice ((Faces × K) × V)) (last first : Usize)
    (hlen : last.val ≤ input.val.length) (hf : first.val ≤ last.val) :
    ∀ counts, table.partition.partition_chunk_loop0 HomeInst home input last
        (Array.repeat 256#usize 0#usize) first = ok counts →
      prefixSum counts 256 = last.val - first.val := by
  unfold table.partition.partition_chunk_loop0
  refine loop_ok_spec
    (measure := fun p : Array Usize 256#usize × Usize => last.val - p.2.val)
    (inv := fun p => countInv HomeInst home input last first p.1 p.2 ∧
      prefixSum p.1 256 = p.2.val - first.val)
    (post := fun counts => prefixSum counts 256 = last.val - first.val)
    (body := fun p => table.partition.partition_chunk_loop0.body HomeInst home input last p.1 p.2)
    ?_ (Array.repeat 256#usize 0#usize, first) ?_
  · rintro ⟨counts, i⟩ ⟨hinv, hsum⟩
    have hinv0 := hinv
    obtain ⟨hfi, hile, _hcnt⟩ := hinv
    simp only [Prod.fst, Prod.snd] at hsum hfi hile
    by_cases hlt : i < last
    · have hltmax : i.val + 1 ≤ Usize.max := by
        have hl := (UScalar.lt_equiv i last).mp hlt
        have := input.property
        omega
      refine ⟨?_, ?_⟩
      · intro y hbody
        obtain ⟨a, i3, hout⟩ := loop0_body_out hlt hbody
        exact absurd hout (by intro h; cases h)
      · intro x' hbody
        obtain ⟨a, i3⟩ := x'
        dsimp only at hbody ⊢
        obtain ⟨⟨⟨face, key⟩, v⟩, s, address, routed, i1, i2, h1, h2, h3, h4, h5, h6, h7, h8⟩ :=
          loop0_body_cont hlt hbody
        have hi3v : i3.val = i.val + 1 := add_one_val hltmax h8
        refine ⟨⟨countInv_step hlt ?_ hinv0 hbody, ?_⟩, ?_⟩
        · have := (UScalar.lt_equiv i last).mp hlt
          omega
        · have hrouted : routed.val < 256 := by
            obtain ⟨r, hr, hb⟩ := WP.spec_imp_exists (shard_lt_256 address)
            rw [h4] at hr
            have hrr : r = routed := (Result.ok.injEq.mp hr).symm
            rwa [hrr] at hb
          have hbnd : routed.val < counts.length := by simpa [Array.length_eq] using hrouted
          have ha : a = counts.set routed i2 := by
            obtain ⟨nv, hnv, hset⟩ := WP.spec_imp_exists
              (Array.update_spec counts routed i2 (by simpa using hbnd))
            rw [h7] at hnv
            have hane : a = nv := Result.ok.injEq.mp hnv
            rw [hane, hset]
          have hi1some : counts.val[routed.val]? = some i1 := index_usize_eq counts routed i1 h5
          have hi2v : i2.val = i1.val + 1 := add_val_of_ok h6
          have hi1v : i1.val = ((counts.val[routed.val]?).getD 0#usize).val := by simp [hi1some]
          have hv : ((counts.val[routed.val]?).getD 0#usize).val ≤ i2.val := by
            rw [hi2v, hi1v]; omega
          have hps : prefixSum (counts.set routed i2) 256 = prefixSum counts 256 + 1 := by
            have hset := prefixSum_set counts routed i2 hrouted hv
            rw [← hi1v, hi2v] at hset
            omega
          rw [ha, hps, hi3v]
          omega
        · rw [hi3v]
          have := (UScalar.lt_equiv i last).mp hlt
          omega
    · refine ⟨?_, ?_⟩
      · intro y hbody
        have hle : last.val ≤ i.val := by
          have := (UScalar.lt_equiv i last).not.mp hlt
          omega
        have hy := loop0_body_done hle hbody
        have hieq : i.val = last.val := le_antisymm hile hle
        rw [hy]
        rw [hieq] at hsum
        exact hsum
      · intro x' hbody
        obtain ⟨a, i3⟩ := x'
        unfold table.partition.partition_chunk_loop0.body at hbody
        rw [if_neg hlt] at hbody
        exact absurd (Result.ok.injEq.mp hbody) (by intro h; cases h)
  · refine ⟨countInv_init HomeInst home input last first hf, ?_⟩
    rw [prefixSum_repeat_zero]
    simp


/-- The prefix loop's invariant: `total` is the running prefix sum and `ends`/`cursor` store every
    earlier prefix sum. -/
def prefixInv (counts : Array Usize 256#usize) (ends : Array Usize 257#usize)
    (cursor : Array Usize 256#usize) (total : Usize) (s : Usize) : Prop :=
  s.val ≤ 256 ∧
  total.val = prefixSum counts s.val ∧
  (∀ t : Usize, t.val < s.val → ends.val[t.val]? = some (usizeOf (prefixSum counts t.val))) ∧
  (∀ t : Usize, t.val < s.val → cursor.val[t.val]? = some (usizeOf (prefixSum counts t.val)))

/-- `Array.update` at an in-range index is `Array.set`. -/
theorem update_eq_set {α : Type} {n : Usize} (v : Array α n) (i : Usize) (x : α)
    (hb : i.val < n.val) (y : Array α n) (h : v.update i x = ok y) : y = v.set i x := by
  obtain ⟨nv, hnv, hset⟩ := WP.spec_imp_exists (Array.update_spec v i x (by simpa using hb))
  rw [h] at hnv
  have hyv : y = nv := Result.ok.injEq.mp hnv
  rw [hyv, hset]

theorem SHARDS_eq : table.partition.SHARDS = ok 256#usize := by
  obtain ⟨r, hr, hv⟩ := WP.spec_imp_exists SHARDS_spec
  rw [hr, hv]

/-- Invert a successful `cont` result of the prefix body. -/
theorem loop1_body_cont {counts : Array Usize 256#usize} {ends : Array Usize 257#usize}
    {cursor : Array Usize 256#usize} {total s : Usize}
    {a : Array Usize 257#usize} {a1 : Array Usize 256#usize} {total1 s1 : Usize}
    (hbody : table.partition.partition_chunk_loop1.body counts ends cursor total s
      = ok (ControlFlow.cont (a, a1, total1, s1))) :
    ∃ (i i1 : Usize), table.partition.SHARDS = ok i ∧ s < i ∧
      ends.update s total = ok a ∧ cursor.update s total = ok a1 ∧
      counts.index_usize s = ok i1 ∧ total + i1 = ok total1 ∧ s + 1#usize = ok s1 := by
  unfold table.partition.partition_chunk_loop1.body at hbody
  simp only [Bind.bind] at hbody
  obtain ⟨i, hi, hbody⟩ := bind_eq_ok (table.partition.SHARDS) _ _ hbody
  by_cases hlt : s < i
  · rw [if_pos hlt] at hbody
    obtain ⟨a', h4, hbody⟩ := bind_eq_ok (ends.update s total) _ _ hbody
    obtain ⟨a1', h5, hbody⟩ := bind_eq_ok (cursor.update s total) _ _ hbody
    obtain ⟨i1, h6, hbody⟩ := bind_eq_ok (counts.index_usize s) _ _ hbody
    obtain ⟨total1', h7, hbody⟩ := bind_eq_ok (total + i1) _ _ hbody
    obtain ⟨s1', h8, hdone⟩ := bind_eq_ok (s + 1#usize) _ _ hbody
    have hpair : (a', a1', total1', s1') = (a, a1, total1, s1) :=
      ControlFlow.cont.inj (Result.ok.injEq.mp hdone)
    have ha : a' = a := congrArg Prod.fst hpair
    have ha1 : a1' = a1 := congrArg (fun q => q.2.1) hpair
    have ht : total1' = total1 := congrArg (fun q => q.2.2.1) hpair
    have hs1 : s1' = s1 := congrArg (fun q => q.2.2.2) hpair
    rw [ha] at h4
    rw [ha1] at h5
    rw [ht] at h7
    rw [hs1] at h8
    exact ⟨i, i1, hi, hlt, h4, h5, h6, h7, h8⟩
  · rw [if_neg hlt] at hbody
    exact absurd (Result.ok.injEq.mp hbody) (by intro h; cases h)

/-- Invert a successful `done` result of the prefix body. -/
theorem loop1_body_done {counts : Array Usize 256#usize} {ends : Array Usize 257#usize}
    {cursor : Array Usize 256#usize} {total s : Usize}
    {e : Array Usize 257#usize} {c : Array Usize 256#usize} {tot : Usize}
    (hbody : table.partition.partition_chunk_loop1.body counts ends cursor total s
      = ok (ControlFlow.done (e, c, tot))) :
    ∃ i : Usize, table.partition.SHARDS = ok i ∧ ¬ s < i ∧ e = ends ∧ c = cursor ∧ tot = total := by
  unfold table.partition.partition_chunk_loop1.body at hbody
  simp only [Bind.bind] at hbody
  obtain ⟨i, hi, hbody⟩ := bind_eq_ok (table.partition.SHARDS) _ _ hbody
  by_cases hlt : s < i
  · rw [if_pos hlt] at hbody
    obtain ⟨_, _, hbody⟩ := bind_eq_ok (ends.update s total) _ _ hbody
    obtain ⟨_, _, hbody⟩ := bind_eq_ok (cursor.update s total) _ _ hbody
    obtain ⟨i1, _, hbody⟩ := bind_eq_ok (counts.index_usize s) _ _ hbody
    obtain ⟨_, _, hbody⟩ := bind_eq_ok (total + i1) _ _ hbody
    obtain ⟨_, _, hdone⟩ := bind_eq_ok (s + 1#usize) _ _ hbody
    exact absurd (Result.ok.injEq.mp hdone) (by intro h; cases h)
  · rw [if_neg hlt] at hbody
    have hpair : (ends, cursor, total) = (e, c, tot) :=
      ControlFlow.done.inj (Result.ok.injEq.mp hbody)
    exact ⟨i, hi, hlt, (congrArg Prod.fst hpair).symm,
      (congrArg (fun q => q.2.1) hpair).symm, (congrArg (fun q => q.2.2) hpair).symm⟩

/-- **The prefix pass.** `partition_chunk_loop1` from the zero arrays stores the prefix sums of
    `counts` in both `ends` and `cursor` and returns their total, under the bound
    `Σ counts ≤ Usize.max` that the count pass supplies. -/
theorem loop1_spec (counts : Array Usize 256#usize) (hsum : prefixSum counts 256 ≤ Usize.max) :
    ∀ e c tot, table.partition.partition_chunk_loop1 counts (Array.repeat 257#usize 0#usize)
        (Array.repeat 256#usize 0#usize) 0#usize 0#usize = ok (e, c, tot) →
      (∀ t : Usize, t.val < 256 → e.val[t.val]? = some (usizeOf (prefixSum counts t.val))) ∧
      (∀ t : Usize, t.val < 256 → c.val[t.val]? = some (usizeOf (prefixSum counts t.val))) ∧
      tot.val = prefixSum counts 256 := by
  unfold table.partition.partition_chunk_loop1
  intro e c tot hok
  refine loop_ok_spec
    (measure := fun p : Array Usize 257#usize × Array Usize 256#usize × Usize × Usize =>
      256 - p.2.2.2.val)
    (inv := fun p => prefixInv counts p.1 p.2.1 p.2.2.1 p.2.2.2)
    (post := fun q : Array Usize 257#usize × Array Usize 256#usize × Usize =>
      (∀ t : Usize, t.val < 256 → q.1.val[t.val]? = some (usizeOf (prefixSum counts t.val))) ∧
      (∀ t : Usize, t.val < 256 → q.2.1.val[t.val]? = some (usizeOf (prefixSum counts t.val))) ∧
      q.2.2.val = prefixSum counts 256)
    (body := fun p => table.partition.partition_chunk_loop1.body counts p.1 p.2.1 p.2.2.1 p.2.2.2)
    ?_ (Array.repeat 257#usize 0#usize, Array.repeat 256#usize 0#usize, 0#usize, 0#usize) ?_
    (e, c, tot) hok
  · rintro ⟨ends, cursor, total, s⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hle, htot, hends, hcur⟩ := hinv
    have hsb : prefixSum counts s.val ≤ Usize.max :=
      le_trans (prefixSum_mono counts (by omega)) hsum
    have htotEq : total = usizeOf (prefixSum counts s.val) :=
      UScalar.eq_of_val_eq (by rw [htot, usizeOf_val _ hsb])
    constructor
    · intro q hbody
      obtain ⟨e, c, tot⟩ := q
      obtain ⟨i, hi, hnlt, he, hc, ht⟩ := loop1_body_done hbody
      have hi256 : i = 256#usize := by
        rw [SHARDS_eq] at hi
        exact Result.ok.injEq.mp hi.symm
      have hs256 : s.val = 256 := by
        rw [hi256] at hnlt
        have : ¬ s.val < 256 := by simpa [UScalar.lt_equiv] using hnlt
        omega
      rw [he, hc, ht]
      exact ⟨fun t ht' => hends t (by rw [hs256]; exact ht'),
        fun t ht' => hcur t (by rw [hs256]; exact ht'), by rw [htot, hs256]⟩
    · intro q' hbody
      obtain ⟨e, c, tot, s'⟩ := q'
      obtain ⟨i, i1, hi, hslt, h4, h5, h6, h7, h8⟩ := loop1_body_cont hbody
      have hi256 : i = 256#usize := by
        rw [SHARDS_eq] at hi
        exact Result.ok.injEq.mp hi.symm
      have hs256 : s.val < 256 := by
        rw [hi256] at hslt
        exact (UScalar.lt_equiv s 256#usize).mp hslt
      have hs257 : s.val < 257 := by omega
      have he : e = ends.set s total := update_eq_set ends s total hs257 e h4
      have hc : c = cursor.set s total := update_eq_set cursor s total hs256 c h5
      have hi1 : counts.val[s.val]? = some i1 := index_usize_eq counts s i1 h6
      have hi1v : i1.val = ((counts.val[s.val]?).getD 0#usize).val := by simp [hi1]
      have htv : total.val + i1.val = prefixSum counts (s.val + 1) := by
        rw [htot, hi1v, prefixSum_succ]
      have hsumle : prefixSum counts (s.val + 1) ≤ Usize.max :=
        le_trans (prefixSum_mono counts (by omega)) hsum
      have hsumle' : total.val + i1.val ≤ Usize.max := by rw [htv]; exact hsumle
      have htv2 : tot.val = total.val + i1.val := add_val hsumle' h7
      have ht1 : tot = usizeOf (prefixSum counts (s.val + 1)) :=
        UScalar.eq_of_val_eq (by rw [htv2, htv, usizeOf_val (prefixSum counts (s.val + 1)) hsumle])
      have hsv : s'.val = s.val + 1 := add_one_val (by scalar_tac) h8
      refine ⟨?_, ?_⟩
      · refine ⟨by rw [hsv]; omega, ?_, ?_, ?_⟩
        · rw [ht1, hsv, usizeOf_val (prefixSum counts (s.val + 1)) hsumle]
        · intro t ht'
          rw [he, Array.set_val_eq]
          by_cases htst : t.val = s.val
          · rw [htst, List.getElem?_set_self (by rw [Array.length_eq]; exact hs257)]
            exact congrArg some (by
              apply UScalar.eq_of_val_eq
              rw [usizeOf_val (prefixSum counts s.val) hsb, htot])
          · rw [List.getElem?_set_ne (by omega)]
            exact hends t (by rw [hsv] at ht'; omega)
        · intro t ht'
          rw [hc, Array.set_val_eq]
          by_cases htst : t.val = s.val
          · rw [htst, List.getElem?_set_self (by rw [Array.length_eq]; exact hs256)]
            exact congrArg some (by
              apply UScalar.eq_of_val_eq
              rw [usizeOf_val (prefixSum counts s.val) hsb, htot])
          · rw [List.getElem?_set_ne (by omega)]
            exact hcur t (by rw [hsv] at ht'; omega)
      · rw [hsv]; omega
  · refine ⟨by norm_num, ?_, ?_, ?_⟩
    · rfl
    · intro t ht; simp at ht
    · intro t ht; simp at ht

/-- The prefix pass's accumulated total is a machine word, so a successful `partition_chunk_loop1`
    bounds `prefixSum counts 256` — no external premise. -/
theorem prefixSum_le_max_of_loop1 (counts : Array Usize 256#usize)
    {e : Array Usize 257#usize} {c : Array Usize 256#usize} {tot : Usize}
    (hok : table.partition.partition_chunk_loop1 counts (Array.repeat 257#usize 0#usize)
        (Array.repeat 256#usize 0#usize) 0#usize 0#usize = ok (e, c, tot)) :
    prefixSum counts 256 ≤ Usize.max := by
  have htot : tot.val = prefixSum counts 256 := by
    unfold table.partition.partition_chunk_loop1 at hok
    refine loop_ok_spec
      (measure := fun p : Array Usize 257#usize × Array Usize 256#usize × Usize × Usize =>
        256 - p.2.2.2.val)
      (inv := fun p => p.2.2.1.val = prefixSum counts p.2.2.2.val ∧ p.2.2.2.val ≤ 256)
      (post := fun q : Array Usize 257#usize × Array Usize 256#usize × Usize =>
        q.2.2.val = prefixSum counts 256)
      (body := fun p => table.partition.partition_chunk_loop1.body counts p.1 p.2.1 p.2.2.1 p.2.2.2)
      ?_ (Array.repeat 257#usize 0#usize, Array.repeat 256#usize 0#usize, 0#usize, 0#usize)
      ?_ (e, c, tot) hok
    · rintro ⟨ends, cursor, total, s⟩ ⟨htot, hsle⟩
      dsimp only at htot hsle ⊢
      constructor
      · intro q hbody
        obtain ⟨e', c', tot'⟩ := q
        obtain ⟨i, hi, hnlt, he, hc, ht⟩ := loop1_body_done hbody
        have hi256 : i = 256#usize := by
          rw [SHARDS_eq] at hi
          exact Result.ok.injEq.mp hi.symm
        have hs256 : s.val = 256 := by
          rw [hi256] at hnlt
          have : ¬ s.val < 256 := by simpa [UScalar.lt_equiv] using hnlt
          omega
        subst ht
        exact htot.trans (congrArg (prefixSum counts) hs256)
      · intro q' hbody
        obtain ⟨e', c', tot', s'⟩ := q'
        obtain ⟨i, i1, hi, hslt, h4, h5, h6, h7, h8⟩ := loop1_body_cont hbody
        have hi256 : i = 256#usize := by
          rw [SHARDS_eq] at hi
          exact Result.ok.injEq.mp hi.symm
        have hs256 : s.val < 256 := by
          rw [hi256] at hslt
          exact (UScalar.lt_equiv s 256#usize).mp hslt
        have hi1 : counts.val[s.val]? = some i1 := index_usize_eq counts s i1 h6
        have hi1v : i1.val = ((counts.val[s.val]?).getD 0#usize).val := by simp [hi1]
        have htv : total.val + i1.val = prefixSum counts (s.val + 1) := by
          rw [htot, hi1v, prefixSum_succ]
        have htv2 : tot'.val = total.val + i1.val := add_val_of_ok h7
        have hsv : s'.val = s.val + 1 := add_one_val (by scalar_tac) h8
        refine ⟨?_, ?_⟩
        · refine ⟨?_, ?_⟩
          · rw [htv2, htv, hsv]
          · rw [hsv]; omega
        · rw [hsv]; omega
    · refine ⟨?_, ?_⟩
      · simp [prefixSum]
      · norm_num
  rw [← htot]
  exact usize_val_le_max tot

/-! ## The scatter pass -/

/-- The input indexes of `[first, i)` routed to shard `s`, in increasing order.  Equal to
    `routedRun i` with the `[0, first)` routed prefix dropped. -/
noncomputable def routedMid {K V H Faces : Type} (HomeInst : table.Home H K Faces) (home : H)
    (input : Slice ((Faces × K) × V)) (first i s : Usize) : List Nat :=
  (routedRun HomeInst home input i s).drop ((routedRun HomeInst home input first s).length)

theorem routedMid_self {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {first s : Usize} :
    routedMid HomeInst home input first first s = [] := by
  unfold routedMid
  simp

/-- With `first = 0` the routed mid-run of a window is the whole routed run. -/
theorem routedMid_zero {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {i s : Usize} :
    routedMid HomeInst home input 0#usize i s = routedRun HomeInst home input i s := by
  unfold routedMid
  rw [routedRun_nil, List.length_nil, List.drop_zero]

/-- One scatter step: the route at `i` appends `i` to its shard's mid-run or leaves it. -/
theorem routedMid_step {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {first i s : Usize} (hfi : first.val ≤ i.val)
    (hi : i.val + 1 ≤ Usize.max) :
    routedMid HomeInst home input first (usizeOf (i.val + 1)) s =
      (if routedTo HomeInst home input s i.val then
        routedMid HomeInst home input first i s ++ [i.val]
      else routedMid HomeInst home input first i s) := by
  unfold routedMid
  rw [routedRun_step hi]
  by_cases h : routedTo HomeInst home input s i.val
  · rw [if_pos h, if_pos h, List.drop_append]
    have hF : (routedRun HomeInst home input first s).length
        ≤ (routedRun HomeInst home input i s).length := routedRun_length_mono hfi
    rw [Nat.sub_eq_zero_of_le hF, List.drop_zero]
  · rw [if_neg h, if_neg h]

/-- The base shard offset `ends[s]` read as a natural. -/
def endsVal (ends : Array Usize 257#usize) (s : Usize) : Nat :=
  ((ends.val[s.val]?).getD 0#usize).val

/-- Invert a successful `cont` result of the scatter body. -/
theorem loop2_body_cont {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {last : Usize} {i : Usize}
    {cursor : Array Usize 256#usize} {buf : alloc.vec.Vec Usize}
    {i3 : Usize} {a : Array Usize 256#usize} {buf1 : alloc.vec.Vec Usize}
    (hlt : i < last)
    (hbody : table.partition.partition_chunk_loop2.body HomeInst home input last i cursor buf
      = ok (ControlFlow.cont (i3, a, buf1))) :
    ∃ (p : (Faces × K) × V) (s : cell.Sigma) (address : table.TableAddress) (s1 i1 i2 : Usize),
      Slice.index_usize input i = ok p ∧
      HomeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceDomainInst.sigma p.1.1 = ok s ∧
      HomeInst.address home s p.1.2 = ok address ∧
      table.shard address = ok s1 ∧
      cursor.index_usize s1 = ok i1 ∧
      buf1 = buf.set i1 i ∧
      i1 + 1#usize = ok i2 ∧
      cursor.update s1 i2 = ok a ∧
      i + 1#usize = ok i3 ∧
      i1.val < buf.val.length := by
  unfold table.partition.partition_chunk_loop2.body at hbody
  rw [if_pos hlt] at hbody
  simp only [Bind.bind] at hbody
  obtain ⟨p, h1, hbody⟩ := bind_eq_ok (Slice.index_usize input i) _ _ hbody
  obtain ⟨⟨face, key⟩, v⟩ := p
  obtain ⟨s, h2, hbody⟩ :=
    bind_eq_ok (HomeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceDomainInst.sigma face) _ _ hbody
  obtain ⟨address, h3, hbody⟩ := bind_eq_ok (HomeInst.address home s key) _ _ hbody
  obtain ⟨s1, h4, hbody⟩ := bind_eq_ok (table.shard address) _ _ hbody
  obtain ⟨i1, h5, hbody⟩ := bind_eq_ok (cursor.index_usize s1) _ _ hbody
  obtain ⟨q, h6, hbody⟩ := bind_eq_ok (alloc.vec.Vec.index_mut
    (core.slice.index.SliceIndexUsizeSlice Std.Usize) buf i1) _ _ hbody
  have hbnd : i1.val < buf.val.length := vec_index_mut_bound buf i1 q h6
  obtain ⟨o, back⟩ := q
  obtain ⟨i2, h7, hbody⟩ := bind_eq_ok (i1 + 1#usize) _ _ hbody
  obtain ⟨a', h8, hbody⟩ := bind_eq_ok (cursor.update s1 i2) _ _ hbody
  obtain ⟨i3', h9, hdone⟩ := bind_eq_ok (i + 1#usize) _ _ hbody
  have hpair : (i3', a', back i) = (i3, a, buf1) :=
    ControlFlow.cont.inj (Result.ok.injEq.mp hdone)
  have hi3 : i3' = i3 := congrArg Prod.fst hpair
  have ha : a' = a := congrArg (fun q => q.2.1) hpair
  have hbi : back i = buf1 := congrArg (fun q => q.2.2) hpair
  have hback : back = buf.set i1 := by
    rw [alloc.vec.Vec.index_mut_slice_index] at h6
    unfold alloc.vec.Vec.index_mut_usize at h6
    simp only [Bind.bind] at h6
    obtain ⟨o', _, h6b⟩ := bind_eq_ok (alloc.vec.Vec.index_usize buf i1) _ _ h6
    exact (congrArg Prod.snd (Result.ok.injEq.mp h6b)).symm
  rw [hi3] at h9
  rw [ha] at h8
  rw [hback] at hbi
  exact ⟨((face, key), v), s, address, s1, i1, i2, h1, h2, h3, h4, h5, hbi.symm, h7, h8, h9, hbnd⟩

/-- Invert a successful `done` result of the scatter body. -/
theorem loop2_body_done {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {last : Usize} {i : Usize}
    {cursor : Array Usize 256#usize} {buf : alloc.vec.Vec Usize} {buf' : alloc.vec.Vec Usize}
    (hnlt : ¬ i < last)
    (hbody : table.partition.partition_chunk_loop2.body HomeInst home input last i cursor buf
      = ok (ControlFlow.done buf')) : buf' = buf := by
  unfold table.partition.partition_chunk_loop2.body at hbody
  rw [if_neg hnlt] at hbody
  exact (ControlFlow.done.inj (Result.ok.injEq.mp hbody)).symm

/-- The scatter loop's invariant: `cursor[s]` is the base offset plus the number of routed indexes of
    `[first, i)` for `s`, and `buf[ends[s] .. cursor[s])` is that routed run. -/
def scatterInv {K V H Faces : Type} (HomeInst : table.Home H K Faces) (home : H)
    (input : Slice ((Faces × K) × V)) (last first : Usize) (ends : Array Usize 257#usize)
    (i : Usize) (cursor : Array Usize 256#usize) (buf : alloc.vec.Vec Usize) : Prop :=
  first.val ≤ i.val ∧ i.val ≤ last.val ∧
  (∀ s : Usize, s.val < 256 →
    cursor.val[s.val]? = some (usizeOf (endsVal ends s
      + (routedMid HomeInst home input first i s).length))) ∧
  (∀ (s : Usize) (p : Nat), s.val < 256 → p < (routedMid HomeInst home input first i s).length →
    buf.val[endsVal ends s + p]? = ((routedMid HomeInst home input first i s)[p]?).map usizeOf)

/-- Setting index `m` of a list and taking `m + 1` appends the new element. -/
theorem take_set_succ {α : Type} {l : List α} {m : Nat} {x : α} (h : m < l.length) :
    (l.set m x).take (m + 1) = l.take m ++ [x] := by
  induction m generalizing l with
  | zero => cases l with
    | nil => simp at h
    | cons a t => simp
  | succ m ih =>
    cases l with
    | nil => simp at h
    | cons a t =>
      simp only [List.set_cons_succ, List.take_succ_cons, List.length_cons] at h ⊢
      rw [ih (by omega), List.cons_append]

/-! ### The scatter invariant step

The four facts the scatter pass consumes: the per-step load of `ends`, its monotonicity, the
routed-prefix property, and the disjointness of the per-shard regions.  They are stated over the
prefix-pass output (`hends`) and the count-pass output (`hcounts`), which `loop0_spec`/`loop1_spec`
supply. -/

/-- `ends[n]` read as a natural. -/
def endsAt (ends : Array Usize 257#usize) (n : Nat) : Nat :=
  ((ends.val[n]?).getD 0#usize).val

theorem endsVal_eq_endsAt (ends : Array Usize 257#usize) (s : Usize) :
    endsVal ends s = endsAt ends s.val := rfl

theorem endsAt_of_getElem? {ends : Array Usize 257#usize} {n x : Nat}
    (h : ends.val[n]? = some (usizeOf x)) (hx : x ≤ Usize.max) :
    endsAt ends n = x := by
  unfold endsAt
  rw [h]
  simp only [Option.getD_some]
  exact usizeOf_val x hx

theorem endsVal_of_getElem? {ends : Array Usize 257#usize} {s : Usize} {x : Nat}
    (h : ends.val[s.val]? = some (usizeOf x)) (hx : x ≤ Usize.max) :
    endsVal ends s = x :=
  endsAt_of_getElem? h hx

theorem endsAt_eq_prefixSum {counts : Array Usize 256#usize} {ends : Array Usize 257#usize}
    (hsum : prefixSum counts 256 ≤ Usize.max)
    (hends : ∀ n, n ≤ 256 → ends.val[n]? = some (usizeOf (prefixSum counts n)))
    {s : Usize} (hs : s.val ≤ 256) :
    endsVal ends s = prefixSum counts s.val :=
  endsAt_of_getElem? (hends s.val hs)
    (le_trans (prefixSum_mono counts (by omega)) hsum)

theorem endsVal_usizeOf {ends : Array Usize 257#usize} {n : Nat} (hn : n ≤ Usize.max) :
    endsVal ends (usizeOf n) = endsAt ends n := by
  unfold endsVal endsAt
  rw [usizeOf_val n hn]

/-- Writing the terminal `ends[256]` does not change `endsVal` at `s < 256`. -/
theorem endsVal_set_of_ne {ends : Array Usize 257#usize} {total : Usize} {s : Usize}
    (hs : s.val < 256) :
    endsVal (ends.set 256#usize total) s = endsVal ends s := by
  unfold endsVal
  rw [Array.set_val_eq, List.getElem?_set_ne (by simpa using (Ne.symm (by omega : s.val ≠ 256)))]

/-- The terminal `ends[256]` reads the written total. -/
theorem endsVal_set_self {ends : Array Usize 257#usize} {total : Usize} :
    endsVal (ends.set 256#usize total) 256#usize = total.val := by
  unfold endsVal
  rw [Array.set_val_eq,
    List.getElem?_set_self (by rw [Array.length_eq]; norm_num)]
  simp

theorem endsVal_repeat_zero (s : Usize) : endsVal (Array.repeat 257#usize 0#usize) s = 0 := by
  unfold endsVal
  rw [Array.repeat_val, List.getElem?_replicate]
  by_cases hs : s.val < 257 <;> simp [hs]

/-- **1 · Load invariant.** Consecutive `ends` differ by the routed load of the shard. -/
theorem endsVal_step {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {last first : Usize} {ends : Array Usize 257#usize}
    {counts : Array Usize 256#usize}
    (hlen : last.val ≤ input.val.length) (_hfl : first.val ≤ last.val)
    (hsum : prefixSum counts 256 ≤ Usize.max)
    (hends : ∀ n, n ≤ 256 → ends.val[n]? = some (usizeOf (prefixSum counts n)))
    (hcounts : ∀ s : Usize, s.val < 256 →
      counts.val[s.val]? = some (usizeOf ((routedRun HomeInst home input last s).length
        - (routedRun HomeInst home input first s).length))) :
    ∀ s : Usize, s.val < 256 →
      endsVal ends (usizeOf (s.val + 1)) - endsVal ends s
        = (routedMid HomeInst home input first last s).length := by
  intro s hs
  have hle : (routedRun HomeInst home input last s).length ≤ last.val := by
    rw [routedRun]
    exact (List.length_filter_le _ _).trans (by simp)
  have hLmax : (routedRun HomeInst home input last s).length
      - (routedRun HomeInst home input first s).length ≤ Usize.max :=
    le_trans (Nat.sub_le _ _) (le_trans hle (le_trans hlen input.property))
  have hsu : prefixSum counts s.val ≤ Usize.max :=
    le_trans (prefixSum_mono counts (by omega)) hsum
  have hsu1 : prefixSum counts (s.val + 1) ≤ Usize.max :=
    le_trans (prefixSum_mono counts (by omega)) hsum
  have h0 : endsVal ends s = prefixSum counts s.val :=
    endsAt_eq_prefixSum hsum hends (by omega)
  have h1 : endsVal ends (usizeOf (s.val + 1)) = prefixSum counts (s.val + 1) := by
    rw [endsVal_usizeOf (by have := le_usize_max_256; omega)]
    exact endsAt_of_getElem? (hends (s.val + 1) (by omega)) hsu1
  have hcntv : ((counts.val[s.val]?).getD 0#usize).val
      = (routedRun HomeInst home input last s).length
        - (routedRun HomeInst home input first s).length := by
    rw [hcounts s hs]
    simp only [Option.getD_some]
    exact usizeOf_val _ hLmax
  rw [h0, h1, prefixSum_succ, hcntv, routedMid, List.length_drop]
  omega

/-- **2 · Monotonicity.** `endsVal ends` is monotone on `[0, 256]`. -/
theorem endsVal_mono {counts : Array Usize 256#usize} {ends : Array Usize 257#usize}
    (hsum : prefixSum counts 256 ≤ Usize.max)
    (hends : ∀ n, n ≤ 256 → ends.val[n]? = some (usizeOf (prefixSum counts n))) :
    ∀ s t : Usize, s.val ≤ t.val → t.val ≤ 256 → endsVal ends s ≤ endsVal ends t := by
  intro s t hst ht
  rw [endsAt_eq_prefixSum hsum hends (by omega), endsAt_eq_prefixSum hsum hends ht]
  exact prefixSum_mono counts hst

/-- **3 · Prefix property.** The routed mid-run of a prefix is a prefix of the whole window's. -/
theorem routedMid_isPrefix {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {first i j s : Usize} (hij : i.val ≤ j.val) :
    routedMid HomeInst home input first i s <+: routedMid HomeInst home input first j s := by
  unfold routedMid
  exact List.IsPrefix.drop (List.IsPrefix.filter _ (range_isPrefix hij)) _

theorem routedMid_length_le {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {first i j s : Usize} (hij : i.val ≤ j.val) :
    (routedMid HomeInst home input first i s).length
      ≤ (routedMid HomeInst home input first j s).length :=
  List.IsPrefix.length_le (routedMid_isPrefix hij)

theorem routedMid_length {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {first i s : Usize} :
    (routedMid HomeInst home input first i s).length
      = (routedRun HomeInst home input i s).length
        - (routedRun HomeInst home input first s).length := by
  unfold routedMid
  rw [List.length_drop]

/-- The routed mid-run of the whole window `[first, last)` has the window's routed load length. -/
theorem endsVal_add_length {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {last first : Usize} {ends : Array Usize 257#usize}
    {counts : Array Usize 256#usize}
    (hlen : last.val ≤ input.val.length) (hfl : first.val ≤ last.val)
    (hsum : prefixSum counts 256 ≤ Usize.max)
    (hends : ∀ n, n ≤ 256 → ends.val[n]? = some (usizeOf (prefixSum counts n)))
    (hcounts : ∀ s : Usize, s.val < 256 →
      counts.val[s.val]? = some (usizeOf ((routedRun HomeInst home input last s).length
        - (routedRun HomeInst home input first s).length))) :
    ∀ s : Usize, s.val < 256 →
      endsVal ends s + (routedMid HomeInst home input first last s).length
        = endsVal ends (usizeOf (s.val + 1)) := by
  intro s hs
  have hsv : (usizeOf (s.val + 1)).val = s.val + 1 := usizeOf_val_of_le_256 (by omega)
  have hle : endsVal ends s ≤ endsVal ends (usizeOf (s.val + 1)) :=
    endsVal_mono hsum hends s (usizeOf (s.val + 1))
      (by rw [hsv]; omega) (by rw [hsv]; omega)
  have hstep := endsVal_step hlen hfl hsum hends hcounts s hs
  rw [Nat.sub_eq_iff_eq_add hle] at hstep
  omega

/-- **4 · Disjointness.** A write to the next free slot of shard `routed` (the index at `i` routes
    to it) lands strictly inside `routed`'s own final region, hence outside every other shard's
    current region. -/
theorem scatter_write_disjoint {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {last first : Usize} {ends : Array Usize 257#usize}
    {counts : Array Usize 256#usize}
    (hlen : last.val ≤ input.val.length) (hfl : first.val ≤ last.val)
    (hsum : prefixSum counts 256 ≤ Usize.max)
    (hends : ∀ n, n ≤ 256 → ends.val[n]? = some (usizeOf (prefixSum counts n)))
    (hcounts : ∀ s : Usize, s.val < 256 →
      counts.val[s.val]? = some (usizeOf ((routedRun HomeInst home input last s).length
        - (routedRun HomeInst home input first s).length)))
    {i routed : Usize} (hfi : first.val ≤ i.val) (hlt : i < last) (hrouted : routed.val < 256)
    (hroute : routedTo HomeInst home input routed i.val = true) :
    ∀ s : Usize, s.val < 256 → s ≠ routed →
      ∀ p : Nat, p < (routedMid HomeInst home input first i s).length →
        endsVal ends routed + (routedMid HomeInst home input first i routed).length
          ≠ endsVal ends s + p := by
  intro s hs hne p hp
  have hi1 : i.val + 1 ≤ last.val := by
    have := (UScalar.lt_equiv i last).mp hlt
    omega
  have hi' : i.val + 1 ≤ Usize.max := by
    have := input.property
    omega
  have hsv : (usizeOf (i.val + 1)).val = i.val + 1 := usizeOf_val (i.val + 1) hi'
  have hstep := routedMid_step (HomeInst := HomeInst) (home := home) (input := input)
    (first := first) (i := i) (s := routed) hfi hi'
  rw [if_pos hroute] at hstep
  have hstep_len : (routedMid HomeInst home input first (usizeOf (i.val + 1)) routed).length
      = (routedMid HomeInst home input first i routed).length + 1 := by
    rw [hstep, List.length_append, List.length_singleton]
  have hpref3 : (routedMid HomeInst home input first (usizeOf (i.val + 1)) routed).length
      ≤ (routedMid HomeInst home input first last routed).length :=
    routedMid_length_le (first := first) (i := usizeOf (i.val + 1)) (j := last) (s := routed)
      (by rw [hsv]; omega)
  have hlen_lt : (routedMid HomeInst home input first i routed).length
      < (routedMid HomeInst home input first last routed).length := by
    omega
  have hwrite : endsVal ends routed + (routedMid HomeInst home input first i routed).length
      < endsVal ends (usizeOf (routed.val + 1)) := by
    have hadd := endsVal_add_length hlen hfl hsum hends hcounts routed hrouted
    omega
  rcases Nat.lt_or_gt_of_ne (fun h => hne (UScalar.eq_of_val_eq h)) with hlt' | hgt'
  · -- `s < routed`: read is below `s`'s endpoint, which is below the write
    have hsp : p < (routedMid HomeInst home input first last s).length :=
      lt_of_lt_of_le hp
        (routedMid_length_le (first := first) (i := i) (j := last) (s := s) (by omega))
    have hsvs : (usizeOf (s.val + 1)).val = s.val + 1 := usizeOf_val_of_le_256 (by omega)
    have hread_lt : endsVal ends s + p < endsVal ends (usizeOf (s.val + 1)) := by
      have hadd := endsVal_add_length hlen hfl hsum hends hcounts s hs
      omega
    have hle : endsVal ends (usizeOf (s.val + 1)) ≤ endsVal ends routed :=
      endsVal_mono hsum hends (usizeOf (s.val + 1)) routed
        (by rw [hsvs]; omega) (by omega)
    omega
  · -- `routed < s`: the write is below `routed`'s endpoint, which is below the read
    have hsvr : (usizeOf (routed.val + 1)).val = routed.val + 1 :=
      usizeOf_val_of_le_256 (by omega)
    have hle : endsVal ends (usizeOf (routed.val + 1)) ≤ endsVal ends s :=
      endsVal_mono hsum hends (usizeOf (routed.val + 1)) s
        (by rw [hsvr]; omega) (by omega)
    omega

/-! ### The scatter step and the loop spec -/

theorem getElem?_append_self {α : Type} (l : List α) (x : α) :
    (l ++ [x])[l.length]? = some x := by
  rw [List.getElem?_append, if_neg (lt_irrefl _)]
  simp

theorem getElem?_append_of_lt {α : Type} (l : List α) (x : α) {p : Nat}
    (h : p < l.length) : (l ++ [x])[p]? = l[p]? := by
  rw [List.getElem?_append, if_pos h]

/-- Invert a successful step of the scatter body with `i < last` into a `cont`. -/
theorem loop2_body_out {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {last : Usize} {i : Usize}
    {cursor : Array Usize 256#usize} {buf : alloc.vec.Vec Usize}
    {out : ControlFlow (Usize × Array Usize 256#usize × alloc.vec.Vec Usize)
      (alloc.vec.Vec Usize)}
    (hlt : i < last)
    (hbody : table.partition.partition_chunk_loop2.body HomeInst home input last i cursor buf
      = ok out) :
    ∃ i3 a buf1, out = ControlFlow.cont (i3, a, buf1) := by
  unfold table.partition.partition_chunk_loop2.body at hbody
  rw [if_pos hlt] at hbody
  simp only [Bind.bind] at hbody
  obtain ⟨p, _, hbody⟩ := bind_eq_ok (Slice.index_usize input i) _ _ hbody
  obtain ⟨⟨face, key⟩, v⟩ := p
  obtain ⟨σ, _, hbody⟩ := bind_eq_ok
    (HomeInst.coordaddressFaceHomeInst.FaceCoordInst.FaceDomainInst.sigma face) _ _ hbody
  obtain ⟨address, _, hbody⟩ := bind_eq_ok (HomeInst.address home σ key) _ _ hbody
  obtain ⟨s1, _, hbody⟩ := bind_eq_ok (table.shard address) _ _ hbody
  obtain ⟨i1, _, hbody⟩ := bind_eq_ok (cursor.index_usize s1) _ _ hbody
  obtain ⟨q, _, hbody⟩ := bind_eq_ok (alloc.vec.Vec.index_mut
    (core.slice.index.SliceIndexUsizeSlice Std.Usize) buf i1) _ _ hbody
  obtain ⟨o, back⟩ := q
  obtain ⟨i2, _, hbody⟩ := bind_eq_ok (i1 + 1#usize) _ _ hbody
  obtain ⟨a', _, hbody⟩ := bind_eq_ok (cursor.update s1 i2) _ _ hbody
  obtain ⟨i3', _, hdone⟩ := bind_eq_ok (i + 1#usize) _ _ hbody
  exact ⟨i3', a', back i, (Result.ok.injEq.mp hdone).symm⟩

/-- One successful `cont` step preserves the scatter invariant. -/
theorem scatterInv_step {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {last first : Usize} {ends : Array Usize 257#usize}
    {counts : Array Usize 256#usize}
    (hlen : last.val ≤ input.val.length) (hfl : first.val ≤ last.val)
    (hsum : prefixSum counts 256 ≤ Usize.max)
    (hends : ∀ n, n ≤ 256 → ends.val[n]? = some (usizeOf (prefixSum counts n)))
    (hcounts : ∀ s : Usize, s.val < 256 →
      counts.val[s.val]? = some (usizeOf ((routedRun HomeInst home input last s).length
        - (routedRun HomeInst home input first s).length)))
    {i : Usize} {cursor : Array Usize 256#usize} {buf : alloc.vec.Vec Usize}
    {i3 : Usize} {a : Array Usize 256#usize} {buf1 : alloc.vec.Vec Usize} (hlt : i < last)
    (hinv : scatterInv HomeInst home input last first ends i cursor buf)
    (hbody : table.partition.partition_chunk_loop2.body HomeInst home input last i cursor buf
      = ok (ControlFlow.cont (i3, a, buf1))) :
    scatterInv HomeInst home input last first ends i3 a buf1 := by
  obtain ⟨⟨⟨face, key⟩, v⟩, σ, address, s1, i1, i2,
    h1, h2, h3, h4, h5, hbuf1, h7, h8, h9, hbnd⟩ :=
    loop2_body_cont (HomeInst := HomeInst) (home := home) (input := input) (last := last)
      (i := i) (cursor := cursor) (buf := buf) (i3 := i3) (a := a) (buf1 := buf1) hlt hbody
  obtain ⟨hfi, hile, hcur, hbuf⟩ := hinv
  have hltN : i.val < last.val := (UScalar.lt_equiv i last).mp hlt
  have himax : i.val ≤ Usize.max := by
    have := input.property
    omega
  have hltmax : i.val + 1 ≤ Usize.max := by
    have := input.property
    omega
  have hbound : i.val < input.val.length := by omega
  have hface : ((face, key), v) = input.val[i.val] := by
    rw [slice_index_ok input i hbound] at h1
    exact (Result.ok.injEq.mp h1).symm
  have hps : producerShard HomeInst home input i = ok s1 :=
    producerShard_eq_ok hbound hface h2 h3 h4
  have hrt : ∀ s' : Usize, routedTo HomeInst home input s' i.val = true ↔ s'.val = s1.val :=
    fun s' => producerShard_routedTo_iff (usizeOf_eq i himax) hps
  have hroute : routedTo HomeInst home input s1 i.val = true := (hrt s1).2 rfl
  have hs1lt : s1.val < 256 := by
    obtain ⟨r, hr, hb⟩ := WP.spec_imp_exists (shard_lt_256 address)
    rw [h4] at hr
    have hrr : r = s1 := (Result.ok.injEq.mp hr).symm
    rwa [hrr] at hb
  have hi3 : i3 = usizeOf (i.val + 1) := add_one_eq hltmax h9
  have hi3v : i3.val = i.val + 1 := add_one_val hltmax h9
  have hi1 : cursor.val[s1.val]? = some i1 := index_usize_eq cursor s1 i1 h5
  have ha : a = cursor.set s1 i2 := update_eq_set cursor s1 i2 hs1lt a h8
  have hbufmax : buf.val.length ≤ Usize.max := by
    have := Slice.length_ineq buf.slice
    simpa [alloc.vec.Vec.val] using this
  have hi2v : i2.val = i1.val + 1 := add_one_val (by omega) h7
  have hi1v : i1.val = endsVal ends s1
      + (routedMid HomeInst home input first i s1).length := by
    have hc := hcur s1 hs1lt
    rw [hi1] at hc
    have heq : i1 = usizeOf (endsVal ends s1
        + (routedMid HomeInst home input first i s1).length) := Option.some.inj hc
    have hXmax : endsVal ends s1
        + (routedMid HomeInst home input first i s1).length ≤ Usize.max := by
      have hadd := endsVal_add_length hlen hfl hsum hends hcounts s1 hs1lt
      have hmle : (routedMid HomeInst home input first i s1).length
          ≤ (routedMid HomeInst home input first last s1).length :=
        routedMid_length_le (first := first) (i := i) (j := last) (s := s1) (by omega)
      have hmono : endsVal ends (usizeOf (s1.val + 1)) ≤ endsVal ends (usizeOf 256) :=
        endsVal_mono hsum hends (usizeOf (s1.val + 1)) (usizeOf 256)
          (by rw [usizeOf_val_of_le_256 (by omega), usizeOf_val_of_le_256 (le_refl 256)]; omega)
          (by rw [usizeOf_val_of_le_256 (le_refl 256)])
      have h256 : endsVal ends (usizeOf 256) = prefixSum counts 256 := by
        rw [endsVal_usizeOf (by have := le_usize_max_256; omega)]
        exact endsAt_of_getElem? (hends 256 (le_refl 256)) hsum
      calc endsVal ends s1 + (routedMid HomeInst home input first i s1).length
          ≤ endsVal ends s1 + (routedMid HomeInst home input first last s1).length := by omega
        _ = endsVal ends (usizeOf (s1.val + 1)) := hadd
        _ ≤ endsVal ends (usizeOf 256) := hmono
        _ = prefixSum counts 256 := h256
        _ ≤ Usize.max := hsum
    rw [heq, usizeOf_val (endsVal ends s1
      + (routedMid HomeInst home input first i s1).length) hXmax]
  have hstep1 : routedMid HomeInst home input first i3 s1
      = routedMid HomeInst home input first i s1 ++ [i.val] := by
    rw [hi3, routedMid_step (HomeInst := HomeInst) (home := home) (input := input)
      (first := first) (i := i) (s := s1) hfi hltmax, if_pos hroute]
  have hstep_ne : ∀ s : Usize, s.val < 256 → s ≠ s1 →
      routedMid HomeInst home input first i3 s = routedMid HomeInst home input first i s := by
    intro s hs hne
    rw [hi3, routedMid_step (HomeInst := HomeInst) (home := home) (input := input)
      (first := first) (i := i) (s := s) hfi hltmax]
    have hfalse : routedTo HomeInst home input s i.val = false := by
      cases h : routedTo HomeInst home input s i.val with
      | true => exact absurd (UScalar.eq_of_val_eq ((hrt s).1 h)) hne
      | false => rfl
    rw [if_neg (by simpa using hfalse)]
  refine ⟨by omega, by omega, ?_, ?_⟩
  · intro s hs
    by_cases hss : s.val = s1.val
    · have hsv : s = s1 := UScalar.eq_of_val_eq hss
      rw [hsv, ha, Array.set_val_eq,
        List.getElem?_set_self (by rw [Array.length_eq]; exact hs1lt)]
      rw [Option.some.injEq]
      apply UScalar.eq_of_val_eq
      have hm1 : (routedMid HomeInst home input first i3 s1).length
          = (routedMid HomeInst home input first i s1).length + 1 := by
        rw [hstep1, List.length_append, List.length_singleton]
      have hXeq : endsVal ends s1 + (routedMid HomeInst home input first i3 s1).length
          = i1.val + 1 := by omega
      have hXmax : endsVal ends s1
          + (routedMid HomeInst home input first i3 s1).length ≤ Usize.max := by
        rw [hXeq]; omega
      rw [usizeOf_val _ hXmax]
      omega
    · have hsne : s ≠ s1 := fun h => hss (by rw [h])
      rw [ha, Array.set_val_eq, List.getElem?_set_ne (Ne.symm hss), hstep_ne s hs hsne]
      exact hcur s hs
  · intro s p hs hp
    by_cases hss : s.val = s1.val
    · have hsv : s = s1 := UScalar.eq_of_val_eq hss
      rw [hsv] at hp ⊢
      have hp' : p < (routedMid HomeInst home input first i s1).length + 1 := by
        rw [hstep1, List.length_append, List.length_singleton] at hp
        exact hp
      rw [hbuf1, alloc.vec.Vec.set_val_eq]
      by_cases hpm : p = (routedMid HomeInst home input first i s1).length
      · subst hpm
        have hidx : endsVal ends s1 + (routedMid HomeInst home input first i s1).length = i1.val :=
          hi1v.symm
        rw [hidx, List.getElem?_set_self hbnd, hstep1, getElem?_append_self]
        rw [Option.map_some]
        exact congrArg some (usizeOf_eq i himax).symm
      · have hplt : p < (routedMid HomeInst home input first i s1).length :=
          lt_of_le_of_ne (Nat.le_of_lt_succ hp') hpm
        have hidx : endsVal ends s1 + p ≠ i1.val := by rw [hi1v]; omega
        rw [List.getElem?_set_ne (Ne.symm hidx), hstep1, getElem?_append_of_lt _ _ hplt]
        exact hbuf s1 p hs1lt hplt
    · have hsne : s ≠ s1 := fun h => hss (by rw [h])
      have hps : p < (routedMid HomeInst home input first i s).length := by
        rw [hstep_ne s hs hsne] at hp; exact hp
      have hidx : endsVal ends s1 + (routedMid HomeInst home input first i s1).length
          ≠ endsVal ends s + p :=
        scatter_write_disjoint hlen hfl hsum hends hcounts hfi hlt hs1lt hroute s hs hsne p hps
      rw [hbuf1, alloc.vec.Vec.set_val_eq, List.getElem?_set_ne (by rw [hi1v]; exact hidx),
        hstep_ne s hs hsne]
      exact hbuf s p hs hps

/-- **The scatter pass.** A successful `partition_chunk_loop2` from the entry cursor
    `cursor[s] = ends[s]` writes, for every shard, the routed indexes of the whole window into
    `buf[ends[s] .. ends[s] + |routedMid|)`. -/
theorem loop2_spec {K V H Faces : Type} (HomeInst : table.Home H K Faces) (home : H)
    (input : Slice ((Faces × K) × V)) (last first : Usize) {ends : Array Usize 257#usize}
    {counts : Array Usize 256#usize}
    (hlen : last.val ≤ input.val.length) (hfl : first.val ≤ last.val)
    (hsum : prefixSum counts 256 ≤ Usize.max)
    (hends : ∀ n, n ≤ 256 → ends.val[n]? = some (usizeOf (prefixSum counts n)))
    (hcounts : ∀ s : Usize, s.val < 256 →
      counts.val[s.val]? = some (usizeOf ((routedRun HomeInst home input last s).length
        - (routedRun HomeInst home input first s).length))) :
    ∀ cursor buf buf',
      (∀ s : Usize, s.val < 256 → cursor.val[s.val]? = some (usizeOf (endsVal ends s))) →
      table.partition.partition_chunk_loop2 HomeInst home input last first cursor buf = ok buf' →
      ∀ s : Usize, s.val < 256 →
        ∀ p : Nat, p < (routedMid HomeInst home input first last s).length →
          buf'.val[endsVal ends s + p]? =
            ((routedMid HomeInst home input first last s)[p]?).map usizeOf := by
  intro cursor buf buf' hcursor hok
  unfold table.partition.partition_chunk_loop2 at hok
  refine loop_ok_spec
    (measure := fun q : Usize × Array Usize 256#usize × alloc.vec.Vec Usize =>
      last.val - q.1.val)
    (inv := fun q => scatterInv HomeInst home input last first ends q.1 q.2.1 q.2.2)
    (post := fun buf' => ∀ s : Usize, s.val < 256 →
      ∀ p : Nat, p < (routedMid HomeInst home input first last s).length →
        buf'.val[endsVal ends s + p]? =
          ((routedMid HomeInst home input first last s)[p]?).map usizeOf)
    (body := fun q => table.partition.partition_chunk_loop2.body HomeInst home input last
      q.1 q.2.1 q.2.2)
    ?_ (first, cursor, buf) ?_ buf' hok
  · rintro ⟨i, cursor, buf⟩ hinv
    dsimp only at hinv ⊢
    have hinv0 := hinv
    obtain ⟨hfi, hile, hcur, hbuf⟩ := hinv
    by_cases hlt : i < last
    · refine ⟨?_, ?_⟩
      · intro y hbody
        obtain ⟨i3, a, buf1, hout⟩ := loop2_body_out hlt hbody
        exact absurd hout (by intro h; cases h)
      · intro q' hbody
        obtain ⟨i3, a, buf1⟩ := q'
        dsimp only at hbody ⊢
        obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, h9, _⟩ :=
          loop2_body_cont (HomeInst := HomeInst) (home := home) (input := input)
            (last := last) (i := i) (cursor := cursor) (buf := buf)
            (i3 := i3) (a := a) (buf1 := buf1) hlt hbody
        have hi3v : i3.val = i.val + 1 := add_one_val (by
          have hl := (UScalar.lt_equiv i last).mp hlt
          have := input.property
          omega) h9
        have hlN : i.val < last.val := (UScalar.lt_equiv i last).mp hlt
        refine ⟨scatterInv_step hlen hfl hsum hends hcounts hlt hinv0 hbody, ?_⟩
        omega
    · refine ⟨?_, ?_⟩
      · intro y hbody
        have hy := loop2_body_done hlt hbody
        have hieq : i = last := by
          apply UScalar.eq_of_val_eq
          have h := (UScalar.lt_equiv i last).not.mp hlt
          omega
        rw [hieq] at hbuf
        rw [hy]
        exact fun s hs p hp => hbuf s p hs hp
      · intro q' hbody
        unfold table.partition.partition_chunk_loop2.body at hbody
        rw [if_neg hlt] at hbody
        exact absurd (Result.ok.injEq.mp hbody) (by intro h; cases h)
  · refine ⟨le_refl _, hfl, ?_, ?_⟩
    · intro s hs
      rw [routedMid_self, List.length_nil, Nat.add_zero]
      exact hcursor s hs
    · intro s p hs hp
      rw [routedMid_self, List.length_nil] at hp
      exact absurd hp (by omega)

/-- The scatter loop only `set`s the buffer, so its length is the entry length. -/
theorem loop2_length {K V H Faces : Type} {HomeInst : table.Home H K Faces} {home : H}
    {input : Slice ((Faces × K) × V)} {last i : Usize}
    {cursor : Array Usize 256#usize} {buf buf' : alloc.vec.Vec Usize}
    (hok : table.partition.partition_chunk_loop2 HomeInst home input last i cursor buf
      = ok buf') :
    buf'.val.length = buf.val.length := by
  unfold table.partition.partition_chunk_loop2 at hok
  refine loop_ok_spec
    (measure := fun p : Usize × Array Usize 256#usize × alloc.vec.Vec Usize => last.val - p.1.val)
    (inv := fun p => p.2.2.val.length = buf.val.length)
    (post := fun buf' => buf'.val.length = buf.val.length)
    (body := fun p => table.partition.partition_chunk_loop2.body HomeInst home input last
      p.1 p.2.1 p.2.2)
    ?_ (i, cursor, buf) ?_ buf' hok
  · rintro ⟨i, cursor, buf0⟩ hinv
    dsimp only at hinv ⊢
    by_cases hlt : i < last
    · refine ⟨?_, ?_⟩
      · intro y hbody
        obtain ⟨i3, a, buf1, hout⟩ := loop2_body_out hlt hbody
        exact absurd hout (by intro h; cases h)
      · intro q' hbody
        obtain ⟨i3, a, buf1⟩ := q'
        dsimp only at hbody ⊢
        obtain ⟨_, _, _, _, _, _, _, _, _, _, _, hbuf1, _, _, h9, _⟩ :=
          loop2_body_cont (HomeInst := HomeInst) (home := home) (input := input)
            (last := last) (i := i) (cursor := cursor) (buf := buf0) (i3 := i3) (a := a)
            (buf1 := buf1) hlt hbody
        refine ⟨?_, ?_⟩
        · rw [hbuf1, alloc.vec.Vec.set_val_eq, List.length_set]
          exact hinv
        · rw [add_one_val (by simpa using add_le_of_ok h9) h9]
          have := (UScalar.lt_equiv i last).mp hlt
          omega
    · refine ⟨?_, ?_⟩
      · intro y hbody
        have hy := loop2_body_done hlt hbody
        rw [hy]
        exact hinv
      · intro q' hbody
        unfold table.partition.partition_chunk_loop2.body at hbody
        rw [if_neg hlt] at hbody
        exact absurd (Result.ok.injEq.mp hbody) (by intro h; cases h)
  · rfl

/-- The prefix pass's `Usize`-indexed endpoint equation at every `n < 256`. -/
theorem hends_of_usize {counts : Array Usize 256#usize} {ends : Array Usize 257#usize}
    (h : ∀ t : Usize, t.val < 256 → ends.val[t.val]? = some (usizeOf (prefixSum counts t.val))) :
    ∀ n, n < 256 → ends.val[n]? = some (usizeOf (prefixSum counts n)) := by
  intro n hn
  have hn256 : n ≤ 256 := by omega
  have h' := h (usizeOf n) (by rw [usizeOf_val_of_le_256 hn256]; omega)
  rwa [usizeOf_val_of_le_256 hn256] at h'

/-- **The run equation.** For a successful `Parts::of`, every shard's buffer region is exactly the
    increasing list of input indexes the producer routes to that shard. -/
theorem parts_of_run_eq {K V H Faces : Type} (HomeInst : table.Home H K Faces) (home : H)
    (input : Slice ((Faces × K) × V)) (workers : Usize) {p : table.partition.Parts}
    (hok : table.partition.Parts.of HomeInst home input workers = ok p) :
    ∀ s : Usize, s.val < 256 →
      ∀ j : Nat, j < (routedRun HomeInst home input (Slice.len input) s).length →
        p.buf.val[endsVal p.ends s + j]? =
          ((routedRun HomeInst home input (Slice.len input) s)[j]?).map usizeOf := by
  unfold table.partition.Parts.of at hok
  dsimp only at hok
  by_cases hempty : Slice.len input = 0#usize
  · intro s hs j hj
    rw [hempty, routedRun_nil] at hj
    simp at hj
  · rw [if_neg hempty] at hok
    simp only [Bind.bind] at hok
    obtain ⟨bc, hpc, hok⟩ := bind_eq_ok
      (table.partition.partition_chunk HomeInst home input 0#usize (Slice.len input)) _ _ hok
    obtain ⟨buf0, ends0⟩ := bc
    have hp : p = { buf := buf0, ends := ends0 } := (Result.ok.injEq.mp hok).symm
    subst hp
    unfold table.partition.partition_chunk at hpc
    simp only [Bind.bind] at hpc
    obtain ⟨counts1, hloop0, hpc⟩ := bind_eq_ok
      (table.partition.partition_chunk_loop0 HomeInst home input (Slice.len input)
        (Array.repeat 256#usize 0#usize) 0#usize) _ _ hpc
    obtain ⟨⟨ends1, cursor1, total⟩, hloop1, hpc⟩ := bind_eq_ok
      (table.partition.partition_chunk_loop1 counts1
        (Array.repeat 257#usize 0#usize) (Array.repeat 256#usize 0#usize) 0#usize 0#usize) _ _ hpc
    obtain ⟨shards, hSH, hpc⟩ := bind_eq_ok table.partition.SHARDS _ _ hpc
    obtain ⟨ends2, hupd, hpc⟩ := bind_eq_ok (ends1.update shards total) _ _ hpc
    obtain ⟨i1v, _hsub, hpc⟩ := bind_eq_ok (Slice.len input - 0#usize) _ _ hpc
    obtain ⟨bufv, _hfe, hpc⟩ := bind_eq_ok
      (alloc.vec.from_elem core.clone.CloneUsize 0#usize i1v) _ _ hpc
    obtain ⟨buf1, hloop2, hdone⟩ := bind_eq_ok
      (table.partition.partition_chunk_loop2 HomeInst home input (Slice.len input) 0#usize
        cursor1 bufv) _ _ hpc
    have hpair : (buf1, ends2) = (buf0, ends0) := Result.ok.injEq.mp hdone
    have hbufeq : buf1 = buf0 := congrArg Prod.fst hpair
    have hendseq : ends2 = ends0 := congrArg Prod.snd hpair
    have hSH256 : shards = 256#usize := Result.ok.injEq.mp (hSH.symm.trans SHARDS_eq)
    rw [hSH256] at hupd
    have hends2eq : ends2 = ends1.set 256#usize total :=
      update_eq_set ends1 256#usize total (by scalar_tac) ends2 hupd
    have hlen : (Slice.len input).val ≤ input.val.length := le_of_eq (Slice.len_val input)
    have hcounts : ∀ s : Usize, s.val < 256 →
        counts1.val[s.val]? = some (usizeOf ((routedRun HomeInst home input (Slice.len input) s).length
          - (routedRun HomeInst home input 0#usize s).length)) :=
      loop0_spec HomeInst home input (Slice.len input) 0#usize hlen (by simp) counts1 hloop0
    have hsum : prefixSum counts1 256 ≤ Usize.max := prefixSum_le_max_of_loop1 counts1 hloop1
    obtain ⟨hends1, hcursor1, htotal⟩ := loop1_spec counts1 hsum ends1 cursor1 total hloop1
    have hends1' : ∀ n, n < 256 → ends1.val[n]? = some (usizeOf (prefixSum counts1 n)) :=
      hends_of_usize hends1
    have hends2 : ∀ n, n ≤ 256 → ends2.val[n]? = some (usizeOf (prefixSum counts1 n)) := by
      intro n hn
      rw [hends2eq, Array.set_val_eq]
      rcases Nat.lt_or_eq_of_le hn with hlt | heq
      · rw [List.getElem?_set_ne (by simpa using (show 256 ≠ n by omega))]
        exact hends1' n hlt
      · subst heq
        change (ends1.val.set 256 total)[256]? = some (usizeOf (prefixSum counts1 256))
        rw [List.getElem?_set_self (by rw [Array.length_eq]; norm_num)]
        simp only [Option.some.injEq]
        exact UScalar.eq_of_val_eq (by rw [usizeOf_val (prefixSum counts1 256) hsum, htotal])
    have hcursor2 : ∀ s : Usize, s.val < 256 →
        cursor1.val[s.val]? = some (usizeOf (endsVal ends2 s)) := by
      intro s hs
      rw [endsAt_eq_prefixSum hsum hends2 (by omega)]
      exact hcursor1 s hs
    have hrun2 := loop2_spec HomeInst home input (Slice.len input) 0#usize hlen (by simp)
      hsum hends2 hcounts cursor1 bufv buf1 hcursor2 hloop2
    intro s hs j hj
    rw [← hendseq, ← hbufeq]
    have hj' : j < (routedMid HomeInst home input 0#usize (Slice.len input) s).length := by
      rwa [routedMid_zero]
    have h := hrun2 s hs j hj'
    rwa [routedMid_zero] at h

/-- **The four `Parts::is_intact` obligations.** For a successful `Parts::of`: the endpoints tile
    (`0`, `input.len`, per-shard monotone and in range), each shard region is strictly increasing,
    the endpoints cover the buffer (`ends[256] = buf.len`), and every entry is a valid in-range index
    that re-routes to its shard. -/
theorem parts_of_shard_run {K V H Faces : Type} (HomeInst : table.Home H K Faces) (home : H)
    (input : Slice ((Faces × K) × V)) (workers : Usize) {p : table.partition.Parts}
    (hok : table.partition.Parts.of HomeInst home input workers = ok p) :
    (endsVal p.ends 0#usize = 0 ∧
     endsVal p.ends 256#usize = (Slice.len input).val ∧
     p.buf.val.length = (Slice.len input).val ∧
     (∀ s : Usize, s.val < 256 →
        endsVal p.ends s ≤ endsVal p.ends (usizeOf (s.val + 1)) ∧
        endsVal p.ends (usizeOf (s.val + 1)) ≤ p.buf.val.length)) ∧
    (∀ (s : Usize) (j j' : Nat), s.val < 256 →
        j < j' → j' < (routedRun HomeInst home input (Slice.len input) s).length →
        ∃ (a b : Usize), p.buf.val[endsVal p.ends s + j]? = some a ∧
          p.buf.val[endsVal p.ends s + j']? = some b ∧ a < b) ∧
    (∀ (s : Usize) (j : Nat), s.val < 256 →
        j < (routedRun HomeInst home input (Slice.len input) s).length →
        ∃ v : Usize, p.buf.val[endsVal p.ends s + j]? = some v ∧
          v.val < (Slice.len input).val ∧ routedTo HomeInst home input s v.val = true) ∧
    (endsVal p.ends 256#usize = p.buf.val.length) := by
  have hok0 := hok
  unfold table.partition.Parts.of at hok
  dsimp only at hok
  have hrun := parts_of_run_eq HomeInst home input workers hok0
  by_cases hempty : Slice.len input = 0#usize
  · rw [if_pos hempty] at hok
    have hp : p = table.partition.Parts.mk (alloc.vec.Vec.new Std.Usize)
        (Array.repeat 257#usize 0#usize) := (Result.ok.injEq.mp hok).symm
    subst hp
    have hL0 : (Slice.len input).val = 0 := by rw [hempty]; rfl
    have hrr : ∀ s : Usize, routedRun HomeInst home input (Slice.len input) s = [] := by
      intro s; rw [hempty, routedRun_nil]
    have hbufz : (alloc.vec.Vec.new Std.Usize).val.length = 0 := by
      simp [alloc.vec.Vec.new, alloc.vec.Vec.from_val]
    refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_, ?_, ?_⟩
    · rw [endsVal_repeat_zero]
    · rw [endsVal_repeat_zero, hL0]
    · rw [hL0]; exact hbufz
    · intro s hs
      rw [endsVal_repeat_zero, endsVal_repeat_zero]
      exact ⟨le_refl _, by rw [hbufz]⟩
    · intro s j j' hs hj hj'
      rw [hrr s] at hj'
      simp at hj'
    · intro s j hs hj
      rw [hrr s] at hj
      simp at hj
    · rw [endsVal_repeat_zero, hbufz]
  · rw [if_neg hempty] at hok
    simp only [Bind.bind] at hok
    obtain ⟨bc, hpc, hok⟩ := bind_eq_ok
      (table.partition.partition_chunk HomeInst home input 0#usize (Slice.len input)) _ _ hok
    obtain ⟨buf0, ends0⟩ := bc
    have hp : p = { buf := buf0, ends := ends0 } := (Result.ok.injEq.mp hok).symm
    subst hp
    unfold table.partition.partition_chunk at hpc
    simp only [Bind.bind] at hpc
    obtain ⟨counts1, hloop0, hpc⟩ := bind_eq_ok
      (table.partition.partition_chunk_loop0 HomeInst home input (Slice.len input)
        (Array.repeat 256#usize 0#usize) 0#usize) _ _ hpc
    obtain ⟨⟨ends1, cursor1, total⟩, hloop1, hpc⟩ := bind_eq_ok
      (table.partition.partition_chunk_loop1 counts1
        (Array.repeat 257#usize 0#usize) (Array.repeat 256#usize 0#usize) 0#usize 0#usize) _ _ hpc
    obtain ⟨shards, hSH, hpc⟩ := bind_eq_ok table.partition.SHARDS _ _ hpc
    obtain ⟨ends2, hupd, hpc⟩ := bind_eq_ok (ends1.update shards total) _ _ hpc
    obtain ⟨i1v, hsub, hpc⟩ := bind_eq_ok (Slice.len input - 0#usize) _ _ hpc
    obtain ⟨bufv, hfe, hpc⟩ := bind_eq_ok
      (alloc.vec.from_elem core.clone.CloneUsize 0#usize i1v) _ _ hpc
    obtain ⟨buf1, hloop2, hdone⟩ := bind_eq_ok
      (table.partition.partition_chunk_loop2 HomeInst home input (Slice.len input) 0#usize
        cursor1 bufv) _ _ hpc
    have hpair : (buf1, ends2) = (buf0, ends0) := Result.ok.injEq.mp hdone
    have hbufeq : buf1 = buf0 := congrArg Prod.fst hpair
    have hendseq : ends2 = ends0 := congrArg Prod.snd hpair
    have hSH256 : shards = 256#usize := Result.ok.injEq.mp (hSH.symm.trans SHARDS_eq)
    rw [hSH256] at hupd
    have hends2eq : ends2 = ends1.set 256#usize total :=
      update_eq_set ends1 256#usize total (by scalar_tac) ends2 hupd
    have hlen : (Slice.len input).val ≤ input.val.length := le_of_eq (Slice.len_val input)
    have hsum : prefixSum counts1 256 ≤ Usize.max := prefixSum_le_max_of_loop1 counts1 hloop1
    obtain ⟨hends1, _hcursor1, htotal_eq⟩ := loop1_spec counts1 hsum ends1 cursor1 total hloop1
    have hsum_eq : prefixSum counts1 256 = (Slice.len input).val := by
      have h := loop0_spec_sum HomeInst home input (Slice.len input) 0#usize hlen (by simp)
        counts1 hloop0
      simpa using h
    have htotal : total.val = (Slice.len input).val := htotal_eq.trans hsum_eq
    have hbufv : bufv.val.length = i1v.val := by
      obtain ⟨v, hv, hvl, _⟩ := WP.spec_imp_exists
        (alloc.vec.from_elem_spec core.clone.CloneUsize 0#usize i1v rfl)
      have hbv : bufv = v := Result.ok.injEq.mp (hfe.symm.trans hv)
      rw [hbv, hvl, List.length_replicate]
    have hi1v : i1v.val = (Slice.len input).val := by
      have h := sub_val_of_ok hsub
      simpa using h
    have hbuf_len : buf0.val.length = (Slice.len input).val := by
      rw [← hbufeq, loop2_length hloop2, hbufv, hi1v]
    have hends2 : ∀ n, n ≤ 256 → ends2.val[n]? = some (usizeOf (prefixSum counts1 n)) := by
      intro n hn
      rw [hends2eq, Array.set_val_eq]
      rcases Nat.lt_or_eq_of_le hn with hlt | heq
      · rw [List.getElem?_set_ne (by simpa using (show 256 ≠ n by omega))]
        exact hends_of_usize hends1 n hlt
      · subst heq
        change (ends1.val.set 256 total)[256]? = some (usizeOf (prefixSum counts1 256))
        rw [List.getElem?_set_self (by rw [Array.length_eq]; norm_num)]
        simp only [Option.some.injEq]
        exact UScalar.eq_of_val_eq (by rw [usizeOf_val (prefixSum counts1 256) hsum, htotal_eq])
    have hends2_le : ∀ s : Usize, s.val ≤ 256 → endsVal ends2 s = prefixSum counts1 s.val :=
      fun s hs => endsAt_eq_prefixSum hsum hends2 hs
    have h0 : endsVal ends0 0#usize = 0 := by
      rw [← hendseq]
      exact (hends2_le 0#usize (by norm_num)).trans (prefixSum_zero counts1)
    have h256 : endsVal ends0 256#usize = (Slice.len input).val := by
      rw [← hendseq, hends2eq, endsVal_set_self]
      exact htotal
    have hmono : ∀ s : Usize, s.val < 256 →
        endsVal ends0 s ≤ endsVal ends0 (usizeOf (s.val + 1)) ∧
        endsVal ends0 (usizeOf (s.val + 1)) ≤ buf0.val.length := by
      intro s hs
      have hsv : (usizeOf (s.val + 1)).val = s.val + 1 := usizeOf_val_of_le_256 (by omega)
      have hsu : (usizeOf (s.val + 1)).val ≤ 256 := by rw [hsv]; omega
      have hupper : endsVal ends0 (usizeOf (s.val + 1)) ≤ buf0.val.length := by
        rw [← hendseq, hends2_le (usizeOf (s.val + 1)) hsu, hsv]
        calc prefixSum counts1 (s.val + 1) ≤ prefixSum counts1 256 := prefixSum_mono counts1 (by omega)
          _ = (Slice.len input).val := hsum_eq
          _ = buf0.val.length := hbuf_len.symm
      have hlower : endsVal ends0 s ≤ endsVal ends0 (usizeOf (s.val + 1)) := by
        rw [← hendseq, hends2_le s (by omega), hends2_le (usizeOf (s.val + 1)) hsu]
        exact prefixSum_mono counts1 (by omega)
      exact ⟨hlower, hupper⟩
    have haj_lt : ∀ (s : Usize) (j aj : Nat), s.val < 256 →
        (routedRun HomeInst home input (Slice.len input) s)[j]? = some aj →
        aj < (Slice.len input).val := by
      intro s j aj _hs haj
      obtain ⟨hjlen, hajeq⟩ := List.getElem?_eq_some_iff.mp haj
      have hm : aj ∈ routedRun HomeInst home input (Slice.len input) s :=
        List.mem_iff_getElem.mpr ⟨j, hjlen, hajeq⟩
      rw [routedRun] at hm
      simpa [List.mem_range] using (List.mem_filter.mp hm).1
    have haj_routed : ∀ (s : Usize) (j aj : Nat),
        (routedRun HomeInst home input (Slice.len input) s)[j]? = some aj →
        routedTo HomeInst home input s aj = true := by
      intro s j aj haj
      obtain ⟨hjlen, hajeq⟩ := List.getElem?_eq_some_iff.mp haj
      have hm : aj ∈ routedRun HomeInst home input (Slice.len input) s :=
        List.mem_iff_getElem.mpr ⟨j, hjlen, hajeq⟩
      rw [routedRun] at hm
      exact (List.mem_filter.mp hm).2
    have horder : ∀ (s : Usize) (j j' : Nat), s.val < 256 → j < j' →
        j' < (routedRun HomeInst home input (Slice.len input) s).length →
        ∃ (a b : Usize), buf0.val[endsVal ends0 s + j]? = some a ∧
          buf0.val[endsVal ends0 s + j']? = some b ∧ a < b := by
      intro s j j' hs hj hj'
      have hjl : j < (routedRun HomeInst home input (Slice.len input) s).length := lt_trans hj hj'
      obtain ⟨aj, haj⟩ : ∃ aj, (routedRun HomeInst home input (Slice.len input) s)[j]? = some aj :=
        ⟨_, List.getElem?_eq_getElem hjl⟩
      obtain ⟨bj, hbj⟩ : ∃ bj, (routedRun HomeInst home input (Slice.len input) s)[j']? = some bj :=
        ⟨_, List.getElem?_eq_getElem hj'⟩
      have hlt := routedRun_getElem?_lt hj haj hbj
      have hbjmax : bj ≤ Usize.max :=
        le_trans (le_of_lt (haj_lt s j' bj hs hbj))
          (le_trans (le_of_eq (Slice.len_val input)) input.property)
      have hajmax : aj ≤ Usize.max := le_trans (le_of_lt hlt) hbjmax
      refine ⟨usizeOf aj, usizeOf bj, ?_, ?_, ?_⟩
      · rw [hrun s hs j hjl, haj, Option.map_some]
      · rw [hrun s hs j' hj', hbj, Option.map_some]
      · exact (UScalar.lt_equiv (usizeOf aj) (usizeOf bj)).mpr (by
          rw [usizeOf_val aj hajmax, usizeOf_val bj hbjmax]
          exact hlt)
    have hrouted_conj : ∀ (s : Usize) (j : Nat), s.val < 256 →
        j < (routedRun HomeInst home input (Slice.len input) s).length →
        ∃ v : Usize, buf0.val[endsVal ends0 s + j]? = some v ∧
          v.val < (Slice.len input).val ∧ routedTo HomeInst home input s v.val = true := by
      intro s j hs hj
      obtain ⟨aj, haj⟩ : ∃ aj, (routedRun HomeInst home input (Slice.len input) s)[j]? = some aj :=
        ⟨_, List.getElem?_eq_getElem hj⟩
      refine ⟨usizeOf aj, ?_, ?_, ?_⟩
      · rw [hrun s hs j hj, haj, Option.map_some]
      · rw [usizeOf_val aj (le_trans (le_of_lt (haj_lt s j aj hs haj))
            (le_trans (le_of_eq (Slice.len_val input)) input.property))]
        exact haj_lt s j aj hs haj
      · rw [usizeOf_val aj (le_trans (le_of_lt (haj_lt s j aj hs haj))
            (le_trans (le_of_eq (Slice.len_val input)) input.property))]
        exact haj_routed s j aj haj
    have hcov : endsVal ends0 256#usize = buf0.val.length := h256.trans hbuf_len.symm
    exact ⟨⟨h0, h256, hbuf_len, hmono⟩, horder, hrouted_conj, hcov⟩

end Apeiron.GrowableSchedule
