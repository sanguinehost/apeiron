-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (HT-BUILD): the executed build refines `Apeiron.Table.build`.
import Map.ExecutedResize
open Aeneas Aeneas.Std Result

/-!
# The executed build refines `Apeiron.Table.build`

`MapRep s m live` (`Stored<u64>`) and `CoordMapRep s m live` (`Regen<u64>`) are the representation
the executed insert preserves: `TableRep ∧ OvWF` and `CoordRep ∧ OvWF`.  Over it:

* `map_get_refines` / `coord_map_get_refines`: the read refines `Apeiron.Table.get`;
* `map_insert_refines` / `coord_map_insert_refines`: the last-write insert refines
  `insertWith (fun new _ => new)` and raises `arenaLen + mainCount` by at most one;
* `with_capacity_mapRep`, `new_mapRep`, `coordinate_with_capacity_mapRep`: the constructors
  establish it at the constant-`0` map, the empty live set and `arenaLen + mainCount = 0`.

`execFold s xs` folds the extracted insert (`word_hashtrinity_insert`) over `xs` in stream order in
the `Result` monad; `execBuild n xs` starts it from `word_hashtrinity_with_capacity n`.  The fold
itself is Lean's `List.foldlM`: `proof/extract/lib.rs` roots no Rust build loop, so the loop is not
extracted; each step is the extracted body.

* `build_refines` / `coord_build_refines`: for `xs.length < nilN`, if the executed build returns
  `ok s`, then `s` represents `Apeiron.Table.build (fun new _ => new) 0 xs` on the keys of `xs`, and
  `arenaLen s + mainCount s ≤ xs.length`.  The merge is last-write and the default `0#i64`, the
  constant map every constructor represents.
* `build_get_refines` / `coord_build_get_refines`: the read of the built table refines `get` of
  the canonical build.

Partiality: partial correctness.  The headlines assume the run returns `ok s`; they do not prove
that it does.  `xs.length < nilN` is what keeps the `u32` arena index of a pushed overflow node
below `NIL` at every step.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-! ## The representation -/

/-- **`MapRep`**: the `Stored<u64>` representation the insert preserves — `TableRep` plus the overflow
tree's shape clause `OvWF`. -/
def MapRep (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) : Prop :=
  TableRep s m live ∧ OvWF s

/-- **`CoordMapRep`**: the `Regen<u64>` representation — `CoordRep ∧ OvWF`. -/
def CoordMapRep (s : Tbl (hashtrinity.algebra.Regen U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) : Prop :=
  CoordRep s m live ∧ OvWF s

theorem mapRep_live_congr {s : Tbl (hashtrinity.algebra.Stored U64)}
    {m : Apeiron.Table.HMap U64 I64} {L L' : U64 → Prop} (h : MapRep s m L)
    (hL : ∀ c, L c ↔ L' c) : MapRep s m L' := by
  have : L = L' := funext fun c => propext (hL c)
  exact this ▸ h

theorem coordMapRep_live_congr {s : Tbl (hashtrinity.algebra.Regen U64)}
    {m : Apeiron.Table.HMap U64 I64} {L L' : U64 → Prop} (h : CoordMapRep s m L)
    (hL : ∀ c, L c ↔ L' c) : CoordMapRep s m L' := by
  have : L = L' := funext fun c => propext (hL c)
  exact this ▸ h

/-- **`map_get_refines` (`Stored<u64>`).** -/
theorem map_get_refines (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) (hR : MapRep s m live) (k : U64) (r : Option I64)
    (h : word_hashtrinity_get s k = ok r) :
    (∀ v, r = some v ↔ live k ∧ v = Apeiron.Table.get m k) ∧ (¬ live k → r = none) :=
  get_refines_uncond s m live hR.1 k r h

/-- **`coord_map_get_refines` (`Regen<u64>`).** -/
theorem coord_map_get_refines (s : Tbl (hashtrinity.algebra.Regen U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) (hR : CoordMapRep s m live) (k : U64)
    (r : Option I64) (h : word_coordmap_get s k = ok r) :
    (∀ v, r = some v ↔ live k ∧ v = Apeiron.Table.get m k) ∧ (¬ live k → r = none) :=
  coord_get_refines_uncond s m live hR.1 k r h

/-- **`map_insert_refines` (`Stored<u64>`, every key).**  The last-write insert refines `insertWith`
and raises the arena plus the main count by at most one. -/
theorem map_insert_refines (s : Tbl (hashtrinity.algebra.Stored U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) (hR : MapRep s m live)
    (hsmall : arenaLen s + mainCount s < nilN) (k : U64) (v : I64)
    (s' : Tbl (hashtrinity.algebra.Stored U64)) (h : word_hashtrinity_insert s k v = ok s') :
    MapRep s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) ∧
      arenaLen s' + mainCount s' ≤ arenaLen s + mainCount s + 1 := by
  by_cases hk : IsSent k
  · obtain ⟨ss, hs, hc⟩ := insert_sent_rep SKS SKS_matches hR.1.1 k v hk s' h
    subst hs
    exact ⟨⟨⟨hc, hR.1.2.1, hR.1.2.2⟩, hR.2⟩, by show arenaLen s + mainCount s ≤ _; omega⟩
  · obtain ⟨⟨hC, hW'⟩, hK, hle⟩ :=
      insert_ns_rep SKS SKS_matches SKC skc_set resizeOK_stored ⟨hR.1.1, hR.2⟩ hR.1.2 hsmall hk v h
    exact ⟨⟨⟨hC, hK⟩, hW'⟩, hle⟩

/-- **`coord_map_insert_refines` (`Regen<u64>`, every key).** -/
theorem coord_map_insert_refines (s : Tbl (hashtrinity.algebra.Regen U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) (hR : CoordMapRep s m live)
    (hsmall : arenaLen s + mainCount s < nilN) (k : U64) (v : I64)
    (s' : Tbl (hashtrinity.algebra.Regen U64)) (h : word_coordmap_insert s k v = ok s') :
    CoordMapRep s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) ∧
      arenaLen s' + mainCount s' ≤ arenaLen s + mainCount s + 1 := by
  by_cases hk : IsSent k
  · obtain ⟨ss, hs, hc⟩ := insert_sent_rep RKS RKS_matches hR.1 k v hk s' h
    subst hs
    exact ⟨⟨hc, hR.2⟩, by show arenaLen s + mainCount s ≤ _; omega⟩
  · obtain ⟨⟨hC, hW'⟩, -, hle⟩ :=
      insert_ns_rep RKS RKS_matches (fun _ _ => True) (fun _ _ _ _ _ _ _ _ _ => trivial)
        resizeOK_regen hR trivial hsmall hk v h
    exact ⟨⟨hC, hW'⟩, hle⟩

/-! ## The constructors -/

theorem new_mapRep (s : Tbl (hashtrinity.algebra.Stored U64)) (h : word_hashtrinity_new = ok s) :
    MapRep s (fun _ => 0#i64) (fun _ => False) ∧ arenaLen s + mainCount s = 0 :=
  ⟨⟨new_rep s h, (new_wf s h).1⟩, (new_wf s h).2⟩

theorem with_capacity_mapRep (cap : Usize) (s : Tbl (hashtrinity.algebra.Stored U64))
    (h : word_hashtrinity_with_capacity cap = ok s) :
    MapRep s (fun _ => 0#i64) (fun _ => False) ∧ arenaLen s + mainCount s = 0 :=
  ⟨⟨with_capacity_rep cap s h, (with_capacity_wf cap s h).1⟩, (with_capacity_wf cap s h).2⟩

theorem coordinate_with_capacity_mapRep (cap : Usize) (s : Tbl (hashtrinity.algebra.Regen U64))
    (h : word_coordmap_coordinate_with_capacity cap = ok s) :
    CoordMapRep s (fun _ => 0#i64) (fun _ => False) ∧ arenaLen s + mainCount s = 0 :=
  ⟨⟨coordinate_with_capacity_rep cap s h, (coordinate_with_capacity_wf cap s h).1⟩,
    (coordinate_with_capacity_wf cap s h).2⟩

/-! ## The executed build -/

/-- The extracted last-write insert folded over `xs`, in stream order. -/
noncomputable def execFold (s : Tbl (hashtrinity.algebra.Stored U64)) (xs : List (U64 × I64)) :
    Result (Tbl (hashtrinity.algebra.Stored U64)) :=
  xs.foldlM (fun t kv => word_hashtrinity_insert t kv.1 kv.2) s

/-- `execFold` from `word_hashtrinity_with_capacity n`. -/
noncomputable def execBuild (n : Usize) (xs : List (U64 × I64)) :
    Result (Tbl (hashtrinity.algebra.Stored U64)) := do
  let s ← word_hashtrinity_with_capacity n
  execFold s xs

/-- `execFold` at `Regen<u64>`: the extracted `word_coordmap_insert` folded over `xs`. -/
noncomputable def coordExecFold (s : Tbl (hashtrinity.algebra.Regen U64)) (xs : List (U64 × I64)) :
    Result (Tbl (hashtrinity.algebra.Regen U64)) :=
  xs.foldlM (fun t kv => word_coordmap_insert t kv.1 kv.2) s

/-- `coordExecFold` from `word_coordmap_coordinate_with_capacity n`. -/
noncomputable def coordExecBuild (n : Usize) (xs : List (U64 × I64)) :
    Result (Tbl (hashtrinity.algebra.Regen U64)) := do
  let s ← word_coordmap_coordinate_with_capacity n
  coordExecFold s xs

/-- The canonical fold `build` is made of. -/
abbrev buildFold (m : Apeiron.Table.HMap U64 I64) (xs : List (U64 × I64)) :
    Apeiron.Table.HMap U64 I64 :=
  xs.foldl (fun m kv => Apeiron.Table.insertWith (fun new _ => new) kv.1 kv.2 m) m

theorem foldlM_cons_ok {σ α : Type} (f : σ → α → Result σ) (x : α) (xs : List α) (s s' : σ)
    (h : (x :: xs).foldlM f s = ok s') : ∃ t, f s x = ok t ∧ xs.foldlM f t = ok s' := by
  rw [List.foldlM_cons] at h
  exact bind_inv h

theorem foldlM_nil_ok {σ α : Type} (f : σ → α → Result σ) (s s' : σ)
    (h : ([] : List α).foldlM f s = ok s') : s' = s := by
  rw [List.foldlM_nil] at h
  exact (Result.ok_injective h).symm

theorem mem_keys_cons (kv : U64 × I64) (xs : List (U64 × I64)) (c : U64) :
    c ∈ (kv :: xs).map Prod.fst ↔ c = kv.1 ∨ c ∈ xs.map Prod.fst := by
  simp

/-- **`fold_refines` (`Stored<u64>`).**  From any `MapRep` state with room for `xs.length` more
nodes below `NIL`, the executed fold represents the canonical fold on `live ∪ keys xs`, and the arena
plus the main count grows by at most `xs.length`.  Time: `xs.length` executed inserts. -/
theorem fold_refines : ∀ (xs : List (U64 × I64)) (s : Tbl (hashtrinity.algebra.Stored U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop), MapRep s m live →
    arenaLen s + mainCount s + xs.length < nilN →
    ∀ s', execFold s xs = ok s' →
      MapRep s' (buildFold m xs) (fun c => live c ∨ c ∈ xs.map Prod.fst) ∧
        arenaLen s' + mainCount s' ≤ arenaLen s + mainCount s + xs.length
  | [], s, m, live, hR, _, s', h => by
    have := foldlM_nil_ok _ s s' h
    subst this
    exact ⟨mapRep_live_congr hR (fun c => by simp), by simp⟩
  | kv :: xs, s, m, live, hR, hcap, s', h => by
    obtain ⟨t, ht, hrest⟩ := foldlM_cons_ok _ kv xs s s' h
    have hlen : (kv :: xs).length = xs.length + 1 := rfl
    obtain ⟨hRt, hlet⟩ := map_insert_refines s m live hR (by omega) kv.1 kv.2 t ht
    obtain ⟨hRs, hles⟩ := fold_refines xs t _ _ hRt (by omega) s' hrest
    refine ⟨mapRep_live_congr hRs (fun c => ?_), by omega⟩
    rw [mem_keys_cons]
    tauto

/-- **`coord_fold_refines` (`Regen<u64>`).** -/
theorem coord_fold_refines : ∀ (xs : List (U64 × I64)) (s : Tbl (hashtrinity.algebra.Regen U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop), CoordMapRep s m live →
    arenaLen s + mainCount s + xs.length < nilN →
    ∀ s', coordExecFold s xs = ok s' →
      CoordMapRep s' (buildFold m xs) (fun c => live c ∨ c ∈ xs.map Prod.fst) ∧
        arenaLen s' + mainCount s' ≤ arenaLen s + mainCount s + xs.length
  | [], s, m, live, hR, _, s', h => by
    have := foldlM_nil_ok _ s s' h
    subst this
    exact ⟨coordMapRep_live_congr hR (fun c => by simp), by simp⟩
  | kv :: xs, s, m, live, hR, hcap, s', h => by
    obtain ⟨t, ht, hrest⟩ := foldlM_cons_ok _ kv xs s s' h
    have hlen : (kv :: xs).length = xs.length + 1 := rfl
    obtain ⟨hRt, hlet⟩ := coord_map_insert_refines s m live hR (by omega) kv.1 kv.2 t ht
    obtain ⟨hRs, hles⟩ := coord_fold_refines xs t _ _ hRt (by omega) s' hrest
    refine ⟨coordMapRep_live_congr hRs (fun c => ?_), by omega⟩
    rw [mem_keys_cons]
    tauto

theorem build_eq_buildFold (xs : List (U64 × I64)) :
    Apeiron.Table.build (fun new _ => new) 0#i64 xs = buildFold (fun _ => 0#i64) xs := rfl

/-! ## The headlines -/

/-- **`build_refines` (`Stored<u64>`).**  For a stream `xs` shorter than `nilN = 2^32 - 1`, if the
executed build (`word_hashtrinity_with_capacity n`, then `word_hashtrinity_insert` at each pair in
order) returns `ok s`, then `s` represents the canonical last-write build
`Apeiron.Table.build (fun new _ => new) 0#i64 xs` on the keys of `xs`, and the overflow arena plus
the main count is at most `xs.length`. -/
theorem build_refines (n : Usize) (xs : List (U64 × I64)) (hlen : xs.length < nilN)
    (s : Tbl (hashtrinity.algebra.Stored U64)) (h : execBuild n xs = ok s) :
    MapRep s (Apeiron.Table.build (fun new _ => new) 0#i64 xs) (fun c => c ∈ xs.map Prod.fst) ∧
      arenaLen s + mainCount s ≤ xs.length := by
  unfold execBuild at h
  obtain ⟨s0, h0, h⟩ := bind_inv h
  obtain ⟨hR0, hz⟩ := with_capacity_mapRep n s0 h0
  obtain ⟨hR, hle⟩ := fold_refines xs s0 _ _ hR0 (by omega) s h
  rw [build_eq_buildFold]
  exact ⟨mapRep_live_congr hR (fun c => by simp), by omega⟩

/-- **`coord_build_refines` (`Regen<u64>`).**  The same from
`word_coordmap_coordinate_with_capacity n` with `word_coordmap_insert`. -/
theorem coord_build_refines (n : Usize) (xs : List (U64 × I64)) (hlen : xs.length < nilN)
    (s : Tbl (hashtrinity.algebra.Regen U64)) (h : coordExecBuild n xs = ok s) :
    CoordMapRep s (Apeiron.Table.build (fun new _ => new) 0#i64 xs)
        (fun c => c ∈ xs.map Prod.fst) ∧
      arenaLen s + mainCount s ≤ xs.length := by
  unfold coordExecBuild at h
  obtain ⟨s0, h0, h⟩ := bind_inv h
  obtain ⟨hR0, hz⟩ := coordinate_with_capacity_mapRep n s0 h0
  obtain ⟨hR, hle⟩ := coord_fold_refines xs s0 _ _ hR0 (by omega) s h
  rw [build_eq_buildFold]
  exact ⟨coordMapRep_live_congr hR (fun c => by simp), by omega⟩

/-- **`build_get_refines` (`Stored<u64>`).**  Reading the executed build at `k` returns
`some (get (build …) k)` when `k` is a key of `xs`, and `none` otherwise. -/
theorem build_get_refines (n : Usize) (xs : List (U64 × I64)) (hlen : xs.length < nilN)
    (s : Tbl (hashtrinity.algebra.Stored U64)) (h : execBuild n xs = ok s) (k : U64)
    (r : Option I64) (hr : word_hashtrinity_get s k = ok r) :
    (∀ v, r = some v ↔
        k ∈ xs.map Prod.fst ∧ v = Apeiron.Table.get (Apeiron.Table.build (fun new _ => new) 0#i64 xs) k) ∧
      (k ∉ xs.map Prod.fst → r = none) :=
  map_get_refines s _ _ (build_refines n xs hlen s h).1 k r hr

/-- **`coord_build_get_refines` (`Regen<u64>`).** -/
theorem coord_build_get_refines (n : Usize) (xs : List (U64 × I64)) (hlen : xs.length < nilN)
    (s : Tbl (hashtrinity.algebra.Regen U64)) (h : coordExecBuild n xs = ok s) (k : U64)
    (r : Option I64) (hr : word_coordmap_get s k = ok r) :
    (∀ v, r = some v ↔
        k ∈ xs.map Prod.fst ∧ v = Apeiron.Table.get (Apeiron.Table.build (fun new _ => new) 0#i64 xs) k) ∧
      (k ∉ xs.map Prod.fst → r = none) :=
  coord_map_get_refines s _ _ (coord_build_refines n xs hlen s h).1 k r hr

end Apeiron.MapExecuted
