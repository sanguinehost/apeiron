/-
Refinement of the extracted MoE router selection (epic `moe-router`).

The Rust body is `src/engine/moe.rs`. `moe.best_unchosen` (`src/engine/moe.rs:258-279`) scans for the
unchosen index with the largest logit, ties broken to the lower index, and reports whether one
exists. `moe.route_topk` (`src/engine/moe.rs:222-251`) runs that scan `k` times, marking each winner
chosen, and appends the winners in selection order.

## Landed here

* `best_unchosen_spec`: the scan returns the minimal-key unchosen index under the selection key
  `(logit descending, index ascending)` when some entry is unchosen, and `(0, false)` when every
  entry is chosen.
* `route_topk_spec`: on `1 ≤ k ≤ n ≤ 2^32-1`, `moe.route_topk logits k` returns `Ok sel` with
  `sel.val.map (·.val)` exactly `Nous.Model.topkSel (logits.val.map (·.val)) k` — the same list, in
  the same order, because both order by `(logit desc, index asc)`; and it refuses `k = 0` and
  `k > n` with `Err (BadK k n)`.

The pure bridge is `idxOrder ls`, the indices of `logits` sorted by the selection key: `topkSel ls k`
is its first `k` entries (`topkSel_eq_take`), and its `p`-th entry is the minimal-key unchosen index
once the first `p` are chosen (`idxAt_spec`).

## Trust boundary

Reaches the extracted `moe.route_topk`, `moe.route_topk_loop(.body)`, `moe.best_unchosen` and
`moe.best_unchosen_loop(.body)` of `proof/ApeironWord.lean` through the `Nous.Model.topkSel`
specification. Does not cover rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V, the `unsafe` container,
nor the gate-weight and expert-combination paths (`gate_weights`, `moe_ffn_fixed`).
-/
import Numeric.AttnLoopScore
import Nous.Model.TopKRouter

namespace apeiron_word

open Aeneas Aeneas.Std Result
open List

noncomputable section

/-! ## 1. The pure selection order -/

/-- The expert indices of `logits` in selection order: sorted by the key
`(logit descending, index ascending)`. -/
def idxOrder (ls : List ℤ) : List ℕ :=
  (Nous.Model.sortedKeys ls).map (fun q => (ofLex q).2)

/-- The selection key of index `i` carrying value `val i`. -/
def keyAt (val : ℕ → ℤ) (i : ℕ) : Nous.Model.RouterKey := Nous.Model.routerKey (val i) i

theorem topkSel_eq_take (ls : List ℤ) (k : ℕ) :
    Nous.Model.topkSel ls k = (idxOrder ls).take k := by
  unfold Nous.Model.topkSel Nous.Model.selKeys idxOrder Nous.Model.sortedKeys
  exact List.map_take

theorem sortedKeys_length (ls : List ℤ) : (Nous.Model.sortedKeys ls).length = ls.length := by
  unfold Nous.Model.sortedKeys Nous.Model.keyList
  simp [List.length_mergeSort]

theorem idxOrder_length (ls : List ℤ) : (idxOrder ls).length = ls.length := by
  unfold idxOrder
  rw [List.length_map, sortedKeys_length]

/-- The `p`-th selected index of `ls`. -/
def idxAt (ls : List ℤ) (p : ℕ) : ℕ := (idxOrder ls).getD p 0

/-- A sorted key is the key of its own index. -/
theorem sortedKey_eq (ls : List ℤ) {q : Nous.Model.RouterKey}
    (hq : q ∈ Nous.Model.sortedKeys ls) :
    q = keyAt (fun i => ls.getD i 0) ((ofLex q).2) := by
  have hq' := (Nous.Model.sortedKeys_perm ls).mem_iff.mp hq
  rw [Nous.Model.keyList_mem] at hq'
  obtain ⟨i, hi⟩ := hq'
  have hgd : ls.getD (i : ℕ) 0 = ls.get i := by
    rw [List.getD_eq_getElem _ _ i.isLt, List.get_eq_getElem]
  have hsnd : (ofLex q).2 = (i : ℕ) := by rw [← hi]; simp [Nous.Model.routerKey]
  rw [hsnd, ← hi]
  simp only [keyAt]
  rw [hgd]

/-- Two keys of the same index are equal. -/
theorem sortedKey_snd_inj (ls : List ℤ) {a b : Nous.Model.RouterKey}
    (ha : a ∈ Nous.Model.sortedKeys ls) (hb : b ∈ Nous.Model.sortedKeys ls)
    (h : (ofLex a).2 = (ofLex b).2) : a = b := by
  rw [sortedKey_eq ls ha, sortedKey_eq ls hb, h]

theorem idxOrder_nodup (ls : List ℤ) : (idxOrder ls).Nodup := by
  unfold idxOrder
  apply List.Nodup.map_on
  · intro a ha b hb hab
    exact sortedKey_snd_inj ls ha hb hab
  · exact Nous.Model.sortedKeys_nodup ls

theorem idxAt_getElem (ls : List ℤ) {p : ℕ} (hp : p < (idxOrder ls).length) :
    idxAt ls p = (idxOrder ls)[p] := by
  unfold idxAt
  exact List.getD_eq_getElem _ _ hp

/-- The key of the `p`-th selected index is the `p`-th sorted key. -/
theorem idxAt_key (ls : List ℤ) {p : ℕ} (hp : p < (idxOrder ls).length) :
    keyAt (fun i => ls.getD i 0) (idxAt ls p) =
      (Nous.Model.sortedKeys ls)[p]'(by rw [sortedKeys_length]; rw [idxOrder_length] at hp; exact hp) := by
  have hp' : p < (Nous.Model.sortedKeys ls).length := by
    rw [sortedKeys_length]; rw [idxOrder_length] at hp; exact hp
  have hidx : idxAt ls p = (ofLex ((Nous.Model.sortedKeys ls)[p]'hp')).2 := by
    rw [idxAt_getElem ls hp]
    unfold idxOrder
    rw [List.getElem_map]
  rw [hidx]
  exact (sortedKey_eq ls (List.getElem_mem hp')).symm

theorem idxAt_mem_range (ls : List ℤ) {p : ℕ} (hp : p < (idxOrder ls).length) :
    idxAt ls p < ls.length := by
  have hp' : p < (Nous.Model.sortedKeys ls).length := by
    rw [sortedKeys_length]; rw [idxOrder_length] at hp; exact hp
  have hmem : (Nous.Model.sortedKeys ls)[p]'hp' ∈ Nous.Model.sortedKeys ls :=
    List.getElem_mem hp'
  have hq := (Nous.Model.sortedKeys_perm ls).mem_iff.mp hmem
  rw [Nous.Model.keyList_mem] at hq
  obtain ⟨i, hi⟩ := hq
  have hidx : idxAt ls p = (i : ℕ) := by
    have h2 : (ofLex ((Nous.Model.sortedKeys ls)[p]'hp')).2 = (i : ℕ) := by
      rw [← hi]; simp [Nous.Model.routerKey]
    have hidx' : idxAt ls p = (ofLex ((Nous.Model.sortedKeys ls)[p]'hp')).2 := by
      rw [idxAt_getElem ls hp]
      unfold idxOrder
      rw [List.getElem_map]
    rw [hidx', h2]
  rw [hidx]
  exact i.isLt

theorem idxAt_not_mem_take (ls : List ℤ) {p : ℕ} (hp : p < (idxOrder ls).length) :
    idxAt ls p ∉ (idxOrder ls).take p := by
  intro hmem
  have hp' : p < (Nous.Model.sortedKeys ls).length := by
    rw [sortedKeys_length]; rw [idxOrder_length] at hp; exact hp
  have hTtake : (idxOrder ls).take p =
      ((Nous.Model.sortedKeys ls).take p).map (fun q => (ofLex q).2) := by
    unfold idxOrder
    exact (List.map_take).symm
  rw [hTtake] at hmem
  obtain ⟨a, ha, haeq⟩ := List.mem_map.mp hmem
  obtain ⟨qq, hqq, hqqeq⟩ := List.getElem_of_mem ha
  have hqqp : qq < p := by rw [List.length_take] at hqq; omega
  have hqqlt : qq < (Nous.Model.sortedKeys ls).length := by
    rw [List.length_take] at hqq; omega
  have hksqq : (Nous.Model.sortedKeys ls)[qq] = a := by
    have hlen : qq < ((Nous.Model.sortedKeys ls).take p).length := by rw [List.length_take]; omega
    have h2 := List.getElem_take (xs := Nous.Model.sortedKeys ls) (j := p) (i := qq) (h := hlen)
    rw [← h2]
    exact hqqeq
  have hidx : idxAt ls p = (ofLex ((Nous.Model.sortedKeys ls)[p]'hp')).2 := by
    rw [idxAt_getElem ls hp]
    unfold idxOrder
    rw [List.getElem_map]
  have hsndeq : (ofLex ((Nous.Model.sortedKeys ls)[qq]'hqqlt)).2 =
      (ofLex ((Nous.Model.sortedKeys ls)[p]'hp')).2 := by
    rw [hksqq, haeq]
    exact hidx
  have heq : (Nous.Model.sortedKeys ls)[qq] = (Nous.Model.sortedKeys ls)[p] :=
    sortedKey_snd_inj ls (List.getElem_mem hqqlt) (List.getElem_mem hp') hsndeq
  have hnod := Nous.Model.sortedKeys_nodup ls
  have hne : ¬ (Nous.Model.sortedKeys ls)[qq] = (Nous.Model.sortedKeys ls)[p] := by
    intro hc
    have hq_eq : qq = p := (List.Nodup.getElem_inj_iff hnod).mp hc
    omega
  exact hne heq

/-- The `p`-th selected index dominates every unchosen index under the selection key. -/
theorem idxAt_le_unchosen (ls : List ℤ) {p : ℕ} (hp : p < (idxOrder ls).length) {j : ℕ}
    (hj : j < ls.length) (hjn : j ∉ (idxOrder ls).take p) :
    keyAt (fun i => ls.getD i 0) (idxAt ls p) ≤ keyAt (fun i => ls.getD i 0) j := by
  have hp' : p < (Nous.Model.sortedKeys ls).length := by
    rw [sortedKeys_length]; rw [idxOrder_length] at hp; exact hp
  -- the sorted position of `j`'s key
  have hkeyj_mem : Nous.Model.routerKey (ls.get ⟨j, hj⟩) j ∈ Nous.Model.sortedKeys ls :=
    Nous.Model.keyAt_mem_sorted ls j hj
  have hkeyj_val : keyAt (fun i => ls.getD i 0) j = Nous.Model.routerKey (ls.get ⟨j, hj⟩) j := by
    simp only [keyAt]
    rw [List.getD_eq_getElem _ _ hj, List.get_eq_getElem]
  obtain ⟨qq, hqq, hqqeq⟩ := List.getElem_of_mem hkeyj_mem
  have hTtake : (idxOrder ls).take p =
      ((Nous.Model.sortedKeys ls).take p).map (fun q => (ofLex q).2) := by
    unfold idxOrder
    exact (List.map_take).symm
  rcases lt_trichotomy qq p with hlt | heq | hgt
  · -- the key lies in the first `p`: its index is chosen
    exfalso
    apply hjn
    rw [hTtake]
    have hm : (Nous.Model.sortedKeys ls)[qq] ∈ (Nous.Model.sortedKeys ls).take p := by
      have hlen : qq < ((Nous.Model.sortedKeys ls).take p).length := by
        rw [List.length_take]; omega
      have h1 : ((Nous.Model.sortedKeys ls).take p)[qq]'hlen ∈ (Nous.Model.sortedKeys ls).take p :=
        List.getElem_mem hlen
      have h2 := List.getElem_take (xs := Nous.Model.sortedKeys ls) (j := p) (i := qq) (h := hlen)
      rwa [h2] at h1
    refine List.mem_map.mpr ⟨_, hm, ?_⟩
    rw [hqqeq]
    simp [Nous.Model.routerKey]
  · -- the key is exactly the `p`-th: same index
    subst heq
    have hk := idxAt_key ls hp
    rw [hkeyj_val, ← hqqeq, hk]
  · -- the key lies after the `p`-th: the sorted order gives the inequality
    have htake : (Nous.Model.sortedKeys ls)[p]'hp' ∈ (Nous.Model.sortedKeys ls).take (p+1) := by
      have hlen : p < ((Nous.Model.sortedKeys ls).take (p+1)).length := by
        rw [List.length_take]; omega
      have h1 : ((Nous.Model.sortedKeys ls).take (p+1))[p]'hlen ∈
          (Nous.Model.sortedKeys ls).take (p+1) := List.getElem_mem hlen
      have h2 := List.getElem_take (xs := Nous.Model.sortedKeys ls) (j := p+1) (i := p) (h := hlen)
      rwa [h2] at h1
    have hdrop : (Nous.Model.sortedKeys ls)[qq] ∈ (Nous.Model.sortedKeys ls).drop (p+1) := by
      have hlen : qq - (p+1) < ((Nous.Model.sortedKeys ls).drop (p+1)).length := by
        rw [List.length_drop]; omega
      have h1 : ((Nous.Model.sortedKeys ls).drop (p+1))[qq - (p+1)]'hlen ∈
          (Nous.Model.sortedKeys ls).drop (p+1) := List.getElem_mem hlen
      have h2 : ((Nous.Model.sortedKeys ls).drop (p+1))[qq - (p+1)]'hlen =
          (Nous.Model.sortedKeys ls)[qq] := by
        have h := List.getElem_drop (xs := Nous.Model.sortedKeys ls) (i := p+1)
          (j := qq - (p+1)) (h := hlen)
        have hadd : p + 1 + (qq - (p+1)) = qq := by omega
        simpa only [hadd] using h
      simpa only [h2] using h1
    have hle : (Nous.Model.sortedKeys ls)[p]'hp' ≤ (Nous.Model.sortedKeys ls)[qq] :=
      ((Nous.Model.sortedKeys_sorted ls).pairwise).rel_of_mem_take_of_mem_drop htake hdrop
    rw [hkeyj_val, ← hqqeq, idxAt_key ls hp]
    exact hle

/-- **Next selected index.** For `p < k ≤ ls.length`, the `p`-th selected index is a valid index,
is not among the first `p` selected indices, and dominates every index outside that prefix under the
selection key. -/
theorem idxAt_spec (ls : List ℤ) {p k : ℕ} (hpk : p < k) (hkl : k ≤ ls.length) :
    idxAt ls p < ls.length ∧
    idxAt ls p ∉ (idxOrder ls).take p ∧
    (∀ j < ls.length, j ∉ (idxOrder ls).take p →
      keyAt (fun i => ls.getD i 0) (idxAt ls p) ≤ keyAt (fun i => ls.getD i 0) j) := by
  have hp : p < (idxOrder ls).length := by rw [idxOrder_length]; omega
  exact ⟨idxAt_mem_range ls hp, idxAt_not_mem_take ls hp,
    fun j hj hjn => idxAt_le_unchosen ls hp hj hjn⟩

/-! ## 2. The `best_unchosen` scan -/

/-- The pure scan state of `best_unchosen` after `j` iterations: the minimal-key unchosen index in
`[0,j)`, or the sentinel `n` when every index in `[0,j)` is chosen, together with its value. -/
def bestScan (n : ℕ) (val : ℕ → ℤ) (taken : ℕ → Bool) : ℕ → ℕ × ℤ
  | 0 => (n, 0)
  | (j+1) =>
    if taken j then bestScan n val taken j
    else if (bestScan n val taken j).1 = n then (j, val j)
    else if val j > (bestScan n val taken j).2 then (j, val j)
    else bestScan n val taken j

theorem bestScan_succ_taken (n : ℕ) (val : ℕ → ℤ) (taken : ℕ → Bool) (j : ℕ)
    (h : taken j = true) : bestScan n val taken (j+1) = bestScan n val taken j := by
  rw [bestScan, if_pos h]

theorem bestScan_succ_none (n : ℕ) (val : ℕ → ℤ) (taken : ℕ → Bool) (j : ℕ)
    (h1 : taken j = false) (h2 : (bestScan n val taken j).1 = n) :
    bestScan n val taken (j+1) = (j, val j) := by
  rw [bestScan, if_neg (by rw [h1]; simp), if_pos h2]

theorem bestScan_succ_greater (n : ℕ) (val : ℕ → ℤ) (taken : ℕ → Bool) (j : ℕ)
    (h1 : taken j = false) (h2 : (bestScan n val taken j).1 ≠ n)
    (h3 : val j > (bestScan n val taken j).2) :
    bestScan n val taken (j+1) = (j, val j) := by
  rw [bestScan, if_neg (by rw [h1]; simp), if_neg h2, if_pos h3]

theorem bestScan_succ_keep (n : ℕ) (val : ℕ → ℤ) (taken : ℕ → Bool) (j : ℕ)
    (h1 : taken j = false) (h2 : (bestScan n val taken j).1 ≠ n)
    (h3 : ¬ val j > (bestScan n val taken j).2) :
    bestScan n val taken (j+1) = bestScan n val taken j := by
  rw [bestScan, if_neg (by rw [h1]; simp), if_neg h2, if_neg h3]

theorem bestScan_fst_le (n : ℕ) (val : ℕ → ℤ) (taken : ℕ → Bool) :
    ∀ j, j ≤ n → (bestScan n val taken j).1 ≤ n := by
  intro j
  induction j with
  | zero => intro _; simp [bestScan]
  | succ j ih =>
    intro hj
    have hj' : j ≤ n := by omega
    by_cases ht : taken j = true
    · rw [bestScan_succ_taken n val taken j ht]; exact ih hj'
    · have hf : taken j = false := Bool.eq_false_iff.mpr ht
      by_cases hn : (bestScan n val taken j).1 = n
      · rw [bestScan_succ_none n val taken j hf hn]; omega
      · by_cases hg : val j > (bestScan n val taken j).2
        · rw [bestScan_succ_greater n val taken j hf hn hg]; omega
        · rw [bestScan_succ_keep n val taken j hf hn hg]; exact ih hj'

theorem bestScan_eq_sentinel (n : ℕ) (val : ℕ → ℤ) (taken : ℕ → Bool) :
    ∀ j, j ≤ n → ((bestScan n val taken j).1 = n ↔ ∀ i < j, taken i = true) := by
  intro j
  induction j with
  | zero => intro _; simp [bestScan]
  | succ j ih =>
    intro hj
    have hj' : j ≤ n := by omega
    have hjn : j < n := by omega
    have hIH := ih hj'
    by_cases ht : taken j = true
    · rw [bestScan_succ_taken n val taken j ht]
      constructor
      · intro h i hi
        by_cases hij : i < j
        · rw [hIH] at h; exact h i hij
        · have hij' : i = j := by omega
          rw [hij']; exact ht
      · intro h; rw [hIH]; exact fun i hi => h i (by omega)
    · have hf : taken j = false := Bool.eq_false_iff.mpr ht
      by_cases hn : (bestScan n val taken j).1 = n
      · rw [bestScan_succ_none n val taken j hf hn]
        constructor
        · intro h; omega
        · intro h
          have := h j (by omega)
          rw [hf] at this
          exact absurd this (by simp)
      · by_cases hg : val j > (bestScan n val taken j).2
        · rw [bestScan_succ_greater n val taken j hf hn hg]
          constructor
          · intro h; omega
          · intro h
            have := h j (by omega)
            rw [hf] at this
            exact absurd this (by simp)
        · rw [bestScan_succ_keep n val taken j hf hn hg]
          constructor
          · intro h; exact absurd h hn
          · intro h
            have := h j (by omega)
            rw [hf] at this
            exact absurd this (by simp)

theorem bestScan_lt_props (n : ℕ) (val : ℕ → ℤ) (taken : ℕ → Bool) :
    ∀ j, j ≤ n → (bestScan n val taken j).1 < n →
      (bestScan n val taken j).1 < j ∧
      taken ((bestScan n val taken j).1) = false ∧
      (bestScan n val taken j).2 = val ((bestScan n val taken j).1) ∧
      ∀ i < j, taken i = false →
        keyAt val ((bestScan n val taken j).1) ≤ keyAt val i := by
  intro j
  induction j with
  | zero => intro _ h; simp [bestScan] at h
  | succ j ih =>
    intro hj hlt
    have hj' : j ≤ n := by omega
    by_cases ht : taken j = true
    · rw [bestScan_succ_taken n val taken j ht] at hlt ⊢
      have h0 := ih hj' hlt
      refine ⟨by omega, h0.2.1, h0.2.2.1, ?_⟩
      intro i hi hif
      by_cases hij : i < j
      · exact h0.2.2.2 i hij hif
      · have hij' : i = j := by omega
        rw [hij'] at hif
        rw [ht] at hif
        exact absurd hif (by simp)
    · have hf : taken j = false := Bool.eq_false_iff.mpr ht
      by_cases hn : (bestScan n val taken j).1 = n
      · rw [bestScan_succ_none n val taken j hf hn] at hlt ⊢
        refine ⟨by omega, hf, rfl, ?_⟩
        intro i hi hif
        by_cases hin : i < j
        · have hs := (bestScan_eq_sentinel n val taken j hj').mp hn
          rw [hs i hin] at hif
          exact absurd hif (by simp)
        · have hij' : i = j := by omega
          rw [hij']
      · have hprev : (bestScan n val taken j).1 < n := by
          have := bestScan_fst_le n val taken j hj'; omega
        by_cases hg : val j > (bestScan n val taken j).2
        · rw [bestScan_succ_greater n val taken j hf hn hg] at hlt ⊢
          have h0 := ih hj' hprev
          refine ⟨by omega, hf, rfl, ?_⟩
          intro i hi hif
          by_cases hij : i < j
          · have hle0 : keyAt val j ≤ keyAt val (bestScan n val taken j).1 := by
              unfold keyAt
              rw [Nous.Model.routerKey_le_iff]
              left
              rw [h0.2.2.1] at hg
              exact hg
            exact le_trans hle0 (h0.2.2.2 i hij hif)
          · have hij' : i = j := by omega
            rw [hij']
        · rw [bestScan_succ_keep n val taken j hf hn hg] at hlt ⊢
          have h0 := ih hj' hlt
          refine ⟨by omega, h0.2.1, h0.2.2.1, ?_⟩
          intro i hi hif
          by_cases hij : i < j
          · exact h0.2.2.2 i hij hif
          · have hij' : i = j := by omega
            rw [hij']
            have hle : val j ≤ val (bestScan n val taken j).1 := by
              rw [h0.2.2.1] at hg
              exact le_of_not_gt hg
            unfold keyAt
            rw [Nous.Model.routerKey_le_iff]
            rcases lt_or_eq_of_le hle with hh | hh
            · left; exact hh
            · right; exact ⟨hh.symm, by omega⟩

theorem bestScan_lt_of_exists (n : ℕ) (val : ℕ → ℤ) (taken : ℕ → Bool)
    (hex : ∃ i < n, taken i = false) : (bestScan n val taken n).1 < n := by
  by_contra h
  have hle := bestScan_fst_le n val taken n (le_refl n)
  have heq : (bestScan n val taken n).1 = n := by omega
  have hall := (bestScan_eq_sentinel n val taken n (le_refl n)).mp heq
  obtain ⟨i, hi, hif⟩ := hex
  have := hall i hi
  rw [hif] at this
  exact absurd this (by simp)

/-! ## 3. The `best_unchosen` loop -/

/-- The loop invariant of `best_unchosen`: `j` is past the scanned prefix, and the running state is
the pure scan `bestScan` over `[0,j)`, with the `chosen` mask and the `logits` values. -/
def bestLoopInv (logits : Slice I64) (chosen : Slice Bool) (n : Usize)
    (best : Usize) (best_val : I64) (j : Usize) : Prop :=
  j.val ≤ n.val ∧ best.val ≤ n.val ∧
  ((best.val, best_val.val) =
    bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
      (fun i => chosen.val.getD i false) j.val)

theorem best_unchosen_loop_body_step (logits : Slice I64) (chosen : Slice Bool) (n : Usize)
    (hn : n.val = logits.val.length) (hclen : chosen.val.length = logits.val.length)
    (best : Usize) (best_val : I64) (j : Usize)
    (hjn : j.val < n.val) (hinv : bestLoopInv logits chosen n best best_val j) :
    WP.spec (moe.best_unchosen_loop.body logits chosen n best best_val j)
      (fun cf => ∃ best2 best_val2 j2,
        cf = ControlFlow.cont (best2, best_val2, j2) ∧
        bestLoopInv logits chosen n best2 best_val2 j2 ∧ j2.val = j.val + 1) := by
  obtain ⟨hjle, hbestle, hinveq⟩ := hinv
  have hb1 : best.val = (bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
      (fun i => chosen.val.getD i false) j.val).1 := (Prod.ext_iff.mp hinveq).1
  have hb2 : best_val.val = (bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
      (fun i => chosen.val.getD i false) j.val).2 := (Prod.ext_iff.mp hinveq).2
  have hjk : j.val < chosen.val.length := by rw [hclen, ← hn]; exact hjn
  have hlen_eq : Slice.len chosen = n := by
    apply UScalar.eq_of_val_eq
    simp only [Slice.len_val, hclen, hn]
  have hjnProp : j < n := (UScalar.lt_equiv j n).2 hjn
  have hidxc : Slice.index_usize chosen j = ok (chosen.val.getD j.val false) := by
    unfold Slice.index_usize
    rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hjk, List.getD_eq_getElem _ _ hjk]
  have hjk2 : j.val < logits.val.length := by rw [← hn]; exact hjn
  have hidxl : Slice.index_usize logits j = ok (logits.val.getD j.val 0#i64) := by
    unfold Slice.index_usize
    rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hjk2, List.getD_eq_getElem _ _ hjk2]
  have h1 : (1#usize).val = 1 := by decide
  have hjmax : j.val + (1#usize).val ≤ Usize.max := by
    rw [h1]
    have hnm : n.val ≤ Usize.max := by rw [hn]; exact Slice.property logits
    omega
  obtain ⟨j2, haddj, hj2⟩ := WP.spec_imp_exists (Usize.add_spec (x := j) (y := 1#usize) hjmax)
  have hj2' : j2.val = j.val + 1 := by rw [hj2, h1]
  rw [WP.spec_equiv_exists]
  by_cases htk : chosen.val.getD j.val false = true
  · refine ⟨ControlFlow.cont (best, best_val, j2), ?_, best, best_val, j2, rfl, ?_, hj2'⟩
    · rw [moe.best_unchosen_loop.body, if_pos hjnProp]
      simp only [hlen_eq, if_pos hjnProp, hidxc, hidxl, bind_tc_ok, if_pos htk, haddj]
      rfl
    · refine ⟨by omega, hbestle, ?_⟩
      rw [hj2', bestScan_succ_taken n.val (fun i => (logits.val.getD i 0#i64).val)
        (fun i => chosen.val.getD i false) j.val htk]
      exact hinveq
  · have hf : chosen.val.getD j.val false = false := Bool.eq_false_iff.mpr htk
    by_cases hbn : best.val = n.val
    · have hbn' : best = n := UScalar.eq_of_val_eq hbn
      have hp1 : (bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
          (fun i => chosen.val.getD i false) j.val).1 = n.val := by
        rw [← hb1, hbn]
      have hsc : bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
          (fun i => chosen.val.getD i false) (j.val + 1) =
          (j.val, (logits.val.getD j.val 0#i64).val) :=
        bestScan_succ_none n.val (fun i => (logits.val.getD i 0#i64).val)
          (fun i => chosen.val.getD i false) j.val hf hp1
      refine ⟨ControlFlow.cont (j, logits.val.getD j.val 0#i64, j2), ?_, j, logits.val.getD j.val 0#i64, j2,
        rfl, ?_, hj2'⟩
      · rw [moe.best_unchosen_loop.body, if_pos hjnProp]
        simp only [hlen_eq, if_pos hjnProp, hidxc, hidxl, bind_tc_ok, if_neg htk, if_pos hbn',
          haddj]
        rfl
      · refine ⟨by omega, ?_, ?_⟩
        · exact hjn.le
        · rw [hj2', hsc]
    · have hbn' : ¬ best = n := fun hc => hbn (by rw [hc])
      by_cases hgt : logits.val.getD j.val 0#i64 > best_val
      · have hgtv : (logits.val.getD j.val 0#i64).val > best_val.val :=
          (IScalar.lt_equiv best_val (logits.val.getD j.val 0#i64)).1 hgt
        have hgtv' : (logits.val.getD j.val 0#i64).val >
            (bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
              (fun i => chosen.val.getD i false) j.val).2 := by
          rw [← hb2]; exact hgtv
        have hp1 : (bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
            (fun i => chosen.val.getD i false) j.val).1 ≠ n.val := by
          intro hc; exact hbn (by rw [hb1, hc])
        have hsc : bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
            (fun i => chosen.val.getD i false) (j.val + 1) =
            (j.val, (logits.val.getD j.val 0#i64).val) :=
          bestScan_succ_greater n.val (fun i => (logits.val.getD i 0#i64).val)
            (fun i => chosen.val.getD i false) j.val hf hp1 hgtv'
        refine ⟨ControlFlow.cont (j, logits.val.getD j.val 0#i64, j2), ?_, j, logits.val.getD j.val 0#i64,
          j2, rfl, ?_, hj2'⟩
        · rw [moe.best_unchosen_loop.body, if_pos hjnProp]
          simp only [hlen_eq, if_pos hjnProp, hidxc, hidxl, bind_tc_ok, if_neg htk, if_neg hbn',
            if_pos hgt, haddj]
          rfl
        · refine ⟨by omega, ?_, ?_⟩
          · exact hjn.le
          · rw [hj2', hsc]
      · have hgtv : ¬ (logits.val.getD j.val 0#i64).val > best_val.val := by
          intro hc
          exact hgt ((IScalar.lt_equiv best_val (logits.val.getD j.val 0#i64)).2 hc)
        have hgtv' : ¬ (logits.val.getD j.val 0#i64).val >
            (bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
              (fun i => chosen.val.getD i false) j.val).2 := by
          rw [← hb2]; exact hgtv
        have hp1 : (bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
            (fun i => chosen.val.getD i false) j.val).1 ≠ n.val := by
          intro hc; exact hbn (by rw [hb1, hc])
        have hsc : bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
            (fun i => chosen.val.getD i false) (j.val + 1) =
            bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
              (fun i => chosen.val.getD i false) j.val :=
          bestScan_succ_keep n.val (fun i => (logits.val.getD i 0#i64).val)
            (fun i => chosen.val.getD i false) j.val hf hp1 hgtv'
        refine ⟨ControlFlow.cont (best, best_val, j2), ?_, best, best_val, j2, rfl, ?_, hj2'⟩
        · rw [moe.best_unchosen_loop.body, if_pos hjnProp]
          simp only [hlen_eq, if_pos hjnProp, hidxc, hidxl, bind_tc_ok, if_neg htk, if_neg hbn',
            if_neg hgt, haddj]
          rfl
        · refine ⟨by omega, hbestle, ?_⟩
          rw [hj2', hsc]
          exact hinveq

theorem best_unchosen_loop_spec (logits : Slice I64) (chosen : Slice Bool) (n : Usize)
    (hn : n.val = logits.val.length) (hclen : chosen.val.length = logits.val.length)
    (hinv : bestLoopInv logits chosen n n 0#i64 0#usize) :
    WP.spec (moe.best_unchosen_loop logits chosen n n 0#i64 0#usize)
      (fun b : Usize => b.val = (bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
        (fun i => chosen.val.getD i false) n.val).1) := by
  have hgoal : WP.spec
      (loop (fun (st : Usize × I64 × Usize) =>
        moe.best_unchosen_loop.body logits chosen n st.1 st.2.1 st.2.2)
        (n, 0#i64, 0#usize))
      (fun b : Usize => b.val = (bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
        (fun i => chosen.val.getD i false) n.val).1) := by
    apply loop.spec_decr_nat
      (measure := fun st : Usize × I64 × Usize => n.val - st.2.2.val)
      (inv := fun st => bestLoopInv logits chosen n st.1 st.2.1 st.2.2)
      (post := fun b : Usize => b.val = (bestScan n.val (fun i => (logits.val.getD i 0#i64).val)
        (fun i => chosen.val.getD i false) n.val).1)
    · intro st hst
      have hsti : bestLoopInv logits chosen n st.1 st.2.1 st.2.2 := hst
      obtain ⟨hle, hbestle, hinvpair⟩ := hst
      by_cases hlt : st.2.2.val < n.val
      · obtain ⟨cf, hcf, hpost⟩ := WP.spec_imp_exists
          (best_unchosen_loop_body_step logits chosen n hn hclen st.1 st.2.1 st.2.2 hlt hsti)
        obtain ⟨best2, best_val2, j2, hcfg, hinv2, hj2⟩ := hpost
        rw [hcfg] at hcf
        simp only [hcf]
        have hj2v : (best2, best_val2, j2).2.2.val = st.2.2.val + 1 := hj2
        exact ⟨hinv2, by omega⟩
      · have hnot : ¬ (st.2.2 < n) := by
          intro hc; exact hlt ((UScalar.lt_equiv _ _).1 hc)
        have hbody : moe.best_unchosen_loop.body logits chosen n st.1 st.2.1 st.2.2
            = ok (ControlFlow.done st.1) := by
          rw [moe.best_unchosen_loop.body, if_neg hnot]
        rw [hbody]
        simp only [WP.spec_ok]
        have heqj : st.2.2.val = n.val := by omega
        rw [heqj] at hinvpair
        exact congrArg Prod.fst hinvpair
    · exact hinv
  unfold moe.best_unchosen_loop
  exact hgoal

/-! ## 4. `best_unchosen`: the minimal-key unchosen index -/

theorem best_unchosen_spec (logits : Slice I64) (chosen : Slice Bool)
    (hlen : chosen.val.length = logits.val.length) :
    (∃ b : Usize, moe.best_unchosen logits chosen = ok (b, true) ∧
        b.val < logits.val.length ∧ chosen.val.getD b.val false = false ∧
        ∀ j, j < logits.val.length → chosen.val.getD j false = false →
          keyAt (fun i => (logits.val.getD i 0#i64).val) b.val ≤
            keyAt (fun i => (logits.val.getD i 0#i64).val) j)
    ∨ (moe.best_unchosen logits chosen = ok (0#usize, false) ∧
        ∀ j, j < logits.val.length → chosen.val.getD j false = true) := by
  have hn : (Slice.len logits).val = logits.val.length := Slice.len_val logits
  have hInv0 : bestLoopInv logits chosen (Slice.len logits) (Slice.len logits) 0#i64 0#usize := by
    refine ⟨by simp, le_refl _, ?_⟩
    simp [bestScan]
  obtain ⟨b, hloop, hbs⟩ := WP.spec_imp_exists
    (best_unchosen_loop_spec logits chosen (Slice.len logits) hn hlen hInv0)
  have hbval : b.val = (bestScan (Slice.len logits).val (fun i => (logits.val.getD i 0#i64).val)
      (fun i => chosen.val.getD i false) (Slice.len logits).val).1 := hbs
  by_cases hex : ∃ i < (Slice.len logits).val, chosen.val.getD i false = false
  · left
    have hS : (bestScan (Slice.len logits).val (fun i => (logits.val.getD i 0#i64).val)
        (fun i => chosen.val.getD i false) (Slice.len logits).val).1 <
        (Slice.len logits).val :=
      bestScan_lt_of_exists _ _ _ hex
    have hprops := bestScan_lt_props (Slice.len logits).val (fun i => (logits.val.getD i 0#i64).val)
      (fun i => chosen.val.getD i false) (Slice.len logits).val (le_refl _) hS
    have hblt : b < Slice.len logits :=
      (UScalar.lt_equiv b (Slice.len logits)).2 (by rw [hbval]; exact hS)
    refine ⟨b, ?_, ?_, ?_, ?_⟩
    · rw [moe.best_unchosen, hloop]
      simp only [bind_tc_ok, if_pos hblt]
    · rw [hbval, hn]; exact hS
    · rw [hbval]; exact hprops.2.1
    · intro j hj hif
      have h := hprops.2.2.2 j (by rw [hn]; exact hj) hif
      rw [hbval]
      exact h
  · right
    have hall : ∀ i < (Slice.len logits).val, chosen.val.getD i false = true := by
      intro i hi
      by_cases h : chosen.val.getD i false = true
      · exact h
      · exfalso
        exact hex ⟨i, hi, Bool.eq_false_iff.mpr h⟩
    have hS : (bestScan (Slice.len logits).val (fun i => (logits.val.getD i 0#i64).val)
        (fun i => chosen.val.getD i false) (Slice.len logits).val).1 =
        (Slice.len logits).val :=
      (bestScan_eq_sentinel _ _ _ _ (le_refl _)).mpr hall
    have hbge : ¬ b < Slice.len logits := by
      intro hc
      have hc' : b.val < (Slice.len logits).val := (UScalar.lt_equiv b (Slice.len logits)).1 hc
      rw [hbval, hS] at hc'
      exact absurd hc' (lt_irrefl _)
    refine ⟨?_, ?_⟩
    · rw [moe.best_unchosen, hloop]
      simp only [bind_tc_ok, if_neg hbge]
    · intro j hj
      exact hall j (by rw [Slice.len_val]; exact hj)

/-! ## 5. Helpers for the selection loop -/

theorem topkSel_take_take (ls : List ℤ) {m p : ℕ} (hp : p ≤ m) :
    (Nous.Model.topkSel ls m).take p = (idxOrder ls).take p := by
  rw [topkSel_eq_take, List.take_take, min_eq_left hp]

theorem list_map_val_getD (l : List I64) {i : ℕ} (hi : i < l.length) :
    (l.map (fun x => x.val)).getD i 0 = (l.getD i 0#i64).val := by
  rw [List.getD_eq_getElem _ _ (by rwa [List.length_map]), List.getElem_map,
    List.getD_eq_getElem _ _ hi]

theorem list_take_succ_getElem {α : Type} (l : List α) {p : ℕ} (hp : p < l.length) :
    l.take (p+1) = l.take p ++ [l[p]] := by
  rw [List.take_add_one, List.getElem?_eq_getElem hp]
  rfl

theorem list_mem_take_succ {α : Type} (l : List α) {p : ℕ} (hp : p < l.length) (j : α) :
    j ∈ l.take (p+1) ↔ j = l[p] ∨ j ∈ l.take p := by
  rw [list_take_succ_getElem l hp, List.mem_append, List.mem_singleton]
  tauto

theorem list_set_true_getD (L : List Bool) {b : ℕ} (hb : b < L.length) {j : ℕ}
    (hj : j < L.length) :
    (L.set b true).getD j false = true ↔ j = b ∨ L.getD j false = true := by
  have hj' : j < (L.set b true).length := by rwa [List.length_set]
  rw [List.getD_eq_getElem _ _ hj']
  by_cases hjb : j = b
  · subst hjb
    rw [List.getElem_set_self]
    simp
  · rw [List.getElem_set_ne (l := L) (i := b) (j := j) (a := true) (by omega) hj']
    rw [List.getD_eq_getElem _ _ hj]
    by_cases h : L[j] = true
    · constructor
      · intro _; exact Or.inr h
      · intro _; exact h
    · have h' : L[j] = false := Bool.eq_false_iff.mpr h
      constructor
      · intro hc; rw [h'] at hc; exact absurd hc (by simp)
      · intro hc; rcases hc with h1 | h2
        · exact absurd h1 hjb
        · rw [h'] at h2; exact absurd h2 (by simp)

theorem alloc.vec.Vec.push_eq {α : Type} (v : alloc.vec.Vec α) (x : α)
    (h : v.val.length + 1 ≤ U32.max) :
    v.push x = ok (alloc.vec.Vec.from (v.val.concat x) (by simp; scalar_tac)) := by
  have hc : (decide (v.val.length + 1 ≤ U32.max)
      || decide (v.val.length + 1 ≤ Usize.max)) = true := by
    rw [decide_eq_true h]; rfl
  unfold alloc.vec.Vec.push
  simp only [hc, ↓reduceDIte]

theorem alloc.vec.Vec.push_val {α : Type} (v : alloc.vec.Vec α) (x : α)
    (h : v.val.length + 1 ≤ U32.max) :
    ∃ v' : alloc.vec.Vec α, v.push x = ok v' ∧ v'.val = v.val ++ [x] :=
  ⟨_, alloc.vec.Vec.push_eq v x h,
    by rw [alloc.vec.Vec.from_val, List.concat_eq_append]⟩

theorem usize_cast_u32_val {b : Usize} (hb : b.val < 2^32) :
    (UScalar.cast UScalarTy.U32 b).val = b.val := by
  rw [UScalar.cast_val_eq]
  have h32 : UScalarTy.U32.numBits = 32 := rfl
  rw [h32]
  exact Nat.mod_eq_of_lt hb

/-! ## 6. The `route_topk` selection loop -/

/-- The loop invariant of `route_topk`: after `pass` passes, `sel` is the first `pass` selected
indices in selection order and `chosen` is exactly their membership mask. -/
def routeInv (logits : Slice I64) (k : Usize) (chosen : alloc.vec.Vec Bool)
    (sel : alloc.vec.Vec U32) (ok1 : Bool) (err : Option moe.MoeError) (pass : Usize) : Prop :=
  ok1 = true ∧ err = none ∧ pass.val ≤ k.val ∧
  chosen.val.length = logits.val.length ∧
  sel.val.map (fun x => x.val) =
    (Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val).take pass.val ∧
  (∀ j < logits.val.length,
    chosen.val.getD j false = true ↔
      j ∈ (Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val).take pass.val)

theorem route_topk_loop_body_step (logits : Slice I64) (k n : Usize)
    (hn : n.val = logits.val.length) (hkN : k.val ≤ n.val) (hnmax : n.val ≤ 4294967295)
    (chosen : alloc.vec.Vec Bool) (sel : alloc.vec.Vec U32) (ok1 : Bool)
    (err : Option moe.MoeError) (pass : Usize)
    (hpass : pass.val < k.val) (hinv : routeInv logits k chosen sel ok1 err pass) :
    WP.spec (moe.route_topk_loop.body logits k n chosen sel ok1 err pass)
      (fun cf => ∃ chosen2 sel2 ok2 err2 pass2,
        cf = ControlFlow.cont (chosen2, sel2, ok2, err2, pass2) ∧
        routeInv logits k chosen2 sel2 ok2 err2 pass2 ∧ pass2.val = pass.val + 1) := by
  obtain ⟨hok, herr, hpassle, hclen, hsel, hmask⟩ := hinv
  set ls : List ℤ := logits.val.map (fun x => x.val) with hls
  set T : List ℕ := Nous.Model.topkSel ls k.val with hT
  have hlenls : ls.length = logits.val.length := by rw [hls]; simp
  have hTlen : T.length = k.val := by
    rw [hT, Nous.Model.topkSel_length, hlenls, ← hn, min_eq_left hkN]
  have hpN : pass.val < logits.val.length := by omega
  have hpT : pass.val < T.length := by rw [hTlen]; exact hpass
  have hTtake : T.take pass.val = (idxOrder ls).take pass.val :=
    topkSel_take_take ls (le_of_lt hpass)
  have hidx_lt : idxAt ls pass.val < ls.length :=
    idxAt_mem_range ls (by rw [idxOrder_length]; omega)
  have hidx_ltN : idxAt ls pass.val < logits.val.length := by rw [← hlenls]; exact hidx_lt
  have hidx_notin : idxAt ls pass.val ∉ T.take pass.val := by
    rw [hTtake]; exact idxAt_not_mem_take ls (by rw [idxOrder_length]; omega)
  have hmaskF : chosen.val.getD (idxAt ls pass.val) false = false := by
    have hm := hmask (idxAt ls pass.val) hidx_ltN
    by_cases h : chosen.val.getD (idxAt ls pass.val) false = true
    · exact absurd (hm.mp h) hidx_notin
    · exact Bool.eq_false_iff.mpr h
  have hsl_val : (alloc.vec.Vec.deref chosen).val = chosen.val := by
    rw [alloc.vec.Vec.deref, Slice.from_val]
  have hsl_len : (alloc.vec.Vec.deref chosen).val.length = logits.val.length := by
    rw [hsl_val]; exact hclen
  have hbest' : ∃ b : Usize,
      moe.best_unchosen logits (alloc.vec.Vec.deref chosen) = ok (b, true) ∧
      b.val < logits.val.length ∧ chosen.val.getD b.val false = false ∧
      ∀ j, j < logits.val.length → chosen.val.getD j false = false →
        keyAt (fun i => (logits.val.getD i 0#i64).val) b.val ≤
          keyAt (fun i => (logits.val.getD i 0#i64).val) j := by
    rcases best_unchosen_spec logits (alloc.vec.Vec.deref chosen) hsl_len with h | h
    · rwa [hsl_val] at h
    · exfalso
      have hex : ∃ j, j < logits.val.length ∧ chosen.val.getD j false = false :=
        ⟨idxAt ls pass.val, hidx_ltN, hmaskF⟩
      obtain ⟨j, hj, hjf⟩ := hex
      have := h.2 j hj
      rw [hsl_val, hjf] at this
      exact absurd this (by simp)
  obtain ⟨b, hbeq, hb_lt, hb_mask, hb_min⟩ := hbest'
  have hidxs := idxAt_spec ls hpass (by rw [hlenls, ← hn]; exact hkN)
  have h1 : keyAt (fun i => (logits.val.getD i 0#i64).val) b.val ≤
      keyAt (fun i => (logits.val.getD i 0#i64).val) (idxAt ls pass.val) :=
    hb_min (idxAt ls pass.val) hidx_ltN hmaskF
  have hb_notin_idx : b.val ∉ (idxOrder ls).take pass.val := by
    intro hmem
    have hmemT : b.val ∈ T.take pass.val := by rw [hTtake]; exact hmem
    have hm := hmask b.val hb_lt
    rw [hb_mask] at hm
    exact absurd (hm.mpr hmemT) (by simp)
  have h2 : keyAt (fun i => ls.getD i 0) (idxAt ls pass.val) ≤
      keyAt (fun i => ls.getD i 0) b.val :=
    hidxs.2.2 b.val (by rw [hlenls]; exact hb_lt) hb_notin_idx
  have hkeyb : keyAt (fun i => ls.getD i 0) b.val =
      keyAt (fun i => (logits.val.getD i 0#i64).val) b.val := by
    simp only [keyAt]
    rw [show ls.getD b.val 0 = (logits.val.getD b.val 0#i64).val from by
      rw [hls, list_map_val_getD logits hb_lt]]
  have hkeyi : keyAt (fun i => ls.getD i 0) (idxAt ls pass.val) =
      keyAt (fun i => (logits.val.getD i 0#i64).val) (idxAt ls pass.val) := by
    simp only [keyAt]
    rw [show ls.getD (idxAt ls pass.val) 0 =
        (logits.val.getD (idxAt ls pass.val) 0#i64).val from by
      rw [hls, list_map_val_getD logits hidx_ltN]]
  have hkeys : keyAt (fun i => (logits.val.getD i 0#i64).val) (idxAt ls pass.val) =
      keyAt (fun i => (logits.val.getD i 0#i64).val) b.val := by
    rw [← hkeyi, ← hkeyb]
    refine le_antisymm h2 ?_
    rw [← hkeyb, ← hkeyi] at h1
    exact h1
  have hbv : b.val = idxAt ls pass.val := (Nous.Model.routerKey_inj.mp hkeys.symm).2
  have hposp : pass.val < (idxOrder ls).length := by rw [idxOrder_length, hlenls]; omega
  have hTtake_succ : T.take (pass.val+1) = T.take pass.val ++ [idxAt ls pass.val] := by
    rw [show T.take (pass.val+1) = (Nous.Model.topkSel ls k.val).take (pass.val+1) from rfl]
    rw [topkSel_take_take ls (by omega)]
    rw [list_take_succ_getElem (idxOrder ls) hposp, ← hTtake, idxAt_getElem ls hposp]
  have hpassProp : pass < k := (UScalar.lt_equiv pass k).2 hpass
  have hb32 : b.val < 2^32 := by
    have h1 : b.val < n.val := by rw [hn]; exact hb_lt
    omega
  have hcast : (UScalar.cast UScalarTy.U32 b).val = b.val := usize_cast_u32_val hb32
  have hbnd : b.val < chosen.val.length := by rw [hclen]; exact hb_lt
  have hidxmut_usize : chosen.index_mut_usize b
      = ok (chosen.val[b.val]'hbnd, chosen.set b) := by
    unfold alloc.vec.Vec.index_mut_usize
    have hindex : alloc.vec.Vec.index_usize chosen b = ok (chosen.val[b.val]'hbnd) := by
      unfold alloc.vec.Vec.index_usize
      rw [alloc.vec.Vec.getElem?_Nat_eq, List.getElem?_eq_getElem hbnd]
    rw [hindex]
    simp only [bind_tc_ok]
  have hsel_len : sel.val.length = (T.take pass.val).length := by
    have := congrArg List.length hsel
    simpa [List.length_map] using this
  have hpushbound : sel.val.length + 1 ≤ U32.max := by
    rw [hsel_len, List.length_take]
    have hmin : min pass.val T.length ≤ pass.val := min_le_left _ _
    have hu : (4294967295 : ℕ) = U32.max := U32.max_eq.symm
    omega
  obtain ⟨sel2, hpush_eq, hsel2⟩ := alloc.vec.Vec.push_val sel (UScalar.cast UScalarTy.U32 b) hpushbound
  have h1u : (1#usize).val = 1 := by decide
  have hpmax : pass.val + (1#usize).val ≤ Usize.max := by
    rw [h1u]
    have hnm : n.val ≤ Usize.max := by rw [hn]; exact Slice.property logits
    omega
  obtain ⟨pass2, haddpass, hpass2⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := pass) (y := 1#usize) hpmax)
  have hpass2' : pass2.val = pass.val + 1 := by rw [hpass2, h1u]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (chosen.set b true, sel2, true, none, pass2), ?_,
    chosen.set b true, sel2, true, none, pass2, rfl, ?_, hpass2'⟩
  · rw [moe.route_topk_loop.body, if_pos hpassProp, if_pos hok]
    simp [hbeq, hidxmut_usize, hpush_eq, haddpass, lift]
    exact herr
  · refine ⟨rfl, rfl, by omega, ?_, ?_, ?_⟩
    · rw [alloc.vec.Vec.set_val_eq, List.length_set]; exact hclen
    · rw [hsel2, List.map_append, hsel]
      simp only [List.map_singleton]
      rw [hcast, hbv, hpass2', ← hTtake_succ]
    · intro j hj
      have hj' : j < chosen.val.length := by rw [hclen]; exact hj
      rw [alloc.vec.Vec.set_val_eq, list_set_true_getD chosen.val hbnd hj']
      rw [hmask j hj]
      rw [hpass2', hTtake_succ, List.mem_append, List.mem_singleton, hbv]
      tauto

/-! ## 7. The loop and the top-level selection -/

theorem route_topk_loop_spec (logits : Slice I64) (k n : Usize)
    (hn : n.val = logits.val.length) (hkN : k.val ≤ n.val) (hnmax : n.val ≤ 4294967295)
    (chosen : alloc.vec.Vec Bool) (sel : alloc.vec.Vec U32) (ok1 : Bool)
    (err : Option moe.MoeError) (pass : Usize)
    (hinv : routeInv logits k chosen sel ok1 err pass) :
    WP.spec (moe.route_topk_loop logits k n chosen sel ok1 err pass)
      (fun b => ∃ sel2 : alloc.vec.Vec U32, b = (sel2, none) ∧
        sel2.val.map (fun x => x.val) =
          Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val) := by
  have hgoal : WP.spec
      (loop (fun (st : alloc.vec.Vec Bool × alloc.vec.Vec U32 × Bool
            × Option moe.MoeError × Usize) =>
          moe.route_topk_loop.body logits k n st.1 st.2.1 st.2.2.1 st.2.2.2.1 st.2.2.2.2)
        (chosen, sel, ok1, err, pass))
      (fun b => ∃ sel2 : alloc.vec.Vec U32, b = (sel2, none) ∧
        sel2.val.map (fun x => x.val) =
          Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val) := by
    apply loop.spec_decr_nat
      (measure := fun st : alloc.vec.Vec Bool × alloc.vec.Vec U32 × Bool
          × Option moe.MoeError × Usize => k.val - st.2.2.2.2.val)
      (inv := fun st => routeInv logits k st.1 st.2.1 st.2.2.1 st.2.2.2.1 st.2.2.2.2)
      (post := fun b => ∃ sel2 : alloc.vec.Vec U32, b = (sel2, none) ∧
        sel2.val.map (fun x => x.val) =
          Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val)
    · intro st hst
      have hsti : routeInv logits k st.1 st.2.1 st.2.2.1 st.2.2.2.1 st.2.2.2.2 := hst
      obtain ⟨hok, herr, hpassle', hclen, hsel, _hmask⟩ := hst
      by_cases hlt : st.2.2.2.2.val < k.val
      · obtain ⟨cf, hcf, hpost⟩ := WP.spec_imp_exists
          (route_topk_loop_body_step logits k n hn hkN hnmax st.1 st.2.1 st.2.2.1 st.2.2.2.1
            st.2.2.2.2 hlt hsti)
        obtain ⟨chosen2, sel2, ok2, err2, pass2, hcfg, hinv2, hp2⟩ := hpost
        rw [hcfg] at hcf
        simp only [hcf]
        have hp2v : (chosen2, sel2, ok2, err2, pass2).2.2.2.2.val = st.2.2.2.2.val + 1 := hp2
        exact ⟨hinv2, by omega⟩
      · have hnot : ¬ (st.2.2.2.2 < k) := by
          intro hc; exact hlt ((UScalar.lt_equiv _ _).1 hc)
        have hbody : moe.route_topk_loop.body logits k n st.1 st.2.1 st.2.2.1 st.2.2.2.1
            st.2.2.2.2 = ok (ControlFlow.done (st.2.1, st.2.2.2.1)) := by
          rw [moe.route_topk_loop.body, if_neg hnot]
        rw [hbody]
        simp only [WP.spec_ok]
        have hpeq : st.2.2.2.2.val = k.val := by omega
        have hTlen : (Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val).length = k.val := by
          rw [Nous.Model.topkSel_length, List.length_map, ← hn, min_eq_left hkN]
        have hfinal : st.2.1.val.map (fun x => x.val) =
            Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val := by
          rw [hsel, hpeq]
          have htake : (Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val).take k.val =
              Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val := by
            have htl := List.take_length
              (l := Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val)
            rwa [hTlen] at htl
          exact htake
        refine ⟨st.2.1, ?_, hfinal⟩
        rw [herr]
    · exact hinv
  unfold moe.route_topk_loop
  exact hgoal

theorem route_topk_spec (logits : Slice I64) (k : Usize)
    (hk0 : 1 ≤ k.val) (hkn : k.val ≤ logits.val.length)
    (hnmax : logits.val.length ≤ 4294967295) :
    ∃ sel : alloc.vec.Vec U32, moe.route_topk logits k = ok (core.result.Result.Ok sel) ∧
      sel.val.map (fun x => x.val) =
        Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val := by
  have hn : (Slice.len logits).val = logits.val.length := Slice.len_val logits
  have hkN : k.val ≤ (Slice.len logits).val := by rw [hn]; exact hkn
  have hnmax' : (Slice.len logits).val ≤ 4294967295 := by rw [hn]; exact hnmax
  have hkne0 : ¬ (k = 0#usize) := by
    intro hc; rw [hc] at hk0; simp at hk0
  have hknle : ¬ (k > Slice.len logits) := by
    intro hc
    have h := (UScalar.lt_equiv (Slice.len logits) k).1 hc
    rw [hn] at h
    omega
  have hnle : ¬ (Slice.len logits > 4294967295#usize) := by
    intro hc
    have h := (UScalar.lt_equiv (4294967295#usize) (Slice.len logits)).1 hc
    have h4 : (4294967295#usize).val = 4294967295 := by decide
    rw [h4] at h
    omega
  obtain ⟨chosen0, hfrom, hchosen0val, hchosen0len⟩ := WP.spec_imp_exists
    (alloc.vec.from_elem_spec core.clone.CloneBool false (Slice.len logits) (by rfl))
  have hInv0 : routeInv logits k chosen0 (alloc.vec.Vec.with_capacity Std.U32 k) true none 0#usize := by
    refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
    · show (0#usize).val ≤ k.val
      simp
    · exact hchosen0len.trans hn
    · show (alloc.vec.Vec.with_capacity Std.U32 k).val.map (fun x => x.val)
          = (Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val).take (0#usize).val
      simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val, List.take_zero]
    · intro j hj
      show (chosen0.val.getD j false = true) ↔
        j ∈ (Nous.Model.topkSel (logits.val.map (fun x => x.val)) k.val).take (0#usize).val
      rw [hchosen0val]
      have hj' : j < (List.replicate (Slice.len logits).val false).length := by
        rw [List.length_replicate]; rw [hn]; exact hj
      rw [List.getD_eq_getElem _ _ hj', List.getElem_replicate]
      simp [List.take_zero]
  obtain ⟨b0, hloop, hpost⟩ := WP.spec_imp_exists
    (route_topk_loop_spec logits k (Slice.len logits) hn hkN hnmax'
      chosen0 (alloc.vec.Vec.with_capacity Std.U32 k) true none 0#usize hInv0)
  obtain ⟨sel2, hb0, hsel2⟩ := hpost
  refine ⟨sel2, ?_, hsel2⟩
  rw [moe.route_topk, if_neg hkne0, if_neg hknle, if_neg hnle, hfrom]
  simp only [bind_tc_ok]
  rw [hloop, hb0]
  simp only [bind_tc_ok]
  simp

theorem route_topk_zero (logits : Slice I64) :
    moe.route_topk logits 0#usize =
      ok (core.result.Result.Err (moe.MoeError.BadK 0#usize (Slice.len logits))) := by
  rw [moe.route_topk, if_pos rfl]

theorem route_topk_bad_k (logits : Slice I64) (k : Usize) (h : Slice.len logits < k) :
    moe.route_topk logits k =
      ok (core.result.Result.Err (moe.MoeError.BadK k (Slice.len logits))) := by
  have hk0 : ¬ (k = 0#usize) := by
    intro hc
    rw [hc] at h
    simp [UScalar.lt_equiv] at h
  rw [moe.route_topk, if_neg hk0, if_pos h]

end

end apeiron_word
