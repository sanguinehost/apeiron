-- Row: `p0-count-body-get-the-a-EF9n3OLyqnX2ouQQWL8w9C` (COUNT-BODY step 2b): the extracted count helpers refine their specifications.
import ApeironWord
import Mathlib.Data.BitVec
import Mathlib.Tactic.IntervalCases
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Refinement lemmas for the five extracted count helpers

`apeiron_word.word_count_{home,admit,histogram,offsets,slot}` (`proof/ApeironWord.lean`, source
`proof/extract/lib.rs`) call the real `FreqMap<K, i64>::count_*` bodies of `src/hashtrinity/freq.rs`
that `FreqMap::count_par` composes.  This module proves each body's functional contract on the
extracted Lean (`U64` coordinate instance, `COUNT_BBITS = 8`, 256 buckets):

* `word_count_home_lt`: `home(c) = ((c·PHI) mod 2^64) / 2^56 < 256`.
* `word_count_admit_spec`: returns `n ≤ count_max`.
* `word_count_slot_spec`: returns `off[h]`, sets `off[h] := off[h] + 1`, other cursors unchanged;
  fails exactly when `h` is out of range or the cursor is `usize::MAX`.
* `word_count_histogram_spec`: `homes[i] = home(keys[i])`, `hist[b] += #{i | home(keys[i]) = b}`;
  `false` implies a bucket count would exceed `u32::MAX`.
* `word_count_offsets_spec`: `true` iff the total is `n`; exclusive prefix sums, per-worker
  offsets and the tiling of `[0, n)`.

Not covered: the `count_par` thread fan-out and the `routed`/`count_aggregate` bodies.
-/

namespace Apeiron.CountHelpers

open Aeneas Aeneas.Std Result
open apeiron_word

theorem bbits_def :
    hashtrinity.freq.FreqMapKI64.COUNT_BBITS U64.Insts.Apeiron_wordHashtrinityAlgebraCoord = 8#u32 := by
  unfold hashtrinity.freq.FreqMapKI64.COUNT_BBITS
  rfl

theorem bbits_val :
    (hashtrinity.freq.FreqMapKI64.COUNT_BBITS U64.Insts.Apeiron_wordHashtrinityAlgebraCoord).val = 8 := by
  unfold hashtrinity.freq.FreqMapKI64.COUNT_BBITS
  rfl

theorem nb_ge : 32 ≤ System.Platform.numBits := by
  rcases System.Platform.numBits_eq with h | h <;> omega

theorem usize_size_ge : 2^32 ≤ Usize.size := by
  have : Usize.size = 2 ^ System.Platform.numBits := by simp [Usize.size, Usize.numBits]
  rw [this]
  exact Nat.pow_le_pow_right (by norm_num) nb_ge

theorem phi_def : hashtrinity.PHI = 11400714819323198485#u64 := by
  unfold hashtrinity.PHI
  rfl

theorem wm_val (c : U64) :
    (core.num.U64.wrapping_mul c hashtrinity.PHI).val = c.val * 11400714819323198485 % 2^64 := by
  rw [phi_def]
  show (c.bv * (11400714819323198485#u64).bv).toNat = _
  rw [BitVec.toNat_mul]
  rfl

theorem buckets_ok :
    hashtrinity.freq.FreqMapKI64.COUNT_BUCKETS U64.Insts.Apeiron_wordHashtrinityAlgebraCoord
      = ok 256#usize := by
  unfold hashtrinity.freq.FreqMapKI64.COUNT_BUCKETS
  rw [bbits_def]
  have h : (1#usize <<< 8#u32) ⦃ r => r.val = 256 ⦄ := by
    step*
    · have := nb_ge; simp; omega
    · have := usize_size_ge
      simp at *
      rw [r_post]
      exact Nat.mod_eq_of_lt (by omega)
  obtain ⟨r, hr, hv⟩ := WP.spec_imp_exists h
  rw [hr]
  congr 1
  exact UScalar.eq_of_val_eq (by simpa using hv)

theorem home_spec (c : U64) :
    word_count_home c ⦃ r => r.val < 256 ∧ r.val = (c.val * 11400714819323198485 % 2^64) / 2^56 ⦄ := by
  unfold word_count_home hashtrinity.freq.FreqMapKI64.count_home
  rw [buckets_ok]
  step*
  · simp [bbits_val]
  · rw [bbits_val] at i1_post
    omega
  · rw [bbits_val] at i1_post
    have hwm : i.val = c.val * 11400714819323198485 % 2^64 := by rw [i_post, wm_val]
    have hi2 : i2.val = (c.val * 11400714819323198485 % 2^64) / 2^56 := by
      rw [i2_post, Nat.shiftRight_eq_div_pow, hwm]
      congr 2
    have hi2lt : i2.val < 256 := by
      rw [hi2]
      have : c.val * 11400714819323198485 % 2^64 < 2^64 := Nat.mod_lt _ (by norm_num)
      omega
    have hsz := usize_size_ge
    have hi3 : i3.val = i2.val := by
      rw [i3_post, UScalar.cast_val_eq]
      have hnb := nb_ge
      simp only [UScalarTy.numBits]
      apply Nat.mod_eq_of_lt
      calc i2.val < 2^32 := by omega
        _ ≤ _ := Nat.pow_le_pow_right (by norm_num) hnb
    have hand : (i3 &&& i5).val = i3.val &&& i5.val := UScalar.val_and _ _
    have h5 : i5.val = 255 := by omega
    have hmask : (i3 &&& i5).val = i2.val := by
      rw [hand, hi3, h5]
      have := Nat.and_two_pow_sub_one_eq_mod i2.val 8
      simp only [show (2:Nat)^8 = 256 by norm_num] at this
      rw [this]
      exact Nat.mod_eq_of_lt hi2lt
    rw [hmask]
    exact ⟨hi2lt, hi2⟩

theorem word_count_home_lt (c : U64) :
    ∃ r, word_count_home c = ok r ∧ r.val < 256 := by
  obtain ⟨r, hr, h1, _⟩ := WP.spec_imp_exists (home_spec c)
  exact ⟨r, hr, h1⟩

/-- `count_admit` returns `n ≤ count_max`, compared in `u64` (`n as u64`; `usize` is at most 64 bits). -/
theorem word_count_admit_spec (n : Usize) (count_max : U64) :
    ∃ b, word_count_admit n count_max = ok b ∧ (b = true ↔ n.val ≤ count_max.val) := by
  unfold word_count_admit hashtrinity.freq.FreqMapKI64.count_admit
  simp only [lift, bind_tc_ok]
  refine ⟨decide (UScalar.cast .U64 n ≤ count_max), rfl, ?_⟩
  have hn : (UScalar.cast .U64 n).val = n.val := by
    rw [UScalar.cast_val_eq]
    apply Nat.mod_eq_of_lt
    have h1 := n.hBounds
    simp only [UScalarTy.numBits] at h1 ⊢
    rcases System.Platform.numBits_eq with h | h <;> rw [h] at h1 <;> omega
  simp only [decide_eq_true_eq, UScalar.le_equiv, hn]

theorem slot_from_val (h : U8) : (core.convert.num.FromUsizeU8.from h).val = h.val :=
  core.convert.num.FromUsizeU8.from_val_eq h

/-- Success branch: returns `off[h]`, increments `off[h]`, leaves every other cursor. -/
theorem word_count_slot_ok (off : Slice Usize) (h : U8)
    (hlt : h.val < off.length) (hmax : off.val[h.val]!.val + 1 ≤ Usize.max) :
    ∃ s off', word_count_slot off h = ok (s, off') ∧ s = off.val[h.val]! ∧
      off'.length = off.length ∧ off'.val[h.val]!.val = off.val[h.val]!.val + 1 ∧
      ∀ i, i ≠ h.val → off'.val[i]! = off.val[i]! := by
  unfold word_count_slot hashtrinity.freq.FreqMapKI64.count_slot
  simp only [lift, bind_tc_ok]
  have hj := slot_from_val h
  generalize core.convert.num.FromUsizeU8.from h = j at hj ⊢
  obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists
    (Slice.index_usize_spec off j (by rw [hj]; exact hlt))
  have hlt' : h.val < off.val.length := hlt
  have hs' : s = off.val[h.val]! := by
    simp [hsv, hj, getElem!_pos, hlt']
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.add_spec (x := s) (y := 1#usize)
    (by rw [hs']; simpa using hmax))
  obtain ⟨off1, hup, hoff1⟩ := WP.spec_imp_exists (Slice.update_spec off j i (by rw [hj]; exact hlt))
  refine ⟨s, off1, ?_, hs', ?_, ?_, ?_⟩
  · simp only [hs, bind_tc_ok, hi, hup]
  · rw [hoff1]; simp [Slice.length]
  · rw [hoff1]
    simp [hj, hlt', getElem!_pos, hiv, hs']
  · intro k hk
    rw [hoff1]
    simp [hj, List.getElem?_set_ne (Ne.symm hk)]

theorem usize_add_one_fail (s : Usize) (h : ¬ s.val + 1 ≤ Usize.max) :
    ∃ e, s + 1#usize = fail e := by
  show ∃ e, UScalar.add s 1#usize = fail e
  unfold UScalar.add UScalar.tryMk UScalar.tryMkOpt
  have hnb : ¬ UScalar.check_bounds UScalarTy.Usize (s.val + (1#usize).val) := by
    intro hc
    have h1 := UScalar.check_bounds_imp_inBounds hc
    have h2 : s.val + 1 ≤ Usize.max := by
      scalar_tac
    exact h h2
  rw [dif_neg hnb]
  exact ⟨_, rfl⟩

/-- Refusal branch: an out-of-range cursor index or a cursor at `usize::MAX` fails. -/
theorem word_count_slot_refuse (off : Slice Usize) (h : U8)
    (hno : ¬ (h.val < off.length ∧ off.val[h.val]!.val + 1 ≤ Usize.max)) :
    ∃ e, word_count_slot off h = fail e := by
  unfold word_count_slot hashtrinity.freq.FreqMapKI64.count_slot
  simp only [lift, bind_tc_ok]
  have hj := slot_from_val h
  generalize core.convert.num.FromUsizeU8.from h = j at hj ⊢
  by_cases hlt : h.val < off.length
  · have hlt' : h.val < off.val.length := hlt
    have hmax : ¬ off.val[h.val]!.val + 1 ≤ Usize.max := fun hm => hno ⟨hlt, hm⟩
    obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists
      (Slice.index_usize_spec off j (by rw [hj]; exact hlt))
    have hs' : s = off.val[h.val]! := by
      simp [hsv, hj, getElem!_pos, hlt']
    obtain ⟨e, he⟩ := usize_add_one_fail s (by rw [hs']; exact hmax)
    exact ⟨e, by simp [hs, he]⟩
  · refine ⟨.arrayOutOfBounds, ?_⟩
    have : ¬ j.val < off.length := by rw [hj]; exact hlt
    simp [Slice.index_usize, getElem?_neg, this, Slice.length] at *

theorem word_count_slot_spec (off : Slice Usize) (h : U8) :
    ((h.val < off.length ∧ off.val[h.val]!.val + 1 ≤ Usize.max) →
      ∃ s off', word_count_slot off h = ok (s, off') ∧ s = off.val[h.val]! ∧
        off'.length = off.length ∧ off'.val[h.val]!.val = off.val[h.val]!.val + 1 ∧
        ∀ i, i ≠ h.val → off'.val[i]! = off.val[i]!) ∧
    (¬ (h.val < off.length ∧ off.val[h.val]!.val + 1 ≤ Usize.max) →
      ∃ e, word_count_slot off h = fail e) :=
  ⟨fun hc => word_count_slot_ok off h hc.1 hc.2, word_count_slot_refuse off h⟩

/-! ### Histogram -/

/-- Home bucket of a coordinate, as a natural number: `(c·PHI mod 2^64) / 2^56`. -/
def homeNat (c : Nat) : Nat := c * 11400714819323198485 % 2^64 / 2^56

/-- Number of keys of `l` whose home bucket is `k`. -/
def cntBucket (l : List U64) (k : Nat) : Nat := (l.map (fun x => homeNat x.val)).count k

theorem cnt_take_succ (l : List U64) (i : Nat) (hi : i < l.length) (k : Nat) :
    cntBucket (l.take (i+1)) k
      = cntBucket (l.take i) k + (if homeNat l[i].val = k then 1 else 0) := by
  rw [List.take_add_one, List.getElem?_eq_getElem hi]
  simp only [cntBucket, Option.toList_some, List.map_append, List.map_cons, List.map_nil,
    List.count_append, List.count_singleton]
  simp

theorem cnt_take_le (l : List U64) (i k : Nat) : cntBucket (l.take i) k ≤ cntBucket l k := by
  unfold cntBucket
  exact ((List.take_sublist i l).map _).count_le k

theorem get!_set {α : Type} [Inhabited α] (l : List α) (i j : Nat) (x : α) (h : i < l.length) :
    (l.set i x)[j]! = if j = i then x else l[j]! := by
  simp only [List.getElem!_eq_getElem?_getD, List.getElem?_set, h]
  by_cases hji : j = i
  · subst hji; simp
  · simp [hji, Ne.symm hji]

/-- Histogram loop invariant after `i` keys. -/
def HistInv (keys : Slice U64) (hist0 : Slice U32) (homes : Slice U8) (hist : Slice U32)
    (ok : Bool) (i : Nat) : Prop :=
  i ≤ keys.length ∧ homes.length = keys.length ∧ hist.length = 256 ∧
  (ok = true →
    (∀ j, j < i → (homes.val[j]!).val = homeNat (keys.val[j]!).val) ∧
    ∀ k, k < 256 → (hist.val[k]!).val = (hist0.val[k]!).val + cntBucket (keys.val.take i) k) ∧
  (ok = false → ∃ k, k < 256 ∧ U32.max < (hist0.val[k]!).val + cntBucket keys.val k)

theorem hist_body_step (keys : Slice U64) (hist0 : Slice U32) (homes : Slice U8)
    (hist : Slice U32) (i : Usize)
    (hinv : HistInv keys hist0 homes hist true i.val) (hlt : i.val < keys.length) :
    ∃ homes' hist' ok' i',
      hashtrinity.freq.FreqMapKI64.count_histogram_loop.body
        U64.Insts.Apeiron_wordHashtrinityAlgebraCoord keys (Slice.len keys) homes hist true i
        = ok (ControlFlow.cont (homes', hist', ok', i')) ∧
      i'.val = i.val + 1 ∧ HistInv keys hist0 homes' hist' ok' i'.val := by
  obtain ⟨hile, hhl, hh256, hok, -⟩ := hinv
  obtain ⟨hhom, hhist⟩ := hok rfl
  unfold hashtrinity.freq.FreqMapKI64.count_histogram_loop.body
  simp only [if_true]
  rw [if_pos (by rw [UScalar.lt_equiv]; simpa [Slice.len] using hlt)]
  obtain ⟨t, ht, htv⟩ := WP.spec_imp_exists (Slice.index_usize_spec keys i hlt)
  have hcoord : U64.Insts.Apeiron_wordHashtrinityAlgebraCoord.coord t = ok t := rfl
  obtain ⟨h, hh, hh256', hhv⟩ := WP.spec_imp_exists (home_spec t)
  have hh' : hashtrinity.freq.FreqMapKI64.count_home U64.Insts.Apeiron_wordHashtrinityAlgebraCoord t
      = ok h := hh
  have hhn : h.val = homeNat (keys.val[i.val]'hlt).val := by rw [hhv, htv]; rfl
  have hihom : i.val < homes.length := by omega
  obtain ⟨⟨px, pb⟩, hp, hpx, hpb⟩ := WP.spec_imp_exists (Slice.index_mut_usize_spec homes i hihom)
  have h8 : (UScalar.cast .U8 h).val = h.val := by
    rw [UScalar.cast_val_eq]; simp only [UScalarTy.numBits]; omega
  have hhb : h.val < hist.length := by omega
  obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists (Slice.index_usize_spec hist h hhb)
  obtain ⟨i4, hi4, hi4v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) (by scalar_tac))
  simp only [ht, hcoord, bind_tc_ok, hh', hp, hi3, lift]
  have hv := U32.checked_add_bv_spec i3 1#u32
  have hi3n : i3.val = (hist.val[h.val]!).val := by
    rw [hi3v]; simp [getElem!_pos, hhb]
  have hhist_h := hhist h.val (by omega)
  have hi4v' : i4.val = i.val + 1 := by rw [hi4v]; simp
  have hkey : keys.val[i.val]! = keys.val[i.val]'hlt := by simp [getElem!_pos, hlt]
  have hhomlen : (pb (UScalar.cast .U8 h)).length = homes.length := by
    rw [hpb]; simp [Slice.length]
  have hhom' : ∀ j, j < i.val + 1 →
      ((pb (UScalar.cast .U8 h)).val[j]!).val = homeNat (keys.val[j]!).val := by
    intro j hj
    rw [hpb, Slice.set_val_eq, get!_set _ _ _ _ (by simpa [Slice.length] using hihom)]
    by_cases hji : j = i.val
    · subst hji
      simp only [if_true]
      rw [h8, hhn, hkey]
    · rw [if_neg hji]
      exact hhom j (by omega)
  have hcnt_succ := cnt_take_succ keys.val i.val hlt h.val
  have hcnt_le := cnt_take_le keys.val (i.val + 1) h.val
  have hcn : homeNat (keys.val[i.val]'hlt).val = h.val := hhn.symm
  rcases hc : i3.checked_add 1#u32 with _ | v
  · rw [hc] at hv
    simp only [hi4, bind_tc_ok]
    refine ⟨pb (UScalar.cast .U8 h), hist, false, i4, rfl, hi4v', ?_⟩
    refine ⟨by omega, by omega, hh256, by simp, fun _ => ⟨h.val, hh256', ?_⟩⟩
    simp only [hcn, if_true] at hcnt_succ
    have hv' : U32.max < i3.val + (1#u32).val := hv
    have h1u : (1#u32).val = 1 := rfl
    omega
  · rw [hc] at hv
    obtain ⟨hv1, hv2, -⟩ := hv
    obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists (Slice.update_spec hist h v hhb)
    simp only [hs, hi4, bind_tc_ok]
    refine ⟨pb (UScalar.cast .U8 h), s, true, i4, rfl, hi4v', ?_⟩
    refine ⟨by omega, by omega, ?_, fun _ => ⟨fun j hj => hhom' j (by omega), ?_⟩, by simp⟩
    · rw [hsv]; simp [Slice.length, hh256]
    · intro k hk
      rw [hsv, Slice.set_val_eq, get!_set _ _ _ _ (by simpa [Slice.length] using hhb), hi4v']
      have hck := cnt_take_succ keys.val i.val hlt k
      rw [hcn] at hck
      have h1u : (1#u32).val = 1 := rfl
      by_cases hkh : k = h.val
      · subst hkh
        simp only [if_true] at hck ⊢
        omega
      · rw [if_neg hkh]
        have : ¬ h.val = k := fun e => hkh e.symm
        simp only [this, if_false] at hck
        rw [hck]
        exact hhist k hk

theorem hist_loop_spec (keys : Slice U64) (hist0 : Slice U32) (homes : Slice U8)
    (hist : Slice U32) (ok1 : Bool) (i : Usize)
    (hinv : HistInv keys hist0 homes hist ok1 i.val) :
    ∃ b homes' hist',
      hashtrinity.freq.FreqMapKI64.count_histogram_loop
        U64.Insts.Apeiron_wordHashtrinityAlgebraCoord keys homes hist (Slice.len keys) ok1 i
        = ok (b, homes', hist') ∧ HistInv keys hist0 homes' hist' b keys.length := by
  have hspec : hashtrinity.freq.FreqMapKI64.count_histogram_loop
        U64.Insts.Apeiron_wordHashtrinityAlgebraCoord keys homes hist (Slice.len keys) ok1 i
      ⦃ r => HistInv keys hist0 r.2.1 r.2.2 r.1 keys.length ⦄ := by
    unfold hashtrinity.freq.FreqMapKI64.count_histogram_loop
    apply loop.spec_decr_nat
      (measure := fun s : Slice U8 × Slice U32 × Bool × Usize => keys.length - s.2.2.2.val)
      (inv := fun s => HistInv keys hist0 s.1 s.2.1 s.2.2.1 s.2.2.2.val)
      (post := fun r : Bool × Slice U8 × Slice U32 => HistInv keys hist0 r.2.1 r.2.2 r.1 keys.length)
    · rintro ⟨homesX, histX, okX, iX⟩ hI
      dsimp only at hI ⊢
      cases okX
      · have hb : hashtrinity.freq.FreqMapKI64.count_histogram_loop.body
            U64.Insts.Apeiron_wordHashtrinityAlgebraCoord keys (Slice.len keys) homesX histX false iX
            = ok (ControlFlow.done (false, homesX, histX)) := by
          unfold hashtrinity.freq.FreqMapKI64.count_histogram_loop.body
          simp
        simp only [hb, Aeneas.Std.WP.spec_ok]
        obtain ⟨-, h2, h3, -, h5⟩ := hI
        exact ⟨le_refl _, h2, h3, by simp, h5⟩
      · by_cases hlt : iX.val < keys.length
        · obtain ⟨homes', hist', ok', i', hb, hi', hI'⟩ :=
            hist_body_step keys hist0 homesX histX iX hI hlt
          simp only [hb, Aeneas.Std.WP.spec_ok]
          exact ⟨hI', by omega⟩
        · have hb : hashtrinity.freq.FreqMapKI64.count_histogram_loop.body
              U64.Insts.Apeiron_wordHashtrinityAlgebraCoord keys (Slice.len keys) homesX histX true iX
              = ok (ControlFlow.done (true, homesX, histX)) := by
            unfold hashtrinity.freq.FreqMapKI64.count_histogram_loop.body
            rw [if_pos rfl, if_neg (by rw [UScalar.lt_equiv]; simpa [Slice.len] using hlt)]
          simp only [hb, Aeneas.Std.WP.spec_ok]
          obtain ⟨h1, h2, h3, h4, -⟩ := hI
          have hn : iX.val = keys.length := le_antisymm h1 (Nat.le_of_not_lt hlt)
          refine ⟨le_refl _, h2, h3, fun _ => ?_, by simp⟩
          rw [hn] at h4
          exact h4 rfl
    · exact hinv
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  exact ⟨r.1, r.2.1, r.2.2, hr, hpost⟩

theorem hist_loop_refused (keys : Slice U64) (homes : Slice U8) (hist : Slice U32) (i : Usize) :
    hashtrinity.freq.FreqMapKI64.count_histogram_loop
      U64.Insts.Apeiron_wordHashtrinityAlgebraCoord keys homes hist (Slice.len keys) false i
      = ok (false, homes, hist) := by
  have hspec : hashtrinity.freq.FreqMapKI64.count_histogram_loop
        U64.Insts.Apeiron_wordHashtrinityAlgebraCoord keys homes hist (Slice.len keys) false i
      ⦃ r => r = (false, homes, hist) ⦄ := by
    unfold hashtrinity.freq.FreqMapKI64.count_histogram_loop
    apply loop.spec_decr_nat
      (measure := fun s : Slice U8 × Slice U32 × Bool × Usize => keys.length - s.2.2.2.val)
      (inv := fun s => s = (homes, hist, false, i))
      (post := fun r : Bool × Slice U8 × Slice U32 => r = (false, homes, hist))
    · rintro ⟨homesX, histX, okX, iX⟩ hI
      simp only [Prod.mk.injEq] at hI
      obtain ⟨rfl, rfl, rfl, rfl⟩ := hI
      have hb : hashtrinity.freq.FreqMapKI64.count_histogram_loop.body
          U64.Insts.Apeiron_wordHashtrinityAlgebraCoord keys (Slice.len keys) homesX histX false iX
          = ok (ControlFlow.done (false, homesX, histX)) := by
        unfold hashtrinity.freq.FreqMapKI64.count_histogram_loop.body
        simp
      simp only [hb, Aeneas.Std.WP.spec_ok]
    · rfl
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  rw [hr, hpost]

/-- `count_histogram` on well-sized slices: `homes'[i] = home(keys[i])` and every bucket count is
`hist[b] + #{i | home(keys[i]) = b}` on `true`; on `false` some bucket would exceed `u32::MAX`.
Wrong-sized slices refuse with both slices unchanged. -/
theorem word_count_histogram_spec (keys : Slice U64) (homes : Slice U8) (hist : Slice U32) :
    (homes.length = keys.length ∧ hist.length = 256 →
      ∃ b homes' hist', word_count_histogram keys homes hist = ok (b, homes', hist') ∧
        homes'.length = keys.length ∧ hist'.length = 256 ∧
        (b = true →
          (∀ j, j < keys.length → (homes'.val[j]!).val = homeNat (keys.val[j]!).val) ∧
          ∀ k, k < 256 → (hist'.val[k]!).val = (hist.val[k]!).val + cntBucket keys.val k) ∧
        (b = false → ∃ k, k < 256 ∧ U32.max < (hist.val[k]!).val + cntBucket keys.val k)) ∧
    (¬ (homes.length = keys.length ∧ hist.length = 256) →
      word_count_histogram keys homes hist = ok (false, homes, hist)) := by
  unfold word_count_histogram hashtrinity.freq.FreqMapKI64.count_histogram
  simp only [buckets_ok, bind_tc_ok]
  have e1 : (homes.len = keys.len) ↔ homes.length = keys.length := by
    rw [UScalar.eq_equiv]; simp
  have e2 : (hist.len = 256#usize) ↔ hist.length = 256 := by
    rw [UScalar.eq_equiv]; simp
  by_cases hc : homes.length = keys.length ∧ hist.length = 256
  · have hok : (if homes.len = keys.len then ok (decide (hist.len = 256#usize)) else ok false)
        = ok true := by
      rw [if_pos (e1.mpr hc.1)]
      simp [e2.mpr hc.2]
    refine ⟨fun _ => ?_, fun hn => absurd hc hn⟩
    simp only [hok, bind_tc_ok]
    have hinv : HistInv keys hist homes hist true (0#usize).val := by
      refine ⟨by simp, hc.1, hc.2, fun _ => ⟨fun j hj => by simp at hj, fun k hk => ?_⟩, by simp⟩
      simp [cntBucket]
    obtain ⟨b, homes', hist', hr, hI⟩ := hist_loop_spec keys hist homes hist true 0#usize hinv
    obtain ⟨-, h2, h3, h4, h5⟩ := hI
    refine ⟨b, homes', hist', hr, h2, h3, fun hb => ?_, h5⟩
    obtain ⟨g1, g2⟩ := h4 hb
    exact ⟨g1, fun k hk => by rw [g2 k hk, List.take_length]⟩
  · have hok : (if homes.len = keys.len then ok (decide (hist.len = 256#usize)) else ok false)
        = ok false := by
      by_cases h1 : homes.len = keys.len
      · rw [if_pos h1]
        have : ¬ hist.len = 256#usize := fun h2 => hc ⟨e1.mp h1, e2.mp h2⟩
        simp [this]
      · rw [if_neg h1]
    refine ⟨fun h => absurd h hc, fun _ => ?_⟩
    simp only [hok, bind_tc_ok]
    exact hist_loop_refused keys homes hist 0#usize

/-! ### Offsets -/

/-- `counts[t·256 + j]` as a natural number: worker `t`, bucket `j`. -/
def cC (counts : Slice U32) (t j : Nat) : Nat := (counts.val[t*256+j]!).val

/-- Column sum of bucket `j` over `w` workers. -/
def colsum (counts : Slice U32) (w j : Nat) : Nat := ∑ t ∈ Finset.range w, cC counts t j

/-- Exclusive prefix sum of the column sums over buckets `< j`. -/
def colsA (counts : Slice U32) (w j : Nat) : Nat := ∑ j' ∈ Finset.range j, colsum counts w j'

theorem colsA_succ (counts : Slice U32) (w j : Nat) :
    colsA counts w (j+1) = colsA counts w j + colsum counts w j := by
  unfold colsA
  rw [Finset.sum_range_succ]

theorem colsA_mono (counts : Slice U32) (w : Nat) {j j' : Nat} (h : j ≤ j') :
    colsA counts w j ≤ colsA counts w j' := by
  unfold colsA
  exact Finset.sum_le_sum_of_subset (Finset.range_mono h)

theorem colsum_prefix_le (counts : Slice U32) (w j t : Nat) (h : t ≤ w) :
    ∑ t' ∈ Finset.range t, cC counts t' j ≤ colsum counts w j := by
  unfold colsum
  exact Finset.sum_le_sum_of_subset (Finset.range_mono h)

/-- Any prefix of the traversal is at most the total. -/
theorem prefix_le_total (counts : Slice U32) (w j t : Nat) (hj : j < 256) (ht : t ≤ w) :
    colsA counts w j + ∑ t' ∈ Finset.range t, cC counts t' j ≤ colsA counts w 256 := by
  have h1 := colsum_prefix_le counts w j t ht
  have h2 := colsA_succ counts w j
  have h3 := colsA_mono counts w (show j + 1 ≤ 256 by omega)
  omega

/-- Offsets already written for buckets `< j`. -/
def WoffCols (counts : Slice U32) (w j : Nat) (woff : Slice Usize) : Prop :=
  ∀ j', j' < j → ∀ t, t < w →
    (woff.val[t*256+j']!).val = colsA counts w j' + ∑ t' ∈ Finset.range t, cC counts t' j'

/-- Inner (worker) loop invariant for bucket `j` after `t` workers. -/
def InInv (counts : Slice U32) (w j : Nat) (woff : Slice Usize) (ok : Bool) (acc : Usize)
    (t : Nat) : Prop :=
  t ≤ w ∧ woff.length = w*256 ∧
  (ok = true →
    acc.val = colsA counts w j + ∑ t' ∈ Finset.range t, cC counts t' j ∧
    WoffCols counts w j woff ∧
    ∀ t', t' < t → (woff.val[t'*256+j]!).val
      = colsA counts w j + ∑ t'' ∈ Finset.range t', cC counts t'' j) ∧
  (ok = false → Usize.max < colsA counts w 256)

theorem cast_u32_usize (x : U32) : (UScalar.cast .Usize x).val = x.val := by
  rw [UScalar.cast_val_eq]
  simp only [UScalarTy.numBits]
  apply Nat.mod_eq_of_lt
  have h1 := x.hBounds
  simp only [UScalarTy.numBits] at h1
  have h2 : 2^32 ≤ 2^System.Platform.numBits := Nat.pow_le_pow_right (by norm_num) nb_ge
  omega

theorem off_inner_step (counts : Slice U32) (w j : Usize) (woff : Slice Usize) (acc t : Usize)
    (hcl : counts.length = w.val*256) (hw : w.val*256 ≤ Usize.max) (hj : j.val < 256)
    (hI : InInv counts w.val j.val woff true acc t.val) (hlt : t.val < w.val) :
    ∃ woff' ok' acc' t',
      hashtrinity.freq.FreqMapKI64.count_offsets_loop0_loop0.body counts w 256#usize j woff true acc t
        = ok (ControlFlow.cont (woff', ok', acc', t')) ∧
      t'.val = t.val + 1 ∧ InInv counts w.val j.val woff' ok' acc' t'.val := by
  obtain ⟨hte, hwl, hok, -⟩ := hI
  obtain ⟨hacc, hcols, hcur⟩ := hok rfl
  unfold hashtrinity.freq.FreqMapKI64.count_offsets_loop0_loop0.body
  simp only [if_true]
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt)]
  have h256 : (256#usize).val = 256 := rfl
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := t) (y := 256#usize) (by rw [h256]; scalar_tac))
  rw [h256] at hiv
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := j) (by scalar_tac))
  obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists
    (Slice.update_spec woff i1 acc (by rw [hi1v, hiv]; omega))
  have hi2 : i1.val < counts.length := by rw [hi1v, hiv]; omega
  obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists (Slice.index_usize_spec counts i1 hi2)
  have hidx : i1.val = t.val*256 + j.val := by rw [hi1v, hiv]
  have hi3n : i3.val = cC counts t.val j.val := by
    rw [hi3v]; unfold cC
    rw [getElem!_pos counts.val _ (by simpa [Slice.length] using (hidx ▸ hi2))]
    simp only [hidx]
  have hi4 : (UScalar.cast .Usize i3).val = cC counts t.val j.val := by
    rw [cast_u32_usize, hi3n]
  obtain ⟨t1, ht1, ht1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := t) (y := 1#usize) (by scalar_tac))
  have ht1v' : t1.val = t.val + 1 := by rw [ht1v]; simp
  have hv := Usize.checked_add_bv_spec acc (UScalar.cast .Usize i3)
  have hsub := prefix_le_total counts w.val j.val (t.val + 1) hj (by omega)
  have hsum := Finset.sum_range_succ (fun t' => cC counts t' j.val) t.val
  have hsl : s.length = w.val * 256 := by rw [hsv]; simp [Slice.length, hwl, Slice.set_val_eq]
  simp only [hi, hi1, hs, hi3, lift, bind_tc_ok]
  rcases hc : acc.checked_add (UScalar.cast .Usize i3) with _ | v
  · rw [hc] at hv
    simp only [bind_tc_ok, ht1]
    refine ⟨s, false, acc, t1, rfl, ht1v', ?_⟩
    refine ⟨by omega, hsl, by simp, fun _ => ?_⟩
    simp only at hv
    rw [hi4] at hv
    omega
  · rw [hc] at hv
    obtain ⟨hv1, hv2, -⟩ := hv
    simp only [bind_tc_ok, ht1]
    refine ⟨s, true, v, t1, rfl, ht1v', ?_⟩
    have hbound : i1.val < woff.val.length := by rw [hidx]; simp [Slice.length] at hwl; omega
    have hsget : ∀ k, (s.val[k]!) = if k = i1.val then acc else woff.val[k]! := by
      intro k; rw [hsv, Slice.set_val_eq]; exact get!_set _ _ _ _ hbound
    refine ⟨by omega, hsl, fun _ => ⟨?_, ?_, ?_⟩, by simp⟩
    · rw [hv2, hi4, ht1v', hacc, hsum]; ring
    · intro j' hj' t0 ht0
      rw [hsget, if_neg (by rw [hidx]; omega)]
      exact hcols j' hj' t0 ht0
    · intro t' ht'
      rw [hsget]
      by_cases hte' : t' = t.val
      · subst hte'
        rw [if_pos hidx.symm]
        exact hacc
      · rw [if_neg (by rw [hidx]; omega)]
        exact hcur t' (by omega)

theorem off_inner_loop (counts : Slice U32) (w j : Usize) (woff : Slice Usize) (ok1 : Bool)
    (acc t : Usize) (hcl : counts.length = w.val*256) (hw : w.val*256 ≤ Usize.max)
    (hj : j.val < 256) (hI : InInv counts w.val j.val woff ok1 acc t.val) :
    ∃ woff' ok' acc',
      hashtrinity.freq.FreqMapKI64.count_offsets_loop0_loop0 counts w woff 256#usize ok1 acc j t
        = ok (woff', ok', acc') ∧ InInv counts w.val j.val woff' ok' acc' w.val := by
  have hspec : hashtrinity.freq.FreqMapKI64.count_offsets_loop0_loop0 counts w woff 256#usize ok1 acc j t
      ⦃ r => InInv counts w.val j.val r.1 r.2.1 r.2.2 w.val ⦄ := by
    unfold hashtrinity.freq.FreqMapKI64.count_offsets_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun s : Slice Usize × Bool × Usize × Usize => w.val - s.2.2.2.val)
      (inv := fun s => InInv counts w.val j.val s.1 s.2.1 s.2.2.1 s.2.2.2.val)
      (post := fun r : Slice Usize × Bool × Usize => InInv counts w.val j.val r.1 r.2.1 r.2.2 w.val)
    · rintro ⟨woffX, okX, accX, tX⟩ hI'
      dsimp only at hI' ⊢
      cases okX
      · have hb : hashtrinity.freq.FreqMapKI64.count_offsets_loop0_loop0.body counts w 256#usize j
            woffX false accX tX = ok (ControlFlow.done (woffX, false, accX)) := by
          unfold hashtrinity.freq.FreqMapKI64.count_offsets_loop0_loop0.body
          simp
        simp only [hb, Aeneas.Std.WP.spec_ok]
        obtain ⟨-, h2, -, h4⟩ := hI'
        exact ⟨le_refl _, h2, by simp, h4⟩
      · by_cases hlt : tX.val < w.val
        · obtain ⟨woff', ok', acc', t', hb, ht', hI''⟩ :=
            off_inner_step counts w j woffX accX tX hcl hw hj hI' hlt
          simp only [hb, Aeneas.Std.WP.spec_ok]
          exact ⟨hI'', by omega⟩
        · have hb : hashtrinity.freq.FreqMapKI64.count_offsets_loop0_loop0.body counts w 256#usize j
              woffX true accX tX = ok (ControlFlow.done (woffX, true, accX)) := by
            unfold hashtrinity.freq.FreqMapKI64.count_offsets_loop0_loop0.body
            rw [if_pos rfl, if_neg (by rw [UScalar.lt_equiv]; exact hlt)]
          simp only [hb, Aeneas.Std.WP.spec_ok]
          obtain ⟨h1, h2, h3, h4⟩ := hI'
          have hn : tX.val = w.val := le_antisymm h1 (Nat.le_of_not_lt hlt)
          rw [hn] at h3
          exact ⟨le_refl _, h2, h3, by simp⟩
    · exact hI
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  exact ⟨r.1, r.2.1, r.2.2, hr, hpost⟩

/-- Outer (bucket) loop invariant after `j` buckets. -/
def OutInv (counts : Slice U32) (w : Nat) (bstart woff : Slice Usize) (ok : Bool) (acc : Usize)
    (j : Nat) : Prop :=
  j ≤ 256 ∧ bstart.length = 257 ∧ woff.length = w*256 ∧
  (ok = true →
    acc.val = colsA counts w j ∧
    (∀ j', j' < j → (bstart.val[j']!).val = colsA counts w j') ∧
    WoffCols counts w j woff) ∧
  (ok = false → Usize.max < colsA counts w 256)

theorem off_outer_step (counts : Slice U32) (w : Usize) (bstart woff : Slice Usize)
    (acc j : Usize) (hcl : counts.length = w.val*256) (hw : w.val*256 ≤ Usize.max)
    (hI : OutInv counts w.val bstart woff true acc j.val) (hlt : j.val < 256) :
    ∃ bstart' woff' ok' acc' j',
      hashtrinity.freq.FreqMapKI64.count_offsets_loop0.body counts w 256#usize bstart woff true acc j
        = ok (ControlFlow.cont (bstart', woff', ok', acc', j')) ∧
      j'.val = j.val + 1 ∧ OutInv counts w.val bstart' woff' ok' acc' j'.val := by
  obtain ⟨hje, hbl, hwl, hok, -⟩ := hI
  obtain ⟨hacc, hbs, hcols⟩ := hok rfl
  unfold hashtrinity.freq.FreqMapKI64.count_offsets_loop0.body
  simp only [if_true]
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt)]
  obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists
    (Slice.update_spec bstart j acc (by omega))
  have hin : InInv counts w.val j.val woff true acc (0#usize).val := by
    refine ⟨by simp, hwl, fun _ => ⟨by simpa using hacc, hcols, fun t' ht' => by simp at ht'⟩, by simp⟩
  obtain ⟨woff1, ok2, acc1, hinner, hI2⟩ :=
    off_inner_loop counts w j woff true acc 0#usize hcl hw hlt hin
  obtain ⟨j1, hj1, hj1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := j) (y := 1#usize) (by scalar_tac))
  have hj1v' : j1.val = j.val + 1 := by rw [hj1v]; simp
  simp only [hs, hinner, hj1, bind_tc_ok]
  refine ⟨s, woff1, ok2, acc1, j1, rfl, hj1v', ?_⟩
  obtain ⟨-, hwl2, hok2, hfal⟩ := hI2
  have hsl : s.length = 257 := by rw [hsv]; simp [Slice.length, hbl, Slice.set_val_eq]
  refine ⟨by omega, hsl, hwl2, fun hok2t => ?_, hfal⟩
  obtain ⟨hacc2, hcols2, hcur2⟩ := hok2 hok2t
  have hbound : j.val < bstart.val.length := by simp [Slice.length] at hbl; omega
  refine ⟨?_, ?_, ?_⟩
  · rw [hj1v', colsA_succ, hacc2]
    rfl
  · intro j' hj'
    rw [hsv, Slice.set_val_eq, get!_set _ _ _ _ hbound]
    by_cases hjj : j' = j.val
    · subst hjj; simp [hacc]
    · rw [if_neg hjj]; exact hbs j' (by omega)
  · intro j' hj' t ht
    by_cases hjj : j' = j.val
    · subst hjj; exact hcur2 t ht
    · exact hcols2 j' (by omega) t ht

theorem off_outer_loop (counts : Slice U32) (w : Usize) (bstart woff : Slice Usize)
    (ok1 : Bool) (acc j : Usize) (hcl : counts.length = w.val*256) (hw : w.val*256 ≤ Usize.max)
    (hI : OutInv counts w.val bstart woff ok1 acc j.val) :
    ∃ bstart' woff' ok' acc',
      hashtrinity.freq.FreqMapKI64.count_offsets_loop0 counts w bstart woff 256#usize ok1 acc j
        = ok (bstart', woff', ok', acc') ∧ OutInv counts w.val bstart' woff' ok' acc' 256 := by
  have hspec : hashtrinity.freq.FreqMapKI64.count_offsets_loop0 counts w bstart woff 256#usize ok1 acc j
      ⦃ r => OutInv counts w.val r.1 r.2.1 r.2.2.1 r.2.2.2 256 ⦄ := by
    unfold hashtrinity.freq.FreqMapKI64.count_offsets_loop0
    apply loop.spec_decr_nat
      (measure := fun s : Slice Usize × Slice Usize × Bool × Usize × Usize => 256 - s.2.2.2.2.val)
      (inv := fun s => OutInv counts w.val s.1 s.2.1 s.2.2.1 s.2.2.2.1 s.2.2.2.2.val)
      (post := fun r : Slice Usize × Slice Usize × Bool × Usize =>
        OutInv counts w.val r.1 r.2.1 r.2.2.1 r.2.2.2 256)
    · rintro ⟨bX, woX, okX, accX, jX⟩ hI'
      dsimp only at hI' ⊢
      cases okX
      · have hb : hashtrinity.freq.FreqMapKI64.count_offsets_loop0.body counts w 256#usize bX woX
            false accX jX = ok (ControlFlow.done (bX, woX, false, accX)) := by
          unfold hashtrinity.freq.FreqMapKI64.count_offsets_loop0.body
          simp
        simp only [hb, Aeneas.Std.WP.spec_ok]
        obtain ⟨-, h2, h3, -, h5⟩ := hI'
        exact ⟨le_refl _, h2, h3, by simp, h5⟩
      · by_cases hlt : jX.val < 256
        · obtain ⟨b', wo', ok', acc', j', hb, hj', hI''⟩ :=
            off_outer_step counts w bX woX accX jX hcl hw hI' hlt
          simp only [hb, Aeneas.Std.WP.spec_ok]
          exact ⟨hI'', by omega⟩
        · have hb : hashtrinity.freq.FreqMapKI64.count_offsets_loop0.body counts w 256#usize bX woX
              true accX jX = ok (ControlFlow.done (bX, woX, true, accX)) := by
            unfold hashtrinity.freq.FreqMapKI64.count_offsets_loop0.body
            rw [if_pos rfl, if_neg (by rw [UScalar.lt_equiv]; exact hlt)]
          simp only [hb, Aeneas.Std.WP.spec_ok]
          obtain ⟨h1, h2, h3, h4, -⟩ := hI'
          have hn : jX.val = 256 := le_antisymm h1 (Nat.le_of_not_lt hlt)
          rw [hn] at h4
          exact ⟨le_refl _, h2, h3, h4, by simp⟩
    · exact hI
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  exact ⟨r.1, r.2.1, r.2.2.1, r.2.2.2, hr, hpost⟩

theorem off_outer_refused (counts : Slice U32) (w : Usize) (bstart woff : Slice Usize)
    (acc j : Usize) :
    hashtrinity.freq.FreqMapKI64.count_offsets_loop0 counts w bstart woff 256#usize false acc j
      = ok (bstart, woff, false, acc) := by
  have hspec : hashtrinity.freq.FreqMapKI64.count_offsets_loop0 counts w bstart woff 256#usize false acc j
      ⦃ r => r = (bstart, woff, false, acc) ⦄ := by
    unfold hashtrinity.freq.FreqMapKI64.count_offsets_loop0
    apply loop.spec_decr_nat
      (measure := fun s : Slice Usize × Slice Usize × Bool × Usize × Usize => 256 - s.2.2.2.2.val)
      (inv := fun s => s = (bstart, woff, false, acc, j))
      (post := fun r : Slice Usize × Slice Usize × Bool × Usize => r = (bstart, woff, false, acc))
    · rintro ⟨bX, woX, okX, accX, jX⟩ hI
      simp only [Prod.mk.injEq] at hI
      obtain ⟨rfl, rfl, rfl, rfl, rfl⟩ := hI
      have hb : hashtrinity.freq.FreqMapKI64.count_offsets_loop0.body counts w 256#usize bX woX
          false accX jX = ok (ControlFlow.done (bX, woX, false, accX)) := by
        unfold hashtrinity.freq.FreqMapKI64.count_offsets_loop0.body
        simp
      simp only [hb, Aeneas.Std.WP.spec_ok]
    · rfl
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  rw [hr, hpost]

theorem word_count_offsets_spec_main (counts : Slice U32) (w n : Usize) (bstart woff : Slice Usize)
    (hw : w.val*256 ≤ Usize.max) (hc : counts.length = w.val*256)
    (hwo : woff.length = w.val*256) (hb : bstart.length = 257) :
    ∃ b bstart' woff', word_count_offsets counts w n bstart woff = ok (b, bstart', woff') ∧
      bstart'.length = 257 ∧ woff'.length = w.val*256 ∧
      (b = true ↔ colsA counts w.val 256 = n.val) ∧
      (b = true →
        (∀ j, j ≤ 256 → (bstart'.val[j]!).val = colsA counts w.val j) ∧
        ∀ j, j < 256 → ∀ t, t < w.val →
          (woff'.val[t*256+j]!).val = colsA counts w.val j + ∑ t' ∈ Finset.range t, cC counts t' j) := by
  unfold word_count_offsets hashtrinity.freq.FreqMapKI64.count_offsets
  simp only [buckets_ok, bind_tc_ok, lift]
  have hv := Usize.checked_mul_bv_spec w 256#usize
  rcases hm : w.checked_mul 256#usize with _ | x
  · rw [hm] at hv
    exfalso
    have : (256#usize).val = 256 := rfl
    simp only [this] at hv
    omega
  · rw [hm] at hv
    obtain ⟨-, hxv, -⟩ := hv
    have h256 : (256#usize).val = 256 := rfl
    rw [h256] at hxv
    have e1 : counts.len = x := by rw [UScalar.eq_equiv]; simp [hxv, hc]
    have e2 : woff.len = x := by rw [UScalar.eq_equiv]; simp [hxv, hwo]
    obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := 256#usize) (y := 1#usize) (by scalar_tac))
    have e3 : bstart.len = i3 := by rw [UScalar.eq_equiv]; simp [hi3v, hb, h256]
    have hok1 : (if counts.len = x then
                if woff.len = x then do
                  let i3 ← 256#usize + 1#usize
                  ok (decide (bstart.len = i3))
                else ok false
              else ok false) = ok true := by
      rw [if_pos e1, if_pos e2]
      simp [hi3, e3]
    simp only [hok1, bind_tc_ok]
    have hinv : OutInv counts w.val bstart woff true 0#usize (0#usize).val := by
      refine ⟨by simp, hb, hwo, fun _ => ⟨?_, fun j' hj' => by simp at hj', fun j' hj' => by simp at hj'⟩, by simp⟩
      simp [colsA]
    obtain ⟨bs1, wo1, ok2, acc, hloop, hI⟩ :=
      off_outer_loop counts w bstart woff true 0#usize 0#usize hc hw hinv
    simp only [hloop, bind_tc_ok]
    obtain ⟨-, hbl1, hwl1, hok2, hfal⟩ := hI
    cases ok2
    · have hmax := hfal rfl
      have hn : n.val ≤ Usize.max := by scalar_tac
      refine ⟨false, bs1, wo1, ?_, hbl1, hwl1, ?_, by simp⟩
      · show (do let bstart2 ← (ok bs1 : Result (Slice Usize)); ok (false, bstart2, wo1)) = _
        simp only [bind_tc_ok]
      simp only [Bool.false_eq_true, false_iff]
      omega
    · obtain ⟨hacc, hbs, hcols⟩ := hok2 rfl
      obtain ⟨bs2, hbs2, hbs2v⟩ := WP.spec_imp_exists
        (Slice.update_spec bs1 256#usize acc (by rw [h256]; omega))
      refine ⟨decide (acc = n), bs2, wo1, ?_, ?_, hwl1, ?_, fun _ => ⟨?_, ?_⟩⟩
      · show (do let bstart2 ← bs1.update 256#usize acc; ok (decide (acc = n), bstart2, wo1)) = _
        rw [hbs2]; simp only [bind_tc_ok]
      · rw [hbs2v]; simp [Slice.length, Slice.set_val_eq, hbl1]
      · simp only [decide_eq_true_eq, UScalar.eq_equiv, hacc]
      · intro j hj
        have hbound : (256#usize).val < bs1.val.length := by rw [h256]; simp [Slice.length] at hbl1; omega
        rw [hbs2v, Slice.set_val_eq, get!_set _ _ _ _ (by simpa using hbound)]
        by_cases hj256 : j = (256#usize).val
        · rw [if_pos hj256, hj256, h256]; exact hacc
        · rw [if_neg hj256]; exact hbs j (by rw [h256] at hj256; omega)
      · exact hcols

theorem word_count_offsets_refuse_mul (counts : Slice U32) (w n : Usize) (bstart woff : Slice Usize)
    (hw : ¬ w.val*256 ≤ Usize.max) :
    word_count_offsets counts w n bstart woff = ok (false, bstart, woff) := by
  unfold word_count_offsets hashtrinity.freq.FreqMapKI64.count_offsets
  simp only [buckets_ok, bind_tc_ok, lift]
  have hv := Usize.checked_mul_bv_spec w 256#usize
  rcases hm : w.checked_mul 256#usize with _ | x
  · rfl
  · rw [hm] at hv
    exfalso
    obtain ⟨h1, -, -⟩ := hv
    have : (256#usize).val = 256 := rfl
    rw [this] at h1
    exact hw h1

theorem word_count_offsets_refuse_len (counts : Slice U32) (w n : Usize) (bstart woff : Slice Usize)
    (hl : ¬ (counts.length = w.val*256 ∧ woff.length = w.val*256 ∧ bstart.length = 257)) :
    word_count_offsets counts w n bstart woff = ok (false, bstart, woff) := by
  unfold word_count_offsets hashtrinity.freq.FreqMapKI64.count_offsets
  simp only [buckets_ok, bind_tc_ok, lift]
  have hv := Usize.checked_mul_bv_spec w 256#usize
  rcases hm : w.checked_mul 256#usize with _ | x
  · rw [hm] at hv
  · rw [hm] at hv
    obtain ⟨-, hxv, -⟩ := hv
    have h256 : (256#usize).val = 256 := rfl
    rw [h256] at hxv
    obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := 256#usize) (y := 1#usize) (by scalar_tac))
    have hok1 : (if counts.len = x then
                if woff.len = x then do
                  let i3 ← 256#usize + 1#usize
                  ok (decide (bstart.len = i3))
                else ok false
              else ok false) = ok false := by
      by_cases e1 : counts.len = x
      · rw [if_pos e1]
        by_cases e2 : woff.len = x
        · rw [if_pos e2]
          have : ¬ bstart.len = i3 := by
            intro e3
            apply hl
            refine ⟨?_, ?_, ?_⟩
            · have := (UScalar.eq_equiv _ _).mp e1; simpa [hxv] using this
            · have := (UScalar.eq_equiv _ _).mp e2; simpa [hxv] using this
            · have := (UScalar.eq_equiv _ _).mp e3; simp [hi3v, h256] at this; simpa using this
          simp [hi3, this]
        · rw [if_neg e2]
      · rw [if_neg e1]
    simp only [hok1, bind_tc_ok]
    rw [off_outer_refused]
    simp

/-- `count_offsets` on `w` workers × 256 buckets (`counts[t·256 + j]`).  `b = true` iff the
column-major total `colsA 256` equals `n`; then `bstart[j]` is the exclusive prefix sum of the
column sums, `woff[t·256+j]` is `bstart[j]` plus the earlier workers' counts, and the windows
`[woff[t·256+j], woff[t·256+j] + counts[t·256+j])` tile `[0, n)` with no overlap and no gap.
Refusals (`false`) leave both output slices unchanged on a length or `w·256` overflow. -/
theorem word_count_offsets_spec (counts : Slice U32) (w n : Usize) (bstart woff : Slice Usize) :
    ((w.val*256 ≤ Usize.max ∧ counts.length = w.val*256 ∧ woff.length = w.val*256 ∧
        bstart.length = 257) →
      ∃ b bstart' woff', word_count_offsets counts w n bstart woff = ok (b, bstart', woff') ∧
        bstart'.length = 257 ∧ woff'.length = w.val*256 ∧
        (b = true ↔ colsA counts w.val 256 = n.val) ∧
        (b = true →
          (bstart'.val[0]!).val = 0 ∧ (bstart'.val[256]!).val = n.val ∧
          (∀ j, j < 256 → (bstart'.val[j+1]!).val = (bstart'.val[j]!).val + colsum counts w.val j) ∧
          (∀ t, t < w.val → ∀ j, j < 256 → (woff'.val[t*256+j]!).val
            = (bstart'.val[j]!).val + ∑ t' ∈ Finset.range t, cC counts t' j) ∧
          (∀ t, t < w.val → ∀ j, j < 256 → (woff'.val[t*256+j]!).val + cC counts t j
            = if t + 1 < w.val then (woff'.val[(t+1)*256+j]!).val else (bstart'.val[j+1]!).val))) ∧
    (¬ (w.val*256 ≤ Usize.max ∧ counts.length = w.val*256 ∧ woff.length = w.val*256 ∧
        bstart.length = 257) →
      word_count_offsets counts w n bstart woff = ok (false, bstart, woff)) := by
  constructor
  · rintro ⟨hw, hc, hwo, hb⟩
    obtain ⟨b, bs, wo, hr, h1, h2, h3, h4⟩ := word_count_offsets_spec_main counts w n bstart woff hw hc hwo hb
    refine ⟨b, bs, wo, hr, h1, h2, h3, fun hbt => ?_⟩
    obtain ⟨hbs, hwo'⟩ := h4 hbt
    have hn : colsA counts w.val 256 = n.val := h3.mp hbt
    have hbs0 := hbs 0 (by omega)
    have hbs256 := hbs 256 le_rfl
    refine ⟨by rw [hbs0]; simp [colsA], by rw [hbs256, hn], fun j hj => ?_, fun t ht j hj => ?_,
      fun t ht j hj => ?_⟩
    · rw [hbs (j+1) (by omega), hbs j (by omega), colsA_succ]
    · rw [hwo' j hj t ht, hbs j (by omega)]
    · rw [hwo' j hj t ht]
      have hsum := Finset.sum_range_succ (fun t' => cC counts t' j) t
      split_ifs with hlt
      · rw [hwo' j hj (t+1) hlt, hsum]; omega
      · have htw : t + 1 = w.val := by omega
        rw [hbs (j+1) (by omega), colsA_succ]
        unfold colsum
        rw [← htw, hsum]
        omega
  · intro hn
    by_cases hw : w.val*256 ≤ Usize.max
    · exact word_count_offsets_refuse_len counts w n bstart woff
        (fun h => hn ⟨hw, h.1, h.2.1, h.2.2⟩)
    · exact word_count_offsets_refuse_mul counts w n bstart woff hw

end Apeiron.CountHelpers
