-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe`: the executed capped probe.
import Map.ExecutedBase
import Count.Helpers
open Aeneas Aeneas.Std Result

/-!
# The executed capped probe

The capped primary probe of `hashtrinity::Table` (`src/hashtrinity/overflow.rs`,
`probe_rest_with`, `probe_capped_with`, `probe_capped_keyed`, and `prefix_masks`), extracted as
`hashtrinity.overflow.Table.*` and `hashtrinity.overflow.prefix_masks*` in `proof/ApeironWord.lean`.

The probe sequence is the extracted `hashtrinity.Table.descend` itself: `Desc ks s c j i` says the
`j`-th probe slot of coordinate `c` in table `s` is `i` (`descend … s c j = ok i`).  The cap is
`limN bits = min 16 (8 + 64 / bits - 1)` (`probe_limit`).  `CappedOK` is the contract of a capped
probe's verdict:

* `Hit i` — `coords[i] = c`;
* `Vacant i` — `coords[i] = EMPTY`, `i` is the `j`-th probe slot for some `j` below the cap, and
  every earlier probe slot holds neither `c` nor `EMPTY`;
* `Blocked` — every probe slot below the cap holds neither `c` nor `EMPTY`.

The key-confirmation is the `KeyStore::matches` of the table's key store; on `u64` keys it is `true`
for both `Stored` and `Regen` (`stored_matches_ok`, `regen_matches_ok`), and every theorem here takes
that as the hypothesis `hm`.

`prefix_masks` branches on the opaque runtime probe `avx2_probe`.  The scalar branch is proved here
(`prefix_masks_loop_spec`).  The AVX2 branch's contract is the named proposition `Avx2Spec`, the
statement a sibling lane (`proof/corpus/Map/PrefixMasks.lean`, `prefix_masks_branches_agree`) proves
for the modelled intrinsics; every theorem that reaches the AVX2 branch carries
`avx2_probe = true → Avx2Spec` as a hypothesis, which is vacuous when the probe is `false`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-- The coordinate instance of `u64`. -/
abbrev CI := U64.Insts.Apeiron_wordHashtrinityAlgebraCoord
/-- The executed table at `u64` keys and `i64` values, key store `KS`. -/
abbrev Tbl (KS : Type) := hashtrinity.Table U64 I64 KS
/-- The confirmation instance the capped probe is called with. -/
abbrev ConfI {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64) :=
  hashtrinity.overflow.ByKey.Insts.Apeiron_wordHashtrinityOverflowConfirm core.cmp.EqU64 CI ks

/-- The probe cap: `min PROBE_CAP (LINEAR_G + 64 / bits - 1)`. -/
def limN (b : Nat) : Nat := min 16 (8 + 64 / b - 1)

theorem limN_pos (b : Nat) : 1 ≤ limN b := by
  unfold limN; generalize 64 / b = x; omega

theorem LINEAR_G_val : hashtrinity.LINEAR_G.val = 8 := by
  unfold hashtrinity.LINEAR_G; rfl

theorem PROBE_CAP_val : hashtrinity.PROBE_CAP.val = 16 := by
  unfold hashtrinity.PROBE_CAP; rfl

theorem cast_usize_u32 (x : U32) : (UScalar.cast .Usize x).val = x.val := by
  rw [UScalar.cast_val_eq]
  apply Nat.mod_eq_of_lt
  have h1 : x.val < 2^32 := by have := x.hBounds; simpa [UScalarTy.numBits] using this
  calc x.val < 2^32 := h1
    _ ≤ _ := Nat.pow_le_pow_right (by norm_num) (by simp [UScalarTy.numBits])

theorem LG_usize : (UScalar.cast .Usize hashtrinity.LINEAR_G).val = 8 := by
  rw [cast_usize_u32, LINEAR_G_val]

theorem slice_len_lt (s : Slice U64) : s.length < 2 ^ System.Platform.numBits := by
  have h := Slice.length_ineq s
  have : Usize.max = 2 ^ System.Platform.numBits - 1 := by
    simp [Usize.max, Usize.numBits]
  have hp : 0 < 2 ^ System.Platform.numBits := by positivity
  simp only [Slice.length]; omega

section
variable {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)

/-- The `j`-th probe slot of coordinate `c` in table `s` is `i`: the extracted `descend`. -/
def Desc (s : Tbl KS) (c : U64) (j : U32) (i : Usize) : Prop :=
  hashtrinity.Table.descend core.cmp.EqU64 CI ks s c j = ok i

/-- Probe step `j` of `c` passes: its slot holds neither `c` nor `EMPTY`. -/
def Pass (s : Tbl KS) (c : U64) (j : U32) : Prop :=
  ∃ i, Desc ks s c j i ∧ ∃ x, s.coords.val[i.val]? = some x ∧ x ≠ c ∧ x.val ≠ EMPTYN

/-- The contract of a capped probe's verdict. -/
def CappedOK (s : Tbl KS) (c : U64) : hashtrinity.Capped → Prop
  | .Hit i => s.coords.val[i.val]? = some c
  | .Vacant i => (∃ x, s.coords.val[i.val]? = some x ∧ x.val = EMPTYN) ∧
      ∃ j : U32, j.val < limN s.bits.val ∧ Desc ks s c j i ∧
        ∀ j' : U32, j'.val < j.val → Pass ks s c j'
  | .Blocked => ∀ j : U32, j.val < limN s.bits.val → Pass ks s c j

theorem desc_det {s : Tbl KS} {c : U64} {j : U32} {i i' : Usize}
    (h1 : Desc ks s c j i) (h2 : Desc ks s c j i') : i = i' := by
  unfold Desc at h1 h2
  rw [h1] at h2
  exact Result.ok_injective h2

/-- The extracted const table: entry `k` of `WINDOW_STEPS_BY_BITS` is `LINEAR_G + 64 / k - 1` for
`1 ≤ k ≤ 64` (the loop `t[b] = LINEAR_G + 64 / b - 1`, `b = 1 .. 64`). -/
theorem window_steps_table (a : Std.Array U32 65#usize)
    (h : hashtrinity.overflow.WINDOW_STEPS_BY_BITS = ok a) (k : Nat) (hk1 : 1 ≤ k) (hk : k ≤ 64) :
    ∃ x : U32, a.val[k]? = some x ∧ x.val = 8 + 64 / k - 1 := by
  unfold hashtrinity.overflow.WINDOW_STEPS_BY_BITS at h
  unfold hashtrinity.overflow.WINDOW_STEPS_BY_BITS_loop at h
  refine loop_pc
    (fun (p : Std.Array U32 65#usize × Usize) => 1 ≤ p.2.val ∧ p.2.val ≤ 65 ∧
      ∀ k : Nat, 1 ≤ k → k < p.2.val → ∃ x : U32, p.1.val[k]? = some x ∧ x.val = 8 + 64 / k - 1)
    (fun a' => ∀ k : Nat, 1 ≤ k → k ≤ 64 → ∃ x : U32, a'.val[k]? = some x ∧ x.val = 8 + 64 / k - 1)
    _ ?_ _ ?_ a h k hk1 hk
  · rintro ⟨t, b⟩ ⟨hb1, hb2, hinv⟩ cf hcf
    dsimp only at hb1 hb2 hinv hcf
    unfold hashtrinity.overflow.WINDOW_STEPS_BY_BITS_loop.body at hcf
    by_cases hlt : b < 65#usize
    · rw [if_pos hlt] at hcf
      have hbv : b.val < 65 := by simpa using hlt
      obtain ⟨i, hi, hcf⟩ := bind_inv hcf
      obtain ⟨i1, hi1, hcf⟩ := bind_inv hcf
      obtain ⟨i2, hi2, hcf⟩ := bind_inv hcf
      obtain ⟨i3, hi3, hcf⟩ := bind_inv hcf
      obtain ⟨a2, ha2, hcf⟩ := bind_inv hcf
      obtain ⟨b1, hb1', hcf⟩ := bind_inv hcf
      have hcf' := (Result.ok_injective hcf).symm
      subst hcf'
      have hiv : i.val = b.val := by
        have : i = UScalar.cast .U32 b := (Result.ok_injective hi).symm
        subst this
        rw [UScalar.cast_val_eq]
        apply Nat.mod_eq_of_lt
        have : b.val < 2 ^ 32 := by omega
        simpa [UScalarTy.numBits] using this
      have hd := (udiv_ok hi1).2
      have hd0 := (udiv_ok hi1).1
      have hs := (uadd_ok hi2).1
      have hu := usub_ok hi3
      have h64 : (64#u32).val = 64 := rfl
      have h1 : (1#u32).val = 1 := rfl
      rw [h64, hiv] at hd
      rw [LINEAR_G_val] at hs
      rw [h1] at hu
      have hb1v := (uadd_ok hb1').1
      have h1u : (1#usize).val = 1 := rfl
      rw [h1u] at hb1v
      have hupd : a2 = t.set b i3 := by
        have := Array.update_spec t b i3 (by simp; omega)
        rw [ha2] at this
        exact (Aeneas.Std.WP.spec_ok a2).mp this
      subst hupd
      dsimp only
      refine ⟨by omega, by omega, ?_⟩
      intro k hk1 hk2
      by_cases hkb : k = b.val
      · subst hkb
        refine ⟨i3, ?_, by omega⟩
        rw [Array.set_val_eq, List.getElem?_set_self]
        have := t.property
        simp at this ⊢
        omega
      · obtain ⟨x, hx1, hx2⟩ := hinv k hk1 (by omega)
        refine ⟨x, ?_, hx2⟩
        rw [Array.set_val_eq, List.getElem?_set_ne]
        · exact hx1
        · omega
    · rw [if_neg hlt] at hcf
      have hbv : 65 ≤ b.val := by simpa using hlt
      have := (Result.ok_injective hcf).symm
      subst this
      intro k hk1 hk2
      exact hinv k hk1 (by omega)
  · refine ⟨by simp, by simp, ?_⟩
    intro k hk1 hk2
    simp at hk2
    omega

theorem probe_limit_val (s : Tbl KS) (l : U32) (hb : 0 < s.bits.val)
    (h : hashtrinity.overflow.Table.probe_limit core.cmp.EqU64 CI ks s = ok l) :
    l.val = limN s.bits.val := by
  unfold hashtrinity.overflow.Table.probe_limit at h
  obtain ⟨w, hw, h⟩ := bind_inv h
  unfold hashtrinity.overflow.Table.window_steps at hw
  obtain ⟨a, ha, hw⟩ := bind_inv hw
  obtain ⟨i1, hi1, hw⟩ := bind_inv hw
  have hi1v : i1.val = s.bits.val := by
    have : i1 = UScalar.cast .Usize s.bits := (Result.ok_injective hi1).symm
    subst this
    exact cast_usize_u32 _
  have hb64 : s.bits.val ≤ 64 := by
    unfold Array.index_usize at hw
    split at hw
    · simp at hw
    · rename_i x hx
      have := (List.getElem?_eq_some_iff.mp hx).1
      simp at this
      omega
  obtain ⟨x, hx1, hx2⟩ := window_steps_table a ha s.bits.val hb hb64
  have hwx : w = x := by
    unfold Array.index_usize at hw
    have hx1' : a[i1]? = some x := by
      show a.val[i1.val]? = some x
      rw [hi1v]; exact hx1
    rw [hx1'] at hw
    exact (Result.ok_injective hw).symm
  subst hwx
  have hwv : w.val = 8 + 64 / s.bits.val - 1 := hx2
  unfold core.cmp.Ord.min.trait_default core.cmp.Ord.min.default core.cmp.Ord.min_body at h
  simp [core.cmp.impls.PartialOrdU32.lt, liftFun2] at h
  unfold limN
  by_cases hlt : w.val < hashtrinity.PROBE_CAP.val
  · rw [if_pos hlt] at h
    have := Result.ok_injective h
    subst this
    rw [PROBE_CAP_val] at hlt
    omega
  · rw [if_neg hlt] at h
    have := Result.ok_injective h
    subst this
    rw [PROBE_CAP_val] at hlt ⊢
    omega

/-- On the linear prefix, the `j`-th probe slot is `home + j` when it does not wrap. -/
theorem desc_lin (s : Tbl KS) (c : U64) (j : U32) (h : Usize)
    (hj : j.val < 8) (hhome : hashtrinity.Table.home core.cmp.EqU64 CI ks s c = ok h)
    (hlen : s.coords.length = 2 ^ s.bits.val) (hlt : h.val + j.val < s.coords.length) :
    ∃ i : Usize, Desc ks s c j i ∧ i.val = h.val + j.val := by
  have hL : (Slice.len s.coords).val = s.coords.length := Slice.len_val _
  have h1 : (1#usize).val = 1 := rfl
  obtain ⟨m, hm, hmv, _⟩ := WP.spec_imp_exists
    (UScalar.sub_spec (x := Slice.len s.coords) (y := 1#usize) (by rw [hL, h1]; omega))
  refine ⟨(core.num.Usize.wrapping_add h (UScalar.cast .Usize j)) &&& m, ?_, ?_⟩
  · unfold Desc hashtrinity.Table.descend
    dsimp only
    rw [hm]
    simp only [bind_tc_ok]
    have hjl : j < hashtrinity.LINEAR_G := by
      rw [UScalar.lt_equiv, LINEAR_G_val]; exact hj
    rw [if_pos hjl, hhome]
    simp [lift]
  · rw [UScalar.val_and, core.num.Usize.wrapping_add_val_eq, cast_usize_u32, hmv, hL, h1, hlen]
    have hsz : h.val + j.val < UScalar.size .Usize := by
      have := slice_len_lt s.coords
      simp only [UScalar.size, UScalarTy.numBits] at *
      omega
    rw [Nat.mod_eq_of_lt hsz]
    rw [Nat.and_two_pow_sub_one_eq_mod]
    apply Nat.mod_eq_of_lt
    omega

/-- The probe sequence depends on the table only through the slot count and `bits`. -/
theorem desc_frame (s s' : Tbl KS) (c : U64) (j : U32) (i : Usize)
    (hlen : s'.coords.length = s.coords.length) (hbits : s'.bits = s.bits)
    (h : Desc ks s c j i) : Desc ks s' c j i := by
  have hL : Slice.len s'.coords = Slice.len s.coords :=
    UScalar.eq_of_val_eq (by rw [Slice.len_val, Slice.len_val, hlen])
  unfold Desc hashtrinity.Table.descend hashtrinity.Table.home at *
  rw [hL, hbits]
  exact h

theorem pass_frame (s s' : Tbl KS) (c : U64) (j : U32)
    (hlen : s'.coords.length = s.coords.length) (hbits : s'.bits = s.bits)
    (hcell : ∀ (i : Nat) (x : U64), s.coords.val[i]? = some x → x.val ≠ EMPTYN → x ≠ c →
      ∃ y, s'.coords.val[i]? = some y ∧ y ≠ c ∧ y.val ≠ EMPTYN)
    (h : Pass ks s c j) : Pass ks s' c j := by
  obtain ⟨i, hd, x, hx, hxc, hxe⟩ := h
  obtain ⟨y, hy, hyc, hye⟩ := hcell i.val x hx hxe hxc
  exact ⟨i, desc_frame ks s s' c j i hlen hbits hd, y, hy, hyc, hye⟩

theorem confirm_ok (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (q : hashtrinity.overflow.ByKey U64) (keys : KS) (i : Usize) :
    (ConfI ks).confirm q keys i = ok true := by
  show hashtrinity.overflow.ByKey.Insts.Apeiron_wordHashtrinityOverflowConfirm.confirm
    core.cmp.EqU64 CI ks q keys i = ok true
  unfold hashtrinity.overflow.ByKey.Insts.Apeiron_wordHashtrinityOverflowConfirm.confirm
  exact hm _ _ _ _

theorem confirm_ok' (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (q : hashtrinity.overflow.ByKey U64) (keys : KS) (i : Usize) :
    hashtrinity.overflow.ByKey.Insts.Apeiron_wordHashtrinityOverflowConfirm.confirm
      core.cmp.EqU64 CI ks q keys i = ok true := by
  unfold hashtrinity.overflow.ByKey.Insts.Apeiron_wordHashtrinityOverflowConfirm.confirm
  exact hm _ _ _ _

/-- The scalar tail of the capped probe from step `j0`, given that every earlier step passed. -/
theorem probe_rest_spec (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (s : Tbl KS) (c : U64) (lim : U32) (hlim : lim.val = limN s.bits.val)
    (q : hashtrinity.overflow.ByKey U64) (j0 : U32)
    (hpre : ∀ j' : U32, j'.val < j0.val → Pass ks s c j') (r : hashtrinity.Capped)
    (h : hashtrinity.overflow.Table.probe_rest_with core.cmp.EqU64 CI ks (ConfI ks) s c j0 lim q
      = ok r) :
    CappedOK ks s c r := by
  unfold hashtrinity.overflow.Table.probe_rest_with hashtrinity.overflow.Table.probe_rest_with_loop
    at h
  refine loop_pc (fun (j : U32) => ∀ j' : U32, j'.val < j.val → Pass ks s c j')
    (fun r => CappedOK ks s c r) _ ?_ j0 hpre r h
  intro j hj cf hcf
  unfold hashtrinity.overflow.Table.probe_rest_with_loop.body at hcf
  by_cases hjl : j < lim
  · rw [if_pos hjl] at hcf
    have hjl' : j.val < lim.val := (UScalar.lt_equiv _ _).mp hjl
    obtain ⟨i, hi, hcf⟩ := bind_inv hcf
    obtain ⟨sc, hsc, hcf⟩ := bind_inv hcf
    have hsc' := get_unchecked_ok _ _ _ hsc
    by_cases hc : sc = c
    · rw [if_pos hc] at hcf
      obtain ⟨b, hb, hcf⟩ := bind_inv hcf
      rw [confirm_ok ks hm] at hb
      have hb' : b = true := (Result.ok_injective hb).symm
      subst hb'
      rw [if_pos rfl] at hcf
      have := (Result.ok_injective hcf).symm
      subst this
      show s.coords.val[i.val]? = some c
      rw [hsc', hc]
    · rw [if_neg hc] at hcf
      by_cases he : sc = hashtrinity.EMPTY
      · rw [if_pos he] at hcf
        have := (Result.ok_injective hcf).symm
        subst this
        refine ⟨⟨sc, hsc', (EMPTY_eq_iff sc).mp he⟩, j, by omega, hi, hj⟩
      · rw [if_neg he] at hcf
        obtain ⟨j1, hj1, hcf⟩ := bind_inv hcf
        have := (Result.ok_injective hcf).symm
        subst this
        have hj1v := (uadd_ok hj1).1
        have h1 : (1#u32).val = 1 := rfl
        rw [h1] at hj1v
        intro j' hj'
        by_cases hjj : j'.val < j.val
        · exact hj j' hjj
        · have hjeq : j' = j := UScalar.eq_of_val_eq (by omega)
          subst hjeq
          exact ⟨i, hi, sc, hsc', hc, fun hh => he ((EMPTY_eq_iff sc).mpr hh)⟩
  · rw [if_neg hjl] at hcf
    have := (Result.ok_injective hcf).symm
    subst this
    intro j' hj'
    have : ¬ j.val < lim.val := fun hh => hjl ((UScalar.lt_equiv _ _).mpr hh)
    exact hj j' (by omega)

/-! ## The linear-prefix masks -/

/-- The contract of `prefix_masks`: bit `k` of `hits` (resp. `empties`) is set iff `k < 8` and the
coordinate at `home + k` is `coord` (resp. `EMPTY`). -/
def MaskSpec (coords : Slice U64) (home : Usize) (coord : U64) (hits empties : U32) : Prop :=
  ∀ k : Nat, (hits.val.testBit k = true ↔ k < 8 ∧ coords.val[home.val + k]? = some coord) ∧
    (empties.val.testBit k = true ↔
      k < 8 ∧ ∃ x, coords.val[home.val + k]? = some x ∧ x.val = EMPTYN)

/-- The AVX2 branch's contract: on a readable 8-lane window, `prefix_masks_avx2` over the window's
pointer returns the masks `MaskSpec` describes.  This is the statement the scalar branch satisfies
(`prefix_masks_loop_spec`); a sibling lane proves the modelled intrinsics agree with it. -/
def Avx2Spec : Prop :=
  ∀ (coords : Slice U64) (home : Usize) (coord : U64) (p p1 : ConstRawPtr U64) (r : U32 × U32),
    home.val + 8 ≤ coords.length → core.slice.Slice.as_ptr coords = ok p →
    core.ptr.const_ptr.RawPtrConstT.add p home = ok p1 →
    hashtrinity.overflow.prefix_masks_avx2 p1 coord = ok r → MaskSpec coords home coord r.1 r.2

theorem from_bool_val (b : Bool) :
    (core.convert.num.FromU32Bool.from b).val = if b then 1 else 0 := by
  unfold core.convert.num.FromU32Bool.from
  cases b <;> rfl

theorem shl_bit_val (b : Bool) (k : Nat) (hk : k < 8) (z : U32)
    (hz : z.val = ((if b then 1 else 0) * 2 ^ k) % 2 ^ UScalarTy.U32.numBits) :
    ∀ t, z.val.testBit t = (b && decide (t = k)) := by
  intro t
  have hnb : UScalarTy.U32.numBits = 32 := rfl
  rw [hnb] at hz
  have h2 : 2 ^ k < 2 ^ 32 := Nat.pow_lt_pow_right (by norm_num) (by omega)
  cases b
  · simp at hz; simp [hz]
  · simp only [if_true, one_mul] at hz
    rw [Nat.mod_eq_of_lt h2] at hz
    rw [hz, Nat.testBit_two_pow]
    simp only [Bool.true_and]
    by_cases h : t = k
    · subst h; simp
    · simp [h, Ne.symm h]

/-- The scalar branch of `prefix_masks` meets `MaskSpec`. -/
theorem prefix_masks_loop_spec (coords : Slice U64) (home : Usize) (coord : U64)
    (r : U32 × U32)
    (h : hashtrinity.overflow.prefix_masks_loop coords home coord 0#u32 0#u32 0#usize = ok r) :
    MaskSpec coords home coord r.1 r.2 := by
  unfold hashtrinity.overflow.prefix_masks_loop at h
  refine loop_pc (fun (st : U32 × U32 × Usize) => st.2.2.val ≤ 8 ∧
      ∀ b : Nat, (st.1.val.testBit b = true ↔ b < st.2.2.val ∧
          coords.val[home.val + b]? = some coord) ∧
        (st.2.1.val.testBit b = true ↔ b < st.2.2.val ∧
          ∃ x, coords.val[home.val + b]? = some x ∧ x.val = EMPTYN))
    (fun (r : U32 × U32) => MaskSpec coords home coord r.1 r.2) _ ?_ (0#u32, 0#u32, 0#usize)
    ?_ r h
  · intro st hst cf hcf
    obtain ⟨hits, empties, k⟩ := st
    obtain ⟨hk8, hb⟩ := hst
    simp only [] at hk8 hb
    unfold hashtrinity.overflow.prefix_masks_loop.body at hcf
    simp only [lift, bind_tc_ok] at hcf
    by_cases hk : k < UScalar.cast .Usize hashtrinity.LINEAR_G
    · rw [if_pos hk] at hcf
      have hk' : k.val < 8 := by rw [UScalar.lt_equiv, LG_usize] at hk; exact hk
      obtain ⟨i1, hi1, hcf⟩ := bind_inv hcf
      obtain ⟨sc, hsc, hcf⟩ := bind_inv hcf
      obtain ⟨i3, hi3, hcf⟩ := bind_inv hcf
      obtain ⟨i5, hi5, hcf⟩ := bind_inv hcf
      obtain ⟨k1, hk1, hcf⟩ := bind_inv hcf
      have := (Result.ok_injective hcf).symm
      subst this
      have hi1v := (uadd_ok hi1).1
      have hsc' := slice_index_ok _ _ _ hsc
      rw [hi1v] at hsc'
      have hi3v := (ushl_ok hi3).2
      have hi5v := (ushl_ok hi5).2
      rw [from_bool_val] at hi3v hi5v
      have hb3 := shl_bit_val (decide (sc = coord)) k.val hk' i3 hi3v
      have hb5 := shl_bit_val (decide (sc = hashtrinity.EMPTY)) k.val hk' i5 hi5v
      have hk1v := (uadd_ok hk1).1
      have h1 : (1#usize).val = 1 := rfl
      rw [h1] at hk1v
      refine ⟨by show k1.val ≤ 8; omega, ?_⟩
      intro b
      dsimp only
      simp only [UScalar.val_or, Nat.testBit_or, hb3, hb5, Bool.or_eq_true, Bool.and_eq_true,
        decide_eq_true_eq]
      rw [(hb b).1, (hb b).2, hk1v]
      constructor
      · constructor
        · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩)
          · exact ⟨by omega, h2⟩
          · subst h2; exact ⟨by omega, by rw [hsc']; simpa using h1⟩
        · rintro ⟨h1, h2⟩
          by_cases hbk : b = k.val
          · subst hbk; right; refine ⟨?_, rfl⟩; rw [hsc'] at h2; simpa using h2
          · left; exact ⟨by omega, h2⟩
      · constructor
        · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩)
          · exact ⟨by omega, h2⟩
          · subst h2
            refine ⟨by omega, sc, hsc', ?_⟩
            have : sc = hashtrinity.EMPTY := by simpa using h1
            exact (EMPTY_eq_iff sc).mp this
        · rintro ⟨h1, x, hx, hxe⟩
          by_cases hbk : b = k.val
          · subst hbk
            right
            refine ⟨?_, rfl⟩
            rw [hsc'] at hx
            have := Option.some.inj hx
            subst this
            simpa using (EMPTY_eq_iff sc).mpr hxe
          · left; exact ⟨by omega, x, hx, hxe⟩
    · rw [if_neg hk] at hcf
      have := (Result.ok_injective hcf).symm
      subst this
      have hk' : ¬ k.val < 8 := by
        intro hh; apply hk; rw [UScalar.lt_equiv, LG_usize]; exact hh
      have hk8' : k.val = 8 := by omega
      intro b
      rw [hk8'] at hb
      exact hb b
  · refine ⟨by simp, ?_⟩
    intro b
    simp

/-- `prefix_masks` meets `MaskSpec` on both branches: the scalar branch by
`prefix_masks_loop_spec`, the AVX2 branch by `Avx2Spec`. -/
theorem prefix_masks_spec
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (coords : Slice U64) (home : Usize) (coord : U64) (r : U32 × U32)
    (hb : home.val + 8 ≤ coords.length)
    (h : hashtrinity.overflow.prefix_masks coords home coord = ok r) :
    MaskSpec coords home coord r.1 r.2 := by
  unfold hashtrinity.overflow.prefix_masks at h
  unfold std_detect.detect.arch.x86.__is_feature_detected.avx2 at h
  simp only [bind_tc_ok] at h
  by_cases hp : std_detect.detect.arch.x86.avx2_probe = true
  · rw [if_pos hp] at h
    obtain ⟨p, hp1, h⟩ := bind_inv h
    obtain ⟨p1, hp2, h⟩ := bind_inv h
    exact hA hp coords home coord p p1 r hb hp1 hp2 h
  · rw [if_neg hp] at h
    exact prefix_masks_loop_spec coords home coord r h

end


/-! ## Trailing zeros -/

theorem tz_go_spec {w : Nat} (x : BitVec w) :
    ∀ fuel acc, (∃ k, k < fuel ∧ x.getLsbD (acc + k) = true) →
      acc ≤ Aeneas.Std.BitVec.trailingZeros.go x fuel acc ∧
      x.getLsbD (Aeneas.Std.BitVec.trailingZeros.go x fuel acc) = true ∧
      ∀ k, acc ≤ k → k < Aeneas.Std.BitVec.trailingZeros.go x fuel acc → x.getLsbD k = false := by
  intro fuel
  induction fuel with
  | zero => intro acc ⟨k, hk, _⟩; omega
  | succ fuel ih =>
    intro acc ⟨k, hk, hb⟩
    by_cases h : x.getLsbD acc = true
    · have e : Aeneas.Std.BitVec.trailingZeros.go x (fuel+1) acc = acc := by
        simp [Aeneas.Std.BitVec.trailingZeros.go, h]
      rw [e]
      exact ⟨le_refl _, h, fun k h1 h2 => by omega⟩
    · have e : Aeneas.Std.BitVec.trailingZeros.go x (fuel+1) acc
          = Aeneas.Std.BitVec.trailingZeros.go x fuel (acc+1) := by
        simp [Aeneas.Std.BitVec.trailingZeros.go, h]
      rw [e]
      have hk0 : k ≠ 0 := by
        intro h0; subst h0; simp at hb; exact h hb
      obtain ⟨a1, a2, a3⟩ := ih (acc + 1) ⟨k - 1, by omega, by
        have : acc + 1 + (k - 1) = acc + k := by omega
        rw [this]; exact hb⟩
      refine ⟨by omega, a2, ?_⟩
      intro j hj1 hj2
      by_cases hj : j = acc
      · subst hj; simpa using h
      · exact a3 j (by omega) hj2

/-- `u32::trailing_zeros` of a nonzero word: the lowest set bit. -/
theorem u32_tz_spec (m : U32) (hm : m.val ≠ 0) (t : U32)
    (ht : core.num.U32.trailing_zeros m = ok t) :
    t.val < 32 ∧ m.val.testBit t.val = true ∧ ∀ k, k < t.val → m.val.testBit k = false := by
  unfold core.num.U32.trailing_zeros at ht
  have ht' := (Result.ok_injective ht).symm
  have hlt : m.val < 2^32 := by have := m.hBounds; simpa [UScalarTy.numBits] using this
  have hex : ∃ k, k < 32 ∧ m.bv.getLsbD (0 + k) = true := by
    by_contra hne
    push Not at hne
    apply hm
    apply Nat.eq_of_testBit_eq
    intro i
    by_cases hi : i < 32
    · have := hne i hi
      have h2 : m.bv.getLsbD i = false := by simpa using this
      have : m.val.testBit i = m.bv.getLsbD i := (BitVec.testBit_toNat m.bv)
      simp [this, h2]
    · rw [Nat.testBit_eq_false_of_lt (by
        calc m.val < 2^32 := hlt
          _ ≤ 2^i := Nat.pow_le_pow_right (by norm_num) (by omega))]
      simp
  obtain ⟨a1, a2, a3⟩ := tz_go_spec m.bv 32 0 hex
  have hbit : ∀ k, m.val.testBit k = m.bv.getLsbD k := fun k => BitVec.testBit_toNat m.bv
  have htz : Aeneas.Std.BitVec.trailingZeros m.bv = Aeneas.Std.BitVec.trailingZeros.go m.bv 32 0 :=
    rfl
  have hlt32 : Aeneas.Std.BitVec.trailingZeros.go m.bv 32 0 < 32 := by
    by_contra hge
    have : m.val.testBit (Aeneas.Std.BitVec.trailingZeros.go m.bv 32 0) = false :=
      Nat.testBit_eq_false_of_lt (by
        calc m.val < 2^32 := hlt
          _ ≤ 2^_ := Nat.pow_le_pow_right (by norm_num) (by omega))
    rw [hbit] at this
    simp [this] at a2
  have htv : t.val = Aeneas.Std.BitVec.trailingZeros.go m.bv 32 0 := by
    rw [ht', ← htz]
    show (BitVec.ofNat 32 (Aeneas.Std.BitVec.trailingZeros m.bv)).toNat = _
    rw [BitVec.toNat_ofNat, htz]
    exact Nat.mod_eq_of_lt (by omega)
  rw [htv]
  exact ⟨hlt32, by rw [hbit]; exact a2, fun k hk => by rw [hbit]; exact a3 k (by omega) hk⟩

theorem ne_zero_of_bne (m : U32) (h : (m != 0#u32) = true) : m.val ≠ 0 := by
  intro h0
  have : m = 0#u32 := UScalar.eq_of_val_eq (by rw [h0]; rfl)
  simp [this] at h

theorem zero_of_not_bne (m : U32) (h : ¬ (m != 0#u32) = true) : m.val = 0 := by
  simpa using h

theorem testBit_zero_of_val (m : U32) (h : m.val = 0) (k : Nat) : m.val.testBit k = false := by
  rw [h]; simp

section
variable {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)

/-- The masked prefix scan of the cold `probe_capped_cold`: a set hit bit is a hit, else the lowest
empty lane is the vacancy, else the scalar tail from step 8.  This is the original full-body scan,
reached by the hot `probe_capped_with` only on the wrapped window or a full window with no hit. -/
theorem pcc_loop_spec (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (s : Tbl KS) (c : U64) (home : Usize) (q : hashtrinity.overflow.ByKey U64) (lim : U32)
    (hlim : lim.val = limN s.bits.val) (h8 : 8 ≤ lim.val)
    (hb : home.val + 8 ≤ s.coords.length)
    (hlanes : ∀ k : U32, k.val < 8 → ∃ i, Desc ks s c k i ∧ i.val = home.val + k.val)
    (hits empties : U32) (hmask : MaskSpec s.coords home c hits empties) (r : hashtrinity.Capped)
    (h : hashtrinity.overflow.Table.probe_capped_cold_loop core.cmp.EqU64 CI ks (ConfI ks)
      s.coords s.vals s.keys s.side s.overflow s.bits s.len s.tombs s._k c home q lim hits empties
      = ok r) :
    CappedOK ks s c r := by
  have hlane_some : ∀ k : Nat, k < 8 → ∃ x, s.coords.val[home.val + k]? = some x := by
    intro k hk
    exact ⟨_, List.getElem?_eq_getElem (by simp only [Slice.length] at hb; omega)⟩
  have hpass : ∀ j' : U32, j'.val < 8 → hits.val.testBit j'.val = false →
      empties.val.testBit j'.val = false → Pass ks s c j' := by
    intro j' hj' hh he
    obtain ⟨i, hd, hiv⟩ := hlanes j' hj'
    obtain ⟨x, hx⟩ := hlane_some j'.val hj'
    refine ⟨i, hd, x, by rw [hiv]; exact hx, ?_, ?_⟩
    · intro hxc
      have := ((hmask j'.val).1).mpr ⟨hj', by rw [hx, hxc]⟩
      rw [hh] at this; exact absurd this (by simp)
    · intro hxe
      have := ((hmask j'.val).2).mpr ⟨hj', x, hx, hxe⟩
      rw [he] at this; exact absurd this (by simp)
  unfold hashtrinity.overflow.Table.probe_capped_cold_loop at h
  refine loop_pc (fun (h1 : U32) => h1 = hits) (fun r => CappedOK ks s c r) _ ?_ hits rfl r h
  intro h1 hh1 cf hcf
  subst hh1
  unfold hashtrinity.overflow.Table.probe_capped_cold_loop.body at hcf
  by_cases hnz : (h1 != 0#u32) = true
  · rw [if_pos hnz] at hcf
    obtain ⟨t, ht, hcf⟩ := bind_inv hcf
    simp only [lift, bind_tc_ok] at hcf
    obtain ⟨slot, hslot, hcf⟩ := bind_inv hcf
    obtain ⟨b, hbb, hcf⟩ := bind_inv hcf
    rw [confirm_ok' ks hm] at hbb
    have hb' : b = true := (Result.ok_injective hbb).symm
    subst hb'
    rw [if_pos rfl] at hcf
    have := (Result.ok_injective hcf).symm
    subst this
    obtain ⟨_, htb, _⟩ := u32_tz_spec h1 (ne_zero_of_bne h1 hnz) t ht
    have hsv := (uadd_ok hslot).1
    rw [cast_usize_u32] at hsv
    obtain ⟨_, hcoord⟩ := ((hmask t.val).1).mp htb
    show s.coords.val[slot.val]? = some c
    rw [hsv]; exact hcoord
  · rw [if_neg hnz] at hcf
    have hz := zero_of_not_bne h1 hnz
    by_cases hne : (empties != 0#u32) = true
    · rw [if_pos hne] at hcf
      obtain ⟨t, ht, hcf⟩ := bind_inv hcf
      simp only [lift, bind_tc_ok] at hcf
      obtain ⟨i5, hi5, hcf⟩ := bind_inv hcf
      have := (Result.ok_injective hcf).symm
      subst this
      obtain ⟨_, htb, htlow⟩ := u32_tz_spec empties (ne_zero_of_bne empties hne) t ht
      have hiv := (uadd_ok hi5).1
      rw [cast_usize_u32] at hiv
      obtain ⟨ht8, x, hx, hxe⟩ := ((hmask t.val).2).mp htb
      refine ⟨⟨x, by rw [hiv]; exact hx, hxe⟩, t, by omega, ?_, ?_⟩
      · obtain ⟨i, hd, hv⟩ := hlanes t ht8
        have : i = i5 := UScalar.eq_of_val_eq (by rw [hv, hiv])
        rw [← this]; exact hd
      · intro j' hj'
        exact hpass j' (by omega) (testBit_zero_of_val h1 hz _) (htlow _ hj')
    · rw [if_neg hne] at hcf
      have hez := zero_of_not_bne empties hne
      obtain ⟨c', hc', hcf⟩ := bind_inv hcf
      have := (Result.ok_injective hcf).symm
      subst this
      refine probe_rest_spec ks hm s c lim hlim q hashtrinity.LINEAR_G ?_ c' hc'
      intro j' hj'
      rw [LINEAR_G_val] at hj'
      exact hpass j' hj' (testBit_zero_of_val h1 hz _) (testBit_zero_of_val empties hez _)

/-- The cold `probe_capped_cold` meets `CappedOK`, on both `prefix_masks` branches.  This is the
original full-body specification; the hot `probe_capped_with` reaches it on the wrapped window or a
full window with no hit, and `probe_capped_keyed` calls it directly on the insert path. -/
theorem pcc_spec (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (s : Tbl KS) (c : U64) (home : Usize) (q : hashtrinity.overflow.ByKey U64)
    (hbits : 0 < s.bits.val)
    (hhome : hashtrinity.Table.home core.cmp.EqU64 CI ks s c = ok home)
    (hlen : s.coords.length = 2 ^ s.bits.val) (r : hashtrinity.Capped)
    (h : hashtrinity.overflow.Table.probe_capped_cold core.cmp.EqU64 CI ks (ConfI ks) s c home q
      = ok r) :
    CappedOK ks s c r := by
  unfold hashtrinity.overflow.Table.probe_capped_cold at h
  obtain ⟨lim, hl, h⟩ := bind_inv h
  have hlim := probe_limit_val ks s lim hbits hl
  dsimp only at h
  have hnone : ∀ j' : U32, j'.val < (0#u32).val → Pass ks s c j' := by
    intro j' hj'; exact absurd hj' (by simp)
  by_cases h8 : lim >= hashtrinity.LINEAR_G
  · rw [if_pos h8] at h
    simp only [lift, bind_tc_ok] at h
    obtain ⟨i1, hi1, h⟩ := bind_inv h
    have hi1v := (uadd_ok hi1).1
    rw [LG_usize] at hi1v
    by_cases hle : i1 <= Slice.len s.coords
    · rw [if_pos hle] at h
      have hle' : home.val + 8 ≤ s.coords.length := by
        rw [UScalar.le_equiv, Slice.len_val] at hle; omega
      obtain ⟨⟨hits, empties⟩, hpm, h⟩ := bind_inv h
      have hmask := prefix_masks_spec hA s.coords home c (hits, empties) hle' hpm
      have h8' : 8 ≤ lim.val := by rw [GE.ge, UScalar.le_equiv, LINEAR_G_val] at h8; exact h8
      exact pcc_loop_spec ks hm s c home q lim hlim h8' hle'
        (fun k hk => desc_lin ks s c k home hk hhome hlen (by omega)) hits empties hmask r h
    · rw [if_neg hle] at h
      exact probe_rest_spec ks hm s c lim hlim q 0#u32 hnone r h
  · rw [if_neg h8] at h
    exact probe_rest_spec ks hm s c lim hlim q 0#u32 hnone r h

/-- The masked prefix scan of the hot `probe_capped_with`: a set hit bit is a hit, else the lowest
empty lane is the vacancy, else the cold continuation `probe_capped_cold`.  It carries the same
`hmask` contract and the same `CappedOK` verdict as the cold scan (`pcc_loop_spec`); only the
no-hit, no-empty tail differs, deferring to `probe_capped_cold`. -/
theorem pcw_loop_spec (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (s : Tbl KS) (c : U64) (home : Usize) (q : hashtrinity.overflow.ByKey U64)
    (hbits : 0 < s.bits.val)
    (hhome : hashtrinity.Table.home core.cmp.EqU64 CI ks s c = ok home)
    (hlen : s.coords.length = 2 ^ s.bits.val)
    (hb : home.val + 8 ≤ s.coords.length)
    (hlanes : ∀ k : U32, k.val < 8 → ∃ i, Desc ks s c k i ∧ i.val = home.val + k.val)
    (hits empties : U32) (hmask : MaskSpec s.coords home c hits empties) (r : hashtrinity.Capped)
    (h : hashtrinity.overflow.Table.probe_capped_with_loop core.cmp.EqU64 CI ks (ConfI ks)
      s.coords s.vals s.keys s.side s.overflow s.bits s.len s.tombs s._k c home q hits empties
      = ok r) :
    CappedOK ks s c r := by
  -- `s.coords.length = 2 ^ bits` and a slice length below `2 ^ numBits` bound `bits` by 64, so
  -- `64 / bits ≥ 1` and `limN bits = min 16 (8 + 64 / bits - 1) ≥ 8`: the hot path skips the cold
  -- body's `limit >= LINEAR_G` gate, and this is the invariant that gate supplied.
  have hlim8 : 8 ≤ limN s.bits.val := by
    have hlt := slice_len_lt s.coords
    rw [hlen] at hlt
    have hb64 : s.bits.val ≤ 64 := by
      rcases System.Platform.numBits_eq with h | h
      · rw [h] at hlt
        have : s.bits.val < 32 := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp hlt
        omega
      · rw [h] at hlt
        have : s.bits.val < 64 := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp hlt
        omega
    unfold limN
    have hd : 1 ≤ 64 / s.bits.val := Nat.div_pos hb64 hbits
    omega
  have hlane_some : ∀ k : Nat, k < 8 → ∃ x, s.coords.val[home.val + k]? = some x := by
    intro k hk
    exact ⟨_, List.getElem?_eq_getElem (by simp only [Slice.length] at hb; omega)⟩
  have hpass : ∀ j' : U32, j'.val < 8 → hits.val.testBit j'.val = false →
      empties.val.testBit j'.val = false → Pass ks s c j' := by
    intro j' hj' hh he
    obtain ⟨i, hd, hiv⟩ := hlanes j' hj'
    obtain ⟨x, hx⟩ := hlane_some j'.val hj'
    refine ⟨i, hd, x, by rw [hiv]; exact hx, ?_, ?_⟩
    · intro hxc
      have := ((hmask j'.val).1).mpr ⟨hj', by rw [hx, hxc]⟩
      rw [hh] at this; exact absurd this (by simp)
    · intro hxe
      have := ((hmask j'.val).2).mpr ⟨hj', x, hx, hxe⟩
      rw [he] at this; exact absurd this (by simp)
  unfold hashtrinity.overflow.Table.probe_capped_with_loop at h
  refine loop_pc (fun (h1 : U32) => h1 = hits) (fun r => CappedOK ks s c r) _ ?_ hits rfl r h
  intro h1 hh1 cf hcf
  subst hh1
  unfold hashtrinity.overflow.Table.probe_capped_with_loop.body at hcf
  by_cases hnz : (h1 != 0#u32) = true
  · rw [if_pos hnz] at hcf
    obtain ⟨t, ht, hcf⟩ := bind_inv hcf
    simp only [lift, bind_tc_ok] at hcf
    obtain ⟨slot, hslot, hcf⟩ := bind_inv hcf
    obtain ⟨b, hbb, hcf⟩ := bind_inv hcf
    rw [confirm_ok' ks hm] at hbb
    have hb' : b = true := (Result.ok_injective hbb).symm
    subst hb'
    rw [if_pos rfl] at hcf
    have := (Result.ok_injective hcf).symm
    subst this
    obtain ⟨_, htb, _⟩ := u32_tz_spec h1 (ne_zero_of_bne h1 hnz) t ht
    have hsv := (uadd_ok hslot).1
    rw [cast_usize_u32] at hsv
    obtain ⟨_, hcoord⟩ := ((hmask t.val).1).mp htb
    show s.coords.val[slot.val]? = some c
    rw [hsv]; exact hcoord
  · rw [if_neg hnz] at hcf
    have hz := zero_of_not_bne h1 hnz
    by_cases hne : (empties != 0#u32) = true
    · rw [if_pos hne] at hcf
      obtain ⟨t, ht, hcf⟩ := bind_inv hcf
      simp only [lift, bind_tc_ok] at hcf
      obtain ⟨i5, hi5, hcf⟩ := bind_inv hcf
      have := (Result.ok_injective hcf).symm
      subst this
      obtain ⟨_, htb, htlow⟩ := u32_tz_spec empties (ne_zero_of_bne empties hne) t ht
      have hiv := (uadd_ok hi5).1
      rw [cast_usize_u32] at hiv
      obtain ⟨ht8, x, hx, hxe⟩ := ((hmask t.val).2).mp htb
      refine ⟨⟨x, by rw [hiv]; exact hx, hxe⟩, t, by omega, ?_, ?_⟩
      · obtain ⟨i, hd, hv⟩ := hlanes t ht8
        have : i = i5 := UScalar.eq_of_val_eq (by rw [hv, hiv])
        rw [← this]; exact hd
      · intro j' hj'
        exact hpass j' (by omega) (testBit_zero_of_val h1 hz _) (htlow _ hj')
    · rw [if_neg hne] at hcf
      have hez := zero_of_not_bne empties hne
      obtain ⟨c', hc', hcf⟩ := bind_inv hcf
      have := (Result.ok_injective hcf).symm
      subst this
      exact pcc_spec ks hm hA s c home q hbits hhome hlen c' hc'

/-- The hot inline `probe_capped_with` meets `CappedOK`: on `home + 8 ≤ len` the branchless prefix
glance and the hit scan decide the hit/vacant verdict, and every other input defers to the cold
`probe_capped_cold`, whose `CappedOK` is `pcc_spec`. -/
theorem pcw_spec (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (s : Tbl KS) (c : U64) (home : Usize) (q : hashtrinity.overflow.ByKey U64)
    (hbits : 0 < s.bits.val)
    (hhome : hashtrinity.Table.home core.cmp.EqU64 CI ks s c = ok home)
    (hlen : s.coords.length = 2 ^ s.bits.val) (r : hashtrinity.Capped)
    (h : hashtrinity.overflow.Table.probe_capped_with core.cmp.EqU64 CI ks (ConfI ks) s c home q
      = ok r) :
    CappedOK ks s c r := by
  unfold hashtrinity.overflow.Table.probe_capped_with at h
  simp only [lift, bind_tc_ok] at h
  obtain ⟨i1, hi1, h⟩ := bind_inv h
  have hi1v := (uadd_ok hi1).1
  rw [LG_usize] at hi1v
  by_cases hle : i1 <= Slice.len s.coords
  · rw [if_pos hle] at h
    have hle' : home.val + 8 ≤ s.coords.length := by
      rw [UScalar.le_equiv, Slice.len_val] at hle; omega
    obtain ⟨⟨hits, empties⟩, hpm, h⟩ := bind_inv h
    have hmask := prefix_masks_spec hA s.coords home c (hits, empties) hle' hpm
    exact pcw_loop_spec ks hm hA s c home q hbits hhome hlen hle'
      (fun k hk => desc_lin ks s c k home hk hhome hlen (by omega)) hits empties hmask r h
  · rw [if_neg hle] at h
    exact pcc_spec ks hm hA s c home q hbits hhome hlen r h

/-- The keyed capped probe (`probe_capped_keyed`): home slot first, then `probe_capped_cold`. -/
theorem pck_spec (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (s : Tbl KS) (c k : U64) (hbits : 0 < s.bits.val) (hlen : s.coords.length = 2 ^ s.bits.val) (r : hashtrinity.Capped)
    (h : hashtrinity.overflow.Table.probe_capped_keyed core.cmp.EqU64 CI ks s c k = ok r) :
    CappedOK ks s c r := by
  unfold hashtrinity.overflow.Table.probe_capped_keyed at h
  obtain ⟨home, hhome, h⟩ := bind_inv h
  obtain ⟨sc, hsc, h⟩ := bind_inv h
  have hsc' := get_unchecked_ok _ _ _ hsc
  have hhlt : home.val < s.coords.length := (List.getElem?_eq_some_iff.mp hsc').1
  have hvac : sc = hashtrinity.EMPTY → CappedOK ks s c (hashtrinity.Capped.Vacant home) := by
    intro he
    obtain ⟨i, hd, hv⟩ := desc_lin ks s c 0#u32 home (by simp) hhome hlen (by simpa using hhlt)
    have : i = home := UScalar.eq_of_val_eq (by rw [hv]; simp)
    subst this
    refine ⟨⟨sc, hsc', (EMPTY_eq_iff sc).mp he⟩, 0#u32, ?_, hd, ?_⟩
    · have := limN_pos s.bits.val; simp; omega
    · intro j' hj'; exact absurd hj' (by simp)
  by_cases hc : sc = c
  · rw [if_pos hc] at h
    obtain ⟨b, hbb, h⟩ := bind_inv h
    rw [hm] at hbb
    have hb' : b = true := (Result.ok_injective hbb).symm
    subst hb'
    rw [if_pos rfl] at h
    have := (Result.ok_injective h).symm
    subst this
    show s.coords.val[home.val]? = some c
    rw [hsc', hc]
  · rw [if_neg hc] at h
    by_cases he : sc = hashtrinity.EMPTY
    · rw [if_pos he] at h
      have := (Result.ok_injective h).symm
      subst this
      exact hvac he
    · rw [if_neg he] at h
      exact pcc_spec ks hm hA s c home _ hbits hhome hlen r h

end

end Apeiron.MapExecuted
