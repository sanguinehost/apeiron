/-
Exact integer Walsh–Hadamard refinement (`APEIRON-TERNARY-W3-T6a`; spec
`docs/epics/ternary-hadamard.md`).

The Rust body is `src/engine/hadamard.rs`. The pure integer reference — `fwhtStage`, `fwhtRef`,
`fwhtRef_segment`, the Sylvester indexing lemmas, the extracted `block_exponent` laws and the integer
window bounds — lives in `Numeric.HadamardStage`, imported here.

Landed here:

* `segSweep`, the lane-sweep value shape, and `map_eq_segSweep_of_lane`, its bridge to a completed
  lane loop;
* the lane, base and stage `loop.spec_decr_nat` assemblies and their invariants;
* `fwht_int_spec`, the `hadamard.fwht_int` law, and `fwht_int_sylvester`, the aligned-segment law.

The axiom cone is `{propext, Classical.choice, Quot.sound}`; no `sorry`, `axiom` or `decide` over an
extracted body appears.
-/
import Numeric.Hadamard.Sylvester
import Numeric.Engine.Norm
import Numeric.Scale
import Numeric.Hadamard.Stage

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result

open scoped Matrix

noncomputable section

/-! ## The integer shape of one lane sweep

`segSweep m base span` is the whole-list value after the innermost loop swept the block
`[base, base + 2·span)`: the two halves become `u+v` and `u−v` lane-wise. -/

/-- The value shape of a completed lane sweep. -/
def segSweep (m : List ℤ) (base span : ℕ) : List ℤ :=
  m.take base ++
    (((m.drop base).take span).zipWith (· + ·) ((m.drop (base + span)).take span)) ++
    (((m.drop base).take span).zipWith (· - ·) ((m.drop (base + span)).take span)) ++
    m.drop (base + 2 * span)

theorem segSweep_length (m : List ℤ) (base span : ℕ) (h : base + 2 * span ≤ m.length) :
    (segSweep m base span).length = m.length := by
  have h1 : (m.take base).length = base := by
    rw [List.length_take]; exact Nat.min_eq_left (by omega)
  have h2 : ((m.drop base).take span).length = span := by
    rw [List.length_take, List.length_drop]
    exact Nat.min_eq_left (by omega)
  have h3 : ((m.drop (base + span)).take span).length = span := by
    rw [List.length_take, List.length_drop]
    exact Nat.min_eq_left (by omega)
  rw [segSweep, List.length_append, List.length_append, List.length_append,
    List.length_zipWith, List.length_zipWith, h1, h2, h3, List.length_drop]
  omega

/-- The completed sweep depends on `m` only through `m.take base` and `m.drop base`. -/
theorem segSweep_take_drop (m : List ℤ) (base span : ℕ) :
    segSweep m base span = m.take base ++ segSweep (m.drop base) 0 span := by
  simp only [segSweep, List.take_zero, List.nil_append, Nat.zero_add,
    Nat.add_zero, List.drop_drop, List.append_assoc]

/-- The completed sweep of a whole `2^(s+1)`-long block is one `fwhtStage s`. -/
theorem segSweep_eq_fwhtStage (s : ℕ) (seg : List ℤ) (hlen : seg.length = 2 ^ (s + 1)) :
    segSweep seg 0 (2 ^ s) = fwhtStage s seg := by
  have hge : 2 ^ (s + 1) ≤ seg.length := le_of_eq hlen.symm
  have hpow : 2 * 2 ^ s = 2 ^ (s + 1) := by rw [pow_two_succ]; ring
  have hhalf : (seg.drop (2 ^ s)).length = 2 ^ s := by
    rw [List.length_drop]; omega
  have htake : (seg.drop (2 ^ s)).take (2 ^ s) = seg.drop (2 ^ s) :=
    List.take_of_length_le (le_of_eq hhalf)
  have hdrop : seg.drop (2 * 2 ^ s) = [] := by
    rw [hpow]; exact List.drop_eq_nil_iff.mpr (le_of_eq hlen)
  have hdrop2 : seg.drop (2 ^ (s + 1)) = [] := List.drop_eq_nil_iff.mpr (le_of_eq hlen)
  rw [segSweep, List.take_zero, List.nil_append, List.drop_zero]
  simp only [Nat.zero_add]
  rw [htake, hdrop, List.append_nil]
  conv_rhs => rw [fwhtStage_of_ge s hge]
  rw [htake, hdrop2, fwhtStage_nil, List.append_nil]

/-- The lane pair content: `base+i` holds `u+v` and `base+span+i` holds `u−v` for the original
pair `(u, v)` of `x0`. -/
def FwhtPairAt (x0 x : Slice I64) (base span i : ℕ) : Prop :=
  ∃ a b c d : I64,
    x0.val[base + i]? = some a ∧ x0.val[base + span + i]? = some b ∧
    x.val[base + i]? = some c ∧ x.val[base + span + i]? = some d ∧
    c.val = a.val + b.val ∧ d.val = a.val - b.val

/-- `i64` checked subtraction succeeds when the difference fits `i64`. -/
theorem fwht_checked_sub_some (x y : I64)
    (hlo : I64.min ≤ x.val - y.val) (hhi : x.val - y.val ≤ I64.max) :
    ∃ z, x.checked_sub y = some z ∧ z.val = x.val - y.val := by
  have hspec := I64.checked_sub_bv_spec x y
  cases hc : core.num.checked_sub_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I64.checked_sub] using hc, hspec.2.2.1⟩

/-- Innermost-loop invariant at lane `lane`: the segment bounds, the not-yet-swept entries equal to
`x0`'s, the entries outside the segment unchanged, and, on the success path, the pair content of the
swept lanes. -/
def FwhtLaneInv (x0 x : Slice I64) (base span lane : ℕ) (ok1 : Bool) : Prop :=
  base + 2 * span ≤ x0.val.length ∧
  x.val.length = x0.val.length ∧
  lane ≤ span ∧
  (∀ i, lane ≤ i → i < span →
    x.val[base + i]? = x0.val[base + i]? ∧
    x.val[base + span + i]? = x0.val[base + span + i]?) ∧
  (∀ j, (j < base ∨ base + 2 * span ≤ j) → x.val[j]? = x0.val[j]?) ∧
  (ok1 = true → ∀ i, i < lane → FwhtPairAt x0 x base span i)

/-- A completed lane sweep carries `segSweep`: the segment's two halves are the lane-wise sum and
difference and everything outside is unchanged. -/
theorem map_eq_segSweep_of_lane (x0 x : Slice I64) (base span : ℕ)
    (hlen : x.val.length = x0.val.length)
    (hbase : base + 2 * span ≤ x0.val.length)
    (hpair : ∀ i, i < span → FwhtPairAt x0 x base span i)
    (hout : ∀ j, (j < base ∨ base + 2 * span ≤ j) → x.val[j]? = x0.val[j]?) :
    x.val.map (·.val) = segSweep (x0.val.map (·.val)) base span := by
  have hm0len : (x0.val.map (·.val)).length = x0.val.length := by simp
  have htake : (x0.val.map (·.val) |>.take base).length = base := by
    rw [List.length_take, hm0len, Nat.min_eq_left (by omega)]
  have hhalf : ((x0.val.map (·.val) |>.drop base).take span).length = span := by
    rw [List.length_take, List.length_drop, hm0len, Nat.min_eq_left (by omega)]
  have hhalf2 : ((x0.val.map (·.val) |>.drop (base + span)).take span).length = span := by
    rw [List.length_take, List.length_drop, hm0len, Nat.min_eq_left (by omega)]
  have hz1 : (((x0.val.map (·.val) |>.drop base).take span).zipWith (· + ·)
      ((x0.val.map (·.val) |>.drop (base + span)).take span)).length = span := by
    rw [List.length_zipWith, hhalf, hhalf2, Nat.min_self]
  have hz2 : (((x0.val.map (·.val) |>.drop base).take span).zipWith (· - ·)
      ((x0.val.map (·.val) |>.drop (base + span)).take span)).length = span := by
    rw [List.length_zipWith, hhalf, hhalf2, Nat.min_self]
  apply List.ext_getElem?
  intro n
  rw [List.getElem?_map]
  by_cases hn : n < x.val.length
  · by_cases h1 : n < base
    · have hb0 : n < x0.val.length := by omega
      have hxvn : x.val[n]? = some x.val[n] := List.getElem?_eq_getElem hn
      have hx0n : x0.val[n]? = some x0.val[n] := List.getElem?_eq_getElem hb0
      have hxn0 : x.val[n] = x0.val[n] := by
        have h := hout n (Or.inl h1)
        rw [hxvn, hx0n] at h
        exact Option.some.inj h
      rw [hxvn, Option.map_some, hxn0]
      rw [segSweep]
      simp only [List.getElem?_append, List.length_append, htake, hz1, hz2,
        List.getElem?_take, List.getElem?_drop, List.getElem?_zipWith]
      rw [if_pos (by omega : n < base + span + span), if_pos (by omega : n < base + span),
        if_pos h1, if_pos h1, List.getElem?_map, hx0n, Option.map_some]
    · simp only [not_lt] at h1
      by_cases h2 : n < base + span
      · obtain ⟨a, b, c, d, ha, hb, hc, hd, hcv, hdv⟩ := hpair (n - base) (by omega)
        have hnb : base + (n - base) = n := by omega
        have hxcn : x.val[n]? = some c := by rw [← hnb]; exact hc
        have hm0a : (x0.val.map (·.val))[base + (n - base)]? = some a.val := by
          rw [List.getElem?_map, ha]; rfl
        have hm0b : (x0.val.map (·.val))[base + span + (n - base)]? = some b.val := by
          rw [List.getElem?_map, hb]; rfl
        rw [hxcn, Option.map_some, hcv]
        rw [segSweep]
        simp only [List.getElem?_append, List.length_append, htake, hz1, hz2,
          List.getElem?_take, List.getElem?_drop, List.getElem?_zipWith]
        rw [if_pos (by omega : n < base + span + span), if_pos h2,
          if_neg (by omega : ¬ n < base)]
        rw [if_pos (by omega : n - base < span), if_pos (by omega : n - base < span)]
        rw [hm0a, hm0b]
      · simp only [not_lt] at h2
        by_cases h3 : n < base + 2 * span
        · obtain ⟨a, b, c, d, ha, hb, hc, hd, hcv, hdv⟩ := hpair (n - (base + span)) (by omega)
          have hnb : base + span + (n - (base + span)) = n := by omega
          have hxdn : x.val[n]? = some d := by rw [← hnb]; exact hd
          have hm0a : (x0.val.map (·.val))[base + (n - (base + span))]? = some a.val := by
            rw [List.getElem?_map, ha]; rfl
          have hm0b : (x0.val.map (·.val))[base + span + (n - (base + span))]? = some b.val := by
            rw [List.getElem?_map, hb]; rfl
          rw [hxdn, Option.map_some, hdv]
          rw [segSweep]
          simp only [List.getElem?_append, List.length_append, htake, hz1, hz2,
            List.getElem?_take, List.getElem?_drop, List.getElem?_zipWith]
          rw [if_pos (by omega : n < base + span + span),
            if_neg (by omega : ¬ n < base + span)]
          rw [if_pos (by omega : n - (base + span) < span),
            if_pos (by omega : n - (base + span) < span)]
          rw [hm0a, hm0b]
        · simp only [not_lt] at h3
          have hb0 : n < x0.val.length := by omega
          have hx0n : x0.val[n]? = some x0.val[n] := List.getElem?_eq_getElem hb0
          rw [hout n (Or.inr h3), hx0n, Option.map_some]
          rw [segSweep]
          simp only [List.getElem?_append, List.length_append, htake, hz1, hz2,
            List.getElem?_take, List.getElem?_drop, List.getElem?_zipWith]
          rw [if_neg (by omega : ¬ n < base + span + span)]
          rw [show base + 2 * span + (n - (base + span + span)) = n by omega]
          rw [List.getElem?_map, hx0n, Option.map_some]
  · simp only [not_lt] at hn
    have hxn : x.val[n]? = none := List.getElem?_eq_none_iff.mpr hn
    rw [hxn]
    show (none : Option ℤ) = (segSweep (x0.val.map (·.val)) base span)[n]?
    rw [List.getElem?_eq_none_iff.mpr (by
      rw [segSweep_length _ _ _ (by rw [hm0len]; omega), hm0len]; omega)]


/-- One innermost-loop body step on the success path: lane `lane < span` reads the original pair
`(u, v)` from `x0`, writes `(u+v, u−v)` into `x` and advances. -/
theorem fwhtLane_body_step (x0 x : Slice I64) (span base : Usize)
    (hov : ∀ i, i < span.val → ∀ a b : I64, x0.val[base.val + i]? = some a →
      x0.val[base.val + span.val + i]? = some b →
      I64.min ≤ a.val + b.val ∧ a.val + b.val ≤ I64.max ∧
      I64.min ≤ a.val - b.val ∧ a.val - b.val ≤ I64.max)
    (lane : Usize) (hinv : FwhtLaneInv x0 x base.val span.val lane.val true)
    (hlane : lane.val < span.val) :
    ∃ (x' : Slice I64) (lane1 : Usize),
      hadamard.fwht_int_loop0_loop0_loop0.body span base x true none lane
          = ok (ControlFlow.cont (x', true, none, lane1)) ∧
      FwhtLaneInv x0 x' base.val span.val lane1.val true ∧ lane1.val = lane.val + 1 := by
  obtain ⟨hb, hlen, hle, hunp, hout, hproc⟩ := hinv
  have hmax : x0.length ≤ Usize.max := Slice.length_ineq x0
  have h1 : base.val + lane.val ≤ x0.val.length := by omega
  have hbmax : base.val + lane.val ≤ Usize.max := le_trans h1 hmax
  obtain ⟨iu, hiu, hiuv⟩ := WP.spec_imp_exists (Usize.add_spec (x := base) (y := lane) hbmax)
  have hiuv' : iu.val = base.val + lane.val := hiuv
  have hiub : iu.val < x.val.length := by rw [hiuv']; omega
  obtain ⟨top, htop, htopv⟩ := WP.spec_imp_exists (Slice.index_usize_spec x iu hiub)
  have htopget : x.val[iu.val]? = some top := by
    rw [List.getElem?_eq_getElem hiub, ← htopv]
  have h2 : iu.val + span.val ≤ x0.val.length := by rw [hiuv']; omega
  have hismax : iu.val + span.val ≤ Usize.max := le_trans h2 hmax
  obtain ⟨iu1, hiu1, hiu1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := iu) (y := span) hismax)
  have hiu1v' : iu1.val = iu.val + span.val := hiu1v
  have hiu1b : iu1.val < x.val.length := by rw [hiu1v', hiuv']; omega
  obtain ⟨bottom, hbottom, hbottomv⟩ := WP.spec_imp_exists (Slice.index_usize_spec x iu1 hiu1b)
  have hbottomget : x.val[iu1.val]? = some bottom := by
    rw [List.getElem?_eq_getElem hiu1b, ← hbottomv]
  have hunp_l := (hunp lane.val le_rfl hlane).1
  have hunp_r := (hunp lane.val le_rfl hlane).2
  have hx0top : x0.val[base.val + lane.val]? = some top := by
    rw [hiuv'] at htopget
    rw [htopget] at hunp_l
    exact hunp_l.symm
  have hiueq3 : iu1.val = base.val + span.val + lane.val := by rw [hiu1v', hiuv']; omega
  have hx0bot : x0.val[base.val + span.val + lane.val]? = some bottom := by
    rw [hiueq3] at hbottomget
    rw [hbottomget] at hunp_r
    exact hunp_r.symm
  have hbnd := hov lane.val hlane top bottom hx0top hx0bot
  obtain ⟨sum, hsum, hsumv⟩ := I64_checked_add_some top bottom hbnd.1 hbnd.2.1
  obtain ⟨diff, hdiff, hdiffv⟩ := fwht_checked_sub_some top bottom hbnd.2.2.1 hbnd.2.2.2
  obtain ⟨x1, hupd1, hx1⟩ := WP.spec_imp_exists (Slice.update_spec x iu sum hiub)
  have hiu1b' : iu1.val < x1.val.length := by
    rw [hx1, Slice.set_val_eq, List.length_set, hiu1v', hiuv']; omega
  obtain ⟨s, hupd2, hs⟩ := WP.spec_imp_exists (Slice.update_spec x1 iu1 diff hiu1b')
  have hl1max : lane.val + (1#usize).val ≤ Usize.max := by simp; omega
  obtain ⟨lane1, haddl, hlane1v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := lane) (y := 1#usize) hl1max)
  have hlane1v' : lane1.val = lane.val + 1 := by rw [hlane1v]; simp
  have hpres : ∀ k, k ≠ iu.val → k ≠ iu1.val → s.val[k]? = x.val[k]? := by
    intro k hk1 hk2
    rw [hs, hx1, Slice.set_val_eq, Slice.set_val_eq]
    rw [List.getElem?_set, List.getElem?_set, if_neg (by omega), if_neg (by omega)]
  have hnew_l : s.val[base.val + lane.val]? = some sum := by
    rw [hs, hx1, Slice.set_val_eq, Slice.set_val_eq]
    rw [List.getElem?_set, List.getElem?_set, if_neg (by omega : ¬ iu1.val = base.val + lane.val),
      if_pos (by omega : iu.val = base.val + lane.val),
      if_pos (by omega : iu.val < x.val.length)]
  have hnew_r : s.val[base.val + span.val + lane.val]? = some diff := by
    have hb2 : iu1.val < (x.val.set iu.val sum).length := by
      rw [List.length_set, hiu1v', hiuv']; omega
    rw [hs, hx1, Slice.set_val_eq, Slice.set_val_eq]
    rw [List.getElem?_set, if_pos (by omega : iu1.val = base.val + span.val + lane.val),
      if_pos hb2]
  have hslen : s.length = x0.length := by
    rw [hs, hx1]
    simp only [Slice.set_length]
    exact hlen
  refine ⟨s, lane1, ?_, ?_, hlane1v'⟩
  · rw [hadamard.fwht_int_loop0_loop0_loop0.body]
    rw [if_pos ((UScalar.lt_equiv lane span).mpr hlane)]
    simp only [if_true, hiu, htop, hiu1, hbottom, lift, hsum, hdiff,
      core.option.Option.ok_or_some, bind_tc_ok, hupd1, hupd2, haddl]
  · refine ⟨hb, hslen, by omega, ?_, ?_, ?_⟩
    · intro i hi hlt
      have h1' := (hunp i (by omega) hlt).1
      have h2' := (hunp i (by omega) hlt).2
      exact ⟨by rw [hpres (base.val + i) (by omega) (by omega)]; exact h1',
        by rw [hpres (base.val + span.val + i) (by omega) (by omega)]; exact h2'⟩
    · intro j hj
      have h0 := hout j hj
      rw [hpres j (by omega) (by omega)]
      exact h0
    · intro _ i hi
      by_cases hie : i = lane.val
      · subst hie
        exact ⟨top, bottom, sum, diff, hx0top, hx0bot, hnew_l, hnew_r, hsumv, hdiffv⟩
      · have hilt : i < lane.val := by omega
        obtain ⟨a, b, c, d, ha, hb', hc, hd, hcv, hdv⟩ := hproc rfl i hilt
        exact ⟨a, b, c, d, ha, hb',
          by rw [hpres (base.val + i) (by omega) (by omega)]; exact hc,
          by rw [hpres (base.val + span.val + i) (by omega) (by omega)]; exact hd,
          hcv, hdv⟩

/-- The innermost lane loop: `loop.spec_decr_nat` with `FwhtLaneInv` and measure `span − lane`, on
the no-overflow path. -/
theorem fwhtLane_loop_spec (x0 : Slice I64) (span base : Usize)
    (hbase : base.val + 2 * span.val ≤ x0.val.length)
    (hov : ∀ i, i < span.val → ∀ a b : I64, x0.val[base.val + i]? = some a →
      x0.val[base.val + span.val + i]? = some b →
      I64.min ≤ a.val + b.val ∧ a.val + b.val ≤ I64.max ∧
      I64.min ≤ a.val - b.val ∧ a.val - b.val ≤ I64.max) :
    WP.spec (hadamard.fwht_int_loop0_loop0_loop0 x0 true none span base 0#usize)
      (fun p => ∃ x' : Slice I64,
        p = (x', true, none) ∧ FwhtLaneInv x0 x' base.val span.val span.val true) := by
  unfold hadamard.fwht_int_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Slice I64 × Bool × Option hadamard.HadamardError × Usize =>
      span.val - s.2.2.2.val)
    (inv := fun s => FwhtLaneInv x0 s.1 base.val span.val s.2.2.2.val true ∧
      s.2.1 = true ∧ s.2.2.1 = none)
  · intro s hs
    rcases s with ⟨x, ok1, err, lane⟩
    dsimp only at hs ⊢
    obtain ⟨hinv, hok, herr⟩ := hs
    by_cases hlt : lane < span
    · have hltv : lane.val < span.val := (UScalar.lt_equiv lane span).mp hlt
      obtain ⟨x1, lane1, hbody, hinv1, hlane1⟩ :=
        fwhtLane_body_step x0 x span base hov lane hinv hltv
      have hbody' : hadamard.fwht_int_loop0_loop0_loop0.body span base x ok1 err lane
          = ok (ControlFlow.cont (x1, true, none, lane1)) := by
        rw [hok, herr]; exact hbody
      rw [hbody']
      exact ⟨⟨hinv1, rfl, rfl⟩, by show span.val - lane1.val < span.val - lane.val; omega⟩
    · have hnot : ¬ (lane < span) := hlt
      have hbody : hadamard.fwht_int_loop0_loop0_loop0.body span base x ok1 err lane
          = ok (ControlFlow.done (x, ok1, err)) := by
        rw [hadamard.fwht_int_loop0_loop0_loop0.body, if_neg hnot]
      rw [hbody]
      have hle : lane.val ≤ span.val := hinv.2.2.1
      have hspan : lane.val = span.val := by
        have h := (UScalar.lt_equiv lane span).not.mp hnot
        omega
      refine ⟨x, ?_, ?_⟩
      · rw [hok, herr]
      · show FwhtLaneInv x0 x base.val span.val span.val true
        rw [hspan] at hinv
        exact hinv
  · refine ⟨?_, rfl, rfl⟩
    refine ⟨hbase, rfl, by simp, ?_, ?_, ?_⟩
    · intro i hi hlt; exact ⟨rfl, rfl⟩
    · intro j hj; rfl
    · intro _ i hi; exact absurd hi (by simp)


theorem getElem?_map_val (v : Slice I64) (m : List ℤ) (hm : v.val.map (·.val) = m) (k : ℕ) :
    (v.val[k]?).map (·.val) = m[k]? := by
  rw [← hm, List.getElem?_map]

theorem append_getElem?_drop {P Q : List ℤ} {base i : ℕ} (hP : P.length = base) :
    (P ++ Q)[base + i]? = Q[i]? := by
  rw [List.getElem?_append_right (by omega), hP]
  congr 1
  omega

theorem drop_getElem?_at (m : List ℤ) (base i : ℕ) : (m.drop base)[i]? = m[base + i]? := by
  rw [List.getElem?_drop]

theorem hov_of_abs_bound (a b : ℤ) (ha : |a| * 2 ≤ i64Ceil) (hb : |b| * 2 ≤ i64Ceil) :
    I64.min ≤ a + b ∧ a + b ≤ I64.max ∧ I64.min ≤ a - b ∧ a - b ≤ I64.max := by
  have hsum : |a + b| ≤ i64Ceil := by
    have h1 : |a| + |b| ≤ i64Ceil := by linarith
    exact le_trans (abs_add_le a b) h1
  have hsub : |a - b| ≤ i64Ceil := by
    have h1 : |a - b| ≤ |a| + |b| := by
      rw [sub_eq_add_neg, ← abs_neg b]; exact abs_add_le a (-b)
    have h2 : |a| + |b| ≤ i64Ceil := by linarith
    exact le_trans h1 h2
  exact ⟨(i64_range_of_abs_le hsum).1, (i64_range_of_abs_le hsum).2,
    (i64_range_of_abs_le hsub).1, (i64_range_of_abs_le hsub).2⟩

theorem segSweep_first (m : List ℤ) (span : ℕ) :
    segSweep m 0 span = segSweep (m.take (2 * span)) 0 span ++ m.drop (2 * span) := by
  have h1 : (m.take (2 * span)).take span = m.take span := by
    rw [List.take_take, Nat.min_eq_left (by omega)]
  have h2 : ((m.take (2 * span)).drop span).take span = (m.drop span).take span := by
    rw [List.drop_take, show 2 * span - span = span by omega, List.take_take, Nat.min_self]
  have h3 : (m.take (2 * span)).drop (2 * span) = [] := by
    rw [List.drop_take, show 2 * span - 2 * span = 0 by omega, List.take_zero]
  simp only [segSweep, List.take_zero, List.nil_append, List.drop_zero, Nat.zero_add, h1, h2, h3,
    List.append_nil, List.append_assoc]

theorem segSweep_stageAt (s base : ℕ) (m : List ℤ)
    (hbase : base + 2 * 2 ^ s ≤ m.length)
    (hdiv : 2 ^ (s + 1) ∣ base) :
    segSweep (fwhtStage s (m.take base) ++ m.drop base) base (2 ^ s)
      = fwhtStage s (m.take (base + 2 * 2 ^ s)) ++ m.drop (base + 2 * 2 ^ s) := by
  have hpad : base ≤ m.length := by omega
  have htakelen : (m.take base).length = base := by
    rw [List.length_take, Nat.min_eq_left hpad]
  have hPlen : (fwhtStage s (m.take base)).length = base := by rw [fwhtStage_length, htakelen]
  have htakediv : 2 ^ (s + 1) ∣ (m.take base).length := by rw [htakelen]; exact hdiv
  have hseg : ((m.drop base).take (2 * 2 ^ s)).length = 2 ^ (s + 1) := by
    rw [List.length_take, List.length_drop, Nat.min_eq_left (by omega)]
    rw [show 2 ^ (s + 1) = 2 * 2 ^ s by rw [pow_succ]; ring]
  have hsegdiv : 2 ^ (s + 1) ∣ ((m.drop base).take (2 * 2 ^ s)).length := by
    rw [hseg]
  have htwospan : 2 ^ s + 2 ^ s = 2 * 2 ^ s := by ring
  rw [segSweep_take_drop]
  rw [List.take_append_of_le_length (le_of_eq hPlen.symm), List.take_of_length_le (le_of_eq hPlen),
    List.drop_append_of_le_length (le_of_eq hPlen.symm), List.drop_of_length_le (le_of_eq hPlen),
    List.nil_append]
  rw [segSweep_first, segSweep_eq_fwhtStage s ((m.drop base).take (2 * 2 ^ s)) hseg,
    List.drop_drop]
  rw [← List.append_assoc,
    ← fwhtStage_append s (m.take base) ((m.drop base).take (2 * 2 ^ s)) htakediv hsegdiv,
    ← htwospan, ← List.take_add]

/-- Base-loop invariant: the prefix `[0, base)` is transformed and the rest is untouched. -/
def FwhtBaseInv (m0 : List ℤ) (span : Usize) (s : ℕ) (x : Slice I64) (base : Usize)
    (ok1 : Bool) (err : Option hadamard.HadamardError) : Prop :=
  ok1 = true ∧ err = none ∧ base.val ≤ m0.length ∧
  2 * span.val ∣ base.val ∧
  x.val.map (·.val) = fwhtStage s (m0.take base.val) ++ m0.drop base.val

/-- One base-loop body step on the success path: run the lane loop on the block `[base, base+2span)`
and advance `base` by `2span`. -/
theorem fwhtBase_body_step (m0 : List ℤ) (span : Usize) (s : ℕ)
    (hspan : span.val = 2 ^ s)
    (hdiv : 2 * span.val ∣ m0.length)
    (hbnd : ∀ j (hj : j < m0.length), |m0[j]| * 2 ≤ i64Ceil)
    (x : Slice I64) (base : Usize)
    (hinv : FwhtBaseInv m0 span s x base true none)
    (hlt : base.val < x.val.length) :
    ∃ x' base', hadamard.fwht_int_loop0_loop0.body span x true none base
        = ok (ControlFlow.cont (x', true, none, base')) ∧
      FwhtBaseInv m0 span s x' base' true none ∧ base'.val = base.val + 2 * span.val := by
  obtain ⟨hok, herr, hble, hbadiv, hmap⟩ := hinv
  have hxlen : x.val.length = m0.length := by
    have h := congrArg List.length hmap
    rw [List.length_map, List.length_append, fwhtStage_length, List.length_take,
      List.length_drop, Nat.min_eq_left (by omega)] at h
    omega
  have hspanpos : 0 < span.val := by rw [hspan]; positivity
  obtain ⟨c, hc⟩ := hdiv
  obtain ⟨d, hd⟩ := hbadiv
  have hbase2 : base.val + 2 * span.val ≤ x.val.length := by
    have h1 : base.val < m0.length := by rw [hxlen] at hlt; exact hlt
    rw [hd, hc] at h1
    have hdc : d < c := Nat.lt_of_mul_lt_mul_left h1
    have hm : d + 1 ≤ c := by omega
    rw [hxlen, hc, hd]
    rw [show (2 * span.val) * d + 2 * span.val = (2 * span.val) * (d + 1) by ring]
    exact Nat.mul_le_mul_left _ hm
  have hP : (fwhtStage s (m0.take base.val)).length = base.val := by
    rw [fwhtStage_length, List.length_take, Nat.min_eq_left (by omega)]
  have hov : ∀ i, i < span.val → ∀ a b : I64, x.val[base.val + i]? = some a →
      x.val[base.val + span.val + i]? = some b →
      I64.min ≤ a.val + b.val ∧ a.val + b.val ≤ I64.max ∧
      I64.min ≤ a.val - b.val ∧ a.val - b.val ≤ I64.max := by
    intro i hi a b ha hb
    have haval : m0[base.val + i]? = some a.val := by
      have h := getElem?_map_val x (fwhtStage s (m0.take base.val) ++ m0.drop base.val) hmap
        (base.val + i)
      rw [ha] at h
      simp only [Option.map_some] at h
      rw [append_getElem?_drop hP] at h
      rw [drop_getElem?_at] at h
      exact h.symm
    have hbval : m0[base.val + span.val + i]? = some b.val := by
      have h := getElem?_map_val x (fwhtStage s (m0.take base.val) ++ m0.drop base.val) hmap
        (base.val + span.val + i)
      rw [hb] at h
      simp only [Option.map_some] at h
      rw [List.getElem?_append_right (by rw [hP]; omega), hP,
        show base.val + span.val + i - base.val = span.val + i by omega] at h
      rw [show (m0.drop base.val)[span.val + i]? = m0[base.val + span.val + i]? from by
        rw [drop_getElem?_at m0 base.val (span.val + i)]
        congr 1
        omega] at h
      exact h.symm
    obtain ⟨hja, hja'⟩ := List.getElem?_eq_some_iff.mp haval
    obtain ⟨hjb, hjb'⟩ := List.getElem?_eq_some_iff.mp hbval
    have hba := hbnd (base.val + i) hja
    have hbb := hbnd (base.val + span.val + i) hjb
    rw [hja'] at hba
    rw [hjb'] at hbb
    exact hov_of_abs_bound a.val b.val hba hbb
  obtain ⟨p, hloop, hpost⟩ := WP.spec_imp_exists (fwhtLane_loop_spec x span base hbase2 hov)
  obtain ⟨x1, hp, hinv1⟩ := hpost
  have hlane_eq : hadamard.fwht_int_loop0_loop0_loop0 x true none span base 0#usize
      = ok (x1, true, none) := by rw [hloop, hp]
  obtain ⟨hb1, hlen1, hle1, hunp1, hout1, hproc1⟩ := hinv1
  have hseg := map_eq_segSweep_of_lane (x0 := x) (x := x1) base.val span.val hlen1 hbase2
    (fun i hi => hproc1 rfl i hi) (fun j hj => hout1 j hj)
  have hbase2' : base.val + 2 * 2 ^ s ≤ m0.length := by
    rw [hspan] at hbase2
    omega
  have hdiv' : 2 ^ (s + 1) ∣ base.val := by
    refine ⟨d, ?_⟩
    rw [hspan] at hd
    rw [hd, show 2 * 2 ^ s = 2 ^ (s + 1) by ring]
  have hstep := segSweep_stageAt s base.val m0 hbase2' hdiv'
  have hnew : x1.val.map (·.val)
      = fwhtStage s (m0.take (base.val + 2 * span.val)) ++ m0.drop (base.val + 2 * span.val) := by
    rw [hseg, hmap, hspan]
    exact hstep
  have hi1max : (2#usize).val * span.val ≤ Usize.max := by
    have h2 : (2#usize).val = 2 := by simp
    have hxmax : x.val.length ≤ Usize.max := Slice.length_ineq x
    have hsp : 2 * span.val ≤ Usize.max := by omega
    rw [h2]; exact hsp
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := 2#usize) (y := span) hi1max)
  have hi1v' : i1.val = 2 * span.val := by rw [hi1v]; simp
  have hbase1max : base.val + i1.val ≤ Usize.max := by
    have hxmax : x.val.length ≤ Usize.max := Slice.length_ineq x
    have hsp : base.val + 2 * span.val ≤ Usize.max := by omega
    rw [hi1v']; exact hsp
  obtain ⟨base1, hbase1, hbase1v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := base) (y := i1) hbase1max)
  have hbase1v' : base1.val = base.val + 2 * span.val := by rw [hbase1v, hi1v']
  refine ⟨x1, base1, ?_, ?_, hbase1v'⟩
  · rw [hadamard.fwht_int_loop0_loop0.body]
    rw [if_pos ((UScalar.lt_equiv base (Slice.len x)).mpr (by rw [Slice.len_val]; exact hlt)),
      if_pos rfl]
    simp only [hlane_eq, hi1, hbase1, bind_tc_ok]
    rfl
  · refine ⟨rfl, rfl, ?_, ?_, ?_⟩
    · rw [hbase1v', ← hxlen]; exact hbase2
    · refine ⟨d + 1, ?_⟩
      rw [hbase1v', hd]; ring
    · rw [hbase1v']; exact hnew

/-- The base loop: `loop.spec_decr_nat` with `FwhtBaseInv` and measure `x0.length − base`, on the
no-overflow path at every block. -/
theorem fwhtBase_loop_spec (x0 : Slice I64) (span : Usize) (s : ℕ)
    (hspan : span.val = 2 ^ s)
    (hdiv : 2 * span.val ∣ x0.val.length)
    (hbnd : ∀ j (hj : j < x0.val.length), |(x0.val[j]).val| * 2 ≤ i64Ceil) :
    WP.spec (hadamard.fwht_int_loop0_loop0 x0 true none span 0#usize)
      (fun p => ∃ x' : Slice I64,
        p = (x', true, none) ∧ x'.val.map (·.val) = fwhtStage s (x0.val.map (·.val))) := by
  have hm0bnd : ∀ j (hj : j < (x0.val.map (·.val)).length),
      |(x0.val.map (·.val))[j]| * 2 ≤ i64Ceil := by
    intro j hj
    rw [List.length_map] at hj
    rw [List.getElem_map]
    exact hbnd j hj
  have hdivm : 2 * span.val ∣ (x0.val.map (·.val)).length := by rwa [List.length_map]
  unfold hadamard.fwht_int_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Slice I64 × Bool × Option hadamard.HadamardError × Usize =>
      x0.val.length - st.2.2.2.val)
    (inv := fun st => FwhtBaseInv (x0.val.map (·.val)) span s st.1 st.2.2.2 st.2.1 st.2.2.1 ∧
      st.2.1 = true ∧ st.2.2.1 = none)
  · intro st hst
    rcases st with ⟨x, ok1, err, base⟩
    dsimp only at hst ⊢
    obtain ⟨hinv, hok, herr⟩ := hst
    by_cases hlt : base < Slice.len x
    · have hltv : base.val < x.val.length := by
        have h := (UScalar.lt_equiv base (Slice.len x)).mp hlt
        rwa [Slice.len_val] at h
      rw [hok, herr] at hinv
      obtain ⟨x1, base1, hbody, hinv1, hbase1⟩ :=
        fwhtBase_body_step (x0.val.map (·.val)) span s hspan hdivm hm0bnd x base hinv hltv
      have hbody' : hadamard.fwht_int_loop0_loop0.body span x ok1 err base
          = ok (ControlFlow.cont (x1, true, none, base1)) := by
        rw [hok, herr]; exact hbody
      rw [hbody']
      refine ⟨⟨hinv1, rfl, rfl⟩, ?_⟩
      have hspanpos : 0 < span.val := by rw [hspan]; positivity
      have hb1le : base1.val ≤ x0.val.length := by
        have := hinv1.2.2.1
        rwa [List.length_map] at this
      have hb0 : base.val ≤ x0.val.length := by
        have := hinv.2.2.1
        rwa [List.length_map] at this
      show x0.val.length - base1.val < x0.val.length - base.val
      have hlt1 : base.val < base1.val := by rw [hbase1]; omega
      omega
    · have hnot : ¬ (base < Slice.len x) := hlt
      have hbody : hadamard.fwht_int_loop0_loop0.body span x ok1 err base
          = ok (ControlFlow.done (x, ok1, err)) := by
        rw [hadamard.fwht_int_loop0_loop0.body, if_neg hnot]
      rw [hbody]
      obtain ⟨hbok, hberr, hble, hbadiv, hmap⟩ := hinv
      have hxle : x.val.length ≤ base.val := by
        have h := (UScalar.lt_equiv base (Slice.len x)).not.mp hnot
        rw [Slice.len_val] at h
        exact Nat.le_of_not_lt h
      have hble' : base.val ≤ x0.val.length := by
        have h := hble; rwa [List.length_map] at h
      have hxlen2 : x.val.length = (x0.val.map (·.val)).length := by
        have h := congrArg List.length hmap
        rw [List.length_map, List.length_append, fwhtStage_length, List.length_take,
          List.length_drop, List.length_map, Nat.min_eq_left hble'] at h
        have h2 : base.val + ((x0.val.map (·.val)).length - base.val)
            = (x0.val.map (·.val)).length := Nat.add_sub_cancel' hble
        rw [List.length_map] at h2
        rw [h2] at h
        rw [List.length_map]
        exact h
      have hbaseeq : base.val = (x0.val.map (·.val)).length :=
        le_antisymm hble (hxlen2 ▸ hxle)
      refine ⟨x, ?_, ?_⟩
      · rw [hbok, hberr]
      · rw [hmap, hbaseeq, List.take_length, List.drop_length, List.append_nil]
  · refine ⟨?_, rfl, rfl⟩
    exact ⟨rfl, rfl, by simp, by simp, by simp⟩


/-- `2^16 ≤ Usize.max`, the bound that makes `span · 2` fit. -/
theorem usize_max_ge_65536 : 2 ^ 16 ≤ Usize.max := by
  have hmax : Usize.max = 2 ^ System.Platform.numBits - 1 := by
    rw [Usize.max, Usize.numBits, UScalarTy.Usize_numBits_eq]
  rw [hmax]
  rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> norm_num

/-- Stage-loop invariant: `stage` completed stages, `span = 2^stage`, the map is `fwhtRef stage`, and
the window scaled by `2^(K − stage)`. -/
def FwhtStageInv (x0 : Slice I64) (K : ℕ) (x : Slice I64) (stage span : Usize)
    (ok1 : Bool) (err : Option hadamard.HadamardError) : Prop :=
  ok1 = true ∧ err = none ∧ stage.val ≤ K ∧ span.val = 2 ^ stage.val ∧
  x.val.length = x0.val.length ∧
  x.val.map (·.val) = fwhtRef stage.val (x0.val.map (·.val)) ∧
  (∀ j (hj : j < x.val.length), |(x.val[j]).val| * 2 ^ (K - stage.val) ≤ i64Ceil)

/-- One stage-loop body step on the success path: run the base loop over every block at span
`2^stage` and advance `stage`, `span`. -/
theorem fwhtStage_body_step (x0 : Slice I64) (K : ℕ) (hK1 : 1 ≤ K) (hK16 : K ≤ 16)
    (hdiv : 2 ^ K ∣ x0.val.length)
    (hwinm : ∀ y ∈ (x0.val.map (·.val)), |y| * 2 ^ K ≤ i64Ceil)
    (k : Usize) (hk : k.val = K)
    (x : Slice I64) (stage span : Usize) (hinv : FwhtStageInv x0 K x stage span true none)
    (hlt : stage.val < K) :
    ∃ x' stage' span', hadamard.fwht_int_loop0.body k x true none stage span
        = ok (ControlFlow.cont (x', true, none, stage', span')) ∧
      FwhtStageInv x0 K x' stage' span' true none ∧ stage'.val = stage.val + 1 ∧
      span'.val = 2 * span.val := by
  obtain ⟨hok, herr, hle, hsp, hlen, hmap, hbound⟩ := hinv
  have hspos : 0 < span.val := by rw [hsp]; positivity
  have hdivs : 2 * span.val ∣ x.val.length := by
    rw [hsp, hlen]
    rw [show 2 * 2 ^ stage.val = 2 ^ (stage.val + 1) by ring]
    exact dvd_trans (Nat.pow_dvd_pow 2 (by omega)) hdiv
  have hbnd : ∀ j (hj : j < x.val.length), |(x.val[j]).val| * 2 ≤ i64Ceil := by
    intro j hj
    have h0 := hbound j hj
    have hle2 : (2 : ℤ) ≤ 2 ^ (K - stage.val) := by
      calc (2 : ℤ) = 2 ^ 1 := by norm_num
        _ ≤ 2 ^ (K - stage.val) := pow_le_pow_right₀ (by norm_num) (by omega)
    calc |(x.val[j]).val| * 2 ≤ |(x.val[j]).val| * 2 ^ (K - stage.val) :=
          mul_le_mul_of_nonneg_left hle2 (abs_nonneg _)
      _ ≤ i64Ceil := h0
  obtain ⟨p, hbase, hpost⟩ := WP.spec_imp_exists
    (fwhtBase_loop_spec x span stage.val hsp hdivs hbnd)
  obtain ⟨x1, hp, hx1map⟩ := hpost
  have hbase_eq : hadamard.fwht_int_loop0_loop0 x true none span 0#usize
      = ok (x1, true, none) := by rw [hbase, hp]
  have hnewmap : x1.val.map (·.val) = fwhtRef (stage.val + 1) (x0.val.map (·.val)) := by
    rw [hx1map, hmap, fwhtRef_succ]
  have hlen1 : x1.val.length = x0.val.length := by
    have h := congrArg List.length hnewmap
    rw [List.length_map, fwhtRef_length, List.length_map] at h
    exact h
  have hnewbound : ∀ j (hj : j < x1.val.length),
      |(x1.val[j]).val| * 2 ^ (K - (stage.val + 1)) ≤ i64Ceil := by
    intro j hj
    have hmem0 : (x1.val[j]).val ∈ x1.val.map (·.val) :=
      List.mem_map.mpr ⟨x1.val[j], List.getElem_mem hj, rfl⟩
    rw [hnewmap] at hmem0
    have h2 := fwhtRef_abs_le K (stage.val + 1) (Nat.succ_le_of_lt hlt) (x0.val.map (·.val))
      hwinm (x1.val[j]).val hmem0
    have hk1 : K - (stage.val + 1) = K - stage.val - 1 := by omega
    rwa [hk1]
  have hspmax : span.val * (2#usize).val ≤ Usize.max := by
    have h2 : span.val * 2 ≤ 2 ^ 16 := by
      rw [show span.val * 2 = 2 * span.val by ring, hsp,
        show 2 * 2 ^ stage.val = 2 ^ (stage.val + 1) by ring]
      exact Nat.pow_le_pow_right (by norm_num) (by omega)
    exact le_trans h2 usize_max_ge_65536
  obtain ⟨span1, hspan1, hspan1v⟩ :=
    WP.spec_imp_exists (Usize.mul_spec (x := span) (y := 2#usize) hspmax)
  have hspan1v' : span1.val = 2 * span.val := by
    rw [hspan1v]; simp [Nat.mul_comm]
  have hstmax : stage.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by simp
    rw [h1]
    have h3 : stage.val + 1 ≤ 16 := by omega
    have h4 : (16 : ℕ) ≤ 2 ^ 16 := by norm_num
    exact le_trans (le_trans h3 h4) usize_max_ge_65536
  obtain ⟨stage1, hstage1, hstage1v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := stage) (y := 1#usize) hstmax)
  have hstage1v' : stage1.val = stage.val + 1 := by rw [hstage1v]; simp
  refine ⟨x1, stage1, span1, ?_, ?_, hstage1v', hspan1v'⟩
  · rw [hadamard.fwht_int_loop0.body]
    rw [if_pos ((UScalar.lt_equiv stage k).mpr (by rw [hk]; exact hlt)), if_pos rfl,
      hbase_eq, hspan1, hstage1]
    simp only [bind_tc_ok]
    rfl
  · refine ⟨rfl, rfl, by omega, ?_, hlen1, ?_, ?_⟩
    · rw [hspan1v', hstage1v', hsp]; ring
    · rw [hstage1v']; exact hnewmap
    · rw [hstage1v']; exact hnewbound

/-- The stage loop: `loop.spec_decr_nat` with `FwhtStageInv` and measure `K − stage`, on the
no-overflow path. -/
theorem fwhtStage_loop_spec (x0 : Slice I64) (k : Usize) (K : ℕ)
    (hk : k.val = K) (hK1 : 1 ≤ K) (hK16 : K ≤ 16)
    (hdiv : 2 ^ K ∣ x0.val.length)
    (hwinm : ∀ y ∈ (x0.val.map (·.val)), |y| * 2 ^ K ≤ i64Ceil) :
    WP.spec (hadamard.fwht_int_loop0 x0 k true none 0#usize 1#usize)
      (fun p => ∃ x' : Slice I64, p = (x', none) ∧
        x'.val.map (·.val) = fwhtRef K (x0.val.map (·.val))) := by
  unfold hadamard.fwht_int_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Slice I64 × Bool × Option hadamard.HadamardError × Usize × Usize =>
      K - st.2.2.2.1.val)
    (inv := fun st => FwhtStageInv x0 K st.1 st.2.2.2.1 st.2.2.2.2 st.2.1 st.2.2.1 ∧
      st.2.1 = true ∧ st.2.2.1 = none)
  · intro st hst
    rcases st with ⟨x, ok1, err, stage, span⟩
    dsimp only at hst ⊢
    obtain ⟨hinv, hok, herr⟩ := hst
    by_cases hlt : stage < k
    · have hltv : stage.val < K := by
        have h := (UScalar.lt_equiv stage k).mp hlt
        rwa [hk] at h
      rw [hok, herr] at hinv
      obtain ⟨x1, stage1, span1, hbody, hinv1, hstage1, hspan1⟩ :=
        fwhtStage_body_step x0 K hK1 hK16 hdiv hwinm k hk x stage span hinv hltv
      have hbody' : hadamard.fwht_int_loop0.body k x ok1 err stage span
          = ok (ControlFlow.cont (x1, true, none, stage1, span1)) := by
        rw [hok, herr]; exact hbody
      rw [hbody']
      refine ⟨⟨hinv1, rfl, rfl⟩, ?_⟩
      show K - stage1.val < K - stage.val
      rw [hstage1]; omega
    · have hnot : ¬ (stage < k) := hlt
      have hbody : hadamard.fwht_int_loop0.body k x ok1 err stage span
          = ok (ControlFlow.done (x, err)) := by
        rw [hadamard.fwht_int_loop0.body, if_neg hnot]
      rw [hbody]
      obtain ⟨hbok, hberr, hble, hsp, hlen, hmap, hbound⟩ := hinv
      have hstageK : stage.val = K := by
        have h := (UScalar.lt_equiv stage k).not.mp hnot
        rw [hk] at h
        omega
      refine ⟨x, ?_, ?_⟩
      · rw [hberr]
      · rw [hmap, hstageK]
  · refine ⟨?_, rfl, rfl⟩
    unfold FwhtStageInv
    dsimp only
    refine ⟨rfl, rfl, by simp, by simp, rfl, ?_, ?_⟩
    · rw [show (0#usize).val = 0 by simp, fwhtRef_zero]
    · intro j hj
      have := hwinm (x0.val[j]).val
        (List.mem_map.mpr ⟨x0.val[j], List.getElem_mem hj, rfl⟩)
      rwa [show (0#usize).val = 0 by simp, Nat.sub_zero]


/-- **`fwht_int` law.** For `block = 2^k` with `1 ≤ k ≤ 16`, `block ∣ x.len` and the no-overflow
window `|x_i| · 2^k ≤ 2^63 − 1`, the extracted `hadamard.fwht_int` returns `Ok ()` with the `k`-stage
reference `fwhtRef k` on the value list. -/
theorem fwht_int_spec (x : Slice I64) (block k : Usize)
    (hb : block.val = 2 ^ k.val) (hk1 : 1 ≤ k.val) (hk16 : k.val ≤ 16)
    (hdiv : block.val ∣ x.val.length)
    (hwin : ∀ i (hi : i < x.val.length), |(x.val[i]).val| * 2 ^ k.val ≤ i64Ceil) :
    ∃ x' : Slice I64, hadamard.fwht_int x block = ok (core.result.Result.Ok (), x') ∧
      x'.val.map (·.val) = fwhtRef k.val (x.val.map (·.val)) := by
  have hexp := block_exponent_spec_pow block k hb hk1 hk16
  have hdiv' : 2 ^ k.val ∣ x.val.length := by rw [← hb]; exact hdiv
  have hwinm : ∀ y ∈ (x.val.map (·.val)), |y| * 2 ^ k.val ≤ i64Ceil := by
    intro y hy
    obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hy
    obtain ⟨i, hi, hiv⟩ := List.mem_iff_getElem.mp hv
    rw [← hiv]
    exact hwin i hi
  have hmod : (Slice.len x).val % block.val = 0 := by
    rw [Slice.len_val]
    exact Nat.dvd_iff_mod_eq_zero.mp hdiv
  have hmul : core.num.Usize.is_multiple_of (Slice.len x) block = ok true := by
    unfold core.num.Usize.is_multiple_of UScalar.is_multiple_of
    rw [hmod]
    rfl
  obtain ⟨p, hloop, hpost⟩ := WP.spec_imp_exists
    (fwhtStage_loop_spec x k k.val rfl hk1 hk16 hdiv' hwinm)
  obtain ⟨x1, hp, hmap⟩ := hpost
  have hloop' : hadamard.fwht_int_loop0 x k true none 0#usize 1#usize
      = ok (x1, none) := by rw [hloop, hp]
  refine ⟨x1, ?_, hmap⟩
  rw [hadamard.fwht_int]
  simp only [hexp, bind_tc_ok, hmul, hloop']
  rfl

/-- An aligned `2^k` segment of a `m·2^k`-long list has length `2^k`. -/
theorem sylvester_seg_length (k m : ℕ) (xs : List ℤ) (hm : xs.length = m * 2 ^ k)
    (i : ℕ) (hi : i < m) :
    ((xs.drop (i * 2 ^ k)).take (2 ^ k)).length = 2 ^ k := by
  rw [List.length_take, List.length_drop, hm]
  have h1 : i * 2 ^ k + 2 ^ k ≤ m * 2 ^ k := by
    have h2 : i + 1 ≤ m := by omega
    calc i * 2 ^ k + 2 ^ k = (i + 1) * 2 ^ k := by ring
      _ ≤ m * 2 ^ k := Nat.mul_le_mul_right _ h2
  exact Nat.min_eq_left (by omega)

/-- **Aligned-segment law.** Under the `fwht_int` hypotheses, each aligned `2^k` segment of `x'` is
`(sylvesterH k).mulVec` applied to the corresponding segment of `x`, indexed through `Fin (2^k)`. -/
theorem fwht_int_sylvester (x : Slice I64) (block k : Usize)
    (hb : block.val = 2 ^ k.val) (hk1 : 1 ≤ k.val) (hk16 : k.val ≤ 16)
    (hdiv : block.val ∣ x.val.length)
    (hwin : ∀ i (hi : i < x.val.length), |(x.val[i]).val| * 2 ^ k.val ≤ i64Ceil)
    (m : ℕ) (hm : x.val.length = m * 2 ^ k.val) (i : ℕ) (hi : i < m) :
    ∃ x' : Slice I64, hadamard.fwht_int x block = ok (core.result.Result.Ok (), x') ∧
      (∀ j : Fin (2 ^ k.val),
        (((x'.val.map (·.val)).drop (i * 2 ^ k.val)).take (2 ^ k.val))[j.val]?
          = some ((sylvesterH k.val).mulVec
              (fun l : Fin (2 ^ k.val) =>
                (((x.val.map (·.val)).drop (i * 2 ^ k.val)).take (2 ^ k.val)).get
                  (Fin.cast (sylvester_seg_length k.val m (x.val.map (·.val))
                    (by rw [List.length_map]; exact hm) i hi).symm l)) j)) := by
  obtain ⟨x', hok, hmap⟩ := fwht_int_spec x block k hb hk1 hk16 hdiv hwin
  refine ⟨x', hok, ?_⟩
  intro j
  have hm' : (x.val.map (·.val)).length = m * 2 ^ k.val := by
    rw [List.length_map]; exact hm
  have hlen : (((x.val.map (·.val)).drop (i * 2 ^ k.val)).take (2 ^ k.val)).length
      = 2 ^ k.val := sylvester_seg_length k.val m (x.val.map (·.val)) hm' i hi
  have hdrop : (x'.val.map (·.val)).drop (i * 2 ^ k.val)
      = (fwhtRef k.val (x.val.map (·.val))).drop (i * 2 ^ k.val) := by rw [hmap]
  have hseg := fwhtRef_segment k.val m (x.val.map (·.val)) hm' i hi
  rw [hdrop, hseg]
  have hpi := fwhtRef_getElem k.val (((x.val.map (·.val)).drop (i * 2 ^ k.val)).take (2 ^ k.val))
    hlen j.val (by rw [hlen]; exact j.isLt)
  rw [List.getElem?_eq_getElem (by rw [fwhtRef_length, hlen]; exact j.isLt)]
  simp only [Option.some.injEq]
  exact hpi

end

end apeiron_word
