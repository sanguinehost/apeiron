/-
Row law for the extracted `Q3_K` dequantizer (`APEIRON-ALGEBRA-G3-DEQUANT-W3-b3d`).

The extracted block reader, the reference scale/code definitions `q3kScaleRef`/`q3kCodeRef` and the
inner-loop specifications are in `Numeric.DequantQ3KBlock`. This module holds the lane and group
loops over the 256-entry value array (`q3kLaneWeight`/`q3kLaneInv`, `q3kGroupInv` and their
specifications), the raw-level weight references `q3kD`/`q3kScaleRefRaw`/`q3kCodeRefRaw`/
`q3kWeight`/`q3kRowValues`, the success domain `q3kBlockDomain`, the row law
`dequant_q3_k_row_spec`, and the one-block joint non-vacuity witness `q3kBlockDomain_witness`.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed
literal facts, never on an extracted body.
-/
import Numeric.DequantQ3KBlock

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## `dequant::q6_k_block` loop scaffolding

The lane loop (`dequant_q3_k_row_loop0_loop0_loop0`) fills the sixteen entries `16·group + lane`
of the 256-entry value array with `slope·codes[16·group + lane]`, from `slope = d·scales[group]`.
`q3kLaneWeight` is the reference value, `q3kLaneInv` the accumulator invariant. -/

/-- Reference value of lane `l` of `group`: `slope` times the signed code at `16·group + l`. -/
def q3kLaneWeight (slope : I64) (codes : Array I8 256#usize) (group l : Nat) : ℤ :=
  slope.val * (codes.val.getD (16 * group + l) 0#i8).val

/-- Lane-loop invariant: lanes `0..lane-1` of `group` hold `q3kLaneWeight`, every other entry is
`vals0`. -/
def q3kLaneInv (vals0 vals : Array I64 256#usize) (codes : Array I8 256#usize) (slope : I64)
    (group lane : Nat) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ lane ≤ 16 ∧
  (∀ j, j < 256 → (vals.val.getD j 0#i64).val =
    if j / 16 = group ∧ j % 16 < lane then q3kLaneWeight slope codes group (j % 16)
    else (vals0.val.getD j 0#i64).val)



/-- `dequant::mul` on the success path returns the exact `i64` product. -/
theorem q3k_mul_ok (a b : I64)
    (hlo : I64.min ≤ a.val * b.val) (hhi : a.val * b.val ≤ I64.max) :
    ∃ z : I64, dequant.mul a b = ok (.Ok z) ∧ z.val = a.val * b.val := by
  have hspec := I64.checked_mul_bv_spec a b
  cases hc : core.num.checked_mul_IScalar a b with
  | none => rw [hc] at hspec; exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      refine ⟨z, ?_, hspec.2.2.1⟩
      unfold dequant.mul
      rw [show I64.checked_mul a b = some z by simpa only [I64.checked_mul] using hc]
      simp only [lift, bind_tc_ok]
      rw [core.option.Option.ok_or_some]

/-- One lane-loop body step on the success path. -/
theorem q3kLane_body_step (vals0 vals : Array I64 256#usize) (codes : Array I8 256#usize)
    (slope : I64) (group lane : Usize) (hgroup : group.val < 16) (hlane : lane.val < 16)
    (hmul : I64.min ≤ slope.val * (codes.val.getD (16 * group.val + lane.val) 0#i8).val ∧
      slope.val * (codes.val.getD (16 * group.val + lane.val) 0#i8).val ≤ I64.max)
    (hinv : q3kLaneInv vals0 vals codes slope group.val lane.val true none) :
    ∃ (ok2 : Bool) (err1 : Option error.EngineError) (vals1 : Array I64 256#usize) (lane2 : Usize),
      dequant.dequant_q3_k_row_loop0_loop0_loop0.body codes group slope true none vals lane
          = ok (ControlFlow.cont (ok2, err1, vals1, lane2)) ∧
      ok2 = true ∧ err1 = none ∧ lane2.val = lane.val + 1 ∧
      q3kLaneInv vals0 vals1 codes slope group.val (lane.val + 1) ok2 err1 := by
  obtain ⟨i, hm16, hiv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := 16#usize) (y := group) (by scalar_tac))
  have hiv' : i.val = 16 * group.val := by rw [hiv]; simp
  obtain ⟨i1, hadd, hi1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := lane) (by rw [hiv']; scalar_tac))
  have hi1' : i1.val = 16 * group.val + lane.val := by rw [hi1v, hiv']
  obtain ⟨i2, hidx, hi2v⟩ := WP.spec_imp_exists
    (Array.index_usize_spec codes i1
      (by show i1.val < codes.val.length; rw [Array.property codes, hi1']; scalar_tac))
  have hi2val : i2.val = (codes.val.getD (16 * group.val + lane.val) 0#i8).val := by
    have hlt : i1.val < codes.val.length := by rw [Array.property codes, hi1']; scalar_tac
    have h : i2.val = (codes.val.getD i1.val 0#i8).val := by
      rw [hi2v]
      exact congrArg (fun x : I8 => x.val) (getD_of_lt_i8 (l := codes.val) hlt).symm
    rwa [hi1'] at h
  obtain ⟨v, hmulv, hvval⟩ := q3k_mul_ok slope (core.convert.num.FromI64I8.from i2)
    (by rw [core.convert.num.FromI64I8.from_val_eq, hi2val]; exact hmul.1)
    (by rw [core.convert.num.FromI64I8.from_val_eq, hi2val]; exact hmul.2)
  obtain ⟨a, hupd, hau⟩ := WP.spec_imp_exists
    (Array.update_spec vals i1 v
      (by show i1.val < vals.val.length; rw [Array.property vals, hi1']; scalar_tac))
  obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
  have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
  have hdiv : i1.val / 16 = group.val := by rw [hi1']; exact div16_mul_add group.val lane.val hlane
  have hmod : i1.val % 16 = lane.val := by rw [hi1']; exact mod16_mul_add group.val lane.val hlane
  refine ⟨true, none, a, lane2, ?_, rfl, rfl, hlane2, ?_⟩
  · rw [dequant.dequant_q3_k_row_loop0_loop0_loop0.body]
    rw [if_pos (by scalar_tac : lane < 16#usize)]
    simp only [if_true, hm16, hadd, hidx, lift, hmulv, hupd, haddl, bind_tc_ok]
    rfl
  · refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    have hvals_len : vals.val.length = 256 := Array.property vals
    rw [hau, Array.set_val_eq, getD_set]
    by_cases hjk : j = i1.val
    · subst hjk
      rw [if_pos ⟨rfl, by rw [hvals_len]; rw [hi1']; omega⟩,
        if_pos ⟨hdiv, by omega⟩, hmod]
      unfold q3kLaneWeight
      rw [hvval, core.convert.num.FromI64I8.from_val_eq, hi2val]
    · rw [if_neg (fun h => hjk h.1), hinv.2.2.2 j hj]
      by_cases hcond : j / 16 = group.val ∧ j % 16 < lane.val
      · rw [if_pos hcond, if_pos ⟨hcond.1, by omega⟩]
      · rw [if_neg hcond]
        have hnot : ¬ (j / 16 = group.val ∧ j % 16 < lane.val + 1) := by
          rintro ⟨h1, h2⟩
          refine hcond ⟨h1, ?_⟩
          by_contra hge
          have hge' : lane.val ≤ j % 16 := not_lt.mp hge
          have hj16 : j = 16 * group.val + j % 16 := eq_of_div16 j group.val h1
          have : j % 16 = lane.val := by omega
          exact hjk (by rw [hj16, this, hi1'])
        rw [if_neg hnot]

/-- The lane loop: `loop.spec_decr_nat` with `q3kLaneInv` and measure `16 - lane`, on the
no-overflow path (`hmul` at every lane).

Complexity: 16 iterations, each O(1) (one array read, one checked multiply, one array write); the
array is 256 words, never resized. -/
theorem q3kLane_loop_spec (vals0 : Array I64 256#usize) (codes : Array I8 256#usize)
    (slope : I64) (group : Usize) (hgroup : group.val < 16)
    (hmul : ∀ l, l < 16 → I64.min ≤ slope.val * (codes.val.getD (16 * group.val + l) 0#i8).val ∧
      slope.val * (codes.val.getD (16 * group.val + l) 0#i8).val ≤ I64.max)
    (hinv0 : q3kLaneInv vals0 vals0 codes slope group.val 0 true none) :
    WP.spec (dequant.dequant_q3_k_row_loop0_loop0_loop0 true none codes vals0 group slope 0#usize)
      (fun b => ∃ vals : Array I64 256#usize,
        b = (true, none, vals) ∧ q3kLaneInv vals0 vals codes slope group.val 16 true none) := by
  unfold dequant.dequant_q3_k_row_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Bool × Option error.EngineError × Array I64 256#usize × Usize =>
      16 - s.2.2.2.val)
    (inv := fun s => q3kLaneInv vals0 s.2.2.1 codes slope group.val s.2.2.2.val s.1 s.2.1)
  · intro s hs
    rcases s with ⟨ok1, err, vals, lane⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    have hok' : ok1 = true := hok
    have herr' : err = none := herr
    rw [hok', herr']
    by_cases hlt : lane.val < 16
    · obtain ⟨ok2, err1, vals1, lane2, hbody, hok2, herr1, hlane2, hinv2⟩ :=
        q3kLane_body_step vals0 vals codes slope group lane hgroup hlt (hmul lane.val hlt)
          ⟨rfl, rfl, hle, hrec⟩
      simp only [hbody]
      refine ⟨by simpa [hlane2] using hinv2, ?_⟩
      show 16 - lane2.val < 16 - lane.val
      omega
    · have hnot : ¬ (lane < 16#usize) := fun hc => hlt ((UScalar.lt_equiv lane 16#usize).1 hc)
      have hbody : dequant.dequant_q3_k_row_loop0_loop0_loop0.body codes group slope true none vals lane
          = ok (ControlFlow.done (true, none, vals)) := by
        rw [dequant.dequant_q3_k_row_loop0_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      refine ⟨vals, rfl, ?_⟩
      have hlane16 : lane.val = 16 := by omega
      refine ⟨rfl, rfl, by omega, ?_⟩
      intro j hj
      rw [hrec j hj, hlane16]
  · exact hinv0

/-! ## `dequant::q3_k_block` group loop

`dequant_q3_k_row_loop0_loop0` runs the sixteen groups; group `g` computes `slope = d·scales[g]`
and invokes the lane loop. `q3kGroupInv` accumulates the reference value
`d·scales[j/16]·codes[j]` for every `j` whose group is complete. -/

/-- Group-loop invariant: every entry whose group is below `group` holds
`d·scales[j/16]·codes[j]`; every other entry is `vals0`. -/
def q3kGroupInv (d : I64) (scales : Array I8 16#usize) (codes : Array I8 256#usize)
    (vals0 vals : Array I64 256#usize) (group : Nat) (ok1 : Bool) (err : Option error.EngineError) :
    Prop :=
  ok1 = true ∧ err = none ∧ group ≤ 16 ∧
  (∀ j, j < 256 → (vals.val.getD j 0#i64).val =
    if j / 16 < group then
      d.val * ((scales.val.getD (j / 16) 0#i8).val - 32) * (codes.val.getD j 0#i8).val
    else (vals0.val.getD j 0#i64).val)

/-- One group-loop body step on the no-overflow path. -/
theorem q3kGroup_body_step (d : I64) (scales : Array I8 16#usize) (codes : Array I8 256#usize)
    (vals0 vals : Array I64 256#usize) (group : Usize) (hgroup : group.val < 16)
    (hsclt : group.val < scales.val.length)
    (hmul : I64.min ≤ d.val * ((scales.val.getD group.val 0#i8).val - 32) ∧
      d.val * ((scales.val.getD group.val 0#i8).val - 32) ≤ I64.max)
    (hlane : ∀ l, l < 16 →
      I64.min ≤ d.val * ((scales.val.getD group.val 0#i8).val - 32)
          * (codes.val.getD (16 * group.val + l) 0#i8).val ∧
      d.val * ((scales.val.getD group.val 0#i8).val - 32)
          * (codes.val.getD (16 * group.val + l) 0#i8).val ≤ I64.max)
    (hinv : q3kGroupInv d scales codes vals0 vals group.val true none) :
    ∃ (ok3 : Bool) (err2 : Option error.EngineError) (vals1 : Array I64 256#usize) (group2 : Usize),
      dequant.dequant_q3_k_row_loop0_loop0.body d scales codes true none vals group
          = ok (ControlFlow.cont (ok3, err2, vals1, group2)) ∧
      ok3 = true ∧ err2 = none ∧ group2.val = group.val + 1 ∧
      q3kGroupInv d scales codes vals0 vals1 (group.val + 1) ok3 err2 := by
  obtain ⟨sc, hidx, hscv⟩ := WP.spec_imp_exists (Array.index_usize_spec scales group hsclt)
  have hscval : sc.val = (scales.val.getD group.val 0#i8).val := by
    rw [hscv]
    exact congrArg (fun x : I8 => x.val) (getD_of_lt_i8 (l := scales.val) hsclt).symm
  obtain ⟨sub, hsub, hsubv⟩ := WP.spec_imp_exists
    (IScalar.sub_spec (x := core.convert.num.FromI64I8.from sc) (y := 32#i64)
      (by rw [core.convert.num.FromI64I8.from_val_eq, hscval]; scalar_tac)
      (by rw [core.convert.num.FromI64I8.from_val_eq, hscval]; scalar_tac))
  have h32 : (32#i64).val = (32 : ℤ) := by decide
  have hsubval : sub.val = (scales.val.getD group.val 0#i8).val - 32 := by
    rw [hsubv, core.convert.num.FromI64I8.from_val_eq, hscval, h32]
  obtain ⟨slope, hmulv, hslopev⟩ := q3k_mul_ok d sub
    (by rw [hsubval]; exact hmul.1)
    (by rw [hsubval]; exact hmul.2)
  have hslope : slope.val = d.val * ((scales.val.getD group.val 0#i8).val - 32) := by
    rw [hslopev, hsubval]
  have hinv0 : q3kLaneInv vals vals codes slope group.val 0 true none := by
    refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    rw [if_neg (by omega : ¬ (j / 16 = group.val ∧ j % 16 < 0))]
  obtain ⟨b1, hloop2, vals1, hb1, hlaneinv⟩ := WP.spec_imp_exists
    (q3kLane_loop_spec vals codes slope group hgroup
      (fun l hl => by rw [hslope]; exact hlane l hl) hinv0)
  have hloop2' : dequant.dequant_q3_k_row_loop0_loop0_loop0 true none codes vals group slope 0#usize
      = ok (true, none, vals1) := by rw [hloop2, hb1]
  obtain ⟨group2, haddg, hgroup2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := group) (y := 1#usize) (by scalar_tac))
  have hgroup2 : group2.val = group.val + 1 := by rw [hgroup2v]; simp
  refine ⟨true, none, vals1, group2, ?_, rfl, rfl, hgroup2, ?_⟩
  · rw [dequant.dequant_q3_k_row_loop0_loop0.body]
    rw [if_pos (by scalar_tac : group < 16#usize)]
    simp only [if_true, hidx, lift, hsub, hmulv, haddg, bind_tc_ok]
    change (dequant.dequant_q3_k_row_loop0_loop0_loop0 true none codes vals group slope 0#usize
        >>= fun b => ok (ControlFlow.cont (b.1, b.2.1, b.2.2, group2)))
      = ok (ControlFlow.cont (true, none, vals1, group2))
    rw [hloop2']
    simp only [bind_tc_ok]
  · refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    have hb := hlaneinv.2.2.2 j hj
    by_cases hlt : j / 16 < group.val
    · rw [hb, if_neg (by omega : ¬ (j / 16 = group.val ∧ j % 16 < 16)),
        hinv.2.2.2 j hj, if_pos hlt, if_pos (by omega)]
    · rw [hb]
      by_cases heq : j / 16 = group.val
      · rw [if_pos ⟨heq, by omega⟩, if_pos (by omega)]
        unfold q3kLaneWeight
        rw [hslope, heq, ← eq_of_div16 j group.val heq]
      · rw [if_neg (fun h => heq h.1), hinv.2.2.2 j hj, if_neg hlt, if_neg (by omega)]

/-- The group loop: `loop.spec_decr_nat` with `q3kGroupInv` and measure `16 - group`, on the
no-overflow path.

Complexity: 16 groups, each running the 16-lane loop; O(1) array writes in total per group. -/
theorem q3kGroup_loop_spec (d : I64) (scales : Array I8 16#usize) (codes : Array I8 256#usize)
    (vals0 : Array I64 256#usize)
    (hmul : ∀ g, g < 16 → I64.min ≤ d.val * ((scales.val.getD g 0#i8).val - 32) ∧
      d.val * ((scales.val.getD g 0#i8).val - 32) ≤ I64.max)
    (hlane : ∀ g, g < 16 → ∀ l, l < 16 →
      I64.min ≤ d.val * ((scales.val.getD g 0#i8).val - 32)
          * (codes.val.getD (16 * g + l) 0#i8).val ∧
      d.val * ((scales.val.getD g 0#i8).val - 32)
          * (codes.val.getD (16 * g + l) 0#i8).val ≤ I64.max)
    (hinv0 : q3kGroupInv d scales codes vals0 vals0 0 true none) :
    WP.spec (dequant.dequant_q3_k_row_loop0_loop0 true none d scales codes vals0 0#usize)
      (fun b => ∃ vals : Array I64 256#usize,
        b = (true, none, vals) ∧ q3kGroupInv d scales codes vals0 vals 16 true none) := by
  unfold dequant.dequant_q3_k_row_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Bool × Option error.EngineError × Array I64 256#usize × Usize =>
      16 - s.2.2.2.val)
    (inv := fun s => q3kGroupInv d scales codes vals0 s.2.2.1 s.2.2.2.val s.1 s.2.1)
  · intro s hs
    rcases s with ⟨ok1, err, vals, group⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    have hok' : ok1 = true := hok
    have herr' : err = none := herr
    rw [hok', herr']
    by_cases hlt : group.val < 16
    · have hsclt : group.val < scales.val.length := by
        rw [Array.property scales]; exact hlt
      obtain ⟨ok3, err2, vals1, group2, hbody, hok3, herr2, hgroup2, hinv2⟩ :=
        q3kGroup_body_step d scales codes vals0 vals group hlt hsclt (hmul group.val hlt)
          (hlane group.val hlt) ⟨rfl, rfl, hle, hrec⟩
      simp only [hbody]
      refine ⟨by simpa [hgroup2] using hinv2, ?_⟩
      show 16 - group2.val < 16 - group.val
      omega
    · have hnot : ¬ (group < 16#usize) := fun hc => hlt ((UScalar.lt_equiv group 16#usize).1 hc)
      have hbody : dequant.dequant_q3_k_row_loop0_loop0.body d scales codes true none vals group
          = ok (ControlFlow.done (true, none, vals)) := by
        rw [dequant.dequant_q3_k_row_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      refine ⟨vals, rfl, ?_⟩
      have hgroup16 : group.val = 16 := by omega
      refine ⟨rfl, rfl, by omega, ?_⟩
      intro j hj
      rw [hrec j hj, hgroup16]
  · exact hinv0

/-! ## The Q3_K weight row: raw references, block domain and loop invariants -/

/-- The finite `f16` scale `d` of block `b`, read at byte `110·b + 108`. -/
def q3kD (raw : Slice U8) (b : Nat) : ℤ := f16Value (f16LeHalf raw (110 * b + 108))

/-- The signed sub-block scale `k` of block `b` at the raw level (12-byte window). -/
def q3kScaleRefRaw (raw : Slice U8) (b k : Nat) : ℤ :=
  q3kScaleRef ((raw.val.drop (110 * b + 96)).take 12) k

/-- The signed code `i` of block `b` at the raw level (`hmask` bytes `0..32`, `qs` bytes `32..96`). -/
def q3kCodeRefRaw (raw : Slice U8) (b i : Nat) : ℤ :=
  q3kCodeRef ((raw.val.drop (110 * b)).take 32) (((raw.val.drop (110 * b)).drop 32).take 64) i

/-- The exact weight `j` of a Q3_K row: `d_{j/256}·(scale_{(j%256)/16} − 32)·q_{j%256}`. -/
def q3kWeight (raw : Slice U8) (j : Nat) : ℤ :=
  q3kD raw (j / 256) * (q3kScaleRefRaw raw (j / 256) ((j % 256) / 16) - 32)
    * q3kCodeRefRaw raw (j / 256) (j % 256)

/-- The first `n` exact Q32.32 integers of a Q3_K row. -/
def q3kRowValues (raw : Slice U8) (n : Nat) : List ℤ := (List.range n).map (q3kWeight raw)

/-- `f16LeHalf` of a 110-byte window at byte 108 is the raw half-word at `s + 108`. -/
lemma f16LeHalf_slice110 (raw : Slice U8) (s : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice s (s + 110)) (he : s + 110 ≤ raw.val.length) :
    f16LeHalf blk 108 = f16LeHalf raw (s + 108) := by
  unfold f16LeHalf
  congr 2
  rw [h, getD_slice raw.val s (s + 110) 108 he (by omega),
    getD_slice raw.val s (s + 110) 109 he (by omega)]

/-- The 12-byte scale window of a 110-byte block window is the raw-level window. -/
lemma q3kScaleWindow_slice (raw : Slice U8) (b : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice (110 * b) (110 * b + 110)) :
    (blk.val.drop 96).take 12 = (raw.val.drop (110 * b + 96)).take 12 := by
  rw [h, List.slice, List.drop_take, List.drop_drop, List.take_take]
  congr 1
  omega

/-- The code planes of a 110-byte block window are the raw-level planes. -/
lemma q3kCodePlanes_slice (raw : Slice U8) (b : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice (110 * b) (110 * b + 110)) :
    blk.val.take 32 = (raw.val.drop (110 * b)).take 32 ∧
    (blk.val.drop 32).take 64 = ((raw.val.drop (110 * b)).drop 32).take 64 := by
  constructor
  · rw [h, List.slice, List.take_take]
    congr 1
    omega
  · rw [h, List.slice, List.drop_take, List.drop_drop, List.take_take]
    congr 1
    omega

lemma q3kScaleRef_slice (raw : Slice U8) (b : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice (110 * b) (110 * b + 110)) (k : Nat) :
    q3kScaleRef ((blk.val.drop 96).take 12) k = q3kScaleRefRaw raw b k := by
  unfold q3kScaleRefRaw
  rw [q3kScaleWindow_slice raw b blk h]

lemma q3kCodeRef_slice (raw : Slice U8) (b : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice (110 * b) (110 * b + 110)) (i : Nat) :
    q3kCodeRef (blk.val.take 32) ((blk.val.drop 32).take 64) i = q3kCodeRefRaw raw b i := by
  obtain ⟨h1, h2⟩ := q3kCodePlanes_slice raw b blk h
  unfold q3kCodeRefRaw
  rw [h1, h2]

/-- The `j`-th entry of block `b`'s 256-value array is raw weight `256·b + j`. -/
lemma q3kWeight_block (raw : Slice U8) (b j : Nat) (hj : j < 256) :
    q3kWeight raw (256 * b + j)
      = q3kD raw b * (q3kScaleRefRaw raw b (j / 16) - 32) * q3kCodeRefRaw raw b j := by
  unfold q3kWeight
  have h1 : (256 * b + j) / 256 = b := by omega
  have h2 : (256 * b + j) % 256 = j := by omega
  rw [h1, h2]

/-- Appending the next `take` raw weights advances the row prefix by `take`. -/
lemma q3kRowValues_append (raw : Slice U8) (m take : Nat) :
    q3kRowValues raw m ++ (List.range take).map (fun j => q3kWeight raw (m + j)) =
      q3kRowValues raw (m + take) := by
  unfold q3kRowValues
  rw [List.range_add, List.map_append, List.map_map]
  congr 1

/-- The success domain of one 110-byte block `b`. -/
def q3kBlockDomain (raw : Slice U8) (b : Usize) : Prop :=
  ∃ blk : Slice U8,
    dequant.block_slice raw b 110#usize = ok (.Ok blk) ∧
    blk.val = raw.val.slice (110 * b.val) (110 * b.val + 110) ∧
    110 * b.val + 110 ≤ raw.val.length ∧
    110 ≤ blk.val.length ∧
    f16Exp (f16LeHalf blk 108) ≠ 31 ∧
    (∀ g, g < 16 → I64.min ≤ q3kD raw b.val * (q3kScaleRefRaw raw b.val g - 32) ∧
      q3kD raw b.val * (q3kScaleRefRaw raw b.val g - 32) ≤ I64.max) ∧
    (∀ g, g < 16 → ∀ l, l < 16 →
      I64.min ≤ q3kD raw b.val * (q3kScaleRefRaw raw b.val g - 32) * q3kCodeRefRaw raw b.val (16 * g + l) ∧
      q3kD raw b.val * (q3kScaleRefRaw raw b.val g - 32) * q3kCodeRefRaw raw b.val (16 * g + l) ≤ I64.max)

/-- Block-loop invariant: after `block` completed blocks the output is the initial output followed
by the first `min (256·block) n` raw weights, with the success flag set. -/
def q3kRowInv (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (blocks block : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ block.val ≤ blocks.val ∧
  dequantVals out.val = dequantVals out0.val ++ q3kRowValues raw (Nat.min (256 * block.val) n.val)

/-- One block-loop body step on the success domain: decode block `block`, run the group loop,
append `take = min 256 (n - 256·block)` values and advance. -/
theorem q3kBlock_body_step (raw : Slice U8) (n blocks : Usize) (out0 out : alloc.vec.Vec I64)
    (block : Usize) (hblock : block.val < blocks.val)
    (hblocks : blocks.val = (n.val + 255) / 256)
    (hdom : q3kBlockDomain raw block)
    (hcap : out.val.length + (n.val - 256 * block.val) ≤ Usize.max)
    (hinv : q3kRowInv raw n out0 out blocks block true none) :
    ∃ (out1 : alloc.vec.Vec I64) (block1 : Usize),
      dequant.dequant_q3_k_row_loop0.body raw n blocks out block true none =
        ok (ControlFlow.cont (out1, block1, true, none)) ∧
      q3kRowInv raw n out0 out1 blocks block1 true none ∧ block1.val = block.val + 1 := by
  obtain ⟨blk, hblk, hslice, hbnd, h110, hfin, hnol, hnol2⟩ := hdom
  obtain ⟨d, scales, codes, hq3, hdv, hscv, hcv⟩ := q3_k_block_ok raw block blk hblk h110 hfin
  have hdv' : d.val = q3kD raw block.val := by
    rw [hdv, f16LeHalf_slice110 raw (110 * block.val) blk hslice hbnd]; rfl
  have hscv' : ∀ k, k < 16 → (scales.val.getD k 0#i8).val = q3kScaleRefRaw raw block.val k := by
    intro k hk; rw [hscv k hk]; exact q3kScaleRef_slice raw block.val blk hslice k
  have hcv' : ∀ i, i < 256 → (codes.val.getD i 0#i8).val = q3kCodeRefRaw raw block.val i := by
    intro i hi; rw [hcv i hi]; exact q3kCodeRef_slice raw block.val blk hslice i
  have hmul : ∀ g, g < 16 → I64.min ≤ d.val * ((scales.val.getD g 0#i8).val - 32) ∧
      d.val * ((scales.val.getD g 0#i8).val - 32) ≤ I64.max := by
    intro g hg
    obtain ⟨h1, h2⟩ := hnol g hg
    rw [hdv', hscv' g hg]
    exact ⟨h1, h2⟩
  have hlane : ∀ g, g < 16 → ∀ l, l < 16 →
      I64.min ≤ d.val * ((scales.val.getD g 0#i8).val - 32) * (codes.val.getD (16 * g + l) 0#i8).val ∧
      d.val * ((scales.val.getD g 0#i8).val - 32) * (codes.val.getD (16 * g + l) 0#i8).val ≤ I64.max := by
    intro g hg l hl
    obtain ⟨h1, h2⟩ := hnol2 g hg l hl
    rw [hdv', hscv' g hg, hcv' (16 * g + l) (by omega)]
    exact ⟨h1, h2⟩
  have hltmul : 256 * block.val < n.val :=
    lt_mul_of_lt_div_ceil_256 (by rw [← hblocks]; exact hblock)
  have hinv0 : q3kGroupInv d scales codes (Array.repeat 256#usize 0#i64)
      (Array.repeat 256#usize 0#i64) 0 true none := by
    refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    rw [if_neg (by omega : ¬ j / 16 < 0)]
  obtain ⟨b1, hloop, vals1, hb1, hgrp⟩ := WP.spec_imp_exists
    (q3kGroup_loop_spec d scales codes (Array.repeat 256#usize 0#i64) hmul hlane hinv0)
  have hloop' : dequant.dequant_q3_k_row_loop0_loop0 true none d scales codes
      (Array.repeat 256#usize 0#i64) 0#usize = ok (true, none, vals1) := by rw [hloop, hb1]
  obtain ⟨i, hiv, hivv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := quant.QK_K) (by simp only [quant.QK_K]; scalar_tac))
  have hivv' : i.val = 256 * block.val := by rw [hivv]; simp [quant.QK_K]; ring
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := n) (y := i) (by rw [hivv']; omega))
  have hi1v' : i1.val = n.val - 256 * block.val := by rw [hi1v.1, hivv']
  obtain ⟨take, htake, htakev⟩ := q6k_min_ok quant.QK_K i1
  have htakev' : take.val = Nat.min 256 (n.val - 256 * block.val) := by
    rw [htakev, hi1v']; simp [quant.QK_K]
  obtain ⟨block1, hblock1a, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) (by scalar_tac))
  have hblock1v' : block1.val = block.val + 1 := by rw [hblock1v]; simp
  have htake_le : take.val ≤ 256 := by rw [htakev']; exact min_le_left _ _
  have htake_X : take.val ≤ n.val - 256 * block.val := by rw [htakev']; exact min_le_right _ _
  have hcap' : out.val.length + take.val ≤ Usize.max := by omega
  obtain ⟨out1, hpush, hpushv⟩ := WP.spec_imp_exists
    (q6k_push_prefix_spec out vals1 take htake_le hcap')
  have hgrp_j : ∀ j, j < 256 → (vals1.val.getD j 0#i64).val =
      d.val * ((scales.val.getD (j / 16) 0#i8).val - 32) * (codes.val.getD j 0#i8).val := by
    intro j hj
    have h := hgrp.2.2.2 j hj
    rwa [if_pos (by omega : j / 16 < 16)] at h
  have hmap : (List.range take.val).map (fun j => (vals1.val.getD j 0#i64).val) =
      (List.range take.val).map (fun j => q3kWeight raw (256 * block.val + j)) := by
    apply List.map_congr_left
    intro j hj
    have hj' : j < take.val := List.mem_range.mp hj
    have hj256 : j < 256 := by omega
    rw [hgrp_j j hj256, hdv', hscv' (j / 16) (by omega), hcv' j hj256,
      q3kWeight_block raw block.val j hj256]
  have hmin : Nat.min (256 * block.val) n.val = 256 * block.val :=
    Nat.min_eq_left (le_of_lt hltmul)
  have hsum : 256 * block.val + take.val = Nat.min (256 * (block.val + 1)) n.val := by
    rw [htakev']; exact mul_block_add_min block.val n.val (le_of_lt hltmul)
  have hinv_new : dequantVals out1.val = dequantVals out0.val ++
      q3kRowValues raw (Nat.min (256 * block1.val) n.val) := by
    rw [hblock1v']
    rw [hpushv, hmap, hinv.2.2.2, hmin, List.append_assoc, q3kRowValues_append, hsum]
  refine ⟨out1, block1, ?_, ⟨rfl, rfl, by omega, hinv_new⟩, hblock1v'⟩
  rw [dequant.dequant_q3_k_row_loop0.body]
  rw [if_pos (by scalar_tac : block < blocks)]
  simp only [if_true]
  rw [hq3]
  simp only [bind_tc_ok]
  rw [hiv]
  simp only [bind_tc_ok]
  rw [hi1]
  simp only [bind_tc_ok]
  rw [htake]
  simp only [bind_tc_ok]
  change (do
    let (ok2, err1, vals1) ←
      dequant.dequant_q3_k_row_loop0_loop0 true none d scales codes
        (Array.repeat 256#usize 0#i64) 0#usize
    let out1 ← if ok2 = true then dequant.push_prefix out vals1 take else ok out
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out1, block1, ok2, err1)))
    = ok (ControlFlow.cont (out1, block1, true, none))
  rw [hloop']
  simp only [bind_tc_ok]
  change (do
    let out1 ← if (true : Bool) = true then dequant.push_prefix out vals1 take else ok out
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out1, block1, true, none)))
    = ok (ControlFlow.cont (out1, block1, true, none))
  rw [hpush]
  rw [hblock1a]
  rw [if_pos rfl]
  simp only [bind_tc_ok]

/-- The block loop: `loop.spec_decr_nat` with `q3kRowInv` and measure `blocks - block`. -/
theorem q3kRow_loop0_spec (raw : Slice U8) (n blocks : Usize) (out0 : alloc.vec.Vec I64)
    (hblocks : blocks.val = (n.val + 255) / 256)
    (hdom : ∀ b : Usize, b.val < blocks.val → q3kBlockDomain raw b)
    (hcap : out0.val.length + n.val ≤ Usize.max) :
    WP.spec (dequant.dequant_q3_k_row_loop0 raw n out0 blocks 0#usize true none)
      (fun p => p.2 = none ∧
        dequantVals p.1.val = dequantVals out0.val ++ q3kRowValues raw n.val) := by
  unfold dequant.dequant_q3_k_row_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
      blocks.val - s.2.1.val)
    (inv := fun s => q3kRowInv raw n out0 s.1 blocks s.2.1 s.2.2.1 s.2.2.2)
  · intro s hs
    rcases s with ⟨out, block, ok1, err⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    rw [hok, herr]
    by_cases hlt : block.val < blocks.val
    · have hltmul : 256 * block.val < n.val :=
        lt_mul_of_lt_div_ceil_256 (by rw [← hblocks]; exact hlt)
      have hmin : Nat.min (256 * block.val) n.val = 256 * block.val :=
        Nat.min_eq_left (le_of_lt hltmul)
      have hlen_out : out.val.length = out0.val.length + 256 * block.val := by
        have h := congrArg List.length hrec
        rw [hmin] at h
        simpa [dequantVals, q3kRowValues, List.length_append, List.length_map,
          List.length_range] using h
      have hcap' : out.val.length + (n.val - 256 * block.val) ≤ Usize.max := by
        rw [hlen_out]; omega
      obtain ⟨out1, block1, hbody, hinv1, hblock1⟩ :=
        q3kBlock_body_step raw n blocks out0 out block hlt hblocks (hdom block hlt) hcap'
          ⟨rfl, rfl, hle, hrec⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv1, ?_⟩
      show blocks.val - block1.val < blocks.val - block.val
      omega
    · have hnot : ¬ (block < blocks) := fun hc => hlt ((UScalar.lt_equiv block blocks).1 hc)
      have hbody : dequant.dequant_q3_k_row_loop0.body raw n blocks out block true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.dequant_q3_k_row_loop0.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨trivial, ?_⟩
      have hblock_eq : block.val = blocks.val := by omega
      rw [hrec, hblock_eq, hblocks]
      have hminend : Nat.min (256 * ((n.val + 255) / 256)) n.val = n.val :=
        Nat.min_eq_right (le_mul_div_ceil_256 n.val)
      rw [hminend]
  · refine ⟨rfl, rfl, by simp, ?_⟩
    have h0 : Nat.min (256 * (0#usize).val) n.val = 0 := by simp
    rw [h0]
    simp [q3kRowValues]

/-- **`Q3_K` row law.** On the block success domain, the extracted `dequant_q3_k_row` succeeds
with `Ok ()` and appends exactly the first `n` raw weights `q3kRowValues raw n` to the output. -/
theorem dequant_q3_k_row_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 255) / 256 → q3kBlockDomain raw b)
    (hcap : out.val.length + n.val ≤ Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.dequant_q3_k_row raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ q3kRowValues raw n.val := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_div_ceil_256_ok n
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (q3kRow_loop0_spec raw n blocks out hblocks
      (fun b hb => hdom b (by rw [← hblocks]; exact hb)) hcap)
  obtain ⟨out1, err⟩ := p
  obtain ⟨herr, hvals⟩ := hp
  have herr' : err = none := herr
  subst err
  refine ⟨out1, ?_, hvals⟩
  unfold dequant.dequant_q3_k_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-! ## One-block joint domain witness -/

/-- Every signed sub-block scale lies in `[-128, 127]`. -/
lemma q3kScaleRef_bounds (l : List U8) (k : Nat) :
    -128 ≤ q3kScaleRef l k ∧ q3kScaleRef l k ≤ 127 := by
  unfold q3kScaleRef
  rw [Int.bmod_eq_emod]
  have h0 : 0 ≤ ((q3kWord l (k / 4) >>> (8 * (k % 4)) : Nat) : ℤ) % 256 :=
    Int.emod_nonneg _ (by norm_num)
  have h1 : ((q3kWord l (k / 4) >>> (8 * (k % 4)) : Nat) : ℤ) % 256 < 256 :=
    Int.emod_lt_of_pos _ (by norm_num)
  split <;> omega

/-- Every signed code lies in `[-4, 3]`. -/
lemma q3kCodeRef_bounds (hm qs : List U8) (i : Nat) :
    -4 ≤ q3kCodeRef hm qs i ∧ q3kCodeRef hm qs i ≤ 3 := by
  unfold q3kCodeRef
  dsimp only
  have hq : ((qs.getD (i / 128 * 32 + i % 32) 0#u8).val >>> (2 * (i % 128 / 32)) &&& 3) ≤ 3 :=
    Nat.and_le_right
  split <;> omega

/-- The 110-byte `Q3_K` witness block: all zeros and `d = 1.0` (`0x00 0x3C`) at byte 108. -/
def q3kWitBytes : List U8 := List.replicate 108 0#u8 ++ [0#u8, 60#u8]

theorem q3kWitBytes_length : q3kWitBytes.length = 110 := by simp [q3kWitBytes]

def q3kWitRaw : Slice U8 := Slice.from q3kWitBytes (by rw [q3kWitBytes_length]; scalar_tac)

theorem q3kWitRaw_val : q3kWitRaw.val = q3kWitBytes := Slice.from_val _ _

lemma q3kWit_f16LeHalf_108 : (f16LeHalf q3kWitRaw 108).val = 15360 := by
  have h0 : q3kWitRaw.val.getD 108 0#u8 = 0#u8 := by rw [q3kWitRaw_val]; decide
  have h1 : q3kWitRaw.val.getD (108 + 1) 0#u8 = 60#u8 := by rw [q3kWitRaw_val]; decide
  have heq : f16LeHalf q3kWitRaw 108 =
      core.num.U16.from_le_bytes (Array.make 2#usize [0#u8, 60#u8]) :=
    f16LeHalf_eq_of_getD q3kWitRaw 108 0#u8 60#u8 h0 h1
  rw [heq, u16_le_0_60]

lemma q3kWit_d : q3kD q3kWitRaw 0 = 2 ^ 32 := by
  have hv := q3kWit_f16LeHalf_108
  unfold q3kD
  rw [show 110 * 0 + 108 = 108 by ring]
  simp only [f16Value, f16Mag, f16Sign, f16Exp, f16Man, hv]
  norm_num

lemma q3kWit_d_finite : f16Exp (f16LeHalf q3kWitRaw 108) ≠ 31 := by
  have hv := q3kWit_f16LeHalf_108
  unfold f16Exp
  rw [hv]
  norm_num

/-- **Joint non-vacuity of `q3kBlockDomain` at block `0`.** A 110-byte all-zero block with
`d = 1.0` satisfies every conjunct of the one-block success domain. -/
theorem q3kBlockDomain_witness : q3kBlockDomain q3kWitRaw 0#usize := by
  obtain ⟨blk, hblk, hslice⟩ := block_slice_ok q3kWitRaw 0#usize 110#usize (by
    rw [q3kWitRaw_val, q3kWitBytes_length]; norm_num)
  refine ⟨blk, hblk, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hslice
  · rw [q3kWitRaw_val, q3kWitBytes_length]; norm_num
  · rw [hslice, List.slice_length]; simp [q3kWitRaw_val, q3kWitBytes_length]
  · have hs : f16LeHalf blk 108 = f16LeHalf q3kWitRaw 108 :=
      f16LeHalf_slice110 q3kWitRaw 0 blk (by simpa using hslice)
        (by rw [q3kWitRaw_val, q3kWitBytes_length])
    rw [hs]
    exact q3kWit_d_finite
  · intro g hg
    have hs := q3kScaleRef_bounds ((q3kWitRaw.val.drop (110 * 0 + 96)).take 12) g
    unfold q3kScaleRefRaw
    rw [show (0#usize).val = 0 from by decide, q3kWit_d, I64.min_eq, I64.max_eq]
    constructor <;> omega
  · intro g hg l hl
    have hs := q3kScaleRef_bounds ((q3kWitRaw.val.drop (110 * 0 + 96)).take 12) g
    have hc := q3kCodeRef_bounds ((q3kWitRaw.val.drop (110 * 0)).take 32)
      (((q3kWitRaw.val.drop (110 * 0)).drop 32).take 64) (16 * g + l)
    unfold q3kScaleRefRaw q3kCodeRefRaw
    rw [show (0#usize).val = 0 from by decide, q3kWit_d, I64.min_eq, I64.max_eq]
    constructor <;> nlinarith

end
