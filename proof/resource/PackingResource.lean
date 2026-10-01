import PackingSource
import Aeneas

/-
SCOPE: resource (work-measure) theorems for the actual extracted body of `Packing::decode_block`
       (`packing_source.packing.Packing.decode_block` and the field/lane loops it calls),
       extracted from the real `src/format/packing.rs` by `proof/resource/Packing_gate.py`.
REGISTER: source/resource. Work is counted in loop continuations, byte loads and appended
       `Vec` elements under the named premises of each theorem; no cycle or wall-clock quantity
       appears. No canonical-ring correspondence and no hardware timing is claimed here.
CORPUS: the generated `packing_source.packing.Packing.{decode_block, decode_block_fields,
       decode_block_fields_loop{0..8}, unpack_code, unpack_code_loop, read_field, total_code_bits,
       block_size, groups, block_bytes}` definitions in `PackingSource.lean`, extracted from
       `src/format/packing.rs` at digest sha256 3e41858ece47c56e (see the gate receipt).
REFERENCE: /home/socol/Workspace/apeiron/src/format/packing.rs (the Rust body) and
       /home/socol/Workspace/apeiron/proof/resource/gate.py (the gate template).
GATE: python3 proof/resource/Packing_gate.py --controls, 540-second task budget, lane-locked.
GRADE: Lean checked only when that gate succeeds; no hardware measurement.

## What is counted, and what is not

Counted here, each on the real extracted body: the per-code-bit transition and its closed-form
continuation count (`bit_plane_*`), the per-group coefficient-field transition and its closed-form
count (`field_*`), the per-lane code transition and its closed-form count (`lane_code_*`), the
malformed-length and geometry refusals of `decode_block_fields` (`fields_truncated`,
`fields_geometry_refusal`), and reachable work on the `F16` face (`f16_lane_work`, `f16_field_work`).
`code_plane_work` is the product of two proved per-loop ranks and nothing more (`rfl`).

Not covered: the payload-copy loops `decode_block_fields_loop0/5` and the nibble-swap loops
`loop1/6`, the duplicate loop bodies `loop2/3/8` (textually equal to `loop7`, which is covered), the
affine epilogue and its `loop0/1/2`, `decode_block`'s own branch composition over those calls, the
signed-code sign-extension face of `unpack_code` (`unpack_code_unsigned` covers the unsigned face
only), and the *values* of the geometry functions `block_size`/`groups`/`block_bytes`, which enter
only as premises (`hneed`, `hbad`). The functional value laws of these loops are the corpus files
`Numeric/Packing{Unpack,DecodeFields,EpilogueFold,Refinement}.lean`, not this file.

Trust boundary: the evidence reaches the Aeneas/Charon model of the extracted Rust body and the Lean
kernel checking of these theorems. It does not cover rustc/LLVM codegen of `src/format/packing.rs`,
the `alloc`/`core` runtime behind the `Vec` model, the kernel ABI, GPU/SPIR-V, or the `unsafe`
container. The Aeneas model, `Vec` capacity, IO/process/dependency provenance and the
`System.Platform.numBits` value used by `scalar_tac` are trusted checking code outside these proofs.
-/

namespace PackingResource

open Aeneas Aeneas.Std Result WP

/-! ## Per-lane code-bit transitions

`packing_source.packing.Packing.unpack_code_loop` is the extracted `while k < total` loop of
`Packing::unpack_code` (`src/format/packing.rs:574-583`): one iteration reads one bit of the code
area of weight `i` at stream position `i·total + k`, ORs it into the accumulator at its plane shift
and advances `k` by one. -/

/-- The plane shift of code bit `k` under the packing's plane order. -/
def planeShift (po : packing_source.packing.PlaneOrder) (total k : Nat) : Nat :=
  match po with
  | .LsbFirst => k
  | .MsbFirst => total - 1 - k

/-- Bit `pos` of the little-endian code-area byte stream. -/
def areaBit (area : List Std.U8) (pos : Nat) : Nat :=
  ((area.getD (pos / 8) 0#u8).val / 2 ^ (pos % 8)) % 2

/-- A stream bit is `0` or `1`. -/
lemma areaBit_le_one (area : List Std.U8) (pos : Nat) : areaBit area pos ≤ 1 := by
  unfold areaBit
  have := Nat.mod_lt ((area.getD (pos / 8) 0#u8).val / 2 ^ (pos % 8)) (by norm_num : 0 < 2)
  omega

/-- The bit pattern of `areaBit · 2^s` as an `I64`, via the extracted `ofNat 64` then `<<< s`. -/
lemma toNat_ofNat_shiftLeft {b s : Nat} (hb : b ≤ 1) (hs : s < 64) :
    ((BitVec.ofNat 64 b) <<< s).toNat = b * 2 ^ s := by
  rw [BitVec.toNat_shiftLeft]
  have hb64 : (BitVec.ofNat 64 b).toNat = b := by
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]
  rw [hb64, Nat.shiftLeft_eq, Nat.mod_eq_of_lt]
  calc b * 2 ^ s ≤ 1 * 2 ^ s := Nat.mul_le_mul_right _ hb
    _ = 2 ^ s := one_mul _
    _ < 2 ^ 64 := Nat.pow_lt_pow_right (by norm_num) hs

/-- **One guarded bit-plane transition.** Under the guard `k < total` and the byte bound, the
extracted body performs exactly one continuation: the code bit at stream position `i·total + k` is
ORed into the accumulator at its plane shift, the descriptor is unchanged, `k` advances by one, and
the rank `total - k` strictly decreases. The step reads one byte of the code area. -/
theorem bit_plane_step (area : Slice Std.U8) (i total : Std.Usize)
    (self : packing_source.packing.Packing) (v : Std.I64) (k : Std.Usize)
    (hk : k.val < total.val) (htot : total.val ≤ 64)
    (hmul : i.val * total.val + total.val ≤ Std.Usize.max)
    (harea : (i.val * total.val + k.val) / 8 < area.val.length) :
    packing_source.packing.Packing.unpack_code_loop.body area i total self v k ⦃ cf =>
      ∃ (vNext : Std.I64) (kNext : Std.Usize),
        cf = ControlFlow.cont (self, vNext, kNext) ∧
        kNext.val = k.val + 1 ∧
        total.val - kNext.val < total.val - k.val ∧
        vNext.bv = v.bv ||| (BitVec.ofNat 64 (areaBit area.val (i.val * total.val + k.val))
          <<< planeShift self.plane_order total.val k.val) ⦄ := by
  unfold packing_source.packing.Packing.unpack_code_loop.body
  have hlt : k < total := by scalar_tac
  simp only [hlt, if_true]
  cases hpo : self.plane_order <;> simp only [planeShift] <;> step* <;>
  · refine ⟨v1, k1, ?_, k1_post, ?_, ?_⟩
    · rw [← hpo]
    · show total.val - k1.val < total.val - k.val
      rw [k1_post]
      omega
    · refine v1_post1.trans ?_
      congr 1
      refine i7_post1.trans ?_
      congr 1
      · rw [bit_post]
        apply BitVec.eq_of_toNat_eq
        have ei2 : i2.val = (i.val * total.val + k.val) / 8 := by scalar_tac
        have ei4 : i4.val = (i.val * total.val + k.val) % 8 := by scalar_tac
        have hi3 : i3.val = (area.val.getD ((i.val * total.val + k.val) / 8) 0#u8).val := by
          rw [List.getD_eq_getElem _ _ harea]
          subst i3_post
          simp only [ei2]
        have e1 : i6.bv.toNat = i5.bv.toNat % 2 := by
          have h := congrArg BitVec.toNat i6_post1
          simpa [BitVec.toNat_and, Nat.and_one_is_mod] using h
        have e2 : i5.bv.toNat = i3.bv.toNat / 2 ^ i4.val := by
          have h := congrArg BitVec.toNat i5_post1
          simpa [BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow] using h
        have e3 : i3.bv.toNat = i3.val := Std.UScalar.bv_toNat i3
        simp only [Std.UScalar.hcast, BitVec.truncate_eq_setWidth, BitVec.toNat_ofNat, areaBit, ←
          hi3]
        simp [BitVec.toNat_setWidth, e1, e2, e3, ei4]
      all_goals scalar_tac

/-- The loop body exits without a continuation exactly when the plane index has reached `total`. -/
theorem bit_plane_exit (area : Slice Std.U8) (i total : Std.Usize)
    (self : packing_source.packing.Packing) (v : Std.I64) (k : Std.Usize)
    (hk : total.val ≤ k.val) :
    packing_source.packing.Packing.unpack_code_loop.body area i total self v k
      = ok (ControlFlow.done (self, v)) := by
  unfold packing_source.packing.Packing.unpack_code_loop.body
  have hnot : ¬ (k < total) := by rw [Std.UScalar.lt_equiv]; exact not_lt_of_ge hk
  rw [if_neg hnot]

/-! ## The bit-plane work measure

`bitPlaneSteps` counts the continuations of the *actual* extracted body: each layer is one
evaluation of `unpack_code_loop.body` at the state the previous continuation produced.
`bit_plane_work` is its closed form, proved by strong induction on the loop's own rank. -/

/-- `bitPlaneSteps … k0 v0 n`: the extracted bit-plane body, entered at plane index `k0` with
accumulator `v0`, performs exactly `n` continuations and then exits. Every layer is an evaluation
of the real extracted body at the state the previous continuation produced. -/
def bitPlaneSteps (area : Slice Std.U8) (i total : Std.Usize)
    (self : packing_source.packing.Packing) (v0 : Std.I64) (k0 : Std.Usize) : Nat → Prop
  | 0 => ∃ v, packing_source.packing.Packing.unpack_code_loop.body area i total self v0 k0
      = ok (ControlFlow.done (self, v))
  | n + 1 => ∃ (v1 : Std.I64) (k1 : Std.Usize),
      packing_source.packing.Packing.unpack_code_loop.body area i total self v0 k0
        = ok (ControlFlow.cont (self, v1, k1)) ∧
      bitPlaneSteps area i total self v1 k1 n

/-- The continuation of the extracted body at plane index `k` under the success domain, together
with its exact plane index. -/
lemma bit_plane_cont (area : Slice Std.U8) (i total : Std.Usize)
    (self : packing_source.packing.Packing) (v0 : Std.I64) (k0 : Std.Usize)
    (hlt : k0.val < total.val) (htot : total.val ≤ 64)
    (hmul : i.val * total.val + total.val ≤ Std.Usize.max)
    (harea : (i.val * total.val + k0.val) / 8 < area.val.length) :
    ∃ (v1 : Std.I64) (k1 : Std.Usize),
      packing_source.packing.Packing.unpack_code_loop.body area i total self v0 k0
        = ok (ControlFlow.cont (self, v1, k1)) ∧ k1.val = k0.val + 1 := by
  obtain ⟨cf, hcf, v1, k1, hcfv, hk1, -, -⟩ :=
    WP.spec_imp_exists (bit_plane_step area i total self v0 k0 hlt htot hmul harea)
  exact ⟨v1, k1, hcf.trans (congrArg ok hcfv), hk1⟩

/-- **Closed-form bit-plane work.** Entered at plane index `k0` on the success domain, the extracted
body performs exactly `total - k0` continuations — one per remaining code bit of the lane — and
then exits; the extracted loop itself terminates with the descriptor unchanged. -/
theorem bit_plane_work (area : Slice Std.U8) (i total : Std.Usize)
    (self : packing_source.packing.Packing) (v0 : Std.I64) (k0 : Std.Usize)
    (hk0 : k0.val ≤ total.val) (htot : total.val ≤ 64)
    (hmul : i.val * total.val + total.val ≤ Std.Usize.max)
    (harea : ∀ j : Std.Usize, j.val < total.val →
      (i.val * total.val + j.val) / 8 < area.val.length) :
    (∃ v', packing_source.packing.Packing.unpack_code_loop self area i total v0 k0
        = ok (self, v')) ∧
      (∃ n, bitPlaneSteps area i total self v0 k0 n ∧ n = total.val - k0.val) := by
  refine ⟨?_, ?_⟩
  · have hloop : packing_source.packing.Packing.unpack_code_loop self area i total v0 k0
        ⦃ (b : packing_source.packing.Packing × Std.I64) => ∃ v', b = (self, v') ⦄ := by
      unfold packing_source.packing.Packing.unpack_code_loop
      apply loop.spec_decr_nat
        (measure := fun s : packing_source.packing.Packing × Std.I64 × Std.Usize =>
          total.val - s.2.2.val)
        (inv := fun s => s.1 = self ∧ k0.val ≤ s.2.2.val)
        (post := fun b : packing_source.packing.Packing × Std.I64 => ∃ v', b = (self, v'))
      · intro x hx
        rcases x with ⟨self', v', k⟩
        dsimp only at hx ⊢
        obtain ⟨hself, hkle⟩ := hx
        by_cases hk : k.val < total.val
        · obtain ⟨v1, k1, hbody, hk1⟩ :=
            bit_plane_cont area i total self' v' k hk htot hmul (harea k hk)
          rw [hbody, hself, WP.spec_ok]
          refine ⟨⟨rfl, ?_⟩, ?_⟩
          · show k0.val ≤ k1.val
            omega
          · show total.val - k1.val < total.val - k.val
            rw [hk1]
            omega
        · rw [bit_plane_exit area i total self' v' k (by omega), hself, WP.spec_ok]
          exact ⟨v', rfl⟩
      · exact ⟨rfl, le_rfl⟩
    obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hloop
    obtain ⟨v', hv'⟩ := hpost
    exact ⟨v', hr.trans (congrArg ok hv')⟩
  · induction hn : total.val - k0.val using Nat.strong_induction_on generalizing k0 v0 with
    | h n ih =>
      by_cases hlt : k0.val < total.val
      · obtain ⟨v1, k1, hbody, hk1⟩ := bit_plane_cont area i total self v0 k0 hlt htot hmul
          (harea k0 hlt)
        have hk1le : k1.val ≤ total.val := by rw [hk1]; omega
        obtain ⟨n', hsteps, hn'⟩ :=
          ih (total.val - k1.val) (by rw [hk1]; omega) v1 k1 hk1le rfl
        exact ⟨n' + 1, ⟨v1, k1, hbody, hsteps⟩, by omega⟩
      · exact ⟨0, ⟨v0, bit_plane_exit area i total self v0 k0 (by omega)⟩, by omega⟩

/-! ## The per-group coefficient-field transitions

`packing_source.packing.Packing.decode_block_fields_loop4` is the extracted `while g < groups` loop
of `Packing::decode_block_fields` (`src/format/packing.rs:672-692`): one iteration reads one group’s
coefficient field(s) at the strand position, pushes the `(a, b)` pair and advances the strand by
`coeffStep b` bytes. -/

/-- The byte stride of one coefficient group: one coefficient byte, plus a min byte when the
grammar carries one. -/
def coeffStep (b : Bool) : Nat := if b = true then 2 else 1

/-- The signed coefficient read block of the extracted body returns an `i32`. -/
lemma readSigned_ok (raw : Slice Std.U8) (pos : Std.Usize) (h : pos.val < raw.val.length) :
    ∃ x : Std.I32,
      (do let i ← Slice.index_usize raw pos
          let i1 ← lift (Std.UScalar.hcast IScalarTy.I8 i)
          ok (IScalar.cast IScalarTy.I32 i1)) = ok x := by
  obtain ⟨i, hi, _⟩ := WP.spec_imp_exists (Slice.index_usize_spec raw pos h)
  refine ⟨IScalar.cast IScalarTy.I32 (Std.UScalar.hcast IScalarTy.I8 i), ?_⟩
  rw [hi]
  simp only [bind_tc_ok, lift]

/-- The signed coefficient read block with the `lift` folded. -/
lemma readSigned_ok_inl (raw : Slice Std.U8) (pos : Std.Usize) (h : pos.val < raw.val.length) :
    ∃ x : Std.I32,
      (do let i ← Slice.index_usize raw pos
          ok (IScalar.cast IScalarTy.I32 (Std.UScalar.hcast IScalarTy.I8 i))) = ok x := by
  obtain ⟨i, hi, _⟩ := WP.spec_imp_exists (Slice.index_usize_spec raw pos h)
  refine ⟨IScalar.cast IScalarTy.I32 (Std.UScalar.hcast IScalarTy.I8 i), ?_⟩
  rw [hi]
  simp only [bind_tc_ok]

/-- The unsigned coefficient read block of the extracted body returns an `i32`. -/
lemma readUnsigned_ok (raw : Slice Std.U8) (pos : Std.Usize) (h : pos.val < raw.val.length) :
    ∃ x : Std.I32,
      (do let i ← Slice.index_usize raw pos
          ok (Std.UScalar.hcast IScalarTy.I32 i)) = ok x := by
  obtain ⟨i, hi, _⟩ := WP.spec_imp_exists (Slice.index_usize_spec raw pos h)
  refine ⟨Std.UScalar.hcast IScalarTy.I32 i, ?_⟩
  rw [hi]
  simp only [bind_tc_ok]

/-- **One guarded coefficient-field transition.** Under the guard `g < groups` and the byte bounds,
the extracted body performs exactly one continuation: it appends one `(a, b)` pair, leaves
`signed_coeff` unchanged, advances the strand by `coeffStep has_min` bytes and the group index by
one, and the rank `groups - g` strictly decreases. -/
theorem field_step (b b1 : Bool) (raw : Slice Std.U8) (groups : Std.Usize)
    (at0 : Std.Usize) (coeffs : alloc.vec.Vec (Std.I32 × Std.I32)) (g : Std.Usize)
    (hlt : g.val < groups.val)
    (hA : at0.val < raw.val.length)
    (hB : b = true → at0.val + 1 < raw.val.length)
    (hcap : coeffs.val.length < Std.Usize.max) :
    ∃ (a m : Std.I32) (atN : Std.Usize) (coeffsN : alloc.vec.Vec (Std.I32 × Std.I32))
      (gN : Std.Usize),
      packing_source.packing.Packing.decode_block_fields_loop4.body b raw groups b1 at0 coeffs g
          = ok (ControlFlow.cont (b1, atN, coeffsN, gN)) ∧
      atN.val = at0.val + coeffStep b ∧ gN.val = g.val + 1 ∧
      coeffsN.val = coeffs.val ++ [(a, m)] ∧
      groups.val - gN.val < groups.val - g.val := by
  obtain ⟨at1, hat1, hat1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := at0) (y := 1#usize) (by scalar_tac))
  have hat1' : at1.val = at0.val + 1 := by rw [hat1v]; rfl
  obtain ⟨g1, hg1, hg1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := g) (y := 1#usize) (by scalar_tac))
  have hg1' : g1.val = g.val + 1 := by rw [hg1v]; rfl
  by_cases hb1 : b1 = true
  · by_cases hb : b = true
    · obtain ⟨xa, hxa⟩ := readSigned_ok raw at0 hA
      have hA1 : at1.val < raw.val.length := by rw [hat1']; exact hB hb
      obtain ⟨xb, hxb⟩ := readSigned_ok_inl raw at1 hA1
      obtain ⟨at3, hat3, hat3v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := at1) (y := 1#usize) (by scalar_tac))
      have hat3' : at3.val = at1.val + 1 := by rw [hat3v]; rfl
      obtain ⟨coeffs2, hpush, hpushv⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.push_spec coeffs (xa, xb) hcap)
      refine ⟨xa, xb, at3, coeffs2, g1, ?_, ?_, hg1', hpushv, ?_⟩
      · unfold packing_source.packing.Packing.decode_block_fields_loop4.body
        rw [if_pos (by rw [Std.UScalar.lt_equiv]; exact hlt), if_pos hb1, hxa]
        simp only [bind_tc_ok, lift]
        rw [hat1]
        simp only [bind_tc_ok]
        rw [if_pos hb, if_pos hb1, hxb]
        simp only [bind_tc_ok]
        rw [hat3]
        simp only [bind_tc_ok]
        change (do let coeffs1 ← coeffs.push (xa, xb)
                   let g2 ← g + 1#usize
                   ok (ControlFlow.cont (b1, at3, coeffs1, g2)))
            = ok (ControlFlow.cont (b1, at3, coeffs2, g1))
        rw [hpush, hg1]
        simp only [bind_tc_ok]
      · show at3.val = at0.val + coeffStep b
        rw [hat3', hat1', coeffStep, if_pos hb]
      · show groups.val - g1.val < groups.val - g.val
        rw [hg1']
        omega
    · obtain ⟨xa, hxa⟩ := readSigned_ok raw at0 hA
      obtain ⟨coeffs2, hpush, hpushv⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.push_spec coeffs (xa, 0#i32) hcap)
      refine ⟨xa, 0#i32, at1, coeffs2, g1, ?_, ?_, hg1', hpushv, ?_⟩
      · unfold packing_source.packing.Packing.decode_block_fields_loop4.body
        rw [if_pos (by rw [Std.UScalar.lt_equiv]; exact hlt), if_pos hb1, hxa]
        simp only [bind_tc_ok, lift]
        rw [hat1]
        simp only [bind_tc_ok]
        rw [if_neg hb]
        simp only [bind_tc_ok]
        change (do let coeffs1 ← coeffs.push (xa, 0#i32)
                   let g2 ← g + 1#usize
                   ok (ControlFlow.cont (b1, at1, coeffs1, g2)))
            = ok (ControlFlow.cont (b1, at1, coeffs2, g1))
        rw [hpush, hg1]
        simp only [bind_tc_ok]
      · show at1.val = at0.val + coeffStep b
        rw [hat1', coeffStep, if_neg hb]
      · show groups.val - g1.val < groups.val - g.val
        rw [hg1']
        omega
  · by_cases hb : b = true
    · obtain ⟨xa, hxa⟩ := readUnsigned_ok raw at0 hA
      have hA1 : at1.val < raw.val.length := by rw [hat1']; exact hB hb
      obtain ⟨xb, hxb⟩ := readUnsigned_ok raw at1 hA1
      obtain ⟨at3, hat3, hat3v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := at1) (y := 1#usize) (by scalar_tac))
      have hat3' : at3.val = at1.val + 1 := by rw [hat3v]; rfl
      obtain ⟨coeffs2, hpush, hpushv⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.push_spec coeffs (xa, xb) hcap)
      refine ⟨xa, xb, at3, coeffs2, g1, ?_, ?_, hg1', hpushv, ?_⟩
      · unfold packing_source.packing.Packing.decode_block_fields_loop4.body
        rw [if_pos (by rw [Std.UScalar.lt_equiv]; exact hlt), if_neg hb1, hxa]
        simp only [bind_tc_ok]
        rw [hat1]
        simp only [bind_tc_ok]
        rw [if_pos hb, if_neg hb1, hxb]
        simp only [bind_tc_ok]
        rw [hat3]
        simp only [bind_tc_ok]
        change (do let coeffs1 ← coeffs.push (xa, xb)
                   let g2 ← g + 1#usize
                   ok (ControlFlow.cont (b1, at3, coeffs1, g2)))
            = ok (ControlFlow.cont (b1, at3, coeffs2, g1))
        rw [hpush, hg1]
        simp only [bind_tc_ok]
      · show at3.val = at0.val + coeffStep b
        rw [hat3', hat1', coeffStep, if_pos hb]
      · show groups.val - g1.val < groups.val - g.val
        rw [hg1']
        omega
    · obtain ⟨xa, hxa⟩ := readUnsigned_ok raw at0 hA
      obtain ⟨coeffs2, hpush, hpushv⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.push_spec coeffs (xa, 0#i32) hcap)
      refine ⟨xa, 0#i32, at1, coeffs2, g1, ?_, ?_, hg1', hpushv, ?_⟩
      · unfold packing_source.packing.Packing.decode_block_fields_loop4.body
        rw [if_pos (by rw [Std.UScalar.lt_equiv]; exact hlt), if_neg hb1, hxa]
        simp only [bind_tc_ok]
        rw [hat1]
        simp only [bind_tc_ok]
        rw [if_neg hb]
        simp only [bind_tc_ok]
        change (do let coeffs1 ← coeffs.push (xa, 0#i32)
                   let g2 ← g + 1#usize
                   ok (ControlFlow.cont (b1, at1, coeffs1, g2)))
            = ok (ControlFlow.cont (b1, at1, coeffs2, g1))
        rw [hpush, hg1]
        simp only [bind_tc_ok]
      · show at1.val = at0.val + coeffStep b
        rw [hat1', coeffStep, if_neg hb]
      · show groups.val - g1.val < groups.val - g.val
        rw [hg1']
        omega

/-- The coefficient loop’s exit branch, taken once the group index has reached `groups`. -/
theorem field_exit (b b1 : Bool) (raw : Slice Std.U8) (groups : Std.Usize)
    (at0 : Std.Usize) (coeffs : alloc.vec.Vec (Std.I32 × Std.I32)) (g : Std.Usize)
    (hg : groups.val ≤ g.val) :
    packing_source.packing.Packing.decode_block_fields_loop4.body b raw groups b1 at0 coeffs g
      = ok (ControlFlow.done (b1, at0, coeffs)) := by
  unfold packing_source.packing.Packing.decode_block_fields_loop4.body
  have hnot : ¬ (g < groups) := by rw [Std.UScalar.lt_equiv]; exact not_lt_of_ge hg
  rw [if_neg hnot]

/-- The coefficient loop invariant: `g - g0` groups read, the strand advanced by that many
`coeffStep b` strides, one `(a, b)` pair appended per group. -/
def FieldInv (b b1 : Bool) (groups at0 : Std.Usize) (g0 : Std.Usize)
    (coeffs0 : alloc.vec.Vec (Std.I32 × Std.I32)) (bflag : Bool) (pos : Std.Usize)
    (coeffs : alloc.vec.Vec (Std.I32 × Std.I32)) (g : Std.Usize) : Prop :=
  bflag = b1 ∧ g0.val ≤ g.val ∧ g.val ≤ groups.val ∧
  pos.val = at0.val + (g.val - g0.val) * coeffStep b ∧
  coeffs.val.length = coeffs0.val.length + (g.val - g0.val)

/-- `n·s + s = (n+1)·s`: one more strand stride. -/
lemma step_succ (n s : Nat) : n * s + s = (n + 1) * s := Nat.succ_mul n s |>.symm

/-- **Closed-form per-field work.** Entered at group index `g0`, the extracted coefficient loop
exits at group index `groups` with the strand advanced by `(groups - g0) · coeffStep has_min` bytes
and exactly `groups - g0` `(a, b)` pairs appended. That difference is the continuation count: every
continuation advances the group index by exactly one and appends exactly one pair (`field_step`),
while the exit branch is taken only at `g = groups` (`field_exit`), so the loop runs once per affine
group of the block. The vector-capacity premise is `coeffs0.length + groups ≤ Usize.max`, the bound
the model's `Vec.push` needs; the payload premise `hA` is the strand position staying inside `raw`. -/
theorem field_work (b b1 : Bool) (raw : Slice Std.U8) (groups at0 : Std.Usize)
    (coeffs0 : alloc.vec.Vec (Std.I32 × Std.I32)) (g0 : Std.Usize)
    (hk0 : g0.val ≤ groups.val)
    (hA : ∀ n : Nat, n ≤ groups.val - g0.val →
      at0.val + n * coeffStep b < raw.val.length)
    (hcap : coeffs0.val.length + groups.val ≤ Std.Usize.max) :
    (∃ (pos' : Std.Usize) (coeffs' : alloc.vec.Vec (Std.I32 × Std.I32)),
      packing_source.packing.Packing.decode_block_fields_loop4 b b1 raw at0 coeffs0 groups g0
          = ok (b1, pos', coeffs') ∧
      pos'.val = at0.val + (groups.val - g0.val) * coeffStep b ∧
      coeffs'.val.length = coeffs0.val.length + (groups.val - g0.val)) := by
  have hloop : packing_source.packing.Packing.decode_block_fields_loop4 b b1 raw at0 coeffs0
    groups g0 ⦃ (r : Bool × Std.Usize × alloc.vec.Vec (Std.I32 × Std.I32)) =>
      ∃ (pos' : Std.Usize) (coeffs' : alloc.vec.Vec (Std.I32 × Std.I32)),
        r = (b1, pos', coeffs') ∧
        pos'.val = at0.val + (groups.val - g0.val) * coeffStep b ∧
        coeffs'.val.length = coeffs0.val.length + (groups.val - g0.val) ⦄ := by
    unfold packing_source.packing.Packing.decode_block_fields_loop4
    apply loop.spec_decr_nat
      (measure := fun s : Bool × Std.Usize × (alloc.vec.Vec (Std.I32 × Std.I32)) × Std.Usize =>
        groups.val - s.2.2.2.val)
      (inv := fun s => FieldInv b b1 groups at0 g0 coeffs0 s.1 s.2.1 s.2.2.1 s.2.2.2)
      (post := fun r : Bool × Std.Usize × alloc.vec.Vec (Std.I32 × Std.I32) =>
        ∃ (pos' : Std.Usize) (coeffs' : alloc.vec.Vec (Std.I32 × Std.I32)),
          r = (b1, pos', coeffs') ∧
          pos'.val = at0.val + (groups.val - g0.val) * coeffStep b ∧
          coeffs'.val.length = coeffs0.val.length + (groups.val - g0.val))
    · intro x hx
      rcases x with ⟨b2, pos, coeffs, g⟩
      dsimp only at hx ⊢
      obtain ⟨hb2, hg0le, hgle, hat, hlen⟩ := hx
      have hmle : g.val - g0.val ≤ groups.val - g0.val := by omega
      by_cases hlt : g.val < groups.val
      · have hA' : pos.val < raw.val.length := by
          rw [hat]
          exact hA (g.val - g0.val) hmle
        have hA1 : b = true → pos.val + 1 < raw.val.length := by
          intro hb
          have h2 : coeffStep b = 2 := by simp [coeffStep, hb]
          have hstep := hA ((g.val - g0.val) + 1) (by omega)
          rw [h2] at hstep
          have hexp : ((g.val - g0.val) + 1) * 2 = (g.val - g0.val) * 2 + 2 := by omega
          rw [hexp] at hstep
          rw [hat, h2]
          omega
        have hcap' : coeffs.val.length < Std.Usize.max := by
          rw [hlen]
          calc coeffs0.val.length + (g.val - g0.val)
              < coeffs0.val.length + groups.val := by omega
            _ ≤ Std.Usize.max := hcap
        obtain ⟨a, m, atN, coeffsN, gN, hbody, hatN, hgN, hlenN, hdecr⟩ :=
          field_step b b2 raw groups pos coeffs g hlt hA' hA1 hcap'
        rw [hbody, WP.spec_ok]
        show FieldInv b b1 groups at0 g0 coeffs0 b2 atN coeffsN gN ∧
          groups.val - gN.val < groups.val - g.val
        unfold FieldInv
        refine ⟨⟨by omega, by omega, by omega, ?_, ?_⟩, ?_⟩
        · show atN.val = at0.val + (gN.val - g0.val) * coeffStep b
          rw [hatN, hgN, hat]
          have hsub : g.val + 1 - g0.val = (g.val - g0.val) + 1 := by omega
          rw [hsub, Nat.add_assoc, step_succ]
        · show coeffsN.val.length = coeffs0.val.length + (gN.val - g0.val)
          rw [hlenN, List.length_append, List.length_cons, List.length_nil, hlen, hgN]
          have hsub : g.val + 1 - g0.val = (g.val - g0.val) + 1 := by omega
          omega
        · exact hdecr
      · rw [field_exit b b2 raw groups pos coeffs g (by omega), WP.spec_ok]
        refine ⟨pos, coeffs, ?_, ?_, ?_⟩
        · simp only [hb2]
        · have hg : g.val = groups.val := by omega
          rw [hat, hg]
        · have hg : g.val = groups.val := by omega
          rw [hlen, hg]
    · refine ⟨rfl, le_rfl, hk0, ?_, ?_⟩
      · show at0.val = at0.val + (g0.val - g0.val) * coeffStep b
        simp only [Nat.sub_self, Nat.zero_mul, Nat.add_zero]
      · show coeffs0.val.length = coeffs0.val.length + (g0.val - g0.val)
        simp only [Nat.sub_self, Nat.add_zero]
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hloop
  obtain ⟨pos', coeffs', hpost', hpos', hlen'⟩ := hpost
  exact ⟨pos', coeffs', hr.trans (congrArg ok hpost'), hpos', hlen'⟩


/-! ## The per-lane code transitions

`packing_source.packing.Packing.decode_block_fields_loop7` is the extracted `while i < block` loop of
`Packing::decode_block_fields` (`src/format/packing.rs:713-716`): one iteration runs the bit-plane
reader `unpack_code` for lane `i` over the code area and appends the resulting code field. -/

/-- The descriptor the extracted code-unpack loop body carries: the body rebuilds the packing from
the individual fields it was given. -/
def codeDescriptor (cw : packing_source.packing.CodeWidth) (i : Std.U8) (i1 i2 : Std.U32)
    (sk : packing_source.packing.ScaleKind) (b b1 b2 : Bool) (i3 : Std.I32)
    (po : packing_source.packing.PlaneOrder) (no : packing_source.packing.NibbleOrder) :
    packing_source.packing.Packing :=
  { width := cw, high_planes := i, block := i1, group := i2, scale_kind := sk, has_min := b,
    signed_code := b1, bias := i3, signed_coeff := b2, plane_order := po, nibble_order := no }

/-- `unpack_code` on an unsigned-code descriptor returns the bit-plane loop’s accumulator: the
two’s-complement sign-extension branch is not taken. -/
theorem unpack_code_unsigned (self : packing_source.packing.Packing) (area : Slice Std.U8)
    (i total : Std.Usize) (htot : total.val ≤ 64)
    (hmul : i.val * total.val + total.val ≤ Std.Usize.max)
    (harea : ∀ j : Std.Usize, j.val < total.val →
      (i.val * total.val + j.val) / 8 < area.val.length)
    (hs : self.signed_code = false) :
    ∃ v', packing_source.packing.Packing.unpack_code self area i total = ok v' := by
  obtain ⟨⟨v', hloop⟩, hcount⟩ :=
    bit_plane_work area i total self 0#i64 0#usize (by simp) htot hmul harea
  have hs' : ¬ (self.signed_code = true) := by simp [hs]
  unfold packing_source.packing.Packing.unpack_code
  rw [hloop]
  simp only [bind_tc_ok]
  refine ⟨v', ?_⟩
  show (if self.signed_code = true then
      (if total < 64#usize then
          (do let i1 ← total - 1#usize
              let sign_bit ← 1#i64 <<< i1
              let i2 ← lift (v' &&& sign_bit)
              if (i2 != 0#i64) = true then
                (do let i3 ← 1#i64 <<< total
                    v' - i3)
              else ok v')
        else ok v')
    else ok v') = ok v'
  rw [if_neg hs']

/-- **One guarded per-lane transition.** Under the guard `i < block` and the code-area bound, the
extracted body performs exactly one continuation: it runs the bit-plane reader for lane `i`,
appends the code field it returns, advances the lane index by one, and the rank `block - i`
strictly decreases. -/
theorem lane_code_step (cw : packing_source.packing.CodeWidth) (hi : Std.U8) (hb gr : Std.U32)
    (sk : packing_source.packing.ScaleKind) (b b1 : Bool) (bias : Std.I32) (b2 : Bool)
    (po : packing_source.packing.PlaneOrder) (no : packing_source.packing.NibbleOrder)
    (total block : Std.Usize) (area : alloc.vec.Vec Std.U8) (codes : alloc.vec.Vec Std.I64)
    (i0 : Std.Usize)
    (hs : b1 = false) (hlt : i0.val < block.val) (htot : total.val ≤ 64)
    (hmul : i0.val * total.val + total.val ≤ Std.Usize.max)
    (harea : (i0.val * total.val + total.val - 1) / 8
      < (alloc.vec.Vec.deref area).val.length)
    (hcap : codes.val.length < Std.Usize.max) :
    ∃ (u : Std.I64) (codesN : alloc.vec.Vec Std.I64) (iN : Std.Usize),
      packing_source.packing.Packing.decode_block_fields_loop7.body cw hi hb gr sk b b1 bias b2 po
          no total block area codes i0 = ok (ControlFlow.cont (codesN, iN)) ∧
      iN.val = i0.val + 1 ∧ codesN.val = codes.val ++ [u] ∧
      block.val - iN.val < block.val - i0.val := by
  have hmulN : i0.val * total.val + total.val ≤ Std.Usize.max := hmul
  have hareaN : ∀ j : Std.Usize, j.val < total.val →
      (i0.val * total.val + j.val) / 8 < (alloc.vec.Vec.deref area).val.length := by
    intro j hj
    have hle : i0.val * total.val + j.val ≤ i0.val * total.val + total.val - 1 := by omega
    exact lt_of_le_of_lt (Nat.div_le_div_right hle) harea
  obtain ⟨u, hunpack⟩ := unpack_code_unsigned
    ({ width := cw, high_planes := hi, block := hb, group := gr, scale_kind := sk, has_min := b,
       signed_code := b1, bias := bias, signed_coeff := b2, plane_order := po,
       nibble_order := no } : packing_source.packing.Packing)
    (alloc.vec.Vec.deref area) i0 total htot hmulN hareaN hs
  obtain ⟨codes2, hpush, hpushv⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec codes u hcap)
  obtain ⟨i1, hadd, hi1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i0) (y := 1#usize) (by scalar_tac))
  have hi1' : i1.val = i0.val + 1 := by rw [hi1v]; rfl
  have hbody : packing_source.packing.Packing.decode_block_fields_loop7.body cw hi hb gr sk b b1
      bias b2 po no total block area codes i0
        = ok (ControlFlow.cont (codes2, i1)) := by
    have hdo : (do
          let i5 ← packing_source.packing.Packing.unpack_code
            { width := cw, high_planes := hi, block := hb, group := gr, scale_kind := sk,
              has_min := b, signed_code := b1, bias := bias, signed_coeff := b2, plane_order := po,
              nibble_order := no } (alloc.vec.Vec.deref area) i0 total
          let codes1 ← codes.push i5
          let i6 ← i0 + 1#usize
          ok (ControlFlow.cont (codes1, i6) :
            ControlFlow (alloc.vec.Vec Std.I64 × Std.Usize) (alloc.vec.Vec Std.I64)))
        = ok (ControlFlow.cont (codes2, i1)) := by
      simp only [hunpack, bind_tc_ok, hpush, hadd]
    unfold packing_source.packing.Packing.decode_block_fields_loop7.body
    rw [if_pos (by rw [Std.UScalar.lt_equiv]; exact hlt)]
    exact hdo
  refine ⟨u, codes2, i1, hbody, hi1', hpushv, ?_⟩
  rw [hi1']
  omega


/-- The per-lane loop invariant: `i - i0` code fields appended, one per lane. -/
def LaneInv (block i0 : Std.Usize) (codes0 : alloc.vec.Vec Std.I64)
    (codes : alloc.vec.Vec Std.I64) (i : Std.Usize) : Prop :=
  i0.val ≤ i.val ∧ i.val ≤ block.val ∧
  codes.val.length = codes0.val.length + (i.val - i0.val)

/-- **Closed-form per-lane work.** Entered at lane index `i0`, the extracted code loop performs
exactly `block - i0` continuations — one per remaining lane of the block — appends that many code
fields and exits with the lane index at `block`. The per-lane work itself is the bit-plane count of
`bit_plane_work`. -/
theorem lane_code_work (cw : packing_source.packing.CodeWidth) (hi : Std.U8) (hb gr : Std.U32)
    (sk : packing_source.packing.ScaleKind) (b b1 : Bool) (bias : Std.I32) (b2 : Bool)
    (po : packing_source.packing.PlaneOrder) (no : packing_source.packing.NibbleOrder)
    (total block : Std.Usize) (area : alloc.vec.Vec Std.U8) (codes0 : alloc.vec.Vec Std.I64)
    (i0 : Std.Usize)
    (hs : b1 = false) (hi0 : i0.val ≤ block.val) (htot : total.val ≤ 64)
    (hmul : ∀ k : Nat, k ≤ block.val → k * total.val + total.val ≤ Std.Usize.max)
    (harea : ∀ k : Nat, k < block.val →
      (k * total.val + total.val - 1) / 8 < (alloc.vec.Vec.deref area).val.length)
    (hcap : codes0.val.length + block.val ≤ Std.Usize.max) :
    ∃ (codes' : alloc.vec.Vec Std.I64),
      packing_source.packing.Packing.decode_block_fields_loop7 cw hi hb gr sk b b1 bias b2 po no
          total block area codes0 i0 = ok codes' ∧
      codes'.val.length = codes0.val.length + (block.val - i0.val) := by
  have hloop : packing_source.packing.Packing.decode_block_fields_loop7 cw hi hb gr sk b b1 bias
      b2 po no total block area codes0 i0
        ⦃ (r : alloc.vec.Vec Std.I64) =>
          ∃ codes' : alloc.vec.Vec Std.I64, r = codes' ∧
            codes'.val.length = codes0.val.length + (block.val - i0.val) ⦄ := by
    unfold packing_source.packing.Packing.decode_block_fields_loop7
    apply loop.spec_decr_nat
      (measure := fun s : alloc.vec.Vec Std.I64 × Std.Usize => block.val - s.2.val)
      (inv := fun s => LaneInv block i0 codes0 s.1 s.2)
      (post := fun r : alloc.vec.Vec Std.I64 => ∃ codes' : alloc.vec.Vec Std.I64, r = codes' ∧
        codes'.val.length = codes0.val.length + (block.val - i0.val))
    · intro x hx
      rcases x with ⟨codes, i⟩
      dsimp only at hx ⊢
      obtain ⟨hi0le, hile, hlen⟩ := hx
      by_cases hlt : i.val < block.val
      · have harea' : (i.val * total.val + total.val - 1) / 8
            < (alloc.vec.Vec.deref area).val.length := harea i.val hlt
        have hcap' : codes.val.length < Std.Usize.max := by
          rw [hlen]
          calc codes0.val.length + (i.val - i0.val) < codes0.val.length + block.val := by omega
            _ ≤ Std.Usize.max := hcap
        obtain ⟨u, codesN, iN, hbody, hiN, hlenN, hdecr⟩ :=
          lane_code_step cw hi hb gr sk b b1 bias b2 po no total block area codes i hs hlt htot
            (hmul i.val hile) harea' hcap'
        rw [hbody, WP.spec_ok]
        show LaneInv block i0 codes0 codesN iN ∧ block.val - iN.val < block.val - i.val
        unfold LaneInv
        refine ⟨⟨by omega, by omega, ?_⟩, ?_⟩
        · rw [hlenN, List.length_append, List.length_cons, List.length_nil, hlen, hiN]
          omega
        · exact hdecr
      · have hieq : i.val = block.val := by omega
        have hdone : packing_source.packing.Packing.decode_block_fields_loop7.body cw hi hb gr sk
            b b1 bias b2 po no total block area codes i
            = ok (ControlFlow.done codes) := by
          unfold packing_source.packing.Packing.decode_block_fields_loop7.body
          rw [if_neg (by rw [Std.UScalar.lt_equiv]; exact hlt)]
        rw [hdone, WP.spec_ok]
        exact ⟨codes, rfl, by rw [hlen, hieq]⟩
    · exact ⟨le_rfl, hi0, by simp⟩
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hloop
  obtain ⟨codes', hpost', hlen'⟩ := hpost
  exact ⟨codes', hr.trans (congrArg ok hpost'), hlen'⟩

/-! ## Refusal branches

`Packing::decode_block_fields` refuses before any field is read when the payload is shorter than the
geometry requires, and it propagates a geometry refusal from `block_bytes` unchanged. Both are
transitions of the real extracted body. -/

/-- The malformed-length refusal: a payload shorter than `block_bytes` yields
`Truncated need (Slice.len raw)` and no field is read. -/
theorem fields_truncated (self : packing_source.packing.Packing) (raw : Slice Std.U8)
    (need : Std.Usize)
    (hneed : packing_source.packing.Packing.block_bytes self
      = ok (core.result.Result.Ok need))
    (hshort : (Slice.len raw).val < need.val) :
    packing_source.packing.Packing.decode_block_fields self raw
      = ok (core.result.Result.Err (packing_source.packing.PackingError.Truncated need
        (Slice.len raw))) := by
  unfold packing_source.packing.Packing.decode_block_fields
  rw [hneed]
  simp only [core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok, lift]
  rw [if_pos (by rw [Std.UScalar.lt_equiv]; exact hshort)]

/-- The geometry refusal: whatever `block_bytes` refuses with, `decode_block_fields` returns
unchanged and reads no field. This covers the `BadWidth`/`BadBlock`/`BadGeometry`/`Overflow`
refusals of the geometry computation. -/
theorem fields_geometry_refusal (self : packing_source.packing.Packing) (raw : Slice Std.U8)
    (e : packing_source.packing.PackingError)
    (hbad : packing_source.packing.Packing.block_bytes self
      = ok (core.result.Result.Err e)) :
    packing_source.packing.Packing.decode_block_fields self raw
      = ok (core.result.Result.Err e) := by
  unfold packing_source.packing.Packing.decode_block_fields
  rw [hbad]
  simp only [core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok, lift,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from, lift]

/-! ## Reachable work on a concrete face of the grammar

`Packing::terminal(CodeWidth::F16)` (`src/format/packing.rs:445-459`) is the `F16` face: one
element per block, identity scale, no min, no coefficient field, and a 16-bit code field per element.
The two values below are the fields the extracted `is_terminal`, `scale_field_bytes` and
`total_code_bits` return for it, evaluated on that descriptor. -/

/-- The `F16` terminal parameter set: one element per block, identity scale, no min, no coefficient
field, `LsbFirst`/`LowHigh` order (`src/format/packing.rs:445-459`). -/
def f16Terminal : packing_source.packing.Packing :=
  { width := packing_source.packing.CodeWidth.F16, high_planes := 0#u8, block := 1#u32,
    group := 1#u32, scale_kind := packing_source.packing.ScaleKind.Identity, has_min := false,
    signed_code := false, bias := 0#i32, signed_coeff := false,
    plane_order := packing_source.packing.PlaneOrder.LsbFirst,
    nibble_order := packing_source.packing.NibbleOrder.LowHigh }

/-- The `F16` face’s own field geometry: terminal, no scale field, sixteen code bits per element. -/
theorem f16Terminal_geometry :
    packing_source.packing.Packing.is_terminal f16Terminal = ok true ∧
    packing_source.packing.Packing.scale_field_bytes f16Terminal = ok 0#usize ∧
    packing_source.packing.Packing.total_code_bits f16Terminal
      = ok (core.result.Result.Ok 16#usize) :=
  ⟨rfl, rfl, rfl⟩

/-- The two-byte code area of one `F16` element, little-endian `0x3c00`. -/
def f16Area : Slice Std.U8 := Slice.from [0x00#u8, 0x3c#u8] (by scalar_tac)

/-- **A reachable per-lane success on the `F16` face.** The extracted bit-plane reader, entered at
plane index `0` of a two-byte lane area, performs exactly 16 continuations — one per code bit of the
face’s 16-bit code field — and exits with the descriptor unchanged. The count is the closed form of
`bit_plane_work` at `total = 16`, `k0 = 0`. -/
theorem f16_lane_work :
    packing_source.packing.Packing.is_terminal f16Terminal = ok true ∧
    packing_source.packing.Packing.total_code_bits f16Terminal
      = ok (core.result.Result.Ok 16#usize) ∧
    (∃ v', packing_source.packing.Packing.unpack_code_loop f16Terminal f16Area 0#usize 16#usize
        0#i64 0#usize = ok (f16Terminal, v')) ∧
    (∃ n, bitPlaneSteps f16Area 0#usize 16#usize f16Terminal 0#i64 0#usize n ∧ n = 16) ∧
    (∃ u, packing_source.packing.Packing.unpack_code f16Terminal f16Area 0#usize 16#usize = ok u) := by
  have h16 : (16 : Nat) ≤ Std.Usize.max := by scalar_tac
  have hgeo : packing_source.packing.Packing.total_code_bits f16Terminal
      = ok (core.result.Result.Ok 16#usize) := rfl
  have hlen : f16Area.val.length = 2 := rfl
  have hz : (0#usize).val = 0 := rfl
  have h16v : (16#usize).val = 16 := rfl
  have harea : ∀ j : Std.Usize, j.val < 16#usize.val →
      (0#usize.val * 16#usize.val + j.val) / 8 < f16Area.val.length := by
    intro j hj
    rw [hlen, hz, h16v]
    omega
  obtain ⟨⟨v', hloop⟩, ⟨n, hsteps, hn⟩⟩ :=
    bit_plane_work f16Area 0#usize 16#usize f16Terminal 0#i64 0#usize (by norm_num)
      (by norm_num) (by norm_num; exact h16) harea
  obtain ⟨u, hunpack⟩ := unpack_code_unsigned f16Terminal f16Area 0#usize 16#usize
    (by norm_num) (by norm_num; exact h16) harea (by rfl)
  refine ⟨rfl, hgeo, ⟨v', hloop⟩, ⟨n, hsteps, hn⟩, ⟨u, hunpack⟩⟩

/-- **A reachable per-field success.** Entered at group index `0` of a one-group block with a
two-byte payload, the extracted coefficient loop performs exactly one continuation, reads the
coefficient byte at the strand position, appends one `(a, b)` pair and exits with the strand at
byte `1`. The count is the closed form of `field_work` at `groups = 1`, `g0 = 0`. -/
theorem f16_field_work :
    ∃ (pos' : Std.Usize) (coeffs' : alloc.vec.Vec (Std.I32 × Std.I32)),
      packing_source.packing.Packing.decode_block_fields_loop4 false false
          (Slice.from [0x01#u8, 0x02#u8] (by scalar_tac)) 0#usize
          (alloc.vec.Vec.new (Std.I32 × Std.I32)) 1#usize 0#usize
          = ok (false, pos', coeffs') ∧
      pos'.val = 1 ∧ coeffs'.val.length = 1 := by
  obtain ⟨pos', coeffs', hloop, hpos', hlen'⟩ :=
    field_work false false (Slice.from [0x01#u8, 0x02#u8] (by scalar_tac)) 1#usize 0#usize
      (alloc.vec.Vec.new (Std.I32 × Std.I32)) 0#usize (by norm_num)
      (by
        intro n hn
        have h2 : (Slice.from [0x01#u8, 0x02#u8] (by scalar_tac : (2:Nat) ≤ Std.Usize.max)).val.length
            = 2 := rfl
        have hstep : coeffStep false = 1 := rfl
        have hz : (0#usize).val = 0 := rfl
        have h1v : (1#usize).val = 1 := rfl
        rw [h1v] at hn
        rw [hstep, h2, hz]
        omega)
      (by simp; scalar_tac)
  exact ⟨pos', coeffs', hloop, hpos', hlen'⟩

/-! ## The block-level code-plane figure

The two per-loop counts compose: the code plane of a block costs one bit-plane loop per lane, and
each of those loops performs one continuation per code bit of the lane. -/

/-- The code-plane work of a block with `block` lanes of `total` code bits each, entered at lane
`i0`: the product of the lane count of `lane_code_work` and the per-lane bit-plane count of
`bit_plane_work` (taken at plane index `0`). -/
def codePlaneWork (block total i0 : Std.Usize) : Nat := (block.val - i0.val) * total.val

/-- The composed figure is exactly the product of the two proved per-loop ranks. The two factors
are `proved` (`lane_code_work` for the lane count, `bit_plane_work` for the per-lane bit count);
this product statement is the definition unfolding, not an independent measurement. -/
theorem code_plane_work (block total i0 : Std.Usize) :
    codePlaneWork block total i0 = (block.val - i0.val) * total.val := rfl

end PackingResource

/-! ## Audit -/

open Lean Elab Command in
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit PackingResource.bit_plane_step
#resource_audit PackingResource.bit_plane_exit
#resource_audit PackingResource.bit_plane_cont
#resource_audit PackingResource.bit_plane_work
#resource_audit PackingResource.field_step
#resource_audit PackingResource.field_exit
#resource_audit PackingResource.field_work
#resource_audit PackingResource.unpack_code_unsigned
#resource_audit PackingResource.lane_code_step
#resource_audit PackingResource.lane_code_work
#resource_audit PackingResource.fields_truncated
#resource_audit PackingResource.fields_geometry_refusal
#resource_audit PackingResource.f16Terminal_geometry
#resource_audit PackingResource.f16_lane_work
#resource_audit PackingResource.f16_field_work
#resource_audit PackingResource.code_plane_work

run_cmd Lean.logInfo "PACKING_KERNEL_OK"
