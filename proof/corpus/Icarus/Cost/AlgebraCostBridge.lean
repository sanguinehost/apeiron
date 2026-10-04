import Icarus.Cost.CostAlgebra
import Icarus.Cost.AlgebraCostCompletePart1
import Icarus.Cost.AlgebraCostCompletePart2

/-
SCOPE:      work/proofs/AlgebraCostBridge.lean (new) and its receipt
            work/proofs/AlgebraCostBridge.md. The alignment AR-27's §11 leaves open: the row "a map to
            `CostAlgebra.Cost` — `ShaCost` is not connected to the registered resource record"
            (`AlgebraCostComplete.lean`, §11). One map `shaToCost` from the twenty-three-field
            `ShaCost σ` attempt measure into the seven-field corpus composition/dominance algebra
            `CostAlgebra.Cost`, with a named disposition for all twenty-three source fields
            (absorbed into a target field, or dropped with a stated reason), the bridge laws
            (zero, additivity, per-field and joint monotonicity, iteration, overlap), the AR-15
            round/chain/block vectors and the AR-20 attempt/range figures instantiated through the
            map, the face verdict at the target, and the corpus algebra's `path` field proved
            invisible to the measure. Complete when the module elaborates against the two compiled
            upstream modules with every headline audited inside {propext, Classical.choice,
            Quot.sound} and zero `sorryAx`.
REGISTER:   resource model / bookkeeping alignment. No production, pin, API, extraction-root,
            corpus or gate-driver edit; no Rust source; no cargo; no regen.sh; no measurement.
            Excluded by construction: latency, wall-clock time, cycle counts, throughput, hash rate,
            bytes per second, energy, joules, watts, hardware, ASIC or GPU figures, miner economics,
            profitability, USD, compiler or codegen cost, the kernel ABI, the `unsafe` container,
            device dispatch cost, constant-time and security properties of any hash, and every
            asymptotic statement. The bridge is an alignment of two formal records; it is not a
            hardware, time or price claim. AGENTS.md:429 - a cost-model proof is not a hardware
            speed measurement.
CORPUS:     `CostAlgebra` (repos/apeiron/proof/resource/CostAlgebra.lean, sha256
            c362ec455954e512c2f2055c04e320f26f70529700ea592aeac7eb7a1b422c95): `Cost` (:46), `zero`
            (:57), `costLe` (:61), `costAdd` (:80), `overlap` (:93), `Cost.le_of_fields` (:98),
            `Cost.eq_of_fields` (:103), `add_monotone` (:157), `add_monotone_left` (:163),
            `overlap_fields` (:184), `tie_not_improvement` (:222), all grade `proved` there.
            `AlgebraCostComplete` (work/proofs/AlgebraCostComplete.lean, sha256
            d6d5ae7f229828122e08dc89e08a57e420e28057e04d73a7f9ada0d907d9fae4): `ShaCost` (:379),
            `ShaCost.zero` (:429), `ShaCost.composed` (:439)/`composed_apply` (:454),
            `ShaCost.rep` (:498)/`rep_succ` (:515)/`rep_add` (:522), `ShaCost.ofRound` (:543),
            `roundCost` (:782)/`roundCost_fields` (:788), `roundChainCost` (:843)/`roundChain64`
            (:867), `blockCost` (:886)/`blockCost_eq` (:896), `attemptCost` (:1576)/`attemptCost_eq`
            (:1602), `rangeCost` (:1662)/`rangeCost_eq` (:1669), `nonceSpace` (:1751),
            `attemptsCost_add` (:1735), `attemptCost_faceElliptic/…Parabolic/…Hyperbolic`
            (:1897/:1905/:1913), `attemptCost_threeFaces` (:1924), grade `proved` there modulo its
            named `model` premises (README: `RotRead`, `AddRead`, the occurrence inventories); the
            bridge reads only kernel-checked definitions and theorems of that module.
REFERENCE:  repos/apeiron/proof/resource/CostBridge.lean (the corpus bridge pattern: `Perf`
            (:58), `toCost` (:102), `toCost_zero` (:106), per-field monotone lemmas (:110-146),
            `loads_invisible_to_order` (:161), `toCost_add` (:170), `runCost_append` (:279)) and
            `CostAlgebraGate.py` (private-package build pattern for the outside-corpus modules).
            repos/apeiron/AGENTS.md:336-341 (§8), :156-167 (§4), :226-240 (§5), :429, :513, :516.
GATE:       three commands under /home/lucas/Workspace/icarus/bin/lane from cwd
            /home/lucas/Workspace/icarus/repos/apeiron/proof/corpus, in order:
            (1) lane lake env lean -R repos/apeiron/proof/resource -o
                /tmp/opencode/alg-cost-bridge/oleans/CostAlgebra.olean
                repos/apeiron/proof/resource/CostAlgebra.lean
            (2) lane lake env lean -R work/proofs -o
                /tmp/opencode/alg-cost-bridge/oleans/AlgebraCostComplete.olean
                work/proofs/AlgebraCostComplete.lean
            (3) lane env LEAN_PATH=/tmp/opencode/alg-cost-bridge/oleans lake env lean
                work/proofs/AlgebraCostBridge.lean
            Wall-clock budget 450 s for the whole invocation; observed 3.5 s + 24.3 s + compile of
            this file.  The oleans are compiled from the verbatim canonical sources; the gate
            records their sha256 before and after.  Mutation control on a throwaway copy of this
            file: a dropped or wrongly mapped field must turn command (3) red at a named headline.
GRADE:      `proved` for every law, closed form, instantiation and invisibility statement below,
            kernel-checked against the two compiled upstream modules, cone
            {propext, Classical.choice, Quot.sound} and zero `sorryAx`, rows at the end of this
            file.  `model` for the map itself: which of the twenty-three fields is absorbed where is
            a representation choice (a reading of the measure onto the registered vocabulary), not a
            hardware or time claim.  Unmeasured and unclaimed: everything on the REGISTER exclusion
            list, in particular no wall-clock, cycle, joule, watt, hash-per-second, USD or
            profitability figure follows from any number here.
-/

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.unusedVariables false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false

universe u

namespace algebra_cost_bridge

open algebra_cost_complete

variable {R : Type u} [CommRing R]

/-! ## 1. The map into the registered resource record -/

/-- **The map from the twenty-three-field attempt measure to the seven-field registered record.**
Field disposition, all twenty-three named, none merged silently:

| `ShaCost σ` field | disposition |
|---|---|
| `mul` `add` `sub` `neg` `cast` `channel` | `ops`, one operation each |
| `bitXor` `bitAnd` `bitNot` `shift` `bitOr` | `ops`, one `Word` operation each |
| `padShift` `padMod` | `ops`, one `Nat` operation each |
| `regroupMul` `regroupAdd` | `ops`, one `Nat` operation each |
| `regroupTrunc` | `ops`, one truncation each |
| `padByte` | `pack`, one byte each (message copy and padding appended) |
| `digestByte` | `pack`, one byte each (digest store) |
| `zeroByte` `lenByte` | not summed: subdivisions of `padByte`; summing them would double-count |
| `blockByteRead` | no target field: host byte traffic; `xfer` is device traffic and would be a false claim |
| `schedWordWrite` | no target field: word writes are not byte moves and not arithmetic |
| `digestWordRead` | no target field: word reads are not byte moves and not arithmetic |
| — | `alloc`, `xfer`, `dispatch`, `peak`, `path` are `0`: the measure counts no allocation, no device transfer, no kernel dispatch, no peak-live storage and no dependent step chain |

`ops` is the only operation-counting field of the target vocabulary, so it is the sum of the
sixteen operation-counted source fields; the merge is lossy across operation kinds (a `Word`-op
regression offset by a ring-op reduction is invisible inside `ops`) and that loss is this bridge's
stated boundary.  `path = 0` is not an omission but a theorem (`shaToCost_path`): the measure
carries no dependent-step count, so the target's scheduling distinction is invisible on the image. -/
def shaToCost (σ : R) (c : ShaCost σ) : CostAlgebra.Cost where
  ops := c.mul + c.add + c.sub + c.neg + c.cast + c.channel + c.bitXor + c.bitAnd + c.bitNot
    + c.shift + c.bitOr + c.padShift + c.padMod + c.regroupMul + c.regroupAdd + c.regroupTrunc
  pack := c.padByte + c.digestByte
  alloc := 0
  xfer := 0
  dispatch := 0
  peak := 0
  path := 0

/-! ## 2. The bridge laws -/

/-- **Zero is preserved.** Doing nothing maps to the registered zero record. -/
theorem shaToCost_zero (σ : R) : shaToCost σ (ShaCost.zero σ) = CostAlgebra.zero := rfl

/-- **The map preserves dependent composition.** Two primitives run after one another perform both
bodies, so both measures add fieldwise and the map commutes with the two `+` operations. -/
theorem shaToCost_add (σ : R) (a b : ShaCost σ) :
    shaToCost σ (a + b) = shaToCost σ a + shaToCost σ b := by
  refine CostAlgebra.Cost.eq_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost, ShaCost.composed_apply, ShaCost.composed, CostAlgebra.add_apply,
      CostAlgebra.costAdd] <;>
    omega

/-- **The map is monotone in the ring multiplication count.** Every other count is held fixed, so
this is the registered order's statement about one source field alone. -/
theorem shaToCost_monotone_mul (σ : R) (c : ShaCost σ) (k : Nat) (h : c.mul ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with mul := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the ring addition count.** -/
theorem shaToCost_monotone_add (σ : R) (c : ShaCost σ) (k : Nat) (h : c.add ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with add := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the ring subtraction count.** -/
theorem shaToCost_monotone_sub (σ : R) (c : ShaCost σ) (k : Nat) (h : c.sub ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with sub := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the ring negation count.** -/
theorem shaToCost_monotone_neg (σ : R) (c : ShaCost σ) (k : Nat) (h : c.neg ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with neg := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the dial-cast count.** -/
theorem shaToCost_monotone_cast (σ : R) (c : ShaCost σ) (k : Nat) (h : c.cast ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with cast := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the channel-read count.** -/
theorem shaToCost_monotone_channel (σ : R) (c : ShaCost σ) (k : Nat) (h : c.channel ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with channel := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the `Word` XOR count.** -/
theorem shaToCost_monotone_bitXor (σ : R) (c : ShaCost σ) (k : Nat) (h : c.bitXor ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with bitXor := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the `Word` AND count.** -/
theorem shaToCost_monotone_bitAnd (σ : R) (c : ShaCost σ) (k : Nat) (h : c.bitAnd ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with bitAnd := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the `Word` NOT count.** -/
theorem shaToCost_monotone_bitNot (σ : R) (c : ShaCost σ) (k : Nat) (h : c.bitNot ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with bitNot := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the `Word` shift count.** -/
theorem shaToCost_monotone_shift (σ : R) (c : ShaCost σ) (k : Nat) (h : c.shift ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with shift := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the `Word` OR count.** -/
theorem shaToCost_monotone_bitOr (σ : R) (c : ShaCost σ) (k : Nat) (h : c.bitOr ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with bitOr := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the length-byte shift count.** -/
theorem shaToCost_monotone_padShift (σ : R) (c : ShaCost σ) (k : Nat) (h : c.padShift ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with padShift := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the length-byte modulo count.** -/
theorem shaToCost_monotone_padMod (σ : R) (c : ShaCost σ) (k : Nat) (h : c.padMod ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with padMod := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the regrouping multiplication count.** -/
theorem shaToCost_monotone_regroupMul (σ : R) (c : ShaCost σ) (k : Nat)
    (h : c.regroupMul ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with regroupMul := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the regrouping addition count.** -/
theorem shaToCost_monotone_regroupAdd (σ : R) (c : ShaCost σ) (k : Nat)
    (h : c.regroupAdd ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with regroupAdd := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the regrouping truncation count.** -/
theorem shaToCost_monotone_regroupTrunc (σ : R) (c : ShaCost σ) (k : Nat)
    (h : c.regroupTrunc ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with regroupTrunc := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the padding byte count.** -/
theorem shaToCost_monotone_padByte (σ : R) (c : ShaCost σ) (k : Nat) (h : c.padByte ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with padByte := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Monotone in the digest byte count.** -/
theorem shaToCost_monotone_digestByte (σ : R) (c : ShaCost σ) (k : Nat)
    (h : c.digestByte ≤ k) :
    shaToCost σ c ≤ shaToCost σ { c with digestByte := k } := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- **Joint monotonicity: no mapped count grows implies no registered resource grows.** `ShaCost`
declares no `LE`/`≤` instance (searched in `AlgebraCostComplete.lean`: no `instance : LE`, no
`costLe`), so the premise is the explicit eighteen-hypothesis componentwise comparison over the
mapped fields; the conclusion is the registered componentwise order. -/
theorem shaToCost_monotone (σ : R) {a b : ShaCost σ}
    (hmul : a.mul ≤ b.mul) (hadd : a.add ≤ b.add) (hsub : a.sub ≤ b.sub) (hneg : a.neg ≤ b.neg)
    (hcast : a.cast ≤ b.cast) (hchannel : a.channel ≤ b.channel) (hbitXor : a.bitXor ≤ b.bitXor)
    (hbitAnd : a.bitAnd ≤ b.bitAnd) (hbitNot : a.bitNot ≤ b.bitNot) (hshift : a.shift ≤ b.shift)
    (hbitOr : a.bitOr ≤ b.bitOr) (hpadShift : a.padShift ≤ b.padShift)
    (hpadMod : a.padMod ≤ b.padMod) (hregroupMul : a.regroupMul ≤ b.regroupMul)
    (hregroupAdd : a.regroupAdd ≤ b.regroupAdd) (hregroupTrunc : a.regroupTrunc ≤ b.regroupTrunc)
    (hpadByte : a.padByte ≤ b.padByte) (hdigestByte : a.digestByte ≤ b.digestByte) :
    shaToCost σ a ≤ shaToCost σ b := by
  refine CostAlgebra.Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;>
    simp only [shaToCost] <;> omega

/-- The `n`-fold dependent sum in the target record, mirroring `ShaCost.rep`. Temporal O(n)
recursive steps; spatial O(1) records. -/
def costRep (n : Nat) (c : CostAlgebra.Cost) : CostAlgebra.Cost :=
  match n with
  | 0 => CostAlgebra.zero
  | n + 1 => c + costRep n c

/-- **The map preserves iteration**: `n` copies of a source cost are `n` copies of its image, so the
map commutes with `ShaCost.rep` and the bridge-local `costRep`. -/
theorem shaToCost_rep (σ : R) (n : Nat) (c : ShaCost σ) :
    shaToCost σ (ShaCost.rep σ n c) = costRep n (shaToCost σ c) := by
  induction n with
  | zero => simp [ShaCost.rep, shaToCost, costRep, CostAlgebra.zero]
  | succ n ih => rw [ShaCost.rep_succ, shaToCost_add, ih]; rfl

/-- **The registered critical path is invisible to the measure.** The measure counts no dependent
step chain, so every image lies in the `path = 0` sub-lattice; "one attempt is a chain of stages" is
not a statement of `ShaCost`. -/
theorem shaToCost_path (σ : R) (c : ShaCost σ) : (shaToCost σ c).path = 0 := rfl

/-- **On the image, concurrent composition equals dependent composition.** With `path = 0` on every
image, `CostAlgebra.overlap` and `CostAlgebra.costAdd` agree: the target's scheduling distinction
(sequence versus concurrent branches) cannot be expressed by any pair of `ShaCost` records. -/
theorem shaToCost_overlap (σ : R) (a b : ShaCost σ) :
    CostAlgebra.overlap (shaToCost σ a) (shaToCost σ b)
      = shaToCost σ a + shaToCost σ b := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ :=
    CostAlgebra.overlap_fields (shaToCost σ a) (shaToCost σ b)
  refine CostAlgebra.Cost.eq_of_fields h1 h2 h3 h4 h5 h6 ?_
  simp [CostAlgebra.overlap, CostAlgebra.costAdd, CostAlgebra.add_apply, shaToCost]

/-- **The five unmapped source fields do not reach the registered record.** Changing `zeroByte`,
`lenByte`, `blockByteRead`, `schedWordWrite` and `digestWordRead` leaves the image definitionally
equal, so the two records tie in the registered order: the bridge reports neither improvement nor
regression from those five counted quantities. This is a boundary of the target vocabulary, the
same one `CostBridge.loads_invisible_to_order` records for `Perf.loads`/`Perf.stores`. -/
theorem shaToCost_unmapped_invisible (σ : R) (c : ShaCost σ) (z l r w d : Nat) :
    shaToCost σ { c with zeroByte := z, lenByte := l, blockByteRead := r, schedWordWrite := w, digestWordRead := d } = shaToCost σ c ∧
      ¬ CostAlgebra.strictlyCheaper (shaToCost σ c)
        (shaToCost σ { c with zeroByte := z, lenByte := l, blockByteRead := r, schedWordWrite := w, digestWordRead := d }) := by
  constructor
  · rfl
  · exact CostAlgebra.tie_not_improvement _

/-! ## 3. AR-15's round, chain and block vectors through the map -/

/-- **One SHA-256 round through the map.** AR-15's round vector
`(36, 25, 0, 0, 0, 6, 8, 5, 1, 12, 6)` maps to 99 operations and no bytes: `ops` is the sum of the
eleven fields, the six ring counts and the five `Word` counts, and the round carries no byte. -/
theorem shaToCost_roundCost (σ : R) :
    shaToCost σ (ShaCost.ofRound (roundCost σ))
      = { ops := 99, pack := 0, alloc := 0, xfer := 0, dispatch := 0, peak := 0, path := 0 } := by
  rw [roundCost_fields]
  rfl

/-- **The 64-round chain through the map**: `64 * 99 = 6336` operations. -/
theorem shaToCost_roundChain64 (σ : R) :
    shaToCost σ (ShaCost.ofRound (roundChainCost σ 64))
      = { ops := 6336, pack := 0, alloc := 0, xfer := 0, dispatch := 0, peak := 0, path := 0 } := by
  rw [roundChain64]
  rfl

/-- **One block transformation through the map**: 64 rounds, 48 schedule steps and the add-back are
`9320` operations in the image. -/
theorem shaToCost_blockCost (σ : R) :
    shaToCost σ (ShaCost.ofRound (blockCost σ))
      = { ops := 9320, pack := 0, alloc := 0, xfer := 0, dispatch := 0, peak := 0, path := 0 } := by
  rw [blockCost_eq]
  rfl

/-! ## 4. AR-20's attempt and range figures through the map -/

/-- **One completed SHA-256 attempt through the map.** The completed attempt's twenty-three fields
(`attemptCost_eq`) map to `18880` operations and `160` bytes: the AR-20 compression counts
(`18640` operations) plus the padding's `16` `Nat` operations and the load's `224` operations, and
the `128` padding bytes plus the `32` digest bytes. The five unmapped fields (`zeroByte` 39,
`lenByte` 8, `blockByteRead` 128, `schedWordWrite` 32, `digestWordRead` 8) are absorbed into no
target field and are not counted in the image. -/
theorem shaToCost_attemptCost (σ : R) :
    shaToCost σ (attemptCost σ)
      = { ops := 18880, pack := 160, alloc := 0, xfer := 0, dispatch := 0, peak := 0, path := 0 } := by
  rw [attemptCost_eq]
  rfl

/-- **A range of `m` attempts against one header through the map.** The padding's `Nat` operations
are paid once (`16`), every other operation and byte is `m` times the per-attempt figure
(`18864` operations and `32` bytes per attempt): `ops = 16 + m * 18864`, `pack = 128 + m * 32`. -/
theorem shaToCost_rangeCost (σ : R) (m : Nat) :
    shaToCost σ (rangeCost σ m)
      = { ops := 16 + m * 18864, pack := 128 + m * 32, alloc := 0, xfer := 0, dispatch := 0,
          peak := 0, path := 0 } := by
  rw [rangeCost_eq]
  refine CostAlgebra.Cost.eq_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_
  all_goals simp [shaToCost]
  all_goals omega

/-- **The full 32-bit nonce space through the map** (`nonceSpace = 2^32 = 4294967296` attempts
against one header): `ops = 16 + 4294967296 * 18864`, `pack = 128 + 4294967296 * 32`. A count, not
a time. -/
theorem shaToCost_rangeCost_nonceSpace (σ : R) :
    shaToCost σ (rangeCost σ nonceSpace)
      = { ops := 16 + 4294967296 * 18864, pack := 128 + 4294967296 * 32, alloc := 0, xfer := 0,
          dispatch := 0, peak := 0, path := 0 } := by
  simpa [nonceSpace] using shaToCost_rangeCost σ nonceSpace

/-- **A split range is the sum of its parts in the image too.** `m + n` composite attempts map to
the registered sum of the two parts, by the additivity of both measures. -/
theorem shaToCost_attemptsCost_split (σ : R) :
    shaToCost σ (attemptsCost σ (2 ^ 31 + 2 ^ 31))
      = shaToCost σ (attemptsCost σ (2 ^ 31)) + shaToCost σ (attemptsCost σ (2 ^ 31)) := by
  simp only [attemptsCost_add, shaToCost_add]

/-! ## 5. The face verdict at the target -/

/-- **The face dial does not enter the registered record.** For any two dial values, the completed
attempt's image is the same `CostAlgebra.Cost` record: `Cost` has no field naming the face, so the
three-face verdict of `attemptCost_faceElliptic`/`…Parabolic`/`…Hyperbolic` maps to a single target
record. The refusal (the count does not depend on the face) is not a `Cost` field and cannot be
stated as one; it is the equality of the images. -/
theorem shaToCost_attemptCost_faceBlind (σ τ : R) :
    shaToCost σ (attemptCost σ) = shaToCost τ (attemptCost τ) := by
  simp only [shaToCost_attemptCost]

/-- **The three faces at the target**: elliptic, parabolic and hyperbolic dials give the same image,
pairwise. -/
theorem shaToCost_threeFaces :
    shaToCost faceElliptic (attemptCost faceElliptic)
        = shaToCost faceParabolic (attemptCost faceParabolic) ∧
      shaToCost faceParabolic (attemptCost faceParabolic)
        = shaToCost faceHyperbolic (attemptCost faceHyperbolic) ∧
      shaToCost faceElliptic (attemptCost faceElliptic)
        = shaToCost faceHyperbolic (attemptCost faceHyperbolic) :=
  ⟨shaToCost_attemptCost_faceBlind _ _, shaToCost_attemptCost_faceBlind _ _,
    shaToCost_attemptCost_faceBlind _ _⟩

/-! ## 6. Audit rows

`#print axioms` for every headline, and the rejecting `#resource_audit` of the imported
`CostAlgebra` module for the load-bearing laws. The closed cone required is
{propext, Classical.choice, Quot.sound} with no `sorryAx`. -/

#print axioms shaToCost
#print axioms shaToCost_zero
#print axioms shaToCost_add
#print axioms shaToCost_monotone_mul
#print axioms shaToCost_monotone_add
#print axioms shaToCost_monotone_sub
#print axioms shaToCost_monotone_neg
#print axioms shaToCost_monotone_cast
#print axioms shaToCost_monotone_channel
#print axioms shaToCost_monotone_bitXor
#print axioms shaToCost_monotone_bitAnd
#print axioms shaToCost_monotone_bitNot
#print axioms shaToCost_monotone_shift
#print axioms shaToCost_monotone_bitOr
#print axioms shaToCost_monotone_padShift
#print axioms shaToCost_monotone_padMod
#print axioms shaToCost_monotone_regroupMul
#print axioms shaToCost_monotone_regroupAdd
#print axioms shaToCost_monotone_regroupTrunc
#print axioms shaToCost_monotone_padByte
#print axioms shaToCost_monotone_digestByte
#print axioms shaToCost_monotone
#print axioms costRep
#print axioms shaToCost_rep
#print axioms shaToCost_path
#print axioms shaToCost_overlap
#print axioms shaToCost_unmapped_invisible
#print axioms shaToCost_roundCost
#print axioms shaToCost_roundChain64
#print axioms shaToCost_blockCost
#print axioms shaToCost_attemptCost
#print axioms shaToCost_rangeCost
#print axioms shaToCost_rangeCost_nonceSpace
#print axioms shaToCost_attemptsCost_split
#print axioms shaToCost_attemptCost_faceBlind
#print axioms shaToCost_threeFaces

#resource_audit algebra_cost_bridge.shaToCost_zero
#resource_audit algebra_cost_bridge.shaToCost_add
#resource_audit algebra_cost_bridge.shaToCost_monotone
#resource_audit algebra_cost_bridge.shaToCost_rep
#resource_audit algebra_cost_bridge.shaToCost_path
#resource_audit algebra_cost_bridge.shaToCost_overlap
#resource_audit algebra_cost_bridge.shaToCost_unmapped_invisible
#resource_audit algebra_cost_bridge.shaToCost_roundCost
#resource_audit algebra_cost_bridge.shaToCost_roundChain64
#resource_audit algebra_cost_bridge.shaToCost_blockCost
#resource_audit algebra_cost_bridge.shaToCost_attemptCost
#resource_audit algebra_cost_bridge.shaToCost_rangeCost
#resource_audit algebra_cost_bridge.shaToCost_rangeCost_nonceSpace
#resource_audit algebra_cost_bridge.shaToCost_attemptsCost_split
#resource_audit algebra_cost_bridge.shaToCost_attemptCost_faceBlind
#resource_audit algebra_cost_bridge.shaToCost_threeFaces

run_cmd Lean.logInfo "ALG_COST_BRIDGE_KERNEL_OK"

end algebra_cost_bridge
