/-
FIPS-202 Keccak-f[1600] and the SHA-3/SHAKE sponge: axiom-cone audit of the exported headlines.

`#audit` collects each constant's transitive axiom dependencies with `Lean.collectAxioms` — the same
mechanism `#print axioms` uses — and calls `throwError` on anything outside
`{propext, Classical.choice, Quot.sound}`, so a `sorryAx` here is a build failure rather than a line
in a log nobody reads. The `#guard` known-answer checks in `Crypto.Sponge` run the untrusted
evaluator and are not proofs; the audit below gates the constants themselves.
-/

import Crypto.Keccak.Constants
import Crypto.Keccak.Round
import Crypto.Keccak.Certificate
import Crypto.Keccak.SpongeCertificate
import Crypto.Sponge
import Lean

set_option autoImplicit false
set_option warningAsError true

open Lean Elab Command in
/-- The three axioms every proof in this bridge may depend on. -/
private def auditClosedAxioms : Lean.NameSet :=
  ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound

open Lean Elab Command in
/-- **The gate.** `#audit foo` collects `foo`'s transitive axiom dependencies with
    `Lean.collectAxioms` and calls `throwError` if any of them falls outside `auditClosedAxioms`.
    Prints the cone on success, so the log carries the same information `#print axioms` did. -/
elab "#audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !auditClosedAxioms.contains a)
    if bad.isEmpty then
      logInfo m!"'{c}' depends on axioms: {axioms.toList}"
    else
      throwError m!"'{c}' depends on axioms outside the closed cone " ++
        m!"(propext, Classical.choice, Quot.sound): {bad.toList}"

-- Constants: round constants and rho offsets (FIPS-202 §3.2)
#audit Apeiron.Crypto.Keccak.RC
#audit Apeiron.Crypto.Keccak.ROT

-- Round: the step functions and the 24-round permutation (FIPS-202 §3.2)
#audit Apeiron.Crypto.Keccak.theta
#audit Apeiron.Crypto.Keccak.rho
#audit Apeiron.Crypto.Keccak.pi
#audit Apeiron.Crypto.Keccak.chi
#audit Apeiron.Crypto.Keccak.iota
#audit Apeiron.Crypto.Keccak.round
#audit Apeiron.Crypto.Keccak.permute
#audit Apeiron.Crypto.Keccak.permute_eq_foldl_round

-- Sponge: parameters, byte order, padding, absorb/squeeze and the named functions (FIPS-202 §3-6)
#audit Apeiron.Crypto.Keccak.rate_sha3_256
#audit Apeiron.Crypto.Keccak.rate_sha3_512
#audit Apeiron.Crypto.Keccak.rate_shake128
#audit Apeiron.Crypto.Keccak.rate_shake256
#audit Apeiron.Crypto.Keccak.domain_sha3
#audit Apeiron.Crypto.Keccak.domain_shake
#audit Apeiron.Crypto.Keccak.byteOf
#audit Apeiron.Crypto.Keccak.xorBlock
#audit Apeiron.Crypto.Keccak.pad
#audit Apeiron.Crypto.Keccak.blocks
#audit Apeiron.Crypto.Keccak.absorb
#audit Apeiron.Crypto.Keccak.squeeze
#audit Apeiron.Crypto.Keccak.sponge
#audit Apeiron.Crypto.Keccak.sha3_256
#audit Apeiron.Crypto.Keccak.sha3_512
#audit Apeiron.Crypto.Keccak.shake128
#audit Apeiron.Crypto.Keccak.shake256

-- Refinement: the extracted `word_keccakf` computes `permute` on `view` (bridge corpus)
#audit apeiron_word.RoundStepDenotes
#audit apeiron_word.WordKeccakfDenotes

/-! ## Sponge certificate (`Crypto.Keccak.SpongeCertificate`, `namespace apeiron_word`)

The absorb loop closure, the byte/state views it reads through, and the parameter bridges to
`Crypto.Keccak`. Pure `rfl`/`decide` helpers (`laneBytes_length`, `blockBytes_length`, the numeric
`rate_lanes_*`/`rate_mod_8_*`/`rate_le_200_*`/`rate_bits_*`/`outLen_*` pins) are not registered. -/

-- Absorb loop: the invariant, one iteration, the loop's specification and the block-loop rewrite
#audit apeiron_word.absorbInv_post
#audit apeiron_word.absorb_step
#audit apeiron_word.absorb_body_step
#audit apeiron_word.absorb_loop_spec
#audit apeiron_word.absorb_block_loop_eq

-- Absorb lane: the extracted lane write is the little-endian fold of the block's byte window
#audit apeiron_word.absorbLane_eq_foldLane

-- Byte and state views: the maps out of the extracted `Array`/`Slice` shapes
#audit apeiron_word.byteView_length
#audit apeiron_word.byteView_digest_length
#audit apeiron_word.byteView_getD
#audit apeiron_word.byteView_repeat
#audit apeiron_word.sliceView_length
#audit apeiron_word.sliceView_len
#audit apeiron_word.sliceView_of_list
#audit apeiron_word.stateView_get
#audit apeiron_word.stateView_zero
#audit apeiron_word.stateView_set_get

-- Pointwise reads: byte windows, lane bytes, and the extracted `byteOf`
#audit apeiron_word.byteWindow_getD
#audit apeiron_word.laneBytes_getD
#audit apeiron_word.byteOf_eq_laneBytes_getD

-- Little-endian packing: the fold over a byte window is `BitVec.fromLEBytes` of that window
#audit apeiron_word.foldLane_eq_fromLEBytes
#audit apeiron_word.foldLane_eq_zero
#audit apeiron_word.from_le_bytes_denote
#audit apeiron_word.from_le_bytes_denote_list
#audit apeiron_word.fromLEBytes_cast_eq
#audit apeiron_word.from_le_bytes_eight
#audit apeiron_word.blockBytes_view
#audit apeiron_word.blockBytes_eq_list

-- State XOR the block window: pointwise read, the fold over the window, and the out-of-range lanes
#audit apeiron_word.blockVec_getD
#audit apeiron_word.blockVec_getD_ge
#audit apeiron_word.xorBlock_get
#audit apeiron_word.xorBlock_blockVec_foldLane
#audit apeiron_word.xorBlock_lane_ge

-- Loop and list infrastructure the absorb step is built from
#audit apeiron_word.foldl_congr
#audit apeiron_word.list_getD_set

-- Extracted reads: the `index_usize` specialisations and the 25-lane width
#audit apeiron_word.array_index_usize_eq
#audit apeiron_word.slice_index_usize_eq
#audit apeiron_word.array_u64_25_length

-- Parameter bridges: the `crypto` constants agree with the canonical rate constants
#audit apeiron_word.rate_256_denote
#audit apeiron_word.rate_512_denote

-- Headline: `Result α` is an `ITree` with no `map`, so the carried `map` agrees on `ok`
#audit apeiron_word.resultMap_ok
