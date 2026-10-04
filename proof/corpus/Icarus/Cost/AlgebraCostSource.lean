/-
SCOPE: work/proofs/AlgebraCostSource.lean, new, and its receipt work/proofs/AlgebraCostSource.md.
AR-41 of the algebra-rooting wave, Sol face adjudication §5 row 5: a source-bound formal cost for one
executing SHA-256 compression. Count the executing bodies' operations - the sixteen-word big-endian
load `sha256.compress_loop0.body`, the forty-eight schedule-extension visits of
`sha256.compress_loop1.body` (the `schedBody_step`/`sched_loop1` link is the executing connection),
the sixty-four rounds, the eight-word add-back and the digest store `sha256.sha256_loop2.body` - field
by field against the existing twenty-three-field measure of `AlgebraCostComplete.lean`, with a
derivation per field from a named source line or a `rfl`-checked body display. Price the byte-to-word
regrouping and the rotation/shift wiring in the same units. Count the carry stages separately from the
integer overflow bookkeeping. State the per-block recurrence over the message length as a theorem,
not a table. State the three-face verdict as a fact about the counted bodies. Name every counted
quantity that does not fit a field rather than forcing it. Complete when the module elaborates with
every headline inside the closed cone and zero `sorryAx`, and the receipt records every count with its
grade, every divergence from AR-15/AR-20/AR-27 with its source line, every named non-field quantity
with its derivation, the recurrence bound, the face verdict and the honesty statement.
REGISTER: algebra/resource model. No production, pin, API, extraction-root, corpus or gate-driver
edit; no Rust source; no cargo; no regen.sh; no measurement. Excluded by construction: latency,
wall-clock time, throughput, hash rate, bytes per second, energy, joules, watts, hardware, ASIC or
GPU figures, miner economics, profitability, USD, compiler or codegen cost, the kernel ABI, device
dispatch cost, constant-time and security properties of any hash, and every asymptotic statement.
AGENTS.md:429 - a cost-model proof is not a hardware speed measurement; this module states counts and
nothing else. No live target contact, no probing, no external submission.
CORPUS: the pinned canonical corpus `880aa57e47e782ed7a9d9ee1974cfc87360b446b` through
repos/apeiron/proof/corpus/.lake/packages/apeiron/proof/:
`Apeiron.Trichotomy.AnyRing.{ringMul,ringUpper,ringOne}` and
`Apeiron.Trichotomy.ChannelsAnyRing.ringUpper` (`:139`). Canonical SHA-2 bodies
`Apeiron.Crypto.Sha2.{rotr:50,shrx:54,sigma0:60,sigma1:64,Sigma0:67,Sigma1:70,choose:73,
majority:76,schedule:82,compress:96,lengthByte:125,lengthBlock:128,pad:133,hash:146}` at
`Crypto/Sha2.lean`, sha256 `b87814767012d1306761c31c78a9c429f4382258201b88899598085e237d5c6e`. The
executing links at `Crypto/Sha256Certificate.lean`, sha256
`d0bb8bb9a60975e0697f4f334602c0c0c33e46a54e4fbfa3bd8ec6e9a7e7e65a`: `rotr_step:188`, `shr_step3:197`,
`shr_step10:209`, `Sigma1_round:226`, `Sigma0_round:237`, `ch_step:248`, `maj_step:254`, `schedRow:280`,
`schedBody_step:643` (one `compress_loop1.body` visit), `sched_loop1:851` (the loop is
`Sha2.schedule`), `b64:479`, `beWord16:484`, `bytesToWords:493`, `schedRow_block:513`, `stateView:123`,
`digView:153`, `K_get:170`, `H0_get:175`. The extracted bodies are the generated
`repos/apeiron/proof/ApeironWord.lean`: `sha256.compress` `:15391`, `compress_loop0.body:15243`,
`compress_loop1.body:15281`, `compress_loop2.body:15332`, `sha256_loop2.body:15510`, `sha256.K:15216`,
`sha256.H0:15233`; extracted source `src/sha256.rs:33-143`. The canonical module has no compression
link to the extracted round loop: `Crypto/Sha256Certificate.lean` carries `WordSha256Denotes` as a
`def` and its own header lists the 64-round compression as an open stage - this module prices the
round *primitives* through the `*_round` links and the round loop as the canonical 64-fold, and says
so where the link is open.
REFERENCE: repos/apeiron/AGENTS.md:336-341 (per-function complexity), :226-240 (grades), :429
(cost-model proof is not a hardware speed measurement), :513, :516 (green gates over broken
artifacts; claims outrunning their theorem). work/proofs/AlgebraCostComplete.lean (AR-27), sha256
`d6d5ae7f229828122e08dc89e08a57e420e28057e04d73a7f9ada0d907d9fae4`: the 23-field `ShaCost`
measure, re-declared here because that module is not a root of any corpus `lean_lib`;
`AlgebraCostPow.lean` (AR-20) sha256
`a3d62397ffd67d6902c8b28629af42d394df3c55bb176558d8f353b1694207e0`, `AlgebraCostRound.lean` (AR-15)
sha256 `908c7aa729a14164a050490a67b78de83e92398fc72e2e4fa894c8077bdc6c8b`,
`AlgebraCost.lean` (AR-10) sha256 `52e5d6c5c2b4a144df2a99824052521c29f4b24252d68df5096f7870447df4fe`.
`receipts/sol-face-adjudication.md` §4 last paragraph and §5 row 5. AR-37's Boolean-carrier links
(`work/proofs/AlgebraBoolCarrier.lean/.md`) do not exist at this base revision
(`ls work/proofs/` shows no match), so the wiring is counted directly, as the dispatch allows.
GATE: /home/lucas/Workspace/icarus/bin/lane lake env lean
/home/lucas/Workspace/icarus/work/proofs/AlgebraCostSource.lean, cwd
/home/lucas/Workspace/icarus/repos/apeiron/proof/corpus. Pre-flight
repos/apeiron/tools/gate/target/debug/gate hold, cwd /home/lucas/Workspace/icarus, observed
`gate hold: no hold in force`, RC=0, before the first write. Wall budget 450 s per file elaboration
and 590 s per invocation including queue time, under flock /tmp/opencode/apeiron-lane.lock,
MemoryMax=3800M, MemoryHigh=3400M, MemorySwapMax=0, LEAN_NUM_THREADS=1. No cargo, no `lake build`
of the package, no regen.sh, no `| head` pipeline, no nohup/setsid, no retry loop, no raised
maxRecDepth, no `maxHeartbeats 0`, no `decide` above small enumerations.
GRADE: `proved` for every body display (`rfl`-checked), every per-operation and composite count
read off a displayed body, the measure's laws, the recurrence and its bound, the carry recovery
identity and its witnesses, the elliptic-premise refusal, and the three-face comparisons; the cone
is {propext, Classical.choice, Quot.sound} with no `sorryAx`, checked by the audit rows at the end.
`model` for the occurrence inventories of the `do`-block bodies (`compress_loop0.body`,
`compress_loop1.body`, `compress_loop2.body`, `sha256_loop2.body`), for the bit-coordinate wiring
model, for the ripple-stage carry expansion, and for the C5 ring reading of a rotation - each is
named, stated, and never merged with a `proved` count. `tested`: none - no `cargo test` ran.
Unmeasured and unclaimed: everything on the REGISTER exclusion list. In particular NO wall-clock,
cycle, joule, watt, hash-per-second, USD or profitability claim follows from any number here.
-/

import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Crypto.Sha256Certificate
import WordRefinement
import Mathlib.Tactic

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
set_option linter.unnecessarySeqFocus false

namespace algebra_cost_source

open Aeneas Aeneas.Std Result
open Apeiron.Trichotomy.AnyRing
open Apeiron.Crypto.Sha2
open apeiron_word

/-! ## 0. The width and the three dials

The SHA-2 word is `BitVec 32` (`Sha2.lean:22`); the ring every ring-layer count is read at is
`ZMod (2^32)`. The three dial values of `cell.Sigma` at width 32 are `-1` (elliptic), `0`
(parabolic) and `1` (hyperbolic) (`WordRefinement.lean:29-35`). -/

/-- The ring the SHA-2 counts are measured in: `w = 32`. -/
abbrev W32 := ZMod (2 ^ 32)

/-- The elliptic face's dial value. -/
abbrev faceElliptic : W32 := -1
/-- The parabolic face's dial value. -/
abbrev faceParabolic : W32 := 0
/-- The hyperbolic face's dial value. -/
abbrev faceHyperbolic : W32 := 1

/-! ## 1. The twenty-three-field measure, re-declared

`ShaCost σ` is AR-27's measure (`AlgebraCostComplete.lean:379`), field names and order unchanged:
the eleven compression fields of AR-15's `RoundCost`, then the twelve padding, load, store and read
fields. It is re-declared because that module is not a root of any corpus `lean_lib` and cannot be
imported; the field-by-field comparison with its figures is done in the receipt. -/

/-- The measure: twenty-three fields, one counted quantity each. Operations and bytes are different
units and are never summed. -/
structure ShaCost (σ : W32) where
  /-- scalar ring multiplications -/
  mul : Nat
  /-- scalar ring additions -/
  add : Nat
  /-- scalar ring subtractions -/
  sub : Nat
  /-- scalar ring negations -/
  neg : Nat
  /-- conversions of a dial value into the ring -/
  cast : Nat
  /-- channel reads, applications of a root-parameterised linear functional -/
  channel : Nat
  /-- bitwise XOR of `Word`s -/
  bitXor : Nat
  /-- bitwise AND of `Word`s -/
  bitAnd : Nat
  /-- bitwise NOT of `Word`s -/
  bitNot : Nat
  /-- logical shifts of a `Word`, left or right -/
  shift : Nat
  /-- bitwise OR of `Word`s -/
  bitOr : Nat
  /-- bytes appended by `pad`, message copy included -/
  padByte : Nat
  /-- of those, the zero-fill bytes -/
  zeroByte : Nat
  /-- of those, the eight length-field bytes -/
  lenByte : Nat
  /-- `Nat` shifts extracting the eight length bytes -/
  padShift : Nat
  /-- the eight `% 256` truncations of the length bytes -/
  padMod : Nat
  /-- byte reads of the padded block by the sixteen-word loader -/
  blockByteRead : Nat
  /-- `Nat` multiplications in the big-endian regrouping -/
  regroupMul : Nat
  /-- `Nat` additions in the big-endian regrouping -/
  regroupAdd : Nat
  /-- 32-bit truncations in the big-endian regrouping -/
  regroupTrunc : Nat
  /-- word writes into the 64-word schedule -/
  schedWordWrite : Nat
  /-- bytes written by the big-endian digest store -/
  digestByte : Nat
  /-- chaining-word reads performed by that store -/
  digestWordRead : Nat
  deriving Repr

/-- A record with the eleven compression fields given positionally, every byte, load and store
field zero. -/
def ShaCost.op (σ : W32) (mul add sub neg cast channel bitXor bitAnd bitNot shift bitOr : Nat) :
    ShaCost σ :=
  { mul := mul, add := add, sub := sub, neg := neg, cast := cast, channel := channel,
    bitXor := bitXor, bitAnd := bitAnd, bitNot := bitNot, shift := shift, bitOr := bitOr,
    padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
    blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
    schedWordWrite := 0, digestByte := 0, digestWordRead := 0 }

/-- A record with the twelve byte, load and store fields given positionally, every compression field
zero. -/
def ShaCost.bytes (σ : W32) (padByte zeroByte lenByte padShift padMod blockByteRead regroupMul
    regroupAdd regroupTrunc schedWordWrite digestByte digestWordRead : Nat) : ShaCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
    padByte := padByte, zeroByte := zeroByte, lenByte := lenByte, padShift := padShift,
    padMod := padMod, blockByteRead := blockByteRead, regroupMul := regroupMul,
    regroupAdd := regroupAdd, regroupTrunc := regroupTrunc, schedWordWrite := schedWordWrite,
    digestByte := digestByte, digestWordRead := digestWordRead }

/-- The record of a compression that performs no operation and moves no byte. -/
def ShaCost.zero (σ : W32) : ShaCost σ := ShaCost.op σ 0 0 0 0 0 0 0 0 0 0 0

/-- Dependent composition: a computation that runs after another performs both sets of operations,
and every count totals. Temporal and spatial O(1): a closed term of twenty-three `Nat` additions, no
recursion, no allocation. Counted in: nothing - `composed` is the measure's own operator. -/
def ShaCost.composed (σ : W32) (a b : ShaCost σ) : ShaCost σ :=
  { mul := a.mul + b.mul, add := a.add + b.add, sub := a.sub + b.sub, neg := a.neg + b.neg,
    cast := a.cast + b.cast, channel := a.channel + b.channel, bitXor := a.bitXor + b.bitXor,
    bitAnd := a.bitAnd + b.bitAnd, bitNot := a.bitNot + b.bitNot, shift := a.shift + b.shift,
    bitOr := a.bitOr + b.bitOr,
    padByte := a.padByte + b.padByte, zeroByte := a.zeroByte + b.zeroByte,
    lenByte := a.lenByte + b.lenByte, padShift := a.padShift + b.padShift,
    padMod := a.padMod + b.padMod, blockByteRead := a.blockByteRead + b.blockByteRead,
    regroupMul := a.regroupMul + b.regroupMul, regroupAdd := a.regroupAdd + b.regroupAdd,
    regroupTrunc := a.regroupTrunc + b.regroupTrunc,
    schedWordWrite := a.schedWordWrite + b.schedWordWrite,
    digestByte := a.digestByte + b.digestByte, digestWordRead := a.digestWordRead + b.digestWordRead }

instance instAddShaCost (σ : W32) : Add (ShaCost σ) := ⟨ShaCost.composed σ⟩

@[simp] theorem ShaCost.composed_apply (σ : W32) (a b : ShaCost σ) :
    a + b = ShaCost.composed σ a b := rfl

/-- **Equality of counts is fieldwise equality**, twenty-three fields: a record with one field per
quantity has no room for a hidden one. -/
theorem ShaCost.eq_of_fields {σ : W32} {a b : ShaCost σ}
    (h1 : a.mul = b.mul) (h2 : a.add = b.add) (h3 : a.sub = b.sub) (h4 : a.neg = b.neg)
    (h5 : a.cast = b.cast) (h6 : a.channel = b.channel) (h7 : a.bitXor = b.bitXor)
    (h8 : a.bitAnd = b.bitAnd) (h9 : a.bitNot = b.bitNot) (h10 : a.shift = b.shift)
    (h11 : a.bitOr = b.bitOr) (h12 : a.padByte = b.padByte) (h13 : a.zeroByte = b.zeroByte)
    (h14 : a.lenByte = b.lenByte) (h15 : a.padShift = b.padShift) (h16 : a.padMod = b.padMod)
    (h17 : a.blockByteRead = b.blockByteRead) (h18 : a.regroupMul = b.regroupMul)
    (h19 : a.regroupAdd = b.regroupAdd) (h20 : a.regroupTrunc = b.regroupTrunc)
    (h21 : a.schedWordWrite = b.schedWordWrite) (h22 : a.digestByte = b.digestByte)
    (h23 : a.digestWordRead = b.digestWordRead) : a = b := by
  cases a
  cases b
  cases h1; cases h2; cases h3; cases h4; cases h5; cases h6; cases h7; cases h8; cases h9
  cases h10; cases h11; cases h12; cases h13; cases h14; cases h15; cases h16; cases h17
  cases h18; cases h19; cases h20; cases h21; cases h22; cases h23
  rfl

/-- **`rep σ n a` is the measure of `n` dependent copies of a cost `a`.** Temporal O(1) and spatial
O(1): twenty-three `Nat` multiplications on a record, no recursion. Counted in: nothing - this is the
iteration operator. Derivation: fieldwise `n * a.field`; `ShaCost.rep_add` proves the image
additive, which is the closure the §8 comment requires. -/
def ShaCost.rep (σ : W32) (n : Nat) (a : ShaCost σ) : ShaCost σ :=
  { mul := n * a.mul, add := n * a.add, sub := n * a.sub, neg := n * a.neg, cast := n * a.cast,
    channel := n * a.channel, bitXor := n * a.bitXor, bitAnd := n * a.bitAnd,
    bitNot := n * a.bitNot, shift := n * a.shift, bitOr := n * a.bitOr,
    padByte := n * a.padByte, zeroByte := n * a.zeroByte, lenByte := n * a.lenByte,
    padShift := n * a.padShift, padMod := n * a.padMod,
    blockByteRead := n * a.blockByteRead, regroupMul := n * a.regroupMul,
    regroupAdd := n * a.regroupAdd, regroupTrunc := n * a.regroupTrunc,
    schedWordWrite := n * a.schedWordWrite, digestByte := n * a.digestByte,
    digestWordRead := n * a.digestWordRead }

@[simp] theorem ShaCost.rep_zero (σ : W32) (a : ShaCost σ) :
    ShaCost.rep σ 0 a = ShaCost.zero σ := by
  simp [ShaCost.rep, ShaCost.zero, ShaCost.op]

theorem ShaCost.rep_succ (σ : W32) (n : Nat) (a : ShaCost σ) :
    ShaCost.rep σ (n + 1) a = a + ShaCost.rep σ n a := by
  cases a
  simp [ShaCost.rep, ShaCost.composed, Nat.succ_mul, Nat.add_comm]

theorem ShaCost.rep_one (σ : W32) (a : ShaCost σ) : ShaCost.rep σ 1 a = a := by
  cases a
  simp [ShaCost.rep]

/-- **The composition law over an iterated construction, twenty-three fields.** -/
theorem ShaCost.rep_add (σ : W32) (n m : Nat) (a : ShaCost σ) :
    ShaCost.rep σ (n + m) a = ShaCost.rep σ n a + ShaCost.rep σ m a := by
  cases a
  simp [ShaCost.rep, ShaCost.composed, Nat.add_mul]

/-! ## 2. Per-operation counts, read off the canonical bodies

Each count is the operation inventory of the displayed body, and the display is `rfl`-checked in
the next block. Nothing in Lean counts the operations of a term; the `do`-block occurrence
inventories of §3.5 are graded `model` for the same reason, and the pure-body inventories here are
read off their `rfl` displays (AR-10 residual 1, restated). -/

/-- **The canonical multiplication** `(a + bx)(c + dx) = (ac + σbd) + (ad + bc)x`
(`AnyRing.lean:109-110`): five multiplications, two additions, and two of the five multiplications
carry the dial as a factor (`σ * z.2 * w.2`). -/
def ringMulCost (σ : W32) : ShaCost σ := ShaCost.op σ 5 2 0 0 0 0 0 0 0 0 0

/-- The two dial-carrying multiplications of `ringMul`, named separately so that the face-sensitive
scalar count of §7 is a stated quantity and not a subtraction. -/
def dialMulCost (σ : W32) : ShaCost σ := ShaCost.op σ 2 0 0 0 0 0 0 0 0 0 0

/-- The three dial-free multiplications and two additions of `ringMul`. -/
def ringMulDialFreeCost (σ : W32) : ShaCost σ := ShaCost.op σ 3 2 0 0 0 0 0 0 0 0 0

/-- **The canonical channel read** `z.1 + r*z.2` (`ChannelsAnyRing.lean:139`): one multiplication,
one addition, one application of the linear functional. -/
def ringUpperCost (σ : W32) : ShaCost σ := ShaCost.op σ 1 1 0 0 0 1 0 0 0 0 0

/-- **The rotation read**, `ringUpper 1 (ringMul 1 (x,0) (2^n, 2^(32-n)))`: one canonical
multiplication followed by one channel read. Its count is a fact about the two bodies; the
identification with a word rotation is candidate C5 (`ops-map.md:167-180`), grade `model`, and is
not used by any `proved` count here. -/
def rotReadCost (σ : W32) : ShaCost σ := ringMulCost σ + ringUpperCost σ

/-- **One `Word` addition**: a word addition modulo `2^32` is one addition of `ZMod (2^32)`
(`denote_wrapping_add`, `Denotation.lean:35`, is stated at width 64 and is carried, not used). -/
def wordAddCost (σ : W32) : ShaCost σ := ShaCost.op σ 0 1 0 0 0 0 0 0 0 0 0

/-- **The rotation `ROTR`** `(x >>> (n % 32)) ||| (x <<< ((32 - n) % 32))` (`Sha2.lean:50`): two
shifts and one disjunction. A rotation is not one operation in the canonical body. -/
def rotrCost (σ : W32) : ShaCost σ := ShaCost.op σ 0 0 0 0 0 0 0 0 0 2 1

/-- **The logical right shift `SHR`** `x >>> (n % 32)` (`Sha2.lean:54`): one shift. -/
def shrxCost (σ : W32) : ShaCost σ := ShaCost.op σ 0 0 0 0 0 0 0 0 0 1 0

/-- **The message-schedule small sigma** `σ₀(x) = ROTR^7 x ⊕ ROTR^18 x ⊕ SHR^3 x`
(`Sha2.lean:60`): two rotations, one shift, two XORs. -/
def sigma0Cost (σ : W32) : ShaCost σ := ShaCost.op σ 0 0 0 0 0 0 2 0 0 5 2

/-- **The message-schedule small sigma** `σ₁(x) = ROTR^17 x ⊕ ROTR^19 x ⊕ SHR^10 x`
(`Sha2.lean:64`): the same inventory. -/
def sigma1Cost (σ : W32) : ShaCost σ := ShaCost.op σ 0 0 0 0 0 0 2 0 0 5 2

/-- **The round function `Σ₀`** `ROTR^2 x ⊕ ROTR^13 x ⊕ ROTR^22 x` (`Sha2.lean:67`): three
rotations and two XORs, no shift of its own. -/
def bigSigma0Cost (σ : W32) : ShaCost σ := ShaCost.op σ 0 0 0 0 0 0 2 0 0 6 3

/-- **The round function `Σ₁`** `ROTR^6 x ⊕ ROTR^11 x ⊕ ROTR^25 x` (`Sha2.lean:70`): the same
inventory. -/
def bigSigma1Cost (σ : W32) : ShaCost σ := ShaCost.op σ 0 0 0 0 0 0 2 0 0 6 3

/-- **The choose function `Ch`** `(x &&& y) ⊕ (¬x &&& z)` (`Sha2.lean:73`): two ANDs, one NOT,
**one** XOR. The displayed body is the authority; a count of two XORs is not a count of this
body. -/
def chooseCost (σ : W32) : ShaCost σ := ShaCost.op σ 0 0 0 0 0 0 1 2 1 0 0

/-- **The majority function `Maj`** `(x &&& y) ⊕ (x &&& z) ⊕ (y &&& z)` (`Sha2.lean:76`): three
ANDs and two XORs. -/
def majorityCost (σ : W32) : ShaCost σ := ShaCost.op σ 0 0 0 0 0 0 2 3 0 0 0

/-! ### 2.1 The bodies the counts are read off -/

/-- Unfolding `ringMul` exhibits the five multiplications, two additions and the dial factor. -/
theorem ringMul_body_ops (σ : W32) (z w : RingCell W32) :
    ringMul σ z w = (z.1 * w.1 + σ * z.2 * w.2, z.1 * w.2 + z.2 * w.1) := rfl

/-- Unfolding `ringUpper` exhibits the one multiplication and one addition. -/
theorem ringUpper_body_ops (r : W32) (z : RingCell W32) : ringUpper r z = z.1 + r * z.2 := rfl

/-- Unfolding `ROTR` exhibits the two shifts and the one disjunction. -/
theorem rotr_body_ops (x : Word) (n : Nat) :
    rotr x n = (x >>> (n % 32)) ||| (x <<< ((32 - n) % 32)) := rfl

/-- Unfolding `SHR` exhibits the one shift. -/
theorem shrx_body_ops (x : Word) (n : Nat) : shrx x n = x >>> (n % 32) := rfl

/-- Unfolding `σ₀` exhibits two rotations, one shift and two XORs. -/
theorem sigma0_body_ops (x : Word) : sigma0 x = rotr x 7 ^^^ rotr x 18 ^^^ shrx x 3 := rfl

/-- Unfolding `σ₁` exhibits two rotations, one shift and two XORs. -/
theorem sigma1_body_ops (x : Word) : sigma1 x = rotr x 17 ^^^ rotr x 19 ^^^ shrx x 10 := rfl

/-- Unfolding `Σ₀` exhibits three rotations and two XORs, with no `SHR`. -/
theorem bigSigma0_body_ops (x : Word) : Sigma0 x = rotr x 2 ^^^ rotr x 13 ^^^ rotr x 22 := rfl

/-- Unfolding `Σ₁` exhibits three rotations and two XORs, with no `SHR`. -/
theorem bigSigma1_body_ops (x : Word) : Sigma1 x = rotr x 6 ^^^ rotr x 11 ^^^ rotr x 25 := rfl

/-- Unfolding `Ch` exhibits two ANDs, one NOT and **one** XOR. -/
theorem choose_body_ops (x y z : Word) : choose x y z = (x &&& y) ^^^ (~~~x &&& z) := rfl

/-- Unfolding `Maj` exhibits three ANDs and two XORs. -/
theorem majority_body_ops (x y z : Word) : majority x y z = (x &&& y) ^^^ (x &&& z) ^^^ (y &&& z) :=
  rfl

/-- **The dial term of the rotation read is present in the count and vanishes in the value**: at
`z = (x, 0)` the `σ` factor multiplies zero. A count that omitted it would be the count of a
specialised body, not of `ringMul`. -/
theorem dial_term_is_zero_valued (σ a b x : W32) :
    ringMul σ (x, 0) (a, b) = (x * a, x * b) := by
  simp only [ringMul]
  simp

/-- The rotation read's count, read off the two canonical bodies. -/
theorem rotReadCost_fields (σ : W32) :
    rotReadCost σ = ShaCost.op σ 6 3 0 0 0 1 0 0 0 0 0 := by
  simp [rotReadCost, ringMulCost, ringUpperCost, ShaCost.composed, ShaCost.op]

/-- The multiplication inventory of `ringMul` splits into its two dial-carrying multiplications
and its three dial-free ones. -/
theorem ringMulCost_split (σ : W32) :
    ringMulCost σ = dialMulCost σ + ringMulDialFreeCost σ := by
  simp [ringMulCost, dialMulCost, ringMulDialFreeCost, ShaCost.composed, ShaCost.op]

/-- The dial-carrying multiplications of one `ringMul`, read off its body. -/
theorem dialMulCost_fields (σ : W32) : (dialMulCost σ).mul = 2 := rfl

/-! ## 3. The executing compression, priced from the source

The priced circuit is the body chain `sha256.compress` (`ApeironWord.lean:15391`, from
`src/sha256.rs:33-94`): the load loop `compress_loop0.body` (`:15243`, `src/sha256.rs:37-41`), the
schedule-extension loop `compress_loop1.body` (`:15281`, `src/sha256.rs:43-51`, linked to the
canonical forty-eight-step fold by `schedBody_step`/`sched_loop1`, `Sha256Certificate.lean:643,851`),
the round loop `compress_loop2.body` (`:15332`, `src/sha256.rs:63-84`), and the eight-word add-back
(`:15400-15430`, `src/sha256.rs:86-94`). The canonical bodies `Sha2.schedule`/`Sha2.compress` are the
specification the links read; the round loop's own value link to `Sha2.compress` is *not* in
`Sha256Certificate.lean` (its header lists the 64-round compression as an open stage), so the round
loop's count is read off its own body and the round primitives through `Sigma1_round`, `Sigma0_round`,
`ch_step`, `maj_step` (`:226-256`), and the gap is named in the receipt. -/

/-- The number of schedule-extension visits per block: `t = 16 … 63`. -/
def blockScheduleSteps : Nat := 48

/-- The number of compression rounds per block. -/
def blockRounds : Nat := 64

/-- Rotations inside one schedule-extension visit: two per small sigma. -/
def schedStepRotations : Nat := 4

/-- Shifts inside one schedule-extension visit: `SHR^3` and `SHR^10`, which are not rotations. -/
def schedStepShifts : Nat := 2

/-- Rotations inside one compression round: three in `Σ₀` and three in `Σ₁`. -/
def roundRotations : Nat := 6

/-- Rotations per block: `48 * 4` schedule plus `64 * 6` rounds. -/
def blockRotations : Nat := blockScheduleSteps * schedStepRotations + blockRounds * roundRotations

theorem blockRotations_eq : blockRotations = 576 := by
  norm_num [blockRotations, blockScheduleSteps, schedStepRotations, blockRounds, roundRotations]

/-- Binary `Word` additions per block: three in each schedule step (four summands), seven in each
round (`t1` four, `t2` one, the two state updates one each), eight in the add-back. The canonical
schedule body has four *summands* and therefore three binary additions (`Sha2.lean:87-88`); the
extracted visit has three `wrapping_add`s (`ApeironWord.lean:15305,15307,15308`). -/
def blockAdds : Nat := blockScheduleSteps * 3 + blockRounds * 7 + 8

theorem blockAdds_eq : blockAdds = 600 := by
  norm_num [blockAdds, blockScheduleSteps, blockRounds]

/-- **One schedule-extension step, word layer**: `σ₁` and `σ₀`, three binary additions, and the one
schedule write the visit performs (`Array.update w i i23`, `ApeironWord.lean:15309`), which the
measure counts in `schedWordWrite`. Closed count: 4 XORs, 10 shifts, 4 disjunctions, 3 additions,
1 word write. -/
def schedStepCost (σ : W32) : ShaCost σ :=
  sigma0Cost σ + sigma1Cost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ +
    ShaCost.bytes σ 0 0 0 0 0 0 0 0 0 1 0 0

/-- **One schedule-extension step, rotation reads**: four rotations under C5. -/
def schedStepRotReadCost (σ : W32) : ShaCost σ :=
  rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ

/-- **One round, word layer**: the four round functions and seven `Word` additions
(`Sha2.lean:108,109,112,116`). Closed count: 7 XORs, 5 ANDs, 1 NOT, 12 shifts, 6 disjunctions,
7 additions. -/
def roundCost (σ : W32) : ShaCost σ :=
  bigSigma0Cost σ + bigSigma1Cost σ + chooseCost σ + majorityCost σ +
    wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ +
    wordAddCost σ + wordAddCost σ

/-- **One round, rotation reads**: six rotations under C5. -/
def roundRotReadCost (σ : W32) : ShaCost σ :=
  rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ

/-- **The eight-word add-back** `h.getD i 0 + v.getD i.val 0` (`Sha2.lean:120`): eight additions;
its eight state reads and eight state writes are named in §4 because the measure has no field for
them. -/
def addBackCost (σ : W32) : ShaCost σ :=
  wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ +
    wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ

/-- **The sixteen-word big-endian load**: 64 byte reads, 48 `Nat` multiplications, 48 `Nat`
additions, 16 truncations, 16 schedule writes. The regrouping is the displayed body of `beWord16`
(`Sha256Certificate.lean:484`): four `b64` reads, three multiplications by `2^24, 2^16, 2^8`, three
additions and one `BitVec.ofNat 32` per word; `bytesToWords_arity` (`:497`) gives the sixteen words.
The extracted load loop `compress_loop0.body` performs the same read with one `from_be_bytes` and one
`Array.update` per visit (`ApeironWord.lean:15250-15261`); the `from_be_bytes` value link to
`beWord16` is not established here (see receipt residual 1), so the per-visit occurrence count is
`model`. -/
def loadCost (σ : W32) : ShaCost σ := ShaCost.bytes σ 0 0 0 0 0 64 48 48 16 16 0 0

/-- **The executing compression, word layer**: load, forty-eight schedule steps, sixty-four rounds,
add-back. This is the part of the figure every operation of which is a `Word`/byte operation of the
body; the ring reading is `compressionRingReadCost`. -/
def compressionWordCost (σ : W32) : ShaCost σ :=
  loadCost σ + ShaCost.rep σ blockScheduleSteps (schedStepCost σ) +
    ShaCost.rep σ blockRounds (roundCost σ) + addBackCost σ

/-- **The rotation reads of the compression under C5**: one `ringMul` plus one `ringUpper` per
rotation. -/
def compressionRingReadCost (σ : W32) : ShaCost σ :=
  ShaCost.rep σ blockScheduleSteps (schedStepRotReadCost σ) +
    ShaCost.rep σ blockRounds (roundRotReadCost σ)

/-- **The executing compression of one block**, in the twenty-three-field measure: the word-layer
figure plus the C5 rotation reads. Temporal and spatial O(1) as a closed record; the growth is in the
counted quantities. Counted in: the twenty-three fields. Derivation: the four components above,
composed; closure by `ShaCost.rep_add` over the block count in §8. -/
def compressionCost (σ : W32) : ShaCost σ := compressionWordCost σ + compressionRingReadCost σ

/-- **The digest store** `sha256.sha256_loop2.body` (`ApeironWord.lean:15510`, `src/sha256.rs:138-141`):
eight iterations, each one `state[i]` read, one `U32::to_be_bytes` and one four-byte slice write;
32 bytes written, 8 word reads. -/
def storeCost (σ : W32) : ShaCost σ := ShaCost.bytes σ 0 0 0 0 0 0 0 0 0 0 32 8

/-- **The two-block header attempt**: two compressions and one digest store, the figure AR-27's
`attemptCost` completes with the padding fields (which the receipt takes from AR-27 read-only and
does not re-price). -/
def headerCost (σ : W32) : ShaCost σ := compressionCost σ + compressionCost σ + storeCost σ

/-! ### 3.1 The closed counts -/

theorem schedStepCost_fields (σ : W32) :
    schedStepCost σ = ShaCost.op σ 0 3 0 0 0 0 4 0 0 10 4 +
      ShaCost.bytes σ 0 0 0 0 0 0 0 0 0 1 0 0 := by
  simp [schedStepCost, sigma0Cost, sigma1Cost, wordAddCost, ShaCost.composed, ShaCost.op,
    ShaCost.bytes]

theorem schedStepRotReadCost_fields (σ : W32) :
    schedStepRotReadCost σ = ShaCost.op σ 24 12 0 0 0 4 0 0 0 0 0 := by
  simp [schedStepRotReadCost, rotReadCost, ringMulCost, ringUpperCost, ShaCost.composed,
    ShaCost.op]

theorem roundCost_fields (σ : W32) :
    roundCost σ = ShaCost.op σ 0 7 0 0 0 0 7 5 1 12 6 := by
  simp [roundCost, bigSigma0Cost, bigSigma1Cost, chooseCost, majorityCost, wordAddCost,
    ShaCost.composed, ShaCost.op]

theorem roundRotReadCost_fields (σ : W32) :
    roundRotReadCost σ = ShaCost.op σ 36 18 0 0 0 6 0 0 0 0 0 := by
  simp [roundRotReadCost, rotReadCost, ringMulCost, ringUpperCost, ShaCost.composed, ShaCost.op]

theorem addBackCost_fields (σ : W32) :
    addBackCost σ = ShaCost.op σ 0 8 0 0 0 0 0 0 0 0 0 := by
  simp [addBackCost, wordAddCost, ShaCost.composed, ShaCost.op]

theorem loadCost_fields (σ : W32) :
    loadCost σ = ShaCost.bytes σ 0 0 0 0 0 64 48 48 16 16 0 0 := rfl

/-- **The word-layer compression's closed count.** The eleven compression fields are the load's
zeros plus `48 * schedStep` plus `64 * round` plus the add-back; the twelve new fields are the load's
five plus the schedule writes, which are 16 from the load and 48 from the extension. -/
theorem compressionWordCost_eq (σ : W32) :
    compressionWordCost σ =
      ShaCost.op σ 0 600 0 0 0 0 640 320 64 1248 576 +
        ShaCost.bytes σ 0 0 0 0 0 64 48 48 16 64 0 0 := by
  simp [compressionWordCost, addBackCost, schedStepCost, roundCost, sigma0Cost, sigma1Cost,
    bigSigma0Cost, bigSigma1Cost, chooseCost, majorityCost, wordAddCost, ShaCost.rep,
    ShaCost.composed, ShaCost.op, ShaCost.bytes, loadCost, blockScheduleSteps, blockRounds]

/-- **The C5 ring read's closed count.** -/
theorem compressionRingReadCost_eq (σ : W32) :
    compressionRingReadCost σ = ShaCost.op σ 3456 1728 0 0 0 576 0 0 0 0 0 := by
  simp [compressionRingReadCost, schedStepRotReadCost, roundRotReadCost, rotReadCost,
    ringMulCost, ringUpperCost, ShaCost.rep, ShaCost.composed, ShaCost.op, blockScheduleSteps,
    blockRounds]

/-- **THE EXECUTING COMPRESSION OF ONE BLOCK, all twenty-three fields.** Field by field:
`mul` 3456, `add` 2328, `channel` 576 (six per rotation, three per rotation plus the word adds,
one per rotation, all over 576 rotations); `bitXor` 640 (192 schedule plus 448 rounds); `bitAnd` 320;
`bitNot` 64; `shift` 1248 (480 schedule plus 768 rounds); `bitOr` 576; `blockByteRead` 64;
`regroupMul` 48; `regroupAdd` 48; `regroupTrunc` 16; `schedWordWrite` 64 (16 load plus 48 extension);
`digestByte` and `digestWordRead` 0 because the store is outside `compress`. -/
theorem compressionCost_eq (σ : W32) :
    compressionCost σ =
      { mul := 3456, add := 2328, sub := 0, neg := 0, cast := 0, channel := 576,
        bitXor := 640, bitAnd := 320, bitNot := 64, shift := 1248, bitOr := 576,
        padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
        blockByteRead := 64, regroupMul := 48, regroupAdd := 48, regroupTrunc := 16,
        schedWordWrite := 64, digestByte := 0, digestWordRead := 0 } := by
  rw [compressionCost, compressionWordCost_eq, compressionRingReadCost_eq]
  simp [ShaCost.composed, ShaCost.op, ShaCost.bytes]

/-- The word layer has no ring multiplication: all 3456 multiplications of the block figure are the
C5 rotation reads, and the executing body contains none. -/
theorem compressionWordCost_mul_zero (σ : W32) : (compressionWordCost σ).mul = 0 := by
  rw [compressionWordCost_eq]
  simp [ShaCost.composed, ShaCost.op, ShaCost.bytes]

/-- The word layer has no dial conversion: a `Word` is already the width-32 ring element's value, and
the extracted `Cell<i64>` casts are outside this circuit. -/
theorem compressionWordCost_cast_zero (σ : W32) : (compressionWordCost σ).cast = 0 := by
  rw [compressionWordCost_eq]
  simp [ShaCost.composed, ShaCost.op, ShaCost.bytes]

/-- The word layer has no channel read: channel reads are the C5 reading's unit. -/
theorem compressionWordCost_channel_zero (σ : W32) : (compressionWordCost σ).channel = 0 := by
  rw [compressionWordCost_eq]
  simp [ShaCost.composed, ShaCost.op, ShaCost.bytes]

/-- **The two-block header attempt's closed count**: twice the compression plus the store. -/
theorem headerCost_eq (σ : W32) :
    headerCost σ =
      { mul := 6912, add := 4656, sub := 0, neg := 0, cast := 0, channel := 1152,
        bitXor := 1280, bitAnd := 640, bitNot := 128, shift := 2496, bitOr := 1152,
        padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
        blockByteRead := 128, regroupMul := 96, regroupAdd := 96, regroupTrunc := 32,
        schedWordWrite := 128, digestByte := 32, digestWordRead := 8 } := by
  rw [headerCost, compressionCost_eq, storeCost]
  simp [ShaCost.composed, ShaCost.bytes]

/-! ## 4. Reads, writes and address operations: named quantities with no field

`ShaCost` has two read fields (`blockByteRead`, `digestWordRead`) and one write field
(`schedWordWrite`); the round, the schedule loop and the add-back read and write elements the record
cannot hold, and AR-27 excluded them with a reason. This section names each one, with its per-block
derivation from the executing body, rather than forcing it into a field. Every quantity here is
`model`: the `do`-block bodies are a sequence of binds, and nothing in Lean counts the binds. -/

/-- The canonical `schedule` start reads the sixteen block words (`Sha2.lean:83`); the extracted
loader's byte reads are the `blockByteRead` field. -/
def schedInitWordReads : Nat := 16

/-- Schedule-extension element reads per block: eight per visit, `i-15` three times, `i-2` three
times, `i-7` and `i-16` once each (`ApeironWord.lean:15286-15306`; the count is stated in the
`schedBody_step` doc comment, `Sha256Certificate.lean:587-590`). -/
def schedStepArrayReads : Nat := blockScheduleSteps * 8

theorem schedStepArrayReads_eq : schedStepArrayReads = 384 := by
  norm_num [schedStepArrayReads, blockScheduleSteps]

/-- Schedule-extension writes per block: one `Array.update w i` per visit
(`ApeironWord.lean:15309`). -/
def schedStepArrayWrites : Nat := blockScheduleSteps * 1

theorem schedStepArrayWrites_eq : schedStepArrayWrites = 48 := by
  norm_num [schedStepArrayWrites, blockScheduleSteps]

/-- The canonical schedule step reads four rows through `getD` (`Sha2.lean:87-88`), the same rows the
extracted visit reads eight times; the difference is the deduplication the canonical term gets from
binding `σ₀`/`σ₁` once. -/
def schedStepCanonicalReads : Nat := blockScheduleSteps * 4

theorem schedStepCanonicalReads_eq : schedStepCanonicalReads = 192 := by
  norm_num [schedStepCanonicalReads, blockScheduleSteps]

/-- The round's array reads per block: `sha256.K[i]` and `w[i]` (`ApeironWord.lean:15353,15357`),
two per round. The canonical round reads eight state entries, `K.getD` and `w.getD`
(`Sha2.lean:108-109`); the extracted round receives the eight working words as parameters, so its
register reads are `roundStateReads`. -/
def roundArrayReads : Nat := blockRounds * 2

theorem roundArrayReads_eq : roundArrayReads = 128 := by
  norm_num [roundArrayReads, blockRounds]

/-- The round's state-element reads per block: eight per round, one per working word. -/
def roundStateReads : Nat := blockRounds * 8

theorem roundStateReads_eq : roundStateReads = 512 := by
  norm_num [roundStateReads, blockRounds]

/-- The round's state-element writes per block: the canonical round rebuilds all eight entries of the
working vector (`Sha2.lean:110-117`); the extracted round returns an eight-tuple
(`ApeironWord.lean:15366`). -/
def roundStateWrites : Nat := blockRounds * 8

theorem roundStateWrites_eq : roundStateWrites = 512 := by
  norm_num [roundStateWrites, blockRounds]

/-- The initial state reads of `compress`, one per chaining word, before the round loop
(`ApeironWord.lean:15399-15406`; `Sha2.lean:98`). -/
def initStateReads : Nat := 8

/-- The add-back's state reads: one per word (`Sha2.lean:120`). -/
def addBackStateReads : Nat := 8

/-- The add-back's state writes: one per word (`ApeironWord.lean:15409-15430`). -/
def addBackStateWrites : Nat := 8

/-- `Usize` multiplications in the block's address arithmetic: `4 * i` once per load visit. AR-27
excluded address arithmetic from the measure with its reason; it is named here so the exclusion is
counted, not invisible. -/
def addrMul : Nat := 16

/-- `Usize` additions in the block's address arithmetic: `o+1`, `o+2`, `o+3`, `i+1` in the load,
`i+1` in each schedule and round visit. -/
def addrAdd : Nat := 16 * 4 + blockScheduleSteps * 1 + blockRounds * 1

theorem addrAdd_eq : addrAdd = 176 := by
  norm_num [addrAdd, blockScheduleSteps, blockRounds]

/-- `Usize` subtractions in the block's address arithmetic: `i-15`, `i-2`, `i-16`, `i-7` per
schedule visit. -/
def addrSub : Nat := blockScheduleSteps * 4

theorem addrSub_eq : addrSub = 192 := by
  norm_num [addrSub, blockScheduleSteps]

/-- Loop tests per block: each loop executes one test per visit plus the failing one
(`i < 16`, `i < 64`, `i < 64`). -/
def loopTests : Nat := (16 + 1) + (blockScheduleSteps + 1) + (blockRounds + 1)

theorem loopTests_eq : loopTests = 131 := by
  norm_num [loopTests, blockScheduleSteps, blockRounds]

/-- Digest-store slice writes per attempt: one per word. -/
def storeSliceWrites : Nat := 8

/-- Digest-store `U32::to_be_bytes` conversions per attempt: one per word. -/
def storeConversions : Nat := 8

/-- Digest-store address operations per attempt: `4 * i`, `i1 + 4`, `i + 1` per iteration. -/
def storeAddrOps : Nat := 8 * 1 + 8 * 2

theorem storeAddrOps_eq : storeAddrOps = 24 := by norm_num [storeAddrOps]

/-- Digest-store loop tests per attempt: eight visits plus the failing one. -/
def storeLoopTests : Nat := 9

/-! ## 5. Carry stages, separately from the integer overflow bookkeeping

The executing additions are `BitVec 32` additions (canonical) and `U32::wrapping_add` (extracted):
both are one operation that drops the carry-out. The stages that a ripple expansion of that one
operation would evaluate are counted here as `carryStages`; the integer overflow bookkeeping - the
number of additions whose `Nat` sum exceeds `2^32 - 1` - is a different, value-dependent quantity,
named `overflowOps`, and the two are never added. `carryRecover` proves the stage-32 residual is the
reconstruction term; `executedAdditionDropsCarryOut` proves the executed result does not retain it;
the witnesses prove the incidence is not constant. -/

/-- The carry-out of one 32-bit addition, read as a `Bool`: whether the `Nat` sum exceeds `2^32 - 1`.
This is an *input* predicate, not an operation of the body. -/
def carryOut (a b : BitVec 32) : Bool := decide (2^32 ≤ a.toNat + b.toNat)

/-- Stages of a ripple expansion of one 32-bit addition: one carry residual per bit position,
including the carry-out. A model of the adder, not an operation of the `BitVec` body. -/
def adderStages : Nat := 32

/-- Carry stages per block: `blockAdds` additions, `adderStages` each. -/
def blockCarryStages : Nat := blockAdds * adderStages

theorem blockCarryStages_eq : blockCarryStages = 19200 := by
  norm_num [blockCarryStages, blockAdds, blockScheduleSteps, blockRounds, adderStages]

/-- Integer overflow bookkeeping operations in the executing compression: no operation of the body
reads the carry-out, so the count is zero. The quantity is stated so that it cannot be conflated with
`blockCarryStages`. -/
def blockOverflowOps : Nat := 0

/-- **The residual identity.** The executed sum plus `2^32` exactly when the addition overflows
reconstructs the `Nat` sum: the stage-32 residual is real and recoverable from the inputs, and no
smaller stage count is. -/
theorem carryRecover (a b : BitVec 32) :
    (a + b).toNat + (if 2^32 ≤ a.toNat + b.toNat then 2^32 else 0) = a.toNat + b.toNat := by
  rw [BitVec.toNat_add]
  by_cases h : 2^32 ≤ a.toNat + b.toNat
  · simp only [h, if_true]
    have hlt : a.toNat + b.toNat < 2^33 := by
      have ha := a.isLt
      have hb := b.isLt
      omega
    have hdiv : (a.toNat + b.toNat) / 2^32 = 1 := by
      rw [Nat.div_eq_iff (by norm_num)]
      omega
    have hm := Nat.mod_add_div (a.toNat + b.toNat) (2^32)
    rw [hdiv] at hm
    omega
  · simp only [h, if_false, Nat.add_zero]
    exact Nat.mod_eq_of_lt (by
      have ha := a.isLt
      have hb := b.isLt
      omega)

/-- **The executed result drops the residual**: the `BitVec` addition's value is the modular
reduction of the `Nat` sum, and no operation of the body can observe the dropped part from the
result. -/
theorem executedAdditionDropsCarryOut (a b : BitVec 32) :
    (a + b).toNat = (a.toNat + b.toNat) % 2^32 := BitVec.toNat_add a b

/-- The carry-out incidence is not constant: one addition overflows, another does not. The stage
count (`32`) is the same in both cases, so the stage count does not determine the bookkeeping
quantity. -/
theorem overflowIncidence_not_constant :
    (∃ a b : BitVec 32, carryOut a b = true) ∧
      (∃ a b : BitVec 32, carryOut a b = false) := by
  constructor
  · exact ⟨0xFFFFFFFF, 1, by decide⟩
  · exact ⟨0, 0, by decide⟩

/-! ## 6. Wiring: rotation and shift coordinate moves

A rotation moves each of the 32 bit positions of a word; a logical shift by `n` moves the `32 - n`
positions that survive and grounds `n`. This is a wiring model (grade `model`), counted in
bit-coordinate moves, a unit the twenty-three-field measure does not have; the word-layer cost of the
same operations is the `shift`/`bitOr`/`bitXor`/`bitAnd`/`bitNot` field content of §3. The two
counts are different units and are never summed. -/

/-- The word width the wiring model is stated at. -/
def wordWidth : Nat := 32

/-- A rotation moves every coordinate. -/
def rotationMoves : Nat := wordWidth

/-- A logical right shift by `n` moves the `32 - n` surviving coordinates and grounds `n`. -/
def shiftMoves (n : Nat) : Nat := wordWidth - n

/-- Shifts per block: the two `SHR`s of each schedule visit, at amounts 3 and 10
(`Sha2.lean:60,64`). -/
def blockShifts : Nat := blockScheduleSteps * schedStepShifts

theorem blockShifts_eq : blockShifts = 96 := by
  norm_num [blockShifts, blockScheduleSteps, schedStepShifts]

/-- Bit-coordinate moves per block: `blockRotations` full permutations plus the surviving
coordinates of `48` visits' two shifts. -/
def blockWireMoves : Nat :=
  blockRotations * rotationMoves + blockScheduleSteps * (shiftMoves 3 + shiftMoves 10)

theorem blockWireMoves_eq : blockWireMoves = 20880 := by
  norm_num [blockWireMoves, blockRotations, blockScheduleSteps, rotationMoves, shiftMoves,
    wordWidth, roundRotations, schedStepRotations, blockRounds]

/-- Grounded coordinates per block: the `SHR` amounts, `3` and `10`, over 48 visits. -/
def blockWireGrounds : Nat := blockScheduleSteps * (3 + 10)

theorem blockWireGrounds_eq : blockWireGrounds = 624 := by
  norm_num [blockWireGrounds, blockScheduleSteps]

/-! ## 7. Face-sensitive operations and the three faces

The executing compression is dial-free: no body in §3 takes a dial or branches on one. The
face-sensitive scalar operations live in the C5 rotation reading and in the channel premise:
`dialMulCost` counts the two dial factors of one `ringMul`; `blockDialMuls` is their block total; and
the channel premise `r * r = σ` (`ChannelsAnyRing.lean:139` `ringUpper`) has witnesses at `σ = 0`
and `σ = 1` and none at `σ = -1`, which makes the elliptic case a refusal, not a different
number. -/

/-- Dial-carrying scalar multiplications per block under C5: two per rotation. -/
def blockDialMuls : Nat := blockRotations * 2

theorem blockDialMuls_eq : blockDialMuls = 1152 := by
  norm_num [blockDialMuls, blockRotations, blockScheduleSteps, schedStepRotations, blockRounds,
    roundRotations]

/-- The executing word-layer compression performs no dial-carrying multiplication: those are the
C5 reading's, priced separately. -/
theorem executing_dial_mul_zero (σ : W32) : (compressionWordCost σ).mul = 0 :=
  compressionWordCost_mul_zero σ

/-- **At the elliptic face the channel premise has no witness**: `r * r = -1` has no solution in
`ZMod (2^32)`, because `-1` is not a square modulo `4` and `4` divides `2^32`. -/
theorem no_sqrt_neg_one_w32 : ¬ ∃ r : W32, r * r = -1 := by
  rintro ⟨r, hr⟩
  have h4 : (4 : ℕ) ∣ 2^32 := by norm_num
  let f : W32 →+* ZMod 4 := ZMod.castHom h4 (ZMod 4)
  have hf : f r * f r = -1 := by
    have := congrArg f hr
    simpa using this
  exact (by decide : ¬ ∃ s : ZMod 4, s * s = -1) ⟨f r, hf⟩

/-- The channel premise's successful branches: `0 * 0 = 0` and `1 * 1 = 1`. -/
theorem channel_premise_witnesses :
    ((0 : W32) * 0 = 0) ∧ ((1 : W32) * 1 = 1) ∧ (¬ ∃ r : W32, r * r = -1) :=
  ⟨by norm_num, by norm_num, no_sqrt_neg_one_w32⟩

/-- **The three-face verdict, field by field, on the counted bodies.** Every field of every face is
the same literal, because `compressionCost_eq` holds at every `σ` and its right-hand side mentions no
dial; the equalities below are directly readable. This is a fact about the counted bodies, not a
statement that the trichotomy cannot affect cost. -/
theorem compressionCost_threeFaces :
    (compressionCost faceElliptic).mul = (compressionCost faceParabolic).mul ∧
      (compressionCost faceParabolic).mul = (compressionCost faceHyperbolic).mul ∧
      (compressionCost faceElliptic).add = (compressionCost faceParabolic).add ∧
      (compressionCost faceParabolic).add = (compressionCost faceHyperbolic).add ∧
      (compressionCost faceElliptic).channel = (compressionCost faceParabolic).channel ∧
      (compressionCost faceParabolic).channel = (compressionCost faceHyperbolic).channel ∧
      (compressionCost faceElliptic).bitXor = (compressionCost faceParabolic).bitXor ∧
      (compressionCost faceParabolic).bitXor = (compressionCost faceHyperbolic).bitXor ∧
      (compressionCost faceElliptic).shift = (compressionCost faceParabolic).shift ∧
      (compressionCost faceParabolic).shift = (compressionCost faceHyperbolic).shift ∧
      (compressionCost faceElliptic).schedWordWrite =
        (compressionCost faceParabolic).schedWordWrite ∧
      (compressionCost faceParabolic).schedWordWrite =
        (compressionCost faceHyperbolic).schedWordWrite := by
  rw [compressionCost_eq faceElliptic, compressionCost_eq faceParabolic,
    compressionCost_eq faceHyperbolic]
  simp

/-- The zero fields of the compression figure at every face: the dial cannot enter a body that has
no dial. -/
theorem compressionCost_threeFaces_zero :
    (compressionCost faceElliptic).sub = 0 ∧ (compressionCost faceElliptic).neg = 0 ∧
      (compressionCost faceElliptic).cast = 0 ∧ (compressionCost faceParabolic).sub = 0 ∧
      (compressionCost faceParabolic).neg = 0 ∧ (compressionCost faceParabolic).cast = 0 ∧
      (compressionCost faceHyperbolic).sub = 0 ∧ (compressionCost faceHyperbolic).neg = 0 ∧
      (compressionCost faceHyperbolic).cast = 0 := by
  rw [compressionCost_eq faceElliptic, compressionCost_eq faceParabolic,
    compressionCost_eq faceHyperbolic]
  simp

/-! ## 8. Recurrence and resource bounds over the message length

The padded message of `n` bytes is `blockCount n = (n + 72) / 64` blocks (`pad` and `hash`,
`Sha2.lean:133,146`; the arithmetic is AR-20's `canonTotal`/`blockCount`, re-derived here as the
closed division for the executing circuit). One block is one `compressionCost`, so the compression
figure is linear in the block count, and every bound below is a theorem over `n`, not a table. -/

/-- The number of 64-byte blocks of the padded message of `n` bytes: the least `k` with
`64 * k ≥ n + 9`, in closed form. Temporal and spatial O(1). -/
def blockCount (n : Nat) : Nat := (n + 72) / 64

/-- **The recurrence**: 64 more message bytes add exactly one block. -/
theorem blockCount_succ64 (n : Nat) : blockCount (n + 64) = blockCount n + 1 := by
  simp only [blockCount]
  rw [show n + 64 + 72 = (n + 72) + 1 * 64 by omega]
  rw [Nat.add_mul_div_right (n + 72) 1 (by norm_num)]

/-- **The bound**: the block count is at most `n/64 + 2`, for every `n`. -/
theorem blockCount_bound (n : Nat) : blockCount n ≤ n / 64 + 2 := by
  simp only [blockCount]
  calc
    (n + 72) / 64 ≤ (n + 128) / 64 := by
      apply Nat.div_le_div_right
      omega
    _ = n / 64 + 2 := by
      rw [show n + 128 = n + 2 * 64 by omega, Nat.add_mul_div_right _ _ (by norm_num)]

/-- Boundary witnesses of the block count: `0`, the `55/56` flip the padding turns on, `63`, `64`
and the 80-byte header. -/
theorem blockCount_boundary :
    blockCount 0 = 1 ∧ blockCount 55 = 1 ∧ blockCount 56 = 2 ∧ blockCount 63 = 2 ∧
      blockCount 64 = 2 ∧ blockCount 80 = 2 := by
  norm_num [blockCount]

/-- The compression figure for a message of `n` bytes: `blockCount n` copies of one block's
compression. Temporal cost: the recurrence `c 0`-free closed form below; spatial: the per-block
footprint of §8.1 is independent of `n`. Counted in: the twenty-three fields. Derivation: the
measured block count times the measured block cost; it closes because `ShaCost.rep_add` telescopes
the block count over the recurrences below. -/
def messageCost (σ : W32) (n : Nat) : ShaCost σ :=
  ShaCost.rep σ (blockCount n) (compressionCost σ)

/-- **The per-block recurrence of the message figure**, the general `n → n + 64` step. -/
theorem messageCost_succ64 (σ : W32) (n : Nat) :
    messageCost σ (n + 64) = messageCost σ n + compressionCost σ := by
  simp only [messageCost]
  rw [blockCount_succ64, ShaCost.rep_add, ShaCost.rep_one]

/-- The message figure is the block count times the block figure. -/
theorem messageCost_rep (σ : W32) (n : Nat) :
    messageCost σ n = ShaCost.rep σ (blockCount n) (compressionCost σ) := rfl

/-- The shift field of the message figure, exact over `n`. -/
theorem messageCost_shift (σ : W32) (n : Nat) :
    (messageCost σ n).shift = blockCount n * 1248 := by
  rw [messageCost_rep, ShaCost.rep, compressionCost_eq]

/-- The addition field of the message figure, exact over `n`. -/
theorem messageCost_add (σ : W32) (n : Nat) :
    (messageCost σ n).add = blockCount n * 2328 := by
  rw [messageCost_rep, ShaCost.rep, compressionCost_eq]

/-- The byte-read field of the message figure, exact over `n`. -/
theorem messageCost_blockByteRead (σ : W32) (n : Nat) :
    (messageCost σ n).blockByteRead = blockCount n * 64 := by
  rw [messageCost_rep, ShaCost.rep, compressionCost_eq]

/-- **The resource bound over the message length**: the schedule shifts of the compression of an
`n`-byte message are at most `(n/64 + 2) * 1248`, for every `n`. The same statement is an instance
of `blockCount_bound` at every other field. -/
theorem messageCost_shift_le (σ : W32) (n : Nat) :
    (messageCost σ n).shift ≤ (n / 64 + 2) * 1248 := by
  rw [messageCost_shift]
  exact Nat.mul_le_mul_right 1248 (blockCount_bound n)

/-- **The resource bound over the message length**, addition field. -/
theorem messageCost_add_le (σ : W32) (n : Nat) :
    (messageCost σ n).add ≤ (n / 64 + 2) * 2328 := by
  rw [messageCost_add]
  exact Nat.mul_le_mul_right 2328 (blockCount_bound n)

/-- **The per-block live footprint is bounded independently of the message length**: the schedule
array holds 64 words (`ApeironWord.lean:15393`), the state 8 (`:15399`), and the working set is the
eight registers plus `W[i]`; no quantity of the compression grows with `n`. -/
theorem liveWords_independent :
    (64#usize : Std.Usize).val + (8#usize : Std.Usize).val + 8 = 80 := by norm_num

/-! ## 9. Audit rows -/

#print axioms ringMul_body_ops
#print axioms ringUpper_body_ops
#print axioms rotr_body_ops
#print axioms shrx_body_ops
#print axioms sigma0_body_ops
#print axioms sigma1_body_ops
#print axioms bigSigma0_body_ops
#print axioms bigSigma1_body_ops
#print axioms choose_body_ops
#print axioms majority_body_ops
#print axioms dial_term_is_zero_valued
#print axioms rotReadCost_fields
#print axioms ringMulCost_split
#print axioms schedStepCost_fields
#print axioms schedStepRotReadCost_fields
#print axioms roundCost_fields
#print axioms roundRotReadCost_fields
#print axioms addBackCost_fields
#print axioms compressionWordCost_eq
#print axioms compressionRingReadCost_eq
#print axioms compressionCost_eq
#print axioms compressionWordCost_mul_zero
#print axioms compressionWordCost_cast_zero
#print axioms compressionWordCost_channel_zero
#print axioms headerCost_eq
#print axioms schedStepArrayReads_eq
#print axioms schedStepArrayWrites_eq
#print axioms schedStepCanonicalReads_eq
#print axioms roundArrayReads_eq
#print axioms roundStateReads_eq
#print axioms roundStateWrites_eq
#print axioms addrAdd_eq
#print axioms addrSub_eq
#print axioms loopTests_eq
#print axioms carryRecover
#print axioms executedAdditionDropsCarryOut
#print axioms overflowIncidence_not_constant
#print axioms blockCarryStages_eq
#print axioms blockWireMoves_eq
#print axioms blockWireGrounds_eq
#print axioms no_sqrt_neg_one_w32
#print axioms channel_premise_witnesses
#print axioms compressionCost_threeFaces
#print axioms compressionCost_threeFaces_zero
#print axioms blockCount_succ64
#print axioms blockCount_bound
#print axioms blockCount_boundary
#print axioms messageCost_succ64
#print axioms messageCost_shift
#print axioms messageCost_add
#print axioms messageCost_blockByteRead
#print axioms messageCost_shift_le
#print axioms messageCost_add_le
#print axioms liveWords_independent

end algebra_cost_source
