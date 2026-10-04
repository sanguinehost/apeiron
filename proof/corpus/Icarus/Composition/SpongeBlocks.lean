/-
  APEIRON-CRYPTO-SPIKE-sponge-blocks

  Correctness of the extracted message-block loop `crypto.sponge_loop0`
  (`proof/ApeironWord.lean`, extracted from `src/crypto.rs:162-166`), against the
  canonical FIPS-202 §4 absorb fold of `proof/corpus/Crypto/Sponge.lean`.

  Scope, and nothing wider:

  * `crypto.absorb_block` (one `rate`-byte block, `src/crypto.rs:126-144`) enters through
    the committed lane-loop contract
    `proof/corpus/Crypto/Keccak/SpongeCertificate.lean:727` `absorb_loop_spec`; the three
    `usize` steps and the `massert` that the extracted `crypto.absorb_block` itself performs
    are discharged here (`absorb_block_spec`), so the caller sees the whole one-block
    absorption;
  * `crypto.keccak_f1600` enters through the committed permutation contract
    `proof/corpus/Crypto/Keccak/Certificate.lean:1625` `keccak_f1600_loop0_spec`, restated in
    the sponge certificate's `stateView` (`keccak_f1600_spec`);
  * the loop proved here is only `crypto.sponge_loop0`, the *full-block* absorb loop of
    `crypto.sponge` (`src/crypto.rs:161-167`).  The padding block
    (`src/crypto.rs:169-179`), the squeeze (`sponge_loop1`/`sponge_loop2`,
    `src/crypto.rs:181-202`) and the `crypto.sha3_256` headline (`src/crypto.rs:206-214`) are
    NOT proved here and are NOT claimed: they are the padding / squeeze / composition owners'
    stages.

  What is proved: for a message of `n = |msg| / rate` full blocks whose length leaves room for
  one more rate block of counter arithmetic (`|msg| + rate ≤ Usize.max`, the extracted
  overflow-checked `usize` addition of `src/crypto.rs:162` being total on that domain), a
  successful run of `crypto.sponge_loop0` from *any* starting state `state0` returns a counter
  `consumed` with `consumed.val = rate * n`, and a state whose canonical `State` value is the
  canonical `permute`/`xorBlock` fold over exactly those `n` real full blocks of the real message
  bytes, starting from `stateView state0`.  The two committed stage contracts above are
  hypotheses of the general theorem `sponge_loop0_spec` and are discharged at the SHA3-256
  instance by closed numeral facts about the rate; `sponge_loop0_sha3_256` is the instance at
  rate 136 from the `Array.repeat 25#usize 0#u64` that `crypto.sponge` itself allocates
  (`src/crypto.rs:158`), and is the successful-branch witness for that `Result`.

  What is NOT claimed:

  * no unconditional whole-message refinement of `crypto.sha3_256`: the top-level
    `crypto.sponge` also pads and squeezes and neither is proved here.  The claim is the block
    loop's own contract;
  * no relation to `Apeiron.Crypto.Keccak.absorb` (`Crypto/Sponge.lean:85`), which needs the
    canonical `blocks`/`pad` unpacking and the padding tail (other owners);
  * no side-channel, constant-time, streaming or memory-safety claim.  The loop's own
    `Result` branches are characterised only where a branch witness is given: the successful
    branch above, and the refusing branch `msgBlock_body_exit`, which declines to absorb a
    short block.

  `maxHeartbeats` is a finite bound (not `0`), used on the three theorems that reduce the
  extracted `do` chains; it is not raised anywhere else in the file.
-/
import Crypto.Keccak.SpongeCertificate
import Crypto.Keccak.Certificate

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result
open Apeiron.Crypto.Keccak (State xorBlock permute)

/-! ## 0. `view` (permutation certificate) is `stateView` (sponge certificate)

`Crypto/Keccak/Certificate.lean` reads a state through `view` (with `getElem`);
`Crypto/Keccak/SpongeCertificate.lean` reads it through `stateView` (with `getD`).  Both are
`Vector.ofFn` over the same 25 lanes, so they are equal and the committed permutation contract
restates in `stateView`.  Copied from
`work/proofs/archive/SpongeComposition.green-increments.lean` (read-only, not imported) into
this file's private declarations; that scratch file is not a dependency. -/

private theorem view_getD (a : Array Std.U64 25#usize) (i : Fin 25) :
    (view a).get i = (a.val.getD i.val (0#u64)).bv := by
  show (view a)[i.val] = _
  simp only [view, Vector.getElem_ofFn]
  rw [Array.getElem_Nat_eq]
  · rw [List.getElem_eq_getD (0#u64)]
  · show (i.val : Nat) < a.val.length
    have : a.val.length = 25 := by simp [a.property]
    omega

private theorem view_eq_stateView (a : Array Std.U64 25#usize) : view a = stateView a := by
  apply Vector.ext
  intro i hi
  show (view a).get ⟨i, hi⟩ = (stateView a).get ⟨i, hi⟩
  exact (view_getD a ⟨i, hi⟩).trans (stateView_get a ⟨i, hi⟩).symm

/-- The committed permutation contract of stage 3, restated in the sponge certificate's
`stateView`: `crypto.keccak_f1600 a` returns an array whose canonical state value is the
canonical `permute` of `a`'s.  No hypothesis beyond the type of `a`. -/
theorem keccak_f1600_spec (a : Array Std.U64 25#usize) :
    crypto.keccak_f1600 a ⦃ fun b => stateView b = permute (stateView a) ⦄ := by
  obtain ⟨b, hb, hv⟩ := WP.spec_imp_exists (keccak_f1600_loop0_spec a)
  refine WP.exists_imp_spec ⟨b, ?_, ?_⟩
  · show crypto.keccak_f1600 a = ok b
    exact hb
  · rw [← view_eq_stateView a, ← view_eq_stateView b, hv]

/-! ## 1. `crypto.absorb_block`: one whole rate block

The committed lane-loop contract `absorb_loop_spec` is stated for
`crypto.absorb_block_loop`.  `crypto.absorb_block` additionally evaluates
`Slice.len block % 8#usize`, asserts it is zero, and evaluates `Slice.len block / 8#usize` as
the lane count.  This section discharges those three steps. -/

set_option maxHeartbeats 400000 in
/-- On a block of exactly `rate` bytes, with `rate % 8 = 0` (the `massert` precondition of
`src/crypto.rs:127`) and `rate / 8 ≤ 25` (the lanes fit the 25-lane state),
`crypto.absorb_block state block` returns the canonical `xorBlock rate` of the start state with
those block bytes, read through `stateView`. -/
theorem absorb_block_spec (state : Array Std.U64 25#usize) (block : Slice Std.U8)
    (rate : Nat) (hrate8 : rate % 8 = 0) (hr25 : rate / 8 ≤ 25)
    (hlen : (sliceView block).length = rate) :
    ∃ s1 : Array Std.U64 25#usize,
      crypto.absorb_block state block = ok s1 ∧
        stateView s1 = xorBlock rate (stateView state) (blockVec rate (sliceView block)) := by
  have hblen : block.val.length = rate := by rw [← sliceView_length]; exact hlen
  -- `Slice.len block % 8#usize`, the `massert` operand of `src/crypto.rs:127`.
  obtain ⟨left_val, hleft, hmod⟩ :=
    WP.spec_imp_exists (Usize.rem_spec (Slice.len block) (y := 8#usize) (by decide))
  have hmod : left_val.val = block.val.length % 8 := hmod
  have hzero : left_val = 0#usize :=
    UScalar.eq_of_val_eq (by rw [hmod, hblen, hrate8]; rfl)
  -- `Slice.len block / 8#usize`, the lane count of `src/crypto.rs:128`.
  obtain ⟨lanes, hlanes, hq⟩ :=
    WP.spec_imp_exists (Usize.div_spec (Slice.len block) (y := 8#usize) (by decide))
  have hq : lanes.val = block.val.length / 8 := hq
  have hlanes' : lanes.val = rate / 8 := by rw [hq, hblen]
  obtain ⟨s1, hs1, hpost⟩ :=
    WP.spec_imp_exists (absorb_loop_spec state block rate lanes hrate8 hlanes' hr25 hlen)
  refine ⟨s1, ?_, ?_⟩
  · unfold crypto.absorb_block
    dsimp only
    rw [hleft]
    simp only [bind_tc_ok]
    simp only [massert, if_pos hzero, bind_tc_ok]
    rw [hlanes]
    simp only [bind_tc_ok]
    exact hs1
  · unfold absorbPost at hpost
    apply Vector.ext
    intro k hk
    show (stateView s1)[k] = (xorBlock rate (stateView state) (blockVec rate (sliceView block)))[k]
    exact hpost ⟨k, hk⟩

/-! ## 2. The canonical fold over the real full blocks of a message -/

/-- Byte `k` of `rate`-byte block `n` of a byte list, zero past the end of the list. -/
def blockByte (rate n k : Nat) (l : List (BitVec 8)) : BitVec 8 := l.getD (rate * n + k) 0

/-- `rate`-byte block `n` of a byte list, in the `Vector.ofFn` shape `xorBlock` reads.  Temporal
cost `rate` out-of-bounds-zero list reads and one vector construction; spatial cost the
returned `rate`-element vector.  Both immediate from the `Vector.ofFn` body. -/
def blockVecN (rate n : Nat) (l : List (BitVec 8)) : Vector (BitVec 8) rate :=
  Vector.ofFn fun i : Fin rate => blockByte rate n i.val l

/-- The FIPS-202 §4 absorb fold over the `n` successive `rate`-byte blocks of `l` starting at
block index `b`: `permute (xorBlock ·)` applied to blocks `b, b + 1, …, b + n - 1` of `l` in
that order, from state `s`.  Temporal cost `n` calls of `permute` (each 24 rounds) plus `n ·
rate` block-byte reads; spatial cost `O(1)` extra words, since the recursion allocates nothing
and the count decreases by one per step.  Both read off the definition. -/
def absorbBlocks (rate b : Nat) (l : List (BitVec 8)) : Nat → State → State
  | 0, s => s
  | k + 1, s => permute (xorBlock rate (absorbBlocks rate b l k s) (blockVecN rate (b + k) l))

/-- The fold's one-block equation: `n + 1` blocks absorb block `b + n` after the first `n`. -/
theorem absorbBlocks_succ (rate b n : Nat) (l : List (BitVec 8)) (s : State) :
    absorbBlocks rate b l (n + 1) s
      = permute (xorBlock rate (absorbBlocks rate b l n s) (blockVecN rate (b + n) l)) := by
  rw [absorbBlocks]

/-- From base block index `0`, the `n + 1`-block fold is block `n` applied after the first `n`
blocks: `0 + n = n`. -/
theorem absorbBlocks_succ_zero (rate n : Nat) (l : List (BitVec 8)) (s : State) :
    absorbBlocks rate 0 l (n + 1) s
      = permute (xorBlock rate (absorbBlocks rate 0 l n s) (blockVecN rate n l)) := by
  rw [absorbBlocks_succ, Nat.zero_add]

/-! ## 3. Slice-offset arithmetic -/

private theorem bv_getD (l : List Std.U8) (k : Nat) :
    (l.map Std.U8.bv).getD k 0 = (l.getD k (0#u8)).bv := by
  have hz : Std.U8.bv (0#u8) = 0 := by decide
  rw [← hz, List.getD_map]

private theorem getD_eq_getElem' (l : List Std.U8) (k : Nat) (d : Std.U8) (h : k < l.length) :
    l.getD k d = l[k] :=
  List.getD_eq_getElem (l := l) (d := d) h

private theorem mulDivSelf (a b : Nat) (h : 0 < b) : (b * a) / b = a := by
  rw [Nat.mul_comm]
  exact Nat.mul_div_cancel a h

/-- The extracted `usize` comparison against the extracted `Slice::len` is the canonical
message-length comparison. -/
private theorem usize_le_slice_len_iff (b : Std.Usize) (msg : Slice Std.U8) :
    b ≤ Slice.len msg ↔ b.val ≤ (sliceView msg).length := by
  have hlen : (Slice.len msg).val = msg.val.length := Slice.len_val msg
  constructor
  · intro h
    have h2 : b.val ≤ (Slice.len msg).val := (UScalar.le_equiv b (Slice.len msg)).mp h
    rw [hlen] at h2
    have h3 : b.val ≤ (sliceView msg).length := by
      rw [sliceView_length]
      exact h2
    exact h3
  · intro h
    have h2 : b.val ≤ (sliceView msg).length := h
    rw [sliceView_length] at h2
    exact (UScalar.le_equiv b (Slice.len msg)).mpr (by omega)

/-- Byte `k` of the window `s = msg[consumed .. b]` is byte `consumed.val + k` of the message,
whenever `k < |s|` and `b ≤ |msg|`.  This is the real-offset reading of the extracted range index
against the canonical block list. -/
theorem slice_window_getD (msg s : Slice Std.U8) (consumed b : Std.Usize) (k : Nat)
    (hsv : s.val = msg.val.slice consumed.val b.val) (hklen : k < s.val.length)
    (hb : b ≤ Slice.len msg) (hkc : consumed.val + k < b.val) :
    (sliceView s).getD k 0 = (sliceView msg).getD (consumed.val + k) 0 := by
  have hlen : (Slice.len msg).val = msg.val.length := Slice.len_val msg
  have hb'' : b.val ≤ (sliceView msg).length := (usize_le_slice_len_iff b msg).mp hb
  rw [sliceView_length] at hb''
  have hb' : b.val ≤ msg.val.length := by omega
  have hklen' : k < (msg.val.slice consumed.val b.val).length := by rw [← hsv]; exact hklen
  simp only [sliceView, bv_getD]
  have hlist : s.val.getD k (0#u8) = msg.val.getD (consumed.val + k) (0#u8) := by
    rw [hsv, getD_eq_getElem' _ _ _ hklen']
    rw [List.getElem_slice consumed.val b.val k msg.val ⟨hb', hkc⟩]
    rw [getD_eq_getElem' _ _ _ (by omega)]
  rw [hlist]

/-- With `consumed.val = rate * n` and `b.val = consumed.val + rate`, the `rate`-byte window
`msg[consumed .. b]` that `crypto.sponge_loop0.body` reads is exactly block `n` of the message:
`blockVec rate (sliceView s) = blockVecN rate n (sliceView msg)`.  Pointwise in `k < rate`,
because `xorBlock` reads each lane through `getD`. -/
theorem blockVec_window (rate : Nat) (msg s : Slice Std.U8) (consumed b : Std.Usize) (n : Nat)
    (hcons : consumed.val = rate * n) (hbval : b.val = consumed.val + rate)
    (hsv : s.val = msg.val.slice consumed.val b.val) (hslen : s.val.length = rate)
    (hb : b ≤ Slice.len msg) :
    blockVec rate (sliceView s) = blockVecN rate n (sliceView msg) := by
  apply Vector.ext
  intro k hk
  show (blockVec rate (sliceView s))[k] = (blockVecN rate n (sliceView msg))[k]
  simp only [blockVec, blockVecN, blockByte, Vector.getElem_ofFn]
  have hklen : k < s.val.length := by omega
  have hkc : consumed.val + k < b.val := by omega
  rw [slice_window_getD msg s consumed b (k := k) hsv hklen hb hkc]
  rw [hcons]

/-! ## 4. The loop invariant

The counter of `crypto.sponge_loop0` is always a whole number of rate blocks, so the invariant
carries that divisibility: without it `consumed.val / rate` need not be the block index. -/

/-- Invariant of the extracted message-block loop at counter `consumed`: the counter is a whole
number of `rate`-byte blocks at or below the message length, and the state is the canonical
FIPS-202 §4 absorb fold of exactly the `consumed.val / rate` full blocks the loop has read,
starting from the loop's input state. -/
def msgBlockInv (msg : Slice Std.U8) (rate : Nat) (state0 state : Array Std.U64 25#usize)
    (consumed : Std.Usize) : Prop :=
  consumed.val = rate * (consumed.val / rate) ∧
  consumed.val ≤ (sliceView msg).length ∧
  stateView state = absorbBlocks rate 0 (sliceView msg) (consumed.val / rate) (stateView state0)

set_option maxHeartbeats 1500000 in
/-- One iteration of `crypto.sponge_loop0` that still has a whole rate block left: it reads
`msg[consumed .. consumed + rate]`, absorbs it as one canonical `xorBlock`, permutes, and
advances the counter by `rate`, preserving `msgBlockInv`. -/
theorem msgBlock_body_step (msg : Slice Std.U8) (rate : Std.Usize) (rateN : Nat)
    (state0 state : Array Std.U64 25#usize) (consumed : Std.Usize)
    (hrateval : rate.val = rateN) (hratepos : 0 < rateN) (hrate8 : rateN % 8 = 0)
    (hr25 : rateN / 8 ≤ 25) (hle : consumed.val + rateN ≤ (sliceView msg).length)
    (hinv : msgBlockInv msg rateN state0 state consumed) :
    ∃ (state1 : Array Std.U64 25#usize) (b : Std.Usize),
      crypto.sponge_loop0.body msg rate state consumed = ok (ControlFlow.cont (state1, b)) ∧
        b.val = consumed.val + rateN ∧ msgBlockInv msg rateN state0 state1 b := by
  obtain ⟨hcons, _, hstate⟩ := hinv
  have hmaxlen : (sliceView msg).length ≤ Usize.max := by
    rw [sliceView_length]
    exact Slice.length_ineq msg
  -- `consumed + rate`: the window end and the next counter.  `hle` puts it in range.
  obtain ⟨b, hb, hbval0⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := consumed) (y := rate) (by omega))
  have hbval : b.val = consumed.val + rateN := by rw [hbval0, hrateval]
  -- The loop guard `consumed + rate <= msg.len`.
  have hguard : b ≤ Slice.len msg := (usize_le_slice_len_iff b msg).mpr (by omega)
  -- The range index `msg[consumed .. b]`.
  have hce : consumed ≤ b := (UScalar.le_equiv consumed b).mpr (by omega)
  have hidx : core.slice.index.Slice.index (core.slice.index.SliceIndexRangeUsizeSlice Std.U8)
        msg { start := consumed, «end» := b }
      ⦃ s => s.val = msg.val.slice consumed.val b.val ∧ s.val.length = b.val - consumed.val ⦄ := by
    rw [Slice.index_SliceIndexRangeUsizeSliceInst]
    exact core.slice.index.SliceIndexRangeUsizeSlice.index.step_spec
      { start := consumed, «end» := b } msg hce hguard
  obtain ⟨s, hs, hsval, hslen0⟩ := WP.spec_imp_exists hidx
  have hslen : s.val.length = rateN := by rw [hslen0, hbval]; omega
  -- The two committed stage contracts, at this block.
  obtain ⟨a1, ha1, hst1⟩ :=
    absorb_block_spec state s rateN hrate8 hr25 (by rw [sliceView_length]; exact hslen)
  obtain ⟨a2, ha2, hst2⟩ := WP.spec_imp_exists (keccak_f1600_spec a1)
  have hbdiv : b.val / rateN = consumed.val / rateN + 1 := by
    have hbform : b.val = rateN * (consumed.val / rateN) + rateN := by omega
    rw [hbform, Nat.add_div_right _ hratepos, mulDivSelf _ _ hratepos]
  refine ⟨a2, b, ?_, hbval, ?_⟩
  · unfold crypto.sponge_loop0.body
    rw [hb]
    simp only [bind_tc_ok]
    rw [if_pos hguard]
    rw [hs]
    simp only [bind_tc_ok]
    rw [ha1]
    simp only [bind_tc_ok]
    rw [ha2]
    simp only [bind_tc_ok]
  · refine ⟨?_, ?_, ?_⟩
    · have h1 : b.val = consumed.val + rateN := hbval
      have h2 : consumed.val + rateN = rateN * (consumed.val / rateN) + rateN := by omega
      have h3 : rateN * (consumed.val / rateN) + rateN
          = rateN * (consumed.val / rateN + 1) := by
        rw [← Nat.succ_eq_add_one, Nat.mul_succ]
      have h4 : rateN * (consumed.val / rateN + 1) = rateN * (b.val / rateN) :=
        congrArg (fun t => rateN * t) hbdiv.symm
      calc b.val = consumed.val + rateN := h1
        _ = rateN * (consumed.val / rateN) + rateN := h2
        _ = rateN * (consumed.val / rateN + 1) := h3
        _ = rateN * (b.val / rateN) := h4
    · have h1 : b.val = consumed.val + rateN := hbval
      omega
    · rw [hst2, hst1, hstate,
        blockVec_window rateN msg s consumed b (consumed.val / rateN) hcons hbval hsval hslen
          hguard,
        hbdiv, absorbBlocks_succ_zero]

/-- The refusing branch of one iteration: when the computed window end `b` of `consumed + rate`
fails the guard `b ≤ msg.len`, the loop body does not index the message, does not absorb, and
returns the current pair unchanged.  The window end `b` is the loop's own `consumed + rate`
evaluation, so no arithmetic about the counter is assumed here. -/
theorem msgBlock_body_exit (msg : Slice Std.U8) (rate : Std.Usize) (b : Std.Usize)
    (state : Array Std.U64 25#usize) (consumed : Std.Usize)
    (hb : consumed + rate = ok b) (hnot : ¬ (b ≤ Slice.len msg)) :
    crypto.sponge_loop0.body msg rate state consumed = ok (ControlFlow.done (state, consumed)) := by
  unfold crypto.sponge_loop0.body
  rw [hb]
  simp only [bind_tc_ok]
  rw [if_neg hnot]

/-! ## 5. The loop

Measure: the number of message bytes still unconsumed.  Each continuing iteration consumes
`rate > 0` bytes, so it strictly decreases.  The domain hypothesis `hwrap` makes the extracted
overflow-checked `usize` addition of `src/crypto.rs:162` total at every reachable counter. -/

/-- The postcondition of the message-block loop: the counter is a whole number of rate blocks at
or below the message length, equal to `rate * (|msg| / rate)`, and the state is the canonical
`permute`/`xorBlock` fold of exactly the `|msg| / rate` real full blocks, from `state0`. -/
def spongeLoop0Post (msg : Slice Std.U8) (rateN : Nat) (state0 : Array Std.U64 25#usize) :
    (Array Std.U64 25#usize × Std.Usize) → Prop :=
  fun r =>
    r.2.val ≤ (sliceView msg).length ∧
    r.2.val = rateN * ((sliceView msg).length / rateN) ∧
    stateView r.1
      = absorbBlocks rateN 0 (sliceView msg) ((sliceView msg).length / rateN) (stateView state0)

set_option maxHeartbeats 400000 in
/-- The extracted message-block loop, against the canonical absorb fold.  For every starting
state, every message whose length leaves one more rate block of counter arithmetic in range
(`|msg| + rate ≤ Usize.max`), and every rate that is a whole number of lanes no larger than the
25-lane state, a successful run of `crypto.sponge_loop0 msg rate state0 0` satisfies
`spongeLoop0Post`: it returns a counter that is a whole number of rate blocks at or below the
message length and equal to `rate * (|msg| / rate)`, together with a state whose canonical
`State` value is the canonical `permute`/`xorBlock` fold of exactly the `|msg| / rate` real full
blocks of the real message bytes, starting from `stateView state0`. -/
theorem sponge_loop0_spec (msg : Slice Std.U8) (rate : Std.Usize) (rateN : Nat)
    (state0 : Array Std.U64 25#usize)
    (hrateval : rate.val = rateN) (hratepos : 0 < rateN) (hrate8 : rateN % 8 = 0)
    (hr25 : rateN / 8 ≤ 25) (hwrap : (sliceView msg).length + rateN ≤ Usize.max) :
    crypto.sponge_loop0 msg rate state0 0#usize ⦃ spongeLoop0Post msg rateN state0 ⦄ := by
  unfold crypto.sponge_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array Std.U64 25#usize × Std.Usize => (sliceView msg).length - s.2.val)
    (inv := fun s => msgBlockInv msg rateN state0 s.1 s.2)
    (post := spongeLoop0Post msg rateN state0)
  · intro s hs
    rcases s with ⟨a1, c1⟩
    have hconsle : c1.val ≤ (sliceView msg).length := (hs : msgBlockInv msg rateN state0 a1 c1).2.1
    have hstate : stateView a1
        = absorbBlocks rateN 0 (sliceView msg) (c1.val / rateN) (stateView state0) :=
      (hs : msgBlockInv msg rateN state0 a1 c1).2.2
    have hcons' : c1.val = rateN * (c1.val / rateN) :=
      (hs : msgBlockInv msg rateN state0 a1 c1).1
    by_cases hfit : c1.val + rateN ≤ (sliceView msg).length
    · obtain ⟨a2, b, hbody, hbval, hinv2⟩ :=
        msgBlock_body_step msg rate rateN state0 a1 c1 hrateval hratepos hrate8 hr25 hfit hs
      simp only [show crypto.sponge_loop0.body msg rate a1 c1
          = ok (ControlFlow.cont (a2, b)) from hbody,
        Aeneas.Std.WP.spec_ok]
      refine ⟨hinv2, ?_⟩
      exact (by rw [hbval]; omega)
    · have hnoadd : c1.val + rateN ≤ Usize.max := by omega
      obtain ⟨b, hb, hbval0⟩ :=
        WP.spec_imp_exists (Usize.add_spec (x := c1) (y := rate) (by omega))
      have hbval : b.val = c1.val + rateN := by rw [hbval0, hrateval]
      have hnot : ¬ (b ≤ Slice.len msg) := by
        intro hc
        have hc' : b.val ≤ (sliceView msg).length := (usize_le_slice_len_iff b msg).mp hc
        rw [hbval] at hc'
        exact hfit hc'
      have hbody := msgBlock_body_exit msg rate b a1 c1 hb hnot
      simp only [show crypto.sponge_loop0.body msg rate a1 c1
          = ok (ControlFlow.done (a1, c1)) from hbody,
        Aeneas.Std.WP.spec_ok]
      have hlt : (sliceView msg).length < c1.val + rateN := by omega
      -- The loop stopped, so `c1.val / rate = |msg| / rate`: the consumed counter is a whole
      -- number of blocks and no further whole block fits.
      have hdiv : c1.val / rateN = (sliceView msg).length / rateN := by
        have hlen' : (sliceView msg).length
            = rateN * ((sliceView msg).length / rateN) + (sliceView msg).length % rateN :=
          (Nat.div_add_mod _ _).symm
        have hrem : (sliceView msg).length % rateN < rateN := Nat.mod_lt _ hratepos
        have hstep : rateN * ((sliceView msg).length / rateN + 1)
            = rateN * ((sliceView msg).length / rateN) + rateN := by
          rw [← Nat.succ_eq_add_one, Nat.mul_succ]
        have hstep' : rateN * (c1.val / rateN + 1)
            = rateN * (c1.val / rateN) + rateN := by
          rw [← Nat.succ_eq_add_one, Nat.mul_succ]
        have hlt' : rateN * ((sliceView msg).length / rateN) + (sliceView msg).length % rateN
            < rateN * (c1.val / rateN) + rateN := by
          rw [hlen', hcons'] at hlt
          exact hlt
        -- `n ≤ q`: `rate * n ≤ |msg| = rate * q + r < rate * q + rate`.
        have hnq : c1.val / rateN ≤ (sliceView msg).length / rateN := by
          have e1 : rateN * (c1.val / rateN)
              < rateN * ((sliceView msg).length / rateN + 1) := by omega
          have e2 : c1.val / rateN < (sliceView msg).length / rateN + 1 :=
            Nat.lt_of_mul_lt_mul_left e1
          omega
        -- `q ≤ n`: `rate * q ≤ |msg| < rate * n + rate`.
        have hqn : (sliceView msg).length / rateN ≤ c1.val / rateN := by
          have e1 : rateN * ((sliceView msg).length / rateN)
              < rateN * (c1.val / rateN + 1) := by omega
          have e2 : (sliceView msg).length / rateN < c1.val / rateN + 1 :=
            Nat.lt_of_mul_lt_mul_left e1
          omega
        exact le_antisymm hnq hqn
      dsimp only [spongeLoop0Post]
      refine ⟨hconsle, ?_, ?_⟩
      · calc c1.val = rateN * (c1.val / rateN) := hcons'
          _ = rateN * ((sliceView msg).length / rateN) := by rw [hdiv]
      · rw [hstate, hdiv]
  · show msgBlockInv msg rateN state0 state0 (0#usize)
    refine ⟨?_, ?_, ?_⟩ <;> simp [msgBlockInv, absorbBlocks]

/-! ## 6. The SHA3-256 instance

Every hypothesis of `sponge_loop0_spec` except the message-length arithmetic is discharged by a
closed numeral fact about the rate that `crypto.sha3_256` passes (`src/crypto.rs:207`). -/

theorem rate_136 : crypto.SHA3_256_RATE.val = 136 := by
  simp [crypto.SHA3_256_RATE, Apeiron.Crypto.Keccak.rate_sha3_256]

/-- The SHA3-256 message-block instance.  `crypto.sponge` starts its absorb loop from
`Array.repeat 25#usize 0#u64` (`src/crypto.rs:158`), whose `stateView` is the canonical zero state
(`stateView_zero`), so this is the successful-branch witness for the
`let (state1, consumed) ← crypto.sponge_loop0 …` binding at rate 136: the loop returns `ok` with
`consumed.val = 136 * (|msg| / 136)` and a state whose canonical value is the canonical
`permute`/`xorBlock` fold over exactly the `|msg| / 136` real full 136-byte blocks of the real
message.

The domain is every message leaving one more 136-byte block of counter arithmetic in range,
`|msg| + 136 ≤ Usize.max`; the excluded population is a message longer than `Usize.max - 136`
bytes, on which the extracted overflow-checked `usize` addition `consumed + rate` of
`src/crypto.rs:162` does not produce a counter and the loop body is not characterised here. -/
theorem sponge_loop0_sha3_256 (msg : Slice Std.U8) (hwrap : (sliceView msg).length + 136
    ≤ Usize.max) :
    ∃ (state1 : Array Std.U64 25#usize) (consumed : Std.Usize),
      crypto.sponge_loop0 msg crypto.SHA3_256_RATE (Array.repeat 25#usize 0#u64) 0#usize
        = ok (state1, consumed) ∧
      consumed.val = 136 * ((sliceView msg).length / 136) ∧
      stateView state1 = absorbBlocks 136 0 (sliceView msg) ((sliceView msg).length / 136)
        (Vector.replicate 25 0) := by
  obtain ⟨r, hr⟩ :=
    WP.spec_imp_exists (sponge_loop0_spec msg crypto.SHA3_256_RATE 136
      (Array.repeat 25#usize 0#u64) rate_136 (by decide) (by decide) (by decide) hwrap)
  dsimp only [spongeLoop0Post] at hr
  refine ⟨r.1, r.2, ?_, hr.2.2.1, ?_⟩
  · show crypto.sponge_loop0 msg crypto.SHA3_256_RATE (Array.repeat 25#usize 0#u64) 0#usize
      = ok (r.1, r.2)
    exact hr.1
  · rw [hr.2.2.2, stateView_zero]

#print axioms sponge_loop0_sha3_256
#print axioms sponge_loop0_spec
#print axioms msgBlock_body_step
#print axioms msgBlock_body_exit
#print axioms blockVec_window
#print axioms slice_window_getD
#print axioms absorb_block_spec

end apeiron_word