/-
  `Icarus/Extraction/Compression.lean` — **O9 (b), the sixty-four-round compression: total success
  and its result.**

  Owned files: this one and the task receipt. No corpus edit, no extraction refresh, no commit.
  Read-only upstream: `Crypto/Sha2.lean`, `Crypto/Sha256Certificate.lean`, and the generated
  `ApeironWord.lean`. The campaign staging source `work/proofs/Sha256Rounds.lean` lives outside
  this package and is cited, not imported.

  ## What this file links

  **Canonical.** `Apeiron.Crypto.Sha2.compress` (`Crypto/Sha2.lean:96-120`) folds
  `(List.range 64)` over its incoming state, each visit rewriting the eight working words with FIPS
  PUB 180-4 §6.2.2's `T₁ = h + Σ₁(e) + Ch(e,f,g) + K[t] + W[t]` and `T₂ = Σ₀(a) + Maj(a,b,c)`, then
  adds the eight working words back into the incoming state. That inner fold is lifted out as
  `roundStep` / `roundsN` / `rounds` below, with `compress_eq_rounds` binding it to the real
  definition by `rfl`.

  **Extracted.** `sha256.compress_loop2`, the generated loop of `src/sha256.rs`'s round loop over
  the working words `a … h` and the schedule array `w`.

  ## What is proved

  1. **§1 working-word views.** `wkState` / `wk8` / `wk9` read the extracted tuples as
     `Sha2.State`, with `wkState_getD` fixing the slot order `a b c d e f g h`.
  2. **§2 the extracted round body.** `roundBody_step`: one visit succeeds, advances the counter,
     and writes exactly the two FIPS round words — read through the corpus's `Sigma1_round`,
     `Sigma0_round`, `ch_step`, `maj_step`, `K_get` and `wView_getD`.
  3. **§3 the canonical round.** `roundStep`, `roundsN`, `rounds`, `rounds_succ`, and
     `compress_eq_rounds` — the latter is the `rfl` that binds every canonical row below to
     `Sha2.compress`.
  4. **§4 the extracted rotation is the canonical round.** `roundBody_vec`, for **every** schedule.
  5. **§5 the invariant.** `roundInv` with `roundInv_init`, `roundInv_step`, `roundInv_post`.
  6. **§6 total success.** `roundLoop_wp` → `roundLoop_spec`: for every schedule array and every
     eight initial words the loop **succeeds** and its working words are the canonical
     sixty-four-round fold. `roundLoop_eq` is the weaker success-conditioned corollary, kept and
     labelled as such.
  7. **§7 the canonical transport.** `roundLoop_canonical` under the schedule hypothesis `hw`, and
     a **witness** that the hypothesis is inhabited for every block (`schedule_premise_inhabited`,
     `wOfVec`, `wView_wOfVec`) — so it is not vacuous.
  8. **§8 the feed-forward link.** `addBack_eq`: the eight add-back writes of `Sha2.compress`'s
     final `Vector.ofFn` are the componentwise sum of the incoming state and the sixty-four round
     fold's working words. This is the add-back layer named in the register; the corpus carries no
     add-back theorem and none is imported here.

  ## Grades and non-claims

  `proved` for §§1–8 from the build gate recorded in the receipt, `#print axioms` rows inside
  `{propext, Classical.choice, Quot.sound}`, no `sorryAx`.

  **Total existence only, and conditional where stated.** `roundLoop_spec` is total over the
  *generic* schedule `wView w` — that is the vector the extracted schedule array denotes, and no
  claim is made here that the extracted pipeline's loading loop produces it from a byte block (O8's
  remaining arc, named in the receipt). `roundLoop_canonical` carries `hw` as a hypothesis and is
  labelled `conditional`; its non-vacuity is the witness in §7. The composed headline
  `word_sha256` is not asserted (O10). Nothing here bears on security, collision or preimage
  resistance, differential cryptanalysis, side channels or constant-time behaviour, and no timing
  is measured.
-/

import Crypto.Sha2
import Crypto.Sha256Certificate
import ApeironWord

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace Icarus
namespace Extraction
namespace Compression

open Apeiron.Crypto.Sha2 (Word State Block)
open Aeneas Aeneas.Std Result
open apeiron_word

/-! ## §1 · Working-word views

The extracted round loop carries its eight working words as eight separate `Std.U32` arguments,
not as a vector. `wkState` is the read into the canonical `Sha2.State`, in the order `a b c d e f g h`
that §6.2.2 step 3 names. -/

/-- The eight extracted working words of the round loop, read as the canonical
`Sha2.State`, in the order `a b c d e f g h`. `O(1)`. -/
def wkState (a b c d e f g h : Std.U32) : State :=
  Vector.ofFn fun i : Fin 8 =>
    (match i.val with
      | 0 => a | 1 => b | 2 => c | 3 => d | 4 => e | 5 => f | 6 => g | _ => h).bv

/-- Reading the working words through `getD` at a slot below eight. `proved`. -/
theorem wkState_getD (a b c d e f g h : Std.U32) (j : Nat) (hj : j < 8) :
    (wkState a b c d e f g h).getD j 0 =
      (match j with
        | 0 => a.bv | 1 => b.bv | 2 => c.bv | 3 => d.bv
        | 4 => e.bv | 5 => f.bv | 6 => g.bv | _ => h.bv) := by
  show (Vector.ofFn _).getD j 0 = _
  rw [ofFn_getD' _ j hj 0]
  have hc : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨ j = 5 ∨ j = 6 ∨ j = 7 := by
    omega
  rcases hc with h | h | h | h | h | h | h | h <;> subst h <;> rfl

/-- Reading the canonical round constants through `getD` at a row below sixty-four: the `getD`
spelling `Sha2.compress`'s fold uses, paired with the corpus's `K_get` read through `get`. `proved`. -/
theorem K_getD (j : Nat) (hj : j < 64) :
    Apeiron.Crypto.Sha2.K.getD j 0 = Apeiron.Crypto.Sha2.K.get ⟨j, hj⟩ := by
  have h : j < Apeiron.Crypto.Sha2.K.toArray.size := by
    show j < 64
    exact hj
  show Apeiron.Crypto.Sha2.K.toArray.getD j 0 = Apeiron.Crypto.Sha2.K.toArray[j]'h
  rw [← Array.getElem_eq_getD (fallback := (0 : Word)) (i := j) (h := h)]

/-! ## §2 · The extracted round body

`sha256.compress_loop2.body` is one FIPS PUB 180-4 §6.2.2 round visit, from `src/sha256.rs`'s round
loop. -/

/-- **`sha256.compress_loop2.body` is one FIPS 180-4 §6.2.2 round visit.** With `i.val < 64` the
body returns `ok (cont (a₁, a, b, c, e₁, e, f, g, i'))` with `i'.val = i.val + 1`, where `T₁` and
`T₂` are the two round words and `a₁ = T₁ + T₂`, `e₁ = d + T₁` are the two recomputed working words.
`proved`. -/
theorem roundBody_step (w : Array Std.U32 64#usize) (a b c d e f g h : Std.U32)
    (i : Std.Usize) (hlt : i.val < 64) :
    ∃ a₁ e₁ t1 t2 : Std.U32, ∃ i' : Std.Usize,
      sha256.compress_loop2.body w a b c d e f g h i
        = ok (ControlFlow.cont (a₁, a, b, c, e₁, e, f, g, i')) ∧
      i'.val = i.val + 1 ∧
      a₁.bv = t1.bv + t2.bv ∧
      e₁.bv = d.bv + t1.bv ∧
      t1.bv = h.bv + Apeiron.Crypto.Sha2.Sigma1 e.bv
        + Apeiron.Crypto.Sha2.choose e.bv f.bv g.bv
        + Apeiron.Crypto.Sha2.K.get ⟨i.val, hlt⟩ + wGet w i.val ∧
      t2.bv = Apeiron.Crypto.Sha2.Sigma0 a.bv + Apeiron.Crypto.Sha2.majority a.bv b.bv c.bv := by
  have h1v : (1#usize : Std.Usize).val = 1 := rfl
  obtain ⟨kw, hkw, hkwv⟩ :=
    WP.spec_imp_exists (Array.index_usize_spec sha256.K i (by simpa using hlt))
  obtain ⟨wi, hwi, hwiv⟩ :=
    WP.spec_imp_exists (Array.index_usize_spec w i (by simpa using hlt))
  have hmax : (64 : Nat) ≤ Std.Usize.max := by scalar_tac
  obtain ⟨i', hi', hival⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) (by rw [h1v]; omega))
  -- The five round words, as the source spells them.
  let s1 : Std.U32 :=
    core.num.U32.rotate_right e 6#u32 ^^^ core.num.U32.rotate_right e 11#u32
      ^^^ core.num.U32.rotate_right e 25#u32
  let ch : Std.U32 := (e &&& f) ^^^ ((~~~e) &&& g)
  let t1 : Std.U32 :=
    core.num.U32.wrapping_add (core.num.U32.wrapping_add
      (core.num.U32.wrapping_add (core.num.U32.wrapping_add h s1) ch) kw) wi
  let s0 : Std.U32 :=
    core.num.U32.rotate_right a 2#u32 ^^^ core.num.U32.rotate_right a 13#u32
      ^^^ core.num.U32.rotate_right a 22#u32
  let maj : Std.U32 := (a &&& b) ^^^ (a &&& c) ^^^ (b &&& c)
  let t2 : Std.U32 := core.num.U32.wrapping_add s0 maj
  let e₁ : Std.U32 := core.num.U32.wrapping_add d t1
  let a₁ : Std.U32 := core.num.U32.wrapping_add t1 t2
  have hival' : i'.val = i.val + 1 := by rw [← h1v]; exact hival
  refine ⟨a₁, e₁, t1, t2, i', ?_, hival', rfl, rfl, ?_, ?_⟩
  · unfold sha256.compress_loop2.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt)]
    simp only [hkw, hwi, hi', lift, bind_tc_ok]
    rfl
  · have hbvv : (core.num.U32.wrapping_add
        (core.num.U32.wrapping_add (core.num.U32.wrapping_add
          (core.num.U32.wrapping_add h s1) ch) kw) wi).bv
        = h.bv + s1.bv + ch.bv + kw.bv + wi.bv := rfl
    dsimp only [t1]
    rw [hbvv,
      show s1.bv = Apeiron.Crypto.Sha2.Sigma1 e.bv from Sigma1_round e,
      show ch.bv = Apeiron.Crypto.Sha2.choose e.bv f.bv g.bv from ch_step e f g,
      show kw.bv = Apeiron.Crypto.Sha2.K.get ⟨i.val, hlt⟩ from by
        rw [hkwv]; exact K_get ⟨i.val, hlt⟩,
      show wi.bv = wGet w i.val from by
        rw [hwiv]
        have hwlt : i.val < w.val.length := by simpa using hlt
        simp only [wGet, getElem_getD _ _ hwlt (0#u32)]]
  · have hbvv : (core.num.U32.wrapping_add s0 maj).bv = s0.bv + maj.bv := rfl
    dsimp only [t2]
    rw [hbvv,
      show s0.bv = Apeiron.Crypto.Sha2.Sigma0 a.bv from Sigma0_round a,
      show maj.bv = Apeiron.Crypto.Sha2.majority a.bv b.bv c.bv from maj_step a b c]

/-! ## §3 · The canonical round

The sixty-four-round fold of `Sha2.compress`, lifted out so that the invariant and the extracted
link can be stated about it. `roundStep` is `Sha2.compress`'s own body, not a second transcription;
`compress_eq_rounds` is the `rfl` that says so. -/

/-- One FIPS 180-4 §6.2.2 round: `T₁ = h + Σ₁(e) + Ch + K[t] + W[t]`, `T₂ = Σ₀(a) + Maj`, and the
eight-way working-word rotation. `O(1)`: eight reads, the two tables, and eight writes. -/
def roundStep (w : Vector Word 64) (acc : State) (t : Nat) : State :=
  let a := acc.getD 0 0
  let b := acc.getD 1 0
  let c := acc.getD 2 0
  let d := acc.getD 3 0
  let e := acc.getD 4 0
  let f := acc.getD 5 0
  let g := acc.getD 6 0
  let hh := acc.getD 7 0
  let t1 := hh + Apeiron.Crypto.Sha2.Sigma1 e + Apeiron.Crypto.Sha2.choose e f g
    + Apeiron.Crypto.Sha2.K.getD t 0 + w.getD t 0
  let t2 := Apeiron.Crypto.Sha2.Sigma0 a + Apeiron.Crypto.Sha2.majority a b c
  Vector.ofFn fun i : Fin 8 =>
    match i.val with
    | 0 => t1 + t2
    | 1 => a
    | 2 => b
    | 3 => c
    | 4 => d + t1
    | 5 => e
    | 6 => f
    | _ => g

/-- The round fold over the first `n` rounds, `n ≤ 64`. `Θ(8n)` element writes for `n ≤ 64`: each of
the `n` visits rewrites all eight working words. -/
def roundsN (w : Vector Word 64) (v0 : State) (n : Nat) : State :=
  (List.range n).foldl (fun acc t => roundStep w acc t) v0

/-- The full sixty-four-round fold of `Sha2.compress`. `O(1)` to name; `Θ(8·64)` element writes to
evaluate. -/
def rounds (w : Vector Word 64) (v0 : State) : State := roundsN w v0 64

/-- **The canonical compression is this round fold, then the add-back.** `Sha2.compress h block` is
`roundStep`-folded sixty-four times over `schedule block` from the incoming state, with the eight
add-back writes of its final `Vector.ofFn` — this is the `rfl` that binds every canonical row below
to production source. `proved`, `rfl`. -/
theorem compress_eq_rounds (h : State) (block : Block) :
    Apeiron.Crypto.Sha2.compress h block
      = Vector.ofFn (fun i : Fin 8 =>
          h.getD i.val 0 + (rounds (Apeiron.Crypto.Sha2.schedule block) h).getD i.val 0) :=
  rfl

/-- One more round is one more `roundStep` at the previous count, by `List.range_succ` and
`List.foldl_append`. `proved`. -/
theorem rounds_succ (w : Vector Word 64) (v0 : State) (n : Nat) :
    roundsN w v0 (n + 1) = roundStep w (roundsN w v0 n) n := by
  show (List.range (n + 1)).foldl (fun acc t => roundStep w acc t) v0
      = roundStep w ((List.range n).foldl (fun acc t => roundStep w acc t) v0) n
  rw [List.range_succ, List.foldl_append]
  rfl

/-! ## §4 · The extracted rotation is the canonical round

For **every** schedule: the only schedule reads are `wGet w i.val` on the extracted side and
`(wView w).getD i.val 0` on the canonical side, which the corpus's `wView_getD` identifies. -/

/-- **The extracted rotation is the canonical round.** One visit of `sha256.compress_loop2.body`
takes the eight working words to exactly `roundStep` of the canonical round fold at the same schedule
row. `proved`. -/
theorem roundBody_vec (w : Array Std.U32 64#usize) (a b c d e f g h : Std.U32)
    (i : Std.Usize) (hlt : i.val < 64)
    (a₁ e₁ t1 t2 : Std.U32) (i' : Std.Usize)
    (ha₁ : a₁.bv = t1.bv + t2.bv)
    (he₁ : e₁.bv = d.bv + t1.bv)
    (ht1 : t1.bv = h.bv + Apeiron.Crypto.Sha2.Sigma1 e.bv
      + Apeiron.Crypto.Sha2.choose e.bv f.bv g.bv
      + Apeiron.Crypto.Sha2.K.get ⟨i.val, hlt⟩ + wGet w i.val)
    (ht2 : t2.bv = Apeiron.Crypto.Sha2.Sigma0 a.bv + Apeiron.Crypto.Sha2.majority a.bv b.bv c.bv) :
    wkState a₁ a b c e₁ e f g = roundStep (wView w) (wkState a b c d e f g h) i.val := by
  refine vec_getD_eq _ _ 0 fun j hj => ?_
  rw [wkState_getD _ _ _ _ _ _ _ _ j hj]
  unfold roundStep
  show _ = (Vector.ofFn _).getD j 0
  rw [ofFn_getD' _ j hj 0, wView_getD w i.val (by omega), K_getD i.val hlt]
  rw [wkState_getD _ _ _ _ _ _ _ _ 0 (by decide), wkState_getD _ _ _ _ _ _ _ _ 1 (by decide),
    wkState_getD _ _ _ _ _ _ _ _ 2 (by decide), wkState_getD _ _ _ _ _ _ _ _ 3 (by decide),
    wkState_getD _ _ _ _ _ _ _ _ 4 (by decide), wkState_getD _ _ _ _ _ _ _ _ 5 (by decide),
    wkState_getD _ _ _ _ _ _ _ _ 6 (by decide), wkState_getD _ _ _ _ _ _ _ _ 7 (by decide)]
  have hc : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨ j = 5 ∨ j = 6 ∨ j = 7 := by
    omega
  rcases hc with h | h | h | h | h | h | h | h <;> subst h <;>
    simp only <;>
    simp only [ha₁, he₁, ht1, ht2]

/-! ## §5 · The round invariant

At loop counter `i.val ≤ 64`, the eight working words are the first `i.val` rounds of the canonical
round fold over the schedule `w`, starting from the incoming state `v0`. -/

/-- **The round invariant.** `proved` below in `roundInv_init`, `roundInv_step`, `roundInv_post`. -/
def roundInv (w : Vector Word 64) (v0 : State) (s : State) (i : Nat) : Prop :=
  i ≤ 64 ∧ s = roundsN w v0 i

/-- **Init.** The round loop starts from the incoming state itself. `proved`. -/
theorem roundInv_init (w : Vector Word 64) (v0 : State) : roundInv w v0 v0 0 := by
  refine ⟨by omega, ?_⟩
  show v0 = roundsN w v0 0
  rfl

/-- **Step.** One visit advances the invariant by one round. **No schedule hypothesis is used.**
`proved`. -/
theorem roundInv_step (w : Vector Word 64) (v0 s : State) (i : Nat)
    (hi : i < 64) (hs : s = roundsN w v0 i) :
    roundInv w v0 (roundStep w s i) (i + 1) := by
  refine ⟨by omega, ?_⟩
  rw [hs, rounds_succ w v0 i]

/-- **Post at the loop exit.** At sixty-four rounds the invariant reads as the full round fold.
`proved`. -/
theorem roundInv_post (w : Vector Word 64) (v0 s : State)
    (h : roundInv w v0 s 64) : s = rounds w v0 :=
  h.2.trans (by show roundsN w v0 64 = rounds w v0; rfl)

/-! ## §6 · The result and loop-state types, and total success -/

/-- The result tuple of `sha256.compress_loop2`: the eight working words. -/
abbrev R8 : Type :=
  Std.U32 × Std.U32 × Std.U32 × Std.U32 × Std.U32 × Std.U32 × Std.U32 × Std.U32

/-- The loop state of `sha256.compress_loop2`: the eight working words and the counter. -/
abbrev R9 : Type :=
  Std.U32 × Std.U32 × Std.U32 × Std.U32 × Std.U32 × Std.U32 × Std.U32 × Std.U32 × Std.Usize

/-- The eight working words of a result tuple, as a `Sha2.State`. `O(1)`. -/
def wk8 (y : R8) : State :=
  wkState y.1 y.2.1 y.2.2.1 y.2.2.2.1 y.2.2.2.2.1 y.2.2.2.2.2.1 y.2.2.2.2.2.2.1
    y.2.2.2.2.2.2.2

/-- The eight working words of a loop state, as a `Sha2.State`. `O(1)`. -/
def wk9 (p : R9) : State :=
  wkState p.1 p.2.1 p.2.2.1 p.2.2.2.1 p.2.2.2.2.1 p.2.2.2.2.2.1 p.2.2.2.2.2.2.1
    p.2.2.2.2.2.2.2.1

/-- The nine-tuple projection of the loop counter. `O(1)`. -/
def i9 (p : R9) : Nat := p.2.2.2.2.2.2.2.2.val

/-- **The round loop's exit, at the generic schedule `wView w`.** `proved`. -/
theorem roundLoop_post (w : Array Std.U32 64#usize)
    (v0 s : State) (h : roundInv (wView w) v0 s 64) :
    s = rounds (wView w) v0 :=
  roundInv_post (wView w) v0 s h

/-- **The sixty-four-round loop, WP style (generic, any schedule).**
`sha256.compress_loop2` carries the eight working words through the sixty-four-round fold of
`roundStep` over `wView w` applied to the incoming state, by `loop.spec_decr_nat` on the measure
`64 - i.val` with invariant `roundInv`. **No schedule premise is required.** `proved`. -/
theorem roundLoop_wp (w : Array Std.U32 64#usize)
    (a b c d e f g h : Std.U32) :
    sha256.compress_loop2 w a b c d e f g h 0#usize ⦃ fun y : R8 =>
      wk8 y = rounds (wView w) (wkState a b c d e f g h) ⦄ := by
  simp only [sha256.compress_loop2]
  apply loop.spec_decr_nat
    (measure := fun p : R9 => 64 - i9 p)
    (inv := fun p : R9 =>
      roundInv (wView w) (wkState a b c d e f g h) (wk9 p) (i9 p))
    (post := fun y : R8 =>
      wk8 y = rounds (wView w) (wkState a b c d e f g h))
  · intro p hp
    obtain ⟨hle, hs⟩ := hp
    by_cases hlt : i9 p < 64
    · obtain ⟨a₁, e₁, t1, t2, i', hb, hi', ha₁, he₁, ht1, ht2⟩ :=
        roundBody_step w p.1 p.2.1 p.2.2.1 p.2.2.2.1 p.2.2.2.2.1 p.2.2.2.2.2.1
          p.2.2.2.2.2.2.1 p.2.2.2.2.2.2.2.1 p.2.2.2.2.2.2.2.2 hlt
      rw [WP.spec_equiv_exists]
      refine ⟨_, hb, ?_⟩
      have hvec := roundBody_vec w p.1 p.2.1 p.2.2.1 p.2.2.2.1 p.2.2.2.2.1 p.2.2.2.2.2.1
        p.2.2.2.2.2.2.1 p.2.2.2.2.2.2.2.1 p.2.2.2.2.2.2.2.2 hlt
        a₁ e₁ t1 t2 i' ha₁ he₁ ht1 ht2
      have hstep : roundInv (wView w) (wkState a b c d e f g h)
          (roundStep (wView w) (wk9 p) (i9 p)) (i9 p + 1) :=
        roundInv_step (wView w) (wkState a b c d e f g h) (wk9 p) (i9 p) hlt hs
      have hstep' : roundInv (wView w) (wkState a b c d e f g h)
          (wkState a₁ p.1 p.2.1 p.2.2.1 e₁ p.2.2.2.2.1 p.2.2.2.2.2.1 p.2.2.2.2.2.2.1) i'.val := by
        dsimp [wk9, i9] at hstep
        rw [← hvec] at hstep
        rw [← hi'] at hstep
        exact hstep
      obtain ⟨hle_i', hs_eq⟩ := hstep'
      refine ⟨⟨hle_i', hs_eq⟩, ?_⟩
      simp only [i9]
      rw [hi']; omega
    · rw [WP.spec_equiv_exists]
      have h64 : i9 p = 64 := by omega
      refine ⟨ControlFlow.done (p.1, p.2.1, p.2.2.1, p.2.2.2.1, p.2.2.2.2.1,
        p.2.2.2.2.2.1, p.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.1), ?_, ?_⟩
      · unfold sha256.compress_loop2.body
        rw [if_neg (by rw [UScalar.lt_equiv]; exact hlt)]
      · calc
          wk8 (p.1, p.2.1, p.2.2.1, p.2.2.2.1, p.2.2.2.2.1,
            p.2.2.2.2.2.1, p.2.2.2.2.2.2.1, p.2.2.2.2.2.2.2.1)
              = wk9 p := rfl
          _ = roundsN (wView w) (wkState a b c d e f g h) (i9 p) := hs
          _ = roundsN (wView w) (wkState a b c d e f g h) 64 := by rw [h64]
          _ = rounds (wView w) (wkState a b c d e f g h) := rfl
  · show roundInv (wView w) (wkState a b c d e f g h)
      (wkState a b c d e f g h) 0
    exact roundInv_init (wView w) (wkState a b c d e f g h)

/-- **Total success of the sixty-four-round compression loop.** Existential total-success form of
`roundLoop_wp`: for every schedule array `w` and eight initial words, the loop **returns some `r`**,
that run **succeeds**, and the working words of `r` are the canonical sixty-four-round fold
`rounds (wView w)` of the incoming state. This is the strongest shape here — no schedule premise and
no equation-only restriction. `proved`. -/
theorem roundLoop_spec (w : Array Std.U32 64#usize)
    (a b c d e f g h : Std.U32) :
    ∃ r : R8,
      sha256.compress_loop2 w a b c d e f g h 0#usize = Result.ok r ∧
      wk8 r = rounds (wView w) (wkState a b c d e f g h) := by
  let Q : Aeneas.Std.WP.Post R8 := fun y : R8 => wk8 y = rounds (wView w) (wkState a b c d e f g h)
  have hwp : (sha256.compress_loop2 w a b c d e f g h 0#usize) ⦃ Q ⦄ :=
    roundLoop_wp w a b c d e f g h
  obtain ⟨r, hr, hPr⟩ := WP.spec_imp_exists hwp
  exact ⟨r, hr, hPr⟩

/-- **Equation-only corollary**, weaker and labelled as such: `roundLoop_spec` implies it, but not
vice versa — this form takes the successful run as a premise, while `roundLoop_spec` guarantees
existence from totality alone. `proved`. -/
theorem roundLoop_eq (w : Array Std.U32 64#usize)
    (a b c d e f g h : Std.U32) (r : R8)
    (hspec : sha256.compress_loop2 w a b c d e f g h 0#usize = Result.ok r) :
    wk8 r = rounds (wView w) (wkState a b c d e f g h) := by
  have hspec' : ∃ r' : R8,
      sha256.compress_loop2 w a b c d e f g h 0#usize = Result.ok r' ∧
      wk8 r' = rounds (wView w) (wkState a b c d e f g h) :=
    roundLoop_spec w a b c d e f g h
  obtain ⟨r', hr', hPr'⟩ := hspec'
  have hr : r = r' := Result.ok_injective (hspec.symm.trans hr')
  rw [hr]; exact hPr'

/-! ## §7 · Canonical transport, and non-vacuity of its hypothesis

`roundLoop_canonical` takes `hw : wView w = Sha2.schedule (bytesToWords blk)` at one array. That
hypothesis is inhabited for every block — `wOfVec` builds an extracted sixty-four-word array from
any canonical `Vector Word 64`, and `wView_wOfVec` proves the read-back — so the conditional
statement is not vacuous. -/

/-- **The round loop's exit, at the canonical schedule.** `proved`. -/
theorem roundLoop_post_canonical (w : Array Std.U32 64#usize)
    (blk : Array Std.U8 64#usize)
    (hw : wView w = Apeiron.Crypto.Sha2.schedule (bytesToWords blk))
    (v0 s : State) (h : roundInv (wView w) v0 s 64) :
    s = rounds (Apeiron.Crypto.Sha2.schedule (bytesToWords blk)) v0 :=
  (roundInv_post (wView w) v0 s h).trans (by rw [hw])

/-- **`sha256.compress_loop2` runs the canonical sixty-four-round fold.** With the local hypothesis
`hw` identifying the extracted schedule array with the canonical schedule of a block, the loop
**succeeds** and its result equals the canonical `Sha2.compress` round fold applied to the initial
state. Grade `conditional`: `hw` is a premise; its non-vacuity is `schedule_premise_inhabited`.
`proved`. -/
theorem roundLoop_canonical (w : Array Std.U32 64#usize)
    (blk : Array Std.U8 64#usize)
    (hw : wView w = Apeiron.Crypto.Sha2.schedule (bytesToWords blk))
    (a b c d e f g h : Std.U32) :
    ∃ r : R8,
      sha256.compress_loop2 w a b c d e f g h 0#usize = Result.ok r ∧
      wk8 r = rounds (Apeiron.Crypto.Sha2.schedule (bytesToWords blk))
        (wkState a b c d e f g h) := by
  obtain ⟨r, hr, hPr⟩ := roundLoop_spec w a b c d e f g h
  refine ⟨r, hr, ?_⟩
  rw [hPr, hw]

/-- The extracted `u32` carrying a canonical word's bit pattern. `O(1)`. -/
def u32OfWord (w : Word) : Std.U32 :=
  U32.ofNatCore w.toNat (by
    have h : w.toNat < 2 ^ 32 := w.isLt
    simpa only [U32.numBits, UScalarTy.U32_numBits_eq] using h)

/-- Reading the extracted word back is the canonical word. `proved`. -/
@[simp] theorem u32OfWord_bv (w : Word) : (u32OfWord w).bv = w := by
  apply BitVec.eq_of_toNat_eq
  simp only [u32OfWord, U32.bv_toNat, U32.ofNatCore_val_eq]

/-- The extracted sixty-four-word schedule array built from a canonical schedule vector: row `i`
holds `v`'s row `i` as an extracted `u32`. `Θ(64)` writes, `O(1)` per row. -/
def wOfVec (v : Vector Word 64) : Array Std.U32 64#usize :=
  Array.from (List.ofFn (fun i : Fin 64 => u32OfWord (v.getD i.val 0))) (by simp)

/-- Reading the built array back through `wView` is the canonical vector. `proved`. -/
theorem wView_wOfVec (v : Vector Word 64) : wView (wOfVec v) = v := by
  refine vec_getD_eq _ _ 0 fun j hj => ?_
  rw [wView_getD _ j hj]
  unfold wGet wOfVec
  rw [Array.from_val]
  rw [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
  rw [dif_pos hj]
  simp

/-- **The witness.** For every block there is an extracted schedule array whose `wView` is the
canonical schedule of that block: the local premise of `roundLoop_canonical` is inhabited, not
vacuous. `proved`. -/
theorem schedule_premise_inhabited (blk : Array Std.U8 64#usize) :
    ∃ w : Array Std.U32 64#usize,
      wView w = Apeiron.Crypto.Sha2.schedule (bytesToWords blk) :=
  ⟨wOfVec (Apeiron.Crypto.Sha2.schedule (bytesToWords blk)), wView_wOfVec _⟩

/-- The eight FIPS 180-4 §5.3.3 initial hash words as extracted `u32`s. `O(1)`. -/
def h0Words : R8 :=
  (u32OfWord (Apeiron.Crypto.Sha2.H0.getD 0 0),
   u32OfWord (Apeiron.Crypto.Sha2.H0.getD 1 0),
   u32OfWord (Apeiron.Crypto.Sha2.H0.getD 2 0),
   u32OfWord (Apeiron.Crypto.Sha2.H0.getD 3 0),
   u32OfWord (Apeiron.Crypto.Sha2.H0.getD 4 0),
   u32OfWord (Apeiron.Crypto.Sha2.H0.getD 5 0),
   u32OfWord (Apeiron.Crypto.Sha2.H0.getD 6 0),
   u32OfWord (Apeiron.Crypto.Sha2.H0.getD 7 0))

/-- **An actual initial state and a reachable schedule, at the FIPS initial hash value.** The loop
runs from the FIPS §5.3.3 words on the extracted array built from the zero block's canonical
schedule, **succeeds**, and returns the canonical sixty-four-round fold. This is the joint witness
obligation 2 asks for: a reachable input pair, not a hypothesis bundle. `proved`. -/
theorem zeroBlock_initialState_certificate :
    ∃ r : R8,
      sha256.compress_loop2
          (wOfVec (Apeiron.Crypto.Sha2.schedule (bytesToWords
            (Array.repeat 64#usize 0#u8))))
          h0Words.1 h0Words.2.1 h0Words.2.2.1 h0Words.2.2.2.1 h0Words.2.2.2.2.1
          h0Words.2.2.2.2.2.1 h0Words.2.2.2.2.2.2.1 h0Words.2.2.2.2.2.2.2 0#usize
          = Result.ok r ∧
      wk8 r = rounds (Apeiron.Crypto.Sha2.schedule (bytesToWords
          (Array.repeat 64#usize 0#u8)))
        (wkState h0Words.1 h0Words.2.1 h0Words.2.2.1 h0Words.2.2.2.1
          h0Words.2.2.2.2.1 h0Words.2.2.2.2.2.1 h0Words.2.2.2.2.2.2.1
          h0Words.2.2.2.2.2.2.2) :=
  roundLoop_canonical _ _ (wView_wOfVec _) _ _ _ _ _ _ _ _

/-! ## §8 · The feed-forward (add-back) link

`Sha2.compress` finishes by adding the sixty-four round fold's eight working words back into the
incoming state (`Crypto/Sha2.lean:120`). `compress_eq_rounds` is that fold; the rows below state the
add-back componentwise, so the *whole* compression is the round fold followed by the add-back, and
`compress_eq_rounds` together with them is the composed canonical statement. -/

/-- **The add-back writes `H' = H + V` componentwise.** `proved`. -/
theorem addBack_getD (h : State) (w : Vector Word 64) (i : Fin 8) :
    (Vector.ofFn (fun j : Fin 8 =>
      h.getD j.val 0 + (rounds w h).getD j.val 0)).getD i.val 0
      = h.getD i.val 0 + (rounds w h).getD i.val 0 := by
  have hi8 : (i : Nat) < 8 := i.isLt
  have hget : (Vector.ofFn (fun j : Fin 8 =>
      h.getD j.val 0 + (rounds w h).getD j.val 0)).getD i.val 0
      = (Vector.ofFn (fun j : Fin 8 =>
        h.getD j.val 0 + (rounds w h).getD j.val 0))[(i : Nat)] := by
    simp [Vector.getD, hi8]
  rw [hget, Vector.getElem_ofFn i.isLt]

/-- **The feed-forward link: the canonical compression's slot `i` is the incoming state's slot plus
the sixty-four-round fold's slot.** With `block`'s own schedule as `w`, this is the composition of
`compress_eq_rounds` with the add-back read, i.e. the feed-forward layer of `Sha2.compress`
discharged on the canonical side. `proved`. -/
theorem compress_feedForward (h : State) (block : Block) (i : Fin 8) :
    (Apeiron.Crypto.Sha2.compress h block).getD i.val 0
      = h.getD i.val 0
        + (rounds (Apeiron.Crypto.Sha2.schedule block) h).getD i.val 0 := by
  rw [compress_eq_rounds]
  exact addBack_getD h _ i

/-! ## §9 · Axiom cone -/

#print axioms wkState_getD
#print axioms K_getD
#print axioms roundBody_step
#print axioms rounds_succ
#print axioms compress_eq_rounds
#print axioms roundBody_vec
#print axioms roundInv_init
#print axioms roundInv_step
#print axioms roundInv_post
#print axioms roundLoop_post
#print axioms roundLoop_wp
#print axioms roundLoop_spec
#print axioms roundLoop_eq
#print axioms roundLoop_post_canonical
#print axioms roundLoop_canonical
#print axioms u32OfWord_bv
#print axioms wView_wOfVec
#print axioms schedule_premise_inhabited
#print axioms zeroBlock_initialState_certificate
#print axioms addBack_getD
#print axioms compress_feedForward

end Compression
end Extraction
end Icarus