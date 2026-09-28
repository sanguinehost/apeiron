import Batteries

/-!
# Seed — the runnable `.seed` pipeline (pure Lean, no Mathlib)
@name How to Shrink Anything to Disk and Get It Back
@child Take any file, split it into "the part you can guess" and "the leftover surprise," store only what's needed, and you can rebuild it perfectly. This is the save-and-restore pipeline.
@scientist The runnable `.seed` pipeline: the floor-with-residue split (prediction = Being arm, delta = the surprise), the codebook dedup, and the disk path (collapse → flatten → LEB128 → bytes), with lossless round-trip proofs.
@engineer `dedup'` with `mem_dedup'` (membership is all the round-trip needs); the floor/residue split and varint pipeline — pure Lean + Batteries, so `sang seedcast`/`reseed` carry no Mathlib (rate-distortion/Null-cell tie stay in the parity layer).

The runnable half of `Substrate/Seed.lean`: the floor-with-residue split, the generator
(genome), and the disk pipeline (collapse → flatten → LEB128 varint → bytes), with their
round-trip proofs. Pure Lean + Batteries — no Mathlib in the execution path, so `sang
seedcast`/`reseed` carry none. The rate-distortion (§8), the Null-cell tie (§9), the
physical-medium reading (§10), and the ℤ trinity/triad generators stay in `Substrate/Seed`
as the Mathlib parity layer; they import this core.

`Mathlib.List.dedup` (the codebook dedup) is re-stood here as the structural `dedup'`
(membership-preserving — the only property the lossless round-trip needs).

## What "lossless round-trip" warrants — and where the disk path stops

`SanguineOS.storage_is_collapse` (law 8) rides `loadBytes_saveBytes` and `Gen.decode_encode`
directly, and `Scribe.lean`, `Runtime/SangNet.lean` and `Runtime/SangClone.lean` all cite them.
The proofs are real; the reading "anything, to disk and back" is one step wider than the types.

* **Proved (native):** `Gen.decode_encode` — `g.decode (g.encode t) = t` for **any** generator
  over any context, by the self-inverse of `⊻` (`Nat.xor_assoc`/`xor_self`/`xor_zero`) in
  `genDecode_genEncode`. Unconditional and genuinely general: `predict` and `step` are
  arbitrary. This is the file's strongest statement.
* **Proved (native):** `rexpand_rcollapse` — the codebook collapse is lossless for any `Floor`,
  the real work being `List.getElem_idxOf` at the dedup index plus `F.recover`. Its only
  hypothesis is the `Floor` structure's own `recover` field, which is what a `Floor` is for.
* **Proved (native):** the varint layer — `deWord_serWord` (strong induction on `n / 128`),
  `deWords_serWords` (with `length_le_serWords` supplying the fuel), `takePairs_flatPairs`,
  `runflatten_rflatten` — composing to `loadR_saveR` and `loadBytes_saveBytes`. Every step is
  proved, none is assumed.
* **Proved (native):** `mem_dedup'` — membership is preserved, which is the only property
  `rexpand_rcollapse` uses. `dedup'` keeping the last occurrence, and `dedup'` producing a
  `Nodup` list, are stated nowhere and are not needed.
* **Next:** that this reaches **bytes**. `saveBytes`/`loadBytes` are
  `List Nat → List Nat`; the only step to a real `ByteArray` is `toByteArray`, which is a `def`
  with **no theorem of any kind**. It maps through `UInt8.ofNat`, which truncates mod 256, and
  nothing here states that `serWords`' output entries are `< 256` — they are, by inspection of
  `serWord` (`[n]` when `n < 128`, else `n % 128 + 128 ≤ 255`), but that is an inspection, not a
  lemma. The header's *"the disk path (collapse → flatten → LEB128 → bytes)"* and `@name`'s
  *"Shrink Anything to Disk and Get It Back"* stop one constructor short of the claim.
* **Next:** the other direction of the round trip.
  `loadBytes_saveBytes` is decode-after-**encode**. Encode-after-decode — that a byte stream the
  loader accepts is one the saver would have produced — is proved nowhere, and
  `Runtime/SangNet.lean` already records that its membrane's self/non-self verdict rests on
  exactly that unproved direction: *"bytes the decoder accepts but would never have produced are
  exactly the gap, and they would read as self."*
* **Anchor:** `Word`'s docstring, *"the proven `ℤ/2ⁿ` machine word as a
  byte-pattern"*. `abbrev Word : Type := Nat` — unbounded, no wraparound. The corpus's ℤ/2ⁿ word
  is `Substrate/Ski/WordReduces.lean`'s positional bit-list, which this file does not import and
  no statement here mentions. `serWord`'s termination in fact relies on `Nat`, not on a width.
* **Anchor:** that a `Floor`'s `key` is a *prediction* (the Being arm) and `delta`
  a *surprise* (the Null arm). `Floor` is four fields with one equation, `combine (key w)
  (delta w) = w`; `idFloor` satisfies it with `key = id` and `delta = 0`, predicting nothing.
  Which floors compress is not addressed — no length, rate or entropy statement is in this file.
* **Open:** the rate–distortion layer (§8), the Null-cell tie (§9), the physical-medium reading
  (§10) and the ℤ trinity generators, all deliberately in `Substrate/Seed.lean`, the Mathlib
  parity layer that imports this core. The section numbering here jumps §2 → §6 for that reason.

-/

namespace Sanguine.Seed

/-- A stored word — the proven `ℤ/2ⁿ` machine word as a byte-pattern. -/
abbrev Word : Type := Nat
/-- A tensor / file is its word (byte) stream. So `.seed` stores **anything**. -/
abbrev Tensor : Type := List Word

-- ═══════════════════════════════════════════════════════════════════
-- DEDUP — pure, structural (re-stood `List.dedup`; membership is all the round-trip needs)
-- ═══════════════════════════════════════════════════════════════════

/-- Membership-preserving dedup (the codebook builder). Keeps the last occurrence; only
    `mem_dedup'` is load-bearing for the lossless round-trip (the table must hold every key). -/
def dedup' {α : Type} [DecidableEq α] : List α → List α
  | []      => []
  | a :: as => if a ∈ as then dedup' as else a :: dedup' as

theorem mem_dedup' {α : Type} [DecidableEq α] {a : α} {l : List α} :
    a ∈ dedup' l ↔ a ∈ l := by
  induction l with
  | nil => simp [dedup']
  | cons b bs ih =>
      by_cases hb : b ∈ bs
      · have hd : dedup' (b :: bs) = dedup' bs := by simp [dedup', hb]
        rw [hd, ih, List.mem_cons]
        constructor
        · intro h; exact Or.inr h
        · intro h; rcases h with rfl | h
          · exact hb
          · exact h
      · have hd : dedup' (b :: bs) = b :: dedup' bs := by simp [dedup', hb]
        rw [hd, List.mem_cons, List.mem_cons, ih]

-- ═══════════════════════════════════════════════════════════════════
-- §1. THE FLOOR — the one per-key datum (prediction + residue + recompose)
-- ═══════════════════════════════════════════════════════════════════

/-- A **floor-with-residue split**: `key` (the floor/prediction — the Being arm), `delta`
    (the residue/deviation — the Null arm), `combine` (recompose), and `recover` (the split
    is lossless). The only per-space datum; correctness is then free. -/
structure Floor where
  key     : Word → Word
  delta   : Word → Word
  combine : Word → Word → Word
  recover : ∀ w, combine (key w) (delta w) = w

/-- The collapsed form: the codebook (distinct floors) + one `(floor index, residue)` per
    element. This is exactly what the byte stream holds. -/
structure RCollapsed where
  table   : List Word
  entries : List (Nat × Word)

/-- **Collapse** — dedup the floors into the codebook; record each element's `(floor index,
    residue)`. -/
def rcollapse (F : Floor) (t : Tensor) : RCollapsed :=
  { table   := dedup' (t.map F.key),
    entries := t.map (fun w => ((dedup' (t.map F.key)).idxOf (F.key w), F.delta w)) }

/-- **Expand** — recombine each floor (by index) with its residue. -/
def rexpand (F : Floor) (c : RCollapsed) : Tensor :=
  c.entries.map (fun e => F.combine (c.table.getD e.1 0) e.2)

/-- **The collapse is lossless for any `Floor`.** Each element's floor sits at its dedup
    index, the codebook holds it there, and `recover` recomposes it with the residue. -/
theorem rexpand_rcollapse (F : Floor) (t : Tensor) : rexpand F (rcollapse F t) = t := by
  have key : ∀ w ∈ t,
      F.combine ((dedup' (t.map F.key)).getD ((dedup' (t.map F.key)).idxOf (F.key w)) 0) (F.delta w) = w := by
    intro w hw
    have hmem : F.key w ∈ dedup' (t.map F.key) :=
      mem_dedup'.mpr (List.mem_map.mpr ⟨w, hw, rfl⟩)
    have hlt : (dedup' (t.map F.key)).idxOf (F.key w) < (dedup' (t.map F.key)).length :=
      List.idxOf_lt_length_of_mem hmem
    have hg : (dedup' (t.map F.key)).getD ((dedup' (t.map F.key)).idxOf (F.key w)) 0 = F.key w := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlt]
      simp [List.getElem_idxOf hlt]
    rw [hg, F.recover]
  calc rexpand F (rcollapse F t)
      = t.map (fun w =>
          F.combine ((dedup' (t.map F.key)).getD ((dedup' (t.map F.key)).idxOf (F.key w)) 0) (F.delta w)) := by
        simp [rexpand, rcollapse, List.map_map, Function.comp]
    _ = t.map id := List.map_congr_left key
    _ = t := List.map_id _

/-- The **identity floor** — `key = id`, residue `0`. The lossless dedup container. -/
def idFloor : Floor where
  key w := w
  delta _ := 0
  combine a _ := a
  recover _ := rfl

-- ═══════════════════════════════════════════════════════════════════
-- §1b. THE GENERATOR — the seed is a GENOME (compression = prediction)
-- ═══════════════════════════════════════════════════════════════════

/-- A **generator** over a context type `Ctx`: the genome `init`, the predictor `predict`
    (Being arm), and `step` (advance the context with the realized value). -/
structure Gen (Ctx : Type) where
  init    : Ctx
  predict : Ctx → Word
  step    : Ctx → Word → Word → Ctx

/-- Encode: walk the stream, emit `residual = actual ⊻ predict ctx` (the Null arm), advance. -/
def genEncode {Ctx : Type} (g : Gen Ctx) : Ctx → Tensor → List Word
  | _,   []        => []
  | ctx, a :: rest => (a ^^^ g.predict ctx) :: genEncode g (g.step ctx (g.predict ctx) a) rest

/-- Decode: recover `actual = residual ⊻ predict ctx`, emit it, advance with the same step. -/
def genDecode {Ctx : Type} (g : Gen Ctx) : Ctx → List Word → Tensor
  | _,   []        => []
  | ctx, r :: rest =>
      let a := r ^^^ g.predict ctx
      a :: genDecode g (g.step ctx (g.predict ctx) a) rest

/-- **The generator round-trips losslessly — for ANY predictor.** Decode inverts encode by
    the self-inverse of `⊻` (`(a ⊻ p) ⊻ p = a`): the residual is the Null arm, the
    prediction the Being arm, and recombination is one xor. -/
theorem genDecode_genEncode {Ctx : Type} (g : Gen Ctx) (ctx : Ctx) (t : Tensor) :
    genDecode g ctx (genEncode g ctx t) = t := by
  induction t generalizing ctx with
  | nil => rfl
  | cons a rest ih =>
      have hx : (a ^^^ g.predict ctx) ^^^ g.predict ctx = a := by
        rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
      simp only [genEncode, genDecode, hx]
      rw [ih]

/-- Encode/decode from the genome `init`. -/
def Gen.encode {Ctx : Type} (g : Gen Ctx) (t : Tensor) : List Word := genEncode g g.init t
def Gen.decode {Ctx : Type} (g : Gen Ctx) (rs : List Word) : Tensor := genDecode g g.init rs

/-- **The genome regenerates the file.** `decode (encode t) = t` — the headline. -/
theorem Gen.decode_encode {Ctx : Type} (g : Gen Ctx) (t : Tensor) :
    g.decode (g.encode t) = t := genDecode_genEncode g g.init t

/-- The **null generator** — predicts `0`, residual = the value itself (no prediction). -/
def rawGen : Gen Unit where
  init := ()
  predict _ := 0
  step _ _ _ := ()

/-- The **order-1 generator** — predicts the *previous* value; residual = `actual ⊻ prev`.
    The simplest real generator (a Markov predictor). -/
def deltaGen : Gen Word where
  init := 0
  predict ctx := ctx
  step _ _ a := a

-- ═══════════════════════════════════════════════════════════════════
-- §2. THE BYTE LAYOUT — S :: table ++ N :: (idx,res pairs), self-delimiting
-- ═══════════════════════════════════════════════════════════════════

def flatPairs : List (Nat × Word) → List Nat
  | []          => []
  | (a, b) :: r => a :: b :: flatPairs r

def takePairs : List Nat → Option (List (Nat × Word))
  | []          => some []
  | [_]         => none
  | a :: b :: r => (takePairs r).map (fun ps => (a, b) :: ps)

theorem takePairs_flatPairs (es : List (Nat × Word)) : takePairs (flatPairs es) = some es := by
  induction es with
  | nil => rfl
  | cons e r ih => obtain ⟨a, b⟩ := e; simp [flatPairs, takePairs, ih]

def rflatten (c : RCollapsed) : List Nat :=
  c.table.length :: (c.table ++ (c.entries.length :: flatPairs c.entries))

def runflatten : List Nat → Option RCollapsed
  | []        => none
  | s :: rest =>
      let table := rest.take s
      match rest.drop s with
      | []         => none
      | _ :: rest2 => (takePairs rest2).map (fun es => { table := table, entries := es })

theorem runflatten_rflatten (c : RCollapsed) : runflatten (rflatten c) = some c := by
  simp [rflatten, runflatten, takePairs_flatPairs]

/-- Save a tensor to the `.seed` word stream under a floor. -/
def saveR (F : Floor) (t : Tensor) : List Nat := rflatten (rcollapse F t)

/-- Load a tensor from a `.seed` word stream. -/
def loadR (F : Floor) (bytes : List Nat) : Option Tensor := (runflatten bytes).map (rexpand F)

/-- **The load path is correct.** A `.seed` stream round-trips to the identical tensor. -/
theorem loadR_saveR (F : Floor) (t : Tensor) : loadR F (saveR F t) = some t := by
  simp only [loadR, saveR, runflatten_rflatten, Option.map_some, rexpand_rcollapse]

-- ═══════════════════════════════════════════════════════════════════
-- §6. WORD → BYTES — LEB128 varint, lossless, so `.seed` is real bytes
-- ═══════════════════════════════════════════════════════════════════

def serWord (n : Nat) : List Nat :=
  if h : n < 128 then [n]
  else (n % 128 + 128) :: serWord (n / 128)
termination_by n
decreasing_by exact Nat.div_lt_self (by omega) (by omega)

def deWord : List Nat → Option (Nat × List Nat)
  | []       => none
  | b :: rest =>
      if b < 128 then some (b, rest)
      else (deWord rest).map (fun p => (b - 128 + 128 * p.1, p.2))

theorem serWord_ne_nil (n : Nat) : serWord n ≠ [] := by rw [serWord]; split <;> simp

theorem deWord_serWord (n : Nat) (rest : List Nat) :
    deWord (serWord n ++ rest) = some (n, rest) := by
  induction n using Nat.strongRecOn with
  | _ n ih =>
    rw [serWord]
    by_cases h : n < 128
    · simp [h, deWord]
    · have hb : ¬ (n % 128 + 128 < 128) := by omega
      rw [dif_neg h, List.cons_append, deWord, if_neg hb,
          ih (n / 128) (Nat.div_lt_self (by omega) (by omega))]
      have hmod := Nat.mod_add_div n 128
      simp only [Option.map_some]
      refine congrArg some (Prod.ext ?_ rfl)
      omega

def serWords (ws : List Nat) : List Nat := ws.flatMap serWord

def deWords : Nat → List Nat → Option (List Nat)
  | _,        []      => some []
  | 0,        _ :: _  => none
  | fuel + 1, b :: bs =>
      match deWord (b :: bs) with
      | some (w, rest) => (deWords fuel rest).map (w :: ·)
      | none           => none

theorem deWords_serWords (ws : List Nat) (fuel : Nat) (hf : ws.length ≤ fuel) :
    deWords fuel (serWords ws) = some ws := by
  induction ws generalizing fuel with
  | nil => simp [serWords, deWords]
  | cons w ws ih =>
      obtain ⟨hb, tb, hbt⟩ := List.exists_cons_of_ne_nil (serWord_ne_nil w)
      cases fuel with
      | zero => simp at hf
      | succ f =>
          have hlen : ws.length ≤ f := by simpa using Nat.le_of_succ_le_succ hf
          have hbytes : serWords (w :: ws) = hb :: (tb ++ serWords ws) := by
            simp [serWords, hbt]
          rw [hbytes, deWords, ← List.cons_append, ← hbt, deWord_serWord]
          dsimp only
          rw [ih f hlen, Option.map_some]

theorem length_le_serWords (l : List Nat) : l.length ≤ (serWords l).length := by
  induction l with
  | nil => simp [serWords]
  | cons a l ih =>
      have h1 : 0 < (serWord a).length := by
        cases h : serWord a with
        | nil => exact absurd h (serWord_ne_nil a)
        | cons => simp
      simp only [serWords, List.flatMap_cons, List.length_append, List.length_cons] at ih ⊢
      omega

def saveBytes (F : Floor) (t : Tensor) : List Nat := serWords (saveR F t)

def loadBytes (F : Floor) (fuel : Nat) (bytes : List Nat) : Option Tensor :=
  (deWords fuel bytes).bind (loadR F)

/-- **The whole disk pipeline round-trips** (collapse + flatten + varint), with adequate fuel. -/
theorem loadBytes_saveBytes (F : Floor) (t : Tensor) :
    loadBytes F (saveBytes F t).length (saveBytes F t) = some t := by
  unfold loadBytes saveBytes
  rw [deWords_serWords (saveR F t) _ (length_le_serWords (saveR F t))]
  exact loadR_saveR F t

/-- A word stream (entries `< 256`) as a real `ByteArray`. -/
def toByteArray (bytes : List Nat) : ByteArray :=
  ⟨(bytes.map (fun n => UInt8.ofNat n)).toArray⟩

end Sanguine.Seed
