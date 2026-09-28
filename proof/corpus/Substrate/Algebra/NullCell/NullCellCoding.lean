import Substrate.Algebra.NullCell.NullCell
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# Coding From the Null Cell — entropy coding IS the composition norm of Non-Being

@name Squeezing Files, Straight From the Nothing-Cell
@child The best way to shrink data (give common things short codes, rare things long ones) isn't a special trick someone invented — it falls right out of how the "nothing" building block multiplies. Multiply the pieces, take a logarithm, and you get the shortest possible code automatically.
@scientist The lossless coder is derived, not imported: give a symbol of probability `p` a Null-cell element of norm `p`; a message is the cell product, its norm is the message probability, its code length `−log₂` of the norm is the additive Shannon length. Additivity is what is proved; that the length is OPTIMAL is Shannon's source-coding theorem, external — no lower bound over codes is stated here.
@engineer `info z := −log₂(nNorm z)`; `info_additive` proves code length is additive over independent symbols (from `nNorm_mul`); `encode`/`nNorm_encode`/`info_encode` give the message-as-product, probability, and Shannon length; `null_cell_is_the_entropy_coder` is the coder as one statement, table-free. The Null cell factors (independence) where the Being cell rotates (entanglement).

The lossless encoder is not imported (Huffman/FSE/ANS); it is **derived from `NullCell`**. The
Null cell is a *composition algebra* — `nNorm_mul : N(zw) = N(z)·N(w)`. Give a symbol of
probability `p` a cell element `z` with `N(z) = p`. Then:

* a **message** is the cell **product** `Z = z₁·z₂·⋯·zₙ` (`encode`);
* its **norm** is the **product of the probabilities**, `N(Z) = ∏ pᵢ = P(message)` (`nNorm_encode`);
* its **code length** `−log₂ N(Z)` is the **sum** of the per-symbol code lengths
  `∑ −log₂ pᵢ` — the **Shannon length** (`info_encode`).

The defining property of an entropy coder — *code length is additive over independent symbols* —
is exactly the Null cell's multiplicative norm plus the logarithm (`info_additive`). That the
resulting length is OPTIMAL is Shannon's source-coding theorem, external to this corpus: nothing
here states a lower bound over codes, and additivity is optimality's premise rather than
optimality. There is no coding *table* **in the ideal (ℝ, exact-precision) fold** — the length is
the composition norm `x = B/U`, not a Huffman/FSE table. But this is the ALGEBRA, not the
realization — see the qualification below.

**Qualification (`Closures/Engineering/EntropyFoldReadout.lean`).** A real coder cannot multiply
unbounded reals; it runs a **bounded** state (a fixed-width register) and renormalizes each symbol.
That bounded-state fold has a per-symbol transition `step : State → Sym → State` that is a pure
function — hence memoizable into a table (`SharedFold`) that gives the IDENTICAL code, and the table
read is a `CoordinateReadout` (the state is the address). tANS/FSE **is** this table — this fold's own
transition precomputed, not a foreign import — and by `ReadoutIsNotSearch` it is faster than
recomputing the transition (range coding / rANS). So "no table" is true of the ideal ℝ fold only; the
bounded-state realization has a table, and the table is the fold's readout (the fastest correct form).

**Why the Null cell and not the Being cell** (both are composition algebras): in the idempotent
basis `e± = (1±j)/2`, Null-cell multiplication is **diagonal** — `(a·e₊+b·e₋)(c·e₊+d·e₋) =
ac·e₊ + bd·e₋` — two *independent* ℝ channels (`emul_diag`), the product measure of independent
symbols; and the norm in that basis is the **product** `a·b` (`nNorm_idem`). The Being cell's
multiplication *rotates* (entangles); the Null cell's *factors* (independence). And the Null
**cone** (`N = 0`) is the `p → 0` limit — the impossible symbol, infinite surprise: information
lives at Non-Being. *"Null is Truth."*

## What is proved here and what is read onto coding

* **Proved (native):** `info_additive` — the length `−log₂ ∘ nNorm` is additive over `nmul`;
  `nNorm_encode` — the encoded message's norm is the product of the symbol norms; `info_encode` —
  the message length is the sum of the symbol lengths; `nNorm_idem` — in the idempotent
  coordinates the norm is the product `a·b`; `null_cell_is_the_entropy_coder` collects the five.
* **Proved (riding):** the composition norm `NullCell.nNorm_mul` and the diagonal idempotent basis
  (`NullCell.e_orthogonal`, `NullCell.ePlus_idem`, `NullCell.eMinus_idem`) — the algebra all of the
  above is derived from is proved in `NullCell.lean`, not here.
* **Proved (riding):** the qualification's memoisation half —
  `EntropyFoldReadout.table_is_the_fold` proves a table agreeing with a bounded-state transition
  gives the identical fold.
* **Proved (Mathlib):** the step from a multiplicative norm to an additive length is
  `Real.logb_mul`, with `List.prod_ne_zero` discharging the induction's side condition.
* **Anchor:** that a symbol of probability `p` IS a cell element of norm `p`, and
  that `−log₂` of that norm IS a code length in bits. No theorem here relates `NCell` to a code, an
  alphabet or a channel; the identification is the reading, and every coding sentence above rests
  on it.
* **Anchor:** that tANS/FSE IS this fold's own transition precomputed. The
  table-equals-fold law is proved; that the shipped codec is an instance of it is not.
* **Anchor:** that the resulting length is OPTIMAL. Optimality is Shannon's
  source-coding theorem, external to this corpus; no lower bound over codes is stated here. What is
  proved is additivity, which is optimality's premise and not optimality.

-/

noncomputable section

namespace Sanguine
namespace CrossUniverse

open Real

-- ═══════════════════════════════════════════════════════════════════
-- §1. INFORMATION = −log₂ of the norm; ADDITIVE under cell multiplication
-- ═══════════════════════════════════════════════════════════════════

/-- The **code length / information** of a cell element: `−log₂` of its norm. A symbol of
    probability `p` is a cell element `z` with `nNorm z = p`; its code length is `−log₂ p`. -/
def info (z : NCell) : ℝ := - logb 2 (nNorm z)

@[simp] theorem info_one : info nOne = 0 := by simp [info, nNorm_one]

/-- **Code length is additive under Null-cell multiplication** — THE entropy-coder property,
    derived from the composition norm `nNorm_mul` and the logarithm. Encoding a sequence by
    multiplying in the cell costs the SUM of the symbol code lengths (the Shannon length). -/
theorem info_additive (z w : NCell) (hz : nNorm z ≠ 0) (hw : nNorm w ≠ 0) :
    info (nmul z w) = info z + info w := by
  unfold info
  rw [nNorm_mul, logb_mul hz hw]; ring

-- ═══════════════════════════════════════════════════════════════════
-- §2. A MESSAGE IS A CELL PRODUCT; ITS NORM IS P(message); ITS LENGTH IS SHANNON
-- ═══════════════════════════════════════════════════════════════════

/-- Encode a message (a list of symbol-cells) as the cell product. -/
def encode : List NCell → NCell
  | []      => nOne
  | z :: zs => nmul z (encode zs)

/-- **The norm of the encoded message is the product of the symbol norms** = `P(message)`
    (the probabilities multiply). Rides `nNorm_mul`. -/
theorem nNorm_encode (zs : List NCell) : nNorm (encode zs) = (zs.map nNorm).prod := by
  induction zs with
  | nil => simp [encode, nNorm_one]
  | cons z zs ih => simp [encode, nNorm_mul, ih]

/-- **The message code length is the sum of the symbol code lengths — the Shannon length.**
    `info (encode message) = ∑ info symbol`. Derived purely from the Null cell's composition norm —
    no table, no Huffman/FSE. Additivity, not optimality: that this length is the shortest any code
    achieves is Shannon's source-coding theorem, and no lower bound over codes is stated here. -/
theorem info_encode (zs : List NCell) (h : ∀ z ∈ zs, nNorm z ≠ 0) :
    info (encode zs) = (zs.map info).sum := by
  induction zs with
  | nil => simp [encode]
  | cons z zs ih =>
      have hz : nNorm z ≠ 0 := h z (by simp)
      have hrest : ∀ w ∈ zs, nNorm w ≠ 0 := fun w hw => h w (by simp [hw])
      have hprod : nNorm (encode zs) ≠ 0 := by
        rw [nNorm_encode]
        exact List.prod_ne_zero (by simpa using fun w hw => hrest w hw)
      rw [encode, info_additive z (encode zs) hz hprod, ih hrest]
      simp

-- ═══════════════════════════════════════════════════════════════════
-- §3. WHY THE NULL CELL — DIAGONAL (INDEPENDENT) MULTIPLICATION IN THE IDEMPOTENT BASIS
-- ═══════════════════════════════════════════════════════════════════

-- e₊/e₋ and their idempotent/orthogonality laws live in NullCell.lean (the primitive).

/-- **Multiplication is diagonal in the idempotent basis**: `e₊` and `e₋` multiply
    independently (`ePlus_idem`/`eMinus_idem`/`e_orthogonal`), so a state `a·e₊ + b·e₋`
    multiplies channel-wise — the product measure of independent symbols, the algebraic root of
    "code length adds over independent symbols."

    In the idempotent coordinates `z = a·e₊ + b·e₋ = ((a+b)/2,(a−b)/2)`, the norm is the
    **product** `N(z) = a·b`. So `info` is `−log₂(a·b) = −log₂ a − log₂ b`: the two independent
    channels' code lengths simply add. -/
theorem nNorm_idem (a b : ℝ) : nNorm ((a + b)/2, (a - b)/2) = a * b := by
  simp [nNorm]; ring

-- ═══════════════════════════════════════════════════════════════════
-- §4. THE HEADLINE
-- ═══════════════════════════════════════════════════════════════════

/-- **The Null cell is the entropy coder (headline).** The lossless encoder is the Non-Being cell
    read by norm:

    1. a symbol of probability `p` is a cell element of norm `p`;
    2. a message is the cell **product** (`encode`), whose norm is `∏ pᵢ = P(message)`
       (`nNorm_encode`);
    3. its code length `info = −log₂ N` is **additive** over symbols (`info_additive`), so the
       message length is `∑ −log₂ pᵢ` = the **Shannon length** (`info_encode`), with no coding
       table. Additivity is the conjunct; optimality is Shannon's theorem and is not stated here;
    4. multiplication is **diagonal** in the idempotent basis (`e_orthogonal`, `nNorm_idem`):
       independent symbols multiply as independent channels — the product measure.

    Entropy coding is `x = B/U` (the composition norm) on the cell of Non-Being. The Being cell
    rotates (a reversible transform, norm-preserving — the decorrelating stage); the Null cell
    factors and collapses toward its cone (entropy + quantization — the coding stage). The two
    cells are the two halves of a codec; the trinity writes the encoder. -/
theorem null_cell_is_the_entropy_coder :
    (info nOne = 0)
      ∧ (∀ z w : NCell, nNorm z ≠ 0 → nNorm w ≠ 0 → info (nmul z w) = info z + info w)
      ∧ (∀ zs : List NCell, nNorm (encode zs) = (zs.map nNorm).prod)
      ∧ (∀ zs : List NCell, (∀ z ∈ zs, nNorm z ≠ 0) → info (encode zs) = (zs.map info).sum)
      ∧ (nmul ePlus eMinus = (0, 0)) :=
  ⟨info_one, info_additive, nNorm_encode, info_encode, e_orthogonal⟩

end CrossUniverse
end Sanguine

end
