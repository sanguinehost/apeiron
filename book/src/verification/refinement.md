# Refinement

`proof/corpus/` (package `apeironWordBridge`) depends by path on `proof/canonical/` (package
`apeiron`). The canonical package holds the mathematics, generic over a commutative ring. The
bridge package holds the statements about extracted Rust.

## The cell

`WordRefinement.lean` proves, over `Apeiron.Trichotomy.AnyRing` instantiated at `ZMod (2^64)`:

| theorem | statement |
|---|---|
| `word_mul_denote` | the extracted `Cell::mul` equals `ringMul` |
| `word_norm_denote` | the extracted `Cell::norm_form` equals `ringNormForm` |
| `word_norm_mul` | norm-form multiplicativity transferred to words |
| `word_mul_conj` | `z · conj(z)` is `(N(z), 0)` transferred to words |

$$\mathrm{denote}\big(\texttt{Cell::mul}(a,\sigma,b)\big) = \mathrm{ringMul}\big(\mathrm{denote}\,\sigma,\ \mathrm{denote}\,a,\ \mathrm{denote}\,b\big)$$

- **say:** the extracted multiplication, read as ring elements, is the canonical multiplication.
- **means:** every algebraic theorem about `ringMul` holds for the word cell at every `σ`, including the dial being carried as a value.
- **vars:** `a, b` are word cells; `σ` is the dial; `denote` maps words to `ZMod (2^64)`.
- **apeiron:** `word_mul_denote`; the canonical side is `Apeiron.Trichotomy.AnyRing.ringMul`.
- **physics:** none.

All four are total equalities. There is no `sorry` and no reduced domain, because wrapping
arithmetic cannot fail.

## Operations that can fail

The scalar carrier step, the exact-half flux and the three-cell line use checked arithmetic and
refuse an unrepresentable intermediate or an odd face difference as `none`. Their theorems are
stated on the success branch with the representability hypotheses, and each has a refusal witness
and a success witness, so the partiality is shown to be real and the success branch is shown to be
inhabited. `proof/README.md` lists the theorems.

## The table

The canonical table is `Apeiron.Table.HMap` with `Apeiron.Table.get`, `Apeiron.Table.insertWith`
and `Apeiron.Table.build`. Two layouts have been related to it.

| layout | status |
|---|---|
| fixed-shard `Table<K, V, H, N>`: one home shard, a `ShardFull` refusal, no probing | proved; revision-bound historical evidence, not a proof of the current layout |
| generic growable `apeiron::Table` (`src/map/table.rs`) | retires once the executed body is verified; no new work targets it |
| executed `freq::Table` (`src/map/hashtrinity/`) | the refinement target; the extraction roots are the `word_hashtrinity_*` and `word_coordmap_*` wrappers; refinement to the canonical map is open |

The executed body is the target because the measured winner is the working artifact. Proving the
slower generic table would certify a loss. The refinement is stated against the body that runs and
is benchmarked.

## What this does not claim

- Not the real-ordered physics. The bridge imports the ring-generic closure, so it claims no nonnegativity, no ordering and no kinetic-energy reading for wrapping words.
- Not totality of the checked operations. They refuse.
- Not that every integer state is iterable, or that an updated value stays a norm-form square.
- Not any property of rustc or LLVM code generation, the machine code or a GPU backend.
