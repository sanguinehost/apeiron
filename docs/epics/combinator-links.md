# Combinator links for the Decode Atlas lenses

## Intent

The Decode Atlas lens survey (`docs/evidence/atlas-lens-corpus.md`) found no combinator-lens content for
five objects: the RoPE torus, the sphere and its rotations, the quantisation lattice, the Gibbs measure
and the weight rectangle. It also found no double-cover theorem (`Spin(2) → SO(2)`) in either corpus.
This epic states and proves the missing links. It reuses the pinned canonical calculus
`Apeiron.Combinator.Basic` (`Comb`, `Step`, `Reduces`, `bracket`, `combinatory_completeness`; confluence
in `Apeiron.Combinator.Confluence`), imported through the `apeiron` package that `proof/corpus/lakefile.lean`
requires.

- Success: each link is a named theorem in `proof/corpus/Nous/Combinator/`, registered in
  `AxiomAudit.lean`, built under the closed axiom cone, with a mutation control that turns it red.
- Two kinds of link, named honestly:
  - combinator terms (Church numerals, `B`, `W`) with reduction theorems in `Comb`;
  - fold links, where an operation is a monoid homomorphism from lists, so every split and every
    merge tree gives the same result (Bird and Meertens' list homomorphisms).
- Not a goal: changing the executing path, the extraction or the canonical corpus.
  `Substrate/` is vendored and stays read-only.

## Spec

| module | object | headline content |
|---|---|---|
| `Nous/Combinator/Church.lean` | base | `church n` as a `Comb` term with `church n f x ↠ fⁿ x`; addition as composition; doubling `church (2n) f x ↠ church n f (church n f x)`; `B` and `W` with their reductions (`W x y ↠ x y y`) |
| `Nous/Combinator/ListHom.lean` | lattice, rectangle, Gibbs | list homomorphism `h (l ++ m) = h l * h m` ⇒ any consecutive partition and any merge tree agree; instances: 32-aligned `quant` (`ColumnCover.quant_locality`), column-cover sums, the online-softmax merge (`osFuse_is_add`, `osFuse_grouping_free`) |
| `Nous/Combinator/Orbit.lean` | torus | the first-order reading of `church n R x` under `R ↦ rotation by θ` is `ρ(nθ)·x` (`rope_orbit`); on a closed ring (`R^N = id`) numerals `n` and `n+N` read the same; the turn ring `Z/2^64` as the instance |
| `Nous/Combinator/Doubling.lean` | sphere | Sylvester/Hadamard of order `2^k` as `church k` of the doubling step (`sylvester_mulVec_butterfly`); the add count `k·2^k` from the recurrence `C(k+1) = 2·C(k) + 2^(k+1)` |
| `Nous/Combinator/DoubleCover.lean` | spinors | squaring is `W mul`; on the unit circle, squaring is onto with fibres `{z, −z}` (covering degree 2 of `Spin(2) → SO(2)`); a Gaussian integer squared gives the exact rational rotation `((a²−b²), 2ab)/(a²+b²)` |

## Plan

| wave | task | file (owner) | depends on | gate | control |
|---|---|---|---|---|---|
| 1 | CL-T1 | `Nous/Combinator/Church.lean` | — | `lake build Nous.Combinator.Church` | drop the `S` in `W`'s term, or shift a numeral |
| 1 | CL-T2 | `Nous/Combinator/ListHom.lean` | — | `lake build Nous.Combinator.ListHom` | weaken the homomorphism hypothesis, or mutate a witness |
| 2 | CL-T3 | `Nous/Combinator/Orbit.lean` | CL-T1, `Numeric/Rope/Turns.lean` | `lake build Nous.Combinator.Orbit` | off-by-one in the ring period |
| 2 | CL-T4 | dropped | — | — | — |
| 2 | CL-T5 | `Nous/Combinator/DoubleCover.lean` | CL-T1 | `lake build Nous.Combinator.DoubleCover` | drop `−z` from the fibre |

CL-T4 (Hadamard as `church k` of the doubling step) is dropped: the Hadamard map is off the decode path on both models, and the link licenses no schedule or byte saving. Status: CL-T1, CL-T2 and `Numeric/Rope/Turns.lean` are registered in `AxiomAudit.lean` (build exit 0, 9063 jobs).

Worker: `gemini-flash` (AGENTS §9.1). The lead registers each module's headlines in `AxiomAudit.lean`
and rebuilds it at the wave barrier.
