# The Song correspondence

Song is a separate project: an operating system written assembly-up in a language called Slang. It
shares no code with this crate and links no Rust or Aeneas runtime. What the two share is the
canonical definition, `R[x]/(x² − σ)` and the table over it.

## What is here

`proof/corpus/Slang/` holds the Lean side of the Slang source-refinement chain: syntax, a machine
model, a decoder, address and table refinements, and a source audit. `proof/corpus/Icarus/` is a
SHA-256 library over the same carrier. `proof/corpus/Nous/` is the model-layer library: attention, routing and decode over the three faces. They are part of this corpus and build with it.

| directory | contents |
|---|---|
| `Slang/` | the Lean machine, decoder, address and table refinements of the Slang chain |
| `Icarus/` | the SHA-256 round step realised over the trichotomy carrier, with a keystone theorem and a cost model |
| `Nous/` | attention, routing and decode over the three faces |

## A stricter cone

The Rust bridge accepts `propext`, `Classical.choice` and `Quot.sound`. The Slang chain accepts
none. Every registered declaration, instance, witness and headline must audit with an empty axiom
set. `noncomputable` and quotient-based representations such as `Finset` and `ZMod` are rejected.

```sh
python3 tools/slang/zero_axioms.py --stage <stage> --controls
```

`proof/corpus/Slang/Audit/Empty.lean` defines the rejecting command. It resolves the name, rejects
an `axiom`, checks the registered type and collects the transitive axioms with
`Lean.collectAxioms`.

## What correspondence needs

Two implementations with different layouts can correspond, but only under a matched admission
relation. The profile fixes four things.

1. **Representation.** Each layout denotes the canonical object, with the domain stated.
2. **State and effects.** Each state transition denotes the canonical transition.
3. **Admission, success and refusal.** The matched domain, each side's bounds, and what each does on unsupported input and on resource exhaustion.
4. **Resources.** Allocation, time and memory are stated.

A worked example, a differential test, identical names or identical layouts do not make a
correspondence. Rust and Slang have different layouts and different proofs of the same object.

The cross-implementation correspondence is **not established**. Each side has its own refinement
to the canonical definition. The required result, one theorem relating the two on a matched
domain, is open.
