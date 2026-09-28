# AI-track re-anchor: trichotomy-first model surface

Merged epic record: `intent/ai-track-reanchor.md`, `spec/ai-track-reanchor.md`, `plan/ai-track-reanchor.md` (one file per epic; see `docs/README.md`).

## Intent — AI-track re-anchor: trichotomy-first model surface

- **Owner:** lead
- **Source revision:** `74c037b`; working tree dirty (stage-2 vendoring, Read composition; see `docs/archive/handoff-openai.md`)
- **Status:** draft
- **Spec:** `docs/epics/ai-track-reanchor.md` (to be written)
- **Plan:** `docs/epics/ai-track-reanchor.md` (to be written)

### Problem

Stage 2 re-established 48 historical modules at the pinned identity. None imports the canonical
object: `rg '^import Apeiron'` over the vendored tree returns nothing, and the canonical-looking
references are prose anchors. The neural headline modules (`MirasCountLM`, `KneserNey`,
`CountLMFloor`, `SparseAttentionBudget`, `MLPrimitivesBridge`, `TrainingCollapse`, `FrontierLadder`)
state count-LM, smoothing, entropy-floor and ridge/Newton mechanisms with no reduction to
`R[x]/(x²−σ)`. The programme thesis — boundedness, O(1) training, finetuning and inference,
test-time/forward-only learning — is a claim about the algebra, so it cannot be carried by those
statements. `lean-dojo/TorchLean` at `de3192df4779877ceb65cfe470a4d4ce9d480a5b` is the
authoritative reference for the *scope* of the surface to rebuild (shape-indexed tensors, autograd,
training, certificate verification; arXiv 2602.22631); it is read, not adopted, and it deliberately
says nothing about the trichotomy `R[x]/(x²−σ)`.

### Vocabulary and attribution

Write standard mathematics: the most common name, the standard notation. The basis is the quotient
`R[x]/(x²−σ)`; its faces are elliptic (`σ<0`, the complex numbers), parabolic (`σ=0`, the dual
numbers) and hyperbolic (`σ>0`, the split-complex numbers). Quote a borrowed object as originator(s)
then reference; the reference is retrieved and read before it is called evidence — a citation alone
is a pointer.

| Object | Standard name | Notation | Originator(s), then reference |
|---|---|---|---|
| Basis | quotient of the polynomial ring | `R[x]/(x²−σ)` | Kronecker (1882); Dedekind (1871) for ideals and quotients |
| `σ<0` | complex numbers | ℂ | Bombelli (1572); Euler (1777) for `i`; Gauss (1831) for the plane reading |
| `σ=0` | dual numbers | `R[ε]/(ε²)` | Clifford (1873); Study (1903) for the kinematic reading |
| `σ>0` | split-complex (hyperbolic) numbers | `R[j]/(j²−1)` | Cockle (1848) |
| Norm form | norm form of the quadratic algebra | `a²−σb²` | Gauss, *Disquisitiones Arithmeticae* (1801); Dedekind (1871) |
| Split exponential | hyperbolic functions | `e^{jφ} = cosh φ + j sinh φ` | Riccati (1757); Lambert (1768) |
| Weighting | softmax (Boltzmann–Gibbs) | `e^{x_i}/Σ_j e^{x_j}` | Boltzmann (1877); Gibbs (1902); Bridle (1990) for the name |

The pre-trichotomy narrative vocabulary is deprecated and must not be reused for any object, module
or model; the translation table is in
[`docs/epics/security-algebra-rebase.md`](security-algebra-rebase.md). `Cell<i64>` and `RingCell`
are Lean identifiers; historical module paths appear as code-span paths only, never as names. The
spec carries the complete attribution table with each reference's retrieval status.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Keep porting the historical ML modules; link at stage 3 | No reduction statement exists for the count-LM/entropy/ridge family; the deferred link is not a claim that can be stated, let alone graded |
| Use the archived split-algebra modules (`Substrate/Algebra/NullCell/…`) as the canonical base | They are parallel split-ℂ definitions, not instances of `Apeiron.Trichotomy.AnyRing.RingCell`; using them as base forks the object of record |
| Adopt TorchLean as a pinned Lake dependency | Puts a foreign semantics layer under our proofs and inherits its runtime trust boundaries and toolchain cadence; the guarantees we want are claims about the algebra, so the surface must be derived from `RingCell`/`Table`, not wrapped around an external model |
| Mirror TorchLean's API and module structure | Only the scope is matched; the construction follows the algebra's own structure, and every reusable piece is generalised (rings, folds, tensors, autograd) where it is useful beyond the model |
| Keep historical names for continuity | A name carries the claim; a name that does not resolve to the canonical object reports a link that does not exist |

### Completion condition

The native surface and the O(1) model are kernel-checked on one identity, with every headline
mentioning only canonical definitions plus the native surface:

```sh
cd proof/corpus && lake build Nous && lake build AxiomAudit
```

### Invariants

1. Every new model-layer definition reduces to `Apeiron.Trichotomy.AnyRing.{RingCell, ringMul, ringNormForm}` or `Apeiron.Table.*`; no parallel carrier is introduced.
2. The vendored historical tree gains no dependents and receives no new names; it is reference material at recorded digests.
3. The surface is rebuilt in-tree on the canonical algebra. TorchLean is reference only: no build dependency, and every consulted statement is cited by file and revision at `de3192df…` (grade `anchor`, not evidence).
4. Closed axiom cone `{propext, Classical.choice, Quot.sound}`; no `sorry`/`axiom`; every Lean command bounded.

### Verification hypotheses

| Claim | Grade | Evidence (command or theorem) |
|---|---|---|
| The native surface builds at the programme identity | tested | `cd proof/corpus && lake build Nous` |
| Generalisable pieces are ring-generic and reusable outside the model | proved | parameterized definitions/instances named per piece |
| Attention weighting is the split face's channel ratio | proved | named bridge theorem over `ringUpper_channel`/`ringLower_channel` and `ringMul` |
| Every model op is a `ringMul`/`ringNormForm`/`Table` instance | proved | per-op named theorems; `#audit` closed cone |
| O(1) training step, stated as an exact recurrence or resource theorem | proved | named theorem with its recurrence; no bare big-O |
| O(1) inference and finetuning bounds | proved | named theorems |
| Test-time / forward-only learning soundness | proved | named theorems over the `ringMul`/`insertWith` update |
| Reference tree unchanged | anchor | `sha256sum -c proof/corpus/Substrate/adapted.sha256` |

### Non-goals

- Porting or extending `MirasCountLM`, `KneserNey`, `CountLMFloor`, `SparseAttentionBudget`, `MLPrimitivesBridge`, `TrainingCollapse`, `FrontierLadder`.
- Reimplementing TorchLean's runtime, CUDA, PyTorch interop or certificate tooling.
- Changing the canonical corpus definitions or the extraction/bridge.
- Any publication before rebase and end-to-end proof.
- Performance claims without a named command, machine, metric and interval.

### Trust boundary

The evidence will not reach: TorchLean's runtime providers and their boundaries (CPU/CUDA/LibTorch,
per its `docs/TRUST_BOUNDARIES.md`) — its code is not in our build; Rust/LLVM codegen, the kernel ABI,
GPU/SPIR-V, the `unsafe` container, tokenizer implementations, training-dynamics empirics, or any
statement about a deployed model. Consulting a TorchLean statement is an anchor, not evidence.

### Open questions (resolved in the spec `docs/epics/ai-track-reanchor.md`)

- Surface boundary: resolved — `Nous/Core` is ring-generic; `Nous/Model` is AI-specific.
- Tree location: resolved — `proof/corpus/Nous/…` with its own `lean_lib`, named `Nous` (νοῦς;
  Anaxagoras, DK 59 B 12; Aristotle, *De Anima* III.4–5; homage, not a claim).
- Architecture: resolved — core first, then streaming state plus table memory plus channel weighting;
  the table-free case is an instance, not a competitor.
- Vendored ML modules: deferred with a trigger — retained checksum-verified until `lake build Nous`
  lands, then a separate task with a build-cost measurement and a reachability check.

## Specification — AI-track re-anchor: trichotomy-first model surface

- **Owner:** Apeiron lead
- **Intent:** [`docs/epics/ai-track-reanchor.md`](ai-track-reanchor.md)
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty (stage-2 vendoring, Read composition; see `docs/archive/handoff-openai.md`)
- **Status:** approved (lead review applied; plan pending in `docs/epics/ai-track-reanchor.md`)

```text
SCOPE:      docs/epics/ai-track-reanchor.md; AI-track specification and the lead-review corrections;
            no source, proof, plan or gate modification
REGISTER:   planning / specification for the AI-track outcome (algebra/refinement); exclusions: no
            runtime dependency, no TorchLean pin, no task allocation
CORPUS:     Apeiron.Table.{HMap,get,insertWith,build} and
            Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm,ringOne,ringMul_comm,ringMul_assoc,ringNormForm_mul},
            Apeiron.Trichotomy.ChannelsAnyRing.{ringUpper,ringLower,RingChannel,ringUpper_channel,ringLower_channel},
            Apeiron.Trichotomy.Basic.{normForm_eq_zero_iff_of_neg,nilpotent_of_disc_zero,zeroDivisorPair_of_disc_one}
            at resolved dependency revision 880aa57e47e782ed7a9d9ee1974cfc87360b446b (package `apeiron`
            in proof/corpus/.lake/packages/apeiron; the sibling tree ../sanguine-apeiron/proof/Apeiron/
            is at the same revision); reference only TorchLean at /tmp/opencode/torchlean-read @
            de3192df4779877ceb65cfe470a4d4ce9d480a5b
REFERENCE:  docs/epics/ai-track-reanchor.md, docs/direction.md; TorchLean NN/API.lean,
            NN/Spec/Core/{Shape,Scalar,Sequence}.lean, NN/Spec/Autograd/AutogradSpec.lean,
            NN/Proofs/Autograd/FDeriv/*
GATE:       doc checks, numeric wall-clock budget 60 s (§10): (a)
            `rg -c 'timeout 540' docs/epics/ai-track-reanchor.md` = 2, (b)
            `rg -n -i 'null[ ]transformer|null[ ]cell|being[ ]cell' docs/epics/ai-track-reanchor.md`
            = no match (patterns bracket-spaced so this gate line cannot match itself); observed both
            as stated after lead review. Legacy `Cell`/`mul`/`normForm` occur only inside verbatim
            quotes of the ℝ-specialised Basic.lean statements.
GRADE:      anchor/planning; not evidence. Lead review corrections: retrieval statuses set to
            `anchor` except the TorchLean pin, contract evidence split into canonical (exists at
            880aa57e…) versus new surface targets, gate commands put in the canonical lane-lock
            form, the channel-ratio precondition stated as unit of `2 * z.1`, the probe-width bound
            removed from the abstract model contract, and the four open decisions taken.
```

### Scope and admissible domain

This specification re-anchors the AI track directly in Apeiron's canonical object: the quotient ring
`R[x]/(x²−σ)`, its two-component carrier `RingCell R`, and the growable flat/spill table
`Apeiron.Table`. It supersedes the historical count-LM/entropy/ridge modules, which remain reference
material; the native surface is rebuilt in Lean 4 without external dependencies. TorchLean
(`de3192df4779877ceb65cfe470a4d4ce9d480a5b`) is consulted solely to delimit the functional scope
(shape-indexed tensors, state-threading sequence traversals, pure VJP operator contracts), never as a
runtime or build dependency.

The surface is factored into two disjoint namespaces:

1. **Ring-generic core (`Nous/Core/`, no AI-specific names):**
   - **Shapes:** shape descriptors `Shape := List Nat`, outermost dimension first; `rank : Shape → Nat`
     and `size : Shape → Nat`; decidable compatibility predicates `CanBroadcastTo`/`BroadcastTo`.
   - **Tensors:** `Tensor R s := Fin (size s) → R` over `[CommRing R]` with pointwise addition,
     subtraction, multiplication, scalar scaling, and coordinate operations.
   - **Folds:** forward accumulator `mapAccum` threading a state over `0 … n-1` and returning
     `{ results : Array Result // results.size = n }`, and reverse `mapAccumRight` over `n-1 … 0`.
   - **Operator contracts:** `OpSpec R σ τ` with `forward : Tensor R σ → Tensor R τ` and the
     vector-Jacobian product `backward : Tensor R σ → Tensor R τ → Tensor R σ`; composition `>>>`
     with the reverse-mode chain rule, no graph or runtime engine.

2. **Trichotomy model layer (`Nous/Model/`, depends on the core and the canonical corpus):**
   - **Carrier:** canonical `RingCell R` with parameter `σ : R`.
   - **Token state:** `ringMul σ` advances point states; `ringOne` is the unit.
   - **Norm form:** `ringNormForm σ z = z.1 ^ 2 - σ * z.2 ^ 2` as the energy/readout.
   - **Table memory:** `Apeiron.Table.HMap K (RingCell R)` with `insertWith (ringMul σ)`.
   - **Channel projections:** `ringUpper r`/`ringLower r` on a root witness `r ^ 2 = σ`; the design
     hypothesis uses their ratio as the weighting function in place of softmax.

| Item | Value |
|---|---|
| Operation | Native tensor, traversal, VJP, and trichotomy-backed model surface |
| Domain | `[CommRing R]`; shapes `s : List Nat`; points `z, w : RingCell R`; keys `K` with `DecidableEq K` |
| Totality | Tensor, shape, fold and ring operations are total; `Table.get` returns a value under the map model; `insertWith` is a total map update |
| Preconditions | Parameter `σ : R`; channel decomposition needs a named witness `r` with `r ^ 2 = σ`; the channel ratio needs `IsUnit (2 * z.1)` |
| Postcondition | Every model operation denotes a composition of `ringMul`, `ringNormForm`, channel projections, and `Apeiron.Table` operations; new headlines audit in `{propext, Classical.choice, Quot.sound}` |

No silent domain reduction: the model is stated over `[CommRing R]`; results that need an ordered or
invertible setting name that setting (`ℝ` for the negative/positive face lemmas, `IsUnit` for the
ratio).

### Representation relation

Surface symbols reduce to canonical declarations at revision `880aa57e…`. `RingCell`, `ringMul`,
`ringNormForm` and `ringOne` are `abbrev`s, so each link is definitional and checked by `rfl`/`simp`
in the surface module that consumes it.

| Surface Symbol | Canonical Declaration | Module | Reduction |
|---|---|---|---|
| `RingCell R` | `RingCell R` (`abbrev`, needs `[CommRing R]`) | `Apeiron.Trichotomy.AnyRing` | definitionally `R × R` |
| `ringMul σ z w` | `ringMul σ z w` (`abbrev`) | `Apeiron.Trichotomy.AnyRing` | definitionally `(z.1 * w.1 + σ * z.2 * w.2, z.1 * w.2 + z.2 * w.1)` |
| `ringNormForm σ z` | `ringNormForm σ z` (`abbrev`) | `Apeiron.Trichotomy.AnyRing` | definitionally `z.1 ^ 2 - σ * z.2 ^ 2` |
| `ringOne` | `ringOne` (`abbrev`) | `Apeiron.Trichotomy.AnyRing` | definitionally `(1, 0)` |
| `ringUpper r z` | `ringUpper r z` (`def`) | `Apeiron.Trichotomy.ChannelsAnyRing` | definitionally `z.1 + r * z.2` |
| `ringLower r z` | `ringLower r z` (`def`) | `Apeiron.Trichotomy.ChannelsAnyRing` | definitionally `z.1 - r * z.2` |
| `RingChannel σ φ` | `RingChannel σ φ` (`def`) | `Apeiron.Trichotomy.ChannelsAnyRing` | four-clause read: add, scale, multiply, unit |
| `HMap K V` | `HMap K V` | `Apeiron.Table.Basic` | map model of the flat/spill table |
| `insertWith` | `insertWith` | `Apeiron.Table.Basic` | `HMap` update under `ringMul σ` |

Rules:

1. No parallel carrier: `RingCell R` is the single canonical carrier of point states.
2. TorchLean modules are neither imported nor mirrored; the pin delimits statement scope only.
3. Every model operation denotes an explicit composition of `ringMul`, `ringNormForm`, channel
   projections and `Apeiron.Table` operations; a construct that does not is out of scope.
4. `Cell`, `mul` and `normForm` appear only where a statement of the ℝ-specialised `Basic.lean` layer
   is quoted verbatim in the contracts table; every new object uses the any-ring names.

### Contracts

Rows marked **canonical** resolve at `880aa57e…` today. Rows marked **target** are new surface
declarations to be written in the plan's waves; the name is fixed here, the proof is not claimed.

| Contract | Statement | Grade | Evidence |
|---|---|---|---|
| `ringMul_comm` | `∀ (σ : R) (z w : RingCell R), ringMul σ z w = ringMul σ w z` | proved | canonical `Apeiron.Trichotomy.AnyRing.ringMul_comm` |
| `ringMul_assoc` | `∀ (σ : R) (z w v : RingCell R), ringMul σ (ringMul σ z w) v = ringMul σ (ringMul σ w v)` | proved | canonical `Apeiron.Trichotomy.AnyRing.ringMul_assoc` |
| `ringNormForm_mul` | `∀ (σ : R) (z w : RingCell R), ringNormForm σ (ringMul σ z w) = ringNormForm σ z * ringNormForm σ w` | proved | canonical `Apeiron.Trichotomy.AnyRing.ringNormForm_mul` |
| `elliptic_no_nontrivial_zero_divisor` | `{σ : ℝ} (hσ : σ < 0) (z : Cell) : normForm σ z = 0 ↔ z = 0` (verbatim, ℝ layer) | proved | canonical `Apeiron.Trichotomy.Basic.normForm_eq_zero_iff_of_neg` |
| `parabolic_nilpotent` | `mul 0 (0, 1) (0, 1) = 0 ∧ ((0, 1) : Cell) ≠ 0` (verbatim, ℝ layer) | proved | canonical `Apeiron.Trichotomy.Basic.nilpotent_of_disc_zero` |
| `hyperbolic_zero_divisors` | `mul 1 (1, 1) (1, -1) = 0 ∧ ((1, 1) : Cell) ≠ 0 ∧ ((1, -1) : Cell) ≠ 0` (verbatim, ℝ layer) | proved | canonical `Apeiron.Trichotomy.Basic.zeroDivisorPair_of_disc_one` |
| `upper_channel_hom` | `{σ r : R} (hr : r ^ 2 = σ) : RingChannel σ (ringUpper r)` | proved | canonical `Apeiron.Trichotomy.ChannelsAnyRing.ringUpper_channel` |
| `lower_channel_hom` | `{σ r : R} (hr : r ^ 2 = σ) : RingChannel σ (ringLower r)` | proved | canonical `Apeiron.Trichotomy.ChannelsAnyRing.ringLower_channel` |
| `opSpec_compose_vjp` | `∀ (op1 : OpSpec R σ τ) (op2 : OpSpec R τ υ), (op1 >>> op2).backward x dL = op1.backward x (op2.backward (op1.forward x) dL)` | proved (target) | new `OpSpec.compose_backward`, `Nous/Core/Autograd.lean` |
| `token_step_fold` | `∀ (σ : R) (xs ys : List (RingCell R)), List.foldl (ringMul σ) ringOne (xs ++ ys) = ringMul σ (List.foldl (ringMul σ) ringOne xs) (List.foldl (ringMul σ) ringOne ys)` and `List.foldl (ringMul σ) ringOne [] = ringOne` | proved (target) | new `token_step_fold`, `Nous/Model/State.lean` |
| `memory_step_model` | `∀ (M : HMap K (RingCell R)), insertWith (ringMul σ) k v M = Function.update M k (…)`; the fixed probe-width bound is an obligation of the executing face, not of the abstract map | proved (target) | new `memory_step_model`, `Nous/Model/Memory.lean`; probe bound named `probe_width_bound`, out of this spec |
| `channelWeight_eq` | `∀ (z : RingCell R), 2 * z.1 ≠ 0 → channelWeight r z = ringUpper r z / (2 * z.1)` where `channelWeight r z = ringUpper r z / (ringUpper r z + ringLower r z)` | proved (target) | new `channelWeight_eq`, `Nous/Model/Weighting.lean` |
| `channelWeight_binary_softmax` | where `u = ringUpper r z > 0` and `l = ringLower r z > 0`, `u / (u + l) = softmax (log u) (log l)` (binary softmax) | proved (target) | new declaration; exact domain `0 < u`, `0 < l` |
| `attention_channel_form` | the attention module's normalized output equals `Σ_i (w_i / Σ_j w_j) v_i` with `w_i` the channel weights, on the module's stated factorizable domain | proved (target) | new `Nous/Model/Attention.lean` |
| `hyperbolic_decomposition` | with `r ^ 2 = σ` and `2`, `r` units, `z ↦ (ringUpper r z, ringLower r z)` is a bijection `RingCell R → R × R` intertwining `ringMul σ` with componentwise multiplication | proved (target) | new `Nous/Model/Decomposition.lean` |
| `channel_diagonal_fold` | the token fold and `memoryStep` factor through `hyperbolic_decomposition` into independent scalar operations per channel | proved (target) | new `Nous/Model/Decomposition.lean` |

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| Incompatible shapes in a binary operation | Static rejection at elaboration | `¬ (s1 = s2)` and `¬ CanBroadcastTo s1 s2` |
| Empty shape product | `Tensor R []` has exactly one element; operations are total | `size [] = 1` |
| Root witness absent for `σ` | Channel decomposition unstatable | `¬ ∃ r : R, r ^ 2 = σ` |
| Denominator not invertible | Channel ratio unavailable | `¬ IsUnit (2 * z.1)` |
| Table reservation refusal (executing face) | Insertion refuses; no partial write | capacity/probe condition of the growable table (out of this spec) |

### The O(1) model starting hypothesis (grade `anchor`)

This section is a design hypothesis at `anchor` grade: the identities hold once the named definitions
exist, and the operation counts are combinatorics over ring operations. No wall-clock, throughput,
memory, training-dynamics or equivalence claim is made here; a measured claim requires the §10
protocol (named machine, metric, interval, before/after).

**The trichotomy architecture.** While the predecessor over-specialized to the split-complex/hyperbolic case (`docs/rfcs/0001-core-architecture/RFC.md#L27`), Apeiron's foundation is `R[x]/(x^2 - \sigma)` across its full family of faces (`docs/direction.md#L24`: *the hyperbolic face is the real split specialization, not a replacement for the full σ-family*). The sequence layer unifies all three operational regimes under a single dial `Sigma`:

- **Hyperbolic face (`σ > 0`, split-complex numbers; Cockle 1848):** Two independent null channels (`ringUpper r z`, `ringLower r z`) with componentwise multiplication (`Nous/Model/Attention.lean`). The channel ratio `upper / lower = e^{2φ}` derives exponential Boltzmann-Gibbs weighting algebraically without transcendental evaluations. Non-trivial idempotents (`P^2 = P`) provide selective associative state erasure and product-key candidate pruning (`Nous/Seq/HyperbolicAddress.lean`).
- **Parabolic face (`σ = 0`, dual numbers; Clifford 1873):** Exactly one scalar channel and a nilpotent tangent `ε^2 = 0` (`Nous/Model/ParabolicAttention.lean`). The first-order truncation `exp(ε φ) = 1 + ε φ` derives native **Linear Attention / Recurrent SSMs** (Katharopoulos et al. 2020) with `K(q, k) = 1 + q^T k`, enabling exact `O(1)`-state linear accumulation (`S_t = S_{t-1} + k_t v_t^T`, `z_t = z_{t-1} + k_t`) and zero transcendental evaluation cost.
- **Elliptic face (`σ < 0`, complex numbers; Bombelli 1572, Gauss 1801):** Division algebra with positive-definite norm form `N_σ(z) = a^2 + |σ| b^2` and zero non-trivial zero divisors (`Nous/Model/EllipticAttention.lean`). Multiplicative norm preservation (`N(q · k) = N(q) N(k)`) guarantees exact isometric state preservation, driving rotary position embeddings (RoPE; Su et al. 2024; `Nous/Model/RopeOrbit.lean`) and unitary phase interference without energy dissipation.
- **Unified superstructure stage:** Assembled in `Nous/Model/TrichotomyAttention.lean`, dispatching across all three faces under a single dial `Sigma`, with linear `O(N)` streaming complexity across all three regimes (`trichotomy_streaming_linear_complexity`).

Whether local writes and dynamic face dispatch separate a useful distribution is the empirical question the falsification point below names.

**Channels and spinors.** The two channels are the eigenspaces of multiplication by the generator
`x = (0, 1)` with eigenvalues `±r`; equivalently, the light-cone coordinates of the two-dimensional
Lorentz structure: for `σ = 1`, a boost multiplies them by `e^{±φ}`, and the norm form is their
product `u * l`. A spinor reading — spinor components as square roots of the light-cone components,
`Cl⁰(1,1) ≅ R ⊕ R`, a null vector as the square of a spinor — is standard (Cartan 1938; Chevalley
1954; Lawson–Michelsohn 1989) and is an `anchor` here, not a claim: a spinor module is not part of
this specification, and the topological double cover is nontrivial only where the group is not simply
connected (the elliptic `SO(2)` case); `SO⁺(1,1)` is simply connected, so the two-channel splitting
does not by itself carry a topological spinor.

1. **Streaming state.** For `S_t, X_t : RingCell R`, `S_{t+1} = ringMul σ S_t X_t` equals
   `(S_t.1 * X_t.1 + σ * S_t.2 * X_t.2, S_t.1 * X_t.2 + S_t.2 * X_t.1)`. Counted in ring operations:
   4 multiplications, 1 scaling by `σ`, 2 additions per token; state width 2 ring elements, constant
   in the sequence length. The count is from the definitional expansion of `ringMul`, nothing else.
2. **Table memory.** `M_{t+1} = insertWith (ringMul σ) k v M_t` on `Apeiron.Table.HMap`. In the
   abstract map model this is one update plus one `ringMul`; the fixed probe-width bound `W` belongs
   to the executing growable table and is a separate obligation (`probe_width_bound`), not claimed
   here. Occupied storage is `N × 2` ring elements for `N` live keys; space is not claimed O(1).
3. **Channel-ratio weighting.** On a root witness `r ^ 2 = σ`, `ringUpper r z + ringLower r z = 2 * z.1`,
   so where `IsUnit (2 * z.1)` the ratio is `w(z) = ringUpper r z / (2 * z.1)`. The design hypothesis
   uses `w` as the weighting function. Where both channels are positive,
   `w(z) = ringUpper r z / (ringUpper r z + ringLower r z)` is the binary logistic/softmax of the
   log-channel pair `(log (ringUpper r z), log (ringLower r z))` — a standard identity, not yet
   formalized in this package, and not an approximation (logistic: Verhulst 1838; distribution:
   Boltzmann 1877, Gibbs 1902; name `softmax`: Bridle 1990). What is **not** claimed: reproduction of
   a context softmax over many keys, any attention model, or any trained capability. `ringUpper r`/`ringLower r` are `RingChannel`s, so the
   channel projections compose multiplicatively across token steps; the ratio itself is recomputed
   from them at each step and is not claimed multiplicative.
4. **Forward-only update.** Because `ringMul` is commutative and associative (`ringMul_comm`,
   `ringMul_assoc`), a token sequence composes as a monoid fold and the memory update is local. The
   design intent is that state and memory update during the forward pass with no unrolled activation
   history; whether that yields useful learning is an empirical question outside this specification.
5. **Softmax reproduction and improvement (targets).** The track owes a concrete relation to softmax,
   not an absence of one. Named targets:
   - **Binary case, exact:** where both channels are positive, `channelWeight r z` is the binary
     softmax of the log-channel pair (`channelWeight_binary_softmax`, target).
   - **Exponential structure, exact:** `ringUpper r`/`ringLower r` are multiplicative
     (`ringUpper_tokenStep`, already proved), so a fold of unit-hyperbola points multiplies its
     channels and accumulates exponent sums with one ring multiplication per item; the normalizer
     `Z = Σ_i w_i` and the weighted value sum `Σ_i w_i v_i` are ring additions, and the streaming
     state has constant width, so normalization is not recomputed over the context
     (`attention_channel_form`, target: the module's normalized output equals the softmax normal form
     `Σ_i (w_i / Σ_j w_j) v_i` on the stated factorizable domain). No random-feature kernel
     approximation is used (contrast random-feature linear attention; see the attribution table);
     the exponential structure is exact.
     What remains per item is the same exponential softmax itself requires; the claimed improvement is
     the constant-width streaming state and exact keyed retrieval through `Apeiron.Table`, to be
     measured, not assumed.
   If a named measurement on a named task and machine shows the assembled model is not competitive
   with a softmax baseline, the model hypothesis fails and is revised; the fallback is a softmax
   readout computed in `R` with the algebra carrying state and memory.

### Evidence plan

1. **Lean gates.** Every gate runs under the lane lock with the declared budget:
   ```sh
   cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous'
   ```
   Budget: 540 s per invocation; 600 s is the failure ceiling (§10); 45 min per task. The wave
   barrier additionally runs bare `lake build` and the §6 Rust set.
2. **Axiom cone.** New headlines are registered by the lead in `proof/corpus/AxiomAudit.lean` and the
   audit builds with `lake build AxiomAudit`; anything outside `{propext, Classical.choice,
   Quot.sound}` fails the build, and `sorry`/`admit` fail the module gate.
3. **Mutation controls.** In a disposable copy of `proof/corpus`, mutate the surface's own content
   definition (for example the component order in `Model/State.lean`'s token step, or the `σ` factor
   in `Model/Memory.lean`'s update) and show the named headline fails with an attributable
   diagnostic; baseline and restore show green. Canonical corpus declarations are not mutated in
   place; their own corpus audit covers them.
4. **Measurement.** None in this specification. When a throughput, latency or memory number is
   claimed, it names command, machine, metric and interval, with before/after (§10).

### Attribution table

Every borrowed object is named by its most common standard term, then the originator(s), then the
reference. **Retrieval status:** only the TorchLean pin and the canonical corpus at
`880aa57e…` are in hand (`read`); every other reference is an `anchor` until retrieved and read
(§9.3). A citation alone is a pointer.

| Object | Standard name | Notation | Originator(s) | Reference | Status |
|---|---|---|---|---|---|
| Basis | Quotient of the polynomial ring | `R[x]/(x²−σ)` | Kronecker (1882); Dedekind (1871) | Kronecker, *Grundzüge einer arithmetischen Theorie der algebraischen Grössen*, J. Reine Angew. Math. 92; Dedekind, Supplement X to Dirichlet, *Vorlesungen über Zahlentheorie* | `anchor` |
| `σ < 0` | Complex numbers (elliptic) | `ℂ` | Bombelli (1572); Euler (1777); Gauss (1831) | Bombelli, *L'Algebra*; Euler, symbol `i`; Gauss, *Theoria residuorum biquadraticorum, Commentatio secunda* | `anchor` |
| `σ = 0` | Dual numbers (parabolic) | `R[ε]/(ε²)` | Clifford (1873); Study (1903) | Clifford, *Preliminary Sketch of Biquaternions*, Proc. London Math. Soc. 4; Study, *Geometrie der Dynamen* | `anchor` |
| `σ > 0` | Split-complex numbers (hyperbolic) | `R[j]/(j²−1)` | Cockle (1848) | Cockle, *On Certain Functions Resembling Quaternions and on a New Imaginary in Algebra*, Philosophical Magazine 33 | `anchor` |
| Norm form | Norm form of the quadratic algebra | `a²−σb²` | Gauss (1801); Dedekind (1871) | Gauss, *Disquisitiones Arithmeticae*; Dedekind, Supplement X | `anchor` |
| Split exponential | Hyperbolic functions | `e^{jφ} = cosh φ + j sinh φ` | Riccati (1757); Lambert (1768) | Riccati, *Opusculorum ad res physicas et mathematicas pertinentium*; Lambert, *Histoire de l'Académie Royale des Sciences et Belles-Lettres de Berlin* | `anchor` |
| Weighting | Softmax / logistic (Boltzmann–Gibbs) | `e^{x_i}/Σ_j e^{x_j}`; binary case `u/(u+l)` | Verhulst (1838) for the logistic; Boltzmann (1877); Gibbs (1902); Bridle (1990) for the name | Verhulst, *Notice sur la loi que la population suit dans son accroissement*; Boltzmann, *Über die Beziehung zwischen dem zweiten Hauptsatze der mechanischen Wärmetheorie…*; Gibbs, *Elementary Principles in Statistical Mechanics*; Bridle, *Probabilistic Interpretation of Feedforward Classification Network Outputs* | `anchor` |
| Folds | List fold and catamorphism | `foldl`, `mapAccum` | Meertens (1986); Bird (1987); Meijer, Fokkinga, Paterson (1991) for the catamorphism | Meertens, *Algorithmics: towards programming as a mathematical activity*; Bird, *An introduction to the theory of lists*; Meijer–Fokkinga–Paterson, *Functional Programming with Bananas, Lenses, Envelopes and Barbed Wire* | `anchor` |
| Hash tables | Open addressing with probe sequences | `HMap K V` | Peterson (1957); Knuth (1963, 1973) | Peterson, *Addressing for Random-Access Storage*, IBM J. Res. Dev. 1(2); Knuth, *The Art of Computer Programming, Vol. 3: Sorting and Searching* | `anchor` |
| Automatic differentiation | Reverse-mode AD (VJP) | `OpSpec R σ τ` | Linnainmaa (1970, 1976); Speelpenning (1980); Griewank (1989) | Linnainmaa, *Taylor expansion of the accumulated rounding error*; Speelpenning, *Compiling Fast Partial Derivatives of Functions*; Griewank, *On Automatic Differentiation* | `anchor` |
| Linear attention / random features | kernel-form attention | `φ(q)ᵀ Σ_i φ(k_i) v_iᵀ` | Katharopoulos et al. (2020); Choromanski et al. (2021) for random-feature softmax approximation | Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*; Choromanski et al., *Rethinking Attention with Performers* | `anchor` |
| Spinors | spin representation; square root of a null vector | light-cone components as squares of spinor components; weights `±φ/2` | Cartan (1938); Chevalley (1954) | Cartan, *Leçons sur la théorie des spineurs*; Chevalley, *The Algebraic Theory of Spinors*; Lawson–Michelsohn, *Spin Geometry* (1989) | `anchor` |
| Scope reference | Verified tensor/autograd specification scope | `Spec.OpSpec`, `Tensor` | TorchLean team | *TorchLean: Verified Deep Learning in Lean 4*, arXiv:2602.22631; pin `de3192df4779877ceb65cfe470a4d4ce9d480a5b` at `/tmp/opencode/torchlean-read` | `read` |

### Open questions and lead decisions

1. **Library name — resolved.** `Nous` (νοῦς), replacing the working name `ApeironAI`; originators
   Anaxagoras (DK 59 B 12) and Aristotle, *De Anima* III.4–5; reference status `anchor` (not
   retrieved). The name is a homage, not a claim, and the trust boundary below is unchanged. Epic
   task IDs keep the track label `AI` (`APEIRON-AI-*`).
2. **Tree location — resolved.** `proof/corpus/Nous/` as a dedicated `lean_lib Nous` in
   `proof/corpus/lakefile.lean` (lead-owned root edit), importing the pinned `apeiron` package at
   `880aa57e…`. One package, one build lane, no new toolchain.
3. **Architecture — resolved sequence.** Ring-generic core first; the model layer second as streaming
   state plus table memory plus channel weighting. The pure-streaming case is the table-free instance
   of the same model, not a competing architecture; a comparison between variants is a measurement
   task that starts only after both exist.
4. **Generalisation boundary — resolved.** `Nous/Core/` is ring-generic with no model names;
   `Nous/Model/` is AI-specific and depends on Core and the canonical corpus. A Core module
   importing a Model module is a defect.
5. **Frozen reference tree — deferred with a trigger.** `SubstrateVendored` stays checksum-verified
   and unchanged; its disposition is reconsidered only after `lake build Nous` lands, in its own
   task carrying a build-cost measurement and a consumer-reachability check.

### Size limits, lane ownership, and trust boundary

- **Size limits:** every owned Lean file ≤1,500 physical lines; ≤12 immediate source files per
  directory. Inventory external trust separately; this spec's TCB is its own modules plus the pinned
  corpus.
- **Lane ownership:** shared caches (`proof/corpus/.lake`, `proof/.lake`, `target/`) are one lane,
  serialized with `flock /tmp/opencode/apeiron-lane.lock`; the lead owns `proof/corpus/lakefile.lean`
  and `proof/corpus/AxiomAudit.lean`.
- **Trust boundary.** This specification and its evidence do not cover:
  - TorchLean runtime engines, PyTorch/LibTorch FFI, or CUDA kernels.
  - rustc/LLVM compilation, target ABI, or GPU/SPIR-V emission.
  - The Rust `unsafe` container; no Rust refinement claim is made for the new surface.
  - Any equivalence between channel-ratio weighting and softmax attention, and any training,
    convergence, or capability claim.
  - Wall-clock performance; the operation counts above are combinatorial, not measured.
  - Tokenizers, datasets, and the frozen 48-module reference tree (out of the new cone).

## Plan — AI-track re-anchor: native surface, then the O(1) model

- **Owner:** Apeiron lead
- **Intent:** [`docs/epics/ai-track-reanchor.md`](ai-track-reanchor.md)
- **Spec:** [`docs/epics/ai-track-reanchor.md`](ai-track-reanchor.md) (`4ce40919049b7bdac10cd19dfe4b89d1b9182820aa10886aff5ee13269e9c8e9`)
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty (`docs/archive/handoff-openai.md`)
- **Status:** approved (lead); no commit or push without explicit user authorization
- **Priority note:** `APEIRON-AI-W4-T1` (assembly) has landed; W4-T2 face instances and W4-T3
  attention are paused. User direction moves the near-term priority to running an existing model under
  the proven Lean + Aeneas Rust structure and comparing against llama.cpp (new parity epic; the old
  `fold` engine inventory is in flight). The landed Nous headlines are registered and the cone is
  closed (`AxiomAudit.lean` `6273f809…`, 3889 jobs).

### Goal and completion condition

The wave-level barrier: every task gate exit 0 with its negative control run and restored, no task
touched another task's file, and the AI lib plus the whole bridge build green on one source identity:

```sh
cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous'
cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build'
cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build AxiomAudit'
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
proof/regen.sh --check
```

Completion condition: `Nous/Core/*` and `Nous/Model/*` compile warning-free, every new
headline is registered in `AxiomAudit.lean` inside the closed cone, the spec's contracts table has
each row either canonical (unchanged) or landed (target proved), and no performance claim is made.

### Non-goals

- No TorchLean dependency, import, or vendoring; the pin stays a read-only reference.
- No change to the canonical corpus at `880aa57e…`, no change to `proof/ApeironWord.lean`, no Rust edits.
- No extraction/refinement of the new surface to Rust in this epic; no measured training or
  throughput claim; no softmax-equivalence or attention claim.
- No change to the frozen 48-module `SubstrateVendored` tree; its disposition is a separate task.
- No AI-track task outside this plan; no task without a gate and a control.

### Wave map

`Nous` is one `lean_lib` in `proof/corpus/lakefile.lean` with `globs := #[.submodules `Nous]`
(one lead edit, W0). `Core/` is ring-generic; `Model/` is AI-specific; a Core module importing a Model
module is a defect. Shared cache `proof/corpus/.lake` is one lane.

Wave 1 first ran under the working name `ApeironAI`; the lead chose `Nous` (spec decisions). The
rename pass moves `proof/corpus/ApeironAI/` to `proof/corpus/Nous/`, renames the `lean_lib`,
namespaces and paths, and re-runs the wave-1 gates and controls so evidence binds to the renamed
digests.

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W0 | lean_lib root (lead) | `proof/corpus/lakefile.lean` | lead | `lake build Nous` (empty lib green) | not applicable — root only |
| W1 | shapes | `Nous/Core/Shape.lean` | deepseek-flash | `lake build Nous.Core.Shape` | `size` drops a factor -> its `#guard`/lemma red |
| W1 | channel ratio | `Nous/Model/Weighting.lean` | gemini-flash | `lake build Nous.Model.Weighting` | denominator sign flip -> `channelWeight_eq` red |
| W1 | name pass `ApeironAI` -> `Nous` (lead) | `proof/corpus/lakefile.lean`, `Nous/Core/Shape.lean`, `Nous/Model/Weighting.lean` | lead | re-run both W1 module gates | W1 controls re-run on the renamed digests |
| W2 | tensors | `Nous/Core/Tensor.lean` | deepseek-flash | `lake build Nous.Core.Tensor` | pointwise add sign flip -> `Tensor.add_comm` red |
| W2 | token state fold | `Nous/Model/State.lean` | gemini-flash | `lake build Nous.Model.State` | component swap in `tokenStep` -> `token_step_fold` red |
| W2 | table memory | `Nous/Model/Memory.lean` | gemini-flash | `lake build Nous.Model.Memory` | `σ` dropped in the update -> `memory_step_model` red |
| W3 | traversals | `Nous/Core/Fold.lean` | deepseek-flash | `lake build Nous.Core.Fold` | accumulator order swap -> fusion lemma red |
| W3 | VJP contracts | `Nous/Core/Autograd.lean` | gemini-flash | `lake build Nous.Core.Autograd` | reversed composition -> `opSpec_compose_vjp` red |
| W4 | model assembly | `Nous/Model/Model.lean` | gemini-flash | `lake build Nous.Model.Model` | state/memory order swap -> model denote lemma red |
| W4 | face instances | `Nous/Model/Instances.lean` | deepseek-flash | `lake build Nous.Model.Instances` | wrong `σ` witness -> instance `#guard` red |
| W4 | channel-form attention | `Nous/Model/Attention.lean` | gemini-flash | `lake build Nous.Model.Attention` | drop the normalizer -> `attention_channel_form` red |
| W4 | hyperbolic decomposition (the bet) | `Nous/Model/Decomposition.lean` | gemini-flash | `lake build Nous.Model.Decomposition` | wrong root witness -> `hyperbolic_decomposition` red |
| W5 | audit registration (lead) | `proof/corpus/AxiomAudit.lean` | lead | `lake build AxiomAudit` | seeded `sorry` in a copy -> cone check red |
| W5 | barrier | none | lead | commands above | W1-W4 controls re-run, all restored |

### Task blocks

#### APEIRON-AI-W0-T1 · dedicated `Nous` library root

- **Scope:** add `lean_lib Nous where globs := #[.submodules `Nous]` to the corpus lakefile
  and a placeholder module so the lib builds; no default-target change.
- **Files:** `proof/corpus/lakefile.lean`, `proof/corpus/Nous/Core/Shape.lean` (placeholder only)
- **Depends on:** none
- **Grade:** tested
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous'`
- **Negative control:** not applicable — root wiring, covered by W1's controls.
- **Evidence to record:** lakefile sha256, gate exit, `ApeironWord.lean` `8e76b917…` unchanged.

#### APEIRON-AI-W1-T1 · shapes

- **Scope:** `Shape := List Nat`, `rank`, `size`, `CanBroadcastTo`, `BroadcastTo` with decidable
  instances and the boundary facts (`size [] = 1`, `size (d :: s) = d * size s`).
- **Files:** `Nous/Core/Shape.lean`
- **Depends on:** W0
- **Grade:** proved
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous.Core.Shape'`
- **Negative control:** in a disposable copy, make `size` drop the head dimension; the boundary
  `#guard`/lemma goes red with an evaluation diagnostic; restore green.
- **Evidence to record:** sha256, gate exit, control baseline/red/restore logs, elapsed.

#### APEIRON-AI-W1-T2 · channel-ratio weighting

- **Scope:** over a field, define `channelWeight r z := ringUpper r z / (ringUpper r z + ringLower r z)`,
  prove `channelWeight_eq` (`= ringUpper r z / (2 * z.1)` under `2 * z.1 ≠ 0`) and the channel
  multiplicativity corollaries from `ringUpper_channel`/`ringLower_channel`; state the zero-denominator
  refusal. No ratio-multiplicativity and no softmax-equivalence claim.
- **Files:** `Nous/Model/Weighting.lean`
- **Depends on:** W0
- **Grade:** proved
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous.Model.Weighting'`
- **Negative control:** flip the sign in the denominator copy -> `channelWeight_eq` red; restore green.
- **Evidence to record:** sha256, gate exit, control logs, canonical names used (`ringUpper`,
  `ringLower`, `RingChannel`, `ringUpper_channel`).

#### APEIRON-AI-W2-T1 · tensors

- **Scope:** `Tensor R s := Fin (size s) → R` over `[CommRing R]`, pointwise `add`/`sub`/`mul`, scalar
  `scale`, `add_comm`/`add_assoc`/`scale_add`; no broadcasting implementation beyond the Shape
  predicates.
- **Files:** `Nous/Core/Tensor.lean`
- **Depends on:** W1-T1
- **Grade:** proved
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous.Core.Tensor'`
- **Negative control:** flip a sign in pointwise add -> `Tensor.add_comm` red; restore green.
- **Evidence to record:** sha256, gate exit, control logs.

#### APEIRON-AI-W2-T2 · token state fold

- **Scope:** `tokenStep σ z w := ringMul σ z w` on canonical `RingCell R`; `token_step_fold` (the fold
  over `xs ++ ys` splits through `ringMul`, the empty fold is `ringOne`) from `ringMul_assoc`; no
  `mapAccum` yet (W3 owns the traversals).
- **Files:** `Nous/Model/State.lean`
- **Depends on:** W0
- **Grade:** proved
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous.Model.State'`
- **Negative control:** swap components in `tokenStep` -> `token_step_fold` red; restore green.
- **Evidence to record:** sha256, gate exit, control logs, canonical `ringMul_comm`/`ringMul_assoc` use.

#### APEIRON-AI-W2-T3 · table memory step

- **Scope:** `memoryStep σ k v M := insertWith (ringMul σ) k v M` over `Apeiron.Table.HMap`;
  `memory_step_model` characterising the map update; explicitly no probe-width claim (that obligation
  is named `probe_width_bound` and belongs to the executing face).
- **Files:** `Nous/Model/Memory.lean`
- **Depends on:** W0
- **Grade:** proved
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous.Model.Memory'`
- **Negative control:** drop the `σ` argument in a copy -> `memory_step_model` red; restore green.
- **Evidence to record:** sha256, gate exit, control logs, canonical `insertWith` usage.

#### APEIRON-AI-W3-T1 · traversals

- **Scope:** `mapAccum` (forward over `0 … n-1`) and `mapAccumRight` (reverse) with the exact
  specification from the spec; agreement with `foldl`/`foldr` for the token step; size-indexed result
  subtype.
- **Files:** `Nous/Core/Fold.lean`
- **Depends on:** W2-T1, W2-T2
- **Grade:** proved
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous.Core.Fold'`
- **Negative control:** swap the accumulator order in a copy -> the fusion lemma red; restore green.
- **Evidence to record:** sha256, gate exit, control logs.

#### APEIRON-AI-W3-T2 · VJP operator contracts

- **Scope:** `OpSpec R σ τ` (`forward`, `backward`), composition `>>>`, `opSpec_compose_vjp` (reverse
  chain rule), identity operator; ring-generic, no model names.
- **Files:** `Nous/Core/Autograd.lean`
- **Depends on:** W2-T1
- **Grade:** proved
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous.Core.Autograd'`
- **Negative control:** reverse the composition order in a copy -> `opSpec_compose_vjp` red; restore green.
- **Evidence to record:** sha256, gate exit, control logs.

#### APEIRON-AI-W4-T1 · model assembly

- **Scope:** the model record combining the streaming state fold (W2-T2, W3-T1), the table memory
  (W2-T3) and the channel ratio (W1-T2); one denote lemma that each field is the named composition,
  no hidden operation.
- **Files:** `Nous/Model/Model.lean`
- **Depends on:** W3-T1, W3-T2, W2-T3, W1-T2
- **Grade:** proved
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous.Model.Model'`
- **Negative control:** swap the state/memory application order in a copy -> the denote lemma red;
  restore green.
- **Evidence to record:** sha256, gate exit, control logs.

#### APEIRON-AI-W4-T2 · face instances

- **Scope:** concrete instances and kernel-evaluated checks at small fixed `σ`: elliptic `σ = -1`
  over `ℤ`/`ℚ`, parabolic `σ = 0`, hyperbolic `σ = 1` with the root witness `r = 1`; kernel `#guard`s
  on the token step and memory update by `rfl`/`decide` where the instances allow.
- **Files:** `Nous/Model/Instances.lean`
- **Depends on:** W4-T1
- **Grade:** tested
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous.Model.Instances'`
- **Negative control:** wrong root witness (`r = 1` for `σ = -1`) -> the instance check red; restore green.
- **Evidence to record:** sha256, gate exit, `#guard` output, control logs.

#### APEIRON-AI-W4-T4 · hyperbolic channel decomposition (the bet)

- **Scope:** with `r ^ 2 = σ` and `2`, `r` units, prove `z ↦ (ringUpper r z, ringLower r z)` is a
  bijection `RingCell R → R × R` intertwining `ringMul σ` with componentwise multiplication
  (`hyperbolic_decomposition`), and that the token fold and `memoryStep` factor through it into
  independent scalar operations per channel (`channel_diagonal_fold`). State the exact invertibility
  hypotheses; make no claim about the elliptic or parabolic faces.
- **Files:** `Nous/Model/Decomposition.lean`
- **Depends on:** W1-T2, W2-T2, W2-T3
- **Grade:** proved
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous.Model.Decomposition'`
- **Negative control:** replace the root witness with a non-root in a disposable copy ->
  `hyperbolic_decomposition` red; restore green.
- **Evidence to record:** sha256, gate exit, control logs, the two unit hypotheses, and the diagonal
  fold/memory forms.

#### APEIRON-AI-W4-T3 · channel-form attention

- **Scope:** the attention module: score the query against keys, encode the scores in the algebra's
  multiplicative channels, accumulate the normalizer `Σ_i w_i` and the weighted value sum
  `Σ_i w_i v_i` with constant-width streaming state, and normalize once; prove
  `attention_channel_form` (the normal form equals `Σ_i (w_i / Σ_j w_j) v_i` on the stated
  factorizable domain) and state the per-item ring-operation count. No random-feature approximation;
  no measured claim.
- **Files:** `Nous/Model/Attention.lean`
- **Depends on:** W1-T2, W2-T2, W2-T3, W4-T4
- **Grade:** proved
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous.Model.Attention'`
- **Negative control:** drop the normalizer in a disposable copy -> `attention_channel_form` red;
  restore green.
- **Evidence to record:** sha256, gate exit, control logs, the exact domain of the equality, and the
  streaming-state width.

#### APEIRON-AI-W5-T1 · headline registration and barrier (lead)

- **Scope:** register every new headline in `AxiomAudit.lean`, run the barrier commands, record
  digests and exits, update this plan's ledger row.
- **Files:** `proof/corpus/AxiomAudit.lean`
- **Depends on:** W4
- **Grade:** proved (cone)
- **Gate:** `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build AxiomAudit'`
- **Negative control:** in a disposable copy, seed `sorry` in one registered declaration -> the cone
  check red; restore green. The W1-W4 controls are re-run at the barrier.
- **Evidence to record:** AxiomAudit sha256, job count, cone lines, all task digests.

### Barrier checklist

- [ ] Every task gate exit 0, with its negative control run and restored.
- [ ] No task touched a file owned by another task; Core does not import Model.
- [ ] `lake build Nous`, bare `lake build`, and `lake build AxiomAudit` pass at one identity.
- [ ] `cargo test`, `cargo clippy`, `cargo fmt` pass; `proof/regen.sh --check` clean (expect no drift).
- [ ] Every new headline registered; cone `{propext, Classical.choice, Quot.sound}` only.
- [ ] Source digests recorded; `ApeironWord.lean` still `8e76b917…`.
- [ ] No performance, softmax-equivalence or training claim made; trust boundary restated in the handoff.

### Falsification point

`channelWeight_binary_softmax` and `attention_channel_form` settle the reproduction side on their
stated domains. The improvement side is a measurement: assemble the model, run a named task on a
named machine with a softmax baseline, and record quality and streaming cost. If the result is not
competitive, the model hypothesis fails and the lead decides between revising the architecture and
falling back to a softmax readout computed in `R` with the algebra carrying state and memory. A
failed measurement is a result, not a reason to relax the claim.

### Stop and rollback conditions

- Any gate reaching 540 s or a task reaching 45 min: stop, record elapsed and the incomplete stage;
  a timeout is a blocked task, not closure.
- Three materially different failed approaches to one obligation: stop and report the smallest
  unresolved goal (statement, exact error, files).
- A contract needs a premise change, a canonical-corpus theorem change, or a claim-grade change:
  stop and escalate; the lead reviews before further edits.
- No commit, push, branch, or worktree without explicit user authorization; controls never edit the
  live tree.
