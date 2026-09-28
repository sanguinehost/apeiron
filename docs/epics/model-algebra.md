# Model algebra: the LLM as one fold over the canonical object

Merged epic record: `intent/model-algebra.md`, `spec/model-algebra.md`, `plan/model-algebra.md` (one file per epic; see `docs/README.md`).

## Intent — Model algebra: the LLM as one fold over the canonical object

- **Owner:** Apeiron lead
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty (tracked files modified; this epic adds two untracked docs)
- **Status:** draft
- **Spec:** [`docs/epics/model-algebra.md`](model-algebra.md)
- **Plan:** `docs/epics/model-algebra.md` (to be written)

```text
SCOPE:      docs/epics/model-algebra.md, docs/epics/model-algebra.md; the governing abstraction and its
            reflection contract; no source, proof, extraction or plan file
REGISTER:   planning / architecture and specification (algebra/refinement/measurement); exclusions: no
            new theorem, no gate run, no claim grade raised
CORPUS:     not applicable to this document; the abstraction names, but does not restate,
            Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm,ringOne,ringMul_comm,ringMul_assoc},
            Apeiron.Trichotomy.ChannelsAnyRing.{ringUpper,ringLower,RingChannel},
            Apeiron.Table.{HMap,get,insertWith,build}, Nous.Model.{tokenStep,tokenFold,memoryStep,step,run},
            Sanguine.AttentionSpinor.{attention_is_spinor_scan,softmax_attention_is_online_spinor}
GATE:       path resolution and whitespace only; command in §Completion condition and in the spec's
            Evidence plan; no compiler gate in this epic
GRADE:      anchor for the abstraction and the reflection contract; the named theorems carry the grade
            their existing files carry (proved or target), restated per row below
```

### Problem

Apeiron verifies kernels, not a model. The executed artifacts are per-operation: the `Cell<i64>` word
kernels (`word_mul_denote`, `word_norm_denote`), the fixed-point numeric refinements
(`fixed_exp_refines_proved`, `fixed_rsqrt_refines_proved`, `fixed_sin_cos_refines_proved`), the landed
ring-generic surface (`Nous.Model.*`), and the ℝ-valued attention theorems in
`proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean`. No single object names an LLM as one fold
of parameterized layers over `RingCell`/`ringMul`, and no contract says which sources are pinned to it
and which way each gate pins them. The gap is therefore structural: a new layer, a new quantization
width, or a new file format re-states the canonical object in its own vocabulary, and drift from the
object is not detected except by a human reading the file. This epic writes the abstraction, the
reflection contract with its three source classes and gate directions, the composed-margin target, the
single canonical packing grammar, and the proof-cost budget that calibrates the plan.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Keep per-layer specs and link them by convention | No shared carrier or law; drift is unguarded, and the canonical `ringMul` link is asserted, not checked. §3 of `AGENTS.md` forbids two production paths; a per-layer restatement is the same defect at document scale. |
| Make the float reference the abstraction | The executing path is integer/fixed-point by owner decision ([`docs/epics/inference-parity.md`](inference-parity.md)); the `f64`/ℝ reading is a test oracle, not a verified artifact. `rustc`/LLVM codegen and IEEE-754 are the named platform boundary. |
| Treat attention as a dense `O(T²)` softmax table | The landed recurrence theorems (`attention_is_spinor_scan`, `softmax_attention_is_online_spinor`) are exact and `O(1)`-state; the dense table adds work with no theorem. The channel-form target (`attention_channel_form`) is the ring-generic reading of the same scan. |
| Import a tensor/autograd library as the abstraction | TorchLean (`de3192df4779877ceb65cfe470a4d4ce9d480a5b`) is a read-only scope pointer. The canonical object is `RingCell`/`ringMul`; a tensor library would be a second foundation and a runtime dependency. |
| Claim the abstraction complete | Only the named operations are covered. An unlisted construct is out of the domain, not silently included. This is stated as a non-goal, not discovered later. |

### Completion condition

The abstraction closes when the named headlines of [`docs/epics/model-algebra.md`](model-algebra.md)
build green at one source identity with the dropped-normalizer control red, and the two documents resolve
as written. The first wave's barrier command:

```sh
cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c \
  'LEAN_NUM_THREADS=1 lake build Nous.Model.ErrorComposition && LEAN_NUM_THREADS=1 lake build Nous.Model.Attention AxiomAudit'
```

`Nous.Model.ErrorComposition` (the parallel `APEIRON-ALGEBRA-D2` module) and `Nous.Model.Attention`
(`APEIRON-ALGEBRA-D3` sibling work) are creation targets; a missing gate is named, not replaced. This
document epic itself closes on the path and whitespace gate in the spec's Evidence plan.

### Invariants

1. The abstraction is stated over `[CommRing R]` with `σ : R` and refines the canonical
   `Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` and
   `Apeiron.Table.{HMap,get,insertWith,build}`; a face that names them by path, namespace or import is
   not rooted.
2. No float runtime theorem is claimed. ℝ is the reference domain; Q32.32 (`Numeric/Scale.lean` `q32`,
   `toReal`) is the executing domain; `ZMod (2^64)` is the word domain.
3. The channel decomposition (`hyperbolic_decomposition`) is used only where `2` and the root witness
   `r` are units. At `Z/2^w` only the ring identity transports; `2 * h = 1` does not (`Spirv/Semantics.lean`
   `two_not_unit`; `ringTrace σ u v = 2 * (ringMul σ u v).1`). This is the 2-adic caveat.
4. Every headline registers in `proof/corpus/AxiomAudit.lean` inside `{propext, Classical.choice,
   Quot.sound}`; the Slang source-refinement chain keeps its separate empty cone and is not imported here.
5. No operation count is a measurement; a cost claim names command, machine, metric and interval
   (`AGENTS.md` §10). Sampling decisions refuse below the composed margin (§Spec, Composed margin).
6. One source class maps to one gate and one direction; no gate enforces two directions.

### Verification hypotheses

| Claim | Grade | Evidence (command or theorem) |
|---|---|---|
| The token step and its fold are the canonical `ringMul` fold | proved | `Nous.Model.token_step_fold`, `Nous.Model.tokenFold_append`, `token_step_fold_contract` |
| The table memory step is the canonical `insertWith` update | proved | `Nous.Model.memory_step_model`, `memory_step_get_self`, `memory_step_get_ne` |
| The assembled run is exactly `(tokenFold, memoryFold)` | proved | `Nous.Model.run_eq`, `run_state_eq_tokenFold`, `run_memory_eq_memoryFold` |
| The hyperbolic face splits into independent channels | proved | `Nous.Model.hyperbolic_decomposition`, `hyperbolic_decomposition_field`, `channel_diagonal_fold` |
| The channel ratio is the binary softmax of the log-channel pair | target | restated in [`docs/epics/model-algebra.md`](model-algebra.md) W3-T2; `Weighting.lean` declines the equivalence as stated (`Weighting.lean:29-30`); domain `0 < u`, `0 < l` |
| Linear attention is the spinor scan; softmax attention is the exact online recurrence | proved | `Sanguine.AttentionSpinor.attention_is_spinor_scan`, `softmax_attention_is_online_spinor` (ℝ; the latter is landed but not yet in `AxiomAudit.lean`) |
| The normalized channel-form attention equals `Σ_i (w_i / Σ_j w_j) v_i` on the factorizable domain | target (open bet) | planned `Nous.Model.attention_channel_form`, `Nous/Model/Attention.lean` (this plan W3-T1) |
| The fixed-point transcendental kernels refine their real specifications | proved | `apeiron_word.fixed_exp_refines_proved`, `fixed_rsqrt_refines_proved`, `fixed_sin_cos_refines_proved` |
| The integer Q4_K/Q6_K unpackers refine their byte-exact references | proved | `apeiron_word.word_q4_k_codes_ok`, `word_q6_k_codes_ok` (`Numeric/FormatRefinement.lean`) |
| The extracted word kernels refine canonical `ringMul`/`ringNormForm` at `ZMod (2^64)` | proved | `apeiron_word.word_mul_denote`, `word_norm_denote` (`proof/corpus/WordRefinement.lean`) |
| GGUF header parse invariants (`HeaderWF`) refine the extracted parser | target | predicates stated in `Numeric/FormatRefinement.lean`; `word_gguf_parse_header` extracted, `proof/regen.sh` root |
| The composed margin refuses the decision below the summed `ε` | proved | `Nous.Model.decision_margin_preserved`, `strictWinner_perturb`, `numeric_fold_margin_usable`, `numeric_decision_unchanged` (`Nous/Model/ErrorComposition.lean`) |
| The packing grammar admits widths 2/3/4/6/8, the native F16/F32 identity face, and the K-quant blocks | target | `APEIRON-ALGEBRA-D3`/`A5`; current tested instance `src/format/quant.rs` (Q4_K 144 B, Q6_K 210 B) |

### Proof-cost budget

Calibrated on the receipts in [`docs/epics/inference-parity.md`](inference-parity.md). Costs are
per slice at the declared 20/30-min soft box; the barrier runs the lane-locked `timeout 540` Lean gate.
The numeric receipts are the calibration because they are the only completed, independently verified
programme slices of comparable shape.

| Wave | Definitions | Lemmas / headlines | Slices | Calibration |
|---|---|---|---|---|
| 1 · reflection | one extraction root and one wrapper per kernel | refinement theorem + refusal theorem + cone registration | 1–2 per kernel | `WordRefinement.lean` (1304 lines) is the canonical face; a fresh word kernel is 1 slice |
| 2 · composed margin (D2) | `Nous.Model.ErrorComposition` error ledger | per-op `ε`, per-layer aggregation, fold composition, margin predicate | 6–10 | rsqrt closure took ~8 slices to 1074 lines; a bound-only composition is cheaper per slice |
| 3 · channel attention | `Nous/Model/Attention.lean` | `attention_channel_form` on the factorizable domain | 4–6 | `RopeRefinement` took ~20 slices to 3524 lines; the attention module carries no transcendental remainder, so fewer |
| 4 · packing (D3) | parameterized grammar + GGUF converter | width/block/scale/min/bit-plane predicates; K-quant instances | 6–10 | `Numeric/FormatRefinement.lean` (1117 lines) lands Q4_K and Q6_K in one wave of ~8 slices |
| 5 · word/ternary faces | extraction roots per face | face instance `#guard`s and refinements | 2–4 per face | `Nous/Model/Instances.lean` is a small fixed-`σ` instance module (W4-T2, not yet present) |

One-person maintenance claim. A single maintainer can hold the *interfaces*: the canonical names, the
three source classes, the packing grammar and the reflection directions. The claim carries these
recorded caveats, each an existing receipt, and is not a throughput claim:

- The two remaining `AGENTS.md` §4 violations are `src/engine/mod.rs` (3477) and
  `src/format/packing.rs` (1711). After the E2 splits, `Numeric/ExpRefinement.lean` (16) and
  `Numeric/RopeRefinement.lean` (15) are aggregators over parts ≤1108; `Numeric/FormatRefinement.lean`
  (1117) and `Numeric/RsqrtRefinement.lean` (1072) are within. Splitting the two remaining files is an
  open obligation, not a settled property.
- Six bounded `set_option` raises existed; five were discharged and one
  (`maxHeartbeats 8000000` on `word_fixed_exp_horner_acc`) is a named accepted opening with owner
  acceptance pending. A maintainer owns that removal.
- `AxiomAudit.lean` imports every lane, so one lane's mid-edit file breaks the shared gate; the
  receipts record this repeatedly. Serialization on `flock /tmp/opencode/apeiron-lane.lock` is the
  current answer, not a solution.
- One slice was escalated to a different worker model (`E2C14`, `xiaomi-pro`); the abstraction's cost
  budget must assume occasional escalation, not forbid it.

### Non-goals

- No float runtime theorem; ℝ is a reference domain, not an executing path.
- No GPU execution semantics. SPIR-V subset verification is the separate
  [`docs/epics/gpu-spv.md`](gpu-spv.md) epic; this abstraction names the seam only.
- No completeness claim for the abstraction. Constructs not listed in the spec's operation table are
  outside the domain.
- No claim that capability, quality or watt goals follow from the algebra. Improvement is a measurement
  against a named baseline, and a failed measurement is a result.
- No token-level correctness theorem. The composed bound plus the runtime margin is the artifact; no
  unconditional per-token guarantee is claimed.
- No new canonical-corpus theorem, no corpus pin change, no `proof/ApeironWord.lean` regeneration in
  this documentation epic.

### Trust boundary

The abstraction does not reach: `rustc`/LLVM codegen, the kernel ABI, the CPU ISA, the GPU/SPIR-V
toolchain, driver and hardware, and the Rust `unsafe` container. It does not prove the allocator or
executor realization, `std::thread::scope`, spawn/join/unwind, or arbitrary-callback panic freedom.
It does not prove the GGUF converter's data values, only the extracted parser's stated invariants; the
`HeaderWF` refinement is a target. It does not cover training, convergence, tokenizers or datasets.
A citation to a theorem or a sibling repository is a pointer until the named declaration is read and the
cone registered; `softmax_attention_is_online_spinor` is registered in `AxiomAudit.lean:1372` (G6 closed
in [`docs/evidence/rigor-baseline.md`](../evidence/rigor-baseline.md)).

### Open questions

- Composed-bound shape: sum of relative `ε` per op, or a Lipschitz-constant product per layer? The
  inference-parity spec states bounds do not compose unconditionally; D2 owns the decision.
- `attention_channel_form` domain: the factorizable-kernel domain is exact only where the key/value
  scores factor; the elliptic and parabolic faces do not split. Does the module state the hyperbolic
  domain only, or a channel-form fold that degenerates correctly? Owner: lead, with this plan W3-T1.
- Packing grammar home: does the grammar live in `Nous.Core` (ring-generic) or `Nous.Model` (AI-specific)?
  A Core module importing a Model module is a defect; a format grammar is AI-specific data.
- Ternary face: `Z/3` arithmetic has no `2`-unit in the same sense and no Aeneas-derived word kernel yet;
  is the balanced-trit plane a packing face only, or a carrier face with its own refinement?

## Specification — Model algebra: the LLM as one fold over the canonical object

- **Owner:** Apeiron lead
- **Intent:** [`docs/epics/model-algebra.md`](model-algebra.md)
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty (tracked files modified; this epic adds two untracked docs)
- **Status:** draft

```text
SCOPE:      docs/epics/model-algebra.md; the abstraction, the reflection contract, the composed-margin
            target, the packing grammar and their evidence plan; no source, proof or extraction file
REGISTER:   architecture and specification (algebra/refinement/measurement); exclusions: this document
            claims no new theorem, runs no Lean or cargo gate, and raises no claim grade
CORPUS:     Apeiron.Trichotomy.AnyRing, Apeiron.Trichotomy.ChannelsAnyRing, Apeiron.Table.Basic,
            Nous.Model.{State,Memory,Model,Weighting,Decomposition},
            Sanguine.AttentionSpinor, apeiron_word.{word_mul_denote,word_norm_denote},
            apeiron_word.{fixed_exp_refines_proved,fixed_rsqrt_refines_proved,fixed_sin_cos_refines_proved},
            apeiron_word.{word_q4_k_codes_ok,word_q6_k_codes_ok}
REFERENCE:  docs/direction.md, docs/kernel-boundary.md, docs/epics/inference-parity.md,
            docs/epics/ai-track-reanchor.md, docs/epics/gpu-spv.md, docs/epics/inference-parity.md
GATE:       path resolution and whitespace only; command in Evidence plan step 0; no compiler gate in
            this document epic
GRADE:      anchor for the abstraction and reflection contract; per-row grades below are the grades the
            named files carry today or the target grade of a named unimplemented declaration
```

### Scope and admissible domain

The abstraction covers the named operations of one decoder-only transformer executed on the integer
path, and nothing else. Each operation is a composition of canonical `ringMul`, `ringNormForm`, channel
projections and `Apeiron.Table` operations, or it is out of domain. Partiality is explicit: the channel
ratio, channel inversion and the sampling decision are partial and carry named preconditions or a
refusal predicate.

| Item | Value |
|---|---|
| Operation | One model forward fold: embed token → per-layer (attention, FFN, norms, RoPE) → logits → sample |
| Domain | `[CommRing R]` with `σ : R` for the algebraic layer; `R = ℝ` reference, Q32.32 executing, `ZMod (2^64)` word, ternary face; keys `K` with `DecidableEq K`; positions `Fin n` |
| Totality | `tokenStep`, `memoryStep`, `step`, `run`, `Tensor` pointwise ops, RoPE rotor are total; `channelWeight` needs `IsUnit (2 * z.1)`; `hyperbolic_decomposition` needs `2`, `r` units; sampling is total with a refusal predicate |
| Preconditions | named root witness `r ^ 2 = σ`; `IsUnit (2 : R)` and `IsUnit r` for the channel bijection; `r ≠ 0` and `(2 : K) ≠ 0` over a field; `fixed_*` input windows |
| Postcondition | every layer denotes a composition of `ringMul`/`ringNormForm`/channel projections/`Apeiron.Table` operations; each new headline audits in `{propext, Classical.choice, Quot.sound}` |

No silent domain reduction: a result that needs an ordered or invertible setting names it (`ℝ` for the
sign lemmas, `IsUnit` for the ratio, `ZMod (2^w)` for the word face). The 2-adic caveat is part of the
domain: the channel decomposition's `2 * h = 1` does not hold at `Z/2^w`.

### The abstraction, operation by operation

Layers are functions `L_i : S_i → S_{i+1}`; the model is `run = L_n ∘ … ∘ L_1` realized as
`List.foldl step` (`Nous.Model.Model.run`, `run_eq`). The table entry names the declaration each layer
denotes and the grade its file carries.

| Layer | Declaration (abstraction) | Canonical denotation | Grade |
|---|---|---|---|
| Token state / update | `tokenStep σ z w := ringMul σ z w`; fold `tokenFold σ xs := xs.foldl (ringMul σ) ringOne` | `Apeiron.Trichotomy.AnyRing.ringMul`, `ringOne` | proved (`Nous/Model/State.lean`: `tokenStep`, `tokenFold`, `token_step_fold`) |
| Token step law | `token_step_fold` (append splits through `ringMul`), base `token_step_fold_nil` | `ringMul_assoc`, `ringMul_one`, `ringOne_mul` | proved |
| Table memory | `memoryStep σ k v M := insertWith (ringMul σ) k v M` | `Apeiron.Table.insertWith` on `HMap K (RingCell R)` | proved (`Nous/Model/Memory.lean`: `memory_step_model`, `memory_step_get_self`, `memory_step_get_ne`) |
| Model assembly | `step`, `memoryFold`, `run`; `run_eq` | pair of the token fold and the memory fold | proved (`Nous/Model/Model.lean`) |
| Attention (landed) | linear-attention step is `AffineStepN` with `A = I`; the run is the fold of injections; softmax attention is the exact online `(m, den, num)` recurrence | ℝ vector space (not yet the canonical `RingCell`) | proved (`AttentionSpinor.attention_is_spinor_scan`, `softmax_attention_is_online_spinor`); the second is landed but not yet in `AxiomAudit.lean` |
| Attention (channel form) | streaming normalizer `Σ_i w_i` and weighted sum `Σ_i w_i v_i`, normalize once; normal form `Σ_i (w_i / Σ_j w_j) v_i` on the factorizable domain | channel projections `ringUpper`/`ringLower` and `ringMul` | target, open bet (`Nous.Model.attention_channel_form`, `Nous/Model/Attention.lean`, this plan W3-T1) |
| FFN pointwise | `Tensor R s → Tensor R s` coordinatewise, composed through `OpSpec` VJP | `Nous.Core.Tensor` pointwise ops; `Nous.Core.Autograd.opSpec_compose_vjp` | proved for the VJP law (`Nous/Core/Autograd.lean`); the activation transcendental is the numeric layer |
| Norms | scaling by a function of the norm form; `channelPair_normForm` factors it | `Apeiron.Trichotomy.AnyRing.ringNormForm σ z = z.1^2 − σ·z.2^2`; `channelPair r z .1 * channelPair r z .2` | anchor: `ringNormForm` is not nonnegative in general, so the scaling needs a positivity hypothesis; the word ring carries no energy claim (`docs/kernel-boundary.md`) |
| RoPE | position-indexed `SO(2)` rotor action on the `(q,k)` pair; group law `R(θ) ∘ R(φ) = R(θ+φ)` | `Sanguine.AttentionSpinor.rotor`, `rotor_compose`, `rotor_isometry`, `so2_spinor_recurrence` | proved over ℝ (`AttentionSpinor`); executing integer path proved (`fixed_sin_cos_refines_proved`, `Numeric/RopeRefinement.lean`) |
| Sampling | total `sample : logits → token`, with a margin predicate refusing when the top-two gap is below the composed `ε` | `StrictWinner`-shaped decision; `strictWinner_perturb`, `decision_margin_preserved` | proved (`Nous/Model/ErrorComposition.lean`); the decision shape is named in [`docs/epics/inference-parity.md`](inference-parity.md) |

Parameterization over the face. The algebraic layers are stated over `[CommRing R]` and `σ : R`, so one
theorem instantiates per face:

| Face | `R` | `σ` | What instantiates | Caveat |
|---|---|---|---|---|
| ℝ reference | `ℝ` | any | `token_step_fold`, `channel_diagonal_fold`, `hyperbolic_decomposition_field`, all `AttentionSpinor` headlines | analytic bounds need `Real.*` remainder lemmas |
| Q32.32 executing | `ℤ`-scaled fixed point | `−1`/`0`/`+1` | `fixed_exp_refines_proved`, `fixed_rsqrt_refines_proved`, `fixed_sin_cos_refines_proved`; the algebra via the word ring | error is the combined `ε·|F| + floor`, not pure relative |
| `Z/2^w` word | `ZMod (2^64)` | `denoteSigma s` | `word_mul_denote`, `word_norm_denote` | 2-adic caveat: `ringTrace = 2 * channel0` and `two_not_unit`; `2 * h = 1` does not transport, so the channel bijection does not instantiate here |
| ternary | `Z/3` / balanced trits | planned | — | anchor; no extracted kernel exists in this tree |

### Reflection contract: three source classes, one direction each

Every executing source belongs to exactly one class. The gate and the direction differ per class; a
class that enforced two directions would prove and consume the same object, which `AGENTS.md` §4 forbids.

| Class | Sources | Gate | Direction | Outside the theorem |
|---|---|---|---|---|
| (a) Extracted-and-refined kernels | `src/`, extracted via pinned Charon/Aeneas into `proof/ApeironWord.lean`; extracted bodies invoked by `proof/extract/lib.rs` wrappers | `proof/regen.sh --check` (drift); `lake build` + `lake build AxiomAudit` (closed cone); mutation control on the real source | **source → abstraction**: the refinement theorem states the extracted body denotes the canonical definition (`word_mul_denote`; `fixed_exp_refines_proved`) | allocator realization, executor/`std::thread::scope`, `rustc`/LLVM, ABI, GPU/SPIR-V, `unsafe` container; wrapper fixes types only |
| (b) Generated / data parameters | tensor shapes, layer schedules, packing parameters (`src/format/quant.rs` block geometry, scale/min layout, bit-plane order), GGUF tensor directory | `cargo test` on the converter (`src/format/gguf.rs`); extracted `word_gguf_parse_header` under `regen.sh`; `#guard`/schema checks; target `HeaderWF` refinement | **abstraction → source**: the grammar fixes the admitted layout; the converter must produce data inside the abstraction's declared domain and refuses malformed input (`Truncated`, `OutOfBounds`, `UnknownType`, `LengthOverflow`) | the parameter values are data, not code; a converter test proves refusal and shape containment, not that the values are correct for a task |
| (c) Tested driver | IO, `HashMap`/`Table` drivers, tokenizer (`src/format/bpe.rs`), harnesses | `cargo test --workspace --all-features --locked`; clippy; fmt | **neither direction as proof**: the driver is the execution substrate; only its kernel fragments cross back through class (a) | everything the driver does outside an extracted kernel: scheduling, allocation, panic, IO ordering |

Rule: an operation crosses into the abstraction only through class (a) or class (b). A driver path that
recomputes a kernel inline is a defect, not a third class.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| `token_step_fold` | `foldl (ringMul σ) ringOne (xs ++ ys) = ringMul σ (foldl … xs) (foldl … ys)` and the empty base | proved | `Nous.Model.token_step_fold`, `token_step_fold_contract` |
| `run_eq` | `run σ M₀ tokens = (tokenFold σ (tokens.map Prod.snd), memoryFold σ M₀ tokens)` | proved | `Nous.Model.run_eq` |
| `hyperbolic_decomposition` | with `r² = σ`, `r·s = 1`, `2·h = 1`: `channelPair r` is a bijection intertwining `ringMul` and addition, preserving `ringOne`, factoring `ringNormForm` | proved | `Nous.Model.hyperbolic_decomposition`, `…_field` |
| `channel_diagonal_fold` | token fold and `memoryStep` factor into independent scalar channel operations | proved | `Nous.Model.channel_diagonal_fold` |
| `attention_is_spinor_scan` | linear attention is an `AffineStepN` with `A = I`; the run equals the fold; composition is associative | proved (ℝ) | `Sanguine.AttentionSpinor.attention_is_spinor_scan` |
| `softmax_attention_is_online_spinor` | the `(m, den, num)` scan computes `(Σ e^{sc}·v)/(Σ e^{sc})` exactly, `O(1)` state | proved (ℝ) | `Sanguine.AttentionSpinor.softmax_attention_is_online_spinor` |
| channel ratio as binary softmax | where `u, l > 0`: `u/(u+l) = Real.exp (Real.log u) / (Real.exp (Real.log u) + Real.exp (Real.log l))`, the binary softmax of the log-channel pair | target | restated in [`docs/epics/model-algebra.md`](model-algebra.md) W3-T2; `Weighting.lean:29-30` declines the equivalence as originally phrased; domain `0 < u`, `0 < l` |
| `attention_channel_form` | normalized output equals `Σ_i (w_i / Σ_j w_j) v_i`, `w_i` the channel weights, on the factorizable domain | target (open bet) | planned `Nous/Model/Attention.lean`; control: drop the normalizer → red |
| `fixed_*_refines` | per-op combined bound `ε·|F (toReal x)| + floor` against the real specification on the named window | proved | `fixed_exp_refines_proved`, `fixed_rsqrt_refines_proved`, `fixed_sin_cos_refines_proved` |
| `word_q4_k_codes_ok`, `word_q6_k_codes_ok` | success yields the byte-exact reference code array on `qk ≤ len` | proved | `Numeric/FormatRefinement.lean`; registered in `AxiomAudit.lean` |
| composed margin | the fold's composed `ε` and the runtime refusal predicate | proved | `Nous.Model.ErrorComposition`: `Layer.budget`, `norm_execComp_sub_refComp_le`, `strictWinner_perturb`, `decision_margin_preserved`, `numeric_fold_margin_usable` |
| packing grammar | widths 2/3/4/6/8, **F16/F32 (the identity face: no quantisation)**, block, scale kind, optional min, bit-plane order, balanced-trit planes | target | `APEIRON-ALGEBRA-D3`/`A5`; current tested instances Q4_K/Q6_K and the F16/F32 element paths |

### Composed margin

Per-op `ε` is the proved artifact of the numeric layer. The inference-parity spec states the discipline
([`docs/epics/inference-parity.md`](inference-parity.md), Contracts): "Bounds do not compose into an
unconditional token theorem — the decision carries a runtime margin." The composed bound is built in
three stages, each a separate declaration in the target module:

| Stage | Object | Form | Grade |
|---|---|---|---|
| per-op | `fixed_exp_refines` | `|toReal v − exp (toReal x)| ≤ 2^-20·exp (toReal x) + 4/q32` on `window` | proved |
| per-op | `fixed_rsqrt_refines` | `|toReal v − 1/√(toReal x)| ≤ 2^-20·(1/√(toReal x)) + 4/q32` on `rsqrt_window` | proved |
| per-op | `fixed_sin_cos_refines` | `|toReal sc.f − f (toReal x)| ≤ 2^-20·|f (toReal x)| + 2^-20` on `trig_window`, both outputs | proved |
| per-layer | `LayerEps` | layer error is the sum over the layer's ops plus the layer's Lipschitz factor on its input error | target (D2) |
| fold | `FoldEps` | the fold's error is the layer sum, or the Lipschitz product, on the stated domain | target (D2) |
| decision | margin predicate | the argmax/sample refuses when the observed top-two score gap is below `FoldEps` | proved (`decision_margin_preserved`), `StrictWinner` shape |

What a composed bound would claim: a finite error budget `ε_fold` for the named operation set on the
named windows, and a total refusal predicate that withholds the decision when the margin is below
`ε_fold`. What it would not claim: no token-level correctness theorem (the decision is refused, not
guaranteed correct, below the margin); no float runtime theorem (the reference is ℝ, the execution is
integer/fixed-point); no tightness claim; no training, capability or quality claim. This restates the
inference-parity trust boundary: "No token or capability theorem; the artifact is the proved `ε` plus
the runtime margin."

### Canonical packing

One parameterized grammar covers every admitted weight layout; a layout not expressible in the grammar
is out of domain. The grammar is a class-(b) object: it constrains the converter, it does not prove the
values. Current tested instances are `src/format/quant.rs` (`q4_k_codes`, `q6_k_codes`;
`Q4_K_BLOCK_BYTES = 144`, `Q6_K_BLOCK_BYTES = 210`), refined in
`Numeric/FormatRefinement.lean` (`q4CodesRef`, `q6CodesRef`, `word_q4_k_codes_ok`, `word_q6_k_codes_ok`).

| Grammar axis | Admitted | Current instance | Grade |
|---|---|---|---|
| code width | 2, 3, 4, 6, 8 bits; **F16/F32 identity (exact f16; round-to-nearest Q32.32 for f32, with refusals)** | 4 (Q4_K), 6 (Q6_K) | tested; general width + native face target (D3/A5) |
| block | weights-per-block `qk`, bytes-per-block; stored as `(qk, bb)` | `block_geom` in `Numeric/FormatRefinement.lean` (144, 210) | proved instance |
| scale kind | per-block f16 scale; K-quant sub-block scales | `f16 d; f16 dmin; u8 scales[12]` | tested |
| min | optional per-block minimum offset | Q4_K has `dmin`; Q6_K none | tested |
| bit-plane order | low/high plane split and offset order (`ql`, `qh`) | Q6_K `ql[128]; qh[64]` | proved (`word_q6_k_codes_ok`) |
| balanced-trit planes | optional signed ternary plane, values `{-1,0,1}` | none in this tree | anchor (planned face) |

GGUF is the tested import converter for class (b): `src/format/gguf.rs` parses the header and tensor
directory, extracted as `word_gguf_parse_header` (`proof/regen.sh` root; `proof/extract/lib.rs:608`).
The header/data invariants `HeaderWF`, `TensorRangeWF`, `magic_ok`, `version_ok` and the
`LengthOverflow`/`UnknownType`/`OutOfBounds` gates are stated in `Numeric/FormatRefinement.lean`; the
refinement theorem for `word_gguf_parse_header` is a residual, not a landed proof. The parallel
`APEIRON-ALGEBRA-D3` task implements the general grammar and the converter; this specification fixes the
axes and the domain, not the code.

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| denominator `2 * z.1` not a unit | channel ratio unavailable | `¬ IsUnit (2 * z.1)` |
| no root witness for `σ` | channel decomposition unstatable | `¬ ∃ r : R, r ^ 2 = σ` |
| `2` or `r` not a unit | channel bijection not available | `IsUnit (2 : R)`, `IsUnit r` |
| argument outside a numeric window | `Result` refusal; no wrap | per-function `hbound` (`fixed_exp_refuses`, `fixed_rsqrt_refuses`, `fixed_sin_cos_refuses`) |
| `i64`/`i128` overflow | refusal; never wraps | checked add/mul in the extracted body |
| margin below composed `ε` | decision refusal | margin predicate (`StrictWinner` shape) |
| short or malformed GGUF/K-quant block | refusal with a named error | `Truncated`, `OutOfBounds`, `UnknownType`, `LengthOverflow` |

An infallible signature over a lossy or partial operation is a defect. `RingCell`, `ringMul`,
`tokenStep`, `memoryStep` are total; the partialities above are named, not hidden behind total types.

### Evidence plan

0. **Document gate.** Every cited path resolves (`rg`/`test -f`), and the two documents carry no
   whitespace diagnostics. Command and result in the task receipt; no compiler gate.
1. **Positive tests.** `cargo test --workspace --all-features --locked` covers the class-(b) converter
   and the class-(c) driver; the Q4_K/Q6_K unpackers are covered by `Numeric/FormatRefinement.lean`.
2. **Proof.** The proof barrier is `cd proof/corpus && flock -w 600
   /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous'` plus
   `lake build AxiomAudit`, on the module set the plan names (`ErrorComposition`, `Attention`). The
   existing surfaces build in `lake build Nous`.
3. **Axiom cone.** Every headline registers in `proof/corpus/AxiomAudit.lean`; anything outside
   `{propext, Classical.choice, Quot.sound}` fails. `softmax_attention_is_online_spinor` is registered at
   `AxiomAudit.lean:1372` (G6 closed); the Slang chain keeps the empty cone and is not imported here.
4. **Negative control.** Per headline: mutate the surface's own definition in a disposable copy — drop
   the normalizer for `attention_channel_form`, wrong root witness for `hyperbolic_decomposition`,
   perturbed coefficient for a numeric bound — and show the named declaration red with an attributable
   diagnostic; restore green. Controls never edit the live tree.
5. **Measurement.** None in this document. A cost claim names command, machine, metric and interval
   (`AGENTS.md` §10); the operation counts are combinatorics, not measured.

### Trust boundary

This specification and its evidence do not cover: `rustc`/LLVM codegen, the kernel ABI, the CPU ISA, the
GPU/SPIR-V toolchain, driver and hardware, and the Rust `unsafe` container; allocator and executor
realization, `std::thread::scope`, spawn/join/unwind, arbitrary-callback panic freedom; the GGUF
converter's data values (only the stated invariants, and `HeaderWF` is a target); the truth of any
citation until the declaration is read and its cone registered; training, convergence, tokenizers and
datasets; and any capability, quality or watt claim. The abstraction is not complete: constructs not in
the operation table are outside the domain. The word ring carries no energy or nonnegativity claim, and
the 2-adic caveat bounds the channel decomposition at `Z/2^w`.

### Open questions

- Composed-bound form: additive sum of per-op relative `ε`, or a Lipschitz-constant product per layer?
  D2 owns the decision and must state the domain where the bound holds.
- `attention_channel_form` domain: hyperbolic-only, or a channel-form fold that degenerates on the
  elliptic and parabolic faces? Owner: lead, with this plan W3-T1.
- Packing grammar home: `Nous.Core` (ring-generic) or `Nous.Model` (AI-specific data)? A Core module
  importing a Model module is a defect; a format grammar is data and belongs with the model.
- Ternary face: `Z/3` packing only, or a carrier face with its own word kernel and refinement? No
  extracted ternary kernel exists in this tree.
- Channel ratio as binary softmax: resolved as a restatement under the log substitution
  (`docs/epics/model-algebra.md` W3-T2); the original phrasing is declined by `Weighting.lean:29-30`.

## Plan — Model algebra: the LLM as one fold over the canonical object

- **Owner:** Apeiron lead
- **Intent:** [`docs/epics/model-algebra.md`](model-algebra.md)
- **Spec:** [`docs/epics/model-algebra.md`](model-algebra.md)
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty (tracked files modified and the two epic docs untracked; the per-file list is `docs/archive/handoff-openai.md`)
- **Status:** draft

```text
SCOPE:      docs/epics/model-algebra.md; write the wave map, task blocks, gates and controls of the
            model-algebra epic. Completion = every intent/spec headline maps to one task with a named
            gate and, where soundness is at stake, a negative control
REGISTER:   planning (architecture/decomposition); exclusions: no source, proof, extraction, corpus or
            claim grade is changed; no gate is run beyond the document gate
CORPUS:     read and named, not restated: Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm,
            ringOne}, Apeiron.Trichotomy.ChannelsAnyRing.{ringUpper,ringLower,RingChannel},
            Apeiron.Table.{HMap,get,insertWith,build},
            Nous.Model.{State,Memory,Model,Weighting,Decomposition,ErrorComposition},
            Sanguine.AttentionSpinor.{attention_is_spinor_scan,softmax_attention_is_online_spinor},
            apeiron_word.{word_mul_denote,word_norm_denote},
            apeiron_word.{fixed_exp_refines_proved,fixed_rsqrt_refines_proved,fixed_sin_cos_refines_proved},
            apeiron_word.{word_q4_k_codes_ok,word_q6_k_codes_ok}
REFERENCE:  docs/epics/model-algebra.md, docs/epics/model-algebra.md, docs/evidence/rigor-baseline.md
            (gap list G1-G7), docs/epics/inference-parity.md, docs/direction.md, docs/kernel-boundary.md -
            derived from the intent's wave table and the rigor baseline's gap list, not invented
GATE:       document gate only: every cited path resolves and `git diff --check` is clean; no compiler
            gate in this planning task
GRADE:      anchor (planning); every mapped headline keeps the grade its file carries today or the
            target grade the spec names; this document raises no grade
```

### Goal and completion condition

The epic closes when the named headlines of [`docs/epics/model-algebra.md`](model-algebra.md)
build green at one source identity, each with a mutation control that turns it red, and the intent and
spec resolve as written. The wave-level barrier, run once at the barrier and not before:

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
proof/regen.sh --check
cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous AxiomAudit'
cd proof/ && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build'
```

The first wave's focused barrier, as the intent states it:

```sh
cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c \
  'LEAN_NUM_THREADS=1 lake build Nous.Model.ErrorComposition && LEAN_NUM_THREADS=1 lake build Nous.Model.Attention AxiomAudit'
```

A creation target that does not exist yet is named as a missing gate, never replaced by a proxy
(`Numeric.PackingRefinement`; `Nous.Model.Attention` and `Nous.Model.Instances` have since landed, see
the status below).

### Non-goals

- No float runtime theorem; ℝ is the reference domain, Q32.32 the executing domain, `ZMod (2^64)` the word domain.
- No GPU/SPIR-V execution semantics; that is [`docs/epics/gpu-spv.md`](gpu-spv.md).
- No completeness claim; an unlisted construct is out of domain.
- No capability, quality or watt claim; improvement is a measurement against a named baseline.
- No token-level correctness theorem; the artifact is the composed bound plus the runtime margin.
- No new canonical-corpus theorem, corpus pin change, or `proof/ApeironWord.lean` regeneration except at the
  extraction-root tasks the wave map names.

### Dependency: the engine split (owned by `docs/epics/inference-parity.md`)

`G3` refines engine kernels, but the kernels are not separately extractable while they share
`src/engine/mod.rs` with `Vec`/`HashMap`/IO. The split (`src/engine/mod.rs` 3477 lines → thin `mod.rs`
plus per-row modules, each ≤1,500) is the prerequisite task of the inference-parity plan (`W4`) and is
**not** re-owned here. This plan consumes its output: the kernel modules become the class-(a) roots.
The behavior-preserving control is the recorded decode checksum (`Engine::step` token-0 argmax `0`,
sum `0x4b39186d63e81773`) reproducing after the split.

### Wave map

One in-progress owner per file; tasks in one wave touch disjoint files. `proof/ApeironWord.lean`,
`target/`, `proof/.lake/` and `proof/corpus/.lake/` are one lane each. A task is ≤3 files, one arc, one
gate whose exit 0 closes it, and a control where soundness is at stake.

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W0 | G4 plan written | `docs/epics/model-algebra.md` | lead | document gate | n/a — planning |
| W0 | intent/spec drift correction and G5 rename | `docs/epics/model-algebra.md`, `docs/epics/model-algebra.md` | lead | document gate | n/a — planning |
| W1 | packing decoder extraction root (G3) | `proof/regen.sh`, `proof/ApeironWord.lean` (regenerated) | lead | `regen.sh --check`; `lake build Numeric.FormatRefinement` | perturb a `packing.rs` constant → regenerated Lean differs → `--check` red |
| W1 | packing decoder refinement | `proof/corpus/Numeric/PackingRefinement.lean` | deepseek-flash | `lake build Numeric.PackingRefinement AxiomAudit` | perturb the affine constant → exactness goal red |
| W1 | `rms_norm_fixed` / `rope_fixed` refinement (G3) | `proof/extract/lib.rs`, `src/engine/<kernel>.rs`, `proof/corpus/Numeric/EngineNormRefinement.lean` | deepseek-flash | `regen.sh --check`; module gate | perturb a coefficient → bound red |
| W1 | `matvec_fixed` / `row_dot_fixed` refinement (G3) | `proof/extract/lib.rs`, `src/engine/<kernel>.rs`, `proof/corpus/Numeric/EngineDotRefinement.lean` | deepseek-flash | `regen.sh --check`; module gate | drop the fixed-point floor → bound red |
| W2 | composed-margin barrier and G7 row | `Nous/Model/ErrorComposition.lean`, `proof/corpus/AxiomAudit.lean` | lead | `lake build Nous.Model.ErrorComposition AxiomAudit` | perturb a per-layer `ε` → `decision_margin_preserved` red |
| W2 | D2 per-layer hypothesis discharge | `Nous/Model/ErrorComposition.lean` | worker | `lake build Nous.Model.ErrorComposition` | flip an `ε` sign → red |
| W3 | G1 `attention_channel_form` | `Nous/Model/Attention.lean` | xiaomi-pro | `lake build Nous.Model.Attention` | drop the normalizer → red |
| W3 | G5 binary-softmax identification (restated) | `Nous/Model/Weighting.lean` | deepseek-flash | `lake build Nous.Model.Weighting` | remove `0 < u` → goal unprovable |
| W4 | F16/F32 identity faces (A5, in flight) | `src/format/packing.rs`, `proof/extract/lib.rs` | deepseek-flash | cargo trio; extraction probe | non-exact f32 round-trip → test red |
| W4 | split `src/format/packing.rs` | `src/format/packing.rs` → `src/format/packing/*.rs` | deepseek-flash | cargo trio; conversion-equality tests | n/a — names preserved |
| W5 | G2 fixed-σ instances | `Nous/Model/Instances.lean` | worker | `lake build Nous.Model.Instances AxiomAudit` | wrong `σ` → instance goal red |

Wave 0 has no control: it changes prose and planning only. Waves open in order; a missing theorem or
control keeps its wave open.

### Status at this revision (source `74c037b`; dirty tree)

The wave map above is the contract; this section records what the gates have closed at extraction
artifact `proof/ApeironWord.lean` `7b4a0831…`, 21265 lines. Digests are recomputed against the working
tree. `proof/regen.sh --check` was re-run at the wave barrier (exit 0); per-file digests and gate
outputs are in `docs/evidence/model-algebra-wave-receipt.md`.

- **G3 kernel refinements** (`proof/corpus/Numeric/`; barrier receipt
  `docs/evidence/model-algebra-wave-receipt.md`, `AxiomAudit.lean` `85c358a8…`, 1390 `#audit`):
  - **Attention.** `attn_head_refines_v2_proved` (`AttnHeadRefines.lean` `6f6d71f4…`) proves the
    corrected target on `attn_domain_v2` with no added hypothesis. The original `attn_head_refines` is
    refuted by `not_attn_head_refines`.
  - **SiLU, RMSNorm, row dot.** `silu_refines_total` (every `I64` input, `SiluSaturation.lean`),
    `rms_norm_refines_proved` and `rms_norm_refines_ideal` (against `x/√(mean(x²)+eps)`,
    `RmsNormIdeal.lean`), and `row_dot_refines_proved`.
  - **Head and sampler.** `lm_head_refines_proved`; the argmax laws.
  - **Dequant.** Row laws with witnesses for F16, Q8_0, Q5_0, Q2_K, Q3_K, Q4_K and Q6_K, and the
    F16/Q4_K/Q6_K refusal equalities.
  - **RoPE and packing.** RoPE head content plus a success-domain witness; packing `unpack_code_ref`
    and the `Fixed`-width epilogue folds.
  - **Block composition.** `Nous/Model/BlockGrid.lean` (`e566dc3a…`) builds the SiLU and RMSNorm
    block stages on explicit grid domains, not `Set.univ`. The matvec stage's defect is still `model`
    grade (no slice ↔ `Fin`-matrix bridge). RMSNorm enters as the scalar stage at a fixed radicand.
  - **Finding (vacuity).** `BlockResidual.mul_ref` pins the gate-product stage's `ref` to the exact
    elementwise product, and `Stage.expansion` is global. A product of unbounded inputs has no global
    Lipschitz constant, so `BlockResidual p h` is uninhabited for every `h ≥ 1`
    (the refutation, now `pinnedMul_stage_impossible` in `Nous/Model/BlockGroundedWitness1.lean`).
    `block_error_bound`, `block_error_bound'` and `block_error_bound_grounded` were therefore vacuous for
    any real FFN. Fixed: `mul_ref_on` (`BlockComposition.lean:384`) requires agreement with the product only
    on `mul.dom`; the refutation is kept as `pinnedMul_stage_impossible`. `Nous/Model/BlockMulBox.lean` adds:
    - a clamped-product stage (`mulBoxStage`, globally 2B-Lipschitz, equal to the product on its box);
    - `blockResidual_nonempty` (inhabited at every width);
    - `fullBlockGrounded_dom_nonempty_111`, the joint witness at `d = p = h = 1` with F16 projection
      data.

    Open: the gate product's kernel refinement. The instance executes the exact product with
    `delta = 0`; grounding it needs the FFN elementwise product extracted.
  - **Open items.** The KV append-success and constructor-projection laws. Open obligations are
    listed in the receipt.
- **Extraction roots** (14) cover the decoder path: `gguf_parse_header`, `q4_k`/`q6_k` codes,
  `packing_decode_block`, `silu`, `rms_norm`, `rope_rotate`, `row_dot`, `matvec`, `lm_head`,
  `kv_state_new`, `kv_append`, `attn_head`, `argmax`.
- **Parser cone closed**: `proof/patches/aeneas-lean-model.patch` (`64054e6b…`) makes the model `toStr`
  kernel-clean (`by rw [U32.max_eq]; decide`, the blocker was `U32.max` being an `irreducible_def`); all
  seven `word_gguf_parse_header_*` refusals report `{propext, Classical.choice, Quot.sound}`. GGUF stays
  the tested import boundary; the canonical packing decoder carries the proved claim.
- **Nous additions**: `SiluChannel.lean` (`cac8d044…`, the SiLU gate is the carrier channel weight),
  `RopeOrbit.lean` (`9d0c8440…`, the position-table group law), and `Attention.lean` W3-T3 (`d4ad5678…`,
  parabolic degeneration to the uniform average and the elliptic exclusion).
- **Cone**: `AxiomAudit.lean` (`32a4342f…`, 1255 `#audit`; job count not re-run here) green; the
  sampler/attn/matvec/lm_head headlines are registered.

### Task blocks

#### W0-T1 · G4 plan written

- **Scope:** this document; every intent/spec headline and every rigor-baseline gap maps to one task with a gate and a control.
- **Files:** `docs/epics/model-algebra.md`
- **Depends on:** none
- **Grade:** anchor
- **Gate:**

```sh
rg -q 'docs/(intent|spec|plan|evidence)/' docs/epics/model-algebra.md \
  && ! rg -q '[ \t]+$' docs/epics/model-algebra.md \
  && git diff --check
```

- **Negative control:** not applicable — planning; the gate asserts path resolution and whitespace only.
- **Evidence to record:** sha256 of this file; the resolved path list; the two-digit hex of `git rev-parse HEAD`.

#### W0-T2 · intent/spec drift correction and G5 rename

- **Scope:** correct descriptive drift found when writing this plan: the `channelWeight_binary_softmax`
  symbol does not exist (G5); the numeric line counts predate the E2 splits; `softmax_attention_is_online_spinor`
  is now registered (G6). No normative change to the abstraction, contract or domain.
- **Files:** `docs/epics/model-algebra.md`, `docs/epics/model-algebra.md`
- **Depends on:** W0-T1
- **Grade:** anchor
- **Gate:** `rg` path-resolution gate of W0-T1 over the two files plus `git diff --check`.
- **Negative control:** not applicable — descriptive correction; the governing source and gate win over the doc
  (`AGENTS.md` §1, failure register "Documents drifted from source").
- **Evidence to record:** before/after sha256 of both files; the quoted lines changed.

#### W1-T1 · packing decoder extraction root

- **Scope:** add `--start-from crate::word_packing_decode_block` to `proof/regen.sh` and regenerate, so the
  canonical decoder's real Rust body is the class-(a) root. Completion: `--check` green at the new identity.
- **Files:** `proof/regen.sh`, `proof/ApeironWord.lean` (regenerated)
- **Depends on:** `APEIRON-ALGEBRA-A5` (wrapper + F16/F32 faces)
- **Grade:** proved (extraction link)
- **Gate:**

```sh
AENEAS_HOME=/home/socol/Workspace/apeiron/target/aeneas proof/regen.sh --check
```

- **Negative control:** in a disposable copy, perturb a `src/format/packing.rs` constant; the regenerated
  `proof/ApeironWord.lean` differs and `regen.sh --check` exits non-zero; restore, `--check` green.
  `~/.cache/miri` must be absent for the run.
- **Evidence to record:** `proof/ApeironWord.lean` sha256 and line count; Aeneas/Charon/Lean revisions; the
  `--check` exit and the mutated-copy diagnostic.

#### W1-T2 · packing decoder refinement

- **Scope:** `Numeric/PackingRefinement.lean` proves `decode_block`'s exact integer law (the affine/code law
  over `ZMod (2^64)`) and its refusal branches, refining the extracted body to the spec's grammar. This is the
  one decoder proof; GGUF stays a tested import boundary.
- **Files:** `proof/corpus/Numeric/PackingRefinement.lean` (new), `proof/corpus/AxiomAudit.lean`
- **Depends on:** W1-T1
- **Grade:** proved
- **Gate:**

```sh
cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Numeric.PackingRefinement AxiomAudit'
```

- **Negative control:** perturb the affine offset or the scale decode in a throwaway copy; the exactness goal
  is red with an attributable diagnostic; restore green.
- **Evidence to record:** file sha256 and line count; the headline names; the `#audit` lines; the control diff.

#### W1-T3 · `rms_norm_fixed` / `rope_fixed` refinement

- **Scope:** root the two engine kernels and prove each extracted body denotes its stated fixed-point
  specification; register in the cone.
- **Files:** `proof/extract/lib.rs`, the engine kernel module the split produces, `proof/corpus/Numeric/EngineNormRefinement.lean` (new)
- **Depends on:** the engine split (`docs/epics/inference-parity.md` W4), W1-T1
- **Grade:** proved
- **Gate:** `AENEAS_HOME=… proof/regen.sh --check`; `lake build Numeric.EngineNormRefinement AxiomAudit`.
- **Negative control:** perturb a coefficient in a throwaway copy → bound red; restore green.
- **Evidence to record:** kernel source sha256; extraction identity; bound statement; control.

#### W1-T4 · `matvec_fixed` / `row_dot_fixed` refinement

- **Scope:** as W1-T3 for the accumulation kernels, including the fixed-point truncation term.
- **Files:** `proof/extract/lib.rs`, the engine kernel module, `proof/corpus/Numeric/EngineDotRefinement.lean` (new)
- **Depends on:** the engine split, W1-T1
- **Grade:** proved
- **Gate:** `AENEAS_HOME=… proof/regen.sh --check`; `lake build Numeric.EngineDotRefinement AxiomAudit`.
- **Negative control:** drop the floor term in a throwaway copy → bound red; restore green.
- **Evidence to record:** as W1-T3.

W1-T3 and W1-T4 share `proof/extract/lib.rs` and `proof.regen.sh`; they take separate lanes (one at a
time) or one wave each. The engine split is a single owner per file and is not parallel with either.

#### W2-T1 · composed-margin barrier and G7

- **Scope:** close G7: the plan and the barrier carry the landed composed-margin module and its control.
  `ErrorComposition.lean` already proves `norm_execComp_sub_refComp_le` (`:127`), `strictWinner_perturb`
  (`:218`) and `decision_margin_preserved` (`:256`); registration is present (`AxiomAudit.lean:1375-1390`).
- **Files:** `Nous/Model/ErrorComposition.lean`, `proof/corpus/AxiomAudit.lean`
- **Depends on:** W1-T3, W1-T4 (for the real-model instantiation in W2-T2)
- **Grade:** proved
- **Gate:**

```sh
cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Nous.Model.ErrorComposition AxiomAudit'
```

- **Negative control:** perturb a per-layer `ε` in a throwaway copy → `decision_margin_preserved` red; restore green.
- **Evidence to record:** file sha256; the three headline names and lines; the control diff.

#### W2-T2 · D2 per-layer hypothesis discharge

- **Scope:** instantiate `Layer`/`Lipschitz`/budget from the W1 extracted-kernel refinements, replacing
  `ErrorComposition.lean`'s stated per-layer hypotheses with proved instances. This is the step that makes the
  composed margin a statement about the executing model rather than a conditional lemma.
- **Files:** `Nous/Model/ErrorComposition.lean`
- **Depends on:** W1-T2, W1-T3, W1-T4
- **Grade:** proved
- **Gate:** `lake build Nous.Model.ErrorComposition AxiomAudit`.
- **Negative control:** flip an `ε` sign or a Lipschitz factor in a throwaway copy → red; restore green.
- **Evidence to record:** which hypotheses are discharged and which remain (the residual is stated, not hidden).

#### W3-T1 · G1 `attention_channel_form`

- **Scope:** `Nous/Model/Attention.lean` proves the streaming normalizer `Σ_i w_i` and weighted sum
  `Σ_i w_i v_i`, normalized once, equal `Σ_i (w_i / Σ_j w_j) v_i` on the factorizable domain; register in the cone.
- **Files:** `Nous/Model/Attention.lean` (new)
- **Depends on:** `Nous.Model.Decomposition` (landed), `AttentionSpinor.softmax_attention_is_online_spinor` (landed, G6 closed)
- **Grade:** proved
- **Gate:** `lake build Nous.Model.Attention`. Registration in `proof/corpus/AxiomAudit.lean` is a lead step at the W2-T1 barrier, not the worker's gate: the audit imports every lane, so a sibling lane mid-edit makes it transiently red (the cross-lane coupling the intent records).
- **Negative control:** drop the normalizer in a throwaway copy → red; restore green.
- **Evidence to record:** the theorem name and statement; the factorizable domain; the control.
- **Escalation:** xiaomi-pro; the open-bet domain question (hyperbolic-only vs degenerate fold) is the
  escalation trigger. If the domain needs an ordered/invertible setting the abstraction lacks, stop and
  escalate to an independent verifier.

#### W3-T2 · G5 binary-softmax identification (restated)

- **Scope:** the real obligation behind G5. `Weighting.lean:29-30` declines softmax equivalence as written;
  the target is the same ratio under the log substitution. For `u, l > 0`:
  `u / (u + l) = Real.exp (Real.log u) / (Real.exp (Real.log u) + Real.exp (Real.log l))`, i.e. the binary
  softmax of the log-channel pair `(log u, log l)`. State it on `ℝ` with the hypotheses `0 < u`, `0 < l`,
  named for the real symbols, and register it.
- **Files:** `Nous/Model/Weighting.lean` (the file that declines the claim; the repaired statement belongs beside its disclaimer)
- **Depends on:** none (`Weighting.lean` landed)
- **Grade:** proved
- **Gate:** `lake build Nous.Model.Weighting`. Registration in `AxiomAudit.lean` is a lead step at the W2-T1 barrier.
- **Negative control:** remove the `0 < u` hypothesis → the goal is unprovable (the identity fails for
  `u ≤ 0`), red; restore green.
- **Evidence to record:** the declaration and statement; the domain; the control.

#### W4-T1 · F16/F32 identity faces (A5, in flight)

- **Scope:** the native identity face of the packing grammar: f16 exact for finite values, Inf/NaN refuse;
  f32 round-to-nearest into Q32.32 with the module's tie convention, refusing overflow/underflow/NaN;
  `from_gguf_kind(0) = F32`, `(1) = F16`. Plus the extraction wrapper and the generated-Lean compile check.
- **Files:** `src/format/packing.rs`, `proof/extract/lib.rs`
- **Depends on:** none
- **Grade:** tested (element paths) → proved (with W1-T2)
- **Gate:** `cargo test --workspace --all-features --locked`; clippy; fmt; then the extraction probe of W1-T1.
- **Negative control:** a non-exact f32 rounding case → round-trip test red; restore green.
- **Evidence to record:** the F16/F32 contract as implemented; test names; the wrapper signature.

#### W4-T2 · split `src/format/packing.rs`

- **Scope:** behavior-preserving split of the now 1,711-line module into a thin `mod.rs` plus submodules,
  each ≤1,500, preserving every public name so `convert_row`/`from_gguf_kind` callers and the conversion-equality
  tests are unchanged.
- **Files:** `src/format/packing.rs` → `src/format/packing/*.rs`
- **Depends on:** W4-T1
- **Grade:** tested
- **Gate:** `cargo test --workspace --all-features --locked`; clippy; fmt.
- **Negative control:** not applicable to soundness — a pure move; the conversion-equality tests are the
  behavior-preserving evidence (before/after identical results for Q8_0/Q5_0/Q2_K/Q3_K/Q4_K/Q6_K and F16/F32).
- **Evidence to record:** before/after line counts; the test list; the equality results.

#### W5-T1 · G2 fixed-σ instances

- **Scope:** `Nous/Model/Instances.lean` fixes the three faces σ = +1 (hyperbolic), σ = 0 (parabolic),
  σ = −1 (elliptic) as canonical `ringMul` instances over the executing/word domains, each with an `#guard` or
  a refinement link where an extracted kernel exists.
- **Files:** `Nous/Model/Instances.lean` (new), `proof/corpus/AxiomAudit.lean`
- **Depends on:** `Nous.Model.Decomposition` (landed)
- **Grade:** proved (faces with a word kernel) / anchor (ternary, no extracted kernel exists)
- **Gate:** `lake build Nous.Model.Instances AxiomAudit`.
- **Negative control:** wrong `σ` (e.g. `σ = 2`) → the instance goal is red; restore green.
- **Evidence to record:** the per-face declaration and grade; the ternary face's excluded population and reason.

### Gap mapping (rigor baseline G1-G7)

| Gap | Task(s) | Status at plan time |
|---|---|---|
| G1 channel-form attention | W3-T1 | open — creation target `Nous/Model/Attention.lean` absent |
| G2 fixed-σ instances | W5-T1 | open — creation target `Nous/Model/Instances.lean` absent |
| G3 per-kernel executing-body refinement | W1-T1..T4 | open — packing root and two engine kernels unrefined |
| G4 written plan | W0-T1 | **this document** |
| G5 binary-softmax identification | W0-T2 (text), W3-T2 (proof) | corrected against source; restated target, open |
| G6 register the attention headline | — | **closed** (`AxiomAudit.lean:1372`) |
| G7 composed-margin row | W2-T1 | open until the barrier row and control are run |

To match the baseline, only G1-G3 and G5 are proof obligations; G4 and G7 are registration and planning;
G6 is closed. To exceed it, each headline must additionally be an extracted-body refinement where a kernel
exists, stay inside `{propext, Classical.choice, Quot.sound}`, and carry a mutation control.

### Size and cost

- `AGENTS.md` §4 limit is 1,500 owned lines per file. After the E2 splits, `Numeric/ExpRefinement.lean` (16)
  and `Numeric/RopeRefinement.lean` (15) are aggregators over parts ≤1,108; `Numeric/RsqrtRefinement.lean`
  (1,072) and `Numeric/FormatRefinement.lean` (1,117) are within. Remaining violations are
  `src/engine/mod.rs` (3,477, split under the inference-parity plan) and `src/format/packing.rs`
  (1,711, W4-T2).
- Slices per wave follow the intent's proof-cost budget (`docs/epics/model-algebra.md:106-112`); a wave
  records its observed slices against the budget at the barrier.
- The single kept `maxHeartbeats 8000000` on `word_fixed_exp_horner_acc` is a named accepted opening with
  owner acceptance pending; it is not silently discharged by this plan.

### Barrier checklist

Run at the wave barrier, not before:

- [ ] Every task gate exits 0, with its negative control run and restored.
- [ ] No task touched a file owned by another task; `proof/ApeironWord.lean` had one writer at a time.
- [ ] `cargo test`, `cargo clippy`, `cargo fmt` pass.
- [ ] `proof/regen.sh --check` reports fresh; bare `lake build` passes in `proof/` and `proof/corpus/`.
- [ ] `lake build AxiomAudit` passes; every new headline is `#audit`ed in the closed cone.
- [ ] Source and tool digests recorded; `proof/ApeironWord.lean` sha256 rebound to the receipts.
- [ ] Claims registered at their evidence grade; no citation called verified; targets still labelled targets.
- [ ] Trust-boundary statement present in the intent and spec; exclusions retained.
- [ ] Unresolved obligations at the barrier do not exceed the opening count (`AGENTS.md` §10).

### Stop and rollback conditions

- Budget expiry (600 s ceiling) is a run failure, not a red gate: stop, record elapsed time, exit status and
  incomplete stages, and report the blocker; do not split the run cosmetically.
- If the packing extraction root cannot produce generated Lean that compiles within budget, stop at W1-T1 and
  report the residual; do not substitute a hand-written model.
- If `attention_channel_form` requires a premise the abstraction does not hold (ordering, invertibility, a
  factorizable kernel the spec excludes), stop at W3-T1 and escalate to an independent verifier with the exact
  premise and the affected contract.
- If a control still passes under its stated mutation, the headline is not closed; redesign the control without
  weakening its contract.
- After three materially different failed approaches to one obstacle, stop and report the smallest unresolved goal.
