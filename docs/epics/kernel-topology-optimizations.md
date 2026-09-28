# Plan — Kernel topology: the native SiLU/RoPE kernels read topologically

- **Epic:** `APEIRON-ALGEBRA-TOPO-DOC`
- **Owner:** Apeiron lead
- **Plan:** [`docs/epics/model-algebra.md`](model-algebra.md)
- **Spec:** [`docs/epics/model-algebra.md`](model-algebra.md)
- **Style:** [`docs/epics/gpu-spv.md`](gpu-spv.md)
- **Source revision:** `74c037b1cfc79b0d6be6d3907398fe926d78d12d`; working tree dirty
- **Status:** draft

```text
SCOPE:      docs/epics/kernel-topology-optimizations.md; the topological reading of the native RoPE,
            SiLU and softmax kernels and the improvements that reading licenses. Completion = every
            improvement names its file and its licensing law, and every cited path resolves
REGISTER:   planning (algebraic reading + engineering inference); exclusions: no source, proof,
            extraction, corpus or claim-grade changes; no runtime measurement is claimed and none is run
CORPUS:     read and named, not restated: Sanguine.AttentionSpinor.{rotor_isometry,rotor_compose,
            rotor_full_turn,so2_spinor_recurrence,softmax_attention_is_online_spinor}
            (proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean);
            Nous.Model.{hyperbolic_decomposition,channel_diagonal_fold}
            (proof/corpus/Nous/Model/Decomposition.lean); Nous.Model.{siluCell_normForm,
            silu_gate_eq_channelWeight,siluReal_eq_mul_channelWeight}
            (proof/corpus/Nous/Model/SiluChannel.lean); Nous.Model.{channelWeight,softmax2,
            channelWeight_binary_softmax} (proof/corpus/Nous/Model/Weighting.lean);
            apeiron_word.{fixed_exp_refines_proved,fixed_sin_cos_refines_proved,silu_window_refines,
            word_silu_fixed_high,word_silu_fixed_low} (proof/corpus/Numeric/);
            apeiron_word.{word_mul_denote,word_norm_denote}
REFERENCE:  docs/epics/model-algebra.md, docs/epics/model-algebra.md, docs/epics/gpu-spv.md,
            docs/epics/inference-parity.md, docs/kernel-boundary.md; sources read in this tree — no
            external paper was retrieved for this plan and no external citation is called evidence
GATE:       document gate only: every backticked .rs/.lean/.md path resolves from the repository root,
            and `rg -n '[ \t]+$'` reports no trailing whitespace. Numeric wall-clock budget 300 s;
            no compiler or Lean gate is run in this planning task
GRADE:      anchor for every improvement (an engineering inference) and for the topological readings of
            the RoPE torus map, the exponential-family reading of softmax and the tropical limit; proved
            where a named in-repo theorem carries the algebraic law the reading uses, labelled at that
            law, not at the improvement. This document raises no grade and makes no measurement
```

## Result

The native kernels implement three algebraic objects that already have proved laws: the RoPE rotation
is the one-parameter `SO(2)` group of `rotor_compose`/`rotor_isometry`; the SiLU gate is the carrier's
binary channel weight of `silu_gate_eq_channelWeight`; and the causal softmax is the exact `(m, den,
num)` merge of `softmax_attention_is_online_spinor`. Reading each kernel on that object names four
concrete changes and one complexity comparison, with no new theorem and no runtime measurement. Every
constant factor below is an inference from source; none is measured, because the measurement branch was
cancelled by the owner and this document makes no benchmark or competitive claim.

## Improvements

Grades name the license, not the rewrite. `proved` = a Lean kernel check passes for the named theorem;
`anchor` = an engineering inference that the source and the reasoning support but no in-repo theorem
states.

| # | Improvement | File | Licensing law (grade) | Improvement grade |
|---|---|---|---|---|
| 1 | RoPE orbit generation from `ρ(1)` by the group law (`rhoMul` / rotor squaring) instead of one `fixed_sin_cos` per position | `src/engine/rope.rs` | `rotor_compose` proved (`proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean`); `fixed_sin_cos_refines_proved` proved (`proof/corpus/Numeric/Rope/Part4.lean`) | anchor — the Q32.32 orbit is not an exact homomorphism (rounding) |
| 2 | RoPE table hoist through the position index, removing the `n_layers·(n_head+n_head_kv)` recompute | `src/engine/rope.rs`, `src/engine/step.rs` | the angle depends only on `(pos, j)` (`src/engine/rope.rs`, `src/engine/step.rs`); no in-repo theorem states cached = recomputed | anchor |
| 3 | Fuse `SiLU` into the gate-projection epilogue | `src/engine/ffn.rs`, `src/engine/silu.rs` | `silu_gate_eq_channelWeight`, `siluReal_eq_mul_channelWeight` proved (`proof/corpus/Nous/Model/SiluChannel.lean`); `silu_window_refines` proved (`proof/corpus/Numeric/EngineSiluRefinement.lean`) | anchor |
| 4 | Linearise the FFN fold to additive natural (log-channel) coordinates | `src/engine/ffn.rs` | `channel_diagonal_fold` proved (`proof/corpus/Nous/Model/Decomposition.lean`); `siluCell_normForm` proved (`proof/corpus/Nous/Model/SiluChannel.lean`) | anchor |
| 5 | Keep attention on the `O(1)`-state LSE monoid instead of the `O(T)` score/weight tables | `src/engine/attn.rs` | `softmax_attention_is_online_spinor` proved (`proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean`) | proved (the law) / anchor (the executing rewrite) |
| 6 | Vectorise the projection loop | `src/engine/matvec.rs` | `ringMul_assoc` through `token_step_fold` proved (`proof/corpus/Nous/Model/State.lean`) | anchor |
| 7 | Evaluate `exp` once per fundamental domain of the `ln2·ℤ` cocycle and reuse it | `src/numeric/fixed_exp.rs` | `fixed_exp_refines_proved` proved (`proof/corpus/Numeric/Exp/Part4.lean`) | proved (present body) / anchor (reuse scheme) |

## 1 · RoPE: one-parameter subgroup, covering section, diagonal quotient

### Reading

Let `d = head_dim`, `n = d/2`, and `ω_j = θ^(-2j/d)` for `j < n`. Define

```text
ρ : ℤ → T^n,   ρ(pos) = (pos · ω_j mod 2π)_{j < n}.
```

Then `ρ(pos₁ + pos₂) = ρ(pos₁) + ρ(pos₂)` in the torus group `T^n`, so `ρ` is a group homomorphism; its
image is a one-parameter subgroup. It is a closed geodesic of `T^n` when the `ω_j` are pairwise
commensurable with `2π`, and dense when they are not (Kronecker's theorem, 1884). No in-repo Lean
theorem states this map; the reading is an **anchor**.

The native table `(cos(pos·ω_j), sin(pos·ω_j))_j` is a section of the covering `ℝ^n → T^n`: the Q32.32
`fixed_sin_cos` selects one branch inside the trig window `[-4096, 4096]` radians (`src/numeric/fixed_trig.rs`).
The correctness obligation of that section is already named and proved as the per-output bound
`fixed_sin_cos_refines_proved` (`proof/corpus/Numeric/Rope/Part4.lean`), composed into the rotation in
`proof/corpus/Numeric/EngineRopeRefinement.lean`.

Relative position lives on the diagonal quotient `(ℤ × ℤ)/diag`: the score reads
`⟨R(θ_q)φ_q, R(θ_k)φ_k⟩ = ⟨φ_q, R(θ_k − θ_q)φ_k⟩` because `R(θ)ᵀR(φ) = R(φ − θ)`. The one-parameter
group law is `rotor_compose` (`proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean`, proved over
ℝ); the isometry that keeps the carry norm fixed is `rotor_isometry` (same file, proved). The
fixed-point table inherits the ℝ law only up to the proved `2^-20` trig bound, so the quotient
statement on the executing face is an **anchor**.

### Improvements

1. **Orbit generation.** Replace one `fixed_sin_cos` evaluation per position with the group law:
   `R(ρ(pos+1)) = R(ρ(pos)) ∘ R(ρ(1))` (one `rotor_compose`-licensed composition per step, `O(1)` per
   rotary dimension), or `O(log pos)` compositions by repeated squaring `R(2θ) = R(θ)²` for random
   access. The ℝ law is `rotor_compose` (proved). The Q32.32 orbit is an **anchor**: composition of two
   rounded rotations is not exactly a rounded rotation, so only the `fixed_sin_cos_refines_proved` bound
   carries over.
2. **Table hoist.** The rotation operator `R(θ(pos, j))` depends only on `(pos, j)`, i.e. it is constant
   on the head and layer fibers. The current path does not exploit this: `src/engine/step.rs` calls
   `rotate_one` for `query` and for `key` inside the `while layer < n_layer` loop, and
   `RopeTable::rotate_head` (`src/engine/rope.rs`) evaluates `fixed_sin_cos(pos · freqs[j])` once per
   `(head, j)` pair. The same `(pos, j)` pair is therefore recomputed
   `n_layers · (n_head + n_head_kv)` times per token. Caching the `(pos, j) → (cos, sin)` pair once per
   position and indexing it removes that factor. Grade **anchor**: the equality of the cached and
   recomputed pair is source-visible but unproved.

The load-time frequency table itself is already position-independent and shared
(`src/engine/mod.rs`); the improvement is on the per-position section, not on `freqs`.

## 2 · SiLU and softmax: channel weight, moment map, log-partition

### Reading

The sigmoid-linear unit carries the carrier's binary channel weight. With
`siluCell x = ((exp x + 1)/2, (exp x − 1)/2)` in `R[x]/(x² − 1)`:

- `siluCell_normForm` : `ringNormForm 1 (siluCell x) = exp x` — the exponential is the norm form, so its
  logarithm is the natural parameter of the point (proved, `proof/corpus/Nous/Model/SiluChannel.lean`).
- `silu_gate_eq_channelWeight` : `siluGate x = channelWeight 1 (siluCell x)` (proved, same file).
- `siluReal_eq_mul_channelWeight` : `siluReal x = x · channelWeight 1 (siluCell x)` (proved, same file).

The channel coordinates diagonalise the fold: `channel_diagonal_fold` proves
`channelPair r (tokenFold σ xs) = ((xs.map (ringUpper r)).prod, (xs.map (ringLower r)).prod)` and the
matching `memoryStep` law (`proof/corpus/Nous/Model/Decomposition.lean`, proved). In log-channel
coordinates the products become sums, which is the additive form the fold takes on the natural
coordinates. The step is the standard group homomorphism `log`; the multiplicative half is the proved
`channel_diagonal_fold`, the additive half is an **anchor**.

Softmax is the moment map of the categorical exponential family and its log-partition is the
log-sum-exp. `softmax2 x y = exp x / (exp x + exp y)` (`proof/corpus/Nous/Model/Weighting.lean`); the
binary identification `channelWeight_binary_softmax` and `ratio_eq_softmax2_log` are proved in the same
file (both `#audit`ed in `proof/corpus/AxiomAudit.lean`), with the domain `0 < u`, `0 < l`. The
exponential-family reading (moment map, natural parameter, log-partition) is an **anchor / model**, not
a proved statement: no in-repo theorem names the moment map. Attribution: Boltzmann (1877), Gibbs
(1902); the name softmax is Bridle (1990).

The `β → ∞` limit of `softmax(β·s)` is the tropical (max-plus) boundary, and the `β = 1` / large-score
regime is the ReLU-like switch. The native code already realises the finite-`β` boundary explicitly:
`src/engine/attn.rs` subtracts the exact maximum (`shifted = score − mx`) and saturates the numerator to
`0` below `-16·ONE`; `src/engine/silu.rs` saturates at `v ≤ -21·ONE → 0` and `v ≥ 16·ONE → v`. The
saturation arms are proved as `word_silu_fixed_high`, `word_silu_fixed_low` and `silu_window_refines`
(`proof/corpus/Numeric/EngineSiluRefinement.lean`). The tropical-limit reading itself is an **anchor**:
no in-repo theorem takes `β → ∞`.

The `O(1)`-state LSE monoid is proved: `softmax_attention_is_online_spinor`
(`proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean`) shows the `(m, den, num)` state merges by
exact `+` in the drag and numerator channels (`oaDen_fuse`, `oaNum_fuse`) and that the scan computes
`(Σ e^{sc}·v)/(Σ e^{sc})` exactly with three running numbers, independent of context length.

### Improvements

1. **Additive natural coordinates.** Where the fold is a product of channel weights, work in
   `log channel` so the fold is a sum. Licensed by `channel_diagonal_fold` (proved) and
   `siluCell_normForm` (proved); the executing rewrite is an **anchor**.
2. **Fundamental-domain `exp`.** `fixed_exp` already reduces by the `ln2·ℤ` cocycle: `exp_reduce`
   (`src/numeric/fixed_exp.rs`) computes `k = round(x / ln2)` and `r = x − k·ln2` with `|r| ≤ ln2/2`, then
   `exp(x) = 2^k · exp(r)`; the degree-7 Horner series is evaluated only on that fundamental domain, and
   `fixed_exp_refines_proved` (`proof/corpus/Numeric/Exp/Part4.lean`) bounds the result. The improvement
   is a reuse/hoist of `exp(r)` across calls sharing an argument, not a change to the reduction; the
   reuse scheme is an **anchor** and the present body is **proved**.
3. **LSE monoid for attention.** Replace the materialised `scores`, `exps` and `weights` vectors of
   `KvState::attn_head` (`src/engine/attn.rs`) with the one-pass `(m, den, num)` merge of
   `softmax_attention_is_online_spinor` (proved). The benefit is spatial (`O(T)` scratch removed), not
   a new temporal order.

## 3 · Complexity

All numbers are **inferred** from the source read in this tree. No row is `measured`: the measurement
branch was cancelled by the owner, and this document makes no benchmark, ratio or competitive claim.
`d = head_dim`, `H = n_head + n_head_kv`, `L = n_layers`, `T = pos + 1`, `n_ff` = FFN hidden width.

| Operation | Native | Ours (proposed) | Temporal | Spatial | Asymptotic in `d`, `T` | Number's basis |
|---|---|---|---|---|---|---|
| RoPE pair rotation | `fixed_sin_cos` per rotary pair: range reduction + degree-8 Horner, then 4 `i128` mults + 2 round-shifts (`src/engine/rope.rs`) | table read + 4 `i128` mults + 2 round-shifts | `O(H·d)` per token | `O(1)` per pair | `O(d)` both | inferred |
| RoPE per-position section | one `fixed_sin_cos` per `(pos, j)`; recomputed `L·H` times per token | `(pos, j)` pair cached once; `O(1)` `rhoMul` per step, `O(log pos)` by squaring | removes factor `L·H` | `O(d)` table per position | `O(d)` both | inferred |
| SiLU activation | `silu_fixed` per `n_ff` row: one `fixed_exp` + one rounded division each (`src/engine/silu.rs`) | unchanged count; fused into the gate epilogue | `O(n_ff)` | `O(n_ff)` materialised → fused | `O(n_ff)` both | inferred |
| FFN block | gate matvec + up matvec + activation + down matvec (`src/engine/ffn.rs`) | fused gate/activation epilogue | `O(n_ff·n_embd)` | `O(n_ff)` scratch | equal | inferred |
| Causal softmax attention | two score/exps passes + weights pass + output pass; scratch `scores`, `exps`, `weights` (`src/engine/attn.rs`) | one `(m, den, num)` pass; state `num` vector + 2 scalars | `O(T·d)` | `O(T)` → `O(d)` | temporal `O(T·d)` both; spatial `O(T)` vs `O(d)` | inferred |
| `exp` | range reduce + degree-7 Horner on `|r| ≤ ln2/2` (`src/numeric/fixed_exp.rs`) | reuse `exp(r)` for shared `r` | `O(1)` per call | `O(1)` | `O(1)` both | inferred |
| Projection | scalar `matvec_fixed` per output row (`src/engine/matvec.rs`) | vectorised rows | `O(n_out·n_in)` | `O(1)` extra | equal | inferred |

Asymptotic order in `d` and `T` is unchanged by every improvement; what changes is the constant factor
(transcendental implementation, table caching, the recompute factor `L·H`) and the attention scratch
from `O(T)` to `O(d)`. A speed or throughput comparison would need a named machine, metric and interval
and a validated instrument (obligation 10 of `AGENTS.md`); that evidence does not exist here and is not
claimed.

## 4 · Ranked plan

Each task names the file it touches and the law that licenses it. These are candidate tasks; none is
started by this document, and each is a ≤3-file, one-gate task when opened.

1. **RoPE table hoist and orbit generation.** Files `src/engine/rope.rs` (a per-position
   `(cos, sin)` index, an orbit step, or both) and `src/engine/step.rs` (hoist the `rotate_one` calls
   out of the layer loop / index the cached section). License: `rotor_compose`, `rotor_isometry`
   (proved, `proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean`) for the group law;
   `fixed_sin_cos_refines_proved` (proved, `proof/corpus/Numeric/Rope/Part4.lean`) for the Q32.32 bound;
   the refinement `proof/corpus/Numeric/EngineRopeRefinement.lean` must be re-established for the new
   body. Gate: `cargo test --workspace --all-features --locked`; `cd proof/corpus && lake build
   Numeric.EngineRopeRefinement AxiomAudit`. Control: perturb a frequency → the rotation test and the
   refinement bound are red. Grade: proved for the ℝ group law, anchor for the executing orbit.
2. **Fuse `SiLU` into the gate-projection epilogue.** Files `src/engine/ffn.rs` and `src/engine/silu.rs`
   (apply the activation to each gate row as the projection writes it, instead of materialising
   `gate_vec`). License: `silu_gate_eq_channelWeight`, `siluReal_eq_mul_channelWeight` (proved,
   `proof/corpus/Nous/Model/SiluChannel.lean`); `silu_window_refines` (proved,
   `proof/corpus/Numeric/EngineSiluRefinement.lean`); `channel_diagonal_fold` (proved,
   `proof/corpus/Nous/Model/Decomposition.lean`) for the diagonal reading. Gate: `cargo test
   --workspace --all-features --locked`; `cd proof/corpus && lake build Numeric.EngineSiluRefinement
   AxiomAudit`. Control: perturb the epilogue rounding → the in-window refinement is red. Grade: anchor.
3. **Vectorise the projection loop.** File `src/engine/matvec.rs`. License: `ringMul_assoc` through
   `token_step_fold` (proved, `proof/corpus/Nous/Model/State.lean`) fixes the reassociation the vector
   reduction performs; `word_mul_denote` fixes the per-word product. Gate: `cargo test --workspace
   --all-features --locked`; `cd proof/corpus && lake build Numeric.EngineDotRefinement AxiomAudit`
   (creation target). Grade: anchor — no per-instruction proof; stays `tested` at best until the
   refinement lands.
4. **Cheaper `exp` on the fundamental domain.** File `src/numeric/fixed_exp.rs` (a shared `exp(r)` for
   arguments with equal `r`, or a lookup over the `ln2·ℤ` fundamental domain). License:
   `fixed_exp_refines_proved` (proved, `proof/corpus/Numeric/Exp/Part4.lean`) for the present body; any
   replacement needs its own refinement before it can be called proved. Gate: `cargo test --workspace
   --all-features --locked`; `cd proof/corpus && lake build Numeric.ExpRefinement AxiomAudit`. Control:
   perturb the Horner coefficient → the refinement bound is red. Grade: proved (present body) / anchor
   (replacement).

Ordering is by touched surface and by how directly a proved law licenses the change; (1) and (2) sit on
the RoPE and FFN kernel paths already extracted in `proof/extract/lib.rs`, which is why their laws
already exist. (3) and (4) are lower ranked because their gates are behavior-preserving or
create a new refinement.

## 5 · Trust boundary

This document is a reading of proved algebraic laws plus engineering inference. It does not claim a
runtime measurement, a speed or throughput comparison, a numerical accuracy improvement, or a token or
capability result. The topological statements on `ℝ` (`rotor_compose`, `rotor_isometry`, the
one-parameter subgroup and its diagonal quotient, the exponential-family reading, the tropical limit)
do not transport exactly to the Q32.32 executing face: only the combined bounds
`fixed_sin_cos_refines_proved`, `fixed_exp_refines_proved`, `silu_window_refines` carry, at `2^-20`
relative plus a fixed floor. `rustc`/LLVM codegen, the allocator, the kernel ABI, `unsafe` containers
and GPU/SPIR-V remain outside every claim here, as named in `docs/kernel-boundary.md` and
`docs/epics/model-algebra.md`. The `Z/2^w` word face degenerates at `2` and is the reason the channel
bijection is not instantiated there (`docs/epics/model-algebra.md`; `proof/corpus/Spirv/Semantics.lean`).
The one measurement that would strengthen any row of §3 is a named command, machine, metric and
interval with a validated instrument; the measurement branch was cancelled by the owner, so none is
claimed.

## 6 · Gate

Document gate for this file:

```sh
# every backticked source or document path resolves from the repository root
rg -N -o '`[^`]+\.(rs|lean|md)`' docs/epics/kernel-topology-optimizations.md \
  | tr -d '`' | sort -u \
  | while IFS= read -r p; do test -e "$p" || printf 'MISSING %s\n' "$p"; done
# no trailing whitespace (rg exits 1 and prints nothing when clean)
rg -n '[ \t]+$' docs/epics/kernel-topology-optimizations.md
```

A replacement or extension of this plan re-runs the two commands; a source or proof change is out of
scope here and takes its own task with the gate and control named in §4.
