# ternary and binary GGUF faces with an exact Walsh–Hadamard rotation, and exact one-bit faces

Merged epic record: `intent/ternary-hadamard.md`, `spec/ternary-hadamard.md`, `plan/ternary-hadamard.md` (one file per epic; see `docs/README.md`).

## Status at this revision

Grades below were re-verified by the lead: gate exit 0, `#print axioms` inside the closed cone, and
the seeded control re-run red. Extraction is `proof/ApeironWord.lean` `72a8b7d4…` (23436 lines;
`regen.sh --check` clean). Every headline is registered in `proof/corpus/AxiomAudit.lean`.

| wave | row | grade | evidence |
|---|---|---|---|
| W1 | six dequant arms (Q1_0, Q2_0, PQ2_0, TQ2_0, TQ1_0, PTQ1_0) against fork-layout vectors | tested | `src/engine/dequant_lowbit.rs` tests; `cargo test` 403 passed |
| W1 | `fwht_fixed` against naive Sylvester for k ≤ 10 | tested | `src/engine/hadamard.rs` tests |
| W2 | extraction of the arms and `word_fwht_fixed` | proved (identity) | `regen.sh --check` exit 0 |
| W3 | row laws and dispatch, all six types | proved | `Numeric/DequantQ1Q2(Block)`, `DequantTQ2_0`, `DequantTQ1_0(Row)`, `DequantPTQ1_0` |
| W3 | refusals (short payload, non-finite scale), all six types | proved | `Numeric/DequantLowBitRefusals` |
| W3 | trit decode = base-3 digits of the fork quantizer | proved | `trit_spec`, `trit_digits` |
| W3 | Sylvester algebra, `H·H = 2^k·I`, `H_2 = channelPair 1` at `σ = 1`, `¬IsUnit 2` | proved | `Numeric/Sylvester` |
| W3 | exact integer FWHT = `sylvesterH k` per aligned segment | proved | `fwht_int_spec`, `fwht_int_sylvester` (`Numeric/HadamardRefinement`, `HadamardStage`) |
| W3 | normalized FWHT, one rounding: `|x' − y/(q32·√2^k)| ≤ 1/(2q32) + |y|/q32·|C_k/q32 − 2^{−k/2}|` | proved | `fwht_fixed_spec`, `fwht_fixed_refines`, `fwht_fixed_refines_sylvester` (`Numeric/HadamardFixedSpec`); table `C_close` |
| W4 | `prism.hadamard.*` loader with the fork's rejection rules | tested | `src/engine/loader.rs` tests |
| W4 | folded projection path (signs, FWHT, then matvec) matches the plain weight | tested | `src/engine/step.rs` `hadamard_fold_*`; version 2 (tied output) is refused, not executed |
| W4 | rotation as a block `Stage` (sup-norm expansion `√2^k`, ℓ2 isometry, involution) | proved | `Nous/Model/RotationStage` |
| W5 | token parity and throughput against the fork on a Bonsai 2 GGUF | open (measured) | quiet-host session; needs the model download |

Open beyond W5:
- version-2 tied-output execution (the inverse transform on the tied embedding);
- the fork's tensor-directory cross-checks in the loader (tied `output.weight` absent, architecture
  routing);
- the loader widens bool-typed sign arrays to 0/1, which the ±1 check then refuses.

The algebraic reading below is proved by the W3 Sylvester row; it is no longer `anchor`.

## Intent — ternary and binary GGUF faces with an exact Walsh–Hadamard rotation

- **Owner:** lead (Claude track)
- **Source revision:** `74c037b`; working tree dirty (model-algebra wave, see
  `docs/epics/model-algebra.md`)
- **Status:** draft
- **Spec:** `docs/epics/ternary-hadamard.md`
- **Plan:** `docs/epics/ternary-hadamard.md`

### Problem

`dequant_row_fixed` (`src/engine/dequant.rs:86`) reads F16, Q8_0, Q5_0, Q2_K, Q3_K, Q4_K and Q6_K
only. The low-bit formats that put 8B–27B models on consumer devices are absent:

- binary `Q1_0` (type 41);
- the four-level `Q2_0` (42, codes {−1, 0, +1, +2});
- ternary `TQ1_0` (34, base-3 dense) and `TQ2_0` (35, 2-bit);
- the PrismML fork's group-128 `PQ2_0` (142) and `PTQ1_0` (143, 1.75 bits per weight).

PrismML's Bonsai 2 27B ships as `PTQ1_0`/`PQ2_0`/`Q2_0` and stores its weights in a
Walsh–Hadamard-rotated basis. Inference must apply the matching activation transform before each
folded matmul; without it the model outputs gibberish. Apeiron has no Hadamard kernel.

Sources, retrieved and read:
- `PrismML-Eng/llama.cpp@adfffbe`: `ggml/src/ggml-common.h:185-316` (block layouts),
  `ggml/src/ggml-quants.c:454-510,2203-2260` (reference dequantizers),
  `ggml/src/ggml-cpu/ops.cpp:12066` (float FWHT), `src/llama-graph.cpp:1555-1578` (sign flip,
  FWHT, then matmul), `conversion/base.py:620-763` (the `prism.hadamard.*` GGUF keys).
- `PrismML-Eng/Bonsai-demo@9ef3205`: `MODEL-FORMATS.md`, `BACKEND-SUPPORT.md`.

Scratch notes: `/home/socol/oc-tmp/spike-prism/FINDINGS.md`.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Link or call the fork's C kernels | No Rust/Aeneas path: no extraction, no refinement. The claim would be `tested` at best, against a float reference. |
| Port the float FWHT as is | `k = log₂ n` rounding stages with no stated bound. The Q32.32 engine is integer, and a float stage breaks the exactness the dequant row laws rely on. |
| Treat the rotation as a dense `n × n` matmul | O(n²) per block instead of O(n log n). It also hides the Sylvester structure the proof uses. |
| Dequantize to Q8_0 at load | Doubles to quadruples resident weight memory. That defeats the consumer-device purpose of 1.75–2.25 bits per weight. |

### The algebraic reading (proved: `Numeric/Sylvester.lean`)

The Sylvester Hadamard matrix is `H_{2^k} = H_2^{⊗k}` with `H_2 = [[1, 1], [1, −1]]`.

`H_2` is `channelPair 1` at `σ = 1`: the change to the idempotent basis `e± = (1 ± x)/2` of the
split-complex face `R[x]/(x² − 1)` (Cockle, 1848). `hyperbolic_decomposition`
(`proof/corpus/Nous/Model/Decomposition.lean:148`) makes that map a ring isomorphism when `2` is
invertible (`2h = 1`). Over ℤ it is not: `H_2 · H_2 = 2·I`. That is exactly the factor `1/√2` per
stage that the float FWHT carries.

Over Q32.32 integers the unnormalized butterflies are exact. All normalization therefore folds into
one final rounding. That is the intended improvement over the float path, which rounds at every
stage.

### Completion condition

Every row in the plan's exit table is green on one source/tool identity:
- the six dequant arms and `fwht_fixed` are extracted from the real source;
- each has a proved row law or refinement under the closed axiom cone;
- `cargo test` covers the nominal and refusal paths;
- a Bonsai-format GGUF block round-trips.

```sh
cargo test --workspace --all-features --locked
AENEAS_HOME=/home/socol/Workspace/apeiron/target/aeneas proof/regen.sh --check
cd proof/corpus && lake build AxiomAudit
```

### Invariants

1. Every dequant arm is exact: `value = d · code` in Q32.32, with the f16 scale decoded by
   `f16_to_fixed_exact` and no rounding. A non-finite scale or a short payload refuses
   (`Result::Err`), never saturates.
2. `fwht_fixed` is total on its checked domain and refuses on `i64` overflow. It rounds exactly once:
   the final normalization by `c ≈ 2^32/√n`.
3. The rotation is applied only to activations entering a weight named in
   `prism.hadamard.weight_names`, with the sign vector first. A folded weight without its transform
   is a load-time refusal, as `llama_verify_hadamard_graph` enforces in the fork.
4. No existing dequant arm, row law or digest-bound claim changes meaning.

### Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| each new dequant arm equals `d·code` exactly, row by row | proved | `dequant_<t>_row_spec` per type |
| the arms agree with the fork's reference dequantizer on shared blocks | tested | Rust test against byte vectors built to the fork's `ggml-common.h` layout |
| unnormalized integer FWHT equals `H_{2^k} · (s ⊙ x)` exactly | proved | `fwht_int_spec` |
| `H_{2^k} · H_{2^k} = 2^k · I` and `H_2 = channelPair 1` at `σ = 1` | proved | `sylvester_mul_self`, `sylvester_one_channelPair` |
| normalized output within `1/(2·2^32) + ‖H(s⊙x)‖∞ · |c/2^32 − 1/√n|` of the real normalized transform | proved | `fwht_fixed_refines` |
| end-to-end token parity with the fork on a Bonsai GGUF | measured | quiet-host session, named command and machine |
| fewer rounding events than the fork's float FWHT (1 vs `log₂ n`) | model/anchor | reading of `ops.cpp:12066` against `fwht_fixed`; becomes measured with the parity run |

### Non-goals

- The fork's SIMD/GPU kernels, or FWHT on GPU.
- Training or quantizing models (`quantize_ptq1_0`) in Part I; load and inference only. Conversion of
  existing models onto one-bit faces is Part II.
- KV-cache 4-bit and mean-centering (`KV-CACHE.md`).
- Performance claims before the quiet-host measurement session.

### Trust boundary

Refinement reaches the extracted Rust bodies (Aeneas model of the source). It does not cover
rustc/LLVM codegen, the GGUF file reader beyond the tested import boundary, the fork's float
reference (it is a comparison target, not evidence), or GPU paths.

### Open questions

- Bonsai 2's `prism.hadamard.block_size`: read from the GGUF metadata at the measurement session. If
  `k` is odd, `c` is irrational and carries a proved coefficient error. If `k` is even, `c = 2^{32−k/2}`
  is exact. Owner: lead.
- Model download (5.9–7.6 GB) for the parity run needs the quiet-host session. Owner: user.

## Specification — ternary and binary GGUF faces with an exact Walsh–Hadamard rotation

- **Owner:** lead (Claude track)
- **Intent:** `docs/epics/ternary-hadamard.md`
- **Source revision:** `74c037b`; working tree dirty
- **Status:** draft

### Scope and admissible domain

#### Dequant arms (`src/engine/dequant.rs`, dispatched by `dequant_row_fixed`)

Common to all six arms: `d` is the block's little-endian f16 scale, decoded exactly to Q32.32 by
`f16_to_fixed`. `n_elems` need not be a multiple of the block size; the last block is partial, as
in the existing arms.

| type | id | QK | bytes | layout | weight `i` of a block |
|---|---|---|---|---|---|
| `Q1_0` | 41 | 128 | 18 | `d`@0, `qs[16]`@2 | `bit(qs[i/8], i%8) ? d : −d` |
| `Q2_0` | 42 | 64 | 18 | `d`@0, `qs[16]`@2 | `(((qs[i/4] >> 2(i%4)) & 3) − 1)·d` |
| `PQ2_0` | 142 | 128 | 34 | `d`@0, `qs[32]`@2 | as `Q2_0` over 128 |
| `TQ2_0` | 35 | 256 | 66 | `qs[64]`@0, `d`@64 | for `j ∈ {0,32}`, `l < 4`, `m < 32`: out index `(j/32)·128 + 32l + m`, code `(qs[j+m] >> 2l) & 3`, value `(code − 1)·d` |
| `TQ1_0` | 34 | 256 | 54 | `qs[48]`@0, `qh[4]`@48, `d`@52 | staged trits (below), stages `[32, 16]` over `qs`, then `qh` |
| `PTQ1_0` | 143 | 128 | 28 | `qs[24]`@0, `qh[2]`@24, `d`@26 | staged trits, stages `[32, 16, 8]` (effective `[16, 8]`), then `qh` |

**Staged trits.** Output order follows `dequantize_row_tq1_0` (`ggml-quants.c:2599`) and
`dequantize_row_ptq1_0` (`:2255`):
- for each stage width `c`, while `j + c ≤ |qs|`: for `n < 5`, `m < c`, emit
  `trit(qs[j+m], n)`; then `j += c`;
- then for `n < 4` and `h < |qh|`, emit `trit(qh[h], n)`.

Here `trit(b, n) = (((b · 3^n) mod 256) · 3) >> 8 ∈ {0, 1, 2}`, and the value is `(trit − 1)·d`.
The Rust body computes it in `u16` with an explicit `& 0xFF`, with no wrapping multiply, so Aeneas
extracts it.

| Item | Value |
|---|---|
| Operation | `dequant_row_fixed(t, raw, n, out)` for `t ∈ {34, 35, 41, 42, 142, 143}` |
| Domain | `raw` holds `⌈n/QK⌉` whole blocks; every block's `d` finite |
| Totality | `Result`: `Err(Quant(Truncated))` on a short payload, `Err(NonFiniteScale)` on a non-finite `d` |
| Postcondition | `out' = out ++ [value(i) for i < n]`, each value exact in Q32.32 |

`|value| ≤ |d| ≤ 65504·2^32 < 2^49`, so no product overflows `i64`. The arms need no overflow
refusal; the proofs record this as a bound, not a hypothesis.

#### Rotation (`src/engine/hadamard.rs`, new)

| Item | Value |
|---|---|
| Operation | `fwht_fixed(x: &mut [i64], signs: Option<&[i8]>, block: usize) -> Result<(), HadamardError>` |
| Domain | `block = 2^k`, `1 ≤ k ≤ 16`; `x.len()` a multiple of `block`; `signs` (if present) of length `x.len()` with entries in {−1, +1}; `|x_i|` small enough that no checked butterfly overflows |
| Totality | `Result`: `Err(NotPowerOfTwo)`, `Err(LengthMismatch)`, `Err(BadSign)`, `Err(Overflow)` |
| Postcondition | per block `B`: `x'_B = round(c · H_{2^k} · (s_B ⊙ x_B) / 2^32)`, where `c` is the Q32.32 constant for `1/√(2^k)` (exact `2^{32−k/2}` for even `k`, `round(2^32/√2^k)` for odd `k`) |

Algorithm:
1. multiply by `s` (checked negation);
2. run `k` in-place butterfly stages `(u, v) ↦ (u + v, u − v)` with checked `i64` add/sub, `len = 1, 2, …, 2^{k−1}`;
3. normalize once: `divRoundNearest(y · c, 2^32)` in `i128`, narrowing checked.

Complexity: `O(n log n)` checked add/sub and `n` multiplications per block of `n = 2^k`; O(1) extra
space (in place). Recurrence: stage `len` touches every element once, and there are `k` stages.

#### Loader metadata (`src/engine/loader.rs`)

Read `prism.hadamard.{version, block_size, transform, axis, sign_mode, weight_names, sign_widths,
sign_values, tied_output}` exactly as `src/llama-model.cpp:1197-1263` of the fork validates them.
Any value the fork rejects is a load refusal here. Weights named in `weight_names` are tagged as
folded; the matmul path applies `fwht_fixed` to the activation before them.

### Representation relation (Rust to Lean)

| Layer | Symbol | Path |
|---|---|---|
| Production Rust | `dequant_<t>_row`, `fwht_fixed` | `src/engine/dequant.rs`, `src/engine/hadamard.rs` |
| Extraction wrapper | `word_fwht_fixed` (the dequant arms are reached through the existing `dequant_row_fixed` root) | `proof/extract/lib.rs` |
| Extracted body | `dequant.dequant_<t>_row`, `hadamard.fwht_fixed` | `proof/ApeironWord.lean` (generated) |
| Canonical definition | `dequantVals`; `sylvesterH k : Matrix (Fin 2^k) (Fin 2^k) ℤ`; `channelPair` | `proof/corpus/Numeric/DequantRefinement.lean`; new `Numeric/Sylvester.lean`; `Nous/Model/Decomposition.lean` |
| Refinement theorem | `dequant_<t>_row_spec`; `fwht_int_spec`; `fwht_fixed_refines` | new `Numeric/Dequant<T>.lean`, `Numeric/HadamardRefinement.lean` |

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| row law, per type | on the domain, `dequantVals out'.val = dequantVals out.val ++ <t>RowValues raw n` | proved | `dequant_<t>_row_spec` |
| dispatch | `dequant_row_fixed <id> = dequant_<t>_row` | proved | `dequant_row_fixed_<t>` |
| trit decode | `trit b n ∈ {0,1,2}`, and the 5 trits of `b < 243` are its base-3 digits, most significant first | proved | `trit_digits` |
| Sylvester | `H_{2^{k+1}} = [[H, H], [H, −H]]`, `H·H = 2^k·I`, `H_2 = channelPair 1` at `σ = 1` | proved | `sylvester_succ`, `sylvester_mul_self`, `sylvester_one_channelPair`, `sylvester_mulVec_butterfly` |
| integer FWHT | the `k` stages compute `H_{2^k}·(s ⊙ x)` exactly (butterfly-stage invariant = `H_{2^j} ⊗ I` on each segment) | proved | `fwht_int_spec` |
| normalized FWHT | `|x'_i − (H(s⊙x))_i/√2^k| ≤ 1/(2·2^32) + |(H(s⊙x))_i|·|c/2^32 − 2^{−k/2}|` | proved | `fwht_fixed_refines` |
| fork parity | the six arms reproduce `dequantize_row_*` on fork-layout byte vectors (float reference compared exactly after Q32.32 scaling) | tested | `src/engine/tests.rs` |
| refusals | short payload, non-finite `d`, bad block size, sign-length mismatch, overflow | tested + proved | tests; refusal equalities |

### Failure semantics

| Input class | Result | Condition |
|---|---|---|
| payload shorter than `⌈n/QK⌉` blocks | `Err(Quant(Truncated))` | first block that does not fit |
| `d` exponent all-ones | `Err(NonFiniteScale(bits))` | `f16_to_fixed` |
| `block` not a power of two, or `k > 16` | `Err(NotPowerOfTwo)` | before any write |
| `x.len() % block ≠ 0`, or `signs.len() ≠ x.len()` | `Err(LengthMismatch)` | before any write |
| sign entry ∉ {−1, +1} | `Err(BadSign)` | before any write |
| butterfly or normalization overflow | `Err(Overflow)` | checked op; `x` state is unspecified on `Err` (documented) |

### Evidence plan

1. **Tests:** `cargo test --workspace --all-features --locked`. Fork-layout vectors per type,
   including a partial last block, each refusal, and FWHT against a naive `O(n²)` Sylvester matmul
   for every `k ≤ 10`.
2. **Proof:** `proof/regen.sh` (lead lane) extracts the new bodies. The row laws and the FWHT
   refinement build under `cd proof/corpus && lake build`.
3. **Axiom cone:** every headline is `#audit`ed in `AxiomAudit.lean`.
4. **Negative controls:**
   - swap two trit stages;
   - flip a `Q2_0` bias (`code − 2`);
   - drop the sign multiply;
   - use `c` for `k + 1`.
   Each runs in a throwaway copy and must turn its gate red.
5. **Measurement:** quiet-host session only. Fork vs Apeiron token parity and throughput on a Bonsai
   GGUF.

### Trust boundary

rustc/LLVM codegen, the GGUF reader beyond its tested boundary, the fork's float kernels (a
comparison target only) and GPU paths are outside the evidence.

### Open questions

- Whether `signs` are stored per folded weight or per width (`sign_widths`). The fork keys them by
  width; the loader follows the fork.

## Plan — ternary and binary GGUF faces with an exact Walsh–Hadamard rotation

- **Owner:** lead (Claude track)
- **Intent:** `docs/epics/ternary-hadamard.md`
- **Spec:** `docs/epics/ternary-hadamard.md`
- **Source revision:** `74c037b`; working tree dirty
- **Status:** draft
- **Epic ID:** `APEIRON-TERNARY`

### Goal and completion condition

The epic closes when every exit-table row is green on one source/tool identity:

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
AENEAS_HOME=/home/socol/Workspace/apeiron/target/aeneas proof/regen.sh --check
cd proof && lake build
cd proof/corpus && lake build && lake build AxiomAudit
```

### Sequencing constraint

W1 (Rust) opens only after the model-algebra wave barrier
(`docs/evidence/model-algebra-wave-receipt.md`), so that barrier's `cargo test` and digests bind to
the pre-epic source. W2 changes `proof/ApeironWord.lean`, which rebinds every extraction digest. It
runs alone in the lead lane, after W1 is green.

W3-T5 (Sylvester algebra) has no extraction dependency and may run at any time.

### Non-goals

As in the intent: no GPU/SIMD kernels, no quantizer, no KV-cache 4-bit, and no performance claim
before the quiet-host session.

### Wave map

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | T1 dequant arms | `src/format/quant.rs`, `src/engine/dequant.rs`, `src/engine/tests.rs` | worker | `cargo test --workspace --all-features --locked engine::` | swap two trit stages → fork-vector test red |
| W1 | T2 FWHT kernel | `src/engine/hadamard.rs` (new), `src/engine/mod.rs` | worker | `cargo test --workspace --all-features --locked hadamard` | drop the sign multiply → naive-Sylvester test red |
| W2 | T1 extraction | `proof/extract/lib.rs`, `proof/ApeironWord.lean` (generated) | lead | `proof/regen.sh` then `proof/regen.sh --check` | none (identity gate); `regen.sh` audits `axiom`/`sorry` |
| W3 | T1 `Q1_0` row law | `proof/corpus/Numeric/DequantQ1_0.lean` | worker | `lake build Numeric.DequantQ1_0` | `bit ? −d : d` → red |
| W3 | T2 `Q2_0` + `PQ2_0` | `Numeric/DequantQ2_0.lean` | worker | `lake build Numeric.DequantQ2_0` | `code − 2` → red |
| W3 | T3 `TQ2_0` | `Numeric/DequantTQ2_0.lean` | worker | `lake build Numeric.DequantTQ2_0` | out index `32l + m` → `l + 4m` → red |
| W3 | T4 trit decode, `TQ1_0`, `PTQ1_0` | `Numeric/DequantTQ1_0.lean` | worker | `lake build Numeric.DequantTQ1_0` | swap stages `[32,16]` → red |
| W3 | T5 Sylvester algebra | `Numeric/Sylvester.lean` | worker | `lake build Numeric.Sylvester` | `H_2 = [[1,1],[1,1]]` → `sylvester_mul_self` red |
| W3 | T6 FWHT refinement | `Numeric/HadamardRefinement.lean` | worker | `lake build Numeric.HadamardRefinement` | `c` for `k+1` → `fwht_fixed_refines` red |
| W3 | T7 refusal equalities | `Numeric/DequantLowBitRefusals.lean` | worker | `lake build Numeric.DequantLowBitRefusals` | wrong variant → red |
| W4 | T1 loader metadata | `src/engine/loader.rs`, `src/engine/tests.rs` | worker | `cargo test … loader` | accept `sign_mode = "bogus"` → refusal test red |
| W4 | T2 folded matmul path | `src/engine/matvec.rs` or `step.rs` | worker | `cargo test … hadamard_fold` | skip the transform → parity test red |
| W4 | T3 rotation stage | `Nous/Model/RotationStage.lean` | worker | `lake build Nous.Model.RotationStage` | `L := 1` in the ℓ∞ norm (true constant `√n`) → red |
| W5 | T1 parity + throughput | none (measurement) | lead, quiet host | named command; machine; tokens/s and top-1 agreement vs fork | fork run without the transform must disagree |

### Task notes

- **W1-T1:** Compute trits in `u16` with `& 0xFF`, with no `wrapping_mul`, so the body stays in the
  Aeneas fragment. Reuse `read_scale`, `block_slice` and `push_prefix`. Build test vectors
  byte-for-byte from `ggml-common.h`.
- **W1-T2:** Checked `i64` butterflies; `i128` normalization through `value::div_round_nearest`.
  The constant `c` comes from a `const` table indexed by `k ≤ 16`, computed and asserted in a test
  against `2^32/√2^k` to half a unit in the last place.
- **W3-T5:** Define `sylvesterH : (k : ℕ) → Matrix (Fin (2^k)) (Fin (2^k)) ℤ` by the block
  recurrence. Prove:
  - entry formula `(−1)^{popcount(i &&& j)}`;
  - `H·H = 2^k·1`;
  - `Hᵀ = H`;
  - `H_1 = !![1,1;1,−1]` equals the matrix of `channelPair 1` at `σ = 1` on `RingCell ℤ`
    (`Nous/Model/Decomposition.lean`);
  - the 2-inverse obstruction: `2` is not a unit in ℤ, so `hyperbolic_decomposition` does not apply
    and the normalization is carried by `c`.
- **W4-T3:** In `Stage`'s ℓ∞ norm the normalized rotation has expansion `√n`, not `1`. State it
  exactly; the ℓ2 isometry is a separate lemma.

### Exit table

| Row | Grade | Gate |
|---|---|---|
| six arms tested against fork layouts | tested | `cargo test` |
| `fwht_fixed` tested against naive Sylvester | tested | `cargo test` |
| extraction fresh | proved (identity) | `regen.sh --check` |
| six row laws + dispatch | proved | W3-T1..T4 modules, `#audit` |
| Sylvester algebra + `channelPair` link | proved | `Numeric.Sylvester`, `#audit` |
| FWHT refinement | proved | `Numeric.HadamardRefinement`, `#audit` |
| refusals | proved + tested | W3-T7, tests |
| loader + folded path | tested | W4 tests |
| parity with the fork on a Bonsai GGUF | measured | W5, quiet host |

### Barrier checklist

At each wave barrier:
- run all §6 commands and the named controls;
- record the sha256 of every new or edited file and of `proof/ApeironWord.lean`/`AxiomAudit.lean`;
- write the wave receipt under `docs/evidence/`.

# Part II — exact one-bit faces and the conversion of existing models

Owner direction (2026-10-09): "go on the 1-bit epic w/ a focus on the formalisation and topology +
note ideally we can use existing models … IDEALLY it's perfectly exact". Part I loads trained
binary/ternary GGUFs; Part II converts trained higher-bit weights onto one-bit faces and states
exactly which readings of "one bit" are exact. Plan row: `LEARN-ONEBIT` ([plan](../plan.md)).

## Intent — one-bit faces

- **Source revision:** `7088d24`; Part I unchanged.
- **Status:** approved (owner go above).

### Problem

"One bit per weight" has four readings. Each one has a different exact statement:

| reading | exact statement | grade at this revision |
|---|---|---|
| storage | A lossless code is injective, so it needs ≥ H bits per symbol. `storage_pigeonhole`, `ternary_lossless_iff` are in `../sanguine/proof/Closures/Engineering/SeedStorage.lean` (`e830d3fb`, sha256 `0bca7cc2…`, lines 152 and 266). Measured int4-code entropy on Qwen2.5-0.5B is H ≈ 3.84, so a pretrained weight is not stored losslessly in one bit. | proved in sanguine (not vendored); H measured there |
| planes | A `b`-bit code is exactly `b` sign planes: `2q − (2^b − 1) = Σ_{k<b} 2^k s_k` with `s_k ∈ {±1}`. The matvec is `Σ_k 2^k (S_k x)`, with adds and subtracts only. Storage stays at `b` bits. | to prove (W6-T2) |
| time | First-order sigma-delta. `s_t = sign(w_t + e_t)` and `e_{t+1} = e_t + w_t − a·s_t` keep `|e_t| ≤ a` whenever `|w_t| ≤ a`. Every prefix sum of the one-bit stream is within one quantum of the prefix sum of `w`, independent of length. The rate moves into length. | to prove (W6-T3) |
| function | `storage_ne_compute` (same file, line 266) says the layer map may collapse, so exactness is owed only on the activation support. The bits needed are the rate of `x ↦ Wx` on that support at the required precision. That rate is measured, not assumed. | measured (W7) |

The executed bridge between "time" and "function" is the error identity of a shaped one-bit
quantizer. Write `r = w − a·s` for the residual of one row. If `r = U ε`, with `U` unit lower
triangular and `|ε_t| ≤ a`, then:

```
rᵀ x = εᵀ (Uᵀ x)          first order (U = difference operator, Abel summation):
Σ_t r_t x_t = e_n x_{n−1} − Σ_{t=1}^{n−1} e_t (x_t − x_{t−1}),   |·| ≤ a(|x_{n−1}| + TV_π(x))
```

The one-bit error therefore reaches the output only through `Uᵀx`:
- In first order, `Uᵀx` is the activation's variation along the input order `π`.
- With `U` the Cholesky factor of `H = E[xxᵀ]` (GPTQ's feedback, `src/learn/gptq.rs`
  `ul_cholesky`, `upper_tri_inverse`), `Uᵀx` is the activation's innovation in that order.
- Where the innovation vanishes the error vanishes exactly.

"Exact one bit" is exactness on the predictable part of the activation, plus a bounded remainder on
the innovation.

### Topology

Two choices are free and exact:
1. **The order of the hidden axis.** A permutation `P` of a hidden axis commutes with every
   elementwise nonlinearity and with RMSNorm (gain permuted alongside). Rewriting
   `W_prev ↦ P W_prev` and `W_next ↦ W_next Pᵀ` leaves the network function unchanged.
2. **The shaping filter `U`.**

The topology question for a single layer is the Hamiltonian path `π` over input channels that
minimises `E Σ_t |x_{π(t)} − x_{π(t−1)}|`. The Gram gives the squared form directly:
`E(x_i − x_j)² = H_ii + H_jj − 2H_ij`. For the residual stream, one `π` is shared by every layer
that reads it.

### The algebra

| face | unit group over ℤ | one-bit reading | grade |
|---|---|---|---|
| hyperbolic `σ = 1` | `{±1, ±j}` ≅ `ℤ/2 × ℤ/2` | In channel coordinates `channelPair 1 z = (z₁ + z₂, z₁ − z₂)`, a unit is an independent sign on each idempotent channel. The one-bit neuron is a sign per channel. `H_1 = channelPair 1`: `sylvester_one_channelPair` (`Numeric/Sylvester.lean`). | channel link proved; unit action to prove (W6-T1) |
| elliptic `σ = −1` | `{±1, ±i}` ≅ `ℤ/4` | A quarter-turn phase. There are no nontrivial idempotents (`idempotent_iff_trivial_of_disc_nonpos`, `../sanguine-apeiron/proof/Apeiron/Trichotomy/Idempotents.lean:307`), so the 2-bit phase does not split into two one-bit channels over ℤ. | to prove (W6-T1) |
| parabolic `σ = 0` | `±1 + bε` (infinite) | Sign plus a nilpotent first-order term: the straight-through derivative carried by dual numbers (Clifford 1873). | model |

### Why the alternatives were rejected

| alternative | reason |
|---|---|
| Claim lossless one-bit storage of pretrained weights | Refuted by `storage_pigeonhole` at the measured H ≈ 3.84 |
| Post-hoc one-trit GPTQ as the answer | Measured 48× the seal's output error (`../spinor-ttl/TERNARY_FINDINGS.md` §13, `a05bb1b`). It is the baseline arm, not the result. |
| Hadamard rotation, then sign | Whitening removes the neighbour correlation that `Uᵀx` exploits. It is kept as a measured arm, not assumed either way. |
| Train one-bit from scratch (BitNet, arXiv 2402.17764) | Not a conversion of an existing model. Non-goal here. |
| A new one-bit kernel before W7 | The rate is unmeasured. A kernel on an unmeasured rate is a tax. |

### Empirical best (§3a)

- **In-tree.** `src/learn/gptq.rs` (sha256 `f27ab5d7…`) ports fold `modelfold/src/quant.rs` `gptq`
  (`d497161f`, sha256 `40424687…`): error feedback under the calibration Gram, with a ternary grid
  and Q4_K/Q5_0/Q6_K grids. It is subsumed and is the base of every W7 arm.
- **External comparator.** llama.cpp `IQ1_S` (1.5625 b/w) and `IQ1_M` (1.75 b/w) with an imatrix,
  scored by paired KLD. Not subsumed; compared.

### Completion condition

1. The W6 Lean rows build and are audited in `AxiomAudit.lean`.
2. W7 reports, on the 24 held-out `ffn_down` Grams of Qwen2.5-0.5B, the relative output error of
   every arm. It runs on the build path (`src/learn/onebit.rs`) with its controls red/green.
3. W9 (planned after W7) reports a whole-model one-bit conversion (`Q1_0`, 1.125 b/w) by paired KLD
   against f16, beside `IQ1_S`/`IQ1_M` and `Q4_K_M`.

### Invariants

1. Every exact row stays exact. The sign-plane decomposition reproduces the code matrix bit for bit.
2. Calibration and evaluation Grams are disjoint text. A number is reported on held-out data only.
3. No inference receipt regresses; Part II touches no `src/gpu`, `src/simt*` or `src/spirv` file.

### Non-goals

- Training from scratch; GPU or SIMD one-bit kernels before W9.
- Any claim of one bit below the entropy of what is stored. The claim is about the function on
  the support.

### Trust boundary

- The Lean rows hold over a linear ordered field or ℤ.
- The W7 quantizers run in `f64` and are `tested`. The `sd1`/`sd1-path` sign stream runs in the checked-`i64`
  `sd_binary_block` (`src/learn/sigma_delta.rs`), extracted and refined; the `f64 → i64` scaling in `sd1_row`
  (`S = 24`) is `tested`, not proved.
- Gram capture (`/home/socol/oc-tmp/gptq-w2/capture_gram.py`) is a measurement instrument outside
  the proof.

## Specification — one-bit faces

### Lean (`proof/corpus/Numeric/`; audits added by the lead at install)

| task | file | statements |
|---|---|---|
| W6-T1 | `OneBitUnits.lean` | Over `RingCell ℤ` at `σ = 1`, multiplication by each of `1, −1, j, −j` acts on `channelPair 1` as the sign pair `(1,1), (−1,−1), (1,−1), (−1,1)`. The four units form `ℤ/2 × ℤ/2`. At `σ = −1`, multiplication by `i` sends `(a, b)` to `(−b, a)`, `i² = −1`, `i⁴ = 1`, so the units form `ℤ/4` with no element of order 2 except `−1`. |
| W6-T2 | `SignPlanes.lean` | For `q < 2^b`, `2q − (2^b − 1) = Σ_{k<b} 2^k (2·bit_k q − 1)`. The map `(Fin b → Bool) → ℤ` is a bijection onto the odd integers of absolute value ≤ `2^b − 1`. Matvec form: `(2Q − (2^b−1)) *ᵥ x = Σ_k 2^k (S_k *ᵥ x)` with `S_k` entries in `{±1}`. A `Q4_K`-shaped instance (`b = 4`) and a `Q8_0`-shaped one (`2q + 1`, `b = 8`) are checked by `decide` on a concrete block. |
| W6-T3 | `SigmaDelta.lean` | Over a `LinearOrderedField`, the first-order quantizer is defined as above. Statements: `sd_state_bound` (`|e_t| ≤ a`); `sd_prefix` (`Σ_{t<T} r_t = e_T`); `sd_abel` (the identity above, exact); `sd_dot_bound`; the shaping identity `r = U *ᵥ ε → r ⬝ᵥ x = ε ⬝ᵥ (Uᵀ *ᵥ x)`; and `shape_null_exact` (`(Uᵀx)_t = 0` wherever `ε_t ≠ 0` gives zero error). A ℚ instance shows plain sign error > sigma-delta error on one vector (non-vacuity). |
| W6-T4 | `HiddenPerm.lean` | `(W_next * Pᵀ) * f (P * W_prev x) = W_next * f (W_prev x)` for a permutation matrix `P` and coordinatewise `f`, with RMSNorm with a permuted gain. This is the exactness of the topology choice. |

The control for T3: in a copy, replace `sign(w_t + e_t)` with `sign(w_t)`. `sd_state_bound` must
fail to build, and a ℚ counterexample shows the state growing linearly. The controls for T1, T2 and
T4: flip one sign in a statement, and the build is red.

### Rust (`src/learn/onebit.rs`, `examples/onebit_diag.rs`, `tests/learn_onebit.rs`)

**Arms.** All arms are per row, with groups of 128 (the `Q1_0` block) and the sign scale
`a = mean|w|` over the group:

| arm | description |
|---|---|
| `rtn` | sign · `a` |
| `sd1` | first-order sigma-delta in natural order |
| `sd1-path` | `sd1` in the greedy nearest-neighbour order on `H_ii + H_jj − 2H_ij` |
| `gptq1` | GPTQ feedback with the grid `{−a, +a}` |
| `gptq1-act` | `gptq1` in act-order (descending `H_ii`) |
| `gptq1-path` | `gptq1` in the path order |
| `had-rtn` | 128-block normalised Walsh–Hadamard on columns of `W` and on `H`, then `rtn` |
| `had-gptq1` | the same rotation, then `gptq1` |

The references are the in-tree ternary `gptq_ternary` (levels 1) and the Q4_K round.

**Metrics.** Per tensor:
- `output_error(W, Ŵ, H_eval)` relative to `output_error(W, 0, H_eval)`, on the held-out Grams
  (`/home/socol/oc-tmp/gptq-diag/gram-eval/`). Calibration uses `/home/socol/oc-tmp/gptq-w2/gram/`
  (51,200 tokens).
- `max_t |ε_t|/a` (stability).
- Bits per weight including scales.

**Controls.**
- An exact `±a` matrix gives zero error in every one-bit arm.
- Zeroing the feedback in a copy raises `sd1` and `gptq1` error.
- Sign planes of a Q4_K code matrix reconstruct bit for bit.

**Model.** Qwen2.5-0.5B (`/home/socol/models/gguf/qwen05b-f16.gguf`). The `ffn_down` input width
is 4864 = 38 × 128.

## Plan — one-bit faces

| wave | task | files (owner) | gate | control |
|---|---|---|---|---|
| W6 | T1 units | `Numeric/OneBitUnits.lean` (DeepSeek) | `lake build Numeric.OneBitUnits` | flipped sign red |
| W6 | T2 planes | `Numeric/SignPlanes.lean` (DeepSeek) | `lake build Numeric.SignPlanes` | flipped plane sign red |
| W6 | T3 sigma-delta + shaping | `Numeric/SigmaDelta.lean` (Sonnet subagent) | `lake build Numeric.SigmaDelta` | no-feedback quantizer red |
| W6 | T4 hidden permutation | `Numeric/HiddenPerm.lean` (DeepSeek, after T1/T2 return) | `lake build Numeric.HiddenPerm` | unpermuted gain red |
| W7 | arms + measurement | `src/learn/onebit.rs`, `examples/onebit_diag.rs`, `tests/learn_onebit.rs` (DeepSeek) | `cargo test --release --locked --test learn_onebit`; `onebit_diag` table | the three W7 controls |
| W8 | refinement | extraction root `sd_binary_row` → `Numeric/SigmaDelta` | `proof/regen.sh --check`; `lake build` | mutated body red |
| W9 | whole model | `Q1_0` writer + paired KLD vs f16, `IQ1_S`, `IQ1_M`, `Q4_K_M` | `llama-perplexity` paired KLD | identical-weights KLD = 0 |

W6 tasks are file-disjoint. The lead adds their `#audit` lines to `AxiomAudit.lean` at install, so
`AxiomAudit.lean` has one owner. W7 runs parallel to W6. W8 and W9 open after W7's table.

### Exit table (Part II)

| row | grade | gate |
|---|---|---|
| unit actions per face | proved | W6-T1 |
| sign-plane exactness | proved | W6-T2 |
| sigma-delta bound, prefix, Abel, shaping identity | proved | W6-T3 |
| hidden-axis permutation exactness | proved | W6-T4 |
| one-bit arms on held-out Grams with controls | tested + measured | W7 |
| executed `sd_binary_block` refines `Numeric.SigmaDelta` | proved | W8 |
| whole-model one-bit KLD against `IQ1_S`/`IQ1_M` | measured | W9 |

### W7 result (row `onebit-exact-one-bit-and-FubKIC3prjKrgnryWWxrF1`)

Command: `cargo run --release --example onebit_diag -- … --index-range 0,4` (509 s; log
`/home/socol/oc-tmp/escalation/ONEBIT-W7-diag-final.log`). Metric: `output_error(W, Ŵ, H_eval)` on held-out Grams,
Qwen2.5-0.5B `ffn_down`, blocks 0–3 of 24, every arm at 1.125 b/w. Grade `tested` (f64; one run, deterministic).

| arm | blk0 | blk1 | blk2 | blk3 | sum |
|---|---|---|---|---|---|
| rtn | 0.360 | 0.358 | 1.50 | 1.34 | 3.560 |
| sd1 | 0.528 | 0.584 | 1.64 | 1.24 | 3.992 |
| sd1-path | 0.478 | 0.560 | 1.40 | 0.900 | 3.341 |
| gptq1 | 0.296 | 0.375 | 1.05 | 0.706 | 2.427 |
| gptq1-act | 0.272 | 0.388 | 0.395 | 0.501 | 1.556 |
| gptq1-path | 0.270 | 0.357 | 1.00 | 0.520 | 2.152 |
| had-rtn | 0.234 | 0.316 | 0.300 | 0.124 | 0.974 |
| had-gptq1 | 0.199 | 0.350 | 0.00546 | 0.00583 | 0.560 |
| ref-gptq3 (levels = 1) | 0.470 | 0.712 | 0.00959 | 0.0173 | 1.209 |

- `had-gptq1` is lowest on 3 of 4 blocks (blk1: `had-rtn` 0.316 < 0.350); sum 0.560 against `rtn` 3.560 (6.36×).
- Plain `sd1` is worse than `rtn` on 3 of 4 blocks. `max|e|/a` is 26–66 on the sd arms: the weights violate the
  `|w| ≤ a` premise of the proved state bound (`a = mean|w|`, heavy tails), so the bound does not apply and the
  feedback accumulates.
- Controls: exact `±a` rows give 0 error in the six sign arms and < 1e-9 through the Hadamard arms; zeroed feedback
  is strictly worse than `sd1`/`gptq1`; the mutation (sd1 feedback zeroed, throwaway copy) turns the test red.
- Open: blocks 4–23 (the exit row asks for all 24); W9 whole-model KLD.

### W8 result (row `onebit-exact-one-bit-and-FubKIC3prjKrgnryWWxrF1`)

- Executed body: `src/learn/sigma_delta.rs` `sd_binary_block`, checked `i64`. Shape mismatch or overflow returns
  `None`; `sd1_row` panics on `None`. It is extracted by `#[path]` through `proof/extract/lib.rs` `word_sd_binary_block`.
- `proof/corpus/Numeric/SigmaDeltaRefinement.lean` (proved, cone `{propext, Classical.choice, Quot.sound}`):
  - `sd_step_spec`
  - `sd_loop_spec`
  - `sd_binary_block_shape`
  - `sd_binary_block_spec`: on `some (e, peak)`, the state is `sdState`, the peak is `sdPeak`, and every sign is
    `sdSign`, read through ℚ.
  - `sd_binary_block_witness`: success, overflow and shape branches on the extraction.
- Control: extracted `u >= 0` → `u > 0` in a throwaway copy turns the build red at lines 92, 137 and 395.
- `sd1_row` scales by `2^24`. The `f16` weights are exact at that scale; the block scale is `round(a·2^24)`.
  - `onebit_diag --index-range 0,4` (524 s), `sd1`/`sd1-path` against W7: max relative change 1.6e-5 (blk0
    `sd1-path` 0.4778163 → 0.4778086); 9 of 24 cells above 1e-6.
  - Every arm ranking above is unchanged. The cause is attributed to the rounded scale; this is inference, not tested.

### W9 result (row `onebit-exact-one-bit-and-FubKIC3prjKrgnryWWxrF1`)

Receipt `docs/evidence/onebit-w9.md`; Lean `proof/corpus/Perf/OnebitW9.lean` (8 verdicts, empty axiom set).
Qwen2.5-0.5B, wikitext-2 test 64×512, paired KLD against f16 on llama.cpp CPU. llama.cpp `IQ1_S`/`IQ1_M` reach only
the 21 `ffn_down` tensors of blocks 3–23 (P1); P2 is all 168 linear tensors on llama.cpp's `Q1_0` base.

| arm | b/w | P1 KLD | P2 KLD |
|---|---:|---:|---:|
| `IQ1_M` imatrix | 1.75 | 0.519 | — |
| `IQ1_S` imatrix | 1.5625 | 0.632 | — |
| `had-gptq1` | 1.125 + online R₁₂₈ | 0.834 | 6.850 |
| `gptq1` | 1.125 | 5.243 | 15.300 |
| `rtn` = llama.cpp `Q1_0` | 1.125 | 10.385 | 13.167 |

- `had-gptq1` beats `Q1_0` 12.4× at equal bits on P1. `IQ1_S` beats it 1.32× with 1.39× the bits.
- Post-training one bit on every linear tensor is not usable (P2 best 6.85 nats, 282× `Q4_K_M`). Whole-model one-bit
  weights go through training: `LEARN-HATTN` T4 and the fold training path.
- Controls: f16 against itself KLD 0; our `Q1_0` writer is byte-identical to llama-quantize's on P1.

# Part III — one-bit attention on the hyperbolic face (`LEARN-HATTN`)

Owner go (2026-10-09): open the fold-training and hyperbolic-attention lane, aiming at attention
that costs O(1) or O(log n) in context length, or is otherwise cheaper than softmax, for
training and finetuning. It must stay compatible with existing models.

## Intent — one-bit attention

### Already proved in this tree (`proof/corpus/Nous/Model/HyperbolicAttention.lean`)

| statement | declaration | grade |
|---|---|---|
| The unnormalized softmax weight `e^s` is the upper idempotent channel of the hyperbolic flow `flow 1 s = (cosh s, sinh s)`, and `e^{-s}` is the lower channel | `softmax_weight_is_upper_channel`, `channelWeight_flow_one` | proved |
| The Qwen score is the elliptic pairing of RoPE-rotated pairs and depends on positions only through `t − s` | `qwenScore_eq_rel`, `qwenScore_relative` | proved |
| Online softmax is an exact streaming scan with a chunk-fusion law | `softmax_attention_is_online_spinor_vec`, `oaNumV_scan_append` | proved |
| `e^{qk}` over real `q, k` has no factorization `Σ_{j<D} φ_j(q) ψ_j(k)` at any `D`, so softmax attention over real scores has no exact fixed-size carry | `exp_kernel_not_finite_rank` | proved |
| A finite-rank content kernel times the hyperbolic boost `e^{λ(t−s)}` is exactly a fixed-size carry | `boost_finite_rank_recurrence_vec` | proved |

The face picture this gives:
- Softmax attention is an elliptic score (the positive-definite dot product, rotated by RoPE)
  fed into a hyperbolic exponent (the `e₊` lane of `exp(jθ)`).
- The correspondence between softmax and the hyperbolic face is exact; it is the first row.
- The negative row is a statement about the function, not the face that computes it. Equal
  denotations keep the same lower bound, so moving faces does not buy exact constant state over
  real scores.
- The fine-grained bound points the same way. Alman and Song, *Fast Attention Requires Bounded
  Entries* (arXiv 2302.13214), and its gradient sequel (arXiv 2402.04497) rule out a truly
  subquadratic approximation once entries reach `Θ(√log n)`, assuming SETH. These are pointers:
  not yet retrieved or read here (§9.3).

### The opening: one-bit keys and queries

Take `q, k ∈ {±1}^d` and a scale `s`. Each product `x_i = q_i k_i` is `±1`, so `x_i² = 1`, and
the hyperbolic flow evaluated at that character is

    e^{s x_i} = cosh s + x_i sinh s.

Multiplying the `d` factors gives, with `x_S = Π_{i∈S} x_i`:

    e^{s q·k} = Π_i (cosh s + q_i k_i sinh s) = Σ_{S ⊆ [d]} cosh^{d−|S|} s · sinh^{|S|} s · q_S k_S.

- R[x]/(x²−1) is the group algebra R[Z₂]. Its d-fold tensor power is R[Z₂^d], and the tensored
  idempotent split `Π (1 ± x_i)/2` is the Walsh–Hadamard transform: the transform Part II
  already uses.
- So the one-bit exponential kernel has exact finite rank 2^d. The features are `q_S`, weighted by
  `λ_S = cosh^{d−|S|} s · sinh^{|S|} s`.
- `boost_finite_rank_recurrence_vec` applies unchanged. Exact attention is a carry of
  `2^d·(d_v + 1)` numbers, independent of the context length `n`.
- Lead check, numpy f64, d = 8, n = 1000, s = 0.37: carry readout vs direct softmax, max
  difference 1.0e-15. This is a sample, not a gate.

The cost of exactness:
- The kernel has full rank 2^d (every `λ_S > 0` for `s > 0`).
- An exact query costs `min(n, 2^d)·(d_v + 1)` multiply-adds. It beats the scan only when
  `n > 2^d`, which means the codes collide. This is the rate rule of `AGENTS.md` §10, met here
  by a finite key domain.

Truncation by degree:
- Write `t = tanh s`, so `e^{s q·k} = cosh^d s · Σ_S t^{|S|} q_S k_S`.
- Keeping `|S| ≤ r` leaves an error of at most `cosh^d s · Σ_{j>r} C(d,j) t^j`, with
  `C(d, ≤r)` features.
- Small `s` is the bounded-entry regime of the pointer above, now with an explicit tail.

### Compatibility with existing models

Existing checkpoints carry real `q, k`.
- Exactness holds after conversion to `b`-bit signs, not to the float model. Conversion changes
  the function.
- The open question is empirical: at which `(b, r)` the converted attention distribution stays
  within tolerance of the float one, and whether `2^b` (exact) or `C(b, ≤r)` (truncated) falls
  below `n` at useful lengths.
- RoPE does not commute with the sign map. Rotating before taking signs keeps the score
  relative only approximately; this is measured, not assumed.

### Success condition

1. **Distribution.** On Qwen2.5-0.5B, some `(b, r)` gives a median per-head attention KL against
   the float softmax of at most 0.05 nats, with feature count `F < n` at `n = 32768`.
2. **Whole model.** The converted model (one-bit q, k projections, fold-trained) has a
   whole-model KLD against f16 no worse than `Q4_K_M`'s.
3. **Training.** Training and finetuning attention through the carry costs linear time in `n`.

The thresholds are the lead's and the owner may revise them.

### Non-goals and non-claims

- No exact sub-quadratic softmax over real scores: proved impossible above.
- No novelty is asserted. Hashing and binary attention, Taylor-feature linear attention (Based,
  arXiv 2402.18668), random features (Performer) and the polynomial method are prior art to
  retrieve and read in `HATTN-T1` before any comparison is written.

## Specification — one-bit attention

- **Domain.** `s : ℝ`; `q k : Fin d → ℝ` with `q_i² = 1` and `k_i² = 1`; values `Fin d_v → ℝ`.
  Total over that domain. Real arithmetic in Lean. The Rust carry is `f64` and tested; an exact
  integer carry over Hamming bins is W2's question.
- **Lean (new module `proof/corpus/Nous/Model/OneBitAttention.lean`).** Statements to prove:
  - `onebit_factor_is_flow`: `cosh s + x sinh s` is `ringUpper 1 (flow 1 s)` at `x = 1` and
    `ringLower 1 (flow 1 s)` at `x = −1`.
  - `onebit_exp_product`: `exp (s * Σ q_i k_i) = Π (cosh s + q_i k_i sinh s)`.
  - `onebit_exp_expansion`: the `Finset.powerset` sum above.
  - `onebit_attention_is_finite_carry`: normalized attention over a list of one-bit keys equals
    the readout of the `2^d`-feature carry, via `boost_finite_rank_recurrence_vec` at `λ = 0`.
  - `onebit_truncation_bound`: the degree-`r` tail bound.
  - A joint witness at `d = 2`.
  - Audit cone `{propext, Classical.choice, Quot.sound}`.
- **Cost model.**
  - Exact carry: `2^d·(d_v + 1)` words; per query `2^d·(d_v + 1)` multiply-adds.
  - Truncated: `C(d, ≤r)` in place of `2^d`.
  - Standard scan: `n·(d + d_v)`.
  - These are combinatorics, not measurements.

## Plan — one-bit attention

| task | scope | files | gate | control |
|---|---|---|---|---|
| `HATTN-T1` (after T2, owner 2026-10-09; cheapest falsifier of compatibility) | **Prior art.** Retrieve and read the pointers above, plus hashing/binary attention, and record each one's relation to the expansion. **Measurement.** On the Qwen2.5-0.5B `q`,`k` dumps (`/home/socol/oc-tmp/faces-qk`, 24 layers, 14 query heads, 2 KV heads, `T = 512`, f32, `cs.bin` RoPE table), RoPE-rotate, apply a fixed normalized Walsh–Hadamard rotation, sign to `b` bits. Report per head the attention KL against float softmax at the best scale `s`, for exact `b ∈ {8, 12, 16}` and truncated `b = 64`, `r ∈ {1, 2, 3}`. Also report distinct-code growth in `t`. The prediction is registered before the run | `tools/hattn/qk_onebit.py`, `docs/evidence/hattn-qk-onebit.md` | `python3 tools/hattn/qk_onebit.py --check` | replace the sign map by the identity in a copy: KL is 0 at every head |
| `HATTN-T2` (first, owner 2026-10-09; closed) | The Lean module above and the cost model | `proof/corpus/Nous/Model/OneBitAttention.lean`, `proof/corpus/AxiomAudit/Remaining.lean` | `cd proof/corpus && lake build Nous.Model.OneBitAttention AxiomAudit` | drop the `sinh` term from the factor in a copy: the build fails at the product theorem |
| `HATTN-T3` (after T1) | Rust exact and truncated carry; tested against float softmax on ±1 inputs; then the fold-trained sign projection of one Qwen layer | `src/learn/hattn.rs`, `src/learn/mod.rs`, `tests/learn_hattn.rs` | `cargo test --release --locked --test learn_hattn` | flip one feature sign in a copy: the carry-vs-scan test fails |
| `HATTN-T4` (after T3) | Whole-model conversion and paired KLD against f16 and `Q4_K_M`; training cost against softmax on the spinor-ttl race corpus | specified at T3's barrier | specified at T3's barrier | specified at T3's barrier |

## Verification — one-bit attention, `HATTN-T2`

Gate: `cd proof/corpus && lake build Nous.Model.OneBitAttention AxiomAudit`, exit 0 (9136 jobs, 13.6 s
warm). `AxiomAudit/Remaining.lean` audits 17 declarations of `Nous.Model.OneBit`; 16 depend on
`{propext, Classical.choice, Quot.sound}`, `carry_query_le_scan` on `{propext, Quot.sound}`.

| statement | content | grade |
|---|---|---|
| `exp_mul_sign` | `e^{s x} = cosh s + x sinh s` on `x ∈ {±1}` | proved |
| `onebit_factor_is_flow` | the factor is `ringUpper 1 (flow 1 s)` at `x = 1`, `ringLower 1 (flow 1 s)` at `x = −1` | proved |
| `onebit_exp_product` | `e^{s q·k} = Π_i (cosh s + q_i k_i sinh s)` on one-bit `q, k` | proved |
| `prod_flow_expansion` | `Π_i (cosh s + x_i sinh s) = Σ_S λ_S x_S`, `λ_S = cosh^{d−|S|} s sinh^{|S|} s`, any real `x` | proved |
| `onebit_exp_expansion` | `e^{s q·k} = Σ_S λ_S q_S k_S` | proved |
| `onebit_attention_is_finite_carry` | normalized attention with per-key scales `a_n` over `N` one-bit keys = read of the carry `(carryNum, carryDen)` indexed by `Finset (Fin d)` | proved |
| `carry_card` | the carry index has `2^d` entries | proved |
| `carryNum_snoc`, `carryDen_snoc` | appending a key adds its own features; the carry is additive in keys | proved |
| `onebit_truncation_bound` | keeping `|S| ≤ r` errs by at most `Σ_{|S|>r} cosh^{d−|S|} s · |sinh s|^{|S|}` | proved |
| `truncated_card` | the truncated carry keeps `Σ_{j≤r} C(d, j)` subsets | proved |
| `carry_query_le_scan` | per-query read `2^d (d_v+1) ≤ n (d+d_v)` when `2^d ≤ n`, `d ≥ 1` | proved |
| `scanCausal_closed` | causal scan pass `2·Σ_{t<n} (t+1)(d+d_v) = n(n+1)(d+d_v)` | proved |
| `carry_causal_le_scan` | causal carry pass `n·2·2^d(d_v+1) ≤` causal scan pass when `4·2^d ≤ n+1`, `d ≥ 1` | proved |
| `onebit_witness` | joint sign witness at `d = 2`; at `d = 8`, `d_v = 64`: carry 16640 words, below a 32768-token read and a 32768-token causal pass | proved |

- The carry theorem is proved by exchanging the two finite sums directly. It does not go through
  `boost_finite_rank_recurrence_vec`, which the spec named.
- The cost model counts real multiply-adds at unit cost (ℕ). It is not a hardware measurement.
- Controls (`lake env lean` on copies in `/home/socol/oc-tmp/hattn-t2-ctl/`; live tree untouched):
  - `drop_sinh.lean` removes `+ x sinh s` from `exp_mul_sign`. Exit 1: the rewrites at lines 64–65
    fail and the theorem stays unsolved.
  - `weak_cost.lean` weakens `carry_causal_le_scan` to `2·2^d ≤ n+1`. Exit 1 at line 264. The
    weakened statement is false, for example at `d = 1`, `d_v = 64`, `n = 3`.
  - `restore.lean`: exit 0.
- Not covered: floating-point evaluation; any map from a real-valued model to signs (T1); the Rust
  carry (T3); hardware cost.
