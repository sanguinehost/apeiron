# ternary and binary GGUF faces with an exact Walsh–Hadamard rotation

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
- Training or quantizing models (`quantize_ptq1_0`); load and inference only.
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
