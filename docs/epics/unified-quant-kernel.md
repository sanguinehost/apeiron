# Unified carrier and quant kernel: quantisation level as one parameter

Merged epic record: intent, spec and plan in one file (see `docs/README.md`). Supersedes the CPU-only `unified-quant-kernel` record, retained verbatim below as wave W2c.

## Intent — Unified carrier

- **Owner:** Lucas 'Paperboy' Rose-Winters (lead: Apeiron lead)
- **Source revision:** `20a307d`; working tree dirty only in files unrelated to this epic
- **Status:** approved by the owner (direction: "single, canonical, composable sources … quantisation is a dial")
- **Spec:** this file
- **Plan:** this file

### Problem

Quantisation level is reimplemented per carrier. `Carrier` (`src/simt/mod.rs:507`) is one enum with one
affine decode `w = d·q + m` per 32-block, yet the executing bodies branch on it in 11+ files
(`src/engine/{mixer,attn,kern/gemm,chart/ffn}`, `src/simt/{gemv,dp4a,superblock}`, `src/gpu/decode/{pack,state,prefill}`,
`src/simt_nn/{spec,attn_dp4a}`), and `src/spirv/` carries three GEMM sources
(`coopmat_gemm.comp`, `int8_coopmat_q8.comp`, `q8_tile.comp`). `tests/gpu_prefill_match.rs` has Q4 and Q8 arms only.
A new level costs a new kernel, a new test arm and a new proof obligation.

The parameter already exists in two places. `Dispatch.dpw` (`proof/corpus/Spirv/Dp4a.lean`, "`OpSUDot` phases per
weight word, 1 for Q8, 2 for Q4") parameterises the served DP4A GEMV module. `sdot_split_ringMul`
(`proof/corpus/Spirv/Semantics.lean:592`) proves the two-phase Q4 split realises channel 0 of canonical `ringMul`.
The epic makes that parameter the only difference between integer carriers and removes the per-carrier copies.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| keep one kernel per carrier, share only tests | the 33.6k lines across 40+ carrier-touching files stay; each level is a new proof |
| a runtime `if bits == 4` inside each kernel | a data-dependent branch in the inner loop; not a statement a theorem indexes by `b` |
| adopt llama.cpp's per-type `mul_mat_*` template set | the multiplication of implementations this epic removes; no refinement to `ringMul` |
| unify F16/F32 and integer carriers in one kernel now | they use different instruction paths (FMA / fp16 coopmat vs `OpSUDot` / int8 MMA); decided by measurement in W4, not assumed |

### Completion condition

One GEMM source, one GEMV source and one attention-weight-read source, each parameterised by bit width `b`, with
a Lean decode lemma indexed by `b`; every per-carrier copy deleted; every measured cell at or above its pre-epic
value.

```sh
cargo test --workspace --all-features --locked
cargo test --release --locked --features gpu --test gpu_prefill_match -- --ignored --test-threads=1
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
proof/regen.sh --check
cd proof/corpus && lake build
python3 tools/rooting/retire_check.py --check
```

### Invariants

1. The decode is one expression `w = d·q + m` with `q = (word >> b·i) & (2^b − 1)`; `b` enters only there and in the phase count.
2. Every executing face refines canonical `ringMul`/`ringNormForm`; `Sigma` is a value in every executing signature.
3. No measured cell regresses: pp512 and PPL per carrier at or above the pre-epic mean, outside the observed spread (§4.10 averaging, ≥5 interleaved runs).
4. Refusal stays refusal: unsupported `b` is a typed error, not a silent fallback to another width.
5. The DP4A path remains the floor; coopmat remains optional and runtime-selected.

### Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| the integer carrier dot at width `b ∈ {2,4,8}` refines canonical `ringMul` by one lemma indexed by `b` | proved | W1 theorem (extends `sdot_split_ringMul`); `lake build` |
| a new width (`b = 2`) is added by instantiating `b` only: no new kernel, no new theorem body | tested + proved | W3 gate: Q2 matches the CPU oracle, same lemma instance |
| one GEMM source at `b ∈ {4,8}` matches each deleted kernel's pp512 within spread | measured | `examples/gpu_ppl.rs`, `engine_profile`, ≥5 interleaved runs, RTX 5080 |
| the prefill attention gain (INF-ATTN-PREFILL) is independent of the weight carrier | measured | INF-CARRIERS table (K and V are f32: inferred from the kernel, not yet measured) |
| F16/F32 weights fold into the same fold at no measured cost, or the exception is named | measured | W4 table |
| lines of per-carrier code and number of carrier-specific sources go down | measured | W5 `rg`/`wc` inventory before and after |

### Non-goals

- New quantisation formats (K-quant superblock layouts beyond what `src/simt/superblock.rs` already carries; open question below).
- Speed targets beyond not regressing; the 2.0× llama.cpp bar stays the programme's, not this epic's.
- Model-architecture changes, KV-cache quantisation, activation quantisation formats.
- Any claim about rustc/LLVM, the Vulkan driver or GPU hardware.

### Trust boundary

Evidence reaches the extracted Rust bodies and the SPIR-V semantics modelled in `proof/corpus/Spirv/`. It does not
reach the Vulkan driver's compilation of the module, the GPU, rustc/LLVM codegen, or the quantisation error of any
particular model (PPL is `tested`, not proved).

### Open questions

- GGUF `Q4_K_M` is a superblock format; does `superblock.rs` reduce to the same `b` with a second scale level, or is the scale hierarchy a second dial? Lead decides after the W0 inventory.
- Can F16/F32 be expressed as `b = 16/32` of the same fold without a second kernel, or is a second instruction path a permanent exception? W4 measures; the owner sees the verdict.

## Specification — Unified carrier

### Scope and admissible domain

| Item | Value |
|---|---|
| Operation | `carrier_dot(b, d, m, word, x)`: dot of `32/b` codes per word against int8 activations, summed over `8/b` `OpSUDot` phases |
| Domain | `b ∈ {1,2,4,8}` (divides 8); 32-block affine `(d, m)`; word-aligned rows; activations int8 |
| Totality | `Result`: `b ∉ {1,2,4,8}` refuses with a typed error naming `b` |

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| decode | `∀ b i, decode b i word = (word >> b·i) & (2^b − 1)` | proved | W1 |
| split | `∑_{phase<8/b} sudot(phase) = ∑ ringMul σ cells .1` at the realised σ | proved | W1 generalising `sdot_split_ringMul` |
| refinement | extracted `b`-parameterised body = canonical dot | proved | W2 `proof/regen.sh` + corpus |
| dispatch | `Dispatch.dpw = 8/b` and the guarded access chains stay in bounds for every `b` | proved | W2 generalising `emittedAccessChains` |
| parity | pp512 and PPL at `b ∈ {4,8}` ≥ pre-epic mean | measured | W3 |

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| unsupported `b` | `Err(UnsupportedWidth(b))` | `b ∉ {1,2,4,8}` |
| row not word-aligned | `Err` | existing pack check |
| coopmat unavailable | DP4A floor, same `b` | runtime selector (existing) |

### Evidence plan

1. `cargo test` plus `gpu_prefill_match` arms for every supported `b` and for F16/F32 where the harness can run them.
2. `proof/regen.sh --check`; `cd proof/corpus && lake build`; axiom cone `{propext, Classical.choice, Quot.sound}` in `AxiomAudit.lean`.
3. Negative control per task: break the shift or mask in the one decode; the `b`-indexed theorem and `gpu_prefill_match` go red.
4. Measurement: paired interleaved old/new arms, ≥5 runs, mean and spread, quiet-host protocol, shared cargo shim.

### Trust boundary

As in the intent. Spec constants (`OpSpecConstant`) are resolved by the driver at pipeline creation; the epic proves the module's semantics, not the driver's specialisation.

### Open questions

- Spec-constant specialisation vs one emitted module per `b` from one generator (`src/spirv/check.rs` already validates emitted modules): W2 picks by what the SPIR-V checker and the semantics model can already express.

## Plan — Unified carrier

### Goal and completion condition

See the intent's completion command. A wave closes when every task gate is green with its negative control run.

### Wave map

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W0 | INF-CARRIERS: measure the prefill gains per carrier and model (running); inventory every per-carrier site | `docs/evidence/unified-carrier-inventory.md` | Sonnet lane | `rg`-based inventory script exit 0; table complete | a site removed from the list makes the count check red |
| W1 | `b`-indexed decode and split lemma over `Spirv/Semantics.lean` | `proof/corpus/Spirv/Carrier.lean`, `proof/corpus/AxiomAudit.lean` | proof lane | `cd proof/corpus && lake build` | change the mask width: lemma red |
| W2 | one `b`-parameterised GEMV (DP4A) body; `Dispatch.dpw = 8/b` proved | `src/simt/dp4a.rs`, `proof/extract/lib.rs`, `proof/corpus/Spirv/Dp4a.lean` | proof lane | `proof/regen.sh --check`; corpus build; `cargo test --lib simt` | `dpw` off by one: access-chain bound red |
| W2c | CPU chart face: the four `q8act_*_row_fast` kernels collapse into one decoder-parameterised accumulator (task `APEIRON-QUANT-UNIFY-W1-T1`, dispatched; record below) | `src/engine/chart/{mod,x86,portable}.rs` | worker lane | `cargo test --locked --lib engine::chart` byte-identical | one codec scale or min mutated: parity red |
| W3 | one `b`-parameterised prefill GEMM replacing `q8_tile.comp`, `int8_coopmat_q8.comp`, `coopmat_gemm.comp` Q4/Q8 arms; Q2 instance as the thesis test | `src/spirv/coopmat_gemm.comp`, `src/gpu/decode/prefill.rs`, `tests/gpu_prefill_match.rs` | worker lane | `gpu_prefill_match` all arms; `gpu_ppl` ×3; paired pp512 | shift/mask broken: first-token mismatch |
| W4 | F16/F32 as `b = 16/32` or a named exception, by measurement | `src/simt/gemv.rs`, `src/gpu/decode/pack.rs` | worker lane | paired table, all cells | wrong unpack: PPL red |
| W5 | delete per-carrier copies; per-carrier line and source counts before and after | per `retire_check` | lead | `python3 tools/rooting/retire_check.py --check` | removal with a live consumer: refused |

W2 and W3 touch disjoint files. W1 gates both. W0 gates the plan rank of W3 and W4.

### Barrier checklist

- [ ] Every task gate exit 0 with its control run and restored.
- [ ] `cargo test`, `cargo clippy`, `cargo fmt`; `proof/regen.sh --check`; both `lake build`s.
- [ ] Paired tables for every previously measured cell; no regression outside spread.
- [ ] Source and tool digests recorded; trust-boundary statement present.

### Stop and rollback conditions

- A measured cell regresses outside spread after the unification: revert that task, report the cell and the mechanism, keep the carrier-specific path for that cell as a named exception.
- Three materially different failed approaches on one proof: stop and report the smallest unresolved goal.

---

## W2c record: unified quant projection kernel (CPU chart face)
Owner: Apeiron lead. Source base: `4b5eff5`. Status: dispatched (byte-identical unification).

### Intent

The executing transformer must not be reimplemented once per quantization format. The slow
`matvec_fixed` layer already keeps one kernel across every kind (`dequant_row_fixed` block codec +
`row_dot_fixed`), but the *fast* prefill layer forks into one dot per family in
`src/engine/chart/mod.rs`:

```
520  fn q8act_q8_0_row_fast(...)
604  fn q8act_q5_0_row_fast(...)
722  fn q8act_q4_k_row_fast(...)
803  fn q8act_q6_k_row_fast(...)
```

Each owns its own `#[target_feature(enable = "avx2")]` accumulator loop and its own portable twin.
Waves 5–6 tuned only the `Q8_0` arm; the dense Qwen3-8B-Q4_K_M prefill measured `ffn` at
~113 GMAC/s versus ~1000 GMAC/s for the Q8_0 hybrid. The two are the same mathematical object;
only the block decode differs. The goal is one accumulator kernel parameterized by a block decoder,
so a new quantization format costs a decoder, not a dot and not a prefill path.

Non-goals: no change to any output value, refusal, fold order or evidence grade. No speed claim is
made by this epic; a measured Q4_K improvement is a separate optimization claim gated by
`tools/dispatch/admit.py --launch`.

### Specification

| boundary | contract |
|---|---|
| Arithmetic | one accumulation kernel computes `acc += act_scale · (w_scale · dot − w_min · xsum)` per 32-lane group, in the existing per-group order; `w_min = 0` for the offset-free families |
| Decode | per-family block decoder yields groups; AVX2 lane primitives (`x86::dot32_ptr`, `q5_0_dot32_ptr`, `q4_k_block_dots`, `q4_k_window_dot_ptr`, `q6_k_half_dots`) are retained, not restated |
| Value identity | every output byte-identical to the current fast rows and to `matvec_fixed`; the existing family parity tests are the oracle |
| Fallback | non-finite weight scale, non-fused family, bound failure and non-multiple widths keep routing to `matvec_fixed`; the refusal/fallback contract is unchanged |
| Extraction | `chart` is outside the extracted fragment (`src/engine/dequant.rs` and `src/engine/matvec.rs` are the extracted roots); `dequant.rs` must not change |
| Complexity | `O(n_in)` time, `O(1)` extra space per row, unchanged |

### Plan

One writer per file. The task touches the three `chart` files, all disjoint from the GPU wave.

| task | owner / files | gate / control |
|---|---|---|
| APEIRON-QUANT-UNIFY-W1-T1 | worker `gemini-flash`; `src/engine/chart/mod.rs`, `src/engine/chart/x86.rs`, `src/engine/chart/portable.rs` | gate `cargo test --locked --lib engine::chart` byte-identical for every family; control mutates one codec scale/min in a throwaway copy and requires a parity failure, then restores green; `cargo test --locked --lib engine::kern engine::step` also green |

Arc Block:

```text
SCOPE:      chart/mod.rs, chart/x86.rs, chart/portable.rs; collapse the four q8act_*_row_fast
            kernels into one decoder-parameterized accumulator. Completion: family parity green.
REGISTER:   structural unification/refactor; no numerical, contract or performance claim.
CORPUS:     chart/ (outside extraction), dequant.rs, matvec.rs read; no canonical theorem changed.
REFERENCE:  /home/socol/Workspace/apeiron/src/engine/chart/mod.rs.
GATE:       cargo test --locked --lib engine::chart; observed red is the seeded codec mutation.
GRADE:      tested on byte identity; any speedup is unregistered and admission-gated.
```

### Open

- Q4_K prefill attribution: whether the dense 8B is falling back to `matvec_fixed` or taking an
  untuned fast arm is unmeasured. Separate bounded task; the collapse does not assume it.
- `engine_profile`'s `--llama` path passes no device pin, so a GPU-visible llama-bench silently
  compares Apeiron CPU against llama.cpp GPU. Harness fix, separate task.
