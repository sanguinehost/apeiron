# nq: fused gate/up SwiGLU and retained attention scratch (1.061× untraced, llama parity)

Criterion: **T2** (turn latency), partial. The subject is the cold CPU prefill of the 0.6B pack at
1,827 positions on the OVH box, width 8: one 1,826-column pass plus the one-column tail. The two
largest leaves are `ffn` (34.66% of the cold wall) and `attn.core` (33.89%); this receipt changes
both in one paired change. The untraced cold turn is **3.934 s** against the base's **4.175 s**
(1.061×, 3/3 rounds) and llama.cpp's **3.936 s** in the same window (1.001×, 2/3 rounds); the traced
window is 3.795 s against 4.049 s and 3.896 s (1.067×, 5/5). The change removes the base's 6.1%
deficit and reaches **parity** with llama; a reliable lead is not established. It does not close T2:
the laptop, the 7B, the 200k context and decode are not measured here.

Grades: `measured` for every wall, phase mean, RSS, minor-fault count, gate exit and the 16-token
stream; `tested` for the equality tests; `model` for the ratios, the `1/(1 − s)` ceilings, the
attribution arithmetic, the retained-buffer sizes, the allocator mechanism and the llama-gap
comparison. Nothing is `proved`.

## Machine, binaries and command

Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC Milan, 32 GiB, 8 sockets × 1 core ×
1 thread), `APEIRON_THREADS=8`, `~/bin/adg` (16 GiB cap, 8 jobs, swap zero). `APEIRON_PREFILL_COLS`
is unset: the 2,048 default splits 1,827 positions into one 1,826-column pass and the one-column
tail (`pass calls 2`). Cache 440 MiB, 420,904,960 B needed. Pack `nq06-v3.nqp`: 196 linears,
0.977441 b/w; KV 28 layers, 8 KV heads × 128, 57,344 values/position, 229,376 B/position.

| arm | tree | sha256 |
|---|---|---|
| base | `82e372a3`; box copy `base2` | `d1b02e32452d19fcc69c604e247d5a10f1001de602fed94d38ee60196937f60a` |
| final | the five modified files | `8a740f0fee047101e1aebf96fd344b7c6bd171309053acb788ae66b7c31518ad` |
| llama | `llama-bench`, llama.cpp `049326a00`, `Qwen3-0.6B-Base.f16.gguf` | — |

The base binary reproduces the merged receipt's `d1b02e32…` exactly from the synced tree; the final
binary is built from the exact tree after `touch src/lib.rs` and `cargo build --release --locked
--example prefix_solve`, with the box copy verified byte-for-byte against the local five files. Two
earlier suites used `933fa98f…` (pre-lint) and `e39bc691…` (pre-unification); the headline suite is
`8a740f0f…`, the unified `u_stage` below.

Command: `prefix_solve --model skeleton-q8head.gguf --pack nq06-v3.nqp --tokenizer tokenizer.json
--len 1827 --differ 0 --decode 0 --mb 440 --phases` (traced) or without `--phases` (untraced), and
`llama-bench -m Qwen3-0.6B-Base.f16.gguf -p 1827 -n 0 -t 8 -r 1`. Rounds interleave base, final and
llama; the gate is load `< 1.5` and no `cargo`/`rustc` before the window.

## The split before the change

Base mean of five traced cold turns (4,049.4 ms). `pool.wait` is an overlay (its time is inside the
leaf that waited), not an additive term. `embed`, `head` and the two `outside` rows are outside
`pass`.

| term | ms | share of wall | `1/(1 − s)` |
|---|---:|---:|---:|
| `ffn` | 1,403.7 | 34.66% | 1.531× |
| `attn.core` | 1,372.4 | 33.89% | 1.513× |
| `attn.proj_in` | 579.0 | 14.30% | 1.167× |
| `attn.proj_out` | 236.4 | 5.84% | 1.062× |
| `norm` | 133.0 | 3.28% | 1.034× |
| `residual` | 64.8 | 1.60% | 1.016× |
| `pass` | 3,824.2 | 94.44% | — |
| `pool.wait` | 205.7 | 5.08% | overlay |
| `embed` | 6.2 | 0.15% | — |
| `head` | 6.6 | 0.16% | outside the pass |
| `outside.alloc` | 180.0 | 4.45% | — |
| `outside.insert` | 32.7 | 0.81% | — |

`ffn` is the dominant leaf and `attn.core` the second, 0.77 pp apart; both are changed. The shares
are mean phase ms over the mean wall, so the last digit is arithmetic (`model`).

## The change

1. **One fused gate/up pass in the one `U` stage.** `Engine::nq_ffn_swiglu_pooled` calls
   `NqLinear::project_gate_up_swiglu_pooled` on the layer's `blk.{l}.4` / `blk.{l}.5` pair. The pair
   enters the existing `project_pooled_body` once with a `PairArgs`: a task packs the gate `t` and
   the up `t` (their `s_pre` differ), folds both `V` stages into two `z` halves, and folds both `U`
   planes in **one** `u_stage` row loop that applies `swiglu_one_carrier` to the pair and writes `h`.
   No gate or up plane is materialized. The one stage is `u_stage<const PAIRED: bool>`: one source
   row/wide-tile loop, specialized at compile time, so the single-linear arm carries no pair branch
   and the paired arm carries no single arm. The separate `u_stage_swiglu` sibling is removed. The
   `F32` carrier takes the eight-lane AVX2 kernel `swiglu_lanes8` (bit-equal to eight scalar calls);
   `Q32` takes the scalar arm. A shape mismatch, an overflow or a `SwiGLU` overflow returns `None`,
   and `proj_cols` runs the separate gate, up and `swiglu_cols` path, which owns the refusal and its
   values.
2. **Retained attention scratch.** `softmax_core_block` takes the stage-1 slot (`PREP_SCRATCH`,
   1,826 × 6,144 × 8 B = **89,751,552 B**) and the tile readout (`TILE_SCRATCH`,
   29 × 8 × 64 × 256 × 8 B = **30,408,704 B**) from thread-local slots and returns them, instead of
   `vec![0; n]` per layer. Both buffers are written in full before any word is read. The reuse is
   the pair's enabling half: fusion alone is refused (below).

## Five paired rounds, traced

Headline suite, final-source binary `8a740f0f…`. Times in seconds; llama is `1827 / pp1827 t/s`.

| round | base | final | llama | base / final |
|---|---:|---:|---:|---:|
| 1 | 3.967 | 3.752 | 3.855 | 1.057 |
| 2 | 3.965 | 3.806 | 3.881 | 1.042 |
| 3 | 4.108 | 3.799 | 3.899 | 1.081 |
| 4 | 4.116 | 3.812 | 3.871 | 1.080 |
| 5 | 4.091 | 3.804 | 3.974 | 1.075 |
| **mean** | **4.049** | **3.795** | **3.896** | **1.067** |
| median | 4.091 | 3.804 | 3.881 | 1.075 |
| spread | 3.81% | 1.60% | 3.07% | 3.80% |

Every base/final round wins; the paired ratios are 1.042–1.081, mean **1.0671**, median **1.0754**
(`model` on the measured walls). The final rate is 481.5 tok/s against the base's 451.2 and llama's
469.0; the final is 1.027× llama's time. RSS is 1,497.0 MiB (base) to 1,524.4 MiB (final), +27.3
MiB, with minor faults 353,599 to 344,185.

## Three paired rounds, untraced

No `--phases`, llama interleaved, same binaries. This is the wall the product sees.

| round | base | final | llama | base / final |
|---|---:|---:|---:|---:|
| 1 | 4.140 | 3.938 | 3.868 | 1.051 |
| 2 | 4.155 | 3.943 | 3.958 | 1.054 |
| 3 | 4.231 | 3.920 | 3.981 | 1.079 |
| **mean** | **4.175** | **3.934** | **3.936** | **1.061** |

Base/final mean ratio **1.0615**, median 1.0538. Against llama the final is **1.001×** at the means
— a dead tie — and wins 2 of 3 rounds; its 0.59% spread is narrower than the base's 2.20%. The
traced window's 1.027× over llama is not reproduced untraced: the trace adds per-call overhead the
fused path reduces, so the traced vs-llama margin is partly instrumentation. Both windows agree on
the base/final gain (1.067 traced, 1.061 untraced); only parity with llama is supported.

## Attribution

Two same-binary pairs isolate the halves from layout. A/B/C is one binary (`prefix_solve.fr`, sha
`4bd4ccfd…`) with the fused branch compiled and gated by a temporary `APEIRON_NQ_FUSE` (removed
before the commit); F/G is one binary (`prefix_solve.f`, sha `bafd2c45…`) with no retention. The arms
predate the unified `u_stage`; the headline suite improves on C's traced wall (3.795 s against
3.933 s) on the committed stage.

| arm | mean, s | control | verdict |
|---|---:|---|---|
| A base | 4.1165 | — | — |
| B retention only (fusion off) | 4.1497 | A | +0.8%, neutral (inside spread) |
| C retention + fusion | 3.9327 | B | −5.23% (1.055×) |
| F fusion code, separate path | 4.1247 | — | — |
| G fusion only | 4.2087 | F | +2.04%, **refused** |

Fusion alone moves `ffn` −154.9 ms but `attn.core` +252.5 ms and the wall +84 ms; retention alone is
flat; the pair gains about 184 ms against the base and 217 ms against retention alone. The pair, not
either half, is the product. The mechanism — the fused pass's transients make the per-layer 86 MiB
stage-1 allocation expensive, and retaining the buffers removes that cost — is `model`, measured
only as the A/B/C wall difference.

## The split after the change

| term | base, ms | final, ms | delta |
|---|---:|---:|---:|
| `attn.core` | 1,372.4 | 1,293.6 | **−78.8** |
| `ffn` | 1,403.7 | 1,219.0 | **−184.7** |
| `pass` | 3,824.2 | 3,561.9 | −262.3 |
| cold wall | 4,049.4 | 3,794.6 | **−254.8** |
| `attn.proj_in` | 579.0 | 554.2 | −24.8 |
| `residual` | 64.8 | 73.8 | +9.0 |
| `embed` | 6.2 | 11.4 | +5.3 |
| `norm` | 133.0 | 137.6 | +4.6 |
| `outside.alloc` | 180.0 | 184.8 | +4.8 |
| `attn.proj_out` | 236.4 | 240.5 | +4.1 |
| `outside.insert` | 32.7 | 34.6 | +2.0 |
| `head` | 6.6 | 7.2 | +0.6 |
| `pool.wait` | 205.7 | 180.1 | −25.6 |

The two targeted leaves move −78.8 and −184.7 ms; the other eight rows move +5.5 ms net (−1.2 ms
across the six non-targeted model leaves, +6.7 ms across the two `outside` rows), inside their run
spread. Post-change, `attn.core` is the dominant leaf at 34.09% (ceiling 1.517×) and `ffn` is
32.13% (ceiling 1.473×); `pass` is 93.87% of the final wall.

## Equality and end-to-end

`engine::kern::nq::tests::the_fused_gate_up_swiglu_equals_the_separate_path` passes: both carriers,
k ∈ {1, 2, 8, 9, 8·width, 8·width + 1} (`NQ_TILE_COLS = 8`), block 0 and `NQ_BLOCK`, cap 0 and
`PLANE_CAP`, with `gate.s_pre ≠ up.s_pre` and `gate.u ≠ up.u` asserted. The fused `h` equals the
separate gate, up and `swiglu_cols` path word for word.
`engine::step::softmax::tests::the_attention_tile_task_agrees_with_the_per_column_task` passes the
tile-vs-per-column equality, but it builds its own `prep`/`tile_buf` vectors and does not call
`softmax_core_block`, so the retained buffers themselves have no direct test; that reuse is `model`
plus the A/B/C wall. `engine::chart::ffn::tests::swiglu_lanes8_equals_eight_scalar_calls`
passes: the entry resolves the AVX2 predicate from the carrier itself (512 deterministic passes over
both carriers, including one saturated lane per fourth pass that must refuse at the same lane as the
scalar oracle), so no caller can hand the `unsafe` kernel a `true` the host does not support.

`generate --prompt "The capital of France is" --n 16`, greedy, `APEIRON_THREADS=8`: the base binary
(rebuilt from `82e372a3`, sha `5e0a4e0c…`) and the final (`60d3d2c4…`) produce the same generated
line, sha `b0385ebc…`; the first three lines are identical and the reports differ only in the two
timing rows. This is a token-stream equality check, not a quality measurement; no PPL is measured
here.

## Gates

| command | result |
|---|---|
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | exit 0 |
| `cargo test --workspace --locked` | stopped at `--lib`: 697 passed, 1 failed, 55 ignored; integration tests, doctests and the workspace members did not run; exit 1 on the known `glslc` check |
| `cargo test --workspace --locked -- --skip coopmat_spirv_is_the_glsl --skip attn_coopmat_spirv_is_the_glsl --skip splitk_reduce_spirv_is_the_glsl` | exit 0: 697 lib tests, every integration test (`spirv_storage_race` 2 passed, 2 filtered) and 7 doctests pass; the three GPU-compiler checks are the box's shaderc 2023.8 limitation |
| `mdbook build book` | exit 0 |
| `mdbook test book -L $HOME/apeiron-mdbook/debug/deps` | exit 0, every chapter |
| `python3 tools/check-claims.py` | OK, every graded claim names a resolvable artifact |
| `python3 tools/book-check.py` | 20 glossary entries, 0 findings |

The lib failure is `gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl`: the box's shaderc
2023.8 rejects `GL_EXT_integer_dot_product`, so the committed `coopmat_gemm.spv` does not reproduce.
The `--all-features` GPU tier needs the `glslc` pin and a device; it runs on neither host. The `unsafe` policy: the new `unsafe` call in `swiglu_lanes8` invokes the pre-existing
`swiglu_f32_avx2_8_into` intrinsic, gated internally by `swiglu_lanes8_avx2(carrier)`
(`src/engine/chart/ffn.rs`), so no caller can supply the predicate. Its safe twin is the eight
scalar `swiglu_one_carrier` calls, and the test above is the equality check; on a host without AVX2
the test exercises the scalar arm only.

## Refusals

1. **"Fusion alone is a win."** Refused: `ffn` falls 154.9 ms, but `attn.core` rises 252.5 ms and
   the wall loses 84 ms (4.2087 vs 4.1247). The fusion is shipped only with the retention.
2. **"Retention alone is a win."** Refused: 4.1497 vs 4.1165 is +0.8%, inside the spread. It is
   shipped as the pair's enabling half, not as a standalone speedup.
3. **"A runtime pair branch in the unified stage is free."** Refused: with the branch, the fused
   pass takes `ffn` from 1,416.5 to 1,282.9 ms (a −133.6 ms gain, `fused2/U` against `fused2/T` in
   the log); the const-generic specialization takes it from 1,432.8 to 1,244.8 (−188.0 ms,
   `fused3`) and 1,403.7 to 1,219.0 (−184.7 ms, headline `fused4`). The branch kept about a third
   of the fused pass's ffn gain out of the wall.
4. **"1.081× is the headline."** Refused: that suite's binary predates the lint-only attribute, and
   the unified code is newer still. The headline is 1.067× traced and 1.061× untraced on the binary
   whose sha matches the committed tree; all three suites win every traced round.
5. **"This closes T2."** Refused: box only, 0.6B only, 1,827 positions, decode 0, q8 head. The
   laptop, the 7B, 200k and decode are open.
6. **"The wall gain is kernel rate alone."** Refused: fusion alone loses; the pair's gain is the
   interaction between the fused pass and the retained scratch (`model`).
7. **"The final is reliably faster than llama."** Refused: untraced 1.001× with 2/3 rounds, traced
   1.027× with 5/5; the untraced margin is zero. The supported claim is parity.
8. **"This reaches the 0.5× campaign target."** Refused: 0.5× in the untraced window would be
   1.968 s per turn; the final is 3.934 s, 2.00× short. No decode is measured.

## Limits

- The box is evidence for the CPU tier, not for the laptop. The target device is the laptop
  (i5-10210U, 15 W, 6 MB L3); no timed result here names it.
- 0.6B, 1,827 positions, cold prefill, `--decode 0`, q8 head, 0.977441 b/w pack. Not the 7B, not the
  200k context, not decode, not the 1.00 b/w product.
- The maintainer gate is not run: the pinned Aeneas/Charon/Lean toolchain is not bootstrapped on
  either host and the committed extraction is stale (`proof/regen.sh --check` fails). The Rust
  arithmetic claim is `tested` by the equality tests, not `proved`.
- The 16-token stream is one prompt; it does not establish quality at any rate.
- The attribution arms predate the unified `u_stage`; they measure the A/B/C and F/G wall splits
  on the pre-unification binaries (the mechanism remains `model`), not the unified stage's code.

## Raw log

`bench/cpu/1795724928-nq-fused-ffn-swiglu.log`: every cold/warm/shared table, the `.time` RSS and
fault rows, the A/B/C and F/G arms, the three five-round suites, the traced and untraced controls,
the intermediate runtime-branch run, the generate diff, the binary shas, and the gate tail.
