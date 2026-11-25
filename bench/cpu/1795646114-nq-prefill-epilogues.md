# nq: the dense epilogue and the pooled residual add

Criterion: **T2** (turn latency), partial: the cold 1,827-position CPU prefill of the 0.6B pack.
Both changes are bit-exact on the PPL stream, so this is not a quality claim. It is not a 7B
result, not a 200k turn and not a decode result.

Grades: `measured` for the walls, the PPL pair, the phase splits and the profile; `tested` for the
three equality/error tests and the two mutation controls. No `proved` claim: the changed bodies sit
in the extraction closure, but the committed extraction predates the grouped change and the Lean
bridge did not run (Omissions). Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC
Milan, 31.3 GiB, 256 MiB L3, 2.4 GHz base, no AVX-512, no VNNI), no GPU. The laptop — the only
device a timed claim may name — is **not measured**. Times UTC; every job under `~/bin/adg run --`.

Model: `~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf`; pack
`~/models/nq-scratch/nq06-v3.nqp` (196 linears, 28 layers, 0.977441 b/w); harness
`examples/prefix_solve.rs`, `--len 1827 --differ 205 --decode 0 --mb 440`, width 8,
`APEIRON_PREFILL_COLS=1024`. PPL: `examples/ppl`, 8,176 tokens of `~/models/wiki.test.raw`, 16
windows of ≤ 512, `--kernel q8act --head-batch 16`.

## What changed

Commits `2594e28b` (`src/engine/kern/nq.rs`) and `bcb3806c` (`src/engine/step/cols.rs`).

### `2594e28b`: the dense projection epilogue

Every linear of the 0.6B v3 pack has `k = 0` salient columns and `res_layers = 0`, so
`project_row`'s sparse correction and residual loops are zero-trip in every hot call while the
entry stayed out of line. `project_scaled(dot, s)` is that branch written once —
`round(dot·s / 2^32)` with the checked product and rounding term and an `i64` result — and
`project_row` early-returns it. The wide U tile hoists `dense` and `s_post[i]` out of its
eight-column inner loop and calls `project_scaled` directly; the general arm still calls
`project_row`, which is the same arithmetic. This is a dial on the one row loop, not a second
projection path.

`the_dense_epilogue_rounds_the_scaled_dot` checks the scalar table (including both refusals) against
the expression written in the test, and the pooled output against the per-column `project_q8` entry
on a `packed_dense` fixture at `k = 1, 7, 33` and `NQ_TILE_COLS × width`. The last count is the
first whose pool band holds eight columns, i.e. the first that reaches the hoisted branch; with the
hoisted scale mutated to `s_post[(i + 1) % out]` the test fails only at `k 64, tile 8`, and
restoring the source turns it green.

### `bcb3806c`: the pooled residual add

The residual add is the output epilogue of `attn.proj_out` and the FFN `down` projection: it
streams `hidden[c]` and the projection's `delta[c]` for every column, two `k × n` planes per layer.
It ran on the calling thread while the pool's workers spun. `add_cols` now splits the columns over
`chunks_mut_global` in `⌈k / (4·width)⌉` tasks and records the lowest failing column through
`FirstErr<usize, StepError>`; the `k < 2` and single-worker cases keep `add_cols_serial`.

On success every word is the serial column-order add's, bit for bit: each column's
`add_assign_fixed` reads and writes only that column. A refusal reports the error of the lowest
failing column; the contents of `hidden` after a refusal are unspecified, and the function says so.
That is the only domain the parallel arm reduces.

`the_residual_add_is_the_column_order` compares the pooled result with the serial loop at `k = 1,
2, 9, 40` and on the common-prefix domain when the two column lists differ.
`the_residual_refusal_is_the_lowest_column` puts an overflow at column 3 against a row-width
mismatch at column 20 in both key orders; the lowest column decides, not the first task to finish.
Changing `delta[start + i]` to `delta[i]` fails both tests; restoring the source turns them green.

## Quality

Same stream and window, two binaries:

| arm | PPL | mean nll | tok/s | wall s |
|---|---:|---:|---:|---:|
| grouped (`ppl.final`, `a540bc6a…`) | 53.991898 | 3.988834 | 168.5 | 48.5 |
| final (`ppl.final2`, `18fe984d…`) | **53.991898** | **3.988834** | 187.1 | 43.7 |

The dense-epilogue arm reproduced the same 53.991898 against the grouped binary in the previous
window, and the pooled add leaves every successful word unchanged, so the final source reproduces
the grouped PPL exactly rather than approximately. No PPL movement is claimed or observed. The
grouped receipt's activation-scale screen stands as the last quality change.

## Result

### Cold prefill: four arms in one window

The serial arm is the final source with `add_cols` forced to `add_cols_serial`; it isolates the
dense epilogue (plus the `#[inline]` source state) from the pooled add. Three interleaved rounds in
one window:

| round | merge4 s | grouped s | serial s | final s |
|:---:|---:|---:|---:|---:|
| 1 | 7.196 | 6.564 | 6.295 | 5.743 |
| 2 | 7.222 | 6.369 | 6.227 | 5.787 |
| 3 | 7.171 | 6.579 | 6.309 | 5.753 |
| mean | **7.196** | **6.504** | **6.277** | **5.761** |
| / grouped | 1.106 | 1 | **0.965** | **0.886** |

Binary md5s: `merge4` `a8c6d5b2…`, grouped `b7058658…`, serial `aa065629…`, final `df6ef8fe…`;
the final's hash was unchanged after the profile run.

Pairs: serial/grouped `0.959, 0.978, 0.959` (mean **0.965**); final/serial `0.912, 0.929, 0.912`
(**0.918**); final/grouped `0.875, 0.909, 0.874` (**0.886**). Every pair moves the same way. The
grouped arm's spread is 6.369–6.579 (3.2% of its mean), the serial arm's 6.227–6.309 (1.3%) and the
final arm's 5.743–5.787 (0.8%); the round-2 grouped run is the low outlier and the pair ratios
absorb it. The rate is **280.9 → 317.1 tok/s** at width 8 (mean walls) and 319.2 tok/s in the phase
run below.

### Phase split

`--phases`, one cold run per arm, same window. Single runs: the paired wall is graded and the split
is read as shares.

| phase | grouped ms | serial ms | final ms | final share |
|---|---:|---:|---:|---:|
| `ffn` | 1958.878 | 1867.216 | **1884.056** | 32.9% |
| `attn.proj_in` | 877.129 | 813.839 | **813.642** | 14.2% |
| `attn.proj_out` | 339.903 | 349.786 | **327.529** | 5.7% |
| `attn.core` | 1768.150 | 1768.933 | **1807.743** | 31.6% |
| `norm` | 320.877 | 346.654 | **362.930** | 6.3% |
| `residual` | 640.339 | 677.589 | **113.173** | **2.0%** |
| `pool.wait` | 143.526 | 173.320 | 124.959 | 2.2% |
| `pass` | 5942.348 | 5864.753 | **5356.677** | 93.6% |
| cold wall | 6.336 s | 6.232 s | **5.723 s** | |

The three projection leaves are **3175.910 → 3030.841 → 3025.227 ms**. The dense epilogue takes
4.6% of the leaves in these single runs (the paired wall says 0.965 for that commit); the pooled add
takes the residual **677.589 → 113.173 ms (0.167)** and the paired wall 0.918. The grouped residual
(640.339) sits below the serial arm's (677.589) on a body neither change edits: the single-run
residual moves with memory state, which is why the paired wall, not the split, is the graded
number. The final residual is 2.0% of the wall, down from 11.4% of the dense arm (10.9% of the
serial control); a residual-only change inherited `1/(1 − 0.114) = 1.13×`, and the measured 0.918
also carries the pooled tasks' dispatch change.

`ffn` is now the largest leaf at 32.9%, ahead of `attn.core` at 31.6%. The projection leaves are
52.9% of the final wall, so a further projection change inherits **`1/(1 − 0.529) = 2.12×`**. The
residual the pooled add leaves open is the 113 ms dispatch+join.

### Where the time is

`perf record -F 499 -g` on the final binary, one cold run (5.878 s, 193.3G cycles), flat by symbol:

| symbol | self |
|---|---:|
| pool worker entry (spin/dispatch) | 21.81% |
| `code_select_row8_avx2` | 20.49% |
| `gemm_tile` (attention qk) | 9.57% |
| `ColPipeline::run` | 9.35% |
| `attn_core_tile_avx2_inner` | 8.69% |
| `quant_act_grouped` | 5.25% |
| `swiglu_f32_avx2_8_into` | 2.81% |
| `rms_norm_fixed` | 2.40% |

`StageCols::dot_row` is 22.77% children, 2.13% self; its child is the selector. The pool entry is
the worker thread's spin/dispatch loop, i.e. idle samples during the caller's serial phases, not
projection work; its share does not fall with this change (21.0% before, 21.8% here) and the pooled
add's work moves into it. The projection terms that remain are the dot family — selector 20.5%,
`dot_row` wrapper 2.1% — and `ColPipeline::run` self (9.35%, the stage build and per-output
epilogue) plus `quant_act_grouped` (5.25%). The next ffn work is in that staging, not in the dot.

### Against llama.cpp

llama.cpp was not re-run in this window: its width-8 row is **4.00 s (3.94–4.06)** from
`1795601655-prefill-onedispatch.md`. `merge4` is the binary of that window; it reads 7.196 s here
where it read 6.957 s there, so this window's factor is **1.034**. Scaling each arm by it:

| arm | wall s | clean-equivalent s | against llama 4.00 s |
|---|---:|---:|---:|
| grouped | 6.504 | 6.288 | 1.57× behind |
| serial (dense) | 6.277 | 6.068 | 1.52× behind |
| final | **5.761** | **5.570** | **1.39× behind** |

The in-window ratio is `5.761 / 4.00 = 1.44×`; the clean-equivalent is **1.39×**. Propagating the
final's 5.743–5.787 s and llama's 3.94–4.06 s gives 1.37–1.42× clean, so the statement is "about
1.4× behind", not a measured interval.

The target is **1×–0.5×** (4.00–2.00 s). **Neither is reached**: parity needs the wall to fall a
further **1.39×** (1.76 s) and half needs a further **2.78×**. The two changes move the paired wall
0.886; the remaining 1.39× is attention (31.6%) and the projection staging (52.9% leaves), not the
residual or the dense-row epilogue.

## Omissions

- The laptop is not measured. The box is shared and its absolute walls are window-specific; the
  interleaved ratio is the graded number and the clean-equivalent is a scaling, not a second run.
- The extraction and bridge are stale: `proof/regen.sh --check` has not run since `57b4352d`, no
  pinned Aeneas/Charon/Lean is installed here, and the grouped change's `quant_act_grouped` bridge
  lemma is still open. `2594e28b` and `bcb3806c` are therefore `tested`, not `proved`. The changed
  `add_cols` body is in the Charon closure, so the maintainer gate inherits it.
- `cargo test --workspace --all-features` needs the `glslc` pin and a GPU device; neither host has
  them. On the box, `cargo test --workspace --locked` is **682 passed, 1 failed, 54 ignored**, and
  the failure is the documented `gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl` byte check
  (no `glslc`). `cargo clippy -j 1 --workspace --all-features --all-targets --locked -- -D
  warnings`, `cargo fmt --all -- --check`, `tools/check-claims.py`, `tools/book-check.py`,
  `mdbook build book` and `mdbook test book` are green on the final source.
- The dense epilogue's gain is measured on a pack where all 196 linears are dense. A pack with
  salient columns or residual layers keeps the general arm and is not measured here.
- The phase splits are single runs; the grouped column of the split is a different run from the
  grouped wall mean. No unchanged leaf's movement is attributed.
- No decode, turn, 7B, context or GPU claim. The `attn.core` share (31.6%) is where the next
  attention change must start; it is not edited here.
