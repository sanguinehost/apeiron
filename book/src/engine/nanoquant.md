# The NanoQuant pack

The 1-bit path replaces a linear `W` (`out × inn`) with a sign-rank factor

```text
Ŵ = diag(s_post) · U · diag(s_mid) · V · diag(s_pre)
U ∈ {±1}^{out × r},  V ∈ {±1}^{r × inn}
```

and serves `y = Ŵ·x` without materialising `Ŵ`. The factors and the three scale vectors are the
`APNQPK01` pack (`format::nq`); the untouched tensors come from a GGUF skeleton, and
`Engine::open_with_nq` installs the pack over it.

A pack is written by `examples/nq_fit.rs` from the fitter (one linear or the whole model) and by
`gpu::nn::train::pack::write_pack` from a block-loop checkpoint tree; `tools/nq_campaign.sh` orders
the campaign's five stages and gates the wall with `tools/fit-budget.py`.

`U` is the output-side factor: each of its `out` rows is a sign pattern over the `rank` features.
`V` is the input-side factor: each of its `rank` rows is a sign pattern over the `inn` inputs.
Rank component `k` contributes the rank-1 matrix `s_mid[k]·(s_post ⊙ u_k)(s_pre ⊙ v_k)^T`, so the
pack is a sum of `rank` signed outer products, not a per-weight sign. The sparse term `S` repairs
`k` input columns per output row in the weight's own units.

## The fit's inputs

The fit reads two statistics per linear and one token tape, all from the reference's calibration
pass (`utils/data_utils.py:19-70,112-248`; `core/importance.py`):

| input | definition | producer |
|---|---|---|
| `i_norm` | mean input Gram diagonal, each sample's rows clipped at the `99.9`th percentile of the row L2 norm, shrunk `0.4` toward the mean | `calib_diag_write` |
| `o_norm` | mean squared output gradient of the language-modelling loss in the reference's `1e6` unit, clipped and shrunk the same way | `calib_onorm_write` |
| tape | 128 samples of 2048 tokens from wikitext2 **train** | `nq_calib_base`, `nq_dump_write` |

The tape is the reference's sampling: concatenate the split's rows with `"\n\n"`, tokenize once,
draw 128 slices of 2048 tokens at `randint(0, len − 2049)`, shuffle the sample list, then draw 128
indices with replacement. The statistics are written under the tag `s128`.
`APEIRON_CALIB_STRATEGY=online` (the default) grows the clip threshold as samples arrive and
rescales the accumulator; `two_phase` profiles every sample first and applies one fixed threshold.
`nq_dump_write` adds `calib_ids.i32`, `ranks.json` and the 112 epoch draws (`7` slots × `2` phases
× `8` epochs; the reference calls `set_seed(0)` at the start of every call, so the eight draws
repeat). `tools/nq_campaign.sh` runs statistics → dump → block loop → model KD → pack → PPL.

The forward is the engine's `Q8Act` prefill, not the reference's bf16 `nn.Linear`, and the device
arithmetic is f32 with Q32.32 solves rather than bf16 with Kahan summation. The sample geometry,
the percentile, the `1e6` unit and the shrinkage are the reference's. The draws are this tree's
xorshift64\*, not Python's MT19937 or NumPy's `RandomState`: same distribution and operation order,
different stream. The model KD recomputes the teacher's logits from its final residual instead of
caching all 128 × 2048 × vocabulary tensors; the student gradient is the same. The residual planes
and salient columns in the pack are extensions the reference does not have; they are off unless a
rate below the sign-factor cap asks for them.

| field | shape | bits |
|---|---|---|
| `U` | `out × ceil(r/32)` words | `32·out·ceil(r/32)` |
| `V` | `r × ceil(inn/32)` words | `32·r·ceil(inn/32)` |
| `s_pre`, `s_mid`, `s_post` | bfloat16 vectors | `16·(inn + r + out)` |
| residual layers (optional) | `res_layers × out × ceil(inn/32)` words | 1 bit per weight per layer |
| salient columns (optional) | `k` indices and `out × k` binary16 | small |

One linear therefore stores `r(out + inn) + 16(inn + r + out)` bits when `inn` and `r` are
multiples of 32, plus `32k + 16·out·k` for the sparse columns. The rate is a property of the rank: a
1.003 b/w plan on a 7B (q/o rank 1776, k/v 432, gate/up/down 2998) stores 0.818 GB over 6.525 G
replaced weights, and computes `r·(out + inn)` sign terms — 6.485 G per token. **Bits and terms move
together**: lowering the rank lowers both the bytes and the multiply count, which is why the rate
target and the speed target are one decision, not two.

A sign term is one bit of `U` or `V` against one int8 activation code. The transport is therefore
`1` bit per term plus the scale vectors, and the arithmetic is a selection, not a multiply.

## The shared qkv block

A layer's `q`, `k` and `v` linears read the same input, and the fit gives the three the same `V`,
`s_pre`, `s_mid` and `salient_idx` block; only `U`, `s_post`, `salient` and the residual differ.
Version 4 stores that block once. A `k` or `v` section with `group = 1` is a member: it keeps its own
fields and binds the four shared ones to the same layer's `q` section, which carries `group = 0`.
The reader resolves the binding, so the kernel receives a complete linear and the arithmetic is
unchanged; `stored_bits` counts the block once. A member must be `k` or `v`, its head must be the
same layer's `q`, and the shared bytes must be equal, or the pack refuses with `SharedHead`.

## The readers

| mode | entry | arithmetic |
|---|---|---|
| exact | `NqLinear::project` | each sign plane expanded to `i64` `±1`, three-byte-limb products, checked `i128` |
| q8 activation | `NqLinear::project_q8` | each stage activation quantised to one int8 code per element; the sign dot is `sum − 2·selected` |
| a column group | `project_pooled`, `project_q8_pooled`, `project_ternary_pooled` | one dispatch per linear; a task owns up to a band of columns, rounded up to a multiple of eight when it holds a group, and runs that band's `t` build, quantization, `V` fold, `z` bound and quantization, reader build and `U` fold before the pass's barrier; each factor row read once and folded against every column it owns |

The pooled entries take the same pre-scale, quantization, reader and accumulation order per column,
so each column equals the single-column entry bit for bit; `proj_cols` takes them for `k > 1` and
reruns the group through the per-column entry on a refusal. Pooling removes the repeated factor
read, not the per-column sign dot: at 229 positions the same-binary A/B is 1.16×
(`bench/cpu/1795498875-nq06-pooled-projection.md`). The pooled body retains its two scratch planes
on the calling thread across calls and writes the `U` stage's output straight into each column's
vector, so a thread allocates a plane only when a call is wider than every earlier one, with no
flat output plane or copy; a plane larger than 32 MiB is allocated for the call instead.

A pooled linear is one parallel region. A task owns up to a contiguous band of columns, rounded up to a
multiple of eight when it holds a group, and runs the band's whole pipeline — `t` build, activation
quantization, `V` fold, `z` bound and quantization, residual readers, `U` fold — and the linear ends
on one barrier. No column's reader is built on the dispatching thread between stages, so under the
`Q8` reader the six dispatches the staged body took are one and the per-column work is spread with
the arithmetic. This
is `ggml_compute_forward_mul_mat`'s shape at `049326a00`
(`ggml/src/ggml-cpu/ggml-cpu.c:1254`): each thread gets a row chunk, converts its `src1` rows inside
that chunk, calls `llama_sgemm` for it, and the pass ends on one `ggml_barrier`;
`tinyBLAS::gemm` (`ggml/src/ggml-cpu/llamafile/sgemm.cpp:583`) likewise dispatches tiles from one
region. At 1,827 positions on the 0.6B pack the box A/B is 1.13× at eight
participants (7.854 → 6.957 s, three pairs), 1.07× at four (12.090 → 11.256 s, two pairs) and
0.936× at one, a 6.8% regression (42.086 → 44.955 s, two pairs); the three projection leaves fall
0.831 while
`attn.core` is 0.981, and they remain 57.3% of the wall, so the ceiling a further projection change
can reach is `1/(1 − 0.573) = 2.34×`. Against llama.cpp in the same window the engine is 1.74× behind
at eight (262.6 against 456.8 tok/s); parity sits inside that ceiling but would take about 3.6× on
the projection leaves themselves, and half of llama is below the 2.91 s floor left when projection
cost is zero. The turn is 9.596 → 8.826 s (0.920×): its prefill falls 0.882× (7.652 → 6.753 s)
while its 64 decode tokens cost 6.6% more after a packed prefill, a measured cost whose cause is not
isolated. A shared tile counter was measured and lost at eight (6.74–6.92 s against 6.63–6.70 s)
while winning at one, so the band stays static
(`bench/cpu/1795601655-prefill-onedispatch.md`; the band's per-column equality is gated by
`cargo test --lib --locked kern::nq` and the pooled path's end-to-end equality by
`cargo test --lib --locked engine::step`). The residual add is serial within a column: a
band-parallel version bought 0.335 s at eight in the same window, but it splits
`step::cols::add_cols`, whose extracted body is refined in
`proof/corpus/Serve/Block/Cols2.lean`: the loop stops at the first refusing column and leaves the
rest untouched — a post-state a pool of bands within a column cannot reproduce. Each column's add is
column-local, so the pooled path splits columns, never a column's add.

Under `Q8` the pooled reader also folds a block of factor rows against one read of the column's
code and shift planes. `APEIRON_NQ_BLOCK` selects the block: `0` and `1` are the per-row loop, the
default is 8, and values above 8 clamp. `code_select_block_avx2` is the body, dispatched from
`code_select_rows_avx2` over `R = 2..=8`, and `code_select_rows_twin` is its per-row definition.
At 1,827 positions the 1.09× over eight same-binary pairs and 1.12× over all fifteen is
`measured` on the 0.6B cold projection sum: `bench/cpu/1795518973-nq-blocking.md` runs
`examples/prefix_solve.rs` and gates the reader's equality with
`cargo test --lib --locked engine::kern::nq`. The default moves 4 → 8 on the paired box screen in
`bench/cpu/1795557505-nq-two-column.md` (mean 1.124× over three alternating pairs), where 8 is the
only block with no remainder row on the production shapes; the laptop confirmation of that default
is open.

The column dial is separate from the row block. `APEIRON_NQ_COLS` sets the columns one weight-word
pass folds: `8` or more selects the eight-column tile, anything else the two-column pair, and an
absent value selects the tile at four or more participants and the pair below. The tile's body
`code_select_row8_avx2` shares one expansion of each weight word across eight activation columns
and computes each column's sign dot with its own codes, shift vectors and `e_min`;
`CodeTerms::selected_row_cols` admits it only when all eight stages are Q8 readers, and the caller
keeps the pair for the leftover columns and on a refusal. Every column's words equal the pair
entry's, as `the_wide_source_equals_eight_per_column_sources` asserts. The isolated projection rate
is 1.19× at four participants and 1.22× at eight, but the no-trace cold 1,827-position wall moves
1.005–1.015× at four (median 1.009, tile 3/3) and 0.992–1.030× at eight (median 1.008, 2/3), at or
inside the unchanged arm's spread; the traced pairs repeat the sign. The projection dot is not the
whole of the dominant leaf, so the kernel gain mostly does not reach the wall
(`bench/cpu/1795569181-nq-eight-column.md`).

The tile's readers are assembled once per column group (`Stage::cols8`) and every row of the group
runs against that assembly (`StageCols::dot_row`). On the box at block 256 the per-row admission was
10.3% of the width-8 `apeiron-q8act` worker's instruction stream and 1.6% of the caller's; after the
hoist it is 1.6% and below 0.5%. The isolated tile rate moves 21.97 → 28.30 G terms/s at one
participant and 130.8 → 142.4 at eight, and the cold wall at block 256 moves 1.085× at eight and
1.082× at four. Width 1 is not paired; the laptop is not measured
(`bench/cpu/1795573862-prefill-block-and-group.md`).

The prefill block is the number of activation columns one pass carries; `APEIRON_PREFILL_COLS`
overrides it. The default is **2048** columns at four or more participants and **256** below. On the
box, 0.6B, cold 1,827 positions, 1024/256 is 1.17× at eight participants and 1.15× at four, and 0.97×
at one; at four the gain appears only with the hoist (1.007× before it, and no width-4 phase split
was run), and the threshold is the lowest measured winning width, not a located crossover. One
1,826-column pass is 1.12× of the two 913-column passes the 1,024 default produces (mean of six
paired rounds in two windows, 1.109–1.128×, every round winning); the removed pass costs 462.8 ms of
pass wall, 393 of 1,295 dispatches and 120.5 ms of caller wait, while `attn.core` moves −103.9 ms
with its score and causal-softmax counts split-invariant
(`bench/cpu/1795683267-nq-prefill-one-pass.md`). Width 4 is 1.054× and 3,000 columns 1.035×
(untraced); widths 2–3, 5–7 at 2,048, prompts past 3,000 columns, the 7B and the laptop are
unmeasured. The gain is per-block work — the pooled residual add, attention's per-pass setup, the
norm dispatch and the pool dispatch — measured in the width-8 phase split; `ffn`, the dominant
phase, is flat across 1024/256 (`1795573862`), while removing the 913-column pass drops it by
196.0 ms (`1795683267`). Median peak RSS over three runs at 1,827 positions at width 8 is
1,362,544 KiB = 1.299 GiB at
block 1,024 and 1,429,424 KiB = 1.363 GiB at 2,048 (+65.3 MiB, +4.9%,
`bench/cpu/1795683267-nq-prefill-one-pass.md`); the earlier 1,294,188 KiB row used a different
harness invocation (`1795573862`).

`project_q8` quantises the stage activation with the same chart `matmat_q8act` uses: `S_b = 2^e_b`,
the smallest power of two with `127·S_b ≥ absmax_b`, one code per element. For a sign plane the
product collapses to a selection: with `c_i` the code and `b_i` the sign bit,

```text
Σ sign_i·c_i = Σ c_i − 2·Σ_{b_i = 0} c_i
```

so the kernel keeps the block's code sum and subtracts twice the codes selected by the sign word.
The selector `code_select_avx2` and its scalar twin `code_select_twin` compute the same integer; the
test `the_code_source_equals_the_scalar_source` asserts equality over the boundary lengths. The
blocked selector `code_select_block_avx2` runs the same arithmetic block for block over `R` rows,
and `the_blocked_code_source_equals_the_per_row_source` asserts it equal to `code_select_rows_twin`.
The `Q8Act` dispatch falls back to `project` when the stage bound refuses, so no activation can
overflow a code path that the exact reader would accept.

## The bound

`project_q8` is not bit-identical to `project`: the int8 activation code is a rounding of the
activation, and a stage's error is bounded by `1/127` of that stage's magnitude. The shipped test
`q8_projection_is_within_the_bound_of_exact` asserts a relative bound of `1/32` of the output
magnitude on the fixture, three times the rounding bound, and asserts the deviation is nonzero — a
route mix-up would leave it. On the production shapes the measured deviation is 0.96–1.04% of `|y|`.

## What is established

| claim | grade | gate |
|---|---|---|
| the AVX2 selector equals its scalar twin | tested | `cargo test --lib --locked nq::tests` |
| the q8 projection stays within `1/32` of the exact reader on the fixture | tested | `cargo test --lib --locked nq::tests` |
| a sign flip of one factor moves the output | tested | `cargo test --lib --locked nq::tests` |
| the staged product equals the sequential product | tested | `cargo test --lib --locked nq::tests` |
| the pooled entries equal the per-column entries at `k = 1..256` | tested | `cargo test --lib --locked nq::tests` |
| the blocked selector equals the per-row source at every block `0, 1…8, 13` | tested | `cargo test --lib --locked nq::tests` |
| a packed prefill equals the per-column prefill, logits word for word | tested | `cargo test --lib --locked step::prefill_tests` |
| the pack's stored bits equal the section payload | tested | `cargo test --lib --locked format::nq` |
| a version-4 member binds to its layer's `q` and the shared block is stored once | tested | `cargo test --lib --locked format::nq` |

The extraction does not include `code_select_avx2` or any AVX2 body: Aeneas does not model
`core::arch`, so an intrinsic body is `tested` against its safe twin and never `proved`. The
blocked bodies `code_select_block_avx2`, `code_select_rows_avx2` and the eight-column
`code_select_row8_avx2` are in the same class. The refinement theorem covers the twin and the
integer arithmetic around it.

## What it does not establish

- The sign-rank factor's approximation error against the dense `W`. That is a property of the fit,
  not of the kernel, and it is measured as perplexity, not proved.
- The 1.00 b/w point: no full 7B pack at that rate exists. The 0.6B v3 sample is 53,815,472 B over
  440,401,920 replaced weights, **0.978 b/w**; the 7B ranks above are arithmetic, and the 8B
  residual pack is 3.13 b/w. Neither is the production pack.
- Any GPU carrier for the sign planes: the Vulkan path serves `F16`/`F32`/`Q8`/`Q4`/`Q2`/`Q3`/`Q5`
  and has no 1-bit variant.
