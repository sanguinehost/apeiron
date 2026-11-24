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
| a column group | `project_pooled`, `project_q8_pooled`, `project_ternary_pooled` | one `V` dispatch and one `U` dispatch for the group; each factor row read once and folded against every column it owns |

The pooled entries take the same pre-scale, quantization, reader and accumulation order per column,
so each column equals the single-column entry bit for bit; `proj_cols` takes them for `k > 1` and
reruns the group through the per-column entry on a refusal. Pooling removes the repeated factor
read, not the per-column sign dot: at 229 positions the same-binary A/B is 1.16× and the ceiling
from the projection's 51.4% share of a 1,827-position cold turn is 2.06×
(`bench/cpu/1795498875-nq06-pooled-projection.md`).

Under `Q8` the pooled reader also folds a block of factor rows against one read of the column's
code and shift planes. `APEIRON_NQ_BLOCK` selects the block: `0` and `1` are the per-row loop, the
default is 4, and values above 8 clamp. `code_select_block_avx2` is the body, dispatched from
`code_select_rows_avx2` over `R = 2..=8`, and `code_select_rows_twin` is its per-row definition.
At 1,827 positions the 1.09× over eight same-binary pairs and 1.12× over all fifteen is
`measured` on the 0.6B cold projection sum: `bench/cpu/1795518973-nq-blocking.md` runs
`examples/prefix_solve.rs` and gates the reader's equality with
`cargo test --lib --locked engine::kern::nq`. Block 8 is not separated from 4 on the laptop, so 4
stays the default.

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
blocked bodies `code_select_block_avx2` and `code_select_rows_avx2` are in the same class. The
refinement theorem covers the twin and the integer arithmetic around it.

## What it does not establish

- The sign-rank factor's approximation error against the dense `W`. That is a property of the fit,
  not of the kernel, and it is measured as perplexity, not proved.
- The 1.00 b/w point: no full 7B pack at that rate exists. The 0.6B v3 sample is 53,815,472 B over
  440,401,920 replaced weights, **0.978 b/w**; the 7B ranks above are arithmetic, and the 8B
  residual pack is 3.13 b/w. Neither is the production pack.
- Any GPU carrier for the sign planes: the Vulkan path serves `F16`/`F32`/`Q8`/`Q4`/`Q2`/`Q3`/`Q5`
  and has no 1-bit variant.
