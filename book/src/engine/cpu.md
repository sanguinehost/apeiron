# CPU path

The CPU engine runs a decoder transformer in integer arithmetic. `apeiron::engine::Engine` opens a
GGUF file, and `step` advances one token. A model is a per-layer schedule of mixers (attention,
`DeltaNet`) and feed-forward blocks. It is configuration and a schedule, not a separate code path
per architecture.

## Fixed point

Every value on the token path is an `i64` in Q32.32: `v` denotes `v / 2^32`.

- **say:** a value `v` stands for the real number `v` divided by two to the power 32.
- **means:** the unit is `2^-32`. Products of two Q32.32 values are Q64.64 and are summed in `i128`.
  One round-to-nearest shift right by 32 (ties upward) returns the sum to Q32.32.
- **vars:** `v`, the stored `i64`.
- **apeiron:** none, anchor. The rounding epilogue is tested, not proved.
- **physics:** none.

A product or sum outside the `i64` range returns an error. Nothing clamps and nothing wraps.

## Kernels

The projection has one path: `matmat_q8act` walks the weight rows once and folds each decoded group
against every activation column. The carrier, column count, dot core and accumulator are dials on
that loop; see [One path](onepath.md). The table below lists the operations around it.

| operation | function | semantics |
|---|---|---|
| projection | `matmat_q8act` | one row loop, `k` columns, carrier dial |
| exact projection (oracle) | `matvec_fixed` | exact `i128` row dot, one rounding per output |
| int8-activation projection | `KernelMode::Q8Act` | defined as `matvec_fixed` on the quantised activation, bit for bit |
| RMS norm | `rms_norm_fixed` | fixed-point `rsqrt` |
| rotary embedding | `RopeTable::rotate` | fixed-point sin and cos, angle window `[-4096, 4096]` rad |
| activation | `silu_fixed` | fixed-point |
| sampling | `argmax_fixed`, `top_p_fixed` | over Q32.32 logits |

`KernelMode::Exact` is the default. `Q8Act` quantises each 32-wide activation block to int8 codes
with a power-of-two scale `S_b = 2^e_b`, the smallest with `127 * S_b >= absmax_b`. The dequantised
activation is exact in Q32.32, and the scale becomes a shift in the kernel. The row dot factors
exactly over weight groups, the inner sums are `i32`, and the group and block products are checked
`i128`. Where a bound cannot rule out an overflow the call falls back to `matvec_fixed`, so no
refusal differs between the two modes.

## SIMD and threads

- The inner group dot runs an AVX2 core on x86-64 behind `is_x86_feature_detected!("avx2")`. A
  portable lane loop is the fallback and the test oracle. Equality of the two is `tested`; the
  intrinsics are not extracted to Lean.
- A persistent worker pool replaces per-call thread spawning. A dispatch is one padded atomic
  store per worker. On a host with efficiency cores, the pool pins one worker per efficiency core
  and chooses per dispatch between a round-robin and a speed-weighted contiguous split by modelled
  makespan. Thread affinity uses `libc` on Linux; that is the crate's only non-optional dependency.

## Key/value cache

The cache has its own carrier dial, `APEIRON_KV_CARRIER`:

| value | plane | bytes per element |
|---|---|---:|
| `q32` | Q32.32 `i64`, position-major | 8 |
| `f32` | binary32, position-major | 4 |
| `i8` | signed code plus one binary32 scale per `(kv_head, position, K or V)`, KV-head-major | 1 + 4/head_dim |

Unset falls back to `APEIRON_F32_CARRIER`: `0` selects `q32`, every other value `f32`. An int8 row
quantises at append to `s = max|x| / 127` and `code = clamp(round_ties_even(x / s), -127, 127)`;
`-128` is never stored, so negation is exact. The read reconstructs `code · s` in binary32.

One pool task owns one KV head and its `group = n_head / n_head_kv` query heads. Under `i8` the
task reads each K and V code row once and folds every query head of the group against it; under
`q32` and `f32` it issues the per-head reads in the same order, so the words are the per-head
words. On a prefill block the `f32` carrier instead dispatches one task per (tile of 32 consecutive
columns, KV head): `attn_core_f32_tile` loads each K and V row once per tile pass and folds every
query head of every column in the tile against it, running each column's own online-softmax
recurrence, so its words equal the per-column read's. `APEIRON_ATTN_TILE=0` selects the per-column
dispatch for a same-binary A/B. The int8 scalar body and its AVX2 body are bit-identical; the AVX2
body is opaque to the extraction and `tested` equal to the scalar twin. The block certificate is
`q32`-only.

## What is established

| claim | grade | gate |
|---|---|---|
| `f16` to Q32.32 conversion is exact for every finite `f16` | tested | `cargo test --workspace --all-features --locked` |
| `Q8Act` equals the exact projection on the quantised activation | tested | `cargo test --workspace --all-features --locked` |
| AVX2 core equals the portable core | tested | `cargo test --workspace --all-features --locked` |
| the int8 KV group read equals the per-head read of each slot | tested | `cargo test --workspace --all-features --locked` |
| the F32 tile read equals the per-column group read, word for word | tested | `cargo test --workspace --all-features --locked` |
| the int8 KV scalar and AVX2 bodies are bit-identical | tested | `cargo test --workspace --all-features --locked` |
| the int8 KV read matches an f64 causal-softmax oracle | tested | `cargo test --workspace --all-features --locked` |
| the executed engine refines a Lean specification | open | none |

The [trust boundary](../verification/trust.md) lists the layers the evidence does not reach.
Throughput numbers and how to reproduce them are in [Results](results.md).
