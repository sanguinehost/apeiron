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

| operation | function | semantics |
|---|---|---|
| projection | `matvec_fixed` | exact `i128` row dot, one rounding per output |
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

## What is established

| claim | grade | gate |
|---|---|---|
| `f16` to Q32.32 conversion is exact for every finite `f16` | tested | `cargo test --workspace --all-features --locked` |
| `Q8Act` equals the exact projection on the quantised activation | tested | `cargo test --workspace --all-features --locked` |
| AVX2 core equals the portable core | tested | `cargo test --workspace --all-features --locked` |
| the executed engine refines a Lean specification | open | none |

The [trust boundary](../verification/trust.md) lists the layers the evidence does not reach.
Throughput numbers and how to reproduce them are in [Results](results.md).
