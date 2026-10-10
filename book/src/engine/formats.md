# Formats

`apeiron::format` reads three things. Every reader returns `Result` and does not panic on file
data. The GGUF reader and `Engine::open` are the crate's untrusted-input surface, and both are
fuzzed (`fuzz/`).

| module | reads |
|---|---|
| `format::gguf` | GGUF header, metadata table, tensor directory |
| `format::quant` | ggml quantisation blocks |
| `format::bpe` | a HuggingFace `tokenizer.json`: byte-level BPE, ids to text and back |

## Quantisation types

`format::quant::block_geom` is the one table of `(weights per block, bytes per block)`. The size
arithmetic, the payload read and the dequantiser dispatch all use it, so they cannot disagree.

| ggml type | notes |
|---|---|
| `F32`, `F16` | one weight per element |
| `Q8_0`, `Q5_0` | block codes with an `f16` scale |
| `Q2_K`, `Q3_K`, `Q4_K`, `Q6_K` | K-quants, super-blocks of 256 |
| `Q1_0`, `Q2_0` | 1-bit and 2-bit block types |
| `TQ1_0`, `TQ2_0` | ternary types |
| `PQ2_0`, `PTQ1_0` | packed variants with a lookup-table kernel |

Any other type id is `QuantError::UnknownType`. The embedding lookup supports `F16`, `Q8_0`, `Q5_0`,
`Q4_K`, `Q6_K`, `Q2_K` and `Q3_K`; `F32` refuses because its domain needs a rounding contract.

## Exact dequantisation

Every finite `f16` is an integer multiple of `2^-24`, so it is an exact multiple of the Q32.32
unit `2^-32`. `f16_to_fixed` decodes it with shifts and adds. The affine `Q4_K` combination, the
`Q8_0` scale times code product and the `F16` weight are therefore integer arithmetic with no
rounding at load.

```rust
extern crate apeiron;
use apeiron::engine::f16_to_fixed;

assert_eq!(f16_to_fixed(0x3C00).unwrap(), 1i64 << 32); // 1.0
assert_eq!(f16_to_fixed(0xBC00).unwrap(), -(1i64 << 32)); // -1.0
assert_eq!(f16_to_fixed(0x0001).unwrap(), 1i64 << 8); // smallest subnormal, 2^-24
assert!(f16_to_fixed(0x7C00).is_err()); // infinity refuses
```

- **say:** an `f16` with bits `h` becomes the `i64` `round_exact(value(h) * 2^32)`.
- **means:** the conversion is exact for every finite `f16`; `Inf` and `NaN` return an error.
- **vars:** `h`, the 16 `f16` bits.
- **apeiron:** none, anchor. The decoder is tested; no Lean theorem covers it.
- **physics:** none.

## Ranges

Each product is checked, and a failure is `EngineError::FixedOverflow`, never a wrap. The ranges the
source documents for a dequantised weight:

| type | range |
|---|---|
| `f16` | `[-2^48, 2^48]` |
| `Q8_0`, `Q5_0` | `[-2^55, 2^55]` |
| `Q2_K`, `Q3_K` | `[-2^58, 2^58]` |
| `Q4_K` | `[-2^59, 2^59]` |
| `Q6_K` | `[-2^60, 2^60]` |
