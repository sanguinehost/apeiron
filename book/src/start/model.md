# Run a model

The engine reads a GGUF file and decodes on the CPU. It uses no float on the token path: every
weight, activation and logit is an `i64` in Q32.32 (an `i64` `v` denotes `v / 2^32`). The
[fixed-point convention](../engine/cpu.md) is in the Engine part.

You need a GGUF model and its `tokenizer.json`. The loader reads `general.architecture` from the
file. The examples and receipts name `qwen2` (Qwen2.5-0.5B) and `qwen35` (Qwen3.5-0.8B) models; no
other architecture is claimed. The quantisation types it reads are in [Formats](../engine/formats.md).

## Greedy decode

```sh
cargo run --release --locked --example engine_decode -- <model.gguf> <tokenizer.json> 16
```

Arguments are `[gguf] [tokenizer.json] [n]`. The example tokenises a fixed prompt, prefills it,
then greedy-decodes `n` tokens. It prints the prompt ids, the generated ids, the detokenised
completion and the wall-clock tokens per second. Two environment variables change the run:

| variable | effect |
|---|---|
| `APEIRON_KERNEL=q8act` | int8-activation projections instead of the exact path |
| `APEIRON_PROMPT_IDS` | read prompt ids from a file instead of tokenising |

Omit the gguf and tokenizer arguments only if you have put the models where the example's defaults
point. Pass both paths explicitly otherwise.

## Check a projection by hand

The kernel under every projection is `matvec_fixed`. It takes the ggml type, the raw row-major
weight bytes, the output and input widths, and a Q32.32 activation. This runs on two rows of `F16`
weights, with no model file.

```rust
extern crate apeiron;
use apeiron::engine::matvec_fixed;
use apeiron::format::quant::GGML_TYPE_F16;

// Rows [1.0, 2.0] and [0.5, -1.0] as little-endian f16 bit patterns.
let w: Vec<u8> = [0x3C00u16, 0x4000, 0x3800, 0xBC00]
    .iter()
    .flat_map(|h| h.to_le_bytes())
    .collect();
// x = [1.0, 0.5] in Q32.32.
let x = [1i64 << 32, 1i64 << 31];

let y = matvec_fixed(GGML_TYPE_F16, &w, 2, 2, &x).unwrap();
assert_eq!(y, vec![2i64 << 32, 0]); // [2.0, 0.0]
```

Each output is an exact `i128` sum of products with one round-to-nearest shift. An overflow is an
`Err`, never a wrap.

Next: [Run on the GPU](vulkan.md).
