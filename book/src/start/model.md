# Run a model

The engine reads a GGUF file and decodes on the CPU. It uses no float on the token path: every
weight, activation and logit is an `i64` in Q32.32 (an `i64` `v` denotes `v / 2^32`). The
[fixed-point convention](../engine/cpu.md) is in the Engine part.

You need a GGUF model and its `tokenizer.json`. The loader reads `general.architecture` from the
file. The examples and receipts name `qwen2` (Qwen2.5-0.5B) and `qwen35` (Qwen3.5-0.8B) models; no
other architecture is claimed. The quantisation types it reads are in [Formats](../engine/formats.md).

## Generate text

```sh
cargo run --release --locked --example generate -- \
  --model <model.gguf> --tokenizer <tokenizer.json> --chat --prompt "What is the capital of France?"
```

The completion streams to stdout. The prompt token count, prefill and decode throughput, and the
stop reason go to stderr. Decoding is greedy.

| flag | effect |
|---|---|
| `--model`, `--tokenizer` | required; the GGUF file and its `tokenizer.json` |
| `--prompt TEXT` | the prompt; read from stdin when absent |
| `--chat` | wrap the prompt as one ChatML turn and stop at the end-of-turn token |
| `--system TEXT` | the system message with `--chat` |
| `--n N` | at most `N` generated tokens, default 64 |
| `--kernel q8act\|exact` | `q8act` (default) runs int8-activation projections; `exact` is the i128 reference path, about ten times slower |

A prompt id at or above the model's vocabulary, an empty prompt, and a prompt plus `n` beyond the
KV capacity are refused with an error.

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
