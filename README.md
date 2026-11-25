# apeiron

`apeiron` is a Rust library for the algebra `R[x]/(x² − σ)` (`Cell<R>` and its norm form
`a² − σb²`), a deterministic flat/spill hash table (`freq::Table`, `CoordMap`, `FreqMap`), and a
CPU/GPU inference engine for GGUF models. `Cell<i64>` word arithmetic is extracted to Lean 4 and
checked against canonical ring definitions in `proof/`. Default builds have zero crate dependencies
(`libc` on Linux for thread affinity). Rust 1.89 or later. Apache-2.0.

```rust
use apeiron::{Cell, Sigma};

let z = Cell::<i64>::of(3, 2);
assert_eq!(z.norm_form(Sigma::Elliptic), 13);   // a² + b²  (σ = -1)
assert_eq!(z.norm_form(Sigma::Parabolic), 9);   // a²       (σ =  0)
assert_eq!(z.norm_form(Sigma::Hyperbolic), 5);  // a² - b²  (σ = +1)
```

## Why one algebra and one path

One ring `R[x]/(x² − σ)` with `σ ∈ {−1, 0, 1}` covers the complex (elliptic), dual (parabolic), and
split-complex (hyperbolic) faces. Every operation takes `Sigma` as a value, so face-specific
capabilities — two independent channels and lossy discard on the hyperbolic face — are enforced by
the type instead of duplicated across modules.

The CPU engine applies the same structure: `matmat_q8act` is the single projection path. It walks
weight rows once, decodes each 32-lane group through a carrier dial (`F16`, `Q8_0`, `Q5_0`, `Q4_K`,
`Q6_K`, ternary, or 1-bit sign factors), and folds that group against all activation columns (`k = 1`
decode, `k > 1` prefill). A weight format is a carrier value, not a new kernel. (`ternary` is an
exploratory dial refuted on quality in [1794834226-ternary-ppl](bench/cpu/1794834226-ternary-ppl.md);
it is not a product route.)

This gives one loop to optimise, one set of bounds to prove, one overflow refusal route, and exact
integer accounting (Q32.32 fixed-point on the token path).

| what this buys | what it costs |
|---|---|
| one row loop, thread split, and checked `i128` refusal path across formats | narrow model coverage (`qwen2`, `qwen3`/`qwen35` GGUF architectures) |
| deterministic fixed-point arithmetic and bounded table probes | scaling overhead on unfitted formats; `Q8_0` VNNI debt still beside the loop |
| every `unsafe` SIMD core carries a safe twin and equality test | slower kernel iteration; extracted bodies need the Lean maintainer gate |

## Why apeiron (and why not)

| | apeiron | Burn / Candle | llama.cpp |
|---|---|---|---|
| scope | narrow vertical stack: algebra, tables, one GGUF path, quantiser, context method | general tensor and ML frameworks (training and inference) | broad LLM inference runtime across many architectures and backends |
| arithmetic | Q32.32 integer path; `Cell<i64>` word bodies refined in Lean 4 | floating-point and quantised tensors; no formal refinement | C/C++ floating-point and quantised kernels; no formal refinement |
| maturity | research crate; not yet a llama.cpp replacement | established Rust ML ecosystems | production baseline for local GGUF inference |

`apeiron` keeps the algebra, hash tables, quantiser, context method, and CPU/Vulkan kernels in one
tree so arithmetic bounds hold end to end. It is not a training framework, not a general tensor
library, and not yet a llama.cpp replacement.

## Use

Clone and build (optional features: `--features gpu` for runtime-loaded Vulkan, `--features derive`
for `Coord` and `Packed`):

```sh
git clone https://github.com/sanguinehost/apeiron
cd apeiron
cargo build --lib
```

Multiply cells on a chosen face, or aggregate keys in the deterministic flat/spill table:

```rust
use apeiron::{Cell, Sigma};
use apeiron::freq::Table;

let p = Cell::<i64>::of(3, 2).mul(Sigma::Hyperbolic, Cell::<i64>::of(1, 5));
assert_eq!((p.a, p.b), (13, 17));

let mut counts = Table::<u64, i64>::new();
for word in [7u64, 9, 7, 7, 9, 4] {
    counts.insert_with(word, 1, |old, new| old + new);
}
assert_eq!(counts.get(&7), Some(&3));
assert_eq!(counts.len(), 3);
```

Build and test the book:

```sh
mdbook build book && mdbook test book -L target/debug/deps
```

Run a model on the CPU (add `--pack <model.nqp>` for a NanoQuant pack; `serve/` is a sibling
workspace that builds `nq-serve`, an OpenAI-compatible HTTP endpoint, keeping `src/` and `examples/`
socket-free):

```sh
python3 tools/nq_convert.py --source <model-nq.gguf> --out-dir <out-dir>
cargo run --release --locked --example generate -- \
  --model <model.gguf> --tokenizer <tokenizer.json> --chat --prompt "What is 17 times 23?"
cargo run --release --locked --example ppl -- \
  --model <model.gguf> --tokenizer <tokenizer.json> --data <wiki.test.raw> --tokens 2048 --ctx 512
cargo run --release --locked --example prefix_solve -- \
  --model <model.gguf> --tokenizer <tokenizer.json> --pack <model.nqp> --len 1827 --differ 205 --decode 16
cargo run --release --locked --example phase_split -- --model <model.gguf> --pack <model.nqp>
cd serve && cargo build --release && ./target/release/nq-serve \
  --model <skeleton.gguf> --pack <model.nqp> --tokenizer <tokenizer.json> --bind 127.0.0.1:8080
```

GGUF models, `.nqp` packs, `tokenizer.json`, and evaluation text (`wiki.test.raw`) are not in the
repository. Receipts in `bench/cpu/` and `bench/gpu/` name every file by SHA-256; only
`tests/fixtures/qwen35-prompt2.ids` is tracked in-tree.

## Status

The goal in [GOAL.md](GOAL.md) is a 7B-class model on a 6.6 GiB laptop at about 30 tok/s CPU decode
with a 200,000-token context and at least 2 GiB free memory. **That goal is open.**

**Dense GGUF CPU parity** (i9-14900KF, 5 paired rounds vs llama.cpp `60bc8866b`, ratio is apeiron
over llama.cpp; Q4_K_M recorded from a dirty tree without first-token agreement):

| metric | checkpoint | apeiron | llama.cpp | ratio | receipt |
|---|---|---:|---:|---:|---|
| pp512 | Qwen3.5-0.8B Q8_0 | 917.82 tok/s | 416.41 tok/s | 2.24× (2.14–2.25) | [1792648053](bench/cpu/1792648053-qwen08b-q8-cpu.md) |
| tg128 | Qwen3.5-0.8B Q8_0 | 92.13 tok/s | 51.42 tok/s | 1.80× (1.70–1.81) | [1792648053](bench/cpu/1792648053-qwen08b-q8-cpu.md) |
| pp512 | Qwen3.5-0.8B Q4_K_M | 658.10 tok/s | 632.67 tok/s | 1.02× (1.00–1.10) | [1792653496](bench/cpu/1792653496-qwen08b-q4km-kqc-live.md) |
| tg128 | Qwen3.5-0.8B Q4_K_M | 84.19 tok/s | 60.39 tok/s | 1.39× (1.33–1.40) | [1792653496](bench/cpu/1792653496-qwen08b-q4km-kqc-live.md) |

**NanoQuant packs:**

- **0.6B cold prefill (1,827 tokens):** behind llama.cpp (`049326a`, F16) by **2.07×** at width 4
  (127.66 vs 264.11 tok/s) and **2.59×** at width 8 (177.28 vs 458.87 tok/s) on an 8-vCPU EPYC box
  ([1795573862](bench/cpu/1795573862-prefill-block-and-group.md)); earlier per-column laptop runs
  were 3.69–4.25× behind ([1795490586](bench/cpu/1795490586-nq06-vs-llamacpp.md)).
- **0.6B decode:** **21.42 tok/s** (20.40–22.43) at width 4 on the i5-10210U laptop with a `Q8_0`
  head ([1794813322](bench/cpu/1794813322-q8-head-fixture.md)).
- **8B quality:** v2 (1.124 b/w on replaced linears) is **1.756×** same-checkpoint Q4_K_M at ctx
  128 (28.31 vs 16.12; [1794653685](bench/cpu/1794653685-nq8b-residual-ppl.md)) and **1.607×** at
  ctx 2048 (`Q8_0` head vs stock `Q6_K`, laptop decode 2.25 tok/s;
  [1794879503](bench/cpu/1794879503-nq8b-1124-ctx2048.md)). Adding two sign-residual layers (v3)
  reaches **1.011×** Q4_K_M at ctx 128 (16.30 vs 16.12) at **3.13 b/w** on replaced linears and
  **80.5%** of Q4_K_M's bytes (4.05 GB vs 5.03 GB; [1794653685](bench/cpu/1794653685-nq8b-residual-ppl.md)).
- **7B target (≤1.00 b/w, ≤1.5× Q4_K_M PPL, ≥20 tok/s):** not met. No 7B claim is made.

**Proved, tested, measured, and open:**

| statement | grade | gate |
|---|---|---|
| `Cell<i64>` multiply and norm form refine canonical ring operations | proved | `word_mul_denote`, `word_norm_denote` in `proof/corpus/WordRefinement.lean` |
| Lean axiom cone stays within `propext`, `Classical.choice`, `Quot.sound` | proved | `proof/corpus/AxiomAudit.lean`, checked by `lake build` |
| SPIR-V fragment header check | proved | `Spirv.checkFragment_header` |
| `Cell`, tables, engine, HashDoS bound, and SPIR-V checker boundaries | tested | `cargo test --workspace --all-features --locked`, `cargo test --test hashdos --locked` |
| CPU throughput against llama.cpp | measured | `cargo run --release --locked --example engine_profile`, receipts in `bench/cpu/` |
| executed `freq::Table` refines `Apeiron.Table`; full SPIR-V checker; 7B/200k goal | open | |

## What this does not claim

- **Cryptography is unaudited.** `aes`, `chacha`, `sha256`, and `crypto` pass known-answer vectors
  only and are not side-channel resistant (`aes` indexes its S-box by secret bytes). Do not use
  them to protect real data.
- **`unsafe` is mostly unproved.** Only the `src/map/table` window model sits under a proof; SIMD
  and Vulkan FFI bodies sit outside the Lean extraction.
- **The GPU path is single-tenant.** Vulkan buffers run without `robustBufferAccess`, workgroup
  memory is not cleared between dispatches, and Vulkan reads GGUF payloads, not `.nqp` packs.
- **Nothing below the extraction is in the evidence.** `rustc` and LLVM code generation, the kernel
  ABI, Vulkan drivers, and GPU hardware are outside every proof.
- **`i64` is not the real trichotomy.** `Z/2⁶⁴` carries the structural half of the algebra, not the
  real classification ([scalars](book/src/theory/scalars.md)). Table callbacks must be total and
  must not allocate, panic, or diverge.

## Documentation and licence

Read the [book](book/src/introduction.md) for installation, algebra, tables, verification, and
engine internals. See [GOAL.md](GOAL.md), [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md),
[proof/README.md](proof/README.md), and [SECURITY.md](SECURITY.md) for criteria, gates, proofs,
and vulnerability reporting. Licence: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
