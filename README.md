# apeiron

A Rust library for the algebra `R[x]/(x² − σ)`, a growable hash table built on it, and an inference
engine for GGUF models that runs on the CPU and, through Vulkan, on the GPU. The algebra and the
word-level arithmetic are specified and checked in Lean 4.

- **`Cell<R>`**: a pair `(a, b)` over a ring `R`, with multiplication `(a₁a₂ + σb₁b₂, a₁b₂ + b₁a₂)`
  and norm form `a² − σb²`. `σ = −1, 0, 1` gives the elliptic, parabolic and hyperbolic faces.
- **`freq::Table`, `CoordMap`, `FreqMap`**: flat tables with a spill region. Lookup is a fixed number
  of probes; allocation failure and size overflow are refused and change nothing.
- **`engine`**: GGUF loading, Q32.32 fixed-point kernels, AVX2 with a portable fallback, and a
  `gpu` feature that loads Vulkan at run time.
- **`proof/`**: Lean 4 extraction of the Rust word bodies with Aeneas and Charon, and the
  refinement theorems that link them to the canonical definitions in `proof/canonical/`.

The default build has no dependencies; Linux adds `libc` for thread affinity. Rust 1.89 or later.

## Use

```sh
git clone https://github.com/sanguinehost/apeiron
cd apeiron
cargo build --lib
```

```rust
use apeiron::{Cell, Sigma};

let x = Cell::<i64>::of(3, 2);
let y = Cell::<i64>::of(1, 5);
let p = x.mul(Sigma::Hyperbolic, y);
assert_eq!((p.a, p.b), (3 + 2 * 5, 3 * 5 + 2 * 1));
```

```rust
use apeiron::freq::Table;

let mut counts = Table::<u64, i64>::new();
for word in [7u64, 9, 7, 7, 9, 4] {
    counts.insert_with(word, 1, |old, new| old + new);
}
assert_eq!(counts.get(&7), Some(&3));
assert_eq!(counts.len(), 3);
```

The [book](book/src/introduction.md) covers installation, the algebra, the tables, running a model
on the CPU and the GPU, and the verification. Build it with `mdbook build book`; its examples run
under `mdbook test book -L target/debug/deps`.

## Inference engine against llama.cpp

Current figures, re-measured 2026-10-29 (CPU) and 2026-10-30 (GPU) with paired, interleaved rounds. `pp512` is 512 prompt
tokens per second, `tg128` is 128 generated tokens per second. Ratios are apeiron over llama.cpp,
the median of 5 paired rounds with the minimum to maximum round in brackets. Every number below is
in a saved receipt registered in `docs/evidence/ledger.jsonl`; `tools/bench-save.sh verify`
fails when one is not.

| metric | model / device | apeiron | llama.cpp | ratio | receipt |
|---|---|---:|---:|---:|---|
| pp512 | Qwen2.5-0.5B Q4_K_M / RTX 5080 | 111,440 tok/s | 61,823 tok/s (CUDA) | **1.73×** (1.71–1.75) | [`1793316889-20261030-qwen05b-q4km-gpu`](docs/evidence/receipts/bench/1793316889-20261030-qwen05b-q4km-gpu.md) |
| tg128 | Qwen2.5-0.5B Q4_K_M / RTX 5080 | 1,026.7 tok/s | 879.0 tok/s (CUDA) | **1.15×** (1.11–1.15) | [`1793316889-20261030-qwen05b-q4km-gpu`](docs/evidence/receipts/bench/1793316889-20261030-qwen05b-q4km-gpu.md) |
| pp512 | Qwen3.5-0.8B Q8_0 / i9-14900KF | 957.36 tok/s | 520.07 tok/s (CPU) | **1.82×** (1.66–1.97) | [`1793260444-20261029-qwen08b-q8-cpu`](bench/cpu/1793260444-20261029-qwen08b-q8-cpu.md) |
| tg128 | Qwen3.5-0.8B Q8_0 / i9-14900KF | 91.76 tok/s | 50.59 tok/s (CPU) | **1.81×** (1.77–1.85) | [`1793260444-20261029-qwen08b-q8-cpu`](bench/cpu/1793260444-20261029-qwen08b-q8-cpu.md) |
| pp512 | Qwen3.5-0.8B Q4_K_M / i9-14900KF | 676.20 tok/s | 694.28 tok/s (CPU) | 0.97× (0.95–0.98) | [`1793260567-20261029-qwen08b-q4km-cpu`](bench/cpu/1793260567-20261029-qwen08b-q4km-cpu.md) |
| tg128 | Qwen3.5-0.8B Q4_K_M / i9-14900KF | 87.33 tok/s | 58.08 tok/s (CPU) | **1.51×** (1.47–1.54) | [`1793260567-20261029-qwen08b-q4km-cpu`](bench/cpu/1793260567-20261029-qwen08b-q4km-cpu.md) |

- Comparator: llama.cpp `ece963f41` (CUDA rows, `llama-bench -ngl 99`) and `0324696b8` (CPU rows, best
  of 16 and 32 threads). Both are newer than the `60bc8866b` of the earlier figures; the Q8_0 prefill
  ratio fell from 2.24× because llama.cpp's own prefill rose from 416 to 520 tok/s, while apeiron's
  rose from 918 to 957.
- Every round passed the quiet-host gate (CPU: busy cores < 1; GPU: load1 < 2 and device
  utilisation < 10%).
- Q4_K_M prefill on the CPU is behind llama.cpp (0.97×); decode is ahead.
- The receipts were recorded from a working tree with uncommitted paths; each names the source hash.
  First-token agreement has not been run on the Q4_K_M files.
- Reproduce: `engine_profile` for the CPU rows (see `book/src/engine/results.md`) and
  `tools/bench-save.sh gpu <gguf> <name>` for the GPU rows.

### Earlier CPU receipts

More models and prompt lengths, Q8_0, measured earlier at revision `04d206013f70` against llama.cpp
`0324696b8` at its best of 8, 16, 24 and 32 threads. `tg` here is 64 generated tokens. Every round
passed the quiet-host gate (busy cores < 0.05).

| weights | pp tokens | rounds | pp ratio | tg ratio | receipt |
|---|---:|---:|---|---|---|
| Qwen3.5-0.8B | 200 | 5 | 2.17 (2.11–2.28) | 1.76 (1.52–1.77) | [`1790795647-w6e-vs-llama-r3`](bench/cpu/1790795647-w6e-vs-llama-r3.md) |
| Qwen3.5-0.8B | 512 | 5 | 2.45 (2.30–2.79) | 1.68 (1.55–1.79) | [`1790798160-w7-q35r2-pp512`](bench/cpu/1790798160-w7-q35r2-pp512.md) |
| qwen05b | 200 | 5 | 1.57 (1.56–1.64) | 1.47 (1.35–1.54) | [`1790798706-w7-qwen05b-r2-pp200`](bench/cpu/1790798706-w7-qwen05b-r2-pp200.md) |
| qwen05b | 512 | 5 | 1.56 (1.37–1.58) | 1.55 (1.45–1.59) | [`1790797618-w7-qwen05b-pp512`](bench/cpu/1790797618-w7-qwen05b-pp512.md) |
| qwen15b | 200 | 5 | 1.77 (1.74–2.00) | 1.10 (1.08–1.14) | [`1790798386-w7-qwen15b-r5-pp200`](bench/cpu/1790798386-w7-qwen15b-r5-pp200.md) |
| qwen15b | 512 | 3 | 1.91 (1.89–1.96) | 1.19 (1.06–1.21) | [`1790797926-w7-qwen15b-pp512`](bench/cpu/1790797926-w7-qwen15b-pp512.md) |

All six were measured at revision `04d206013f70`, source sha256 `0218e7d911a3…`, with the default `f32`
carrier. Each receipt holds the model and prompt digests, the machine, the compiler and the
per-phase trace. Reproduce one with:

```sh
cargo run --release --locked --example engine_profile -- \
  --model <Qwen3.5-0.8B-Q8_0.gguf> --ids tests/fixtures/qwen35-prompt2.ids \
  --pp 200 --tg 64 --reps 5 \
  --llama <path/to/llama-bench> --llama-threads 8,16,24,32 --max-busy 0.05 --label repro --out <dir>
```

All rows are one machine (the i9-14900KF above); `measured` here means that host and no other. The
GGUF files are not distributed; each receipt names the model by sha256. Only the first row's prompt
(`tests/fixtures/qwen35-prompt2.ids`, sha256 `894a166a…`) is in the repository. The other five
name their prompt ids by sha256 in the receipt and are not reproducible from a clone.

Throughput is accepted only alongside agreement with the comparator on the same model (first
tokens and perplexity); the book's [Results](book/src/engine/results.md) chapter gives the protocol
and the GPU commands (`gpu_decode_bench`, `gpu_ppl`). The hash table comparison against
`hashbrown` is not in these tables: its ratios are not claimed here.

## Quantisation: the open gap

The goal is a 27B-class model in at most 4 GB of resident weights (about 1.18 bits per weight) that
stays within 10% of Q4_K_M: perplexity at most 1.15× and zero-shot task accuracy at least 0.90× of
Q4_K_M's, from the same run. That bar is not met yet.

| metric | model / setting | apeiron | Q4_K_M | ratio | bar |
|---|---|---:|---:|---:|---:|
| wikitext-2 perplexity | Qwen3-0.6B-Base, 1.18 b/w, 1024-step KD | 24.5751 | 13.7633 | 1.785× | 1.15× |
| mean 0-shot accuracy (boolq, piqa, hellaswag, winogrande, arc_easy, arc_challenge) | same | 0.4194 | 0.5556 | 0.755× | 0.90× |
| relative Hessian distortion, all 28 blocks | same, Fold E8 lattice codebook vs NanoQuant's sign-rank factorisation | 0.0149 | 0.0805 (NanoQuant) | 5.39× lower | n/a |

- NanoQuant as shipped, on the same model and harness: perplexity 24.3317 (1.768×), accuracy 0.4235
  (0.762×). At equal bits the port is 1.010× the reference's perplexity and 0.99× its accuracy, with
  5.39× lower layer distortion; both are short of the bar.
- Qwen3.8-27B (48 DeltaNet and 16 attention layers) has been run only as its Q4_K_M file (15.3 GiB):
  prompt processing 586.4 tok/s and generation 4.85–5.90 tok/s on an RTX 5080 with 38 layers
  offloaded plus CPU, prefix-cache hit 98.3%. No 1.18 b/w 27B has been built; its 3.753 GiB is the
  arithmetic N·1.18/8, not a measurement. Quantisation quality is measured only on Qwen3-0.6B-Base.
- Figures are copied from the recorded measurements, not re-run for this release.

## What is proved, tested and open

| statement | grade | gate |
|---|---|---|
| `Cell<i64>` multiply and norm form refine the canonical ring operations for every `σ` and input | proved | `word_mul_denote`, `word_norm_denote` in `proof/corpus/WordRefinement.lean` |
| the Lean axiom cone of every headline is within `propext`, `Classical.choice`, `Quot.sound` | proved | `proof/corpus/AxiomAudit.lean`, run by `lake build` |
| `Cell`, the tables, the engine and the SPIR-V fragment checker behave as documented on their boundary cases | tested | `cargo test --workspace --all-features --locked` |
| the SPIR-V fragment header check | proved | `Spirv.checkFragment_header` |
| the HashDoS bound on the tables | tested | `cargo test --test hashdos --locked` |
| speed against llama.cpp | measured | `cargo run --release --locked --example engine_profile`, receipts in `bench/cpu/` |
| the executed `freq::Table` body refines the canonical `Apeiron.Table` | open | |
| the whole SPIR-V fragment checker | open | |

Two extraction facts bound the proofs: the Lean files are generated from the Rust source by Aeneas
and Charon at pinned revisions (`proof/README.md`), and `proof/regen.sh --check` fails if the
committed extraction differs from a fresh one; `cd proof/corpus && lake build` checks the theorems.

## What this does not claim

- **Cryptography is unaudited.** `aes`, `chacha`, `sha256` and `crypto` are tested against
  known-answer vectors only and are not side-channel resistant (`aes` indexes its S-box by secret
  bytes). Do not use them to protect real data.
- **`unsafe` is mostly unproved.** Count it with
  `rg -n '\bunsafe\b' src --glob '*.rs' | rg -v ':\s*(//|\*)' | wc -l`; `SAFETY:` comments are
  counted with `rg SAFETY src --glob '*.rs' | wc -l`. Only the `src/table` window model is under a proof.
- **The GPU path is single-tenant.** Vulkan buffers run without `robustBufferAccess` and workgroup
  memory is not cleared between dispatches.
- **No claim below the extraction.** rustc and LLVM code generation, the kernel ABI, Vulkan drivers
  and GPU hardware are outside the evidence.
- **`i64` is not the real trichotomy.** `Z/2⁶⁴` carries the structural half of the algebra
  (commutativity, associativity, multiplicative norm) and not the classification; the book's
  [scalars](book/src/theory/scalars.md) chapter states where it stops.
- **Callbacks.** The tables assume `Eq`, `Clone` and merge callbacks that are total and stable; a
  callback that allocates, panics or diverges is outside the statements.

Security reports: [SECURITY.md](SECURITY.md). Licence: Apache-2.0 ([LICENSE](LICENSE),
[NOTICE](NOTICE)).
