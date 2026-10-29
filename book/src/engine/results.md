# Results

This page gives the commands that produce throughput numbers and the protocol they follow. It
carries no table of ratios, because a number from a document is a hypothesis until the command is
re-run. The ratios for a release are in the repository README with the identity they were measured
at.

## Protocol

- A number is a measurement when a named command ran and its output was read.
- Report the mean over at least three runs with the observed spread. A single run is a sample.
- A change inside the spread seen on unchanged code is not a change.
- Name the machine, the metric and the interval. Report before and after.
- Interleave the arms (apeiron and the comparator) round by round, so drift in host load hits both.
- Gate on a quiet host. `engine_profile` measures busy cores before each round and rejects a round
  above `--max-busy`.
- Force a rebuild (`touch src/lib.rs`) or use a private target directory before a timed run, so the
  binary matches the source being quoted.

## CPU: prefill and decode against llama.cpp

```sh
cargo run --release --locked --example engine_profile -- \
  --model <model.gguf> --ids <prompt.ids> \
  --pp 200 --tg 64 --reps 5 \
  --llama <path/to/llama-bench> --llama-threads 8,16,24,32 \
  --max-busy 0.05 --label <name> --out <dir>
```

`pp` is prompt tokens per second and `tg` is generated tokens per second. The comparator runs at
each thread count in `--llama-threads` and the best is used. Each run writes a receipt with the
model digest, the source digest, the machine, the compiler version, the comparator revision and the
per-round ratios. A prompt id file for a 0.8B model is in `tests/fixtures/`.

## Saving a measurement

A figure in the README or `perf-predictions.md` is a row in `docs/evidence/figures.json`: a receipt
path, a key in it, and the text as quoted. One tool owns the path from run to document:

```sh
tools/bench-save.sh gpu <model.gguf> <name>   # interleaved GPU rounds, writes and registers a receipt
tools/bench-save.sh register bench/cpu/<run>.json bench/cpu/<run>.md   # index an engine_profile receipt
tools/bench-save.sh verify
```

Every writer indexes the receipt in `docs/evidence/ledger.jsonl` with its sha256 and copies it to
`$APEIRON_RECEIPT_MIRROR` (default `~/Backups/apeiron-receipts`). `verify` fails on an unregistered
or edited receipt, a receipt missing from the mirror, and a quoted figure that differs from its
receipt or is absent from the document. Commit the receipt in the same change as the figure.

## GPU decode

```sh
cargo run --release --locked --features gpu --example gpu_decode_bench -- <model.gguf> --gen 128 --reps 5 --carrier q4
cargo run --release --locked --features gpu --example gpu_decode_bench -- <model.gguf> --prefill 512 --reps 5 --carrier q4
```

`gpu_decode_bench` reports `tg` and `pp` from device timing; `tools/bench-save.sh gpu` runs it
against `llama-bench` and saves the receipt. `gpu_ppl` was deleted with the examples in `e0d991c5`, so
no GPU-path perplexity check against llama.cpp exists at this revision.

## Hash tables

```sh
cargo bench --locked --bench map
cargo bench --locked --bench count
```

These compare the tables with `hashbrown` under `FxHasher` and XXH3. A win means every workload
column, with the ratio per column reported, losses as plainly as wins.

## Correctness before speed

A throughput number is accepted only with agreement against the comparator on the same model: the
first generated tokens match and perplexity matches. The [evidence](../verification/evidence.md)
chapter lists the grade each claim carries. Speed is `measured`, never `proved`.
