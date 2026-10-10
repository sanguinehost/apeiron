# Working on apeiron

Instructions for contributors and for coding agents. Human contributors also read
[CONTRIBUTING.md](CONTRIBUTING.md).

apeiron is `R[x]/(x² − σ)`, its `Cell<i64>` word kernel, the flat/spill tables, and an inference
engine. The Rust is extracted to Lean with Aeneas and Charon; the Lean theorems link the extracted
bodies to the canonical definitions in `proof/canonical/`. The [book](book/src/introduction.md) is
the reader's documentation.

## Orient

1. `book/src/introduction.md`, then the chapter for the area you change.
2. `proof/README.md`: extraction, the bridge, exact commands, what is not established.
3. The source you will change, read in full. A summary of a file is not the file.

Where a document disagrees with the source or a gate, the source and the gate win. Fix the
document.

## Gates

Run these before a change is proposed. All are offline once the toolchains are installed.

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
python3 tools/check-claims.py
python3 tools/book-check.py
mdbook build book && mdbook test book -L target/debug/deps
```

Proof work adds:

```sh
proof/bootstrap.sh              # build the pinned Aeneas and Charon (once)
proof/regen.sh --check          # fail if the committed extraction differs from the source
cd proof/corpus && lake build   # the bridge, including the axiom audit
```

Run Lean from the package directory, never `lake -d` from elsewhere: the toolchain is chosen by the
package's `lean-toolchain`. Lean builds take several GB; bound memory (for example
`systemd-run --user --scope -p MemoryMax=28G -p MemorySwapMax=0`) and use `LEAN_NUM_THREADS=1`.
`set_option maxHeartbeats 0` and unbounded kernel reduction are not accepted in a committed proof.

## Evidence grades

Every stated result carries one grade, and the grade names the artifact behind it.

| grade | meaning |
|---|---|
| `proved` | a Lean kernel check passes for a named theorem; for Rust, the extraction is from the real source and a refinement theorem links the body to the canonical definition, with every axiom within `propext`, `Classical.choice`, `Quot.sound` |
| `tested` | a named `cargo test` runs the path and its boundary cases |
| `measured` | a named command ran on a named machine and its output was read |
| `model` or `anchor` | a mapping or a design assumption; true as a mapping, not evidence of correctness |

A citation to a theorem, a paper or another repository is a pointer. It is `proved` only when the
theorem exists and the link in this repository is checked. `python3 tools/check-claims.py` enforces
that a graded claim names an artifact that resolves.

## Definition of done

- The change is on the path the crate builds and runs. No stub, parallel copy or dead code.
- Partial operations state their domain: total, `Option`, `Result`, checked or panicking. No silent
  domain reduction, no infallible signature over an operation that can lose bits.
- A Rust claim in Lean is made over the real source: the extraction includes it by `#[path]` and
  calls the real body. A wrapper may fix a type; it does not restate the formula.
- A mutation of the real source in a throwaway copy turns the proof or test red, and restoring it
  turns it green. A check that cannot fail carries no weight.
- A performance claim gives the command, machine, metric and interval, and is the mean over at least
  three runs with the observed spread. A change inside the spread seen on unchanged code is not a
  change. Rebuild before a timed run (`touch src/lib.rs`) so the binary matches the source.
- The change states what it does not cover: rustc and LLVM code generation, the kernel ABI, Vulkan
  drivers and the GPU, and the `unsafe` container.
- A function whose cost is not O(1) carries a comment giving its time and space, the unit they are
  counted in and the recurrence or loop that derives them.

## Writing

- State mechanism and evidence. A table does not "want", a map does not "refuse".
- Use the standard name for a mathematical object, with attribution for borrowed ones: norm form
  (Gauss 1801), dual numbers (Clifford 1873), split-complex numbers (Cockle 1848).
- Present tense for what is. No dates in prose; git has them.
- Fewest words that keep every path, hash, identifier and number. Tables and bullets over paragraphs.
- A comment carries a reason the code cannot state. It does not restate the declaration.

## Boundaries

- Do not run `git stash`, `reset`, `checkout`, `restore` or `clean` over a tree with work you did
  not make.
- A change touches the files its task names. If it needs another file, sequence it separately.
- Build products (`target/`, `proof/.lake/`, `proof/corpus/.lake/`) are not tracked and are not
  shared between incompatible toolchains.
- No network listener, server or client in `src/` or `examples/`. A change that adds one is
  security-relevant; see `SECURITY.md`.
- Vendored third-party code is limited to benchmark and comparison fixtures.
