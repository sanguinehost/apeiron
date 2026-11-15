# Working on apeiron

Instructions for contributors and for coding agents. Human contributors also read
[CONTRIBUTING.md](CONTRIBUTING.md).

apeiron is `R[x]/(x² − σ)`, its `Cell<i64>` word kernel, the flat/spill tables, and an inference
engine. The Rust is extracted to Lean with Aeneas and Charon; the Lean theorems link the extracted
bodies to the canonical definitions in `proof/canonical/`. The [book](book/src/introduction.md) is
the reader's documentation.

## The goal

Run a 7B-class model on the 6.6 GiB laptop at about 30 tok/s on the CPU with a 200,000-token
context, leaving at least 2 GiB available for four or five coding-agent sessions. Serve it through
`serve/` into `../intent`, replace DeepSeek usage to a measured degree, and ship the quantiser, the
context method and the Vulkan/SPIR-V kernels as open source, with the kernels formally verified
through Aeneas and Lean.

`GOAL.md` is the canonical statement of work, the acceptance criteria and the current status. The
criteria are Q1 rate, Q2 quality, M1 memory, T1 decode, T2 turn latency, C1 context state, C2
context quality, I1 Intent integration, I2 DeepSeek replacement, V1–V4 Vulkan/SPIR-V/Lean, O1 open
source, G1–G3 gates and receipts, and S1 load safety.

The card in `../sanguine.sh` is an artifact of the goal, not the goal. A pack, a benchmark row, a
12-row sample, a 0.6B result or a single speedup is an artifact. A receipt advances a criterion; it
does not close it. Every receipt names the criterion it advances and the part it leaves open.

## Resource discipline

Every heavy command runs through [`~/Workspace/tools/agent-guard`](/home/lucas/Workspace/tools/agent-guard).
The policy is [`~/Workspace/RESOURCE_GUARD.md`](/home/lucas/Workspace/RESOURCE_GUARD.md). One heavy
job runs at a time across the workspace; a second waits for the first. The guard waits for every
other `cargo`, `rustc`, `lean`, `lake` or C compiler process, refuses below 1 GiB available, and caps memory at
`clamp(MemAvailable - 1 GiB, 500 MiB, 2 GiB)`: 3 GiB free permits at most 2 GiB, 1 GiB free permits
500 MiB. Swap is capped at zero.

Use `agent-guard run -- <command>` for a build, test or gate. Do not run `cargo`, `bun`, `npm`,
`mix`, `make`, `just` or `docker build` outside it. In an OpenCode session the shell shims apply the
same rule; do not call a real binary by absolute path or reset `PATH`. A command that spawns an
uncapped build is not admitted and its result is not evidence. On this 6.6 GiB laptop a hardcoded
`MemoryMax=8G` or `28G` is a bug; use the derived cap.

## Orient

1. `GOAL.md`, then `book/src/introduction.md`, then the chapter for the area you change.
2. `proof/README.md`: extraction, the bridge, exact commands, what is not established.
3. The source you will change, read in full. A summary of a file is not the file.

Where a document disagrees with the source or a gate, the source and the gate win. Fix the
document.

## Initiative

For agents. The task as stated is the authorisation; do not ask for it again.

- Act on a clear request. Do the whole job, in order, without a confirmation between steps.
- Read-only commands (`git fetch`, `git status`, `git log`, `cargo test`, `ls`, `grep`) run without
  asking, in any repository under `~/Workspace`.
- Local, reversible changes inside the task run without asking: fast-forward `git pull --ff-only`,
  add a remote, create a file the task needs. Say what changed in the report.
- A blocker inside the task gets diagnosed and fixed in the same turn: read the failing script or
  log, find the cause, apply the fix. Report the cause and the fix, not the question.
- Scope the work to the target. Skip unrelated clones and large third-party trees; a command
  that runs more than a minute goes to the background while the next step proceeds.
- Pick the default and state it. Ask only when the answer changes what is built and the code,
  the docs and the request do not decide it.
- Report in the order result, evidence, what remains. No preamble, no restating the request.
- Still confirm first, every time: history rewrites (`reset`, forced push, rebase of shared or
  diverged branches), deleting files with work you did not make, and publishing outside the
  repository. A diverged branch is reported with the counts and the options.
- The derivation gate (`.claude/hooks/derivation_gate.py`, `.intent/`, `.claude/settings*.json`)
  is the owner's control. An agent does not edit, stub or route around it; on a gate failure it
  reports the cause and the one-line fix for the owner to run with `!`.

## Gates

Two tiers. A contributor runs the first before proposing a change; CI runs both.

**Contributor gate** (Rust and documentation; minutes, no Lean toolchain):

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
python3 tools/check-claims.py
python3 tools/book-check.py
mdbook build book && mdbook test book -L target/debug/deps
```

**Maintainer gate** (a change under `proof/` or to a Rust file the extraction includes; needs the
pinned Aeneas, Charon and Lean, and a Lean build of several GB):

```sh
proof/bootstrap.sh              # build the pinned Aeneas and Charon (once)
proof/regen.sh --check          # fail if the committed extraction differs from the source
cd proof/corpus && lake build   # the bridge, including the axiom audit
```

Run Lean from the package directory, never `lake -d` from elsewhere: the toolchain is chosen by the
package's `lean-toolchain`. Lean builds can take several GB. Run them through
`/home/lucas/Workspace/tools/agent-guard run -- lake build` and use `LEAN_NUM_THREADS=1`. If the
build needs more than the derived cap, it does not run on this laptop; use the larger box. Do not
raise the cap here. `set_option maxHeartbeats 0` and unbounded kernel reduction are not accepted in
a committed proof.

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

- The change advances a named criterion in `GOAL.md`. The receipt names the criterion id, the grade
  and the scope, including what the result does not cover. A narrower claim says it is narrower.
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
