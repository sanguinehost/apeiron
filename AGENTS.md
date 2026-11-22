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

## One path

The CPU engine has one projection path, not one per quantisation format. `matmat_q8act` is the
entry. It walks the weight rows once, decodes each 32-lane group through a carrier dial, and folds
that group against every activation column in the call. Decode is `k = 1`; prefill is `k > 1`; both
take the same loop. A format is a carrier value, not a kernel.

| dial | values | changes |
|---|---|---|
| carrier | `F16`, `Q8_0`, `Q5_0`, `Q4_K`, `Q6_K`, ternary, packed | block geometry and code unpack |
| columns `k` | `1..4096` | activation columns folded per weight read |
| dot core | AVX2, AVX-VNNI, portable | the 32-lane integer dot |
| accumulator | power-of-two `i64` lanes, checked `i128` | range proof and refusal route |

`ternary` is an exploratory activation dial opened by the agent in W2, not a requested path, and it
is refuted on quality: 77,741,390 PPL against the same-binary q8act control's 72.988013. It is not
a product route, and no claim rests on it.

Adding a bit depth adds a carrier row. It does not add a projection function, a second row loop, a
`gemm_*` or `*_row_*` sibling, or an `if ggml_type == ...` beside an existing kernel. A carrier's
block core may use a different instruction; that is a dot dial on the same loop.

The one exception is the correctness oracle: `matvec_fixed` on the dequantised weights, or the
per-column `matvec_q8act`. It exists to validate the fast path, never as a production route. An
acceptance run keeps it to at most about ten tokens. A verification campaign too slow for this
laptop runs on a rented host (vast.ai or GCP), not on the laptop and not as a second path.

Debt, not precedent: the pooled `Q8_0` VNNI entry and the pooled `Q4_K`/`Q6_K` entries still sit
beside the shared loop. They fold into it next. New work does not copy their shape.

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
- Independent angles run in parallel, not in one lane. A task with two or three disjoint parts (one
  carrier per agent, one criterion per agent, one receipt per agent) spawns one subagent per part
  in one batch; each gets its own files, its own artifact and its own gate command, and returns
  both. The main lane integrates and runs the shared gate. Serialising independent reads or
  measurements is a defect, not caution.
- Pick the default and state it. Ask only when the answer changes what is built and the code,
  the docs and the request do not decide it.
- Report in the order result, evidence, what remains. No preamble, no restating the request. A
  result that is short of the target says so in that sentence, with the factor: "8.1–12.2 tok/s
  against 24 at 200k, 2.0–3.0× short", not "the ceiling is 26.96".
- Still confirm first, every time: history rewrites (`reset`, forced push, rebase of shared or
  diverged branches), deleting files with work you did not make, and publishing outside the
  repository. A diverged branch is reported with the counts and the options.
- The derivation gate (`.claude/hooks/derivation_gate.py`, `.intent/`, `.claude/settings*.json`)
  is the owner's control. An agent does not edit, stub or route around it; on a gate failure it
  reports the cause and the one-line fix for the owner to run with `!`.

## Dogfood

A harness defect met while doing the work — a tool that hangs, a command that kills its own shell,
a job record that cannot say what it ran — is filed where the agent cannot rewrite it. The only
write under `.intent/` an agent makes is this append-only log; the derivation gate's own files stay
the owner's.

```sh
intent friction add --root ~/Workspace/apeiron-nq --class tool --evidence <path>:<line> \
  --observed '<the exact failure, the command and what it returned>'
intent friction screen <id> --component … --edit … --symptom … --falsifier … --metric … \
  --cost … --guard … --prune …
```

`add` has no bar. `screen` is the bar: it names the missing field instead of accepting a candidate
that cannot be evaluated. A candidate names a mechanism in `intent`, not in this repository; the
receipt carries the ids it filed.

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

On this laptop run clippy with `-j 1` (the derived cap terminates `-j 2`). A mixed-feature
`target/` makes `mdbook test` fail with `E0464` (several `apeiron` rlibs); build once into a clean
`CARGO_TARGET_DIR` with `cargo build --lib --all-features --locked` and point `-L` at its
`debug/deps`. Neither is a code defect; a receipt says which directory the result came from.

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

**Performance gate** (any change that claims a speedup): the phase split above on the unmodified
tree, with `--target` set to the phase the change edits. It needs a model and a pack, so it is not
part of the contributor gate; a receipt that claims a speedup without it is refused. The gate is
executable, not documentary: exit 1 is the refusal.

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

## Ceilings

A speed target or claim is checked against the device ceiling before it is called reachable or
unreachable; the ceiling is computed, not asserted. Decode is bounded twice:

```text
tok/s ≤ min( bandwidth ÷ bytes-per-token , issue-rate ÷ instructions-per-weight )
```

`bytes-per-token` counts every tensor read on every token, the output head included; an embedding
lookup is not read; the KV read is part of it. `issue-rate` is physical cores × clock × IPC, and
`instructions-per-weight` is counted from the kernel or its assembly. A receipt states both bounds,
with the spec peak and the measured bandwidth as separate rows, and names the binding one. A target
is quoted with its margin ("30.4 tok/s of 42.7 GB/s, 98.8% of spec peak"), never with an adjective.
A ceiling computed before the measurement does not become optional because a benchmark looks good.

A floor is not a target. `GOAL.md` T1 is 30 tok/s at 2k and 24 at 200k; every ceiling is quoted
beside the target and the gap to it, not beside the floor alone. A gap inside the run's spread is not
a verdict: 19.9 against a floor of 20 is zero margin, stated as the utilization it implies, never as
a refusal. A configuration's ceiling is never the model's ceiling — the pack geometry belongs in the
same sentence as the number: the same sign dot spans 19.91 with the `Q6_K` head and 26.96 with the
fitted head, and the `Q4_K_M` pack is 6.40 under a different carrier.

Decode is bounded a third time, by the pipeline: a phase that holds share `s` of the token cannot
exceed `1/(1 − s)`, whatever its kernel does. See [The dominant term](#the-dominant-term).

## The dominant term

A token is a pipeline. A speed change can only move the phase that owns the time, so the split is
measured before the target is chosen, not after the number disappoints. The ceiling a target
inherits is `1/(1 − share)`: a phase at 18% cannot exceed 1.22× however fast its kernel becomes.

```sh
cargo run --release --locked --example phase_split -- \
  --model <gguf> [--pack <nqp>] [--target <phase>]
```

`phase_split` takes `APEIRON_TRACE` over a decode run, prints each leaf phase's ms/token and share
of wall, and names the dominant term (a leaf, the head, or the pass's untraced remainder). With
`--target` it exits 1 unless that term is the largest; the refusal prints both shares and both
ceilings. `untraced` is a legal target and means "instrument the pass first". `pool.wait` is an
overlay: a large value means the work was cut too fine to schedule, not that the kernel is slow.

The measured case: the 0.6B v3 fixture put 82% of a decode token in the tied `F16` head
(`q8act_plan(F16) = Reference`: one scalar dequantization and one checked `i128` dot per weight) and
18% in all 196 NQ projections. Any sign-dot work was capped at 1.22×. Moving the head to a `Q8_0`
`output.weight` put it on the fused block reader and took 3.44 → 21.42 tok/s at width 4
(`bench/cpu/1794813322-q8-head-fixture.md`). The reader, not the arithmetic, owned the token.

A performance receipt quotes the split it measured, the target's share, its `1/share` ceiling, and
the post-change split. A receipt whose target was not the dominant term is refused; the fix is to
change the target, not to explain the ceiling. The same rule applies inside a phase: time the
sub-steps (weight read, decode, dot, activation pack, pool dispatch) before choosing one.

## Unsafe

An `unsafe` block is admitted only with a safe twin in the crate: the same input domain, the same
arithmetic and the same bounds, written without `unsafe`. The twin is the extracted body, and a
`proved` theorem over it is the arithmetic claim. The intrinsic body is `tested` equal to the twin
over the boundary lengths, never `proved`: Aeneas does not model `core::arch`, inline assembly or
raw-pointer arithmetic, so nothing that reaches those operations can appear in an extracted body.
A block whose logic has no twin, or whose twin has no equality test, is not admissible; rewrite it
in safe Rust or remove it. Adding a twin to `proof/extract/lib.rs` is part of adding the `unsafe`.

The safe twin is not a second production route. It is the specification the kernel is checked
against, the same way `matvec_q8act` is the oracle for the pooled loop.

The surface that predates this rule is debt, listed in `proof/README.md`; it is not precedent.
`sign_dot_avx2` has its safe twin `sign_dot_i64` and the equality test
`the_vector_source_equals_the_scalar_source`; the `Q4_K` packed lane `q4_k_row_lanes_avx2` has only
the oracle equality `pooled_q4_k_matches_per_column`, so it is `tested` debt: its arithmetic is
still duplicated rather than expressed by a safe twin.

## The product line

The target is one product: a 7B pack at **≤1.00 b/w average** — NanoQuant's published 1.00-bit
point, with 0.80 b/w as the sub-1-bit point — **≤1.5× Q4_K_M PPL** on the same head and token
stream, and **≥20 tok/s** decode at 2k as the working target (`GOAL.md` T1 is 30 tok/s at 2k and 24
at 200k). Each number is checked by an artifact, not inferred from a smaller model or a
single layer.

- **Compression.** The pack header is the rate. A per-layer allocation states every layer's rate and
  the weighted average; the average is the criterion. The dense sign-factor cap for a linear `m × n`
  at rank `m` is `(m+n)/n` b/w; a plan above the cap names the residual mechanism, and `nq_fit`
  writes no residual today. For 3584×18944 the cap is 1.19 b/w, so a 3.0 b/w layer-0 plan is not
  executable by `nq_fit`.
- **Quality.** A PPL receipt names the model, the pack, the token count, the context, the head and
  the teacher. A partial pack is labeled partial; a layer control is not a model result. The 0.6B
  rows are lower bounds, not the product. The minimum line is 1.5× Q4; above it the lever is the
  rate or the layer allocation, not a re-fit of the same artifact. **The preconditioner is part of
  the result.** The reference path uses the activation Gram diagonal (`nq_fit --diag-dir`,
  `g{l}.{tag}.{gram}.diag.i64`); the example's column-RMS fallback is not a quality path — at
  0.977M ppm it scores 1,225,029 PPL where the same rate on the reference path scores 57.045. A
  control and its treatment use the same preconditioner; changing it is not a tree comparison.
- **Rate.** A 20 tok/s claim names the 7B, the phase split, the target phase's share, and the
  `1/(1−s)` ceiling. The CPU measured points are q8 11.0 and int4 14–18; neither reaches 20, so
  20 tok/s on this CPU is refused until a carrier appears whose quality screen and kernel both clear
  it. An agent-opened exploratory ternary activation reached 63 tok/s but scored 77,741,390 PPL
  against the q8 control's 72.988013; it is not a candidate. The GPU path is a separate claim with
  its own receipt.

**Fit budget.** A fit campaign multiplies the measured per-linear iteration by the linear count and
by T before it starts. The 7B `down` iteration is 94 s on the AVX-VNNI box; one layer's seven kinds
are 324 s; 196 linears are 2.52 h/iteration; T=200 is 504 h, not 18 h. A receipt that quotes a
per-layer figure as a model figure is wrong by the layer count. `tools/fit-budget.py` computes the
wall and exits 1 over the cap:

```sh
python3 tools/fit-budget.py --iteration-seconds 9072 --iters 200 --cap-hours 6
# 504.00 h > 6.00 h: refuse (exit 1)
python3 tools/fit-budget.py --iteration-seconds 9072 --iters 2 --cap-hours 6
# 5.04 h <= 6.00 h: admit (exit 0)
```

The cap selects the host, not T. T is the campaign's quality parameter; when the projected wall
exceeds the cap, rent a faster host or change the algorithm (the GPU limb path), never lower T to
fit the cap. The 7B T=2 campaign did the reverse: it lowered T to fit a 6 h CPU cap while a
bit-exact GPU path that fits the same `down` shape in 2.4 s per 100 iterations sat in the tree.

**Fit preflight.** A full-model campaign fits layer 0 at the campaign's T on the device path and
scores an 8-token PPL on that one-layer pack before spending the full wall. A `norm: OutOfRange`, a
nonzero exit, an `admm_loss` outside the reference band, or a CPU-only build stops it. A layer-0
fit at a smaller T is a shape test, not a preflight: the 7B T=2 campaign ran **6.63 h** after layer
0 had printed v/o losses of **9,094/10,546**, and the one-layer smoke refuses in about a second. A
fit whose forward cannot run is not an upper bound. The campaign runs `nq_fit --require-device`;
the flag exits 1 when the binary was built without `--features gpu` or no cooperative-matrix device
loads.

## Context is two tiers

Long context is bounded hot state plus a cold tier read by selection, not a larger dense cache. The
design is settled and the 1M path is only through the second tier; the port is the open half.

- **Hot.** The dense `KvState` ring holds the most recent positions and is read every token. It is
  bounded by the position budget, not by the context length: a few thousand positions at 4 bit. At
  1,827 positions the 0.6B reads 229,376 B/position of F32 KV and `attn.core` owns 49.8% of the
  token — the replacement targets attention, not the head (`bench/cpu/1795170568-nq06-context-split.md`).
- **Cold.** Sealed block rows live outside the dense plane, one record per `(layer, kv-head, block)`,
  in the canonical flat/spill `Table` (`src/engine/step/cold_store.rs`), and are gathered by block. A
  block the store does not hold is a refusal, never a read of the plane the rows came from. The
  record format exists and is re-exported; it is **not** wired into the step path and has no test
  yet. PLAN S5 is that wiring plus the file tier, S8 repeats it on the 8B. Do not report C1 from the
  component's existence.
- **Selection.** Past the hot ring a token attends to the blocks selection keeps plus the ring. The
  hot path is O(1) or O(log n); a cold gather is O(blocks kept) and the store payload is O(tokens).
  The 512 MB block cache, not a larger ring, is what makes 1M reachable at the byte budget
  (PLAN §1, S5, S8).

Two cold objects exist in the lineage and are not interchangeable. `FinderRecall.ColdStore`
(`AC2-MEM`, 9 B/token) stores token ids and a hash chain and recalls a token by exact recompute; it
holds no KV vectors and cannot serve a row gather. `ColdStore` here holds Q32.32 rows and returns
them bit-for-bit at 8 B/word, so its payload is O(tokens) and is **not** 9 B/token. The long-context
solution already found is the pair: recall/selection chooses positions, the block store reads the
chosen rows. It was built in the `fold`/bedrock line (`ColdDecode`, `ColdTier`, `FinderRecall`,
cited in `cold_store.rs`), which is not in this repository; this repository has the record format and
the plan, not the serving path. Do not re-derive either, and do not quote one at the other's byte
count.

Accounting: C1's 1 GiB at 200k is **1.4976 bits/value** over the 7B geometry, so the cold carrier is
sign or 1.25 bit; a 4-bit cold store is 3.23 GB and is a disk figure. Report resident (hot ring,
index, block cache) and cold payload/disk separately, never summed into a flat claim. The
file-backed plane (`KvState::with_file`) is M1's mechanism, not C1's. Open: C2 quality at 200k —
a sign-only carrier changes every score, and the certificate proves which blocks contribute exactly
zero, not that a sign-only block is exact — plus the 200k/1M NIAH gate
(`bench/notes/context-budget.md`, `GOAL.md` C1/C2).

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
- A speed change names the traced phase it targets and that phase's share on the unmodified tree;
  `examples/phase_split --target` exits 0 there. A change to a phase that is not the dominant term is
  refused, whatever its local ratio.
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
