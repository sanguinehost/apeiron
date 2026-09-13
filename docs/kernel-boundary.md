# The boundary that "zero dependencies" stops at

**2026-09-11 — context for the hashtrinity / masquerade work.**

This note exists because "zero non-optional runtime dependencies, full stop" is a claim
about *userspace* dependencies, and it is easy to read as a claim about the whole
machine. It is not. Below the Rust crates there is a kernel you do not own, and below
that, hardware. `hashtrinity`'s own reactor already knows this — `MASQUERADE.md:145`
names the inline-`asm!` `syscall` its one true cut. This document is the general form of
that cut: which layers a sovereign stack actually controls, and which ones it calls into.

---

## 1. Userspace and kernel are two privilege levels, not two folders

On x86-64 the CPU has four protection levels, called **rings**: ring 0 (most
privileged) through ring 3 (least). Linux uses two of them.

| | userspace | kernel |
|---|---|---|
| ring | 3 | 0 |
| can execute `syscall` to cross in | yes | n/a |
| can touch I/O ports, page tables, interrupt descriptors | no | yes |
| can map physical memory, program DMA, set timers | no | yes |
| scheduled by | the kernel | itself / interrupts |

The crossing is a **gate**, not a function call. `syscall` saves the user context,
loads the kernel stack and entry point from registers the kernel configured at boot, and
switches ring 3 → ring 0. There is no userspace instruction that performs I/O, allocates
a page, or reads a clock at ring 3; the kernel is the only thing that talks to hardware.

This is why **a userspace runtime can never be "dependency-free" in the absolute.** It
can have no *crate* dependencies; it still has the kernel ABI as its target. A Rust
program with an empty `[dependencies]` and a hundred lines of `std` still calls
`read`/`write`/`mmap`/`epoll_wait`/`io_uring_enter`. `hashtrinity`'s reactor goes one
step further and issues the `syscall` instruction directly via `asm!`, with `libc`
removed — but that is *removing a wrapper*, not *removing the boundary*. The syscall
numbers, calling convention and struct layouts are now hand-maintained facts about
Linux, x86-64, and the current kernel ABI. That is a real (and reasonable) trade; it is
not sovereignty over the kernel.

**The correct sentence is:** zero non-optional *userspace* runtime dependencies, and the
kernel ABI is the floor.

---

## 2. The three different things called "ring"

The word does a lot of work in this stack and the three senses are unrelated. Worth
keeping apart:

1. **Protection rings** (above) — CPU privilege levels, ring 0 vs ring 3. Hardware.
2. **io_uring's submission and completion rings** — two shared-memory circular buffers
   between userspace and the kernel. Software/ABI.
3. **Algebraic rings** — `Z/2^64`, `R`, `R[x]/(x²−σ)`. The two-slot algebra apeiron is
   built from. Pure mathematics, no kernel in it at all.

The masquerade/transport work lives in (1) and (2). apeiron lives in (3). They meet only
where a program stores its state in a table and does I/O with the result.

---

## 3. io_uring, concretely

`io_uring` is a completion-based I/O interface. The kernel and the process share a pair
of lock-free ring buffers, mapped with `mmap`:

```
        userspace                                   kernel
   ┌──────────────────────┐                  ┌──────────────────────┐
   │  SQ ring (submission)│  ---- mmap ----> │  consumes SQEs       │
   │  [Sqe][Sqe][Sqe]…    │  tail →, head ←  │                      │
   ├──────────────────────┤                  ├──────────────────────┤
   │  CQ ring (completion)│  <--- mmap ----  │  posts CQEs          │
   │  [Cqe][Cqe][Cqe]…    │  head →, tail ←  │                      │
   └──────────────────────┘                  └──────────────────────┘
```

- **SQE** (submission queue entry): "please do this op" — opcode, fd, buffer, offset.
  Userspace writes an SQE at the SQ tail, then **publishes** it.
- **CQE** (completion queue entry): "that op finished, here is the result." The kernel
  writes it at the CQ tail; userspace consumes from the CQ head.
- The rings are **shared memory**. Ordering is not free: publishing the SQ tail is a
  release store, reading the CQ tail is an acquire load, and both sides must not read a
  slot the other has not finished writing. This is why any implementation carries a
  memory-ordering argument, not just an index argument.
- **`io_uring_enter`** is the syscall that submits SQEs and/or waits for completions.
- **`SQPOLL`** lets the kernel poll the SQ ring itself, so the common path issues *no*
  syscall per submission — at the cost of a busy kernel thread. That is the optimisation
  `hashtrinity`'s `IoUringDriver::new_sqpoll` is reaching for.
- **Provided buffers / multishot** let the kernel pick up receive buffers the application
  registered in advance and keep one recv live across many arrivals, closing the
  small-chunk gap that readiness APIs otherwise pay a syscall per chunk for.

The contrast with **epoll** is the important one:

- **epoll is readiness-based**: the kernel tells you an fd is *ready*; you then call
  `read`/`write` yourself, and that call can still block or return `EAGAIN`.
- **io_uring is completion-based**: you describe the whole operation up front; the kernel
  reports the *result*.

`mio` is an epoll wrapper; `tokio` is a scheduler and reactor built on readiness. So
"reimplement `mio`/`tokio` on our own reactor" is reimplementing the userspace side of a
kernel interface whose hard parts — ring memory, barriers, ABI structs — are fixed by the
kernel. The reactor can be faster or smaller, but it is not a new substrate.

---

## 4. What is actually sovereign, layer by layer

Reading bottom-up, marking who controls each layer:

| layer | who owns it | verifiable here? |
|---|---|---|
| silicon | the CPU vendor | no |
| firmware / microcode | vendor / board | no |
| **kernel (Linux + ABI)** | upstream / distro | no (huge, C, ring 0) |
| syscall wrappers (`libc`, or inline `asm!`) | us / libc | the *wrappers* yes, the kernel no |
| runtime / reactor / scheduler | us | yes, if written in a verifiable subset |
| **the algebra (`Cell`, `R[x]/(x²−σ)`)** | us | **yes — Lean proves it; Aeneas refines the `Cell<i64>` kernel** |
| **storage (`Table<K, V>`)** | us | Sole representation; declaration only at the boundary commit. Executing constructor, reads, writes, build and layout must all be refined before acceptance. |
| application / game logic | us | yes |
| language and compiler (`song` / Slang) | us | toolchain correctness is its own problem |

Sovereignty is not "no dependencies." It is: **know which layers are yours, make those
layers correct, and keep the boundary to the ones you do not own thin, explicit and
audited.** A stack that reimplements `tokio` but has an unproven primitive has chosen the
wrong layer to own. A stack that owns one small, *proven* primitive and calls a thin,
audited kernel boundary is stronger, not weaker.

That is the division of labour the apeiron work is built on:

- **Lean (`sanguine-apeiron`)** proves the algebra over an arbitrary commutative ring,
  once, so `Z/2^64` and `R` are instances rather than analogies.
- **Aeneas + Charon** extract the *actual Rust* kernels into Lean, so "the proof is about
  this code" is a machine-checked claim and not a comment. Penglai already does this:
  `proof/PhysicsMath.lean` is generated from `src/physics/math.rs`, and
  `density_exact` / `phase_flux_exact` / `reverse_phase_flux_exact` are proved against the
  extracted bodies.
- **Kani** stays for the narrow, bit-precise facts (the 60 harnesses in `src/proofs.rs`)
  where a witness over all inputs at a width is cheaper than a symbolic argument.
- **Storage has one proof boundary.** The executing `Table` must refine the corpus directly.
  Neither a reference container nor a tested-only optimized container is an accepted alternative.
  The former storage proofs remain historical evidence about deleted source, not this table.
- **The kernel boundary** stays `unsafe`, small, and named — the reactor's syscall cut,
  the byte↔value cast in `codec::web`. Those are the places that
  are *allowed* to be unsafe and *must* carry a written invariant.

### Frozen single-table boundary

Binding lead decision supersedes `da01485` and the two-container storage clauses in
`docs/direction.md`, `docs/flash-handoff.md` and the earlier storage intent/spec. Those files are
historical for storage until their owners reconcile them. This is an `anchor`, not a new proof.
The supplied checkout is the authorized boundary-edit lane.

**Authority read:** gen2 branch `apeiron-rebuild`, clean revision
`50c94192399add3b9f8e64bd14e36d8cba270705`, at
`../sanguine-apeiron/proof/Apeiron/Table/{Basic,Adjacency,Home}.lean`.
The bridge still pins `d9aefc93522e2109bc96158526de6dbd52477276`; moving that pin and checking
the imported theorems is a separate acceptance task. Reading the new revision is not a build.

#### Representation and public contract

- **One type:** `apeiron::Table<K, V>`, defined in `src/table.rs`; no aliases or wrapper containers.
  `src/map.rs`, `src/safe_table.rs` and `src/set.rs` are deleted, including `Map`, `Words`,
  `Exposed`, `ExposedWords`, `SafeTable`, `TableFull`, `Set`, `WordSet`, `ExposedSet` and the map
  entry/iterator/storage-policy types. The former `CountTable`, regions, scanner and worker code
  are removed from `src/table.rs`, not retained under a second name. Algebra modules remain.
- **Current artifact:** a declaration with private default, home function and initialized-vector
  shard fields, with no constructor or methods. It is not an executing implementation. Library
  export and the merge example name only `Table`; legacy tests, examples, benches, fuzz and Kani
  callers intentionally remain red pending semantic migration. No alias makes their old API work.
- **Required API:** `new(d, home) -> Self`, `get(&self, &K) -> &V`,
  `insert_with(&mut self, f, k, v) -> Result<(), StorageError>`, and
  `build(f, d, kvs: Vec<(K,V)>, home) -> Result<Self, StorageError>`.
  `home: fn(&K) -> u64`; writes use `K: Eq`, `V: Clone`, and `f: Fn(V,V) -> V`.
  `new` is allocation-free (empty shard vector); the first write initializes 256 shards.
  `get` returns the default even before allocation. No `Option`-valued absence is imposed:
  `V = Option<W>` is an ordinary instance. Presence uses `bool`, count a chosen numeric type,
  adjacency a list value; none introduces another container name.
- **Semantics:** `HMap K V := K → V`. Let `D(t)(k)` be the unique matching override in shard
  `home(k) mod 256`, or the stored default. `Rep(t,h)` requires zero or 256 initialized shards,
  globally unique keys, every entry in its home shard, and `∀ k, D(t)(k) = h(k)` (through key/value
  denotations when Rust and Lean types differ). Constructors establish `Rep(t, fun _ => d)`;
  reads denote `Apeiron.Table.get`; successful writes establish
  `Rep(t', Apeiron.Table.insertWith f k v h)`. Even the first write computes **`f v d`**, not `v`.
  `build` starts from `d` and applies exactly those writes to the finite stream. This realizes
  the constant-default, finite-update fragment of total functions; it does not enumerate all `K → V`.
- **Layout:** one home-sharded representation with initialized `Vec<(K,V)>` entries, bounded
  scalar search and full-key equality to resolve collisions. Growth preserves denotation; it is
  not another table or a fixed-capacity-four reference. There is no raw-iteration public contract.
  The home callback must be total, stable and respect key equality. Generic `Eq`, `Clone` and merge
  callbacks need explicit denotation/termination contracts; Rust trait names alone prove no laws.
- **Resources:** the canonical functions are total; physical allocation is not. Checked capacity
  arithmetic and fallible reservation report `StorageError` before changing logical entries.
  Reservation failure leaves denotation unchanged (not necessarily allocation capacity); `build`
  returns no partial table. Prove success on the explicit resource-admitted domain, exact checked
  refusal conditions and witnesses, not only an implication conditional on a returned success.
  Allocation failure is not logical fullness. Generic callbacks can panic/diverge; their total
  pure contracts must be discharged for each claimed instance. No unwind theorem is assumed.

#### Exact theorem links and limits

`get_build_acc` fixes **new-first**, per-key stream-order accumulation for every merge.
`build_eq_of_key_streams_eq` permits stable partition across keys for every merge.
`build_gauge_invariance` permits arbitrary stream permutations **only with commutativity and
associativity**, plus a proved permutation of the input. It licenses home sorting/sharding at
the semantic level, not Rust allocation, routing, joins or scheduling. The first implementation
preserves per-key order, so replace and append remain supported; `Table/Adjacency.lean`'s
`Apeiron.Table.get_build_replace`,
`get_build_append`, `build_replace_swap_ne` and `build_append_swap_ne` supply the order controls.

`Table/Home.lean`'s `Apeiron.Table.one_half_decides` assumes `σ > 0`, real cells and an actual
decryptor for the lower channel;
`opaque_needs_lower` explains why a lossy readout cannot decide identity alone.
`coord_home_collisionFree` is for `id : ℕ → ℕ`, **not** truncation to 256 machine shards;
`opaque_home_collides` exhibits that truncation can collide. Keeping and comparing full keys is
the selected collision-safe layout. Dropping them requires a separately checked word encoding/
decoder link; no real null-basis theorem automatically transfers to `ZMod (2^64)`.

| Mechanism | Boundary decision |
|---|---|
| Safe scalar home routing, search, growth, writes, stable build | Inside the same executing-source extraction and direct corpus refinement; all presently open. |
| SSE2/SWAR dispatch and raw `MaybeUninit` lifecycle | Removed. No intrinsic or heap-simulation axiom is allowed to replace their missing proofs. A later optimization must refine this same type and executing path before acceptance. |
| Scoped worker scheduling and joins | Removed. Pure shard independence can be proved; that does not prove thread memory semantics, panic propagation or liveness. Reintroduction requires a checked implementation route, not gauge invariance alone. |
| Rust/Lean translation and modeled `Vec` primitives | Explicit external trust inventory: Charon/Aeneas translation fidelity, model correspondence and Lean kernel. Extraction must still reject unmodeled operations. |
| Allocator/OS, rustc/LLVM, ABI, hardware, GPU | Outside source-level refinement; no allocation-availability, binary-execution, ring-0 or GPU theorem. These assumptions cannot include the desired table transition simulation itself. |

A stored-state application additionally needs its actual read/compute/write composition linked;
the existing scalar/carrier proofs neither establish that composition nor the new table.

---

## 5. Why this matters to the masquerade

The masquerade replaces userspace crates with hashtrinity faces. That is a legitimate
goal with a real payoff: determinism, replay, one `Table` underneath, no supply-chain
drift. But three things are worth stating plainly:

1. **The dependency count is a userspace metric.** Driving `cargo tree | wc -l` down does
   not reduce the kernel surface. At some point the work stops removing dependencies and
   starts re-implementing crates whose correctness is not the bottleneck.
2. **`tokio`/`iroh` are not the primitive.** They are I/O and scheduling on top of a
   kernel that already exists. Replacing them is a runtime project; verifying the table
   is a *correctness* project. They compete for the same time and are not equally
   load-bearing for the guarantees the stack wants to advertise.
3. **The proven layer should be depended on, not duplicated.** If apeiron's algebra is the verified
   primitive, the sovereign move is for hashtrinity (and everything above it) to *depend on* apeiron
   and shrink to adapters, rather than to keep growing a second, unverified implementation of the
   same ideas beside it. The container is the second obligation after the algebra, not a sideline: it is
   tested today and is in scope to be proved, described by its fragment membership rather
   than as permanently unverified.

The floor is the kernel. Above it, pick the smallest layer that must be right, prove it,
and let everything else be an adapter.
