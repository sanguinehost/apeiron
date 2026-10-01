# Attn extraction drift: the Aeneas root does not compile the current source

Found while extracting `word_kv_append`. Grade: **blocker / trust-boundary finding**, not a cost
claim. It does not touch the non-attn roots already gated this session (row-dot `matvec.rs`,
capacity `table/resources.rs`).

## Finding

`proof/extract/lib.rs` (mtime 2026-09-28) is stale relative to `src/engine/attn.rs` (mtime
2026-10-01). The extraction unit does not compile, so `proof/regen.sh --check` cannot pass and the
committed `proof/ApeironWord.lean` (mtime 2026-09-29) cannot be regenerated from HEAD.

Reproduce:

```sh
cargo build --locked --offline --lib --manifest-path proof/extract/Cargo.toml \
  --target-dir /tmp/extract-check
# exit 101:
#   error[E0583] file not found for module `f32_core`
#     (= note: create ../../src/engine/f32_core.rs; attn.rs:12 `pub mod f32_core;`
#        resolves relative to the build of the #[path]-included attn.rs, not src/engine/attn/)
#   error[E0433] cannot find `mixer` in `engine`   (src/engine/attn.rs:6)
#   error[E0432] unresolved import `crate::engine::mixer`  (src/engine/attn.rs:7)
```

`src/engine/attn.rs` now imports `crate::engine::mixer::{f32_carrier, carrier_from_env, Carrier}`
(attn.rs:6-7) and declares `pub mod f32_core;` (attn.rs:12, file present at
`src/engine/attn/f32_core.rs`). `proof/extract/lib.rs`'s `engine` mirror (`lib.rs:89-95`) does not
declare `mixer`, and the `#[path]` module cannot find the submodule.

## Consequence

The committed extraction carries the pre-F32 `KvState` (`proof/ApeironWord.lean:2801-2810`:
`n_layer, ..., v : Vec<Vec<I64>>`), while the source has
`KvState { carrier: Carrier, storage: KvStorage }` (`src/engine/attn.rs:112-121`). So the committed
attn/KV headlines — `word_kv_append_ok`, `word_kv_append_capacity_refusal`,
`KvReachable_append_geometry`, and any `word_attn_head` / `word_kv_state_new` refinement — are
stated against a body that no longer exists in the source. This is the "green gates over broken
artifacts" class: the theorems still check, but they refine a stale denotation.

`proof/ApeironWord.lean` is additionally dirty-modified in the working tree (30 insertions / 30
deletions, line-comment updates only, written 2026-09-29), so the working extraction is also not the
committed one.

## Fix options (out of the cost-model worker's scope)

- (a) Update `proof/extract/lib.rs`: add the `crate::engine::mixer` mirror (the module the current
  `attn.rs` imports) and fix the `pub mod f32_core;` resolution (a `#[path]` to
  `../../src/engine/attn/f32_core.rs`, or declare `f32_core` under the attn mirror). This changes the
  extraction identity and needs a fresh `proof/regen.sh`, so it is a proof-owner change.
- (b) Change `src/engine/attn.rs` back to a form the pinned root can translate. Not a cost-model
  change.

Either way it is an architecture escalation: it moves the extraction source set and must be
re-established before any attn-family cost task or admission contract is registered.

## Independently blocked (Aeneas limitation, second blocker)

Even with (a), `KvState::append` is not translatable by the pinned Aeneas: the `KvStorage::F32` arm
returns from inside two loops (`src/engine/attn.rs:934,942`), and Aeneas fails closed with
"Returns inside of nested loops are not supported yet". `-loops-no-rec`, `-loops-to-rec` and
`--ullbc` all fail the same way. This needs a source rewrite (hoist the `is_finite` check out of the
loop, or return `Result` from a helper) or a newer Aeneas pin.

## Not affected

`matvec.rs` (row-dot), `table/resources.rs` (capacity), the numeric kernels and the format kernels
extract and gate green this session; their roots do not import `engine::mixer`.
