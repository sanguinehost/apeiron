# Locator reserve quadratic and REG-1 budget discharge

> (historical; the harness was removed in c9a2d0c) The `examples/reg1`, `examples/smhasher`, `examples/fold` and `examples/support` harnesses, and the comparison dev-dependencies (criterion, hashbrown, foldhash, fxhash, twox-hash, wyhash, ahash, rustc-hash), were removed from the tree in `c9a2d0c`. References below record that harness as it then stood; the named paths are absent and no command naming them runs now.

The parallel build's locator reservation walked every identity node per reserved key, so the quadratic
grew with the table and the full REG-1 campaign never completed inside its budget. The fix is in
`src/table/locator.rs` and `src/coord/locator_core.rs`: `observe_locator_receipt` reads
`Directory::identity_node_count()` instead of summing `EdgeList::capacity()` (the constant one edge
record per node), and `reserve_path`/`reserve_bytes` pass the borrowed `QueryIdentity` view instead of
materializing a per-key `Vec<u8>`. The unchanged required campaign then completed in 477.79 seconds
below the 600-second ceiling with its controls. Budget discharge is pending lead acceptance; G1 ratio
acceptance remains open. No commit, pin change or Kani run is part of this work.

## Binding

Base `262bb0a94eff2cc7d5655122a954f44d74de96b6` plus accepted dirty source, merged with the assurance
factoring. Generated `proof/ApeironWord.lean` sha256
`491b49704216000c723de5a1a59122db97ea99c2fb464eb6c2fa939d94a4f344` supersedes the factoring refresh
identity `5599fe7d…`.

| artifact | sha256 |
|---|---|
| `src/table/locator.rs` | `be55f3a6fc111376fde9e428b0c11a05fc6cc40f4194791dd583b1b7af2347a8` |
| `src/coord/locator_core.rs` | `56be11683f4a4aed1e01815c520884b85b677114f06391a69a7a3b54dcc90ff2` |
| harness `examples/reg1/{acceptance,performance}.rs` | phase-marker edits; the factoring receipt records them as concurrent |

Host `leviathan`, Intel Core i9-14900KF, 32 threads, kernel `7.2.6-1-cachyos`; rustc
`1.97.0-nightly (17584a181 2026-04-13)`, cargo `1.97.0-nightly (eb94155a9 2026-04-09)`; release
profile, cwd `/home/socol/Workspace/apeiron`. Raw artifacts:

| raw artifact | sha256 |
|---|---|
| `/tmp/opencode/full-required.log` | `c082a4d6c77f487ea1bde759c4ae30b2449e174488afb7ab7f3116b1e0adba49` |
| `/tmp/opencode/full-required-time.txt` | `1f0058788336f96b97ac77569b746ed3ec0f73b0c5375aec90a9b9f7483b70bc` |
| `/tmp/opencode/smoke-accept.log` | `f1a2b72b31fd4545b113757531443995b932af7a2514e48633b12f106c7b72e6` |
| `/tmp/opencode/micro-view-time.txt` | `1fda88fa3da5e0035c19b43e2d53c3e4ceb7deedaf882e019c39da7fc42fab24` |

Tree note: later tasks moved the current digests (`src/table/parallel.rs` `14b49724…`,
`src/table/locator.rs` `f12d0413…`, generated `proof/ApeironWord.lean` `d0a34ee5…`). The campaign
numbers in this receipt bind to the digests above; the later task evidence is in
`docs/epics/build-cost-optimization.md`, and no claim here changed.

## Root cause and fix

`observe_locator_receipt` summed `node.edges.capacity()` over `directory.identity_nodes()`.
`EdgeList::capacity` is the constant one edge record reserved when an identity node is created, so
the sum equals `identity_node_count()` — an O(1) value computed in O(nodes) per reserved key and
O(nodes²) per build. At one million keys the first build never finished: `micro_spatial` exceeded
180 s with no phase line, one thread at 100% CPU, last sampled symbol
`apeiron::table::observe_locator_receipt`. The fix reads the count directly; the follow-up view
change removes two per-key buffers. Measured: n1M spatial build 1354.3 ms → 1311.1 ms (view change
alone ~3%; the receipt's phase table splits the remainder).

The pair doubles as the baseline/red/restore control for the discharge: the pre-fix source is the
mutation — `micro_spatial` produced no phase line in 180 s and the full campaign was killed at its
600 s bound with no pair table (`docs/evidence/open-area-performance.md`); the fixed source restores
completion with its controls.

## Discharge run

```sh
flock /tmp/opencode/apeiron-lane.lock timeout 600 \
  /usr/bin/time -v -o /tmp/opencode/full-required-time.txt \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls
```

Exit 1 (strict acceptance rejection), wall 7:57.79 = **477.79 s**, not killed; the wrapper bounds the
timed command only, so lock acquisition is outside the measurement. The earlier attempt used the
same check and profile with `timeout 600s` inside `flock`; it set `CARGO_TARGET_DIR` and
`CARGO_BUILD_JOBS=2` for its worktree and this run did not. Registered population, acceptance rule
and statistical protocol are unchanged: 160 serial plus 2 spatial cells, three campaigns, two
warmups, 15 samples per campaign, 4096 serial operations, interval maximum ≤ 1.0; the run's own
registration validation reported `missing=0`.

| phase | elapsed |
|---|---|
| `perf.witness` | 1.456 s |
| `perf.strata` | 21.051 s |
| `accept.strata` | 20.763 s |
| `accept.spatial` | 386.484 s |
| `accept.build_campaign.serial` | 13.812 s |
| `accept.evaluate` | 0.001 s |
| `accept.controls` | 34.145 s |
| total (wall) | 477.79 s |

```text
acceptance_phase=per-pair profile=full registered_cells=160 spatial_cells=2 json=false
HEADLINE registered=2592 load_bearing=2268 applicable=1450 pass=325 fail=1125 missing=0 \
  concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false
```

55 control observations ran, all `observed=red` with `child_exit=2`.

## Gates

| gate | result |
|---|---|
| `proof/regen.sh` (after each source change) | exit 0; 1708 jobs |
| `proof/regen.sh --check` | "extraction is fresh" |
| `cd proof/corpus && lake build` | exit 0; 3493 jobs |
| `cargo test --locked --lib` | exit 0; 91 passed, 1 ignored |
| `cargo test --locked --example reg1` | exit 0; 14 passed |
| `python3 tools/check-assurance.py --check` | exit 0 |
| `python3 tools/check-claims.py --check` | exit 0 |

Not run: the full workspace Rust suite (the factoring receipt's 555.752 s run predates the phase
markers and both source changes).

## Trust boundary and residual

No rustc/LLVM codegen, allocator/unsafe-container or machine-execution claim; extraction freshness and
the bridge are separate checks and prove no runtime behavior. G1 remains open: 325 passing /
1125 failing pairs, `concurrency_unmatched=2`, `address_face_unmatched=4`, `full_acceptance=false`.
The serial reserve-all and rebase still dominate the build (micro n1M build 1311.1 ms against the
post-scatter micro receipt's 27.7 ms), which is the next G1 optimization target, not a budget
blocker. Future wrapper calls place `timeout` outside `flock` so lock acquisition is bounded too.
Budget discharge requires lead acceptance of this receipt.

## Re-run at the W2-T1 source identity

The unchanged command was re-run after W1-T1–W1-T4 and W2-T1–W2-T3 under `timeout 600`; the source
identity is `/tmp/opencode/w4-micro-identity.txt` (generated `ApeironWord.lean` `8e76b917…`,
`src/coord/locator_core.rs` `f2708a6f…`, `src/table/locator.rs` `a017cae6…`, `src/table.rs`
`b5ddd10f…`, `src/table/parallel.rs` `8018d8f4…`, `src/table/scan.rs` `ec633507…`), raw log
`/tmp/opencode/w4-reg1-full.log`. Result: exit 1 (strict acceptance), wall **316.0 s** below the
600-second ceiling, 55 controls observed red as expected, `HEADLINE registered=2592 load_bearing=2268
applicable=1450 pass=308 fail=1142 missing=0 concurrency_unmatched=2 address_face_unmatched=4
full_acceptance=false`. `micro_spatial` at the same identity: 1M build 625.3 ms (reserve 587.0,
kernel 10.25, assemble 15.0, end-to-end 648.2, `digest_ok=true`), against 724.6 ms after W1 and
1311.1 pre-fix.

The pass count moved from 325 to 308 between the two runs with both unmatched counts unchanged; the
17 pairs are ratio/interval-borderline flips under the faster build. No per-pair attribution was
run, so no claim is made about them. G1 remains open.
