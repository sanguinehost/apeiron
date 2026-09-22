# First-cut baseline receipt (APEIRON-PERF-REWORK-W3-T1)

Binding: this receipt binds to the dirty-tree digests below, **not to HEAD**.
`apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`. The captured
`src/table.rs` is uncommitted; HEAD's copy differs. Any later W4-T1 rewrite is
compared against the digests in §1, not against a commit.

## 1 · First-cut source snapshot

| artifact | sha256 | lines |
|---|---|---|
| live `src/table.rs` (at capture) | `c34ddf17ef2ed7a81fd4b3e16447479b55d737af8f5253b99dd397414d9b1dc1` | 389 |
| snapshot `docs/evidence/first-cut/table.rs` | `c34ddf17ef2ed7a81fd4b3e16447479b55d737af8f5253b99dd397414d9b1dc1` | 389 |
| live `tests/table.rs` (at capture) | `cf9969bf0c2ededd97cb23c6a36f7abfcd9614f3c2b7e84595f97962ecf10641` | 59 |
| snapshot `docs/evidence/first-cut/tests-table.rs` | `cf9969bf0c2ededd97cb23c6a36f7abfcd9614f3c2b7e84595f97962ecf10641` | 59 |

`cmp` reports both snapshot pairs byte-identical to the live files at capture
(`TABLE_SNAPSHOT_IDENTICAL`, `TESTS_SNAPSHOT_IDENTICAL`). Re-checked after the
strict run: live digests unchanged.

## 2 · Harness identity at capture

| artifact | sha256 | lines |
|---|---|---|
| `examples/reg1/main.rs` | `0e132290a0b084a03fb99feac977de6d0f6fe910b62fae7693a65adb8f735dd7` | 606 |
| `examples/reg1/strata.rs` | `895ff8ef65b2a6e53b81bdd31647efd8659a6783187eb8a09fa9c3a399fb0d3b` | 1490 |
| `examples/reg1/performance.rs` | `bd73bb70116dc798c007838d7764164c07c5270f0530ed74248ccc076d472238` | 1182 |
| `examples/reg1/matrix.rs` | `620eae3ef85390f18f840937ce96a0cc606d9fb2c06e63a112eaa3459c6095b1` | 1385 |
| `examples/reg1/workload.rs` | `100371053e8e00a50eb91e90887c41659996aa30b6ee6392a8a4493e1d63cf5d` | 817 |
| `examples/reg1/correctness.rs` | `4294f5ec2a6172d76c8e893ccf31973aaf5a7b06d5c839b02d3edd4473e62379` | 553 |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` | 1636 |

`matrix.json` re-checked after the strict run: digest unchanged; the run does not
write it.

## 3 · Prior before-measurements (handoff-supplied)

Recorded as stated by the W3-T1 handoff; no in-repo citation for these numbers was
located, so they are prior measurements with source not in this tree.

| configuration | ratio (replacement / wrapping-add) | n |
|---|---|---|
| historical `CountTable` / hashtrinity | `0.2813` | 1M |
| bounded fixed-16 shared witness | `2.4647` / `2.4286` | — |
| first growable cut | `9.7270` / `9.6198` | — |

The fresh strict run (§4) emits first-cut shared fixed-16 ratios `9.5316` /
`9.5121`. The two first-cut pairs differ across measurement moments; the fresh
run's numbers are authoritative for this receipt's source digest.

## 4 · Strict acceptance run on the first-cut source

Command (cwd `/home/socol/Workspace/apeiron`):

```sh
cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

Exit code: **1** (strict acceptance failure; the run is the evidence, nothing was fixed).
A cargo warning `error: duplicate key` from the hashtrinity git checkout
(`opsfold/Cargo.toml:61`) is printed before `Finished`; it is non-fatal and does
not determine the exit status.

`strata_summary`:

```text
strata_summary cells=160 measured=160 pass=24 fail=136 not_applicable=0 missing=0 rejected=0 resource_cells=3 resource_not_applicable=1 resource_missing=2 campaigns=3 warmup=2 samples=15 ops=4096 ht_rev=4cc14471ae24f3f204fb0b0e91736ef75c909358 rustc="rustc 1.97.0-nightly (17584a181 2026-04-13)" cpu="Intel(R) Core(TM) i9-14900KF"
```

`strict_acceptance` (from the `axis=performance` evidence line):

```text
strict_acceptance=fail applicable=160 failing=136 worst_cell=occ16-direct-index-resident-absent-miss-wrapping-add ratio=45.4000
```

| field | value |
|---|---|
| failing cells | 136 / 160 |
| worst cell | `occ16-direct-index-resident-absent-miss-wrapping-add` |
| worst ratio | `45.4000` |
| ratio limit | `1.00` |
| first failing cell | `occ1-direct-index-resident-present-hit-replacement` ratio `4.2256` |
| shared fixed-16 axis | replacement `9.5316`, wrapping-add `9.5121` |
| candidate/hashtrinity | `2.5810` (experimental, not the REG-1 axis) |
| candidate/current | `0.2713` (experimental, not the REG-1 axis) |

Axis verdicts: `identity=pass`, `round_trip=boundary`, `determinism=missing`,
`growth_merge_refusal=pass`, `performance=fail`, `coverage=missing`.
`report_complete=false`, `reg1_pass=false`.

## 5 · Controls

Observed: **25 control lines, 25 red, 0 green** (11 distinct control names:
`retired-shardfull-receipt` ×8, `retired-shardfull-digest` ×8,
`fabricated-current-cell`, `wrong-workload-identity`, `invalid-samples`,
`absent-samples`, `cached-home-only-collision`, `wrong-oracle-digest`,
`deleted-required-current-cell`, `selected-implementation-mismatch`,
`strict-acceptance-single-failing-cell`).

The W3-T1 handoff expected `8/8` red controls. That count is **not** reproduced by
this source revision; the observed count is 25/25. Recorded as observed; the
handoff figure is not adopted.

## 6 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux (rolling), kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads |
| memory | 62 GiB |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target | `x86_64-linux`, release |
| hashtrinity rev | `4cc14471ae24f3f204fb0b0e91736ef75c909358` |
| apeiron head | `c450c222905c050f793a23332708c230c658f82c` (dirty) |

Raw run log: `/tmp/opencode/reg1-strict-baseline.log`,
sha256 `28268f42821ec47444d121918a722fdd8fc36a697e912cc85b31a4260d6a9afe`
(ephemeral; the exact lines above are the durable record).

## 7 · Scope and exclusions

This receipt is the "before" for the W4-T1 rewrite. It captures source identity
and a red strict-acceptance result only.

- In scope: identity of the first-cut `Table`, the harness, and the strict gate
  result on this machine/toolchain.
- Out of scope (section-8 obligations not evidenced here): `Sigma` in every
  executing signature, `remove`, SIMD, allocator behaviour, threading, std
  internals, rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V.
- No history, Kani, `proof/regen.sh`, or `lake build` was run for this receipt.
- Grade: snapshot and harness identity are digest receipts (`tested` identity);
  the strict run is a `measured`/gate result. The prior numbers in §3 are
  handoff-supplied prior measurements, source not located.

## 8 · Open items

- **REG-1 OPEN**: the strict performance acceptance fails 136/160 on this source.
- **G1 OPEN**: the performance objective is not met by the first growable cut.
