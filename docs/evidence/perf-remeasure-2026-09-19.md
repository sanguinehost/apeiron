# REG-1 performance remeasurement (APEIRON-PERF-REMEASURE-T1)

Fresh strict-gate measurement on the current source. The last matched run
([`perf-global-baseline.md`](perf-global-baseline.md)) binds `src/table.rs` sha256
`e56014da…` (pre-rework fixed-shard); the executing source is now the growable
flat/spill table at sha256 `a89e9ec7…`. Those numbers are stale; this receipt
supersedes them for the current identity only.

## 1 · Binding

Dirty tree, not HEAD. `apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`.

| artifact | sha256 |
|---|---|
| `src/table.rs` | `a89e9ec743405ac7d80183e431f6e932ba90353bc777f142865f44005261d09a` |
| `src/table/scan.rs` | `698b1ecaa6c15cf1f5faa8c3286dc3f60eb9898b13c89215085a021e9f9a7578` |
| `src/table/resources.rs` | `5b04be96131a360d8245ee3061c58dba85059f39a3efeeb0261442a7051368bf` |
| `examples/reg1/main.rs` | `8c1787fe63bac5d225a53f4535d190a009e6f5480284c1a55e222412ae573268` |
| `examples/reg1/performance.rs` | `ea77fbe2f2cc1248283624f5153eae2669cff4bfeb6efe03c104dc4508c01930` |
| `examples/reg1/acceptance.rs` | `f0f9aff07ff9b6feeef761f6a940386a89e3f07ba5861b39509ec4ca4e060c33` |
| `examples/reg1/strata.rs` | `98cfd2be21eb5a94587d00a90f2dfcf7c446951c97300d1ca5510ab36c01b777` |
| `examples/reg1/matrix.rs` | `405351ee86cc9c4cfb003d11f62dfaf3cb518ce314a9df29a4aef205bf87cf90` |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` |
| `Cargo.toml` | `9d4452267341ef7ee72f23fbccee3a96e3585974cb7478b8b3f8e46325b8711f` |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |

The run writes none of these files.

## 2 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux, kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads |
| memory | 62 GiB |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target / profile | `x86_64-linux`, release |
| hashtrinity rev | `4cc14471ae24f3f204fb0b0e91736ef75c909358` |
| run log | `/tmp/opencode/reg1-full-20260919-161738.log`, sha256 `2d4eb91b1ce8afd60d969434eaf69735faeb7a4935d952227f588efa3968ff06`, 2690 lines |

## 3 · Commands, exit codes, elapsed

All heavy commands ran under `nice -n 10` and a hard `timeout`. No Lean, Kani or
other build ran concurrently; `/tmp` had 8.6 G free and load average was 1.36 at
start.

```sh
# 1. build (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 1800 cargo build --release --locked --example reg1
# exit 0; wall 0:02.42 (release example already cached); max RSS 542696 KiB

# 2. strict full gate (cwd /home/socol/Workspace/apeiron)
TS=$(date +%Y%m%d-%H%M%S)
/usr/bin/time -v nice -n 10 timeout 3600 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls \
  2>&1 | tee /tmp/opencode/reg1-full-$TS.log
# exit 1 (strict acceptance failure); wall 0:32:33.57; user 2335.82 s; max RSS 749488 KiB
# timeout 3600 NOT reached; step 3 (smoke fallback) not run
```

Non-fatal cargo warning: `error: duplicate key` from the hashtrinity git checkout
`opsfold/Cargo.toml:61`; it precedes `Finished` and does not set the exit status.

## 4 · HEADLINE

```text
HEADLINE registered=2592 load_bearing=1458 applicable=650 pass=78 fail=572 missing=0 concurrency_unmatched=0 address_face_unmatched=6 full_acceptance=false
```

Verdict: **fail** (`full_acceptance=false`). `missing=0`, `concurrency_unmatched=0`;
6 failures are the separate address-face class, the other 566 are ratio failures.

Metric: per-pair `selected/incumbent` ratio of pooled **median elapsed ns** over one
deterministic serial schedule (`STRATA_OPS = 4096` registered ops, `campaigns=3
warmup=2 samples=15`), interval = min/max campaign ratio; acceptance requires interval
max `<= 1.0`. `ns/op = median_ns / 4096` for serial rows. Spatial rows are elapsed
wall ns of the packed parallel count at matching worker width (no ns/op stated).

`acceptance_phase=per-pair profile=full registered_cells=160 spatial_cells=2 json=false`.
The table-mode run prints no `acceptance_summary`/`strata_summary` (those are `--json` only).

## 5 · Worst failing pairs (by interval max)

| family | cell | selected | incumbent | ratio | interval | sel ns/op | inc ns/op |
|---|---|---|---|---|---|---|---|
| current-strata | `occ16-seeded-random-absent-miss-replacement` | `apeiron_table` w1 | `hashbrown_fxhash` w1 | 373.8538 | [370.3637,375.1818] | 314.62 | 0.84 |
| current-strata | `occ16-seeded-random-absent-miss-wrapping-add` | `apeiron_table` w1 | `hashbrown_fxhash` w1 | 369.1557 | [368.1812,370.4833] | 310.93 | 0.84 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table` w1 | `hashbrown_fxhash` w1 | 365.5041 | [363.7280,367.1765] | 369.43 | 1.01 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table` w1 | `hashbrown_fxhash` w1 | 358.6002 | [356.5349,360.1007] | 369.63 | 1.03 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table` w1 | `hashtrinity_table` w1 | 268.1054 | [266.5928,269.6074] | 369.43 | 1.38 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table` w1 | `hashtrinity_table` w1 | 265.1506 | [264.0869,267.0088] | 369.63 | 1.39 |
| current-strata | `occ8-direct-index-resident-admitted-fresh-insert-wrapping-add` | `apeiron_table` w1 | `hashtrinity_table` w1 | 254.6132 | [249.8611,257.7129] | 550.32 | 2.16 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table` w1 | `hashbrown_foldhash` w1 | 251.8732 | [247.8753,255.3626] | 369.63 | 1.47 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table` w1 | `hashbrown_foldhash` w1 | 249.7008 | [248.3792,253.2675] | 369.43 | 1.48 |
| current-strata | `occ8-direct-index-resident-admitted-fresh-insert-replacement` | `apeiron_table` w1 | `hashtrinity_table` w1 | 250.1037 | [249.2837,251.4613] | 554.18 | 2.22 |
| current-strata | `occ16-direct-index-resident-growth-insert-replacement` | `apeiron_table` w1 | `hashtrinity_table` w1 | 231.8779 | [209.7051,234.0361] | 1653.21 | 7.13 |
| current-strata | `occ16-direct-index-resident-growth-insert-wrapping-add` | `apeiron_table` w1 | `hashtrinity_table` w1 | 215.3154 | [212.2631,233.0231] | 1641.10 | 7.62 |

The catastrophic rows are the **spill-path distributions** (`seeded-random`,
`offset-beyond-direct`) on absent-miss and the **growth-insert** cells. The resident
direct-index path is not pathological: `occ16-direct-index-resident-absent-miss-replacement`
selects 10984 ns (2.68 ns/op), ratio 4.6780 vs `hashtrinity_table` and 1.8234 vs
`hashbrown_foldhash`; `occ16-direct-index-resident-present-hit-replacement` selects
10263 ns, ratio 3.3860 vs `hashtrinity_table`.

## 6 · Best passing pairs (by ratio ascending)

| family | cell | selected | incumbent | ratio | interval |
|---|---|---|---|---|---|
| current-strata | `occ16-direct-index-resident-existing-update-wrapping-add` | `apeiron_table` w1 | `std_siphash` w1 | 0.3150 | [0.3141,0.3162] |
| current-strata | `occ16-direct-index-resident-existing-update-replacement` | `apeiron_table` w1 | `std_siphash` w1 | 0.3154 | [0.3159,0.3184] |
| current-strata | `occ8-direct-index-resident-existing-update-wrapping-add` | `apeiron_table` w1 | `std_siphash` w1 | 0.3168 | [0.3141,0.3185] |
| current-strata | `occ8-direct-index-resident-existing-update-replacement` | `apeiron_table` w1 | `std_siphash` w1 | 0.3202 | [0.3163,0.3214] |
| current-strata | `occ4-direct-index-resident-existing-update-wrapping-add` | `apeiron_table` w1 | `std_siphash` w1 | 0.3230 | [0.3209,0.3231] |
| current-strata | `occ1-direct-index-resident-existing-update-replacement` | `apeiron_table` w1 | `std_siphash` w1 | 0.3253 | [0.3248,0.3319] |
| current-strata | `occ4-direct-index-resident-existing-update-replacement` | `apeiron_table` w1 | `std_siphash` w1 | 0.3269 | [0.3238,0.3278] |
| current-strata | `occ2-direct-index-resident-existing-update-wrapping-add` | `apeiron_table` w1 | `std_siphash` w1 | 0.3269 | [0.3265,0.3272] |
| current-strata | `occ2-direct-index-resident-existing-update-replacement` | `apeiron_table` w1 | `std_siphash` w1 | 0.3298 | [0.3289,0.3308] |
| current-strata | `occ1-direct-index-resident-existing-update-wrapping-add` | `apeiron_table` w1 | `std_siphash` w1 | 0.3298 | [0.3214,0.3320] |
| current-strata | `occ2-clustered-skewed-admitted-fresh-insert-replacement` | `apeiron_table` w1 | `hashtrinity_table` w1 | 0.3308 | [0.3298,0.3316] |
| current-strata | `occ2-clustered-skewed-admitted-fresh-insert-wrapping-add` | `apeiron_table` w1 | `hashtrinity_table` w1 | 0.3323 | [0.3318,0.3343] |

All 78 passes are `existing-update` or `clustered-skewed` cells against `std_siphash`
or `hashtrinity_table`.

## 7 · Address-face unmatched (6 pairs, fail regardless of ratio)

| family | cell | selected | incumbent | ratio | interval |
|---|---|---|---|---|---|
| spatial | `spatial/n100000` | `apeiron_table` w24 | `hashtrinity_count_par` w24 | 1.6967 | [1.6297,1.7154] |
| spatial | `spatial/n100000` | `apeiron_table` w1 | `hashtrinity_get_each` w1 | 7.4258 | [7.1715,8.0202] |
| spatial | `spatial/n100000` | `apeiron_table` w32 | `hashbrown_rayon` w32 | 1.6991 | [1.6196,1.7571] |
| spatial | `spatial/n1000000` | `apeiron_table` w24 | `hashtrinity_count_par` w24 | 6.8355 | [6.8527,6.9322] |
| spatial | `spatial/n1000000` | `apeiron_table` w1 | `hashtrinity_get_each` w1 | 80.2115 | [79.2462,81.7946] |
| spatial | `spatial/n1000000` | `apeiron_table` w32 | `hashbrown_rayon` w32 | 2.5814 | [2.4410,2.7745] |

The two `par16` spatial incumbents are address-face `matched` and fail on ratio:
`foldhash_par16` 4.3537 / 18.8233 and `fxhash_par16` 5.4138 / 17.2777 at
`n100000` / `n1000000`.

## 8 · Per-incumbent tallies (load-bearing; 160 serial + 2 spatial cells each)

| incumbent | pass | fail | n/a | baseline pass/fail |
|---|---|---|---|---|
| `hashtrinity_table` | 20 | 140 | 2 | 14 / 146 |
| `hashbrown_foldhash` | 0 | 160 | 2 | 11 / 149 |
| `hashbrown_fxhash` | 0 | 160 | 2 | 10 / 150 |
| `std_siphash` | 58 | 102 | 2 | 98 / 62 |
| `hashtrinity_count_par` | 0 | 2 | 160 | 0 / 2 |
| `hashtrinity_get_each` | 0 | 2 | 160 | 0 / 2 |
| `hashbrown_rayon` | 0 | 2 | 160 | 0 / 2 |
| `foldhash_par16` | 0 | 2 | 160 | 0 / 2 |
| `fxhash_par16` | 0 | 2 | 160 | 0 / 2 |

Informational incumbents (`xxhash_xxh3`, `xxhash_xxh64`, `wyhash`, `ahash`, `fnv`,
`blake3`, `sha256`): 162 info rows each, excluded from the load-bearing pass.

Delta vs [`perf-global-baseline.md`](perf-global-baseline.md) (different `src/table.rs`
digest, and its harness lacked the address-face gate): aggregate worst serial ratio
rose from `r_max=29.5121` to `375.1818`, both against `hashbrown_fxhash`, on different
absent-miss cells. Total applicable passes fell 133 → 78. The regression tracks the
growable spill/growth path, not the resident path.

## 9 · Controls

19/19 controls red (`child_exit=2`), 0 green: `wrong-workload-identity`,
`invalid-samples`, `absent-samples`, `cached-home-only-collision`,
`wrong-oracle-digest`, `deleted-required-current-cell`,
`selected-implementation-mismatch`, `strict-acceptance-single-failing-cell`,
`spatial-wrong-packed-input`, `spatial-wrong-bucket-routing`,
`spatial-wrong-worker-width`, `acceptance-failing-non-hashtrinity-pair`,
`acceptance-omitted-incumbent-pair`, `acceptance-unauthorized-not-applicable`,
`acceptance-crossing-ratio-interval`, `acceptance-wrong-oracle-digest`,
`acceptance-serial-adapter-wrong-result`, `acceptance-concurrency-mismatch`,
`acceptance-address-face-mismatch`. Additional registration controls
(`placement-only-address`, `retired-shardfull-{receipt,digest}` ×16) also red.

## 10 · Trust and measurement boundary

- **Measured**: strict per-pair `selected/incumbent` ratio on this machine, toolchain
  and dirty-source identity; the 2592-row pair table; the 19 controls; the fail count
  and the worst/best pairs above.
- **Not measured / out of scope**: `cargo test`, `cargo clippy`, `cargo fmt`,
  `cargo kani`, `proof/regen.sh`, Lean builds, coverage census, the `10^8` identity
  row, cross-process determinism, codec round-trip, `remove`, SIMD, allocator
  behaviour, rustc/LLVM codegen, kernel ABI, GPU/SPIR-V, and the unextracted
  `std::thread::scope` spatial orchestration. `report_complete=false`,
  `reg1_pass=false`.
- **Grade**: `measured` for the gate result; the digests and log hash are identity
  receipts. The 19 controls establish the harness rejects corrupted observations,
  not that the table is correct.

## 11 · Residual and next bounded task

- Residual (uncovered): the raw 2690-line pair table is retained only at the `/tmp`
  log path with the sha256 in §2; durable retention is not included here. The
  per-cell `strata_summary`/`acceptance_summary` lines are `--json`-only and were not
  captured in table mode.
- Next bounded task: attribute the spill-path and growth-insert regression to a named
  `src/table.rs` code region (the `seeded-random`/`offset-beyond-direct` absent-miss and
  `growth-insert` cells at 250–375× and 215–232×), with a focused before/after
  measurement on the same 4096-op schedule. Do not weaken the `<= 1.0` contract.
