# REG-1 performance remeasurement, post correctness lifecycle (APEIRON-PERF-REMEASURE-T3)

Fresh strict-gate measurement on the current source, after the spill tag
fingerprint (`tag = TAG_LIVE | ((place >> 8) as u8 & 0x7f)`), the checked
fallible receipted summary lifecycle, the fallible density planner, the parallel
serial reserve-all + seam, and the intact-flag `is_intact`. The prior receipt
([`perf-remeasure-2026-09-19-post-spill.md`](perf-remeasure-2026-09-19-post-spill.md))
binds `src/table.rs` sha256 `79621926…`; the executing source is now
`00c3c7f8…`. The prior numbers are stale for this identity; this receipt
supersedes them for the current identity only.

## 1 · Binding

Dirty tree, not HEAD. `apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`.

| artifact | sha256 |
|---|---|
| `src/table.rs` | `00c3c7f82c840bd6be18978be5f356d97c3918278f373396f4a0c9417ae65e01` |
| `src/table/scan.rs` | `698b1ecaa6c15cf1f5faa8c3286dc3f60eb9898b13c89215085a021e9f9a7578` |
| `src/table/partition.rs` | `7207a4e9784b5c4153856f2c73d1d78df3dadf25a617d6aae809670cac4b2bf2` |
| `src/table/parallel.rs` | `66f0ba73450cd834da9f483403d03aa20503374cff5765e54c7ed8562d52ed46` |
| `src/table/resources.rs` | `a1d6e242c912706752375027c21b255677830ec74254d89fc8d5dd1324c06009` |
| `examples/reg1/main.rs` | `8c1787fe63bac5d225a53f4535d190a009e6f5480284c1a55e222412ae573268` |
| `examples/reg1/performance.rs` | `ea77fbe2f2cc1248283624f5153eae2669cff4bfeb6efe03c104dc4508c01930` |
| `examples/reg1/acceptance.rs` | `d2f8c0346a4b2b3a82498c706f8dcf721e0f1df65902a942973c5d28f2326eb2` |
| `examples/reg1/strata.rs` | `98cfd2be21eb5a94587d00a90f2dfcf7c446951c97300d1ca5510ab36c01b777` |
| `examples/reg1/matrix.rs` | `405351ee86cc9c4cfb003d11f62dfaf3cb518ce314a9df29a4aef205bf87cf90` |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` |
| `Cargo.toml` | `9d4452267341ef7ee72f23fbccee3a96e3585974cb7478b8b3f8e46325b8711f` |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |

Changed since the post-spill receipt: `src/table.rs` (`79621926…` → `00c3c7f8…`),
`src/table/partition.rs` (`2e9a2f56…` → `7207a4e9…`),
`src/table/resources.rs` (`5b04be96…` → `a1d6e242…`); `src/table/parallel.rs` is
new to the binding table; `src/table/scan.rs` is unchanged (`698b1eca…`). The run
writes none of these files. The reg1 harness sources (`main.rs`, `performance.rs`,
`acceptance.rs`, `strata.rs`, `matrix.rs`, `matrix.json`, `Cargo.toml`,
`Cargo.lock`) are byte-identical to the post-spill receipt.

## 2 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux, kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads |
| CPU governor / turbo | `powersave`; `intel_pstate/no_turbo=0` |
| memory | 62 GiB |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target / profile | `x86_64-linux`, release |
| hashtrinity rev | `4cc14471ae24f3f204fb0b0e91736ef75c909358` |
| run log | `/tmp/opencode/reg1-full-postcorrect-20260919-180612.log`, sha256 `c8d43f2cd165c251b20e02170820d79e2ba31bc2aeb27565455b4df2197d924b`, 2690 lines |

The host was otherwise idle: load average `0.65/0.86/1.20` at start and
`0.67/0.87/1.14` after the run; no Lean, Kani, coverage or other build ran
concurrently.

## 3 · Commands, exit codes, elapsed

All heavy commands ran under `nice -n 10` and a hard `timeout`.

```sh
# 1. build (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 1800 cargo build --release --locked --example reg1
# exit 0; wall 0:02.50; user 11.71 s; max RSS 518900 KiB
# log /tmp/opencode/reg1-build-postcorrect-20260919-180607.log
#     sha256 1f5bf417f456012a92e9b331127b08cc69a3a41c29ed57f4f23a4e80a08760cf

# 2. strict full gate (cwd /home/socol/Workspace/apeiron)
TS=20260919-180612
/usr/bin/time -v nice -n 10 timeout 3600 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls \
  > /tmp/opencode/reg1-full-postcorrect-$TS.log 2>&1
# exit 1 (strict acceptance failure); wall 0:01:40.11; user 191.82 s; system 7.87 s;
# 199% CPU; max RSS 714868 KiB. timeout 3600 NOT reached; smoke fallback not run.
```

Non-fatal cargo warning: `error: duplicate key` from the hashtrinity git checkout
`opsfold/Cargo.toml:61`; it precedes `Finished` and does not set the exit status.

## 4 · HEADLINE

```text
HEADLINE registered=2592 load_bearing=2268 applicable=1450 pass=421 fail=1029 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false
```

Verdict: **fail** (`full_acceptance=false`). `missing=0`; 2 concurrency-unmatched,
4 address-face-unmatched, the other 1023 are ratio failures. Against the post-spill
receipt (`pass=387 fail=1063`, same `registered`/`load_bearing`/`applicable`):
`pass` rose 387 → 421, `fail` fell 1063 → 1029.

Metric: per-pair `selected/incumbent` ratio of pooled **median elapsed ns** over one
deterministic serial schedule (`STRATA_OPS = 4096` registered ops, `campaigns=3
warmup=2 samples=15`), interval = min/max campaign ratio; acceptance requires interval
max `<= 1.0`. `ns/op = median_ns / 4096` for serial rows. Spatial rows are elapsed
wall ns of the packed parallel count at matching worker width (no ns/op stated).
The pooled point ratio can fall marginally outside the campaign min/max interval
(observed on several cells, both directions); acceptance reads the interval max.

`acceptance_phase=per-pair profile=full registered_cells=160 spatial_cells=2 json=false`.

## 5 · Worst failing pairs (top-20 current-strata, by interval max)

`ns/op` is `sel_median_ns / 4096`.

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| current-strata | `occ16-seeded-random-absent-miss-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 20.4682 | [20.0871,20.5922] | 16.161 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 19.8637 | [19.1943,20.2727] | 18.923 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 19.1104 | [18.5575,19.3316] | 18.812 |
| current-strata | `occ16-seeded-random-absent-miss-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 18.2012 | [18.1236,18.5058] | 15.993 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table w1` | `hashbrown_foldhash w1` | 14.9341 | [14.8725,15.0975] | 18.923 |
| current-strata | `occ8-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 14.6136 | [14.5484,14.6337] | 10.949 |
| current-strata | `occ8-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 14.5840 | [14.5000,14.6471] | 10.931 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table w1` | `hashbrown_foldhash w1` | 14.1460 | [14.0701,14.2779] | 18.812 |
| current-strata | `occ16-clustered-skewed-absent-miss-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 13.2175 | [13.1593,13.3719] | 10.907 |
| current-strata | `occ16-clustered-skewed-absent-miss-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 13.1937 | [13.1811,13.2302] | 10.894 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 13.0705 | [12.9891,13.1988] | 18.923 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 12.8636 | [12.7463,13.1009] | 18.812 |
| current-strata | `occ16-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 12.3244 | [12.2749,12.5069] | 10.092 |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 12.2664 | [12.3033,12.3288] | 9.814 |
| current-strata | `occ2-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 12.2547 | [12.1547,12.2719] | 8.727 |
| current-strata | `occ2-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 12.1343 | [12.0876,12.2423] | 8.645 |
| current-strata | `occ4-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 12.1534 | [11.9544,12.1743] | 8.780 |
| current-strata | `occ4-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 12.0674 | [12.0383,12.1629] | 8.697 |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 11.8644 | [11.8848,11.9381] | 9.098 |
| current-strata | `occ1-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 11.7557 | [11.7570,11.9275] | 9.009 |

The worst cell moved from the spill **present-hit** path to the spill
**absent-miss** path. The new worst present-hit cells are `occ8-offset-beyond-direct-present-hit-*`
(14.58/14.61×, ~10.9 ns/op); `occ16` present-hit is 12.27/12.32× at 9.81/10.09 ns/op,
matching the ~9.9 ns/op micro measurement.

## 6 · Spatial failures (10 rows, none passing)

| cell | selected | incumbent | ratio | interval | concurrency | address_face | basis |
|---|---|---|---|---|---|---|---|
| `spatial/n100000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 0.7953 | [0.7830,0.8718] | matched | unmatched | parallel |
| `spatial/n100000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 2.6974 | [2.5125,2.9699] | MISMATCH | unmatched | serial |
| `spatial/n100000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.1167 | [1.0911,1.1674] | matched | unmatched | parallel |
| `spatial/n100000` | `apeiron_table w16` | `foldhash_par16 w16` | 2.6115 | [2.5237,2.8098] | matched | matched | parallel |
| `spatial/n100000` | `apeiron_table w16` | `fxhash_par16 w16` | 3.1125 | [2.7908,3.4410] | matched | matched | parallel |
| `spatial/n1000000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 2.8653 | [2.5369,3.1616] | matched | unmatched | parallel |
| `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 23.0538 | [22.0638,23.1265] | MISMATCH | unmatched | serial |
| `spatial/n1000000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.0689 | [1.0193,1.0728] | matched | unmatched | parallel |
| `spatial/n1000000` | `apeiron_table w16` | `foldhash_par16 w16` | 5.6274 | [5.3468,6.1887] | matched | matched | parallel |
| `spatial/n1000000` | `apeiron_table w16` | `fxhash_par16 w16` | 6.0770 | [5.8106,6.2807] | matched | matched | parallel |

`spatial/n100000` vs `count_par` is under the ratio limit (0.7953) but fails the
address-face gate. The two `get_each` rows are counted as concurrency-unmatched
(2); the four `count_par`/`hashbrown_rayon` rows are the address-face-unmatched (4).
The four `foldhash_par16`/`fxhash_par16` rows are ratio failures.

## 7 · Best passing pairs (top-10 by ratio ascending)

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| current-strata | `occ8-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0757 | [0.0733,0.0784] | 3.888 |
| current-strata | `occ1-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0784 | [0.0784,0.0785] | 3.885 |
| current-strata | `occ8-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0785 | [0.0781,0.0785] | 3.897 |
| current-strata | `occ16-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0787 | [0.0786,0.0804] | 3.897 |
| current-strata | `occ16-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0788 | [0.0786,0.0832] | 3.906 |
| current-strata | `occ1-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0790 | [0.0785,0.0792] | 3.911 |
| current-strata | `occ2-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0790 | [0.0784,0.0819] | 3.889 |
| current-strata | `occ4-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0820 | [0.0807,0.0825] | 3.890 |
| current-strata | `occ4-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0821 | [0.0788,0.0823] | 3.887 |
| current-strata | `occ2-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0824 | [0.0820,0.0828] | 3.892 |

All ten are `existing-update` on the resident direct-index path, as in the post-spill
receipt.

## 8 · Post-spill vs post-correctness on the previously-worst cells

`pre` is the post-spill receipt (its §5/§6 "post" columns); `post` is this run. Same
cell, incumbent and worker width. `ns/op = sel_median_ns / 4096`.

### Spill present-hit

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval |
|---|---|---|---|---|---|---|
| `occ16-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table w1` | 30.9208 | 24.69 | 12.3244 | 10.092 | [12.2749,12.5069] |
| `occ16-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table w1` | 30.7018 | 24.74 | 12.2664 | 9.814 | [12.3033,12.3288] |
| `occ16-seeded-random-present-hit-replacement` | `hashtrinity_table w1` | 22.5222 | 24.68 | 9.6735 | 10.200 | [9.4370,9.8420] |
| `occ16-seeded-random-present-hit-wrapping-add` | `hashtrinity_table w1` | 22.3041 | 24.69 | 9.7360 | 10.280 | [9.7871,9.8435] |

The tag fingerprint cut the spill present-hit selected cost from ~24.7 to ~9.8–10.3
ns/op (ratio ~30.9 → ~12.3; the incumbent `hashtrinity_table` side is stable at
~0.80–0.82 ns/op in both runs).

### Spill absent-miss

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval |
|---|---|---|---|---|---|---|
| `occ16-seeded-random-absent-miss-replacement` | `hashbrown_fxhash w1` | 23.5900 | 19.68 | 20.4682 | 16.161 | [20.0871,20.5922] |
| `occ16-seeded-random-absent-miss-wrapping-add` | `hashbrown_fxhash w1` | 23.7463 | 19.68 | 18.2012 | 15.993 | [18.1236,18.5058] |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashbrown_fxhash w1` | 17.4845 | 17.61 | 19.1104 | 18.812 | [18.5575,19.3316] |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashbrown_fxhash w1` | 18.1126 | 17.67 | 19.8637 | 18.923 | [19.1943,20.2727] |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashtrinity_table w1` | 9.5818 | 17.61 | 12.8636 | 18.812 | [12.7463,13.1009] |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashtrinity_table w1` | 9.6489 | 17.67 | 13.0705 | 18.923 | [12.9891,13.1988] |

The `seeded-random` absent-miss selected cost fell ~19.7 → ~16.0–16.2 ns/op; the
`offset-beyond-direct` absent-miss selected cost **rose** ~17.6 → ~18.8–18.9 ns/op.
The `hashtrinity_table` ratios rose more than the selected cost alone because the
incumbent side measured faster this run (derived ~1.84 → ~1.46 ns/op on that cell);
that is a measurement-attribution fact, not a change to the pinned incumbent.

### Growth-insert

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval |
|---|---|---|---|---|---|---|
| `occ16-direct-index-resident-growth-insert-replacement` | `hashtrinity_table w1` | 1.5275 | 13.31 | 1.5364 | 12.561 | [1.5075,1.5367] |
| `occ16-direct-index-resident-growth-insert-wrapping-add` | `hashtrinity_table w1` | 1.4905 | 12.84 | 1.5227 | 12.571 | [1.5261,1.5443] |

Essentially unchanged; still the only growth-insert pairs below ~1.6×, and still
failing the `<= 1.0` interval contract.

### Resident absent-miss (named in the post-spill §5 note)

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval |
|---|---|---|---|---|---|---|
| `occ16-direct-index-resident-absent-miss-replacement` | `hashtrinity_table w1` | 4.8359 | 2.65 | 4.9286 | 2.645 | [4.7962,4.9293] |
| `occ16-direct-index-resident-absent-miss-replacement` | `hashbrown_foldhash w1` | 2.1052 | 2.65 | 1.9998 | 2.645 | [1.9202,2.0638] |

## 9 · Per-incumbent tallies (load-bearing; 160 serial + 2 spatial cells each)

| incumbent | pass | fail | n/a | info | post-spill pass/fail |
|---|---|---|---|---|---|
| `hashtrinity_table` | 28 | 132 | 2 | 0 | 28 / 132 |
| `hashbrown_foldhash` | 22 | 138 | 2 | 0 | 14 / 146 |
| `hashbrown_fxhash` | 0 | 160 | 2 | 0 | 0 / 160 |
| `std_siphash` | 92 | 68 | 2 | 0 | 86 / 74 |
| `xxhash_xxh3` | 144 | 16 | 2 | 0 | 152 / 8 |
| `xxhash_xxh64` | 42 | 118 | 2 | 0 | 32 / 128 |
| `wyhash` | 13 | 147 | 2 | 0 | 12 / 148 |
| `ahash` | 34 | 126 | 2 | 0 | 26 / 134 |
| `fnv` | 46 | 114 | 2 | 0 | 37 / 123 |
| `hashtrinity_count_par` | 0 | 2 | 160 | 0 | 0 / 2 |
| `hashtrinity_get_each` | 0 | 2 | 160 | 0 | 0 / 2 |
| `hashbrown_rayon` | 0 | 2 | 160 | 0 | 0 / 2 |
| `foldhash_par16` | 0 | 2 | 160 | 0 | 0 / 2 |
| `fxhash_par16` | 0 | 2 | 160 | 0 | 0 / 2 |
| `blake3` | 0 | 0 | 0 | 162 | 0 / 0 |
| `sha256` | 0 | 0 | 0 | 162 | 0 / 0 |

## 10 · Controls

55 control observations, 0 green (`observed=red`): the 19 acceptance/registration
controls (`wrong-workload-identity`, `invalid-samples`, `absent-samples`,
`cached-home-only-collision`, `wrong-oracle-digest`, `deleted-required-current-cell`,
`selected-implementation-mismatch`, `strict-acceptance-single-failing-cell`,
`spatial-wrong-packed-input`, `spatial-wrong-bucket-routing`,
`spatial-wrong-worker-width`, `acceptance-failing-non-hashtrinity-pair`,
`acceptance-omitted-incumbent-pair`, `acceptance-unauthorized-not-applicable`,
`acceptance-crossing-ratio-interval`, `acceptance-wrong-oracle-digest`,
`acceptance-serial-adapter-wrong-result`, `acceptance-concurrency-mismatch`,
`acceptance-address-face-mismatch`), `placement-only-address`,
`fabricated-current-cell`, and 16× `retired-shardfull-receipt` + 16×
`retired-shardfull-digest` (all red; some print twice).

## 11 · Trust and measurement boundary

- **Measured**: strict per-pair `selected/incumbent` ratio on this machine,
  toolchain and dirty-source identity; the 2592-row pair table; the 55 control
  observations; the fail counts and the worst/best pairs above; the derived
  `ns/op = median_ns / 4096`.
- **Not measured / out of scope**: `cargo test`, `cargo clippy`, `cargo fmt`,
  `cargo kani`, `proof/regen.sh`, Lean builds, coverage census, the `10^8`
  identity row, cross-process determinism, codec round-trip, `remove`, SIMD,
  allocator behaviour, rustc/LLVM codegen, kernel ABI, GPU/SPIR-V, and the
  unextracted `std::thread::scope` spatial orchestration. `report_complete=false`,
  `reg1_pass=false`. Extraction of `is_intact` is blocked separately by Aeneas and
  is not exercised here.
- **Grade**: `measured` for the gate result; the digests and log hash are identity
  receipts. The 55 controls establish the harness rejects corrupted observations,
  not that the table is correct.

## 12 · Residual and next bounded task

- Residual (uncovered): the raw 2690-line pair table is retained only at the
  `/tmp` log path with the sha256 in §2; durable retention is not included here.
  The per-cell `strata_summary`/`acceptance_summary` lines are `--json`-only and
  were not captured in table mode.
- Next bounded task: attribute the spill **absent-miss** probe cost — the new worst
  cells `occ16-seeded-random-absent-miss-*` (20.47/18.20×, ~16.0–16.2 ns/op vs
  `hashbrown_fxhash`) and `occ16-offset-beyond-direct-absent-miss-*`
  (19.86/19.11×, ~18.8–18.9 ns/op) — to a named `src/table.rs`/`src/table/scan.rs`
  region, with a focused before/after measurement on the same 4096-op schedule.
  Do not weaken the `<= 1.0` contract.
