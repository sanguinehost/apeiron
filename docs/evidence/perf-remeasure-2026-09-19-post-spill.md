# REG-1 performance remeasurement, post spill/bitmap fix (APEIRON-PERF-REMEASURE-T2)

Fresh strict-gate measurement on the current source, after the per-region Bloom
summary and the O(live) extent bitmap landed in `src/table.rs`/`src/table/scan.rs`.
The prior receipt ([`perf-remeasure-2026-09-19.md`](perf-remeasure-2026-09-19.md))
binds `src/table.rs` sha256 `a89e9ec7…`; the executing source is now
`79621926…`. The prior numbers are stale for this identity; this receipt
supersedes them for the current identity only.

## 1 · Binding

Dirty tree, not HEAD. `apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`.

| artifact | sha256 |
|---|---|
| `src/table.rs` | `79621926ba4c7b279d5511decba596fcd7b9cee06e90ea0b4525edc83fc0d5f3` |
| `src/table/scan.rs` | `698b1ecaa6c15cf1f5faa8c3286dc3f60eb9898b13c89215085a021e9f9a7578` |
| `src/table/partition.rs` | `2e9a2f56337ef1e23fe016627e3b615f456cdeb6460cbf60432118ea19d5962d` |
| `src/table/resources.rs` | `5b04be96131a360d8245ee3061c58dba85059f39a3efeeb0261442a7051368bf` |
| `examples/reg1/main.rs` | `8c1787fe63bac5d225a53f4535d190a009e6f5480284c1a55e222412ae573268` |
| `examples/reg1/performance.rs` | `ea77fbe2f2cc1248283624f5153eae2669cff4bfeb6efe03c104dc4508c01930` |
| `examples/reg1/acceptance.rs` | `d2f8c0346a4b2b3a82498c706f8dcf721e0f1df65902a942973c5d28f2326eb2` |
| `examples/reg1/strata.rs` | `98cfd2be21eb5a94587d00a90f2dfcf7c446951c97300d1ca5510ab36c01b777` |
| `examples/reg1/matrix.rs` | `405351ee86cc9c4cfb003d11f62dfaf3cb518ce314a9df29a4aef205bf87cf90` |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` |
| `Cargo.toml` | `9d4452267341ef7ee72f23fbccee3a96e3585974cb7478b8b3f8e46325b8711f` |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |

Changed since the prior receipt: `src/table.rs` (`a89e9ec7…` → `79621926…`),
`examples/reg1/acceptance.rs` (`f0f9aff0…` → `d2f8c034…`). `src/table/partition.rs`
is new to the binding table. The run writes none of these files.

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
| run log | `/tmp/opencode/reg1-full-postfix-20260919-172248.log`, sha256 `6ad4e40b135a3b23ba5a72e3edcf40070c6054327b27e78255d4ce2acb7bc384`, 2690 lines |

## 3 · Commands, exit codes, elapsed

All heavy commands ran under `nice -n 10` and a hard `timeout`. Load average was
0.62/1.02/1.27 at start; no Lean, Kani or other build ran concurrently.

```sh
# 1. build (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 1800 cargo build --release --locked --example reg1
# exit 0; wall 0:02.25 (release example already cached); max RSS 553036 KiB

# 2. strict full gate (cwd /home/socol/Workspace/apeiron)
TS=$(date +%Y%m%d-%H%M%S)
/usr/bin/time -v nice -n 10 timeout 3600 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls \
  2>&1 | tee /tmp/opencode/reg1-full-postfix-$TS.log
# exit 1 (strict acceptance failure); wall 0:01:39.98; user 208.45 s; max RSS 761636 KiB
# timeout 3600 NOT reached; step 3 (smoke fallback) not run
```

Non-fatal cargo warning: `error: duplicate key` from the hashtrinity git checkout
`opsfold/Cargo.toml:61`; it precedes `Finished` and does not set the exit status.

## 4 · HEADLINE

```text
HEADLINE registered=2592 load_bearing=2268 applicable=1450 pass=387 fail=1063 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false
```

Verdict: **fail** (`full_acceptance=false`). `missing=0`; 2 concurrency-unmatched,
4 address-face-unmatched, the other 1057 are ratio failures.

Metric: per-pair `selected/incumbent` ratio of pooled **median elapsed ns** over one
deterministic serial schedule (`STRATA_OPS = 4096` registered ops, `campaigns=3
warmup=2 samples=15`), interval = min/max campaign ratio; acceptance requires interval
max `<= 1.0`. `ns/op = median_ns / 4096` for serial rows. Spatial rows are elapsed
wall ns of the packed parallel count at matching worker width (no ns/op stated).

`acceptance_phase=per-pair profile=full registered_cells=160 spatial_cells=2 json=false`.

`load_bearing` rose 1458 → 2268 because `xxhash_xxh3`, `xxhash_xxh64`, `wyhash`,
`ahash` and `fnv` are now load-bearing. Raw pass/fail counts are therefore not
directly comparable to the prior receipt; the per-cell comparison in §6 is.

## 5 · Worst failing pairs (by interval max)

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| current-strata | `occ16-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 30.9208 | [30.6770,31.1075] | 24.69 |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 30.7018 | [30.2096,31.0015] | 24.74 |
| current-strata | `occ16-seeded-random-absent-miss-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 23.7463 | [23.5694,23.7792] | 19.68 |
| current-strata | `occ16-seeded-random-absent-miss-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 23.5900 | [23.5045,23.7635] | 19.68 |
| current-strata | `occ16-seeded-random-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 22.5222 | [20.2366,23.5354] | 24.68 |
| current-strata | `occ16-seeded-random-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 22.3041 | [21.6493,22.7078] | 24.69 |
| spatial | `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 20.7457 | [19.5026,22.2537] | n/a |
| current-strata | `occ16-seeded-random-present-hit-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 21.6546 | [21.5898,21.8550] | 24.69 |
| current-strata | `occ16-seeded-random-present-hit-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 21.6632 | [21.6600,21.6699] | 24.68 |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 21.5063 | [21.4638,21.5557] | 24.74 |
| current-strata | `occ16-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 21.4739 | [21.4177,21.5253] | 24.69 |
| current-strata | `occ8-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 19.7966 | [19.7317,19.8755] | 14.92 |
| current-strata | `occ8-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 19.7706 | [19.7261,19.8178] | 14.92 |
| current-strata | `occ16-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashbrown_foldhash w1` | 18.1096 | [17.9594,18.7543] | 24.69 |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashbrown_foldhash w1` | 18.4782 | [18.2654,18.6754] | 24.74 |
| current-strata | `occ16-seeded-random-present-hit-replacement` | `apeiron_table w1` | `hashbrown_foldhash w1` | 18.4729 | [18.2330,18.5913] | 24.68 |
| current-strata | `occ16-seeded-random-present-hit-wrapping-add` | `apeiron_table w1` | `hashbrown_foldhash w1` | 18.4102 | [17.6420,18.5549] | 24.69 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 18.1126 | [17.9611,18.1722] | 17.67 |
| current-strata | `occ16-offset-beyond-direct-absent-miss-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 17.4845 | [17.4201,17.5700] | 17.61 |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `wyhash w1` | 14.8015 | [14.7380,14.8657] | 24.74 |

The worst cell moved from the spill absent-miss path to the spill **present-hit**
path (`offset-beyond-direct`/`seeded-random`, ~24.7 ns/op, 30.9× worst). The
resident direct-index path is not the worst: `occ16-direct-index-resident-absent-miss-replacement`
selects 10871 ns (2.65 ns/op), ratio 4.8359 vs `hashtrinity_table` and 2.1052 vs
`hashbrown_foldhash`.

## 6 · Pre-fix vs post-fix on the previously-worst cells

Same cell, incumbent and worker width as the prior receipt §5; `pre` is
[`perf-remeasure-2026-09-19.md`](perf-remeasure-2026-09-19.md).

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval |
|---|---|---|---|---|---|---|
| `occ16-seeded-random-absent-miss-replacement` | `hashbrown_fxhash w1` | 373.8538 | 314.62 | 23.5900 | 19.68 | [23.5045,23.7635] |
| `occ16-seeded-random-absent-miss-wrapping-add` | `hashbrown_fxhash w1` | 369.1557 | 310.93 | 23.7463 | 19.68 | [23.5694,23.7792] |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashbrown_fxhash w1` | 365.5041 | 369.43 | 17.4845 | 17.61 | [17.4201,17.5700] |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashbrown_fxhash w1` | 358.6002 | 369.63 | 18.1126 | 17.67 | [17.9611,18.1722] |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashtrinity_table w1` | 268.1054 | 369.43 | 9.5818 | 17.61 | [9.5500,9.6441] |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashtrinity_table w1` | 265.1506 | 369.63 | 9.6489 | 17.67 | [9.5685,9.7562] |
| `occ8-direct-index-resident-admitted-fresh-insert-wrapping-add` | `hashtrinity_table w1` | 254.6132 | 550.32 | 3.4306 | 7.22 | [3.4160,3.4852] |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashbrown_foldhash w1` | 251.8732 | 369.63 | 14.0958 | 17.67 | [13.9219,14.1339] |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashbrown_foldhash w1` | 249.7008 | 369.43 | 13.9781 | 17.61 | [13.7856,14.1991] |
| `occ8-direct-index-resident-admitted-fresh-insert-replacement` | `hashtrinity_table w1` | 250.1037 | 554.18 | 3.5356 | 6.98 | [3.5094,3.5634] |
| `occ16-direct-index-resident-growth-insert-replacement` | `hashtrinity_table w1` | 231.8779 | 1653.21 | 1.5275 | 13.31 | [1.4823,1.6279] |
| `occ16-direct-index-resident-growth-insert-wrapping-add` | `hashtrinity_table w1` | 215.3154 | 1641.10 | 1.4905 | 12.84 | [1.4804,1.5031] |

The spill absent-miss cells fell ~15–21× in ratio and ~15–21× in absolute ns/op
(314.62 → 19.68 on the seeded-random pair); growth-insert fell ~145–152× in ratio
(1653.21 → 13.31 ns/op). The resident absent-miss pair also improved
(prior 4.6780 vs `hashtrinity_table` → now 4.8359; 1.8234 vs `hashbrown_foldhash`
→ now 2.1052). Every compared cell still fails the `<= 1.0` interval contract.

## 7 · Best passing pairs (by ratio ascending)

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| current-strata | `occ16-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0840 | [0.0838,0.0843] | 3.93 |
| current-strata | `occ1-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0841 | [0.0839,0.0852] | 3.93 |
| current-strata | `occ16-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0843 | [0.0839,0.0883] | 3.95 |
| current-strata | `occ1-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0848 | [0.0842,0.0851] | 3.96 |
| current-strata | `occ2-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0855 | [0.0854,0.0857] | 4.00 |
| current-strata | `occ4-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0855 | [0.0854,0.0861] | 4.00 |
| current-strata | `occ2-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0857 | [0.0854,0.0858] | 4.00 |
| current-strata | `occ4-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0864 | [0.0853,0.0882] | 4.05 |
| current-strata | `occ8-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0865 | [0.0862,0.0866] | 4.05 |
| current-strata | `occ8-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0866 | [0.0860,0.0867] | 4.06 |

All ten are `existing-update` on the resident direct-index path; the same cells
passed against `std_siphash` in the prior receipt. The pass set is now dominated by
`xxhash_xxh3` (152 passes), `std_siphash` (86) and `fnv` (37).

## 8 · Non-ratio failures

Concurrency-unmatched (2, both the combined `get_each` arm):

| family | cell | selected | incumbent | ratio | interval |
|---|---|---|---|---|---|
| spatial | `spatial/n100000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 2.7219 | [2.0541,3.0854] |
| spatial | `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 20.7457 | [19.5026,22.2537] |

Address-face unmatched (4):

| family | cell | selected | incumbent | ratio | interval |
|---|---|---|---|---|---|
| spatial | `spatial/n100000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 0.7603 | [0.6334,0.8278] |
| spatial | `spatial/n100000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.1455 | [1.0325,1.3819] |
| spatial | `spatial/n1000000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 2.7739 | [2.7143,2.9200] |
| spatial | `spatial/n1000000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.1169 | [1.0711,1.1820] |

`n100000` vs `count_par` is under the ratio limit (0.7603) but fails the address-face
gate, so it is not a pass. The two `get_each` rows hit the concurrency check first
(`continue`) and are counted once, as concurrency-unmatched.

## 9 · Per-incumbent tallies (load-bearing; 160 serial + 2 spatial cells each)

| incumbent | pass | fail | n/a | info |
|---|---|---|---|---|
| `hashtrinity_table` | 28 | 132 | 2 | 0 |
| `hashbrown_foldhash` | 14 | 146 | 2 | 0 |
| `hashbrown_fxhash` | 0 | 160 | 2 | 0 |
| `std_siphash` | 86 | 74 | 2 | 0 |
| `xxhash_xxh3` | 152 | 8 | 2 | 0 |
| `xxhash_xxh64` | 32 | 128 | 2 | 0 |
| `wyhash` | 12 | 148 | 2 | 0 |
| `ahash` | 26 | 134 | 2 | 0 |
| `fnv` | 37 | 123 | 2 | 0 |
| `hashtrinity_count_par` | 0 | 2 | 160 | 0 |
| `hashtrinity_get_each` | 0 | 2 | 160 | 0 |
| `hashbrown_rayon` | 0 | 2 | 160 | 0 |
| `foldhash_par16` | 0 | 2 | 160 | 0 |
| `fxhash_par16` | 0 | 2 | 160 | 0 |
| `blake3` | 0 | 0 | 0 | 162 |
| `sha256` | 0 | 0 | 0 | 162 |

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
  observations; the fail counts and the worst/best pairs above.
- **Not measured / out of scope**: `cargo test`, `cargo clippy`, `cargo fmt`,
  `cargo kani`, `proof/regen.sh`, Lean builds, coverage census, the `10^8`
  identity row, cross-process determinism, codec round-trip, `remove`, SIMD,
  allocator behaviour, rustc/LLVM codegen, kernel ABI, GPU/SPIR-V, and the
  unextracted `std::thread::scope` spatial orchestration. `report_complete=false`,
  `reg1_pass=false`.
- **Grade**: `measured` for the gate result; the digests and log hash are identity
  receipts. The 55 controls establish the harness rejects corrupted observations,
  not that the table is correct.

## 12 · Residual and next bounded task

- Residual (uncovered): the raw 2690-line pair table is retained only at the
  `/tmp` log path with the sha256 in §2; durable retention is not included here.
  The per-cell `strata_summary`/`acceptance_summary` lines are `--json`-only and
  were not captured in table mode.
- Next bounded task: attribute the spill **present-hit** probe cost — the new
  worst cells `occ16-offset-beyond-direct-present-hit-{replacement,wrapping-add}`
  (30.70/30.92×, ~24.7 ns/op) and `occ16-seeded-random-present-hit-*`
  (21.7/22.5×) — to a named `src/table.rs`/`src/table/scan.rs` region, with a
  focused before/after measurement on the same 4096-op schedule. Do not weaken the
  `<= 1.0` contract.
