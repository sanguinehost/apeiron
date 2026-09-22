# REG-1 performance remeasurement, post stable scatter (APEIRON-PERF-REMEASURE-T4)

Fresh strict-gate measurement on the current source, after the spill tag
fingerprint (`tag = TAG_LIVE | ((place >> 8) as u8 & 0x7f)`), the specialized
read path (single-exit `spill_read`, inlined resident), the totalized summary,
and the parallel stable scatter. The prior receipt
([`perf-remeasure-2026-09-19-post-correctness.md`](perf-remeasure-2026-09-19-post-correctness.md))
binds `src/table.rs` sha256 `00c3c7f8…`; the executing source is now `5f260801…`.
The prior numbers are stale for this identity; this receipt supersedes them for
the current identity only.

Two full passes ran on the same build. Pass 1 (the command below) is the primary
receipt; pass 2 is the reproducibility control in §4a. The two HEADLINEs differ by
one pair (`pass=500 fail=950` vs `pass=501 fail=949`).

## 1 · Binding

Dirty tree, not HEAD. `apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`.

| artifact | sha256 |
|---|---|
| `src/table.rs` | `5f2608010137b16b195baad62cb5214bcadeeea13a9df4da39e19f4bf543a61e` |
| `src/table/state.rs` | `799834665532bd4361ecd7fbd3c1497fffaef42a7044139de9e91fdd29e1f3e8` |
| `src/table/partition.rs` | `7cf4724447b5ef8de8c4d4c2a175432d2284e3c1152b1f41c7f374cd13688c64` |
| `src/table/parallel.rs` | `b22650c1c6db76a7865e782fc0fc56174d3b8b6ac2c7eaccc0af087815df193f` |
| `src/table/scan.rs` | `698b1ecaa6c15cf1f5faa8c3286dc3f60eb9898b13c89215085a021e9f9a7578` |
| `src/table/resources.rs` | `a1d6e242c912706752375027c21b255677830ec74254d89fc8d5dd1324c06009` |
| `proof/ApeironWord.lean` | `4c7bb34f8e4926daacf349ba828d5cf0d24dc725e3b96ceec3fda3fe48ca8bf6` |
| `examples/reg1/main.rs` | `8c1787fe63bac5d225a53f4535d190a009e6f5480284c1a55e222412ae573268` |
| `examples/reg1/performance.rs` | `ea77fbe2f2cc1248283624f5153eae2669cff4bfeb6efe03c104dc4508c01930` |
| `examples/reg1/acceptance.rs` | `d2f8c0346a4b2b3a82498c706f8dcf721e0f1df65902a942973c5d28f2326eb2` |
| `examples/reg1/strata.rs` | `98cfd2be21eb5a94587d00a90f2dfcf7c446951c97300d1ca5510ab36c01b777` |
| `examples/reg1/matrix.rs` | `405351ee86cc9c4cfb003d11f62dfaf3cb518ce314a9df29a4aef205bf87cf90` |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` |
| `Cargo.toml` | `9d4452267341ef7ee72f23fbccee3a96e3585974cb7478b8b3f8e46325b8711f` |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |

Changed since the post-correctness receipt: `src/table.rs`
(`00c3c7f8…` → `5f260801…`), `src/table/partition.rs` (`7207a4e9…` →
`7cf47244…`), `src/table/parallel.rs` (`66f0ba73…` → `b22650c1…`); `state.rs`
was absent from the post-correctness binding table. `scan.rs` (`698b1eca…`) and
`resources.rs` (`a1d6e242…`) are unchanged. The reg1 harness sources (`main.rs`,
`performance.rs`, `acceptance.rs`, `strata.rs`, `matrix.rs`, `matrix.json`,
`Cargo.toml`, `Cargo.lock`) are byte-identical to the post-correctness receipt.
The run writes none of these files.

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
| run 1 log | `/tmp/opencode/reg1-full-postscatter-20260919-210424.log`, sha256 `3916a46821de226202e143f8f5ab2bb0fb73e960cc44d46de36420c61cd79ce3`, 2690 lines |
| run 2 log | `/tmp/opencode/reg1-full-postscatter-20260919-210709.log`, sha256 `7e1ffe7255fa2c49ac8042ee12d1dd438e7c4f24ea9a06e871a9c8c44ea10182`, 2690 lines |
| build log | `/tmp/opencode/reg1-build-postscatter-20260919-210419.log`, sha256 `10c8c46765848a03433fe7129b5da86d7c32510dc4e98a6c495a1f602fde88eb`, 30 lines |

No concurrent build ran (no cargo/Lean/Kani/coverage). A desktop session
(Hyprland, Discord, ChatGPT, sunshine) was resident, so the host was not fully
idle: load average `1.61/1.27/1.15` before run 1, `1.54/1.32/1.18` after run 1,
`4.17/2.40/1.60` after run 2. This is higher than the post-correctness receipt's
`0.65/0.86/1.20`; the effect is quantified in §4a.

## 3 · Commands, exit codes, elapsed

All heavy commands ran under `nice -n 10` and a hard `timeout`.

```sh
# 1. build (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 1800 cargo build --release --locked --example reg1
# exit 0; wall 0:02.30; user 11.22 s; max RSS 565788 KiB

# 2. strict full gate, pass 1 (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 3600 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls \
  > /tmp/opencode/reg1-full-postscatter-20260919-210424.log 2>&1
# exit 1 (strict acceptance failure); wall 0:01:28.63; user 189.04 s; system 8.07 s;
# 222% CPU; max RSS 821704 KiB. timeout 3600 NOT reached; smoke fallback not run.

# 3. strict full gate, pass 2 (reproducibility control)
/usr/bin/time -v nice -n 10 timeout 3600 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls \
  > /tmp/opencode/reg1-full-postscatter-20260919-210709.log 2>&1
# exit 1; wall 0:01:28.54; user 186.54 s; system 8.35 s; 220% CPU; max RSS 757900 KiB.
```

Non-fatal cargo warning: `error: duplicate key` from the hashtrinity git checkout
`opsfold/Cargo.toml:61`; it precedes `Finished` and does not set the exit status.

## 4 · HEADLINE

```text
HEADLINE registered=2592 load_bearing=2268 applicable=1450 pass=500 fail=950 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false
```

Verdict: **fail** (`full_acceptance=false`). `missing=0`; 2 concurrency-unmatched,
4 address-face-unmatched, the other 944 are ratio failures. Against the
post-correctness receipt (`pass=421 fail=1029`, same `registered`/`load_bearing`/
`applicable`): `pass` rose 421 → 500, `fail` fell 1029 → 950.

Metric: per-pair `selected/incumbent` ratio of pooled **median elapsed ns** over one
deterministic serial schedule (`STRATA_OPS = 4096` registered ops, `campaigns=3
warmup=2 samples=15`), interval = min/max campaign ratio; acceptance requires interval
max `<= 1.0`. `ns/op = median_ns / 4096` for serial rows. Spatial rows are elapsed
wall ns of the packed parallel count at matching worker width (no ns/op stated).
The pooled point ratio can fall outside the campaign min/max interval (observed on
several cells, both directions); acceptance reads the interval max.

`acceptance_phase=per-pair profile=full registered_cells=160 spatial_cells=2 json=false`.

### 4a · Reproducibility (pass 2)

Pass 2 HEADLINE: `pass=501 fail=949 missing=0 concurrency_unmatched=2
address_face_unmatched=4`. 29 of 2592 pairs (1.1%) changed status between the two
passes. The selected-side median moved more than 20% on exactly three cells:

| cell | pass 1 sel ns/op | pass 2 sel ns/op |
|---|---|---|
| `occ1-offset-beyond-direct-present-hit-replacement` | 6.450 | 10.339 |
| `occ1-seeded-random-present-hit-replacement` | 10.895 | 6.083 |
| `occ1-seeded-random-present-hit-wrapping-add` | 9.773 | 6.177 |

The `occ1` present-hit selected cost is order-unstable: the offset-beyond and
seeded-random replacement cells exchange ~6.2 and ~10.3 ns/op between passes. Rows
carrying these cells are marked `†` in §5; their ratios are not a stable ranking
signal. The largest ratio swings overall are against `xxhash_xxh3`, whose incumbent
side sits near 0.7–0.8 ns/op so small absolute noise becomes a large ratio change;
the selected side is stable there.

## 5 · Worst failing pairs (top-20 by interval max, pass 1)

`ns/op` is `sel_median_ns / 4096`. `†` marks a row whose selected cell is
order-unstable between passes (§4a).

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| spatial | `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 20.0474 | [19.8189,20.8308] | — |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` † | `apeiron_table w1` | `hashbrown_fxhash w1` | 6.1200 | [5.8410,12.8527] | 6.450 |
| current-strata | `occ1-seeded-random-present-hit-replacement` † | `apeiron_table w1` | `hashbrown_fxhash w1` | 10.3349 | [8.9431,11.4965] | 10.895 |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` † | `apeiron_table w1` | `hashbrown_foldhash w1` | 5.2724 | [5.0395,11.0525] | 6.450 |
| current-strata | `occ1-seeded-random-present-hit-wrapping-add` † | `apeiron_table w1` | `hashbrown_fxhash w1` | 9.3093 | [8.1506,10.7577] | 9.773 |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 10.2197 | [10.1991,10.2151] | 8.119 |
| current-strata | `occ16-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 10.0217 | [9.8928,10.1629] | 8.133 |
| current-strata | `occ1-seeded-random-present-hit-replacement` † | `apeiron_table w1` | `hashbrown_foldhash w1` | 8.5951 | [7.3920,9.5964] | 10.895 |
| current-strata | `occ8-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.8014 | [8.7901,8.9352] | 6.674 |
| current-strata | `occ8-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.8709 | [8.8691,8.9128] | 6.675 |
| current-strata | `occ1-seeded-random-present-hit-wrapping-add` † | `apeiron_table w1` | `hashbrown_foldhash w1` | 7.6190 | [5.3492,8.8575] | 9.773 |
| current-strata | `occ1-offset-beyond-direct-present-hit-wrapping-add` † | `apeiron_table w1` | `hashtrinity_table w1` | 8.5953 | [8.5658,8.8288] | 6.155 |
| current-strata | `occ1-clustered-skewed-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.7713 | [8.7176,8.8262] | 6.733 |
| current-strata | `occ2-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.6612 | [8.6271,8.6680] | 6.185 |
| current-strata | `occ2-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.6316 | [8.6266,8.6557] | 6.172 |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` † | `apeiron_table w1` | `hashtrinity_table w1` | 8.7629 | [8.5870,8.6119] | 6.450 |
| current-strata | `occ1-clustered-skewed-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.5911 | [8.4599,8.6100] | 6.242 |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` † | `apeiron_table w1` | `wyhash w1` | 4.0784 | [3.8988,8.5328] | 6.450 |
| current-strata | `occ4-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.4284 | [8.4092,8.4408] | 6.282 |
| current-strata | `occ4-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.4231 | [8.4203,8.4238] | 6.280 |

The worst **stable** current-strata cells are `occ16-offset-beyond-direct-present-hit-*`
vs `hashtrinity_table` (10.22/10.02×, ~8.1 ns/op), reproduced in pass 2
(10.24/10.24×). The spill **absent-miss** cells that led the post-correctness
top-20 (20.5/18.2×) no longer appear; they fell to 4.39/4.44× (§8).

## 6 · Spatial failures (10 rows, none passing)

| cell | selected | incumbent | ratio | interval | concurrency | address_face | basis |
|---|---|---|---|---|---|---|---|
| `spatial/n100000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 1.2654 | [1.1046,1.4231] | matched | unmatched | parallel |
| `spatial/n100000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 2.6256 | [2.1731,2.7975] | MISMATCH | unmatched | serial |
| `spatial/n100000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.2810 | [1.2265,1.3082] | matched | unmatched | parallel |
| `spatial/n100000` | `apeiron_table w16` | `foldhash_par16 w16` | 2.9151 | [3.2415,3.4652] | matched | matched | parallel |
| `spatial/n100000` | `apeiron_table w16` | `fxhash_par16 w16` | 3.0204 | [3.1397,3.4449] | matched | matched | parallel |
| `spatial/n1000000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 2.7982 | [2.5884,2.8670] | matched | unmatched | parallel |
| `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 20.0474 | [19.8189,20.8308] | MISMATCH | unmatched | serial |
| `spatial/n1000000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.1893 | [1.1302,1.3046] | matched | unmatched | parallel |
| `spatial/n1000000` | `apeiron_table w16` | `foldhash_par16 w16` | 6.2851 | [5.8889,6.6411] | matched | matched | parallel |
| `spatial/n1000000` | `apeiron_table w16` | `fxhash_par16 w16` | 5.7893 | [5.4921,5.9296] | matched | matched | parallel |

The two `get_each` rows are the concurrency-unmatched (2); the four
`count_par`/`hashbrown_rayon` rows are the address-face-unmatched (4); the four
`foldhash_par16`/`fxhash_par16` rows are ratio failures. Unlike the post-correctness
run, `spatial/n100000` vs `count_par` now also exceeds the ratio limit (0.7953 →
1.2654) and fails on both gates; pass 2 measured 1.1634, also above 1.0.

## 7 · Best passing pairs (top-10 by ratio ascending)

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| current-strata | `occ8-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0582 | [0.0576,0.0596] | 1.337 |
| current-strata | `occ8-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0604 | [0.0601,0.0607] | 1.334 |
| current-strata | `occ16-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0612 | [0.0611,0.0614] | 1.344 |
| current-strata | `occ1-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0614 | [0.0606,0.0622] | 1.349 |
| current-strata | `occ16-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0615 | [0.0610,0.0622] | 1.352 |
| current-strata | `occ16-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0615 | [0.0609,0.0616] | 3.033 |
| current-strata | `occ2-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0623 | [0.0620,0.0627] | 3.011 |
| current-strata | `occ1-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0638 | [0.0617,0.0646] | 1.402 |
| current-strata | `occ1-direct-index-resident-existing-update-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0642 | [0.0640,0.0643] | 3.004 |
| current-strata | `occ1-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0643 | [0.0642,0.0643] | 3.007 |

All ten are resident direct-index paths. The absent-miss rows displaced the
`existing-update` rows that led the post-correctness receipt: the resident
absent-miss selected cost fell to ~1.33–1.40 ns/op.

## 8 · Post-correctness vs post-scatter on the previously-worst cells

`pre` is the post-correctness receipt (§5/§6/§8); `post` is pass 1; `post 2` is
pass 2. Same cell, incumbent and worker width. `ns/op = sel_median_ns / 4096`.
Ratios compare the selected side to a *re-measured* incumbent, so a ratio move is
not attributable to the source alone; the incumbent ns/op is given where it moved.

### Spill absent-miss

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval | post 2 ratio |
|---|---|---|---|---|---|---|---|
| `occ16-seeded-random-absent-miss-replacement` | `hashbrown_fxhash` | 20.4682 | 16.161 | 4.3922 | 3.675 | [4.3883,4.3915] | 4.5303 |
| `occ16-seeded-random-absent-miss-wrapping-add` | `hashbrown_fxhash` | 18.2012 | 15.993 | 4.4443 | 3.680 | [4.4440,4.4510] | 4.4529 |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashbrown_fxhash` | 19.1104 | 18.812 | 3.3645 | 3.353 | [3.3302,3.4225] | 3.4279 |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashbrown_fxhash` | 19.8637 | 18.923 | 3.4118 | 3.354 | [2.9292,3.4301] | 3.4249 |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashtrinity_table` | 12.8636 | 18.812 | 2.4425 | 3.353 | [2.4347,2.4561] | 2.4278 |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashtrinity_table` | 13.0705 | 18.923 | 2.4346 | 3.354 | [2.4286,2.4397] | 2.4356 |

The selected cost fell ~16.0–18.9 → ~3.35–3.68 ns/op (ratio ~18–20 → ~2.4–4.4).
This is the largest change in the receipt and reproduces across passes.

### Spill present-hit

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval | post 2 ratio |
|---|---|---|---|---|---|---|---|
| `occ16-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 12.2664 | 9.814 | 10.2197 | 8.119 | [10.1991,10.2151] | 10.2429 |
| `occ16-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 12.3244 | 10.092 | 10.0217 | 8.133 | [9.8928,10.1629] | 10.2404 |
| `occ16-seeded-random-present-hit-replacement` | `hashtrinity_table` | 9.6735 | 10.200 | 7.9158 | 8.428 | [7.9445,8.0205] | 7.8422 |
| `occ16-seeded-random-present-hit-wrapping-add` | `hashtrinity_table` | 9.7360 | 10.280 | 8.0865 | 8.402 | [7.9468,8.1921] | 8.0126 |
| `occ8-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 14.5840 | 10.931 | 8.8709 | 6.675 | [8.8691,8.9128] | 8.8545 |
| `occ8-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 14.6136 | 10.949 | 8.8014 | 6.674 | [8.7901,8.9352] | 8.8385 |
| `occ2-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 12.1343 | 8.645 | 8.6316 | 6.172 | [8.6266,8.6557] | 8.6434 |
| `occ2-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 12.2547 | 8.727 | 8.6612 | 6.185 | [8.6271,8.6680] | 8.6329 |
| `occ4-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 12.0674 | 8.697 | 8.4284 | 6.282 | [8.4092,8.4408] | 8.4117 |
| `occ4-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 12.1534 | 8.780 | 8.4231 | 6.280 | [8.4203,8.4238] | 8.3925 |
| `occ1-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 11.8644 | 9.098 | 8.7629 | 6.450 | [8.5870,8.6119] | 8.8303 |
| `occ1-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 11.7557 | 9.009 | 8.5953 | 6.155 | [8.5658,8.8288] | 8.5996 |

The selected cost fell at every occupancy (~8.6–10.9 → ~6.2–8.4 ns/op); the
incumbent `hashtrinity_table` side is stable (~0.73–1.07 ns/op). The largest
selected-side gain is `occ8` (~10.9 → ~6.67 ns/op). The `occ1` replacement row is
order-unstable (§4a): pass 2 selected 10.339 ns/op, ratio 9.8032 vs
`hashbrown_fxhash`.

### Growth-insert

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval | post 2 ratio |
|---|---|---|---|---|---|---|---|
| `occ16-direct-index-resident-growth-insert-replacement` | `hashtrinity_table` | 1.5364 | 12.561 | 2.3943 | 12.173 | [2.2984,2.5230] | 2.4066 |
| `occ16-direct-index-resident-growth-insert-wrapping-add` | `hashtrinity_table` | 1.5227 | 12.571 | 2.3358 | 12.051 | [2.3374,2.4518] | 2.4543 |

The selected cost held (~12.56 → ~12.0–12.2 ns/op). The ratio rose ~1.53 → ~2.34–2.39
because the re-measured `hashtrinity_table` incumbent fell ~8.18 → ~5.0–5.1 ns/op
(derived from `ratio` and `sel ns/op`). This is a cross-session incumbent drift, not
a selected-side regression.

### Resident absent-miss

| cell | incumbent | pre ratio | pre ns/op | post ratio | post ns/op | post interval | post 2 ratio |
|---|---|---|---|---|---|---|---|
| `occ16-direct-index-resident-absent-miss-replacement` | `hashtrinity_table` | 4.9286 | 2.645 | 2.4699 | 1.344 | [2.4650,2.4828] | 2.4401 |
| `occ16-direct-index-resident-absent-miss-replacement` | `hashbrown_foldhash` | 1.9998 | 2.645 | 0.9616 | 1.344 | [0.9510,0.9815] | 0.9753 |

The resident absent-miss selected cost halved (~2.65 → ~1.34 ns/op); the pair against
`hashbrown_foldhash` now passes (`<= 1.0`) in both passes.

### Spatial

| cell | selected | incumbent | pre ratio | post ratio | post interval | post 2 ratio |
|---|---|---|---|---|---|---|
| `spatial/n100000` | `w24` | `hashtrinity_count_par w24` | 0.7953 | 1.2654 | [1.1046,1.4231] | 1.1634 |
| `spatial/n100000` | `w1` | `hashtrinity_get_each w24` | 2.6974 | 2.6256 | [2.1731,2.7975] | 2.3892 |
| `spatial/n100000` | `w32` | `hashbrown_rayon w32` | 1.1167 | 1.2810 | [1.2265,1.3082] | 1.4814 |
| `spatial/n100000` | `w16` | `foldhash_par16 w16` | 2.6115 | 2.9151 | [3.2415,3.4652] | 2.7230 |
| `spatial/n100000` | `w16` | `fxhash_par16 w16` | 3.1125 | 3.0204 | [3.1397,3.4449] | 3.2467 |
| `spatial/n1000000` | `w24` | `hashtrinity_count_par w24` | 2.8653 | 2.7982 | [2.5884,2.8670] | 2.8411 |
| `spatial/n1000000` | `w1` | `hashtrinity_get_each w24` | 23.0538 | 20.0474 | [19.8189,20.8308] | 19.5996 |
| `spatial/n1000000` | `w32` | `hashbrown_rayon w32` | 1.0689 | 1.1893 | [1.1302,1.3046] | 1.1984 |
| `spatial/n1000000` | `w16` | `foldhash_par16 w16` | 5.6274 | 6.2851 | [5.8889,6.6411] | 6.2174 |
| `spatial/n1000000` | `w16` | `fxhash_par16 w16` | 6.0770 | 5.7893 | [5.4921,5.9296] | 5.9478 |

Spatial ratios are mixed and within ~±25% of the prior session except the two
`count_par` rows, which are the address-face-unmatched rows regardless of ratio.
The `n100000` selected wall cost held ~1104 → ~1096 ns across passes; the derived
incumbent is ~1104/1.2654 = ~872 ns (pass 1) and ~1096/1.1634 = ~942 ns (pass 2).
The post-correctness receipt did not record this row's selected ns, so its incumbent
side is not derivable; the ratio rise is consistent with a faster incumbent.

## 9 · Per-incumbent tallies (load-bearing; 160 serial + 2 spatial cells each)

Pass 1:

| incumbent | pass | fail | n/a | info | post-correctness pass/fail |
|---|---|---|---|---|---|
| `hashtrinity_table` | 28 | 132 | 2 | 0 | 28 / 132 |
| `hashbrown_foldhash` | 25 | 135 | 2 | 0 | 22 / 138 |
| `hashbrown_fxhash` | 0 | 160 | 2 | 0 | 0 / 160 |
| `std_siphash` | 123 | 37 | 2 | 0 | 92 / 68 |
| `xxhash_xxh3` | 152 | 8 | 2 | 0 | 144 / 16 |
| `xxhash_xxh64` | 54 | 106 | 2 | 0 | 42 / 118 |
| `wyhash` | 20 | 140 | 2 | 0 | 13 / 147 |
| `ahash` | 43 | 117 | 2 | 0 | 34 / 126 |
| `fnv` | 55 | 105 | 2 | 0 | 46 / 114 |
| `hashtrinity_count_par` | 0 | 2 | 160 | 0 | 0 / 2 |
| `hashtrinity_get_each` | 0 | 2 | 160 | 0 | 0 / 2 |
| `hashbrown_rayon` | 0 | 2 | 160 | 0 | 0 / 2 |
| `foldhash_par16` | 0 | 2 | 160 | 0 | 0 / 2 |
| `fxhash_par16` | 0 | 2 | 160 | 0 | 0 / 2 |
| `blake3` | 0 | 0 | 0 | 162 | 0 / 0 |
| `sha256` | 0 | 0 | 0 | 162 | 0 / 0 |

Every incumbent except `hashtrinity_table` and `hashbrown_fxhash` gained passes;
`hashtrinity_table` is unchanged at 28/132.

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
`acceptance-address-face-mismatch`), `placement-only-address` (×2),
`fabricated-current-cell` (×2), and 16× `retired-shardfull-receipt` + 16×
`retired-shardfull-digest` (all red). Identical set to the post-correctness receipt.

## 11 · Trust and measurement boundary

- **Measured**: strict per-pair `selected/incumbent` ratio on this machine,
  toolchain and dirty-source identity; the 2592-row pair table; the 55 control
  observations; the fail counts and the worst/best pairs above; the derived
  `ns/op = median_ns / 4096`; the pass-2 reproducibility check (1.1% status flips).
- **Not measured / out of scope**: `cargo test`, `cargo clippy`, `cargo fmt`,
  `cargo kani`, `proof/regen.sh`, Lean builds, coverage census, the `10^8`
  identity row, cross-process determinism, codec round-trip, `remove`, SIMD,
  allocator behaviour, rustc/LLVM codegen, kernel ABI, GPU/SPIR-V, and the
  unextracted `std::thread::scope` spatial orchestration. `report_complete=false`,
  `reg1_pass=false`. The extraction identity (`ApeironWord.lean` `4c7bb34f…`) is
  recorded but its freshness was not re-run here.
- **Grade**: `measured` for the gate result; the digests and log hashes are identity
  receipts. The 55 controls establish the harness rejects corrupted observations,
  not that the table is correct. The two-pass comparison bounds, but does not
  eliminate, run-to-run variance; the host was not fully idle (§2).
- **Cell semantics**: no cell showed a semantically wrong value (wrong operation,
  wrong address face, wrong status class). The `occ1` present-hit and `xxhash_xxh3`
  swings are timing/ordering effects, not semantics; no Sol escalation triggered.

## 12 · Residual and next bounded task

- Residual (uncovered): the raw 2690-line pair tables are retained only at the
  `/tmp` log paths with their sha256 in §2; durable retention is not included here.
  The per-cell `strata_summary`/`acceptance_summary` lines are `--json`-only and
  were not captured in table mode. The `occ1` present-hit order-instability is
  characterized but not root-caused; it affects three cells and one pair in the
  HEADLINE.
- Next bounded task: attribute the stable worst current-strata ratio — the spill
  **present-hit** path `occ16-offset-beyond-direct-present-hit-*` vs
  `hashtrinity_table` (~10.2×, ~8.1 ns/op, reproduced) and `occ8-*` (~8.8×,
  ~6.7 ns/op) — to a named `src/table.rs`/`src/table/scan.rs` region, with a
  focused before/after measurement on the same 4096-op schedule. Secondary: the
  `spatial/n100000` `hashtrinity_count_par` address-face row now also exceeds the
  ratio limit. Do not weaken the `<= 1.0` contract.
