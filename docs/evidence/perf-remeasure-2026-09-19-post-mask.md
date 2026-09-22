# REG-1 performance remeasurement, post summary-mask (APEIRON-PERF-REMEASURE-T5)

Fresh strict-gate measurement on the current source, after the `least_set_lane`
halving search in `spill_read` (`src/table/scan.rs`), the power-of-two summary
mask fast path (`src/table/state.rs`) and the single-key-qualification fix in
`src/table/parallel.rs` (`get_each`). The prior receipt
([`perf-remeasure-2026-09-19-post-scatter.md`](perf-remeasure-2026-09-19-post-scatter.md))
binds `src/table.rs` sha256 `5f260801…` and HEADLINE `pass=500 fail=950`; the
executing source is now `e568d0c3…`. This receipt supersedes the post-scatter
numbers for the current identity only.

Three full passes ran on one build. Pass 1 (the command below) is the primary
receipt; passes 2 and 3 are the reproducibility control in §4a. The three
HEADLINEs are `pass=561`, `pass=540`, `pass=547`.

## 1 · Binding

Dirty tree, not HEAD. `apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`.

| artifact | sha256 |
|---|---|
| `src/table.rs` | `e568d0c318801304d17ad304a1f719d509de28b35b33764ad6980619e3e4856d` |
| `src/table/state.rs` | `8dad06425b7ae68b47657e97d7f31ece917b0dd87b7c520d7e1b4f321bf64f8a` |
| `src/table/scan.rs` | `ec633507a8ad456a7cb11553f559cf1abc08d3d4984a118e145eb5aa1ad91e97` |
| `src/table/parallel.rs` | `bd843ddcf5fb8074736a52ffbca4fa0bc1bf872839d770100e54c81f59ab1290` |
| `src/table/partition.rs` | `7cf4724447b5ef8de8c4d4c2a175432d2284e3c1152b1f41c7f374cd13688c64` |
| `proof/ApeironWord.lean` | `9af54e7d2467a5a19ebccfd7fd70691072a17fda76e78e2366c3832675749d3e` |
| `examples/reg1/main.rs` | `8c1787fe63bac5d225a53f4535d190a009e6f5480284c1a55e222412ae573268` |
| `examples/reg1/performance.rs` | `ea77fbe2f2cc1248283624f5153eae2669cff4bfeb6efe03c104dc4508c01930` |
| `examples/reg1/acceptance.rs` | `d2f8c0346a4b2b3a82498c706f8dcf721e0f1df65902a942973c5d28f2326eb2` |
| `examples/reg1/strata.rs` | `98cfd2be21eb5a94587d00a90f2dfcf7c446951c97300d1ca5510ab36c01b777` |
| `examples/reg1/matrix.rs` | `405351ee86cc9c4cfb003d11f62dfaf3cb518ce314a9df29a4aef205bf87cf90` |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` |
| `Cargo.toml` | `bf5d26106064e944b620e9922a2e30004b9d24ef8dffb0b8698c1f678fd91654` |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |

Changed since the post-scatter receipt: `src/table.rs` (`5f260801…` →
`e568d0c3…`), `src/table/state.rs` (`79983466…` → `8dad0642…`), `src/table/scan.rs`
(`698b1eca…` → `ec633507…`), `src/table/parallel.rs` (`b22650c1…` → `bd843ddc…`),
`Cargo.toml` (`9d445226…` → `bf5d2610…`). `src/table/partition.rs` (`7cf47244…`)
is unchanged. The `Cargo.toml` delta adds the off-by-default `read-stats` feature
(§11). The reg1 harness sources and `Cargo.lock` are byte-identical to the
post-scatter receipt. The run writes none of these files. `src/table.rs` changed
beyond the three named edits; the receipt binds its digest and does not attribute
the delta.

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
| run 1 log | `/tmp/opencode/reg1-full-post-mask-20260920-080201.log`, sha256 `157b0108470b1f163c4af858a1839b527a12429f834528fe2f1faaef692eb618`, 2690 lines |
| run 2 log | `/tmp/opencode/reg1-full-post-mask-20260920-080329.log`, sha256 `2848212bab6f9af51b9fe7a57e5f089ffd93a3b42d22e1e6d3d7dcd80d47c6f8`, 2690 lines |
| run 3 log | `/tmp/opencode/reg1-full-post-mask-20260920-080535.log`, sha256 `2e2f4f0e9e360e5b7bb1c30e1669d92d5f55ec91ebdd6891eb36579c97de0bb7`, 2690 lines |
| build log | `/tmp/opencode/reg1-build-post-mask-20260920-080157.log`, sha256 `bdaeb7034df08ce2bb2245b7eabab478ab7d49488257a1a30b7056f5b4e51f0e`, 30 lines |

No concurrent cargo/Lean/Kani/coverage build ran; `pgrep` found none before run 1.
A `btrfs-cleaner` kernel thread and a desktop session were resident, so the host
was not fully idle: load average `1.00/1.60/1.67` before run 1, `1.85/1.95/1.80`
after run 3.

## 3 · Commands, exit codes, elapsed

All heavy commands ran under `nice -n 10` and a hard `timeout`.

```sh
# 1. build (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 1800 cargo build --release --locked --example reg1
# exit 0; wall 0:02.42; user 11.55 s; max RSS 554484 KiB; "Finished in 2.37s"

# 2. strict full gate, pass 1 (cwd /home/socol/Workspace/apeiron)
/usr/bin/time -v nice -n 10 timeout 3600 \
  cargo run --release --locked --example reg1 -- \
  --check performance --profile full --accept --controls \
  > /tmp/opencode/reg1-full-post-mask-20260920-080201.log 2>&1
# exit 1 (strict acceptance failure); wall 0:01:25.97; user 173.78 s; system 8.28 s;
# 211% CPU; max RSS 782996 KiB. timeout 3600 NOT reached.

# 3. strict full gate, pass 2 (reproducibility control)
#    same command -> /tmp/opencode/reg1-full-post-mask-20260920-080329.log
# exit 1; wall 0:01:25.79; user 171.02 s; system 8.67 s; 209% CPU; max RSS 868028 KiB.

# 4. strict full gate, pass 3 (reproducibility control)
#    same command -> /tmp/opencode/reg1-full-post-mask-20260920-080535.log
# exit 1; wall 0:01:25.74; user 174.79 s; system 8.56 s; 213% CPU; max RSS 738904 KiB.
```

Non-fatal cargo warning: `error: duplicate key` from the hashtrinity git checkout
`opsfold/Cargo.toml:61`; it precedes `Finished` and does not set the exit status.

## 4 · HEADLINE

```text
HEADLINE registered=2592 load_bearing=2268 applicable=1450 pass=561 fail=889 missing=0 concurrency_unmatched=2 address_face_unmatched=4 full_acceptance=false
```

Verdict: **fail** (`full_acceptance=false`). `missing=0`; 2 concurrency-unmatched,
4 address-face-unmatched, the other 883 are ratio failures. Against the
post-scatter receipt (`pass=500 fail=950`, same `registered`/`load_bearing`/
`applicable`): `pass` rose 500 → 561, `fail` fell 950 → 889 on pass 1.

Metric: per-pair `selected/incumbent` ratio of pooled **median elapsed ns** over
one deterministic serial schedule (`STRATA_OPS = 4096` registered ops,
`campaigns=3 warmup=2 samples=15`), interval = min/max campaign ratio; acceptance
requires interval max `<= 1.0`. `ns/op = median_ns / 4096` for serial rows.
Spatial rows carry raw elapsed wall ns for the packed operation, not ns/op; their
ratio is `selected_wall_ns / incumbent_wall_ns`. The pooled point ratio can fall
outside the campaign min/max interval; acceptance reads the interval max.

`acceptance_phase=per-pair profile=full registered_cells=160 spatial_cells=2 json=false`.

### 4a · Reproducibility (passes 2 and 3)

| pass | pass | fail | status flips vs pass 1 |
|---|---|---|---|
| 1 | 561 | 889 | — |
| 2 | 540 | 910 | 47 |
| 3 | 547 | 903 | 54 |

The three-run `pass` range is 540–561 (mean 549.3), 21 pairs wide (1.4% of
`applicable`). The post-scatter control was 1 pair wide (500/501); run-to-run
variance is larger on this identity. Every post-mask run still exceeds every
post-scatter run by at least 39 pairs. The selected-side median moved more than
20% on the four `occ1-*` present-hit cells, marked `†` in §5:

| cell | pass 1 sel ns/op | pass 2 sel ns/op |
|---|---|---|
| `occ1-direct-index-resident-present-hit-replacement` | 1.903 | 3.363 |
| `occ1-direct-index-resident-present-hit-wrapping-add` | 1.768 | 2.289 |
| `occ1-offset-beyond-direct-present-hit-replacement` | 7.192 | 12.166 |
| `occ1-offset-beyond-direct-present-hit-wrapping-add` | 7.346 | 10.318 |

42 of the 47 pass-1/pass-2 flips are pairs whose ratio sits within ~7% of 1.0
(0.93–1.07), where absolute noise changes the verdict. The 5 far flips are
`occ1-direct-index-resident-present-hit-*` vs `xxhash_xxh64` (0.75/0.72 pass →
1.13/0.90 fail), `occ1-clustered-skewed-present-hit-replacement` vs
`xxhash_xxh3` (0.33 pass → 1.07 fail), and
`occ16-direct-index-resident-existing-update-*` vs `hashtrinity_table`
(1.73/1.76 fail → 0.66/0.66 pass). The `occ1` present-hit selected cost is
order-unstable, so its rows are not a stable ranking signal.

## 5 · Worst failing pairs (top-20 by interval max, pass 1)

`ns/op` is `sel_median_ns / 4096` for serial rows; spatial carries raw ms. `†`
marks an order-unstable selected cell (§4a).

| family | cell | selected | incumbent | ratio | interval | sel |
|---|---|---|---|---|---|---|
| spatial | `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 17.5729 | [17.4485,17.7193] | 276.930 ms |
| current-strata | `occ1-offset-beyond-direct-present-hit-wrapping-add` † | `apeiron_table w1` | `hashtrinity_table w1` | 10.0696 | [9.9751,10.1370] | 7.346 ns/op |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` † | `apeiron_table w1` | `hashtrinity_table w1` | 10.0365 | [10.0124,10.0526] | 7.192 ns/op |
| current-strata | `occ1-clustered-skewed-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 10.0187 | [9.9006,9.9494] | 7.328 ns/op |
| current-strata | `occ1-clustered-skewed-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 9.9973 | [9.8944,9.9322] | 7.317 ns/op |
| current-strata | `occ2-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 9.7508 | [9.6861,9.8460] | 7.004 ns/op |
| current-strata | `occ2-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 9.7848 | [9.7247,9.8134] | 7.028 ns/op |
| current-strata | `occ4-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.9191 | [8.9108,8.9334] | 6.672 ns/op |
| current-strata | `occ4-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.8887 | [8.8754,8.9129] | 6.651 ns/op |
| current-strata | `occ8-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.7307 | [8.7173,8.8315] | 6.586 ns/op |
| current-strata | `occ8-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.7487 | [8.7277,8.7586] | 6.570 ns/op |
| current-strata | `occ16-offset-beyond-direct-present-hit-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 8.2160 | [8.1221,8.2785] | 6.908 ns/op |
| current-strata | `occ16-offset-beyond-direct-present-hit-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 8.2200 | [8.0886,8.2396] | 6.887 ns/op |
| current-strata | `occ1-offset-beyond-direct-present-hit-wrapping-add` † | `apeiron_table w1` | `hashbrown_fxhash w1` | 6.9600 | [6.1405,6.9713] | 7.346 ns/op |
| current-strata | `occ1-clustered-skewed-present-hit-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 6.9541 | [6.7226,6.9661] | 7.317 ns/op |
| current-strata | `occ1-clustered-skewed-present-hit-wrapping-add` | `apeiron_table w1` | `hashbrown_fxhash w1` | 6.9498 | [5.8053,6.9553] | 7.328 ns/op |
| current-strata | `occ1-offset-beyond-direct-present-hit-replacement` † | `apeiron_table w1` | `hashbrown_fxhash w1` | 6.8203 | [5.8779,6.9211] | 7.192 ns/op |
| current-strata | `occ1-direct-index-resident-admitted-fresh-insert-wrapping-add` | `apeiron_table w1` | `hashtrinity_table w1` | 6.7675 | [6.7271,6.8236] | 1.698 ns/op |
| current-strata | `occ1-direct-index-resident-admitted-fresh-insert-replacement` | `apeiron_table w1` | `hashtrinity_table w1` | 6.7678 | [6.7055,6.8193] | 1.694 ns/op |
| current-strata | `occ2-clustered-skewed-present-hit-replacement` | `apeiron_table w1` | `hashbrown_fxhash w1` | 6.7007 | [6.6827,6.7615] | 7.023 ns/op |

The worst **stable** current-strata cells are the spill present-hit rows at
`occ2`/`occ4`/`occ8`/`occ16` vs `hashtrinity_table` (8.22–9.78×, ~6.6–7.0 ns/op),
reproduced in passes 2 and 3. The two
`occ1-direct-index-resident-admitted-fresh-insert-*` rows (6.77×) are not new:
the post-scatter receipt measured 6.86/6.92×, so they only entered the top-20
because the `occ1` present-hit rows above them reordered. The spill absent-miss
cells that led the post-correctness top-20 remain off this list.

## 6 · Spatial rows (raw wall ns; all 10 rows fail)

| cell | selected | incumbent | ratio | interval | concurrency | address_face |
|---|---|---|---|---|---|---|
| `spatial/n100000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 1.3786 | [1.0584,1.8408] | matched | unmatched |
| `spatial/n100000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 2.3105 | [2.4786,2.7020] | MISMATCH | unmatched |
| `spatial/n100000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.2626 | [1.0639,1.3597] | matched | unmatched |
| `spatial/n100000` | `apeiron_table w16` | `foldhash_par16 w16` | 2.8983 | [2.6416,3.0252] | matched | matched |
| `spatial/n100000` | `apeiron_table w16` | `fxhash_par16 w16` | 2.8851 | [2.7775,3.2596] | matched | matched |
| `spatial/n1000000` | `apeiron_table w24` | `hashtrinity_count_par w24` | 2.6677 | [2.5128,2.6803] | matched | unmatched |
| `spatial/n1000000` | `apeiron_table w1` | `hashtrinity_get_each w24` | 17.5729 | [17.4485,17.7193] | MISMATCH | unmatched |
| `spatial/n1000000` | `apeiron_table w32` | `hashbrown_rayon w32` | 1.1487 | [1.0580,1.1652] | matched | unmatched |
| `spatial/n1000000` | `apeiron_table w16` | `foldhash_par16 w16` | 5.8085 | [5.4319,6.2537] | matched | matched |
| `spatial/n1000000` | `apeiron_table w16` | `fxhash_par16 w16` | 5.3499 | [5.2835,5.6112] | matched | matched |

The two `get_each` rows are the concurrency-unmatched (2); the four
`count_par`/`hashbrown_rayon` rows are the address-face-unmatched (4); the four
`foldhash_par16`/`fxhash_par16` rows are ratio failures. The `n100000`
`count_par` ratio is unstable across passes (1.379/0.967/1.025) and is
address-face-unmatched regardless of ratio. The `n1000000` `hashbrown_rayon`
ratio is likewise unstable (1.149/0.980/1.094). The `get_each` selected wall
cost fell 308.9 → 276.9 ms (pass 1; 273.4/269.0 in passes 2/3), consistent with
the `get_each` fix, but the row is concurrency-MISMATCH and therefore not
applicable.

## 7 · Best passing pairs (top-10 by ratio ascending, pass 1)

| family | cell | selected | incumbent | ratio | interval | sel ns/op |
|---|---|---|---|---|---|---|
| current-strata | `occ1-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0595 | [0.0594,0.0602] | 3.006 |
| current-strata | `occ8-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0604 | [0.0603,0.0606] | 1.329 |
| current-strata | `occ2-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0606 | [0.0606,0.0610] | 1.329 |
| current-strata | `occ2-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0607 | [0.0606,0.0610] | 1.332 |
| current-strata | `occ16-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0607 | [0.0606,0.0608] | 1.332 |
| current-strata | `occ8-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0608 | [0.0605,0.0642] | 1.334 |
| current-strata | `occ16-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0608 | [0.0607,0.0610] | 1.335 |
| current-strata | `occ4-direct-index-resident-absent-miss-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0609 | [0.0609,0.0610] | 1.333 |
| current-strata | `occ4-direct-index-resident-absent-miss-wrapping-add` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0609 | [0.0605,0.0612] | 1.331 |
| current-strata | `occ4-direct-index-resident-existing-update-replacement` | `apeiron_table w1` | `xxhash_xxh3 w1` | 0.0610 | [0.0610,0.0612] | 3.025 |

All ten are resident direct-index paths against `xxhash_xxh3`. The resident
absent-miss selected cost is ~1.33 ns/op, unchanged from post-scatter.

## 8 · Post-scatter vs post-mask on the previously-worst cells

`pre` is the post-scatter receipt; `post` is pass 1; `post 2` is pass 2. Same
cell, incumbent and worker width. `ns/op = sel_median_ns / 4096`. Ratios compare
the selected side to a *re-measured* incumbent, so a ratio move is not
attributable to the source alone; the incumbent ns/op is given where it moved.

### Spill present-hit

| cell | incumbent | pre ratio | pre sel | post ratio | post sel | post interval | post 2 sel |
|---|---|---|---|---|---|---|---|
| `occ16-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 10.2197 | 8.119 | 8.2200 | 6.887 | [8.0886,8.2396] | 6.544 |
| `occ16-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 10.0217 | 8.133 | 8.2160 | 6.908 | [8.1221,8.2785] | 6.548 |
| `occ8-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 8.8709 | 6.675 | 8.7307 | 6.586 | [8.7173,8.8315] | 6.586 |
| `occ8-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 8.8014 | 6.674 | 8.7487 | 6.570 | [8.7277,8.7586] | 6.572 |
| `occ4-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 8.4284 | 6.282 | 8.9191 | 6.672 | [8.9108,8.9334] | 6.677 |
| `occ4-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 8.4231 | 6.280 | 8.8887 | 6.651 | [8.8754,8.9129] | 6.707 |
| `occ2-offset-beyond-direct-present-hit-replacement` | `hashtrinity_table` | 8.6316 | 6.172 | 9.7848 | 7.028 | [9.7247,9.8134] | 7.054 |
| `occ2-offset-beyond-direct-present-hit-wrapping-add` | `hashtrinity_table` | 8.6612 | 6.185 | 9.7508 | 7.004 | [9.6861,9.8460] | 6.999 |
| `occ1-offset-beyond-direct-present-hit-replacement` † | `hashtrinity_table` | 8.7629 | 6.450 | 10.0365 | 7.192 | [10.0124,10.0526] | 12.166 |
| `occ1-offset-beyond-direct-present-hit-wrapping-add` † | `hashtrinity_table` | 8.5953 | 6.155 | 10.0696 | 7.346 | [9.9751,10.1370] | 10.318 |

The selected cost improves only at `occ16` (8.12 → 6.89 ns/op, −15.2%); `occ8` is
flat (6.67 → 6.59); `occ4`, `occ2` and `occ1` regress (6.28 → 6.67, 6.17 → 7.03,
6.45 → 7.19). The incumbent `hashtrinity_table` side is stable at 0.72–0.84
ns/op, so the ratio follows the selected side. The micro-benchmark's present-hit
−15.6% holds on this schedule only at `occ16`; the low-occupancy cells are not
improved. The `occ1` rows are order-unstable (§4a).

### Spill absent-miss

| cell | incumbent | pre ratio | pre sel | post ratio | post sel | post interval | post 2 sel |
|---|---|---|---|---|---|---|---|
| `occ16-seeded-random-absent-miss-replacement` | `hashbrown_fxhash` | 4.3922 | 3.675 | 3.3602 | 2.794 | [3.3534,3.3810] | 2.789 |
| `occ16-seeded-random-absent-miss-wrapping-add` | `hashbrown_fxhash` | 4.4443 | 3.680 | 3.3564 | 2.791 | [3.3418,3.3802] | 2.789 |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashbrown_fxhash` | 3.3645 | 3.353 | 2.5141 | 2.476 | [2.5063,2.5226] | 2.474 |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashbrown_fxhash` | 3.4118 | 3.354 | 2.4972 | 2.476 | [2.4957,2.5021] | 2.474 |
| `occ16-offset-beyond-direct-absent-miss-replacement` | `hashtrinity_table` | 2.4425 | 3.353 | 1.7970 | 2.476 | [1.7873,1.8043] | 2.474 |
| `occ16-offset-beyond-direct-absent-miss-wrapping-add` | `hashtrinity_table` | 2.4346 | 3.354 | 1.7987 | 2.476 | [1.7898,1.8042] | 2.474 |

The selected cost fell 3.675 → 2.794 (−24.0%) and 3.353 → 2.476 (−26.2%) ns/op,
reproducing in passes 2 and 3. This matches the −23.9% absent-miss micro claim.
The `hashbrown_fxhash` incumbent is stable at ~0.83–0.99 ns/op.

### Growth-insert

| cell | incumbent | pre ratio | pre sel | pre inc | post ratio | post sel | post inc | post 2 ratio |
|---|---|---|---|---|---|---|---|---|
| `occ16-direct-index-resident-growth-insert-replacement` | `hashtrinity_table` | 2.3943 | 12.173 | 5.084 | 1.4629 | 11.880 | 8.121 | 2.1049 |
| `occ16-direct-index-resident-growth-insert-wrapping-add` | `hashtrinity_table` | 2.3358 | 12.051 | 5.159 | 1.4575 | 11.881 | 8.152 | 2.0941 |

The selected cost holds (12.17 → 11.88 ns/op). The ratio move 2.39 → 1.46 is
incumbent-side: the re-measured `hashtrinity_table` rose 5.08 → 8.12 ns/op on
pass 1 and fell to ~5.6 on pass 2 (ratio 2.10). This is cross-session incumbent
drift, not a selected-side change.

### Resident absent-miss

| cell | incumbent | pre ratio | pre sel | post ratio | post sel | post interval | post 2 ratio |
|---|---|---|---|---|---|---|---|
| `occ16-direct-index-resident-absent-miss-replacement` | `hashtrinity_table` | 2.4699 | 1.344 | 2.4757 | 1.332 | [2.4721,2.4749] | 2.4660 |
| `occ16-direct-index-resident-absent-miss-replacement` | `hashbrown_foldhash` | 0.9616 | 1.344 | 0.9416 | 1.332 | [0.9329,0.9501] | 1.0395 |
| `occ16-direct-index-resident-absent-miss-wrapping-add` | `hashbrown_foldhash` | 0.9859 | 1.352 | 0.9468 | 1.335 | [0.9355,0.9511] | 1.0286 |

The selected cost is unchanged (~1.33 ns/op). The pair against
`hashbrown_foldhash` passes on pass 1 (0.942/0.947) but fails on pass 2
(1.040/1.029): the two sides are within ~5%, so this pair is not a stable signal.

### Spatial

Raw elapsed wall ns (not ns/op); `sel`/`inc` are pass-1 wall ns.

| cell | incumbent | pre ratio | post ratio | post sel | post inc | post 2 ratio |
|---|---|---|---|---|---|---|
| `spatial/n100000` | `hashtrinity_count_par w24` | 1.2654 | 1.3786 | 4.301 ms | 3.119 ms | 0.9669 |
| `spatial/n100000` | `hashtrinity_get_each w24` | 2.6256 | 2.3105 | 8.947 ms | 3.872 ms | 2.0179 |
| `spatial/n100000` | `hashbrown_rayon w32` | 1.2810 | 1.2626 | 3.915 ms | 3.101 ms | 1.3093 |
| `spatial/n100000` | `foldhash_par16 w16` | 2.9151 | 2.8983 | 3.179 ms | 1.097 ms | 2.8091 |
| `spatial/n100000` | `fxhash_par16 w16` | 3.0204 | 2.8851 | 2.881 ms | 0.999 ms | 2.8763 |
| `spatial/n1000000` | `hashtrinity_count_par w24` | 2.7982 | 2.6677 | 41.701 ms | 15.632 ms | 2.5032 |
| `spatial/n1000000` | `hashtrinity_get_each w24` | 20.0474 | 17.5729 | 276.930 ms | 15.759 ms | 17.7872 |
| `spatial/n1000000` | `hashbrown_rayon w32` | 1.1893 | 1.1487 | 46.803 ms | 40.745 ms | 0.9800 |
| `spatial/n1000000` | `foldhash_par16 w16` | 6.2851 | 5.8085 | 49.585 ms | 8.537 ms | 5.3684 |
| `spatial/n1000000` | `fxhash_par16 w16` | 5.7893 | 5.3499 | 45.195 ms | 8.448 ms | 5.4976 |

The prior receipt stated "~1104 ns" for the `n100000` `count_par` selected wall;
that figure is `sel_median_ns / 4096` (4522358 / 4096 = 1104.1), not the raw
elapsed wall ns (4.52 ms). This receipt reports the raw measurement. Ratios are
mixed and within ~±25% of post-scatter; the `get_each` rows are
concurrency-MISMATCH and the `count_par`/`hashbrown_rayon` rows are
address-face-unmatched, so no spatial row is applicable.

## 9 · Per-incumbent tallies (load-bearing; 160 serial + 2 spatial cells each)

Pass 1:

| incumbent | pass | fail | n/a | info | post-scatter pass/fail |
|---|---|---|---|---|---|
| `hashtrinity_table` | 32 | 128 | 2 | 0 | 28 / 132 |
| `hashbrown_foldhash` | 26 | 134 | 2 | 0 | 25 / 135 |
| `hashbrown_fxhash` | 0 | 160 | 2 | 0 | 0 / 160 |
| `std_siphash` | 128 | 32 | 2 | 0 | 123 / 37 |
| `xxhash_xxh3` | 156 | 4 | 2 | 0 | 152 / 8 |
| `xxhash_xxh64` | 73 | 87 | 2 | 0 | 54 / 106 |
| `wyhash` | 20 | 140 | 2 | 0 | 20 / 140 |
| `ahash` | 47 | 113 | 2 | 0 | 43 / 117 |
| `fnv` | 79 | 81 | 2 | 0 | 55 / 105 |
| `hashtrinity_count_par` | 0 | 2 | 160 | 0 | 0 / 2 |
| `hashtrinity_get_each` | 0 | 2 | 160 | 0 | 0 / 2 |
| `hashbrown_rayon` | 0 | 2 | 160 | 0 | 0 / 2 |
| `foldhash_par16` | 0 | 2 | 160 | 0 | 0 / 2 |
| `fxhash_par16` | 0 | 2 | 160 | 0 | 0 / 2 |
| `blake3` | 0 | 0 | 0 | 162 | 0 / 0 |
| `sha256` | 0 | 0 | 0 | 162 | 0 / 0 |

Every incumbent except `hashbrown_fxhash` and `wyhash` gained passes;
`hashbrown_foldhash` gained one, `hashtrinity_table` four, `fnv` 24.

## 10 · Controls

55 control observations, 0 green (`observed=red`), identical in all three passes
and to the post-scatter receipt: the 19 acceptance/registration controls
(`wrong-workload-identity`, `invalid-samples`, `absent-samples`,
`cached-home-only-collision`, `wrong-oracle-digest`, `deleted-required-current-cell`,
`selected-implementation-mismatch`, `strict-acceptance-single-failing-cell`,
`spatial-wrong-packed-input`, `spatial-wrong-bucket-routing`,
`spatial-wrong-worker-width`, `acceptance-failing-non-hashtrinity-pair`,
`acceptance-omitted-incumbent-pair`, `acceptance-unauthorized-not-applicable`,
`acceptance-crossing-ratio-interval`, `acceptance-wrong-oracle-digest`,
`acceptance-serial-adapter-wrong-result`, `acceptance-concurrency-mismatch`,
`acceptance-address-face-mismatch`), `placement-only-address` (×2),
`fabricated-current-cell` (×2), and 16× `retired-shardfull-receipt` + 16×
`retired-shardfull-digest` (all red).

## 11 · Trust and measurement boundary

- **Measured**: strict per-pair `selected/incumbent` ratio on this machine,
  toolchain and dirty-source identity; the 2592-row pair table; the 55 control
  observations; the fail counts and the worst/best pairs above; the derived
  `ns/op = median_ns / 4096`; the three-pass reproducibility check (47–57 status
  flips, 1.8–2.2%).
- **Not measured / out of scope**: `cargo test`, `cargo clippy`, `cargo fmt`,
  `cargo kani`, `proof/regen.sh`, Lean builds, coverage census, the `10^8`
  identity row, cross-process determinism, codec round-trip, `remove`, SIMD,
  allocator behaviour, rustc/LLVM codegen, kernel ABI, GPU/SPIR-V, and the
  unextracted `std::thread::scope` spatial orchestration. `report_complete=false`,
  `reg1_pass=false`. The extraction identity (`ApeironWord.lean` `9af54e7d…`) is
  recorded but its freshness was not re-run here.
- **Timed path is counter-free**: `Cargo.toml` adds the `read-stats` feature, off
  by default; the `--release --locked --example reg1` invocation enables no
  feature, so the `thread_local` counters in `src/table.rs` are compiled out of
  the timed path.
- **Grade**: `measured` for the gate result; the digests and log hashes are
  identity receipts. The 55 controls establish the harness rejects corrupted
  observations, not that the table is correct. The three-pass comparison bounds,
  but does not eliminate, run-to-run variance; the host was not fully idle (§2)
  and the 21-pair spread exceeds the post-scatter control's 1-pair spread.
- **Cell semantics**: no cell showed a semantically wrong value (wrong operation,
  wrong address face, wrong status class). The `occ1` present-hit and the
  near-1.0 ratio swings are timing/ordering effects, not semantics; no Sol
  escalation triggered. The present-hit micro claim (−15.6%) is not uniform over
  occupancy on this schedule; that is a measurement result, not a semantic error.

## 12 · Residual and next bounded task

- Residual (uncovered): the raw 2690-line pair tables are retained only at the
  `/tmp` log paths with their sha256 in §2; durable retention is not included
  here. The per-cell `strata_summary`/`acceptance_summary` lines are `--json`-only
  and were not captured in table mode. The `occ1` present-hit order-instability is
  characterized but not root-caused; it affects four cells and several near-1.0
  pairs. The present-hit regression at `occ2`/`occ4`/`occ1` (ratio 8.63 → 9.78,
  8.43 → 8.92, 8.76 → 10.04 vs `hashtrinity_table`) is measured but not
  attributed.
- Next bounded task: attribute the worst **stable** current-strata ratio — the
  spill **present-hit** path at `occ2-offset-beyond-direct-present-hit-*` vs
  `hashtrinity_table` (~9.78×, ~7.0 ns/op) and the low-occupancy `occ4`/`occ1`
  cells — to a named `src/table.rs`/`src/table/scan.rs`/`src/table/state.rs`
  region, with a focused before/after measurement on the same 4096-op schedule.
  Determine whether the power-of-two summary mask fast path is bypassed at low
  occupancy. Secondary: the `occ1` present-hit order-instability. Do not weaken
  the `<= 1.0` contract.

## Arc Block

```text
SCOPE:      docs/evidence/perf-remeasure-2026-09-19-post-mask.md (new receipt only);
            src/*, tests/*, proof/*, examples/* untouched
REGISTER:   measurement; no algebra/refinement/state obligation
CORPUS:     no canonical-corpus match (performance receipt); ApeironWord.lean digest recorded, freshness not re-run
REFERENCE:  docs/evidence/perf-remeasure-2026-09-19-post-scatter.md; examples/reg1/acceptance.rs, performance.rs, strata.rs
GATE:       cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
            cwd /home/socol/Workspace/apeiron; observed red exit 1, HEADLINE pass=561 fail=889
GRADE:      measured (three passes); digests and log hashes are identity receipts
```
