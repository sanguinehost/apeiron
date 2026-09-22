# Global baseline receipt (APEIRON-PERF-GLOBAL-W7-T1)

Binding: this receipt binds to the dirty-tree digests below, **not to HEAD**.
`apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`. The captured
`src/table.rs` and `src/table/scan.rs` are uncommitted; HEAD's copies differ. The
flat-resident rework is compared against the digests in §1, not against a commit.

## 1 · Source and harness snapshot

| artifact | sha256 | lines |
|---|---|---|
| `src/table.rs` | `e56014dade8f385a83f1a2c0721fb186a5d1bd580a64486c5ef36ad17d2d17ea` | 1330 |
| `src/table/scan.rs` | `698b1ecaa6c15cf1f5faa8c3286dc3f60eb9898b13c89215085a021e9f9a7578` | 210 |
| `tests/table.rs` | `12ad029f27c0e982c44fe4248797b3b0dc6bb2248c0bebd7d18527221844bab5` | 356 |
| `examples/reg1/main.rs` | `51e1c10c2fbff6af48cddf82bd63ae3f3f2df318bb20b6c189517a6ebd0ac46c` | 620 |
| `examples/reg1/matrix.rs` | `405351ee86cc9c4cfb003d11f62dfaf3cb518ce314a9df29a4aef205bf87cf90` | 1465 |
| `examples/reg1/performance.rs` | `ab92dfc2ec9d2ce4944acece2e4b96fb4c0d2dfe714d3e5f97f64e79db7c7bae` | 1193 |
| `examples/reg1/strata.rs` | `fdf56add117ccb1da0c413bc0be43063b9ed3f8975488de6f3b641f1993da90b` | 1498 |
| `examples/reg1/acceptance.rs` | `183c019c8c9447e1518b2c76ce577eb910b0e92bb24fa302a74576e34e88f3c0` | 1102 |
| `examples/reg1/history.rs` | `46353c85ad4352c59f7ec9c3e5f2a42df532adfd3a2fbafc0379025cc11af057` | 1498 |
| `examples/reg1/adapters/serial.rs` | `b7fa9a735ad673b46899e0eabb049db63b720cede09de2e8ea98b11d33c744b3` | 356 |
| `examples/reg1/adapters/spatial.rs` | `e22cc9a471e5075241cf2eb509cf298ba8e8c1f3dc13270bbd368c212bac3ba2` | 851 |
| `examples/reg1/matrix.json` | `068f717041e5acfbf5dc802a1636c8fafe5981bf3ece717c85fe6961733c8bac` | 1636 |
| `Cargo.toml` | `9d4452267341ef7ee72f23fbccee3a96e3585974cb7478b8b3f8e46325b8711f` | 151 |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` | 899 |

The run does not write any of these files; re-checked after the run, digests
unchanged.

## 2 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux (rolling), kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads |
| memory | 62 GiB |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target | `x86_64-linux`, release |
| serial harness threads | `1` (`single_thread=true`, from the run's own header) |
| hashtrinity rev | `4cc14471ae24f3f204fb0b0e91736ef75c909358` |
| apeiron head | `c450c222905c050f793a23332708c230c658f82c` (dirty) |

## 3 · Run and result

Command (cwd `/home/socol/Workspace/apeiron`):

```sh
cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

Exit code: **1** (strict acceptance failure; the run is the evidence, nothing was
fixed). A cargo warning `error: duplicate key` from the hashtrinity git checkout
(`opsfold/Cargo.toml:61`) is printed before `Finished`; it is non-fatal and does
not determine the exit status.

Failure class (stderr, last line):

```text
acceptance failed: applicable=650 failing=517 missing=0 unauthorized=0 full_acceptance=false
```

`acceptance_summary`:

```text
acceptance_summary registered=2592 load_bearing=1458 applicable=650 pass=133 fail=517 missing=0 unauthorized=0 not_applicable=808 informational=1134 worst_pair=spatial/n1000000/foldhash_par16 worst_ratio=147.2329 full_acceptance=false
```

`acceptance_first_failure`:

```text
acceptance_first_failure=(occ1-direct-index-resident-present-hit-replacement, hashtrinity_table) rmax=4.0406 > 1.00
```

`strata_summary` (the 160-cell serial fixed-shard axis):

```text
strata_summary cells=160 measured=160 pass=20 fail=140 not_applicable=0 missing=0 rejected=0 resource_cells=3 resource_not_applicable=1 resource_missing=2 campaigns=3 warmup=2 samples=15 ops=4096 ht_rev=4cc14471ae24f3f204fb0b0e91736ef75c909358 rustc="rustc 1.97.0-nightly (17584a181 2026-04-13)" cpu="Intel(R) Core(TM) i9-14900KF"
```

`report_complete=false`, `reg1_pass=false`.

## 4 · Per-incumbent ratios and pass/fail (all registered incumbents)

`pairs` is the count of measured pairs whose oracle digest agreed; `applicable` is
the tally's evaluated-pair count, with `pass`/`fail`/`not_applicable` summing to it.
Load-bearing rows carry the acceptance verdict.

Verbatim per-incumbent ratio lines:

```text
acceptance_ratio incumbent=hashtrinity_table class=load-bearing pairs=160 digest_fail=0 r_min=0.2884 r_max=9.2695
acceptance_ratio incumbent=hashtrinity_count_par class=load-bearing pairs=2 digest_fail=0 r_min=8.7423 r_max=78.0214
acceptance_ratio incumbent=hashtrinity_get_each class=load-bearing pairs=2 digest_fail=0 r_min=8.9823 r_max=81.6256
acceptance_ratio incumbent=hashbrown_foldhash class=load-bearing pairs=160 digest_fail=0 r_min=0.3779 r_max=16.1926
acceptance_ratio incumbent=hashbrown_fxhash class=load-bearing pairs=160 digest_fail=0 r_min=0.4578 r_max=29.5121
acceptance_ratio incumbent=hashbrown_rayon class=load-bearing pairs=2 digest_fail=0 r_min=7.2338 r_max=31.2551
acceptance_ratio incumbent=std_siphash class=load-bearing pairs=160 digest_fail=0 r_min=0.1600 r_max=3.8239
acceptance_ratio incumbent=foldhash_par16 class=load-bearing pairs=2 digest_fail=0 r_min=16.1193 r_max=147.2329
acceptance_ratio incumbent=fxhash_par16 class=load-bearing pairs=2 digest_fail=0 r_min=19.4473 r_max=133.1571
```

| incumbent | class | pairs | r_min | r_max | applicable | pass | fail | not_applicable |
|---|---|---|---|---|---|---|---|---|
| `hashtrinity_table` | load-bearing | 160 | 0.2884 | 9.2695 | 162 | 14 | 146 | 2 |
| `hashtrinity_count_par` | load-bearing | 2 | 8.7423 | 78.0214 | 162 | 0 | 2 | 160 |
| `hashtrinity_get_each` | load-bearing | 2 | 8.9823 | 81.6256 | 162 | 0 | 2 | 160 |
| `hashbrown_foldhash` | load-bearing | 160 | 0.3779 | 16.1926 | 162 | 11 | 149 | 2 |
| `hashbrown_fxhash` | load-bearing | 160 | 0.4578 | 29.5121 | 162 | 10 | 150 | 2 |
| `hashbrown_rayon` | load-bearing | 2 | 7.2338 | 31.2551 | 162 | 0 | 2 | 160 |
| `std_siphash` | load-bearing | 160 | 0.1600 | 3.8239 | 162 | 98 | 62 | 2 |
| `foldhash_par16` | load-bearing | 2 | 16.1193 | 147.2329 | 162 | 0 | 2 | 160 |
| `fxhash_par16` | load-bearing | 2 | 19.4473 | 133.1571 | 162 | 0 | 2 | 160 |
| `xxhash_xxh3` | informational | 160 | 0.0488 | 2.3875 | 0 | 0 | 0 | 0 (informational=162) |
| `xxhash_xxh64` | informational | 160 | 0.3339 | 9.5952 | 0 | 0 | 0 | 0 (informational=162) |
| `wyhash` | informational | 160 | 0.3895 | 14.6033 | 0 | 0 | 0 | 0 (informational=162) |
| `ahash` | informational | 160 | 0.3528 | 13.1617 | 0 | 0 | 0 | 0 (informational=162) |
| `fnv` | informational | 160 | 0.3232 | 8.1112 | 0 | 0 | 0 | 0 (informational=162) |
| `blake3` | informational | 160 | 0.0231 | 0.4897 | 0 | 0 | 0 | 0 (informational=162) |
| `sha256` | informational | 160 | 0.0286 | 0.7166 | 0 | 0 | 0 | 0 (informational=162) |

All `digest_fail=0`. The 160-pair incumbents are the serial fixed-shard strata
face; the 2-pair incumbents are the packed spatial `n100000`/`n1000000` face. The
`not_applicable` counts are the registered per-face exclusions (serial incumbents on
the spatial face and spatial incumbents on the strata face), not missing adapters
(`missing=0`).

## 5 · Spatial selected-vs-incumbent medians

`spatial_protocol` sizes `[100000, 1000000]`, campaigns `3`, warmup `2`, samples
`15`, metric `elapsed_ns_median`, oracle `std::HashMap` count folded in query order.

### `spatial/n100000` — selected `selected_median_ns=19707218`

| incumbent | selected ns | incumbent ns | ratio | ratio_interval |
|---|---|---|---|---|
| `hashtrinity_count_par` | 19707218 | 2210482 | 8.9153 | [8.7423, 9.0556] |
| `hashtrinity_get_each` | 19707218 | 2184165 | 9.0228 | [8.9823, 9.4479] |
| `hashbrown_rayon` | 19707218 | 2640717 | 7.4628 | [7.2338, 7.6639] |
| `foldhash_par16` | 19707218 | 1166252 | 16.8979 | [16.1193, 17.8666] |
| `fxhash_par16` | 19707218 | 980370 | 20.1018 | [19.4473, 21.1001] |

### `spatial/n1000000` — selected `selected_median_ns=1235053749`

| incumbent | selected ns | incumbent ns | ratio | ratio_interval |
|---|---|---|---|---|
| `hashtrinity_count_par` | 1235053749 | 15892921 | 77.7109 | [77.1799, 78.0214] |
| `hashtrinity_get_each` | 1235053749 | 15717175 | 78.5799 | [76.9795, 81.6256] |
| `hashbrown_rayon` | 1235053749 | 41241983 | 29.9465 | [28.4135, 31.2551] |
| `foldhash_par16` | 1235053749 | 8521887 | 144.9273 | [129.2260, 147.2329] |
| `fxhash_par16` | 1235053749 | 9367241 | 131.8482 | [130.3004, 133.1571] |

`selected_digest_ok=true` and `digest_ok=true` on every spatial cell.

## 6 · Baseline deficits (this is the "before")

Serial (160 strata cells, 3 campaigns): the measured serial worst against
`hashbrown_fxhash` is `r_max=29.5121`; `std_siphash` is the closest incumbent at
`r_max=3.8239`. Spatial at `n=1000000`: `hashtrinity_count_par` median ratio
`77.7109` (interval max `78.0214`) and `foldhash_par16` median ratio `144.9273`
(interval max `147.2329`).

The task handoff states the same two deficits with prior approximate figures:
serial direct-index-resident worst ~28x behind `hashbrown_fxhash`; spatial n=1M
79.9x behind `hashtrinity_count_par` and 138x behind `foldhash_par16`;
`std_siphash` closest at 3.77. Those figures are prior measurements from a
different run moment; the live figures in this section are authoritative for this
receipt's digests. Measured deltas: hashbrown_fxhash +1.5, count_par −2.2,
foldhash_par16 +6.9, std_siphash +0.05. No speed claim is made.

The per-incumbent ratio print does not name the serial cell carrying the
`hashbrown_fxhash` maximum; the earlier no-spatial run attributed the serial
`hashbrown_fxhash` worst to `occ16-clustered-skewed-absent-miss-replacement`
(ratio `28.2913`). This receipt records the aggregate `r_max=29.5121` only.

## 7 · Controls

Observed: **53 red control lines, 0 green** (22 distinct names). Distinct counts:
`retired-shardfull-digest` ×16, `retired-shardfull-receipt` ×16,
`fabricated-current-cell` ×2, `placement-only-address` ×2, and one each of
`wrong-workload-identity`, `invalid-samples`, `absent-samples`,
`cached-home-only-collision`, `wrong-oracle-digest`,
`deleted-required-current-cell`, `selected-implementation-mismatch`,
`strict-acceptance-single-failing-cell`, `spatial-wrong-packed-input`,
`spatial-wrong-bucket-routing`, `spatial-wrong-worker-width`,
`acceptance-failing-non-hashtrinity-pair`, `acceptance-omitted-incumbent-pair`,
`acceptance-unauthorized-not-applicable`, `acceptance-crossing-ratio-interval`,
`acceptance-wrong-oracle-digest`, `acceptance-serial-adapter-wrong-result`. Every
acceptance/performance control child exits 2.

## 8 · Raw run logs

| artifact | sha256 | lines |
|---|---|---|
| `docs/evidence/perf-global-baseline.log` (stdout) | `92badfedd7f756d903119bce389a926fab1fa7fea27f4af4f843705c2277d62d` | 492 |
| `docs/evidence/perf-global-baseline.stderr.log` (stderr) | `9f7a6ec3d036a2947145c25408b67ab133740e4b9c741bc17102bd6b8e91e43d` | 8 |

## 9 · Scope and exclusions

- In scope: identity of the pre-rework `src/table.rs`/`src/table/scan.rs` and
  harness, the strict per-pair acceptance result, and the serial + spatial ratios
  on this machine/toolchain. This is the "before" for the flat-resident rework.
- Out of scope (obligations not evidenced here): `Sigma` in every executing
  signature, `remove`, SIMD, allocator behaviour, threading memory semantics,
  std internals, rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V. The spatial
  `std::thread::scope` orchestration is an unextracted boundary.
- No history, Kani, `proof/regen.sh`, or `lake build` was run for this receipt.
- Grade: snapshot, harness and log identity are digest receipts (`tested`
  identity); the run is a `measured`/gate result on the recorded machine and tool
  identity.

## 10 · Open items

- **REG-1 OPEN**: the strict performance acceptance exits 1 with
  `full_acceptance=false`, 517 of 650 applicable pairs failing.
- **G1 OPEN**: the performance objective is not met by the current representation;
  the serial and spatial deficits in §6 are the baseline to be reduced.
