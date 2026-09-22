# Micro spatial phase split, post stable scatter (APEIRON-PERF-SCATTER-W3-T1)

Focused release micro-benchmark of the packed spatial build+probe face on the current
source, after the stable chunk scatter landed in `src/table/partition.rs` and the
reserve-all integration in `src/table/parallel.rs`. No source, test, proof or example
file was edited; this run measures the committed harness against the working source.

## 1 · Binding

Dirty tree, not HEAD. `apeiron_head=c450c222905c050f793a23332708c230c658f82c dirty=yes`.
The run writes none of these files.

| artifact | sha256 |
|---|---|
| `src/table.rs` | `5f2608010137b16b195baad62cb5214bcadeeea13a9df4da39e19f4bf543a61e` |
| `src/table/partition.rs` | `7cf4724447b5ef8de8c4d4c2a175432d2284e3c1152b1f41c7f374cd13688c64` |
| `src/table/parallel.rs` | `b22650c1c6db76a7865e782fc0fc56174d3b8b6ac2c7eaccc0af087815df193f` |
| `src/table/scan.rs` | `698b1ecaa6c15cf1f5faa8c3286dc3f60eb9898b13c89215085a021e9f9a7578` |
| `examples/micro_spatial.rs` | `40b374540eb1c6ca1a6d13396eb916d2873a13bef9985baa38676b973b18d2df` |
| `Cargo.toml` | `9d4452267341ef7ee72f23fbccee3a96e3585974cb7478b8b3f8e46325b8711f` |
| `Cargo.lock` | `043b2a05b7c9b2b25229b96d2a39e50dc3403a840f66c9cf29e855dab4b16c1d` |

Changed since [`perf-remeasure-2026-09-19-post-spill.md`](perf-remeasure-2026-09-19-post-spill.md):
`src/table.rs` (`79621926…` → `5f260801…`), `src/table/partition.rs`
(`2e9a2f56…` → `7cf47244…`), `src/table/parallel.rs` (new to the binding table).

## 2 · Machine and tool identity

| item | value |
|---|---|
| host | `leviathan` |
| OS | CachyOS Linux, kernel `7.2.3-1-cachyos` |
| CPU | Intel(R) Core(TM) i9-14900KF, 32 threads, ~4.0 GHz nominal |
| governor | `powersave` |
| memory | 62 GiB total, 31 GiB available at run |
| load average | 0.99 / 1.01 / 0.90 at run start; no Lean, Kani or other build concurrent |
| rustc | `rustc 1.97.0-nightly (17584a181 2026-04-13)` |
| cargo | `cargo 1.97.0-nightly (eb94155a9 2026-04-09)` |
| target / profile | `x86_64-linux`, release |
| run log (stdout) | `/tmp/opencode/micro-spatial-20260919-204725.log`, sha256 `0c51eecfef1620cad19535a3c15c2b8a34f4afb685c4308501c8cb874c488f4f`, 7 lines |
| run log (stderr) | `/tmp/opencode/micro-spatial-20260919-204725.log.stderr`, sha256 `5450e94afb4cdc734698eb64504c8a512eec276e0eba039d3bb416cc466a2d4d` |

Non-fatal cargo warning `error: duplicate key` from the hashtrinity git checkout
`opsfold/Cargo.toml:61` precedes `Finished` and does not set the exit status.

## 3 · Command, exit, elapsed

```sh
# cwd /home/socol/Workspace/apeiron
/usr/bin/time -v nice -n 10 timeout 1800 \
  cargo run --release --locked --example micro_spatial -- --controls
# exit 0; wall 0:01.34; user 6.08 s; sys 0.48 s; max RSS 152716 KiB; timeout 1800 NOT reached
```

The harness protocol: `71942db scene(n)` packed-u64 build+probe, `workers=16`,
`warmup=2`, `samples=15`, `metric=elapsed_ns_median`, oracle `std::HashMap` count
folded in query order. `digest_ok=true` for both sizes: the FNV digest of the probe
readback equals the independent oracle.

The `--controls` argument is inert in this harness: `examples/micro_spatial.rs` reads
no `std::env::args`, so the flag changes nothing. The only in-harness check is the
printed `digest_ok`.

## 4 · Phase table (median ns, workers=16)

Each phase value is the median of the 15 timed samples after 2 warmups. Percentages
are the harness's own display: each field over the six-field subtotal it tracks. The
`reserve` phase (`PhaseNanos::reserve`, added with reserve-all) is not read by this
harness, so it is absent from both the printed total and the percentages.

### Build

| cell | qualify | partition | validate | schedule | kernel | assemble | six-field total | build median |
|---|---|---|---|---|---|---|---|---|
| `spatial/n100000` | 95596 (6.5%) | 470045 (32.1%) | 316854 (21.7%) | 62135 (4.2%) | 511732 (35.0%) | 6166 (0.4%) | 1462528 | 1675838 |
| `spatial/n1000000` | 1481721 (6.1%) | 3959142 (16.2%) | 6548069 (26.8%) | 63224 (0.3%) | 12352666 (50.6%) | 10294 (0.0%) | 24415116 | 27671723 |

### Probe

| cell | qualify | partition | validate | schedule | kernel | assemble | six-field total | probe median |
|---|---|---|---|---|---|---|---|---|
| `spatial/n100000` | 118028 (7.4%) | 317425 (19.8%) | 151052 (9.4%) | 60780 (3.8%) | 460051 (28.7%) | 495233 (30.9%) | 1602569 | 1777546 |
| `spatial/n1000000` | 2344010 (9.4%) | 2411899 (9.7%) | 4478654 (18.0%) | 64292 (0.3%) | 11164580 (44.9%) | 4388490 (17.7%) | 24851925 | 24654940 |

### End-to-end

| cell | population | ops | build median ns | probe median ns | end-to-end median ns | ns/op |
|---|---|---|---|---|---|---|
| `spatial/n100000` | 100000 | 99943 | 1675838 | 1777546 | 3478495 | 34.7850 |
| `spatial/n1000000` | 1000000 | 993259 | 27671723 | 24654940 | 52820140 | 52.8201 |

Unprinted `reserve` + accounting gap, from `build/probe median − six-field total`:
n100k build 213310 ns (~12.7%), n1M build 3256607 ns (~11.8%), n100k probe 174977 ns
(~9.8%), n1M probe −196985 ns (median non-additivity, within sample noise). The
`reserve` phase is material at ~12% of the build and is not in the harness output.

## 5 · Partition/validate before and after

The "before" row is the historical anchor stated in
[`docs/intent/parallel-stable-scatter.md`](../intent/parallel-stable-scatter.md)
(Problem): partition plus `Parts::is_intact` about 28% of the spatial build, per-shard
kernel 46–55%, assemble about 0% build / 16% probe. That anchor carries no absolute ns
and is **not** a fresh W0 measurement; no W0 receipt exists in `docs/evidence/`. The
"after" row is §4. Post-scatter `partition` now includes the checked chunk plan,
scatter-scratch reservation, stable scatter and stable concatenation, so part of the
kernel→partition shift is re-attribution, not pure added cost.

| run | kernel before | kernel after | partition+validate before | partition+validate after | assemble before | assemble after |
|---|---|---|---|---|---|---|
| build n100k | 46–55% | 35.0% | ~28% | 53.8% | ~0% | 0.4% |
| build n1M | 46–55% | 50.6% | ~28% | 43.0% | ~0% | 0.0% |
| probe n100k | 46–55% | 28.7% | ~28% | 29.2% | ~16% | 30.9% |
| probe n1M | 46–55% | 44.9% | ~28% | 27.7% | ~16% | 17.7% |

Direction: build `partition+validate` rose (28% → 43–54%) and build `kernel` fell
(46–55% → 35–51%) as the scatter cost moved into `partition`. Probe
`partition+validate` is roughly unchanged (27.7–29.2%); probe `assemble` rose to
17.7–30.9%. No speedup claim is made; the strict pair gate is the acceptance path.

## 6 · Measured vs not

- **Measured**: the phase medians, six-field totals, end-to-end build+probe medians and
  ns/op in §4, on this machine, toolchain, dirty-source identity and single process
  invocation, with the internal oracle digest matching (`digest_ok=true`). Grade:
  `measured`.
- **Interval**: none. The harness prints one median per phase from 15 samples; it does
  not print min/max, and this receipt is one invocation, not repeated campaigns. The
  numbers are single-run medians.
- **Not measured / out of scope**: strict REG-1 (`examples/reg1`, `--check performance
  --profile full --accept --controls`), `cargo test`, `cargo clippy`, `cargo fmt`,
  `cargo kani`, `proof/regen.sh`, Lean builds, the `reserve` phase (not read by the
  harness), the W3 digest-corruption rejecting control, cross-run intervals, and the
  `std::thread::scope` executor boundary (spawn/join/unwind/liveness remain
  unextracted). This diagnostic cannot pass G1.
- **Grade**: `measured` for §4; `anchor` for the §5 before column; the digests and log
  hashes are identity receipts.

## 7 · Residual and next bounded task

- Residual: the strict spatial pair gate
  (`cargo run --release --locked --example reg1 -- --check performance --profile full
  --accept --controls`) is pending; every spatial pair must be recorded and no interval
  maximum above `1.0` closes G1. Also pending: a fresh pre-scatter baseline (the W0
  receipt is absent, so §5's before column is the historical anchor), and the W3
  digest-corruption rejecting control, which was not run.
- Next bounded task: APEIRON-PERF-SCATTER-W4-T1, run the strict full-profile registry
  and record every spatial pair (including concurrency/address-face outcomes and every
  named incumbent) with its intervals and the raw-log hash. Do not weaken the `<= 1.0`
  contract.
