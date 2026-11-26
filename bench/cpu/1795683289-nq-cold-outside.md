# nq: cold outside-pass split (`207.4 ms` alloc, `210.3 ms` insert) and parallel `insert` (`4.675 → 4.498 s`)

Criterion: **T2** (turn latency), partial: the cold 1,827-position CPU prefill of the 0.6B pack at
block 1024 (`2 × 913 + 1`), width 8, on the OVH box (`1795674629-nq-prefill-serial.md`'s workload).
It is not a 7B result, not a 200k context turn, not a decode result, and not measured on the laptop.

**Measured split and single-term fix.** On the instrumentation-only engine (`prefix_solve.before`),
three `--phases` runs split the `418.486–431.299 ms` (`model`: `cold_ms − pass_ms`) outside-pass
interval into two nearly equal halves: `outside.alloc` (`decode_state`) median **207.366 ms**
(range 203.849–209.785 ms, spread 2.91%) and `outside.insert` (`PrefixCache::insert`) median
**210.259 ms** (range 200.168–211.270 ms, spread 5.55%), plus the traced greedy head **5.489 ms**
and a **5.682 ms** (`model`) unsplit remainder (probes, dense `lm_head_fixed_sched`,
`argmax_fixed`, checksum, drops). `outside.insert` is the higher median (`210.259` vs
`207.366 ms`) and higher mean (`207.232` vs `207.000 ms`), though the two medians differ by
**2.893 ms (1.40%, `model`)**, inside each term's 3-run spread. Only `insert` was changed
(`KvState::clone_planes_parallel` over `std::thread::scope`, up to 8 threads); `decode_state`
(`203.849–209.785 ms` in Step 2, `195.886–208.992 ms` in traced `before`, `206.043–211.535 ms` in
traced `after`) was left untouched per the single-term rule.

In the quiet-window interleaved paired benchmark (3 rounds rotating `llama-bench`, `before`,
`after`), cold `prefix_solve` drops from median **4.675 s** (`4.639–4.726 s`, spread 1.88%) to
median **4.498 s** (`4.497–4.513 s`, spread 0.36%): **−177 ms (−3.79%, `model`: `after_med − before_med`)**,
med/med ratio **0.9621×** (**1.0394×** speedup, `model`), with zero overlap between the two
distributions (`126 ms` separation between `after` max `4.513 s` and `before` min `4.639 s`,
`126–228 ms` per-round deltas `before − after`, and `229 ms` max cross-arm span `4.726 − 4.497 s`,
`model`). In the interleaved `--phases` pair, `outside.insert` drops from median **206.795 ms**
(`200.889–210.339 ms`) to **42.041 ms** (`35.790–42.571 ms`; **−164.754 ms**, **4.92×** faster,
`model`: `206.795 / 42.041`). `outside.alloc` remains **210.744 ms** (traced `after` median,
realized in `after-p3`, `4.6%` of that turn's `4,565 ms` cold wall, `1/(1 − 210.744/4565) = 1.048×`
ceiling, `model`; or `4.685%` / `1.049×` against the untraced `after` median `4.498 s`, `model`)
and is the remaining outside-pass term.

Grades: `measured` for every wall, counter, and phase duration read directly from tool output;
`tested` for `the_parallel_clone_equals_the_serial_clone`, the workspace test suite, and the clippy
and doc checks; `model` for every derived wall (`llama` seconds `1827 / tps`, `cold_ms − pass_ms`,
unsplit remainder `outside − head − alloc − insert`, gaps, ratios, percentages, and `1/(1 − s)`
ceilings), each named with its formula. No `proved` claim: Aeneas and Charon are not installed on
either host (Lean `v4.31.0` exists locally in `~/.elan/toolchains`, without `target/aeneas` or
built `proof/corpus`). The raw lines are in `1795683289-nq-cold-outside.log`.

Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC Milan, 8 sockets × 1 core × 1 thread,
31.3 GiB, 256 MiB L3, no AVX-512, no VNNI, no GPU). The laptop — the only device a timed claim may
name — is **not measured**. Times UTC; every box job under `~/bin/adg run --` (16 GiB cap, 8 jobs,
swap zero).

**Base and provenance.** Worktree branch `area1-cold-outside`, base HEAD
`1a4df32f6cd6b9bb0be7487dbac143f23b5f5e29`. Modified source files (sha256 identical locally and on
the box): `examples/prefix_solve.rs` (`bff0e46d…`), `src/engine/trace.rs` (`3ee5c83f…`),
`src/engine/prefix/mod.rs` (`873799be…`), `src/engine/attn.rs` (`f5b9cc62…`),
`src/engine/attn/cert.rs` (`0dba341d…`). Binaries: `prefix_solve.before` (`a3039f27…`, fresh
`target/` build of base HEAD + instrumentation only), `prefix_solve.after` (`c962b582…`, rebuilt on
changed sources with parallel plane clone), `llama-bench` `3305fbed…` (`049326a00` (9949)).
Fixtures: `skeleton-q8head.gguf` `107c8edd…`, `nq06-v3.nqp` `2e4670a4…`, `tokenizer.json`
`c0382117…`, `Qwen3-0.6B-Base.f16.gguf` `a33dce69…`.

## 1. The split of the outside-pass interval

Instrumentation was added at the non-extracted call site (`src/engine/prefix/mod.rs`) and
`src/engine/trace.rs` / `examples/prefix_solve.rs`: `record_alloc(ns)`, `record_insert(ns)`,
`outside_split()` gated by `trace::enabled()`, printing `outside.alloc` and `outside.insert` under
`--phases`.

Three runs of `prefix_solve.before --phases` at `08:38:02 UTC` (start load `0.63`, `measured`):

| term | run 1 (ms) | run 2 (ms) | run 3 (ms) | median (ms) | mean (ms) | spread (`model`) | grade |
|---|---:|---:|---:|---:|---:|---:|---|
| `cold` turn (`cold_s × 1000`) | 4,646 | 4,712 | 4,812 | **4,712** | 4,723.3 | 3.57% | `measured` |
| `pass` | 4,218.888 | 4,293.514 | 4,380.701 | **4,293.514** | 4,297.701 | 3.84% | `measured` |
| `outside pass` (`cold − pass`) | 427.112 | 418.486 | 431.299 | **427.112** | 425.632 | 3.06% | `model` |
| `outside.alloc` (`decode_state`) | 203.849 | 207.366 | 209.785 | **207.366** | 207.000 | 2.91% | `measured` |
| `outside.insert` (`cache.insert`) | 211.270 | 200.168 | 210.259 | **210.259** | 207.232 | 5.55% | `measured` |
| `head` (`head_greedy_state`) | 5.489 | 5.445 | 5.573 | **5.489** | 5.502 | 2.35% | `measured` |
| remaining outside (`outside − head − alloc − insert`) | 6.504 | 5.507 | 5.682 | **5.682** | 5.898 | 18.10% | `model` |
| `pool.wait` (overlay, 1,295 calls) | 276.813 | 288.671 | 288.959 | **288.671** | 284.814 | 4.39% | `measured` |

`outside.alloc` and `outside.insert` together account for `417.625 ms` (97.8% of the `427.112 ms`
outside-pass median, `model`); the unsplit non-KV remainder (probes, `lm_head_fixed_sched`,
`argmax_fixed`, checksum, drops) is **5.682 ms** (`model`), so the `427.112 ms` outside-pass median
(`418.486–431.299 ms`, `model`) is split ~50/50 between `decode_state` and `cache.insert`. Under the
Area 1 outside-pass scope, `outside.insert` is the higher median (`210.259` vs `207.366 ms`) and
higher mean (`207.232` vs `207.000 ms`), with a single-term ceiling of
`1/(1 − 210.259/4712) = 1.047×` (`model`), though `210.259` vs `207.366 ms` sits within both
3-run spreads (see `## Review` for the whole-turn and co-dominance critique).

## 2. What changed (`insert` only)

- `src/engine/attn.rs`: added `KvState::clone_planes_parallel(&self) -> Self` and
  `KvState::clone_for_snapshot(&self) -> Self` (safe Rust, zero `unsafe`). Owned `(k, v)` plane
  pairs (`ks, vs` too for `I8`) are cloned across `std::thread::scope` over
  `min(available_parallelism, 8)` chunks into pre-sized destination planes (chunk 0 on the caller
  thread, remaining chunks on scoped worker threads). Mapped states (`map.is_some()`), states with
  `< 8` KV planes, and `workers <= 1` fall back to serial `Clone`. `clone_for_snapshot()` also falls
  back to serial `Clone` under `cfg!(test)` (unit tests), while
  `the_parallel_clone_equals_the_serial_clone` calls `clone_planes_parallel()` directly.
- `src/engine/prefix/mod.rs`: `PrefixCache::insert` calls `kv.clone_for_snapshot()` instead of
  `kv.clone()`.
- `src/engine/attn.rs` and `src/engine/attn/cert.rs`: added
  `the_parallel_clone_equals_the_serial_clone` covering `Q32`, `F32`, and `I8` at `n_layer = 2`
  (fallback) and `n_layer = 5` (parallel), non-zero per-layer `occupied`, `KvBlockIndex` with
  non-zero `CertTotals`, and Linux file-backed mapped fallback, asserting bitwise equality of all
  plane words, `occupied`, `carrier`, geometry, and `blocks` against serial `Clone` (`tested`).

## 3. Paired before/after

### Untraced interleaved pair (quiet window, 08:43:32 UTC, load 0.57)

Three rounds rotating order: `(llama, before, after)`, `(before, after, llama)`, `(after, llama, before)`.
`spread (model)` is `(max − min) / min`.

| arm | r1 | r2 | r3 | mean | median | spread (`model`) | grade |
|---|---:|---:|---:|---:|---:|---:|---|
| `llama pp1827`, t/s | 469.62 | 473.29 | 470.54 | 471.15 | **470.54** | 0.78% | `measured` |
| `llama`, s (`1827 / tps`) | 3.8904 | 3.8602 | 3.8828 | 3.8778 | **3.8828** | 0.78% | `model` |
| `before` cold, s | 4.639 | 4.675 | 4.726 | 4.680 | **4.675** | 1.88% | `measured` |
| `after` cold, s | 4.513 | 4.497 | 4.498 | 4.503 | **4.498** | 0.36% | `measured` |
| `before − after`, ms | 126 | 178 | 228 | 177.3 | **178** (`177` med diff) | — | `model` |
| `after / before` ratio | 0.9728 | 0.9619 | 0.9518 | 0.9622 | **0.9619** (`0.9621` med/med) | 2.22% | `model` |
| `before − llama`, s | 0.7486 | 0.8148 | 0.8432 | 0.8022 | **0.8148** (`0.7922` med/med) | 12.64% | `model` |
| `after − llama`, s | 0.6226 | 0.6368 | 0.6152 | 0.6249 | **0.6226** (`0.6152` med/med) | 3.50% | `model` |
| `before / llama` | 1.1924 | 1.2111 | 1.2172 | 1.2069 | **1.2111** (`1.2040` med/med) | 2.07% | `model` |
| `after / llama` | 1.1600 | 1.1650 | 1.1585 | 1.1612 | **1.1600** (`1.1585` med/med) | 0.56% | `model` |

Every `after` run (`4.497–4.513 s`) is faster than every `before` run (`4.639–4.726 s`), with zero
overlap: **126 ms** (`4.639 − 4.513`, `model`) between `after` max and `before` min, **126–228 ms**
per-round deltas (`model`), and **229 ms** (`4.726 − 4.497`, `model`) widest cross-arm span.

### Traced interleaved pair (`--phases`, 08:45:34 UTC, load 1.42)

Three rounds rotating `(before, after)`, `(after, before)`, `(before, after)`:

| metric (cold turn) | `before` (p1 / p2 / p3) | `before` med (spread, `model`) | `after` (p1 / p2 / p3) | `after` med (spread, `model`) | `Δ med` (`after − before`) | grade |
|---|---:|---:|---:|---:|---:|---|
| `cold`, ms | 4,649 / 4,772 / 4,774 | **4,772** (2.69%) | 4,650 / 4,675 / 4,565 | **4,650** (2.41%) | −122 ms | `measured` (`Δ`: `model`) |
| `pass`, ms | 4,234.596 / 4,350.551 / 4,345.609 | **4,345.609** (2.74%) | 4,384.063 / 4,414.657 / 4,306.665 | **4,384.063** (2.51%) | +38.454 ms | `measured` (`Δ`: `model`) |
| `pool.wait`, ms | 279.011 / 275.467 / 256.210 | **275.467** (8.90%) | 298.096 / 300.800 / 299.256 | **299.256** (0.91%) | +23.789 ms | `measured` (`Δ`: `model`) |
| `outside.alloc`, ms | 195.886 / 208.992 / 206.446 | **206.446** (6.69%) | 211.535 / 206.043 / 210.744 | **210.744** (2.67%) | +4.298 ms | `measured` (`Δ`: `model`) |
| `outside.insert`, ms | 206.795 / 200.889 / 210.339 | **206.795** (4.70%) | 42.571 / 42.041 / 35.790 | **42.041** (18.95%) | **−164.754 ms** (**4.92×**, `model`) | `measured` (`Δ`/ratio: `model`) |
| `head`, ms | 5.355 / 5.618 / 5.707 | **5.618** (6.57%) | 5.839 / 5.536 / 5.798 | **5.798** (5.47%) | +0.180 ms | `measured` (`Δ`: `model`) |
| `outside pass` (`cold − pass`) | 414.404 / 421.449 / 428.391 | **421.449** (3.38%) | 265.937 / 260.343 / 258.335 | **260.343** (2.94%) | **−161.106 ms** (−38.2%) | `model` |
| remaining (`outside − head − alloc − insert`) | 6.368 / 5.950 / 5.899 | **5.950** (7.95%) | 5.992 / 6.723 / 6.003 | **6.003** (12.20%) | +0.053 ms | `model` |

On the subsequent `warm` and `shared` turns (`--phases`), `outside.insert` also drops from medians
**26.718 ms** (`warm`, range `26.246–27.269 ms`, `measured`) and **26.212 ms** (`shared`, range
`26.040–27.493 ms`, `measured`) to **9.397 ms** (`8.836–9.666 ms`, `measured`; **2.84×**, `model`:
`26.718 / 9.397`) and **8.503 ms** (`8.424–9.858 ms`, `measured`; **3.08×**, `model`:
`26.212 / 8.503`).

## 4. Gates

Run on the OVH box under `~/bin/adg run --`, exact exit codes and outcomes for the contributor
gate (`test` without `--all-features`, which needs the `glslc` pin and a device):

| command | exit | result | grade |
|---|---:|---|---|
| `cargo build --release --locked --example prefix_solve` | 0 | reproduces `prefix_solve.after` sha256 `c962b582…` | `tested` |
| `cargo fmt --all -- --check` | 0 | pass | `tested` |
| `cargo clippy --workspace --all-features --all-targets --locked -j 1 -- -D warnings` | 0 | pass, no warnings | `tested` |
| `cargo test --workspace --locked --no-fail-fast` | 101 | lib 692 passed, 1 failed (the `coopmat_spirv_is_the_glsl` `glslc` pin), 55 ignored; `spirv_storage_race` 2 passed, 2 failed (`attn_coopmat_spirv_is_the_glsl` SIGSEGV, `splitk_reduce_spirv_is_the_glsl` stale SPV); every other suite passed | `tested` |
| `cargo test --lib --locked the_parallel_clone_equals_the_serial_clone` | 0 | 1 passed; 0 failed | `tested` |
| `python3 tools/check-claims.py` | 0 | 45 graded claims OK | `tested` |
| `python3 tools/book-check.py` | 0 | 20 glossary entries, 0 findings | `tested` |
| `mdbook build book` | 0 | pass | `tested` |
| `mdbook test book -L $HOME/apeiron-mdbook1/debug/deps` | 0 | every chapter | `tested` |

An earlier `--all-targets` run without `--all-features` reported 8 pre-existing lib warnings in
unmodified files (`pool.rs`, `place.rs`, `gpu/nn/fwd/{lin,mod}.rs`, `learn/nanoquant/step.rs`); the
exact contributor command with `--all-features` passes.

## 5. Commands

```sh
# build on box (~/Workspace/apeiron-nq-area1)
~/bin/adg run -- cargo build --release --locked --example prefix_solve
# turn command (untraced pair; --phases added for split and traced pair)
~/bin/adg run -- /tmp/area1/prefix_solve.<arm> \
  --model "$GGUF" --pack "$NQP" --tokenizer "$TOK" \
  --len 1827 --differ 0 --decode 0 --mb 440 [--phases]
# paired quiet-window script (untraced 3-round rotated + traced 3-round rotated)
~/bin/adg run -- bash /tmp/area1/pair.sh   # full script body in .log
# llama-bench baseline
~/llama.cpp/build/bin/llama-bench -m "$LLAMA" -p 1827 -n 0 -t 8 -r 1
# equality artifact and box gates
~/bin/adg run -- cargo fmt --all -- --check
~/bin/adg run -- cargo test --lib --locked the_parallel_clone_equals_the_serial_clone
~/bin/adg run -- cargo test --workspace --locked [--no-fail-fast]
~/bin/adg run -- cargo clippy --workspace --all-targets --locked -j 1 [-- -D warnings]
# doc checks (the final run below is on the box, under ~/bin/adg run --)
python3 tools/check-claims.py
python3 tools/book-check.py
```

## 6. Review

`gemini-4-hydrogen-xhigh` ran (`agy --dangerously-skip-permissions --model gemini-4-hydrogen-xhigh --output-format stream-json --print-timeout 0`). Pass 1 returned **FAIL — 5 refusals** (Check 4 passed):

1. **Arithmetic and cross-run figures (fixed in Pass 2):** `126–229 ms` min/max wording (`126 ms` min-to-max separation vs `229 ms` max cross-arm span); unresolved `207–211 ms` `decode_state` range (replaced with per-stage ranges); `~421 ms` in §1 (replaced with `427.112 ms` median); `+0.615 s` / `1.158×` called paired median instead of `med/med` (paired medians are `+0.6226 s` / `1.1600×`, `med/med` `+0.6152 s` / `1.1585×`); `210.744 ms` share quoted as `4.5%` instead of `after-p3`'s `4.6%` (`210.744 / 4565 = 4.617%`, `1.048×` ceiling).
2. **Missing commands and summarized gate outputs in `.log` and §5 (fixed in Pass 2):** added the instrumentation-only `git diff`, verbatim `before` and `after` build outputs, `pair.sh` body, `sha256sum` commands, verbatim `cargo test`/`clippy` runner/failure/warning outputs, and separated local vs box checks. (`before` built from a clean `target/`, `after` recompiled `apeiron` on changed sources; no manual `touch src/lib.rs` ran between them.)
3. **Derived numbers not graded `model` (fixed in Pass 2):** marked `Δ med`, `4.92×`, `2.84×`, `3.08×`, `−177 ms`, `0.9621×`, `1.0394×`, and `med/med` figures as `model` with their formulas.

Pass 2 (`--resume 2c450edc-27e7-4859-a82d-53c52f73f772`) confirmed Pass 1 Refusals 1–3 resolved and raised **3 minor factual/log items (fixed after Pass 2)**: `tests/gpu_*.rs` count `19` → `25` (`26` total `gpu_*.rs` minus `gpu_spv_check.rs`); `.log` clarification that the paired output lines come from `/tmp/area1/pair/*.wall` plus the `build 1` `cp` to `prefix_solve.before`; `4.616%` → `4.617%` and `"after-p3 median"` → `"traced after median, realized in after-p3"` (plus `spread` column `(model)` tags).

Two structural refusals from Pass 1 (re-confirmed in Pass 2) remain open:

4. **Maintainer gate on extraction-included files and shifted `trace.rs` spans (remains open, documented in §7):** `src/engine/attn.rs` (including `src/engine/attn/cert.rs` via `#[path = "attn/cert.rs"]`) and `src/engine/trace.rs` are listed in `proof/regen.sh` `SOURCE_FILES`. Adding fields/functions above `Span`/`span`/`q8_bytes`/`macs` in `src/engine/trace.rs` shifts the four `trace.*` `Source:` spans in `proof/ApeironWord.lean` (`139..187` → `177..225`), so `proof/regen.sh --check` will detect drift until regenerated; Aeneas/Charon is absent on both hosts.
5. **Dominant-term / single-term scope (remains open):** on the whole cold turn, `outside.insert` (`4.46%`, `1.047×` ceiling, `model`) is smaller than `ffn` (`33.7%`), `attn.core` (`31.8%`), `attn.proj_in` (`13.0%`), and `attn.proj_out` (`5.3%`), so targeting it rests on the Area 1 brief's outside-pass scope rather than whole-turn dominance (no separate owner waiver in-tree). Within the outside-pass interval, `outside.insert` (`210.259 ms`) and `outside.alloc` (`207.366 ms`) are co-dominant within spread (`2.893 ms` / `1.40%`, `model`), so choosing `insert` follows the higher median/mean and the Area 1 brief's single-term rule while leaving `outside.alloc` untouched.

## 7. Omissions and open items

- **T2 (open)**: `GOAL.md` T2 requires a 1,000-token incremental turn at 200k context in `≤ 60 s`
  plus a 200k full prefill on a 7B-class model; this receipt measures only a 0.6B cold 1,827-token
  CPU prefill on the OVH box (paired median gap `+0.6226 s`, paired median ratio `1.1600×`;
  `med/med` gap `+0.6152 s`, `med/med` ratio `1.1585×` vs llama `pp1827`, `model`).
- **Target device**: the laptop (`i5-10210U`) was not measured.
- **Untouched `outside.alloc`**: `decode_state` (`outside.alloc` median `210.744 ms` in traced
  `after`, range `206.043–211.535 ms`, `4.6%` of `after-p3`'s `4,565 ms` cold turn, `1.048×`
  ceiling, `model`; median `206.446 ms` in traced `before`, `195.886–208.992 ms`) remains
  untouched per the single-term rule and is now the dominant outside-pass term.
- **Maintainer gate (open)**: Aeneas and Charon are not installed on either host (Lean `v4.31.0`
  exists locally in `~/.elan/toolchains`, without `target/aeneas` or built `proof/corpus`), so
  `proof/regen.sh --check` and `cd proof/corpus && lake build` did not run. `src/engine/attn.rs`
  (with `#[path = "attn/cert.rs"]` for `src/engine/attn/cert.rs`) and `src/engine/trace.rs` are
  listed in `proof/regen.sh` `SOURCE_FILES`. `crate::trace` is `--opaque` (`proof/regen.sh:299`),
  though inserting `Counters` fields and `record_alloc`/`record_insert`/`outside_split` above
  `Span`/`span`/`q8_bytes`/`macs` shifts the four `trace.*` `Source:` line spans in
  `proof/ApeironWord.lean` (`139..187` → `177..225`), so `proof/regen.sh --check` will fail on
  span drift until regenerated; new `KvState` methods sit after extracted `attn.rs` methods and
  are called only from unextracted `src/engine/prefix/mod.rs`, and `cert.rs` edits are test-only.
- **GPU / `--all-features` / book build**: `--all-features` and GPU device tests were not run (no
  GPU and mismatched `glslc` 2023.8 on the box); the final gate ran `mdbook build`/`test` on the
  unmodified book (exit 0, §4).

## 7. Merged with area 2

The two lanes are separate binaries, and every number above is from `prefix_solve.after`
(`c962b582…`) on the area-1 tree. On the merged tree — this commit `678b9e97` on top of area-2
`0b61fcbb` (the 2,048 default) — the cold 1,827-token turn at width 8 is mean **4.078 s**
(4.158 / 3.990 / 4.085) against llama's interleaved **3.958 s** (459.53 / 460.87 / 464.52 t/s),
**1.030×** (paired 1.046 / 1.007 / 1.039; median 1.039), 3.0% slower, closing 85.1% of the
1,024-arm gap (`model` on the means). The merged binary is `d1b02e32…`; peak RSS median is
1,532,916 KiB = 1.462 GiB, +101.1 MiB over the area-2 2,048 row, a delta whose cause is not
isolated. The untouched `decode_state` allocation (`outside.alloc`, 207 ms) is 5.1% of the merged
turn; its single-term ceiling is `1/(1 − 207/4078) = 1.054×` (`model`). Table and raw lines:
`bench/cpu/1795683267-nq-prefill-one-pass.md`, `## The merged tree`. The merged change is not
re-measured on the laptop.
