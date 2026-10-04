# M-M3a + B-VECTOR — integration receipt

Rows: `APEIRON-M-M3a` (`apeiron-m-m3a-land-the-m-Clps3Wt72UJCDxUGLcOlOe`), `APEIRON-B-VECTOR`
(`apeiron-b-vector-restore-HC9kdgd1vcpPWFnHT0qdDH`), integration `APEIRON-INSTALL`
(`apeiron-install-install-Hi069dVc2zNzjVUIXIzHtV`).

**`src/lib.rs`** — restore `#[cfg(feature = "gpu")] pub mod gpu;` (dropped by merge `920f87c`).
Gate: `cargo build --locked --features gpu` exit 0, `Finished dev profile … in 6.43s`.

**`src/table.rs`, `src/table/state.rs`, `src/table/tests.rs`** — M-M3a's **address half only**: a shared
`flat_index(lane_bits, lanes, lane, shard)`, `Flat::lane_bits`, `Flat::index` as shift+add, and
`Flat::reserve` refusing a non-power-of-two lane count, with the bijection + refusal tests. The
in-slot word fusion is **not** taken (measured +3 insns / +8 B/slot; refuted).

Gates on the lane trees: `cargo test --locked --lib table` 106 passed; `--test table --test
native_locator` 19 + 59 passed; `cargo fmt --all -- --check` clean. Control: drop the power-of-two
refusal -> the `geteach` out-of-range panic plus `flat_reserve_refuses_a_non_power_of_two_lane_count`
fire; restore -> green. Measured: resident hit path **37 -> 34** instructions (`objdump -d`, private
target per variant, base reproduced independently).

**Acceptance stream (this commit's checkpoint).** Five paired full-profile runs, guard limits opened so
contention is averaged rather than refused: four completed (run 3 hit the 560 s timeout). Per-pair
**mean** across runs — `hashtrinity_table` **1.795** (1.716–1.885), `hashtrinity_count_par` **0.953**
(0.917–0.985), `hashtrinity_get_each` **1.007** (0.972–1.050). `full_acceptance=false`; the map face is
still ~3.6x short of the 0.5 target and the count face is at parity, not below. The 32 absolute-baseline
`regression_failure` rows are the contended box (load1 75), not a code regression: the within-run ratios
are the measurement and they moved only within their spread.

Open: `proof/regen.sh --check` **will drift** on the `Flat` layout change — regeneration is a separate
barrier task. Grade: `measured` (campaign + instruction counts) / `tested` (lane gates). No `proved`.
