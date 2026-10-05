# Subsumed map face — landing receipt

Row: `apeiron-map-parity-wire-EhXlkMtFhAOPAkVDTSGtyQ` (APEIRON-MAP-PARITY).
Related: `apeiron-map-register-the-GWgtrZKctFxA8re9qryHse` (APEIRON-MAP-REGISTER).

## What lands

The **selected map arm executes the subsumed incumbent body**. `examples/reg1/adapters/selected_map.rs`
(new) constructs and measures `apeiron::hashtrinity::Table` for the map face; the vendor tree
`src/hashtrinity/**` (36 files) is the subsumed artifact; `Cargo.toml` puts it on the build path.

Registration is **truthful**, which is what makes this a measurement and not a diagnostic
(Sol, `PARITY-CELL-CLOSE`):
- `examples/reg1/acceptance.rs` `SELECTED_IMPL = "apeiron_hashtrinity_map"` (was `"apeiron_table"`).
- `examples/reg1/matrix.json` `current_strata[].impl_id = "apeiron_hashtrinity_map"`,
  `implementation = "apeiron::hashtrinity::Table<StrataKey,i64>"` (was `apeiron::Table<StrataKey,i64,StrataHome,16>`).
- `examples/reg1/matrix.rs` `EXECUTING_SELECTED_ID` / `EXECUTING_SELECTED_SYMBOL`, and the check
  compares the registry against them, so a mismatch refuses.

## Measurement

Live tree, `reg1 --check performance --profile full --accept --controls`,
`REG1_TARGET_RATIO=half REG1_LOAD1_LIMIT=1000000 REG1_HOST_BUSY_LIMIT=1.0 REG1_CONTENTION_TOKENS=''`,
`measurement_verdict=measured`:

| face | ratio_mean | min | max |
|---|---|---|---|
| `hashtrinity_table` | **0.9757** | 0.3527 | 1.1758 |
| `hashtrinity_count_par` | 1.0054 | 0.8170 | 1.8678 |
| `hashtrinity_get_each` | 1.0373 | 0.9102 | 1.3945 |

`HEADLINE pass=706 fail=794 of 1500`, `full_acceptance=false` (the registered target is the 0.50
`hashtrinity-half`, which is an aspiration; parity is the victory condition).

The six-campaign measurement recorded by the MAP-PARITY lane on the same wire:
`hashtrinity_table` mean **0.9912**, sd 0.0073, min 0.9799, max 1.0004.

## Gates (§6, on the live tree, this source)

- `cargo build --workspace --all-features --locked` — exit 0.
- `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` — exit 0.
- `cargo fmt --all -- --check` — exit 0.

## Trust boundary and open items (obligation 11)

- The vendored tree is **51,479 lines across 36 files** — 34x the §4 1,500-line bound. It is the
  subsumed artifact under test, not checking code, but the size acceptance is recorded as open.
- The executed body and the refined body are **not yet the same object**: `proof/corpus/Map/`
  refines `apeiron::Table.get`, while this wire executes `apeiron::hashtrinity::Table`. Closing
  that identity is the phase-1 verification obligation.
- The `count_par` and `get_each` means are just above 1.00 on this single run and within their
  spread; a paired campaign is required before parity is claimed for those two faces.
- `examples/reg1/adapters/insert_charge.rs` is a stray instrument from a cancelled lane with **0
  references**; it is deliberately **not** included.
- A lane wrote to the live tree during this work; the foreign `src/table.rs` patch
  (MAP-WORST-CELL's, which regresses the headline 32 -> 30) is preserved at
  `/home/socol/oc-tmp/land-parity/FOREIGN-insert-cost.patch` and is **not** included.
