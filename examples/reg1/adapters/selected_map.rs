//! The selected side of a map pair: the incumbent's own body, compiled into this crate.
//!
//! `strata.rs` measures every current-strata cell against the pinned `hashtrinity::Table`. This
//! child module holds the selected side's three bodies — build, resident touch, timed schedule —
//! plus the identity that says which table they ran on. It is a separate file to keep `strata.rs`
//! below the source-size cap, and it is included by `strata.rs` so every access path stays
//! `crate::strata::{..}`.
//!
//! **The selected side is the incumbent's algorithm, not a faster one.** Before the wire it was
//! this crate's own `apeiron::Table`, so the `hashtrinity_table` pair began around 1.75 and every
//! mechanism attempt had been made from below parity. Now it is `src/hashtrinity/` — the same
//! source at the same pinned revision `4cc14471ae24f3f204fb0b0e91736ef75c909358`, compiled into
//! this crate — and the incumbent arms keep executing the unchanged external comparator through
//! the `hashtrinity` git dependency. The two differ only in which crate owns the type, so the pair
//! begins at parity and a ratio below `1.00` from here is what the vendored body removed.
//!
//! The recorded implementation identity is read from the running expression by [`type_name_of`],
//! not written beside it: [`VENDORED_MAP_SYMBOL`] is what the vendored body must produce and
//! [`EXTERNAL_MAP_SYMBOL`] is what the unchanged comparator produces, and
//! [`super::validate_selected_impl`] rejects a cell whose recorded symbol is the second one.

use std::hint::black_box;
use std::time::Instant;

use super::{fold, initial_value, nanos_since, MergeFn, RunOutcome, StrataKey, Trace, FNV_OFFSET};

/// The selected side's type: the same table at the same pinned revision, compiled into this crate
/// as `src/hashtrinity/` and reached through `apeiron::freq::Table`.
pub(crate) type SelectedMap = apeiron::freq::Table<StrataKey, i64>;

/// The compiler's own name for the type of `value`, as `type_name` on the value rather than on a
/// type written beside it.
///
/// `std::any::type_name::<T>()` names a type the caller names, which is a label. Threading the
/// value through a generic shim infers `T` from the running expression, so the recorded identity
/// changes when the expression does. `O(1)`: a `&'static str` the compiler already carries.
pub(super) fn type_name_of<T: ?Sized>(value: &T) -> &'static str {
    let _ = value;
    std::any::type_name::<T>()
}

/// The registered implementation identity of the selected map arm.
///
/// The selected side of a map pair must execute the incumbent's body **compiled into this crate**
/// (`src/hashtrinity/`, `hashtrinity` at rev `4cc14471ae24f3f204fb0b0e91736ef75c909358`), not the
/// unchanged external comparator that the `hashtrinity` git dependency supplies. The two are the
/// same source at the same revision and differ only in which crate owns the type, so the compiler's
/// own name is the discriminator: the vendored table is named under this crate's module, the
/// external one under the dependency's.
///
/// The prefix names the **defining** module, which is what `std::any::type_name` reports and not
/// the `apeiron::freq` path this crate re-exports it under — measured on this tree:
/// `type_name::<apeiron::freq::Table<u64, i64>>()` is `apeiron::hashtrinity::Table<u64, i64>`, while
/// `type_name::<hashtrinity::Table<u64, i64>>()` is `hashtrinity::Table<u64, i64>`. Only the prefix
/// is compared, because the key type's own module path (`reg1::strata::StrataKey`) is not part of
/// the registered contract.
pub(crate) const VENDORED_MAP_SYMBOL: &str = "apeiron::hashtrinity::Table<";

/// The same implementation as the unchanged external comparator the incumbent arms execute.
/// Registered so a mismatch is reported against a named alternative and not against an absence.
pub(crate) const EXTERNAL_MAP_SYMBOL: &str = "hashtrinity::Table<";

/// Build the **selected** side's table, returning it, its live length, and the compiler's name for
/// its type.
///
/// The same schedule as the incumbent's `build_ht` — `with_capacity(keys.len())`, then one
/// `insert_with(key, initial_value(key), merge)` per present key — over [`SelectedMap`]. The
/// identity is read from the table this body built by [`type_name_of`], so redirecting the body to
/// the external comparator changes the recorded symbol and
/// [`super::validate_selected_impl`] rejects the cell by name.
pub(super) fn build_selected_map(
    keys: &[StrataKey],
    merge: MergeFn,
) -> (SelectedMap, usize, &'static str) {
    let mut table = SelectedMap::with_capacity(keys.len());
    for key in keys {
        table.insert_with(*key, initial_value(*key), merge);
    }
    let len = table.len();
    let impl_symbol = type_name_of(&table);
    (table, len, impl_symbol)
}

/// Touch every present key so the selected table is resident before a timed trace.
///
/// The selected side's table stores keys exactly, so every present key reads back and the
/// `unwrap_or(0)` arm is unreachable for this population; it is written because it is the shape
/// the incumbent's own read path has, and the two sides must differ only in which crate owns the
/// type. Temporal and spatial `O(keys.len())`, one hashable `get` per key and no allocation.
pub(super) fn touch_selected_map(table: &SelectedMap, keys: &[StrataKey]) {
    let mut acc = 0i64;
    for key in keys {
        acc = acc.wrapping_add(table.get(key).copied().unwrap_or(0));
    }
    black_box(acc);
}

/// Run the timed schedule against the **selected** side's table, then the untimed canonical
/// readback.
///
/// The incumbent's own schedule over [`SelectedMap`]: the same lookup loop, the same
/// `insert_with(k, v, merge)` fold and the same canonical readback as `measure_ht`, on the type
/// this crate owns rather than the one the git dependency owns. Nothing is excluded from the timed
/// region, so both sides' `total_nanos` cover the same work.
///
/// Temporal `O(lookups + inserts + present)` and spatial `O(1)` beyond the trace: two folds per
/// scheduled op, one per present key in the readback, and no allocation in either timed window.
pub(super) fn measure_selected_map(
    table: &mut SelectedMap,
    trace: &Trace,
    keys: &[StrataKey],
    merge: MergeFn,
) -> RunOutcome {
    let table = black_box(table);
    let mut hash = FNV_OFFSET;
    let start = Instant::now();
    for key in &trace.lookups {
        fold(&mut hash, table.get(black_box(key)).copied().unwrap_or(0));
    }
    let lookup_nanos = nanos_since(start);
    let start = Instant::now();
    for (key, value) in &trace.inserts {
        table.insert_with(black_box(*key), black_box(*value), merge);
        fold(&mut hash, 0);
        fold(&mut hash, *value);
    }
    let insert_nanos = nanos_since(start);
    for key in keys {
        fold(&mut hash, table.get(black_box(key)).copied().unwrap_or(0));
    }
    RunOutcome {
        digest: hash,
        lookup_nanos,
        insert_nanos,
    }
}

/// Check that a cell's selected side executed the **vendored** incumbent.
///
/// A map cell's ratio only means something about `src/hashtrinity/` if the selected side runs that
/// code. Before the wire the selected side ran this crate's own `Table`, so the pair began far
/// above parity and every edit to the vendored copy moved no number; the recorded symbol is what
/// makes that state detectable rather than invisible. The symbol is read from the running
/// expression ([`type_name_of`]) and compared against [`VENDORED_MAP_SYMBOL`]; the unchanged
/// external comparator ([`EXTERNAL_MAP_SYMBOL`]) does not name this crate's module and is rejected
/// by name.
///
/// `O(1)` time and space: one prefix comparison per cell.
pub(super) fn validate_selected_impl(cell: &super::CellSamples) -> Result<(), String> {
    if !cell.selected_impl_symbol.starts_with(VENDORED_MAP_SYMBOL) {
        return Err(format!(
            "strata cell {} selected side executes {} but the registered selected map \
             implementation is the vendored incumbent, whose symbol begins {VENDORED_MAP_SYMBOL}; \
             {} is the unchanged external comparator and the selected side must not be it",
            cell.cell_id, cell.selected_impl_symbol, EXTERNAL_MAP_SYMBOL
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        build_selected_map, validate_selected_impl, EXTERNAL_MAP_SYMBOL, VENDORED_MAP_SYMBOL,
    };
    use crate::strata::{
        all_specs, make_plan, merge_fn, probe_capacity, run_cell, validate_strata, CellSpec,
        Distribution, MergeMode, Operation, StrataKey, StrataWitness, SHARDS,
    };
    use hashtrinity::Coord;

    /// The smallest occupancy, one merge and one distribution: the identity check reads one
    /// symbol, so the cheapest cell that reaches a timed run exercises it without a campaign's
    /// worth of work. One campaign of two samples keeps the control inside the test's own budget.
    const CONTROL: CellSpec = CellSpec {
        occupancy: 1,
        distribution: Distribution::DirectIndexResident,
        operation: Operation::PresentHit,
        merge: MergeMode::Replacement,
    };

    fn one_cell() -> crate::strata::CellSamples {
        run_cell(CONTROL, 1)
    }

    fn witness_of(cell: crate::strata::CellSamples) -> StrataWitness {
        // The summary counts are derived from the cell rather than written beside it, for the same
        // reason the symbol is: a witness whose counts were chosen by hand would reject a measured
        // cell for an unrelated reason and the identity check inside it would never be reached.
        let (passed, failed) = match cell.status {
            crate::strata::CellStatus::Pass => (1, 0),
            crate::strata::CellStatus::Fail => (0, 1),
            _ => (0, 0),
        };
        StrataWitness {
            cells: vec![cell],
            resources: Vec::new(),
            profile: "selected-map-identity-control".to_string(),
            campaigns: 1,
            warmup: 1,
            // `run_cell` appends `STRATA_SAMPLES` samples per campaign whatever the witness
            // declares, and `validate_cell` checks the product, so the witness states the real
            // sample count rather than a convenient one.
            samples: crate::strata::STRATA_SAMPLES,
            ops: 4096,
            measured: 1,
            missing: 0,
            rejected: 0,
            passed,
            failed,
            not_applicable: 0,
            ht_rev: String::new(),
            rustc: String::new(),
            cpu: String::new(),
        }
    }

    /// The registered identity symbols are the compiler's, not a written-down pair of strings that
    /// happens to differ.
    ///
    /// A discriminating check whose two sides were chosen by hand discriminates nothing: both
    /// literals could be wrong in the same direction and every run would still agree with them. So
    /// both are read back off the two real table types — the vendored one and the external
    /// comparator's — and the check is only meaningful because they come apart.
    #[test]
    fn registered_map_symbols_are_the_compilers_own() {
        let vendored = std::any::type_name::<apeiron::freq::Table<StrataKey, i64>>();
        let external = std::any::type_name::<hashtrinity::Table<StrataKey, i64>>();
        assert!(
            vendored.starts_with(VENDORED_MAP_SYMBOL),
            "the vendored table is named {vendored}, which does not begin the registered \
             {VENDORED_MAP_SYMBOL}"
        );
        assert!(
            external.starts_with(EXTERNAL_MAP_SYMBOL),
            "the external comparator is named {external}, which does not begin the registered \
             {EXTERNAL_MAP_SYMBOL}"
        );
        assert!(
            !external.starts_with(VENDORED_MAP_SYMBOL),
            "the external comparator's symbol {external} is accepted as the vendored one, so the \
             identity check cannot fail"
        );
    }

    /// The gate itself: an uncorrupted cell runs the vendored body, records the compiler's name for
    /// the type it built, and passes [`super::validate_selected_impl`] and the whole witness.
    ///
    /// This is what the mutation control attacks: redirecting
    /// [`build_selected_map`] back to `hashtrinity::Table` changes the recorded symbol and fails
    /// [`selected_map_identity_mismatch_rejects`].
    #[test]
    fn selected_map_runs_the_vendored_body() {
        let cell = one_cell();
        // `Pass` or `Fail` are the two *measured* verdicts; this control asserts the identity, not
        // the ratio. A one-campaign debug build's ratio straddles `1.00` by a few parts per
        // thousand, so pinning the verdict here would make the control a speed test.
        assert!(
            matches!(
                cell.status,
                crate::strata::CellStatus::Pass | crate::strata::CellStatus::Fail
            ),
            "the control cell must be measured, not missing or rejected: {} ({:?})",
            cell.reason,
            cell.status
        );
        assert!(
            cell.selected_impl_symbol.starts_with(VENDORED_MAP_SYMBOL),
            "cell {} recorded implementation {} but the registered selected map implementation \
             begins {VENDORED_MAP_SYMBOL}",
            cell.cell_id,
            cell.selected_impl_symbol
        );
        validate_selected_impl(&cell).expect("the vendored symbol is the registered one");
        // The identity is not the only thing that has to hold: the whole witness validates.
        validate_strata(&witness_of(cell)).unwrap_or_else(|e| {
            panic!("a cell whose selected side is the vendored incumbent must pass: {e}")
        });
    }

    /// The rejecting control: a redirected selected side is rejected by name, and the message names
    /// the cell, the arm's registered symbols and the alternative.
    ///
    /// Without this, a check that accepted everything would also pass the gate above.
    #[test]
    fn selected_map_identity_mismatch_rejects() {
        let mut cell = one_cell();
        let cell_id = cell.cell_id.clone();
        cell.selected_impl_symbol = EXTERNAL_MAP_SYMBOL;
        let error = validate_selected_impl(&cell)
            .expect_err("the external comparator is not the registered selected implementation");
        for expected in [cell_id.as_str(), EXTERNAL_MAP_SYMBOL, VENDORED_MAP_SYMBOL] {
            assert!(
                error.contains(expected),
                "the mismatch must name {expected}; the message was: {error}"
            );
        }
        // The whole witness rejects too, so the check is on the mandatory path and not only
        // callable by hand.
        let error = validate_strata(&witness_of(cell))
            .expect_err("a cell whose selected side is the external comparator must reject");
        assert!(
            error.contains(VENDORED_MAP_SYMBOL),
            "the witness-level rejection must name the registered symbol; it was: {error}"
        );
    }

    /// The two sides' bodies place a key at the same slot, which is the property the pair's claim
    /// — that the two sides differ only in which crate owns the type — rests on. If the vendored
    /// `Coord` impl disagreed with the external one the ratio would compare different placements
    /// rather than the same one in two crates.
    #[test]
    fn vendored_and_external_coord_impls_agree() {
        let key = StrataKey {
            tag: 0x0123_4567_89AB_CDEF,
        };
        assert_eq!(
            <StrataKey as Coord>::coord(&key),
            <StrataKey as apeiron::freq::Coord>::coord(&key),
            "the two crates' Coord impls must return the same address for the same key"
        );
    }

    /// The build the selected side runs admits the whole present set, and the recorded symbol comes
    /// off the built table rather than off a type written beside it.
    #[test]
    fn build_reports_the_compilers_name_and_full_length() {
        let (capacity, bits) = probe_capacity(SHARDS * CONTROL.occupancy);
        let plan = make_plan(CONTROL, capacity, bits).expect("the control plan is constructible");
        let merge = merge_fn(CONTROL.merge);
        let (_, len, symbol) = build_selected_map(&plan.present, merge);
        assert_eq!(len, plan.present.len());
        assert!(
            symbol.starts_with(VENDORED_MAP_SYMBOL),
            "the build reported {symbol}, which does not begin {VENDORED_MAP_SYMBOL}"
        );
        // And the enumeration is non-empty, so the one-cell witness is a real cell.
        assert!(
            !all_specs().is_empty(),
            "the strata enumeration must contain the cell this control measures"
        );
    }
}
