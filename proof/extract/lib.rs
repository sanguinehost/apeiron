//! Extraction root for Aeneas.
//!
//! This is a development-only translation unit, not a crate in the Apeiron workspace.
//! Charon drives rustc over this single file; the modules it pulls in by `#[path]` are the
//! real `src/scalar.rs`, `src/cell.rs`, `src/exchange.rs`, `src/carrier.rs`, `src/table.rs` and
//! `src/coord.rs`, so the extracted definitions track the source the library builds. The wrappers
//! below exist only to give Charon concrete, monomorphic roots and to call the real methods; they do
//! not restate any formula.
//!
//! Regenerate with `proof/regen.sh`.

#[path = "../../src/carrier.rs"]
pub mod carrier;
#[path = "../../src/cell.rs"]
pub mod cell;
#[path = "../../src/coord.rs"]
pub mod coord;
#[path = "../../src/exchange.rs"]
pub mod exchange;
#[path = "../../src/scalar.rs"]
pub mod scalar;
#[path = "../../src/table.rs"]
pub mod table;

/// The carried-key word instance used to probe the executing table bodies.
pub type WordStoredTable = table::Table<i64, i64, table::WordHome, 16, table::keyrep::Stored<i64>>;
/// The historical default-face word root remains the stored-carrier instance.
pub type WordTable = WordStoredTable;
/// The elliptic word table retains the real regenerated identity word.
pub type WordRegenTable =
    table::Table<i64, i64, table::WordHome, 16, table::keyrep::Regen<table::keyrep::I64WordCodec>>;

/// `Cell::<i64>::mul`, rooted so Charon does not have to translate the whole crate.
pub fn word_mul(a: cell::Cell<i64>, sigma: cell::Sigma, b: cell::Cell<i64>) -> cell::Cell<i64> {
    a.mul(sigma, b)
}

/// `Cell::<i64>::norm_form`, rooted for the same reason.
pub fn word_norm(a: cell::Cell<i64>, sigma: cell::Sigma) -> i64 {
    a.norm_form(sigma)
}

/// `carrier::point_density`, the phase-cell density, rooted at the concrete `i64` scalar.
///
/// This calls the real `src/carrier.rs` generic kernel with `R = i64`; the `Ring` instance is the
/// same `i64` instance `word_mul`/`word_norm` use, so the extracted `carrier.point_density` is the
/// production body rather than a restatement.
pub fn word_point_density(sigma: cell::Sigma, amplitude: i64, direction: cell::Cell<i64>) -> i64 {
    carrier::point_density(sigma, amplitude, direction)
}

/// `carrier::point_flux`, the phase-cell flux, rooted at the concrete `i64` scalar.
pub fn word_point_flux(
    sigma: cell::Sigma,
    amplitude: i64,
    direction: cell::Cell<i64>,
    previous: i64,
    next: i64,
) -> i64 {
    carrier::point_flux(sigma, amplitude, direction, previous, next)
}

/// `carrier::conservative_density_step`, the exact checked scalar update, at the concrete `i64`
/// scalar. This calls the real `src/carrier.rs` body; it does not restate the formula.
pub fn word_conservative_density_step(
    density: i64,
    incoming: i64,
    outgoing: i64,
    source: i64,
) -> Option<i64> {
    carrier::conservative_density_step(density, incoming, outgoing, source)
}

/// `carrier::diffusive_flux_half`, the exact checked face half-flux, at the concrete `i64` scalar.
pub fn word_diffusive_flux_half(left: i64, right: i64) -> Option<i64> {
    carrier::diffusive_flux_half(left, right)
}

/// `carrier::diffusive_line3_step`, the exact three-cell synchronous update, at the concrete
/// `i64` scalar. This calls the real `src/carrier.rs` body; it does not restate the composition.
pub fn word_diffusive_line3_step(
    old: carrier::DensityLine3,
    source: carrier::DensityLine3,
) -> Option<carrier::DensityLine3> {
    carrier::diffusive_line3_step(old, source)
}

/// The derived impulse `p - q`, rooted at the concrete `i64` scalar.
pub fn word_exchange_impulse(p: cell::Cell<i64>, q: cell::Cell<i64>) -> cell::Cell<i64> {
    exchange::exchange_impulse(p, q)
}

/// The equal-mass elastic exchange, through the derived impulse.
pub fn word_elastic_exchange(p: cell::Cell<i64>, q: cell::Cell<i64>) -> exchange::Exchange<i64> {
    exchange::elastic_exchange(p, q)
}

/// The checked-claim kernel, rooted at the concrete `i64` scalar.
///
/// `accept_derived_impulse` compares the claim against the derived impulse with an explicit `if`,
/// so Charon translates the branch and the derived `Cell` equality rather than the external
/// `core.bool.Bool.then` combinator. `regen.sh`'s audit still rejects any generated `axiom`.
pub fn word_accept_derived_impulse(
    p: cell::Cell<i64>,
    q: cell::Cell<i64>,
    claimed: cell::Cell<i64>,
) -> Option<exchange::Exchange<i64>> {
    exchange::accept_derived_impulse(p, q, claimed)
}

/// Executing-table construction with a concrete word home callback.
pub fn word_table_new(default: i64) -> WordTable {
    WordTable::new(default, table::WordHome)
}

/// Executing-table total lookup, returning a copied word value.
pub fn word_table_get(state: &WordTable, key: i64) -> i64 {
    *state.get(&key)
}

/// Executing-table insertion, with the successor state returned explicitly.
pub fn word_table_insert(
    mut state: WordTable,
    key: i64,
    value: i64,
) -> (WordTable, Result<(), table::StorageError>) {
    let result = state.insert_with(&table::WordMerge, key, value);
    (state, result)
}

/// Executing-table replacement insertion with the concrete new-first callback.
pub fn word_table_insert_replacement(
    mut state: WordTable,
    key: i64,
    value: i64,
) -> (WordTable, Result<(), table::StorageError>) {
    let result = state.insert_with(&table::ReplacementMerge, key, value);
    (state, result)
}

/// Executing-table stream build with concrete word callbacks.
pub fn word_table_build(
    default: i64,
    kvs: &[(i64, i64)],
) -> Result<WordTable, table::StorageError> {
    WordTable::build(&table::WordMerge, default, kvs, table::WordHome)
}

/// The real elliptic codec encoding operation.
pub fn word_i64_codec_encode(sigma: cell::Sigma, key: i64) -> u64 {
    let home = table::WordHome;
    let address = table::Home::address(&home, sigma, &key);
    <table::keyrep::I64WordCodec as table::keyrep::ExactCodec<i64, table::WordHome>>::encode(
        sigma, &key, &address,
    )
}

/// The real elliptic codec decoding operation.
pub fn word_i64_codec_decode(sigma: cell::Sigma, identity: u64) -> i64 {
    <table::keyrep::I64WordCodec as table::keyrep::ExactCodec<i64, table::WordHome>>::decode(
        sigma, identity,
    )
}

/// The real regenerated-key hold operation.
pub fn word_regen_hold(
    sigma: cell::Sigma,
    key: i64,
) -> table::keyrep::Regen<table::keyrep::I64WordCodec> {
    let home = table::WordHome;
    let address = table::Home::address(&home, sigma, &key);
    <table::keyrep::Regen<table::keyrep::I64WordCodec> as table::keyrep::KeyRep<
        i64,
        table::WordHome,
    >>::hold(sigma, &key, &address)
}

/// The real regenerated-key identity comparison.
pub fn word_regen_matches(
    sigma: cell::Sigma,
    held: table::keyrep::Regen<table::keyrep::I64WordCodec>,
    key: i64,
) -> bool {
    let home = table::WordHome;
    let address = table::Home::address(&home, sigma, &key);
    <table::keyrep::Regen<table::keyrep::I64WordCodec> as table::keyrep::KeyRep<
        i64,
        table::WordHome,
    >>::matches(sigma, &held, &key, &address)
}

/// The real regenerated-key cold materialization operation.
pub fn word_regen_key_owned(
    sigma: cell::Sigma,
    held: table::keyrep::Regen<table::keyrep::I64WordCodec>,
) -> i64 {
    <table::keyrep::Regen<table::keyrep::I64WordCodec> as table::keyrep::KeyRep<
        i64,
        table::WordHome,
    >>::key_owned(sigma, &held)
}

/// The real regenerated-key address recovery operation.
pub fn word_regen_address_held(
    sigma: cell::Sigma,
    held: table::keyrep::Regen<table::keyrep::I64WordCodec>,
) -> table::TableAddress {
    let home = table::WordHome;
    <table::keyrep::Regen<table::keyrep::I64WordCodec> as table::keyrep::KeyRep<
        i64,
        table::WordHome,
    >>::address_held(&home, sigma, &held)
}

/// Real construction of the `Sigma`-qualified stored-key table.
pub fn word_stored_table_new_at(default: i64) -> WordStoredTable {
    WordStoredTable::new(default, table::WordHome)
}

/// Real `get_at` through the stored-key carrier.
///
/// This is the selected public-read extraction root: Charon translates the real
/// [`table::Table::get_at`] body, its inlined resident classification/index, and the shared
/// read-only `spill_read` matcher. The wrapper only fixes the carrier and copies the borrowed
/// result; it does not restate resident or spill matching.
pub fn word_stored_table_get_at(state: &WordStoredTable, sigma: cell::Sigma, key: i64) -> i64 {
    *state.get_at(sigma, &key)
}

/// Real `insert_with_at` through the stored-key carrier.
pub fn word_stored_table_insert_with_at(
    mut state: WordStoredTable,
    sigma: cell::Sigma,
    key: i64,
    value: i64,
) -> (WordStoredTable, Result<(), table::StorageError>) {
    let result = state.insert_with_at(&table::WordMerge, sigma, key, value);
    (state, result)
}

/// Real `build_at` through the stored-key carrier.
pub fn word_stored_table_build_at(
    default: i64,
    kvs: &[((cell::Sigma, i64), i64)],
) -> Result<WordStoredTable, table::StorageError> {
    WordStoredTable::build_at(&table::WordMerge, default, kvs, table::WordHome)
}

/// Real construction of the `Sigma`-qualified regenerated-key table.
pub fn word_regen_table_new_at(default: i64) -> WordRegenTable {
    WordRegenTable::new(default, table::WordHome)
}

/// Real `get_at` through the regenerated-key carrier.
///
/// The regenerated-carrier public-read extraction root, selected alongside
/// [`word_stored_table_get_at`]. Same real [`table::Table::get_at`] body and shared `spill_read`
/// matcher; only the carrier differs.
pub fn word_regen_table_get_at(state: &WordRegenTable, sigma: cell::Sigma, key: i64) -> i64 {
    *state.get_at(sigma, &key)
}

/// Real `insert_with_at`, through the resident-or-spill update path.
pub fn word_regen_table_insert_with_at(
    mut state: WordRegenTable,
    sigma: cell::Sigma,
    key: i64,
    value: i64,
) -> (WordRegenTable, Result<(), table::StorageError>) {
    let result = state.insert_with_at(&table::WordMerge, sigma, key, value);
    (state, result)
}

/// Real `build_at`; its short-input branch is `build_at_sequential` and its long-input branch
/// invokes the real deterministic partition and LPT schedule before the same insertion kernel.
pub fn word_regen_table_build_at(
    default: i64,
    kvs: &[((cell::Sigma, i64), i64)],
) -> Result<WordRegenTable, table::StorageError> {
    WordRegenTable::build_at(&table::WordMerge, default, kvs, table::WordHome)
}

/// Reads the real non-allocating prepared spill core through the stored carrier.
pub fn word_stored_prepared_shard_get(
    state: &WordStoredTable,
    sigma: cell::Sigma,
    key: i64,
) -> i64 {
    *state.get_prepared_shard_at(sigma, &key)
}

/// Applies the real non-allocating prepared spill core through the stored carrier.
/// The caller is the resource adapter and supplies an initialized region with capacity.
pub fn word_stored_prepared_shard_apply(
    mut state: WordStoredTable,
    sigma: cell::Sigma,
    key: i64,
    value: i64,
) -> (WordStoredTable, Result<(), table::StorageError>) {
    let result = state.apply_prepared_shard_at(&table::WordMerge, sigma, &key, value);
    (state, result)
}

/// Reads the real non-allocating prepared spill core through the regenerated carrier.
pub fn word_regen_prepared_shard_get(state: &WordRegenTable, sigma: cell::Sigma, key: i64) -> i64 {
    *state.get_prepared_shard_at(sigma, &key)
}

/// Applies the real non-allocating prepared spill core through the regenerated carrier.
/// The caller is the resource adapter and supplies an initialized region with capacity.
pub fn word_regen_prepared_shard_apply(
    mut state: WordRegenTable,
    sigma: cell::Sigma,
    key: i64,
    value: i64,
) -> (WordRegenTable, Result<(), table::StorageError>) {
    let result = state.apply_prepared_shard_at(&table::WordMerge, sigma, &key, value);
    (state, result)
}

/// Real straight-line read apply through the stored carrier.
///
/// The composite decision walk (`Table::search_decision`, and the shared `spill_locate` scan it
/// uses) stays unextracted under `PinnedJoinBoundary`; the caller supplies the already-computed
/// [`table::SearchDecision`]. This calls the real `Table::apply_get`, which indexes the selected
/// real resident/spill slot and returns the real default for every non-`Found` classification.
pub fn word_stored_apply_get(state: &WordStoredTable, decision: table::SearchDecision) -> i64 {
    *state.apply_get(decision)
}

/// Real straight-line insert apply through the stored carrier.
///
/// Consumes the real [`table::ApplyOutcome`] explicitly: `Committed` reports no pending entry;
/// `NeedsGrowth` returns the pending `(key, value)` unchanged for the unextracted resource
/// adapter. The call performs no probe, reservation or allocation.
pub fn word_stored_apply_insert(
    mut state: WordStoredTable,
    decision: table::SearchDecision,
    sigma: cell::Sigma,
    key: i64,
    value: i64,
    address: table::TableAddress,
) -> (WordStoredTable, Option<(i64, i64)>) {
    match state.apply_insert(decision, &table::WordMerge, sigma, key, value, &address) {
        table::ApplyOutcome::Committed => (state, None),
        table::ApplyOutcome::NeedsGrowth(key, value) => (state, Some((key, value))),
    }
}

/// Real straight-line read apply through the regenerated carrier.
pub fn word_regen_apply_get(state: &WordRegenTable, decision: table::SearchDecision) -> i64 {
    *state.apply_get(decision)
}

/// Real straight-line insert apply through the regenerated carrier.
pub fn word_regen_apply_insert(
    mut state: WordRegenTable,
    decision: table::SearchDecision,
    sigma: cell::Sigma,
    key: i64,
    value: i64,
    address: table::TableAddress,
) -> (WordRegenTable, Option<(i64, i64)>) {
    match state.apply_insert(decision, &table::WordMerge, sigma, key, value, &address) {
        table::ApplyOutcome::Committed => (state, None),
        table::ApplyOutcome::NeedsGrowth(key, value) => (state, Some((key, value))),
    }
}

/// Real deterministic LPT schedule completeness check.
pub fn word_schedule_is_complete(
    loads: &[usize; table::partition::SHARDS],
    workers: usize,
) -> bool {
    table::partition::Schedule::lpt(loads, workers).is_complete()
}

/// `coord::finish`, the splitmix64 finalizer, rooted at the concrete `u64` word.
///
/// This calls the real `src/coord.rs` body; it does not restate the xor-shift chain.
pub fn word_coord_finish(h: u64) -> u64 {
    coord::finish(h)
}

/// `coord::spread`, the one-widening-multiply address split, rooted at the concrete `u64` word.
///
/// This calls the real `src/coord.rs` body; it does not restate the widening multiply or the
/// rotate/xor that folds the two halves.
pub fn word_coord_spread(x: u64) -> coord::Addr {
    coord::spread(x)
}

/// `coord::unspread`, the identity-word inverse multiply, rooted at the concrete `u64` word.
pub fn word_coord_unspread(id: u64) -> u64 {
    coord::unspread(id)
}

/// The identity word of a `coord::Addr`, read as a word.
///
/// A field projection of the real `coord::Addr`; it selects the identity half that
/// [`word_coord_spread`] produced and states no arithmetic of its own.
pub fn word_coord_id(address: coord::Addr) -> u64 {
    address.id
}

/// The placement word of a `coord::Addr`, read as a word.
///
/// A field projection of the real `coord::Addr`; it selects the placement half that
/// [`word_coord_spread`] produced and states no arithmetic of its own.
pub fn word_coord_place(address: coord::Addr) -> u64 {
    address.place
}

/// The real production per-entry relocation core through the stored carrier.
///
/// The caller is the unextracted prepared-resource adapter: it supplies the checked `RebuildPlan`
/// and a materialized `RebuildReserved` with vacant destination occupancy. This calls the real
/// `Table::relocate_entry` body and returns the updated table with the reserved storage; it does
/// not restate the flat/spill destination write.
pub fn word_stored_relocate_entry(
    mut state: WordStoredTable,
    plan: &table::RebuildPlan,
    mut reserved: table::RebuildReserved<table::keyrep::Stored<i64>, i64>,
    entry_at: usize,
) -> (
    WordStoredTable,
    table::RebuildReserved<table::keyrep::Stored<i64>, i64>,
) {
    state.relocate_entry(plan, &mut reserved, entry_at);
    (state, reserved)
}

/// The real production per-entry relocation core through the regenerated carrier.
///
/// Same real body as [`word_stored_relocate_entry`], specialized at the elliptic regenerated
/// carrier. The adapter supplies the checked plan and materialized reserved storage.
pub fn word_regen_relocate_entry(
    mut state: WordRegenTable,
    plan: &table::RebuildPlan,
    mut reserved: table::RebuildReserved<table::keyrep::Regen<table::keyrep::I64WordCodec>, i64>,
    entry_at: usize,
) -> (
    WordRegenTable,
    table::RebuildReserved<table::keyrep::Regen<table::keyrep::I64WordCodec>, i64>,
) {
    state.relocate_entry(plan, &mut reserved, entry_at);
    (state, reserved)
}

/// The real production relocation traversal through the stored carrier.
///
/// The caller is the unextracted prepared-resource adapter: it supplies the checked `RebuildPlan`
/// and a materialized `RebuildReserved` with vacant destination occupancy. This calls the real
/// `Table::relocate_all` body, which walks `plan.entries` in order through the real
/// `Table::relocate_entry`; it does not restate the traversal or the per-entry write.
pub fn word_stored_relocate_all(
    mut state: WordStoredTable,
    plan: &table::RebuildPlan,
    mut reserved: table::RebuildReserved<table::keyrep::Stored<i64>, i64>,
) -> (
    WordStoredTable,
    table::RebuildReserved<table::keyrep::Stored<i64>, i64>,
) {
    state.relocate_all(plan, &mut reserved);
    (state, reserved)
}

/// The real production relocation traversal through the regenerated carrier.
///
/// Same real body as [`word_stored_relocate_all`], specialized at the elliptic regenerated carrier.
pub fn word_regen_relocate_all(
    mut state: WordRegenTable,
    plan: &table::RebuildPlan,
    mut reserved: table::RebuildReserved<table::keyrep::Regen<table::keyrep::I64WordCodec>, i64>,
) -> (
    WordRegenTable,
    table::RebuildReserved<table::keyrep::Regen<table::keyrep::I64WordCodec>, i64>,
) {
    state.relocate_all(plan, &mut reserved);
    (state, reserved)
}

/// The real pending-insertion and commit core through the stored carrier.
///
/// The caller is the unextracted prepared-resource adapter: it supplies the materialized
/// `RebuildReserved` with the relocation already applied and the precomputed `SearchDecision` for
/// the pending `(sigma, key)`. This calls the real `Table::apply_pending_and_commit` body, which
/// consumes the decision without re-probing, merges once, and commits the reserved storage; it does
/// not restate the class/location match.
pub fn word_stored_apply_pending_and_commit(
    mut state: WordStoredTable,
    reserved: table::RebuildReserved<table::keyrep::Stored<i64>, i64>,
    decision: table::SearchDecision,
    sigma: cell::Sigma,
    value: i64,
    held: table::keyrep::Stored<i64>,
    address: table::TableAddress,
) -> WordStoredTable {
    state.apply_pending_and_commit(
        reserved,
        decision,
        &table::WordMerge,
        sigma,
        value,
        held,
        &address,
    );
    state
}

/// The real pending-insertion and commit core through the regenerated carrier.
///
/// Same real body as [`word_stored_apply_pending_and_commit`], specialized at the elliptic
/// regenerated carrier. The adapter supplies the relocated reserved storage and the decision.
pub fn word_regen_apply_pending_and_commit(
    mut state: WordRegenTable,
    reserved: table::RebuildReserved<table::keyrep::Regen<table::keyrep::I64WordCodec>, i64>,
    decision: table::SearchDecision,
    sigma: cell::Sigma,
    value: i64,
    held: table::keyrep::Regen<table::keyrep::I64WordCodec>,
    address: table::TableAddress,
) -> WordRegenTable {
    state.apply_pending_and_commit(
        reserved,
        decision,
        &table::WordMerge,
        sigma,
        value,
        held,
        &address,
    );
    state
}

/// The real nonallocating publication step through the stored carrier.
///
/// Calls the real `table::resources::publish_prepared` at `R = Stored<i64>`, `V = i64`; it moves
/// the adapter's exact success tuple into `PreparedResult`, or maps `Err(e)` to
/// `PreparedResult::Refused`. It does not plan, reserve, read capacity, initialize or certify a
/// predicate.
pub fn word_stored_publish_prepared(
    result: Result<
        table::resources::PreparedEnvelope<table::keyrep::Stored<i64>, i64>,
        table::StorageError,
    >,
) -> table::resources::PreparedResult<table::keyrep::Stored<i64>, i64> {
    table::resources::publish_prepared(result)
}

/// The real nonallocating publication step through the regenerated carrier.
///
/// Same real body as [`word_stored_publish_prepared`], specialized at the elliptic regenerated
/// carrier.
pub fn word_regen_publish_prepared(
    result: Result<
        table::resources::PreparedEnvelope<table::keyrep::Regen<table::keyrep::I64WordCodec>, i64>,
        table::StorageError,
    >,
) -> table::resources::PreparedResult<table::keyrep::Regen<table::keyrep::I64WordCodec>, i64> {
    table::resources::publish_prepared(result)
}

/// The real production scalar capacity comparison.
///
/// Calls the real `table::resources::capacity_bound_holds`; it does not restate the comparison.
pub fn word_capacity_bound_holds(required_len: usize, observed_capacity: usize) -> bool {
    table::resources::capacity_bound_holds(required_len, observed_capacity)
}
