//! Extraction root for Aeneas.
//!
//! This is a development-only translation unit, not a crate in the Apeiron workspace.
//! Charon drives rustc over this single file; the modules it pulls in by `#[path]` are the
//! real `src/scalar.rs`, `src/cell.rs`, `src/exchange.rs`, `src/carrier.rs`, `src/table.rs`,
//! `src/coord.rs` and `src/numeric/{fixed_exp,fixed_rsqrt,fixed_trig}.rs`, so the extracted
//! definitions track the source the
//! library builds. The wrappers
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
#[path = "../../src/numeric/fixed_rsqrt.rs"]
pub mod fixed_rsqrt;
#[path = "../../src/numeric/fixed_trig.rs"]
pub mod fixed_trig;
#[path = "../../src/format/gguf.rs"]
pub mod gguf;

#[path = "../../src/numeric/fixed_exp.rs"]
pub mod fixed_exp;
#[path = "../../src/numeric/register.rs"]
pub mod register;

/// Mirror of `crate::numeric` for the extraction unit: the real `src/engine/{silu,norm}.rs` import
/// `crate::numeric::fixed_exp::fixed_exp` and `crate::numeric::fixed_rsqrt::{fixed_rsqrt,RsqrtError}`,
/// and `src/engine/rope.rs` imports `crate::numeric::fixed_trig::fixed_sin_cos`, so the three
/// `numeric::*` aliases name the modules above.
pub mod numeric {
    pub use super::{fixed_exp, fixed_ln, fixed_rsqrt, fixed_trig, register};
}

/// Mirror of `crate::format` for the extraction unit: the real `src/engine/error.rs` imports
/// `crate::format::gguf::GgufError` and `crate::format::quant::QuantError`, and
/// `src/engine/dequant.rs` and `src/engine/matvec.rs` import `crate::format::quant`, so these
/// aliases name the crate-root modules declared below.
pub mod format {
    pub use super::{gguf, packing, quant};
}

#[path = "../../src/engine/value.rs"]
pub mod value;
#[path = "../../src/engine/silu.rs"]
pub mod silu;
#[path = "../../src/engine/norm.rs"]
pub mod norm;
#[path = "../../src/engine/rope.rs"]
pub mod rope;
#[path = "../../src/engine/error.rs"]
pub mod error;
#[path = "../../src/engine/f32_convert.rs"]
pub mod f32_convert;
#[path = "../../src/engine/dequant.rs"]
pub mod dequant;
#[path = "../../src/engine/matvec.rs"]
pub mod matvec;
#[path = "../../src/engine/attn.rs"]
pub mod attn;
#[path = "../../src/engine/mixer.rs"]
pub mod mixer;
#[path = "../../src/engine/lm_head.rs"]
pub mod lm_head;
#[path = "../../src/engine/sampler.rs"]
pub mod sampler;
#[path = "../../src/engine/hadamard.rs"]
pub mod hadamard;
#[path = "../../src/engine/ffn.rs"]
pub mod ffn;
#[path = "../../src/engine/moe.rs"]
pub mod moe;
#[path = "../../src/engine/kern/pow2.rs"]
pub mod pow2;

/// Mirror of `crate::engine` for the extraction unit: `crate::engine::silu` (the real kernel's
/// module path) aliases [`silu`], `crate::engine::value` aliases [`value`], `crate::engine::norm`
/// and `crate::engine::rope` alias the two kernel modules, and `ONE`/`FIXED_FRAC_BITS` are the
/// engine-root constants the kernels read.
///
/// The kernel modules are declared at the crate root (a `#[path]` inside an inline module would
/// resolve against a virtual directory and cannot traverse its non-existent `..`); their
/// `super::value` and `super::FIXED_FRAC_BITS` therefore resolve to the crate-root [`value`] and
/// the crate-root aliases of `ONE`/`FIXED_FRAC_BITS` below, which name the same items. The same
/// reason puts [`attn`]/[`lm_head`]/[`sampler`] at the crate root: their `super::matvec`,
/// `super::value` and `super::MatvecError` imports then name the crate-root items re-exported
/// under the engine mirror.
pub mod engine {
    pub use super::{
        attn, chart, error, f32_convert, ffn, fused, hadamard, kern, lm_head, loader, matvec, mixer,
        moe, norm, par, rope, sampler, silu, step, trace, value,
    };
    pub use super::rope::{RopeError, RopeTable};

    /// The engine refusal type at its real `crate::engine::EngineError` path.
    pub use super::error::EngineError;
    /// The attention refusal and cache at their real `crate::engine` paths.
    pub use super::attn::{AttnError, KvState};
    /// The LM-head refusal at its real `crate::engine::LmHeadError` path.
    pub use super::lm_head::LmHeadError;
    /// The projection refusal at its real `crate::engine::MatvecError` path.
    pub use super::matvec::MatvecError;
    /// The sampler refusal at its real `crate::engine::SamplerError` path.
    pub use super::sampler::SamplerError;

    /// The remaining engine-root re-exports the mirrored `src/engine/{loader,step,chart,kern,
    /// trace,mixer}.rs` closure reads at `crate::engine::*` (the real `src/engine/mod.rs:219-234`
    /// re-export block, which this mirror reproduces because the mirrored submodules reach it by
    /// absolute `crate::engine::` paths rather than `super::`).
    pub use super::{
        argmax_fixed, f16_to_fixed, f32_to_fixed, ffn_fixed, matvec_fixed, row_dot_fixed,
        rms_norm_fixed, silu_fixed, top_p_fixed, FfnError, NormError, ProjWeights,
    };
    pub use super::loader::{AffineGeom, AffineVectors, Engine, MixerKind, TensorEntry};
    pub use super::step::{DecodeState, KernelMode, MtpState, SpecStats, StepError, StepResult};

    /// Fractional bits of the executing fixed-point representation (Q32.32).
    pub const FIXED_FRAC_BITS: u32 = 32;
    /// The Q32.32 unit: `ONE` denotes `1.0`.
    pub const ONE: i64 = 1 << FIXED_FRAC_BITS;
}

/// Crate-root aliases of the engine-root constants, so the top-level [`norm`]/[`rope`] mirrors read
/// the same `ONE`/`FIXED_FRAC_BITS` their real `crate::engine` parent defines.
pub use engine::{FIXED_FRAC_BITS, ONE};

/// Crate-root alias of the engine refusal type, so the top-level [`dequant`]/[`matvec`] mirrors read
/// the same `EngineError` their real `crate::engine` parent defines.
pub use error::EngineError;

/// Crate-root alias of the projection refusal, so the top-level [`lm_head`] mirror reads the same
/// `MatvecError` its real `crate::engine` parent re-exports.
pub use matvec::MatvecError;

#[path = "../../src/format/packing.rs"]
pub mod packing;
#[path = "../../src/format/quant.rs"]
pub mod quant;
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
/// The caller supplies the already-computed [`table::SearchDecision`]; separate selected roots
/// extract `Table::search_decision` and the shared `spill_locate` scan, but their bridge contract
/// remains a premise. This calls the real `Table::apply_get`, which indexes the selected
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

/// The real fixed-point `exp` body, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/numeric/fixed_exp.rs` body; it does not restate the `ln 2` reduction,
/// the degree-7 Horner polynomial or the `2^k` scaling. The wrapper only fixes the argument and
/// result types.
pub fn word_fixed_exp(x: i64) -> Result<i64, fixed_exp::ExpError> {
    fixed_exp::fixed_exp(x)
}

/// The real power-of-two activation block scale that `quant_act` uses.
///
/// Calls `src/engine/kern/pow2.rs` `act_scale`; the wrapper does not restate the doubling loop.
pub fn word_act_scale(absmax: u64) -> Option<u64> {
    pow2::act_scale(absmax)
}

/// The real `f16` weight-scale split `d = m·2^k` that the power-of-two GEMM uses.
///
/// Calls `src/engine/kern/pow2.rs` `f16_split`; the wrapper only fixes the types.
pub fn word_f16_split(h: u16) -> Option<(i32, u32)> {
    pow2::f16_split(h)
}

/// The real final rounding `round_shift_32(acc·2^s)` of the power-of-two GEMM.
///
/// Calls `src/engine/kern/pow2.rs` `finish`; the wrapper only fixes the types.
pub fn word_pow2_finish(acc: i64, s: u32) -> Option<i64> {
    pow2::finish(acc, s)
}

/// The real fixed-point `SiLU` body, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/engine/silu.rs` body, which `crate::engine::silu_fixed` also delegates
/// to; it does not restate the two saturation branches, the `fixed_exp` call or the
/// round-to-nearest division. The wrapper only fixes the argument and result types.
pub fn word_silu_fixed(v: i64) -> Result<i64, engine::silu::SiluError> {
    engine::silu::silu_fixed(v)
}

/// The real fixed-point `RMSNorm` body, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/engine/norm.rs` body, which `crate::engine::rms_norm_fixed` also
/// re-exports; it does not restate the `i128` sum of squares, the round-to-nearest mean, the
/// `eps_fixed` addend, the `fixed_rsqrt` call or the value-path epilogue. The wrapper only fixes
/// the row and epsilon types.
pub fn word_rms_norm_fixed(
    row: &[i64],
    eps_fixed: i64,
) -> Result<Vec<i64>, engine::norm::NormError> {
    engine::norm::rms_norm_fixed(row, eps_fixed)
}

/// The real per-token `RoPE` rotation body, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/engine/rope.rs` body, which `crate::engine::rope_rotate_fixed` also
/// re-exports; it does not restate the checked angle product, the `fixed_sin_cos` pair or the
/// round-to-nearest epilogue. The wrapper only fixes the vector, position and table types.
pub fn word_rope_rotate_fixed(
    q: &mut [i64],
    k: &mut [i64],
    pos: u32,
    table: &engine::rope::RopeTable,
) -> Result<(), engine::rope::RopeError> {
    engine::rope::rope_rotate_fixed(q, k, pos, table)
}

/// This calls the real `src/numeric/fixed_rsqrt.rs` body; it does not restate the scaled integer
/// square root, the division or the rounding. The wrapper only fixes the argument and result types.
pub fn word_fixed_rsqrt(x: i64) -> Result<i64, fixed_rsqrt::RsqrtError> {
    fixed_rsqrt::fixed_rsqrt(x)
}

/// This calls the real `src/numeric/fixed_trig.rs` body; it does not restate the `2π` reduction,
/// the quadrant fold or the Taylor polynomials. The wrapper only fixes the argument and result types.
pub fn word_fixed_sin_cos(x: i64) -> Result<fixed_trig::SinCos, fixed_trig::TrigError> {
    fixed_trig::fixed_sin_cos(x)
}

/// The real GGUF header/metadata/tensor-directory parse over a caller-supplied byte slice.
///
/// This calls the real `format::gguf::parse_header` body; the wrapper fixes only the file-length
/// argument to the slice length. It does not restate the magic, version or directory arithmetic.
pub fn word_gguf_parse_header(buf: &[u8]) -> Result<gguf::GgufHeader, gguf::GgufError> {
    gguf::parse_header(buf, buf.len() as u64)
}

/// The real Q4_K integer 4-bit code unpack over one 144-byte super-block.
///
/// This calls the real `format::quant::q4_k_codes` body; the wrapper fixes only the array type. It
/// does not restate the nibble or scale-pair layout.
pub fn word_q4_k_codes(blk: &[u8]) -> Result<[u8; quant::QK_K], quant::QuantError> {
    quant::q4_k_codes(blk)
}

/// The real Q4_K block dequant into a fixed 256-weight observation array.
///
/// This calls the real `format::quant::dequant_q4_k_block` body; the wrapper fixes only the output
/// type. It does not restate the `d·sc·code − dmin·min` scaling.
pub fn word_q4_k_dequant(
    blk: &[u8],
    out: &mut [f32; quant::QK_K],
) -> Result<(), quant::QuantError> {
    quant::dequant_q4_k_block(blk, out)
}

/// The real Q6_K integer 6-bit code unpack over one 210-byte super-block.
///
/// This calls the real `format::quant::q6_k_codes` body; the wrapper fixes only the array type.
pub fn word_q6_k_codes(blk: &[u8]) -> Result<[i8; quant::QK_K], quant::QuantError> {
    quant::q6_k_codes(blk)
}

/// The real Q6_K block dequant into a fixed 256-weight observation array.
///
/// This calls the real `format::quant::dequant_q6_k_block` body; the wrapper fixes only the output
/// type.
pub fn word_q6_k_dequant(
    blk: &[u8],
    out: &mut [f32; quant::QK_K],
) -> Result<(), quant::QuantError> {
    quant::dequant_q6_k_block(blk, out)
}

/// The real canonical packing block decode over a caller-supplied byte slice.
///
/// This calls the real `format::packing::Packing::decode_block` body; the wrapper fixes only the
/// `Packing` type. It does not restate the scale decode, the code unpack or the affine epilogue.
pub fn word_packing_decode_block(
    p: packing::Packing,
    raw: &[u8],
) -> Result<Vec<i64>, packing::PackingError> {
    p.decode_block(raw)
}

/// The real MoE top-k selection, rooted at the concrete Q32.32 `i64` router logits.
///
/// This calls the real `src/engine/moe.rs` body (k passes of a max scan, lower index on ties); the
/// wrapper only fixes the slice and result types.
pub fn word_route_topk(logits: &[i64], k: usize) -> Result<Vec<u32>, engine::moe::MoeError> {
    engine::moe::route_topk(logits, k)
}

/// The real MoE gate weights over the selected experts, rooted at the concrete Q32.32 `i64` words.
///
/// This calls the real `src/engine/moe.rs` body (max-shifted fixed `exp`, checked total, rounded
/// division); the wrapper only fixes the slice and result types.
pub fn word_gate_weights(
    logits: &[i64],
    sel: &[u32],
    renorm: bool,
) -> Result<Vec<i64>, engine::moe::MoeError> {
    engine::moe::gate_weights(logits, sel, renorm)
}

/// The real exact-integer Walsh–Hadamard rotation, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/engine/hadamard.rs` body (sign multiply, `k` checked butterfly stages,
/// one normalization by the `C[k]` table); the wrapper only fixes the slice and result types.
pub fn word_fwht_fixed(
    x: &mut [i64],
    signs: Option<&[i8]>,
    block: usize,
) -> Result<(), engine::hadamard::HadamardError> {
    engine::hadamard::fwht_fixed(x, signs, block)
}

/// The real register stage over `R[x]/(x^2 - sigma)`, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/numeric/register.rs` `stage_mul` body (the span power-of-two check, the
/// length check, the block loop and the per-lane `Cell::checked_mul`); the wrapper only fixes the
/// slice and result types. It does not restate the pair map.
pub fn word_stage_mul(
    x: &mut [i64],
    span: usize,
    w: cell::Cell<i64>,
    sigma: cell::Sigma,
) -> Result<(), register::RegisterError> {
    register::stage_mul(x, span, w, sigma)
}

/// The real fixed-point projection kernel, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/engine/matvec.rs` body, which dequantizes one weight row at a time with
/// the real `dequant_row_fixed` and reduces it with the real `row_dot_fixed`; the wrapper only
/// fixes the payload, shape and result types. It does not restate the row stride, the dequantizer
/// dispatch or the `i128` dot and its `>> 32` epilogue.
pub fn word_matvec_fixed(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    x: &[i64],
) -> Result<Vec<i64>, engine::matvec::MatvecError> {
    engine::matvec::matvec_fixed(ggml_type, raw, n_out, n_in, x)
}

/// The real row-dot helper of the projection kernel, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/engine/matvec.rs` body; it does not restate the checked `i128`
/// multiply-add accumulation or the single round-to-nearest `>> 32` epilogue. The wrapper only
/// fixes the weight-row and activation types.
pub fn word_row_dot_fixed(w_row: &[i64], x: &[i64]) -> Result<i64, engine::matvec::MatvecError> {
    engine::matvec::row_dot_fixed(w_row, x)
}

/// The real LM-head application, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/engine/lm_head.rs` `lm_head_fixed` body, which
/// `crate::engine::Engine::lm_head_fixed` also delegates to after its checked tensor read; it does
/// not restate the activation-width guard, the quant-family check or the `vocab`-row projection
/// with its single `>> 32` epilogue. It composes the rooted [`engine::matvec::matvec_fixed`], so
/// the projection externals already resolve. The wrapper only fixes the payload and activation
/// types.
pub fn word_lm_head_fixed(
    ggml_type: u32,
    raw: &[u8],
    vocab: usize,
    n_embd: usize,
    x: &[i64],
) -> Result<Vec<i64>, engine::lm_head::LmHeadError> {
    engine::lm_head::lm_head_fixed(ggml_type, raw, vocab, n_embd, x)
}

/// The real `KvState::with_carrier` constructor at the concrete Q32.32 carrier, rooted at the
/// concrete Q32.32 `i64` word.
///
/// This calls the real `src/engine/attn.rs` body, which computes the load-time
/// `fixed_rsqrt(head_dim)` score scale and allocates one zeroed plane per layer; it does not restate
/// the geometry guard, the scale call or the plane sizing. The wrapper only fixes the geometry
/// types and the carrier.
///
/// **Coverage consequence.** This wrapper replaced a call to `KvState::new`, which is
/// `Self::with_carrier(.., carrier_from_env())` — an environment read behind a `OnceLock` that the
/// pinned Aeneas does not model. `KvState::new` and `mixer::carrier_from_env` therefore have zero
/// extracted coverage: no axiom and no body in the generated corpus covers either, and neither is
/// checked by `proof/corpus/`. `KvState::new` is not on the production path (production constructs
/// through `KvState::with_carrier(.., carrier)`), so this states a non-claim rather than an
/// omission, but it is a non-claim: the environment-driven carrier selection is unverified.
pub fn word_kv_state_new(
    n_layer: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    cap: usize,
) -> Result<attn::KvState, attn::AttnError> {
    attn::KvState::with_carrier(
        n_layer,
        n_head,
        n_head_kv,
        head_dim,
        cap,
        mixer::Carrier::Q32,
    )
}

/// The real `KvState::append`, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/engine/attn.rs` body; it does not restate the width, capacity or
/// position guard, or the position-major row copy. The wrapper only fixes the row types.
pub fn word_kv_append(
    state: &mut attn::KvState,
    layer: usize,
    k_rows: &[i64],
    v_rows: &[i64],
) -> Result<usize, attn::AttnError> {
    state.append(layer, k_rows, v_rows)
}

/// The real per-query-head causal attention read, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/engine/attn.rs` `KvState::attn_head` body, whose score shift, softmax
/// normalisation and value accumulation are the production path; it does not restate the
/// `row_dot_fixed` score, the `fixed_exp` softmax, the round-to-nearest weights or the epilogue.
pub fn word_attn_head(
    state: &attn::KvState,
    layer: usize,
    q_head: usize,
    q: &[i64],
    pos: usize,
    out: &mut [i64],
) -> Result<(), attn::AttnError> {
    state.attn_head(layer, q_head, q, pos, out)
}

/// The real greedy `argmax_fixed` sampler, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/engine/sampler.rs` body; it does not restate the strict-`>` tie rule or
/// the index-range refusal. The wrapper only fixes the logit type.
pub fn word_argmax_fixed(logits: &[i64]) -> Result<u32, sampler::SamplerError> {
    sampler::argmax_fixed(logits)
}

/// The real integer nucleus filter `top_p_fixed`, rooted at the concrete Q32.32 `i64` word.
///
/// This calls the real `src/engine/sampler.rs` body; it does not restate the maximum shift, the
/// `fixed_exp` window, the checked `i128` total, the round-to-nearest probabilities, the descending
/// order or the cumulative prefix rule. The wrapper only fixes the logit and threshold types.
pub fn word_top_p_fixed(logits: &[i64], p: i64) -> Result<Vec<u32>, sampler::SamplerError> {
    sampler::top_p_fixed(logits, p)
}

/// Mirror of `crate::spirv` for the extraction unit: the real `src/spirv/mod.rs` declares
/// `pub mod check;`, so the `#[path]`-declared `spirv` module resolves `check` against the real
/// `src/spirv/` directory and pulls the real `src/spirv/check.rs`. The checker body itself imports
/// only the standard library, so no other mirror is needed.
#[path = "../../src/spirv/mod.rs"]
pub mod spirv;

/// The real SPIR-V DP4A fragment checker, rooted at the concrete `&[u32]` word slice.
///
/// This calls the real `src/spirv/check.rs` `check_fragment` body; it does not restate the header,
/// D1-D4, storage, decoration or `OpSUDot` obligations. The wrapper only fixes the slice element
/// type (the emitted words are `u32`), so Charon translates the production checker rather than a
/// copy.
pub fn word_check_fragment(words: &[u32]) -> Result<(), spirv::check::CheckError> {
    spirv::check_fragment(words)
}

/// Mirror of `crate::policy` for the extraction unit: the real `src/policy/mod.rs` declares
/// `pub mod decide;`, so the `#[path]`-declared `policy` module resolves `decide` against the real
/// `src/policy/` directory and pulls the real `src/policy/decide.rs`. The decision core imports only
/// the standard library, so no other mirror is needed.
#[path = "../../src/policy/mod.rs"]
pub mod policy;

/// The real bounded authorization decision core `policy::decide`, rooted at the concrete `&[Rule]`,
/// `&[Atom]` and `&[u64]` word slices.
///
/// This calls the real `src/policy/decide.rs` `decide` body; it does not restate the passes over
/// `rules`, the Kleene guard evaluation, the mask arithmetic or the `Option` min-id fold. The
/// wrapper fixes nothing: `decide` is already monomorphic over the concrete `Rule`, `Atom`, `u64`,
/// `u32` and `u64` argument types, and the call is the whole body.
pub fn word_policy_decide(
    rules: &[policy::Rule],
    atoms: &[policy::Atom],
    facts: &[u64],
    known: u64,
    action: u32,
    want: u64,
) -> policy::Verdict {
    policy::decide(rules, atoms, facts, known, action, want)
}

/// Row-chunked projection helpers the real `src/engine/ffn.rs` imports (`super::par`) and their
/// fused block-wise dot (`super::fused`). Neither is an extraction root; they are declared last so
/// no extracted item's source span moves.
#[path = "../../src/engine/fused.rs"]
pub mod fused;
#[path = "../../src/engine/par.rs"]
pub mod par;

/// Mirrors of the `src/engine/mixer.rs` closure: mixer reads `super::{chart, step, value}`,
/// `super::{rms_norm_fixed, silu_fixed, AffineGeom, AffineVectors}` and `crate::numeric::fixed_ln`,
/// and its children `conv_f32`/`f32_carrier` read `crate::engine::step`. Declared last so no earlier
/// extracted item's source span moves; the crate-root aliases below carry the names mixer's
/// `super::` imports need, since the crate-root [`mixer`] mirror's parent is the crate root.
#[path = "../../src/engine/loader.rs"]
pub mod loader;
#[path = "../../src/engine/step.rs"]
pub mod step;
#[path = "../../src/engine/chart/mod.rs"]
pub mod chart;
#[path = "../../src/engine/kern/mod.rs"]
pub mod kern;
#[path = "../../src/engine/trace.rs"]
pub mod trace;
#[path = "../../src/numeric/fixed_ln.rs"]
pub mod fixed_ln;

/// Engine-root re-exports the crate-root mirror modules read through `super::` (mirrors of
/// `src/engine/mod.rs:219-234`). The crate-root [`mixer`]/[`step`]/[`loader`] mirrors' parent is the
/// crate root, so these names must exist here rather than only under the engine mirror.
pub use attn::{AttnError, KvState};
pub use dequant::f16_to_fixed;
pub use f32_convert::{f32_to_fixed, F32Error};
pub use ffn::{ffn_fixed, silu_fixed, FfnError, ProjWeights};
pub use hadamard::{fwht_fixed, HadamardError};
pub use lm_head::LmHeadError;
pub use loader::{AffineGeom, AffineVectors, Engine, MixerKind, TensorEntry};
pub use matvec::{matvec_fixed, row_dot_fixed};
pub use mixer::AffineState;
pub use moe::{gate_weights, moe_ffn_fixed, route_topk, ExpertWeights, MoeError};
pub use norm::{rms_norm_fixed, NormError};
pub use rope::{rope_rotate_fixed, RopeError, RopeTable};
pub use sampler::{argmax_fixed, top_p_fixed, SamplerError};
pub use step::{DecodeState, KernelMode, MtpState, SpecStats, StepError, StepResult};
pub use value::fixed_vector_checksum;

/// FIPS-202 hash construction, declared last so no earlier extracted item's source span moves.
/// `src/crypto.rs` imports only `core`, so no mirror above is needed.
#[path = "../../src/crypto.rs"]
pub mod crypto;

/// FIPS 180-4 SHA-256. `src/sha256.rs` imports only `core`, so no mirror above is needed.
#[path = "../../src/sha256.rs"]
pub mod sha256;

/// `crypto::keccak_f1600`, the Keccak-f[1600] permutation, at the fixed 25-lane state.
///
/// This calls the real `src/crypto.rs` body; the wrapper fixes nothing beyond naming the state
/// type, which is already concrete. It does not restate the theta/rho/pi/chi/iota steps.
pub fn word_keccakf(state: &mut [u64; 25]) {
    crypto::keccak_f1600(state)
}

/// `crypto::sha3_256`, rooted at the concrete `&[u8]` message and `[u8; 32]` digest. The sponge
/// bodies are translatable after the `absorb_block`/`sponge` `while` rewrites; the extraction audit
/// verifies the remaining cone.
pub fn word_sha3_256(msg: &[u8]) -> [u8; 32] {
    crypto::sha3_256(msg)
}

/// `crypto::sha3_512`, rooted at the concrete `&[u8]` message and `[u8; 64]` digest.
pub fn word_sha3_512(msg: &[u8]) -> [u8; 64] {
    crypto::sha3_512(msg)
}

/// `sha256::sha256`, the real FIPS 180-4 SHA-256 body at the concrete `&[u8]`/`[u8; 32]` types.
///
/// This calls the real `src/sha256.rs` body; the wrapper fixes the slice and array types and does
/// not restate the padding, the message schedule or the compression rounds.
pub fn word_sha256(msg: &[u8]) -> [u8; 32] {
    sha256::sha256(msg)
}

/// `sha256::sha256d`, the double SHA-256 composition, at the concrete `&[u8]`/`[u8; 32]` types.
///
/// This calls the real `src/sha256.rs` `sha256d` body, which applies the real [`sha256::sha256`]
/// twice; the wrapper fixes the slice and array types and does not restate the composition.
pub fn word_sha256d(msg: &[u8]) -> [u8; 32] {
    sha256::sha256d(msg)
}

/// RFC 8439. `src/chacha.rs` imports only `core`, so no mirror above is needed. Declared last so
/// no earlier extracted item's source span moves.
#[path = "../../src/chacha.rs"]
pub mod chacha;

/// `chacha::chacha20poly1305_encrypt`, the RFC 8439 section 2.8.1 AEAD seal, at the concrete
/// `&[u8; 32]` key, `&[u8; 12]` nonce and `&[u8]` aad/plaintext types.
///
/// This calls the real `src/chacha.rs` body; the wrapper fixes the slice and array types and does
/// not restate the one-time-key derivation, the keystream xor, the MAC-input padding or the tag
/// append. The body is total (infallible, returning `Vec<u8>`), so this root has no `Result`.
pub fn word_chacha20poly1305_encrypt(
    key: &[u8; chacha::KEY_LEN],
    nonce: &[u8; chacha::NONCE_LEN],
    aad: &[u8],
    plaintext: &[u8],
) -> Vec<u8> {
    chacha::chacha20poly1305_encrypt(key, nonce, aad, plaintext)
}

/// `chacha::chacha20poly1305_decrypt`, the RFC 8439 section 2.8.2 AEAD open, at the same concrete
/// key/nonce types.
///
/// This calls the real `src/chacha.rs` body; the wrapper fixes the types and does not restate the
/// short-ciphertext refusal, the constant-time tag comparison or the keystream xor. It is the only
/// partial entry point: it yields `Ok(Vec<u8>)` or `Err(chacha::Error)`.
pub fn word_chacha20poly1305_decrypt(
    key: &[u8; chacha::KEY_LEN],
    nonce: &[u8; chacha::NONCE_LEN],
    aad: &[u8],
    ciphertext_and_tag: &[u8],
) -> Result<Vec<u8>, chacha::Error> {
    chacha::chacha20poly1305_decrypt(key, nonce, aad, ciphertext_and_tag)
}

/// FIPS 197. `src/aes.rs` imports only `core`, so no mirror above is needed. Declared after
/// `chacha` so no earlier extracted item's source span moves.
#[path = "../../src/aes.rs"]
pub mod aes;

/// `aes::aes128_encrypt_block`, the FIPS 197 section 6.1 cipher, at the concrete `&[u8; 16]` key
/// and block types.
///
/// This calls the real `src/aes.rs` body; the wrapper fixes nothing beyond naming the array types,
/// which are already concrete. It does not restate the key expansion, SubBytes, ShiftRows,
/// MixColumns or AddRoundKey steps.
pub fn word_aes128_encrypt_block(
    key: &[u8; aes::KEY_LEN],
    block: &[u8; aes::BLOCK_LEN],
) -> [u8; aes::BLOCK_LEN] {
    aes::aes128_encrypt_block(key, block)
}

/// `aes::aes128_decrypt_block`, the FIPS 197 section 6.2 equivalent inverse cipher, at the same
/// concrete key and block types.
///
/// This calls the real `src/aes.rs` body and does not restate InvShiftRows, InvSubBytes,
/// InvMixColumns or the round-key order. The body is total (infallible, returning `[u8; 16]`), so
/// this root has no `Result`.
pub fn word_aes128_decrypt_block(
    key: &[u8; aes::KEY_LEN],
    block: &[u8; aes::BLOCK_LEN],
) -> [u8; aes::BLOCK_LEN] {
    aes::aes128_decrypt_block(key, block)
}

/// `crypto::shake128`, rooted at the concrete `&[u8]` message and `&mut [u8]` out-parameter.
///
/// Not an extraction root: its output length is unbounded at the call site, so the root would be
/// slice-length-parametric rather than monomorphic.
pub fn word_shake128(msg: &[u8], out: &mut [u8]) {
    crypto::shake128(msg, out)
}

/// `crypto::shake256`, rooted as `word_shake128` is, and likewise not an extraction root.
pub fn word_shake256(msg: &[u8], out: &mut [u8]) {
    crypto::shake256(msg, out)
}

// ── The selected map face: the ported incumbent, reached by `#[path]` ───────────────────────────
//
// `src/hashtrinity/` is the executed body of the registered selected map arm
// (`examples/reg1/adapters/selected_map.rs`), so it is reached here by `#[path]` — the extracted
// object and the executed object are the same source. The wrappers below fix only the key type
// (`u64`, which `hashtrinity/algebra.rs` implements `Coord` for) and the table's default key store;
// they do not restate the probe, the descent, the resize or the merge.
//
// Route 1 (`KeyReadout`): `hashtrinity::KeyStore` is GAT-free, so these storage/probe/insert bodies
// carry no generic associated type and Aeneas's `translate_fwd_ty` does not refuse them. The
// readback faces (`iter`, `encircle_floor`, the `face`/`ctrl_proto` iterators) need the
// `KeyReadout` extension trait and are not on this path.
//
// The mutator root is merge-free: it calls `insert_merge_last_write`, the Map face's
// `insert_with(k, v, set)` specialised to its own body. `insert_with` is the same core plus a
// caller-supplied `impl FnOnce(V, V) -> V` merge, and a `fn` item passed into it is an arrow type —
// `Arrow types are not supported yet` at the wrapper, which is the first-order/higher-order
// boundary, not a missing rule. The concrete face carries the same semantics and translates.
//
// The module is declared last, as `crypto`/`sha256`/`chacha`/`aes` are, so no earlier extracted
// item's source span moves.
#[path = "../../src/hashtrinity/mod.rs"]
pub mod hashtrinity;

/// Real `hashtrinity::Table::<u64, i64>::new` — the stored-key constructor (`with_bits(3)`).
pub fn word_hashtrinity_new() -> hashtrinity::Table<u64, i64> {
    hashtrinity::Table::new()
}

/// Real `hashtrinity::Table::with_capacity`, the constructor `build_selected_map` uses.
pub fn word_hashtrinity_with_capacity(cap: usize) -> hashtrinity::Table<u64, i64> {
    hashtrinity::Table::with_capacity(cap)
}

/// Real `hashtrinity::Table::get`, the coordinate read, returning the stored value by read.
pub fn word_hashtrinity_get(state: &hashtrinity::Table<u64, i64>, key: u64) -> Option<i64> {
    match state.get(&key) {
        Some(v) => Some(*v),
        None => None,
    }
}

/// Real `hashtrinity::Table::insert_merge_last_write` — the Map face's mutator (`put`/`last write`)
/// over the real `insert_core`, with the merge in its own body instead of an arrow-typed argument.
pub fn word_hashtrinity_insert(
    mut state: hashtrinity::Table<u64, i64>,
    key: u64,
    value: i64,
) -> hashtrinity::Table<u64, i64> {
    state.insert_merge_last_write(key, value);
    state
}

/// Real `CoordMap::<u64, i64>::coordinate_with_capacity` — the `Regen` constructor.
pub fn word_coordmap_coordinate_with_capacity(cap: usize) -> hashtrinity::CoordMap<u64, i64> {
    hashtrinity::CoordMap::coordinate_with_capacity(cap)
}

/// Real `CoordMap::get` (`Regen` carrier).
pub fn word_coordmap_get(state: &hashtrinity::CoordMap<u64, i64>, key: u64) -> Option<i64> {
    match state.get(&key) {
        Some(v) => Some(*v),
        None => None,
    }
}

/// Real `CoordMap::insert_merge_last_write` (`Regen` carrier).
pub fn word_coordmap_insert(
    mut state: hashtrinity::CoordMap<u64, i64>,
    key: u64,
    value: i64,
) -> hashtrinity::CoordMap<u64, i64> {
    state.insert_merge_last_write(key, value);
    state
}

// The count face's sequential helpers (`hashtrinity::FreqMap::count_*`), `K = u64`. Each wrapper
// fixes the key type and calls the real body; the thread fan-out of `count_par` stays outside.

/// Real `FreqMap::<u64, i64>::count_home`.
pub fn word_count_home(c: u64) -> usize {
    hashtrinity::FreqMap::<u64, i64>::count_home(c)
}

/// Real `FreqMap::<u64, i64>::count_admit`.
pub fn word_count_admit(n: usize, count_max: u64) -> bool {
    hashtrinity::FreqMap::<u64, i64>::count_admit(n, count_max)
}

/// Real `FreqMap::<u64, i64>::count_histogram`.
pub fn word_count_histogram(keys: &[u64], homes: &mut [u8], hist: &mut [u32]) -> bool {
    hashtrinity::FreqMap::<u64, i64>::count_histogram(keys, homes, hist)
}

/// Real `FreqMap::<u64, i64>::count_offsets`.
pub fn word_count_offsets(
    counts: &[u32],
    w: usize,
    n: usize,
    bstart: &mut [usize],
    woff: &mut [usize],
) -> bool {
    hashtrinity::FreqMap::<u64, i64>::count_offsets(counts, w, n, bstart, woff)
}

/// Real `FreqMap::<u64, i64>::count_slot`.
pub fn word_count_slot(off: &mut [usize], h: u8) -> usize {
    hashtrinity::FreqMap::<u64, i64>::count_slot(off, h)
}

/// Real `FreqMap::<u64, i64>::count_aggregate`.
pub fn word_count_aggregate(routed: &[u64], lo: usize, hi: usize) -> hashtrinity::FreqMap<u64, i64> {
    hashtrinity::FreqMap::<u64, i64>::count_aggregate(routed, lo, hi)
}

// The exact-integer Gram fold (`src/learn/gram.rs`), reached by `#[path]` so the extracted body is
// the real source. The module is dependency-free; the wrapper fixes nothing and calls the real
// function.
#[path = "../../src/learn/gram.rs"]
pub mod gram;

/// Real `learn::gram::gram_fold_i8`: the exact checked `i64` fold `g += XᵀX` over a batch of `i8`
/// rows. The wrapper calls the real body; it does not restate the product or the accumulation.
pub fn word_gram_fold_i8(
    g: Vec<i64>,
    x: &[i8],
    rows: usize,
    d: usize,
) -> Result<Vec<i64>, gram::GramError> {
    gram::gram_fold_i8(g, x, rows, d)
}

// The exact-integer sigma-delta block (`src/learn/sigma_delta.rs`), reached by `#[path]` so the
// extracted body is the real source. The module is dependency-free; the wrapper fixes nothing and
// calls the real function.
#[path = "../../src/learn/sigma_delta.rs"]
pub mod sigma_delta;

/// Real `learn::sigma_delta::sd_binary_block`: the checked `i64` first-order sigma-delta block. It
/// returns the final error state and the peak `max |e_t|`, or `None` on a shape mismatch or a
/// checked overflow. The wrapper calls the real body; it does not restate the sign test or the
/// recurrence.
pub fn word_sd_binary_block(w: &[i64], a: i64, s: &mut [i8]) -> Option<(i64, i64)> {
    sigma_delta::sd_binary_block(w, a, s)
}

// The one-bit arm bit budget (`src/learn/onebit_bits.rs`), reached by `#[path]` so the extracted
// bodies are the real source. The module depends only on `core`; each wrapper fixes nothing and
// calls the real function.
#[path = "../../src/learn/onebit_bits.rs"]
pub mod onebit_bits;

/// Real `learn::onebit_bits::total_bits`: `nsign + nr + nc + nd + ncorr`, `None` on a checked
/// overflow or a width that does not divide `group`.
pub fn word_onebit_total_bits(c: &onebit_bits::Cfg) -> Option<u64> {
    onebit_bits::total_bits(c)
}

/// Real `learn::onebit_bits::bits_sign`.
pub fn word_onebit_bits_sign(c: &onebit_bits::Cfg) -> Option<u64> {
    onebit_bits::bits_sign(c)
}

/// Real `learn::onebit_bits::bits_r`.
pub fn word_onebit_bits_r(c: &onebit_bits::Cfg) -> Option<u64> {
    onebit_bits::bits_r(c)
}

/// Real `learn::onebit_bits::bits_c`.
pub fn word_onebit_bits_c(c: &onebit_bits::Cfg) -> Option<u64> {
    onebit_bits::bits_c(c)
}

/// Real `learn::onebit_bits::bits_d`.
pub fn word_onebit_bits_d(c: &onebit_bits::Cfg) -> Option<u64> {
    onebit_bits::bits_d(c)
}

/// Real `learn::onebit_bits::bits_corr`.
pub fn word_onebit_bits_corr(c: &onebit_bits::Cfg) -> Option<u64> {
    onebit_bits::bits_corr(c)
}

/// Real `learn::onebit_bits::params`.
pub fn word_onebit_params(c: &onebit_bits::Cfg) -> Option<u64> {
    onebit_bits::params(c)
}

/// Real `learn::onebit_bits::index_bits`.
pub fn word_onebit_index_bits(c: &onebit_bits::Cfg) -> Option<u64> {
    onebit_bits::index_bits(c)
}

// The one-bit backward passes (`src/learn/grad/{dense,attn,head}.rs`), reached through `learn_port.rs`.
#[path = "learn_port.rs"]
pub mod learn_port;
