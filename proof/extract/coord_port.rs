//! Extraction roots for the coordinate surface: `src/map/coord.rs` and
//! `src/map/coord/{address,locator,locator_core}.rs`.
//!
//! The real `coord` module is declared in `lib.rs` by `#[path]`, and its `address` and `locator`
//! children resolve from `src/map/coord/`; `locator_core` is `pub(crate)` inside `locator`, which
//! the extraction unit sees crate-wide. Every wrapper below fixes a concrete root and calls the
//! real body, so no formula, match arm or refusal order is restated here.
//!
//! # Merge (`proof/extract/lib.rs`)
//!
//! The unit is declared at the crate root:
//!
//! ```text
//! #[path = "coord_port.rs"]
//! pub mod coord_port;
//! ```
//!
//! and `proof/regen.sh` lists every public wrapper below as a `--start-from` root.
//!
//! # Unrooted surface
//!
//! * The `ExactIdentity` trait declaration and its `i64`, `u64`, `Vec<u8>` and `Arc<T>` impls: the
//!   first has no body, and Charon names a primitive's impl at the scalar (`I64.Insts.…`), not
//!   under this module.
//! * `reservation_failure::{arm, arm_late_edge, disarm, consume, consume_late_edge}` are
//!   `#[cfg(feature = "injection")]` and the extraction unit does not enable that feature.
//! * `Directory::reserve_all_capacity` is private to `locator_core`; it is reached through
//!   [`word_directory_reserve_or_linked_residue`].

use crate::cell::Sigma;
use crate::coord::address::{
    face_index_policy, AddressValue, EllipticPacking, FaceCoord, FaceDomain, FaceResidueCapability,
    FaceTableAddress, H3_40, LocatorSigma, ParabolicDual, Residue, WordFaceHome,
};
use crate::coord::locator::locator_core::{
    Directory, Edge, EdgeList, EdgeWindows, FoundOrStaged, IdentityNode, IdentityNodeIter,
    IdentityNodes, Refusal, StagedPath,
};
use crate::coord::locator::{
    coefficient_bits, residue_digits, Absence, CoveredScope, FaceIndex, Inserted, Locator, Lookup,
};

// ── `coord` (src/map/coord.rs) ──────────────────────────────────────────────────────────────

/// The real `Coord::coord_with` default body, through a `u64` receiver.
pub fn word_coord_with(key: u64, pepper: u64) -> u64 {
    <u64 as crate::coord::Coord>::coord_with(&key, pepper)
}

/// The real `coord::inverse_odd`: Newton's iteration on the odd-multiply inverse.
pub fn word_coord_inverse_odd(a: u64) -> u64 {
    crate::coord::inverse_odd(a)
}

/// The real `coord::bucket`: the avalanched address's top `bits`.
pub fn word_coord_bucket(address: u64, bits: u32) -> u64 {
    crate::coord::bucket(address, bits)
}

/// The real `coord::mix`: one multiply-add step of a lossy fold.
pub fn word_coord_mix(acc: u64, next: u64) -> u64 {
    crate::coord::mix(acc, next)
}

/// The real `coord::pack`: the concatenating, exactly invertible address.
pub fn word_coord_pack(hi: u64, lo: u64, shift: u32) -> u64 {
    crate::coord::pack(hi, lo, shift)
}

/// The real `i21::new`, the checked 21-bit constructor.
pub fn word_coord_i21_new(v: i32) -> Option<crate::coord::i21> {
    crate::coord::i21::new(v)
}

/// The real `i21::get`.
pub fn word_coord_i21_get(v: crate::coord::i21) -> i32 {
    v.get()
}

/// The real `Coord for i21` body.
pub fn word_coord_i21_coord(v: crate::coord::i21) -> u64 {
    <crate::coord::i21 as crate::coord::Coord>::coord(&v)
}

/// The real `Packed for (i21, i21, i21)` body, which reaches the private `i21::bits`.
pub fn word_coord_i21_tuple_packed(
    a: crate::coord::i21,
    b: crate::coord::i21,
    c: crate::coord::i21,
) -> u64 {
    <(crate::coord::i21, crate::coord::i21, crate::coord::i21) as crate::coord::Packed>::packed(&(
        a, b, c,
    ))
}

/// The real `Packed for (i21, i21, i21)` inverse, which reaches the private `i21::from_bits`.
pub fn word_coord_i21_tuple_unpack(
    word: u64,
) -> (crate::coord::i21, crate::coord::i21, crate::coord::i21) {
    <(crate::coord::i21, crate::coord::i21, crate::coord::i21) as crate::coord::Packed>::unpack(word)
}

// ── `coord::address` ────────────────────────────────────────────────────────────────────────

/// The real `H3_40::new`: validation of the frozen constants, then the sealed token.
pub fn word_h3_40_new() -> Option<H3_40> {
    H3_40::new()
}

/// The real `H3_40::q`.
pub fn word_h3_40_q(profile: H3_40) -> u64 {
    profile.q()
}

/// The real `H3_40::inverse_of_two`.
pub fn word_h3_40_inverse_of_two(profile: H3_40) -> u64 {
    profile.inverse_of_two()
}

/// The real `H3_40::r`.
pub fn word_h3_40_r(profile: H3_40) -> u64 {
    profile.r()
}

/// The real `H3_40::s`.
pub fn word_h3_40_s(profile: H3_40) -> u64 {
    profile.s()
}

/// The real `H3_40::version`.
pub fn word_h3_40_version(profile: H3_40) -> u32 {
    profile.version()
}

/// The real checked `Residue::new`.
pub fn word_residue_new(profile: H3_40, value: u64) -> Option<Residue> {
    Residue::new(profile, value)
}

/// The real `ParabolicDual::scalar` (`fstHom`).
pub fn word_parabolic_dual_scalar(dual: ParabolicDual) -> u64 {
    dual.scalar()
}

/// The real `ParabolicDual::nilpotent`.
pub fn word_parabolic_dual_nilpotent(dual: ParabolicDual) -> u64 {
    dual.nilpotent()
}

/// The real `AddressValue::face`.
pub fn word_address_value_face(value: AddressValue) -> Sigma {
    value.face()
}

/// The real `AddressValue::elliptic_word`.
pub fn word_address_value_elliptic_word(value: AddressValue) -> Option<u64> {
    value.elliptic_word()
}

/// The real `AddressValue::word`.
pub fn word_address_value_word(value: AddressValue) -> Option<u64> {
    value.word()
}

/// The real `AddressValue::residue`.
pub fn word_address_value_residue(value: AddressValue) -> Option<Residue> {
    value.residue()
}

/// The real `AddressValue::profile`.
pub fn word_address_value_profile(value: AddressValue) -> Option<H3_40> {
    value.profile()
}

/// The real `FaceTableAddress::value`.
pub fn word_face_table_address_value(address: FaceTableAddress) -> AddressValue {
    address.value()
}

/// The real `FaceTableAddress::face`.
pub fn word_face_table_address_face(address: FaceTableAddress) -> Sigma {
    address.face()
}

/// The real `LocatorSigma::dial`.
pub fn word_locator_sigma_dial(sigma: LocatorSigma) -> Sigma {
    sigma.dial()
}

/// The real `face_index_policy`.
pub fn word_face_index_policy(sigma: Sigma) -> crate::coord::address::FaceIndexPolicy {
    face_index_policy(sigma)
}

/// The real `FaceResidueCapability<u64> for WordFaceHome` parabolic body.
pub fn word_word_face_parabolic_dual_u64(home: WordFaceHome, key: u64) -> ParabolicDual {
    home.parabolic_dual(&key)
}

/// The real `FaceResidueCapability<u64> for WordFaceHome` hyperbolic body.
pub fn word_word_face_hyperbolic_scalar_u64(
    home: WordFaceHome,
    profile: H3_40,
    key: u64,
) -> Residue {
    home.hyperbolic_scalar(profile, &key)
}

/// The real `FaceResidueCapability<i64> for WordFaceHome` parabolic body.
pub fn word_word_face_parabolic_dual_i64(home: WordFaceHome, key: i64) -> ParabolicDual {
    home.parabolic_dual(&key)
}

/// The real `FaceResidueCapability<i64> for WordFaceHome` hyperbolic body.
pub fn word_word_face_hyperbolic_scalar_i64(
    home: WordFaceHome,
    profile: H3_40,
    key: i64,
) -> Residue {
    home.hyperbolic_scalar(profile, &key)
}

/// The real `EllipticPacking<u64> for WordFaceHome` pack body.
pub fn word_word_face_elliptic_pack_u64(home: WordFaceHome, key: u64) -> u64 {
    home.elliptic_pack(&key)
}

/// The real `EllipticPacking<u64> for WordFaceHome` unpack body.
pub fn word_word_face_elliptic_unpack_u64(home: WordFaceHome, word: u64) -> Option<u64> {
    home.elliptic_unpack(word)
}

/// The real `EllipticPacking<i64> for WordFaceHome` pack body.
pub fn word_word_face_elliptic_pack_i64(home: WordFaceHome, key: i64) -> u64 {
    home.elliptic_pack(&key)
}

/// The real `EllipticPacking<i64> for WordFaceHome` unpack body.
pub fn word_word_face_elliptic_unpack_i64(home: WordFaceHome, word: u64) -> Option<i64> {
    home.elliptic_unpack(word)
}

/// The real `FaceDomain<u64, WordFaceHome> for Sigma` `sigma` body.
pub fn word_sigma_face_domain_sigma(sigma: Sigma) -> Sigma {
    <Sigma as FaceDomain<u64, WordFaceHome>>::sigma(sigma)
}

/// The real `FaceDomain<u64, WordFaceHome> for Sigma` `from_sigma` body.
pub fn word_sigma_face_domain_from_sigma(sigma: Sigma) -> Option<Sigma> {
    <Sigma as FaceDomain<u64, WordFaceHome>>::from_sigma(sigma)
}

/// The real `FaceDomain<u64, WordFaceHome> for Sigma` `encode` body.
pub fn word_sigma_face_domain_encode(
    sigma: Sigma,
    profile: H3_40,
    key: u64,
) -> FaceTableAddress {
    <Sigma as FaceDomain<u64, WordFaceHome>>::encode(sigma, &WordFaceHome, profile, &key)
}

/// The real `FaceDomain<u64, WordFaceHome> for LocatorSigma` `sigma` body.
pub fn word_locator_sigma_face_domain_sigma(sigma: LocatorSigma) -> Sigma {
    <LocatorSigma as FaceDomain<u64, WordFaceHome>>::sigma(sigma)
}

/// The real `FaceDomain<u64, WordFaceHome> for LocatorSigma` `from_sigma` body.
pub fn word_locator_sigma_face_domain_from_sigma(sigma: Sigma) -> Option<LocatorSigma> {
    <LocatorSigma as FaceDomain<u64, WordFaceHome>>::from_sigma(sigma)
}

/// The real `FaceDomain<u64, WordFaceHome> for LocatorSigma` `encode` body.
pub fn word_locator_sigma_face_domain_encode(
    sigma: LocatorSigma,
    profile: H3_40,
    key: u64,
) -> FaceTableAddress {
    <LocatorSigma as FaceDomain<u64, WordFaceHome>>::encode(sigma, &WordFaceHome, profile, &key)
}

/// The real `FaceCoord<u64> for WordFaceHome` `coord` default body.
pub fn word_word_face_coord_u64(
    home: &WordFaceHome,
    profile: H3_40,
    face: Sigma,
    key: u64,
) -> FaceTableAddress {
    <WordFaceHome as FaceCoord<u64>>::coord(home, profile, face, &key)
}

// ── `coord::locator` ────────────────────────────────────────────────────────────────────────

/// The real `Locator::new` for a word-keyed locator.
pub fn word_locator_new(profile: H3_40) -> Locator<u64, u64> {
    Locator::new(profile)
}

/// The real `Locator::with_max_reps`.
pub fn word_locator_with_max_reps(profile: H3_40, max_reps: usize) -> Locator<u64, u64> {
    Locator::with_max_reps(profile, max_reps)
}

/// The real `Locator::profile`.
pub fn word_locator_profile(locator: &Locator<u64, u64>) -> H3_40 {
    locator.profile()
}

/// The real `Locator::len`.
pub fn word_locator_len(locator: &Locator<u64, u64>) -> usize {
    locator.len()
}

/// The real `Locator::is_empty`.
pub fn word_locator_is_empty(locator: &Locator<u64, u64>) -> bool {
    locator.is_empty()
}

/// The real `Locator::coverage_established`.
pub fn word_locator_coverage_established(locator: &Locator<u64, u64>) -> bool {
    locator.coverage_established()
}

/// The real `Locator::residue_node_count`.
pub fn word_locator_residue_node_count(locator: &Locator<u64, u64>) -> usize {
    locator.residue_node_count()
}

/// The real `Locator::parabolic_node_count`.
pub fn word_locator_parabolic_node_count(locator: &Locator<u64, u64>) -> usize {
    locator.parabolic_node_count()
}

/// The real `Locator::identity_node_count`.
pub fn word_locator_identity_node_count(locator: &Locator<u64, u64>) -> usize {
    locator.identity_node_count()
}

/// The real `Locator::edge_count`.
pub fn word_locator_edge_count(locator: &Locator<u64, u64>) -> usize {
    locator.edge_count()
}

/// The real `Locator::max_identity_fanout`.
pub fn word_locator_max_identity_fanout(locator: &Locator<u64, u64>) -> usize {
    locator.max_identity_fanout()
}

/// The real `Locator::invalidate_coverage`.
pub fn word_locator_invalidate_coverage(locator: &mut Locator<u64, u64>) {
    locator.invalidate_coverage()
}

/// The real `Locator::insert`.
pub fn word_locator_insert(
    locator: &mut Locator<u64, u64>,
    index: FaceIndex,
    key: u64,
    value: u64,
) -> Result<Inserted, Refusal> {
    locator.insert(index, &key, value)
}

/// The real `Locator::get`.
pub fn word_locator_get<'a>(
    locator: &'a Locator<u64, u64>,
    index: FaceIndex,
    key: u64,
) -> Lookup<'a, u64> {
    locator.get(index, &key)
}

/// The real `Locator::lookup_absent`.
pub fn word_locator_lookup_absent(
    locator: &Locator<u64, u64>,
    index: FaceIndex,
    key: u64,
) -> Absence {
    locator.lookup_absent(index, &key)
}

/// The real `FaceIndex::face`.
pub fn word_face_index_face(index: FaceIndex) -> Sigma {
    index.face()
}

/// The real `residue_digits`, reaching `Residue as ResidueIndex::digits`.
pub fn word_residue_digits(residue: Residue) -> [u8; 40] {
    residue_digits(residue)
}

/// The real `coefficient_bits`, reaching `ParabolicDual as ResidueIndex::digits`.
pub fn word_coefficient_bits(coefficient: u64) -> [u8; 64] {
    coefficient_bits(coefficient)
}

/// The real `CoveredScope::index`.
pub fn word_covered_scope_index(scope: CoveredScope) -> FaceIndex {
    scope.index()
}

/// The real `CoveredScope::face`.
pub fn word_covered_scope_face(scope: CoveredScope) -> Sigma {
    scope.face()
}

/// The real `CoveredScope::residue`.
pub fn word_covered_scope_residue(scope: CoveredScope) -> Option<Residue> {
    scope.residue()
}

/// The real `CoveredScope::coefficient`.
pub fn word_covered_scope_coefficient(scope: CoveredScope) -> Option<ParabolicDual> {
    scope.coefficient()
}

/// The real `CoveredScope::residue_levels`.
pub fn word_covered_scope_residue_levels(scope: CoveredScope) -> usize {
    scope.residue_levels()
}

/// The real `CoveredScope::identity_bytes`.
pub fn word_covered_scope_identity_bytes(scope: CoveredScope) -> usize {
    scope.identity_bytes()
}

// ── `coord::locator::locator_core` ──────────────────────────────────────────────────────────

/// The real `Directory::new`.
pub fn word_directory_new() -> Directory {
    Directory::new()
}

/// The real `Directory::arena_len` at the hyperbolic face.
pub fn word_directory_arena_len_residue(directory: &Directory) -> usize {
    directory.arena_len::<Residue>()
}

/// The real `Directory::arena_len` at the parabolic face.
pub fn word_directory_arena_len_parabolic(directory: &Directory) -> usize {
    directory.arena_len::<ParabolicDual>()
}

/// The real `Directory::max_identity_fanout`.
pub fn word_directory_max_identity_fanout(directory: &Directory) -> usize {
    directory.max_identity_fanout()
}

/// The real `Directory::identity_nodes` through `IdentityNodes::get`.
pub fn word_directory_identity_nodes_get(
    directory: &Directory,
    index: usize,
) -> Option<IdentityNode<'_>> {
    directory.identity_nodes().get(index)
}

/// The real `Directory::representative_terminal` at the hyperbolic face.
pub fn word_directory_representative_terminal_residue(
    directory: &Directory,
    index: Residue,
    identity: u64,
) -> Option<u32> {
    directory.representative_terminal::<Residue, u64>(&index, &identity)
}

/// The real `Directory::reserve_or_linked` at the hyperbolic face.
pub fn word_directory_reserve_or_linked_residue(
    directory: &mut Directory,
    index: Residue,
    key: u64,
) -> Result<FoundOrStaged<((), StagedPath)>, Refusal> {
    directory.reserve_or_linked::<Residue, u64>(&index, &key)
}

/// The real `EdgeList::len`.
pub fn word_edge_list_len(edges: EdgeList<'_>) -> usize {
    edges.len()
}

/// The real `EdgeList::windows`.
pub fn word_edge_list_windows<'a>(edges: &'a EdgeList<'a>, width: usize) -> EdgeWindows<'a> {
    edges.windows(width)
}

/// The real `Iterator for EdgeWindows::next`.
pub fn word_edge_windows_next(windows: &mut EdgeWindows<'_>) -> Option<[Edge; 2]> {
    windows.next()
}

/// The real `IdentityNodes::get`.
pub fn word_identity_nodes_get(
    nodes: IdentityNodes<'_>,
    index: usize,
) -> Option<IdentityNode<'_>> {
    nodes.get(index)
}

/// The real `IdentityNodes::iter`.
pub fn word_identity_nodes_iter(nodes: IdentityNodes<'_>) -> IdentityNodeIter<'_> {
    nodes.iter()
}

/// The real `IntoIterator for IdentityNodes::into_iter`.
pub fn word_identity_nodes_into_iter(nodes: IdentityNodes<'_>) -> IdentityNodeIter<'_> {
    nodes.into_iter()
}

/// The real `Iterator for IdentityNodeIter::next`.
pub fn word_identity_node_iter_next<'a>(
    iter: &mut IdentityNodeIter<'a>,
) -> Option<IdentityNode<'a>> {
    iter.next()
}
