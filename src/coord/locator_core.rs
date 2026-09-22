#![allow(
    clippy::cast_possible_truncation,
    clippy::manual_let_else,
    clippy::manual_map,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::len_zero,
    clippy::bool_to_int_with_if,
    reason = "extraction-facing scalar forms: the pinned extractor emits an `axiom` for \
              `is_empty`, `div_ceil`, `Option::map`, `let-else`, integer `TryFrom` and the \
              narrowing casts, so the explicit forms are kept and these lints stand"
)]
//! The shared directory engine behind [`crate::coord::locator`].
//!
//! This module is private to [`crate::coord::locator`]. It holds the face-independent machinery the
//! parabolic and hyperbolic faces share: the residue arenas, the generic residue-path traversal,
//! the exact-identity byte trie, the reserve-before-retain machinery and the one-shot failure
//! seams. The caller supplies the face's [`ResidueIndex`] parameter; everything downstream of the
//! digit sequence is one implementation and one set of invariants.
//!
//! A terminal is a bounded opaque handle (`u32`) into the caller's representative registry. The
//! engine owns no value and retains no second key: the face check and the exact-identity comparison
//! stay with the caller.
//!
//! # Reserve-before-retain ordering
//!
//! [`Directory::reserve_for`] follows `checked sizing -> reserve all -> stage path -> reserve every
//! identity edge -> reserve retained identity`. Every fallible allocation happens before retention.
//! An armed [`reservation_failure`] entry refusal short-circuits the step; an armed
//! [`reservation_failure::arm_late_edge`] refusal fires at the identity-edge reservation. Both arms
//! are one-shot and leave the committed links, counts and coverage unchanged.

use crate::cell::Sigma;
use crate::coord::address::{ParabolicDual, Residue};
use crate::table::keyrep::{ExactCodec, I64WordCodec};
use crate::table::{TableAddress, WordHome};

use super::{LEVELS, MAX_LEVELS, PARABOLIC_LEVELS};

/// The largest number of children one residue node may hold, across the admitted faces.
const MAX_ARITY: usize = 3;

/// The sentinel for an absent child slot or identity-trie root.
///
/// A residue node stores child node indices and an identity-trie root index, never a value. Every
/// stored index is a `u32` node index bounded strictly below `u32::MAX` by the checked sizing in
/// [`Directory::reserve_for`] (`len + growth <= u32::MAX`), so `u32::MAX` is never a live index.
/// Storing this sentinel instead of `None` keeps each traversal a plain `u32` read, so Aeneas does
/// not have to expand an `Option` enumeration in an array position.
const NO_CHILD: u32 = u32::MAX;

/// Why a write did not commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The locator has no elliptic face; elliptic addresses are exact packings.
    WrongFace,
    /// A checked count or handle bound did not fit `usize` or `u32`.
    SizeOverflow,
    /// The configured representative limit would be exceeded.
    Capacity,
    /// The backing reservation could not be obtained.
    AllocationFailed,
    /// The coverage invariant is not established, so the write cannot be placed soundly.
    Uncovered,
}

/// Bound for [`ExactIdentity::Retained`]: a retained identity is `Clone + Eq`.
///
/// Spelled as a helper trait rather than `Clone + Eq` inline because the pinned Aeneas emits one
/// trait-clause field per bound, named after the trait: with `ExactIdentity: Eq` the `Eq` bound on
/// `Self` and on `Retained` both become a field `corecmpEqInst`, and the generated structure does
/// not elaborate. The blanket impl makes this equivalent to `Clone + Eq`.
pub trait RetainedKey: Clone + Eq {}
impl<T: Clone + Eq> RetainedKey for T {}

/// The exact identity of a key: a finite byte encoding `E` with `E(k) = E(k')` iff `k = k'`.
///
/// The locator stores only [`ExactIdentity::Retained`], so a key that is too large to retain
/// can be replaced by an independently lossless codec. The residue is never consulted for
/// identity.
pub trait ExactIdentity: Eq {
    /// The retained exact identity. Equality on this type decides key equality.
    type Retained: RetainedKey;

    /// Storage reserved by [`ExactIdentity::reserve_retain`] for one retained identity.
    ///
    /// `()` for an identity that does not allocate.
    type Reserved;

    /// Retains the exact identity of `self`, allocating infallibly.
    ///
    /// The write path uses [`ExactIdentity::reserve_retain`] and
    /// [`ExactIdentity::retain_reserved`] instead, so allocation stays fallible and gated.
    fn retain(&self) -> Self::Retained;

    /// Reserves the storage [`ExactIdentity::retain_reserved`] needs.
    ///
    /// # Errors
    ///
    /// Returns [`Refusal::AllocationFailed`] when the reservation cannot be obtained.
    fn reserve_retain(&self) -> Result<Self::Reserved, Refusal>;

    /// Retains the exact identity after [`ExactIdentity::reserve_retain`] succeeded. Does not
    /// allocate.
    fn retain_reserved(&self, reserved: Self::Reserved) -> Self::Retained;

    /// Decides equality of `self` with a retained identity, without allocating.
    fn matches_retained(&self, retained: &Self::Retained) -> bool;

    /// Number of bytes in the exact encoding `E(self)`.
    fn encoding_len(&self) -> usize;

    /// Byte `index` of `E(self)`.
    ///
    /// # Panics
    ///
    /// Panics when `index >= self.encoding_len()`.
    fn encoding_byte(&self, index: usize) -> u8;

    /// Number of bytes in the exact encoding of a retained identity.
    fn retained_encoding_len(retained: &Self::Retained) -> usize;

    /// Byte `index` of the exact encoding of a retained identity.
    ///
    /// # Panics
    ///
    /// Panics when `index >= Self::retained_encoding_len(retained)`.
    fn retained_encoding_byte(retained: &Self::Retained, index: usize) -> u8;
}

/// A zero placement address. `I64WordCodec::encode` ignores its `TableAddress` argument; the
/// value is passed only to satisfy the reviewed codec signature.
fn identity_address() -> TableAddress {
    TableAddress {
        place: 0,
        direct_word: None,
    }
}

/// The reviewed `I64WordCodec` exact identity word for a word key.
fn word_identity(key: i64) -> u64 {
    <I64WordCodec as ExactCodec<i64, WordHome>>::encode(
        Sigma::Hyperbolic,
        &key,
        &identity_address(),
    )
}

/// The word recovered from an `I64WordCodec` identity word.
fn word_from_identity(identity: u64) -> i64 {
    <I64WordCodec as ExactCodec<i64, WordHome>>::decode(Sigma::Hyperbolic, identity)
}

impl ExactIdentity for i64 {
    type Retained = u64;
    type Reserved = ();

    fn retain(&self) -> u64 {
        word_identity(*self)
    }

    fn reserve_retain(&self) -> Result<(), Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> u64 {
        word_identity(*self)
    }

    fn matches_retained(&self, retained: &u64) -> bool {
        word_identity(*self) == *retained
    }

    fn encoding_len(&self) -> usize {
        8
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        // The identity bytes are the original bit pattern, never the residue.
        self.cast_unsigned().to_be_bytes()[index]
    }

    fn retained_encoding_len(_: &u64) -> usize {
        8
    }

    fn retained_encoding_byte(retained: &u64, index: usize) -> u8 {
        word_from_identity(*retained).cast_unsigned().to_be_bytes()[index]
    }
}

impl ExactIdentity for u64 {
    type Retained = u64;
    type Reserved = ();

    fn retain(&self) -> u64 {
        word_identity(self.cast_signed())
    }

    fn reserve_retain(&self) -> Result<(), Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> u64 {
        word_identity(self.cast_signed())
    }

    fn matches_retained(&self, retained: &u64) -> bool {
        word_identity(self.cast_signed()) == *retained
    }

    fn encoding_len(&self) -> usize {
        8
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        self.to_be_bytes()[index]
    }

    fn retained_encoding_len(_: &u64) -> usize {
        8
    }

    fn retained_encoding_byte(retained: &u64, index: usize) -> u8 {
        word_from_identity(*retained).cast_unsigned().to_be_bytes()[index]
    }
}

#[allow(
    clippy::ptr_arg,
    reason = "the retained identity is the byte vector itself; a slice parameter would not name \
              the associated type the locator stores"
)]
impl ExactIdentity for Vec<u8> {
    type Retained = Vec<u8>;
    type Reserved = Vec<u8>;

    fn retain(&self) -> Vec<u8> {
        self.clone()
    }

    fn reserve_retain(&self) -> Result<Vec<u8>, Refusal> {
        let mut reserved = Vec::new();
        reserved
            .try_reserve_exact(self.len())
            .map_err(|_| Refusal::AllocationFailed)?;
        Ok(reserved)
    }

    fn retain_reserved(&self, mut reserved: Vec<u8>) -> Vec<u8> {
        // The capacity was reserved, so this extends in place and does not allocate.
        reserved.extend_from_slice(self);
        reserved
    }

    fn matches_retained(&self, retained: &Vec<u8>) -> bool {
        self == retained
    }

    fn encoding_len(&self) -> usize {
        self.len()
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        self[index]
    }

    fn retained_encoding_len(retained: &Vec<u8>) -> usize {
        retained.len()
    }

    fn retained_encoding_byte(retained: &Vec<u8>, index: usize) -> u8 {
        retained[index]
    }
}

/// A shared key delegates its exact identity to the pointee.
///
/// `Arc<T>` compares its pointees for equality, so the exact identity of `Arc<T>` is exactly the
/// pointee's: every operation forwards to `**self` and `Arc` contributes no identity byte and no
/// reservation of its own. `T::Retained` is unchanged, so a shared key and its pointee retain the
/// same value and reach the same trie terminal.
impl<T: ExactIdentity> ExactIdentity for std::sync::Arc<T> {
    type Retained = T::Retained;
    type Reserved = T::Reserved;

    fn retain(&self) -> Self::Retained {
        T::retain(self)
    }

    fn reserve_retain(&self) -> Result<Self::Reserved, Refusal> {
        T::reserve_retain(self)
    }

    fn retain_reserved(&self, reserved: Self::Reserved) -> Self::Retained {
        T::retain_reserved(self, reserved)
    }

    fn matches_retained(&self, retained: &Self::Retained) -> bool {
        T::matches_retained(self, retained)
    }

    fn encoding_len(&self) -> usize {
        T::encoding_len(self)
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        T::encoding_byte(self, index)
    }

    fn retained_encoding_len(retained: &Self::Retained) -> usize {
        T::retained_encoding_len(retained)
    }

    fn retained_encoding_byte(retained: &Self::Retained, index: usize) -> u8 {
        T::retained_encoding_byte(retained, index)
    }
}

/// Deterministic one-shot reservation-failure injection for [`Directory::reserve_for`].
///
/// This is an inherent seam, not a `#[cfg(test)]` one: the zero-effects tests live in
/// `tests/native_locator.rs`, an external integration crate that cannot observe `#[cfg(test)]`
/// items. Two one-shot arms exist, both consumed by the same insert:
///
/// * [`arm`] fires at the entry to the reserve-all phase, before any bound check, arena
///   reservation, node, edge, payload or link is written.
/// * [`arm_late_edge`] fires at the identity-edge reservation inside the reserve-all phase, after
///   the staged residue path exists but before retention. Moving that reservation after
///   [`ExactIdentity::retain_reserved`] is the ordering the seam is meant to reject.
///
/// Either hit leaves the committed links, payloads, counts and coverage assertion unchanged and
/// publishes nothing. A production call never arms either, and each arm is one-shot, so it cannot
/// leak into a later operation on the same thread.
#[doc(hidden)]
#[cfg(feature = "injection")]
pub mod reservation_failure {
    use std::cell::Cell;

    use super::Refusal;

    thread_local! {
        static ARMED: Cell<Option<Refusal>> = const { Cell::new(None) };
        static LATE_EDGE: Cell<Option<Refusal>> = const { Cell::new(None) };
    }

    /// Arms a one-shot refusal of `refusal`'s kind at the reserve-all entry on the calling thread.
    pub fn arm(refusal: Refusal) {
        ARMED.with(|cell| cell.set(Some(refusal)));
    }

    /// Arms a one-shot refusal of `refusal`'s kind at the identity-edge reservation on the calling
    /// thread.
    pub fn arm_late_edge(refusal: Refusal) {
        LATE_EDGE.with(|cell| cell.set(Some(refusal)));
    }

    /// Clears any armed refusal on the calling thread.
    pub fn disarm() {
        ARMED.with(|cell| cell.set(None));
        LATE_EDGE.with(|cell| cell.set(None));
    }

    /// Consumes the entry arm, if any, and reports the armed refusal.
    pub(crate) fn consume() -> Option<Refusal> {
        ARMED.with(|cell| {
            let armed = cell.get();
            if armed.is_some() {
                cell.set(None);
            }
            armed
        })
    }

    /// Consumes the late-edge arm, if any, and reports the armed refusal.
    pub(crate) fn consume_late_edge() -> Option<Refusal> {
        LATE_EDGE.with(|cell| {
            let armed = cell.get();
            if armed.is_some() {
                cell.set(None);
            }
            armed
        })
    }
}

/// A face's residue index: the digit sequence and arena its trie traverses.
///
/// The two admitted faces differ only in this parameter. The parabolic face reads the 64 bits of
/// the scalar coefficient `a` in a binary (arity-2) trie; the hyperbolic face reads the 40
/// canonical base-3 digits of a residue in a ternary (arity-3) trie. Everything downstream of the
/// digit sequence is shared.
pub(crate) trait ResidueIndex {
    /// Number of residue levels the trie traverses.
    const LEVELS: usize;
    /// Number of children one node may hold: 2 for the parabolic bit, 3 for the hyperbolic digit.
    const ARITY: usize;
    /// Which root handle this face owns in the shared residue arena.
    const ARENA: u32;
    /// Writes the canonical digits least-significant first into `out`.
    fn digits(&self, out: &mut [u8]);
}

impl ResidueIndex for ParabolicDual {
    const LEVELS: usize = super::PARABOLIC_LEVELS;
    const ARITY: usize = 2;
    const ARENA: u32 = 1;

    #[allow(
        clippy::cast_possible_truncation,
        reason = "`(a >> level) & 1` is 0 or 1, so the narrowing to u8 cannot truncate"
    )]
    fn digits(&self, out: &mut [u8]) {
        let mut level = 0;
        while level < out.len() {
            out[level] = ((self.residue() >> level) & 1) as u8;
            level += 1;
        }
    }
}

impl ResidueIndex for Residue {
    const LEVELS: usize = super::LEVELS;
    const ARITY: usize = 3;
    const ARENA: u32 = 0;

    #[allow(
        clippy::cast_possible_truncation,
        reason = "`value % 3` is below 3, so the narrowing to u8 cannot truncate"
    )]
    fn digits(&self, out: &mut [u8]) {
        let mut value = self.get();
        let mut level = 0;
        while level < out.len() {
            out[level] = (value % 3) as u8;
            value /= 3;
            level += 1;
        }
    }
}

/// One byte-labelled edge, copied out of the flat sibling-edge arena.
#[derive(Clone, Copy, Default)]
pub(crate) struct Edge {
    pub(crate) label: u8,
    /// The edge's child identity node.
    ///
    /// Part of the copied edge record a validator sees; the engine traverses the flat edge arena
    /// directly, so the current cold validators inspect only `label` and `terminal`.
    #[allow(
        dead_code,
        reason = "the copied edge record carries its target for a validator; the engine reads the flat arena"
    )]
    pub(crate) child: u32,
}

/// The sorted, duplicate-free outgoing edge list of one identity node, as a view over the flat
/// sibling-edge arena.
///
/// Reading a record copies at most one edge; no `Vec` is materialized and no borrow outlives a
/// step. [`EdgeList::capacity`] reports the one edge record every identity node reserves when it is
/// created; [`EdgeList::len`] walks the sibling chain for the cold validators.
#[derive(Clone, Copy)]
pub(crate) struct EdgeList<'a> {
    edge_words: &'a [u32],
    first_edge: u32,
}

impl EdgeList<'_> {
    /// The number of edges on the sibling chain.
    pub(crate) fn len(&self) -> usize {
        let mut len = 0;
        let mut edge = self.first_edge;
        while edge != NO_CHILD {
            len += 1;
            edge = read_word(self.edge_words, EDGE_STRIDE * edge as usize + EDGE_NEXT);
        }
        len
    }

    /// The number of edge records reserved for this node.
    ///
    /// Every identity node reserves exactly one outgoing edge record when
    /// [`Directory::new_identity_node`] creates it, so a copied edge list reports one. The
    /// reservation receipt observes this per-node count; the flat arena's total record capacity is
    /// counted by stride in [`Directory`].
    #[allow(
        clippy::unused_self,
        reason = "the copied edge view is the call-site interface `table::locator` observes \
                  (`node.edges.capacity()`); the per-view capacity is the constant one reserved \
                  edge record, and the method form is preserved"
    )]
    pub(crate) const fn capacity(&self) -> usize {
        1
    }

    /// Consecutive adjacent-pair windows over the sorted chain.
    ///
    /// The cold validators use the width-2 window; the width argument is accepted for call-site
    /// compatibility with the former `Vec::windows`. Only `width == 2` yields windows.
    pub(crate) fn windows(&self, width: usize) -> EdgeWindows<'_> {
        debug_assert!(
            width == 2,
            "the copied edge view implements the adjacent-pair window"
        );
        EdgeWindows {
            edge_words: self.edge_words,
            start: self.first_edge,
            width,
        }
    }
}

/// The [`EdgeList::windows`] iterator over copied adjacent edge pairs.
pub(crate) struct EdgeWindows<'a> {
    edge_words: &'a [u32],
    start: u32,
    width: usize,
}

impl Iterator for EdgeWindows<'_> {
    type Item = [Edge; 2];

    fn next(&mut self) -> Option<[Edge; 2]> {
        if self.width != 2 || self.start == NO_CHILD {
            return None;
        }
        let first_at = EDGE_STRIDE * self.start as usize;
        let second = read_word(self.edge_words, first_at + EDGE_NEXT);
        if second == NO_CHILD {
            return None;
        }
        let second_at = EDGE_STRIDE * second as usize;
        let window = [
            Edge {
                label: label_byte(self.edge_words, first_at + EDGE_LABEL),
                child: read_word(self.edge_words, first_at + EDGE_CHILD),
            },
            Edge {
                label: label_byte(self.edge_words, second_at + EDGE_LABEL),
                child: read_word(self.edge_words, second_at + EDGE_CHILD),
            },
        ];
        self.start = second;
        Some(window)
    }
}

/// One identity node as a copied view: its terminal handle and a view of its outgoing edges.
#[derive(Clone, Copy)]
pub(crate) struct IdentityNode<'a> {
    pub(crate) terminal: Option<u32>,
    pub(crate) edges: EdgeList<'a>,
}

/// A view of every identity node, for the caller's cold coverage check.
///
/// The engine's storage is the flat `identity_words`/`edge_words` arenas, so a node's edges are a
/// view rather than a slice. The handle borrows the directory for `'a`, and [`IdentityNodes::get`]
/// returns an owned [`IdentityNode`] view, so a validator may hold a record across statements. This
/// is a cold diagnostic view; no traversal reads through it and none of it allocates.
#[derive(Clone, Copy)]
pub(crate) struct IdentityNodes<'a> {
    directory: &'a Directory,
}

impl<'a> IdentityNodes<'a> {
    /// The copied view of identity node `index`, or `None` out of range.
    pub(crate) fn get(self, index: usize) -> Option<IdentityNode<'a>> {
        self.directory.identity_node_view(index)
    }

    /// Iterates the copied views in node order.
    pub(crate) fn iter(self) -> IdentityNodeIter<'a> {
        IdentityNodeIter {
            directory: self.directory,
            index: 0,
            count: self.directory.identity_node_count(),
        }
    }
}

impl<'a> IntoIterator for IdentityNodes<'a> {
    type Item = IdentityNode<'a>;
    type IntoIter = IdentityNodeIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// The [`IdentityNodes`] iterator over copied views.
pub(crate) struct IdentityNodeIter<'a> {
    directory: &'a Directory,
    index: usize,
    count: usize,
}

impl<'a> Iterator for IdentityNodeIter<'a> {
    type Item = IdentityNode<'a>;

    fn next(&mut self) -> Option<IdentityNode<'a>> {
        if self.index >= self.count {
            return None;
        }
        let node = self.directory.identity_node_at(self.index);
        self.index += 1;
        Some(node)
    }
}

/// What one generic traversal found.
pub(crate) enum Walk {
    /// The complete residue and identity path ended at a terminal handle.
    Terminal(u32),
    /// The path stopped; `residue_levels` and `identity_bytes` describe the covered prefix.
    Miss {
        residue_levels: usize,
        identity_bytes: usize,
    },
}

/// The staged node indices of a new representative's complete path.
pub(crate) struct StagedPath {
    pub(crate) residue_path: [u32; MAX_LEVELS + 1],
    pub(crate) identity_path: Vec<u32>,
}

/// The shared directory engine: one flat scalar residue arena for both faces and one flat
/// exact-identity byte trie (identity-node records plus sorted sibling-edge records).
///
/// Every arena is a `Vec<u32>` of fixed-stride records, so each traversal step is a scalar indexed
/// load or store through [`read_word`]/[`store_word`], and no borrowed node, subslice or nested
/// field projection survives a step. A record handle is a logical record number, stable across
/// `Vec` growth; [`NO_CHILD`] tags an absent child, identity root, terminal, first edge or next
/// edge. The engine owns no value and no key: a terminal is a bounded opaque handle into the
/// caller's representative registry. Both admitted faces share this one implementation and its
/// coverage invariant; they differ only in their [`ResidueIndex`] parameter and their root handle.
pub(crate) struct Directory {
    /// Residue node records, stride 4: `[child0, child1, child2, identity_root]`.
    ///
    /// Node 0 is the hyperbolic root and node 1 the parabolic root; every descendant keeps its
    /// root's ownership, so equal residue values on the two faces never alias. An empty vector is
    /// two implicit empty roots with logical counts 1/1; the first reservation materializes both.
    residue_words: Vec<u32>,
    /// Identity node records, stride 2: `[terminal_handle, first_edge]`.
    identity_words: Vec<u32>,
    /// Sibling-edge records, stride 3: `[byte_label, child_identity_node, next_edge]`.
    edge_words: Vec<u32>,
    /// The number of logical hyperbolic residue nodes, including the implicit root.
    hyperbolic_nodes: usize,
    /// The number of logical parabolic residue nodes, including the implicit root.
    parabolic_nodes: usize,
}

/// The stride, in `u32` words, of one residue node record.
const RESIDUE_STRIDE: usize = 4;
/// The word offset of the identity-trie root in a residue record.
const RESIDUE_IDENTITY: usize = 3;
/// The stride, in `u32` words, of one identity node record.
const IDENTITY_STRIDE: usize = 2;
/// The word offset of the terminal handle in an identity record.
const IDENTITY_TERMINAL: usize = 0;
/// The word offset of the first edge in an identity record.
const IDENTITY_FIRST_EDGE: usize = 1;
/// The stride, in `u32` words, of one sibling-edge record.
const EDGE_STRIDE: usize = 3;
/// The word offset of the byte label in an edge record.
const EDGE_LABEL: usize = 0;
/// The word offset of the child identity node in an edge record.
const EDGE_CHILD: usize = 1;
/// The word offset of the next sibling edge in an edge record.
const EDGE_NEXT: usize = 2;

/// One indexed scalar load from a flat word arena. Returns a copied word, never a reference.
fn read_word(words: &[u32], index: usize) -> u32 {
    words[index]
}

/// One indexed scalar load narrowed to a byte label; an out-of-range word becomes `0`.
fn label_byte(words: &[u32], index: usize) -> u8 {
    let value = read_word(words, index);
    if value < 256 {
        value as u8
    } else {
        0
    }
}

/// One indexed scalar store into a flat word arena. Returns no borrow.
fn store_word(words: &mut [u32], index: usize, value: u32) {
    words[index] = value;
}

impl Directory {
    /// The empty directory: no records, two implicit residue roots, no identity nodes.
    pub(crate) fn new() -> Self {
        Self {
            residue_words: Vec::new(),
            identity_words: Vec::new(),
            edge_words: Vec::new(),
            hyperbolic_nodes: 1,
            parabolic_nodes: 1,
        }
    }

    /// The number of materialized residue records.
    fn residue_record_count(&self) -> usize {
        self.residue_words.len() / RESIDUE_STRIDE
    }

    /// The number of occupied nodes in `I`'s residue arena, including its root.
    pub(crate) fn arena_len<I: ResidueIndex>(&self) -> usize {
        if I::ARENA == <Residue as ResidueIndex>::ARENA {
            self.hyperbolic_nodes
        } else {
            self.parabolic_nodes
        }
    }

    /// The number of identity nodes.
    pub(crate) fn identity_node_count(&self) -> usize {
        self.identity_words.len() / IDENTITY_STRIDE
    }

    /// The total number of byte-labelled identity edges.
    pub(crate) fn edge_count(&self) -> usize {
        self.edge_words.len() / EDGE_STRIDE
    }

    /// The largest number of edges on one identity node.
    pub(crate) fn max_identity_fanout(&self) -> usize {
        let nodes = self.identity_node_count();
        let mut max = 0;
        let mut node = 0;
        while node < nodes {
            let mut fanout = 0;
            let mut edge = read_word(
                &self.identity_words,
                IDENTITY_STRIDE * node + IDENTITY_FIRST_EDGE,
            );
            while edge != NO_CHILD {
                fanout += 1;
                edge = read_word(&self.edge_words, EDGE_STRIDE * edge as usize + EDGE_NEXT);
            }
            if fanout > max {
                max = fanout;
            }
            node += 1;
        }
        max
    }

    /// A copied view of every identity node, for the caller's coverage check.
    pub(crate) fn identity_nodes(&self) -> IdentityNodes<'_> {
        IdentityNodes { directory: self }
    }

    /// The copied record of identity node `index`, or `None` out of range.
    fn identity_node_view(&self, index: usize) -> Option<IdentityNode<'_>> {
        if IDENTITY_STRIDE * index >= self.identity_words.len() {
            return None;
        }
        Some(self.identity_node_at(index))
    }

    /// The copied view of identity node `index`. The caller has checked the range.
    fn identity_node_at(&self, index: usize) -> IdentityNode<'_> {
        let terminal = read_word(
            &self.identity_words,
            IDENTITY_STRIDE * index + IDENTITY_TERMINAL,
        );
        let terminal = if terminal == NO_CHILD {
            None
        } else {
            Some(terminal)
        };
        let first_edge = read_word(
            &self.identity_words,
            IDENTITY_STRIDE * index + IDENTITY_FIRST_EDGE,
        );
        IdentityNode {
            terminal,
            edges: EdgeList {
                edge_words: &self.edge_words,
                first_edge,
            },
        }
    }

    /// The identity-trie root handle stored on a residue node.
    fn residue_identity_root(&self, node: u32) -> u32 {
        read_word(
            &self.residue_words,
            RESIDUE_STRIDE * node as usize + RESIDUE_IDENTITY,
        )
    }

    /// The child handle at `digit` on a residue node, or [`NO_CHILD`] when the record is absent
    /// (an implicit root) or the child is unset.
    fn residue_child(&self, node: u32, digit: u8) -> u32 {
        debug_assert!(usize::from(digit) < MAX_ARITY);
        let at = RESIDUE_STRIDE * node as usize + usize::from(digit);
        if at < self.residue_words.len() {
            read_word(&self.residue_words, at)
        } else {
            NO_CHILD
        }
    }

    /// The complete face residue and identity traversal, read-only.
    pub(crate) fn walk<I: ResidueIndex, K: ExactIdentity>(&self, index: &I, key: &K) -> Walk {
        let mut digits = [0u8; MAX_LEVELS];
        index.digits(&mut digits[..I::LEVELS]);
        let root = I::ARENA;
        let (node, edge_depth) =
            walk_residue(&self.residue_words, &digits, I::LEVELS, I::ARITY, root);
        if node == NO_CHILD {
            Walk::Miss {
                residue_levels: edge_depth,
                identity_bytes: 0,
            }
        } else {
            let identity_root = self.residue_identity_root(node);
            if identity_root == NO_CHILD {
                Walk::Miss {
                    residue_levels: I::LEVELS,
                    identity_bytes: 0,
                }
            } else {
                self.walk_identity::<I, K>(identity_root, key)
            }
        }
    }

    /// The exact-identity traversal from a residue leaf, read-only.
    fn walk_identity<I: ResidueIndex, K: ExactIdentity>(&self, root: u32, key: &K) -> Walk {
        let len = key.encoding_len();
        let mut idnode = root;
        let mut miss: Option<Walk> = None;
        let mut index = 0;
        while index < len && miss.is_none() {
            let label = key.encoding_byte(index);
            let child = find_edge(&self.identity_words, &self.edge_words, idnode, label);
            if child == NO_CHILD {
                miss = Some(Walk::Miss {
                    residue_levels: I::LEVELS,
                    identity_bytes: index,
                });
            } else {
                idnode = child;
            }
            index += 1;
        }
        if let Some(miss) = miss {
            miss
        } else {
            let terminal = read_word(
                &self.identity_words,
                IDENTITY_STRIDE * idnode as usize + IDENTITY_TERMINAL,
            );
            if terminal == NO_CHILD {
                Walk::Miss {
                    residue_levels: I::LEVELS,
                    identity_bytes: len,
                }
            } else {
                Walk::Terminal(terminal)
            }
        }
    }

    /// The terminal identity node a retained identity reaches from a residue leaf, or `None` if a
    /// path edge is missing.
    pub(crate) fn representative_terminal<I: ResidueIndex, K: ExactIdentity>(
        &self,
        index: &I,
        identity: &K::Retained,
    ) -> Option<u32> {
        let mut digits = [0u8; MAX_LEVELS];
        index.digits(&mut digits[..I::LEVELS]);
        let mut node = I::ARENA;
        let mut missing = false;
        let mut depth = 0;
        while depth < I::LEVELS && !missing {
            let digit = digits[depth];
            let child = self.residue_child(node, digit);
            if child == NO_CHILD {
                missing = true;
            } else {
                node = child;
            }
            depth += 1;
        }
        if missing {
            None
        } else {
            let root = self.residue_identity_root(node);
            if root == NO_CHILD {
                None
            } else {
                self.identity_terminal::<K>(root, identity)
            }
        }
    }

    /// The terminal identity node a retained identity reaches from a residue leaf.
    fn identity_terminal<K: ExactIdentity>(
        &self,
        root: u32,
        identity: &K::Retained,
    ) -> Option<u32> {
        let len = K::retained_encoding_len(identity);
        let mut node = root;
        let mut found = true;
        let mut byte = 0;
        while byte < len && found {
            let label = K::retained_encoding_byte(identity, byte);
            let child = find_edge(&self.identity_words, &self.edge_words, node, label);
            if child == NO_CHILD {
                found = false;
            } else {
                node = child;
            }
            byte += 1;
        }
        if found {
            Some(node)
        } else {
            None
        }
    }

    /// Checks count, byte-size and handle bounds and reserves all new storage.
    ///
    /// This is the whole reserve-all phase: every residue, identity and edge record, the complete
    /// staged residue and identity path, and the retained-identity storage. The staged path is
    /// created here, but no link is published; on any nested reservation failure the arenas are
    /// truncated to their committed lengths, so a failure leaves the committed state unchanged. An
    /// armed [`reservation_failure`] entry refusal is consumed by the caller before this step; an
    /// armed [`reservation_failure::arm_late_edge`] refusal fires at the identity-edge reservation
    /// inside [`Directory::stage_reserved`]. Both arms are one-shot.
    pub(crate) fn reserve_for<I: ResidueIndex, K: ExactIdentity>(
        &mut self,
        index: &I,
        key: &K,
    ) -> Result<(K::Reserved, StagedPath), Refusal> {
        let encoding_len = key.encoding_len();
        let new_identity = encoding_len.checked_add(1).ok_or(Refusal::SizeOverflow)?;
        let residue_records = self.residue_record_count();
        let residue_base = match residue_records.checked_add(2) {
            Some(value) => value,
            None => return Err(Refusal::SizeOverflow),
        };
        let residue_levels = match residue_base.checked_add(LEVELS) {
            Some(value) => value,
            None => return Err(Refusal::SizeOverflow),
        };
        let residue_end = match residue_levels.checked_add(PARABOLIC_LEVELS) {
            Some(value) => value,
            None => return Err(Refusal::SizeOverflow),
        };
        let identity_end = self
            .identity_node_count()
            .checked_add(new_identity)
            .ok_or(Refusal::SizeOverflow)?;
        let edge_reserve = new_identity.checked_mul(2).ok_or(Refusal::SizeOverflow)?;
        let edge_end = self
            .edge_count()
            .checked_add(edge_reserve)
            .ok_or(Refusal::SizeOverflow)?;
        if residue_end > u32::MAX as usize
            || identity_end > u32::MAX as usize
            || edge_end > u32::MAX as usize
        {
            return Err(Refusal::SizeOverflow);
        }
        let residue_delta = match residue_end.checked_sub(residue_records) {
            Some(value) => value,
            None => return Err(Refusal::SizeOverflow),
        };
        let residue_additional = match residue_delta.checked_mul(RESIDUE_STRIDE) {
            Some(value) => value,
            None => return Err(Refusal::SizeOverflow),
        };
        let identity_additional = new_identity
            .checked_mul(IDENTITY_STRIDE)
            .ok_or(Refusal::SizeOverflow)?;
        let edge_additional = edge_reserve
            .checked_mul(EDGE_STRIDE)
            .ok_or(Refusal::SizeOverflow)?;
        if self.residue_words.try_reserve(residue_additional).is_err()
            || self
                .identity_words
                .try_reserve(identity_additional)
                .is_err()
            || self.edge_words.try_reserve(edge_additional).is_err()
        {
            return Err(Refusal::AllocationFailed);
        }
        let path_len = encoding_len.checked_add(1).ok_or(Refusal::SizeOverflow)?;
        let mut identity_path = Vec::new();
        identity_path
            .try_reserve_exact(path_len)
            .map_err(|_| Refusal::AllocationFailed)?;
        let saved_residue = self.residue_words.len();
        let saved_identity = self.identity_words.len();
        let saved_edges = self.edge_words.len();
        let saved_hyperbolic = self.hyperbolic_nodes;
        let saved_parabolic = self.parabolic_nodes;
        self.materialize_roots();
        let staged = match self.stage_reserved::<I, K>(index, key, identity_path) {
            Ok(staged) => staged,
            Err(refusal) => {
                self.rollback(
                    saved_residue,
                    saved_identity,
                    saved_edges,
                    saved_hyperbolic,
                    saved_parabolic,
                );
                return Err(refusal);
            }
        };
        let reserved = match key.reserve_retain() {
            Ok(reserved) => reserved,
            Err(refusal) => {
                self.rollback(
                    saved_residue,
                    saved_identity,
                    saved_edges,
                    saved_hyperbolic,
                    saved_parabolic,
                );
                return Err(refusal);
            }
        };
        Ok((reserved, staged))
    }

    /// Materializes both implicit residue roots, leaving the logical node counts unchanged.
    fn materialize_roots(&mut self) {
        if self.residue_words.len() == 0 {
            let mut word = 0;
            while word < 2 * RESIDUE_STRIDE {
                self.residue_words.push(NO_CHILD);
                word += 1;
            }
        }
    }

    /// Stages the complete new path into the pre-reserved arenas without publishing any link.
    ///
    /// Every arena was reserved by [`Directory::reserve_for`] before retention, and each identity
    /// edge slot is reserved through [`Directory::reserve_edge`] here, so staging performs the
    /// fallible edge reservations before retention and never after it. On any nested reservation
    /// failure the caller truncates the staged arenas to their saved lengths, so nothing is
    /// observable through the caller's read.
    fn stage_reserved<I: ResidueIndex, K: ExactIdentity>(
        &mut self,
        index: &I,
        key: &K,
        mut identity_path: Vec<u32>,
    ) -> Result<StagedPath, Refusal> {
        let mut residue_path = [0u32; MAX_LEVELS + 1];
        let mut digits = [0u8; MAX_LEVELS];
        index.digits(&mut digits[..I::LEVELS]);
        residue_path[0] = I::ARENA;
        // Single exit: each loop records the first refusal and stops, and the one `Result` return
        // follows both. A `?` in a loop is an early return, which Aeneas does not translate.
        let mut refusal: Option<Refusal> = None;
        // A `break` inside a `for` loop is an Aeneas-shape failure; the refusal guard is in the
        // `while` condition instead, so each loop is single-exit.
        let mut depth = 0;
        while depth < I::LEVELS && refusal.is_none() {
            let digit = digits[depth];
            debug_assert!(usize::from(digit) < I::ARITY);
            let parent = residue_path[depth];
            let child = self.residue_child(parent, digit);
            if child == NO_CHILD {
                match u32::try_from(self.residue_record_count()) {
                    Ok(next) => {
                        self.push_residue_default();
                        self.bump_face_nodes::<I>();
                        residue_path[depth + 1] = next;
                    }
                    Err(_) => refusal = Some(Refusal::SizeOverflow),
                }
            } else {
                residue_path[depth + 1] = child;
            }
            depth += 1;
        }
        let mut node = 0u32;
        if refusal.is_none() {
            let leaf = residue_path[I::LEVELS];
            let root = self.residue_identity_root(leaf);
            if root == NO_CHILD {
                match self.new_identity_node() {
                    Ok(root) => {
                        identity_path.clear();
                        identity_path.push(root);
                        node = root;
                    }
                    Err(err) => refusal = Some(err),
                }
            } else {
                identity_path.clear();
                identity_path.push(root);
                node = root;
            }
        }
        if refusal.is_none() {
            let byte_len = key.encoding_len();
            let mut byte_index = 0;
            while byte_index < byte_len && refusal.is_none() {
                let label = key.encoding_byte(byte_index);
                let child = find_edge(&self.identity_words, &self.edge_words, node, label);
                if child == NO_CHILD {
                    match self.reserve_edge() {
                        Ok(()) => match self.new_identity_node() {
                            Ok(next) => {
                                identity_path.push(next);
                                node = next;
                            }
                            Err(err) => refusal = Some(err),
                        },
                        Err(err) => refusal = Some(err),
                    }
                } else {
                    identity_path.push(child);
                    node = child;
                }
                byte_index += 1;
            }
        }
        match refusal {
            Some(refusal) => Err(refusal),
            None => Ok(StagedPath {
                residue_path,
                identity_path,
            }),
        }
    }

    /// Appends one default residue record for the selected face.
    fn push_residue_default(&mut self) {
        let mut word = 0;
        while word < RESIDUE_STRIDE {
            self.residue_words.push(NO_CHILD);
            word += 1;
        }
    }

    /// Counts one new node in the selected face's logical arena.
    fn bump_face_nodes<I: ResidueIndex>(&mut self) {
        if I::ARENA == <Residue as ResidueIndex>::ARENA {
            self.hyperbolic_nodes += 1;
        } else {
            self.parabolic_nodes += 1;
        }
    }

    /// Reserves one edge record, after checking the late-edge failure seam.
    ///
    /// The seam is consumed at the real edge reservation, so a forced late failure occurs before
    /// the retained identity is reserved or filled. The capacity change alone is not observable, so
    /// a failed reservation leaves the committed state unchanged.
    fn reserve_edge(&mut self) -> Result<(), Refusal> {
        #[cfg(feature = "injection")]
        if let Some(refusal) = reservation_failure::consume_late_edge() {
            return Err(refusal);
        }
        if self.edge_words.try_reserve(EDGE_STRIDE).is_err() {
            return Err(Refusal::AllocationFailed);
        }
        Ok(())
    }

    /// Appends one identity node with its edge slot reserved before retention.
    fn new_identity_node(&mut self) -> Result<u32, Refusal> {
        let Ok(index) = u32::try_from(self.identity_node_count()) else {
            return Err(Refusal::SizeOverflow);
        };
        self.identity_words.push(NO_CHILD);
        self.identity_words.push(NO_CHILD);
        self.reserve_edge()?;
        Ok(index)
    }

    /// Discards staged records after a failed reservation.
    fn rollback(
        &mut self,
        residue_len: usize,
        identity_len: usize,
        edge_len: usize,
        hyperbolic_nodes: usize,
        parabolic_nodes: usize,
    ) {
        self.residue_words.truncate(residue_len);
        self.identity_words.truncate(identity_len);
        self.edge_words.truncate(edge_len);
        self.hyperbolic_nodes = hyperbolic_nodes;
        self.parabolic_nodes = parabolic_nodes;
    }

    /// Publishes the staged path. Every write is infallible: all storage was reserved.
    pub(crate) fn link<I: ResidueIndex, K: ExactIdentity>(
        &mut self,
        index: &I,
        key: &K,
        staged: &StagedPath,
        handle: u32,
    ) {
        let mut digits = [0u8; MAX_LEVELS];
        index.digits(&mut digits[..I::LEVELS]);
        let mut depth = 0;
        while depth < I::LEVELS {
            let digit = digits[depth];
            debug_assert!(usize::from(digit) < I::ARITY);
            let parent = staged.residue_path[depth] as usize;
            let at = RESIDUE_STRIDE * parent + usize::from(digit);
            if read_word(&self.residue_words, at) == NO_CHILD {
                store_word(&mut self.residue_words, at, staged.residue_path[depth + 1]);
            }
            depth += 1;
        }
        let leaf = staged.residue_path[I::LEVELS] as usize;
        let root_at = RESIDUE_STRIDE * leaf + RESIDUE_IDENTITY;
        if read_word(&self.residue_words, root_at) == NO_CHILD {
            store_word(&mut self.residue_words, root_at, staged.identity_path[0]);
        }
        let byte_len = key.encoding_len();
        let mut byte_index = 0;
        while byte_index < byte_len {
            let parent = staged.identity_path[byte_index];
            let label = key.encoding_byte(byte_index);
            let child = staged.identity_path[byte_index + 1];
            self.link_identity_edge(parent, label, child);
            byte_index += 1;
        }
        let terminal_node = staged.identity_path[key.encoding_len()];
        store_word(
            &mut self.identity_words,
            IDENTITY_STRIDE * terminal_node as usize + IDENTITY_TERMINAL,
            handle,
        );
    }

    /// Publishes one staged identity edge after the read-only sorted search has ended.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "reserve_for checked that every edge record handle fits below the NO_CHILD sentinel"
    )]
    fn link_identity_edge(&mut self, parent: u32, label: u8, child: u32) {
        let first_at = IDENTITY_STRIDE * parent as usize + IDENTITY_FIRST_EDGE;
        let mut previous = NO_CHILD;
        let mut edge = read_word(&self.identity_words, first_at);
        while edge != NO_CHILD
            && read_word(&self.edge_words, EDGE_STRIDE * edge as usize + EDGE_LABEL)
                < u32::from(label)
        {
            previous = edge;
            edge = read_word(&self.edge_words, EDGE_STRIDE * edge as usize + EDGE_NEXT);
        }
        let mut present = false;
        if edge != NO_CHILD {
            present = read_word(&self.edge_words, EDGE_STRIDE * edge as usize + EDGE_LABEL)
                == u32::from(label);
        }
        if !present {
            let new_edge = self.edge_count() as u32;
            self.edge_words.push(u32::from(label));
            self.edge_words.push(child);
            self.edge_words.push(edge);
            if previous == NO_CHILD {
                store_word(&mut self.identity_words, first_at, new_edge);
            } else {
                store_word(
                    &mut self.edge_words,
                    EDGE_STRIDE * previous as usize + EDGE_NEXT,
                    new_edge,
                );
            }
        }
    }
}

/// Walks one residue arena from `root`, returning the reached node or `NO_CHILD` and the depth of
/// the last edge followed.
///
/// Single-exit scalar `while`: one indexed load per level through [`read_word`], no borrowed node
/// or nested field projection surviving a step. An implicit empty root (record absent) has every
/// child absent.
fn walk_residue(
    residue_words: &[u32],
    digits: &[u8; MAX_LEVELS],
    levels: usize,
    arity: usize,
    root: u32,
) -> (u32, usize) {
    let mut node = root;
    let mut depth = 0;
    while depth < levels && node != NO_CHILD {
        let digit = digits[depth];
        debug_assert!(usize::from(digit) < arity);
        let at = RESIDUE_STRIDE * node as usize + usize::from(digit);
        node = if at < residue_words.len() {
            read_word(residue_words, at)
        } else {
            NO_CHILD
        };
        depth += 1;
    }
    // `depth` advanced once per edge followed, so the last edge followed sits at `depth - 1`;
    // `saturating_sub` keeps the no-iteration cases at zero. Deriving the reported depth after the
    // loop rather than carrying a second counter keeps the loop body a single scalar update, which
    // Aeneas's loop fixed point closes without a second carried variable.
    (node, depth.saturating_sub(1))
}

/// The child of the sibling edge labelled `label` on `node`, or `NO_CHILD` when absent.
///
/// Single-exit scalar `while` over the sorted sibling chain, one indexed load per record; no
/// borrowed edge record survives a step.
fn find_edge(identity_words: &[u32], edge_words: &[u32], node: u32, label: u8) -> u32 {
    let mut edge = read_word(
        identity_words,
        IDENTITY_STRIDE * node as usize + IDENTITY_FIRST_EDGE,
    );
    let mut found = NO_CHILD;
    while edge != NO_CHILD && found == NO_CHILD {
        let at = EDGE_STRIDE * edge as usize;
        if read_word(edge_words, at + EDGE_LABEL) == u32::from(label) {
            found = read_word(edge_words, at + EDGE_CHILD);
        }
        edge = read_word(edge_words, at + EDGE_NEXT);
    }
    found
}

#[cfg(test)]
mod tests {
    use super::ExactIdentity;
    use std::sync::Arc;

    fn query_bytes<K: ExactIdentity>(key: &K) -> Vec<u8> {
        (0..key.encoding_len())
            .map(|index| key.encoding_byte(index))
            .collect()
    }

    /// `Arc` adds no identity of its own: a shared key's exact bytes are the pointee's, and the
    /// shared key retains the pointee's identity. Dropping or shifting the leading byte in the
    /// delegation changes the byte vector and fails the equality against the pointee.
    #[test]
    fn arc_delegates_the_exact_identity_to_the_pointee() {
        let word = 0x0102_0304_0506_0708i64;
        let shared = Arc::new(word);

        let shared_bytes = query_bytes(&shared);
        assert_eq!(shared_bytes, query_bytes(&word));
        assert_eq!(shared_bytes, word.cast_unsigned().to_be_bytes().to_vec());

        let retained = shared.retain();
        assert_eq!(retained, word.retain());
        assert!(shared.matches_retained(&retained));
        assert!(!shared.matches_retained(&0i64.retain()));

        let retained_bytes: Vec<u8> =
            (0..<Arc<i64> as ExactIdentity>::retained_encoding_len(&retained))
                .map(|index| <Arc<i64> as ExactIdentity>::retained_encoding_byte(&retained, index))
                .collect();
        assert_eq!(retained_bytes, shared_bytes);
    }
}
