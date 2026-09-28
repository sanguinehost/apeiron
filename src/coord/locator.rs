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
//! A sparse, face-indexed residue locator with a shared exact-identity byte trie.
//!
//! The locator is the physical half of the parabolic and hyperbolic faces in
//! the algebra-native-addressing note. It is a standalone structure: it is not wired into
//! [`crate::table::Table`] and it changes no existing read or write.
//!
//! # Per-face indexing
//!
//! * **Elliptic** has no locator. Its exact address is the packing `pack`; the locator refuses
//!   [`FaceIndex::Elliptic`] with [`Refusal::WrongFace`] on a write and answers reads with
//!   [`Lookup::Unknown`] / [`Absence::Unknown`]. The exact-address path stays in
//!   [`crate::coord::address`].
//! * **Parabolic** (`sigma = 0`, the dual numbers `Z/2^64[ε]/ε²`) is indexed by the scalar
//!   coefficient `a = fstHom(a + bε)`, the residue modulo `(ε)`. The trie is binary over the 64
//!   bits of `a`, least-significant bit first. The nilpotent coefficient `b` is never read: it
//!   cannot decide identity or absence, and two keys with equal `a` share one fiber.
//! * **Hyperbolic** (`sigma > 0`, `Z/q`, `q = 3^40`) keeps the ternary trie over the canonical
//!   base-3 digits of a [`Residue`], least-significant digit first, for exactly [`LEVELS`] = 40
//!   levels.
//!
//! Both faces share one exact-identity byte trie. Each face has its own residue root, so equal
//! residue-path values across faces enter different identity subtrees and never alias keys. A
//! terminal is accepted only after the stored representative's face matches and its full identity
//! compares equal.
//!
//! The face-independent machinery — the residue arenas, the generic residue-path traversal, the
//! exact-identity byte trie, the reserve-before-retain ordering and the failure seams — is in the
//! private [`locator_core`] module. This file is the face binding: it maps a [`FaceIndex`] to the
//! face's residue index and interprets terminal handles against the representative registry.
//!
//! # Structure
//!
//! * A **residue trie** over the canonical base-3 digits of a [`Residue`]. Depth `j` reads
//!   `floor(a / 3^j) mod 3`, least-significant digit first, for exactly [`LEVELS`] = 40 levels.
//!   Each node has three optional child handles; a missing edge is data, not a zero sentinel.
//!   The residue zero and every leading zero digit are ordinary data.
//! * A **coefficient trie** over the 64 bits of the parabolic scalar coefficient. Depth `j` reads
//!   bit `j` of `a`, least-significant first, for exactly [`PARABOLIC_LEVELS`] = 64 levels. Each
//!   node has two optional child handles.
//! * At a residue leaf, an **identity trie** over the exact full-key byte encoding `E(k)`
//!   ([`ExactIdentity`]). A node has an optional terminal representative and a sorted,
//!   duplicate-free list of at most [`MAX_LABELS`] = 256 byte-labelled edges. The terminal marker
//!   is separate from every byte, so an empty key, a prefix and a key containing zero differ.
//!
//! The residue is many-to-one and never decides identity: only the identity path does, and only
//! after a full-key comparison at the terminal. Word keys reuse the reviewed
//! [`I64WordCodec`] as their retained exact identity.
//!
//! # Coverage
//!
//! Every live representative has its complete residue path and its complete identity path ending
//! at that representative; every terminal points back to the one live representative that owns it;
//! and encoded keys and edges are unique. [`Locator::new`] establishes the invariant on the empty
//! structure, and every successful insert preserves it. A missing edge or terminal therefore
//! certifies absence. When the invariant is not established, the locator returns
//! [`Absence::Unknown`] instead of a certificate; it never reports absence for a present key.
//!
//! # Reserve-before-retain ordering
//!
//! [`Locator::insert`] follows `checked sizing -> reserve all -> retain identity -> publish ->
//! commit`. [`Locator::reserve_for`] checks every count and handle bound, reserves the residue,
//! coefficient, identity and representative arenas, reserves the staged identity path, stages the
//! complete new path into the reserved arenas, reserves every identity edge slot on that path, and
//! reserves the retained-identity storage ([`ExactIdentity::reserve_retain`]);
//! [`ExactIdentity::retain_reserved`] fills that storage and [`Locator::publish`] publishes. Every
//! fallible allocation happens before retention, so no allocation occurs after the reserve-all
//! phase. An armed [`reservation_failure`] entry refusal or
//! [`reservation_failure::arm_late_edge`] edge refusal short-circuits the reserve-all phase, so the
//! committed links, payloads, counts and coverage are unchanged and there is no partial
//! publication.
//!
//! The engine stores both faces' residue records in one flat `Vec<u32>` arena with disjoint roots,
//! and the identity trie in flat identity-node and sibling-edge record arenas; every traversal is a
//! single-exit scalar `while` over indexed loads and stores. A cold validator sees copied
//! terminal/edge records rather than the arenas.
//!
//! # Bounds and memory
//!
//! For `n` live representatives and `S = sum over live k of |E(k)|`:
//!
//! * hyperbolic residue nodes `<= 1 + 40*n`; parabolic coefficient nodes `<= 1 + 64*n`; identity
//!   nodes `<= n + S`; edges `<= n + S`; payloads `= n`.
//! * a word lookup follows at most 40 ternary residue edges or 64 binary coefficient edges, then 8
//!   identity edges, inspects at most `8 * 256 = 2048` byte-edge labels and performs exactly one
//!   full-key comparison. These are design bounds, not measured latency.
//! * both faces share one residue arena of 4-word records; the identity trie is 2-word node records
//!   and 3-word sibling-edge records; a representative carries the retained identity (one word for
//!   a word key), the value, the face index and one link word. Allocator metadata is not counted.
//!   No `q`-sized allocation exists.

use crate::cell::Sigma;
use crate::coord::address::{ParabolicDual, Residue, H3_40};

#[path = "locator_core.rs"]
pub(crate) mod locator_core;

#[cfg(feature = "injection")]
pub use locator_core::reservation_failure;
pub use locator_core::{ExactIdentity, Refusal};

use locator_core::{Directory, ResidueIndex, StagedPath, Walk};

/// Number of ternary residue levels: `H3_40::K`.
const LEVELS: usize = H3_40::K as usize;
/// Number of binary coefficient levels: the 64 bits of a word.
const PARABOLIC_LEVELS: usize = u64::BITS as usize;
/// The largest per-face residue path length.
const MAX_LEVELS: usize = if LEVELS > PARABOLIC_LEVELS {
    LEVELS
} else {
    PARABOLIC_LEVELS
};
/// The largest number of byte-labelled edges one identity node may hold.
const MAX_LABELS: usize = 256;

/// Exact-identity instances for the small integer keys used by the regression tests.
///
/// Each width encodes as its fixed-width original-bit bytes, most-significant first, with no
/// truncation and no reduced residue: for an unsigned key `E(k) = k.to_be_bytes()`, and for a
/// signed key the same bytes on the two's-complement bit pattern `k.cast_unsigned()`. The retained
/// identity is that exact unsigned bit pattern, so equality on the retained value is exactly key
/// equality. The shared engine and the public trait path are unchanged; these instances only widen
/// the admitted key set.
macro_rules! exact_identity_for_integer {
    ($signed:ty, $unsigned:ty, $bytes:literal) => {
        impl ExactIdentity for $signed {
            type Retained = $unsigned;
            type Reserved = ();

            fn retain(&self) -> $unsigned {
                self.cast_unsigned()
            }

            fn reserve_retain(&self) -> Result<(), Refusal> {
                Ok(())
            }

            fn retain_reserved(&self, (): ()) -> $unsigned {
                self.cast_unsigned()
            }

            fn matches_retained(&self, retained: &$unsigned) -> bool {
                self.cast_unsigned() == *retained
            }

            fn encoding_len(&self) -> usize {
                $bytes
            }

            fn encoding_byte(&self, index: usize) -> u8 {
                self.cast_unsigned().to_be_bytes()[index]
            }

            fn retained_encoding_len(_: &$unsigned) -> usize {
                $bytes
            }

            fn retained_encoding_byte(retained: &$unsigned, index: usize) -> u8 {
                retained.to_be_bytes()[index]
            }
        }

        impl ExactIdentity for $unsigned {
            type Retained = $unsigned;
            type Reserved = ();

            fn retain(&self) -> $unsigned {
                *self
            }

            fn reserve_retain(&self) -> Result<(), Refusal> {
                Ok(())
            }

            fn retain_reserved(&self, (): ()) -> $unsigned {
                *self
            }

            fn matches_retained(&self, retained: &$unsigned) -> bool {
                *self == *retained
            }

            fn encoding_len(&self) -> usize {
                $bytes
            }

            fn encoding_byte(&self, index: usize) -> u8 {
                self.to_be_bytes()[index]
            }

            fn retained_encoding_len(_: &$unsigned) -> usize {
                $bytes
            }

            fn retained_encoding_byte(retained: &$unsigned, index: usize) -> u8 {
                retained.to_be_bytes()[index]
            }
        }
    };
}

exact_identity_for_integer!(i8, u8, 1);
exact_identity_for_integer!(i32, u32, 4);

/// The face-selected index a locator operation addresses.
///
/// The elliptic face has no locator: its exact address is the packing, not a trie, so
/// [`FaceIndex::Elliptic`] is refused. The parabolic and hyperbolic faces select a residue trie;
/// the exact identity is always the shared byte trie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceIndex {
    /// The elliptic face: exact packing, no locator.
    Elliptic,
    /// The parabolic face: the dual coefficient `a + b ε`, indexed by `a = fstHom`.
    Parabolic(ParabolicDual),
    /// The hyperbolic face: a canonical residue in `Z/q`.
    Hyperbolic(Residue),
}

impl FaceIndex {
    /// The face this index belongs to.
    #[must_use]
    pub const fn face(self) -> Sigma {
        match self {
            Self::Elliptic => Sigma::Elliptic,
            Self::Parabolic(_) => Sigma::Parabolic,
            Self::Hyperbolic(_) => Sigma::Hyperbolic,
        }
    }

    /// The number of residue levels this index's trie traverses.
    #[must_use]
    const fn levels(self) -> usize {
        match self {
            Self::Elliptic => 0,
            Self::Parabolic(_) => PARABOLIC_LEVELS,
            Self::Hyperbolic(_) => LEVELS,
        }
    }
}

/// The canonical base-3 digits of a residue, least-significant first, one per level.
///
/// Temporal O(40): exactly [`LEVELS`] remainder/division steps, independent of the residue.
/// Spatial O(1): one 40-byte array.
#[must_use]
pub fn residue_digits(residue: Residue) -> [u8; LEVELS] {
    let mut digits = [0u8; LEVELS];
    <Residue as ResidueIndex>::digits(&residue, &mut digits);
    digits
}

/// The canonical bits of a parabolic scalar coefficient, least-significant first.
///
/// Temporal O(64): exactly [`PARABOLIC_LEVELS`] bit reads, independent of the coefficient.
/// Spatial O(1): one 64-byte array.
#[must_use]
pub fn coefficient_bits(coefficient: u64) -> [u8; PARABOLIC_LEVELS] {
    let mut bits = [0u8; PARABOLIC_LEVELS];
    <ParabolicDual as ResidueIndex>::digits(&ParabolicDual::new(coefficient, 0), &mut bits);
    bits
}

/// The scope over which absence was certified by a complete traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub struct CoveredScope {
    index: FaceIndex,
    residue_levels: usize,
    identity_bytes: usize,
}

impl CoveredScope {
    /// The queried face index.
    #[must_use]
    pub const fn index(self) -> FaceIndex {
        self.index
    }

    /// The face the certificate was issued under.
    #[must_use]
    pub const fn face(self) -> Sigma {
        self.index.face()
    }

    /// The queried hyperbolic residue, `None` off the hyperbolic face.
    #[must_use]
    pub const fn residue(self) -> Option<Residue> {
        match self.index {
            FaceIndex::Hyperbolic(residue) => Some(residue),
            _ => None,
        }
    }

    /// The queried parabolic dual coefficient, `None` off the parabolic face.
    #[must_use]
    pub const fn coefficient(self) -> Option<ParabolicDual> {
        match self.index {
            FaceIndex::Parabolic(dual) => Some(dual),
            _ => None,
        }
    }

    /// The number of residue levels traversed before the missing edge.
    #[must_use]
    pub const fn residue_levels(self) -> usize {
        self.residue_levels
    }

    /// The number of identity bytes traversed before the missing edge.
    #[must_use]
    pub const fn identity_bytes(self) -> usize {
        self.identity_bytes
    }
}

/// The result of a borrowed read.
#[derive(Debug, PartialEq, Eq)]
#[must_use]
pub enum Lookup<'a, V> {
    /// A representative with a matching full identity is present.
    Present(&'a V),
    /// Absence is certified over the covered scope.
    Absent(CoveredScope),
    /// Absence cannot be certified, or the structure did not confirm the identity.
    Unknown,
}

/// The result of a dedicated absence query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum Absence {
    /// Absence is certified over the covered scope.
    Absent(CoveredScope),
    /// Absence cannot be certified. This includes a present key, for which the query is
    /// vacuously not an absence certificate.
    Unknown,
}

/// The outcome of a successful write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inserted {
    /// A new representative was published.
    Fresh,
    /// An existing representative's value was replaced.
    Updated,
}

/// What one traversal found.
enum WalkResult {
    Found(usize),
    StaleTerminal,
    Miss {
        residue_levels: usize,
        identity_bytes: usize,
    },
}

/// One live representative: the face index, the retained exact identity, the value and the
/// reverse link.
struct Representative<R, V> {
    index: FaceIndex,
    identity: R,
    value: V,
    identity_node: u32,
}

/// A sparse, face-indexed residue trie followed by a shared exact-identity byte trie.
pub struct Locator<K: ExactIdentity, V> {
    profile: H3_40,
    directory: Directory,
    reps: Vec<Representative<K::Retained, V>>,
    coverage_established: bool,
    max_reps: Option<usize>,
}

impl<K: ExactIdentity, V> Locator<K, V> {
    /// The empty locator: one residue root and one coefficient root, with no representatives.
    ///
    /// Empty construction establishes the coverage invariant.
    #[must_use]
    pub fn new(profile: H3_40) -> Self {
        Self {
            profile,
            directory: Directory::new(),
            reps: Vec::new(),
            coverage_established: true,
            max_reps: None,
        }
    }

    /// The empty locator with a checked representative limit.
    #[must_use]
    pub fn with_max_reps(profile: H3_40, max_reps: usize) -> Self {
        let mut locator = Self::new(profile);
        locator.max_reps = Some(max_reps);
        locator
    }

    /// The sealed profile.
    #[must_use]
    pub const fn profile(&self) -> H3_40 {
        self.profile
    }

    /// The number of live representatives.
    #[must_use]
    pub fn len(&self) -> usize {
        self.reps.len()
    }

    /// Whether there are no live representatives.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.reps.len() == 0
    }

    /// Whether the coverage invariant is currently asserted.
    #[must_use]
    pub const fn coverage_established(&self) -> bool {
        self.coverage_established
    }

    /// The number of occupied hyperbolic residue nodes.
    #[must_use]
    pub fn residue_node_count(&self) -> usize {
        self.directory.arena_len::<Residue>()
    }

    /// The number of occupied parabolic coefficient nodes.
    #[must_use]
    pub fn parabolic_node_count(&self) -> usize {
        self.directory.arena_len::<ParabolicDual>()
    }

    /// The number of identity nodes.
    #[must_use]
    pub fn identity_node_count(&self) -> usize {
        self.directory.identity_node_count()
    }

    /// The total number of byte-labelled identity edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.directory.edge_count()
    }

    /// The largest number of edges on one identity node, at most [`MAX_LABELS`].
    #[must_use]
    pub fn max_identity_fanout(&self) -> usize {
        self.directory.max_identity_fanout()
    }

    /// Marks the coverage invariant as not established.
    ///
    /// The empty constructor and every successful insert establish or preserve the invariant.
    /// This exists so a locator whose provenance is not trusted — for example one recovered from
    /// an untrusted snapshot — withholds absence certificates. It never changes a stored key or
    /// value, and it is safe and deterministic.
    pub fn invalidate_coverage(&mut self) {
        self.coverage_established = false;
    }

    /// Structurally verifies the coverage invariant.
    ///
    /// Checks the sorted, duplicate-free, bounded edge lists; the reverse terminal link of every
    /// representative; and that each representative's complete face residue and identity paths
    /// reach its own terminal. Temporal O(n * (64 + |E|)); no allocation.
    #[must_use]
    pub fn check_coverage(&self) -> bool {
        for (index, node) in self.directory.identity_nodes().iter().enumerate() {
            if node.edges.len() > MAX_LABELS {
                return false;
            }
            if node
                .edges
                .windows(2)
                .any(|pair| pair[0].label >= pair[1].label)
            {
                return false;
            }
            if let Some(terminal) = node.terminal {
                let Some(rep) = self.reps.get(terminal as usize) else {
                    return false;
                };
                let expected = if index > u32::MAX as usize {
                    u32::MAX
                } else {
                    index as u32
                };
                if rep.identity_node != expected {
                    return false;
                }
            }
        }
        for (index, rep) in self.reps.iter().enumerate() {
            let Some(idnode) = self.representative_terminal(rep) else {
                return false;
            };
            let expected = if index > u32::MAX as usize {
                u32::MAX
            } else {
                index as u32
            };
            let Some(node) = self.directory.identity_nodes().get(idnode as usize) else {
                return false;
            };
            if node.terminal != Some(expected) {
                return false;
            }
            if rep.identity_node != idnode {
                return false;
            }
        }
        true
    }

    /// Inserts `key` with `value` at the semantic face index, replacing an equal key's value.
    ///
    /// # Errors
    ///
    /// Returns [`Refusal::WrongFace`] on the elliptic face, [`Refusal::Capacity`] when the
    /// configured representative limit is reached, [`Refusal::SizeOverflow`] when a checked count
    /// or handle bound does not fit, [`Refusal::AllocationFailed`] when a reservation cannot be
    /// obtained, and [`Refusal::Uncovered`] when the coverage invariant is not established. Every
    /// error leaves the committed state unchanged and publishes nothing.
    pub fn insert(&mut self, index: FaceIndex, key: &K, value: V) -> Result<Inserted, Refusal> {
        if matches!(index, FaceIndex::Elliptic) {
            return Err(Refusal::WrongFace);
        }
        match self.walk(index, key) {
            WalkResult::Found(index) => {
                self.reps[index].value = value;
                return Ok(Inserted::Updated);
            }
            WalkResult::StaleTerminal => return Err(Refusal::Uncovered),
            WalkResult::Miss { .. } => {}
        }
        if let Some(max) = self.max_reps {
            if self.reps.len() >= max {
                return Err(Refusal::Capacity);
            }
        }
        let (reserved, staged) = self.reserve_for(index, key)?;
        let retained = key.retain_reserved(reserved);
        if self.reps.len() > u32::MAX as usize {
            // Refusal after staging: the scratch returns to the idle slot with nothing published.
            self.directory.recycle_staged(staged);
            return Err(Refusal::SizeOverflow);
        }
        let handle = self.reps.len() as u32;
        let identity_node = staged.identity_path[key.encoding_len()];
        // Publish the complete path first, then commit the payload the terminal now names.
        // Publication consumes the staged path and recycles its scratch buffer.
        self.publish(index, key, staged, handle);
        self.reps.push(Representative {
            index,
            identity: retained,
            value,
            identity_node,
        });
        Ok(Inserted::Fresh)
    }

    /// Borrows the value for an exactly matching key.
    pub fn get(&self, index: FaceIndex, key: &K) -> Lookup<'_, V> {
        if matches!(index, FaceIndex::Elliptic) {
            return Lookup::Unknown;
        }
        match self.walk(index, key) {
            WalkResult::Found(index) => Lookup::Present(&self.reps[index].value),
            WalkResult::StaleTerminal => Lookup::Unknown,
            WalkResult::Miss {
                residue_levels,
                identity_bytes,
            } => match self.certify_absent(index, residue_levels, identity_bytes) {
                Absence::Absent(scope) => Lookup::Absent(scope),
                Absence::Unknown => Lookup::Unknown,
            },
        }
    }

    /// Answers whether the key is absent, returning a certificate only under coverage.
    pub fn lookup_absent(&self, index: FaceIndex, key: &K) -> Absence {
        if matches!(index, FaceIndex::Elliptic) {
            return Absence::Unknown;
        }
        match self.walk(index, key) {
            WalkResult::Found(_) | WalkResult::StaleTerminal => Absence::Unknown,
            WalkResult::Miss {
                residue_levels,
                identity_bytes,
            } => self.certify_absent(index, residue_levels, identity_bytes),
        }
    }

    /// The complete face residue and identity traversal, read-only.
    ///
    /// The shared engine returns a terminal handle; the face and full-key identity are confirmed
    /// here against the representative that handle names.
    fn walk(&self, index: FaceIndex, key: &K) -> WalkResult {
        let walked = match index {
            FaceIndex::Elliptic => {
                return WalkResult::Miss {
                    residue_levels: 0,
                    identity_bytes: 0,
                };
            }
            FaceIndex::Parabolic(dual) => self.directory.walk(&dual, key),
            FaceIndex::Hyperbolic(residue) => self.directory.walk(&residue, key),
        };
        match walked {
            Walk::Terminal(handle) => {
                let representative = &self.reps[handle as usize];
                if representative.index.face() == index.face()
                    && key.matches_retained(&representative.identity)
                {
                    WalkResult::Found(handle as usize)
                } else {
                    WalkResult::StaleTerminal
                }
            }
            Walk::Miss {
                residue_levels,
                identity_bytes,
            } => WalkResult::Miss {
                residue_levels,
                identity_bytes,
            },
        }
    }

    /// The terminal identity node a retained representative reaches, or `None` if a path edge is
    /// missing.
    fn representative_terminal(&self, rep: &Representative<K::Retained, V>) -> Option<u32> {
        match rep.index {
            FaceIndex::Elliptic => None,
            FaceIndex::Parabolic(dual) => self
                .directory
                .representative_terminal::<ParabolicDual, K>(&dual, &rep.identity),
            FaceIndex::Hyperbolic(residue) => self
                .directory
                .representative_terminal::<Residue, K>(&residue, &rep.identity),
        }
    }

    /// Turns a miss into a certificate only when coverage is established.
    fn certify_absent(
        &self,
        index: FaceIndex,
        residue_levels: usize,
        identity_bytes: usize,
    ) -> Absence {
        if !self.coverage_established {
            return Absence::Unknown;
        }
        // An identity miss is a certificate only if the canonical face residue path completed.
        if identity_bytes > 0 && residue_levels != index.levels() {
            return Absence::Unknown;
        }
        Absence::Absent(CoveredScope {
            index,
            residue_levels,
            identity_bytes,
        })
    }

    /// Checks count, byte-size and handle bounds and reserves all new storage.
    ///
    /// The representative arena is reserved here; the shared engine reserves the residue,
    /// coefficient and identity arenas, stages the complete path and reserves the retained
    /// identity. An armed [`reservation_failure`] entry refusal is consumed before any bound check
    /// or reservation; an armed [`reservation_failure::arm_late_edge`] refusal fires at the
    /// identity-edge reservation inside the engine. Both arms are one-shot.
    fn reserve_for(
        &mut self,
        index: FaceIndex,
        key: &K,
    ) -> Result<(K::Reserved, StagedPath), Refusal> {
        #[cfg(feature = "injection")]
        if let Some(refusal) = reservation_failure::consume() {
            return Err(refusal);
        }
        let rep_end = self
            .reps
            .len()
            .checked_add(1)
            .ok_or(Refusal::SizeOverflow)?;
        if rep_end > u32::MAX as usize {
            return Err(Refusal::SizeOverflow);
        }
        if self.reps.try_reserve(1).is_err() {
            return Err(Refusal::AllocationFailed);
        }
        match index {
            FaceIndex::Elliptic => Err(Refusal::WrongFace),
            FaceIndex::Parabolic(dual) => self.directory.reserve_for(&dual, key),
            FaceIndex::Hyperbolic(residue) => self.directory.reserve_for(&residue, key),
        }
    }

    /// Publishes the staged path into the face's residue arena and the shared identity trie,
    /// consuming the staged path and recycling its scratch buffer.
    fn publish(&mut self, index: FaceIndex, key: &K, staged: StagedPath, handle: u32) {
        match index {
            // Unreachable (`insert` refuses the elliptic face before staging); the defensive arm
            // still returns the scratch to the idle slot rather than dropping it.
            FaceIndex::Elliptic => self.directory.recycle_staged(staged),
            FaceIndex::Parabolic(dual) => self.directory.link(&dual, key, staged, handle),
            FaceIndex::Hyperbolic(residue) => self.directory.link(&residue, key, staged, handle),
        }
    }
}
