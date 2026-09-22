//! Fallible vector reservations and the joint prepared-resource handoff.
//!
//! The handoff types below are the unextracted adapter's envelope: a checked plan, its
//! capacity-reserved (then materialized) storage, the observed capacity receipt and the pending
//! held carrier. [`publish_prepared`] is the nonallocating publication step production consumes
//! before relocation; [`capacity_bound_holds`] is the separately rooted scalar comparison.

use super::{RebuildPlan, RebuildReserved, StorageError, SHARDS};

/// A checked target length beside the capacity the successful reservation observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CapacityBound {
    /// The checked target vector length.
    pub(crate) required_len: usize,
    /// The real `Vec::capacity()` observed after reserve-all and before initialization.
    pub(crate) observed_capacity: usize,
}

/// Per-region tag, payload and summary capacity bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RegionReservation {
    /// The region's tag-vector bound.
    pub(crate) tags: CapacityBound,
    /// The region's payload-vector bound.
    pub(crate) payloads: CapacityBound,
    /// The region's occupancy-summary bound; zero for an empty region.
    pub(crate) summary: CapacityBound,
}

/// The fixed-size capacity observations for one reserved rebuild.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReservationReceipt {
    /// The flat resident-area bound.
    pub(crate) flat: CapacityBound,
    /// The 256-entry region-directory bound.
    pub(crate) directory: CapacityBound,
    /// One tag/payload bound pair per physical shard.
    pub(crate) regions: [RegionReservation; SHARDS],
}

/// The checked locator reservation targets for one staged directory path.
///
/// The eight fields mirror `Apeiron.NativeLocator.LocatorReservationReceipt` in
/// `proof/corpus/Native/LocatorSpec.lean`: the binary and ternary residue nodes, the identity nodes
/// and byte-labelled edges, the representative handles, the reverse links, the staged path entries
/// and the retained-identity bytes. Every target is computed with checked arithmetic before any
/// reservation, so a sizing overflow is refused before an allocation is attempted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LocatorReservationPlan {
    /// Binary (parabolic) residue nodes including the arena root.
    pub(crate) parabolic_nodes: usize,
    /// Ternary (hyperbolic) residue nodes including the arena root.
    pub(crate) residue_nodes: usize,
    /// Exact-identity byte-trie nodes, one per encoding byte plus the root.
    pub(crate) identity_nodes: usize,
    /// Byte-labelled identity edge slots, one per new identity node.
    pub(crate) identity_edges: usize,
    /// Representative handles named by the reverse-link registry.
    pub(crate) representatives: usize,
    /// Reverse-link entries `(identity_node, payload_location)`.
    pub(crate) reverse_links: usize,
    /// Staged identity-path entries, one per encoding byte plus the root.
    pub(crate) staged_path_entries: usize,
    /// Bytes the retained exact identity reserves.
    pub(crate) retained_identity_bytes: usize,
}

/// The observed locator capacity bounds for one reserved directory path.
///
/// The observed value is the real `Vec::capacity()` where the storage exposes one: the reverse
/// links, the staged path entries and each identity node's edge vector. The residue and identity
/// node counts are the post-reservation element counts; a `Vec`'s capacity is never below its
/// length, so the bound is a sound (conservative) witness. The retained-identity buffer is opaque
/// (`ExactIdentity::Reserved`); its reservation is the engine's `reserve_retain`, so that bound
/// records the checked encoding-length target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LocatorReservationReceipt {
    /// Binary residue-node bound.
    pub(crate) parabolic_nodes: CapacityBound,
    /// Ternary residue-node bound.
    pub(crate) residue_nodes: CapacityBound,
    /// Identity-node bound.
    pub(crate) identity_nodes: CapacityBound,
    /// Identity-edge bound over every node's edge capacity.
    pub(crate) identity_edges: CapacityBound,
    /// Representative-handle bound.
    pub(crate) representatives: CapacityBound,
    /// Reverse-link registry bound.
    pub(crate) reverse_links: CapacityBound,
    /// Staged identity-path bound.
    pub(crate) staging: CapacityBound,
    /// Retained-identity bound.
    pub(crate) retained_identity: CapacityBound,
}

impl LocatorReservationReceipt {
    /// The allocation-free empty receipt: no locator storage was reserved.
    pub(crate) const EMPTY: Self = Self {
        parabolic_nodes: CapacityBound {
            required_len: 0,
            observed_capacity: 0,
        },
        residue_nodes: CapacityBound {
            required_len: 0,
            observed_capacity: 0,
        },
        identity_nodes: CapacityBound {
            required_len: 0,
            observed_capacity: 0,
        },
        identity_edges: CapacityBound {
            required_len: 0,
            observed_capacity: 0,
        },
        representatives: CapacityBound {
            required_len: 0,
            observed_capacity: 0,
        },
        reverse_links: CapacityBound {
            required_len: 0,
            observed_capacity: 0,
        },
        staging: CapacityBound {
            required_len: 0,
            observed_capacity: 0,
        },
        retained_identity: CapacityBound {
            required_len: 0,
            observed_capacity: 0,
        },
    };

    /// Reports whether every observed bound meets its checked target.
    ///
    /// The receipt is the refusal witness: a bound that does not hold is a checked refusal, not a
    /// panic, so the transaction can stutter before publishing the staged destination.
    #[must_use]
    pub(crate) fn holds(&self) -> bool {
        capacity_bound_holds(
            self.parabolic_nodes.required_len,
            self.parabolic_nodes.observed_capacity,
        ) && capacity_bound_holds(
            self.residue_nodes.required_len,
            self.residue_nodes.observed_capacity,
        ) && capacity_bound_holds(
            self.identity_nodes.required_len,
            self.identity_nodes.observed_capacity,
        ) && capacity_bound_holds(
            self.identity_edges.required_len,
            self.identity_edges.observed_capacity,
        ) && capacity_bound_holds(
            self.representatives.required_len,
            self.representatives.observed_capacity,
        ) && capacity_bound_holds(
            self.reverse_links.required_len,
            self.reverse_links.observed_capacity,
        ) && capacity_bound_holds(self.staging.required_len, self.staging.observed_capacity)
            && capacity_bound_holds(
                self.retained_identity.required_len,
                self.retained_identity.observed_capacity,
            )
    }
}

/// Adapter-produced resource data: the table and locator receipts and the pending held carrier.
pub(crate) struct PreparedOutcome<R> {
    /// The observed capacity bounds for the materialized table storage.
    pub(crate) reservation: ReservationReceipt,
    /// The observed capacity bounds for the jointly reserved locator path.
    pub(crate) locator: LocatorReservationReceipt,
    /// The pending key's held carrier, built exactly once by the adapter.
    pub(crate) pending_held: R,
}

/// The adapter's exact success tuple: checked plan, materialized storage and resource outcome.
pub(crate) type PreparedEnvelope<R, V> = (RebuildPlan, RebuildReserved<R, V>, PreparedOutcome<R>);

/// The published adapter result consumed by the relocation/commit core.
#[allow(
    clippy::large_enum_variant,
    reason = "the prepared envelope is moved once per rebuild; boxing its storage or outcome would add an allocation the resource boundary does not authorize"
)]
pub(crate) enum PreparedResult<R, V> {
    /// The adapter completed planning, reservation, capacity checking and initialization.
    Prepared {
        /// The checked plan.
        plan: RebuildPlan,
        /// The materialized reserved storage.
        storage: RebuildReserved<R, V>,
        /// The receipt and pending carrier.
        outcome: PreparedOutcome<R>,
    },
    /// A checked sizing or reservation failure; no plan, storage or carrier is exposed.
    Refused(StorageError),
}

/// Reserves `additional` items, mapping an allocator refusal to [`StorageError::AllocationFailed`].
pub(crate) fn try_reserve_exact<T>(
    items: &mut Vec<T>,
    additional: usize,
) -> Result<(), StorageError> {
    items
        .try_reserve_exact(additional)
        .map_err(|_| StorageError::AllocationFailed)
}

/// The production scalar capacity comparison: the observed capacity meets the checked target.
pub(crate) fn capacity_bound_holds(required_len: usize, observed_capacity: usize) -> bool {
    required_len <= observed_capacity
}

/// Nonallocating publication: moves the adapter's exact success tuple into [`PreparedResult`], or
/// maps `Err(e)` to [`PreparedResult::Refused`].
///
/// It does not plan, reserve, read capacity, initialize or certify a predicate.  The error arm
/// carries no plan, storage, receipt or held carrier.
pub(crate) fn publish_prepared<R, V>(
    result: Result<PreparedEnvelope<R, V>, StorageError>,
) -> PreparedResult<R, V> {
    match result {
        Ok((plan, storage, outcome)) => PreparedResult::Prepared {
            plan,
            storage,
            outcome,
        },
        Err(error) => PreparedResult::Refused(error),
    }
}
