// Handle-only directory staging and the table storage-coverage checker.
//
// Included textually by `include!("table/locator.rs")` from `src/table.rs`, so these items are
// definitions of the `table` module and read the table's private storage. The file reuses the one
// shared directory engine in `crate::coord::locator::locator_core`; it builds no second trie, and
// the directory owns no key and no value. No read path consumes the directory here.

/// The largest number of byte-labelled edges one identity node may hold.
///
/// This is the engine's design bound; the checker rejects a directory that exceeds it.
const MAX_IDENTITY_LABELS: usize = 256;

impl<K, V, H, const N: usize, R> Table<K, V, H, N, R>
where
    K: ExactIdentity,
    H: Home<K>,
    R: KeyRep<K, H>,
{
    /// Structurally checks table storage coverage `C`.
    ///
    /// Returns `true` exactly when the attached directory covers every live table payload: each
    /// live payload's complete residue and exact-identity path ends at that payload's own terminal,
    /// each terminal points back to the live payload that owns it, path edges and terminals are
    /// unique, and no intermediate staging is outstanding. A `false` result is not an absence
    /// certificate; it means the coverage invariant is not established, so no absence may be issued
    /// and no read may consume the directory.
    ///
    /// Temporal `O(slots + live * (64 + |E|) + edges)`: one pass over the table's storage, one
    /// directory walk per live payload and one pass over the identity nodes, counted in slots, key
    /// bytes and edges. Spatial `O(live + edges)` for the location and reverse-link working sets.
    /// This is a cold diagnostic: it materializes each payload's key through the carrier, so it may
    /// clone (`Stored`) or decode (`Regen`) and is not on any read or write path.
    #[must_use]
    pub fn check_storage_coverage(&self) -> bool {
        // Unlinked intermediate staging can issue no certificate.
        if self.locator.staged.is_some() {
            return false;
        }
        let Some(directory) = &self.locator.directory else {
            // The allocation-free empty state covers exactly the empty table.
            return self.live_locations().len() == 0;
        };
        // Structural: bounded, sorted, duplicate-free identity edges.
        for node in directory.identity_nodes() {
            if node.edges.len() > MAX_IDENTITY_LABELS {
                return false;
            }
            if node
                .edges
                .windows(2)
                .any(|pair| pair[0].label >= pair[1].label)
            {
                return false;
            }
        }
        // One reverse link per live payload, with distinct locations.
        let live = self.live_locations();
        if self.locator.links.len() != live.len() {
            return false;
        }
        let mut linked: Vec<PayloadLocation> = self
            .locator
            .links
            .iter()
            .map(|(_, location)| *location)
            .collect();
        linked.sort_unstable();
        linked.dedup();
        if linked.len() != live.len() {
            return false;
        }
        // Forward: every link names a live payload whose complete path reaches its own terminal.
        for (handle, (identity_node, location)) in self.locator.links.iter().enumerate() {
            let Some((sigma, carrier)) = self.payload_at(*location) else {
                return false;
            };
            let key = R::key_owned(sigma, carrier);
            let walked = match sigma {
                // The elliptic face has no locator; an elliptic payload cannot be covered here.
                Sigma::Elliptic => return false,
                Sigma::Parabolic => {
                    let index = self.home.parabolic_dual(&key);
                    directory.walk::<ParabolicDual, K>(&index, &key)
                }
                Sigma::Hyperbolic => {
                    let Some(profile) = self.locator.profile else {
                        return false;
                    };
                    let index = self.home.hyperbolic_scalar(profile, &key);
                    directory.walk::<Residue, K>(&index, &key)
                }
            };
            if handle > u32::MAX as usize {
                return false;
            }
            let expected = handle as u32;
            if !matches!(walked, Walk::Terminal(found) if found == expected) {
                return false;
            }
            let node_index = *identity_node as usize;
            let Some(node) = directory.identity_nodes().get(node_index) else {
                return false;
            };
            if node.terminal != Some(expected) {
                return false;
            }
        }
        // Reverse: every terminal names its own reverse link, so terminals are unique.
        for (index, node) in directory.identity_nodes().iter().enumerate() {
            if let Some(handle) = node.terminal {
                let handle = handle as usize;
                let Some((identity_node, _)) = self.locator.links.get(handle) else {
                    return false;
                };
                if index > u32::MAX as usize || index as u32 != *identity_node {
                    return false;
                }
            }
        }
        true
    }

    /// The live payload locations in resident-then-spill storage order.
    ///
    /// Temporal `O(slots)`; spatial `O(live)`.
    fn live_locations(&self) -> Vec<PayloadLocation> {
        let mut locations = Vec::new();
        for (index, slot) in self.resident.slots.iter().enumerate() {
            if slot.live {
                locations.push(PayloadLocation::Resident(index));
            }
        }
        for (shard, region) in self.shards.iter().enumerate() {
            for (index, tag) in region.tags.iter().enumerate() {
                if *tag != 0 {
                    locations.push(PayloadLocation::Spill(shard, index));
                }
            }
        }
        locations
    }

    /// Resolves a location to its stored face and held carrier, or `None` when it is not live.
    fn payload_at(&self, location: PayloadLocation) -> Option<(Sigma, &R)> {
        match location {
            PayloadLocation::Resident(index) => {
                let slot = match self.resident.slots.get(index) {
                    Some(slot) => slot,
                    None => return None,
                };
                slot.live.then_some((slot.sigma, &slot.held))
            }
            PayloadLocation::Spill(shard, index) => {
                let region = match self.shards.get(shard) {
                    Some(region) => region,
                    None => return None,
                };
                let tag = match region.tags.get(index) {
                    Some(tag) => *tag,
                    None => return None,
                };
                if tag == 0 {
                    return None;
                }
                let slot = match region.slots.get(index) {
                    Some(slot) => slot,
                    None => return None,
                };
                Some((slot.0, &slot.1))
            }
        }
    }
}

/// The checked reservation plan for one complete path in a directory.
///
/// This is a pure sizing function: it reserves nothing and reads no directory. The eight targets
/// mirror `Apeiron.NativeLocator.LocatorReservationReceipt`. Only the selected face's residue arena
/// grows; the other arena keeps its single fresh root. The identity targets count one node per
/// encoding byte plus the root, one edge per new identity node and the encoding length plus one
/// staged path entries. Every arithmetic step is checked, so a sizing overflow is refused before
/// any reservation.
fn locator_path_plan<I: ResidueIndex>(
    encoding_len: usize,
) -> Result<LocatorReservationPlan, StorageError> {
    let identity_entries = encoding_len
        .checked_add(1)
        .ok_or(StorageError::SizeOverflow)?;
    let levels = I::LEVELS;
    // A fresh `Directory` has one root node in each residue arena; only `I`'s arena grows.
    let (residue_nodes, parabolic_nodes) = if I::ARENA == <Residue as ResidueIndex>::ARENA {
        (
            1usize
                .checked_add(levels)
                .ok_or(StorageError::SizeOverflow)?,
            1usize,
        )
    } else {
        (
            1usize,
            1usize
                .checked_add(levels)
                .ok_or(StorageError::SizeOverflow)?,
        )
    };
    Ok(LocatorReservationPlan {
        parabolic_nodes,
        residue_nodes,
        identity_nodes: identity_entries,
        identity_edges: identity_entries,
        representatives: 1,
        reverse_links: 1,
        staged_path_entries: identity_entries,
        retained_identity_bytes: encoding_len,
    })
}

/// Observes the locator capacity receipt for a reserved path.
///
/// Every field pairs the checked plan target with what the reservation actually observed: the
/// residue and identity node counts, the summed identity-edge capacity, the reverse-link registry
/// capacity, the staged identity-path capacity and the retained encoding length. A `Vec`'s capacity
/// is never below its length, so the node-count bounds are sound. The receipt's
/// [`LocatorReservationReceipt::holds`] is the refusal witness.
fn observe_locator_receipt(
    plan: &LocatorReservationPlan,
    staged: &StagedPath,
    links_capacity: usize,
    directory: &Directory,
) -> LocatorReservationReceipt {
    let mut edge_capacity = 0usize;
    for node in directory.identity_nodes() {
        edge_capacity = edge_capacity.saturating_add(node.edges.capacity());
    }
    LocatorReservationReceipt {
        parabolic_nodes: locator_bound(plan.parabolic_nodes, directory.arena_len::<ParabolicDual>()),
        residue_nodes: locator_bound(plan.residue_nodes, directory.arena_len::<Residue>()),
        identity_nodes: locator_bound(plan.identity_nodes, directory.identity_node_count()),
        identity_edges: locator_bound(plan.identity_edges, edge_capacity),
        representatives: locator_bound(plan.representatives, links_capacity),
        reverse_links: locator_bound(plan.reverse_links, links_capacity),
        staging: locator_bound(plan.staged_path_entries, staged.identity_path.capacity()),
        retained_identity: locator_bound(
            plan.retained_identity_bytes,
            plan.retained_identity_bytes,
        ),
    }
}

/// Maps a locator refusal to the table's existing resource error.
///
/// A positive-capacity `ShardFull` is never invented: the locator has no capacity error that maps
/// to it, so capacity, face and coverage refusals take the allocation error.
fn locator_refusal(refusal: Refusal) -> StorageError {
    match refusal {
        Refusal::AllocationFailed => StorageError::AllocationFailed,
        Refusal::SizeOverflow => StorageError::SizeOverflow,
        Refusal::Capacity | Refusal::WrongFace | Refusal::Uncovered => {
            StorageError::AllocationFailed
        }
    }
}

/// Pairs a checked target with the capacity the reservation observed.
fn locator_bound(required_len: usize, observed_capacity: usize) -> CapacityBound {
    CapacityBound {
        required_len,
        observed_capacity,
    }
}

/// A borrowed query key viewed as its carrier's exact-identity bytes, without a copy.
///
/// The directory's `link` and `walk` read only `encoding_len`/`encoding_byte`; this view forwards
/// those to the carrier, so publication never allocates a second key buffer. Its equality is
/// byte-wise over the exact identity encoding, which the directory never consults, so the write
/// path needs no `K: Eq` bound. The retention methods are unreachable for a view and are never
/// called.
struct QueryIdentity<'a, K, H, R> {
    key: &'a K,
    // A plain tuple phantom, not `fn() -> (H, R)`: the arrow type is an Aeneas-unsupported shape,
    // and the view is a transient local, so the marker carries no ownership or drop obligation.
    _carrier: PhantomData<(H, R)>,
}

/// Byte-wise equality of two query keys' exact identity encodings.
///
/// This is a free function, not the `PartialEq::eq` body: a loop written directly in a generic
/// `PartialEq` impl is an Aeneas-internal failure, while the same loop in a free function
/// translates.
fn query_identity_bytes_equal<K, H, R: KeyRep<K, H>>(left: &K, right: &K) -> bool {
    let len = R::query_encoding_len(left);
    if len != R::query_encoding_len(right) {
        return false;
    }
    let mut index = 0;
    while index < len {
        if R::query_encoding_byte(left, index) != R::query_encoding_byte(right, index) {
            return false;
        }
        index += 1;
    }
    true
}

impl<K, H, R> PartialEq for QueryIdentity<'_, K, H, R>
where
    R: KeyRep<K, H>,
{
    /// Byte-wise equality of the two keys' exact identity encodings.
    ///
    /// The directory never consults this equality; it exists only to satisfy [`ExactIdentity`]'s
    /// `Eq` supertrait. `ExactIdentity`'s law makes the encoding injective, so this agrees with
    /// `K`'s own equality while requiring no `K: Eq` bound on the write path.
    fn eq(&self, other: &Self) -> bool {
        query_identity_bytes_equal::<K, H, R>(self.key, other.key)
    }
}
impl<K, H, R> Eq for QueryIdentity<'_, K, H, R> where R: KeyRep<K, H> {}

impl<K, H, R> ExactIdentity for QueryIdentity<'_, K, H, R>
where
    R: KeyRep<K, H>,
{
    type Retained = ();
    type Reserved = ();

    fn retain(&self) {}
    fn reserve_retain(&self) -> Result<(), Refusal> {
        Ok(())
    }
    fn retain_reserved(&self, (): ()) {}
    #[allow(
        clippy::ignored_unit_patterns,
        reason = "the view's unit retained type is the trait's required signature; it is never compared"
    )]
    fn matches_retained(&self, _: &()) -> bool {
        false
    }
    fn encoding_len(&self) -> usize {
        R::query_encoding_len(self.key)
    }
    fn encoding_byte(&self, index: usize) -> u8 {
        R::query_encoding_byte(self.key, index)
    }
    #[allow(
        clippy::ignored_unit_patterns,
        reason = "the view's unit retained type is the trait's required signature; it is never read"
    )]
    fn retained_encoding_len(_: &()) -> usize {
        0
    }
    #[allow(
        clippy::ignored_unit_patterns,
        reason = "the view's unit retained type is the trait's required signature; it is never read"
    )]
    fn retained_encoding_byte(_: &(), _: usize) -> u8 {
        0
    }
}

/// Reports whether two payload locations name the same storage cell.
///
/// A scalar variant comparison rather than `PartialEq` on [`PayloadLocation`]: the derived
/// structural equality compares `&usize` fields through the standard library's
/// `PartialEq<&usize> for &usize`, whose signature the extraction leaves unmodeled, while this
/// compares copied `usize` values directly. The comparison is exactly the derived one.
fn same_location(left: PayloadLocation, right: PayloadLocation) -> bool {
    match (left, right) {
        (PayloadLocation::Resident(left), PayloadLocation::Resident(right)) => left == right,
        (
            PayloadLocation::Spill(left_shard, left_index),
            PayloadLocation::Spill(right_shard, right_index),
        ) => left_shard == right_shard && left_index == right_index,
        _ => false,
    }
}

impl TableLocator {
    /// The frozen hyperbolic profile, initialized on first use.
    fn profile_or_init(&mut self) -> Result<H3_40, StorageError> {
        if let Some(profile) = self.profile {
            return Ok(profile);
        }
        let profile = H3_40::new().ok_or(StorageError::SizeOverflow)?;
        self.profile = Some(profile);
        Ok(profile)
    }

    /// Reserves and stages the pending key's complete directory path, privately.
    ///
    /// The only admitted order is `checked sizing -> reserve the reverse-link registry -> reserve
    /// every binary/ternary residue node, identity node, identity edge, staged path entry and
    /// retained-identity buffer -> observe and check the receipt -> stage`. The engine's
    /// `reserve_for` performs the node/edge/staging/retained reservation and rolls its arenas back
    /// on failure. A refusal at any step leaves the committed directory, reverse links, payloads
    /// and counts unchanged, publishes nothing and leaves no staging behind.
    ///
    /// The path is reserved on the committed directory, before any payload effect, and is published
    /// by the write or commit path through [`TableLocator::publish_for_query`] or
    /// [`TableLocator::publish_for_held`]. While it is staged, `check_storage_coverage` issues no
    /// certificate.
    fn reserve_path<K, H, R>(
        &mut self,
        home: &H,
        sigma: Sigma,
        key: &K,
    ) -> Result<LocatorReservationReceipt, StorageError>
    where
        K: Clone,
        H: Home<K>,
        R: KeyRep<K, H>,
    {
        // The locator-free elliptic face reserves nothing.
        if matches!(sigma, Sigma::Elliptic) {
            return Ok(LocatorReservationReceipt::EMPTY);
        }
        let encoding_len = R::query_encoding_len(key);
        let mut bytes: Vec<u8> = Vec::new();
        try_reserve_exact(&mut bytes, encoding_len)?;
        let mut i = 0;
        while i < encoding_len {
            bytes.push(R::query_encoding_byte(key, i));
            i += 1;
        }
        // The allocation-free empty state materializes on its first locator reservation.
        if self.directory.is_none() {
            self.directory = Some(Directory::new());
        }
        if self.links.len() > u32::MAX as usize {
            return Err(StorageError::SizeOverflow);
        }
        let handle = self.links.len() as u32;
        self.links
            .try_reserve(1)
            .map_err(|_| StorageError::AllocationFailed)?;
        let links_capacity = self.links.capacity();
        match sigma {
            // Already returned above: the elliptic face has no locator.
            Sigma::Elliptic => Ok(LocatorReservationReceipt::EMPTY),
            Sigma::Parabolic => {
                let index = home.parabolic_dual(key);
                self.reserve_bytes::<ParabolicDual>(&index, &bytes, links_capacity, handle)
            }
            Sigma::Hyperbolic => {
                let profile = self.profile_or_init()?;
                let index = home.hyperbolic_scalar(profile, key);
                self.reserve_bytes::<Residue>(&index, &bytes, links_capacity, handle)
            }
        }
    }

    /// Reserves and stages one exact byte key's complete path on the committed directory.
    fn reserve_bytes<I: ResidueIndex>(
        &mut self,
        index: &I,
        bytes: &[u8],
        links_capacity: usize,
        handle: u32,
    ) -> Result<LocatorReservationReceipt, StorageError> {
        let plan = locator_path_plan::<I>(bytes.len())?;
        let mut key: Vec<u8> = Vec::new();
        try_reserve_exact(&mut key, bytes.len())?;
        key.extend_from_slice(bytes);
        // Direct reborrow of the owned `Option<Directory>` field: the branch-local binding is a
        // plain `&mut Directory`, so no `Result<&mut Directory, _>` or its `?`-desugared
        // `ControlFlow` enters the translated body. `reserve_path` materializes the directory
        // before this call, so the `None` arm is the same defensive refusal as before.
        let Some(directory) = &mut self.directory else {
            return Err(StorageError::AllocationFailed);
        };
        let (retained, staged) = directory
            .reserve_for::<I, Vec<u8>>(index, &key)
            .map_err(locator_refusal)?;
        let receipt = observe_locator_receipt(&plan, &staged, links_capacity, directory);
        if !receipt.holds() {
            return Err(StorageError::AllocationFailed);
        }
        // The retained identity buffer is reserved and released: the table's own carrier is the
        // authoritative key store, so the directory owns no second copy. `let _ =` drops it without
        // the `core::mem::drop` external, which the extraction leaves unmodeled.
        let _ = retained;
        self.staged = Some((handle, staged));
        Ok(receipt)
    }

    /// Publishes the staged path as the live reverse link at `location`, if one is staged.
    ///
    /// The payload effect has already run when this is called; the directory's `link` performs no
    /// fallible allocation because every node and edge was reserved before retention.
    fn publish_for_query<K, H, R>(
        &mut self,
        home: &H,
        sigma: Sigma,
        key: &K,
        location: PayloadLocation,
    ) where
        K: Clone,
        H: Home<K>,
        R: KeyRep<K, H>,
    {
        let Some((handle, staged)) = self.staged.take() else {
            return;
        };
        let encoding_len = R::query_encoding_len(key);
        match sigma {
            Sigma::Elliptic => {}
            Sigma::Parabolic => {
                let index = home.parabolic_dual(key);
                let view = QueryIdentity::<K, H, R> {
                    key,
                    _carrier: PhantomData,
                };
                // Reborrow the owned field directly: the binding is `&mut Directory`, not the
                // `Option<&mut Directory>` that `Option::as_mut` would form.
                if let Some(directory) = &mut self.directory {
                    directory.link::<ParabolicDual, _>(&index, &view, &staged, handle);
                }
            }
            Sigma::Hyperbolic => {
                let Some(profile) = self.profile else {
                    return;
                };
                let index = home.hyperbolic_scalar(profile, key);
                let view = QueryIdentity::<K, H, R> {
                    key,
                    _carrier: PhantomData,
                };
                if let Some(directory) = &mut self.directory {
                    directory.link::<Residue, _>(&index, &view, &staged, handle);
                }
            }
        }
        self.record_link(handle, staged.identity_path[encoding_len], location);
    }

    /// Publishes the staged path for a live held carrier, materializing its key on this cold path.
    fn publish_for_held<K, H, R>(
        &mut self,
        home: &H,
        sigma: Sigma,
        held: &R,
        location: PayloadLocation,
    ) where
        K: Clone,
        H: Home<K>,
        R: KeyRep<K, H>,
    {
        if self.staged.is_none() {
            return;
        }
        let key = R::key_owned(sigma, held);
        self.publish_for_query::<K, H, R>(home, sigma, &key, location);
    }

    /// Records one reverse link, appending the handle's slot or overwriting a retarget.
    fn record_link(&mut self, handle: u32, identity_node: u32, location: PayloadLocation) {
        let at = handle as usize;
        if at < self.links.len() {
            self.links[at] = (identity_node, location);
        } else {
            self.links.push((identity_node, location));
        }
    }

    /// The terminal handle the committed directory assigns to a live held carrier, or `None`.
    ///
    /// Temporal `O(64 + |E|)` directory steps plus one carrier key materialization; spatial `O(1)`
    /// beyond the materialized key. This is the relocation-side lookup: the directory is unchanged
    /// by a relocation, so the handle is recovered from the payload identity and only its recorded
    /// location is retargeted.
    fn linked_handle<K, H, R>(&self, home: &H, sigma: Sigma, held: &R) -> Option<u32>
    where
        K: Clone,
        H: Home<K>,
        R: KeyRep<K, H>,
    {
        let directory = self.directory.as_ref()?;
        if matches!(sigma, Sigma::Elliptic) {
            return None;
        }
        let key = R::key_owned(sigma, held);
        let view = QueryIdentity::<K, H, R> {
            key: &key,
            _carrier: PhantomData,
        };
        match sigma {
            Sigma::Elliptic => None,
            Sigma::Parabolic => {
                let index = home.parabolic_dual(&key);
                match directory.walk::<ParabolicDual, _>(&index, &view) {
                    Walk::Terminal(handle) => Some(handle),
                    Walk::Miss { .. } => None,
                }
            }
            Sigma::Hyperbolic => {
                let profile = match self.profile {
                    Some(profile) => profile,
                    None => return None,
                };
                let index = home.hyperbolic_scalar(profile, &key);
                match directory.walk::<Residue, _>(&index, &view) {
                    Walk::Terminal(handle) => Some(handle),
                    Walk::Miss { .. } => None,
                }
            }
        }
    }

    /// The terminal handle the committed directory assigns to a query key, or `None`.
    ///
    /// Temporal `O(64 + |E|)` directory steps and no key materialization; spatial `O(1)`. This is
    /// the query-side counterpart of [`TableLocator::linked_handle`], used by the parallel
    /// directory assembly to find a key already linked without a hold or a second key buffer. A
    /// miss is `None`, never an absence certificate.
    fn linked_handle_query<K, H, R>(&self, home: &H, sigma: Sigma, key: &K) -> Option<u32>
    where
        K: Clone,
        H: Home<K>,
        R: KeyRep<K, H>,
    {
        let directory = self.directory.as_ref()?;
        if matches!(sigma, Sigma::Elliptic) {
            return None;
        }
        let view = QueryIdentity::<K, H, R> {
            key,
            _carrier: PhantomData,
        };
        match sigma {
            Sigma::Elliptic => None,
            Sigma::Parabolic => {
                let index = home.parabolic_dual(key);
                match directory.walk::<ParabolicDual, _>(&index, &view) {
                    Walk::Terminal(handle) => Some(handle),
                    Walk::Miss { .. } => None,
                }
            }
            Sigma::Hyperbolic => {
                let profile = self.profile?;
                let index = home.hyperbolic_scalar(profile, key);
                match directory.walk::<Residue, _>(&index, &view) {
                    Walk::Terminal(handle) => Some(handle),
                    Walk::Miss { .. } => None,
                }
            }
        }
    }

    /// Rebases one terminal's recorded payload location without touching the directory.
    ///
    /// The terminal handle is unchanged by assembly; only the payload location it names moves. A
    /// handle with no recorded link is left alone. This allocates nothing.
    fn set_link_location(&mut self, handle: u32, location: PayloadLocation) {
        // Guarded index plus a copied old pair and a whole-pair write: no `get_mut` result or
        // mutable-reference tuple is formed. An out-of-range handle is a no-op.
        let at = handle as usize;
        if at < self.links.len() {
            let (identity_node, _) = self.links[at];
            self.links[at] = (identity_node, location);
        }
    }

    /// Retargets one relocated representative's reverse link without a merge.
    ///
    /// The terminal handle is unchanged by relocation; only the payload location it names moves.
    /// A link that does not name `old` (for example an already-retargeted one) is left alone.
    fn retarget_held<K, H, R>(
        &mut self,
        home: &H,
        sigma: Sigma,
        held: &R,
        old: PayloadLocation,
        new: PayloadLocation,
    ) where
        K: Clone,
        H: Home<K>,
        R: KeyRep<K, H>,
    {
        if same_location(old, new) {
            return;
        }
        let Some(handle) = self.linked_handle::<K, H, R>(home, sigma, held) else {
            return;
        };
        let at = handle as usize;
        // Guarded index plus a copied old pair: a missing handle stays a no-op and only a link
        // naming `old` is rewritten, as one whole-pair write.
        if at < self.links.len() {
            let (identity_node, current) = self.links[at];
            if same_location(current, old) {
                self.links[at] = (identity_node, new);
            }
        }
    }
}

impl<K, V, H, const N: usize, R> Table<K, V, H, N, R>
where
    K: Clone,
    V: Clone,
    H: Home<K>,
    R: KeyRep<K, H>,
{
    /// Reserves and stages the pending key's directory path before any payload effect.
    ///
    /// This is the locator half of the joint table/directory transaction. The write and rebuild
    /// paths call it before their first hold, clone, merge or relocation, so no locator failure can
    /// occur after a table merge. A refusal leaves the committed table and directory unchanged.
    fn reserve_locator_path(
        &mut self,
        sigma: Sigma,
        key: &K,
    ) -> Result<LocatorReservationReceipt, StorageError> {
        self.locator.reserve_path::<K, H, R>(&self.home, sigma, key)
    }

    /// Reserves and links the directory path of every distinct non-elliptic input key before any
    /// worker callback.
    ///
    /// This is the parallel build's locator half of the joint transaction. Every residue node,
    /// identity node, identity edge, staged path entry and reverse link is reserved here, on the
    /// calling thread, before the scoped fill workers run, so a locator refusal starts no worker and
    /// runs no hold, clone, merge or relocation. Each distinct key's complete path is linked once: a
    /// repeat key is found through [`TableLocator::linked_handle_query`] and skipped, so no
    /// duplicate node or edge is staged or merged. The recorded payload location is a placeholder,
    /// rebased by [`Self::rebase_directory`] once the assembled payload locations are known.
    ///
    /// Temporal `O(|kvs| * (64 + |E|))` directory steps, counted in input entries and key bytes;
    /// spatial `O(distinct keys * |E|)` directory storage. This is a cold build-time pass.
    pub(crate) fn preassemble_directory(
        &mut self,
        kvs: &[((H::Faces, K), V)],
    ) -> Result<(), StorageError> {
        let mut i = 0;
        while i < kvs.len() {
            let (face, key) = &kvs[i].0;
            let sigma = face.sigma();
            if !matches!(sigma, Sigma::Elliptic)
                && self
                    .locator
                    .linked_handle_query::<K, H, R>(&self.home, sigma, key)
                    .is_none()
            {
                self.reserve_locator_path(sigma, key)?;
                self.locator.publish_for_query::<K, H, R>(
                    &self.home,
                    sigma,
                    key,
                    PayloadLocation::Spill(0, 0),
                );
            }
            i += 1;
        }
        Ok(())
    }

    /// Rebases every preassembled terminal to the assembled payload's real location.
    ///
    /// Runs after the worker regions are assembled into the table and after the last fallible
    /// reservation, so it allocates nothing: it walks the committed directory for the query key's
    /// handle, classifies the assembled payload and overwrites the placeholder location. A key that
    /// is not live or not linked is a checked refusal, not a partial certificate.
    ///
    /// Temporal `O(|kvs| * (64 + |E| + probe))` directory steps and spill probes, counted in input
    /// entries and key bytes; spatial `O(1)` beyond the directory already built.
    pub(crate) fn rebase_directory(
        &mut self,
        kvs: &[((H::Faces, K), V)],
    ) -> Result<(), StorageError> {
        let mut i = 0;
        while i < kvs.len() {
            let (face, key) = &kvs[i].0;
            let sigma = face.sigma();
            if !matches!(sigma, Sigma::Elliptic) {
                let handle = self
                    .locator
                    .linked_handle_query::<K, H, R>(&self.home, sigma, key)
                    .ok_or(StorageError::AllocationFailed)?;
                let address = Home::address(&self.home, sigma, key);
                let location = match self.search_decision(sigma, key, &address).location {
                    SearchLocation::Flat(index) => PayloadLocation::Resident(index),
                    SearchLocation::Spill(shard, index) => PayloadLocation::Spill(shard, index),
                    SearchLocation::NoIndex => return Err(StorageError::AllocationFailed),
                };
                self.locator.set_link_location(handle, location);
            }
            i += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod locator_tests {
    use super::*;
    use crate::coord::address::LocatorSigma;
    #[cfg(feature = "injection")]
    use crate::coord::locator::reservation_failure;
    use crate::table::keyrep::{I64WordCodec, Regen};

    fn profile() -> H3_40 {
        H3_40::new().expect("the frozen H3_40 constants validate")
    }

    /// A placement-only word home: every key routes to spill, so spill growth and relocation are
    /// reachable without a resident extent.
    #[derive(Clone, Copy)]
    struct PlacementHome;

    impl Home<i64> for PlacementHome {
        fn home(&self, key: &i64) -> u64 {
            key.cast_unsigned()
        }

        fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
            TableAddress {
                place: self.home(key),
                direct_word: None,
            }
        }
    }

    impl FaceResidueCapability<i64> for PlacementHome {
        fn parabolic_dual(&self, key: &i64) -> ParabolicDual {
            WordHome.parabolic_dual(key)
        }

        fn hyperbolic_scalar(&self, profile: H3_40, key: &i64) -> Residue {
            WordHome.hyperbolic_scalar(profile, key)
        }
    }

    impl FaceCoord<i64> for PlacementHome {
        type Faces = LocatorSigma;
    }

    impl FaceHome<i64> for PlacementHome {}

    /// The empty constructor allocates no directory: the attached state is `None`, and the empty
    /// table is covered.
    #[test]
    fn empty_locator_allocates_no_directory() {
        let table = Table::<i64, i64, WordHome, 16>::new(0, WordHome);
        assert!(table.locator.profile.is_none());
        assert!(table.locator.directory.is_none());
        assert!(table.locator.links.is_empty());
        assert!(table.locator.staged.is_none());
        assert!(table.check_storage_coverage());
    }

    /// The first locator-face insert materializes the directory, links the payload and leaves no
    /// staging; the elliptic face has no locator and materializes nothing.
    #[test]
    fn real_inserts_maintain_coverage_and_the_elliptic_face_has_no_directory() {
        let mut table = Table::<i64, i64, WordHome, 16>::new(0, WordHome);
        table
            .insert_with_at(&ReplacementMerge, Sigma::Parabolic, 7, 41)
            .unwrap();
        assert!(table.locator.directory.is_some(), "the first insert materializes the directory");
        assert!(table.locator.staged.is_none(), "publication clears the staging");
        assert_eq!(table.locator.links.len(), 1);
        assert!(table.check_storage_coverage());

        let mut elliptic = Table::<i64, i64, WordHome, 16>::new(0, WordHome);
        elliptic
            .insert_with_at(&ReplacementMerge, Sigma::Elliptic, 7, 41)
            .unwrap();
        assert!(elliptic.locator.directory.is_none(), "elliptic has no locator");
        assert!(elliptic.locator.links.is_empty());
    }

    /// Every fresh insert maintains `C` on both carriers and both filtered faces, including a
    /// default-valued key.
    #[test]
    fn fresh_inserts_maintain_coverage_on_both_carriers_and_faces() {
        fn exercise<R: KeyRep<i64, WordHome>>(label: &str) {
            let mut table = Table::<i64, i64, WordHome, 16, R>::new(-1, WordHome);
            table
                .insert_with_at(&ReplacementMerge, Sigma::Parabolic, 7, 41)
                .unwrap();
            table
                .insert_with_at(&ReplacementMerge, Sigma::Parabolic, 8, -1)
                .unwrap();
            table
                .insert_with_at(&ReplacementMerge, Sigma::Hyperbolic, 7, 42)
                .unwrap();
            assert_eq!(table.len(), 3, "{label}: exact live length");
            assert_eq!(table.locator.links.len(), 3, "{label}: one reverse link per payload");
            assert!(table.check_storage_coverage(), "{label}: coverage after fresh inserts");
            assert_eq!(*table.get_at(Sigma::Parabolic, &7), 41);
            assert_eq!(*table.get_at(Sigma::Parabolic, &8), -1);
            assert_eq!(*table.get_at(Sigma::Hyperbolic, &7), 42);
        }
        exercise::<Stored<i64>>("stored");
        exercise::<Regen<I64WordCodec>>("regen");
    }

    /// A found update keeps the existing link: the length and the reverse-link count are unchanged
    /// and `C` still holds.
    #[test]
    fn found_updates_keep_existing_links() {
        let mut table = Table::<i64, i64, WordHome, 16>::new(0, WordHome);
        table
            .insert_with_at(&ReplacementMerge, Sigma::Parabolic, 7, 41)
            .unwrap();
        let links = table.locator.links.clone();
        table
            .insert_with_at(&ReplacementMerge, Sigma::Parabolic, 7, 99)
            .unwrap();
        assert_eq!(table.len(), 1, "an update does not change the live length");
        assert_eq!(table.locator.links, links, "an update keeps the existing reverse link");
        assert_eq!(*table.get_at(Sigma::Parabolic, &7), 99);
        assert!(table.check_storage_coverage());
    }

    /// Spill-region growth relocates every existing payload; each move retargets its reverse link
    /// and `C` holds after every insert.
    #[test]
    fn spill_growth_retargets_existing_links_and_keeps_coverage() {
        let mut table = Table::<i64, i64, PlacementHome, 16>::new(-1, PlacementHome);
        let merge = ReplacementMerge;
        let mut n = 0;
        let mut key = 0i64;
        while n < 40 {
            table
                .insert_with_at(&merge, LocatorSigma::Parabolic, key, key + 1)
                .unwrap();
            assert!(table.check_storage_coverage(), "coverage after insert {n}");
            key += 256;
            n += 1;
        }
        assert_eq!(table.len(), 40);
        let mut n = 0;
        let mut key = 0i64;
        while n < 40 {
            assert_eq!(
                *table.get_at(LocatorSigma::Parabolic, &key),
                key + 1,
                "readback after spill growth {n}"
            );
            key += 256;
            n += 1;
        }
    }

    /// The I6 rejecting control fixture: a real constructor and real inserts drive a spill-to-flat
    /// relocation, and `C` plus readback must hold. Omitting the relocation retarget leaves the
    /// moved payload's link naming the vacated spill slot, so `check_storage_coverage` rejects.
    fn spill_to_flat_relocation_keeps_coverage<R: KeyRep<i64, WordHome>>(label: &str) {
        let mut table = Table::<i64, i64, WordHome, 16, R>::new(0, WordHome);
        let merge = ReplacementMerge;
        // Key 256 routes to lane 1 while the resident extent is empty and the density is too low to
        // grow, so it spills.
        table
            .insert_with_at(&merge, Sigma::Parabolic, 256, 1)
            .unwrap();
        // Key 0 grows the resident extent to one lane; key 256 stays in spill.
        table.insert_with_at(&merge, Sigma::Parabolic, 0, 2).unwrap();
        for key in 1..256_i64 {
            table
                .insert_with_at(&merge, Sigma::Parabolic, key, key + 1)
                .unwrap();
        }
        assert!(table.check_storage_coverage(), "{label}: coverage before the relocation");
        // Key 257 doubles the extent: key 256 relocates from spill to flat index 1.
        table
            .insert_with_at(&merge, Sigma::Parabolic, 257, 9)
            .unwrap();

        assert_eq!(table.len(), 258, "{label}: exact live length");
        let relocated = table.search_decision_at(Sigma::Parabolic, &256);
        assert_eq!(relocated.class, SearchClass::Found, "{label}: relocated key found");
        assert_eq!(
            relocated.location,
            SearchLocation::Flat(1),
            "{label}: key 256 moved from spill to flat"
        );
        assert_eq!(*table.get_at(Sigma::Parabolic, &256), 1, "{label}: relocated readback");
        assert_eq!(*table.get_at(Sigma::Parabolic, &0), 2);
        assert_eq!(*table.get_at(Sigma::Parabolic, &257), 9);
        assert!(
            table.check_storage_coverage(),
            "{label}: coverage after the spill-to-flat relocation"
        );
    }

    #[test]
    fn spill_to_flat_relocation_keeps_coverage_on_both_carriers() {
        spill_to_flat_relocation_keeps_coverage::<Stored<i64>>("stored");
        spill_to_flat_relocation_keeps_coverage::<Regen<I64WordCodec>>("regen");
    }

    /// Rejecting control: an empty directory beside a live payload must fail `C`. The real insert
    /// first establishes coverage; removing the directory then uncovers the live payload.
    #[test]
    fn empty_directory_beside_a_live_payload_is_rejected() {
        let mut table = Table::<i64, i64, WordHome, 16>::new(0, WordHome);
        table
            .insert_with_at(&ReplacementMerge, Sigma::Parabolic, 7, 41)
            .unwrap();
        assert!(table.check_storage_coverage());
        table.locator.directory = None;
        assert!(
            !table.check_storage_coverage(),
            "the allocation-free empty directory must not cover a live payload"
        );
        table.locator.directory = Some(Directory::new());
        assert!(
            !table.check_storage_coverage(),
            "a materialized but unlinked directory must not cover a live payload"
        );
    }

    /// Unlinked intermediate staging can issue no certificate, even over an otherwise covered
    /// table.
    #[test]
    fn unlinked_staging_issues_no_certificate() {
        let mut table = Table::<i64, i64, WordHome, 16>::new(0, WordHome);
        table
            .insert_with_at(&ReplacementMerge, Sigma::Parabolic, 7, 41)
            .unwrap();
        assert!(table.check_storage_coverage());
        table
            .reserve_locator_path(Sigma::Parabolic, &8)
            .expect("a fresh path reserves");
        assert!(table.locator.staged.is_some());
        assert!(!table.check_storage_coverage());
        table.locator.staged = None;
        assert!(table.check_storage_coverage());
    }

    /// The checked locator reservation receipt observes real capacities that meet every checked
    /// target, including the scratch staging buffer and the reverse-link registry.
    #[test]
    fn locator_reservation_receipt_observes_every_checked_target() {
        let mut table = Table::<i64, i64, WordHome, 16>::new(0, WordHome);
        let key = 7i64;

        let receipt = table
            .reserve_locator_path(Sigma::Parabolic, &key)
            .expect("a fresh word path reserves");

        assert!(
            receipt.holds(),
            "every observed bound must meet its checked target"
        );
        assert_eq!(receipt.parabolic_nodes.required_len, 1 + u64::BITS as usize);
        // Only the selected face's arena grows; the other keeps its single fresh root.
        assert_eq!(receipt.residue_nodes.required_len, 1);
        assert_eq!(receipt.identity_nodes.required_len, key.encoding_len() + 1);
        assert_eq!(receipt.identity_edges.required_len, key.encoding_len() + 1);
        assert_eq!(receipt.staging.required_len, key.encoding_len() + 1);
        assert_eq!(receipt.retained_identity.required_len, key.encoding_len());
        // The scratch staging buffer, identity edges and reverse-link registry report real
        // capacities.
        assert!(receipt.staging.observed_capacity >= receipt.staging.required_len);
        assert!(receipt.reverse_links.observed_capacity >= receipt.reverse_links.required_len);
        assert!(receipt.identity_edges.observed_capacity >= receipt.identity_edges.required_len);

        // The reservation stages a path but publishes no reverse link.
        assert!(table.locator.directory.is_some());
        assert!(table.locator.links.is_empty());
        assert!(table.locator.staged.is_some());
    }

    /// The elliptic face has no locator, so its reservation is the empty receipt and materializes
    /// no directory.
    #[test]
    fn elliptic_reserves_an_empty_receipt_and_no_directory() {
        let mut table = Table::<i64, i64, WordHome, 16>::new(0, WordHome);
        let receipt = table
            .reserve_locator_path(Sigma::Elliptic, &7)
            .expect("the elliptic face reserves nothing");
        assert_eq!(receipt, LocatorReservationReceipt::EMPTY);
        assert!(receipt.holds());
        assert!(table.locator.directory.is_none());
        assert!(table.locator.staged.is_none());
    }

    /// A forced late identity-edge refusal inside the joint rebuild transaction refuses the whole
    /// prepare, so the apply runs no hold, clone, merge or relocation and the table storage is
    /// unchanged. The locator reservation runs before the holds, so no locator failure occurs after
    /// a table merge.
    #[cfg(feature = "injection")]
    #[test]
    fn locator_refusal_precedes_the_table_merge() {
        let mut table = Table::<i64, i64, WordHome, 16>::new(0, WordHome);
        table
            .insert_with_at(&ReplacementMerge, Sigma::Parabolic, 7, 41)
            .unwrap();
        table.locator.profile = Some(profile());
        table.locator.directory = Some(Directory::new());

        let before_len = table.len();
        let before_lanes = table.resident.lanes;
        let before_tags: Vec<u8> = table.shards[0].tags.clone();
        let before_slots = table.shards[0].slots.len();

        // A checked rebuild plan that would grow the resident area and commit if it were applied.
        let mut spill_loads = vec![0usize; SHARDS];
        spill_loads[0] = 1;
        let plan = RebuildPlan {
            lanes: 1,
            entries: Vec::new(),
            spill_loads,
        };
        let key = 99i64;
        let address = Home::address(&WordHome, Sigma::Parabolic, &key);

        reservation_failure::arm_late_edge(Refusal::AllocationFailed);
        let prepared = table.prepare_from_plan(plan, Sigma::Parabolic, &key, &address);
        reservation_failure::disarm();

        assert!(matches!(
            prepared,
            PreparedResult::Refused(StorageError::AllocationFailed)
        ));
        let outcome = table.apply_prepared_rebuild(
            prepared,
            &ReplacementMerge,
            Sigma::Parabolic,
            key,
            99,
            &address,
        );
        assert_eq!(outcome, Err(StorageError::AllocationFailed));

        // No table merge ran: length, resident extent, tags and payloads are unchanged.
        assert_eq!(table.len(), before_len);
        assert_eq!(table.resident.lanes, before_lanes);
        assert_eq!(table.shards[0].tags, before_tags);
        assert_eq!(table.shards[0].slots.len(), before_slots);
        assert_eq!(*table.get_at(Sigma::Parabolic, &7), 41);
        assert!(table.locator.staged.is_none(), "a refusal stages nothing");
    }
}
