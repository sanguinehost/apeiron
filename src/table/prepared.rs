/// Reserves one spill region per shard for a checked rebuild plan.
///
/// This is a free function over the carrier `R` and value `V` alone, not a method of the generic
/// `Table`: the region construction is a separately named, monomorphizable item, so the extraction
/// pipeline instantiates it from the concrete roots instead of translating the whole `Table`
/// binder inside the loop. Exactly one implementation exists; both carriers share it. The
/// reservation order is unchanged: the region directory is reserved first, then each nonempty
/// shard's region is sized and reserved in shard order, and the first error is returned with no
/// partial vector. Temporal `O(SHARDS)` checked reservations; spatial `O(SHARDS)` region headers
/// plus their reserved payload storage.
fn reserve_plan_regions<R: Clone, V: Clone, const N: usize>(
    spill_loads: &[usize],
) -> Result<Vec<Region<R, V>>, StorageError> {
    let mut regions: Vec<Region<R, V>> = Vec::new();
    try_reserve_exact(&mut regions, SHARDS)?;
    let mut reservation_error = None;
    let mut shard_at = 0;
    while shard_at < SHARDS && reservation_error.is_none() {
        let count = spill_loads[shard_at];
        let region = if count == 0 {
            Ok(Region::empty())
        } else {
            match slots_for(count, N) {
                Ok(slots) => Region::reserve_empty(slots),
                Err(error) => Err(error),
            }
        };
        match region {
            Ok(region) => regions.push(region),
            Err(error) => reservation_error = Some(error),
        }
        shard_at += 1;
    }
    match reservation_error {
        Some(error) => Err(error),
        None => Ok(regions),
    }
}

/// Initializes every reserved spill region with the pending held carrier.
///
/// The free-function counterpart of [`reserve_plan_regions`], over `R` and `V` alone: it runs
/// after the last fallible reservation, allocates nothing and replaces no reserved vector.
/// Temporal `O(regions)` payload clones; spatial `O(1)` beyond the regions already reserved.
fn initialize_reserved_regions<R: Clone, V: Clone>(
    regions: &mut [Region<R, V>],
    sigma: Sigma,
    held: &R,
    default: &V,
) {
    let mut shard_at = 0;
    while shard_at < regions.len() {
        regions[shard_at].initialize(sigma, held, default);
        shard_at += 1;
    }
}

/// Marks one in-bounds selected flat slot; `None` leaves the bitmap unchanged.
fn mark_selected_occupied(occupied: &mut [bool], selected: Option<usize>) {
    if let Some(index) = selected {
        occupied[index] = true;
    }
}

/// Increments one spill load or returns the checked-sizing error without changing that load.
fn increment_spill_load(spill_loads: &mut [usize], spill: usize) -> Option<StorageError> {
    match spill_loads[spill].checked_add(1) {
        Some(count) => {
            spill_loads[spill] = count;
            None
        }
        None => Some(StorageError::SizeOverflow),
    }
}

/// Accounts for a spill only when no flat slot was selected.
fn account_spill_load(
    spill_loads: &mut [usize],
    spill: usize,
    selected: Option<usize>,
) -> Option<StorageError> {
    if selected.is_some() {
        None
    } else {
        increment_spill_load(spill_loads, spill)
    }
}

/// Reads one already-prepared spill region.  The caller supplies the routed region and its
/// initialized capacity; this core does not plan, reserve, or allocate.
///
/// The face token is carried to this shared consumer and erased to the stored [`Sigma`] only at
/// the [`spill_read`] storage boundary.
pub(crate) fn get_prepared_shard<'a, K, H: FaceCoord<K>, R: KeyRep<K, H>, V>(
    region: &'a Region<R, V>,
    face: H::Faces,
    address: &TableAddress,
    key: &K,
    n: usize,
) -> Option<&'a V> {
    match spill_read::<K, H, R, V>(region, face.sigma(), address, key, n) {
        Some(index) => Some(&region.slots[index].value),
        None => None,
    }
}

/// Applies one entry to an already-prepared spill region and reports where it landed.
///
/// Returns `(fresh, slot)`: `fresh` is `true` exactly for a new key, and `slot` is the region slot
/// the entry occupies after the merge. A repeated key reports `false` and its existing slot, so a
/// caller recording placement learns one slot per key for every entry of that key.
///
/// The prepared-resource adapter supplies an initialized region with a vacancy for a new key.
/// This core performs no fallible allocation.
pub(crate) fn apply_prepared_shard_placed<K, V, H, R, F, const N: usize>(
    region: &mut Region<R, V>,
    merge: &F,
    default: &V,
    face: H::Faces,
    key: &K,
    value: V,
    address: &TableAddress,
) -> Result<(bool, usize), StorageError>
where
    V: Clone,
    H: Home<K>,
    R: KeyRep<K, H>,
    F: Merge<V>,
{
    // Erasure boundary: the typed token becomes the stored `Sigma` here and nowhere earlier.
    let sigma = face.sigma();
    match spill_locate::<K, H, R, V>(region, sigma, address, key, N) {
        SpillSite::Found(index) => {
            let old = region.slots[index].value.clone();
            region.slots[index].value = merge.merge(value, old);
            Ok((false, index))
        }
        SpillSite::Vacant(index) => {
            region.publish_live(
                index,
                (
                    sigma,
                    R::hold(sigma, key, address),
                    merge.merge(value, default.clone()),
                ),
                address.place,
            );
            Ok((true, index))
        }
        SpillSite::Grow => Err(StorageError::SizeOverflow),
    }
}

/// Applies one entry to an already-prepared spill region.
///
/// The boolean projection of [`apply_prepared_shard_placed`] for callers that do not record the
/// placement: `true` exactly for a fresh key. The prepared-resource adapter supplies an initialized
/// region with a vacancy for a new key; this core performs no fallible allocation.
pub(crate) fn apply_prepared_shard<K, V, H, R, F, const N: usize>(
    region: &mut Region<R, V>,
    merge: &F,
    default: &V,
    face: H::Faces,
    key: &K,
    value: V,
    address: &TableAddress,
) -> Result<bool, StorageError>
where
    V: Clone,
    H: Home<K>,
    R: KeyRep<K, H>,
    F: Merge<V>,
{
    match apply_prepared_shard_placed::<K, V, H, R, F, N>(
        region, merge, default, face, key, value, address,
    ) {
        Ok((fresh, _slot)) => Ok(fresh),
        Err(error) => Err(error),
    }
}

/// Reserves a shard through the resource adapter, then applies its stable partition runs.
/// The allocation is deliberately outside [`apply_prepared_shard`].
pub(crate) fn build_prepared_shard<K, V, H, R, F, const N: usize>(
    home: &H,
    merge: &F,
    default: &V,
    input: &[((H::Faces, K), V)],
    parts: &partition::Parts,
    shard_at: usize,
    load: usize,
) -> Result<Region<R, V>, StorageError>
where
    K: Clone,
    V: Clone,
    H: Home<K>,
    R: KeyRep<K, H>,
    F: Merge<V>,
{
    if load == 0 {
        return Ok(Region::empty());
    }
    let first = parts
        .runs(shard_at)
        .find_map(|run| match run.first() {
            Some(value) => Some(*value),
            None => None,
        })
        .ok_or(StorageError::SizeOverflow)?;
    let ((face, key), _) = &input[first];
    let address = Home::address(home, face.sigma(), key);
    // Reserve the complete checked region (tags, payloads, summary) before the first hold or
    // clone; a refusal here leaves no partial region and runs no callback.
    let mut region = Region::reserve_empty(slots_for(load, N)?)?;
    let sigma = face.sigma();
    let held = R::hold(sigma, key, &address);
    region.initialize(sigma, &held, default);
    for run in parts.runs(shard_at) {
        for &index in run {
            let ((entry_face, key), value) = &input[index];
            let address = Home::address(home, entry_face.sigma(), key);
            apply_prepared_shard::<K, V, H, R, F, N>(
                &mut region,
                merge,
                default,
                *entry_face,
                key,
                value.clone(),
                &address,
            )?;
        }
    }
    Ok(region)
}

#[cfg(test)]
mod prepared_placement_tests {
    use super::keyrep::KeyRep;
    use super::{
        apply_prepared_shard, apply_prepared_shard_placed, spill_locate, Region, ReplacementMerge,
        SpillSite, Stored, TableAddress, WordHome,
    };
    use crate::cell::Sigma;

    /// A fresh entry reports `(true, slot)`; a repeat key reports `(false, same slot)` with the
    /// merge applied, so a caller that records placement learns one slot per key.
    #[test]
    fn placement_reports_the_slot_for_fresh_and_repeat_entries() {
        let address = TableAddress {
            place: 0,
            direct_word: None,
        };
        let held = <Stored<i64> as KeyRep<i64, WordHome>>::hold(Sigma::Parabolic, &7, &address);
        let mut region = Region::reserve_empty(32).expect("two initialized groups");
        region.initialize(Sigma::Parabolic, &held, &0);

        let first =
            apply_prepared_shard_placed::<i64, i64, WordHome, Stored<i64>, ReplacementMerge, 16>(
                &mut region,
                &ReplacementMerge,
                &0,
                Sigma::Parabolic,
                &7,
                70,
                &address,
            )
            .expect("a fresh entry fits");
        assert_eq!(
            first,
            (true, 0),
            "a fresh key lands in the first vacant slot"
        );
        assert_eq!(
            region.slots[0].value, 70,
            "the placed slot holds the merged value"
        );
        assert_eq!(
            spill_locate::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &address,
                &7,
                16,
            ),
            SpillSite::Found(0),
            "the placed key is found in the reported slot"
        );

        let second =
            apply_prepared_shard_placed::<i64, i64, WordHome, Stored<i64>, ReplacementMerge, 16>(
                &mut region,
                &ReplacementMerge,
                &0,
                Sigma::Parabolic,
                &7,
                80,
                &address,
            )
            .expect("a repeat entry fits");
        assert_eq!(second, (false, 0), "a repeat key keeps its slot");
        assert_eq!(
            region.slots[0].value, 80,
            "the merge result replaces the value"
        );

        let projected =
            apply_prepared_shard::<i64, i64, WordHome, Stored<i64>, ReplacementMerge, 16>(
                &mut region,
                &ReplacementMerge,
                &0,
                Sigma::Parabolic,
                &7,
                90,
                &address,
            )
            .expect("the projection fits");
        assert!(!projected, "the projection reports the repeat as not fresh");
        assert_eq!(
            region.slots[0].value, 90,
            "the projection applies the same merge"
        );
    }

    /// Distinct keys report distinct slots, so recorded placements partition by key.
    #[test]
    fn distinct_keys_report_distinct_slots() {
        let address = TableAddress {
            place: 0,
            direct_word: None,
        };
        let held_seven =
            <Stored<i64> as KeyRep<i64, WordHome>>::hold(Sigma::Parabolic, &7, &address);
        let mut region = Region::reserve_empty(32).expect("two initialized groups");
        region.initialize(Sigma::Parabolic, &held_seven, &0);

        let seven =
            apply_prepared_shard_placed::<i64, i64, WordHome, Stored<i64>, ReplacementMerge, 16>(
                &mut region,
                &ReplacementMerge,
                &0,
                Sigma::Parabolic,
                &7,
                70,
                &address,
            )
            .expect("the first key fits");
        let eight =
            apply_prepared_shard_placed::<i64, i64, WordHome, Stored<i64>, ReplacementMerge, 16>(
                &mut region,
                &ReplacementMerge,
                &0,
                Sigma::Parabolic,
                &8,
                80,
                &address,
            )
            .expect("the second key fits");
        assert!(seven.0 && eight.0, "both keys are fresh");
        assert_ne!(seven.1, eight.1, "distinct keys occupy distinct slots");
    }
}
