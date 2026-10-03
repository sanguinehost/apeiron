// The bit-exact read, ported from incumbent `fold hashtrinity/src/lib.rs:1124` (`get_exact`).
// Included textually by `include!("table/getexact.rs")` from `src/table.rs`, so these items stay
// definitions of the `table` module and keep their generated type identities. Bodies, layout, field
// privacy, signatures and visibility are unchanged; this is a physical-file cut only.
//
// The incumbent's `get_exact` differs from its sibling `get` in two ways, and only one of them was
// missing here. Its unconditional `(Sigma, K)` compare — `full.asm` `probe_get_exact` `188c3`-`188cb`,
// loading the `keys` store and comparing the query against it before the `vals` load at `188cd` —
// already reached apeiron at `54ce281`: `spill_read` runs `matches` at both its home site
// (`src/table/probe.rs`) and its tail, and the resident path compared `(Sigma, K)` inline. What the
// incumbent's `Option` return adds, and what [`Table::get`] cannot express, is *absence reported
// rather than substituted*: `get` and `get_at` are total and return `self.default`, so a stored
// value equal to the default is indistinguishable from an absent key.
//
// So this file adds the `Option` read and shares the walk with the total read, rather than adding a
// second probe: `resident_exact` holds the one resident decision and `get_exact_spill` holds the one
// spill `spill_read` call, and `get_at`/`get_spill` are re-expressed over the same two pieces. The
// three-state [`ResidentRead`] keeps the terminal direct-cover vacancy distinct from a full-key
// reject; collapsing `Vacant` into `Miss` would let a vacant direct word fall through to a spill
// region, and collapsing it into `Hit` would return a wrong value.

/// The three outcomes of the resident direct-cover read, kept distinct because only one of them is
/// terminal for the total read.
///
/// `Vacant` is the direct word resolving to a reserved-but-unpublished resident slot: the resident
/// initialization fills all `SHARDS * lanes` slots `live: false`, so this is a slot inside the
/// resident extent that no entry occupies. `Miss` is a full-key reject on an occupied direct word —
/// or a direct word outside the extent entirely — which is **not** absence: the entry may live in a
/// spill region. `Hit` carries the stored value.
///
/// The two reads treat `Vacant` differently, and that asymmetry is the contract:
///
///   * [`Table::get_at`] answers `&self.default` and runs **no** spill lookup. This is the resident
///     kernel that landed at `54ce281`, unchanged: an unwritten direct word ends the read.
///   * [`Table::get_exact_at`] falls through to the spill probe. An unpublished resident slot is not
///     an absence certificate for a key that may be homed in spill, and never reporting a false
///     absence is the exact read's whole purpose.
///
/// Collapsing `Vacant` into `Miss` would change `get_at`; collapsing it into `Hit` would return an
/// unpublished slot's payload, which is exactly the defect the incumbent's `get_exact` excludes.
enum ResidentRead<'a, V> {
    /// The direct word is unoccupied: terminal absence, no spill lookup.
    Vacant,
    /// The direct word is occupied but the `(Sigma, K)` compare rejected.
    Miss,
    /// The direct word holds the query under its own face.
    Hit(&'a V),
}

impl<K, V, H: Home<K>, const N: usize, R: KeyRep<K, H>> Table<K, V, H, N, R> {
    /// The resident half of the read, shared by [`Table::get_at`] and [`Table::get_exact_at`] so the
    /// full `(Sigma, K)` compare has exactly one definition on this path.
    ///
    /// The compare is unconditional: `flat.sigma == sigma && R::matches(...)`. A coordinate or
    /// fingerprint match never returns a value on its own. Temporal `O(1)`; spatial `O(1)`.
    #[inline]
    fn resident_exact(&self, sigma: Sigma, address: &TableAddress, key: &K) -> ResidentRead<'_, V> {
        if let Some(word) = address.direct_word {
            if let Some(index) = self.resident.index(word) {
                let flat = &self.resident.slots[index];
                if flat.live {
                    if flat.sigma == sigma && R::matches(sigma, &flat.held, key, address) {
                        return ResidentRead::Hit(&flat.value);
                    }
                    return ResidentRead::Miss;
                }
                return ResidentRead::Vacant;
            }
        }
        ResidentRead::Miss
    }

    /// Bit-exact read: absence is `None`, never a substituted default.
    ///
    /// [`Table::get`] and [`Table::get_at`] are total — they substitute `self.default` when nothing
    /// is stored, so a stored value equal to the default is indistinguishable from an absent one.
    /// This read reports absence instead, and shares one walk with [`Table::get_at`]: the same
    /// `resident_exact` decision, the same `spill_read` body, the same unconditional `(Sigma, K)`
    /// compare. It is the port of `fold hashtrinity/src/lib.rs:1124` (`get_exact`), whose `Option`
    /// return is the whole behavioural difference from its sibling `get` — in the pinned
    /// disassembly one compare, `full.asm` `probe_get_exact` `188c3`–`188cb` loading the `keys`
    /// store and comparing the query against it before the `vals` load at `188cd`.
    ///
    /// **Admissible domain.** Total over every `face: H::Faces` and every `key: &K`. `None` is
    /// returned on the two absence certificates [`Self::get_exact_spill`] admits — a zero
    /// coordinate word at the home slot, or a completed bounded cycle — and on nothing else. The
    /// resident half contributes no absence of its own: [`ResidentRead::Vacant`] and
    /// [`ResidentRead::Miss`] both continue to the spill probe, so neither an unpublished nor a
    /// rejected resident slot can produce a false absence. `None` is never returned for a stored
    /// entry, and never for an entry stored under a different face or a different key: the resident
    /// compare is `flat.sigma == sigma && R::matches(..)` and the spill compare is `matches(..)` at
    /// `src/table/state.rs`.
    ///
    /// Temporal `O(capacity)` slot visits on the collision tail and `O(1)` otherwise, counted as in
    /// [`Table::get_at`]; spatial `O(1)`.
    #[must_use]
    pub fn get_exact_at(&self, face: H::Faces, key: &K) -> Option<&V> {
        let sigma = face.sigma();
        let address = Home::address(&self.home, sigma, key);
        match self.resident_exact(sigma, &address, key) {
            ResidentRead::Vacant | ResidentRead::Miss => {
                self.get_exact_spill(sigma, address.place, address.direct_word, key)
            }
            ResidentRead::Hit(value) => Some(value),
        }
    }

    /// The absence-reporting form of [`Table::get_spill`]: the one `spill_read` call on this path,
    /// shared with the total read, so there is no second probe.
    ///
    /// `None` is the same two-absence-cases certificate the total read turns into
    /// `self.default`; the routing is identical, including forwarding `direct_word` unchanged.
    #[must_use]
    #[inline(never)]
    fn get_exact_spill(
        &self,
        sigma: Sigma,
        place: u64,
        direct_word: Option<u32>,
        key: &K,
    ) -> Option<&V> {
        let address = TableAddress { place, direct_word };
        let shard_at = shard(&address);
        let index = spill_read::<K, H, R, V>(&self.shards[shard_at], sigma, &address, key, N)?;
        Some(&self.shards[shard_at].slots[index].value)
    }
}