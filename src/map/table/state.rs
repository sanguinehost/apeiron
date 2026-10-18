// Region, decision, scalar-geometry and rebuild-plan state for the growable table.
// Included textually by `include!("table/state.rs")` from `src/map/table.rs`, so these items stay
// definitions of the `table` module and keep their generated type identities. Bodies, layout,
// field privacy, signatures and visibility are unchanged; this is a physical-file cut only.

/// Assumed upper bound, in bytes, of one flat slot, used only for the byte-budget heuristic.
///
/// The executing slots are fixed-width words; `core::mem::size_of` is an unmodeled Aeneas external,
/// so the budget uses this bound. A larger element type is capped to correspondingly fewer lanes.
const ASSUMED_SLOT_BYTES: usize = 32;

/// The vacant-slot marker of a region's coordinate array.
///
/// Ported from `fold hashtrinity/src/lib.rs:298` (`const EMPTY: u64 = u64::MAX`). A live coordinate
/// never equals it: [`Region::coord_of`] nudges the one colliding value aside, so
/// `coord == COORD_EMPTY` unambiguously means "vacant" and nothing else. This is the word that
/// makes an absent read `O(1)`: one load, one compare, `ret` (`full.asm` `probe_get`,
/// `cmp $0xffffffffffffffff`).
const COORD_EMPTY: u64 = u64::MAX;

/// The odd spread multiplier [`Region::home`] mixes a coordinate by, before the top `bits` of the
/// product choose the home lane.
///
/// fxhash 0.2.1's `SEED64` (`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/
/// fxhash-0.2.1/lib.rs:46`, sha256 `885b72a1c977dc1c3ff4e0652719e62b6d8c3e713403e571afd581283f72df7b`),
/// the multiplier the `hashbrown` `fxhash` arm this face is measured against multiplies by. It is
/// the seed alone: fxhash's own chain rotates and xors before the multiply, and this constant does
/// neither — the measured arm is the multiply, and the reduction end is the top `bits`, as in the
/// incumbent's Fibonacci form.
///
/// It is odd, so the multiply-shift loses no key: equality of the top `bits` of `c1 * HOME_SPREAD`
/// and `c2 * HOME_SPREAD` makes `HOME_SPREAD * (c1 - c2)` vanish modulo `2^(64 - bits)`, and an odd
/// multiplier is invertible on `Z/2^(64 - bits)`, so `c1` and `c2` agree modulo `2^(64 - bits)`.
/// Distinct coordinates below that modulus therefore never share a home lane.
///
/// Distinct from [`TAG_SPREAD`], which stays `crate::map::coord::PHI` for the tag fingerprint: the tag
/// reads seven bits of `place * TAG_SPREAD` and the home reads `bits` bits of `coord * HOME_SPREAD`
/// of a different operand, and each constant's stated separation property is its own.
const HOME_SPREAD: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// Test-only record of the spill publication steps, so the contract order can be checked on the
/// real write path. Inert outside `cfg(test)`; it is not a production or reference path.
#[cfg(test)]
mod publication_trace {
    use std::cell::RefCell;

    /// The payload write of one fresh live spill slot.
    pub(super) const PAYLOAD: u8 = 0;
    /// The nonzero tag publication of one fresh live spill slot.
    pub(super) const TAG: u8 = 1;

    thread_local! {
        static EVENTS: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }

    /// Clears the recorded steps on this thread.
    pub(super) fn clear() {
        EVENTS.with(|events| events.borrow_mut().clear());
    }

    /// Records one publication step on this thread.
    pub(super) fn record(step: u8) {
        EVENTS.with(|events| events.borrow_mut().push(step));
    }

    /// Returns the steps recorded on this thread since the last [`clear`].
    pub(super) fn events() -> Vec<u8> {
        EVENTS.with(|events| events.borrow().clone())
    }
}

/// Test-only census of the multiply-shift branch of [`Region::home`]: first-fit slot visits per key
/// over a natural and a congruence-controlled coordinate population.
///
/// `home` has two branches. This census is on the multiply-shift branch, entered when
/// `coord >= coords.len()`; the other branch is `coord as usize` with no constant. The registered map
/// population takes the multiply-shift branch on 99.9998% of `home` calls, so the constant is read
/// here and nowhere else on that face.
///
/// **The mirror is differentially controlled.** [`high`] and [`low`] below restate `home`'s
/// expression with the spread as a parameter so both candidate constants can be run in one process;
/// `home_mirror_agrees_with_the_region` runs the real [`Region::home`] on a real [`Region`] and
/// asserts the landed mirror equals it on every key of both populations. A census figure is then the
/// real method's cost, not the mirror's.
///
/// **The control is the congruence population.** [`controlled`] clears each coordinate's low
/// `bits` bits, so every key in the region is congruent modulo the region width. That is the
/// population on which a *low*-window reduction of any odd spread collapses to one lane — every
/// product has `bits` trailing zero bits — and it is the arm
/// `home_fxhash_high_survives_the_congruence_control` uses to separate a real improvement from a
/// relabeling: the low-window arm must fail here, or the control cannot see anything.
#[cfg(test)]
mod home_census {
    use super::{Region, HOME_SPREAD, SHARD_BITS};
    use crate::cell::Sigma;
    use crate::map::coord::PHI;

    /// fxhash 0.2.1's `SEED64`, transcribed at [`super::HOME_SPREAD`]. Held separately from the
    /// landed constant so the census can run the incumbent and the candidate side by side.
    const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

    /// The region capacities the census reports, each a power of two so the probe mask is
    /// `capacity - 1` and the load factor is the `1/2` `slots_for` holds.
    const CAPACITIES: [usize; 2] = [1 << 8, 1 << 12];

    /// `home`'s multiply-shift expression with `spread` as a parameter, top window: the structure
    /// [`Region::home`] ships.
    fn high(c: u64, spread: u64, bits: u32, bound: usize) -> usize {
        ((c.wrapping_mul(spread) >> (64 - bits)) as usize) & (bound - 1)
    }

    /// The same product read from the *low* `bits` bits, which is how `hashbrown` reduces its
    /// `h1`. The negative arm: under [`controlled`] every product has `bits` trailing zero bits, so
    /// this returns one lane for the whole population.
    fn low(c: u64, spread: u64, bound: usize) -> usize {
        (c.wrapping_mul(spread) as usize) & (bound - 1)
    }

    /// `load` distinct coordinates spread over the whole addressable word by one fixed odd multiply.
    ///
    /// The coordinate is the *wide* half of a place word, so it is masked to `64 - SHARD_BITS` bits:
    /// `Region::coord_of` recovers `place >> SHARD_BITS` and nothing narrower survives a shift, so a
    /// coordinate with bits above `64 - SHARD_BITS` set has no place word and could not reach `home`
    /// at all. Every coordinate is at or above `capacity`, so all of them take the multiply-shift
    /// branch rather than the direct index.
    fn natural(load: usize) -> Vec<u64> {
        let wide = u64::MAX >> SHARD_BITS;
        (0..load)
            .map(|j| PHI.wrapping_mul(j as u64 + 1) & wide)
            .collect()
    }

    /// The congruence control: each coordinate of [`natural`] with its low `bits` bits cleared, so
    /// every key is congruent modulo `2^bits`, the region width.
    ///
    /// The coordinates stay distinct: equality of two masked `PHI * (j + 1)` values modulo `2^bits`
    /// forces `j1 = j2` for `j < load <= capacity <= 2^bits`, because `PHI` is invertible on
    /// `Z/2^bits`. So the control removes the low window only, and every key must still find its own
    /// lane through the high window.
    fn controlled(coords: &[u64], bits: u32) -> Vec<u64> {
        let keep = u64::MAX << bits;
        coords.iter().map(|c| c & keep).collect()
    }

    /// First-fit slot visits per key: the mean number of lanes walked from the home lane to the
    /// first vacancy, inserting `coords` in order into an empty `bound`-lane region at load factor
    /// `1/2`. `bound` is a power of two and `coords.len() == bound / 2`, so a cycle always finds a
    /// vacancy within the region's own bound.
    fn first_fit_probes_per_key(home_of: impl Fn(u64) -> usize, coords: &[u64], bound: usize) -> f64 {
        let mask = bound - 1;
        let mut occupied = vec![false; bound];
        // The count is accumulated in `f64` and the divisor is taken through `u32`, so no `usize` is
        // narrowed into a float: a region this census builds holds at most `1 << 12` keys, and the
        // mean of exact integers below `2^53` is exact in `f64` to the four decimals printed.
        let mut visits = 0.0f64;
        for &c in coords {
            let mut step = 0usize;
            loop {
                let slot = (home_of(c) + step) & mask;
                visits += 1.0;
                if !occupied[slot] {
                    occupied[slot] = true;
                    break;
                }
                step += 1;
            }
        }
        visits / f64::from(coords.len() as u32)
    }

    /// The landed mirror equals the real [`Region::home`] on every key of both populations.
    ///
    /// The rejection: a mirror that read a different operand, a different reduction end or a stale
    /// constant diverges from the region here, before any census figure is read.
    #[test]
    fn home_mirror_agrees_with_the_region() {
        for &bound in &CAPACITIES {
            let bits = bound.trailing_zeros();
            let mut region = Region::<u64, u64>::reserve_empty(bound).expect("one census region");
            region.initialize(Sigma::Parabolic, &0, &0);
            assert_eq!(region.slots.len(), bound, "the census region's lane count");
            assert_eq!(region.bits, bits, "the census region's shift width");

            let coords = natural(bound / 2);
            let held = controlled(&coords, bits);
            let population = coords.iter().chain(held.iter());
            for c in population {
                assert_eq!(
                    region.home(c << SHARD_BITS),
                    high(*c, HOME_SPREAD, bits, bound),
                    "landed mirror against home at bound {bound}, coord {c}"
                );
            }
        }
    }

    /// The landed reduction is the top window, and the top window survives the congruence control.
    ///
    /// Three arms per capacity and per population: the incumbent's `PHI` and the landed
    /// `HOME_SPREAD` in the top window [`Region::home`] ships, and `FX_SEED` in the low window as the
    /// negative arm. Under [`controlled`] every product has `bits` trailing zero bits, so the low
    /// window returns one lane for the whole population and its first-fit cost is the linear pile-up
    /// `(load + 1) / 2`; the assertions below are that this collapse is real and that the landed arm
    /// is nowhere near it. That is the control's own rejection, and it is the property that separates
    /// a reduction end from a relabeling.
    ///
    /// The natural-population figures are **printed, not asserted**. That population is one
    /// deterministic key set, `PHI * (j + 1)`, and at these loads the gap between the two top
    /// windows is a handful of collisions — a single sample of which constant happens to spread one
    /// fixed set better, not a distribution, and not the registered population. A direction asserted
    /// on it would be a sample asserted as a result. The registered-population ratio is the
    /// campaign's number, not this one's.
    #[test]
    fn home_fxhash_high_survives_the_congruence_control() {
        println!("census: first-fit slot visits per key");
        for &bound in &CAPACITIES {
            let bits = bound.trailing_zeros();
            let coords = natural(bound / 2);
            let load = bound / 2;
            // The landed arm runs through the shipped `Region::home` on a real `Region`, not through
            // the mirror, so a figure in this table is the executing method's cost. The mirror is
            // differentially pinned to it by `home_mirror_agrees_with_the_region`, and it is here
            // only because the other two arms need a different constant or a different window.
            let mut region = Region::<u64, u64>::reserve_empty(bound).expect("one census region");
            region.initialize(Sigma::Parabolic, &0, &0);
            let populations = [
                ("natural", coords.clone()),
                ("congruence-control", controlled(&coords, bits)),
            ];
            for (name, population) in populations {
                let incumbent =
                    first_fit_probes_per_key(|c| high(c, PHI, bits, bound), &population, bound);
                let landed = first_fit_probes_per_key(
                    |c| region.home(c << SHARD_BITS),
                    &population,
                    bound,
                );
                let low_window =
                    first_fit_probes_per_key(|c| low(c, FX_SEED, bound), &population, bound);
                let pile_up = f64::midpoint(f64::from(load as u32), 1.0);
                println!(
                    "census: bound={bound} load={load} population={name} \
                     phi-high={incumbent:.4} fxhash-high={landed:.4} fxhash-low={low_window:.4} \
                     collapsed-floor={pile_up:.4}"
                );

                // The landed arm is the shipped method: it must agree with the mirror on every key,
                // here as well as in the differential test.
                for &c in &population {
                    assert_eq!(
                        region.home(c << SHARD_BITS),
                        high(c, HOME_SPREAD, bits, bound),
                        "bound {bound}: the landed arm and the mirror disagree at coord {c}"
                    );
                }

                if name != "congruence-control" {
                    continue;
                }
                // The control's own rejection: the low window collapses to one lane.
                assert!(
                    low_window >= pile_up - 0.5,
                    "bound {bound}: the low-window negative arm must collapse under the control, \
                     observed {low_window:.4} against the pile-up floor {pile_up:.4}"
                );
                // The claim: the landed arm does not collapse. Under a one-lane home every key pays
                // the pile-up cost; a top window does not, and must stay near the ideal of one visit.
                assert!(
                    landed < pile_up / 4.0,
                    "bound {bound}: fxhash-high {landed:.4} collapses toward the low-window \
                     pile-up {pile_up:.4}"
                );
                assert!(
                    low_window > landed * 4.0,
                    "bound {bound}: the control must separate the arms, low-window {low_window:.4} \
                     against landed {landed:.4}"
                );
            }
        }
    }
}

/// One spill slot: the occupancy marker, the face, the held carrier and the payload.
///
/// `tag` is the occupancy marker and fingerprint in one `u8`: `0` is a vacancy, and any nonzero
/// value is the live marker [`TAG_LIVE`] OR the query fingerprint of the stored key's place. The
/// tag is a fingerprint only — a matching tag still requires the full `(Sigma, K)` compare — and it
/// is derived, never stored twice.
pub(crate) struct SpillSlot<R, V> {
    /// `0` while vacant; the nonzero occupancy marker after publication.
    pub(crate) tag: u8,
    /// The face this slot's payload belongs to.
    pub(crate) sigma: Sigma,
    /// The held carrier; the key material the full compare runs against.
    pub(crate) held: R,
    /// The merged value.
    pub(crate) value: V,
}

impl<R, V> SpillSlot<R, V> {
    /// Reports whether this slot is occupied: the nonzero-tag test.
    #[inline]
    pub(crate) fn live(&self) -> bool {
        self.tag != 0
    }
}

/// One open-addressed spill region per shard: a single flat slot array with linear probing.
///
/// **Invariants.** `slots.len()` is the capacity and is either `0` or a power of two, so the probe
/// mask is `slots.len() - 1`; `live` is the exact number of occupied slots, so
/// `live <= slots.len()` always. A slot is occupied exactly when its `tag` is nonzero.
///
/// **Probe sequence.** Slot `i` of a query with placement `place` is visited at
/// `((home(place) + step) & mask)` for `step = 0, 1, ..., capacity - 1` — linear probing over one
/// cycle, in increasing index with wraparound. The **bound is the capacity**: after `capacity`
/// steps every slot has been visited exactly once, so a cycle that finds no vacancy has observed
/// every slot occupied. That is the absence certificate the read path relies on, and it is what
/// `SpillProbeBoundAbsentSound` states: no key can be stored in a slot of a completed cycle without
/// that cycle having visited it.
///
/// **No deletion.** Nothing in the table removes a spill entry; a repeated key merges in place.
/// There are therefore no tombstones, and a vacant coordinate terminates a probe correctly: no live
/// slot can sit beyond a vacancy, because a linear insert only ever places at the first vacancy at
/// or after its home. (A future deletion would have to introduce a tombstone and would void this
/// argument.)
///
/// **Publication order.** A fresh write stores `sigma`, `held` and `value` first, then writes the
/// nonzero `tag`, then writes the coordinate word. The coordinate write is the occupancy
/// publication point, so no search-visible state has a live coordinate over an unpublished payload.
///
/// **The resident kernel is the incumbent's.** `coords` is the contiguous coordinate array of
/// `fold hashtrinity/src/lib.rs:409` (`coords: Box<[u64]>`): one `u64` per slot holding the placed
/// key's own coordinate word, or [`COORD_EMPTY`] while vacant. One word carries occupancy, identity
/// and the absence certificate at once, so an absent read is one load and one compare
/// (`full.asm` `probe_get`: `mov (%rcx,%rax,8),%rcx` then `cmp $0xffffffffffffffff`). The per-slot
/// `tag` stays as the published occupancy marker the relocation and coverage passes read; it is not
/// on the probe path.
pub(crate) struct Region<R, V> {
    /// The open-addressed slot array; length is the capacity, a power of two or zero.
    slots: Vec<SpillSlot<R, V>>,
    /// The contiguous coordinate array: one placed coordinate word per slot index, or
    /// [`COORD_EMPTY`] while vacant. Its length equals [`Self::slots`].
    coords: Vec<u64>,
    /// The incumbent's `bits` for this capacity: `log2(slots.len())`, or `0` while there is no
    /// storage. Stored rather than recomputed, because `usize::trailing_zeros` has no field in the
    /// pinned Aeneas Lean model and the fail-closed audit rejects the external (see
    /// `density_below`'s note on the same rule).
    bits: u32,
    /// The number of occupied slots.
    live: usize,
}
impl<R, V> Region<R, V> {
    fn empty() -> Self {
        Self {
            slots: Vec::new(),
            coords: Vec::new(),
            bits: 0,
            live: 0,
        }
    }

    /// The probe bound of this region: its capacity, or `0` when it has no storage.
    ///
    /// This is the number of distinct slots one bounded linear-probe cycle visits, and therefore the
    /// absence certificate's width. `O(1)`; spatial `O(1)`.
    #[inline]
    fn bound(&self) -> usize {
        self.slots.len()
    }

    /// A placement's coordinate word: `place` with the low shard byte removed, and the reserved
    /// vacant marker nudged aside.
    ///
    /// The low [`SHARD_BITS`] bits of `place` select the region (see [`shard`]), so they are
    /// constant across one region's whole population and carry no information inside it; the
    /// coordinate word is what remains. The nudge is `Table::coord_of` of
    /// `fold hashtrinity/src/lib.rs:548-561`, which maps the one colliding value aside so a live
    /// coordinate is never [`COORD_EMPTY`]. That nudge aliases the top of the range, the same
    /// 2⁻⁶⁴ identity risk the coordinate already trusts and unreachable for the coordinate words
    /// this table homes.
    #[inline]
    fn coord_of(place: u64) -> u64 {
        let c = place >> SHARD_BITS;
        if c == COORD_EMPTY {
            COORD_EMPTY - 1
        } else {
            c
        }
    }

    /// The home slot of `place`: the incumbent's home, `fold hashtrinity/src/lib.rs:565-572`.
    ///
    /// A coordinate below the region capacity indexes directly, with no hash; a larger one takes the
    /// top `bits` of `coord · HOME_SPREAD`, the multiply-shift whose multiply is on the fall-through
    /// branch in `full.asm` `probe_get` (`1822b`/`18235`) while the direct index is the taken branch
    /// (`18225`/`18229`). [`HOME_SPREAD`] is odd, so the multiply-shift is a bijection on the word
    /// and loses no key. Both branches are reduced through the cycle mask `bound - 1`, so the
    /// returned index is always in `[0, bound)`: it is the probe cycle's step-0 lane and is safe to
    /// read directly. `O(1)`; spatial `O(1)`.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the cycle mask keeps the value below the slots capacity, so it fits usize"
    )]
    #[inline]
    fn home(&self, place: u64) -> usize {
        let bound = self.slots.len();
        // A region with no storage never reaches a probe; the guard also keeps the shift below 64.
        if bound == 0 {
            return 0;
        }
        // The resident home index is the probe cycle's step-0 lane. The tail sweep and
        // `spill_locate` both apply the cycle mask `bound - 1` to `home + step`; the resident probe
        // indexes `coords`/`slots` with the home lane directly, so the home index must lie in
        // `[0, bound)` before it is used. `coord_of` is bounded by `coords.len()` and by the top
        // `bits` of the spread product, neither of which is the slots capacity, so reduce the value
        // through the same mask. On a capacity that is a power of two this is the identity for every
        // value already below capacity; it is what rules out a home at or above the capacity.
        let mask = bound - 1;
        let c = Self::coord_of(place);
        let len = self.coords.len();
        if len == 0 {
            return 0;
        }
        if c < len as u64 {
            (c as usize) & mask
        } else {
            ((c.wrapping_mul(HOME_SPREAD) >> (64 - self.bits)) as usize) & mask
        }
    }

    /// Publishes one fresh live spill slot in the contract order: payload, then the nonzero tag,
    /// then the coordinate word.
    ///
    /// The coordinate write is the occupancy publication point; no fallible operation and no
    /// search-visible handoff runs between the payload store and the coordinate store, so every
    /// search-visible state with a live coordinate has its published payload.
    #[inline]
    fn publish_live(&mut self, index: usize, payload: (Sigma, R, V), place: u64) {
        self.slots[index].sigma = payload.0;
        self.slots[index].held = payload.1;
        self.slots[index].value = payload.2;
        #[cfg(test)]
        publication_trace::record(publication_trace::PAYLOAD);
        self.slots[index].tag = tag(place);
        self.coords[index] = Self::coord_of(place);
        self.live += 1;
        #[cfg(test)]
        publication_trace::record(publication_trace::TAG);
    }
}
impl<R, V> Region<R, V> {
    /// The occupancy marker of every slot, for a before/after no-mutation assertion.
    ///
    /// Test-only: the read path never materializes this vector.
    #[cfg(test)]
    fn markers(&self) -> Vec<u8> {
        self.slots.iter().map(|slot| slot.tag).collect()
    }
}
impl<R: Clone, V: Clone> Region<R, V> {
    /// Reserves `capacity` initialized-length slots of storage without writing any slot.
    ///
    /// The capacity is checked before any reservation, so a sizing overflow refuses without a
    /// partial allocation. Both arrays are checked, so a refused second reservation commits neither.
    /// `O(capacity)`; spatial `O(capacity)` slots and `O(capacity)` coordinate words.
    fn reserve_empty(capacity: usize) -> Result<Self, StorageError> {
        let mut slots = Vec::new();
        try_reserve_exact(&mut slots, capacity)?;
        let mut coords = Vec::new();
        try_reserve_exact(&mut coords, capacity)?;
        Ok(Self {
            slots,
            coords,
            bits: 0,
            live: 0,
        })
    }

    /// Writes the vacant marker into every slot, filling both arrays to their reserved capacity.
    ///
    /// Temporal `O(capacity)` clones; spatial `O(capacity)`.
    fn initialize(&mut self, sigma: Sigma, held: &R, default: &V) {
        let capacity = self.slots.capacity();
        while self.slots.len() < capacity {
            self.slots.push(SpillSlot {
                tag: 0,
                sigma,
                held: held.clone(),
                value: default.clone(),
            });
        }
        while self.coords.len() < capacity {
            self.coords.push(COORD_EMPTY);
        }
        // The capacity is a power of two, so this counts the doublings that produce it: `bits` is
        // `log2(capacity)`, the shift width the multiply-shift home needs.
        let mut width = capacity;
        let mut bits = 0u32;
        while width > 1 {
            width /= 2;
            bits += 1;
        }
        self.bits = bits;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpillSite {
    Found(usize),
    Vacant(usize),
    Grow,
}
/// Classification returned by the unextracted composite probe walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchClass {
    /// The selected location holds the queried full logical key.
    Found,
    /// The selected location is the first vacant lane in probe order.
    Vacant,
    /// No indexed location is available; prepared spill growth is required.
    Exhausted,
}

/// Region and index selected by a [`SearchDecision`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchLocation {
    /// A shard-major resident-slot index.
    Flat(usize),
    /// A spill shard and its slot index.
    Spill(usize, usize),
    /// No index accompanies exhaustion.
    NoIndex,
}

/// A value-only result of the composite direct, window, and coverage-sweep probe walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchDecision {
    /// The probe classification.
    pub class: SearchClass,
    /// The location paired with [`Self::class`].
    pub location: SearchLocation,
}

impl SearchDecision {
    fn found_flat(index: usize) -> Self {
        Self {
            class: SearchClass::Found,
            location: SearchLocation::Flat(index),
        }
    }
    fn vacant_flat(index: usize) -> Self {
        Self {
            class: SearchClass::Vacant,
            location: SearchLocation::Flat(index),
        }
    }
    fn from_spill(shard: usize, site: SpillSite) -> Self {
        match site {
            SpillSite::Found(index) => Self {
                class: SearchClass::Found,
                location: SearchLocation::Spill(shard, index),
            },
            SpillSite::Vacant(index) => Self {
                class: SearchClass::Vacant,
                location: SearchLocation::Spill(shard, index),
            },
            SpillSite::Grow => Self {
                class: SearchClass::Exhausted,
                location: SearchLocation::NoIndex,
            },
        }
    }
}
/// Straight-line result of [`Table::apply_insert`].
///
/// `Committed` records an update or initialization of the indexed real slot.  `NeedsGrowth` returns
/// the pending entry unchanged so the caller can perform the resource step; neither variant
/// allocates, reserves or runs a probe loop.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ApplyOutcome<K, V> {
    /// The indexed real slot was updated or initialized.
    Committed,
    /// No indexed location exists; the caller must prepare growth for the returned pending entry.
    NeedsGrowth(K, V),
}

#[inline]
fn valid_n(n: usize) -> bool {
    n != 0 && n <= MAX_GROUP && n.is_power_of_two()
}
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    reason = "only the low shard byte is retained"
)]
pub(crate) fn shard(address: &TableAddress) -> usize {
    match address.direct_word {
        Some(word) => (word as usize) & SHARD_MASK,
        None => (address.place as usize) & SHARD_MASK,
    }
}
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the fingerprint shift leaves the top seven bits, so the u8 cast drops no retained \
              bit"
)]
fn tag(place: u64) -> u8 {
    let spread = place.wrapping_mul(TAG_SPREAD);
    TAG_LIVE | (((spread >> TAG_FINGERPRINT_SHIFT) as u8) & TAG_FINGERPRINT_MASK)
}
#[inline]
fn matches<K, H, R: KeyRep<K, H>, V>(
    slot: &SpillSlot<R, V>,
    sigma: Sigma,
    key: &K,
    address: &TableAddress,
) -> bool {
    slot.sigma == sigma && R::matches(sigma, &slot.held, key, address)
}

/// The capacity a region holding `load` live entries reserves, or `Ok(0)` for an empty region.
///
/// One flat open-addressed region needs a power-of-two capacity so the probe mask is
/// `capacity - 1`, and it needs at least one vacant slot, so the capacity is the least power of two
/// `>= 2 * load` and at least `2 * n`.  Every reserved region therefore satisfies
/// `capacity >= 2 * load`, and a region the insert path builds itself is reserved for the whole
/// group width, so it holds a vacancy at every load up to `n`: the load factor stays at or below
/// `1/2`, which is the classic linear-probing bound, under which an absent read stops at the first
/// vacancy after `1 / (1 - alpha)` slots on average instead of completing a whole cycle.
///
/// The floor is `2 * n` rather than `n` because `n` is a load a shard fills, not a capacity.  A
/// floor of `n` reserved `n` slots for the region's first entry, and the insert path then filled
/// those `n` slots: at load factor `1` the bounded read cycle has no vacancy to stop on, so every
/// absent read cost the region's whole capacity in tag compares, and the completed-cycle absence
/// certificate became reachable through [`Table::insert_with`] rather than only through a
/// deliberately filled region.
///
/// Temporal `O(log(load))` doublings; spatial `O(1)` words beyond the reserved region.
fn slots_for(load: usize, n: usize) -> Result<usize, StorageError> {
    if load == 0 {
        return Ok(0);
    }
    let twice = load.checked_mul(2).ok_or(StorageError::SizeOverflow)?;
    // `n.max(2)` keeps the zero-width floor of the original rule; `valid_n` admits `n = 1`, so
    // `2 * n` is a power of two above the smallest group the table accepts.
    let floor = n
        .max(2)
        .checked_mul(2)
        .ok_or(StorageError::SizeOverflow)?;
    let mut capacity = floor;
    while capacity < twice {
        capacity = capacity.checked_mul(2).ok_or(StorageError::SizeOverflow)?;
    }
    Ok(capacity)
}

/// Builds a [`CapacityBound`] from a checked target and the capacity the reservation observed.
///
/// The production scalar comparison runs here.  A successful reservation always meets its checked
/// target, so a violated bound is an internal invariant failure, not a new refusal case or a late
/// `Err` after cloning.
fn capacity_bound(required_len: usize, observed_capacity: usize) -> CapacityBound {
    assert!(
        capacity_bound_holds(required_len, observed_capacity),
        "reserved capacity is below the checked target"
    );
    CapacityBound {
        required_len,
        observed_capacity,
    }
}
/// A total-default table with flat certified resident cells and grouped spill regions.
struct FlatSlot<R, V> {
    live: bool,
    sigma: Sigma,
    held: R,
    value: V,
}

/// The resident slot index of `(lane, shard)`: `(shard << lane_bits) + lane`.
///
/// The incumbent computes `home(coord) = coord` below `len` and masks with `len - 1` because its
/// array is power-of-two (`fold hashtrinity/src/lib.rs:565-572`, `:617-618`). Under this flat's own
/// power-of-two lane count the same law holds with the lane as the low half and the shard as the
/// high half, which is a shift and an add in place of a multiply by the lane count and its
/// `checked_mul`/`checked_add` plumbing. The add is the or: `lane < lanes` is tested here and
/// `lanes == 1 << lane_bits` is the reservation's invariant, so the two halves are disjoint and the
/// sum carries into neither.
///
/// **No overflow check is needed.** `shard < SHARDS` and `lane < lanes` give
/// `index <= (SHARDS - 1) * lanes + lanes - 1 = lanes * SHARDS - 1`, and `Flat::reserve` established
/// `lanes * SHARDS` with `checked_mul`, so the sum fits `usize` and the shift amount is at most
/// `usize::BITS - 1`.
///
/// `None` unless `lane < lanes`, so a violated premise costs the spill route instead of a wrong
/// index; `lanes == 0` is rejected by that test. The power-of-two premise is [`Flat::reserve`]'s
/// refusal, not a test here: `lane_bits` is derived once from an admitted lane count, so it is
/// `log2(lanes)` wherever it is read.
#[inline]
fn flat_index(lane_bits: u32, lanes: usize, lane: usize, shard: usize) -> Option<usize> {
    if lane >= lanes {
        return None;
    }
    Some((shard << lane_bits) + lane)
}

/// The resident area: shard-major slot lanes, one index per `(shard, lane)` word.
struct Flat<R, V> {
    /// `0`, or a power of two. See [`Flat::reserve`].
    lanes: usize,
    /// `log2(lanes)`, derived once from the admitted lane count: the index law's shift amount.
    lane_bits: u32,
    /// The payload slots, in index order.
    slots: Vec<FlatSlot<R, V>>,
}

#[derive(Clone, Copy)]
enum RebuildSource {
    Resident(usize),
    Spill(usize, usize),
}

struct RebuildPlanEntry {
    source: RebuildSource,
    flat: Option<usize>,
    spill: usize,
}

pub(crate) struct RebuildPlan {
    lanes: usize,
    entries: Vec<RebuildPlanEntry>,
    spill_loads: Vec<usize>,
}

pub(crate) struct RebuildReserved<R, V> {
    resident: Flat<R, V>,
    regions: Vec<Region<R, V>>,
}

impl<R, V> Flat<R, V> {
    fn empty() -> Self {
        Self {
            lanes: 0,
            lane_bits: 0,
            slots: Vec::new(),
        }
    }
    /// The resident slot index of one direct placement word, or `None` when its lane is beyond
    /// this flat's extent.
    ///
    /// The lane admission test is kept because it is the growth trigger: a word whose lane exceeds
    /// the reserved extent is the `selected_lanes` widening signal, not a failed lookup.
    #[inline]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the lane and shard are the two halves of the same `u32` placement word, split by \
                  `SHARD_BITS`/`SHARD_MASK`; `lane < self.lanes` is tested inside `flat_index`"
    )]
    fn index(&self, word: u32) -> Option<usize> {
        let lane = (word >> SHARD_BITS) as usize;
        flat_index(self.lane_bits, self.lanes, lane, (word as usize) & SHARD_MASK)
    }
    fn byte_len_for(lanes: usize) -> Option<usize> {
        lanes.checked_mul(SHARDS)?.checked_mul(ASSUMED_SLOT_BYTES)
    }
}

impl<R: Clone, V: Clone> Flat<R, V> {
    fn reserve(lanes: usize) -> Result<Self, StorageError> {
        let count = lanes
            .checked_mul(SHARDS)
            .ok_or(StorageError::SizeOverflow)?;
        // The power-of-two invariant `flat_index`'s index law needs, refused on the checked
        // boundary rather than absorbed: `StorageError::SizeOverflow`'s own contract admits an
        // invalid group width, and a non-power-of-two lane count is one, since `log2(lanes)` is not
        // an integer and the shift would alias two admitted words onto one slot. `lanes` is `0` or a
        // power of two on the production path — `selected_lanes` (`src/map/table.rs:476-482`) returns a
        // doubling loop's result and `Flat::empty` is the only other constructor — so this arm is
        // unreachable from `Table` and is the located failure a violated invariant gets instead of
        // a wrong index. `lane_bits` is derived here, once, so the read path's index law is a shift
        // and an add.
        if !lanes.is_power_of_two() {
            return Err(StorageError::SizeOverflow);
        }
        let mut slots = Vec::new();
        try_reserve_exact(&mut slots, count)?;
        Ok(Self {
            lanes,
            lane_bits: lanes.trailing_zeros(),
            slots,
        })
    }

    fn initialize(&mut self, sigma: Sigma, held: &R, value: &V) {
        let count = self.lanes * SHARDS;
        while self.slots.len() < count {
            self.slots.push(FlatSlot {
                live: false,
                sigma,
                held: held.clone(),
                value: value.clone(),
            });
        }
    }
}

/// Where one linked representative's payload lives in the table's own storage.
///
/// This names a location; it is not a second store. The key and value remain in the named flat or
/// spill slot, so the directory still owns no key and no value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PayloadLocation {
    /// A live resident flat slot.
    Resident(usize),
    /// A live slot `(shard, index)` in a spill region.
    Spill(usize, usize),
}

/// The handle-only directory state attached to [`Table`].
///
/// The shared [`Directory`] owns no key and no value: terminal handle `i` names the payload at
/// `links[i].1`, and `links[i].0` is the terminal identity node that handle must occupy. The state
/// therefore records reverse links, not a duplicate key/value map. `staged` is the private staged
/// destination of an in-flight write; while it is `Some`, no coverage certificate may be issued.
pub(crate) struct TableLocator {
    /// The sealed hyperbolic profile used to rebuild a hyperbolic payload's residue path.
    profile: Option<H3_40>,
    /// The shared directory engine, or `None` for the allocation-free empty state.
    directory: Option<Directory>,
    /// Reverse links `(identity_node, payload_location)` indexed by terminal handle.
    ///
    /// The registry is built by the cold [`Table::assemble_directory`] pass from the table's own
    /// live payloads; a merge does not push a link. It stays empty until that pass runs, so a fresh
    /// insert writes only the payload stores.
    links: Vec<(u32, PayloadLocation)>,
    /// The number of representatives admitted against the reserved link extent.
    ///
    /// This is [`Self::links`]'s admitted count while the registry is not materialized: a merge
    /// increments it into the extent [`TableLocator::reserve_batch_paths`] reserved, and refuses
    /// when the extent is exhausted. [`Table::assemble_directory`] resets it to the rebuilt
    /// registry length.
    admitted: usize,
    /// The staged destination of a pending write, if any.
    staged: Option<(u32, StagedPath)>,
}

impl TableLocator {
    /// The empty state: no directory, no links, no staging and no allocation.
    pub(crate) const fn new() -> Self {
        Self {
            profile: None,
            directory: None,
            links: Vec::new(),
            admitted: 0,
            staged: None,
        }
    }
}
