//! What `#[derive(Coord)]` and `#[derive(Packed)]` have to get right.
//!
//! The contract is `Hash`'s contract, and it is the only one that matters: **`a == b` must imply
//! `a.coord() == b.coord()`**. Break it and a lookup misses a key that is present, silently, for
//! one value in the corpus. Everything else here is either that property under a different shape,
//! or the claim that a derived struct addresses the same way the equivalent tuple does -- which is
//! what makes the generated code reviewable at all.

#![expect(
    clippy::cast_possible_truncation,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    reason = "this is a test, not library code, and each lint below is listed because this file actually \
              triggers it: sizes and indices narrowed to build keys, bounded by the loop that produced them; \
              constants declared next to the loop that uses them; the algebra's own names. Listed \
              individually rather than suppressed wholesale, so a pedantic lint this file does not yet \
              trigger still surfaces"
)]

#![cfg(feature = "derive")]

use apeiron::coord::{Coord, Packed};
use apeiron::map::{Map, Words};
use apeiron::{Coord, Packed};
use std::collections::HashSet;

// -- the shapes the real corpus actually uses ----------------------------------------------------

/// Four of five user-defined key types in the codebase this was aimed at are exactly this.
#[derive(Coord, Packed, PartialEq, Eq, Clone, Copy, Debug, Hash)]
struct BlockId(u32);

/// The fifth.
#[derive(Coord, PartialEq, Eq, Clone, Debug, Hash)]
struct VPath(String);

#[derive(Coord, PartialEq, Eq, Clone, Debug, Hash)]
struct ObjectId {
    area: u32,
    index: u32,
    kind: u8,
}

#[derive(Coord, PartialEq, Eq, Clone, Debug, Hash)]
struct Wide(u32, u32, u16, u8, String);

#[derive(Coord, PartialEq, Eq, Clone, Debug, Hash)]
struct Unit;

#[derive(Coord, PartialEq, Eq, Clone, Debug, Hash)]
enum Slot {
    Empty,
    Indexed(u32),
    Named { area: String, n: u16 },
}

#[derive(Coord, PartialEq, Eq, Clone, Debug, Hash)]
struct Boxed<T> {
    inner: T,
    tag: u8,
}

/// A field `Eq` ignores must be a field the address ignores too, or the contract breaks.
#[derive(Coord, Clone, Debug)]
struct Cached {
    id: u32,
    #[coord(skip)]
    #[allow(dead_code)]
    memo: u64,
}
impl PartialEq for Cached {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Cached {}

// -- the contract --------------------------------------------------------------------------------

/// Equal keys, equal addresses. Checked under several peppers, because `coord_with` is a second
/// code path and a derive that threaded the pepper into only some fields would pass at zero.
#[test]
fn equal_keys_have_equal_addresses() {
    let peppers = [0u64, 1, 0x5EED_1234_ABCD_9876, u64::MAX];
    for p in peppers {
        for i in 0..if cfg!(miri) { 25 } else { 200u32 } {
            let a = ObjectId { area: i, index: i * 3, kind: (i % 7) as u8 };
            let b = a.clone();
            assert_eq!(a.coord_with(p), b.coord_with(p));

            let a = Slot::Named { area: format!("area{i}"), n: i as u16 };
            assert_eq!(a.coord_with(p), a.clone().coord_with(p));

            // The skipped field must not reach the address, or `Eq` and `coord` disagree.
            let x = Cached { id: i, memo: 0 };
            let y = Cached { id: i, memo: u64::MAX };
            assert_eq!(x, y);
            assert_eq!(x.coord_with(p), y.coord_with(p), "a skipped field leaked into the address");
        }
    }
}

/// A derived struct must address the same way the equivalent tuple does. This is what makes the
/// generated fold reviewable: it is not a second scheme, it is the one already in `coord.rs`.
#[test]
fn a_derived_struct_is_the_tuple_it_is_made_of() {
    for p in [0u64, 0x5EED] {
        for i in 0..if cfg!(miri) { 40 } else { 300u32 } {
            let s = ObjectId { area: i, index: i.wrapping_mul(7), kind: (i % 251) as u8 };
            let t = (s.area, s.index, s.kind);
            assert_eq!(s.coord_with(p), t.coord_with(p), "3-field struct vs 3-tuple");
        }
    }
}

/// A newtype is its field, with no fold in between. The table avalanches afterwards regardless, so
/// an extra `finish` here would be a multiply and two shifts spent to no end.
#[test]
fn a_newtype_is_its_field() {
    for i in 0..if cfg!(miri) { 60 } else { 1000u32 } {
        assert_eq!(BlockId(i).coord(), i.coord());
        assert_eq!(BlockId(i).coord_with(99), i.coord_with(99));
    }
    for s in ["", "a", "resref_0001", "a rather longer path/into/the/tree.mdl"] {
        assert_eq!(VPath(s.into()).coord(), s.to_string().coord());
    }
}

/// `Packed` promises an exact round trip. A newtype forwards, so it inherits one -- and that is
/// what lets these keys run on the elliptic face and store no key at all.
#[test]
fn a_derived_packed_newtype_round_trips_exactly() {
    // 542,000 round trips natively; Miri interprets each one. A coarser stride still walks the
    // whole `u32` range, which is what the round trip has to hold over.
    let stride = if cfg!(miri) { 7_919_000 } else { 7919 };
    for i in (0..u32::MAX).step_by(stride) {
        assert_eq!(BlockId::unpack(BlockId(i).packed()), BlockId(i));
    }
    for i in [0u32, 1, u32::MAX, u32::MAX - 1] {
        assert_eq!(BlockId::unpack(BlockId(i).packed()), BlockId(i));
    }
}

// -- distinctness ---------------------------------------------------------------------------------

/// Distinct keys should get distinct addresses. Not a guarantee -- a 64-bit address of a
/// `String`-carrying struct is a discard and has fibers by counting -- but a derive that dropped a
/// field, or folded the fields into each other, would collide here in the thousands rather than
/// not at all.
#[test]
fn distinct_keys_get_distinct_addresses() {
    // 60 x 60 x 6 is 21,600 folds; Miri does not need the whole grid to see a collision.
    const GRID: u32 = if cfg!(miri) { 12 } else { 60 };
    let mut seen = HashSet::new();
    for a in 0..GRID {
        for i in 0..GRID {
            for k in 0..6u8 {
                assert!(
                    seen.insert(ObjectId { area: a, index: i, kind: k }.coord()),
                    "collision at ({a}, {i}, {k})"
                );
            }
        }
    }

    let mut seen = HashSet::new();
    for i in 0..if cfg!(miri) { 150 } else { 5000u32 } {
        assert!(seen.insert(Wide(i, i * 3, i as u16, (i % 251) as u8, format!("s{i}")).coord()));
    }
}

/// Two variants carrying the same payload are different keys, so the discriminant has to reach the
/// address. Folding only the fields would make `Indexed(0)` and a zero-ish `Named` collide.
#[test]
fn enum_variants_do_not_share_addresses() {
    let mut seen = HashSet::new();
    assert!(seen.insert(Slot::Empty.coord()));
    for i in 0..if cfg!(miri) { 50 } else { 500u32 } {
        assert!(seen.insert(Slot::Indexed(i).coord()), "Indexed({i})");
        assert!(
            seen.insert(Slot::Named { area: format!("a{i}"), n: i as u16 }.coord()),
            "Named({i})"
        );
    }
    // And the unit variant is not just zero.
    assert_ne!(Slot::Empty.coord(), 0);
    assert_ne!(Slot::Empty.coord(), Slot::Indexed(0).coord());
}

/// The bound goes on the field type, not on the type parameter: `Boxed<Vec<u8>>` needs
/// `Vec<u8>: Coord`, which a blanket `T: Coord` neither implies nor is implied by.
#[test]
fn a_generic_key_bounds_its_fields() {
    let a = Boxed { inner: 7u32, tag: 1 };
    let b = Boxed { inner: 7u32, tag: 2 };
    assert_ne!(a.coord(), b.coord());

    let c = Boxed { inner: "res".to_string(), tag: 1 };
    let d = Boxed { inner: "res".to_string(), tag: 1 };
    assert_eq!(c.coord(), d.coord());

    let e = Boxed { inner: vec![1u8, 2, 3], tag: 9 };
    assert_ne!(e.coord(), Boxed { inner: vec![1u8, 2, 4], tag: 9 }.coord());
}

#[test]
fn a_unit_struct_has_a_stable_address() {
    assert_eq!(Unit.coord(), Unit.coord());
    assert_ne!(Unit.coord(), 0);
}

// -- and they work as keys -------------------------------------------------------------------------

/// The point of all of it: these types go into tables without a line of hand-written glue.
#[test]
fn derived_keys_work_on_both_faces() {
    // Elliptic: `BlockId` is `Packed`, so no key is stored and none is ever compared.
    const N: u32 = if cfg!(miri) { 150 } else { 5000 };
    let mut ids: Words<BlockId, &str> = Words::new();
    for i in 0..N {
        ids.insert(BlockId(i), "block");
    }
    assert_eq!(ids.len(), N as usize);
    assert_eq!(ids.get(&BlockId(N - 1)), Some(&"block"));
    assert_eq!(ids.get(&BlockId(N)), None);
    assert_eq!(ids.face(), apeiron::Sigma::Elliptic);
    // The key really does come back off the address alone.
    assert!(ids.keys().any(|k| k == BlockId(123)));

    // Parabolic: `VPath` wraps a `String`, so the key rides along and is compared.
    let mut paths: Map<VPath, u32> = Map::new();
    const PATHS: u32 = if cfg!(miri) { 80 } else { 2000 };
    for i in 0..PATHS {
        paths.insert(VPath(format!("data/{i}.mdl")), i);
    }
    let last = PATHS - 1;
    assert_eq!(paths.get(&VPath(format!("data/{last}.mdl"))), Some(&last));
    assert_eq!(paths.face(), apeiron::Sigma::Parabolic);

    // Structs and enums, through the entry API.
    const OBJS: u32 = if cfg!(miri) { 10 } else { 40 };
    let mut objs: Map<ObjectId, u32> = Map::new();
    for a in 0..OBJS {
        for i in 0..OBJS {
            *objs.entry(ObjectId { area: a, index: i, kind: 3 }).or_insert(0) += 1;
        }
    }
    assert_eq!(objs.len(), (OBJS * OBJS) as usize);
    assert_eq!(objs.get(&ObjectId { area: 7, index: 9, kind: 3 }), Some(&1));
    assert_eq!(objs.get(&ObjectId { area: 7, index: 9, kind: 4 }), None);

    let mut slots: Map<Slot, u32> = Map::new();
    slots.insert(Slot::Empty, 0);
    slots.insert(Slot::Indexed(7), 1);
    slots.insert(Slot::Named { area: "a".into(), n: 2 }, 2);
    assert_eq!(slots.len(), 3);
    assert_eq!(slots.get(&Slot::Indexed(7)), Some(&1));
    assert_eq!(slots.get(&Slot::Indexed(8)), None);
}

/// **The derive propagates `Coord::SPREAD`.**
///
/// It did not, and the consequence was silent: a derived newtype over `String` took the `false`
/// default, so `already_mixed` kept the redundant widening multiply that a hand-written
/// `impl Coord for String` skips. Nothing failed -- the addresses were fine, the probe was flat,
/// the tests passed, and every lookup paid for a second mix it did not need. `VPath(String)` is one
/// of the five real key shapes this crate was aimed at.
#[test]
fn a_derived_newtype_inherits_whether_its_fold_is_already_mixed() {
    // The runtime half: the derived const must agree with the field it came from.
    assert_eq!(<VPath as Coord>::SPREAD, <String as Coord>::SPREAD, "VPath must agree with String");
    assert_eq!(<BlockId as Coord>::SPREAD, <u32 as Coord>::SPREAD, "BlockId must agree with u32");
}

// And the values themselves, as **compile-time** assertions, because they are constants and a
// runtime check of a constant is weaker than the compiler already is. A derive that stopped
// propagating `SPREAD` would fail the build rather than a test run.
//
// A newtype over a byte fold is already mixed and must say so; over an integer it is not, and the
// spread is the only thing mixing it *and* the only thing carrying its pepper.
const _: () = assert!(<VPath as Coord>::SPREAD, "a String newtype should skip the second mix");
const _: () = assert!(!<BlockId as Coord>::SPREAD, "a u32 newtype must keep the spread");
// Composites take the conjunction, so all-integer structs and enums keep it, and a mixed one is
// conservative rather than clever.
const _: () = assert!(!<ObjectId as Coord>::SPREAD);
const _: () = assert!(!<Slot as Coord>::SPREAD);
const _: () = assert!(!<Wide as Coord>::SPREAD);
// A unit struct folds to a constant, which no pepper reaches.
const _: () = assert!(!<Unit as Coord>::SPREAD);

/// The obligation that comes with claiming it: the pepper must actually reach the fold, or skipping
/// the spread drops it silently. `tests/the_faces_keep_their_promises.rs` states this for the
/// hand-written impls; this is the derived half.
#[test]
fn a_derived_type_that_claims_spread_consumes_the_pepper() {
    fn obliged<T: Coord>(label: &str, k: &T) {
        if !T::SPREAD {
            return;
        }
        let mut seen = std::collections::HashSet::new();
        for p in [0u64, 1, 0x5EED_1234_ABCD_9876, u64::MAX] {
            seen.insert(k.coord_with(p));
        }
        assert_eq!(seen.len(), 4, "{label} claims SPREAD but ignores the pepper");
    }
    obliged("VPath", &VPath("data/meshes/c_human.mdl".into()));
    obliged("BlockId", &BlockId(7));
    obliged("ObjectId", &ObjectId { area: 1, index: 2, kind: 3 });
}

/// **The derive propagates `Coord::CHEAP_EQ`.**
///
/// The identity word beside a stored key is an early reject in front of the key compare, and it
/// only pays when the compare it skips is a `memcmp`. Measured, `Map<K, V>::contains_key`: a `u64`
/// key costs 34.82 instructions and 7.28 cycles with the reject and 32.82 and 6.14 without, while a
/// 42-byte shared-prefix `String` costs 131.65 and **33.73** with it against 129.73 and 34.18
/// without. So the answer is the key type's, and a derived newtype should inherit it rather than
/// take the conservative default and pay a cycle it does not owe.
///
/// Unlike `SPREAD` this is a hint and not an obligation: `Stored::holds` decides on the key
/// whatever the identity words say -- `proofs::the_parabolic_face_decides_on_the_key` -- so a wrong
/// value here costs instructions and never correctness.
#[test]
fn a_derived_newtype_inherits_whether_its_key_is_cheap_to_compare() {
    assert_eq!(
        <VPath as Coord>::CHEAP_EQ,
        <String as Coord>::CHEAP_EQ,
        "VPath must agree with String"
    );
    assert_eq!(
        <BlockId as Coord>::CHEAP_EQ,
        <u32 as Coord>::CHEAP_EQ,
        "BlockId must agree with u32"
    );
}

// The same values as compile-time assertions, for the same reason.
const _: () = assert!(<BlockId as Coord>::CHEAP_EQ, "a u32 newtype compares in one instruction");
const _: () = assert!(!<VPath as Coord>::CHEAP_EQ, "a String newtype does not");
// Conjunction over the fields: all-integer composites are cheap, and one byte field is enough to
// make the whole thing not -- `Wide`'s trailing `String` is what decides it.
const _: () = assert!(<ObjectId as Coord>::CHEAP_EQ);
const _: () = assert!(!<Wide as Coord>::CHEAP_EQ);
const _: () = assert!(!<Slot as Coord>::CHEAP_EQ);
// Zero fields claims nothing: there is no compare for the hint to sit in front of.
const _: () = assert!(!<Unit as Coord>::CHEAP_EQ);
