//! What `#[derive(Coord)]` and `#[derive(Packed)]` have to get right.
//!
//! The contract is `Hash`'s contract, and it is the only one that matters: **`a == b` must imply
//! `a.coord() == b.coord()`**. Break it and a lookup misses a key that is present, silently, for
//! one value in the corpus. Everything else here is either that property under a different shape,
//! or the claim that a derived struct addresses the same way the equivalent tuple does -- which is
//! what makes the generated code reviewable at all.

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
        for i in 0..200u32 {
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
        for i in 0..300u32 {
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
    for i in 0..1000u32 {
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
    for i in (0..u32::MAX).step_by(7919) {
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
    let mut seen = HashSet::new();
    for a in 0..60u32 {
        for i in 0..60u32 {
            for k in 0..6u8 {
                assert!(
                    seen.insert(ObjectId { area: a, index: i, kind: k }.coord()),
                    "collision at ({a}, {i}, {k})"
                );
            }
        }
    }

    let mut seen = HashSet::new();
    for i in 0..5000u32 {
        assert!(seen.insert(Wide(i, i * 3, i as u16, (i % 251) as u8, format!("s{i}")).coord()));
    }
}

/// Two variants carrying the same payload are different keys, so the discriminant has to reach the
/// address. Folding only the fields would make `Indexed(0)` and a zero-ish `Named` collide.
#[test]
fn enum_variants_do_not_share_addresses() {
    let mut seen = HashSet::new();
    assert!(seen.insert(Slot::Empty.coord()));
    for i in 0..500u32 {
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
    let mut ids: Words<BlockId, &str> = Words::new();
    for i in 0..5000u32 {
        ids.insert(BlockId(i), "block");
    }
    assert_eq!(ids.len(), 5000);
    assert_eq!(ids.get(&BlockId(4999)), Some(&"block"));
    assert_eq!(ids.get(&BlockId(5000)), None);
    assert_eq!(ids.face(), apeiron::Keystore::Being);
    // The key really does come back off the address alone.
    assert!(ids.keys().any(|k| k == BlockId(123)));

    // Parabolic: `VPath` wraps a `String`, so the key rides along and is compared.
    let mut paths: Map<VPath, u32> = Map::new();
    for i in 0..2000u32 {
        paths.insert(VPath(format!("data/{i}.mdl")), i);
    }
    assert_eq!(paths.get(&VPath("data/1999.mdl".into())), Some(&1999));
    assert_eq!(paths.face(), apeiron::Keystore::BeingOverUnbeing);

    // Structs and enums, through the entry API.
    let mut objs: Map<ObjectId, u32> = Map::new();
    for a in 0..40u32 {
        for i in 0..40u32 {
            *objs.entry(ObjectId { area: a, index: i, kind: 3 }).or_insert(0) += 1;
        }
    }
    assert_eq!(objs.len(), 1600);
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
