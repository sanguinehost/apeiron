//! The acceptance test: can a real codebase's key shapes go in without hand-written glue?
//!
//! Every type and every collection shape below was taken from a survey of Lyrium, which is the
//! codebase this crate is meant to serve. Nothing here is invented to be easy:
//!
//! | shape | how many, there | covered by |
//! |---|---|---|
//! | `newtype(u32)` handles | 33 declarations, 4 types | `derive(Coord, Packed)`, elliptic face |
//! | `newtype(String)` | 1 | `derive(Coord)`, parabolic face |
//! | `(String, u16)` composite | 6, every VFS archive index | `Locate` for `(&str, u16)` |
//! | `String` keys | 47 | the borrowed surface |
//! | `u32 -> String` at TLK scale | 1, up to a million entries | `Words` |
//! | `HashSet<u32>` / `HashSet<String>` | 8 + ~8 inferred | `Set`, `WordSet` |
//! | `BTreeSet<u32>` for deterministic order | 13 | `WordSet` iteration |
//!
//! **If this file compiles, the port needs no glue.** That is the whole claim, and it is worth
//! testing as one thing rather than as seven, because the failure it guards against is not a bug --
//! it is discovering halfway through a migration that one shape needs an impl nobody can write.

#![expect(
    clippy::cast_possible_truncation,
    reason = "this is a test, not library code, and each lint below is listed because this file actually \
              triggers it: sizes and indices narrowed to build keys, bounded by the loop that produced them. \
              Listed individually rather than suppressed wholesale, so a pedantic lint this file does not \
              yet trigger still surfaces"
)]
#![cfg(feature = "derive")]

use apeiron::coord::Coord as _;
use apeiron::map::Exposed;
use apeiron::set::{Set, WordSet};
use apeiron::{Coord, Map, Packed, Words};
use std::num::NonZeroU64;

// -- the five user-defined key types, verbatim in shape -------------------------------------------

/// `BlockId`, `InstId`, `ValueId`, `FunctionId` are all exactly this, and account for 33 of the 34
/// user-keyed map declarations. `Packed` is what puts them on the elliptic face.
#[derive(Coord, Packed, PartialEq, Eq, Clone, Copy, Debug, PartialOrd, Ord)]
struct BlockId(u32);

#[derive(Coord, Packed, PartialEq, Eq, Clone, Copy, Debug)]
struct ValueId(u32);

/// The fifth: a `String` newtype, deliberately case-sensitive where it came from.
#[derive(Coord, PartialEq, Eq, Clone, Debug)]
struct VPath(String);

/// A foreign key type -- `bevy::Entity` -- cannot be derived on, so this is the documented pattern:
/// wrap it, and forward to the `u64` it already knows how to make of itself.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
struct EntityKey(u64);
impl apeiron::coord::Coord for EntityKey {
    fn coord(&self) -> u64 {
        self.0
    }
}
impl apeiron::coord::Packed for EntityKey {
    fn packed(&self) -> u64 {
        self.0
    }
    fn unpack(c: u64) -> Self {
        Self(c)
    }
}

/// The resource-type tag half of a VFS key.
type ResType = u16;

/// Miri interprets rather than runs, so a hundred thousand entries is minutes rather than
/// milliseconds. The shapes are what this file is about -- every one of them is exercised at a
/// fortieth the size, and native keeps the real scale.
#[cfg(miri)]
const SCALE: usize = 40;
#[cfg(not(miri))]
const SCALE: usize = 1;

#[test]
fn the_ir_handles_run_on_the_elliptic_face_storing_no_key() {
    // `HashMap<BlockId, CfgNode>` and friends: 33 declarations across the decompiler IR.
    let mut preds: Words<BlockId, Vec<BlockId>> = Words::new();
    for b in 0..(5_000 / SCALE) as u32 {
        preds.insert(BlockId(b), vec![BlockId(b.saturating_sub(1))]);
    }
    assert_eq!(preds.len(), 5_000 / SCALE);
    assert_eq!(preds.face(), apeiron::Sigma::Elliptic);
    assert_eq!(
        preds
            .get(&BlockId((5_000 / SCALE) as u32 - 1))
            .map(Vec::len),
        Some(1)
    );
    assert_eq!(preds.get(&BlockId((5_000 / SCALE) as u32)), None);

    // The key really does come back off the address -- nothing else is stored.
    let mut ids: Vec<BlockId> = preds.keys().collect();
    ids.sort_unstable();
    assert_eq!(ids.first(), Some(&BlockId(0)));
    assert_eq!(ids.last(), Some(&BlockId((5_000 / SCALE) as u32 - 1)));

    // A second handle type is a different type, so it cannot be looked up in the first map by
    // accident even though both are newtypes over u32 with the same address.
    let mut values: Words<ValueId, u32> = Words::new();
    values.insert(ValueId(7), 1);
    assert_eq!(values.get(&ValueId(7)), Some(&1));
    assert_eq!(BlockId(7).coord(), ValueId(7).coord()); // same address, different tables

    // And the foreign-type wrapper.
    let mut ecs: Words<EntityKey, &str> = Words::new();
    ecs.insert(EntityKey(0x0000_0001_0000_002A), "door");
    assert_eq!(ecs.get(&EntityKey(0x0000_0001_0000_002A)), Some(&"door"));
}

#[test]
fn the_vfs_composite_index_probes_without_allocating() {
    // `HashMap<(String, u16), ResourceLocation>` -- six of these, one per archive kind, each built
    // once at mount and probed on every resource lookup for the life of the process.
    let mut index: Map<(String, ResType), (u32, usize)> = Map::new();
    for i in 0..(20_000 / SCALE) as u32 {
        index.insert((format!("nw_wswmg{i:05}"), 2_002), (i, i as usize * 16));
    }

    // The probe a resource manager actually writes. No `String` is materialised, and the resource
    // type stays the bare `u16` it is at the call site -- `(&str, &u16)` would compile and would
    // read like a mistake.
    assert_eq!(index.get_at(("nw_wswmg00042", 2_002u16)), Some(&(42, 672)));
    assert_eq!(index.get_at(("nw_wswmg00042", 2_003u16)), None);
    assert_eq!(index.get_at(("nw_missing", 2_002u16)), None);

    // And the owned surface sees the same table.
    assert_eq!(
        index.get(&("nw_wswmg00042".to_string(), 2_002)),
        Some(&(42, 672))
    );
    assert_eq!(index.len(), 20_000 / SCALE);
}

#[test]
fn the_talk_table_holds_a_hundred_thousand_strings() {
    // `HashMap<u32, String>`, keyed by StrRef. Non-sequential in DA:O -- 0..274K and 5M+ ranges --
    // so this is deliberately sparse rather than dense.
    let half = (60_000 / SCALE) as u32;
    let mut tlk: Words<u32, String> = Words::with_capacity(2 * half as usize);
    for i in 0..half {
        tlk.insert(i, format!("line {i}"));
    }
    for i in 0..half {
        tlk.insert(5_000_000 + i * 3, format!("dlc line {i}"));
    }
    assert_eq!(tlk.len(), 2 * half as usize);
    assert_eq!(
        tlk.get(&(half - 1)).map(String::as_str),
        Some(format!("line {}", half - 1).as_str())
    );
    assert_eq!(tlk.get(&5_000_003).map(String::as_str), Some("dlc line 1"));
    assert_eq!(tlk.get(&5_000_001), None);

    // The probe stays short across a sparse, structured key set -- which is the thing a bad address
    // function would ruin here and nowhere else.
    let s = tlk.stats();
    assert!(
        s.longest_probe < 32,
        "sparse StrRefs degraded the probe: {s:?}"
    );
}

#[test]
fn a_path_newtype_and_the_string_maps() {
    let mut paths: Map<VPath, u64> = Map::new();
    for i in 0..(5_000 / SCALE) as u32 {
        paths.insert(VPath(format!("data/meshes/{i}.nif")), u64::from(i));
    }
    let last = (5_000 / SCALE) as u32 - 1;
    assert_eq!(
        paths.get(&VPath(format!("data/meshes/{last}.nif"))),
        Some(&u64::from(last))
    );
    // Case-sensitive, as its source is.
    assert_eq!(paths.get(&VPath(format!("DATA/MESHES/{last}.NIF"))), None);

    // The 47 plain `HashMap<String, _>` declarations: GFF field tables, 2DA columns, MDL nodes.
    let mut fields: Map<String, i64> = Map::new();
    for name in [
        "Tag",
        "TemplateResRef",
        "LocName",
        "Appearance_Type",
        "FactionID",
    ] {
        fields.insert(name.to_string(), 1);
    }
    *fields.entry_at("Tag").or_insert(0) += 41;
    assert_eq!(fields.get_at("Tag"), Some(&42));
    assert_eq!(fields["Tag"], 42); // Index, so `map[&k]` reads as it does with std
}

#[test]
fn the_sets_cover_what_the_sets_are_for() {
    // `HashSet<u32>` -- declared_params, declared_values, expanded_nodes, block_starts.
    let mut declared: WordSet<u32> = WordSet::new();
    for i in 0..(2_000 / SCALE) as u32 {
        assert!(declared.insert(i));
    }
    assert!(!declared.insert(0));
    assert!(declared.contains(&((2_000 / SCALE) as u32 - 1)));

    // `HashSet<ValueId>` -- hoisted_values.
    let mut hoisted: WordSet<ValueId> = WordSet::new();
    hoisted.insert(ValueId(3));
    assert!(hoisted.contains(&ValueId(3)));

    // `HashSet<String>` -- declared_names.
    let mut names: Set<String> = Set::new();
    assert!(names.insert_at("nX_0001"));
    assert!(!names.insert_at("nX_0001"));
    assert!(names.contains_at("nX_0001"));

    // `BTreeSet<u32>` -- 13 of the 14, holding jump targets and block ids, ordered so the
    // decompiler's output can be fingerprinted in a test. That is determinism, not ordering, and
    // this gives it without paying for a tree.
    let targets = || {
        (0..(500 / SCALE) as u32)
            .map(|i| i.wrapping_mul(2_654_435_761) % 10_000)
            .collect::<WordSet<u32>>()
    };
    let a: Vec<u32> = targets().into_iter().collect();
    let b: Vec<u32> = targets().into_iter().collect();
    assert_eq!(a, b, "iteration order moved between two identical builds");
    assert_eq!(a.len(), (500 / SCALE));
}

#[test]
fn the_untrusted_indexes_are_peppered_by_construction() {
    // Every VFS index is keyed by names out of an archive header, and `GffStruct::fields` by labels
    // read verbatim from a module file. A module author picks those keys and how many there are.
    // `Exposed` is how that gets said at the declaration instead of in a comment.
    let pepper = NonZeroU64::new(0x5EED_1234_ABCD_9876).expect("nonzero");
    let mut index: Exposed<(String, ResType), u32> = Exposed::new(pepper);
    for i in 0..(10_000 / SCALE) as u32 {
        index.insert((format!("res{i:05}"), 2_002), i);
    }
    let last = (10_000 / SCALE) as u32 - 1;
    assert_eq!(
        index.get_at((format!("res{last:05}").as_str(), 2_002u16)),
        Some(&last)
    );
    assert!(index.stats().peppered);
    assert!(!index.stats().looks_aimed());

    // A GFF field table, the other untrusted shape.
    let mut fields: Exposed<String, i64> = Exposed::new(pepper);
    fields.insert("Tag".into(), 1);
    assert_eq!(fields.get_at("Tag"), Some(&1));
    assert!(fields.stats().peppered);
}

/// The whole surface a mechanical port reaches for, in one place. This compiling is the point.
#[test]
fn the_std_shaped_surface_is_all_there() {
    let mut m: Map<String, u32> = Map::from([("a".into(), 1), ("b".into(), 2), ("c".into(), 3)]);
    assert_eq!(m.len(), 3);
    assert_eq!(m["a"], 1);

    assert_eq!(m.get_at("a"), Some(&1));
    assert_eq!(
        m.get_key_value(&"a".to_string())
            .map(|(k, v)| (k.as_str(), *v)),
        Some(("a", 1))
    );
    *m.get_mut_at("a").unwrap() += 10;
    assert_eq!(m["a"], 11);

    m.entry("d".into()).or_insert(4);
    m.entry("d".into()).and_modify(|v| *v += 1).or_insert(0);
    assert_eq!(m["d"], 5);
    assert_eq!(m.entry("d".into()).key(), "d");

    for (_, v) in &mut m {
        *v += 100;
    }
    assert_eq!((&m).into_iter().count(), 4);
    assert_eq!(m.keys().count(), 4);
    assert_eq!(m.values().copied().sum::<u32>(), 111 + 102 + 103 + 105);
    m.values_mut().for_each(|v| *v -= 100);

    let [x, y] = m.get_disjoint_mut([&"a".to_string(), &"b".to_string()]);
    *x.unwrap() += 1;
    *y.unwrap() += 1;
    assert_eq!(m["a"], 12);

    let taken: Vec<(String, u32)> = m.extract_if(|k, _| k == "c").collect();
    assert_eq!(taken.len(), 1);

    m.reserve(100);
    assert!(m.try_reserve(100).is_ok());
    m.shrink_to_fit();
    assert!(m.capacity() >= m.len());

    let keys: Vec<String> = m.clone().into_keys().collect();
    let vals: Vec<u32> = m.clone().into_values().collect();
    assert_eq!(keys.len(), 3);
    assert_eq!(vals.len(), 3);

    assert_eq!(m.drain().count(), 3);
    assert!(m.is_empty());

    m.extend([("z".to_string(), 9)]);
    assert_eq!(m.remove_entry(&"z".to_string()), Some(("z".into(), 9)));

    // And `Debug` shows the entries, so a failing assert says what was in it.
    let d: Map<u32, u32> = Map::from([(1, 2)]);
    assert_eq!(format!("{d:?}"), "{1: 2}");
    assert!(d.summary().contains("1 live"));
}
