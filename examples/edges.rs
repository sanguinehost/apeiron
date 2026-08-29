//! Edges a port hits on day one.
#![expect(
    clippy::cast_possible_truncation,
    clippy::unreadable_literal,
    reason = "this is a measurement harness, not library code, and each lint below is listed because this \
              file actually triggers it: sizes and indices narrowed to build keys, bounded by the loop that \
              produced them; a constant quoted from elsewhere. Listed individually rather than suppressed \
              wholesale, so a pedantic lint this file does not yet trigger still surfaces"
)]

use apeiron::coord::Coord;
use apeiron::map::{Map, Words};

/// What a user type costs today. There is no `derive(Coord)`, so this is hand-written.
#[derive(PartialEq, Eq, Clone, Debug)]
struct ObjectId { area: u32, index: u32, kind: u8 }

impl Coord for ObjectId {
    fn coord(&self) -> u64 { (self.area, self.index, self.kind).coord() }
    fn coord_with(&self, p: u64) -> u64 { (self.area, self.index, self.kind).coord_with(p) }
}

fn main() {
    // 1. zero-sized values (a set, essentially)
    let mut set: Map<String, ()> = Map::new();
    for i in 0..1000u32 { set.insert(format!("k{i}"), ()); }
    assert_eq!(set.len(), 1000);
    assert!(set.contains_at("k500"));
    println!("zero-sized values           ok  ({} slots)", set.slots());

    // 2. a user key type
    let mut m: Map<ObjectId, &str> = Map::new();
    for a in 0..40u32 { for i in 0..40u32 {
        m.insert(ObjectId { area: a, index: i, kind: 3 }, "door");
    }}
    assert_eq!(m.len(), 1600);
    assert_eq!(m.get(&ObjectId { area: 7, index: 9, kind: 3 }), Some(&"door"));
    assert_eq!(m.get(&ObjectId { area: 7, index: 9, kind: 4 }), None);
    println!("user key type               ok  (hand-written Coord, {} lines)", 2);

    // 3. big values
    let mut big: Map<u64, [u8; 512]> = Map::new();
    for i in 0..5000u64 { big.insert(i, [i as u8; 512]); }
    assert_eq!(big.get(&4999).map(|v| v[0]), Some(4999u64 as u8));
    println!("512-byte values             ok  ({} MiB of slots)", big.slots() * 528 / 1048576);

    // 4. a reserve nobody can satisfy must fail loudly, not wrap to a small table
    let r = std::panic::catch_unwind(|| {
        let mut m: Words<u64, u64> = Words::new();
        m.reserve(usize::MAX / 4);
        m.slots()
    });
    println!("reserve(usize::MAX/4)       {}", match r {
        Ok(n) => format!("RETURNED {n} slots -- did it really allocate that?"),
        Err(_) => "panicked (allocation failure), which is what std does".into(),
    });

    // 5. capacity honesty: with_capacity(n) must not grow on the n-th insert
    for n in [1usize, 7, 8, 14, 15, 100, 1000] {
        let mut m: Words<u64, u64> = Words::with_capacity(n);
        let before = m.slots();
        for i in 0..n as u64 { m.insert(i, i); }
        let grew = m.slots() != before;
        println!("with_capacity({n:>5}) -> {before:>6} slots, grew filling it: {grew}");
    }
}
