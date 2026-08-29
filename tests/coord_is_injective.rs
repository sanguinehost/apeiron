//! The lossy fold spreads keys, including negative ones.
//!
//! This is the gate hashtrinity did not have. Its `(A, B)` fold collides on 49% of a signed grid
//! and on 72% of a signed 3-tuple grid, and nothing failed, because every test used non-negative
//! keys -- where that fold is perfect. A spatial index over a world whose origin sits in the
//! middle of it is exactly the case that breaks.
//!
//! These grids happen to be collision-free, and that is a measurement rather than a guarantee:
//! the fold is a discard and is not injective on the full domain. See
//! `each_face_gets_the_fold_it_needs.rs` for the explicit collision, and use `Packed` where the
//! address has to determine the key.

use apeiron::coord::Coord;
use std::collections::HashSet;

fn distinct<K: Coord>(keys: &[K]) -> usize {
    keys.iter().map(Coord::coord).collect::<HashSet<_>>().len()
}

/// Half-width of each grid. Miri interprets every fold and every `HashSet` insert, so the wide
/// grid's 1,048,576 keys is a run that does not finish -- it was the single slowest thing in the
/// Miri gate by a wide margin. A grid that straddles zero is what these tests are for, and a
/// smaller one straddles it just as well; the full sizes still run natively, which is where the
/// collision search has to be exhaustive.
const PAIR: i32 = if cfg!(miri) { 16 } else { 64 };
const WIDE: i32 = if cfg!(miri) { 32 } else { 512 };
const TRIPLE: i32 = if cfg!(miri) { 6 } else { 16 };

#[cfg_attr(miri, ignore = "pure arithmetic: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn a_signed_pair_grid_is_spread_without_collision() {
    let keys: Vec<(i32, i32)> =
        (-PAIR..PAIR).flat_map(|x| (-PAIR..PAIR).map(move |y| (x, y))).collect();
    assert_eq!(distinct(&keys), keys.len(), "signed {}x{} grid collides", PAIR * 2, PAIR * 2);
}

#[cfg_attr(miri, ignore = "pure arithmetic: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn a_wide_signed_pair_grid_is_spread_without_collision() {
    let keys: Vec<(i32, i32)> =
        (-WIDE..WIDE).flat_map(|x| (-WIDE..WIDE).map(move |y| (x, y))).collect();
    assert_eq!(distinct(&keys), keys.len(), "signed {}x{} grid collides", WIDE * 2, WIDE * 2);
}

#[cfg_attr(miri, ignore = "pure arithmetic: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn a_signed_triple_grid_is_spread_without_collision() {
    let keys: Vec<(i32, i32, i32)> = (-TRIPLE..TRIPLE)
        .flat_map(|x| {
            (-TRIPLE..TRIPLE).flat_map(move |y| (-TRIPLE..TRIPLE).map(move |z| (x, y, z)))
        })
        .collect();
    assert_eq!(distinct(&keys), keys.len(), "signed {}^3 grid collides", TRIPLE * 2);
}

/// The non-negative case, which the old fold also passed. Kept so a change that fixes the signed
/// case by breaking the unsigned one is caught.
#[cfg_attr(miri, ignore = "pure arithmetic: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn an_unsigned_grid_is_still_spread_without_collision() {
    let n = if cfg!(miri) { 32u32 } else { 128 };
    let keys: Vec<(u32, u32)> = (0..n).flat_map(|x| (0..n).map(move |y| (x, y))).collect();
    assert_eq!(distinct(&keys), keys.len());
}

/// Mixed widths and signs, which is what a real key looks like: a cell coordinate and a
/// generation, say.
#[cfg_attr(miri, ignore = "pure arithmetic: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn mixed_width_keys_do_not_collide() {
    let keys: Vec<(i16, u32, i64)> = (-8i16..8)
        .flat_map(|a| (0u32..8).flat_map(move |b| (-8i64..8).map(move |c| (a, b, c))))
        .collect();
    assert_eq!(distinct(&keys), keys.len());
}
