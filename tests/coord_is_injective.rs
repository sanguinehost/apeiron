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

#[test]
fn a_signed_pair_grid_is_spread_without_collision() {
    let keys: Vec<(i32, i32)> = (-64..64).flat_map(|x| (-64..64).map(move |y| (x, y))).collect();
    assert_eq!(distinct(&keys), keys.len(), "signed 128x128 grid collides");
}

#[test]
fn a_wide_signed_pair_grid_is_spread_without_collision() {
    let keys: Vec<(i32, i32)> =
        (-512..512).flat_map(|x| (-512..512).map(move |y| (x, y))).collect();
    assert_eq!(distinct(&keys), keys.len(), "signed 1024x1024 grid collides");
}

#[test]
fn a_signed_triple_grid_is_spread_without_collision() {
    let keys: Vec<(i32, i32, i32)> = (-16..16)
        .flat_map(|x| (-16..16).flat_map(move |y| (-16..16).map(move |z| (x, y, z))))
        .collect();
    assert_eq!(distinct(&keys), keys.len(), "signed 32^3 grid collides");
}

/// The non-negative case, which the old fold also passed. Kept so a change that fixes the signed
/// case by breaking the unsigned one is caught.
#[test]
fn an_unsigned_grid_is_still_spread_without_collision() {
    let keys: Vec<(u32, u32)> = (0..128).flat_map(|x| (0..128).map(move |y| (x, y))).collect();
    assert_eq!(distinct(&keys), keys.len());
}

/// Mixed widths and signs, which is what a real key looks like: a cell coordinate and a
/// generation, say.
#[test]
fn mixed_width_keys_do_not_collide() {
    let keys: Vec<(i16, u32, i64)> = (-8i16..8)
        .flat_map(|a| (0u32..8).flat_map(move |b| (-8i64..8).map(move |c| (a, b, c))))
        .collect();
    assert_eq!(distinct(&keys), keys.len());
}
