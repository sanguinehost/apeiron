//! Injectivity is required on one face, irrelevant on the second, impossible on the third.
//!
//! This test exists because "the collision rate" was measured once and reported as settled. It is
//! three different questions:
//!
//! - **Elliptic (B).** The address determines the key, so the fold must be a bijection. Nothing
//!   is measured here -- it either round-trips or it does not.
//! - **Parabolic (B/U).** The key is carried beside the address, so a collision costs a comparison
//!   and never an error. Distribution matters, injectivity does not.
//! - **Hyperbolic (NB).** The address is a discard. Injectivity is not merely unnecessary, it is
//!   unavailable: two words do not fit in one.

use apeiron::coord::{i21, mix, Coord, Packed, PHI};
use std::collections::HashSet;

/// The bijection round-trips, over the whole of a representative slice.
#[cfg_attr(miri, ignore = "pure arithmetic: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn a_packed_key_round_trips_exactly() {
    for x in [i32::MIN, -70_000, -1, 0, 1, 70_000, i32::MAX] {
        for y in [i32::MIN, -3, 0, 3, i32::MAX] {
            assert_eq!(<(i32, i32)>::unpack((x, y).packed()), (x, y));
        }
    }
    for x in 0u32..256 {
        for y in [0u32, 1, 4_294_967_295] {
            assert_eq!(<(u32, u32)>::unpack((x, y).packed()), (x, y));
        }
    }
}

/// And it is injective on a grid where the arithmetic fold is not.
#[cfg_attr(miri, ignore = "pure arithmetic: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn packing_separates_every_key_in_a_signed_grid() {
    let keys: Vec<(i32, i32)> = (-64..64).flat_map(|x| (-64..64).map(move |y| (x, y))).collect();
    let addresses: HashSet<u64> = keys.iter().map(Packed::packed).collect();
    assert_eq!(addresses.len(), keys.len());
}

/// The 21-bit coordinate refuses what will not fit, rather than truncating it.
///
/// This is the bijection's precondition made unrepresentable instead of documented.
#[cfg_attr(miri, ignore = "pure arithmetic: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn a_narrow_coordinate_refuses_an_out_of_range_value() {
    assert!(i21::new(0).is_some());
    assert!(i21::new(i21::MAX).is_some());
    assert!(i21::new(i21::MIN).is_some());
    assert!(i21::new(i21::MAX + 1).is_none());
    assert!(i21::new(i21::MIN - 1).is_none());

    for x in [i21::MIN, -1, 0, 1, i21::MAX] {
        for y in [i21::MIN, 0, i21::MAX] {
            for z in [i21::MIN, 0, i21::MAX] {
                let k = (
                    i21::new(x).unwrap(),
                    i21::new(y).unwrap(),
                    i21::new(z).unwrap(),
                );
                assert_eq!(<(i21, i21, i21)>::unpack(k.packed()), k, "{x} {y} {z}");
            }
        }
    }
}

/// The arithmetic fold is **not** injective, with the witness.
///
/// `f(a, b) = a * PHI + b` collides when `d * PHI = -e (mod 2^64)` for `|d|, |e| < 2^32`. PHI is
/// the golden ratio, so its continued-fraction convergents are Fibonacci numbers, and 2971215073
/// is one: that is where a multiplier stops spreading. Asserting this rather than hoping it is
/// absent is the difference between a discard and a broken bijection.
#[cfg_attr(miri, ignore = "pure arithmetic: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn the_arithmetic_fold_is_not_a_bijection_and_here_is_the_collision() {
    let a = mix(2_971_215_073, 50_920_843);
    let b = mix(0, 0);
    assert_eq!(a, b, "the Fibonacci convergent collides with the origin");

    // And it is not an accident of that pair: the same d shifts any key onto another.
    for base in [1u64, 12_345, 999_999] {
        assert_eq!(
            mix(base.wrapping_add(2_971_215_073), 50_920_843),
            mix(base, 0),
            "the collision translates"
        );
    }
    assert_ne!(PHI, 0);
}

/// What the discard is judged on instead: how evenly it spreads.
///
/// Measured as distinct addresses over a grid far larger than the table that would hold it, which
/// is the regime a lossy fold is actually used in.
#[cfg_attr(miri, ignore = "pure arithmetic: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn the_discard_spreads_evenly_on_the_grids_that_use_it() {
    let keys: Vec<(i32, i32)> = (-512..512).flat_map(|x| (-512..512).map(move |y| (x, y))).collect();
    let distinct: HashSet<u64> = keys.iter().map(Coord::coord).collect();
    // Not a proof of injectivity -- there is none to have -- but the rate a discard must hit.
    let collisions = keys.len() - distinct.len();
    assert!(
        collisions * 1000 < keys.len(),
        "the discard collided on {collisions} of {} keys, over one in a thousand",
        keys.len()
    );
}
