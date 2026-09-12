//! The claims `coord::spread`'s doc makes about the two halves, checked rather than asserted.
//!
//! Takes nothing, checks four identities over the address, and fails if any stops holding:
//! `lo` is additive, `hi` is additive up to the carry, `place` is the stated `GF(2)` form, and the
//! low window degenerates while the tag does not.
//!
//! Counts scale under Miri, as the rest of the crate's tests do: these are arithmetic identities
//! with no unsafe in them, so a Miri run wants the path taken, not the sample size.

use apeiron::coord::{spread, PHI};

#[expect(
    clippy::cast_possible_truncation,
    reason = "the two truncations ARE the split, exactly as in `coord::spread` -- taking each half \
              of the 128-bit product is the whole point of the function"
)]
fn parts(x: u64) -> (u64, u64) {
    let w = u128::from(x).wrapping_mul(u128::from(PHI));
    (w as u64, (w >> 64) as u64)
}

const fn reps(native: usize, miri: usize) -> usize {
    if cfg!(miri) {
        miri
    } else {
        native
    }
}

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// `lo` is a unit multiple of a ring quotient, so it is additive exactly. This is what makes it
/// the identity word: additive and injective.
#[test]
fn the_low_half_is_additive() {
    let mut s = 1u64;
    for _ in 0..reps(200_000, 200) {
        let x = xorshift(&mut s);
        let y = x.rotate_left(17);
        assert_eq!(
            parts(x.wrapping_add(y)).0,
            parts(x).0.wrapping_add(parts(y).0)
        );
    }
}

/// `hi` is additive only up to the carry out of `lo`. That carry term is the whole nonlinearity of
/// the address, and the test is not vacuous only if carries actually occur -- so it counts them.
#[test]
fn the_high_half_is_additive_up_to_the_carry() {
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut carried = 0u32;
    for _ in 0..reps(200_000, 200) {
        let v = xorshift(&mut s);
        // Halved so `x + y` cannot itself wrap; the identity is about `lo`'s carry, not the input's.
        let (x, y) = (v >> 1, v.rotate_left(31) >> 1);
        let ((lx, hx), (ly, hy)) = (parts(x), parts(y));
        let carry = u64::from(lx.checked_add(ly).is_none());
        carried += u32::try_from(carry).unwrap();
        assert_eq!(parts(x + y).1, hx.wrapping_add(hy).wrapping_add(carry));
    }
    assert!(
        carried > u32::try_from(reps(200_000, 200)).unwrap() / 4,
        "only {carried} carries: the identity was not exercised"
    );
}

/// `place_i == lo_i ^ hi_((i - 32) mod 64)`: sixty-four independent `GF(2)` functionals of the
/// product's 128 bits. This is what makes disjoint slices of `place` independent evidence, which
/// is what lets `tag_of` sit directly above the descent.
#[test]
fn every_bit_of_place_is_one_lo_bit_and_one_hi_bit() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..reps(100_000, 100) {
        let x = xorshift(&mut s);
        let (lo, hi) = parts(x);
        let p = spread(x).place;
        for i in 0..64u32 {
            let want = ((lo >> i) & 1) ^ ((hi >> ((i + 32) % 64)) & 1);
            assert_eq!((p >> i) & 1, want, "bit {i} of place for key {x}");
        }
    }
}

/// Why the descent takes its window from the top. `hi`'s upper half is zero for any key whose
/// product does not fill 128 bits, so a low window is the raw low bits of `lo` -- the measured
/// flatness failure recorded on `map::Geom::first`, here as its cause.
#[test]
fn the_low_window_degenerates_and_the_tag_does_not() {
    let (mut collapsed, mut n) = (0u32, 0u32);
    let mut tags = std::collections::HashSet::new();
    for x in 0..reps(200_000, 4_000) as u64 {
        let (lo, hi) = parts(x);
        let p = spread(x).place;
        tags.insert(p >> 57);
        if hi >> 32 == 0 {
            n += 1;
            collapsed += u32::from(p & 0xFFFF_FFFF == lo & 0xFFFF_FFFF);
        }
    }
    assert!(
        n > u32::try_from(reps(200_000, 4_000)).unwrap() / 2,
        "not enough short products to say anything: {n}"
    );
    assert_eq!(
        collapsed, n,
        "the low window did not collapse; the rejection needs re-deriving"
    );
    assert_eq!(
        tags.len(),
        128,
        "the tag slice lost values: {} of 128",
        tags.len()
    );
}
