//! The two tag-scan paths must be the same function.
//!
//! A group scan is sixteen lanes at once, and there are two ways to get them: `_mm_cmpeq_epi8`
//! plus `_mm_movemask_epi8` on `x86_64`, and a SWAR word pair everywhere else. Only one of them
//! runs on any given build, so the other is exactly the kind of code that rots. These tests run
//! both and assert they agree, against a scalar reference that is obviously right.

#![expect(
    clippy::cast_possible_truncation,
    clippy::items_after_statements,
    clippy::verbose_bit_mask,
    reason = "this is a test, not library code, and each lint below is listed because this file actually \
              triggers it: sizes and indices narrowed to build keys, bounded by the loop that produced them; \
              constants declared next to the loop that uses them; the mask is the definition being checked. \
              Listed individually rather than suppressed wholesale, so a pedantic lint this file does not \
              yet trigger still surfaces"
)]

use apeiron::table::scan;

/// The definition, one lane at a time.
fn reference_match(line: &[u8; 16], tag: u8) -> u16 {
    let mut m = 0u16;
    for (i, &b) in line.iter().enumerate() {
        if b == tag {
            m |= 1 << i;
        }
    }
    m
}

/// A slot is FREE exactly when its tag has the top bit set; the sentinels set it and `tag_of`
/// never does, so this is the reference for `scan::free`. See the note on the polarity in
/// `apeiron::table::tag_of`.
/// A slot is occupied exactly when its tag has the top bit clear; `tag_of` always clears it and an
/// empty slot's tag is zero, so this is "not empty".
fn reference_free(line: &[u8; 16]) -> u16 {
    let mut m = 0u16;
    for (i, &b) in line.iter().enumerate() {
        if b & 0x80 != 0 {
            m |= 1 << i;
        }
    }
    m
}

fn lines() -> Vec<[u8; 16]> {
    let mut out = Vec::new();
    out.push([0u8; 16]);
    out.push([0xFFu8; 16]);
    // One occupied lane at each position, the rest empty: the placement case.
    for i in 0..16 {
        let mut l = [0u8; 16];
        l[i] = (i as u8 * 7) % 128;
        out.push(l);
    }
    // Every lane occupied but one: the "group is full except here" case.
    for i in 0..16 {
        let mut l = [0u8; 16];
        l[i] = 0;
        out.push(l);
    }
    // Pseudorandom fill over the real alphabet: 0xFF, or seven bits with the top clear.
    let mut r = 0x243F_6A88_85A3_08D3u64;
    let mut next = || {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        r
    };
    // Miri emulates the SSE2 intrinsics an instruction at a time, and this file is three tests x
    // `lines()` x 256 tags. At the native count that is several million emulated comparisons and
    // the run does not finish in any useful time. The structured cases above are the ones that
    // discriminate -- all-empty, all-full, one lane either way -- so they stay whole, and only the
    // random fill is cut. Same idiom as `SCALE` in `tests/the_counts_are_exact.rs`.
    const FILL: usize = if cfg!(miri) { 40 } else { 4000 };
    for _ in 0..FILL {
        let mut l = [0u8; 16];
        for b in &mut l {
            let v = next();
            *b = if v & 3 == 0 { 0xFF } else { (v >> 8) as u8 & 0x7F };
        }
        out.push(l);
    }
    out
}

#[test]
fn the_swar_scan_is_the_scalar_definition() {
    for l in lines() {
        for tag in (0u8..=255).step_by(if cfg!(miri) { 17 } else { 1 }) {
            assert_eq!(scan::match_tag_swar(&l, tag), reference_match(&l, tag), "line {l:?} tag {tag}");
        }
        assert_eq!(scan::free_swar(&l), reference_free(&l), "line {l:?}");
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn the_simd_scan_is_the_swar_scan() {
    for l in lines() {
        for tag in (0u8..=255).step_by(if cfg!(miri) { 17 } else { 1 }) {
            assert_eq!(
                scan::match_tag_simd(&l, tag),
                scan::match_tag_swar(&l, tag),
                "line {l:?} tag {tag}"
            );
        }
        assert_eq!(scan::free_simd(&l), scan::free_swar(&l), "line {l:?}");
    }
}

/// Whichever one the build chose is the one the table actually calls, so check it too.
#[test]
fn the_chosen_scan_is_the_scalar_definition() {
    for l in lines() {
        for tag in (0u8..=255).step_by(if cfg!(miri) { 17 } else { 1 }) {
            assert_eq!(scan::match_tag(&l, tag), reference_match(&l, tag));
        }
        assert_eq!(scan::free(&l), reference_free(&l));
    }
}
