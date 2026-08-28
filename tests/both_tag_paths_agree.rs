//! The two tag-scan paths must be the same function.
//!
//! A group scan is sixteen lanes at once, and there are two ways to get them: `_mm_cmpeq_epi8`
//! plus `_mm_movemask_epi8` on x86_64, and a SWAR word pair everywhere else. Only one of them
//! runs on any given build, so the other is exactly the kind of code that rots. These tests run
//! both and assert they agree, against a scalar reference that is obviously right.

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

/// A slot is occupied exactly when its tag has the top bit set; `tag_of` always sets it and an
/// empty slot's tag is zero, so this is "not empty".
fn reference_occupied(line: &[u8; 16]) -> u16 {
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
        l[i] = 0x80 | (i as u8 * 7 % 128);
        out.push(l);
    }
    // Every lane occupied but one: the "group is full except here" case.
    for i in 0..16 {
        let mut l = [0x80u8; 16];
        l[i] = 0;
        out.push(l);
    }
    // Pseudorandom fill over the real alphabet: zero, or 0x80 | seven bits.
    let mut r = 0x243F_6A88_85A3_08D3u64;
    let mut next = || {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        r
    };
    for _ in 0..4000 {
        let mut l = [0u8; 16];
        for b in &mut l {
            let v = next();
            *b = if v & 3 == 0 { 0 } else { 0x80 | (v >> 8) as u8 & 0x7F };
        }
        out.push(l);
    }
    out
}

#[test]
fn the_swar_scan_is_the_scalar_definition() {
    for l in lines() {
        for tag in 0u8..=255 {
            assert_eq!(scan::match_tag_swar(&l, tag), reference_match(&l, tag), "line {l:?} tag {tag}");
        }
        assert_eq!(scan::occupied_swar(&l), reference_occupied(&l), "line {l:?}");
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn the_simd_scan_is_the_swar_scan() {
    for l in lines() {
        for tag in 0u8..=255 {
            assert_eq!(
                scan::match_tag_simd(&l, tag),
                scan::match_tag_swar(&l, tag),
                "line {l:?} tag {tag}"
            );
        }
        assert_eq!(scan::occupied_simd(&l), scan::occupied_swar(&l), "line {l:?}");
    }
}

/// Whichever one the build chose is the one the table actually calls, so check it too.
#[test]
fn the_chosen_scan_is_the_scalar_definition() {
    for l in lines() {
        for tag in 0u8..=255 {
            assert_eq!(scan::match_tag(&l, tag), reference_match(&l, tag));
        }
        assert_eq!(scan::occupied(&l), reference_occupied(&l));
    }
}
