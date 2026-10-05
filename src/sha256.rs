//! FIPS 180-4: SHA-256, a pure safe-Rust compression-function hash with no dependencies.
//!
//! Ported from `work/crypto-sha2/src/lib.rs`; the FIPS 180-4 known-answer tests are
//! `tests/sha256.rs`.
//!
//! Reference: FIPS PUB 180-4, "Secure Hash Standard (August 2015)", sections 6.2.1 (padding),
//! 6.2.2 (message schedule) and 6.2.3 (compression function).

/// Round constants `K[0..64]`, the first 32 bits of the fractional parts of the cube roots of the
/// first 64 primes (FIPS 180-4 section 4.2.2).
const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// The initial hash value `H(0)`: the first 32 bits of the fractional parts of the square roots of
/// the first eight primes (FIPS 180-4 section 5.3.3).
const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// Compress one 64-byte block into the running state.
///
/// Cost: 64 rounds over a 64-word message schedule built from the 16 block words, so 256 `u32`
/// additions, 96 rotations and 128 bitwise operations per block; `O(1)` in any external input.
/// Spatial cost is the 64-word schedule, 256 bytes on the stack.
#[allow(
    clippy::many_single_char_names,
    reason = "FIPS 180-4 6.2.3 names the working variables a..h, T1, T2 and the schedule W"
)]
fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    // Message schedule: big-endian words, then the 48-word extension.
    let mut w = [0u32; 64];
    let mut i = 0usize;
    while i < 16 {
        let o = 4 * i;
        w[i] = u32::from_be_bytes([block[o], block[o + 1], block[o + 2], block[o + 3]]);
        i += 1;
    }
    let mut i = 16usize;
    while i < 64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
        i += 1;
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    let mut i = 0usize;
    while i < 64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
        i += 1;
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Compute the SHA-256 digest of `msg`.
///
/// Total on every input: the message is padded to `floor(msg.len() / 64) + 1` or `+ 2` blocks and
/// each block is compressed once, so `O(msg.len() / 64)` compressions, `O(msg.len())` time in
/// bytes. Spatial cost is the 8-word state plus a 64-word schedule and a 128-byte tail buffer, all
/// on the stack, independent of `msg.len()`.
#[must_use]
pub fn sha256(msg: &[u8]) -> [u8; 32] {
    let mut state = H0;

    // Full 64-byte blocks.
    let full = msg.len() / 64;
    let mut i = 0usize;
    while i < full {
        let mut block = [0u8; 64];
        let o = i * 64;
        block.copy_from_slice(&msg[o..o + 64]);
        compress(&mut state, &block);
        i += 1;
    }

    // Final block(s): remainder || 0x80 || zero pad || bit length (64-bit big-endian). The length
    // field needs the last 9 bytes, so a remainder of 56 or more bytes needs a second block.
    let rest = &msg[full * 64..];
    let mut tail = [0u8; 128];
    tail[..rest.len()].copy_from_slice(rest);
    tail[rest.len()] = 0x80;

    let tail_len = if rest.len() < 56 { 64 } else { 128 };
    let bit_len = (msg.len() as u64).wrapping_mul(8);
    tail[tail_len - 8..tail_len].copy_from_slice(&bit_len.to_be_bytes());

    let mut i = 0usize;
    while i < tail_len {
        let mut block = [0u8; 64];
        block.copy_from_slice(&tail[i..i + 64]);
        compress(&mut state, &block);
        i += 64;
    }

    let mut out = [0u8; 32];
    let mut i = 0usize;
    while i < 8 {
        out[4 * i..4 * i + 4].copy_from_slice(&state[i].to_be_bytes());
        i += 1;
    }
    out
}

/// Compute the double SHA-256 of `msg`: `sha256(sha256(msg))`, the Bitcoin block and
/// transaction-commitment construction.
///
/// Temporal cost is two applications of [`sha256`], so `O(msg.len() / 64)` compressions, `O(msg.len())`
/// time in bytes: the first call pads to `floor(msg.len() / 64) + 1` or `+ 2` blocks, the second pads
/// the fixed 32-byte digest to exactly one block. Spatial cost is two rounds of the 8-word state, the
/// 64-word schedule and the 128-byte tail buffer, all on the stack, independent of `msg.len()`; the
/// first digest is 32 bytes and is a value, not an allocation. Total and infallible on every input.
#[must_use]
pub fn sha256d(msg: &[u8]) -> [u8; 32] {
    sha256(&sha256(msg))
}
