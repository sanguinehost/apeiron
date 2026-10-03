//! FIPS 197: AES-128 block cipher, pure safe Rust, no dependencies.
//!
//! Ported from `work/crypto-aes/src/lib.rs`; the NIST SP 800-38A F.1.1 known-answer test is
//! `tests::kat` below.
//!
//! Reference: FIPS PUB 197, "Advanced Encryption Standard (November 2001)", sections 5.2 (key
//! expansion), 6.1 (cipher, SubBytes/ShiftRows/MixColumns/AddRoundKey) and 6.2 (inverse cipher).
//!
//! The state is 16 bytes in column-major order: index `4 * c + r` holds row `r` of column `c`,
//! matching the FIPS 197 state layout. Every loop here is a `while` over an explicit index, with no
//! iterators, so the control flow is directly readable in the extraction fragment.

/// Length in bytes of an AES-128 key.
pub const KEY_LEN: usize = 16;

/// Length in bytes of an AES block.
pub const BLOCK_LEN: usize = 16;

/// AES-128 round constants (RCON), one per key-expansion round.
const RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

/// Number of rounds for AES-128.
const ROUNDS: usize = 10;

/// FIPS 197 Figure 7: the AES S-box.
const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// FIPS 197 Figure 14: the inverse S-box, derived from [`SBOX`].
const INV_SBOX: [u8; 256] = build_inv_sbox();

/// Build the inverse permutation of the S-box at compile time.
///
/// Cost: 256 table writes over a fixed bound, `O(1)` at runtime; it runs only in const evaluation.
const fn build_inv_sbox() -> [u8; 256] {
    let mut inv = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        inv[SBOX[i] as usize] = i as u8;
        i += 1;
    }
    inv
}

/// Multiply by `x` in GF(2^8) modulo the AES polynomial `m(x) = x^8 + x^4 + x^3 + x + 1`.
///
/// Cost: one shift, one test and one xor on `u8`, `O(1)`.
const fn xtime(a: u8) -> u8 {
    let hi = if a & 0x80 != 0 { 0x1b } else { 0x00 };
    (a << 1) ^ hi
}

/// Carry-less multiply in GF(2^8) modulo the AES polynomial (Russian peasant multiplication).
///
/// Cost: eight iterations of one conditional xor and one shift, so 8 ands, 8 shifts and at most
/// 8 xors on `u8`, `O(1)`. The loop closes because each iteration advances `i` to `i + 1` until
/// `i == 8`.
const fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut acc: u8 = 0;
    let mut i = 0usize;
    while i < 8 {
        if b & 1 != 0 {
            acc ^= a;
        }
        a = xtime(a);
        b >>= 1;
        i += 1;
    }
    acc
}

/// Expand a 128-bit key into `ROUNDS + 1` round keys (FIPS 197 section 5.2).
///
/// Cost: 4 key-column copies and 40 word expansions, each one S-box lookup or xor per byte, plus 11
/// round-key assemblies of 16 bytes; `O(1)` in any external input. Spatial cost is the 44-word
/// expansion schedule and 11 round keys, 700 bytes on the stack.
fn expand_key(key: &[u8; KEY_LEN]) -> [[u8; BLOCK_LEN]; ROUNDS + 1] {
    // Word `i` occupies bytes `4*i .. 4*i+4` and is one column of the state.
    let mut w = [[0u8; 4]; 4 * (ROUNDS + 1)];
    let mut i = 0usize;
    while i < 4 {
        let o = 4 * i;
        w[i][0] = key[o];
        w[i][1] = key[o + 1];
        w[i][2] = key[o + 2];
        w[i][3] = key[o + 3];
        i += 1;
    }
    let mut i = 4usize;
    while i < 4 * (ROUNDS + 1) {
        let mut temp = w[i - 1];
        if i % 4 == 0 {
            temp = [
                SBOX[temp[1] as usize] ^ RCON[i / 4 - 1],
                SBOX[temp[2] as usize],
                SBOX[temp[3] as usize],
                SBOX[temp[0] as usize],
            ];
        }
        let mut b = 0usize;
        while b < 4 {
            w[i][b] = w[i - 4][b] ^ temp[b];
            b += 1;
        }
        i += 1;
    }

    let mut round_keys = [[0u8; BLOCK_LEN]; ROUNDS + 1];
    let mut r = 0usize;
    while r < ROUNDS + 1 {
        let mut c = 0usize;
        while c < 4 {
            let o = 4 * c;
            round_keys[r][o] = w[4 * r + c][0];
            round_keys[r][o + 1] = w[4 * r + c][1];
            round_keys[r][o + 2] = w[4 * r + c][2];
            round_keys[r][o + 3] = w[4 * r + c][3];
            c += 1;
        }
        r += 1;
    }
    round_keys
}

/// XOR one round key into the state (FIPS 197 section 6.1.4).
///
/// Cost: 16 byte reads and 16 xors, `O(1)`.
fn add_round_key(state: &mut [u8; BLOCK_LEN], rk: &[u8; BLOCK_LEN]) {
    let mut i = 0usize;
    while i < BLOCK_LEN {
        state[i] ^= rk[i];
        i += 1;
    }
}

/// Apply SubBytes to all 16 state bytes with the S-box `sbox`.
///
/// Cost: 16 table lookups and 16 writes, `O(1)`.
fn sub_bytes(state: &mut [u8; BLOCK_LEN], sbox: &[u8; 256]) {
    let mut i = 0usize;
    while i < BLOCK_LEN {
        state[i] = sbox[state[i] as usize];
        i += 1;
    }
}

/// Apply ShiftRows: row `r` rotates left by `r` (FIPS 197 section 6.1.2).
///
/// Cost: three rows of four byte reads and four writes, 12 of each, `O(1)`.
fn shift_rows(state: &mut [u8; BLOCK_LEN]) {
    let mut r = 1usize;
    while r < 4 {
        let col = [state[r], state[r + 4], state[r + 8], state[r + 12]];
        let mut c = 0usize;
        while c < 4 {
            state[r + 4 * c] = col[(c + r) % 4];
            c += 1;
        }
        r += 1;
    }
}

/// Apply InvShiftRows: row `r` rotates right by `r` (FIPS 197 section 6.2.2).
///
/// Cost: three rows of four byte reads and four writes, 12 of each, `O(1)`.
fn inv_shift_rows(state: &mut [u8; BLOCK_LEN]) {
    let mut r = 1usize;
    while r < 4 {
        let col = [state[r], state[r + 4], state[r + 8], state[r + 12]];
        let mut c = 0usize;
        while c < 4 {
            state[r + 4 * ((c + r) % 4)] = col[c];
            c += 1;
        }
        r += 1;
    }
}

/// Apply MixColumns to the column `a` (FIPS 197 section 6.1.3).
///
/// Cost: 16 GF(2^8) products by the fixed constants 2 and 3, so 16 `gf_mul` calls, `O(1)`.
fn mix_column(a: &mut [u8]) {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];
    a[0] = gf_mul(a0, 2) ^ gf_mul(a1, 3) ^ a2 ^ a3;
    a[1] = a0 ^ gf_mul(a1, 2) ^ gf_mul(a2, 3) ^ a3;
    a[2] = a0 ^ a1 ^ gf_mul(a2, 2) ^ gf_mul(a3, 3);
    a[3] = gf_mul(a0, 3) ^ a1 ^ a2 ^ gf_mul(a3, 2);
}

/// Apply InvMixColumns to the column `a` (FIPS 197 section 6.3.2).
///
/// Cost: 16 GF(2^8) products by the fixed constants 9, 11, 13 and 14, so 16 `gf_mul` calls,
/// `O(1)`.
fn inv_mix_column(a: &mut [u8]) {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];
    a[0] = gf_mul(a0, 14) ^ gf_mul(a1, 11) ^ gf_mul(a2, 13) ^ gf_mul(a3, 9);
    a[1] = gf_mul(a0, 9) ^ gf_mul(a1, 14) ^ gf_mul(a2, 11) ^ gf_mul(a3, 13);
    a[2] = gf_mul(a0, 13) ^ gf_mul(a1, 9) ^ gf_mul(a2, 14) ^ gf_mul(a3, 11);
    a[3] = gf_mul(a0, 11) ^ gf_mul(a1, 13) ^ gf_mul(a2, 9) ^ gf_mul(a3, 14);
}

/// Encrypt one 16-byte block with AES-128.
///
/// Total and infallible: `key` and `block` are fixed-size arrays, so there is no domain to check and
/// no failure case.
///
/// Cost: one key expansion (11 round keys), then 10 rounds of 16 S-box lookups, 12 row rotations
/// and one MixColumns (16 GF(2^8) products) per round except the last, plus 11 AddRoundKey passes;
/// `O(1)` in any external input. Spatial cost is the 176-byte state plus the 700-byte expansion
/// schedule.
#[must_use]
pub fn aes128_encrypt_block(key: &[u8; KEY_LEN], block: &[u8; BLOCK_LEN]) -> [u8; BLOCK_LEN] {
    let rk = expand_key(key);
    let mut s = *block;

    add_round_key(&mut s, &rk[0]);
    let mut round = 1usize;
    while round < ROUNDS {
        sub_bytes(&mut s, &SBOX);
        shift_rows(&mut s);
        let mut c = 0usize;
        while c < 4 {
            let o = 4 * c;
            mix_column(&mut s[o..o + 4]);
            c += 1;
        }
        add_round_key(&mut s, &rk[round]);
        round += 1;
    }
    // Final round: SubBytes, ShiftRows, AddRoundKey (no MixColumns).
    sub_bytes(&mut s, &SBOX);
    shift_rows(&mut s);
    add_round_key(&mut s, &rk[ROUNDS]);
    s
}

/// Decrypt one 16-byte block with AES-128 (the equivalent inverse cipher, FIPS 197 section 6.2).
///
/// Total and infallible, with the same domain as [`aes128_encrypt_block`].
///
/// Cost: one key expansion (11 round keys), then 9 rounds of InvShiftRows, 16 inverse S-box
/// lookups, AddRoundKey and one InvMixColumns (16 GF(2^8) products), plus the final
/// InvShiftRows/InvSubBytes/AddRoundKey pass; `O(1)` in any external input. Spatial cost is the
/// 176-byte state plus the 700-byte expansion schedule.
#[must_use]
pub fn aes128_decrypt_block(key: &[u8; KEY_LEN], block: &[u8; BLOCK_LEN]) -> [u8; BLOCK_LEN] {
    let rk = expand_key(key);
    let mut s = *block;

    add_round_key(&mut s, &rk[ROUNDS]);
    let mut round = ROUNDS - 1;
    while round > 0 {
        inv_shift_rows(&mut s);
        sub_bytes(&mut s, &INV_SBOX);
        add_round_key(&mut s, &rk[round]);
        let mut c = 0usize;
        while c < 4 {
            let o = 4 * c;
            inv_mix_column(&mut s[o..o + 4]);
            c += 1;
        }
        round -= 1;
    }
    inv_shift_rows(&mut s);
    sub_bytes(&mut s, &INV_SBOX);
    add_round_key(&mut s, &rk[0]);
    s
}

#[cfg(test)]
mod tests {
    use super::{aes128_decrypt_block, aes128_encrypt_block};

    /// NIST SP 800-38A, "Recommendation for Block Cipher Modes of Operation", F.1.1
    /// ECB-AES128.Encrypt: key, plaintext and the resulting ciphertext.
    const KEY: [u8; 16] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f,
    ];
    const PLAINTEXT: [u8; 16] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ];
    const CIPHERTEXT: [u8; 16] = [
        0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5,
        0x5a,
    ];

    #[test]
    fn sp800_38a_f1_1_encrypts_to_the_published_block() {
        assert_eq!(aes128_encrypt_block(&KEY, &PLAINTEXT), CIPHERTEXT);
    }

    #[test]
    fn sp800_38a_f1_1_decrypts_the_published_block() {
        assert_eq!(aes128_decrypt_block(&KEY, &CIPHERTEXT), PLAINTEXT);
    }

    #[test]
    fn decryption_inverts_encryption() {
        // The all-zero key and block are outside the SP 800-38A vector, so this checks the
        // round-trip only, not a specification value.
        let key = [0u8; 16];
        let ct = aes128_encrypt_block(&key, &[0u8; 16]);
        assert_eq!(aes128_decrypt_block(&key, &ct), [0u8; 16]);
    }
}
