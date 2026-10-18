//! FIPS-202: the Keccak-f[1600] permutation and the sponge construction.
//!
//! Provides the SHA3-256 and SHA3-512 fixed-output hash functions and the SHAKE128 / SHAKE256
//! extendable-output functions (XOFs). Ported from `work/crypto-rs/src/lib.rs`; the NIST
//! known-answer tests are `tests/keccak.rs`.
//!
//! Reference: FIPS PUB 202, "SHA-3 Standard: Permutation-Based Hash and Extendable-Output Functions
//! (August 2015)".
#![allow(
    clippy::manual_is_multiple_of,
    reason = "the rate and block-length preconditions are written as `n % k == 0` to match \
              FIPS-202 section 5.1 literally; this module is a port of \
              `work/crypto-rs/src/lib.rs` and is kept line-identical to that KAT-passing source"
)]

/// Rate (in bytes) for SHA3-256: capacity 512 bits.
const SHA3_256_RATE: usize = 136;
/// Rate (in bytes) for SHA3-512: capacity 1024 bits.
const SHA3_512_RATE: usize = 72;
/// Rate (in bytes) for SHAKE128: capacity 256 bits.
const SHAKE128_RATE: usize = 168;
/// Rate (in bytes) for SHAKE256: capacity 512 bits.
const SHAKE256_RATE: usize = 136;

/// Round constants `RC[i]` for the i-th of the 24 rounds.
const RC: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// Rotation offsets `r[x, y]` in the rho and pi steps, indexed `x + 5 * y`.
const ROT: [u32; 25] = [
    0, 1, 62, 28, 27, // y = 0
    36, 44, 6, 55, 20, // y = 1
    3, 10, 43, 25, 39, // y = 2
    41, 45, 15, 21, 8, // y = 3
    18, 2, 61, 56, 14, // y = 4
];

/// The Keccak-f[1600] permutation, applied in place to the 25-lane state.
///
/// Lanes are indexed `A[x + 5 * y]`.
///
/// Cost: 24 rounds, each over the 25 lanes and a 25-lane scratch array, so `O(1)` time in the
/// message length (24 * 5 column XORs, 25 rotates and 25 chi terms per round) and 25 lanes of
/// scratch, independent of any input length.
pub fn keccak_f1600(state: &mut [u64; 25]) {
    let mut round = 0usize;
    while round < 24 {
        let rc = RC[round];

        // theta
        let mut c = [0u64; 5];
        let mut x = 0usize;
        while x < 5 {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
            x += 1;
        }
        let mut x = 0usize;
        while x < 5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            let mut y = 0usize;
            while y < 5 {
                state[x + 5 * y] ^= d;
                y += 1;
            }
            x += 1;
        }

        // rho and pi
        let mut b = [0u64; 25];
        let mut y = 0usize;
        while y < 5 {
            let mut x = 0usize;
            while x < 5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] = state[x + 5 * y].rotate_left(ROT[x + 5 * y]);
                x += 1;
            }
            y += 1;
        }

        // chi
        let mut y = 0usize;
        while y < 5 {
            let mut x = 0usize;
            while x < 5 {
                state[x + 5 * y] =
                    b[x + 5 * y] ^ ((!b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);
                x += 1;
            }
            y += 1;
        }

        // iota
        state[0] ^= rc;
        round += 1;
    }
}

/// XOR `block` into the first `block.len() / 8` lanes of `state`.
///
/// Total on `block.len() % 8 == 0`, which every caller holds by construction; the debug assertion
/// records that precondition.
fn absorb_block(state: &mut [u64; 25], block: &[u8]) {
    debug_assert_eq!(block.len() % 8, 0);
    let lanes = block.len() / 8;
    let mut i = 0usize;
    while i < lanes {
        let o = i * 8;
        state[i] ^= u64::from_le_bytes([
            block[o],
            block[o + 1],
            block[o + 2],
            block[o + 3],
            block[o + 4],
            block[o + 5],
            block[o + 6],
            block[o + 7],
        ]);
        i += 1;
    }
}

/// The core sponge: absorb `msg` at `rate`, then write `out.len()` bytes of output.
///
/// `pad10` is the leading pad nibble: the domain separation bits `01` for SHA-3 (`0x06`) and
/// `1111` for SHAKE (`0x1f`). FIPS-202 pad10\*1 appends a final `1` bit, which lands in the low bit
/// of the block's last byte (`pad_final`, `0x80`) for both SHA-3 and SHAKE.
///
/// Cost: `ceil((msg.len() + 1) / rate) + ceil(out.len() / rate)` permutations, so `O(msg.len() +
/// out.len())` time in bytes; spatial cost is the fixed 25-lane state plus two 200-byte stack
/// buffers, independent of both lengths. The `debug_assert` states the domain `rate % 8 == 0 &&
/// rate <= 200`, which the four rate constants hold.
fn sponge(msg: &[u8], rate: usize, pad10: u8, pad_final: u8, out: &mut [u8]) {
    debug_assert!(rate % 8 == 0 && rate <= 200);
    let mut state = [0u64; 25];

    // Absorb full blocks.
    let mut consumed = 0usize;
    while consumed + rate <= msg.len() {
        absorb_block(&mut state, &msg[consumed..consumed + rate]);
        keccak_f1600(&mut state);
        consumed += rate;
    }

    // Absorb the final (possibly empty) padded block.
    let rem = &msg[consumed..];
    let mut last = [0u8; 200];
    last[..rem.len()].copy_from_slice(rem);
    last[rem.len()] ^= pad10;
    last[rate - 1] ^= pad_final;
    absorb_block(&mut state, &last[..rate]);
    keccak_f1600(&mut state);

    // Squeeze.
    let mut squeezed = [0u8; 200];
    let mut i = 0usize;
    while i < rate / 8 {
        let o = i * 8;
        squeezed[o..o + 8].copy_from_slice(&state[i].to_le_bytes());
        i += 1;
    }
    let mut off = 0usize;
    let mut filled = 0usize;
    while off < out.len() {
        if filled == rate {
            keccak_f1600(&mut state);
            let mut i = 0usize;
            while i < rate / 8 {
                let o = i * 8;
                squeezed[o..o + 8].copy_from_slice(&state[i].to_le_bytes());
                i += 1;
            }
            filled = 0;
        }
        let n = core::cmp::min(rate - filled, out.len() - off);
        out[off..off + n].copy_from_slice(&squeezed[filled..filled + n]);
        off += n;
        filled += n;
    }
}

/// SHA3-256 of `msg`.
///
/// Cost: `(msg.len() / 136 + 1)` permutations, `O(msg.len())` in bytes; `O(1)` space.
#[must_use]
pub fn sha3_256(msg: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    sponge(msg, SHA3_256_RATE, 0x06, 0x80, &mut out);
    out
}

/// SHA3-512 of `msg`.
///
/// Cost: `(msg.len() / 72 + 1)` permutations, `O(msg.len())` in bytes; `O(1)` space.
#[must_use]
pub fn sha3_512(msg: &[u8]) -> [u8; 64] {
    let mut out = [0u8; 64];
    sponge(msg, SHA3_512_RATE, 0x06, 0x80, &mut out);
    out
}

/// SHAKE128 extendable-output function, writing `out.len()` bytes into `out`.
///
/// Cost: `(msg.len() / 168 + 1 + out.len() / 168)` permutations, `O(msg.len() + out.len())` in
/// bytes; `O(1)` space.
pub fn shake128(msg: &[u8], out: &mut [u8]) {
    sponge(msg, SHAKE128_RATE, 0x1f, 0x80, out);
}

/// SHAKE256 extendable-output function, writing `out.len()` bytes into `out`.
///
/// Cost: `(msg.len() / 136 + 1 + out.len() / 136)` permutations, `O(msg.len() + out.len())` in
/// bytes; `O(1)` space.
pub fn shake256(msg: &[u8], out: &mut [u8]) {
    sponge(msg, SHAKE256_RATE, 0x1f, 0x80, out);
}
