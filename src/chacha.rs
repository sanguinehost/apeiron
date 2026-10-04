#![allow(
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    reason = "extraction-facing scalar forms: the pinned extractor emits an `axiom` for the \
              narrowing casts and integer `TryFrom`, so the explicit `as u32` repacking of the \
              Poly1305 limbs is kept; the RFC 8439 quarter-round names `a`..`d` and `r`/`s`/`h`/`g` \
              are the specification's names for the quantities; `ChaCha20` and `Poly1305` are the \
              specification's names for the algorithms, so prose is not backticked around them"
)]
//! RFC 8439: ChaCha20, Poly1305, and the ChaCha20-Poly1305 AEAD, in pure safe Rust with no
//! dependencies.
//!
//! Ported from `work/crypto-aead/src/lib.rs`; the RFC 8439 known-answer tests are
//! `tests/chacha.rs`.
//!
//! Reference: RFC 8439, "ChaCha20 and Poly1305 for IETF Protocols", sections 2.3 (block
//! function), 2.4 (stream cipher), 2.5 (Poly1305 authenticator) and 2.8 (AEAD).
//!
//! Every primitive is constant-time with respect to key and message content: the Poly1305 modular
//! reduction and the AEAD tag comparison use arithmetic or a constant-time equality, never an
//! early-exit branch. Every loop here is a `while` over an explicit index, with no iterators, so
//! the control flow is directly readable in the extraction fragment.

/// Length in bytes of a ChaCha20 key.
pub const KEY_LEN: usize = 32;

/// Length in bytes of the IETF ChaCha20 nonce (RFC 8439 uses a 96-bit nonce).
pub const NONCE_LEN: usize = 12;

/// Length in bytes of the one-time Poly1305 key.
pub const ONETIME_KEY_LEN: usize = 32;

/// Length in bytes of a ChaCha20 keystream block.
pub const BLOCK_LEN: usize = 64;

/// Length in bytes of a Poly1305 authentication tag.
pub const TAG_LEN: usize = 16;

/// Errors returned by the AEAD entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The supplied key was not [`KEY_LEN`] bytes.
    InvalidKeyLength,
    /// The supplied nonce was not [`NONCE_LEN`] bytes.
    InvalidNonceLength,
    /// The supplied ciphertext was shorter than a single tag.
    CiphertextTooShort,
    /// The tag did not verify; the plaintext was discarded and not returned.
    AuthenticationFailed,
}

/// `"expand 32-byte k"` as four little-endian words.
const CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

/// The ChaCha20 quarter round (RFC 8439, section 2.1).
///
/// Cost: eight additions and eight bitwise operations on `u32`, `O(1)` in any external input.
fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(7);
}

/// Reads the little-endian word at `bytes[o..o + 4]`.
///
/// Cost: four byte reads and one `O(1)` word assembly; the slice bound is a panic, so the caller
/// supplies four readable bytes.
fn le32_at(bytes: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
}

/// Writes `v` little-endian at `out[o..o + 4]`.
///
/// Cost: four byte writes, `O(1)`.
fn write_le32_at(out: &mut [u8], o: usize, v: u32) {
    let b = v.to_le_bytes();
    out[o] = b[0];
    out[o + 1] = b[1];
    out[o + 2] = b[2];
    out[o + 3] = b[3];
}

/// Runs the 20-round ChaCha20 permutation in place on the 16-word state.
///
/// Cost: 10 rounds of 8 quarter rounds, so 80 quarter rounds, 320 additions and 320 rotations on
/// `u32`, independent of every input. Spatial cost is the caller's 16 words.
fn permute(state: &mut [u32; 16]) {
    let mut round = 0usize;
    while round < 10 {
        // Column round, then diagonal round.
        quarter_round(state, 0, 4, 8, 12);
        quarter_round(state, 1, 5, 9, 13);
        quarter_round(state, 2, 6, 10, 14);
        quarter_round(state, 3, 7, 11, 15);
        quarter_round(state, 0, 5, 10, 15);
        quarter_round(state, 1, 6, 11, 12);
        quarter_round(state, 2, 7, 8, 13);
        quarter_round(state, 3, 4, 9, 14);
        round += 1;
    }
}

/// Builds the 16-word ChaCha20 state: constants, key, counter, nonce.
///
/// Cost: eight key words and three nonce words read, 16 words of scratch, `O(1)`.
fn state_from(key: &[u8; KEY_LEN], counter: u32, nonce: &[u8; NONCE_LEN]) -> [u32; 16] {
    let mut s = [0u32; 16];
    let mut i = 0usize;
    while i < 4 {
        s[i] = CONSTANTS[i];
        i += 1;
    }
    let mut i = 0usize;
    while i < 8 {
        s[4 + i] = le32_at(key, i * 4);
        i += 1;
    }
    s[12] = counter;
    let mut i = 0usize;
    while i < 3 {
        s[13 + i] = le32_at(nonce, i * 4);
        i += 1;
    }
    s
}

/// Generates one 64-byte ChaCha20 keystream block.
///
/// Cost: one 20-round permutation plus 16 word additions, 64 bytes written; `O(1)` in any
/// external input. Spatial cost is 32 words (working state and initial state).
#[must_use]
pub fn chacha20_block(
    key: &[u8; KEY_LEN],
    counter: u32,
    nonce: &[u8; NONCE_LEN],
) -> [u8; BLOCK_LEN] {
    let mut state = state_from(key, counter, nonce);
    let initial = state;
    permute(&mut state);

    let mut block = [0u8; BLOCK_LEN];
    let mut i = 0usize;
    while i < 16 {
        write_le32_at(&mut block, i * 4, state[i].wrapping_add(initial[i]));
        i += 1;
    }
    block
}

/// Encrypts or decrypts `data` in place with ChaCha20, XORing the keystream over it.
///
/// The block counter starts at `counter` and increments once per 64-byte block, wrapping at
/// `u32::MAX`. ChaCha20 is its own inverse, so this serves both directions.
///
/// Cost: `ceil(data.len() / 64)` permutations, so `O(data.len() / 64)` 20-round permutations and
/// `data.len()` XORs, with one 64-byte block of scratch. The loop closes because each iteration
/// advances the offset by one block until it reaches `data.len()`.
pub fn chacha20_xor(key: &[u8; KEY_LEN], counter: u32, nonce: &[u8; NONCE_LEN], data: &mut [u8]) {
    let mut block_index = 0usize;
    let mut offset = 0usize;
    while offset < data.len() {
        let ks = chacha20_block(key, counter.wrapping_add(block_index as u32), nonce);
        let end = offset + BLOCK_LEN;
        let stop = if end < data.len() { end } else { data.len() };
        let mut i = offset;
        while i < stop {
            data[i] ^= ks[i - offset];
            i += 1;
        }
        offset = stop;
        block_index += 1;
    }
}

/// Poly1305 clamps `r` per RFC 8439 section 2.5.1, i.e.
/// `r &= 0x0ffffffc0ffffffc0ffffffc0fffffff`.
///
/// Bytes 4, 8 and 12 keep their low two bits cleared (`0xfc`), matching the RFC mask applied
/// byte-wise; clearing the high two bits instead yields wrong tags for most keys.
///
/// Cost: seven byte masks, `O(1)`.
fn poly1305_clamp(r: &mut [u8; 16]) {
    r[3] &= 15;
    r[7] &= 15;
    r[11] &= 15;
    r[15] &= 15;
    r[4] &= 0xfc;
    r[8] &= 0xfc;
    r[12] &= 0xfc;
}

/// Splits the clamped `r` into five 26-bit limbs.
///
/// Each limb is masked after shifting: a 32-bit window shifted by 2, 4, 6 or 8 bits still holds
/// bits above 26, which would silently corrupt every product.
///
/// Cost: five 32-bit window reads and masks, `O(1)`.
fn poly1305_limbs(r: &[u8; 16]) -> [u64; 5] {
    [
        u64::from(le32_at(r, 0)) & 0x3ff_ffff,
        (u64::from(le32_at(r, 3)) >> 2) & 0x3ff_ffff,
        (u64::from(le32_at(r, 6)) >> 4) & 0x3ff_ffff,
        (u64::from(le32_at(r, 9)) >> 6) & 0x3ff_ffff,
        (u64::from(le32_at(r, 12)) >> 8) & 0x3ff_ffff,
    ]
}

/// Computes the Poly1305 tag of `msg` under the 32-byte one-time key.
///
/// Cost: `ceil(msg.len() / 16)` block absorptions, so `O(msg.len() / 16)` rounds of 25 limb
/// products on `u64` plus the final reduction and padding add. Spatial cost is 5 `h` limbs, 10
/// `r`/`5r` limbs and one 16-byte partial block, all fixed. The loop closes because each
/// iteration consumes 16 bytes.
#[must_use]
pub fn poly1305_mac(otk: &[u8; ONETIME_KEY_LEN], msg: &[u8]) -> [u8; TAG_LEN] {
    let mut r = [0u8; 16];
    let mut i = 0usize;
    while i < 16 {
        r[i] = otk[i];
        i += 1;
    }
    poly1305_clamp(&mut r);
    let mut s = [0u8; 16];
    let mut i = 0usize;
    while i < 16 {
        s[i] = otk[16 + i];
        i += 1;
    }

    let rl = poly1305_limbs(&r);
    // s_i = 5 * r_i, which keeps every product inside 64 bits.
    let mut s_limbs = [0u64; 5];
    let mut i = 0usize;
    while i < 5 {
        s_limbs[i] = rl[i] * 5;
        i += 1;
    }

    let mut h = [0u64; 5];
    let mut offset = 0usize;

    // Every block except the last one carries the implicit high bit 2^128.
    while msg.len() - offset >= 16 {
        let block = [
            msg[offset],
            msg[offset + 1],
            msg[offset + 2],
            msg[offset + 3],
            msg[offset + 4],
            msg[offset + 5],
            msg[offset + 6],
            msg[offset + 7],
            msg[offset + 8],
            msg[offset + 9],
            msg[offset + 10],
            msg[offset + 11],
            msg[offset + 12],
            msg[offset + 13],
            msg[offset + 14],
            msg[offset + 15],
        ];
        poly1305_absorb(&mut h, &block, &rl, &s_limbs, 1u64 << 24);
        offset += 16;
    }
    let rest = msg.len() - offset;
    if rest > 0 {
        // The final short block is zero-padded and carries the bit 2^8 in place of 2^128.
        let mut last = [0u8; 16];
        let mut i = 0usize;
        while i < rest {
            last[i] = msg[offset + i];
            i += 1;
        }
        last[rest] = 1;
        poly1305_absorb(&mut h, &last, &rl, &s_limbs, 0);
    }

    poly1305_finish(h, &s)
}

/// Absorbs one 16-byte block into `h`, then reduces the product back to five 26-bit limbs.
///
/// `d = h * r` over `2^130 - 5`, using the `5 * r` limbs for the reduction of the top limb.
///
/// Cost: 25 limb products, 5 carry propagations and 5 limb extractions, `O(1)`.
fn poly1305_absorb(h: &mut [u64; 5], block: &[u8; 16], rl: &[u64; 5], sl: &[u64; 5], hibit: u64) {
    let t0 = u64::from(le32_at(block, 0));
    let t1 = u64::from(le32_at(block, 4));
    let t2 = u64::from(le32_at(block, 8));
    let t3 = u64::from(le32_at(block, 12));

    h[0] += t0 & 0x3ff_ffff;
    h[1] += ((t0 >> 26) | (t1 << 6)) & 0x3ff_ffff;
    h[2] += ((t1 >> 20) | (t2 << 12)) & 0x3ff_ffff;
    h[3] += ((t2 >> 14) | (t3 << 18)) & 0x3ff_ffff;
    // The mask is essential: `hibit` must not leave spare bits in h[4], which would let it grow
    // past 64 bits across many full blocks.
    h[4] += ((t3 >> 8) | hibit) & 0x3ff_ffff;

    let d0 = h[0] * rl[0] + h[1] * sl[4] + h[2] * sl[3] + h[3] * sl[2] + h[4] * sl[1];
    let d1 = h[0] * rl[1] + h[1] * rl[0] + h[2] * sl[4] + h[3] * sl[3] + h[4] * sl[2];
    let d2 = h[0] * rl[2] + h[1] * rl[1] + h[2] * rl[0] + h[3] * sl[4] + h[4] * sl[3];
    let d3 = h[0] * rl[3] + h[1] * rl[2] + h[2] * rl[1] + h[3] * rl[0] + h[4] * sl[4];
    let d4 = h[0] * rl[4] + h[1] * rl[3] + h[2] * rl[2] + h[3] * rl[1] + h[4] * rl[0];

    // Propagate carries, keeping the top limb bounded.
    let mut c = d0 >> 26;
    h[0] = d0 & 0x3ff_ffff;
    let d1 = d1 + c;
    c = d1 >> 26;
    h[1] = d1 & 0x3ff_ffff;
    let d2 = d2 + c;
    c = d2 >> 26;
    h[2] = d2 & 0x3ff_ffff;
    let d3 = d3 + c;
    c = d3 >> 26;
    h[3] = d3 & 0x3ff_ffff;
    let d4 = d4 + c;
    c = d4 >> 26;
    h[4] = d4 & 0x3ff_ffff;
    h[0] += c * 5;
    c = h[0] >> 26;
    h[0] &= 0x3ff_ffff;
    h[1] += c;
}

/// Reduces `h` modulo `2^130 - 5`, adds the `s` pad, and serialises the tag.
///
/// Cost: two 5-step carry propagations, one 5-step conditional select, 4 word additions and 4
/// word writes, `O(1)`.
fn poly1305_finish(h: [u64; 5], s: &[u8; 16]) -> [u8; TAG_LEN] {
    let mut h = h;

    let mut c = h[1] >> 26;
    h[1] &= 0x3ff_ffff;
    h[2] += c;
    c = h[2] >> 26;
    h[2] &= 0x3ff_ffff;
    h[3] += c;
    c = h[3] >> 26;
    h[3] &= 0x3ff_ffff;
    h[4] += c;
    c = h[4] >> 26;
    h[4] &= 0x3ff_ffff;
    h[0] += c * 5;
    c = h[0] >> 26;
    h[0] &= 0x3ff_ffff;
    h[1] += c;

    // g = h + 5, kept in parallel to decide whether h >= 2^130 - 5.
    let mut g = [0u64; 5];
    let mut c = h[0].wrapping_add(5) >> 26;
    g[0] = h[0].wrapping_add(5) & 0x3ff_ffff;
    let mut i = 1usize;
    while i < 4 {
        g[i] = h[i].wrapping_add(c);
        c = g[i] >> 26;
        g[i] &= 0x3ff_ffff;
        i += 1;
    }
    g[4] = h[4].wrapping_add(c).wrapping_sub(1 << 26);

    // Select g if it did not borrow (the mask is all ones exactly then), else keep h.
    let mask = (g[4] >> 63).wrapping_sub(1);
    // Read both limbs into scalars before writing `h`, so no projection of `h` or `g` is held
    // across a write to `h` (Aeneas rejects that shape).
    let h0 = h[0];
    let h1 = h[1];
    let h2 = h[2];
    let h3 = h[3];
    let h4 = h[4];
    let g0 = g[0];
    let g1 = g[1];
    let g2 = g[2];
    let g3 = g[3];
    let g4 = g[4];
    h[0] = (h0 & !mask) | (g0 & mask);
    h[1] = (h1 & !mask) | (g1 & mask);
    h[2] = (h2 & !mask) | (g2 & mask);
    h[3] = (h3 & !mask) | (g3 & mask);
    h[4] = (h4 & !mask) | (g4 & mask);

    // Repack the limbs into four 32-bit words, keeping the low 32 bits of each.
    let f0 = (h[0] | (h[1] << 26)) as u32;
    let f1 = ((h[1] >> 6) | (h[2] << 20)) as u32;
    let f2 = ((h[2] >> 12) | (h[3] << 14)) as u32;
    let f3 = ((h[3] >> 18) | (h[4] << 8)) as u32;

    // Add the s pad with carry.
    let mut f = [f0, f1, f2, f3];
    let mut carry = 0u32;
    let mut i = 0usize;
    while i < 4 {
        let pad = le32_at(s, i * 4);
        let sum = u64::from(f[i]) + u64::from(pad) + u64::from(carry);
        f[i] = sum as u32;
        carry = (sum >> 32) as u32;
        i += 1;
    }

    let mut tag = [0u8; TAG_LEN];
    let mut i = 0usize;
    while i < 4 {
        write_le32_at(&mut tag, i * 4, f[i]);
        i += 1;
    }
    tag
}

/// Compares two tags in constant time, returning true when they are equal.
///
/// Cost: one pass over `min(a.len(), b.len())` bytes with an early return on a length mismatch;
/// the length of a tag is not secret.
fn tags_equal(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    let mut i = 0usize;
    while i < a.len() {
        diff |= a[i] ^ b[i];
        i += 1;
    }
    diff == 0
}

/// Builds the ChaCha20-Poly1305 MAC input: AAD, ciphertext, both zero-padded to a 16-byte
/// boundary, followed by their 64-bit little-endian lengths (RFC 8439 section 2.8.1).
///
/// Cost: one buffer of `aad.len() + ciphertext.len() + 32` bytes rounded up per field to a 16-byte
/// multiple, so `O(aad.len() + ciphertext.len())` bytes written; one allocation.
fn mac_input(aad: &[u8], ciphertext: &[u8]) -> Vec<u8> {
    let aad_pad = (16 - aad.len() % 16) % 16;
    let ct_pad = (16 - ciphertext.len() % 16) % 16;
    let mut buf = Vec::with_capacity(aad.len() + aad_pad + ciphertext.len() + ct_pad + 16);
    buf.extend_from_slice(aad);
    buf.resize(buf.len() + aad_pad, 0);
    buf.extend_from_slice(ciphertext);
    buf.resize(buf.len() + ct_pad, 0);
    buf.extend_from_slice(&(aad.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(ciphertext.len() as u64).to_le_bytes());
    buf
}

/// Derives the Poly1305 one-time key from ChaCha20 block 0 (RFC 8439 section 2.6).
///
/// Cost: one 20-round permutation, `O(1)`.
fn onetime_key(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN]) -> [u8; ONETIME_KEY_LEN] {
    let block = chacha20_block(key, 0, nonce);
    let mut otk = [0u8; ONETIME_KEY_LEN];
    let mut i = 0usize;
    while i < ONETIME_KEY_LEN {
        otk[i] = block[i];
        i += 1;
    }
    otk
}

/// Encrypts and authenticates `plaintext`, returning `ciphertext || tag` (RFC 8439 section 2.8.1).
///
/// The one-time key is taken from block 0 and the payload is encrypted starting at block 1.
/// Total, and infallible for well-formed lengths: `key` is a fixed-size array, so there is no
/// domain to check.
///
/// Cost: `ceil(plaintext.len() / 64)` keystream permutations for the payload plus two for the
/// MAC input over `aad.len() + ciphertext.len() + 32` bytes, and one Poly1305 pass over that
/// input. Spatial cost is the returned `plaintext.len() + 16` bytes plus one MAC-input buffer.
#[must_use]
pub fn chacha20poly1305_encrypt(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    plaintext: &[u8],
) -> Vec<u8> {
    let otk = onetime_key(key, nonce);

    let mut out = Vec::with_capacity(plaintext.len() + TAG_LEN);
    out.extend_from_slice(plaintext);
    chacha20_xor(key, 1, nonce, &mut out);

    let input = mac_input(aad, &out);
    let tag = poly1305_mac(&otk, &input);
    out.extend_from_slice(&tag);
    out
}

/// Verifies and decrypts `ciphertext || tag`, returning the plaintext (RFC 8439 section 2.8.2).
///
/// The tag is verified in constant time before any plaintext is produced; a failure returns
/// [`Error::AuthenticationFailed`] and no plaintext, and an input shorter than one tag returns
/// [`Error::CiphertextTooShort`].
///
/// Cost: one Poly1305 pass over `aad.len() + ciphertext.len() + 32` bytes, one constant-time tag
/// comparison, and on success `ceil(ciphertext.len() / 64)` keystream permutations. Spatial cost
/// is the MAC-input buffer and, on success, the `ciphertext.len()`-byte plaintext.
///
/// # Errors
///
/// [`Error::CiphertextTooShort`] when `ciphertext_and_tag` is shorter than [`TAG_LEN`], and
/// [`Error::AuthenticationFailed`] when the tag does not verify under `key`, `nonce` and `aad`. No
/// other input is rejected.
pub fn chacha20poly1305_decrypt(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    ciphertext_and_tag: &[u8],
) -> Result<Vec<u8>, Error> {
    if ciphertext_and_tag.len() < TAG_LEN {
        return Err(Error::CiphertextTooShort);
    }
    let split = ciphertext_and_tag.len() - TAG_LEN;
    let ciphertext = &ciphertext_and_tag[..split];
    let tag = &ciphertext_and_tag[split..];

    let otk = onetime_key(key, nonce);
    let input = mac_input(aad, ciphertext);
    let expected = poly1305_mac(&otk, &input);
    if !tags_equal(&expected, tag) {
        return Err(Error::AuthenticationFailed);
    }

    let mut out = Vec::with_capacity(ciphertext.len());
    out.extend_from_slice(ciphertext);
    chacha20_xor(key, 1, nonce, &mut out);
    Ok(out)
}

/// Encrypts with byte-slice `key` and `nonce`, checking their lengths.
///
/// The only partial cases are the two length checks: a `key` of [`KEY_LEN`] bytes and a `nonce`
/// of [`NONCE_LEN`] bytes succeed, any other length returns [`Error::InvalidKeyLength`] or
/// [`Error::InvalidNonceLength`] without touching `plaintext`.
///
/// # Errors
///
/// [`Error::InvalidKeyLength`] when `key` is not [`KEY_LEN`] bytes, and
/// [`Error::InvalidNonceLength`] when `nonce` is not [`NONCE_LEN`] bytes. The key length is checked
/// first; on success there is no further failure.
pub fn chacha20poly1305_encrypt_slice(
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>, Error> {
    let key: &[u8; KEY_LEN] = key.try_into().map_err(|_| Error::InvalidKeyLength)?;
    let nonce: &[u8; NONCE_LEN] = nonce.try_into().map_err(|_| Error::InvalidNonceLength)?;
    Ok(chacha20poly1305_encrypt(key, nonce, aad, plaintext))
}

/// Decrypts with byte-slice `key` and `nonce`, checking their lengths.
///
/// The length failures are [`Error::InvalidKeyLength`] and [`Error::InvalidNonceLength`], checked
/// before the ciphertext; an authentic but wrong-length key or nonce never decrypts.
///
/// # Errors
///
/// [`Error::InvalidKeyLength`] when `key` is not [`KEY_LEN`] bytes, [`Error::InvalidNonceLength`]
/// when `nonce` is not [`NONCE_LEN`] bytes, and otherwise whatever
/// [`chacha20poly1305_decrypt`] returns for the ciphertext. The key length is checked first.
pub fn chacha20poly1305_decrypt_slice(
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
    ciphertext_and_tag: &[u8],
) -> Result<Vec<u8>, Error> {
    let key: &[u8; KEY_LEN] = key.try_into().map_err(|_| Error::InvalidKeyLength)?;
    let nonce: &[u8; NONCE_LEN] = nonce.try_into().map_err(|_| Error::InvalidNonceLength)?;
    chacha20poly1305_decrypt(key, nonce, aad, ciphertext_and_tag)
}
