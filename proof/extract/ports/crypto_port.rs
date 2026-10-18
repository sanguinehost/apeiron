//! Extraction roots for the hashtrinity crypto face (`src/map/hashtrinity/crypto.rs`).
//!
//! The module is already in the translation unit as `crate::hashtrinity` (declared in `lib.rs` over
//! the real `src/map/hashtrinity/mod.rs`), so each `word_crypto_*` wrapper fixes the monomorphic
//! root and calls the real `crate::hashtrinity::crypto::<module>::<fn>` body; it restates no
//! formula. Roots are grouped as the public surface is: hashes, MAC/KDF, AEAD/stream cipher,
//! signatures and the protocol helpers.
//!
//! A root whose body reaches an item the pinned Aeneas does not model (a `Vec` mutation without a
//! model, `try_into` at a slice width, an OS read) surfaces as an unlisted axiom and is dropped
//! here; the refusal is recorded against its name, not waived.

use crate::hashtrinity::crypto;

// ── Hashes ───────────────────────────────────────────────────────────────────────────────────────

/// Hex is not rooted: `to_hex` reaches the `ToHex` iterator implementation (`hex.rs`).
pub fn word_crypto_ct_eq(a: &[u8], b: &[u8]) -> bool {
    crypto::ct_eq(a, b)
}

/// FIPS 180-4 SHA-256 at the concrete `&[u8]`/`[u8; 32]` types.
pub fn word_crypto_sha256(msg: &[u8]) -> [u8; 32] {
    crypto::sha256(msg)
}

/// FIPS 180-4 SHA-512 at the concrete `&[u8]`/`[u8; 64]` types.
pub fn word_crypto_sha512(msg: &[u8]) -> [u8; 64] {
    crypto::sha512(msg)
}

/// FIPS 180-4 SHA-384 (SHA-512 fold, 384 IV, truncated to 48 bytes).
pub fn word_crypto_sha384(msg: &[u8]) -> [u8; 48] {
    crypto::sha384(msg)
}

/// RFC 1321 MD5 at the concrete `&[u8]`/`[u8; 16]` types.
pub fn word_crypto_md5(msg: &[u8]) -> [u8; 16] {
    crypto::md5(msg)
}

/// RFC 7693 BLAKE2b at a caller-fixed output length.
pub fn word_crypto_blake2b(msg: &[u8], out_len: usize) -> Vec<u8> {
    crypto::blake2::blake2b(msg, out_len)
}

/// RFC 7693 BLAKE2b-512.
pub fn word_crypto_blake2b512(msg: &[u8]) -> [u8; 64] {
    crypto::blake2::blake2b512(msg)
}

/// RIPEMD-160 (the Bitcoin address hash's outer fold).
pub fn word_crypto_ripemd160(msg: &[u8]) -> [u8; 20] {
    crypto::ripemd160::ripemd160(msg)
}

/// Bitcoin `Hash160` = RIPEMD160(SHA256(x)).
pub fn word_crypto_hash160(data: &[u8]) -> [u8; 20] {
    crypto::ripemd160::hash160(data)
}

// ── MAC / KDF ────────────────────────────────────────────────────────────────────────────────────

/// RFC 2104 HMAC-SHA256.
pub fn word_crypto_hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    crypto::hmac_sha256(key, msg)
}

/// RFC 2104 HMAC-SHA512.
pub fn word_crypto_hmac_sha512(key: &[u8], msg: &[u8]) -> [u8; 64] {
    crypto::hmac_sha512(key, msg)
}

/// RFC 2104 HMAC-SHA384.
pub fn word_crypto_hmac_sha384(key: &[u8], msg: &[u8]) -> [u8; 48] {
    crypto::hmac_sha384(key, msg)
}

/// RFC 8018 PBKDF2-HMAC-SHA256.
pub fn word_crypto_pbkdf2_sha256(pass: &[u8], salt: &[u8], iters: u32, dklen: usize) -> Vec<u8> {
    crypto::pbkdf2_sha256(pass, salt, iters, dklen)
}

/// RFC 5869 HKDF-Extract (SHA-256).
pub fn word_crypto_hkdf_extract_sha256(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    crypto::hkdf_extract_sha256(salt, ikm)
}

/// RFC 5869 HKDF-Expand (SHA-256).
pub fn word_crypto_hkdf_expand_sha256(prk: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    crypto::hkdf_expand_sha256(prk, info, len)
}

/// RFC 5869 HKDF (SHA-256).
pub fn word_crypto_hkdf_sha256(salt: &[u8], ikm: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    crypto::hkdf_sha256(salt, ikm, info, len)
}

/// RFC 5869 HKDF (SHA-384).
pub fn word_crypto_hkdf_sha384(salt: &[u8], ikm: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    crypto::hkdf_sha384(salt, ikm, info, len)
}

/// RFC 5869 HKDF (SHA-512).
pub fn word_crypto_hkdf_sha512(salt: &[u8], ikm: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    crypto::hkdf_sha512(salt, ikm, info, len)
}

// ── AEAD / stream cipher ────────────────────────────────────────────────────────────────────────

/// RFC 8439 section 2.3 ChaCha20 keystream block.
pub fn word_crypto_chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    crypto::chacha_poly::chacha20_block(key, counter, nonce)
}

// ── Signatures ──────────────────────────────────────────────────────────────────────────────────

/// FIPS 186-4 ECDSA P-256 verification at the raw `r‖s` signature.
pub fn word_crypto_ecdsa_p256_verify(pubkey: &[u8; 64], hash: &[u8; 32], sig: &[u8; 64]) -> bool {
    crypto::ecdsa_p256_verify(pubkey, hash, sig)
}

/// P-256 public key from a 32-byte secret.
pub fn word_crypto_p256_pubkey(secret: &[u8; 32]) -> [u8; 64] {
    crypto::p256_pubkey(secret)
}

/// secp256k1 public key (uncompressed `x‖y`).
pub fn word_crypto_secp256k1_pubkey(secret: &[u8; 32]) -> [u8; 64] {
    crypto::secp256k1::pubkey(secret)
}

/// FIPS 186-4 ECDSA P-384 verification.
pub fn word_crypto_ecdsa_p384_verify(
    pubkey: &[u8; 96],
    hash: &[u8; 48],
    sig: &[u8; 96],
) -> bool {
    crypto::p384::ecdsa_p384_verify(pubkey, hash, sig)
}

/// RSASSA-PKCS1-v1_5 SHA-256 verification over a big-endian modulus.
pub fn word_crypto_rsa_pkcs1_sha256_verify(
    n_be: &[u8],
    e: u64,
    sig_be: &[u8],
    msg_hash: &[u8; 32],
) -> bool {
    crypto::rsa::rsa_pkcs1_sha256_verify(n_be, e, sig_be, msg_hash)
}

// ── Protocol helpers ───────────────────────────────────────────────────────────────────────────

/// BIP-173/BIP-350 8-bit to 5-bit regrouping.
pub fn word_crypto_bech32_convert_bits(
    data: &[u8],
    from: u32,
    to: u32,
    pad: bool,
) -> Option<Vec<u8>> {
    crypto::bech32::convert_bits(data, from, to, pad)
}

// ── Big-integer limbs, DER and the RSA seams ────────────────────────────────────────────────────
//
// `Bn` is the real `Vec<u64>` little-endian limb vector (`type Bn = Vec<u64>` in `rsa`); a wrapper
// names it at the concrete `Vec<u64>` type and calls the real body.

/// `rsa::bn_from_u64`: the one-limb value.
pub fn word_crypto2_bn_from_u64(x: u64) -> Vec<u64> {
    crypto::rsa::bn_from_u64(x)
}

/// `rsa::bn_is_zero`.
pub fn word_crypto2_bn_is_zero(a: &Vec<u64>) -> bool {
    crypto::rsa::bn_is_zero(a)
}

/// `rsa::bn_is_even`.
pub fn word_crypto2_bn_is_even(a: &Vec<u64>) -> bool {
    crypto::rsa::bn_is_even(a)
}

/// `rsa::bn_add`: limb-wise carry add.
pub fn word_crypto2_bn_add(a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
    crypto::rsa::bn_add(a, b)
}

/// `rsa::bn_shr1`: the right shift one used by Miller-Rabin's `2^s * d` split.
pub fn word_crypto2_bn_shr1(a: &Vec<u64>) -> Vec<u64> {
    crypto::rsa::bn_shr1(a)
}

/// `rsa::bn_sub1`.
pub fn word_crypto2_bn_sub1(a: &Vec<u64>) -> Vec<u64> {
    crypto::rsa::bn_sub1(a)
}

/// `rsa::bn_divmod`: `(x / m, x % m)` by bit-by-bit long division.
pub fn word_crypto2_bn_divmod(x: &Vec<u64>, m: &Vec<u64>) -> (Vec<u64>, Vec<u64>) {
    crypto::rsa::bn_divmod(x, m)
}

/// `rsa::bn_modexp_big`: square-and-multiply over every bit of a bignum exponent.
pub fn word_crypto2_bn_modexp_big(base: &Vec<u64>, e: &Vec<u64>, m: &Vec<u64>) -> Vec<u64> {
    crypto::rsa::bn_modexp_big(base, e, m)
}

/// `rsa::mod_inverse`: `e^-1 mod phi` by the extended Euclidean algorithm.
pub fn word_crypto2_mod_inverse(e: u64, phi: &Vec<u64>) -> Option<Vec<u64>> {
    crypto::rsa::mod_inverse(e, phi)
}

/// `rsa::bitlen`: the bit length of a big-endian integer.
pub fn word_crypto2_bitlen(n_be: &[u8]) -> usize {
    crypto::rsa::bitlen(n_be)
}

/// `rsa::mgf1`: the RFC 8017 MGF1 mask over SHA-256.
pub fn word_crypto2_mgf1(seed: &[u8], len: usize) -> Vec<u8> {
    crypto::rsa::mgf1(seed, len)
}

/// `rsa::emsa_pss_encode`: the EMSA-PSS encoded message.
pub fn word_crypto2_emsa_pss_encode(m_hash: &[u8; 32], em_bits: usize, salt: &[u8]) -> Vec<u8> {
    crypto::rsa::emsa_pss_encode(m_hash, em_bits, salt)
}

/// `rsa::rsa_pkcs1_sign_generic`: `sig = EM^d mod n` for a DigestInfo prefix.
pub fn word_crypto2_rsa_pkcs1_sign_generic(
    n_be: &[u8],
    d_be: &[u8],
    di: &[u8],
    hash: &[u8],
) -> Vec<u8> {
    crypto::rsa::rsa_pkcs1_sign_generic(n_be, d_be, di, hash)
}

/// `rsa::rsa_pkcs1_sha384_verify`.
pub fn word_crypto2_rsa_pkcs1_sha384_verify(
    n_be: &[u8],
    e: u64,
    sig_be: &[u8],
    msg_hash: &[u8; 48],
) -> bool {
    crypto::rsa::rsa_pkcs1_sha384_verify(n_be, e, sig_be, msg_hash)
}

/// `rsa::rsa_pkcs1_sha512_verify`.
pub fn word_crypto2_rsa_pkcs1_sha512_verify(
    n_be: &[u8],
    e: u64,
    sig_be: &[u8],
    msg_hash: &[u8; 64],
) -> bool {
    crypto::rsa::rsa_pkcs1_sha512_verify(n_be, e, sig_be, msg_hash)
}

/// `rsa::rsa_pkcs1_sha256_sign`.
pub fn word_crypto2_rsa_pkcs1_sha256_sign(
    n_be: &[u8],
    d_be: &[u8],
    msg_hash: &[u8; 32],
) -> Vec<u8> {
    crypto::rsa::rsa_pkcs1_sha256_sign(n_be, d_be, msg_hash)
}

/// `rsa::rsa_pkcs1_sha384_sign`.
pub fn word_crypto2_rsa_pkcs1_sha384_sign(
    n_be: &[u8],
    d_be: &[u8],
    msg_hash: &[u8; 48],
) -> Vec<u8> {
    crypto::rsa::rsa_pkcs1_sha384_sign(n_be, d_be, msg_hash)
}

/// `rsa::rsa_pkcs1_sha512_sign`.
pub fn word_crypto2_rsa_pkcs1_sha512_sign(
    n_be: &[u8],
    d_be: &[u8],
    msg_hash: &[u8; 64],
) -> Vec<u8> {
    crypto::rsa::rsa_pkcs1_sha512_sign(n_be, d_be, msg_hash)
}

/// `rsa::rsa_pss_sha256_sign`.
pub fn word_crypto2_rsa_pss_sha256_sign(
    n_be: &[u8],
    d_be: &[u8],
    m_hash: &[u8; 32],
    salt: &[u8],
) -> Vec<u8> {
    crypto::rsa::rsa_pss_sha256_sign(n_be, d_be, m_hash, salt)
}

/// `rsa::rsa_pss_sha256_verify`.
pub fn word_crypto2_rsa_pss_sha256_verify(
    n_be: &[u8],
    e: u64,
    sig_be: &[u8],
    m_hash: &[u8; 32],
    s_len: usize,
) -> bool {
    crypto::rsa::rsa_pss_sha256_verify(n_be, e, sig_be, m_hash, s_len)
}

/// `rsa::rsa_algid`: the rsaEncryption AlgorithmIdentifier.
pub fn word_crypto2_rsa_algid() -> Vec<u8> {
    crypto::rsa::rsa_algid()
}

/// `rsa::trim_int`: strip the DER sign-guard byte.
pub fn word_crypto2_trim_int(b: &[u8]) -> Vec<u8> {
    crypto::rsa::trim_int(b)
}

/// `rsa::rsa_to_pkcs8_der`.
pub fn word_crypto2_rsa_to_pkcs8_der(n_be: &[u8], e: u64, d_be: &[u8]) -> Vec<u8> {
    crypto::rsa::rsa_to_pkcs8_der(n_be, e, d_be)
}

/// `rsa::rsa_from_pkcs8_der`.
pub fn word_crypto2_rsa_from_pkcs8_der(d: &[u8]) -> Option<(Vec<u8>, u64, Vec<u8>)> {
    crypto::rsa::rsa_from_pkcs8_der(d)
}

// ── DER encoder (`certgen`) and TLV walker (`x509`) ─────────────────────────────────────────────

/// `certgen::der_len`: the DER length octets.
pub fn word_crypto2_der_len(n: usize) -> Vec<u8> {
    crypto::certgen::der_len(n)
}

/// `certgen::tlv`: tag-length-value.
pub fn word_crypto2_der_tlv(tag: u8, c: &[u8]) -> Vec<u8> {
    crypto::certgen::tlv(tag, c)
}

/// `certgen::seq`: a SEQUENCE over the encoded items.
pub fn word_crypto2_der_seq(items: &[Vec<u8>]) -> Vec<u8> {
    crypto::certgen::seq(items)
}

/// `certgen::set`: a SET.
pub fn word_crypto2_der_set(inner: &[u8]) -> Vec<u8> {
    crypto::certgen::set(inner)
}

/// `certgen::der_int`: a non-negative INTEGER from a big-endian magnitude.
pub fn word_crypto2_der_int(mag: &[u8]) -> Vec<u8> {
    crypto::certgen::der_int(mag)
}

/// `certgen::oid`.
pub fn word_crypto2_der_oid(bytes: &[u8]) -> Vec<u8> {
    crypto::certgen::oid(bytes)
}

/// `certgen::bitstr` (0 unused bits).
pub fn word_crypto2_der_bitstr(data: &[u8]) -> Vec<u8> {
    crypto::certgen::bitstr(data)
}

/// `certgen::ctx_explicit`: `[n] EXPLICIT`.
pub fn word_crypto2_der_ctx_explicit(n: u8, inner: &[u8]) -> Vec<u8> {
    crypto::certgen::ctx_explicit(n, inner)
}

/// `certgen::algid_ed25519`.
pub fn word_crypto2_der_algid_ed25519() -> Vec<u8> {
    crypto::certgen::algid_ed25519()
}

/// `certgen::name_cn`: the commonName RDN sequence.
pub fn word_crypto2_der_name_cn(cn: &str) -> Vec<u8> {
    crypto::certgen::name_cn(cn)
}

/// `certgen::validity`: the fixed window.
pub fn word_crypto2_der_validity() -> Vec<u8> {
    crypto::certgen::validity()
}

/// `certgen::spki`: SubjectPublicKeyInfo over a 32-byte ed25519 key.
pub fn word_crypto2_der_spki(pubkey: &[u8; 32]) -> Vec<u8> {
    crypto::certgen::spki(pubkey)
}

/// `x509::der_tlv`: one DER element at `off`.
pub fn word_crypto2_x509_der_tlv(d: &[u8], off: usize) -> Option<(u8, usize, usize, usize)> {
    crypto::x509::der_tlv(d, off)
}

/// `x509::der_children`: the immediate children of a constructed element.
pub fn word_crypto2_x509_der_children(d: &[u8], cs: usize, ce: usize) -> Vec<(u8, usize, usize)> {
    crypto::x509::der_children(d, cs, ce)
}

// ── Base64 decoders ─────────────────────────────────────────────────────────────────────────────

/// `age::b64d`: standard base64 without padding.
pub fn word_crypto2_age_b64d(s: &str) -> Option<Vec<u8>> {
    crypto::age::b64d(s)
}

/// `jwt::b64url_decode`: base64url without padding.
pub fn word_crypto2_jwt_b64url_decode(s: &str) -> Option<Vec<u8>> {
    crypto::jwt::b64url_decode(s)
}