//! The cryptographic primitives. None is audited and none is side-channel reviewed: do not use
//! them to protect real data. Their rooting onto `R[x]/(x² − σ)` is in progress.

/// AES-128 (FIPS 197). Tested against the SP 800-38A known-answer vector only. Not side-channel
/// resistant: the S-box is indexed by secret bytes.
pub mod aes;
/// ChaCha20-Poly1305 (RFC 8439). Tested against the RFC 8439 known-answer vectors only. Its
/// constant-time construction is not measured.
pub mod chacha;
/// SHA-3 and SHAKE (FIPS 202). Tested against the NIST known-answer vectors only.
pub mod keccak;
/// SHA-256 (FIPS 180-4). Tested against the FIPS 180-4 known-answer vectors only.
pub mod sha256;
