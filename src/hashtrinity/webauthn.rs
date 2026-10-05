//! A native WebAuthn Level 2 relying-party implementation — passkey registration and
//! authentication only (the sole flow any consumer in this ecosystem uses), over
//! hashtrinity-crypto's existing ECDSA-P256/RSA-PKCS1v1.5/SHA-256 primitives. No `openssl`, no
//! vendored attestation-CA trust bundle: attestation-statement verification is out of scope by
//! design (every consumer's config requests `AttestationConveyancePreference::None` with no CA
//! list, so upstream `webauthn-rs` itself never verifies an attestation chain either — only the
//! credential's own assertion signature is ever checked, both here and there).
//!
//! Sibling crates depend on this via Cargo's `package = "hashtrinity-webauthn"` rename, e.g.
//! `webauthn-rs = { package = "hashtrinity-webauthn", path = "..." }` — so `use
//! webauthn_rs::prelude::*;` at the call site needs no changes at all, same as `webauthn-rs-proto`
//! for the handful of `webauthn_rs_proto::{...}` direct (non-prelude) imports.
pub mod b64data {
    //! `Base64UrlSafeData` — every binary field in the WebAuthn JSON wire format (challenge, credential
    //! IDs, `attestationObject`, `clientDataJSON`, signatures, `rawId`, `userHandle`) is RFC 4648 §5
    //! URL-safe, no-pad base64 as a JSON string. Reuses `crate::webauthn::base64`'s `URL_SAFE_NO_PAD` engine
    //! (the same masquerade already used throughout the sauthy/hashtrinity ecosystem).
    use crate::base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use serde::de::{Error as DeError, Visitor};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::fmt;

    #[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
    pub struct Base64UrlSafeData(pub Vec<u8>);

    impl From<Vec<u8>> for Base64UrlSafeData {
        fn from(v: Vec<u8>) -> Self {
            Self(v)
        }
    }
    impl AsRef<[u8]> for Base64UrlSafeData {
        fn as_ref(&self) -> &[u8] {
            &self.0
        }
    }
    /// Mirrors the real crate's `HumanBinaryData` — it derefs to `Vec<u8>`, so call sites reach straight
    /// through for e.g. `.to_vec()`/`.len()`/slicing without an explicit `.0`.
    impl std::ops::Deref for Base64UrlSafeData {
        type Target = Vec<u8>;
        fn deref(&self) -> &Vec<u8> {
            &self.0
        }
    }
    impl std::ops::DerefMut for Base64UrlSafeData {
        fn deref_mut(&mut self) -> &mut Vec<u8> {
            &mut self.0
        }
    }

    impl Serialize for Base64UrlSafeData {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            s.serialize_str(&URL_SAFE_NO_PAD.encode(&self.0))
        }
    }

    struct B64Visitor;
    impl Visitor<'_> for B64Visitor {
        type Value = Base64UrlSafeData;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a base64url string")
        }
        fn visit_str<E: DeError>(self, v: &str) -> Result<Self::Value, E> {
            URL_SAFE_NO_PAD
                .decode(v)
                .map(Base64UrlSafeData)
                .map_err(|e| E::custom(format!("invalid base64url: {e:?}")))
        }
    }
    impl<'de> Deserialize<'de> for Base64UrlSafeData {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            d.deserialize_str(B64Visitor)
        }
    }

    /// A raw credential ID — the real crate aliases `CredentialID = HumanBinaryData`, same base64url
    /// wire shape as every other binary field here.
    pub type CredentialID = Base64UrlSafeData;

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn round_trips_through_json() {
            let v = Base64UrlSafeData(vec![0, 1, 2, 250, 251, 252, 253, 254, 255]);
            let json = serde_json::to_string(&v).unwrap();
            assert!(!json.contains('+') && !json.contains('/') && !json.contains('='));
            let back: Base64UrlSafeData = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }
}
pub mod cbor {
    //! A minimal CBOR (RFC 8949) decoder — just the subset WebAuthn's `attestationObject` and
    //! `credentialPublicKey` (COSE_Key) actually use: unsigned/negative integers, byte strings, text
    //! strings, arrays, maps, and the two booleans, all DEFINITE-length (canonical CBOR, which is what
    //! CTAP2/WebAuthn authenticators emit — indefinite-length items never appear here). No encoder: this
    //! crate only ever PARSES CBOR a browser/authenticator produced, never emits it.

    #[derive(Debug, Clone, PartialEq)]
    pub enum CborValue {
        UInt(u64),
        NInt(i64), // already resolved to the actual negative value
        Bytes(Vec<u8>),
        Text(String),
        Array(Vec<CborValue>),
        Map(Vec<(CborValue, CborValue)>), // small maps in practice; linear lookup is fine
        Bool(bool),
        Null,
    }

    impl CborValue {
        pub fn as_bytes(&self) -> Option<&[u8]> {
            if let CborValue::Bytes(b) = self {
                Some(b)
            } else {
                None
            }
        }
        pub fn as_text(&self) -> Option<&str> {
            if let CborValue::Text(t) = self {
                Some(t)
            } else {
                None
            }
        }
        pub fn as_uint(&self) -> Option<u64> {
            if let CborValue::UInt(u) = self {
                Some(*u)
            } else {
                None
            }
        }
        pub fn as_map(&self) -> Option<&[(CborValue, CborValue)]> {
            if let CborValue::Map(m) = self {
                Some(m)
            } else {
                None
            }
        }
        /// Look up a map value by an integer key (COSE_Key fields are always small ints, positive or negative).
        pub fn get_int(&self, key: i64) -> Option<&CborValue> {
            self.as_map()?
                .iter()
                .find(|(k, _)| match k {
                    CborValue::UInt(u) => *u as i64 == key,
                    CborValue::NInt(n) => *n == key,
                    _ => false,
                })
                .map(|(_, v)| v)
        }
    }

    fn read_len(d: &[u8], off: usize, info: u8) -> Option<(u64, usize)> {
        match info {
            0..=23 => Some((info as u64, off)),
            24 => Some((*d.get(off)? as u64, off + 1)),
            25 => Some((
                u16::from_be_bytes(d.get(off..off + 2)?.try_into().ok()?) as u64,
                off + 2,
            )),
            26 => Some((
                u32::from_be_bytes(d.get(off..off + 4)?.try_into().ok()?) as u64,
                off + 4,
            )),
            27 => Some((
                u64::from_be_bytes(d.get(off..off + 8)?.try_into().ok()?),
                off + 8,
            )),
            _ => None, // indefinite-length / reserved — not used by canonical CBOR, unsupported here
        }
    }

    /// Decode ONE CBOR item starting at `off`. Returns the value and the offset just past it.
    pub fn decode(d: &[u8], off: usize) -> Option<(CborValue, usize)> {
        let b0 = *d.get(off)?;
        let major = b0 >> 5;
        let info = b0 & 0x1f;
        let off = off + 1;
        match major {
            0 => {
                let (v, off) = read_len(d, off, info)?;
                Some((CborValue::UInt(v), off))
            }
            1 => {
                let (v, off) = read_len(d, off, info)?;
                Some((CborValue::NInt(-1 - v as i64), off))
            }
            2 => {
                let (len, off) = read_len(d, off, info)?;
                let len = len as usize;
                Some((CborValue::Bytes(d.get(off..off + len)?.to_vec()), off + len))
            }
            3 => {
                let (len, off) = read_len(d, off, info)?;
                let len = len as usize;
                let s = String::from_utf8(d.get(off..off + len)?.to_vec()).ok()?;
                Some((CborValue::Text(s), off + len))
            }
            4 => {
                let (n, mut off) = read_len(d, off, info)?;
                let mut items = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    let (v, next) = decode(d, off)?;
                    items.push(v);
                    off = next;
                }
                Some((CborValue::Array(items), off))
            }
            5 => {
                let (n, mut off) = read_len(d, off, info)?;
                let mut items = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    let (k, next) = decode(d, off)?;
                    off = next;
                    let (v, next2) = decode(d, off)?;
                    off = next2;
                    items.push((k, v));
                }
                Some((CborValue::Map(items), off))
            }
            7 => match info {
                20 => Some((CborValue::Bool(false), off)),
                21 => Some((CborValue::Bool(true), off)),
                22 | 23 => Some((CborValue::Null, off)), // null / undefined, treated the same (unused fields)
                _ => None,
            },
            _ => None,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn decodes_flat_map() {
            // {"fmt": "none"} — 0xa1 (map,1) 0x63 "fmt" 0x64 "none"
            let bytes = [0xa1, 0x63, b'f', b'm', b't', 0x64, b'n', b'o', b'n', b'e'];
            let (v, next) = decode(&bytes, 0).unwrap();
            assert_eq!(next, bytes.len());
            let m = v.as_map().unwrap();
            assert_eq!(m.len(), 1);
            assert_eq!(m[0].0.as_text(), Some("fmt"));
            assert_eq!(m[0].1.as_text(), Some("none"));
        }

        #[test]
        fn decodes_cose_key_shape() {
            // COSE EC2 key: {1: 2, 3: -7, -1: 1, -2: h'01..', -3: h'02..'} with small ints as map keys
            let x = [1u8; 32];
            let y = [2u8; 32];
            let mut bytes = vec![0xa5]; // map(5)
            bytes.extend([0x01, 0x02]); // 1: 2 (kty EC2)
            bytes.extend([0x03, 0x26]); // 3: -7 (alg ES256) -> NInt encoding: -7 = -1-6, info=6 -> 0x26
            bytes.extend([0x20, 0x01]); // -1: 1 (crv P-256) -> key -1 encoded as 0x20
            bytes.push(0x21);
            bytes.push(0x58);
            bytes.push(32);
            bytes.extend(x); // -2: bytes(32)
            bytes.push(0x22);
            bytes.push(0x58);
            bytes.push(32);
            bytes.extend(y); // -3: bytes(32)
            let (v, next) = decode(&bytes, 0).unwrap();
            assert_eq!(next, bytes.len());
            assert_eq!(v.get_int(1).unwrap().as_uint(), Some(2));
            if let CborValue::NInt(alg) = v.get_int(3).unwrap() {
                assert_eq!(*alg, -7);
            } else {
                panic!("expected NInt");
            }
            assert_eq!(v.get_int(-2).unwrap().as_bytes(), Some(&x[..]));
            assert_eq!(v.get_int(-3).unwrap().as_bytes(), Some(&y[..]));
        }
    }
}
pub mod cose {
    //! COSE_Key (RFC 9053) parsing and `authenticatorData` (WebAuthn §6.1) parsing — the two structures
    //! an authenticator's CBOR actually carries. Only the two algorithms sauthy's `Webauthn` config ever
    //! negotiates (`COSEAlgorithm::secure_algs() == [ES256, RS256]`) are represented; verification for
    //! both reuses hashtrinity-crypto's existing ECDSA-P256 / RSA-PKCS1v1.5 primitives unchanged.
    use crate::crypto::{ecdsa_p256_verify, rsa_pkcs1_sha256_verify, sha256};
    use crate::webauthn::cbor::{decode, CborValue};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum CosePublicKey {
        Es256 { x: [u8; 32], y: [u8; 32] },
        Rs256 { n: Vec<u8>, e: Vec<u8> },
    }

    impl CosePublicKey {
        /// COSE algorithm identifier (RFC 9053 §2 / IANA COSE registry) — ES256 = -7, RS256 = -257.
        pub fn cose_alg(&self) -> i64 {
            match self {
                CosePublicKey::Es256 { .. } => -7,
                CosePublicKey::Rs256 { .. } => -257,
            }
        }

        /// Verify a WebAuthn signature: `sig` over `signed_data` (already `authData ++ sha256(clientDataJSON)`
        /// per §7.2 step 20 — the caller assembles that; this only does the algorithm-specific check).
        pub fn verify(&self, signed_data: &[u8], sig: &[u8]) -> bool {
            let hash = sha256(signed_data);
            match self {
                CosePublicKey::Es256 { x, y } => {
                    let Some((r, s)) = der_decode_ecdsa_sig(sig) else {
                        return false;
                    };
                    let mut pubkey = [0u8; 64];
                    pubkey[..32].copy_from_slice(x);
                    pubkey[32..].copy_from_slice(y);
                    let mut raw_sig = [0u8; 64];
                    raw_sig[..32].copy_from_slice(&r);
                    raw_sig[32..].copy_from_slice(&s);
                    ecdsa_p256_verify(&pubkey, &hash, &raw_sig)
                }
                CosePublicKey::Rs256 { n, e } => {
                    let mut e_val = 0u64;
                    for &b in e.iter() {
                        e_val = (e_val << 8) | b as u64;
                    }
                    rsa_pkcs1_sha256_verify(n, e_val, sig, &hash)
                }
            }
        }
    }

    /// Left-pad/trim a DER INTEGER's magnitude to exactly 32 bytes (P-256 field element width). DER
    /// integers prepend a `0x00` sign guard when the top bit is set, and may be naturally shorter than
    /// 32 bytes when the value itself has leading zero bytes.
    fn fixed32(mag: &[u8]) -> Option<[u8; 32]> {
        let trimmed = if mag.len() > 1 && mag[0] == 0 {
            &mag[1..]
        } else {
            mag
        };
        if trimmed.len() > 32 {
            return None;
        }
        let mut out = [0u8; 32];
        out[32 - trimmed.len()..].copy_from_slice(trimmed);
        Some(out)
    }

    /// ECDSA-Sig-Value ::= SEQUENCE { r INTEGER, s INTEGER } — the DER shape WebAuthn's ES256 assertion
    /// signatures are always encoded in (RFC 3279 §2.2.3), unlike hashtrinity-crypto's raw-r‖s internal
    /// convention. Reuses the same DER TLV walker `x509.rs`/`ed25519.rs`'s masquerade already use.
    fn der_decode_ecdsa_sig(d: &[u8]) -> Option<([u8; 32], [u8; 32])> {
        let (tag, cs, ce, _) = crate::crypto::x509::der_tlv(d, 0)?;
        if tag != 0x30 {
            return None;
        }
        let children = crate::crypto::x509::der_children(d, cs, ce);
        let (rt, rs, re) = *children.first()?;
        let (st, ss, se) = *children.get(1)?;
        if rt != 0x02 || st != 0x02 {
            return None;
        }
        Some((fixed32(&d[rs..re])?, fixed32(&d[ss..se])?))
    }

    /// Parse a COSE_Key CBOR map into the two shapes WebAuthn's negotiated algorithms produce.
    pub fn parse_cose_key(v: &CborValue) -> Option<CosePublicKey> {
        let kty = v.get_int(1)?.as_uint()?; // 1 = EC2, 2 = OKP, 3 = RSA (RFC 9053 Table 21)
        match kty {
            2 => {
                let x = v.get_int(-2)?.as_bytes()?;
                let y = v.get_int(-3)?.as_bytes()?;
                if x.len() != 32 || y.len() != 32 {
                    return None;
                }
                let mut xa = [0u8; 32];
                xa.copy_from_slice(x);
                let mut ya = [0u8; 32];
                ya.copy_from_slice(y);
                Some(CosePublicKey::Es256 { x: xa, y: ya })
            }
            3 => {
                let n = v.get_int(-1)?.as_bytes()?.to_vec();
                let e = v.get_int(-2)?.as_bytes()?.to_vec();
                Some(CosePublicKey::Rs256 { n, e })
            }
            _ => None,
        }
    }

    /// The parsed `authenticatorData` (WebAuthn §6.1). `attested` is `Some` only for registration
    /// (the AT flag set); authentication's authData never carries a fresh credential.
    pub struct AuthData {
        pub rp_id_hash: [u8; 32],
        pub user_present: bool,
        pub user_verified: bool,
        pub backup_eligible: bool,
        pub backup_state: bool,
        pub sign_count: u32,
        pub attested: Option<AttestedCredentialData>,
    }
    pub struct AttestedCredentialData {
        pub credential_id: Vec<u8>,
        pub public_key: CosePublicKey,
    }

    pub fn parse_auth_data(d: &[u8]) -> Option<AuthData> {
        if d.len() < 37 {
            return None;
        }
        let mut rp_id_hash = [0u8; 32];
        rp_id_hash.copy_from_slice(&d[0..32]);
        let flags = d[32];
        let sign_count = u32::from_be_bytes(d[33..37].try_into().ok()?);
        let at_flag = flags & 0x40 != 0;
        let attested = if at_flag {
            let mut off = 37;
            if d.len() < off + 18 {
                return None;
            } // 16-byte AAGUID + 2-byte credIdLen
            off += 16; // AAGUID — unused by sauthy, skip
            let cred_id_len = u16::from_be_bytes(d[off..off + 2].try_into().ok()?) as usize;
            off += 2;
            let credential_id = d.get(off..off + cred_id_len)?.to_vec();
            off += cred_id_len;
            let (key_cbor, _next) = decode(d, off)?;
            let public_key = parse_cose_key(&key_cbor)?;
            Some(AttestedCredentialData {
                credential_id,
                public_key,
            })
        } else {
            None
        };
        Some(AuthData {
            rp_id_hash,
            user_present: flags & 0x01 != 0,
            user_verified: flags & 0x04 != 0,
            backup_eligible: flags & 0x08 != 0,
            backup_state: flags & 0x10 != 0,
            sign_count,
            attested,
        })
    }

    /// `attestationObject` (registration response) — `{"fmt": tstr, "authData": bstr, "attStmt": map}`.
    /// `attStmt` is never inspected: sauthy's config always requests `AttestationConveyancePreference::
    /// None` and never supplies a trust-anchor CA list, so upstream `webauthn-rs` itself skips
    /// attestation-statement/chain verification for every credential this server ever creates — only
    /// `authData` (which carries the credential's own public key) matters here.
    pub fn parse_attestation_object(d: &[u8]) -> Option<AuthData> {
        let (obj, _) = decode(d, 0)?;
        let map = obj.as_map()?;
        let auth_data = map
            .iter()
            .find(|(k, _)| k.as_text() == Some("authData"))?
            .1
            .as_bytes()?;
        parse_auth_data(auth_data)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The RS256 branch: a COSE RSA key parses correctly and `CosePublicKey::verify` dispatches to
        /// `rsa_pkcs1_sha256_verify` — the ceremony logic itself is already proven end to end via the
        /// ES256 integration test (`tests/passkey_roundtrip.rs`); this covers the RSA-specific path it   CITATION-EXEMPT: deleted by `d84bda3`, 25 crates → 12
        /// doesn't exercise.
        #[test]
        fn rs256_cose_key_verifies() {
            let (n_be, e, d_be) = crate::crypto::rsa_generate_key(1024);
            let key = CosePublicKey::Rs256 {
                n: n_be.clone(),
                e: e.to_be_bytes().to_vec(),
            };
            assert_eq!(key.cose_alg(), -257);

            let signed_data = b"authData || sha256(clientDataJSON)";
            let hash = sha256(signed_data);
            let sig = crate::crypto::rsa_pkcs1_sha256_sign(&n_be, &d_be, &hash);
            assert!(key.verify(signed_data, &sig));

            let mut bad_sig = sig.clone();
            *bad_sig.last_mut().unwrap() ^= 1;
            assert!(!key.verify(signed_data, &bad_sig));
        }

        #[test]
        fn parses_cose_rsa_key_from_cbor() {
            // {1: 3, 3: -257, -1: h'...' (n), -2: h'...' (e)}
            let n = vec![0xAAu8; 4];
            let e = vec![0x01, 0x00, 0x01];
            let mut bytes = vec![0xa4]; // map(4)
            bytes.extend([0x01, 0x03]); // kty: RSA
            bytes.push(0x03);
            bytes.push(0x39);
            bytes.extend(256u16.to_be_bytes()); // alg: -257 (2-byte NInt: info=25)
            bytes.push(0x20);
            bytes.push(0x44);
            bytes.extend(&n); // -1: bytes(4)
            bytes.push(0x21);
            bytes.push(0x43);
            bytes.extend(&e); // -2: bytes(3)
            let (v, _) = decode(&bytes, 0).unwrap();
            let key = parse_cose_key(&v).unwrap();
            assert_eq!(key, CosePublicKey::Rs256 { n, e });
        }
    }
}
pub mod types {
    //! Two families of types, per the mapped sauthy usage:
    //! - Browser wire types (`CreationChallengeResponse`, `RequestChallengeResponse`,
    //!   `RegisterPublicKeyCredential`, `PublicKeyCredential`) — a real browser's `navigator.credentials
    //!   .create()`/`.get()` produces/consumes this exact JSON shape (camelCase, base64url), so field
    //!   names and casing here are load-bearing, not a free choice.
    //! - Server ceremony/storage types (`PasskeyRegistration`, `PasskeyAuthentication`, `Passkey`) — only
    //!   ever round-trip through this crate's own `serde_json` calls, so only self-consistency matters.
    use crate::webauthn::b64data::{Base64UrlSafeData, CredentialID};
    use crate::webauthn::cose::CosePublicKey;
    use serde::{Deserialize, Serialize};

    // ── Registration (browser wire types) ───────────────────────────────────────────────────────────

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RelyingParty {
        pub id: String,
        pub name: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct User {
        pub id: Base64UrlSafeData,
        pub name: String,
        #[serde(rename = "displayName")]
        pub display_name: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PubKeyCredParams {
        #[serde(rename = "type")]
        pub type_: String,
        pub alg: i64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PublicKeyCredentialDescriptor {
        #[serde(rename = "type")]
        pub type_: String,
        pub id: Base64UrlSafeData,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub transports: Option<Vec<String>>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub enum AuthenticatorAttachment {
        Platform,
        CrossPlatform,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum ResidentKeyRequirement {
        Discouraged,
        Preferred,
        Required,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum UserVerificationPolicy {
        Required,
        Preferred,
        Discouraged,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct AuthenticatorSelectionCriteria {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub authenticator_attachment: Option<AuthenticatorAttachment>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub resident_key: Option<ResidentKeyRequirement>,
        pub require_resident_key: bool,
        pub user_verification: UserVerificationPolicy,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct PublicKeyCredentialCreationOptions {
        pub rp: RelyingParty,
        pub user: User,
        pub challenge: Base64UrlSafeData,
        pub pub_key_cred_params: Vec<PubKeyCredParams>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub timeout: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub exclude_credentials: Option<Vec<PublicKeyCredentialDescriptor>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub authenticator_selection: Option<AuthenticatorSelectionCriteria>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attestation: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extensions: Option<serde_json::Value>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CreationChallengeResponse {
        pub public_key: PublicKeyCredentialCreationOptions,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct AuthenticatorAttestationResponseRaw {
        pub attestation_object: Base64UrlSafeData,
        pub client_data_json: Base64UrlSafeData,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub transports: Option<Vec<String>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RegisterPublicKeyCredential {
        pub id: String,
        #[serde(rename = "rawId")]
        pub raw_id: Base64UrlSafeData,
        pub response: AuthenticatorAttestationResponseRaw,
        #[serde(rename = "type")]
        pub type_: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extensions: Option<serde_json::Value>,
    }

    // ── Authentication (browser wire types) ─────────────────────────────────────────────────────────

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct PublicKeyCredentialRequestOptions {
        pub challenge: Base64UrlSafeData,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub timeout: Option<u32>,
        pub rp_id: String,
        pub allow_credentials: Vec<PublicKeyCredentialDescriptor>,
        pub user_verification: UserVerificationPolicy,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extensions: Option<serde_json::Value>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RequestChallengeResponse {
        pub public_key: PublicKeyCredentialRequestOptions,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct AuthenticatorAssertionResponseRaw {
        pub authenticator_data: Base64UrlSafeData,
        pub client_data_json: Base64UrlSafeData,
        pub signature: Base64UrlSafeData,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user_handle: Option<Base64UrlSafeData>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PublicKeyCredential {
        pub id: String,
        #[serde(rename = "rawId")]
        pub raw_id: Base64UrlSafeData,
        pub response: AuthenticatorAssertionResponseRaw,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub extensions: Option<serde_json::Value>,
        #[serde(rename = "type")]
        pub type_: String,
    }

    // ── The client-data-JSON envelope (not CBOR — plain JSON, base64url-decoded from *ClientDataJSON) ──

    #[derive(Debug, Deserialize)]
    pub struct CollectedClientData {
        #[serde(rename = "type")]
        pub type_: String,
        pub challenge: String, // itself base64url text, compared as a string against the stored challenge
        pub origin: String,
    }

    // ── Ceremony state — self-consistent JSON only, never sent to a browser ────────────────────────────

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PasskeyRegistration {
        pub challenge_b64: String,
        pub rp_id_hash: [u8; 32],
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PasskeyAuthentication {
        pub challenge_b64: String,
        pub rp_id_hash: [u8; 32],
        pub credentials: Vec<(CredentialID, CosePublicKey, u32)>,
    }

    // ── Stored credential + results ─────────────────────────────────────────────────────────────────

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Passkey {
        cred_id: CredentialID,
        public_key: CosePublicKey,
        sign_count: u32,
        user_verified: bool,
        backup_eligible: bool,
        backup_state: bool,
    }

    impl Passkey {
        pub fn new(
            cred_id: Vec<u8>,
            public_key: CosePublicKey,
            sign_count: u32,
            user_verified: bool,
            backup_eligible: bool,
            backup_state: bool,
        ) -> Self {
            Self {
                cred_id: cred_id.into(),
                public_key,
                sign_count,
                user_verified,
                backup_eligible,
                backup_state,
            }
        }
        pub fn cred_id(&self) -> &CredentialID {
            &self.cred_id
        }
        pub fn public_key(&self) -> &CosePublicKey {
            &self.public_key
        }
        pub fn sign_count(&self) -> u32 {
            self.sign_count
        }
        pub fn is_user_verified(&self) -> bool {
            self.user_verified
        }

        /// Apply a fresh `AuthenticationResult` (sign counter bump, backup-state changes) after a
        /// successful assertion — mirrors the real crate's mutate-in-place `update_credential`, which
        /// returns `Some(true/false)` when persistence is needed, `None` when nothing changed.
        pub fn update_credential(&mut self, result: &AuthenticationResult) -> Option<bool> {
            if result.cred_id != self.cred_id.0 {
                return None;
            }
            let changed = result.counter != self.sign_count || result.backup_state != self.backup_state;
            self.sign_count = result.counter;
            self.backup_state = result.backup_state;
            if changed {
                Some(result.needs_update)
            } else {
                None
            }
        }
    }

    /// Mirrors `webauthn-rs`'s `danger-credential-internals`-gated `Credential` — sauthy only ever
    /// reads `.user_verified`, so that's the entire surface here.
    #[derive(Debug, Clone)]
    pub struct Credential {
        pub user_verified: bool,
    }
    impl From<Passkey> for Credential {
        fn from(pk: Passkey) -> Self {
            Self {
                user_verified: pk.user_verified,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct AuthenticationResult {
        pub(crate) cred_id: Vec<u8>,
        pub(crate) counter: u32,
        pub(crate) user_verified: bool,
        pub(crate) backup_eligible: bool,
        pub(crate) backup_state: bool,
        pub(crate) needs_update: bool,
    }
    impl AuthenticationResult {
        pub fn cred_id(&self) -> &[u8] {
            &self.cred_id
        }
        pub fn counter(&self) -> u32 {
            self.counter
        }
        pub fn user_verified(&self) -> bool {
            self.user_verified
        }
        pub fn backup_eligible(&self) -> bool {
            self.backup_eligible
        }
        pub fn backup_state(&self) -> bool {
            self.backup_state
        }
        pub fn needs_update(&self) -> bool {
            self.needs_update
        }
    }

    /// Opaque error — every sauthy call site only ever uses `Debug` (`error!(?err, ...)`) or `Display`
    /// (`err.to_string()` / `format!("{err}")`), never matches a specific variant (confirmed by a
    /// repo-wide grep for `WebauthnError` turning up zero pattern matches).
    #[derive(Debug)]
    pub struct WebauthnError(pub String);
    impl std::fmt::Display for WebauthnError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "webauthn error: {}", self.0)
        }
    }
    impl std::error::Error for WebauthnError {}

    pub type WebauthnResult<T> = Result<T, WebauthnError>;
}
pub mod webauthn {
    //! `Webauthn`/`WebauthnBuilder` and the four ceremony methods sauthy actually calls
    //! (`start`/`finish_passkey_registration`, `start`/`finish_passkey_authentication` — confirmed the
    //! only entry points in use; security-key/attested-passkey/conditional-UI flows are not implemented,
    //! matching what sauthy's own `WebauthnBuilder` chain — just `new().rp_name(...).build()`, no
    //! `allow_subdomains`/`append_allowed_origin`/attestation-CA config — ever exercises).
    use crate::base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use crate::crypto::{rand_bytes, sha256};
    use crate::webauthn::b64data::{Base64UrlSafeData, CredentialID};
    use crate::webauthn::cose;
    use crate::webauthn::types::*;

    /// A minimal origin holder — WebAuthn only ever needs exact-string origin comparison against
    /// `clientDataJSON.origin` (sauthy never calls `.allow_subdomains()`/`.append_allowed_origin()`, so
    /// upstream itself only ever checks a single fixed origin here) — not general URL manipulation, so
    /// this isn't the full `url` crate, just enough to `parse`/store/compare/display one.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Url(String);
    impl Url {
        pub fn parse(s: &str) -> WebauthnResult<Self> {
            if !(s.starts_with("http://") || s.starts_with("https://")) {
                return Err(WebauthnError(format!("not an http(s) URL: {s}")));
            }
            Ok(Self(s.trim_end_matches('/').to_string()))
        }
        pub fn as_str(&self) -> &str {
            &self.0
        }
    }
    impl std::fmt::Display for Url {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    pub struct WebauthnBuilder {
        rp_id: String,
        rp_origin: Url,
        rp_name: String,
    }
    impl WebauthnBuilder {
        pub fn new(rp_id: &str, rp_origin: &Url) -> WebauthnResult<Self> {
            if rp_id.is_empty() {
                return Err(WebauthnError("rp_id must not be empty".into()));
            }
            Ok(Self {
                rp_id: rp_id.to_string(),
                rp_origin: rp_origin.clone(),
                rp_name: rp_id.to_string(),
            })
        }
        pub fn rp_name(mut self, rp_name: &str) -> Self {
            self.rp_name = rp_name.to_string();
            self
        }
        pub fn build(self) -> WebauthnResult<Webauthn> {
            let rp_id_hash = sha256(self.rp_id.as_bytes());
            Ok(Webauthn {
                rp_id: self.rp_id,
                rp_origin: self.rp_origin,
                rp_name: self.rp_name,
                rp_id_hash,
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct Webauthn {
        rp_id: String,
        rp_origin: Url,
        rp_name: String,
        rp_id_hash: [u8; 32],
    }

    /// The only algorithms sauthy's config ever negotiates (`COSEAlgorithm::secure_algs()` upstream) —
    /// ES256 (-7) and RS256 (-257), in that preference order.
    const PUB_KEY_CRED_PARAMS: [(&str, i64); 2] = [("public-key", -7), ("public-key", -257)];
    const DEFAULT_TIMEOUT_MS: u32 = 300_000; // matches `DEFAULT_AUTHENTICATOR_TIMEOUT` = 300s upstream

    impl Webauthn {
        pub fn start_passkey_registration(
            &self,
            user_unique_id: uuid::Uuid,
            user_name: &str,
            user_display_name: &str,
            exclude_credentials: Option<Vec<CredentialID>>,
        ) -> WebauthnResult<(CreationChallengeResponse, PasskeyRegistration)> {
            let challenge = rand_bytes(32);
            let challenge_b64 = URL_SAFE_NO_PAD.encode(&challenge);
            let ccr = CreationChallengeResponse {
                public_key: PublicKeyCredentialCreationOptions {
                    rp: RelyingParty {
                        id: self.rp_id.clone(),
                        name: self.rp_name.clone(),
                    },
                    user: User {
                        id: Base64UrlSafeData(user_unique_id.as_bytes().to_vec()),
                        name: user_name.to_string(),
                        display_name: user_display_name.to_string(),
                    },
                    challenge: Base64UrlSafeData(challenge),
                    pub_key_cred_params: PUB_KEY_CRED_PARAMS
                        .iter()
                        .map(|(t, a)| PubKeyCredParams {
                            type_: t.to_string(),
                            alg: *a,
                        })
                        .collect(),
                    timeout: Some(DEFAULT_TIMEOUT_MS),
                    exclude_credentials: exclude_credentials.map(|ids| {
                        ids.into_iter()
                            .map(|id| PublicKeyCredentialDescriptor {
                                type_: "public-key".to_string(),
                                id,
                                transports: None,
                            })
                            .collect()
                    }),
                    authenticator_selection: None,
                    attestation: Some("none".to_string()),
                    extensions: None,
                },
            };
            Ok((
                ccr,
                PasskeyRegistration {
                    challenge_b64,
                    rp_id_hash: self.rp_id_hash,
                },
            ))
        }

        pub fn finish_passkey_registration(
            &self,
            reg: &RegisterPublicKeyCredential,
            state: &PasskeyRegistration,
        ) -> WebauthnResult<Passkey> {
            let cdj: CollectedClientData = serde_json::from_slice(&reg.response.client_data_json.0)
                .map_err(|e| WebauthnError(format!("malformed clientDataJSON: {e}")))?;
            if cdj.type_ != "webauthn.create" {
                return Err(WebauthnError(format!(
                    "wrong clientData type: {}",
                    cdj.type_
                )));
            }
            if cdj.challenge != state.challenge_b64 {
                return Err(WebauthnError("registration challenge mismatch".into()));
            }
            if cdj.origin != self.rp_origin.as_str() {
                return Err(WebauthnError(format!("origin mismatch: {}", cdj.origin)));
            }

            let auth_data = cose::parse_attestation_object(&reg.response.attestation_object.0)
                .ok_or_else(|| WebauthnError("malformed attestationObject".into()))?;
            if auth_data.rp_id_hash != state.rp_id_hash {
                return Err(WebauthnError("rpIdHash mismatch".into()));
            }
            let attested = auth_data.attested.ok_or_else(|| {
                WebauthnError("attestationObject has no attested credential data".into())
            })?;

            Ok(Passkey::new(
                attested.credential_id,
                attested.public_key,
                auth_data.sign_count,
                auth_data.user_verified,
                auth_data.backup_eligible,
                auth_data.backup_state,
            ))
        }

        pub fn start_passkey_authentication(
            &self,
            creds: &[Passkey],
        ) -> WebauthnResult<(RequestChallengeResponse, PasskeyAuthentication)> {
            if creds.is_empty() {
                return Err(WebauthnError(
                    "no credentials to authenticate against".into(),
                ));
            }
            let challenge = rand_bytes(32);
            let challenge_b64 = URL_SAFE_NO_PAD.encode(&challenge);
            let rcr = RequestChallengeResponse {
                public_key: PublicKeyCredentialRequestOptions {
                    challenge: Base64UrlSafeData(challenge),
                    timeout: Some(DEFAULT_TIMEOUT_MS),
                    rp_id: self.rp_id.clone(),
                    allow_credentials: creds
                        .iter()
                        .map(|pk| PublicKeyCredentialDescriptor {
                            type_: "public-key".to_string(),
                            id: pk.cred_id().clone(),
                            transports: None,
                        })
                        .collect(),
                    user_verification: UserVerificationPolicy::Preferred,
                    extensions: None,
                },
            };
            let credentials = creds
                .iter()
                .map(|pk| {
                    (
                        pk.cred_id().clone(),
                        pk.public_key().clone(),
                        pk.sign_count(),
                    )
                })
                .collect();
            Ok((
                rcr,
                PasskeyAuthentication {
                    challenge_b64,
                    rp_id_hash: self.rp_id_hash,
                    credentials,
                },
            ))
        }

        pub fn finish_passkey_authentication(
            &self,
            reg: &PublicKeyCredential,
            state: &PasskeyAuthentication,
        ) -> WebauthnResult<AuthenticationResult> {
            let (cred_id, pubkey, old_counter) = state
                .credentials
                .iter()
                .find(|(id, _, _)| id.0 == reg.raw_id.0)
                .ok_or_else(|| WebauthnError("credential not found among the allowed set".into()))?;

            let cdj: CollectedClientData = serde_json::from_slice(&reg.response.client_data_json.0)
                .map_err(|e| WebauthnError(format!("malformed clientDataJSON: {e}")))?;
            if cdj.type_ != "webauthn.get" {
                return Err(WebauthnError(format!(
                    "wrong clientData type: {}",
                    cdj.type_
                )));
            }
            if cdj.challenge != state.challenge_b64 {
                return Err(WebauthnError("authentication challenge mismatch".into()));
            }
            if cdj.origin != self.rp_origin.as_str() {
                return Err(WebauthnError(format!("origin mismatch: {}", cdj.origin)));
            }

            let auth_data_bytes = &reg.response.authenticator_data.0;
            let auth_data = cose::parse_auth_data(auth_data_bytes)
                .ok_or_else(|| WebauthnError("malformed authenticatorData".into()))?;
            if auth_data.rp_id_hash != state.rp_id_hash {
                return Err(WebauthnError("rpIdHash mismatch".into()));
            }

            let client_data_hash = sha256(&reg.response.client_data_json.0);
            let mut signed_data = auth_data_bytes.clone();
            signed_data.extend_from_slice(&client_data_hash);
            if !pubkey.verify(&signed_data, &reg.response.signature.0) {
                return Err(WebauthnError(
                    "assertion signature verification failed".into(),
                ));
            }

            // Anti-replay (WebAuthn L2 §7.2 step 21): the sign counter must strictly increase, unless
            // the authenticator doesn't implement one (both stored and fresh counters are 0 — allowed).
            if !(auth_data.sign_count == 0 && *old_counter == 0) && auth_data.sign_count <= *old_counter
            {
                return Err(WebauthnError(
                    "sign counter did not increase — possible cloned authenticator".into(),
                ));
            }

            Ok(AuthenticationResult {
                cred_id: cred_id.0.clone(),
                counter: auth_data.sign_count,
                user_verified: auth_data.user_verified,
                backup_eligible: auth_data.backup_eligible,
                backup_state: auth_data.backup_state,
                needs_update: auth_data.sign_count != *old_counter,
            })
        }
    }
}

pub use cose::CosePublicKey;
pub use uuid::Uuid;
pub use webauthn::{Url, Webauthn, WebauthnBuilder};

// `webauthn_rs_proto::{AuthenticatorSelectionCriteria, ResidentKeyRequirement,
// UserVerificationPolicy}` is imported directly (not via `prelude`) at one sauthy call site —
// re-exported at the crate root to match.
pub use types::{
    AuthenticatorAttachment, AuthenticatorSelectionCriteria, ResidentKeyRequirement,
    UserVerificationPolicy,
};

pub mod prelude {
    pub use crate::webauthn::b64data::{Base64UrlSafeData, CredentialID};
    pub use crate::webauthn::types::{
        AuthenticationResult, AuthenticatorAssertionResponseRaw,
        AuthenticatorAttestationResponseRaw, CreationChallengeResponse, Credential, Passkey,
        PasskeyAuthentication, PasskeyRegistration, PublicKeyCredential,
        PublicKeyCredentialCreationOptions, PublicKeyCredentialDescriptor,
        PublicKeyCredentialRequestOptions, RegisterPublicKeyCredential, RelyingParty,
        RequestChallengeResponse, User, WebauthnError, WebauthnResult,
    };
    pub use crate::webauthn::webauthn::{Url, Webauthn, WebauthnBuilder};
    pub use crate::webauthn::{
        AuthenticatorSelectionCriteria, ResidentKeyRequirement, UserVerificationPolicy,
    };
    pub use uuid::Uuid;
}
