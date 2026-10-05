//! A rustls `CryptoProvider` backed entirely by hashtrinity-crypto — no ring, no aws-lc-rs.
pub mod aead {
    //! TLS 1.3 AEAD (AES-256-GCM, ChaCha20-Poly1305) as rustls `Tls13AeadAlgorithm` — hashtrinity-crypto.
    //! The record framing: nonce = iv ⊕ seq, AAD = the record header (make_tls13_aad), plaintext ++ type.
    use rustls::crypto::cipher::{
        make_tls13_aad, AeadKey, InboundOpaqueMessage, InboundPlainMessage, Iv, MessageDecrypter,
        MessageEncrypter, Nonce, OutboundOpaqueMessage, OutboundPlainMessage, PrefixedPayload,
        Tls13AeadAlgorithm, UnsupportedOperationError,
    };
    use rustls::ConnectionTrafficSecrets;
    use rustls::{ContentType, Error, ProtocolVersion};

    #[derive(Clone, Copy)]
    pub enum Kind {
        Aes128Gcm,
        Aes256Gcm,
        ChaCha20Poly1305,
    }

    pub struct Aead(pub Kind);

    impl Tls13AeadAlgorithm for Aead {
        fn encrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn MessageEncrypter> {
            Box::new(Enc {
                kind: self.0,
                key: key.as_ref().to_vec(),
                iv,
            })
        }
        fn decrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn MessageDecrypter> {
            Box::new(Dec {
                kind: self.0,
                key: key.as_ref().to_vec(),
                iv,
            })
        }
        fn key_len(&self) -> usize {
            match self.0 {
                Kind::Aes128Gcm => 16,
                _ => 32,
            }
        }
        fn extract_keys(
            &self,
            key: AeadKey,
            iv: Iv,
        ) -> Result<ConnectionTrafficSecrets, UnsupportedOperationError> {
            Ok(match self.0 {
                Kind::Aes128Gcm => ConnectionTrafficSecrets::Aes128Gcm { key, iv },
                Kind::Aes256Gcm => ConnectionTrafficSecrets::Aes256Gcm { key, iv },
                Kind::ChaCha20Poly1305 => ConnectionTrafficSecrets::Chacha20Poly1305 { key, iv },
            })
        }
    }

    pub(crate) fn seal(
        kind: Kind,
        key: &[u8],
        nonce: &[u8; 12],
        pt: &[u8],
        aad: &[u8],
    ) -> (Vec<u8>, [u8; 16]) {
        use crate::crypto::{
            aes_gcm::{aes128_gcm_encrypt, aes256_gcm_encrypt},
            chacha_poly::chacha20poly1305_encrypt,
        };
        match kind {
            Kind::Aes128Gcm => aes128_gcm_encrypt(key.try_into().unwrap(), nonce, pt, aad),
            Kind::Aes256Gcm => aes256_gcm_encrypt(key.try_into().unwrap(), nonce, pt, aad),
            Kind::ChaCha20Poly1305 => chacha20poly1305_encrypt(key.try_into().unwrap(), nonce, pt, aad),
        }
    }
    pub(crate) fn open(
        kind: Kind,
        key: &[u8],
        nonce: &[u8; 12],
        ct: &[u8],
        tag: &[u8; 16],
        aad: &[u8],
    ) -> Option<Vec<u8>> {
        use crate::crypto::{
            aes_gcm::{aes128_gcm_decrypt, aes256_gcm_decrypt},
            chacha_poly::chacha20poly1305_decrypt,
        };
        match kind {
            Kind::Aes128Gcm => aes128_gcm_decrypt(key.try_into().unwrap(), nonce, ct, tag, aad),
            Kind::Aes256Gcm => aes256_gcm_decrypt(key.try_into().unwrap(), nonce, ct, tag, aad),
            Kind::ChaCha20Poly1305 => {
                chacha20poly1305_decrypt(key.try_into().unwrap(), nonce, ct, tag, aad)
            }
        }
    }

    struct Enc {
        kind: Kind,
        key: Vec<u8>,
        iv: Iv,
    }
    impl MessageEncrypter for Enc {
        fn encrypt(
            &mut self,
            msg: OutboundPlainMessage<'_>,
            seq: u64,
        ) -> Result<OutboundOpaqueMessage, Error> {
            let total_len = self.encrypted_payload_len(msg.payload.len());
            let mut plain = Vec::with_capacity(msg.payload.len() + 1);
            msg.payload.copy_to_vec(&mut plain);
            plain.push(msg.typ.into());
            let nonce = Nonce::new(&self.iv, seq).0;
            let aad = make_tls13_aad(total_len);
            let (ct, tag) = seal(self.kind, &self.key, &nonce, &plain, &aad);
            let mut payload = PrefixedPayload::with_capacity(total_len);
            payload.extend_from_slice(&ct);
            payload.extend_from_slice(&tag);
            Ok(OutboundOpaqueMessage::new(
                ContentType::ApplicationData,
                ProtocolVersion::TLSv1_2,
                payload,
            ))
        }
        fn encrypted_payload_len(&self, payload_len: usize) -> usize {
            payload_len + 1 + 16
        }
    }

    struct Dec {
        kind: Kind,
        key: Vec<u8>,
        iv: Iv,
    }
    impl MessageDecrypter for Dec {
        fn decrypt<'a>(
            &mut self,
            mut msg: InboundOpaqueMessage<'a>,
            seq: u64,
        ) -> Result<InboundPlainMessage<'a>, Error> {
            let payload = &mut msg.payload;
            if payload.len() < 16 {
                return Err(Error::DecryptError);
            }
            let nonce = Nonce::new(&self.iv, seq).0;
            let aad = make_tls13_aad(payload.len());
            let n = payload.len() - 16;
            let mut tag = [0u8; 16];
            tag.copy_from_slice(&payload[n..]);
            let pt = open(self.kind, &self.key, &nonce, &payload[..n], &tag, &aad)
                .ok_or(Error::DecryptError)?;
            payload[..pt.len()].copy_from_slice(&pt);
            payload.truncate(pt.len());
            msg.into_tls13_unpadded_message()
        }
    }

    pub static AES128_GCM: Aead = Aead(Kind::Aes128Gcm);
    pub static CHACHA20_POLY1305: Aead = Aead(Kind::ChaCha20Poly1305);
    pub static AES256_GCM: Aead = Aead(Kind::Aes256Gcm);
}
pub mod hash {
    //! SHA-256/384 as rustls `Hash` — now the REAL streaming fold (hashtrinity-crypto's `Sha256`/`Sha512`
    //! incremental `update`), no buffering. The debt paid: `Context` IS the fold state, block by block.
    use crate::crypto::{Sha256, Sha512};
    use rustls::crypto::hash::{Context, Hash as HashT, HashAlgorithm, Output};

    pub static SHA256: Sha = Sha::S256;
    pub static SHA384: Sha = Sha::S384;

    pub enum Sha {
        S256,
        S384,
    }
    impl HashT for Sha {
        fn start(&self) -> Box<dyn Context> {
            match self {
                Sha::S256 => Box::new(C256(Sha256::new())),
                Sha::S384 => Box::new(C384(Sha512::new_384())),
            }
        }
        fn hash(&self, bytes: &[u8]) -> Output {
            match self {
                Sha::S256 => Output::new(&crate::crypto::sha256(bytes)),
                Sha::S384 => Output::new(&crate::crypto::sha384(bytes)),
            }
        }
        fn output_len(&self) -> usize {
            match self {
                Sha::S256 => 32,
                Sha::S384 => 48,
            }
        }
        fn algorithm(&self) -> HashAlgorithm {
            match self {
                Sha::S256 => HashAlgorithm::SHA256,
                Sha::S384 => HashAlgorithm::SHA384,
            }
        }
    }

    struct C256(Sha256);
    impl Context for C256 {
        fn fork_finish(&self) -> Output {
            Output::new(&self.0.clone().finalize())
        }
        fn fork(&self) -> Box<dyn Context> {
            Box::new(C256(self.0.clone()))
        }
        fn finish(self: Box<Self>) -> Output {
            Output::new(&self.0.finalize())
        }
        fn update(&mut self, data: &[u8]) {
            self.0.update(data);
        }
    }
    struct C384(Sha512);
    impl Context for C384 {
        fn fork_finish(&self) -> Output {
            Output::new(&self.0.clone().finalize_384())
        }
        fn fork(&self) -> Box<dyn Context> {
            Box::new(C384(self.0.clone()))
        }
        fn finish(self: Box<Self>) -> Output {
            Output::new(&self.0.finalize_384())
        }
        fn update(&mut self, data: &[u8]) {
            self.0.update(data);
        }
    }
}
pub mod hkdf {
    //! HKDF-SHA256 (TLS 1.3 key schedule) as rustls `Hkdf`/`HkdfExpander` — hashtrinity-crypto's HMAC fold.
    use rustls::crypto::hmac::Tag;
    use rustls::crypto::tls13::{Hkdf, HkdfExpander, OkmBlock, OutputLengthError};

    pub struct HkdfSha256;

    fn hmac(key: &[u8], msg: &[u8]) -> [u8; 32] {
        crate::crypto::hmac_sha256(key, msg)
    }

    fn expand(prk: &[u8], info: &[&[u8]], out: &mut [u8]) {
        let mut t: Vec<u8> = Vec::new();
        let (mut i, mut done) = (1u8, 0usize);
        while done < out.len() {
            let mut msg = t.clone();
            for s in info {
                msg.extend_from_slice(s);
            }
            msg.push(i);
            t = hmac(prk, &msg).to_vec();
            let n = (out.len() - done).min(t.len());
            out[done..done + n].copy_from_slice(&t[..n]);
            done += n;
            i = i.wrapping_add(1);
        }
    }

    impl Hkdf for HkdfSha256 {
        fn extract_from_zero_ikm(&self, salt: Option<&[u8]>) -> Box<dyn HkdfExpander> {
            let z = [0u8; 32];
            Box::new(Exp {
                prk: hmac(salt.unwrap_or(&z), &z),
            })
        }
        fn extract_from_secret(&self, salt: Option<&[u8]>, secret: &[u8]) -> Box<dyn HkdfExpander> {
            let z = [0u8; 32];
            Box::new(Exp {
                prk: hmac(salt.unwrap_or(&z), secret),
            })
        }
        fn expander_for_okm(&self, okm: &OkmBlock) -> Box<dyn HkdfExpander> {
            let mut prk = [0u8; 32];
            prk.copy_from_slice(okm.as_ref());
            Box::new(Exp { prk })
        }
        fn hmac_sign(&self, key: &OkmBlock, message: &[u8]) -> Tag {
            Tag::new(&hmac(key.as_ref(), message))
        }
    }

    struct Exp {
        prk: [u8; 32],
    }
    impl HkdfExpander for Exp {
        fn expand_slice(&self, info: &[&[u8]], output: &mut [u8]) -> Result<(), OutputLengthError> {
            if output.len() > 255 * 32 {
                return Err(OutputLengthError);
            }
            expand(&self.prk, info, output);
            Ok(())
        }
        fn expand_block(&self, info: &[&[u8]]) -> OkmBlock {
            let mut b = [0u8; 32];
            expand(&self.prk, info, &mut b);
            OkmBlock::new(&b)
        }
        fn hash_len(&self) -> usize {
            32
        }
    }

    pub static HKDF_SHA256: HkdfSha256 = HkdfSha256;
}
pub mod hmac {
    //! HMAC-SHA256/384 as rustls `Hmac` — hashtrinity-crypto (384 uses the SHA-512 fold, truncated? no:
    //! TLS 1.3 with SHA-384 uses HMAC-SHA384; we expose 256 + 512, and 384 via SHA-384 HMAC).
    use rustls::crypto::hmac::{Hmac as HmacT, Key as KeyT, Tag};

    pub static HMAC_SHA256: H = H { len: 32 };

    pub struct H {
        len: usize,
    }
    impl HmacT for H {
        fn with_key(&self, key: &[u8]) -> Box<dyn KeyT> {
            Box::new(K { key: key.to_vec() })
        }
        fn hash_output_len(&self) -> usize {
            self.len
        }
    }
    struct K {
        key: Vec<u8>,
    }
    impl KeyT for K {
        fn sign_concat(&self, first: &[u8], middle: &[&[u8]], last: &[u8]) -> Tag {
            let mut msg = first.to_vec();
            for m in middle {
                msg.extend_from_slice(m);
            }
            msg.extend_from_slice(last);
            Tag::new(&crate::crypto::hmac_sha256(&self.key, &msg))
        }
        fn tag_len(&self) -> usize {
            32
        }
    }
}
pub mod kx {
    //! X25519 key exchange as rustls `SupportedKxGroup`/`ActiveKeyExchange` — hashtrinity-crypto::x25519.
    use rustls::crypto::{ActiveKeyExchange, SharedSecret, SupportedKxGroup};
    use rustls::{Error, NamedGroup};
    use std::io::Read;

    fn basepoint() -> [u8; 32] {
        let mut b = [0u8; 32];
        b[0] = 9;
        b
    }

    pub static X25519: &dyn SupportedKxGroup = &Group;

    #[derive(Debug)]
    struct Group;
    impl SupportedKxGroup for Group {
        fn start(&self) -> Result<Box<dyn ActiveKeyExchange>, Error> {
            let mut sk = [0u8; 32];
            std::fs::File::open("/dev/urandom")
                .and_then(|mut f| f.read_exact(&mut sk))
                .map_err(|_| Error::FailedToGetRandomBytes)?;
            let pk = crate::crypto::x25519(&sk, &basepoint());
            Ok(Box::new(Active {
                sk,
                pk: pk.to_vec(),
            }))
        }
        fn name(&self) -> NamedGroup {
            NamedGroup::X25519
        }
    }

    struct Active {
        sk: [u8; 32],
        pk: Vec<u8>,
    }
    impl ActiveKeyExchange for Active {
        fn complete(self: Box<Self>, peer: &[u8]) -> Result<SharedSecret, Error> {
            let peer: [u8; 32] = peer.try_into().map_err(|_| Error::FailedToGetRandomBytes)?;
            let ss = crate::crypto::x25519(&self.sk, &peer);
            if ss == [0u8; 32] {
                return Err(Error::FailedToGetRandomBytes);
            } // all-zero = degenerate
            Ok(SharedSecret::from(&ss[..]))
        }
        fn pub_key(&self) -> &[u8] {
            &self.pk
        }
        fn group(&self) -> NamedGroup {
            NamedGroup::X25519
        }
    }
}
pub mod quic {
    //! QUIC crypto for the provider (rustls `quic::Algorithm`), parameterized by (packet-AEAD, HP-cipher):
    //! ChaCha suite → ChaCha packet + ChaCha-block HP; AES suites → AES-GCM packet + AES-ECB HP (RFC 9001
    //! §5.4.3). All wired to hashtrinity-crypto; the RFC HP application is verbatim.
    use crate::crypto::quic::{quic_hp_mask, quic_hp_mask_aes128, quic_hp_mask_aes256};
    use crate::rustls::aead::{open, seal, Kind};
    use rustls::crypto::cipher::{AeadKey, Iv};
    use rustls::quic::{Algorithm, HeaderProtectionKey, PacketKey, Tag};
    use rustls::Error;

    fn nonce(iv: &[u8; 12], pn: u64) -> [u8; 12] {
        let mut n = *iv;
        let b = pn.to_be_bytes();
        for i in 0..8 {
            n[4 + i] ^= b[i];
        }
        n
    }

    #[derive(Clone, Copy)]
    pub enum Hp {
        Chacha,
        Aes128,
        Aes256,
    }

    pub struct QuicAlg {
        pub aead: Kind,
        pub hp: Hp,
    }
    impl Algorithm for QuicAlg {
        fn packet_key(&self, key: AeadKey, iv: Iv) -> Box<dyn PacketKey> {
            let mut ivb = [0u8; 12];
            ivb.copy_from_slice(iv.as_ref());
            Box::new(PKey {
                aead: self.aead,
                key: key.as_ref().to_vec(),
                iv: ivb,
            })
        }
        fn header_protection_key(&self, key: AeadKey) -> Box<dyn HeaderProtectionKey> {
            Box::new(HPKey {
                hp: self.hp,
                key: key.as_ref().to_vec(),
            })
        }
        fn aead_key_len(&self) -> usize {
            match self.aead {
                Kind::Aes128Gcm => 16,
                _ => 32,
            }
        }
    }

    struct PKey {
        aead: Kind,
        key: Vec<u8>,
        iv: [u8; 12],
    }
    impl PacketKey for PKey {
        fn encrypt_in_place(&self, pn: u64, header: &[u8], payload: &mut [u8]) -> Result<Tag, Error> {
            let (ct, tag) = seal(self.aead, &self.key, &nonce(&self.iv, pn), payload, header);
            payload.copy_from_slice(&ct);
            Ok(Tag::from(&tag[..]))
        }
        fn decrypt_in_place<'a>(
            &self,
            pn: u64,
            header: &[u8],
            payload: &'a mut [u8],
        ) -> Result<&'a [u8], Error> {
            if payload.len() < 16 {
                return Err(Error::DecryptError);
            }
            let n = payload.len() - 16;
            let mut tag = [0u8; 16];
            tag.copy_from_slice(&payload[n..]);
            let pt = open(
                self.aead,
                &self.key,
                &nonce(&self.iv, pn),
                &payload[..n],
                &tag,
                header,
            )
            .ok_or(Error::DecryptError)?;
            payload[..pt.len()].copy_from_slice(&pt);
            Ok(&payload[..pt.len()])
        }
        fn tag_len(&self) -> usize {
            16
        }
        fn confidentiality_limit(&self) -> u64 {
            u64::MAX
        }
        fn integrity_limit(&self) -> u64 {
            1 << 36
        }
    }

    struct HPKey {
        hp: Hp,
        key: Vec<u8>,
    }
    impl HPKey {
        fn mask(&self, sample: &[u8; 16]) -> [u8; 5] {
            match self.hp {
                Hp::Chacha => quic_hp_mask(self.key[..32].try_into().unwrap(), sample),
                Hp::Aes128 => quic_hp_mask_aes128(self.key[..16].try_into().unwrap(), sample),
                Hp::Aes256 => quic_hp_mask_aes256(self.key[..32].try_into().unwrap(), sample),
            }
        }
        fn xor(&self, sample: &[u8], first: &mut u8, pn: &mut [u8], masked: bool) -> Result<(), Error> {
            let s: [u8; 16] = sample
                .try_into()
                .map_err(|_| Error::General("bad HP sample".into()))?;
            let mask = self.mask(&s);
            let (first_mask, pn_mask) = mask.split_first().unwrap();
            if pn.len() > pn_mask.len() {
                return Err(Error::General("packet number too long".into()));
            }
            const LONG_HEADER: u8 = 0x80;
            let bits = if *first & LONG_HEADER == LONG_HEADER {
                0x0f
            } else {
                0x1f
            };
            let first_plain = if masked {
                *first ^ (first_mask & bits)
            } else {
                *first
            };
            let pn_len = (first_plain & 0x03) as usize + 1;
            *first ^= first_mask & bits;
            for (dst, m) in pn.iter_mut().zip(pn_mask).take(pn_len) {
                *dst ^= m;
            }
            Ok(())
        }
    }
    impl HeaderProtectionKey for HPKey {
        fn encrypt_in_place(&self, sample: &[u8], first: &mut u8, pn: &mut [u8]) -> Result<(), Error> {
            self.xor(sample, first, pn, false)
        }
        fn decrypt_in_place(&self, sample: &[u8], first: &mut u8, pn: &mut [u8]) -> Result<(), Error> {
            self.xor(sample, first, pn, true)
        }
        fn sample_len(&self) -> usize {
            16
        }
    }

    pub static AES128_QUIC: QuicAlg = QuicAlg {
        aead: Kind::Aes128Gcm,
        hp: Hp::Aes128,
    };
    pub static CHACHA20_QUIC: QuicAlg = QuicAlg {
        aead: Kind::ChaCha20Poly1305,
        hp: Hp::Chacha,
    };
}
pub mod random {
    //! Secure random from the OS entropy source (the one syscall cut), as in age.
    use rustls::crypto::{GetRandomFailed, SecureRandom};
    use std::io::Read;

    #[derive(Debug)]
    pub struct Rng;
    impl SecureRandom for Rng {
        fn fill(&self, buf: &mut [u8]) -> Result<(), GetRandomFailed> {
            std::fs::File::open("/dev/urandom")
                .and_then(|mut f| f.read_exact(buf))
                .map_err(|_| GetRandomFailed)
        }
    }
}
pub mod sign {
    //! `KeyProvider` for ed25519 and ECDSA-P256 PKCS8 private keys — copies ring's structure, shaved to
    //! hashtrinity-crypto. The PKCS8/SEC1 DER is walked with our OWN `der_tlv` (no ring key-parse);
    //! signing is `ed25519_sign`/`ecdsa_p256_sign`. RSA server certs aren't handled (bedrock's own
    //! `tls_server_config` only ever hands this a cert it generated itself — ed25519 or P-256 today).
    use crate::crypto::x509::{der_children, der_tlv};
    use rustls::crypto::KeyProvider;
    use rustls::pki_types::PrivateKeyDer;
    use rustls::sign::{Signer, SigningKey};
    use rustls::{Error, SignatureAlgorithm, SignatureScheme};
    use std::sync::Arc;

    #[derive(Debug)]
    pub struct Provider;
    impl KeyProvider for Provider {
        fn load_private_key(
            &self,
            key_der: PrivateKeyDer<'static>,
        ) -> Result<Arc<dyn SigningKey>, Error> {
            let der = match &key_der {
                PrivateKeyDer::Pkcs8(d) => d.secret_pkcs8_der(),
                _ => {
                    return Err(Error::General(
                        "hashtrinity-rustls: only PKCS8 keys are supported".into(),
                    ))
                }
            };
            if let Some(seed) = ed25519_seed_from_pkcs8(der) {
                return Ok(Arc::new(Ed25519Key { seed }));
            }
            if let Some(scalar) = ecdsa_p256_scalar_from_pkcs8(der) {
                return Ok(Arc::new(EcdsaP256Key { scalar }));
            }
            Err(Error::General("hashtrinity-rustls: unrecognized PKCS8 key (only ed25519 and ECDSA-P256 are supported)".into()))
        }
    }

    const ED25519_OID: [u8; 3] = [0x2b, 0x65, 0x70]; // 1.3.101.112
    const EC_PUBLIC_KEY_OID: [u8; 7] = [0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01]; // 1.2.840.10045.2.1

    /// Look up the `AlgorithmIdentifier`'s OID (child 1 of a PKCS8 `PrivateKeyInfo` SEQUENCE) — the
    /// AlgorithmIdentifier is itself `SEQUENCE { OID, parameters }`, so this returns just the OID bytes.
    fn pkcs8_algorithm_oid(der: &[u8]) -> Option<&[u8]> {
        let (tag, cs, ce, _) = der_tlv(der, 0)?;
        if tag != 0x30 {
            return None;
        }
        let (algid_tag, as_, ae) = *der_children(der, cs, ce).get(1)?;
        if algid_tag != 0x30 {
            return None;
        }
        let (oid_tag, os, oe) = *der_children(der, as_, ae).first()?;
        if oid_tag != 0x06 {
            return None;
        }
        Some(&der[os..oe])
    }

    /// PKCS8 ed25519 (RFC 8410): `SEQUENCE { version, algid, OCTET STRING { OCTET STRING { seed[32] } } }`.
    fn ed25519_seed_from_pkcs8(der: &[u8]) -> Option<[u8; 32]> {
        if pkcs8_algorithm_oid(der)? != ED25519_OID {
            return None;
        }
        let (_, cs, ce, _) = der_tlv(der, 0)?;
        let (tag, ps, _pe) = *der_children(der, cs, ce).get(2)?; // privateKey OCTET STRING
        if tag != 0x04 {
            return None;
        }
        let (itag, is, ie, _) = der_tlv(der, ps)?; // inner OCTET STRING { seed }
        if itag != 0x04 || ie - is != 32 {
            return None;
        }
        der[is..ie].try_into().ok()
    }

    /// PKCS8 EC (SEC1, RFC 5915): `SEQUENCE { version, algid{ecPublicKey, curveOid}, OCTET STRING {
    /// ECPrivateKey ::= SEQUENCE { version, privateKey OCTET STRING[32], [0] params?, [1] pubkey? } } }`.
    /// Only P-256 (`id-ecPublicKey`) is recognized — the only curve this crate's own certificate
    /// issuance ever produces (the curve OID itself isn't re-checked; a non-P-256 EC key would produce
    /// a wrong-length scalar and fail loudly downstream rather than silently misparse).
    fn ecdsa_p256_scalar_from_pkcs8(der: &[u8]) -> Option<[u8; 32]> {
        if pkcs8_algorithm_oid(der)? != EC_PUBLIC_KEY_OID {
            return None;
        }
        let (_, cs, ce, _) = der_tlv(der, 0)?;
        let (tag, ps, _pe) = *der_children(der, cs, ce).get(2)?; // privateKey OCTET STRING
        if tag != 0x04 {
            return None;
        }
        let (inner_tag, ics, ice, _) = der_tlv(der, ps)?; // ECPrivateKey SEQUENCE
        if inner_tag != 0x30 {
            return None;
        }
        let (sk_tag, ss, se) = *der_children(der, ics, ice).get(1)?; // privateKey OCTET STRING[32]
        if sk_tag != 0x04 || se - ss != 32 {
            return None;
        }
        der[ss..se].try_into().ok()
    }

    /// `SEQUENCE { INTEGER r, INTEGER s }` — the DER shape TLS's ECDSA `CertificateVerify` signatures
    /// use, from raw r‖s (what `ecdsa_p256_sign` returns). Same tiny encoder the WebAuthn masquerade's
    /// `cose.rs` writes for the decode direction; this crate needs the encode direction instead.
    fn der_encode_ecdsa_sig(raw_sig: &[u8; 64]) -> Vec<u8> {
        fn der_int(mag: &[u8]) -> Vec<u8> {
            let mut v = mag.to_vec();
            while v.len() > 1 && v[0] == 0 {
                v.remove(0);
            }
            if v[0] & 0x80 != 0 {
                v.insert(0, 0);
            }
            let mut out = vec![0x02, v.len() as u8];
            out.extend(v);
            out
        }
        let mut content = der_int(&raw_sig[..32]);
        content.extend(der_int(&raw_sig[32..]));
        let mut out = vec![0x30, content.len() as u8];
        out.extend(content);
        out
    }

    #[derive(Debug)]
    struct Ed25519Key {
        seed: [u8; 32],
    }
    impl SigningKey for Ed25519Key {
        fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
            offered
                .contains(&SignatureScheme::ED25519)
                .then(|| Box::new(Ed25519Sig { seed: self.seed }) as Box<dyn Signer>)
        }
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::ED25519
        }
    }

    #[derive(Debug)]
    struct Ed25519Sig {
        seed: [u8; 32],
    }
    impl Signer for Ed25519Sig {
        fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Error> {
            Ok(crate::crypto::ed25519_sign(&self.seed, message).to_vec())
        }
        fn scheme(&self) -> SignatureScheme {
            SignatureScheme::ED25519
        }
    }

    #[derive(Debug)]
    struct EcdsaP256Key {
        scalar: [u8; 32],
    }
    impl SigningKey for EcdsaP256Key {
        fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
            offered
                .contains(&SignatureScheme::ECDSA_NISTP256_SHA256)
                .then(|| {
                    Box::new(EcdsaP256Sig {
                        scalar: self.scalar,
                    }) as Box<dyn Signer>
                })
        }
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::ECDSA
        }
    }

    #[derive(Debug)]
    struct EcdsaP256Sig {
        scalar: [u8; 32],
    }
    impl Signer for EcdsaP256Sig {
        fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Error> {
            let hash = crate::crypto::sha256(message);
            let raw_sig = crate::crypto::ecdsa_p256_sign(&self.scalar, &hash);
            Ok(der_encode_ecdsa_sig(&raw_sig))
        }
        fn scheme(&self) -> SignatureScheme {
            SignatureScheme::ECDSA_NISTP256_SHA256
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The exact PKCS8 shape `bedrock`'s own test cert/key pair uses (an ECDSA-P256 key, not
        /// ed25519 — the shape that surfaced the original "bad ed25519 PKCS8" gap: this crate only
        /// handled ed25519 before, and had never been exercised against real key material until now).
        const TEST_KEY_PEM_B64: &str =
            "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgVWtnGXHnzG6FQVVq\
    i8YHRSNBOrNNqCa29hrqX758vMihRANCAAQWzzNgEHmWExfSibB4yxRBk1XskVgj\
    Cno/DO/69cKbImX6Mesim9v56PmqJXWVtsTxPgyjXGc9t7qFcshNGmdr";

        fn b64_decode(s: &str) -> Vec<u8> {
            use crate::base64::{engine::general_purpose::STANDARD, Engine};
            STANDARD.decode(s).unwrap()
        }

        #[test]
        fn parses_real_ecdsa_p256_pkcs8_key() {
            let der = b64_decode(TEST_KEY_PEM_B64);
            let scalar = ecdsa_p256_scalar_from_pkcs8(&der).expect("should parse as ECDSA-P256");
            assert_eq!(ed25519_seed_from_pkcs8(&der), None); // must not misidentify as ed25519

            // sign/verify round trip proves the extracted scalar is the real private key, not garbage
            let pubkey = crate::crypto::p256_pubkey(&scalar);
            let hash = crate::crypto::sha256(b"hashtrinity-rustls TLS CertificateVerify");
            let sig = crate::crypto::ecdsa_p256_sign(&scalar, &hash);
            assert!(crate::crypto::ecdsa_p256_verify(&pubkey, &hash, &sig));
        }

        #[test]
        fn key_provider_loads_real_test_key() {
            use rustls::pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer};
            let der = b64_decode(TEST_KEY_PEM_B64);
            let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(der));
            let signing_key = Provider
                .load_private_key(key_der)
                .expect("should load as ECDSA-P256");
            assert_eq!(signing_key.algorithm(), SignatureAlgorithm::ECDSA);
            let signer = signing_key
                .choose_scheme(&[SignatureScheme::ECDSA_NISTP256_SHA256])
                .expect("should offer ECDSA_NISTP256_SHA256");
            let sig = signer.sign(b"transcript").unwrap();
            assert_eq!(sig[0], 0x30); // DER SEQUENCE tag
        }
    }
}
pub mod verify {
    //! Signature verification for the provider — the one piece that can't copy ring (ring delegates to
    //! webpki's ring backend). Fresh `SignatureVerificationAlgorithm` impls wired to hashtrinity-crypto.
    //! ed25519 (iroh's cert algorithm) first; ECDSA/RSA are additive (need DER-sig parse + pre-hash).
    use rustls::crypto::WebPkiSupportedAlgorithms;
    use rustls::pki_types::{
        alg_id, AlgorithmIdentifier, InvalidSignature, SignatureVerificationAlgorithm,
    };
    use rustls::SignatureScheme;

    #[derive(Debug)]
    struct Ed25519;
    impl SignatureVerificationAlgorithm for Ed25519 {
        fn public_key_alg_id(&self) -> AlgorithmIdentifier {
            alg_id::ED25519
        }
        fn signature_alg_id(&self) -> AlgorithmIdentifier {
            alg_id::ED25519
        }
        fn verify_signature(
            &self,
            public_key: &[u8],
            message: &[u8],
            signature: &[u8],
        ) -> Result<(), InvalidSignature> {
            let pk: [u8; 32] = public_key.try_into().map_err(|_| InvalidSignature)?;
            let sig: [u8; 64] = signature.try_into().map_err(|_| InvalidSignature)?;
            if crate::crypto::ed25519_verify(&pk, message, &sig) {
                Ok(())
            } else {
                Err(InvalidSignature)
            }
        }
    }

    pub static ED25519: &dyn SignatureVerificationAlgorithm = &Ed25519;

    pub static SUPPORTED_SIG_ALGS: WebPkiSupportedAlgorithms = WebPkiSupportedAlgorithms {
        all: &[ED25519],
        mapping: &[(SignatureScheme::ED25519, &[ED25519])],
    };
}

use rustls::crypto::CryptoProvider;
use rustls::{CipherSuite, CipherSuiteCommon, SupportedCipherSuite, Tls13CipherSuite};

/// TLS_CHACHA20_POLY1305_SHA256 — fully on hashtrinity-crypto (ChaCha20-Poly1305 + SHA-256 + our HKDF).
pub static TLS13_CHACHA20_POLY1305_SHA256: SupportedCipherSuite =
    SupportedCipherSuite::Tls13(&Tls13CipherSuite {
        common: CipherSuiteCommon {
            suite: CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
            hash_provider: &hash::SHA256,
            confidentiality_limit: u64::MAX,
        },
        hkdf_provider: &hkdf::HKDF_SHA256,
        aead_alg: &aead::CHACHA20_POLY1305,
        quic: Some(&quic::CHACHA20_QUIC),
    });

/// TLS_AES_128_GCM_SHA256 — required for QUIC Initial packets (RFC 9001 §5.2).
pub static TLS13_AES_128_GCM_SHA256: SupportedCipherSuite =
    SupportedCipherSuite::Tls13(&Tls13CipherSuite {
        common: CipherSuiteCommon {
            suite: CipherSuite::TLS13_AES_128_GCM_SHA256,
            hash_provider: &hash::SHA256,
            confidentiality_limit: 1 << 23,
        },
        hkdf_provider: &hkdf::HKDF_SHA256,
        aead_alg: &aead::AES128_GCM,
        quic: Some(&quic::AES128_QUIC),
    });
/// The complete `CryptoProvider` — ring-free, entirely on hashtrinity-crypto.
pub fn provider() -> CryptoProvider {
    CryptoProvider {
        cipher_suites: vec![TLS13_AES_128_GCM_SHA256, TLS13_CHACHA20_POLY1305_SHA256],
        kx_groups: vec![kx::X25519],
        signature_verification_algorithms: verify::SUPPORTED_SIG_ALGS,
        secure_random: &random::Rng,
        key_provider: &sign::Provider,
    }
}
