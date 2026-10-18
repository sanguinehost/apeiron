//! Compile-time key storage policies for [`super::Table`].
//!
//! `Stored` is the parabolic carrier: it retains `K`. `Regen` is the elliptic
//! carrier: it retains the exact identity word and recovers `K` only through a
//! reviewed inverse codec. Neither carrier stores `Sigma`.

use core::marker::PhantomData;

use crate::map::coord::address::{FaceDomain, FaceHome, FaceTableAddress, H3_40};
use crate::map::coord::locator::ExactIdentity;
use crate::{cell::Sigma, map::coord};

use super::{Home, TableAddress};

mod sealed {
    pub trait Sealed {}
}

/// What one live slot retains for a key face.
///
/// `matches` is the hot identity operation. `key_owned` is intentionally a
/// cold materialization operation; the table's reads and updates do not call
/// it.
pub trait KeyRep<K, H>: sealed::Sealed + Clone {
    /// Retains the representation selected by this carrier.
    fn hold(sigma: Sigma, key: &K, address: &TableAddress) -> Self;
    /// Decides identity for a query without materializing a stored key.
    fn matches(sigma: Sigma, carrier: &Self, key: &K, address: &TableAddress) -> bool;
    /// Decides identity between two retained carriers during relocation.
    fn matches_held(sigma: Sigma, carrier: &Self, other: &Self) -> bool;
    /// Materializes a key on a cold path.
    fn key_owned(sigma: Sigma, carrier: &Self) -> K;
    /// Recovers an address for relocation without cloning an arbitrary carried key.
    fn address_held(home: &H, sigma: Sigma, carrier: &Self) -> TableAddress;

    /// Number of exact identity bytes retained by this carrier.
    fn held_encoding_len(carrier: &Self) -> usize;
    /// Byte `index` of the retained exact identity.
    ///
    /// # Panics
    ///
    /// Panics when `index >= Self::held_encoding_len(carrier)`.
    fn held_encoding_byte(carrier: &Self, index: usize) -> u8;
    /// Number of exact identity bytes of a query key.
    fn query_encoding_len(key: &K) -> usize;
    /// Byte `index` of the query key's exact identity.
    ///
    /// # Panics
    ///
    /// Panics when `index >= Self::query_encoding_len(key)`.
    fn query_encoding_byte(key: &K, index: usize) -> u8;
    /// The semantic face address of the retained carrier.
    ///
    /// The carrier supplies the key without a duplicate store: `Stored` borrows its retained `K`,
    /// `Regen` decodes its independent exact identity on this cold path.
    fn face_address_held(
        home: &H,
        profile: H3_40,
        face: H::Faces,
        carrier: &Self,
    ) -> FaceTableAddress
    where
        H: FaceHome<K>;
}

/// Parabolic policy: retain the complete key.
///
/// An opaque `Eq`-only key is rejected at compile time: the sealed instance requires
/// [`ExactIdentity`], so there is no carrier for a key with no equality-faithful byte encoding.
///
/// ```compile_fail
/// use apeiron::map::coord::address::{
///     EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, ParabolicDual, Residue, H3_40,
/// };
/// use apeiron::map::table::keyrep::{KeyRep, Stored};
/// use apeiron::map::table::{Home, TableAddress};
/// use apeiron::Sigma;
///
/// #[derive(Clone, PartialEq, Eq)]
/// struct OpaqueEq(u8);
///
/// struct OpaqueHome;
///
/// impl FaceResidueCapability<OpaqueEq> for OpaqueHome {
///     fn parabolic_dual(&self, _: &OpaqueEq) -> ParabolicDual { ParabolicDual::new(0, 0) }
///     fn hyperbolic_scalar(&self, profile: H3_40, _: &OpaqueEq) -> Residue {
///         Residue::new(profile, 0).expect("zero is canonical")
///     }
/// }
///
/// impl EllipticPacking<OpaqueEq> for OpaqueHome {
///     fn elliptic_pack(&self, _: &OpaqueEq) -> u64 { 0 }
///     fn elliptic_unpack(&self, _: u64) -> Option<OpaqueEq> { Some(OpaqueEq(0)) }
/// }
///
/// impl FaceCoord<OpaqueEq> for OpaqueHome { type Faces = Sigma; }
/// impl FaceHome<OpaqueEq> for OpaqueHome {}
///
/// impl Home<OpaqueEq> for OpaqueHome {
///     fn home(&self, _: &OpaqueEq) -> u64 { 0 }
///     fn address(&self, _: Sigma, _: &OpaqueEq) -> TableAddress {
///         TableAddress { place: 0, direct_word: None }
///     }
/// }
///
/// fn requires<R: KeyRep<OpaqueEq, OpaqueHome>>() {}
///
/// // Rejected: `OpaqueEq` has no `ExactIdentity`, so `Stored<OpaqueEq>` cannot implement `KeyRep`.
/// requires::<Stored<OpaqueEq>>();
/// ```
#[derive(Clone)]
pub struct Stored<K> {
    key: K,
}

impl<K> sealed::Sealed for Stored<K> {}

impl<K: ExactIdentity + Clone, H: Home<K>> KeyRep<K, H> for Stored<K> {
    fn hold(_: Sigma, key: &K, _: &TableAddress) -> Self {
        Self { key: key.clone() }
    }

    fn matches(_: Sigma, carrier: &Self, key: &K, _: &TableAddress) -> bool {
        carrier.key == *key
    }

    fn matches_held(_: Sigma, carrier: &Self, other: &Self) -> bool {
        carrier.key == other.key
    }

    fn key_owned(_: Sigma, carrier: &Self) -> K {
        carrier.key.clone()
    }

    fn address_held(home: &H, sigma: Sigma, carrier: &Self) -> TableAddress {
        Home::address(home, sigma, &carrier.key)
    }

    fn held_encoding_len(carrier: &Self) -> usize {
        carrier.key.encoding_len()
    }

    fn held_encoding_byte(carrier: &Self, index: usize) -> u8 {
        carrier.key.encoding_byte(index)
    }

    fn query_encoding_len(key: &K) -> usize {
        key.encoding_len()
    }

    fn query_encoding_byte(key: &K, index: usize) -> u8 {
        key.encoding_byte(index)
    }

    fn face_address_held(
        home: &H,
        profile: H3_40,
        face: H::Faces,
        carrier: &Self,
    ) -> FaceTableAddress
    where
        H: FaceHome<K>,
    {
        FaceHome::address(home, profile, face, &carrier.key)
    }
}

/// Exact elliptic codecs admitted to [`Regen`].
///
/// The sealed set prevents a caller from selecting key-dropping storage with
/// an unreviewed non-injective codec. Its required law is, on the codec's
/// admitted domain: `decode(sigma, encode(sigma, key, address)) == key`.
/// ```compile_fail
/// use apeiron::map::table::keyrep::ExactCodec;
/// struct Lossy;
/// // Fails: only codecs reviewed in this module can implement ExactCodec.
/// impl ExactCodec<u64, ()> for Lossy {
///     fn encode(_: apeiron::Sigma, _: &u64, _: &apeiron::map::table::TableAddress) -> u64 { 0 }
///     fn decode(_: apeiron::Sigma, _: u64) -> u64 { 0 }
/// }
/// ```
pub trait ExactCodec<K, H>: sealed::Sealed
where
    K: ExactIdentity,
    H: Home<K>,
{
    /// The exact identity coordinate retained by `Regen`.
    fn encode(sigma: Sigma, key: &K, address: &TableAddress) -> u64;
    /// The verified inverse of [`Self::encode`], never an inverse of `place`.
    fn decode(sigma: Sigma, identity: u64) -> K;
    /// Number of exact identity bytes of a retained identity word.
    fn identity_encoding_len(identity: u64) -> usize;
    /// Byte `index` of a retained identity word's exact identity bytes.
    ///
    /// These bytes must equal the query's canonical bytes after [`Self::encode`], so `Regen`
    /// retains the exact key identity rather than a reduced residue.
    ///
    /// # Panics
    ///
    /// Panics when `index >= Self::identity_encoding_len(identity)`.
    fn identity_encoding_byte(identity: u64, index: usize) -> u8;
}

/// Elliptic policy: retain only an exact identity coordinate.
pub struct Regen<C> {
    identity: u64,
    codec: PhantomData<C>,
}

impl<C> Clone for Regen<C> {
    fn clone(&self) -> Self {
        Self {
            identity: self.identity,
            codec: PhantomData,
        }
    }
}

impl<C> sealed::Sealed for Regen<C> {}

impl<K: ExactIdentity, H: Home<K>, C: ExactCodec<K, H>> KeyRep<K, H> for Regen<C> {
    fn hold(sigma: Sigma, key: &K, address: &TableAddress) -> Self {
        Self {
            identity: C::encode(sigma, key, address),
            codec: PhantomData,
        }
    }

    fn matches(sigma: Sigma, carrier: &Self, key: &K, address: &TableAddress) -> bool {
        carrier.identity == C::encode(sigma, key, address)
    }

    fn matches_held(_: Sigma, carrier: &Self, other: &Self) -> bool {
        carrier.identity == other.identity
    }

    fn key_owned(sigma: Sigma, carrier: &Self) -> K {
        C::decode(sigma, carrier.identity)
    }

    fn address_held(home: &H, sigma: Sigma, carrier: &Self) -> TableAddress {
        let key = C::decode(sigma, carrier.identity);
        Home::address(home, sigma, &key)
    }

    fn held_encoding_len(carrier: &Self) -> usize {
        C::identity_encoding_len(carrier.identity)
    }

    fn held_encoding_byte(carrier: &Self, index: usize) -> u8 {
        C::identity_encoding_byte(carrier.identity, index)
    }

    fn query_encoding_len(key: &K) -> usize {
        key.encoding_len()
    }

    fn query_encoding_byte(key: &K, index: usize) -> u8 {
        key.encoding_byte(index)
    }

    fn face_address_held(
        home: &H,
        profile: H3_40,
        face: H::Faces,
        carrier: &Self,
    ) -> FaceTableAddress
    where
        H: FaceHome<K>,
    {
        let key = C::decode(face.sigma(), carrier.identity);
        FaceHome::address(home, profile, face, &key)
    }
}

/// Reviewed exact codec for the `i64` word coordinate.
///
/// It stores `Addr.id = x * PHI` and uses `unspread` as its inverse. The
/// codec law is `decode(s, encode(s, k, a)) == k` for every `i64`; `sigma` is
/// carried by the surrounding slot and does not alter this word bijection.
pub struct I64WordCodec;

impl sealed::Sealed for I64WordCodec {}

impl<H: Home<i64>> ExactCodec<i64, H> for I64WordCodec {
    fn encode(_: Sigma, key: &i64, address: &TableAddress) -> u64 {
        let id = coord::spread((*key).cast_unsigned()).id;
        debug_assert_eq!(id, address_identity(address, *key));
        id
    }

    fn decode(_: Sigma, identity: u64) -> i64 {
        coord::unspread(identity).cast_signed()
    }

    fn identity_encoding_len(_: u64) -> usize {
        8
    }

    fn identity_encoding_byte(identity: u64, index: usize) -> u8 {
        // Recover the original bit pattern by word arithmetic; the residue is never reused.
        coord::unspread(identity).to_be_bytes()[index]
    }
}

#[inline]
fn address_identity(address: &TableAddress, key: i64) -> u64 {
    let id = coord::spread(key.cast_unsigned()).id;
    let _ = address;
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::table::{ReplacementMerge, Table, WordHome};

    struct PanicDecodeCodec;

    impl sealed::Sealed for PanicDecodeCodec {}

    impl ExactCodec<i64, WordHome> for PanicDecodeCodec {
        fn encode(_: Sigma, key: &i64, _: &TableAddress) -> u64 {
            coord::spread((*key).cast_unsigned()).id
        }

        fn decode(_: Sigma, _: u64) -> i64 {
            panic!("direct resident hit materialized a stored key")
        }

        fn identity_encoding_len(_: u64) -> usize {
            8
        }

        fn identity_encoding_byte(identity: u64, index: usize) -> u8 {
            coord::unspread(identity).to_be_bytes()[index]
        }
    }

    /// A direct elliptic hit compares `Held` to the query coordinate. Calling
    /// `key_owned` would call this codec's panicking `decode` and fail.
    #[test]
    fn direct_regen_hit_never_materializes_the_stored_key() {
        let mut table = Table::<i64, i64, WordHome, 16, Regen<PanicDecodeCodec>>::new(0, WordHome);
        table
            .insert_with_at(&ReplacementMerge, Sigma::Elliptic, 7, 41)
            .unwrap();
        assert_eq!(*table.get_at(Sigma::Elliptic, &7), 41);
        table
            .insert_with_at(&ReplacementMerge, Sigma::Elliptic, 7, 42)
            .unwrap();
        assert_eq!(*table.get_at(Sigma::Elliptic, &7), 42);
    }
}
