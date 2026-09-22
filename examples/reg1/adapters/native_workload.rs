//! The bootstrap workloads' keys and homes, with their exact identity and face capabilities.
//!
//! This child module is included by `workload.rs` and re-exported, so the module-local names
//! `ConstantKey`, `ConstantHome`, `ShardKey` and `ShardHome` are unchanged. The definitions moved
//! here to keep `workload.rs` below the source-size cap; every workload, stream, home and oracle
//! is unchanged.
//!
//! [`ShardKey`] has two equality fields, `tag: u32` and `home: u64`, so its exact identity
//! ([`ExactIdentity`]) is the 12-byte concatenation of both fields, most-significant byte first.
//! It cannot be packed into a `u64`; its [`FaceCoord::Faces`] domain is [`LocatorSigma`], which
//! admits the parabolic and hyperbolic faces and no elliptic packing. Both residue encodings fold
//! the whole 96-bit key with widened `u128` arithmetic:
//!
//! * [`ShardHome::parabolic_dual`] writes the 96-bit integer `home << 32 | tag` as the dual number
//!   `a + b ε`: `a` is the low 64 bits, `b` the high 32 bits.
//! * [`ShardHome::hyperbolic_scalar`] reduces that same 96-bit integer modulo `q = 3^40`.
//!
//! [`ConstantKey`] is a word wrapper whose equality is exactly `tag` equality, so its exact
//! identity is the full 32-bit tag and its elliptic packing is the identity on the widened word.
//! Its `Home` placement stays the constant `0`; the semantic face map is separate.
//!
//! The `Home` implementations here are the physical placement adapters, never the semantic face
//! address. The physical placement and the [`FaceHome`] map are kept separate.

use apeiron::coord::address::{
    EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, LocatorSigma, ParabolicDual,
    Residue, H3_40,
};
use apeiron::coord::locator::{ExactIdentity, Refusal};
use apeiron::coord::{spread, Packed};
use apeiron::table::{Home, TableAddress};
use apeiron::Sigma;
use hashtrinity::Coord;

/// A key with a *constant* placement word. Identity is the tag; both candidate tables receive the
/// same constant, isolating the two failure modes under test: `hashtrinity` coalesces all 17 keys
/// onto one coordinate, `apeiron` admits at most `N` keys per home shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ConstantKey {
    pub(crate) tag: u32,
}

const CONSTANT_COORD: u64 = 1;

impl Coord for ConstantKey {
    fn coord(&self) -> u64 {
        CONSTANT_COORD
    }
}

/// The full-width exact identity of a [`ConstantKey`]: the four big-endian bytes of `tag`.
///
/// The retained identity is the tag itself, so equality on the retained value is exactly key
/// equality. Nothing here reduces, truncates or hashes the tag; the high bits are load-bearing
/// identity.
impl ExactIdentity for ConstantKey {
    type Retained = u32;
    type Reserved = ();

    fn retain(&self) -> u32 {
        self.tag
    }

    fn reserve_retain(&self) -> Result<(), Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> u32 {
        self.tag
    }

    fn matches_retained(&self, retained: &u32) -> bool {
        self.tag == *retained
    }

    fn encoding_len(&self) -> usize {
        4
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        self.tag.to_be_bytes()[index]
    }

    fn retained_encoding_len(_: &u32) -> usize {
        4
    }

    fn retained_encoding_byte(retained: &u32, index: usize) -> u8 {
        retained.to_be_bytes()[index]
    }
}

/// The Apeiron `Home` that maps every key to shard 0, so the fixed-shard capacity is exercised.
/// Placement-only: it certifies no exact coordinate word, so `direct_word` is `None`.
pub(crate) struct ConstantHome;

impl Home<ConstantKey> for ConstantHome {
    fn home(&self, _key: &ConstantKey) -> u64 {
        0
    }

    fn address(&self, _sigma: Sigma, _key: &ConstantKey) -> TableAddress {
        TableAddress {
            place: 0,
            direct_word: None,
        }
    }
}

/// The semantic face capability of [`ConstantHome`]: the full 32-bit tag on all three faces.
///
/// The physical placement above stays the constant `0`; this map is the semantic face address and
/// must not be read as that placement.
impl FaceResidueCapability<ConstantKey> for ConstantHome {
    fn parabolic_dual(&self, key: &ConstantKey) -> ParabolicDual {
        ParabolicDual::new(spread(u64::from(key.tag)).place, 0)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &ConstantKey) -> Residue {
        Residue::reduce(profile, u64::from(key.tag))
    }
}

impl EllipticPacking<ConstantKey> for ConstantHome {
    fn elliptic_pack(&self, key: &ConstantKey) -> u64 {
        <u32 as Packed>::packed(&key.tag)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<ConstantKey> {
        Some(ConstantKey {
            tag: <u32 as Packed>::unpack(word),
        })
    }
}

impl FaceCoord<ConstantKey> for ConstantHome {
    type Faces = Sigma;
}

impl FaceHome<ConstantKey> for ConstantHome {}

/// A key whose placement is an explicit field, so a workload controls its home shard directly.
///
/// `Coord` is the distinct tag, so the pinned hashtrinity table keeps every key separate; `Home`
/// is the field, so keys sharing a home necessarily share an Apeiron shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ShardKey {
    pub(crate) tag: u32,
    pub(crate) home: u64,
}

impl Coord for ShardKey {
    fn coord(&self) -> u64 {
        u64::from(self.tag)
    }
}

/// The exact identity of a [`ShardKey`]: the 12 bytes `tag.to_be_bytes() ++ home.to_be_bytes()`.
///
/// Both equality fields are retained: two keys with the same `tag` and different `home` have
/// distinct identities and distinct encodings. The 96-bit key cannot be packed into a `u64`, so
/// there is no elliptic packing and the face domain is [`LocatorSigma`].
///
/// Temporal O(1): two field reads and one 12-byte encoding. Spatial O(1): no allocation.
impl ExactIdentity for ShardKey {
    type Retained = (u32, u64);
    type Reserved = ();

    fn retain(&self) -> (u32, u64) {
        (self.tag, self.home)
    }

    fn reserve_retain(&self) -> Result<(), Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> (u32, u64) {
        (self.tag, self.home)
    }

    fn matches_retained(&self, retained: &(u32, u64)) -> bool {
        self.tag == retained.0 && self.home == retained.1
    }

    fn encoding_len(&self) -> usize {
        12
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        if index < 4 {
            self.tag.to_be_bytes()[index]
        } else {
            self.home.to_be_bytes()[index - 4]
        }
    }

    fn retained_encoding_len(_: &(u32, u64)) -> usize {
        12
    }

    fn retained_encoding_byte(retained: &(u32, u64), index: usize) -> u8 {
        if index < 4 {
            retained.0.to_be_bytes()[index]
        } else {
            retained.1.to_be_bytes()[index - 4]
        }
    }
}

pub(crate) struct ShardHome;

impl Home<ShardKey> for ShardHome {
    fn home(&self, key: &ShardKey) -> u64 {
        key.home
    }

    /// Placement-only: the explicit `home` field is the placement and no exact coordinate word is
    /// certified, so `direct_word` is `None`.
    fn address(&self, _sigma: Sigma, key: &ShardKey) -> TableAddress {
        TableAddress {
            place: key.home,
            direct_word: None,
        }
    }
}

/// The 96-bit key integer `home << 32 | tag`.
///
/// Temporal O(1), spatial O(1). The widening to `u128` is exact: `home` contributes bits `32..96`
/// and `tag` bits `0..32`.
fn wide_key(key: &ShardKey) -> u128 {
    (u128::from(key.home) << 32) | u128::from(key.tag)
}

/// The semantic face capability of [`ShardHome`]: the two locator faces over the 96-bit key.
///
/// Both maps fold the whole key with widened `u128` arithmetic. The identity bytes, not either
/// residue, decide equality, so a shared residue fiber is not a key collision.
impl FaceResidueCapability<ShardKey> for ShardHome {
    fn parabolic_dual(&self, key: &ShardKey) -> ParabolicDual {
        let wide = wide_key(key);
        let scalar = u64::try_from(wide & u128::from(u64::MAX)).expect("the low 64 bits fit u64");
        let nilpotent = u64::try_from(wide >> 64).expect("the high 32 bits fit u64");
        ParabolicDual::new(scalar, nilpotent)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &ShardKey) -> Residue {
        let reduced = u64::try_from(wide_key(key) % u128::from(profile.q()))
            .expect("a remainder modulo q fits u64");
        Residue::new(profile, reduced).expect("a remainder modulo q is canonical")
    }
}

impl FaceCoord<ShardKey> for ShardHome {
    type Faces = LocatorSigma;
}

impl FaceHome<ShardKey> for ShardHome {}

#[cfg(test)]
mod tests {
    use super::{ConstantHome, ConstantKey, ShardHome, ShardKey};
    use apeiron::coord::address::{
        EllipticPacking, FaceResidueCapability, LocatorSigma, Residue, H3_40,
    };
    use apeiron::coord::locator::ExactIdentity;
    use apeiron::table::Home;

    /// Both `ShardKey` fields are load-bearing identity.
    ///
    /// Two keys with the same `tag` and different `home` are distinct, have distinct 12-byte
    /// encodings, and distinct retained identities. Omitting `home` from the identity bytes or the
    /// retained value fails this test, which is the I3-H rejecting control.
    #[test]
    fn shard_key_same_tag_different_home_identity() {
        let a = ShardKey { tag: 7, home: 1 };
        let b = ShardKey { tag: 7, home: 2 };
        assert_ne!(a, b);

        // The exact encoding is 12 bytes: four tag bytes then eight home bytes.
        assert_eq!(ShardKey::encoding_len(&a), 12);
        let a_bytes: [u8; 12] = core::array::from_fn(|i| a.encoding_byte(i));
        let b_bytes: [u8; 12] = core::array::from_fn(|i| b.encoding_byte(i));
        assert_ne!(a_bytes, b_bytes);
        assert_eq!(&a_bytes[0..4], &a.tag.to_be_bytes());
        assert_eq!(&a_bytes[4..12], &a.home.to_be_bytes());

        // The retained identity keeps both fields and discriminates the home.
        let retained = a.retain();
        assert_eq!(retained, (a.tag, a.home));
        assert!(a.matches_retained(&retained));
        assert!(!b.matches_retained(&retained));
        let retained_bytes: [u8; 12] =
            core::array::from_fn(|i| ShardKey::retained_encoding_byte(&retained, i));
        assert_eq!(retained_bytes, a_bytes);
    }

    /// Both locator faces fold the whole 96-bit key with widened arithmetic.
    #[test]
    fn shard_key_face_encodings_fold_both_fields() {
        let profile = H3_40::new().expect("the frozen H3_40 token validates");
        let home = ShardHome;
        let key = ShardKey {
            tag: 0xDEAD_BEEF,
            home: 0x1234_5678_9ABC_DEF0,
        };
        let wide = (u128::from(key.home) << 32) | u128::from(key.tag);

        let dual = home.parabolic_dual(&key);
        assert_eq!(
            dual.scalar(),
            u64::try_from(wide & u128::from(u64::MAX)).expect("the low 64 bits fit u64")
        );
        assert_eq!(
            dual.nilpotent(),
            u64::try_from(wide >> 64).expect("the high 32 bits fit u64")
        );

        let expected =
            u64::try_from(wide % u128::from(profile.q())).expect("the remainder fits u64");
        assert_eq!(
            home.hyperbolic_scalar(profile, &key),
            Residue::new(profile, expected).expect("the remainder is canonical")
        );

        // `ShardHome` names the locator domain, so no elliptic packing exists for the 96-bit key.
        assert_eq!(
            core::any::TypeId::of::<
                <ShardHome as apeiron::coord::address::FaceCoord<ShardKey>>::Faces,
            >(),
            core::any::TypeId::of::<LocatorSigma>()
        );
    }

    /// `ConstantKey` keeps its full 32-bit tag while its physical placement stays the constant `0`.
    #[test]
    fn constant_key_full_width_round_trip() {
        let low = ConstantKey { tag: 0x0000_0001 };
        let high = ConstantKey { tag: 0x8000_0001 };
        assert_ne!(low, high);

        assert_eq!(ConstantKey::encoding_len(&low), 4);
        let key_bytes: [u8; 4] = core::array::from_fn(|i| low.encoding_byte(i));
        assert_eq!(key_bytes, low.tag.to_be_bytes());

        let retained = low.retain();
        assert!(low.matches_retained(&retained));
        assert!(!high.matches_retained(&retained));
        let retained_bytes: [u8; 4] =
            core::array::from_fn(|i| ConstantKey::retained_encoding_byte(&retained, i));
        assert_eq!(retained_bytes, key_bytes);

        let home = ConstantHome;
        assert_eq!(home.elliptic_pack(&high), u64::from(high.tag));
        assert_eq!(home.elliptic_unpack(u64::from(high.tag)), Some(high));
        // The physical placement is constant; the semantic map is separate.
        assert_eq!(home.home(&high), 0);
    }
}
