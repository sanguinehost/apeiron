//! The current-strata key and its homes, with their exact identity and face capabilities.
//!
//! This child module is included by `strata.rs` and re-exported, so the access paths
//! `crate::strata::{StrataKey, StrataHome, RawStrataHome}` are unchanged. The definitions moved
//! here to keep `strata.rs` below the source-size cap; the strata enumeration and every input are
//! unchanged.
//!
//! Each named home carries the same semantic capability — the exact elliptic packing
//! ([`EllipticPacking`]) and both locator residues ([`FaceResidueCapability`]) — while its `Home`
//! implementation remains a separate physical placement adapter, never the semantic face address:
//!
//! * [`StrataHome`] places a key at `spread(tag).place`.
//! * [`RawStrataHome`] is the raw physical placement, `place = tag`.
//!
//! `StrataKey` is a word wrapper whose equality is exactly `tag` equality, so its exact identity
//! ([`ExactIdentity`]) is the full 64-bit tag and its elliptic packing is the identity on the word.
//! The high tag bits are load-bearing identity, never a reduced residue.

use apeiron::coord::address::{
    EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, ParabolicDual, Residue, H3_40,
};
use apeiron::coord::locator::{ExactIdentity, Refusal};
use apeiron::coord::{spread, Packed};
use apeiron::table::{Home, TableAddress};
use apeiron::Sigma;
use hashtrinity::Coord;

/// A REG-1 current-strata coordinate word. Equality is exactly equality on `tag`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StrataKey {
    pub(crate) tag: u64,
}

impl Coord for StrataKey {
    fn coord(&self) -> u64 {
        self.tag
    }
}

/// The full-width exact identity of a [`StrataKey`]: the eight big-endian bytes of `tag`.
///
/// The retained identity is the tag itself, so equality on the retained value is exactly key
/// equality. Nothing here reduces, truncates or hashes the tag; the high bits are load-bearing
/// identity.
impl ExactIdentity for StrataKey {
    type Retained = u64;
    type Reserved = ();

    fn retain(&self) -> u64 {
        self.tag
    }

    fn reserve_retain(&self) -> Result<(), Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> u64 {
        self.tag
    }

    fn matches_retained(&self, retained: &u64) -> bool {
        self.tag == *retained
    }

    fn encoding_len(&self) -> usize {
        8
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        self.tag.to_be_bytes()[index]
    }

    fn retained_encoding_len(_: &u64) -> usize {
        8
    }

    fn retained_encoding_byte(retained: &u64, index: usize) -> u8 {
        retained.to_be_bytes()[index]
    }
}

/// The capability-bearing current-strata home.
///
/// The `Home` implementation below is the physical placement adapter (`spread(tag).place`); the
/// semantic face map is [`FaceHome`] and must not be read as that physical placement.
pub(crate) struct StrataHome;

impl Home<StrataKey> for StrataHome {
    fn home(&self, key: &StrataKey) -> u64 {
        spread(key.tag).place
    }

    fn address(&self, _sigma: Sigma, key: &StrataKey) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: u32::try_from(key.tag).ok(),
        }
    }
}

impl FaceResidueCapability<StrataKey> for StrataHome {
    fn parabolic_dual(&self, key: &StrataKey) -> ParabolicDual {
        ParabolicDual::new(spread(key.tag).place, 0)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &StrataKey) -> Residue {
        Residue::reduce(profile, key.tag)
    }
}

impl EllipticPacking<StrataKey> for StrataHome {
    fn elliptic_pack(&self, key: &StrataKey) -> u64 {
        <u64 as Packed>::packed(&key.tag)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<StrataKey> {
        Some(StrataKey {
            tag: <u64 as Packed>::unpack(word),
        })
    }
}

impl FaceCoord<StrataKey> for StrataHome {
    type Faces = Sigma;
}

impl FaceHome<StrataKey> for StrataHome {}

/// The raw physical current-strata home: `place = tag`, with the direct word certified below
/// `2^32`.
///
/// Its `Home` implementation is the raw coordinate adapter; the semantic face map is [`FaceHome`]
/// and is kept separate from that physical placement.
pub(crate) struct RawStrataHome;

impl Home<StrataKey> for RawStrataHome {
    fn home(&self, key: &StrataKey) -> u64 {
        key.tag
    }

    fn address(&self, _sigma: Sigma, key: &StrataKey) -> TableAddress {
        TableAddress {
            place: key.tag,
            direct_word: u32::try_from(key.tag).ok(),
        }
    }
}

impl FaceResidueCapability<StrataKey> for RawStrataHome {
    fn parabolic_dual(&self, key: &StrataKey) -> ParabolicDual {
        ParabolicDual::new(spread(key.tag).place, 0)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &StrataKey) -> Residue {
        Residue::reduce(profile, key.tag)
    }
}

impl EllipticPacking<StrataKey> for RawStrataHome {
    fn elliptic_pack(&self, key: &StrataKey) -> u64 {
        <u64 as Packed>::packed(&key.tag)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<StrataKey> {
        Some(StrataKey {
            tag: <u64 as Packed>::unpack(word),
        })
    }
}

impl FaceCoord<StrataKey> for RawStrataHome {
    type Faces = Sigma;
}

impl FaceHome<StrataKey> for RawStrataHome {}

#[cfg(test)]
mod tests {
    use super::{StrataHome, StrataKey};
    use apeiron::coord::address::{EllipticPacking, FaceResidueCapability, Residue, H3_40};
    use apeiron::coord::locator::ExactIdentity;
    use apeiron::coord::spread;

    /// The full-width identity, packing and residue round-trips for `StrataKey`.
    ///
    /// The high tag bits are load-bearing: two keys differing only in the top bit must have
    /// distinct identities. A `retain`, `matches_retained` or `encoding_byte` that ignored them
    /// fails this test, which is the I3-F rejecting control.
    #[test]
    fn strata_key_full_width_round_trip() {
        let low = StrataKey {
            tag: 0x0000_0000_0000_0001,
        };
        let high = StrataKey {
            tag: 0x8000_0000_0000_0001,
        };
        assert_ne!(low, high);

        // The key bytes are the full eight big-endian tag bytes.
        let key_bytes: [u8; 8] = core::array::from_fn(|i| low.encoding_byte(i));
        assert_eq!(key_bytes, low.tag.to_be_bytes());
        assert_eq!(StrataKey::encoding_len(&low), 8);

        // The retained identity is the full tag, and it discriminates the high bit.
        let retained = low.retain();
        assert_eq!(retained, low.tag);
        assert!(low.matches_retained(&retained));
        assert!(!high.matches_retained(&retained));

        // The retained bytes equal the key bytes.
        let retained_bytes: [u8; 8] =
            core::array::from_fn(|i| StrataKey::retained_encoding_byte(&retained, i));
        assert_eq!(retained_bytes, key_bytes);

        // Elliptic packing is exact over the full width.
        let home = StrataHome;
        let packed = home.elliptic_pack(&high);
        assert_eq!(packed, high.tag);
        assert_eq!(home.elliptic_unpack(packed), Some(high));
        assert_eq!(home.elliptic_unpack(low.tag), Some(low));

        // Both face residues are the specified maps, at full width.
        let profile = H3_40::new().expect("the frozen H3_40 token validates");
        assert_eq!(home.parabolic_dual(&high).residue(), spread(high.tag).place);
        assert_eq!(
            home.hyperbolic_scalar(profile, &high),
            Residue::reduce(profile, high.tag)
        );
    }
}
