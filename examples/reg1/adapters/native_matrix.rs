//! The shared fixed-16 workload key and its home, with their exact identity and face capabilities.
//!
//! This child module is included by `matrix.rs` and re-exported, so the access paths
//! `crate::matrix::{SharedKey, SharedHome}` are unchanged. The definitions moved here to keep
//! `matrix.rs` below the source-size cap; the workload, its rows and every resource outcome are
//! unchanged.
//!
//! [`SharedHome`] carries the same semantic capability — the exact elliptic packing
//! ([`EllipticPacking`]) and both locator residues ([`FaceResidueCapability`]) — while its `Home`
//! implementation remains a separate physical placement adapter, never the semantic face address:
//! the raw physical placement is `home = u64::from(tag)`, and the semantic face map is
//! [`FaceHome`]. The two are kept separate.
//!
//! [`SharedKey`] is a word wrapper whose equality is exactly `tag` equality, so its exact identity
//! ([`ExactIdentity`]) is the full 32-bit tag and its elliptic packing is the identity on the
//! widened word. The tag is the whole identity: no high tag bit is truncated, masked or reduced.

use apeiron::coord::address::{
    EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, ParabolicDual, Residue, H3_40,
};
use apeiron::coord::locator::{ExactIdentity, Refusal};
use apeiron::coord::{spread, Packed};
use apeiron::table::{Home, TableAddress};
use apeiron::Sigma;
use hashtrinity::Coord;

/// The shared workload key. `tag` is the hashtrinity `Coord` (injective across the admitted set, so
/// identity is preserved).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SharedKey {
    pub(crate) tag: u32,
}

impl Coord for SharedKey {
    fn coord(&self) -> u64 {
        u64::from(self.tag)
    }
}

/// The full-width exact identity of a [`SharedKey`]: the four big-endian bytes of `tag`.
///
/// The retained identity is the tag itself, so equality on the retained value is exactly key
/// equality. Nothing here reduces, truncates or hashes the tag; the high bits are load-bearing
/// identity.
impl ExactIdentity for SharedKey {
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

/// The explicit Apeiron placement: the full injective coordinate.
///
/// This is the benchmark-boundary change. The historical home returned `tag % 256`; this home
/// returns the full `tag`, and both `apeiron::Table` and the candidate reduce `% 256` internally
/// before selecting a shard. Placement is therefore identical, while the candidate can cache the
/// full home word and filter on it. [`crate::matrix::run_home_boundary`] records both properties.
///
/// The `Home` implementation below is the physical placement adapter; the semantic face map is
/// [`FaceHome`] and must not be read as that physical placement.
pub(crate) struct SharedHome;

impl Home<SharedKey> for SharedHome {
    fn home(&self, key: &SharedKey) -> u64 {
        u64::from(key.tag)
    }

    /// The production address for the exact word `tag`: `place = spread(tag).place` and
    /// `direct_word = Some(tag as u32)` exactly when `tag < 2^32` (always here, since `tag: u32`).
    /// `home` keeps returning the full tag for the benchmark-only candidate's cached-word filter;
    /// only `address`, the interface the production table calls, carries the certified direct word.
    fn address(&self, _sigma: Sigma, key: &SharedKey) -> TableAddress {
        let word = u64::from(key.tag);
        TableAddress {
            place: spread(word).place,
            direct_word: u32::try_from(word).ok(),
        }
    }
}

impl FaceResidueCapability<SharedKey> for SharedHome {
    fn parabolic_dual(&self, key: &SharedKey) -> ParabolicDual {
        ParabolicDual::new(spread(u64::from(key.tag)).place, 0)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &SharedKey) -> Residue {
        // The tag is widened to `u64` before the total remainder reduction: the high tag bits are
        // part of the residue, never truncated.
        Residue::reduce(profile, u64::from(key.tag))
    }
}

impl EllipticPacking<SharedKey> for SharedHome {
    fn elliptic_pack(&self, key: &SharedKey) -> u64 {
        <u32 as Packed>::packed(&key.tag)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<SharedKey> {
        Some(SharedKey {
            tag: <u32 as Packed>::unpack(word),
        })
    }
}

impl FaceCoord<SharedKey> for SharedHome {
    type Faces = Sigma;
}

impl FaceHome<SharedKey> for SharedHome {}

#[cfg(test)]
mod tests {
    use super::{SharedHome, SharedKey};
    use apeiron::coord::address::{EllipticPacking, FaceResidueCapability, Residue, H3_40};
    use apeiron::coord::locator::ExactIdentity;
    use apeiron::coord::spread;

    /// The full-width identity, packing and residue round-trips for `SharedKey`.
    ///
    /// The tag is the whole 32-bit identity: two keys differing only in the top bit must have
    /// distinct identities and distinct exact bytes. The hyperbolic oracle widens the tag to `u64`
    /// and reduces it; an encoding that truncated the high tag bits fails the `u32::MAX` case
    /// below, which is the I3-G rejecting control.
    #[test]
    fn shared_key_full_width_round_trip() {
        let low = SharedKey { tag: 0x0000_0001 };
        let high = SharedKey { tag: 0x8000_0001 };
        assert_ne!(low, high);

        // The key bytes are the full four big-endian tag bytes.
        let key_bytes: [u8; 4] = core::array::from_fn(|i| low.encoding_byte(i));
        assert_eq!(key_bytes, low.tag.to_be_bytes());
        assert_eq!(SharedKey::encoding_len(&low), 4);

        // The retained identity is the full tag, and it discriminates the high bit.
        let retained = low.retain();
        assert_eq!(retained, low.tag);
        assert!(low.matches_retained(&retained));
        assert!(!high.matches_retained(&retained));

        // The retained bytes equal the key bytes.
        let retained_bytes: [u8; 4] =
            core::array::from_fn(|i| SharedKey::retained_encoding_byte(&retained, i));
        assert_eq!(retained_bytes, key_bytes);

        // Elliptic packing is exact over the full width.
        let home = SharedHome;
        let packed = home.elliptic_pack(&high);
        assert_eq!(packed, u64::from(high.tag));
        assert_eq!(home.elliptic_unpack(packed), Some(high));
        assert_eq!(home.elliptic_unpack(u64::from(low.tag)), Some(low));

        // Both face residues are the specified maps, at full width. The hyperbolic oracle widens
        // the tag to `u64` before reducing, so `u32::MAX` must reduce to its own value, not to a
        // high-bit-truncated residue.
        let profile = H3_40::new().expect("the frozen H3_40 token validates");
        let max = SharedKey { tag: u32::MAX };
        assert_eq!(
            home.hyperbolic_scalar(profile, &max),
            Residue::reduce(profile, u64::from(u32::MAX))
        );
        assert_eq!(
            home.parabolic_dual(&high).residue(),
            spread(u64::from(high.tag)).place
        );
        assert_eq!(
            home.hyperbolic_scalar(profile, &high),
            Residue::reduce(profile, u64::from(high.tag))
        );
    }
}
