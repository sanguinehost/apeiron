//! The additive W1 address contract: face-tagged values and their laws.
//!
//! These tests exercise the library bodies in `src/map/coord/address.rs`. They do not migrate a table
//! read, write, carrier or storage path; the old `WordHome`/`TableAddress` fixture is checked
//! unchanged at the end.

use apeiron::map::coord::address::{
    AddressValue, EllipticPacking, FaceCoord, FaceDomain, FaceHome, FaceResidueCapability,
    FaceTableAddress, LocatorSigma, ParabolicDual, Residue, WordFaceHome, H3_40,
};
use apeiron::map::coord::locator::{ExactIdentity, Refusal};
use apeiron::map::coord::{spread, Packed};
use apeiron::map::table::{Home, WordHome};
use apeiron::Sigma;

fn profile() -> H3_40 {
    H3_40::new().expect("the frozen H3_40 constants validate")
}

fn elliptic(address: FaceTableAddress) -> u64 {
    address
        .value()
        .elliptic_word()
        .expect("an elliptic address carries an exact word")
}

fn residue_value(address: FaceTableAddress) -> u64 {
    address
        .value()
        .residue()
        .expect("a hyperbolic address carries a residue")
        .get()
}

/// A physical locator is a checked conversion, never a cast of the semantic address.
fn checked_locator(word: u64) -> Option<u32> {
    u32::try_from(word).ok()
}

#[test]
fn profile_is_valid_and_rejects_other_parameters() {
    let p = profile();
    assert_eq!(p.q(), 12_157_665_459_056_928_801);
    assert_eq!(p.inverse_of_two(), 6_078_832_729_528_464_401);
    assert_eq!(p.r(), 1);
    assert_eq!(p.s(), 1);
    assert_eq!(p.version(), 1);

    assert!(H3_40::validate(
        1,
        3,
        40,
        H3_40::Q,
        1,
        1,
        H3_40::INVERSE_OF_TWO
    ));
    // `2` is a unit: the modulus is odd and its inverse satisfies `2 * h = 1 (mod q)`.
    assert_eq!(H3_40::Q % 2, 1);
    assert_eq!(2 * H3_40::INVERSE_OF_TWO % H3_40::Q, 1);

    // A wrong version, base, exponent, modulus, root, inverse root or half is rejected.
    let half = H3_40::INVERSE_OF_TWO;
    assert!(!H3_40::validate(2, 3, 40, H3_40::Q, 1, 1, half));
    assert!(!H3_40::validate(1, 5, 40, H3_40::Q, 1, 1, half));
    assert!(!H3_40::validate(1, 3, 39, H3_40::Q, 1, 1, half));
    assert!(!H3_40::validate(1, 3, 40, H3_40::Q + 1, 1, 1, half));
    assert!(!H3_40::validate(1, 3, 40, H3_40::Q, 2, 1, half));
    assert!(!H3_40::validate(1, 3, 40, H3_40::Q, 1, 2, half));
    assert!(!H3_40::validate(1, 3, 40, H3_40::Q, 1, 1, half + 1));
    assert!(!H3_40::validate(1, 3, 40, H3_40::Q, 1, 1, 0));
}

#[test]
fn residue_constructor_rejects_noncanonical() {
    let p = profile();
    assert_eq!(Residue::new(p, 0).map(Residue::get), Some(0));
    assert_eq!(Residue::new(p, 1).map(Residue::get), Some(1));
    assert_eq!(
        Residue::new(p, H3_40::Q - 1).map(Residue::get),
        Some(H3_40::Q - 1)
    );
    assert_eq!(Residue::new(p, H3_40::Q), None);
    assert_eq!(Residue::new(p, H3_40::Q + 1), None);
    assert_eq!(Residue::new(p, u64::MAX), None);
}

#[test]
fn word_reduction_matches_integer_remainder() {
    let p = profile();
    let mut samples = vec![
        0,
        1,
        H3_40::Q - 1,
        H3_40::Q,
        H3_40::Q + 1,
        1u64 << 63,
        u64::MAX,
    ];
    samples.extend((0u64..64).map(|i| H3_40::Q.wrapping_add(i)));
    samples.extend([2, 3, 10, 1_000_000_007, (1u64 << 63) + 12_345]);

    for word in samples {
        let expected = u64::try_from(u128::from(word) % u128::from(H3_40::Q))
            .expect("a remainder modulo q is below 2^64");
        assert_eq!(Residue::reduce(p, word).get(), expected, "reduce({word})");
        // Total and deterministic: the same word reduces the same way.
        assert_eq!(Residue::reduce(p, word), Residue::reduce(p, word));
    }
    assert_eq!(
        Residue::reduce(p, u64::MAX).get(),
        6_289_078_614_652_622_814
    );
}

#[test]
fn elliptic_full_word_is_not_reduced() {
    let p = profile();
    let h = WordFaceHome;
    let words = [
        0u64,
        1,
        H3_40::Q - 1,
        H3_40::Q,
        H3_40::Q + 1,
        1u64 << 63,
        u64::MAX,
    ];
    for word in words {
        assert_eq!(elliptic(h.coord(p, Sigma::Elliptic, &word)), word);
        assert_eq!(
            residue_value(h.coord(p, Sigma::Hyperbolic, &word)),
            Residue::reduce(p, word).get()
        );
        // The elliptic exact packing round-trips over the whole word.
        assert_eq!(
            <u64 as Packed>::unpack(<u64 as Packed>::packed(&word)),
            word
        );
        assert_eq!(
            elliptic(h.coord(p, Sigma::Elliptic, &word)),
            <u64 as Packed>::packed(&word)
        );
    }
    // The decisive boundary: elliptic `q` stays `q`; hyperbolic `q` becomes zero.
    assert_eq!(elliptic(h.coord(p, Sigma::Elliptic, &H3_40::Q)), H3_40::Q);
    assert_eq!(residue_value(h.coord(p, Sigma::Hyperbolic, &H3_40::Q)), 0);
}

#[test]
fn parabolic_uses_existing_spread() {
    let p = profile();
    let h = WordFaceHome;
    for word in [0u64, 1, 7, 1u64 << 32, 1u64 << 63, H3_40::Q, u64::MAX] {
        let address = h.coord(p, Sigma::Parabolic, &word);
        assert_eq!(address.value().word(), Some(spread(word).place));
        assert_eq!(address.face(), Sigma::Parabolic);
    }
    let high = 0xDEAD_BEEF_0000_0001u64;
    assert_eq!(
        h.coord(p, Sigma::Parabolic, &high).value().word(),
        Some(spread(high).place)
    );
}

#[test]
fn faces_and_profile_are_preserved() {
    let p = profile();
    let h = WordFaceHome;
    for key in [0i64, 1, -1, 7, i64::MIN, i64::MAX] {
        for sigma in [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic] {
            let via_home = FaceHome::address(&h, p, sigma, &key);
            let via_coord = FaceCoord::coord(&h, p, sigma, &key);
            assert_eq!(via_home, via_coord);
            assert_eq!(via_home.face(), sigma);
            // Stable and equality-respecting: repeats and a copy agree.
            assert_eq!(via_home, FaceHome::address(&h, p, sigma, &key));
            let copied = via_home;
            assert_eq!(copied, via_home);
        }
    }

    // All three variants are reachable and report their face.
    let on_elliptic = FaceTableAddress::new(AddressValue::EllipticWord(9));
    let on_parabolic = FaceTableAddress::new(AddressValue::ParabolicWord(9));
    let on_hyperbolic =
        FaceTableAddress::new(AddressValue::HyperbolicResidue(p, Residue::reduce(p, 9)));
    assert_eq!(on_elliptic.face(), Sigma::Elliptic);
    assert_eq!(on_parabolic.face(), Sigma::Parabolic);
    assert_eq!(on_hyperbolic.face(), Sigma::Hyperbolic);
    assert_eq!(on_elliptic.value().elliptic_word(), Some(9));
    assert_eq!(on_parabolic.value().word(), Some(9));
    assert_eq!(on_hyperbolic.value().residue().map(Residue::get), Some(9));
    // The sealed profile travels with the hyperbolic address.
    assert_eq!(on_hyperbolic.value().profile(), Some(p));
}

#[test]
fn signed_words_keep_bits_and_collisions_keep_identity() {
    let p = profile();
    let h = WordFaceHome;
    for key in [i64::MIN, -1, 0, 1, i64::MAX] {
        let word = key.cast_unsigned();
        assert_eq!(elliptic(h.coord(p, Sigma::Elliptic, &key)), word);
        assert_eq!(<i64 as Packed>::unpack(<i64 as Packed>::packed(&key)), key);
        assert_eq!(
            residue_value(h.coord(p, Sigma::Hyperbolic, &key)),
            Residue::reduce(p, word).get()
        );
    }
    // Distinct original word keys `0` and `q` share one hyperbolic residue but stay distinct.
    assert_eq!(
        residue_value(h.coord(p, Sigma::Hyperbolic, &0u64)),
        residue_value(h.coord(p, Sigma::Hyperbolic, &H3_40::Q))
    );
    assert_eq!(residue_value(h.coord(p, Sigma::Hyperbolic, &0u64)), 0);
    assert_ne!(
        h.coord(p, Sigma::Elliptic, &0u64),
        h.coord(p, Sigma::Elliptic, &H3_40::Q)
    );
}

#[test]
fn legacy_word_home_is_unchanged() {
    for sigma in [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic] {
        for key in [
            0i64,
            7,
            -1,
            i64::MAX,
            i64::MIN,
            (1i64 << 32) - 1,
            1i64 << 32,
        ] {
            let word = key.cast_unsigned();
            let address = Home::address(&WordHome, sigma, &key);
            assert_eq!(address.place, spread(word).place, "place at {sigma:?}");
            assert_eq!(
                address.direct_word,
                u32::try_from(word).ok(),
                "direct_word at {sigma:?}"
            );
        }
    }
}

#[test]
fn locator_is_not_a_cast() {
    // The semantic address keeps all 64 bits; it is never narrowed to a physical index.
    let full = FaceTableAddress::new(AddressValue::EllipticWord(u64::MAX));
    assert_eq!(full.value().elliptic_word(), Some(u64::MAX));
    assert_ne!(full.value().elliptic_word(), Some(u64::from(u32::MAX)));

    // A physical locator is a separate, checked conversion: a cast would truncate, this rejects.
    assert_eq!(checked_locator(u64::MAX), None);
    assert_eq!(checked_locator(u64::from(u32::MAX)), Some(u32::MAX));
    assert_eq!(checked_locator(7), Some(7));

    // The hyperbolic address keeps the full-width residue, not a narrowed locator.
    let p = profile();
    let address = FaceTableAddress::new(AddressValue::HyperbolicResidue(
        p,
        Residue::reduce(p, u64::MAX),
    ));
    assert_eq!(
        address.value().residue().map(Residue::get),
        Some(6_289_078_614_652_622_814)
    );
}

#[test]
fn residue_capability_is_the_specified_face_map() {
    let p = profile();
    let h = WordFaceHome;
    for word in [0u64, 1, 7, H3_40::Q - 1, H3_40::Q, 1u64 << 63, u64::MAX] {
        let dual: ParabolicDual = h.parabolic_dual(&word);
        // The parabolic address is `ParabolicWord(parabolic_dual(k).residue())`.
        assert_eq!(
            dual.residue(),
            spread(word).place,
            "parabolic dual of {word}"
        );
        // The nilpotent coefficient is carried, never an index.
        assert_eq!(dual.nilpotent(), 0);
        // The hyperbolic address is the canonical `H3_40` residue.
        assert_eq!(
            h.hyperbolic_scalar(p, &word),
            Residue::reduce(p, word),
            "hyperbolic scalar of {word}"
        );
    }
    for key in [i64::MIN, -1, 0, 1, i64::MAX] {
        let word = key.cast_unsigned();
        assert_eq!(h.parabolic_dual(&key).residue(), spread(word).place);
        assert_eq!(h.hyperbolic_scalar(p, &key), Residue::reduce(p, word));
    }
}

#[test]
fn elliptic_packing_round_trips_on_its_domain() {
    let h = WordFaceHome;
    for word in [0u64, 1, 7, H3_40::Q - 1, H3_40::Q, 1u64 << 63, u64::MAX] {
        assert_eq!(h.elliptic_pack(&word), <u64 as Packed>::packed(&word));
        assert_eq!(h.elliptic_unpack(h.elliptic_pack(&word)), Some(word));
    }
    for key in [i64::MIN, -1, 0, 1, i64::MAX] {
        assert_eq!(h.elliptic_pack(&key), <i64 as Packed>::packed(&key));
        assert_eq!(h.elliptic_unpack(h.elliptic_pack(&key)), Some(key));
    }
}

#[test]
fn locator_sigma_admits_only_the_filtered_faces() {
    // The elliptic face has no locator and is deliberately unnameable in this enum.
    assert_eq!(LocatorSigma::token_for(Sigma::Elliptic), None);
    assert_eq!(
        LocatorSigma::token_for(Sigma::Parabolic),
        Some(LocatorSigma::Parabolic)
    );
    assert_eq!(
        LocatorSigma::token_for(Sigma::Hyperbolic),
        Some(LocatorSigma::Hyperbolic)
    );
    assert_eq!(LocatorSigma::Parabolic.dial(), Sigma::Parabolic);
    assert_eq!(LocatorSigma::Hyperbolic.dial(), Sigma::Hyperbolic);

    // Both admitted domains default to parabolic.
    assert_eq!(
        <Sigma as FaceDomain<u64, WordFaceHome>>::DEFAULT,
        Sigma::Parabolic
    );
    assert_eq!(
        <LocatorSigma as FaceDomain<u64, WordFaceHome>>::DEFAULT,
        LocatorSigma::Parabolic
    );

    // The trait-level restriction matches the enum: elliptic has no locator token.
    assert_eq!(
        <LocatorSigma as FaceDomain<u64, WordFaceHome>>::from_sigma(Sigma::Elliptic),
        None
    );
    assert_eq!(
        <Sigma as FaceDomain<u64, WordFaceHome>>::from_sigma(Sigma::Elliptic),
        Some(Sigma::Elliptic)
    );
}

#[test]
fn locator_domain_shares_the_face_dispatch() {
    let p = profile();
    let h = WordFaceHome;
    for word in [0u64, 1, 7, 1u64 << 32, 1u64 << 63, H3_40::Q, u64::MAX] {
        let parabolic = <LocatorSigma as FaceDomain<u64, WordFaceHome>>::encode(
            LocatorSigma::Parabolic,
            &h,
            p,
            &word,
        );
        let hyperbolic = <LocatorSigma as FaceDomain<u64, WordFaceHome>>::encode(
            LocatorSigma::Hyperbolic,
            &h,
            p,
            &word,
        );
        // The locator-only domain uses the same encoding as the all-face `Sigma` domain.
        assert_eq!(parabolic, h.coord(p, Sigma::Parabolic, &word));
        assert_eq!(hyperbolic, h.coord(p, Sigma::Hyperbolic, &word));
        assert_eq!(parabolic.face(), Sigma::Parabolic);
        assert_eq!(hyperbolic.face(), Sigma::Hyperbolic);
    }
}

#[test]
fn word_face_home_admits_the_sigma_domain() {
    let p = profile();
    let h = WordFaceHome;
    for sigma in [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic] {
        let via_coord = h.coord(p, sigma, &7u64);
        assert_eq!(via_coord.face(), sigma);
        assert_eq!(via_coord, FaceHome::address(&h, p, sigma, &7u64));
    }
    // The all-face domain requires exact packing: the elliptic word is the exact pack.
    assert_eq!(
        elliptic(h.coord(p, Sigma::Elliptic, &7u64)),
        h.elliptic_pack(&7u64)
    );
    assert_eq!(
        h.coord(p, Sigma::Elliptic, &7u64).value().elliptic_word(),
        Some(7)
    );
}

// ── 96-bit locator-domain witness ────────────────────────────────────────────────────────────

/// A 12-byte key with two equality fields.
///
/// Its exact identity is the 12-byte concatenation `tag.to_be_bytes() ++ home.to_be_bytes()`, so
/// neither field is truncated. The 96-bit integer cannot be packed into a `u64`, so the home below
/// names [`LocatorSigma`] and supplies no elliptic packing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct WideKey {
    tag: u32,
    home: u64,
}

/// The 96-bit key integer `home << 32 | tag`.
fn wide_integer(key: &WideKey) -> u128 {
    (u128::from(key.home) << 32) | u128::from(key.tag)
}

/// The 12-byte exact identity of a [`WideKey`]: four tag bytes then eight `home` bytes.
impl ExactIdentity for WideKey {
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

/// A 96-bit home: the locator-only domain.
///
/// It supplies [`FaceResidueCapability`] only, so the compile-positive `type Faces = LocatorSigma`
/// below holds and no elliptic packing exists. Both residues fold the whole 96-bit key with
/// widened `u128` arithmetic.
struct WideHome;

impl FaceResidueCapability<WideKey> for WideHome {
    fn parabolic_dual(&self, key: &WideKey) -> ParabolicDual {
        let wide = wide_integer(key);
        ParabolicDual::new(
            u64::try_from(wide & u128::from(u64::MAX)).expect("the low 64 bits fit u64"),
            u64::try_from(wide >> 64).expect("the high 32 bits fit u64"),
        )
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &WideKey) -> Residue {
        let reduced = u64::try_from(wide_integer(key) % u128::from(profile.q()))
            .expect("a remainder modulo q fits u64");
        Residue::new(profile, reduced).expect("a remainder modulo q is canonical")
    }
}

// Compile-positive: a home with only the residue capability names the locator domain. The
// all-face `Sigma` domain would require `EllipticPacking` for this 96-bit key, which does not
// exist, so `type Faces = LocatorSigma` is the only compiling choice.
impl FaceCoord<WideKey> for WideHome {
    type Faces = LocatorSigma;
}

impl FaceHome<WideKey> for WideHome {}

#[test]
fn locator_domain_admits_a_twelve_byte_key() {
    let p = profile();
    let h = WideHome;
    let key = WideKey {
        tag: 0xDEAD_BEEF,
        home: 0x1234_5678_9ABC_DEF0,
    };

    // The compile-positive domain witness: the home names `LocatorSigma`, not `Sigma`.
    assert_eq!(
        core::any::TypeId::of::<<WideHome as FaceCoord<WideKey>>::Faces>(),
        core::any::TypeId::of::<LocatorSigma>()
    );

    // Both locator tokens are admitted and encode through the shared dispatch.
    let parabolic = <LocatorSigma as FaceDomain<WideKey, WideHome>>::encode(
        LocatorSigma::Parabolic,
        &h,
        p,
        &key,
    );
    let hyperbolic = <LocatorSigma as FaceDomain<WideKey, WideHome>>::encode(
        LocatorSigma::Hyperbolic,
        &h,
        p,
        &key,
    );
    assert_eq!(parabolic.face(), Sigma::Parabolic);
    assert_eq!(hyperbolic.face(), Sigma::Hyperbolic);
    assert_eq!(
        parabolic.value().word(),
        Some(h.parabolic_dual(&key).residue())
    );
    assert_eq!(
        hyperbolic.value().residue(),
        Some(h.hyperbolic_scalar(p, &key))
    );

    // The elliptic token is not admitted on this domain: a compile-time absence, not a branch.
    assert_eq!(
        <LocatorSigma as FaceDomain<WideKey, WideHome>>::from_sigma(Sigma::Elliptic),
        None
    );
}

#[test]
fn wide_key_residue_collision_keeps_exact_identity() {
    let p = profile();
    let h = WideHome;

    // `home = 0` and `home = q` are distinct keys whose 96-bit integers both reduce to residue 0.
    let zero = WideKey { tag: 0, home: 0 };
    let q_home = WideKey {
        tag: 0,
        home: H3_40::Q,
    };
    assert_ne!(zero, q_home);
    assert_eq!(
        h.hyperbolic_scalar(p, &zero),
        h.hyperbolic_scalar(p, &q_home)
    );
    assert_eq!(h.hyperbolic_scalar(p, &zero).get(), 0);

    // The exact identity bytes and the retained identity are the whole 12-byte key, never the
    // shared residue. A residue-only adapter collapses the two keys and fails these assertions.
    let zero_bytes: [u8; 12] = core::array::from_fn(|i| zero.encoding_byte(i));
    let q_bytes: [u8; 12] = core::array::from_fn(|i| q_home.encoding_byte(i));
    assert_ne!(zero_bytes, q_bytes);
    assert_eq!(&zero_bytes[0..4], &zero.tag.to_be_bytes());
    assert_eq!(&q_bytes[4..12], &q_home.home.to_be_bytes());
    assert_ne!(zero.retain(), q_home.retain());
    assert!(!q_home.matches_retained(&zero.retain()));
    let retained_bytes: [u8; 12] =
        core::array::from_fn(|i| WideKey::retained_encoding_byte(&zero.retain(), i));
    assert_eq!(retained_bytes, zero_bytes);
}
