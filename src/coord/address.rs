//! Face-tagged semantic addresses and the one admitted hyperbolic profile.
//!
//! This module is contract vocabulary, not storage. It separates a semantic [`AddressValue`]
//! from a physical locator: [`FaceTableAddress`] carries the address and nothing else. The
//! executing table keeps its own [`crate::table::TableAddress`]; nothing here is wired into a
//! table read or write.
//!
//! Faces follow the per-face contract in `docs/epics/algebra-native-addressing.md`:
//!
//! | face | [`Sigma`] | address ring | word map |
//! |---|---|---|---|
//! | elliptic | `Elliptic` | `Z/2^64` | the exact unsigned word |
//! | parabolic | `Parabolic` | `Z/2^64` | [`spread`]`.place` |
//! | hyperbolic | `Hyperbolic` | `Z/3^40` | [`Residue::reduce`] |

use crate::cell::Sigma;
use crate::coord::{spread, Packed};

/// The one admitted hyperbolic profile: `p = 3`, `k = 40`, `q = 3^40`.
///
/// The field is private and [`H3_40::new`] validates the frozen constants, so a raw modulus
/// cannot forge a token. A validated token is immutable and copied by value; there is no mutable
/// global modulus and no per-query profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct H3_40 {
    _sealed: (),
}

/// The plan's name for [`H3_40`]: the sealed, validated hyperbolic profile.
pub type HyperbolicProfile = H3_40;

impl H3_40 {
    /// The prime base `p`.
    pub const P: u64 = 3;
    /// The exponent `k`.
    pub const K: u32 = 40;
    /// The modulus `q = 3^40 = 12157665459056928801`, the largest power of `3` that fits a `u64`.
    pub const Q: u64 = 12_157_665_459_056_928_801;
    /// The unit root `r`, with `r * r = sigma = 1`.
    pub const R: u64 = 1;
    /// The inverse unit root `s`, with `r * s = 1`.
    pub const S: u64 = 1;
    /// The inverse of `2` modulo `q`: `2 * INVERSE_OF_TWO mod q = 1`.
    pub const INVERSE_OF_TWO: u64 = 6_078_832_729_528_464_401;
    /// The profile and encoding version.
    pub const VERSION: u32 = 1;

    /// Validates an exact parameter tuple against the frozen constants.
    ///
    /// Requires the version, `(base, exponent) = (3, 40)`, `40` checked multiplications by `3`,
    /// the width `2^63 < q < 2^64 < 3^41`, an odd modulus, and the unit and root equations in
    /// `u128`. Any altered modulus, exponent, root, half or version returns `false`.
    #[must_use]
    pub fn validate(
        version: u32,
        base: u64,
        exponent: u32,
        modulus: u64,
        root: u64,
        inverse_root: u64,
        half: u64,
    ) -> bool {
        if version != Self::VERSION || base != Self::P || exponent != Self::K {
            return false;
        }
        let mut power: u64 = 1;
        let mut step = 0;
        while step < exponent {
            match power.checked_mul(base) {
                Some(next) => power = next,
                None => return false,
            }
            step += 1;
        }
        if power != modulus {
            return false;
        }
        // Width: `q` is the largest power of `3` that fits `u64`.
        if modulus <= (1u64 << 63) || modulus.checked_mul(base).is_some() {
            return false;
        }
        // `2` is a unit exactly when the modulus is odd.
        if modulus % 2 != 1 {
            return false;
        }
        // Unit and root equations, widened so no product wraps.
        let wide = u128::from(modulus);
        if (u128::from(root) * u128::from(inverse_root)) % wide != 1 {
            return false;
        }
        if (2 * u128::from(half)) % wide != 1 {
            return false;
        }
        if (u128::from(root) * u128::from(root)) % wide != 1 {
            return false;
        }
        root == Self::R && inverse_root == Self::S && half == Self::INVERSE_OF_TWO
    }

    /// The one admitted token, or `None` if the frozen constants fail [`H3_40::validate`].
    #[must_use]
    pub fn new() -> Option<Self> {
        if Self::validate(
            Self::VERSION,
            Self::P,
            Self::K,
            Self::Q,
            Self::R,
            Self::S,
            Self::INVERSE_OF_TWO,
        ) {
            Some(Self { _sealed: () })
        } else {
            None
        }
    }

    /// The modulus `q`.
    #[must_use]
    pub const fn q(self) -> u64 {
        Self::Q
    }

    /// The inverse of `2` modulo `q`.
    #[must_use]
    pub const fn inverse_of_two(self) -> u64 {
        Self::INVERSE_OF_TWO
    }

    /// The unit root `r`.
    #[must_use]
    pub const fn r(self) -> u64 {
        Self::R
    }

    /// The inverse unit root `s`.
    #[must_use]
    pub const fn s(self) -> u64 {
        Self::S
    }

    /// The profile and encoding version.
    #[must_use]
    pub const fn version(self) -> u32 {
        Self::VERSION
    }
}

/// A canonical hyperbolic residue: an integer in `[0, q)`.
///
/// The field is private. [`Residue::new`] is checked and [`Residue::reduce`] is the total word
/// reduction, so a raw word cannot become a residue without passing through one of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Residue(u64);

impl Residue {
    /// The checked canonical constructor: `Some` exactly for `value < q`.
    #[must_use]
    pub const fn new(profile: H3_40, value: u64) -> Option<Self> {
        if value < profile.q() {
            Some(Self(value))
        } else {
            None
        }
    }

    /// The total word reduction `w < q ? w : w - q`.
    ///
    /// Since `u64::MAX < 2 * q`, the subtraction is exactly integer remainder modulo `q`, with no
    /// underflow or wrapping. This is the hyperbolic successor to the old unkeyed fold; it is not
    /// a cast, a mask or [`spread`].
    #[must_use]
    pub const fn reduce(profile: H3_40, word: u64) -> Self {
        let q = profile.q();
        if word < q {
            Self(word)
        } else {
            Self(word - q)
        }
    }

    /// The residue as an integer in `[0, q)`.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A parabolic dual number `a + b ε` in `Z/2^64[ε]/ε²`, with `ε² = 0`.
///
/// The parabolic address is the scalar coefficient `a`, the residue modulo the ideal `(ε)`. In the
/// `TrivSqZeroExt` representation this is `fstHom(a + bε) = a`. The nilpotent coefficient `b` is
/// carried so a caller can state the full ring element; the locator indexes by `a` and never reads
/// `b`, so `b` cannot decide identity or absence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParabolicDual {
    scalar: u64,
    nilpotent: u64,
}

impl ParabolicDual {
    /// The dual number `scalar + nilpotent * ε`.
    #[must_use]
    pub const fn new(scalar: u64, nilpotent: u64) -> Self {
        Self { scalar, nilpotent }
    }

    /// The scalar coefficient `a`.
    #[must_use]
    pub const fn scalar(self) -> u64 {
        self.scalar
    }

    /// The nilpotent coefficient `b`.
    #[must_use]
    pub const fn nilpotent(self) -> u64 {
        self.nilpotent
    }

    /// `fstHom`: the residue modulo `(ε)`, which is exactly the scalar coefficient `a`.
    #[must_use]
    pub const fn residue(self) -> u64 {
        self.scalar
    }
}

/// The semantic address of one face, tagged by its variant.
///
/// The variant determines the face ([`AddressValue::face`]); a caller cannot supply a [`Sigma`]
/// inconsistent with the payload. Equal numeric payloads on different variants are different
/// addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddressValue {
    /// The elliptic face: the exact unsigned word in `Z/2^64`.
    EllipticWord(u64),
    /// The parabolic face: the spreading fold `spread(word).place` in `Z/2^64`.
    ParabolicWord(u64),
    /// The hyperbolic face: the validated profile and a canonical residue in `Z/q`.
    HyperbolicResidue(H3_40, Residue),
}

impl AddressValue {
    /// The face this address belongs to.
    #[must_use]
    pub const fn face(self) -> Sigma {
        match self {
            Self::EllipticWord(_) => Sigma::Elliptic,
            Self::ParabolicWord(_) => Sigma::Parabolic,
            Self::HyperbolicResidue(_, _) => Sigma::Hyperbolic,
        }
    }

    /// The exact word of an elliptic address, `None` on the other faces.
    #[must_use]
    pub const fn elliptic_word(self) -> Option<u64> {
        match self {
            Self::EllipticWord(word) => Some(word),
            _ => None,
        }
    }

    /// The word of an elliptic or parabolic address, `None` on the hyperbolic face.
    #[must_use]
    pub const fn word(self) -> Option<u64> {
        match self {
            Self::EllipticWord(word) | Self::ParabolicWord(word) => Some(word),
            Self::HyperbolicResidue(_, _) => None,
        }
    }

    /// The canonical residue of a hyperbolic address, `None` on the other faces.
    #[must_use]
    pub const fn residue(self) -> Option<Residue> {
        match self {
            Self::HyperbolicResidue(_, residue) => Some(residue),
            _ => None,
        }
    }

    /// The sealed profile of a hyperbolic address, `None` on the other faces.
    #[must_use]
    pub const fn profile(self) -> Option<H3_40> {
        match self {
            Self::HyperbolicResidue(profile, _) => Some(profile),
            _ => None,
        }
    }
}

/// A semantic face address with no physical locator.
///
/// The value determines the face. There is deliberately no `usize`/`u32` index and no exactness
/// flag: a locator is a separate, checked concern, never a cast of this value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceTableAddress {
    value: AddressValue,
}

impl FaceTableAddress {
    /// Wraps a semantic address.
    #[must_use]
    pub const fn new(value: AddressValue) -> Self {
        Self { value }
    }

    /// The semantic address.
    #[must_use]
    pub const fn value(self) -> AddressValue {
        self.value
    }

    /// The face the address belongs to.
    #[must_use]
    pub const fn face(self) -> Sigma {
        self.value.face()
    }
}

/// The sealed supertrait closing [`FaceDomain`] to the two admitted domains.
///
/// A downstream crate cannot name this module, so it cannot implement [`FaceDomain`] for its own
/// type. The only implementations are [`Sigma`] and [`LocatorSigma`].
mod sealed {
    /// Closed to this module: only [`Sigma`] and [`LocatorSigma`] implement it.
    pub trait Sealed {}
}

/// The two faces that carry a secondary locator.
///
/// The elliptic face is exact packing with no locator, so it is deliberately absent from this
/// enum. A [`FaceDomain`] whose token is [`LocatorSigma`] therefore cannot name the elliptic face:
/// the restriction is a type, not a run-time `UnsupportedFace` branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum LocatorSigma {
    /// `sigma = 0`: the dual numbers, indexed by the scalar coefficient `a`.
    #[default]
    Parabolic,
    /// `sigma > 0`: the split-complex numbers, indexed by the `H3_40` scalar residue.
    Hyperbolic,
}

impl LocatorSigma {
    /// The dial this token selects.
    #[must_use]
    pub const fn dial(self) -> Sigma {
        match self {
            Self::Parabolic => Sigma::Parabolic,
            Self::Hyperbolic => Sigma::Hyperbolic,
        }
    }

    /// The locator token for an admitted dial, `None` for elliptic.
    #[must_use]
    pub const fn token_for(sigma: Sigma) -> Option<Self> {
        match sigma {
            Sigma::Parabolic => Some(Self::Parabolic),
            Sigma::Hyperbolic => Some(Self::Hyperbolic),
            Sigma::Elliptic => None,
        }
    }
}

impl sealed::Sealed for Sigma {}
impl sealed::Sealed for LocatorSigma {}

/// The residue maps every admitted home supplies for the two locator faces.
///
/// The parabolic address is the scalar coefficient of the dual number; the hyperbolic address is
/// the canonical residue modulo the `H3_40` modulus. Implementing this is a claim that the returned
/// values are the specified encodings, not a proof certificate.
pub trait FaceResidueCapability<K> {
    /// The dual number whose scalar coefficient `a` indexes the parabolic face.
    fn parabolic_dual(&self, key: &K) -> ParabolicDual;

    /// The canonical `H3_40` residue indexing the hyperbolic face.
    fn hyperbolic_scalar(&self, profile: H3_40, key: &K) -> Residue;
}

/// Exact at-most-word packing: the elliptic capability.
///
/// The law is `elliptic_unpack(elliptic_pack(k)) == Some(k)` on the declared packable domain. This
/// is the only face whose address determines the key, so a home without this capability cannot use
/// the all-face [`Sigma`] domain.
///
/// ```compile_fail
/// use apeiron::coord::address::{
///     FaceCoord, FaceResidueCapability, ParabolicDual, Residue, H3_40,
/// };
/// use apeiron::Sigma;
///
/// struct ResidueOnly;
///
/// impl FaceResidueCapability<u64> for ResidueOnly {
///     fn parabolic_dual(&self, _key: &u64) -> ParabolicDual {
///         ParabolicDual::new(0, 0)
///     }
///     fn hyperbolic_scalar(&self, profile: H3_40, _key: &u64) -> Residue {
///         Residue::new(profile, 0).expect("zero is canonical")
///     }
/// }
///
/// // Rejected: `ResidueOnly` has no `EllipticPacking`, so `Sigma: FaceDomain<u64, ResidueOnly>`
/// // does not hold and the elliptic token cannot be named on this home.
/// impl FaceCoord<u64> for ResidueOnly {
///     type Faces = Sigma;
/// }
/// ```
pub trait EllipticPacking<K> {
    /// The exact word for `key`.
    fn elliptic_pack(&self, key: &K) -> u64;

    /// The key that word came from, `None` off the packing's image.
    fn elliptic_unpack(&self, word: u64) -> Option<K>;
}

/// A sealed face-domain token: the compile-time set of faces a home admits.
///
/// The only implementations are [`Sigma`] (all three faces, requiring [`EllipticPacking`]) and
/// [`LocatorSigma`] (parabolic and hyperbolic only). Both default to parabolic.
/// [`FaceDomain::encode`] is the shared exhaustive dispatch; it is not a user callback.
pub trait FaceDomain<K, H: ?Sized>: sealed::Sealed + Copy {
    /// The default face of this domain: parabolic for both admitted domains.
    const DEFAULT: Self;

    /// The [`Sigma`] dial this token selects.
    fn sigma(self) -> Sigma;

    /// The token for `sigma`, `None` when the domain does not admit that face.
    fn from_sigma(sigma: Sigma) -> Option<Self>;

    /// Encodes `key` on this face using the home's capabilities.
    fn encode(self, home: &H, profile: H3_40, key: &K) -> FaceTableAddress;
}

/// The admitted face encoding of a key: the capability behind a home.
///
/// This is a law, not a certificate: implementing it claims that `coord` returns the specified
/// face encoding for every admitted `(profile, face, key)`. The associated [`FaceCoord::Faces`]
/// token is the home's whole admitted domain, so a locator-only home cannot be passed elliptic.
pub trait FaceCoord<K>: FaceResidueCapability<K> {
    /// The faces this home admits.
    type Faces: FaceDomain<K, Self>;

    /// Encodes `key` on `face`. The default is the domain's shared dispatch.
    fn coord(&self, profile: H3_40, face: Self::Faces, key: &K) -> FaceTableAddress {
        face.encode(self, profile, key)
    }
}

/// A stable, equality-respecting face map supplied to a table.
///
/// `address` must agree with [`FaceCoord::coord`] and be deterministic for the lifetime of the
/// home. There is no blanket implementation for placement-only closures.
pub trait FaceHome<K>: FaceCoord<K> {
    /// The stable map. The default delegates to [`FaceCoord::coord`].
    fn address(&self, profile: H3_40, face: Self::Faces, key: &K) -> FaceTableAddress {
        self.coord(profile, face, key)
    }
}

impl<K, H: ?Sized> FaceDomain<K, H> for Sigma
where
    H: FaceResidueCapability<K> + EllipticPacking<K>,
{
    const DEFAULT: Self = Self::Parabolic;

    fn sigma(self) -> Sigma {
        self
    }

    fn from_sigma(sigma: Sigma) -> Option<Self> {
        Some(sigma)
    }

    fn encode(self, home: &H, profile: H3_40, key: &K) -> FaceTableAddress {
        match LocatorSigma::token_for(self) {
            Some(locator) => locator.encode(home, profile, key),
            None => FaceTableAddress::new(AddressValue::EllipticWord(home.elliptic_pack(key))),
        }
    }
}

impl<K, H: ?Sized> FaceDomain<K, H> for LocatorSigma
where
    H: FaceResidueCapability<K>,
{
    const DEFAULT: Self = Self::Parabolic;

    fn sigma(self) -> Sigma {
        self.dial()
    }

    fn from_sigma(sigma: Sigma) -> Option<Self> {
        Self::token_for(sigma)
    }

    fn encode(self, home: &H, profile: H3_40, key: &K) -> FaceTableAddress {
        let value = match self {
            Self::Parabolic => AddressValue::ParabolicWord(home.parabolic_dual(key).residue()),
            Self::Hyperbolic => {
                AddressValue::HyperbolicResidue(profile, home.hyperbolic_scalar(profile, key))
            }
        };
        FaceTableAddress::new(value)
    }
}

/// The word face instance: `u64` and `i64` keys on all three faces.
///
/// `i64` is interpreted as its two's-complement bit pattern, never as a signed remainder. Its
/// admitted domain is [`Sigma`], so it must supply exact elliptic packing.
#[derive(Clone, Copy, Debug, Default)]
pub struct WordFaceHome;

impl FaceResidueCapability<u64> for WordFaceHome {
    fn parabolic_dual(&self, key: &u64) -> ParabolicDual {
        ParabolicDual::new(spread(*key).place, 0)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &u64) -> Residue {
        Residue::reduce(profile, *key)
    }
}

impl FaceResidueCapability<i64> for WordFaceHome {
    fn parabolic_dual(&self, key: &i64) -> ParabolicDual {
        ParabolicDual::new(spread(key.cast_unsigned()).place, 0)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &i64) -> Residue {
        Residue::reduce(profile, key.cast_unsigned())
    }
}

impl EllipticPacking<u64> for WordFaceHome {
    fn elliptic_pack(&self, key: &u64) -> u64 {
        <u64 as Packed>::packed(key)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<u64> {
        Some(<u64 as Packed>::unpack(word))
    }
}

impl EllipticPacking<i64> for WordFaceHome {
    fn elliptic_pack(&self, key: &i64) -> u64 {
        <i64 as Packed>::packed(key)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<i64> {
        Some(<i64 as Packed>::unpack(word))
    }
}

impl FaceCoord<u64> for WordFaceHome {
    type Faces = Sigma;
}

impl FaceCoord<i64> for WordFaceHome {
    type Faces = Sigma;
}

impl FaceHome<u64> for WordFaceHome {}

impl FaceHome<i64> for WordFaceHome {}

/// The physical index a face selects, mirroring `Apeiron.NativeLocator.FaceIndexPolicy`.
///
/// Elliptic addresses are exact packings with no secondary locator; parabolic and hyperbolic
/// addresses are located by a binary and a ternary residue trie respectively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceIndexPolicy {
    /// Elliptic: the exact `pack`, no locator.
    ExactPackNoLocator,
    /// Parabolic: a base-2 residue trie over the scalar coefficient `a`.
    Base2ResidueTrie,
    /// Hyperbolic: the base-3 residue trie over the canonical residue.
    Base3ResidueTrie,
}

/// The index policy of a face.
#[must_use]
pub const fn face_index_policy(sigma: Sigma) -> FaceIndexPolicy {
    match sigma {
        Sigma::Elliptic => FaceIndexPolicy::ExactPackNoLocator,
        Sigma::Parabolic => FaceIndexPolicy::Base2ResidueTrie,
        Sigma::Hyperbolic => FaceIndexPolicy::Base3ResidueTrie,
    }
}
