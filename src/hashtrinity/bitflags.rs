//! `bitflags` masquerade — a `bitflags!` macro generating the same public API as the real crate
//! (v2): a newtype over an integer, named associated consts, the set-algebra methods, operator
//! overloads, and the `"A | B | 0x4"` pretty-print. Zero dependencies — this REPLACES `bitflags`,
//! it doesn't wrap it.
//!
//! ```ignore
//! use apeiron::hashtrinity::bitflags;
//!
//! bitflags! {
//!     #[derive(PartialEq, Eq)]
//!     pub struct Flags: u32 {
//!         const A = 0b001;
//!         const B = 0b010;
//!         const C = 0b100;
//!         const AB = Self::A.bits() | Self::B.bits();
//!     }
//! }
//!
//! let e = Flags::A | Flags::C;
//! assert!(e.contains(Flags::A));
//! assert!(!e.contains(Flags::B));
//! assert_eq!(format!("{e:?}"), "A | C");
//! assert_eq!((Flags::AB - Flags::A).bits(), Flags::B.bits());
//! ```
//!
//! ## What's ported and what's dropped
//!
//! The real crate splits its own generated type into an internal/public layer specifically so
//! `Debug` can be baked in on the hidden inner type while leaving the OUTER type's `Debug` (and
//! `Clone`/`Copy`/`PartialEq`/`Eq`/`Hash`/…) entirely up to the caller's own `#[derive(..)]` — the
//! two compose (`#[derive(Debug)]` on a tuple struct prints `Outer(<field's Debug>)`), which is
//! how real usage gets `Flags(A | C)` even though the crate's own tests write `#[derive(Debug,
//! ...)]` on every invocation (verified against the real crate directly: dropping the two-layer
//! split and baking `Debug` in here directly, instead, means callers must NOT also `#[derive(Debug)]`
//! — a real, deliberate deviation, not an oversight: replicating the internal/public split just to
//! reproduce the `Outer(...)` wrapping isn't worth doubling every type this macro generates.
//! `Clone`/`Copy`/`PartialEq`/`Eq`/`PartialOrd`/`Ord`/`Hash` are NOT baked in here either — same as
//! the real crate, entirely on the caller's `#[derive(..)]`, and for the same reason (baking any
//! of them in here would conflict with a caller who derives it too, verified as an `E0119` hard
//! error either way — the `Flags` trait's default methods are written to never need `Self: Copy`
//! so leaving it out costs nothing).
//!
//! Also not built: signed integer storage (real but rare — flags are unsigned bit patterns in
//! practice), the `arbitrary`/`bytemuck` integrations, and the full `const _ = !0;`
//! "these bits are all I expect" compatibility-mask syntax. Narrow, honestly-labeled gaps, not
//! silently dropped surface (see `MASQUERADE.md`).
//!
//! ## Serde — the `AsTerm` analog, not real `serde`
//!
//! The real crate's `serde` feature serializes human-readable output as the same `"A | B"` string
//! `Debug`/`Display` produce. Feature `syntax` here does the same thing through `AsTerm`
//! (`codec::syntax`, hashtrinity's own serde-derive analog) instead of bridging to real `serde` —
//! nothing here has ever depended on it, so there's nothing to bridge to.

/// A single named (or unnamed) flag definition — mirrors the real crate's `Flag<B>`.
#[derive(Debug)]
pub struct Flag<B> {
    name: &'static str,
    value: B,
}

impl<B> Flag<B> {
    pub const fn new(name: &'static str, value: B) -> Self {
        Flag { name, value }
    }
    pub const fn name(&self) -> &'static str {
        self.name
    }
    pub const fn value(&self) -> &B {
        &self.value
    }
    pub const fn is_named(&self) -> bool {
        !self.name.is_empty()
    }
}

/// An unsigned integer usable as flags storage. Implemented for every unsigned integer width —
/// signed storage is real in the original crate but unused in practice, and isn't built here.
pub trait Bits:
    Copy
    + Clone
    + PartialEq
    + Eq
    + core::ops::BitOr<Output = Self>
    + core::ops::BitAnd<Output = Self>
    + core::ops::BitXor<Output = Self>
    + core::ops::Not<Output = Self>
{
    const EMPTY: Self;
    const ALL: Self;
}

macro_rules! impl_bits {
    ($($t:ty),* $(,)?) => {
        $(
            impl Bits for $t {
                const EMPTY: Self = 0;
                const ALL: Self = <$t>::MAX;
            }
        )*
    };
}
impl_bits!(u8, u16, u32, u64, u128, usize);

/// A set of defined flags over an underlying [`Bits`] type — mirrors the real crate's `Flags`
/// trait. Every method except `bits`/`from_bits_retain`/`FLAGS` is a default derived from those
/// three plus set arithmetic on the underlying integer; the `bitflags!` macro only needs to
/// supply the three, matching how little is genuinely irreducible here.
pub trait Flags: Sized + 'static {
    const FLAGS: &'static [Flag<Self>];
    type Bits: Bits;

    fn bits(&self) -> Self::Bits;
    fn from_bits_retain(bits: Self::Bits) -> Self;

    fn empty() -> Self {
        Self::from_bits_retain(Self::Bits::EMPTY)
    }
    fn all() -> Self {
        let mut acc = Self::Bits::EMPTY;
        for f in Self::FLAGS {
            acc = acc | f.value().bits();
        }
        Self::from_bits_retain(acc)
    }
    fn from_bits(bits: Self::Bits) -> Option<Self> {
        let truncated = Self::from_bits_truncate(bits);
        if truncated.bits() == bits {
            Some(truncated)
        } else {
            None
        }
    }
    fn from_bits_truncate(bits: Self::Bits) -> Self {
        Self::from_bits_retain(bits & Self::all().bits())
    }
    fn from_name(name: &str) -> Option<Self> {
        if name.is_empty() {
            return None;
        }
        Self::FLAGS
            .iter()
            .find(|f| f.name() == name)
            .map(|f| Self::from_bits_retain(f.value().bits()))
    }
    fn is_empty(&self) -> bool {
        self.bits() == Self::Bits::EMPTY
    }
    fn is_all(&self) -> bool {
        Self::all().bits() | self.bits() == self.bits()
    }
    fn intersects(&self, other: Self) -> bool {
        self.bits() & other.bits() != Self::Bits::EMPTY
    }
    fn contains(&self, other: Self) -> bool {
        self.bits() & other.bits() == other.bits()
    }
    fn insert(&mut self, other: Self) {
        *self = Self::from_bits_retain(self.bits() | other.bits());
    }
    fn remove(&mut self, other: Self) {
        *self = Self::from_bits_retain(self.bits() & !other.bits());
    }
    fn toggle(&mut self, other: Self) {
        *self = Self::from_bits_retain(self.bits() ^ other.bits());
    }
    fn set(&mut self, other: Self, value: bool) {
        if value {
            self.insert(other);
        } else {
            self.remove(other);
        }
    }
    #[must_use]
    fn intersection(self, other: Self) -> Self {
        Self::from_bits_retain(self.bits() & other.bits())
    }
    #[must_use]
    fn union(self, other: Self) -> Self {
        Self::from_bits_retain(self.bits() | other.bits())
    }
    #[must_use]
    fn difference(self, other: Self) -> Self {
        Self::from_bits_retain(self.bits() & !other.bits())
    }
    #[must_use]
    fn symmetric_difference(self, other: Self) -> Self {
        Self::from_bits_retain(self.bits() ^ other.bits())
    }
    #[must_use]
    fn complement(self) -> Self {
        Self::from_bits_truncate(!self.bits())
    }
    fn iter(&self) -> Iter<Self> {
        Iter::new(Self::from_bits_retain(self.bits()))
    }
    fn iter_names(&self) -> IterNames<Self> {
        IterNames::new(Self::from_bits_retain(self.bits()))
    }
}

/// Yields each contained, defined, NAMED flag once (multi-bit flags that fully cover an
/// already-yielded flag are skipped — same semantics as the real crate's `IterNames`).
pub struct IterNames<B: Flags> {
    flags: &'static [Flag<B>],
    idx: usize,
    remaining: B,
}

impl<B: Flags> IterNames<B> {
    fn new(source: B) -> Self {
        IterNames {
            flags: B::FLAGS,
            idx: 0,
            remaining: source,
        }
    }
    pub fn remaining(&self) -> &B {
        &self.remaining
    }
}

impl<B: Flags> Iterator for IterNames<B> {
    type Item = (&'static str, B);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(flag) = self.flags.get(self.idx) {
            self.idx += 1;
            if self.remaining.is_empty() {
                return None;
            }
            if !flag.is_named() {
                continue;
            }
            // Reconstruct a fresh `B` from the raw (trivially-`Copy`) bits at each use site
            // rather than binding one `candidate` and reusing it — `contains`/`remove` take
            // `Self` by value, and `B` itself has no `Copy` bound (real `bitflags` does the same,
            // for the same reason: nothing here should force a `Copy`/`Clone` derive on callers).
            let bits = flag.value().bits();
            if self.remaining.contains(B::from_bits_retain(bits)) {
                self.remaining.remove(B::from_bits_retain(bits));
                return Some((flag.name(), B::from_bits_retain(bits)));
            }
        }
        None
    }
}

/// Yields each contained flags value: named flags first, then any leftover unnamed bits as one
/// final value — mirrors the real crate's `Iter` (its `into_iter`/`from_iter` round-trip).
pub struct Iter<B: Flags> {
    inner: IterNames<B>,
    done: bool,
}

impl<B: Flags> Iter<B> {
    fn new(source: B) -> Self {
        Iter {
            inner: IterNames::new(source),
            done: false,
        }
    }
}

impl<B: Flags> Iterator for Iter<B> {
    type Item = B;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some((_, flag)) => Some(flag),
            None if !self.done => {
                self.done = true;
                let remaining = self.inner.remaining();
                if remaining.is_empty() {
                    None
                } else {
                    Some(B::from_bits_retain(remaining.bits()))
                }
            }
            None => None,
        }
    }
}

/// Format a flags value as `"A | B | 0x4"` — named flags bar-separated, then any unnamed
/// remaining bits as one trailing hex number. Mirrors the real crate's `parser::to_writer`.
pub fn format_flags<B: Flags>(flags: &B) -> std::string::String
where
    B::Bits: std::fmt::LowerHex,
{
    use std::fmt::Write;
    use std::string::String;
    let mut out = String::new();
    let mut first = true;
    let mut iter = IterNames::new(B::from_bits_retain(flags.bits()));
    for (name, _) in &mut iter {
        if !first {
            out.push_str(" | ");
        }
        first = false;
        out.push_str(name);
    }
    let remaining = iter.remaining().bits();
    if remaining != B::Bits::EMPTY {
        if !first {
            out.push_str(" | ");
        }
        let _ = write!(out, "{remaining:#x}");
    }
    out
}

/// Declares a bitflags struct — see the module doc for a full example and what's ported.
#[macro_export]
macro_rules! bitflags {
    (
        $(#[$outer:meta])*
        $vis:vis struct $Name:ident: $T:ty {
            $(
                $(#[$inner:meta])*
                const $Flag:ident = $value:expr;
            )*
        }
    ) => {
        // Deliberately no baked-in `#[derive(..)]` here — same discipline the real crate uses:
        // any trait bakeD in here (Clone/Copy/PartialEq/Eq/Hash/...) would conflict with a
        // caller's own `#[derive(..)]` in `$(#[$outer])*` for the exact same trait (E0119,
        // verified). The `Flags` trait's default methods are written to never need `Self: Copy`
        // (they reconstruct via `from_bits_retain` from the raw, trivially-`Copy` integer instead
        // of duplicating a `Self` value) precisely so this macro doesn't have to choose for you.
        $(#[$outer])*
        $vis struct $Name($T);

        impl $Name {
            $(
                $(#[$inner])*
                #[allow(non_upper_case_globals)]
                pub const $Flag: Self = Self($value);
            )*

            /// A flags value with all bits unset.
            #[inline]
            pub const fn empty() -> Self {
                Self(0)
            }

            /// A flags value with all defined flags' bits set.
            #[inline]
            pub const fn all() -> Self {
                Self(0 $(| Self::$Flag.0)*)
            }

            /// The underlying bits.
            #[inline]
            pub const fn bits(&self) -> $T {
                self.0
            }

            /// Convert from bits exactly — `None` if any unknown bit is set.
            #[inline]
            pub const fn from_bits(bits: $T) -> ::core::option::Option<Self> {
                let truncated = Self::from_bits_truncate(bits).0;
                if truncated == bits {
                    ::core::option::Option::Some(Self(bits))
                } else {
                    ::core::option::Option::None
                }
            }

            /// Convert from bits, unsetting any unknown bits.
            #[inline]
            pub const fn from_bits_truncate(bits: $T) -> Self {
                Self(bits & Self::all().0)
            }

            /// Convert from bits exactly, keeping unknown bits.
            #[inline]
            pub const fn from_bits_retain(bits: $T) -> Self {
                Self(bits)
            }

            /// The flags value of the defined flag with this name, if any.
            pub fn from_name(name: &str) -> ::core::option::Option<Self> {
                <Self as $crate::bitflags::Flags>::from_name(name)
            }

            #[inline]
            pub const fn is_empty(&self) -> bool {
                self.0 == 0
            }
            #[inline]
            pub const fn is_all(&self) -> bool {
                Self::all().0 | self.0 == self.0
            }
            #[inline]
            pub const fn intersects(&self, other: Self) -> bool {
                self.0 & other.0 != 0
            }
            #[inline]
            pub const fn contains(&self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }
            #[inline]
            pub fn insert(&mut self, other: Self) {
                self.0 |= other.0;
            }
            #[inline]
            pub fn remove(&mut self, other: Self) {
                self.0 &= !other.0;
            }
            #[inline]
            pub fn toggle(&mut self, other: Self) {
                self.0 ^= other.0;
            }
            #[inline]
            pub fn set(&mut self, other: Self, value: bool) {
                if value {
                    self.insert(other);
                } else {
                    self.remove(other);
                }
            }
            #[inline]
            pub const fn intersection(self, other: Self) -> Self {
                Self(self.0 & other.0)
            }
            #[inline]
            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }
            #[inline]
            pub const fn difference(self, other: Self) -> Self {
                Self(self.0 & !other.0)
            }
            #[inline]
            pub const fn symmetric_difference(self, other: Self) -> Self {
                Self(self.0 ^ other.0)
            }
            #[inline]
            pub const fn complement(self) -> Self {
                Self::from_bits_truncate(!self.0)
            }

            /// Yield each contained flags value: defined flags first, then any leftover unknown
            /// bits as one final value.
            #[inline]
            pub fn iter(&self) -> $crate::bitflags::Iter<Self> {
                $crate::bitflags::Flags::iter(self)
            }
            /// Yield each contained, defined, NAMED flag.
            #[inline]
            pub fn iter_names(&self) -> $crate::bitflags::IterNames<Self> {
                $crate::bitflags::Flags::iter_names(self)
            }
        }

        impl $crate::bitflags::Flags for $Name {
            const FLAGS: &'static [$crate::bitflags::Flag<Self>] = &[
                $($crate::bitflags::Flag::new(stringify!($Flag), Self::$Flag),)*
            ];
            type Bits = $T;

            #[inline]
            fn bits(&self) -> $T {
                self.0
            }
            #[inline]
            fn from_bits_retain(bits: $T) -> Self {
                Self(bits)
            }
        }

        impl ::core::default::Default for $Name {
            #[inline]
            fn default() -> Self {
                Self::empty()
            }
        }

        impl ::core::ops::BitOr for $Name {
            type Output = Self;
            #[inline]
            fn bitor(self, other: Self) -> Self {
                self.union(other)
            }
        }
        impl ::core::ops::BitOrAssign for $Name {
            #[inline]
            fn bitor_assign(&mut self, other: Self) {
                self.insert(other);
            }
        }
        impl ::core::ops::BitAnd for $Name {
            type Output = Self;
            #[inline]
            fn bitand(self, other: Self) -> Self {
                self.intersection(other)
            }
        }
        impl ::core::ops::BitAndAssign for $Name {
            #[inline]
            fn bitand_assign(&mut self, other: Self) {
                // NOT `*self = self.intersection(other)` — `intersection` takes `self` by value,
                // and moving out of a `&mut Self` receiver needs `Copy` (E0507, verified) that
                // this type deliberately doesn't require. `self.0 &= other.0` mutates the
                // wrapped integer in place instead, exactly like `insert`/`remove`/`toggle` above.
                self.0 &= other.0;
            }
        }
        impl ::core::ops::BitXor for $Name {
            type Output = Self;
            #[inline]
            fn bitxor(self, other: Self) -> Self {
                self.symmetric_difference(other)
            }
        }
        impl ::core::ops::BitXorAssign for $Name {
            #[inline]
            fn bitxor_assign(&mut self, other: Self) {
                self.toggle(other);
            }
        }
        impl ::core::ops::Sub for $Name {
            type Output = Self;
            #[inline]
            fn sub(self, other: Self) -> Self {
                self.difference(other)
            }
        }
        impl ::core::ops::SubAssign for $Name {
            #[inline]
            fn sub_assign(&mut self, other: Self) {
                self.remove(other);
            }
        }
        impl ::core::ops::Not for $Name {
            type Output = Self;
            #[inline]
            fn not(self) -> Self {
                self.complement()
            }
        }

        impl ::core::iter::Extend<$Name> for $Name {
            fn extend<I: ::core::iter::IntoIterator<Item = Self>>(&mut self, iter: I) {
                for item in iter {
                    self.insert(item);
                }
            }
        }
        impl ::core::iter::FromIterator<$Name> for $Name {
            fn from_iter<I: ::core::iter::IntoIterator<Item = Self>>(iter: I) -> Self {
                let mut out = Self::empty();
                out.extend(iter);
                out
            }
        }
        impl ::core::iter::IntoIterator for $Name {
            type Item = Self;
            type IntoIter = $crate::bitflags::Iter<Self>;
            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }

        impl ::core::fmt::Debug for $Name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                if self.is_empty() {
                    ::core::write!(f, "{:#x}", 0)
                } else {
                    ::core::fmt::Display::fmt(self, f)
                }
            }
        }
        impl ::core::fmt::Display for $Name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::write!(f, "{}", $crate::bitflags::format_flags(self))
            }
        }

        #[cfg(feature = "ht-syntax")]
        impl $crate::codec::syntax::AsTerm for $Name {
            fn to_term(&self) -> $crate::codec::syntax::SyntaxTerm {
                $crate::codec::syntax::SyntaxTerm::B($crate::codec::syntax::Atom::Str(
                    $crate::bitflags::format_flags(self),
                ))
            }
            fn from_term(term: &$crate::codec::syntax::SyntaxTerm) -> ::core::result::Result<Self, $crate::codec::syntax::SyntaxError> {
                let s = $crate::codec::syntax::term_str(term)?;
                let mut out = Self::empty();
                let s = s.trim();
                if s.is_empty() {
                    return ::core::result::Result::Ok(out);
                }
                for part in s.split('|') {
                    let part = part.trim();
                    if let ::core::option::Option::Some(hex) = part.strip_prefix("0x") {
                        let bits = <$T>::from_str_radix(hex, 16)
                            .map_err(|_| $crate::codec::syntax::SyntaxError::new("bad hex flag"))?;
                        out.insert(Self::from_bits_retain(bits));
                    } else {
                        out.insert(Self::from_name(part).ok_or_else(|| {
                            $crate::codec::syntax::SyntaxError::new(::std::format!("unknown flag: {part}"))
                        })?);
                    }
                }
                ::core::result::Result::Ok(out)
            }
        }
    };
}
