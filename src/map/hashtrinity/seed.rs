//! The **coordinate seed** — storing a coordinate key in fewer bits than the full 64.
//!
//! A slot already tells you where a key lives: its home slot is the top `bits` of `c·PHI`. So a
//! coordinate key doesn't need to store the whole coordinate — only the **residual**, the low
//! `64 − bits` of `c·PHI` that the slot index doesn't already pin down. Because [`PHI`] is odd,
//! `c ↦ c·PHI` is a bijection, so the split `c·PHI = (home | residual)` loses nothing and the
//! coordinate — hence the key ([`CoordInv::from_coord`]) — regrows exactly from `(home, residual)`.
//!
//! At a table with `n` keys and `2^bits` slots (load `α = n/2^bits`), the fraction of keys sitting
//! at their own home slot — the ones this compact form fits — is about `(1 − (1 − 1/2^bits)^n)/α`
//! (≈ 0.82 at α ≈ 0.4); the rest, displaced by collisions, fall back to the full coordinate.
//!
//! [`CoordSeed::home`] matches the [`Table`](crate::map::hashtrinity::Table)'s own home slot, so the seed's split is
//! the table's addressing, not a separate scheme.

use crate::map::hashtrinity::{Coord, CoordInv, PHI};

/// The multiplicative inverse of [`PHI`] in ℤ/2⁶⁴ — `PHI` is odd, hence a unit, so `unfold ∘ fold`
/// is the identity. Hensel lift: a 5-bit-correct seed (`3x ^ 2`) whose accuracy doubles per Newton
/// step (5 → 10 → 20 → 40 → 80 ≥ 64), so four steps suffice. Checked `PHI * PHI_INV == 1` in tests.
pub const PHI_INV: u64 = {
    let x = PHI;
    let mut inv = x.wrapping_mul(3) ^ 2;
    inv = inv.wrapping_mul(2u64.wrapping_sub(x.wrapping_mul(inv)));
    inv = inv.wrapping_mul(2u64.wrapping_sub(x.wrapping_mul(inv)));
    inv = inv.wrapping_mul(2u64.wrapping_sub(x.wrapping_mul(inv)));
    inv = inv.wrapping_mul(2u64.wrapping_sub(x.wrapping_mul(inv)));
    inv
};

/// The coordinate fold `c ↦ c·PHI` — a bijection of 2⁶⁴ (PHI is odd). `home` reads its top bits; the
/// seed keeps the rest.
#[inline]
pub fn fold(c: u64) -> u64 {
    c.wrapping_mul(PHI)
}

/// The inverse fold `p ↦ p·PHI⁻¹` — recovers the coordinate from the folded word.
/// `unfold(fold(c)) == c`.
#[inline]
pub fn unfold(p: u64) -> u64 {
    p.wrapping_mul(PHI_INV)
}

/// Stores a coordinate key at slot granularity `bits`: the residual, and regrows the key from it.
///
/// `fold(c)` splits into the top `bits` ([`home`](Self::home), the slot) and the low `64 − bits`
/// ([`residual`](Self::residual), what's stored). The split is lossless, so `(home, residual)`
/// reconstruct `fold(c)`, hence `c` ([`unfold`]), hence the key ([`CoordInv::from_coord`]) — the
/// readout carries `64 − bits` bits instead of the whole coordinate.
///
/// The residual width `64 − bits` is also the false-positive budget: two coordinates with the same
/// residual are 1-in-`2^(64−bits)`. It's exact, not a guess — the residual always fits in `64 − bits`
/// bits. Only home-resident keys use this compact form; collision-displaced keys keep the full
/// coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoordSeed {
    /// The home-window width in bits: `home = fold(c) >> (64 - bits)`, the slot index. The table has
    /// `2^bits` slots, matching [`Table`](crate::map::hashtrinity::Table)'s `bits`.
    pub bits: u32,
}

impl CoordSeed {
    /// A coordinate seed whose home window is `bits` wide (the table has `2^bits` slots).
    #[inline]
    pub fn new(bits: u32) -> Self {
        // `bits >= 1 && bits <= 64`, not `(1..=64).contains(&bits)`: the range's `contains` is an
        // unmodeled external in the extraction, and the two tests are the same predicate.
        debug_assert!(bits >= 1 && bits <= 64, "bits must be in 1..=64");
        CoordSeed { bits }
    }

    /// The home slot of `c`: the top `bits` of `fold(c)` — the same slot the table addresses it to.
    #[inline]
    pub fn home(&self, c: u64) -> u64 {
        fold(c) >> (64 - self.bits)
    }

    /// The residual (the stored part): the low `64 − bits` of `fold(c)` — everything the slot index
    /// doesn't already carry. Always `< 2^(64−bits)`.
    #[inline]
    pub fn residual(&self, c: u64) -> u64 {
        fold(c) & ((1u64 << (64 - self.bits)) - 1)
    }

    /// Regrow the coordinate from `(home, residual)`: reassemble `fold(c)` (home and residual are
    /// disjoint bit ranges, so `|` = `+`) and [`unfold`]. `regrow_coord(home c, residual c) == c`.
    #[inline]
    pub fn regrow_coord(&self, home: u64, residual: u64) -> u64 {
        unfold((home << (64 - self.bits)) | residual)
    }

    /// Regrow the key from `(home, residual)`: regrow the coordinate, then [`CoordInv::from_coord`].
    #[inline]
    pub fn regrow<K: CoordInv>(&self, home: u64, residual: u64) -> K {
        K::from_coord(self.regrow_coord(home, residual))
    }

    /// The seed of a key: `(home, residual)` — what a coordinate readout stores per slot instead of
    /// the whole coordinate. `regrow(seed_of(k)) == k` for every [`CoordInv`] key.
    #[inline]
    pub fn seed_of<K: Coord>(&self, k: &K) -> (u64, u64) {
        let c = k.coord();
        (self.home(c), self.residual(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_is_a_unit() {
        // The Hensel-lifted inverse: PHI · PHI_INV ≡ 1 (mod 2^64), so the fold is invertible.
        assert_eq!(
            PHI.wrapping_mul(PHI_INV),
            1,
            "PHI_INV is not PHI's inverse mod 2^64"
        );
    }

    #[test]
    fn fold_unfold_round_trips() {
        for c in [0u64, 1, 2, 42, u64::MAX, PHI, 12345678901234567890] {
            assert_eq!(unfold(fold(c)), c, "unfold∘fold lost {c}");
        }
    }

    /// Slot + residual regrow the key, and the residual fits in `64 − bits`, at every granularity.
    #[test]
    fn seed_regrows_the_key() {
        for bits in [3u32, 8, 17, 18, 21, 32, 63] {
            let s = CoordSeed::new(bits);
            for k in [0u64, 1, 7, 100_000, u64::MAX, 0xDEAD_BEEF_CAFE_F00D] {
                let (home, residual) = s.seed_of(&k);
                assert!(home < (1u64 << bits), "bits={bits}: home out of window");
                assert!(
                    residual < (1u64 << (64 - bits)),
                    "bits={bits}: residual wider than seed"
                );
                let regrown: u64 = s.regrow(home, residual);
                assert_eq!(regrown, k, "bits={bits}: seed lost key {k}");
            }
        }
        // signed / narrower CoordInv keys regrow the same way.
        let s = CoordSeed::new(18);
        for k in [-1i32, 0, 1, 42, i32::MIN, i32::MAX] {
            let (home, residual) = s.seed_of(&k);
            let regrown: i32 = s.regrow(home, residual);
            assert_eq!(regrown, k, "signed key {k} lost");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The Gen stream codec — absorbed from seedfold (the storage face's pure core):
// predictor + XOR residual, lossless for EVERY predictor ().
// ─────────────────────────────────────────────────────────────────────────────

/// The stream word — u64, the coordinate word (XOR = the discrete self-inverse /).
pub type Word = u64;

pub trait Gen {
    /// The generator's context — the running genome state threaded through the stream.
    type Ctx: Clone;

    /// The genome the stream regrows from (`Gen.init`).
    fn init(&self) -> Self::Ctx;
    /// The prediction at the current context (`Gen.predict : Ctx → Word`).
    fn predict(&self, ctx: &Self::Ctx) -> Word;
    /// Advance from the prediction and the actual value (`Gen.step : Ctx → Word → Word → Ctx`).
    fn step(&self, ctx: &Self::Ctx, predicted: Word, actual: Word) -> Self::Ctx;

    /// Emit residual `actual ^ predict` (the boundary), advance (`genEncode g ctx t`).
    fn gen_encode(&self, ctx: Self::Ctx, t: &[Word]) -> Vec<Word> {
        let mut ctx = ctx;
        let mut out = Vec::with_capacity(t.len());
        for &a in t {
            let p = self.predict(&ctx);
            out.push(a ^ p); // the residual — the Null arm U, the stored boundary
            ctx = self.step(&ctx, p, a);
        }
        out
    }

    /// Recover `actual = residual ^ predict`, emit, advance (`genDecode g ctx r`).
    fn gen_decode(&self, ctx: Self::Ctx, rs: &[Word]) -> Vec<Word> {
        let mut ctx = ctx;
        let mut out = Vec::with_capacity(rs.len());
        for &x in rs {
            let p = self.predict(&ctx);
            let a = x ^ p; // XOR self-inverse ⇒ undoes gen_encode
            out.push(a);
            ctx = self.step(&ctx, p, a);
        }
        out
    }

    /// Encode a tensor from the genome (`Gen.encode g t = genEncode g g.init t`).
    fn encode(&self, t: &[Word]) -> Vec<Word> {
        self.gen_encode(self.init(), t)
    }

    /// Decode a residual stream back to the tensor (`Gen.decode`). **Headline** (`Gen.decode_encode`):
    /// `g.decode(g.encode(t)) == t` for every `g`, `t` — store the boundary, regrow the bulk.
    fn decode(&self, rs: &[Word]) -> Vec<Word> {
        self.gen_decode(self.init(), rs)
    }
}

/// The **order-1 generator**: predict the previous value (`deltaGen`). A correlated stream codes to
/// near-zero residual — the seed is the genome plus almost nothing.
pub struct DeltaGen;

impl Gen for DeltaGen {
    type Ctx = Word;
    fn init(&self) -> Word {
        0
    }
    fn predict(&self, ctx: &Word) -> Word {
        *ctx
    }
    fn step(&self, _ctx: &Word, _predicted: Word, actual: Word) -> Word {
        actual
    }
}
