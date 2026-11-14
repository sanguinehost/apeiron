//! The `NanoQuant` sign-rank product in Q32.32:
//! `y = s_post ⊙ U (s_mid ⊙ (V (s_pre ⊙ x))) + Σ_j salient[·, j]·x[idx[j]] + Σ_l s_res_l ⊙ (sign_l x)`.
//!
//! [`NqLinear::project`] is the CPU kernel the packed linears run. `x`, `t = s_pre ⊙ x`, `z = V t`
//! and `y` are fixed-point vectors; `U`, `V` and every residual plane are sign matrices stored as
//! bits, so each product is a sum of add/subtracts and never a multiply. The only rounding is the
//! documented Q32.32 epilogues: one [`mul_q32`] per pre-scaled activation, one per mid-scaled
//! rank, and one on the sum of the post-scaled output, the sparse correction and every dense
//! residual. A factor or output that leaves `i64` refuses with [`NqError::Overflow`].
//!
//! **Accumulator width.** A sum of `n` terms each at most `max` in magnitude is at most `n·max`,
//! checked before the pass. When that bound fits `i64` (every real activation), the sums run in
//! `i64`; otherwise the same sign-dot source runs in `i128` so cancellation still decides the
//! result instead of a conservative refusal. The bound is a guard, not a second kernel.
//!
//! **One face.** This is not a second forward and not a dtype-specific copy: the one dispatch
//! point `Engine::step_proj_sched` selects it for a tensor the pack replaces, exactly as it
//! selects the payload kernels for a stored tensor. A fitted tensor has no GGUF row, so the
//! kernel mode chooses nothing here; the sign product is the weight. The pack reader decodes the
//! three scale vectors and every residual row scale once ([`PackedLinear`], `bf16_to_fixed`) and
//! the sparse correction with the `f16_to_fixed` the other `F16` rows use.
//!
//! **Bit convention.** Bit `1` in `U`/`V` is `+1`, bit `0` is `-1` (the exporter's convention; the
//! dense reference words of `nq-ref7` reproduce with it). The trailing bits of the last word are
//! padding and are ignored. [`sign_dot!`] maps bit `1` to `-t`; `U` and `V` are always read as a
//! product, so the shared inversion cancels, while a residual plane is read once and its dot is
//! negated.
//!
//! **Parallelism.** The `V` rows split across the engine's persistent pool: task `k` owns one
//! `z_k`, so the `rank × inn` adds are shared. The `U` rows then split the same way over output
//! rows. Both dispatches write disjoint chunks through [`Pool::chunks_mut`]; the caller is the
//! step's main thread, never a pool task.
//!
//! **Cost.** One projection is `O(rank·inn + out·rank + res_layers·out·inn)` additions,
//! `O(out · inn)` weights replaced, `O(rank)` scratch for `z`, `O(inn)` for `t` and `O(out)`
//! output.

#![allow(
    clippy::many_single_char_names,
    reason = "`x`, `t`, `z`, `y`, `U`, `V` are the derivation's names; the section counters are short"
)]

use crate::engine::chart::pool::Pool;
use crate::engine::num::dequant::{bf16_to_fixed, f16_to_fixed};
use crate::engine::num::value::mul_q32;
use crate::format::nq::{PackedLinear, Plane};

/// A refused sign-rank product or factor decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NqError {
    /// The activation width is not the linear's input width.
    Shape {
        /// Activation length.
        x: usize,
        /// Input width.
        n_in: usize,
    },
    /// A packed factor's field lengths disagree with its dimensions.
    Packed {
        /// Which field.
        field: &'static str,
        /// Length found.
        have: usize,
        /// Length the dimensions require.
        expected: usize,
    },
    /// A 16-bit scale (bfloat16 or binary16) is not finite or leaves the Q32.32 grid.
    Scale {
        /// The raw binary16 bit pattern.
        bits: u16,
    },
    /// A salient index is outside the input width or repeats an earlier index.
    SalientIndex {
        /// Index found.
        index: usize,
        /// Input width.
        inn: usize,
    },
    /// A checked fixed-point product, sum or narrowing left its range.
    Overflow,
}

impl core::fmt::Display for NqError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NqError::Shape { x, n_in } => {
                write!(f, "nq projection: activation {x}, want {n_in}")
            }
            NqError::Packed {
                field,
                have,
                expected,
            } => write!(f, "nq factor: {field} has {have}, want {expected}"),
            NqError::Scale { bits } => write!(f, "nq scale bits {bits:#06x}"),
            NqError::SalientIndex { index, inn } => {
                write!(f, "nq salient index {index} outside {inn}")
            }
            NqError::Overflow => write!(f, "nq projection: fixed-point overflow"),
        }
    }
}

impl std::error::Error for NqError {}

/// The sign dot `Σ (2·bit − 1)·t[i]` of one packed row, written once for both accumulator widths.
///
/// `$acc` is `i64` when the caller's `n·max|t|` bound fits and `i128` otherwise; the bit mask is
/// all ones for `1` and zero for `0`, so `(t ^ mask) - mask` is `t` or `-t` without a branch.
macro_rules! sign_dot {
    ($t:expr, $words:expr, $len:expr, $acc:ty) => {{
        let mut acc: $acc = 0;
        let mut i = 0usize;
        for &wordbits in $words {
            let mut bits = wordbits;
            let end = (i + 32).min($len);
            while i < end {
                let mask = -<$acc>::from(bits & 1);
                acc += ($t[i] as $acc ^ mask) - mask;
                bits >>= 1;
                i += 1;
            }
        }
        acc
    }};
}

/// One fitted linear in the form the kernel executes: bits, Q32.32 scales, the sparse residual and
/// the dense sign-residual layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NqLinear {
    out: usize,
    inn: usize,
    rank: usize,
    u_words: usize,
    v_words: usize,
    k: usize,
    u: Plane<u32>,
    v: Plane<u32>,
    s_pre: Box<[i64]>,
    s_mid: Box<[i64]>,
    s_post: Box<[i64]>,
    salient_idx: Box<[u32]>,
    salient: Box<[i64]>,
    res_layers: usize,
    res_words: usize,
    res: Plane<u32>,
    s_res: Box<[i64]>,
}

impl NqLinear {
    /// Decode a packed linear: every scale word through [`bf16_to_fixed`], every sign word copied.
    ///
    /// # Errors
    ///
    /// [`NqError::Packed`] for a field whose length disagrees with its dimensions, and
    /// [`NqError::Scale`] for a binary16 scale that is not finite or does not fit Q32.32.
    pub fn from_packed(p: PackedLinear) -> Result<Self, NqError> {
        let shape = checked_shape(&p)?;
        let mut s_pre = Vec::with_capacity(p.inn);
        for &bits in p.s_pre.iter() {
            s_pre.push(bf16_to_fixed(bits).map_err(|_| NqError::Scale { bits })?);
        }
        let mut s_mid = Vec::with_capacity(p.rank);
        for &bits in p.s_mid.iter() {
            s_mid.push(bf16_to_fixed(bits).map_err(|_| NqError::Scale { bits })?);
        }
        let mut s_post = Vec::with_capacity(p.out);
        for &bits in p.s_post.iter() {
            s_post.push(bf16_to_fixed(bits).map_err(|_| NqError::Scale { bits })?);
        }
        let mut salient = Vec::with_capacity(shape.salient_len);
        for &bits in p.salient.iter() {
            salient.push(f16_to_fixed(bits).map_err(|_| NqError::Scale { bits })?);
        }
        let mut s_res = Vec::with_capacity(p.s_res.len());
        for &bits in p.s_res.iter() {
            s_res.push(bf16_to_fixed(bits).map_err(|_| NqError::Scale { bits })?);
        }
        let mut seen = vec![false; p.inn];
        for &j in p.salient_idx.iter() {
            let j = j as usize;
            if j >= p.inn || seen[j] {
                return Err(NqError::SalientIndex {
                    index: j,
                    inn: p.inn,
                });
            }
            seen[j] = true;
        }
        Ok(Self {
            out: p.out,
            inn: p.inn,
            rank: p.rank,
            u_words: shape.u_words,
            v_words: shape.v_words,
            k: shape.k,
            u: p.u,
            v: p.v,
            s_pre: s_pre.into_boxed_slice(),
            s_mid: s_mid.into_boxed_slice(),
            s_post: s_post.into_boxed_slice(),
            salient_idx: p.salient_idx.to_vec().into_boxed_slice(),
            salient: salient.into_boxed_slice(),
            res_layers: shape.res_layers,
            res_words: shape.res_words,
            res: p.res,
            s_res: s_res.into_boxed_slice(),
        })
    }

    /// Output rows.
    #[must_use]
    pub fn out(&self) -> usize {
        self.out
    }

    /// Input columns.
    #[must_use]
    pub fn inn(&self) -> usize {
        self.inn
    }

    /// The dense weights the linear replaces.
    #[must_use]
    pub fn weights(&self) -> u64 {
        (self.out as u64).saturating_mul(self.inn as u64)
    }

    /// The stored bits: the two sign planes, the three scale vectors, the index list, the sparse
    /// residual and every dense sign-residual layer.
    #[must_use]
    pub fn stored_bits(&self) -> u64 {
        let signs = self
            .u
            .len()
            .saturating_add(self.v.len())
            .saturating_add(self.res.len());
        let scales = self
            .s_pre
            .len()
            .saturating_add(self.s_mid.len())
            .saturating_add(self.s_post.len())
            .saturating_add(self.salient.len())
            .saturating_add(self.s_res.len());
        (signs as u64)
            .saturating_mul(32)
            .saturating_add((self.salient_idx.len() as u64).saturating_mul(32))
            .saturating_add((scales as u64).saturating_mul(16))
    }

    /// `round(s_post_i · Σ_k U_ik · z_k / 2^32)` plus `Σ_j salient_ij·x_idx[j]` and
    /// `Σ_l s_res_li·Σ_j sign_lij·x_j`, for every output row, with
    /// `z = s_mid ⊙ (V (s_pre ⊙ x))`.
    ///
    /// # Errors
    ///
    /// [`NqError::Shape`] for an activation of the wrong width and [`NqError::Overflow`] when a
    /// pre-scaled activation, a mid-scaled rank, a rank sum, a dense residual or a post-scaled
    /// output leaves `i64`. No value is clamped.
    ///
    /// # Complexity
    ///
    /// `O(rank·inn + out·rank + out·k + res_layers·out·inn)` additions and products; scratch
    /// `O(inn + rank)` words and `O(out)` output.
    pub fn project(&self, x: &[i64]) -> Result<Vec<i64>, NqError> {
        if x.len() != self.inn {
            return Err(NqError::Shape {
                x: x.len(),
                n_in: self.inn,
            });
        }
        // t_j = round(s_pre_j · x_j / 2^32).
        let mut t: Vec<i64> = Vec::with_capacity(self.inn);
        let mut max_abs_t = 0u64;
        let mut max_abs_x = 0u64;
        let mut j = 0usize;
        while j < self.inn {
            max_abs_x = max_abs_x.max(x[j].unsigned_abs());
            match mul_q32(x[j], self.s_pre[j]) {
                Some(v) => {
                    max_abs_t = max_abs_t.max(v.unsigned_abs());
                    t.push(v);
                }
                None => return Err(NqError::Overflow),
            }
            j += 1;
        }
        // Every `V` row sums `inn` terms of at most `max_abs_t`, so `inn·max_abs_t` bounds it.
        // When the bound fits `i64`, every partial sum does too and the narrow source is exact. The
        // product saturates because the factors can leave `u128`; a saturated bound is simply not
        // narrow and takes the wide source.
        let bound_v = u128::from(max_abs_t).saturating_mul(self.inn as u128);
        let narrow_v = bound_v <= i64::MAX as u128;
        let pool = Pool::global();
        let width = pool.width().max(1);
        let mut z: Vec<i64> = vec![0; self.rank];
        let z_chunk = self.rank.div_ceil(width.saturating_mul(4)).max(1);
        let overflow = std::sync::atomic::AtomicBool::new(false);
        pool.chunks_mut(&mut z, z_chunk, &|_, at, zs| {
            let mut d = 0usize;
            while d < zs.len() {
                let k = at + d;
                let vrow = &self.v[k * self.v_words..(k + 1) * self.v_words];
                let dot = if narrow_v {
                    sign_dot!(t, vrow, self.inn, i64)
                } else if let Ok(v) = i64::try_from(sign_dot!(t, vrow, self.inn, i128)) {
                    v
                } else {
                    overflow.store(true, std::sync::atomic::Ordering::Relaxed);
                    0
                };
                match mul_q32(dot, self.s_mid[k]) {
                    Some(v) => zs[d] = v,
                    None => overflow.store(true, std::sync::atomic::Ordering::Relaxed),
                }
                d += 1;
            }
        });
        if overflow.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(NqError::Overflow);
        }
        // The mid scale changes the bound: every `U` row sums `rank` terms of at most
        // `max_abs_z`, so `rank·max_abs_z` bounds it. The serial scan is `O(rank)` against the
        // `O(out·rank)` product that follows.
        let mut max_abs_z = 0u64;
        for &v in &z {
            max_abs_z = max_abs_z.max(v.unsigned_abs());
        }
        let bound_u = u128::from(max_abs_z).saturating_mul(self.rank as u128);
        let narrow_u = bound_u <= i64::MAX as u128;
        // The dense sign dot sums `inn` activation words; it is narrow exactly when that bound is.
        let bound_x = u128::from(max_abs_x).saturating_mul(self.inn as u128);
        let narrow_x = bound_x <= i64::MAX as u128;
        let y = self.project_rows(pool, &z, x, narrow_u, narrow_x, &overflow);
        if overflow.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(NqError::Overflow);
        }
        Ok(y)
    }

    /// The `U` stage of [`NqLinear::project`]: `round(s_post_i·Σ_k U_ik·z_k / 2^32)` plus the
    /// sparse residual, one row per output element, parallel over `pool` chunks.
    ///
    /// A refusal sets `overflow` and moves on rather than unwinding, because the body runs on pool
    /// workers; the first stage shares the flag, so the caller reads it once after both stages.
    fn project_rows(
        &self,
        pool: &Pool,
        z: &[i64],
        x: &[i64],
        narrow_u: bool,
        narrow_x: bool,
        overflow: &std::sync::atomic::AtomicBool,
    ) -> Vec<i64> {
        let width = pool.width().max(1);
        let mut y: Vec<i64> = vec![0; self.out];
        let y_chunk = self.out.div_ceil(width.saturating_mul(4)).max(1);
        pool.chunks_mut(&mut y, y_chunk, &|_, at, ys| {
            let mut d = 0usize;
            while d < ys.len() {
                let i = at + d;
                let urow = &self.u[i * self.u_words..(i + 1) * self.u_words];
                // A wide `U` sum is not a refusal by itself: the post-scale can bring it back
                // into `i64`, so only the scaled output is checked.
                let acc = if narrow_u {
                    i128::from(sign_dot!(z, urow, self.rank, i64))
                } else {
                    sign_dot!(z, urow, self.rank, i128)
                };
                let Some(low) = acc.checked_mul(i128::from(self.s_post[i])) else {
                    overflow.store(true, std::sync::atomic::Ordering::Relaxed);
                    d += 1;
                    continue;
                };
                // The sparse residual lives in the weight's units: `Σ_j w_ij·x_idx[j]` is a
                // Q64.32 sum like `low`, and one rounding decides the output. Both the term
                // products and the sums are checked: an extreme factor refuses instead of
                // wrapping in release.
                let mut corr: i128 = 0;
                let mut c = 0usize;
                while c < self.k {
                    let src = self.salient_idx[c] as usize;
                    let term =
                        i128::from(self.salient[i * self.k + c]).checked_mul(i128::from(x[src]));
                    if let Some(v) = term.and_then(|t| corr.checked_add(t)) {
                        corr = v;
                    } else {
                        overflow.store(true, std::sync::atomic::Ordering::Relaxed);
                        break;
                    }
                    c += 1;
                }
                // The dense sign residual: `Σ_l s_res[l,i] · Σ_j sign_lij·x_j` is a Q64.32 sum
                // like `corr`, so the row keeps one rounding after every term is added.
                let mut res: i128 = 0;
                let mut l = 0usize;
                while l < self.res_layers {
                    let at = (l * self.out + i) * self.res_words;
                    let srow = &self.res[at..at + self.res_words];
                    // The bits in `U` and `V` are consumed as a product, so their shared
                    // inversion cancels; a residual plane is read once, and the negated
                    // `sign_dot!` restores the format's bit-1-is-`+1` convention.
                    let dot = if narrow_x {
                        i128::from(-sign_dot!(x, srow, self.inn, i64))
                    } else {
                        -sign_dot!(x, srow, self.inn, i128)
                    };
                    let term = dot.checked_mul(i128::from(self.s_res[l * self.out + i]));
                    let Some(v) = term.and_then(|t| res.checked_add(t)) else {
                        overflow.store(true, std::sync::atomic::Ordering::Relaxed);
                        break;
                    };
                    res = v;
                    l += 1;
                }
                if l < self.res_layers {
                    d += 1;
                    continue;
                }
                if let Some(v) = low
                    .checked_add(corr)
                    .and_then(|v| v.checked_add(res))
                    .and_then(|v| v.checked_add(1_i128 << 31))
                    .and_then(|v| i64::try_from(v >> 32).ok())
                {
                    ys[d] = v;
                } else {
                    overflow.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                d += 1;
            }
        });
        y
    }
}

/// The decoded geometry of one packed linear: strides, sparse columns and residual planes.
struct Shape {
    u_words: usize,
    v_words: usize,
    salient_len: usize,
    k: usize,
    res_words: usize,
    res_layers: usize,
}

/// Validate a [`PackedLinear`]'s dimensions and section lengths.
///
/// Returns the U/V strides, the sparse residual's row count and column count, and the dense
/// residual's layer count and row stride, all checked against the declared `out`, `inn` and
/// `rank`.
fn checked_shape(p: &PackedLinear) -> Result<Shape, NqError> {
    if p.out == 0 || p.inn == 0 || p.rank == 0 {
        return Err(NqError::Packed {
            field: "dimensions",
            have: 0,
            expected: 1,
        });
    }
    let u_words = p.rank.div_ceil(32);
    let v_words = p.inn.div_ceil(32);
    let u_len = p.out.checked_mul(u_words).ok_or(NqError::Overflow)?;
    let v_len = p.rank.checked_mul(v_words).ok_or(NqError::Overflow)?;
    if p.u.len() != u_len {
        return Err(NqError::Packed {
            field: "U",
            have: p.u.len(),
            expected: u_len,
        });
    }
    if p.v.len() != v_len {
        return Err(NqError::Packed {
            field: "V",
            have: p.v.len(),
            expected: v_len,
        });
    }
    if p.s_pre.len() != p.inn {
        return Err(NqError::Packed {
            field: "s_pre",
            have: p.s_pre.len(),
            expected: p.inn,
        });
    }
    if p.s_mid.len() != p.rank {
        return Err(NqError::Packed {
            field: "s_mid",
            have: p.s_mid.len(),
            expected: p.rank,
        });
    }
    if p.s_post.len() != p.out {
        return Err(NqError::Packed {
            field: "s_post",
            have: p.s_post.len(),
            expected: p.out,
        });
    }
    let k = p.salient_idx.len();
    if k > p.inn {
        return Err(NqError::Packed {
            field: "salient_idx",
            have: k,
            expected: p.inn,
        });
    }
    let salient_len = p.out.checked_mul(k).ok_or(NqError::Overflow)?;
    if p.salient.len() != salient_len {
        return Err(NqError::Packed {
            field: "salient",
            have: p.salient.len(),
            expected: salient_len,
        });
    }
    let res_words = p.inn.div_ceil(32);
    let res_len = p
        .res_layers
        .checked_mul(p.out)
        .and_then(|x| x.checked_mul(res_words))
        .ok_or(NqError::Overflow)?;
    if p.res.len() != res_len {
        return Err(NqError::Packed {
            field: "res",
            have: p.res.len(),
            expected: res_len,
        });
    }
    let s_res_len = p.res_layers.checked_mul(p.out).ok_or(NqError::Overflow)?;
    if p.s_res.len() != s_res_len {
        return Err(NqError::Packed {
            field: "s_res",
            have: p.s_res.len(),
            expected: s_res_len,
        });
    }
    Ok(Shape {
        u_words,
        v_words,
        salient_len,
        k,
        res_words,
        res_layers: p.res_layers,
    })
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "test-only small indices and the staged oracle index their vectors as written"
)]
mod tests {
    use super::*;
    use crate::engine::num::value::round_shift_32;
    use crate::format::nq::PackedLinear;

    /// A deterministic xorshift word source.
    fn next_u32(state: &mut u64) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state as u32
    }

    /// `f32` to binary16 bits, round-to-nearest (test-only; scales only).
    fn bf16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let round = 0x7fff + ((bits >> 16) & 1);
        ((bits.wrapping_add(round)) >> 16) as u16
    }

    fn packed(reps: u64) -> PackedLinear {
        let (out, inn, rank) = (37usize, 29usize, 23usize);
        let u_words = rank.div_ceil(32);
        let v_words = inn.div_ceil(32);
        let mut state = 0x1234_5678_9abc_def0u64 ^ reps;
        let mut u = Vec::with_capacity(out * u_words);
        let mut v = Vec::with_capacity(rank * v_words);
        for _ in 0..out * u_words {
            u.push(next_u32(&mut state));
        }
        for _ in 0..rank * v_words {
            v.push(next_u32(&mut state));
        }
        let res_layers = 1usize;
        let res_words = inn.div_ceil(32);
        let mut planes = Vec::with_capacity(res_layers * out * res_words);
        for _ in 0..res_layers * out * res_words {
            planes.push(next_u32(&mut state));
        }
        PackedLinear {
            layer: 0,
            kind: 0,
            out,
            inn,
            rank,
            u: u.into(),
            v: v.into(),
            s_pre: (0..inn)
                .map(|i| bf16_bits(0.01 + 0.001 * (i % 7) as f32))
                .collect::<Vec<u16>>()
                .into(),
            s_mid: (0..rank)
                .map(|i| bf16_bits(0.5 + 0.01 * (i % 3) as f32))
                .collect::<Vec<u16>>()
                .into(),
            s_post: (0..out)
                .map(|i| bf16_bits(0.02 + 0.002 * (i % 5) as f32))
                .collect::<Vec<u16>>()
                .into(),
            salient_idx: vec![1, 7].into(),
            salient: (0..out * 2)
                .map(|i| 0x2400u16 + (i % 9) as u16)
                .collect::<Vec<u16>>()
                .into(),
            res_layers,
            res: planes.into(),
            s_res: (0..res_layers * out)
                .map(|i| bf16_bits(0.0005 + 0.0001 * (i % 7) as f32))
                .collect::<Vec<u16>>()
                .into(),
        }
    }

    /// The staged product written out one term at a time: the kernel must equal it word for word.
    fn sequential(p: &PackedLinear, x: &[i64]) -> Vec<i64> {
        let s_pre: Vec<i64> = p
            .s_pre
            .iter()
            .map(|&b| bf16_to_fixed(b).expect("scale"))
            .collect();
        let s_mid: Vec<i64> = p
            .s_mid
            .iter()
            .map(|&b| bf16_to_fixed(b).expect("scale"))
            .collect();
        let s_post: Vec<i64> = p
            .s_post
            .iter()
            .map(|&b| bf16_to_fixed(b).expect("scale"))
            .collect();
        let salient: Vec<i64> = p
            .salient
            .iter()
            .map(|&b| f16_to_fixed(b).expect("salient"))
            .collect();
        let s_res: Vec<i64> = p
            .s_res
            .iter()
            .map(|&b| bf16_to_fixed(b).expect("residual scale"))
            .collect();
        let t: Vec<i64> = (0..p.inn)
            .map(|j| mul_q32(x[j], s_pre[j]).expect("t"))
            .collect();
        let v_words = p.inn.div_ceil(32);
        let z: Vec<i64> = (0..p.rank)
            .map(|k| {
                let mut acc: i128 = 0;
                for j in 0..p.inn {
                    let bit = (p.v[k * v_words + j / 32] >> (j % 32)) & 1;
                    acc += if bit == 1 {
                        i128::from(t[j])
                    } else {
                        -i128::from(t[j])
                    };
                }
                mul_q32(i64::try_from(acc).expect("z"), s_mid[k]).expect("mid")
            })
            .collect();
        let u_words = p.rank.div_ceil(32);
        (0..p.out)
            .map(|i| {
                let mut acc: i128 = 0;
                for k in 0..p.rank {
                    let bit = (p.u[i * u_words + k / 32] >> (k % 32)) & 1;
                    acc += if bit == 1 {
                        i128::from(z[k])
                    } else {
                        -i128::from(z[k])
                    };
                }
                let mut corr: i128 = 0;
                for j in 0..p.salient_idx.len() {
                    corr += i128::from(salient[i * p.salient_idx.len() + j])
                        * i128::from(x[p.salient_idx[j] as usize]);
                }
                let mut res: i128 = 0;
                for l in 0..p.res_layers {
                    let mut acc = 0i128;
                    for j in 0..p.inn {
                        let bit = (p.res[(l * p.out + i) * p.res_words() + j / 32] >> (j % 32)) & 1;
                        acc += if bit == 1 {
                            i128::from(x[j])
                        } else {
                            -i128::from(x[j])
                        };
                    }
                    res += acc * i128::from(s_res[l * p.out + i]);
                }
                i64::try_from(round_shift_32(acc * i128::from(s_post[i]) + corr + res)).expect("y")
            })
            .collect()
    }

    #[test]
    fn the_kernel_equals_the_sequential_staged_product() {
        let p = packed(1);
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        let x: Vec<i64> = (0..p.inn)
            .map(|i| ((i as i64 * 37 + 11) % 101 - 50) << 24)
            .collect();
        assert_eq!(linear.project(&x).expect("project"), sequential(&p, &x));
    }

    #[test]
    fn the_wide_accumulator_agrees_when_the_bound_leaves_i64() {
        // Unit scales, sign planes all `-1`, and an alternating activation that cancels inside
        // `V`: `z` is one 2^60 term, the `U` sum is 23·2^60 (past `i64`), and the 2^-8 post-scale
        // brings the output back. `29·2^60` is past `i64`, so both wide sources run.
        let mut p = packed(4);
        p.v.fill(0);
        p.u.fill(0);
        p.s_pre.fill(bf16_bits(1.0));
        p.s_mid.fill(bf16_bits(1.0));
        p.s_post.fill(bf16_bits(1.0 / 256.0));
        p.salient.fill(0);
        p.res_layers = 0;
        p.res.clear();
        p.s_res.clear();
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        let x: Vec<i64> = (0..p.inn)
            .map(|i| {
                if i % 2 == 0 {
                    1i64 << 60
                } else {
                    -(1i64 << 60)
                }
            })
            .collect();
        assert_eq!(linear.project(&x).expect("project"), sequential(&p, &x));
    }

    #[test]
    fn wrong_width_and_bad_scale_refuse() {
        let mut p = packed(2);
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        assert!(matches!(
            linear.project(&[0; 28]),
            Err(NqError::Shape { x: 28, n_in: 29 })
        ));
        p.s_pre[0] = 0x7f80; // +inf
        assert!(matches!(
            NqLinear::from_packed(p),
            Err(NqError::Scale { .. })
        ));
    }

    #[test]
    fn a_single_sign_flip_moves_the_output() {
        let p = packed(3);
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        let x: Vec<i64> = (0..p.inn).map(|i| (i as i64 + 1) << 26).collect();
        let a = linear.project(&x).expect("a");
        let mut q = p;
        q.u[0] ^= 1;
        let b = NqLinear::from_packed(q)
            .expect("decode")
            .project(&x)
            .expect("b");
        assert_ne!(a, b);
    }

    #[test]
    fn a_residual_sign_flip_moves_the_output() {
        let p = packed(6);
        assert_eq!(p.res_layers, 1);
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        let x: Vec<i64> = (0..p.inn).map(|i| (i as i64 + 1) << 26).collect();
        let a = linear.project(&x).expect("a");
        let mut q = p;
        q.res[0] ^= 1;
        let b = NqLinear::from_packed(q)
            .expect("decode")
            .project(&x)
            .expect("b");
        assert_ne!(a, b);
    }
}
