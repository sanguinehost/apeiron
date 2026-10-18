//! Attention: the per-layer key/value cache and one causal query head (port-map row 5).

use super::ONE;
use crate::engine::mixer::f32_carrier::q32_to_f32;
use crate::engine::mixer::{carrier_from_env, Carrier};
use crate::engine::num::fp::F32;
use crate::engine::num::value::{div_round_nearest, round_shift_32};
use crate::engine::ops::matvec::row_dot_fixed;
use crate::numeric::fixed_exp::fixed_exp;
use crate::numeric::fixed_rsqrt::fixed_rsqrt;
use std::cell::Cell;

#[path = "attn/f32_core.rs"]
pub mod f32_core;

// Test seam for `APEIRON_EXACT_SCAN`: a per-thread override of the flag, `None` falling through
// to the process value. The process parse lives in `src/engine/step/cols.rs` where the flag is
// consumed; the seam is here because `src/engine/tests.rs` cannot name the private
// `engine::step::cols` module.
thread_local! {
    static EXACT_SCAN_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// The thread's forced exact-scan value, or `None` to fall through to the process flag.
pub(crate) fn exact_scan_override() -> Option<bool> {
    EXACT_SCAN_OVERRIDE.with(Cell::get)
}

/// Force the exact-scan flag on this thread, or clear it with `None` (the test seam).
#[cfg(test)]
pub(crate) fn set_exact_scan_override(value: Option<bool>) {
    EXACT_SCAN_OVERRIDE.with(|cell| cell.set(value));
}

/// A refused fixed-point attention read (port-map row 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnError {
    /// A geometry parameter is zero, or `n_head` is not a multiple of `n_head_kv`.
    BadGeometry,
    /// `head_dim` left the `fixed_rsqrt` window, so the `1 / sqrt(head_dim)` score scale is
    /// undefined.
    ScaleOutOfRange,
    /// A layer index is at or past the configured layer count.
    LayerOutOfRange {
        /// Layer index supplied.
        layer: usize,
        /// Configured layer count.
        n_layer: usize,
    },
    /// A query-head index is at or past `n_head`.
    HeadOutOfRange {
        /// Query-head index supplied.
        head: usize,
        /// Configured query-head count.
        n_head: usize,
    },
    /// The key/value row block, the query or the output is not the required width.
    RowLength {
        /// Required width.
        expected: usize,
        /// Width supplied.
        got: usize,
    },
    /// The attention position is at or past the number of positions appended for the layer.
    PositionOutOfRange {
        /// Position supplied.
        pos: usize,
        /// Positions appended for the layer.
        len: usize,
    },
    /// The layer already holds `cap` positions, so no further append is admitted.
    CapacityExceeded {
        /// Configured position capacity.
        cap: usize,
    },
    /// A [`fixed_exp`] call in the score-shifted softmax refused. The shift is `<= 0` and its only
    /// reachable window edge `-16*ONE` is admitted, so with the underflow saturation this variant is
    /// retained but unreachable.
    ExpOutOfRange,
    /// An `i128` accumulation or an `i64` narrowing in the score, softmax or output left its range.
    Overflow,
    /// The softmax total is zero, so the probability normalisation is undefined.
    ZeroSum,
}

/// Typed storage for key and value planes: either Q32.32 fixed-point `i64` or binary32 `f32`.
#[derive(Debug, Clone)]
pub enum KvStorage {
    /// Fixed-point Q32.32 key and value planes.
    Q32 {
        /// Position-major key planes, one per layer.
        k: Vec<Vec<i64>>,
        /// Position-major value planes, one per layer.
        v: Vec<Vec<i64>>,
    },
    /// Native binary32 floating-point key and value planes.
    F32 {
        /// Position-major key planes, one per layer.
        k: Vec<Vec<F32>>,
        /// Position-major value planes, one per layer.
        v: Vec<Vec<F32>>,
    },
}

/// Allocate the zeroed native binary32 key and value planes for `n_layer` layers of `plane` values.
///
/// This is the carrier's binary32 allocation and nothing else: the geometry guards, the
/// `fixed_rsqrt` window and the checked plane product are decided by [`KvState::with_carrier`]
/// before it is called, and it cannot refuse. It is a named function so that the extraction
/// boundary can be the allocation rather than the whole constructor — the constructor's guards and
/// its Q32.32 allocation stay translated.
///
/// # Complexity
///
/// `O(n_layer · plane)` time and space in `f32` words: one zeroed `plane`-value plane, `n_layer`
/// clones of it into the key planes and the move of the blank into the value plane. The count is
/// two `vec![blank; n_layer]` repetitions over a `plane`-element element, so it closes at
/// `2 · n_layer · plane` writes.
fn alloc_f32_storage(plane: usize, n_layer: usize) -> KvStorage {
    let blank = vec![F32::ZERO; plane];
    let k = vec![blank.clone(); n_layer];
    let v = vec![blank; n_layer];
    KvStorage::F32 { k, v }
}

/// The single-token key/value cache of port-map row 5: one flat Q32.32 or binary32 plane per layer.
///
/// Keys and values are stored position-major, so position `p` carries all `n_head_kv` rows
/// contiguously and the row for `(p, kv_head)` is the `head_dim` values at
/// `(p * n_head_kv + kv_head) * head_dim`. One KV head serves `n_head / n_head_kv` query heads
/// (GQA). [`KvState::append`] writes a whole token's KV rows at the next position of a layer and
/// returns that position; [`KvState::attn_head`] reads one query head over the causal prefix
/// `0..=pos`.
#[derive(Debug, Clone)]
pub struct KvState {
    carrier: Carrier,
    n_layer: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    cap: usize,
    scale: i64,
    occupied: Vec<usize>,
    storage: KvStorage,
}

/// The Q32.32 position-major value of `(s, kvh, i)` in a layer's key or value plane.
///
/// The checked index product is the one the pinned Charon preset keeps as `checked.*` (it cannot
/// prove the bound), so the real `(s · n_head_kv + kvh) · head_dim + i` arithmetic appears here as
/// checked steps. The guards in [`KvState::attn_head`] admit only in-range positions.
fn plane_at(kv: usize, kvh: usize, hd: usize, s: usize, i: usize) -> Result<usize, AttnError> {
    let base = s.checked_mul(kv).ok_or(AttnError::Overflow)?;
    let base = base.checked_add(kvh).ok_or(AttnError::Overflow)?;
    let base = base.checked_mul(hd).ok_or(AttnError::Overflow)?;
    base.checked_add(i).ok_or(AttnError::Overflow)
}

/// Fold one key into the fused state: `den += exp_word` and `num[comp] += exp_word · v[comp]` for
/// each `comp < hd`, every `i128` step checked. Stops at the first failure, which is returned; the
/// partial update is discarded by the caller. Time `O(hd)` checked multiply-adds, space `O(1)`,
/// no allocation.
#[allow(
    clippy::too_many_arguments,
    reason = "flat scalar and slice parameters keep the helper free of nested regions for extraction"
)]
fn fold_key(
    score: i64,
    m: i64,
    vplane: &[i64],
    kv: usize,
    kvh: usize,
    hd: usize,
    key: usize,
    den: &mut i128,
    num: &mut [i128],
) -> Result<(), AttnError> {
    let exp_word = exp_at(score, m)?;
    let next = den
        .checked_add(i128::from(exp_word))
        .ok_or(AttnError::Overflow)?;
    *den = next;
    let mut comp = 0usize;
    let mut res: Result<(), AttnError> = Ok(());
    while comp < hd && res.is_ok() {
        res = fold_comp(exp_word, vplane, kv, kvh, hd, key, comp, num);
        comp += 1;
    }
    res
}

/// The key loop of the fused fold: [`fold_key`] for each `scores[key]`, in index order, stopping at
/// the first error. Time `O(count · hd)` checked operations, space `O(1)`.
#[allow(
    clippy::too_many_arguments,
    reason = "flat scalar and slice parameters keep the helper free of nested regions for extraction"
)]
fn fold_keys(
    scores: &[i64],
    m: i64,
    vplane: &[i64],
    kv: usize,
    kvh: usize,
    hd: usize,
    den: &mut i128,
    num: &mut [i128],
) -> Result<(), AttnError> {
    let count = scores.len();
    let mut key = 0usize;
    let mut res: Result<(), AttnError> = Ok(());
    while key < count && res.is_ok() {
        res = fold_key(scores[key], m, vplane, kv, kvh, hd, key, den, num);
        key += 1;
    }
    res
}

/// One component of [`fold_key`]: `num[comp] += exp_word · v[plane_at(comp)]`, checked. Straight
/// line, no loop; `O(1)` checked operations.
#[allow(
    clippy::too_many_arguments,
    reason = "flat scalar and slice parameters keep the helper free of nested regions for extraction"
)]
fn fold_comp(
    exp_word: i64,
    vplane: &[i64],
    kv: usize,
    kvh: usize,
    hd: usize,
    key: usize,
    comp: usize,
    num: &mut [i128],
) -> Result<(), AttnError> {
    let at = plane_at(kv, kvh, hd, key, comp)?;
    let vb = *vplane.get(at).ok_or(AttnError::Overflow)?;
    let term = i128::from(exp_word)
        .checked_mul(i128::from(vb))
        .ok_or(AttnError::Overflow)?;
    let next = num[comp].checked_add(term).ok_or(AttnError::Overflow)?;
    num[comp] = next;
    Ok(())
}

/// The checked `[base, base + head_dim)` bounds of the `head_dim`-wide row of key `s` in a
/// position-major plane: the same three `checked.*` index steps as [`plane_at`], with the row end
/// as one more. Test-only: the index oracle the hoisted `first + s·stride` base in
/// [`KvState::attn_head`] is pinned against, so the checked chain and the product stay the same
/// integer.
#[cfg(test)]
fn row_bounds(kv: usize, kvh: usize, hd: usize, s: usize) -> Result<(usize, usize), AttnError> {
    let base = s.checked_mul(kv).ok_or(AttnError::Overflow)?;
    let base = base.checked_add(kvh).ok_or(AttnError::Overflow)?;
    let base = base.checked_mul(hd).ok_or(AttnError::Overflow)?;
    let end = base.checked_add(hd).ok_or(AttnError::Overflow)?;
    Ok((base, end))
}

/// One Q32.32 causal key score: `row_dot_fixed` of `q` against key `s`, then the load-time `scale`
/// epilogue. The real `row_dot_fixed`, `checked_mul` and `round_shift_32` bodies are called.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
              immediately before it; `i64::try_from` emits an unmodeled external the fail-closed \
              extraction audit rejects"
)]
fn score_at(
    kplane: &[i64],
    q: &[i64],
    scale: i64,
    kv: usize,
    kvh: usize,
    hd: usize,
    s: usize,
) -> Result<i64, AttnError> {
    let base = s.checked_mul(kv).ok_or(AttnError::Overflow)?;
    let base = base.checked_add(kvh).ok_or(AttnError::Overflow)?;
    let base = base.checked_mul(hd).ok_or(AttnError::Overflow)?;
    let end = base.checked_add(hd).ok_or(AttnError::Overflow)?;
    let raw = row_dot_fixed(&kplane[base..end], q).map_err(|_| AttnError::Overflow)?;
    let prod = i128::from(raw)
        .checked_mul(i128::from(scale))
        .ok_or(AttnError::Overflow)?;
    let rounded = round_shift_32(prod);
    if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
        return Err(AttnError::Overflow);
    }
    Ok(rounded as i64)
}

/// One softmax numerator: the exact maximum shift, the `fixed_exp` window with the documented
/// underflow saturation below `-16·ONE`, and no clamping.
fn exp_at(score: i64, mx: i64) -> Result<i64, AttnError> {
    let shifted = score.checked_sub(mx).ok_or(AttnError::Overflow)?;
    if shifted < -16 * ONE {
        Ok(0)
    } else {
        fixed_exp(shifted).map_err(|_| AttnError::ExpOutOfRange)
    }
}

/// One Q32.32 softmax weight: the round-to-nearest division `round(e·2^32 / total)`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
              immediately before it; `i64::try_from` emits an unmodeled external the fail-closed \
              extraction audit rejects"
)]
fn weight_at(e: i64, total: i128) -> Result<i64, AttnError> {
    let scaled = i128::from(e)
        .checked_mul(i128::from(ONE))
        .ok_or(AttnError::Overflow)?;
    let p = div_round_nearest(scaled, total);
    if p > i128::from(i64::MAX) || p < i128::from(i64::MIN) {
        return Err(AttnError::Overflow);
    }
    Ok(p as i64)
}

/// The round-up `u64` reciprocal `floor(2^64 / total) + 1` of [`weight_at_recip`], or `None` when
/// `total` leaves the window `2 ..= u64::MAX/2` the two-operand product bound needs.
///
/// `floor(2^64/total) + 1` is the classical round-up multiplier: `n2 · recip / 2^64` is then
/// `floor(n2/total)` or one greater, which the single subtract fixup in [`weight_at_recip`] removes.
/// The truncated `floor(2^64/total)` reciprocal is one *below* `floor(n2/total)` whenever
/// `2^64 mod total != 0` and the product lands on the low side (for example `total = 3`, `e = 2`),
/// and a subtract-only fixup cannot repair that.
///
/// # Complexity
///
/// O(1) time and space: one `u128` division, one add and two comparisons.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the `total >= 2` guard makes the quotient `<= 2^63`, so both casts are exact; \
              `i64::try_from`/`u64::try_from` would emit unmodeled externals the fail-closed \
              extraction audit rejects"
)]
fn recip_u64(total: i128) -> Option<u64> {
    if total < 2 || total > i128::from(u64::MAX / 2) {
        return None;
    }
    let base = ((1u128 << 64) / (total as u128)) as u64;
    base.checked_add(1)
}

/// One softmax weight through the `u64` round-up reciprocal: the same
/// `div_round_nearest(e·2^32, total)` value as [`weight_at`], with the division replaced by one
/// `u128` multiply and shift.
///
/// The fast path applies exactly when `e >= 0`, `e·2^32 + total/2` fits `u64`, and `total` fits the
/// `recip_u64` window (guaranteed by the caller's `Some`); every other operand — a negative `e`, an
/// `i128` overflow, or a numerator above `u64::MAX` — delegates to the literal i128 [`weight_at`].
///
/// # Complexity
///
/// O(1) time and space: one `u128` multiply, one shift and one fixup comparison.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "every `as` is guarded: `e >= 0` fixes the sign of `n`, `n <= i128::from(u64::MAX)` \
              and the caller's `total <= u64::MAX/2` fix the widths, and the `i64` narrowing is \
              range-checked against i128::from(i64::MAX)/i64::MIN immediately before it; the \
              `try_from` forms emit unmodeled externals the fail-closed extraction audit rejects"
)]
fn weight_at_recip(e: i64, total: i128, recip: u64) -> Result<i64, AttnError> {
    let scaled = i128::from(e).checked_mul(i128::from(ONE));
    let narrowed = match scaled {
        Some(s) => match s.checked_add(total / 2) {
            Some(n) => {
                if e >= 0 {
                    if n <= i128::from(u64::MAX) {
                        Some(n as u64)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            None => None,
        },
        None => None,
    };
    match narrowed {
        Some(n2) => {
            let d = total as u64;
            let q = ((u128::from(n2) * u128::from(recip)) >> 64) as u64;
            let q = if u128::from(q) * u128::from(d) > u128::from(n2) {
                q - 1
            } else {
                q
            };
            let p = i128::from(q);
            if p > i128::from(i64::MAX) || p < i128::from(i64::MIN) {
                return Err(AttnError::Overflow);
            }
            Ok(p as i64)
        }
        None => weight_at(e, total),
    }
}

/// One `p_s · v_s` term of the output component `i`. Test-only: the key-outer production output
/// pass folds the same terms into all components at once, and this per-term oracle keeps the
/// old per-component arithmetic exercised by the output-pin test.
#[cfg(test)]
fn value_term(
    vplane: &[i64],
    weights: &[i64],
    kv: usize,
    kvh: usize,
    hd: usize,
    s: usize,
    i: usize,
) -> Result<i128, AttnError> {
    let at = plane_at(kv, kvh, hd, s, i)?;
    let vb = *vplane.get(at).ok_or(AttnError::Overflow)?;
    i128::from(weights[s])
        .checked_mul(i128::from(vb))
        .ok_or(AttnError::Overflow)
}

/// The single-component output accumulation: the checked `i128` sum over the causal prefix, then
/// one `round_shift_32` epilogue. The position loop carries a `bool` failure flag so no early
/// return appears inside a loop, the extractable form the pinned fragment admits.
///
/// Test-only oracle for the per-component scan the key-outer production pass replaces.
#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
              immediately before it; `i64::try_from` emits an unmodeled external the fail-closed \
              extraction audit rejects"
)]
fn value_component(
    vplane: &[i64],
    weights: &[i64],
    kv: usize,
    kvh: usize,
    hd: usize,
    pos: usize,
    i: usize,
) -> Result<i64, AttnError> {
    let mut acc: i128 = 0;
    let mut s = 0usize;
    let mut ok = true;
    let mut err: Option<AttnError> = None;
    while s <= pos && ok {
        match value_term(vplane, weights, kv, kvh, hd, s, i) {
            Ok(term) => {
                if let Some(a) = acc.checked_add(term) {
                    acc = a;
                } else {
                    err = Some(AttnError::Overflow);
                    ok = false;
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        s += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let rounded = round_shift_32(acc);
    if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
        return Err(AttnError::Overflow);
    }
    Ok(rounded as i64)
}

/// The fused online-softmax state of one query head: the fixed max-shift score and the exact
/// `i128` denominator and numerator sums of the causal prefix at that shift.
///
/// `den = Σ_{j≤pos} e_j` and `num_i = Σ_{j≤pos} e_j·v_{j,i}`, with `e_j = exp_at(score_j, m)`.
/// Both sums are exact checked `i128` additions, so they are independent of the order and of the
/// tile grouping of the keys (integer addition is associative and commutative). In real reading
/// `num_i/den` is `Σ_j softmax(score)_j·v_{j,i}` — the common shift cancels — the dense softmax
/// attention `Numeric.EngineAttnRefinement.denseAttnReal`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttnFusedState {
    /// The fixed max-shift score `m`.
    pub m: i64,
    /// The exact softmax denominator `Σ_j e_j` (`> 0` for a state that was returned).
    pub den: i128,
    /// The exact numerator `Σ_j e_j·v_{j,i}`, `head_dim` wide.
    pub num: Vec<i128>,
}

/// Round-to-nearest `num / den` (`den > 0`) for a signed numerator: the sign-symmetric companion
/// of [`round_shift_32`], ties away from zero. The error against the exact quotient is below
/// `1/2 + 1/(2·den)`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
              immediately before it; `i64::try_from` emits an unmodeled external the fail-closed \
              extraction audit rejects"
)]
fn round_div_signed(num: i128, den: i128) -> Result<i64, AttnError> {
    let q = if num >= 0 {
        (num + den / 2) / den
    } else {
        // `(num - den/2) / den` rounds the magnitude to nearest without negating `num`; the plain
        // form stays total at `i128::MIN`.
        (num - den / 2) / den
    };
    if q > i128::from(i64::MAX) || q < i128::from(i64::MIN) {
        return Err(AttnError::Overflow);
    }
    Ok(q as i64)
}

impl KvState {
    /// The shared guard chain of the fused read, in the order [`KvState::attn_head`] checks it
    /// (layer, query head, query width, position); returns the GQA KV head `q_head` reads.
    fn fused_kvh(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
    ) -> Result<usize, AttnError> {
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        if q_head >= self.n_head {
            return Err(AttnError::HeadOutOfRange {
                head: q_head,
                n_head: self.n_head,
            });
        }
        if q.len() != self.head_dim {
            return Err(AttnError::RowLength {
                expected: self.head_dim,
                got: q.len(),
            });
        }
        let len = self.occupied[layer];
        if pos >= len {
            return Err(AttnError::PositionOutOfRange { pos, len });
        }
        Ok(q_head / (self.n_head / self.n_head_kv))
    }

    /// The score prefix `0..=pos` of the fused read: one [`score_at`] per causal key, the same
    /// scores [`KvState::attn_head`] computes. Returns the GQA KV head and the score list.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time in checked `i128` multiply-adds, `O(pos + 1)` space for the
    /// score list: `O(pos + 1)` calls into `O(head_dim)` each, accumulated once per key.
    fn score_prefix(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
    ) -> Result<(usize, Vec<i64>), AttnError> {
        let kvh = self.fused_kvh(layer, q_head, q, pos)?;
        let hd = self.head_dim;
        let kv = self.n_head_kv;
        let slots = pos.checked_add(1).ok_or(AttnError::Overflow)?;
        let mut scores: Vec<i64> = Vec::with_capacity(slots);
        let kplane = match &self.storage {
            KvStorage::Q32 { k, .. } => &k[layer],
            KvStorage::F32 { .. } => return Err(AttnError::BadGeometry),
        };
        let mut s = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while s <= pos && ok {
            match score_at(kplane, q, self.scale, kv, kvh, hd, s) {
                Ok(score) => scores.push(score),
                Err(e) => {
                    err = Some(e);
                    ok = false;
                }
            }
            s += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        Ok((kvh, scores))
    }

    /// Fold the precomputed score prefix into a fused `(m, den, num)` state at the fixed shift
    /// `m`, one key at a time in index order. `tile` has no effect: the `i128` sums are the same
    /// for every value, including `0`, which `attn_fused_state` and `attn_head_fused` refuse with
    /// [`AttnError::BadGeometry`] before this call.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time in checked `i128` multiply-adds and one [`fixed_exp`] per
    /// key, `O(head_dim)` space for `num` plus `O(1)` for the key cursor; the op count is
    /// independent of `tile`.
    fn fused_fold(
        &self,
        layer: usize,
        kvh: usize,
        scores: &[i64],
        m: i64,
        _tile: usize,
    ) -> Result<AttnFusedState, AttnError> {
        let hd = self.head_dim;
        let kv = self.n_head_kv;
        let vplane = match &self.storage {
            KvStorage::Q32 { v, .. } => &v[layer],
            KvStorage::F32 { .. } => return Err(AttnError::BadGeometry),
        };
        let mut den: i128 = 0;
        let mut num: Vec<i128> = vec![0i128; hd];
        fold_keys(scores, m, &vplane[..], kv, kvh, hd, &mut den, &mut num[..])?;
        if den <= 0 {
            return Err(AttnError::ZeroSum);
        }
        Ok(AttnFusedState { m, den, num })
    }

    /// The exact maximum score over the causal prefix `0..=pos`: pass 1 of the fused read, the
    /// shift the fold fixes for every key. The prefix is nonempty (`pos < occupied`), so the
    /// result is a real score.
    ///
    /// # Errors
    ///
    /// The [`KvState::attn_head`] guard chain and its `AttnError::Overflow` on a score epilogue.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time, `O(pos + 1)` space for the score prefix.
    #[must_use = "the max shift is the fold's fixed `m`"]
    pub fn attn_max(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
    ) -> Result<i64, AttnError> {
        let (_kvh, scores) = self.score_prefix(layer, q_head, q, pos)?;
        let mut mx = i64::MIN;
        let mut s = 0usize;
        while s < scores.len() {
            if scores[s] > mx {
                mx = scores[s];
            }
            s += 1;
        }
        Ok(mx)
    }

    /// The fused `(m, den, num)` state of the causal prefix at the fixed shift `m`, folded key by
    /// key. `m` is caller-supplied: the online read passes the exact maximum from
    /// [`KvState::attn_max`], so every key is shifted by one common value and the exact `i128`
    /// sums do not depend on `tile`.
    ///
    /// # Errors
    ///
    /// The [`KvState::attn_head`] guard chain; [`AttnError::BadGeometry`] when `tile == 0`;
    /// [`AttnError::ZeroSum`] when `den == 0` (every shift below the `fixed_exp` window);
    /// [`AttnError::Overflow`] when an `i128` accumulation leaves range.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time, `O(head_dim)` space beyond the `O(pos + 1)` score prefix;
    /// independent of `tile`.
    pub fn attn_fused_state(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
        m: i64,
        tile: usize,
    ) -> Result<AttnFusedState, AttnError> {
        if tile == 0 {
            return Err(AttnError::BadGeometry);
        }
        let (kvh, scores) = self.score_prefix(layer, q_head, q, pos)?;
        self.fused_fold(layer, kvh, &scores, m, tile)
    }

    /// The fused online attention read of one query head at `pos`, the exact-scan companion of
    /// [`KvState::attn_head`]: the exact maximum score over the causal prefix, the exact
    /// `(m, den, num)` fold in `tile`-key groups, and one normalisation `round(num_i / den)` per
    /// output component. The per-probability path of [`KvState::attn_head`] is unchanged and
    /// remains the oracle.
    ///
    /// # Errors
    ///
    /// The [`KvState::attn_head`] guard chain; [`AttnError::BadGeometry`] when `tile == 0`;
    /// [`AttnError::ZeroSum`] when the fold's denominator is zero (unreachable at the exact
    /// maximum, retained for a caller-supplied `m`); [`AttnError::Overflow`] when an `i128`
    /// accumulation or an `i64` narrowing leaves range.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time in checked `i128` multiply-adds and one [`fixed_exp`] per
    /// key, `O(pos + 1)` space for the score prefix and `O(head_dim)` for the fused state; the
    /// per-position weight vector and its `O(pos + 1)` divisions are removed.
    pub fn attn_head_fused(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
        tile: usize,
        out: &mut [i64],
    ) -> Result<(), AttnError> {
        if tile == 0 {
            return Err(AttnError::BadGeometry);
        }
        if self.carrier == Carrier::F32 {
            return self.attn_head_f32(layer, q_head, q, pos, out);
        }
        if out.len() != self.head_dim {
            return Err(AttnError::RowLength {
                expected: self.head_dim,
                got: out.len(),
            });
        }
        let (kvh, scores) = self.score_prefix(layer, q_head, q, pos)?;
        // Pass 1 of the online read: the exact global maximum, the one shift every tile folds
        // under. A per-tile maximum would rescale the running sums and make the result depend on
        // the tile boundaries.
        let mut mx = i64::MIN;
        let mut s = 0usize;
        while s < scores.len() {
            if scores[s] > mx {
                mx = scores[s];
            }
            s += 1;
        }
        let state = self.fused_fold(layer, kvh, &scores, mx, tile)?;
        let mut i = 0usize;
        while i < self.head_dim {
            out[i] = round_div_signed(state.num[i], state.den)?;
            i += 1;
        }
        Ok(())
    }

    /// Allocate a zeroed cache for `n_layer` layers, `n_head` query heads, `n_head_kv` KV heads,
    /// `head_dim` values per head and at most `cap` positions per layer, resolving the carrier
    /// from the process environment (`carrier_from_env()`).
    ///
    /// The load-time score scale `fixed_rsqrt(head_dim)` is computed once here, from the same
    /// integer kernel the executing path uses; `head_dim` is admitted in the `fixed_rsqrt` window.
    ///
    /// # Errors
    ///
    /// Returns [`AttnError::BadGeometry`] for a zero parameter or `n_head` not a multiple of
    /// `n_head_kv`, [`AttnError::ScaleOutOfRange`] when `head_dim` leaves the `fixed_rsqrt` window,
    /// and [`AttnError::Overflow`] when the per-layer plane length leaves `usize`.
    ///
    /// # Complexity
    ///
    /// `O(n_layer · n_head_kv · cap · head_dim)` time and space: one zeroed plane per layer.
    pub fn new(
        n_layer: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        cap: usize,
    ) -> Result<KvState, AttnError> {
        Self::with_carrier(
            n_layer,
            n_head,
            n_head_kv,
            head_dim,
            cap,
            carrier_from_env(),
        )
    }

    /// Allocate a zeroed cache with an explicitly specified numerical [`Carrier`].
    ///
    /// Under [`Carrier::Q32`], allocates Q32.32 `i64` planes. Under [`Carrier::F32`], allocates
    /// native binary32 `f32` planes (50% memory footprint, zero per-call float conversion).
    ///
    /// # Errors
    ///
    /// Same geometry and range refusals as [`KvState::new`].
    ///
    /// Every geometry and range guard is decided before the first allocation; the carrier's
    /// binary32 allocation is the single operation behind [`alloc_f32_storage`].
    pub(crate) fn with_carrier(
        n_layer: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        cap: usize,
        carrier: Carrier,
    ) -> Result<KvState, AttnError> {
        if n_layer == 0 || n_head == 0 || n_head_kv == 0 || head_dim == 0 || cap == 0 {
            return Err(AttnError::BadGeometry);
        }
        if !n_head.is_multiple_of(n_head_kv) {
            return Err(AttnError::BadGeometry);
        }
        // `head_dim` widened after an explicit range check: `i64::try_from` emits an unmodeled
        // external the fail-closed extraction audit rejects. The check is exact on 64-bit, where
        // `head_dim as i64` is negative exactly when `head_dim > i64::MAX`.
        #[allow(
            clippy::cast_possible_wrap,
            reason = "the `hd_i64 < 0` check immediately above the cast is the range guard; the \
                      cast is exact for every value it admits"
        )]
        let hd_signed = head_dim as i64;
        if hd_signed < 0 {
            return Err(AttnError::ScaleOutOfRange);
        }
        // `checked_shl` is unmodeled; `head_dim · 2^FIXED_FRAC_BITS` is the same radicand and
        // overflows `i64` on exactly the inputs the shift rejects.
        let radicand = hd_signed
            .checked_mul(ONE)
            .ok_or(AttnError::ScaleOutOfRange)?;
        let scale = fixed_rsqrt(radicand).map_err(|_| AttnError::ScaleOutOfRange)?;
        // `Option::and_then` is unmodeled; the nested match is the same checked product.
        let plane = match n_head_kv.checked_mul(cap) {
            Some(x) => match x.checked_mul(head_dim) {
                Some(plane) => plane,
                None => return Err(AttnError::Overflow),
            },
            None => return Err(AttnError::Overflow),
        };
        let storage = match carrier {
            Carrier::Q32 => {
                let blank = vec![0i64; plane];
                let k = vec![blank.clone(); n_layer];
                let v = vec![blank; n_layer];
                KvStorage::Q32 { k, v }
            }
            Carrier::F32 => alloc_f32_storage(plane, n_layer),
        };
        Ok(KvState {
            carrier,
            n_layer,
            n_head,
            n_head_kv,
            head_dim,
            cap,
            scale,
            occupied: vec![0usize; n_layer],
            storage,
        })
    }

    /// The numerical carrier for this key/value cache.
    #[must_use]
    #[allow(
        dead_code,
        reason = "public-in-crate accessor for the resolved carrier"
    )]
    pub(crate) fn carrier(&self) -> Carrier {
        self.carrier
    }

    /// The configured layer count.
    #[must_use]
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    /// The configured query-head count.
    #[must_use]
    pub fn n_head(&self) -> usize {
        self.n_head
    }

    /// The configured KV-head count.
    #[must_use]
    pub fn n_head_kv(&self) -> usize {
        self.n_head_kv
    }

    /// The per-head width.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The per-layer position capacity.
    #[must_use]
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// The load-time Q32.32 score scale `fixed_rsqrt(head_dim)`.
    #[must_use]
    pub fn scale(&self) -> i64 {
        self.scale
    }

    /// Positions appended for `layer`.
    ///
    /// # Errors
    ///
    /// Returns [`AttnError::LayerOutOfRange`] for a layer index at or past the layer count.
    pub fn occupied(&self, layer: usize) -> Result<usize, AttnError> {
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        Ok(self.occupied[layer])
    }

    /// Append one token's KV rows at the layer's next position and return that position.
    ///
    /// `k_rows` and `v_rows` each hold the `n_head_kv` heads of one token's `head_dim`-wide rows
    /// (`n_head_kv · head_dim` values), KV head `h` first. This is the single-token step of the
    /// fold `kv_write`; the position advances only after the whole token's rows are written.
    ///
    /// # Errors
    ///
    /// Returns [`AttnError::LayerOutOfRange`] for a layer index at or past the layer count,
    /// [`AttnError::RowLength`] when a block is not `n_head_kv · head_dim` wide, and
    /// [`AttnError::CapacityExceeded`] when the layer already holds `cap` positions. No row is
    /// clamped or truncated.
    ///
    /// **The destination plane's contents are unspecified on `Err`.** The `F32` carrier writes
    /// the whole row before raising [`AttnError::Overflow`], so a rejecting element leaves a
    /// partially-written row past that element rather than an untouched tail; the earlier
    /// raise-at-first-failure guarantee is not preserved. `occupied` is not advanced and no
    /// caller reads the plane on `Err`.
    ///
    /// # Complexity
    ///
    /// `O(n_head_kv · head_dim)` time and no allocation: the two row blocks are copied into the
    /// layer planes.
    pub fn append(
        &mut self,
        layer: usize,
        k_rows: &[i64],
        v_rows: &[i64],
    ) -> Result<usize, AttnError> {
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        let width = self
            .n_head_kv
            .checked_mul(self.head_dim)
            .ok_or(AttnError::Overflow)?;
        if k_rows.len() != width {
            return Err(AttnError::RowLength {
                expected: width,
                got: k_rows.len(),
            });
        }
        if v_rows.len() != width {
            return Err(AttnError::RowLength {
                expected: width,
                got: v_rows.len(),
            });
        }
        let pos = self.occupied[layer];
        if pos >= self.cap {
            return Err(AttnError::CapacityExceeded { cap: self.cap });
        }
        // Checked forms throughout: a plain `*`/`+` on `usize` is lowered by the pinned Charon
        // preset to `checked.*` with an overflow assert, which the Aeneas fragment rejects as an
        // unimplemented binary operation. Each product is bounded by the checked `width` and the
        // admitted `pos < cap`, so no reachable input changes branch.
        let base = pos.checked_mul(width).ok_or(AttnError::Overflow)?;
        let end = base.checked_add(width).ok_or(AttnError::Overflow)?;
        match &mut self.storage {
            KvStorage::Q32 { k, v } => {
                k[layer][base..end].copy_from_slice(k_rows);
                v[layer][base..end].copy_from_slice(v_rows);
            }
            // Index `while` loops, not `iter_mut().zip`, and one `Err` after the loops. Both
            // removals are forced by the pinned Aeneas model, and neither is a change of shape
            // choice: the model has no field for `Iterator::zip` (prepass reports `Could not
            // find the information for item 'zip'`), and a `return` whose target is the function
            // boundary crossed from inside a loop nest is refused outright (`Returns inside of
            // nested loops are not supported yet`). Arithmetic is unchanged: the same
            // `q32_to_f32` on the same element in the same order, the same store to the same
            // slot, and the same `is_finite` test on the same value, over the index range
            // `0..width` that the two length checks above have already made equal to
            // `k_dst.len() == v_dst.len()`.
            KvStorage::F32 { k, v } => {
                let mut bad = 0u32;
                let k_dst = &mut k[layer][base..end];
                let mut i = 0usize;
                while i < width {
                    let f = q32_to_f32(k_rows[i]);
                    bad += u32::from(!f.is_finite());
                    k_dst[i] = f;
                    i += 1;
                }
                let v_dst = &mut v[layer][base..end];
                let mut i = 0usize;
                while i < width {
                    let f = q32_to_f32(v_rows[i]);
                    bad += u32::from(!f.is_finite());
                    v_dst[i] = f;
                    i += 1;
                }
                if bad != 0 {
                    return Err(AttnError::Overflow);
                }
            }
        }
        self.occupied[layer] = pos.checked_add(1).ok_or(AttnError::Overflow)?;
        Ok(pos)
    }

    /// Read one query head at position `pos` over the causal prefix `0..=pos`.
    ///
    /// The score is `round(round(Σ q·k_j / 2^32) · scale / 2^32)` — [`row_dot_fixed`] then one
    /// value-path epilogue with the load-time `scale`. The softmax subtracts the exact maximum and
    /// evaluates [`fixed_exp`] on `shifted = score - mx >= -16·ONE`; a `shifted < -16·ONE` saturates
    /// its numerator to `0`. The total is a checked `i128` sum, each weight is the round-to-nearest
    /// `p_j = round(e_j·2^32 / Σ e)`, and the output is `round(Σ p_j·v_j / 2^32)` under the same
    /// epilogue. Query head `q_head` reads KV head `q_head / (n_head / n_head_kv)` (GQA); positions
    /// past `pos` are never read (causal mask).
    ///
    /// # Errors
    ///
    /// Returns [`AttnError::LayerOutOfRange`], [`AttnError::HeadOutOfRange`],
    /// [`AttnError::RowLength`] or [`AttnError::PositionOutOfRange`] for a bad index or width,
    /// [`AttnError::ZeroSum`] only for a genuinely empty total, and [`AttnError::Overflow`] when an
    /// `i128` accumulation or an `i64` narrowing leaves its range. Scores are exact; only the
    /// softmax numerator below `-16·ONE` saturates, so no attended input is clamped.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time in checked integer multiply-adds and `fixed_exp` calls, with
    /// `O(pos + 1)` scratch space for the scores and exponentials; `out` is `head_dim`
    /// wide.
    #[allow(
        clippy::too_many_lines,
        reason = "the one linear score/softmax/weight/value composition of port-map row 5; the \
                  four bodies keep the fold's stage order and each carries the extraction \
                  fragment's explicit failure-flag loop shape"
    )]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
                  immediately before it; `i64::try_from` emits an unmodeled external the fail-closed \
                  extraction audit rejects"
    )]
    pub fn attn_head(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
        out: &mut [i64],
    ) -> Result<(), AttnError> {
        if self.carrier == Carrier::F32 {
            return self.attn_head_f32(layer, q_head, q, pos, out);
        }
        let hd = self.head_dim;
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        if q_head >= self.n_head {
            return Err(AttnError::HeadOutOfRange {
                head: q_head,
                n_head: self.n_head,
            });
        }
        if q.len() != hd {
            return Err(AttnError::RowLength {
                expected: hd,
                got: q.len(),
            });
        }
        if out.len() != hd {
            return Err(AttnError::RowLength {
                expected: hd,
                got: out.len(),
            });
        }
        let len = self.occupied[layer];
        if pos >= len {
            return Err(AttnError::PositionOutOfRange { pos, len });
        }
        let (kplane, vplane) = match &self.storage {
            KvStorage::Q32 { k, v } => (&k[layer], &v[layer]),
            KvStorage::F32 { .. } => return Err(AttnError::BadGeometry),
        };
        let group = self.n_head / self.n_head_kv;
        let kvh = q_head / group;
        let kv = self.n_head_kv;
        let slots = pos.checked_add(1).ok_or(AttnError::Overflow)?;
        // The position-major row `(s·kv + kvh)·hd` is the hoisted `first + s·stride`; the guards
        // admit only `s <= pos < occupied <= cap`, so `base + hd <= plane.len()` and the product is
        // exact without a per-key checked chain. One bound check per key is the same refusal the
        // checked index chain produced.
        let stride = kv.checked_mul(hd).ok_or(AttnError::Overflow)?;
        let first = kvh.checked_mul(hd).ok_or(AttnError::Overflow)?;

        // Scores: one Q32.32 dot per causal key, then one epilogue against the load-time scale.
        // Each loop carries a `bool` failure flag and stops on the first refusal, so no early
        // return appears inside a body; the refusals, arithmetic and order are unchanged.
        let mut scores: Vec<i64> = Vec::with_capacity(slots);
        let mut mx = i64::MIN;
        let mut s = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while s <= pos && ok {
            let base = first + s * stride;
            if base > kplane.len().saturating_sub(hd) {
                err = Some(AttnError::Overflow);
                ok = false;
            } else {
                let end = base + hd;
                if let Ok(raw) = row_dot_fixed(&kplane[base..end], q) {
                    if let Some(prod) = i128::from(raw).checked_mul(i128::from(self.scale)) {
                        let rounded = round_shift_32(prod);
                        if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
                            err = Some(AttnError::Overflow);
                            ok = false;
                        } else {
                            let score = rounded as i64;
                            if score > mx {
                                mx = score;
                            }
                            scores.push(score);
                        }
                    } else {
                        err = Some(AttnError::Overflow);
                        ok = false;
                    }
                } else {
                    err = Some(AttnError::Overflow);
                    ok = false;
                }
            }
            s += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }

        // Softmax: exact max shift, `fixed_exp` over `[-16·ONE, 0]`, underflow saturation below
        // `-16·ONE`, checked i128 total.
        let mut exps: Vec<i64> = Vec::with_capacity(slots);
        let mut total: i128 = 0;
        let mut s = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while s <= pos && ok {
            match exp_at(scores[s], mx) {
                Ok(e) => {
                    if let Some(t) = total.checked_add(i128::from(e)) {
                        total = t;
                        exps.push(e);
                    } else {
                        err = Some(AttnError::Overflow);
                        ok = false;
                    }
                }
                Err(e) => {
                    err = Some(e);
                    ok = false;
                }
            }
            s += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        if total <= 0 {
            return Err(AttnError::ZeroSum);
        }

        // The weights fold into the output pass: each key's round-to-nearest division is resolved
        // immediately before that key's row accumulates, replacing the separate `weights` vector
        // and its pass. The reciprocal and its Option are resolved once per call; the output loop
        // only picks the fast or the literal i128 path. `exps` are non-negative with `total >= ONE`
        // here, so the weight never refuses where the old weights pass would have; a refusal at key
        // `s` is raised with the same `AttnError` at the same ascending key. A refused read still
        // leaves `out` untouched, because `out` is written only after every key folds.
        let recip = recip_u64(total);

        // Output: one key-outer pass. Each key's weight is resolved, then its `head_dim` row is read
        // contiguously once and its `weight·v[s, i]` products accumulate into the independent
        // per-component checked `i128` sums. For each `i` the terms are the same checked products in
        // the same ascending `s` order as the per-component oracle, so every `acc[i]` is the
        // identical sum and the single `round_shift_32` epilogue plus its `i64` range check are
        // unchanged; only the interleaving of independent sums changes (integer addition is
        // associative and commutative). One `bool` failure flag per loop, and no early return inside
        // a body, keep the pinned extraction fragment's shape.
        let mut acc: Vec<i128> = vec![0i128; hd];
        let mut s = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while s <= pos && ok {
            let w = match recip {
                Some(r) => weight_at_recip(exps[s], total, r),
                None => weight_at(exps[s], total),
            };
            match w {
                Ok(weight) => {
                    let base = first + s * stride;
                    if base > vplane.len().saturating_sub(hd) {
                        err = Some(AttnError::Overflow);
                        ok = false;
                    } else {
                        let end = base + hd;
                        let row = &vplane[base..end];
                        let w = i128::from(weight);
                        let mut i = 0usize;
                        while i < hd && ok {
                            if let Some(term) = w.checked_mul(i128::from(row[i])) {
                                if let Some(next) = acc[i].checked_add(term) {
                                    acc[i] = next;
                                } else {
                                    err = Some(AttnError::Overflow);
                                    ok = false;
                                }
                            } else {
                                err = Some(AttnError::Overflow);
                                ok = false;
                            }
                            i += 1;
                        }
                    }
                }
                Err(e) => {
                    err = Some(e);
                    ok = false;
                }
            }
            s += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        let mut i = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while i < hd && ok {
            let rounded = round_shift_32(acc[i]);
            if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
                err = Some(AttnError::Overflow);
                ok = false;
            } else {
                out[i] = rounded as i64;
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        Ok(())
    }

    /// Read one query head at position `pos` over the causal prefix `0..=pos` in binary32 (f32).
    ///
    /// Reads slices directly from the native binary32 KV cache with zero per-call float conversion,
    /// executes the online-softmax attention core via [`f32_core::attn_core_f32_strided`], and
    /// converts output components back to Q32.32 via round-to-nearest. Refuses non-finite numbers
    /// and out-of-range outputs with [`AttnError::Overflow`].
    ///
    /// # Errors
    ///
    /// Same guard chain as [`KvState::attn_head`]: [`AttnError::LayerOutOfRange`],
    /// [`AttnError::HeadOutOfRange`], [`AttnError::RowLength`], [`AttnError::PositionOutOfRange`],
    /// [`AttnError::Overflow`] on non-finite floats or overflow, and [`AttnError::ZeroSum`] if
    /// the softmax sum is not positive.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time: dot products and value accumulations.
    /// Zero heap allocation on hot path for `head_dim <= 256`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::similar_names,
        reason = "Q32.32 to f32 conversions and stack scratch buffers"
    )]
    pub fn attn_head_f32(
        &self,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
        out: &mut [i64],
    ) -> Result<(), AttnError> {
        let hd = self.head_dim;
        if layer >= self.n_layer {
            return Err(AttnError::LayerOutOfRange {
                layer,
                n_layer: self.n_layer,
            });
        }
        if q_head >= self.n_head {
            return Err(AttnError::HeadOutOfRange {
                head: q_head,
                n_head: self.n_head,
            });
        }
        if q.len() != hd {
            return Err(AttnError::RowLength {
                expected: hd,
                got: q.len(),
            });
        }
        if out.len() != hd {
            return Err(AttnError::RowLength {
                expected: hd,
                got: out.len(),
            });
        }
        let len = self.occupied[layer];
        if pos >= len {
            return Err(AttnError::PositionOutOfRange { pos, len });
        }
        let group = self.n_head / self.n_head_kv;
        let kvh = q_head / group;
        let kv = self.n_head_kv;

        let stride = kv.checked_mul(hd).ok_or(AttnError::Overflow)?;
        let first = kvh.checked_mul(hd).ok_or(AttnError::Overflow)?;

        let (kplane, vplane) = match &self.storage {
            KvStorage::F32 { k, v } => (&k[layer], &v[layer]),
            KvStorage::Q32 { .. } => return Err(AttnError::BadGeometry),
        };

        let last_base = first + pos * stride;
        if last_base > kplane.len().saturating_sub(hd)
            || last_base > vplane.len().saturating_sub(hd)
        {
            return Err(AttnError::Overflow);
        }

        // Each branch lends its scratch buffers to `attn_head_f32_rows`: a join of a stack and a heap
        // `&mut` is refused by the pinned Aeneas model. Stack for hd <= 256, heap otherwise.
        if hd <= 256 {
            let mut q_stack = [F32::ZERO; 256];
            let mut out_stack = [F32::ZERO; 256];
            Self::attn_head_f32_rows(
                &kplane[first..],
                &vplane[first..],
                q,
                hd,
                stride,
                pos,
                out,
                &mut q_stack[..hd],
                &mut out_stack[..hd],
            )
        } else {
            let mut q_heap = vec![F32::ZERO; hd];
            let mut out_heap = vec![F32::ZERO; hd];
            Self::attn_head_f32_rows(
                &kplane[first..],
                &vplane[first..],
                q,
                hd,
                stride,
                pos,
                out,
                &mut q_heap[..],
                &mut out_heap[..],
            )
        }
    }

    /// Q32.32 query to f32, the f32 core over `kx`/`vx`, and the f32 result back to Q32.32, with the
    /// two scratch buffers supplied by the caller (`q_f32`, `out_f32`, each of length `hd`).
    #[allow(
        clippy::too_many_arguments,
        reason = "one scratch buffer per caller branch keeps the Aeneas-visible borrows single-level"
    )]
    fn attn_head_f32_rows(
        kx: &[F32],
        vx: &[F32],
        q: &[i64],
        hd: usize,
        stride: usize,
        pos: usize,
        out: &mut [i64],
        q_f32: &mut [F32],
        out_f32: &mut [F32],
    ) -> Result<(), AttnError> {
        // Index `while` loop, and the one `Err` raised after it. The removal is forced by the
        // pinned Aeneas model, which refuses this body with `Breaks to outer loops are not
        // supported yet`: the `return Err(AttnError::Overflow)` sat inside the `for` body, and an
        // early exit from a loop body is the shape its translator cannot lower. No `zip` appears in
        // this body, so that refusal does not apply here. Arithmetic is unchanged: the same
        // `q32_to_f32` on the same element in the same ascending order, the same store to the same
        // slot, and the same `is_finite` test on the same value, over `0..hd`, which is exactly the
        // old `for` range. `q_f32` is a local scratch buffer with no reader after this loop, so
        // writing it whole before raising rather than up to the first non-finite element is not
        // observable.
        let mut bad = 0u32;
        let mut i = 0usize;
        while i < hd {
            let f = q32_to_f32(q[i]);
            bad += u32::from(!f.is_finite());
            q_f32[i] = f;
            i += 1;
        }
        if bad != 0 {
            return Err(AttnError::Overflow);
        }

        // `hd as i64` is exact: `hd` is a head width, far below 2^53.
        #[allow(clippy::cast_possible_wrap, reason = "head width < 2^53")]
        let scale = F32::ONE / F32::from_i64(hd as i64).sqrt();

        f32_core::attn_core_f32_strided(q_f32, kx, vx, hd, stride, pos, scale, out_f32)?;

        // Index `while` loop with a failure flag, and the one `Err` raised after it, for the same
        // asserted reason: the `?` inside the `for` body is an early exit from a loop body and is
        // refused with `Breaks to outer loops are not supported yet`. The flag is what holds the
        // early exit outside the loop without changing what `out` carries: the old `?` stored
        // `out[i]` at every index before the first failing conversion and returned at that index,
        // and `while i < hd && ok` stores the same prefix and stops at the same index, raising the
        // same `AttnError` after the loop.
        let mut i = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while i < hd && ok {
            match f32_core::f32_to_q32_checked(out_f32[i]) {
                Ok(v) => out[i] = v,
                Err(e) => {
                    err = Some(e);
                    ok = false;
                }
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }

        Ok(())
    }

    /// Rewind every layer to `m` positions. Causal attention makes the rows at `0..m` independent
    /// of later appends, so the rewound state equals one that stopped at `m`; rows at `m..` stay in
    /// the planes and the next [`KvState::append`] overwrites position `m`.
    ///
    /// # Errors
    ///
    /// Returns [`AttnError::PositionOutOfRange`] when a layer holds fewer than `m` positions; no
    /// layer is changed.
    ///
    /// # Complexity
    ///
    /// `O(n_layer)` time: one check pass and one store pass over the per-layer counters; no
    /// allocation.
    pub fn truncate(&mut self, m: usize) -> Result<(), AttnError> {
        let mut bad: Option<usize> = None;
        let mut i = 0usize;
        while i < self.occupied.len() && bad.is_none() {
            if self.occupied[i] < m {
                bad = Some(self.occupied[i]);
            }
            i += 1;
        }
        if let Some(len) = bad {
            return Err(AttnError::PositionOutOfRange { pos: m, len });
        }
        self.occupied.fill(m);
        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "test harnesses, carrier comparisons and multi-token sequence setups"
)]
mod tests {
    use super::*;

    /// A deterministic Q32.32 word in `[-4·ONE, 4·ONE)` from a 64-bit LCG state.
    fn word(state: &mut u64) -> i64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let wide = i64::try_from(*state >> 1).expect("the top 63 bits fit i64");
        wide % (8 * ONE) - 4 * ONE
    }

    /// The per-component oracle of the output stage: the score, softmax and weight stages of
    /// [`KvState::attn_head`] rebuilt from the live helpers, then one test-only [`value_component`]
    /// call per output component. This is the scan the key-outer production pass replaces.
    fn oracle_head(
        kv: &KvState,
        layer: usize,
        q_head: usize,
        q: &[i64],
        pos: usize,
    ) -> Result<Vec<i64>, AttnError> {
        let hd = kv.head_dim;
        let kv_n = kv.n_head_kv;
        let kvh = q_head / (kv.n_head / kv_n);
        let (kplane, vplane) = match &kv.storage {
            KvStorage::Q32 { k, v } => (&k[layer], &v[layer]),
            KvStorage::F32 { .. } => return Err(AttnError::BadGeometry),
        };
        let slots = pos + 1;
        let mut scores: Vec<i64> = Vec::with_capacity(slots);
        let mut mx = i64::MIN;
        let mut s = 0usize;
        while s <= pos {
            let sc = score_at(kplane, q, kv.scale, kv_n, kvh, hd, s)?;
            if sc > mx {
                mx = sc;
            }
            scores.push(sc);
            s += 1;
        }
        let mut exps: Vec<i64> = Vec::with_capacity(slots);
        let mut total: i128 = 0;
        let mut s = 0usize;
        while s <= pos {
            let e = exp_at(scores[s], mx)?;
            total = total
                .checked_add(i128::from(e))
                .ok_or(AttnError::Overflow)?;
            exps.push(e);
            s += 1;
        }
        if total <= 0 {
            return Err(AttnError::ZeroSum);
        }
        let mut weights: Vec<i64> = Vec::with_capacity(slots);
        let mut s = 0usize;
        while s <= pos {
            weights.push(weight_at(exps[s], total)?);
            s += 1;
        }
        let mut out: Vec<i64> = Vec::with_capacity(hd);
        let mut i = 0usize;
        while i < hd {
            out.push(value_component(vplane, &weights, kv_n, kvh, hd, pos, i)?);
            i += 1;
        }
        Ok(out)
    }

    #[test]
    fn attn_head_key_outer_output_matches_component_oracle() {
        // n_head 4 over n_head_kv 2 gives a GQA group of 2; head_dim 3 is neither 1 nor a power of
        // two, so several query heads read distinct KV rows over non-trivial causal prefixes.
        let mut kv = KvState::with_carrier(1, 4, 2, 3, 6, Carrier::Q32).expect("geometry");
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut s = 0usize;
        while s < 6 {
            let k = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let v = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            assert_eq!(kv.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let q = [word(&mut seed), word(&mut seed), word(&mut seed)];
        let mut pos = 0usize;
        while pos < 6 {
            let mut head = 0usize;
            while head < 4 {
                let expected = oracle_head(&kv, 0, head, &q, pos).expect("per-component oracle");
                let mut got = [0i64; 3];
                kv.attn_head(0, head, &q, pos, &mut got)
                    .expect("key-outer read");
                assert_eq!(got.as_slice(), expected.as_slice(), "pos {pos} head {head}");
                head += 1;
            }
            pos += 1;
        }

        // A non-`head_dim`-wide output refuses before the output pass runs.
        assert_eq!(
            kv.attn_head(0, 0, &q, 0, &mut [0i64; 2]),
            Err(AttnError::RowLength {
                expected: 3,
                got: 2,
            })
        );
        // A position at or past the occupied count refuses before the output pass runs.
        let len = kv.occupied(0).expect("occupied");
        assert_eq!(
            kv.attn_head(0, 0, &q, len, &mut [0i64; 3]),
            Err(AttnError::PositionOutOfRange { pos: len, len })
        );
    }

    #[test]
    fn attn_head_f32_carrier_matches_oracle() {
        let mut kv_q32 = KvState::with_carrier(1, 4, 2, 3, 6, Carrier::Q32).expect("geometry");
        let mut kv_f32 = KvState::with_carrier(1, 4, 2, 3, 6, Carrier::F32).expect("geometry");
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut s = 0usize;
        while s < 6 {
            let k = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let v = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            assert_eq!(kv_q32.append(0, &k, &v), Ok(s));
            assert_eq!(kv_f32.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let q = [word(&mut seed), word(&mut seed), word(&mut seed)];
        let mut pos = 0usize;
        while pos < 6 {
            let mut head = 0usize;
            while head < 4 {
                let mut got_q32 = [0i64; 3];
                let mut got_f32 = [0i64; 3];
                kv_q32
                    .attn_head(0, head, &q, pos, &mut got_q32)
                    .expect("q32 read");
                kv_f32
                    .attn_head(0, head, &q, pos, &mut got_f32)
                    .expect("f32 read");
                for (&g, &e) in got_f32.iter().zip(got_q32.iter()) {
                    let diff = (g - e).abs();
                    assert!(
                        diff <= 2000,
                        "pos {pos} head {head}: got_f32 {g} got_q32 {e} diff {diff}"
                    );
                }
                head += 1;
            }
            pos += 1;
        }
    }

    /// The reciprocal weight path must equal the literal [`weight_at`] oracle (or refuse with the
    /// same `AttnError`) on the window boundary, the `e = 2^32` argmax fallback, xorshift draws,
    /// and the parity ties of each total.
    #[test]
    fn weight_at_recip_matches_weight_at_oracle() {
        let totals: [i128; 7] = [
            2,
            3,
            1 << 31,
            1 << 32,
            (1 << 32) + 1,
            1 << 40,
            i128::from(u64::MAX / 2),
        ];
        let mut x = 0x243F_6A88_85A3_08D3_u64;
        for &total in &totals {
            let recip = recip_u64(total).expect("total is in the reciprocal window");
            let total_i64 = i64::try_from(total).expect("total <= i64::MAX");
            let mut e_values: Vec<i64> = vec![
                0,
                1,
                total_i64 / 2,
                total_i64 - 1,
                total_i64,
                (1 << 32) - 1,
                1 << 32,
                -1,
                i64::MIN,
            ];
            let mut draw = 0usize;
            while draw < 128 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                e_values.push(i64::try_from(x >> 1).expect("the top 63 bits fit i64"));
                draw += 1;
            }
            // The parity ties: `e << 32` congruent to `total/2` mod total, when such an `e` exists
            // in the small search window. `total/2` is the oracle's own rounding bias.
            let half = (total / 2) % total;
            let mut e = 0i64;
            while e < 4096 {
                if (i128::from(e) << 32) % total == half {
                    e_values.push(e);
                }
                e += 1;
            }
            for &e in &e_values {
                let expected = weight_at(e, total);
                let got = weight_at_recip(e, total, recip);
                assert_eq!(got, expected, "total {total} e {e}");
            }
        }
        // `recip_u64` refuses outside its window, and the `None` dispatch is the literal oracle.
        assert!(recip_u64(0).is_none());
        assert!(recip_u64(1).is_none());
        assert!(recip_u64(i128::from(u64::MAX)).is_none());
        let total = i128::from(u64::MAX);
        let e = 7i64;
        let dispatched = match recip_u64(total) {
            Some(r) => weight_at_recip(e, total, r),
            None => weight_at(e, total),
        };
        assert_eq!(dispatched, weight_at(e, total));
    }

    /// The hoisted `first + s·stride` base must equal the checked index oracles [`row_bounds`] and
    /// [`plane_at`] on the rows the output test reads.
    #[test]
    fn hoisted_row_base_matches_plane_index_oracles() {
        let kv = 2usize;
        let hd = 3usize;
        let stride = kv.checked_mul(hd).expect("stride");
        let mut kvh = 0usize;
        while kvh < kv {
            let first = kvh.checked_mul(hd).expect("first");
            let mut s = 0usize;
            while s < 6 {
                let base = first + s * stride;
                let (rb_base, rb_end) = row_bounds(kv, kvh, hd, s).expect("row_bounds");
                assert_eq!(rb_base, base, "kvh {kvh} s {s}");
                assert_eq!(rb_end, base + hd, "kvh {kvh} s {s}");
                let mut i = 0usize;
                while i < hd {
                    assert_eq!(
                        plane_at(kv, kvh, hd, s, i).expect("plane_at"),
                        base + i,
                        "kvh {kvh} s {s} i {i}"
                    );
                    i += 1;
                }
                s += 1;
            }
            kvh += 1;
        }
    }

    /// The round epilogue shared by the two fold-shape helpers: one `round_shift_32` plus the same
    /// `i64` range check [`KvState::attn_head`] applies.
    ///
    /// # Complexity
    ///
    /// `O(hd)` time and space: one `head_dim`-wide output.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
                  immediately before it"
    )]
    fn round_acc(acc: &[i128]) -> Result<Vec<i64>, AttnError> {
        let mut out: Vec<i64> = Vec::with_capacity(acc.len());
        let mut i = 0usize;
        while i < acc.len() {
            let rounded = round_shift_32(acc[i]);
            if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
                return Err(AttnError::Overflow);
            }
            out.push(rounded as i64);
            i += 1;
        }
        Ok(out)
    }

    /// The key-outer output fold in the pre-inline shape: the whole prefix's weights are resolved
    /// into a `Vec` first (the r26 dispatch), then each key's row is read and folded. Test-only
    /// oracle for the inline shape; the production inline path is [`KvState::attn_head`].
    ///
    /// # Complexity
    ///
    /// `O(keys · hd)` time, `O(keys + hd)` space (the weights vector and the accumulator).
    fn weights_vec_fold(
        vplane: &[i64],
        exps: &[i64],
        total: i128,
        kv: usize,
        kvh: usize,
        hd: usize,
    ) -> Result<Vec<i64>, AttnError> {
        let recip = recip_u64(total);
        let mut weights: Vec<i64> = Vec::with_capacity(exps.len());
        let mut s = 0usize;
        while s < exps.len() {
            let w = match recip {
                Some(r) => weight_at_recip(exps[s], total, r)?,
                None => weight_at(exps[s], total)?,
            };
            weights.push(w);
            s += 1;
        }
        let mut acc: Vec<i128> = vec![0i128; hd];
        let mut s = 0usize;
        while s < exps.len() {
            let base = plane_at(kv, kvh, hd, s, 0)?;
            if base > vplane.len().saturating_sub(hd) {
                return Err(AttnError::Overflow);
            }
            let row = &vplane[base..base + hd];
            let w = i128::from(weights[s]);
            let mut i = 0usize;
            while i < hd {
                acc[i] = acc[i]
                    .checked_add(
                        w.checked_mul(i128::from(row[i]))
                            .ok_or(AttnError::Overflow)?,
                    )
                    .ok_or(AttnError::Overflow)?;
                i += 1;
            }
            s += 1;
        }
        round_acc(&acc)
    }

    /// The key-outer output fold in the inline shape: each key's weight is resolved immediately
    /// before its row is read, with no weights buffer. Mirrors the production loop of
    /// [`KvState::attn_head`] so the two shapes can be compared for the same `exps`/`total`.
    ///
    /// # Complexity
    ///
    /// `O(keys · hd)` time, `O(hd)` space (no weights vector).
    fn inline_fold(
        vplane: &[i64],
        exps: &[i64],
        total: i128,
        kv: usize,
        kvh: usize,
        hd: usize,
    ) -> Result<Vec<i64>, AttnError> {
        let recip = recip_u64(total);
        let mut acc: Vec<i128> = vec![0i128; hd];
        let mut s = 0usize;
        while s < exps.len() {
            let w = match recip {
                Some(r) => weight_at_recip(exps[s], total, r)?,
                None => weight_at(exps[s], total)?,
            };
            let base = plane_at(kv, kvh, hd, s, 0)?;
            if base > vplane.len().saturating_sub(hd) {
                return Err(AttnError::Overflow);
            }
            let row = &vplane[base..base + hd];
            let w = i128::from(w);
            let mut i = 0usize;
            while i < hd {
                acc[i] = acc[i]
                    .checked_add(
                        w.checked_mul(i128::from(row[i]))
                            .ok_or(AttnError::Overflow)?,
                    )
                    .ok_or(AttnError::Overflow)?;
                i += 1;
            }
            s += 1;
        }
        round_acc(&acc)
    }

    /// The inline weight fold must equal the weights-`Vec` fold for the same scores/exps/total,
    /// including a total outside the reciprocal window (`recip_u64` returns `None`, so the literal
    /// [`weight_at`] dispatch runs). `attn_head`'s reachable totals are all `>= ONE` (the argmax key
    /// contributes `fixed_exp(0) = ONE`), so its dispatch never returns `None`; the `None` case is
    /// therefore exercised at the fold level, where the two shapes are the only variable.
    #[test]
    fn inline_weight_fold_matches_weights_vec_oracle() {
        // hd 3, kv 2, kvh 1: row `s` starts at `(2·s + 1)·3`, so the two shapes cannot share a
        // trivial base. Three keys fit the 18-word plane.
        let (kv, kvh, hd) = (2usize, 1usize, 3usize);
        let mut vplane = vec![0i64; 3 * kv * hd];
        let mut seed = 0xDEAD_BEEF_1234_5678_u64;
        let mut at = 0usize;
        while at < vplane.len() {
            vplane[at] = word(&mut seed);
            at += 1;
        }
        let cases: [(&[i64], i128); 4] = [
            (&[ONE, ONE, ONE / 2], 5 * i128::from(ONE) / 2),
            (&[ONE, 0, ONE / 4], 5 * i128::from(ONE) / 4),
            // `total = 1` is below `recip_u64`'s lower window edge, so the dispatch takes `None`.
            (&[0, 1], 1),
            // A total above `ONE` but well inside the window.
            (&[3, 5, 8], 16),
        ];
        for (exps, total) in cases {
            let expected = weights_vec_fold(&vplane, exps, total, kv, kvh, hd);
            let got = inline_fold(&vplane, exps, total, kv, kvh, hd);
            assert_eq!(got, expected, "exps {exps:?} total {total}");
            if total < 2 {
                assert!(
                    recip_u64(total).is_none(),
                    "total {total} leaves the window"
                );
            }
        }
    }

    /// The first refusing key and its `AttnError` must be the same in the inline and weights-`Vec`
    /// shapes: a weight refusal at key 1 with key 0 admitted, and a bound refusal at key 1 with a
    /// one-row-short plane. Both shapes scan keys ascending, so both stop at the same key.
    #[test]
    fn first_refusing_key_matches_weights_vec_shape() {
        let (kv, kvh, hd) = (2usize, 1usize, 3usize);
        let mut vplane = vec![0i64; 3 * kv * hd];
        let mut seed = 0x0F1E_2D3C_4B5A_6978_u64;
        let mut at = 0usize;
        while at < vplane.len() {
            vplane[at] = word(&mut seed);
            at += 1;
        }
        // Weight refusal at key 1: key 0's weight is admitted, key 1's `e·2^32` leaves `i64`.
        let exps = [1i64, i64::MAX];
        let total = 2i128;
        assert_eq!(weight_at(exps[0], total), Ok(1i64 << 31));
        assert_eq!(weight_at(exps[1], total), Err(AttnError::Overflow));
        assert_eq!(
            weights_vec_fold(&vplane, &exps, total, kv, kvh, hd),
            Err(AttnError::Overflow)
        );
        assert_eq!(
            inline_fold(&vplane, &exps, total, kv, kvh, hd),
            Err(AttnError::Overflow)
        );

        // Bound refusal at key 1: `short` holds key 0's row but cuts key 1's.
        let short = &vplane[..kvh * hd + hd];
        let exps = [ONE, ONE];
        let total = 2 * i128::from(ONE);
        assert_eq!(
            weights_vec_fold(short, &exps, total, kv, kvh, hd),
            Err(AttnError::Overflow)
        );
        assert_eq!(
            inline_fold(short, &exps, total, kv, kvh, hd),
            Err(AttnError::Overflow)
        );
    }

    #[test]
    fn carrier_explicit_selection() {
        let kv_q32 = KvState::with_carrier(1, 2, 2, 8, 4, Carrier::Q32).expect("geometry q32");
        let kv_f32 = KvState::with_carrier(1, 2, 2, 8, 4, Carrier::F32).expect("geometry f32");
        assert_eq!(kv_q32.carrier(), Carrier::Q32);
        assert_eq!(kv_f32.carrier(), Carrier::F32);
        assert!(matches!(kv_q32.storage, KvStorage::Q32 { .. }));
        assert!(matches!(kv_f32.storage, KvStorage::F32 { .. }));

        let k = [0i64; 16];
        let v = [0i64; 16];
        let mut kv_q32 = kv_q32;
        let mut kv_f32 = kv_f32;
        kv_q32.append(0, &k, &v).expect("append q32");
        kv_f32.append(0, &k, &v).expect("append f32");

        let q = [0i64; 8];
        let mut out_q32 = [0i64; 8];
        let mut out_f32 = [0i64; 8];
        kv_q32
            .attn_head(0, 0, &q, 0, &mut out_q32)
            .expect("read q32");
        kv_f32
            .attn_head(0, 0, &q, 0, &mut out_f32)
            .expect("read f32");
        assert_eq!(out_q32, out_f32);
    }

    #[test]
    fn truncate_and_rollback_hold_for_both_carriers() {
        for &carrier in &[Carrier::Q32, Carrier::F32] {
            let mut kv = KvState::with_carrier(1, 2, 2, 4, 8, carrier).expect("geometry");
            let mut seed = 0x1234_5678_9ABC_DEF0_u64;

            let mut k_tokens = Vec::new();
            let mut v_tokens = Vec::new();
            for _ in 0..6 {
                let k = [
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                ];
                let v = [
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                    word(&mut seed),
                ];
                k_tokens.push(k);
                v_tokens.push(v);
            }

            for s in 0..6 {
                assert_eq!(kv.append(0, &k_tokens[s], &v_tokens[s]), Ok(s));
            }
            assert_eq!(kv.occupied(0), Ok(6));

            // Truncate to 3 positions
            assert_eq!(kv.truncate(3), Ok(()));
            assert_eq!(kv.occupied(0), Ok(3));

            // Append 2 replacement tokens
            let new_k1 = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let new_v1 = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let new_k2 = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let new_v2 = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            assert_eq!(kv.append(0, &new_k1, &new_v1), Ok(3));
            assert_eq!(kv.append(0, &new_k2, &new_v2), Ok(4));
            assert_eq!(kv.occupied(0), Ok(5));

            // Create a fresh kv state that directly appended the exact same sequence
            let mut fresh = KvState::with_carrier(1, 2, 2, 4, 8, carrier).expect("geometry");
            for s in 0..3 {
                assert_eq!(fresh.append(0, &k_tokens[s], &v_tokens[s]), Ok(s));
            }
            assert_eq!(fresh.append(0, &new_k1, &new_v1), Ok(3));
            assert_eq!(fresh.append(0, &new_k2, &new_v2), Ok(4));

            let q = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let mut out_rewound = [0i64; 4];
            let mut out_direct = [0i64; 4];

            kv.attn_head(0, 0, &q, 4, &mut out_rewound)
                .expect("rewound read");
            fresh
                .attn_head(0, 0, &q, 4, &mut out_direct)
                .expect("direct read");

            assert_eq!(
                out_rewound, out_direct,
                "carrier {carrier:?} rewound vs direct mismatch"
            );

            // Truncate past occupied refuses and changes nothing
            assert_eq!(
                kv.truncate(6),
                Err(AttnError::PositionOutOfRange { pos: 6, len: 5 })
            );
            assert_eq!(kv.occupied(0), Ok(5));
        }
    }
}
