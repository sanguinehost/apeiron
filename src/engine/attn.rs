//! Attention: the per-layer key/value cache and one causal query head (port-map row 5).

use super::matvec::row_dot_fixed;
use super::value::{div_round_nearest, round_shift_32};
use super::ONE;
use crate::numeric::fixed_exp::fixed_exp;
use crate::numeric::fixed_rsqrt::fixed_rsqrt;
use std::cell::Cell;

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

/// The single-token key/value cache of port-map row 5: one flat Q32.32 plane per layer.
///
/// Keys and values are stored position-major, so position `p` carries all `n_head_kv` rows
/// contiguously and the row for `(p, kv_head)` is the `head_dim` values at
/// `(p * n_head_kv + kv_head) * head_dim`. One KV head serves `n_head / n_head_kv` query heads
/// (GQA). [`KvState::append`] writes a whole token's KV rows at the next position of a layer and
/// returns that position; [`KvState::attn_head`] reads one query head over the causal prefix
/// `0..=pos`.
#[derive(Debug, Clone)]
pub struct KvState {
    n_layer: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    cap: usize,
    scale: i64,
    occupied: Vec<usize>,
    k: Vec<Vec<i64>>,
    v: Vec<Vec<i64>>,
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

/// The checked `[base, base + head_dim)` bounds of the `head_dim`-wide row of key `s` in a
/// position-major plane: the same three `checked.*` index steps as [`plane_at`], with the row end
/// as one more. The key-outer output pass reads the whole row from `base`, so it computes the
/// index product once per key instead of once per `(key, component)`.
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
        let kplane = &self.k[layer];
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
    /// `m`, in `tile`-key groups. The tile boundaries only re-parenthesise the same `i128` sums,
    /// so the returned `den`/`num` are the same for every `tile`.
    ///
    /// # Complexity
    ///
    /// `O((pos + 1) · head_dim)` time in checked `i128` multiply-adds and one [`fixed_exp`] per
    /// key, `O(head_dim)` space for `num` plus `O(1)` for the group cursor; the op count is
    /// independent of `tile`.
    fn fused_fold(
        &self,
        layer: usize,
        kvh: usize,
        scores: &[i64],
        m: i64,
        tile: usize,
    ) -> Result<AttnFusedState, AttnError> {
        let hd = self.head_dim;
        let kv = self.n_head_kv;
        let vplane = &self.v[layer];
        let count = scores.len();
        let mut den: i128 = 0;
        let mut num: Vec<i128> = vec![0i128; hd];
        let mut start = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while start < count && ok {
            let end = match start.checked_add(tile) {
                Some(last) => last.min(count),
                None => count,
            };
            let mut key = start;
            while key < end && ok {
                match exp_at(scores[key], m) {
                    Ok(exp_word) => {
                        if let Some(next) = den.checked_add(i128::from(exp_word)) {
                            den = next;
                        } else {
                            err = Some(AttnError::Overflow);
                            ok = false;
                        }
                        let mut comp = 0usize;
                        while comp < hd && ok {
                            match plane_at(kv, kvh, hd, key, comp) {
                                Ok(at) => {
                                    if let Some(&vb) = vplane.get(at) {
                                        if let Some(term) =
                                            i128::from(exp_word).checked_mul(i128::from(vb))
                                        {
                                            if let Some(next) = num[comp].checked_add(term) {
                                                num[comp] = next;
                                            } else {
                                                err = Some(AttnError::Overflow);
                                                ok = false;
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
                                Err(geo_err) => {
                                    err = Some(geo_err);
                                    ok = false;
                                }
                            }
                            comp += 1;
                        }
                    }
                    Err(exp_err) => {
                        err = Some(exp_err);
                        ok = false;
                    }
                }
                key += 1;
            }
            start = end;
        }
        if let Some(e) = err {
            return Err(e);
        }
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

    /// The fused `(m, den, num)` state of the causal prefix at the fixed shift `m`, folded in
    /// `tile`-key groups. `m` is caller-supplied: the online read passes the exact maximum from
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
    /// `head_dim` values per head and at most `cap` positions per layer.
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
        let blank = vec![0i64; plane];
        let k = vec![blank.clone(); n_layer];
        let v = vec![blank; n_layer];
        Ok(KvState {
            n_layer,
            n_head,
            n_head_kv,
            head_dim,
            cap,
            scale,
            occupied: vec![0usize; n_layer],
            k,
            v,
        })
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
        self.k[layer][base..end].copy_from_slice(k_rows);
        self.v[layer][base..end].copy_from_slice(v_rows);
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
    /// `O(pos + 1)` scratch space for the scores, exponentials and weights; `out` is `head_dim`
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
        let slots = pos.checked_add(1).ok_or(AttnError::Overflow)?;

        // Scores: one Q32.32 dot per causal key, then one epilogue against the load-time scale.
        // Each loop carries a `bool` failure flag and stops on the first refusal, so no early
        // return appears inside a body; the refusals, arithmetic and order are unchanged.
        let mut scores: Vec<i64> = Vec::with_capacity(slots);
        let mut mx = i64::MIN;
        let kplane = &self.k[layer];
        let mut s = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while s <= pos && ok {
            match score_at(kplane, q, self.scale, kv, kvh, hd, s) {
                Ok(score) => {
                    if score > mx {
                        mx = score;
                    }
                    scores.push(score);
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

        // Weights: one round-to-nearest division per causal key.
        let mut weights: Vec<i64> = Vec::with_capacity(slots);
        let mut s = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while s <= pos && ok {
            match weight_at(exps[s], total) {
                Ok(w) => weights.push(w),
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

        // Output: one key-outer pass. Each key's `head_dim` row is read contiguously once and its
        // `weights[s]·v[s, i]` products accumulate into the independent per-component checked
        // `i128` sums. For each `i` the terms are the same checked products in the same ascending
        // `s` order as the per-component oracle, so every `acc[i]` is the identical sum and the
        // single `round_shift_32` epilogue plus its `i64` range check are unchanged; only the
        // interleaving of independent sums changes (integer addition is associative and
        // commutative). One `bool` failure flag per loop, and no early return inside a body, keep
        // the pinned extraction fragment's shape.
        let vplane = &self.v[layer];
        let mut acc: Vec<i128> = vec![0i128; hd];
        let mut s = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while s <= pos && ok {
            match row_bounds(kv, kvh, hd, s) {
                Ok((base, end)) => {
                    let row = &vplane[base..end];
                    let w = i128::from(weights[s]);
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
        if let Some(&len) = self.occupied.iter().find(|&&occ| occ < m) {
            return Err(AttnError::PositionOutOfRange { pos: m, len });
        }
        self.occupied.fill(m);
        Ok(())
    }
}

#[cfg(test)]
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
        let kplane = &kv.k[layer];
        let vplane = &kv.v[layer];
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
        let mut kv = KvState::new(1, 4, 2, 3, 6).expect("geometry");
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
}
