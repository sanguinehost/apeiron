//! Attention: the per-layer key/value cache and one causal query head (port-map row 5).

use super::matvec::row_dot_fixed;
use super::value::{div_round_nearest, round_shift_32};
use super::ONE;
use crate::numeric::fixed_exp::fixed_exp;
use crate::numeric::fixed_rsqrt::fixed_rsqrt;

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

/// One `p_s · v_s` term of the output component `i`.
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

impl KvState {
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

        // Output: checked i128 value accumulation, one value-path epilogue per component.
        let vplane = &self.v[layer];
        let mut i = 0usize;
        let mut ok = true;
        let mut err: Option<AttnError> = None;
        while i < hd && ok {
            match value_component(vplane, &weights, kv, kvh, hd, pos, i) {
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
        if let Some(&len) = self.occupied.iter().find(|&&occ| occ < m) {
            return Err(AttnError::PositionOutOfRange { pos: m, len });
        }
        self.occupied.fill(m);
        Ok(())
    }
}
