//! The fine-grained Mixture-of-Experts router and routed FFN in Q32.32 (epic `moe-router`).
//!
//! [`route_topk`] is the exact, deterministic top-`k` selection over the integer router logits:
//! the `k` largest logits, ties broken by the lower expert index. Softmax is strictly monotone, so
//! the top-`k` set is the same on logits and on probabilities; selection therefore never calls
//! `exp` and never rounds, and it is independent of the order in which experts are stored.
//! [`gate_weights`] evaluates the gate weights. With renormalization the `k` selected exponentials
//! are normalised among themselves — the common softmax denominator cancels, so `k` exponentials
//! are evaluated instead of `E`; without it the full softmax is evaluated and its values are read
//! at the selected indices. [`moe_ffn_fixed`] composes the router with the dense expert FFN:
//! `Σ_{e ∈ top-k} w_e · ffn_fixed(expert_e, x)` plus the optional shared expert, every term on the
//! proved dense path.
//!
//! The exponential is [`fixed_exp`] under the attention softmax's exact max shift and underflow
//! saturation: `exp(l - m)` with `l - m < -16·ONE` saturates to `0`, and each weight is the
//! round-to-nearest division `round(e · 2^32 / Σ e)` of the attention weight step. No float appears
//! on the executing path.

use super::ffn::{ffn_fixed, FfnError, ProjWeights};
use super::matvec::{matvec_fixed, MatvecError};
use super::value::{div_round_nearest, round_shift_32};
use super::ONE;
use crate::numeric::fixed_exp::fixed_exp;

/// A refused fixed-point `MoE` router step (epic `moe-router`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoeError {
    /// `k` is zero or exceeds the expert count, so no top-`k` set is defined.
    BadK {
        /// Requested selection width.
        k: usize,
        /// Number of experts, the logit-vector length.
        n_experts: usize,
    },
    /// The activation is empty, so the router projection is undefined.
    EmptyInput,
    /// A selected expert index is at or past the logit count.
    IndexOutOfRange {
        /// Index supplied.
        index: u32,
        /// Logit-vector length.
        n_logits: usize,
    },
    /// A [`fixed_exp`] argument on the shifted-softmax path left its window. The exact maximum
    /// shift makes every argument `<= 0`, and the `-16·ONE` underflow saturation covers the lower
    /// edge, so this is unreachable on the executing path and retained for the extraction.
    ExpOutOfRange,
    /// The softmax total is zero, so the weight normalisation is undefined. The exact maximum
    /// contributes `exp(0) = ONE`, so the total is at least `ONE`; retained for the extraction.
    ZeroSum,
    /// An `i128` product or sum, an `i64` narrowing, or a `u32` expert-index narrowing left its
    /// range.
    Overflow,
    /// The router projection refused through [`MatvecError`].
    Matvec(MatvecError),
    /// An expert or the shared FFN refused through [`FfnError`].
    Ffn(FfnError),
}

/// One expert's three dense `SwiGLU` projections plus its hidden width, the argument
/// [`moe_ffn_fixed`] passes to [`ffn_fixed`].
#[derive(Debug, Clone, Copy)]
pub struct ExpertWeights<'a> {
    /// The gate projection, `n_ff × x.len()`.
    pub gate: ProjWeights<'a>,
    /// The up projection, `n_ff × x.len()`.
    pub up: ProjWeights<'a>,
    /// The down projection, `x.len() × n_ff`.
    pub down: ProjWeights<'a>,
    /// The hidden width `n_ff` of this expert.
    pub n_ff: usize,
}

/// One softmax numerator: the exact max shift, the [`fixed_exp`] window, and the attention
/// softmax's underflow saturation below `-16·ONE`.
fn exp_shifted(v: i64, mx: i64) -> Result<i64, MoeError> {
    let shifted = v.checked_sub(mx).ok_or(MoeError::Overflow)?;
    if shifted < -16 * ONE {
        Ok(0)
    } else {
        fixed_exp(shifted).map_err(|_| MoeError::ExpOutOfRange)
    }
}

/// One Q32.32 gate weight: the round-to-nearest division `round(e·2^32 / total)`, the attention
/// weight step.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `as i64` narrowing is range-checked against i128::from(i64::MAX)/i64::MIN \
              immediately before it; `i64::try_from` emits an unmodeled external the fail-closed \
              extraction audit rejects"
)]
fn weight_of(e: i64, total: i128) -> Result<i64, MoeError> {
    let scaled = i128::from(e)
        .checked_mul(i128::from(ONE))
        .ok_or(MoeError::Overflow)?;
    let p = div_round_nearest(scaled, total);
    if p > i128::from(i64::MAX) || p < i128::from(i64::MIN) {
        return Err(MoeError::Overflow);
    }
    Ok(p as i64)
}

/// The checked `i128` sum of the shifted exponentials of `logits`.
fn sum_shifted(logits: &[i64], mx: i64) -> Result<i128, MoeError> {
    let mut total: i128 = 0;
    let mut i = 0usize;
    let mut ok = true;
    let mut err: Option<MoeError> = None;
    while i < logits.len() && ok {
        match exp_shifted(logits[i], mx) {
            Ok(e) => {
                if let Some(t) = total.checked_add(i128::from(e)) {
                    total = t;
                } else {
                    err = Some(MoeError::Overflow);
                    ok = false;
                }
            }
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
    Ok(total)
}

/// The checked `i128` sum of the shifted exponentials of `logits` at the selected indices.
fn sum_shifted_selected(logits: &[i64], sel: &[u32], mx: i64) -> Result<i128, MoeError> {
    let mut total: i128 = 0;
    let mut j = 0usize;
    let mut ok = true;
    let mut err: Option<MoeError> = None;
    while j < sel.len() && ok {
        match exp_shifted(logits[sel[j] as usize], mx) {
            Ok(e) => {
                if let Some(t) = total.checked_add(i128::from(e)) {
                    total = t;
                } else {
                    err = Some(MoeError::Overflow);
                    ok = false;
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        j += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(total)
}

/// The round-to-nearest gate weights of the selected logits under the shift `mx` and the checked
/// total `total`.
fn weights_shifted(
    logits: &[i64],
    sel: &[u32],
    mx: i64,
    total: i128,
) -> Result<Vec<i64>, MoeError> {
    let mut out: Vec<i64> = Vec::with_capacity(sel.len());
    let mut j = 0usize;
    let mut ok = true;
    let mut err: Option<MoeError> = None;
    while j < sel.len() && ok {
        match exp_shifted(logits[sel[j] as usize], mx) {
            Ok(e) => match weight_of(e, total) {
                Ok(w) => out.push(w),
                Err(e) => {
                    err = Some(e);
                    ok = false;
                }
            },
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        j += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(out)
}

/// The stable top-`k` expert selection of the router (epic `moe-router`, Spec row "selection").
///
/// Returns the `k` expert indices with the largest logits, ordered by `(logit desc, index asc)`:
/// ties break to the lower expert index, so the result is deterministic and independent of the
/// order in which experts are stored. Selection reads the integer logits only; no `exp` and no
/// rounding enters it. Softmax is strictly monotone, so this is also the top-`k` set of the
/// probabilities.
///
/// # Errors
///
/// Returns [`MoeError::BadK`] for `k = 0` or `k > logits.len()`, and [`MoeError::Overflow`] when
/// the logit count exceeds `u32::MAX` so an index cannot be represented (unreachable for a real
/// expert table).
///
/// # Complexity
///
/// `O(E·k)` time in integer comparisons (`k` passes of one maximum scan over `E` logits) and
/// `O(E + k)` space for the chosen mask and the result, where `E = logits.len()`. The two loops
/// carry a `bool` failure flag and stop on the first refusal, so no `?` and no early `return`
/// appears inside either body — the extractable form the pinned Charon/Aeneas fragment admits.
#[allow(
    clippy::cast_possible_truncation,
    reason = "`best < n <= 0xFFFF_FFFF` is established by the up-front count guard, so the `as u32` \
              narrows exactly; `u32::try_from` emits an unmodeled external the fail-closed \
              extraction audit rejects"
)]
pub fn route_topk(logits: &[i64], k: usize) -> Result<Vec<u32>, MoeError> {
    let n = logits.len();
    if k == 0 || k > n {
        return Err(MoeError::BadK { k, n_experts: n });
    }
    if n > 0xFFFF_FFFFusize {
        return Err(MoeError::Overflow);
    }
    let mut chosen: Vec<bool> = vec![false; n];
    let mut sel: Vec<u32> = Vec::with_capacity(k);
    let mut ok = true;
    let mut err: Option<MoeError> = None;
    let mut pass = 0usize;
    while pass < k && ok {
        let (best, have) = best_unchosen(logits, &chosen);
        if have {
            chosen[best] = true;
            sel.push(best as u32);
        } else {
            // Unreachable: `pass < k <= n` leaves at least one unchosen entry in every pass.
            err = Some(MoeError::BadK { k, n_experts: n });
            ok = false;
        }
        pass += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(sel)
}

/// The index of the largest logit not yet chosen, lower index on ties, and whether one exists.
///
/// One scan over `E = logits.len()` entries: `O(E)` time, `O(1)` space. Split out of
/// [`route_topk`] so each extracted function carries a single loop; the pinned Aeneas fragment
/// could not compute the joined loop state when this scan was nested inside the selection loop.
fn best_unchosen(logits: &[i64], chosen: &[bool]) -> (usize, bool) {
    let n = logits.len();
    let mut best = n;
    let mut best_val: i64 = 0;
    let mut j = 0usize;
    while j < n {
        let taken = if j < chosen.len() { chosen[j] } else { true };
        if !taken {
            let v = logits[j];
            if best == n || v > best_val {
                best = j;
                best_val = v;
            }
        }
        j += 1;
    }
    if best < n {
        (best, true)
    } else {
        (0, false)
    }
}

/// The Q32.32 gate weights at the selected expert indices.
///
/// `sel` holds the indices [`route_topk`] returned, in selection order. With `renorm` the weights
/// are the softmax over the selected logits only: the exact maximum shift, the attention softmax's
/// `fixed_exp`, the checked `i128` total and one round-to-nearest division per selected logit.
/// Without `renorm` the full softmax over all logits is evaluated (still `k` divisions, one per
/// selected index) and its values are read at `sel`.
///
/// # Errors
///
/// Returns [`MoeError::BadK`] for an empty `sel`, [`MoeError::IndexOutOfRange`] for a selected
/// index at or past `logits.len()`, and [`MoeError::Overflow`] when a total or a narrow leaves its
/// range. The attention softmax's underflow saturation applies; nothing is clamped.
///
/// # Complexity
///
/// `O(E + k)` time (`E = logits.len()`) and `O(k)` space for the result: the exponentials are
/// evaluated on demand in the weight pass. Both passes carry a `bool` failure flag so no early
/// `return` appears inside a loop.
pub fn gate_weights(logits: &[i64], sel: &[u32], renorm: bool) -> Result<Vec<i64>, MoeError> {
    let n = logits.len();
    if sel.is_empty() {
        return Err(MoeError::BadK { k: 0, n_experts: n });
    }
    let mut j = 0usize;
    let mut ok = true;
    let mut err: Option<MoeError> = None;
    while j < sel.len() && ok {
        if sel[j] as usize >= n {
            err = Some(MoeError::IndexOutOfRange {
                index: sel[j],
                n_logits: n,
            });
            ok = false;
        }
        j += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }

    let mx;
    let total;
    if renorm {
        // The maximum and total are over the selected logits only; the common softmax
        // denominator cancels, so `k` exponentials suffice.
        let mut m = i64::MIN;
        let mut j = 0usize;
        while j < sel.len() {
            let l = logits[sel[j] as usize];
            if l > m {
                m = l;
            }
            j += 1;
        }
        mx = m;
        total = sum_shifted_selected(logits, sel, mx)?;
    } else {
        // The maximum and total are over all logits: the full softmax, read at the selection.
        let mut m = i64::MIN;
        let mut i = 0usize;
        while i < n {
            if logits[i] > m {
                m = logits[i];
            }
            i += 1;
        }
        mx = m;
        total = sum_shifted(logits, mx)?;
    }
    if total <= 0 {
        return Err(MoeError::ZeroSum);
    }
    weights_shifted(logits, sel, mx, total)
}

/// One routed `MoE` FFN token: `router·x`, the top-`k` selection, the gate weights, and
/// `Σ_{e ∈ top-k} w_e · ffn_fixed(expert_e, x)` plus the optional shared expert.
///
/// The router projection is [`matvec_fixed`] over the `experts.len()` rows of `router`; the logits
/// feed [`route_topk`] and [`gate_weights`]. Each selected expert is evaluated by the proved dense
/// [`ffn_fixed`] and combined by a checked `i128` multiply-accumulate `Σ_e w_e·y_e`, reduced once
/// through the value-path epilogue [`round_shift_32`]. The shared expert, always active, is
/// evaluated by the same dense path and added to the reduced routed sum.
///
/// # Errors
///
/// Returns [`MoeError::EmptyInput`] for an empty `x`, [`MoeError::BadK`] from [`route_topk`] when
/// `k = 0` or `k > experts.len()`, [`MoeError::IndexOutOfRange`] from [`gate_weights`],
/// [`MoeError::Matvec`] for a router shape, payload or quant refusal, [`MoeError::Ffn`] for an
/// expert or shared FFN refusal, and [`MoeError::Overflow`] when an accumulation or narrowing
/// leaves its range. Nothing is clamped.
///
/// # Complexity
///
/// `O(E·n_in + E·k + k·n_ff·n_in)` time in checked integer multiply-adds, where `E` is the expert
/// count, `n_in = x.len()`, `k` the selected width and `n_ff` the expert hidden width; `O(n_in)`
/// accumulator, `O(E + k)` for the logits and the selection, and `O(n_ff)` scratch per expert.
#[allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    reason = "the kernel keeps the engine's short names (`k`, `x`, `e`, `i`) so the extracted body \
              reads like `matvec_fixed` and `attn_head`; the one `as i64` narrowing is range-checked \
              against i128::from(i64::MAX)/i64::MIN immediately before it, and `i64::try_from` emits \
              an unmodeled external the fail-closed extraction audit rejects"
)]
pub fn moe_ffn_fixed(
    router: ProjWeights<'_>,
    experts: &[ExpertWeights<'_>],
    shared: Option<ExpertWeights<'_>>,
    k: usize,
    renorm: bool,
    x: &[i64],
) -> Result<Vec<i64>, MoeError> {
    if x.is_empty() {
        return Err(MoeError::EmptyInput);
    }
    let n_in = x.len();
    let n_experts = experts.len();
    let logits =
        matvec_fixed(router.ggml_type, router.raw, n_experts, n_in, x).map_err(MoeError::Matvec)?;
    let sel = route_topk(&logits, k)?;
    let weights = gate_weights(&logits, &sel, renorm)?;

    // Checked `i128` multiply-accumulate of the selected expert outputs.
    let mut acc: Vec<i128> = vec![0; n_in];
    let mut e = 0usize;
    let mut ok = true;
    let mut err: Option<MoeError> = None;
    while e < sel.len() && ok {
        let expert = &experts[sel[e] as usize];
        match ffn_fixed(expert.gate, expert.up, expert.down, expert.n_ff, x) {
            Ok(y) => {
                let w = i128::from(weights[e]);
                let mut i = 0usize;
                while i < n_in && ok {
                    if let Some(prod) = i128::from(y[i]).checked_mul(w) {
                        if let Some(s) = acc[i].checked_add(prod) {
                            acc[i] = s;
                        } else {
                            err = Some(MoeError::Overflow);
                            ok = false;
                        }
                    } else {
                        err = Some(MoeError::Overflow);
                        ok = false;
                    }
                    i += 1;
                }
            }
            Err(fe) => {
                err = Some(MoeError::Ffn(fe));
                ok = false;
            }
        }
        e += 1;
    }
    if let Some(er) = err {
        return Err(er);
    }

    // One value-path epilogue per component of the routed sum.
    let mut out: Vec<i64> = Vec::with_capacity(n_in);
    let mut i = 0usize;
    let mut ok = true;
    let mut err: Option<MoeError> = None;
    while i < n_in && ok {
        let rounded = round_shift_32(acc[i]);
        if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
            err = Some(MoeError::Overflow);
            ok = false;
        } else {
            out.push(rounded as i64);
        }
        i += 1;
    }
    if let Some(er) = err {
        return Err(er);
    }

    // The always-active shared expert, added to the reduced routed sum.
    if let Some(sh) = shared {
        let sy = ffn_fixed(sh.gate, sh.up, sh.down, sh.n_ff, x).map_err(MoeError::Ffn)?;
        let mut i = 0usize;
        let mut ok = true;
        let mut err: Option<MoeError> = None;
        while i < n_in && ok {
            if let Some(v) = out[i].checked_add(sy[i]) {
                out[i] = v;
            } else {
                err = Some(MoeError::Overflow);
                ok = false;
            }
            i += 1;
        }
        if let Some(er) = err {
            return Err(er);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::f16_to_fixed;
    use crate::format::quant::GGML_TYPE_F16;

    /// The largest f16 exponent field, so `moe_f16_val` can reject a non-finite value.
    const MOE_F16_INF_EXP: u16 = 0x1f;

    /// The Q32.32 unit as an `f64`: the denominator that turns a fixed-point value into its real
    /// value in the `f64` reference.
    const MOE_ONE_F64: f64 = 4_294_967_296.0;

    /// The small exact f16 values the random test draws from: `0`, `±0.125`, `±0.25`, `±0.5`,
    /// `±1`, `±2`. Values this small keep the expert `SiLU` arguments inside its exact window.
    const MOE_PALETTE: [u16; 11] = [
        0x0000, 0x3000, 0xB000, 0x3400, 0xB400, 0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000,
    ];

    /// The exact real value of an f16 bit pattern, for the `f64` reference.
    fn moe_f16_val(bits: u16) -> f64 {
        let sign = if bits >> 15 == 1 { -1.0_f64 } else { 1.0_f64 };
        let e = (bits >> 10) & 0x1f;
        let m = bits & 0x3ff;
        if e == 0 {
            sign * f64::from(m) * 2f64.powi(-24)
        } else if e == MOE_F16_INF_EXP {
            f64::NAN
        } else {
            sign * (f64::from(m) + 1024.0) * 2f64.powi(i32::from(e) - 25)
        }
    }

    /// A row-major `F16` payload from explicit bit patterns.
    fn moe_f16_raw(bits: &[u16]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::with_capacity(bits.len() * 2);
        for &b in bits {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    /// A deterministic pseudo-random `u64` (Knuth's LCG), test-only.
    fn moe_lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    /// One palette value drawn from the LCG.
    fn moe_pick(state: &mut u64) -> u16 {
        let idx = usize::try_from(moe_lcg(state) % 11).expect("index < 11");
        MOE_PALETTE[idx]
    }

    /// One expert's weights borrowed from caller-owned payloads.
    fn moe_expert<'a>(
        gate: &'a [u8],
        up: &'a [u8],
        down: &'a [u8],
        n_ff: usize,
    ) -> ExpertWeights<'a> {
        ExpertWeights {
            gate: ProjWeights {
                ggml_type: GGML_TYPE_F16,
                raw: gate,
            },
            up: ProjWeights {
                ggml_type: GGML_TYPE_F16,
                raw: up,
            },
            down: ProjWeights {
                ggml_type: GGML_TYPE_F16,
                raw: down,
            },
            n_ff,
        }
    }

    /// A real-valued row dot against the same operands, for the `f64` reference.
    fn moe_dot_ref(row: &[u16], x: &[f64]) -> f64 {
        let mut s = 0.0_f64;
        let mut i = 0usize;
        while i < row.len() {
            s += moe_f16_val(row[i]) * x[i];
            i += 1;
        }
        s
    }

    /// The fold's `SiLU`, in real arithmetic.
    fn moe_silu_ref(v: f64) -> f64 {
        v / (1.0 + (-v).exp())
    }

    /// One dense `SwiGLU` expert in real arithmetic: `down(silu(gate·x) ⊙ (up·x))`.
    fn moe_ffn_ref(gate: &[u16], up: &[u16], down: &[u16], n_ff: usize, x: &[f64]) -> Vec<f64> {
        let n_in = x.len();
        let h: Vec<f64> = (0..n_ff)
            .map(|i| {
                let g = moe_dot_ref(&gate[i * n_in..(i + 1) * n_in], x);
                let u = moe_dot_ref(&up[i * n_in..(i + 1) * n_in], x);
                moe_silu_ref(g) * u
            })
            .collect();
        (0..n_in)
            .map(|j| moe_dot_ref(&down[j * n_ff..(j + 1) * n_ff], &h))
            .collect()
    }

    /// The softmax in real arithmetic, for the `f64` reference.
    fn moe_softmax_ref(v: &[f64]) -> Vec<f64> {
        let mx = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = v.iter().map(|&x| (x - mx).exp()).collect();
        let total: f64 = ex.iter().sum();
        ex.iter().map(|&e| e / total).collect()
    }

    /// The `(logit desc, index asc)` top-`k` reference over integer logits.
    fn moe_topk_ref(logits: &[i64], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].cmp(&logits[a]).then(a.cmp(&b)));
        idx.truncate(k);
        idx
    }

    /// The `(logit desc, index asc)` top-`k` reference over real logits.
    fn moe_topk_ref_f64(logits: &[f64], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .expect("finite logits")
                .then(a.cmp(&b))
        });
        idx.truncate(k);
        idx
    }

    #[test]
    fn moe_route_topk_matches_reference_and_breaks_ties_low() {
        // Equal logits at indices 0/2/4 and at 3/5: the lower index wins every tie.
        let logits: [i64; 6] = [ONE / 2, -ONE, ONE / 2, 2 * ONE, ONE / 2, 2 * ONE];
        assert_eq!(route_topk(&logits, 3), Ok(vec![3, 5, 0]));
        assert_eq!(moe_topk_ref(&logits, 3), vec![3, 5, 0]);
    }

    #[test]
    fn moe_route_topk_matches_reference_over_a_range_of_k() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let logits: Vec<i64> = (0..40)
            .map(|_| {
                let r = moe_lcg(&mut state) >> 11;
                i64::try_from(r % (7 * 1_000_003) as u64).expect("small") * 1_000_003 - 3 * ONE
            })
            .collect();
        for k in [1usize, 2, 5, 17, 40] {
            let got = route_topk(&logits, k).expect("valid k");
            let want: Vec<u32> = moe_topk_ref(&logits, k)
                .iter()
                .map(|&i| u32::try_from(i).expect("small"))
                .collect();
            assert_eq!(got, want, "k={k}");
        }
    }

    #[test]
    fn moe_route_topk_is_order_independent() {
        let logits: [i64; 8] = [
            5 * ONE,
            -3 * ONE,
            9 * ONE,
            ONE,
            7 * ONE,
            0,
            8 * ONE,
            2 * ONE,
        ];
        let sel = route_topk(&logits, 3).expect("top 3");
        assert_eq!(sel, vec![2, 6, 4]);
        // Permute the stored order; the selected expert values and the mapped-back set are equal.
        let perm: [usize; 8] = [3, 0, 7, 2, 5, 1, 6, 4];
        let permuted: Vec<i64> = perm.iter().map(|&i| logits[i]).collect();
        let sel_p = route_topk(&permuted, 3).expect("top 3 permuted");
        let mut mapped: Vec<usize> = sel_p
            .iter()
            .map(|&i| perm[usize::try_from(i).expect("small")])
            .collect();
        let mut orig: Vec<usize> = sel
            .iter()
            .map(|&i| usize::try_from(i).expect("small"))
            .collect();
        mapped.sort_unstable();
        orig.sort_unstable();
        assert_eq!(mapped, orig);
        let values: Vec<i64> = sel_p
            .iter()
            .map(|&i| permuted[usize::try_from(i).expect("small")])
            .collect();
        let want_values: Vec<i64> = sel
            .iter()
            .map(|&i| logits[usize::try_from(i).expect("small")])
            .collect();
        assert_eq!(values, want_values);
    }

    #[test]
    fn moe_gate_weights_renorm_sums_to_one_within_k_units() {
        let logits: [i64; 5] = [ONE / 3, -7 * ONE / 10, 11 * ONE / 10, 0, 9 * ONE / 10];
        for k in [1usize, 2, 3, 5] {
            let sel = route_topk(&logits, k).expect("valid k");
            let w = gate_weights(&logits, &sel, true).expect("renorm weights");
            assert!(w.iter().all(|&x| x >= 0), "negative weight k={k}");
            let sum: i64 = w.iter().sum();
            assert!(
                (sum - ONE).abs() <= i64::try_from(k).expect("small"),
                "k={k} sum={sum} ONE={ONE}"
            );
        }
    }

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        reason = "the fixed-point weights and logits are read as f64 only to compare against the \
                  f64 reference softmax; magnitudes are < 2^40, so the cast error is far below the \
                  1e-6 tolerance"
    )]
    fn moe_gate_weights_renorm_matches_selected_softmax_reference() {
        let logits: [i64; 4] = [4 * ONE / 10, -2 * ONE / 10, ONE, ONE / 10];
        let sel = route_topk(&logits, 3).expect("top 3");
        let w = gate_weights(&logits, &sel, true).expect("renorm weights");
        // Reference: softmax over the selected real logits, in f64.
        let sel_f: Vec<f64> = sel
            .iter()
            .map(|&i| {
                let v = usize::try_from(i).expect("small");
                logits[v] as f64 / MOE_ONE_F64
            })
            .collect();
        let mx = sel_f.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = sel_f.iter().map(|&x| (x - mx).exp()).collect();
        let total: f64 = ex.iter().sum();
        for (j, &e) in ex.iter().enumerate() {
            let want = e / total;
            let got = w[j] as f64 / MOE_ONE_F64;
            assert!((got - want).abs() < 1e-6, "j={j} got={got} want={want}");
        }
    }

    #[test]
    fn moe_gate_weights_full_and_renorm_paths_agree_on_selected_ratios() {
        let logits: [i64; 5] = [4 * ONE / 10, -2 * ONE / 10, ONE, ONE / 10, 6 * ONE / 10];
        let sel = route_topk(&logits, 3).expect("top 3");
        let q = gate_weights(&logits, &sel, true).expect("renorm");
        let p = gate_weights(&logits, &sel, false).expect("full");
        assert!(p.iter().all(|&x| x > 0), "full-softmax value underflowed");
        // q_j/p_j is the same normaliser ratio for every selected index; with one rounding unit
        // per weight the cross-product difference is bounded by 2·ONE.
        for j in 0..sel.len() {
            for i in j + 1..sel.len() {
                let lhs = i128::from(q[j]) * i128::from(p[i]);
                let rhs = i128::from(q[i]) * i128::from(p[j]);
                assert!(
                    (lhs - rhs).abs() <= 2 * i128::from(ONE),
                    "j={j} i={i} lhs={lhs} rhs={rhs}"
                );
            }
        }
    }

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        reason = "the fixed-point i64 outputs are read as f64 only to compare against the f64 \
                  reference; magnitudes are < 2^40, so the cast error is far below the tolerance. \
                  The test is one flat end-to-end fixture; splitting it would hide the shared \
                  weight generation from the reference it feeds"
    )]
    fn moe_ffn_matches_f64_reference_with_random_f16_experts() {
        const N_IN: usize = 2;
        const N_FF: usize = 2;
        const E: usize = 3;
        const K: usize = 2;

        // 1.0, 0.5.
        let x_bits: [u16; N_IN] = [0x3C00, 0x3800];
        // Router logits are separated (1.0, 0.25, -0.5), so quantization cannot flip the top-2.
        let router_bits: [u16; E * N_IN] = [
            0x3C00, 0x0000, // 1.0
            0x3400, 0x0000, // 0.25
            0xB800, 0x0000, // -0.5
        ];
        let router_raw = moe_f16_raw(&router_bits);

        let mut state = 0x0BAD_F00D_1234_5678u64;
        let mut expert_bits: Vec<([u16; N_FF * N_IN], [u16; N_FF * N_IN], [u16; N_IN * N_FF])> =
            Vec::with_capacity(E);
        let mut expert_raw: Vec<[Vec<u8>; 3]> = Vec::with_capacity(E);
        for _ in 0..E {
            let mut g = [0u16; N_FF * N_IN];
            let mut u = [0u16; N_FF * N_IN];
            let mut d = [0u16; N_IN * N_FF];
            for slot in &mut g {
                *slot = moe_pick(&mut state);
            }
            for slot in &mut u {
                *slot = moe_pick(&mut state);
            }
            for slot in &mut d {
                *slot = moe_pick(&mut state);
            }
            expert_raw.push([moe_f16_raw(&g), moe_f16_raw(&u), moe_f16_raw(&d)]);
            expert_bits.push((g, u, d));
        }
        let mut sg = [0u16; N_FF * N_IN];
        let mut su = [0u16; N_FF * N_IN];
        let mut sd = [0u16; N_IN * N_FF];
        for slot in &mut sg {
            *slot = moe_pick(&mut state);
        }
        for slot in &mut su {
            *slot = moe_pick(&mut state);
        }
        for slot in &mut sd {
            *slot = moe_pick(&mut state);
        }
        let shared_raw = [moe_f16_raw(&sg), moe_f16_raw(&su), moe_f16_raw(&sd)];

        let experts: Vec<ExpertWeights> = expert_raw
            .iter()
            .map(|r| moe_expert(&r[0], &r[1], &r[2], N_FF))
            .collect();
        let shared = moe_expert(&shared_raw[0], &shared_raw[1], &shared_raw[2], N_FF);

        let x_fixed: Vec<i64> = x_bits
            .iter()
            .map(|&b| f16_to_fixed(b).expect("finite"))
            .collect();
        let xf: Vec<f64> = x_bits.iter().map(|&b| moe_f16_val(b)).collect();

        // The fixed-point selection must agree with the real-valued reference selection.
        let logits_fixed =
            matvec_fixed(GGML_TYPE_F16, &router_raw, E, N_IN, &x_fixed).expect("router logits");
        let sel_fixed: Vec<usize> = route_topk(&logits_fixed, K)
            .expect("top k")
            .iter()
            .map(|&i| usize::try_from(i).expect("small"))
            .collect();
        let logits_ref: Vec<f64> = (0..E)
            .map(|e| moe_dot_ref(&router_bits[e * N_IN..(e + 1) * N_IN], &xf))
            .collect();
        let sel_ref = moe_topk_ref_f64(&logits_ref, K);
        assert_eq!(sel_fixed, sel_ref, "selection disagrees with the reference");

        let router = ProjWeights {
            ggml_type: GGML_TYPE_F16,
            raw: &router_raw,
        };
        for renorm in [true, false] {
            let probs = moe_softmax_ref(&logits_ref);
            let mut w_ref: Vec<f64> = sel_ref.iter().map(|&i| probs[i]).collect();
            if renorm {
                let sum: f64 = w_ref.iter().sum();
                for w in &mut w_ref {
                    *w /= sum;
                }
            }
            let mut want = [0.0_f64; N_IN];
            for (j, &i) in sel_ref.iter().enumerate() {
                let out = moe_ffn_ref(
                    &expert_bits[i].0,
                    &expert_bits[i].1,
                    &expert_bits[i].2,
                    N_FF,
                    &xf,
                );
                for (dst, src) in want.iter_mut().zip(&out) {
                    *dst += w_ref[j] * src;
                }
            }
            let shared_out = moe_ffn_ref(&sg, &su, &sd, N_FF, &xf);
            for (dst, src) in want.iter_mut().zip(&shared_out) {
                *dst += src;
            }
            let got = moe_ffn_fixed(router, &experts, Some(shared), K, renorm, &x_fixed)
                .expect("routed ffn");
            for (d, &expected) in want.iter().enumerate() {
                let value = got[d] as f64 / MOE_ONE_F64;
                assert!(
                    (value - expected).abs() < 1e-3,
                    "renorm={renorm} d={d} got={value} want={expected}"
                );
            }
        }
    }

    #[test]
    fn moe_router_refuses_bad_k_shapes_indices_and_kernels() {
        // k = 0 and k > E.
        assert_eq!(
            route_topk(&[], 0),
            Err(MoeError::BadK { k: 0, n_experts: 0 })
        );
        assert_eq!(
            route_topk(&[ONE, 2 * ONE, 3 * ONE], 0),
            Err(MoeError::BadK { k: 0, n_experts: 3 })
        );
        assert_eq!(
            route_topk(&[ONE, 2 * ONE, 3 * ONE], 4),
            Err(MoeError::BadK { k: 4, n_experts: 3 })
        );
        // Empty selection and an out-of-range index refuse.
        assert_eq!(
            gate_weights(&[ONE, 2 * ONE], &[], true),
            Err(MoeError::BadK { k: 0, n_experts: 2 })
        );
        assert_eq!(
            gate_weights(&[ONE, 2 * ONE], &[2], true),
            Err(MoeError::IndexOutOfRange {
                index: 2,
                n_logits: 2
            })
        );
        // Empty activation, then an empty expert table.
        let empty: [ExpertWeights; 0] = [];
        let unused = ProjWeights {
            ggml_type: GGML_TYPE_F16,
            raw: &[],
        };
        assert_eq!(
            moe_ffn_fixed(unused, &empty, None, 1, true, &[]),
            Err(MoeError::EmptyInput)
        );
        assert_eq!(
            moe_ffn_fixed(unused, &empty, None, 1, true, &[ONE]),
            Err(MoeError::BadK { k: 1, n_experts: 0 })
        );
        // The router projection refuses an unsupported quant and a short payload.
        let x = [ONE, ONE];
        let expert_raw = moe_f16_raw(&[0x3C00, 0x0000, 0x0000, 0x3C00]);
        let down_raw = moe_f16_raw(&[0x3C00, 0x0000, 0x0000, 0x3C00]);
        let expert = moe_expert(&expert_raw, &expert_raw, &down_raw, 2);
        let bad_quant = ProjWeights {
            ggml_type: 0xFFFF,
            raw: &[],
        };
        assert!(matches!(
            moe_ffn_fixed(bad_quant, std::slice::from_ref(&expert), None, 1, true, &x),
            Err(MoeError::Matvec(MatvecError::UnsupportedQuant { .. }))
        ));
        let short = ProjWeights {
            ggml_type: GGML_TYPE_F16,
            raw: &[0u8, 0u8],
        };
        assert!(matches!(
            moe_ffn_fixed(short, std::slice::from_ref(&expert), None, 1, true, &x),
            Err(MoeError::Matvec(MatvecError::ShortPayload { .. }))
        ));
        // A degenerate expert geometry refuses through the dense FFN.
        let router_raw = moe_f16_raw(&[0x3C00, 0x0000]);
        let router = ProjWeights {
            ggml_type: GGML_TYPE_F16,
            raw: &router_raw,
        };
        let degenerate = moe_expert(&expert_raw, &expert_raw, &down_raw, 0);
        assert_eq!(
            moe_ffn_fixed(router, std::slice::from_ref(&degenerate), None, 1, true, &x),
            Err(MoeError::Ffn(FfnError::BadGeometry))
        );
    }
}
