//! Exact tensor-parallel FFN: [`ffn_tp`] runs one
//! [`ffn_q8act`] FFN (`SwiGLU`) block as a column cover of the hidden dimension in one
//! the process pool dispatch.
//!
//! Task `i` owns hidden slice `S_i`, aligned to the down projection's block width; it computes the
//! gate and up rows of `S_i` with the chart fast row kernels over the shared `quant_act(x)`, applies
//! the same `SwiGLU` elementwise, quantizes its slice locally (the quantizer is per 32-wide block,
//! so an aligned slice quantizes locally), and accumulates the down projection's columns of `S_i`
//! into an exact `i128` partial vector. The single reduction `Σ_i partial_i` then takes one
//! [`round_shift_32`] and one `i64` range check per output row — the epilogue `q8act_rows` applies
//! to the whole-row accumulator.
//!
//! The licensing statements are `Numeric/Matvec/ColumnCover.lean` (`split_ffn_exact`: the unsplit FFN
//! `r (D·Q(σ(Gx, Ux)))` equals the slice reduction for any partition of the hidden blocks;
//! `column_cover_round`: one rounding over a cover equals the rounding of the covered sum;
//! `quant_locality`: a 32-aligned slice quantizes locally), with the per-block quantizer itself
//! `Numeric/Matvec/ChartDot.lean` `quantBlock`/`dequantBlock`.
//!
//! The sliced path is taken only when all three projections' plans are `Fused` and fully fast and
//! every slice stays on its non-fallback path; any other case — a refusal, a non-finite weight
//! scale, a bound without a proof, a short payload, a non-sliced family — delegates to
//! [`ffn_q8act`], whose result is the contract by definition, so no `Err` value can differ.
#![allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    dead_code,
    reason = "`ffn_tp` is the whole dispatch; its named stages are one body, and the index algebra \
              (`o` output row, `t` slice, `q`/`h` activations) mirrors the factorization. The entry \
              point lands in this wave; its step-integration consumer (L8-T3, `step.rs`) is a \
              separate task"
)]

use crate::engine::chart::ffn::ffn_q8act;
use crate::engine::chart::pool::{Excl, Flag};
use crate::engine::chart::task::IndexTask;
use crate::engine::chart::{
    q8act_plan, q8act_row_i128_fast, q8act_rows, quant_act, Q8ActFast, Q8ActPlan, QuantAct,
};
use crate::engine::ffn::{silu_fixed, FfnError, ProjWeights};
use crate::engine::matvec::MatvecError;
use crate::engine::value::round_shift_32;
use crate::engine::EngineError;
use crate::format::quant;

/// The `SwiGLU` element of one hidden index, the same arithmetic
/// [`swiglu_hidden`](crate::engine::chart::ffn::swiglu_hidden) applies:
/// `round(silu(gate)·up / 2^32)` with the `i128` product and the `i64` narrowing checked.
///
/// # Errors
///
/// [`FfnError::Overflow`] when the product or the narrowing leaves its range, and
/// [`silu_fixed`]'s own refusal.
fn swiglu_one(gate: i64, up: i64) -> Result<i64, FfnError> {
    let s = silu_fixed(gate)?;
    let prod = i128::from(s)
        .checked_mul(i128::from(up))
        .ok_or(FfnError::Overflow)?;
    i64::try_from(round_shift_32(prod)).map_err(|_| FfnError::Overflow)
}

/// One hidden slice's outputs: the local quantization of the slice's `SwiGLU` hidden and its exact
/// `i128` down-projection partial.
struct SliceData {
    /// The slice-local `quant_act` of the slice's `SwiGLU` hidden.
    quant: QuantAct,
    /// `partial[o]` is the slice's exact `i128` down accumulator for output row `o`.
    partial: Vec<i128>,
}

/// One FFN (`SwiGLU`) block as a column cover in one pool dispatch, bit-identical to
/// [`ffn_q8act`] including its `Err` values.
///
/// The hidden dimension `[0, n_ff)` is cut into slices aligned to the down projection's block
/// width `bw` — 32 for `Q5_0`/`Q8_0`, 256 for `Q4_K`/`Q6_K`. Task `i` owns columns
/// `[i·bw, min((i+1)·bw, n_ff))`, computes their `n_ff`-row gate and up outputs with the chart fast
/// row kernels, applies [`swiglu_one`] to its slice, quantizes the slice locally and accumulates
/// the down projection's columns of the slice into one `n_embd`-long exact `i128` partial. The
/// reduction over the slices then takes one [`round_shift_32`] and one `i64` range check per output
/// row, exactly as `q8act_rows` does.
///
/// # Errors
///
/// [`FfnError::BadGeometry`] for a zero `n_ff` or an empty `x`; otherwise the first refusal
/// [`ffn_q8act`] returns — a projection shape, payload, quant or accumulator refusal, or
/// [`FfnError::Overflow`] from the elementwise pass.
///
/// # Complexity
///
/// `O(n_ff·n_embd)` time in exact integer multiply-adds over the three projections plus
/// `O(n_ff)` for the elementwise pass, cut into `⌈n_ff/bw⌉` pool tasks; `O(n_embd)` output and
/// `O(n_ff + n_embd·⌈n_ff/bw⌉)` scratch across the tasks.
pub(crate) fn ffn_tp(
    gate: ProjWeights<'_>,
    up: ProjWeights<'_>,
    down: ProjWeights<'_>,
    n_ff: usize,
    x: &[i64],
) -> Result<Vec<i64>, FfnError> {
    let n_embd = x.len();
    let delegate = || ffn_q8act(gate, up, down, n_ff, x);
    if n_ff == 0 || n_embd == 0 {
        return Err(FfnError::BadGeometry);
    }
    // The down projection's column slices are its own block width; only the four families with a
    // fully-fast chart row are sliced, everything else is `ffn_q8act` itself.
    let Some((down_block_width, down_block_bytes)) = quant::block_geom(down.ggml_type) else {
        return delegate();
    };
    if !matches!(
        down.ggml_type,
        quant::GGML_TYPE_Q8_0
            | quant::GGML_TYPE_Q5_0
            | quant::GGML_TYPE_Q4_K
            | quant::GGML_TYPE_Q6_K
    ) {
        return delegate();
    }
    let Some(q) = quant_act(x) else {
        return Err(FfnError::Matvec(MatvecError::Dequant(
            EngineError::FixedOverflow,
        )));
    };
    let gate_plan = q8act_plan(gate.ggml_type, n_embd, &q);
    let up_plan = q8act_plan(up.ggml_type, n_embd, &q);
    let (Q8ActPlan::Fused(gfast), Q8ActPlan::Fused(ufast)) = (&gate_plan, &up_plan) else {
        return delegate();
    };
    if !gfast.fast || !ufast.fast {
        return delegate();
    }
    let Some((gqk, gbb)) = quant::block_geom(gate.ggml_type) else {
        return delegate();
    };
    let Some((uqk, ubb)) = quant::block_geom(up.ggml_type) else {
        return delegate();
    };
    let (Some(gate_row_bytes), Some(up_row_bytes), Some(down_row_bytes)) = (
        n_embd.div_ceil(gqk).checked_mul(gbb),
        n_embd.div_ceil(uqk).checked_mul(ubb),
        n_ff.div_ceil(down_block_width)
            .checked_mul(down_block_bytes),
    ) else {
        return delegate();
    };
    let tasks = n_ff.div_ceil(down_block_width);
    let slots: Excl<Vec<(usize, SliceData)>> = Excl::new(Vec::with_capacity(tasks));
    let dirty = crate::engine::chart::pool::Flag::new();
    let ctx = SliceCtx {
        gate,
        up,
        down,
        geom: SliceGeom {
            n_embd,
            n_ff,
            gate_row_bytes,
            up_row_bytes,
            down_row_bytes,
            down_block_width,
            down_block_bytes,
        },
        q: &q,
        gfast,
        ufast,
        dirty: &dirty,
        slots: &slots,
    };
    crate::engine::chart::pool::run_global(tasks, &ctx);
    if dirty.get() {
        return delegate();
    }
    let mut results: Vec<Option<SliceData>> = Vec::with_capacity(tasks);
    let mut t = 0usize;
    while t < tasks {
        results.push(None);
        t += 1;
    }
    for (t, d) in slots.into_inner() {
        results[t] = Some(d);
    }
    // The whole-vector quantization is the slices' local quantizations concatenated: the quantizer
    // is per 32-wide block and every slice starts on a 32-aligned boundary, so the whole-call plan
    // of `ffn_q8act`'s down projection is decided without re-quantizing the hidden. A `Fused` fast
    // plan is exactly the case the slice partials reproduce bit for bit.
    let mut scales: Vec<i64> = Vec::with_capacity(n_ff / 32);
    let mut codes: Vec<i8> = Vec::with_capacity(n_ff);
    for r in &results {
        let Some(d) = r else {
            return delegate();
        };
        scales.extend_from_slice(d.quant.scales());
        codes.extend_from_slice(d.quant.codes());
    }
    let down_plan = q8act_plan(down.ggml_type, n_ff, &QuantAct::from_parts(scales, codes));
    if !matches!(&down_plan, Q8ActPlan::Fused(f) if f.fast) {
        return delegate();
    }
    let Some(need) = n_embd.checked_mul(down_row_bytes) else {
        return delegate();
    };
    if down.raw.len() < need {
        return delegate();
    }
    let mut y: Vec<i64> = Vec::with_capacity(n_embd);
    let mut out_row = 0usize;
    while out_row < n_embd {
        let mut acc: i128 = 0;
        let mut i = 0usize;
        while i < results.len() {
            let Some(d) = &results[i] else {
                return delegate();
            };
            let Some(v) = acc.checked_add(d.partial[out_row]) else {
                return delegate();
            };
            acc = v;
            i += 1;
        }
        let Ok(v) = i64::try_from(round_shift_32(acc)) else {
            return Err(FfnError::Matvec(MatvecError::OutputOverflow));
        };
        y.push(v);
        out_row += 1;
    }
    Ok(y)
}

/// The exact geometry of one [`ffn_tp`] slice task: the hidden and output extents and the row and
/// block byte counts of the three projections. Numbers only, so it is passed by value.
#[derive(Clone, Copy)]
struct SliceGeom {
    n_embd: usize,
    n_ff: usize,
    gate_row_bytes: usize,
    up_row_bytes: usize,
    down_row_bytes: usize,
    down_block_width: usize,
    down_block_bytes: usize,
}

/// The shared inputs of one [`ffn_tp`] dispatch, borrowed for the pool.
struct SliceCtx<'a> {
    gate: ProjWeights<'a>,
    up: ProjWeights<'a>,
    down: ProjWeights<'a>,
    geom: SliceGeom,
    q: &'a QuantAct,
    gfast: &'a Q8ActFast,
    ufast: &'a Q8ActFast,
    dirty: &'a Flag,
    slots: &'a Excl<Vec<(usize, SliceData)>>,
}

impl IndexTask for SliceCtx<'_> {
    fn run(&self, t: usize) {
        ffn_slice_task(
            t,
            self.geom,
            self.gate.raw,
            self.gate.ggml_type,
            self.up.raw,
            self.up.ggml_type,
            self.down.raw,
            self.down.ggml_type,
            self.q,
            self.gfast,
            self.ufast,
            self.dirty,
            self.slots,
        );
    }
}

/// The body of one [`ffn_tp`] slice task: hidden slice `t`, computed and accumulated into `slots`.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the task's shared inputs travel as one argument each so the trait method is one call"
)]
fn ffn_slice_task(
    t: usize,
    geom: SliceGeom,
    gate_raw: &[u8],
    gate_type: u32,
    up_raw: &[u8],
    up_type: u32,
    down_raw: &[u8],
    down_type: u32,
    q: &QuantAct,
    gfast: &Q8ActFast,
    ufast: &Q8ActFast,
    dirty: &Flag,
    slots: &Excl<Vec<(usize, SliceData)>>,
) {
    let c0 = t * geom.down_block_width;
    let c1 = (c0 + geom.down_block_width).min(geom.n_ff);
    let cols = c1 - c0;
    // The slice's gate and up rows, on the same row kernels the whole-projection call uses
    // (including the cross-row tiled cores), so the sliced and unsliced rows are equal.
    let (Some(gbytes), Some(ubytes)) = (
        gate_raw.get(c0 * geom.gate_row_bytes..c1 * geom.gate_row_bytes),
        up_raw.get(c0 * geom.up_row_bytes..c1 * geom.up_row_bytes),
    ) else {
        dirty.set();
        return;
    };
    let need_g = cols * geom.gate_row_bytes;
    let need_u = cols * geom.up_row_bytes;
    let (Ok(gate_rows), Ok(up_rows)) = (
        q8act_rows(
            gate_type,
            gbytes,
            cols,
            geom.n_embd,
            geom.gate_row_bytes,
            need_g,
            q,
            gfast,
        ),
        q8act_rows(
            up_type,
            ubytes,
            cols,
            geom.n_embd,
            geom.up_row_bytes,
            need_u,
            q,
            ufast,
        ),
    ) else {
        dirty.set();
        return;
    };
    let mut hidden: Vec<i64> = Vec::with_capacity(cols);
    let mut j = 0usize;
    let mut ok = true;
    while j < cols && ok {
        match swiglu_one(gate_rows[j], up_rows[j]) {
            Ok(hv) => hidden.push(hv),
            Err(_) => ok = false,
        }
        j += 1;
    }
    if !ok {
        dirty.set();
        return;
    }
    let Some(qh) = quant_act(&hidden) else {
        dirty.set();
        return;
    };
    let local_plan = q8act_plan(down_type, cols, &qh);
    let Q8ActPlan::Fused(dfast) = &local_plan else {
        dirty.set();
        return;
    };
    if !dfast.fast {
        dirty.set();
        return;
    }
    let off = (c0 / geom.down_block_width) * geom.down_block_bytes;
    let slice_bytes = cols.div_ceil(geom.down_block_width) * geom.down_block_bytes;
    let mut partial: Vec<i128> = Vec::with_capacity(geom.n_embd);
    let mut row = 0usize;
    while row < geom.n_embd && ok {
        let start = row * geom.down_row_bytes + off;
        match down_raw
            .get(start..start + slice_bytes)
            .and_then(|r| q8act_row_i128_fast(down_type, r, cols, &qh, dfast))
        {
            Some(acc) => partial.push(acc),
            None => ok = false,
        }
        row += 1;
    }
    if !ok {
        dirty.set();
        return;
    }
    slots.lock().push((t, SliceData { quant: qh, partial }));
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::many_single_char_names,
        clippy::needless_range_loop,
        reason = "test-only fixture/reference arithmetic on bounded small values; not extracted"
    )]

    use super::*;
    use crate::engine::ONE;
    use crate::format::quant::{
        GGML_TYPE_F16, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
        Q4_K_BLOCK_BYTES, Q5_0_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES,
    };
    use std::time::Instant;

    /// `f16` `2^-8` in Q32.32 scale units: the finite weight scale the fixtures use.
    const D_SCALE: u16 = 0x1C00;
    /// `f16` `2^-9`: the `Q4_K` min scale.
    const D_MIN: u16 = 0x1800;

    /// A deterministic pseudo-random `u64` (Knuth's LCG), test-only.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    /// One random byte from the generator.
    fn byte(state: &mut u64) -> u8 {
        (lcg(state) & 0xFF) as u8
    }

    fn pw(ggml_type: u32, raw: &[u8]) -> ProjWeights<'_> {
        ProjWeights { ggml_type, raw }
    }

    /// A `Q8_0` payload of `rows × cols` (`cols` a multiple of 32): scale `2^-8`, random codes.
    fn q8_0_raw(rows: usize, cols: usize, salt: u64) -> Vec<u8> {
        let blocks = cols.div_ceil(32);
        let mut out: Vec<u8> = Vec::with_capacity(rows * blocks * Q8_0_BLOCK_BYTES);
        let mut state = salt;
        for _ in 0..rows {
            for _ in 0..blocks {
                out.extend_from_slice(&D_SCALE.to_le_bytes());
                for _ in 0..32 {
                    out.push(byte(&mut state));
                }
            }
        }
        out
    }

    /// A `Q5_0` payload of `rows × cols` (`cols` a multiple of 32): scale `2^-8`, random planes.
    fn q5_0_raw(rows: usize, cols: usize, salt: u64) -> Vec<u8> {
        let blocks = cols.div_ceil(32);
        let mut out: Vec<u8> = Vec::with_capacity(rows * blocks * Q5_0_BLOCK_BYTES);
        let mut state = salt;
        for _ in 0..rows {
            for _ in 0..blocks {
                out.extend_from_slice(&D_SCALE.to_le_bytes());
                for _ in 0..4 {
                    out.push(byte(&mut state));
                }
                for _ in 0..16 {
                    out.push(byte(&mut state));
                }
            }
        }
        out
    }

    /// A `Q4_K` payload of `rows × cols` (`cols` a multiple of 64): scale `2^-8`, min `2^-9`,
    /// random packed scales and nibbles.
    fn q4_k_raw(rows: usize, cols: usize, salt: u64) -> Vec<u8> {
        let blocks = cols.div_ceil(256);
        let mut out: Vec<u8> = Vec::with_capacity(rows * blocks * Q4_K_BLOCK_BYTES);
        let mut state = salt;
        for _ in 0..rows {
            for _ in 0..blocks {
                out.extend_from_slice(&D_SCALE.to_le_bytes());
                out.extend_from_slice(&D_MIN.to_le_bytes());
                for _ in 0..12 {
                    out.push(byte(&mut state));
                }
                for _ in 0..128 {
                    out.push(byte(&mut state));
                }
            }
        }
        out
    }

    /// A `Q6_K` payload of `rows × cols` (`cols` a multiple of 256): scale `2^-8`, random planes
    /// and signed sub-block scales.
    fn q6_k_raw(rows: usize, cols: usize, salt: u64) -> Vec<u8> {
        let blocks = cols.div_ceil(256);
        let mut out: Vec<u8> = Vec::with_capacity(rows * blocks * Q6_K_BLOCK_BYTES);
        let mut state = salt;
        for _ in 0..rows {
            for _ in 0..blocks {
                for _ in 0..128 {
                    out.push(byte(&mut state));
                }
                for _ in 0..64 {
                    out.push(byte(&mut state));
                }
                for _ in 0..16 {
                    out.push(byte(&mut state));
                }
                out.extend_from_slice(&D_SCALE.to_le_bytes());
            }
        }
        out
    }

    /// Finite `f16` bit patterns the fixture draws from: `0`, `±0.125`, `±0.25`, `±0.5`, `±1`,
    /// `±2` and `±4`.
    const PALETTE: [u16; 13] = [
        0x0000, 0x3000, 0xB000, 0x3400, 0xB400, 0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000,
        0x4200, 0xC200,
    ];

    /// A row-major `F16` payload of `rows × cols` drawn from the palette.
    fn f16_raw(rows: usize, cols: usize, salt: u64) -> Vec<u8> {
        let mut state = salt;
        let mut out: Vec<u8> = Vec::with_capacity(rows * cols * 2);
        for _ in 0..rows * cols {
            let idx = usize::try_from(lcg(&mut state) % 13).expect("index < 13");
            out.extend_from_slice(&PALETTE[idx].to_le_bytes());
        }
        out
    }

    /// A payload of `rows × cols` in `ty`'s fixture form.
    fn weights(ty: u32, rows: usize, cols: usize, salt: u64) -> Vec<u8> {
        match ty {
            GGML_TYPE_Q8_0 => q8_0_raw(rows, cols, salt),
            GGML_TYPE_Q5_0 => q5_0_raw(rows, cols, salt),
            GGML_TYPE_Q4_K => q4_k_raw(rows, cols, salt),
            GGML_TYPE_Q6_K => q6_k_raw(rows, cols, salt),
            GGML_TYPE_F16 => f16_raw(rows, cols, salt),
            other => panic!("no fixture for type {other}"),
        }
    }

    /// A Q32.32 activation with `±1/4`- and `0`-magnitude entries, as in the chart tests.
    fn activation(n: usize, salt: i64) -> Vec<i64> {
        let mut x: Vec<i64> = Vec::with_capacity(n);
        let mut i = 0usize;
        while i < n {
            let j = i64::try_from(i).expect("small") * 3 + salt;
            x.push((j.rem_euclid(7) - 3) * (ONE / 8));
            i += 1;
        }
        x
    }

    /// `ffn_tp` equals `ffn_q8act` on one case: `(gate/up type, down type, n_embd, n_ff)`.
    fn check(case: (u32, u32, usize, usize), salt: u64) {
        let (gu, dt, n_embd, n_ff) = case;
        let x = activation(n_embd, i64::try_from(salt).expect("small") % 11);
        let g = weights(gu, n_ff, n_embd, salt);
        let u = weights(gu, n_ff, n_embd, salt + 1);
        let d = weights(dt, n_embd, n_ff, salt + 2);
        let (gw, uw, dw) = (pw(gu, &g), pw(gu, &u), pw(dt, &d));
        let want = ffn_q8act(gw, uw, dw, n_ff, &x)
            .unwrap_or_else(|e| panic!("reference refused {case:?}: {e:?}"));
        let got = ffn_tp(gw, uw, dw, n_ff, &x)
            .unwrap_or_else(|e| panic!("ffn_tp refused {case:?}: {e:?}"));
        assert_eq!(got, want, "case {case:?}");
    }

    /// The four fully-fast down families, sliced, over ragged and odd slice counts: the `Q4_K`
    /// cases with `n_ff` not a multiple of 256 give a short last slice, and the `Q6_K` and
    /// `Q5_0`/`Q8_0` cases give one to three and seven slices.
    #[test]
    fn ffn_tp_equals_ffn_q8act_on_every_fast_down_family() {
        const CASES: &[(u32, u32, usize, usize)] = &[
            (GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 32, 32),
            (GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 64, 96),
            (GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, 32, 224),
            (GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, 64, 160),
            (GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 64, 64),
            (GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 64, 192),
            (GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 128, 320),
            (GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 128, 448),
            (GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 64, 512),
            (GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, 256, 256),
            (GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, 256, 768),
            (GGML_TYPE_Q5_0, GGML_TYPE_Q6_K, 64, 256),
            (GGML_TYPE_Q5_0, GGML_TYPE_Q6_K, 896, 1024),
        ];
        for (i, &case) in CASES.iter().enumerate() {
            check(case, 0x5EED + 7 * u64::try_from(i).expect("small"));
        }
    }

    /// The non-sliced families — `F16` on either side, the real Qwen gate/up type with an `F16`
    /// down — take `ffn_q8act` itself and still agree bit for bit.
    #[test]
    fn ffn_tp_equals_ffn_q8act_on_the_non_sliced_families() {
        for &case in &[
            (GGML_TYPE_F16, GGML_TYPE_F16, 32, 96),
            (GGML_TYPE_Q5_0, GGML_TYPE_F16, 32, 128),
            (GGML_TYPE_F16, GGML_TYPE_Q5_0, 32, 128),
            (GGML_TYPE_F16, GGML_TYPE_Q6_K, 64, 256),
        ] {
            check(case, 0xBEEF);
        }
    }

    /// A truncated down payload and the degenerate geometries refuse with the same [`FfnError`]
    /// `ffn_q8act` returns.
    #[test]
    fn ffn_tp_refuses_exactly_as_ffn_q8act() {
        const N_EMBD: usize = 64;
        const N_FF: usize = 256;
        let x = activation(N_EMBD, 5);
        let g = q5_0_raw(N_FF, N_EMBD, 1);
        let u = q5_0_raw(N_FF, N_EMBD, 2);
        let mut d = q6_k_raw(N_EMBD, N_FF, 3);
        d.truncate(d.len() - 1);
        let (gw, uw, dw) = (
            pw(GGML_TYPE_Q5_0, &g),
            pw(GGML_TYPE_Q5_0, &u),
            pw(GGML_TYPE_Q6_K, &d),
        );
        let got = ffn_tp(gw, uw, dw, N_FF, &x);
        assert_eq!(got, ffn_q8act(gw, uw, dw, N_FF, &x));
        assert!(
            matches!(got, Err(FfnError::Matvec(MatvecError::ShortPayload { .. }))),
            "got {got:?}"
        );
        assert_eq!(ffn_tp(gw, uw, dw, 0, &x), Err(FfnError::BadGeometry));
        assert_eq!(ffn_tp(gw, uw, dw, 0, &x), ffn_q8act(gw, uw, dw, 0, &x));
        assert_eq!(
            ffn_tp(gw, uw, dw, N_FF, &[]),
            ffn_q8act(gw, uw, dw, N_FF, &[])
        );
    }

    /// Best-of-[`REPS`] and median microseconds of `ffn_tp` and `ffn_q8act` for one FFN shape with
    /// `Q5_0` gate/up and a `Q6_K` down projection, interleaved one rep each.
    fn time_shape(n_ff: usize, n_embd: usize, salt: u64) -> (Vec<u128>, Vec<u128>) {
        const REPS: usize = 50;
        let x = activation(n_embd, 3);
        let g = q5_0_raw(n_ff, n_embd, salt);
        let u = q5_0_raw(n_ff, n_embd, salt + 1);
        let d = q6_k_raw(n_embd, n_ff, salt + 2);
        let (gw, uw, dw) = (
            pw(GGML_TYPE_Q5_0, &g),
            pw(GGML_TYPE_Q5_0, &u),
            pw(GGML_TYPE_Q6_K, &d),
        );
        let want = ffn_q8act(gw, uw, dw, n_ff, &x).expect("reference");
        assert_eq!(ffn_tp(gw, uw, dw, n_ff, &x).expect("tp"), want);
        let mut tp: Vec<u128> = Vec::with_capacity(REPS);
        let mut base: Vec<u128> = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            std::hint::black_box(ffn_tp(gw, uw, dw, n_ff, &x).expect("tp"));
            tp.push(t.elapsed().as_micros());
            let t = Instant::now();
            std::hint::black_box(ffn_q8act(gw, uw, dw, n_ff, &x).expect("reference"));
            base.push(t.elapsed().as_micros());
        }
        tp.sort_unstable();
        base.sort_unstable();
        (tp, base)
    }

    /// The real `qwen05b-Q4_K_M.gguf` layer-0 FFN shapes and types, read off the file's directory
    /// (`blk.0.ffn_gate`/`ffn_up` are `Q5_0` `[896, 4864]`, `blk.0.ffn_down` is `Q6_K` `[4864, 896]`),
    /// with a synthetic finite payload of those shapes and types — the hidden width is 19 `Q6_K`
    /// blocks, so the cover has 19 slices over the pool's eight participants. The second row is the
    /// balanced control shape (`n_ff` a multiple of `8·256`), one 256-column slice per participant.
    /// Best-of-50 and median, in microseconds:
    ///
    /// ```sh
    /// APEIRON_THREADS=8 cargo test --release --all-features \
    ///     engine::kern::tp::tests::ffn_tp_timing_qwen_layer0 -- --ignored --nocapture
    /// ```
    ///
    /// The falsifier is the reported ratio; nothing here is asserted about it.
    #[test]
    #[ignore = "timing measurement; run with --ignored --nocapture and APEIRON_THREADS=8"]
    fn ffn_tp_timing_qwen_layer0() {
        for (n_ff, note) in [(4864usize, "real layer-0"), (2048, "balanced control")] {
            let (tp, base) = time_shape(n_ff, 896, 0x1111);
            let mid = tp.len() / 2;
            let faster = tp[mid] < base[mid];
            println!(
                "n_ff {n_ff} ({note}, {} down blocks, {} slices): ffn_tp best {} us median {} us | \
                 ffn_q8act best {} us median {} us | median ratio {:.3} | ffn_tp faster = {faster}",
                n_ff / 256,
                n_ff.div_ceil(256),
                tp[0],
                tp[mid],
                base[0],
                base[mid],
                tp[mid] as f64 / base[mid] as f64
            );
        }
    }
}
