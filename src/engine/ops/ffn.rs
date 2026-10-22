//! The dense FFN (`SwiGLU`) block and its scalar `SiLU` (port-map row 6).

use super::matvec::MatvecError;
use super::par::{matvec_rows_chunked, row_chunk_budget};
use crate::engine::num::value::round_shift_32;

#[path = "silu.rs"]
pub mod silu;

/// A refused fixed-point FFN (`SwiGLU`) step (port-map row 6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfnError {
    /// The geometry is degenerate: the hidden width `n_ff` is zero or the activation `x` is empty,
    /// so the FFN block is undefined.
    BadGeometry,
    /// A [`fixed_exp`](crate::numeric::fixed_exp::fixed_exp) argument on the exact `SiLU` path
    /// left its window `[-16·ONE, 21·ONE]`. Values outside the open interval `(-21·ONE, 16·ONE)`
    /// saturate instead, so this is unreachable for a well-formed call and is retained for the
    /// in-window path.
    ExpOutOfRange,
    /// The `SiLU` denominator `ONE + fixed_exp(-v)` is zero, so the division is undefined.
    ZeroDenominator,
    /// An `i128` product or an `i64` narrowing left its range.
    Overflow,
    /// A projection refused through [`MatvecError`].
    Matvec(MatvecError),
}

impl From<MatvecError> for FfnError {
    fn from(e: MatvecError) -> Self {
        FfnError::Matvec(e)
    }
}

impl From<silu::SiluError> for FfnError {
    fn from(e: silu::SiluError) -> Self {
        match e {
            silu::SiluError::ExpOutOfRange => FfnError::ExpOutOfRange,
            silu::SiluError::ZeroDenominator => FfnError::ZeroDenominator,
            silu::SiluError::Overflow => FfnError::Overflow,
        }
    }
}

/// One projection's weight payload for [`ffn_fixed`]: the ggml type plus the row-major quantized
/// bytes, the same pair [`matvec_fixed`] consumes.
#[derive(Debug, Clone, Copy)]
pub struct ProjWeights<'a> {
    /// ggml type id of the quantized rows.
    pub ggml_type: u32,
    /// Row-major quantized weight bytes.
    pub raw: &'a [u8],
}

/// The three `n_ff`-word vectors an FFN block computes before the down projection, retained by
/// [`ffn_fixed_tapped`]: the gate projection (`silu`'s argument), the up projection (`silu`'s
/// multiplier) and the `SwiGLU` product `h` (the down projection's input). Each is moved out of
/// [`ffn_fixed_chunks`], not copied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfnStages {
    /// `gate = matvec_rows_chunked(W_gate, x)`, the `SiLU` argument.
    pub gate: Vec<i64>,
    /// `up = matvec_rows_chunked(W_up, x)`, the `SwiGLU` multiplier.
    pub up: Vec<i64>,
    /// `h_i = round(silu(gate_i)·up_i / 2^32)`, the down projection's input.
    pub h: Vec<i64>,
}

/// Fixed-point `SiLU` (port-map row 6): `silu(v) = v / (1 + exp(-v))` in Q32.32.
///
/// The real body is [`silu::silu_fixed`], the scalar kernel isolated from the projection kernels so
/// the Charon/Aeneas extraction can root it. This entry point only maps its [`silu::SiluError`]
/// onto [`FfnError`], so every `crate::engine::silu_fixed` caller is unchanged. The saturation
/// bounds and the in-window quotient are documented at [`silu::silu_fixed`].
///
/// # Errors
///
/// [`FfnError::ExpOutOfRange`] (unreachable in the exact window), [`FfnError::ZeroDenominator`] and
/// [`FfnError::Overflow`]; see [`silu::silu_fixed`] for the domain.
pub fn silu_fixed(v: i64) -> Result<i64, FfnError> {
    silu::silu_fixed(v).map_err(FfnError::from)
}

/// One token of the dense FFN (`SwiGLU`) block (port-map row 6), in the order of the fold
/// `moe.rs::swiglu`.
///
/// `gate = matvec_rows_chunked(W_gate, x)` and `up = matvec_rows_chunked(W_up, x)` over `n_ff` rows
/// each, then `h_i = round(silu_fixed(gate_i)·up_i / 2^32)` under the value-path epilogue
/// [`round_shift_32`], then `y = matvec_rows_chunked(W_down, h)` over `x.len()` rows. `W_gate`/`W_up`
/// are `n_ff × x.len()`; `W_down` is `x.len() × n_ff`, so the block preserves the residual width.
/// The gate is the `SiLU` argument and the `up` row is the multiplier — the fold's `silu(gate) *
/// up`, not `gate * silu(up)`.
///
/// The three projections are the row-parallel [`matvec_rows_chunked`] kernel over
/// [`row_chunk_budget`]-sized row chunks, so the block uses the machine's parallelism as the
/// decode step's other projections do; the chunked concatenation is bit-identical to the unchunked
/// call. The elementwise `SiLU` pass is unchanged and single-threaded.
///
/// # Errors
///
/// Returns [`FfnError::BadGeometry`] for a zero `n_ff` or an empty `x`, [`FfnError::Matvec`] for a
/// projection shape, payload, quant or accumulator refusal, and [`FfnError::Overflow`] when an
/// elementwise product or a narrowing leaves its range. Gate logits saturate inside [`silu_fixed`];
/// no input is clamped.
///
/// # Complexity
///
/// `O(n_ff·n_embd)` time in checked integer multiply-adds over the three projections plus
/// `O(n_ff)` for the elementwise pass, `O(n_ff)` scratch for the gate and up vectors and the hidden
/// vector, `O(n_embd)` for the output, and `O(chunks)` scoped-thread handles owned by the chunked
/// kernel; no weight payload is copied.
pub fn ffn_fixed(
    gate: ProjWeights<'_>,
    up: ProjWeights<'_>,
    down: ProjWeights<'_>,
    n_ff: usize,
    x: &[i64],
) -> Result<Vec<i64>, FfnError> {
    let n_embd = x.len();
    match ffn_fixed_chunks(
        gate,
        up,
        down,
        n_ff,
        x,
        row_chunk_budget(n_ff),
        row_chunk_budget(n_embd),
    ) {
        Ok((y, _)) => Ok(y),
        Err(e) => Err(e),
    }
}

/// [`ffn_fixed`] with the gate, up and `SwiGLU` vectors retained: the same block bit for bit, plus
/// an [`FfnStages`] holding the three `n_ff`-word rows the down projection consumed.
///
/// The executing body is [`ffn_fixed_chunks`], the one [`ffn_fixed`] runs, so the returned output is
/// not a second implementation of the block; it returns the vectors the body already
/// computes. The engine's tap path uses this to record the FFN pre-activations and product.
///
/// # Errors
///
/// As [`ffn_fixed`]: [`FfnError::BadGeometry`] for a zero `n_ff` or an empty `x`,
/// [`FfnError::Matvec`] for a projection refusal, and [`FfnError::Overflow`] for an elementwise
/// product or narrowing outside its range. On a refusal no [`FfnStages`] is returned.
///
/// # Complexity
///
/// Exactly [`ffn_fixed`]'s `O(n_ff·n_embd)` time and `O(n_ff)` scratch; the difference is that the
/// `O(n_ff)` gate, up and `h` vectors are moved into the returned [`FfnStages`] instead of dropped.
pub fn ffn_fixed_tapped(
    gate: ProjWeights<'_>,
    up: ProjWeights<'_>,
    down: ProjWeights<'_>,
    n_ff: usize,
    x: &[i64],
) -> Result<(Vec<i64>, FfnStages), FfnError> {
    let n_embd = x.len();
    ffn_fixed_chunks(
        gate,
        up,
        down,
        n_ff,
        x,
        row_chunk_budget(n_ff),
        row_chunk_budget(n_embd),
    )
}

/// [`ffn_fixed`]'s body with the hidden projections' row-chunk counts supplied by the caller,
/// returning the output with the gate, up and `SwiGLU` vectors it computed.
///
/// [`ffn_fixed`] passes [`row_chunk_budget`]`(n_ff)` for the `n_ff`-row gate and up projections and
/// [`row_chunk_budget`]`(n_embd)` for the `n_embd`-row down projection. Splitting the two counts out
/// lets the equivalence test drive every split, including counts that do not divide a row count and
/// counts of one row per chunk; the executing path always passes the budget.
///
/// The three vectors are moved into the returned [`FfnStages`], not copied; [`ffn_fixed`] drops
/// them. The stages are returned by value rather than through a borrowed sink so the body stays in
/// the extracted fragment (`proof/extract/lib.rs` includes this file).
fn ffn_fixed_chunks(
    gate: ProjWeights<'_>,
    up: ProjWeights<'_>,
    down: ProjWeights<'_>,
    n_ff: usize,
    x: &[i64],
    gate_chunks: usize,
    down_chunks: usize,
) -> Result<(Vec<i64>, FfnStages), FfnError> {
    let n_embd = x.len();
    if n_ff == 0 || n_embd == 0 {
        return Err(FfnError::BadGeometry);
    }
    let gate_vec = matvec_rows_chunked(gate.ggml_type, gate.raw, n_ff, n_embd, x, gate_chunks)?;
    let up_vec = matvec_rows_chunked(up.ggml_type, up.raw, n_ff, n_embd, x, gate_chunks)?;
    let mut h: Vec<i64> = Vec::with_capacity(n_ff);
    let mut i = 0usize;
    let mut err: Option<FfnError> = None;
    while i < n_ff && err.is_none() {
        match silu_fixed(gate_vec[i]) {
            Err(failure) => err = Some(failure),
            Ok(s) => match i128::from(s).checked_mul(i128::from(up_vec[i])) {
                None => err = Some(FfnError::Overflow),
                Some(prod) => match i64::try_from(round_shift_32(prod)) {
                    Ok(val) => h.push(val),
                    Err(_) => err = Some(FfnError::Overflow),
                },
            },
        }
        i += 1;
    }
    if let Some(failure) = err {
        return Err(failure);
    }
    let y = matvec_rows_chunked(down.ggml_type, down.raw, n_embd, n_ff, &h, down_chunks)?;
    Ok((
        y,
        FfnStages {
            gate: gate_vec,
            up: up_vec,
            h,
        },
    ))
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
    use crate::engine::matvec_fixed;
    use crate::format::quant::GGML_TYPE_F16;

    /// Finite `f16` bit patterns the fixture draws from: `0`, `±0.125`, `±0.25`, `±0.5`, `±1`, `±2`
    /// and `±4`. All are exact small values that keep the `n_in = 32` dots inside `i128` and the
    /// gate inside the `SiLU` saturation bounds.
    const FFN_PALETTE: [u16; 13] = [
        0x0000, 0x3000, 0xB000, 0x3400, 0xB400, 0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000,
        0x4200, 0xC200,
    ];

    /// A deterministic pseudo-random `u64` (Knuth's LCG), test-only.
    fn ffn_lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    /// A row-major `F16` payload of `rows × cols` drawn from the palette.
    fn ffn_f16_raw(rows: usize, cols: usize, state: &mut u64) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::with_capacity(rows * cols * 2);
        let mut i = 0usize;
        while i < rows * cols {
            let idx = usize::try_from(ffn_lcg(state) % 13).expect("index < 13");
            out.extend_from_slice(&FFN_PALETTE[idx].to_le_bytes());
            i += 1;
        }
        out
    }

    /// The Q32.32 activation the fixture projects: `0` and `±0.25`, so the `SiLU` arguments stay
    /// small and no projection refuses.
    fn ffn_activation(n: usize) -> Vec<i64> {
        let mut x: Vec<i64> = Vec::with_capacity(n);
        let mut i = 0usize;
        while i < n {
            x.push((i64::try_from(i % 5).expect("small") - 2) * (crate::engine::ONE / 4));
            i += 1;
        }
        x
    }

    /// The sequential reference: one [`matvec_fixed`] per projection and the elementwise
    /// `round(silu(gate)·up / 2^32)`, built independently of [`ffn_fixed_chunks`] so a defect in the
    /// chunked body cannot move both sides of the comparison together.
    fn ffn_sequential_ref(
        gate: ProjWeights<'_>,
        up: ProjWeights<'_>,
        down: ProjWeights<'_>,
        n_ff: usize,
        x: &[i64],
    ) -> Vec<i64> {
        let n_embd = x.len();
        let gate_vec = matvec_fixed(gate.ggml_type, gate.raw, n_ff, n_embd, x).expect("gate");
        let up_vec = matvec_fixed(up.ggml_type, up.raw, n_ff, n_embd, x).expect("up");
        let mut h: Vec<i64> = Vec::with_capacity(n_ff);
        let mut i = 0usize;
        while i < n_ff {
            let s = silu_fixed(gate_vec[i]).expect("silu");
            let prod = i128::from(s)
                .checked_mul(i128::from(up_vec[i]))
                .expect("in range");
            h.push(i64::try_from(round_shift_32(prod)).expect("in range"));
            i += 1;
        }
        matvec_fixed(down.ggml_type, down.raw, n_embd, n_ff, &h).expect("down")
    }

    /// The row-parallel FFN equals the sequential FFN bit-for-bit for several hidden widths and
    /// every split tried, including splits that do not divide a row count and one row per chunk. The
    /// sequential reference is [`ffn_sequential_ref`], one `matvec_fixed` per projection.
    #[test]
    fn ffn_chunked_equals_sequential_on_random_f16_weights() {
        const N_IN: usize = 32;
        /// Fixed gate/down splits; every pair exercises uneven division and uneven chunk order.
        const SPLITS: [(usize, usize); 7] =
            [(2, 1), (1, 2), (3, 2), (2, 3), (5, 7), (7, 5), (1, 1)];
        let x = ffn_activation(N_IN);
        let widths = [1usize, 7, 63, 64, 65, 128, 131, 200];
        for (case, &n_ff) in widths.iter().enumerate() {
            let mut state = 0x0123_4567_89AB_CDEFu64 ^ u64::try_from(case).expect("small");
            let gate = ffn_f16_raw(n_ff, N_IN, &mut state);
            let up = ffn_f16_raw(n_ff, N_IN, &mut state);
            let down = ffn_f16_raw(N_IN, n_ff, &mut state);
            let weights = || {
                (
                    ProjWeights {
                        ggml_type: GGML_TYPE_F16,
                        raw: &gate,
                    },
                    ProjWeights {
                        ggml_type: GGML_TYPE_F16,
                        raw: &up,
                    },
                    ProjWeights {
                        ggml_type: GGML_TYPE_F16,
                        raw: &down,
                    },
                )
            };
            let (wg, wu, wd) = weights();
            let want = ffn_sequential_ref(wg, wu, wd, n_ff, &x);
            assert_eq!(
                ffn_fixed(wg, wu, wd, n_ff, &x).expect("budget"),
                want,
                "n_ff {n_ff}: the budget-driven path disagrees with the sequential reference"
            );
            for &(gc, dc) in &SPLITS {
                let (wg, wu, wd) = weights();
                assert_eq!(
                    ffn_fixed_chunks(wg, wu, wd, n_ff, &x, gc, dc)
                        .expect("chunked")
                        .0,
                    want,
                    "n_ff {n_ff}: gate chunks {gc}, down chunks {dc}"
                );
            }
            // One row per chunk, and more chunks than rows (clamped inside the kernel).
            for &(gc, dc) in &[(n_ff, N_IN), (n_ff + 3, N_IN + 3)] {
                let (wg, wu, wd) = weights();
                assert_eq!(
                    ffn_fixed_chunks(wg, wu, wd, n_ff, &x, gc, dc)
                        .expect("chunked-max")
                        .0,
                    want,
                    "n_ff {n_ff}: gate chunks {gc}, down chunks {dc}"
                );
            }
        }
    }
}
