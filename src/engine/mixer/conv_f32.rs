//! The prefill (`k > 1`) f32 convolution: `conv_silu_cols_f32`, pooled over channel ranges.
//!
//! Each pool task owns channels `[c0, c1)` of every row: it converts the Q32 inputs it reads to
//! f32 in registers (each input word once), writes both the f32 column cache the heads dispatch
//! reads and the Q32 output columns, and advances its own channels of the window. The call allocates
//! the `k` Q32 output columns the signature returns, one flat `k · channels` copy of the input rows,
//! and `O(k + kernel)` scratch words per pool range; the f32 cache is the state-wide `conv_shared`
//! buffer, reused at its high-water capacity.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "dense numeric kernels name operands as in the recurrences they implement"
)]

use super::{alloc_cols, col_writers, f32_carrier, AffineLayer, ColWriters, ConvState, Flat};
use crate::engine::chart::pool::{set_written_len, set_written_lens};
use crate::engine::chart::task::IndexTask;
use crate::engine::fp::F32;
use crate::engine::step::StepError;

#[cfg(test)]
pub(super) mod seg {
    //! Test-only segment timers (ns) for the real `conv_silu_cols_f32`: 0 setup (validation,
    //! buffer take, output column allocation), 1 pool wall, 2 task time (sum over tasks), 3 task
    //! time (max over tasks), 4 `set_len`, 5 buffer restore.
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    pub(crate) static NS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
    pub(crate) static SKIP_READOUT: AtomicBool = AtomicBool::new(false);
    pub(crate) fn add(i: usize, t0: std::time::Instant) {
        NS[i].fetch_add(
            u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Relaxed,
        );
    }
    pub(crate) fn max(i: usize, t0: std::time::Instant) {
        NS[i].fetch_max(
            u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Relaxed,
        );
    }
}

/// Raw pointer that crosses pool threads; task `t` touches only channels `[t·per, t·per + per)`.
struct Shared<T>(*mut T);
// SAFETY: the dispatch gives each task a disjoint channel range of every row.
unsafe impl<T> Send for Shared<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    /// Store `value` at element `i`.
    ///
    /// # Safety
    ///
    /// `i` is inside the allocation and no other live writer targets element `i`.
    #[inline]
    unsafe fn write(&self, i: usize, value: T) {
        // SAFETY: the caller guarantees the index and the disjointness.
        unsafe { self.0.add(i).write(value) };
    }
}

/// History row `r` (`r < cols`: window, else converted input row `r − cols` of the flat `xs`) at
/// channel `c`.
#[inline]
fn hist_at(window: &[F32], xs: &[i64], cols: usize, ch: usize, r: usize, c: usize) -> F32 {
    if r < cols {
        return window[r * ch + c];
    }
    f32_carrier::q32_to_f32(xs[(r - cols) * ch + c])
}

/// Channel `c` of all `k` columns: `ys[j]` is the f32 `SiLU` output, `yq[j]` its Q32 readout. `taps`
/// and `hist` are `kernel`-word scratch. The first refusal is returned; entries before it are
/// written, later ones are not.
///
/// # Complexity
///
/// `O(k · kernel)` f32 FMAs plus `k` `SiLU`s and readouts, `O(kernel)` scratch words: the outer loop
/// runs `k` times, the tap loop `kernel` times.
#[allow(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    reason = "w, x, c, t are the convolution's symbols; the geometry is passed as scalars so every \
              reference parameter is single-level"
)]
fn conv_channel(
    w: &[i64],
    window: &[F32],
    xs: &[i64],
    kernel: usize,
    cols: usize,
    ch: usize,
    k: usize,
    c: usize,
    taps: &mut [F32],
    hist: &mut [F32],
    ys: &mut [F32],
    yq: &mut [i64],
) -> Result<(), StepError> {
    let base = c * kernel;
    let mut tap = 0;
    while tap < kernel {
        taps[tap] = f32_carrier::q32_to_f32(w[base + tap]);
        hist[tap] = hist_at(window, xs, cols, ch, tap, c);
        tap += 1;
    }
    let mut err: Option<StepError> = None;
    let mut j = 0;
    while j < k && err.is_none() {
        let mut acc = F32::ZERO;
        let mut tap = 0;
        while tap < kernel {
            acc = taps[tap].fma(hist[tap], acc);
            tap += 1;
        }
        if acc.is_finite() {
            let y = f32_carrier::silu_f32_scalar(acc);
            match f32_carrier::f32_readout_to_q32(y) {
                Ok(y_q32) => {
                    ys[j] = y;
                    yq[j] = y_q32;
                    if j + 1 < k {
                        hist.copy_within(1..kernel, 0);
                        hist[kernel - 1] = hist_at(window, xs, cols, ch, j + kernel, c);
                    }
                }
                Err(e) => err = Some(e),
            }
        } else {
            err = Some(StepError::Overflow);
        }
        j += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// Publish channel `c` of every column: the pool's disjoint-write point.
///
/// # Safety
///
/// `c` is in this task's channel range; every column `j < ys.len()` reserved `ch` words and the f32
/// cache holds `ys.len() · ch`.
#[inline]
unsafe fn scatter(
    out_f32: &Shared<F32>,
    out_q32: &ColWriters,
    ch: usize,
    c: usize,
    ys: &[F32],
    yq: &[i64],
) {
    let mut j = 0;
    while j < ys.len() {
        // SAFETY: the caller's contract covers `(j, c)`.
        unsafe {
            out_q32.write(j, c, yq[j]);
            out_f32.write(j * ch + c, ys[j]);
        }
        j += 1;
    }
}

/// Channels `[c0, c1)` of every column, any kernel width, channel tail, no AVX2.
///
/// # Complexity
///
/// `O((c1 − c0) · k · kernel)` f32 FMAs, `O(k + kernel)` scratch words.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry is scalars so every reference parameter is single-level"
)]
fn range_scalar(
    w: &[i64],
    window: &[F32],
    xs: &[i64],
    kernel: usize,
    cols: usize,
    ch: usize,
    k: usize,
    out_f32: &Shared<F32>,
    out_q32: &ColWriters,
    c0: usize,
    c1: usize,
) -> Result<(), StepError> {
    let mut taps = vec![F32::ZERO; kernel];
    let mut hist = vec![F32::ZERO; kernel];
    let mut ys = vec![F32::ZERO; k];
    let mut yq = vec![0i64; k];
    let mut err: Option<StepError> = None;
    let mut c = c0;
    while c < c1 && err.is_none() {
        match conv_channel(
            w, window, xs, kernel, cols, ch, k, c, &mut taps, &mut hist, &mut ys, &mut yq,
        ) {
            // SAFETY: `c` is in this task's range; column `j` reserved `ch` words.
            Ok(()) => unsafe { scatter(out_f32, out_q32, ch, c, &ys, &yq) },
            Err(e) => err = Some(e),
        }
        c += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// Four-tap path over whole 8-channel groups `[c0, c_end)`. History rows slide through four
/// registers, so each input row is converted once per group. SIMD twin of [`range_scalar`]: intrinsic
/// body, opaque to the extraction.
///
/// # Safety
///
/// AVX2 and FMA are available; `kernel == 4`; `c0 <= c_end <= ch`, `(c_end − c0) % 8 == 0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry is scalars so every reference parameter is single-level"
)]
unsafe fn range_avx2(
    w: &[i64],
    window: &[F32],
    xs: &[i64],
    cols: usize,
    ch: usize,
    k: usize,
    out_f32: &Shared<F32>,
    out_q32: &ColWriters,
    c0: usize,
    c_end: usize,
) -> Result<(), StepError> {
    use std::arch::x86_64::{__m256, _mm256_loadu_ps, _mm256_storeu_ps};
    let load8 = |r: usize, c: usize| -> __m256 {
        if r < cols {
            // SAFETY: `r < cols`, `c + 8 <= ch`.
            unsafe { _mm256_loadu_ps(window.as_ptr().add(r * ch + c).cast::<f32>()) }
        } else {
            let at = (r - cols) * ch + c;
            let x = &xs[at..at + 8];
            let mut b = [F32::ZERO; 8];
            let mut i = 0;
            while i < 8 {
                b[i] = f32_carrier::q32_to_f32(x[i]);
                i += 1;
            }
            // SAFETY: `b` holds 8 f32.
            unsafe { _mm256_loadu_ps(b.as_ptr().cast::<f32>()) }
        }
    };
    let mut err: Option<StepError> = None;
    let mut c = c0;
    while c < c_end && err.is_none() {
        let mut t = [[F32::ZERO; 8]; 4];
        #[allow(
            clippy::needless_range_loop,
            reason = "a Range iterator is outside the extraction fragment"
        )]
        for lane in 0..8 {
            let base = (c + lane) * 4;
            let mut tap = 0;
            while tap < 4 {
                t[tap][lane] = f32_carrier::q32_to_f32(w[base + tap]);
                tap += 1;
            }
        }
        // SAFETY: each `t[i]` holds 8 f32.
        let (v0, v1, v2, v3) = unsafe {
            (
                _mm256_loadu_ps(t[0].as_ptr().cast::<f32>()),
                _mm256_loadu_ps(t[1].as_ptr().cast::<f32>()),
                _mm256_loadu_ps(t[2].as_ptr().cast::<f32>()),
                _mm256_loadu_ps(t[3].as_ptr().cast::<f32>()),
            )
        };
        let (mut h0, mut h1, mut h2) = (load8(0, c), load8(1, c), load8(2, c));
        let mut j = 0;
        while j < k && err.is_none() {
            let h3 = load8(j + 3, c);
            // SAFETY: this function enables AVX2 and FMA.
            let vy = unsafe { f32_carrier::conv_silu_tap4_avx2(v0, v1, v2, v3, h0, h1, h2, h3) };
            (h0, h1, h2) = (h1, h2, h3);
            // SAFETY: `(j, c..c + 8)` is in this task's channel range; column `j` reserved
            // `ch` words; the f32 cache holds `k · ch`.
            unsafe {
                _mm256_storeu_ps(out_f32.0.add(j * ch + c).cast::<f32>(), vy);
            }
            #[cfg(test)]
            let skip = seg::SKIP_READOUT.load(std::sync::atomic::Ordering::Relaxed);
            #[cfg(not(test))]
            let skip = false;
            if !skip {
                // SAFETY: column `j` reserved `ch` words and `c + 8 <= ch`.
                let res = unsafe {
                    let dst = out_q32.0[j].add(c).cast::<i64>();
                    f32_carrier::readout8_avx2(vy, dst)
                };
                if let Err(e) = res {
                    err = Some(e);
                }
            }
            j += 1;
        }
        c += 8;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// Channels `[c0, c1)` of every column: AVX2 over whole 8-channel groups when `avx2`, the scalar
/// path for the rest.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry is scalars so every reference parameter is single-level"
)]
fn range(
    w: &[i64],
    window: &[F32],
    xs: &[i64],
    kernel: usize,
    cols: usize,
    ch: usize,
    k: usize,
    out_f32: &Shared<F32>,
    out_q32: &ColWriters,
    c0: usize,
    c1: usize,
    avx2: bool,
) -> Result<(), StepError> {
    let mut c = c0;
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        let c_end = c0 + (c1 - c0) / 8 * 8;
        // SAFETY: AVX2 and FMA were detected by the caller; `[c0, c_end)` lies inside `[0, ch)`.
        unsafe { range_avx2(w, window, xs, cols, ch, k, out_f32, out_q32, c0, c_end)? };
        c = c_end;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    if c < c1 {
        range_scalar(w, window, xs, kernel, cols, ch, k, out_f32, out_q32, c, c1)?;
    }
    Ok(())
}

/// The new window: the history's last `cols` rows of the flat `xs` (`k` rows of `ch`). It moves
/// only on success, so a refusal leaves the window as it was.
fn advance_window(conv: &mut ConvState, xs: &[i64], k: usize, cols: usize, ch: usize) {
    match conv {
        ConvState::F32(window) => {
            let mut r = 0;
            while r < cols {
                if k + r < cols {
                    window.copy_within((k + r) * ch..(k + r + 1) * ch, r * ch);
                } else {
                    let mut c = 0;
                    while c < ch {
                        window[r * ch + c] = f32_carrier::q32_to_f32(xs[(k + r - cols) * ch + c]);
                        c += 1;
                    }
                }
                r += 1;
            }
        }
        ConvState::Q32(window) => {
            let mut next = vec![0i64; cols * ch];
            let mut r = 0;
            while r < cols {
                let mut c = 0;
                while c < ch {
                    next[r * ch + c] = if k + r < cols {
                        window[(k + r) * ch + c]
                    } else {
                        xs[(k + r - cols) * ch + c]
                    };
                    c += 1;
                }
                r += 1;
            }
            *window = next;
        }
    }
}

/// The shared borrows of the pooled convolution dispatch; `run` is the single call to [`pooled_task`].
struct PooledCtx<'a> {
    w: &'a [i64],
    window: &'a [F32],
    xs: &'a [i64],
    /// `[kernel, cols, ch, k, per]`.
    dims: [usize; 5],
    out_f32: &'a Shared<F32>,
    out_q32: &'a ColWriters,
    avx2: bool,
    first: &'a crate::engine::chart::pool::FirstErr<usize, StepError>,
}

impl IndexTask for PooledCtx<'_> {
    fn run(&self, t: usize) {
        pooled_task(
            self.w,
            self.window,
            self.xs,
            self.dims,
            self.out_f32,
            self.out_q32,
            self.avx2,
            self.first,
            t,
        );
    }
}

/// Pool task `t`: channels `[t·per, (t+1)·per) ∩ [0, ch)` through [`range`]; a refusal is recorded
/// under the range's first channel.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry is scalars so every reference parameter is single-level"
)]
fn pooled_task(
    w: &[i64],
    window: &[F32],
    xs: &[i64],
    dims: [usize; 5],
    out_f32: &Shared<F32>,
    out_q32: &ColWriters,
    avx2: bool,
    first: &crate::engine::chart::pool::FirstErr<usize, StepError>,
    t: usize,
) {
    let [kernel, cols, ch, k, per] = dims;
    let c0 = t * per;
    let c1 = (c0 + per).min(ch);
    if c0 >= c1 {
        return;
    }
    #[cfg(test)]
    let t_task = std::time::Instant::now();
    let res = range(
        w, window, xs, kernel, cols, ch, k, out_f32, out_q32, c0, c1, avx2,
    );
    #[cfg(test)]
    {
        seg::add(2, t_task);
        seg::max(3, t_task);
    }
    if let Err(e) = res {
        first.record(c0, e);
    }
}

/// One pool dispatch over `max(4 · width, 96)` channel ranges. A range can refuse only with
/// `Overflow`, so which range's refusal is kept is immaterial.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry is scalars so every reference parameter is single-level"
)]
fn run_pooled(
    w: &[i64],
    window: &[F32],
    xs: &[i64],
    kernel: usize,
    cols: usize,
    ch: usize,
    k: usize,
    out_f32: &Shared<F32>,
    out_q32: &ColWriters,
    avx2: bool,
) -> Result<(), StepError> {
    let target_tasks = (4 * crate::engine::chart::pool::width_global()).max(96);
    let per = ch.div_ceil(target_tasks).max(8).div_ceil(8) * 8;
    let tasks = ch.div_ceil(per);
    let first: crate::engine::chart::pool::FirstErr<usize, StepError> =
        crate::engine::chart::pool::FirstErr::new();
    #[cfg(test)]
    let t_pool = std::time::Instant::now();
    let ctx = PooledCtx {
        w,
        window,
        xs,
        dims: [kernel, cols, ch, k, per],
        out_f32,
        out_q32,
        avx2,
        first: &first,
    };
    crate::engine::chart::pool::run_global(tasks, &ctx);
    #[cfg(test)]
    seg::add(1, t_pool);
    first.take().map_or(Ok(()), |(_, e)| Err(e))
}

/// The causal depthwise f32 convolution and `SiLU` of `k > 1` columns; window, output columns and
/// the f32 column cache as [`super::conv_silu_cols`] describes. The input rows arrive as the flat
/// `k · channels` buffer of [`super::flatten_cols`].
///
/// # Complexity
///
/// `O(k · kernel · channels)` f32 FMAs plus `k · channels` `SiLU`s and Q32 readouts, pooled over
/// `max(4 · width, 96)` channel ranges; `O((kernel − 1) · channels)` window moves; `k` output
/// column allocations. Words moved: `k · channels` Q32 writes, `k · channels` f32 writes, one
/// read of every input word.
pub(super) fn conv_silu_cols_f32(
    layer: &mut AffineLayer,
    w: &[i64],
    kernel: usize,
    flat: &Flat,
) -> Result<Vec<Vec<i64>>, StepError> {
    #[cfg(test)]
    let t_setup = std::time::Instant::now();
    let (k, ch) = (flat.k, flat.ch);
    let xs: &[i64] = &flat.data;
    let cols = kernel
        .checked_sub(1)
        .ok_or(StepError::BadConfig(String::from("conv_kernel")))?;
    if k == 0 || flat.ragged || w.len() != kernel * ch || layer.conv.len() != cols * ch {
        return Err(StepError::BadConfig(String::from("conv geometry")));
    }

    // A Q32 window (test and benchmark callers) is converted into scratch and rebuilt serially.
    let mut window_storage: Vec<F32> = Vec::new();
    if let ConvState::Q32(v) = &layer.conv {
        let mut wi = 0usize;
        while wi < v.len() {
            window_storage.push(f32_carrier::q32_to_f32(v[wi]));
            wi += 1;
        }
    }
    let window: &[F32] = match &layer.conv {
        ConvState::F32(v) => v.as_slice(),
        ConvState::Q32(_) => window_storage.as_slice(),
    };

    // The f32 column cache is the buffer shared by every layer of the state, locked for the
    // call. It is emptied and marked invalid until the dispatch completes; `clear` leaves it
    // empty, so `reserve_exact` moves no words, and the capacity is reached once and kept. It is
    // never zero-filled because the dispatch writes every word before `set_len`.
    let shared_arc = std::sync::Arc::clone(&layer.conv_shared);
    let mut shared = shared_arc.lock();
    shared.k = 0;
    shared.buf.clear();
    shared.buf.reserve_exact(k * ch);
    let by_col_f32 = &mut shared.buf;

    let mut by_col = alloc_cols(k, ch);
    let writers = col_writers(&mut by_col);
    let out_f32 = Shared(by_col_f32.spare_capacity_mut().as_mut_ptr().cast::<F32>());
    #[cfg(target_arch = "x86_64")]
    let avx2 = kernel == 4
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let avx2 = false;
    #[cfg(test)]
    seg::add(0, t_setup);

    let result = if k == 1 {
        range(
            w, window, xs, kernel, cols, ch, k, &out_f32, &writers, 0, ch, avx2,
        )
    } else {
        run_pooled(w, window, xs, kernel, cols, ch, k, &out_f32, &writers, avx2)
    };

    #[cfg(test)]
    let t_end = std::time::Instant::now();
    if result.is_ok() {
        // SAFETY: a refusal-free dispatch wrote all `ch` words of every column.
        unsafe { set_written_lens(&mut by_col, ch) };
        // SAFETY: `by_col_f32` was cleared (length 0) and `reserve_exact(k · ch)` ran, so its
        // capacity is at least `k · ch` whatever its previous length; the same dispatch wrote all
        // `k · ch` words of the spare capacity, so the new length covers initialised words only.
        unsafe { set_written_len(by_col_f32, k * ch) };
        advance_window(&mut layer.conv, xs, k, cols, ch);
        shared.owner = layer.slot;
        shared.k = k;
    }
    #[cfg(test)]
    seg::add(5, t_end);
    result.map(|()| by_col)
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "test data generation and oracle indexing"
)]
mod tests {
    use super::super::flatten_cols;
    use super::super::{AffineGeom, AffineState, Carrier, ConvState};
    use super::{conv_silu_cols_f32, seg};
    use crate::engine::fp::F32;
    use crate::engine::mixer::f32_carrier::{
        conv_silu_tap4_scalar, f32_readout_to_q32, q32_to_f32, readout8_avx2,
    };
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::Instant;

    fn t(x: f32) -> F32 {
        F32::from_bits(x.to_bits())
    }
    fn bits(v: &[F32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        #[allow(clippy::cast_precision_loss, reason = "test data")]
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }

    fn geom(ch_heads: (usize, usize, usize, usize)) -> AffineGeom {
        AffineGeom {
            conv_kernel: 4,
            n_k_heads: ch_heads.0,
            n_v_heads: ch_heads.1,
            head_k: ch_heads.2,
            head_v: ch_heads.3,
        }
    }

    /// Vector readout equals the scalar readout word for word on every class of lane.
    #[test]
    fn readout8_equals_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let mut vals: Vec<f32> = vec![
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            2.0f32.powi(-33),
            -(2.0f32.powi(-33)),
            2.0f32.powi(-32),
            -(2.0f32.powi(-32)),
            1.5 * 2.0f32.powi(-32),
            -1.5 * 2.0f32.powi(-32),
            2.5 * 2.0f32.powi(-32),
            0.5,
            -0.5,
            1.0,
            -1.0,
            262_143.75,
            -262_143.75,
            524_287.0,
            524_288.0,
            -524_288.0,
            2.0f32.powi(20),
            2.0f32.powi(30),
            2.0f32.powi(31) - 256.0,
            -(2.0f32.powi(31)),
            2.0f32.powi(31),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MAX,
        ];
        for _ in 0..200_000 {
            let e = (rng.next() % 50) as i32 - 35;
            vals.push((rng.unit() as f32) * 2.0f32.powi(e));
        }
        for _ in 0..200_000 {
            vals.push(f32::from_bits(rng.next() as u32));
        }
        // Blocks of 8 from the list, plus blocks with one special lane among ordinary ones.
        let mut blocks: Vec<[f32; 8]> = vals
            .chunks(8)
            .filter(|c| c.len() == 8)
            .map(|c| c.try_into().unwrap())
            .collect();
        for (i, &v) in vals.iter().take(400).enumerate() {
            let mut b = [0.25f32, -0.75, 3.0, -3.5, 1.0e-3, -1.0e-9, 100.5, -0.0];
            b[i % 8] = v;
            blocks.push(b);
        }
        for b in blocks {
            let expect: Result<Vec<i64>, _> = b.iter().map(|&o| f32_readout_to_q32(t(o))).collect();
            let mut got = [0i64; 8];
            // SAFETY: AVX2 checked; `got` holds 8 words.
            let res = unsafe {
                readout8_avx2(
                    std::arch::x86_64::_mm256_loadu_ps(b.as_ptr()),
                    got.as_mut_ptr(),
                )
            };
            match expect {
                Ok(e) => {
                    assert!(res.is_ok(), "{b:?}");
                    assert_eq!(&e[..], &got[..], "{b:?}");
                }
                Err(_) => assert!(res.is_err(), "{b:?}"),
            }
        }
    }

    /// The pooled call equals a serial scalar oracle over three successive calls: columns, f32
    /// cache bits and window, with a channel tail (43 = 5 groups of 8 + 3) and `k < kernel - 1`.
    #[test]
    fn conv_f32_calls_equal_scalar_oracle() {
        let g = geom((1, 1, 8, 27));
        let ch = 2 * 8 + 27;
        let mut rng = Rng(0x0BAD_CAFE_1357_9BDF);
        let w: Vec<i64> = (0..4 * ch)
            .map(|_| (rng.unit() * 4_294_967_296.0) as i64)
            .collect();
        let taps: Vec<[F32; 4]> = (0..ch)
            .map(|c| std::array::from_fn(|t| q32_to_f32(w[c * 4 + t])))
            .collect();
        let mut st = AffineState::with_slices_carrier(&g, 1, 1, Carrier::F32);
        let layer = st.layer_mut(0).unwrap();
        let mut win: Vec<Vec<F32>> = vec![vec![F32::ZERO; ch]; 3];
        for &k in &[5usize, 2, 1, 40] {
            let xs: Vec<Vec<i64>> = (0..k)
                .map(|_| {
                    (0..ch)
                        .map(|_| (rng.unit() * 6.0 * 4_294_967_296.0) as i64)
                        .collect()
                })
                .collect();
            let refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
            let got = conv_silu_cols_f32(layer, &w, 4, &flatten_cols(&refs)).unwrap();
            let mut hist = win.clone();
            hist.extend(
                xs.iter()
                    .map(|x| x.iter().map(|&v| q32_to_f32(v)).collect::<Vec<F32>>()),
            );
            let (cache, tag_k, owner) = {
                let sh = layer.conv_shared.lock();
                (sh.buf.clone(), sh.k, sh.owner)
            };
            assert_eq!((tag_k, owner), (k, 0));
            assert_eq!(cache.len(), k * ch);
            for j in 0..k {
                for c in 0..ch {
                    let y = conv_silu_tap4_scalar(
                        taps[c],
                        [hist[j][c], hist[j + 1][c], hist[j + 2][c], hist[j + 3][c]],
                    );
                    assert_eq!(
                        cache[j * ch + c].to_bits(),
                        y.to_bits(),
                        "k={k} j={j} c={c}"
                    );
                    assert_eq!(
                        got[j][c],
                        f32_readout_to_q32(y).unwrap(),
                        "k={k} j={j} c={c}"
                    );
                }
            }
            win = hist[k..k + 3].to_vec();
            let ConvState::F32(lw) = &layer.conv else {
                unreachable!()
            };
            for (r, row) in win.iter().enumerate() {
                for c in 0..ch {
                    assert_eq!(
                        lw[r * ch + c].to_bits(),
                        row[c].to_bits(),
                        "window k={k} r={r} c={c}"
                    );
                }
            }
        }
    }

    /// A refusal leaves the window unchanged and the f32 cache marked invalid.
    #[test]
    fn conv_f32_refusal_keeps_window() {
        let g = geom((1, 1, 8, 27));
        let ch = 2 * 8 + 27;
        let w = vec![1i64 << 32; 4 * ch];
        let mut st = AffineState::with_slices_carrier(&g, 1, 1, Carrier::F32);
        let layer = st.layer_mut(0).unwrap();
        let mut xs: Vec<Vec<i64>> = vec![vec![1 << 32; ch]; 6];
        let refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
        conv_silu_cols_f32(layer, &w, 4, &flatten_cols(&refs)).unwrap();
        let before = layer.conv.len();
        let ConvState::F32(w0) = &layer.conv else {
            unreachable!()
        };
        let w0 = w0.clone();
        xs[3][ch - 1] = i64::MAX;
        let big = vec![i64::MAX; 4 * ch];
        let refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
        assert!(conv_silu_cols_f32(layer, &big, 4, &flatten_cols(&refs)).is_err());
        let ConvState::F32(w1) = &layer.conv else {
            unreachable!()
        };
        assert_eq!(layer.conv.len(), before);
        assert_eq!(bits(&w0), bits(w1));
        assert_eq!(layer.conv_shared.lock().k, 0);
    }

    /// Two layers of one state share one buffer; the tag names the layer that wrote it, and a
    /// reused buffer whose previous length was larger (then smaller) holds exactly the new call.
    #[test]
    fn conv_f32_shared_buffer_tags_layer() {
        let g = geom((1, 1, 8, 27));
        let ch = 2 * 8 + 27;
        let w = vec![1i64 << 30; 4 * ch];
        let mut st = AffineState::with_slices_carrier(&g, 2, 1, Carrier::F32);
        let mut prev_ptr = None;
        for (slot, k) in [(0usize, 9usize), (1, 3), (0, 9), (1, 12)] {
            let layer = st.layer_mut(slot).unwrap();
            let xs: Vec<Vec<i64>> = (0..k)
                .map(|j| vec![(i64::try_from(slot * 100 + j + 1).unwrap()) << 28; ch])
                .collect();
            let refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
            conv_silu_cols_f32(layer, &w, 4, &flatten_cols(&refs)).unwrap();
            let sh = layer.conv_shared.lock();
            assert_eq!((sh.owner, sh.k, sh.buf.len()), (slot, k, k * ch));
            assert!(sh.buf.capacity() >= k * ch);
            // the shared buffer is reused (not reallocated) while it is large enough
            if k * ch <= 9 * ch {
                if let Some(p) = prev_ptr {
                    assert_eq!(sh.buf.as_ptr(), p);
                }
                prev_ptr = Some(sh.buf.as_ptr());
            }
        }
        let l0 = st.layer_mut(0).unwrap();
        let sh = l0.conv_shared.lock();
        assert_eq!(sh.owner, 1, "layer 1 wrote last");
    }

    /// Real-shape (k = 200, 6144 channels, 4 taps) attribution of one persistent layer's call,
    /// driven the way `affine_cols` drives it: persistent state, output dropped after the heads.
    /// Prints per-segment ms (median of 15 warm calls) and the caller-side drop.
    #[test]
    #[ignore = "release segment attribution: cargo test --release --lib conv_f32_segments -- --ignored --nocapture"]
    fn conv_f32_segments() {
        let (kernel, ch, k) = (4usize, 6144usize, 200usize);
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let w: Vec<i64> = (0..kernel * ch)
            .map(|_| (rng.unit() * 4_294_967_296.0) as i64)
            .collect();
        let xs: Vec<Vec<i64>> = (0..k)
            .map(|_| {
                (0..ch)
                    .map(|_| (rng.unit() * 4_294_967_296.0) as i64)
                    .collect()
            })
            .collect();
        let refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
        let mut st =
            AffineState::with_slices_carrier(&geom((16, 16, 128, 128)), 1, 1, Carrier::F32);
        let layer = st.layer_mut(0).unwrap();
        let names = [
            "setup(validate,take,cols alloc)",
            "pool_wall",
            "task_time(sum)",
            "task_time(max)",
            "set_len",
            "window+restore",
        ];
        for skip in [false, true] {
            seg::SKIP_READOUT.store(skip, Relaxed);
            let mut rows: Vec<[f64; 9]> = Vec::new();
            for it in 0..18 {
                for n in &seg::NS {
                    n.store(0, Relaxed);
                }
                let t0 = Instant::now();
                let out = conv_silu_cols_f32(layer, &w, kernel, &flatten_cols(&refs)).unwrap();
                let total = t0.elapsed().as_secs_f64() * 1e3;
                let t1 = Instant::now();
                drop(out);
                let drop_ms = t1.elapsed().as_secs_f64() * 1e3;
                if it >= 3 {
                    let mut r = [0.0; 9];
                    for (slot, n) in r.iter_mut().zip(&seg::NS) {
                        *slot = n.load(Relaxed) as f64 / 1e6;
                    }
                    r[6] = total;
                    r[7] = drop_ms;
                    rows.push(r);
                }
            }
            println!("skip_i64_readout={skip}");
            for (i, n) in names.iter().enumerate() {
                println!("  {n:<32} {:.3} ms", med(&rows, i));
            }
            println!("  {:<32} {:.3} ms", "call total", med(&rows, 6));
            println!("  {:<32} {:.3} ms", "caller drop(Vec<Vec>)", med(&rows, 7));
        }
        seg::SKIP_READOUT.store(false, Relaxed);
    }

    fn med(rows: &[[f64; 9]], i: usize) -> f64 {
        let mut v: Vec<f64> = rows.iter().map(|r| r[i]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }
}
