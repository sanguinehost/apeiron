//! The prefill (`k > 1`) f32 convolution: `conv_silu_cols_f32`, pooled over channel ranges.
//!
//! Each pool task owns channels `[c0, c1)` of every row: it converts the Q32 inputs it reads to
//! f32 in registers (each input word once), writes both the f32 column cache the heads dispatch
//! reads and the Q32 output columns, and advances its own channels of the window. No scratch is
//! allocated per call except the `k` Q32 output columns the signature returns; the f32 cache is
//! the state-wide `conv_shared` buffer, reused at its high-water capacity.

use super::{f32_carrier, AffineLayer, ColWriters, ConvState};
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

/// Everything a channel-range task reads or writes.
struct Job<'a> {
    w: &'a [i64],
    xs: &'a [&'a [i64]],
    kernel: usize,
    cols: usize,
    ch: usize,
    k: usize,
    /// History window, `cols` rows of `ch` f32; read only during the dispatch.
    window: Shared<f32>,
    out_f32: Shared<f32>,
    out_q32: &'a ColWriters,
}

impl Job<'_> {
    /// History row `r` (`r < cols`: window, else converted input row `r − cols`) at channel `c`.
    #[inline]
    fn hist(&self, r: usize, c: usize) -> f32 {
        if r < self.cols {
            // SAFETY: `r < cols`, `c < ch`; the window holds `cols · ch` words.
            unsafe { *self.window.0.add(r * self.ch + c) }
        } else {
            f32_carrier::q32_to_f32(self.xs[r - self.cols][c])
        }
    }

    /// Channels `[c0, c1)` of every column.
    fn range(&self, c0: usize, c1: usize, avx2: bool) -> Result<(), StepError> {
        let mut c = c0;
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            let c_end = c0 + (c1 - c0) / 8 * 8;
            // SAFETY: AVX2 and FMA were detected by the caller; `[c0, c_end)` lies inside `[0, ch)`.
            unsafe { self.range_avx2(c0, c_end)? };
            c = c_end;
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = avx2;
        if c < c1 {
            self.range_scalar(c, c1)?;
        }
        Ok(())
    }

    /// Generic per-channel path: any kernel width, channel tail, no AVX2.
    #[allow(
        clippy::many_single_char_names,
        clippy::needless_range_loop,
        reason = "w, x, c, t are the convolution's symbols"
    )]
    fn range_scalar(&self, c0: usize, c1: usize) -> Result<(), StepError> {
        let kernel = self.kernel;
        let mut taps = vec![0.0f32; kernel];
        let mut hist = vec![0.0f32; kernel];
        for c in c0..c1 {
            let base = c * kernel;
            for (slot, &w_val) in taps.iter_mut().zip(&self.w[base..base + kernel]) {
                *slot = f32_carrier::q32_to_f32(w_val);
            }
            for (idx, slot) in hist.iter_mut().enumerate() {
                *slot = self.hist(idx, c);
            }
            for j in 0..self.k {
                let mut acc = 0.0f32;
                for tap in 0..kernel {
                    acc = taps[tap].mul_add(hist[tap], acc);
                }
                if !acc.is_finite() {
                    return Err(StepError::Overflow);
                }
                let y = f32_carrier::silu_f32_scalar(acc);
                let y_q32 = f32_carrier::f32_readout_to_q32(y)?;
                // SAFETY: `(j, c)` with `c` in this task's range; column `j` reserved `ch` words.
                unsafe {
                    self.out_q32.write(j, c, y_q32);
                    *self.out_f32.0.add(j * self.ch + c) = y;
                }
                if j + 1 < self.k {
                    hist.copy_within(1..kernel, 0);
                    hist[kernel - 1] = self.hist(j + kernel, c);
                }
            }
        }
        Ok(())
    }

    /// Four-tap path over whole 8-channel groups `[c0, c_end)`. History rows slide through four
    /// registers, so each input row is converted once per group.
    ///
    /// # Safety
    ///
    /// AVX2 and FMA are available; `kernel == 4`; `c0 <= c_end <= ch`, `(c_end − c0) % 8 == 0`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn range_avx2(&self, c0: usize, c_end: usize) -> Result<(), StepError> {
        use std::arch::x86_64::{__m256, _mm256_loadu_ps, _mm256_storeu_ps};
        let (ch, k) = (self.ch, self.k);
        let load8 = |r: usize, c: usize| -> __m256 {
            if r < self.cols {
                // SAFETY: `r < cols`, `c + 8 <= ch`.
                unsafe { _mm256_loadu_ps(self.window.0.add(r * ch + c)) }
            } else {
                let x = &self.xs[r - self.cols][c..c + 8];
                let mut b = [0.0f32; 8];
                for (d, &s) in b.iter_mut().zip(x) {
                    *d = f32_carrier::q32_to_f32(s);
                }
                // SAFETY: `b` holds 8 f32.
                unsafe { _mm256_loadu_ps(b.as_ptr()) }
            }
        };
        let mut c = c0;
        while c < c_end {
            let mut t = [[0.0f32; 8]; 4];
            for lane in 0..8 {
                let base = (c + lane) * 4;
                for (tap, row) in t.iter_mut().enumerate() {
                    row[lane] = f32_carrier::q32_to_f32(self.w[base + tap]);
                }
            }
            // SAFETY: each `t[i]` holds 8 f32.
            let (v0, v1, v2, v3) = unsafe {
                (
                    _mm256_loadu_ps(t[0].as_ptr()),
                    _mm256_loadu_ps(t[1].as_ptr()),
                    _mm256_loadu_ps(t[2].as_ptr()),
                    _mm256_loadu_ps(t[3].as_ptr()),
                )
            };
            let (mut h0, mut h1, mut h2) = (load8(0, c), load8(1, c), load8(2, c));
            for j in 0..k {
                let h3 = load8(j + 3, c);
                // SAFETY: this function enables AVX2 and FMA.
                let vy =
                    unsafe { f32_carrier::conv_silu_tap4_avx2(v0, v1, v2, v3, h0, h1, h2, h3) };
                (h0, h1, h2) = (h1, h2, h3);
                // SAFETY: `(j, c..c + 8)` is in this task's channel range; column `j` reserved
                // `ch` words; the f32 cache holds `k · ch`.
                unsafe {
                    _mm256_storeu_ps(self.out_f32.0.add(j * ch + c), vy);
                    #[cfg(test)]
                    if seg::SKIP_READOUT.load(std::sync::atomic::Ordering::Relaxed) {
                        continue;
                    }
                    let dst = self.out_q32.0[j].add(c).cast::<i64>();
                    f32_carrier::readout8_avx2(vy, dst)?;
                }
            }
            c += 8;
        }
        Ok(())
    }
}

/// The new window: the history's last `cols` rows. It moves only on success, so a refusal leaves
/// the window as it was.
fn advance_window(conv: &mut ConvState, xs: &[&[i64]], k: usize, cols: usize, ch: usize) {
    match conv {
        ConvState::F32(window) => {
            for r in 0..cols {
                if k + r < cols {
                    window.copy_within((k + r) * ch..(k + r + 1) * ch, r * ch);
                } else {
                    for (d, &s) in window[r * ch..(r + 1) * ch]
                        .iter_mut()
                        .zip(xs[k + r - cols])
                    {
                        *d = f32_carrier::q32_to_f32(s);
                    }
                }
            }
        }
        ConvState::Q32(window) => {
            let mut next = vec![0i64; cols * ch];
            for r in 0..cols {
                for c in 0..ch {
                    next[r * ch + c] = if k + r < cols {
                        window[(k + r) * ch + c]
                    } else {
                        xs[k + r - cols][c]
                    };
                }
            }
            *window = next;
        }
    }
}

/// One pool dispatch over `max(4 · width, 96)` channel ranges. A range can refuse only with
/// `Overflow`, so which range's refusal is kept is immaterial.
fn run_pooled(job: &Job<'_>, avx2: bool) -> Result<(), StepError> {
    let ch = job.ch;
    let pool = crate::engine::chart::pool::Pool::global();
    let target_tasks = (4 * pool.width()).max(96);
    let per = ch.div_ceil(target_tasks).max(8).div_ceil(8) * 8;
    let tasks = ch.div_ceil(per);
    let first: std::sync::Mutex<Option<StepError>> = std::sync::Mutex::new(None);
    #[cfg(test)]
    let t_pool = std::time::Instant::now();
    pool.run(tasks, &|t| {
        let c0 = t * per;
        let c1 = (c0 + per).min(ch);
        if c0 >= c1 {
            return;
        }
        #[cfg(test)]
        let t_task = std::time::Instant::now();
        let res = job.range(c0, c1, avx2);
        #[cfg(test)]
        {
            seg::add(2, t_task);
            seg::max(3, t_task);
        }
        if let Err(e) = res {
            let mut f = first
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if f.is_none() {
                *f = Some(e);
            }
        }
    });
    #[cfg(test)]
    seg::add(1, t_pool);
    first.into_inner().unwrap_or_default().map_or(Ok(()), Err)
}

/// The causal depthwise f32 convolution and `SiLU` of `k > 1` columns; window, output columns and
/// the f32 column cache as [`super::conv_silu_cols`] describes.
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
    xs: &[&[i64]],
) -> Result<Vec<Vec<i64>>, StepError> {
    #[cfg(test)]
    let t_setup = std::time::Instant::now();
    let k = xs.len();
    let ch = xs.first().map_or(0, |x| x.len());
    let cols = kernel
        .checked_sub(1)
        .ok_or(StepError::BadConfig("conv_kernel"))?;
    if k == 0
        || xs.iter().any(|x| x.len() != ch)
        || w.len() != kernel * ch
        || layer.conv.len() != cols * ch
    {
        return Err(StepError::BadConfig("conv geometry"));
    }

    // A Q32 window (test and benchmark callers) is converted into scratch and rebuilt serially.
    let mut window_storage: Vec<f32> = Vec::new();
    let window_ptr = match &mut layer.conv {
        ConvState::F32(v) => v.as_mut_ptr(),
        ConvState::Q32(v) => {
            window_storage.extend(v.iter().copied().map(f32_carrier::q32_to_f32));
            window_storage.as_mut_ptr()
        }
    };

    // The f32 column cache is the buffer shared by every layer of the state, locked for the
    // call. It is emptied and marked invalid until the dispatch completes; `clear` leaves it
    // empty, so `reserve_exact` moves no words, and the capacity is reached once and kept. It is
    // never zero-filled because the dispatch writes every word before `set_len`.
    let shared_arc = std::sync::Arc::clone(&layer.conv_shared);
    let mut shared = shared_arc
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    shared.k = 0;
    shared.buf.clear();
    shared.buf.reserve_exact(k * ch);
    let by_col_f32 = &mut shared.buf;

    let mut by_col: Vec<Vec<i64>> = (0..k).map(|_| Vec::with_capacity(ch)).collect();
    let writers = ColWriters(
        by_col
            .iter_mut()
            .map(|v| v.spare_capacity_mut().as_mut_ptr())
            .collect(),
    );
    let job = Job {
        w,
        xs,
        kernel,
        cols,
        ch,
        k,
        window: Shared(window_ptr),
        out_f32: Shared(by_col_f32.spare_capacity_mut().as_mut_ptr().cast::<f32>()),
        out_q32: &writers,
    };
    let avx2 = kernel == 4
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma");
    #[cfg(test)]
    seg::add(0, t_setup);

    let result = if k == 1 {
        job.range(0, ch, avx2)
    } else {
        run_pooled(&job, avx2)
    };

    #[cfg(test)]
    let t_end = std::time::Instant::now();
    if result.is_ok() {
        for col in &mut by_col {
            // SAFETY: a refusal-free dispatch wrote all `ch` words of every column.
            unsafe { col.set_len(ch) };
        }
        // SAFETY: `by_col_f32` was cleared (length 0) and `reserve_exact(k · ch)` ran, so its
        // capacity is at least `k · ch` whatever its previous length; the same dispatch wrote all
        // `k · ch` words of the spare capacity, so the new length covers initialised words only.
        unsafe { by_col_f32.set_len(k * ch) };
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
    use super::super::{AffineGeom, AffineState, Carrier, ConvState};
    use super::{conv_silu_cols_f32, seg};
    use crate::engine::mixer::f32_carrier::{
        conv_silu_tap4_scalar, f32_readout_to_q32, q32_to_f32, readout8_avx2,
    };
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::Instant;

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
            let expect: Result<Vec<i64>, _> = b.iter().map(|&o| f32_readout_to_q32(o)).collect();
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
        let taps: Vec<[f32; 4]> = (0..ch)
            .map(|c| std::array::from_fn(|t| q32_to_f32(w[c * 4 + t])))
            .collect();
        let mut st = AffineState::with_slices_carrier(&g, 1, 1, Carrier::F32);
        let layer = st.layer_mut(0).unwrap();
        let mut win: Vec<Vec<f32>> = vec![vec![0.0; ch]; 3];
        for &k in &[5usize, 2, 1, 40] {
            let xs: Vec<Vec<i64>> = (0..k)
                .map(|_| {
                    (0..ch)
                        .map(|_| (rng.unit() * 6.0 * 4_294_967_296.0) as i64)
                        .collect()
                })
                .collect();
            let refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
            let got = conv_silu_cols_f32(layer, &w, 4, &refs).unwrap();
            let mut hist = win.clone();
            hist.extend(
                xs.iter()
                    .map(|x| x.iter().map(|&v| q32_to_f32(v)).collect::<Vec<f32>>()),
            );
            let (cache, tag_k, owner) = {
                let sh = layer.conv_shared.lock().unwrap();
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
        conv_silu_cols_f32(layer, &w, 4, &refs).unwrap();
        let before = layer.conv.len();
        let ConvState::F32(w0) = &layer.conv else {
            unreachable!()
        };
        let w0 = w0.clone();
        xs[3][ch - 1] = i64::MAX;
        let big = vec![i64::MAX; 4 * ch];
        let refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
        assert!(conv_silu_cols_f32(layer, &big, 4, &refs).is_err());
        let ConvState::F32(w1) = &layer.conv else {
            unreachable!()
        };
        assert_eq!(layer.conv.len(), before);
        assert_eq!(&w0, w1);
        assert_eq!(layer.conv_shared.lock().unwrap().k, 0);
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
            conv_silu_cols_f32(layer, &w, 4, &refs).unwrap();
            let sh = layer.conv_shared.lock().unwrap();
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
        let sh = l0.conv_shared.lock().unwrap();
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
                let out = conv_silu_cols_f32(layer, &w, kernel, &refs).unwrap();
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
