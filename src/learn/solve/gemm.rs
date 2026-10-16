//! Blocked, threaded, exact-accumulation Q32.32 matrix products: [`gemm_nn`] (`C = A·B`) and
//! [`gemm_nt`] (`C = A·Bᵀ`), the one GEMM the fold's low-rank corrector needs.
//!
//! Entry for entry, `C[i,j] = round32(Σ_l A[i,l]·B[l,j])` ([`gemm_nn`]) or
//! `C[i,j] = round32(Σ_l A[i,l]·B[j,l])` ([`gemm_nt`]): each Q32.32 `i64` product is formed exactly
//! in `i128`, the contraction sum is accumulated exactly in `i128`, and a single round-to-nearest
//! shift by 32 is applied once to the finished sum. This is the product `eigen_parts`
//! (`src/learn/fold/fit/corrector.rs`) forms as `w1 = ΔW·Hl` (`o×n×n`, [`gemm_nn`]) and
//! `m = w1·ΔWᵀ` (`o×o×n`, [`gemm_nt`]) in scalar `f64` — 140 s at `o = 896`, `n = 4864` — and the
//! one the later `stage`/`gptq_e8` steps need. [`gemm_nt`] is the cache-friendly form: both
//! operands are read as contiguous dot products.
//!
//! # Reuse
//!
//! The rounding ([`super::syrk::round32`]), the `i64` narrowing ([`super::syrk::narrow`]) and the
//! magnitude-bound scan ([`super::syrk::within_bound`] over [`super::syrk::BOUND_BITS`]) are the
//! `syrk` module's, promoted to `pub(super)` rather than copied, and [`MAX_K`] is derived from that
//! same `BOUND_BITS`. The blocked/threaded shape is `syrk`'s too: a plain `fn` block body driven by
//! `std::thread::scope` over disjoint output row chunks, one first-error return, `threads ≤ 1` the
//! same body with no spawn.
//!
//! # Relation to `dense::linear_fwd`
//!
//! `crate::learn::grad::dense::linear_fwd(x, w, n, i, o)` already computes exactly
//! `gemm_nt(x, w, n, i, o)`: `round32(Σ_k x[r,k]·w[c,k])` with `x` `n×i`, `w` `o×i`, the same
//! Q32.32 product, the same `(v + 2³¹) >> 32` rounding and the same `k`-ascending accumulation — as
//! a single-threaded, unbounded triple loop. [`gemm_nt`] is its blocked, threaded, bound-checked
//! generalisation and returns the same bits on the shared domain, so **`linear_fwd` should delegate
//! to `gemm_nt(x, w, n, i, o, 1)`** and map [`GemmError`], keeping only its `Empty`
//! zero-dimension policy; `dense.rs` is not edited here.
//!
//! # Exactness and the accepted domain
//!
//! Every `a` and `b` entry satisfies `|v| < 2³⁶` in raw Q32.32 units (real magnitudes below 16) and
//! the contraction length satisfies `k < MAX_K` with [`MAX_K`] `= 1 << (127 − 2·BOUND_BITS) = 2⁵⁵`.
//! The product of two entries is then below `2⁷²`, so the exact `i128` contraction sum is below
//! `k·2⁷² < 2¹²⁷`, and the round-half add `2³¹` keeps it inside `i128`
//! (`(2⁵⁵−1)·(2³⁶−1)² + 2³¹ < 2¹²⁷`). A violated bound is refused, not wrapped:
//! [`GemmError::InputBound`] for an entry at `2³⁶`, [`GemmError::Overflow`] for `k ≥ MAX_K`, for a
//! dimension product that leaves `usize`, or for a rounded entry that leaves `i64`.
//!
//! # Blocking and the tile
//!
//! The contraction is cut into `KC`-deep cache blocks; within a block an `MC × NC` output tile
//! accumulates in a `i128` array while the packed `MC × KC` A panel and `NC × KC` B panel are
//! read. With `MC = NC = 64`, `KC = 128` the A and B panels are `64 KiB` each and the accumulator
//! tile `64 KiB` — a `192 KiB` working set that fits this box's per-core L2 (`32 MiB` across 12
//! instances, `≥ 2 MiB` per core). The inner kernel is a `4 × 4` tile of `i128` accumulators
//! ([`MR`] `×` [`NR`]) held across the block's `kk` loop, a plain safe-Rust loop with no `unsafe`
//! and no intrinsics; the compiler lowers `i64 × i64 → i128` to a widening multiply and a
//! two-word add and does not vectorise it, which the `#[ignore]` timing test reports as a measured
//! GMAC/s rather than assumes away.
//!
//! Grade: `tested` (`tests/learn_solve_gemm.rs`); no body here is extracted or refined.

#![allow(
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::must_use_candidate,
    clippy::similar_names,
    reason = "`usize::from` is not usable in the `const MAX_K` initialiser, so the `BOUND_BITS as \
              usize` widening stays; the single-letter names are the operands' index symbols (`m`, `k`, `n`, `i`, `j`, \
              `l`, `kb`, `kc`, `ii`, `jj`) that read closer to the arithmetic than spelled-out \
              names; the `4 × 4` register tile names `c00..c33`, `a0..a3`, `b0..b3`. The `Result` \
              returns are already `#[must_use]`."
)]

use super::q32::{from_grid, panel_shift, to_grid, INPUT_BOUND};
use super::syrk::{narrow, round32, within_bound, BOUND_BITS};

/// The largest accepted contraction length: `k < MAX_K` keeps the exact `i128` sum plus the
/// round-half add inside `i128` (`(MAX_K−1)·(2³⁶−1)² + 2³¹ < 2¹²⁷`). `127 − 2·BOUND_BITS = 55`.
pub const MAX_K: usize = 1 << (127 - 2 * BOUND_BITS as usize);

/// Rows of the A cache panel: `MC · KC · 8 B = 64 KiB`.
const MC: usize = 64;
/// Columns of the B cache panel: `NC · KC · 8 B = 64 KiB`.
const NC: usize = 64;
/// Contraction depth of a cache block: `(MC + NC) · KC · 8 B = 128 KiB` of panel.
const KC: usize = 128;
/// Rows of the register tile of `i128` accumulators.
const MR: usize = 4;
/// Columns of the register tile of `i128` accumulators.
const NR: usize = 4;

/// A refused Q32.32 matrix product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmError {
    /// A dimension is zero (`m == 0`, `k == 0` or `n == 0`), so the product has no entries.
    ZeroDim,
    /// A slice length disagrees with the declared shape (`a` is `[m, k]`, `b` is `[k, n]` for
    /// [`gemm_nn`] and `[n, k]` for [`gemm_nt`]).
    ShapeMismatch,
    /// An entry of `a` or `b` reaches the documented magnitude bound `2³⁶`.
    InputBound,
    /// `k ≥ `[`MAX_K`], a dimension product left `usize`, or a rounded entry left `i64`.
    Overflow,
}

/// Accumulate one `MR × NR` register tile of the `acc` block into `acc` over the packed A and B
/// panels: `acc[i,j] += Σ_kk ap[i·kc+kk]·bp[j·kc+kk]` for `i ∈ [ii, ii+MR)`, `j ∈ [jj, jj+NR)`.
///
/// `acc` is row-major `[mc, NC]`, `ap` row-major `[mc, kc]` and `bp` row-major `[nc, kc]`, with
/// `kc` the packed block depth. The read rows and columns are in range because `MC`/`NC` are
/// multiples of `MR`/`NR` and `ii ≤ mc ≤ MC`, `jj ≤ nc ≤ NC`. Rows and columns past `mc`/`nc` hold
/// pack padding and touch only accumulator entries the caller does not store. The `i128` products
/// and adds cannot overflow under the [`MAX_K`] contract, so they are unchecked here; the caller
/// checks the rounding add and the narrowing.
///
/// # Complexity
/// Time: `MR·NR·kc` `i128` multiply-accumulates and `MR·NR·(kc+1)` `i128` loads/stores — the `kk`
/// loop carries the `MR·NR` accumulators, one A-row element and one B-row element per step.
/// Space: `O(1)` — the accumulators are register-resident and nothing is allocated.
fn micro_4x4(ap: &[i64], bp: &[i64], acc: &mut [i128], kc: usize, ii: usize, jj: usize) {
    let a0 = &ap[ii * kc..ii * kc + kc];
    let a1 = &ap[(ii + 1) * kc..(ii + 1) * kc + kc];
    let a2 = &ap[(ii + 2) * kc..(ii + 2) * kc + kc];
    let a3 = &ap[(ii + 3) * kc..(ii + 3) * kc + kc];
    let b0 = &bp[jj * kc..jj * kc + kc];
    let b1 = &bp[(jj + 1) * kc..(jj + 1) * kc + kc];
    let b2 = &bp[(jj + 2) * kc..(jj + 2) * kc + kc];
    let b3 = &bp[(jj + 3) * kc..(jj + 3) * kc + kc];
    let base = ii * NC + jj;
    let mut c00 = acc[base];
    let mut c01 = acc[base + 1];
    let mut c02 = acc[base + 2];
    let mut c03 = acc[base + 3];
    let mut c10 = acc[base + NC];
    let mut c11 = acc[base + NC + 1];
    let mut c12 = acc[base + NC + 2];
    let mut c13 = acc[base + NC + 3];
    let mut c20 = acc[base + 2 * NC];
    let mut c21 = acc[base + 2 * NC + 1];
    let mut c22 = acc[base + 2 * NC + 2];
    let mut c23 = acc[base + 2 * NC + 3];
    let mut c30 = acc[base + 3 * NC];
    let mut c31 = acc[base + 3 * NC + 1];
    let mut c32 = acc[base + 3 * NC + 2];
    let mut c33 = acc[base + 3 * NC + 3];
    let mut kk = 0usize;
    while kk < kc {
        let x0 = i128::from(a0[kk]);
        let x1 = i128::from(a1[kk]);
        let x2 = i128::from(a2[kk]);
        let x3 = i128::from(a3[kk]);
        let y0 = i128::from(b0[kk]);
        let y1 = i128::from(b1[kk]);
        let y2 = i128::from(b2[kk]);
        let y3 = i128::from(b3[kk]);
        c00 += x0 * y0;
        c01 += x0 * y1;
        c02 += x0 * y2;
        c03 += x0 * y3;
        c10 += x1 * y0;
        c11 += x1 * y1;
        c12 += x1 * y2;
        c13 += x1 * y3;
        c20 += x2 * y0;
        c21 += x2 * y1;
        c22 += x2 * y2;
        c23 += x2 * y3;
        c30 += x3 * y0;
        c31 += x3 * y1;
        c32 += x3 * y2;
        c33 += x3 * y3;
        kk += 1;
    }
    acc[base] = c00;
    acc[base + 1] = c01;
    acc[base + 2] = c02;
    acc[base + 3] = c03;
    acc[base + NC] = c10;
    acc[base + NC + 1] = c11;
    acc[base + NC + 2] = c12;
    acc[base + NC + 3] = c13;
    acc[base + 2 * NC] = c20;
    acc[base + 2 * NC + 1] = c21;
    acc[base + 2 * NC + 2] = c22;
    acc[base + 2 * NC + 3] = c23;
    acc[base + 3 * NC] = c30;
    acc[base + 3 * NC + 1] = c31;
    acc[base + 3 * NC + 2] = c32;
    acc[base + 3 * NC + 3] = c33;
}

/// Pack rows `[i0, i0+mc)` and contraction columns `[kb, kb+kc)` of the row-major `[m, k]` `a` into
/// the contiguous `ap`, row stride `kc`.
///
/// # Complexity
/// `mc·kc` `i64` copies and `O(1)` extra space; derivation: one `counter`-free row copy per row, one
/// [`copy_from_slice`](slice::copy_from_slice) of `kc` words each.
fn pack_a(a: &[i64], k: usize, i0: usize, mc: usize, kb: usize, kc: usize, ap: &mut [i64]) {
    let mut i = 0usize;
    while i < mc {
        let src = (i0 + i) * k + kb;
        let dst = i * kc;
        ap[dst..dst + kc].copy_from_slice(&a[src..src + kc]);
        i += 1;
    }
}

/// Pack rows `[j0, j0+nc)` and contraction columns `[kb, kb+kc)` of the row-major `[n, k]` `b` (the
/// [`gemm_nt`] layout) into the contiguous `bp`, row stride `kc`.
///
/// # Complexity
/// `nc·kc` `i64` copies and `O(1)` extra space; derivation as [`pack_a`].
fn pack_b_nt(b: &[i64], k: usize, j0: usize, nc: usize, kb: usize, kc: usize, bp: &mut [i64]) {
    let mut j = 0usize;
    while j < nc {
        let src = (j0 + j) * k + kb;
        let dst = j * kc;
        bp[dst..dst + kc].copy_from_slice(&b[src..src + kc]);
        j += 1;
    }
}

/// Pack output columns `[j0, j0+nc)` and contraction rows `[kb, kb+kc)` of the row-major `[k, n]`
/// `b` (the [`gemm_nn`] layout) into the `[nc, kc]` `bp`, transposing the block so the inner kernel
/// reads B as contiguous rows.
///
/// # Complexity
/// `nc·kc` `i64` copies and `O(1)` extra space; derivation: one `kc`-long row segment of `b` per
/// column `j`, transposed on the copy.
fn pack_b_nn(b: &[i64], n: usize, j0: usize, nc: usize, kb: usize, kc: usize, bp: &mut [i64]) {
    let mut j = 0usize;
    while j < nc {
        let mut kk = 0usize;
        while kk < kc {
            bp[j * kc + kk] = b[(kb + kk) * n + j0 + j];
            kk += 1;
        }
        j += 1;
    }
}

/// Compute all `rows` output rows of one worker's `c` (row stride `n`) for `C = A·B` (`nn`) or
/// `C = A·Bᵀ` (`nt`) with `B` in the `nn` (`[k, n]`, `true`) or `nt` (`[n, k]`, `false`) layout, in
/// `MC × NC` output blocks over `KC`-deep packed panels with the [`micro_4x4`] register kernel. `a` is this worker's `rows × k`
/// row block, so the local row `i` of `c` is row `i` of `a`.
///
/// A given output entry is accumulated over the `k`-blocks in order and within a block over `kk`
/// in order — that is ascending `k`, exactly once and by this worker alone — so the result is
/// independent of the block geometry and the thread count. `c.len()` must be `rows·n`.
///
/// # Complexity
/// Time: `rows·k·n` `i128` multiply-accumulates plus `rows·k·n·(1/MC + 1/NC)` `i64` panel copies
/// (`rows·k·n/32` at the chosen tiles). Derivation: the output is `⌈rows/MC⌉·⌈n/NC⌉` blocks, each
/// cost `k·MC·NC` MACs and `k·(MC + NC)` panel copies. Space: `O(MC·KC + NC·KC) i64` for the panels
/// plus `O(MC·NC) i128` for the accumulator tile, per worker, reused across the whole block.
fn block_rows(
    c: &mut [i64],
    rows: usize,
    a: &[i64],
    b: &[i64],
    n: usize,
    k: usize,
    nn: bool,
) -> Result<(), GemmError> {
    if rows == 0 {
        return Ok(());
    }
    let mut acc = vec![0i128; MC * NC];
    let mut ap = vec![0i64; MC * KC];
    let mut bp = vec![0i64; NC * KC];
    let mut i0 = 0usize;
    while i0 < rows {
        let mc = (i0 + MC).min(rows) - i0;
        let mut jb = 0usize;
        while jb < n {
            let nc = (jb + NC).min(n) - jb;
            acc[..].fill(0);
            let mut kb = 0usize;
            while kb < k {
                let kc = (kb + KC).min(k) - kb;
                pack_a(a, k, i0, mc, kb, kc, &mut ap);
                if nn {
                    pack_b_nn(b, n, jb, nc, kb, kc, &mut bp);
                } else {
                    pack_b_nt(b, k, jb, nc, kb, kc, &mut bp);
                }
                let mut ii = 0usize;
                while ii < mc {
                    let mut jj = 0usize;
                    while jj < nc {
                        micro_4x4(&ap, &bp, &mut acc, kc, ii, jj);
                        jj += NR;
                    }
                    ii += MR;
                }
                kb += kc;
            }
            let mut i = 0usize;
            while i < mc {
                let mut j = 0usize;
                while j < nc {
                    let Some(r) = round32(acc[i * NC + j]) else {
                        return Err(GemmError::Overflow);
                    };
                    let Some(v) = narrow(r) else {
                        return Err(GemmError::Overflow);
                    };
                    c[(i0 + i) * n + jb + j] = v;
                    j += 1;
                }
                i += 1;
            }
            jb += nc;
        }
        i0 += mc;
    }
    Ok(())
}

/// Split `c` (`m × n`, row stride `n`) into `parts` contiguous row chunks and run [`block_rows`] on
/// each in scoped threads; returns the first refusal. Every worker owns a disjoint row range of `c`
/// and is handed the matching `a` row block (`&a[lo·k .. hi·k]`), so no two workers write one output
/// entry and none reads another's A rows. `parts ≥ 2`: the caller runs [`block_rows`] itself for
/// `threads ≤ 1`, so the serial arithmetic is not behind this thread boundary.
///
/// # Complexity
/// Time: [`block_rows`] plus `O(m)` split bookkeeping. Space: `O(parts)` chunk descriptors plus each
/// worker's `O(MC·KC + NC·KC + MC·NC)` scratch.
///
/// # Panics
/// Re-raises the payload of a worker panic on the calling thread; the body does not panic on caller
/// input, so a worker panic is an internal bug.
#[allow(
    clippy::too_many_arguments,
    reason = "the driver takes the output slice plus the two operands and the `m`, `n`, `k` geometry \
              and the layout flag; a struct would only relocate the fields."
)]
fn block_rows_par(
    c: &mut [i64],
    m: usize,
    n: usize,
    parts: usize,
    a: &[i64],
    b: &[i64],
    k: usize,
    nn: bool,
) -> Result<(), GemmError> {
    let base = m / parts;
    let rem = m % parts;
    let mut lo = 0usize;
    let mut rest: &mut [i64] = c;
    let mut groups: Vec<(&[i64], usize, &mut [i64])> = Vec::with_capacity(parts);
    let mut t = 0usize;
    while t < parts {
        let hi = lo + base + usize::from(t < rem);
        let (head, tail) = rest.split_at_mut((hi - lo) * n);
        groups.push((&a[lo * k..hi * k], hi - lo, head));
        rest = tail;
        lo = hi;
        t += 1;
    }
    let err: Option<GemmError> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(groups.len());
        for (ablock, rows, part) in groups {
            handles.push(scope.spawn(move || block_rows(part, rows, ablock, b, n, k, nn)));
        }
        let mut first: Option<GemmError> = None;
        let mut panicked = None;
        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if first.is_none() {
                        first = Some(e);
                    }
                }
                Err(p) => {
                    if panicked.is_none() {
                        panicked = Some(p);
                    }
                }
            }
        }
        if let Some(p) = panicked {
            std::panic::resume_unwind(p);
        }
        first
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Check the shared domain of [`gemm_nn`] and [`gemm_nt`]: a non-zero `m`, `k`, `n`; `k < `[`MAX_K`]
/// and dimension products that fit `usize`; `a.len() == m·k` and `b.len() == k·n` (the length is the
/// same in either layout); and every entry of `a` and `b` below `2³⁶`.
///
/// # Complexity
/// `O(m·k + k·n)` bound comparisons and `O(1)` extra space.
fn check(a: &[i64], b: &[i64], m: usize, k: usize, n: usize) -> Result<(), GemmError> {
    if m == 0 || k == 0 || n == 0 {
        return Err(GemmError::ZeroDim);
    }
    if k >= MAX_K {
        return Err(GemmError::Overflow);
    }
    let Some(ak) = m.checked_mul(k) else {
        return Err(GemmError::Overflow);
    };
    let Some(kn) = k.checked_mul(n) else {
        return Err(GemmError::Overflow);
    };
    if n.checked_mul(m).is_none() {
        return Err(GemmError::Overflow);
    }
    if a.len() != ak || b.len() != kn {
        return Err(GemmError::ShapeMismatch);
    }
    if !within_bound(a, Some(b)) {
        return Err(GemmError::InputBound);
    }
    Ok(())
}

/// `C[m,n] = A[m,k]·B[k,n]` in Q32.32: `a` row-major `[m, k]`, `b` row-major `[k, n]`, `c`
/// row-major `[m, n]` with `C[i,j] = round32(Σ_l a[i,l]·b[l,j])`, one rounding per entry.
///
/// `threads` splits the output rows across `std::thread::scope` workers; `threads ≤ 1` runs the same
/// body with no spawn. Each entry is accumulated by exactly one worker over ascending `k`, so the
/// result is identical for every thread count.
///
/// # Errors
/// [`GemmError::ZeroDim`] if `m`, `k` or `n` is zero; [`GemmError::ShapeMismatch`] if `a.len() !=
/// m·k` or `b.len() != k·n`; [`GemmError::InputBound`] if an entry reaches `2³⁶`;
/// [`GemmError::Overflow`] if `k ≥ `[`MAX_K`], a dimension product leaves `usize`, or a rounded
/// entry leaves `i64`.
///
/// # Complexity
/// Time: `m·k·n` `i128` multiply-accumulates plus `m·k·n·(1/MC + 1/NC)` panel copies, split across
/// `threads`; the derivation is in [`block_rows`]. Space: `O(m·n)` for `c` plus each worker's
/// `O(MC·KC + NC·KC + MC·NC)` scratch.
pub fn gemm_nn(
    a: &[i64],
    b: &[i64],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) -> Result<Vec<i64>, GemmError> {
    run(a, b, m, k, n, threads, true)
}

/// `x` as a Q32.32 panel below the [`gemm_nn`] input bound, with the exponent `s` it was scaled by.
///
/// `s = `[`panel_shift`]`(max|x|)` is the least `s ≥ 0` with `max|x|·2⁻ˢ ≤ 8`, an exact scaling, so
/// the only loss is the one grid rounding of [`to_grid`]. A non-finite entry is
/// [`GemmError::InputBound`].
///
/// # Complexity
/// Time `O(x.len())`: one magnitude scan, one scale and one rounding per entry. Space `O(x.len())`.
fn to_panel(x: &[f64]) -> Result<(Vec<i64>, i32), GemmError> {
    let mut max = 0.0_f64;
    for &v in x {
        if !v.is_finite() {
            return Err(GemmError::InputBound);
        }
        max = max.max(v.abs());
    }
    let shift = panel_shift(max);
    let scale = 2.0_f64.powi(-shift);
    let mut out = Vec::with_capacity(x.len());
    for &v in x {
        out.push(to_grid(v * scale, INPUT_BOUND).ok_or(GemmError::InputBound)?);
    }
    Ok((out, shift))
}

/// `C[m,n] = A[m,k]·B[k,n]` for `f64` operands on the Q32.32 [`gemm_nn`] path: the one place an
/// `f64` matrix product crosses the grid.
///
/// Each operand is scaled by the exact power of two `2⁻ˢ` with `s = `[`panel_shift`]`(max|·|)`, the
/// least `s ≥ 0` that brings its largest entry to `≤ 8`, below the [`gemm_nn`] bound `16` with
/// head-room ([`super::q32::fill_scale`] fills to `2³¹`, above that bound, so it is not used). The
/// grid product carries the scale `2⁻(sₐ+s_b)`, undone exactly by `2^(sₐ+s_b)` on the read-back.
/// Entries are rounded to the `2⁻³²` grid once on the way in and the product once per entry on the
/// way out; a panel already within `8` is not scaled. Bit-identical for every `threads`.
///
/// # Errors
/// As [`gemm_nn`] for the shape, with a non-finite entry or an entry that does not fit the grid
/// after scaling as [`GemmError::InputBound`], and an undo exponent or a read-back entry that
/// leaves `f64`'s finite range as [`GemmError::Overflow`].
///
/// # Complexity
/// Time `m·k·n` `i128` multiply-accumulates in [`gemm_nn`] plus `O(m·k + k·n)` grid conversions and
/// `O(m·n)` exact undo multiplies. Space `O(m·n)` for the result plus the two `i64` panels.
pub fn gemm_nn_f64(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) -> Result<Vec<f64>, GemmError> {
    product_f64(a, b, m, k, n, threads, true)
}

/// `C[m,n] = A[m,k]·B[n,k]ᵀ` for `f64` operands on the Q32.32 [`gemm_nt`] path: [`gemm_nn_f64`]
/// with `B` stored row-major `[n, k]`. Same scaling rule and thread independence.
///
/// # Errors
/// As [`gemm_nn_f64`], with `b` of shape `[n, k]`.
///
/// # Complexity
/// As [`gemm_nn_f64`].
pub fn gemm_nt_f64(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) -> Result<Vec<f64>, GemmError> {
    product_f64(a, b, m, k, n, threads, false)
}

/// The shared body of [`gemm_nn_f64`] and [`gemm_nt_f64`]; `nn` selects the `B` layout.
///
/// # Complexity
/// As [`gemm_nn_f64`].
fn product_f64(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
    nn: bool,
) -> Result<Vec<f64>, GemmError> {
    let (pa, sa) = to_panel(a)?;
    let (pb, sb) = to_panel(b)?;
    let c = if nn {
        gemm_nn(&pa, &pb, m, k, n, threads)?
    } else {
        gemm_nt(&pa, &pb, m, k, n, threads)?
    };
    let back = sa
        .checked_add(sb)
        .map(|e| 2.0_f64.powi(e))
        .filter(|v| v.is_finite())
        .ok_or(GemmError::Overflow)?;
    let mut out = Vec::with_capacity(c.len());
    for &v in &c {
        let x = from_grid(v) * back;
        if !x.is_finite() {
            return Err(GemmError::Overflow);
        }
        out.push(x);
    }
    Ok(out)
}

/// `C[m,n] = A[m,k]·B[n,k]ᵀ` in Q32.32: `a` row-major `[m, k]`, `b` row-major `[n, k]`, `c`
/// row-major `[m, n]` with `C[i,j] = round32(Σ_l a[i,l]·b[j,l])`, one rounding per entry. Both
/// operand rows are contiguous in the contraction index, so this is the cache-friendly form and the
/// one `m = w1·ΔWᵀ` needs.
///
/// `threads` splits the output rows exactly as [`gemm_nn`] does, with the same thread-count
/// independence.
///
/// # Errors
/// As [`gemm_nn`], with `b` of shape `[n, k]`.
///
/// # Complexity
/// Time and space as [`gemm_nn`], with `B` read as contiguous rows.
pub fn gemm_nt(
    a: &[i64],
    b: &[i64],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) -> Result<Vec<i64>, GemmError> {
    run(a, b, m, k, n, threads, false)
}

/// The shared body of [`gemm_nn`] and [`gemm_nt`]: validate, allocate `C`, and drive
/// [`block_rows_par`] with `nn` selecting the B-panel layout.
///
/// # Complexity
/// As [`gemm_nn`].
fn run(
    a: &[i64],
    b: &[i64],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
    nn: bool,
) -> Result<Vec<i64>, GemmError> {
    check(a, b, m, k, n)?;
    let mut c = vec![0i64; m * n];
    if threads <= 1 {
        block_rows(&mut c, m, a, b, n, k, nn)?;
    } else {
        block_rows_par(&mut c, m, n, threads, a, b, k, nn)?;
    }
    Ok(c)
}
