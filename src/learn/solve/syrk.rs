//! Weighted exact-integer SYRK: `G = Σ_o w_o · x_o x_oᵀ` over Q32.32 rows with one rounding.
//!
//! [`wsyrk`] forms the weighted Gram `G[i,j] = round(Σ_o w[o]·x[o,i]·x[o,j] / 2⁶⁴)` of an
//! `o × n` Q32.32 panel in blocked, threaded, checked `i128` arithmetic; [`hadamard_into`] applies
//! the elementwise `A ∘ hl` factor. Both are the exact-integer body of the `A`-formation in
//! `crate::learn::fold::fit::steps::c_step` (`A[i,j] = hl[i,j]·Σ_o Sh[o,i]·Sh[o,j]·g_row[o]`),
//! where the Python-ported loop accumulates in f64 with a strided scalar reduction.
//!
//! # Exactness and the intermediate width
//!
//! Each factor is Q32.32. `x[o,i]·x[o,j]` is Q64.64 and exact as an `i128` (the product of two
//! `i64`); weighting by the Q32.32 `w[o]` multiplies the scale again to Q96.96, also exact in
//! `i128`. The `o`-sum is accumulated exactly in `i128`, and a single round-to-nearest shift by 64
//! — `(v + 2⁶³) >> 64`, the same add-half-then-shift rule as `crate::learn::grad::dense`'s `round32`
//! — returns Q32.32. No intermediate, term, or output is rounded.
//!
//! # Accepted domain
//!
//! `|x[o,i]| < 2^k` and, when present, `|w[o]| < 2^k`, with `k = 36` in raw Q32.32 units (real
//! magnitudes below 16), and `o < 2¹⁸`. Every triple product is then below `2¹⁰⁸` and the exact
//! `i128` sum below `2¹²⁶`, inside `i128` with room for the round-half add `2⁶³`. A violated bound is
//! refused, not wrapped. `w = None` is the unit weight, raw `2³²`.
//!
//! Grade: `tested` (`tests/learn_solve_syrk.rs`); the bodies are not extracted or refined.

#![allow(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::must_use_candidate,
    clippy::similar_names,
    reason = "the widening `i128::from` forms are total and the narrowing `as i64` is range-checked \
              against `i64::MAX`/`i64::MIN` immediately before it; the single-letter names are the \
              recurrence symbols of `c_step` (`o`, `n`, `i`, `j`) plus the block geometry (`jb`, \
              `j1`, `nb`), which reads closer to the arithmetic than spelled-out names."
)]

/// Raw Q32.32 magnitude bound exponent: every `x` and `w` entry satisfies `|v| < 2³⁶`.
const BOUND_BITS: u32 = 36;
/// The largest accepted `o`: `o < 2¹⁸` keeps `o · 2^{3·BOUND_BITS}` inside `i128`.
const MAX_O: usize = 1 << 18;
/// The unit weight in raw Q32.32 (`1.0`), the value used when `w` is `None`.
const UNIT: i128 = 1_i128 << 32;
/// Column-block width of the blocked accumulation (64 Q32.32 words = 512 B: one cache line per
/// accumulator row).
const BLOCK: usize = 64;

/// A refused SYRK or hadamard fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyrkError {
    /// A slice length disagrees with the declared `o × n` shape.
    ShapeMismatch,
    /// `n == 0`: the output Gram has no entries.
    ZeroN,
    /// `o == 0`: the reduction has no rows.
    ZeroO,
    /// An `x` or `w` entry reaches the documented magnitude bound `2^k` (`k = 36`).
    InputBound,
    /// A dimension product, the `o < 2¹⁸` domain bound, or a checked `i128` operation left its range.
    Overflow,
}

/// `(v + 2³¹) >> 32` — round-to-nearest of an `i128` Q64.64 product to Q32.32, or `None` when the
/// rounding add leaves `i128`.
fn round32(v: i128) -> Option<i128> {
    v.checked_add(1_i128 << 31).map(|s| s >> 32)
}

/// `(v + 2⁶³) >> 64` — round-to-nearest of the Q96.96 triple-product accumulator to Q32.32, or
/// `None` when the rounding add leaves `i128`.
fn round64(v: i128) -> Option<i128> {
    v.checked_add(1_i128 << 63).map(|s| s >> 64)
}

/// `v` as `i64`, or `None` when it is outside `i64`.
fn narrow(v: i128) -> Option<i64> {
    if v > i128::from(i64::MAX) || v < i128::from(i64::MIN) {
        None
    } else {
        Some(v as i64)
    }
}

/// `true` iff every `x` entry, and every `w` entry when present, is below the magnitude bound
/// `2^k`. `unsigned_abs` is total, so `i64::MIN` is refused rather than wrapping.
///
/// # Complexity
/// `O(o·n)` comparisons, `O(1)` extra space.
fn within_bound(x: &[i64], w: Option<&[i64]>) -> bool {
    let limit = 1_u64 << BOUND_BITS;
    if !x.iter().all(|&v| v.unsigned_abs() < limit) {
        return false;
    }
    match w {
        Some(ws) => ws.iter().all(|&v| v.unsigned_abs() < limit),
        None => true,
    }
}

/// Accumulate the block's owned `acc` rows `[lo, hi)` of the column block `[jb, j1)` over every `o`
/// row: `acc[i−lo, j−jb] += w[o]·x[o,i]·x[o,j]` for `j ≥ max(i, jb)` in checked `i128`. `acc` is
/// row-major `[hi−lo, j1−jb]`, so worker `t` owns rows `[lo, hi)` of the block and writes disjoint
/// entries; each entry is accumulated once, in `o` order, so the result is worker-count-independent.
///
/// # Complexity
/// `o·Σ_{i<lo..hi}(j1 − max(i, jb))` checked multiply-accumulates and `O(1)` extra words; every
/// index is in range by construction (`hi ≤ j1`, `acc.len() = (hi−lo)·(j1−jb)`).
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel is handed the block accumulator slice plus the read-only panel and the \
              block geometry (`lo`, `hi`, `o`, `n`, `jb`, `j1`); a struct would only relocate the \
              fields."
)]
fn block_rows(
    acc: &mut [i128],
    lo: usize,
    hi: usize,
    x: &[i64],
    w: Option<&[i64]>,
    o: usize,
    n: usize,
    jb: usize,
    j1: usize,
) -> Result<(), SyrkError> {
    let nb = j1 - jb;
    let mut oo = 0usize;
    while oo < o {
        let ww = match w {
            Some(ws) => i128::from(ws[oo]),
            None => UNIT,
        };
        let xo = &x[oo * n..oo * n + n];
        let mut i = lo;
        while i < hi {
            let Some(a) = ww.checked_mul(i128::from(xo[i])) else {
                return Err(SyrkError::Overflow);
            };
            let row = &mut acc[(i - lo) * nb..(i - lo) * nb + nb];
            let mut j = if i > jb { i } else { jb };
            while j < j1 {
                let Some(t) = a.checked_mul(i128::from(xo[j])) else {
                    return Err(SyrkError::Overflow);
                };
                match row[j - jb].checked_add(t) {
                    Some(s) => row[j - jb] = s,
                    None => return Err(SyrkError::Overflow),
                }
                j += 1;
            }
            i += 1;
        }
        oo += 1;
    }
    Ok(())
}

/// Split the `rows × nb` block accumulator `acc` into `parts` contiguous row chunks and run
/// [`block_rows`] on each in scoped threads; returns the first refusal. Every worker owns a
/// disjoint row range, so no two workers write one entry. `parts ≤ 1` runs the serial body with no
/// spawn. `O(rows·nb)` split work; `O(parts)` bookkeeping.
///
/// # Panics
/// Re-raises the payload of a worker panic on the calling thread; the kernel does not panic on
/// caller input, so a worker panic is an internal bug.
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel is handed the block accumulator slice plus the read-only panel and the \
              block geometry (`rows`, `nb`, `parts`, `o`, `n`, `jb`, `j1`); a struct would only \
              relocate the fields."
)]
fn block_rows_par(
    acc: &mut [i128],
    rows: usize,
    nb: usize,
    parts: usize,
    x: &[i64],
    w: Option<&[i64]>,
    o: usize,
    n: usize,
    jb: usize,
    j1: usize,
) -> Result<(), SyrkError> {
    if parts <= 1 {
        return block_rows(acc, 0, rows, x, w, o, n, jb, j1);
    }
    let base = rows / parts;
    let rem = rows % parts;
    let mut lo = 0usize;
    let mut rest: &mut [i128] = acc;
    let mut groups: Vec<(usize, usize, &mut [i128])> = Vec::with_capacity(parts);
    let mut t = 0usize;
    while t < parts {
        let hi = lo + base + usize::from(t < rem);
        let (head, tail) = rest.split_at_mut((hi - lo) * nb);
        groups.push((lo, hi, head));
        rest = tail;
        lo = hi;
        t += 1;
    }
    let err: Option<SyrkError> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(groups.len());
        for (lo, hi, part) in groups {
            handles.push(scope.spawn(move || block_rows(part, lo, hi, x, w, o, n, jb, j1)));
        }
        let mut first: Option<SyrkError> = None;
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

/// Weighted Q32.32 Gram (SYRK) fold: returns `G` row-major `[n, n]` with
/// `G[i,j] = round(Σ_o w[o]·x[o,i]·x[o,j] / 2⁶⁴)` in Q32.32, `x` row-major `[o, n]`.
///
/// Only the upper triangle is accumulated, in column blocks of [`BLOCK`]; the lower triangle is
/// mirrored from it, so `G` is exactly symmetric (`G[i,j] == G[j,i]` bit-for-bit) and each
/// upper-triangle pair is accumulated once. `threads` splits the output-block rows across
/// `std::thread::scope` workers; `threads ≤ 1` runs the same body with no spawn. Because each entry
/// is accumulated by exactly one worker in `o` order, the result is identical for every thread
/// count.
///
/// `w = None` is the unit weight (raw `2³²`). The accepted domain, the intermediate width, and the
/// single rounding are stated in the module documentation.
///
/// # Errors
/// [`SyrkError::ZeroN`] if `n == 0`; [`SyrkError::ZeroO`] if `o == 0`;
/// [`SyrkError::ShapeMismatch`] if `x.len() != o·n`, `w.len() != o`, or `o·n` leaves `usize`;
/// [`SyrkError::Overflow`] if `n·n` or the `o < 2¹⁸` domain bound is violated, or a checked `i128`
/// operation leaves its range; [`SyrkError::InputBound`] if an `x` or `w` entry reaches `2³⁶`.
///
/// # Complexity
/// Time: `o·n(n+1)/2` checked multiply-accumulates — the block loop partitions the columns into
/// `⌈n/BLOCK⌉` blocks, accumulates the pairs `(i,j)` with `i ≤ j < j1` once per block, and the blocks
/// partition the upper triangle, so each of the `n(n+1)/2` pairs is counted exactly once — plus an
/// `O(o·n)` bound scan and `O(n²)` round/store/mirror. The multiply-accumulates are split across
/// `threads`. Space: `O(n²)` for `G` and `O(n·BLOCK)` `i128` for the reused block accumulator.
pub fn wsyrk(
    x: &[i64],
    w: Option<&[i64]>,
    o: usize,
    n: usize,
    threads: usize,
) -> Result<Vec<i64>, SyrkError> {
    if n == 0 {
        return Err(SyrkError::ZeroN);
    }
    if o == 0 {
        return Err(SyrkError::ZeroO);
    }
    let Some(nn) = n.checked_mul(n) else {
        return Err(SyrkError::Overflow);
    };
    let Some(on) = o.checked_mul(n) else {
        return Err(SyrkError::ShapeMismatch);
    };
    if x.len() != on {
        return Err(SyrkError::ShapeMismatch);
    }
    if let Some(ws) = w {
        if ws.len() != o {
            return Err(SyrkError::ShapeMismatch);
        }
    }
    if o >= MAX_O {
        return Err(SyrkError::Overflow);
    }
    if !within_bound(x, w) {
        return Err(SyrkError::InputBound);
    }
    let nthreads = threads.max(1);
    let mut g = vec![0i64; nn];
    let mut acc = vec![0i128; n * BLOCK.min(n)];
    let mut jb = 0usize;
    while jb < n {
        let j1 = (jb + BLOCK).min(n);
        let nb = j1 - jb;
        let used = j1 * nb;
        acc[..used].fill(0);
        block_rows_par(&mut acc[..used], j1, nb, nthreads, x, w, o, n, jb, j1)?;
        let mut i = 0usize;
        while i < j1 {
            let jstart = if i > jb { i } else { jb };
            let row = i * nb;
            let mut j = jstart;
            while j < j1 {
                let Some(shifted) = round64(acc[row + (j - jb)]) else {
                    return Err(SyrkError::Overflow);
                };
                let Some(v) = narrow(shifted) else {
                    return Err(SyrkError::Overflow);
                };
                g[i * n + j] = v;
                if i != j {
                    g[j * n + i] = v;
                }
                j += 1;
            }
            i += 1;
        }
        jb = j1;
    }
    Ok(g)
}

/// Elementwise `a[k] = round(a[k]·h[k] / 2³²)` in Q32.32 — the `A ∘ hl` factor of `c_step`
/// (`A[i,j] = hl[i,j]·acc[i,j]`), one round-to-nearest per entry.
///
/// # Errors
/// [`SyrkError::ShapeMismatch`] if `a.len() != h.len()`; [`SyrkError::Overflow`] if a product, the
/// rounding add, or the narrowing leaves its range.
///
/// # Complexity
/// `O(a.len())` checked multiply-rounds and `O(1)` extra space; derivation: one loop, one product
/// and one round per entry.
pub fn hadamard_into(a: &mut [i64], h: &[i64]) -> Result<(), SyrkError> {
    if a.len() != h.len() {
        return Err(SyrkError::ShapeMismatch);
    }
    let n = a.len();
    let mut k = 0usize;
    while k < n {
        let Some(p) = i128::from(a[k]).checked_mul(i128::from(h[k])) else {
            return Err(SyrkError::Overflow);
        };
        let Some(r) = round32(p) else {
            return Err(SyrkError::Overflow);
        };
        let Some(v) = narrow(r) else {
            return Err(SyrkError::Overflow);
        };
        a[k] = v;
        k += 1;
    }
    Ok(())
}
