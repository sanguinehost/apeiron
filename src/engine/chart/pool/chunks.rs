//! Typed chunked and row-band dispatch over the process pool.
//!
//! Every entry runs its tasks through [`dispatch`], which runs `Pool::run` (the one copy of the
//! dispatch protocol) and returns the error of the lowest task index. A task writes only its own
//! chunk. The entries allocate nothing per dispatch except `for_vec_chunks` and `for_pbands`, which
//! allocate one descriptor per output (and `for_pbands` one column-pointer and one fill vector per
//! band).

use super::{FirstErr, Pool, SendPtr};
use crate::engine::chart::task::{BandTask, Task1, Task2, Task3, Task4, VecTask};

/// Runs `f(0) … f(tasks − 1)`, each once, and returns the error of the lowest failing task.
///
/// Every task runs even after a failure. Cost: `tasks` calls; the error slot is locked only on `Err`.
fn dispatch<E: Send, F: Fn(usize) -> Result<(), E> + Sync>(tasks: usize, f: &F) -> Result<(), E> {
    let errs: FirstErr<usize, E> = FirstErr::new();
    Pool::global().run(tasks, &|i| {
        if let Err(e) = f(i) {
            errs.record(i, e);
        }
    });
    match errs.take() {
        Some((_, e)) => Err(e),
        None => Ok(()),
    }
}

/// Chunk `i` of a `len`-element output split into `chunk`-element pieces (the last may be shorter).
///
/// # Safety
///
/// `base` addresses `len` initialised elements that stay mutably borrowed for the whole dispatch,
/// and each task index is passed at most once per `base`, so the returned slices are disjoint.
unsafe fn chunk_part<'a, A>(base: SendPtr<A>, len: usize, chunk: usize, i: usize) -> &'a mut [A] {
    let start = i * chunk;
    let end = (start + chunk).min(len);
    std::slice::from_raw_parts_mut(base.get().add(start), end - start)
}

#[allow(
    dead_code,
    reason = "consumed by the FORM-PARALLEL L1-L4 lanes; exercised by this module's tests"
)]
/// Splits `out` into `chunk`-element pieces and runs `k.run(task, piece)` on each, `task` being the
/// piece index. Equal to the sequential loop over the pieces, with the lowest-index `Err` returned.
///
/// # Complexity
///
/// `⌈out.len() / chunk⌉` calls; `O(1)` extra space.
pub(crate) fn for_chunks<A: Send, E: Send, K: Task1<A, E>>(
    out: &mut [A],
    chunk: usize,
    k: &K,
) -> Result<(), E> {
    let chunk = chunk.max(1);
    let (len, base) = (out.len(), SendPtr(out.as_mut_ptr()));
    dispatch(len.div_ceil(chunk), &|i| {
        // SAFETY: `out` is borrowed mutably for the dispatch; each task index runs once.
        let a = unsafe { chunk_part(base, len, chunk, i) };
        k.run(i, a)
    })
}

#[allow(
    dead_code,
    reason = "consumed by the FORM-PARALLEL L1-L4 lanes; exercised by this module's tests"
)]
/// [`for_chunks`] over two outputs split with their own chunk sizes. The task counts must agree.
pub(crate) fn for_chunks2<A: Send, B: Send, E: Send, K: Task2<A, B, E>>(
    oa: &mut [A],
    ka: usize,
    ob: &mut [B],
    kb: usize,
    k: &K,
) -> Result<(), E> {
    let (ka, kb) = (ka.max(1), kb.max(1));
    let tasks = oa.len().div_ceil(ka);
    assert_eq!(
        tasks,
        ob.len().div_ceil(kb),
        "for_chunks2: task counts differ"
    );
    let (na, nb) = (oa.len(), ob.len());
    let (pa, pb) = (SendPtr(oa.as_mut_ptr()), SendPtr(ob.as_mut_ptr()));
    dispatch(tasks, &|i| {
        // SAFETY: as in `for_chunks`, for each output; the two outputs are distinct borrows.
        let (xa, xb) = unsafe { (chunk_part(pa, na, ka, i), chunk_part(pb, nb, kb, i)) };
        k.run(i, xa, xb)
    })
}

/// [`for_chunks`] over three outputs. The task counts must agree.
pub(crate) fn for_chunks3<A: Send, B: Send, C: Send, E: Send, K: Task3<A, B, C, E>>(
    oa: &mut [A],
    ka: usize,
    ob: &mut [B],
    kb: usize,
    oc: &mut [C],
    kc: usize,
    k: &K,
) -> Result<(), E> {
    let (ka, kb, kc) = (ka.max(1), kb.max(1), kc.max(1));
    let tasks = oa.len().div_ceil(ka);
    assert_eq!(
        tasks,
        ob.len().div_ceil(kb),
        "for_chunks3: task counts differ"
    );
    assert_eq!(
        tasks,
        oc.len().div_ceil(kc),
        "for_chunks3: task counts differ"
    );
    let (na, nb, nc) = (oa.len(), ob.len(), oc.len());
    let (pa, pb, pc) = (
        SendPtr(oa.as_mut_ptr()),
        SendPtr(ob.as_mut_ptr()),
        SendPtr(oc.as_mut_ptr()),
    );
    dispatch(tasks, &|i| {
        // SAFETY: as in `for_chunks`, for each output; the three outputs are distinct borrows.
        let (xa, xb, xc) = unsafe {
            (
                chunk_part(pa, na, ka, i),
                chunk_part(pb, nb, kb, i),
                chunk_part(pc, nc, kc, i),
            )
        };
        k.run(i, xa, xb, xc)
    })
}

/// [`for_chunks`] over four outputs. The task counts must agree.
#[allow(
    dead_code,
    reason = "consumed by the FORM-PARALLEL L1-L4 lanes; exercised by this module's tests"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "four outputs, each with its chunk size, and the task"
)]
pub(crate) fn for_chunks4<A: Send, B: Send, C: Send, D: Send, E: Send, K: Task4<A, B, C, D, E>>(
    oa: &mut [A],
    ka: usize,
    ob: &mut [B],
    kb: usize,
    oc: &mut [C],
    kc: usize,
    od: &mut [D],
    kd: usize,
    k: &K,
) -> Result<(), E> {
    let (ka, kb, kc, kd) = (ka.max(1), kb.max(1), kc.max(1), kd.max(1));
    let tasks = oa.len().div_ceil(ka);
    assert_eq!(
        tasks,
        ob.len().div_ceil(kb),
        "for_chunks4: task counts differ"
    );
    assert_eq!(
        tasks,
        oc.len().div_ceil(kc),
        "for_chunks4: task counts differ"
    );
    assert_eq!(
        tasks,
        od.len().div_ceil(kd),
        "for_chunks4: task counts differ"
    );
    let (na, nb, nc, nd) = (oa.len(), ob.len(), oc.len(), od.len());
    let (pa, pb, pc, pd) = (
        SendPtr(oa.as_mut_ptr()),
        SendPtr(ob.as_mut_ptr()),
        SendPtr(oc.as_mut_ptr()),
        SendPtr(od.as_mut_ptr()),
    );
    dispatch(tasks, &|i| {
        // SAFETY: as in `for_chunks`, for each output; the four outputs are distinct borrows.
        let (xa, xb, xc, xd) = unsafe {
            (
                chunk_part(pa, na, ka, i),
                chunk_part(pb, nb, kb, i),
                chunk_part(pc, nc, kc, i),
                chunk_part(pd, nd, kd, i),
            )
        };
        k.run(i, xa, xb, xc, xd)
    })
}

/// Splits several `Vec` outputs, output `i` into `chunk[i]`-element pieces, and runs
/// `k.run(task, output, offset, piece)` once per piece. Tasks are numbered across the outputs in order,
/// and no piece crosses a `Vec`.
///
/// # Complexity
///
/// `Σ_i ⌈outs[i].len() / chunk[i]⌉` calls; `O(outs.len())` extra space for the task table.
#[allow(
    dead_code,
    reason = "consumed by the FORM-PARALLEL L1-L4 lanes; exercised by this module's tests"
)]
pub(crate) fn for_vec_chunks<T: Send, E: Send, K: VecTask<T, E>>(
    outs: &mut [Vec<T>],
    chunk: &[usize],
    k: &K,
) -> Result<(), E> {
    assert_eq!(
        outs.len(),
        chunk.len(),
        "for_vec_chunks: one chunk size per output"
    );
    // (base, length, chunk size, first task index) per output.
    let mut table: Vec<(SendPtr<T>, usize, usize, usize)> = Vec::with_capacity(outs.len());
    let mut tasks = 0;
    for (v, &c) in outs.iter_mut().zip(chunk) {
        let c = c.max(1);
        table.push((SendPtr(v.as_mut_ptr()), v.len(), c, tasks));
        tasks += v.len().div_ceil(c);
    }
    dispatch(tasks, &|t| {
        // The last output whose first task is at or below `t` owns `t`; it has at least one task.
        let i = table.partition_point(|e| e.3 <= t) - 1;
        let (base, len, c, first) = table[i];
        let j = t - first;
        // SAFETY: task `t` is piece `j` of output `i`; pieces partition each `Vec`, and the `Vec`s are
        // borrowed mutably for the dispatch.
        let piece = unsafe { chunk_part(base, len, c, j) };
        k.run(t, i, j * c, piece)
    })
}

#[allow(
    dead_code,
    reason = "consumed by the FORM-PARALLEL L1-L4 lanes; exercised by this module's tests"
)]
/// A row band of [`for_pbands`]: `rows()` slots per column, filled with `push` in row order.
pub(crate) struct PBand<T> {
    cols: Vec<SendPtr<T>>,
    fill: Vec<usize>,
    rows: usize,
}

#[allow(
    dead_code,
    reason = "consumed by the FORM-PARALLEL L1-L4 lanes; exercised by this module's tests"
)]
impl<T> PBand<T> {
    /// The number of rows this band must fill in every column.
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Writes the next row of column `c`. Panics when column `c` already holds `rows()` values.
    pub(crate) fn push(&mut self, c: usize, v: T) {
        assert!(
            self.fill[c] < self.rows,
            "PBand::push: column {c} past its band"
        );
        // SAFETY: slot `fill[c]` of column `c` lies in this band's rows of that column's reserved
        // capacity, and is written once; no other band addresses it.
        unsafe { self.cols[c].get().add(self.fill[c]).write(v) };
        self.fill[c] += 1;
    }
}

#[allow(
    dead_code,
    reason = "consumed by the FORM-PARALLEL L1-L4 lanes; exercised by this module's tests"
)]
/// Fills `n` rows of every column of `cols` through row bands of `rows_per` rows. Band `t` covers
/// rows `[t·rows_per, min((t+1)·rows_per, n))` and is passed to `k.run(t, first_row, band)`.
///
/// Each column must be empty on entry; its spare capacity receives the rows, and the columns are
/// lengthened to `n` only when every task succeeds. A band that returns `Ok` without filling every
/// column panics after the dispatch.
///
/// # Complexity
///
/// `⌈n / rows_per⌉` calls; one allocation per band for its column pointers and fill counts.
pub(crate) fn for_pbands<T: Send, E: Send, K: BandTask<T, E>>(
    cols: &mut [Vec<T>],
    n: usize,
    rows_per: usize,
    k: &K,
) -> Result<(), E> {
    let rows_per = rows_per.max(1);
    for col in cols.iter_mut() {
        assert!(col.is_empty(), "for_pbands: column must be empty on entry");
        col.reserve(n);
    }
    let bases: Vec<SendPtr<T>> = cols.iter_mut().map(|c| SendPtr(c.as_mut_ptr())).collect();
    dispatch(n.div_ceil(rows_per), &|t| {
        let r0 = t * rows_per;
        let rows = (r0 + rows_per).min(n) - r0;
        let mut band = PBand {
            cols: bases
                .iter()
                .map(|p| SendPtr(p.get().wrapping_add(r0)))
                .collect(),
            fill: vec![0; bases.len()],
            rows,
        };
        k.run(t, r0, &mut band)?;
        assert!(
            band.fill.iter().all(|&k| k == rows),
            "for_pbands: band {t} not full"
        );
        Ok(())
    })?;
    for col in cols.iter_mut() {
        // SAFETY: the bands cover `0..n` and every band returned `Ok` with all its slots written, so
        // the first `n` slots of each column are initialised; `reserve(n)` made the capacity.
        unsafe { col.set_len(n) };
    }
    Ok(())
}

/// Make the first `n` words of `col`'s spare capacity its contents.
///
/// # Safety
///
/// `n` is within `col.capacity()` and words `len..n` were initialised through the column's spare
/// capacity (the pool boundary's disjoint writes); the elements need no drop.
pub(crate) unsafe fn set_written_len<T>(col: &mut Vec<T>, n: usize) {
    // SAFETY: the caller guarantees the extent and the initialisation.
    unsafe { col.set_len(n) };
}

/// [`set_written_len`] on every column of `cols`.
///
/// # Safety
///
/// [`set_written_len`]'s contract holds for each column.
pub(crate) unsafe fn set_written_lens<T>(cols: &mut [Vec<T>], n: usize) {
    for col in cols {
        // SAFETY: the caller guarantees the extent and the initialisation of every column.
        unsafe { col.set_len(n) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Closure adapters: the production entries take Task traits; a test passes the body as a closure.
    struct Fk<F>(F);
    impl<A, E, F: Fn(usize, &mut [A]) -> Result<(), E> + Sync> Task1<A, E> for Fk<F> {
        fn run(&self, t: usize, a: &mut [A]) -> Result<(), E> {
            (self.0)(t, a)
        }
    }
    impl<A, B, E, F: Fn(usize, &mut [A], &mut [B]) -> Result<(), E> + Sync> Task2<A, B, E> for Fk<F> {
        fn run(&self, t: usize, a: &mut [A], b: &mut [B]) -> Result<(), E> {
            (self.0)(t, a, b)
        }
    }
    impl<T, E, F: Fn(usize, usize, usize, &mut [T]) -> Result<(), E> + Sync> VecTask<T, E> for Fk<F> {
        fn run(&self, t: usize, o: usize, lo: usize, p: &mut [T]) -> Result<(), E> {
            (self.0)(t, o, lo, p)
        }
    }
    impl<T, E, F: Fn(usize, usize, &mut PBand<T>) -> Result<(), E> + Sync> BandTask<T, E> for Fk<F> {
        fn run(&self, t: usize, r0: usize, b: &mut PBand<T>) -> Result<(), E> {
            (self.0)(t, r0, b)
        }
    }
    fn for_chunks<A: Send, E: Send>(
        out: &mut [A],
        chunk: usize,
        f: impl Fn(usize, &mut [A]) -> Result<(), E> + Sync,
    ) -> Result<(), E> {
        super::for_chunks(out, chunk, &Fk(f))
    }
    fn for_chunks2<A: Send, B: Send, E: Send>(
        oa: &mut [A],
        ka: usize,
        ob: &mut [B],
        kb: usize,
        f: impl Fn(usize, &mut [A], &mut [B]) -> Result<(), E> + Sync,
    ) -> Result<(), E> {
        super::for_chunks2(oa, ka, ob, kb, &Fk(f))
    }
    fn for_vec_chunks<T: Send, E: Send>(
        outs: &mut [Vec<T>],
        chunk: &[usize],
        f: impl Fn(usize, usize, usize, &mut [T]) -> Result<(), E> + Sync,
    ) -> Result<(), E> {
        super::for_vec_chunks(outs, chunk, &Fk(f))
    }
    fn for_pbands<T: Send, E: Send>(
        cols: &mut [Vec<T>],
        n: usize,
        rows_per: usize,
        f: impl Fn(usize, usize, &mut PBand<T>) -> Result<(), E> + Sync,
    ) -> Result<(), E> {
        super::for_pbands(cols, n, rows_per, &Fk(f))
    }

    fn xorshift(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn for_chunks_equals_seq() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for len in [0usize, 1, 5, 64, 1000] {
            for chunk in [1usize, 3, 7, 100, 2000] {
                for _ in 0..8 {
                    let tasks = len.div_ceil(chunk);
                    let fail: Vec<bool> = (0..tasks)
                        .map(|_| xorshift(&mut seed).is_multiple_of(4))
                        .collect();
                    let want: Result<(), usize> = match fail.iter().position(|&b| b) {
                        Some(t) => Err(t),
                        None => Ok(()),
                    };
                    let runs: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
                    let mut out = vec![0u64; len];
                    let got: Result<(), usize> =
                        for_chunks(&mut out, chunk, |t: usize, dst: &mut [u64]| {
                            runs[t].fetch_add(1, Ordering::Relaxed);
                            for (k, x) in dst.iter_mut().enumerate() {
                                *x = (t * 1000 + k) as u64 + 1;
                            }
                            if fail[t] {
                                Err(t)
                            } else {
                                Ok(())
                            }
                        });
                    assert_eq!(got, want, "len {len} chunk {chunk}");
                    let mut expect = vec![0u64; len];
                    for t in 0..tasks {
                        let lo = t * chunk;
                        for k in 0..(lo + chunk).min(len) - lo {
                            expect[lo + k] = (t * 1000 + k) as u64 + 1;
                        }
                    }
                    assert_eq!(out, expect, "len {len} chunk {chunk}");
                    assert!(runs.iter().all(|r| r.load(Ordering::Relaxed) == 1));
                }
            }
        }
    }

    #[test]
    fn for_chunks_two_outputs_disjoint() {
        let mut a = vec![0usize; 100];
        let mut b = vec![0usize; 100];
        let r: Result<(), ()> = for_chunks2(
            &mut a,
            7,
            &mut b,
            7,
            |t: usize, x: &mut [usize], y: &mut [usize]| {
                for v in x.iter_mut() {
                    *v = t + 1;
                }
                for v in y.iter_mut() {
                    *v = 200 - t;
                }
                Ok(())
            },
        );
        assert_eq!(r, Ok(()));
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(*x, k / 7 + 1);
            assert_eq!(*y, 200 - k / 7);
        }
    }

    #[test]
    fn for_pbands_fills_every_row() {
        let (n, cols_n) = (50usize, 3usize);
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); cols_n];
        let r: Result<(), ()> = for_pbands(&mut cols, n, 8, |_t, r0, band: &mut PBand<usize>| {
            for r in 0..band.rows() {
                for c in 0..cols_n {
                    band.push(c, (r0 + r) * 10 + c);
                }
            }
            Ok(())
        });
        assert_eq!(r, Ok(()));
        for (c, col) in cols.iter().enumerate() {
            assert_eq!(col.len(), n);
            for (row, v) in col.iter().enumerate() {
                assert_eq!(*v, row * 10 + c);
            }
        }
    }

    #[test]
    fn for_vec_chunks_splits_within_each_vec() {
        let mut outs: Vec<Vec<usize>> = vec![vec![0; 10], vec![0; 4], vec![0; 0]];
        let chunk = [3usize, 4, 2];
        let r: Result<(), ()> = for_vec_chunks(&mut outs, &chunk, |t, i, lo, dst: &mut [usize]| {
            for (k, x) in dst.iter_mut().enumerate() {
                *x = t * 100 + i * 10 + lo + k;
            }
            Ok(())
        });
        assert_eq!(r, Ok(()));
        assert_eq!(outs[0].len(), 10);
        assert_eq!(outs[1].len(), 4);
        // Output 0 holds tasks 0..4 (⌈10/3⌉), output 1 task 4, output 2 none.
        assert_eq!(outs[0][9], 3 * 100 + 9);
        assert_eq!(outs[1][0], 4 * 100 + 10);
    }

    #[test]
    #[should_panic(expected = "a Q8Act pool task panicked")]
    fn pband_underfill_panics() {
        let mut cols: Vec<Vec<u32>> = vec![Vec::new(); 2];
        let _ = for_pbands(&mut cols, 10, 4, |_t, _r0, band: &mut PBand<u32>| {
            band.push(0, 1);
            Ok::<(), ()>(())
        });
    }
}
