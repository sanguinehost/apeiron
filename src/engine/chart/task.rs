//! Task traits for the process pool's dispatch entries (`chart::pool`).
//!
//! A pool entry takes `&K` with `K` one of these traits instead of a closure: Aeneas translates a
//! user trait and its impl from one translator, so the extracted call and the Lean instance agree,
//! while a std `Fn*` impl generated for a closure does not match the hand-written model trait
//! (FORM-PARALLEL addendum D5). A call site defines a small context struct holding the shared
//! borrows and implements the trait by calling the named free function that is the task body.
//!
//! Every trait is `Sync`: the pool runs `run` from several threads on one `&self`.

use super::pool::PBand;

/// One output split into pieces: `run(t, piece)` for piece `t`.
pub(crate) trait Task1<A, E>: Sync {
    fn run(&self, t: usize, a: &mut [A]) -> Result<(), E>;
}

/// Two outputs split with their own piece sizes: `run(t, piece_a, piece_b)`.
pub(crate) trait Task2<A, B, E>: Sync {
    fn run(&self, t: usize, a: &mut [A], b: &mut [B]) -> Result<(), E>;
}

/// Three outputs split with their own piece sizes.
pub(crate) trait Task3<A, B, C, E>: Sync {
    fn run(&self, t: usize, a: &mut [A], b: &mut [B], c: &mut [C]) -> Result<(), E>;
}

/// Four outputs split with their own piece sizes.
pub(crate) trait Task4<A, B, C, D, E>: Sync {
    fn run(&self, t: usize, a: &mut [A], b: &mut [B], c: &mut [C], d: &mut [D]) -> Result<(), E>;
}

/// Several `Vec` outputs: `run(t, output, offset, piece)` for piece `t`, `offset` its start in `output`.
pub(crate) trait VecTask<T, E>: Sync {
    fn run(&self, t: usize, out: usize, offset: usize, piece: &mut [T]) -> Result<(), E>;
}

/// Row bands of several columns: `run(t, first_row, band)` fills `band.rows()` rows of every column.
pub(crate) trait BandTask<T, E>: Sync {
    fn run(&self, t: usize, first_row: usize, band: &mut PBand<T>) -> Result<(), E>;
}

/// A task of index `t` that writes through its own captured disjoint state and cannot fail.
pub(crate) trait IndexTask: Sync {
    fn run(&self, t: usize);
}

/// One output split into pieces, infallible: `run(c, start, piece)`, `start` the piece's offset.
pub(crate) trait ChunkTask<T>: Sync {
    fn run(&self, c: usize, start: usize, piece: &mut [T]);
}

/// One head of a per-head dispatch: `run(h, dst)` fills head `h`'s piece `dst`, refusing with `E`.
pub(crate) trait HeadTask<T, E>: Sync {
    fn run(&self, h: usize, dst: &mut [T]) -> Result<(), E>;
}

/// Test adapter: a closure as a [`HeadTask`]. Production call sites implement the trait on a context.
#[cfg(test)]
pub(crate) struct FnHead<F>(pub F);

#[cfg(test)]
impl<T, E, F: Fn(usize, &mut [T]) -> Result<(), E> + Sync> HeadTask<T, E> for FnHead<F> {
    fn run(&self, h: usize, dst: &mut [T]) -> Result<(), E> {
        (self.0)(h, dst)
    }
}

/// The lexicographic `(a, b)` order as one `u128`, for [`super::pool::FirstErr`] keys: a std tuple
/// `Ord` instance does not extract, a scalar one does.
pub(crate) fn pair_key(a: usize, b: usize) -> u128 {
    ((a as u128) << 64) | (b as u128)
}
