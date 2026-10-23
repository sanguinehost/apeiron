//! The ADMM iterate `Z_x`, `Z`, `U`, `X` resident on the device between outer iterations, bit for
//! bit.
//!
//! **Contract.** After [`Side::load`] on both sides, one outer iteration is [`Side::step`] on each
//! side and then [`commit`]: the step runs `X = x_update(Z_x, Z, U)` and `rank1(X + U)` on the
//! frame's buffers, and the commit applies `U += X − Z'`, `Z ← Z'` and the peer's `Z_x ← Z'`, where
//! `Z'` is the `rank1` result. Each value equals `learn::nanoquant::side_step`'s in every bit: the
//! products are those of `gpu::limb::admm` and `gpu::limb::rank1`, and the two maps
//! (`rank1_map.comp` modes 6 and 7) are the CPU's `x + u` and `u + (x − z)` in binary64 with no
//! contraction. A `None` from either side's step precedes every commit write, so `Z` and `U` are
//! still those of the last committed iteration and [`Side::state`] reads them back for the host.
//!
//! **Route.** The step stages `[rho, reg]` and the host inverse's `H⁻¹` and reads back `hw` and the
//! flag words; the iterate itself crosses the bus only at [`Side::load`] and [`Side::state`].

use std::sync::PoisonError;

use super::admm::Side;
use super::device;
use super::rank1::MAX_ITERS;
use crate::engine::kern::limb::MAX_INNER;

impl Side {
    /// Uploads the iterate `Z_x` (`n × r`), `Z` and `U` (`k × r`); `None` on a length mismatch or no
    /// frame.
    ///
    /// # Complexity
    ///
    /// `O((n + 2k)·r)` words staged; one submit.
    pub(crate) fn load(&self, zx: &[f64], z: &[f64], du: &[f64]) -> Option<()> {
        self.frame.as_deref()?.load(zx, z, du)
    }

    /// One side's step on the resident iterate: `X` by `x_update` with `inv` the host inverse of
    /// `hw` (`spd_inverse`), then `rank1(X + U)` from `start` (`k` long when `start_rows`, else
    /// `r`), its result left on the device for [`commit`]. `None` on a device flag, a refusal of
    /// `inv`, or an operand outside the frame; the iterate is then unchanged.
    ///
    /// # Complexity
    ///
    /// Host: `O(r²)` words staged and read, plus `inv`. Device: `x_update`'s three limb products
    /// and `rank1`'s `2·iters + 1` matvecs; three submits.
    pub(crate) fn step<E>(
        &mut self,
        (rho, reg): (f64, f64),
        inv: impl FnOnce(&[i64], usize) -> Result<Vec<i64>, E>,
        iters: usize,
        start_rows: bool,
        start: &[f64],
    ) -> Option<()> {
        let f = self.frame.as_deref_mut()?;
        let ls = if start_rows { f.k } else { f.r };
        if start.len() != ls || f.k > MAX_INNER || iters > MAX_ITERS {
            return None;
        }
        f.x_submit(1, rho, reg, inv)?.ok()?;
        // SAFETY: `start`'s length and `iters` are checked above; the resident form uploads no `m`.
        unsafe { f.rank1_submit(iters, start_rows, None, start) }?;
        Some(())
    }

    /// `[X, Z, U][i]` of the resident iterate, `k × r`; `None` with no frame.
    ///
    /// # Complexity
    ///
    /// One submit; `O(k·r)` words read.
    pub(crate) fn state(&self, i: usize) -> Option<Vec<f64>> {
        self.frame.as_deref()?.read(i)
    }
}

/// Commits one outer iteration of both sides after each [`Side::step`] returned `Some`: per side
/// `U += X − Z'`, `Z ← Z'`, and `Z'` into the peer's `Z_x`. `None` before any write when a side
/// holds no frame or `rank1`, or the two frames are not transposes (`k_a = n_b`, `n_a = k_b`,
/// `r_a = r_b`).
///
/// # Complexity
///
/// `O((k_a + k_b)·r)` device words mapped and copied; two submits, a recording on the first commit
/// per pair.
pub(crate) fn commit(a: &mut Side, b: &mut Side) -> Option<()> {
    let (fa, fb) = (a.frame.as_deref_mut()?, b.frame.as_deref_mut()?);
    if fa.k != fb.n || fa.n != fb.k || fa.r != fb.r || fa.r1.is_none() || fb.r1.is_none() {
        return None;
    }
    let (za, zb) = (fa.iterate[0], fb.iterate[0]);
    let dev = device()?;
    let guard = dev.lock().unwrap_or_else(PoisonError::into_inner);
    let vk = &guard.vk;
    // SAFETY: the device mutex is held; each peer `Z_x` is `n × r` = the other side's `k × r`.
    unsafe {
        let ca = fa.r1.as_deref_mut()?.commit(vk, zb);
        let cb = fb.r1.as_deref_mut()?.commit(vk, za);
        vk.submit_wait(ca);
        vk.submit_wait(cb);
    }
    Some(())
}
