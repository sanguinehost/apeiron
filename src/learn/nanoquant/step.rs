//! The ADMM outer step of `factorize`: each side's step on its resident frame or the CPU, and the
//! outer iterations on the device-resident iterate (`crate::gpu::limb::commit`).

use super::{rank1, spd_inverse, x_update, NqConfig, NqError, Rng, LIMBS};

/// One side's resident frame (`crate::gpu::limb::Side`); uninhabited without the `gpu` feature.
#[cfg(feature = "gpu")]
type Side = crate::gpu::limb::Side;
#[cfg(not(feature = "gpu"))]
type Side = std::convert::Infallible;

/// The resident frame of the `k × n` target words `w` at rank `r` when the device takes the
/// side's products; `None` runs every step on the CPU.
///
/// # Complexity
///
/// As `crate::gpu::limb::Side::new`: `O(k·n)` bytes staged once per factorisation.
#[cfg_attr(not(feature = "gpu"), allow(clippy::unnecessary_wraps))]
pub(super) fn resident(w: &[i64], k: usize, n: usize, r: usize) -> Option<Side> {
    #[cfg(feature = "gpu")]
    if crate::gpu::limb::takes(k, r, n) {
        return Side::new(w, k, n, r, LIMBS);
    }
    let _ = (w, k, n, r);
    None
}

/// [`x_update`] on the side's resident frame when it holds one and the device completes the step,
/// the CPU otherwise; the two are equal in every bit (`gpu::limb::admm`).
///
/// # Complexity
///
/// As [`x_update`].
#[allow(
    clippy::too_many_arguments,
    reason = "the ADMM step's operands are distinct matrices"
)]
fn x_step(
    side: &mut Option<Side>,
    zx: &[f64],
    n: usize,
    r: usize,
    wp: &crate::engine::kern::limb::Prepared,
    k: usize,
    z: &[f64],
    du: &[f64],
    rho: f64,
    reg: f64,
) -> Result<Vec<f64>, NqError> {
    #[cfg(feature = "gpu")]
    if let Some(done) = side
        .as_mut()
        .and_then(|s| s.x_update(zx, z, du, rho, reg, spd_inverse))
    {
        return done;
    }
    #[cfg(not(feature = "gpu"))]
    let _ = side;
    x_update(zx, n, r, wp, k, z, du, rho, reg)
}

/// `len` Gaussian draws from `rng`, a [`rank1`] start. `O(len)` draws.
pub(super) fn draw(rng: &mut Rng, len: usize) -> Vec<f64> {
    (0..len).map(|_| rng.normal()).collect()
}

/// [`rank1`] from `start`, on the side's resident frame when it holds one and the device completes
/// it, the CPU otherwise; the two are equal in every bit (`gpu::limb::rank1`). [`rank1`]'s cost.
pub(super) fn rank1_step(
    side: &mut Option<Side>,
    m: &[f64],
    (rows, cols): (usize, usize),
    iters: usize,
    start_rows: bool,
    start: Vec<f64>,
) -> Result<Vec<f64>, NqError> {
    #[cfg(feature = "gpu")]
    if let Some(out) = side
        .as_mut()
        .and_then(|s| s.rank1(m, iters, start_rows, &start))
    {
        return Ok(out);
    }
    #[cfg(not(feature = "gpu"))]
    let _ = side;
    rank1(m, rows, cols, iters, start_rows, start)
}

/// One side's ADMM step on a `k × r` iterate: `X` by [`x_step`], `Z = rank1(X + U)` from `start`
/// by [`rank1_step`], and `U += X − Z` in place once both complete. Returns `X` and the `rank1`
/// result, so the caller orders the two sides' refusals as the serial loop does.
/// The frame `(zx, n, r, wp, k)` is [`x_update`]'s.
///
/// # Complexity
///
/// [`x_step`] plus [`rank1_step`] plus `2·k·r` adds.
#[allow(
    clippy::type_complexity,
    reason = "the two stages' results stay separate"
)]
pub(super) fn side_step(
    side: &mut Option<Side>,
    (zx, n, r, wp, k): (
        &[f64],
        usize,
        usize,
        &crate::engine::kern::limb::Prepared,
        usize,
    ),
    z: &[f64],
    du: &mut [f64],
    (rho, reg, inner): (f64, f64, usize),
    start_rows: bool,
    start: Vec<f64>,
) -> Result<(Vec<f64>, Result<Vec<f64>, NqError>), NqError> {
    let x = x_step(side, zx, n, r, wp, k, z, du, rho, reg)?;
    let t: Vec<f64> = x.iter().zip(du.iter()).map(|(x, u)| x + u).collect();
    let zn = rank1_step(side, &t, (k, r), inner, start_rows, start);
    if let Ok(zn) = &zn {
        for (u, (l, z)) in du.iter_mut().zip(x.iter().zip(zn)) {
            *u += l - z;
        }
    }
    Ok((x, zn))
}

/// The outer iterations on the device-resident iterate while both sides hold frames: per
/// iteration the starts of `A` (`lens[0]` long) and `B` (`lens[1]`) are drawn in the serial order,
/// both sides step on two threads ([`Side`]`::step`) and both commit. Returns the first iteration
/// not committed with its drawn starts, after reading `Z` and `U` of both sides back into `z` and
/// `u`; or `outer_iters`, after reading `X`, `Z` and `U` back. Every value read is the host loop's
/// in every bit (`gpu::limb::resident`), so the host loop resumes at the returned iteration.
/// `APEIRON_ADMM_RESIDENT=0` runs none.
///
/// # Complexity
///
/// Per iteration the device work of two [`side_step`]s and `O(r²)` words across the bus per side;
/// `O((o + n)·r)` words at the load and the readback.
#[allow(
    clippy::type_complexity,
    reason = "the three iterate matrices of the two sides"
)]
pub(super) fn device_iters(
    sides: [&mut Option<Side>; 2],
    [ls, z, u]: [[&mut Vec<f64>; 2]; 3],
    rng: &mut Rng,
    cfg: &NqConfig,
    lens: [usize; 2],
) -> (usize, Option<(Vec<f64>, Vec<f64>)>) {
    #[cfg(feature = "gpu")]
    {
        let [Some(a), Some(b)] = sides.map(Option::as_mut) else {
            return (0, None);
        };
        if cfg.outer_iters == 0
            || std::env::var("APEIRON_ADMM_RESIDENT").is_ok_and(|v| v.trim() == "0")
            || a.load(&z[1][..], &z[0][..], &u[0][..]).is_none()
            || b.load(&z[0][..], &z[1][..], &u[1][..]).is_none()
        {
            return (0, None);
        }
        let read = |s: &Side, i: usize| s.state(i).expect("a loaded frame reads its iterate back");
        for t in 0..cfg.outer_iters {
            let (sa, sb) = (draw(rng, lens[0]), draw(rng, lens[1]));
            let p = (cfg.schedule.rho(t as f64 / cfg.outer_iters as f64), cfg.reg);
            let it = cfg.inner_iters;
            let (ra, rb) = std::thread::scope(|s| {
                let hb = s.spawn(|| b.step(p, spd_inverse, it, true, &sb));
                (a.step(p, spd_inverse, it, false, &sa), hb.join())
            });
            let rb = rb.unwrap_or_else(|e| std::panic::resume_unwind(e));
            if ra.is_none() || rb.is_none() || crate::gpu::limb::commit(a, b).is_none() {
                for (i, s) in [&*a, &*b].into_iter().enumerate() {
                    *z[i] = read(s, 1);
                    *u[i] = read(s, 2);
                }
                return (t, Some((sa, sb)));
            }
        }
        for (i, s) in [&*a, &*b].into_iter().enumerate() {
            *ls[i] = read(s, 0);
            *z[i] = read(s, 1);
            *u[i] = read(s, 2);
        }
        (cfg.outer_iters, None)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (sides, ls, z, u, rng, cfg, lens);
        (0, None)
    }
}
