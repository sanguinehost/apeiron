//! The carrier point readout over any ring: density and phase flux.
//!
//! This is the Rust kernel for `Apeiron.Physics.CarrierAnyRing`. A phase cell is the algebra's
//! product `(a, 0) * u`, where the amplitude `a` stretches the direction `u`. Its size --
//! [`Cell::norm_form`] -- is the point density, and the density carried by the phase's forward
//! difference `next - previous` is the point flux.
//!
//! Both functions are generic over the crate's [`Ring`] contract and take the dial as an
//! argument, so the real and machine-word readouts are instantiations rather than restatements:
//! `Z/2^64` is a ring like any other. The unit direction does not consult the dial
//! (`CarrierAnyRing.pointDensity_ringOne_independent`), which is why a machine-word kernel can fix
//! it and still have the value be the one every face gives.
//!
//! The product is written out as the algebra's own `mul` rather than the slot-by-slot stretch
//! `a * u`: the two agree (`CarrierAnyRing.ringMul_mk_zero_eq_smul`) but only propositionally, and
//! the product is the construction the algebra's multiplication laws are stated over.
//!
//! # The exact evolution step
//!
//! [`conservative_density_step`] and [`diffusive_flux_half`] are a **nondimensional exact
//! integer kernel** for one carrier update. They move whole units between a cell and its faces
//! with checked arithmetic: the conservative step is `density - (outgoing - incoming) + source`,
//! and a face's diffusive half-flux is `-(right - left) / 2` only when the difference is exactly
//! even. An operation that cannot be represented returns `None` rather than wrapping or rounding.
//!
//! [`diffusive_line3_step`] composes them into the smallest fixed-size synchronous update: a
//! closed three-cell line with `nu = 1/2`, the two interior faces read once each and the two
//! endpoint faces fixed at zero. It is still not a material model and still partial: an odd
//! interior difference or an unrepresentable intermediate refuses the whole three-cell update.
//!
//! This is deliberately not a material model. There is no coefficient, no time step, no unit and
//! no calibrated viscosity; the caller supplies the face fluxes and sources and decides what a
//! step means. It is not a water solver, and a conserved total is a statement about the
//! arithmetic the caller feeds it, not about any particular physical regime.

use crate::cell::{Cell, Sigma};
use crate::scalar::Ring;

/// The density a phase cell carries, over any ring.
///
/// Temporal and spatial O(1): no loop or recursion; one `Cell::mul` and one `Cell::norm_form`, each
/// a fixed count of scalar `mul`/`add`/`sub` on the two slots. No allocation. The scalar operations
/// are the only cost, so the constant is not visible from the signature.
///
/// `CarrierAnyRing.pointDensity`: the norm form of `(a, 0) * u`, the cell the amplitude `a`
/// stretches the direction `u` into. The dial is an argument because the product and the norm form
/// both consult it; at the unit direction the answer does not depend on it
#[must_use]
pub fn point_density<R: Ring>(sigma: Sigma, amplitude: R, direction: Cell<R>) -> R {
    Cell::new(amplitude, R::ZERO)
        .mul(sigma, direction)
        .norm_form(sigma)
}

/// The phase flux a cell carries across one forward step, over any ring.
///
/// Temporal and spatial O(1): no loop or recursion; one [`point_density`], then one scalar `sub`
/// and one scalar `mul`. No allocation. The scalar operations are the only cost, so the constant is
/// not visible from the signature.
///
/// `CarrierAnyRing.pointFlux`: [`point_density`] at the amplitude and direction, times the forward
/// phase difference `next - previous`. The single phase fixes the flux's direction; no independent
/// momentum is supplied.
#[must_use]
pub fn point_flux<R: Ring>(
    sigma: Sigma,
    amplitude: R,
    direction: Cell<R>,
    previous: R,
    next: R,
) -> R {
    point_density(sigma, amplitude, direction).mul(next.sub(previous))
}

/// One conservative density update, exactly and with checked intermediates.
///
/// The update is `density - (outgoing - incoming) + source`: the cell loses its outgoing face
/// flux, gains its incoming face flux and gains whatever the source contributes. Every
/// intermediate is checked, so an unrepresentable result is reported as `None` rather than
/// wrapped. `None` is a refusal of the whole update, not a partial one.
///
/// This is the **nondimensional exact kernel** described in the module documentation. `incoming`,
/// `outgoing` and `source` are whole units the caller has already decided on; there is no
/// coefficient, time step or unit here, and this is not a calibrated viscosity or a water solver.
#[must_use]
#[expect(
    clippy::question_mark,
    clippy::manual_let_else,
    reason = "`?` desugars through `Try::branch`, which the Aeneas extraction has no model for; \
              the explicit `match` with early `return None` is the form it can read"
)]
pub fn conservative_density_step(
    density: i64,
    incoming: i64,
    outgoing: i64,
    source: i64,
) -> Option<i64> {
    // Explicit `match` rather than `?`: the `?` operator desugars through `Try::branch`, which the
    // extraction has no model for, so an early `return None` is the form it can read.
    let net = match outgoing.checked_sub(incoming) {
        Some(net) => net,
        None => return None,
    };
    let after_net = match density.checked_sub(net) {
        Some(after_net) => after_net,
        None => return None,
    };
    after_net.checked_add(source)
}

/// Half of the diffusive flux across a face, or `None` when it does not exist.
///
/// The result is the signed left-to-right flux `-(right - left) / 2`: negative when the right
/// neighbour is denser, so the direction is down the gradient. A half-unit flux exists only when
/// `right - left` is exactly even; an odd difference is refused rather than rounded, and an
/// unrepresentable difference is refused rather than wrapped. Every intermediate is checked.
///
/// This is the same **nondimensional exact kernel** as [`conservative_density_step`]. The factor
/// of one half is the two-cell face share, not a fitted diffusivity, and no time step is implied.
#[must_use]
#[expect(
    clippy::question_mark,
    clippy::manual_let_else,
    reason = "`?` desugars through `Try::branch`, which the Aeneas extraction has no model for; \
              the explicit `match` with early `return None` is the form it can read"
)]
pub fn diffusive_flux_half(left: i64, right: i64) -> Option<i64> {
    // Explicit `match` rather than `?` for the same reason as `conservative_density_step`.
    let difference = match right.checked_sub(left) {
        Some(difference) => difference,
        None => return None,
    };
    let remainder = match difference.checked_rem(2) {
        Some(remainder) => remainder,
        None => return None,
    };
    if remainder != 0 {
        return None;
    }
    // `-(right - left) / 2` as `0 - ((right - left) / 2)`. The subtraction cannot overflow here:
    // the halved difference has magnitude at most `2^62`.
    let half = match difference.checked_div(2) {
        Some(half) => half,
        None => return None,
    };
    0i64.checked_sub(half)
}

/// Three adjacent density samples: the smallest closed line a diffusive step updates.
///
/// The field names are positions, not roles; [`diffusive_line3_step`] reads `left`/`center` and
/// `center`/`right` as the two interior faces and treats the two ends as closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensityLine3 {
    /// The left cell.
    pub left: i64,
    /// The center cell.
    pub center: i64,
    /// The right cell.
    pub right: i64,
}

/// One synchronous exact diffusive update of a closed three-cell line, `nu = 1/2`.
///
/// The two interior faces are derived **once each** from the old state with
/// [`diffusive_flux_half`] -- `-(center - left) / 2` and `-(right - center) / 2` -- and then
/// shared: the left cell's outgoing face is the center cell's incoming face, and the center
/// cell's outgoing face is the right cell's incoming face. The two endpoint faces are closed and
/// carry zero. Each of the three cells is updated by [`conservative_density_step`] with its own
/// source, so the update is `old - (outgoing - incoming) + source` per cell.
///
/// Every local is computed before the result is constructed, so a refusal at any face or cell
/// returns `None` and no partially updated line. The operation is **partial** on purpose: an odd
/// interior difference has no exact half, and an unrepresentable intermediate is refused rather
/// than wrapped. There is no coefficient, time step, unit, float or rounding here.
#[must_use]
#[expect(
    clippy::question_mark,
    clippy::manual_let_else,
    reason = "`?` desugars through `Try::branch`, which the Aeneas extraction has no model for; \
              the explicit `match` with early `return None` is the form it can read"
)]
pub fn diffusive_line3_step(old: DensityLine3, source: DensityLine3) -> Option<DensityLine3> {
    // Explicit `match` rather than `?` for the same reason as the two kernels above. Every local
    // is bound before `Some`, so a late refusal discards the earlier results rather than
    // returning a partial line.
    let face_left = match diffusive_flux_half(old.left, old.center) {
        Some(face_left) => face_left,
        None => return None,
    };
    let face_right = match diffusive_flux_half(old.center, old.right) {
        Some(face_right) => face_right,
        None => return None,
    };
    let left = match conservative_density_step(old.left, 0, face_left, source.left) {
        Some(left) => left,
        None => return None,
    };
    let center = match conservative_density_step(old.center, face_left, face_right, source.center) {
        Some(center) => center,
        None => return None,
    };
    let right = match conservative_density_step(old.right, face_right, 0, source.right) {
        Some(right) => right,
        None => return None,
    };
    Some(DensityLine3 {
        left,
        center,
        right,
    })
}
