// Test-only `f64` oracle for the 128-block Hadamard transform: the deleted `steps::fwht128`
// boundary, rebuilt on the word transform `fwht_words`. Not a production path. The module keeps no
// top-level `use` items so it can be included at a test crate root without name collisions.

/// The normalised 128-block Walsh–Hadamard transform of an `f64` slice, quantised once on entry and
/// read back once on exit, exactly as the deleted boundary did.
///
/// # Errors
/// `StepsError::ShapeMismatch` on a non-multiple length or an entry outside the grid.
pub fn fwht128(x: &[f64]) -> Result<Vec<f64>, apeiron::learn::fold::fit::steps::StepsError> {
    use apeiron::learn::fold::fit::steps::{fwht_words, StepsError};
    use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};
    let mut y: Vec<i64> = x
        .iter()
        .map(|&v| to_grid(v, GRID_BOUND))
        .collect::<Option<_>>()
        .ok_or(StepsError::ShapeMismatch)?;
    fwht_words(&mut y)?;
    Ok(y.into_iter().map(from_grid).collect())
}
