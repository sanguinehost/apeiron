//! Test-only `f64` oracle carried over from the retired `src/learn/ridge.rs`: the plain `Cholesky`
//! factor `L` and the dense SPD solve `A·X = B`. The bodies keep that module's arithmetic; the
//! crate itself computes in Q32.32 (`crate::learn::solve::{chol,q32}`) and no `src/` file calls
//! this.
//!
//! Each test binary that needs it includes
//! `#[path = "support/ridge_oracle.rs"] mod ridge_oracle;` and calls `ridge_oracle::solve_spd` /
//! `ridge_oracle::cholesky`. Grade `tested`: this is the reference input to the solve tests, not a
//! crate path.

#![allow(
    dead_code,
    reason = "one shared oracle module compiled into binaries that call one fn"
)]
#![allow(clippy::needless_range_loop)]

/// Refusal from the oracle: `"shape mismatch"` when a slice length disagrees with the declared
/// shape, `"not positive definite"` when a `Cholesky` pivot is non-positive or NaN.
type Refusal = &'static str;

/// `Cholesky` factor `L` (lower-triangular, row-major `[d, d]`) of SPD `a`. O(d³/3) time;
/// O(d²) space.
///
/// # Complexity
/// Time: `d(d+1)/2` pivots, each a dot product of length `j` over `f64` multiply-adds, so
/// `Σ_{i} Σ_{j≤i} j = O(d³/3)` multiply-adds, the measure this bounds. Space: `d²` `f64` output =
/// `8·d²` bytes, plus the `d²` output slice itself; no other allocation.
///
/// # Errors
/// `"shape mismatch"` if `a.len() != d*d`; `"not positive definite"` if a diagonal pivot is
/// non-positive or NaN.
pub fn cholesky(a: &[f64], d: usize) -> Result<Vec<f64>, Refusal> {
    if a.len() != d * d {
        return Err("shape mismatch");
    }
    let mut l = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s.is_nan() || s <= 0.0 {
                    return Err("not positive definite");
                }
                l[i * d + j] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Ok(l)
}

/// Solve the SPD system `A · X = Bmat` for `X`; `a` row-major `[d, d]`, `bmat` row-major
/// `[d, out]`, return `X` row-major `[d, out]`. `Cholesky` factor, then a forward and a back
/// substitution per right-hand-side column.
///
/// # Complexity
/// Time: one factor `O(d³/3)` plus `2·out` triangular substitutions of `d²/2` multiply-adds each,
/// `O(d³ + d²·out)` in `f64` multiply-adds. Space: the `d²` factor plus the `d·out` solution,
/// `8·(d² + d·out)` bytes.
///
/// # Errors
/// `"shape mismatch"` if `a.len() != d*d` or `bmat.len() != d*out`; `"not positive definite"` from
/// [`cholesky`].
pub fn solve_spd(a: &[f64], bmat: &[f64], d: usize, out: usize) -> Result<Vec<f64>, Refusal> {
    if a.len() != d * d || bmat.len() != d * out {
        return Err("shape mismatch");
    }
    let l = cholesky(a, d)?;
    let mut x = bmat.to_vec();
    // Forward: x[i] = (b[i] - Σ_{k<i} L[i,k] x[k]) / L[i,i].
    for i in 0..d {
        let lii = l[i * d + i];
        for o in 0..out {
            let mut s = x[i * out + o];
            for k in 0..i {
                s -= l[i * d + k] * x[k * out + o];
            }
            x[i * out + o] = s / lii;
        }
    }
    // Back: x[i] = (y[i] - Σ_{k>i} L[k,i] x[k]) / L[i,i].
    for i in (0..d).rev() {
        let lii = l[i * d + i];
        for o in 0..out {
            let mut s = x[i * out + o];
            for k in (i + 1)..d {
                s -= l[k * d + i] * x[k * out + o];
            }
            x[i * out + o] = s / lii;
        }
    }
    Ok(x)
}
