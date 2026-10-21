use apeiron::learn::fold::fit::steps::fwht_words;
use apeiron::learn::solve::chol::chol_solve;
use apeiron::learn::solve::q32::{ilog2_pos, narrow, round_q32_even, rshift_rne};

use super::FoldFitError;

/// Columns per `r` group scale.
pub const GROUP: usize = 128;
/// Exponent a shifted panel is filled to.
const BOUND_BITS: u32 = 35;
/// `1.0` as a raw word.
const UNIT_RAW: i64 = 1 << 32;
/// `round(1e-12·2⁶²)`.
const EPS_Q: i64 = 4_611_686;

/// The `fold_fit::metric(H, ridge)` body in words, for the diagnostic below.
pub fn metric_words(h: &[f64], n: usize, ridge: i64) -> Vec<i64> {
    let mut sum: i128 = 0;
    for i in 0..n {
        sum += i128::from(super::grid(h[i * n + i]));
    }
    let mean = sum / n as i128;
    let add = round_q32_even(i128::from(ridge) * mean).expect("metric ridge");
    let mut hl = super::qw(h);
    for i in 0..n {
        hl[i * n + i] += add;
    }
    hl
}

fn shape() -> FoldFitError {
    FoldFitError::ShapeMismatch
}

/// `a·b` as a Q32.32 word, one ties-even rounding.
fn mul_word(a: i64, b: i64) -> Result<i64, FoldFitError> {
    round_q32_even(i128::from(a) * i128::from(b)).map_err(|_| shape())
}

/// `(v + 2³¹) >> 32`: the exact product's single rounding (`syrk::round32`).
fn round32(v: i128) -> Result<i64, FoldFitError> {
    let s = v.checked_add(1_i128 << 31).ok_or_else(shape)?;
    i64::try_from(s >> 32).map_err(|_| shape())
}

/// `(v + 2⁶³) >> 64`: the Q96.96 triple-product accumulator's single rounding
/// (`syrk::round64`).
fn round64(v: i128) -> Result<i64, FoldFitError> {
    let s = v.checked_add(1_i128 << 63).ok_or_else(shape)?;
    i64::try_from(s >> 64).map_err(|_| shape())
}

/// `round_q64`: the old body's Q64.64 accumulator rounding to a Q32.32 word.
fn round_q64(v: i128) -> Result<i64, FoldFitError> {
    round_q32_even(v).map_err(|_| shape())
}

/// `v·2⁻ᵏ`: round-to-nearest for `k > 0`, exact checked left shift for `k < 0`.
fn shift_word(v: i64, k: i32) -> Result<i64, FoldFitError> {
    match k.signum() {
        1 => rshift_rne(i128::from(v), k as u32)
            .and_then(narrow)
            .ok_or_else(shape),
        -1 => i128::from(v)
            .checked_shl((-k) as u32)
            .and_then(narrow)
            .ok_or_else(shape),
        _ => Ok(v),
    }
}

fn max_mag(x: &[i64]) -> u128 {
    x.iter()
        .fold(0u128, |m, &v| m.max(u128::from(v.unsigned_abs())))
}

/// The signed `k` with `max·2⁻ᵏ ∈ (2³⁴, 2³⁵]` raw.
fn panel_shift_words(max_mag: u128) -> i32 {
    match ilog2_pos(max_mag) {
        Some(bits) => bits as i32 + 1 - BOUND_BITS as i32,
        None => 0,
    }
}

/// The exact `2⁻ˢ` rescale keeping the Hadamard product a Q32.32 word.
fn product_shift(a: u128, h: u128) -> i32 {
    let bl = |m: u128| ilog2_pos(m).map_or(0, |b| b as i32 + 1);
    (bl(a) + bl(h) - 32 - 62).max(0)
}

fn ridge_word(mean: i128) -> Result<i64, FoldFitError> {
    let scaled = mean.checked_mul(i128::from(EPS_Q)).ok_or_else(shape)?;
    rshift_rne(scaled, 62).and_then(narrow).ok_or_else(shape)
}

fn shift_in_place(x: &mut [i64], k: i32) -> Result<(), FoldFitError> {
    if k == 0 {
        return Ok(());
    }
    for v in x {
        *v = shift_word(*v, k)?;
    }
    Ok(())
}

fn shift_copy(x: &[i64], k: i32) -> Result<Vec<i64>, FoldFitError> {
    let mut y = x.to_vec();
    shift_in_place(&mut y, k)?;
    Ok(y)
}

fn add_ridge(a: &mut [i64], n: usize, ridge: i64) -> Result<(), FoldFitError> {
    for i in 0..n {
        let d = &mut a[i * n + i];
        *d = d.checked_add(ridge).ok_or_else(shape)?;
    }
    Ok(())
}

fn diag_sum(a: &[i64], n: usize) -> i128 {
    (0..n).map(|i| i128::from(a[i * n + i])).sum()
}

fn fill_columns(x: &mut [i64], o: usize, n: usize) -> Result<Vec<i32>, FoldFitError> {
    let mut e = vec![0i32; n];
    for (j, ej) in e.iter_mut().enumerate() {
        let m = (0..o).fold(0u128, |m, q| m.max(u128::from(x[q * n + j].unsigned_abs())));
        *ej = (-panel_shift_words(m)).max(0);
        for q in 0..o {
            x[q * n + j] = shift_word(x[q * n + j], -*ej)?;
        }
    }
    Ok(e)
}

fn filled_diag_mean(a: &[i64], e: &[i32], n: usize) -> Result<i128, FoldFitError> {
    let mut acc = 0i128;
    for (j, &ej) in e.iter().enumerate() {
        acc += i128::from(shift_word(a[j * n + j], 2 * ej)?);
    }
    Ok(acc / n as i128)
}

fn add_filled_ridge(a: &mut [i64], e: &[i32], n: usize, ridge: i64) -> Result<(), FoldFitError> {
    for (j, &ej) in e.iter().enumerate() {
        let d = &mut a[j * n + j];
        *d = d
            .checked_add(shift_word(ridge, -2 * ej)?)
            .ok_or_else(shape)?;
    }
    Ok(())
}

fn row_weight(g_row: Option<&[i64]>, o: usize) -> Result<Vec<i64>, FoldFitError> {
    let Some(g) = g_row else {
        return Ok(vec![UNIT_RAW; o]);
    };
    if g.len() != o {
        return Err(shape());
    }
    let mut v = g.to_vec();
    let k = panel_shift_words(max_mag(&v)).max(0);
    shift_in_place(&mut v, k)?;
    Ok(v)
}

/// The transpose of a row-major `cols × rows` matrix into `rows × cols`.
fn transpose_grid(m: &[i64], rows: usize, cols: usize) -> Vec<i64> {
    let mut t = vec![0i64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[r * cols + c] = m[c * rows + r];
        }
    }
    t
}

fn group_scale(
    s: &[i64],
    r: &[i64],
    o: usize,
    n: usize,
    g: usize,
) -> Result<Vec<i64>, FoldFitError> {
    let mut b = vec![0i64; o * n];
    for oo in 0..o {
        for i in 0..n {
            b[oo * n + i] = mul_word(s[oo * n + i], r[oo * g + i / GROUP])?;
        }
    }
    Ok(b)
}

fn col_scale(x: &[i64], c: &[i64], n: usize) -> Result<Vec<i64>, FoldFitError> {
    let mut out = vec![0i64; x.len()];
    for i in 0..x.len() {
        out[i] = mul_word(x[i], c[i % n])?;
    }
    Ok(out)
}

fn metric_scale(hl: &[i64], c: &[i64], n: usize) -> Result<Vec<i64>, FoldFitError> {
    let mut out = vec![0i64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = mul_word(mul_word(hl[i * n + j], c[i])?, c[j])?;
        }
    }
    Ok(out)
}

fn had_gram_grid(h: &[i64], n: usize) -> Result<Vec<i64>, FoldFitError> {
    let mut t = h.to_vec();
    fwht_words(&mut t).map_err(|_| shape())?;
    let mut u = transpose_grid(&t, n, n);
    fwht_words(&mut u).map_err(|_| shape())?;
    Ok(transpose_grid(&u, n, n))
}

/// `C[i,j] = round32(Σ_l a[i,l]·b[l,j])`: the exact `gemm_nn` body.
pub fn gemm_nn(
    a: &[i64],
    b: &[i64],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<i64>, FoldFitError> {
    if m == 0 || k == 0 || n == 0 || a.len() != m * k || b.len() != k * n {
        return Err(shape());
    }
    let mut c = vec![0i64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: i128 = 0;
            for l in 0..k {
                acc += i128::from(a[i * k + l]) * i128::from(b[l * n + j]);
            }
            c[i * n + j] = round32(acc)?;
        }
    }
    Ok(c)
}

/// `round64(Σ_o w[o]·x[o,i]·x[o,j])`, mirrored: the exact weighted Gram.
pub fn wsyrk(x: &[i64], w: Option<&[i64]>, o: usize, n: usize) -> Result<Vec<i64>, FoldFitError> {
    if n == 0 || o == 0 || x.len() != o * n || w.is_some_and(|ws| ws.len() != o) {
        return Err(shape());
    }
    let mut g = vec![0i64; n * n];
    for i in 0..n {
        for j in i..n {
            let mut acc: i128 = 0;
            for oo in 0..o {
                let wv = match w {
                    Some(ws) => i128::from(ws[oo]),
                    None => i128::from(UNIT_RAW),
                };
                acc += wv * i128::from(x[oo * n + i]) * i128::from(x[oo * n + j]);
            }
            let v = round64(acc)?;
            g[i * n + j] = v;
            g[j * n + i] = v;
        }
    }
    Ok(g)
}

fn rhs_grid(x: &[i64], y: &[i64], w: &[i64], o: usize, n: usize) -> Result<Vec<i64>, FoldFitError> {
    let mut out = vec![0i64; n];
    for i in 0..n {
        let mut acc = 0i128;
        for oo in 0..o {
            let p = mul_word(x[oo * n + i], y[oo * n + i])?;
            acc = i128::from(p)
                .checked_mul(i128::from(w[oo]))
                .and_then(|v| acc.checked_add(v))
                .ok_or_else(shape)?;
        }
        out[i] = round_q64(acc)?;
    }
    Ok(out)
}

fn t_rows(p: &[i64], wshl: &[i64], o: usize, n: usize, g: usize) -> Result<Vec<i64>, FoldFitError> {
    let mut t = vec![0i64; o * g];
    for oo in 0..o {
        for gg in 0..g {
            let mut acc = 0i128;
            for k in 0..GROUP {
                let idx = oo * n + gg * GROUP + k;
                acc = i128::from(p[idx])
                    .checked_mul(i128::from(wshl[idx]))
                    .and_then(|v| acc.checked_add(v))
                    .ok_or_else(shape)?;
            }
            t[oo * g + gg] = round_q64(acc)?;
        }
    }
    Ok(t)
}

fn m_band(
    band: &mut [i128],
    kg: &[i64],
    p: &[i64],
    n: usize,
    g: usize,
) -> Result<(), FoldFitError> {
    for oo in 0..band.len() / g {
        for gg2 in 0..g {
            let mut acc = band[oo * g + gg2];
            for k in 0..GROUP {
                let idx = oo * n + gg2 * GROUP + k;
                acc = i128::from(kg[idx])
                    .checked_mul(i128::from(p[idx]))
                    .and_then(|v| acc.checked_add(v))
                    .ok_or_else(shape)?;
            }
            band[oo * g + gg2] = acc;
        }
    }
    Ok(())
}

fn metric_grid(a: &[i64], b: &[i64], o: usize, n: usize) -> Result<Vec<i64>, FoldFitError> {
    let ka = panel_shift_words(max_mag(a));
    let kb = panel_shift_words(max_mag(b));
    let as_ = shift_copy(a, ka)?;
    let bs = shift_copy(b, kb)?;
    let mut out = gemm_nn(&as_, &bs, o, n, n)?;
    shift_in_place(&mut out, -(ka + kb))?;
    Ok(out)
}

/// `Ws·Hl` on the exact body.
pub fn mul_metric(ws: &[i64], hl: &[i64], o: usize, n: usize) -> Result<Vec<i64>, FoldFitError> {
    if ws.len() != o * n || hl.len() != n * n {
        return Err(shape());
    }
    if o == 0 || n == 0 {
        return Ok(vec![0i64; o * n]);
    }
    metric_grid(ws, hl, o, n)
}

/// `J(Ws, Wp, Hl)` on the exact body.
pub fn objective(
    ws: &[i64],
    wp: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
) -> Result<i64, FoldFitError> {
    if ws.len() != o * n || wp.len() != o * n || hl.len() != n * n {
        return Err(shape());
    }
    let mut d = vec![0i64; o * n];
    for i in 0..d.len() {
        d[i] = ws[i].checked_sub(wp[i]).ok_or_else(shape)?;
    }
    if o == 0 || n == 0 {
        return Ok(0);
    }
    let y = metric_grid(&d, hl, o, n)?;
    let mut acc = 0i128;
    for i in 0..d.len() {
        acc = i128::from(d[i])
            .checked_mul(i128::from(y[i]))
            .and_then(|v| acc.checked_add(v))
            .ok_or_else(shape)?;
    }
    round_q64(acc)
}

/// The `c_step` body on the exact Gram.
pub fn c_step_wshl(
    ws: &[i64],
    hl: &[i64],
    wshl: &[i64],
    s: &[i64],
    r: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
) -> Result<Vec<i64>, FoldFitError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(shape());
    }
    let g = n / GROUP;
    if ws.len() != o * n
        || hl.len() != n * n
        || wshl.len() != o * n
        || s.len() != o * n
        || r.len() != o * g
        || g_row.is_some_and(|gr| gr.len() != o)
    {
        return Err(shape());
    }
    let mut sh = group_scale(s, r, o, n, g)?;
    let k = panel_shift_words(max_mag(&sh));
    shift_in_place(&mut sh, k)?;
    let e = fill_columns(&mut sh, o, n)?;
    let gr_g = row_weight(g_row, o)?;
    let mut a = wsyrk(&sh, Some(&gr_g), o, n)?;
    let sh2 = product_shift(max_mag(&a), max_mag(hl));
    shift_in_place(&mut a, sh2)?;
    hadamard_into(&mut a, hl)?;
    let mean = filled_diag_mean(&a, &e, n)?;
    let mut ridge = ridge_word(mean)?;
    if mean > 0 && ridge == 0 {
        ridge = 1;
    }
    add_filled_ridge(&mut a, &e, n, ridge)?;
    let mut b = rhs_grid(wshl, &sh, &gr_g, o, n)?;
    shift_in_place(&mut b, sh2)?;
    let solved = chol_solve(&a, &b, n, 1).map_err(|_| FoldFitError::NotPositiveDefinite)?;
    (0..n).map(|j| shift_word(solved.x[j], k - e[j])).collect()
}

/// `A ⊙ Hl` in place, one `round32` per entry (`syrk::hadamard_into`).
fn hadamard_into(a: &mut [i64], h: &[i64]) -> Result<(), FoldFitError> {
    if a.len() != h.len() {
        return Err(shape());
    }
    for i in 0..a.len() {
        a[i] = round32(i128::from(a[i]) * i128::from(h[i]))?;
    }
    Ok(())
}

/// The `r_step` body on the exact Gram.
pub fn r_step_wshl(
    ws: &[i64],
    hl: &[i64],
    wshl: &[i64],
    s: &[i64],
    c: &[i64],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, FoldFitError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(shape());
    }
    let g = n / GROUP;
    if ws.len() != o * n
        || hl.len() != n * n
        || wshl.len() != o * n
        || s.len() != o * n
        || c.len() != n
    {
        return Err(shape());
    }
    if o == 0 {
        return Ok(Vec::new());
    }
    let mut p = col_scale(s, c, n)?;
    let kp = panel_shift_words(max_mag(&p));
    shift_in_place(&mut p, kp)?;
    let kh = panel_shift_words(max_mag(hl));
    let hl_s = shift_copy(hl, kh)?;
    let t = t_rows(&p, wshl, o, n, g)?;
    let mut ms = vec![0i128; g * o * g];
    let mut panel = vec![0i64; o * GROUP];
    for gg in 0..g {
        for oo in 0..o {
            panel[oo * GROUP..(oo + 1) * GROUP]
                .copy_from_slice(&p[oo * n + gg * GROUP..oo * n + (gg + 1) * GROUP]);
        }
        let kg = gemm_nn(
            &panel,
            &hl_s[gg * GROUP * n..(gg + 1) * GROUP * n],
            o,
            GROUP,
            n,
        )?;
        m_band(&mut ms[gg * o * g..(gg + 1) * o * g], &kg, &p, n, g)?;
    }
    let mut out = vec![0i64; o * g];
    let mut m = vec![0i64; g * g];
    for oo in 0..o {
        for r0 in 0..g {
            for c0 in 0..g {
                m[r0 * g + c0] = round_q64(ms[(r0 * o + oo) * g + c0])?;
            }
        }
        let ridge = ridge_word(diag_sum(&m, g) / g as i128)?;
        add_ridge(&mut m, g, ridge)?;
        let solved = chol_solve(&m, &t[oo * g..(oo + 1) * g], g, 1)
            .map_err(|_| FoldFitError::NotPositiveDefinite)?;
        out[oo * g..(oo + 1) * g].copy_from_slice(&solved.x);
    }
    shift_in_place(&mut out, kp + kh)?;
    Ok(out)
}

/// The `d_step` body on the exact Gram.
pub fn d_step_wshl(
    ws: &[i64],
    hl: &[i64],
    wshl: &[i64],
    s_rot: &[i64],
    r: &[i64],
    c: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
) -> Result<Vec<i64>, FoldFitError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(shape());
    }
    let g = n / GROUP;
    if ws.len() != o * n
        || hl.len() != n * n
        || wshl.len() != o * n
        || s_rot.len() != o * n
        || r.len() != o * g
        || c.len() != n
        || g_row.is_some_and(|gr| gr.len() != o)
    {
        return Err(shape());
    }
    let mut g_rot = col_scale(wshl, c, n)?;
    fwht_words(&mut g_rot).map_err(|_| shape())?;
    let h_rot = had_gram_grid(&metric_scale(hl, c, n)?, n)?;
    let mut b_scaled = group_scale(s_rot, r, o, n, g)?;
    let k = panel_shift_words(max_mag(&b_scaled));
    shift_in_place(&mut b_scaled, k)?;
    let e = fill_columns(&mut b_scaled, o, n)?;
    let gr_g = row_weight(g_row, o)?;
    let mut a = wsyrk(&b_scaled, Some(&gr_g), o, n)?;
    let sd = product_shift(max_mag(&a), max_mag(&h_rot));
    shift_in_place(&mut a, sd)?;
    hadamard_into(&mut a, &h_rot)?;
    let mean = filled_diag_mean(&a, &e, n)?;
    let ridge = ridge_word(mean.max(0))?.max(1);
    add_filled_ridge(&mut a, &e, n, ridge)?;
    let mut b = rhs_grid(&b_scaled, &g_rot, &gr_g, o, n)?;
    shift_in_place(&mut b, sd)?;
    let solved = chol_solve(&a, &b, n, 1).map_err(|_| FoldFitError::NotPositiveDefinite)?;
    (0..n).map(|j| shift_word(solved.x[j], k - e[j])).collect()
}
