//! Bit budget of the one-bit transformer arm `...-sink-fnorm-grow-res-qmlp-nb`, split by component.
//!
//! Each function is the closed form of one counter in `tools/onebit/fold_transfer.py` (removed; at 2f15d6a9)
//! (`nsign`, `nr`, `nc`, `nd`, `ncorr`, accumulated at lines 1214, 1237, 1248 and 1260), evaluated
//! on the layer-uniform shape `(d_model, kv, ff, layers, k_rows)`. Every layer holds twelve
//! `fold_fit` calls `(O, n)`:
//!
//! | plane | tensors | `(O, n)` |
//! |---|---|---|
//! | 1 | q, o | `(d, d)` |
//! | 1 | k, v | `(kv, d)` |
//! | 1 | gate, up | `(ff, d)` |
//! | 1 | down | `(d, ff)` |
//! | 2 | o | `(d, d)` |
//! | 2 | k, v | `(kv, d)` |
//! | 2 | gate, up (top `k_rows` rows by target row mass) | `(k_rows, d)` |
//!
//! The scale widths are those of the active `fold_fit` branch (`bscale and had`, not `cgauge`,
//! `fold_transfer.py:829-836`): `d` is `quant_c_u8(bits=5)`, `c` is `quant_c_u8(bits=6)`, `r` is
//! `quant_r_hier`. The corrector (`lowrank_corrector(u8_corr=True)`, `fold_transfer.py:549-555`)
//! runs on q (rank 2) and down (rank 1) and is counted unconditionally
//! (`ncorr += b_corr`, `fold_transfer.py:1260`).
//!
//! The row-index term is reported by [`index_bits`] and is not part of [`total_bits`]: the Python
//! counter omits it, and the receipt total equals the sum of the five components. The caller
//! supplies `index_bits_per_set`, a bound with `C(ff, k_rows) <= 2^index_bits_per_set`.
//!
//! This module depends only on `core`, so `proof/extract/lib.rs` can pull it in by `#[path]`.
//! Every add and multiply is checked `u64`: an overflow, or a width that does not divide `group`,
//! returns `None` and nothing wraps. Each function is a fixed number of checked operations and no
//! loop: O(1) time and space.

/// Shape, plane set and scale widths of the arm. All fields are bit counts or tensor extents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cfg {
    /// Model width `d` (columns of q, k, v, gate, up; rows of o, down).
    pub d_model: u64,
    /// Key/value width `kv` (rows of k and v).
    pub kv: u64,
    /// MLP hidden width `ff` (rows of gate and up; columns of down).
    pub ff: u64,
    /// Number of transformer layers.
    pub layers: u64,
    /// Rows of gate and up that receive a second sign plane.
    pub k_rows: u64,
    /// Scale group width along `n` (`GROUP`, `fold_transfer.py:37`); must divide every `n`.
    pub group: u64,
    /// One affine header: two `fp16` scalars (`quant_c_u8`, `fold_transfer.py:511`).
    pub header_bits: u64,
    /// Bits per `(row, group)` level in `quant_r_hier` (`fold_transfer.py:540`).
    pub r_level_bits: u64,
    /// Bits per row of the group-mean index `u6` (`fold_transfer.py:540`, the `8 * O` term).
    pub r_mu_bits: u64,
    /// Bits per row of the spread index (4 levels; the rest of the `8 * O` term).
    pub r_sp_bits: u64,
    /// Bits per column of `c` (`quant_c_u8(bits=6)`, `fold_transfer.py:836`).
    pub c_bits: u64,
    /// Bits per column of `d` (`quant_c_u8(bits=5)`, `fold_transfer.py:830`).
    pub d_bits: u64,
    /// Bits per entry of a corrector factor (`quant_c_u8(bits=8)`, `fold_transfer.py:551-554`).
    pub corr_bits: u64,
    /// Corrector rank on q (`rk`, `fold_transfer.py:1255`).
    pub rank_q: u64,
    /// Corrector rank on down.
    pub rank_down: u64,
    /// Row-index sets per layer (gate and up).
    pub index_sets: u64,
    /// Bits to name one `k_rows`-subset of `ff` rows.
    pub index_bits_per_set: u64,
}

impl Cfg {
    /// Qwen2.5-0.5B, arm `...-sink-fnorm-grow-res-qmlp-nb`; `index_bits_per_set = 2044 =
    /// ceil(log2 C(4864, 416))`.
    #[must_use]
    pub const fn qwen25_05b_nb() -> Self {
        Self {
            d_model: 896,
            kv: 128,
            ff: 4864,
            layers: 24,
            k_rows: 416,
            group: 128,
            header_bits: 32,
            r_level_bits: 4,
            r_mu_bits: 6,
            r_sp_bits: 2,
            c_bits: 6,
            d_bits: 5,
            corr_bits: 8,
            rank_q: 2,
            rank_down: 1,
            index_sets: 2,
            index_bits_per_set: 2044,
        }
    }
}

/// Checked `a + b`.
fn add(a: u64, b: u64) -> Option<u64> {
    a.checked_add(b)
}

/// Checked `a * b`.
fn mul(a: u64, b: u64) -> Option<u64> {
    a.checked_mul(b)
}

/// Sign bits of one layer: `3 d^2 + 4 kv d + 3 ff d + 2 k_rows d` (plane 1 on all seven tensors,
/// plane 2 on k, v, o and on `k_rows` rows of gate and up).
fn sign_layer(c: &Cfg) -> Option<u64> {
    let dd = mul(c.d_model, c.d_model)?;
    let kvd = mul(c.kv, c.d_model)?;
    let ffd = mul(c.ff, c.d_model)?;
    let krd = mul(c.k_rows, c.d_model)?;
    let t = add(mul(3, dd)?, mul(4, kvd)?)?;
    let t = add(t, mul(3, ffd)?)?;
    add(t, mul(2, krd)?)
}

/// `nsign`: sign bits over all layers (`fold_transfer.py:1214,1237,1248`).
#[must_use]
pub fn bits_sign(c: &Cfg) -> Option<u64> {
    mul(c.layers, sign_layer(c)?)
}

/// `ntot`: weights of the seven linears over all layers; plane 1 only (`fold_transfer.py:1214`).
#[must_use]
pub fn params(c: &Cfg) -> Option<u64> {
    let dd = mul(c.d_model, c.d_model)?;
    let kvd = mul(c.kv, c.d_model)?;
    let ffd = mul(c.ff, c.d_model)?;
    let t = add(mul(2, dd)?, mul(2, kvd)?)?;
    mul(c.layers, add(t, mul(3, ffd)?)?)
}

/// `quant_r_hier` bits for an `(O, n)` call: `r_level_bits * O * (n / group)`
/// `+ (r_mu_bits + r_sp_bits) * O + 2 * header_bits` (`fold_transfer.py:540`, 4*O*G + 8*O + 64).
#[allow(
    clippy::manual_is_multiple_of,
    reason = "`%` is the operator the Aeneas extraction fragment supports; `is_multiple_of` is not"
)]
fn call_r(o: u64, n: u64, c: &Cfg) -> Option<u64> {
    if c.group == 0 || n % c.group != 0 {
        return None;
    }
    let groups = n / c.group;
    let levels = mul(mul(c.r_level_bits, o)?, groups)?;
    let rows = mul(add(c.r_mu_bits, c.r_sp_bits)?, o)?;
    add(add(levels, rows)?, mul(2, c.header_bits)?)
}

/// `quant_c_u8(bits)` bits for a vector of length `n`: `bits * n + header_bits`
/// (`fold_transfer.py:511`).
fn call_vec(bits: u64, n: u64, c: &Cfg) -> Option<u64> {
    add(mul(bits, n)?, c.header_bits)
}

/// `nr`: `quant_r_hier` bits over all layers (`fold_transfer.py:1214,1237,1248`). Per layer the
/// twelve calls are `3 x (d,d)`, `4 x (kv,d)`, `2 x (ff,d)`, `1 x (d,ff)`, `2 x (k_rows,d)`.
#[must_use]
pub fn bits_r(c: &Cfg) -> Option<u64> {
    let t = mul(3, call_r(c.d_model, c.d_model, c)?)?;
    let t = add(t, mul(4, call_r(c.kv, c.d_model, c)?)?)?;
    let t = add(t, mul(2, call_r(c.ff, c.d_model, c)?)?)?;
    let t = add(t, call_r(c.d_model, c.ff, c)?)?;
    let t = add(t, mul(2, call_r(c.k_rows, c.d_model, c)?)?)?;
    mul(c.layers, t)
}

/// Sum of `bits * n + header` over the twelve calls of a layer: eleven have `n = d_model`, one
/// (down) has `n = ff`.
fn vec_layer(bits: u64, c: &Cfg) -> Option<u64> {
    let t = mul(11, call_vec(bits, c.d_model, c)?)?;
    add(t, call_vec(bits, c.ff, c)?)
}

/// `nc`: column-scale bits over all layers, `quant_c_u8(bits=6)` per call
/// (`fold_transfer.py:836,1214,1237,1248`).
#[must_use]
pub fn bits_c(c: &Cfg) -> Option<u64> {
    mul(c.layers, vec_layer(c.c_bits, c)?)
}

/// `nd`: `d`-scale bits over all layers, `quant_c_u8(bits=5)` per call
/// (`fold_transfer.py:830,1214,1237,1248`).
#[must_use]
pub fn bits_d(c: &Cfg) -> Option<u64> {
    mul(c.layers, vec_layer(c.d_bits, c)?)
}

/// Bits of a rank-`rank` corrector on an `(O, n)` tensor: `rank * (corr_bits * (O + n) + 2 *
/// header_bits)` (`lowrank_corrector(u8_corr=True)`, `fold_transfer.py:555`, `rank*(8*(O+n)+64)`).
fn corr_call(rank: u64, o: u64, n: u64, c: &Cfg) -> Option<u64> {
    let width = mul(c.corr_bits, add(o, n)?)?;
    mul(rank, add(width, mul(2, c.header_bits)?)?)
}

/// `ncorr`: corrector bits over all layers, q at `rank_q` and down at `rank_down`
/// (`fold_transfer.py:1255-1260`).
#[must_use]
pub fn bits_corr(c: &Cfg) -> Option<u64> {
    let q = corr_call(c.rank_q, c.d_model, c.d_model, c)?;
    let down = corr_call(c.rank_down, c.d_model, c.ff, c)?;
    mul(c.layers, add(q, down)?)
}

/// Receipt total `nsign + nr + nc + nd + ncorr` (`fold_transfer.py:1404`); excludes
/// [`index_bits`].
#[must_use]
pub fn total_bits(c: &Cfg) -> Option<u64> {
    let t = add(bits_sign(c)?, bits_r(c)?)?;
    let t = add(t, bits_c(c)?)?;
    let t = add(t, bits_d(c)?)?;
    add(t, bits_corr(c)?)
}

/// Side information the decoder needs for the top-`k_rows` row choice: `layers * index_sets *
/// index_bits_per_set`. Not counted by the Python estimator; not in [`total_bits`].
#[must_use]
pub fn index_bits(c: &Cfg) -> Option<u64> {
    mul(mul(c.layers, c.index_sets)?, c.index_bits_per_set)
}
