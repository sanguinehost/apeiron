//! Fit one layer's `NanoQuant` linears from a GGUF teacher and write an `APNQPK01` pack.
//!
//! `tests/learn_nanoquant.rs::nq_fit` drives `learn::nanoquant::factorize` from a cached
//! calibration Gram and a factor cache; this example is the smallest standalone driver: it opens
//! the teacher, reads the seven linears of each requested layer through
//! `learn::grad::load::load_matrix`, fits them at the ranks the bit budget names, and writes the
//! pack `src/format/nq.rs` serves. It is an example, not a library entry point: the only public
//! surface it uses is already public for the ignored test.
//!
//! The preconditioners are the reference's two calibration statistics: the mean input Gram
//! diagonal `g{l}.{tag}.{gram}.diag.i64` (`tests/learn_calib.rs::calib_diag_write`) and the mean
//! squared output gradient `g{l}.{tag}.{kind}.o_norm.i64`
//! (`tests/learn_calib.rs::calib_onorm_write`), both with the reference's per-sample
//! `99.9`th-percentile row clip (`online` by default), `0.4` covariance shrinkage and the
//! reference's `1e6` output-gradient unit already applied, header `(1, width)`, Q32.32. This is the
//! onepath quality fit: a rate below the sign-factor cap spends its extra bits on the same call's
//! `--residual` planes, and the pack header is the rate. There is no second fitter and no
//! fallback preconditioner. `--ones-norm` is the reference-parity exemption, and
//! `--allow-column-rms` names a shape test; neither writes a quality point.
//!
//! ```text
//! cargo run --release --locked --example nq_fit -- \
//!   --model <teacher.gguf> --out <pack.nqp> --layers 0 \
//!   --kinds q,k,v,o,gate,up,down --iters 400 --ranks q:800,k:608,v:608,o:800,gate:864,up:864,down:864 \
//!   --diag-dir <dir> --o-norm-dir <dir> --diag-tag s128 [--residual 1] [--rel] [--require-device] \
//!   [--reg <ridge>] [--seed <u64>] [--salient-frac 0.001] [--schedule linear|cubic]
//! ```
//!
//! `--salient-frac <f>` removes the `ceil(f·inn)` input columns with the largest calibration Gram
//! diagonal from the fit before ADMM; the served pack restores them exactly (`s_pre[idx] = 0`,
//! `sal = f16(W[:, idx] − R[:, idx])`), so this flag is only the first half of the term: the fit
//! must not spend rank on them. Without it the Gram's energy can collapse the factor onto a few
//! columns (`l06 down`: 99.85% of `tr(H)` in 13 of 12,288), which no per-rank weight can undo.
//!
//! `--residual <planes>` fits that many dense sign-residual layers
//! (`learn::nanoquant::dense_residual`) in the same call: one bit per weight per plane plus one
//! bfloat16 row scale, from
//! `R_0 = W − Ŵ`. This is the rate dial's upper range, not a second approach: a 4-bit point is the
//! same `nq_fit` with the planes the header counts, and a 1-bit point is the same call with zero.
//! The sparse values absorb the residual at the salient columns (the pack stays exact there).
//!
//! `--rel` prints the relative Frobenius error of the served factor, `‖W − Ŵ‖/‖W‖`; it costs one
//! `NqFactor::weight()` per linear. The pack is written after every requested section fits, so a
//! partial list writes a partial pack (a legal `APNQPK01`; the engine serves the omitted linears
//! from the skeleton).
//!
//! `--schedule <linear|cubic>` selects the ADMM penalty `ρ(t/T)`: `linear` is `ρ = x` (the
//! reference CLI default) and `cubic` is `ρ = min(1, x)³` (the value the reference's resolved 8B
//! config carries). Both reach `ρ = 1` at the last iteration; they differ in the early part.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::struct_excessive_bools,
    reason = "the driver's dimensions are usize below 2^32, its rates are f64 reports, the loops \
              are index maps into row-major weight and sign arrays, and its flags are independent \
              command-line switches"
)]

use std::error::Error;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::format::nq::{write, NqPack, PackedLinear};
use apeiron::learn::grad::load::load_matrix;
use apeiron::learn::nanoquant::{
    dense_residual, factorize, rank_for_bits, NqConfig, NqFactor, NqResidual, Schedule,
};
use apeiron::learn::solve::q32::from_grid;

/// The seven fitted linears in pack-kind order: `(kind index, tensor suffix, Gram kind)`.
const KINDS: [(&str, usize, &str, &str); 7] = [
    ("q", 0, "attn_q", "qkv"),
    ("k", 1, "attn_k", "qkv"),
    ("v", 2, "attn_v", "qkv"),
    ("o", 3, "attn_output", "o"),
    ("gate", 4, "ffn_gate", "gate_up"),
    ("up", 5, "ffn_up", "gate_up"),
    ("down", 6, "ffn_down", "down"),
];

/// The reference model's module suffix per kind, the name of its `i_norm`/`o_norm` dump.
const REF: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

struct Args {
    model: String,
    out: Option<String>,
    layers: Vec<usize>,
    kinds: Vec<usize>,
    iters: usize,
    ppm: u64,
    ranks: Option<Vec<usize>>,
    diag_dir: Option<String>,
    diag_tag: String,
    rel: bool,
    require_device: bool,
    reg: Option<f64>,
    seed: u64,
    salient_frac: f64,
    /// Dense sign-residual planes per section: one bit per weight per plane plus one bfloat16
    /// row scale. The rate dial's upper range, fit in the same call as the factor.
    residual: usize,
    /// Permitted only as a shape test: the column-RMS preconditioner is not a quality path.
    allow_column_rms: bool,
    schedule: Schedule,
    /// Write the fitted linear's `W`, `i_norm` and `o_norm` here and stop: the reference
    /// `factorize_admm_nanoquant` reads the same three arrays, so a parity run starts from the
    /// same numbers.
    dump: Option<String>,
    /// Directory of the reference's per-linear `i_norm`/`o_norm` dumps (little-endian f32, named
    /// `model.layers.{l}.{q,k,v,o,gate,up,down}_proj.{i,o}.f32`). Absent stats stop the fit
    /// rather than fall back.
    norm_dir: Option<String>,
    /// Directory of `g{l}.{tag}.{kind}.o_norm.i64` (`tests/learn_calib.rs::calib_onorm_write`),
    /// the output statistic `factorize` weights the rows by. Part of the onepath result, like the
    /// input diagonal: a fit with `o_norm = 1` ignores which output rows the loss moves.
    o_norm_dir: Option<String>,
    /// Fit with `i_norm = o_norm = 1`, so the goal is `W` itself and a parity run against the
    /// reference excludes the preconditioner (whose floor is ours alone).
    ones_norm: bool,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut a = Args {
        model: String::new(),
        out: None,
        layers: vec![0],
        kinds: (0..7).collect(),
        iters: 400,
        ppm: 1_174_000,
        ranks: None,
        diag_dir: None,
        diag_tag: String::from("s128"),
        rel: false,
        require_device: false,
        reg: None,
        seed: 0,
        salient_frac: 0.0,
        residual: 0,
        allow_column_rms: false,
        schedule: Schedule::Linear,
        dump: None,
        norm_dir: None,
        o_norm_dir: None,
        ones_norm: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--out" => a.out = Some(val()?),
            "--layers" => {
                a.layers = val()?
                    .split(',')
                    .map(|s| s.trim().parse())
                    .collect::<Result<_, _>>()?;
            }
            "--kinds" => {
                a.kinds = val()?
                    .split(',')
                    .map(|s| {
                        KINDS
                            .iter()
                            .position(|k| k.0 == s.trim())
                            .ok_or_else(|| format!("kind {s:?}"))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
            }
            "--iters" => a.iters = val()?.parse()?,
            "--bits-ppm" => a.ppm = val()?.parse()?,
            "--ranks" => {
                let mut r = Vec::with_capacity(7);
                let mut seen = [false; 7];
                for item in val()?.split(',') {
                    let (name, rank) = item.split_once(':').ok_or("--ranks wants kind:rank")?;
                    let at = KINDS
                        .iter()
                        .position(|k| k.0 == name.trim())
                        .ok_or_else(|| format!("rank kind {name:?}"))?;
                    if seen[at] {
                        return Err(format!("--ranks repeats {name}").into());
                    }
                    seen[at] = true;
                    while r.len() <= at {
                        r.push(0);
                    }
                    r[at] = rank.trim().parse()?;
                }
                if seen.iter().any(|s| !s) {
                    return Err("--ranks must name every kind".into());
                }
                a.ranks = Some(r);
            }
            "--diag-dir" => a.diag_dir = Some(val()?),
            "--diag-tag" => a.diag_tag = val()?,
            "--rel" => a.rel = true,
            "--require-device" => a.require_device = true,
            "--reg" => a.reg = Some(val()?.parse()?),
            "--seed" => a.seed = val()?.parse()?,
            "--salient-frac" => a.salient_frac = val()?.parse()?,
            "--residual" => a.residual = val()?.parse()?,
            "--allow-column-rms" => a.allow_column_rms = true,
            "--dump" => a.dump = Some(val()?),
            "--norm-dir" => a.norm_dir = Some(val()?),
            "--o-norm-dir" => a.o_norm_dir = Some(val()?),
            "--ones-norm" => a.ones_norm = true,
            "--schedule" => {
                a.schedule = match val()?.as_str() {
                    "linear" => Schedule::Linear,
                    "cubic" => Schedule::Cubic,
                    other => return Err(format!("schedule {other:?}").into()),
                };
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if a.model.is_empty() {
        return Err("--model is required".into());
    }
    Ok(a)
}

/// The reference's `i_norm`/`o_norm` vector at `path` (little-endian f32), `want` entries.
fn read_f32(path: &str, want: usize) -> Option<Vec<f64>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() != 4 * want {
        return None;
    }
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f64::from(f32::from_le_bytes(*c)))
            .collect(),
    )
}

/// The mean Gram diagonal at `{dir}/g{l}.{tag}.{gram}.diag.i64` (`(1, width)` u64 header, Q32.32
/// words), `Ok(None)` when the file is absent, and an error when it is present but is not exactly
/// the `(1, inn)` file the fit named.
fn read_stat(
    dir: &str,
    layer: usize,
    tag: &str,
    name: &str,
    what: &str,
    inn: usize,
) -> Result<Option<Vec<f64>>, Box<dyn Error>> {
    let path = format!("{dir}/g{layer}.{tag}.{name}.{what}.i64");
    let Ok(bytes) = std::fs::read(&path) else {
        return Ok(None);
    };
    if bytes.len() < 16 {
        return Err(format!("{path}: shorter than its 16-byte header").into());
    }
    let word = |at: usize| -> u64 {
        u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))
    };
    let (rows, width) = (word(0) as usize, word(8) as usize);
    if rows != 1 || width != inn || bytes.len() != 16 + 8 * inn {
        return Err(format!(
            "{path}: header ({rows}, {width}), {} B; want (1, {inn}), {} B",
            16 + 8 * width,
            16 + 8 * inn
        )
        .into());
    }
    Ok(Some(
        (0..width)
            .map(|i| from_grid(i64::try_from(word((2 + i) * 8)).expect("diag word")))
            .collect(),
    ))
}

/// Column RMS of a row-major `out × inn` matrix, the fallback input statistic.
fn col_rms(w: &[i64], out: usize, inn: usize) -> Vec<f64> {
    let mut d = vec![0.0f64; inn];
    for i in 0..out {
        for j in 0..inn {
            let x = from_grid(w[i * inn + j]);
            d[j] += x * x;
        }
    }
    for x in &mut d {
        *x /= out as f64;
    }
    d
}

/// `±1` signs `rows × cols` packed into little-endian u32 words, bit 1 for `+1`.
fn pack_signs(signs: &[i8], rows: usize, cols: usize) -> Vec<u32> {
    let words = cols.div_ceil(32);
    let mut out = vec![0u32; rows * words];
    for i in 0..rows {
        for j in 0..cols {
            if signs[i * cols + j] > 0 {
                out[i * words + j / 32] |= 1 << (j % 32);
            }
        }
    }
    out
}

/// `f64` to bfloat16 bits, round to nearest even through `f32`.
fn bf16(x: f64) -> u16 {
    let bits = (x as f32).to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round)) >> 16) as u16
}

/// `f64` to binary16 bits, round to nearest even. Subnormals flush to zero and overflow
/// saturates: a salient column's entry is a weight, and both cases are outside the pack's range.
fn f16(x: f64) -> u16 {
    let bits = (x as f32).to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let e = i32::try_from((bits >> 23) & 0xff).expect("8 bits");
    let man = bits & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00;
    }
    let mut exp = e - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        return sign;
    }
    let mut m = man >> 13;
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m & 1 == 1) {
        m += 1;
    }
    if m == 0x400 {
        m = 0;
        exp += 1;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | (m as u16)
}

/// The served dense form of `f` at `out × inn`, Q32.32: `NqFactor::weight()`.
fn rel_fro(f: &NqFactor, w: &[i64], mask: &[bool]) -> Result<f64, Box<dyn Error>> {
    rel_fro_mid(f, w, mask, &f.s_mid)
}

/// [`rel_fro`] with an explicit per-rank weight, for the two export arms on one fit.
fn rel_fro_mid(f: &NqFactor, w: &[i64], mask: &[bool], mid: &[f64]) -> Result<f64, Box<dyn Error>> {
    let hat = f.weight_with_mid(mid).map_err(|e| format!("{e:?}"))?;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    let inn = f.inn;
    for (at, (&a, &b)) in hat.iter().zip(w).enumerate() {
        if !mask[at % inn] {
            continue;
        }
        let e = from_grid(a) - from_grid(b);
        num += e * e;
        let v = from_grid(b);
        den += v * v;
    }
    Ok((num / den.max(f64::MIN_POSITIVE)).sqrt())
}

/// The served product on the pack's bfloat16 scales: [`NqFactor::weight`] after rounding each
/// scale vector to the grid `PackedLinear` stores. The dense residual is fit against this, not
/// against the unrounded f64 factor the ADMM exits with.
fn served_weight(f: &NqFactor) -> Result<Vec<i64>, Box<dyn Error>> {
    let pre: Vec<f64> = f.s_pre.iter().map(|&x| bf16_value(bf16(x))).collect();
    let mid: Vec<f64> = f.s_mid.iter().map(|&x| bf16_value(bf16(x))).collect();
    let post: Vec<f64> = f.s_post.iter().map(|&x| bf16_value(bf16(x))).collect();
    f.weight_with_scales(&pre, &mid, &post)
        .map_err(|e| format!("{e:?}").into())
}

/// The bfloat16 value of `bits`, for fitting against the stored scale.
fn bf16_value(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

fn packed_linear(
    layer: usize,
    kind: usize,
    f: &NqFactor,
    salient_idx: &[u32],
    residual: &NqResidual,
) -> PackedLinear {
    let u = pack_signs(&f.u, f.out, f.rank);
    let v = pack_signs(&f.v, f.rank, f.inn);
    debug_assert_eq!(u.len(), f.out * f.rank.div_ceil(32));
    debug_assert_eq!(v.len(), f.rank * f.inn.div_ceil(32));
    PackedLinear {
        layer,
        kind,
        out: f.out,
        inn: f.inn,
        rank: f.rank,
        u: u.into(),
        v: v.into(),
        s_pre: f
            .s_pre
            .iter()
            .map(|&x| bf16(x))
            .collect::<Vec<u16>>()
            .into(),
        s_mid: f
            .s_mid
            .iter()
            .map(|&x| bf16(x))
            .collect::<Vec<u16>>()
            .into(),
        s_post: f
            .s_post
            .iter()
            .map(|&x| bf16(x))
            .collect::<Vec<u16>>()
            .into(),
        salient_idx: salient_idx.to_vec().into(),
        salient: residual
            .salient
            .iter()
            .map(|&x| f16(x))
            .collect::<Vec<u16>>()
            .into(),
        res_layers: residual.s_res.len() / f.out,
        shared: false,
        res: residual.res.clone().into(),
        s_res: residual
            .s_res
            .iter()
            .map(|&x| bf16(x))
            .collect::<Vec<u16>>()
            .into(),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "the campaign's argument checks, per-kind fit loop and report are one driver"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let a = parse_args()?;
    // One quality path. The calibration Gram diagonal is the reference preconditioner and part of
    // the result: a column-RMS fit is a shape test whose PPL says nothing about a pack. Refuse it
    // by default rather than writing a pack that looks like a dial point and is not one.
    if !a.ones_norm && a.diag_dir.is_none() && a.norm_dir.is_none() && !a.allow_column_rms {
        return Err(
            "no onepath fit without the calibration Gram diagonal: pass --diag-dir <dir> \
                    (the reference preconditioner), --norm-dir for reference parity, or \
                    --ones-norm for a ones-preconditioner parity run; --allow-column-rms names a \
                    shape test, not a quality pack"
                .into(),
        );
    }
    if !a.ones_norm && a.norm_dir.is_none() && a.o_norm_dir.is_none() && !a.allow_column_rms {
        return Err(
            "no onepath fit without the calibration output statistic: pass --o-norm-dir <dir> \
                    (calib_onorm_write's `o_norm.i64`), --norm-dir for reference parity, or \
                    --ones-norm for a ones-preconditioner parity run"
                .into(),
        );
    }
    // A plane is one bit per weight plus one row scale; past 32 the header is already beyond any
    // rate this product serves, and the allocation is no longer a typo away from the grid.
    if a.residual > 32 {
        return Err("--residual: at most 32 planes, one bit per weight each".into());
    }
    // A full campaign's wall is measured on the device path. Refuse a CPU-only build before the
    // first linear rather than discovering it after the wall, as campaign 2 did.
    #[cfg(feature = "gpu")]
    println!(
        "limb device: {}",
        if apeiron::gpu::limb_available() {
            "present"
        } else {
            "absent"
        }
    );
    if a.require_device {
        #[cfg(feature = "gpu")]
        if !apeiron::gpu::limb_available() {
            return Err(
                "--require-device: no Vulkan device with cooperative matrices; run on a GPU host"
                    .into(),
            );
        }
        #[cfg(not(feature = "gpu"))]
        return Err("--require-device: built without --features gpu".into());
    }
    let engine = Engine::open(&a.model)?;
    let cfg =
        apeiron::learn::grad::model::Config::from_engine(&engine).map_err(|e| format!("{e:?}"))?;
    let q_out = cfg.q_out().map_err(|e| format!("{e:?}"))?;
    let (n, f, kv) = (cfg.n_embd, cfg.n_ff, cfg.n_head_kv * cfg.head_dim);
    // The seven linears' `(out, in)` at the Qwen3 geometry the pack kinds name.
    let shape = |kind: usize| -> (usize, usize) {
        match kind {
            0 => (q_out, n),
            1 | 2 => (kv, n),
            3 => (n, q_out),
            4 | 5 => (f, n),
            _ => (n, f),
        }
    };
    let mut linears = Vec::with_capacity(a.layers.len() * a.kinds.len());
    let t_all = Instant::now();
    for &layer in &a.layers {
        for &kind in &a.kinds {
            let (name, at, tensor, gram) = KINDS[kind];
            let (out, inn) = shape(kind);
            let mut w = load_matrix(&engine, &format!("blk.{layer}.{tensor}.weight"), out, inn)
                .map_err(|e| format!("{e:?}"))?;
            let rank = a
                .ranks
                .as_ref()
                .map_or_else(|| rank_for_bits(out, inn, a.ppm, 32), |r| r[at]);
            let (norm_in, norm_out) = a.norm_dir.as_deref().map_or((None, None), |d| {
                let at = format!("model.layers.{layer}.{}", REF[at]);
                (
                    read_f32(&format!("{d}/{at}.i.f32"), inn),
                    read_f32(&format!("{d}/{at}.o.f32"), out),
                )
            });
            if let Some(d) = a.norm_dir.as_deref() {
                assert!(
                    norm_in.is_some() && norm_out.is_some(),
                    "l{layer} {name}: no reference stats for model.layers.{layer}.{} under {d}",
                    REF[at]
                );
            }
            let i_norm = if a.ones_norm {
                vec![1.0f64; inn]
            } else if let Some(x) = norm_in {
                x
            } else if let Some(d) = a.diag_dir.as_deref() {
                read_stat(d, layer, &a.diag_tag, gram, "diag", inn)?.ok_or_else(|| {
                    format!(
                        "l{layer} {name}: no {d}/g{layer}.{}.{gram}.diag.i64; the onepath fit \
                         refuses rather than falling back to column RMS",
                        a.diag_tag
                    )
                })?
            } else {
                col_rms(&w, out, inn)
            };
            assert_eq!(i_norm.len(), inn, "l{layer} {name}: diag width");
            // The salient columns leave the fit and are stored exactly. Zeroing them before ADMM
            // keeps the activation Gram's energy from collapsing the factor onto a few columns:
            // on `l06 down` 99.85% of `tr(H)` sits in 13 of 12,288 input columns.
            let k = if a.salient_frac > 0.0 {
                ((a.salient_frac * inn as f64).ceil() as usize).min(inn)
            } else {
                0
            };
            let mut mask = vec![true; inn];
            let (mut salient_idx, mut salient_dense) = (Vec::new(), Vec::new());
            if k > 0 {
                let mut order: Vec<usize> = (0..inn).collect();
                order.sort_by(|&x, &y| i_norm[y].total_cmp(&i_norm[x]));
                order.truncate(k);
                order.sort_unstable();
                // The removed columns are served by the salient residual, not by the factor:
                // `y += Σ_j sal[·, j]·x[idx[j]]` (`src/engine/kern/nq.rs`). Record the exact
                // values before zeroing and force `s_pre` to zero so the factor contributes none.
                salient_idx = order.iter().map(|&j| j as u32).collect();
                salient_dense = vec![0.0f64; out * k];
                for i in 0..out {
                    for (at, &j) in order.iter().enumerate() {
                        salient_dense[i * k + at] = from_grid(w[i * inn + j]);
                        w[i * inn + j] = 0;
                    }
                }
                for &j in &order {
                    mask[j] = false;
                }
            }
            let o_norm = if a.ones_norm {
                vec![1.0f64; out]
            } else if let Some(x) = norm_out {
                x
            } else if let Some(d) = a.o_norm_dir.as_deref() {
                read_stat(d, layer, &a.diag_tag, name, "o_norm", out)?.ok_or_else(|| {
                    format!(
                        "l{layer} {name}: no {d}/g{layer}.{}.{name}.o_norm.i64; the onepath fit \
                         refuses rather than weighting every output row equally",
                        a.diag_tag
                    )
                })?
            } else {
                vec![1.0f64; out]
            };
            if let Some(dir) = a.dump.as_deref() {
                std::fs::create_dir_all(dir)?;
                let mut wb = Vec::with_capacity(w.len() * 4);
                for &word in &w {
                    wb.extend_from_slice(&(from_grid(word) as f32).to_le_bytes());
                }
                std::fs::write(format!("{dir}/w.f32"), &wb)?;
                let mut ib = Vec::with_capacity(inn * 8);
                for &x in &i_norm {
                    ib.extend_from_slice(&x.to_le_bytes());
                }
                std::fs::write(format!("{dir}/i_norm.f64"), &ib)?;
                let mut ob = Vec::with_capacity(out * 8);
                for &x in &o_norm {
                    ob.extend_from_slice(&x.to_le_bytes());
                }
                std::fs::write(format!("{dir}/o_norm.f64"), &ob)?;
                std::fs::write(
                    format!("{dir}/shape.txt"),
                    format!("{out} {inn} {rank} {}\n", a.iters),
                )?;
                println!("dump {dir}: l{layer:02} {name} {out}x{inn} r {rank}");
                return Ok(());
            }
            let mut c = NqConfig {
                rank,
                outer_iters: a.iters,
                ..NqConfig::reference(rank)
            };
            if let Some(reg) = a.reg {
                c.reg = reg;
            }
            c.seed = a.seed;
            c.schedule = a.schedule;
            let t = Instant::now();
            let mut factor =
                factorize(&w, out, inn, &i_norm, &o_norm, &c).map_err(|e| format!("{e:?}"))?;
            if k > 0 {
                for &j in &salient_idx {
                    factor.s_pre[j as usize] = 0.0;
                }
            }
            // The rate dial's upper range: the same fit spends its extra bits on the dense sign
            // residual, and the sparse values absorb both terms at the salient columns.
            let residual = if a.residual > 0 {
                let that = served_weight(&factor)?;
                dense_residual(
                    &w,
                    &that,
                    &salient_dense,
                    &salient_idx,
                    out,
                    inn,
                    a.residual,
                )
                .map_err(|e| format!("{e:?}"))?
            } else {
                NqResidual {
                    res: Vec::new(),
                    s_res: Vec::new(),
                    salient: salient_dense.clone(),
                    rel: 0.0,
                }
            };
            let wall = t.elapsed().as_secs_f64();
            let rel = if a.rel {
                let one = vec![1.0f64; rank];
                format!(
                    " relF {:.5} relF_one {:.5}",
                    rel_fro(&factor, &w, &mask)?,
                    rel_fro_mid(&factor, &w, &mask, &one)?
                )
            } else {
                String::new()
            };
            let bits = 32
                * (out * rank.div_ceil(32)
                    + rank * inn.div_ceil(32)
                    + a.residual * out * inn.div_ceil(32)
                    + salient_idx.len())
                + 16 * (inn + rank + out + out * salient_idx.len() + a.residual * out);
            println!(
                "l{layer:02} {name:4} {out}x{inn} r {rank} k {k} res {} b/w {:.5} {wall:.2} s \
                 admm_loss {:.5} relR {:.5}{rel}",
                a.residual,
                bits as f64 / (out * inn) as f64,
                factor.admm_loss,
                residual.rel
            );
            linears.push(packed_linear(layer, at, &factor, &salient_idx, &residual));
        }
        if let Some(path) = a.out.as_deref() {
            checkpoint(path, cfg.n_layer, &linears)?;
        }
    }
    println!(
        "fit {} linears in {:.1} s",
        linears.len(),
        t_all.elapsed().as_secs_f64()
    );
    if let Some(path) = &a.out {
        checkpoint(path, cfg.n_layer, &linears)?;
    }
    Ok(())
}

/// Write the accumulated pack atomically. Called after every layer so a fit that fails late keeps
/// the finished sections instead of losing the whole run to one refused linear.
///
/// # Errors
///
/// Any [`NqPack::new`] ordering error or [`write`] layout error, and the filesystem's.
fn checkpoint(path: &str, layers: usize, linears: &[PackedLinear]) -> Result<(), Box<dyn Error>> {
    let pack = NqPack::new(layers, linears.to_vec())?;
    let bytes = write(&pack)?;
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)?;
    println!(
        "pack {path} {} B, {} sections, {:.6} b/w",
        bytes.len(),
        pack.linears.len(),
        pack.bits_per_weight().unwrap_or(0.0)
    );
    Ok(())
}
