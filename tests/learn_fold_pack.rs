//! `tests/learn_fold_pack.rs` — the fold pack: the student's `E8 ∪ O8` planes written by
//! `fold_pack_of_planes`, read back by `fold_plane_coded`, their weight bit-identical, their rate
//! counted from the bytes on disk, and a one-bit code flip that the round trip must see (red) and
//! its restore (green). The real `Qwen2.5-0.5B` checkpoint measurement is the ignored
//! `fold_pack_real_checkpoint`. Grade `tested`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the fixture indexes `o/n/g/b` planes the way the writer does, and its deterministic \
              LCG maps a `u64` into a small code range by remainder"
)]

use std::fs;

use apeiron::format::packing::{fold_planes_read, fold_planes_write, FoldPlane};
use apeiron::learn::fold::finish::plane::CodedPlane;
use apeiron::learn::fold::fit::icm::{codebook, BLOCK};
use apeiron::learn::fold::fit::quant::r_level;
use apeiron::learn::fold::fit::steps::GROUP;
use apeiron::learn::fold::layer::tensor::SinkPlane;
use apeiron::learn::fold::run::transfer::{
    checkpoint_present, fold_pack_of_planes, fold_pack_read, fold_pack_write, fold_plane_coded,
    load, FoldPackStats,
};
use apeiron::learn::onebit_bits::{bits_corr, params, Cfg};

/// `ONE`, the Q32.32 unit.
const ONE: i64 = 1 << 32;

/// A deterministic xorshift64.
fn next(seed: &mut u64) -> u64 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    x
}

/// A column-scale field that is a fixed point of the writer's own quantizer at `bits`: `n` words on
/// the `2^bits`-point grid between two binary16 endpoints, in file order. The endpoint words are
/// the field's own minimum and maximum, as `quant_c_q32` rounds them from the extremes.
fn column_field(n: usize, c_min: i64, c_max: i64, bits: u32, seed: &mut u64) -> Vec<i64> {
    let span = i128::from(c_max - c_min);
    let levels = (1_i128 << bits) - 1;
    let word = |q: i128| {
        i64::try_from(i128::from(c_min) + (q * span + levels / 2) / levels).expect("grid word")
    };
    let mut out = Vec::with_capacity(n);
    out.push(word(0));
    out.push(word(levels));
    for _ in 2..n {
        let q = i128::from(next(seed) % u64::try_from(levels).expect("levels"));
        out.push(word(q));
    }
    out
}

/// A production-shape `E8 ∪ O8` plane (`896 × 896`) whose rows take 64 distinct `(lo, sp)` grids, so
/// the pack's dictionary path is the 1-byte index the fitted planes use. Returns the sink plane, the
/// codebook index of every 8-block, and the grids the rows were built on.
fn fitted_plane() -> (SinkPlane, Vec<u8>, Vec<i64>, Vec<i64>) {
    let (o, n) = (896_usize, 896_usize);
    let (g, nb) = (n / GROUP, n / BLOCK);
    let cb = codebook();
    let mut idx = Vec::with_capacity(o * nb);
    let mut s_rot = Vec::with_capacity(o * n);
    for b in 0..o * nb {
        let j = (b * 37 + 11) % 256;
        idx.push(u8::try_from(j).unwrap());
        s_rot.extend_from_slice(&cb[j]);
    }

    let mut grids = Vec::with_capacity(64);
    for t in 0..64_i64 {
        grids.push((ONE / 4 + t * (ONE / 512), ONE / 3 + t * (ONE / 1024)));
    }
    let mut seed = 0x5eed_1234_u64;
    let mut rh = Vec::with_capacity(o * g);
    for i in 0..o {
        let (lo, sp) = grids[i % 64];
        for _ in 0..g {
            let k = u8::try_from(next(&mut seed) % 16).unwrap();
            rh.push(r_level(lo, sp, k).expect("grid level"));
        }
    }

    let (lo, sp): (Vec<i64>, Vec<i64>) = (0..o).map(|i| grids[i % 64]).unzip();
    let ch = column_field(n, ONE / 4, ONE / 4 + ONE / 2, 6, &mut seed);
    let dh = column_field(n, ONE / 8, ONE / 8 + ONE / 4, 5, &mut seed);
    let plane = SinkPlane {
        s_rot,
        rh,
        ch,
        dh,
        rows: None,
        dense: None,
        factors: None,
    };
    (plane, idx, lo, sp)
}

/// The plane the fit recomposes: the same code field, grids and scales with zero latents.
fn reference(p: &SinkPlane, idx: Vec<u8>, lo: Vec<i64>, sp: Vec<i64>) -> CodedPlane {
    let (o, n) = (p.rh.len() / (p.ch.len() / GROUP), p.ch.len());
    CodedPlane {
        o,
        n,
        idx,
        s_fix: None,
        rows: p.rows.clone(),
        lo,
        sp,
        r0: p.rh.clone(),
        c0: p.ch.clone(),
        d0: p.dh.clone(),
        pa: vec![0_i64; o * (n / GROUP)],
        pb: vec![0_i64; n],
        pd: vec![0_i64; n],
    }
}

/// The writer's bytes for one plane, its rate, and the loader's weight: bit-identical to the fitted
/// plane, at or under `1.18` bits/weight counted from the file, and a flipped code byte changes the
/// weight while the restore returns it (the negative control).
#[test]
fn fold_pack_is_lossless_under_118_and_sees_a_code_bit_flip() {
    let (plane, idx, lo, sp) = fitted_plane();
    let want = reference(&plane, idx, lo, sp).weight();

    let (records, stats): (Vec<FoldPlane>, FoldPackStats) =
        fold_pack_of_planes(&[&plane]).expect("pack the fitted plane");
    assert_eq!(stats.planes, 1);
    assert_eq!(stats.skipped_dense, 0);
    assert_eq!(stats.weights as usize, plane.s_rot.len());

    let bytes = fold_planes_write(&records).expect("write the layout");
    assert_eq!(stats.bytes as usize, bytes.len());
    let milli = stats.bits_per_weight_milli().expect("rate");
    eprintln!(
        "fold pack: {} planes, {} bytes, {} weights, {}.{:03} b/w",
        stats.planes,
        stats.bytes,
        stats.weights,
        milli / 1000,
        milli % 1000
    );
    assert!(
        milli <= 1180,
        "pack rate {}.{:03} b/w exceeds 1.18",
        milli / 1000,
        milli % 1000
    );

    let path = "/tmp/apeiron/pack118-synthetic.pack";
    fs::write(path, &bytes).expect("write pack file");
    let back = fold_pack_read(path).expect("read the pack");
    assert_eq!(back, records, "records round-trip");
    let got = fold_plane_coded(&back[0]).expect("coded plane").weight();
    assert_eq!(got, want, "loaded weight is the fitted weight");

    // The control: one bit of the first codebook index. The two codebook rows differ, so the
    // loaded weight must differ; restoring the byte must return it.
    let code_at = 8 + 4 + 4 + 4 + 1;
    assert_eq!(bytes[code_at], records[0].idx[0]);
    let flipped = bytes[code_at] ^ 1;
    assert_ne!(
        codebook()[usize::from(flipped)],
        codebook()[usize::from(bytes[code_at])],
        "the flipped code must name a different block"
    );
    let mut red = bytes.clone();
    red[code_at] = flipped;
    let red_record = fold_planes_read(&red).expect("a flipped code is still a record");
    let red_weight = fold_plane_coded(&red_record[0]).expect("coded").weight();
    assert_ne!(
        red_weight, want,
        "a flipped code bit must change the weight"
    );
    let mut green = red.clone();
    green[code_at] = bytes[code_at];
    assert_eq!(&green, &bytes, "restore");
}

/// The real checkpoint's planes through the writer: the measured bytes/weight, and every stored
/// block, row scale and column scale decoded back to the fit's own word.
///
/// The held-out number is not measurable here: the checkpoint holds 4 of the model's 24 layers and
/// the engine has no fold-pack kernel, so this reports the pack's rate on the fitted weights, not a
/// model quality result.
#[test]
#[ignore = "reads the 4-layer resumable transfer checkpoint; --release --ignored --nocapture"]
fn fold_pack_real_checkpoint() {
    const CKPT: &str = "~/q-tr1/ckpt";
    let dir = std::env::var("APEIRON_TRANSFER_DIR").unwrap_or_else(|_| CKPT.to_string());
    assert!(
        checkpoint_present(&dir),
        "blocker: no transfer checkpoint at {dir}"
    );
    let st = load(&dir).expect("load the checkpoint");
    let cb = codebook();
    let mut total = FoldPackStats {
        bytes: 0,
        weights: 0,
        planes: 0,
        skipped_dense: 0,
    };
    let mut records_all: Vec<FoldPlane> = Vec::new();
    let (mut grid_all, mut head_all) = (0u64, 0u64);
    for (l, fits) in st.fits.iter().enumerate() {
        let planes: Vec<&SinkPlane> = fits.iter().flat_map(|f| f.planes.iter()).collect();
        let (records, stats) = fold_pack_of_planes(&planes).expect("pack layer");
        for (rec, p) in records
            .iter()
            .zip(planes.iter().filter(|p| p.dense.is_none()))
        {
            for (b, blk) in p.s_rot.chunks_exact(BLOCK).enumerate() {
                assert_eq!(
                    cb[usize::from(rec.idx[b])].as_slice(),
                    blk,
                    "layer {l} block {b}"
                );
            }
            let coded = fold_plane_coded(rec).expect("coded plane");
            assert_eq!(coded.r0, p.rh, "layer {l} row scales");
            assert_eq!(coded.c0, p.ch, "layer {l} column scales");
            assert_eq!(coded.d0, p.dh, "layer {l} Hadamard scales");
        }
        eprintln!(
            "layer {l}: {} planes, {} bytes, {} weights, {}.{:03} b/w, {} dense skipped",
            stats.planes,
            stats.bytes,
            stats.weights,
            stats.bits_per_weight_milli().unwrap_or(0) / 1000,
            stats.bits_per_weight_milli().unwrap_or(0) % 1000,
            stats.skipped_dense
        );
        let mut grid = 0usize;
        let mut head = 0usize;
        for r in &records {
            let mut pairs: Vec<(i64, i64)> =
                r.lo.iter().copied().zip(r.sp.iter().copied()).collect();
            pairs.sort_unstable();
            pairs.dedup();
            let at = if pairs.len() <= 255 { r.o } else { 2 * r.o };
            grid += 16 * pairs.len() + at;
            head += r.o + 8;
        }
        eprintln!(
            "layer {l}: row grid {grid} bytes as a dictionary, {head} as the carried RHier header"
        );
        grid_all += u64::try_from(grid).expect("grid bytes");
        head_all += u64::try_from(head).expect("header bytes");
        total.bytes += stats.bytes;
        total.weights += stats.weights;
        total.planes += stats.planes;
        total.skipped_dense += stats.skipped_dense;
        records_all.extend(records);
    }
    let path = "/tmp/apeiron/pack118.pack";
    let written = fold_pack_write(&st.fits, path).expect("write the pack");
    assert_eq!(written.planes, records_all.len());
    assert_eq!(written.weights, total.weights);
    assert_eq!(written.skipped_dense, total.skipped_dense);
    let total = FoldPackStats {
        bytes: written.bytes,
        ..total
    };
    let milli = total.bits_per_weight_milli().expect("rate");
    let cfg = Cfg::qwen25_05b_nb();
    let body = params(&cfg).expect("body params");
    let corr = bits_corr(&cfg).expect("corrector bits");
    let layers = u64::try_from(st.fits.len()).expect("layers");
    let per_layer = total.bytes / layers;
    let projected = per_layer * 8 * cfg.layers;
    let body_rate = (projected * 1000 + body / 2) / body;
    let with_corr = ((projected + corr) * 1000 + body / 2) / body;
    eprintln!(
        "total: {} layers, {} planes, {} bytes, {} weights, {}.{:03} b/w over packed weights, \
         {} dense skipped",
        st.fits.len(),
        total.planes,
        total.bytes,
        total.weights,
        milli / 1000,
        milli % 1000,
        total.skipped_dense
    );
    eprintln!(
        "model: 24 uniform layers would store {} bytes = {} bits over {} body params = \
         {}.{:03} b/w, {}.{:03} with the corrector ({corr} bits)",
        projected / 8,
        projected,
        body,
        body_rate / 1000,
        body_rate % 1000,
        with_corr / 1000,
        with_corr % 1000
    );
    // A projection, not a measurement: the byte counts are the ones those layers wrote, less the
    // row-grid dictionaries they wrote and plus the header those same rows would carry
    // (`o + 8` bytes a plane). The measured rate above stays the stored file's own.
    let carried = total.bytes - grid_all + head_all;
    let projected = (carried / layers) * 8 * cfg.layers;
    let carried_rate = (projected * 1000 + body / 2) / body;
    eprintln!(
        "projected with the carried header: row grid {grid_all} -> {head_all} bytes, {} over 4 \
         layers = {} bits over {body} body params = {}.{:03} b/w",
        carried,
        projected,
        carried_rate / 1000,
        carried_rate % 1000
    );
    let back = fold_pack_read(path).expect("read the pack");
    assert_eq!(back, records_all, "records round-trip through the file");
}
