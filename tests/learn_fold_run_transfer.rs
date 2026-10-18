//! `tests/learn_fold_run_transfer.rs` — the resumable model-level transfer run: the one-shot /
//! checkpoint-resumed equivalence, the checkpoint's digest refusal, the student-`GGUF` identity on
//! the real `Qwen2.5-0.5B` F16 teacher, and the layer-by-layer transfer run. Grade `tested`.
//!
//! The synthetic fixture is `tests/learn_fold_driver.rs`'s small model (three layers):
//! `hidden = 128`, `n_head = 2`, `head_dim = 64`, `n_kv_head = 1`, `inter = 128`, `t = 8`,
//! `n_seq = 3`, `slab = 2`, `qmlp_rows = 16`.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unreadable_literal,
    reason = "the fixed-seed LCG maps integers to `f64` by division, the fixture compares driver \
              outputs bit-for-bit by construction, and the `q/k/v/o/i/j/l` names are the driver's \
              tensor vocabulary"
)]

use std::fs;
use std::time::Instant;

use apeiron::crypto::sha256::sha256;
use apeiron::engine::{Engine, KernelMode};
use apeiron::learn::fold::finish::train::{assemble, Frozen};
use apeiron::learn::fold::fit::corrector::Factors;
use apeiron::learn::fold::fit::icm::{codebook, BLOCK};
use apeiron::learn::fold::fit::quant::{quant_c_u8, quant_r_hier};
use apeiron::learn::fold::fit::steps::GROUP;
use apeiron::learn::fold::layer::driver::{
    layer_words, model_finish, model_init, model_step, model_step_stages, quantise_model,
    DriverCfg, ModelState,
};
use apeiron::learn::fold::layer::tensor::{SinkPlane, TensorBits, TensorFit, TensorKind};
use apeiron::learn::fold::run::convert::{student_to_grad, teacher_from_grad, Teacher};
use apeiron::learn::fold::run::finish::{finish_model, planes_from_fits, tunables_from_weights};
use apeiron::learn::fold::run::npy::{chunks, read_npy_i4, split};
use apeiron::learn::fold::run::transfer::{
    checkpoint_present, h0_from_chunks, load, save, write_student_gguf, TransferError,
};
use apeiron::learn::fold::stats::parts::{parts, LayerCfg, RopeTables};
use apeiron::learn::grad::model::{Config, LayerWeights, Weights};
use apeiron::learn::solve::q32::{to_grid, GRID_BOUND};

const HID: usize = 128;
const N_HEAD: usize = 2;
const HEAD_DIM: usize = 64;
const N_KV: usize = 1;
const INTER: usize = 128;
const T: usize = 8;
const N_SEQ: usize = 3;
const SLAB: usize = 2;
const QMLP: usize = 16;
const EPS: f64 = 1e-6;
const THETA: f64 = 1e4;

/// The real `Qwen2.5-0.5B` F16 teacher.
const MODEL: &str = "~/models/gguf/qwen05b-f16.gguf";

/// The C4 calibration tokens (`kl_distill.train_ids`, disjoint from the `seal/calib-c4.txt` shard).
const TOKENS: &str = "~/lanes/w18/kd/c4data/train-c4.qwen.npy";

/// Deterministic xorshift64 LCG.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[lo, hi)` from the top 53 bits.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        lo + (hi - lo) * u
    }
}

/// A standard-normal sample (sum of twelve uniforms minus six).
fn gauss(rng: &mut Lcg) -> f64 {
    (0..12).map(|_| rng.uniform(0.0, 1.0)).sum::<f64>() - 6.0
}

/// `x` as a `Q32.32` word.
fn q(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("on grid")
}

/// A slice as words.
fn qv(x: &[f64]) -> Vec<i64> {
    x.iter().map(|&v| q(v)).collect()
}

fn rand_mat(o: usize, n: usize, scale: f64, rng: &mut Lcg) -> Vec<f64> {
    (0..o * n).map(|_| scale * gauss(rng)).collect()
}

fn rand_bias(n: usize, rng: &mut Lcg) -> Vec<f64> {
    (0..n).map(|_| 0.05 * gauss(rng)).collect()
}

fn rand_ln(n: usize, rng: &mut Lcg) -> Vec<f64> {
    (0..n).map(|_| 1.0 + 0.02 * gauss(rng)).collect()
}

fn layer_cfg() -> LayerCfg {
    LayerCfg {
        n_head: N_HEAD,
        n_kv_head: N_KV,
        head_dim: HEAD_DIM,
        hidden: HID,
        inter: INTER,
        eps: EPS,
        theta: THETA,
    }
}

/// The production driver configuration at the test geometry, at the minimal fit step count: these
/// tests compare two runs of the same fit (one-shot against checkpoint-resumed), so a converged fit
/// buys nothing and the full pass would put the binary over the gate's 30 s budget. `fold_fit`'s own
/// test carries the refit pass.
fn driver_cfg() -> DriverCfg {
    let mut cfg = DriverCfg::production(layer_cfg(), T, N_SEQ);
    cfg.slab = SLAB;
    cfg.qmlp_rows = QMLP;
    cfg.fit.inner_iters = 0;
    cfg.fit.sweeps = 1;
    cfg.fit.e8_beam = 1;
    cfg.nb_sweeps = 1;
    cfg.nb_rounds = 1;
    cfg
}

fn make_layer(cfg: &LayerCfg, rng: &mut Lcg) -> LayerWeights {
    let qo = cfg.n_head * cfg.head_dim;
    let kvo = cfg.n_kv_head * cfg.head_dim;
    let s = 1.0 / (cfg.hidden as f64).sqrt();
    LayerWeights {
        attn_norm: qv(&rand_ln(cfg.hidden, rng)),
        ffn_norm: qv(&rand_ln(cfg.hidden, rng)),
        wq: qv(&rand_mat(qo, cfg.hidden, s, rng)),
        bq: Some(qv(&rand_bias(qo, rng))),
        wk: qv(&rand_mat(kvo, cfg.hidden, s, rng)),
        bk: Some(qv(&rand_bias(kvo, rng))),
        wv: qv(&rand_mat(kvo, cfg.hidden, s, rng)),
        bv: Some(qv(&rand_bias(kvo, rng))),
        wo: qv(&rand_mat(cfg.hidden, qo, s, rng)),
        w_gate: qv(&rand_mat(cfg.inter, cfg.hidden, s, rng)),
        w_up: qv(&rand_mat(cfg.inter, cfg.hidden, s, rng)),
        w_down: qv(&rand_mat(cfg.hidden, cfg.inter, s, rng)),
    }
}

fn rand_hidden(rows: usize, hid: usize, scale: f64, rng: &mut Lcg) -> Vec<i64> {
    (0..rows * hid).map(|_| q(scale * gauss(rng))).collect()
}

/// A scratch directory under the lane's disk, never the tmpfs.
fn scratch(name: &str) -> String {
    let root = std::env::var("APEIRON_TRANSFER_TEST_DIR")
        .unwrap_or_else(|_| "~/q-tr1/transfer-test".to_string());
    let dir = format!("{root}/{name}-{}", std::process::id());
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// The three-layer fixture: config, `RoPE` tables, the teacher layers, the final norm, `h0` and the
/// tied embedding.
fn model_fixture() -> (
    DriverCfg,
    RopeTables,
    Vec<LayerWeights>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    usize,
) {
    let cfg = driver_cfg();
    let rope = RopeTables::new(&cfg.layer, T).expect("rope");
    let mut rng = Lcg::new(31);
    let layers = vec![
        make_layer(&cfg.layer, &mut rng),
        make_layer(&cfg.layer, &mut rng),
        make_layer(&cfg.layer, &mut rng),
    ];
    let fnorm: Vec<i64> = (0..HID)
        .map(|i| q(if i % 2 == 0 { 0.5 } else { 2.0 }))
        .collect();
    let h0 = rand_hidden(N_SEQ * T, HID, 1.0, &mut rng);
    let vocab = 7;
    let embed = rand_hidden(vocab, HID, 0.5, &mut rng);
    (cfg, rope, layers, fnorm, h0, embed, vocab)
}

/// Assert two layers are word-identical, field by field.
fn assert_layer_bits(got: &LayerWeights, want: &LayerWeights, l: usize) {
    for (name, a, b) in [
        ("attn_norm", &got.attn_norm, &want.attn_norm),
        ("ffn_norm", &got.ffn_norm, &want.ffn_norm),
        ("wq", &got.wq, &want.wq),
        ("wk", &got.wk, &want.wk),
        ("wv", &got.wv, &want.wv),
        ("wo", &got.wo, &want.wo),
        ("w_gate", &got.w_gate, &want.w_gate),
        ("w_up", &got.w_up, &want.w_up),
        ("w_down", &got.w_down, &want.w_down),
    ] {
        assert_eq!(a, b, "layer {l} {name} words");
    }
    for (name, a, b) in [
        ("bq", &got.bq, &want.bq),
        ("bk", &got.bk, &want.bk),
        ("bv", &got.bv, &want.bv),
    ] {
        assert_eq!(a, b, "layer {l} {name} words");
    }
}

/// The bit code of a [`TensorKind`], in the checkpoint's `q, k, v, o, gate, up, down` order.
fn kind_code(k: TensorKind) -> u64 {
    match k {
        TensorKind::Q => 0,
        TensorKind::K => 1,
        TensorKind::V => 2,
        TensorKind::O => 3,
        TensorKind::Gate => 4,
        TensorKind::Up => 5,
        TensorKind::Down => 6,
    }
}

/// FNV-1a over every fit field the checkpoint format carries: kind, bits, planes (their four
/// buffers, `rows`, `dense`), `top_rows` and `db`. `wp`, `nb` and `factors` are not in the
/// `fits-<l>.bin` form (the finisher rebuilds `wp` from the planes), so they are not part of the
/// state a `save`/`load` round trip defines.
fn fits_digest(fits: &[Vec<TensorFit>]) -> u64 {
    let mut d = fnv1a(0xcbf2_9ce4_8422_2325, b"fits");
    d = fnv1a(d, &u64::try_from(fits.len()).expect("fits").to_le_bytes());
    for layer in fits {
        d = fnv1a(d, &u64::try_from(layer.len()).expect("fits").to_le_bytes());
        for f in layer {
            d = fnv1a(d, &kind_code(f.kind).to_le_bytes());
            for b in [f.bits.nsign, f.bits.nr, f.bits.nc, f.bits.nd, f.bits.ncorr] {
                d = fnv1a(d, &b.to_le_bytes());
            }
            d = fnv1a(
                d,
                &u64::try_from(f.planes.len()).expect("planes").to_le_bytes(),
            );
            for p in &f.planes {
                for v in [&p.s_rot, &p.rh, &p.ch, &p.dh] {
                    d = fnv_buf(d, v);
                }
                d = fnv1a(
                    d,
                    &[u8::from(p.rows.is_some()), u8::from(p.dense.is_some())],
                );
                if let Some(rows) = &p.rows {
                    d = fnv1a(d, &u64::try_from(rows.len()).expect("rows").to_le_bytes());
                    for &i in rows {
                        d = fnv1a(d, &u64::try_from(i).expect("row").to_le_bytes());
                    }
                }
                if let Some(dense) = &p.dense {
                    d = fnv_buf(d, dense);
                }
            }
            d = fnv1a(
                d,
                &[u8::from(f.top_rows.is_some()), u8::from(f.db.is_some())],
            );
            if let Some(rows) = &f.top_rows {
                d = fnv1a(d, &u64::try_from(rows.len()).expect("top").to_le_bytes());
                for &i in rows {
                    d = fnv1a(d, &u64::try_from(i).expect("top").to_le_bytes());
                }
            }
            if let Some(db) = &f.db {
                d = fnv_buf(d, db);
            }
        }
    }
    d
}

/// Assert two fit lists agree on every field the checkpoint carries, and on their digest.
fn assert_fits_persisted(got: &[Vec<TensorFit>], want: &[Vec<TensorFit>], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: layer count");
    for (l, (gl, wl)) in got.iter().zip(want).enumerate() {
        assert_eq!(gl.len(), wl.len(), "{what}: layer {l} fit count");
        for (a, b) in gl.iter().zip(wl) {
            assert_eq!(a.kind, b.kind, "{what}: layer {l} kind");
            assert_eq!(a.bits, b.bits, "{what}: layer {l} bits");
            assert_eq!(a.top_rows, b.top_rows, "{what}: layer {l} top_rows");
            assert_eq!(a.db, b.db, "{what}: layer {l} db");
            assert!(a.wp.is_empty(), "{what}: wp is not in the fits form");
            assert!(a.nb.is_none(), "{what}: nb is not in the fits form");
            assert_eq!(a.planes.len(), b.planes.len(), "{what}: layer {l} planes");
            for (pa, pb) in a.planes.iter().zip(&b.planes) {
                assert_eq!(pa.s_rot, pb.s_rot, "{what}: layer {l} s_rot");
                assert_eq!(pa.rh, pb.rh, "{what}: layer {l} rh");
                assert_eq!(pa.ch, pb.ch, "{what}: layer {l} ch");
                assert_eq!(pa.dh, pb.dh, "{what}: layer {l} dh");
                assert_eq!(pa.rows, pb.rows, "{what}: layer {l} rows");
                assert_eq!(pa.dense, pb.dense, "{what}: layer {l} dense");
            }
        }
    }
    assert_eq!(fits_digest(got), fits_digest(want), "{what}: fits digest");
}

/// Assert a loaded state equals the state it was saved from on the persisted projection: the next
/// index, both hidden states, the bit sums, every layer word, the fits' checkpoint fields and, when
/// set, the in-progress layer's stage, `lq`, fits, bits and `hs_mid`.
fn assert_state_persisted(got: &ModelState, want: &ModelState, what: &str) {
    assert_eq!(got.next, want.next, "{what}: next");
    assert_eq!(got.ht, want.ht, "{what}: ht");
    assert_eq!(got.hs, want.hs, "{what}: hs");
    assert_eq!(got.bits, want.bits, "{what}: bits");
    assert_eq!(got.layers.len(), want.layers.len(), "{what}: layers");
    for (l, (a, b)) in got.layers.iter().zip(&want.layers).enumerate() {
        assert_layer_bits(a, b, l);
    }
    assert_fits_persisted(&got.fits, &want.fits, what);
    match (&got.partial, &want.partial) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            assert_eq!(a.stage, b.stage, "{what}: progress stage");
            assert_eq!(a.bits, b.bits, "{what}: progress bits");
            assert_eq!(a.hs_mid, b.hs_mid, "{what}: progress hs_mid");
            assert_layer_bits(&a.lq, &b.lq, want.next);
            assert_fits_persisted(
                std::slice::from_ref(&a.fits),
                std::slice::from_ref(&b.fits),
                what,
            );
        }
        _ => panic!("{what}: progress presence differs"),
    }
}

/// [1] resume equivalence: one-shot `quantise_model` against `model_step(0..1)` → `save` → `load`
/// → `model_step(1..3)` → `model_finish`: every output tensor and `fnorm` bit-identical.
#[test]
fn resume_equivalence() {
    let t0 = Instant::now();
    let (cfg, rope, layers, fnorm, h0, embed, vocab) = model_fixture();
    let lr = q(0.15);

    let one = quantise_model(&cfg, &rope, &layers, &fnorm, &embed, vocab, &h0, 3, lr)
        .expect("one-shot model");
    assert_eq!(one.layers.len(), 3);
    assert_eq!(one.fits.len(), 3);

    let dir = scratch("resume");
    let mut st = model_init(&h0);
    model_step(&cfg, &rope, &layers, &fnorm, &mut st, 1).expect("layer 0");
    assert_eq!(st.next, 1);
    let bits_after_0 = st.bits;
    save(&dir, &st).expect("save");
    assert!(checkpoint_present(&dir), "manifest not written");

    let mut resumed = load(&dir).expect("load");
    assert_eq!(resumed.next, 1);
    assert_eq!(resumed.bits, bits_after_0);
    assert_eq!(resumed.layers.len(), 1);
    assert_layer_bits(&resumed.layers[0], &one.layers[0], 0);
    assert_eq!(resumed.ht, st.ht, "ht round trip");
    assert_eq!(resumed.hs, st.hs, "hs round trip");

    model_step(&cfg, &rope, &layers, &fnorm, &mut resumed, 3).expect("layers 1..3");
    let res = model_finish(&cfg, resumed, &fnorm, &embed, vocab, lr).expect("finish");
    assert_eq!(res.layers.len(), one.layers.len());
    for l in 0..one.layers.len() {
        assert_layer_bits(&res.layers[l], &one.layers[l], l);
    }
    assert_eq!(res.fnorm, one.fnorm, "fnorm words");
    assert_eq!(res.bits, one.bits, "bit budget");
    assert_eq!(res.fnorm.len(), HID);

    fs::remove_dir_all(&dir).expect("clean scratch");
    eprintln!("[1] resume equivalence ok in {:?}", t0.elapsed());
}

/// [2] checkpoint refusal: one flipped byte in `ht.bin` makes `load` refuse on the digest.
#[test]
fn checkpoint_digest_refusal() {
    let (cfg, rope, layers, fnorm, h0, _, _) = model_fixture();
    let dir = scratch("digest");
    let mut st = model_init(&h0);
    model_step(&cfg, &rope, &layers, &fnorm, &mut st, 1).expect("layer 0");
    save(&dir, &st).expect("save");

    let ht = format!("{dir}/ht.bin");
    let mut bytes = fs::read(&ht).expect("read ht");
    bytes[0] ^= 0x40;
    fs::write(&ht, &bytes).expect("write ht");
    let err = load(&dir).expect_err("a flipped byte must refuse");
    match &err {
        TransferError::Digest { file, .. } => assert_eq!(file, "ht.bin"),
        other => panic!("expected a digest refusal, got {other:?}"),
    }

    // Restoring the byte restores the checkpoint.
    bytes[0] ^= 0x40;
    fs::write(&ht, &bytes).expect("restore ht");
    let back = load(&dir).expect("restored checkpoint");
    assert_eq!(back.next, 1);
    assert_layer_bits(&back.layers[0], &st.layers[0], 0);

    fs::remove_dir_all(&dir).expect("clean scratch");
}

/// The byte length of every file in a scratch directory.
fn dir_bytes(dir: &str) -> u64 {
    fs::read_dir(dir)
        .expect("scratch dir")
        .filter_map(Result::ok)
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

/// [1a] stage-granular resume: the model fixture run as one-stage rounds with `save` + `load`
/// between EVERY stage reaches the one-shot `model_step` state, word for word on the populated
/// projection (the fits' `wp`/`nb`/`factors` are not in the checkpoint form), and every pause reads
/// its progress back exactly.
#[test]
fn stage_resume_equivalence() {
    let t0 = Instant::now();
    let (cfg, rope, layers, fnorm, h0, _, _) = model_fixture();
    let stages_per_layer = 4usize;
    let total_stages = stages_per_layer * layers.len();

    let mut one = model_init(&h0);
    model_step(&cfg, &rope, &layers, &fnorm, &mut one, layers.len()).expect("one-shot model");
    assert!(one.partial.is_none(), "a finished model keeps no progress");
    assert_eq!(one.next, layers.len());

    let dir = scratch("stage-resume");
    let mut st = model_init(&h0);
    let mut pauses = 0usize;
    let mut max_pause_bytes = 0usize;
    let mut max_checkpoint_bytes = 0u64;
    let mut io = std::time::Duration::ZERO;
    for k in 0..total_stages {
        model_step_stages(&cfg, &rope, &layers, &fnorm, &mut st, 1).expect("one stage");
        let done = st.next == layers.len() && st.partial.is_none();
        assert_eq!(
            done,
            k + 1 == total_stages,
            "stage {k} finished the model early"
        );
        if done {
            break;
        }
        // The unit of resumption is a stage: every boundary is written and read back.
        let t_io = Instant::now();
        save(&dir, &st).expect("save stage");
        let back = load(&dir).expect("load stage");
        io += t_io.elapsed();
        assert_state_persisted(&back, &st, &format!("stage pause {k}"));
        st = back;
        pauses += 1;
        let bytes: u64 = [
            "progress-lq.bin",
            "progress-fits.bin",
            "progress-hs-mid.bin",
        ]
        .iter()
        .map(|n| {
            fs::metadata(format!("{dir}/{n}"))
                .expect("progress file")
                .len()
        })
        .sum();
        max_pause_bytes = max_pause_bytes.max(usize::try_from(bytes).expect("pause bytes"));
        max_checkpoint_bytes = max_checkpoint_bytes.max(dir_bytes(&dir));
    }
    assert_eq!(pauses, total_stages - 1, "one save/load per stage boundary");

    // The finished state round-trips through the layer-boundary format (no progress line).
    save(&dir, &st).expect("save finished");
    let back = load(&dir).expect("load finished");
    assert!(back.partial.is_none(), "no progress at a layer boundary");
    assert_state_persisted(&back, &st, "finished checkpoint");
    assert_state_persisted(&back, &one, "one-shot state");
    eprintln!(
        "[1a] {total_stages} stages, {pauses} pause round trips, largest pause {max_pause_bytes} \
         bytes, largest checkpoint {max_checkpoint_bytes} bytes, save+load {:.1} ms, one-shot \
         state matched in {:?}",
        io.as_secs_f64() * 1e3,
        t0.elapsed()
    );

    fs::remove_dir_all(&dir).expect("clean scratch");
}

/// [1b] progress refusal: one flipped byte in `progress-hs-mid.bin` makes `load` refuse on the
/// digest, and the restore loads the stage boundary back.
#[test]
fn stage_progress_digest_refusal() {
    let (cfg, rope, layers, fnorm, h0, _, _) = model_fixture();
    let dir = scratch("stage-digest");
    let mut st = model_init(&h0);
    model_step_stages(&cfg, &rope, &layers, &fnorm, &mut st, 1).expect("stage 0");
    assert_eq!(st.partial.as_ref().map(|p| p.stage), Some(1));
    save(&dir, &st).expect("save stage");

    let path = format!("{dir}/progress-hs-mid.bin");
    let mut bytes = fs::read(&path).expect("read progress");
    bytes[0] ^= 0x40;
    fs::write(&path, &bytes).expect("write progress");
    match load(&dir).expect_err("a flipped progress byte must refuse") {
        TransferError::Digest { file, .. } => assert_eq!(file, "progress-hs-mid.bin"),
        other => panic!("expected a digest refusal, got {other:?}"),
    }

    bytes[0] ^= 0x40;
    fs::write(&path, &bytes).expect("restore progress");
    let back = load(&dir).expect("restored stage");
    assert_state_persisted(&back, &st, "restored progress");
    assert_eq!(back.next, 0);
    assert_eq!(back.partial.as_ref().map(|p| p.stage), Some(1));

    fs::remove_dir_all(&dir).expect("clean scratch");
}

/// The finisher fixture geometry (the tiny Qwen2 of `tests/learn_fold_run_finish.rs`): two layers,
/// `n_embd` 128, `n_ff` 256, vocabulary 64, a 16-token batch.
const FE: usize = 128;
const FNH: usize = 2;
const FNKV: usize = 1;
const FHD: usize = 64;
const FFF: usize = 256;
const FV: usize = 64;
const FL: usize = 2;
const FT: usize = 16;
const FEPS: i64 = 42_950;
const FKINDS: usize = 7;

fn finish_cfg() -> Config {
    Config {
        n_layer: FL,
        n_head: FNH,
        n_head_kv: FNKV,
        n_embd: FE,
        n_ff: FFF,
        vocab: FV,
        head_dim: FHD,
        eps: FEPS,
        rope_theta: THETA,
    }
}

/// `(o_total, n)` of linear kind `k`, in `q, k, v, o, gate, up, down` order.
fn finish_dims(k: usize) -> (usize, usize) {
    match k {
        0 => (FNH * FHD, FE),
        1 | 2 => (FNKV * FHD, FE),
        3 => (FE, FNH * FHD),
        4 | 5 => (FFF, FE),
        _ => (FE, FFF),
    }
}

fn finish_kind(k: usize) -> TensorKind {
    match k {
        0 => TensorKind::Q,
        1 => TensorKind::K,
        2 => TensorKind::V,
        3 => TensorKind::O,
        4 => TensorKind::Gate,
        5 => TensorKind::Up,
        _ => TensorKind::Down,
    }
}

/// The exact code field of a list of codebook indices, `o × n` row-major.
fn code_field(idx: &[u8], o: usize, n: usize) -> Vec<i64> {
    let cb = codebook();
    let nb = n / BLOCK;
    let mut s = vec![0_i64; o * n];
    for oi in 0..o {
        for m in 0..nb {
            let j = idx[oi * nb + m] as usize;
            for k in 0..BLOCK {
                s[oi * n + m * BLOCK + k] = cb[j][k];
            }
        }
    }
    s
}

/// One stored pack at the quantizers' fixed points: `(s_rot, rh, ch, dh)`.
fn pack_words(rng: &mut Lcg, o: usize, n: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
    let idx: Vec<u8> = (0..o * (n / BLOCK))
        .map(|_| u8::try_from(rng.next_u64() % 256).expect("codebook index"))
        .collect();
    let s = code_field(&idx, o, n);
    let g = n / GROUP;
    let r_raw: Vec<i64> = (0..o * g)
        .map(|_| q(0.5 + 0.5 * rng.uniform(0.0, 1.0)))
        .collect();
    let rh = quant_r_hier(&r_raw, o, g).expect("row scales").rq;
    let c_raw: Vec<i64> = (0..n)
        .map(|_| q(0.1 + 0.1 * rng.uniform(0.0, 1.0)))
        .collect();
    let ch = quant_c_u8(&c_raw, 6).expect("column scales").0;
    let d_raw: Vec<i64> = (0..n)
        .map(|_| q(0.5 + 0.5 * rng.uniform(0.0, 1.0)))
        .collect();
    let dh = quant_c_u8(&d_raw, 5).expect("hadamard scales").0;
    (s, rh, ch, dh)
}

/// The seven [`TensorFit`]s of one layer, in `q, k, v, o, gate, up, down` order: a pack plane per
/// kind, a rows-less second plane for `gate`/`up` recovered from `top_rows`, a dense corrector (with
/// the unread `Factors`) on `q`/`down`, the `q`/`k`/`v` bias delta and the per-fit `bits`. `wp` is
/// empty: neither the checkpoint nor the finisher reads it.
fn finish_fits(rng: &mut Lcg) -> Vec<TensorFit> {
    let mut out = Vec::with_capacity(FKINDS);
    for k in 0..FKINDS {
        let (o, n) = finish_dims(k);
        let (s_rot, rh, ch, dh) = pack_words(rng, o, n);
        let mut planes = vec![SinkPlane {
            s_rot,
            rh,
            ch,
            dh,
            rows: None,
            dense: None,
            factors: None,
        }];
        let mut top_rows = None;
        if k == 4 || k == 5 {
            let top: Vec<usize> = vec![1, 5, 9, 13];
            let (s2, rh2, ch2, dh2) = pack_words(rng, top.len(), n);
            planes.push(SinkPlane {
                s_rot: s2,
                rh: rh2,
                ch: ch2,
                dh: dh2,
                rows: None,
                dense: None,
                factors: None,
            });
            top_rows = Some(top);
        }
        if k == 0 || k == 6 {
            let dense: Vec<i64> = (0..o * n).map(|_| q(0.02 * gauss(rng))).collect();
            planes.push(SinkPlane {
                s_rot: Vec::new(),
                rh: Vec::new(),
                ch: Vec::new(),
                dh: Vec::new(),
                rows: None,
                dense: Some(dense),
                factors: Some(Factors {
                    u: vec![q(0.1), q(0.2)],
                    v: vec![q(0.3), q(0.4)],
                }),
            });
        }
        let db = if k < 3 {
            Some((0..o).map(|_| q(0.05 * gauss(rng))).collect())
        } else {
            None
        };
        out.push(TensorFit {
            kind: finish_kind(k),
            wp: Vec::new(),
            planes,
            bits: TensorBits {
                nsign: (k + 1) as u64,
                nr: 100 + k as u64,
                nc: 200 + k as u64,
                nd: 300 + k as u64,
                ncorr: 400 + k as u64,
            },
            nb: None,
            top_rows,
            db,
        });
    }
    out
}

/// One student layer at the finisher's widths. The `q`/`k`/`v` biases are present because the
/// checkpoint's layer format requires them; the round trip feeds the tunable build a bias-less copy
/// so the fit's `db` (the field under test) supplies the bias base.
fn finish_student(rng: &mut Lcg) -> LayerWeights {
    let jitter =
        |rng: &mut Lcg, n: usize| -> Vec<i64> { (0..n).map(|_| q(0.01 * gauss(rng))).collect() };
    LayerWeights {
        attn_norm: (0..FE)
            .map(|_| q(0.8 + 0.2 * rng.uniform(0.0, 1.0)))
            .collect(),
        wq: jitter(rng, FNH * FHD * FE),
        bq: Some(jitter(rng, FNH * FHD)),
        wk: jitter(rng, FNKV * FHD * FE),
        bk: Some(jitter(rng, FNKV * FHD)),
        wv: jitter(rng, FNKV * FHD * FE),
        bv: Some(jitter(rng, FNKV * FHD)),
        wo: jitter(rng, FE * FNH * FHD),
        ffn_norm: (0..FE)
            .map(|_| q(0.8 + 0.2 * rng.uniform(0.0, 1.0)))
            .collect(),
        w_gate: jitter(rng, FFF * FE),
        w_up: jitter(rng, FFF * FE),
        w_down: jitter(rng, FE * FFF),
    }
}

/// FNV-1a over a length-prefixed word buffer.
fn fnv_buf(mut h: u64, v: &[i64]) -> u64 {
    h = fnv1a(h, &u64::try_from(v.len()).expect("len").to_le_bytes());
    for &w in v {
        h = fnv1a(h, &w.to_le_bytes());
    }
    h
}

/// FNV-1a over every tensor of a [`Weights`], length-prefixed per buffer.
fn weights_digest(w: &Weights) -> u64 {
    let mut d: u64 = 0xcbf2_9ce4_8422_2325;
    for (name, v) in [
        ("token_embd", &w.token_embd),
        ("output_norm", &w.output_norm),
        ("lm_head", &w.lm_head),
    ] {
        d = fnv1a(d, name.as_bytes());
        d = fnv_buf(d, v);
    }
    for (l, lw) in w.layers.iter().enumerate() {
        d = fnv1a(d, &u64::try_from(l).expect("layer").to_le_bytes());
        for (name, v) in [
            ("attn_norm", &lw.attn_norm),
            ("ffn_norm", &lw.ffn_norm),
            ("wq", &lw.wq),
            ("wk", &lw.wk),
            ("wv", &lw.wv),
            ("wo", &lw.wo),
            ("w_gate", &lw.w_gate),
            ("w_up", &lw.w_up),
            ("w_down", &lw.w_down),
        ] {
            d = fnv1a(d, name.as_bytes());
            d = fnv_buf(d, v);
        }
        for (name, b) in [("bq", &lw.bq), ("bk", &lw.bk), ("bv", &lw.bv)] {
            d = fnv1a(d, name.as_bytes());
            d = fnv1a(d, &[u8::from(b.is_some())]);
            if let Some(v) = b {
                d = fnv_buf(d, v);
            }
        }
    }
    d
}

/// The digest of the weights [`assemble`] builds from `fits`: the finisher's own path from a fit
/// list to the student's tensors.
fn assembled_digest(
    cfg: &Config,
    fits: &[Vec<TensorFit>],
    layers: &[LayerWeights],
    fnorm: &[i64],
    emb: &[i64],
    head: &[i64],
) -> u64 {
    let planes = planes_from_fits(cfg, fits).expect("planes");
    let tun = tunables_from_weights(layers, fnorm, fits).expect("tunables");
    let frozen = Frozen::new(emb, head);
    let w = assemble(cfg, &planes, &tun, &frozen).expect("assemble");
    weights_digest(&w)
}

/// [2a] fit round trip: `save` then `load` restores the fits so that the finisher assembles the same
/// student `Weights` from the loaded fits as from the original ones.
#[test]
fn checkpoint_round_trips_fits() {
    let t0 = Instant::now();
    let cfg = finish_cfg();
    let mut rng = Lcg::new(0xF17);
    let fits: Vec<Vec<TensorFit>> = (0..FL).map(|_| finish_fits(&mut rng)).collect();
    let layers: Vec<LayerWeights> = (0..FL).map(|_| finish_student(&mut rng)).collect();
    // The tunable build sees no student bias, so the fit's `db` is the bias base.
    let mut tun_layers = layers.clone();
    for lw in &mut tun_layers {
        lw.bq = None;
        lw.bk = None;
        lw.bv = None;
    }
    let fnorm: Vec<i64> = (0..FE)
        .map(|_| q(0.8 + 0.2 * rng.uniform(0.0, 1.0)))
        .collect();
    let emb: Vec<i64> = (0..FV * FE).map(|_| q(0.1 * gauss(&mut rng))).collect();
    let head: Vec<i64> = (0..FV * FE).map(|_| q(0.1 * gauss(&mut rng))).collect();
    let ht = vec![q(0.1); FT * FE];
    let hs = vec![q(-0.1); FT * FE];
    let st = ModelState {
        next: FL,
        ht: ht.clone(),
        hs: hs.clone(),
        layers: layers.clone(),
        bits: TensorBits::default(),
        fits: fits.clone(),
        partial: None,
    };

    let dir = scratch("fits");
    save(&dir, &st).expect("save");
    let back = load(&dir).expect("load");
    assert_eq!(back.next, FL);
    assert_eq!(back.ht, ht, "ht round trip");
    assert_eq!(back.hs, hs, "hs round trip");
    assert_eq!(back.bits, st.bits);
    assert_eq!(back.fits.len(), FL);
    let mut seen_factors = false;
    for (got, want) in back.fits.iter().zip(&fits) {
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(want) {
            seen_factors |= b.planes.iter().any(|p| p.factors.is_some());
            assert_eq!(a.kind, b.kind, "fit kind");
            assert_eq!(a.bits, b.bits, "fit bits");
            assert_eq!(a.top_rows, b.top_rows, "top_rows");
            assert_eq!(a.db, b.db, "db");
            assert!(a.wp.is_empty(), "wp is not persisted");
            assert!(a.nb.is_none(), "nb is not persisted");
            assert_eq!(a.planes.len(), b.planes.len());
            for (pa, pb) in a.planes.iter().zip(&b.planes) {
                assert_eq!(pa.s_rot, pb.s_rot);
                assert_eq!(pa.rh, pb.rh);
                assert_eq!(pa.ch, pb.ch);
                assert_eq!(pa.dh, pb.dh);
                assert_eq!(pa.rows, pb.rows);
                assert_eq!(pa.dense, pb.dense);
                assert!(pa.factors.is_none(), "factors are not persisted");
            }
        }
    }
    assert!(seen_factors, "the fixture must carry a corrector's factors");

    let want = assembled_digest(&cfg, &fits, &tun_layers, &fnorm, &emb, &head);
    let got = assembled_digest(&cfg, &back.fits, &tun_layers, &fnorm, &emb, &head);
    assert_eq!(got, want, "assembled weights digest");
    println!(
        "fits round-trip weights digest {want:#018x} in {:?}",
        t0.elapsed()
    );

    fs::remove_dir_all(&dir).expect("clean scratch");
}

/// [2b] fits refusal: one flipped byte in `fits-0.bin` makes `load` refuse on the digest, and the
/// restore loads the fits back.
#[test]
fn fits_digest_refusal() {
    let (cfg, rope, layers, fnorm, h0, _, _) = model_fixture();
    let dir = scratch("fits-digest");
    let mut st = model_init(&h0);
    model_step(&cfg, &rope, &layers, &fnorm, &mut st, 1).expect("layer 0");
    save(&dir, &st).expect("save");

    let path = format!("{dir}/fits-0.bin");
    let mut bytes = fs::read(&path).expect("read fits");
    bytes[0] ^= 0x40;
    fs::write(&path, &bytes).expect("write fits");
    match load(&dir).expect_err("a flipped fits byte must refuse") {
        TransferError::Digest { file, .. } => assert_eq!(file, "fits-0.bin"),
        other => panic!("expected a digest refusal, got {other:?}"),
    }

    bytes[0] ^= 0x40;
    fs::write(&path, &bytes).expect("restore fits");
    let back = load(&dir).expect("restored checkpoint");
    assert_eq!(back.fits.len(), 1);
    assert_eq!(back.fits[0].len(), st.fits[0].len());
    for (a, b) in back.fits[0].iter().zip(&st.fits[0]) {
        assert_eq!(a.kind, b.kind);
        assert_eq!(a.planes.len(), b.planes.len());
        assert_eq!(a.db, b.db);
    }

    fs::remove_dir_all(&dir).expect("clean scratch");
}

/// [3] `GGUF` identity (ignored, real model): rewrite the teacher's own weights through
/// `write_student_gguf` and compare the 8-token logits and the file digest.
#[test]
#[ignore = "loads the 994 MB F16 Qwen2.5-0.5B and writes a ~1 GB GGUF; --release --ignored --nocapture"]
fn gguf_identity() {
    let t0 = Instant::now();
    let root = std::env::var("APEIRON_GGUF_IDENTITY_DIR")
        .unwrap_or_else(|_| "~/q-tr1/gguf-identity".to_string());
    fs::create_dir_all(&root).expect("identity dir");
    let out = format!("{root}/student-f16.gguf");

    let src = Engine::open(MODEL).expect("open f16 teacher");
    let (w, _cfg) = Weights::from_engine(&src).expect("load teacher weights");
    eprintln!("[3] teacher loaded at {:.1}s", t0.elapsed().as_secs_f64());
    write_student_gguf(MODEL, &w, &out).expect("write student gguf");
    eprintln!("[3] written at {:.1}s", t0.elapsed().as_secs_f64());

    let a = fs::read(MODEL).expect("read teacher file");
    let b = fs::read(&out).expect("read student file");
    let ha = sha256(&a);
    let hb = sha256(&b);
    let same = ha == hb;
    eprintln!(
        "[3] sha256 teacher {} / student {}: {}",
        hex(&ha),
        hex(&hb),
        if same {
            "IDENTICAL (byte-faithful writer)"
        } else {
            "DIFFERENT"
        }
    );
    // The payload must be the teacher's, byte for byte, except where the teacher stores IEEE-754
    // negative zero: `Weights::from_engine` reads `-0.0` as the integer zero, so the re-encode
    // emits `+0.0`. Any other difference is a changed weight — an F16 narrowing that is not the
    // teacher's own value. The logits comparison below cannot see a one-ULP weight change
    // (fixed-point logits are quantised to `2^-32`), so this byte check is the writer's identity
    // witness and the mutation control's target.
    assert_eq!(a.len(), b.len(), "student file length");
    let mut neg_zero = 0usize;
    let mut other = 0usize;
    let mut first: Option<(usize, u8, u8)> = None;
    for (i, (&ta, &tb)) in a.iter().zip(b.iter()).enumerate() {
        if ta == tb {
            continue;
        }
        let word_is_neg_zero =
            i % 2 == 1 && ta == 0x80 && tb == 0x00 && a[i - 1] == 0x00 && b[i - 1] == 0x00;
        if word_is_neg_zero {
            neg_zero += 1;
        } else {
            other += 1;
            if first.is_none() {
                first = Some((i, ta, tb));
            }
        }
    }
    eprintln!(
        "[3] payload: {neg_zero} negative-zero words re-encoded, {other} other differing bytes"
    );
    assert_eq!(
        other, 0,
        "student payload differs from the teacher beyond the negative-zero re-encode; \
         first at offset {first:?}"
    );
    drop(a);
    drop(b);

    let dst = Engine::open(&out).expect("open student gguf");
    let prompt: [u32; 8] = [9707, 11, 1879, 30, 13, 264, 1129, 311];
    let mut sa = src.state(prompt.len() + 4).expect("teacher state");
    let mut sb = dst.state(prompt.len() + 4).expect("student state");
    for (i, &tok) in prompt.iter().enumerate() {
        let la = src
            .step_logits_state(&mut sa, tok, KernelMode::Exact)
            .expect("teacher logits");
        let lb = dst
            .step_logits_state(&mut sb, tok, KernelMode::Exact)
            .expect("student logits");
        assert_eq!(la, lb, "8-token logits differ at token {i}");
    }
    // The head, exercised on a fixed output-normed state.
    let n_embd = i64::try_from(src.config.n_embd).expect("n_embd");
    let x: Vec<i64> = (0..n_embd).map(|i| (i - 448) * 97_321).collect();
    let ha = src.lm_head_fixed(&x).expect("teacher head");
    let hb = dst.lm_head_fixed(&x).expect("student head");
    assert_eq!(ha, hb, "lm_head_fixed logits differ");

    eprintln!("[3] 8-token prefill + lm_head_fixed logits bit-identical");
    eprintln!("[3] ok in {:?}", t0.elapsed());
}

/// [4] `transfer_qwen05b` (ignored): the resumable layer-by-layer transfer on the real 0.5B model,
/// checkpointed under `APEIRON_TRANSFER_DIR` and stopped when the declared budget would overrun.
#[test]
#[ignore = "loads and converts the 0.5B model, quantises one layer per ~10^11 flops; --release --ignored --nocapture"]
fn transfer_qwen05b() {
    let t_test = Instant::now();
    let dir = std::env::var("APEIRON_TRANSFER_DIR")
        .unwrap_or_else(|_| "~/q-tr1/ckpt".to_string());
    let ids_raw = fs::read(TOKENS).expect("read token npy");
    let ids = read_npy_i4(&ids_raw).expect("parse token npy");
    drop(ids_raw);
    let chunks = chunks(&ids, 512).expect("512-token chunks");
    drop(ids);
    let sp = split(chunks.len(), 50, 20, 40).expect("50/20/40 split");
    let cal: Vec<Vec<u32>> = sp.cal.iter().map(|&c| chunks[c].clone()).collect();
    let n_seq = cal.len();
    let t = cal[0].len();

    let engine = Engine::open(MODEL).expect("open teacher");
    let (weights, config) = Weights::from_engine(&engine).expect("load weights");
    let teacher = teacher_from_grad(&weights, &config).expect("teacher");
    eprintln!(
        "[4] teacher loaded at {:.1}s: n_layer {} hidden {} inter {} vocab {}",
        t_test.elapsed().as_secs_f64(),
        teacher.n_layer,
        teacher.layer_cfg.hidden,
        teacher.layer_cfg.inter,
        teacher.vocab
    );
    let cfg = DriverCfg::production(teacher.layer_cfg, t, n_seq);
    let rope = RopeTables::new(&cfg.layer, t).expect("rope");

    let mut st = if checkpoint_present(&dir) {
        let st = load(&dir).expect("resume checkpoint");
        eprintln!(
            "[4] resumed at layer {} of {} after {:.1}s",
            st.next,
            teacher.n_layer,
            t_test.elapsed().as_secs_f64()
        );
        st
    } else {
        let h0 = h0_from_chunks(&teacher, &cal).expect("h0");
        let st = model_init(&h0);
        eprintln!("[4] fresh state, h0 {} entries", h0.len());
        st
    };

    // A round runs whole stages while the elapsed wall plus 1.25x the longest stage seen stays under
    // the budget; the progress of a layer left mid-way is checkpointed and resumed by the next round.
    let budget = 500.0_f64;
    let t_loop = Instant::now();
    let mut stage_max = 0.0_f64;
    while st.next < teacher.n_layer {
        if t_loop.elapsed().as_secs_f64() + 1.25 * stage_max > budget {
            eprintln!(
                "[4] stop: {:.1}s + 1.25 x longest stage {stage_max:.1}s > {budget:.0}s budget",
                t_loop.elapsed().as_secs_f64()
            );
            break;
        }
        let (l, stage0) = (st.next, st.partial.as_ref().map_or(0, |p| p.stage));
        let t_stage = Instant::now();
        model_step_stages(&cfg, &rope, &teacher.layers, &teacher.fnorm, &mut st, 1)
            .expect("stage step");
        let wall = t_stage.elapsed().as_secs_f64();
        stage_max = stage_max.max(wall);
        eprintln!(
            "[4] layer {l} stage {stage0}: wall {wall:.1}s, rss {:.2} GB, bits {} {} {} {} {}",
            rss_gb(),
            st.bits.nsign,
            st.bits.nr,
            st.bits.nc,
            st.bits.nd,
            st.bits.ncorr
        );
    }
    save(&dir, &st).expect("checkpoint");
    eprintln!(
        "[4] loop end at next {} after {:.1}s ({:.1}s total)",
        st.next,
        t_loop.elapsed().as_secs_f64(),
        t_test.elapsed().as_secs_f64()
    );

    if st.next == teacher.n_layer {
        let res = model_finish(
            &cfg,
            st,
            &teacher.fnorm,
            &teacher.embed,
            teacher.vocab,
            q(0.15),
        )
        .expect("finish");
        // The finishing P-step: `APEIRON_FINISH_STEPS` > 0 runs `finish_model` on the driver path
        // (the restored fits included) and writes the finished student; 0 (the default) writes the
        // fitted student alone.
        let steps: u64 = std::env::var("APEIRON_FINISH_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let warmup: u64 = std::env::var("APEIRON_FINISH_WARMUP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let (student, source) = if steps > 0 {
            let fin =
                finish_model(&config, &res, &weights, &cal, steps, warmup).expect("finish_model");
            eprintln!(
                "[4] finish_model {steps} steps (warmup {warmup}): {} loss words, last {:?}",
                fin.loss.len(),
                fin.loss.last()
            );
            (fin.weights, "finished")
        } else {
            (
                student_to_grad(&weights, &res.layers, &res.fnorm).expect("student weights"),
                "fitted",
            )
        };
        let out = format!("{dir}/student-f16.gguf");
        write_student_gguf(MODEL, &student, &out).expect("write student gguf");
        let file = fs::read(&out).expect("read student");
        let linear: u64 = res
            .layers
            .iter()
            .map(|l| {
                (l.wq.len()
                    + l.wk.len()
                    + l.wv.len()
                    + l.wo.len()
                    + l.w_gate.len()
                    + l.w_up.len()
                    + l.w_down.len()) as u64
            })
            .sum();
        let bits = res.bits.nsign + res.bits.nr + res.bits.nc + res.bits.nd + res.bits.ncorr;
        eprintln!(
            "[4] wrote {out} ({source}): {} sha256, {:.2} b/w ({} bits / {} linear weights), \
             rss {:.2} GB",
            hex(&sha256(&file)),
            bits as f64 / linear as f64,
            bits,
            linear,
            rss_gb()
        );
        eprintln!("[4] ok in {:?}", t_test.elapsed());
    } else {
        eprintln!(
            "[4] incomplete: rerun to resume at layer {} (the multi-invocation campaign is the \
             lead's); rss {:.2} GB",
            st.next,
            rss_gb()
        );
    }
}

/// Probe (ignored): one layer's wall and its propagation/stage split at a reduced sequence count,
/// for projecting the production shape when one production layer alone exceeds the budget.
#[test]
#[ignore = "loads the 0.5B model and quantises one layer at a reduced batch; --release --ignored --nocapture"]
fn layer_wall_probe() {
    let t0 = Instant::now();
    let n_seq: usize = std::env::var("APEIRON_PROBE_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let t: usize = std::env::var("APEIRON_PROBE_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let ids_raw = fs::read(TOKENS).expect("read token npy");
    let ids = read_npy_i4(&ids_raw).expect("parse token npy");
    drop(ids_raw);
    let chunks = chunks(&ids, 512).expect("512-token chunks");
    drop(ids);
    let sp = split(chunks.len(), 50, 20, 40).expect("50/20/40 split");
    let cal: Vec<Vec<u32>> = sp
        .cal
        .iter()
        .take(n_seq)
        .map(|&c| chunks[c][..t].to_vec())
        .collect();
    let k = cal.len();

    let engine = Engine::open(MODEL).expect("open teacher");
    let (weights, config) = Weights::from_engine(&engine).expect("load weights");
    let teacher = teacher_from_grad(&weights, &config).expect("teacher");
    drop(weights);
    let cfg = DriverCfg::production(teacher.layer_cfg, t, k);
    let rope = RopeTables::new(&cfg.layer, t).expect("rope");
    let hid = cfg.layer.hidden;
    let h0 = h0_from_chunks(&teacher, &cal).expect("h0");
    let mut st = model_init(&h0);
    drop(h0);
    eprintln!(
        "[probe] k {k} t {t}, loaded at {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let t_prop = Instant::now();
    let _ = parts(
        &cfg.layer,
        &rope,
        &layer_words(&teacher.layers[0]).expect("teacher words"),
        &st.ht[0..t * hid],
        t,
    )
    .expect("propagation probe");
    let prop1 = t_prop.elapsed().as_secs_f64();

    let t_step = Instant::now();
    model_step(&cfg, &rope, &teacher.layers, &teacher.fnorm, &mut st, 1).expect("layer step");
    let step = t_step.elapsed().as_secs_f64();
    let prop = prop1 * k as f64;
    eprintln!(
        "[probe] k={k}: layer wall {step:.1}s (propagation {prop:.1}s = {prop1:.2}s/seq, \
         stage folds {:.1}s), rss {:.2} GB, bits {} {} {} {} {}",
        (step - prop).max(0.0),
        rss_gb(),
        st.bits.nsign,
        st.bits.nr,
        st.bits.nc,
        st.bits.nd,
        st.bits.ncorr
    );
    eprintln!(
        "[probe] per-sequence layer {:.2}s -> k=50 projects {:.0}s/layer, 24 layers {:.0}s",
        step / k as f64,
        step / k as f64 * 50.0,
        step / k as f64 * 50.0 * 24.0
    );
}

/// Peak resident set size in GB from `/proc/self/status`.
fn rss_gb() -> f64 {
    let text = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0.0);
            return kb / 1e6;
        }
    }
    0.0
}

/// Lowercase hex of a digest.
fn hex(d: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in d {
        s.push(char::from_digit(u32::from(b >> 4), 16).unwrap_or('0'));
        s.push(char::from_digit(u32::from(b & 0x0f), 16).unwrap_or('0'));
    }
    s
}

/// FNV-1a 64 fold of `bytes` into `h`.
fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// FNV-1a 64 over the little-endian bytes of the words.
fn fnv_words(v: &[i64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &w in v {
        h = fnv1a(h, &w.to_le_bytes());
    }
    h
}

/// The numerical-recipes LCG the recorded checkpoint fixture was generated with.
struct WitnessLcg(u64);

impl WitnessLcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn word(&mut self) -> i64 {
        i64::try_from(self.next() >> 33).expect("31-bit") % 2001 - 1000
    }
    fn fill(&mut self, n: usize) -> Vec<i64> {
        (0..n).map(|_| self.word()).collect()
    }
}

/// The recorded fixture's geometry: `hidden` 32, one head of 8, `inter` 16, `vocab` 8, two layers.
const WHID: usize = 32;
const WQ: usize = 8;
const WINTER: usize = 16;
const WVOCAB: usize = 8;

fn witness_config() -> apeiron::learn::grad::model::Config {
    apeiron::learn::grad::model::Config {
        n_layer: 2,
        n_head: 1,
        n_head_kv: 1,
        n_embd: WHID,
        n_ff: WINTER,
        vocab: WVOCAB,
        head_dim: 8,
        eps: 4_295,
        rope_theta: 10_000.0,
    }
}

fn witness_teacher(seed: u64) -> Teacher {
    let mut rng = WitnessLcg::new(seed);
    let layer = |rng: &mut WitnessLcg| apeiron::learn::grad::model::LayerWeights {
        attn_norm: rng.fill(WHID),
        wq: rng.fill(WQ * WHID),
        bq: Some(rng.fill(WQ)),
        wk: rng.fill(WQ * WHID),
        bk: Some(rng.fill(WQ)),
        wv: rng.fill(WQ * WHID),
        bv: Some(rng.fill(WQ)),
        wo: rng.fill(WHID * WQ),
        ffn_norm: rng.fill(WHID),
        w_gate: rng.fill(WINTER * WHID),
        w_up: rng.fill(WINTER * WHID),
        w_down: rng.fill(WHID * WINTER),
    };
    let w = Weights {
        layers: vec![layer(&mut rng), layer(&mut rng)],
        output_norm: rng.fill(WHID),
        token_embd: rng.fill(WVOCAB * WHID),
        lm_head: rng.fill(WVOCAB * WHID),
    };
    teacher_from_grad(&w, &witness_config()).expect("witness teacher")
}

/// The recorded pre-migration digest: `h0_from_chunks`, the three checkpoint files, and the loaded
/// `next`/`ht`/`hs` words, folded in the recorded order.
fn witness_transfer(seed: u64) -> u64 {
    let t = witness_teacher(seed);
    let mut rng = WitnessLcg::new(seed ^ 0x5151);
    let chunks: Vec<Vec<u32>> = (0..3)
        .map(|_| {
            (0..4)
                .map(|_| u32::try_from(rng.next() % (WVOCAB as u64)).expect("token"))
                .collect()
        })
        .collect();
    let h0w = h0_from_chunks(&t, &chunks).expect("h0");
    let st = model_init(&h0w);
    let dir = scratch(&format!("witness-{seed:016x}"));
    save(&dir, &st).expect("save");
    let mut h = fnv1a(0xcbf2_9ce4_8422_2325, b"h0");
    h = fnv1a(h, &i64::try_from(h0w.len()).expect("len").to_le_bytes());
    h = fnv1a(h, &fnv_words(&h0w).to_le_bytes());
    for name in ["ht.bin", "hs.bin", "manifest.txt"] {
        let bytes = fs::read(format!("{dir}/{name}")).expect("read checkpoint");
        h = fnv1a(h, &fnv1a(0, name.as_bytes()).to_le_bytes());
        h = fnv1a(h, &bytes);
    }
    let back = load(&dir).expect("load");
    h = fnv1a(h, b"load");
    h = fnv1a(h, &i64::try_from(back.next).expect("next").to_le_bytes());
    h = fnv1a(h, &fnv_words(&back.ht).to_le_bytes());
    h = fnv1a(h, &fnv_words(&back.hs).to_le_bytes());
    fs::remove_dir_all(&dir).expect("clean scratch");
    h
}

/// The recorded pre-migration digests of the transfer face, from the source at base `57d21789`.
/// The checkpoint bytes and the gathered rows are word-identical: the format is unchanged.
/// The recorded digests of the transfer face from base `57d21789`, folded in the recorded order. The
/// `ht.bin`/`hs.bin` bytes are unchanged; the digest moved with the manifest's `v2`→`v3` magic, the
/// format bump that adds the per-layer fits (this fixture saves `next = 0`, so no layer or fits
/// file participates).
#[test]
fn recorded_checkpoint_digests_hold() {
    for (seed, want) in [
        (0x7A11_0001u64, 0xd133_827f_6162_b26fu64),
        (0x7A11_0002, 0x6ae1_86b9_e88e_3270),
    ] {
        assert_eq!(witness_transfer(seed), want, "seed {seed:016x}");
    }
}
