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

use apeiron::engine::{Engine, KernelMode};
use apeiron::learn::fold::layer::driver::{
    layer_words, model_finish, model_init, model_step, quantise_model, DriverCfg, LayerWeights,
};
use apeiron::learn::fold::run::convert::teacher_from_grad;
use apeiron::learn::fold::run::npy::{chunks, read_npy_i4, split};
use apeiron::learn::fold::run::transfer::{
    checkpoint_present, h0_from_chunks, load, save, write_student_gguf, TransferError,
};
use apeiron::learn::fold::stats::parts::{parts, LayerCfg, RopeTables};
use apeiron::learn::grad::model::Weights;
use apeiron::learn::solve::q32::{to_grid, GRID_BOUND};
use apeiron::sha256::sha256;

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
const MODEL: &str = "/home/socol/models/gguf/qwen05b-f16.gguf";

/// The C4 calibration tokens (`kl_distill.train_ids`, disjoint from the `seal/calib-c4.txt` shard).
const TOKENS: &str = "/home/socol/oc-tmp/lanes/w18/kd/c4data/train-c4.qwen.npy";

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
        .unwrap_or_else(|_| "/home/socol/oc-tmp/q-tr1/transfer-test".to_string());
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

/// [1] resume equivalence: one-shot `quantise_model` against `model_step(0..1)` → `save` → `load`
/// → `model_step(1..3)` → `model_finish`: every output tensor and `fnorm` bit-identical.
#[test]
fn resume_equivalence() {
    let t0 = Instant::now();
    let (cfg, rope, layers, fnorm, h0, embed, vocab) = model_fixture();
    let lr = 0.15;

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

/// [3] `GGUF` identity (ignored, real model): rewrite the teacher's own weights through
/// `write_student_gguf` and compare the 8-token logits and the file digest.
#[test]
#[ignore = "loads the 994 MB F16 Qwen2.5-0.5B and writes a ~1 GB GGUF; --release --ignored --nocapture"]
fn gguf_identity() {
    let t0 = Instant::now();
    let root = std::env::var("APEIRON_GGUF_IDENTITY_DIR")
        .unwrap_or_else(|_| "/home/socol/oc-tmp/q-tr1/gguf-identity".to_string());
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
        .unwrap_or_else(|_| "/home/socol/oc-tmp/q-tr1/ckpt".to_string());
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
    let hid = cfg.layer.hidden;

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

    let budget = 520.0_f64;
    let t_loop = Instant::now();
    let mut last_wall: Option<f64> = None;
    while st.next < teacher.n_layer {
        if let Some(last) = last_wall {
            if t_loop.elapsed().as_secs_f64() + last > budget {
                eprintln!(
                    "[4] stop: {:.1}s + last layer {last:.1}s > {budget:.0}s budget",
                    t_loop.elapsed().as_secs_f64()
                );
                break;
            }
        }
        let l = st.next;
        let t_layer = Instant::now();
        // Propagation proxy: the teacher forward of this layer over four sequences, scaled.
        let probe = 4.min(n_seq);
        let t_fwd = Instant::now();
        for s in 0..probe {
            let row = s * t * hid;
            let _ = parts(
                &cfg.layer,
                &rope,
                &layer_words(&teacher.layers[l]).expect("teacher words"),
                &st.ht[row..row + t * hid],
                t,
            )
            .expect("propagation probe");
        }
        let fwd = t_fwd.elapsed().as_secs_f64() * (n_seq as f64 / probe as f64);
        model_step(&cfg, &rope, &teacher.layers, &teacher.fnorm, &mut st, l + 1)
            .expect("layer step");
        save(&dir, &st).expect("checkpoint");
        let wall = t_layer.elapsed().as_secs_f64();
        last_wall = Some(wall);
        let layer_bytes = fs::metadata(format!("{dir}/layer-{l}.bin"))
            .expect("layer file")
            .len();
        eprintln!(
            "[4] layer {l}: wall {wall:.1}s (propagation {fwd:.1}s, stage folds {:.1}s), \
             rss {:.2} GB, bits {} {} {} {} {}, checkpoint {:.1} MB",
            (wall - fwd).max(0.0),
            rss_gb(),
            st.bits.nsign,
            st.bits.nr,
            st.bits.nc,
            st.bits.nd,
            st.bits.ncorr,
            layer_bytes as f64 / 1e6
        );
    }
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
            0.15,
        )
        .expect("finish");
        let student =
            apeiron::learn::fold::run::convert::student_to_grad(&weights, &res.layers, &res.fnorm)
                .expect("student weights");
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
            "[4] wrote {out}: {} sha256, {:.2} b/w ({} bits / {} linear weights), rss {:.2} GB",
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
