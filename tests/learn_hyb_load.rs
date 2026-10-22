//! `tests/learn_hyb_load.rs` — the per-layer, name-keyed linear loader on the `Qwen3.5-0.8B` proxy:
//! an affine (`Gated DeltaNet`) layer's `attn_qkv`/`attn_gate`/`ssm_out` set and a gated-attention
//! layer's doubled-`attn_q` set, each with its input's tap slot, and the `Weights::from_engine`
//! refusal that names the missing `attn_q`. Grade `tested`.
//!
//! The file is read only through the engine's own directory, and one layer is loaded at a time: the
//! embedding is ~254M words, so the whole model is never materialised.

use apeiron::engine::{Engine, MixerKind};
use apeiron::learn::grad::load::{
    layer_from_engine, LayerLinears, TAP_ATTN_OUT, TAP_LN1_OUT, TAP_LN2_OUT, TAP_M,
};
use apeiron::learn::grad::model::{Config, ModelError, Weights};

/// The `Qwen3.5-0.8B` proxy the fold transfer targets (arch `qwen35`, 24 trunk layers, every fourth
/// layer gated attention over `full_attention_interval = 4`).
const QWEN35: &str = "/home/socol/oc-tmp/kld08/qwen08b-bf16.gguf";

/// `qwen35.embedding_length`.
const N_EMBD: usize = 1024;
/// `qwen35.feed_forward_length`.
const N_FF: usize = 3584;
/// `qwen35.attention.head_count · key_length` = `8 · 256`.
const Q_OUT: usize = 2048;
/// `qwen35.attention.head_count_kv · key_length` = `2 · 256`.
const KV_OUT: usize = 512;
/// `qwen35.ssm.time_step_rank · (inner_size / time_step_rank)` = `16 · 128`, the affine mixer's value
/// width and the `attn_gate` rows.
const VAL: usize = 2048;
/// `2 · group_count · state_size + VAL` = `2 · 16 · 128 + 2048`, the `attn_qkv` rows.
const QKV: usize = 6144;

/// The model path, overridable so the same assertions run against another quantization of the same
/// directory.
fn model_path() -> String {
    std::env::var("APEIRON_QWEN35").unwrap_or_else(|_| QWEN35.to_string())
}

/// One line per linear: `name rows×cols tap=… w=… bias=…`.
fn table(l: usize, ll: &LayerLinears) -> String {
    let mut rows = vec![format!(
        "layer {l} mixer={:?} attn_norm={} ffn_norm={}",
        ll.mixer,
        ll.attn_norm.len(),
        ll.ffn_norm.len()
    )];
    for x in &ll.linears {
        rows.push(format!(
            "  {:<11} {:>5}x{:<5} tap={} w={} bias={}",
            x.name,
            x.rows,
            x.cols,
            x.tap,
            x.w.len(),
            match &x.bias {
                Some(b) => b.len().to_string(),
                None => String::from("none"),
            }
        ));
    }
    rows.join("\n")
}

/// The `(name, rows, cols, tap)` list of `ll`, for one comparison against the expected set.
fn shape_list(ll: &LayerLinears) -> Vec<(&'static str, usize, usize, usize)> {
    ll.linears
        .iter()
        .map(|x| (x.name, x.rows, x.cols, x.tap))
        .collect()
}

/// Every loaded matrix is `rows · cols` words, carries no bias on this model, and sits on the tap
/// slot its input is recorded at.
fn check_linears(l: usize, ll: &LayerLinears) {
    for x in &ll.linears {
        assert_eq!(
            x.w.len(),
            x.rows * x.cols,
            "layer {l} {}: {} words != {}×{}",
            x.name,
            x.w.len(),
            x.rows,
            x.cols
        );
        assert!(x.bias.is_none(), "layer {l} {} carries no bias", x.name);
    }
}

/// Layer 0's affine set, with the directory's shapes and the tap slots the engine records its inputs
/// at.
fn affine_layer(engine: &Engine, cfg: &Config) -> LayerLinears {
    let ll = layer_from_engine(engine, cfg, 0).expect("affine layer 0");
    assert_eq!(ll.mixer, MixerKind::Affine);
    assert_eq!(
        (ll.attn_norm.len(), ll.ffn_norm.len()),
        (N_EMBD, N_EMBD),
        "layer 0's two norm gains"
    );
    assert_eq!(
        shape_list(&ll),
        vec![
            ("attn_qkv", QKV, N_EMBD, TAP_LN1_OUT),
            ("attn_gate", VAL, N_EMBD, TAP_LN1_OUT),
            ("ssm_out", N_EMBD, VAL, TAP_ATTN_OUT),
            ("ffn_gate", N_FF, N_EMBD, TAP_LN2_OUT),
            ("ffn_up", N_FF, N_EMBD, TAP_LN2_OUT),
            ("ffn_down", N_EMBD, N_FF, TAP_M),
        ],
        "affine layer 0's fitted set"
    );
    check_linears(0, &ll);
    ll
}

/// A gated-attention layer's set: `attn_q` doubled for the output gate, no `attn_qkv`/`ssm_out`.
fn softmax_layer(engine: &Engine, cfg: &Config) -> (usize, LayerLinears) {
    let ec = &engine.config;
    let soft = ec
        .mixers
        .iter()
        .position(|m| *m == MixerKind::Softmax)
        .expect("a gated-attention layer");
    // `qwen35.full_attention_interval = 4`: layer `l` is softmax iff `(l + 1) % 4 == 0`.
    assert_eq!((soft + 1) % 4, 0, "first softmax layer {soft}");
    let ll = layer_from_engine(engine, cfg, soft).expect("gated-attention layer");
    assert_eq!(ll.mixer, MixerKind::Softmax);
    assert_eq!(
        shape_list(&ll),
        vec![
            ("attn_q", 2 * Q_OUT, N_EMBD, TAP_LN1_OUT),
            ("attn_k", KV_OUT, N_EMBD, TAP_LN1_OUT),
            ("attn_v", KV_OUT, N_EMBD, TAP_LN1_OUT),
            ("attn_output", N_EMBD, Q_OUT, TAP_ATTN_OUT),
            ("ffn_gate", N_FF, N_EMBD, TAP_LN2_OUT),
            ("ffn_up", N_FF, N_EMBD, TAP_LN2_OUT),
            ("ffn_down", N_EMBD, N_FF, TAP_M),
        ],
        "gated-attention layer {soft}'s fitted set"
    );
    check_linears(soft, &ll);
    (soft, ll)
}

#[test]
fn qwen35_trunk_loads() {
    let path = model_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skip qwen35_trunk_loads: {path} absent");
        return;
    }
    let engine = Engine::open(&path).unwrap_or_else(|e| panic!("open {path}: {e:?}"));
    let cfg = Config::from_engine(&engine).expect("qwen35 config");
    let ec = &engine.config;
    assert_eq!((cfg.n_embd, cfg.n_ff), (N_EMBD, N_FF), "directory dims");
    assert_eq!(
        (cfg.q_out().unwrap(), cfg.kv_out().unwrap()),
        (Q_OUT, KV_OUT)
    );
    let geom = ec.affine.expect("qwen35 carries the affine geometry");
    assert_eq!(
        (geom.n_k_heads * geom.head_k, geom.n_v_heads * geom.head_v),
        (VAL, VAL),
        "key and value widths"
    );
    assert!(ec.attn_gate, "qwen35's attn_q carries the output gate");
    assert_eq!(ec.mixers.len(), cfg.n_layer, "one mixer per trunk layer");
    assert_eq!(ec.mixers[0], MixerKind::Affine, "layer 0 is DeltaNet");

    let affine = affine_layer(&engine, &cfg);
    let (soft, attn) = softmax_layer(&engine, &cfg);
    println!("{}\n{}", table(0, &affine), table(soft, &attn));

    // The fitted backward is the softmax schedule, so the whole-model load refuses the first affine
    // layer by the name the backward would have read.
    match Weights::from_engine(&engine) {
        Err(ModelError::Missing(name)) => assert_eq!(name, "blk.0.attn_q.weight"),
        Err(other) => panic!("from_engine refused with {other:?}, not Missing(attn_q)"),
        Ok(_) => panic!("from_engine loaded a model with an affine layer"),
    }
}
