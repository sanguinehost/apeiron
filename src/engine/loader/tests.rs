use super::*;
use crate::format::gguf::{parse_header, DEFAULT_ALIGNMENT};
use crate::format::quant;

/// Minimal little-endian writer for a synthetic header.
struct W(Vec<u8>);
impl W {
    fn new() -> Self {
        W(Vec::new())
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }
    fn string(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.bytes(s.as_bytes());
    }
}

/// A Qwen2 header with a `[4, 8]` `Q8_0` embedding at data offset 0.
fn qwen2_header(embd_offset: u64, file_len: u64) -> GgufHeader {
    qwen2_header_typed(quant::GGML_TYPE_Q8_0, [4, 8], embd_offset, file_len)
}

/// A Qwen2 header with one `dims` tensor of `ggml_type`; the row dim must equal the `[4]`
/// embedding length the metadata declares.
fn qwen2_header_typed(
    ggml_type: u32,
    dims: [u64; 2],
    embd_offset: u64,
    file_len: u64,
) -> GgufHeader {
    let mut w = W::new();
    w.bytes(b"GGUF");
    w.u32(3);
    w.u64(1); // 1 tensor
    w.u64(6); // 6 kv
    w.string("general.architecture");
    w.u32(8); // STRING
    w.string("qwen2");
    for (k, v) in [
        ("qwen2.block_count", 1u32),
        ("qwen2.embedding_length", 4),
        ("qwen2.feed_forward_length", 8),
        ("qwen2.attention.head_count", 2),
        ("qwen2.attention.head_count_kv", 1),
    ] {
        w.string(k);
        w.u32(4); // UINT32
        w.u32(v);
    }
    w.string("token_embd.weight");
    w.u32(2);
    w.u64(dims[0]);
    w.u64(dims[1]);
    w.u32(ggml_type);
    w.u64(embd_offset);
    parse_header(&w.0, file_len).expect("synthetic header parses")
}

/// A Qwen2 header whose float config keys are `FLOAT32`/`FLOAT64`, as GGUF writers store them.
fn qwen2_header_with_float_meta(rope: f32, eps: f64) -> GgufHeader {
    let mut w = W::new();
    w.bytes(b"GGUF");
    w.u32(3);
    w.u64(1); // 1 tensor
    w.u64(8); // 8 kv
    w.string("general.architecture");
    w.u32(8); // STRING
    w.string("qwen2");
    for (k, v) in [
        ("qwen2.block_count", 1u32),
        ("qwen2.embedding_length", 4),
        ("qwen2.feed_forward_length", 8),
        ("qwen2.attention.head_count", 2),
        ("qwen2.attention.head_count_kv", 1),
    ] {
        w.string(k);
        w.u32(4); // UINT32
        w.u32(v);
    }
    w.string("qwen2.rope.freq_base");
    w.u32(6); // FLOAT32
    w.u32(rope.to_bits());
    w.string("qwen2.attention.layer_norm_rms_epsilon");
    w.u32(12); // FLOAT64
    w.u64(eps.to_bits());
    w.string("token_embd.weight");
    w.u32(2);
    w.u64(4);
    w.u64(8);
    w.u32(quant::GGML_TYPE_Q8_0);
    w.u64(0);
    parse_header(&w.0, 1 << 16).expect("synthetic header parses")
}

#[test]
fn table_records_q5_0_geometry_and_checks_the_extent() {
    let h = qwen2_header_typed(quant::GGML_TYPE_Q5_0, [4, 8], 0, 1 << 16);
    let t = build_table(&h, 1 << 16).expect("table builds");
    let e = t.get("token_embd.weight").expect("embedding entry");
    assert_eq!(e.quant, QuantKind::Q5_0);
    assert_eq!(e.quant.name(), "Q5_0");
    assert_eq!(e.n_elements, 32);
    assert_eq!(e.n_bytes, Some(22)); // 32 weights = 1 Q5_0 block x 22 bytes
}

#[test]
fn config_resolves_from_qwen2_metadata() {
    let h = qwen2_header(0, 1 << 16);
    let c = resolve_config(&h).expect("config resolves");
    assert_eq!(c.n_layer, 1);
    assert_eq!(c.n_head, 2);
    assert_eq!(c.n_head_kv, 1);
    assert_eq!(c.n_embd, 4);
    assert_eq!(c.n_ff, 8);
    assert_eq!(c.vocab_size, 8);
    assert_eq!(c.head_dim, 2);
    assert!((c.rope_theta - DEFAULT_ROPE_THETA).abs() < f64::EPSILON);
}

/// A block count above the tensor directory is refused before the per-layer schedule is
/// allocated (fuzz `gguf_open` seed `block-count-u32-max`: a 4 GiB `Vec` from a 9600-byte file).
#[test]
fn config_refuses_more_layers_than_tensors() {
    let mut w = W::new();
    w.bytes(b"GGUF");
    w.u32(3);
    w.u64(1); // 1 tensor
    w.u64(6); // 6 kv
    w.string("general.architecture");
    w.u32(8); // STRING
    w.string("qwen2");
    for (k, v) in [
        ("qwen2.block_count", u32::MAX),
        ("qwen2.embedding_length", 4),
        ("qwen2.feed_forward_length", 8),
        ("qwen2.attention.head_count", 2),
        ("qwen2.attention.head_count_kv", 1),
    ] {
        w.string(k);
        w.u32(4); // UINT32
        w.u32(v);
    }
    w.string("token_embd.weight");
    w.u32(2);
    w.u64(4);
    w.u64(8);
    w.u32(quant::GGML_TYPE_Q8_0);
    w.u64(0);
    let h = parse_header(&w.0, 1 << 16).expect("synthetic header parses");
    match resolve_config(&h) {
        Err(EngineError::ConfigMismatch(m)) => assert!(m.contains("block_count"), "{m}"),
        other => panic!("expected a block_count refusal, got {other:?}"),
    }
}

#[test]
fn config_interprets_float_metadata_at_the_boundary() {
    let h = qwen2_header_with_float_meta(1e6f32, 1e-5f64);
    let c = resolve_config(&h).expect("config resolves");
    assert_eq!(c.rope_theta.to_bits(), f64::from(1e6f32).to_bits());
    assert_eq!(c.rms_eps.to_bits(), 1e-5f64.to_bits());
}

#[test]
fn table_refuses_a_tensor_past_eof() {
    let h = qwen2_header(1 << 20, 1 << 16);
    let e = build_table(&h, 1 << 16).expect_err("offset past EOF refuses");
    assert!(matches!(e, EngineError::TensorOutOfBounds { .. }));
    assert!(e.to_string().contains("past EOF"));
}

#[test]
fn table_records_checked_extent_and_quant_kind() {
    let h = qwen2_header(0, 1 << 16);
    let t = build_table(&h, 1 << 16).expect("table builds");
    let e = t.get("token_embd.weight").expect("embedding entry");
    assert_eq!(e.quant, QuantKind::Q8_0);
    assert_eq!(e.n_elements, 32);
    assert_eq!(e.n_bytes, Some(34)); // 32 weights = 1 Q8_0 block x 34 bytes
    assert_eq!(e.byte_range.start, h.data0);
    assert_eq!(e.byte_range.end, 1 << 16);
}

#[test]
fn quant_kind_classifies_known_ids() {
    assert_eq!(QuantKind::from_ggml(0), QuantKind::F32);
    assert_eq!(QuantKind::from_ggml(1), QuantKind::F16);
    assert_eq!(QuantKind::from_ggml(12), QuantKind::Q4K);
    assert_eq!(QuantKind::from_ggml(14), QuantKind::Q6K);
    assert_eq!(QuantKind::from_ggml(8), QuantKind::Q8_0);
    assert_eq!(QuantKind::from_ggml(6), QuantKind::Q5_0);
    assert_eq!(QuantKind::from_ggml(7), QuantKind::Other(7));
}

/// A header carrying only the given metadata, so no architecture or tensor directory is needed.
fn h_meta(kv: Vec<(&str, GVal)>) -> GgufHeader {
    GgufHeader {
        version: 3,
        kv: kv
            .into_iter()
            .map(|(k, v)| (k.as_bytes().to_vec(), v))
            .collect(),
        tensors: Vec::new(),
        alignment: DEFAULT_ALIGNMENT,
        header_end: 0,
        data0: 0,
    }
}

fn s(v: &str) -> GVal {
    GVal::Str(v.as_bytes().to_vec())
}

fn ia(v: &[i64]) -> GVal {
    GVal::Arr(5, v.iter().map(|x| GVal::Int(i128::from(*x), 5)).collect())
}

fn set(kv: &mut Vec<(&'static str, GVal)>, key: &'static str, v: GVal) {
    for e in kv.iter_mut() {
        if e.0 == key {
            e.1 = v;
            return;
        }
    }
    kv.push((key, v));
}

/// A valid `identity` fold contract at `version`; `tied` adds `tied_output = true`.
fn valid_kv(version: u32, tied: bool) -> Vec<(&'static str, GVal)> {
    let mut kv: Vec<(&'static str, GVal)> = vec![
        ("prism.hadamard.version", GVal::Int(i128::from(version), 4)),
        ("prism.hadamard.block_size", GVal::Int(2, 4)),
        ("prism.hadamard.transform", s(HADAMARD_TRANSFORM)),
        ("prism.hadamard.axis", s(HADAMARD_AXIS)),
        ("prism.hadamard.sign_mode", s("identity")),
        (
            "prism.hadamard.weight_names",
            GVal::Arr(8, vec![s("blk.0.ffn_gate.weight")]),
        ),
    ];
    if tied {
        kv.push(("prism.hadamard.tied_output", GVal::Bool(true)));
    }
    kv
}

/// `valid_kv(1, false)` with `key` replaced, or appended when absent.
fn replaced(key: &'static str, v: GVal) -> Vec<(&'static str, GVal)> {
    let mut kv = valid_kv(1, false);
    set(&mut kv, key, v);
    kv
}

/// A version-2 tied contract in `explicit` mode with the given sign table.
fn explicit_kv(widths: &[i64], values: &[i64]) -> Vec<(&'static str, GVal)> {
    let mut kv = valid_kv(2, true);
    kv.retain(|e| e.0 != "prism.hadamard.sign_mode");
    kv.push(("prism.hadamard.sign_mode", s("explicit")));
    set(&mut kv, "prism.hadamard.sign_widths", ia(widths));
    set(&mut kv, "prism.hadamard.sign_values", ia(values));
    kv
}

fn refusal(kv: Vec<(&str, GVal)>) -> String {
    match hadamard_meta(&h_meta(kv)) {
        Err(e) => e.to_string(),
        Ok(v) => panic!("expected a refusal, got {v:?}"),
    }
}

#[test]
fn hadamard_meta_absent_version_leaves_config_unchanged() {
    let h = h_meta(vec![("qwen2.block_count", GVal::Int(2, 4))]);
    assert_eq!(hadamard_meta(&h), Ok(None));
}

#[test]
fn hadamard_meta_accepts_each_sign_mode() {
    let id = hadamard_meta(&h_meta(valid_kv(1, false)))
        .expect("identity metadata accepts")
        .expect("contract present");
    assert_eq!((id.version, id.block_size, id.tied_output), (1, 2, false));
    assert_eq!(id.weight_names, vec!["blk.0.ffn_gate.weight".to_string()]);
    assert_eq!(id.signs, None);

    let ex = hadamard_meta(&h_meta(explicit_kv(&[2], &[1, -1])))
        .expect("explicit metadata accepts")
        .expect("contract present");
    assert_eq!(ex.signs, Some(vec![(2u32, vec![1i8, -1i8])]));
}

/// Each rule the fork rejects at load: one case, asserting that rule's refusal.
#[test]
fn hadamard_meta_refuses_each_fork_rejection() {
    assert!(refusal(replaced("prism.hadamard.version", GVal::Int(3, 4)))
        .contains("version 3 is not 1 or 2"));
    assert!(refusal(valid_kv(2, false)).contains("version 2 requires tied_output=true"));
    assert!(refusal(valid_kv(1, true)).contains("version 1 forbids it"));
    for bs in [0i64, 3, 24] {
        let msg = refusal(replaced(
            "prism.hadamard.block_size",
            GVal::Int(i128::from(bs), 4),
        ));
        assert!(msg.contains("zero or not a power of two"), "bs {bs}: {msg}");
    }
    for (key, needle) in [
        (
            "prism.hadamard.transform",
            "unsupported prism.hadamard.transform",
        ),
        ("prism.hadamard.axis", "unsupported prism.hadamard.axis"),
        (
            "prism.hadamard.sign_mode",
            "unsupported prism.hadamard.sign_mode",
        ),
    ] {
        let msg = refusal(replaced(key, s("bogus")));
        assert!(msg.contains(needle), "{key}: {msg}");
    }
    assert!(refusal(replaced(
        "prism.hadamard.weight_names",
        GVal::Arr(8, Vec::new())
    ))
    .contains("weight_names is empty"));
    let mut missing = valid_kv(1, false);
    missing.retain(|e| e.0 != "prism.hadamard.block_size");
    assert!(refusal(missing).contains("block_size"));
    // explicit mode with no widths would leave an empty sign table
    assert!(refusal(explicit_kv(&[], &[])).contains("sign_widths is empty"));
    // width zero is not positive
    assert!(refusal(explicit_kv(&[0], &[])).contains("sign width 0"));
    // width not a multiple of block_size
    assert!(refusal(explicit_kv(&[3], &[1, 1, 1])).contains("sign width 3"));
    // a sign value outside {-1, +1}
    assert!(refusal(explicit_kv(&[2], &[1, 2])).contains("must be +/-1"));
    // sign_values shorter than a declared width
    assert!(refusal(explicit_kv(&[4], &[1, 1])).contains("is too short"));
    // widths cover fewer values than present
    assert!(refusal(explicit_kv(&[2], &[1, 1, 1])).contains("sign_values length mismatch"));
}
