//! Gate A (forward logits bit-identical to `Engine::step_logits`) and gate B (backward analytic
//! gradient against a central difference of the KL loss) over a 2-layer `qwen2` `F16`/`F32` fixture.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "test-only fixture/difference arithmetic on bounded small values; not extracted"
)]

use super::*;
use crate::engine::{Engine, ONE};
use crate::format::quant::{GGML_TYPE_F16, GGML_TYPE_F32};
use crate::learn::grad::head::kl_row;

/// Embedding width and query projection width.
const E: usize = 8;
const NH: usize = 2;
const NKV: usize = 1;
const HD: usize = 4;
const QW: usize = NH * HD; // 8
const KVW: usize = NKV * HD; // 4
const FF: usize = 16;
const V: usize = 16;
const L: usize = 2;
const TOKENS: [u32; 4] = [1, 3, 5, 2];

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}
fn kv_u32(b: &mut Vec<u8>, k: &str, v: u32) {
    put_str(b, k);
    put_u32(b, 4);
    put_u32(b, v);
}
fn kv_str(b: &mut Vec<u8>, k: &str, v: &str) {
    put_str(b, k);
    put_u32(b, 8);
    put_str(b, v);
}

/// Round-to-nearest-even `f16` bits for a finite normal `f32` in `(2^-14, 65504)`.
fn f16_round(x: f32) -> u16 {
    let bits = x.to_bits();
    if bits << 1 == 0 {
        return 0;
    }
    let sign = ((bits >> 16) & 0x8000) as u16;
    let e = i32::try_from((bits >> 23) & 0xff).expect("8 bits") - 127 + 15;
    let mut m = i32::try_from((bits >> 13) & 0x3ff).expect("10 bits");
    let rem = bits & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m & 1 == 1) {
        m += 1;
    }
    let carry = m >> 10;
    m &= 0x3ff;
    sign | (((e + carry) as u16) << 10) | (m as u16)
}

/// Deterministic `F16` payload of `rows · cols` weights in `{-2, -1, 0, 1, 2}·2^-5`; `salt`
/// separates tensors.
fn f16_weight(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
    const SCALE: f32 = 1.0 / 32.0;
    let mut b: Vec<u8> = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        for c in 0..cols {
            let code = i32::try_from((r * 7 + c * 3 + salt * 11) % 5).expect("small") - 2;
            b.extend_from_slice(&f16_round(code as f32 * SCALE).to_le_bytes());
        }
    }
    b
}

/// All-`1.0` `F32` gain of `n` entries.
fn f32_gain(n: usize) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(n * 4);
    for _ in 0..n {
        b.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
    }
    b
}

/// A deterministic nonzero `F32` bias of `n` entries in `{-0.5, -0.25, 0, 0.25, 0.5}`.
fn f32_bias(n: usize, salt: usize) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(n * 4);
    for i in 0..n {
        let code = i32::try_from((i + salt * 2) % 5).expect("small") - 2;
        b.extend_from_slice(&(code as f32 * 0.25).to_bits().to_le_bytes());
    }
    b
}

/// A 2-layer `qwen2` fixture with nonzero attention biases, a stored `output.weight` head, `n_embd
/// 8`, 2 query heads over 1 KV head of width 4, `n_ff 16`, vocabulary 16.
fn build_fixture() -> Vec<u8> {
    let nl = L;
    let u = |x: usize| u64::try_from(x).expect("fixture dims");
    let mut ts: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    let mut push = |name: String, dims: Vec<u64>, ty: u32, payload: Vec<u8>| {
        ts.push((name, dims, ty, payload));
    };
    for l in 0..nl {
        push(
            format!("blk.{l}.attn_norm.weight"),
            vec![u(E)],
            GGML_TYPE_F32,
            f32_gain(E),
        );
        push(
            format!("blk.{l}.attn_q.weight"),
            vec![u(E), u(QW)],
            GGML_TYPE_F16,
            f16_weight(QW, E, 1 + l),
        );
        push(
            format!("blk.{l}.attn_q.bias"),
            vec![u(QW)],
            GGML_TYPE_F32,
            f32_bias(QW, 1 + l),
        );
        push(
            format!("blk.{l}.attn_k.weight"),
            vec![u(E), u(KVW)],
            GGML_TYPE_F16,
            f16_weight(KVW, E, 3 + l),
        );
        push(
            format!("blk.{l}.attn_k.bias"),
            vec![u(KVW)],
            GGML_TYPE_F32,
            f32_bias(KVW, 3 + l),
        );
        push(
            format!("blk.{l}.attn_v.weight"),
            vec![u(E), u(KVW)],
            GGML_TYPE_F16,
            f16_weight(KVW, E, 5 + l),
        );
        push(
            format!("blk.{l}.attn_v.bias"),
            vec![u(KVW)],
            GGML_TYPE_F32,
            f32_bias(KVW, 5 + l),
        );
        push(
            format!("blk.{l}.attn_output.weight"),
            vec![u(QW), u(E)],
            GGML_TYPE_F16,
            f16_weight(E, QW, 7 + l),
        );
        push(
            format!("blk.{l}.ffn_norm.weight"),
            vec![u(E)],
            GGML_TYPE_F32,
            f32_gain(E),
        );
        push(
            format!("blk.{l}.ffn_gate.weight"),
            vec![u(E), u(FF)],
            GGML_TYPE_F16,
            f16_weight(FF, E, 9 + l),
        );
        push(
            format!("blk.{l}.ffn_up.weight"),
            vec![u(E), u(FF)],
            GGML_TYPE_F16,
            f16_weight(FF, E, 11 + l),
        );
        push(
            format!("blk.{l}.ffn_down.weight"),
            vec![u(FF), u(E)],
            GGML_TYPE_F16,
            f16_weight(E, FF, 13 + l),
        );
    }
    push(
        "output_norm.weight".to_string(),
        vec![u(E)],
        GGML_TYPE_F32,
        f32_gain(E),
    );
    push(
        "token_embd.weight".to_string(),
        vec![u(E), u(V)],
        GGML_TYPE_F16,
        f16_weight(V, E, 21),
    );
    push(
        "output.weight".to_string(),
        vec![u(E), u(V)],
        GGML_TYPE_F16,
        f16_weight(V, E, 23),
    );

    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, u64::try_from(ts.len()).expect("small"));
    put_u64(&mut b, 6);
    kv_str(&mut b, "general.architecture", "qwen2");
    kv_u32(&mut b, "qwen2.block_count", nl as u32);
    kv_u32(&mut b, "qwen2.attention.head_count", NH as u32);
    kv_u32(&mut b, "qwen2.attention.head_count_kv", NKV as u32);
    kv_u32(&mut b, "qwen2.embedding_length", E as u32);
    kv_u32(&mut b, "qwen2.feed_forward_length", FF as u32);
    let mut off = 0u64;
    for (name, dims, ty, payload) in &ts {
        put_str(&mut b, name);
        put_u32(&mut b, u32::try_from(dims.len()).expect("rank 2"));
        for &d in dims {
            put_u64(&mut b, d);
        }
        put_u32(&mut b, *ty);
        put_u64(&mut b, off);
        off += u64::try_from(payload.len()).expect("small");
    }
    let pad = (32 - b.len() % 32) % 32;
    b.resize(b.len() + pad, 0);
    for (_, _, _, payload) in &ts {
        b.extend_from_slice(payload);
    }
    b
}

/// Write the fixture to a per-process temp file and open it.
fn open_fixture(tag: &str) -> (Engine, String) {
    open_bytes(tag, &build_fixture())
}

/// Write `bytes` to a per-process temp file and open it.
fn open_bytes(tag: &str, bytes: &[u8]) -> (Engine, String) {
    let path =
        std::env::temp_dir().join(format!("apeiron_grad_model_{}_{tag}", std::process::id()));
    std::fs::write(&path, bytes).expect("write fixture");
    let path = path.to_str().expect("utf8 temp path").to_string();
    (Engine::open(&path).expect("open fixture"), path)
}

/// Gate A: `forward`'s logits equal `Engine::step_logits` bit-for-bit at every position.
///
/// The engine's default carrier is the `f32` one (`APEIRON_F32_CARRIER != "0"`), whose attention
/// and `SwiGLU` are native binary32; the port is a pure-Q32 composition, so the gate compares
/// against the Q32 carrier. `carrier_from_env` caches the variable in a process-wide `OnceLock`
/// that any earlier test may already have set to `f32`, so the comparison runs in a child process
/// of this test binary with `APEIRON_F32_CARRIER=0` and only the child test selected.
#[test]
fn forward_logits_match_step_logits_bit_for_bit() {
    let out = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "learn::grad::model::tests::gate_a_q32_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("APEIRON_F32_CARRIER", "0")
        .output()
        .expect("spawn gate A child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "gate A child failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The gate A body; run only by [`forward_logits_match_step_logits_bit_for_bit`] in a child
/// process whose carrier is Q32.
#[test]
#[ignore = "run in a child process with APEIRON_F32_CARRIER=0 by forward_logits_match_step_logits_bit_for_bit"]
fn gate_a_q32_child() {
    let (engine, path) = open_fixture("gateA");
    let mut kv = engine.decode_state(TOKENS.len()).expect("kv");
    assert_eq!(
        format!("{:?}", kv.carrier()),
        "Q32",
        "the Q32 gate needs the Q32 KvState carrier (APEIRON_F32_CARRIER=0)"
    );
    let (w, cfg) = Weights::from_engine(&engine).expect("weights");
    let (logits, tape) = forward(&w, &cfg, &TOKENS).expect("forward");
    assert_eq!(logits.len(), TOKENS.len() * V);
    assert_eq!(tape.layers.len(), L);
    for (t, &tok) in TOKENS.iter().enumerate() {
        let want = engine.step_logits(&mut kv, tok).expect("engine logits");
        assert_eq!(want.len(), V);
        let got = &logits[t * V..(t + 1) * V];
        if got != want.as_slice() {
            let (idx, g, e) = got
                .iter()
                .zip(&want)
                .enumerate()
                .find(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| (i, *a, *b))
                .expect("a difference exists");
            panic!("position {t}: logit[{idx}] forward {g} != engine {e}");
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// The KL objective and its logit gradient at the student's `forward`; the teacher logits are fixed.
fn objective(w: &Weights, cfg: &Config, tokens: &[u32], teacher: &[i64]) -> (i64, Vec<i64>) {
    let (logits, _) = forward(w, cfg, tokens).expect("forward");
    let t = tokens.len();
    let v = cfg.vocab;
    let mut loss = 0i64;
    let mut dz = vec![0i64; t * v];
    for r in 0..t {
        let (l, g) =
            kl_row(&logits[r * v..(r + 1) * v], &teacher[r * v..(r + 1) * v]).expect("kl_row");
        loss += l;
        dz[r * v..(r + 1) * v].copy_from_slice(&g);
    }
    (loss, dz)
}

/// A coordinate inside one weight tensor.
#[derive(Clone, Copy)]
enum Coord {
    AttnNorm(usize, usize),
    Wq(usize, usize, usize),
    Wk(usize, usize, usize),
    Wv(usize, usize, usize),
    Wo(usize, usize, usize),
    WGate(usize, usize, usize),
    WDown(usize, usize, usize),
    TokenEmbd(usize, usize),
}

/// The `(buffer, index)` a coordinate names, plus a name for the failure message.
fn slot(w: &Weights, c: Coord) -> (&Vec<i64>, usize, String) {
    match c {
        Coord::AttnNorm(l, i) => (
            &w.layers[l].attn_norm,
            i,
            format!("layer{l}.attn_norm[{i}]"),
        ),
        Coord::Wq(l, r, k) => (&w.layers[l].wq, r * E + k, format!("layer{l}.wq[{r},{k}]")),
        Coord::Wk(l, r, k) => (&w.layers[l].wk, r * E + k, format!("layer{l}.wk[{r},{k}]")),
        Coord::Wv(l, r, k) => (&w.layers[l].wv, r * E + k, format!("layer{l}.wv[{r},{k}]")),
        Coord::Wo(l, r, k) => (&w.layers[l].wo, r * QW + k, format!("layer{l}.wo[{r},{k}]")),
        Coord::WGate(l, r, k) => (
            &w.layers[l].w_gate,
            r * E + k,
            format!("layer{l}.w_gate[{r},{k}]"),
        ),
        Coord::WDown(l, r, k) => (
            &w.layers[l].w_down,
            r * FF + k,
            format!("layer{l}.w_down[{r},{k}]"),
        ),
        Coord::TokenEmbd(r, k) => (&w.token_embd, r * E + k, format!("token_embd[{r},{k}]")),
    }
}

fn tensor_mut(w: &mut Weights, c: Coord) -> (&mut Vec<i64>, usize) {
    match c {
        Coord::AttnNorm(l, i) => (&mut w.layers[l].attn_norm, i),
        Coord::Wq(l, r, k) => (&mut w.layers[l].wq, r * E + k),
        Coord::Wk(l, r, k) => (&mut w.layers[l].wk, r * E + k),
        Coord::Wv(l, r, k) => (&mut w.layers[l].wv, r * E + k),
        Coord::Wo(l, r, k) => (&mut w.layers[l].wo, r * QW + k),
        Coord::WGate(l, r, k) => (&mut w.layers[l].w_gate, r * E + k),
        Coord::WDown(l, r, k) => (&mut w.layers[l].w_down, r * FF + k),
        Coord::TokenEmbd(r, k) => (&mut w.token_embd, r * E + k),
    }
}

fn element(w: &Weights, c: Coord) -> i64 {
    let (v, i, _) = slot(w, c);
    v[i]
}

fn add_element(w: &mut Weights, c: Coord, d: i64) {
    let (v, i) = tensor_mut(w, c);
    v[i] += d;
}

fn grad_element(g: &Grads, c: Coord) -> i64 {
    match c {
        Coord::AttnNorm(l, i) => g.layers[l].attn_norm[i],
        Coord::Wq(l, r, k) => g.layers[l].wq[r * E + k],
        Coord::Wk(l, r, k) => g.layers[l].wk[r * E + k],
        Coord::Wv(l, r, k) => g.layers[l].wv[r * E + k],
        Coord::Wo(l, r, k) => g.layers[l].wo[r * QW + k],
        Coord::WGate(l, r, k) => g.layers[l].w_gate[r * E + k],
        Coord::WDown(l, r, k) => g.layers[l].w_down[r * FF + k],
        Coord::TokenEmbd(r, k) => g.token_embd[r * E + k],
    }
}

/// Add a deterministic nonzero nudge to every projection and embedding element; the teacher's
/// weights. `salt` separates the run, `amp` is the nudge amplitude in Q32.32 units.
fn perturb_teacher(w: &mut Weights, salt: u64, amp: i64) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ salt.wrapping_mul(0x2545_F491_4F6C_DD1D);
    let span = u64::try_from(2 * amp).expect("positive");
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % span) as i64 - amp
    };
    for lw in &mut w.layers {
        for buf in [&mut lw.wq, &mut lw.wk, &mut lw.wv, &mut lw.wo] {
            for x in buf.iter_mut() {
                *x += next();
            }
        }
        for buf in [&mut lw.w_gate, &mut lw.w_up, &mut lw.w_down] {
            for x in buf.iter_mut() {
                *x += next();
            }
        }
    }
    for x in &mut w.token_embd {
        *x += next();
    }
    for x in &mut w.lm_head {
        *x += next();
    }
}

/// Gate B: the backward's analytic gradient matches a central difference of the summed KL loss at
/// sampled coordinates spanning `wq`, `wk`, `wv`, `wo`, `w_gate`, `w_down`, `attn_norm` and
/// `token_embd`. `H` is `2^-12` of a Q32.32 unit; the tolerance is [`FD_TOL`] Q32.32 units.
#[test]
fn backward_matches_finite_difference() {
    // FD step `2^20` Q32.32 units (`2^-12` of a unit). The forward rounds once per stage, so
    // `L(w)` is a staircase whose step is a few Q32.32 ULPs; over `2H = 2^21` Q32.32 units the
    // central difference averages the staircase to `O(noise·ONE/(2H))`, while the `O(H²)`
    // curvature term stays well below it (the teacher's nudge is `2^26`, so `ΔL ≫` the step).
    const H: i64 = 1 << 20;
    // Tolerance in Q32.32 gradient units, derived: a per-evaluation loss noise of `≤ 8` Q32.32
    // ULPs (one rounding per stage, `T = 4` positions) contributes `8·ONE/(2H) = 8·2^11 = 2^14`
    // after the `ONE/(2H)` rescale; the `O(H²)` truncation and the backward's own per-stage
    // roundings are smaller at this fixture. The measured worst coordinate is `wv` at `7820`
    // (`0.48·2^14`); the mutation control moves a gradient by `O(1)` relative, far above it.
    const FD_TOL: i64 = 1 << 14;

    let (engine, path) = open_fixture("gateB");
    let (w, cfg) = Weights::from_engine(&engine).expect("weights");
    let (_, tape) = forward(&w, &cfg, &TOKENS).expect("forward");

    let mut w_teacher = w.clone();
    perturb_teacher(&mut w_teacher, 1, 1 << 26);
    let (teacher, _) = forward(&w_teacher, &cfg, &TOKENS).expect("teacher");
    let (base_loss, dz) = objective(&w, &cfg, &TOKENS, &teacher);
    assert_ne!(base_loss, 0, "the KL base loss is zero");
    let grads = backward(&w, &cfg, &tape, &dz).expect("backward");

    let coords = [
        Coord::AttnNorm(0, 1),
        Coord::Wq(0, 0, 0),
        Coord::Wk(0, 0, 1),
        Coord::Wv(0, 1, 2),
        Coord::Wo(0, 0, 0),
        Coord::WGate(0, 2, 1),
        Coord::WDown(0, 0, 0),
        Coord::TokenEmbd(TOKENS[0] as usize, 0),
    ];
    for c in coords {
        let (_, _, name) = slot(&w, c);
        let analytic = grad_element(&grads, c);
        let orig = element(&w, c);
        let mut wp = w.clone();
        add_element(&mut wp, c, H);
        let lp = objective(&wp, &cfg, &TOKENS, &teacher).0;
        let mut wm = w.clone();
        add_element(&mut wm, c, -H);
        let lm = objective(&wm, &cfg, &TOKENS, &teacher).0;
        // `lp − lm` is Δ(L·2^32); `H` is the step in `w`; multiplying by `ONE/(2H)` gives
        // `ONE·∂L/∂W`, the convention the backward's gradients are returned in.
        let fd = (lp - lm) as f64 * (ONE as f64) / (2.0 * H as f64);
        let delta = (analytic as f64 - fd).abs();
        eprintln!(
            "{name}: analytic {analytic} fd {fd:.3} delta {delta:.3} (orig {orig}, dL {})",
            lp - lm
        );
        assert!(
            delta <= FD_TOL as f64,
            "{name}: analytic {analytic} vs fd {fd:.4} differ by {delta:.4} > {FD_TOL}"
        );
    }
    let _ = std::fs::remove_file(&path);
}
