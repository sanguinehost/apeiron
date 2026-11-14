//! Browser ABI over apeiron's own inference engine.
//!
//! The arithmetic is the CPU engine's: [`Session`] owns the same `Engine`, `Bpe` and `KvState` and
//! calls the same `Engine::step_logits` / `Engine::step_with_kv` bodies. The only work added here
//! is the Q32.32 → `f32` conversion at the ABI edge.
//!
//! The `#[wasm_bindgen]` surface exists only on `wasm32`; on every other target the crate is an
//! empty rlib so the host workspace build and `cargo test --workspace` link no wasm glue.

/// The default KV capacity when the GGUF states no `context_length`, and the cap on a stated one:
/// a 24-layer, 2-KV-head, 64-wide model holds about 100 MB of Q32.32 keys and values here, which a
/// browser tab can carry. A longer context is a v2 entry point, not a silent allocation.
#[cfg(any(target_arch = "wasm32", test))]
const DEFAULT_KV_CAP: usize = 2048;

#[cfg(any(target_arch = "wasm32", test))]
mod session {
    use apeiron::engine::{Engine, KvState};
    use apeiron::format::bpe::Bpe;

    /// The v1 browser handle: the loaded engine, its tokenizer and one decode cache.
    pub struct Session {
        engine: Engine,
        bpe: Bpe,
        kv: KvState,
    }

    impl Session {
        /// Load an in-memory GGUF and the bytes of its `tokenizer.json`.
        ///
        /// No filesystem is read: the GGUF image and tokenizer document are the caller's buffers.
        /// The KV cache is sized to the file's `context_length`, capped at [`super::DEFAULT_KV_CAP`].
        ///
        /// # Errors
        ///
        /// `Engine::from_bytes`'s config/tensor refusals, `Bpe::from_tokenizer_bytes`'s tokenizer
        /// refusals, and `Engine::decode_state`'s KV geometry refusal, as text.
        pub fn load(gguf: &[u8], tokenizer: &[u8]) -> Result<Session, String> {
            let engine = Engine::from_bytes("wasm", gguf.to_vec()).map_err(|e| e.to_string())?;
            let bpe = Bpe::from_tokenizer_bytes(tokenizer)?;
            let ctx = engine.config.context_length;
            let cap = if ctx == 0 {
                super::DEFAULT_KV_CAP
            } else {
                ctx.min(super::DEFAULT_KV_CAP)
            };
            let kv = engine.decode_state(cap).map_err(|e| e.to_string())?;
            Ok(Session { engine, bpe, kv })
        }

        /// The token ids of `text`.
        ///
        /// # Errors
        ///
        /// `Bpe::encode`'s refusal as text.
        pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
            self.bpe.encode(text).map_err(|e| e.to_string())
        }

        /// One decode step: the Q32.32 logits of `token_id`, with the KV cache advanced by one
        /// position.
        ///
        /// # Errors
        ///
        /// `Engine::step_logits`'s `StepError` as text.
        pub fn step_logits(&mut self, token_id: u32) -> Result<Vec<i64>, String> {
            self.engine
                .step_logits(&mut self.kv, token_id)
                .map_err(|e| e.to_string())
        }

        /// One decode step through the engine's greedy sampler: the chosen token id, with the KV
        /// cache advanced by one position.
        ///
        /// # Errors
        ///
        /// `Engine::step_with_kv`'s `StepError` as text.
        pub fn step_greedy(&mut self, token_id: u32) -> Result<u32, String> {
            self.engine
                .step_with_kv(&mut self.kv, token_id)
                .map(|r| r.token)
                .map_err(|e| e.to_string())
        }

        /// One decode step returning the logits as `f32` for JavaScript.
        ///
        /// # Errors
        ///
        /// As [`Session::step_logits`].
        pub fn step(&mut self, token_id: u32) -> Result<Vec<f32>, String> {
            let logits = self.step_logits(token_id)?;
            Ok(logits.into_iter().map(q32_to_f32).collect())
        }

        /// The text of `ids`.
        ///
        /// # Errors
        ///
        /// `Bpe::decode`'s unknown-id refusal as text.
        pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
            self.bpe.decode(ids).map_err(|e| e.to_string())
        }

        /// Rewind the KV cache and step position to zero. The planes are not zeroed: the next
        /// append overwrites position zero, and causal reads never pass the occupied count.
        pub fn reset(&mut self) {
            // `truncate(0)` refuses only when a layer sits below `0`, impossible for `usize`
            // counters; the reset is total.
            let _ = self.kv.truncate(0);
        }

        /// The KV capacity the handle was loaded with.
        pub fn capacity(&self) -> usize {
            self.kv.cap()
        }
    }

    /// Q32.32 (`i64`, value `v / 2^32`) to the JavaScript number. The conversion is the ABI edge;
    /// the kernel keeps `i64` throughout.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "the ABI returns f32 to JavaScript; the Q32.32 word stays i64 in the kernel, and \
                  the logit magnitude is far inside f32's exponent range"
    )]
    fn q32_to_f32(v: i64) -> f32 {
        (v as f64 / 4_294_967_296.0) as f32
    }
}

#[cfg(any(target_arch = "wasm32", test))]
pub use session::Session;

#[cfg(target_arch = "wasm32")]
mod abi {
    use wasm_bindgen::prelude::*;

    use crate::Session;

    /// The loaded model handle. Every method delegates to [`Session`], which calls the CPU engine's
    /// own bodies; the wrapper adds no arithmetic.
    #[wasm_bindgen]
    pub struct Model {
        session: Session,
    }

    fn js_err(e: String) -> JsValue {
        JsValue::from_str(&e)
    }

    /// Load an in-memory GGUF and tokenizer document. No filesystem is read.
    ///
    /// # Errors
    ///
    /// The engine, tokenizer or KV-geometry refusal as a JavaScript string.
    #[wasm_bindgen]
    pub fn load(gguf: &[u8], tokenizer: &[u8]) -> Result<Model, JsValue> {
        Session::load(gguf, tokenizer)
            .map(|session| Model { session })
            .map_err(js_err)
    }

    #[wasm_bindgen]
    impl Model {
        /// Encode `text` to token ids.
        ///
        /// # Errors
        ///
        /// The tokenizer refusal as a JavaScript string.
        pub fn encode(&self, text: &str) -> Result<Vec<u32>, JsValue> {
            self.session.encode(text).map_err(js_err)
        }

        /// One decode step: the logits as `f32`, KV advanced one position.
        ///
        /// # Errors
        ///
        /// The step refusal as a JavaScript string.
        pub fn step(&mut self, token_id: u32) -> Result<Vec<f32>, JsValue> {
            self.session.step(token_id).map_err(js_err)
        }

        /// One greedy decode step: the chosen token id, KV advanced one position.
        ///
        /// # Errors
        ///
        /// The step refusal as a JavaScript string.
        pub fn step_greedy(&mut self, token_id: u32) -> Result<u32, JsValue> {
            self.session.step_greedy(token_id).map_err(js_err)
        }

        /// Decode token ids to text.
        ///
        /// # Errors
        ///
        /// The tokenizer refusal as a JavaScript string.
        pub fn decode(&self, ids: &[u32]) -> Result<String, JsValue> {
            self.session.decode(ids).map_err(js_err)
        }

        /// Clear the KV cache and the step position.
        pub fn reset(&mut self) {
            self.session.reset();
        }

        /// The KV capacity this handle was loaded with.
        pub fn capacity(&self) -> usize {
            self.session.capacity()
        }
    }
}

#[cfg(test)]
mod tests {
    use apeiron::engine::{argmax_fixed, Engine};
    use apeiron::format::bpe::Bpe;
    use apeiron::format::quant;

    use super::Session;

    // ── synthetic GGUF (the fixture shape of `src/engine/tests.rs`, one layer) ────────────────

    struct Gw(Vec<u8>);

    impl Gw {
        fn new() -> Self {
            Gw(Vec::new())
        }
        fn u8(&mut self, v: u8) {
            self.0.push(v);
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
            self.u64(u64::try_from(s.len()).expect("small"));
            self.bytes(s.as_bytes());
        }
    }

    fn push_t(
        ts: &mut Vec<(String, u32, Vec<u64>, Vec<u8>)>,
        name: String,
        ggml_type: u32,
        dims: Vec<u64>,
        payload: Vec<u8>,
    ) {
        ts.push((name, ggml_type, dims, payload));
    }

    fn d1(n: usize) -> Vec<u64> {
        vec![u64::try_from(n).expect("small")]
    }

    fn d2(a: usize, b: usize) -> Vec<u64> {
        vec![
            u64::try_from(a).expect("small"),
            u64::try_from(b).expect("small"),
        ]
    }

    fn synth_code(r: usize, c: usize, salt: usize) -> i8 {
        let rr = i64::try_from(r).expect("small");
        let cc = i64::try_from(c).expect("small");
        let ss = i64::try_from(salt).expect("small");
        i8::try_from((rr * 3 + cc * 5 + ss) % 5 - 2).expect("in [-2, 2]")
    }

    /// One `Q8_0` weight of `n_out` rows over `n_in = 32`.
    fn synth_q8_weight(n_out: usize, d_bits: u16, salt: usize) -> Vec<u8> {
        const N_IN: usize = 32;
        let mut out: Vec<u8> = Vec::with_capacity(n_out * (2 + N_IN));
        let mut r = 0usize;
        while r < n_out {
            out.extend_from_slice(&d_bits.to_le_bytes());
            let mut c = 0usize;
            while c < N_IN {
                out.push(synth_code(r, c, salt).to_le_bytes()[0]);
                c += 1;
            }
            r += 1;
        }
        out
    }

    fn synth_f32_vec(n: usize, f: impl Fn(usize) -> f32) -> Vec<u8> {
        let mut b: Vec<u8> = Vec::with_capacity(n * 4);
        let mut i = 0usize;
        while i < n {
            b.extend_from_slice(&f(i).to_le_bytes());
            i += 1;
        }
        b
    }

    /// A one-layer dense `qwen2` GGUF: `n_embd 32`, `n_ff 32`, 4 query heads over 2 KV heads of
    /// width 8, vocabulary 16.
    fn synth_gguf_bytes() -> Vec<u8> {
        const NE: usize = 32;
        const NF: usize = 32;
        const NH: usize = 4;
        const NKV: usize = 2;
        const HD: usize = 8;
        const V: usize = 16;
        const Q_OUT: usize = NH * HD;
        const KV_OUT: usize = NKV * HD;
        const D_PROJ: u16 = 0x2800; // f16 2^-5
        const D_EMB: u16 = 0x3C00; // f16 1.0
        let gain = |_: usize| 1.0_f32;
        let bias = |i: usize| match i % 3 { 0 => -1.0_f32, 1 => 0.0, _ => 1.0 } / 64.0;

        let mut ts: Vec<(String, u32, Vec<u64>, Vec<u8>)> = Vec::new();
        push_t(
            &mut ts,
            "blk.0.attn_norm.weight".to_string(),
            quant::GGML_TYPE_F32,
            d1(NE),
            synth_f32_vec(NE, gain),
        );
        push_t(
            &mut ts,
            "blk.0.attn_q.weight".to_string(),
            quant::GGML_TYPE_Q8_0,
            d2(NE, Q_OUT),
            synth_q8_weight(Q_OUT, D_PROJ, 1),
        );
        push_t(
            &mut ts,
            "blk.0.attn_q.bias".to_string(),
            quant::GGML_TYPE_F32,
            d1(Q_OUT),
            synth_f32_vec(Q_OUT, bias),
        );
        push_t(
            &mut ts,
            "blk.0.attn_k.weight".to_string(),
            quant::GGML_TYPE_Q8_0,
            d2(NE, KV_OUT),
            synth_q8_weight(KV_OUT, D_PROJ, 2),
        );
        push_t(
            &mut ts,
            "blk.0.attn_k.bias".to_string(),
            quant::GGML_TYPE_F32,
            d1(KV_OUT),
            synth_f32_vec(KV_OUT, bias),
        );
        push_t(
            &mut ts,
            "blk.0.attn_v.weight".to_string(),
            quant::GGML_TYPE_Q8_0,
            d2(NE, KV_OUT),
            synth_q8_weight(KV_OUT, D_PROJ, 3),
        );
        push_t(
            &mut ts,
            "blk.0.attn_v.bias".to_string(),
            quant::GGML_TYPE_F32,
            d1(KV_OUT),
            synth_f32_vec(KV_OUT, bias),
        );
        push_t(
            &mut ts,
            "blk.0.attn_output.weight".to_string(),
            quant::GGML_TYPE_Q8_0,
            d2(NE, NE),
            synth_q8_weight(NE, D_PROJ, 4),
        );
        push_t(
            &mut ts,
            "blk.0.ffn_norm.weight".to_string(),
            quant::GGML_TYPE_F32,
            d1(NE),
            synth_f32_vec(NE, gain),
        );
        push_t(
            &mut ts,
            "blk.0.ffn_gate.weight".to_string(),
            quant::GGML_TYPE_Q8_0,
            d2(NE, NF),
            synth_q8_weight(NF, D_PROJ, 5),
        );
        push_t(
            &mut ts,
            "blk.0.ffn_up.weight".to_string(),
            quant::GGML_TYPE_Q8_0,
            d2(NE, NF),
            synth_q8_weight(NF, D_PROJ, 6),
        );
        push_t(
            &mut ts,
            "blk.0.ffn_down.weight".to_string(),
            quant::GGML_TYPE_Q8_0,
            d2(NF, NE),
            synth_q8_weight(NE, D_PROJ, 7),
        );
        push_t(
            &mut ts,
            "output_norm.weight".to_string(),
            quant::GGML_TYPE_F32,
            d1(NE),
            synth_f32_vec(NE, gain),
        );
        push_t(
            &mut ts,
            "token_embd.weight".to_string(),
            quant::GGML_TYPE_Q8_0,
            d2(NE, V),
            synth_q8_weight(V, D_EMB, 9),
        );

        let mut h = Gw::new();
        h.bytes(b"GGUF");
        h.u32(3);
        h.u64(u64::try_from(ts.len()).expect("small"));
        h.u64(8);
        h.string("general.architecture");
        h.u32(8);
        h.string("qwen2");
        for (k, v) in [
            ("qwen2.block_count", u32::try_from(1).expect("small")),
            (
                "qwen2.attention.head_count",
                u32::try_from(NH).expect("small"),
            ),
            (
                "qwen2.attention.head_count_kv",
                u32::try_from(NKV).expect("small"),
            ),
            ("qwen2.embedding_length", u32::try_from(NE).expect("small")),
            (
                "qwen2.feed_forward_length",
                u32::try_from(NF).expect("small"),
            ),
        ] {
            h.string(k);
            h.u32(4);
            h.u32(v);
        }
        h.string("qwen2.rope.freq_base");
        h.u32(6);
        h.u32(10_000.0_f32.to_bits());
        h.string("qwen2.attention.layer_norm_rms_epsilon");
        h.u32(6);
        h.u32(1e-5_f32.to_bits());

        let mut off = 0u64;
        for (name, ty, dims, payload) in &ts {
            h.string(name);
            h.u32(u32::try_from(dims.len()).expect("small"));
            for &d in dims {
                h.u64(d);
            }
            h.u32(*ty);
            h.u64(off);
            off += u64::try_from(payload.len()).expect("small");
        }
        let header_len = h.0.len();
        let pad = (32 - header_len % 32) % 32;
        let mut i = 0usize;
        while i < pad {
            h.u8(0);
            i += 1;
        }
        for (_, _, _, payload) in &ts {
            h.bytes(payload);
        }
        h.0
    }

    /// `a..p` map to ids `0..16`; no merges. Enough to round-trip the two-letter probe and to
    /// prove the ABI reads the tokenizer bytes rather than a path.
    fn synth_tokenizer_json() -> Vec<u8> {
        let mut vocab = String::new();
        let mut i = 0usize;
        while i < 16 {
            if i > 0 {
                vocab.push(',');
            }
            let c = char::from(b'a' + u8::try_from(i).expect("small"));
            vocab.push('"');
            vocab.push(c);
            vocab.push_str("\":");
            vocab.push_str(&i.to_string());
            i += 1;
        }
        format!(
            r#"{{"pre_tokenizer":{{"type":"Sequence","pretokenizers":[{{"type":"Split","pattern":{{"Regex":"\\p{{N}}"}},"behavior":"Isolated"}},{{"type":"ByteLevel","use_regex":false}}]}},
"model":{{"type":"BPE","byte_fallback":false,"vocab":{{{vocab}}},"merges":[]}}}}"#
        )
        .into_bytes()
    }

    #[test]
    fn session_matches_the_engine_bodies_and_round_trips() {
        let gguf = synth_gguf_bytes();
        let tok = synth_tokenizer_json();

        // The direct engine path: the same loader and the same step bodies the ABI claims.
        let engine = Engine::from_bytes("parity", gguf.clone()).expect("direct engine");
        let bpe = Bpe::from_tokenizer_bytes(&tok).expect("direct tokenizer");
        let mut kv = engine.decode_state(8).expect("direct kv");

        let mut session = Session::load(&gguf, &tok).expect("session load");
        assert!(session.capacity() >= 1);

        // Encode/decode through the handle and through the direct tokenizer body.
        let ids = session.encode("ab").expect("encode");
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(bpe.encode("ab").expect("direct encode"), ids);
        assert_eq!(session.decode(&ids).expect("decode"), "ab");
        assert_eq!(bpe.decode(&ids).expect("direct decode"), "ab");

        // Two forward steps: the wrapper's logits equal the direct `step_logits` word for word.
        let t0 = 3u32;
        let s0 = session.step_logits(t0).expect("session step 0");
        let e0 = engine.step_logits(&mut kv, t0).expect("engine step 0");
        assert_eq!(s0, e0, "first step: wrapper and engine disagree");
        let t1 = argmax_fixed(&e0).expect("greedy token");
        let s1 = session.step_logits(t1).expect("session step 1");
        let e1 = engine.step_logits(&mut kv, t1).expect("engine step 1");
        assert_eq!(s1, e1, "second step: wrapper and engine disagree");

        // The `f32` ABI edge is the same Q32.32 words divided by 2^32.
        let mut session_f32 = Session::load(&gguf, &tok).expect("session f32");
        let f = session_f32.step(t0).expect("f32 step");
        assert_eq!(f.len(), s0.len());
        let mut i = 0usize;
        while i < f.len() {
            let want = (s0[i] as f64 / 4_294_967_296.0) as f32;
            assert_eq!(f[i], want, "f32 logit {i}");
            i += 1;
        }

        // Two greedy steps through `step_with_kv`, the body behind `step_greedy`.
        let engine_g = Engine::from_bytes("parity-greedy", gguf.clone()).expect("greedy engine");
        let mut kv_g = engine_g.decode_state(8).expect("greedy kv");
        let mut session_g = Session::load(&gguf, &tok).expect("greedy session");
        let mut tok_id = t0;
        let mut step = 0usize;
        while step < 2 {
            let direct = engine_g
                .step_with_kv(&mut kv_g, tok_id)
                .expect("direct greedy step");
            let wrapped = session_g.step_greedy(tok_id).expect("wrapped greedy step");
            assert_eq!(wrapped, direct.token, "greedy step {step}");
            tok_id = wrapped;
            step += 1;
        }

        // `reset` rewinds to position zero: the first step repeats exactly.
        session.reset();
        let after_reset = session.step_logits(t0).expect("step after reset");
        assert_eq!(after_reset, s0, "reset did not rewind the KV cache");
    }
}
