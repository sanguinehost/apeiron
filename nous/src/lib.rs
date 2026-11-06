#![forbid(unsafe_code)]
//! `nous`: a loopback OpenAI-compatible provider for the apeiron engine.
//!
//! The library owns the HTTP surface and the provider object; `nous-serve` is the CLI wrapper.
//! It is deliberately a separate package from `apeiron`: the engine has no listener, and this
//! crate is the opt-in local adapter for standard AI clients.
//!
//! - Loopback only: a non-loopback `--addr` is refused. There is no authentication and no TLS;
//!   the process is a local tool, not a service.
//! - One request at a time: the engine call is blocking and the next connection waits in the
//!   listen queue. A client that disconnects mid-stream ends that request at the next write.
//! - Decoding is greedy (`argmax`); `temperature`, `top_p`, `tools` and `stop` in the request are
//!   ignored. The model answers in text only, so a client that needs tool calls gets none.
//! - The prompt is ChatML (`<|im_start|>role\n…<|im_end|>\n` then `<|im_start|>assistant\n`).
//!   A base model was not trained on it.
//! - Stops at `<|im_end|>`, `<|endoftext|>` or the tokenizer's EOS, at `max_tokens` (default 512),
//!   or when the context `--ctx` is full (`finish_reason` `length`).
//!
//! Cost per request: prefill P prompt tokens, decode at most `n` more: P + n engine steps and
//! P + n KV positions, `P + n <= --ctx`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::bpe::Bpe;
use serde_json::{json, Value};

pub mod fold_image;

pub struct Provider {
    engine: Engine,
    bpe: Bpe,
    name: String,
    ctx: usize,
    mode: KernelMode,
    stops: Vec<u32>,
}

fn single_token(bpe: &Bpe, s: &str) -> Option<u32> {
    match bpe.encode(s) {
        Ok(v) if v.len() == 1 => Some(v[0]),
        _ => None,
    }
}

/// ChatML of `messages`; an array `content` contributes its `text` parts.
pub fn chatml(messages: &[Value]) -> Result<String, String> {
    let mut s = String::new();
    for m in messages {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or("a message has no role")?;
        let role = match role {
            "developer" => "system",
            r => r,
        };
        let content = match m.get("content") {
            Some(Value::String(t)) => t.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
            Some(Value::Null) | None => String::new(),
            Some(_) => return Err("message content is neither a string nor an array".into()),
        };
        s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    s.push_str("<|im_start|>assistant\n");
    Ok(s)
}

pub struct Done {
    pub produced: usize,
    pub reason: &'static str,
    pub decode_s: f64,
}

impl Provider {
    /// Open a model, optionally with a fold pack. The tokenizer is the Qwen-family BPE JSON.
    pub fn open(
        model: &str,
        fold: &str,
        tokenizer: &str,
        name: String,
        ctx: usize,
        mode: KernelMode,
    ) -> Result<Self, String> {
        let bpe = Bpe::from_tokenizer_json(tokenizer).map_err(|e| format!("{tokenizer}: {e:?}"))?;
        let engine = if fold.is_empty() {
            Engine::open(model).map_err(|e| format!("{model}: {e:?}"))?
        } else {
            fold_image::load(model, fold)?
        };
        if !fold.is_empty() {
            let (bytes, weights) = engine.fold_storage();
            eprintln!(
                "nous: fold {} faces {:.1} MiB / {} weights = {:.4} bits/weight",
                fold,
                bytes as f64 / 1048576.0,
                weights,
                if weights == 0 {
                    0.0
                } else {
                    bytes as f64 * 8.0 / weights as f64
                }
            );
        }
        let mut stops: Vec<u32> = bpe.eos_id().into_iter().collect();
        for t in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(id) = single_token(&bpe, t) {
                if !stops.contains(&id) {
                    stops.push(id);
                }
            }
        }
        Ok(Self {
            engine,
            bpe,
            name,
            ctx,
            mode,
            stops,
        })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn context(&self) -> usize {
        self.ctx
    }

    /// Greedy generation; `emit` receives each complete UTF-8 piece in order.
    pub fn generate(
        &self,
        ids: &[u32],
        max_new: usize,
        mut emit: impl FnMut(&str) -> std::io::Result<()>,
    ) -> Result<Done, String> {
        let mut st = self
            .engine
            .state(ids.len() + max_new)
            .map_err(|e| format!("{e:?}"))?;
        let mut next = self
            .engine
            .prefill_state(&mut st, ids, self.mode)
            .map_err(|e| format!("{e:?}"))?;
        let mut pending: Vec<u32> = Vec::new();
        let (mut produced, mut reason) = (0usize, "length");
        let t_dec = Instant::now();
        while produced < max_new {
            if self.stops.contains(&next) {
                reason = "stop";
                break;
            }
            pending.push(next);
            produced += 1;
            let piece = self.bpe.decode(&pending).map_err(|e| format!("{e:?}"))?;
            if !piece.ends_with('\u{FFFD}') {
                emit(&piece).map_err(|e| format!("client: {e}"))?;
                pending.clear();
            }
            if produced < max_new {
                next = self
                    .engine
                    .next_token_state(&mut st, next, self.mode)
                    .map_err(|e| format!("{e:?}"))?;
            }
        }
        if !pending.is_empty() {
            let piece = self.bpe.decode(&pending).map_err(|e| format!("{e:?}"))?;
            emit(&piece).map_err(|e| format!("client: {e}"))?;
        }
        Ok(Done {
            produced,
            reason,
            decode_s: t_dec.elapsed().as_secs_f64(),
        })
    }
}

/// CORS is intentionally permissive for loopback-only local tools. The listener refuses a
/// non-loopback address, so this does not expose the engine to a network.
const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: authorization, content-type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

fn respond(s: &mut TcpStream, code: u16, status: &str, body: &Value) {
    let b = body.to_string();
    let _ = write!(
        s,
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json; charset=utf-8\r\n{CORS}Cache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
        b.len()
    );
}

fn err(s: &mut TcpStream, code: u16, status: &str, msg: &str) {
    respond(
        s,
        code,
        status,
        &json!({"error": {"message": msg, "type": "invalid_request_error"}}),
    );
}

fn preflight(s: &mut TcpStream) {
    let _ = write!(
        s,
        "HTTP/1.1 204 No Content\r\n{CORS}Content-Length: 0\r\nConnection: close\r\n\r\n"
    );
}

/// OpenAI model object. `owned_by` is the provider name, not the derived model name.
pub fn model_card(name: &str, ctx: usize) -> Value {
    json!({
        "id": name,
        "object": "model",
        "created": 0,
        "owned_by": "nous",
        "context_window": ctx,
    })
}

/// OpenAI `/v1/models` list.
pub fn models_json(name: &str, ctx: usize) -> Value {
    json!({"object": "list", "data": [model_card(name, ctx)]})
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn handle(srv: &Provider, mut s: TcpStream) -> Result<(), String> {
    let mut r = BufReader::new(s.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    r.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let (method, path) = (
        method.to_string(),
        path.split('?').next().unwrap_or("").to_string(),
    );
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).map_err(|e| e.to_string())? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v
                    .trim()
                    .parse()
                    .map_err(|_| "bad content-length".to_string())?;
            }
        }
    }
    if method == "OPTIONS" {
        preflight(&mut s);
        return Ok(());
    }
    if len > 8 << 20 {
        err(&mut s, 413, "Payload Too Large", "request over 8 MiB");
        return Ok(());
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).map_err(|e| e.to_string())?;
    let route = path.trim_start_matches("/v1");
    if method == "GET" {
        if route == "/models" {
            respond(&mut s, 200, "OK", &models_json(&srv.name, srv.ctx));
            return Ok(());
        }
        if let Some(id) = route.strip_prefix("/models/") {
            if id == srv.name {
                respond(&mut s, 200, "OK", &model_card(&srv.name, srv.ctx));
            } else {
                err(&mut s, 404, "Not Found", "unknown model");
            }
            return Ok(());
        }
    }
    match (method.as_str(), route) {
        ("GET", "/health") | ("GET", "/") => respond(&mut s, 200, "OK", &json!({"status": "ok"})),
        ("POST", "/chat/completions") => chat(srv, &mut s, &body)?,
        _ => err(
            &mut s,
            404,
            "Not Found",
            &format!("no route {method} {path}"),
        ),
    }
    Ok(())
}

fn chat(srv: &Provider, s: &mut TcpStream, body: &[u8]) -> Result<(), String> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            err(s, 400, "Bad Request", &format!("invalid JSON: {e}"));
            return Ok(());
        }
    };
    let Some(messages) = req.get("messages").and_then(Value::as_array) else {
        err(s, 400, "Bad Request", "messages is required");
        return Ok(());
    };
    let text = match chatml(messages) {
        Ok(t) => t,
        Err(m) => {
            err(s, 400, "Bad Request", &m);
            return Ok(());
        }
    };
    let ids = srv.bpe.encode(&text).map_err(|e| format!("{e:?}"))?;
    let vocab = srv.engine.config.vocab_size;
    if ids.is_empty() || ids.iter().any(|&i| i as usize >= vocab) {
        err(
            s,
            400,
            "Bad Request",
            "the prompt is empty or holds an id outside the vocabulary",
        );
        return Ok(());
    }
    if ids.len() + 1 > srv.ctx {
        err(
            s,
            400,
            "Bad Request",
            &format!("prompt is {} tokens; the context is {}", ids.len(), srv.ctx),
        );
        return Ok(());
    }
    let want = ["max_completion_tokens", "max_tokens"]
        .iter()
        .find_map(|k| req.get(*k).and_then(Value::as_u64))
        .map_or(512, |v| v as usize);
    let max_new = want.min(srv.ctx - ids.len()).max(1);
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let (id, created) = (format!("chatcmpl-{}", now()), now());
    let t0 = Instant::now();
    eprintln!(
        "nous: chat prompt {} tokens, max_new {max_new}, stream {stream}",
        ids.len()
    );

    if stream {
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n{CORS}\r\n"
        );
        s.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
        let chunk = |s: &mut TcpStream, delta: Value, fin: Value| -> std::io::Result<()> {
            let v = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": srv.name,
                           "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]});
            write!(s, "data: {v}\n\n")?;
            s.flush()
        };
        chunk(s, json!({"role": "assistant", "content": ""}), Value::Null)
            .map_err(|e| e.to_string())?;
        let d = srv.generate(&ids, max_new, |p| {
            chunk(s, json!({"content": p}), Value::Null)
        })?;
        chunk(s, json!({}), json!(d.reason)).map_err(|e| e.to_string())?;
        let usage = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": srv.name,
            "choices": [], "usage": {"prompt_tokens": ids.len(), "completion_tokens": d.produced, "total_tokens": ids.len() + d.produced}});
        write!(s, "data: {usage}\n\ndata: [DONE]\n\n").map_err(|e| e.to_string())?;
        s.flush().map_err(|e| e.to_string())?;
        eprintln!(
            "nous: done {} tokens in {:.1}s ({}), decode {:.2}s = {:.1} ms/token",
            d.produced,
            t0.elapsed().as_secs_f64(),
            d.reason,
            d.decode_s,
            1e3 * d.decode_s / d.produced.max(1) as f64
        );
    } else {
        let mut out = String::new();
        let d = srv.generate(&ids, max_new, |p| {
            out.push_str(p);
            Ok(())
        })?;
        let v = json!({"id": id, "object": "chat.completion", "created": created, "model": srv.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": d.reason}],
            "usage": {"prompt_tokens": ids.len(), "completion_tokens": d.produced, "total_tokens": ids.len() + d.produced}});
        respond(s, 200, "OK", &v);
        eprintln!(
            "nous: done {} tokens in {:.1}s ({}), decode {:.2}s = {:.1} ms/token",
            d.produced,
            t0.elapsed().as_secs_f64(),
            d.reason,
            d.decode_s,
            1e3 * d.decode_s / d.produced.max(1) as f64
        );
    }
    Ok(())
}

/// Run the provider from CLI arguments. The binary is a thin wrapper around this function.
///
/// # Errors
/// Missing paths, an unknown flag, a non-loopback address, or an engine/tokenizer load failure.
pub fn run() -> Result<(), String> {
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        println!(
            "nous-serve --model <gguf> --tokenizer <tokenizer.json> [--fold <pack.fold>]\n\
             \x20          [--addr 127.0.0.1:8095] [--name qwen2.5-7b-e8-exact]\n\
             \x20          [--ctx 4096] [--kernel exact|q8act]\n\n\
             OpenAI-compatible loopback provider. Default kernel is exact.\n\
             q8act is fast and quality-broken; use it only for plumbing."
        );
        return Ok(());
    }
    let (mut model, mut tok, mut fold) = (String::new(), String::new(), String::new());
    let (mut addr, mut name, mut ctx) = (
        "127.0.0.1:8095".to_string(),
        "qwen2.5-7b-e8-exact".to_string(),
        4096usize,
    );
    let mut mode = KernelMode::Exact;
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{f} needs a value"));
        match f.as_str() {
            "--model" => model = v()?,
            "--tokenizer" => tok = v()?,
            "--fold" => fold = v()?,
            "--addr" => addr = v()?,
            "--name" => name = v()?,
            "--ctx" => ctx = v()?.parse().map_err(|e| format!("--ctx: {e}"))?,
            "--kernel" => {
                mode = match v()?.as_str() {
                    "q8act" => KernelMode::Q8Act,
                    "exact" => KernelMode::Exact,
                    o => return Err(format!("--kernel is q8act|exact, got {o:?}")),
                }
            }
            o => return Err(format!("unknown argument {o:?}")),
        }
    }
    if model.is_empty() || tok.is_empty() {
        return Err("--model and --tokenizer are required".into());
    }
    let sock: std::net::SocketAddr = addr.parse().map_err(|e| format!("--addr: {e}"))?;
    if !sock.ip().is_loopback() {
        return Err(format!(
            "--addr {sock} is not loopback; this server has no authentication"
        ));
    }
    let srv = Provider::open(&model, &fold, &tok, name, ctx, mode)?;
    let listener = TcpListener::bind(sock).map_err(|e| format!("bind {sock}: {e}"))?;
    eprintln!(
        "nous: {model} on http://{sock}/v1 as {:?}, ctx {}, stops {:?}",
        srv.name(),
        srv.context(),
        srv.stops
    );
    for conn in listener.incoming() {
        match conn {
            Ok(s) => {
                if let Err(e) = handle(&srv, s) {
                    eprintln!("nous: request failed: {e}");
                }
            }
            Err(e) => eprintln!("nous: accept: {e}"),
        }
    }
    Ok(())
}
