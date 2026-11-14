//! An OpenAI-compatible `/v1/chat/completions` endpoint over the apeiron engine.
//!
//! This is the embedding application's listener: the apeiron crate itself opens
//! no socket (`SECURITY.md`, "No network listener"). It exists so Intent's
//! `local` provider (`crates/intent-agent/src/providers/local.rs` in the intent
//! repository) can point at a full Qwen3 pack served from this machine.
//!
//! Run:
//! `cargo run --release -- --model skeleton.gguf --pack model.nqp \
//!    --tokenizer tokenizer.json --bind 127.0.0.1:8080`
//!
//! Supported routes:
//! - `POST /v1/chat/completions` — `messages` in, greedy completion out; `stream`
//!   selects SSE (`data: {...}` … `data: [DONE]`) or one JSON body.
//! - `GET /v1/models` — the served model id, for clients that probe it.
//! - `GET /health` — `ok`.
//!
//! Domain: greedy sampling only (`temperature`, `top_p` and `tools` are
//! accepted and ignored); a request's decoded output is capped by
//! `--max-tokens` regardless of the ceiling the client sends, because the KV
//! state is allocated up front. One request is decoded at a time.

use std::error::Error;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::ExitCode;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::bpe::Bpe;
use apeiron::format::nq;
use serde_json::{json, Value};

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

struct Args {
    model: String,
    pack: Option<String>,
    tokenizer: String,
    bind: String,
    model_name: String,
    kernel: KernelMode,
    max_tokens: usize,
    think: bool,
    warmup: bool,
    /// Serve only the conversation, not Intent's assembled system prompt. The
    /// loop and its transport stay real; the prompt the model reads is the
    /// user's text alone. A demo for a small KV/memory budget, not the default.
    drop_system: bool,
    /// End the turn at the first `.`, `!`, `?` or newline. The base checkpoint
    /// does not emit `<|im_end|>` on its own, and a length cut is a turn error
    /// in Intent; this is an explicit stop rule for the hello demo.
    stop_at_sentence: bool,
    /// Replaces every system/developer message with this one line. The compact
    /// counterpart of `--drop-system`: the model still gets an instruction, not
    /// Intent's 2.6k-token assembled prompt.
    system: Option<String>,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut a = Args {
        model: String::new(),
        pack: None,
        tokenizer: String::new(),
        bind: "127.0.0.1:8080".to_string(),
        model_name: "nq-qwen3".to_string(),
        kernel: KernelMode::Q8Act,
        max_tokens: 256,
        think: false,
        warmup: true,
        drop_system: false,
        stop_at_sentence: false,
        system: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--pack" => a.pack = Some(val()?),
            "--tokenizer" => a.tokenizer = val()?,
            "--bind" => a.bind = val()?,
            "--model-name" => a.model_name = val()?,
            "--max-tokens" => a.max_tokens = val()?.parse()?,
            "--kernel" => {
                a.kernel = match val()?.as_str() {
                    "exact" => KernelMode::Exact,
                    "q8act" => KernelMode::Q8Act,
                    other => return Err(format!("--kernel is q8act|exact, got {other:?}").into()),
                }
            }
            "--think" => a.think = true,
            "--no-warmup" => a.warmup = false,
            "--drop-system" => a.drop_system = true,
            "--stop-at-sentence" => a.stop_at_sentence = true,
            "--system" => a.system = Some(val()?),
            "-h" | "--help" => {
                println!(
                    "nq-serve --model <gguf> [--pack <nqp>] --tokenizer <json> \
                     [--bind 127.0.0.1:8080] [--model-name nq-qwen3] \
                     [--max-tokens 256] [--kernel q8act|exact] [--think] [--no-warmup] \
                     [--drop-system] [--stop-at-sentence] [--system TEXT]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if a.model.is_empty() || a.tokenizer.is_empty() {
        return Err("--model and --tokenizer are required".into());
    }
    if a.max_tokens == 0 {
        return Err("--max-tokens must be positive".into());
    }
    Ok(a)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("nq-serve: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let t0 = Instant::now();
    let bpe = Bpe::from_tokenizer_json(&args.tokenizer)?;
    let engine = match &args.pack {
        Some(path) => {
            let pack = nq::open(path).map_err(|e| e.to_string())?;
            eprintln!(
                "pack          {} linears, {:.6} b/w",
                pack.linears.len(),
                pack.bits_per_weight().unwrap_or(0.0)
            );
            Engine::open_with_nq(&args.model, pack)?
        }
        None => Engine::open(&args.model)?,
    };
    eprintln!(
        "loaded        {} layers, vocab {}, {:.3} s",
        engine.config.n_layer,
        engine.config.vocab_size,
        t0.elapsed().as_secs_f64()
    );
    if args.warmup {
        warm(&engine, &bpe, args.kernel)?;
    }
    let listener = TcpListener::bind(&args.bind)?;
    eprintln!(
        "listening     http://{}/v1  model={} kernel={:?} max_tokens={} think={}",
        args.bind, args.model_name, args.kernel, args.max_tokens, args.think
    );
    let stop = stop_ids(&bpe);
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                if let Err(e) = serve_one(&engine, &bpe, &args, &stop, &mut stream) {
                    eprintln!("request       failed: {e}");
                }
            }
            Err(e) => eprintln!("accept        failed: {e}"),
        }
    }
    Ok(())
}

/// Read every projection once so the first real request finds the pack in page
/// cache; `--no-warmup` leaves that cost on the first request.
fn warm(engine: &Engine, bpe: &Bpe, mode: KernelMode) -> Result<(), Box<dyn Error>> {
    // More than one token so the multi-column projections and their F16 tile
    // builds are paid before the first client request, not inside it.
    let ids = bpe.encode("Hello, how are you today?")?;
    let t0 = Instant::now();
    let mut st = engine.state(ids.len() + 1)?;
    let _ = engine.prefill_state(&mut st, &ids, mode)?;
    eprintln!("warmup        {:.3} s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// Every end-of-turn id this tokenizer declares. A model that emits any of them
/// has finished the assistant turn, and the id is never streamed as text.
fn stop_ids(bpe: &Bpe) -> Vec<u32> {
    let mut ids = Vec::new();
    for name in ["<|im_end|>", "<|endoftext|>", "<|end|>", "</s>"] {
        if let Some(id) = bpe.special_id(name) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn serve_one(
    engine: &Engine,
    bpe: &Bpe,
    args: &Args,
    stop: &[u32],
    stream: &mut TcpStream,
) -> Result<(), Box<dyn Error>> {
    let Some(req) = read_request(stream)? else {
        return Ok(());
    };
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => write_plain(stream, 200, "ok\n"),
        ("GET", "/v1/models") => {
            let body = json!({
                "object": "list",
                "data": [{
                    "id": args.model_name,
                    "object": "model",
                    "created": epoch_secs(),
                    "owned_by": "local",
                }]
            });
            write_json(stream, 200, &body)
        }
        ("POST", "/v1/chat/completions") => chat(engine, bpe, args, stop, stream, &req.body),
        _ => write_plain(stream, 404, "not found\n"),
    }
}

fn chat(
    engine: &Engine,
    bpe: &Bpe,
    args: &Args,
    stop: &[u32],
    stream: &mut TcpStream,
    body: &[u8],
) -> Result<(), Box<dyn Error>> {
    let v: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return write_error(stream, 400, &format!("request body is not JSON: {e}")),
    };
    let model = v
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&args.model_name)
        .to_string();
    let Some(messages) = v.get("messages").and_then(Value::as_array) else {
        return write_error(stream, 400, "messages must be an array");
    };
    let mut messages: Vec<Value> = if args.drop_system {
        messages
            .iter()
            .filter(|m| {
                !matches!(
                    m.get("role").and_then(Value::as_str),
                    Some("system" | "developer")
                )
            })
            .cloned()
            .collect()
    } else {
        messages.clone()
    };
    if let Some(system) = &args.system {
        messages.retain(|m| {
            !matches!(
                m.get("role").and_then(Value::as_str),
                Some("system" | "developer")
            )
        });
        messages.insert(0, json!({"role": "system", "content": system}));
    }
    let want_stream = v.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let requested = v
        .get("max_tokens")
        .and_then(Value::as_u64)
        .or_else(|| v.get("max_completion_tokens").and_then(Value::as_u64))
        .unwrap_or(args.max_tokens as u64);
    let max_new = usize::try_from(requested)
        .unwrap_or(args.max_tokens)
        .min(args.max_tokens)
        .max(1);

    let prompt = match render_chatml(&messages, args.think) {
        Ok(p) => p,
        Err(e) => return write_error(stream, 400, &e),
    };
    let ids = match bpe.encode(&prompt) {
        Ok(ids) => ids,
        Err(e) => return write_error(stream, 400, &format!("tokenize: {e}")),
    };
    if ids.is_empty() {
        return write_error(stream, 400, "the rendered prompt is empty");
    }
    let vocab = engine.config.vocab_size;
    if let Some(&bad) = ids.iter().find(|&&id| id as usize >= vocab) {
        return write_error(
            stream,
            400,
            &format!("prompt id {bad} >= model vocabulary {vocab}"),
        );
    }
    eprintln!(
        "request       prompt {} tok ({} messages)",
        ids.len(),
        messages.len()
    );

    // The KV state is allocated for prompt + output up front, so the served
    // ceiling is the smaller of the client's and `--max-tokens`.
    let mut st = match engine.state(ids.len() + max_new + 8) {
        Ok(st) => st,
        Err(e) => return write_error(stream, 500, &format!("state: {e}")),
    };

    let id = format!("chatcmpl-{}-{}", epoch_secs(), std::process::id());
    let created = epoch_secs();
    if want_stream {
        write_stream_headers(stream)?;
    }

    let t0 = Instant::now();
    let mut next = match engine.prefill_state(&mut st, &ids, args.kernel) {
        Ok(t) => t,
        Err(e) => {
            if want_stream {
                let _ = write_sse(
                    stream,
                    &json!({"error": {"message": format!("prefill: {e}"), "type": "engine_error"}}),
                );
                let _ = stream.write_all(b"data: [DONE]\n\n");
                return Ok(());
            }
            return write_error(stream, 500, &format!("prefill: {e}"));
        }
    };
    let prefill = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    if want_stream {
        // After prefill, so a client's time-to-first-event is the time to the
        // first generated token, not to an empty role chunk.
        write_sse(
            stream,
            &json!({
                "id": id, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            }),
        )?;
    }

    let mut text = String::new();
    let mut pending: Vec<u32> = Vec::new();
    let mut produced = 0usize;
    let mut finish = "length";
    // The first token came out of prefill; later ones each take a full step.
    let mut current = Some(next);
    while let Some(token) = current.take() {
        if stop.contains(&token) {
            finish = "stop";
            break;
        }
        pending.push(token);
        produced += 1;
        let piece = bpe.decode(&pending)?;
        // A token can end mid-character; hold ids until the bytes decode.
        if !piece.ends_with('\u{FFFD}') {
            let mut emit = piece.as_str();
            let mut stop_here = false;
            if args.stop_at_sentence {
                if let Some(i) = piece.find(['.', '!', '?', '\n']) {
                    emit = &piece[..=i];
                    stop_here = true;
                }
            }
            if want_stream {
                write_sse(stream, &delta_chunk(&id, created, &model, emit, None))?;
            }
            text.push_str(emit);
            pending.clear();
            if stop_here {
                finish = "stop";
                break;
            }
        }
        if produced >= max_new {
            break;
        }
        next = match engine.next_token_state(&mut st, token, args.kernel) {
            Ok(t) => t,
            Err(e) => {
                if want_stream {
                    let _ = write_sse(
                        stream,
                        &json!({"error": {"message": format!("step: {e}"), "type": "engine_error"}}),
                    );
                    let _ = stream.write_all(b"data: [DONE]\n\n");
                    return Ok(());
                }
                return write_error(stream, 500, &format!("step: {e}"));
            }
        };
        current = Some(next);
    }
    if !pending.is_empty() {
        let piece = bpe.decode(&pending)?;
        if want_stream {
            write_sse(stream, &delta_chunk(&id, created, &model, &piece, None))?;
        }
        text.push_str(&piece);
    }
    let decode = t1.elapsed().as_secs_f64();

    if want_stream {
        let done = json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": ids.len(),
                "completion_tokens": produced,
                "total_tokens": ids.len() + produced,
            }
        });
        write_sse(stream, &done)?;
        stream.write_all(b"data: [DONE]\n\n")?;
        stream.flush()?;
    } else {
        let body = json!({
            "id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": ids.len(),
                "completion_tokens": produced,
                "total_tokens": ids.len() + produced,
            }
        });
        write_json(stream, 200, &body)?;
    }

    let ttft = prefill;
    eprintln!(
        "request       prompt {} tok prefill {:.3} s, decode {:.3} s, {} tok, {} \
         ({:.2} tok/s end-to-end)",
        ids.len(),
        prefill,
        decode,
        produced,
        finish,
        produced as f64 / (ttft + decode).max(1e-9),
    );
    Ok(())
}

/// ChatML for the Qwen3 template. With thinking off, the generation prompt ends
/// in the empty think block the template uses for `enable_thinking=False`, so
/// the answer starts immediately.
fn render_chatml(messages: &[Value], think: bool) -> Result<String, String> {
    let mut out = String::new();
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let role = match role {
            "developer" => "system",
            "system" | "user" | "assistant" | "tool" => role,
            _ => "user",
        };
        let content = message_text(m.get("content"))?;
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    if !think {
        out.push_str("<think>\n\n</think>\n\n");
    }
    Ok(out)
}

/// A message `content` is a string, or an array of content parts whose text
/// parts are concatenated (the OpenAI multimodal spelling).
fn message_text(content: Option<&Value>) -> Result<String, String> {
    match content {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for p in parts {
                if let Some(t) = p.get("text").and_then(Value::as_str) {
                    out.push_str(t);
                }
            }
            Ok(out)
        }
        Some(Value::Null) | None => Ok(String::new()),
        Some(_) => Err("message content must be a string or an array of parts".into()),
    }
}

fn delta_chunk(id: &str, created: u64, model: &str, delta: &str, finish: Option<&str>) -> Value {
    json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": finish}]
    })
}

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── HTTP/1.1, one request per connection ─────────────────────────────────────

fn read_request(stream: &mut TcpStream) -> Result<Option<Request>, Box<dyn Error>> {
    stream.set_read_timeout(Some(Duration::from_secs(120)))?;
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(p) = find(&buf, b"\r\n\r\n") {
            break p;
        }
        if buf.len() > MAX_HEADER_BYTES {
            write_plain(stream, 431, "request header too large\n")?;
            return Ok(None);
        }
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = std::str::from_utf8(&buf[..head_end]).map_err(|_| "header is not UTF-8")?;
    let mut lines = head.split("\r\n");
    let request_line = lines.next().ok_or("missing request line")?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().ok_or("missing method")?.to_string();
    let path = parts.next().ok_or("missing path")?.to_string();

    let mut content_length = 0usize;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("content-length") {
            content_length = value.trim().parse()?;
        }
        if name.trim().eq_ignore_ascii_case("transfer-encoding") {
            write_plain(stream, 411, "chunked request bodies are not supported\n")?;
            return Ok(None);
        }
    }
    if content_length > MAX_BODY_BYTES {
        write_plain(stream, 413, "request body too large\n")?;
        return Ok(None);
    }

    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Err("request body ended early".into());
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    Ok(Some(Request { method, path, body }))
}

fn write_plain(stream: &mut TcpStream, status: u16, body: &str) -> Result<(), Box<dyn Error>> {
    let response = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: text/plain; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        status_text(status),
        body.len()
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn write_json(stream: &mut TcpStream, status: u16, body: &Value) -> Result<(), Box<dyn Error>> {
    let body = serde_json::to_vec(body)?;
    let response = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        status_text(status),
        body.len()
    );
    stream.write_all(response.as_bytes())?;
    stream.write_all(&body)?;
    stream.flush()?;
    Ok(())
}

fn write_error(stream: &mut TcpStream, status: u16, message: &str) -> Result<(), Box<dyn Error>> {
    write_json(
        stream,
        status,
        &json!({"error": {"message": message, "type": "invalid_request_error"}}),
    )
}

fn write_stream_headers(stream: &mut TcpStream) -> Result<(), Box<dyn Error>> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
    )?;
    stream.flush()?;
    Ok(())
}

fn write_sse(stream: &mut TcpStream, value: &Value) -> Result<(), Box<dyn Error>> {
    stream.write_all(b"data: ")?;
    stream.write_all(&serde_json::to_vec(value)?)?;
    stream.write_all(b"\n\n")?;
    stream.flush()?;
    Ok(())
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        411 => "Length Required",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        _ => "Error",
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_ends_in_the_empty_think_block() {
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let prompt = render_chatml(&messages, false).unwrap();
        assert_eq!(
            prompt,
            "<|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn chatml_leaves_the_think_block_out_when_thinking() {
        let messages = vec![json!({"role": "user", "content": "Hi"})];
        let prompt = render_chatml(&messages, true).unwrap();
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
        assert!(!prompt.contains("<think>"));
    }

    #[test]
    fn a_content_part_array_is_its_text_parts() {
        let messages = vec![json!({
            "role": "user",
            "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "World"}]
        })];
        let prompt = render_chatml(&messages, false).unwrap();
        assert!(prompt.starts_with("<|im_start|>user\nHello World<|im_end|>\n"));
    }

    #[test]
    fn developer_is_rendered_as_system() {
        let messages = vec![json!({"role": "developer", "content": "be brief"})];
        let prompt = render_chatml(&messages, false).unwrap();
        assert!(prompt.starts_with("<|im_start|>system\nbe brief<|im_end|>\n"));
    }
}
