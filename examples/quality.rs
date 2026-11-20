//! Zero-shot multiple-choice scoring on an item file: the Q2 0-shot arm.
//!
//! Run:
//! `cargo run --release --locked --example quality -- --model <gguf> [--pack <nqp>]
//!  --tokenizer <tokenizer.json> --items <task-200.jsonl> --out <scored.jsonl>
//!  [--kernel q8act|exact|ternary] [--head-batch 32] [--ctx 2048] [--max-refused 0] [--rebound]
//!  [--limit N] [--no-share]`
//!
//! One item is one question with 2–4 candidates. Each candidate is scored as
//! `Σ ln p(continuation | prompt)` in f64 from the Q32.32 head logits: the joined string is prefilled
//! once, and positions `boundary−1 … len−2` are projected in `--head-batch` chunks, so one head
//! projection per scored position. Choices that share a prompt prefill it once per item and rewind
//! the keys and values to the boundary (`--no-share` forces the one-prefill-per-choice path, which
//! is the reference the shared path is checked against). `acc` is `argmax ll`, `acc_norm` is `argmax ll / norm_len` (the
//! lm-eval rule), `acc_tok` is `argmax ll / n_tok` (the upstream anchor's per-token rule). Ties go
//! to the first maximum, as `np.argmax` does, and are counted.
//!
//! Boundary: the default refuses a candidate whose joined encoding does not start with the prompt's
//! encoding, because the continuation's first token then differs from the one scored by a separate
//! prefill. `--rebound` scores lm-eval's way instead (slice the joined encoding at the prompt's token
//! count), which is what a deployed runner does. Refusals are recorded per item and excluded from
//! the denominators; both arms refuse the same items because both use this tokenizer.
//!
//! The item file is built by `tools/quality-items.py`. Domain: the file must parse, every id must be
//! below the model vocabulary, every candidate must leave at least one scored token, and a context
//! longer than `--ctx` is left-truncated and marked. Cost is one prefill per candidate over at most
//! `--ctx` tokens plus one head projection per scored token; space is `O(ctx·n_embd)` states and
//! `O(head-batch·vocab)` logits.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "counts and log-probabilities are reported as f64 far below 2^53; token counts index a usize"
)]

use std::error::Error;
use std::fmt::Write as _;
use std::io::{BufWriter, Write};
use std::process::ExitCode;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::bpe::Bpe;
use apeiron::format::nq;

struct Args {
    model: String,
    pack: Option<String>,
    tokenizer: String,
    items: String,
    out: String,
    mode: KernelMode,
    head_batch: usize,
    ctx: usize,
    max_refused: usize,
    rebound: bool,
    no_share: bool,
    limit: Option<usize>,
    check: bool,
}

struct Candidate {
    prompt: String,
    continuation: String,
    norm_len: usize,
}

struct Item {
    task: String,
    /// The row index in the source split; the item file's `item` key.
    index: usize,
    gold: usize,
    candidates: Vec<Candidate>,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut a = Args {
        model: String::new(),
        pack: None,
        tokenizer: String::new(),
        items: String::new(),
        out: String::new(),
        mode: KernelMode::Q8Act,
        head_batch: 32,
        ctx: 2048,
        max_refused: 0,
        rebound: false,
        no_share: false,
        limit: None,
        check: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--pack" => a.pack = Some(val()?),
            "--tokenizer" => a.tokenizer = val()?,
            "--items" => a.items = val()?,
            "--out" => a.out = val()?,
            "--head-batch" => a.head_batch = val()?.parse()?,
            "--ctx" => a.ctx = val()?.parse()?,
            "--max-refused" => a.max_refused = val()?.parse()?,
            "--limit" => a.limit = Some(val()?.parse()?),
            "--rebound" => a.rebound = true,
            "--no-share" => a.no_share = true,
            "--check-items" => a.check = true,
            "--kernel" => {
                a.mode = match val()?.as_str() {
                    "exact" => KernelMode::Exact,
                    "q8act" => KernelMode::Q8Act,
                    "ternary" => KernelMode::Ternary,
                    other => {
                        return Err(format!("--kernel is q8act|ternary|exact, got {other:?}").into())
                    }
                }
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if a.items.is_empty() {
        return Err("--items is required".into());
    }
    if !a.check && (a.model.is_empty() || a.tokenizer.is_empty() || a.out.is_empty()) {
        return Err(
            "--model, --tokenizer, --items and --out are required unless --check-items".into(),
        );
    }
    if a.head_batch == 0 || a.ctx < 2 {
        return Err("--head-batch must be positive and --ctx at least 2".into());
    }
    Ok(a)
}

// ---- the item reader: the fixed schema `tools/quality-items.py` writes -------------------------

struct Cursor<'a> {
    text: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text: text.as_bytes(),
            at: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.get(self.at).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        self.skip_ws();
        if self.peek() != Some(byte) {
            return Err(format!("expected {:?} at byte {}", byte as char, self.at));
        }
        self.at += 1;
        Ok(())
    }

    fn eat(&mut self, byte: u8) -> bool {
        self.skip_ws();
        if self.peek() == Some(byte) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let mut value = 0u32;
        for _ in 0..4 {
            let byte = self.peek().ok_or("truncated \\u escape")?;
            let digit = (byte as char).to_digit(16).ok_or("bad \\u digit")?;
            value = value * 16 + digit;
            self.at += 1;
        }
        Ok(value)
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let byte = self.peek().ok_or("unterminated string")?;
            self.at += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = self.peek().ok_or("truncated escape")?;
                    self.at += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let high = self.hex4()?;
                            let scalar = if (0xD800..0xDC00).contains(&high) {
                                self.expect(b'\\')?;
                                self.expect(b'u')?;
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return Err(String::from("bad surrogate pair"));
                                }
                                0x1_0000 + ((high - 0xD800) << 10) + (low - 0xDC00)
                            } else {
                                high
                            };
                            out.push(char::from_u32(scalar).ok_or("bad scalar")?);
                        }
                        other => return Err(format!("bad escape \\{}", other as char)),
                    }
                }
                _ => {
                    // A UTF-8 continuation byte is copied verbatim; the item files are UTF-8.
                    let start = self.at - 1;
                    let tail = &self.text[start..];
                    let width = utf8_width(byte);
                    if tail.len() < width {
                        return Err(String::from("truncated UTF-8"));
                    }
                    let piece = std::str::from_utf8(&tail[..width]).map_err(|e| e.to_string())?;
                    out.push_str(piece);
                    self.at = start + width;
                }
            }
        }
    }

    fn integer(&mut self) -> Result<i64, String> {
        self.skip_ws();
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.at += 1;
        }
        let text = std::str::from_utf8(&self.text[start..self.at]).map_err(|e| e.to_string())?;
        text.parse().map_err(|_| format!("bad integer {text:?}"))
    }

    /// Skip one value of any shape, for keys this reader does not use.
    fn skip_value(&mut self) -> Result<(), String> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => self.string().map(|_| ()),
            Some(b'{') => {
                self.at += 1;
                if self.eat(b'}') {
                    return Ok(());
                }
                loop {
                    self.string()?;
                    self.expect(b':')?;
                    self.skip_value()?;
                    if self.eat(b'}') {
                        return Ok(());
                    }
                    self.expect(b',')?;
                }
            }
            Some(b'[') => {
                self.at += 1;
                if self.eat(b']') {
                    return Ok(());
                }
                loop {
                    self.skip_value()?;
                    if self.eat(b']') {
                        return Ok(());
                    }
                    self.expect(b',')?;
                }
            }
            Some(b't') => self.literal("true"),
            Some(b'f') => self.literal("false"),
            Some(b'n') => self.literal("null"),
            _ => self.integer().map(|_| ()),
        }
    }

    fn literal(&mut self, word: &str) -> Result<(), String> {
        if self.text[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(())
        } else {
            Err(format!("expected {word} at byte {}", self.at))
        }
    }
}

/// The byte width of a UTF-8 sequence from its lead byte.
const fn utf8_width(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead < 0xE0 {
        2
    } else if lead < 0xF0 {
        3
    } else {
        4
    }
}

fn parse_item(line: &str) -> Result<Item, String> {
    let mut c = Cursor::new(line);
    c.expect(b'{')?;
    let (mut task, mut index, mut gold) = (String::new(), 0usize, 0usize);
    let mut candidates = Vec::new();
    loop {
        let key = c.string()?;
        c.expect(b':')?;
        match key.as_str() {
            "task" => task = c.string()?,
            "item" => index = usize::try_from(c.integer()?).map_err(|_| "negative item")?,
            "gold" => gold = usize::try_from(c.integer()?).map_err(|_| "negative gold")?,
            "candidates" => {
                c.expect(b'[')?;
                if !c.eat(b']') {
                    loop {
                        let mut prompt = String::new();
                        let mut continuation = String::new();
                        let mut norm_len = 0usize;
                        c.expect(b'{')?;
                        loop {
                            let key = c.string()?;
                            c.expect(b':')?;
                            match key.as_str() {
                                "prompt" => prompt = c.string()?,
                                "continuation" => continuation = c.string()?,
                                "norm_len" => {
                                    norm_len = usize::try_from(c.integer()?)
                                        .map_err(|_| "negative norm")?;
                                }
                                _ => c.skip_value()?,
                            }
                            if c.eat(b'}') {
                                break;
                            }
                            c.expect(b',')?;
                        }
                        candidates.push(Candidate {
                            prompt,
                            continuation,
                            norm_len,
                        });
                        if c.eat(b']') {
                            break;
                        }
                        c.expect(b',')?;
                    }
                }
            }
            _ => c.skip_value()?,
        }
        if c.eat(b'}') {
            break;
        }
        c.expect(b',')?;
    }
    Ok(Item {
        task,
        index,
        gold,
        candidates,
    })
}

// ---- scoring ----------------------------------------------------------------------------------

/// The log-sum-exp of one logit vector in nats at `target`.
fn log_softmax(logits: &[i64], target: usize) -> Result<f64, String> {
    let scale = 1.0 / 4_294_967_296.0; // Q32.32
    let mut max = f64::NEG_INFINITY;
    for &v in logits {
        let f = v as f64 * scale;
        if f > max {
            max = f;
        }
    }
    let mut sum = 0.0f64;
    for &v in logits {
        sum += (v as f64 * scale - max).exp();
    }
    let target = *logits
        .get(target)
        .ok_or_else(|| format!("target {target} outside the vocabulary"))?;
    Ok(target as f64 * scale - max - sum.ln())
}

struct Scored {
    ll: f64,
    n_tok: usize,
    n_bytes: usize,
    refused: Option<&'static str>,
    truncated: bool,
}

#[allow(
    clippy::too_many_arguments,
    reason = "the scorer's inputs are the engine, the tokenizer and four protocol dials"
)]
fn score_candidate(
    engine: &Engine,
    bpe: &Bpe,
    ctx: usize,
    mode: KernelMode,
    head_batch: usize,
    rebound: bool,
    cand: &Candidate,
) -> Result<Scored, String> {
    let prompt_ids = bpe.encode(&cand.prompt).map_err(|e| e.to_string())?;
    if prompt_ids.is_empty() {
        return Ok(Scored {
            ll: 0.0,
            n_tok: 0,
            n_bytes: cand.continuation.len(),
            refused: Some("empty prompt"),
            truncated: false,
        });
    }
    let joined = format!("{}{}", cand.prompt, cand.continuation);
    let mut whole = bpe.encode(&joined).map_err(|e| e.to_string())?;
    let mut boundary = prompt_ids.len();
    let starts = whole.len() >= boundary && whole[..boundary] == prompt_ids[..];
    if !starts && !rebound {
        return Ok(Scored {
            ll: 0.0,
            n_tok: 0,
            n_bytes: cand.continuation.len(),
            refused: Some("boundary"),
            truncated: false,
        });
    }
    if boundary >= whole.len() {
        return Ok(Scored {
            ll: 0.0,
            n_tok: 0,
            n_bytes: cand.continuation.len(),
            refused: Some("empty span"),
            truncated: false,
        });
    }
    let mut truncated = false;
    if whole.len() > ctx {
        let start = whole.len() - ctx;
        truncated = true;
        if boundary <= start {
            return Ok(Scored {
                ll: 0.0,
                n_tok: 0,
                n_bytes: cand.continuation.len(),
                refused: Some("context overflow"),
                truncated,
            });
        }
        boundary -= start;
        whole = whole[start..].to_vec();
    }
    if let Some(&bad) = whole
        .iter()
        .find(|&&id| id as usize >= engine.config.vocab_size)
    {
        return Err(format!(
            "id {bad} >= vocabulary {}",
            engine.config.vocab_size
        ));
    }
    let mut st = engine.state(whole.len()).map_err(|e| format!("{e:?}"))?;
    let states = engine
        .prefill_states(&mut st, &whole, mode)
        .map_err(|e| format!("{e:?}"))?;
    if states.len() != whole.len() {
        return Err(format!("prefill returned {} states", states.len()));
    }
    let mut ll = 0.0f64;
    let mut at = boundary - 1;
    while at + 1 < whole.len() {
        let take = (whole.len() - 1 - at).min(head_batch);
        let logits = engine
            .lm_head_cols(&states[at..at + take], mode)
            .map_err(|e| format!("lm_head: {e:?}"))?;
        if logits.len() != take {
            return Err(format!("head returned {} of {take} rows", logits.len()));
        }
        for (c, row) in logits.iter().enumerate() {
            ll += log_softmax(row, whole[at + c + 1] as usize)?;
        }
        at += take;
    }
    Ok(Scored {
        ll,
        n_tok: whole.len() - boundary,
        n_bytes: cand.continuation.len(),
        refused: None,
        truncated,
    })
}

/// Score every choice of an item through one prefill of the shared prompt.
///
/// boolq, piqa, hellaswag and the arc tasks give every choice the same prompt, so `score_candidate`
/// prefills it once per choice. This prefills it once per item and rewinds the keys and values to
/// the boundary between choices, then reads the head over the last prompt state and the choice's
/// continuation states — the same head rows in the same order as the joint pass.
///
/// Returns `None` when sharing cannot reproduce `score_candidate` exactly, and the caller then
/// scores each choice the joint way: prompts that differ (winogrande puts the options in the
/// prompt), a continuation off the prompt's token boundary, left truncation at `ctx`, an empty
/// continuation, an affine carrier whose state this does not rewind, or an id past the vocabulary.
///
/// # Complexity
///
/// Time is `O((prompt + Σ continuations) · weights)` for the projections and
/// `O(choices · continuation · vocab)` for the head, the same work as the joint pass minus
/// `(choices − 1) · prompt` prefill columns. Space is one `ctx` decode state.
fn score_item_shared(
    engine: &Engine,
    bpe: &Bpe,
    ctx: usize,
    mode: KernelMode,
    head_batch: usize,
    candidates: &[Candidate],
) -> Result<Option<Vec<Scored>>, String> {
    let Some(first) = candidates.first() else {
        return Ok(None);
    };
    let prompt_ids = bpe.encode(&first.prompt).map_err(|e| e.to_string())?;
    if prompt_ids.is_empty() {
        return Ok(None);
    }
    let mut conts = Vec::with_capacity(candidates.len());
    for cand in candidates {
        let own_prompt = bpe.encode(&cand.prompt).map_err(|e| e.to_string())?;
        let whole = bpe
            .encode(&format!("{}{}", cand.prompt, cand.continuation))
            .map_err(|e| e.to_string())?;
        if own_prompt != prompt_ids
            || !whole.starts_with(&prompt_ids)
            || whole.len() <= prompt_ids.len()
            || whole.len() > ctx
        {
            return Ok(None);
        }
        if whole
            .iter()
            .any(|&id| id as usize >= engine.config.vocab_size)
        {
            return Ok(None);
        }
        conts.push(whole[prompt_ids.len()..].to_vec());
    }
    let mut st = engine
        .state(ctx.min(prompt_ids.len() + conts.iter().map(Vec::len).max().unwrap_or(0)))
        .map_err(|e| format!("{e:?}"))?;
    if st.affine.is_some() {
        return Ok(None);
    }
    let prompt_states = engine
        .prefill_states(&mut st, &prompt_ids, mode)
        .map_err(|e| format!("{e:?}"))?;
    if prompt_states.len() != prompt_ids.len() {
        return Err(format!(
            "prompt prefill returned {} states",
            prompt_states.len()
        ));
    }
    let last = prompt_states
        .last()
        .ok_or_else(|| "empty prompt states".to_string())?;
    let mut out = Vec::with_capacity(candidates.len());
    for (cand, cont) in candidates.iter().zip(&conts) {
        st.kv
            .truncate(prompt_ids.len())
            .map_err(|e| format!("rewind: {e:?}"))?;
        let cont_states = engine
            .prefill_states(&mut st, cont, mode)
            .map_err(|e| format!("{e:?}"))?;
        if cont_states.len() != cont.len() {
            return Err(format!(
                "continuation prefill returned {} states of {}",
                cont_states.len(),
                cont.len()
            ));
        }
        let mut rows = Vec::with_capacity(cont.len());
        rows.push(last.clone());
        rows.extend(cont_states[..cont.len() - 1].iter().cloned());
        let mut ll = 0.0f64;
        let mut at = 0usize;
        while at < rows.len() {
            let take = (rows.len() - at).min(head_batch);
            let logits = engine
                .lm_head_cols(&rows[at..at + take], mode)
                .map_err(|e| format!("lm_head: {e:?}"))?;
            if logits.len() != take {
                return Err(format!("head returned {} of {take} rows", logits.len()));
            }
            for (c, row) in logits.iter().enumerate() {
                ll += log_softmax(row, cont[at + c] as usize)?;
            }
            at += take;
        }
        out.push(Scored {
            ll,
            n_tok: cont.len(),
            n_bytes: cand.continuation.len(),
            refused: None,
            truncated: false,
        });
    }
    Ok(Some(out))
}

/// The first index of the largest value and how many entries attain it.
fn argmax(values: &[f64]) -> (usize, usize) {
    let mut best = 0usize;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    let top = values[best].to_bits();
    let ties = values.iter().filter(|v| v.to_bits() == top).count();
    (best, ties)
}

fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmHWM:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

/// Parse and validate the item file. With `--tokenizer` it also reports the tokenized length
/// distribution, the strict-boundary refusal count and how many candidates would be truncated at
/// `--ctx`, so a run plan does not discover an over-long item on the target host.
fn check_items(a: &Args) -> Result<(), Box<dyn Error>> {
    let text = std::fs::read_to_string(&a.items)?;
    let bpe = if a.tokenizer.is_empty() {
        None
    } else {
        Some(Bpe::from_tokenizer_json(&a.tokenizer)?)
    };
    let (mut items, mut candidates) = (0usize, 0usize);
    let (mut max_tokens, mut over, mut boundary) = (0usize, 0usize, 0usize);
    let (mut total, mut scored) = (0u64, 0u64);
    let mut task = String::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let item = parse_item(line)?;
        if task.is_empty() {
            task.clone_from(&item.task);
        }
        if item.task != task {
            return Err(format!("mixed tasks {task:?} and {:?} in one file", item.task).into());
        }
        if item.candidates.len() < 2 || item.gold >= item.candidates.len() {
            return Err(format!("item {}: bad gold or choice count", item.index).into());
        }
        if item
            .candidates
            .iter()
            .any(|c| c.prompt.is_empty() || c.norm_len == 0)
        {
            return Err(format!("item {}: empty prompt or zero norm_len", item.index).into());
        }
        items += 1;
        candidates += item.candidates.len();
        let Some(bpe) = &bpe else { continue };
        for cand in &item.candidates {
            let prompt = bpe.encode(&cand.prompt).map_err(|e| e.to_string())?;
            let joined = bpe
                .encode(&format!("{}{}", cand.prompt, cand.continuation))
                .map_err(|e| e.to_string())?;
            if joined.len() > a.ctx {
                over += 1;
            }
            if joined.len() < prompt.len() || joined[..prompt.len()] != prompt[..] {
                boundary += 1;
            }
            max_tokens = max_tokens.max(joined.len());
            total += joined.len() as u64;
            scored += joined.len().saturating_sub(prompt.len()) as u64;
        }
    }
    println!(
        "{{\"check\":true,\"task\":\"{}\",\"items\":{items},\"candidates\":{candidates},\
         \"max_tokens\":{max_tokens},\"prefill_tokens\":{total},\"scored_tokens\":{scored},\
         \"over_ctx\":{over},\"strict_boundary_refusals\":{boundary}}}",
        json_escape(&task)
    );
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "one sequential per-item pass: splitting it would carry the report state across functions"
)]
fn run(a: &Args) -> Result<(), Box<dyn Error>> {
    if a.check {
        return check_items(a);
    }
    let engine = match &a.pack {
        Some(path) => {
            let pack = nq::open(path).map_err(|e| e.to_string())?;
            println!(
                "pack: {} linears, {:.6} b/w, {} factor bytes",
                pack.linears.len(),
                pack.bits_per_weight().unwrap_or(0.0),
                pack.stored_bits().div_ceil(8)
            );
            Engine::open_with_nq(&a.model, pack)?
        }
        None => Engine::open(&a.model)?,
    };
    let bpe = Bpe::from_tokenizer_json(&a.tokenizer)?;
    let ctx = match engine.config.context_length {
        0 => a.ctx,
        n => a.ctx.min(n),
    };
    let text = std::fs::read_to_string(&a.items)?;
    let mut out = BufWriter::new(std::fs::File::create(&a.out)?);
    let t0 = Instant::now();
    let (mut n, mut scored_n, mut refused_n, mut truncated_n, mut ties) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut acc, mut acc_norm, mut acc_tok) = (0usize, 0usize, 0usize);
    let mut task = String::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        if a.limit.is_some_and(|limit| n >= limit) {
            break;
        }
        let item = parse_item(line)?;
        if task.is_empty() {
            task.clone_from(&item.task);
        }
        if item.task != task {
            return Err(format!("mixed tasks {task:?} and {:?} in one file", item.task).into());
        }
        if item.candidates.len() < 2 || item.gold >= item.candidates.len() {
            return Err(format!("item {}: bad gold or choice count", item.index).into());
        }
        n += 1;
        let started = Instant::now();
        let mut sll = Vec::with_capacity(item.candidates.len());
        let mut stok = Vec::with_capacity(item.candidates.len());
        let mut sbytes = Vec::with_capacity(item.candidates.len());
        let mut snorm = Vec::with_capacity(item.candidates.len());
        let mut refused = None;
        let mut truncated = false;
        let shared = if a.no_share {
            None
        } else {
            score_item_shared(&engine, &bpe, ctx, a.mode, a.head_batch, &item.candidates)?
        };
        let scored = match shared {
            Some(shared) => shared,
            None => {
                let mut joint = Vec::with_capacity(item.candidates.len());
                for cand in &item.candidates {
                    joint.push(score_candidate(
                        &engine,
                        &bpe,
                        ctx,
                        a.mode,
                        a.head_batch,
                        a.rebound,
                        cand,
                    )?);
                }
                joint
            }
        };
        for (cand, s) in item.candidates.iter().zip(scored) {
            sll.push(s.ll);
            stok.push(s.n_tok);
            sbytes.push(s.n_bytes);
            snorm.push(cand.norm_len);
            if let Some(reason) = s.refused {
                refused = Some(reason);
            }
            truncated |= s.truncated;
        }
        let seconds = started.elapsed().as_secs_f64();
        let mut line = String::new();
        let _ = write!(
            line,
            "{{\"task\":\"{}\",\"item\":{},\"gold\":{},\"ll\":[",
            json_escape(&item.task),
            item.index,
            item.gold
        );
        for (i, v) in sll.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            let _ = write!(line, "{v:.9}");
        }
        line.push_str("],\"n_tok\":[");
        for (i, v) in stok.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            let _ = write!(line, "{v}");
        }
        line.push_str("],\"n_bytes\":[");
        for (i, v) in sbytes.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            let _ = write!(line, "{v}");
        }
        line.push_str("],\"norm_len\":[");
        for (i, v) in snorm.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            let _ = write!(line, "{v}");
        }
        line.push_str("],\"correct\":{");
        if let Some(reason) = refused {
            refused_n += 1;
            let _ = write!(
                line,
                "\"acc\":false,\"acc_norm\":false,\"acc_tok\":false}},\"refused\":\"{reason}\","
            );
        } else {
            scored_n += 1;
            truncated_n += usize::from(truncated);
            let (acc_i, tie) = argmax(&sll);
            ties += usize::from(tie > 1);
            let norm: Vec<f64> = sll
                .iter()
                .zip(&snorm)
                .map(|(ll, n)| *ll / (*n).max(1) as f64)
                .collect();
            let tok: Vec<f64> = sll
                .iter()
                .zip(&stok)
                .map(|(ll, n)| *ll / (*n).max(1) as f64)
                .collect();
            let (norm_i, _) = argmax(&norm);
            let (tok_i, _) = argmax(&tok);
            acc += usize::from(acc_i == item.gold);
            acc_norm += usize::from(norm_i == item.gold);
            acc_tok += usize::from(tok_i == item.gold);
            let _ = write!(
                line,
                "\"acc\":{},\"acc_norm\":{},\"acc_tok\":{}}},\"refused\":null,",
                acc_i == item.gold,
                norm_i == item.gold,
                tok_i == item.gold
            );
        }
        let _ = writeln!(line, "\"truncated\":{truncated},\"seconds\":{seconds:.3}}}");
        out.write_all(line.as_bytes())?;
        if n % 10 == 0 || n == 1 {
            eprintln!("item {n} elapsed {:.1} s", t0.elapsed().as_secs_f64());
        }
    }
    out.flush()?;
    let seconds = t0.elapsed().as_secs_f64();
    let share = |k: usize| k as f64 / scored_n.max(1) as f64;
    println!(
        "{{\"summary\":true,\"task\":\"{}\",\"items\":{n},\"scored\":{scored_n},\"refused\":{refused_n},\
         \"truncated\":{truncated_n},\"ties\":{ties},\"acc\":{:.6},\"acc_norm\":{:.6},\"acc_tok\":{:.6},\
         \"seconds\":{seconds:.1},\"kernel\":\"{:?}\"}}",
        json_escape(&task),
        share(acc),
        share(acc_norm),
        share(acc_tok),
        a.mode
    );
    if let Some(kib) = peak_rss_kib() {
        println!("peak rss {:.1} MiB", kib as f64 / 1024.0);
    }
    if refused_n > a.max_refused {
        return Err(format!(
            "{refused_n} refused items exceed --max-refused {}",
            a.max_refused
        )
        .into());
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse_args().and_then(|a| run(&a)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("quality: {e}");
            ExitCode::FAILURE
        }
    }
}
