//! Byte-level BPE tokenizer for the served Qwen/GPT-2-family checkpoints.
//!
//! Ported from the sibling `fold` tree (`fold/modelfold/src/text.rs`, module
//! `text::tokenizer`), which reads a HuggingFace `tokenizer.json` through hashtrinity's JSON
//! parser. This port keeps the same pre-tokenization, merge-rank and byte-level decode
//! algorithms, but reads the file with a self-contained checked JSON reader and returns
//! [`BpeError`] refusals instead of panicking.
//!
//! The served vocabulary is Qwen2.5-0.5B: `model.type = "BPE"` with a cl100k `Split` regex
//! (single-digit `\p{N}` cap) followed by the GPT-2 `ByteLevel` alphabet, and 22 added tokens
//! (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, …) that are matched literally before BPE.
//!
//! # Trust boundary
//!
//! * The `normalizer` stage is not applied. Qwen declares NFC; on the reference vectors here
//!   (ASCII, precomposed Unicode, CJK) NFC is the identity, so the ported stream is exact.
//!   Text that is not already NFC-normalized may segment differently, as it does in `fold`.
//! * Only byte-level BPE is built. A `SentencePiece`/`Metaspace` checkpoint, a non-`BPE`
//!   `model.type`, or a `byte_fallback` model is refused rather than approximated.
//! * Codegen, the kernel ABI and the inference engine are out of scope; this module ends at
//!   `Vec<u32>` ids.

use std::collections::HashMap;

/// The GPT-2 `bytes_to_unicode` alphabet maps byte 0x20 (space) to U+0120, the char a merged
/// token carries when it begins with a space.
#[cfg(test)]
const SPACE_CHAR: char = '\u{0120}';

/// The separator joining a merge pair `(a, b)` into one rank-key string. Byte-level pieces use
/// chars from U+0021 upward (space is `Ġ`), so U+0000 occurs in no piece and the split back to
/// `(a, b)` is unique.
const RANK_SEP: char = '\u{0000}';

/// A refused tokenizer operation. Integer- and string-carrying so callers can match without
/// formatting; every variant is reachable only from file data, never from a panic path.
#[derive(Debug, PartialEq, Eq)]
pub enum BpeError {
    /// The tokenizer file could not be read.
    Io(String),
    /// The JSON reader refused at byte `at` with `msg`.
    Json {
        /// Byte offset where the parse stopped.
        at: usize,
        /// Static description of the refusal.
        msg: &'static str,
    },
    /// A required member of `tokenizer.json` is absent or has the wrong shape.
    Missing(&'static str),
    /// The checkpoint is not a byte-level BPE this module can run.
    Unsupported(&'static str),
    /// A merge entry was not a `"a b"` string or a two-element array.
    MergeMalformed(usize),
    /// A merge piece contains the rank-key separator, so the key would not be injective.
    MergeSeparator(usize),
    /// A `decode` id has no token string.
    UnknownToken(u32),
    /// A BPE piece resolved to no vocabulary entry.
    UnknownPiece(String),
}

impl core::fmt::Display for BpeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BpeError::Io(e) => write!(f, "io: {e}"),
            BpeError::Json { at, msg } => write!(f, "json at byte {at}: {msg}"),
            BpeError::Missing(what) => write!(f, "tokenizer.json: missing {what}"),
            BpeError::Unsupported(why) => write!(f, "unsupported checkpoint: {why}"),
            BpeError::MergeMalformed(rank) => write!(f, "merge {rank} is not a pair"),
            BpeError::MergeSeparator(rank) => {
                write!(f, "merge {rank} piece contains the rank-key separator")
            }
            BpeError::UnknownToken(id) => write!(f, "no token for id {id}"),
            BpeError::UnknownPiece(p) => write!(f, "no vocabulary entry for piece {p:?}"),
        }
    }
}

impl std::error::Error for BpeError {}

/// A loaded byte-level BPE tokenizer. Built by [`Bpe::from_tokenizer_json`].
pub struct Bpe {
    /// token string → id, for the encode lookup.
    vocab: HashMap<String, u32>,
    /// id → token string, for decode. Holes where a vocab id is absent hold `""`.
    id_to_tok: Vec<String>,
    /// merge rank keys `a ‖ RANK_SEP ‖ b` → rank.
    ranks: HashMap<String, u32>,
    /// added tokens matched literally in the raw text, longest first.
    specials: Vec<(String, u32)>,
    /// byte → printable char (the GPT-2 alphabet).
    b2u: [char; 256],
    /// inverse of `b2u`, indexed by the char's scalar value; `None` outside the alphabet.
    u2b: Vec<Option<u8>>,
    /// the `\p{N}` quantifier's upper bound read from the declared split regex (1 for Qwen).
    digit_cap: usize,
}

impl Bpe {
    /// Load a byte-level BPE tokenizer from a HuggingFace `tokenizer.json`.
    ///
    /// Refuses, rather than approximating, a checkpoint whose `model.type` is not `BPE`, whose
    /// `model.byte_fallback` is set, or whose `pre_tokenizer` declares no `Split` regex.
    #[allow(
        clippy::too_many_lines,
        reason = "one linear, stage-per-block loader; splitting it would spread the field paths \
                  that a reader must check together"
    )]
    pub fn from_tokenizer_json(path: &str) -> Result<Bpe, BpeError> {
        let raw = std::fs::read(path).map_err(|e| BpeError::Io(e.to_string()))?;
        let root = parse_document(&raw)?;

        let model = obj_get(&root, "model").ok_or(BpeError::Missing("model"))?;
        match obj_get(model, "type").and_then(as_str) {
            Some("BPE") => {}
            _ => return Err(BpeError::Unsupported("model.type is not BPE")),
        }
        if matches!(obj_get(model, "byte_fallback"), Some(Json::Bool(true))) {
            return Err(BpeError::Unsupported(
                "byte_fallback models are not byte-level BPE",
            ));
        }
        let digit_cap = detect_digit_cap(&root)?;

        let vobj = obj_get(model, "vocab").ok_or(BpeError::Missing("model.vocab"))?;
        let Json::Obj(vps) = vobj else {
            return Err(BpeError::Missing("model.vocab object"));
        };

        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(vps.len());
        let mut maxid: usize = 0;
        let mut vi = 0;
        while vi < vps.len() {
            let id = as_u32(&vps[vi].1).ok_or(BpeError::Missing("integer vocab id"))?;
            if id as usize > maxid {
                maxid = id as usize;
            }
            vocab.insert(vps[vi].0.clone(), id);
            vi += 1;
        }

        let mut id_to_tok: Vec<String> = Vec::with_capacity(maxid + 1);
        let mut k = 0;
        while k <= maxid {
            id_to_tok.push(String::new());
            k += 1;
        }
        vi = 0;
        while vi < vps.len() {
            let id = as_u32(&vps[vi].1).ok_or(BpeError::Missing("integer vocab id"))?;
            id_to_tok[id as usize].clone_from(&vps[vi].0);
            vi += 1;
        }

        let marr = obj_get(model, "merges")
            .and_then(as_arr)
            .ok_or(BpeError::Missing("model.merges array"))?;
        let mut ranks: HashMap<String, u32> = HashMap::with_capacity(marr.len());
        let mut mi = 0;
        while mi < marr.len() {
            let (a, b) = merge_pair(&marr[mi], mi)?;
            if contains_nul(&a) || contains_nul(&b) {
                return Err(BpeError::MergeSeparator(mi));
            }
            let mut key = String::with_capacity(a.len() + 1 + b.len());
            key.push_str(&a);
            key.push(RANK_SEP);
            key.push_str(&b);
            ranks.insert(key, mi as u32);
            mi += 1;
        }

        let mut specials: Vec<(String, u32)> = Vec::new();
        if let Some(ats) = obj_get(&root, "added_tokens").and_then(as_arr) {
            let mut ai = 0;
            while ai < ats.len() {
                let content = obj_get(&ats[ai], "content").and_then(as_str);
                let id = obj_get(&ats[ai], "id").and_then(as_u32);
                if let (Some(c), Some(id)) = (content, id) {
                    let cs = c.to_string();
                    specials.push((cs.clone(), id));
                    while id_to_tok.len() <= id as usize {
                        id_to_tok.push(String::new());
                    }
                    id_to_tok[id as usize] = cs;
                }
                ai += 1;
            }
        }
        specials.sort_by_key(|s| core::cmp::Reverse(s.0.len()));

        let b2u = bytes_to_unicode();
        let mut span = 0usize;
        let mut bi = 0;
        while bi < 256 {
            let c = b2u[bi] as usize;
            if c + 1 > span {
                span = c + 1;
            }
            bi += 1;
        }
        let mut u2b: Vec<Option<u8>> = Vec::with_capacity(span);
        let mut ui = 0;
        while ui < span {
            u2b.push(None);
            ui += 1;
        }
        bi = 0;
        while bi < 256 {
            u2b[b2u[bi] as usize] = Some(bi as u8);
            bi += 1;
        }

        Ok(Bpe {
            vocab,
            id_to_tok,
            ranks,
            specials,
            b2u,
            u2b,
            digit_cap,
        })
    }

    /// Number of ids the vocabulary and added tokens span (max id + 1).
    pub fn vocab_size(&self) -> usize {
        self.id_to_tok.len()
    }

    /// The id of a named added token, or `None`.
    pub fn special_id(&self, name: &str) -> Option<u32> {
        let mut i = 0;
        while i < self.specials.len() {
            if self.specials[i].0 == name {
                return Some(self.specials[i].1);
            }
            i += 1;
        }
        None
    }

    /// The end-of-turn token id among the conventional chat names, or `None`.
    pub fn eos_id(&self) -> Option<u32> {
        let names = [
            "<|im_end|>",
            "<|eot_id|>",
            "<|endoftext|>",
            "</s>",
            "<|end|>",
        ];
        let mut i = 0;
        while i < names.len() {
            if let Some(id) = self.special_id(names[i]) {
                return Some(id);
            }
            i += 1;
        }
        None
    }

    /// Encode text → token ids. Added tokens are split out first (earliest occurrence, longest
    /// at a tie), then each remaining span is pre-tokenized and merged by rank.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, BpeError> {
        let mut out: Vec<u32> = Vec::new();
        self.encode_into(text, &mut out)?;
        Ok(out)
    }

    /// Decode ids → text by the GPT-2 byte-level inverse: every char of a token that came from
    /// a byte maps back to it, and chars outside the alphabet pass through as their own UTF-8.
    pub fn decode(&self, ids: &[u32]) -> Result<String, BpeError> {
        let mut bytes: Vec<u8> = Vec::new();
        let mut i = 0;
        while i < ids.len() {
            let id = ids[i];
            let Some(tok) = self.id_to_tok.get(id as usize) else {
                return Err(BpeError::UnknownToken(id));
            };
            let cs = decode_utf8(tok);
            let mut j = 0;
            while j < cs.len() {
                let ch = cs[j];
                let cu = ch as usize;
                let mapped = if cu < self.u2b.len() {
                    self.u2b[cu]
                } else {
                    None
                };
                if let Some(b) = mapped {
                    bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                j += 1;
            }
            i += 1;
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Split out the earliest added token, then recurse on the tail.
    fn encode_into(&self, text: &str, out: &mut Vec<u32>) -> Result<(), BpeError> {
        if text.is_empty() {
            return Ok(());
        }
        let bytes = text.as_bytes();
        let mut hit: Option<(usize, usize, u32)> = None;
        let mut si = 0;
        while si < self.specials.len() {
            let (c, id) = &self.specials[si];
            if let Some(p) = find_sub(bytes, c.as_bytes()) {
                let replace = match hit {
                    None => true,
                    Some((bp, _, _)) => p < bp,
                };
                if replace {
                    hit = Some((p, c.len(), *id));
                }
            }
            si += 1;
        }
        match hit {
            Some((p, len, id)) => {
                self.bpe_span(&text[..p], out)?;
                out.push(id);
                self.encode_into(&text[p + len..], out)
            }
            None => self.bpe_span(text, out),
        }
    }

    /// Pre-tokenize a span and run BPE on each piece.
    fn bpe_span(&self, text: &str, out: &mut Vec<u32>) -> Result<(), BpeError> {
        let pieces = pretokenize_cl100k(text, self.digit_cap);
        let mut key = String::new();
        let mut pi = 0;
        while pi < pieces.len() {
            let mut sym = self.init_syms_bytes(&pieces[pi]);
            self.merge(&mut sym, &mut key);
            let mut si = 0;
            while si < sym.len() {
                let s = std::mem::take(&mut sym[si]);
                match self.vocab.get(s.as_str()) {
                    Some(&id) => out.push(id),
                    None => return Err(BpeError::UnknownPiece(s)),
                }
                si += 1;
            }
            pi += 1;
        }
        Ok(())
    }

    /// One `b2u` char per byte of a piece.
    fn init_syms_bytes(&self, piece: &str) -> Vec<String> {
        let bytes = piece.as_bytes();
        let mut sym: Vec<String> = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            let ch = self.b2u[bytes[i] as usize];
            sym.push(ch.to_string());
            i += 1;
        }
        sym
    }

    /// The rank of adjacent pair `(a, b)`, if declared. `key` is caller-owned scratch so the
    /// merge loop allocates nothing.
    #[inline]
    fn rank_of(&self, a: &str, b: &str, key: &mut String) -> Option<u32> {
        key.clear();
        key.push_str(a);
        key.push(RANK_SEP);
        key.push_str(b);
        self.ranks.get(key.as_str()).copied()
    }

    /// Greedily apply the lowest-rank adjacent merge (leftmost at a tie) until none applies.
    fn merge(&self, sym: &mut Vec<String>, key: &mut String) {
        while sym.len() >= 2 {
            let mut best: Option<(usize, u32)> = None;
            let mut i = 0;
            while i + 1 < sym.len() {
                if let Some(r) = self.rank_of(&sym[i], &sym[i + 1], key) {
                    let replace = match best {
                        None => true,
                        Some((_, br)) => r < br,
                    };
                    if replace {
                        best = Some((i, r));
                    }
                }
                i += 1;
            }
            let Some((pos, _)) = best else { break };
            let tail = std::mem::take(&mut sym[pos + 1]);
            sym[pos].push_str(&tail);
            sym.remove(pos + 1);
        }
    }
}

// ── pre-tokenization ────────────────────────────────────────────────────────────────────────────

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_digit(c: char) -> bool {
    c.is_numeric()
}

fn is_punct(c: char) -> bool {
    !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric()
}

pub(crate) fn is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// Build a string from `chars[lo..hi]`, pushed one char at a time.
fn chars_to_string(chars: &[char], lo: usize, hi: usize) -> String {
    let mut s = String::new();
    let mut i = lo;
    while i < hi {
        s.push(chars[i]);
        i += 1;
    }
    s
}

/// Decode a valid `&str` into chars with an indexed UTF-8 walk (no iterator, so the routine is
/// translatable).
fn decode_utf8(text: &str) -> Vec<char> {
    let b = text.as_bytes();
    let n = b.len();
    let mut out: Vec<char> = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let x = b[i];
        let (cp, len) = if x < 0x80 {
            (x as u32, 1)
        } else if (x & 0xE0) == 0xC0 && i + 1 < n {
            ((((x & 0x1F) as u32) << 6) | ((b[i + 1] & 0x3F) as u32), 2)
        } else if (x & 0xF0) == 0xE0 && i + 2 < n {
            (
                (((x & 0x0F) as u32) << 12)
                    | (((b[i + 1] & 0x3F) as u32) << 6)
                    | ((b[i + 2] & 0x3F) as u32),
                3,
            )
        } else if (x & 0xF8) == 0xF0 && i + 3 < n {
            (
                (((x & 0x07) as u32) << 18)
                    | (((b[i + 1] & 0x3F) as u32) << 12)
                    | (((b[i + 2] & 0x3F) as u32) << 6)
                    | ((b[i + 3] & 0x3F) as u32),
                4,
            )
        } else {
            (0xFFFD, 1)
        };
        out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
        i += len;
    }
    out
}

/// cl100k/GPT-4-family split (Qwen): contractions (case-insensitive),
/// `[^\r\n\p{L}\p{N}]?\p{L}+`, `\p{N}{1,digit_cap}`, ` ?[^\s\p{L}\p{N}]+[\r\n]*`, `\s*[\r\n]+`,
/// and the trailing-whitespace rules. Mirrors `fold`'s `pretokenize_cl100k`.
fn pretokenize_cl100k(text: &str, digit_cap: usize) -> Vec<String> {
    let chars = decode_utf8(text);
    let n = chars.len();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];

        // contractions (?i): 's 't 're 've 'm 'll 'd
        if c == '\'' && i + 1 < n {
            let mut two = String::new();
            two.push(chars[i + 1]);
            if i + 2 < n {
                two.push(chars[i + 2]);
            }
            let two_hit = two.len() == 2
                && (two.eq_ignore_ascii_case("re")
                    || two.eq_ignore_ascii_case("ve")
                    || two.eq_ignore_ascii_case("ll"));
            if two_hit {
                out.push(chars_to_string(&chars, i, i + 3));
                i += 3;
                continue;
            }
            let one = chars[i + 1].to_ascii_lowercase();
            if one == 's' || one == 't' || one == 'm' || one == 'd' {
                out.push(chars_to_string(&chars, i, i + 2));
                i += 2;
                continue;
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+ : letters, absorbing one preceding non-L/N char.
        let absorb =
            !is_nl(c) && !is_letter(c) && !is_digit(c) && i + 1 < n && is_letter(chars[i + 1]);
        if is_letter(c) || absorb {
            let ls = if is_letter(c) { i } else { i + 1 };
            let mut j = ls;
            while j < n && is_letter(chars[j]) {
                j += 1;
            }
            out.push(chars_to_string(&chars, i, j));
            i = j;
            continue;
        }

        // \p{N}{1,digit_cap} : a digit run, greedy up to the cap.
        if is_digit(c) {
            let cap = if digit_cap == 0 { 1 } else { digit_cap };
            let mut j = i;
            while j < n && j - i < cap && is_digit(chars[j]) {
                j += 1;
            }
            out.push(chars_to_string(&chars, i, j));
            i = j;
            continue;
        }

        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        let lead_space = c == ' ' && i + 1 < n && is_punct(chars[i + 1]);
        let pstart = if lead_space { i + 1 } else { i };
        if pstart < n && is_punct(chars[pstart]) {
            let mut j = pstart;
            while j < n && is_punct(chars[j]) {
                j += 1;
            }
            while j < n && is_nl(chars[j]) {
                j += 1;
            }
            out.push(chars_to_string(&chars, i, j));
            i = j;
            continue;
        }

        // `\s*[\r\n]+` : a whitespace run containing a newline, trimmed back to the last newline.
        let mut j = i;
        while j < n && chars[j].is_whitespace() {
            j += 1;
        }
        let mut has_nl = false;
        let mut k = i;
        while k < j {
            if is_nl(chars[k]) {
                has_nl = true;
            }
            k += 1;
        }
        if has_nl {
            let mut e = j;
            while e > i && !is_nl(chars[e - 1]) {
                e -= 1;
            }
            out.push(chars_to_string(&chars, i, e));
            i = e;
            continue;
        }
        i = whitespace_run(&chars, i, &mut out);
    }
    out
}

/// A whitespace run emits as one piece, except a non-final run leaves its last char to attach to
/// the next word (`\s+(?!\S)`). Returns the new cursor.
fn whitespace_run(chars: &[char], i: usize, out: &mut Vec<String>) -> usize {
    let n = chars.len();
    let mut j = i;
    while j < n && chars[j].is_whitespace() {
        j += 1;
    }
    if j == n {
        out.push(chars_to_string(chars, i, j));
        j
    } else if j - i >= 2 {
        out.push(chars_to_string(chars, i, j - 1));
        j - 1
    } else {
        out.push(chars_to_string(chars, i, j));
        j
    }
}

// ── byte alphabet ───────────────────────────────────────────────────────────────────────────────

/// The GPT-2 reversible byte→unicode map: printable bytes map to themselves and the rest to
/// U+0100+n, so every byte is one printable char.
fn bytes_to_unicode() -> [char; 256] {
    let mut bs: [u32; 256] = [0; 256];
    let mut bn = 0usize;
    let mut b = b'!' as u32;
    while b <= b'~' as u32 {
        bs[bn] = b;
        bn += 1;
        b += 1;
    }
    b = 0xA1;
    while b <= 0xAC {
        bs[bn] = b;
        bn += 1;
        b += 1;
    }
    b = 0xAE;
    while b <= 0xFF {
        bs[bn] = b;
        bn += 1;
        b += 1;
    }
    let visible = bn;
    let mut cs: [u32; 256] = [0; 256];
    let mut i = 0;
    while i < visible {
        cs[i] = bs[i];
        i += 1;
    }
    let mut n = 0u32;
    b = 0;
    while b < 256 {
        let mut found = false;
        let mut k = 0;
        while k < visible {
            if bs[k] == b {
                found = true;
            }
            k += 1;
        }
        if !found {
            bs[bn] = b;
            cs[bn] = 256 + n;
            bn += 1;
            n += 1;
        }
        b += 1;
    }
    let mut out = ['\0'; 256];
    i = 0;
    while i < 256 {
        out[bs[i] as usize] = char::from_u32(cs[i]).unwrap_or('\0');
        i += 1;
    }
    out
}

/// The first byte index of `needle` in `hay`, or `None`.
#[allow(
    clippy::len_zero,
    reason = "the extraction discipline keeps the `len() == 0` form; `is_empty` is one of the \
              container calls the refinement lane excludes"
)]
pub(crate) fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() == 0 || needle.len() > hay.len() {
        return None;
    }
    let mut found: Option<usize> = None;
    let mut i = 0;
    // The first complete match is carried out of the scan instead of returned from inside it; the
    // cursor stops advancing on the match, so the result is the same index for every input.
    while i + needle.len() <= hay.len() && found.is_none() {
        let mut k = 0;
        let mut ok = true;
        while k < needle.len() {
            if hay[i + k] != needle[k] {
                ok = false;
                break;
            }
            k += 1;
        }
        if ok {
            found = Some(i);
        }
        i += 1;
    }
    found
}

pub(crate) fn contains_nul(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    let mut found = false;
    // A carried flag instead of an early `return`: the scan always reads the whole slice and the
    // result is the same at every length, so no refusal ordering changes.
    while i < b.len() {
        if b[i] == 0 {
            found = true;
        }
        i += 1;
    }
    found
}

/// One merge entry as `(a, b)`: a `"a b"` string or a two-element array of strings.
fn merge_pair(j: &Json, rank: usize) -> Result<(String, String), BpeError> {
    match j {
        Json::Str(s) => match s.split_once(' ') {
            Some((a, b)) => Ok((a.to_string(), b.to_string())),
            None => Err(BpeError::MergeMalformed(rank)),
        },
        Json::Arr(p) => {
            if p.len() != 2 {
                return Err(BpeError::MergeMalformed(rank));
            }
            let a = as_str(&p[0]).ok_or(BpeError::MergeMalformed(rank))?;
            let b = as_str(&p[1]).ok_or(BpeError::MergeMalformed(rank))?;
            Ok((a.to_string(), b.to_string()))
        }
        _ => Err(BpeError::MergeMalformed(rank)),
    }
}

// ── regime detection ────────────────────────────────────────────────────────────────────────────

/// Read the `\p{N}` run cap from the declared split regex; unrecognized shapes fall back to 1.
/// Refuses when no `Split` stage with a regex is declared.
fn detect_digit_cap(root: &Json) -> Result<usize, BpeError> {
    let pt = obj_get(root, "pre_tokenizer").ok_or(BpeError::Unsupported("no pre_tokenizer"))?;
    let mut stages: Vec<&Json> = Vec::new();
    if let Some(inner) = obj_get(pt, "pretokenizers").and_then(as_arr) {
        let mut i = 0;
        while i < inner.len() {
            stages.push(&inner[i]);
            i += 1;
        }
    } else {
        stages.push(pt);
    }
    let mut i = 0;
    while i < stages.len() {
        if obj_get(stages[i], "type").and_then(as_str) == Some("Split") {
            let re = obj_get(stages[i], "pattern")
                .and_then(|p| obj_get(p, "Regex"))
                .and_then(as_str);
            if let Some(pattern) = re {
                return Ok(digit_quantifier_cap(pattern));
            }
        }
        i += 1;
    }
    Err(BpeError::Unsupported("no Split regex pre_tokenizer"))
}

/// A literal scan for `\p{N}{1,N}`; bare `\p{N}` (Qwen) yields 1.
fn digit_quantifier_cap(pattern: &str) -> usize {
    let needle = "\\p{N}{1,";
    if let Some(start) = pattern.find(needle) {
        let rest = &pattern[start + needle.len()..];
        if let Some(end) = rest.find('}') {
            if let Ok(n) = rest[..end].parse::<usize>() {
                return if n < 1 { 1 } else { n };
            }
        }
    }
    1
}

// ── self-contained JSON reader ──────────────────────────────────────────────────────────────────

/// A parsed JSON value. Not `Clone`: a recursive payload with a derived clone is what the
/// format layer's extraction discipline keeps out.
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

fn obj_get<'a>(j: &'a Json, k: &str) -> Option<&'a Json> {
    match j {
        Json::Obj(ps) => {
            let mut i = 0;
            while i < ps.len() {
                if ps[i].0 == k {
                    return Some(&ps[i].1);
                }
                i += 1;
            }
            None
        }
        _ => None,
    }
}

fn as_str(j: &Json) -> Option<&str> {
    match j {
        Json::Str(s) => Some(s),
        _ => None,
    }
}

fn as_arr(j: &Json) -> Option<&Vec<Json>> {
    match j {
        Json::Arr(v) => Some(v),
        _ => None,
    }
}

/// A JSON number that is a non-negative integer within `u32`.
fn as_u32(j: &Json) -> Option<u32> {
    match j {
        Json::Num(n) => {
            if *n >= 0.0 && *n <= 4_294_967_295.0 && n.fract() == 0.0 {
                Some(*n as u32)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(crate) fn hex_val(c: u8) -> Option<u32> {
    match c {
        b'0'..=b'9' => Some((c - b'0') as u32),
        b'a'..=b'f' => Some((c - b'a') as u32 + 10),
        b'A'..=b'F' => Some((c - b'A') as u32 + 10),
        _ => None,
    }
}

struct JsonReader<'a> {
    b: &'a [u8],
    i: usize,
}

impl JsonReader<'_> {
    fn err(&self, msg: &'static str) -> BpeError {
        BpeError::Json { at: self.i, msg }
    }

    fn skip_ws(&mut self) {
        while self.i < self.b.len() {
            let c = self.b[self.i];
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<Json, BpeError> {
        self.skip_ws();
        if self.i >= self.b.len() {
            return Err(self.err("unexpected end of input"));
        }
        let c = self.b[self.i];
        match c {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => {
                let s = self.parse_string()?;
                Ok(Json::Str(s))
            }
            b't' => {
                self.lit(b"true")?;
                Ok(Json::Bool(true))
            }
            b'f' => {
                self.lit(b"false")?;
                Ok(Json::Bool(false))
            }
            b'n' => {
                self.lit(b"null")?;
                Ok(Json::Null)
            }
            b'-' | b'0'..=b'9' => {
                let n = self.parse_number()?;
                Ok(Json::Num(n))
            }
            _ => Err(self.err("unexpected byte")),
        }
    }

    fn lit(&mut self, s: &[u8]) -> Result<(), BpeError> {
        let mut k = 0;
        while k < s.len() {
            if self.i >= self.b.len() || self.b[self.i] != s[k] {
                return Err(self.err("bad literal"));
            }
            self.i += 1;
            k += 1;
        }
        Ok(())
    }

    fn parse_number(&mut self) -> Result<f64, BpeError> {
        let start = self.i;
        if self.i < self.b.len() && self.b[self.i] == b'-' {
            self.i += 1;
        }
        while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
            self.i += 1;
        }
        if self.i < self.b.len() && self.b[self.i] == b'.' {
            self.i += 1;
            while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
                self.i += 1;
            }
        }
        if self.i < self.b.len() && (self.b[self.i] == b'e' || self.b[self.i] == b'E') {
            self.i += 1;
            if self.i < self.b.len() && (self.b[self.i] == b'+' || self.b[self.i] == b'-') {
                self.i += 1;
            }
            while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
                self.i += 1;
            }
        }
        let Ok(s) = core::str::from_utf8(&self.b[start..self.i]) else {
            return Err(self.err("number is not utf8"));
        };
        s.parse::<f64>().map_err(|_| self.err("bad number"))
    }

    fn parse_u4(&mut self) -> Result<u32, BpeError> {
        let mut v = 0u32;
        let mut k = 0;
        while k < 4 {
            if self.i >= self.b.len() {
                return Err(self.err("short \\u escape"));
            }
            match hex_val(self.b[self.i]) {
                Some(x) => v = v * 16 + x,
                None => return Err(self.err("bad hex digit")),
            }
            self.i += 1;
            k += 1;
        }
        Ok(v)
    }

    fn parse_string(&mut self) -> Result<String, BpeError> {
        if self.i >= self.b.len() || self.b[self.i] != b'"' {
            return Err(self.err("expected string"));
        }
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            if self.i >= self.b.len() {
                return Err(self.err("unterminated string"));
            }
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    if self.i >= self.b.len() {
                        return Err(self.err("unterminated escape"));
                    }
                    let e = self.b[self.i];
                    self.i += 1;
                    match e {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0C),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let hi = self.parse_u4()?;
                            let cp = if (0xD800..=0xDBFF).contains(&hi) {
                                if self.i + 1 >= self.b.len()
                                    || self.b[self.i] != b'\\'
                                    || self.b[self.i + 1] != b'u'
                                {
                                    return Err(self.err("lone high surrogate"));
                                }
                                self.i += 2;
                                let lo = self.parse_u4()?;
                                if !(0xDC00..=0xDFFF).contains(&lo) {
                                    return Err(self.err("bad low surrogate"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            let Some(ch) = char::from_u32(cp) else {
                                return Err(self.err("bad codepoint"));
                            };
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err(self.err("bad escape")),
                    }
                }
                _ => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| self.err("string is not utf8"))
    }

    fn parse_array(&mut self) -> Result<Json, BpeError> {
        self.i += 1; // '['
        let mut v: Vec<Json> = Vec::new();
        self.skip_ws();
        if self.i < self.b.len() && self.b[self.i] == b']' {
            self.i += 1;
            return Ok(Json::Arr(v));
        }
        loop {
            v.push(self.parse_value()?);
            self.skip_ws();
            if self.i >= self.b.len() {
                return Err(self.err("unterminated array"));
            }
            let c = self.b[self.i];
            self.i += 1;
            if c == b',' {
                continue;
            }
            if c == b']' {
                return Ok(Json::Arr(v));
            }
            return Err(self.err("expected , or ]"));
        }
    }

    fn parse_object(&mut self) -> Result<Json, BpeError> {
        self.i += 1; // '{'
        let mut ps: Vec<(String, Json)> = Vec::new();
        self.skip_ws();
        if self.i < self.b.len() && self.b[self.i] == b'}' {
            self.i += 1;
            return Ok(Json::Obj(ps));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.i >= self.b.len() || self.b[self.i] != b':' {
                return Err(self.err("expected :"));
            }
            self.i += 1;
            let val = self.parse_value()?;
            ps.push((key, val));
            self.skip_ws();
            if self.i >= self.b.len() {
                return Err(self.err("unterminated object"));
            }
            let c = self.b[self.i];
            self.i += 1;
            if c == b',' {
                continue;
            }
            if c == b'}' {
                return Ok(Json::Obj(ps));
            }
            return Err(self.err("expected , or }"));
        }
    }
}

/// Parse one JSON document and reject trailing bytes.
fn parse_document(b: &[u8]) -> Result<Json, BpeError> {
    let mut r = JsonReader { b, i: 0 };
    let v = r.parse_value()?;
    r.skip_ws();
    if r.i != b.len() {
        return Err(BpeError::Json {
            at: r.i,
            msg: "trailing bytes after document",
        });
    }
    Ok(v)
}

/// Push a char into a JSON string body, escaping only `"` and `\`.
#[cfg(test)]
fn push_json_char(out: &mut String, c: char) {
    if c == '"' || c == '\\' {
        out.push('\\');
    }
    out.push(c);
}

/// Write a fixture to the OS temp dir and return its path, unique per process.
#[cfg(test)]
fn write_temp(name: &str, body: &str) -> String {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("{}_{}", std::process::id(), name));
    std::fs::write(&path, body).expect("write fixture");
    path.to_str().expect("utf8 temp path").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The alphabet maps space to `Ġ`, printable ASCII to itself, and is a bijection onto 256
    /// distinct chars.
    #[test]
    fn byte_alphabet_is_a_bijection_with_space_as_g() {
        let m = bytes_to_unicode();
        assert_eq!(m[b' ' as usize], SPACE_CHAR);
        assert_eq!(m[b'a' as usize], 'a');
        let mut seen: Vec<char> = Vec::new();
        let mut i = 0;
        while i < 256 {
            let mut k = 0;
            while k < seen.len() {
                assert_ne!(seen[k], m[i], "byte {i} aliases byte with same char");
                k += 1;
            }
            seen.push(m[i]);
            i += 1;
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn pretokenizer_matches_the_reference_splits() {
        // cl100k: a letter run absorbs one preceding non-L/N char; `_case` is one piece.
        let p = pretokenize_cl100k("_case", 1);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], "_case");
        // a space rides with the following letter run.
        let p = pretokenize_cl100k("Hello world", 1);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0], "Hello");
        assert_eq!(p[1], " world");
        // Qwen's bare \p{N}: each digit is its own piece.
        let p = pretokenize_cl100k("3.14159", 1);
        let want = ["3", ".", "1", "4", "1", "5", "9"];
        assert_eq!(p.len(), want.len());
        let mut i = 0;
        while i < want.len() {
            assert_eq!(p[i], want[i]);
            i += 1;
        }
        // trailing run of 2+ whitespace leaves the last char for the next word's ` ?`.
        let p = pretokenize_cl100k("  hi", 1);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0], " ");
        assert_eq!(p[1], " hi");
    }

    /// A fixture with all 256 byte-level pieces, one merge, and one added token; verifies
    /// encode/decode, empty and unknown behavior, and each refusal without the model on disk.
    #[test]
    fn fixture_encode_decode_and_refusals() {
        let b2u = bytes_to_unicode();
        let mut vocab = String::new();
        let mut i = 0;
        while i < 256 {
            if i > 0 {
                vocab.push(',');
            }
            vocab.push('"');
            push_json_char(&mut vocab, b2u[i]);
            vocab.push_str("\":");
            vocab.push_str(&i.to_string());
            i += 1;
        }
        vocab.push_str(",\"ab\":256");
        let full = format!(
            r#"{{"added_tokens":[{{"id":300,"content":"<x>","special":true}}],
"pre_tokenizer":{{"type":"Sequence","pretokenizers":[{{"type":"Split","pattern":{{"Regex":"\\p{{N}}"}},"behavior":"Isolated"}},{{"type":"ByteLevel","use_regex":false}}]}},
"model":{{"type":"BPE","byte_fallback":false,"vocab":{{{vocab}}},"merges":["a b"]}}}}"#
        );
        let path = write_temp("int_w3t26_fixture.json", &full);
        let bpe = Bpe::from_tokenizer_json(&path).expect("fixture loads");

        // all 256 byte chars resolve; one merge produces id 256.
        assert_eq!(bpe.encode("ab").expect("encode"), vec![256]);
        assert_eq!(bpe.encode("a b").expect("encode"), vec![97, 32, 98]);
        assert_eq!(bpe.encode("").expect("empty"), Vec::<u32>::new());
        assert_eq!(bpe.encode("<x>").expect("special"), vec![300]);
        // a UTF-8 char outside ASCII goes through as two byte-level pieces.
        assert_eq!(bpe.encode("é").expect("encode"), vec![195, 169]);
        assert_eq!(bpe.decode(&[256]).expect("decode"), "ab");
        assert_eq!(bpe.decode(&[97, 32, 98]).expect("decode"), "a b");
        assert_eq!(bpe.decode(&[195, 169]).expect("decode"), "é");
        assert_eq!(bpe.decode(&[300]).expect("decode"), "<x>");
        assert_eq!(bpe.decode(&[]).expect("decode empty"), "");
        assert_eq!(
            bpe.decode(&bpe.encode("a b").expect("enc")).expect("dec"),
            "a b"
        );
        assert_eq!(bpe.special_id("<x>"), Some(300));
        assert_eq!(bpe.eos_id(), None);

        // unknown decode id is a refusal, not a panic.
        assert_eq!(bpe.decode(&[999_999]), Err(BpeError::UnknownToken(999_999)));

        // malformed JSON refuses.
        let bad = write_temp("int_w3t26_bad.json", "{\"model\":");
        assert!(matches!(
            Bpe::from_tokenizer_json(&bad),
            Err(BpeError::Json { .. })
        ));

        // a merge without a space refuses.
        let no_space = full.replace("\"a b\"", "\"ab\"");
        let p2 = write_temp("int_w3t26_nospace.json", &no_space);
        assert!(matches!(
            Bpe::from_tokenizer_json(&p2),
            Err(BpeError::MergeMalformed(0))
        ));

        // a pre_tokenizer with no Split regex refuses.
        let no_split = full.replace("\"Split\"", "\"Digits\"");
        let p3 = write_temp("int_w3t26_nosplit.json", &no_split);
        assert!(matches!(
            Bpe::from_tokenizer_json(&p3),
            Err(BpeError::Unsupported(_))
        ));
    }

    /// Reference ids from `fold`'s tokenizer (`fold/modelfold/src/text.rs`,
    /// `text::tokenizer::Tokenizer`) run over Qwen2.5-0.5B's `tokenizer.json`; the same process
    /// round-tripped every string. Skipped when the checkpoint is not on this box.
    #[test]
    fn qwen2_5_0_5b_matches_fold_reference() {
        let Some(path) = crate::engine::testdata::model_path("models/qwen2.5-0.5b/tokenizer.json")
        else {
            return;
        };
        let bpe = Bpe::from_tokenizer_json(&path).expect("load Qwen2.5-0.5B");
        let vectors: [(&str, &[u32]); 10] = [
            ("Hello world", &[9707, 1879]),
            ("Hello, world!", &[9707, 11, 1879, 0]),
            (
                "The quick brown fox jumps over the lazy dog.",
                &[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13],
            ),
            (
                "def f(x):\n    return x + 1\n",
                &[750, 282, 2075, 982, 262, 470, 856, 488, 220, 16, 198],
            ),
            ("你好，世界", &[108_386, 3837, 99489]),
            (
                "<|im_start|>user\nHello<|im_end|>\n",
                &[151_644, 872, 198, 9707, 151_645, 198],
            ),
            (" leading", &[6388]),
            ("", &[]),
            ("3.14159", &[18, 13, 16, 19, 16, 20, 24]),
            ("2 + 2 = 4", &[17, 488, 220, 17, 284, 220, 19]),
        ];
        let mut i = 0;
        while i < vectors.len() {
            let (text, want) = vectors[i];
            let got = bpe.encode(text).expect("encode");
            assert_eq!(got, want, "encode {text:?}");
            let back = bpe.decode(&got).expect("decode");
            assert_eq!(back, text, "round-trip {text:?}");
            i += 1;
        }
        assert_eq!(bpe.special_id("<|im_start|>"), Some(151_644));
        assert_eq!(bpe.eos_id(), Some(151_645));
        assert_eq!(bpe.vocab_size(), 151_665);
    }
}
