//! The resumable model-level transfer run: the embedded calibration batch, one layer at a time
//! through [`crate::learn::fold::layer::driver::model_step`], a crash-safe checkpoint, and the
//! student `GGUF` the engine loads.
//!
//! # What the pieces port
//!
//! - [`h0_from_chunks`] is `ht = cfg.E[ids_cal.cpu()]` (`fold_transfer.py:1514`; the same embedding
//!   gather is `h = cfg.E[ids.cpu()]` at `:94`): one token-embedding row per calibration token,
//!   row-major `[n_seq·t, hidden]`.
//! - [`save`]/[`load`] make the [`ModelState`] loop resumable: raw little-endian `i64` `Q32.32`
//!   words per finished layer (`layer-<l>.bin`) and per hidden state (`ht.bin`, `hs.bin`), with a
//!   text manifest naming each file's shape and SHA-256. A load refuses a digest or shape mismatch,
//!   and every data file is written to a temporary name and renamed, so a killed run leaves the
//!   previous checkpoint valid.
//! - [`write_student_gguf`] re-encodes the fitted `Q32.32` [`Weights`] into the teacher's own `GGUF`
//!   directory: every tensor keeps its name, dimensions, `ggml` type and offset, so the directory
//!   is byte-unchanged and only payloads move. The payload is the
//!   [`crate::learn::fold::run::convert`] scalar encoder for the tensor's type (binary16 RNE,
//!   binary32 integer map), refusing a value outside that type's finite range and never clamping.
//!
//! The checkpoint keeps the driver's word [`ModelState`] as its format (`ht.bin`, `hs.bin`,
//! `layer-<l>.bin`, eight bytes per `i64` word). That is a **format change**: the previous
//! checkpoint stored eight-byte IEEE-754 binary64 values, so `MANIFEST_MAGIC` is `v2` and a `v1`
//! manifest is refused rather than misread.
//!
//! # Subsumption
//!
//! Empirical best for "resumable layer-wise PTQ run writing a GGUF": llama.cpp `llama-quantize`
//! (C++, no resume) and GPTQModel/AutoGPTQ (Python, layer-wise, no resume). `not-subsumable`: both
//! are outside this crate's build path and neither runs this `fold_gqa` transfer, so there is no
//! upstream body to subsume; the transfer core is the in-tree
//! [`crate::learn::fold::layer::driver`].
//!
//! # Grade
//!
//! `tested`: `tests/learn_fold_run_transfer.rs` on the crate's build path (resume equivalence and
//! the checkpoint refusal), plus the ignored real-model identity and transfer runs. Nothing here is
//! `proved`.
//!
//! # Trust boundary
//!
//! Not covered: rustc/LLVM codegen, the engine's `GGUF` loader (taken as its own contract), the
//! calibration corpus, and the `fits` the checkpoint deliberately omits. The checkpoint binds each
//! file's bytes, not the `h0` or the configuration that produced them.

#![allow(
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "`l`/`n`/`o` are the driver's tensor-dimension names, and the `ht`/`hs` and \
              `*_len`/`*_sha` pairs are the two hidden states and their manifest fields, named \
              after the files they describe"
)]

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::format::gguf::{Gguf, GgufError};
use crate::format::quant;
use crate::learn::fold::layer::driver::{DriverError, LayerWeights, ModelState};
use crate::learn::fold::layer::tensor::TensorBits;
use crate::learn::fold::run::convert::{encode_f16, encode_f32, ConvertError, Teacher};
use crate::learn::gguf::write_file;
use crate::learn::grad::model::Weights;
use crate::sha256::sha256;

/// The checkpoint manifest file name.
const MANIFEST: &str = "manifest.txt";

/// The manifest's first line: format identity and version. `v2` stores `i64` `Q32.32` words; `v1`
/// stored eight-byte IEEE-754 binary64 values and is refused.
const MANIFEST_MAGIC: &str = "apeiron-fold-transfer v2";

/// The field count of one serialized [`LayerWeights`]: `attn_norm, ffn_norm, wq, bq, wk, bk, wv,
/// bv, wo, w_gate, w_up, w_down`.
const LAYER_FIELDS: usize = 12;

/// A refused transfer-run operation.
#[derive(Clone, Debug, PartialEq)]
pub enum TransferError {
    /// A file-system operation failed; the payload names the path and the OS text.
    Io(String),
    /// The checkpoint manifest is missing, malformed, or names a state that is not the geometry it
    /// declares.
    Manifest(String),
    /// A checkpoint file's SHA-256 disagrees with the manifest.
    Digest {
        /// The file whose digest moved.
        file: String,
        /// The manifest's digest.
        want: String,
        /// The file's digest.
        got: String,
    },
    /// A length, tensor shape or file size disagrees with the declared geometry.
    Shape(String),
    /// The `Q32.32` value has no finite binary16 encoding.
    F16(String),
    /// The teacher stores a tensor in a `ggml` type this writer does not re-encode.
    Type(u32),
    /// The teacher `GGUF` was refused.
    Gguf(GgufError),
    /// A `GGUF` encoder refused a `Q32.32` word.
    Convert(ConvertError),
    /// The driver refused a layer.
    Driver(DriverError),
}

impl From<GgufError> for TransferError {
    fn from(e: GgufError) -> Self {
        Self::Gguf(e)
    }
}

impl From<ConvertError> for TransferError {
    fn from(e: ConvertError) -> Self {
        Self::Convert(e)
    }
}

impl From<DriverError> for TransferError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

impl core::fmt::Display for TransferError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(m) => write!(f, "io: {m}"),
            Self::Manifest(m) => write!(f, "checkpoint manifest: {m}"),
            Self::Digest { file, want, got } => {
                write!(f, "checkpoint digest: {file}: {got} != {want}")
            }
            Self::Shape(m) => write!(f, "shape: {m}"),
            Self::F16(m) => write!(f, "F16 encoding refused: {m}"),
            Self::Type(t) => write!(f, "ggml type {t} has no re-encoder here"),
            Self::Gguf(e) => write!(f, "teacher GGUF: {e}"),
            Self::Convert(e) => write!(f, "weights conversion: {e}"),
            Self::Driver(e) => write!(f, "driver: {e}"),
        }
    }
}

impl std::error::Error for TransferError {}

/// The first `File::create`-style failure with its path.
fn io_err(path: &Path, e: &std::io::Error) -> TransferError {
    TransferError::Io(format!("{}: {e}", path.display()))
}

/// `a · b`, refused on overflow.
fn area(a: usize, b: usize, what: &str) -> Result<usize, TransferError> {
    a.checked_mul(b)
        .ok_or_else(|| TransferError::Shape(format!("{what}: {a}·{b} overflows usize")))
}

/// The lowercase hex of a digest.
fn hex(d: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in d {
        s.push(char::from_digit(u32::from(b >> 4), 16).unwrap_or('0'));
        s.push(char::from_digit(u32::from(b & 0x0f), 16).unwrap_or('0'));
    }
    s
}

/// The little-endian bytes of an `i64` word vector.
fn word_bytes(v: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode little-endian `i64` words from whole 8-byte groups.
fn words_from_bytes(b: &[u8]) -> Result<Vec<i64>, TransferError> {
    if !b.len().is_multiple_of(8) {
        return Err(TransferError::Shape(format!(
            "{} bytes is not a whole number of i64 words",
            b.len()
        )));
    }
    let mut out = Vec::with_capacity(b.len() / 8);
    for c in b.chunks_exact(8) {
        let mut w = [0u8; 8];
        w.copy_from_slice(c);
        out.push(i64::from_le_bytes(w));
    }
    Ok(out)
}

/// Write `bytes` to `name.tmp` and rename it over `name`, so a killed write leaves the previous
/// file intact.
///
/// # Complexity
/// Time `O(bytes)`; space `O(bytes)` for the temporary copy.
///
/// # Errors
/// [`TransferError::Io`] on a create, write or rename failure.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), TransferError> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes).map_err(|e| io_err(&tmp, &e))?;
    fs::rename(&tmp, path).map_err(|e| io_err(path, &e))
}

/// The token-embedding rows of the calibration chunks: `h0[row·hidden + j] = embed[id·hidden + j]`
/// over every token of every chunk, in chunk order (`ht = cfg.E[ids_cal.cpu()]`,
/// `fold_transfer.py:1514`; the same gather is `:94`).
///
/// A row copy, not a computed map: each output row is one `memcpy` of `hidden` words, so the pass
/// is memory-bound and a pool dispatch costs about what the copy saves. It stays a serial checked
/// copy; every token id is bounds-checked before its row is read.
///
/// # Complexity
/// Time and space `O(n_tok·hidden)`: one checked copy of `hidden` words per token, two sequential
/// passes over the token list.
///
/// # Errors
/// [`TransferError::Shape`] for a token id `>= teacher.vocab`, an embedding row past the table, or a
/// length product that overflows.
pub fn h0_from_chunks(teacher: &Teacher, chunks: &[Vec<u32>]) -> Result<Vec<i64>, TransferError> {
    let hid = teacher.layer_cfg.hidden;
    let n_tok = chunks
        .iter()
        .try_fold(0usize, |n, c| n.checked_add(c.len()));
    let tok = n_tok.ok_or_else(|| TransferError::Shape("token count".to_string()))?;
    let mut out = Vec::with_capacity(area(tok, hid, "h0")?);
    for c in chunks {
        for &id in c {
            let id = usize::try_from(id)
                .map_err(|_| TransferError::Shape(format!("token {id} is not a usize")))?;
            if id >= teacher.vocab {
                return Err(TransferError::Shape(format!(
                    "token {id} >= vocab {}",
                    teacher.vocab
                )));
            }
            let at = area(id, hid, "embedding row")?;
            let row = teacher.embed.get(at..at + hid).ok_or_else(|| {
                TransferError::Shape(format!("embedding row {id} past the table"))
            })?;
            out.extend_from_slice(row);
        }
    }
    Ok(out)
}

/// `true` when `dir` holds a checkpoint manifest, i.e. [`load`] can be called.
#[must_use]
pub fn checkpoint_present(dir: &str) -> bool {
    Path::new(dir).join(MANIFEST).is_file()
}

/// The owned lengths of one layer's twelve vectors at `(hidden, q_out, kv_out, inter)`, checked.
fn layer_lens(
    hidden: usize,
    q_out: usize,
    kv_out: usize,
    inter: usize,
) -> Result<[usize; LAYER_FIELDS], TransferError> {
    Ok([
        hidden,
        hidden,
        area(q_out, hidden, "q")?,
        q_out,
        area(kv_out, hidden, "k")?,
        kv_out,
        area(kv_out, hidden, "v")?,
        kv_out,
        area(hidden, q_out, "o")?,
        area(inter, hidden, "gate")?,
        area(inter, hidden, "up")?,
        area(hidden, inter, "down")?,
    ])
}

/// The twelve vectors of a layer in the manifest's fixed order. The fold driver requires the three
/// attention biases, so a layer without them is refused here.
fn layer_fields(l: &LayerWeights) -> Result<[&[i64]; LAYER_FIELDS], TransferError> {
    let missing = || TransferError::Shape("a layer bias is absent".to_string());
    Ok([
        &l.attn_norm,
        &l.ffn_norm,
        &l.wq,
        l.bq.as_deref().ok_or_else(missing)?,
        &l.wk,
        l.bk.as_deref().ok_or_else(missing)?,
        &l.wv,
        l.bv.as_deref().ok_or_else(missing)?,
        &l.wo,
        &l.w_gate,
        &l.w_up,
        &l.w_down,
    ])
}

/// The geometry one checkpoint layer file declares: `(hidden, q_out, kv_out, inter)`.
fn layer_geometry(l: &LayerWeights) -> Result<(usize, usize, usize, usize), TransferError> {
    let hidden = l.attn_norm.len();
    if hidden == 0 || l.ffn_norm.len() != hidden {
        return Err(TransferError::Shape("layer norm widths".to_string()));
    }
    if !l.wq.len().is_multiple_of(hidden)
        || !l.wk.len().is_multiple_of(hidden)
        || !l.w_gate.len().is_multiple_of(hidden)
    {
        return Err(TransferError::Shape("layer weight width".to_string()));
    }
    let q_out = l.wq.len() / hidden;
    let kv_out = l.wk.len() / hidden;
    let inter = l.w_gate.len() / hidden;
    Ok((hidden, q_out, kv_out, inter))
}

/// Write `st` as a checkpoint under `dir`: `ht.bin`, `hs.bin`, one `layer-<l>.bin` per finished
/// layer, and `manifest.txt` last (each through [`write_atomic`]).
///
/// The manifest records `next`, the layer geometry, the five `bits` fields, and every file's length
/// and SHA-256; the per-layer `fits` are not written, because the student `GGUF` needs only the
/// weights and `bits`. A state with `next == 0` writes the two hidden-state files and a manifest
/// with no layer lines (the geometry is unknown until the first layer is quantised).
///
/// # Complexity
/// Time and space `O(bytes written)` = `O((next·(hidden² + hidden·inter) + n_seq·t·hidden)·8)`; the
/// per-layer buffers are built one at a time.
///
/// # Errors
/// [`TransferError::Shape`] when `next`, `layers` and `fits` disagree or a layer's vectors do not
/// match the geometry its first layer declares; [`TransferError::Io`] on a write failure.
pub fn save(dir: &str, st: &ModelState) -> Result<(), TransferError> {
    if st.next != st.layers.len() || st.fits.len() != st.layers.len() {
        return Err(TransferError::Shape(format!(
            "next {} / layers {} / fits {}",
            st.next,
            st.layers.len(),
            st.fits.len()
        )));
    }
    let dirp = Path::new(dir);
    fs::create_dir_all(dirp).map_err(|e| io_err(dirp, &e))?;

    let mut manifest = String::new();
    let (hidden, geom) = if st.layers.is_empty() {
        (0usize, None)
    } else {
        let g = layer_geometry(&st.layers[0])?;
        (g.0, Some(g))
    };
    let _ = writeln!(manifest, "{MANIFEST_MAGIC}");
    let _ = writeln!(manifest, "next {}", st.next);
    let _ = writeln!(manifest, "hidden {hidden}");
    let _ = writeln!(
        manifest,
        "bits {} {} {} {} {}",
        st.bits.nsign, st.bits.nr, st.bits.nc, st.bits.nd, st.bits.ncorr
    );

    for (name, v) in [("ht.bin", &st.ht), ("hs.bin", &st.hs)] {
        let bytes = word_bytes(v);
        write_atomic(&dirp.join(name), &bytes)?;
        let _ = writeln!(
            manifest,
            "{} {} {}",
            name,
            bytes.len(),
            hex(&sha256(&bytes))
        );
    }

    if let Some((hid, q_out, kv_out, inter)) = geom {
        let want = layer_lens(hid, q_out, kv_out, inter)?;
        for (l, lw) in st.layers.iter().enumerate() {
            let fields = layer_fields(lw)?;
            let mut bytes = Vec::new();
            for (i, f) in fields.iter().enumerate() {
                if f.len() != want[i] {
                    return Err(TransferError::Shape(format!(
                        "layer {l} field {i}: {} != {}",
                        f.len(),
                        want[i]
                    )));
                }
                bytes.extend_from_slice(&word_bytes(f));
            }
            let name = format!("layer-{l}.bin");
            write_atomic(&dirp.join(&name), &bytes)?;
            let _ = writeln!(
                manifest,
                "layer {l} {hid} {q_out} {kv_out} {inter} {} {}",
                bytes.len() / 8,
                hex(&sha256(&bytes))
            );
        }
    }
    write_atomic(&dirp.join(MANIFEST), manifest.as_bytes())
}

/// One parsed manifest line.
struct ManifestHead {
    next: usize,
    hidden: usize,
    bits: TensorBits,
    ht_len: usize,
    ht_sha: String,
    hs_len: usize,
    hs_sha: String,
}

/// Split a manifest line into its whitespace-separated tokens.
fn tokens(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

/// Parse a decimal token as a `usize`.
fn num(tok: &str, what: &str) -> Result<usize, TransferError> {
    tok.parse::<usize>()
        .map_err(|_| TransferError::Manifest(format!("{what}: {tok}")))
}

/// Parse a decimal token as a `u64`.
fn num_u64(tok: &str, what: &str) -> Result<u64, TransferError> {
    tok.parse::<u64>()
        .map_err(|_| TransferError::Manifest(format!("{what}: {tok}")))
}

/// Parse the manifest head (`next`, `hidden`, `bits`, the two hidden-state lines).
fn parse_head(lines: &[&str]) -> Result<ManifestHead, TransferError> {
    if lines.len() < 6 || lines[0] != MANIFEST_MAGIC {
        return Err(TransferError::Manifest("header".to_string()));
    }
    let t = tokens(lines[1]);
    if t.len() != 2 || t[0] != "next" {
        return Err(TransferError::Manifest("next".to_string()));
    }
    let next = num(t[1], "next")?;
    let t = tokens(lines[2]);
    if t.len() != 2 || t[0] != "hidden" {
        return Err(TransferError::Manifest("hidden".to_string()));
    }
    let hidden = num(t[1], "hidden")?;
    let t = tokens(lines[3]);
    if t.len() != 6 || t[0] != "bits" {
        return Err(TransferError::Manifest("bits".to_string()));
    }
    let bits = TensorBits {
        nsign: num_u64(t[1], "nsign")?,
        nr: num_u64(t[2], "nr")?,
        nc: num_u64(t[3], "nc")?,
        nd: num_u64(t[4], "nd")?,
        ncorr: num_u64(t[5], "ncorr")?,
    };
    let (ht_len, ht_sha) = hidden_line(lines[4], "ht.bin")?;
    let (hs_len, hs_sha) = hidden_line(lines[5], "hs.bin")?;
    Ok(ManifestHead {
        next,
        hidden,
        bits,
        ht_len,
        ht_sha,
        hs_len,
        hs_sha,
    })
}

/// Parse a hidden-state manifest line `ht.bin <len> <sha>`.
fn hidden_line(line: &str, name: &str) -> Result<(usize, String), TransferError> {
    let t = tokens(line);
    if t.len() != 3 || t[0] != name {
        return Err(TransferError::Manifest(name.to_string()));
    }
    Ok((num(t[1], name)?, t[2].to_string()))
}

/// Read one checkpoint file, refusing a length or SHA-256 that disagrees with the manifest.
///
/// # Complexity
/// Time `O(bytes)` for the read and one SHA-256 pass; space `O(bytes)`.
fn read_checked(
    dir: &Path,
    name: &str,
    len: usize,
    sha: Option<&str>,
) -> Result<Vec<u8>, TransferError> {
    let path = dir.join(name);
    let bytes = fs::read(&path).map_err(|e| io_err(&path, &e))?;
    if bytes.len() != len {
        return Err(TransferError::Shape(format!(
            "{name}: {} bytes != manifest {len}",
            bytes.len()
        )));
    }
    if let Some(want) = sha {
        let got = hex(&sha256(&bytes));
        if got != want {
            return Err(TransferError::Digest {
                file: name.to_string(),
                want: want.to_string(),
                got,
            });
        }
    }
    Ok(bytes)
}

/// Split one layer buffer at `lens` into the twelve vectors, in the manifest's order.
///
/// # Errors
/// [`TransferError::Shape`] when the buffer length is not exactly `Σ lens`.
fn split_layer(
    bytes: &[u8],
    lens: &[usize; LAYER_FIELDS],
    l: usize,
) -> Result<LayerWeights, TransferError> {
    let total: usize = lens.iter().try_fold(0usize, |a, &b| {
        a.checked_add(b)
            .ok_or_else(|| TransferError::Shape(format!("layer {l} length")))
    })?;
    if bytes.len() != total * 8 {
        return Err(TransferError::Shape(format!(
            "layer {l}: {} bytes != {}",
            bytes.len(),
            total * 8
        )));
    }
    let mut at = 0usize;
    let mut parts: Vec<Vec<i64>> = Vec::with_capacity(LAYER_FIELDS);
    for &n in lens {
        let b = bytes
            .get(at..at + n * 8)
            .ok_or_else(|| TransferError::Shape(format!("layer {l} field split")))?;
        parts.push(words_from_bytes(b)?);
        at += n * 8;
    }
    let take = |i: usize| -> Vec<i64> { parts[i].clone() };
    Ok(LayerWeights {
        attn_norm: take(0),
        ffn_norm: take(1),
        wq: take(2),
        bq: Some(take(3)),
        wk: take(4),
        bk: Some(take(5)),
        wv: take(6),
        bv: Some(take(7)),
        wo: take(8),
        w_gate: take(9),
        w_up: take(10),
        w_down: take(11),
    })
}

/// Read the checkpoint under `dir` back into a [`ModelState`] whose `next` is the manifest's, whose
/// layers are the finished prefix, and whose `fits` are one empty vector per finished layer.
///
/// # Complexity
/// Time and space `O(bytes read)` = `O((next·(hidden² + hidden·inter) + n_seq·t·hidden)·8)`.
///
/// # Errors
/// [`TransferError::Io`] on a missing or unreadable file; [`TransferError::Manifest`] on a malformed
/// manifest; [`TransferError::Shape`] on a length, layer count or geometry disagreement;
/// [`TransferError::Digest`] on a file whose SHA-256 is not the manifest's.
pub fn load(dir: &str) -> Result<ModelState, TransferError> {
    let dirp = Path::new(dir);
    let manifest_path = dirp.join(MANIFEST);
    let text = fs::read_to_string(&manifest_path).map_err(|e| io_err(&manifest_path, &e))?;
    let lines: Vec<&str> = text.lines().collect();
    let head = parse_head(&lines)?;
    if lines.len() != 6 + head.next {
        return Err(TransferError::Manifest(format!(
            "{} lines for next {}",
            lines.len(),
            head.next
        )));
    }
    let ht = words_from_bytes(&read_checked(
        dirp,
        "ht.bin",
        head.ht_len,
        Some(&head.ht_sha),
    )?)?;
    let hs = words_from_bytes(&read_checked(
        dirp,
        "hs.bin",
        head.hs_len,
        Some(&head.hs_sha),
    )?)?;
    if hs.len() != ht.len() {
        return Err(TransferError::Shape(format!(
            "hs {} != ht {}",
            hs.len(),
            ht.len()
        )));
    }

    let mut layers: Vec<LayerWeights> = Vec::with_capacity(head.next);
    for l in 0..head.next {
        let t = tokens(lines[6 + l]);
        if t.len() != 8 || t[0] != "layer" || num(t[1], "layer index")? != l {
            return Err(TransferError::Manifest(format!("layer line {l}")));
        }
        let hid = num(t[2], "hidden")?;
        let q_out = num(t[3], "q_out")?;
        let kv_out = num(t[4], "kv_out")?;
        let inter = num(t[5], "inter")?;
        if hid != head.hidden {
            return Err(TransferError::Shape(format!(
                "layer {l} hidden {hid} != {}",
                head.hidden
            )));
        }
        let n = num(t[6], "layer length")?;
        let bytes = n
            .checked_mul(8)
            .ok_or_else(|| TransferError::Shape(format!("layer {l} byte count")))?;
        let name = format!("layer-{l}.bin");
        let raw = read_checked(dirp, &name, bytes, Some(t[7]))?;
        let lens = layer_lens(hid, q_out, kv_out, inter)?;
        layers.push(split_layer(&raw, &lens, l)?);
    }
    Ok(ModelState {
        next: head.next,
        ht,
        hs,
        layers,
        bits: head.bits,
        fits: vec![Vec::new(); head.next],
    })
}

/// The student values of the `GGUF` tensor `name`, or `None` when the teacher carries a tensor the
/// [`Weights`] struct does not ([`write_student_gguf`] then copies its teacher payload).
fn student_values<'a>(w: &'a Weights, name: &[u8]) -> Option<&'a [i64]> {
    let s = core::str::from_utf8(name).ok()?;
    match s {
        "token_embd.weight" => Some(&w.token_embd),
        "output_norm.weight" => Some(&w.output_norm),
        "output.weight" => Some(&w.lm_head),
        _ => {
            let rest = s.strip_prefix("blk.")?;
            let (num, suffix) = rest.split_once('.')?;
            let l: usize = num.parse().ok()?;
            let lw = w.layers.get(l)?;
            match suffix {
                "attn_norm.weight" => Some(&lw.attn_norm),
                "attn_q.weight" => Some(&lw.wq),
                "attn_q.bias" => lw.bq.as_deref(),
                "attn_k.weight" => Some(&lw.wk),
                "attn_k.bias" => lw.bk.as_deref(),
                "attn_v.weight" => Some(&lw.wv),
                "attn_v.bias" => lw.bv.as_deref(),
                "attn_output.weight" => Some(&lw.wo),
                "ffn_norm.weight" => Some(&lw.ffn_norm),
                "ffn_gate.weight" => Some(&lw.w_gate),
                "ffn_up.weight" => Some(&lw.w_up),
                "ffn_down.weight" => Some(&lw.w_down),
                _ => None,
            }
        }
    }
}

/// Re-encode `v` into the teacher's `ggml` type.
///
/// # Errors
/// [`TransferError::Type`] for a type other than F16 or F32; [`TransferError::F16`] when a word has
/// no finite binary16 encoding; [`TransferError::Convert`] for any other refusal.
fn encode(ggml_type: u32, v: &[i64]) -> Result<Vec<u8>, TransferError> {
    match ggml_type {
        quant::GGML_TYPE_F16 => encode_f16(v).map_err(|e| TransferError::F16(e.to_string())),
        quant::GGML_TYPE_F32 => Ok(encode_f32(v)),
        other => Err(TransferError::Type(other)),
    }
}

/// Write `student` as a `GGUF` under the teacher's directory at `teacher_path`, to `out_path`.
///
/// The teacher header is parsed by [`Gguf::open`]; every tensor keeps its name, dimensions, `ggml`
/// type and offset, so the directory is unchanged. Each tensor the student carries is re-encoded
/// from its `Q32.32` words ([`encode`]) — one pooled pass per tensor — and a tensor name the
/// [`Weights`] struct does not carry copies the teacher payload byte-for-byte.
///
/// # Complexity
/// Time and space `O(file bytes)` = `O(2·W)` for an F16 teacher of `W` weights: one pooled encode
/// pass per tensor plus one header pass; the whole data section is held once before the atomic
/// write.
///
/// # Errors
/// [`TransferError::Gguf`] on a teacher refusal; [`TransferError::Type`] for a source type other
/// than F16/F32 or a tensor with no geometry; [`TransferError::Shape`] when a tensor's element
/// count, payload size or offset disagrees with the student; [`TransferError::F16`] for a value
/// outside binary16's finite range; [`TransferError::Io`] on a write failure.
pub fn write_student_gguf(
    teacher_path: &str,
    student: &Weights,
    out_path: &str,
) -> Result<(), TransferError> {
    let g = Gguf::open(teacher_path)?;
    let data_len = g
        .file_len
        .checked_sub(g.header.data0)
        .ok_or_else(|| TransferError::Shape("teacher data section".to_string()))?;
    let data_len = usize::try_from(data_len)
        .map_err(|_| TransferError::Shape("teacher data section".to_string()))?;
    let mut data = vec![0u8; data_len];
    for t in &g.header.tensors {
        let n = t.n_bytes()?.ok_or(TransferError::Type(t.ggml_type))?;
        let n_el = t.n_elements()?;
        let off = usize::try_from(t.offset)
            .map_err(|_| TransferError::Shape("tensor offset".to_string()))?;
        let n = usize::try_from(n).map_err(|_| TransferError::Shape("tensor size".to_string()))?;
        let slot = data
            .get_mut(off..off + n)
            .ok_or_else(|| TransferError::Shape("tensor past the data section".to_string()))?;
        let name = String::from_utf8_lossy(&t.name).into_owned();
        if let Some(v) = student_values(student, &t.name) {
            if u64::try_from(v.len()).ok() != Some(n_el) {
                return Err(TransferError::Shape(format!(
                    "{name}: {} weights != {n_el} directory elements",
                    v.len()
                )));
            }
            let bytes = encode(t.ggml_type, v)?;
            if bytes.len() != slot.len() {
                return Err(TransferError::Shape(format!("{name}: encoded size")));
            }
            slot.copy_from_slice(&bytes);
        } else {
            let raw = g.tensor_bytes(t)?;
            if raw.len() != slot.len() {
                return Err(TransferError::Shape(format!(
                    "{name}: teacher payload size"
                )));
            }
            slot.copy_from_slice(&raw);
        }
    }
    let file = write_file(&g.header, &data)?;
    write_atomic(Path::new(out_path), &file)
}
