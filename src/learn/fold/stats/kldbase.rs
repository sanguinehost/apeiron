//! Reader of llama.cpp's `--kl-divergence-base` file, streamed one chunk at a time.
//!
//! llama.cpp `ece963f41b0b02d7a0d61436ae365762c073a4c8` writes `"_logits_"`, `n_ctx`, `n_vocab`,
//! `n_chunk`, the tokens, then per chunk `n_ctx − 1 − n_ctx/2` rows of `nv = 2·((n_vocab+1)/2) + 4`
//! `u16` (`tools/perplexity/perplexity.cpp:516–527`, `:1695–1800`, `:191`, `:203`); row `r` is the
//! distribution at position `n_ctx/2 + r` predicting the token one position later (`:1871`). The
//! writer's BOS substitution at position 0 stores the original token and does not apply here. A bad
//! signature, `n_ctx < 2`, a zero width or count, a negative token and a truncated block are
//! `InvalidData`. Grade `tested`.
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// llama.cpp's signature: written at `perplexity.cpp:468`, checked at `:1711`.
const MAGIC: [u8; 8] = *b"_logits_";

fn invalid(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.to_string())
}

/// The row width in `u16` words, `2·((n_vocab + 1)/2) + 4` (`perplexity.cpp:191`).
fn row_words(n_vocab: usize) -> Option<usize> {
    let half = n_vocab.checked_add(1)?.checked_div(2)?;
    half.checked_mul(2)?.checked_add(4)
}

/// A streamed base file: header and tokens on [`KldBase::open`], one row block per
/// [`KldBase::next_chunk`].
#[derive(Debug)]
pub struct KldBase {
    file: File,
    n_ctx: usize,
    n_vocab: usize,
    n_chunk: usize,
    next: usize,
    nv: usize,
    tokens: Vec<u32>,
}

impl KldBase {
    /// Open a base file and read its header and token block. Time and space `O(n_ctx·n_chunk)`:
    /// the token block is read and converted once, no row block.
    ///
    /// # Errors
    ///
    /// `InvalidData` for a bad signature, `n_ctx < 2`, a zero width or count, a negative token id
    /// and a truncated header, token or row block.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut h = [0u8; 20];
        file.read_exact(&mut h)?;
        if h[..8] != MAGIC {
            return Err(invalid("not a llama.cpp `_logits_` base file"));
        }
        let field = |i: usize| {
            let mut b = [0u8; 4];
            b.copy_from_slice(&h[8 + i..12 + i]);
            usize::try_from(u32::from_le_bytes(b)).map_err(|_| invalid("header field"))
        };
        let (n_ctx, n_vocab, n_chunk) = (field(0)?, field(4)?, field(8)?);
        if n_ctx < 2 || n_vocab == 0 || n_chunk == 0 {
            return Err(invalid("n_ctx < 2, n_vocab == 0 or n_chunk == 0"));
        }
        let nv = row_words(n_vocab).ok_or_else(|| invalid("width"))?;
        let count = n_ctx.checked_mul(n_chunk).ok_or_else(|| invalid("count"))?;
        let mut raw = vec![0u8; count.checked_mul(4).ok_or_else(|| invalid("bytes"))?];
        file.read_exact(&mut raw)?;
        let tokens = raw
            .chunks_exact(4)
            .map(|b| u32::try_from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect::<Result<Vec<u32>, _>>()
            .map_err(|_| invalid("negative token id"))?;
        Ok(Self {
            file,
            n_ctx,
            n_vocab,
            n_chunk,
            next: 0,
            nv,
            tokens,
        })
    }

    /// Chunks stored (`n_chunk`).
    #[must_use]
    pub fn n_chunk(&self) -> usize {
        self.n_chunk
    }

    /// Read the next chunk's tokens and row block; `InvalidData` once every chunk has been read.
    /// Time `O(n_ctx + rows·n_vocab)`; space `O(rows·n_vocab + n_ctx)` on [`KldChunk`]. `next·n_ctx`
    /// cannot leave `usize`: `open` checked `n_ctx·n_chunk`.
    ///
    /// # Errors
    ///
    /// `InvalidData` once every chunk has been read and for a truncated row block.
    pub fn next_chunk(&mut self) -> io::Result<KldChunk> {
        if self.next >= self.n_chunk {
            return Err(invalid("all chunks read"));
        }
        let rows = self.n_ctx - 1 - self.n_ctx / 2;
        let words = rows.checked_mul(self.nv).ok_or_else(|| invalid("rows"))?;
        let mut raw = vec![0u8; words.checked_mul(2).ok_or_else(|| invalid("bytes"))?];
        self.file.read_exact(&mut raw)?;
        let at = self.next * self.n_ctx;
        let tokens = self
            .tokens
            .get(at..at + self.n_ctx)
            .ok_or_else(|| invalid("tokens"))?;
        self.next += 1;
        let words = raw
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        Ok(KldChunk {
            tokens: tokens.to_vec(),
            n_vocab: self.n_vocab,
            rows,
            nv: self.nv,
            words,
        })
    }
}

/// One chunk of a base file: its tokens and its `rows` rows of `nv` `u16` words.
#[derive(Debug, Clone)]
pub struct KldChunk {
    /// The chunk's own evaluation tokens.
    pub tokens: Vec<u32>,
    /// Vocabulary width of a row.
    pub n_vocab: usize,
    /// Scored rows, `n_ctx − 1 − n_ctx/2`.
    pub rows: usize,
    nv: usize,
    words: Vec<u16>,
}

impl KldChunk {
    /// Row `r`'s dequantized `log p(v)`: `scale·code[v] + min_log_prob` as two `f32` words then the
    /// `u16` codes (`perplexity.cpp:203`); `None` past the row or the vocabulary.
    #[must_use]
    pub fn log_prob(&self, r: usize, v: usize) -> Option<f64> {
        if v >= self.n_vocab {
            return None;
        }
        let at = r.checked_mul(self.nv)?;
        let word = |i: usize| {
            let b = *self.words.get(at + i + 1)?;
            Some(u32::from(*self.words.get(at + i)?) | (u32::from(b) << 16))
        };
        let scale = f64::from(f32::from_bits(word(0)?));
        let min = f64::from(f32::from_bits(word(2)?));
        Some(scale.mul_add(f64::from(*self.words.get(at + 4 + v)?), min))
    }

    /// Row `r`'s lowest maximum code index, as llama.cpp's strict comparison keeps it.
    #[must_use]
    pub fn argmax(&self, r: usize) -> Option<usize> {
        let at = r.checked_mul(self.nv)?.checked_add(4)?;
        let row = self.words.get(at..at.checked_add(self.n_vocab)?)?;
        let max = |(_, c): &(usize, &u16)| **c;
        row.iter().enumerate().rev().max_by_key(max).map(|(i, _)| i)
    }
}
