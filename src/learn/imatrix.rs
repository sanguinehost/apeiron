//! llama.cpp importance-matrix reader: per-tensor activation vectors from an imatrix `GGUF`.
//!
//! An imatrix file stores, for every calibrated tensor `name`, two `F32` tensors:
//!
//! * `encode[0]`: `<name>.in_sum2`, shape `[nval/nmat, nmat]` — the raw per-column sum of squared
//!   activations `Σ_t x_{t,j}²`, per matrix `j`;
//! * `encode[1]`: `<name>.counts`, shape `[1, nmat]` — the number of activation rows accumulated
//!   for each matrix.
//!
//! They are written by `tools/imatrix/imatrix.cpp:611-629` (`save_imatrix`:
//! `ggml_new_tensor_2d(..., GGML_TYPE_F32, nval/nmat, nmat)` named `"%s.in_sum2"` and
//! `ggml_new_tensor_2d(..., GGML_TYPE_F32, 1, nmat)` named `"%s.counts"`), both type-checked `F32`.
//! The loader pairs them by suffix and reads the raw sums and `lround`-ed counts
//! (`common/imatrix-loader.cpp:118-168`). `llama-quantize` then forms the importance vector that it
//! hands to `quantize_row_q4_K_impl` as `quant_weights`: it divides the sums by the per-matrix count,
//! and substitutes `1` where the count is `0`
//! (`tools/quantize/quantize.cpp:196-212`: `e[j*ne0 + i] = entry.sums[j*ne0 + i] / count` for
//! `count > 0`, else `1`). That division is the whole contract of this module: [`read_imatrix`]
//! returns exactly that per-column vector, so [`crate::learn::seal`]'s `qw` is the same vector
//! llama.cpp weights its `Q4_K` fit with.
//!
//! llama.cpp rev `ece963f41` (`tools/imatrix/imatrix.cpp` sha256 `a56c35fe501a2b4d…`,
//! `common/imatrix-loader.cpp` sha256 `fcd471a2c502f5d7…`, `tools/quantize/quantize.cpp` sha256
//! `5a3e413ca1022d34…`).
//!
//! **Evidence.** The layout round-trip, the division and the count-zero substitution are `tested`
//! (`#[cfg(test)]` below). This is a byte-layout reader, not a refinement, so no `proved` claim is
//! made.
//!
//! **Complexity** (AGENTS §8). [`read_imatrix`] is `O(bytes of the imatrix payload)` time — one pass
//! to index the directory plus one read per entry — and `O(Σ nval)` space for the returned vectors.
//! [`ImportanceMatrix::vector`] is `O(1)` amortized (one hash lookup).

use std::collections::HashMap;

use crate::format::gguf::{Gguf, GgufError, TensorDesc};

/// A refused imatrix operation.
#[derive(Debug)]
pub enum ImatrixError {
    /// The underlying GGUF reader refused.
    Gguf(GgufError),
    /// An `in_sum2` tensor has no matching `counts` tensor, or the two disagree in length.
    Mismatched {
        /// The base tensor name the pair describes.
        name: String,
        /// What disagrees.
        detail: &'static str,
    },
    /// The element counts of an `in_sum2` and its `counts` do not divide into whole matrices.
    Shape {
        /// The base tensor name.
        name: String,
        /// Number of summed values.
        nval: usize,
        /// Number of per-matrix counts.
        nmat: usize,
    },
    /// An element count did not fit the host `usize`.
    LengthOverflow,
}

impl core::fmt::Display for ImatrixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ImatrixError::Gguf(e) => write!(f, "{e}"),
            ImatrixError::Mismatched { name, detail } => {
                write!(f, "imatrix entry {name}: {detail}")
            }
            ImatrixError::Shape { name, nval, nmat } => {
                write!(
                    f,
                    "imatrix entry {name}: {nval} values over {nmat} matrices"
                )
            }
            ImatrixError::LengthOverflow => write!(f, "imatrix length overflow"),
        }
    }
}

impl std::error::Error for ImatrixError {}

impl From<GgufError> for ImatrixError {
    fn from(e: GgufError) -> Self {
        ImatrixError::Gguf(e)
    }
}

/// Per-tensor importance vectors read from a llama.cpp imatrix `GGUF`.
///
/// Each vector is the exact per-column quantity `llama-quantize` uses as `quant_weights`: the
/// `in_sum2` sums divided by the per-matrix `counts`, with `1` where a count is `0`.
pub struct ImportanceMatrix {
    vectors: HashMap<String, Vec<f32>>,
    chunk_count: u32,
    chunk_size: u32,
}

impl ImportanceMatrix {
    /// The importance vector for tensor `name`, or `None` when the imatrix has no entry. Length is
    /// the tensor's `nval/nmat` (the contraction dimension for a dense tensor).
    pub fn vector(&self, name: &str) -> Option<&[f32]> {
        self.vectors.get(name).map(Vec::as_slice)
    }

    /// Number of tensor entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether there are no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// `imatrix.chunk_count` metadata: the number of chunks the matrix was computed over, or `0`.
    #[must_use]
    pub fn chunk_count(&self) -> u32 {
        self.chunk_count
    }

    /// `imatrix.chunk_size` metadata: the tokens per chunk, or `0`.
    #[must_use]
    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }
}

/// The base name when `name` ends in `suffix`, as `&str` (the base is always valid UTF-8 for the
/// ASCII tensor names an imatrix carries); `None` otherwise.
fn strip_suffix<'a>(name: &'a [u8], suffix: &[u8]) -> Option<&'a str> {
    let n = name.len();
    let s = suffix.len();
    if n < s || &name[n - s..] != suffix {
        return None;
    }
    core::str::from_utf8(&name[..n - s]).ok()
}

/// Decode a raw `F32` payload into `f32` values (little-endian), one per element.
///
/// A payload whose length is not a whole number of `f32` refuses through [`ImatrixError::Shape`]'s
/// caller; here a trailing partial word is dropped by `chunks_exact`. Cost `O(bytes)`.
fn f32_payload(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Index the imatrix directory by base tensor name, returning `(in_sum2, counts)` descriptors.
///
/// The reader keys entries exactly as llama.cpp's loader does (`common/imatrix-loader.cpp:118-133`):
/// strip `.in_sum2` and `.counts`, pair the halves. Cost `O(tensors)`.
fn pair_descriptors(tensors: &[TensorDesc]) -> HashMap<String, (Option<usize>, Option<usize>)> {
    let mut pairs: HashMap<String, (Option<usize>, Option<usize>)> = HashMap::new();
    for (i, t) in tensors.iter().enumerate() {
        if let Some(base) = strip_suffix(&t.name, b".in_sum2") {
            pairs.entry(base.to_string()).or_insert((None, None)).0 = Some(i);
        } else if let Some(base) = strip_suffix(&t.name, b".counts") {
            pairs.entry(base.to_string()).or_insert((None, None)).1 = Some(i);
        }
    }
    pairs
}

/// Read a llama.cpp imatrix `GGUF` into per-tensor importance vectors.
///
/// The returned map keys are base tensor names (`blk.3.ffn_down.weight`), valued by the
/// `in_sum2/counts` vector (`tools/quantize/quantize.cpp:196-212`). A missing or length-mismatched
/// half of an entry, or a value count that is not a whole multiple of the matrix count, refuses.
///
/// # Errors
///
/// * any [`GgufError`] from opening or reading the file, wrapped as [`ImatrixError::Gguf`];
/// * [`ImatrixError::Mismatched`] for an `in_sum2` with no paired `counts` (or vice versa);
/// * [`ImatrixError::Shape`] when a value count does not divide across the matrices.
pub fn read_imatrix(path: &str) -> Result<ImportanceMatrix, ImatrixError> {
    let g = Gguf::open(path)?;

    let mut chunk_count = 0u32;
    let mut chunk_size = 0u32;
    for (k, v) in &g.header.kv {
        if k.as_slice() == b"imatrix.chunk_count" {
            if let Some(x) = v.as_i64() {
                chunk_count = u32::try_from(x).unwrap_or(0);
            }
        } else if k.as_slice() == b"imatrix.chunk_size" {
            if let Some(x) = v.as_i64() {
                chunk_size = u32::try_from(x).unwrap_or(0);
            }
        }
    }

    let pairs = pair_descriptors(&g.header.tensors);

    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, (si, ci)) in pairs {
        let (Some(si), Some(ci)) = (si, ci) else {
            return Err(ImatrixError::Mismatched {
                name,
                detail: "in_sum2 and counts are not both present",
            });
        };
        let sums_bytes = g.tensor_bytes(&g.header.tensors[si])?;
        let counts_bytes = g.tensor_bytes(&g.header.tensors[ci])?;
        let sums = f32_payload(&sums_bytes);
        let counts = f32_payload(&counts_bytes);
        let nval = sums.len();
        let nmat = counts.len();
        if nmat == 0 || !nval.is_multiple_of(nmat) {
            return Err(ImatrixError::Shape { name, nval, nmat });
        }
        let ne0 = nval / nmat;
        let mut tau = vec![0f32; nval];
        for j in 0..nmat {
            // `imatrix-loader.cpp:166` rounds the count; `quantize.cpp:202` casts it to `f32`.
            let count = counts[j].round();
            for i in 0..ne0 {
                tau[j * ne0 + i] = if count > 0.0 {
                    sums[j * ne0 + i] / count
                } else {
                    1.0
                };
            }
        }
        vectors.insert(name, tau);
    }

    Ok(ImportanceMatrix {
        vectors,
        chunk_count,
        chunk_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::gguf::{GVal, GgufHeader};
    use crate::format::quant::GGML_TYPE_F32;
    use crate::learn::gguf::write_file;

    /// Build a synthetic imatrix file: one entry `<name>` with `sums` declared at `dims_sums` and
    /// `counts` at `[counts.len()]`, written with the crate's own GGUF writer, so this exercises
    /// writer and reader together.
    fn write_synth(
        name: &str,
        sums: &[f32],
        counts: &[f32],
        dims_sums: Vec<u64>,
    ) -> std::path::PathBuf {
        let mut sums_bytes = Vec::new();
        for v in sums {
            sums_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut counts_bytes = Vec::new();
        for v in counts {
            counts_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let tensors = vec![
            TensorDesc {
                name: format!("{name}.in_sum2").into_bytes(),
                dims: dims_sums,
                ggml_type: GGML_TYPE_F32,
                offset: 0,
            },
            TensorDesc {
                name: format!("{name}.counts").into_bytes(),
                dims: vec![counts.len() as u64],
                ggml_type: GGML_TYPE_F32,
                offset: 0,
            },
        ];
        let kv = vec![
            (b"general.alignment".to_vec(), GVal::Int(32, 4)),
            (b"imatrix.chunk_count".to_vec(), GVal::Int(7, 4)),
            (b"imatrix.chunk_size".to_vec(), GVal::Int(512, 4)),
        ];
        let mut header = GgufHeader {
            version: 3,
            kv,
            tensors,
            alignment: 32,
            header_end: 0,
            data0: 0,
        };
        let mut data = Vec::new();
        data.extend_from_slice(&sums_bytes);
        while !data.len().is_multiple_of(32) {
            data.push(0);
        }
        let counts_off = data.len();
        data.extend_from_slice(&counts_bytes);
        header.tensors[1].offset = counts_off as u64;

        let out = write_file(&header, &data).expect("write synthetic imatrix");
        let dir = std::env::temp_dir().join(format!(
            "apeiron-imatrix-{}-{}.gguf",
            std::process::id(),
            name.replace('.', "_")
        ));
        std::fs::write(&dir, out).expect("write temp imatrix");
        dir
    }

    #[test]
    fn division_reproduces_quantize_weights() {
        let sums = vec![8.0f32, 16.0, 40.0, -2.0, 800.0, 4.0];
        let counts = vec![4.0f32];
        let path = write_synth("blk.0.ffn_down.weight", &sums, &counts, vec![6]);
        let m = read_imatrix(path.to_str().unwrap()).expect("read");
        let tau = m.vector("blk.0.ffn_down.weight").expect("entry present");
        assert_eq!(tau, &[2.0f32, 4.0, 10.0, -0.5, 200.0, 1.0]);
        assert_eq!(m.len(), 1);
        assert_eq!(m.chunk_count(), 7);
        assert_eq!(m.chunk_size(), 512);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn zero_count_substitutes_one() {
        let sums = vec![3.0f32, 9.0];
        let counts = vec![0.0f32];
        let path = write_synth("blk.1.attn_q.weight", &sums, &counts, vec![2]);
        let m = read_imatrix(path.to_str().unwrap()).expect("read");
        let tau = m.vector("blk.1.attn_q.weight").expect("entry present");
        assert_eq!(tau, &[1.0f32, 1.0]);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn missing_entry_is_none() {
        let path = write_synth("blk.2.attn_k.weight", &[1.0f32, 2.0], &[2.0f32], vec![2]);
        let m = read_imatrix(path.to_str().unwrap()).expect("read");
        assert!(m.vector("blk.9.ffn_up.weight").is_none());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn multi_matrix_entry_flattens_row_major() {
        // 2 matrices, ne0 = 2: sums [a,b | c,d]; counts [2, 4] -> [a/2, b/2, c/4, d/4].
        let sums = vec![2.0f32, 4.0, 8.0, 16.0];
        let counts = vec![2.0f32, 4.0];
        let path = write_synth("blk.3.ffn_gate.weight", &sums, &counts, vec![2, 2]);
        let m = read_imatrix(path.to_str().unwrap()).expect("read");
        assert_eq!(
            m.vector("blk.3.ffn_gate.weight").unwrap(),
            &[1.0f32, 2.0, 2.0, 4.0]
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn shape_mismatch_is_refused() {
        // 3 summed values declared as one matrix, but 2 counts: 3 does not divide across 2.
        let sums = vec![1.0f32, 2.0, 3.0];
        let counts = vec![1.0f32, 1.0];
        let path = write_synth("blk.4.attn_v.weight", &sums, &counts, vec![3]);
        assert!(matches!(
            read_imatrix(path.to_str().unwrap()),
            Err(ImatrixError::Shape { .. })
        ));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn strip_suffix_only_matches_tail() {
        assert_eq!(
            strip_suffix(b"blk.0.attn_q.weight.in_sum2", b".in_sum2"),
            Some("blk.0.attn_q.weight")
        );
        assert_eq!(strip_suffix(b"blk.0.attn_q.weight", b".in_sum2"), None);
        assert_eq!(strip_suffix(b".in_sum2", b".in_sum2"), Some(""));
    }

    #[test]
    fn f32_payload_decodes_le() {
        let mut b = Vec::new();
        b.extend_from_slice(&1.0f32.to_le_bytes());
        b.extend_from_slice(&(-2.5f32).to_le_bytes());
        assert_eq!(f32_payload(&b), &[1.0f32, -2.5]);
    }
}
