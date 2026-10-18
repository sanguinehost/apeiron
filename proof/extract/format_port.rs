//! Extraction roots for the format layer's import/encode side (`src/format/{packing,bpe}.rs`).
//!
//! `packing` and `quant` are declared in `lib.rs`; the wrappers below root their public bodies so
//! Charon translates the real functions. The `bpe` module is declared here because `lib.rs`'s
//! `format` mirror needs the real `src/format/bpe.rs` under a name the crate root does not already
//! carry. Every wrapper calls the real body and restates no formula.

#[path = "../../src/format/bpe.rs"]
pub mod bpe;

/// `PackingError::from(QuantError)`, the real `From` impl body.
pub fn word_packing_error_from_quant(
    e: crate::quant::QuantError,
) -> crate::packing::PackingError {
    crate::packing::PackingError::from(e)
}

/// The real `Packing::validate` body over the whole parameter set.
pub fn word_packing_validate(p: &crate::packing::Packing) -> Result<(), crate::packing::PackingError> {
    p.validate()
}

/// The real `Packing::from_gguf_kind` body; it reaches `Packing::terminal` for the two float faces.
pub fn word_packing_from_gguf_kind(kind: u32) -> Option<crate::packing::Packing> {
    crate::packing::Packing::from_gguf_kind(kind)
}

/// The real `Packing::encode_block` body; it reaches `Packing::pack_codes` and `write_field`.
pub fn word_packing_encode_block(
    p: &crate::packing::Packing,
    b: &crate::packing::CanonicalBlock,
) -> Result<Vec<u8>, crate::packing::PackingError> {
    p.encode_block(b)
}

/// The real `Packing::row_decode_fixed` body over a caller-supplied canonical row.
pub fn word_packing_row_decode_fixed(
    p: &crate::packing::Packing,
    raw: &[u8],
    n: usize,
) -> Result<Vec<i64>, crate::packing::PackingError> {
    p.row_decode_fixed(raw, n)
}

/// The real `import_row` body; it reaches `block_slice`, every `import_*` block reader and
/// `read_field`.
pub fn word_packing_import_row(
    kind: u32,
    p: &crate::packing::Packing,
    raw: &[u8],
    n: usize,
) -> Result<Vec<u8>, crate::packing::PackingError> {
    crate::packing::import_row(kind, p, raw, n)
}

/// The real `convert_row` body, the public entry of the import path.
pub fn word_packing_convert_row(
    kind: u32,
    raw: &[u8],
    n: usize,
) -> Result<Vec<i64>, crate::packing::PackingError> {
    crate::packing::convert_row(kind, raw, n)
}

/// The real `q4_k_scale_mins` body, the eight `(scale, min)` pairs of one super-block.
pub fn word_quant_q4_k_scale_mins(
    blk: &[u8],
) -> Result<[(u8, u8); 8], crate::quant::QuantError> {
    crate::quant::q4_k_scale_mins(blk)
}

/// The real `ggml_type_name` body.
pub fn word_quant_ggml_type_name(t: u32) -> String {
    crate::quant::ggml_type_name(t)
}

/// The real `bpe::hex_val` body, the hex-digit scalar of the JSON reader's unicode escapes.
pub fn word_bpe_hex_val(c: u8) -> Option<u32> {
    crate::format_port::bpe::hex_val(c)
}

/// The real `bpe::contains_nul` body over a caller-supplied piece.
pub fn word_bpe_contains_nul(s: &str) -> bool {
    crate::format_port::bpe::contains_nul(s)
}

/// The real `bpe::is_nl` body, the carriage-return/line-feed predicate of the split rules.
pub fn word_bpe_is_nl(c: char) -> bool {
    crate::format_port::bpe::is_nl(c)
}

/// The real `bpe::find_sub` body, the first-index byte search the added-token split uses.
pub fn word_bpe_find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    crate::format_port::bpe::find_sub(hay, needle)
}
