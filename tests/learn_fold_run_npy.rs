//! `tests/learn_fold_run_npy.rs` — the NumPy `.npy` `int32` reader and the `CHUNK = 512`
//! chunking and calibration / in-sample / held-out split of the fold-transfer runs
//! (`src/learn/fold/run/npy.rs`).
//!
//! Grade `tested`: every assertion here executes the crate's build path. What the Python selects is
//! read out of `fold_transfer.py:2763-2778` (`CHUNK = 512`; `nch = len(ids) // CHUNK`;
//! `ids_cal = ids_all[:n_cal]`; `ids_ev = ids_all[n_cal:n_cal+n_eval]`; `ids_in = ids_all[:n_in]`),
//! `pv_finish.py:1089-1095, 1401-1404` and `kl_distill.py:95-103`.
//!
//! The two generators in that path are torch's, not numpy's: the calibration-row permutation for
//! `seed != 0` is `torch.randperm(n_cal, generator=torch.Generator().manual_seed(sd))`
//! (`fold_transfer.py:2797`) and the training order is the same `torch.randperm` over the training
//! chunks (`pv_finish.py:1404`, `kl_distill.py:103`). Neither is ported; the chunk index list
//! enters as a slice, and these tests cover the deterministic `seed = 0` identity the module
//! computes.

#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "test harness: byte-level fixture building, integer casts in range and unwrapped \
              `Result`s are the assertions"
)]

use apeiron::learn::fold::run::npy::{chunks, read_npy_i4, split, NpyError};

/// The NumPy magic prefix, `numpy.lib.format.MAGIC_PREFIX`.
const MAGIC: &[u8] = b"\x93NUMPY";

/// The path of the C4 shard 00001 token file (`pv_finish.py:49`, `kl_distill.py:26`).
const C4_NPY: &str = "/home/socol/oc-tmp/lanes/w18/kd/c4data/train-c4.qwen.npy";

/// The Python's `CHUNK` (`fold_transfer.py:41`).
const CHUNK: usize = 512;

/// A header dict text space-padded to numpy's 64-byte alignment and newline-terminated, exactly as
/// `numpy.lib.format._write_array_header` lays it out.
fn padded(dict: &str) -> Vec<u8> {
    let mut h = dict.as_bytes().to_vec();
    while !(10 + h.len() + 1).is_multiple_of(64) {
        h.push(b' ');
    }
    h.push(b'\n');
    h
}

/// The header dict text of a `.npy` file with the three keys the reader requires.
fn header_text(descr: &str, fortran: &str, shape: &str) -> Vec<u8> {
    padded(&format!(
        "{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': {shape}, }}"
    ))
}

/// A whole `.npy` file from a version pair, a header and a payload.
fn file_bytes(version: (u8, u8), header: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.push(version.0);
    out.push(version.1);
    match version.0 {
        1 => out.extend_from_slice(&u16::try_from(header.len()).unwrap().to_le_bytes()),
        2 => out.extend_from_slice(&u32::try_from(header.len()).unwrap().to_le_bytes()),
        _ => {}
    }
    out.extend_from_slice(header);
    out.extend_from_slice(payload);
    out
}

/// The little-endian `i4` payload of `words`.
fn i4(words: &[i32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A v1.0 file with the given header fields and payload.
fn npy_v1(descr: &str, fortran: &str, shape: &str, payload: &[u8]) -> Vec<u8> {
    file_bytes((1, 0), &header_text(descr, fortran, shape), payload)
}

/// Overwrite the declared header length field of a v1.0 file.
fn with_len_field(mut bytes: Vec<u8>, len: u16) -> Vec<u8> {
    bytes[8..10].copy_from_slice(&len.to_le_bytes());
    bytes
}

/// The five-word fixture: two negative, zero, and the `i32` ends.
fn five_words() -> [i32; 5] {
    [7, -1, 0, 1234, i32::MIN]
}

/// A v1.0 header carrying five `i4` words round-trips, with the payload at the byte offset the
/// declared header length names.
#[test]
fn v1_header_five_i32_round_trips() {
    let words = five_words();
    let header = header_text("<i4", "False", "(5,)");
    let bytes = file_bytes((1, 0), &header, &i4(&words));
    assert_eq!(header.len(), 118);
    assert_eq!(bytes.len(), 10 + header.len() + 4 * 5);
    assert_eq!(read_npy_i4(&bytes).unwrap(), words.to_vec());
}

/// A v2.0 header (4-byte declared length) reads the same words.
#[test]
fn v2_header_five_i32_round_trips() {
    let words = five_words();
    let header = header_text("<i4", "False", "(5,)");
    let bytes = file_bytes((2, 0), &header, &i4(&words));
    assert_eq!(bytes.len(), 12 + header.len() + 4 * 5);
    assert_eq!(read_npy_i4(&bytes).unwrap(), words.to_vec());
}

/// A big-endian word width is not this reader's dtype.
#[test]
fn big_endian_descriptor_is_unsupported() {
    let bytes = npy_v1(">i4", "False", "(5,)", &i4(&five_words()));
    assert_eq!(read_npy_i4(&bytes), Err(NpyError::Unsupported));
}

/// `int8` and 64-bit integers are not `i4`.
#[test]
fn narrow_and_wide_integer_descriptors_are_unsupported() {
    let narrow = npy_v1("|i1", "False", "(5,)", &[1, 2, 3, 4, 5]);
    assert_eq!(read_npy_i4(&narrow), Err(NpyError::Unsupported));
    let wide = npy_v1("<i8", "False", "(5,)", &[0u8; 40]);
    assert_eq!(read_npy_i4(&wide), Err(NpyError::Unsupported));
}

/// A Fortran-ordered file is refused rather than silently read as C order.
#[test]
fn fortran_order_is_unsupported() {
    let bytes = npy_v1("<i4", "True", "(5,)", &i4(&five_words()));
    assert_eq!(read_npy_i4(&bytes), Err(NpyError::Unsupported));
}

/// `(2, 5)` and `()` are not 1-D.
#[test]
fn non_1d_shapes_are_unsupported() {
    let two_d = npy_v1(
        "<i4",
        "False",
        "(2, 5)",
        &i4(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    );
    assert_eq!(read_npy_i4(&two_d), Err(NpyError::Unsupported));
    let zero_d = npy_v1("<i4", "False", "()", &[]);
    assert_eq!(read_npy_i4(&zero_d), Err(NpyError::Unsupported));
}

/// A payload shorter than `4·N` is refused, not zero-filled.
#[test]
fn truncated_payload_is_refused() {
    let full = npy_v1("<i4", "False", "(5,)", &i4(&five_words()));
    let short = full[..full.len() - 4].to_vec();
    assert_eq!(read_npy_i4(&short), Err(NpyError::ShapeMismatch));
}

/// A payload longer than `4·N` is refused: the shape and the payload must agree exactly.
#[test]
fn payload_longer_than_the_shape_is_refused() {
    let words = [1i32, 2, 3, 4, 5, 6, 7, 8];
    let bytes = npy_v1("<i4", "False", "(5,)", &i4(&words));
    assert_eq!(read_npy_i4(&bytes), Err(NpyError::ShapeMismatch));
}

/// A declared header length that runs past the buffer is refused, not clamped.
#[test]
fn header_length_beyond_the_buffer_is_refused() {
    let full = npy_v1("<i4", "False", "(5,)", &i4(&five_words()));
    let lying = with_len_field(full, 2000);
    assert_eq!(read_npy_i4(&lying), Err(NpyError::Truncated));
}

/// A declared header length that stops short of the terminating newline is refused.
#[test]
fn header_length_short_of_the_newline_is_refused() {
    let full = npy_v1("<i4", "False", "(5,)", &i4(&five_words()));
    let lying = with_len_field(full, 40);
    assert_eq!(read_npy_i4(&lying), Err(NpyError::MalformedHeader));
}

/// A wrong magic and a version other than 1.0 / 2.0 are refused.
#[test]
fn magic_and_version_are_checked() {
    let mut wrong_magic = npy_v1("<i4", "False", "(5,)", &i4(&five_words()));
    wrong_magic[0] = b'X';
    assert_eq!(read_npy_i4(&wrong_magic), Err(NpyError::NotNpy));
    let wrong_minor = file_bytes(
        (1, 1),
        &header_text("<i4", "False", "(5,)"),
        &i4(&five_words()),
    );
    assert_eq!(read_npy_i4(&wrong_minor), Err(NpyError::Unsupported));
}

/// A header missing one of the three dict keys is refused, as is a header without its terminating
/// newline.
#[test]
fn missing_key_is_refused() {
    let dict = padded("{'descr': '<i4', 'shape': (5,), }");
    let bytes = file_bytes((1, 0), &dict, &i4(&five_words()));
    assert_eq!(read_npy_i4(&bytes), Err(NpyError::MalformedHeader));
    let mut header = header_text("<i4", "False", "(5,)");
    header.truncate(header.len() - 1);
    let bytes = file_bytes((1, 0), &header, &i4(&five_words()));
    assert_eq!(read_npy_i4(&bytes), Err(NpyError::MalformedHeader));
}

/// `1030` ids at `CHUNK = 512` give two chunks of exactly `512`, dropping the trailing 6 tokens
/// (`nch = len(ids) // CHUNK`).
#[test]
fn chunks_1030_ids_at_512_give_two() {
    let ids: Vec<i32> = (0..1030).collect();
    let rows = chunks(&ids, CHUNK).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].len(), CHUNK);
    assert_eq!(rows[1].len(), CHUNK);
    assert_eq!(rows[0][0], 0);
    assert_eq!(rows[0][CHUNK - 1], 511);
    assert_eq!(rows[1][0], 512);
    assert_eq!(rows[1][CHUNK - 1], 1023);
    // Two chunks of 512 of the 1030 tokens: 6 dropped, none duplicated.
    assert_eq!(rows.iter().map(Vec::len).sum::<usize>(), 2 * CHUNK);
}

/// An empty stream and a stream shorter than one chunk both give no rows.
#[test]
fn chunks_below_one_chunk_give_nothing() {
    assert!(chunks(&[], CHUNK).unwrap().is_empty());
    assert_eq!(chunks(&[1, 2, 3], CHUNK).unwrap().len(), 0);
    assert_eq!(chunks(&[1, 2, 3], 3).unwrap().len(), 1);
}

/// A negative token id has no `u32` encoding and is refused.
#[test]
fn chunks_refuse_a_negative_id() {
    assert_eq!(chunks(&[0, -1, 2], 1), Err(NpyError::NegativeToken));
    assert_eq!(chunks(&[i32::MIN], 1), Err(NpyError::NegativeToken));
}

/// `chunk = 0` is refused rather than dividing by zero.
#[test]
fn chunks_refuse_a_zero_chunk() {
    assert_eq!(chunks(&[1, 2, 3], 0), Err(NpyError::ZeroChunk));
}

/// The split is the Python's three slices: `cal = 0..n_cal`, `in_sample = 0..n_in`,
/// `heldout = n_cal..n_cal+n_eval`.
#[test]
fn split_is_the_python_prefix_and_offset() {
    let (n_chunks, n_cal, n_in, n_eval) = (140, 100, 20, 40);
    let s = split(n_chunks, n_cal, n_in, n_eval).unwrap();
    assert_eq!(s.cal.len(), n_cal);
    assert_eq!(s.cal[0], 0);
    assert_eq!(s.cal[n_cal - 1], n_cal - 1);
    assert_eq!(s.in_sample.len(), n_in);
    assert_eq!(s.in_sample, s.cal[..n_in].to_vec());
    assert_eq!(s.heldout.len(), n_eval);
    assert_eq!(s.heldout[0], n_cal);
    assert_eq!(s.heldout[n_eval - 1], n_cal + n_eval - 1);

    // Disjoint: calibration against held-out, in-sample against held-out.
    assert!(s.cal.iter().all(|c| !s.heldout.contains(c)));
    assert!(s.in_sample.iter().all(|c| !s.heldout.contains(c)));
    // In range.
    assert!(s.heldout.iter().all(|&c| c < n_chunks));
    // No duplicates.
    let mut all = s.cal.clone();
    all.extend_from_slice(&s.heldout);
    let mut sorted = all.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), all.len());
}

/// The exact-boundary split (`n_cal + n_eval == n_chunks`) is accepted; a range past the stream and
/// an in-sample prefix wider than the calibration set are refused.
#[test]
fn split_refuses_indices_outside_the_stream() {
    let ok = split(140, 100, 20, 40).unwrap();
    assert_eq!(ok.heldout[39], 139);
    assert_eq!(
        split(100, 80, 20, 40),
        Err(NpyError::SplitOutOfRange),
        "n_cal + n_eval > n_chunks"
    );
    assert_eq!(
        split(120, 10, 20, 40),
        Err(NpyError::SplitOutOfRange),
        "n_in > n_cal would name chunks outside the calibration set"
    );
    assert_eq!(
        split(usize::MAX, usize::MAX, 1, 1),
        Err(NpyError::SplitOutOfRange)
    );
}

/// The C4 shard 00001 token file (`train-c4.qwen.npy`, sha256
/// `74945dcc18b06e3bff23aa4dd78454bc1206cd425d75b5ac03dc181854c7a2f5`) reads to its declared
/// `(5219098,)` `i4` payload and chunks into `10193` rows of `512` with `282` tokens dropped.
///
/// `#[ignore]` because the file is outside the repository; run it with
/// `cargo test --locked --test learn_fold_run_npy -- --ignored --nocapture`.
#[test]
#[ignore = "reads /home/socol/oc-tmp/lanes/w18/kd/c4data/train-c4.qwen.npy, outside the crate"]
fn chunking_the_c4_token_file() {
    let bytes = std::fs::read(C4_NPY).expect("the C4 token file is on disk");
    let ids = read_npy_i4(&bytes).expect("a v1.0 <i4 (5219098,) file");
    assert_eq!(ids.len(), 5_219_098);
    let nch = ids.len() / CHUNK;
    assert_eq!(nch, 10_193);
    assert_eq!(ids.len() % CHUNK, 282);
    let rows = chunks(&ids, CHUNK).expect("non-negative Qwen token ids");
    assert_eq!(rows.len(), nch);
    assert_eq!(rows[0].len(), CHUNK);
    println!("first 8 ids: {:?}", &ids[..8]);
    assert_eq!(
        rows[0][..8],
        [1655, 29948, 17131, 6982, 582, 2884, 7079, 304]
    );
    let s = split(nch, 100, 20, 40).expect("the Python's default split fits");
    assert_eq!(s.cal.len(), 100);
    assert_eq!(s.heldout.len(), 40);
    assert_eq!(s.in_sample.len(), 20);
    assert!(s.heldout.iter().all(|&c| c < nch));
}
