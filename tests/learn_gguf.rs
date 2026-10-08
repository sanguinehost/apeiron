//! Round-trip tests for the GGUF v3 writer in [`apeiron::learn::gguf`]: a synthetic header with one
//! metadata value of every kind and two tensors, the Qwen2.5-0.5B `Q4_K_M` fixture written back
//! byte-for-byte, and the dimension-product overflow refusal.

use apeiron::format::gguf::{parse_header, GVal, GgufError, GgufHeader, TensorDesc};
use apeiron::format::quant::{GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0};
use apeiron::learn::gguf::{write_file, write_header};

/// A 32-byte `F32` tensor followed by a 68-byte `Q8_0` tensor (two 34-byte blocks), at aligned
/// offsets.
fn sample_header() -> (GgufHeader, Vec<u8>) {
    let kv = vec![
        (b"general.alignment".to_vec(), GVal::Int(32, 4)),
        (b"k.bool".to_vec(), GVal::Bool(true)),
        (b"k.f32".to_vec(), GVal::F32Bits(1.5f32.to_bits(), 6)),
        (b"k.f64".to_vec(), GVal::F64Bits(1.5f64.to_bits(), 12)),
        (b"k.i8".to_vec(), GVal::Int(-5, 1)),
        (b"k.i16".to_vec(), GVal::Int(-300, 3)),
        (b"k.i32".to_vec(), GVal::Int(-2_000_000_000, 5)),
        (b"k.i64".to_vec(), GVal::Int(-7, 11)),
        (b"k.u8".to_vec(), GVal::Int(200, 0)),
        (b"k.u16".to_vec(), GVal::Int(60_000, 2)),
        (b"k.u32".to_vec(), GVal::Int(4_000_000_000, 4)),
        (b"k.u64".to_vec(), GVal::Int(i128::from(u64::MAX), 10)),
        (b"k.str".to_vec(), GVal::Str(b"hello".to_vec())),
        (
            b"k.arr_i32".to_vec(),
            GVal::Arr(5, vec![GVal::Int(1, 5), GVal::Int(-2, 5), GVal::Int(3, 5)]),
        ),
        (
            b"k.arr_str".to_vec(),
            GVal::Arr(8, vec![GVal::Str(b"a".to_vec()), GVal::Str(b"bc".to_vec())]),
        ),
    ];
    let tensors = vec![
        TensorDesc {
            name: b"t.f32".to_vec(),
            dims: vec![4, 2],
            ggml_type: GGML_TYPE_F32,
            offset: 0,
        },
        TensorDesc {
            name: b"t.q8".to_vec(),
            dims: vec![64],
            ggml_type: GGML_TYPE_Q8_0,
            offset: 32,
        },
    ];
    let header = GgufHeader {
        version: 3,
        kv,
        tensors,
        alignment: 32,
        header_end: 0,
        data0: 0,
    };
    // 8 f32 values (32 bytes), then 68 raw Q8_0 bytes (2 blocks of a 2-byte scale + 32 i8 codes).
    let mut data = Vec::new();
    for i in 0u8..8 {
        data.extend_from_slice(&f32::from(i).to_le_bytes());
    }
    data.extend((0u8..68).map(|b| b.wrapping_mul(7)));
    (header, data)
}

#[test]
fn synthetic_round_trip() {
    let (header, data) = sample_header();
    let out = write_file(&header, &data).expect("write synthetic file");
    let h = parse_header(&out, out.len() as u64).expect("round trip header parses");

    assert_eq!(h.version, 3);
    assert_eq!(h.alignment, 32);
    assert_eq!(h.kv, header.kv, "metadata round trip");
    assert_eq!(h.tensors.len(), header.tensors.len());
    for (a, b) in h.tensors.iter().zip(header.tensors.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.dims, b.dims);
        assert_eq!(a.ggml_type, b.ggml_type);
        assert_eq!(a.offset, b.offset);
    }

    let d0 = usize::try_from(h.data0).expect("data0 fits usize");
    assert_eq!(&out[d0..d0 + 32], &data[..32], "f32 payload");
    assert_eq!(&out[d0 + 32..d0 + 100], &data[32..], "q8_0 payload");
    assert_eq!(out.len(), d0 + data.len());
}

#[test]
fn real_fixture_round_trips_byte_identical() {
    let Some(path) = apeiron::engine::testdata::model_path("models/gguf/qwen05b-Q4_K_M.gguf")
    else {
        eprintln!("skip: fixture absent");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let header = parse_header(&bytes, bytes.len() as u64).expect("parse fixture header");
    let data = &bytes[usize::try_from(header.data0).expect("data0 fits usize")..];
    let out = write_file(&header, data).expect("write fixture back");

    assert_eq!(out.len(), bytes.len(), "round trip length");
    if let Some(i) = out.iter().zip(bytes.iter()).position(|(a, b)| a != b) {
        panic!(
            "byte mismatch at {i}: wrote {:#04x}, expected {:#04x}",
            out[i], bytes[i]
        );
    }
}

/// The padding follows `general.alignment` in the metadata, as the reader does, not the
/// `alignment` field of the header value handed to the writer.
#[test]
fn padding_follows_metadata_alignment() {
    let (mut header, data) = sample_header();
    header.kv[0].1 = GVal::Int(64, 4);
    header.alignment = 7;
    let out = write_file(&header, &data).expect("write");
    let h = parse_header(&out, out.len() as u64).expect("parse");
    assert_eq!(h.alignment, 64);
    assert_eq!(h.data0 % 64, 0);
    let d0 = usize::try_from(h.data0).expect("data0 fits usize");
    assert_eq!(&out[d0..], &data[..], "payload at the reader's data0");
}

#[test]
fn dims_product_overflow_is_refused() {
    let header = GgufHeader {
        version: 3,
        kv: vec![(b"general.alignment".to_vec(), GVal::Int(32, 4))],
        tensors: vec![TensorDesc {
            name: b"big".to_vec(),
            dims: vec![u64::MAX, 2],
            ggml_type: GGML_TYPE_Q4_K,
            offset: 0,
        }],
        alignment: 32,
        header_end: 0,
        data0: 0,
    };
    assert_eq!(write_header(&header), Err(GgufError::LengthOverflow));
    assert_eq!(write_file(&header, &[]), Err(GgufError::LengthOverflow));
}
