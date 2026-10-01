//! Actual CPU observations for the native signed Q8 GPU baseline.
//!
//! SPIQ8V01 is little-endian. Refusal values are placeholders, never golden outputs.
//! `QuantAct::from_parts` is unchecked: this producer validates the joint domain first.

use super::{exact_dot, f16_split, QuantAct, ACT_QK, Q8_0_BLOCK_BYTES};
use crate::engine::chart::quant_act;

struct Case {
    id: u32,
    rows: usize,
    tokens: usize,
    blocks: usize,
    raw: Vec<u8>,
    scales: Vec<i64>,
    codes: Vec<i8>,
    refusal: bool,
}

impl Case {
    /// O(M*K + B*K) words and O(32*(M+B)*K) writes, from the allocated lengths,
    /// for M rows, B tokens, K blocks. Every geometry multiplication is checked.
    fn new(id: u32, rows: usize, tokens: usize, blocks: usize, refusal: bool) -> Self {
        assert!((1..=257).contains(&rows) && (1..=200).contains(&tokens));
        assert!((1..=64).contains(&blocks));
        let row_bytes = rows
            .checked_mul(blocks)
            .unwrap()
            .checked_mul(Q8_0_BLOCK_BYTES)
            .unwrap();
        let act_blocks = tokens.checked_mul(blocks).unwrap();
        let mut case = Self {
            id,
            rows,
            tokens,
            blocks,
            raw: vec![0; row_bytes],
            scales: vec![1; act_blocks],
            codes: vec![0; act_blocks.checked_mul(ACT_QK).unwrap()],
            refusal,
        };
        for row in 0..rows {
            for block in 0..blocks {
                case.set_half(row, block, 0x3c00);
            }
        }
        case
    }

    fn set_half(&mut self, row: usize, block: usize, half: u16) {
        let at = (row * self.blocks + block) * Q8_0_BLOCK_BYTES;
        self.raw[at..at + 2].copy_from_slice(&half.to_le_bytes());
    }

    fn weight(&mut self, row: usize, block: usize, lane: usize, code: i8) {
        self.raw[(row * self.blocks + block) * Q8_0_BLOCK_BYTES + 2 + lane] = code.cast_unsigned();
    }

    fn activation(&mut self, token: usize, block: usize, lane: usize, code: i8) {
        self.codes[(token * self.blocks + block) * ACT_QK + lane] = code;
    }
}

/// O(M*K + B*K) words written; lane loops have exactly 32 iterations per block.
/// The deterministic arithmetic remains bounded before narrowing to signed byte codes.
fn varied(id: u32, rows: usize, tokens: usize, blocks: usize) -> Case {
    let mut case = Case::new(id, rows, tokens, blocks, false);
    let halves = [0x3c00, 0xbc00, 0x3800, 0xb800, 0x0001, 0x8001, 0, 0x8000];
    for row in 0..rows {
        for block in 0..blocks {
            case.set_half(row, block, halves[(row + block) % halves.len()]);
            for lane in 0..ACT_QK {
                let code = i16::try_from((row * 29 + block * 17 + lane * 11) % 256).unwrap() - 128;
                case.weight(row, block, lane, i8::try_from(code).unwrap());
            }
        }
    }
    for token in 0..tokens {
        for block in 0..blocks {
            case.scales[token * blocks + block] = if (token + block) % 11 == 0 {
                0
            } else {
                1 << ((token + 3 * block) % 11)
            };
            for lane in 0..ACT_QK {
                let code = i16::try_from((token * 31 + block * 7 + lane * 19) % 255).unwrap() - 127;
                case.activation(token, block, lane, i8::try_from(code).unwrap());
            }
        }
    }
    case
}

// Domain classifications validate the registered refusal populations; exact_dot supplies goldens.
#[derive(Debug, PartialEq)]
enum DomainEnd {
    Finite(i128),
    NonFinite,
    ProductOverflow,
    PositiveAccumulatorOverflow,
    NegativeAccumulatorOverflow,
}

/// O(32*K) lane products and O(1) scratch; the loops visit every block and 32 lanes.
fn check_rounding_domain(row: &[u8], blocks: usize, q: &QuantAct) -> DomainEnd {
    let mut acc = 0i128;
    for block in 0..blocks {
        let at = block * Q8_0_BLOCK_BYTES;
        let Some((m, k)) = f16_split(u16::from_le_bytes([row[at], row[at + 1]])) else {
            return DomainEnd::NonFinite;
        };
        let mut dot = 0i128;
        for lane in 0..ACT_QK {
            dot += i128::from(row[at + 2 + lane].cast_signed())
                * i128::from(q.codes()[block * ACT_QK + lane]);
        }
        let Some(term) = (i128::from(m) << k)
            .checked_mul(i128::from(q.scales()[block]))
            .and_then(|value| value.checked_mul(dot))
        else {
            return DomainEnd::ProductOverflow;
        };
        let Some(next) = acc.checked_add(term) else {
            return if term > 0 {
                DomainEnd::PositiveAccumulatorOverflow
            } else {
                DomainEnd::NegativeAccumulatorOverflow
            };
        };
        acc = next;
    }
    assert!(
        acc <= i128::MAX - (1i128 << 31),
        "rounding-add panic domain is excluded"
    );
    DomainEnd::Finite(acc)
}

/// O(32*(M+B)*K + 32*M*B*K) operations and O(M*B + (M+B)*K) words:
/// packing visits each lane once; `exact_dot` visits each block/lane for every output.
fn encode(case: &Case, output: &mut Vec<u8>) -> (usize, usize) {
    let Case {
        id,
        rows,
        tokens,
        blocks,
        raw,
        scales,
        codes,
        refusal,
    } = case;
    assert_eq!(raw.len(), rows * blocks * Q8_0_BLOCK_BYTES);
    assert_eq!(scales.len(), tokens * blocks);
    assert_eq!(codes.len(), tokens * blocks * ACT_QK);
    assert!(codes.iter().all(|code| (-127..=127).contains(code)));
    assert!(scales.iter().all(|scale| *scale == 0
        || (*scale > 0 && scale.unsigned_abs().is_power_of_two() && scale.trailing_zeros() <= 58)));
    for value in [*rows, *tokens, *blocks, usize::try_from(*id).unwrap()] {
        output.extend_from_slice(&u32::try_from(value).unwrap().to_le_bytes());
    }
    // W[(block*8+quad)*M+row], four signed lanes per little-endian word.
    for block in 0..*blocks {
        for quad in 0..8 {
            for row in 0..*rows {
                let at = (row * blocks + block) * Q8_0_BLOCK_BYTES + 2 + 4 * quad;
                output.extend_from_slice(&raw[at..at + 4]);
            }
        }
    }
    for block in 0..*blocks {
        for row in 0..*rows {
            let at = (row * blocks + block) * Q8_0_BLOCK_BYTES;
            let half = u16::from_le_bytes([raw[at], raw[at + 1]]);
            let (mantissa, exponent) = f16_split(half).unwrap_or((0, u32::MAX));
            output.extend_from_slice(&mantissa.to_le_bytes());
            output.extend_from_slice(&exponent.to_le_bytes());
        }
    }
    for group in codes.chunks_exact(4) {
        output.extend(group.iter().map(|code| code.cast_unsigned()));
    }
    for scale in scales {
        let exponent = if *scale == 0 {
            u32::MAX
        } else {
            scale.trailing_zeros()
        };
        output.extend_from_slice(&exponent.to_le_bytes());
    }
    let mut golden = Vec::with_capacity(rows * tokens);
    let mut refused = Vec::with_capacity(rows * tokens);
    for token in 0..*tokens {
        let q = QuantAct::from_parts(
            scales[token * blocks..(token + 1) * blocks].to_vec(),
            codes[token * blocks * ACT_QK..(token + 1) * blocks * ACT_QK].to_vec(),
        );
        assert_eq!(q.scales().len(), *blocks);
        assert_eq!(q.codes().len(), blocks * ACT_QK);
        for row in 0..*rows {
            let row_slice =
                &raw[row * blocks * Q8_0_BLOCK_BYTES..(row + 1) * blocks * Q8_0_BLOCK_BYTES];
            let domain = check_rounding_domain(row_slice, *blocks, &q);
            if *id == 4 {
                assert_eq!(domain, DomainEnd::NonFinite);
            }
            if *id == 9 || *id == 10 {
                assert!(matches!(
                    domain,
                    DomainEnd::PositiveAccumulatorOverflow | DomainEnd::NegativeAccumulatorOverflow
                ));
            }
            if *id == 8 {
                let DomainEnd::Finite(acc) = domain else {
                    panic!("narrowing witness must reach rounding")
                };
                assert!(((acc + (1i128 << 31)) >> 32) > i128::from(i64::MAX));
            }
            let observed = exact_dot(row_slice, *blocks, &q);
            assert_eq!(
                observed.is_none(),
                *refusal,
                "case {id}, token {token}, row {row}"
            );
            golden.push(observed.unwrap_or(0));
            refused.push(u32::from(observed.is_none()));
        }
    }
    if *id == 6 {
        assert_eq!(
            &golden[..4],
            &[1, 0, 2, -1],
            "signed half-way rounding witnesses"
        );
    }
    for value in &golden {
        output.extend_from_slice(&value.to_le_bytes());
    }
    for flag in &refused {
        output.extend_from_slice(&flag.to_le_bytes());
    }
    let failures = refused.iter().filter(|flag| **flag == 1).count();
    (golden.len() - failures, failures)
}

/// O(32*(M+B)*K) initialized input bytes/words per case; registered shapes are bounded
/// by M<=257, B<=200, K<=64. The two large shapes exercise prefill and row/token tails.
fn populations() -> Vec<Case> {
    let mut zero = varied(0, 3, 5, 2);
    zero.scales.fill(0);
    let signed = varied(1, 5, 3, 3);
    let mut subnormal = varied(2, 3, 5, 2);
    for row in 0..3 {
        for block in 0..2 {
            subnormal.set_half(
                row,
                block,
                if (row + block) % 2 == 0 {
                    0x03ff
                } else {
                    0x83ff
                },
            );
        }
    }
    subnormal.scales.fill(1 << 18);
    let mut signed_zero = varied(3, 3, 5, 2);
    for row in 0..3 {
        for block in 0..2 {
            signed_zero.set_half(row, block, if row % 2 == 0 { 0 } else { 0x8000 });
        }
    }
    let mut invalid = Case::new(4, 3, 5, 1, true);
    for (row, half) in [0x7c00, 0xfc00, 0x7e01].into_iter().enumerate() {
        invalid.set_half(row, 0, half);
    }
    invalid.scales.fill(0);
    let mut extremes = Case::new(5, 3, 5, 1, false);
    for row in 0..3 {
        for lane in 0..ACT_QK {
            extremes.weight(row, 0, lane, if row % 2 == 0 { -128 } else { 127 });
        }
    }
    extremes.codes.fill(-127);
    let mut ties = Case::new(6, 4, 5, 1, false);
    for (row, code) in [1, -1, 3, -3].into_iter().enumerate() {
        ties.set_half(row, 0, 1);
        ties.weight(row, 0, 0, code);
    }
    for token in 0..5 {
        ties.activation(token, 0, 0, 1);
    }
    ties.scales.fill(1 << 23);
    let mut high = Case::new(7, 3, 5, 1, false);
    for row in 0..3 {
        high.set_half(row, 0, if row % 2 == 0 { 1 } else { 0x8001 });
        high.weight(row, 0, 0, 1);
    }
    for token in 0..5 {
        high.activation(token, 0, 0, 1);
    }
    high.scales.fill(1 << 58);
    let mut narrow = Case::new(8, 3, 5, 1, true);
    for row in 0..3 {
        narrow.set_half(row, 0, 0x7bff);
        narrow.weight(row, 0, 0, 1);
    }
    for token in 0..5 {
        narrow.activation(token, 0, 0, 1);
    }
    narrow.scales.fill(1 << 58);
    let mut overflow_pos = Case::new(9, 1, 1, 8, true);
    let mut overflow_neg = Case::new(10, 1, 1, 8, true);
    let mut cancellation = Case::new(11, 1, 1, 10, false);
    for case in [&mut overflow_pos, &mut overflow_neg, &mut cancellation] {
        case.scales.fill(1 << 58);
        case.codes.fill(-127);
        for block in 0..case.blocks {
            case.set_half(
                0,
                block,
                if case.id == 11 && block % 2 == 1 {
                    0xfbff
                } else {
                    0x7bff
                },
            );
            for lane in 0..ACT_QK {
                case.weight(0, block, lane, if case.id == 10 { 127 } else { -128 });
            }
        }
    }
    vec![
        zero,
        signed,
        subnormal,
        signed_zero,
        invalid,
        extremes,
        ties,
        high,
        narrow,
        overflow_pos,
        overflow_neg,
        cancellation,
        varied(12, 257, 31, 64),
        varied(13, 33, 200, 4),
    ]
}

#[test]
#[ignore = "produces explicit native GPU parity fixtures"]
/// O(sum(M*B*K*32)) scalar operations and O(serialized bytes) space across 14 cases;
/// encode visits every output's 32-lane blocks, and writing visits each serialized byte once.
fn native_q8_prefill_fixture() {
    let path = std::env::var_os("APEIRON_NATIVE_Q8_FIXTURE")
        .expect("APEIRON_NATIVE_Q8_FIXTURE must name the output file");
    let cases = populations();
    assert_eq!(cases.len(), 14);
    assert_eq!(cases.iter().filter(|case| case.refusal).count(), 4);
    let mut output = b"SPIQ8V01".to_vec();
    output.extend_from_slice(&u32::try_from(cases.len()).unwrap().to_le_bytes());
    let mut success = 0usize;
    let mut refusal = 0usize;
    for case in &cases {
        let (yes, no) = encode(case, &mut output);
        success += yes;
        refusal += no;
        eprintln!(
            "NATIVE_Q8_CASE id={} rows={} tokens={} blocks={} some={} refusal={}",
            case.id, case.rows, case.tokens, case.blocks, yes, no
        );
    }
    assert_eq!((success, refusal), (14_678, 32));
    std::fs::write(&path, &output).expect("write native GPU fixture");
    eprintln!(
        "NATIVE_Q8_FIXTURE_OK cases={} some={success} refusal={refusal} bytes={} path={}",
        cases.len(),
        output.len(),
        std::path::Path::new(&path).display()
    );
}

struct RealQuantizerCase {
    id: u32,
    rows: usize,
    blocks: usize,
    raw: Vec<u8>,
    quantized: Vec<QuantAct>,
}

impl RealQuantizerCase {
    /// `O(B·32·T + M·B)` time and `O(B·32·T + M·B·34)` bytes, where T is activation count;
    /// it quantizes each input once and initializes every row/block weight record.
    fn new(id: u32, rows: usize, activations: Vec<Vec<i64>>) -> Self {
        assert!(!activations.is_empty());
        let n_in = activations[0].len();
        assert!(n_in > 0 && n_in.is_multiple_of(ACT_QK));
        assert!(activations.iter().all(|x| x.len() == n_in));
        let blocks = n_in / ACT_QK;
        assert!((1..=64).contains(&blocks));
        let quantized = activations
            .into_iter()
            .map(|x| {
                let q = quant_act(&x).expect("registered real-quantizer fixture input quantizes");
                assert_eq!(q.len(), x.len());
                assert!(q.dequantizes());
                let decoded = q.dequant().expect("constructor result dequantizes");
                let reconstructed = QuantAct::from_parts(q.scales().to_vec(), q.codes().to_vec());
                assert_eq!(reconstructed.scales(), q.scales());
                assert_eq!(reconstructed.codes(), q.codes());
                assert_eq!(reconstructed.dequant(), Some(decoded));
                q
            })
            .collect();
        let row_bytes = rows
            .checked_mul(blocks)
            .unwrap()
            .checked_mul(Q8_0_BLOCK_BYTES)
            .unwrap();
        let mut case = Self {
            id,
            rows,
            blocks,
            raw: vec![0; row_bytes],
            quantized,
        };
        for row in 0..rows {
            for block in 0..blocks {
                case.set_half(row, block, 0x3c00);
            }
        }
        case
    }

    fn set_half(&mut self, row: usize, block: usize, half: u16) {
        let at = (row * self.blocks + block) * Q8_0_BLOCK_BYTES;
        self.raw[at..at + 2].copy_from_slice(&half.to_le_bytes());
    }

    fn weight(&mut self, row: usize, block: usize, lane: usize, code: i8) {
        self.raw[(row * self.blocks + block) * Q8_0_BLOCK_BYTES + 2 + lane] = code.cast_unsigned();
    }
}

/// `O(B·32)` writes and `O(B·32)` bytes for B activation blocks.
fn constant_activation(blocks: usize, value: i64) -> Vec<i64> {
    vec![value; blocks * ACT_QK]
}

/// `O(32)` writes and `O(32)` bytes for one activation block.
fn unit_at(value: i64) -> Vec<i64> {
    let mut x = vec![0; ACT_QK];
    x[0] = value;
    x
}

/// `O(B·32)` arithmetic operations and `O(B·32)` bytes for B activation blocks.
fn varied_activation(blocks: usize, seed: usize) -> Vec<i64> {
    (0..blocks * ACT_QK)
        .map(|i| i64::try_from((i * 29 + seed * 17) % 63).unwrap() - 31)
        .collect()
}

/// `O(M·B·32)` writes and `O(1)` extra space: each weight lane and block scale is assigned once.
fn populate_weighted(case: &mut RealQuantizerCase, seed: usize) {
    for row in 0..case.rows {
        for block in 0..case.blocks {
            let half = match (row + block + seed) % 4 {
                0 => 0x3c00, // finite normal
                1 => 0x0001, // finite subnormal
                2 => 0xbc00, // negative finite normal
                _ => 0x8000, // signed zero
            };
            case.set_half(row, block, half);
            for lane in 0..ACT_QK {
                let code =
                    i16::try_from((row * 17 + block * 29 + lane * 13 + seed) % 255).unwrap() - 127;
                case.weight(row, block, lane, i8::try_from(code).unwrap());
            }
        }
    }
}

/// `O(M*K + B*K + M*B*K*32)` operations and `O(serialized bytes)` storage for M weight
/// rows, B quantized activation vectors and K blocks; each matrix row, token and lane is visited.
fn encode_real_inputs(case: &RealQuantizerCase, output: &mut Vec<u8>) {
    let tokens = case.quantized.len();
    output.extend_from_slice(&u32::try_from(case.rows).unwrap().to_le_bytes());
    output.extend_from_slice(&u32::try_from(tokens).unwrap().to_le_bytes());
    output.extend_from_slice(&u32::try_from(case.blocks).unwrap().to_le_bytes());
    output.extend_from_slice(&case.id.to_le_bytes());

    for block in 0..case.blocks {
        for quad in 0..8 {
            for row in 0..case.rows {
                let at = (row * case.blocks + block) * Q8_0_BLOCK_BYTES + 2 + 4 * quad;
                output.extend_from_slice(&case.raw[at..at + 4]);
            }
        }
    }
    for block in 0..case.blocks {
        for row in 0..case.rows {
            let at = (row * case.blocks + block) * Q8_0_BLOCK_BYTES;
            let half = u16::from_le_bytes([case.raw[at], case.raw[at + 1]]);
            let (mantissa, exponent) = f16_split(half).unwrap_or((0, u32::MAX));
            output.extend_from_slice(&mantissa.to_le_bytes());
            output.extend_from_slice(&exponent.to_le_bytes());
        }
    }
    for q in &case.quantized {
        assert_eq!(q.codes().len(), case.blocks * ACT_QK);
        for group in q.codes().chunks_exact(4) {
            output.extend(group.iter().map(|code| code.cast_unsigned()));
        }
    }
    for q in &case.quantized {
        assert_eq!(q.scales().len(), case.blocks);
        for &scale in q.scales() {
            assert!(scale == 0 || (scale > 0 && scale.unsigned_abs().is_power_of_two()));
            let exponent = if scale == 0 {
                u32::MAX
            } else {
                scale.trailing_zeros()
            };
            assert!(scale == 0 || exponent <= 58);
            output.extend_from_slice(&exponent.to_le_bytes());
        }
    }
}

/// `O(M·B·K·32)` operations and `O(M·B)` scratch; actual `exact_dot` observes each row/token pair.
fn encode_real_quantizer(case: &RealQuantizerCase, output: &mut Vec<u8>) -> (usize, usize) {
    encode_real_inputs(case, output);
    let tokens = case.quantized.len();
    let mut expected = Vec::with_capacity(case.rows * tokens);
    let mut refused = Vec::with_capacity(case.rows * tokens);
    let mut arithmetic_refusals = 0;
    for q in &case.quantized {
        for row in 0..case.rows {
            let row_slice = &case.raw
                [row * case.blocks * Q8_0_BLOCK_BYTES..(row + 1) * case.blocks * Q8_0_BLOCK_BYTES];
            let domain = check_rounding_domain(row_slice, case.blocks, q);
            if case.id == 105 && row > 0 {
                assert_eq!(
                    domain,
                    DomainEnd::NonFinite,
                    "non-finite DM must refuse even at zero activation"
                );
            }
            if case.id == 108 {
                let DomainEnd::Finite(acc) = domain else {
                    panic!("narrowing case must reach round/narrow")
                };
                assert!(((acc + (1i128 << 31)) >> 32) > i128::from(i64::MAX));
            }
            if case.id == 109 {
                assert_eq!(
                    domain,
                    DomainEnd::PositiveAccumulatorOverflow,
                    "positive i128 overflow witness must be reachable"
                );
            }
            if case.id == 110 {
                assert_eq!(
                    domain,
                    DomainEnd::NegativeAccumulatorOverflow,
                    "negative i128 overflow witness must be reachable"
                );
            }
            if case.id == 111 {
                assert_eq!(
                    domain,
                    DomainEnd::Finite(0),
                    "alternating large terms cancel without overflow"
                );
            }
            if !matches!(domain, DomainEnd::Finite(_)) {
                arithmetic_refusals += 1;
            }
            let value = exact_dot(row_slice, case.blocks, q);
            expected.push(value.unwrap_or(0));
            refused.push(u32::from(value.is_none()));
        }
    }
    if case.id == 106 || case.id == 107 {
        let tail = expected.len() - 1;
        assert_eq!(
            refused[tail], 0,
            "registered row/token tail must be a successful observation"
        );
        assert_ne!(
            expected[tail], 0,
            "registered row/token tail must be nonzero"
        );
    }
    for value in &expected {
        output.extend_from_slice(&value.to_le_bytes());
    }
    for flag in &refused {
        output.extend_from_slice(&flag.to_le_bytes());
    }
    let refusals = refused.iter().filter(|flag| **flag == 1).count();
    eprintln!(
        "REAL_Q8_CASE id={} rows={} tokens={} blocks={} some={} refusal={} arithmetic_refusals={}",
        case.id,
        case.rows,
        tokens,
        case.blocks,
        expected.len() - refusals,
        refusals,
        arithmetic_refusals
    );
    (expected.len() - refusals, refusals)
}

/// `O(total input lanes + M·B·K·32)` exact CPU observations; all activation columns enter through
/// `quant_act`, and `exact_dot` supplies every fixture golden.
fn real_quantizer_boundary_cases() -> Vec<RealQuantizerCase> {
    let zeros = constant_activation(1, 0);
    let mut zero = RealQuantizerCase::new(100, 2, vec![zeros.clone()]);
    zero.weight(0, 0, 0, 127);

    let mut boundary = RealQuantizerCase::new(
        101,
        1,
        [127, -127, 128, -128].into_iter().map(unit_at).collect(),
    );
    assert_eq!(
        boundary
            .quantized
            .iter()
            .map(|q| q.scales()[0])
            .collect::<Vec<_>>(),
        vec![1, 1, 2, 2]
    );
    boundary.weight(0, 0, 0, 1);
    assert_eq!(
        boundary
            .quantized
            .iter()
            .map(|q| q.codes()[0])
            .collect::<Vec<_>>(),
        vec![127, -127, 64, -64],
        "actual quant_act ±127/±128 scale transition witnesses"
    );

    let tie_values = {
        let mut x = vec![0; ACT_QK];
        x[..3].copy_from_slice(&[3, -3, 130]);
        x
    };
    let mut ties = RealQuantizerCase::new(102, 3, vec![tie_values]);
    assert_eq!(ties.quantized[0].scales(), &[2]);
    assert_eq!(
        &ties.quantized[0].codes()[..3],
        &[2, -2, 65],
        "actual quant_act signed tie must round away from zero"
    );
    for row in 0..3 {
        ties.weight(row, 0, row, 1);
    }

    let mut mixed = vec![0; 2 * ACT_QK];
    for lane in 0..ACT_QK {
        mixed[lane] = if lane % 2 == 0 { 127 } else { -127 };
        mixed[ACT_QK + lane] = if lane % 2 == 0 {
            63i64 << 57
        } else {
            -(63i64 << 57)
        };
    }
    let mut mixed = RealQuantizerCase::new(103, 1, vec![mixed]);
    assert_eq!(mixed.quantized[0].scales(), &[1, 1i64 << 56]);
    assert_eq!(mixed.quantized[0].codes()[0], 127);
    assert_eq!(mixed.quantized[0].codes()[ACT_QK], 126);
    mixed.weight(0, 0, 0, 1);
    mixed.weight(0, 1, 0, 1);

    let mut high =
        RealQuantizerCase::new(104, 1, vec![unit_at(63i64 << 57), unit_at(-(63i64 << 57))]);
    assert!(high.quantized.iter().all(|q| q.scales() == [1i64 << 56]));
    assert_eq!(high.quantized[0].codes()[0], 126);
    assert_eq!(high.quantized[1].codes()[0], -126);
    high.set_half(0, 0, 0x0001);
    high.weight(0, 0, 0, 1);

    let min = quant_act(&[i64::MIN; ACT_QK]).expect("i64::MIN lies on the actual 2^57 grid");
    assert_eq!(min.scales(), &[1i64 << 57]);
    assert!(min.dequant().unwrap().iter().all(|&v| v == i64::MIN));
    assert!(
        quant_act(&[i64::MAX; ACT_QK]).is_none(),
        "i64::MAX rounds to code 64 and its dequantized product is out of range"
    );
    assert!(
        quant_act(&[i64::MIN, i64::MAX]).is_none(),
        "mixed i64::MIN/MAX retains the i64::MAX constructor refusal"
    );

    let mut nonfinite = RealQuantizerCase::new(105, 3, vec![zeros]);
    nonfinite.set_half(1, 0, 0x7c00);
    nonfinite.set_half(2, 0, 0x7e01);
    nonfinite.weight(0, 0, 0, 1);

    vec![zero, boundary, ties, mixed, high, nonfinite]
}

/// `O(total input lanes + M·B·K·32)` operations and `O(total input/weight bytes)` space;
/// constructor inputs cover nonzero tails, signed overflow and cancellation.
fn real_quantizer_large_cases() -> Vec<RealQuantizerCase> {
    let mut large_activations: Vec<_> = (0..31).map(|token| varied_activation(64, token)).collect();
    for activation in &mut large_activations {
        activation[0] = 31;
    }
    let mut large = RealQuantizerCase::new(106, 257, large_activations);
    populate_weighted(&mut large, 0);
    large.set_half(256, 0, 0x3c00);
    for block in 0..large.blocks {
        for lane in 0..ACT_QK {
            large.weight(256, block, lane, 0);
        }
    }
    large.weight(256, 0, 0, 1);
    let mut large_tail_activations: Vec<_> = (0..200)
        .map(|token| varied_activation(4, token + 31))
        .collect();
    for activation in &mut large_tail_activations {
        activation[0] = 31;
    }
    let mut large_tail = RealQuantizerCase::new(107, 33, large_tail_activations);
    populate_weighted(&mut large_tail, 1);
    large_tail.set_half(32, 0, 0x3c00);
    for block in 0..large_tail.blocks {
        for lane in 0..ACT_QK {
            large_tail.weight(32, block, lane, 0);
        }
    }
    large_tail.weight(32, 0, 0, 1);

    let mut narrow = RealQuantizerCase::new(108, 1, vec![unit_at(63i64 << 57)]);
    narrow.set_half(0, 0, 0x7bff);
    narrow.weight(0, 0, 0, 1);

    let overflow_values = constant_activation(32, 63i64 << 57);
    let mut overflow_positive = RealQuantizerCase::new(109, 1, vec![overflow_values.clone()]);
    let mut overflow_negative = RealQuantizerCase::new(110, 1, vec![overflow_values.clone()]);
    let mut cancellation =
        RealQuantizerCase::new(111, 1, vec![constant_activation(64, 63i64 << 57)]);
    for block in 0..32 {
        overflow_positive.set_half(0, block, 0x7bff);
        overflow_negative.set_half(0, block, 0x7bff);
        for lane in 0..ACT_QK {
            overflow_positive.weight(0, block, lane, 127);
            overflow_negative.weight(0, block, lane, -128);
        }
    }
    for block in 0..64 {
        cancellation.set_half(0, block, if block % 2 == 0 { 0x7bff } else { 0xfbff });
        for lane in 0..ACT_QK {
            cancellation.weight(0, block, lane, -128);
        }
    }
    let q = &overflow_positive.quantized[0];
    assert!(q.scales().iter().all(|&s| s == 1i64 << 56));
    assert!(q.codes().iter().all(|&c| c == 126));
    assert_eq!(
        exact_dot(&cancellation.raw, 64, &cancellation.quantized[0]),
        Some(0)
    );

    vec![
        large,
        large_tail,
        narrow,
        overflow_positive,
        overflow_negative,
        cancellation,
    ]
}

/// `O(total input lanes + M·B·K·32)` operations and `O(total fixture bytes)` space;
/// concatenate constructor boundary cases and large/arithmetic cases without changing their order.
fn real_quantizer_populations() -> Vec<RealQuantizerCase> {
    let mut cases = real_quantizer_boundary_cases();
    cases.extend(real_quantizer_large_cases());
    cases
}

#[test]
#[ignore = "produces actual-quantizer native GPU parity fixtures"]
/// `O(total activations + Σ(M·B·K·32))` operations and `O(serialized bytes)` storage; the
/// registered matrix row/token tails are 257x31x64 and 33x200x4.
fn native_real_quantizer_q8_fixture() {
    let path = std::env::var_os("APEIRON_NATIVE_Q8_FIXTURE")
        .expect("APEIRON_NATIVE_Q8_FIXTURE must name the output file");
    let cases = real_quantizer_populations();
    assert_eq!(cases.len(), 12);
    let mut output = b"SPIQ8V01".to_vec();
    output.extend_from_slice(&u32::try_from(cases.len()).unwrap().to_le_bytes());
    let mut success = 0usize;
    let mut refusal = 0usize;
    for case in &cases {
        let (yes, no) = encode_real_quantizer(case, &mut output);
        success += yes;
        refusal += no;
    }
    eprintln!(
        "REAL_Q8_FIXTURE_OK cases={} some={success} refusal={refusal} bytes={} path={}",
        cases.len(),
        output.len(),
        std::path::Path::new(&path).display()
    );
    std::fs::write(&path, &output).expect("write real-quantizer GPU fixture");
}
