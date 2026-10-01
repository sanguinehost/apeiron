//! Actual CPU observations for the native signed Q8 GPU baseline.
//!
//! SPIQ8V01 is little-endian. Refusal values are placeholders, never golden outputs.
//! `QuantAct::from_parts` is unchecked: this producer validates the joint domain first.

use super::{exact_dot, f16_split, QuantAct, ACT_QK, Q8_0_BLOCK_BYTES};

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
    AccumulatorOverflow,
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
            return DomainEnd::AccumulatorOverflow;
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
                assert_eq!(domain, DomainEnd::AccumulatorOverflow);
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
