//! `tests/learn_fold_gptq_e8_feedback.rs` — the inter-group feedback of `gptq_e8` after its
//! restructure: bit-identity against the pre-change body, and the speed of
//! the restructured body against that body, both at the `down` geometry.
//!
//! [`inter_group_feedback_base`] is the body at `4aa0ec23` (`gptq_e8_carrier.rs:567-641`, the
//! function the restructure replaced), copied here as the reference: column loop outermost, the `gs`
//! products accumulated at stride `n` in `u.hat`. It uses only the public `Fx`, `UFac`,
//! `GptqE8Error` and `rshift_rne` plus the five helpers [`base_mag_u128`], [`base_narrow_q`],
//! [`base_blen`], [`base_shift_left_i128`] and [`base_round_at`], themselves copies of the private
//! ones so the reference does not depend on the production module.
//!
//! [`inter_group_feedback_bitequal`] is the witness: the production body and the reference agree on
//! `wm` and `e_row`, bit for bit, on three seeded `(64, 512)` inputs (the third drives the headroom
//! halving), one seeded `(896, 4864)` input, every group `g0`, and on a refusal input — both refuse
//! `Overflow` and leave `wm`/`e_row` untouched, because the overflowing term is the first term of
//! every row.
//!
//! `gfeed_speed` is `#[ignore]`d: an interleaved base/new measurement of the feedback summed over all
//! groups and of the whole fit at `(896, 4864)`, `beam = 4`, `wscale`, `act_order`.
//!
//! Grade: `tested` for the witness, `measured` for the printed wall-clock.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::items_after_statements,
    clippy::manual_range_contains,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "a test: it maps a fixed LCG to words by casts, mirrors the reference's row-major \
              indices and single-letter names, and keeps the measurement in one body. The two \
              extra allows keep the copied reference body character-identical to the base source, \
              which carries the same allows (`gptq_e8_carrier.rs:10`, `:691`)"
)]

use apeiron::learn::fold::fit::gptq_e8::{feedback_u, gptq_e8, GptqE8Error, UFac, GROUP};
use apeiron::learn::fold::fit::gptq_e8_carrier::inter_group_feedback;
use apeiron::learn::solve::q32::{ilog2_pos, rshift_rne, Fx};
use std::time::Instant;

/// The `down` damping coefficient `0.02·(4864/896)` as a Q32.32 word.
const DAMP: f64 = 0.108_6;
/// The `down` geometry.
const DOWN: (usize, usize) = (896, 4864);

// ---------------------------------------------------------------- fixtures

/// `x` as the Q32.32 word `round(x·2³²)`.
fn word(x: f64) -> i64 {
    (x * 4_294_967_296.0).round() as i64
}

/// `len` words from `seed`: a dense LCG sequence in `±2^52`.
fn lcg_words(len: usize, seed: u64) -> Vec<i64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as i64 - (1_i64 << 52)
        })
        .collect()
}

/// A dense symmetric strictly diagonally dominant `[n, n]` word Hessian: off-diagonal entries
/// `±2^20`, diagonal the row's absolute off-diagonal sum plus `2^24`. Strict dominance makes it
/// positive definite and its inverse factor dense, the shape a production Gram gives `U`.
fn spd_words(n: usize, seed: u64) -> Vec<i64> {
    let mut state = seed;
    let mut h = vec![0i64; n * n];
    let mut row_abs = vec![0i128; n];
    for i in 0..n {
        for j in (i + 1)..n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = (((state >> 11) & 0x1F_FFFF) as i64) - (1_i64 << 20);
            h[i * n + j] = v;
            h[j * n + i] = v;
            row_abs[i] += i128::from(v.abs());
            row_abs[j] += i128::from(v.abs());
        }
    }
    for i in 0..n {
        h[i * n + i] = (row_abs[i] + (1_i128 << 24)) as i64;
    }
    h
}

/// `len` working-row words: `±2^41` (the ordinary case), or `±(2^62 + ·)` when `headroom`, which sits
/// at `blen = 63` so the first update trips the row-exponent halving.
fn wm_words(len: usize, seed: u64, headroom: bool) -> Vec<i64> {
    lcg_words(len, seed)
        .iter()
        .map(|&v| {
            if headroom {
                let mag = (1u64 << 62) | (v.unsigned_abs() & ((1 << 20) - 1));
                if v < 0 {
                    -(mag as i64)
                } else {
                    mag as i64
                }
            } else {
                v >> 12
            }
        })
        .collect()
}

// ------------------------------------------------- copies of the base helpers

/// `|v|` as a `u128` (`i128::MIN` included); copy of the base private helper.
fn base_mag_u128(v: i128) -> u128 {
    if v < 0 {
        (-(v + 1)) as u128 + 1
    } else {
        v as u128
    }
}

/// `v` as `i64` or `Overflow`; copy of the base private helper.
fn base_narrow_q(v: i128) -> Result<i64, GptqE8Error> {
    if v > i128::from(i64::MAX) || v < i128::from(i64::MIN) {
        Err(GptqE8Error::Overflow)
    } else {
        Ok(v as i64)
    }
}

/// The magnitude class `⌈log2(|v| + 1)⌉`; copy of the base private helper.
fn base_blen(v: i128) -> i32 {
    if v == 0 {
        0
    } else {
        match ilog2_pos(base_mag_u128(v)) {
            Some(b) => b.cast_signed() + 1,
            None => 0,
        }
    }
}

/// `v · 2^k` for `k ≥ 0`, `Some(0)` for `v == 0`, `None` outside `i128`; copy of the base helper.
fn base_shift_left_i128(v: i128, k: i32) -> Option<i128> {
    if v == 0 {
        return Some(0);
    }
    if k < 0 || k >= 127 {
        return None;
    }
    v.checked_mul(1_i128 << k)
}

/// [`base_shift_left_i128`] with a round-to-nearest right shift for `k < 0`; copy of `round_at`.
fn base_round_at(v: i128, k: i64) -> Result<i128, GptqE8Error> {
    if k >= 0 {
        if k > i64::from(i32::MAX) {
            return Err(GptqE8Error::Overflow);
        }
        base_shift_left_i128(v, k as i32).ok_or(GptqE8Error::Overflow)
    } else {
        if -k > i64::from(u32::MAX) {
            return Err(GptqE8Error::Overflow);
        }
        rshift_rne(v, (-k) as u32).ok_or(GptqE8Error::Overflow)
    }
}

// ------------------------------------------------------------- the base body

/// The `inter_group_feedback` body at `4aa0ec23`, copied verbatim apart from its name.
#[allow(
    clippy::too_many_arguments,
    reason = "the reference keeps the production signature it is compared through"
)]
fn inter_group_feedback_base(
    wm: &mut [i64],
    e_row: &mut [i32],
    e: &[Fx],
    u: &UFac,
    n: usize,
    g0: usize,
    gs: usize,
) -> Result<(), GptqE8Error> {
    let g1 = g0 + gs;
    if g1 > n
        || n == 0
        || !wm.len().is_multiple_of(n)
        || e_row.len() != wm.len() / n
        || e.len() != e_row.len() * gs
    {
        return Err(GptqE8Error::ShapeMismatch);
    }
    let o = e_row.len();
    let tail = n - g1;
    for r in 0..o {
        // The shared exponent of this row's residual terms.
        let mut emin = i32::MAX;
        for j in 0..gs {
            let x = &e[r * gs + j];
            if x.m != 0 {
                emin = emin.min(x.e);
            }
        }
        let emin = if emin == i32::MAX { 0 } else { emin };
        for t in 0..tail {
            let c = g1 + t;
            let dc = u.d[c];
            let mut sum: i128 = 0;
            for j in 0..gs {
                let h = u.hat[(g0 + j) * n + c];
                let x = &e[r * gs + j];
                if h == 0 || x.m == 0 {
                    continue;
                }
                let pr = i128::from(x.m)
                    .checked_mul(i128::from(h))
                    .ok_or(GptqE8Error::Overflow)?;
                let term = base_shift_left_i128(pr, x.e - emin).ok_or(GptqE8Error::Overflow)?;
                sum = sum.checked_add(term).ok_or(GptqE8Error::Overflow)?;
            }
            // `Δw = −sum·2^(emin − 32 − d_c)`, one RNE to the row grid.
            let mut sh = i64::from(emin) - 32 - i64::from(dc) - i64::from(e_row[r]);
            let mut v = i128::from(wm[r * n + c])
                .checked_add(base_round_at(
                    sum.checked_neg().ok_or(GptqE8Error::Overflow)?,
                    sh,
                )?)
                .ok_or(GptqE8Error::Overflow)?;
            if base_blen(v) > 62 {
                // The headroom rule: halve the whole row (RNE) and re-round this entry.
                for q in 0..n {
                    let cur = i128::from(wm[r * n + q]);
                    let half = rshift_rne(cur, 1).ok_or(GptqE8Error::Overflow)?;
                    wm[r * n + q] = base_narrow_q(half)?;
                }
                e_row[r] = e_row[r].checked_add(1).ok_or(GptqE8Error::Overflow)?;
                sh -= 1;
                v = i128::from(wm[r * n + c])
                    .checked_add(base_round_at(
                        sum.checked_neg().ok_or(GptqE8Error::Overflow)?,
                        sh,
                    )?)
                    .ok_or(GptqE8Error::Overflow)?;
            }
            wm[r * n + c] = base_narrow_q(v)?;
        }
    }
    Ok(())
}

// -------------------------------------------------------------- the witness

/// A residual word `e` for `o` rows at `gs`: nonzero mantissas in `±(2^20 + i)` and exponents in
/// `{0, 1, 2}`, so the `2^(e_j − emin)` shift is not the identity.
fn residuals(o: usize, gs: usize, seed: u64) -> Vec<Fx> {
    lcg_words(o * gs, seed)
        .iter()
        .enumerate()
        .map(|(i, &m)| Fx {
            m: m.abs() % (1 << 20) + 1 + i as i64,
            e: (i % 3) as i32,
        })
        .collect()
}

/// The feedback after the restructure writes the bits the pre-change body wrote, on every group of
/// the three `(64, 512)` seeds (the third near the row-exponent headroom, so the halving fires) and
/// of the seeded `(896, 4864)` input; and both refuse `Overflow` on a residual exponent that leaves
/// the shift range, leaving `wm`/`e_row` as they were.
#[test]
fn inter_group_feedback_bitequal() {
    let damp = word(DAMP);
    // (o, n, seed, e_row, headroom?) — the third (64, 512) case sits at `blen = 63`.
    let cases: [(usize, usize, u64, i32, bool); 4] = [
        (64, 512, 0x0000_1111_2222_3333, -24, false),
        (64, 512, 0x4444_5555_6666_7777, -26, false),
        (64, 512, 0x8888_9999_AAAA_BBBB, -30, true),
        (DOWN.0, DOWN.1, 0xCCCC_DDDD_EEEE_FFFF, -24, false),
    ];
    for (o, n, seed, er, headroom) in cases {
        let h = spd_words(n, seed ^ 0x5EED_5EED_5EED_5EED);
        let ufac = feedback_u(&h, n, damp).expect("SPD factor");
        let wm = wm_words(o * n, seed, headroom);
        let e_row = vec![er; o];
        let e = residuals(o, GROUP, seed ^ 0x0F0F_0F0F_0F0F_0F0F);
        let mut halved = false;
        for g0 in (0..n).step_by(GROUP) {
            let gs = GROUP.min(n - g0);
            assert_eq!(gs, GROUP, "every witness case is a whole number of groups");
            let (mut w_base, mut er_base) = (wm.clone(), e_row.clone());
            let (mut w_new, mut er_new) = (wm.clone(), e_row.clone());
            inter_group_feedback_base(&mut w_base, &mut er_base, &e, &ufac, n, g0, gs)
                .expect("base feedback");
            inter_group_feedback(&mut w_new, &mut er_new, &e, &ufac, n, g0, gs)
                .expect("new feedback");
            if g0 + gs < n {
                assert_ne!(w_base, wm, "the feedback did not move the rows ({o}, {n})");
            }
            assert_eq!(w_base, w_new, "wm diverged at g0 = {g0} ({o}, {n})");
            assert_eq!(er_base, er_new, "e_row diverged at g0 = {g0} ({o}, {n})");
            halved |= er_new.iter().any(|&v| v != er);
        }
        assert_eq!(
            halved, headroom,
            "the headroom case and the halving it fires disagree ({o}, {n})"
        );
    }
}

/// The refusal: with `e[1].e = 200` the term `m·h·2^200` leaves `i128` on the first column, in every
/// row, before either body writes an entry. Both refuse `Overflow` and `wm`/`e_row` are unchanged.
#[test]
fn inter_group_feedback_refuses_the_same_overflow() {
    let (o, n) = (64usize, 512usize);
    let damp = word(DAMP);
    let h = spd_words(n, 0x1234_5678_9ABC_DEF0);
    let ufac = feedback_u(&h, n, damp).expect("SPD factor");
    let wm = wm_words(o * n, 0x0BAD_0BAD_0BAD_0BAD, false);
    let e_row = vec![-24i32; o];
    let mut e = vec![Fx { m: 0, e: 0 }; o * GROUP];
    for r in 0..o {
        e[r * GROUP] = Fx { m: 1 << 20, e: 0 };
        e[r * GROUP + 1] = Fx { m: 1 << 20, e: 200 };
    }
    let (mut wb, mut eb) = (wm.clone(), e_row.clone());
    let (mut wn, mut en) = (wm.clone(), e_row.clone());
    assert_eq!(
        inter_group_feedback_base(&mut wb, &mut eb, &e, &ufac, n, 0, GROUP),
        Err(GptqE8Error::Overflow)
    );
    assert_eq!(
        inter_group_feedback(&mut wn, &mut en, &e, &ufac, n, 0, GROUP),
        Err(GptqE8Error::Overflow)
    );
    assert_eq!(
        (wb, eb),
        (wm.clone(), e_row.clone()),
        "the base body wrote on its refusal"
    );
    assert_eq!((wn, en), (wm, e_row), "the new body wrote on its refusal");
}

// ---------------------------------------------------------------- the bench

/// Interleaved base/new wall-clock, `n = 3` pairs, at the `down` geometry:
///
/// (a) [`inter_group_feedback_base`] against [`inter_group_feedback`], summed over every group
///     `g0` of one `(896, 4864)` fit;
/// (b) the whole [`gptq_e8`] at `(896, 4864)`, `beam = 4`, `wscale = true`, `act_order = true`.
///
/// The whole-fit base arm is the measured new total minus the measured new feedback plus the
/// measured base feedback in the same pair: the feedback is the only term whose body changed, and
/// both of its arms sit in the same process on the same fixture, so the identity carries the
/// unchanged work (`feedback_u`, the two beam passes, the pick, the scatters) across exactly.
#[test]
#[ignore = "measurement: --release --ignored --nocapture"]
fn gfeed_speed() {
    let (o, n) = DOWN;
    let damp = word(DAMP);
    let h = spd_words(n, 0x5EED_0F1A_5EED_0F1A);
    let w = lcg_words(o * n, 0x1234_5678_9ABC_DEF0);
    let t = Instant::now();
    let ufac = feedback_u(&h, n, damp).expect("SPD factor");
    eprintln!("GFEED setup: feedback_u {:.3} s", t.elapsed().as_secs_f64());
    eprintln!(
        "GFEED machine: available_parallelism={:?}",
        std::thread::available_parallelism()
    );
    let er = vec![-14i32; o];
    let e = residuals(o, GROUP, 0x00FF_00FF_00FF_00FF);
    let groups: Vec<usize> = (0..n).step_by(GROUP).collect();
    let (mut a_base, mut a_new, mut b_new) = (Vec::new(), Vec::new(), Vec::new());
    for rep in 0..3 {
        let mut wb = w.clone();
        let mut erb = er.clone();
        let mut wn = w.clone();
        let mut ern = er.clone();
        let t = Instant::now();
        for &g0 in &groups {
            let gs = GROUP.min(n - g0);
            inter_group_feedback_base(&mut wb, &mut erb, &e, &ufac, n, g0, gs).expect("base");
        }
        let tb = t.elapsed().as_secs_f64();
        let t = Instant::now();
        for &g0 in &groups {
            let gs = GROUP.min(n - g0);
            inter_group_feedback(&mut wn, &mut ern, &e, &ufac, n, g0, gs).expect("new");
        }
        let tn = t.elapsed().as_secs_f64();
        assert_eq!((wb, erb), (wn, ern), "the arms disagree at rep {rep}");
        let t = Instant::now();
        let out = gptq_e8(&w, &h, o, n, damp, true, true, 4).expect("gptq_e8");
        let tq = t.elapsed().as_secs_f64();
        eprintln!(
            "GFEED rep {rep}: (a) base {tb:.3} s new {tn:.3} s ratio {:.1}x | (b) new {tq:.3} s chk {}",
            tb / tn,
            out.scale[0]
        );
        a_base.push(tb);
        a_new.push(tn);
        b_new.push(tq);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let spread = |v: &[f64]| {
        (
            v.iter().copied().fold(f64::INFINITY, f64::min),
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let (mb, mn, mq) = (mean(&a_base), mean(&a_new), mean(&b_new));
    let (lob, hib) = spread(&a_base);
    let (lon, hin) = spread(&a_new);
    let (loq, hiq) = spread(&b_new);
    let base_fit = mq - mn + mb;
    eprintln!(
        "GFEED (a) mean base {mb:.3} [{lob:.3}, {hib:.3}] new {mn:.3} [{lon:.3}, {hin:.3}] ratio {:.1}x",
        mb / mn
    );
    eprintln!(
        "GFEED (b) mean new {mq:.3} [{loq:.3}, {hiq:.3}] base-equivalent {base_fit:.3} s ratio {:.2}x",
        base_fit / mq
    );
}
