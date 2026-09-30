#![allow(
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    reason = "test harnesses use standard mathematical names, timing loops and literals"
)]

use super::*;
use crate::engine::mixer::{delta_head_step_fast, delta_head_step_ref};

#[inline]
#[allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    reason = "conversion to Q32.32 fixed point for test harnesses"
)]
fn f32_to_q32(x: f32) -> i64 {
    (f64::from(x) * ((1i64 << 32) as f64)).round() as i64
}

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
    fn next_f32_range(&mut self, low: f32, high: f32) -> f32 {
        low + self.next_f32() * (high - low)
    }
}

#[test]
fn f32_avx2_equals_scalar() {
    let mut rng = SimpleRng::new(0x1234_5678_9ABC_DEF0);
    let dk = 128;
    let w = 32;

    let mut s_scalar = vec![0.0f32; dk * w];
    let mut s_avx2 = vec![0.0f32; dk * w];
    let mut o_scalar = vec![0.0f32; w];
    let mut o_avx2 = vec![0.0f32; w];

    for step in 0..1200 {
        let mut k = Vec::with_capacity(dk);
        let mut q = Vec::with_capacity(dk);
        let mut k_sq = 0.0f32;
        let mut q_sq = 0.0f32;
        for _ in 0..dk {
            let ki = rng.next_f32_range(-1.0, 1.0);
            let qi = rng.next_f32_range(-1.0, 1.0);
            k_sq += ki * ki;
            q_sq += qi * qi;
            k.push(ki);
            q.push(qi);
        }
        let k_norm = k_sq.sqrt();
        let q_norm = q_sq.sqrt();
        let q_scale = 1.0f32 / (128.0f32).sqrt();
        for ki in &mut k {
            *ki /= k_norm;
        }
        for qi in &mut q {
            *qi = (*qi / q_norm) * q_scale;
        }

        let mut v = Vec::with_capacity(w);
        for _ in 0..w {
            v.push(rng.next_f32_range(-4.0, 4.0));
        }

        let decay = rng.next_f32_range(0.5, 0.999);
        let beta = rng.next_f32_range(0.0, 1.0);

        delta_step_f32_scalar(&mut s_scalar, &mut o_scalar, &q, &k, &v, decay, beta);
        delta_step_f32_avx2(&mut s_avx2, &mut o_avx2, &q, &k, &v, decay, beta);

        for j in 0..w {
            assert_eq!(
                    o_scalar[j].to_bits(),
                    o_avx2[j].to_bits(),
                    "readout mismatch at step {step}, col {j}: scalar={:e} ({:08x}) vs avx2={:e} ({:08x})",
                    o_scalar[j],
                    o_scalar[j].to_bits(),
                    o_avx2[j],
                    o_avx2[j].to_bits(),
                );
        }

        for idx in 0..s_scalar.len() {
            assert_eq!(
                s_scalar[idx].to_bits(),
                s_avx2[idx].to_bits(),
                "state mismatch at step {step}, idx {idx}: scalar={:e} vs avx2={:e}",
                s_scalar[idx],
                s_avx2[idx],
            );
        }
    }
}

#[test]
fn exp_f32_accuracy_over_domain() {
    let mut max_rel_err = 0.0f64;
    let mut worst_x = 0.0f32;
    let mut x = -88.0f32;
    while x <= 88.0f32 {
        let actual = f64::from(exp_f32_scalar(x));
        let expected = (f64::from(x)).exp();
        let rel_err = (actual - expected).abs() / expected;
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
            worst_x = x;
        }
        x += 0.005;
    }
    println!(
        "exp_f32 max relative error over [-88, 88]: {:.6e} at x = {:.4}",
        max_rel_err, worst_x
    );
    assert!(
        max_rel_err < 2.0e-7,
        "max relative error {max_rel_err:e} exceeded 2e-7 bound"
    );
}

#[test]
fn exp_and_silu_avx2_equals_scalar() {
    let mut x = -88.0f32;
    while x <= 88.0f32 {
        let xs = [
            x,
            x + 0.001,
            x + 0.002,
            x + 0.003,
            x + 0.004,
            x + 0.005,
            x + 0.006,
            x + 0.007,
        ];
        let mut exp_scalar = [0.0f32; 8];
        let mut silu_scalar = [0.0f32; 8];
        for i in 0..8 {
            exp_scalar[i] = exp_f32_scalar(xs[i]);
            silu_scalar[i] = silu_f32_scalar(xs[i]);
        }

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let vx = _mm256_loadu_ps(xs.as_ptr());
            let vexp = exp_f32_avx2(vx);
            let vsilu = silu_f32_avx2(vx);
            let mut exp_avx2 = [0.0f32; 8];
            let mut silu_avx2 = [0.0f32; 8];
            _mm256_storeu_ps(exp_avx2.as_mut_ptr(), vexp);
            _mm256_storeu_ps(silu_avx2.as_mut_ptr(), vsilu);

            for i in 0..8 {
                assert_eq!(
                    exp_scalar[i].to_bits(),
                    exp_avx2[i].to_bits(),
                    "exp mismatch at x={}: scalar={:e} vs avx2={:e}",
                    xs[i],
                    exp_scalar[i],
                    exp_avx2[i]
                );
                assert_eq!(
                    silu_scalar[i].to_bits(),
                    silu_avx2[i].to_bits(),
                    "silu mismatch at x={}: scalar={:e} vs avx2={:e}",
                    xs[i],
                    silu_scalar[i],
                    silu_avx2[i]
                );
            }
        }
        x += 0.008;
    }

    // Test boundary values
    let special = [
        0.0f32,
        -0.0f32,
        -90.0f32,
        90.0f32,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0f32,
    ];
    for &s in &special {
        let e_s = exp_f32_scalar(s);
        let si_s = silu_f32_scalar(s);
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let vx = _mm256_set1_ps(s);
            let vexp = exp_f32_avx2(vx);
            let vsilu = silu_f32_avx2(vx);
            let mut exp_a = [0.0f32; 8];
            let mut silu_a = [0.0f32; 8];
            _mm256_storeu_ps(exp_a.as_mut_ptr(), vexp);
            _mm256_storeu_ps(silu_a.as_mut_ptr(), vsilu);
            if s.is_nan() {
                assert!(exp_a[0].is_nan());
                assert!(silu_a[0].is_nan());
            } else {
                assert_eq!(e_s.to_bits(), exp_a[0].to_bits());
                assert_eq!(si_s.to_bits(), silu_a[0].to_bits());
            }
        }
    }
}

#[test]
fn conv_silu_tap4_avx2_equals_scalar() {
    let mut rng = SimpleRng::new(0xdead_beef_cafe_babe);
    for _ in 0..1000 {
        let mut taps_s = [[0.0f32; 4]; 8];
        let mut hist_s = [[0.0f32; 4]; 8];
        let mut tap0 = [0.0f32; 8];
        let mut tap1 = [0.0f32; 8];
        let mut tap2 = [0.0f32; 8];
        let mut tap3 = [0.0f32; 8];
        let mut hist0 = [0.0f32; 8];
        let mut hist1 = [0.0f32; 8];
        let mut hist2 = [0.0f32; 8];
        let mut hist3 = [0.0f32; 8];

        for lane in 0..8 {
            for t in 0..4 {
                let tv = rng.next_f32_range(-2.0, 2.0);
                let hv = rng.next_f32_range(-4.0, 4.0);
                taps_s[lane][t] = tv;
                hist_s[lane][t] = hv;
            }
            tap0[lane] = taps_s[lane][0];
            tap1[lane] = taps_s[lane][1];
            tap2[lane] = taps_s[lane][2];
            tap3[lane] = taps_s[lane][3];
            hist0[lane] = hist_s[lane][0];
            hist1[lane] = hist_s[lane][1];
            hist2[lane] = hist_s[lane][2];
            hist3[lane] = hist_s[lane][3];
        }

        let mut out_scalar = [0.0f32; 8];
        for lane in 0..8 {
            out_scalar[lane] = conv_silu_tap4_scalar(taps_s[lane], hist_s[lane]);
        }

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let v_tap0 = _mm256_loadu_ps(tap0.as_ptr());
            let v_tap1 = _mm256_loadu_ps(tap1.as_ptr());
            let v_tap2 = _mm256_loadu_ps(tap2.as_ptr());
            let v_tap3 = _mm256_loadu_ps(tap3.as_ptr());
            let v_hist0 = _mm256_loadu_ps(hist0.as_ptr());
            let v_hist1 = _mm256_loadu_ps(hist1.as_ptr());
            let v_hist2 = _mm256_loadu_ps(hist2.as_ptr());
            let v_hist3 = _mm256_loadu_ps(hist3.as_ptr());

            let vy = conv_silu_tap4_avx2(
                v_tap0, v_tap1, v_tap2, v_tap3, v_hist0, v_hist1, v_hist2, v_hist3,
            );
            let mut out_avx2 = [0.0f32; 8];
            _mm256_storeu_ps(out_avx2.as_mut_ptr(), vy);

            for lane in 0..8 {
                assert_eq!(
                    out_scalar[lane].to_bits(),
                    out_avx2[lane].to_bits(),
                    "mismatch at lane {lane}: scalar={:e} vs avx2={:e}",
                    out_scalar[lane],
                    out_avx2[lane]
                );
            }
        }
    }
}

#[test]
fn f32_tracks_q32_oracle() {
    let decays = [0.5f32, 0.9f32, 0.99f32, 0.999f32];
    let dk = 128;
    let w = 32;
    let steps = 1024;
    let q_scale = 1.0f32 / (dk as f32).sqrt();
    let norm_q = f64::from(q_scale);

    println!(
        "\n{:<7} | {:<12} | {:<12} | {:<14} | {:<12} | {:<14} | {:<10}",
        "decay",
        "max |Δo|",
        "max |o|",
        "max |Δo|/|o|",
        "meas ε",
        "bound ‖q‖ε/(1-α)",
        "actual/bound"
    );
    println!("{:-<95}", "");

    for &decay in &decays {
        let mut rng = SimpleRng::new(0x4242_4242_1337_0001);
        let mut s_f32 = vec![0.0f32; dk * w];
        let mut s_q32 = vec![0i64; dk * w];
        let mut c_q32 = 1i64 << 32;
        let mut o_f32 = vec![0.0f32; w];

        let mut max_abs_diff_o = 0.0f32;
        let mut max_abs_o = 0.0f32;
        let mut max_eps = 0.0f64;

        let decay_q32 = f32_to_q32(decay);

        for _step in 0..steps {
            let mut k = Vec::with_capacity(dk);
            let mut k_sq = 0.0f32;
            for _ in 0..dk {
                let ki = rng.next_f32_range(-1.0, 1.0);
                k_sq += ki * ki;
                k.push(ki);
            }
            let k_norm = k_sq.sqrt();
            for ki in &mut k {
                *ki /= k_norm;
            }

            let mut q = Vec::with_capacity(dk);
            let mut q_sq = 0.0f32;
            for _ in 0..dk {
                let qi = rng.next_f32_range(-1.0, 1.0);
                q_sq += qi * qi;
                q.push(qi);
            }
            let q_norm = q_sq.sqrt();
            for qi in &mut q {
                *qi = (*qi / q_norm) * q_scale;
            }

            let beta = rng.next_f32_range(0.0, 1.0);
            let mut v = Vec::with_capacity(w);
            for _ in 0..w {
                v.push(rng.next_f32_range(-4.0, 4.0));
            }

            let k_q32: Vec<i64> = k.iter().map(|&x| f32_to_q32(x)).collect();
            let q_q32: Vec<i64> = q.iter().map(|&x| f32_to_q32(x)).collect();
            let v_q32: Vec<i64> = v.iter().map(|&x| f32_to_q32(x)).collect();
            let beta_q32 = f32_to_q32(beta);

            let s_f32_before = s_f32.clone();

            delta_step_f32_avx2(&mut s_f32, &mut o_f32, &q, &k, &v, decay, beta);
            let o_q32 = delta_head_step_ref(
                &mut s_q32, &mut c_q32, &q_q32, &k_q32, &v_q32, decay_q32, beta_q32,
            )
            .expect("oracle step failed");

            for j in 0..w {
                let o_q_float = q32_to_f32(o_q32[j]);
                let diff = (o_f32[j] - o_q_float).abs();
                if diff > max_abs_diff_o {
                    max_abs_diff_o = diff;
                }
                let mag = o_q_float.abs().max(o_f32[j].abs());
                if mag > max_abs_o {
                    max_abs_o = mag;
                }
            }

            for j in 0..w {
                let mut sk = 0.0f64;
                for i in 0..dk {
                    sk += f64::from(s_f32_before[i * w + j]) * f64::from(k[i]);
                }
                let mut sum_diff_sq = 0.0f64;
                for i in 0..dk {
                    let exact_col_i = f64::from(decay)
                        * (f64::from(s_f32_before[i * w + j])
                            - f64::from(beta) * sk * f64::from(k[i]))
                        + f64::from(beta) * f64::from(v[j]) * f64::from(k[i]);
                    let diff_i = f64::from(s_f32[i * w + j]) - exact_col_i;
                    sum_diff_sq += diff_i * diff_i;
                }
                let col_err = sum_diff_sq.sqrt();
                if col_err > max_eps {
                    max_eps = col_err;
                }
            }
        }

        let bound = norm_q * max_eps / (1.0 - f64::from(decay));
        let rel_err = f64::from(max_abs_diff_o) / f64::from(max_abs_o);
        let ratio = f64::from(max_abs_diff_o) / bound;

        println!(
            "{:<7.3} | {:<12.6e} | {:<12.6} | {:<14.6e} | {:<12.6e} | {:<14.6e} | {:<10.4}",
            decay, max_abs_diff_o, max_abs_o, rel_err, max_eps, bound, ratio
        );

        assert!(
            rel_err <= 1e-3,
            "decay {decay}: relative error {rel_err:e} exceeded 1e-3"
        );
    }
}

#[test]
#[ignore = "timing benchmark: run with cargo test --release --lib f32_vs_q32_timing -- --ignored --nocapture"]
fn f32_vs_q32_timing() {
    let dk = 128;
    let w = 32;
    let steps = 200;
    let reps = 200;
    let runs = 5;
    let total_steps = steps * reps;
    let total_elem_steps = (total_steps * dk * w) as f64;

    let mut rng = SimpleRng::new(0xDEAD_BEEF_CAFE_BABE);
    let q_scale = 1.0f32 / (dk as f32).sqrt();

    let mut k_steps = Vec::with_capacity(steps);
    let mut q_steps = Vec::with_capacity(steps);
    let mut v_steps = Vec::with_capacity(steps);
    let mut k_q32_steps = Vec::with_capacity(steps);
    let mut q_q32_steps = Vec::with_capacity(steps);
    let mut v_q32_steps = Vec::with_capacity(steps);

    let decay = 0.99f32;
    let beta = 0.8f32;
    let decay_q32 = f32_to_q32(decay);
    let beta_q32 = f32_to_q32(beta);

    for _ in 0..steps {
        let mut k = Vec::with_capacity(dk);
        let mut q = Vec::with_capacity(dk);
        let mut k_sq = 0.0f32;
        let mut q_sq = 0.0f32;
        for _ in 0..dk {
            let ki = rng.next_f32_range(-1.0, 1.0);
            let qi = rng.next_f32_range(-1.0, 1.0);
            k_sq += ki * ki;
            q_sq += qi * qi;
            k.push(ki);
            q.push(qi);
        }
        let k_norm = k_sq.sqrt();
        let q_norm = q_sq.sqrt();
        for ki in &mut k {
            *ki /= k_norm;
        }
        for qi in &mut q {
            *qi = (*qi / q_norm) * q_scale;
        }

        let mut v = Vec::with_capacity(w);
        for _ in 0..w {
            v.push(rng.next_f32_range(-4.0, 4.0));
        }

        let k_q32: Vec<i64> = k.iter().map(|&x| f32_to_q32(x)).collect();
        let q_q32: Vec<i64> = q.iter().map(|&x| f32_to_q32(x)).collect();
        let v_q32: Vec<i64> = v.iter().map(|&x| f32_to_q32(x)).collect();

        k_steps.push(k);
        q_steps.push(q);
        v_steps.push(v);
        k_q32_steps.push(k_q32);
        q_q32_steps.push(q_q32);
        v_q32_steps.push(v_q32);
    }

    println!("\n=== f32 vs Q32.32 Timing (single-thread, best of 5, 200 steps x 200 reps = 40,000 steps) ===");

    let mut s_f32 = vec![0.0f32; dk * w];
    let mut s_q32 = vec![0i64; dk * w];
    let mut c_q32 = 1i64 << 32;
    let mut o_f32 = vec![0.0f32; w];
    for s in 0..steps {
        delta_step_f32_avx2(
            &mut s_f32,
            &mut o_f32,
            &q_steps[s],
            &k_steps[s],
            &v_steps[s],
            decay,
            beta,
        );
        let _ = delta_head_step_fast(
            &mut s_q32,
            &mut c_q32,
            &q_q32_steps[s],
            &k_q32_steps[s],
            &v_q32_steps[s],
            decay_q32,
            beta_q32,
        );
    }

    let mut q32_best_ns = f64::INFINITY;
    for run in 0..runs {
        let mut total_dur = std::time::Duration::ZERO;
        for _rep in 0..reps {
            s_q32.fill(0);
            c_q32 = 1i64 << 32;
            let t0 = std::time::Instant::now();
            for step in 0..steps {
                let o = delta_head_step_fast(
                    &mut s_q32,
                    &mut c_q32,
                    &q_q32_steps[step],
                    &k_q32_steps[step],
                    &v_q32_steps[step],
                    decay_q32,
                    beta_q32,
                )
                .unwrap();
                std::hint::black_box(o);
            }
            total_dur += t0.elapsed();
        }
        let ns_per_elem = (total_dur.as_nanos() as f64) / total_elem_steps;
        println!(
            "Q32.32 run {run}: {:.3} ms total, {:.4} ns / elem-step",
            total_dur.as_secs_f64() * 1000.0,
            ns_per_elem
        );
        if ns_per_elem < q32_best_ns {
            q32_best_ns = ns_per_elem;
        }
    }

    let mut f32_best_ns = f64::INFINITY;
    for run in 0..runs {
        let mut total_dur = std::time::Duration::ZERO;
        for _rep in 0..reps {
            s_f32.fill(0.0);
            let t0 = std::time::Instant::now();
            for step in 0..steps {
                delta_step_f32_avx2(
                    &mut s_f32,
                    &mut o_f32,
                    &q_steps[step],
                    &k_steps[step],
                    &v_steps[step],
                    decay,
                    beta,
                );
                std::hint::black_box(&o_f32);
            }
            total_dur += t0.elapsed();
        }
        let ns_per_elem = (total_dur.as_nanos() as f64) / total_elem_steps;
        println!(
            "f32 AVX2 run {run}: {:.3} ms total, {:.4} ns / elem-step",
            total_dur.as_secs_f64() * 1000.0,
            ns_per_elem
        );
        if ns_per_elem < f32_best_ns {
            f32_best_ns = ns_per_elem;
        }
    }

    let speedup = q32_best_ns / f32_best_ns;
    println!("\nSummary (best of 5):");
    println!(
        "  Q32.32 (delta_head_step_fast): {:.4} ns / elem-step ({:.2} µs / 4096-elem step)",
        q32_best_ns,
        q32_best_ns * 4096.0 / 1000.0
    );
    println!(
        "  f32 AVX2+FMA (delta_step_f32_avx2): {:.4} ns / elem-step ({:.2} µs / 4096-elem step)",
        f32_best_ns,
        f32_best_ns * 4096.0 / 1000.0
    );
    println!("  Speedup: {speedup:.2}x");
}

#[test]
fn f32_readout_to_q32_overflow_and_nan() {
    assert!(matches!(
        f32_readout_to_q32(f32::NAN),
        Err(StepError::Overflow)
    ));
    assert!(matches!(
        f32_readout_to_q32(f32::INFINITY),
        Err(StepError::Overflow)
    ));
    assert!(matches!(
        f32_readout_to_q32(f32::NEG_INFINITY),
        Err(StepError::Overflow)
    ));
    assert!(matches!(
        f32_readout_to_q32(1e20f32),
        Err(StepError::Overflow)
    ));
    assert!(matches!(
        f32_readout_to_q32(-1e20f32),
        Err(StepError::Overflow)
    ));

    assert_eq!(f32_readout_to_q32(0.0).unwrap(), 0);
    assert_eq!(f32_readout_to_q32(1.0).unwrap(), 1i64 << 32);
    assert_eq!(f32_readout_to_q32(-1.0).unwrap(), -(1i64 << 32));
    assert_eq!(f32_readout_to_q32(0.5).unwrap(), 1i64 << 31);
    assert_eq!(f32_readout_to_q32(-0.5).unwrap(), -(1i64 << 31));
}

#[test]
fn affine_head_f32_state_carrier() {
    use crate::engine::mixer::{AffineHead, Carrier, HeadState};

    let mut head = AffineHead::with_carrier(128 * 32, Carrier::F32);
    let HeadState::F32 { ref mut s } = head.state else {
        panic!("expected F32 head state");
    };
    assert_eq!(s.len(), 128 * 32);

    for &x in s.iter() {
        assert_eq!(x.to_bits(), 0.0f32.to_bits());
    }

    s[42] = 2.5f32;

    let cloned = head.clone();
    let HeadState::F32 { s: ref s_cloned } = cloned.state else {
        panic!("expected F32 head state");
    };
    assert_eq!(s_cloned[42].to_bits(), 2.5f32.to_bits());

    let mut target = AffineHead::with_carrier(128 * 32, Carrier::F32);
    target.clone_from(&head);
    let HeadState::F32 { s: ref s_target } = target.state else {
        panic!("expected F32 head state");
    };
    assert_eq!(s_target[42].to_bits(), 2.5f32.to_bits());
}

#[test]
fn l2_norm_avx2_equals_scalar() {
    let mut rng = SimpleRng::new(0xABCD_EF01_2345_6789);
    for len in [8, 16, 32, 64, 128, 256] {
        let mut x_scalar = vec![0.0f32; len];
        let mut x_avx2 = vec![0.0f32; len];
        for i in 0..len {
            let v = rng.next_f32_range(-10.0, 10.0);
            x_scalar[i] = v;
            x_avx2[i] = v;
        }
        let eps = 1e-6f32;
        let q_scale = 1.0f32 / (len as f32).sqrt();

        l2_normalize_f32_in_place_scalar(&mut x_scalar, eps, q_scale).unwrap();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            l2_normalize_f32_in_place_avx2_inner(&mut x_avx2, eps, q_scale).unwrap();
            for i in 0..len {
                assert_eq!(
                    x_scalar[i].to_bits(),
                    x_avx2[i].to_bits(),
                    "L2 norm mismatch at len {len}, idx {i}: scalar={:e} vs avx2={:e}",
                    x_scalar[i],
                    x_avx2[i]
                );
            }
        }
    }
}

#[test]
fn stage2_gated_norm_avx2_equals_scalar() {
    let mut rng = SimpleRng::new(0x5678_9ABC_DEF0_1234);
    let head_v = 128;
    let mut o = vec![0.0f32; head_v];
    let mut z = vec![0i64; head_v];
    let mut ssm_norm = vec![0.0f32; head_v];
    for i in 0..head_v {
        o[i] = rng.next_f32_range(-5.0, 5.0);
        z[i] = f32_to_q32(rng.next_f32_range(-5.0, 5.0));
        ssm_norm[i] = rng.next_f32_range(0.1, 2.0);
    }
    let eps = 1e-6f32;
    let mut dst_scalar = vec![0i64; head_v];
    let mut dst_avx2 = vec![0i64; head_v];

    stage2_gated_norm_f32_scalar(&o, &z, &ssm_norm, eps, &mut dst_scalar).unwrap();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        stage2_gated_norm_f32_avx2_inner(&o, &z, &ssm_norm, eps, &mut dst_avx2).unwrap();
        for i in 0..head_v {
            assert_eq!(
                dst_scalar[i], dst_avx2[i],
                "stage2 norm mismatch at idx {i}: scalar={} vs avx2={}",
                dst_scalar[i], dst_avx2[i]
            );
        }
    }
}

#[test]
#[allow(
    clippy::cast_possible_truncation,
    reason = "test harnesses use intentional truncations and bit conversions"
)]
fn f32_readout_equivalence_tests() {
    let old_readout = |o: f32| -> Result<i64, StepError> {
        if !o.is_finite() {
            return Err(StepError::Overflow);
        }
        let scaled = f64::from(o) * 4_294_967_296.0;
        let rounded = scaled.round();
        if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&rounded) {
            return Err(StepError::Overflow);
        }
        #[allow(clippy::cast_possible_truncation, reason = "bounds checked above")]
        Ok(rounded as i64)
    };

    // (i) Every f32 in [-2^20, 2^20] that sits exactly on a .5 tie after scaling
    let scale = 4_294_967_296.0;
    for n in -1_048_576_i64..=1_048_576_i64 {
        for &sign in &[1.0, -1.0] {
            let target_scaled = (n.abs() as f64 + 0.5) * sign;
            let o = (target_scaled / scale) as f32;
            let scaled = f64::from(o) * scale;
            if (scaled.abs() - scaled.abs().floor() - 0.5).abs() < 1e-9 {
                let old_res = old_readout(o);
                let new_res = f32_readout_to_q32(o);
                assert_eq!(
                    old_res, new_res,
                    "Mismatch at tie point n={n}, sign={sign}, o={o}"
                );
            }
        }
    }

    // (ii) Range-check boundaries
    let boundaries = [
        0.0f32,
        -0.0f32,
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0001),
        2147483647.0f32,
        -2147483647.0f32,
        2147483648.0f32,
        -2147483648.0f32,
        2147483649.0f32,
        -2147483649.0f32,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e20f32,
        -1e20f32,
    ];
    for &b in &boundaries {
        assert_eq!(
            old_readout(b),
            f32_readout_to_q32(b),
            "Boundary mismatch for {b}"
        );
    }
    let u_max = 2147483648.0f32.to_bits();
    for u in (u_max - 50_000)..(u_max + 50_000) {
        let f_pos = f32::from_bits(u);
        let f_neg = -f_pos;
        assert_eq!(old_readout(f_pos), f32_readout_to_q32(f_pos));
        assert_eq!(old_readout(f_neg), f32_readout_to_q32(f_neg));
    }

    // (iii) Deterministic 2^24-sample sweep of random f32 bit patterns that are finite and in range
    let mut rng = SimpleRng::new(0xDEAD_BEEF_CAFE_BABE);
    let n_samples = 1 << 24;
    for _ in 0..n_samples {
        let bits = rng.next_u64() as u32;
        let f = f32::from_bits(bits);
        if f.is_finite() && f.abs() < 2147483648.0 {
            let ro = old_readout(f);
            let rn = f32_readout_to_q32(f);
            assert_eq!(ro, rn, "Sweep mismatch for bits {bits:#010x} (f={f})");
        }
    }
}

#[test]
#[ignore = "release microbench of conv_silu_cols_f32 at k=200 with the real shape (6144 channels, 4 taps)"]
#[allow(
    clippy::cast_possible_truncation,
    reason = "test harnesses use intentional float-to-int truncations"
)]
fn bench_conv_silu_cols_f32_k200() {
    use std::time::Instant;
    let kernel = 4;
    let channels = 6144;
    let k = 200;
    let mut rng = SimpleRng::new(0xCAFE_BABE_0123_4567);
    let weights: Vec<i64> = (0..kernel * channels)
        .map(|_| (rng.next_f32_range(-1.0, 1.0) * 4_294_967_296.0) as i64)
        .collect();
    let xs: Vec<Vec<i64>> = (0..k)
        .map(|_| {
            (0..channels)
                .map(|_| (rng.next_f32_range(-1.0, 1.0) * 4_294_967_296.0) as i64)
                .collect()
        })
        .collect();
    let col_refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();

    let g = crate::engine::mixer::AffineGeom {
        conv_kernel: kernel,
        n_k_heads: 16,
        n_v_heads: 16,
        head_k: 128,
        head_v: 128,
    };

    let mut samples_us = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut st = crate::engine::mixer::AffineState::with_slices_carrier(
            &g,
            1,
            1,
            crate::engine::mixer::Carrier::F32,
        );
        let layer = st.layer_mut(0).unwrap();
        let t0 = Instant::now();
        let out = crate::engine::mixer::conv_silu_cols_carrier(
            layer,
            &weights,
            kernel,
            &col_refs,
            crate::engine::mixer::Carrier::F32,
        )
        .unwrap();
        std::hint::black_box(out);
        samples_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = samples_us[0];
    let median = samples_us[2];
    println!(
        "conv_silu_cols_f32 k=200: best={:.2} µs ({:.3} ms), median={:.2} µs ({:.3} ms)",
        best,
        best / 1000.0,
        median,
        median / 1000.0
    );
}

#[allow(
    clippy::too_many_arguments,
    reason = "test harness replicates the oracle signature"
)]
fn old_stage2_gated_norm_oracle(
    raw_f32: &[f32],
    cols: &[crate::engine::mixer::AffineCol<'_>],
    ssm_norm_f: &[f32],
    eps_f: f32,
    n_v_heads: usize,
    head_v: usize,
    slices: usize,
    k: usize,
) -> Result<Vec<Vec<i64>>, StepError> {
    let val_dim = n_v_heads * head_v;
    let w = head_v / slices;
    let mut out = vec![0i64; k * val_dim];

    let gate_norm_f32 = |h: usize, dst: &mut [i64]| -> Result<(), StepError> {
        let vr = h * head_v..(h + 1) * head_v;
        let mut o_vec;
        let mut o_stack = [0.0f32; 256];
        let o_slice = if head_v <= 256 {
            &mut o_stack[..head_v]
        } else {
            o_vec = vec![0.0f32; head_v];
            &mut o_vec[..]
        };
        for (c, (_, z, ..)) in cols.iter().enumerate() {
            for sl in 0..slices {
                let at = (h * slices + sl) * k * w + c * w;
                o_slice[sl * w..(sl + 1) * w].copy_from_slice(&raw_f32[at..at + w]);
            }
            let zh = &z[vr.clone()];
            let dcol = &mut dst[c * head_v..(c + 1) * head_v];
            stage2_gated_norm_f32(o_slice, zh, ssm_norm_f, eps_f, dcol)?;
        }
        Ok(())
    };

    for h in 0..n_v_heads {
        let at = h * k * head_v;
        gate_norm_f32(h, &mut out[at..at + k * head_v])?;
    }

    let mut cols_out: Vec<Vec<i64>> = (0..k).map(|_| Vec::with_capacity(val_dim)).collect();
    for h in 0..n_v_heads {
        for (c, col) in cols_out.iter_mut().enumerate() {
            let at = h * k * head_v + c * head_v;
            col.extend_from_slice(&out[at..at + head_v]);
        }
    }
    Ok(cols_out)
}

#[test]
#[allow(
    clippy::similar_names,
    clippy::cast_possible_truncation,
    reason = "test harness uses standard geometry variables and truncation"
)]
fn stage2_heads_cols_f32_equals_old_oracle() {
    let n_v_heads = 16;
    let head_v = 128;
    let slices = 4;
    let w = head_v / slices;
    let val_dim = n_v_heads * head_v;
    let eps_f = 1e-5f32;

    let mut rng = SimpleRng::new(0xABCD_EF01_2345_6789);

    let ssm_norm_f: Vec<f32> = (0..head_v).map(|_| rng.next_f32_range(0.5, 1.5)).collect();

    for &k in &[2, 7, 64, 200] {
        let n_tasks = n_v_heads * slices;
        let raw_f32: Vec<f32> = (0..n_tasks * k * w)
            .map(|_| rng.next_f32_range(-2.0, 2.0))
            .collect();

        let dummy = Vec::new();
        let z_data: Vec<Vec<i64>> = (0..k)
            .map(|_| {
                (0..val_dim)
                    .map(|_| f32_to_q32(rng.next_f32_range(-3.0, 3.0)))
                    .collect()
            })
            .collect();
        let cols: Vec<crate::engine::mixer::AffineCol<'_>> = z_data
            .iter()
            .map(|z| {
                (
                    dummy.as_slice(),
                    z.as_slice(),
                    dummy.as_slice(),
                    dummy.as_slice(),
                )
            })
            .collect();

        let oracle_out = old_stage2_gated_norm_oracle(
            &raw_f32,
            &cols,
            &ssm_norm_f,
            eps_f,
            n_v_heads,
            head_v,
            slices,
            k,
        )
        .expect("oracle failed");

        // Test both sequential and parallel paths
        for &parallel in &[false, true] {
            let new_out = crate::engine::mixer::stage2_heads_cols_f32(
                &raw_f32,
                &cols,
                &ssm_norm_f,
                eps_f,
                n_v_heads,
                head_v,
                slices,
                k,
                parallel,
            )
            .expect("new stage2 failed");

            assert_eq!(
                new_out.len(),
                oracle_out.len(),
                "k={k}, parallel={parallel}: column count mismatch"
            );
            for c in 0..k {
                assert_eq!(
                    new_out[c], oracle_out[c],
                    "k={k}, parallel={parallel}, column {c}: output word mismatch"
                );
            }
        }
    }
}

#[test]
#[ignore = "release microbench of prefill heads call at k=200 with real shape (16 heads, head dim 128), pooled"]
#[allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    reason = "timing benchmark harness"
)]
fn bench_delta_heads_cols_f32_k200() {
    use std::time::Instant;
    let g = crate::engine::mixer::AffineGeom {
        conv_kernel: 4,
        n_k_heads: 16,
        n_v_heads: 16,
        head_k: 128,
        head_v: 128,
    };
    let k = 200;
    let eps = 4295i64;
    let slices = crate::engine::mixer::head_slices(g.head_v);

    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let p = crate::engine::mixer::AffineVectors {
        conv: vec![0; g.conv_kernel * (2 * key_dim + val_dim)],
        dt: vec![0; g.n_v_heads],
        a: vec![-1000; g.n_v_heads],
        norm: vec![1i64 << 32; g.head_v],
    };
    let mut rng = SimpleRng::new(0x1234_5678_9ABC_DEF0);
    let mut word = || (rng.next_u64() as i64) % (1i64 << 33);
    let cols_data: Vec<[Vec<i64>; 4]> = (0..k)
        .map(|_| {
            let conv: Vec<i64> = (0..2 * key_dim + val_dim).map(|_| word()).collect();
            let z: Vec<i64> = (0..val_dim).map(|_| word()).collect();
            let b: Vec<i64> = (0..g.n_v_heads).map(|_| word().abs()).collect();
            let a: Vec<i64> = vec![0i64; g.n_v_heads];
            [conv, z, b, a]
        })
        .collect();
    let col_refs: Vec<crate::engine::mixer::AffineCol<'_>> = cols_data
        .iter()
        .map(|[c, z, b, a]| (c.as_slice(), z.as_slice(), b.as_slice(), a.as_slice()))
        .collect();

    // Warmup
    let mut st = crate::engine::mixer::AffineState::with_slices_carrier(
        &g,
        1,
        slices,
        crate::engine::mixer::Carrier::F32,
    );
    let layer = st.layer_mut(0).unwrap();
    let _ = crate::engine::mixer::delta_heads_cols_carrier(
        layer,
        &g,
        &p,
        &col_refs,
        eps,
        true, // pooled
        None,
        crate::engine::mixer::Carrier::F32,
    )
    .unwrap();

    let mut samples_us = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut st = crate::engine::mixer::AffineState::with_slices_carrier(
            &g,
            1,
            slices,
            crate::engine::mixer::Carrier::F32,
        );
        let layer = st.layer_mut(0).unwrap();
        let t0 = Instant::now();
        let out = crate::engine::mixer::delta_heads_cols_carrier(
            layer,
            &g,
            &p,
            &col_refs,
            eps,
            true, // pooled
            None,
            crate::engine::mixer::Carrier::F32,
        )
        .unwrap();
        std::hint::black_box(out);
        samples_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = samples_us[0];
    let median = samples_us[2];
    println!(
        "delta_heads_cols_f32 k=200 pooled: best={:.2} µs ({:.3} ms), median={:.2} µs ({:.3} ms)",
        best,
        best / 1000.0,
        median,
        median / 1000.0
    );
}
