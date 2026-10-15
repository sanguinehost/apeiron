use super::*;

#[test]
fn conv_silu_cols_refuses_i128_accumulator_overflow() {
    let g = AffineGeom {
        conv_kernel: 18,
        n_k_heads: 1,
        n_v_heads: 1,
        head_k: 1,
        head_v: 1,
    };
    let mut state = AffineState::with_slices_carrier(&g, 1, 1, Carrier::Q32);
    let layer = state.layer_mut(0).expect("constructed layer");
    let channels = 2 * g.n_k_heads * g.head_k + g.n_v_heads * g.head_v;
    let scale = 1_i64 << 62;
    for r in 0..g.conv_kernel - 1 {
        let value = if r < 9 { scale } else { -scale };
        layer.conv.as_q32_mut()[r * channels..(r + 1) * channels].fill(value);
    }
    let weights = vec![scale; g.conv_kernel * channels];
    let input = vec![-scale; channels];
    let old_window = layer.conv.clone();

    let result = conv_silu_cols_carrier(layer, &weights, g.conv_kernel, &[&input], Carrier::Q32);

    assert!(matches!(result, Err(StepError::Overflow)));
    assert_eq!(layer.conv, old_window);

    // The same cancellation is valid when every prefix remains within i128.
    let safe_scale = 1_i64 << 60;
    for r in 0..g.conv_kernel - 1 {
        let value = if r < 9 { safe_scale } else { -safe_scale };
        layer.conv.as_q32_mut()[r * channels..(r + 1) * channels].fill(value);
    }
    let safe_weights = vec![safe_scale; g.conv_kernel * channels];
    let safe_input = vec![-safe_scale; channels];
    let result = conv_silu_cols_carrier(
        layer,
        &safe_weights,
        g.conv_kernel,
        &[&safe_input],
        Carrier::Q32,
    );
    assert_eq!(result, Ok(vec![vec![0; channels]]));
}

#[test]
fn conv_silu_cols_refuses_rounding_bias_overflow() {
    let g = AffineGeom {
        conv_kernel: 3,
        n_k_heads: 1,
        n_v_heads: 1,
        head_k: 1,
        head_v: 1,
    };
    let mut state = AffineState::with_slices_carrier(&g, 1, 1, Carrier::Q32);
    let layer = state.layer_mut(0).expect("constructed layer");
    let channels = 2 * g.n_k_heads * g.head_k + g.n_v_heads * g.head_v;
    layer.conv.as_q32_mut()[0] = i64::MIN;
    layer.conv.as_q32_mut()[channels] = i64::MAX;
    let mut weights = vec![0; g.conv_kernel * channels];
    weights[0] = i64::MIN;
    weights[1] = i64::MAX;
    weights[2] = 2;
    let mut input = vec![0; channels];
    input[0] = i64::MAX;
    let old_window = layer.conv.clone();

    let result = conv_silu_cols_carrier(layer, &weights, g.conv_kernel, &[&input], Carrier::Q32);

    assert!(matches!(result, Err(StepError::Overflow)));
    assert_eq!(layer.conv, old_window);
}

/// One `k`-column call equals `k` successive one-column calls on a cloned layer: every output
/// word and the resulting `layer.conv` window. `channels = 100` is not a multiple of 8, and
/// `k = 33` exercises the last column, whose window advance has no successor row to load.
#[test]
#[allow(
    clippy::cast_possible_wrap,
    reason = "xorshift words reinterpreted as signed fixed-point test inputs"
)]
fn conv_silu_cols_k_columns_equal_k_single_column_calls() {
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 1,
        n_v_heads: 1,
        head_k: 16,
        head_v: 68,
    };
    let channels = 2 * g.n_k_heads * g.head_k + g.n_v_heads * g.head_v;
    assert_eq!(channels, 100);
    let mut rng = 0x243f_6a88_85a3_08d3_u64;
    let mut word = |bound: i64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as i64) % bound
    };
    let weights: Vec<i64> = (0..g.conv_kernel * channels)
        .map(|_| word(1 << 20))
        .collect();
    let mut state_all = AffineState::with_slices(&g, 1, 1);
    {
        let layer = state_all.layer_mut(0).expect("constructed layer");
        match &mut layer.conv {
            ConvState::Q32(conv) => {
                for v in conv {
                    *v = word(1 << 20);
                }
            }
            ConvState::F32(conv) => {
                for v in conv {
                    *v = f32_carrier::q32_to_f32(word(1 << 20));
                }
            }
        }
    }
    let mut state_one = state_all.clone();
    let layer_all = state_all.layer_mut(0).expect("constructed layer");
    let layer_one = state_one.layer_mut(0).expect("constructed layer");
    for k in [2usize, 5, 33] {
        let xs: Vec<Vec<i64>> = (0..k)
            .map(|_| (0..channels).map(|_| word(1 << 20)).collect())
            .collect();
        let cols: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
        let got_all =
            conv_silu_cols(layer_all, &weights, g.conv_kernel, &cols).expect("no overflow");
        let mut got_one: Vec<Vec<i64>> = Vec::with_capacity(k);
        for x in &xs {
            let single = conv_silu_cols(layer_one, &weights, g.conv_kernel, &[x.as_slice()])
                .expect("no overflow");
            got_one.push(single.into_iter().next().expect("one column"));
        }
        assert_eq!(got_all, got_one, "outputs, k = {k}");
        assert_eq!(layer_all.conv, layer_one.conv, "window, k = {k}");
    }
}

/// Value-column slicing is bit-identical to whole heads: outputs over several multi-column
/// passes, every slice's state words and its deferred decay, for 2, 4 and 8 slices, pooled
/// and serial, including renormalisations (a strong decay drives `c` under the threshold).
#[test]
#[allow(
    clippy::cast_possible_wrap,
    reason = "xorshift words reinterpreted as signed fixed-point test inputs"
)]
fn sliced_heads_equal_whole_heads() {
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 2,
        n_v_heads: 4,
        head_k: 128,
        head_v: 128,
    };
    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let mut rng = 0x9e37_79b9_7f4a_7c15_u64;
    let mut word = |bound: i64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as i64) % bound
    };
    let p = AffineVectors {
        conv: vec![0; g.conv_kernel * (2 * key_dim + val_dim)],
        dt: (0..g.n_v_heads).map(|_| word(1 << 32)).collect(),
        a: (0..g.n_v_heads)
            .map(|_| -(word(1 << 34).abs() + (1 << 32)))
            .collect(),
        norm: (0..g.head_v).map(|_| word(1 << 33)).collect(),
    };
    // `q = k` per column keeps `kᵀq̃ = 1/√head_k` and `β ≥ ½`, so every `o_h` stays inside the
    // gated norm's range `[2⁻²⁰, 2²⁰]` mean square.
    let passes: Vec<Vec<[Vec<i64>; 4]>> = (0..4)
        .map(|_| {
            (0..3)
                .map(|_| {
                    let keys: Vec<i64> = (0..key_dim).map(|_| word(1 << 33)).collect();
                    let mut conv = keys.clone();
                    conv.extend_from_slice(&keys);
                    conv.extend((0..val_dim).map(|_| word(1 << 33)));
                    [
                        conv,
                        (0..val_dim).map(|_| word(1 << 33)).collect(),
                        (0..g.n_v_heads).map(|_| word(1 << 33).abs()).collect(),
                        (0..g.n_v_heads).map(|_| word(1 << 33)).collect(),
                    ]
                })
                .collect()
        })
        .collect();
    let run = |slices: usize, parallel: bool| {
        let mut st = AffineState::with_slices_carrier(&g, 1, slices, Carrier::Q32);
        let layer = st.layer_mut(0).expect("layer");
        let mut outs = Vec::new();
        for pass in &passes {
            let cols: Vec<AffineCol<'_>> = pass
                .iter()
                .map(|[c, z, b, a]| (c.as_slice(), z.as_slice(), b.as_slice(), a.as_slice()))
                .collect();
            outs.push(
                delta_heads_cols_carrier(layer, &g, &p, &cols, 1, parallel, None, Carrier::Q32)
                    .expect("step"),
            );
        }
        let (words, scales) = layer_state_words(layer, &g, slices);
        (outs, words, scales)
    };
    let (want_out, want_words, want_scales) = run(1, false);
    for slices in [2usize, 4, 8] {
        for parallel in [false, true] {
            let (out, words, scales) = run(slices, parallel);
            assert_eq!(
                out, want_out,
                "outputs, {slices} slices, parallel {parallel}"
            );
            assert_eq!(
                words, want_words,
                "state, {slices} slices, parallel {parallel}"
            );
            for (h, c) in scales {
                assert_eq!(c, want_scales[h].1, "decay of head {h}, {slices} slices");
            }
        }
    }
}

#[test]
#[allow(
    clippy::cast_possible_wrap,
    reason = "xorshift words reinterpreted as signed fixed-point test inputs"
)]
fn shared_decay_table_equals_per_column_gates() {
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 2,
        n_v_heads: 4,
        head_k: 128,
        head_v: 128,
    };
    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let mut rng = 0x1234_5678_9abc_def0_u64;
    let mut word = |bound: i64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as i64) % bound
    };
    let p = AffineVectors {
        conv: vec![0; g.conv_kernel * (2 * key_dim + val_dim)],
        dt: (0..g.n_v_heads).map(|_| word(1 << 32)).collect(),
        a: (0..g.n_v_heads)
            .map(|_| -(word(1 << 34).abs() + (1 << 32)))
            .collect(),
        norm: (0..g.head_v).map(|_| word(1 << 33)).collect(),
    };
    let cols: Vec<[Vec<i64>; 4]> = (0..3)
        .map(|_| {
            let keys: Vec<i64> = (0..key_dim).map(|_| word(1 << 33)).collect();
            let mut conv = keys.clone();
            conv.extend_from_slice(&keys);
            conv.extend((0..val_dim).map(|_| word(1 << 33)));
            [
                conv,
                (0..val_dim).map(|_| word(1 << 33)).collect(),
                (0..g.n_v_heads).map(|_| word(1 << 33).abs()).collect(),
                (0..g.n_v_heads).map(|_| word(1 << 33)).collect(),
            ]
        })
        .collect();
    let eps = 4295i64;

    // Sliced/whole path: the per-(head, column) β/decay table.
    let mut st = AffineState::with_slices_carrier(&g, 1, 1, Carrier::Q32);
    let layer = st.layer_mut(0).expect("layer");
    let col_refs: Vec<AffineCol<'_>> = cols
        .iter()
        .map(|[c, z, b, a]| (c.as_slice(), z.as_slice(), b.as_slice(), a.as_slice()))
        .collect();
    let shared = delta_heads_cols_carrier(layer, &g, &p, &col_refs, eps, true, None, Carrier::Q32)
        .expect("step");

    // Oracle: `delta_heads_custom` recomputes β = σ(b_h), α = exp(a_h · softplus(ssm + dt_h))
    // per call, so a corrupted table entry cannot agree with it.
    let mut st_o = AffineState::with_slices_carrier(&g, 1, 1, Carrier::Q32);
    let layer_o = st_o.layer_mut(0).expect("layer");
    let step =
        |s: &mut [i64], c: &mut i64, q: &[i64], k: &[i64], v: &[i64], decay: i64, beta: i64| {
            delta_head_step(s, c, q, k, v, decay, beta)
        };
    let mut oracle = Vec::with_capacity(cols.len());
    for [c, z, b, a] in &cols {
        oracle.push(
            delta_heads_custom(layer_o, &g, &p, c, z, b, a, eps, true, step).expect("oracle"),
        );
    }
    assert_eq!(
        shared, oracle,
        "shared β/decay table differs from per-column gates"
    );
}

#[test]
#[allow(
    clippy::unreadable_literal,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "random synthetic test data generation"
)]
fn test_fast_eq_ref_bit_for_bit_random_and_renorm() {
    let dk = 128;
    let dv = 128;
    let mut rng = 9_876_543_210_123_456_789_u64;
    let mut next_u64 = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    for trial in 0..50 {
        let mut s_ref = vec![0i64; dk * dv];
        let mut s_fast = vec![0i64; dk * dv];
        for i in 0..dk * dv {
            let v = ((next_u64() as i64) % (1 << 32)).max(-(1 << 32));
            s_ref[i] = v;
            s_fast[i] = v;
        }

        // Start near renorm threshold on half of the trials
        let mut c_ref = if trial % 2 == 0 {
            RENORM_THRESHOLD + (next_u64() as i64 % 500)
        } else {
            1i64 << 32
        };
        let mut c_fast = c_ref;

        let mut q = vec![0i64; dk];
        let mut k = vec![0i64; dk];
        let mut v = vec![0i64; dv];
        for i in 0..dk {
            k[i] = ((next_u64() as i64) % (1 << 31)).max(-(1 << 31));
            q[i] = ((next_u64() as i64) % (1 << 28)).max(-(1 << 28));
        }
        for j in 0..dv {
            v[j] = ((next_u64() as i64) % (1 << 34)).max(-(1 << 34));
        }
        let decay = (0.95 * (1u64 << 32) as f64) as i64;
        let beta = (0.8 * (1u64 << 32) as f64) as i64;

        let o_ref = delta_head_step_ref(&mut s_ref, &mut c_ref, &q, &k, &v, decay, beta)
            .expect("ref step ok");
        let o_fast = delta_head_step_fast(&mut s_fast, &mut c_fast, &q, &k, &v, decay, beta)
            .expect("fast step ok");

        assert_eq!(o_ref, o_fast, "trial {trial}: o mismatch");
        assert_eq!(c_ref, c_fast, "trial {trial}: c mismatch");
        assert_eq!(s_ref, s_fast, "trial {trial}: state matrix mismatch");
    }
}

/// AVX2 write updates match the scalar formula across widths, signs, zero keys, and refusal.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_write_matches_scalar_across_value_widths() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    for dv in [4usize, 16, 32, 64, 128] {
        let k: Vec<i64> = (0..128)
            .map(|i| match i % 5 {
                0 => 0,
                1 => 1_i64 << 32,
                2 => -(1_i64 << 32),
                3 => (1_i64 << 31) + 17,
                _ => -((1_i64 << 31) + 29),
            })
            .collect();
        let d_prime: Vec<i64> = (0..dv)
            .map(|j| {
                let magnitude = if j % 3 == 0 {
                    1_i64 << 46
                } else {
                    (1_i64 << 40) + (i64::try_from(j).expect("dv <= 128") * 37)
                };
                if j % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect();
        let initial: Vec<i64> = (0..128 * dv)
            .map(|n| {
                let magnitude = (1_i64 << 43) + i64::try_from(n % 97).expect("remainder < 97");
                if n % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect();

        let mut fast = initial.clone();
        let result = unsafe { avx2_kern::write_pass_avx2(&mut fast, &k, &d_prime, dv) };
        assert_eq!(result, Ok(()), "AVX2 dv={dv}");

        let mut scalar = initial.clone();
        for i in 0..128 {
            for j in 0..dv {
                let delta = mul_q32(k[i], d_prime[j]).expect("bounded product");
                scalar[i * dv + j] = scalar[i * dv + j]
                    .checked_add(delta)
                    .expect("bounded state addition");
            }
        }
        assert_eq!(fast, scalar, "scalar parity dv={dv}");

        // The state is fully updated before the bound scan reports overflow.
        let overflow_k = vec![1_i64 << 32; 128];
        let overflow_d = vec![1_i64 << 33; dv];
        let overflow_initial = vec![STATE_BOUND - 1; 128 * dv];
        let mut overflow_state = overflow_initial;
        let overflow_result = unsafe {
            avx2_kern::write_pass_avx2(&mut overflow_state, &overflow_k, &overflow_d, dv)
        };
        assert_eq!(overflow_result, Err(StepError::Overflow), "refusal dv={dv}");
        assert_eq!(
            overflow_state,
            vec![STATE_BOUND - 1 + (1_i64 << 33); 128 * dv],
            "all cells updated before refusal, dv={dv}"
        );
    }
}

/// The AVX2 write predicate refuses exactly `|res| > STATE_BOUND`: the boundary word
/// `-(STATE_BOUND + 1)` must refuse, `-STATE_BOUND` must pass. One-word edge that the old
/// `-B - 1` below-bound accepted.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_write_refuses_exact_negative_bound() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let dv = 4usize;
    let mut k = vec![0i64; 128];
    k[0] = 1i64 << 32; // mul_q32(1 << 32, -1) = -1

    let mut s = vec![0i64; 128 * dv];
    s[0] = -STATE_BOUND;
    let d_prime = [-1i64, 0, 0, 0]; // res[0] = -(STATE_BOUND + 1)
    let result = unsafe { avx2_kern::write_pass_avx2(&mut s, &k, &d_prime, dv) };
    assert_eq!(
        result,
        Err(StepError::Overflow),
        "res = -(STATE_BOUND + 1) must refuse"
    );

    let mut s_in = vec![0i64; 128 * dv];
    s_in[0] = -STATE_BOUND;
    let zero = [0i64; 4]; // res[0] = -STATE_BOUND
    let result = unsafe { avx2_kern::write_pass_avx2(&mut s_in, &k, &zero, dv) };
    assert_eq!(result, Ok(()), "res = -STATE_BOUND must pass");
}

#[test]
#[allow(
    clippy::unreadable_literal,
    clippy::cast_possible_wrap,
    clippy::explicit_iter_loop,
    reason = "synthetic state renormalization test"
)]
fn test_renormalisation_preserves_state_within_bound() {
    let dk = 128;
    let dv = 128;
    let len = dk * dv;
    let mut s_tilde = vec![0i64; len];
    let mut rng = 4_242_424_242_u64;
    for x in &mut s_tilde {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        *x = ((rng as i64) % (1 << 35)).max(-(1 << 35));
    }

    // Near threshold c = 2^-8
    let c: i64 = RENORM_THRESHOLD - 100;
    let mut s_folded = s_tilde.clone();
    for x in &mut s_folded {
        *x = mul_q32(*x, c).expect("mul");
    }

    // Compare un-renormalized exact S = (S̃ · c) / 2^32 vs S_folded
    let mut max_diff: i64 = 0;
    for i in 0..len {
        let exact_s =
            i64::try_from(round_shift_32(i128::from(s_tilde[i]) * i128::from(c))).unwrap();
        let diff = (s_folded[i] - exact_s).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    // Rounding bound: folding rounds each product to nearest integer, so difference is 0 LSB.
    assert_eq!(
        max_diff, 0,
        "folding S̃ · c equals un-renormalized rounded value bit for bit"
    );
}

#[test]
#[allow(
    clippy::unreadable_literal,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "8-step test with realistic model activation ranges"
)]
fn test_delta_head_step_8_real_steps_drift() {
    if crate::engine::testdata::model_path("models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf").is_none() {
        eprintln!("skip 8-step test: model not found");
        return;
    }

    let dk = 128;
    let dv = 128;
    let mut s_v0 = vec![0i64; dk * dv];
    let mut s_new = vec![0i64; dk * dv];
    let mut c_new = 1i64 << 32;

    let mut rng = 555_123_489_u64;
    let mut next_u64 = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    let decay = (0.95 * (1u64 << 32) as f64) as i64;
    let beta = (0.75 * (1u64 << 32) as f64) as i64;

    let mut max_diff_overall = 0i64;

    for step in 0..8 {
        let mut q_raw = vec![0i64; dk];
        let mut k_raw = vec![0i64; dk];
        let mut v = vec![0i64; dv];
        for i in 0..dk {
            q_raw[i] = (next_u64() as i64 % 20000) - 10000;
            k_raw[i] = (next_u64() as i64 % 20000) - 10000;
        }
        for j in 0..dv {
            v[j] = ((next_u64() as i64 % 50000) - 25000) << 24;
        }

        let eps = 1;
        let q_hat = l2_normalize(&q_raw, eps).expect("norm q");
        let k_hat = l2_normalize(&k_raw, eps).expect("norm k");
        let head_k_fixed = i64::try_from(dk).unwrap() << 32;
        let q_scale = fixed_rsqrt(head_k_fixed).expect("rsqrt");
        let mut q_tilde = Vec::with_capacity(dk);
        for &x in &q_hat {
            q_tilde.push(mul_q32(x, q_scale).expect("mul"));
        }

        let mut s_fast = s_new.clone();
        let mut c_fast = c_new;

        let o_v0 =
            delta_head_step_v0(&mut s_v0, &q_tilde, &k_hat, &v, decay, beta).expect("v0 step ok");
        let o_ref = delta_head_step_ref(&mut s_new, &mut c_new, &q_tilde, &k_hat, &v, decay, beta)
            .expect("ref step ok");
        let o_fast =
            delta_head_step_fast(&mut s_fast, &mut c_fast, &q_tilde, &k_hat, &v, decay, beta)
                .expect("fast step ok");
        assert_eq!(o_ref, o_fast, "step {step}: fast != ref bit-for-bit");
        assert_eq!(c_new, c_fast, "step {step}: c_new != c_fast");
        assert_eq!(s_new, s_fast, "step {step}: s_new != s_fast");

        let mut step_max_diff = 0i64;
        for j in 0..dv {
            let diff = (o_ref[j] - o_v0[j]).abs();
            if diff > step_max_diff {
                step_max_diff = diff;
            }
        }
        if step_max_diff > max_diff_overall {
            max_diff_overall = step_max_diff;
        }
        println!("Step {step}: max |new - v0| = {step_max_diff} counts in Q32.32");
    }
    println!("Max |new - v0| over 8 real steps: {max_diff_overall} counts");
    // Due to deferred decay rounding, difference is small (within ~100 counts in Q32.32, i.e. ~2e-8)
    assert!(
        max_diff_overall <= 256,
        "max |new - v0| {max_diff_overall} must be <= 256 counts"
    );
}

/// Helper evaluating one layer of heads using an explicit per-head step function.
#[allow(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    reason = "test benchmark helper mirroring delta_heads"
)]
fn delta_heads_custom<F>(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    conv: &[i64],
    z: &[i64],
    b: &[i64],
    a: &[i64],
    eps: i64,
    parallel: bool,
    step_fn: F,
) -> Result<Vec<i64>, StepError>
where
    F: FnMut(&mut [i64], &mut i64, &[i64], &[i64], &[i64], i64, i64) -> Result<Vec<i64>, StepError>
        + Send
        + Sync
        + Clone,
{
    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let head_k_fixed = i64::try_from(g.head_k).unwrap() << 32;
    let q_scale = fixed_rsqrt(head_k_fixed).map_err(|_| StepError::Overflow)?;
    let (q_all, rest) = conv.split_at(key_dim);
    let (k_all, v_all) = rest.split_at(key_dim);
    let mut out = vec![0i64; val_dim];
    let head = |h: usize, dst: &mut [i64]| -> Result<(), StepError> {
        let kh = h % g.n_k_heads;
        let kr = kh * g.head_k..(kh + 1) * g.head_k;
        let vr = h * g.head_v..(h + 1) * g.head_v;
        let q_hat = l2_normalize(&q_all[kr.clone()], eps)?;
        let k_hat = l2_normalize(&k_all[kr], eps)?;
        let mut q_tilde = Vec::with_capacity(g.head_k);
        for &x in &q_hat {
            q_tilde.push(mul_q32(x, q_scale).ok_or(StepError::Overflow)?);
        }
        let beta = fixed_sigmoid(b[h]).map_err(|_| StepError::Overflow)?;
        let sp = fixed_softplus(a[h].checked_add(p.dt[h]).ok_or(StepError::Overflow)?)
            .map_err(|_| StepError::Overflow)?;
        let decay = fixed_exp(mul_q32(p.a[h], sp).ok_or(StepError::Overflow)?)
            .map_err(|_| StepError::Overflow)?;
        let o = {
            let mut head_guard = layer.heads[h].lock();
            let HeadState::Q32 {
                ref mut s,
                ref mut c,
            } = head_guard.state
            else {
                return Err(StepError::BadConfig(String::from("affine carrier")));
            };
            (step_fn.clone())(s, c, &q_tilde, &k_hat, &v_all[vr.clone()], decay, beta)?
        };
        let normed = rms_norm_fixed(&o, eps).map_err(StepError::Norm)?;
        let zh = &z[vr];
        let mut j = 0usize;
        while j < g.head_v {
            let gained = mul_q32(normed[j], p.norm[j]).ok_or(StepError::Overflow)?;
            let gate = silu_fixed(zh[j]).map_err(StepError::Ffn)?;
            dst[j] = mul_q32(gained, gate).ok_or(StepError::Overflow)?;
            j += 1;
        }
        Ok(())
    };
    if let Some((_, e)) = attn_heads(
        g.n_v_heads,
        g.head_v,
        &mut out,
        parallel,
        &crate::engine::chart::task::FnHead(head),
    ) {
        return Err(e);
    }
    Ok(out)
}

/// `chunk_matches_oracle`: the chunked composition tracks the per-step oracle within the epic's
/// bound `ε/(1 − α)` (`Nous.Seq.ChunkedRounding.chunked_discrepancy_uniform_bound`), ε = 1 LSB and
/// α = 99/100, i.e. 100 LSB, on random heads at chunk lengths 2, 8 and `CHUNK_MAX`.
#[test]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "random synthetic head data; the trial loop is the test"
)]
fn chunk_matches_oracle_within_bound() {
    let dk = 128usize;
    let dv = 128usize;
    // α = 99/100: the epic's witness_delta_net_100_lsb instance, bound ε/(1 − α) = 100 LSB.
    let decay = (0.99 * (1u64 << 32) as f64) as i64;
    let eps = 4295i64;
    let q_scale = fixed_rsqrt(i64::try_from(dk).expect("dk fits i64") << 32).expect("rsqrt");
    let mut rng = 0x51ed_2701_9e37_79b9_u64;
    let mut word = |bound: i64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as i64) % bound
    };
    let mut max_out_dev = 0i64;
    let mut max_state_dev = 0i64;
    for chunk in [2usize, 8, CHUNK_MAX] {
        for _ in 0..2 {
            let entry: Vec<i64> = (0..dk * dv).map(|_| word(1 << 30)).collect();
            let mut q = vec![0i64; chunk * dk];
            let mut k = vec![0i64; chunk * dk];
            let mut v = vec![0i64; chunk * dv];
            let mut beta = vec![0i64; chunk];
            let decay_step = vec![decay; chunk];
            for t in 0..chunk {
                let q_raw: Vec<i64> = (0..dk).map(|_| word(1 << 20)).collect();
                let k_raw: Vec<i64> = (0..dk).map(|_| word(1 << 20)).collect();
                let q_hat = l2_normalize(&q_raw, eps).expect("q norm");
                let k_hat = l2_normalize(&k_raw, eps).expect("k norm");
                for (i, (&qh, &kh)) in q_hat.iter().zip(&k_hat).enumerate() {
                    q[t * dk + i] = mul_q32(qh, q_scale).expect("q scale");
                    k[t * dk + i] = kh;
                }
                for jv in 0..dv {
                    v[t * dv + jv] = word(1 << 33);
                }
                beta[t] = word(1 << 31).abs() + (1 << 30);
            }

            // Oracle: `chunk` one-step calls on the deferred-decay carrier.
            let mut s_ora = entry.clone();
            let mut c_ora = 1i64 << 32;
            let mut want = Vec::with_capacity(chunk * dv);
            for t in 0..chunk {
                let o = delta_head_step_ref(
                    &mut s_ora,
                    &mut c_ora,
                    &q[t * dk..(t + 1) * dk],
                    &k[t * dk..(t + 1) * dk],
                    &v[t * dv..(t + 1) * dv],
                    decay_step[t],
                    beta[t],
                )
                .expect("oracle step");
                want.extend_from_slice(&o);
            }

            // Candidate: one chunked composition from the same entry.
            let mut s_chk = entry.clone();
            let mut c_chk = 1i64 << 32;
            let got = delta_head_chunk(
                &mut s_chk,
                &mut c_chk,
                &q,
                &k,
                &v,
                &decay_step,
                &beta,
                dk,
                dv,
                chunk,
            )
            .expect("chunk");
            assert_eq!(c_chk, 1i64 << 32, "chunk exit decay, chunk {chunk}");

            for (a, b) in got.iter().zip(&want) {
                max_out_dev = max_out_dev.max((a - b).abs());
            }
            // Real states: the chunk carries `c = 1`, the oracle its deferred `c_ora`.
            for (w, &w_ora) in s_chk.iter().zip(&s_ora) {
                let real_ora = mul_q32(c_ora, w_ora).expect("oracle real state");
                max_state_dev = max_state_dev.max((w - real_ora).abs());
            }
        }
    }
    println!("chunk deviation: readout {max_out_dev} LSB, state {max_state_dev} LSB");
    assert!(
        max_out_dev <= 100,
        "readout deviation {max_out_dev} LSB exceeds the 100 LSB bound for alpha = 99/100"
    );
    assert!(
        max_state_dev <= 100,
        "state deviation {max_state_dev} LSB exceeds the 100 LSB bound for alpha = 99/100"
    );
}

/// The chunk kernel is a pure function of its inputs: two calls from the same entry produce the
/// same readouts, state and exit decay.
#[test]
#[allow(
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    reason = "xorshift words reinterpreted as signed fixed-point test inputs; q, k, v, s, c are \
              the recurrence's symbols"
)]
fn chunk_is_deterministic() {
    let dk = 16usize;
    let dv = 16usize;
    let chunk = 8usize;
    let mut rng = 0x0f0f_1234_5678_9abc_u64;
    let mut word = |bound: i64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as i64) % bound
    };
    let entry: Vec<i64> = (0..dk * dv).map(|_| word(1 << 28)).collect();
    let q: Vec<i64> = (0..chunk * dk).map(|_| word(1 << 24)).collect();
    let k: Vec<i64> = (0..chunk * dk).map(|_| word(1 << 24)).collect();
    let v: Vec<i64> = (0..chunk * dv).map(|_| word(1 << 30)).collect();
    let decay: Vec<i64> = (0..chunk).map(|_| word(1 << 32).abs()).collect();
    let beta: Vec<i64> = (0..chunk).map(|_| word(1 << 31).abs()).collect();
    let run = || {
        let mut s = entry.clone();
        let mut c = 1i64 << 32;
        let o = delta_head_chunk(&mut s, &mut c, &q, &k, &v, &decay, &beta, dk, dv, chunk)
            .expect("chunk");
        (o, s, c)
    };
    assert_eq!(run(), run(), "chunk kernel is not deterministic");
}

/// `refuses_on_width`: a chunk outside `1..=CHUNK_MAX`, a state word over [`STATE_BOUND`], a
/// non-positive deferred decay and a geometry mismatch each refuse, and no refusal writes a state
/// word.
#[test]
#[allow(
    clippy::many_single_char_names,
    reason = "q, k, v, s, c, d, b are the recurrence's and the gate's symbols"
)]
fn chunk_refuses_on_width() {
    let (dk, dv, chunk) = (8usize, 8usize, 2usize);
    let entry = vec![0i64; dk * dv];
    let q = vec![0i64; chunk * dk];
    let k = vec![0i64; chunk * dk];
    let v = vec![0i64; chunk * dv];
    let decay = vec![1i64 << 32; chunk];
    let beta = vec![1i64 << 31; chunk];
    let one = 1i64 << 32;

    // Chunk length over CHUNK_MAX and chunk length zero.
    for bad in [CHUNK_MAX + 1, 0] {
        let (mut s, mut c) = (entry.clone(), one);
        let d = vec![1i64 << 32; bad.max(1)];
        let b = vec![1i64 << 31; bad.max(1)];
        let qb = vec![0i64; bad * dk];
        let kb = vec![0i64; bad * dk];
        let vb = vec![0i64; bad * dv];
        let r = delta_head_chunk(&mut s, &mut c, &qb, &kb, &vb, &d, &b, dk, dv, bad);
        assert!(
            matches!(r, Err(StepError::BadConfig(_))),
            "chunk length {bad} must refuse with BadConfig, got {r:?}"
        );
        assert_eq!(s, entry, "refusal wrote a state word, chunk length {bad}");
        assert_eq!(c, one, "refusal wrote the decay, chunk length {bad}");
    }

    // A state word over STATE_BOUND.
    let mut wide = entry.clone();
    wide[0] = STATE_BOUND + 1;
    let mut c = one;
    let r = delta_head_chunk(&mut wide, &mut c, &q, &k, &v, &decay, &beta, dk, dv, chunk);
    assert!(matches!(r, Err(StepError::Overflow)));
    assert_eq!(wide[0], STATE_BOUND + 1, "refusal wrote a state word");
    assert_eq!(c, one, "refusal wrote the decay");

    // Non-positive deferred decay.
    let mut s = entry.clone();
    let mut c = 0i64;
    let r = delta_head_chunk(&mut s, &mut c, &q, &k, &v, &decay, &beta, dk, dv, chunk);
    assert!(matches!(r, Err(StepError::Overflow)));
    assert_eq!(s, entry, "refusal wrote a state word");
    assert_eq!(c, 0, "refusal wrote the decay");

    // Geometry mismatch: `v` carries a whole chunk of the wrong value width.
    let mut s = entry.clone();
    let mut c = one;
    let r = delta_head_chunk(
        &mut s,
        &mut c,
        &q,
        &k,
        &vec![0i64; chunk * (dv + 4)],
        &decay,
        &beta,
        dk,
        dv,
        chunk,
    );
    assert!(matches!(r, Err(StepError::BadConfig(_))));
    assert_eq!(s, entry, "refusal wrote a state word");
}

/// A one-step chunk is the step: `chunk == 1` reduces to [`delta_head_step_ref`] with bit-identical
/// readouts and state.
#[test]
#[allow(
    clippy::cast_possible_wrap,
    reason = "xorshift words reinterpreted as signed fixed-point test inputs"
)]
fn chunk_c1_equals_step() {
    let dk = 32usize;
    let dv = 32usize;
    let mut rng = 0xdead_beef_cafe_1234_u64;
    let mut word = |bound: i64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as i64) % bound
    };
    let entry: Vec<i64> = (0..dk * dv).map(|_| word(1 << 28)).collect();
    let q: Vec<i64> = (0..dk).map(|_| word(1 << 24)).collect();
    let k: Vec<i64> = (0..dk).map(|_| word(1 << 24)).collect();
    let v: Vec<i64> = (0..dv).map(|_| word(1 << 30)).collect();
    let decay = [word(1 << 32).abs()];
    let beta = [word(1 << 31).abs()];

    let mut s_ora = entry.clone();
    let mut c_ora = 1i64 << 32;
    let want = delta_head_step_ref(&mut s_ora, &mut c_ora, &q, &k, &v, decay[0], beta[0])
        .expect("oracle step");
    let mut s_chk = entry.clone();
    let mut c_chk = 1i64 << 32;
    let got = delta_head_chunk(&mut s_chk, &mut c_chk, &q, &k, &v, &decay, &beta, dk, dv, 1)
        .expect("one-step chunk");
    assert_eq!(got, want, "one-step chunk readout differs from the step");
    assert_eq!(s_chk, s_ora, "one-step chunk state differs from the step");
    assert_eq!(c_chk, c_ora, "one-step chunk decay differs from the step");
}

/// A random multi-column affine block on one layer. `q = k` per column, a unity `ssm_norm` gain,
/// `a = dt = 0`, and `ssm_a` chosen so the engine's `α = exp(ssm_a · softplus(0))` is the witness
/// decay near 0.98. Returns the layer's parameters and the `k` columns.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "random synthetic fixed-point inputs; α target is a real constant"
)]
fn random_affine_block(g: &AffineGeom, k: usize, seed: u64) -> (AffineVectors, Vec<[Vec<i64>; 4]>) {
    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let sp = fixed_softplus(0).expect("softplus(0)");
    let target = 0.98f64;
    let ssm_a = (target.ln() * (1u64 << 32) as f64 * (1u64 << 32) as f64 / sp as f64) as i64;
    let p = AffineVectors {
        conv: vec![0; g.conv_kernel * (2 * key_dim + val_dim)],
        dt: vec![0; g.n_v_heads],
        a: vec![ssm_a; g.n_v_heads],
        norm: vec![1i64 << 32; g.head_v],
    };
    let mut rng = seed;
    let mut word = |bound: i64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as i64) % bound
    };
    let cols = (0..k)
        .map(|_| {
            let keys: Vec<i64> = (0..key_dim).map(|_| word(1 << 33)).collect();
            let mut conv = keys.clone();
            conv.extend_from_slice(&keys);
            conv.extend((0..val_dim).map(|_| word(1 << 33)));
            [
                conv,
                (0..val_dim).map(|_| word(1 << 33)).collect(),
                (0..g.n_v_heads).map(|_| word(1 << 33).abs()).collect(),
                vec![0i64; g.n_v_heads],
            ]
        })
        .collect();
    (p, cols)
}

/// Every head's state words in whole-head layout plus each slice's deferred decay `(head, c)`.
fn layer_state_words(
    layer: &AffineLayer,
    g: &AffineGeom,
    slices: usize,
) -> (Vec<i64>, Vec<(usize, i64)>) {
    let w = g.head_v / slices;
    let mut words = vec![0i64; g.n_v_heads * g.head_k * g.head_v];
    let mut scales = Vec::new();
    for (t, m) in layer.heads.iter().enumerate() {
        let head = m.lock();
        let (h, sl) = (t / slices, t % slices);
        let HeadState::Q32 { ref s, c } = head.state else {
            panic!("expected Q32 head state");
        };
        for i in 0..g.head_k {
            let at = h * g.head_k * g.head_v + i * g.head_v + sl * w;
            words[at..at + w].copy_from_slice(&s[i * w..(i + 1) * w]);
        }
        scales.push((h, c));
    }
    (words, scales)
}

/// `flag_off_is_baseline`: the stage-1 per-step path is bit-identical to the existing one-column
/// oracle for a multi-column block (outputs, state words and deferred decay).
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the test builds a random block, runs both paths and compares them"
)]
fn exact_scan_off_is_bit_identical_to_the_per_step_oracle() {
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 2,
        n_v_heads: 4,
        head_k: 128,
        head_v: 128,
    };
    let slices = 1usize;
    let eps = 4295i64;
    let (p, cols) = random_affine_block(&g, 3, 0x0f0f_0f0f_1234_5678);

    let mut st_off = AffineState::with_slices_carrier(&g, 1, slices, Carrier::Q32);
    let layer_off = st_off.layer_mut(0).expect("layer");
    let refs: Vec<AffineCol<'_>> = cols
        .iter()
        .map(|[c, z, b, a]| (c.as_slice(), z.as_slice(), b.as_slice(), a.as_slice()))
        .collect();
    let (out_off, _raw_off) = delta_heads_cols_impl(
        layer_off,
        &g,
        &p,
        &refs,
        eps,
        true,
        None,
        false,
        Carrier::Q32,
    )
    .expect("per-step");
    let (words_off, scales_off) = layer_state_words(layer_off, &g, slices);

    let mut st_o = AffineState::with_slices_carrier(&g, 1, slices, Carrier::Q32);
    let layer_o = st_o.layer_mut(0).expect("oracle layer");
    let step =
        |s: &mut [i64], c: &mut i64, q: &[i64], k: &[i64], v: &[i64], decay: i64, beta: i64| {
            delta_head_step(s, c, q, k, v, decay, beta)
        };
    let mut out_oracle = Vec::with_capacity(cols.len());
    for [conv, z, b, a] in &cols {
        out_oracle.push(
            delta_heads_custom(layer_o, &g, &p, conv, z, b, a, eps, true, step).expect("oracle"),
        );
    }
    let (words_oracle, scales_oracle) = layer_state_words(layer_o, &g, slices);

    assert_eq!(
        out_off, out_oracle,
        "flag-off outputs differ from the oracle"
    );
    assert_eq!(
        words_off, words_oracle,
        "flag-off state differs from the oracle"
    );
    assert_eq!(
        scales_off, scales_oracle,
        "flag-off deferred decay differs from the oracle"
    );
}

/// f32 carrier tracks the Q32 per-step oracle within acceptable numerical tolerance.
#[test]
#[allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    reason = "test harnesses comparing f32 vs q32 parity and formatting stats"
)]
fn f32_carrier_tracks_q32_per_step_oracle() {
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 2,
        n_v_heads: 4,
        head_k: 128,
        head_v: 128,
    };
    let slices = 1usize;
    let eps = 4295i64;
    let (p, cols) = random_affine_block(&g, 3, 0x0f0f_0f0f_1234_5678);

    let mut st_f32 = AffineState::with_slices_carrier(&g, 1, slices, Carrier::F32);
    let layer_f32 = st_f32.layer_mut(0).expect("layer");
    let refs: Vec<AffineCol<'_>> = cols
        .iter()
        .map(|[c, z, b, a]| (c.as_slice(), z.as_slice(), b.as_slice(), a.as_slice()))
        .collect();
    let (out_f32, _raw_f32) = delta_heads_cols_impl(
        layer_f32,
        &g,
        &p,
        &refs,
        eps,
        true,
        None,
        false,
        Carrier::F32,
    )
    .expect("f32 step");

    let mut st_o = AffineState::with_slices_carrier(&g, 1, slices, Carrier::Q32);
    let layer_o = st_o.layer_mut(0).expect("oracle layer");
    let step =
        |s: &mut [i64], c: &mut i64, q: &[i64], k: &[i64], v: &[i64], decay: i64, beta: i64| {
            delta_head_step(s, c, q, k, v, decay, beta)
        };
    let mut out_oracle = Vec::with_capacity(cols.len());
    for [conv, z, b, a] in &cols {
        out_oracle.push(
            delta_heads_custom(layer_o, &g, &p, conv, z, b, a, eps, true, step).expect("oracle"),
        );
    }

    let mut max_abs_diff = 0.0f64;
    let mut max_rel_diff = 0.0f64;
    for (col_f32, col_q32) in out_f32.iter().zip(&out_oracle) {
        for (&yf, &yq) in col_f32.iter().zip(col_q32) {
            let ff = (yf as f64) / 4_294_967_296.0;
            let fq = (yq as f64) / 4_294_967_296.0;
            let diff = (ff - fq).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
            let denom = ff.abs().max(fq.abs());
            if denom > 1e-6 {
                let rel = diff / denom;
                if rel > max_rel_diff {
                    max_rel_diff = rel;
                }
            }
        }
    }
    println!("f32_carrier vs Q32 oracle: max |Δo| = {max_abs_diff:.6e}, max relative diff = {max_rel_diff:.6e}");
    assert!(
        max_abs_diff < 1e-3,
        "max absolute readout deviation {max_abs_diff:e} exceeded 1e-3"
    );
    assert!(
        max_rel_diff < 1e-3,
        "max relative readout deviation {max_rel_diff:e} exceeded 1e-3"
    );
}

/// `chunk_matches_oracle` at the wiring: a `CHUNK_MAX + 6`-column block run through the chunked
/// stage-1 path agrees with the per-step path on every readout and on the real exit state within
/// the epic's `ε/(1 − α)` bound (`Nous.Seq.ChunkedRounding.chunked_discrepancy_uniform_bound`).
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the test builds a random block, runs both paths and compares them"
)]
fn exact_scan_on_agrees_with_per_step_within_bound() {
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 2,
        n_v_heads: 4,
        head_k: 128,
        head_v: 128,
    };
    let slices = 1usize;
    let eps = 4295i64;
    let (p, cols) = random_affine_block(&g, CHUNK_MAX + 6, 0x9e37_79b9_7f4a_7c15);
    let refs: Vec<AffineCol<'_>> = cols
        .iter()
        .map(|[c, z, b, a]| (c.as_slice(), z.as_slice(), b.as_slice(), a.as_slice()))
        .collect();

    let run = |chunked: bool| {
        let mut st = AffineState::with_slices_carrier(&g, 1, slices, Carrier::Q32);
        let layer = st.layer_mut(0).expect("layer");
        let (out, raw) =
            delta_heads_cols_impl(layer, &g, &p, &refs, eps, true, None, chunked, Carrier::Q32)
                .expect("wiring step");
        let (words, scales) = layer_state_words(layer, &g, slices);
        (out, raw, words, scales)
    };
    let (_out_off, raw_off, words_off, scales_off) = run(false);
    let (_out_on, raw_on, words_on, scales_on) = run(true);

    // The chunked path stores the real state at `c = 1`.
    assert!(
        scales_on.iter().all(|&(_, c)| c == 1i64 << 32),
        "chunked exit decay is not 1.0"
    );
    let one = 1i64 << 32;
    let sp = fixed_softplus(0).expect("softplus(0)");
    let decay = fixed_exp(mul_q32(p.a[0], sp).expect("ssm_a · softplus")).expect("exp");
    assert!(decay > 0 && decay < one, "decay {decay} outside (0, 1)");
    let bound = one / (one - decay);

    let mut dev_out = 0i64;
    for (a, b) in raw_on.iter().zip(&raw_off) {
        dev_out = dev_out.max((a - b).abs());
    }
    let mut dev_state = 0i64;
    for (h, &(_, c_off)) in scales_off.iter().enumerate() {
        let base = h * g.head_k * g.head_v;
        for i in base..base + g.head_k * g.head_v {
            let real_off = mul_q32(c_off, words_off[i]).expect("oracle real state");
            dev_state = dev_state.max((words_on[i] - real_off).abs());
        }
    }
    println!(
        "exact-scan wiring deviation: outputs {dev_out} LSB, state {dev_state} LSB, bound {bound} \
         LSB (alpha = {decay} / 2^32, {} columns)",
        cols.len()
    );
    assert!(
        dev_out <= bound,
        "output deviation {dev_out} LSB exceeds the {bound} LSB bound"
    );
    assert!(
        dev_state <= bound,
        "state deviation {dev_state} LSB exceeds the {bound} LSB bound"
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the test runs the same block at several slice counts and compares"
)]
fn exact_scan_chunked_is_slice_separable() {
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 2,
        n_v_heads: 4,
        head_k: 128,
        head_v: 128,
    };
    let eps = 4295i64;
    let (p, cols) = random_affine_block(&g, 5, 0x1357_9bdf_0246_8ace);
    let refs: Vec<AffineCol<'_>> = cols
        .iter()
        .map(|[c, z, b, a]| (c.as_slice(), z.as_slice(), b.as_slice(), a.as_slice()))
        .collect();
    let run = |slices: usize| {
        let mut st = AffineState::with_slices_carrier(&g, 1, slices, Carrier::Q32);
        let layer = st.layer_mut(0).expect("layer");
        let (out, _raw) =
            delta_heads_cols_impl(layer, &g, &p, &refs, eps, true, None, true, Carrier::Q32)
                .expect("chunked step");
        let (words, scales) = layer_state_words(layer, &g, slices);
        assert!(
            scales.iter().all(|&(_, c)| c == 1i64 << 32),
            "chunked exit decay is not 1.0 at {slices} slices"
        );
        (out, words)
    };
    let (out1, words1) = run(1);
    for slices in [2usize, 4] {
        let (out, words) = run(slices);
        assert_eq!(out1, out, "chunked outputs differ at {slices} slices");
        assert_eq!(words1, words, "chunked state differs at {slices} slices");
    }
}

#[test]
#[ignore = "microbench: cargo test --release -- --ignored bench_delta_heads_50_steps"]
#[allow(
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    reason = "microbench test harness"
)]
fn bench_delta_heads_50_steps() {
    use std::time::Instant;
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 16,
        n_v_heads: 16,
        head_k: 128,
        head_v: 128,
    };
    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let conv = vec![123_456_789_i64; 2 * key_dim + val_dim];
    let z = vec![1_122_334_455_i64; val_dim];
    let b = vec![556_677_889_i64; g.n_v_heads];
    let a = vec![998_877_665_i64; g.n_v_heads];
    let eps = 4295i64;

    let p = AffineVectors {
        conv: vec![0; g.conv_kernel * (2 * key_dim + val_dim)],
        dt: vec![1_000_000_i64; g.n_v_heads],
        a: vec![-10_000_000_i64; g.n_v_heads],
        norm: vec![1i64 << 32; g.head_v],
    };

    let n_steps = 50;

    // 1. Measure v0 (today's baseline)
    let mut st_v0 = AffineState::with_slices_carrier(&g, 1, 1, Carrier::Q32);
    let layer_v0 = st_v0.layer_mut(0).unwrap();
    let step_v0 =
        |s: &mut [i64], _c: &mut i64, q: &[i64], k: &[i64], v: &[i64], decay: i64, beta: i64| {
            delta_head_step_v0(s, q, k, v, decay, beta)
        };
    for _ in 0..5 {
        let _ =
            delta_heads_custom(layer_v0, &g, &p, &conv, &z, &b, &a, eps, true, step_v0).unwrap();
    }
    let mut times_v0 = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let t0 = Instant::now();
        let _ =
            delta_heads_custom(layer_v0, &g, &p, &conv, &z, &b, &a, eps, true, step_v0).unwrap();
        times_v0.push(t0.elapsed().as_micros());
    }
    times_v0.sort_unstable();
    let median_v0 = times_v0[n_steps / 2];

    // 2. Measure ref (new scalar reference)
    let mut st_ref = AffineState::with_slices_carrier(&g, 1, 1, Carrier::Q32);
    let layer_ref = st_ref.layer_mut(0).unwrap();
    let step_ref =
        |s: &mut [i64], c: &mut i64, q: &[i64], k: &[i64], v: &[i64], decay: i64, beta: i64| {
            delta_head_step_ref(s, c, q, k, v, decay, beta)
        };
    for _ in 0..5 {
        let _ =
            delta_heads_custom(layer_ref, &g, &p, &conv, &z, &b, &a, eps, true, step_ref).unwrap();
    }
    let mut times_ref = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let t0 = Instant::now();
        let _ =
            delta_heads_custom(layer_ref, &g, &p, &conv, &z, &b, &a, eps, true, step_ref).unwrap();
        times_ref.push(t0.elapsed().as_micros());
    }
    times_ref.sort_unstable();
    let median_ref = times_ref[n_steps / 2];

    // 3. Measure fast (AVX2 SIMD)
    let mut st_fast = AffineState::with_slices_carrier(&g, 1, head_slices(g.head_v), Carrier::Q32);
    let layer_fast = st_fast.layer_mut(0).unwrap();
    for _ in 0..5 {
        let _ = delta_heads(layer_fast, &g, &p, &conv, &z, &b, &a, eps, true).unwrap();
    }
    let mut times_fast = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let t0 = Instant::now();
        let _ = delta_heads(layer_fast, &g, &p, &conv, &z, &b, &a, eps, true).unwrap();
        times_fast.push(t0.elapsed().as_micros());
    }
    times_fast.sort_unstable();
    let median_fast = times_fast[n_steps / 2];

    println!(
        "MICROBENCH delta_heads (16 heads pooled, 50 steps):\n  v0 (baseline): {median_v0} µs\n  ref (scalar):   {median_ref} µs\n  fast (AVX2):    {median_fast} µs"
    );
}

#[test]
#[ignore = "timing benchmark: cargo test --release --lib bench_delta_heads_cols_f32_vs_q32 -- --ignored --nocapture"]
#[allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::uninlined_format_args,
    reason = "timing benchmark harness comparing Q32 vs F32 full DeltaNet stages"
)]
fn bench_delta_heads_cols_f32_vs_q32() {
    use std::time::Instant;
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 16,
        n_v_heads: 16,
        head_k: 128,
        head_v: 128,
    };
    let k = 200; // prefill 200 tokens
    let eps = 4295i64;
    let slices = head_slices(g.head_v);

    let (p, cols) = random_affine_block(&g, k, 0x1234_5678_9ABC_DEF0);
    let col_refs: Vec<AffineCol<'_>> = cols
        .iter()
        .map(|[c, z, b, a]| (c.as_slice(), z.as_slice(), b.as_slice(), a.as_slice()))
        .collect();

    let runs = 5;

    // Warmup Q32
    let mut st_q32 = AffineState::with_slices_carrier(&g, 1, slices, Carrier::Q32);
    let layer_q32 = st_q32.layer_mut(0).unwrap();
    let _ = delta_heads_cols_impl(
        layer_q32,
        &g,
        &p,
        &col_refs,
        eps,
        false,
        None,
        false,
        Carrier::Q32,
    )
    .unwrap();

    let mut best_q32_us = f64::INFINITY;
    for run in 0..runs {
        let mut st = AffineState::with_slices_carrier(&g, 1, slices, Carrier::Q32);
        let layer = st.layer_mut(0).unwrap();
        let t0 = Instant::now();
        let (out, _) = delta_heads_cols_impl(
            layer,
            &g,
            &p,
            &col_refs,
            eps,
            false,
            None,
            false,
            Carrier::Q32,
        )
        .unwrap();
        let dur_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
        std::hint::black_box(out);
        println!(
            "  Q32 run {run}: {dur_us:.2} µs ({:.3} ms)",
            dur_us / 1000.0
        );
        if dur_us < best_q32_us {
            best_q32_us = dur_us;
        }
    }

    // Warmup F32
    let mut st_f32 = AffineState::with_slices_carrier(&g, 1, slices, Carrier::F32);
    let layer_f32 = st_f32.layer_mut(0).unwrap();
    let _ = delta_heads_cols_impl(
        layer_f32,
        &g,
        &p,
        &col_refs,
        eps,
        false,
        None,
        false,
        Carrier::F32,
    )
    .unwrap();

    let mut best_f32_us = f64::INFINITY;
    for run in 0..runs {
        let mut st = AffineState::with_slices_carrier(&g, 1, slices, Carrier::F32);
        let layer = st.layer_mut(0).unwrap();
        let t0 = Instant::now();
        let (out, _) = delta_heads_cols_impl(
            layer,
            &g,
            &p,
            &col_refs,
            eps,
            false,
            None,
            false,
            Carrier::F32,
        )
        .unwrap();
        let dur_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
        std::hint::black_box(out);
        println!(
            "  F32 run {run}: {dur_us:.2} µs ({:.3} ms)",
            dur_us / 1000.0
        );
        if dur_us < best_f32_us {
            best_f32_us = dur_us;
        }
    }

    let speedup = best_q32_us / best_f32_us;
    println!(
        "\n=== DeltaNet Stages (0, 0.5, 1, 2) Timing (k={k} tokens, single-thread, best of {runs}) ==="
    );
    println!("  Q32.32: {:.3} ms", best_q32_us / 1000.0);
    println!("  F32:    {:.3} ms", best_f32_us / 1000.0);
    println!("  Speedup: {speedup:.2}x");
}

#[test]
#[allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args,
    reason = "test harnesses comparing f32 vs q32 parity and formatting stats"
)]
fn f32_conv_silu_tracks_q32_oracle() {
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 2,
        n_v_heads: 4,
        head_k: 16,
        head_v: 32,
    };
    let channels = 2 * g.n_k_heads * g.head_k + g.n_v_heads * g.head_v;
    let mut rng = 0x5a5a_5a5a_3c3c_3c3c_u64;
    let mut word = |bound: i64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng.cast_signed() % bound
    };
    let weights: Vec<i64> = (0..g.conv_kernel * channels)
        .map(|_| word(1 << 30))
        .collect();
    let k = 10;
    let xs: Vec<Vec<i64>> = (0..k)
        .map(|_| (0..channels).map(|_| word(1 << 30)).collect())
        .collect();
    let col_refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();

    let mut st_q32 = AffineState::with_slices_carrier(&g, 1, 1, Carrier::Q32);
    let layer_q32 = st_q32.layer_mut(0).expect("layer");
    let mut st_f32 = AffineState::with_slices_carrier(&g, 1, 1, Carrier::F32);
    let layer_f32 = st_f32.layer_mut(0).expect("layer");

    let out_q32 =
        conv_silu_cols_carrier(layer_q32, &weights, g.conv_kernel, &col_refs, Carrier::Q32)
            .expect("q32 conv");
    let out_f32 =
        conv_silu_cols_carrier(layer_f32, &weights, g.conv_kernel, &col_refs, Carrier::F32)
            .expect("f32 conv");

    let mut max_abs_diff = 0.0f64;
    let mut max_rel_diff = 0.0f64;
    for (col_f, col_q) in out_f32.iter().zip(&out_q32) {
        for (&yf, &yq) in col_f.iter().zip(col_q) {
            let ff = (yf as f64) / 4_294_967_296.0;
            let fq = (yq as f64) / 4_294_967_296.0;
            let diff = (ff - fq).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
            let denom = ff.abs().max(fq.abs());
            if denom > 1e-6 {
                let rel = diff / denom;
                if rel > max_rel_diff {
                    max_rel_diff = rel;
                }
            }
        }
    }
    println!("f32 conv+silu vs Q32 oracle: max |Δo| = {max_abs_diff:.6e}, max relative diff = {max_rel_diff:.6e}");
    assert!(
        max_abs_diff < 1e-3,
        "max absolute conv deviation {max_abs_diff:e} exceeded 1e-3"
    );
    assert!(
        max_rel_diff < 1e-3,
        "max relative conv deviation {max_rel_diff:e} exceeded 1e-3"
    );
}

#[test]
#[ignore = "timing benchmark: cargo test --release --lib bench_conv_silu_f32_prefill_and_decode -- --ignored --nocapture"]
#[allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    reason = "timing benchmark harness comparing Q32 vs F32 conv_silu"
)]
fn bench_conv_silu_f32_prefill_and_decode() {
    use std::time::Instant;
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 16,
        n_v_heads: 16,
        head_k: 128,
        head_v: 128,
    };
    let channels = 2 * g.n_k_heads * g.head_k + g.n_v_heads * g.head_v;
    let mut rng = 0x5a5a_5a5a_3c3c_3c3c_u64;
    let mut word = |bound: i64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng.cast_signed() % bound
    };
    let weights: Vec<i64> = (0..g.conv_kernel * channels)
        .map(|_| word(1 << 30))
        .collect();

    // 1. Decode benchmark (k = 1, 200 reps)
    let decode_reps = 200;
    let decode_xs: Vec<Vec<i64>> = (0..decode_reps)
        .map(|_| (0..channels).map(|_| word(1 << 30)).collect())
        .collect();

    let runs = 5;

    // Decode Q32
    let mut best_decode_q32_us = f64::INFINITY;
    for _ in 0..runs {
        let mut st = AffineState::with_slices_carrier(&g, 1, 1, Carrier::Q32);
        let layer = st.layer_mut(0).unwrap();
        let t0 = Instant::now();
        for x in &decode_xs {
            let out = conv_silu_cols_carrier(
                layer,
                &weights,
                g.conv_kernel,
                &[x.as_slice()],
                Carrier::Q32,
            )
            .unwrap();
            std::hint::black_box(out);
        }
        let dur_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / (decode_reps as f64);
        if dur_us < best_decode_q32_us {
            best_decode_q32_us = dur_us;
        }
    }

    // Decode F32
    let mut best_decode_f32_us = f64::INFINITY;
    for _ in 0..runs {
        let mut st = AffineState::with_slices_carrier(&g, 1, 1, Carrier::F32);
        let layer = st.layer_mut(0).unwrap();
        let t0 = Instant::now();
        for x in &decode_xs {
            let out = conv_silu_cols_carrier(
                layer,
                &weights,
                g.conv_kernel,
                &[x.as_slice()],
                Carrier::F32,
            )
            .unwrap();
            std::hint::black_box(out);
        }
        let dur_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / (decode_reps as f64);
        if dur_us < best_decode_f32_us {
            best_decode_f32_us = dur_us;
        }
    }

    // 2. Prefill benchmark (k = 200)
    let k_prefill = 200;
    let prefill_xs: Vec<Vec<i64>> = (0..k_prefill)
        .map(|_| (0..channels).map(|_| word(1 << 30)).collect())
        .collect();
    let prefill_refs: Vec<&[i64]> = prefill_xs.iter().map(Vec::as_slice).collect();

    // Prefill Q32
    let mut best_prefill_q32_us = f64::INFINITY;
    for _ in 0..runs {
        let mut st = AffineState::with_slices_carrier(&g, 1, 1, Carrier::Q32);
        let layer = st.layer_mut(0).unwrap();
        let t0 = Instant::now();
        let out =
            conv_silu_cols_carrier(layer, &weights, g.conv_kernel, &prefill_refs, Carrier::Q32)
                .unwrap();
        std::hint::black_box(out);
        let dur_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
        if dur_us < best_prefill_q32_us {
            best_prefill_q32_us = dur_us;
        }
    }

    // Prefill F32
    let mut best_prefill_f32_us = f64::INFINITY;
    for _ in 0..runs {
        let mut st = AffineState::with_slices_carrier(&g, 1, 1, Carrier::F32);
        let layer = st.layer_mut(0).unwrap();
        let t0 = Instant::now();
        let out =
            conv_silu_cols_carrier(layer, &weights, g.conv_kernel, &prefill_refs, Carrier::F32)
                .unwrap();
        std::hint::black_box(out);
        let dur_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
        if dur_us < best_prefill_f32_us {
            best_prefill_f32_us = dur_us;
        }
    }

    println!("\n=== Conv+SiLU Timing (channels={channels}, kernel=4, best of {runs}) ===");
    println!(
        "  Decode (k=1) Q32: {:.2} µs/token ({:.4} ms/token)",
        best_decode_q32_us,
        best_decode_q32_us / 1000.0
    );
    println!(
        "  Decode (k=1) F32: {:.2} µs/token ({:.4} ms/token)",
        best_decode_f32_us,
        best_decode_f32_us / 1000.0
    );
    println!(
        "  Decode Speedup:   {:.2}x",
        best_decode_q32_us / best_decode_f32_us
    );
    println!(
        "  Prefill (k=200) Q32: {:.2} µs/token ({:.3} ms total)",
        best_prefill_q32_us / (k_prefill as f64),
        best_prefill_q32_us / 1000.0
    );
    println!(
        "  Prefill (k=200) F32: {:.2} µs/token ({:.3} ms total)",
        best_prefill_f32_us / (k_prefill as f64),
        best_prefill_f32_us / 1000.0
    );
    println!(
        "  Prefill Speedup:     {:.2}x",
        best_prefill_q32_us / best_prefill_f32_us
    );
}

#[test]
#[ignore = "timing benchmark: cargo test --release --lib bench_affine_heads_f32_decode -- --ignored --nocapture"]
#[allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    reason = "timing benchmark harness comparing Q32 vs F32 decode heads"
)]
fn bench_affine_heads_f32_decode() {
    use std::time::Instant;
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 16,
        n_v_heads: 16,
        head_k: 128,
        head_v: 128,
    };
    let eps = 4295i64;
    let slices = head_slices(g.head_v);
    let (p, cols) = random_affine_block(&g, 1, 0x1234_5678_9ABC_DEF0);
    let col_refs: Vec<AffineCol<'_>> = cols
        .iter()
        .map(|[c, z, b, a]| (c.as_slice(), z.as_slice(), b.as_slice(), a.as_slice()))
        .collect();

    let runs = 5;
    let reps = 200;

    let mut best_q32_us = f64::INFINITY;
    for _ in 0..runs {
        let mut st = AffineState::with_slices_carrier(&g, 1, slices, Carrier::Q32);
        let layer = st.layer_mut(0).unwrap();
        let t0 = Instant::now();
        for _ in 0..reps {
            let (out, _) = delta_heads_cols_impl(
                layer,
                &g,
                &p,
                &col_refs,
                eps,
                true,
                None,
                false,
                Carrier::Q32,
            )
            .unwrap();
            std::hint::black_box(out);
        }
        let dur_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / (reps as f64);
        if dur_us < best_q32_us {
            best_q32_us = dur_us;
        }
    }

    let mut best_f32_us = f64::INFINITY;
    for _ in 0..runs {
        let mut st = AffineState::with_slices_carrier(&g, 1, slices, Carrier::F32);
        let layer = st.layer_mut(0).unwrap();
        let t0 = Instant::now();
        for _ in 0..reps {
            let (out, _) = delta_heads_cols_impl(
                layer,
                &g,
                &p,
                &col_refs,
                eps,
                true,
                None,
                false,
                Carrier::F32,
            )
            .unwrap();
            std::hint::black_box(out);
        }
        let dur_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / (reps as f64);
        if dur_us < best_f32_us {
            best_f32_us = dur_us;
        }
    }

    // Fused conv + DeltaNet decode
    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let channels = 2 * key_dim + val_dim;
    let mut rng = 0x5a5a_1234_u64;
    let mut word = |bound: i64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as i64) % bound
    };
    let raw_input: Vec<i64> = (0..channels).map(|_| word(1 << 30)).collect();
    let z_in: Vec<i64> = (0..val_dim).map(|_| word(1 << 30)).collect();
    let b_in: Vec<i64> = (0..g.n_v_heads).map(|_| word(1 << 30).abs()).collect();
    let a_in: Vec<i64> = vec![0i64; g.n_v_heads];

    let mut best_fused_f32_us = f64::INFINITY;
    for _ in 0..runs {
        let mut st = AffineState::with_slices_carrier(&g, 1, slices, Carrier::F32);
        let layer = st.layer_mut(0).unwrap();
        let t0 = Instant::now();
        for _ in 0..reps {
            let conv_staged =
                conv_silu_cols(layer, &p.conv, g.conv_kernel, &[raw_input.as_slice()]).unwrap();
            let col_fused: Vec<AffineCol<'_>> = vec![(
                conv_staged[0].as_slice(),
                z_in.as_slice(),
                b_in.as_slice(),
                a_in.as_slice(),
            )];
            let (out, _) = delta_heads_cols_impl(
                layer,
                &g,
                &p,
                &col_fused,
                eps,
                true,
                None,
                false,
                Carrier::F32,
            )
            .unwrap();
            std::hint::black_box(out);
        }
        let dur_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / (reps as f64);
        if dur_us < best_fused_f32_us {
            best_fused_f32_us = dur_us;
        }
    }

    println!("\n=== Affine Heads Decode Timing (k=1, best of {runs}) ===");
    println!(
        "  Q32 (per layer): {:.2} µs/token ({:.4} ms/token for 18 layers)",
        best_q32_us,
        best_q32_us * 18.0 / 1000.0
    );
    println!(
        "  F32 unfused (per layer): {:.2} µs/token ({:.4} ms/token for 18 layers)",
        best_f32_us,
        best_f32_us * 18.0 / 1000.0
    );
    println!(
        "  F32 fused conv+DeltaNet (per layer): {:.2} µs/token ({:.4} ms/token for 18 layers)",
        best_fused_f32_us,
        best_fused_f32_us * 18.0 / 1000.0
    );
    println!("  Speedup vs Q32:  {:.2}x", best_q32_us / best_fused_f32_us);
}

#[test]
#[allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test comparing Q32 vs F32 fused conv+deltanet"
)]
fn test_fused_conv_deltanet_tracks_oracle_at_k1() {
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 16,
        n_v_heads: 16,
        head_k: 128,
        head_v: 128,
    };
    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let ch = 2 * key_dim + val_dim;
    let slices = 1usize;
    let eps = 4295i64;

    let mut rng = 0xbeef_cafe_1234_5678_u64;
    let mut word = |bound: i64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as i64) % bound
    };

    let sp = fixed_softplus(0).expect("softplus(0)");
    let target = 0.98f64;
    let ssm_a = (target.ln() * (1u64 << 32) as f64 * (1u64 << 32) as f64 / sp as f64) as i64;

    let p = AffineVectors {
        conv: (0..g.conv_kernel * ch).map(|_| word(1 << 30)).collect(),
        dt: vec![0i64; g.n_v_heads],
        a: vec![ssm_a; g.n_v_heads],
        norm: vec![1i64 << 32; g.head_v],
    };

    let raw_input: Vec<i64> = (0..ch).map(|_| word(1 << 30)).collect();
    let z: Vec<i64> = (0..val_dim).map(|_| word(1 << 30)).collect();
    let b: Vec<i64> = (0..g.n_v_heads).map(|_| word(1 << 30).abs()).collect();
    let a: Vec<i64> = vec![0i64; g.n_v_heads];

    let mut st_q32 = AffineState::with_slices_carrier(&g, 1, slices, Carrier::Q32);
    let layer_q32 = st_q32.layer_mut(0).expect("layer");
    let conv_q32 = conv_silu_cols(layer_q32, &p.conv, g.conv_kernel, &[raw_input.as_slice()])
        .expect("q32 conv");
    let cols_q32: Vec<AffineCol<'_>> = vec![(
        conv_q32[0].as_slice(),
        z.as_slice(),
        b.as_slice(),
        a.as_slice(),
    )];
    let out_q32 =
        delta_heads_cols_carrier(layer_q32, &g, &p, &cols_q32, eps, true, None, Carrier::Q32)
            .expect("q32 heads");

    let mut st_f32 = AffineState::with_slices_carrier(&g, 1, slices, Carrier::F32);
    let layer_f32 = st_f32.layer_mut(0).expect("layer");
    let conv_f32 = conv_silu_cols(layer_f32, &p.conv, g.conv_kernel, &[raw_input.as_slice()])
        .expect("f32 conv");
    let cols_f32: Vec<AffineCol<'_>> = vec![(
        conv_f32[0].as_slice(),
        z.as_slice(),
        b.as_slice(),
        a.as_slice(),
    )];
    let out_f32 =
        delta_heads_cols_carrier(layer_f32, &g, &p, &cols_f32, eps, true, None, Carrier::F32)
            .expect("f32 heads");

    let mut max_abs_diff = 0.0f64;
    for (yf, yq) in out_f32[0].iter().zip(&out_q32[0]) {
        let ff = (*yf as f64) / 4_294_967_296.0;
        let fq = (*yq as f64) / 4_294_967_296.0;
        let diff = (ff - fq).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    println!("fused conv+deltanet k=1 vs Q32 oracle: max |Δo| = {max_abs_diff:.6e}");
    assert!(max_abs_diff < 1e-3, "max abs diff {max_abs_diff:e} >= 1e-3");
}

#[test]
#[allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test comparing Q32 vs F32 fused conv+deltanet"
)]
fn test_fused_conv_deltanet_tracks_oracle_at_k5() {
    let g = AffineGeom {
        conv_kernel: 4,
        n_k_heads: 16,
        n_v_heads: 16,
        head_k: 128,
        head_v: 128,
    };
    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let ch = 2 * key_dim + val_dim;
    let slices = 1usize;
    let eps = 4295i64;
    let k = 5;

    let mut rng = 0xcafe_beef_8765_4321_u64;
    let mut word = |bound: i64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as i64) % bound
    };

    let sp = fixed_softplus(0).expect("softplus(0)");
    let target = 0.98f64;
    let ssm_a = (target.ln() * (1u64 << 32) as f64 * (1u64 << 32) as f64 / sp as f64) as i64;

    let p = AffineVectors {
        conv: (0..g.conv_kernel * ch).map(|_| word(1 << 30)).collect(),
        dt: vec![0i64; g.n_v_heads],
        a: vec![ssm_a; g.n_v_heads],
        norm: vec![1i64 << 32; g.head_v],
    };

    let raw_inputs: Vec<Vec<i64>> = (0..k)
        .map(|_| (0..ch).map(|_| word(1 << 30)).collect())
        .collect();
    let raw_refs: Vec<&[i64]> = raw_inputs.iter().map(Vec::as_slice).collect();

    let z: Vec<Vec<i64>> = (0..k)
        .map(|_| (0..val_dim).map(|_| word(1 << 30)).collect())
        .collect();
    let b: Vec<Vec<i64>> = (0..k)
        .map(|_| (0..g.n_v_heads).map(|_| word(1 << 30).abs()).collect())
        .collect();
    let a: Vec<Vec<i64>> = (0..k).map(|_| vec![0i64; g.n_v_heads]).collect();

    let mut st_q32 = AffineState::with_slices_carrier(&g, 1, slices, Carrier::Q32);
    let layer_q32 = st_q32.layer_mut(0).expect("layer");
    let conv_q32 = conv_silu_cols(layer_q32, &p.conv, g.conv_kernel, &raw_refs).expect("q32 conv");
    let cols_q32: Vec<AffineCol<'_>> = (0..k)
        .map(|c| {
            (
                conv_q32[c].as_slice(),
                z[c].as_slice(),
                b[c].as_slice(),
                a[c].as_slice(),
            )
        })
        .collect();
    let out_q32 =
        delta_heads_cols_carrier(layer_q32, &g, &p, &cols_q32, eps, true, None, Carrier::Q32)
            .expect("q32 heads");

    let mut st_f32 = AffineState::with_slices_carrier(&g, 1, slices, Carrier::F32);
    let layer_f32 = st_f32.layer_mut(0).expect("layer");
    let conv_f32 = conv_silu_cols(layer_f32, &p.conv, g.conv_kernel, &raw_refs).expect("f32 conv");
    let cols_f32: Vec<AffineCol<'_>> = (0..k)
        .map(|c| {
            (
                conv_f32[c].as_slice(),
                z[c].as_slice(),
                b[c].as_slice(),
                a[c].as_slice(),
            )
        })
        .collect();
    let out_f32 =
        delta_heads_cols_carrier(layer_f32, &g, &p, &cols_f32, eps, true, None, Carrier::F32)
            .expect("f32 heads");

    let mut max_abs_diff = 0.0f64;
    for c in 0..k {
        for (yf, yq) in out_f32[c].iter().zip(&out_q32[c]) {
            let ff = (*yf as f64) / 4_294_967_296.0;
            let fq = (*yq as f64) / 4_294_967_296.0;
            let diff = (ff - fq).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
        }
    }
    println!("fused conv+deltanet k=5 vs Q32 oracle: max |Δo| = {max_abs_diff:.6e}");
    assert!(max_abs_diff < 1e-3, "max abs diff {max_abs_diff:e} >= 1e-3");
}
