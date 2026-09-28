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
    let mut state = AffineState::with_slices(&g, 1, 1);
    let layer = state.layer_mut(0).expect("constructed layer");
    let channels = 2 * g.n_k_heads * g.head_k + g.n_v_heads * g.head_v;
    let scale = 1_i64 << 62;
    for r in 0..g.conv_kernel - 1 {
        let value = if r < 9 { scale } else { -scale };
        layer.conv[r * channels..(r + 1) * channels].fill(value);
    }
    let weights = vec![scale; g.conv_kernel * channels];
    let input = vec![-scale; channels];
    let old_window = layer.conv.clone();

    let result = conv_silu_cols(layer, &weights, g.conv_kernel, &[&input]);

    assert!(matches!(result, Err(StepError::Overflow)));
    assert_eq!(layer.conv, old_window);

    // The same cancellation is valid when every prefix remains within i128.
    let safe_scale = 1_i64 << 60;
    for r in 0..g.conv_kernel - 1 {
        let value = if r < 9 { safe_scale } else { -safe_scale };
        layer.conv[r * channels..(r + 1) * channels].fill(value);
    }
    let safe_weights = vec![safe_scale; g.conv_kernel * channels];
    let safe_input = vec![-safe_scale; channels];
    let result = conv_silu_cols(layer, &safe_weights, g.conv_kernel, &[&safe_input]);
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
    let mut state = AffineState::with_slices(&g, 1, 1);
    let layer = state.layer_mut(0).expect("constructed layer");
    let channels = 2 * g.n_k_heads * g.head_k + g.n_v_heads * g.head_v;
    layer.conv[0] = i64::MIN;
    layer.conv[channels] = i64::MAX;
    let mut weights = vec![0; g.conv_kernel * channels];
    weights[0] = i64::MIN;
    weights[1] = i64::MAX;
    weights[2] = 2;
    let mut input = vec![0; channels];
    input[0] = i64::MAX;
    let old_window = layer.conv.clone();

    let result = conv_silu_cols(layer, &weights, g.conv_kernel, &[&input]);

    assert!(matches!(result, Err(StepError::Overflow)));
    assert_eq!(layer.conv, old_window);
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
        let mut st = AffineState::with_slices(&g, 1, slices);
        let layer = st.layer_mut(0).expect("layer");
        let mut outs = Vec::new();
        for pass in &passes {
            let cols: Vec<AffineCol<'_>> = pass
                .iter()
                .map(|[c, z, b, a]| (c.as_slice(), z.as_slice(), b.as_slice(), a.as_slice()))
                .collect();
            outs.push(delta_heads_cols(layer, &g, &p, &cols, 1, parallel, None).expect("step"));
        }
        // Every head's state in whole-head layout, with each slice's `c`.
        let w = g.head_v / slices;
        let mut words = vec![0i64; g.n_v_heads * g.head_k * g.head_v];
        let mut scales = Vec::new();
        for (t, m) in layer.heads.iter().enumerate() {
            let head = m.lock().expect("head");
            let (h, sl) = (t / slices, t % slices);
            for i in 0..g.head_k {
                let at = h * g.head_k * g.head_v + i * g.head_v + sl * w;
                words[at..at + w].copy_from_slice(&head.s[i * w..(i + 1) * w]);
            }
            scales.push((h, head.c));
        }
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
    const QWEN35_08B_PATH: &str = "~/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf";
    if !std::path::Path::new(QWEN35_08B_PATH).exists() {
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

        let o_v0 = delta_head_step_v0(&mut s_v0, &q_tilde, &k_hat, &v, decay, beta)
            .expect("v0 step ok");
        let o_ref =
            delta_head_step_ref(&mut s_new, &mut c_new, &q_tilde, &k_hat, &v, decay, beta)
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
    F: FnMut(
            &mut [i64],
            &mut i64,
            &[i64],
            &[i64],
            &[i64],
            i64,
            i64,
        ) -> Result<Vec<i64>, StepError>
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
            let mut head_guard = layer.heads[h]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let AffineHead { s, c } = &mut *head_guard;
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
    if let Some((_, e)) = attn_heads(g.n_v_heads, g.head_v, &mut out, parallel, head) {
        return Err(e);
    }
    Ok(out)
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
    let eps = 1i64;

    let p = AffineVectors {
        conv: vec![0; g.conv_kernel * (2 * key_dim + val_dim)],
        dt: vec![1_000_000_i64; g.n_v_heads],
        a: vec![-10_000_000_i64; g.n_v_heads],
        norm: vec![1i64 << 32; g.head_v],
    };

    let n_steps = 50;

    // 1. Measure v0 (today's baseline)
    let mut st_v0 = AffineState::with_slices(&g, 1, 1);
    let layer_v0 = st_v0.layer_mut(0).unwrap();
    let step_v0 =
        |s: &mut [i64],
         _c: &mut i64,
         q: &[i64],
         k: &[i64],
         v: &[i64],
         decay: i64,
         beta: i64| { delta_head_step_v0(s, q, k, v, decay, beta) };
    for _ in 0..5 {
        let _ = delta_heads_custom(layer_v0, &g, &p, &conv, &z, &b, &a, eps, true, step_v0)
            .unwrap();
    }
    let mut times_v0 = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let t0 = Instant::now();
        let _ = delta_heads_custom(layer_v0, &g, &p, &conv, &z, &b, &a, eps, true, step_v0)
            .unwrap();
        times_v0.push(t0.elapsed().as_micros());
    }
    times_v0.sort_unstable();
    let median_v0 = times_v0[n_steps / 2];

    // 2. Measure ref (new scalar reference)
    let mut st_ref = AffineState::with_slices(&g, 1, 1);
    let layer_ref = st_ref.layer_mut(0).unwrap();
    let step_ref =
        |s: &mut [i64], c: &mut i64, q: &[i64], k: &[i64], v: &[i64], decay: i64, beta: i64| {
            delta_head_step_ref(s, c, q, k, v, decay, beta)
        };
    for _ in 0..5 {
        let _ = delta_heads_custom(layer_ref, &g, &p, &conv, &z, &b, &a, eps, true, step_ref)
            .unwrap();
    }
    let mut times_ref = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let t0 = Instant::now();
        let _ = delta_heads_custom(layer_ref, &g, &p, &conv, &z, &b, &a, eps, true, step_ref)
            .unwrap();
        times_ref.push(t0.elapsed().as_micros());
    }
    times_ref.sort_unstable();
    let median_ref = times_ref[n_steps / 2];

    // 3. Measure fast (AVX2 SIMD)
    let mut st_fast = AffineState::new(&g, 1);
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
