//! **The gated delta rule on the device** — the recurrence that is 48 of Qwen3.8-27B's 64 blocks
//! and the one mechanism in that model with no GPU kernel at all.
//!
//! It is not attention: there is no KV cache and no softmax. Each v-head carries a FIXED `[hd, hd]`
//! state that every token decays, corrects by a rank-1 update, and reads. The state does not grow
//! with context, which is why only the 16 full-attention layers need a KV tier.
//!
//! The reference is `bedrock::serve::model::deltanet_step`, itself transcribed from llama.cpp's
//! `ggml_compute_forward_gated_delta_net`. Per v-head, exactly:
//!
//! ```text
//!   M[j][i] *= exp(g)                        decay, one scalar per head
//!   d[j]     = (v[j] − dot(M[j], k)) · beta  the delta rule's correction
//!   M[j]    += d[j] · k                      rank-1 update
//!   o[j]     = dot(M[j], q) / sqrt(hd)       readout
//! ```
//!
//! **The mapping is one workgroup per HEAD and one thread per ROW `j`.** That is what makes this a
//! tractable kernel rather than a scan: within a token, row `j` is touched only by thread `j` —
//! decay, the `k` dot, the rank-1 update and the `q` dot all read and write `M[j]` alone. The only
//! shared data is `k` and `q`, staged once per head. The sequentiality of the delta rule is ACROSS
//! tokens, not within one, and decode feeds tokens one at a time regardless.
//!
//! Two passes over the row, not one, and the reason is the data dependence: `d[j]` needs the whole
//! `dot(M[j], k)` before the update can start, and the readout needs the UPDATED row. Fusing them
//! would compute the `q` dot against a half-updated state.
//!
//! **The state is stored TRANSPOSED** (`M[j][i] = S[i][j]`), as the reference stores it, so a row of
//! `M` is a column of `S` and every inner loop is a contiguous walk. Getting that backwards is
//! silent: the arithmetic still runs and the model still emits tokens.
//!
//! q/k heads are FEWER than v heads and the map is **modulo** (`kh = h % n_k_heads`), not divide —
//! head 17 reads q/k head 1, not head 5.

use super::*;

/// Bindings: 0 = `M` (f32, `n_v_heads·hd·hd`, read-write), 1 = `k`, 2 = `q` (f32, `n_k_heads·hd`,
/// already L2-normed), 3 = `v` (f32, `n_v_heads·hd`), 4 = `g` (f32, `n_v_heads`, the LOG decay),
/// 5 = `beta` (f32, `n_v_heads`, already sigmoid'd), 6 = `o` (f32, `n_v_heads·hd`, written).
/// Dispatch `(1, n_v_heads, 1)`.
///
/// **`off` = `(q, k, v)` element offsets into their bindings**, which is what lets a chain bind ONE
/// buffer three times instead of copying. The conv emits q, k and v fused in a single `[conv_dim]`
/// vector (q at 0, k at `key_dim`, v at `2·key_dim`) and Vulkan has no sub-buffer view here, so the
/// alternative is a copy stage — three transfers of the layer's whole activation, 48 times a token,
/// to move data that never left the device. Pass `(0, 0, 0)` with three distinct buffers for the
/// standalone form.
pub fn deltanet_rule_module(
    hd: u32,
    n_v_heads: u32,
    n_k_heads: u32,
    off: (u32, u32, u32),
) -> Vec<u32> {
    const WG: u32 = 256;
    let mut k = Kb::new();
    let mm = k.ssbo_f(0);
    let kk = k.ssbo_f(1);
    let qq = k.ssbo_f(2);
    let vv = k.ssbo_f(3);
    let gg = k.ssbo_f(4);
    let bb = k.ssbo_f(5);
    let oo = k.ssbo_f(6);
    let (shk, shkp) = k.shared_f32(hd);
    let (shq, shqp) = k.shared_f32(hd);

    let tid = k.gid_comp(0);
    let h = k.gid_comp(1);

    let c0 = k.cu(0);
    let chd = k.cu(hd);
    let cwg = k.cu(WG);
    let cnk = k.cu(n_k_heads);

    // q/k heads are fewer than v heads and the map is MODULO, not divide.
    let kh = k.umod(h, cnk);
    let hbase = k.imul(kh, chd);
    let vhead = k.imul(h, chd);
    let (cqo, cko, cvo) = (k.cu(off.0), k.cu(off.1), k.cu(off.2));
    let qbase = k.iadd(hbase, cqo);
    let kbase = k.iadd(hbase, cko);
    let vbase = k.iadd(vhead, cvo);
    // `o` is its own buffer and is NOT offset — the readout is the layer's product, not a window
    // into the fused activation.
    let obase = vhead;
    let chdhd = k.cu(hd * hd);
    let mbase = k.imul(h, chdhd);

    // one scalar per head: the decay is exp(g), the correction step is beta
    let gv = k.load_f(gg, h);
    let decay = k.ext1(GLSL_EXP, gv);
    let beta = k.load_f(bb, h);
    // `1/sqrt(hd)` is a compile-time constant — no reason to compute it per invocation
    let scale = k.cf(1.0f32 / (hd as f32).sqrt());

    // stage this head's k and q once; every row below reads all of both
    k.loop_step(tid, chd, cwg, |k, i| {
        let ki = k.iadd(kbase, i);
        let kvv = k.load_f(kk, ki);
        k.shared_store(shk, shkp, i, kvv);
        let qi = k.iadd(qbase, i);
        let qvv = k.load_f(qq, qi);
        k.shared_store(shq, shqp, i, qvv);
    });
    k.wg_barrier();

    k.loop_step(tid, chd, cwg, |k, j| {
        let jrow = k.imul(j, chd);
        let rowbase = k.iadd(mbase, jrow);

        // ── pass 1: decay the row in place, and dot it with k ──
        let acc = k.fvar_f();
        let z = k.cf(0.0);
        k.store_var(acc, z);
        k.loop_u(c0, chd, |k, i| {
            let idx = k.iadd(rowbase, i);
            let m0 = k.load_f(mm, idx);
            let m1 = k.fmul(m0, decay);
            k.store_f(mm, idx, m1);
            let kv = k.shared_load(shk, shkp, i);
            let p = k.fmul(m1, kv);
            let a0 = k.load_var_f(acc);
            let a1 = k.fadd(a0, p);
            k.store_var(acc, a1);
        });
        let dot_k = k.load_var_f(acc);
        let vj_i = k.iadd(vbase, j);
        let vj = k.load_f(vv, vj_i);
        let diff = k.fsub(vj, dot_k);
        let delta = k.fmul(diff, beta);

        // ── pass 2: rank-1 update, and dot the UPDATED row with q ──
        let acc2 = k.fvar_f();
        k.store_var(acc2, z);
        k.loop_u(c0, chd, |k, i| {
            let idx = k.iadd(rowbase, i);
            let m0 = k.load_f(mm, idx);
            let kv = k.shared_load(shk, shkp, i);
            let dk = k.fmul(delta, kv);
            let m1 = k.fadd(m0, dk);
            k.store_f(mm, idx, m1);
            let qv = k.shared_load(shq, shqp, i);
            let p = k.fmul(m1, qv);
            let a0 = k.load_var_f(acc2);
            let a1 = k.fadd(a0, p);
            k.store_var(acc2, a1);
        });
        let dot_q = k.load_var_f(acc2);
        let ov = k.fmul(dot_q, scale);
        let oi = k.iadd(obase, j);
        k.store_f(oo, oi, ov);
    });

    k.finish(WG)
}

/// [`deltanet_rule_module`] as bytes.
pub fn deltanet_rule_bytes(
    hd: u32,
    n_v_heads: u32,
    n_k_heads: u32,
    off: (u32, u32, u32),
) -> Vec<u8> {
    words_to_bytes(&deltanet_rule_module(hd, n_v_heads, n_k_heads, off))
}

/// **The causal depthwise conv + SiLU, and the conv history's shift, in one kernel.**
///
/// One thread per CHANNEL, and that is what makes the state shift safe to do here rather than in a
/// second pass: channel `ch` owns `state[ch·taps .. (ch+1)·taps]` outright, so the read-then-write
/// walk races nothing. Doing it as a separate dispatch would cost a second barrier per layer across
/// 48 layers for no correctness gain.
///
/// The conv weights are **channel-major** — `w[ch·d_conv + r]` — which is what BOTH the HF
/// `[conv_dim, 1, d_conv]` and the GGUF `[4, 6144]` layouts already are (ggml's `ne[0]` is the
/// fastest-varying dimension, so `[4, 6144]` is 6144 groups of 4, not 4 rows of 6144). A transposed
/// read is still a well-formed convolution over the wrong weights, and silent.
///
/// The tap window is `[state ... , this token]`: the newest sample is the LAST tap, and the history
/// then drops its oldest and appends this token's PRE-conv value.
///
/// Bindings: 0 = `qkv` (f32, `conv_dim`, the projection's output), 1 = `w` (f32,
/// `conv_dim·d_conv`), 2 = `state` (f32, `conv_dim·(d_conv−1)`, read-write), 3 = `y` (f32,
/// `conv_dim`, written). Dispatch ≥ `conv_dim`.
pub fn deltanet_conv_silu_module(conv_dim: u32, d_conv: u32) -> Vec<u32> {
    const LOCAL: u32 = 64;
    let taps = d_conv - 1;
    let mut k = Kb::new();
    let qkv = k.ssbo_f(0);
    let ww = k.ssbo_f(1);
    let st = k.ssbo_f(2);
    let yy = k.ssbo_f(3);

    let ch = k.gid_comp(0);
    let ccd = k.cu(conv_dim);
    let guard = k.ult(ch, ccd);
    k.iff(guard, |k| {
        let cdc = k.cu(d_conv);
        let ctaps = k.cu(taps);
        let kb = k.imul(ch, cdc);
        let sb = k.imul(ch, ctaps);
        let c0 = k.cu(0);

        let acc = k.fvar_f();
        let z = k.cf(0.0);
        k.store_var(acc, z);
        // the history taps, oldest first
        k.loop_u(c0, ctaps, |k, r| {
            let wi = k.iadd(kb, r);
            let si = k.iadd(sb, r);
            let p = {
                let a = k.load_f(ww, wi);
                let b = k.load_f(st, si);
                k.fmul(a, b)
            };
            let a0 = k.load_var_f(acc);
            let a1 = k.fadd(a0, p);
            k.store_var(acc, a1);
        });
        // and this token, which is the LAST tap
        let xv = k.load_f(qkv, ch);
        let wl = {
            let i = k.iadd(kb, ctaps);
            k.load_f(ww, i)
        };
        let p = k.fmul(wl, xv);
        let a0 = k.load_var_f(acc);
        let a = k.fadd(a0, p);

        // SiLU: a / (1 + exp(−a))
        let f = k.float;
        let na = k.e(OP_FNEG, f, &[a]);
        let ea = k.ext1(GLSL_EXP, na);
        let one = k.cf(1.0);
        let den = k.fadd(one, ea);
        let sil = k.fdiv(a, den);
        k.store_f(yy, ch, sil);

        // advance the history: drop the oldest, append this token's PRE-conv value. Emitted only
        // when there is a history to shift — `d_conv = 1` is a pointwise conv and has none, and a
        // `taps − 1` computed on `u32` would wrap to 4 294 967 295.
        if taps >= 1 {
            if taps >= 2 {
                let cshift = k.cu(taps - 1);
                k.loop_u(c0, cshift, |k, r| {
                    let d = k.iadd(sb, r);
                    let s = {
                        let one_ = k.cu(1);
                        let rp = k.iadd(r, one_);
                        k.iadd(sb, rp)
                    };
                    let v = k.load_f(st, s);
                    k.store_f(st, d, v);
                });
            }
            let last = {
                let t = k.cu(taps - 1);
                k.iadd(sb, t)
            };
            k.store_f(st, last, xv);
        }
    });
    k.finish(LOCAL)
}

/// **Per-head L2 normalise, in place** — `x / sqrt(Σx² + eps)`.
///
/// NOT an RMS norm, and the difference is not cosmetic: the reference calls `ggml_l2_norm`, which
/// has **no `1/n`** and **no learned gain**. Reusing [`qk_norm_module`](super::qk_norm_module) here
/// would scale every key by `sqrt(hd)` = 11.3 — finite, plausible, and wrong in a way the delta
/// rule absorbs into `beta` rather than blowing up.
///
/// `off` lets q and k be normalised inside the SAME fused `y` buffer (q at 0, k at `key_dim`); it is
/// baked per module, so a layer builds two pipelines off one kernel.
///
/// Bindings: 0 = `x` (f32, read-write). Dispatch ≥ `heads`.
/// **[`deltanet_l2_heads_module`] over BOTH head windows in ONE dispatch.**
///
/// The serving chain normalised q and k with two launches of the same kernel — `for off in [0, kd]`
/// — which is **96 dispatches for 48 recurrent layers** on Qwen3.8-27B, 592 µs of a 21 ms token in
/// a kernel that moves almost nothing. `results/dispatch-barriers-are-the-token.md` measures the
/// inter-dispatch drain at **6.2 µs** and 72.5 % of the 0.8B's token, so a launch that exists only
/// because a constant differs is pure loss.
///
/// One pipeline, `2·heads` groups: the lower half takes `off_a`, the upper `off_b`. Each group
/// walks the same `base = h·hd + off` over the same `hd` elements in the same order its own launch
/// did, so no sum is re-associated. That the fused form reproduces the pair is **gated, not
/// argued**: SEQHASH is unchanged at `a1d9716181890a88` on the 27B and `e3cc2998bfa01325` on the
/// 0.8B across every run of both arms (`results/best-of-3-is-not-a-measurement.md`), and the pair
/// is still reachable at `DEC_DNL2FUSE=0` so the gate can be re-run.
///
/// Bindings: 0 = `x` (f32, read-write). Dispatch ≥ `2·heads`.
pub fn deltanet_l2_heads2_module(
    heads: u32,
    hd: u32,
    eps: f32,
    off_a: u32,
    off_b: u32,
) -> Vec<u32> {
    assert!(
        off_b >= off_a,
        "the second window must not precede the first"
    );
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let g = k.gid_comp(0);
    let ch = k.cu(heads);
    let c2h = k.cu(2 * heads);
    let guard = k.ult(g, c2h);
    k.iff(guard, |k| {
        let chd = k.cu(hd);
        // lower half -> window A, upper half -> window B; `h` is the head within its window
        let h = k.umod(g, ch);
        let sel = k.udiv(g, ch);
        // affine in the half index, so no bool/select is needed: the guard above admits only
        // `g < 2·heads`, which leaves `sel = g / heads` at 0 or 1 and `coff` at `off_a` or `off_b`
        let ca = k.cu(off_a);
        let cd = k.cu(off_b - off_a);
        let step = k.imul(sel, cd);
        let coff = k.iadd(ca, step);
        let base = {
            let b0 = k.imul(h, chd);
            k.iadd(b0, coff)
        };
        let c0 = k.cu(0);
        let acc = k.fvar_f();
        let z = k.cf(0.0);
        k.store_var(acc, z);
        k.loop_u(c0, chd, |k, i| {
            let idx = k.iadd(base, i);
            let v = k.load_f(x, idx);
            let sq = k.fmul(v, v);
            let a0 = k.load_var_f(acc);
            let a1 = k.fadd(a0, sq);
            k.store_var(acc, a1);
        });
        let ss = k.load_var_f(acc);
        let ce = k.cf(eps);
        let se = k.fadd(ss, ce);
        let inv = k.ext1(GLSL_INVERSE_SQRT, se);
        k.loop_u(c0, chd, |k, i| {
            let idx = k.iadd(base, i);
            let v = k.load_f(x, idx);
            let o = k.fmul(v, inv);
            k.store_f(x, idx, o);
        });
    });
    k.finish(1)
}

/// [`deltanet_l2_heads2_module`] as bytes.
pub fn deltanet_l2_heads2_bytes(heads: u32, hd: u32, eps: f32, off_a: u32, off_b: u32) -> Vec<u8> {
    words_to_bytes(&deltanet_l2_heads2_module(heads, hd, eps, off_a, off_b))
}

pub fn deltanet_l2_heads_module(heads: u32, hd: u32, eps: f32, off: u32) -> Vec<u32> {
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let h = k.gid_comp(0);
    let ch = k.cu(heads);
    let guard = k.ult(h, ch);
    k.iff(guard, |k| {
        let chd = k.cu(hd);
        let coff = k.cu(off);
        let base = {
            let b0 = k.imul(h, chd);
            k.iadd(b0, coff)
        };
        let c0 = k.cu(0);
        let acc = k.fvar_f();
        let z = k.cf(0.0);
        k.store_var(acc, z);
        k.loop_u(c0, chd, |k, i| {
            let idx = k.iadd(base, i);
            let v = k.load_f(x, idx);
            let sq = k.fmul(v, v);
            let a0 = k.load_var_f(acc);
            let a1 = k.fadd(a0, sq);
            k.store_var(acc, a1);
        });
        let ss = k.load_var_f(acc);
        let ce = k.cf(eps);
        let se = k.fadd(ss, ce);
        let inv = k.ext1(GLSL_INVERSE_SQRT, se);
        k.loop_u(c0, chd, |k, i| {
            let idx = k.iadd(base, i);
            let v = k.load_f(x, idx);
            let o = k.fmul(v, inv);
            k.store_f(x, idx, o);
        });
    });
    k.finish(1)
}

/// **The two per-head scalars the delta rule consumes** — `beta` and the LOG decay `g`.
///
/// `beta ← sigmoid(beta_raw)` and `g ← softplus(alpha + dt_bias) · a`, where `a` already carries
/// `−exp(A_log)` so the forward only multiplies. Both are computed in place, which is what lets the
/// rule kernel read bindings 4 and 5 with no copy between them.
///
/// **Softplus is emitted as `max(t, 0) + log(1 + exp(−|t|))`, not as `log(1 + exp(t))`.** The naive
/// form overflows f32 at `t > 88`, and the reference guards it with a `t > 20` branch. The form
/// here needs no branch and no threshold: `exp(−|t|)` stays inside `(0, 1]` whatever the input, so
/// the `log` never sees a large argument.
///
/// That the two forms agree is not asserted here, it is MEASURED: `deltanet_gpu_layer`'s CPU arm
/// runs the reference's branch form and this kernel runs the reformulation, on the same `alpha` and
/// `dt_bias`, and `g` feeds `exp(g)` — a decay, so any disagreement compounds through the sequence
/// rather than cancelling. 64 carried tokens at 48 heads agree to 8.44e−6 on the recurrent state.
///
/// Bindings: 0 = `beta` (f32, `n_v_heads`, read-write: raw in, sigmoid out), 1 = `g` (f32,
/// `n_v_heads`, read-write: `alpha` in, log-decay out), 2 = `dt_bias`, 3 = `a`. Dispatch ≥
/// `n_v_heads`.
pub fn deltanet_gk_module(n_v_heads: u32) -> Vec<u32> {
    let mut k = Kb::new();
    let bb = k.ssbo_f(0);
    let gg = k.ssbo_f(1);
    let dt = k.ssbo_f(2);
    let aa = k.ssbo_f(3);
    let h = k.gid_comp(0);
    let cnv = k.cu(n_v_heads);
    let guard = k.ult(h, cnv);
    k.iff(guard, |k| {
        let f = k.float;
        let one = k.cf(1.0);
        let zero = k.cf(0.0);

        // beta = sigmoid(beta_raw)
        let braw = k.load_f(bb, h);
        let nb = k.e(OP_FNEG, f, &[braw]);
        let eb = k.ext1(GLSL_EXP, nb);
        let db = k.fadd(one, eb);
        let beta = k.fdiv(one, db);
        k.store_f(bb, h, beta);

        // g = softplus(alpha + dt_bias) · a
        let al = k.load_f(gg, h);
        let bias = k.load_f(dt, h);
        let t = k.fadd(al, bias);
        let at = k.ext1(GLSL_FABS, t);
        let nat = k.e(OP_FNEG, f, &[at]);
        let ent = k.ext1(GLSL_EXP, nat);
        let l1 = k.fadd(one, ent);
        let lg = k.ext1(GLSL_LOG, l1);
        let pos = k.ext2(GLSL_FMAX, t, zero);
        let sp = k.fadd(pos, lg);
        let av = k.load_f(aa, h);
        let g = k.fmul(sp, av);
        k.store_f(gg, h, g);
    });
    k.finish(1)
}

/// **The gated output norm** — `o ← RMSNorm(o) · norm · silu(z)`, per v-head, in place.
///
/// This is an RMS norm (it HAS the `1/n` and it HAS a learned gain), unlike the L2 above, and the
/// two sit four lines apart in the reference. The gain is indexed by `j` WITHIN the head — one
/// `[head_v_dim]` vector shared by all 48 heads, not a `[value_dim]` vector.
///
/// Bindings: 0 = `o` (f32, `n_v_heads·hd`, read-write), 1 = `z` (f32, `n_v_heads·hd`, the gate
/// projection, un-activated), 2 = `norm` (f32, `hd`). Dispatch ≥ `n_v_heads`.
pub fn deltanet_gate_norm_module(hd: u32, n_v_heads: u32, eps: f32) -> Vec<u32> {
    let mut k = Kb::new();
    let oo = k.ssbo_f(0);
    let zz = k.ssbo_f(1);
    let nn = k.ssbo_f(2);
    let h = k.gid_comp(0);
    let cnv = k.cu(n_v_heads);
    let guard = k.ult(h, cnv);
    k.iff(guard, |k| {
        let f = k.float;
        let chd = k.cu(hd);
        let base = k.imul(h, chd);
        let c0 = k.cu(0);
        let acc = k.fvar_f();
        let z0 = k.cf(0.0);
        k.store_var(acc, z0);
        k.loop_u(c0, chd, |k, j| {
            let idx = k.iadd(base, j);
            let v = k.load_f(oo, idx);
            let sq = k.fmul(v, v);
            let a0 = k.load_var_f(acc);
            let a1 = k.fadd(a0, sq);
            k.store_var(acc, a1);
        });
        let ss = k.load_var_f(acc);
        let invn = k.cf(1.0 / hd as f32);
        let mean = k.fmul(ss, invn);
        let ce = k.cf(eps);
        let me = k.fadd(mean, ce);
        let inv = k.ext1(GLSL_INVERSE_SQRT, me);
        let one = k.cf(1.0);
        k.loop_u(c0, chd, |k, j| {
            let idx = k.iadd(base, j);
            let v = k.load_f(oo, idx);
            let gain = k.load_f(nn, j);
            let zv = k.load_f(zz, idx);
            let nz = k.e(OP_FNEG, f, &[zv]);
            let ez = k.ext1(GLSL_EXP, nz);
            let dz = k.fadd(one, ez);
            let sz = k.fdiv(zv, dz);
            let t0 = k.fmul(v, inv);
            let t1 = k.fmul(t0, gain);
            let out = k.fmul(t1, sz);
            k.store_f(oo, idx, out);
        });
    });
    k.finish(1)
}

/// [`deltanet_conv_silu_module`] as bytes.
pub fn deltanet_conv_silu_bytes(conv_dim: u32, d_conv: u32) -> Vec<u8> {
    words_to_bytes(&deltanet_conv_silu_module(conv_dim, d_conv))
}

/// [`deltanet_l2_heads_module`] as bytes.
pub fn deltanet_l2_heads_bytes(heads: u32, hd: u32, eps: f32, off: u32) -> Vec<u8> {
    words_to_bytes(&deltanet_l2_heads_module(heads, hd, eps, off))
}

/// [`deltanet_gk_module`] as bytes.
pub fn deltanet_gk_bytes(n_v_heads: u32) -> Vec<u8> {
    words_to_bytes(&deltanet_gk_module(n_v_heads))
}

/// [`deltanet_gate_norm_module`] as bytes.
pub fn deltanet_gate_norm_bytes(hd: u32, n_v_heads: u32, eps: f32) -> Vec<u8> {
    words_to_bytes(&deltanet_gate_norm_module(hd, n_v_heads, eps))
}

fn words_to_bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}
