//! **The two small kernels the routed FFN needs around the gemvs.**
//!
//! `gemv_dp4a_moe` reads `SEL[slot]` and folds one expert's chart; [`moe_topk_module`] produces
//! that `SEL` (and the mixture weights beside it), and [`silu_mul_q_slot_module`] is the SwiGLU
//! between the two routed gemvs. Neither is a matmul, but both have to run ON the device, because
//! a decode command buffer is recorded once and replayed: a top-k that went through the host would
//! stall every layer.
//!
//! **There is no combine kernel.** A third module used to live here — `moe_combine`, one dispatch
//! doing `x += Σ_s swt[s]·edn[s]`. The reference does that multiply in the down projection's
//! epilogue instead (vLLM `fused_moe.py::fused_moe_kernel`'s `MUL_ROUTED_WEIGHT`), which is where
//! the value already sits in a register; the sum is then the seg-reduce the chain already runs.
//! It was retired for `gemv_dp4a_moe_bytes(.., weighted = true)` — one dispatch a routed layer,
//! priced at ~1.3 µs marginal (`results/dispatch-cost.md`), ~62 µs/token at 48 layers
//! (`results/moe-reference-set.md` §1). Its gate moved with it, into
//! `examples/moe_gemv_check.rs`'s combine-equivalence section.
//!
//! The arithmetic is the same as `bedrock`'s `route_topk`, deliberately, including the trap it
//! documents: a router bias steers SELECTION and must not enter the mixture weight.

use super::*;

/// **Top-k routing, on the device.** One invocation reads `n_experts` logits and writes the
/// selected expert ids and their mixture weights.
///
/// Selection is by score + bias, the weight is the unbiased score, and the sigmoid family always
/// renormalises — matching `route_topk` term for term, because the CPU path is the reference this
/// is graded against.
///
/// Softmax weights are computed against the FULL normaliser (`exp(l − max) / Z` over all experts)
/// rather than only over the selected k. With `norm_topk_prob` that normaliser cancels, so it looks
/// like wasted work — but without it the dropped mass is the answer, and computing it over the
/// selected k would silently renormalise a model that must not be.
///
/// The selection loop is O(n_used · n_experts) with an O(n_used) exclusion check per candidate —
/// 256 experts and top-8 is ~2k iterations in ONE invocation. That is the shape to beat if this
/// ever shows up in a profile, not a shape to defend.
///
/// Bindings: 0 = logits (f32, `n_experts`), 1 = sel (u32, `n_used`), 2 = swt (f32, `n_used`),
/// 3 = bias (f32, `n_experts`; bind zeros when the model has none). Dispatch (1, 1, 1).
pub fn moe_topk_module(
    n_experts: u32,
    n_used: u32,
    sigmoid: bool,
    norm: bool,
    routed_scale: f32,
) -> Vec<u32> {
    let mut k = Kb::new();
    let logits = k.ssbo_f(0);
    let sel = k.ssbo_u(1);
    let swt = k.ssbo_f(2);
    let bias = k.ssbo_f(3);

    let gi = k.gid_comp(0);
    let zero_u = k.cu(0);
    let bt = k.bool_t;
    let only0 = k.e(OP_IEQ, bt, &[gi, zero_u]);

    k.iff(only0, |k| {
        let c0 = k.cu(0);
        let c1 = k.cu(1);
        let cne = k.cu(n_experts);
        let cnu = k.cu(n_used);
        let f0 = k.cf(0.0);
        let f1 = k.cf(1.0);
        let neg = k.cf(-1.0e30);

        // ── pass 1: the full-population statistics the softmax weight needs ──
        let mx = k.fvar_f();
        k.store_var(mx, neg);
        k.loop_u(c0, cne, |k, e| {
            let l = k.load_f(logits, e);
            let m0 = k.load_var_f(mx);
            let bt = k.bool_t;
            let gt = k.e(OP_FORD_GREATER_THAN, bt, &[l, m0]);
            k.iff(gt, |k| k.store_var(mx, l));
        });
        let zsum = k.fvar_f();
        k.store_var(zsum, f0);
        if !sigmoid {
            k.loop_u(c0, cne, |k, e| {
                let l = k.load_f(logits, e);
                let m0 = k.load_var_f(mx);
                let d = k.fsub(l, m0);
                let ex = k.ext1(GLSL_EXP, d);
                let z0 = k.load_var_f(zsum);
                let z1 = k.fadd(z0, ex);
                k.store_var(zsum, z1);
            });
        }

        // ── pass 2: pick n_used, excluding what is already picked ──
        k.loop_u(c0, cnu, |k, s| {
            let best = k.fvar_f();
            let bi = k.fvar_u();
            k.store_var(best, neg);
            k.store_var(bi, c0);

            k.loop_u(c0, cne, |k, e| {
                // already selected?
                let taken = k.fvar_u();
                k.store_var(taken, c0);
                k.loop_u(c0, s, |k, t| {
                    let prev = k.load_u(sel, t);
                    let bt = k.bool_t;
                    let same = k.e(OP_IEQ, bt, &[prev, e]);
                    k.iff(same, |k| k.store_var(taken, c1));
                });
                let tk = k.load_var_u(taken);
                let bt = k.bool_t;
                let free = k.e(OP_IEQ, bt, &[tk, c0]);
                k.iff(free, |k| {
                    // the SELECTION key: score + bias
                    let l = k.load_f(logits, e);
                    let score = if sigmoid {
                        let nl = k.fsub(f0, l);
                        let ex = k.ext1(GLSL_EXP, nl);
                        let den = k.fadd(f1, ex);
                        k.fdiv(f1, den)
                    } else {
                        l
                    };
                    let bv = k.load_f(bias, e);
                    let key = k.fadd(score, bv);
                    let b0 = k.load_var_f(best);
                    let bt = k.bool_t;
                    let gt = k.e(OP_FORD_GREATER_THAN, bt, &[key, b0]);
                    k.iff(gt, |k| {
                        k.store_var(best, key);
                        k.store_var(bi, e);
                    });
                });
            });

            // the WEIGHT is the unbiased score of the winner
            let e = k.load_var_u(bi);
            k.store_u(sel, s, e);
            let l = k.load_f(logits, e);
            let wv = if sigmoid {
                let nl = k.fsub(f0, l);
                let ex = k.ext1(GLSL_EXP, nl);
                let den = k.fadd(f1, ex);
                k.fdiv(f1, den)
            } else {
                let m0 = k.load_var_f(mx);
                let d = k.fsub(l, m0);
                let ex = k.ext1(GLSL_EXP, d);
                let z = k.load_var_f(zsum);
                k.fdiv(ex, z)
            };
            k.store_f(swt, s, wv);
        });

        // ── renormalise: always for sigmoid (its scores sum to nothing in particular) ──
        if norm || sigmoid {
            let tot = k.fvar_f();
            k.store_var(tot, f0);
            k.loop_u(c0, cnu, |k, s| {
                let v = k.load_f(swt, s);
                let t0 = k.load_var_f(tot);
                let t1 = k.fadd(t0, v);
                k.store_var(tot, t1);
            });
            let t = k.load_var_f(tot);
            let bt = k.bool_t;
            let pos = k.e(OP_FORD_GREATER_THAN, bt, &[t, f0]);
            k.iff(pos, |k| {
                k.loop_u(c0, cnu, |k, s| {
                    let v = k.load_f(swt, s);
                    let t = k.load_var_f(tot);
                    let d = k.fdiv(v, t);
                    k.store_f(swt, s, d);
                });
            });
        }
        if routed_scale != 1.0 {
            let cs = k.cf(routed_scale);
            k.loop_u(c0, cnu, |k, s| {
                let v = k.load_f(swt, s);
                let p = k.fmul(v, cs);
                k.store_f(swt, s, p);
            });
        }
    });
    k.finish(1)
}

/// [`moe_topk_module`] as bytes.
pub fn moe_topk_bytes(
    n_experts: u32,
    n_used: u32,
    sigmoid: bool,
    norm: bool,
    routed_scale: f32,
) -> Vec<u8> {
    moe_topk_module(n_experts, n_used, sigmoid, norm, routed_scale)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

/// **SwiGLU + activation quantise for ALL routed slots, in one dispatch.**
///
/// The dense path's `silu_mul_q` owns one vector in one workgroup. The routed path needs
/// `n_used` of them, and issuing `n_used` dispatches per layer would cost ~384 extra dispatches a
/// token at 48 layers — more than the whole dense chain. So the slot becomes `gid.y`: one
/// workgroup per slot, each with its own shared scratch, each reading and writing its own stride.
///
/// The gemv wrote `egu[slot·(2·dff) + …]`, so gate is at `slot·2·dff + i` and up at
/// `slot·2·dff + dff + i`; the quantised activation lands at `slot·(dff/4)` words with its scales
/// at `slot·(dff/32)`.
///
/// Bindings: 0 = g (f32, the gemv output — g and u are the same buffer, `u_off` apart), 1 = u,
/// 2 = xq (packed codes), 3 = scales, 4 = xs. Dispatch `groups = (1, n_used)`.
pub fn silu_mul_q_slot_module(dff: u32, dpw: u32) -> Vec<u32> {
    const WG: u32 = 256;
    let mut k = Kb::new();
    let g = k.ssbo_f(0);
    let u = k.ssbo_f(1);
    let xq = k.ssbo_u(2);
    let scales = k.ssbo_f(3);
    let xs = k.ssbo_f(4);
    let (nrm, nrmp) = k.shared_f32(dff);
    let sint = k.sint_ty();

    let tid = k.gid_comp(0);
    let slot = k.gid_comp(1);

    // the three strides this slot owns
    let c_in = k.cu(2 * dff); //   the gemv writes [slot · 2·dff]
    let in_base = k.imul(slot, c_in);
    let c_q = k.cu(dff / 4); //    one u32 per 4 codes
    let q_base = k.imul(slot, c_q);
    let c_b = k.cu(dff / 32); //   one scale + one sum per 32-block
    let b_base = k.imul(slot, c_b);

    let cwg = k.cu(WG);
    let cn = k.cu(dff);
    let cuo = k.cu(dff); //        u sits one dff past g within the same slot
    k.loop_step(tid, cn, cwg, |k, i| {
        let gi = k.iadd(in_base, i);
        let gv = k.load_f(g, gi);
        let ui0 = k.iadd(i, cuo);
        let ui = k.iadd(in_base, ui0);
        let uv = k.load_f(u, ui);
        let f = k.float;
        let ng = k.e(OP_FNEG, f, &[gv]);
        let eg = k.ext1(GLSL_EXP, ng);
        let one = k.cf(1.0);
        let den = k.fadd(one, eg);
        let sil = k.fdiv(gv, den);
        let ov = k.fmul(sil, uv);
        k.shared_store(nrm, nrmp, i, ov);
    });
    k.wg_barrier();
    k.emit_quantize_from_shared(
        nrm,
        nrmp,
        xq,
        scales,
        xs,
        dff,
        dpw,
        sint,
        tid,
        WG,
        Some((q_base, b_base)),
    );
    k.finish(WG)
}

/// [`silu_mul_q_slot_module`] as bytes.
pub fn silu_mul_q_slot_bytes(dff: u32, dpw: u32) -> Vec<u8> {
    silu_mul_q_slot_module(dff, dpw)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}
