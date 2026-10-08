//! One decoder layer's GPU dispatch sub-graph, ported from fold `bedrock/src/serve/gpu.rs`.
//!
//! Ported verbatim (`gpu.rs:527-797`): `make_moe_gemv`, `make_gemv3`, `make_fused_gemv`,
//! `make_actq_gemv`. `resolve_decode`, `make_step*`, `Step::with_bytes`, `bar_mode`, `step_barrier`
//! and `rec_*` (`gpu.rs:484-526`, `798-944`) are already in [`super`] from the previous slice.
//! [`record_layer`] is the dense-attention layer section of `GpuDecode::new` (`gpu.rs:2300-3160`),
//! fused default path only (`DEC_FOLD`, `DEC_FUSENORM`, `DEC_ACTQFUSE`, `DEC_COMBFUSE`,
//! `DEC_ROPEFUSE` all at their default 1), restricted to a dense layer with no QK-norm, attention
//! gate, knot, compact pool, routed experts or recurrent block.
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's bedrock/src/serve/gpu.rs; lint profile is fold's, not apeiron's"
)]

use super::state::{LayerCfg, LayerW, Scratch, Stream};
use super::*;

/// **The ROUTED gemv** — `n_used` slots in ONE dispatch, each folding whichever expert
/// `sel[slot]` names.
///
/// No split-K: `gid.y` is the slot, so the two claims on that axis are mutually exclusive
/// (the kernel asserts it). `bytes` is what a token actually MOVES — `n_used` experts' charts,
/// not the whole concatenated tensor — which is the only figure the roofline row can mean here.
///
/// `weighted` binds `swt` at 7 and multiplies the router's mixture weight into the store
/// (vLLM's `MUL_ROUTED_WEIGHT`). It belongs on the DOWN projection only — the gate/up gemv passes
/// `false`, and `swt` is then unread. The weighted slots stay separate rows of `y`; the seg-reduce
/// that already folds split-K partials sums them, so there is no combine dispatch.
///
/// `x_slot` is the ACTIVATION's twin of the same distinction, and it is the down projection's too:
/// gate/up folds the layer's one normed hidden against `n_used` experts, while down folds `n_used`
/// DIFFERENT vectors — the per-slot SwiGLU outputs `silu_mul_q_slot` wrote at `s·(k/4)` words and
/// `s·(k/32)` blocks. Without it every slot reads slot 0's activation.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles.
#[allow(clippy::too_many_arguments)]
pub unsafe fn make_moe_gemv(
    g: &GpuCtx,
    carrier: Carrier,
    w: Handle,
    dm: Handle,
    x: Handle,
    xs_sx: (Handle, Handle),
    sel: Handle,
    swt: Handle,
    y: Handle,
    m: u32,
    k: u32,
    n_used: u32,
    weighted: bool,
    x_slot: bool,
) -> Gemv {
    // `code_words_per_row`, not `k/pack`: `Carrier::Q3`/`Q5` are TWO planes and `pack()` describes
    // only plane A, so `k/pack` under-strides the expert chart for a two-plane carrier.
    let wpe = carrier.code_words_per_row(k) * m;
    let dmpe = (k / 32) * m;
    let mut bufs = vec![w, x, y, dm, xs_sx.0, xs_sx.1, sel];
    if weighted {
        bufs.push(swt);
    }
    let (l1, d1) = g.vk.ssbo_set(&bufs);
    // THROUGH THE SPEC, like the other four. This was the last dispatcher reaching a `gemv_*`
    // emitter directly — the shape `results/shape-derived-gemv.md` describes, where a policy lands
    // on one sibling and another sibling never sees it. `legalize` now names the axis a MoE spec
    // cannot honour instead of the call dropping it in silence.
    let spec = GemvSpec::moe(carrier, wpe, dmpe, weighted, x_slot);
    let pipe = g.vk.pipeline(
        &spec
            .emit(m, k, local_wg())
            .expect("MoE gemv spec is constructed legal by GemvSpec::moe"),
        l1,
    );
    Gemv {
        pipe,
        layout: l1,
        dset: d1,
        groups: (m.div_ceil(local_wg()), n_used),
        stage2: None,
        segs: 1,
        bytes: chart_bytes(m, k, carrier) * n_used as u64,
        m,
        k,
    }
}

/// **The 3-trit basis matmul** — three Q2 plane-gemvs (place values {1,3,9} baked into each plane's
/// `dm`) summed into `y` (`dot_plane_sum`). fold=false so each plane fully reduces before the sum;
/// `tmp` is scratch for planes 1–2. The per-node barriers (rec_gemv/rec_step) make the shared
/// `partials`/`tmp` safe. One matmul → 3 gemvs + 2 adds, reusing the proven ternary Q2 kernel.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles.
#[allow(clippy::too_many_arguments)]
pub unsafe fn make_gemv3(
    g: &GpuCtx,
    planes: &[(Handle, Handle)],
    x: Handle,
    xs_sx: (Handle, Handle),
    y: Handle,
    tmp: Handle,
    partials: Handle,
    ones: Handle,
    m: u32,
    k: u32,
) -> Vec<Node> {
    let mut out = vec![Node::Gemv(make_gemv(
        g,
        Carrier::Q2,
        planes[0].0,
        planes[0].1,
        x,
        xs_sx,
        y,
        partials,
        ones,
        m,
        k,
        false,
        ScaleFace::Dm32, // planes are their own face; the two-level chart is per-Proj
    ))];
    for pl in &planes[1..] {
        out.push(Node::Gemv(make_gemv(
            g,
            Carrier::Q2,
            pl.0,
            pl.1,
            x,
            xs_sx,
            tmp,
            partials,
            ones,
            m,
            k,
            false,
            ScaleFace::Dm32, // as above
        )));
        out.push(Node::Step(make_step_named(
            g,
            &add_inplace_bytes(m, local_wg()),
            &[y, tmp],
            m.div_ceil(local_wg()),
            1,
            "add_inplace",
        )));
    }
    out
}

/// **The RMSNorm+quant co-reduction folded into the DP4A gemv** (sanguine `FusionHidesLatency`):
/// one dispatch replaces `[rmsnorm_fused_q → gemv_dp4a_dyn]`, the norm's sumsq/absmax hiding behind
/// the gemv's weight transport (the norm rides x-in-shared). `residual = Some(res_segs)` folds the
/// residual branch (split-K partials at `res`) into a NEW stream buffer `x_out` (≠ `x_in`, so every
/// WG writes the identical sum — no cross-WG race); the caller ping-pongs. Grid divisor 256 = the
/// kernel's forced WG (byte-matches rmsnorm's reduction tree). Split-K stage 2 as `make_gemv`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles.
#[allow(clippy::too_many_arguments)]
pub unsafe fn make_fused_gemv(
    g: &GpuCtx,
    carrier: Carrier,
    w: Handle,
    dm: Handle,
    gain: Handle,
    x_in: Handle,
    x_out: Handle,
    res: Handle,
    y: Handle,
    partials: Handle,
    ones: Handle,
    m: u32,
    k: u32,
    eps: f32,
    residual: Option<u32>,
    // **`two_level` = the `qsub`/`qsup` scale face** at binding 4 in place of `dm` (`tl_face`).
    // This path serves `attn_qkv` and `ffn_gateup` whenever `fusenorm` is on — the two largest
    // projections in the stack — so omitting it read a `[sub][sup]` buffer as f16 pairs and the
    // 0.8B emitted token 0 forty times at a healthy tok/s with zero argmax mismatches.
    scale: ScaleFace,
) -> Gemv {
    const FWG: u32 = 256; // the fused kernel forces LocalSize 256 (rmsnorm reduction byte-identity)
    let chart = carrier.is_chart();
    // the third sibling on the same policy — see `make_actq_gemv`
    let spec = resolve_decode(
        g,
        GemvSpec::decode(carrier, false).fused(Fusion::Norm { eps, residual }),
        m,
        k,
        scale,
    );
    let s = spec.reduce.segs();
    let out1 = if s > 1 { partials } else { y };
    // carrier is the dial: chart binds DM at slot 4 (DP4A); direct (F16) has no DM, so res/x_out
    // shift down one. `gemv_norm_bytes` matches this binding layout by carrier.
    let bufs: Vec<Handle> = match (chart, residual) {
        (true, Some(_)) => vec![w, x_in, gain, out1, dm, res, x_out],
        (true, None) => vec![w, x_in, gain, out1, dm],
        (false, Some(_)) => vec![w, x_in, gain, out1, res, x_out],
        (false, None) => vec![w, x_in, gain, out1],
    };
    let (l1, d1) = g.vk.ssbo_set(&bufs);
    let pipe = g.vk.pipeline(
        &spec
            .emit(m, k, FWG)
            .unwrap_or_else(|e| panic!("illegal fused gemv spec at {m}x{k}: {e}")),
        l1,
    );
    let stage2 = (s > 1).then(|| {
        let (l2, d2) = g.vk.ssbo_set(&[partials, ones, y]);
        let p2 =
            g.vk.pipeline(&gemv_bytes(m, s, local_wg(), 1, Carrier::F32), l2);
        (p2, l2, d2, (m.div_ceil(local_wg()), 1u32))
    });
    Gemv {
        pipe,
        layout: l1,
        dset: d1,
        groups: spec.groups(m, FWG),
        stage2,
        segs: 1,
        bytes: spec.chart_bytes(m, k),
        m,
        k,
    }
}

/// **`[DEC-FUSE]` the activation-producing gemv** — [`gemv_actq_module`] wired as a `Gemv` node:
/// the f32 activation source `src` is turned into the int8 chart inside the kernel, so the producer
/// dispatch (`quantize_vec` for o, `silu_mul_q` for down) and its barrier are gone. `fold` keeps the
/// stage-1 partials in `y` for the next fused-norm gemv to reduce, exactly as [`make_gemv`] does.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles.
#[allow(clippy::too_many_arguments)]
pub unsafe fn make_actq_gemv(
    g: &GpuCtx,
    carrier: Carrier,
    w: Handle,
    dm: Handle,
    src: Handle,
    y: Handle,
    m: u32,
    k: u32,
    act: ActQ,
    // aux = `(part_m, part_l)` — only for `ActQ::FlashCombine`, where `src` is `part_o`
    aux: Option<(Handle, Handle)>,
    // **`two_level` = the `qsub`/`qsup` scale face** at binding 3 in place of `dm` (`tl_face`).
    // This path serves `o`, `ffn_down` and the delta-net `out` — all three of which the packer can
    // put on that face, so omitting it here would read a `[sub][sup]` buffer as though it were
    // f16 pairs: finite logits, wrong model.
    scale: ScaleFace,
) -> Gemv {
    const FWG: u32 = 256; // gemv_actq forces LocalSize 256 (the shared quantize tree)
                          // THE POINT OF THE SPEC: this sibling now resolves through the SAME policy `make_gemv` does.
                          // `results/shape-derived-gemv.md` found the coop rule wired into `make_gemv` alone while
                          // `actqfuse` routed `ffn_down` here — one policy, two dispatchers, and the shape the policy
                          // targeted flowed through the one without it.
    let spec = resolve_decode(
        g,
        GemvSpec::decode(carrier, true).fused(Fusion::ActQ(act)),
        m,
        k,
        scale,
    );
    let s = spec.reduce.segs();
    let mut bufs = vec![w, src, y, dm];
    if let Some((a, b)) = aux {
        bufs.push(a);
        bufs.push(b);
    }
    let (layout, dset) = g.vk.ssbo_set(&bufs);
    let pipe = g.vk.pipeline(
        &spec
            .emit(m, k, FWG)
            .unwrap_or_else(|e| panic!("illegal actq gemv spec at {m}x{k}: {e}")),
        layout,
    );
    Gemv {
        pipe,
        layout,
        dset,
        groups: spec.groups(m, FWG),
        stage2: None,
        segs: s,
        bytes: spec.chart_bytes(m, k),
        m,
        k,
    }
}

/// One dense attention + SwiGLU layer's dispatch nodes on the fused default path
/// (`gpu.rs:2316-3160`, branches `fusenorm && ropefuse && actqfuse && combfuse`, not routed, not
/// recurrent, no QK-norm/gate/knot). Layer `li > 0` folds the previous layer's `d` partials into
/// the stream inside its qkv gemv; `li == 0` reads the embedding as is. `st` advances to the state
/// the next layer consumes.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles.
pub unsafe fn record_layer(
    g: &GpuCtx,
    c: &LayerCfg,
    l: &LayerW,
    sc: &Scratch,
    li: usize,
    st: &mut Stream,
) -> Vec<Node> {
    let (hidden, dff, nh, nkv, hd, tmax) = (c.hidden, c.dff, c.nh, c.nkv, c.hd, c.tmax);
    let (qn, qkv_m) = (c.qn(), c.qkv_m());
    let (s_qkv, nsplit) = (c.s_qkv(), c.nsplit());
    let mut nodes: Vec<Node> = Vec::new();
    // qkv: the RMSNorm rides the projection's weight transport; the split-K stage 2 and the bias
    // move into the rope kernel's prologue (`gm.stage2 = None`, `qkvp` holds the partials).
    if li == 0 {
        let mut gm = make_fused_gemv(
            g,
            l.qkv_c,
            l.qkv_w,
            l.qkv_dm,
            l.ln1,
            st.cur,
            st.alt,
            st.cur,
            sc.qkvp,
            sc.qkvp,
            sc.ones,
            qkv_m,
            hidden,
            c.eps,
            None,
            ScaleFace::Dm32,
        );
        gm.stage2 = None;
        nodes.push(Node::Gemv(gm));
    } else {
        let mut gm = make_fused_gemv(
            g,
            l.qkv_c,
            l.qkv_w,
            l.qkv_dm,
            l.ln1,
            st.cur,
            st.alt,
            sc.d,
            sc.qkvp,
            sc.qkvp,
            sc.ones,
            qkv_m,
            hidden,
            c.eps,
            Some(st.prev_ffn_segs),
            ScaleFace::Dm32,
        );
        gm.stage2 = None;
        nodes.push(Node::Gemv(gm));
        std::mem::swap(&mut st.cur, &mut st.alt);
    }
    let mut bufs = vec![sc.qkv, l.kcq, l.vc, sc.params, l.sk, sc.qkvp];
    if let Some(b) = l.qkv_b {
        bufs.push(b);
    }
    if envu("DEC_ROPEFLASH", 1) == 1 {
        // `DEC_ROPEFLASH` (default 1; 0 records the two-Step path): the `fused_rope_kv_fold` Step
        // and its pipeline barrier are folded into `flash_partial`'s prologue — one dispatch
        // instead of two. `sc.qkv` is written by neither
        // step on this path and nothing downstream reads it (only `flash_partial` did), so the
        // prologue computes q straight into workgroup memory.
        let mut bufs = vec![
            l.kcq, l.vc, sc.part_o, sc.part_m, sc.part_l, sc.params, l.sk, sc.qkvp,
        ];
        if let Some(b) = l.qkv_b {
            bufs.push(b);
        }
        nodes.push(Node::Step(make_step_named(
            g,
            &flash_partial_fold_bytes(
                nh,
                nkv,
                hd,
                tmax,
                c.scale(),
                nsplit,
                64,
                c.theta,
                s_qkv,
                l.qkv_b.is_some(),
                c.rope_dims,
            ),
            &bufs,
            nh * nsplit,
            1,
            "flash_partial_fold",
        )));
    } else {
        nodes.push(Node::Step(make_step_named(
            g,
            &fused_rope_kv_fold_bytes(
                nh,
                nkv,
                hd,
                tmax,
                c.theta,
                0,
                s_qkv,
                l.qkv_b.is_some(),
                c.rope_dims,
                false,
            ),
            &bufs,
            nh + nkv,
            1,
            "fused_rope_kv_fold",
        )));
        nodes.push(Node::Step(make_step_named(
            g,
            &flash_partial_bytes(
                nh,
                nkv,
                hd,
                tmax,
                c.scale(),
                nsplit,
                64,
                false,
                false,
                false,
                0,
            ),
            &[
                sc.qkv, l.kcq, l.vc, sc.part_o, sc.part_m, sc.part_l, sc.params, l.sk,
            ],
            nh * nsplit,
            1,
            "flash_partial",
        )));
    }
    // the split-K attention merge IS the o-projection's activation
    nodes.push(Node::Gemv(make_actq_gemv(
        g,
        l.o_c,
        l.o_w,
        l.o_dm,
        sc.part_o,
        sc.o,
        hidden,
        qn,
        ActQ::FlashCombine(hd, nsplit),
        Some((sc.part_m, sc.part_l)),
        ScaleFace::Dm32,
    )));
    // residual + FFN RMSNorm ride the gate/up projection's transport
    nodes.push(Node::Gemv(make_fused_gemv(
        g,
        l.gu_c,
        l.gu_w,
        l.gu_dm,
        l.ln2,
        st.cur,
        st.alt,
        sc.o,
        sc.gu,
        sc.partials,
        sc.ones,
        2 * dff,
        hidden,
        c.eps,
        Some(c.segs_o()),
        ScaleFace::Dm32,
    )));
    std::mem::swap(&mut st.cur, &mut st.alt);
    nodes.push(Node::Gemv(make_actq_gemv(
        g,
        l.dn_c,
        l.dn_w,
        l.dn_dm,
        sc.gu,
        sc.d,
        hidden,
        dff,
        ActQ::SiluMul(dff),
        None,
        ScaleFace::Dm32,
    )));
    st.prev_ffn_segs = c.segs_d();
    nodes
}

/// The embedding gather node (`gpu.rs:2295-2301`): `x[i]` from the chart row named by
/// `params[1]`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles.
pub unsafe fn embed_step(
    g: &GpuCtx,
    c: &LayerCfg,
    vocab: u32,
    embed_w: Handle,
    embed_dm: Handle,
    sc: &Scratch,
) -> Step {
    make_step_named(
        g,
        &embed_chart_bytes(c.hidden, vocab, local_wg(), c.head_carrier()),
        &[embed_w, embed_dm, sc.params, sc.x],
        c.hidden.div_ceil(local_wg()),
        1,
        "embed_chart",
    )
}

/// The final norm node (`gpu.rs:3213-3221`, `final_rms_f32`): folds the last layer's `d` partials
/// into the stream in place and writes the normed f32 vector to `out`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles.
pub unsafe fn final_norm_step(
    g: &GpuCtx,
    c: &LayerCfg,
    st: &Stream,
    sc: &Scratch,
    gain: Handle,
    out: Handle,
) -> Step {
    make_step_named(
        g,
        &rmsnorm_add_seg_bytes(c.hidden, c.eps, st.prev_ffn_segs),
        &[st.cur, gain, sc.d, out],
        1,
        1,
        "rmsnorm_add_seg",
    )
}

/// Records every node of `nodes` into `cb`; returns the dispatch count.
///
/// # Safety
/// Records into a command buffer in the recording state.
pub unsafe fn rec_nodes(vk: &Vk, cb: CommandBuffer, nodes: &[Node]) -> u32 {
    nodes
        .iter()
        .map(|n| match n {
            Node::Step(s) => rec_step(vk, cb, s),
            Node::Gemv(gm) => rec_gemv(vk, cb, gm),
        })
        .sum()
}
