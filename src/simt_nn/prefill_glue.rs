//! simt_nn::prefill_glue — the two kernels that join the batched prefill family to the decode
//! serving buffers: the batched embed gather and the KV re-layout.
//!
//! Not ported from fold: fold's prefill hands its KV back to the host (`GpuPrefillStack::run`
//! folds the readback mirrors into `CpuState`) and seeds the decode cache from the host
//! (`GpuDecode::seed_kv`). Here both stay on the device: [`kv_seed_module`] is `seed_kv`'s body
//! (K codes packed 4/word dim-axis, scales transposed `[pos][kvh]` -> `[kvh][pos]`, V
//! `code·scale -> f16`) as one dispatch per layer.

use super::*;

/// **Batched embed gather from the chart** — [`embed_chart_module`] over a chunk of tokens.
/// Bindings: 0 = W (u32), 1 = DM (u32 f16 pairs), 2 = toks (u32, `toks[i]` = token of position
/// `i`), 3 = x out (f32, `hidden` per position). Dispatch `(ceil(hidden/local), B)`;
/// `gl_GlobalInvocationId.y` is the position.
pub fn embed_chart_batch_module(
    hidden: u32,
    vocab: u32,
    local: u32,
    carrier: crate::simt::Carrier,
) -> Vec<u32> {
    let pack = carrier.pack();
    let width = 32 / pack;
    let mask = carrier.qmask();
    let mut k = Kb::new();
    let w = k.ssbo_u(0);
    let dm = k.ssbo_u(1);
    let toks = k.ssbo_u(2);
    let x = k.ssbo_f(3);
    let i = k.gid_comp(0);
    let p = k.gid_comp(1);
    let ch = k.cu(hidden);
    let guard = k.ult(i, ch);
    k.iff(guard, |k| {
        let tok = k.load_u(toks, p);
        let cv = k.cu(vocab);
        let cp = k.cu(pack);
        let iw = k.udiv(i, cp);
        let widx0 = k.imul(iw, cv);
        let widx = k.iadd(widx0, tok);
        let word = k.load_u(w, widx);
        let lane = k.umod(i, cp);
        let cwd = k.cu(width);
        let sh = k.imul(lane, cwd);
        let u = k.uint;
        let shifted = k.e(OP_SHIFT_RIGHT_LOGICAL2, u, &[word, sh]);
        let cm = k.cu(mask);
        let q = k.e(OP_BITWISE_AND, u, &[shifted, cm]);
        let qf = k.utof(q);
        let c32 = k.cu(32);
        let b = k.udiv(i, c32);
        let dmi0 = k.imul(b, cv);
        let dmi = k.iadd(dmi0, tok);
        let word2 = k.load_u(dm, dmi);
        let unp = {
            let (v2, g) = (k.vec2f, k.glsl);
            k.e(OP_EXT_INST, v2, &[g, GLSL_UNPACK_HALF2X16, word2])
        };
        let f = k.float;
        let d = k.e(OP_COMPOSITE_EXTRACT, f, &[unp, 0]);
        let lo = k.e(OP_COMPOSITE_EXTRACT, f, &[unp, 1]);
        let dq = k.fmul(d, qf);
        let val = k.fadd(dq, lo);
        let ch2 = k.cu(hidden);
        let xo = k.imul(p, ch2);
        let xi = k.iadd(xo, i);
        k.store_f(x, xi, val);
    });
    k.finish(local)
}

/// **KV re-layout, prefill cache -> decode cache** — one layer, positions `pos0 .. pos0+B`.
///
/// Prefill layout ([`qkv_rope_batch_module`]): `kc` f32 codes `[(kvh·hd+d)·tmax + s]`, `ks` f32
/// `[s·nkv + kvh]`, `vc` f32 codes `[(kvh·tmax+s)·hd + d]`, `vs` f32 `[s·nkv + kvh]`.
/// Decode layout ([`fused_rope_kv_fold_module`], `v8 = false`): `kcq` u32, 4 int8 codes along the
/// dim axis per word `[(kvh·(hd/4)+w)·tmax + s]`; `sk` f32 `[kvh·tmax + s]`; `vdec` f16
/// `[(kvh·tmax+s)·hd + d]` holding `code·vs`.
///
/// Bindings: 0 = kc, 1 = ks, 2 = vc, 3 = vs, 4 = params (`params[0]` = `pos0`), 5 = kcq, 6 = sk,
/// 7 = vdec. Dispatch `(nkv, B)` at LocalSize `hd`.
pub fn kv_seed_module(nkv: u32, hd: u32, tmax: u32) -> Vec<u32> {
    assert!(hd % 4 == 0, "K words pack 4 codes along the head dim");
    let mut k = Kb::new();
    let kc = k.ssbo_f(0);
    let ks = k.ssbo_f(1);
    let vc = k.ssbo_f(2);
    let vs = k.ssbo_f(3);
    let params = k.ssbo_u(4);
    let kcq = k.ssbo_u(5);
    let sk = k.ssbo_f(6);
    let vdec = k.ssbo_f16(7);
    let chd = k.cu(hd);
    let gid = k.gid_comp(0);
    let i = k.gid_comp(1);
    let kvh = k.udiv(gid, chd);
    let lane = k.umod(gid, chd);
    let c0 = k.cu(0);
    let pos0 = k.load_u(params, c0);
    let pos = k.iadd(pos0, i);
    let ct = k.cu(tmax);
    let cnkv = k.cu(nkv);
    let sidx0 = k.imul(pos, cnkv);
    let sidx = k.iadd(sidx0, kvh); // scale slot [pos·nkv + kvh]
    let kvt = k.imul(kvh, ct);
    // K word: lanes < hd/4 each pack dims 4w..4w+3
    let chd4 = k.cu(hd / 4);
    let packer = k.ult(lane, chd4);
    k.iff(packer, |k| {
        let u = k.uint;
        let one = k.cf(1.0);
        let mut word = k.cu(0);
        for j in 0..4u32 {
            let c4 = k.cu(4);
            let l4 = k.imul(lane, c4);
            let cj = k.cu(j);
            let d = k.iadd(l4, cj);
            let kh0 = k.imul(kvh, chd);
            let row = k.iadd(kh0, d);
            let rt = k.imul(row, ct);
            let src = k.iadd(rt, pos);
            let v = k.load_f(kc, src);
            let byte = k.int8_code(v, one);
            let csh = k.cu(8 * j);
            let sh = k.e(OP_SHIFT_LEFT_LOGICAL, u, &[byte, csh]);
            word = k.e(OP_BITWISE_OR, u, &[word, sh]);
        }
        let wr0 = k.imul(kvh, chd4);
        let wr1 = k.iadd(wr0, lane);
        let wr2 = k.imul(wr1, ct);
        let dst = k.iadd(wr2, pos);
        k.store_u(kcq, dst, word);
    });
    let is0 = k.e(OP_IEQ, k.bool_t, &[lane, c0]);
    k.iff(is0, |k| {
        let s = k.load_f(ks, sidx);
        let di = k.iadd(kvt, pos);
        k.store_f(sk, di, s);
    });
    // V: f16(code · scale)
    let vi0 = k.iadd(kvt, pos);
    let vi1 = k.imul(vi0, chd);
    let vi = k.iadd(vi1, lane);
    let code = k.load_f(vc, vi);
    let sc = k.load_f(vs, sidx);
    let val = k.fmul(code, sc);
    k.store_f16(vdec, vi, val);
    k.finish(hd)
}

/// [`embed_chart_batch_module`] as bytes.
pub fn embed_chart_batch_bytes(
    hidden: u32,
    vocab: u32,
    local: u32,
    carrier: crate::simt::Carrier,
) -> Vec<u8> {
    bytes(embed_chart_batch_module(hidden, vocab, local, carrier))
}
/// [`kv_seed_module`] as bytes.
pub fn kv_seed_bytes(nkv: u32, hd: u32, tmax: u32) -> Vec<u8> {
    bytes(kv_seed_module(nkv, hd, tmax))
}
