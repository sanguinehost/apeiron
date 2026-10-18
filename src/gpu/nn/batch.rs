//! simt_nn::batch — the batched (prefill-stage) kernels: the `QSrc` quantize helpers, the batch
//! RMSNorm/quantize forms (dpw and superblock), the last-row gathers, the batched SwiGLU and
//! the fused QKV+RoPE batch kernel. Cut verbatim from `simt_nn.rs`.

use super::*;

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// THE BATCHED PREFILL FAMILY — the stages that kept the activation on the HOST
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
// `[PREFILL-GPU]` measured the wall: the batch-tiled DP4A GEMM reaches 20 953 tok/s standalone but
// only 1 316 tok/s wired, because only the PROJECTIONS were on the device — 73 % of the wired chunk
// was host int8 quantize + staged upload + readback + one queue drain per projection (96 per chunk).
// The lever named there is not a faster GEMM: it is keeping the activation ON the device across the
// whole layer stack, which needs every OTHER stage batched over positions as a device kernel too.
//
// These are those stages. Each is dispatched `(gx, B)` — **`gl_GlobalInvocationId.y` is the position
// in the chunk** — so one recorded command buffer walks the entire stack for a whole chunk and the
// host touches the stream exactly twice (one embedding upload, one hidden/logit readback).
//
// The activation chart they produce/consume is byte-identical to [`Kb::emit_quantize_from_shared`]'s
// (per-32-block absmax/127, round-half-away, deinterleaved `dpw` word order) — the same chart
// `simt::gemv_dp4a_batch_module` reads and the same one `cpu_matmat_f32_into` builds on the host,
// only never leaving VRAM.

/// Where [`Kb::quantize_at`] reads its `n` f32 values from: workgroup-shared (`var`, elem-ptr) —
/// the norm's output that never lands in global — or a global f32 SSBO at a base offset.
#[derive(Clone, Copy)]
enum QSrc {
    Shared(u32, u32),
    /// Shared with one pad float per 32 (`idx + idx/32`): the per-block walk reads stride 32, which
    /// on an unpadded array lands every lane on one bank.
    SharedPad(u32, u32),
    Global(u32, u32),
}

impl QSrc {
    fn load(self, k: &mut Kb, idx: u32) -> u32 {
        match self {
            QSrc::Shared(v, p) => k.shared_load(v, p, idx),
            QSrc::SharedPad(v, p) => {
                let c32 = k.cu(32);
                let q = k.udiv(idx, c32);
                let pi = k.iadd(idx, q);
                k.shared_load(v, p, pi)
            }
            QSrc::Global(b, off) => {
                let i = k.iadd(off, idx);
                k.load_f(b, i)
            }
        }
    }
}

impl Kb {
    /// **The DP4A activation chart, at an offset** — [`Kb::emit_quantize_from_shared`] with a batch
    /// row's base offsets (`qoff` into the word stream, `boff` into the per-32-block scale/sum
    /// streams) and a choice of source. Same absmax/127 scale, same round-half-away code, same
    /// deinterleaved word order, so a row quantized here is bit-identical to one the host quantizer
    /// in `servefold::gpu` produced for the same f32 input.
    #[allow(clippy::too_many_arguments)]
    fn quantize_at(
        &mut self,
        src: QSrc,
        xq: u32,
        qoff: u32,
        scales: u32,
        xs: u32,
        boff: u32,
        n: u32,
        dpw: u32,
        tid: u32,
        wg: u32,
    ) {
        let sint = self.sint_ty();
        // the per-block scales cross threads (written by block owners, read by every word lane):
        // they travel through workgroup memory, which the barrier below orders.
        let (ssc, sscp) = self.shared_f32(n / 32);
        let cwg = self.cu(wg);
        let cblk = self.cu(n / 32);
        self.loop_step(tid, cblk, cwg, |k, b| {
            let c32 = k.cu(32);
            let base = k.imul(b, c32);
            let mxv = k.fvar_f();
            let sumv = k.fvar_f();
            let tiny = k.cf(1.0e-12);
            k.store_var(mxv, tiny);
            let z = k.cf(0.0);
            k.store_var(sumv, z);
            let c0u = k.cu(0);
            let c32b = k.cu(32);
            k.loop_u(c0u, c32b, |k, j| {
                let idx = k.iadd(base, j);
                let v = src.load(k, idx);
                let f = k.float;
                let gl = k.glsl;
                let av = k.e(OP_EXT_INST, f, &[gl, GLSL_FABS, v]);
                let m0 = k.load_var_f(mxv);
                let m1 = k.ext2(GLSL_FMAX, m0, av);
                k.store_var(mxv, m1);
                let s0 = k.load_var_f(sumv);
                let s1 = k.fadd(s0, v);
                k.store_var(sumv, s1);
            });
            let mx = k.load_var_f(mxv);
            let inv127 = k.cf(1.0 / 127.0);
            let sc = k.fmul(mx, inv127);
            let si = k.iadd(boff, b);
            k.store_f(scales, si, sc);
            k.shared_store(ssc, sscp, b, sc);
            let sm = k.load_var_f(sumv);
            k.store_f(xs, si, sm);
        });
        self.wg_barrier();
        let cwords = self.cu(n / 4);
        self.loop_step(tid, cwords, cwg, |k, c| {
            let cdpw = k.cu(dpw);
            let cc = k.udiv(c, cdpw);
            let ll = k.umod(c, cdpw);
            let c4d = k.cu(4 * dpw);
            let base0 = k.imul(cc, c4d);
            let base = k.iadd(base0, ll);
            let c32 = k.cu(32);
            let kb = k.udiv(base0, c32);
            let sx = k.shared_load(ssc, sscp, kb);
            let word = k.fvar_u();
            let z = k.cu(0);
            k.store_var(word, z);
            let c0u = k.cu(0);
            let four = k.cu(4);
            k.loop_u(c0u, four, |k, b| {
                let bd = k.imul(b, cdpw);
                let idx = k.iadd(base, bd);
                let v = src.load(k, idx);
                let byte = k.int8_code(v, sx);
                let u = k.uint;
                let c8 = k.cu(8);
                let sh = k.imul(b, c8);
                let shifted = k.e(OP_SHIFT_LEFT_LOGICAL, u, &[byte, sh]);
                let w0 = k.load_var_u(word);
                let w1 = k.e(OP_BITWISE_OR, u, &[w0, shifted]);
                k.store_var(word, w1);
            });
            let wv = k.load_var_u(word);
            let psu = k.ptr_sb_u;
            let c0b = k.cu(0);
            let ci = k.iadd(qoff, c);
            let p = k.e(OP_ACCESS_CHAIN, psu, &[xq, c0b, ci]);
            k.ins(OP_STORE, &[p, wv]);
        });
        let _ = sint;
    }

    /// `round(v/scale)` clamped to ±127, returned as `(byte, signed bits)` — the packed code AND the
    /// two's-complement i32 the superblock chart's `Σq` accumulates. One conversion feeds both, so
    /// the sum costs a wrapping `OpIAdd` and not a second quantize.
    fn int8_code_pair(&mut self, v: u32, scale: u32) -> (u32, u32) {
        let q = self.fdiv(v, scale);
        let half = self.cf(0.5);
        let nhalf = self.cf(-0.5);
        let zf = self.cf(0.0);
        let bt = self.bool_t;
        let pos = self.e(OP_FORD_GREATER_THAN, bt, &[q, zf]);
        let f = self.float;
        let adj = self.e(OP_SELECT, f, &[pos, half, nhalf]);
        let qa = self.fadd(q, adj);
        let hi = self.cf(127.0);
        let lo = self.cf(-127.0);
        let g = self.glsl;
        let q1 = self.e(OP_EXT_INST, f, &[g, GLSL_FMIN, qa, hi]);
        let q2 = self.e(OP_EXT_INST, f, &[g, GLSL_FMAX2, q1, lo]);
        let s = self.sint_ty();
        let si = self.e(OP_CONVERT_F_TO_S, s, &[q2]);
        let u = self.uint;
        let sb = self.e(OP_BITCAST, u, &[si]);
        let cff = self.cu(0xFF);
        let byte = self.e(OP_BITWISE_AND, u, &[sb, cff]);
        (byte, sb)
    }

    /// **The SUPERBLOCK activation chart** — what `simt::gemv_dp4a_sb_module` reads, and the only
    /// thing standing between that kernel's measured 61 010 tok/s and the wired stack.
    ///
    /// Against [`Kb::quantize_at`]'s per-32 chart, three things change and nothing else does:
    ///   * ONE absmax/127 scale per **256** values (`xs[boff + sb]`), not per 32;
    ///   * `Σq` per superblock as an **integer** (`ys[boff + sb]`, i32 bits), not `Σx` as a float —
    ///     the kernel's `Σ w·x = s·sx·(Σ code·q − Σ q)` needs the offset inside the integer ring
    ///     (`TernaryDP4A.ternary_dot_two_mask`: the `code = 1` monad lanes cancel against `Σq`);
    ///   * the codes are in **natural** word order (word `c` holds values `4c..4c+4`), not the
    ///     `dpw`-deinterleaved order, because the superblock weight layout already pairs weight word
    ///     `8·unit + q` with activation word `32·unit + 8·p + q`.
    ///
    /// It keeps the per-32 kernel's PARALLELISM, which is the reason it is written in four phases
    /// rather than one thread per superblock: `n/32` threads do the absmax and the packing (28 of
    /// them at `n = 896`, 152 at `n = 4864`), and only the two tiny cross-block folds — 8 partials
    /// per superblock — run at `nsb` width. A one-thread-per-superblock quantizer would have left 4
    /// of 256 lanes doing 512 serial iterations each.
    ///
    /// `pmax`/`psum` are `(var, elem-ptr)` shared arrays of `n/32` entries each (224 B at `n = 896`).
    /// The last superblock may be PARTIAL (`n = 896` is 3 full ones plus a 128-wide tail, exactly the
    /// odd unit `gemv_dp4a_sb_module` emits straight-line), so both folds run to
    /// `min(8·sb + 8, n/32)`.
    #[allow(clippy::too_many_arguments)]
    fn quantize_sb_at(
        &mut self,
        src: QSrc,
        xq: u32,
        qoff: u32,
        xs: u32,
        ys: u32,
        boff: u32,
        n: u32,
        tid: u32,
        wg: u32,
        pmax: (u32, u32),
        psum: (u32, u32),
    ) {
        let nblk = n / 32;
        let nsb = n.div_ceil(256);
        let cwg = self.cu(wg);
        let cblk = self.cu(nblk);
        let cnsb = self.cu(nsb);
        let cnblk = self.cu(nblk);
        // the per-superblock scale crosses threads (written by the superblock owner in phase B,
        // read by every block lane in phase C): it travels through workgroup memory. Reaching it
        // through the `xs` SSBO is the cross-invocation storage race — under GLSL450 a
        // WorkgroupMemory barrier does not make a storage-buffer write visible cross-invocation.
        let (sxs, sxsp) = self.shared_f32(nsb);
        // ── phase A: absmax over each 32-block, one thread per block
        self.loop_step(tid, cblk, cwg, |k, b| {
            let c32 = k.cu(32);
            let base = k.imul(b, c32);
            let mxv = k.fvar_f();
            let z = k.cf(0.0);
            k.store_var(mxv, z);
            let c0u = k.cu(0);
            let c32b = k.cu(32);
            k.loop_u(c0u, c32b, |k, j| {
                let idx = k.iadd(base, j);
                let v = src.load(k, idx);
                let av = k.ext1(GLSL_FABS, v);
                let m0 = k.load_var_f(mxv);
                let m1 = k.ext2(GLSL_FMAX, m0, av);
                k.store_var(mxv, m1);
            });
            let mx = k.load_var_f(mxv);
            k.shared_store(pmax.0, pmax.1, b, mx);
        });
        self.wg_barrier();
        // ── phase B: fold the ≤8 block maxima of a superblock into its ONE scale
        self.loop_step(tid, cnsb, cwg, |k, sb| {
            let c8 = k.cu(8);
            let lo = k.imul(sb, c8);
            let hi0 = k.iadd(lo, c8);
            let hi = k.umin(hi0, cnblk);
            let mxv = k.fvar_f();
            let z = k.cf(0.0);
            k.store_var(mxv, z);
            k.loop_u(lo, hi, |k, j| {
                let v = k.shared_load(pmax.0, pmax.1, j);
                let m0 = k.load_var_f(mxv);
                let m1 = k.ext2(GLSL_FMAX, m0, v);
                k.store_var(mxv, m1);
            });
            let mx = k.load_var_f(mxv);
            // `quantize_act_sb`'s rule exactly: absmax/127, or 1.0 on an all-zero superblock
            let inv127 = k.cf(1.0 / 127.0);
            let sc0 = k.fmul(mx, inv127);
            let zf = k.cf(0.0);
            let one = k.cf(1.0);
            let bt = k.bool_t;
            let pos = k.e(OP_FORD_GREATER_THAN, bt, &[mx, zf]);
            let f = k.float;
            let sc = k.e(OP_SELECT, f, &[pos, sc0, one]);
            let si = k.iadd(boff, sb);
            k.store_f(xs, si, sc);
            k.shared_store(sxs, sxsp, sb, sc);
        });
        self.wg_barrier();
        // ── phase C: pack the 8 natural-order words of each 32-block, and its partial Σq
        self.loop_step(tid, cblk, cwg, |k, b| {
            let c8 = k.cu(8);
            let sbi0 = k.udiv(b, c8);
            let sx = k.shared_load(sxs, sxsp, sbi0);
            let c32 = k.cu(32);
            let base = k.imul(b, c32);
            let wbase0 = k.imul(b, c8);
            let wbase = k.iadd(qoff, wbase0);
            let acc = k.fvar_u();
            let z = k.cu(0);
            k.store_var(acc, z);
            let c0u = k.cu(0);
            let c8b = k.cu(8);
            k.loop_u(c0u, c8b, |k, w| {
                let c4 = k.cu(4);
                let e0 = k.imul(w, c4);
                let ebase = k.iadd(base, e0);
                let word = k.fvar_u();
                let zz = k.cu(0);
                k.store_var(word, zz);
                let c0t = k.cu(0);
                let c4b = k.cu(4);
                k.loop_u(c0t, c4b, |k, t| {
                    let idx = k.iadd(ebase, t);
                    let v = src.load(k, idx);
                    let (byte, sgn) = k.int8_code_pair(v, sx);
                    let u = k.uint;
                    let c8s = k.cu(8);
                    let sh = k.imul(t, c8s);
                    let shifted = k.e(OP_SHIFT_LEFT_LOGICAL, u, &[byte, sh]);
                    let w0 = k.load_var_u(word);
                    let w1 = k.e(OP_BITWISE_OR, u, &[w0, shifted]);
                    k.store_var(word, w1);
                    let a0 = k.load_var_u(acc);
                    let a1 = k.iadd(a0, sgn);
                    k.store_var(acc, a1);
                });
                let wv = k.load_var_u(word);
                let wi = k.iadd(wbase, w);
                k.store_u(xq, wi, wv);
            });
            let av = k.load_var_u(acc);
            k.shared_store_u(psum.0, psum.1, b, av);
        });
        self.wg_barrier();
        // ── phase D: fold the ≤8 block sums into the superblock's Σq (i32 bits, wrapping)
        self.loop_step(tid, cnsb, cwg, |k, sb| {
            let c8 = k.cu(8);
            let lo = k.imul(sb, c8);
            let hi0 = k.iadd(lo, c8);
            let hi = k.umin(hi0, cnblk);
            let acc = k.fvar_u();
            let z = k.cu(0);
            k.store_var(acc, z);
            k.loop_u(lo, hi, |k, j| {
                let v = k.shared_load_u(psum.0, psum.1, j);
                let a0 = k.load_var_u(acc);
                let a1 = k.iadd(a0, v);
                k.store_var(acc, a1);
            });
            let av = k.load_var_u(acc);
            let si = k.iadd(boff, sb);
            k.store_u(ys, si, av);
        });
    }

    /// `round(v/scale)` clamped to ±127 as an **f32** (not the packed byte) — the int8 KV code the
    /// attention kernel reads back as a float. Same map as [`Kb::int8_code`] and as `CpuState::
    /// kv_write`'s `(x/sc).round().clamp(-127,127)`: add ±½ then truncate toward zero.
    fn int8_code_f(&mut self, v: u32, scale: u32) -> u32 {
        let q = self.fdiv(v, scale);
        let half = self.cf(0.5);
        let nhalf = self.cf(-0.5);
        let zf = self.cf(0.0);
        let bt = self.bool_t;
        let pos = self.e(OP_FORD_GREATER_THAN, bt, &[q, zf]);
        let f = self.float;
        let adj = self.e(OP_SELECT, f, &[pos, half, nhalf]);
        let qa = self.fadd(q, adj);
        let hi = self.cf(127.0);
        let lo = self.cf(-127.0);
        let g = self.glsl;
        let q1 = self.e(OP_EXT_INST, f, &[g, GLSL_FMIN, qa, hi]);
        let q2 = self.e(OP_EXT_INST, f, &[g, GLSL_FMAX2, q1, lo]);
        let s = self.sint_ty();
        let si = self.e(OP_CONVERT_F_TO_S, s, &[q2]);
        self.e(OP_CONVERT_S_TO_F, f, &[si])
    }
}

/// **Batched RMSNorm + residual + activation-quantize — one workgroup per POSITION.**
///
/// `residual = true` first folds the previous projection's output into the residual stream in place
/// (`x[i] += res[i]`, the `x += o` / `x += down` the CPU stack does between norms), then normalizes
/// with `gain`, writes the f32 normalized vector to `h`, and emits the DP4A activation chart the
/// next projection consumes — all without the value ever crossing PCIe. That fusion is the point:
/// the host quantize was 73 % of `[PREFILL-GPU]`'s wired chunk precisely because the norm's output
/// lived on the host.
///
/// Bindings: 0 = x (f32 residual stream, `n` per position, read-write), 1 = gain, 2 = res (f32,
/// `n` per position; bound but unread when `residual = false`), 3 = xq (u32 chart words),
/// 4 = scales (f32 per 32-block), 5 = xs (f32 `Σx` per 32-block), 6 = h (f32 normalized out).
/// Dispatch `(1, B)` at LocalSize 256; `gl_GlobalInvocationId.y` is the position.
///
/// `ks > 1` (requires `residual`): binding 2 holds the `ks` split-K partial planes of the down GEMM,
/// `[ks][ceil(B/64)*64][n]` floats, and binding 7 is the prefill `params` (`params[1]` = `B`). The
/// residual prologue forms `res[idx] = (p0 + p1) + ... + p(ks-1)` in registers, a left fold of f32
/// adds in plane order (`splitk_reduce.comp`'s order), then `x += res` as for `ks = 1`.
pub fn rmsnorm_batch_q_module(n: u32, eps: f32, dpw: u32, residual: bool, ks: u32) -> Vec<u32> {
    rmsnorm_batch_q_impl(n, eps, dpw, residual, true, ks)
}

/// [`rmsnorm_batch_q_module`] with binding 6 (`h`) **not written**. The resident prefill path binds
/// `hbuf` to every intermediate norm (norm1/norm2 of each layer) and reads it from no later node;
/// only the final norm's `hout` is read back to the host (`prefill.rs`, the last-row hidden copy).
/// Skipping the dead store removes `n·4` bytes of L2 write traffic per node with the chart
/// (`xq`/`scales`/`xs`) and the shared `nrm` path arithmetically untouched. The caller must not read
/// binding 6 when it uses this entry.
pub fn rmsnorm_batch_q_noh_module(n: u32, eps: f32, dpw: u32, residual: bool, ks: u32) -> Vec<u32> {
    rmsnorm_batch_q_impl(n, eps, dpw, residual, false, ks)
}

/// Shared body of [`rmsnorm_batch_q_module`] / [`rmsnorm_batch_q_noh_module`]. `emit_h` selects
/// whether binding 6 is written; `ks` is the split-K plane count of the residual (1 = one plane).
fn rmsnorm_batch_q_impl(
    n: u32,
    eps: f32,
    dpw: u32,
    residual: bool,
    emit_h: bool,
    ks: u32,
) -> Vec<u32> {
    const WG: u32 = 256;
    const SPLITK_BN: u32 = 64; // `coopmat_gemm.comp` BN: the split kernel pads tokens to this
    assert!(n % 32 == 0, "the activation chart is per-32-block");
    assert!(
        ks >= 1 && (ks == 1 || residual),
        "split planes feed the residual"
    );
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let gain = k.ssbo_f(1);
    let res = k.ssbo_f(2);
    let xq = k.ssbo_u(3);
    let scales = k.ssbo_f(4);
    let xs = k.ssbo_f(5);
    let hout = k.ssbo_f(6);
    let params = (ks > 1).then(|| k.ssbo_u(7));
    let (red, redp) = k.shared_f32(WG);
    let (nrm, nrmp) = k.shared_f32(n);
    let tid = k.gid_comp(0);
    let i = k.gid_comp(1);
    let cn = k.cu(n);
    let cwg = k.cu(WG);
    let xoff = k.imul(i, cn);
    // plane stride in floats: `ceil(B/64)*64 * n` (the split kernel's padded token count, as
    // `gl_NumWorkGroups.y * BN` is in `splitk_reduce.comp`), `B` read from `params[1]`
    let plane = params.map(|pb| {
        let c1 = k.cu(1);
        let tokens = k.load_u(pb, c1);
        let cbn = k.cu(SPLITK_BN);
        let cbn1 = k.cu(SPLITK_BN - 1);
        let up = k.iadd(tokens, cbn1);
        let tiles = k.udiv(up, cbn);
        let padded = k.imul(tiles, cbn);
        k.imul(padded, cn)
    });
    let cq = k.cu(n / 4);
    let qoff = k.imul(i, cq);
    let cb = k.cu(n / 32);
    let boff = k.imul(i, cb);
    // ── one pass over the row: every global load (gain, x, residual planes) of every strided trip is
    // issued before any is consumed, and the row stays in registers (`sv`) through the sumsq and the
    // normalise. Each lane still owns elements `tid + t·WG` and adds their squares in ascending `t`,
    // so Σx² (and with it every value the chart is built from) is bit-identical to the loop form.
    // A trip with `tid + t·WG ≥ n` (only the last, when `n % WG != 0`) loads under a guard into a
    // variable preset to 0.0, which adds exactly 0.0 to the accumulator and is never stored.
    let trips = n.div_ceil(WG);
    let tail = n % WG != 0;
    let mut gvs = Vec::new();
    let mut xvs = Vec::new();
    let mut rvs: Vec<Vec<u32>> = Vec::new();
    for t in 0..trips {
        let j = k.strided_idx(tid, t, WG);
        let idx = k.iadd(xoff, j);
        let guarded = tail && t == trips - 1;
        if !guarded {
            gvs.push(k.load_f(gain, j));
            xvs.push(k.load_f(x, idx));
            let mut planes = Vec::new();
            if residual {
                planes.push(k.load_f(res, idx));
                if let Some(plane) = plane {
                    for sp in 1..ks {
                        let cs = k.cu(sp);
                        let po = k.imul(cs, plane);
                        let pidx = k.iadd(po, idx);
                        planes.push(k.load_f(res, pidx));
                    }
                }
            }
            rvs.push(planes);
        } else {
            let gvar = k.fvar_f();
            let xvar = k.fvar_f();
            let z = k.cf(0.0);
            k.store_var(gvar, z);
            k.store_var(xvar, z);
            let rvars: Vec<u32> = (0..if residual { ks } else { 0 })
                .map(|_| {
                    let v = k.fvar_f();
                    k.store_var(v, z);
                    v
                })
                .collect();
            let ok = k.ult(j, cn);
            k.iff(ok, |k| {
                let g = k.load_f(gain, j);
                k.store_var(gvar, g);
                let xv = k.load_f(x, idx);
                k.store_var(xvar, xv);
                for (sp, rvar) in rvars.iter().enumerate() {
                    let pidx = match plane {
                        Some(plane) if sp > 0 => {
                            let cs = k.cu(sp as u32);
                            let po = k.imul(cs, plane);
                            k.iadd(po, idx)
                        }
                        _ => idx,
                    };
                    let pv = k.load_f(res, pidx);
                    k.store_var(*rvar, pv);
                }
            });
            gvs.push(k.load_var_f(gvar));
            xvs.push(k.load_var_f(xvar));
            rvs.push(rvars.iter().map(|v| k.load_var_f(*v)).collect());
        }
    }
    let acc = k.fvar_f();
    let zero = k.cf(0.0);
    k.store_var(acc, zero);
    let mut svs = Vec::new();
    for t in 0..trips as usize {
        let mut sv = xvs[t];
        if residual {
            let mut rv = rvs[t][0];
            for pv in &rvs[t][1..] {
                rv = k.fadd(rv, *pv);
            }
            sv = k.fadd(sv, rv);
            let j = k.strided_idx(tid, t as u32, WG);
            let idx = k.iadd(xoff, j);
            if tail && t as u32 == trips - 1 {
                let ok = k.ult(j, cn);
                k.iff(ok, |k| k.store_f(x, idx, sv));
            } else {
                k.store_f(x, idx, sv);
            }
        }
        svs.push(sv);
        let sq = k.fmul(sv, sv);
        let a0 = k.load_var_f(acc);
        let a1 = k.fadd(a0, sq);
        k.store_var(acc, a1);
    }
    let af = k.load_var_f(acc);
    k.shared_store(red, redp, tid, af);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, tid, false);
    let c0 = k.cu(0);
    let tot = k.shared_load(red, redp, c0);
    let invn = k.cf(1.0 / n as f32);
    let mean = k.fmul(tot, invn);
    let ce = k.cf(eps);
    let me = k.fadd(mean, ce);
    let inv = k.ext1(GLSL_INVERSE_SQRT, me);
    for t in 0..trips as usize {
        let j = k.strided_idx(tid, t as u32, WG);
        let sc = k.fmul(svs[t], inv);
        let v = k.fmul(sc, gvs[t]);
        let idx = k.iadd(xoff, j);
        if tail && t as u32 == trips - 1 {
            let ok = k.ult(j, cn);
            k.iff(ok, |k| {
                k.shared_store(nrm, nrmp, j, v);
                if emit_h {
                    k.store_f(hout, idx, v);
                }
            });
        } else {
            k.shared_store(nrm, nrmp, j, v);
            if emit_h {
                k.store_f(hout, idx, v);
            }
        }
    }
    k.wg_barrier();
    k.quantize_at(
        QSrc::Shared(nrm, nrmp),
        xq,
        qoff,
        scales,
        xs,
        boff,
        n,
        dpw,
        tid,
        WG,
    );
    k.finish(WG)
}

/// **Batched activation quantize from a global f32 buffer** — the attention output before the
/// o-projection and the SwiGLU output before the down-projection, neither of which comes out of a
/// norm. Bindings: 0 = src (f32, `n` per position), 1 = xq, 2 = scales, 3 = xs. Dispatch `(1, B)`
/// at LocalSize 256.
pub fn quantize_batch_module(n: u32, dpw: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(n % 32 == 0, "the activation chart is per-32-block");
    let mut k = Kb::new();
    let src = k.ssbo_f(0);
    let xq = k.ssbo_u(1);
    let scales = k.ssbo_f(2);
    let xs = k.ssbo_f(3);
    let tid = k.gid_comp(0);
    let i = k.gid_comp(1);
    let cn = k.cu(n);
    let soff = k.imul(i, cn);
    let cq = k.cu(n / 4);
    let qoff = k.imul(i, cq);
    let cb = k.cu(n / 32);
    let boff = k.imul(i, cb);
    k.quantize_at(
        QSrc::Global(src, soff),
        xq,
        qoff,
        scales,
        xs,
        boff,
        n,
        dpw,
        tid,
        WG,
    );
    k.finish(WG)
}

/// **Batched RMSNorm + residual + SUPERBLOCK activation-quantize.** [`rmsnorm_batch_q_module`] with
/// the per-256 chart `simt::gemv_dp4a_sb_module` reads (see [`Kb::quantize_sb_at`]): the norm, the
/// residual fold and the f32 write are the SAME code and the same numbers; only the activation face
/// the next projection consumes changes.
///
/// Bindings: 0 = x (f32 residual stream, read-write), 1 = gain, 2 = res, 3 = xq (u32 codes, natural
/// order), 4 = xs (f32 absmax/127 per 256), 5 = ys (u32 `Σq` bits per 256), 6 = h (f32 normalized).
/// Dispatch `(1, B)` at LocalSize 256.
pub fn rmsnorm_batch_qsb_module(n: u32, eps: f32, residual: bool) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(
        n % 32 == 0,
        "the activation chart folds 32-blocks into superblocks"
    );
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let gain = k.ssbo_f(1);
    let res = k.ssbo_f(2);
    let xq = k.ssbo_u(3);
    let xs = k.ssbo_f(4);
    let ys = k.ssbo_u(5);
    let hout = k.ssbo_f(6);
    let (red, redp) = k.shared_f32(WG);
    let (nrm, nrmp) = k.shared_f32(n);
    let pmax = k.shared_f32(n / 32);
    let psum = k.shared_u32(n / 32);
    let tid = k.gid_comp(0);
    let i = k.gid_comp(1);
    let cn = k.cu(n);
    let cwg = k.cu(WG);
    let xoff = k.imul(i, cn);
    let cq = k.cu(n / 4);
    let qoff = k.imul(i, cq);
    let cb = k.cu(n.div_ceil(256));
    let boff = k.imul(i, cb);
    if residual {
        k.loop_step(tid, cn, cwg, |k, j| {
            let idx = k.iadd(xoff, j);
            let xv = k.load_f(x, idx);
            let rv = k.load_f(res, idx);
            let s = k.fadd(xv, rv);
            k.store_f(x, idx, s);
        });
    }
    let acc = k.fvar_f();
    let zero = k.cf(0.0);
    k.store_var(acc, zero);
    k.loop_step(tid, cn, cwg, |k, j| {
        let idx = k.iadd(xoff, j);
        let v = k.load_f(x, idx);
        let sq = k.fmul(v, v);
        let a0 = k.load_var_f(acc);
        let a1 = k.fadd(a0, sq);
        k.store_var(acc, a1);
    });
    let af = k.load_var_f(acc);
    k.shared_store(red, redp, tid, af);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, tid, false);
    let c0 = k.cu(0);
    let tot = k.shared_load(red, redp, c0);
    let invn = k.cf(1.0 / n as f32);
    let mean = k.fmul(tot, invn);
    let ce = k.cf(eps);
    let me = k.fadd(mean, ce);
    let inv = k.ext1(GLSL_INVERSE_SQRT, me);
    k.loop_step(tid, cn, cwg, |k, j| {
        let idx = k.iadd(xoff, j);
        let xv = k.load_f(x, idx);
        let gv = k.load_f(gain, j);
        let sc = k.fmul(xv, inv);
        let v = k.fmul(sc, gv);
        k.shared_store(nrm, nrmp, j, v);
        k.store_f(hout, idx, v);
    });
    k.wg_barrier();
    k.quantize_sb_at(
        QSrc::Shared(nrm, nrmp),
        xq,
        qoff,
        xs,
        ys,
        boff,
        n,
        tid,
        WG,
        pmax,
        psum,
    );
    k.finish(WG)
}

/// **Batched SUPERBLOCK activation quantize from a global f32 buffer** — the attention output before
/// the o-projection and the SwiGLU output before the down-projection. Bindings: 0 = src (f32),
/// 1 = xq, 2 = xs, 3 = ys. Dispatch `(1, B)` at LocalSize 256.
pub fn quantize_batch_sb_module(n: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(
        n % 32 == 0,
        "the activation chart folds 32-blocks into superblocks"
    );
    let mut k = Kb::new();
    let src = k.ssbo_f(0);
    let xq = k.ssbo_u(1);
    let xs = k.ssbo_f(2);
    let ys = k.ssbo_u(3);
    let pmax = k.shared_f32(n / 32);
    let psum = k.shared_u32(n / 32);
    let tid = k.gid_comp(0);
    let i = k.gid_comp(1);
    let cn = k.cu(n);
    let soff = k.imul(i, cn);
    let cq = k.cu(n / 4);
    let qoff = k.imul(i, cq);
    let cb = k.cu(n.div_ceil(256));
    let boff = k.imul(i, cb);
    k.quantize_sb_at(
        QSrc::Global(src, soff),
        xq,
        qoff,
        xs,
        ys,
        boff,
        n,
        tid,
        WG,
        pmax,
        psum,
    );
    k.finish(WG)
}

/// [`last_row_gather_module`] over the SUPERBLOCK chart — the same mandala-closed copy of the
/// chunk's last quantized row into row 0, with `nsb` per-256 entries instead of `n/32` per-32 ones
/// and `Σq` (u32 bits) in place of `Σx` (f32). Bindings: 0 = xq src, 1 = xs src, 2 = ys src (u32),
/// 3 = xq dst, 4 = xs dst, 5 = ys dst, 6 = params (`params[1]` = `n`). Dispatch `(1, 1)`.
pub fn last_row_gather_sb_module(qwords: u32, nsb: u32, qmask: u32, bmask: u32) -> Vec<u32> {
    const WG: u32 = 256;
    let mut k = Kb::new();
    let qsrc = k.ssbo_u(0);
    let xssrc = k.ssbo_f(1);
    let yssrc = k.ssbo_u(2);
    let qdst = k.ssbo_u(3);
    let xsdst = k.ssbo_f(4);
    let ysdst = k.ssbo_u(5);
    let params = k.ssbo_u(6);
    let tid = k.gid_comp(0);
    let c1 = k.cu(1);
    let cwg = k.cu(WG);
    let n = k.load_u(params, c1);
    let r = k.e(OP_ISUB, k.uint, &[n, c1]);
    let cq = k.cu(qwords);
    let cb = k.cu(nsb);
    let qbase = k.imul(r, cq);
    let bbase = k.imul(r, cb);
    k.loop_step(tid, cq, cwg, |k, j| {
        let s0 = k.iadd(qbase, j);
        let s = k.mandala(s0, qmask);
        let v = k.load_u(qsrc, s);
        k.store_u(qdst, j, v);
    });
    k.loop_step(tid, cb, cwg, |k, j| {
        let s0 = k.iadd(bbase, j);
        let s = k.mandala(s0, bmask);
        let sc = k.load_f(xssrc, s);
        k.store_f(xsdst, j, sc);
        let sm = k.load_u(yssrc, s);
        k.store_u(ysdst, j, sm);
    });
    k.finish(WG)
}

/// [`rmsnorm_batch_qsb_module`] as bytes.
pub fn rmsnorm_batch_qsb_bytes(n: u32, eps: f32, residual: bool) -> Vec<u8> {
    bytes(rmsnorm_batch_qsb_module(n, eps, residual))
}
/// [`quantize_batch_sb_module`] as bytes.
pub fn quantize_batch_sb_bytes(n: u32) -> Vec<u8> {
    bytes(quantize_batch_sb_module(n))
}
/// [`last_row_gather_sb_module`] as bytes.
pub fn last_row_gather_sb_bytes(qwords: u32, nsb: u32, qmask: u32, bmask: u32) -> Vec<u8> {
    bytes(last_row_gather_sb_module(qwords, nsb, qmask, bmask))
}

/// **Gather the chunk's LAST quantized row into row 0** — the whole of the prefill lm head's input
/// when nobody is scoring the interior positions.
///
/// During prefill only the last position's logits are needed, to sample the first generated token;
/// the other `n−1` rows exist for the harness's bits/token and the cloze ranking. Rather than run
/// the lm head over all `n` rows and push `n·vocab` f32 across PCIe, this copies row `n−1` of the
/// quantized activation (`xq`, its per-32-block `scales` and `xs`) to row 0 of a small destination,
/// so the SAME lm GEMM pipeline over ONE tile produces that row. At b = 640 on qwen2.5-0.5b that is
/// a 371 MiB host write replaced by a 19 MiB one, and 640 rows of GEMM replaced by 32.
///
/// **Every load is a mandala access.** The source index is closed by `&&& (2^k − 1)` against the
/// buffer's power-of-two capacity ([`Kb::mandala`]), so `n` — which arrives from a host-written
/// params buffer and is therefore untrusted at this boundary — cannot form an address outside the
/// region: `masked_load_never_oob`. The destination indices are the loop counters, bounded by
/// construction. The mask is the null cell's NB select, so the bound and the addressing are one
/// instruction, not a branch (`MandalaMemory` §3).
///
/// `qwords` = `hidden/4` u32 per row, `blocks` = `hidden/32` scale entries per row; `qmask`/`bmask`
/// are `2^k − 1` for the source capacities. Bindings: 0 = xq src (u32), 1 = scales src (f32),
/// 2 = xs src (f32), 3 = xq dst, 4 = scales dst, 5 = xs dst, 6 = params (u32, `params[1]` = `n`).
/// Dispatch `(1, 1)` at LocalSize 256.
pub fn last_row_gather_module(qwords: u32, blocks: u32, qmask: u32, bmask: u32) -> Vec<u32> {
    const WG: u32 = 256;
    let mut k = Kb::new();
    let qsrc = k.ssbo_u(0);
    let scsrc = k.ssbo_f(1);
    let smsrc = k.ssbo_f(2);
    let qdst = k.ssbo_u(3);
    let scdst = k.ssbo_f(4);
    let smdst = k.ssbo_f(5);
    let params = k.ssbo_u(6);
    let tid = k.gid_comp(0);
    let c1 = k.cu(1);
    let cwg = k.cu(WG);
    let n = k.load_u(params, c1);
    // the last position of the chunk. `run` refuses n == 0, and the mandala mask below makes the
    // wrap harmless in any case — there is no key that escapes the petal.
    let r = k.e(OP_ISUB, k.uint, &[n, c1]);
    let cq = k.cu(qwords);
    let cb = k.cu(blocks);
    let qbase = k.imul(r, cq);
    let bbase = k.imul(r, cb);
    k.loop_step(tid, cq, cwg, |k, j| {
        let s0 = k.iadd(qbase, j);
        let s = k.mandala(s0, qmask);
        let v = k.load_u(qsrc, s);
        k.store_u(qdst, j, v);
    });
    k.loop_step(tid, cb, cwg, |k, j| {
        let s0 = k.iadd(bbase, j);
        let s = k.mandala(s0, bmask);
        let sc = k.load_f(scsrc, s);
        k.store_f(scdst, j, sc);
        let sm = k.load_f(smsrc, s);
        k.store_f(smdst, j, sm);
    });
    k.finish(WG)
}

/// [`last_row_gather_module`] as bytes.
pub fn last_row_gather_bytes(qwords: u32, blocks: u32, qmask: u32, bmask: u32) -> Vec<u8> {
    bytes(last_row_gather_module(qwords, blocks, qmask, bmask))
}

/// **Batched SwiGLU + activation-quantize, one workgroup per position** — `a[j] = silu(gate[j])·up[j]`
/// staged in workgroup memory, then the DP4A chart of [`Kb::quantize_at`] built from it. Replaces
/// the pair (SwiGLU to a global `acts` buffer, then [`quantize_batch_module`] over it): `a` is
/// bit-identical (it still passes through memory, so no multiply-add contraction reaches the
/// quantizer's `Σx`), and never crosses global memory.
///
/// Bindings: 0 = gu (f32, `2·dff` per position: gate then up), 1 = xq, 2 = scales, 3 = xs.
/// Dispatch `(1, B)` at LocalSize 256.
///
/// Cost per workgroup: `dff/256` iterations of 2 loads + 1 exp + 1 div, `dff/32` serial 32-element
/// block walks, `dff/4` words; shared `4·(dff + dff/32 + dff/32)` bytes (20.2 KB at `dff = 4864`).
pub fn swiglu_quant_batch_module(dff: u32, dpw: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(dff % 32 == 0, "the activation chart is per-32-block");
    let mut k = Kb::new();
    let gu = k.ssbo_f(0);
    let xq = k.ssbo_u(1);
    let scales = k.ssbo_f(2);
    let xs = k.ssbo_f(3);
    let (act, actp) = k.shared_f32(dff + dff / 32);
    let tid = k.gid_comp(0);
    let i = k.gid_comp(1);
    let cd = k.cu(dff);
    let c2d = k.cu(2 * dff);
    let gbase = k.imul(i, c2d);
    let cwg = k.cu(WG);
    k.loop_step(tid, cd, cwg, |k, j| {
        let gi = k.iadd(gbase, j);
        let g = k.load_f(gu, gi);
        let ui0 = k.iadd(gi, cd);
        let u = k.load_f(gu, ui0);
        let ng = k.e(OP_FNEG, k.float, &[g]);
        let ex = k.ext1(GLSL_EXP, ng);
        let one = k.cf(1.0);
        let den = k.fadd(one, ex);
        let sil = k.fdiv(g, den);
        let v = k.fmul(sil, u);
        let c32 = k.cu(32);
        let q = k.udiv(j, c32);
        let pj = k.iadd(j, q);
        k.shared_store(act, actp, pj, v);
    });
    k.wg_barrier();
    let cq = k.cu(dff / 4);
    let qoff = k.imul(i, cq);
    let cb = k.cu(dff / 32);
    let boff = k.imul(i, cb);
    k.quantize_at(
        QSrc::SharedPad(act, actp),
        xq,
        qoff,
        scales,
        xs,
        boff,
        dff,
        dpw,
        tid,
        WG,
    );
    k.finish(WG)
}

/// `subgroupMax(v)` — `OpGroupNonUniformFMax` over the whole subgroup, every lane receives it.
fn sg_fmax(k: &mut Kb, v: u32) -> u32 {
    const OP_GROUP_NON_UNIFORM_FMAX: u32 = 358;
    k.sg_arith = true;
    let f = k.float;
    let sc = k.cu(3);
    k.e(OP_GROUP_NON_UNIFORM_FMAX, f, &[sc, 0, v])
}

/// **Batched QKV epilogue — bias, QK-norm, RoPE and the KV append, one workgroup per (head-slot,
/// position).** The whole per-position tail of the attention block that used to sit on the host
/// between the QKV projection and the o-projection.
///
/// Slot layout mirrors the QKV row layout exactly, so `base = i·qkvm + slot·hd` addresses all three
/// faces with one expression: slots `[0, nh)` are Q heads (bias → q-norm → RoPE), `[nh, nh+nkv)` are
/// K heads (bias → k-norm → RoPE → K-cache append), `[nh+nkv, nh+2·nkv)` are V heads (bias → V-cache
/// append, unroped, as the CPU path does).
///
/// The KV codes are written as **f32-valued int8 codes** (`round(x/absmax·127)`, the identical map
/// `CpuState::kv_write` applies) rather than packed bytes: the CPU attention scores an f32 query
/// against int8 key codes, and holding the codes as f32 lets [`attn_prefill_module`] reproduce that
/// product exactly instead of introducing a second int8 quantization of the QUERY. They are written
/// twice — into the device cache (`kc`/`vc`, the shape the attention reads) and into a compact
/// per-chunk mirror (`kout`/`vout`, position-major) the host folds back into `CpuState` so decode
/// continues on the CPU from the state prefill built.
///
/// Bindings: 0 = qkv (f32, `qkvm` per position, read-write), 1 = bias (f32 `qkvm`; unread when
/// `bias = false`), 2 = qg, 3 = kg (f32 `hd` QK-norm gains; unread when `qknorm = false`),
/// 4 = kc (f32 `[(kvh·hd+d)·tmax + s]`), 5 = ks (f32 `[s·nkv + kvh]`), 6 = vc (f32
/// `[(kvh·tmax+s)·hd + d]`), 7 = vs (f32 `[s·nkv + kvh]`), 8 = params (`params[0]` = `pos0`),
/// 9 = kout (f32 `[(i·nkv+kvh)·hd + d]`), 10 = ksout (f32 `[i·nkv+kvh]`), 11 = vout, 12 = vsout,
/// 13 = kcq (u32 decode K `[(kvh·(hd/4)+w)·tmax + s]`, 4 int8 codes/word),
/// 14 = sk (f32 decode K scale `[kvh·tmax + s]`), 15 = vdec (f16 decode V `[(kvh·tmax+s)·hd + d]`,
/// `code·scale`). 13–15 are the [`super::prefill_glue::kv_seed_module`] decode layout, written here
/// from the same `code`/`sc` arithmetic so the standalone seed node is not needed; the K packing
/// gathers codes through the workgroup `red` array.
/// `nplanes` is the split-K plane count of the qkv GEMM (1 = one plane, read from binding 0). For
/// `nplanes > 1` binding 16 holds the partial planes `[nplanes][ceil(B/64)*64][qkvm]` (`B` =
/// `params[1]`) and the input is their left fold `((p0 + p1) + …)` in plane order
/// (`splitk_reduce.comp`'s order), taken before the bias; binding 0 is then only written (the
/// rotated Q the attention reads).
/// `rows` positions share one workgroup (lane `r·hd + d` is dim `d` of row `r`; arithmetic per row
/// unchanged): dispatch `(nh + 2·nkv, ceil(B/rows))` at LocalSize `hd·rows`; a row past `params[1]`
/// computes on row 0's data and stores nothing.
#[allow(clippy::too_many_arguments)]
pub fn qkv_rope_batch_module(
    nh: u32,
    nkv: u32,
    hd: u32,
    qkvm: u32,
    tmax: u32,
    theta: f32,
    eps: f32,
    bias: bool,
    qknorm: bool,
    nplanes: u32,
    rows: u32,
) -> Vec<u32> {
    const SPLITK_BN: u32 = 64; // `coopmat_gemm.comp` BN: the split kernel pads tokens to this
    assert!(nplanes >= 1, "at least one plane");
    assert!(rows >= 1, "at least one row per workgroup");
    assert!(
        hd % 64 == 0,
        "a lane owns the RoPE pair (d, d + hd/2); hd/2 is a whole number of 32-lane subgroups"
    );
    let half = hd / 2;
    let nsg = half / 32; // subgroups per row
    let mut k = Kb::new();
    let qkv = k.ssbo_f(0);
    let bias_b = k.ssbo_f(1);
    let qg = k.ssbo_f(2);
    let kg = k.ssbo_f(3);
    let kc = k.ssbo_f(4);
    let ks = k.ssbo_f(5);
    let vc = k.ssbo_f(6);
    let vs = k.ssbo_f(7);
    let params = k.ssbo_u(8);
    let kout = k.ssbo_f(9);
    let ksout = k.ssbo_f(10);
    let vout = k.ssbo_f(11);
    let vsout = k.ssbo_f(12);
    let kcq = k.ssbo_u(13);
    let sk = k.ssbo_f(14);
    let vdec = k.ssbo_f16(15);
    let planes = (nplanes > 1).then(|| k.ssbo_f(16));
    let (red, redp) = k.shared_f32(hd * rows);
    let (smax, smaxp) = k.shared_f32(rows * nsg); // per-(row, subgroup) |v| maxima, nsg > 1 only
    let chd = k.cu(hd);
    let chalf = k.cu(half);
    let cloc = k.cu(half * rows);
    let gid = k.gid_comp(0);
    let iy = k.gid_comp(1);
    let slot = k.udiv(gid, cloc); // WG-uniform: LocalSize == (hd/2)·rows
    let local = k.umod(gid, cloc);
    let r = k.udiv(local, chalf); // the row of this lane within the workgroup
    let lane = k.umod(local, chalf); // the RoPE pair (lane, lane + hd/2) this lane owns
    let rbase = k.imul(r, chd); // the row's slice of `red`
    let crows = k.cu(rows);
    let iyr = k.imul(iy, crows);
    let i = k.iadd(iyr, r);
    let c0 = k.cu(0);
    let c1t = k.cu(1);
    let tokens = k.load_u(params, c1t);
    let active = k.ult(i, tokens); // the last tile may hold rows past the chunk
    let pos0 = k.load_u(params, c0);
    let pos = k.iadd(pos0, i);
    let cqkvm = k.cu(qkvm);
    let iload = k.e(OP_SELECT, k.uint, &[active, i, c0]); // an inactive row reads row 0
    let rowb = k.imul(iload, cqkvm);
    let slotb = k.imul(slot, chd);
    let base = k.iadd(rowb, slotb);
    let idx = k.iadd(base, lane); // element `lane`; element `lane + hd/2` is `idx + hd/2`
                                  // The lane's pair lives in registers (`vlo`, `vhi`) until the stores: the RoPE partner of
                                  // element `d` is `d ± hd/2`, which this lane also owns, so rotation needs no shared memory.
    let vlo = k.fvar_f();
    let vhi = k.fvar_f();
    let plane = planes.map(|_| {
        // plane stride in floats `ceil(B/64)*64 * qkvm`, `B` from `params[1]` (as the norm fold)
        let cbn = k.cu(SPLITK_BN);
        let cbn1 = k.cu(SPLITK_BN - 1);
        let up = k.iadd(tokens, cbn1);
        let tiles = k.udiv(up, cbn);
        let padded = k.imul(tiles, cbn);
        k.imul(padded, cqkvm)
    });
    for (h, var) in [vlo, vhi].into_iter().enumerate() {
        let off = k.cu(h as u32 * half);
        let ix = k.iadd(idx, off);
        let mut v = if let (Some(pl), Some(plane)) = (planes, plane) {
            let mut acc = k.load_f(pl, ix);
            for sp in 1..nplanes {
                let cs = k.cu(sp);
                let po = k.imul(cs, plane);
                let pidx = k.iadd(po, ix);
                let pv = k.load_f(pl, pidx);
                acc = k.fadd(acc, pv);
            }
            acc
        } else {
            k.load_f(qkv, ix)
        };
        if bias {
            let bl = k.iadd(slotb, lane);
            let bi = k.iadd(bl, off);
            let bv = k.load_f(bias_b, bi);
            v = k.fadd(v, bv);
        }
        k.store_var(var, v);
    }
    let cnh = k.cu(nh);
    let cnhk = k.cu(nh + nkv);
    let is_k = k.e(OP_UGE, k.bool_t, &[slot, cnh]);
    let is_v = k.e(OP_UGE, k.bool_t, &[slot, cnhk]);
    let rotates = k.ult(slot, cnhk); // Q and K heads rotate; V does not
                                     // ── QK-norm (Q heads against `qg`, K heads against `kg`): the sum of squares is a per-lane
                                     //    pair sum, then a `log2(hd/2)` shared tree per row. The grouping differs from a `hd`-lane
                                     //    tree; the prefill planner passes `qknorm = false`.
    if qknorm {
        k.iff(rotates, |k| {
            let a = k.load_var_f(vlo);
            let b = k.load_var_f(vhi);
            let sa = k.fmul(a, a);
            let sb = k.fmul(b, b);
            let sq = k.fadd(sa, sb);
            let rl = k.iadd(rbase, lane);
            k.shared_store(red, redp, rl, sq);
            k.wg_barrier();
            let mut stride = half / 2;
            while stride >= 1 {
                let cs = k.cu(stride);
                let lt = k.ult(lane, cs);
                k.iff(lt, |k| {
                    let x = k.shared_load(red, redp, rl);
                    let ti = k.iadd(rl, cs);
                    let y = k.shared_load(red, redp, ti);
                    let z = k.fadd(x, y);
                    k.shared_store(red, redp, rl, z);
                });
                k.wg_barrier();
                stride /= 2;
            }
            let ss = k.shared_load(red, redp, rbase);
            let invhd = k.cf(1.0 / hd as f32);
            let mean = k.fmul(ss, invhd);
            let ce = k.cf(eps);
            let me = k.fadd(mean, ce);
            let inv = k.ext1(GLSL_INVERSE_SQRT, me);
            let f = k.float;
            for (v, var, h) in [(a, vlo, 0u32), (b, vhi, 1u32)] {
                let go = k.cu(h * half);
                let gi = k.iadd(lane, go);
                let gq_v = k.load_f(qg, gi);
                let gk_v = k.load_f(kg, gi);
                let gv = k.e(OP_SELECT, f, &[is_k, gk_v, gq_v]);
                let sc = k.fmul(v, inv);
                let o = k.fmul(sc, gv);
                k.store_var(var, o);
            }
            k.wg_barrier(); // every lane has read `red`
        });
    }
    // ── RoPE: the pair is `(vlo, vhi)`; the range reduction is `fused_rope_kv_module`'s (GPU
    //    sin/cos lose all range past ~2¹² rad, which is why >4k context collapsed before it existed).
    k.iff(rotates, |k| {
        let jf = k.utof(lane);
        let two_over_hd = k.cf(-2.0 / hd as f32 * theta.ln());
        let expo = k.fmul(jf, two_over_hd);
        let invf = k.ext1(GLSL_EXP, expo);
        let posf = k.utof(pos);
        let ang0 = k.fmul(posf, invf);
        let inv2pi = k.cf(1.0 / (2.0 * std::f32::consts::PI));
        let nr0 = k.fmul(ang0, inv2pi);
        let nrd = k.ext1(GLSL_ROUND, nr0);
        let cc1 = k.cf(6.283203125_f32);
        let cc2 = k.cf(2.0 * std::f32::consts::PI - 6.283203125_f32);
        let nc1 = k.fmul(nrd, cc1);
        let a1 = k.fsub(ang0, nc1);
        let nc2 = k.fmul(nrd, cc2);
        let ang = k.fsub(a1, nc2);
        let cs = k.ext1(GLSL_COS, ang);
        let sn = k.ext1(GLSL_SIN, ang);
        let x0 = k.load_var_f(vlo);
        let x1 = k.load_var_f(vhi);
        let a = k.fmul(x0, cs);
        let b = k.fmul(x1, sn);
        let r_lo = k.fsub(a, b);
        let cc = k.fmul(x1, cs);
        let dd = k.fmul(x0, sn);
        let r_hi = k.fadd(cc, dd);
        k.store_var(vlo, r_lo);
        k.store_var(vhi, r_hi);
    });
    let lo_f = k.load_var_f(vlo);
    let hi_f = k.load_var_f(vhi);
    let chalf_off = k.cu(half);
    let idx_hi = k.iadd(idx, chalf_off);
    k.iff(active, |k| {
        k.store_f(qkv, idx, lo_f);
        k.store_f(qkv, idx_hi, hi_f);
    });
    // ── KV append: absmax/127 per (head, position), the code map of `CpuState::kv_write`
    let does_kv = k.e(OP_UGE, k.bool_t, &[slot, cnh]);
    k.iff(does_kv, |k| {
        let f = k.float;
        let gl = k.glsl;
        let c0b = k.cu(0);
        // absmax: |lo|, |hi| in the lane, then `subgroupMax` over the row's 32-lane subgroups.
        // `fmax` is exactly associative and commutative, so the value is the shared tree's.
        let al = k.e(OP_EXT_INST, f, &[gl, GLSL_FABS, lo_f]);
        let ah = k.e(OP_EXT_INST, f, &[gl, GLSL_FABS, hi_f]);
        let am = k.ext2(GLSL_FMAX, al, ah);
        let sgm = sg_fmax(k, am);
        let amax = if nsg == 1 {
            sgm
        } else {
            let c32 = k.cu(32);
            let sgi = k.udiv(local, c32);
            let sgl = k.umod(local, c32);
            let lead = k.e(OP_IEQ, k.bool_t, &[sgl, c0b]);
            k.iff(lead, |k| k.shared_store(smax, smaxp, sgi, sgm));
            k.wg_barrier();
            let csg = k.cu(nsg);
            let sgr = k.imul(r, csg); // this row's first subgroup slot
            let mut m = k.shared_load(smax, smaxp, sgr);
            for g in 1..nsg {
                let cg = k.cu(g);
                let sgg = k.iadd(sgr, cg);
                let o = k.shared_load(smax, smaxp, sgg);
                m = k.ext2(GLSL_FMAX, m, o);
            }
            m
        };
        let inv127 = k.cf(1.0 / 127.0);
        let sc0 = k.fmul(amax, inv127);
        let zf = k.cf(0.0);
        let one = k.cf(1.0);
        let bt = k.bool_t;
        let posv = k.e(OP_FORD_GREATER_THAN, bt, &[sc0, zf]);
        let sc = k.e(OP_SELECT, f, &[posv, sc0, one]);
        let code_lo = k.int8_code_f(lo_f, sc);
        let code_hi = k.int8_code_f(hi_f, sc);
        let kvh_k = k.e(OP_ISUB, k.uint, &[slot, cnh]);
        let kvh_v = k.e(OP_ISUB, k.uint, &[slot, cnhk]);
        let u = k.uint;
        let kvh = k.e(OP_SELECT, u, &[is_v, kvh_v, kvh_k]);
        let cnkv = k.cu(nkv);
        let ct = k.cu(tmax);
        let kvt = k.imul(kvh, ct); // decode K/V head row base [kvh·tmax]
        let srow = k.imul(pos, cnkv);
        let sidx = k.iadd(srow, kvh); // scale slot [pos·nkv + kvh]
        let orow0 = k.imul(i, cnkv);
        let orow = k.iadd(orow0, kvh);
        let ocidx0 = k.imul(orow, chd);
        let ocidx = k.iadd(ocidx0, lane); // compact mirror slot [(i·nkv+kvh)·hd + d], d = lane
        let ocidx_hi = k.iadd(ocidx, chalf_off); // d = lane + hd/2
        let is0 = k.e(OP_IEQ, k.bool_t, &[lane, c0b]);
        let notv = k.e(OP_LOGICAL_NOT, bt, &[is_v]);
        k.iff(notv, |k| {
            let kb0 = k.imul(kvh, chd);
            let kb1 = k.iadd(kb0, lane);
            let kb2 = k.imul(kb1, ct);
            let kidx = k.iadd(kb2, pos); // K cache [(kvh·hd+d)·tmax + s], d = lane
            let kb1h = k.iadd(kb1, chalf_off);
            let kb2h = k.imul(kb1h, ct);
            let kidx_hi = k.iadd(kb2h, pos); // d = lane + hd/2
            k.iff(active, |k| {
                k.store_f(kc, kidx, code_lo);
                k.store_f(kout, ocidx, code_lo);
                k.store_f(kc, kidx_hi, code_hi);
                k.store_f(kout, ocidx_hi, code_hi);
                k.iff(is0, |k| {
                    k.store_f(ks, sidx, sc);
                    k.store_f(ksout, orow, sc);
                });
            });
            // ── decode K: pack 4 int8 codes/word along the dim axis — `kv_seed_module`'s map, with
            //    this lane's two `code`s staged in `red` for the cross-lane gather. The branch is
            //    workgroup-uniform (`slot` is), so the barrier is uniform.
            let rl = k.iadd(rbase, lane);
            let rh = k.iadd(rl, chalf_off);
            k.shared_store(red, redp, rl, code_lo);
            k.shared_store(red, redp, rh, code_hi);
            k.wg_barrier();
            let chd4 = k.cu(hd / 4);
            let packer = k.ult(lane, chd4);
            k.iff(active, |k| {
                k.iff(packer, |k| {
                    let u = k.uint;
                    let one = k.cf(1.0);
                    let mut word = k.cu(0);
                    for j in 0..4u32 {
                        let c4 = k.cu(4);
                        let l4 = k.imul(lane, c4);
                        let cj = k.cu(j);
                        let d0 = k.iadd(l4, cj);
                        let d = k.iadd(rbase, d0);
                        let cv = k.shared_load(red, redp, d);
                        let byte = k.int8_code(cv, one);
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
                // decode K scale, transposed `[kvh·tmax + s]` (lane 0; `sc` is uniform)
                k.iff(is0, |k| {
                    let di = k.iadd(kvt, pos);
                    k.store_f(sk, di, sc);
                });
            });
        });
        k.iff(is_v, |k| {
            k.iff(active, |k| {
                let vb0 = k.imul(kvh, ct);
                let vb1 = k.iadd(vb0, pos);
                let vb2 = k.imul(vb1, chd);
                let vidx = k.iadd(vb2, lane); // V cache [(kvh·tmax+s)·hd + d], d = lane
                let vidx_hi = k.iadd(vidx, chalf_off); // d = lane + hd/2
                k.store_f(vc, vidx, code_lo);
                k.store_f(vout, ocidx, code_lo);
                k.store_f(vc, vidx_hi, code_hi);
                k.store_f(vout, ocidx_hi, code_hi);
                k.iff(is0, |k| {
                    k.store_f(vs, sidx, sc);
                    k.store_f(vsout, orow, sc);
                });
                // ── decode V: f16(code·scale), bit-identical to `kv_seed_module`'s read of `vc`/`vs`
                let val_lo = k.fmul(code_lo, sc);
                k.store_f16(vdec, vidx, val_lo);
                let val_hi = k.fmul(code_hi, sc);
                k.store_f16(vdec, vidx_hi, val_hi);
            });
        });
    });
    k.finish(half * rows)
}

/// [`rmsnorm_batch_q_module`] as bytes.
pub fn rmsnorm_batch_q_bytes(n: u32, eps: f32, dpw: u32, residual: bool, ks: u32) -> Vec<u8> {
    bytes(rmsnorm_batch_q_module(n, eps, dpw, residual, ks))
}
/// [`rmsnorm_batch_q_noh_module`] as bytes — the resident prefill's intermediate-norm module
/// (`h`/binding 6 unread; writes only the chart and, conditionally, the residual stream).
pub fn rmsnorm_batch_q_noh_bytes(n: u32, eps: f32, dpw: u32, residual: bool, ks: u32) -> Vec<u8> {
    bytes(rmsnorm_batch_q_noh_module(n, eps, dpw, residual, ks))
}
/// [`quantize_batch_module`] as bytes.
pub fn quantize_batch_bytes(n: u32, dpw: u32) -> Vec<u8> {
    bytes(quantize_batch_module(n, dpw))
}
/// [`swiglu_quant_batch_module`] as bytes.
pub fn swiglu_quant_batch_bytes(dff: u32, dpw: u32) -> Vec<u8> {
    bytes(swiglu_quant_batch_module(dff, dpw))
}
/// [`qkv_rope_batch_module`] as bytes.
#[allow(clippy::too_many_arguments)]
pub fn qkv_rope_batch_bytes(
    nh: u32,
    nkv: u32,
    hd: u32,
    qkvm: u32,
    tmax: u32,
    theta: f32,
    eps: f32,
    bias: bool,
    qknorm: bool,
    ks: u32,
    rows: u32,
) -> Vec<u8> {
    bytes(qkv_rope_batch_module(
        nh, nkv, hd, qkvm, tmax, theta, eps, bias, qknorm, ks, rows,
    ))
}
