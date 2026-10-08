//! The DP4A attention lever: `Kb` builders for int8 q·k scores via `OpSDot`, moved verbatim out of
//! `simt_nn/mod.rs` (fold `modelfold/src/simt_nn/mod.rs`) to keep each file under 1500 lines.

use super::*;

impl Kb {
    /// A Workgroup-scope shared u32 array (the packed int8 query codes); returns (var, elem ptr).
    pub(super) fn shared_u32(&mut self, len: u32) -> (u32, u32) {
        let clen = self.cu(len);
        let (arr, ptr_arr, var, ptr_el) = (self.id(), self.id(), self.id(), self.id());
        let u = self.uint;
        self.ty(OP_TYPE_ARRAY, &[arr, u, clen]);
        self.ty(OP_TYPE_POINTER, &[ptr_arr, SC_WORKGROUP, arr]);
        self.ty(OP_VARIABLE, &[ptr_arr, var, SC_WORKGROUP]);
        self.ty(OP_TYPE_POINTER, &[ptr_el, SC_WORKGROUP, u]);
        (var, ptr_el)
    }
    pub(super) fn shared_load_u(&mut self, var: u32, ptr_el: u32, idx: u32) -> u32 {
        let u = self.uint;
        let p = self.e(OP_ACCESS_CHAIN, ptr_el, &[var, idx]);
        self.e(OP_LOAD, u, &[p])
    }
    pub(super) fn shared_store_u(&mut self, var: u32, ptr_el: u32, idx: u32, val: u32) {
        let p = self.e(OP_ACCESS_CHAIN, ptr_el, &[var, idx]);
        self.ins(OP_STORE, &[p, val]);
    }
    /// Store f32 `val` reinterpreted into `buf[idx]` of a u32 SSBO (the int8 K cache word).
    pub(super) fn store_u(&mut self, buf: u32, idx: u32, val: u32) {
        let psu = self.ptr_sb_u;
        let c0 = self.cu(0);
        let p = self.e(OP_ACCESS_CHAIN, psu, &[buf, c0, idx]);
        self.ins(OP_STORE, &[p, val]);
    }
    /// A function-scope signed-i32 variable (the DP4A accumulator).
    pub(super) fn fvar_s(&mut self) -> u32 {
        let s = self.sint_ty();
        if self.ptr_fn_s == 0 {
            self.ptr_fn_s = self.id();
            let p = self.ptr_fn_s;
            self.ty(OP_TYPE_POINTER, &[p, SC_FUNCTION, s]);
        }
        let t = self.ptr_fn_s;
        self.fvar(t)
    }
    pub(super) fn sconst0(&mut self) -> u32 {
        if self.sint0 == 0 {
            let s = self.sint_ty();
            self.sint0 = self.id();
            let z = self.sint0;
            self.ty(OP_CONSTANT, &[s, z, 0]);
        }
        self.sint0
    }
    pub(super) fn load_var_s(&mut self, var: u32) -> u32 {
        let s = self.sint_ty();
        self.e(OP_LOAD, s, &[var])
    }
    /// The int8 code of `round(v/scale)` clamped ±127, as the low byte (two's-complement u32) —
    /// the SAME chart the activation quantizer packs (`emit_quantize_from_shared`), one element.
    pub(super) fn int8_code(&mut self, v: u32, scale: u32) -> u32 {
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
        let ub = self.e(OP_BITCAST, u, &[si]);
        let cff = self.cu(0xFF);
        self.e(OP_BITWISE_AND, u, &[ub, cff])
    }
    /// **The inverse of [`Self::int8_code`]**: byte `sub` of `word`, sign-extended, as f32.
    ///
    /// Sign extension without `OpShiftRightArithmetic`: `(x ^ 0x80) − 0x80` wraps in uint and
    /// bitcasts to exactly `i8`'s value over the whole `0..255` range. This is the THIRD site
    /// needing it — `dp4a_chart_fold`'s two-level prelude has the other — so it is a helper rather
    /// than a fourth transcription of a five-instruction idiom whose failure mode is a silently
    /// wrong sign on half the codes.
    pub(super) fn int8_lane_f(&mut self, word: u32, sub: u32) -> u32 {
        let u = self.uint;
        let c8 = self.cu(8);
        let sh = self.imul(sub, c8);
        let sd = self.e(OP_SHIFT_RIGHT_LOGICAL2, u, &[word, sh]);
        let cff = self.cu(0xff);
        let b = self.e(OP_BITWISE_AND, u, &[sd, cff]);
        let c80 = self.cu(0x80);
        let xr = self.e(OP_BITWISE_XOR, u, &[b, c80]);
        let sb = self.e(OP_ISUB, u, &[xr, c80]);
        let s = self.sint_ty();
        let si = self.e(OP_BITCAST, s, &[sb]);
        let f = self.float;
        self.e(OP_CONVERT_S_TO_F, f, &[si])
    }
    /// Pack 4 f32 values `buf[base..base+4]` into one int8 word (4 codes) with `scale`.
    pub(super) fn pack_int8_global(&mut self, buf: u32, base: u32, scale: u32) -> u32 {
        let word = self.fvar_u();
        let z = self.cu(0);
        self.store_var(word, z);
        let c0 = self.cu(0);
        let four = self.cu(4);
        self.loop_u(c0, four, |k, b| {
            let idx = k.iadd(base, b);
            let v = k.load_f(buf, idx);
            let byte = k.int8_code(v, scale);
            let u = k.uint;
            let c8 = k.cu(8);
            let sh = k.imul(b, c8);
            let shifted = k.e(OP_SHIFT_LEFT_LOGICAL, u, &[byte, sh]);
            let w0 = k.load_var_u(word);
            let w1 = k.e(OP_BITWISE_OR, u, &[w0, shifted]);
            k.store_var(word, w1);
        });
        self.load_var_u(word)
    }
    /// Reduce `absmax_{i<n} |buf[base+i]|` into `red[0]` (shared, power-of-two `wg`); each lane
    /// folds its strided slice, a shared tree-reduce closes the max. Floor `red[0] ≥ 1e-12`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn absmax_reduce_global(
        &mut self,
        buf: u32,
        base: u32,
        red: u32,
        redp: u32,
        n: u32,
        wg: u32,
        tid: u32,
    ) {
        let cwg = self.cu(wg);
        let cn = self.cu(n);
        let mxv = self.fvar_f();
        let tiny = self.cf(1.0e-12);
        self.store_var(mxv, tiny);
        self.loop_step(tid, cn, cwg, |k, i| {
            let idx = k.iadd(base, i);
            let v = k.load_f(buf, idx);
            let f = k.float;
            let g = k.glsl;
            let av = k.e(OP_EXT_INST, f, &[g, GLSL_FABS, v]);
            let m0 = k.load_var_f(mxv);
            let m1 = k.ext2(GLSL_FMAX, m0, av);
            k.store_var(mxv, m1);
        });
        let a = self.load_var_f(mxv);
        self.shared_store(red, redp, tid, a);
        self.wg_barrier();
        self.tree_reduce(red, redp, wg, tid, true);
    }
    /// The integer q·k for position `s`: `Σ_w OpSDot(qq[w], Kq[(kbase_w+w)·ct+s])` → i32 → f32.
    /// `qq` is the shared packed-int8 query, `kcq` the packed-int8 K cache (word-i-major). This
    /// is the gemv's `SealChartFold` on the SCORE — 4 code·MACs/instruction, K one byte not two.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dp4a_score(
        &mut self,
        kcq: u32,
        qq: u32,
        qqp: u32,
        kbase_w: u32,
        s: u32,
        ct: u32,
        nwords: u32,
    ) -> u32 {
        self.dp4a = true;
        let sint = self.sint_ty();
        let iacc = self.fvar_s();
        let z = self.sconst0();
        self.store_var(iacc, z);
        let c0 = self.cu(0);
        let cnw = self.cu(nwords);
        self.loop_u(c0, cnw, |k, w| {
            let qw = k.shared_load_u(qq, qqp, w);
            let krow = k.iadd(kbase_w, w);
            let kw1 = k.imul(krow, ct);
            let kidx = k.iadd(kw1, s);
            let kw = k.load_u(kcq, kidx);
            let d4 = k.e(OP_SDOT, sint, &[qw, kw, PACKED_VEC_4X8]);
            let a0 = k.load_var_s(iacc);
            let a1 = k.e(OP_IADD, sint, &[a0, d4]);
            k.store_var(iacc, a1);
        });
        let isum = self.load_var_s(iacc);
        let f = self.float;
        self.e(OP_CONVERT_S_TO_F, f, &[isum])
    }
    /// One output row's affine-chart DP4A dot: `Σ_{b<blocks} d_b·(Σ q·x)_b·sx_b + m_b·xs_b`,
    /// `w` k-packed col-major (`widx = (b·cpb+cj)·cm + r`), `dm` f16-packed `(d,m)` at `b·cm+r`.
    /// `act_shared` selects the activation-chart source — shared `(xq,sx,xs)` (the fused-norm
    /// gemv's in-kernel chart) when true, else global buffers (a megakernel's cross-barrier
    /// chart) when false; `xqp/sxp/xsp` are the shared array's byte-offset id, ignored when
    /// global. Literal extraction of `gemv_dp4a_norm_module`'s inlined block loop — same emission
    /// order, so results stay bit-identical to the pre-refactor kernel.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dp4a_chart_fold(
        &mut self,
        w: u32,
        scale: ChartScale,
        r: u32,
        cm: u32,
        start_blk: u32,
        bound_blk: u32,
        xq: u32,
        xqp: u32,
        sx: u32,
        sxp: u32,
        xs: u32,
        xsp: u32,
        carrier: crate::simt::Carrier,
        act_shared: bool,
        // **Block offset the STAGED activation is zero-based against**, `None` when the shared
        // arrays span the whole `k` (every caller but the windowed `gemv_actq`).
        //
        // Split-K already windows the WORK to `[start_blk, bound_blk)`, but the arrays were still
        // allocated for all of `k` — 5.25 B per k element, which is 91 392 B at k=17408 against
        // this device's 49 152 B `maxComputeSharedMemorySize`
        // (`results/gemv-shared-is-over-the-limit.md`). Sizing them to the WINDOW and re-basing the
        // reads is what takes that back. `None` emits no subtraction at all, so the SPIR-V of every
        // other kernel is byte-identical rather than merely equivalent.
        xq_base_blk: Option<u32>,
        // **`Carrier::Q3`'s plane-B element offset within `w`**, 0 for every single-plane carrier.
        // Three bits do not divide 32, so Q3 rides Q2's phase loop for the low two bits and folds
        // the high bit in from a second plane appended to the SAME buffer — see
        // `q3_planes_gpu_chart` for the layout and why plane B is bit-permuted.
        plane_b_off: u32,
    ) -> u32 {
        self.dp4a = true;
        let q3 = carrier == crate::simt::Carrier::Q3;
        let q5 = carrier == crate::simt::Carrier::Q5;
        assert!(
            (!q3 && !q5) || plane_b_off != 0,
            "a two-plane chart (Carrier::Q3/Q5) without a plane-B offset would decode only the low \
bits of every weight"
        );
        let sint = self.sint_ty();
        let codes_per_word = carrier.pack();
        let dpw = codes_per_word / 4;
        let cpb = 32 / codes_per_word;
        let wbits = 32 / codes_per_word;
        let byte_mask = 0x0101_0101u32 * ((1u32 << wbits) - 1);
        let split = dpw > 1;

        // ── the row's scale source, resolved ONCE outside the block loop ──
        //
        // SINGLE-LEVEL (`sup == 0`): `dm[b·cm + r]` is an f16 `(d, lo)` pair — **1.0 bits/weight**
        // of metadata, which `results/where-the-27b-bits-go.md` measures as the largest single
        // quality lever on the 27B: at a fixed budget it is 0.5 bits/weight that the CODES could
        // have had, worth ~2x MSE by the ladder in `results/quantiser-rung-ladder.md`.
        //
        // TWO-LEVEL (`sup != 0`): `sup[r]` is the row's f16 `(dsup, losup)` super-pair and
        // `sub[b·mrow + r/2]` packs TWO rows' `(d8, l8)` bytes per word — **0.5 bits/weight**. The
        // decode must reproduce `fold_scales_two_level` exactly: `d = d8·dsup` with d8 UNSIGNED
        // 0..255, `lo = i8(l8)·losup` with l8 SIGNED -127..127. Sign-extending the wrong lane
        // still emits plausible logits, which is the failure mode this face already has a history
        // of — `two_level_gpu_chart_is_a_permutation_of_the_row_chart` pins the layout and
        // `gpu_proj_check` pins this arithmetic against the CPU row chart.
        let two_level = scale.sup_off != 0;
        let dm = scale.dm;
        let (dsup, losup, csh, rhalf, cmrow) = if two_level {
            let u = self.uint;
            let coff = self.cu(scale.sup_off);
            let sidx = self.iadd(coff, r);
            let supw = self.load_u(dm, sidx);
            let v2 = self.vec2f;
            let g = self.glsl;
            let unp = self.e(OP_EXT_INST, v2, &[g, GLSL_UNPACK_HALF2X16, supw]);
            let f = self.float;
            let ds = self.e(OP_COMPOSITE_EXTRACT, f, &[unp, 0]);
            let ls = self.e(OP_COMPOSITE_EXTRACT, f, &[unp, 1]);
            // `r & 1` selects the 16-bit half (shift = par·16), `r >> 1` the word.
            let c1 = self.cu(1);
            let par = self.e(OP_BITWISE_AND, u, &[r, c1]);
            let c4 = self.cu(4);
            let sh = self.e(OP_SHIFT_LEFT_LOGICAL, u, &[par, c4]);
            let rh = self.e(OP_SHIFT_RIGHT_LOGICAL2, u, &[r, c1]);
            (ds, ls, sh, rh, self.cu(scale.mrow))
        } else {
            (0, 0, 0, 0, 0)
        };

        let accf = self.fvar_f();
        let zf = self.cf(0.0);
        self.store_var(accf, zf);
        self.loop_u(start_blk, bound_blk, |kb, b| {
            // block prelude: the block's (d, lo) off whichever scale face is bound, and its sx / Σx
            let (dval, mval) = if two_level {
                let u = kb.uint;
                let bm = kb.imul(b, cmrow);
                let sidx = kb.iadd(bm, rhalf);
                let word = kb.load_u(dm, sidx);
                let half = kb.e(OP_SHIFT_RIGHT_LOGICAL2, u, &[word, csh]);
                let cff = kb.cu(0xff);
                let d8 = kb.e(OP_BITWISE_AND, u, &[half, cff]);
                let c8 = kb.cu(8);
                let hi = kb.e(OP_SHIFT_RIGHT_LOGICAL2, u, &[half, c8]);
                let l8 = kb.e(OP_BITWISE_AND, u, &[hi, cff]);
                // Sign-extend the byte without OpShiftRightArithmetic: `(x ^ 0x80) - 0x80` wraps in
                // uint and bitcasts to exactly i8's value over the whole 0..255 range.
                let c80 = kb.cu(0x80);
                let xr = kb.e(OP_BITWISE_XOR, u, &[l8, c80]);
                let sb = kb.e(OP_ISUB, u, &[xr, c80]);
                let s = kb.sint_ty();
                let si8 = kb.e(OP_BITCAST, s, &[sb]);
                let f = kb.float;
                let d8f = kb.e(OP_CONVERT_U_TO_F, f, &[d8]);
                let l8f = kb.e(OP_CONVERT_S_TO_F, f, &[si8]);
                (kb.fmul(d8f, dsup), kb.fmul(l8f, losup))
            } else {
                let bm = kb.imul(b, cm);
                let dmidx = kb.iadd(bm, r);
                let dmw = kb.load_u(dm, dmidx);
                let v2 = kb.vec2f;
                let g = kb.glsl;
                let unp = kb.e(OP_EXT_INST, v2, &[g, GLSL_UNPACK_HALF2X16, dmw]);
                let f = kb.float;
                (
                    kb.e(OP_COMPOSITE_EXTRACT, f, &[unp, 0]),
                    kb.e(OP_COMPOSITE_EXTRACT, f, &[unp, 1]),
                )
            };
            let sxv = if act_shared {
                let b = match xq_base_blk {
                    Some(base) => {
                        let u = kb.uint;
                        kb.e(OP_ISUB, u, &[b, base])
                    }
                    None => b,
                };
                kb.shared_load(sx, sxp, b)
            } else {
                kb.load_f(sx, b)
            };
            let xsv = if act_shared {
                let b = match xq_base_blk {
                    Some(base) => {
                        let u = kb.uint;
                        kb.e(OP_ISUB, u, &[b, base])
                    }
                    None => b,
                };
                kb.shared_load(xs, xsp, b)
            } else {
                kb.load_f(xs, b)
            };
            // Q3/Q5: the high-bit plane, ONE word per (32-block, row) — loop-invariant across `cj`
            // and `l`, so it is fetched here rather than per phase.
            let bword = if q3 || q5 {
                let coff = kb.cu(plane_b_off);
                let bm2 = kb.imul(b, cm);
                let bi1 = kb.iadd(bm2, r);
                let bi = kb.iadd(bi1, coff);
                kb.load_u(w, bi)
            } else {
                0
            };
            // inner: cpb weight words, each 1 (Q8) / 2 (Q4) / 4 (Q2) phases of OpSUDot
            let accq = kb.fvar_s();
            let s0 = kb.sconst0();
            kb.store_var(accq, s0);
            let ccpb = kb.cu(cpb);
            let c0b = kb.cu(0);
            kb.loop_u(c0b, ccpb, |kb, cj| {
                let ccpb2 = kb.cu(cpb);
                let cbase = kb.imul(b, ccpb2);
                let c = kb.iadd(cbase, cj);
                let widx1 = kb.imul(c, cm);
                let widx = kb.iadd(widx1, r);
                let wword = kb.load_u(w, widx);
                let xbase = if split {
                    let cdp = kb.cu(dpw);
                    kb.imul(c, cdp)
                } else {
                    c
                };
                let u = kb.uint;
                for l in 0..dpw {
                    let src = if !split {
                        wword
                    } else if l == 0 {
                        let mask = kb.cu(byte_mask);
                        kb.e(OP_BITWISE_AND, u, &[wword, mask])
                    } else {
                        let csh = kb.cu(wbits * l);
                        let shd = kb.e(OP_SHIFT_RIGHT_LOGICAL2, u, &[wword, csh]);
                        let mask = kb.cu(byte_mask);
                        kb.e(OP_BITWISE_AND, u, &[shd, mask])
                    };
                    // Q3: four CONTIGUOUS high bits for this phase, expanded to one per byte at
                    // bit 2. `(x · 0x00204081) & 0x01010101` scatters nibble bit `i` to byte `i`;
                    // it is only valid because `q3_planes_gpu_chart` permuted plane B so a phase's
                    // bits are adjacent — at the natural stride of 4 the partial products collide.
                    let src = if q3 {
                        let cjsh = kb.cu(16);
                        let sh1 = kb.imul(cj, cjsh);
                        let cl4 = kb.cu(4 * l);
                        let sh = kb.iadd(sh1, cl4);
                        let shifted = kb.e(OP_SHIFT_RIGHT_LOGICAL2, u, &[bword, sh]);
                        let cf = kb.cu(0xF);
                        let nib = kb.e(OP_BITWISE_AND, u, &[shifted, cf]);
                        let spread = kb.cu(0x0020_4081);
                        let mul = kb.e(OP_IMUL, u, &[nib, spread]);
                        let one = kb.cu(0x0101_0101);
                        let bits = kb.e(OP_BITWISE_AND, u, &[mul, one]);
                        let c2 = kb.cu(2);
                        let hi = kb.e(OP_SHIFT_LEFT_LOGICAL, u, &[bits, c2]);
                        kb.e(OP_BITWISE_OR, u, &[src, hi])
                    } else if q5 {
                        // Q5: the plane-B word holds, contiguously per phase, the fifth bit of this
                        // phase's four codes (`pack_q5_0_raw` permutes them so they are adjacent —
                        // at the natural nibble stride the `0x0204_0810` partial products collide).
                        // `x · 0x0204_0810 & 0x1010_1010` scatters nibble bit `i` to bit 4 of byte
                        // `i`, exactly the lane plane A placed the nibble's low four bits in.
                        let cjsh = kb.cu(8);
                        let sh1 = kb.imul(cj, cjsh);
                        let cl4 = kb.cu(4 * l);
                        let sh = kb.iadd(sh1, cl4);
                        let shifted = kb.e(OP_SHIFT_RIGHT_LOGICAL2, u, &[bword, sh]);
                        let cf = kb.cu(0xF);
                        let nib = kb.e(OP_BITWISE_AND, u, &[shifted, cf]);
                        let spread = kb.cu(0x0204_0810);
                        let mul = kb.e(OP_IMUL, u, &[nib, spread]);
                        let hi_mask = kb.cu(0x1010_1010);
                        let hi = kb.e(OP_BITWISE_AND, u, &[mul, hi_mask]);
                        kb.e(OP_BITWISE_OR, u, &[src, hi])
                    } else {
                        src
                    };
                    let xqi = if l == 0 {
                        xbase
                    } else {
                        let cl = kb.cu(l);
                        kb.iadd(xbase, cl)
                    };
                    let xw = if act_shared {
                        // words are 8 per 32-block, so the element base is `8 * base_blk`
                        let xqi = match xq_base_blk {
                            Some(base) => {
                                let c8 = kb.cu(8);
                                let woff = kb.imul(base, c8);
                                let u = kb.uint;
                                kb.e(OP_ISUB, u, &[xqi, woff])
                            }
                            None => xqi,
                        };
                        kb.shared_load_u(xq, xqp, xqi)
                    } else {
                        kb.load_u(xq, xqi)
                    };
                    let d4 = kb.e(OP_SUDOT, sint, &[xw, src, PACKED_VEC_4X8]);
                    let a0 = kb.load_var_s(accq);
                    let a1 = kb.e(OP_IADD, sint, &[a0, d4]);
                    kb.store_var(accq, a1);
                }
            });
            // block epilogue: acc += (d·float(Σqx))·sx + m·Σx
            let accqi = kb.load_var_s(accq);
            let f2 = kb.float;
            let accqf = kb.e(OP_CONVERT_S_TO_F, f2, &[accqi]);
            let dq = kb.fmul(dval, accqf);
            let dqs = kb.fmul(dq, sxv);
            let mxs = kb.fmul(mval, xsv);
            let s2 = kb.fadd(dqs, mxs);
            let a0 = kb.load_var_f(accf);
            let a1 = kb.fadd(a0, s2);
            kb.store_var(accf, a1);
        });
        self.load_var_f(accf)
    }

    /// The direct (F16 / F32) gemv fold — the non-chart sibling of [`Kb::dp4a_chart_fold`].
    /// Accumulates `Σ_k W[k·M+r] · x̂[k]` over the element window `[start_k, bound_k)`, reading the
    /// normalized activation `x̂` from **shared** `nrm` (the fused rmsnorm's output — no global
    /// round-trip). F16 weights are 2 rows per u32 word (`UnpackHalf2x16`, dynamic lane `r % 2`),
    /// byte-matching `simt::gemv_module`'s F16 layout; F32 is a direct load. `m % pack == 0`.
    pub(super) fn direct_fold(
        &mut self,
        w: u32,
        r: u32,
        start_k: u32,
        bound_k: u32,
        nrm: u32,
        nrmp: u32,
        m: u32,
        carrier: crate::simt::Carrier,
    ) -> u32 {
        let pack = carrier.pack(); // F16 → 2, F32 → 1
        let f = self.float;
        let accf = self.fvar_f();
        let zf = self.cf(0.0);
        self.store_var(accf, zf);
        // per-invocation: the row's word column (r / pack) and F16 unpack lane (r % pack)
        let (wcol, lane) = if pack > 1 {
            let cpack = self.cu(pack);
            (self.udiv(r, cpack), self.umod(r, cpack))
        } else {
            (r, 0)
        };
        let words_per_col = m / pack;
        self.loop_u(start_k, bound_k, |kb, kid| {
            let wval = if pack > 1 {
                let cwords = kb.cu(words_per_col);
                let wi1 = kb.imul(kid, cwords);
                let widx = kb.iadd(wi1, wcol);
                let wraw = kb.load_u(w, widx);
                let v2 = kb.vec2f;
                let g = kb.glsl;
                let unp = kb.e(OP_EXT_INST, v2, &[g, GLSL_UNPACK_HALF2X16, wraw]);
                kb.e(OP_VECTOR_EXTRACT_DYNAMIC, f, &[unp, lane])
            } else {
                let cm = kb.cu(m);
                let wi1 = kb.imul(kid, cm);
                let widx = kb.iadd(wi1, r);
                kb.load_f(w, widx)
            };
            let xv = kb.shared_load(nrm, nrmp, kid);
            let prod = kb.fmul(wval, xv);
            let a0 = kb.load_var_f(accf);
            let a1 = kb.fadd(a0, prod);
            kb.store_var(accf, a1);
        });
        self.load_var_f(accf)
    }
}
