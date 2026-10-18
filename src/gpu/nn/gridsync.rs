//! simt_nn::gridsync — **what a device-wide barrier COSTS**, measured, so the megakernel question
//! is decided by `megakernel_needs_few_barriers` instead of by intuition.
//!
//! Sanguine `Closures/Engineering/SynchronizationCost.megakernel_needs_few_barriers` says a fused
//! kernel wins only when `n_bar·β < n_disp·L` — strictly fewer syncs than dispatches saved, priced
//! at their real costs. `L` (the per-dispatch drain) is measured: `DEC_BAR=9` against `DEC_BAR=0`
//! puts it at ~4.18 µs on this card (`results/decode-surface-audit.md`). `β` — the cost of one
//! device-scope grid barrier — was never measured on its own. It was only ever inferred as one of
//! two lumped causes in the 2026-07-08 chart-megakernel loss (`hashtrinity/GPU_PRIMITIVES.md`:
//! "RMSNorm+quantize recomputed redundantly across all `g` workgroups, and a device-scope
//! grid-barrier spin"), which is exactly the shape of attribution that names two mechanisms and
//! tests neither.
//!
//! So this measures β alone, with nothing else in the kernel.
//!
//! **The barrier is sense-reversing over device-scope atomics** — the shape CUDA's own
//! `grid_group::sync()` has (`cooperative_groups.h:178-182`, a spin-wait over a driver-provided
//! barrier struct). Each workgroup flips a private sense bit, lane 0 arrives on a shared counter,
//! the last arriver resets the counter and publishes the new sense, everyone else spins on it.
//! `vulkan_memory_model_device_scope` is already requested and enabled (`gpu/vulkan.rs:1219`).
//!
//! **Every spin is CAPPED, and that is not a detail.** A grid barrier deadlocks by construction if
//! the grid is larger than what the device can hold resident: the workgroups that never got
//! scheduled cannot arrive, and the ones that did spin forever. On a desktop card that is a hung
//! display, not a failed test. The spin is therefore a bounded loop — it gives up after
//! `SPIN_CAP` iterations, records that it did in `ctl[CTL_OVERFLOW]`, and lets the kernel retire.
//! A run that trips the cap is reported as INVALID rather than as a fast barrier, because a
//! barrier that gave up is not a barrier that was cheap.

use super::*;

const OP_ATOMIC_LOAD: u32 = 227;
const OP_ATOMIC_STORE: u32 = 228;
const OP_ATOMIC_IADD: u32 = 234;
const OP_LOGICAL_OR: u32 = 166;
const OP_BITWISE_XOR: u32 = 198;

const SCOPE_DEVICE: u32 = 1;
/// `AcquireRelease | UniformMemory` — the read-modify-write's semantics.
const SEM_DEV_ACQREL: u32 = 0x48;
/// `Acquire | UniformMemory`. `OpAtomicLoad` may not carry Release or AcquireRelease.
const SEM_DEV_ACQ: u32 = 0x42;
/// `Release | UniformMemory`. `OpAtomicStore` may not carry Acquire or AcquireRelease.
const SEM_DEV_REL: u32 = 0x44;

/// Slot layout of the control buffer (binding 0).
pub const CTL_COUNT: u32 = 0;
pub const CTL_SENSE: u32 = 1;
pub const CTL_OVERFLOW: u32 = 2;
pub const CTL_LEN: u32 = 4;

/// How many times a spinning workgroup re-reads the sense word before giving up. Sized so a
/// healthy barrier (which resolves in hundreds of reads) has orders of magnitude of headroom,
/// while a deadlocked grid still retires in well under a second per barrier.
pub const SPIN_CAP: u32 = 1 << 16;

impl Kb {
    /// `ctl[i]`, as a pointer for the atomic ops.
    fn ctl_ptr(&mut self, ctl: u32, i: u32) -> u32 {
        let ci = self.cu(i);
        let p = self.ptr_sb_u;
        let zero = self.cu(0);
        self.e(OP_ACCESS_CHAIN, p, &[ctl, zero, ci])
    }

    /// **One device-wide barrier.** `sense` is the workgroup's private sense variable (an fvar
    /// holding 0/1), `tid` the lane index, `g` the workgroup count the grid was launched with.
    ///
    /// Returns nothing; on exit every workgroup that reached it has observed every write made by
    /// every other workgroup before it — or the spin cap tripped and `ctl[CTL_OVERFLOW]` is set.
    fn grid_barrier(&mut self, ctl: u32, sense: u32, tid: u32, g: u32) {
        let (u, b) = (self.uint, self.bool_t);
        let (zero, one) = (self.cu(0), self.cu(1));

        // flip this workgroup's sense: every lane computes the same value
        let s0 = self.load_var_u(sense);
        let s1 = self.e(OP_BITWISE_XOR, u, &[s0, one]);
        self.store_var(sense, s1);

        // all lanes of this workgroup must be done before it ARRIVES
        self.wg_barrier();

        let is0 = self.e(OP_IEQ, b, &[tid, zero]);
        self.iff(is0, |k| {
            let cnt = k.ctl_ptr(ctl, CTL_COUNT);
            let sen = k.ctl_ptr(ctl, CTL_SENSE);
            let (sc, sa, sr, sl) = (
                k.cu(SCOPE_DEVICE),
                k.cu(SEM_DEV_ACQREL),
                k.cu(SEM_DEV_REL),
                k.cu(SEM_DEV_ACQ),
            );
            let onec = k.cu(1);
            let old = k.e(OP_ATOMIC_IADD, u, &[cnt, sc, sa, onec]);
            let last = k.cu(g - 1);
            let is_last = k.e(OP_IEQ, k.bool_t, &[old, last]);
            // the last arriver releases everyone: reset the counter, publish the new sense
            k.iff(is_last, |k| {
                let z = k.cu(0);
                k.ins(OP_ATOMIC_STORE, &[cnt, sc, sr, z]);
                let sv = k.load_var_u(sense);
                k.ins(OP_ATOMIC_STORE, &[sen, sc, sr, sv]);
            });
            // everyone else spins until the sense matches — BOUNDED, see the module note
            let notlast = k.e(OP_LOGICAL_NOT, k.bool_t, &[is_last]);
            k.iff(notlast, |k| {
                let want = k.load_var_u(sense);
                let cap = k.cu(SPIN_CAP);
                let ctr = k.fvar_u();
                let z = k.cu(0);
                k.store_var(ctr, z);
                let (hdr, chk, bod, cont, merge) = (k.id(), k.id(), k.id(), k.id(), k.id());
                k.ins(OP_BRANCH, &[hdr]);
                k.ins(OP_LABEL, &[hdr]);
                k.ins(OP_LOOP_MERGE, &[merge, cont, 0]);
                k.ins(OP_BRANCH, &[chk]);
                k.ins(OP_LABEL, &[chk]);
                let i = k.load_var_u(ctr);
                let got = k.e(OP_ATOMIC_LOAD, u, &[sen, sc, sl]);
                let ready = k.e(OP_IEQ, k.bool_t, &[got, want]);
                let over = k.e(OP_UGE, k.bool_t, &[i, cap]);
                let stop = k.e(OP_LOGICAL_OR, k.bool_t, &[ready, over]);
                let go = k.e(OP_LOGICAL_NOT, k.bool_t, &[stop]);
                k.ins(OP_BRANCH_CONDITIONAL, &[go, bod, merge]);
                k.ins(OP_LABEL, &[bod]);
                k.ins(OP_BRANCH, &[cont]);
                k.ins(OP_LABEL, &[cont]);
                let i2 = k.load_var_u(ctr);
                let o = k.cu(1);
                let inext = k.iadd(i2, o);
                k.store_var(ctr, inext);
                k.ins(OP_BRANCH, &[hdr]);
                k.ins(OP_LABEL, &[merge]);
                // record a give-up: a barrier that timed out must not read as a cheap barrier
                let i3 = k.load_var_u(ctr);
                let tripped = k.e(OP_UGE, k.bool_t, &[i3, cap]);
                k.iff(tripped, |k| {
                    let ov = k.ctl_ptr(ctl, CTL_OVERFLOW);
                    let oneu = k.cu(1);
                    let u2 = k.uint;
                    let _ = k.e(OP_ATOMIC_IADD, u2, &[ov, sc, sa, oneu]);
                });
            });
        });
        // release the rest of the workgroup only once lane 0 has been let through
        self.wg_barrier();
    }
}

/// **The β kernel: `rounds` grid barriers and nothing else.**
///
/// Launched as `g` workgroups of `local` lanes. Time it at two round counts and the SLOPE is β,
/// with the launch and teardown differenced away — the same two-point form the dispatch floor is
/// read with, so the two costs `megakernel_needs_few_barriers` compares are measured the same way.
///
/// `out[wg]` is written at the end purely so the barriers cannot be eliminated as dead.
pub fn gridsync_module(g: u32, rounds: u32, local: u32, work: u32) -> Vec<u32> {
    let mut k = Kb::new();
    let ctl = k.ssbo_u(0);
    let out = k.ssbo_u(1);
    let scratch = k.ssbo_u(2);
    let tid = k.local_index();
    let sense = k.fvar_u();
    let z = k.cu(0);
    k.store_var(sense, z);

    let rc = k.cu(rounds);
    let one = k.cu(1);
    k.loop_step(z, rc, one, |k, _| {
        // Dirty `work` words per workgroup BEFORE each barrier. A barrier's real cost is not the
        // instruction, it is making the previous stage's writes visible — which is why the decode
        // path's drain is 4.18 µs where an empty chain's is 1.07. Measuring beta with nothing to
        // flush measures the cheap half and would overstate the megakernel.
        if work > 0 {
            let wg = k.gid_comp(0);
            let cw = k.cu(work);
            let base = k.imul(wg, cw);
            let cl = k.cu(local);
            k.loop_step(tid, cw, cl, |k, i| {
                let idx = k.iadd(base, i);
                k.store_u(scratch, idx, idx);
            });
        }
        k.grid_barrier(ctl, sense, tid, g);
    });

    let is0 = k.e(OP_IEQ, k.bool_t, &[tid, z]);
    k.iff(is0, |k| {
        // GlobalInvocationId.x of lane 0 IS the workgroup's slot times `local` — any distinct
        // index will do; this exists only so the barriers are not dead code.
        let wg = k.gid_comp(0);
        // Report the give-up count HERE rather than leaving it in `ctl`: `ctl` must stay in device
        // memory or the barrier's atomics would cross PCIe and the measurement would be of the
        // link, not of the barrier. `out` is host-visible and written once per kernel, not per
        // barrier, so reading the verdict costs nothing that lands in the slope.
        let ov = k.ctl_ptr(ctl, CTL_OVERFLOW);
        let (sc, sl) = (k.cu(SCOPE_DEVICE), k.cu(SEM_DEV_ACQ));
        let u2 = k.uint;
        // The barriers cannot be eliminated as dead regardless of what is written here: the last
        // arriver's `OpAtomicStore` to `ctl` is an observable side effect.
        let n = k.e(OP_ATOMIC_LOAD, u2, &[ov, sc, sl]);
        k.store_u(out, wg, n);
    });
    k.finish(local)
}

pub fn gridsync_bytes(g: u32, rounds: u32, local: u32, work: u32) -> Vec<u8> {
    bytes(gridsync_module(g, rounds, local, work))
}

/// **The L kernel: touch one word, and nothing else.** Chained `n` times with the SAME barrier the
/// decode path uses, the slope in `n` is the per-dispatch drain — the `L` of the law, measured on
/// this card rather than carried over from a note. It must do a dependent read-modify-write so the
/// chain is a real hazard and the barrier cannot be optimised away.
pub fn nulldispatch_module(local: u32, work: u32) -> Vec<u32> {
    let mut k = Kb::new();
    let out = k.ssbo_u(0);
    let scratch = k.ssbo_u(1);
    let tid = k.local_index();
    let z = k.cu(0);
    // the SAME per-workgroup working set the barrier kernel dirties, so beta and L are compared
    // at equal memory pressure and the difference is the SYNC, not the traffic
    if work > 0 {
        let wg = k.gid_comp(0);
        let cw = k.cu(work);
        let base = k.imul(wg, cw);
        let cl = k.cu(local);
        k.loop_step(tid, cw, cl, |k, i| {
            let idx = k.iadd(base, i);
            k.store_u(scratch, idx, idx);
        });
    }
    let is0 = k.e(OP_IEQ, k.bool_t, &[tid, z]);
    k.iff(is0, |k| {
        let v = k.load_u(out, z);
        let one = k.cu(1);
        let n = k.iadd(v, one);
        k.store_u(out, z, n);
    });
    k.finish(local)
}

pub fn nulldispatch_bytes(local: u32, work: u32) -> Vec<u8> {
    bytes(nulldispatch_module(local, work))
}
