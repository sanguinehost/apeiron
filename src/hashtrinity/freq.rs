//! The NB face — `FreqMap` and the `FreqLine` group-probe engine it rides on. Split out of `lib.rs`,
//! re-exported from it, so every path is unchanged. A separate structure from `Table`: SwissTable
//! control bytes, no key column, no deletions.

use crate::hashtrinity::{bits_for_slots, par, Coord, CoordInv, Regen, Table, EMPTY, PHI};
use std::mem::MaybeUninit;

impl<K: Coord + Eq + Copy + Sync> FreqMap<K, i64> {
    /// **The parallel two-level GROUP BY count**, on the group-probe (NB) engine (`GroupProbe.lean`:
    /// uniform golden fold ⇒ no identity-home block, 16-wide tag glance ⇒ miss filter, bounded
    /// terminal sweep). Partition `keys` by home bucket (top bits of `k·PHI` — balanced, no hot
    /// bucket, `RadixDescent.home_balanced`), then aggregate each bucket into its own `FreqMap`
    /// across the pool. Buckets hold disjoint keys ⇒ **no merge**: a key is in exactly one returned
    /// map and `∑ maps[b].get(k)` is its count — the disjoint-fold case of
    /// `SharedFold.interleave_exact` (independent per-bucket folds, any schedule, exact).
    /// Below the crossover (small `n` / single core —
    /// `GatherCrossover`) it falls back to one serial aggregation. Cost is linear in `n` GIVEN a
    /// per-probe bound (`AggregationBound.build_linear_of_bounded_probes`, where that bound is a
    /// hypothesis, refuted worst-case); `2^-64` fold-collision fp (the join/count use case).
    #[must_use]
    pub fn count_par(keys: &[K]) -> Vec<Self> {
        let pool = par::Pool::global();
        let w = pool.width();
        let n = keys.len();
        // Serial fallback below the crossover: one map, no partition cost.
        if w <= 1 || n < (1 << 15) {
            let mut m = Self::with_capacity(n);
            for &k in keys {
                m.add(&k, 1);
            }
            return vec![m];
        }
        const BBITS: u32 = 8; // 256 home buckets
        let b = 1usize << BBITS;
        let chunk = n.div_ceil(w);
        // Bucket = the TOP BBITS of the Fibonacci-spread coord (`c·PHI`) — `home_balanced` spread
        // (raw `coord()` is the identity for integers, top bits 0 ⇒ all in bucket 0; the `·PHI` fixes it).
        #[inline]
        // R5: the trailing mask is the identity on this expression's range and is what lets the
        // compiler discharge the two index bounds checks. BOUND: for any `c`, `c·PHI` is a 64-bit
        // word, so `(c·PHI) >> (64 - BBITS)` is its top `BBITS` bits and lies in `[0, 2^BBITS)` =
        // `[0, b)`; masking by `2^BBITS - 1` therefore leaves the value unchanged, while stating
        // `idx < b = local.len() = off.len()` to the optimiser, which removes one `cmp`+`jae` from
        // each of the two partition loops (2 branches/key). Not an approximation: same index.
        fn home(c: u64) -> usize {
            (c.wrapping_mul(PHI) >> (64 - BBITS)) as usize & ((1 << BBITS) - 1)
        }

        // R1: phase 1 folds every key once and phase 2 needs the *same* bucket of the *same* key, so
        // phase 2 reads the bucket back instead of refolding it. `BBITS = 8` gives `home ∈ [0, 2^8)
        // = [0, b)`, so one byte per key carries it exactly; the two phases walk the same `[lo, hi)`
        // per worker, so the absolute index is the correspondence. Cost: one `n`-byte scratch
        // allocation, written once in phase 1 and read once in phase 2.
        const _: () = assert!(BBITS <= 8);
        let mut homes: Vec<u8> = vec![0u8; n];
        let hptr = par::SendPtr(homes.as_mut_ptr());

        // Phase 1 — per-worker per-bucket histogram (row t = counts[t*b .. t*b+b]).
        let mut counts = vec![0u32; w * b];
        let cptr = par::SendPtr(counts.as_mut_ptr());
        pool.dispatch(w, |t| {
            let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
            let mut local = vec![0u32; b];
            for (i, &k) in keys[lo..hi].iter().enumerate() {
                let h = home(k.coord());
                // SAFETY: task t writes homes[lo+i]; disjoint by t; the barrier orders it before
                // phase 2's read.
                unsafe { *hptr.add(lo + i) = h as u8 };
                local[h] += 1;
            }
            for (j, &c) in local.iter().enumerate() {
                // SAFETY: task t owns row t (disjoint by t); barrier orders before the read below.
                unsafe { *cptr.add(t * b + j) = c };
            }
        });
        // Prefix sums (serial, w*b work): contiguous bucket ranges + each (worker,bucket) write offset.
        let mut bstart = vec![0usize; b + 1];
        for j in 0..b {
            let s: usize = (0..w).map(|t| counts[t * b + j] as usize).sum();
            bstart[j + 1] = bstart[j] + s;
        }
        let mut woff = vec![0usize; w * b];
        for j in 0..b {
            let mut acc = bstart[j];
            for t in 0..w {
                woff[t * b + j] = acc;
                acc += counts[t * b + j] as usize;
            }
        }

        // Phase 2 — parallel scatter into `routed`, grouped by bucket (disjoint offset ranges ⇒ no
        // contention). `K: Copy`, so writing uninitialised slots by pointer then `set_len` is sound.
        let mut routed: Vec<K> = Vec::with_capacity(n);
        let rptr = par::SendPtr(routed.as_mut_ptr());
        let woff_ref = &woff;
        pool.dispatch(w, |t| {
            let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
            let mut off: Vec<usize> = (0..b).map(|j| woff_ref[t * b + j]).collect();
            // The zip over two equal-length slices is what keeps R1 free of a per-key bounds
            // branch: `keys[lo..hi]` and `homes[lo..hi]` are sliced once, so the range check is
            // per loop, not per key. `enumerate` + `homes[lo + i]` costs one `cmp`/`jae` per key.
            for (&h, &k) in homes[lo..hi].iter().zip(&keys[lo..hi]) {
                let j = usize::from(h); // R1: carried, not refolded
                // SAFETY: prefix-summed offsets are disjoint across (t,j); each slot written once.
                unsafe { *rptr.add(off[j]) = k };
                off[j] += 1;
            }
        });
        // SAFETY: every slot 0..n was written exactly once by phase 2 (Σ counts = n).
        unsafe { routed.set_len(n) };

        // Phase 3 — aggregate each bucket into its own group-probe map, in parallel (disjoint keys
        // ⇒ no merge; schedule-free, `Web.preduceAt_schedule_free`). Static bucket chunks per worker.
        // R4: the placeholder vector existed only so phase 3 could write through a raw pointer, and
        // every one of its `b` maps was built by `with_capacity(0)` → `with_bits(3)` forced to
        // `max(4)` = 16 slots, i.e. 2 allocations each, then overwritten and dropped: `2b` = 512
        // allocations and 512 frees per call for nothing. `Vec<Option<Self>>` is the same shape of
        // out-parameter with a `None` that allocates nothing, so phase 3 writes `Some(m)` and the
        // unwrap at the end is one `Vec<Self>` allocation instead of 512.
        let mut maps: Vec<Option<Self>> = (0..b).map(|_| None).collect();
        let mptr = par::SendPtr(maps.as_mut_ptr());
        let bstart_ref = &bstart;
        let routed_ref = &routed;
        let bchunk = b.div_ceil(w);
        pool.dispatch(w, |t| {
            for j in (t * bchunk)..((t + 1) * bchunk).min(b) {
                let (lo, hi) = (bstart_ref[j], bstart_ref[j + 1]);
                let mut m = Self::with_capacity(hi - lo);
                for &k in &routed_ref[lo..hi] {
                    m.add(&k, 1);
                }
                // SAFETY: task t owns bucket range [t*bchunk,..); disjoint j ⇒ disjoint maps[j].
                unsafe { *mptr.add(j) = Some(m) }
            }
        });
        // Every slot is written: the `w` dispatches partition `[0, b)` by `bchunk` with no gap.
        maps.into_iter().map(|slot| slot.expect("phase 3 wrote every bucket")).collect()
    }
}

impl<K: CoordInv + Eq + Copy + Sync> Table<K, i64, Regen<K>> {
    /// **The parallel two-level GROUP BY count, KEYS RETAINED** (the Regen face), on the group-probe (NB) engine (`GroupProbe.lean`:
    /// uniform golden fold ⇒ no identity-home block, 16-wide tag glance ⇒ miss filter, bounded
    /// terminal sweep). Partition `keys` by home bucket (top bits of `k·PHI` — balanced, no hot
    /// bucket, `RadixDescent.home_balanced`), then aggregate each bucket into its own `FreqMap`
    /// across the pool. Buckets hold disjoint keys ⇒ **no merge**: a key is in exactly one returned
    /// map and `∑ maps[b].get(k)` is its count — the disjoint-fold case of
    /// `SharedFold.interleave_exact` (independent per-bucket folds, any schedule, exact).
    /// Below the crossover (small `n` / single core —
    /// `GatherCrossover`) it falls back to one serial aggregation. Cost is linear in `n` GIVEN a
    /// per-probe bound (`AggregationBound.build_linear_of_bounded_probes`, where that bound is a
    /// hypothesis, refuted worst-case); `2^-64` fold-collision fp (the join/count use case).
    #[must_use]
    pub fn count_par(keys: &[K]) -> Vec<Self> {
        let pool = par::Pool::global();
        let w = pool.width();
        let n = keys.len();
        // Serial fallback below the crossover: one map, no partition cost.
        if w <= 1 || n < (1 << 15) {
            let mut m = Self::coordinate_with_capacity(n);
            for &k in keys {
                m.add(k, 1);
            }
            return vec![m];
        }
        const BBITS: u32 = 8; // 256 home buckets
        let b = 1usize << BBITS;
        let chunk = n.div_ceil(w);
        // Bucket = the TOP BBITS of the Fibonacci-spread coord (`c·PHI`) — `home_balanced` spread
        // (raw `coord()` is the identity for integers, top bits 0 ⇒ all in bucket 0; the `·PHI` fixes it).
        #[inline]
        fn home(c: u64) -> usize {
            (c.wrapping_mul(PHI) >> (64 - BBITS)) as usize
        }

        // Phase 1 — per-worker per-bucket histogram (row t = counts[t*b .. t*b+b]).
        let mut counts = vec![0u32; w * b];
        let cptr = par::SendPtr(counts.as_mut_ptr());
        pool.dispatch(w, |t| {
            let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
            let mut local = vec![0u32; b];
            for &k in &keys[lo..hi] {
                local[home(k.coord())] += 1;
            }
            for (j, &c) in local.iter().enumerate() {
                // SAFETY: task t owns row t (disjoint by t); barrier orders before the read below.
                unsafe { *cptr.add(t * b + j) = c };
            }
        });
        // Prefix sums (serial, w*b work): contiguous bucket ranges + each (worker,bucket) write offset.
        let mut bstart = vec![0usize; b + 1];
        for j in 0..b {
            let s: usize = (0..w).map(|t| counts[t * b + j] as usize).sum();
            bstart[j + 1] = bstart[j] + s;
        }
        let mut woff = vec![0usize; w * b];
        for j in 0..b {
            let mut acc = bstart[j];
            for t in 0..w {
                woff[t * b + j] = acc;
                acc += counts[t * b + j] as usize;
            }
        }

        // Phase 2 — parallel scatter into `routed`, grouped by bucket (disjoint offset ranges ⇒ no
        // contention). `K: Copy`, so writing uninitialised slots by pointer then `set_len` is sound.
        let mut routed: Vec<K> = Vec::with_capacity(n);
        let rptr = par::SendPtr(routed.as_mut_ptr());
        let woff_ref = &woff;
        pool.dispatch(w, |t| {
            let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
            let mut off: Vec<usize> = (0..b).map(|j| woff_ref[t * b + j]).collect();
            for &k in &keys[lo..hi] {
                let j = home(k.coord());
                // SAFETY: prefix-summed offsets are disjoint across (t,j); each slot written once.
                unsafe { *rptr.add(off[j]) = k };
                off[j] += 1;
            }
        });
        // SAFETY: every slot 0..n was written exactly once by phase 2 (Σ counts = n).
        unsafe { routed.set_len(n) };

        // Phase 3 — aggregate each bucket into its own group-probe map, in parallel (disjoint keys
        // ⇒ no merge; schedule-free, `Web.preduceAt_schedule_free`). Static bucket chunks per worker.
        let mut maps: Vec<Self> = (0..b).map(|_| Self::coordinate_with_capacity(0)).collect();
        let mptr = par::SendPtr(maps.as_mut_ptr());
        let bstart_ref = &bstart;
        let routed_ref = &routed;
        let bchunk = b.div_ceil(w);
        pool.dispatch(w, |t| {
            for j in (t * bchunk)..((t + 1) * bchunk).min(b) {
                let (lo, hi) = (bstart_ref[j], bstart_ref[j + 1]);
                let mut m = Self::coordinate_with_capacity(hi - lo);
                for &k in &routed_ref[lo..hi] {
                    m.add(k, 1);
                }
                // SAFETY: task t owns bucket range [t*bchunk,..); disjoint j ⇒ disjoint maps[j].
                unsafe { *mptr.add(j) = m };
            }
        });
        maps
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// THE NB FACE — the frequency map, on a control-filtered frequency line.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// The frequency line: coords (frequencies) + values + a **packed byte control**, group probe.
///
/// The control is the first of the two flaw readings the Null cell forces on any lookup of opaque
/// data (sanguine `GoldenRouting.golden_control_is_the_first_flaw`, standing on
/// `NullReadout.one_flaw_is_blind`/`readout_injective`): a coarse reading computed from the query
/// itself, then an exact confirm against the stored residual (the Seed — the only bits that must
/// persist). Its group-granular control flow is `GroupProbe.lean` — SwissTable's three bare
/// assertions, proved:
///
/// * **the glance** — one `u64` ctrl word holds 8 byte lanes (a group of 8 slots, whose 8 coords
///   are exactly one 64-byte cache line); an exact SWAR haszero detector winnows all 8 at once,
///   and a filter is allowed to be winnowed sloppily because no-false-negatives + exact confirm =
///   exact (`GroupProbe.winnow_exact`);
/// * **the stop** — a probe halts at the first group holding an empty lane; sound because
///   placement is first-fit and occupancy is monotone — `FreqMap` never deletes
///   (`GroupProbe.stop_not_before_placement`; one deletion voids the licence,
///   `deletion_breaks_the_stop` — SwissTable's DELETED≠EMPTY tombstone, forced);
/// * **the tag** — each lane is `{0 = empty, else the fold's low byte, 0 remapped to 1}`. The tag
///   must be a window DISJOINT from the address windows: a top-window tag is a function of the
///   home address and rejects nothing exactly where the probe starts
///   (`GroupProbe.top_window_is_home_determined` — the bug the earlier top-bits band carried);
///   the low window has exact false-positive mass `2⁻ᵗ` globally and within every home fiber
///   (`low_window_balanced`, `home_and_tag_jointly_balanced`). Its width is the `F = D + λR` rate
///   dial, and a full byte is the point that MATCHES SwissTable's h2: the earlier 4-bit nibble
///   was measured slower on misses (its `2⁻⁴` false-positive mass forced a cold coord read on
///   ~1/3 of misses), the dial's distortion term made visible — the byte pays it down to `2⁻⁸`.
///
/// The **group sequence is the PHI-descent** read at group granularity
/// (`GroupProbe.group_descent_dichotomy` — the closure: separate within the digit walk, or
/// share a pile of exactly `2^(64 mod gbits)` folds (`pile_card`), tag-resolved when the residual
/// fits the tag window (`pile_resolved_by_tag`); no mention of the tag in the walk itself): an
/// adversary who defeats the filter still meets the descent's SEPARATION — which bounds aliasing,
/// not probe length (`AggregationBound.descent_can_be_wholly_blocked`). The control can only
/// shortcut, never lengthen; that part is unaffected.
///
/// **The readout depth is a constant, not the word-length walk.** `probe`'s glance-then-confirm
/// is exactly `SpinorReadout.resolve₂` (proven in sanguine
/// `proof/Closures/Engineering/CoordinateReadout.lean`; retired from this crate's corpus — see
/// THEOREMS.md External): flaw one = the home group, flaw
/// two = the low-window `tag`, and the stored-coord confirm (`slots[i].0 == f`) fires ONLY when
/// both collide. So the readout is a **spinor — it resolves in depth ≤ 2, the double cover**
/// (`SpinorReadout.spinor_depth_two`), not the variable radix length `D = ⌈W/b⌉` that
/// `RadixDescent.chain_separates_within_radix` bounds the single-projection walk by. Three facts,
/// none assuming injectivity: the verdict is the exact membership test for every key
/// (`resolve₂_correct`); the confirm fires iff the two windows jointly collide
/// (`resolve₂_touch_iff_double_collision`), which is the top-`(t+b)` fiber — of exactly
/// `2^(W−t−b)` folds (`home_and_tag_jointly_balanced`; the adjacent-window twin is
/// `SpinorReadout.concrete_spinor_depth_two`'s `top_pair_fiber_card`); everywhere else the depth-2
/// coordinate read IS the whole lookup, no key touched (`resolve₂_no_touch_of_separated`). This is
/// the readout's proof — depth-2, exact, flood-bounded — superseding the older
/// "separate-within-`D+1`" sketch for the deployed descend.
struct FreqLine<V> {
    ctrl: Box<[u8]>, // one control byte per slot; group g = bytes 16g..16g+16, the
    // SSE register SwissTable scans. `ctrl[i] == EMPTY_CTRL (0x80)`
    // ⟺ slot i empty; a full slot's byte is its 7-bit tag.
    slots: Box<[(u64, MaybeUninit<V>)]>, // (frequency, value) fused — one cache line per confirm/hit;
    // `slots[i].0 == EMPTY` ⟺ empty (the in-band Null the
    // Drop/Clone/grow liveness test reads). The miss path never
    // reads it (the tag lives in `ctrl`); a hit reads one line:
    // the payload is a projection of the confirm's own cell
    // (`GroupProbe.fused_payload_from_one_cell`), where split
    // arrays FORCE a second load (`split_payload_needs_second_cell`).
    gshift: u32,  // home group = (p >> gshift) & gmask — precomputed, one shift+and
    gmask: usize, // = (#groups) − 1 = (2^bits / 16) − 1
    bits: u32,
    len: usize,
}

/// The group width — 16 control bytes scanned in one glance, exactly SwissTable's SSE group. The
/// `GroupProbe.winnow_exact` licence is `∀ w`, so the width is a free efficiency knob; 16 is the
/// SSE register, and the byte tag makes the scan one `_mm_cmpeq_epi8` + `_mm_movemask_epi8`.
pub(crate) const GROUP: usize = 16;

/// **The glance** — 16 tag lanes compared to `tag` in one movemask (`SimtSelect.movemask`,
/// `GroupProbe.winnow_exact`): the exact SwissTable move (`_mm_cmpeq_epi8` + `_mm_movemask_epi8`,
/// SSE2 baseline), a portable byte loop off x86_64. ONE implementation, shared by the hashmap
/// ([`FreqLine::match_tag`]) and the codec's row match-finder (`codec::lz`) — the SAME coordinate
/// readout / group-probe at both call sites (this is the point of `CoordinateReadout`: SwissTable
/// lookup and zstd-style match finding are one operation). `base + GROUP ≤ ctrl.len()`.
#[inline]
pub(crate) fn group_tag_mask(ctrl: &[u8], base: usize, tag: u8) -> u16 {
    debug_assert!(base + GROUP <= ctrl.len());
    // SAFETY (both paths): base + GROUP ≤ ctrl.len() (caller-guaranteed), so ptr..ptr+16 is in-bounds.
    let ptr = unsafe { ctrl.as_ptr().add(base) };
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        _mm_movemask_epi8(_mm_cmpeq_epi8(
            _mm_loadu_si128(ptr.cast::<__m128i>()),
            _mm_set1_epi8(tag as i8),
        )) as u16
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut m = 0u16;
        for k in 0..GROUP {
            m |= u16::from(unsafe { *ptr.add(k) } == tag) << k;
        }
        m
    }
}
/// The empty-slot control byte: high bit set. A full slot's tag is 7-bit (high bit clear), so
/// empty detection is a single `_mm_movemask_epi8` of the sign bits — no compare (SwissTable's
/// `0xFF`-empty encoding, the reason the miss-deciding op is one instruction). Proven: the
/// sentinel can never collide with a tag (`GroupProbe.sentinel_never_a_tag`) and the sign bit
/// reads emptiness exactly (`flag_bit_reads_emptiness`) — the byte is the Null cell in 8 bits,
/// one Being/Non-Being flag + the coarse flaw, and 7 is the byte's maximum tag width, the
/// minimum false-positive mass (`tag_width_is_maximal`), not a tuning.
const EMPTY_CTRL: u8 = 0x80;
/// `0x80` per byte (the SWAR-fallback sign-bit mask); only used off x86_64.
#[cfg(not(target_arch = "x86_64"))]
const LANE_HI: u64 = 0x8080_8080_8080_8080;

impl<V> FreqLine<V> {
    fn with_bits(bits: u32) -> Self {
        let bits = bits.max(4); // at least one full 16-lane group
        let n = 1usize << bits;
        let gbits = bits - 4; // groups = 2^gbits
        FreqLine {
            ctrl: vec![EMPTY_CTRL; n].into_boxed_slice(),
            slots: std::iter::repeat_with(|| (EMPTY, MaybeUninit::uninit()))
                .take(n)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            gshift: if gbits == 0 { 0 } else { 64 - gbits },
            gmask: (n / GROUP) - 1,
            bits,
            len: 0,
        }
    }

    /// The lane tag of a frequency: the fold's low **7 bits** (high bit clear, so it never aliases
    /// the `EMPTY_CTRL` sentinel). 128 states = 7 bits — exactly SwissTable's h2 width, the point on
    /// the `F = D + λR` dial with false-positive mass `2⁻⁷`. The LOW window,
    /// disjoint from the address windows the descent consumes (`GroupProbe.tagWindow`; a top window
    /// would be home-determined and blind, `top_window_is_home_determined`).
    #[inline]
    fn tag(f: u64) -> u8 {
        (f.wrapping_mul(PHI) & 0x7F) as u8
    }

    #[inline]
    fn set_lane(&mut self, i: usize, s: u8) {
        // SAFETY: i < slots.len() = ctrl.len() = 2^bits.
        unsafe { *self.ctrl.get_unchecked_mut(i) = s };
    }

    /// **The glance** — the 16-lane tag mask of group `g` (`GroupProbe.winnow_exact`): the exact
    /// SwissTable move (`_mm_cmpeq_epi8` + `_mm_movemask_epi8`, SSE2 baseline), a portable byte loop
    /// elsewhere. Split from the empty mask so a *hit* pays only this one movemask — the empty mask
    /// (the stop) is computed lazily by [`match_empty`](Self::match_empty), only when no tag matched
    /// (exactly hashbrown's `match_tag` / `match_empty` split). The group is 16-aligned, so the load
    /// never straddles a cache line.
    #[inline]
    fn match_tag(&self, g: usize, tag: u8) -> u16 {
        // g < ctrl.len()/16 (descend_group masks to that), so base + 16 ≤ ctrl.len().
        group_tag_mask(&self.ctrl, g * GROUP, tag)
    }

    /// The empty-lane mask of group `g` — the stop signal. A full slot's tag has the high bit clear
    /// and `EMPTY_CTRL` has it set, so this is a single sign-bit `_mm_movemask_epi8` (no compare).
    /// Called only after [`match_tag`](Self::match_tag) finds no candidate: the group's cache line is
    /// already resident from that load, so the reload is an L1 hit, and the hit path skips it entirely.
    #[inline]
    fn match_empty(&self, g: usize) -> u16 {
        let base = g * GROUP;
        // SAFETY: g < ctrl.len()/16, so base + 16 ≤ ctrl.len().
        let ptr = unsafe { self.ctrl.as_ptr().add(base) };
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            // SAFETY: SSE2 baseline; ptr..ptr+16 in-bounds.
            unsafe { _mm_movemask_epi8(_mm_loadu_si128(ptr.cast::<__m128i>())) as u16 }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut empties = 0u16;
            for k in 0..GROUP {
                // SAFETY: base + k < base + 16 ≤ ctrl.len().
                let b = unsafe { *ptr.add(k) };
                empties |= (((b & 0x80) != 0) as u16) << k;
            }
            empties
        }
    }

    /// The home group — the common case (`j = 0`), one precomputed shift + and. The PHI-descent's
    /// top window read at group granularity.
    #[inline]
    fn home_group(&self, p: u64) -> usize {
        (p >> self.gshift) as usize & self.gmask
    }

    /// The group descent for the overflow path (`j ≥ 1`): successive `(bits−4)`-bit windows of
    /// `p` (`GroupProbe.topDigit`), then a linear sweep once the word is spent. `gbits` need not
    /// divide 64, so this is NOT a bare `chain_separates_within_radix` instantiation — the
    /// closure is `GroupProbe.group_descent_dichotomy`: distinct folds either separate within the
    /// digit walk, or share the whole consumed prefix and land in a pile of exactly `2^(64 mod
    /// gbits)` folds (`pile_card`), which the tag resolves at the first glance when that residual
    /// fits the tag window (`pile_resolved_by_tag` — the 4M table has `r = 7 = t`, the word fully
    /// consumed). The sweep offset is `j − full + 1`: offset 0 re-glanced the group the last
    /// digit step had just scanned (`zero_offset_duplicates`, the bug the proof caught); distinct
    /// offsets below `2^gbits` never repeat (`sweep_period_no_repeat`). Cold — group overflow only.
    #[inline]
    fn descend_group(&self, p: u64, j: u32) -> usize {
        let gbits = self.bits - 4;
        if gbits == 0 {
            return 0; // a single group
        }
        let used = gbits * (j + 1);
        if used <= 64 {
            (p >> (64 - used)) as usize & self.gmask
        } else {
            let full = 64 / gbits;
            ((p >> (64 - gbits * full)) as usize + (j - full) as usize + 1) & self.gmask
        }
    }

    /// The group probe. Per group: one SSE glance winnows all 16 lanes against the tag, then
    /// confirm candidates on the coord (the exact flaw); stop at the first group with an empty
    /// lane (sound: first-fit placement + monotone occupancy, `stop_not_before_placement`).
    /// Returns the slot and whether the frequency is present; on a miss the slot is the first
    /// empty lane in probe order — exactly where an insert must place (first-fit).
    #[inline]
    ///
    /// R2: the tag is returned, not recomputed. `insert_with`'s fresh-lane path wrote
    /// `Self::tag(f) = (f·PHI) & 0x7F`, a second application of the multiply `probe` had already
    /// done; returning the value removes the `lea;lea;and` triple the multiplier constant folds to
    /// from every fresh insert, on the serial `add` path and on `count_par`'s phase 3 alike.
    /// Callers that do not place a lane (`get`, `grow`'s re-place probe) drop it.
    fn probe_tagged(&self, f: u64) -> (usize, bool, u8) {
        let p = f.wrapping_mul(PHI);
        let tag = (p & 0x7F) as u8; // = tag(f), reusing p
        let mut g = self.home_group(p); // j = 0, the common case: one shift + and
        let mut j = 0u32;
        loop {
            let base = g * GROUP;
            let mut matches = self.match_tag(g, tag); // the hit path: one movemask, no empty scan
            while matches != 0 {
                let i = base + matches.trailing_zeros() as usize;
                // SAFETY: base + lane < ctrl.len() = slots.len().
                if unsafe { self.slots.get_unchecked(i).0 } == f {
                    return (i, true, tag);
                }
                matches &= matches - 1;
            }
            // No tag candidate here — only NOW read the empty mask (the stop): an empty lane means
            // the key, had it been inserted, would sit at or before this group ⇒ absent, and the
            // first empty lane is the insert point. The group's line is already resident from the
            // tag load, so this reload is an L1 hit; the hit path above never runs it.
            let empties = self.match_empty(g);
            if empties != 0 {
                return (base + empties.trailing_zeros() as usize, false, tag);
            }
            j += 1;
            g = self.descend_group(p, j);
        }
    }

    /// [`probe_tagged`](Self::probe_tagged) without the tag.
    #[inline]
    fn probe(&self, f: u64) -> (usize, bool) {
        let (i, found, _tag) = self.probe_tagged(f);
        (i, found)
    }

    #[inline]
    fn get(&self, f: u64) -> Option<&V> {
        let (i, found) = self.probe(f);
        if found {
            Some(unsafe { self.slots.get_unchecked(i).1.assume_init_ref() })
        } else {
            None
        }
    }

    /// Prefetch the home group's control line + slot line for fold `p` — the pipeline warm-up.
    #[inline]
    fn prefetch_group(&self, p: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let base = self.home_group(p) * GROUP;
            _mm_prefetch(self.ctrl.as_ptr().add(base).cast::<i8>(), _MM_HINT_T0);
            _mm_prefetch(self.slots.as_ptr().add(base).cast::<i8>(), _MM_HINT_T0);
        }
        let _ = p;
    }

    /// Iterate live `(coord, &value)` slots — a live slot is `slots[i].0 != EMPTY`.
    fn iter_live(&self) -> impl Iterator<Item = (u64, &V)> {
        self.slots.iter().filter_map(|(c, v)| {
            if *c == EMPTY {
                None
            } else {
                // SAFETY: coord ≠ EMPTY ⇒ the value was initialized on insert.
                Some((*c, unsafe { v.assume_init_ref() }))
            }
        })
    }

    /// Reduce the whole table in parallel over live values — `encircle` on the group-probe engine
    /// (mirror of `Table::encircle`; schedule-free, `Web.preduceAt_schedule_free`).
    fn encircle<M, R, C>(&self, id: M, residual: R, combine: C) -> M
    where
        M: Send + Sync + Clone,
        R: Fn(&V) -> M + Sync,
        C: Fn(M, M) -> M + Sync,
        V: Sync,
    {
        let fold_range = |start: usize, end: usize| -> M {
            let mut acc = id.clone();
            for i in start..end {
                if self.slots[i].0 != EMPTY {
                    // SAFETY: live coord ⇒ value initialized.
                    acc = combine(
                        acc,
                        residual(unsafe { self.slots.get_unchecked(i).1.assume_init_ref() }),
                    );
                }
            }
            acc
        };
        let n = self.slots.len();
        let pool = par::Pool::global();
        if pool.width() <= 1 || n < 1 << 14 {
            return fold_range(0, n);
        }
        let tasks = (pool.width() * 4).min(n);
        let chunk = n.div_ceil(tasks);
        let mut partials: Vec<M> = (0..tasks).map(|_| id.clone()).collect();
        let pptr = par::SendPtr(partials.as_mut_ptr());
        pool.dispatch(tasks, |t| {
            let start = t * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            // SAFETY: task t writes only partials[t]; disjoint by t; barrier orders before merge.
            unsafe { *pptr.add(t) = fold_range(start, end) };
        });
        partials.into_iter().fold(id, &combine)
    }

    fn insert_with(&mut self, f: u64, new: V, merge: impl FnOnce(V, V) -> V) {
        if (self.len + 1) * 8 >= self.slots.len() * 7 {
            self.grow();
        }
        let (i, found, tag) = self.probe_tagged(f); // R2: the tag `probe` already computed
        if found {
            // SAFETY: found ⇒ slot i is live ⇒ its value is initialized.
            let slot = unsafe { self.slots.get_unchecked_mut(i) };
            let old = unsafe { slot.1.assume_init_read() };
            if std::mem::needs_drop::<V>() {
                // panic-safe: blank the COORD only, so a merge panic leaves a TOMBSTONE — Drop/
                // Clone/grow skip it (coord == EMPTY, no double-free) but the control byte keeps
                // its tag, so probes for OTHER keys do not stop here. Blanking the lane too would
                // be a deletion mid-chain, exactly `GroupProbe.deletion_breaks_the_stop`: a key
                // placed past this group would false-miss after the panic. DELETED ≠ EMPTY. Only
                // when `V` has a destructor — a non-drop `V` (the counter/accumulator case)
                // cannot double-free, so the dance is skipped and the merge writes in place.
                slot.0 = EMPTY;
                let merged = merge(new, old);
                let slot = unsafe { self.slots.get_unchecked_mut(i) };
                slot.1 = MaybeUninit::new(merged);
                slot.0 = f;
            } else {
                slot.1 = MaybeUninit::new(merge(new, old));
            }
        } else {
            // SAFETY: i < slots.len() (probe returns an in-range empty slot).
            let slot = unsafe { self.slots.get_unchecked_mut(i) };
            slot.0 = f;
            slot.1 = MaybeUninit::new(new);
            self.set_lane(i, tag); // R2: not `Self::tag(f)`, which refolds f·PHI
            self.len += 1;
        }
    }

    #[cold]
    fn grow(&mut self) {
        let mut bigger = Self::with_bits(self.bits + 1);
        for i in 0..self.slots.len() {
            let c = self.slots[i].0;
            if c != EMPTY {
                let (j, _) = bigger.probe(c); // re-place along the same probe sequence
                let v = unsafe { self.slots.get_unchecked_mut(i).1.assume_init_read() };
                self.slots[i].0 = EMPTY; // blank the source so the old line's Drop skips it
                let dst = unsafe { bigger.slots.get_unchecked_mut(j) };
                dst.0 = c;
                dst.1 = MaybeUninit::new(v);
                bigger.set_lane(j, Self::tag(c));
                bigger.len += 1;
            }
        }
        *self = bigger;
    }

    /// Test-only: the worst probe length (in group steps) over all present keys — the flood bound.
    /// A flood clusters keys at one home group; the descent must still separate them within a few
    /// group steps, so this stays tiny (`G` linear + radix depth), never `O(n)`.
    #[cfg(test)]
    fn max_probe_groups(&self) -> u32 {
        let mut worst = 0;
        for i in 0..self.slots.len() {
            let f = self.slots[i].0;
            if f == EMPTY {
                continue;
            }
            let p = f.wrapping_mul(PHI);
            let tag = (p & 0x7F) as u8;
            let mut g = self.home_group(p);
            let mut j = 0u32;
            let mut groups = 1u32;
            loop {
                let base = g * GROUP;
                let mut m = self.match_tag(g, tag);
                let mut hit = false;
                while m != 0 {
                    if self.slots[base + m.trailing_zeros() as usize].0 == f {
                        hit = true;
                        break;
                    }
                    m &= m - 1;
                }
                if hit || self.match_empty(g) != 0 {
                    break;
                }
                j += 1;
                groups += 1;
                g = self.descend_group(p, j);
                assert!(
                    groups < 100_000,
                    "probe did not terminate — flood unbounded"
                );
            }
            worst = worst.max(groups);
        }
        worst
    }
}

impl<V> Drop for FreqLine<V> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<V>() {
            for (c, v) in self.slots.iter_mut() {
                if *c != EMPTY {
                    // SAFETY: coord != EMPTY ⇒ value initialized.
                    unsafe { v.assume_init_drop() };
                }
            }
        }
    }
}

impl<V: Clone> Clone for FreqLine<V> {
    fn clone(&self) -> Self {
        let slots = self
            .slots
            .iter()
            .map(|(c, v)| {
                if *c == EMPTY {
                    (EMPTY, MaybeUninit::uninit())
                } else {
                    (*c, MaybeUninit::new(unsafe { v.assume_init_ref() }.clone()))
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        FreqLine {
            ctrl: self.ctrl.clone(),
            slots,
            gshift: self.gshift,
            gmask: self.gmask,
            bits: self.bits,
            len: self.len,
        }
    }
}

/// **The frequency map — the NB (Non-Being) face of the storage trinity.**
///
/// The opaque golden router. An opaque key routes by its FOLD (its frequency, a `u64`); the map
/// keeps no key at all — the frequency IS the address (`key_routes_iff_fold_distinguishes`). The
/// 16-byte pointer and the key bytes are gone; only the 8-byte frequency remains, and its storage is
/// **independent of key length** (17 B/slot — 8-byte coord + 1-byte tag + value — whether the key
/// is 8 bytes or a kilobyte).
///
/// The three storage faces, the trinity `x = B/U`:
/// * [`CoordMap`] / [`Regen`] — **B** (Being): coordinate keys, address round-trips, stores nothing.
/// * [`Table`] / [`Stored`] — **B/U**: opaque keys, keeps the key to confirm exactly.
/// * `FreqMap` — **NB** (Non-Being): opaque keys, the fold collapses the key, only the frequency
///   survives; stores nothing, and rejects misses through the control — the Null cell's coarse
///   first reading (`golden_control_is_the_first_flaw`), whose cache-resident density is the
///   derived description — while the PHI-descent underneath is untouched (the filter adds no probe
///   surface; the descent's own worst case is the measured one, not a proved one).
///
/// **The one tradeoff.** The fold is lossy, so two distinct keys with the same 64-bit fold collide,
/// and a non-member query whose fold equals a member's is a false positive — both `2^-64` (zero over
/// millions of probes). This is the map for dedup, joins, caches, counters, indexes on high-entropy
/// keys, where `2^-64` is "never". A map that must reject arbitrary non-members exactly, or return
/// the key, wants [`Table`]. `FreqMap` carries frequencies, not keys.
#[derive(Clone, Debug, Default)]
pub struct FreqMap<K, V> {
    inner: FreqLine<V>,
    _k: std::marker::PhantomData<K>,
}

impl<V> std::fmt::Debug for FreqLine<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FreqLine")
            .field("bits", &self.bits)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}
impl<V> Default for FreqLine<V> {
    fn default() -> Self {
        Self::with_bits(3)
    }
}

impl<K: Coord, V> FreqMap<K, V> {
    /// The key's frequency (its fold), dodging the two reserved coordinate sentinels `EMPTY` and
    /// `EMPTY-1`. Branchless: `min(c, EMPTY-2)` folds exactly those two top values to `EMPTY-2`
    /// and leaves every other frequency untouched (one `cmp`+`cmov`, no branch on the hot path).
    /// Proven: the clamp never emits a sentinel (`GroupProbe.clamp_avoids_sentinels`), is the
    /// identity below the top (`clamp_injective_below`), and its whole distortion is one 3-point
    /// fiber — added collision mass `2/2^64` (`clamp_top_fiber`), the price of the in-band Null.
    #[inline]
    fn freq(k: &K) -> u64 {
        k.coord().min(EMPTY - 2)
    }

    /// A frequency map sized for the **group probe's optimum — dense** (`≈ 1.5·cap` slots, load
    /// ~0.33–0.67). The byte-tag group probe (`GroupProbe.lean`) rejects a miss with one control
    /// word regardless of load — probe work is occupancy over the group width, `w`-fold amplified
    /// against load (`GroupProbe.full_groups_bounded_by_occupancy`, the load inversion) — so,
    /// unlike the old direct-slot band probe, which wanted a sparse table to keep hits
    /// collision-free, density is now strictly better: a smaller footprint
    /// means fewer cold lines on every out-of-cache access. Measured: at 1M/4M keys this matches
    /// hashbrown on misses (the area the sparse table lost, +67%) and stays within ~15% on hits,
    /// where `≈ 3·cap` (the former read-tuned sizing) was slower on BOTH. The load knob is
    /// `LoadFactorFrontier`'s, not a security flag — the probe is always immune.
    pub fn with_capacity(cap: usize) -> Self {
        FreqMap {
            inner: FreqLine::with_bits(bits_for_slots(cap.saturating_mul(3) / 2)),
            _k: std::marker::PhantomData,
        }
    }

    /// Alias of [`with_capacity`](Self::with_capacity) — kept for source compatibility. The group probe's optimum is the
    /// same dense sizing for read- and stream-heavy work alike; the former sparse/dense split was
    /// calibrated for the retired direct-slot probe.
    pub fn stream_with_capacity(cap: usize) -> Self {
        Self::with_capacity(cap)
    }

    /// Look up a key by its frequency. `2^-64` false-positive on non-members (see the type docs).
    #[inline]
    pub fn get(&self, k: &K) -> Option<&V> {
        self.inner.get(Self::freq(k))
    }

    /// Insert `k → new`, resolving a frequency clash with `f` (`|new, old| …`).
    #[inline]
    pub fn insert_with(&mut self, k: &K, new: V, f: impl FnOnce(V, V) -> V) {
        self.inner.insert_with(Self::freq(k), new, f);
    }

    /// Insert `k → v`, last write wins.
    #[inline]
    pub fn insert(&mut self, k: &K, v: V) {
        self.inner.insert_with(Self::freq(k), v, |new, _| new);
    }

    /// The number of live frequencies.
    pub fn len(&self) -> usize {
        self.inner.len
    }
    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }
    /// The slot capacity (the frequency line length).
    pub fn capacity(&self) -> usize {
        self.inner.slots.len()
    }

    /// Iterate the live values (arbitrary order). The key is not recoverable from the NB fold —
    /// `values` is the reduce input; use [`CoordMap`] if you need the keys back.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.inner.iter_live().map(|(_, v)| v)
    }

    /// **The stream read, pipelined** ([`Table::get_each`] on the group probe): probe each key,
    /// prefetching the home group `P` folds ahead so the miss latency is hidden
    /// (`MemoryLevelParallelism.optimal_depth_hides`). Visits `(key, Option<&value>)` in order.
    pub fn get_each<'a, F>(&'a self, keys: &'a [K], mut visit: F)
    where
        F: FnMut(&'a K, Option<&'a V>),
    {
        const P: usize = 16;
        let m = &self.inner;
        let n = keys.len();
        if n <= P || m.slots.len() <= (1 << 15) {
            for k in keys {
                visit(k, m.get(Self::freq(k)));
            }
            return;
        }
        for k in &keys[..P] {
            m.prefetch_group(Self::freq(k).wrapping_mul(PHI));
        }
        for i in 0..n {
            visit(&keys[i], m.get(Self::freq(&keys[i])));
            if i + P < n {
                m.prefetch_group(Self::freq(&keys[i + P]).wrapping_mul(PHI));
            }
        }
    }

    /// Reduce the whole table in parallel over live values — [`Table::encircle`] on the group probe.
    pub fn encircle<M, R, C>(&self, id: M, residual: R, combine: C) -> M
    where
        M: Send + Sync + Clone,
        R: Fn(&V) -> M + Sync,
        C: Fn(M, M) -> M + Sync,
        V: Sync,
    {
        self.inner.encircle(id, residual, combine)
    }

    /// **The threaded stream read** ([`Table::get_all`] on the group probe): split the keys one slice
    /// per core, each pipelines internally into a disjoint output slice. A read can't affect another,
    /// so the split is safe and `out[i] = get(keys[i])`, order preserved.
    pub fn get_all<'a>(&'a self, keys: &'a [K]) -> Vec<Option<&'a V>>
    where
        K: Sync,
        V: Sync,
    {
        let n = keys.len();
        let mut out: Vec<Option<&'a V>> = vec![None; n];
        let pool = par::Pool::global();
        if pool.width() <= 1 || n < 1 << 15 {
            let mut w = 0;
            self.get_each(keys, |_, v| {
                out[w] = v;
                w += 1;
            });
            return out;
        }
        let tasks = pool.width();
        let chunk = n.div_ceil(tasks);
        let optr = par::SendPtr(out.as_mut_ptr());
        pool.dispatch(tasks, |t| {
            let start = t * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            let mut w = start;
            self.get_each(&keys[start..end], |_, v| {
                // SAFETY: worker t writes only its disjoint [start,end); barrier before the read.
                unsafe { *optr.add(w) = v };
                w += 1;
            });
        });
        out
    }

    /// **The fused parallel key-stream reduce** ([`Table::get_reduce`] on the group probe): each core
    /// pipelines its slice and folds locally, partials combined once — no `N`-alloc.
    pub fn get_reduce<'a, M, Rf, C>(&'a self, keys: &'a [K], id: M, map: Rf, combine: C) -> M
    where
        K: Sync,
        V: Sync,
        M: Send + Sync + Clone,
        Rf: Fn(Option<&'a V>) -> M + Sync,
        C: Fn(M, M) -> M + Sync,
    {
        let reduce_range = |start: usize, end: usize| -> M {
            let mut acc = id.clone();
            self.get_each(&keys[start..end], |_, v| {
                acc = combine(core::mem::replace(&mut acc, id.clone()), map(v));
            });
            acc
        };
        let n = keys.len();
        let pool = par::Pool::global();
        if pool.width() <= 1 || n < 1 << 15 {
            return reduce_range(0, n);
        }
        let tasks = (pool.width() * 4).min(n);
        let chunk = n.div_ceil(tasks);
        let mut partials: Vec<M> = (0..tasks).map(|_| id.clone()).collect();
        let pptr = par::SendPtr(partials.as_mut_ptr());
        pool.dispatch(tasks, |t| {
            let start = t * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            // SAFETY: task t writes only partials[t]; disjoint by t; barrier before merge.
            unsafe { *pptr.add(t) = reduce_range(start, end) };
        });
        partials.into_iter().fold(id, &combine)
    }
}

impl<K: Coord> FreqMap<K, i64> {
    /// Count: add `v` to the running total at `k`'s frequency (the B/U accumulator face).
    #[inline]
    pub fn add(&mut self, k: &K, v: i64) {
        self.inner
            .insert_with(Self::freq(k), v, |new, old| new + old);
    }
}

#[cfg(test)]
mod freqline_flood {
    //! Phase-4 gate: the ·PHI-collapse reads home/tag/descent from the fold's own bits, so an
    //! adversary who controls the fold's top window controls the home group directly. Flood immunity
    //! must then rest ENTIRELY on the descent separating distinct folds within the radix depth (not on
    //! any home-spread). These build the sharpest floods and assert the probe stays bounded.
    use super::{FreqMap, PHI};

    fn assert_bounded(keys: &[u64], label: &str) {
        let mut m: FreqMap<u64, i64> = FreqMap::with_capacity(keys.len());
        for &k in keys {
            m.insert(&k, 1);
        }
        assert_eq!(m.len(), keys.len(), "{label}: all distinct keys stored");
        for &k in keys {
            assert!(m.get(&k).is_some(), "{label}: present key resolves");
        }
        let worst = m.inner.max_probe_groups();
        assert!(
            worst < 128,
            "{label}: flood must stay radix-bounded, got worst {worst} groups (n = {})",
            keys.len()
        );
    }

    /// Worst case for a top-window-first descent read from the fold's own bits: small sequential
    /// integers. `freq(k) = k`, so every key's home window (top bits) is 0 AND the descent's first
    /// several windows are 0 — they differ only in the low bits. If the descent still separates them
    /// within the radix depth, the collapse is flood-safe for the hardest input.
    #[test]
    fn small_sequential_ints_stay_bounded() {
        let keys: Vec<u64> = (0..12_000u64).collect();
        assert_bounded(&keys, "small-sequential");
    }

    /// The classic multiples attack — all keys share the low bits; here also the top bits (they are
    /// `i << 40`, so for i < 2^12 the top ~12 bits vary but the middle is zero). Distinct folds,
    /// so the descent separates.
    #[test]
    fn multiples_attack_stays_bounded() {
        let keys: Vec<u64> = (1..=12_000u64).map(|i| i << 20).collect();
        assert_bounded(&keys, "multiples");
    }

    /// Keys whose top window collides directly (the collapse's home is the fold's top window): pin the
    /// top 16 bits to 0, vary the rest. Every key homes to group 0.
    #[test]
    fn pinned_home_window_stays_bounded() {
        // top 16 bits zero; the low 48 bits are a PHI-scrambled counter (distinct, dense).
        let keys: Vec<u64> = (0..12_000u64).map(|i| i.wrapping_mul(PHI) >> 16).collect();
        assert_bounded(&keys, "pinned-home");
    }
}
