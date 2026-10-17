//! Reduced joint block-ICM post-pass on a finished two-plane pair: the one-bit fold transfer's
//! `nb_refine` (T3).
//!
//! Port of `tools/onebit/fold_transfer.py` (removed; at `2f15d6a9`) lines 715–753 (`NB_SWEEPS` and
//! `nb_refine`), which runs at most `rounds` rounds of
//!
//! 1. a joint `E8 ∪ O8` block-ICM pass over both planes' stored codes ([`e8_joint_icm_2plane`]),
//! 2. a joint least-squares refresh of both planes' group scales ([`r_step2`]) quantised by
//!    [`quant_r_hier`], then one [`c_step`] per plane against the residual of the other plane,
//!    quantised by [`quant_c_u8`] at `6` bits,
//!
//! keeping a round only when `J` strictly falls, and reverting (stopping) otherwise. The two
//! quantisers are the stored formats, so the post-pass writes no new bits outside them: the kept
//! round's row scale is a [`quant_r_hier`] value and its column scale is a `6`-bit
//! [`quant_c_u8`] value.
//!
//! # Frames
//!
//! A pack is `(S_rot, r, c, d)` with weight `plane_w = fwht128(S_rot ⊙ d) · r_rep · c`
//! ([`icm::plane_w`]): `S_rot` is the code in the **rotated** frame — one
//! [`crate::learn::code::e8`] codebook row per 8-column block — and `fwht128(S_rot ⊙ d)` is the
//! transformed code `S_e` the script passes to [`r_step2`] and to `compose`. Both calls here follow
//! that convention: [`e8_joint_icm_2plane`] takes `S_rot`, while [`r_step2`] and [`compose`] take
//! `S_e = fwht128(S_rot ⊙ d)`, and the packs returned carry `S_rot` (the ICM output) with the
//! quantised `r` and `c`.
//!
//! # Objective
//!
//! The two `J` evaluations and the round-acceptance test use [`steps::objective`], the script's
//! `J(Ws, Wp, Hl)` (line 434), which is **unweighted**: `g_row` enters only the joint ICM block
//! costs and the two `c_step`s, exactly as in `nb_refine`, which calls `J(Ws, W1 + W2, Hl)` with no
//! `g_row`. [`JointStats`] reports the joint ICM's own row-weighted `J`. Each round's `W1 + W2` is
//! re-derived from the round's own packs by [`icm::plane_w`], the same determination the next call's
//! entry `J` enumerates, so a kept round's `j_nb` is the returned pair's `J` and repeated runs are
//! non-increasing exactly rather than up to the grid unit.
//!
//! # Partiality
//!
//! No body here panics on caller input. [`nb_refine`] refuses a slice length that disagrees with
//! `o`/`n` and an `n` that is not a positive multiple of [`GROUP`], with [`NbError`]; every input is
//! a Q32.32 word, so the finiteness refusal the `f64` entry needed no longer exists. The inner
//! steps' refusals are carried through. The algorithm terminates after at most `rounds` rounds,
//! each of which either strictly lowers `J` or ends the loop, so the result never has a `J` above
//! `j_seq`.
//!
//! # Grade
//!
//! `tested` (`tests/learn_fold_nb_refine.rs`). No extraction or refinement links this body to a Lean
//! theorem, so nothing here is `proved`.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "the body is the ported reference: it indexes row-major `[o, n]` buffers by \
              `o/k/r1/r2`, names the packs `p1/p2` and the scales `r/c/d` as the script does, and \
              the option surface (g_row, icm_ridge, sweeps, rounds) mirrors the Python signature; a \
              bag struct would hide which arguments are metric options and which are shapes; the \
              one function is long because it is one ported body"
)]

use crate::learn::fold::fit::icm::{
    bin_into, code_transform_q, e8_joint_icm_2plane, plane_w, r_step2, BinOp, IcmError, JointStats,
    Plane, PlaneRef, BLOCK,
};
use crate::learn::fold::fit::quant::{quant_c_u8, quant_r_hier, QuantError};
use crate::learn::fold::fit::steps::{c_step, compose, objective, StepsError, GROUP};

/// Rows the joint ICM hands to its argmin loop at a time: the script's `row_chunk` default, `256`.
///
/// It bounds the working set only — rows are independent given the residual field, so it does not
/// change the result (`tests/learn_fold_icm.rs` checks that) — and `o ≤ 256` in the ported pipeline
/// means one chunk.
pub const ROW_CHUNK: usize = 256;

/// Bits per column scale in the `c` quantiser of the post-pass (`quant_c_u8(., 6)`).
pub const C_BITS: u32 = 6;

/// Block sweeps of each joint ICM pass: the script's `NB_SWEEPS` default (`2`), passed to
/// [`e8_joint_icm_2plane`] when the caller has no `ONEBIT_NB_SWEEPS` override.
pub const NB_SWEEPS: usize = 2;

/// A refusal from [`nb_refine`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NbError {
    /// A slice length disagrees with the declared shape (`o`, `n`, `o·(n/8)`, `o·(n/128)`, `n`, or
    /// the `g_row` length `o`).
    Shape,
    /// `n` is not a positive multiple of [`GROUP`], which the Hadamard passes need.
    Width,
    /// The joint ICM pass refused ([`IcmError`]).
    Icm(IcmError),
    /// A closed-form step refused ([`StepsError`]).
    Steps(StepsError),
    /// A scale quantiser refused ([`QuantError`]).
    Quant(QuantError),
}

impl From<IcmError> for NbError {
    fn from(e: IcmError) -> Self {
        Self::Icm(e)
    }
}

impl From<StepsError> for NbError {
    fn from(e: StepsError) -> Self {
        Self::Steps(e)
    }
}

impl From<QuantError> for NbError {
    fn from(e: QuantError) -> Self {
        Self::Quant(e)
    }
}

/// The log of one [`nb_refine`] run.
#[derive(Clone, Debug, PartialEq)]
pub struct NbLog {
    /// `J(ws, W1 + W2, hl)` of the input pair: [`steps::objective`], the unweighted script `J`, as a
    /// Q32.32 word.
    pub j_seq: i64,
    /// `J` of the returned pair; equal to [`NbLog::j_seq`] when no round was kept.
    pub j_nb: i64,
    /// `moves` summed over the rounds that ran.
    pub moves: usize,
    /// Rounds kept, each of which strictly lowered `J`.
    pub rounds_kept: usize,
    /// The joint ICM's counters for every round that ran, in order.
    pub per_round: Vec<JointStats>,
}

/// The result of [`nb_refine`]: the final plane weights, the final packs and the log.
#[derive(Clone, Debug)]
pub struct NbRefined {
    /// Final weight of plane 1, row-major `[o, n]`, Q32.32.
    pub w1: Vec<i64>,
    /// Final weight of plane 2, row-major `[o, n]`, Q32.32.
    pub w2: Vec<i64>,
    /// Final pack of plane 1: `(S_rot, idx, r, c, d)` with `r` the quantised row scale and `c` on
    /// the [`C_BITS`]-bit grid.
    pub p1: Plane,
    /// Final pack of plane 2, same convention.
    pub p2: Plane,
    /// The run log.
    pub log: NbLog,
}

/// A borrowed view of an owned [`Plane`].
fn pref(p: &Plane) -> PlaneRef<'_> {
    PlaneRef {
        s: &p.s,
        idx: &p.idx,
        r: &p.r,
        c: &p.c,
        d: &p.d,
    }
}

/// Copy a borrowed plane into an owned one. `O(o·n)` time and space.
fn to_owned_plane(p: PlaneRef<'_>) -> Plane {
    Plane {
        s: p.s.to_vec(),
        idx: p.idx.to_vec(),
        r: p.r.to_vec(),
        c: p.c.to_vec(),
        d: p.d.to_vec(),
    }
}

/// `fwht128(s ⊙ d)`, the transformed code `S_e` of a pack, in the script's multiplication order
/// (`s[o,i] * d[i]` first, then the block transform), on the `Q32.32` grid.
///
/// # Errors
/// [`NbError::Shape`] if `s.len() != o·n` or `d.len() != n`; [`NbError::Icm`] on a width the
/// transform refuses or an `i64` overflow.
///
/// # Complexity
/// Temporal `O(o·n·log2(GROUP))` grid operations; spatial `O(o·n)`.
fn transformed_code(s: &[i64], d: &[i64], o: usize, n: usize) -> Result<Vec<i64>, NbError> {
    if s.len() != o * n || d.len() != n {
        return Err(NbError::Shape);
    }
    Ok(code_transform_q(s, d, o, n)?)
}

/// `a op b` elementwise on the `Q32.32` grid.
///
/// # Errors
/// [`NbError::Shape`] on a length disagreement; [`NbError::Icm`] on an `i64` overflow.
///
/// # Complexity
/// Temporal `O(a.len())` checked adds; spatial `O(a.len())`.
fn combine(a: &[i64], b: &[i64], op: BinOp) -> Result<Vec<i64>, NbError> {
    if a.len() != b.len() {
        return Err(NbError::Shape);
    }
    let mut x = a.to_vec();
    bin_into(&mut x, b, op)?;
    Ok(x)
}

/// Shape preflight for [`nb_refine`]: the lengths the inner steps re-check.
///
/// # Errors
/// [`NbError::Shape`] on a length disagreement, [`NbError::Width`] if `n` is not a positive multiple
/// of [`GROUP`].
///
/// # Complexity
/// Temporal `O(1)`; spatial `O(1)`.
fn check_inputs(
    ws: &[i64],
    hl: &[i64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
) -> Result<(), NbError> {
    if o == 0 || n == 0 {
        return Err(NbError::Shape);
    }
    if !n.is_multiple_of(GROUP) {
        return Err(NbError::Width);
    }
    let g = n / GROUP;
    let nb = n / BLOCK;
    for p in [p1, p2] {
        if p.s.len() != o * n
            || p.idx.len() != o * nb
            || p.r.len() != o * g
            || p.c.len() != n
            || p.d.len() != n
        {
            return Err(NbError::Shape);
        }
    }
    if ws.len() != o * n || hl.len() != n * n {
        return Err(NbError::Shape);
    }
    if let Some(gr) = g_row {
        if gr.len() != o {
            return Err(NbError::Shape);
        }
    }
    Ok(())
}

/// Reduced joint block-ICM post-pass on a finished two-plane pair: at most `rounds` rounds of
/// (joint `E8 ∪ O8` block ICM over both packs; `r_step2` + [`quant_r_hier`]; one [`c_step`] and
/// `6`-bit [`quant_c_u8`] per plane against the other plane's residual), each kept only when the
/// unweighted `J(ws, w1 + w2, hl)` strictly falls, and stopping otherwise.
///
/// `rounds == 0` returns the input packs unchanged (with `w1`, `w2` their [`plane_w`] weights and
/// `j_nb == j_seq`). `sweeps` is the block-sweep count of each joint ICM pass (`NB_SWEEPS = 2` in
/// the script); `icm_ridge` is applied inside the joint ICM only, as in the script.
///
/// # Errors
/// [`NbError`] on a shape or width disagreement, or a refusal from one of the inner steps
/// (`NotPositiveDefinite` reaches here from [`r_step2`]/[`c_step`]).
///
/// # Complexity
///
/// Temporal: `rounds · (e8_joint_icm_2plane + r_step2 + 2 c_step + 2 quant)` plus `(1 + rounds)`
/// evaluations of `J` at `o·n²` multiply-adds each, plus `O(rounds·o·n·log2(128))` for the two
/// `fwht128` passes and `O(rounds·o·n)` for the compositions. The derivation is the script's: every
/// round closes in one joint ICM pass, one two-plane `r_step2`, two `c_step`s with a `6`-bit
/// quantiser each, and is judged by one `J`; the entry and the return contribute the remaining two
/// `J`s, each one `n`-term reduction per `(o, i)` entry of `D·Hl` and one multiply-add per entry
/// ([`steps::objective`]). Spatial: `O(o·n)` for the packs and the weights, plus
/// `e8_joint_icm_2plane`'s own `O(n²)` metric buffers and `(n/8)·256²` `f64` for its `X` tensor
/// (`16 MB` at `n = 256`), and `O(o·n)` per `c_step`.
pub fn nb_refine(
    ws: &[i64],
    hl: &[i64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
    icm_ridge: i64,
    sweeps: usize,
    rounds: usize,
) -> Result<NbRefined, NbError> {
    check_inputs(ws, hl, p1, p2, o, n, g_row)?;
    let g = n / GROUP;

    // W1, W2 = plane_w(P1), plane_w(P2); j_seq = J(Ws, W1 + W2, Hl).
    let w1_start = plane_w(p1, o, n)?;
    let w2_start = plane_w(p2, o, n)?;
    let wsum = combine(&w1_start, &w2_start, BinOp::Add)?;
    let j_seq = objective(ws, &wsum, hl, o, n)?;

    let mut cur1 = to_owned_plane(p1);
    let mut cur2 = to_owned_plane(p2);
    let mut w1 = w1_start;
    let mut w2 = w2_start;
    let mut j_cur = j_seq;
    let mut log = NbLog {
        j_seq,
        j_nb: j_seq,
        moves: 0,
        rounds_kept: 0,
        per_round: Vec::new(),
    };

    for _ in 0..rounds {
        let (na1, na2, st) = e8_joint_icm_2plane(
            ws,
            hl,
            pref(&cur1),
            pref(&cur2),
            o,
            n,
            sweeps,
            icm_ridge,
            g_row,
            ROW_CHUNK,
            None,
        )?;
        log.moves += st.moves;
        log.per_round.push(st);
        if st.moves == 0 {
            break;
        }
        // S1e, S2e = fwht128(S_rot ⊙ d) of the new packs.
        let s1e = transformed_code(&na1.s, &na1.d, o, n)?;
        let s2e = transformed_code(&na2.s, &na2.d, o, n)?;
        let (r1, r2) = r_step2(
            ws,
            hl,
            PlaneRef {
                s: &s1e,
                idx: &na1.idx,
                r: &na1.r,
                c: &na1.c,
                d: &na1.d,
            },
            PlaneRef {
                s: &s2e,
                idx: &na2.idx,
                r: &na2.r,
                c: &na2.c,
                d: &na2.d,
            },
            o,
            n,
        )?;
        // The quantisers carry words; the row scale and the column scale are read in place.
        let rh1 = quant_r_hier(&r1, o, g)?.rq;
        let rh2 = quant_r_hier(&r2, o, g)?.rq;
        // c1n = quant_c_u8(c_step(ws - compose(S2e, rh2, c2), hl, S1e, rh1), 6).0
        let other2 = compose(&s2e, &rh2, &na2.c, o, n)?;
        let resid1 = combine(ws, &other2, BinOp::Sub)?;
        let c1_raw = c_step(&resid1, hl, &s1e, &rh1, o, n, g_row)?;
        let (c1n, _) = quant_c_u8(&c1_raw, C_BITS)?;
        // The two round packs, and their weights re-derived from those packs through `plane_w`: the
        // same determination the next call's entry `J` enumerates, so a kept round's logged `J` is
        // the returned pair's own value and repeated runs are non-increasing exactly.
        let np1 = Plane {
            s: na1.s,
            idx: na1.idx,
            r: rh1,
            c: c1n,
            d: na1.d,
        };
        let w1n = plane_w(pref(&np1), o, n)?;
        // c2n = quant_c_u8(c_step(ws - W1n, hl, S2e, rh2), 6).0
        let resid2 = combine(ws, &w1n, BinOp::Sub)?;
        let c2_raw = c_step(&resid2, hl, &s2e, &rh2, o, n, g_row)?;
        let (c2n, _) = quant_c_u8(&c2_raw, C_BITS)?;
        let np2 = Plane {
            s: na2.s,
            idx: na2.idx,
            r: rh2,
            c: c2n,
            d: na2.d,
        };
        let w2n = plane_w(pref(&np2), o, n)?;
        let wn = combine(&w1n, &w2n, BinOp::Add)?;
        let j_new = objective(ws, &wn, hl, o, n)?;
        // A round is kept only on a strict fall of the unweighted `J`, on the exact grid word.
        let keep = j_new < j_cur;
        if !keep {
            break;
        }
        cur1 = np1;
        cur2 = np2;
        w1 = w1n;
        w2 = w2n;
        j_cur = j_new;
        log.rounds_kept += 1;
    }
    log.j_nb = j_cur;
    Ok(NbRefined {
        w1,
        w2,
        p1: cur1,
        p2: cur2,
        log,
    })
}
