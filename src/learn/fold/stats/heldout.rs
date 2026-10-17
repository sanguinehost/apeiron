//! Held-out quality of a Q32.32 student against a Q32.32 teacher over token sequences: mean forward
//! `KL(teacher ‖ student)`, mean total variation, top-1 argmax agreement, and mean next-token
//! negative log-likelihood of each model.
//!
//! Ports the evaluator of `~/lanes/port/spec/fold_transfer.py` `kl_eval` and
//! `spec_stats` (lines 2151–2220) and the true-label NLL of `~/lanes/port/spec/
//! pv_finish.py` (`c4_nll`, `n = B·(T − 1)`). Two logit sources feed one evaluator
//! ([`Acc::score_logits`]): the whole-model [`crate::learn::grad::model::forward`] ([`held_out`],
//! [`held_out_seq`]) and the engine's shipped [`KernelMode::Q8Act`] decode step
//! ([`held_out_engines`], [`held_out_engines_per_seq`]). Each row carries the next-token
//! distribution `p = softmax(z)`. Over every scored position the port returns the mean
//!
//! - forward KL `Σ_x p_t(x)·(log p_t(x) − log p_s(x))` (`kls/n`, nats/token), read from
//!   [`kl_row`]`(z_s, z_t)`;
//! - total variation `1 − Σ_x min(p_t(x), p_s(x)) = ½Σ_x |p_t(x) − p_s(x)|` (`1 − accept`, the
//!   `spec_stats` acceptance complement), read from [`tv_row`]`(z_s, z_t)`;
//! - top-1 agreement `[argmax p_t = argmax p_s]` (`agree/n`);
//! - next-token NLL `−log p(token_{t+1} | position t)`, read from [`log_softmax_row`], summed over
//!   the `T − 1` positions of each sequence that predict a following token (the last position,
//!   which predicts nothing, is excluded).
//!
//! # Carrier and rules
//!
//! Logits are Q32.32 `i64` (`ONE = 2^32`), the head's carrier. Every per-row loss is the head's own
//! exact-`i128`, once-rounded Q32.32 value; the evaluator sums those per-row values in checked
//! `i128` and converts to `f64` **once**, at the end, dividing by the position count. No per-row
//! `f64` and no float softmax appear on the path. `KL(p_t ‖ p_s)` is asymmetric, so the argument
//! order of [`kl_row`] is semantic (student first, teacher second).
//!
//! # Partiality
//!
//! The evaluator refuses, through [`HeldOutError`], an empty sequence list, a sequence shorter than
//! two tokens (no next-token position exists, so the NLL is undefined), a zero vocabulary, a logits
//! buffer that is not `t × vocab`, a whole-model `forward` refusal (empty sequence, token past the
//! vocabulary, shape, overflow), an engine `state`/`step_logits_state` refusal and a head-row
//! refusal. It never panics and takes no `unsafe`.
//!
//! # Evidence grade
//!
//! `tested`: `tests/learn_fold_heldout.rs` on the crate's build path — identical models give zero
//! KL/TV and unit top-1, a perturbed `lm_head` row matches an independent `f64` softmax-KL
//! recomputation to `1e-6` relative, the mean over several sequences is the token-weighted mean of
//! the per-sequence values, and the named refusals hold. `heldout_engine_matches_grad_qwen05b` is a
//! **measurement**, not a parity control: it prints the `Q8Act` engine path's `|ΔKL|`, `|ΔTV|` and
//! top-1 against the `forward` path on the same 16 real tokens and asserts only that it ran.
//! No Lean refinement links this file; it is not on the extracted fragment.
//!
//! # Engine source and subsumption
//!
//! The engine source runs [`Engine::step_logits_state`] under [`KernelMode::Q8Act`] over the
//! sequence in token order: the shipped decode step (`Engine::next_token_state`, the CPU speed
//! receipts' `tg` workload) with the `Q8Act` head, so the measured logits are the engine's own
//! arithmetic, not a second forward. The empirical best for this comparison outside the crate is
//! llama.cpp's `llama-perplexity --kl-divergence`
//! (`~/llama.cpp/tools/perplexity/perplexity.cpp:1695`, mean at `:1949`); it is
//! `not-subsumable` — C++, outside this crate's build path, and the head arithmetic under test is
//! the crate's own, which that tool cannot substitute.

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "the i128/f64 boundary is the one deliberate conversion, made once per aggregate after \
              the exact integer sums; `t`/`v`/`r` are the schedule's position, vocabulary and row \
              indices"
)]

use crate::engine::{Engine, KernelMode, ONE};
use crate::learn::grad::head::{kl_row, log_softmax_row, tv_row, HeadError};
use crate::learn::grad::model::{forward, Config, ModelError, Weights};

/// A refused held-out evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeldOutError {
    /// A whole-model [`forward`] refused: an empty sequence, a token past the vocabulary, a shape
    /// disagreement or an overflow.
    Model(ModelError),
    /// A head row refused: a shape or overflow, or the unreachable `exp`/`ln` refusals.
    Head(HeadError),
    /// The evaluator's own shape check: an empty sequence list, a sequence shorter than two tokens,
    /// a zero vocabulary, a logits buffer that is not `t × vocab`, or token metadata that does not
    /// cover the scored rows.
    Shape(String),
    /// A checked `i128` accumulation left its range.
    Overflow,
    /// The engine source refused, or the two engines disagree on geometry: the carried text is the
    /// engine error's own message, prefixed by the phase and position.
    Engine(String),
}

/// The per-sequence wall of the engine source, in nanoseconds: the shipped
/// [`Engine::step_logits_state`] decode step over every position, teacher and student.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EngineWalls {
    /// Teacher and student [`Engine::step_logits_state`] wall plus row scoring, summed.
    pub wall_ns: u64,
}

/// The scored held-out quality of a student against a teacher.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeldOut {
    /// Scored positions `Σ_i T_i`: the denominator of [`HeldOut::kl`], [`HeldOut::tv`] and
    /// [`HeldOut::top1`].
    pub tokens: u64,
    /// Mean forward `KL(teacher ‖ student)` per position, in nats.
    pub kl: f64,
    /// Mean total variation `½Σ_x |p_teacher(x) − p_student(x)|` per position, in `[0, 1]`.
    pub tv: f64,
    /// Fraction of positions where the teacher and student `argmax` agree, in `[0, 1]`.
    pub top1: f64,
    /// Mean `−log p_student(next token)` in nats, over the `tokens − n_sequences` next-token
    /// positions (each sequence's last position is excluded).
    pub nll_student: f64,
    /// Mean `−log p_teacher(next token)` in nats, over the same positions as
    /// [`HeldOut::nll_student`].
    pub nll_teacher: f64,
}

/// Exact Q32.32 accumulators over the scored positions; the division into the `f64` means happens
/// once, in [`Acc::finish`].
#[derive(Debug, Clone, Copy, Default)]
struct Acc {
    /// Scored positions, `Σ_i T_i`.
    tokens: u64,
    /// Next-token positions, `Σ_i (T_i − 1)`.
    pairs: u64,
    /// `Σ` of the per-row forward KL in Q32.32.
    kl: i128,
    /// `Σ` of the per-row total variation in Q32.32.
    tv: i128,
    /// Positions whose teacher and student `argmax` agree.
    top1: u64,
    /// `Σ` of the teacher `log p(next token)` in Q32.32 (negative).
    nll_t: i128,
    /// `Σ` of the student `log p(next token)` in Q32.32 (negative).
    nll_s: i128,
}

/// First index of the row maximum; `None` for an empty row. Ties keep the lower index, as the
/// reference `argmax` does.
fn argmax(row: &[i64]) -> Option<usize> {
    let mut best: Option<(usize, i64)> = None;
    for (i, &z) in row.iter().enumerate() {
        match best {
            Some((_, b)) if b >= z => {}
            _ => best = Some((i, z)),
        }
    }
    best.map(|(i, _)| i)
}

/// Checked `i128` addition.
fn add(a: i128, b: i128) -> Result<i128, HeldOutError> {
    match a.checked_add(b) {
        Some(s) => Ok(s),
        None => Err(HeldOutError::Overflow),
    }
}

impl Acc {
    /// Accumulate paired logit rows: the per-row forward KL and total variation, the top-1
    /// agreement, and the next-token log-probabilities.
    ///
    /// `zt` and `zs` are `rows × vocab` Q32.32 row-major buffers; `tokens` covers the scored rows
    /// and is indexed at `r + 1` for the next-token position of row `r`, so a caller scoring a
    /// suffix of a sequence passes the suffix's first token onward and every row but the last still
    /// sees its successor. A row with no successor is excluded from the NLL exactly as a whole
    /// sequence's final position is.
    ///
    /// # Errors
    ///
    /// [`HeldOutError::Shape`] for a zero vocabulary, a logits length that is not `rows × vocab`,
    /// mismatched teacher/student lengths, or a token list shorter than the scored rows;
    /// [`HeldOutError::Head`] for a head-row refusal and [`HeldOutError::Overflow`] for a checked
    /// `i128` accumulation.
    ///
    /// # Complexity
    ///
    /// Time `O(rows·vocab)`: one [`kl_row`], one [`tv_row`], two `argmax` scans and at most two
    /// [`log_softmax_row`] calls per row, each `O(vocab)`. Space `O(1)`: no buffer is allocated;
    /// the logits belong to the caller.
    fn score_logits(
        &mut self,
        zt: &[i64],
        zs: &[i64],
        vocab: usize,
        tokens: &[u32],
    ) -> Result<(), HeldOutError> {
        if vocab == 0 {
            return Err(HeldOutError::Shape("zero vocabulary".to_string()));
        }
        if !zt.len().is_multiple_of(vocab) || zs.len() != zt.len() {
            return Err(HeldOutError::Shape(format!(
                "logits {} / {} not a multiple of vocab {vocab}",
                zt.len(),
                zs.len()
            )));
        }
        let rows = zt.len() / vocab;
        if tokens.len() < rows {
            return Err(HeldOutError::Shape(format!(
                "{} tokens for {rows} scored rows",
                tokens.len()
            )));
        }
        for r in 0..rows {
            let teacher_row = zt
                .get(r * vocab..(r + 1) * vocab)
                .ok_or_else(|| HeldOutError::Shape("teacher row past logits".to_string()))?;
            let student_row = zs
                .get(r * vocab..(r + 1) * vocab)
                .ok_or_else(|| HeldOutError::Shape("student row past logits".to_string()))?;
            let (kl, _) = kl_row(student_row, teacher_row).map_err(HeldOutError::Head)?;
            let (tv, _) = tv_row(student_row, teacher_row).map_err(HeldOutError::Head)?;
            self.kl = add(self.kl, i128::from(kl))?;
            self.tv = add(self.tv, i128::from(tv))?;
            if argmax(teacher_row) == argmax(student_row) {
                self.top1 += 1;
            }
            self.tokens += 1;
            if let Some(&next_tok) = tokens.get(r + 1) {
                let next = usize::try_from(next_tok).map_err(|_| HeldOutError::Overflow)?;
                let row_logp = |z: &[i64]| -> Result<i64, HeldOutError> {
                    let lp = log_softmax_row(z).map_err(HeldOutError::Head)?;
                    lp.get(next).copied().ok_or_else(|| {
                        HeldOutError::Shape("next token past vocabulary".to_string())
                    })
                };
                let lp_t = row_logp(teacher_row)?;
                let lp_s = row_logp(student_row)?;
                self.nll_t = add(self.nll_t, i128::from(lp_t))?;
                self.nll_s = add(self.nll_s, i128::from(lp_s))?;
                self.pairs += 1;
            }
        }
        Ok(())
    }

    /// Score one sequence: run the teacher's and the student's [`forward`], then accumulate the
    /// per-row KL and TV, the top-1 agreement, and the next-token log-probabilities.
    ///
    /// # Complexity
    ///
    /// Time `O(cost(forward_t) + cost(forward_s) + t·vocab)` where a [`forward`] is
    /// `O(t·n_embd·(n_embd + q_out + 2·kv_out + n_ff) + n_head·t²·head_dim + t·vocab·n_embd)`; the
    /// evaluator's own head cost is `O(t·vocab)`. Space `O(t·vocab)` for the two logit buffers plus
    /// the `Tape` a [`forward`] returns and this call drops.
    fn score(
        &mut self,
        teacher: &Weights,
        student: &Weights,
        cfg: &Config,
        tokens: &[u32],
    ) -> Result<(), HeldOutError> {
        let t = tokens.len();
        if t < 2 {
            return Err(HeldOutError::Shape(format!(
                "sequence length {t} < 2: no next-token position"
            )));
        }
        let v = cfg.vocab;
        if v == 0 {
            return Err(HeldOutError::Shape("zero vocabulary".to_string()));
        }
        let (lt, _tape_t) = forward(teacher, cfg, tokens).map_err(HeldOutError::Model)?;
        let (ls, _tape_s) = forward(student, cfg, tokens).map_err(HeldOutError::Model)?;
        if lt.len() != t * v || ls.len() != t * v {
            return Err(HeldOutError::Shape(format!(
                "logits {} × {} != t·vocab {}",
                lt.len(),
                ls.len(),
                t * v
            )));
        }
        self.score_logits(&lt, &ls, v, tokens)
    }

    /// Score one sequence on the shipped engine path: the teacher's and the student's
    /// [`Engine::step_logits_state`] under [`KernelMode::Q8Act`] — the decode step behind
    /// `Engine::next_token_state` and the CPU speed receipts' `tg` workload — run over the sequence
    /// in token order, each row pair entering the same [`Acc::score_logits`] the [`forward`] source
    /// uses.
    ///
    /// # Errors
    ///
    /// [`HeldOutError::Shape`] for a sequence shorter than two tokens or a zero vocabulary;
    /// [`HeldOutError::Engine`] for a `state`/`step_logits_state` refusal, a teacher/student
    /// vocabulary or embedding-width disagreement, or a logits row of the wrong width; plus
    /// [`Acc::score_logits`]'s refusals.
    ///
    /// # Complexity
    ///
    /// Time `O(cost(step_t) + cost(step_s) + t·vocab)`: `t` single-column `Q8Act` decode steps per
    /// engine, each `O(n_layer·(n_embd² + n_embd·n_ff) + pos·n_head·head_dim + vocab·n_embd)` for
    /// its position `pos`, then the scoring `O(t·vocab)`. Space `O(t·n_embd)` for the two KV
    /// caches plus one `O(vocab)` logit row per engine; no `t·vocab` buffer is live.
    fn score_engines(
        &mut self,
        teacher: &Engine,
        student: &Engine,
        tokens: &[u32],
    ) -> Result<EngineWalls, HeldOutError> {
        let t = tokens.len();
        if t < 2 {
            return Err(HeldOutError::Shape(format!(
                "sequence length {t} < 2: no next-token position"
            )));
        }
        let v = teacher.config.vocab_size;
        if v == 0 {
            return Err(HeldOutError::Shape("zero vocabulary".to_string()));
        }
        if student.config.vocab_size != v || student.config.n_embd != teacher.config.n_embd {
            return Err(HeldOutError::Engine(format!(
                "student geometry {}×{} != teacher {v}×{}",
                student.config.vocab_size, student.config.n_embd, teacher.config.n_embd
            )));
        }

        let mut st_t = teacher
            .state(t)
            .map_err(|e| HeldOutError::Engine(format!("teacher state: {e}")))?;
        let mut st_s = student
            .state(t)
            .map_err(|e| HeldOutError::Engine(format!("student state: {e}")))?;
        let start = std::time::Instant::now();
        for (r, &tok) in tokens.iter().enumerate() {
            let zt = teacher
                .step_logits_state(&mut st_t, tok, KernelMode::Q8Act)
                .map_err(|e| HeldOutError::Engine(format!("teacher step {r}: {e}")))?;
            let zs = student
                .step_logits_state(&mut st_s, tok, KernelMode::Q8Act)
                .map_err(|e| HeldOutError::Engine(format!("student step {r}: {e}")))?;
            if zt.len() != v || zs.len() != v {
                return Err(HeldOutError::Engine(format!(
                    "logits rows {} / {} != vocab {v}",
                    zt.len(),
                    zs.len()
                )));
            }
            let tail = tokens
                .get(r..)
                .ok_or_else(|| HeldOutError::Shape(format!("position {r} past sequence")))?;
            self.score_logits(&zt, &zs, v, tail)?;
        }
        Ok(EngineWalls {
            wall_ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
        })
    }

    /// Divide the exact accumulators into the `f64` means, the one point of conversion.
    ///
    /// # Errors
    ///
    /// [`HeldOutError::Shape`] when no next-token position was scored.
    ///
    /// # Complexity
    /// `O(1)`.
    fn finish(self) -> Result<HeldOut, HeldOutError> {
        if self.tokens == 0 || self.pairs == 0 {
            return Err(HeldOutError::Shape("no scored positions".to_string()));
        }
        let one = ONE as f64;
        let n = self.tokens as f64;
        let m = self.pairs as f64;
        Ok(HeldOut {
            tokens: self.tokens,
            kl: self.kl as f64 / one / n,
            tv: self.tv as f64 / one / n,
            top1: self.top1 as f64 / n,
            nll_student: -(self.nll_s as f64) / one / m,
            nll_teacher: -(self.nll_t as f64) / one / m,
        })
    }
}

/// The held-out quality of `student` against `teacher` over one token sequence.
///
/// # Errors
///
/// As [`HeldOutError`]: [`HeldOutError::Shape`] for a sequence shorter than two tokens (or a zero
/// vocabulary), [`HeldOutError::Model`] for a [`forward`] refusal (a token past the vocabulary, an
/// empty sequence, a shape or overflow), [`HeldOutError::Head`] for a head-row refusal, and
/// [`HeldOutError::Overflow`] for a checked-`i128` accumulation.
///
/// # Complexity
///
/// As [`Acc::score`].
pub fn held_out_seq(
    teacher: &Weights,
    student: &Weights,
    cfg: &Config,
    tokens: &[u32],
) -> Result<HeldOut, HeldOutError> {
    let mut acc = Acc::default();
    acc.score(teacher, student, cfg, tokens)?;
    acc.finish()
}

/// The held-out quality of `student` against `teacher` over `sequences`: each sequence is scored on
/// its own causal [`forward`] and the aggregates are pooled as exact integers, so the reported means
/// are the token-weighted means of the per-sequence values
/// ([`held_out_seq`]) — KL, TV and top-1 weighted by each sequence's `T_i` positions, the NLL by its
/// `T_i − 1` next-token positions.
///
/// # Errors
///
/// [`HeldOutError::Shape`] when `sequences` is empty, plus every refusal of [`held_out_seq`].
///
/// # Complexity
///
/// Time and space are the sum over the sequences of [`held_out_seq`]'s cost; the pooling is the
/// exact integer accumulation of [`Acc`].
pub fn held_out(
    teacher: &Weights,
    student: &Weights,
    cfg: &Config,
    sequences: &[&[u32]],
) -> Result<HeldOut, HeldOutError> {
    if sequences.is_empty() {
        return Err(HeldOutError::Shape("no sequences".to_string()));
    }
    let mut acc = Acc::default();
    for &seq in sequences {
        acc.score(teacher, student, cfg, seq)?;
    }
    acc.finish()
}

/// [`held_out`] on the engine's shipped path: the teacher's and the student's
/// [`Engine::step_logits_state`](crate::engine::Engine) under [`KernelMode::Q8Act`] produce each
/// position's logits, which enter the same [`Acc::score_logits`] accumulator. Values pool exactly
/// as [`held_out`] pools them: token-weighted means over the per-sequence scores.
///
/// # Errors
///
/// [`HeldOutError::Shape`] when `sequences` is empty, plus every refusal of the engine source
/// ([`HeldOutError::Engine`] for a state or step refusal and for a teacher/student vocabulary or
/// embedding-width disagreement; [`HeldOutError::Head`] and [`HeldOutError::Overflow`] from the row
/// accumulation).
///
/// # Complexity
///
/// Time and space are the sum over the sequences of the engine source's cost; the pooling is the
/// exact integer accumulation of [`Acc`].
pub fn held_out_engines(
    teacher: &Engine,
    student: &Engine,
    sequences: &[&[u32]],
) -> Result<HeldOut, HeldOutError> {
    if sequences.is_empty() {
        return Err(HeldOutError::Shape("no sequences".to_string()));
    }
    let mut acc = Acc::default();
    for &seq in sequences {
        acc.score_engines(teacher, student, seq)?;
    }
    acc.finish()
}

/// One [`HeldOut`] per sequence of [`held_out_engines`], for the per-chunk spread.
///
/// # Errors
///
/// [`HeldOutError::Shape`] when `sequences` is empty, plus every [`held_out_engines`] refusal.
///
/// # Complexity
///
/// As [`held_out_engines`], with `O(sequences.len())` returned values.
pub fn held_out_engines_per_seq(
    teacher: &Engine,
    student: &Engine,
    sequences: &[&[u32]],
) -> Result<Vec<HeldOut>, HeldOutError> {
    Ok(held_out_engines_per_seq_timed(teacher, student, sequences)?.0)
}

/// [`held_out_engines_per_seq`] plus each sequence's [`EngineWalls`] wall.
///
/// # Errors
///
/// As [`held_out_engines_per_seq`].
///
/// # Complexity
///
/// As [`held_out_engines_per_seq`]; the two `Instant` reads per sequence are `O(1)`.
pub fn held_out_engines_per_seq_timed(
    teacher: &Engine,
    student: &Engine,
    sequences: &[&[u32]],
) -> Result<(Vec<HeldOut>, Vec<EngineWalls>), HeldOutError> {
    if sequences.is_empty() {
        return Err(HeldOutError::Shape("no sequences".to_string()));
    }
    let mut scores = Vec::with_capacity(sequences.len());
    let mut walls = Vec::with_capacity(sequences.len());
    for &seq in sequences {
        let mut acc = Acc::default();
        walls.push(acc.score_engines(teacher, student, seq)?);
        scores.push(acc.finish()?);
    }
    Ok((scores, walls))
}
