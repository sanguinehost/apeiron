//! simt_nn::spec — **the gemv SPEC: five orthogonal axes, ONE selection, ONE legality relation.**
//!
//! [`crate::simt::Carrier`]'s doc states the rule this module extends to the rest of the kernel:
//! *"precision is one dial, not a kernel zoo"*. That held for the code width and for nothing else.
//! The other four axes a gemv varies along — where the SCALES live, how rows MAP onto invocations,
//! what is FUSED into the dispatch, how split-K partials REDUCE — were encoded in *function
//! identity*: five sibling `make_*` dispatchers in `bedrock::serve::gpu` choosing among 19
//! `gemv_*` emitters, with the cross-cutting concerns repeated as `bool` flags on each
//! (`two_level` appears on three of them, each carrying its own hand-written warning about the
//! same bug).
//!
//! **That is not a style complaint; it has a measured cost.** `results/shape-derived-gemv.md`:
//!
//! > *"The cause: `ffn_down` does not dispatch through `make_gemv`. `actqfuse` is on by default,
//! > so the down projection goes through `make_actq_gemv`, which computes `segs(m, k/32)` and
//! > emits `gemv_actq_bytes_sc` with **no cooperative branch at all**. The policy was wired into
//! > the wrong one of two sibling functions, and the A/B could only ever have measured noise."*
//!
//! A shape-selection policy was written once, landed on one sibling, and the shape it targeted
//! flowed through the other. Nothing in the type system could have caught it, because "which
//! kernel" was a function name and "which policy applies" was a comment.
//!
//! **The fix is `Carrier`'s own shape, applied to the remaining axes.** A [`GemvSpec`] is a value.
//! A policy is a total function on that value ([`GemvSpec::legalize`]). Selection is one `match`
//! ([`GemvSpec::emit`]). Byte accounting is one method ([`GemvSpec::chart_bytes`]). A caller can no
//! longer *reach* an emitter without going through the spec, so a policy cannot miss a sibling.
//!
//! **Three illegal combinations become unrepresentable rather than asserted.** `make_gemv` opened
//! with `assert!(!(two_level && sb), …)` — two booleans encoding three mutually exclusive scale
//! layouts, so the fourth state had to be excluded at runtime. [`ScaleFace`] has three variants and
//! the assert is deleted, not moved.

use super::*;
use crate::simt::Carrier;

/// **Where a chart's SCALES live, and at what granularity.** Orthogonal to [`Carrier`], which says
/// how wide the CODES are — the two were entangled only because the superblock face arrived as its
/// own emitter family rather than as a value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScaleFace {
    /// Per-32 `(d, m)` affine pairs in the `dm` stream — the default chart.
    Dm32,
    /// The `qsub`/`qsup` two-level face: `sup` shares `dm`'s buffer at a compile-time offset, so
    /// the binding list is unchanged and only the module differs. Half the scale metadata for
    /// identical values (`results/scale-ladder.md`).
    TwoLevel,
    /// The SUPERBLOCK face — `rowq2s`/`dsb`, one f16 scale per 256 weights, transposed so lane `r`
    /// and lane `r+1` read adjacent words. `dm` carries the `dsb` stream; the six bindings are
    /// unchanged. Q2 codes only today — see [`GemvSpec::legalize`].
    Superblock,
}

/// **How output rows are MAPPED onto invocations.** Was: three emitter families plus a
/// `DEC_COOP` env read buried in `make_gemv`, invisible to the four sibling dispatchers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mapping {
    /// One thread per output row, occupancy bought with split-K. The shipped default.
    Row,
    /// Cooperative lanes share a row and reduce in the workgroup — complete in ONE dispatch, so
    /// `segs` is forced to 1. Declares a constant 256 B of shared at any `k`.
    ///
    /// ⛔ **Measured 2.1× WORSE at the long-`k` shape it was aimed at** (`ffn_down` m5120 k17408:
    /// 536 → 251 GB/s, `results/shape-derived-gemv.md`). Retained as an override for the in-chain
    /// A/B, not as a policy target.
    Coop {
        /// Output rows per workgroup.
        rows_per_wg: u32,
        /// `local_size.x`.
        local: u32,
        /// Lanes per row; at 32 (NVIDIA's warp) the reduce is a shuffle with no shared array.
        subgroup: u32,
    },
    /// The superblock register tile — `rt` rows × `bt` columns per invocation, `uu` 128-weight
    /// units unrolled. `bt` MUST be 1 under split-K (`WorkgroupId.y` is then the segment, not the
    /// batch column) — enforced in [`GemvSpec::legalize`], previously a comment at the call site.
    SbTile {
        /// Batch (column) tile.
        bt: u32,
        /// Row tile.
        rt: u32,
        /// 128-weight units unrolled per inner iteration (1 or 2).
        uu: u32,
    },
}

/// **What is FOLDED INTO the gemv dispatch.** Was: *which of three functions you called* — which is
/// precisely how the coop policy missed `ffn_down`.
///
/// The motive for fusing is not arithmetic and not bytes: a compute→compute `vkCmdPipelineBarrier`
/// drains this box's pipeline for a measured 3.71 µs, and at batch 1 the token is barriers
/// (`results/dispatch-barriers-are-the-token.md`: 72.5 % of the 0.8B token). So `Fusion` is a
/// dispatch-COUNT axis, and making it a value is what lets it be traded per shape.
// no `Eq`: `Norm` carries the rmsnorm epsilon, and an f32 has no total equality
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fusion {
    /// Bare gemv; the activation arrives already normed and quantized.
    None,
    /// RMSNorm + gain (+ optional residual) folded in. Serves `attn_qkv` and `ffn_gateup`.
    Norm {
        /// RMSNorm epsilon.
        eps: f32,
        /// Residual source stride, if the residual add is folded in too.
        residual: Option<u32>,
    },
    /// The activation producer + its int8 quantize folded in. Serves `o`, `ffn_down` and the
    /// delta-net `out`.
    ///
    /// ⚠ **This variant stages the whole activation in shared at 5.25 B per `k` element**, so it is
    /// the one [`GemvSpec::row_shared_bytes`] speaks about. At `k = 17408` that is 91 392 B against
    /// this box's 49 152 limit — the driver accepts the over-declaration silently and charges ~42 %
    /// of the kernel's bandwidth for it.
    ActQ(ActQ),
}

/// **How split-K partials are COMBINED.** Was `fold: bool` beside a separately-computed `segs`,
/// with the invariant "`segs` means something different in each case" left to the reader.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reduce {
    /// `segs` partials land in a scratch buffer; a second `gemv(F32)` dispatch reduces them.
    /// `segs == 1` means no second dispatch.
    Stage2(u32),
    /// `segs` partials are written straight into `y` (sized `segs·m`); the CONSUMER reduces, so
    /// there is no stage-2 dispatch at any `segs`.
    Fold(u32),
}

impl Reduce {
    /// The split-K segment count, whichever face this is.
    pub const fn segs(self) -> u32 {
        match self {
            Reduce::Stage2(s) | Reduce::Fold(s) => s,
        }
    }
    /// Whether a second reduce dispatch is needed.
    pub const fn needs_stage2(self) -> bool {
        matches!(self, Reduce::Stage2(s) if s > 1)
    }
}

/// Why a [`GemvSpec`] cannot be dispatched as written. Each variant is a rule that was previously a
/// `assert!`, a doc comment, or nothing at all.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SpecError {
    /// A chart carrier's `k` must be a multiple of 32 (the block granularity).
    KNotBlockAligned {
        /// The offending `k`.
        k: u32,
    },
    /// [`ScaleFace::Superblock`] carries 2-bit codes only — the emitter hardcodes `0x03030303`
    /// and the `>> 2p` plane shift.
    SuperblockNeedsQ2 {
        /// The carrier that was asked for.
        carrier: Carrier,
    },
    /// [`ScaleFace::Superblock`] requires [`Mapping::SbTile`], and vice versa: the transposed
    /// `sb·16m + rowc + wq·m` layout is only read by that kernel.
    SuperblockNeedsSbTile,
    /// `bt > 1` under split-K claims `WorkgroupId.y` twice.
    SbTileBatchUnderSplitK {
        /// The offending batch tile.
        bt: u32,
        /// The segment count that conflicts with it.
        segs: u32,
    },
    /// [`Mapping::Coop`] produces the final value in one dispatch; split-K partials are meaningless.
    CoopWithSplitK {
        /// The segment count that was requested.
        segs: u32,
    },
    /// A direct (non-chart) carrier has no scale stream to place.
    DirectCarrierHasNoScaleFace {
        /// The carrier that was asked for.
        carrier: Carrier,
        /// The face that was requested for it.
        scale: ScaleFace,
    },
    /// **No fused kernel exists for this mapping.** `gemv_actq`/`gemv_norm` are row-mapped only —
    /// there is no cooperative and no superblock variant of either.
    ///
    /// This is the boundary `results/shape-derived-gemv.md` walked into from the other side. The
    /// coop policy lived inside `make_gemv`, so when `actqfuse` routed `ffn_down` to
    /// `make_actq_gemv` the policy simply did not apply — silently, and the A/B could only measure
    /// noise. Naming the combination illegal converts that silence into a refusal at the one place
    /// that builds pipelines: a fused shape is now KNOWN not to take the mapping, rather than
    /// appearing to take it and quietly not doing so.
    /// A MoE spec asked for something `gemv_dp4a_moe_bytes` cannot do. It has no split-K, no
    /// cooperative mapping, no fusion and no alternate scale face — so rather than dropping the
    /// request silently (which is what calling the emitter directly did), say which axis was
    /// unhonourable.
    MoeSupportsRowDm32Only {
        /// The axis that could not be honoured.
        axis: &'static str,
    },
    NoFusedKernelForMapping {
        /// The mapping that has no fused variant.
        mapping: Mapping,
    },
    /// **This face runs the DP4A chart fold and a direct carrier has no chart.** Two axes demand
    /// it and one conspicuously does not: [`Fusion::ActQ`] quantizes the activation *into* the
    /// chart (`gemv_actq_module_sc`), [`Mapping::Coop`] folds through `dp4a_chart_fold`
    /// unconditionally — while [`Fusion::Norm`] IS carrier-generic (F16 direct or chart DP4A).
    ///
    /// **Both rules were found by `emit_is_total_wherever_legalize_passes`, not by reading.** They
    /// lived as an `assert!` in one emitter (`simt_nn/norm.rs:348`) and as an unchecked `u32`
    /// multiply in the other (`simt_nn/mod.rs:1425`: at F16 `wbits` is 16 and
    /// `0x01010101 * ((1<<16)-1)` overflows), so a spec could pass `legalize` and still abort at
    /// pipeline-build time. Neither is reachable from the served stack — `resolve_decode` gates
    /// coop on `is_chart()` exactly as `make_gemv` did — but a legality relation that does not
    /// contain a legality rule is the same defect one level up, which is why the sweep is a test
    /// and not a doc comment.
    NeedsChartCarrier {
        /// Which face demanded a chart.
        face: &'static str,
        /// The direct carrier that was asked for.
        carrier: Carrier,
    },
    /// **[`Carrier::Q3`] is only wired against [`ScaleFace::TwoLevel`]**, and the reason is the
    /// whole point of the carrier: its 3.5 bpw is 3.0 of codes plus 0.5 of scale, so the
    /// single-level face would make it 4.0 and pointless (`results/scale-ladder.md` — our 3-bit
    /// chart ties llama.cpp IQ3_XXS at 3.5 bpw; at 4.0 it does not).
    ///
    /// A *design* constraint that lived as an `assert!` at `simt/dp4a.rs:368`. Fourth rule the
    /// sweep pulled out of an emitter and into the relation.
    Q3NeedsTwoLevel,
    /// **[`Carrier::Q5`] is a TWO-PLANE chart, and only the fused folds read plane B.** The bare row
    /// (`gemv_dp4a_dyn_bytes_sc`), the cooperative and the superblock emitters never thread
    /// `plane_b_off` into a decode that consumes it, and the MoE emitter has no plane-B at all — so
    /// the fifth bit of every code would be dropped and the kernel would serve a plausible Q4. Only
    /// [`Fusion::ActQ`]/[`Fusion::Norm`] pass `carrier.plane_b_off(m, k)` into
    /// `Kb::dp4a_chart_fold`, and only on [`ScaleFace::Dm32`]: the two-level and superblock faces
    /// store no plane-B word.
    Q5NeedsFusedFold {
        /// The axis that could not be honoured.
        axis: &'static str,
    },
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecError::KNotBlockAligned { k } => {
                write!(f, "chart carriers need k % 32 == 0, got k={k}")
            }
            SpecError::MoeSupportsRowDm32Only { axis } => write!(
                f,
                "gemv_dp4a_moe has no split-K, no cooperative mapping, no fusion and one scale \
face — the `{axis}` axis cannot be honoured on a MoE spec. Calling the emitter directly \
dropped this request silently; that is what routing it through GemvSpec removes."
            ),
            SpecError::SuperblockNeedsQ2 { carrier } => write!(
                f,
                "the superblock scale face carries 2-bit codes only, got {}",
                carrier.name()
            ),
            SpecError::SuperblockNeedsSbTile => write!(
                f,
                "the superblock scale face and the SbTile mapping imply each other — the \
transposed chart is read by no other kernel"
            ),
            SpecError::SbTileBatchUnderSplitK { bt, segs } => write!(
                f,
                "bt={bt} under segs={segs}: WorkgroupId.y is the SEGMENT under split-K, so the \
batch tile must be 1"
            ),
            SpecError::CoopWithSplitK { segs } => write!(
                f,
                "the cooperative mapping is complete in one dispatch; segs={segs} would reduce an \
identity"
            ),
            SpecError::DirectCarrierHasNoScaleFace { carrier, scale } => write!(
                f,
                "{} is a direct carrier and has no scale stream to place on {scale:?}",
                carrier.name()
            ),
            SpecError::NoFusedKernelForMapping { mapping } => write!(
                f,
                "gemv_actq/gemv_norm are row-mapped only — there is no fused variant for \
{mapping:?}"
            ),
            SpecError::NeedsChartCarrier { face, carrier } => write!(
                f,
                "{face} runs the DP4A chart fold — {} is a direct carrier and has no chart",
                carrier.name()
            ),
            SpecError::Q3NeedsTwoLevel => write!(
                f,
                "Carrier::Q3 is only wired against the two-level scale face — 3.0 bits of codes \
plus 0.5 of scale is the 3.5 bpw that ties IQ3_XXS; the single-level face makes it 4.0"
            ),
            SpecError::Q5NeedsFusedFold { axis } => write!(
                f,
                "Carrier::Q5 is a two-plane chart and only the fused folds read plane B, so the \
`{axis}` axis cannot be honoured — this emitter would drop the fifth bit of every code and serve \
a plausible Q4. Q5 needs a Fusion::ActQ/Fusion::Norm fold on ScaleFace::Dm32, no experts."
            ),
        }
    }
}

impl std::error::Error for SpecError {}

/// **The gemv, as a value.** Five orthogonal axes; every kernel in the family is a point in this
/// space, and every policy is a function on it.
///
/// Buffers are deliberately NOT here. The spec answers *which module, what geometry, how many
/// bytes* — the dispatcher still owns handles, because the binding LIST varies with [`Fusion`]
/// while everything else does not.
/// **Per-EXPERT weight indirection** — the MoE gemv's extra axis.
///
/// `make_moe_gemv` was the fifth sibling, and the only one that reached a `gemv_*` emitter without
/// passing through this module: it called `gemv_dp4a_moe_bytes` directly, hardcoding `segs = 1` and
/// never consulting `resolve_decode`. That is the same shape as the bug in
/// `results/shape-derived-gemv.md` — a policy that cannot see one of its call sites — and it is live
/// on the 30B-A3B's hot path.
///
/// Routing it here does NOT give it split-K or the cooperative mapping: `gemv_dp4a_moe_bytes` has
/// neither. What it gives is a REFUSAL instead of silence, so a future policy that sets `Reduce`
/// or `Mapping::Coop` on a MoE spec fails loudly rather than being quietly dropped on the floor
/// (see [`GemvSpec::legalize`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Experts {
    /// Weight words per expert — `(k/pack)·m`, the stride `sel` indexes.
    pub wpe: u32,
    /// Scale words per expert — `(k/32)·m`.
    pub dmpe: u32,
    /// Multiply each expert's contribution by its router weight (`swt`).
    pub weighted: bool,
    /// The activation is indexed per selected expert rather than shared.
    pub x_slot: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GemvSpec {
    /// Code width — the dial that was already a dial.
    pub carrier: Carrier,
    /// Scale granularity and layout.
    pub scale: ScaleFace,
    /// Row → invocation mapping.
    pub mapping: Mapping,
    /// What rides along in the same dispatch.
    pub fusion: Fusion,
    /// How split-K partials come back together.
    pub reduce: Reduce,
    /// Per-expert indirection, `None` for every dense gemv.
    pub experts: Option<Experts>,
}

impl GemvSpec {
    /// The shipped default for a chart carrier: per-32 scales, row mapping, nothing fused,
    /// stage-2 reduce over `segs`.
    pub const fn row(carrier: Carrier, segs: u32) -> Self {
        Self {
            carrier,
            scale: ScaleFace::Dm32,
            mapping: Mapping::Row,
            fusion: Fusion::None,
            reduce: Reduce::Stage2(segs),
            experts: None,
        }
    }

    /// The MoE gemv: row-mapped, per-32 scales, nothing fused, no split-K — plus [`Experts`].
    pub const fn moe(carrier: Carrier, wpe: u32, dmpe: u32, weighted: bool, x_slot: bool) -> Self {
        Self {
            carrier,
            scale: ScaleFace::Dm32,
            mapping: Mapping::Row,
            fusion: Fusion::None,
            reduce: Reduce::Stage2(1),
            experts: Some(Experts {
                wpe,
                dmpe,
                weighted,
                x_slot,
            }),
        }
    }

    /// This spec with the two-level (`qsub`/`qsup`) scale face selected when `on`.
    ///
    /// Exists so the three dispatchers that carried `two_level: bool` can be ported without
    /// restating the mapping — and so that omitting it is a visible `false` at one site rather than
    /// a forgotten argument at three. Forgetting it read a `[sub][sup]` buffer as f16 pairs: finite
    /// logits, wrong model, and the 0.8B emitting token 0 forty times at a healthy tok/s.
    /// A direct carrier has no scale stream, and the shipped `chart_bytes_tl` IGNORES `two_level`
    /// for one — so this is a no-op there rather than a spec [`ScaleFace::TwoLevel`] that
    /// [`Self::legalize`] would then reject. Setting `scale` directly still errors, which is the
    /// case worth catching.
    pub const fn with_two_level(mut self, on: bool) -> Self {
        if on && self.carrier.is_chart() {
            self.scale = ScaleFace::TwoLevel;
        }
        self
    }

    /// This spec on the superblock face — `ScaleFace::Superblock` and the `SbTile` mapping are one
    /// choice, so they are made together and [`Self::legalize`] rejects them apart.
    pub const fn superblock(bt: u32, rt: u32, uu: u32, segs: u32) -> Self {
        Self {
            carrier: Carrier::Q2,
            scale: ScaleFace::Superblock,
            mapping: Mapping::SbTile { bt, rt, uu },
            fusion: Fusion::None,
            reduce: Reduce::Stage2(segs),
            experts: None,
        }
    }

    /// This spec with `fusion` folded in.
    pub const fn fused(mut self, fusion: Fusion) -> Self {
        self.fusion = fusion;
        self
    }

    /// **The decode-time spec as the CALLER states it** — carrier, scale face, and which reduce
    /// FACE it wants; the mapping and the segment count are the dispatcher's to resolve from the
    /// shape (`bedrock::serve::gpu::resolve_decode`).
    ///
    /// `segs` starts at 1, so a spec that never reaches the resolver is correct-but-unsplit rather
    /// than illegal — a missing policy must not be a wrong answer.
    pub const fn decode(carrier: Carrier, fold: bool) -> Self {
        Self {
            carrier,
            scale: ScaleFace::Dm32,
            mapping: Mapping::Row,
            fusion: Fusion::None,
            reduce: if fold {
                Reduce::Fold(1)
            } else {
                Reduce::Stage2(1)
            },
            experts: None,
        }
    }

    /// [`Self::decode`] on the superblock face. `bt`/`rt`/`uu` stay 1 until a decode tile sweep
    /// says otherwise — `bt` is REQUIRED to be 1 under split-K anyway.
    pub const fn decode_superblock(fold: bool) -> Self {
        Self {
            carrier: Carrier::Q2,
            scale: ScaleFace::Superblock,
            mapping: Mapping::SbTile {
                bt: 1,
                rt: 1,
                uu: 1,
            },
            fusion: Fusion::None,
            reduce: if fold {
                Reduce::Fold(1)
            } else {
                Reduce::Stage2(1)
            },
            experts: None,
        }
    }

    /// This spec with the segment count replaced, keeping whichever reduce FACE it already carries.
    /// The face is the caller's choice; the count is the shape's.
    pub const fn with_segs(mut self, segs: u32) -> Self {
        self.reduce = match self.reduce {
            Reduce::Fold(_) => Reduce::Fold(segs),
            Reduce::Stage2(_) => Reduce::Stage2(segs),
        };
        self
    }

    /// This spec with the reduce face replaced.
    pub const fn reducing(mut self, reduce: Reduce) -> Self {
        self.reduce = reduce;
        self
    }

    /// **Shared memory the ROW mapping declares, in bytes — the constant that must not drift.**
    ///
    /// `gemv_actq`/`gemv_norm` stage the whole activation in shared: `shared_f32(n)` +
    /// `shared_u32(n/4)` + two `shared_f32(blocks)` = **exactly 5.25 B per `k` element**, and
    /// split-K does NOT reduce it (segs 1…32 at k=17408 all report 91 392).
    /// `results/gemv-shared-is-over-the-limit.md` measured it; `bedrock/src/serve/gpu.rs` used to
    /// re-derive it inline with the comment *"re-deriving 5.25 B/k at the next call site is how a
    /// constant drifts from the measurement that produced it"*. This is that one call site.
    ///
    /// Returns 0 for mappings that do not stage in shared ([`Mapping::Coop`] declares a constant
    /// 256 B at any `k` and is structurally immune; [`Mapping::SbTile`] allocated zero shared at
    /// `bt4×rt8` before `SB_LDS`).
    pub fn row_shared_bytes(&self, k: u32) -> u32 {
        match self.mapping {
            Mapping::Row if self.carrier.is_chart() => (f64::from(k) * 5.25) as u32,
            _ => 0,
        }
    }

    /// Whether this spec declares more shared memory than the device permits. The driver ACCEPTS
    /// the over-declaration — `vkCreateComputePipelines` returns success — so this never surfaces
    /// as an error, only as bandwidth (~42 % of `ffn_down`'s).
    pub fn over_shared_limit(&self, k: u32, max_shared: u32) -> bool {
        self.row_shared_bytes(k) > max_shared
    }

    /// **The legality relation — every cross-axis rule in the family, in one total function.**
    ///
    /// Previously: one `assert!` in `make_gemv`, one `assert!` inside the superblock emitter, two
    /// doc comments at call sites, and one rule (`Coop ⇒ segs == 1`) that was maintained by the
    /// caller happening to write `let s = if coop { 1 } else { segs(m, span) }`.
    ///
    /// Called by [`Self::emit`], so an illegal spec cannot reach a pipeline.
    pub fn legalize(&self, k: u32) -> Result<(), SpecError> {
        if self.carrier.is_chart() && !k.is_multiple_of(32) {
            return Err(SpecError::KNotBlockAligned { k });
        }
        // MoE first: its emitter supports exactly one point on the other four axes, so a spec that
        // strays gets a NAMED axis rather than a generic complaint about mapping or scale.
        if self.experts.is_some() {
            let axis = if self.scale != ScaleFace::Dm32 {
                Some("scale")
            } else if !matches!(self.mapping, Mapping::Row) {
                Some("mapping")
            } else if !matches!(self.fusion, Fusion::None) {
                Some("fusion")
            } else if self.reduce.segs() != 1 {
                Some("reduce")
            } else if !self.carrier.is_chart() {
                Some("carrier")
            } else {
                None
            };
            if let Some(axis) = axis {
                return Err(SpecError::MoeSupportsRowDm32Only { axis });
            }
        }
        if !self.carrier.is_chart() && self.scale != ScaleFace::Dm32 {
            return Err(SpecError::DirectCarrierHasNoScaleFace {
                carrier: self.carrier,
                scale: self.scale,
            });
        }
        let segs = self.reduce.segs();
        // The fused emitters take no mapping argument, so `emit`'s fusion arm would otherwise
        // SWALLOW a non-row mapping — silently producing the row kernel for a spec that asked for
        // coop. That silence is the defect this module exists to remove, so it is a refusal.
        if !matches!(self.fusion, Fusion::None) && !matches!(self.mapping, Mapping::Row) {
            return Err(SpecError::NoFusedKernelForMapping {
                mapping: self.mapping,
            });
        }
        if self.carrier == Carrier::Q3 && self.scale != ScaleFace::TwoLevel {
            return Err(SpecError::Q3NeedsTwoLevel);
        }
        // Q5's high bit lives in a plane only the fused folds read; every other emitter would serve
        // the low four bits of every weight (a plausible Q4). Name the axis, as the MoE rule does.
        if self.carrier == Carrier::Q5 {
            let axis = if self.experts.is_some() {
                Some("experts")
            } else if self.scale != ScaleFace::Dm32 {
                Some("scale")
            } else if matches!(self.fusion, Fusion::None) {
                Some("fusion")
            } else {
                None
            };
            if let Some(axis) = axis {
                return Err(SpecError::Q5NeedsFusedFold { axis });
            }
        }
        if !self.carrier.is_chart() {
            if matches!(self.fusion, Fusion::ActQ(_)) {
                return Err(SpecError::NeedsChartCarrier {
                    face: "gemv_actq",
                    carrier: self.carrier,
                });
            }
            if matches!(self.mapping, Mapping::Coop { .. }) {
                return Err(SpecError::NeedsChartCarrier {
                    face: "gemv_coop",
                    carrier: self.carrier,
                });
            }
        }
        match (self.scale, self.mapping) {
            (ScaleFace::Superblock, Mapping::SbTile { bt, .. }) => {
                // **This is now a PACKER limit, not an emitter one.** `gemv_dp4a_sb_module` takes a
                // `Carrier` and derives its mask/shift/stride from `pack()`, so Q4 and Q8 emit
                // correctly. What does not exist is a chart to feed them: the packer's only
                // superblock face is `q2sb_chart`, which asserts `code <= 2` and lays
                // `s·(code − 1)` with ONE stream and no `lo`, licensed by
                // `native_ternary_zero_residual` — an argument about NATIVELY TERNARY tensors that
                // does not extend to a symmetric 4-bit block at 256-wide granularity.
                //
                // So the refusal is kept deliberately: opening it would let a caller build a kernel
                // with no data behind it, which is precisely the silent-wrong-answer class this
                // module exists to make unrepresentable. It lifts when `rowq4s`/`rowq8s` exist AND
                // their quality is gated, not when the emitter compiles.
                if self.carrier != Carrier::Q2 {
                    return Err(SpecError::SuperblockNeedsQ2 {
                        carrier: self.carrier,
                    });
                }
                if bt > 1 && segs > 1 {
                    return Err(SpecError::SbTileBatchUnderSplitK { bt, segs });
                }
            }
            (ScaleFace::Superblock, _) | (_, Mapping::SbTile { .. }) => {
                return Err(SpecError::SuperblockNeedsSbTile)
            }
            (_, Mapping::Coop { .. }) => {
                if segs > 1 {
                    return Err(SpecError::CoopWithSplitK { segs });
                }
            }
            (_, Mapping::Row) => {}
        }
        Ok(())
    }

    /// **Chart bytes a token MOVES under this spec** — the roofline denominator, defined once.
    ///
    /// Was two `if sb { sb_chart_bytes(..) } else { chart_bytes_tl(..) }` expressions inside one
    /// function (`make_gemv` computed it at both exit points) plus independent copies in the other
    /// four dispatchers; `bedrock::serve::gpu`'s `chart_bytes_tl`/`sb_chart_bytes` now forward here.
    ///
    /// ⚠ **The three bodies below are MOVED, not re-derived.** Getting this wrong does not just
    /// misreport a number: it reports a kernel as closer to the card's limit than it is — the first
    /// two-level face left this at the single-level formula and `DEC_PROF` on the 27B duly claimed
    /// 11 735 MB of weight traffic against 11.60 GB actually resident. `spec_bytes_match_the_shipped_
    /// formulas` pins all three against the shapes in the served stack.
    pub fn chart_bytes(&self, m: u32, k: u32) -> u64 {
        let (m64, k64) = (u64::from(m), u64::from(k));
        match self.scale {
            // 2 bits per weight of codes plus one f16 per 256 — 2.0625 bits/weight against the
            // per-32 chart's 3.0 at the same code width, which is the whole reason the face exists.
            ScaleFace::Superblock => (k64 / 16) * m64 * 4 + u64::from(k.div_ceil(256)) * m64 * 4,
            _ if !self.carrier.is_chart() => m64 * k64 * 4 / u64::from(self.carrier.pack()),
            _ => {
                // `code_words_per_row` and not `k / pack`: Carrier::Q3 is TWO planes and `pack()`
                // only describes the first, so `pack()` here would undercount it by a third.
                let codes = u64::from(self.carrier.code_words_per_row(k)) * m64 * 4;
                // The two-level face stores a `(d8, l8)` BYTE pair per (block, row) packed two rows
                // to a word, plus one f16 super-pair per row — half the per-32 scale stream.
                let scales = if self.scale == ScaleFace::TwoLevel {
                    (k64 / 32) * u64::from(m.div_ceil(2)) * 4 + m64 * 4
                } else {
                    (k64 / 32) * m64 * 4
                };
                codes + scales
            }
        }
    }

    /// **The dispatch grid** for stage 1, given the row-mapping workgroup size.
    pub fn groups(&self, m: u32, local: u32) -> (u32, u32) {
        match self.mapping {
            Mapping::Coop { rows_per_wg, .. } => (m.div_ceil(rows_per_wg), 1),
            _ => (m.div_ceil(local), self.reduce.segs()),
        }
    }

    /// **THE SELECTION — one `match`, replacing the four-way `if` chain in `make_gemv` and the
    /// independent choices made by its four siblings.**
    ///
    /// `local` is the row-mapping workgroup size; the fused kernels force 256 internally
    /// (`gemv_actq` for its shared quantize tree, `gemv_norm` for rmsnorm reduction byte-identity)
    /// and ignore it.
    ///
    /// # Errors
    /// [`SpecError`] if the spec violates a cross-axis rule — see [`Self::legalize`].
    pub fn emit(&self, m: u32, k: u32, local: u32) -> Result<Vec<u8>, SpecError> {
        self.legalize(k)?;
        let segs = self.reduce.segs();
        let two_level = self.scale == ScaleFace::TwoLevel;
        // the expert axis is not a point in the (fusion, mapping) space -- `legalize` has already
        // pinned the other four axes, so this is a total dispatch, not a special case that escapes
        if let Some(e) = self.experts {
            return Ok(crate::simt::gemv_dp4a_moe_bytes(
                m,
                k,
                local,
                self.carrier,
                e.wpe,
                e.dmpe,
                e.weighted,
                e.x_slot,
            ));
        }
        Ok(match (self.fusion, self.mapping) {
            // ── fused: the activation producer or the norm rides along ─────────────────────────
            (Fusion::ActQ(act), _) => gemv_actq_bytes_sc(m, k, segs, self.carrier, act, two_level),
            (Fusion::Norm { eps, residual }, _) => {
                gemv_norm_bytes_sc(m, k, 256, segs, self.carrier, eps, residual, two_level)
            }
            // ── bare, by mapping ───────────────────────────────────────────────────────────────
            (
                Fusion::None,
                Mapping::Coop {
                    rows_per_wg,
                    local: lw,
                    subgroup,
                },
            ) => gemv_coop_bytes_sc(m, k, lw, rows_per_wg, self.carrier, subgroup, two_level),
            (Fusion::None, Mapping::SbTile { bt, rt, uu }) => {
                crate::simt::gemv_dp4a_sb_bytes(m, k, local, bt, rt, uu, segs, self.carrier)
            }
            (Fusion::None, Mapping::Row) if self.carrier.is_chart() => {
                crate::simt::gemv_dp4a_dyn_bytes_sc(m, k, local, segs, self.carrier, two_level)
            }
            // direct carriers (F32/F16) — no chart, no scale stream.
            //
            // ⚠ `make_gemv`'s direct branch hardcoded `Carrier::F16` here and dropped the caller's
            // carrier on the floor. Only F16 reaches it in the served stack (the stage-2 reduce
            // names `Carrier::F32` through its own `gemv_bytes` call, not through a spec), so this
            // is behaviour-preserving today — but a silently ignored argument is the shape of bug
            // this module exists to remove, so the spec honours what it was given.
            (Fusion::None, Mapping::Row) => {
                crate::simt::gemv_bytes(m, k, local, segs, self.carrier)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `assert!(!(two_level && sb))` that opened `make_gemv` is gone because the state it
    /// guarded cannot be written down. This test records that, so a future `bool`-ification is a
    /// red test rather than a silent regression.
    #[test]
    fn scale_faces_are_mutually_exclusive_by_construction() {
        let s = GemvSpec::row(Carrier::Q2, 1).with_two_level(true);
        assert_eq!(s.scale, ScaleFace::TwoLevel);
        assert_eq!(
            GemvSpec::superblock(1, 1, 1, 1).scale,
            ScaleFace::Superblock
        );
    }

    #[test]
    fn superblock_and_sbtile_imply_each_other() {
        let mut s = GemvSpec::superblock(1, 1, 1, 1);
        s.mapping = Mapping::Row;
        assert_eq!(s.legalize(4096), Err(SpecError::SuperblockNeedsSbTile));
        let mut s = GemvSpec::row(Carrier::Q2, 1);
        s.mapping = Mapping::SbTile {
            bt: 1,
            rt: 1,
            uu: 1,
        };
        assert_eq!(s.legalize(4096), Err(SpecError::SuperblockNeedsSbTile));
    }

    /// `bt` must be 1 under split-K — the superblock kernel's own doc, previously restated as a
    /// comment at the one call site that got it right.
    #[test]
    fn sbtile_batch_conflicts_with_split_k() {
        let s = GemvSpec::superblock(4, 1, 1, 8);
        assert_eq!(
            s.legalize(4096),
            Err(SpecError::SbTileBatchUnderSplitK { bt: 4, segs: 8 })
        );
        assert!(GemvSpec::superblock(4, 1, 1, 1).legalize(4096).is_ok());
        assert!(GemvSpec::superblock(1, 1, 1, 8).legalize(4096).is_ok());
    }

    #[test]
    fn coop_refuses_split_k() {
        let mut s = GemvSpec::row(Carrier::Q4, 8);
        s.mapping = Mapping::Coop {
            rows_per_wg: 4,
            local: 64,
            subgroup: 32,
        };
        assert_eq!(s.legalize(4096), Err(SpecError::CoopWithSplitK { segs: 8 }));
        assert!(s.reducing(Reduce::Stage2(1)).legalize(4096).is_ok());
    }

    /// **Routing must not change a byte.** `make_moe_gemv` used to call `gemv_dp4a_moe_bytes`
    /// directly; it now goes through [`GemvSpec::emit`]. An end-to-end SEQHASH cannot gate that
    /// today — the only MoE checkpoint here is 57 GB of safetensors whose packed recipe does not
    /// fit the card beside a desktop — so the equivalence is asserted at the emitter instead, which
    /// is where the risk actually lives.
    #[test]
    fn routing_moe_through_the_spec_emits_the_same_module() {
        for (m, k, local, weighted, x_slot, carrier) in [
            (2048u32, 1024u32, 128u32, true, false, Carrier::Q2),
            (2048, 1024, 128, false, true, Carrier::Q4),
            (768, 2048, 64, true, true, Carrier::Q8),
        ] {
            let (wpe, dmpe) = ((k / carrier.pack()) * m, (k / 32) * m);
            let direct =
                crate::simt::gemv_dp4a_moe_bytes(m, k, local, carrier, wpe, dmpe, weighted, x_slot);
            let viaspec = GemvSpec::moe(carrier, wpe, dmpe, weighted, x_slot)
                .emit(m, k, local)
                .expect("GemvSpec::moe constructs a legal spec");
            assert_eq!(
                direct, viaspec,
                "m{m} k{k} {carrier:?} w{weighted} x{x_slot}"
            );
        }
    }

    /// The MoE emitter supports exactly one point on the other four axes. Each stray axis must be
    /// NAMED — the failure this replaces is `make_moe_gemv` calling the emitter directly, where a
    /// `Reduce` or `Mapping` a policy had set was dropped with no diagnostic at all.
    #[test]
    fn a_moe_spec_names_the_axis_it_cannot_honour() {
        let base = GemvSpec::moe(Carrier::Q2, 4096, 512, true, false);
        assert!(
            base.legalize(1024).is_ok(),
            "the constructed MoE spec must be legal"
        );
        let cases = [
            (
                "scale",
                GemvSpec {
                    scale: ScaleFace::TwoLevel,
                    ..base
                },
            ),
            (
                "mapping",
                GemvSpec {
                    mapping: Mapping::Coop {
                        rows_per_wg: 4,
                        local: 128,
                        subgroup: 32,
                    },
                    ..base
                },
            ),
            (
                "reduce",
                GemvSpec {
                    reduce: Reduce::Stage2(8),
                    ..base
                },
            ),
            (
                "carrier",
                GemvSpec {
                    carrier: Carrier::F16,
                    ..base
                },
            ),
        ];
        for (want, spec) in cases {
            match spec.legalize(1024) {
                Err(SpecError::MoeSupportsRowDm32Only { axis }) => assert_eq!(axis, want),
                other => panic!("{want}: expected a named MoE axis refusal, got {other:?}"),
            }
        }
    }

    #[test]
    fn superblock_is_q2_only() {
        let mut s = GemvSpec::superblock(1, 1, 1, 1);
        s.carrier = Carrier::Q4;
        assert_eq!(
            s.legalize(4096),
            Err(SpecError::SuperblockNeedsQ2 {
                carrier: Carrier::Q4
            })
        );
    }

    /// **[`Carrier::Q5`] is a two-plane chart: its fifth bit is read by the fused folds only.** Every
    /// other emitter would serve a plausible Q4 (finite, wrong), so each dishonourable axis is
    /// refused by name — and `ActQ`/`Norm` on `Dm32`, which thread `plane_b_off` into the fold, is
    /// the point that legalizes.
    #[test]
    fn q5_needs_a_fused_fold_that_reads_plane_b() {
        let ok = GemvSpec::decode(Carrier::Q5, true).fused(Fusion::ActQ(ActQ::Copy));
        assert!(
            ok.legalize(896).is_ok(),
            "ActQ + Dm32 + Q5 is the served fold"
        );
        // the fused-norm emitter threads plane_b_off too (`gemv_norm_module_sc`)
        let norm = GemvSpec::decode(Carrier::Q5, true).fused(Fusion::Norm {
            eps: 1e-6,
            residual: None,
        });
        assert!(norm.legalize(896).is_ok());
        // the bare row — and the coop/sbtile emitters it shares `Fusion::None` with — drop plane B
        let bare = GemvSpec::decode(Carrier::Q5, true);
        assert_eq!(
            bare.legalize(896),
            Err(SpecError::Q5NeedsFusedFold { axis: "fusion" })
        );
        // the MoE emitter has no plane-B word at all
        let moe = GemvSpec::moe(Carrier::Q5, 896 * 896 / 8, 896 * 896 / 32, false, false);
        assert_eq!(
            moe.legalize(896),
            Err(SpecError::Q5NeedsFusedFold { axis: "experts" })
        );
        // any scale face but Dm32 stores no plane-B word
        let mut tl = GemvSpec::decode(Carrier::Q5, true).fused(Fusion::ActQ(ActQ::Copy));
        tl.scale = ScaleFace::TwoLevel;
        assert_eq!(
            tl.legalize(896),
            Err(SpecError::Q5NeedsFusedFold { axis: "scale" })
        );
    }

    #[test]
    fn direct_carriers_have_no_scale_face() {
        // `with_two_level` is a no-op on a direct carrier, matching the shipped `chart_bytes_tl`
        let s = GemvSpec::row(Carrier::F16, 1).with_two_level(true);
        assert_eq!(s.scale, ScaleFace::Dm32);
        assert!(s.legalize(4096).is_ok());
        // ...but naming the face directly is the error worth having
        let mut s = GemvSpec::row(Carrier::F16, 1);
        s.scale = ScaleFace::TwoLevel;
        assert!(matches!(
            s.legalize(4096),
            Err(SpecError::DirectCarrierHasNoScaleFace { .. })
        ));
        assert!(
            GemvSpec::row(Carrier::F16, 1).legalize(4097).is_ok(),
            "direct carriers ignore the block alignment"
        );
    }

    /// **The `shape-derived-gemv.md` defect, as a test.** `emit`'s fusion arm ignores the mapping
    /// (the fused emitters take no mapping argument), so a fused spec carrying `Coop` would have
    /// silently produced the ROW kernel — the same silence that made the in-chain A/B measure
    /// noise. `legalize` refuses it instead.
    #[test]
    fn a_fused_spec_cannot_silently_swallow_its_mapping() {
        for fusion in [
            Fusion::ActQ(ActQ::Copy),
            Fusion::Norm {
                eps: 1e-6,
                residual: None,
            },
        ] {
            let mut s = GemvSpec::decode(Carrier::Q4, true).fused(fusion);
            s.mapping = Mapping::Coop {
                rows_per_wg: 4,
                local: 64,
                subgroup: 32,
            };
            assert!(
                matches!(s.legalize(4096), Err(SpecError::NoFusedKernelForMapping { .. })),
                "a fused {fusion:?} on a cooperative mapping must be refused, not silently row-mapped"
            );
            assert!(
                s.emit(5120, 4096, 256).is_err(),
                "emit must not build that pipeline"
            );
            // ...and the row mapping, which DOES have a fused kernel, still works
            let ok = GemvSpec::decode(Carrier::Q4, true).fused(fusion);
            assert!(ok.legalize(4096).is_ok());
        }
    }

    /// Every legal point in the space emits SOMETHING — `emit` is total where `legalize` passes,
    /// so a new axis value cannot fall through the match.
    #[test]
    fn emit_is_total_wherever_legalize_passes() {
        let carriers = [
            Carrier::F16,
            Carrier::Q8,
            Carrier::Q4,
            Carrier::Q3,
            Carrier::Q2,
        ];
        let mut checked = 0;
        for c in carriers {
            for tl in [false, true] {
                for fold in [false, true] {
                    for segs in [1u32, 4] {
                        let mut specs = vec![
                            GemvSpec::decode(c, fold).with_two_level(tl).with_segs(segs),
                            GemvSpec::decode(c, fold)
                                .with_two_level(tl)
                                .with_segs(segs)
                                .fused(Fusion::ActQ(ActQ::Copy)),
                        ];
                        let mut coop = GemvSpec::decode(c, fold).with_two_level(tl).with_segs(1);
                        coop.mapping = Mapping::Coop {
                            rows_per_wg: 4,
                            local: 64,
                            subgroup: 32,
                        };
                        specs.push(coop);
                        specs.push(GemvSpec::decode_superblock(fold).with_segs(segs));
                        for s in specs {
                            if s.legalize(4096).is_ok() {
                                assert!(
                                    !s.emit(5120, 4096, 128).expect("legal ⇒ emits").is_empty(),
                                    "{s:?} legalized but emitted nothing"
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(
            checked > 60,
            "the sweep should cover the space, covered {checked}"
        );
    }

    /// **The pin: `chart_bytes` reproduces the three shipped formulas exactly.**
    ///
    /// These literals are computed from `bedrock::serve::gpu`'s `chart_bytes_tl` and
    /// `sb_chart_bytes` as they stood before the move, over the four shapes the corpus names
    /// (`results/gemv-shared-is-over-the-limit.md`, `results/shape-derived-gemv.md`). modelfold
    /// cannot call bedrock, so the originals are pinned as values — which is the stronger gate
    /// anyway: it stays red if BOTH copies drift together.
    #[test]
    fn spec_bytes_match_the_shipped_formulas() {
        // (name, m, k, [q4_dm32, q4_tl, q2_dm32, q2_tl, q3_dm32, f16, sb])
        let cases: &[(&str, u32, u32, [u64; 7])] = &[
            (
                "ffn_gateup",
                34816,
                5120,
                [
                    111_411_200,
                    100_409_344,
                    66_846_720,
                    55_844_864,
                    89_128_960,
                    356_515_840,
                    47_349_760,
                ],
            ),
            (
                "dn_out",
                5120,
                6144,
                [
                    19_660_800, 17_715_200, 11_796_480, 9_850_880, 15_728_640, 62_914_560,
                    8_355_840,
                ],
            ),
            (
                "ffn_down",
                5120,
                17408,
                [
                    55_705_600,
                    50_155_520,
                    33_423_360,
                    27_873_280,
                    44_564_480,
                    178_257_920,
                    23_674_880,
                ],
            ),
            (
                "lm_head",
                151936,
                896,
                [
                    85_084_160,
                    77_183_488,
                    51_050_496,
                    43_149_824,
                    68_067_328,
                    272_269_312,
                    36_464_640,
                ],
            ),
        ];
        for &(name, m, k, want) in cases {
            let b = |s: GemvSpec| s.chart_bytes(m, k);
            assert_eq!(b(GemvSpec::row(Carrier::Q4, 1)), want[0], "{name} q4 dm32");
            assert_eq!(
                b(GemvSpec::row(Carrier::Q4, 1).with_two_level(true)),
                want[1],
                "{name} q4 two-level"
            );
            assert_eq!(b(GemvSpec::row(Carrier::Q2, 1)), want[2], "{name} q2 dm32");
            assert_eq!(
                b(GemvSpec::row(Carrier::Q2, 1).with_two_level(true)),
                want[3],
                "{name} q2 two-level"
            );
            assert_eq!(b(GemvSpec::row(Carrier::Q3, 1)), want[4], "{name} q3 dm32");
            assert_eq!(b(GemvSpec::row(Carrier::F16, 1)), want[5], "{name} f16");
            assert_eq!(b(GemvSpec::superblock(1, 1, 1, 1)), want[6], "{name} sb");
        }
    }

    /// The 5.25 B/k constant lives at exactly one site, and the shapes
    /// `results/gemv-shared-is-over-the-limit.md` measured reproduce.
    #[test]
    fn row_shared_reproduces_the_measured_table() {
        let s = GemvSpec::row(Carrier::Q2, 1);
        assert_eq!(s.row_shared_bytes(5120), 26_880); // ffn_gateup
        assert_eq!(s.row_shared_bytes(6144), 32_256); // dn_out
        assert_eq!(s.row_shared_bytes(17408), 91_392); // ffn_down — 1.86x the 49152 limit
        assert!(s.over_shared_limit(17408, 49_152));
        assert!(!s.over_shared_limit(6144, 49_152));
        // split-K does NOT reduce it: the window is the same at every segment count
        for segs in [1u32, 4, 16, 32] {
            assert_eq!(
                s.reducing(Reduce::Stage2(segs)).row_shared_bytes(17408),
                91_392
            );
        }
    }

    /// Coop is structurally immune to the shared limit — the reason it was reached for, kept as a
    /// property now that the mapping rule itself is refuted.
    #[test]
    fn coop_declares_no_row_staging() {
        let mut s = GemvSpec::row(Carrier::Q2, 1);
        s.mapping = Mapping::Coop {
            rows_per_wg: 4,
            local: 64,
            subgroup: 32,
        };
        assert_eq!(s.row_shared_bytes(17408), 0);
        assert!(!s.over_shared_limit(17408, 49_152));
    }

    #[test]
    fn reduce_faces_agree_on_segs_and_differ_on_stage2() {
        assert_eq!(Reduce::Stage2(8).segs(), 8);
        assert_eq!(Reduce::Fold(8).segs(), 8);
        assert!(Reduce::Stage2(8).needs_stage2());
        assert!(!Reduce::Stage2(1).needs_stage2());
        assert!(!Reduce::Fold(8).needs_stage2(), "fold's consumer reduces");
    }
}
