import Lake
open Lake DSL

package apeironWordBridge where
  leanOptions := #[
    { name := `autoImplicit, value := false },
    { name := `warningAsError, value := true }
  ]

-- The extraction/denotation package next door.
require apeironWordProof from ".."

-- The canonical mathematical corpus refined by this bridge, in-tree at `proof/canonical/`
-- (byte-identical to canonical revision `880aa57e47e782ed7a9d9ee1974cfc87360b446b`, `proof/`).
require «apeiron» from "../canonical"

-- Pin the same Mathlib revision as the corpus, last so this package's choice wins any transitive
-- resolution. The corpus is no longer on a 4.30.0-rc2-era Mathlib.
require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "v4.31.0"

@[default_target]
lean_lib ApeironWordBridge where
  -- The `TableRefinement*`, `TableRefinement/*`, and `Joint.TableInstance*` modules prove the
  -- revision-bound fixed-shard layout. They remain on disk as historical evidence but are not
  -- roots of the executing growable bridge because the current extraction has no fixed-shard API.
  roots := #[`WordRefinement, `Native.HyperbolicBase, `Native.LocatorSpec,
    -- `Native.HashtrinityBasis` was committed in 637e093 on no build path (roots carries no
    -- `.submodules` glob). Registered with its 15 in-module `#audit` rows.
    `Native.HashtrinityBasis,
    `Security.InterpretationWitnesses,
    -- Added with the HashDOS verification lane. Two transcription hypotheses (the multiplier
    -- constant identity and the slot arithmetic) are now discharged against the generated
    -- `table.Region.home`; the input inversion, the power-of-two bound and the `descend` binding
    -- stay blocked (no `Table::descend` body exists in the extraction).
    `Security.HashDos,
    `Growable.Geometry, `Growable.Identity, `Growable.Scan,
    `Growable.ReadLoop, `Growable.LocateLoop, `Growable.ReadState, `Kepler.Certificate,
    `Growable.Search, `Growable.Read, `Growable.ReadComposition, `Growable.RelocationWF, `Growable.Classifier,
    `Growable.Relocation, `Growable.Refinement, `Growable.Resources, `Growable.Representation,
    `Growable.Rebuild,
    -- Added with the executed-body verification wave. `ApeironWordBridge` sets `roots` without
    -- `.submodules` globs, so an import of a root is built only when it is itself a root.
    -- `Growable.InsertRefine` was imported by `Growable.GaplessRead`/`Growable.BuildRefine`,
    -- neither reachable, so it was ungated.
    -- `Growable.BuildRefine` was on no build path (roots carries no `.submodules` glob, and its
    -- only importer, `Growable/GaplessRead.lean`, is unreachable). Registered + gated.
    -- `GaplessRead`/`HomeBound` were also on no build path: registered, so their headlines are
    -- checked by a scoped `lake build` rather than by an ad-hoc `lake lean`.
    `Growable.GaplessRead, `Growable.HomeBound,
    `Growable.InsertRefine, `Growable.Schedule, `Growable.BuildRefine, `Growable.CostModel,
    -- `Growable.Batched` was committed on no build path: `roots` carries no `.submodules` glob and
    -- nothing imported it, so its 6 `#print axioms` rows were never evaluated. Registered.
    `Growable.Batched,
    -- `Growable.FaceBound` was committed on no build path: `roots` carries no `.submodules` glob and
    -- nothing imported it, so its 6 `#audit` rows were never evaluated. Registered.
    `Growable.FaceBound,
    `Map.Rep, `Map.Read, `Map.Selected, `Map.PrefixMasks,
    -- The executed map face.
    `Map.ExecutedBase, `Map.ExecutedProbe, `Map.ExecutedOverflow, `Map.ExecutedRep,
    `Map.ExecutedRead, `Map.ExecutedReadTotal, `Map.ExecutedReadUncond, `Map.ExecutedSelected,
    `Map.ExecutedReadCoordTotal, `Map.ExecutedSelectedCoord, `Map.Executed,
    `Map.ExecutedOverflowAvlBase, `Map.ExecutedOverflowAvl, `Map.ExecutedOverflowChain, `Map.ExecutedOverflowDescend, `Map.ExecutedOverflowInsert,
    `Map.ExecutedInsertSide, `Map.ExecutedInsertSlot, `Map.ExecutedInsert, `Map.ExecutedResizeOrder, `Map.ExecutedResize,
    -- The executed build.
    `Map.ExecutedBuild,
    `Map.ExecutedBuildWitness, `Map.ExecutedInsertTotal,
    -- Totality of resize.
    `Map.ExecutedResizeTotal,
    -- Totality of
    -- `grow_for_one`, with the resize witness split out of `Map.ExecutedResizeTotal`.
    `Map.ExecutedGrowTotal,
    -- Stage 3, the executed
    -- insert is total.
    `Map.ExecutedInsertOk,
    -- Stage 3, the executed
    -- insert is total on a reserved key (the side-store path).
    `Map.ExecutedInsertSentOk,
    -- The `Stored<u64>`
    -- size carrier and its key-store totality lemmas.
    `Map.ExecutedInsertStored,
    -- The live-entry
    -- counter `Hlive` and the per-`place_capped` joint bound for the linear fold.
    `Map.ExecutedInsertLinear,
        `Count.Rep, `Count.Read, `Count.Helpers, `Count.FreqRep, `Count.OverflowOps, `Count.FreqAdd, `Count.FreqSideOv, `Count.FreqGrow, `Count.FreqInsert, `Count.Aggregate,
    -- `Flux.Discrete` is imported by the registered root `AxiomAudit` (`AxiomAudit.lean:3`) and
    -- `#audit`-ed by it (`:6`), but no `lean_lib` globbed it, so `lake build AxiomAudit` failed
    -- with `unknown module prefix 'Flux'` and that audit row was never evaluated. Registered.
    `Flux.Discrete,
    -- Exact-integer Gram fold refinement (`Learn/GramExact.lean`): the extracted `gram_fold_i8`
    -- against the canonical `Sanguine.TrainingCollapse.gram_stack`.
    `Learn.GramExact,
    -- E8 codebook refinement (`Learn/E8Code.lean`): the extracted `learn_port.e8` codeword
    -- and nearest-codeword bodies against integer specifications over `ℤ`.
    `Learn.E8Code,
    -- Exact-integer `src/learn/grad/dense.rs` refinement (`Learn/Grad/DenseLinearSpec.lean` is the
    -- independent `ℤ`/`List ℤ` model; `Learn/Grad/DenseLinear.lean` refines the extracted
    -- `linear_fwd` / `linear_bwd_input` / `linear_bwd_weight` against it; `DenseLinearWitness`
    -- shows the admissible domain and every refusal arm are inhabited; `AxiomAudit` audits them).
    `Learn.Grad.DenseLinearSpec, `Learn.Grad.DenseLinear, `Learn.Grad.DenseLinearWitness,
    -- Extracted RoPE forward/backward refinement (`Learn/Grad/Rope.lean`): `learn_port.attn.rope_fwd`
    -- against `rotHeadsState`/`RopeTable.rotate_one`, and `learn_port.attn.rope_bwd`/`rope_elem_bwd`
    -- against the transpose kernel `ropeBwdPair` accumulated as `ropeBwdPrefix`.
    `Learn.Grad.Rope,
    -- Exact-integer head refinement (`Learn/Grad/Head.lean`): the extracted `learn_port.head`
    -- rounding, narrowing, sign and checked-arithmetic bodies against their `ℤ` specifications,
    -- and the `log_softmax_row` projection over the conditional `softmax_row`.
    `Learn.Grad.Head,
    -- The extracted `_loop` bodies of `learn_port.head` (`Learn/Grad/HeadLoop.lean`) against
    -- their `ℤ` prefix-sum/accumulator models, over `loop.spec_decr_nat`.
    `Learn.Grad.HeadLoop,
    -- W1b backward-pass refinements (`src/learn/grad/dense.rs`): the extracted `learn_port.dense`
    -- `RMSNorm`/`SwiGLU` forward and backward maps against the `ℤ` specification.
    `Learn.Grad.DenseNormSpec, `Learn.Grad.DenseNorm, `Learn.Grad.DenseNormBwd,
    -- The remaining extracted `_loop` bodies of `learn_port.head` (`Learn/Grad/HeadRows.lean`):
    -- the `W_head` backward, the TV gradient and the row/logits/softmax loops.
    `Learn.Grad.HeadRows,
    `AxiomAudit, `Joint.WordBridge,
    `Iota.Audit.Empty,
    `Iota.Syntax, `Iota.Decode,
    `Iota.DecodeTest, `Iota.Word, `Iota.Machine, `Iota.MachineAudit,
    `Iota.Generated.Table, `Iota.Generated.TableDerivation,
    `Iota.Generated.TableSupport,
    `Iota.Source.Bytes, `Iota.Source.Audit.Bytes,
    `Iota.Source.Lexing, `Iota.Source.Audit.Lexing, `Iota.Source.Grammar,
    `Iota.Source.GrammarAudit,     `Iota.Arena.Bounded, `Iota.Arena.StateResult, `Iota.Arena.HandleAudit,
    `Iota.Address.Faces, `Iota.Address.FacesAudit,
    `Iota.Memo.Key, `Iota.Memo.KeyAudit,
    `Iota.Memo.Value, `Iota.Memo.ValueAudit,
    `Iota.Memo.Store, `Iota.Memo.StoreAudit, `Iota.Memo.Witness, `Iota.Memo.WitnessAudit,
    `Iota.Memo.Fold, `Iota.Memo.FoldAudit,
    `Iota.Source.Arena, `Iota.Source.ArenaAudit,
    `Iota.Source.Closure, `Iota.Source.ClosureAudit, `Iota.Source.ClosedBinding,
    `Iota.Source.ClosedBindingAudit,
    `Iota.Source.DecodeProof, `Iota.Source.DecodeProofAudit, `Iota.SourceAudit,
    `Iota.Source.Tokenizer.Utf8, `Iota.Source.Tokenizer.Utf8Audit,
    `Iota.Source.Tokenizer.Helpers, `Iota.Source.Tokenizer.Scan,
    `Iota.Source.Tokenizer.ScanAudit, `Iota.Source.Tokenizer.Tokenize,
    `Iota.Source.Tokenizer.TokenizeAudit,
    `Iota.Source.Parsing.Selection, `Iota.Source.Parsing.SelectionAudit,
    `Iota.Source.Parsing.Expression, `Iota.Source.Parsing.ExpressionAudit,
    `Iota.Source.Parsing.Statement, `Iota.Source.Parsing.StatementAudit,
    `Iota.Source.Parsing.Validation, `Iota.Source.Parsing.ValidationAudit,
    `Iota.Source.Parsing.Tokens, `Iota.Source.Parsing.TokensAudit,
    `Iota.Refinement.WordRing, `Iota.Refinement.WordRingAudit,
    `Iota.Refinement.WordCoreAudit, `Iota.Refinement.TableRepAudit,
    `Iota.Refinement.WordDenotation, `Iota.Refinement.WordDenotationAudit,
    `Iota.Refinement.Context, `Iota.Refinement.ContextAudit, `Iota.Table.TableRep,
    `Iota.Table.TableInit, `Iota.Table.TableWitness, `Iota.Table.TableProbe, `Iota.Table.TableRead,
    `Iota.Table.ReadWitness, `Iota.Table.TableWrite, `Iota.Table.TableAdmission,
    `Iota.Table.WriteWitness, `Iota.Table.TableTermination, `Iota.Table.TableResources,
    `Iota.Table.ResourceWitness,
    `Iota.Joint.SongTable, `Iota.Joint.SongTableControl,
    `Iota.Joint.AccumulatorProfile,
    `Iota.Generated.SourceWitness.Parsing.Table.Chunk01,
    `Iota.Generated.SourceWitness.Parsing.Table.Chunk02,
    `Iota.Generated.SourceWitness.Parsing.Table.Chunk03,
    `Iota.Generated.SourceWitness.Parsing.Table.Chunk04,
    `Iota.Generated.SourceWitness.Parsing.Table.Chunk05,
    `Iota.Generated.SourceWitness.Parsing.Table.Chunk06,
    `Iota.Generated.SourceWitness.Parsing.Table.Chunk07,
    `Iota.Generated.SourceWitness.Parsing.Table.Chunk08,
    `Iota.Generated.SourceWitness.Parsing.Table.Chunk09,
    `Iota.Generated.SourceWitness.Parsing.Cell.Chunk01,
    `Iota.Generated.SourceWitness.Parsing.Cell.Chunk02,
    `Iota.Generated.SourceWitness.Parsing.Cell.Chunk03,
    `Iota.Generated.SourceWitness.Parsing.Audit,
    `Iota.Generated.SourceWitness.Tokenizer.Cell.Chunk01,
    `Iota.Generated.SourceWitness.Tokenizer.Cell.Chunk02,
    `Iota.Generated.SourceWitness.Tokenizer.Table.Chunk01,
    `Iota.Generated.SourceWitness.Tokenizer.Table.Chunk02,
    `Iota.Generated.SourceWitness.Tokenizer.Table.Chunk03,
    `Iota.Generated.SourceWitness.Tokenizer.Std.Chunk01,
    `Iota.Generated.SourceWitness.Tokenizer.Std.Chunk02,
    `Iota.Generated.SourceWitness.Tokenizer.Std.Chunk03,
    `Iota.Generated.SourceWitness.Tokenizer.Std.Chunk04,
    `Iota.Generated.SourceWitness.Tokenizer.Std.Chunk05,
    `Iota.Generated.SourceWitness.Tokenizer.Std.Chunk06,
    `Iota.Generated.SourceWitness.Tokenizer.Std.Chunk07,
    `Iota.Generated.SourceWitness.Tokenizer.Std.Chunk08,
    `Iota.Generated.SourceWitness.Tokenizer.Std.Chunk09,
    `Iota.Generated.SourceWitness.Tokenizer.Std.Chunk10,
    `Iota.Generated.SourceWitness.Tokenizer.Std.Chunk11,
    `Iota.Generated.SourceWitness.Tokenizer.Audit,
    `Iota.Generated.SourceWitness.Decoder.ChunksA,
    `Iota.Generated.SourceWitness.Decoder.ChunksB,
    `Iota.Generated.SourceWitness.Decoder.Assembly,
    `Iota.Generated.SourceWitness.Decoder.Audit]

-- Stage-2 neural rebase: the vendored historical subtrees, byte-identical copies from
-- `sanguine/proof` (manifest `Substrate/vendor.sha256`, regeneration `tools/vendor-neural.sh`).
-- A separate default target owns module discovery for the vendored files; `ApeironWordBridge`
-- does not import them until the wave-1 audit registrations do.
@[default_target]
lean_lib SubstrateVendored where
  globs := #[.submodules `Substrate, .submodules `Closures]

-- AI-track re-anchor: native trichotomy-first surface and model
-- (docs/epics/ai-track-reanchor.md, docs/epics/ai-track-reanchor.md). One library, module discovery
-- under Nous/; Core/ is ring-generic and Model/ is AI-specific.
lean_lib Nous where
  globs := #[.submodules `Nous]

-- Inference-parity numeric kernels: fixed-point Rust bodies and their real-valued refinements
-- (docs/epics/inference-parity.md, docs/epics/inference-parity.md). Module discovery under Numeric/.
lean_lib Numeric where
  globs := #[.submodules `Numeric]

-- GPU/SPIR-V subset verification: mechanized fragment semantics, decoder soundness and the
-- checker/DP4A refinements (docs/epics/gpu-spv.md, docs/epics/gpu-spv.md). Module discovery under
-- Spirv/; roots are registered here by the lead before each task dispatches.
lean_lib Spirv where
  globs := #[.submodules `Spirv]

-- Performance theory (docs/epics/resource-guided-evolution.md): the resource carrier `Perf.Res`,
-- the roofline lower bound, the move theorem and the fusion results. Module discovery under Perf/.
lean_lib Perf where
  globs := #[.submodules `Perf]

-- Bounded authorization decision core (docs/epics/policy-engine.md): the canonical definition staged
-- here until the corpus migration, its laws, and (W2) the analysis layer over Apeiron.Decider.
lean_lib Policy where
  globs := #[.submodules `Policy]

-- FIPS-202 Keccak-f[1600] constants and the refinement chain over them. Module discovery under
-- Crypto/.
lean_lib Crypto where
  globs := #[.submodules `Crypto]

-- SHA-256/SHA-256d rooting onto the trichotomy (docs/epics/icarus-sha256-rooting.md).
-- Module discovery under Icarus/.
lean_lib Icarus where
  globs := #[.submodules `Icarus]
