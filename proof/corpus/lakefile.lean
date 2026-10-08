import Lake
open Lake DSL

package apeironWordBridge where
  leanOptions := #[
    { name := `autoImplicit, value := false },
    { name := `warningAsError, value := true }
  ]

-- The extraction/denotation package next door.
require apeironWordProof from ".."

-- The canonical mathematical corpus at the exact revision refined by this bridge.
require «apeiron» from git "git@gitlab.tas.twn.network:lucas/sanguine.git" @
  "880aa57e47e782ed7a9d9ee1974cfc87360b446b" / "proof"

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
    -- Row: `apeiron-native-gate-DbKpTzo95pI4CWY7PekAxE` (APEIRON-NATIVE-GATE).
    `Native.HashtrinityBasis,
    `Security.InterpretationWitnesses,
    -- Added with the HashDOS verification lane. Two transcription hypotheses (the multiplier
    -- constant identity and the slot arithmetic) are now discharged against the generated
    -- `table.Region.home`; the input inversion, the power-of-two bound and the `descend` binding
    -- stay blocked (no `Table::descend` body exists in the extraction).
    -- Row: `p1-hashdos-formalise-the-CuQBvGaIws0dH6rzQYsl7j`.
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
    -- Row: `apeiron-verify-wave-land-nFsIRO8H9pKb1njJApKly` (APEIRON-VERIFY-WAVE).
    -- `Growable.BuildRefine` was on no build path (roots carries no `.submodules` glob, and its
    -- only importer, `Growable/GaplessRead.lean`, is unreachable). Registered + gated.
    -- Row: `apeiron-roots-schedule-GLorJM9lzcN49Cr1tCCT0s` (APEIRON-BUILDREFINE-RECONCILE).
    -- `GaplessRead`/`HomeBound` were also on no build path: registered, so their headlines are
    -- checked by a scoped `lake build` rather than by an ad-hoc `lake lean`.
    `Growable.GaplessRead, `Growable.HomeBound,
    -- Row: `apeiron-corpus-register-GPgkBfYXcWVRfHB3t9Mpwl` (APEIRON-CORPUS-REACHABILITY).
    `Growable.InsertRefine, `Growable.Schedule, `Growable.BuildRefine, `Growable.CostModel,
    -- `Growable.Batched` was committed on no build path: `roots` carries no `.submodules` glob and
    -- nothing imported it, so its 6 `#print axioms` rows were never evaluated. Registered.
    `Growable.Batched,
    -- `Growable.FaceBound` was committed on no build path: `roots` carries no `.submodules` glob and
    -- nothing imported it, so its 6 `#audit` rows were never evaluated. Registered.
    `Growable.FaceBound,
    `Map.Rep, `Map.Read, `Map.Selected, `Map.PrefixMasks,
    -- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe` (APEIRON-HT-READ): the executed map face.
    `Map.ExecutedBase, `Map.ExecutedProbe, `Map.ExecutedOverflow, `Map.ExecutedRep,
    `Map.ExecutedRead, `Map.ExecutedReadTotal, `Map.ExecutedReadUncond, `Map.ExecutedSelected,
    `Map.ExecutedReadCoordTotal, `Map.ExecutedSelectedCoord, `Map.Executed,
    `Map.ExecutedOverflowAvlBase, `Map.ExecutedOverflowAvl, `Map.ExecutedOverflowChain, `Map.ExecutedOverflowDescend, `Map.ExecutedOverflowInsert,
    `Map.ExecutedInsertSide, `Map.ExecutedInsertSlot, `Map.ExecutedInsert, `Map.ExecutedResizeOrder, `Map.ExecutedResize,
    -- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (APEIRON-HT-INSERT, HT-BUILD): the executed build.
    `Map.ExecutedBuild,
    `Map.ExecutedBuildWitness, `Map.ExecutedInsertTotal,
        `Count.Rep, `Count.Read, `Count.Helpers, `Count.FreqRep, `Count.OverflowOps, `Count.FreqAdd, `Count.FreqSideOv, `Count.FreqGrow, `Count.FreqInsert, `Count.Aggregate,
    -- `Flux.Discrete` is imported by the registered root `AxiomAudit` (`AxiomAudit.lean:3`) and
    -- `#audit`-ed by it (`:6`), but no `lean_lib` globbed it, so `lake build AxiomAudit` failed
    -- with `unknown module prefix 'Flux'` and that audit row was never evaluated. Registered.
    `Flux.Discrete,
    -- Exact-integer Gram fold refinement (`Learn/GramExact.lean`): the extracted `gram_fold_i8`
    -- against the canonical `Sanguine.TrainingCollapse.gram_stack`.
    `Learn.GramExact,
    `AxiomAudit, `Joint.WordBridge,
    `Slang.Audit.Empty,
    `Slang.Syntax, `Slang.Decode,
    `Slang.DecodeTest, `Slang.Word, `Slang.Machine, `Slang.MachineAudit,
    `Slang.Generated.Table, `Slang.Generated.TableDerivation,
    `Slang.Generated.TableSupport,
    `Slang.Source.Bytes, `Slang.Source.Audit.Bytes,
    `Slang.Source.Lexing, `Slang.Source.Audit.Lexing, `Slang.Source.Grammar,
    `Slang.Source.GrammarAudit,     `Slang.Arena.Bounded, `Slang.Arena.StateResult, `Slang.Arena.HandleAudit,
    `Slang.Address.Faces, `Slang.Address.FacesAudit,
    `Slang.Memo.Key, `Slang.Memo.KeyAudit,
    `Slang.Memo.Value, `Slang.Memo.ValueAudit,
    `Slang.Memo.Store, `Slang.Memo.StoreAudit, `Slang.Memo.Witness, `Slang.Memo.WitnessAudit,
    `Slang.Memo.Fold, `Slang.Memo.FoldAudit,
    `Slang.Source.Arena, `Slang.Source.ArenaAudit,
    `Slang.Source.Closure, `Slang.Source.ClosureAudit, `Slang.Source.ClosedBinding,
    `Slang.Source.ClosedBindingAudit,
    `Slang.Source.DecodeProof, `Slang.Source.DecodeProofAudit, `Slang.SourceAudit,
    `Slang.Source.Tokenizer.Utf8, `Slang.Source.Tokenizer.Utf8Audit,
    `Slang.Source.Tokenizer.Helpers, `Slang.Source.Tokenizer.Scan,
    `Slang.Source.Tokenizer.ScanAudit, `Slang.Source.Tokenizer.Tokenize,
    `Slang.Source.Tokenizer.TokenizeAudit,
    `Slang.Source.Parsing.Selection, `Slang.Source.Parsing.SelectionAudit,
    `Slang.Source.Parsing.Expression, `Slang.Source.Parsing.ExpressionAudit,
    `Slang.Source.Parsing.Statement, `Slang.Source.Parsing.StatementAudit,
    `Slang.Source.Parsing.Validation, `Slang.Source.Parsing.ValidationAudit,
    `Slang.Source.Parsing.Tokens, `Slang.Source.Parsing.TokensAudit,
    `Slang.Refinement.WordRing, `Slang.Refinement.WordRingAudit,
    `Slang.Refinement.WordCoreAudit, `Slang.Refinement.TableRepAudit,
    `Slang.Refinement.WordDenotation, `Slang.Refinement.WordDenotationAudit,
    `Slang.Refinement.Context, `Slang.Refinement.ContextAudit, `Slang.Table.TableRep,
    `Slang.Table.TableInit, `Slang.Table.TableWitness, `Slang.Table.TableProbe, `Slang.Table.TableRead,
    `Slang.Table.ReadWitness, `Slang.Table.TableWrite, `Slang.Table.TableAdmission,
    `Slang.Table.WriteWitness, `Slang.Table.TableTermination, `Slang.Table.TableResources,
    `Slang.Table.ResourceWitness,
    `Slang.Joint.SongTable, `Slang.Joint.SongTableControl,
    `Slang.Joint.AccumulatorProfile,
    `Slang.Generated.SourceWitness.Parsing.Table.Chunk01,
    `Slang.Generated.SourceWitness.Parsing.Table.Chunk02,
    `Slang.Generated.SourceWitness.Parsing.Table.Chunk03,
    `Slang.Generated.SourceWitness.Parsing.Table.Chunk04,
    `Slang.Generated.SourceWitness.Parsing.Table.Chunk05,
    `Slang.Generated.SourceWitness.Parsing.Table.Chunk06,
    `Slang.Generated.SourceWitness.Parsing.Table.Chunk07,
    `Slang.Generated.SourceWitness.Parsing.Table.Chunk08,
    `Slang.Generated.SourceWitness.Parsing.Table.Chunk09,
    `Slang.Generated.SourceWitness.Parsing.Cell.Chunk01,
    `Slang.Generated.SourceWitness.Parsing.Cell.Chunk02,
    `Slang.Generated.SourceWitness.Parsing.Cell.Chunk03,
    `Slang.Generated.SourceWitness.Parsing.Audit,
    `Slang.Generated.SourceWitness.Tokenizer.Cell.Chunk01,
    `Slang.Generated.SourceWitness.Tokenizer.Cell.Chunk02,
    `Slang.Generated.SourceWitness.Tokenizer.Table.Chunk01,
    `Slang.Generated.SourceWitness.Tokenizer.Table.Chunk02,
    `Slang.Generated.SourceWitness.Tokenizer.Table.Chunk03,
    `Slang.Generated.SourceWitness.Tokenizer.Std.Chunk01,
    `Slang.Generated.SourceWitness.Tokenizer.Std.Chunk02,
    `Slang.Generated.SourceWitness.Tokenizer.Std.Chunk03,
    `Slang.Generated.SourceWitness.Tokenizer.Std.Chunk04,
    `Slang.Generated.SourceWitness.Tokenizer.Std.Chunk05,
    `Slang.Generated.SourceWitness.Tokenizer.Std.Chunk06,
    `Slang.Generated.SourceWitness.Tokenizer.Std.Chunk07,
    `Slang.Generated.SourceWitness.Tokenizer.Std.Chunk08,
    `Slang.Generated.SourceWitness.Tokenizer.Std.Chunk09,
    `Slang.Generated.SourceWitness.Tokenizer.Std.Chunk10,
    `Slang.Generated.SourceWitness.Tokenizer.Std.Chunk11,
    `Slang.Generated.SourceWitness.Tokenizer.Audit,
    `Slang.Generated.SourceWitness.Decoder.ChunksA,
    `Slang.Generated.SourceWitness.Decoder.ChunksB,
    `Slang.Generated.SourceWitness.Decoder.Assembly,
    `Slang.Generated.SourceWitness.Decoder.Audit]

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
