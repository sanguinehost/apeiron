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
require «apeiron» from git "git@github.com:sanguinehost/sanguine.git" @
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
    `Security.InterpretationWitnesses,
    `Growable.Geometry, `Growable.Identity, `Growable.Scan,
    `Growable.ReadLoop, `Growable.LocateLoop, `Growable.ReadState, `Kepler.Certificate,
    `Growable.Search, `Growable.Read, `Growable.ReadComposition, `Growable.RelocationWF, `Growable.Classifier,
    `Growable.Relocation, `Growable.Refinement, `Growable.Resources, `Growable.Representation,
    `Growable.Rebuild,
    -- Added with the executed-body verification wave. `ApeironWordBridge` sets `roots` without
    -- `.submodules` globs, so an import of a root is built only when it is itself a root.
    -- `Growable.InsertRefine` was imported by `Growable.GaplessRead`/`Growable.BuildRefine`,
    -- neither reachable, so it was ungated.
    -- Row: `apeiron-verify-wave-land-nFsIRO8H9pKb1njJApKly`.
    `Growable.InsertRefine, `Growable.CostModel,
    `Map.Rep, `Map.Read, `Map.Selected,
    `Count.Rep, `Count.Read,
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
-- `sanguine/proof` (manifest `Substrate/vendor.sha256`, regeneration `tools/vendor-neural.sh (unpublished)`).
-- A separate default target owns module discovery for the vendored files; `ApeironWordBridge`
-- does not import them until the wave-1 audit registrations do.
@[default_target]
lean_lib SubstrateVendored where
  globs := #[.submodules `Substrate, .submodules `Closures]

-- AI-track re-anchor: native trichotomy-first surface and model
--. One library, module discovery
-- under Nous/; Core/ is ring-generic and Model/ is AI-specific.
lean_lib Nous where
  globs := #[.submodules `Nous]

-- Inference-parity numeric kernels: fixed-point Rust bodies and their real-valued refinements
--. Module discovery under Numeric/.
lean_lib Numeric where
  globs := #[.submodules `Numeric]

-- GPU/SPIR-V subset verification: mechanized fragment semantics, decoder soundness and the
-- checker/DP4A refinements. Module discovery under
-- Spirv/; roots are registered here by the lead before each task dispatches.
lean_lib Spirv where
  globs := #[.submodules `Spirv]

-- Bounded authorization decision core: the canonical definition staged
-- here until the corpus migration, its laws, and (W2) the analysis layer over Apeiron.Decider.
lean_lib Policy where
  globs := #[.submodules `Policy]

-- FIPS-202 Keccak-f[1600] constants and the refinement chain over them. Module discovery under
-- Crypto/.
lean_lib Crypto where
  globs := #[.submodules `Crypto]

-- SHA-256/SHA-256d rooting onto the trichotomy.
-- Module discovery under Icarus/.
lean_lib Icarus where
  globs := #[.submodules `Icarus]
