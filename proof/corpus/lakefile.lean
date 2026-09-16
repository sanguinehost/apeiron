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
  "8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f" / "proof"

-- Pin the same Mathlib revision as the corpus, last so this package's choice wins any transitive
-- resolution. The corpus is no longer on a 4.30.0-rc2-era Mathlib.
require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "v4.31.0"

@[default_target]
lean_lib ApeironWordBridge where
  roots := #[`WordRefinement, `TableRefinement, `TableRefinementGeneric,
    `TableRefinementBuild, `TableRefinementAdmission, `TableRefinementBuildGeneric,
    `TableRefinementReachablePrefixExactN, `TableRefinementAdmissionGeneric,
    `TableRefinementPhysical, `TableRefinementPhysicalBuild, `AxiomAudit,
    `Joint.TableInstance, `Joint.TableInstanceAudit, `Joint.WordBridge, `Slang.Audit.Empty,
    `Slang.Syntax, `Slang.Decode,
    `Slang.DecodeTest, `Slang.Word, `Slang.Machine, `Slang.MachineAudit,
    `Slang.Generated.Table, `Slang.Generated.TableDerivation,
    `Slang.Source.Bytes, `Slang.Source.BytesAudit,
    `Slang.Source.Lexing, `Slang.Source.LexingAudit, `Slang.Source.Grammar,
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
    `Slang.Joint.AccumulatorProfile]
