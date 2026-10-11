import Native.HyperbolicBase
import ApeironWord
import Growable.Identity
import Mathlib.Algebra.TrivSqZeroExt.Basic
import Lean

/-!
# Logical specification of the algebra-native locator

This module is generic over the physical `entryAt` observation. The adapter to
`GrowableSearch.entryAt`, `GrowableSearch.Absent`, `CompleteCoverage`, `DirectCover`, and
`SearchDecisionCorrect` is deliberately missing: importing the current module rebuilds the known-red
pre-summary `Growable.Search`. Summary deletion must supply that adapter instead of duplicating or
repairing the old vocabulary here.
-/

namespace Apeiron.NativeLocator

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

abbrev Address := Apeiron.NativeAddress.H3_40

/-! ## Three-arena scalar directory -/

/-- Handles are stable integer record numbers. They are never byte offsets or borrowed addresses. -/
abbrev ArenaHandle := UInt32

/-- The sole missing-child/edge/terminal marker; zero remains an ordinary handle and label. -/
def NO_CHILD : ArenaHandle := 4294967295

def residueStride : Nat := 4
def identityStride : Nat := 2
def siblingEdgeStride : Nat := 3

/-- The two filtered faces have distinct roots in the shared residue-record arena. -/
inductive ResidueRoot where
  | hyperbolic
  | parabolic
  deriving DecidableEq

def residueRootHandle : ResidueRoot → ArenaHandle
  | .hyperbolic => 0
  | .parabolic => 1

/-- A non-sentinel record handle denotes an initialized record, not a flattened word offset. -/
def InitializedHandle (handle : ArenaHandle) (recordCount : Nat) : Prop :=
  handle ≠ NO_CHILD ∧ handle.toNat < recordCount

/-- One copied scalar load; out-of-range reads produce the sentinel and cannot yield a live link. -/
def scalarWordAt (words : List UInt32) (index : Nat) : UInt32 := words.getD index NO_CHILD

/-- Checked stride arithmetic ties logical records to flat scalar words and the allocation bound. -/
structure CheckedRecordArena (words : List UInt32) (stride recordCount wordLimit byteLimit : Nat) :
    Prop where
  recordCount_before_sentinel : recordCount < NO_CHILD.toNat
  words_exact : words.length = stride * recordCount
  words_within_limit : words.length ≤ wordLimit
  bytes_exact : 4 * words.length ≤ byteLimit

/--
The production directory has exactly three flat scalar arenas. Every observer returns copied scalar
words or handles; no field has a node, subslice, iterator, or borrowed result type.
-/
structure ScalarArenaDirectory (State : Type*) where
  residueWords : State → List UInt32
  identityWords : State → List UInt32
  siblingEdgeWords : State → List UInt32
  residueRecordCount : State → Nat
  identityRecordCount : State → Nat
  siblingEdgeRecordCount : State → Nat
  representativeCount : State → Nat
  hyperbolicNodeCount : State → Nat
  parabolicNodeCount : State → Nat
  wordLimit : State → Nat
  byteLimit : State → Nat
  residueChild : State → ArenaHandle → Fin 3 → ArenaHandle
  residueIdentityRoot : State → ArenaHandle → ArenaHandle
  identityTerminal : State → ArenaHandle → ArenaHandle
  identityFirstEdge : State → ArenaHandle → ArenaHandle
  edgeLabel : State → ArenaHandle → UInt32
  edgeChildIdentity : State → ArenaHandle → ArenaHandle
  edgeNext : State → ArenaHandle → ArenaHandle
  residueOwnedBy : State → ResidueRoot → ArenaHandle → Prop
  sortedUniqueSiblingChain : State → ArenaHandle → Prop
  acyclicSiblingChain : State → ArenaHandle → Prop

/-- Empty storage has two implicit roots; materialized storage contains both root records. -/
def ResidueArenaMaterialization {State : Type*} (directory : ScalarArenaDirectory State)
    (state : State) : Prop :=
  (directory.residueWords state = [] ∧ directory.residueRecordCount state = 2) ∨
    CheckedRecordArena (directory.residueWords state) residueStride
      (directory.residueRecordCount state) (directory.wordLimit state) (directory.byteLimit state)

/--
The scalar-arena shape: checked strides, disjoint face ownership, sentinel-safe links, and sorted
acyclic identity sibling chains. Parabolic child 2 is always `NO_CHILD`.
-/
structure ScalarArenaShape {State : Type*} (directory : ScalarArenaDirectory State)
    (state : State) : Prop where
  residueLayout : ResidueArenaMaterialization directory state
  identityLayout : CheckedRecordArena (directory.identityWords state) identityStride
    (directory.identityRecordCount state) (directory.wordLimit state) (directory.byteLimit state)
  siblingEdgeLayout : CheckedRecordArena (directory.siblingEdgeWords state) siblingEdgeStride
    (directory.siblingEdgeRecordCount state) (directory.wordLimit state) (directory.byteLimit state)
  residueCounts : directory.residueRecordCount state =
    directory.hyperbolicNodeCount state + directory.parabolicNodeCount state
  rootsInitialized : ∀ root,
    InitializedHandle (residueRootHandle root) (directory.residueRecordCount state)
  rootsOwned : ∀ root, directory.residueOwnedBy state root (residueRootHandle root)
  rootOwnershipDisjoint : ∀ handle,
    ¬ (directory.residueOwnedBy state .hyperbolic handle ∧
      directory.residueOwnedBy state .parabolic handle)
  residueChildWords : ∀ parent digit,
    InitializedHandle parent (directory.residueRecordCount state) →
      directory.residueChild state parent digit =
        scalarWordAt (directory.residueWords state)
          (residueStride * parent.toNat + digit.val)
  residueIdentityWords : ∀ parent,
    InitializedHandle parent (directory.residueRecordCount state) →
      directory.residueIdentityRoot state parent =
        scalarWordAt (directory.residueWords state) (residueStride * parent.toNat + 3)
  identityTerminalWords : ∀ identity,
    InitializedHandle identity (directory.identityRecordCount state) →
      directory.identityTerminal state identity =
        scalarWordAt (directory.identityWords state) (identityStride * identity.toNat)
  identityFirstEdgeWords : ∀ identity,
    InitializedHandle identity (directory.identityRecordCount state) →
      directory.identityFirstEdge state identity =
        scalarWordAt (directory.identityWords state) (identityStride * identity.toNat + 1)
  edgeLabelWords : ∀ edge,
    InitializedHandle edge (directory.siblingEdgeRecordCount state) →
      directory.edgeLabel state edge =
        scalarWordAt (directory.siblingEdgeWords state) (siblingEdgeStride * edge.toNat)
  edgeChildWords : ∀ edge,
    InitializedHandle edge (directory.siblingEdgeRecordCount state) →
      directory.edgeChildIdentity state edge =
        scalarWordAt (directory.siblingEdgeWords state) (siblingEdgeStride * edge.toNat + 1)
  edgeNextWords : ∀ edge,
    InitializedHandle edge (directory.siblingEdgeRecordCount state) →
      directory.edgeNext state edge =
        scalarWordAt (directory.siblingEdgeWords state) (siblingEdgeStride * edge.toNat + 2)
  childInitialized : ∀ root parent digit,
    directory.residueOwnedBy state root parent →
      let child := directory.residueChild state parent digit
      child = NO_CHILD ∨
        InitializedHandle child (directory.residueRecordCount state) ∧
          directory.residueOwnedBy state root child
  parabolicChild2Absent : ∀ parent,
    directory.residueOwnedBy state .parabolic parent →
      directory.residueChild state parent ⟨2, by decide⟩ = NO_CHILD
  residueIdentityInitialized : ∀ parent,
    InitializedHandle parent (directory.residueRecordCount state) →
      let identity := directory.residueIdentityRoot state parent
      identity = NO_CHILD ∨ InitializedHandle identity (directory.identityRecordCount state)
  terminalInitialized : ∀ identity,
    InitializedHandle identity (directory.identityRecordCount state) →
      let terminal := directory.identityTerminal state identity
      terminal = NO_CHILD ∨ terminal.toNat < directory.representativeCount state
  firstEdgeInitialized : ∀ identity,
    InitializedHandle identity (directory.identityRecordCount state) →
      let edge := directory.identityFirstEdge state identity
      edge = NO_CHILD ∨ InitializedHandle edge (directory.siblingEdgeRecordCount state)
  edgeLinksInitialized : ∀ edge,
    InitializedHandle edge (directory.siblingEdgeRecordCount state) →
      InitializedHandle (directory.edgeChildIdentity state edge)
          (directory.identityRecordCount state) ∧
        (directory.edgeNext state edge = NO_CHILD ∨
          InitializedHandle (directory.edgeNext state edge)
            (directory.siblingEdgeRecordCount state))
  edgeLabelByte : ∀ edge,
    InitializedHandle edge (directory.siblingEdgeRecordCount state) →
      (directory.edgeLabel state edge).toNat < 256
  siblingChains : ∀ identity,
    InitializedHandle identity (directory.identityRecordCount state) →
      directory.sortedUniqueSiblingChain state identity ∧
        directory.acyclicSiblingChain state identity

/-- A complete filtered path ends in initialized residue and identity records under one face root. -/
def ArenaPathEndpoint {State : Type*} (directory : ScalarArenaDirectory State)
    (state : State) : Prop :=
  ∃ root residue identity,
    directory.residueOwnedBy state root residue ∧
      InitializedHandle residue (directory.residueRecordCount state) ∧
      directory.residueIdentityRoot state residue = identity ∧
      InitializedHandle identity (directory.identityRecordCount state)

/-- Loop-carried traversal state consists only of copied scalar values. -/
structure ScalarTraversalCursor where
  depth : Nat
  handle : ArenaHandle
  missing : Bool
  stoppingDepth : Nat
  deriving DecidableEq

/--
Traversal reads one copied handle per step. This signature cannot carry a borrowed node, child array,
subslice, iterator, or arena reference from one step to the next.
-/
structure ScalarTraversalContract {State Query : Type*}
    (directory : ScalarArenaDirectory State) where
  residueStep : State → Query → ScalarTraversalCursor → ScalarTraversalCursor
  identityStep : State → Query → ScalarTraversalCursor → ScalarTraversalCursor
  residueStep_copiedHandle : ∀ state query cursor,
    let handle := (residueStep state query cursor).handle
    handle = NO_CHILD ∨ InitializedHandle handle (directory.residueRecordCount state)
  identityStep_copiedHandle : ∀ state query cursor,
    let handle := (identityStep state query cursor).handle
    handle = NO_CHILD ∨ InitializedHandle handle (directory.identityRecordCount state)
  sentinelStops : ∀ state query cursor,
    cursor.handle = NO_CHILD →
      (residueStep state query cursor).missing = true ∧
        (identityStep state query cursor).missing = true

/-- Linking consumes copied handles and preserves the complete arena shape. -/
structure ScalarLinkingPreservation {State : Type*} (directory : ScalarArenaDirectory State)
    (link : State → ArenaHandle → ArenaHandle → State → Prop) : Prop where
  preservesShape : ∀ before parent child after,
    ScalarArenaShape directory before → link before parent child after →
      ScalarArenaShape directory after
  copiedHandlesOnly : ∀ before parent child after,
    link before parent child after →
      parent ≠ NO_CHILD ∧ child ≠ NO_CHILD

/-- A representative is live when the physical observation decodes its full key and value. -/
def Live {State Representative Sigma Key Value : Type}
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (state : State) (representative : Representative) (query : Sigma × Key) : Prop :=
  ∃ value, entryAt state representative = some (query, value)

/--
A channel link is an in-bounds (`Fin d`) association whose representative still denotes a live
payload. The raw association is supplied by the future locator.
-/
def Link {State Representative Sigma Key Value : Type} {d : Nat}
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (association : State → Fin d → Address → Representative → Prop)
    (state : State) (channel : Fin d) (value : Address) (representative : Representative) : Prop :=
  association state channel value representative ∧
    ∃ query, Live entryAt state representative query

/-- No full decoded key has two live physical representatives. -/
def UniqueDecodedKeys {State Representative Sigma Key Value : Type}
    (entryAt : State → Representative → Option ((Sigma × Key) × Value)) (state : State) : Prop :=
  ∀ representative₁ representative₂ query value₁ value₂,
    entryAt state representative₁ = some (query, value₁) →
    entryAt state representative₂ = some (query, value₂) →
    representative₁ = representative₂ ∧ value₁ = value₂

/--
The bidirectional locator invariant. Coverage publishes every projection of one live full-key
representative. Soundness rejects stale, cross-key, and out-of-range associations by recovering that
same representative and its projected value.
-/
structure LocatorInvariant {State Representative Sigma Key Value : Type} {d : Nat}
    (directory : ScalarArenaDirectory State)
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (address : Sigma × Key → Address) (projection : Fin d → Address → Address)
    (association : State → Fin d → Address → Representative → Prop) (state : State) : Prop where
  arenaShape : ScalarArenaShape directory state
  uniqueDecodedKeys : UniqueDecodedKeys entryAt state
  coverage : ∀ representative query, Live entryAt state representative query →
    ∀ channel, Link entryAt association state channel (projection channel (address query))
      representative
  linksUseScalarArenas : ∀ channel value representative,
    Link entryAt association state channel value representative → ArenaPathEndpoint directory state
  linkSound : ∀ channel value representative,
    Link entryAt association state channel value representative →
      ∃ query, Live entryAt state representative query ∧
        value = projection channel (address query)

/--
An adapter identifies an implementation's existing absence vocabulary with no live full-key
representative. The successor source bridge must instantiate this with `GrowableSearch.Absent`.
-/
structure AbsentAdapter {State Representative Sigma Key Value : Type}
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (absent : State → Sigma × Key → Prop) : Prop where
  exact : ∀ state query, absent state query ↔ ¬ ∃ representative, Live entryAt state representative query

/-- Missing any required projected association excludes every live representative of the query. -/
theorem absent_of_projected_missing
    {State Representative Sigma Key Value : Type} {d : Nat}
    {entryAt : State → Representative → Option ((Sigma × Key) × Value)}
    {absent : State → Sigma × Key → Prop} {address : Sigma × Key → Address}
    {projection : Fin d → Address → Address}
    {association : State → Fin d → Address → Representative → Prop} {state : State}
    {query : Sigma × Key}
    (adapter : AbsentAdapter entryAt absent)
    {directory : ScalarArenaDirectory State}
    (invariant : LocatorInvariant directory entryAt address projection association state)
    (missing : ∃ channel, ¬ ∃ representative,
      Link entryAt association state channel (projection channel (address query)) representative) :
    absent state query := by
  rw [adapter.exact]
  obtain ⟨channel, hmissing⟩ := missing
  intro present
  obtain ⟨representative, hlive⟩ := present
  exact hmissing ⟨representative, invariant.coverage representative query hlive channel⟩

/-- Presence requires one physical representative of the complete `(Sigma, key)` identity. -/
def PresenceObligation {State Representative Sigma Key Value : Type}
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (state : State) (sigma : Sigma) (key : Key) : Prop :=
  ∃ representative value, entryAt state representative = some ((sigma, key), value)

/-- Component-wise hits may use different representatives; this proposition does not require one. -/
def ComponentConjunction {State Representative Sigma Key Value : Type} {d : Nat}
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (address : Sigma × Key → Address) (projection : Fin d → Address → Address)
    (association : State → Fin d → Address → Representative → Prop)
    (state : State) (query : Sigma × Key) : Prop :=
  ∀ channel, ∃ representative,
    Link entryAt association state channel (projection channel (address query)) representative

/-- A common live representative supplies every component; the converse is deliberately absent. -/
theorem componentConjunction_of_presence
    {State Representative Sigma Key Value : Type} {d : Nat}
    {entryAt : State → Representative → Option ((Sigma × Key) × Value)}
    {address : Sigma × Key → Address} {projection : Fin d → Address → Address}
    {association : State → Fin d → Address → Representative → Prop} {state : State}
    {query : Sigma × Key}
    {directory : ScalarArenaDirectory State}
    (invariant : LocatorInvariant directory entryAt address projection association state)
    (presence : PresenceObligation entryAt state query.1 query.2) :
    ComponentConjunction entryAt address projection association state query := by
  obtain ⟨representative, value, hentry⟩ := presence
  intro channel
  exact ⟨representative, invariant.coverage representative query ⟨value, hentry⟩ channel⟩

/-- Component conjunction is insufficient exactly when its witnesses do not share a full-key rep. -/
def MarginalsFromDifferentKeys
    {State Representative Sigma Key Value : Type} {d : Nat}
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (address : Sigma × Key → Address) (projection : Fin d → Address → Address)
    (association : State → Fin d → Address → Representative → Prop)
    (state : State) (query : Sigma × Key) : Prop :=
  ComponentConjunction entryAt address projection association state query ∧
    ¬ PresenceObligation entryAt state query.1 query.2

/-- The missing source theorem must rule out every known false exclusion mechanism. -/
inductive ExclusionCounterexampleClass where
  | unpublishedLink
  | staleLink
  | relocatedOutOfRange
  | matchingRepresentativeAfterEarlyVacancy
  | marginalsFromDifferentKeys
  deriving DecidableEq

/--
Exact missing source obligation: every reachable state instantiates the existing absence adapter and
the locator invariant, and a reported missing projection is the `Link` negation consumed above.
-/
structure SourceLinkObligation
    {State Representative Sigma Key Value : Type} {d : Nat}
    (directory : ScalarArenaDirectory State)
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (absent : State → Sigma × Key → Prop) (address : Sigma × Key → Address)
    (projection : Fin d → Address → Address)
    (association : State → Fin d → Address → Representative → Prop)
    (reachable : State → Prop) (lookupMissing : State → Sigma × Key → Fin d → Prop) : Prop where
  absentAdapter : AbsentAdapter entryAt absent
  reachableInvariant : ∀ state, reachable state →
    LocatorInvariant directory entryAt address projection association state
  missingReflectsAssociation : ∀ state query channel,
    reachable state → lookupMissing state query channel →
      ¬ ∃ representative,
        Link entryAt association state channel (projection channel (address query)) representative

/-- Work charged to the locator rather than hidden in allocator or callback terms. -/
structure WorkCounts where
  comparisons : Nat
  bytesInspected : Nat
  bytesCopied : Nat
  indexSteps : Nat
  linkWrites : Nat
  nodesInitialized : Nat
  growthSteps : Nat
  relocationSteps : Nat
  fullKeyComparisons : Nat

def WorkCounts.total (work : WorkCounts) : Nat :=
  work.comparisons + work.bytesInspected + work.bytesCopied + work.indexSteps + work.linkWrites +
    work.nodesInitialized + work.growthSteps + work.relocationSteps + work.fullKeyComparisons

/-- Operations remain ordered, so this quantifies over adversarial insertion order. -/
structure Operation where
  keyBytes : List UInt8
  isInsert : Bool

/-- Maximum live-key count over the initial state, every intermediate, and the final state. -/
def maximumLive {State : Type} (liveCount : State → Nat)
    (initial : State) (intermediate : List State) (final : State) : Nat :=
  ((initial :: intermediate) ++ [final]).foldl (fun maximum state => max maximum (liveCount state)) 0

/--
The required HashDOS proposition. It ranges over every pepper and operation sequence, so known
pepper, equal descriptors, common prefixes, key bytes, and insertion order cannot become premises.
`WorkCounts` explicitly charges growth, relocation, and repeated full-key comparisons. Allocation
and callback realization remain separate charged terms.
-/
structure HashDOSBound (Profile Pepper State : Type)
    (empty : Profile → State) (reachable : Profile → State → Prop)
    (run : Profile → Pepper → State → List Operation → List State → State → Prop)
    (descriptor : Profile → Pepper → List UInt8 → Address)
    (liveCount : State → Nat) (work : Profile → Pepper → State → List Operation → WorkCounts)
    (allocatorCost callbackCost : Profile → Pepper → State → List Operation → Nat)
    (profileConstant : Profile → Nat) (potential : Profile → State → Nat) : Prop where
  potential_empty : ∀ profile, potential profile (empty profile) = 0
  potential_storage_bound : ∀ profile state, reachable profile state →
    potential profile state ≤ profileConstant profile * (liveCount state + 1)
  aggregate : ∀ profile pepper initial intermediate final operations,
    reachable profile initial → run profile pepper initial operations intermediate final →
      let operationCount := operations.length
      let inputBytes := (operations.map (fun operation => operation.keyBytes.length)).sum
      let liveMaximum := maximumLive liveCount initial intermediate final
      (work profile pepper initial operations).total + potential profile final ≤
        potential profile initial +
          profileConstant profile * operationCount * (1 + Nat.log2 (liveMaximum + 1)) +
          profileConstant profile * inputBytes +
          allocatorCost profile pepper initial operations +
          callbackCost profile pepper initial operations
  lookupWorstCase : ∀ profile pepper state keyBytes,
    reachable profile state →
      (work profile pepper state [⟨keyBytes, false⟩]).total ≤
        profileConstant profile * (40 + 256 * keyBytes.length + 1)

/-- Equal descriptors are included because the aggregate theorem has no collision premise. -/
theorem equalDescriptors_are_covered {Profile Pepper State : Type}
    {empty : Profile → State} {reachable : Profile → State → Prop}
    {run : Profile → Pepper → State → List Operation → List State → State → Prop}
    {descriptor : Profile → Pepper → List UInt8 → Address}
    {liveCount : State → Nat} {work : Profile → Pepper → State → List Operation → WorkCounts}
    {allocatorCost callbackCost : Profile → Pepper → State → List Operation → Nat}
    {profileConstant : Profile → Nat} {potential : Profile → State → Nat}
    (bound : HashDOSBound Profile Pepper State empty reachable run descriptor liveCount work
      allocatorCost callbackCost profileConstant potential)
    (profile : Profile) (pepper : Pepper) (initial : State) (intermediate : List State)
    (final : State) (operations : List Operation)
    (_equalDescriptors : ∃ value, ∀ operation, operation ∈ operations →
      descriptor profile pepper operation.keyBytes = value)
    (hreachable : reachable profile initial)
    (hrun : run profile pepper initial operations intermediate final) :
    let operationCount := operations.length
    let inputBytes := (operations.map (fun operation => operation.keyBytes.length)).sum
    let liveMaximum := maximumLive liveCount initial intermediate final
    (work profile pepper initial operations).total + potential profile final ≤
      potential profile initial +
        profileConstant profile * operationCount * (1 + Nat.log2 (liveMaximum + 1)) +
        profileConstant profile * inputBytes + allocatorCost profile pepper initial operations +
        callbackCost profile pepper initial operations :=
  bound.aggregate profile pepper initial intermediate final operations hreachable hrun

/-- The trie plus pepper is a candidate, but no source-linked `HashDOSBound` inhabitant exists. -/
inductive TriePepperCostVerdict where
  | notYetImplied
  deriving DecidableEq

def triePepperCostVerdict : TriePepperCostVerdict := .notYetImplied

theorem triePepper_cost_not_yet_implied : triePepperCostVerdict = .notYetImplied := rfl

/-! ## Face-indexed residue contract -/

/-- The three executing address faces. This is contract vocabulary, not the Rust `Sigma` type. -/
inductive Face where
  | elliptic
  | parabolic
  | hyperbolic
  deriving DecidableEq

/-- The physical index selected by each face. -/
inductive FaceIndexPolicy where
  | exactPackNoLocator
  | base2ResidueTrie
  | base3ResidueTrie
  deriving DecidableEq

def faceIndexPolicy : Face → FaceIndexPolicy
  | .elliptic => .exactPackNoLocator
  | .parabolic => .base2ResidueTrie
  | .hyperbolic => .base3ResidueTrie

/--
Elliptic admission uses an exact key packing into the selected field. There is no secondary
locator: an empty packed address certifies absence only when `addressCovered` covers every live
representative and the address is empty in that covered storage.
-/
structure EllipticExactAddress (Key Carrier : Type*) [Field Carrier] where
  pack : Key → Carrier
  unpack : Carrier → Option Key
  live : Key → Prop
  addressCovered : Carrier → Prop
  addressEmpty : Carrier → Prop
  pack_exact : ∀ key, unpack (pack key) = some key
  empty_exact_address_absent : ∀ key,
    addressCovered (pack key) → addressEmpty (pack key) → ¬ live key

/-- The coefficient ring used by the parabolic residue trie. -/
abbrev ParabolicResidueRing (k : Nat) := ZMod (2 ^ k)

/-- `R[ε]/ε²`, represented by Mathlib's square-zero extension. -/
abbrev ParabolicDual (k : Nat) :=
  TrivSqZeroExt (ParabolicResidueRing k) (ParabolicResidueRing k)

/-- Quotienting the dual numbers by `(ε)` retains the scalar coefficient `a`. -/
def parabolicResidueHom (k : Nat) :
    ParabolicDual k →+* ParabolicResidueRing k :=
  (TrivSqZeroExt.fstHom (ParabolicResidueRing k) (ParabolicResidueRing k)
    (ParabolicResidueRing k)).toRingHom

/-- The parabolic locator indexes the complete base-2 residue of `a`, never `b`. -/
def parabolicLocatorIndex (k : Nat) (z : ParabolicDual k) : ParabolicResidueRing k :=
  parabolicResidueHom k z

/-- The parabolic residue map preserves zero, one, addition, and multiplication. -/
theorem parabolic_residue_hom (k : Nat) (z w : ParabolicDual k) :
    parabolicResidueHom k 0 = 0 ∧
      parabolicResidueHom k 1 = 1 ∧
      parabolicResidueHom k (z + w) = parabolicResidueHom k z + parabolicResidueHom k w ∧
      parabolicResidueHom k (z * w) = parabolicResidueHom k z * parabolicResidueHom k w := by
  simp

/-- The scalar hyperbolic address embeds as `(a,0)` in the quadratic algebra. -/
def hyperbolicScalarEmbedding (a : Address) : Trichotomy.AnyRing.RingCell Address := (a, 0)

/-- At `σ=1`, `r=s=1`, both channel reads of the scalar embedding are the same residue. -/
theorem hyperbolic_scalar_channels_equal (a : Address) :
    (Trichotomy.AnyRing.ringUpper 1 (hyperbolicScalarEmbedding a),
      Trichotomy.AnyRing.ringLower 1 (hyperbolicScalarEmbedding a)) = (a, a) := by
  simp [hyperbolicScalarEmbedding, Trichotomy.AnyRing.ringUpper,
    Trichotomy.AnyRing.ringLower]

/-- H3_40 therefore indexes one scalar residue, not a redundant channel pair. -/
def hyperbolicLocatorIndex (a : Address) : Address := a

theorem face_index_policy_exact :
    faceIndexPolicy .elliptic = .exactPackNoLocator ∧
      faceIndexPolicy .parabolic = .base2ResidueTrie ∧
      faceIndexPolicy .hyperbolic = .base3ResidueTrie := by
  decide

/-! ## Coverage of the table's authoritative storage -/

/--
Coverage is over the table's live storage, not merely representatives owned by a standalone
locator. `completePath` includes the complete face residue path and exact-identity path.
`reverseLink` is the terminal-to-authoritative-payload association.
-/
structure TableStorageCoverage {State Representative Sigma Key Value : Type}
    (directory : ScalarArenaDirectory State)
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (completePath : State → Sigma × Key → Representative → Prop)
    (reverseLink : State → Representative → Sigma × Key → Prop) (state : State) : Prop where
  arenaShape : ScalarArenaShape directory state
  uniqueDecodedKeys : UniqueDecodedKeys entryAt state
  everyLiveRepresentativeReachable : ∀ representative query,
    Live entryAt state representative query → completePath state query representative
  completePathsUseScalarArenas : ∀ query representative,
    completePath state query representative → ArenaPathEndpoint directory state
  completePathSound : ∀ query representative,
    completePath state query representative → Live entryAt state representative query
  reverseLinksSound : ∀ representative query,
    reverseLink state representative query ↔
      completePath state query representative ∧ Live entryAt state representative query

/-- A state transition preserves the complete table/locator coverage relation. -/
def PreservesTableCoverage {State Representative Sigma Key Value : Type}
    (directory : ScalarArenaDirectory State)
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (completePath : State → Sigma × Key → Representative → Prop)
    (reverseLink : State → Representative → Sigma × Key → Prop)
    (step : State → State → Prop) : Prop :=
  ∀ before after,
    TableStorageCoverage directory entryAt completePath reverseLink before → step before after →
      TableStorageCoverage directory entryAt completePath reverseLink after

/-- Constructor and every state-changing table path must establish or preserve coverage. -/
structure TableCoverageTransitions {State Representative Sigma Key Value : Type}
    (directory : ScalarArenaDirectory State)
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (completePath : State → Sigma × Key → Representative → Prop)
    (reverseLink : State → Representative → Sigma × Key → Prop)
    (empty : State) (insert growth relocation merge : State → State → Prop) : Prop where
  constructor : TableStorageCoverage directory entryAt completePath reverseLink empty
  insertion : PreservesTableCoverage directory entryAt completePath reverseLink insert
  growth : PreservesTableCoverage directory entryAt completePath reverseLink growth
  relocation : PreservesTableCoverage directory entryAt completePath reverseLink relocation
  merge : PreservesTableCoverage directory entryAt completePath reverseLink merge

/-- A missing complete path excludes a live table representative only under table-wide coverage. -/
theorem table_absent_of_complete_path_missing
    {State Representative Sigma Key Value : Type}
    {entryAt : State → Representative → Option ((Sigma × Key) × Value)}
    {completePath : State → Sigma × Key → Representative → Prop}
    {reverseLink : State → Representative → Sigma × Key → Prop}
    {directory : ScalarArenaDirectory State}
    {state : State} {query : Sigma × Key}
    (coverage : TableStorageCoverage directory entryAt completePath reverseLink state)
    (missing : ¬ ∃ representative, completePath state query representative) :
    ¬ ∃ representative, Live entryAt state representative query := by
  intro present
  obtain ⟨representative, live⟩ := present
  exact missing ⟨representative, coverage.everyLiveRepresentativeReachable representative query live⟩

/-- `Absent` carries the proof preconditions; `Unknown` carries no absence conclusion. -/
inductive CertifiedDecision (covered missing : Prop) : Type where
  | absent (hcovered : covered) (hmissing : missing)
  | unknown

theorem certified_absent_requires_coverage {covered missing : Prop}
    (decision : CertifiedDecision covered missing) :
    (∃ hcovered hmissing, decision = .absent hcovered hmissing) → covered ∧ missing := by
  rintro ⟨hcovered, hmissing, rfl⟩
  exact ⟨hcovered, hmissing⟩

/-- Conjuncts already present in this module versus obligations requiring actual table bodies. -/
inductive CoverageConjunct where
  | locatorUnique
  | locatorCoverage
  | locatorLinkSound
  | tableCompletePath
  | tableReverseLinks
  | constructor
  | insertion
  | growth
  | relocation
  | merge
  deriving DecidableEq

def provedByAbstractLocatorSpec : CoverageConjunct → Bool
  | .locatorUnique | .locatorCoverage | .locatorLinkSound => true
  | .tableCompletePath | .tableReverseLinks | .constructor | .insertion | .growth
  | .relocation | .merge => false

theorem locatorSpec_coverage_boundary :
    provedByAbstractLocatorSpec .locatorUnique = true ∧
      provedByAbstractLocatorSpec .locatorCoverage = true ∧
      provedByAbstractLocatorSpec .locatorLinkSound = true ∧
      provedByAbstractLocatorSpec .tableCompletePath = false ∧
      provedByAbstractLocatorSpec .tableReverseLinks = false ∧
      provedByAbstractLocatorSpec .constructor = false ∧
      provedByAbstractLocatorSpec .insertion = false ∧
      provedByAbstractLocatorSpec .growth = false ∧
      provedByAbstractLocatorSpec .relocation = false ∧
      provedByAbstractLocatorSpec .merge = false := by
  decide

/-! ## Admission and resource ordering -/

/-- Equality-faithful finite identity bytes, independent of the face residue. -/
structure ExactIdentityContract (Key Retained : Type*) where
  retain : Key → Retained
  bytes : Key → List UInt8
  retainedBytes : Retained → List UInt8
  retain_bytes : ∀ key, retainedBytes (retain key) = bytes key
  bytes_exact : ∀ left right, bytes left = bytes right ↔ left = right

/-- The source capability includes the finite length charged by the locator's byte walk. -/
structure FiniteExactIdentityContract (Key Retained : Type*)
    extends ExactIdentityContract Key Retained where
  byteLength : Key → Nat
  byteLength_exact : ∀ key, (bytes key).length = byteLength key
  retainedByteLength : Retained → Nat
  retainedByteLength_exact : ∀ retained,
    (retainedBytes retained).length = retainedByteLength retained
  queryEncodingByte : Key → Nat → UInt8
  queryEncodingByte_exact : ∀ key index,
    queryEncodingByte key index = (bytes key).getD index 0
  heldEncodingByte : Retained → Nat → UInt8
  heldEncodingByte_exact : ∀ retained index,
    heldEncodingByte retained index = (retainedBytes retained).getD index 0

/-- Byte keys inhabit the capability with their bytes as the retained exact identity. -/
def byteKeyExactIdentity : FiniteExactIdentityContract (List UInt8) (List UInt8) where
  retain := id
  bytes := id
  retainedBytes := id
  retain_bytes := by intro key; rfl
  bytes_exact := by intro left right; rfl
  byteLength := List.length
  byteLength_exact := by intro key; rfl
  retainedByteLength := List.length
  retainedByteLength_exact := by intro retained; rfl
  queryEncodingByte := fun key index => key.getD index 0
  queryEncodingByte_exact := by intro key index; rfl
  heldEncodingByte := fun retained index => retained.getD index 0
  heldEncodingByte_exact := by intro retained index; rfl

/--
`Arc<T>` adds no identity data: the Rust instance delegates retention and both indexed byte views to
`T`. This contract-level operation records that transparent delegation without introducing a second
identity encoding.
-/
def arcDelegatedExactIdentity {Key Retained : Type*}
    (identity : FiniteExactIdentityContract Key Retained) :
    FiniteExactIdentityContract Key Retained := identity

theorem arc_identity_delegates {Key Retained : Type*}
    (identity : FiniteExactIdentityContract Key Retained) :
    (arcDelegatedExactIdentity identity).retain = identity.retain ∧
      (arcDelegatedExactIdentity identity).bytes = identity.bytes ∧
      (arcDelegatedExactIdentity identity).retainedBytes = identity.retainedBytes ∧
      (arcDelegatedExactIdentity identity).queryEncodingByte = identity.queryEncodingByte ∧
      (arcDelegatedExactIdentity identity).heldEncodingByte = identity.heldEncodingByte := by
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- `Regen` is admitted only when its retained word is an exact identity codec, not a residue. -/
structure RegenIdentityContract (Key Identity : Type*) extends ExactIdentityContract Key Identity where
  decode : Identity → Key
  decode_encode : ∀ key, decode (retain key) = key
  encode_decode : ∀ identity, retain (decode identity) = identity

/-- An opaque `K : Eq` needs both witnesses; `Eq` and a placement closure supply neither. -/
structure OpaqueKeyLocatorAdmission (Key Residue Retained : Type*) where
  residue : Key → Residue
  identity : ExactIdentityContract Key Retained
  homeDenotesResidue : (Key → Residue) → Prop
  admittedHome : homeDenotesResidue residue

/-! ## Table-facing face and home capabilities -/

/-- The immutable validated profile token accepted by the executing hyperbolic encoder. -/
inductive H3_40Token where
  | validated
  deriving DecidableEq

/-- A face address is semantic data; no constructor is a physical table index. -/
inductive FaceAddress where
  | elliptic (packed : UInt64)
  | parabolic (residue : ParabolicResidueRing 64)
  | hyperbolic (profile : H3_40Token) (residue : Address)

/-- The two residue maps required on every admitted face domain. -/
structure FaceResidueCapability (Key Home : Type*) where
  parabolicDual : Home → Key → ParabolicDual 64
  hyperbolicScalar : Home → H3_40Token → Key → Address

/-- Exact at-most-word packing is separate from the two locator residues. -/
structure EllipticPacking (Key Home : Type*) where
  pack : Home → Key → UInt64
  unpack : Home → UInt64 → Option Key
  exact : ∀ home key, unpack home (pack home key) = some key

/-- The only face tokens accepted by a secondary locator. Elliptic is unrepresentable here. -/
inductive LocatorSigma where
  | parabolic
  | hyperbolic
  deriving DecidableEq

def LocatorSigma.face : LocatorSigma → Face
  | .parabolic => .parabolic
  | .hyperbolic => .hyperbolic

def LocatorSigma.fromFace : Face → Option LocatorSigma
  | .elliptic => none
  | .parabolic => some .parabolic
  | .hyperbolic => some .hyperbolic

/-- The sealed Rust `FaceDomain` choices, represented extensionally. -/
inductive FaceDomain where
  | sigma
  | locatorSigma
  deriving DecidableEq

def FaceDomain.Faces : FaceDomain → Type
  | .sigma => Face
  | .locatorSigma => LocatorSigma

def FaceDomain.default : (domain : FaceDomain) → domain.Faces
  | .sigma => .parabolic
  | .locatorSigma => .parabolic

def FaceDomain.face : {domain : FaceDomain} → domain.Faces → Face
  | .sigma, face => face
  | .locatorSigma, face => face.face

def FaceDomain.fromFace : (domain : FaceDomain) → Face → Option domain.Faces
  | .sigma, face => some face
  | .locatorSigma, face => LocatorSigma.fromFace face

/-- Sigma-domain encoding requires exact packing; locator-domain encoding cannot name Elliptic. -/
def FaceDomain.encode {Key Home : Type*}
    (residue : FaceResidueCapability Key Home) (home : Home) (profile : H3_40Token) :
    (domain : FaceDomain) →
      (match domain with | .sigma => EllipticPacking Key Home | .locatorSigma => PUnit) →
      domain.Faces → Key → FaceAddress
  | .sigma, packing, .elliptic, key => .elliptic (packing.pack home key)
  | .sigma, _, .parabolic, key =>
      .parabolic (parabolicResidueHom 64 (residue.parabolicDual home key))
  | .sigma, _, .hyperbolic, key =>
      .hyperbolic profile (residue.hyperbolicScalar home profile key)
  | .locatorSigma, _, .parabolic, key =>
      .parabolic (parabolicResidueHom 64 (residue.parabolicDual home key))
  | .locatorSigma, _, .hyperbolic, key =>
      .hyperbolic profile (residue.hyperbolicScalar home profile key)

/--
The selected typed home. Its dependent `packing` field is an exact packing contract only for the
Sigma domain and `PUnit` for the locator-only domain, so a locator-only home neither supplies nor
silently fabricates elliptic packing.
-/
structure FaceHome (Key Home : Type*) where
  domain : FaceDomain
  residue : FaceResidueCapability Key Home
  packing : match domain with | .sigma => EllipticPacking Key Home | .locatorSigma => PUnit
  address : Home → H3_40Token → domain.Faces → Key → FaceAddress

/--
The laws of Rust `FaceHome<K>`. Its address is stable, equality-respecting, and agrees with the
shared exhaustive `FaceDomain` encoding. A placement-only `Fn(&K) → u64` supplies none of these
equations, and no closure blanket exists.
-/
structure FaceHomeLaws (Key Home : Type*) (capability : FaceHome Key Home) : Prop where
  equalityRespecting : ∀ home profile face left right, left = right →
    capability.address home profile face left = capability.address home profile face right
  addressAgrees : ∀ home profile face key,
    capability.address home profile face key =
      FaceDomain.encode capability.residue home profile capability.domain capability.packing face key

/--
Rust `Home<K> : FaceHome<K>` keeps a separate physical placement interface. Both operations are
mandatory here, matching removal of the placement-address default; neither result is a semantic
face address or an absence certificate.
-/
structure TableHomeCapability (Key Home PhysicalAddress : Type*) where
  semantic : FaceHome Key Home
  home : Home → Key → UInt64
  physicalAddress : Home → Face → Key → PhysicalAddress

/-- The complete capability consumed by a table-backed locator. -/
structure FaceLocatorCapability (Key Retained Home : Type*) where
  identity : FiniteExactIdentityContract Key Retained
  home : FaceHome Key Home
  homeLaws : FaceHomeLaws Key Home home

/-- The all-face domain necessarily carries exact packing. -/
def sigmaDomainAdmitted {Key Home : Type*}
    (residue : FaceResidueCapability Key Home) (packing : EllipticPacking Key Home)
    (home : Home) (profile : H3_40Token) (face : Face) (key : Key) : FaceAddress :=
  FaceDomain.encode residue home profile .sigma packing face key

/-- The locator-only domain has exactly Parabolic and Hyperbolic tokens. -/
def locatorDomainAdmitted {Key Home : Type*}
    (residue : FaceResidueCapability Key Home)
    (home : Home) (profile : H3_40Token) (face : LocatorSigma) (key : Key) : FaceAddress :=
  FaceDomain.encode residue home profile .locatorSigma ⟨⟩ face key

theorem typed_face_domain_exact :
    FaceDomain.fromFace .sigma .elliptic = some .elliptic ∧
      FaceDomain.fromFace .locatorSigma .elliptic = none ∧
      FaceDomain.fromFace .locatorSigma .parabolic = some .parabolic ∧
      FaceDomain.fromFace .locatorSigma .hyperbolic = some .hyperbolic := by
  exact ⟨rfl, rfl, rfl, rfl⟩

/-! ## Generated elliptic word packing -/

namespace Native.Address

/-- The extracted word codec decodes its own elliptic packed image. -/
theorem pack_roundtrip (key : Aeneas.Std.I64) :
    (do
      let packed ←
        apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressEllipticPackingI64.elliptic_pack
          () key
      apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressEllipticPackingI64.elliptic_unpack
        () packed) = Aeneas.Std.Result.ok (some key) := by
  simp [apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressEllipticPackingI64.elliptic_pack,
    apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressEllipticPackingI64.elliptic_unpack,
    apeiron_word.coord.address.WordFaceHome.Insts.Apeiron_wordCoordAddressEllipticPackingI64.elliptic_pack,
    apeiron_word.coord.address.WordFaceHome.Insts.Apeiron_wordCoordAddressEllipticPackingI64.elliptic_unpack,
    apeiron_word.I64.Insts.Apeiron_wordCoordPacked.packed,
    apeiron_word.I64.Insts.Apeiron_wordCoordPacked.unpack,
    Aeneas.Std.UScalar.hcast, Aeneas.Std.IScalar.hcast]

end Native.Address

/-! ## Generated all-face word addresses -/

namespace Native.Address

open Aeneas Aeneas.Std Result

/-- The generated validator reaches its sole sealed profile from the frozen constants. -/
theorem h3_40_new_ok :
    ∃ profile : apeiron_word.coord.address.H3_40,
      apeiron_word.coord.address.H3_40.new = Aeneas.Std.Result.ok (some profile) := by
  let HInv : Aeneas.Std.U64 × Aeneas.Std.U32 → Prop := fun state =>
    state.2.val ≤ 40 ∧ state.1.val = 3 ^ state.2.val
  have hloop :
      apeiron_word.coord.address.H3_40.validate_loop
          apeiron_word.coord.address.H3_40.P
          apeiron_word.coord.address.H3_40.K
          apeiron_word.coord.address.H3_40.Q
          apeiron_word.coord.address.H3_40.R
          apeiron_word.coord.address.H3_40.S
          apeiron_word.coord.address.H3_40.INVERSE_OF_TWO 1#u64 0#u32 =
        Aeneas.Std.Result.ok true := by
    unfold apeiron_word.coord.address.H3_40.validate_loop
    have hs : loop
        (fun state =>
          apeiron_word.coord.address.H3_40.validate_loop.body
            apeiron_word.coord.address.H3_40.P
            apeiron_word.coord.address.H3_40.K
            apeiron_word.coord.address.H3_40.Q
            apeiron_word.coord.address.H3_40.R
            apeiron_word.coord.address.H3_40.S
            apeiron_word.coord.address.H3_40.INVERSE_OF_TWO state.1 state.2)
        (1#u64, 0#u32) ⦃ result => result = true ⦄ := by
      apply loop.spec_decr_nat
        (measure := fun state : Aeneas.Std.U64 × Aeneas.Std.U32 => 40 - state.2.val)
        (inv := HInv)
        (post := fun result => result = true)
      · intro state hstate
        rcases state with ⟨power, step⟩
        rcases hstate with ⟨hstep, hpower⟩
        dsimp only [HInv] at hstep hpower ⊢
        unfold apeiron_word.coord.address.H3_40.validate_loop.body
        by_cases hlt : step < apeiron_word.coord.address.H3_40.K
        · rw [if_pos hlt]
          have hstepLt : step.val < 40 := by
            simpa [apeiron_word.coord.address.H3_40.K] using hlt
          simp only [Aeneas.Std.lift, Aeneas.Std.bind_tc_ok]
          simp only [apeiron_word.coord.address.H3_40.P]
          cases hmul : Aeneas.Std.U64.checked_mul power 3#u64 with
          | none =>
            have hover := Aeneas.Std.U64.checked_mul_bv_spec power 3#u64
            rw [hmul] at hover
            simp only at hover
            rw [hpower] at hover
            have hpowLe : 3 ^ (step.val + 1) ≤ 3 ^ 40 := by
              exact Nat.pow_le_pow_right (by norm_num) (Nat.succ_le_iff.mpr hstepLt)
            have hmax : 3 ^ 40 ≤ Aeneas.Std.U64.max := by
              rw [Aeneas.Std.U64.max_eq]
              norm_num
            norm_num at hover
            rw [pow_succ] at hpowLe
            omega
          | some next =>
            have hnext := Aeneas.Std.U64.checked_mul_bv_spec power 3#u64
            rw [hmul] at hnext
            simp only at hnext
            simp only
            have haddBound : step.val + 1 ≤ Aeneas.Std.U32.max := by
              rw [Aeneas.Std.U32.max_eq]
              omega
            step
            scalar_tac
        · rw [if_neg hlt]
          have hstepEq : step.val = 40 := by
            have hnot : ¬step.val < 40 := by
              simpa [apeiron_word.coord.address.H3_40.K] using hlt
            omega
          have hpowerEq : power = apeiron_word.coord.address.H3_40.Q := by
            apply Aeneas.Std.UScalar.eq_of_val_eq
            rw [hpower, hstepEq]
            simp [apeiron_word.coord.address.H3_40.Q]
          subst power
          simp only [apeiron_word.coord.address.H3_40.Q,
            apeiron_word.coord.address.H3_40.P,
            apeiron_word.coord.address.H3_40.R,
            apeiron_word.coord.address.H3_40.S,
            apeiron_word.coord.address.H3_40.INVERSE_OF_TWO]
          rw [if_neg (by simp)]
          step
          have hi : i.val = 9223372036854775808 := by
            rw [i_post]
            rw [Nat.shiftLeft_eq]
            norm_num [Aeneas.Std.U64.size, Aeneas.Std.U64.numBits]
          rw [if_neg]
          · step
            cases o with
            | some value =>
              simp only at o_post
              rw [Aeneas.Std.U64.max_eq] at o_post
              norm_num at o_post
            | none =>
              simp only [Option.isSome, Bool.false_eq_true, if_false]
              step
              have hi1 : i1 = 1#u64 := by
                apply Aeneas.Std.UScalar.eq_of_val_eq
                rw [i1_post]
                norm_num
              subst i1
              rw [if_neg (by simp)]
              simp only [Aeneas.Std.lift, Aeneas.Std.bind_tc_ok]
              step
              step
              have hi5 : i5 = 1#u128 := by
                apply Aeneas.Std.UScalar.eq_of_val_eq
                rw [i5_post, i4_post]
                simp only [Aeneas.Std.core.convert.num.FromU128U64.from_val_eq]
                norm_num
              subst i5
              rw [if_neg (by simp)]
              step
              step
              have hi8 : i8 = 1#u128 := by
                apply Aeneas.Std.UScalar.eq_of_val_eq
                rw [i8_post, i7_post]
                simp only [Aeneas.Std.core.convert.num.FromU128U64.from_val_eq]
                norm_num
              subst i8
              rw [if_neg (by simp)]
              step
              step
              have hi12 : i12 = 1#u128 := by
                apply Aeneas.Std.UScalar.eq_of_val_eq
                rw [i12_post, i11_post]
                simp only [Aeneas.Std.core.convert.num.FromU128U64.from_val_eq]
                norm_num
              subst i12
              simp
          · change ¬12157665459056928801 ≤ i.val
            rw [hi]
            norm_num
      · exact ⟨by norm_num, by norm_num⟩
    obtain ⟨result, hresult, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hs
    rw [hpost] at hresult
    exact hresult
  refine ⟨{ _sealed := () }, ?_⟩
  have hvalidate :
      apeiron_word.coord.address.H3_40.validate
          apeiron_word.coord.address.H3_40.VERSION
          apeiron_word.coord.address.H3_40.P
          apeiron_word.coord.address.H3_40.K
          apeiron_word.coord.address.H3_40.Q
          apeiron_word.coord.address.H3_40.R
          apeiron_word.coord.address.H3_40.S
          apeiron_word.coord.address.H3_40.INVERSE_OF_TWO = ok true := by
    unfold apeiron_word.coord.address.H3_40.validate
    rw [if_neg (by simp), if_neg (by simp), if_neg (by simp)]
    exact hloop
  simp [apeiron_word.coord.address.H3_40.new, hvalidate]

/-- The generated default face-home delegation from a validated generated profile. -/
def wordFaceAddress (profile : apeiron_word.coord.address.H3_40)
    (face : apeiron_word.cell.Sigma) (key : Aeneas.Std.I64) :
    Aeneas.Std.Result apeiron_word.coord.address.FaceTableAddress :=
  apeiron_word.coord.address.FaceHome.address.default
    apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressFaceHomeI64Sigma () profile face key

/-- The scalar hyperbolic address embeds as the canonical cell `(a, 0)`. -/
def hyperbolicScalarCell (residue : Aeneas.Std.U64) :
    Apeiron.Trichotomy.AnyRing.RingCell Apeiron.NativeAddress.H3_40 :=
  (((residue.val : Nat) : Apeiron.NativeAddress.H3_40), 0)

/-- The scalar embedding uses canonical `ringMul` and `ringNormForm`; it makes no encoding-homomorphism claim. -/
theorem hyperbolic_scalar_denote (residue : Aeneas.Std.U64) :
    Apeiron.Trichotomy.AnyRing.ringMul (1 : Apeiron.NativeAddress.H3_40)
        (hyperbolicScalarCell residue) (1, 0) = hyperbolicScalarCell residue ∧
      Apeiron.Trichotomy.AnyRing.ringNormForm (1 : Apeiron.NativeAddress.H3_40)
        (hyperbolicScalarCell residue) = ((residue.val : Nat) : Apeiron.NativeAddress.H3_40) ^ 2 := by
  simp [hyperbolicScalarCell]

/-- The real default delegation denotes the generated elliptic, parabolic, and hyperbolic bodies. -/
theorem home_face_denote (profile : apeiron_word.coord.address.H3_40)
    (_hprofile : apeiron_word.coord.address.H3_40.new = Aeneas.Std.Result.ok (some profile))
    (key : Aeneas.Std.I64) :
    wordFaceAddress profile .Elliptic key =
      (do
        let packed ←
          apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressEllipticPackingI64.elliptic_pack
            () key
        apeiron_word.coord.address.FaceTableAddress.new (.EllipticWord packed)) ∧
    wordFaceAddress profile .Parabolic key =
      (do
        let dual ←
          apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressFaceResidueCapabilityI64.parabolic_dual
            () key
        let residue ← apeiron_word.coord.address.ParabolicDual.residue dual
        apeiron_word.coord.address.FaceTableAddress.new (.ParabolicWord residue)) ∧
    wordFaceAddress profile .Hyperbolic key =
      (do
        let residue ←
          apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressFaceResidueCapabilityI64.hyperbolic_scalar
            () profile key
        apeiron_word.coord.address.FaceTableAddress.new (.HyperbolicResidue profile residue)) := by
  simp [wordFaceAddress, apeiron_word.coord.address.FaceHome.address.default,
    apeiron_word.cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.encode,
    apeiron_word.coord.address.LocatorSigma.token_for,
    apeiron_word.coord.address.LocatorSigma.Insts.Apeiron_wordCoordAddressFaceDomain.encode]

/-- One generated constructor witness supplies all three generated face-dispatch equations. -/
theorem home_face_joint_witness :
    ∃ profile : apeiron_word.coord.address.H3_40,
      apeiron_word.coord.address.H3_40.new = Aeneas.Std.Result.ok (some profile) ∧
      ∀ key : Aeneas.Std.I64,
        wordFaceAddress profile .Elliptic key =
            (do
              let packed ←
                apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressEllipticPackingI64.elliptic_pack
                  () key
              apeiron_word.coord.address.FaceTableAddress.new (.EllipticWord packed)) ∧
          wordFaceAddress profile .Parabolic key =
            (do
              let dual ←
                apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressFaceResidueCapabilityI64.parabolic_dual
                  () key
              let residue ← apeiron_word.coord.address.ParabolicDual.residue dual
              apeiron_word.coord.address.FaceTableAddress.new (.ParabolicWord residue)) ∧
          wordFaceAddress profile .Hyperbolic key =
            (do
              let residue ←
                apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressFaceResidueCapabilityI64.hyperbolic_scalar
                  () profile key
              apeiron_word.coord.address.FaceTableAddress.new
                (.HyperbolicResidue profile residue)) := by
  obtain ⟨profile, hprofile⟩ := h3_40_new_ok
  exact ⟨profile, hprofile, fun key => home_face_denote profile hprofile key⟩


/-- The carried-key face calls the real face-home address on its retained key. -/
def storedFaceAddressHeld (profile : apeiron_word.coord.address.H3_40)
    (face : apeiron_word.cell.Sigma) (held : Apeiron.GrowableIdentity.StoredI64) :
    Aeneas.Std.Result apeiron_word.coord.address.FaceTableAddress :=
  apeiron_word.table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.face_address_held
    Apeiron.GrowableIdentity.wordExactIdentityInst Aeneas.Std.core.clone.CloneI64
    Apeiron.GrowableIdentity.wordHomeInst
    apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressFaceHomeI64Sigma
    () profile face held

/-- The regenerating face decodes its held identity before calling the real face-home address. -/
def regenFaceAddressHeld (profile : apeiron_word.coord.address.H3_40)
    (face : apeiron_word.cell.Sigma) (held : Apeiron.GrowableIdentity.RegenI64) :
    Aeneas.Std.Result apeiron_word.coord.address.FaceTableAddress :=
  apeiron_word.table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.face_address_held
    Apeiron.GrowableIdentity.wordExactIdentityInst Apeiron.GrowableIdentity.wordHomeInst
    Apeiron.GrowableIdentity.wordCodecInst
    apeiron_word.table.WordHome.Insts.Apeiron_wordCoordAddressFaceHomeI64Sigma
    () profile face held

/-- Held stored and regenerating addresses agree with the query address on every admitted face. -/
theorem held_address_agrees (profile : apeiron_word.coord.address.H3_40)
    (face : apeiron_word.cell.Sigma) (key : Aeneas.Std.I64)
    (heldStored : Apeiron.GrowableIdentity.StoredI64)
    (heldRegen : Apeiron.GrowableIdentity.RegenI64)
    (hstored : heldStored = { key := key })
    (hregen : apeiron_word.word_regen_hold face key = Aeneas.Std.Result.ok heldRegen) :
    storedFaceAddressHeld profile face heldStored = wordFaceAddress profile face key ∧
      regenFaceAddressHeld profile face heldRegen = wordFaceAddress profile face key := by
  subst heldStored
  constructor
  · rfl
  · have hdecode := Apeiron.GrowableIdentity.regen_hold_decodes face key
    rw [hregen] at hdecode
    simp only [Aeneas.Std.WP.spec_ok] at hdecode
    unfold regenFaceAddressHeld
    unfold apeiron_word.table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.face_address_held
    dsimp only [Apeiron.GrowableIdentity.wordCodecInst,
      apeiron_word.table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces]
    simp only [apeiron_word.cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma,
      Aeneas.Std.bind_tc_ok]
    change (do
      let decoded ← Apeiron.GrowableIdentity.decodeRegen face heldRegen
      wordFaceAddress profile face decoded) = wordFaceAddress profile face key
    rw [hdecode]
    simp

/-- One generated profile and one common key produce agreeing stored and regenerating addresses. -/
theorem held_address_joint_witness :
    ∃ profile key heldRegen,
      apeiron_word.coord.address.H3_40.new = Aeneas.Std.Result.ok (some profile) ∧
      apeiron_word.word_regen_hold apeiron_word.cell.Sigma.Parabolic key =
        Aeneas.Std.Result.ok heldRegen ∧
      storedFaceAddressHeld profile apeiron_word.cell.Sigma.Parabolic { key := key } =
        wordFaceAddress profile apeiron_word.cell.Sigma.Parabolic key ∧
      regenFaceAddressHeld profile apeiron_word.cell.Sigma.Parabolic heldRegen =
        wordFaceAddress profile apeiron_word.cell.Sigma.Parabolic key := by
  obtain ⟨profile, hprofile⟩ := h3_40_new_ok
  obtain ⟨heldRegen, hhold, _⟩ := Aeneas.Std.WP.spec_imp_exists
    (Apeiron.GrowableIdentity.regen_hold_decodes apeiron_word.cell.Sigma.Parabolic 0#i64)
  exact ⟨profile, 0#i64, heldRegen, hprofile, hhold,
    (held_address_agrees profile apeiron_word.cell.Sigma.Parabolic 0#i64 { key := 0#i64 }
      heldRegen rfl hhold).1,
    (held_address_agrees profile apeiron_word.cell.Sigma.Parabolic 0#i64 { key := 0#i64 }
      heldRegen rfl hhold).2⟩

end Native.Address

/-- A many-to-one residue cannot be reused as an equality-faithful identity encoding. -/
theorem manyToOne_residue_not_exactIdentity
    {Key Residue Retained : Type*} (identity : ExactIdentityContract Key Retained)
    (residue : Key → Residue) (encodeResidue : Residue → List UInt8) (left right : Key)
    (distinct : left ≠ right) (sameResidue : residue left = residue right)
    (decodeAsIdentity : ∀ key, identity.bytes key = encodeResidue (residue key)) : False := by
  apply distinct
  rw [← identity.bytes_exact]
  rw [decodeAsIdentity left, decodeAsIdentity right, sameResidue]

/-- The population-complete `(K,H,R)` decision required before table wiring. -/
inductive TableAdmissionCombination where
  | storedExactIdentityFaceHome
  | regenI64WordCodecFaceHome
  | byteKeysFiniteEncodingFaceHome
  | byteKeysWithoutEllipticPackingSigmaHome
  | arcDelegatedIdentityFaceHome
  | opaqueEqArbitraryHome
  | placementClosureHome
  | regenDecodeFaceResidue
  deriving DecidableEq

def tableCombinationAdmitted : TableAdmissionCombination → Bool
  | .storedExactIdentityFaceHome => true
  | .regenI64WordCodecFaceHome => true
  -- This row denotes the explicit `LocatorSigma` byte-key domain, not all-face packing.
  | .byteKeysFiniteEncodingFaceHome => true
  | .byteKeysWithoutEllipticPackingSigmaHome => false
  | .arcDelegatedIdentityFaceHome => true
  | .opaqueEqArbitraryHome => false
  | .placementClosureHome => false
  | .regenDecodeFaceResidue => false

theorem table_admission_map_exact :
    tableCombinationAdmitted .storedExactIdentityFaceHome = true ∧
      tableCombinationAdmitted .regenI64WordCodecFaceHome = true ∧
      tableCombinationAdmitted .byteKeysFiniteEncodingFaceHome = true ∧
      tableCombinationAdmitted .byteKeysWithoutEllipticPackingSigmaHome = false ∧
      tableCombinationAdmitted .arcDelegatedIdentityFaceHome = true ∧
      tableCombinationAdmitted .opaqueEqArbitraryHome = false ∧
      tableCombinationAdmitted .placementClosureHome = false ∧
      tableCombinationAdmitted .regenDecodeFaceResidue = false := by
  decide

/-- The lead must select one migration; this contract does not silently select either one. -/
inductive ApiMigrationOption where
  | explicitCapabilityApi
  | preserveEntireCurrentApi
  deriving DecidableEq

inductive ApiMigrationObligation where
  | capabilityOnEveryLocatorOperation
  | compatibilityFallbackPreservesAllCurrentPopulations
  deriving DecidableEq

def migrationObligation : ApiMigrationOption → ApiMigrationObligation
  | .explicitCapabilityApi => .capabilityOnEveryLocatorOperation
  | .preserveEntireCurrentApi => .compatibilityFallbackPreservesAllCurrentPopulations

theorem migration_options_exact :
    migrationObligation .explicitCapabilityApi = .capabilityOnEveryLocatorOperation ∧
      migrationObligation .preserveEntireCurrentApi =
        .compatibilityFallbackPreservesAllCurrentPopulations := by
  decide

inductive ApiMigrationDecision where
  | explicitCapabilityLive
  deriving DecidableEq

def apiMigrationDecision : ApiMigrationDecision := .explicitCapabilityLive

/--
The headline name is retained so an established audit entry is not deleted silently. Its old
`pendingLead` statement is stale: the selected explicit-capability migration now executes.
-/
theorem api_migration_is_pending : apiMigrationDecision = .explicitCapabilityLive := rfl

/-! ## Source-linked table coverage and cost obligations -/

/-- Absence stated solely through the table's authoritative `entryAt` observation. -/
def EntryAtAbsent {State Representative Sigma Key Value : Type}
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (state : State) (query : Sigma × Key) : Prop :=
  ¬ ∃ representative value, entryAt state representative = some (query, value)

theorem entryAtAbsent_iff_noLive
    {State Representative Sigma Key Value : Type}
    {entryAt : State → Representative → Option ((Sigma × Key) × Value)}
    {state : State} {query : Sigma × Key} :
    EntryAtAbsent entryAt state query ↔ ¬ ∃ representative, Live entryAt state representative query := by
  simp [EntryAtAbsent, Live]

/--
The source adapter is parameterized by the generated table observation. Its coverage field is
therefore coverage of table storage, not coverage of a standalone locator-owned value store.
-/
structure TableEntryAtCoverageAdapter {State Representative Sigma Key Value : Type}
    (directory : ScalarArenaDirectory State)
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (completePath : State → Sigma × Key → Representative → Prop)
    (reverseLink : State → Representative → Sigma × Key → Prop) (state : State) : Prop where
  storage : TableStorageCoverage directory entryAt completePath reverseLink state
  absence : ∀ query, EntryAtAbsent entryAt state query ↔
    ¬ ∃ representative, Live entryAt state representative query

theorem tableEntryAt_absent_of_missing
    {State Representative Sigma Key Value : Type}
    {entryAt : State → Representative → Option ((Sigma × Key) × Value)}
    {completePath : State → Sigma × Key → Representative → Prop}
    {reverseLink : State → Representative → Sigma × Key → Prop}
    {directory : ScalarArenaDirectory State}
    {state : State} {query : Sigma × Key}
    (adapter : TableEntryAtCoverageAdapter directory entryAt completePath reverseLink state)
    (missing : ¬ ∃ representative, completePath state query representative) :
    EntryAtAbsent entryAt state query := by
  rw [adapter.absence]
  exact table_absent_of_complete_path_missing adapter.storage missing

/-- Every production transition that can create, move, or assemble a representative is explicit. -/
structure SourceCoverageObligations {State Representative Sigma Key Value : Type}
    (directory : ScalarArenaDirectory State)
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (completePath : State → Sigma × Key → Representative → Prop)
    (reverseLink : State → Representative → Sigma × Key → Prop)
    (empty : State) (serialInsert parallelInsert growth relocation merge : State → State → Prop)
    (build : State → Prop) (link : State → ArenaHandle → ArenaHandle → State → Prop) where
  scalarTraversal : ScalarTraversalContract (Query := Sigma × Key) directory
  scalarLinking : ScalarLinkingPreservation directory link
  constructor : TableStorageCoverage directory entryAt completePath reverseLink empty
  serialInsertion : PreservesTableCoverage directory entryAt completePath reverseLink serialInsert
  parallelInsertion : PreservesTableCoverage directory entryAt completePath reverseLink parallelInsert
  buildResult : ∀ state, build state →
    TableStorageCoverage directory entryAt completePath reverseLink state
  growthPreservation : PreservesTableCoverage directory entryAt completePath reverseLink growth
  relocationPreservation : PreservesTableCoverage directory entryAt completePath reverseLink relocation
  mergePreservation : PreservesTableCoverage directory entryAt completePath reverseLink merge

/-- Bytes and comparisons are charged separately; an `Unknown` sweep is never hidden. -/
structure SourceWorkCharge where
  residueSteps : Nat
  identityBytesInspected : Nat
  identityBytesCopied : Nat
  edgeLabelComparisons : Nat
  fullKeyComparisons : Nat
  handleWrites : Nat
  reverseLinkWrites : Nat
  unknownSweepComparisons : Nat

def SourceWorkCharge.bytes (charge : SourceWorkCharge) : Nat :=
  charge.identityBytesInspected + charge.identityBytesCopied

def SourceWorkCharge.comparisons (charge : SourceWorkCharge) : Nat :=
  charge.edgeLabelComparisons + charge.fullKeyComparisons + charge.unknownSweepComparisons

/-- Exact source counters required before a population-independent lookup claim. -/
structure SourceLinkedCostObligation (State Query : Type*)
    (face : Query → Face) (identityBytes : Query → Nat)
    (liveCount : State → Nat) (sourceWork : State → Query → WorkCounts)
    (charge : State → Query → SourceWorkCharge) : Prop where
  parabolicDepth : ∀ state query, face query = .parabolic →
    (charge state query).residueSteps ≤ 64
  hyperbolicDepth : ∀ state query, face query = .hyperbolic →
    (charge state query).residueSteps ≤ 40
  identityFanoutCharged : ∀ state query,
    (charge state query).edgeLabelComparisons ≤ 256 * identityBytes query
  sourceBytesInspected : ∀ state query,
    (sourceWork state query).bytesInspected = (charge state query).identityBytesInspected
  sourceBytesCopied : ∀ state query,
    (sourceWork state query).bytesCopied = (charge state query).identityBytesCopied
  sourceComparisons : ∀ state query,
    (sourceWork state query).comparisons =
      (charge state query).edgeLabelComparisons + (charge state query).fullKeyComparisons +
        (charge state query).unknownSweepComparisons
  bytesCharged : ∀ state query,
    (charge state query).bytes =
      (charge state query).identityBytesInspected + (charge state query).identityBytesCopied
  comparisonsCharged : ∀ state query,
    (charge state query).comparisons =
      (charge state query).edgeLabelComparisons + (charge state query).fullKeyComparisons +
        (charge state query).unknownSweepComparisons
  unknownSweepCharged : ∀ state query,
    (charge state query).unknownSweepComparisons ≤ liveCount state
  populationIndependentLookup : ∃ constant, ∀ state query,
    (charge state query).comparisons ≤ constant * (identityBytes query + 1)

inductive CurrentSourceBlocker where
  | identityEdgeReserveAfterRetain
  | unknownThenPopulationSweep
  | scalarArenaTranslationUnproved
  deriving DecidableEq

/-!
`reserveBeforeRetainSatisfiedByCurrentLocator` describes the superseded nested-arena source. It is
stale for the selected scalar-arena representation; no source-linked scalar-arena receipt exists.
-/
def reserveBeforeRetainSatisfiedByCurrentLocator : Bool := true
def scalarArenaReserveBeforeRetainSourceLinked : Bool := false
def populationIndependentLookupEstablishedByUnknownSweep : Bool := false

/-- The old reserve result is retained plainly as stale; scalar source binding and cost remain open. -/
theorem current_source_blockers_exact :
    reserveBeforeRetainSatisfiedByCurrentLocator = true ∧
      scalarArenaReserveBeforeRetainSourceLinked = false ∧
      populationIndependentLookupEstablishedByUnknownSweep = false := by
  decide

/-! ## Named extraction roots for the wiring task -/

structure ExtractionTarget where
  sourceModule : String
  productionFunction : String
  wrapperRoot : String
  deriving DecidableEq

def requiredExtractionTargets : List ExtractionTarget := [
  ⟨"src/coord/address.rs", "<WordFaceHome as FaceCoord<i64>>::coord", "word_face_address"⟩,
  ⟨"src/coord/locator_core.rs", "Directory::decide", "word_stored_locator_decide"⟩,
  ⟨"src/coord/locator_core.rs", "Directory::decide", "word_regen_locator_decide"⟩,
  ⟨"src/coord/locator_core.rs", "Directory::publish", "word_stored_locator_publish"⟩,
  ⟨"src/coord/locator_core.rs", "Directory::publish", "word_regen_locator_publish"⟩,
  ⟨"src/coord/locator_core.rs", "Directory::relink", "word_stored_locator_relink"⟩,
  ⟨"src/coord/locator_core.rs", "Directory::relink", "word_regen_locator_relink"⟩,
  ⟨"src/coord/locator_core.rs", "Directory::entry", "word_stored_locator_entry"⟩,
  ⟨"src/coord/locator_core.rs", "Directory::entry", "word_regen_locator_entry"⟩,
  ⟨"src/table.rs", "Table::get_at", "word_stored_table_get_at"⟩,
  ⟨"src/table.rs", "Table::get_at", "word_regen_table_get_at"⟩,
  ⟨"src/table.rs", "get_prepared_shard", "word_stored_prepared_shard_get"⟩,
  ⟨"src/table.rs", "get_prepared_shard", "word_regen_prepared_shard_get"⟩,
  ⟨"src/table.rs", "Table::apply_prepared_shard_at", "word_stored_prepared_shard_apply"⟩,
  ⟨"src/table.rs", "Table::apply_prepared_shard_at", "word_regen_prepared_shard_apply"⟩,
  ⟨"src/table.rs", "Table::apply_insert", "word_stored_apply_insert"⟩,
  ⟨"src/table.rs", "Table::apply_insert", "word_regen_apply_insert"⟩,
  ⟨"src/table.rs", "Table::relocate_entry", "word_stored_relocate_entry"⟩,
  ⟨"src/table.rs", "Table::relocate_entry", "word_regen_relocate_entry"⟩,
  ⟨"src/table.rs", "Table::relocate_all", "word_stored_relocate_all"⟩,
  ⟨"src/table.rs", "Table::relocate_all", "word_regen_relocate_all"⟩,
  ⟨"src/table.rs", "Table::apply_pending_and_commit", "word_stored_apply_pending_and_commit"⟩,
  ⟨"src/table.rs", "Table::apply_pending_and_commit", "word_regen_apply_pending_and_commit"⟩,
  ⟨"src/table.rs", "Table::build_at", "word_stored_table_build_at"⟩,
  ⟨"src/table.rs", "Table::build_at", "word_regen_table_build_at"⟩,
  ⟨"src/table/resources.rs", "publish_prepared", "word_stored_publish_prepared"⟩,
  ⟨"src/table/resources.rs", "publish_prepared", "word_regen_publish_prepared"⟩,
  ⟨"src/table/resources.rs", "capacity_bound_holds", "word_capacity_bound_holds"⟩,
  ⟨"src/table/parallel.rs", "Table::build_at_parallel", "word_stored_build_at_parallel"⟩,
  ⟨"src/table/parallel.rs", "Table::build_at_parallel", "word_regen_build_at_parallel"⟩,
  ⟨"src/table/parallel.rs", "Table::get_each_at_parallel", "word_stored_get_each_at_parallel"⟩,
  ⟨"src/table/parallel.rs", "Table::get_each_at_parallel", "word_regen_get_each_at_parallel"⟩]

theorem extraction_targets_exact : requiredExtractionTargets = [
    ⟨"src/coord/address.rs", "<WordFaceHome as FaceCoord<i64>>::coord", "word_face_address"⟩,
    ⟨"src/coord/locator_core.rs", "Directory::decide", "word_stored_locator_decide"⟩,
    ⟨"src/coord/locator_core.rs", "Directory::decide", "word_regen_locator_decide"⟩,
    ⟨"src/coord/locator_core.rs", "Directory::publish", "word_stored_locator_publish"⟩,
    ⟨"src/coord/locator_core.rs", "Directory::publish", "word_regen_locator_publish"⟩,
    ⟨"src/coord/locator_core.rs", "Directory::relink", "word_stored_locator_relink"⟩,
    ⟨"src/coord/locator_core.rs", "Directory::relink", "word_regen_locator_relink"⟩,
    ⟨"src/coord/locator_core.rs", "Directory::entry", "word_stored_locator_entry"⟩,
    ⟨"src/coord/locator_core.rs", "Directory::entry", "word_regen_locator_entry"⟩,
    ⟨"src/table.rs", "Table::get_at", "word_stored_table_get_at"⟩,
    ⟨"src/table.rs", "Table::get_at", "word_regen_table_get_at"⟩,
    ⟨"src/table.rs", "get_prepared_shard", "word_stored_prepared_shard_get"⟩,
    ⟨"src/table.rs", "get_prepared_shard", "word_regen_prepared_shard_get"⟩,
    ⟨"src/table.rs", "Table::apply_prepared_shard_at", "word_stored_prepared_shard_apply"⟩,
    ⟨"src/table.rs", "Table::apply_prepared_shard_at", "word_regen_prepared_shard_apply"⟩,
    ⟨"src/table.rs", "Table::apply_insert", "word_stored_apply_insert"⟩,
    ⟨"src/table.rs", "Table::apply_insert", "word_regen_apply_insert"⟩,
    ⟨"src/table.rs", "Table::relocate_entry", "word_stored_relocate_entry"⟩,
    ⟨"src/table.rs", "Table::relocate_entry", "word_regen_relocate_entry"⟩,
    ⟨"src/table.rs", "Table::relocate_all", "word_stored_relocate_all"⟩,
    ⟨"src/table.rs", "Table::relocate_all", "word_regen_relocate_all"⟩,
    ⟨"src/table.rs", "Table::apply_pending_and_commit", "word_stored_apply_pending_and_commit"⟩,
    ⟨"src/table.rs", "Table::apply_pending_and_commit", "word_regen_apply_pending_and_commit"⟩,
    ⟨"src/table.rs", "Table::build_at", "word_stored_table_build_at"⟩,
    ⟨"src/table.rs", "Table::build_at", "word_regen_table_build_at"⟩,
    ⟨"src/table/resources.rs", "publish_prepared", "word_stored_publish_prepared"⟩,
    ⟨"src/table/resources.rs", "publish_prepared", "word_regen_publish_prepared"⟩,
    ⟨"src/table/resources.rs", "capacity_bound_holds", "word_capacity_bound_holds"⟩,
    ⟨"src/table/parallel.rs", "Table::build_at_parallel", "word_stored_build_at_parallel"⟩,
    ⟨"src/table/parallel.rs", "Table::build_at_parallel", "word_regen_build_at_parallel"⟩,
    ⟨"src/table/parallel.rs", "Table::get_each_at_parallel", "word_stored_get_each_at_parallel"⟩,
    ⟨"src/table/parallel.rs", "Table::get_each_at_parallel", "word_regen_get_each_at_parallel"⟩] := rfl

inductive StandaloneIdentityPopulation where
  | i8Word
  | u8Word
  | i32Word
  | u32Word
  | i64Word
  | u64Word
  | byteVector
  | arcDelegatingExactIdentity
  | opaqueEq
  deriving DecidableEq

def standaloneLocatorHasExactIdentity : StandaloneIdentityPopulation → Bool
  | .i8Word | .u8Word | .i32Word | .u32Word | .i64Word | .u64Word | .byteVector
  | .arcDelegatingExactIdentity => true
  | .opaqueEq => false

inductive HomePopulation where
  | faceAwareCertified
  | blanketU64Closure
  deriving DecidableEq

def homeAdmitted : HomePopulation → Bool
  | .faceAwareCertified => true
  | .blanketU64Closure => false

/-- Carrier admission keeps identity independent of a many-to-one residue. -/
inductive CarrierPopulation where
  | storedWithExactIdentity
  | regenI64WordCodec
  | regenDecodeResidue
  deriving DecidableEq

def carrierAdmitted : CarrierPopulation → Bool
  | .storedWithExactIdentity => true
  | .regenI64WordCodec => true
  | .regenDecodeResidue => false

theorem admission_boundary :
    standaloneLocatorHasExactIdentity .i8Word = true ∧
      standaloneLocatorHasExactIdentity .u8Word = true ∧
      standaloneLocatorHasExactIdentity .i32Word = true ∧
      standaloneLocatorHasExactIdentity .u32Word = true ∧
      standaloneLocatorHasExactIdentity .i64Word = true ∧
      standaloneLocatorHasExactIdentity .u64Word = true ∧
      standaloneLocatorHasExactIdentity .byteVector = true ∧
      standaloneLocatorHasExactIdentity .arcDelegatingExactIdentity = true ∧
      standaloneLocatorHasExactIdentity .opaqueEq = false ∧
      homeAdmitted .faceAwareCertified = true ∧
      homeAdmitted .blanketU64Closure = false ∧
      carrierAdmitted .storedWithExactIdentity = true ∧
      carrierAdmitted .regenI64WordCodec = true ∧
      carrierAdmitted .regenDecodeResidue = false := by
  decide

inductive CommitPhase where
  | checkedSizing
  | reserveAll
  | observeCapacities
  | retainIdentity
  | stagePrivate
  | publishLinks
  | mergeOnce
  | commit
  deriving DecidableEq

/-- Physical capacities are observed in records and scalar words, with checked stride conversion. -/
structure ScalarArenaCapacity where
  residueRecords : Nat
  identityRecords : Nat
  siblingEdgeRecords : Nat
  residueWords : Nat
  identityWords : Nat
  siblingEdgeWords : Nat
  residueStride_exact : residueWords = residueStride * residueRecords
  identityStride_exact : identityWords = identityStride * identityRecords
  siblingEdgeStride_exact : siblingEdgeWords = siblingEdgeStride * siblingEdgeRecords

/--
The successor receipt retains eight logical reservation categories. The two residue-face counts feed
one flat residue arena; identity nodes and sorted sibling edges feed the other two arenas.
-/
structure LocatorReservationReceipt where
  parabolicResidueRecords : Nat
  hyperbolicResidueRecords : Nat
  identityRecords : Nat
  siblingEdgeRecords : Nat
  representatives : Nat
  reverseLinks : Nat
  stagedPathEntries : Nat
  retainedIdentityBytes : Nat
  committed : ScalarArenaCapacity
  observed : ScalarArenaCapacity
  observedResidueSufficient :
    committed.residueRecords + parabolicResidueRecords + hyperbolicResidueRecords ≤
      observed.residueRecords
  observedIdentitySufficient :
    committed.identityRecords + identityRecords ≤ observed.identityRecords
  observedSiblingEdgesSufficient :
    committed.siblingEdgeRecords + siblingEdgeRecords ≤ observed.siblingEdgeRecords

/--
The only fallible allocation phase is `reserveAll`. Actual scalar-arena capacities are observed
after reservation and before retention. Staging is private; failure may change raw Vec capacities,
but it restores staged logical lengths and stutters on the represented committed state.
-/
structure ReserveBeforeRetainContract (State Committed Scratch : Type*)
    (Coverage : State → Prop) where
  successOrder : List CommitPhase
  successOrder_exact : successOrder = [CommitPhase.checkedSizing, CommitPhase.reserveAll,
    CommitPhase.observeCapacities, CommitPhase.retainIdentity, CommitPhase.stagePrivate,
    CommitPhase.publishLinks, CommitPhase.mergeOnce, CommitPhase.commit]
  allocationPhase : CommitPhase → Bool
  allocationPhase_exact : ∀ phase, allocationPhase phase = true ↔ phase = .reserveAll
  receipt : State → LocatorReservationReceipt
  observeStorage : State → ScalarArenaCapacity
  receipt_observesStorage : ∀ state, (receipt state).observed = observeStorage state
  committedState : State → Committed
  privateScratch : State → Scratch
  scratchReleased : Scratch → Prop
  stagedLogicalLengthsRestored : State → State → Prop
  forcedReservationFailure : State → State → Prop
  forcedFailure_stutters : ∀ before after,
    forcedReservationFailure before after → committedState after = committedState before
  forcedFailure_restoresStaging : ∀ before after,
    forcedReservationFailure before after → stagedLogicalLengthsRestored before after
  forcedFailure_releasesScratch : ∀ before after,
    forcedReservationFailure before after → scratchReleased (privateScratch after)
  forcedFailure_preservesCoverage : ∀ before after,
    Coverage before → forcedReservationFailure before after → Coverage after
  rootsMaterializedBeforeEffects : ∀ state,
    2 ≤ (receipt state).observed.residueRecords
  seam : String
  seam_exact : seam = "reservation_failure::arm"

/--
Joint non-vacuity obligation for empty construction, one successful transition, and a late
reservation failure. Successful output coverage is derived from preservation, not stored as an
assumed postcondition.
-/
structure ScalarArenaJointWitnesses
    {State Representative Sigma Key Value Committed Scratch : Type}
    (directory : ScalarArenaDirectory State)
    (entryAt : State → Representative → Option ((Sigma × Key) × Value))
    (completePath : State → Sigma × Key → Representative → Prop)
    (reverseLink : State → Representative → Sigma × Key → Prop)
    (successStep : State → State → Prop) where
  emptyState : State
  emptyCoverage : TableStorageCoverage directory entryAt completePath reverseLink emptyState
  successBefore : State
  successAfter : State
  successInputCoverage :
    TableStorageCoverage directory entryAt completePath reverseLink successBefore
  successTransition : successStep successBefore successAfter
  successPreserves :
    PreservesTableCoverage directory entryAt completePath reverseLink successStep
  lateFailureBefore : State
  lateFailureAfter : State
  lateFailureInputCoverage :
    TableStorageCoverage directory entryAt completePath reverseLink lateFailureBefore
  resourceContract : ReserveBeforeRetainContract State Committed Scratch
    (fun state => TableStorageCoverage directory entryAt completePath reverseLink state)
  lateFailure : resourceContract.forcedReservationFailure lateFailureBefore lateFailureAfter

theorem scalarArena_joint_success_coverage
    {State Representative Sigma Key Value Committed Scratch : Type}
    {directory : ScalarArenaDirectory State}
    {entryAt : State → Representative → Option ((Sigma × Key) × Value)}
    {completePath : State → Sigma × Key → Representative → Prop}
    {reverseLink : State → Representative → Sigma × Key → Prop}
    {successStep : State → State → Prop}
    (witnesses : ScalarArenaJointWitnesses (Committed := Committed) (Scratch := Scratch)
      directory entryAt completePath reverseLink successStep) :
    TableStorageCoverage directory entryAt completePath reverseLink witnesses.successAfter :=
  witnesses.successPreserves witnesses.successBefore witnesses.successAfter
    witnesses.successInputCoverage witnesses.successTransition

theorem scalarArena_joint_failure_committed_stutter
    {State Representative Sigma Key Value Committed Scratch : Type}
    {directory : ScalarArenaDirectory State}
    {entryAt : State → Representative → Option ((Sigma × Key) × Value)}
    {completePath : State → Sigma × Key → Representative → Prop}
    {reverseLink : State → Representative → Sigma × Key → Prop}
    {successStep : State → State → Prop}
    (witnesses : ScalarArenaJointWitnesses (Committed := Committed) (Scratch := Scratch)
      directory entryAt completePath reverseLink successStep) :
    witnesses.resourceContract.committedState witnesses.lateFailureAfter =
      witnesses.resourceContract.committedState witnesses.lateFailureBefore :=
  witnesses.resourceContract.forcedFailure_stutters witnesses.lateFailureBefore
    witnesses.lateFailureAfter witnesses.lateFailure

/-! ## Integration replacement map -/

inductive SummarySite where
  | writeProbe
  | readProbe
  | relocationProbe
  | reservationReceipt
  deriving DecidableEq

inductive SummaryReplacement where
  | exactAddressCoverage
  | residueAbsentOrUnknown
  | locatorPathExtension
  | locatorRelink
  | noSummaryReceipt
  | locatorCapacityReceipt
  deriving DecidableEq

def summaryReplacement : Face → SummarySite → SummaryReplacement
  | .elliptic, .writeProbe => .exactAddressCoverage
  | .elliptic, .readProbe => .exactAddressCoverage
  | .elliptic, .relocationProbe => .exactAddressCoverage
  | .elliptic, .reservationReceipt => .noSummaryReceipt
  | .parabolic, .writeProbe => .locatorPathExtension
  | .parabolic, .readProbe => .residueAbsentOrUnknown
  | .parabolic, .relocationProbe => .locatorRelink
  | .parabolic, .reservationReceipt => .locatorCapacityReceipt
  | .hyperbolic, .writeProbe => .locatorPathExtension
  | .hyperbolic, .readProbe => .residueAbsentOrUnknown
  | .hyperbolic, .relocationProbe => .locatorRelink
  | .hyperbolic, .reservationReceipt => .locatorCapacityReceipt

/-- The shifted tag fingerprint remains only a candidate prefilter followed by full identity. -/
inductive FingerprintRole where
  | candidatePrefilterRequiringFullIdentity
  deriving DecidableEq

def retainedFingerprintRole : FingerprintRole := .candidatePrefilterRequiringFullIdentity

theorem integration_replacement_exact :
    summaryReplacement .elliptic .readProbe = .exactAddressCoverage ∧
      summaryReplacement .parabolic .readProbe = .residueAbsentOrUnknown ∧
      summaryReplacement .hyperbolic .readProbe = .residueAbsentOrUnknown ∧
      summaryReplacement .parabolic .writeProbe = .locatorPathExtension ∧
      summaryReplacement .hyperbolic .writeProbe = .locatorPathExtension ∧
      summaryReplacement .parabolic .relocationProbe = .locatorRelink ∧
      summaryReplacement .hyperbolic .relocationProbe = .locatorRelink ∧
      summaryReplacement .elliptic .reservationReceipt = .noSummaryReceipt ∧
      summaryReplacement .parabolic .reservationReceipt = .locatorCapacityReceipt ∧
      summaryReplacement .hyperbolic .reservationReceipt = .locatorCapacityReceipt ∧
      retainedFingerprintRole = .candidatePrefilterRequiringFullIdentity := by
  decide

/-! ## AEN-PIPE extraction ratification -/

/-- The explicit Charon entry encoding; omission of the three lowerings is intentional. -/
def aenExplicitOptions : List String := ["--lift-associated-types '*'", "--treat-box-as-builtin",
  "--reconstruct-fallible-operations", "--reconstruct-asserts", "--reconstruct-matches",
  "--hide-marker-traits", "--hide-allocator", "--remove-unused-self-clauses",
  "--remove-adt-clauses", "--unbind-item-vars", "--deallocate-all-locals", "--monomorphize",
  "--monomorphize-mut=all"]

def aenDroppedLowerings : List String :=
  ["--no-gen-tuple-structs", "--ops-to-function-calls", "--index-to-function-calls"]

/-- These existing surfaces type against or unfold generated declarations and require transport. -/
def loweringSensitiveBridgeSurface : List String := ["WordRefinement.*",
  "TableRefinement.CoordIdentity.coord_spread_id_injective", "Growable.Identity.*",
  "Growable.Geometry.*", "Growable.Scan.*", "Growable.Search.*", "Growable.Relocation.*",
  "Growable.Refinement.*", "Growable.Resources.*", "Growable.Representation.*",
  "Growable.Rebuild.*", "Joint.WordBridge.*"]

/-- Types and hypotheses freeze at their current declarations; specialization changes bodies only. -/
def statementTransportFreeze : List (String × String) :=
  [("WordRefinement.*", "all I64 inputs and every Sigma"),
   ("TableRefinement.CoordIdentity.*", "current word identity theorem types"),
   ("Growable.*", "current generic premises, both carriers, canonical HMap (Sigma × I64)"),
   ("Joint.WordBridge.*", "current relations and canonical replacement-map statement"),
   ("Native.LocatorSpec.*", "generic State/Representative/Sigma/Key/Value; no N=16 contraction")]

def tableStorageCoverageFields : List String := ["arenaShape", "uniqueDecodedKeys",
  "everyLiveRepresentativeReachable", "completePathsUseScalarArenas", "completePathSound",
  "reverseLinksSound"]

def locatorReservationCategories : List String := ["parabolicResidueRecords",
  "hyperbolicResidueRecords", "identityRecords", "siblingEdgeRecords", "representatives",
  "reverseLinks", "stagedPathEntries", "retainedIdentityBytes"]

/-- Proved registry exactness; translator semantics and source preservation remain open obligations. -/
theorem aen_pipeline_ratification_exact : aenExplicitOptions.length = 13 ∧
    aenDroppedLowerings.length = 3 ∧ tableStorageCoverageFields.length = 6 ∧
    locatorReservationCategories.length = 8 := by decide

open Lean Elab Command in
private def locatorClosedAxioms : Lean.NameSet :=
  ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound

open Lean Elab Command in
elab "#audit " id:ident : command => do
  let constants ← liftCoreM <| realizeGlobalConstWithInfos id
  for constant in constants do
    let axioms ← liftTermElabM <| Lean.collectAxioms constant
    let bad := axioms.filter (fun n => !locatorClosedAxioms.contains n)
    if bad.isEmpty then
      logInfo m!"'{constant}' depends on axioms: {axioms.toList}"
    else
      throwError m!"'{constant}' depends on axioms outside the closed cone: {bad.toList}"

#audit NO_CHILD
#audit scalarWordAt
#audit CheckedRecordArena
#audit ScalarArenaDirectory
#audit ScalarArenaShape
#audit ArenaPathEndpoint
#audit ScalarTraversalContract
#audit ScalarLinkingPreservation
#audit LocatorInvariant
#audit absent_of_projected_missing
#audit componentConjunction_of_presence
#audit SourceLinkObligation
#audit HashDOSBound
#audit equalDescriptors_are_covered
#audit triePepper_cost_not_yet_implied
#audit EllipticExactAddress
#audit parabolicResidueHom
#audit parabolic_residue_hom
#audit hyperbolic_scalar_channels_equal
#audit face_index_policy_exact
#audit TableStorageCoverage
#audit TableCoverageTransitions
#audit table_absent_of_complete_path_missing
#audit certified_absent_requires_coverage
#audit locatorSpec_coverage_boundary
#audit ExactIdentityContract
#audit FiniteExactIdentityContract
#audit byteKeyExactIdentity
#audit arc_identity_delegates
#audit RegenIdentityContract
#audit OpaqueKeyLocatorAdmission
#audit FaceResidueCapability
#audit EllipticPacking
#audit FaceDomain
#audit LocatorSigma
#audit FaceHome
#audit FaceHomeLaws
#audit TableHomeCapability
#audit FaceLocatorCapability
#audit typed_face_domain_exact
#audit Native.Address.pack_roundtrip
#audit Native.Address.h3_40_new_ok
#audit Native.Address.hyperbolic_scalar_denote
#audit Native.Address.home_face_denote
#audit Native.Address.home_face_joint_witness
#audit Native.Address.held_address_agrees
#audit Native.Address.held_address_joint_witness
#audit manyToOne_residue_not_exactIdentity
#audit admission_boundary
#audit table_admission_map_exact
#audit migration_options_exact
#audit api_migration_is_pending
#audit TableEntryAtCoverageAdapter
#audit tableEntryAt_absent_of_missing
#audit SourceCoverageObligations
#audit SourceLinkedCostObligation
#audit current_source_blockers_exact
#audit extraction_targets_exact
#audit ReserveBeforeRetainContract
#audit LocatorReservationReceipt
#audit ScalarArenaJointWitnesses
#audit scalarArena_joint_success_coverage
#audit scalarArena_joint_failure_committed_stutter
#audit integration_replacement_exact
#audit aen_pipeline_ratification_exact

end Apeiron.NativeLocator
