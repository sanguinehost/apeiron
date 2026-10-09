import Apeiron.Trichotomy.Basic
import Apeiron.Trichotomy.Idempotents
import Apeiron.Trichotomy.NullBasis
import Apeiron.Trichotomy.Channels
import Apeiron.Trichotomy.Implicative
import Apeiron.Trichotomy.Trace
import Apeiron.Trichotomy.Derived.Differential
import Apeiron.Trichotomy.Derived.Projection
import Apeiron.Trichotomy.Derived.Gates
import Apeiron.Trichotomy.SplitComplex
import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.Roots
import Apeiron.Trichotomy.Flow
import Apeiron.Trichotomy.Derived.CStar

import Apeiron.Doubling.CayleyDickson
import Apeiron.Doubling.Octonion
import Apeiron.Doubling.SplitOctonion
import Apeiron.Doubling.Sedenion
import Apeiron.Doubling.Bott

import Apeiron.Lattice.IntegralForm
import Apeiron.Lattice.EvenUnimodular

import Apeiron.Projective.Basic
import Apeiron.Projective.Bulk
import Apeiron.Projective.Length

import Apeiron.Bridge.Cell
import Apeiron.Bridge.Prover0
import Apeiron.Bridge.ExecutionContract

import Apeiron.Combinator.Basic
import Apeiron.Combinator.Confluence
import Apeiron.Combinator.Omega
import Apeiron.Combinator.Pair
import Apeiron.Combinator.Model
import Apeiron.Combinator.Invariant
import Apeiron.Combinator.Wires
import Apeiron.Combinator.Linear.Cost
import Apeiron.Combinator.Denotation
import Apeiron.Combinator.Verdict
import Apeiron.Combinator.Recursion
import Apeiron.Combinator.Resonance
import Apeiron.Combinator.RelationalPaths
import Apeiron.Combinator.RelationalApplication
import Apeiron.Combinator.BoundedRoutes

import Apeiron.Capability.Basic
import Apeiron.Capability.Extent
import Apeiron.Capability.Naming
import Apeiron.Capability.MaskedTable
import Apeiron.Boundary.Basic
import Apeiron.Region.Graded
import Apeiron.Classification.Margin
import Apeiron.Classification.ScaleInvariant
import Apeiron.Transport.Basic
import Apeiron.Physics.Models.Baroclinic
import Apeiron.Physics.Quadratic
import Apeiron.Physics.Wick
import Apeiron.Physics.Madelung
import Apeiron.Physics.Models.CellField
import Apeiron.Physics.Flavour
import Apeiron.Physics.Models.Mixing
import Apeiron.Physics.Ladder
import Apeiron.Physics.Models.Chaos
import Apeiron.Physics.Models.ClassicalWaves
import Apeiron.Physics.Models.Schrodinger
import Apeiron.Physics.Models.VanVleck

import Apeiron.Physics.CarrierAnyRing
import Apeiron.Physics.CarrierFiniteAnyRing
import Apeiron.Physics.CarrierBodyWorkAnyRing
import Apeiron.Physics.CarrierDissipationAnyRing
import Apeiron.Physics.CarrierVolumeAnyRing
import Apeiron.Physics.WaveAnyRing
import Apeiron.Physics.Carrier
import Apeiron.Physics.Models.CarrierFlow
import Apeiron.Physics.Carrier.Derived.Circulation
import Apeiron.Physics.Carrier.Derived.Authority
import Apeiron.Physics.Carrier.Derived.BodyField
import Apeiron.Physics.Models.WaveTime

import Apeiron.Cosmology.Majorana
import Apeiron.Cosmology.NeutrinoSum
import Apeiron.Linear.Basic

import Apeiron.Table.Basic
import Apeiron.Table.OrderBlind
import Apeiron.Table.Adjacency
import Apeiron.Table.Home
import Apeiron.Table.WordIdentity

import Apeiron.Machine.Encoding

import Apeiron.Parser.Expression
import Apeiron.Parser.Arithmetic

import Apeiron.Decider.Face
import Apeiron.Decider.Wire
import Apeiron.Decider.Propagate
import Apeiron.Decider.Eliminate
import Apeiron.Decider.Residue
import Apeiron.Decider.UnitProjection
import Apeiron.Decider.CompleteDecision
import Apeiron.Decider.CompleteCertificate
import Apeiron.Decider.Cert
import Apeiron.Decider.Extended.CuttingPlane
import Apeiron.Decider.Stream
import Apeiron.Decider.Model
import Apeiron.Decider.Parity
import Apeiron.Decider.Extended.CertParity

import Apeiron.Fragment.Basic
import Apeiron.Numeric.Accumulator


/-!
# Apeiron

The library root. `Apeiron.Trichotomy.AnyRing` is the generic base: the two-slot algebra over an
arbitrary commutative ring, from which the rest of the library is built. `Apeiron.Trichotomy.Basic`
is its specialization to the reals, where `σ` ranges over `ℝ` and the classification runs by its
sign.
-/
