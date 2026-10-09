import Apeiron.Physics.Models.VanVleck.Elliptic
import Apeiron.Physics.Models.VanVleck.Parabolic
import Apeiron.Physics.Models.VanVleck.Hyperbolic

/-!
# Van Vleck formulas on the three quadratic faces

@name Pointwise Van Vleck exactness
@cite schrodinger-1926
@cite van-vleck-1928
@cite morette-1951
@cite feynman-hibbs-1965
@cite vattay-2026-reply
@grade proved

The three imported modules check the free, harmonic, and inverted-oscillator polar formulas
against the one-dimensional analytic Schrodinger residual. Their density prefactors are also
identified independently by mixed source-position derivatives of their actions.

## Root

`Physics.actual_schrodinger_iff_madelung` supplies the analytic polar decomposition used by each
pointwise PDE theorem. `Physics.det_generator` and `Physics.generator_trichotomy` classify the
associated classical generators. The quantum phase carrier remains elliptic on every face.

The cited converse says that complete semiclassical propagator exactness characterizes
at-most-quadratic Hamiltonians. That converse is not proved here. These modules do not define a
quantum evolution operator, delta initial condition, kernel composition, unitarity, caustic
continuation, Maslov index, or path integral.
-/
