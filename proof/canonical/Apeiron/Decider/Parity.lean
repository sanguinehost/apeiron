import Apeiron.Decider.Model

/-!
# Parity: the same systems, the same verdicts

@name Parity
@grade proved

A model of a decider is worth what it agrees with. This file runs
`Apeiron/Decider/Model.lean`'s fold on systems taken from `../song`'s own test corpus, from the
predecessor's proven model, and from a generated sweep, and checks the verdict against what the
compiled `../song/build/ncone0` returned on the same bytes. Every expected verdict below was read
off that program's standard output, not chosen.

## Where the systems come from

- the six of `../song/src/prover/cone/tests/qflia_ground.slang`, verbatim
- the twelve `#guard` systems of `../song-ref/lean/Song/proof/prover/ConeFold.lean`, which are
  live call sites of the predecessor corpus translated to rows
- `../song/src/prover/cone/ncone0.slang`'s own header example, the one its witness caution is for
- the two loci of `cell.slang`'s C4, a crossing off the cone and a point on the cone that is not
  a crossing
- five greatest-common-divisor shapes, which are the cutting plane on its own
- 220 generated systems, four columns, up to five rows, coefficients from minus three to three,
  drawn once under a fixed seed
- two capped variants, explained below
- five further hand-written shapes: two equality pairs, a one-sided system, a free column, and
  the empty system

Those are the 253 systems the file decides twice.

## What is kernel-checked, what is gated, and what is printed

The corpus is checked at three strengths and the difference is stated rather than blurred.

- `parity_kernel` and `parity_no_contradiction` cover the first twenty four systems by kernel
  `decide`, so they depend on nothing but the kernel. Those twenty four are the six of
  `../song/src/prover/cone/tests/qflia_ground.slang`, the twelve `#guard` systems of the
  predecessor's `ConeFold.lean`, `ncone0`'s header witness, the two loci of `cell.slang`'s C4, and
  three of the five cutting-plane shapes. `Apeiron/Audit.lean` audits these.
- The two whole-corpus booleans `List.all corpus agrees` and `List.all corpus noContradiction` are
  gated at the foot of the file by `#guard`. `#guard` evaluates its argument at elaboration time
  and raises an error, failing the build, unless the result is the literal `true`. It introduces
  no declaration, so nothing it computes enters any theorem's axiom cone and no declaration depends
  on `Lean.ofReduceBool`, the axiom `native_decide` would introduce.
- The counts and the names of the systems that differ are `#eval` diagnostics. They print and
  never fail the build; they are not the gate.

The gate is not kernel reduction. `#guard` runs its argument through Lean's evaluator rather than
the kernel's reducer, so the whole-corpus check is a gate on the evaluation, not a proof, and the
proved part stays the twenty four above. It is strictly stronger than a bare `#eval`, for which a
false result prints and the build still exits zero.

The reason for the split is measured. A kernel evaluation of this fold costs seconds per system
and the memory is not released between declarations: a run over all 253 at kernel strength
reached three gigabytes and was killed by the build's memory cap. `agreesV` tests the refusal
first so that `||` short-circuits before the fold runs on a system the compiled decider refused,
which helps and does not close the gap. `Apeiron/Decider/CertParity.lean`'s `cert_replay_all`
checks a different property, that the compiled decider's own UNSAT certificates replay, and does
not stand in for this check.

## What agreement means, and the six that differ

On 247 of the 253 the two return the same verdict. On six the compiled decider returns MAYBE
where this model returns UNSAT, and on none does either return a verdict the other contradicts.
MAYBE is a refusal rather than an answer, so `agrees` counts it as agreement and
`parity_no_contradiction` states the stronger fact separately on the kernel-checked core: the
model never returns SAT where the decider returned UNSAT, and never UNSAT where it returned SAT.
The whole-corpus form is gated by `#guard` rather than proved.

The cause of the six is measured, not guessed. Each has a column unbounded below, so propagation
walks the bound down without ever crossing; after enough passes the products leave `NC_MAG` and
`mulChk` sets the overflow flag, and `sayUnsat` downgrades to MAYBE. `r64_capped` and
`r89_capped` are two of those systems with each column bounded below by minus one hundred, which
is a weakening that cannot create unsatisfiability: the compiled decider then returns UNSAT and
prints a certificate, which `Apeiron/Decider/CertParity.lean` replays. So the six are the
overflow guard costing coverage, which is what `Decider.sum_can_leave_range` says the guard is
for, and not a disagreement about any system's answer.

## Root

This file states no new mathematics. It exercises `decide_unsat_sound`, and through it
`propUnsat_sound`, `eliminate_sound` and the trichotomy theorems those rest on. The systems are
data and the theorems below are kernel-checked evaluations of the fold on that data.
-/

namespace Apeiron
namespace Decider

open Fragment

-- Both limits are raised for the whole file rather than per theorem: the default heartbeat
-- budget stops a block part way through, and the default recursion depth stops `decide` on a
-- list literal.
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

/-- A row of four columns, written as a constant and a list of coefficients.

    (c : ℤ) (as : List ℤ)                    [the constant term and the coefficients]
    : Row 4                                  [the row over four columns]

    So: the wire's own shape. A short list leaves the remaining columns at zero. -/
def mkRow (c : ℤ) (as : List ℤ) : Row 4 := ⟨c, fun i => as.getD i.val 0⟩

/-- A `g` line of the wire.

    (c : ℤ) (as : List ℤ)                    [the constant term and the coefficients]
    : Line 4                                 [the inequality those numbers assert] -/
def G (c : ℤ) (as : List ℤ) : Line 4 := .ge (mkRow c as)

/-- An `e` line of the wire.

    (c : ℤ) (as : List ℤ)                    [the constant term and the coefficients]
    : Line 4                                 [the equality those numbers assert] -/
def E (c : ℤ) (as : List ℤ) : Line 4 := .eq (mkRow c as)

/-- One corpus entry: a name, a system, and what the compiled decider answered.

    : Type                                   [a name, a wire system, and a verdict] -/
abbrev Entry : Type := String × List (Line 4) × Verdict

/-- Two verdicts agree when they are the same, or the second is a refusal.

    (a b : Verdict)                          [the model's answer and the compiled decider's]
    : Bool                                   [true when they agree]

    So: MAYBE is a refusal rather than an answer, so a MAYBE from the compiled decider is not a
    verdict the model can contradict. `noConV` states the stronger relation. The refusal is
    tested first on purpose: `||` is short-circuiting, so the kernel never runs the model on a
    system the compiled decider refused, and those are exactly the systems whose propagation
    diverges and costs the most to evaluate. The statement is unchanged by the order. -/
def agreesV (a b : Verdict) : Bool := b == Verdict.maybe || a == b

/-- Two verdicts contradict only when one is UNSAT and the other SAT.

    (a b : Verdict)                          [the model's answer and the compiled decider's]
    : Bool                                   [true unless one says UNSAT and the other SAT] -/
def noConV (a b : Verdict) : Bool :=
  !((a == Verdict.unsat && b == Verdict.sat) || (a == Verdict.sat && b == Verdict.unsat))

/-- Agreement rules out contradiction.

    (a b : Verdict)                          [two verdicts]
    agreesV a b = true                       [they agree]
    noConV a b = true                        [so they do not contradict]

    So: nine cases, checked. This is why the corpus below needs one kernel evaluation per
    system rather than two. -/
theorem noConV_of_agreesV (a b : Verdict) (h : agreesV a b = true) : noConV a b = true := by
  cases a <;> cases b <;> simp_all [agreesV, noConV]

/-- The model agrees with the compiled decider on one entry.

    (e : Entry)                              [a corpus entry]
    : Bool                                   [true when the two verdicts agree] -/
def agrees (e : Entry) : Bool := agreesV (decideLines e.2.1) e.2.2

/-- The two verdicts are the same, on the nose.

    (e : Entry)                              [a corpus entry]
    : Bool                                   [true when the two verdicts are identical] -/
def identical (e : Entry) : Bool := decideLines e.2.1 == e.2.2

/-- Neither side answers where the other answers the opposite.

    (e : Entry)                              [a corpus entry]
    : Bool                                   [true unless one says UNSAT and the other SAT] -/
def noContradiction (e : Entry) : Bool := noConV (decideLines e.2.1) e.2.2

/-- Corpus block 0, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block0 : List Entry := [
  ("ground_unsat", [G (-1) [0, 0]], Verdict.unsat),
  ("trivial_sat", [G 5 [1], G 5 [(-1)]], Verdict.sat),
  ("bound_unsat", [G (-2) [1], G 0 [(-1)]], Verdict.unsat),
  ("eq_unsat", [E 0 [1], G (-1) [1]], Verdict.unsat),
  ("fm_unsat", [G (-1) [1, 0], G (-1) [0, 1], G 1 [(-1), (-1)]], Verdict.unsat),
  ("fm_sat", [G 0 [1, 0], G 0 [0, 1], G 10 [(-1), (-1)]], Verdict.sat),
  ("hdr_witness", [G (-5) [1, 0], G (-100) [1, 1]], Verdict.sat),
  ("cone_lt_asym", [G (-1) [(-1), 1], G (-1) [1, (-1)]], Verdict.unsat),
  ("cone_trans", [G (-1) [(-1), 1, 0], G 0 [0, (-1), 1], G 0 [1, 0, (-1)]], Verdict.unsat),
  ("cone_gcd_eq", [E (-1) [2]], Verdict.unsat),
  ("cone_combine_gcd", [G (-5) [2], G 5 [(-2)]], Verdict.unsat),
  ("cone_sat_box", [G 0 [1, 0], G 0 [0, 1], G 3 [(-1), (-1)]], Verdict.sat)
]

/-- Corpus block 1, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block1 : List Entry := [
  ("cone_maybe_inexact", [G 0 [2, (-3)], G 1 [(-2), 3]], Verdict.sat),
  ("cone_chain",
    [G 0 [(-1), 1, 0, 0], G 0 [0, (-1), 1, 0], G 0 [0, 0, (-1), 1], G (-1) [1, 0, 0, (-1)]],
    Verdict.unsat),
  ("cone_ground_false", [G (-1) []], Verdict.unsat),
  ("oserial_a",
    [E 0 [1, 0, (-1), 0], E 0 [0, 1, 0, (-1)], G (-1) [1, 1, (-1), (-1)]],
    Verdict.unsat),
  ("oserial_b",
    [E 0 [1, 0, (-1), 0], E 0 [0, 1, 0, (-1)], G (-1) [(-1), (-1), 1, 1]],
    Verdict.unsat),
  ("ogeom", [G 0 [1], G (-1) [0]], Verdict.unsat),
  ("omlp", [G (-1) [1, 0], G 0 [0, 1], G 0 [(-1), 0]], Verdict.unsat),
  ("cross_off_cone", [G (-2) [1], G 1 [(-1)]], Verdict.unsat),
  ("cone_not_cross", [G 0 [1], G 1 [(-1)]], Verdict.sat),
  ("cut_2x_eq_1", [E (-1) [2]], Verdict.unsat),
  ("cut_3x_eq_1", [E (-1) [3]], Verdict.unsat),
  ("cut_2x_2y_odd", [E (-1) [2, 2]], Verdict.sat)
]

/-- Corpus block 2, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block2 : List Entry := [
  ("cut_6x_eq_4", [E (-4) [6]], Verdict.unsat),
  ("cut_4x6y_eq_3", [E (-3) [4, 6]], Verdict.sat),
  ("eq_pair_sat", [E 0 [1, (-1)], G (-3) [1, 0]], Verdict.sat),
  ("eq_pair_unsat", [E 0 [1, (-1)], G (-3) [1, 0], G 2 [0, (-1)]], Verdict.unsat),
  ("one_sided", [G 0 [1, 1], G 0 [1, 0]], Verdict.sat),
  ("free_col", [G (-1) [1, 0, 0, 0]], Verdict.sat),
  ("empty_sys", [], Verdict.sat),
  ("r64_capped",
    [G (-1) [0, (-3)], G 0 [1, (-2)], G (-4) [2, (-1)], G 2 [(-2), 2], G 100 [1, 0], G 100
    [0, 1]],
    Verdict.unsat),
  ("r89_capped",
    [G 0 [(-2), 3], G (-4) [0, 1], G (-5) [(-3), (-3)], G 0 [(-1), 1], G 2 [3, 1], G 100 [1,
    0], G 100 [0, 1]],
    Verdict.unsat),
  ("r0", [G (-6) [(-1), 3], G 2 [(-3), 3], G 5 [(-3), 1], G 5 [0, 0], G (-2) [0, 3]], Verdict.sat),
  ("r1", [G (-4) [3, 2, 1], G (-3) [(-3), (-3), (-3)]], Verdict.sat),
  ("r2", [G 3 [3, 0, 2], G 6 [(-2), (-1), 0], G 5 [(-1), 3, 0]], Verdict.sat)
]

/-- Corpus block 3, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block3 : List Entry := [
  ("r3", [G (-4) [0], G 3 [(-2)], G 4 [(-1)]], Verdict.unsat),
  ("r4", [G 0 [(-3)], G 3 [(-2)]], Verdict.sat),
  ("r5", [G 1 [1, 1]], Verdict.sat),
  ("r6",
    [G (-4) [3, (-2)], E (-4) [(-1), 0], G (-5) [3, 0], G (-6) [(-2), (-1)], G 3 [(-3),
    (-1)]],
    Verdict.unsat),
  ("r7",
    [G (-1) [2, 1, 0], G (-3) [(-2), 1, (-3)], G 2 [(-3), 1, 1], G 5 [0, 0, (-2)], E 2
    [(-2), 3, (-1)]],
    Verdict.unsat),
  ("r8",
    [E (-5) [2, (-3)], G 2 [0, (-3)], E (-6) [(-3), (-1)], G 2 [3, 3], G (-1) [2, (-2)]],
    Verdict.unsat),
  ("r9", [G 1 [0, 0, 3]], Verdict.sat),
  ("r10",
    [G 1 [(-2), (-3), 2], G (-2) [0, 3, 2], G (-1) [(-2), (-1), (-3)], G (-2) [(-1), (-2),
    (-1)]],
    Verdict.maybe),
  ("r11", [G 3 [1, 1]], Verdict.sat),
  ("r12",
    [G (-6) [(-3), 0, 0, 2], G (-5) [(-1), 1, (-1), (-2)], G 3 [0, (-2), 3, 3], G 5 [0,
    (-1), 0, (-1)], G (-6) [0, 1, 1, 2]],
    Verdict.sat),
  ("r13", [G 5 [(-2), (-2), (-3), (-2)]], Verdict.sat),
  ("r14", [G 1 [0, (-2)], G 6 [3, 0]], Verdict.sat)
]

/-- Corpus block 4, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block4 : List Entry := [
  ("r15", [E (-4) [(-3), 1, 3], G 2 [(-3), 2, (-3)]], Verdict.sat),
  ("r16", [G (-6) [0, 0, 1, 2], E (-4) [1, (-1), 0, 0]], Verdict.sat),
  ("r17", [G 5 [3, (-2), 1]], Verdict.sat),
  ("r18", [G (-3) [(-1), 1], G (-3) [1, 1], G 2 [0, 2], G 0 [2, 3]], Verdict.sat),
  ("r19",
    [E 4 [(-3), (-2), (-3)], G (-2) [(-3), (-2), 2], G (-3) [1, (-3), 3], G (-6) [1, 3,
    (-3)]],
    Verdict.maybe),
  ("r20",
    [G 6 [0, (-2), 1, (-3)], G 4 [3, 1, 3, (-3)], G 3 [(-2), (-2), (-1), 0], G 3 [2, (-2),
    (-2), 1], E (-6) [3, (-1), (-2), 3]],
    Verdict.maybe),
  ("r21", [G (-2) [(-2), 1], E (-5) [(-1), 2], G 1 [(-1), (-1)]], Verdict.sat),
  ("r22", [G 6 [(-2), 0]], Verdict.sat),
  ("r23",
    [G 0 [2, 1, 2, 0], G 3 [0, 1, (-1), (-2)], G (-2) [3, (-1), 3, 2], G 0 [(-2), 2, 2, 3],
    E (-2) [(-1), (-1), 2, 2]],
    Verdict.maybe),
  ("r24", [G 3 [(-1), (-3)], E (-5) [1, 1], G 6 [2, (-3)], G 5 [3, 0]], Verdict.sat),
  ("r25", [E (-2) [1, 0, 3], G 5 [(-2), 0, 0]], Verdict.sat),
  ("r26", [G (-2) [3, 2], G 6 [(-2), (-1)], E (-2) [1, 2], G (-2) [0, 0]], Verdict.unsat)
]

/-- Corpus block 5, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block5 : List Entry := [
  ("r27",
    [G 3 [(-2), 2, (-1)], G (-3) [2, (-3), 1], E (-1) [(-2), 1, (-1)], G (-2) [(-3), 2,
    (-2)], G 4 [0, (-1), (-2)]],
    Verdict.sat),
  ("r28", [G 3 [(-3), 0, (-1)]], Verdict.sat),
  ("r29",
    [G (-5) [3, 2, (-1), (-2)], G (-3) [2, (-3), (-2), (-2)], E (-2) [(-2), (-2), (-3), 1]],
    Verdict.sat),
  ("r30", [G (-1) [1, (-2), 3], G 1 [(-2), 0, 1], G (-5) [1, (-2), 1]], Verdict.sat),
  ("r31",
    [G 1 [3, (-3)], G 3 [2, (-3)], G 1 [2, 3], G (-6) [3, 0], G (-1) [(-3), (-3)]],
    Verdict.maybe),
  ("r32", [E 0 [(-1)]], Verdict.sat),
  ("r33", [G (-3) [(-1), (-3), (-2)], G 6 [(-3), (-3), (-2)]], Verdict.sat),
  ("r34",
    [G (-6) [(-1), 2], G 0 [(-2), 1], G 2 [(-3), 2], G 2 [3, 0], E 4 [(-1), (-2)]],
    Verdict.unsat),
  ("r35", [G (-2) [2], G 1 [(-1)], G (-3) [(-2)], G (-1) [0], G (-5) [1]], Verdict.unsat),
  ("r36", [G (-1) [2, (-1), (-2)], G (-5) [3, 3, (-3)], G (-1) [1, (-3), (-1)]], Verdict.sat),
  ("r37",
    [G 0 [0, (-2), (-1)], G (-1) [3, (-2), 1], G (-4) [3, (-3), (-3)], G (-4) [3, (-1),
    (-2)]],
    Verdict.sat),
  ("r38",
    [G 5 [(-1), (-3), 2], G 4 [(-1), (-3), (-2)], G 5 [0, 3, (-1)], G (-2) [(-2), 2, (-1)]],
    Verdict.sat)
]

/-- Corpus block 6, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block6 : List Entry := [
  ("r39",
    [G (-5) [2, (-3)], G (-6) [3, 0], G 2 [(-2), 3], E 5 [(-3), 1], G 0 [1, (-3)]],
    Verdict.maybe),
  ("r40", [G (-2) [(-3)], G 0 [(-3)], G (-1) [0], G 4 [(-3)], G (-6) [(-2)]], Verdict.unsat),
  ("r41", [G (-1) [(-2), (-2)], G (-2) [(-1), 0], G (-4) [0, 3], G (-4) [2, 2]], Verdict.unsat),
  ("r42", [E (-6) [3, (-1)], G (-1) [2, (-2)], G (-4) [3, 1], G 1 [1, 1]], Verdict.sat),
  ("r43",
    [G (-1) [2, (-2)], G (-1) [1, 0], G (-2) [0, (-2)], G 5 [3, 2], E 3 [(-2), 2]],
    Verdict.unsat),
  ("r44", [G (-5) [1, 2], G (-1) [2, 3], G 6 [(-2), 1]], Verdict.sat),
  ("r45", [E 5 [(-1), 3], G 4 [3, (-2)]], Verdict.sat),
  ("r46", [E 2 [2, 3, (-1), 0]], Verdict.sat),
  ("r47", [G (-4) [0, 2, (-3), 3], G 3 [(-3), (-1), (-3), (-3)]], Verdict.sat),
  ("r48", [G (-5) [1, (-3)], G 3 [2, 3]], Verdict.sat),
  ("r49",
    [G 4 [3, (-1)], G 4 [3, 1], G (-3) [0, (-3)], G 3 [3, 2], E (-3) [(-2), (-1)]],
    Verdict.maybe),
  ("r50", [E 6 [2, 0, 0]], Verdict.sat)
]

/-- Corpus block 7, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block7 : List Entry := [
  ("r51", [G 6 [(-1), 3], G 3 [1, 0], E (-1) [(-3), (-3)], E 1 [(-3), 3]], Verdict.unsat),
  ("r52", [G 1 [1, (-1)]], Verdict.sat),
  ("r53", [G 3 [2, (-3)], G 3 [2, (-2)], G (-3) [(-1), (-3)]], Verdict.sat),
  ("r54",
    [G (-3) [(-3), 3, 3], G 5 [(-2), 2, 1], E (-1) [2, 0, (-1)], G 5 [(-1), 0, (-3)], G 6
    [(-2), (-3), 2]],
    Verdict.maybe),
  ("r55", [G 1 [2], G 1 [1]], Verdict.sat),
  ("r56",
    [G 6 [3, 0, (-3)], G 6 [(-1), (-3), 2], G 4 [(-2), 3, 1], G 6 [(-3), 1, (-1)], E (-1)
    [2, 2, 2]],
    Verdict.maybe),
  ("r57", [E (-6) [1, 1, 3], G 5 [(-3), 1, (-2)]], Verdict.sat),
  ("r58", [E (-5) [(-1), (-1)], G 2 [2, 3]], Verdict.sat),
  ("r59",
    [G (-3) [3, (-2), (-3)], G 2 [1, 2, (-3)], E 0 [1, (-2), 2], G 1 [(-3), (-1), 2], G 0
    [1, (-2), 1]],
    Verdict.maybe),
  ("r60", [G 5 [3, 1, 1], G (-4) [(-1), 3, 3], G (-3) [2, 2, 3], G 5 [3, 1, (-2)]], Verdict.sat),
  ("r61",
    [E 0 [3, (-1), 0], E 4 [0, (-2), (-2)], G 5 [(-3), 3, 0], G (-5) [(-3), 2, (-1)], E 5
    [3, (-1), (-2)]],
    Verdict.unsat),
  ("r62", [G 6 [0, 2, 1, (-3)], E (-3) [(-1), (-2), (-1), 2], E (-1) [1, 1, 2, 3]], Verdict.sat)
]

/-- Corpus block 8, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block8 : List Entry := [
  ("r63", [G (-2) [(-1), 2], E 4 [(-3), 3], G 4 [(-2), 2]], Verdict.maybe),
  ("r64", [G (-1) [0, (-3)], G 0 [1, (-2)], G (-4) [2, (-1)], G 2 [(-2), 2]], Verdict.maybe),
  ("r65",
    [G (-1) [(-2), (-1), (-3)], G (-4) [(-3), 3, 1], E 3 [1, 1, 0], G 1 [(-3), 2, 2], G 6
    [(-2), 2, 0]],
    Verdict.maybe),
  ("r66", [G (-6) [(-2), (-2)], G 2 [2, (-1)]], Verdict.sat),
  ("r67", [G (-1) [2], G (-2) [3], G (-2) [0], G (-5) [0]], Verdict.unsat),
  ("r68", [G 6 [0, 0, 3]], Verdict.sat),
  ("r69", [G (-3) [2, 2, (-3)]], Verdict.sat),
  ("r70", [G 6 [3, 0, (-1)], G (-5) [3, 2, 3]], Verdict.sat),
  ("r71", [G 0 [1, 3], G 0 [(-1), (-1)]], Verdict.sat),
  ("r72", [G 4 [2, 0]], Verdict.sat),
  ("r73",
    [G (-5) [(-1), 1, 0], G (-6) [0, 3, 2], E 1 [(-3), (-3), (-2)], E 1 [2, 2, (-3)]],
    Verdict.unsat),
  ("r74", [E 3 [(-1), 3, 2], E 6 [3, (-3), (-1)]], Verdict.sat)
]

/-- Corpus block 9, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block9 : List Entry := [
  ("r75", [G 0 [3, 3], E 1 [3, (-1)], G (-2) [(-2), (-3)], G 3 [(-1), (-2)]], Verdict.unsat),
  ("r76", [G 0 [(-3), (-3), 3]], Verdict.sat),
  ("r77",
    [E 5 [(-2), (-2), 0, (-1)], G 5 [(-1), (-3), 3, 3], G (-5) [3, (-3), (-3), 0]],
    Verdict.maybe),
  ("r78", [G 4 [3]], Verdict.sat),
  ("r79", [G 2 [3, (-1), 3], G 5 [(-3), 2, 2], G (-1) [(-1), (-2), 0]], Verdict.sat),
  ("r80", [G (-3) [(-2), 2], E (-2) [2, 1], E 1 [3, 3], G 0 [2, (-2)]], Verdict.unsat),
  ("r81", [G (-4) [0, 0]], Verdict.unsat),
  ("r82",
    [G (-6) [(-3), (-1), 2, (-1)], G 4 [3, (-2), (-3), (-1)], G 2 [(-1), 0, 1, 0], G 2 [1,
    1, (-2), (-3)]],
    Verdict.sat),
  ("r83", [G (-3) [3, (-3), 2, (-2)]], Verdict.sat),
  ("r84",
    [G (-1) [0, 0, (-3), (-3)], G (-4) [(-3), 3, (-3), 1], G 3 [0, (-2), 2, 1]],
    Verdict.sat),
  ("r85",
    [G 3 [(-1), (-3), (-3)], G 3 [(-2), 3, (-3)], E 5 [(-3), (-3), (-3)], E (-1) [(-1),
    (-2), (-2)]],
    Verdict.unsat),
  ("r86", [G 0 [0], E (-2) [1], G (-1) [(-1)], G 0 [2]], Verdict.unsat)
]

/-- Corpus block 10, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block10 : List Entry := [
  ("r87",
    [G 0 [(-1), 0, (-2), (-1)], E 4 [3, (-3), 0, (-2)], G 5 [1, 0, (-1), (-3)]],
    Verdict.sat),
  ("r88", [G 6 [(-2)]], Verdict.sat),
  ("r89",
    [G 0 [(-2), 3], G (-4) [0, 1], G (-5) [(-3), (-3)], G 0 [(-1), 1], G 2 [3, 1]],
    Verdict.maybe),
  ("r90", [G (-6) [(-2), (-3), 0], G 2 [(-2), (-2), 0]], Verdict.sat),
  ("r91", [E (-3) [(-2)]], Verdict.unsat),
  ("r92", [G 6 [0, 3, 3], G 5 [1, (-3), 3]], Verdict.sat),
  ("r93",
    [E (-5) [1, 3, 2, (-1)], G 3 [(-2), 1, (-3), 1], G 1 [(-2), (-1), (-3), (-1)], G (-6)
    [(-1), (-3), 1, (-1)], E 5 [3, (-1), 0, (-1)]],
    Verdict.sat),
  ("r94", [G 6 [0, 1], E (-3) [2, 3], G 4 [3, 1], G 2 [3, (-3)]], Verdict.sat),
  ("r95", [G 5 [(-1), (-3), (-1)], G 4 [3, (-2), 2], G 4 [0, 1, 3]], Verdict.sat),
  ("r96", [G 2 [3], G (-4) [1], G 0 [0], E (-5) [(-3)]], Verdict.unsat),
  ("r97",
    [G 2 [1, (-2), 0, 3], G (-5) [3, (-3), 2, (-1)], E 2 [3, (-1), 2, (-2)], G (-5) [(-2),
    1, (-1), 0]],
    Verdict.sat),
  ("r98",
    [G (-4) [0, (-2)], G (-6) [2, (-1)], G (-1) [3, (-3)], G 3 [(-3), (-3)], G (-4) [3, 2]],
    Verdict.maybe)
]

/-- Corpus block 11, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block11 : List Entry := [
  ("r99", [E 4 [1, (-3)]], Verdict.sat),
  ("r100", [G 2 [3, (-2), (-1)], G 2 [0, 1, (-2)], G 6 [(-3), 0, 1]], Verdict.sat),
  ("r101", [E 4 [2], G (-2) [(-1)]], Verdict.sat),
  ("r102",
    [G 1 [(-3), 0, (-3), 3], G (-1) [1, 1, 2, (-2)], G (-1) [3, (-3), (-3), 1]],
    Verdict.sat),
  ("r103",
    [E 5 [2, (-2), 2], G 4 [(-1), 3, (-2)], G (-1) [1, (-2), (-2)], G (-5) [3, (-1), (-2)],
    G (-1) [(-3), 1, 2]],
    Verdict.unsat),
  ("r104", [G 0 [(-2)]], Verdict.sat),
  ("r105", [E 0 [(-1)], G 6 [(-3)], G (-6) [0]], Verdict.unsat),
  ("r106", [G 4 [2], G (-4) [(-3)]], Verdict.maybe),
  ("r107", [G 6 [(-1), 2, (-3)], G 4 [1, 2, 2], E (-2) [(-1), (-2), 3]], Verdict.sat),
  ("r108",
    [G 6 [(-3), 1], G (-4) [3, (-2)], E 0 [(-1), (-1)], G 0 [2, 2], G (-5) [1, 0]],
    Verdict.unsat),
  ("r109", [G 5 [0, 3, (-2)], G (-6) [(-2), (-3), 3], E (-3) [3, (-3), 1]], Verdict.maybe),
  ("r110", [G (-4) [1, (-1)], G (-3) [1, (-3)], G 1 [1, (-2)], G (-1) [3, (-2)]], Verdict.sat)
]

/-- Corpus block 12, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block12 : List Entry := [
  ("r111", [G (-1) [0, (-2)]], Verdict.sat),
  ("r112",
    [G 0 [(-3), 0, 2], G (-6) [1, (-3), 2], G 5 [1, 1, (-3)], G 3 [3, 3, (-1)]],
    Verdict.maybe),
  ("r113", [G 4 [1, 0], E 5 [2, 0]], Verdict.unsat),
  ("r114", [G 6 [(-1)]], Verdict.sat),
  ("r115", [G 0 [3, (-2), (-3)]], Verdict.sat),
  ("r116",
    [G 0 [3, 0, (-2), (-1)], G (-1) [3, (-3), (-1), 1], G 4 [3, (-3), 3, 3], E (-2) [2,
    (-3), 0, (-3)]],
    Verdict.maybe),
  ("r117", [G 3 [(-2), (-1), 3, 2], G (-5) [(-3), (-2), 0, 2]], Verdict.sat),
  ("r118", [E (-4) [(-2), 3]], Verdict.sat),
  ("r119", [G 2 [2, 0, 2], G (-3) [1, (-2), (-2)]], Verdict.sat),
  ("r120", [G (-2) [1, (-2)], G 4 [0, (-1)]], Verdict.sat),
  ("r121", [G (-3) [(-1), (-2), (-3)], G (-1) [(-1), 0, 3], G (-3) [0, 3, (-1)]], Verdict.maybe),
  ("r122", [E 1 [3], G (-5) [2], E 6 [(-3)]], Verdict.unsat)
]

/-- Corpus block 13, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block13 : List Entry := [
  ("r123", [G 3 [2, 1, 2, 0], G 2 [3, (-3), 1, (-1)]], Verdict.sat),
  ("r124", [G 1 [1, 3, (-1), 2], G (-2) [0, 2, (-3), 1]], Verdict.sat),
  ("r125", [G 4 [(-3), 0], E (-2) [3, 3], G (-3) [(-3), 2], G (-3) [(-2), 1]], Verdict.maybe),
  ("r126", [G 4 [(-3), 3, (-2)], G 5 [(-1), 3, 2]], Verdict.sat),
  ("r127", [G 4 [(-3), (-3)], E (-3) [3, (-2)], G 4 [(-3), (-2)]], Verdict.sat),
  ("r128", [G 3 [0, 2], G (-2) [(-1), (-2)], G 3 [0, (-1)]], Verdict.sat),
  ("r129",
    [G 2 [(-2), 3, 3, 2], G 5 [2, 2, 3, 1], G 0 [0, 1, 3, 3], G (-1) [0, (-2), 3, (-2)], G
    (-1) [(-2), 1, 3, 0]],
    Verdict.maybe),
  ("r130", [G 3 [(-1), 0], G (-3) [(-2), 0], G 1 [(-2), 0]], Verdict.sat),
  ("r131", [G (-3) [3, (-3), (-3), 2], G (-4) [1, 3, 0, 0]], Verdict.sat),
  ("r132", [G (-2) [(-3), 0, (-3)], G 5 [1, 0, (-1)]], Verdict.sat),
  ("r133",
    [G (-2) [0, 0, 0, (-1)], E 3 [2, (-3), (-1), (-3)], G (-4) [(-1), 0, 2, (-1)], G 4 [0,
    0, (-2), (-3)]],
    Verdict.sat),
  ("r134",
    [G (-2) [0, 3], E (-5) [(-2), (-3)], G 4 [1, (-2)], G (-1) [3, 1], E 3 [(-1), 3]],
    Verdict.unsat)
]

/-- Corpus block 14, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block14 : List Entry := [
  ("r135",
    [G (-6) [(-2), (-1), (-2), 1], G 1 [1, (-2), (-1), 1], G 5 [(-1), (-3), (-2), 0]],
    Verdict.sat),
  ("r136", [G (-2) [1, 0], G (-5) [1, 0]], Verdict.sat),
  ("r137",
    [G 3 [3, 0, 2], G (-5) [2, 2, (-1)], G (-2) [1, (-2), (-2)], G (-3) [(-3), 3, 2]],
    Verdict.unsat),
  ("r138", [G (-4) [(-1), 0], E 4 [(-2), (-2)], G 1 [3, (-1)], E 6 [0, 0]], Verdict.unsat),
  ("r139", [E 1 [3, 1, (-3)], G (-5) [(-1), (-2), 3]], Verdict.sat),
  ("r140",
    [E 2 [(-1), (-3), 1, (-3)], G 6 [2, 2, 2, 2], G 2 [(-2), (-1), (-1), 0], G (-2) [(-2),
    2, 1, (-2)]],
    Verdict.sat),
  ("r141", [G 3 [1], G 0 [(-2)], G (-5) [3]], Verdict.unsat),
  ("r142", [G (-5) [2, 3], G 0 [2, 0], G 6 [(-2), 3], G 3 [(-3), (-1)]], Verdict.sat),
  ("r143", [G 2 [0, (-3), 0]], Verdict.sat),
  ("r144", [E 1 [0, 3, 0]], Verdict.unsat),
  ("r145", [G 5 [(-2), 2, 2], G 5 [2, (-3), 1], G 6 [2, (-1), (-3)]], Verdict.maybe),
  ("r146", [G 5 [0, (-2), 2, (-2)], E (-1) [2, (-2), 2, 0], G 0 [(-1), 0, 0, 3]], Verdict.maybe)
]

/-- Corpus block 15, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block15 : List Entry := [
  ("r147", [G 3 [1, 2]], Verdict.sat),
  ("r148", [G 0 [(-3)], E (-3) [0], E (-5) [0], E 6 [(-2)]], Verdict.unsat),
  ("r149", [G 3 [(-1), 3], G 0 [2, (-3)], E 1 [1, 1], G 5 [0, 2], G 1 [1, 0]], Verdict.sat),
  ("r150",
    [E (-1) [(-1), 3, 0, 3], G 2 [(-3), 2, 0, (-2)], G (-1) [(-1), (-3), (-1), 2]],
    Verdict.sat),
  ("r151", [G (-5) [1, (-3), (-3)], G 0 [3, 0, 0]], Verdict.sat),
  ("r152", [G 1 [(-3), 1], G (-1) [0, 1]], Verdict.sat),
  ("r153",
    [G (-1) [3, (-1)], G (-5) [(-2), (-2)], G (-3) [0, 1], G (-5) [(-3), 2]],
    Verdict.unsat),
  ("r154", [E (-6) [0], E (-1) [2], G (-2) [3], E (-3) [3]], Verdict.unsat),
  ("r155", [G 3 [1, 1], E (-6) [3, 1], G (-4) [1, 0]], Verdict.sat),
  ("r156",
    [G (-4) [(-3), (-3), (-2)], G 2 [(-1), (-1), 2], G (-3) [(-3), 1, (-2)], G 5 [(-2), 0,
    0]],
    Verdict.sat),
  ("r157", [G 1 [(-2)], G 6 [0], G (-5) [(-1)], G (-5) [0]], Verdict.unsat),
  ("r158", [G (-5) [(-1), (-2)], G 4 [1, 0], G 0 [(-1), 3]], Verdict.sat)
]

/-- Corpus block 16, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block16 : List Entry := [
  ("r159", [G (-1) [2, 3], G (-4) [(-2), (-3)]], Verdict.unsat),
  ("r160", [G 0 [(-2), (-2), (-1)]], Verdict.sat),
  ("r161", [G 1 [3], G 4 [(-3)]], Verdict.sat),
  ("r162", [G (-6) [3, 1]], Verdict.sat),
  ("r163", [G (-1) [2, (-1)], G (-3) [(-3), (-1)]], Verdict.maybe),
  ("r164", [G 4 [0, 1], G (-3) [(-3), (-3)], G 6 [3, 0]], Verdict.sat),
  ("r165", [G 1 [3, (-3)], E (-1) [3, 2], G 4 [(-3), 3], G 6 [(-1), 3]], Verdict.sat),
  ("r166", [G (-4) [(-2), 0, 1]], Verdict.sat),
  ("r167", [G (-3) [1, (-1)], G 1 [0, 1], G 0 [(-2), 0], G 1 [3, (-3)]], Verdict.unsat),
  ("r168", [G 1 [(-1), (-3)], E 5 [0, 1], G (-2) [1, 2]], Verdict.sat),
  ("r169", [G (-4) [3, 3], G 5 [3, 0]], Verdict.sat),
  ("r170", [G (-4) [(-1), (-2), 0], G (-5) [3, (-2), 0], G (-6) [2, 1, (-2)]], Verdict.sat)
]

/-- Corpus block 17, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block17 : List Entry := [
  ("r171", [G 6 [0, 2], E 6 [(-3), 0], G (-6) [(-1), (-2)]], Verdict.unsat),
  ("r172",
    [E 3 [(-2), (-2), (-1)], G 2 [(-2), 0, (-2)], G 5 [(-1), (-2), (-1)], G 5 [3, (-2), 3],
    G (-5) [(-3), (-3), (-3)]],
    Verdict.maybe),
  ("r173", [G 6 [(-3), (-1), (-3)], E (-6) [3, 2, 3], G (-2) [2, (-3), (-3)]], Verdict.maybe),
  ("r174", [G (-2) [0, 3, (-2)], G 6 [(-1), 1, 0]], Verdict.sat),
  ("r175",
    [G (-3) [(-2), (-1), (-1), 3], G 5 [2, (-2), 0, (-3)], G 0 [3, (-2), 3, 3]],
    Verdict.maybe),
  ("r176", [E 5 [0, 2], G (-1) [1, (-3)], G 5 [(-1), 0], G (-4) [2, 1]], Verdict.unsat),
  ("r177",
    [G (-6) [(-1), 2, (-2)], E 0 [(-3), 0, 2], G 0 [(-2), (-2), 0], G 1 [(-2), 1, 2], G 3
    [3, (-2), 1]],
    Verdict.unsat),
  ("r178", [G 1 [1, 0], E (-2) [3, (-2)], G 1 [3, (-2)]], Verdict.sat),
  ("r179", [E 4 [2], G 5 [(-2)], G (-6) [3], G 3 [(-1)], G (-1) [(-2)]], Verdict.unsat),
  ("r180",
    [G 5 [3, 2, 0], G 3 [(-3), (-1), 2], G (-6) [3, (-3), (-3)], G (-2) [(-2), 0, 2]],
    Verdict.maybe),
  ("r181", [E (-1) [3, 1]], Verdict.sat),
  ("r182",
    [G 4 [(-1), 1], G 3 [3, (-3)], G 0 [(-3), 3], G (-6) [2, (-1)], G 4 [2, 0]],
    Verdict.sat)
]

/-- Corpus block 18, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block18 : List Entry := [
  ("r183", [G 4 [2, 3], G 2 [(-3), 0], G (-3) [0, 0]], Verdict.unsat),
  ("r184", [E 5 [(-3), 0, 2, 1], G 5 [0, (-1), 0, 0]], Verdict.sat),
  ("r185", [G 3 [3], G (-5) [(-1)], G (-6) [(-3)], G (-2) [1], G (-6) [(-1)]], Verdict.unsat),
  ("r186", [E (-3) [(-3), 0], G (-2) [(-1), 1], G 5 [3, (-1)], G 1 [3, (-1)]], Verdict.unsat),
  ("r187", [G 2 [3, (-2), 3]], Verdict.sat),
  ("r188", [G (-4) [2, 1, 3, 3], G (-3) [(-2), (-1), 3, 3]], Verdict.sat),
  ("r189", [G (-2) [(-3), 0], G 6 [0, 2], G (-4) [(-3), 0], G 5 [0, (-2)]], Verdict.sat),
  ("r190", [G 3 [3, 3], G (-4) [(-1), 1]], Verdict.sat),
  ("r191", [G 6 [3], E (-5) [(-1)], G 1 [1]], Verdict.unsat),
  ("r192", [E (-1) [3, 2], G (-1) [2, (-1)]], Verdict.sat),
  ("r193", [G (-1) [2, 1, 0], G (-5) [(-3), 3, 0]], Verdict.maybe),
  ("r194",
    [E (-2) [(-1), 2], E 2 [(-3), (-3)], G 0 [(-3), (-1)], G (-2) [0, (-2)], G (-6) [(-1),
    (-1)]],
    Verdict.unsat)
]

/-- Corpus block 19, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block19 : List Entry := [
  ("r195",
    [E 6 [0, 2], G 4 [(-3), 2], G 5 [0, (-2)], G 6 [0, (-1)], G (-3) [(-1), 3]],
    Verdict.maybe),
  ("r196", [G 0 [2]], Verdict.sat),
  ("r197", [G (-1) [1], G (-1) [(-2)]], Verdict.unsat),
  ("r198", [G 4 [(-2), 0], G 4 [0, 3], G 5 [3, 0], G (-5) [3, (-2)], G 4 [3, 1]], Verdict.maybe),
  ("r199", [G 2 [(-3), (-2), 3], G 0 [(-3), (-2), 3], G 5 [(-3), 3, 3]], Verdict.maybe),
  ("r200", [G (-3) [3], G 0 [(-1)], G 5 [1]], Verdict.unsat),
  ("r201", [G 4 [2, (-2), (-2)], G 3 [(-1), (-2), (-1)]], Verdict.sat),
  ("r202",
    [G 6 [(-3), (-3), (-2), 2], G 4 [(-1), 2, (-2), (-1)], G 2 [2, (-3), (-1), 3]],
    Verdict.sat),
  ("r203", [G 3 [0, 2], G 1 [(-3), (-3)], G 3 [0, 3]], Verdict.maybe),
  ("r204", [G 2 [2, (-1), 0, (-1)]], Verdict.sat),
  ("r205", [G 4 [1], G (-4) [0]], Verdict.unsat),
  ("r206",
    [G 2 [0, (-3), (-2), 2], G 4 [1, 0, (-2), (-1)], G 2 [3, 2, 1, 3], G 2 [0, 2, (-1),
    (-1)]],
    Verdict.sat)
]

/-- Corpus block 20, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block20 : List Entry := [
  ("r207", [G 0 [3, 0, (-1)], G 4 [(-3), 1, (-3)]], Verdict.sat),
  ("r208", [G 2 [2, (-2)], E 2 [3, 0], E 5 [0, 2], G (-2) [0, (-3)]], Verdict.unsat),
  ("r209", [G (-2) [2], E (-5) [(-2)], E 2 [(-2)]], Verdict.unsat),
  ("r210", [G 3 [(-1), 3, (-1)], G (-1) [0, (-1), 0]], Verdict.sat),
  ("r211", [G 1 [(-2), (-1), (-3)]], Verdict.sat),
  ("r212", [G 1 [(-2), (-3), 3], G 5 [3, 2, 1], E 6 [2, 2, 2]], Verdict.maybe),
  ("r213", [G 0 [0], E 1 [(-1)], G 0 [2], G (-5) [(-3)]], Verdict.unsat),
  ("r214", [E 1 [0, (-2)], G 5 [(-3), 1]], Verdict.unsat),
  ("r215", [G (-3) [0, (-2)], G 1 [0, 1], G (-2) [0, (-2)]], Verdict.unsat),
  ("r216",
    [E (-2) [(-1), 0, (-3)], G 5 [(-2), 3, (-1)], G 4 [0, 0, (-1)], G 2 [3, 2, (-2)]],
    Verdict.maybe),
  ("r217",
    [G 4 [0, (-1)], G (-2) [(-3), (-3)], E (-1) [2, (-3)], E (-1) [3, 3], G 1 [0, (-1)]],
    Verdict.unsat),
  ("r218", [G (-4) [2], G 3 [1], E 6 [1], G 6 [1]], Verdict.unsat)
]

/-- Corpus block 21, as the compiled decider answered it.

    : List Entry                             [named systems with the verdicts read off] -/
def block21 : List Entry := [
  ("r219", [G 1 [3], E (-6) [1], G 4 [1], G 1 [(-2)], G 3 [(-2)]], Verdict.unsat)
]

/-- Every system in the corpus, with the verdict the compiled decider gave it.

    : List Entry                             [the whole corpus, in blocks of twelve] -/
def corpus : List Entry :=
  block0 ++ block1 ++ block2 ++ block3 ++ block4 ++ block5 ++ block6 ++ block7 ++ block8 ++
    block9 ++ block10 ++ block11 ++ block12 ++ block13 ++ block14 ++ block15 ++ block16 ++
    block17 ++ block18 ++ block19 ++ block20 ++ block21

/-- The part of the corpus the kernel checks, twenty four systems.

    : List Entry                             [the first two blocks of the corpus] -/
def kernelCorpus : List Entry := block0 ++ block1

/-- Block 0 agrees, checked by the kernel.

    List.all block0 agrees = true          [every entry of the block agrees] -/
theorem parity_block0 : List.all block0 agrees = true := by decide

/-- Block 1 agrees, checked by the kernel.

    List.all block1 agrees = true          [every entry of the block agrees] -/
theorem parity_block1 : List.all block1 agrees = true := by decide





















/-- **The parity theorem.** The model and the compiled decider agree, kernel-checked.

    List.all kernelCorpus agrees = true      [every entry of the checked part agrees]

    So: the twenty four systems of the checked core, each decided twice, once by
    `../song/build/ncone0` on the bytes and once here by the kernel with no compiler in the
    way. Those twenty four are the six of
    `../song/src/prover/cone/tests/qflia_ground.slang`, the twelve `#guard` systems of the
    predecessor's `ConeFold.lean`, `ncone0`'s header witness, the two loci of `cell.slang`'s C4,
    and three of the five cutting-plane shapes. Agreement means the same verdict, or a refusal by
    the compiled decider, and `parity_no_contradiction` rules out the case that matters. The other
    two hundred and twenty nine are covered by the two `#guard` gates at the foot of the file,
    which fail the build if either whole-corpus boolean is false; that is runtime evidence rather
    than a theorem. -/
theorem parity_kernel : List.all kernelCorpus agrees = true := by
  simp [kernelCorpus, List.all_append, parity_block0, parity_block1]

/-- **Neither side ever contradicts the other.**

    List.all kernelCorpus noContradiction    [no checked system is UNSAT one side, SAT the other]
    = true                                   [the whole list of them comes out true]

    So: the statement that matters for trust. Where the two differ, one of them refused. The
    whole corpus carries the same property as a build-failing `#guard` below, evaluated at
    elaboration time rather than proved. -/
theorem parity_no_contradiction : List.all kernelCorpus noContradiction = true := by
  rw [List.all_eq_true]
  intro e he
  exact noConV_of_agreesV _ _ (List.all_eq_true.mp parity_kernel e he)

/- The whole corpus, gated at build time rather than proved.

The two whole-corpus booleans are checked by `#guard`, not by a theorem. `#guard` evaluates its
argument during elaboration and raises an error, failing the build, unless the result is the
literal `true`. It introduces no declaration, so no declaration depends on `Lean.ofReduceBool`.
That axiom, which says the compiler and the kernel agree on Booleans, is outside the closed cone
`Apeiron/Audit.lean` gates on, and no declaration in this file or in that cone uses it.

The mechanism is not kernel reduction. `#guard` runs its argument through Lean's evaluator rather
than the kernel's reducer, so the gate is runtime evidence on the evaluation, not a proof, and it
is strictly stronger than a bare `#eval`, for which a false result prints and the build still
exits zero.

Why the whole corpus is not a kernel theorem: a kernel evaluation of this fold costs seconds per
system and the memory is not released between declarations, so the whole corpus at kernel
strength does not fit the three gigabyte budget this corpus builds under. Measured, not guessed:
a run over all 253 reached 3.0 gigabytes and was killed. `parity_kernel` is therefore the checked
core, and it is the part chosen for content rather than for cheapness: every system from
`../song`'s own test file and every one from the predecessor's proven model is in it. -/

/-- **The SAT verdict is not sound, and here is the system.**

    decideLines [E (-3) [4, 6, 0, 0]] = Verdict.sat       [the decider answers SAT]
    ∧                                                     [and, on top of that]
    ¬ ∃ ρ : Fin 4 → ℤ                                     [no assignment of whole numbers]
    SatLines ρ [E (-3) [4, 6, 0, 0]]                      [satisfies that system]

    So: `4x + 6y = 3` has no whole-number solution, because two divides the left side and not
    the right, and both this model and `../song/build/ncone0` answer SAT on it. Checked on the
    running program: `printf '2\ne -3 4 6\n;\n' | ncone0` prints `SAT` and exits 1, with no
    `# model` line, so no witness accompanies the claim.

    Why the fold reaches it: the equality loads as two rows with coefficients `[4, 6]` and
    `[-4, -6]`, the elimination of the first column reduces both multipliers by their common
    factor four, and the combined row is identically zero, which is a row that holds. The
    greatest-common-divisor tightening never runs on the pair, because it runs on the combined
    row and that row has no coefficients left. Nothing crossed, every pair resolved exactly, so
    the run claims SAT.

    What this costs and does not cost. UNSAT is the verdict that discharges a law, and
    `decide_unsat_sound` and `checkCert_sound` are about UNSAT alone, so no proof `../song`
    accepts rests on this. What a false SAT does is report a law as failing when it holds, with
    no counterexample to show for it, which is the failure mode `ncone0`'s own witness
    construction exists to catch and did not: `saySat` prints SAT whether or not `tryWitness`
    produced anything. Stated here rather than in `theory/DEAD_ENDS.md` because it is a fact
    about the decider that a reader of this file needs, and because the model reproducing it
    faithfully is what makes it a parity result rather than a divergence. -/
theorem sat_verdict_is_not_sound :
    decideLines [E (-3) [4, 6, 0, 0]] = Verdict.sat ∧
      ¬ ∃ ρ : Fin 4 → ℤ, SatLines ρ [E (-3) [4, 6, 0, 0]] := by
  refine ⟨by decide, ?_⟩
  rintro ⟨ρ, h⟩
  have hr := h (E (-3) [4, 6, 0, 0]) (List.mem_singleton_self _)
  simp only [Line.Sat, Row.value, E, mkRow, Fin.sum_univ_four, List.getD] at hr
  norm_num at hr
  omega

-- The counts, printed at build time over the WHOLE corpus, not only the kernel-checked part:
-- how many systems, how many verdicts are identical, how many agree under `agrees`, and which
-- ones differ. The six that differ are the compiled decider refusing where this model decides,
-- which the header explains. These `#eval` lines are diagnostics: they print and never fail the
-- build, and they are not the gate.
#eval corpus.length
#eval kernelCorpus.length
#eval (corpus.filter identical).length
#eval (corpus.filter agrees).length
#eval (corpus.filter (fun e => !identical e)).map (fun e => e.1)

-- The whole-corpus booleans, as build gates. `#guard` raises an error, failing the build, unless
-- its argument evaluates to the literal `true`; it adds no declaration, so no declaration depends
-- on `Lean.ofReduceBool`. This is what makes a false whole-corpus boolean a red build rather than
-- a line in a log. The evaluation is the compiled evaluator `#eval` uses, not kernel reduction, so
-- the gate is runtime evidence, not a proof; the proved part is `parity_kernel` above.
#guard List.all corpus agrees
#guard List.all corpus noContradiction

end Decider
end Apeiron
