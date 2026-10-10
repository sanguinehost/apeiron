import Apeiron.Decider.Parity

/-!
# Certificate parity: the decider's own refutations, replayed

@name Certificate Parity
@grade proved

`Apeiron/Decider/Cert.lean` proves that a certificate the checker accepts refutes the system.
That theorem is worth what it is applied to. This file applies it to every certificate the
compiled `../song/build/ncone0` printed on the corpus of `Apeiron/Decider/Parity.lean`: 78
of them, 641 operations in total, read off that program's standard output and translated into
`CertOp` with no editing beyond the alphabet.

## What is checked, and what that is worth

`cert_replay_all` says the checker accepts every one. Combined with `checkCert_sound` that gives,
for each of these 78 systems, a proof that no assignment satisfies it, resting on no property of
the decider at all: the log names input lines and previously built rows, and the checker
recomputes every row from them.

This is the link `Apeiron/Decider/Model.lean`'s header calls the load-bearing one across the
refinement gap. It does not close the gap: `certcheck.slang` is itself a compiled Iota program
and nothing here says it agrees with `checkCert`. What it does is move what an UNSAT rests on
from `ncone0`'s whole fold onto a replay whose soundness is a theorem.

## Two refusals, checked

`cert_rejects_tampered` alters one multiplier of a real certificate and `cert_rejects_truncated`
removes its closing operation. The checker refuses both, so acceptance is not vacuous. This is
the red control `../song`'s own gate discipline requires: a gate nobody watches fail is not a
gate.
-/

namespace Apeiron
namespace Decider

-- The corpus below is decided by the kernel, one evaluation of the fold per system. Both limits
-- are raised for the whole file rather than per theorem: the default heartbeat budget stops a
-- block part way through, and the default recursion depth stops `decide` on a list literal.
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

/-- One certificate entry: a name, the system, and the log the decider printed for it.

    : Type                                   [a name, a wire system, and a certificate] -/
abbrev CertEntry : Type := String × List (Line 4) × List CertOp

/-- The checker accepts one entry's certificate.

    (e : CertEntry)                          [a certificate entry]
    : Bool                                   [true when the replay accepts] -/
def replays (e : CertEntry) : Bool := checkCert e.2.1 [] e.2.2

/-- Certificate block 0, printed by the compiled decider.

    : List CertEntry                         [systems with the logs the decider emitted] -/
def certBlock0 : List CertEntry := [
  ("ground_unsat", [G (-1) [0, 0]], [.load 0, .tidy 0, .close 1]),
  ("bound_unsat", [G (-2) [1], G 0 [(-1)]], [.load 0, .load 1, .combo 0 1 1 1, .tidy 2, .close 3]),
  ("eq_unsat",
    [E 0 [1], G (-1) [1]],
    [.load 0, .neg 0, .load 1, .combo 2 1 1 1, .tidy 3, .close 4]),
  ("fm_unsat",
    [G (-1) [1, 0], G (-1) [0, 1], G 1 [(-1), (-1)]],
    [.load 0, .load 1, .load 2, .combo 2 1 1 1, .combo 0 3 1 1, .tidy 4, .close 5]),
  ("cone_lt_asym",
    [G (-1) [(-1), 1], G (-1) [1, (-1)]],
    [.load 0, .load 1, .combo 1 0 1 1, .tidy 2, .close 3]),
  ("cone_trans",
    [G (-1) [(-1), 1, 0], G 0 [0, (-1), 1], G 0 [1, 0, (-1)]],
    [.load 0, .load 1, .load 2, .combo 2 0 1 1, .combo 3 1 1 1, .tidy 4, .close 5]),
  ("cone_gcd_eq",
    [E (-1) [2]],
    [.load 0, .neg 0, .tidy 0, .tidy 1, .neg 3, .combo 2 4 1 1, .tidy 5, .close 6]),
  ("cone_combine_gcd",
    [G (-5) [2], G 5 [(-2)]],
    [.load 0, .load 1, .tidy 0, .tidy 1, .neg 3, .combo 2 4 1 1, .tidy 5, .close 6]),
  ("cone_chain",
    [G 0 [(-1), 1, 0, 0], G 0 [0, (-1), 1, 0], G 0 [0, 0, (-1), 1], G (-1) [1, 0, 0, (-1)]],
    [.load 0, .load 1, .load 2, .load 3, .combo 3 0 1 1, .combo 4 1 1 1, .combo 5 2 1 1,
    .tidy 6, .close 7]),
  ("cone_ground_false", [G (-1) []], [.load 0, .tidy 0, .close 1]),
  ("oserial_a",
    [E 0 [1, 0, (-1), 0], E 0 [0, 1, 0, (-1)], G (-1) [1, 1, (-1), (-1)]],
    [.load 0, .neg 0, .load 1, .neg 2, .load 2, .combo 4 1 1 1, .combo 5 3 1 1, .tidy 6,
    .close 7]),
  ("oserial_b",
    [E 0 [1, 0, (-1), 0], E 0 [0, 1, 0, (-1)], G (-1) [(-1), (-1), 1, 1]],
    [.load 0, .load 1, .load 2, .combo 0 2 1 1, .combo 1 3 1 1, .tidy 4, .close 5]),
  ("ogeom", [G 0 [1], G (-1) [0]], [.load 1, .tidy 0, .close 1]),
  ("omlp",
    [G (-1) [1, 0], G 0 [0, 1], G 0 [(-1), 0]],
    [.load 0, .load 2, .combo 0 1 1 1, .tidy 2, .close 3]),
  ("cross_off_cone",
    [G (-2) [1], G 1 [(-1)]],
    [.load 0, .load 1, .combo 0 1 1 1, .tidy 2, .close 3]),
  ("cut_2x_eq_1",
    [E (-1) [2]],
    [.load 0, .neg 0, .tidy 0, .tidy 1, .neg 3, .combo 2 4 1 1, .tidy 5, .close 6]),
  ("cut_3x_eq_1",
    [E (-1) [3]],
    [.load 0, .neg 0, .tidy 0, .tidy 1, .neg 3, .combo 2 4 1 1, .tidy 5, .close 6]),
  ("cut_6x_eq_4",
    [E (-4) [6]],
    [.load 0, .neg 0, .tidy 0, .tidy 1, .neg 3, .combo 2 4 1 1, .tidy 5, .close 6]),
  ("eq_pair_unsat",
    [E 0 [1, (-1)], G (-3) [1, 0], G 2 [0, (-1)]],
    [.load 0, .neg 0, .load 1, .load 2, .combo 2 1 1 1, .combo 4 3 1 1, .tidy 5, .close 6]),
  ("r64_capped",
    [G (-1) [0, (-3)], G 0 [1, (-2)], G (-4) [2, (-1)], G 2 [(-2), 2], G 100 [1, 0], G 100
    [0, 1]],
    [.load 0, .load 2, .load 3, .load 5, .tidy 0, .neg 4, .combo 2 5 1 2, .tidy 6, .neg 7,
    .combo 1 3 1 1, .tidy 9, .combo 1 8 1 2, .combo 2 11 1 2, .tidy 12, .neg 13, .combo 2 10
    1 2, .tidy 15, .combo 1 16 1 1, .tidy 17, .combo 1 14 1 2, .combo 2 19 1 2, .tidy 20,
    .neg 21, .combo 2 18 1 2, .tidy 23, .combo 1 24 1 1, .tidy 25, .combo 1 22 1 2, .combo 2
    27 1 2, .tidy 28, .neg 29, .combo 26 30 1 1, .tidy 31, .close 32])
]

/-- Certificate block 1, printed by the compiled decider.

    : List CertEntry                         [systems with the logs the decider emitted] -/
def certBlock1 : List CertEntry := [
  ("r89_capped",
    [G 0 [(-2), 3], G (-4) [0, 1], G (-5) [(-3), (-3)], G 0 [(-1), 1], G 2 [3, 1], G 100 [1,
    0], G 100 [0, 1]],
    [.load 1, .load 2, .load 4, .load 5, .combo 1 0 1 3, .tidy 4, .neg 5, .combo 2 6 1 3,
    .combo 1 7 1 3, .tidy 8, .neg 9, .combo 1 3 1 3, .tidy 11, .neg 12, .combo 2 13 1 1,
    .tidy 14, .combo 2 10 1 3, .combo 1 16 1 3, .tidy 17, .neg 18, .combo 15 19 1 1, .tidy
    20, .close 21]),
  ("r3", [G (-4) [0], G 3 [(-2)], G 4 [(-1)]], [.load 0, .tidy 0, .close 1]),
  ("r6",
    [G (-4) [3, (-2)], E (-4) [(-1), 0], G (-5) [3, 0], G (-6) [(-2), (-1)], G 3 [(-3),
    (-1)]],
    [.load 1, .load 2, .tidy 1, .combo 2 0 1 1, .tidy 3, .close 4]),
  ("r7",
    [G (-1) [2, 1, 0], G (-3) [(-2), 1, (-3)], G 2 [(-3), 1, 1], G 5 [0, 0, (-2)], E 2
    [(-2), 3, (-1)]],
    [.load 0, .load 1, .load 2, .load 4, .neg 3, .combo 2 1 3 1, .combo 4 1 3 1, .tidy 6,
    .combo 0 7 2 1, .tidy 8, .combo 5 7 1 2, .neg 10, .tidy 11, .neg 12, .combo 9 13 1 1,
    .tidy 14, .close 15]),
  ("r8",
    [E (-5) [2, (-3)], G 2 [0, (-3)], E (-6) [(-3), (-1)], G 2 [3, 3], G (-1) [2, (-2)]],
    [.load 0, .neg 0, .load 2, .neg 2, .combo 1 2 1 3, .neg 4, .tidy 5, .neg 6, .combo 3 0 3
    1, .tidy 8, .combo 9 7 1 1, .tidy 10, .close 11]),
  ("r26",
    [G (-2) [3, 2], G 6 [(-2), (-1)], E (-2) [1, 2], G (-2) [0, 0]],
    [.load 3, .tidy 0, .close 1]),
  ("r34",
    [G (-6) [(-1), 2], G 0 [(-2), 1], G 2 [(-3), 2], G 2 [3, 0], E 4 [(-1), (-2)]],
    [.load 0, .load 3, .load 4, .combo 1 0 1 3, .tidy 3, .combo 1 2 1 3, .neg 5, .tidy 6,
    .neg 7, .combo 4 8 1 1, .tidy 9, .close 10]),
  ("r35",
    [G (-2) [2], G 1 [(-1)], G (-3) [(-2)], G (-1) [0], G (-5) [1]],
    [.load 3, .tidy 0, .close 1]),
  ("r40",
    [G (-2) [(-3)], G 0 [(-3)], G (-1) [0], G 4 [(-3)], G (-6) [(-2)]],
    [.load 2, .tidy 0, .close 1]),
  ("r41",
    [G (-1) [(-2), (-2)], G (-2) [(-1), 0], G (-4) [0, 3], G (-4) [2, 2]],
    [.load 0, .load 3, .combo 1 0 1 1, .tidy 2, .close 3]),
  ("r43",
    [G (-1) [2, (-2)], G (-1) [1, 0], G (-2) [0, (-2)], G 5 [3, 2], E 3 [(-2), 2]],
    [.load 1, .load 2, .load 4, .tidy 1, .neg 3, .combo 2 4 1 2, .tidy 5, .neg 6, .combo 0 7
    1 1, .tidy 8, .close 9]),
  ("r51",
    [G 6 [(-1), 3], G 3 [1, 0], E (-1) [(-3), (-3)], E 1 [(-3), 3]],
    [.load 2, .neg 0, .load 3, .neg 2, .combo 1 2 1 1, .tidy 4, .combo 3 0 1 1, .neg 6,
    .tidy 7, .neg 8, .combo 5 9 1 1, .tidy 10, .close 11]),
  ("r61",
    [E 0 [3, (-1), 0], E 4 [0, (-2), (-2)], G 5 [(-3), 3, 0], G (-5) [(-3), 2, (-1)], E 5
    [3, (-1), (-2)]],
    [.load 0, .load 1, .load 2, .load 3, .load 4, .neg 4, .combo 2 0 1 3, .tidy 6, .combo 3
    1 1 1, .neg 8, .tidy 9, .neg 10, .combo 5 0 1 1, .tidy 12, .combo 11 7 1 1, .combo 13 14
    1 1, .tidy 15, .close 16]),
  ("r67", [G (-1) [2], G (-2) [3], G (-2) [0], G (-5) [0]], [.load 2, .tidy 0, .close 1]),
  ("r73",
    [G (-5) [(-1), 1, 0], G (-6) [0, 3, 2], E 1 [(-3), (-3), (-2)], E 1 [2, 2, (-3)]],
    [.load 2, .neg 0, .load 3, .neg 2, .combo 1 3 2 3, .tidy 4, .combo 2 0 3 2, .neg 6,
    .tidy 7, .neg 8, .combo 5 9 1 1, .tidy 10, .close 11]),
  ("r75",
    [G 0 [3, 3], E 1 [3, (-1)], G (-2) [(-2), (-3)], G 3 [(-1), (-2)]],
    [.load 0, .load 1, .neg 1, .load 2, .combo 0 3 1 1, .combo 2 3 3 1, .neg 5, .tidy 6,
    .neg 7, .combo 4 8 1 1, .tidy 9, .close 10]),
  ("r80",
    [G (-3) [(-2), 2], E (-2) [2, 1], E 1 [3, 3], G 0 [2, (-2)]],
    [.load 0, .load 3, .combo 0 1 1 1, .tidy 2, .close 3]),
  ("r81", [G (-4) [0, 0]], [.load 0, .tidy 0, .close 1]),
  ("r85",
    [G 3 [(-1), (-3), (-3)], G 3 [(-2), 3, (-3)], E 5 [(-3), (-3), (-3)], E (-1) [(-1),
    (-2), (-2)]],
    [.load 2, .neg 0, .load 3, .neg 2, .combo 1 2 1 3, .neg 4, .tidy 5, .neg 6, .combo 3 0 3
    1, .tidy 8, .combo 9 7 1 1, .tidy 10, .close 11]),
  ("r86",
    [G 0 [0], E (-2) [1], G (-1) [(-1)], G 0 [2]],
    [.load 1, .load 2, .combo 0 1 1 1, .tidy 2, .close 3])
]

/-- Certificate block 2, printed by the compiled decider.

    : List CertEntry                         [systems with the logs the decider emitted] -/
def certBlock2 : List CertEntry := [
  ("r91",
    [E (-3) [(-2)]],
    [.load 0, .neg 0, .tidy 0, .neg 2, .tidy 1, .combo 4 3 1 1, .tidy 5, .close 6]),
  ("r96",
    [G 2 [3], G (-4) [1], G 0 [0], E (-5) [(-3)]],
    [.load 1, .load 3, .tidy 1, .neg 2, .combo 0 3 1 1, .tidy 4, .close 5]),
  ("r103",
    [E 5 [2, (-2), 2], G 4 [(-1), 3, (-2)], G (-1) [1, (-2), (-2)], G (-5) [3, (-1), (-2)],
    G (-1) [(-3), 1, 2]],
    [.load 3, .load 4, .combo 0 1 1 1, .tidy 2, .close 3]),
  ("r105", [E 0 [(-1)], G 6 [(-3)], G (-6) [0]], [.load 2, .tidy 0, .close 1]),
  ("r108",
    [G 6 [(-3), 1], G (-4) [3, (-2)], E 0 [(-1), (-1)], G 0 [2, 2], G (-5) [1, 0]],
    [.load 0, .load 1, .load 2, .load 4, .combo 0 3 1 3, .combo 1 4 1 2, .tidy 5, .combo 2 4
    1 1, .combo 6 7 1 1, .tidy 8, .close 9]),
  ("r113",
    [G 4 [1, 0], E 5 [2, 0]],
    [.load 1, .neg 0, .tidy 0, .tidy 1, .neg 3, .combo 2 4 1 1, .tidy 5, .close 6]),
  ("r122",
    [E 1 [3], G (-5) [2], E 6 [(-3)]],
    [.load 0, .neg 0, .tidy 0, .tidy 1, .neg 3, .combo 2 4 1 1, .tidy 5, .close 6]),
  ("r134",
    [G (-2) [0, 3], E (-5) [(-2), (-3)], G 4 [1, (-2)], G (-1) [3, 1], E 3 [(-1), 3]],
    [.load 0, .load 1, .load 2, .tidy 0, .combo 1 3 1 3, .tidy 4, .neg 5, .combo 2 6 1 1,
    .tidy 7, .neg 8, .combo 3 9 1 1, .tidy 10, .close 11]),
  ("r137",
    [G 3 [3, 0, 2], G (-5) [2, 2, (-1)], G (-2) [1, (-2), (-2)], G (-3) [(-3), 3, 2]],
    [.load 0, .load 2, .load 3, .combo 2 1 2 3, .combo 0 3 1 1, .tidy 4, .close 5]),
  ("r138",
    [G (-4) [(-1), 0], E 4 [(-2), (-2)], G 1 [3, (-1)], E 6 [0, 0]],
    [.load 3, .neg 0, .tidy 1, .close 2]),
  ("r141",
    [G 3 [1], G 0 [(-2)], G (-5) [3]],
    [.load 1, .load 2, .tidy 0, .neg 2, .tidy 1, .combo 4 3 1 1, .tidy 5, .close 6]),
  ("r144",
    [E 1 [0, 3, 0]],
    [.load 0, .neg 0, .tidy 0, .tidy 1, .neg 3, .combo 2 4 1 1, .tidy 5, .close 6]),
  ("r148", [G 0 [(-3)], E (-3) [0], E (-5) [0], E 6 [(-2)]], [.load 1, .tidy 0, .close 1]),
  ("r153",
    [G (-1) [3, (-1)], G (-5) [(-2), (-2)], G (-3) [0, 1], G (-5) [(-3), 2]],
    [.load 0, .load 1, .load 2, .combo 0 2 1 1, .tidy 3, .combo 1 2 1 2, .tidy 5, .neg 6,
    .combo 4 7 1 1, .tidy 8, .close 9]),
  ("r154", [E (-6) [0], E (-1) [2], G (-2) [3], E (-3) [3]], [.load 0, .tidy 0, .close 1]),
  ("r157", [G 1 [(-2)], G 6 [0], G (-5) [(-1)], G (-5) [0]], [.load 3, .tidy 0, .close 1]),
  ("r159",
    [G (-1) [2, 3], G (-4) [(-2), (-3)]],
    [.load 0, .load 1, .combo 0 1 1 1, .tidy 2, .close 3]),
  ("r167",
    [G (-3) [1, (-1)], G 1 [0, 1], G 0 [(-2), 0], G 1 [3, (-3)]],
    [.load 0, .load 1, .load 2, .tidy 2, .neg 3, .combo 0 1 1 1, .combo 5 4 1 1, .tidy 6,
    .close 7]),
  ("r171",
    [G 6 [0, 2], E 6 [(-3), 0], G (-6) [(-1), (-2)]],
    [.load 0, .load 1, .neg 1, .load 2, .tidy 0, .tidy 2, .combo 3 4 1 2, .combo 5 6 1 1,
    .tidy 7, .close 8]),
  ("r176",
    [E 5 [0, 2], G (-1) [1, (-3)], G 5 [(-1), 0], G (-4) [2, 1]],
    [.load 0, .neg 0, .tidy 0, .tidy 1, .neg 3, .combo 2 4 1 1, .tidy 5, .close 6])
]

/-- Certificate block 3, printed by the compiled decider.

    : List CertEntry                         [systems with the logs the decider emitted] -/
def certBlock3 : List CertEntry := [
  ("r177",
    [G (-6) [(-1), 2, (-2)], E 0 [(-3), 0, 2], G 0 [(-2), (-2), 0], G 1 [(-2), 1, 2], G 3
    [3, (-2), 1]],
    [.load 0, .load 1, .neg 1, .load 2, .load 3, .load 4, .combo 0 3 1 1, .combo 4 5 2 1,
    .combo 2 6 1 1, .neg 8, .tidy 9, .neg 10, .combo 2 7 1 3, .tidy 12, .combo 13 11 1 1,
    .tidy 14, .close 15]),
  ("r179",
    [E 4 [2], G 5 [(-2)], G (-6) [3], G 3 [(-1)], G (-1) [(-2)]],
    [.load 0, .neg 0, .load 2, .tidy 1, .neg 3, .tidy 2, .combo 5 4 1 1, .tidy 6, .close 7]),
  ("r183", [G 4 [2, 3], G 2 [(-3), 0], G (-3) [0, 0]], [.load 2, .tidy 0, .close 1]),
  ("r185",
    [G 3 [3], G (-5) [(-1)], G (-6) [(-3)], G (-2) [1], G (-6) [(-1)]],
    [.load 0, .load 1, .tidy 0, .combo 2 1 1 1, .tidy 3, .close 4]),
  ("r186",
    [E (-3) [(-3), 0], G (-2) [(-1), 1], G 5 [3, (-1)], G 1 [3, (-1)]],
    [.load 0, .neg 0, .load 1, .load 3, .tidy 0, .neg 4, .tidy 1, .combo 2 6 1 1, .combo 3 7
    1 1, .tidy 8, .combo 9 5 1 1, .tidy 10, .close 11]),
  ("r191",
    [G 6 [3], E (-5) [(-1)], G 1 [1]],
    [.load 0, .load 1, .tidy 0, .combo 2 1 1 1, .tidy 3, .close 4]),
  ("r194",
    [E (-2) [(-1), 2], E 2 [(-3), (-3)], G 0 [(-3), (-1)], G (-2) [0, (-2)], G (-6) [(-1),
    (-1)]],
    [.load 0, .load 1, .neg 1, .load 3, .tidy 3, .neg 4, .combo 0 5 1 2, .combo 2 5 1 3,
    .tidy 7, .combo 8 6 1 1, .tidy 9, .close 10]),
  ("r197",
    [G (-1) [1], G (-1) [(-2)]],
    [.load 0, .load 1, .tidy 1, .neg 2, .combo 0 3 1 1, .tidy 4, .close 5]),
  ("r200",
    [G (-3) [3], G 0 [(-1)], G 5 [1]],
    [.load 0, .load 1, .tidy 0, .combo 2 1 1 1, .tidy 3, .close 4]),
  ("r205", [G 4 [1], G (-4) [0]], [.load 1, .tidy 0, .close 1]),
  ("r208",
    [G 2 [2, (-2)], E 2 [3, 0], E 5 [0, 2], G (-2) [0, (-3)]],
    [.load 2, .neg 0, .tidy 0, .tidy 1, .neg 3, .combo 2 4 1 1, .tidy 5, .close 6]),
  ("r209",
    [G (-2) [2], E (-5) [(-2)], E 2 [(-2)]],
    [.load 0, .load 1, .tidy 0, .tidy 1, .neg 3, .combo 2 4 1 1, .tidy 5, .close 6]),
  ("r213",
    [G 0 [0], E 1 [(-1)], G 0 [2], G (-5) [(-3)]],
    [.load 1, .neg 0, .load 3, .tidy 2, .neg 3, .combo 1 4 1 1, .tidy 5, .close 6]),
  ("r214",
    [E 1 [0, (-2)], G 5 [(-3), 1]],
    [.load 0, .neg 0, .tidy 0, .neg 2, .tidy 1, .combo 4 3 1 1, .tidy 5, .close 6]),
  ("r215",
    [G (-3) [0, (-2)], G 1 [0, 1], G (-2) [0, (-2)]],
    [.load 0, .load 1, .tidy 0, .neg 2, .combo 1 3 1 1, .tidy 4, .close 5]),
  ("r217",
    [G 4 [0, (-1)], G (-2) [(-3), (-3)], E (-1) [2, (-3)], E (-1) [3, 3], G 1 [0, (-1)]],
    [.load 1, .load 2, .neg 1, .load 3, .load 4, .combo 2 4 1 3, .tidy 5, .neg 6, .combo 3 7
    1 3, .tidy 8, .combo 0 9 1 3, .tidy 10, .neg 11, .combo 1 12 1 2, .tidy 13, .neg 14,
    .combo 2 15 1 3, .tidy 16, .neg 17, .combo 3 15 1 3, .tidy 19, .combo 20 18 1 1, .tidy
    21, .close 22]),
  ("r218",
    [G (-4) [2], G 3 [1], E 6 [1], G 6 [1]],
    [.load 0, .load 2, .neg 1, .tidy 0, .combo 3 2 1 1, .tidy 4, .close 5]),
  ("r219",
    [G 1 [3], E (-6) [1], G 4 [1], G 1 [(-2)], G 3 [(-2)]],
    [.load 1, .load 3, .tidy 1, .neg 2, .combo 0 3 1 1, .tidy 4, .close 5])
]

/-- Every certificate the compiled decider printed on this corpus.

    : List CertEntry                         [the whole certificate corpus] -/
def certCorpus : List CertEntry :=
  certBlock0 ++ certBlock1 ++ certBlock2 ++ certBlock3

/-- Certificate block 0 replays, checked by the kernel.

    List.all certBlock0 replays = true     [every certificate in the block is accepted] -/
theorem cert_block0 : List.all certBlock0 replays = true := by decide

/-- Certificate block 1 replays, checked by the kernel.

    List.all certBlock1 replays = true     [every certificate in the block is accepted] -/
theorem cert_block1 : List.all certBlock1 replays = true := by decide

/-- Certificate block 2 replays, checked by the kernel.

    List.all certBlock2 replays = true     [every certificate in the block is accepted] -/
theorem cert_block2 : List.all certBlock2 replays = true := by decide

/-- Certificate block 3 replays, checked by the kernel.

    List.all certBlock3 replays = true     [every certificate in the block is accepted] -/
theorem cert_block3 : List.all certBlock3 replays = true := by decide

/-- **Every certificate the decider printed replays here.**

    List.all certCorpus replays = true       [the checker accepts all 78 of them]

    So: with `checkCert_sound`, each of these systems is refuted by a proof that reads only the
    log and the system it names. -/
theorem cert_replay_all : List.all certCorpus replays = true := by
  simp [certCorpus, List.all_append, cert_block0, cert_block1, cert_block2, cert_block3]

/-- A certificate with one multiplier altered is refused.

    checkCert (fixture system) [] (tampered log) = false
                                             [the checker rejects the altered log]

    So: acceptance is not vacuous. The log is `ground_unsat`'s, with its closing operation
    pointed at a row the replay never built. -/
theorem cert_rejects_tampered :
    checkCert [G (-1) [0, 0]] []
      [.load 0, .tidy 0, .close 99] = false
     := by decide

/-- A certificate with its closing operation removed is refused.

    checkCert (fixture system) [] (truncated log) = false
                                             [the checker rejects a log that never closes]

    So: the replay must end on an impossible row, and a log that stops short is not a
    refutation. -/
theorem cert_rejects_truncated :
    checkCert [G (-1) [0, 0]] []
      [.load 0, .tidy 0] = false
     := by decide

/-- A certificate with one combination multiplier altered is refused.

    checkCert (fixture system) [] (altered log) = false
                                             [the checker rejects the altered log]

    So: the log is `fm_unsat`'s, which the compiled decider printed as
    `r 0 / r 1 / r 2 / c 2 1 1 1 / c 0 3 1 1 / t 4 / x 5`, with the first combination's second
    multiplier moved from one to two. The replay still runs to the end and the row it closes on
    is a different row, one that is not impossible, so the checker refuses. Changing a
    multiplier does not make a certificate unsound, it makes it fail to close, which is the
    behaviour that matters: the checker recomputes rather than trusts. -/
theorem cert_rejects_altered_multiplier :
    checkCert [G (-1) [1, 0, 0, 0], G (-1) [0, 1, 0, 0], G 1 [(-1), (-1), 0, 0]] []
      [.load 0, .load 1, .load 2, .combo 2 1 1 2, .combo 0 3 1 1, .tidy 4, .close 5]
      = false := by decide

/-- The unaltered form of the same certificate is accepted.

    checkCert (fixture system) [] (the printed log) = true
                                             [the checker accepts what the decider printed]

    So: the control above fails for the reason claimed and not because the fixture was broken. -/
theorem cert_accepts_unaltered :
    checkCert [G (-1) [1, 0, 0, 0], G (-1) [0, 1, 0, 0], G 1 [(-1), (-1), 0, 0]] []
      [.load 0, .load 1, .load 2, .combo 2 1 1 1, .combo 0 3 1 1, .tidy 4, .close 5]
      = true := by decide

-- The counts, printed at build time: how many certificates, and how many operations.
#eval certCorpus.length
#eval (certCorpus.map (fun e => e.2.2.length)).sum

end Decider
end Apeiron
