import Lean

/-!
# The notation registry

One canonical meaning per symbol. `Apeiron.Linter.Notation` checks every declaration's
statement against this list, so it is the single source of truth: a symbol cannot mean one
thing in one file and something else in another.

Knuth, *Mathematical Writing*, rule 14: "Don't use the same notation for two different
things. Conversely, use consistent notation for the same thing when it appears in several
places." Halmos: "Consistency ... is a major virtue and its opposite is a cardinal sin in
exposition."

`tag` is written for a reader who has not seen the symbol before. No jargon inside a tag; a
term that needs explaining gets its own entry.

`token` entries match on word boundaries, so `mul` does not fire inside `Trichotomy.mul` or
`normForm_mul`. `chars` entries match as substrings, which is what operators need.

## Contested symbols

This corpus draws on fields that disagree about notation, and a symbol meaning one thing in
algebra and another in physics is a defect rather than two conventions. `AGENTS.md` §6a carries
the rule. Each row below is a collision found in a source this corpus cites, the decision, and
the reason. A decision without a reason gets re-litigated.

**`norm`, resolved.** Was `z.1² − σ z.2²`, which is negative at `σ = 1`, `z = (0, 1)`. Mathlib
proves `norm_nonneg`, so the word already meant something incompatible in the library this
corpus imports. Renamed `normForm`: the norm form of a quadratic algebra is indefinite by
nature, and that is what a number theorist greps for.

**`σ`, kept.** Contested with Pauli matrices, standard deviation, a σ-algebra and stress. Kept
because it is the only symbol that ranges over all three faces, and the discriminant reading is
the honest one.

**`ϵ`, refused for σ.** It means the dual-number generator, an arbitrarily small
quantity, the Levi-Civita symbol, machine epsilon and the empty string, depending on the field.
`ϵ` is a generator with `ϵ² = 0`; σ is the parameter a generator squares to. `i`, `ϵ` and
`j` are what `σ` specialises to at the three faces, so they name different objects and must not
share a symbol.

**`S`, unresolved.** The duplicating combinator here, and entropy in `S = Area/4G`.

**`K`, unresolved.** The discarding combinator here, and Kaucher's `𝕂ℝ`, which this corpus
already cites. Two meanings in one corpus today.

**`T`, unresolved.** A transport law here, and the stress tensor in every conformal field theory
paper read here.

**`Ω`, taken from Ramallo, meaning fixed.** The conformal factor of the line element, his
equation (8). Elsewhere it is solid angle, the ohm, the sample space of a probability, and the
lower bound in complexity theory. None of those are in scope here and the holographic reading is
the one the corpus cites, so `Ω` is the factor here.

**`L`, contested and undecided.** The length unit of the line element here, which is Ramallo's
`AdS` radius. It is also `L_n` in the Galilean conformal algebra of Bagchi, Basu, Grumiller and
Riegler, which `theory/LITERATURE.md` §G lists as the next thing to formalise. Both meanings
would be in the same file. Decide before that arrives.

**`c`, reserve now.** Unused here; the central charge `c_L`, `c_M` and the speed of light in the
sources. Decide before the physics arrives.

Unresolved means the collision is recorded and the decision is not yet made. Making it before
the second meaning arrives is cheaper than after.
-/

namespace Apeiron.Linter

/-- How a registry entry is matched against the text of a statement. -/
inductive MatchKind
  /-- Match on word boundaries. For identifiers. -/
  | token
  /-- Match as a substring. For operators and single characters. -/
  | chars
  deriving BEq, Repr, Inhabited

/-- One symbol, and the plain-language meaning that must accompany it. -/
structure Entry where
  /-- The symbol as it appears in source. -/
  symbol : String
  /-- The bracketed meaning a reader gets. -/
  tag : String
  /-- Whether the symbol is standard mathematics or this library's own vocabulary. -/
  corpus : Bool := true
  /-- How to match it. -/
  kind : MatchKind := .chars
  deriving Repr, Inhabited

/-- Standard mathematical notation, glossed once here and once at each use. -/
def ambient : List Entry := [
  ⟨"∃", "there is at least one", false, .chars⟩,
  ⟨"∀", "for every single one", false, .chars⟩,
  ⟨"↔", "exactly when, true in both directions", false, .chars⟩,
  ⟨"→", "takes the left and gives the right; between statements, if the left holds so " ++
    "does the right", false, .chars⟩,
  ⟨"∧", "both are true", false, .chars⟩,
  ⟨"∨", "at least one is true", false, .chars⟩,
  ⟨"¬", "this is not true", false, .chars⟩,
  ⟨"≠", "is not equal to", false, .chars⟩,
  ⟨"≤", "is smaller than or equal to", false, .chars⟩,
  ⟨"<", "is smaller than", false, .chars⟩,
  ⟨"∈", "is one of the things in", false, .chars⟩,
  ⟨"ℝ", "the real numbers: every number on the number line", false, .chars⟩,
  ⟨"ℂ", "the complex numbers: pairs of reals where the second squares to −1", false, .chars⟩,
  ⟨"ℕ", "the counting numbers 0, 1, 2, 3, and so on", false, .chars⟩,
  ⟨"×", "paired with, so ℝ × ℝ means two real numbers side by side", false, .chars⟩,
  ⟨"•", "stretch by, a number times a point", false, .chars⟩,
  ⟨"≃ₗ", "can be relabelled into, keeping addition and stretching intact", false, .chars⟩,
  ⟨"⁻¹", "one divided by", false, .chars⟩,
  ⟨"Real.sqrt", "the number that squares to give this", false, .token⟩,
  ⟨"Module.finrank", "how many independent directions the space has", false, .token⟩,
  ⟨"Basis", "a set of directions every point is built from, exactly one way", false, .token⟩,
  ⟨"Complex.normSq", "the size of a complex number, squared", false, .token⟩,
  ⟨"IsUnit", "can be divided by, meaning something multiplies with it to give 1", false, .token⟩,
  ⟨"Prod.fst", "the first of the pair", false, .token⟩,
  ⟨"Prod.snd", "the second of the pair", false, .token⟩,
  ⟨"Submodule.span", "everything reachable by stretching and adding the given points",
    false, .token⟩,
  ⟨"algebraMap", "read a plain number as a point of the algebra", false, .token⟩,
  ⟨"Bool", "true or false, nothing else", false, .token⟩,
  ⟨"List", "a finite sequence of things, in order", false, .token⟩,
  ⟨"AdjoinRoot", "the ring left after imposing one polynomial relation", false, .token⟩,
  ⟨"root", "the element the imposed relation is about", false, .token⟩,
  ⟨"Module.Basis", "a set of directions every element is built from, exactly one way",
    false, .token⟩,
  ⟨"++", "the two lists joined, the left one's entries first", false, .chars⟩,
  ⟨"Perm", "is a rearrangement of", false, .chars⟩,
  ⟨"flatten", "the inner lists joined into one, in order", false, .chars⟩,
  ⟨"reverse", "the same list back to front", false, .chars⟩,
  ⟨"headD", "the first element, or the given default if there is none", false, .chars⟩,
  ⟨"getLastD", "the last element, or the given default if there is none", false, .chars⟩,
  ⟨"∫", "added up continuously over", false, .chars⟩,
  ⟨"deriv", "how fast the thing is changing, at that point", false, .token⟩,
  ⟨"HasDerivAt", "is changing at that rate at that time", false, .token⟩,
  ⟨"Real.log", "the power e must be raised to for this", false, .token⟩,
  ⟨"IntervalIntegrable", "can be added up over that stretch", false, .token⟩,
  ⟨"Set.uIcc", "every number between the two, either order", false, .token⟩,
  ⟨"volume", "ordinary length, as a way of adding up", false, .token⟩
]

/-- This library's own vocabulary. -/
def corpusEntries : List Entry := [
  ⟨"σ", "the parameter: what the adjoined direction squares to, below zero, zero, or above",
    true, .chars⟩,
  ⟨"Cell", "a point: two real numbers written side by side, a + b·x", true, .token⟩,
  ⟨"mul", "multiply two points together, the σ way", true, .token⟩,
  ⟨"normForm", "the norm form: a point times its conjugate, which can come out negative",
    true, .token⟩,
  ⟨"form", "how much two points pull on each other, measured the σ way", true, .token⟩,
  ⟨"HasNullPair", "holds two directions of size zero that still pull on each other",
    true, .token⟩,
  ⟨"classification", "the proof that every two-direction algebra is one of the three",
    true, .token⟩,
  ⟨"pairBasis", "the two directions 1 and ω, used as the grid to write points on", true, .token⟩,
  ⟨"Idempotent", "unchanged when multiplied by itself, Boole's x² = x", true, .token⟩,
  ⟨"idemPlus", "the point (1/2, 1/(2√σ)), unchanged by squaring when σ is above zero",
    true, .token⟩,
  ⟨"idemMinus", "the same point with the second slot negated", true, .token⟩,
  ⟨"upper", "how far a point lies along the first direction of size zero", true, .token⟩,
  ⟨"lower", "how far a point lies along the other direction of size zero", true, .token⟩,
  ⟨"ofHalves", "the point having the two given coordinates", true, .token⟩,
  ⟨"InCell", "the numbers a point stands for, between its two channel readings", true, .token⟩,
  ⟨"Crossing", "the lower reading has overtaken the upper one, so the interval is empty",
    true, .token⟩,
  ⟨"meet", "the point holding the tighter of each channel, so the two intervals intersected",
    true, .token⟩,
  ⟨"boolVal", "a truth value as a number: one for true, zero for false", true, .token⟩,
  ⟨"cellOfBool", "a truth value carried by the algebra: true is one corner, false the other",
    true, .token⟩,
  ⟨"fold", "multiply every point in a list together, starting from the unit", true, .token⟩,
  ⟨"proj", "multiply a point by a fixed point, keeping some channels and emptying others",
    true, .token⟩,
  ⟨"Discards", "some projection kills a non-zero point while keeping another", true, .token⟩,
  ⟨"Comb", "a term: two constants, and one term applied to another", true, .token⟩,
  ⟨"Step", "one term rewrites to another in a single step", true, .token⟩,
  ⟨"Reduces", "one term reaches another by any number of steps", true, .token⟩,
  ⟨"Par", "one term reduces to another, contracting any number of redexes at once",
    true, .token⟩,
  ⟨"NormalForm", "no rule applies anywhere inside the term, so nothing rewrites it",
    true, .token⟩,
  ⟨"HMap", "a table: a total function from keys to values, absence being the default",
    true, .token⟩,
  ⟨"insertWith", "put a value at a key, merging with whatever was there", true, .token⟩,
  ⟨"build", "fold the write over a stream of pairs, starting from the default", true, .token⟩,
  ⟨"iota", "the single constant that hands its argument the other two", true, .token⟩,
  ⟨"app", "one term applied to another", true, .token⟩,
  ⟨"var", "a variable, named by a counting number", true, .token⟩,
  ⟨"subst", "replace one named variable by a term", true, .token⟩,
  ⟨"hasVar", "true exactly when the term holds that named variable", true, .token⟩,
  ⟨"bracket", "abstraction: rebuild a term as one that takes the named variable as an " ++
    "argument", true, .token⟩,
  ⟨"SC", "a point in the σ = 1 algebra: two reals written a + b·j", true, .token⟩,
  ⟨"j", "the special direction that squares to give 1, not −1 the way i does", true, .token⟩,
  ⟨"one", "the point (1, 0), which changes nothing when you multiply by it", true, .token⟩,
  ⟨"nullPlus", "the point 1 + j, which has size zero even though it is not zero", true, .token⟩,
  ⟨"nullMinus", "the point 1 − j, the other one with size zero", true, .token⟩,
  ⟨"ePlus", "the point halfway to 1 + j, which gives itself back when multiplied by itself",
    true, .token⟩,
  ⟨"eMinus", "the point halfway to 1 − j, which also gives itself back", true, .token⟩,
  ⟨"Channel", "a read: one number off each point, respecting add, stretch, multiply, and one",
    true, .token⟩,
  ⟨"ratio", "the first coordinate divided by the second: the point read as one fraction",
    true, .token⟩,
  ⟨"ProjPair", "a point of the projective line: two slots, the second off the cone so it can " ++
    "be divided by", true, .token⟩,
  ⟨"dehomUpper", "the pair read as one number on the first channel", true, .token⟩,
  ⟨"dehomLower", "the pair read as one number on the other channel", true, .token⟩,
  ⟨"act", "a two-by-two grid of numbers recombining a pair of points", true, .token⟩,
  ⟨"conj", "the same point with its second slot negated", true, .token⟩,
  ⟨"smul", "a number stretching both slots of a point", true, .token⟩,
  ⟨"Tower", "a tree: a plain function at each leaf, one point packing two subtrees at each branch",
    true, .token⟩,
  ⟨"leaf", "a tree that is a single function", true, .token⟩,
  ⟨"node", "two subtrees packed into the two channels of one point", true, .token⟩,
  ⟨"state", "the number a tree works out to at each input", true, .token⟩,
  ⟨"pack", "the term that hands any argument its two stored terms, in order", true, .token⟩,
  ⟨"TwoSlot", "one container holds two things, and each can be read back on its own",
    true, .token⟩,
  ⟨"rel", "the relation the quotient imposes: the generator squares to σ", true, .token⟩,
  ⟨"toQuot", "send a pair across into the quotient, as first slot plus second times the root",
    true, .token⟩,
  ⟨"replace", "the merge that keeps the new value and discards the old", true, .token⟩,
  ⟨"quotBasis", "the two directions 1 and the generator, spanning the quotient", true, .token⟩,
  ⟨"Ω", "the conformal factor: what the flat bracket is multiplied by at each height",
    true, .chars⟩,
  ⟨"lineElement", "the displacement's size over the point's height squared", true, .token⟩,
  ⟨"genLineElement", "the conformal factor squared, times the displacement's size", true, .token⟩,
  ⟨"ScaleInvariant", "stretching everything by one factor leaves the line element alone",
    true, .token⟩,
  ⟨"mobius", "the numerator times the denominator's reciprocal", true, .token⟩,
  ⟨"det2", "what the grid scales areas by", true, .token⟩,
  ⟨"speed", "the square root of the line element, along a path", true, .token⟩,
  ⟨"arcLength", "the speed, added up over the times", true, .token⟩,
  ⟨"vertical", "the path that holds its first slot and rises", true, .token⟩,
  ⟨"functionDistortion", "the precision-weighted squared residual, over a per-channel energy",
    true, .token⟩,
  ⟨"osFuse", "combines two online-softmax states into one", true, .token⟩,
  ⟨"osVal", "an online-softmax state, to the un-normalized drag it stands for", true, .token⟩,
  ⟨"tileTotal", "a list of tile states, to their total", true, .token⟩,
  ⟨"Joinable", "two terms, to whether they are equal by computation", true, .token⟩,
  ⟨"verdict", "a fuel budget, two terms, to the three-valued answer", true, .token⟩,
  ⟨"normalize", "a fuel budget and a term, to the reached floor", true, .token⟩,
  ⟨"hasRedex", "a term, to whether some rule fires somewhere in it", true, .token⟩,
  ⟨"Precedes", "both channel readings of one point are at or below the other's", true, .token⟩,
  ⟨"rescaleFace", "the second slot stretched so one face's algebra reads as another's",
    true, .token⟩,
  ⟨"Implicative", "an order where every set has a meet, carrying one implication",
    true, .token⟩,
  ⟨"implApply", "applying, read off the order: the meet of every conclusion two elements " ++
    "support", true, .token⟩,
  ⟨"nodeOp", "the branch rule of the tree, read as an operation on two plain numbers",
    true, .token⟩,
  ⟨"Engeler", "a token of the graph model: an atom, or a list of tokens pointing at one",
    true, .token⟩,
  ⟨"feed", "hand the right element to the left one: the targets of arrows whose whole source " ++
    "is held", true, .token⟩,
  ⟨"kPoint", "the discarding constant, as a set of arrows", true, .token⟩,
  ⟨"sPoint", "the duplicating constant, as a set of arrows", true, .token⟩,
  ⟨"Survivor", "a point of the algebra unchanged by squaring", true, .token⟩,
  ⟨"leafList", "the constants and variables of a term, left to right", true, .token⟩,
  ⟨"MulInvariant", "sends applying one term to another to multiplying their points",
    true, .token⟩,
  ⟨"dupFree", "true exactly when the term holds no duplicating constant", true, .token⟩,
  ⟨"discardFree", "true exactly when the term holds no discarding constant", true, .token⟩,
  ⟨"freeCell", "the point carrying both constant-free readings, one on each channel",
    true, .token⟩,
  ⟨"feedback", "go straight, or out round the loop and back: the block with its loop solved",
    true, .token⟩,
  ⟨"loopSum", "one term per trip round the loop, added up to a given depth", true, .token⟩,
  ⟨"cpow", "what n trips round the loop do, multiplied out", true, .token⟩,
  ⟨"affineIter", "the affine loop run that many times: multiply by the weight, add the " ++
    "injection", true, .token⟩,
  ⟨"restPoint", "where the affine loop comes to rest: the injection times the loop's resolvent",
    true, .token⟩,
  ⟨"aitken", "the first of three points, less their first difference squared over their " ++
    "second difference", true, .token⟩,
  ⟨"leftIn", "a wire moved into the first of the two tracks", true, .token⟩,
  ⟨"rightIn", "a wire moved into the second of the two tracks", true, .token⟩,
  ⟨"leftOut", "what sits in the first track of a wire", true, .token⟩,
  ⟨"rightOut", "what sits in the second track of a wire", true, .token⟩,
  ⟨"Route", "a weighted wire map: each position goes nowhere, or to one with one weight",
    true, .token⟩,
  ⟨"rcomp", "follow one route and then another, multiplying the weights", true, .token⟩,
  ⟨"rid", "the route joining every position to itself, weighted by the unit", true, .token⟩,
  ⟨"rnone", "the route that joins nothing", true, .token⟩,
  ⟨"rtensor", "two routes side by side, one on the even positions and one on the odd",
    true, .token⟩,
  ⟨"rpow", "a route composed with itself that many times", true, .token⟩,
  ⟨"inL", "place a wire in the even positions", true, .token⟩,
  ⟨"inR", "place a wire in the odd positions", true, .token⟩,
  ⟨"outL", "read the even positions back", true, .token⟩,
  ⟨"outR", "read the odd positions back", true, .token⟩,
  ⟨"copy", "place a wire in one of countably many disjoint copies", true, .token⟩,
  ⟨"uncopy", "read one of the countably many copies back", true, .token⟩,
  ⟨"TracesTo", "the trace of the first route is exactly the second", true, .token⟩,
  ⟨"rplus", "the first route where it is defined, the second elsewhere", true, .token⟩,
  ⟨"bang", "the same route running inside every one of the countably many copies",
    true, .token⟩,
  ⟨"traceRoute", "the trace of a route, taken as a route", true, .token⟩,
  ⟨"rapp", "plug the argument into the loop and close it: applying one route to another",
    true, .token⟩,
  ⟨"RunsN", "a path of a given length through a route, staying inside a set until it leaves",
    true, .token⟩,
  ⟨"Runs", "a path through a route, staying inside a set until it leaves", true, .token⟩,
  ⟨"runsRoute", "the route whose steps are whole paths through a set", true, .token⟩,
  ⟨"extPos", "a position of the wire outside both loops", true, .token⟩,
  ⟨"outPos", "a position of the loop that faces the argument", true, .token⟩,
  ⟨"innPos", "a position of the loop a stored route keeps to itself", true, .token⟩,
  ⟨"outerSet", "the positions of the loop that faces the argument", true, .token⟩,
  ⟨"innerSet", "the positions of the loop a stored route keeps to itself", true, .token⟩,
  ⟨"capp", "apply a route to the exponential of another, so the argument serves every consumer",
    true, .token⟩,
  ⟨"conjRoute", "a route read through a relabelling of the wire", true, .token⟩,
  ⟨"argPos", "a position of the channel a route faces its argument through", true, .token⟩,
  ⟨"wIdx", "the change of layout a contraction needs, its labels to its argument's",
    true, .token⟩,
  ⟨"dupArg", "the argument a contraction sees: the identity outside, the route on each loop",
    true, .token⟩,
  ⟨"innArgSet", "the positions of the second use of a duplicated argument", true, .token⟩,
  ⟨"zPos", "the place of an outer argument position on the composite's wire", true, .token⟩,
  ⟨"zSet", "the track the composite offers its outer argument", true, .token⟩,
  ⟨"xPos", "the place of the first route's position on the combined wire", true, .token⟩,
  ⟨"yPos", "the place of the second route's position on the combined wire", true, .token⟩,
  ⟨"flipPos", "a position read the other way round: even becomes odd", true, .token⟩,
  ⟨"bSwapFun", "the wire: what one route asks is what the other is asked", true, .token⟩,
  ⟨"swapRoute", "the wire, taken as a route weighted by the unit", true, .token⟩,
  ⟨"pairRoute", "two routes laid side by side on one wire", true, .token⟩,
  ⟨"bStore", "two routes side by side with the wire between them", true, .token⟩,
  ⟨"bComb", "the composite of two routes, on the composite's own wire", true, .token⟩,
  ⟨"bArg", "the outer argument, laid on the fourth track alone", true, .token⟩,
  ⟨"bPlug", "both routes with the outer argument, before the wire", true, .token⟩,
  ⟨"bBig", "both routes, the wire, and the outer argument, as one route", true, .token⟩,
  ⟨"xOut", "where the first route's step lands after the wire", true, .token⟩,
  ⟨"cSwapFun", "exchange the second argument's track with the first argument's", true, .token⟩,
  ⟨"cSwap", "the exchange of the two argument tracks, as a permutation", true, .token⟩,
  ⟨"cStore", "the first route with the second plugged at its second argument", true, .token⟩,
  ⟨"cComb", "the exchanging combinator's stored route", true, .token⟩,
  ⟨"cMach", "the exchanging combinator's machine, carrying both loops", true, .token⟩,
  ⟨"fedRoute", "a machine after the argument's own channel is collapsed", true, .token⟩,
  ⟨"z1Set", "the copy of the argument the first route consumes", true, .token⟩,
  ⟨"z2Set", "the copy of the argument the second route consumes", true, .token⟩,
  ⟨"sxPos", "the first route's place on the substituting combinator's wire", true, .token⟩,
  ⟨"syPos", "the second route's place on the substituting combinator's wire", true, .token⟩,
  ⟨"sPair", "two routes laid on the five tracks, each with its own copy", true, .token⟩,
  ⟨"sStore", "the two routes side by side with the wire between them", true, .token⟩,
  ⟨"sComb", "the substituting combinator's stored route", true, .token⟩,
  ⟨"sPlug", "the pair with the doubled argument, before the wire", true, .token⟩,
  ⟨"sBig", "the substituting combinator's machine, carrying every loop", true, .token⟩,
  ⟨"dRoute", "the route that reads one copy of the exponential", true, .token⟩,
  ⟨"digFun", "read a tag and a tagged position as a pair of tags and a position", true, .token⟩,
  ⟨"digInv", "read a pair of tags and a position as a tag and a tagged position", true, .token⟩,
  ⟨"dig", "the associativity of Cantor's pairing, as a bijection", true, .token⟩,
  ⟨"diggingRoute", "the route that makes a reusable copy of a reusable copy", true, .token⟩,
  ⟨"copySplitFun", "read a position whose payload is tagged as a tagged position", true, .token⟩,
  ⟨"copySplitInv", "read a tagged position as a position whose payload is tagged", true, .token⟩,
  ⟨"copySplit", "the tag read outside or inside, as a bijection", true, .token⟩,
  ⟨"copyPos", "the whole of a wire, embedded in one copy", true, .token⟩,
  ⟨"copyOff", "the loop positions outside one copy", true, .token⟩,
  ⟨"bangShift", "the exponential of a route, read on the wire an application uses", true, .token⟩,
  ⟨"contractSplitFun", "read a side and a tag as a tag which records the side", true, .token⟩,
  ⟨"contractSplitInv", "read a tag which records a side as a side and a tag", true, .token⟩,
  ⟨"contractSplit", "the comultiplication: one reusable copy read as a pair", true, .token⟩,
  ⟨"storeRoute", "the route that stores its argument under a relabelling", true, .token⟩,
  ⟨"argExtFun", "a relabelling of the loop half, doing nothing on the outside", true, .token⟩,
  ⟨"argExt", "a relabelling of the wire acting on the argument channel alone", true, .token⟩,
  ⟨"cRoute", "the exchanging combinator as a constant route", true, .token⟩,
  ⟨"fRoute", "functoriality as a constant route", true, .token⟩,
  ⟨"wRoute", "the contracting combinator as a constant route", true, .token⟩,
  ⟨"bFun", "the composing combinator's wiring, as a map of the counting numbers", true, .token⟩,
  ⟨"bRoute", "the composing combinator as a constant route", true, .token⟩,
  ⟨"bStored", "the first route stored on its tracks, argument channel folded back", true, .token⟩,
  ⟨"yOut", "where the second route's step lands after the wire", true, .token⟩,
  ⟨"sRoute", "the substituting combinator as a term in the basis", true, .token⟩,
  ⟨"kCombinator", "the discarding combinator of the ordinary algebra, dereliction and all",
    true, .token⟩,
  ⟨"sCombinator", "the substituting combinator of the ordinary algebra, promotion and all",
    true, .token⟩,
  ⟨"sPromote", "the piece that makes a copy reusable again", true, .token⟩,
  ⟨"sShare", "the piece that holds the promotion aside", true, .token⟩,
  ⟨"sSplit", "the piece that hands one copy to each half", true, .token⟩,
  ⟨"denote", "a term read as a weighted wire map", true, .token⟩,
  ⟨"linearDenote", "a term read as a wire map with no exponential anywhere", true, .token⟩,
  ⟨"AppTrips", "how many times an application goes round the feedback loop", true, .token⟩,
  ⟨"Row", "a row: a constant plus whole-number multiples of the columns, asserted to be at " ++
    "least zero", true, .token⟩,
  ⟨"Expr", "an expression a law or spec row is allowed to state", true, .token⟩,
  ⟨"denote", "the single row an expression turns into", true, .token⟩,
  ⟨"Holds", "the row's value at this assignment is at least zero", true, .token⟩,
  ⟨"Keyed", "every opaque column holds the value it stands for", true, .token⟩,
  ⟨"opaqueCol", "a nonlinear operation taken as one column, carrying nothing about its " ++
    "operands", true, .token⟩,
  ⟨"crossingRow", "the row saying the bottom endpoint has passed the top one", true, .token⟩,
  ⟨"Box", "every column sits between its declared caps", true, .token⟩,
  ⟨"combo", "a nonnegative-weighted sum of rows", true, .token⟩,
  ⟨"Cmp", "a comparison: at most, below, equal, at least, or above", true, .token⟩,
  ⟨"flow", "the size-one points reached by turning, shearing or boosting for a given time",
    true, .token⟩,
  ⟨"mover", "a profile carried rigidly along one ray of the cone", true, .token⟩,
  ⟨"BalancedChannels", "the turning point sits on the hyperbolic cone, so its two coordinates " ++
    "weigh the same", true, .token⟩,
  ⟨"OnTheMonad", "three times the cone's azimuth is posited to equal its quotient", true, .token⟩,
  ⟨"trinitySum", "three weights placed on the three phases a third of a turn apart",
    true, .token⟩,
  ⟨"MixingIsRootRatio", "the mixing weight is posited to be the ratio of the two masses",
    true, .token⟩,
  ⟨"planeWave", "one travelling mode of fixed frequency and momentum", true, .token⟩,
  ⟨"waveFst", "the first slot of a size times a flow", true, .token⟩,
  ⟨"waveSnd", "the second slot of a size times a flow", true, .token⟩,
  ⟨"curv", "the one point that, multiplied into the flow, is the wave's second derivative",
    true, .token⟩,
  ⟨"koideQuotient", "the three sizes added, over the square of their square roots added",
    true, .token⟩,
  ⟨"dimOf", "how many real directions so many definite blocks and so many cells make",
    true, .token⟩,
  ⟨"koideForm", "twice the square of the three square roots added, less three times their " ++
    "squares added", true, .token⟩,
  ⟨"coneAngleSq", "how squarely the three square roots point along the all-equal direction",
    true, .token⟩,
  ⟨"trinity", "one amplitude sampled at three times a third of a turn apart", true, .token⟩,
  ⟨"weightSpread", "the size of the three-phase sum of three weights, squared", true, .token⟩,
  ⟨"Double", "the Cayley-Dickson double of an algebra: two of its elements side by side",
    true, .token⟩,
  ⟨"ofPair", "read a pair of base elements as one element of the double", true, .token⟩,
  ⟨"doubleNorm", "the doubled quadratic form: the base form of the first slot, less σ " ++
    "times that of the second", true, .token⟩,
  ⟨"Quat", "the real quaternions, four real numbers with Hamilton's product", true, .token⟩,
  ⟨"Octonion", "the octonions: the double of the quaternions at σ = −1",
    true, .token⟩,
  ⟨"SplitOct", "the split octonions: the same double at σ = 1", true, .token⟩,
  ⟨"Sedenion", "the sedenions: the double of the octonions at σ = −1", true, .token⟩,
  ⟨"normO", "the octonion norm form, a sum of eight real squares", true, .token⟩,
  ⟨"normS", "the split octonion norm form, four real squares less four", true, .token⟩,
  ⟨"normSed", "the sedenion norm form, a sum of sixteen real squares", true, .token⟩,
  ⟨"breakLeft", "the first of the two non-zero sedenions whose product is zero", true, .token⟩,
  ⟨"breakRight", "the second of them", true, .token⟩,
  ⟨"IntForm", "an integral symmetric bilinear form, given by its table of pairings",
    true, .token⟩,
  ⟨"sigPos", "how many plus ones a sign vector has", true, .token⟩,
  ⟨"sigNeg", "how many entries of a sign vector are not plus one", true, .token⟩,
  ⟨"hyperbolicPlane", "the rank-two even form whose table of pairings is zero on the diagonal " ++
    "and one off it", true, .token⟩,
  ⟨"nullBasis", "the two directions of size zero of the σ = 1 algebra", true, .token⟩,
  ⟨"hSign", "one plus and one minus, the signs of the rank-two even form", true, .token⟩,
  ⟨"E8", "the rank-eight definite even form, whose table of pairings is the E8 Cartan matrix",
    true, .token⟩,
  ⟨"oCoord", "the eight real coordinates of an octonion", true, .token⟩,
  ⟨"oForm", "the pairing that the octonion norm form is the diagonal of", true, .token⟩,
  ⟨"e8Oct", "the eight simple roots of E8, written as octonions", true, .token⟩,
  ⟨"e8B", "those eight roots written as rows of coordinates", true, .token⟩,
  ⟨"e8GramInv", "an integer matrix undoing the E8 Cartan matrix exactly", true, .token⟩,
  ⟨"evenUnimodular", "so many copies of the rank-eight block and so many of the rank-two one",
    true, .token⟩,
  ⟨"euSign", "the sign vector of that family", true, .token⟩,
  ⟨"EUIdx", "the index set of that family", true, .token⟩,
  ⟨"isSqUpTo", "true when some whole number up to the bound squares to this", true, .token⟩,
  ⟨"renorm", "compose a map with itself and rescale the line", true, .token⟩,
  ⟨"bulk", "the map x over one minus a x, the tangency normal form", true, .token⟩,
  ⟨"gaussStep", "one over the number, less one", true, .token⟩,
  ⟨"IsRenormMultiplier", "the rate at which renormalisation stretches one direction at a " ++
    "fixed point", true, .token⟩,
  ⟨"Bnd", "one column's interval: a bottom and a top, either of which may be absent",
    true, .token⟩,
  ⟨"CRow", "a certificate row: a bound from each side, and one coefficient per column",
    true, .token⟩,
  ⟨"CertOp", "one line of a certificate: read a system row, combine, negate, meet, tidy, close",
    true, .token⟩,
  ⟨"Verdict", "the decider's answer: refuted, satisfied, or not settled", true, .token⟩,
  ⟨"intervalOf", "the point of the algebra whose two readings are the doubled endpoints",
    true, .token⟩,
  ⟨"checkCert", "replay a certificate against a system and say whether it closes",
    true, .token⟩,
  ⟨"Outcome", "one bounded search result: a value found, or the budget run out", true, .token⟩,
  ⟨"Traces", "the union over every path length through a route", true, .token⟩,
  ⟨"direct", "what the route does without entering the loop", true, .token⟩,
  ⟨"entry", "the way from the even wire into the odd one", true, .token⟩,
  ⟨"leave", "the way from the odd wire out to the even one", true, .token⟩,
  ⟨"body", "the way from the odd wire back to the odd wire, one trip round", true, .token⟩,
  ⟨"traceWithin", "the bounded search for the trace, up to a trip budget", true, .token⟩,
  ⟨"runsWithin", "the bounded search for a path, up to a step budget", true, .token⟩,
  ⟨"pointDensity", "the amount: the size of the phase cell at a place", true, .token⟩,
  ⟨"pointFlux", "the first-direction flux: amount times phase difference", true, .token⟩,
  ⟨"ringBasis", "the basis point: the amplitude in the first slot, zero in the second",
    true, .token⟩,
  ⟨"ringWave", "the wave point: an amplitude coefficient on the unit direction times a phase " ++
    "point", true, .token⟩,
  ⟨"ringStencil4_normForm", "the four-sample sum less four times the centre, exact at every " ++
    "step", true, .token⟩,
  ⟨"ringDet2", "the diagonal product less the off-diagonal product, what the grid scales " ++
    "areas by", true, .token⟩,
  ⟨"ringRecip", "the conjugate stretched by an explicit inverse of the point's size",
    true, .token⟩,
  ⟨"RingCell", "a point: two ring elements side by side, a + b·x", true, .token⟩,
  ⟨"ringMul", "multiply two points together, the σ way", true, .token⟩,
  ⟨"ringNormForm", "the norm form: a point times its conjugate, which can come out negative",
    true, .token⟩,
  ⟨"ringOne", "the point (1, 0), which changes nothing when you multiply by it", true, .token⟩,
  ⟨"ringConj", "the same point with its second slot negated", true, .token⟩,
  ⟨"ringIdemPlus", "the point h in the first slot and h·s in the second, an idempotent when " ++
    "the witnesses hold", true, .token⟩,
  ⟨"ringIdemMinus", "the mirror of ringIdemPlus, with the second slot negated", true, .token⟩,
  ⟨"ringCurv", "the curvature point: the second-order data of the amplitude and the phase",
    true, .token⟩,
  ⟨"ringJet", "the finite jet: the wave's second derivative assembled from its coefficient " ++
    "data", true, .token⟩,
  ⟨"ringDoubleNorm", "the doubled quadratic form: the base form of the first slot, less σ " ++
    "times that of the second", true, .token⟩,
  ⟨"ringTrinitySum", "three ring weights placed on the three order-three phases", true, .token⟩,
  ⟨"ringWeightSpread", "the size of the three-phase sum over a ring: the squared length of the " ++
    "amplitude", true, .token⟩,
  ⟨"ringMobiusNum", "the numerator of a Möbius map over a ring, written with no division",
    true, .token⟩,
  ⟨"ringMobiusDen", "the denominator of a Möbius map over a ring, written with no division",
    true, .token⟩,
  ⟨"ringMobiusTangent", "the displacement factor of a Möbius map, with the denominator's " ++
    "inverse carried", true, .token⟩,
  ⟨"RingProjPair", "a point of the projective line over a ring: two slots, the second's size " ++
    "a unit", true, .token⟩,
  ⟨"ringProjScale", "a projective point with both slots stretched by a scalar that can be " ++
    "divided by", true, .token⟩,
  ⟨"ringBasisFaceX", "the first-coordinate x-face read: the positive x-face value leaving an " ++
    "anchor", true, .token⟩,
  ⟨"ringBasisFaceY", "the second-coordinate y-face read: the positive y-face value leaving an " ++
    "anchor", true, .token⟩,
  ⟨"ringSampleDensity3", "the generic point density at one extended sample of a " ++
    "three-dimensional field", true, .token⟩,
  ⟨"ringCarrierDensity3", "the generic density at the logical cell's shifted three-dimensional " ++
    "sample", true, .token⟩,
  ⟨"ringFaceFluxX", "the single-valued forward x-face flux: amount times phase difference",
    true, .token⟩,
  ⟨"RingChannel", "a read of the ring: one element off each point, respecting the operations",
    true, .token⟩,
  ⟨"ringUpper", "how far a point lies along the first direction of size zero, over a ring",
    true, .token⟩,
  ⟨"ringLower", "how far a point lies along the other direction of size zero, over a ring",
    true, .token⟩,
  ⟨"ringOfHalves", "the ring point having the two given coordinates", true, .token⟩,
  ⟨"ringFold", "multiply every point in a list together over a ring, starting from the unit",
    true, .token⟩,
  ⟨"ringBoolVal", "a truth value as a ring element: one for true, zero for false", true, .token⟩,
  ⟨"ringCellOfBool", "a truth value carried by the ring: true is one corner, false the other",
    true, .token⟩,
  ⟨"scalarCarrierStep", "the scalar carrier update: the old value, less the outgoing minus " ++
    "incoming flux, plus the source", true, .token⟩,
  ⟨"DoublesTo", "twice the first value is the second: an explicit witness that a half exists",
    true, .token⟩,
  ⟨"exactHalfFlux", "the exact half diffusive flux: the value that doubles to the forward " ++
    "difference, with no division", true, .token⟩
]

/-- Every registered symbol, longest first so that `Real.sqrt` is tried before `√`. -/
def registry : List Entry :=
  (ambient ++ corpusEntries).toArray.qsort
    (fun a b => a.symbol.length > b.symbol.length) |>.toList

end Apeiron.Linter
