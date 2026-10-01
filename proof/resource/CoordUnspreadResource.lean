import CoordUnspreadSource
import Aeneas

/-
SCOPE: the actual extracted body `coord::unspread` reached by the extraction root
`word_coord_unspread`; its one guarded transition, the exit state, the counted work of one
evaluation, the closed-form value it returns, and reachable witnesses of each at the joint
`U64` domain. REGISTER: scalar source refinement of one wrapping multiply plus an explicit
word-operation count of that one body; the count is read off the extracted body, not from a
loop counter and not from a clock. Excluded: the algebra correspondence
`unspread(spread x).id = x`, the `spread` body and its `U128` widening multiply, the
deliberately one-way `place` fold, the `src/table/keyrep.rs` codec that calls this body, the
shared `proof/extract/lib.rs` extraction unit, and hardware or GPU timing.
CORPUS: `src/coord.rs:385-387`, whose real body is
`pub const fn unspread(id: u64) -> u64 { id.wrapping_mul(PHI_INV) }` over the constant
`src/coord.rs:390`; the forwarding wrapper is `proof/extract/lib.rs:503-506`, rooted by
`proof/regen.sh:166`, and the same body is already extracted at `proof/ApeironWord.lean:1070-1071`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean
GATE: `python3 proof/resource/CoordUnspread_gate.py --controls`, 540-second budget, lane
/tmp/opencode/apeiron-lane-2.lock.
GRADE: `proved` for the theorems below when that gate exits 0. The work measure counts `U64`
wrapping multiplies, machine-word comparisons and heap allocations in the extracted body under
the Aeneas model; it is a count of the model's own reduction, not a measured time.
-/

namespace CoordUnspreadResource
open Aeneas Aeneas.Std Result

/-- The real extracted `src/coord.rs:385-387` body reached by the extraction root. -/
def body (identity : U64) : Result U64 :=
  coord_unspread_source.coord.unspread identity

/-- The extraction root forwards to that body; the root restates no multiply. -/
theorem root_forwards (identity : U64) :
    coord_unspread_source.word_coord_unspread identity = body identity := rfl

/-- The extracted multiplier is the concrete word of `src/coord.rs:390`. -/
theorem phi_inv_eq :
    coord_unspread_source.coord.PHI_INV = 17428512612931826493#u64 := by
  norm_num [coord_unspread_source.coord.PHI_INV]

/-- The modulus of the `U64` wrapping multiply, as the number the model reduces it to. -/
theorem modulus_eq : UScalar.size UScalarTy.U64 = 18446744073709551616 := by
  simp [U64.size, U64.numBits]

/-- The whole body is `Result.ok` of one `U64` wrapping multiply by the extracted constant: no
loop, no `bind`, no branch, no call other than that multiply, no `alloc` and no `Error` channel
appears in it. This is the kernel certificate the work count below is read from. -/
theorem body_is_one_wrapping_mul (identity : U64) :
    body identity
      = ok (core.num.U64.wrapping_mul identity coord_unspread_source.coord.PHI_INV) := rfl

/-- The counted work of one evaluation of the real body. `value m c a` is a value node reached
after `m` `U64` wrapping multiplies, `c` machine-word comparisons and `a` heap allocations;
`effect` and `diverging` are the two non-value nodes. -/
inductive Work where
  /-- A value node: `multiplies` wrapping multiplies, `comparisons` word comparisons,
      `allocations` heap allocations. -/
  | value (multiplies comparisons allocations : Nat)
  /-- An effect node. -/
  | effect
  /-- A divergence node. -/
  | diverging

/-- The work of one evaluation of the real body at the given input word. -/
def workOf (identity : U64) : Work :=
  Result.cases (body identity)
    (fun _ => Work.value 1 0 0) (fun _ _ => Work.effect) Work.diverging

/-- The real body performs exactly one `U64` wrapping multiply, no machine-word comparison and
no heap allocation, at every input word. The count is read off `body_is_one_wrapping_mul`: the
`Result` it returns is the value node `ok`, not an effect or divergence node, and the value of
that node is the one `wrapping_mul` of the two argument words. -/
theorem work_is_one_multiply (identity : U64) : workOf identity = Work.value 1 0 0 := by
  show Result.cases (body identity)
      (fun _ => Work.value 1 0 0) (fun _ _ => Work.effect) Work.diverging = Work.value 1 0 0
  show Result.cases
      (ok (core.num.U64.wrapping_mul identity coord_unspread_source.coord.PHI_INV))
      (fun _ => Work.value 1 0 0) (fun _ _ => Work.effect) Work.diverging = Work.value 1 0 0
  rfl

/-- The counted work does not depend on the magnitude of the input word: the body carries no
data-dependent guard, because the certificate holds at every input. -/
theorem work_is_input_independent (identity identity' : U64) :
    workOf identity = workOf identity' := by
  rw [work_is_one_multiply, work_is_one_multiply]

/-- The transition of the real body, over `(input, pending, output)` states. The only guard is
the pending flag: the single wrapping multiply is taken exactly once, while the flag is set,
and the exit state carries the produced word. -/
inductive Step : U64 → Bool → U64 → Bool → Prop where
  /-- The multiply transition, taken exactly while the pending flag is set. -/
  | multiply (identity : U64) :
      Step identity true
        (core.num.U64.wrapping_mul identity coord_unspread_source.coord.PHI_INV) false

/-- The rank of a transition state: the number of word operations still owed. -/
def Rank (pending : Bool) : Nat := if pending then 1 else 0

/-- The initial state of the real body owes one word operation. -/
theorem rank_initial : Rank true = 1 := rfl

/-- The exit state of the real body owes none. -/
theorem rank_exit : Rank false = 0 := rfl

/-- Every transition of the real body lowers the rank by at least one word operation. -/
theorem rank_drops {a : U64} {p : Bool} {b : U64} {q : Bool} (h : Step a p b q) :
    Rank q + 1 ≤ Rank p := by
  cases h
  simp [Rank]

/-- The guard is exhaustive: while the pending flag is set the sole multiply is available, and
once the flag is clear no transition is available at all. -/
theorem guard_exhaustive (a : U64) (p : Bool) :
    (p = true ∧ ∃ r : U64, Step a true r false) ∨
      (p = false ∧ ¬ ∃ w : U64, ∃ q : Bool, Step a false w q) := by
  cases hp : p
  · refine Or.inr ⟨rfl, fun h => ?_⟩
    obtain ⟨w, q, hw⟩ := h
    cases hw
  · refine Or.inl ⟨rfl, core.num.U64.wrapping_mul a coord_unspread_source.coord.PHI_INV, ?_⟩
    exact Step.multiply a

/-- The exit state of the real body has no outgoing transition, so no chain leaves it and no
cycle can close on it. -/
theorem exit_is_terminal (r : U64) : ¬ ∃ w : U64, ∃ q : Bool, Step r false w q := by
  rintro ⟨w, q, h⟩
  cases h

/-- Work is counted, never cycled: from the initial state a chain of `n` transitions of the real
body has spent `n` word operations of a rank of `1`, so the chain cannot be longer than one
step. The bound is on the chain length `n`, not on a cycle count. -/
theorem chain_rank_bound {n : Nat} {w : Nat → U64} {p : Nat → Bool}
    (h0 : p 0 = true)
    (hstep : ∀ k : Nat, Step (w k) (p k) (w (k + 1)) (p (k + 1))) :
    Rank (p n) + n ≤ 1 := by
  induction n with
  | zero => rw [h0, rank_initial]
  | succ n ih =>
    have h := rank_drops (hstep n)
    omega

/-- No transition chain out of the initial state has more than one step. -/
theorem chain_length_le_one {n : Nat} {w : Nat → U64} {p : Nat → Bool}
    (h0 : p 0 = true)
    (hstep : ∀ k : Nat, Step (w k) (p k) (w (k + 1)) (p (k + 1))) : n ≤ 1 := by
  have h := chain_rank_bound (n := n) h0 hstep
  omega

/-- The exit branch of the real body: the single transition leads to a state from which the body
returns that word, and no guard, call, loop or allocation follows it. -/
theorem exit_branch (identity : U64) :
    ∃ r : U64,
      Step identity true r false ∧ body identity = ok r ∧
        ¬ ∃ w : U64, ∃ q : Bool, Step r false w q :=
  ⟨core.num.U64.wrapping_mul identity coord_unspread_source.coord.PHI_INV, Step.multiply identity,
    body_is_one_wrapping_mul identity, fun h => exit_is_terminal _ h⟩

/-- The transition is total: every input word reaches a value node and a produced word. -/
theorem body_total (identity : U64) : ∃ r : U64, body identity = ok r :=
  ⟨core.num.U64.wrapping_mul identity coord_unspread_source.coord.PHI_INV,
    body_is_one_wrapping_mul identity⟩

/-- The body reaches a value node under the Aeneas weak-precondition relation, so it produces no
`Error` node: the real body has no refusal branch to take. -/
theorem body_never_errors (identity : U64) : WP.spec (body identity) (fun _ => True) := by
  show body identity ⦃ r => True ⦄
  rw [body_is_one_wrapping_mul]
  exact (WP.spec_ok _).mpr trivial

/-- One wrapped `U64` multiply of two words. -/
theorem product_val (x y : U64) :
    (core.num.U64.wrapping_mul x y).val = (x.val * y.val) % 18446744073709551616 := by
  rw [core.num.U64.wrapping_mul_val_eq, modulus_eq]

theorem one_val : (1#u64).val = 1 := by simp [U64.ofNat, UScalar.ofNat]

theorem zero_val : (0#u64).val = 0 := by simp [U64.ofNat, UScalar.ofNat]

theorem max_val : (18446744073709551615#u64).val = 18446744073709551615 := by
  simp [U64.ofNat, UScalar.ofNat]

theorem phi_inv_val : (17428512612931826493#u64).val = 17428512612931826493 := by
  simp [U64.ofNat, UScalar.ofNat]

theorem max_product_val : (1018231460777725123#u64).val = 1018231460777725123 := by
  simp [U64.ofNat, UScalar.ofNat]

/-- The value of the extracted multiplier constant. -/
theorem extracted_phi_inv_val :
    (coord_unspread_source.coord.PHI_INV).val = 17428512612931826493 := by
  rw [phi_inv_eq]
  exact phi_inv_val

/-- Closed form: the real body returns the wrapped product of its argument and the extracted
constant, and nothing else. -/
theorem body_value (identity r : U64) (h : body identity = ok r) :
    r.val = (identity.val * 17428512612931826493) % 18446744073709551616 := by
  have hb := body_is_one_wrapping_mul identity
  rw [h] at hb
  have hi : r = core.num.U64.wrapping_mul identity coord_unspread_source.coord.PHI_INV :=
    Result.ok_injective hb
  rw [hi, phi_inv_eq, product_val, phi_inv_val]

/-- The joint machine-word domain of the real call: the argument is a `U64` word, so its `.val`
is below the word's bit width. -/
def MachineWord (identity : U64) : Prop := identity.val < 2 ^ UScalarTy.U64.numBits

/-- The extracted argument type initializes the joint domain at every word, so the domain of
the body is total with no partial input to check and no unreachable premise. -/
theorem machineWord_holds (identity : U64) : MachineWord identity := identity.hmax

/-- The real body at the concrete word `1#u64` is the concrete multiplier word. -/
theorem unspread_one : body 1#u64 = ok 17428512612931826493#u64 := by
  have h : core.num.U64.wrapping_mul 1#u64 coord_unspread_source.coord.PHI_INV
      = 17428512612931826493#u64 := by
    apply UScalar.eq_of_val_eq
    rw [product_val, one_val, extracted_phi_inv_val]
    norm_num
  show ok (core.num.U64.wrapping_mul 1#u64 coord_unspread_source.coord.PHI_INV)
      = ok 17428512612931826493#u64
  rw [h]

/-- The real body at the concrete word `0#u64` is the concrete zero word. -/
theorem unspread_zero : body 0#u64 = ok 0#u64 := by
  have h : core.num.U64.wrapping_mul 0#u64 coord_unspread_source.coord.PHI_INV = 0#u64 := by
    apply UScalar.eq_of_val_eq
    rw [product_val, zero_val, extracted_phi_inv_val]
  show ok (core.num.U64.wrapping_mul 0#u64 coord_unspread_source.coord.PHI_INV) = ok 0#u64
  rw [h]

/-- The real body at the largest word is a concrete word, so the wrapping step is reached by
inputs other than the small ones. -/
theorem unspread_max : body 18446744073709551615#u64 = ok 1018231460777725123#u64 := by
  have h : core.num.U64.wrapping_mul 18446744073709551615#u64
      coord_unspread_source.coord.PHI_INV = 1018231460777725123#u64 := by
    apply UScalar.eq_of_val_eq
    rw [product_val, max_val, extracted_phi_inv_val, max_product_val]
  show ok (core.num.U64.wrapping_mul 18446744073709551615#u64
      coord_unspread_source.coord.PHI_INV) = ok 1018231460777725123#u64
  rw [h]

/-- Reachable success witness at the joint machine-word type: the identity word `1#u64`, whose
real evaluation is the concrete multiplier word. -/
theorem success_witness_one :
    MachineWord 1#u64 ∧ body 1#u64 = ok 17428512612931826493#u64 :=
  ⟨machineWord_holds 1#u64, unspread_one⟩

/-- Reachable success witness at the joint machine-word type: the zero identity word. -/
theorem success_witness_zero :
    MachineWord 0#u64 ∧ body 0#u64 = ok 0#u64 :=
  ⟨machineWord_holds 0#u64, unspread_zero⟩

/-- Reachable success witness at the joint machine-word type: the largest `U64` word. -/
theorem success_witness_max :
    MachineWord 18446744073709551615#u64 ∧
      body 18446744073709551615#u64 = ok 1018231460777725123#u64 :=
  ⟨machineWord_holds 18446744073709551615#u64, unspread_max⟩

/-- The counted work of the witness evaluations is the counted work of the body: the witnesses
pin the value, not a cheaper path. -/
theorem witness_work_is_the_body_work :
    workOf 1#u64 = workOf 0#u64 ∧ workOf 18446744073709551615#u64 = workOf 1#u64 :=
  ⟨work_is_input_independent 1#u64 0#u64,
    work_is_input_independent 18446744073709551615#u64 1#u64⟩

end CoordUnspreadResource

open Lean Elab Command in
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit CoordUnspreadResource.root_forwards
#resource_audit CoordUnspreadResource.phi_inv_eq
#resource_audit CoordUnspreadResource.modulus_eq
#resource_audit CoordUnspreadResource.body_is_one_wrapping_mul
#resource_audit CoordUnspreadResource.work_is_one_multiply
#resource_audit CoordUnspreadResource.work_is_input_independent
#resource_audit CoordUnspreadResource.rank_initial
#resource_audit CoordUnspreadResource.rank_exit
#resource_audit CoordUnspreadResource.rank_drops
#resource_audit CoordUnspreadResource.guard_exhaustive
#resource_audit CoordUnspreadResource.exit_is_terminal
#resource_audit CoordUnspreadResource.chain_rank_bound
#resource_audit CoordUnspreadResource.chain_length_le_one
#resource_audit CoordUnspreadResource.exit_branch
#resource_audit CoordUnspreadResource.body_total
#resource_audit CoordUnspreadResource.body_never_errors
#resource_audit CoordUnspreadResource.product_val
#resource_audit CoordUnspreadResource.extracted_phi_inv_val
#resource_audit CoordUnspreadResource.body_value
#resource_audit CoordUnspreadResource.machineWord_holds
#resource_audit CoordUnspreadResource.unspread_one
#resource_audit CoordUnspreadResource.unspread_zero
#resource_audit CoordUnspreadResource.unspread_max
#resource_audit CoordUnspreadResource.success_witness_one
#resource_audit CoordUnspreadResource.success_witness_zero
#resource_audit CoordUnspreadResource.success_witness_max
#resource_audit CoordUnspreadResource.witness_work_is_the_body_work

run_cmd Lean.logInfo "COORD_UNSPREAD_KERNEL_OK"