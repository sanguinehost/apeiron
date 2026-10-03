import Dispatch.Extracted

namespace Dispatch.Spec
open dispatch_extract Aeneas Aeneas.Std

deriving instance DecidableEq for core.Stage
deriving instance DecidableEq for core.Digest

/-- An observation satisfies its ordered obligation with the registered identity. -/
def Valid (o : core.Observation) (stage : core.Stage) (identity : core.Digest) : Prop :=
  o.stage = stage ∧ o.exit = 0#i32 ∧ o.identity = identity

instance (o : core.Observation) (stage : core.Stage) (identity : core.Digest) :
    Decidable (Valid o stage identity) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def next : core.Stage → Option core.Stage
  | .Extraction => some .Dependencies
  | .Dependencies => some .Certificate
  | .Certificate => some .Revalidation
  | .Revalidation => some .Done
  | .Done => none

def start (reg : Option core.Digest) (observed : core.Digest) : Option core.Session :=
  if reg = some observed then some ⟨observed, .Extraction⟩ else none

def advance (s : core.Session) (o : core.Observation) : Option core.Session :=
  if Valid o s.next s.identity then
    (next s.next).map fun stage => ⟨s.identity, stage⟩
  else none

def finish (s : core.Session) (observed : core.Digest) : Option core.Permit :=
  if s.next = .Done ∧ observed = s.identity then some ⟨s.identity⟩ else none

/-- Joint acceptance domain; no antecedent asserts that a permit already exists. -/
def Admissible (reg : Option core.Digest) (initial : core.Digest)
    (a b c d : core.Observation) (fresh : core.Digest) : Prop :=
  reg = some initial ∧ Valid a .Extraction initial ∧ Valid b .Dependencies initial ∧
    Valid c .Certificate initial ∧ Valid d .Revalidation initial ∧ fresh = initial

instance (reg : Option core.Digest) (initial : core.Digest)
    (a b c d : core.Observation) (fresh : core.Digest) :
    Decidable (Admissible reg initial a b c d fresh) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

def complete (reg : Option core.Digest) (initial : core.Digest)
    (a b c d : core.Observation) (fresh : core.Digest) : Option core.Permit :=
  if Admissible reg initial a b c d fresh then some ⟨initial⟩ else none

end Dispatch.Spec
