/-
# State and result arenas

The two record types stored in the bounded arenas: a state record carries the
machine environment, memory and call depth, and a result record carries a raw
word. `StateArena`, `ResultArena`, `StateHandle` and `ResultHandle` are the
fixed-capacity specializations and their handles, so a state handle cannot be
used against a result arena and vice versa.
-/
import Iota.Arena.Bounded

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Arena

/-- The machine state stored in a state arena slot. -/
structure StateRecord where
  environment : Iota.Machine.Environment
  memory : Iota.Machine.Memory
  callDepth : Nat

/-- The word result stored in a result arena slot. -/
structure ResultRecord where
  raw : Iota.Word

/-- A bounded arena of machine states. -/
abbrev StateArena (capacity : Nat) := Arena StateRecord capacity

/-- A bounded arena of word results. -/
abbrev ResultArena (capacity : Nat) := Arena ResultRecord capacity

/-- A handle into a specific state arena. -/
abbrev StateHandle {capacity : Nat} (arena : StateArena capacity) := Handle arena

/-- A handle into a specific result arena. -/
abbrev ResultHandle {capacity : Nat} (arena : ResultArena capacity) := Handle arena

end Iota.Arena
