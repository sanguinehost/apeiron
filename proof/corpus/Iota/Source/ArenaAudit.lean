import Iota.Source.Arena
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Source.ArenaAudit

/-- The local positive fixture the zero-axiom runner mutates to prove the command rejects a
nonempty cone. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

#audit_empty Iota.Source.slice : type_of% @Iota.Source.slice
#audit_empty Iota.Source.CheckedInterval : type_of% @Iota.Source.CheckedInterval
#audit_empty Iota.Source.CheckedInterval.contents : type_of% @Iota.Source.CheckedInterval.contents
#audit_empty Iota.Source.ArenaFile : type_of% @Iota.Source.ArenaFile
#audit_empty Iota.Source.SourceArena : type_of% @Iota.Source.SourceArena
#audit_empty Iota.Source.FileId : type_of% @Iota.Source.FileId
#audit_empty Iota.Source.SourceArena.file : type_of% @Iota.Source.SourceArena.file
#audit_empty Iota.Source.rebaseToken : type_of% @Iota.Source.rebaseToken
#audit_empty Iota.Source.SourceRef : type_of% @Iota.Source.SourceRef
#audit_empty Iota.Source.SourceRef.byte_in_bounds : type_of% @Iota.Source.SourceRef.byte_in_bounds
#audit_empty Iota.Source.SourceRef.token_in_bounds : type_of% @Iota.Source.SourceRef.token_in_bounds
#audit_empty Iota.Source.SourceRef.exact_tokens : type_of% @Iota.Source.SourceRef.exact_tokens
#audit_empty Iota.Source.copiedTokens_rejected : type_of% @Iota.Source.copiedTokens_rejected
#audit_empty Iota.Source.copiedTokens_not_a_ref : type_of% @Iota.Source.copiedTokens_not_a_ref

#audit_empty Iota.Source.ParsedRef : type_of% @Iota.Source.ParsedRef
#audit_empty Iota.Source.ParsedRef.exact_decode : type_of% @Iota.Source.ParsedRef.exact_decode
#audit_empty Iota.Source.ParsedRef.program_determined : type_of% @Iota.Source.ParsedRef.program_determined
#audit_empty Iota.Source.ParsedRef.mismatched_program_rejected : type_of% @Iota.Source.ParsedRef.mismatched_program_rejected
#audit_empty Iota.Source.ParsedRef.context : type_of% @Iota.Source.ParsedRef.context
#audit_empty Iota.Source.ParsedRef.context_constants : type_of% @Iota.Source.ParsedRef.context_constants

#audit_empty Iota.Source.ArenaAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Iota.Source.ArenaAudit
