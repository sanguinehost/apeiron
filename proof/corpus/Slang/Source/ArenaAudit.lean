import Slang.Source.Arena
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source.ArenaAudit

/-- The local positive fixture the zero-axiom runner mutates to prove the command rejects a
nonempty cone. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

#audit_empty Slang.Source.slice : type_of% @Slang.Source.slice
#audit_empty Slang.Source.CheckedInterval : type_of% @Slang.Source.CheckedInterval
#audit_empty Slang.Source.CheckedInterval.contents : type_of% @Slang.Source.CheckedInterval.contents
#audit_empty Slang.Source.ArenaFile : type_of% @Slang.Source.ArenaFile
#audit_empty Slang.Source.SourceArena : type_of% @Slang.Source.SourceArena
#audit_empty Slang.Source.FileId : type_of% @Slang.Source.FileId
#audit_empty Slang.Source.SourceArena.file : type_of% @Slang.Source.SourceArena.file
#audit_empty Slang.Source.rebaseToken : type_of% @Slang.Source.rebaseToken
#audit_empty Slang.Source.SourceRef : type_of% @Slang.Source.SourceRef
#audit_empty Slang.Source.SourceRef.byte_in_bounds : type_of% @Slang.Source.SourceRef.byte_in_bounds
#audit_empty Slang.Source.SourceRef.token_in_bounds : type_of% @Slang.Source.SourceRef.token_in_bounds
#audit_empty Slang.Source.SourceRef.exact_tokens : type_of% @Slang.Source.SourceRef.exact_tokens
#audit_empty Slang.Source.copiedTokens_rejected : type_of% @Slang.Source.copiedTokens_rejected
#audit_empty Slang.Source.copiedTokens_not_a_ref : type_of% @Slang.Source.copiedTokens_not_a_ref

#audit_empty Slang.Source.ArenaAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Slang.Source.ArenaAudit
