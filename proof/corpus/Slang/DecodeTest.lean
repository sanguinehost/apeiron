import Slang.Decode

namespace Slang.DecodeTest

open Slang Slang.Decode

-- Temporal O(b) for b = byte length of the resolved file: `IO.FS.readBinFile` reads the whole file,
-- while the `pathExists` probe and path concatenation are O(|relative|). Spatial O(b): the returned
-- ByteArray plus the read buffer. Operations counted: filesystem probes, bytes read.
def readSongFile (relative : String) : IO ByteArray := do
  let nearby := "../../song/" ++ relative
  if ← System.FilePath.pathExists nearby then IO.FS.readBinFile nearby
  else IO.FS.readBinFile ("../../../song/" ++ relative)

def require (condition : Bool) (message : String) : IO Unit :=
  if condition then pure () else throw (IO.userError message)

-- Temporal O(t + s) for the table and std source byte lengths: `readSongFile` reads both files and
-- `decodeClosure` is linear in them, while the membership/equality checks scan the O(1) selected
-- constant and function name lists. Spatial O(t + s): the two ByteArrays plus the decoded
-- token/declaration sets. Operations counted: bytes read, decoded tokens, `require` comparisons.
def main : IO Unit := do
  let table ← readSongFile "src/table/table.slang"
  let std ← readSongFile "src/kernel/std.slang"
  let decoded ← match decodeClosure table std with
    | .ok decoded => pure decoded
    | .error message => throw (IO.userError message)
  let program := decoded.val.program
  require (program.sourceBytes = table.size) "source byte count differs"
  require (program.consumedBytes = table.size) "source was not completely consumed"
  require (program.constants.map (·.name) = selectedConstantNames) "constant declarations differ"
  require (program.constants.map (fun decl => (decl.name, decl.value)) == [
    ("TABLE_PRESENT", .literal 1),
    ("TABLE_EMPTY", .binary .sub (.literal 0) (.literal 1)),
    ("TABLE_STRIDE", .literal 2),
    ("MERGE_SET", .literal 1), ("MERGE_MAP", .literal 2),
    ("MERGE_ACC", .literal 3), ("MERGE_GRAPH", .literal 4),
    ("TABLE_PHI", .literal 2654435769),
    ("COORD_CEILING", .literal 8796093022208)]) "constant values differ"
  require (program.functions.map (·.name) = selectedFunctionNames) "function declarations differ"
  let insert ← match program.functions.find? (·.name = "tableInsertWith") with
    | some decl => pure decl
    | none => throw (IO.userError "tableInsertWith absent")
  require (insert.parameters = ["base", "cap", "key", "value", "merge"]) "tableInsertWith parameters differ"
  require (insert.body.length = 13) "tableInsertWith statement count differs"
  require (insert.body.getLast? == some (.return_ (.call "tableSlotValSet"
    [.name "base", .name "slot", .name "newVal"]))) "tableInsertWith final return differs"
  let spread ← match program.functions.find? (·.name = "tableSpread") with
    | some decl => pure decl
    | none => throw (IO.userError "tableSpread absent")
  require (spread.body.head? == some (.bind "pos" (.conditional
    (.binary .sub (.literal 0) (.name "k"))
    (.binary .lt (.name "k") (.literal 0)) (.name "k"))))
    "tableSpread inline conditional precedence differs"
  require (spread.body.getLast? == some (.return_ (.binary .bitAnd
    (.binary .mul (.name "pos") (.name "TABLE_PHI")) (.literal 4611686018427387903))))
    "tableSpread precedence differs"
  let slotSet ← match program.functions.find? (·.name = "tableSlotSet") with
    | some decl => pure decl
    | none => throw (IO.userError "tableSlotSet absent")
  require (slotSet.body.length = 3) "tableSlotSet did not consume both stores and return"
  let truncated := table.extract 0 (table.size / 2)
  require (match decodeTable truncated with | .error _ => true | .ok _ => false)
    "truncated selected body decoded"
  let changed := table ++ " unsupported;".toUTF8
  require (match decodeTable changed, decodeTable table with
    | .ok left, .ok right => !(left.val == right.val)
    | _, _ => true) "altered source decoded identically"

#eval main

end Slang.DecodeTest
