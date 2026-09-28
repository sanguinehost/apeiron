import Spirv.Fragment

/-!
# Decoding the SPIR-V binary of the integer DP4A fragment

`decode : List U32Word → Option Module` parses a SPIR-V 1.3 binary (little-endian `u32` words,
`Kb::finish` in `~/fold/modelfold/src/simt_nn/mod.rs`) into the fragment
syntax of `Spirv.Fragment`, refusing malformed input. The header is
`magic · version · generator · bound · schema`; each instruction begins with
`(WordCount << 16) | Opcode`, and the `WordCount - 1` operand words follow.

The decoder is two-phase: `parseInsts` fixes instruction boundaries from the high-16-bit
`WordCount` (D1) and maps each opcode to an admitted `Inst`; `wfCheckB` then scans the linear
instruction list, enforcing the id and type obligations (D2–D4). `decode_sound` states that a
decoded module is `WellFormed` and `IsDp4aFragment`.

## The D1–D4 obligations

| | statement | theorem |
|---|---|---|
| D1 boundaries | a zero `WordCount`, or one that overruns the remaining words, makes `parseInsts` return `none` | `D1_wordCount_zero`, `D1_overrun` |
| D2 ids | each instruction's used ids are defined by its prefix, and its result id is fresh | `D2_defUses`, `D2_unique`, `WellFormed.defUses`, `WellFormed.unique` |
| D3 types | on `OpSUDot` the result type is an integer of width ≥ 32 and both operands are 32-bit integers | `D3_typeOK`, `WellFormed.typeOK` |
| D4 storage | `OpTypePointer`/`OpVariable` use an admitted storage class and `OpDecorate`/`OpMemberDecorate` an admitted decoration | `D4_storageOK`, `WellFormed.storageOK` |

`OpEntryPoint`'s entry and interface ids, branch/merge label targets, and — per SPIR-V 1.3 §2.4 —
the `OpExecutionMode` entry point, the `OpDecorate` target and the `OpMemberDecorate` type are
SPIR-V forward references and are exempt from definition-before-use (`Inst.uses` omits them);
`OpSUDot` only carries `PackedVectorFormat4x8Bit` (`0`); any other format is refused.
-/

namespace Spirv

/-! ## Little-endian bytes -/

/-- A binary byte, held in `Nat` and reduced modulo `256` on use. -/
abbrev Byte := Nat

/-- The four little-endian bytes of a word, least-significant first. -/
def bytesOfWord (w : U32Word) : List Byte :=
  [w % 256, w / 256 % 256, w / 65536 % 256, w / 16777216 % 256]

/-- The little-endian byte string of a word list. -/
def bytes : List U32Word → List Byte
  | [] => []
  | w :: rest => bytesOfWord w ++ bytes rest

/-- Reassemble a word from four little-endian bytes. -/
def wordOfBytes (b0 b1 b2 b3 : Byte) : U32Word :=
  b0 + b1 * 256 + b2 * 65536 + b3 * 16777216

/-- Reassemble a word list from a byte string, four bytes per word; a short tail is zero-padded. -/
def wordsOfBytes : List Byte → List U32Word
  | [] => []
  | [b0] => [wordOfBytes b0 0 0 0]
  | [b0, b1] => [wordOfBytes b0 b1 0 0]
  | [b0, b1, b2] => [wordOfBytes b0 b1 b2 0]
  | b0 :: b1 :: b2 :: b3 :: rest => wordOfBytes b0 b1 b2 b3 :: wordsOfBytes rest

/-- A word below `2 ^ 32` is recovered from its own byte decomposition. -/
theorem wordOfBytes_bytesOfWord (w : U32Word) (h : w < 2 ^ 32) :
    wordOfBytes (w % 256) (w / 256 % 256) (w / 65536 % 256) (w / 16777216 % 256) = w := by
  have h1 := Nat.mod_add_div w 256
  have h2 := Nat.mod_add_div (w / 256) 256
  have h3 := Nat.mod_add_div (w / 65536) 256
  have e2 : w / 256 / 256 = w / 65536 := by rw [Nat.div_div_eq_div_mul]
  have e3 : w / 65536 / 256 = w / 16777216 := by rw [Nat.div_div_eq_div_mul]
  have hd : w / 16777216 < 256 := by
    rw [Nat.div_lt_iff_lt_mul (by norm_num : (0 : Nat) < 16777216)]
    norm_num
    omega
  have e4 : w / 16777216 % 256 = w / 16777216 := Nat.mod_eq_of_lt hd
  have h2' : w / 256 % 256 + 256 * (w / 65536) = w / 256 := by rw [← e2]; exact h2
  have h3' : w / 65536 % 256 + 256 * (w / 16777216) = w / 65536 := by rw [← e3]; exact h3
  simp only [wordOfBytes]
  rw [e4]
  linear_combination h1 + 256 * h2' + 65536 * h3'

/-- A four-byte group reassembles into one word. -/
theorem wordsOfBytes_four (b0 b1 b2 b3 : Byte) (rest : List Byte) :
    wordsOfBytes (b0 :: b1 :: b2 :: b3 :: rest) =
      wordOfBytes b0 b1 b2 b3 :: wordsOfBytes rest := rfl

/-- The little-endian byte order round trip: `wordsOfBytes ∘ bytes` is the identity on words
below `2 ^ 32`, the range the emitter produces. -/
theorem words_of_bytes_bytes (ws : List U32Word) (h : ∀ w ∈ ws, w < 2 ^ 32) :
    wordsOfBytes (bytes ws) = ws := by
  induction ws with
  | nil => rfl
  | cons w t ih =>
      have hw : w < 2 ^ 32 := h w (by simp)
      have ht : ∀ x ∈ t, x < 2 ^ 32 := fun x hx => h x (by simp [hx])
      show wordsOfBytes (bytesOfWord w ++ bytes t) = w :: t
      simp only [bytesOfWord, List.cons_append, List.nil_append]
      rw [wordsOfBytes_four, wordOfBytes_bytesOfWord w hw, ih ht]

/-! ## Instruction parsing (D1) -/

/-- Does the word carry a zero byte? The terminator of a SPIR-V literal string. -/
def hasZeroByte (w : U32Word) : Bool :=
  w % 256 == 0 || w / 256 % 256 == 0 || w / 65536 % 256 == 0 || w / 16777216 % 256 == 0

/-- Split off a nul-terminated literal string, returning the name words and the rest. -/
def takeLiteral : List U32Word → Option (List U32Word × List U32Word)
  | [] => none
  | w :: t =>
      if hasZeroByte w then some ([w], t)
      else (takeLiteral t).map (fun p => (w :: p.1, p.2))

/-- The ASCII name held by a list of literal words, up to the first zero byte. -/
def litString (ws : List U32Word) : String :=
  String.ofList ((bytes ws).takeWhile (· != 0) |>.map Char.ofNat)

/-- Map an opcode and its operand words to an admitted instruction. Opcodes outside the admitted
set, wrong operand counts, and `OpSUDot` formats other than `PackedVectorFormat4x8Bit` return
`none`. -/
def decodeInst (op : Nat) (ops : List U32Word) : Option Inst :=
  match op, ops with
  | 17, [cap] => some (.capability cap)
  | 10, _ => some (.extension (litString ops))
  | 11, r :: name => some (.extInstImport r (litString name))
  | 12, ty :: r :: set :: instr :: args => some (.extInst ty r set instr args)
  | 14, [addr, mem] => some (.memoryModel addr mem)
  | 15, model :: entry :: rest =>
      match takeLiteral rest with
      | some (_, iface) => some (.entryPoint model entry iface)
      | none => none
  | 16, entry :: mode :: opts => some (.executionMode entry mode opts)
  | 71, target :: dec :: lits => some (.decorate target dec lits)
  | 72, ty :: member :: dec :: lits => some (.memberDecorate ty member dec lits)
  | 19, [r] => some (.typeVoid r)
  | 20, [r] => some (.typeBool r)
  | 21, [r, width, signed] => some (.typeInt r width signed)
  | 22, [r, width] => some (.typeFloat r width)
  | 23, [r, comp, count] => some (.typeVector r comp count)
  | 29, [r, elem] => some (.typeRuntimeArray r elem)
  | 30, r :: members => some (.typeStruct r members)
  | 32, [r, sc, pointee] => some (.typePointer r sc pointee)
  | 33, r :: ret :: params => some (.typeFunction r ret params)
  | 43, [ty, r, v] => some (.constant ty r v)
  | 54, [ty, r, ctl, sig] => some (.function ty r ctl sig)
  | 56, [] => some .functionEnd
  | 59, [ty, r, sc] => some (.variable ty r sc none)
  | 59, [ty, r, sc, init] => some (.variable ty r sc (some init))
  | 61, [ty, r, ptr] => some (.load ty r ptr)
  | 62, [ptr, val] => some (.store ptr val)
  | 65, ty :: r :: base :: idxs => some (.accessChain ty r base idxs)
  | 81, ty :: r :: agg :: idxs => some (.compositeExtract ty r agg idxs)
  | 128, [ty, r, a, b] => some (.iadd ty r a b)
  | 132, [ty, r, a, b] => some (.imul ty r a b)
  | 169, [ty, r, c, a, b] => some (.select ty r c a b)
  | 176, [ty, r, a, b] => some (.ult ty r a b)
  | 194, [ty, r, a, s] => some (.shiftRightLogical ty r a s)
  | 199, [ty, r, a, b] => some (.bitAnd ty r a b)
  | 4452, [ty, r, a, b, fmt] =>
      if fmt = 0 then some (.sudot ty r a b .packed4x8) else none
  | 111, [ty, r, a] => some (.convertSToF ty r a)
  | 129, [ty, r, a, b] => some (.fadd ty r a b)
  | 133, [ty, r, a, b] => some (.fmul ty r a b)
  | 246, [merge, cont, ctl] => some (.loopMerge merge cont ctl)
  | 247, [merge, ctl] => some (.selectionMerge merge ctl)
  | 248, [l] => some (.label l)
  | 249, [l] => some (.branch l)
  | 250, [c, l1, l2] => some (.branchCond c l1 l2)
  | 253, [] => some .ret
  | _, _ => none

/-- Parse a word stream into instructions, with `fuel` a bound on the remaining word count. A
`WordCount` of zero, or one that overruns the remaining words, returns `none`. -/
def parseInstsAux : Nat → List U32Word → Option (List Inst)
  | 0, _ => some []
  | _ + 1, [] => some []
  | fuel + 1, h :: t =>
      if h / 65536 = 0 then none
      else if t.length < h / 65536 - 1 then none
      else
        match decodeInst (h % 65536) (t.take (h / 65536 - 1)) with
        | none => none
        | some i => (parseInstsAux fuel (t.drop (h / 65536 - 1))).map (i :: ·)

/-- Parse the instruction stream of a module. -/
def parseInsts (ws : List U32Word) : Option (List Inst) :=
  parseInstsAux ws.length ws

/-- D1: a zero `WordCount` refuses. -/
theorem D1_wordCount_zero {h : U32Word} {t : List U32Word} (hw : h / 65536 = 0) :
    parseInsts (h :: t) = none := by
  simp only [parseInsts, List.length_cons, parseInstsAux]
  rw [if_pos hw]

/-- D1: a `WordCount` that overruns the remaining words refuses. -/
theorem D1_overrun {h : U32Word} {t : List U32Word} (hw : h / 65536 ≠ 0)
    (hlen : t.length < h / 65536 - 1) :
    parseInsts (h :: t) = none := by
  simp only [parseInsts, List.length_cons, parseInstsAux]
  rw [if_neg hw, if_pos hlen]

/-! ## The linear id/type scan (D2–D4) -/

/-- A shallow type tag, mirroring `VType` but with decidable equality. -/
inductive TyTag where
  | int (w : Nat) (signed : Bool)
  | bool
  | float (w : Nat)
  | vec (n : Nat) (elem : TyTag)
  | ptr (sc : Nat) (pointee : TyTag)
  | runtimeArray (elem : TyTag)
  | fn
  | st
  | void
  deriving DecidableEq, Repr

/-- The scan state: ids defined so far and their type tags. -/
structure WFState where
  defs : List SpvId
  tys : List (SpvId × TyTag)
  deriving Repr

/-- The initial state: no ids. -/
def WFState.init : WFState := ⟨[], []⟩

/-- Look an id up in a type-tag association list. -/
def lookupTy : List (SpvId × TyTag) → SpvId → Option TyTag
  | [], _ => none
  | (k, v) :: t, x => if k = x then some v else lookupTy t x

/-- The result id an instruction defines, if any. -/
def Inst.resultId : Inst → Option SpvId
  | .extInstImport r _ => some r
  | .extInst _ r _ _ _ => some r
  | .typeVoid r => some r
  | .typeBool r => some r
  | .typeInt r _ _ => some r
  | .typeFloat r _ => some r
  | .typeVector r _ _ => some r
  | .typeRuntimeArray r _ => some r
  | .typeStruct r _ => some r
  | .typePointer r _ _ => some r
  | .typeFunction r _ _ => some r
  | .constant _ r _ => some r
  | .function _ r _ _ => some r
  | .variable _ r _ _ => some r
  | .load _ r _ => some r
  | .accessChain _ r _ _ => some r
  | .compositeExtract _ r _ _ => some r
  | .iadd _ r _ _ => some r
  | .imul _ r _ _ => some r
  | .select _ r _ _ _ => some r
  | .ult _ r _ _ => some r
  | .shiftRightLogical _ r _ _ => some r
  | .bitAnd _ r _ _ => some r
  | .sudot _ r _ _ _ => some r
  | .convertSToF _ r _ => some r
  | .fadd _ r _ _ => some r
  | .fmul _ r _ _ => some r
  | .label l => some l
  | _ => none

/-- Ids an instruction uses and that must be defined by its prefix. Branch/merge label targets and
`OpEntryPoint`'s entry/interface ids are SPIR-V forward references and are not listed. SPIR-V 1.3
§2.4 also permits a forward reference for the `OpExecutionMode` entry point, the `OpDecorate` target
and the `OpMemberDecorate` type, so those operands are exempt as well. -/
def Inst.uses : Inst → List SpvId
  | .capability _ => []
  | .extension _ => []
  | .extInstImport _ _ => []
  | .extInst ty _ set _ args => ty :: set :: args
  | .memoryModel _ _ => []
  | .entryPoint _ _ _ => []
  | .executionMode _ _ _ => []
  | .decorate _ _ _ => []
  | .memberDecorate _ _ _ _ => []
  | .typeVoid _ => []
  | .typeBool _ => []
  | .typeInt _ _ _ => []
  | .typeFloat _ _ => []
  | .typeVector _ comp _ => [comp]
  | .typeRuntimeArray _ elem => [elem]
  | .typeStruct _ members => members
  | .typePointer _ _ pointee => [pointee]
  | .typeFunction _ fret params => fret :: params
  | .constant ty _ _ => [ty]
  | .function ty _ _ sig => [ty, sig]
  | .functionEnd => []
  | .variable ty _ _ init => ty :: init.toList
  | .load ty _ ptr => [ty, ptr]
  | .store ptr val => [ptr, val]
  | .accessChain ty _ base idxs => ty :: base :: idxs
  | .compositeExtract ty _ agg _ => [ty, agg]
  | .iadd ty _ a b => [ty, a, b]
  | .imul ty _ a b => [ty, a, b]
  | .select ty _ c a b => [ty, c, a, b]
  | .ult ty _ a b => [ty, a, b]
  | .shiftRightLogical ty _ a s => [ty, a, s]
  | .bitAnd ty _ a b => [ty, a, b]
  | .sudot ty _ a b _ => [ty, a, b]
  | .convertSToF ty _ a => [ty, a]
  | .fadd ty _ a b => [ty, a, b]
  | .fmul ty _ a b => [ty, a, b]
  | .loopMerge _ _ _ => []
  | .selectionMerge _ _ => []
  | .label _ => []
  | .branch _ => []
  | .branchCond c _ _ => [c]
  | .ret => []

/-- The type tag an instruction gives its result, resolving the type operand against the scan
state. -/
def resultTag (st : WFState) (i : Inst) : Option TyTag :=
  match i with
  | .typeVoid _ => some .void
  | .typeBool _ => some .bool
  | .typeInt _ w s => some (.int w (s != 0))
  | .typeFloat _ w => some (.float w)
  | .typeVector _ comp n => (lookupTy st.tys comp).map (.vec n)
  | .typeRuntimeArray _ elem => (lookupTy st.tys elem).map .runtimeArray
  | .typeStruct _ _ => some .st
  | .typePointer _ sc pointee => (lookupTy st.tys pointee).map (.ptr sc)
  | .typeFunction _ _ _ => some .fn
  | .constant ty _ _ => lookupTy st.tys ty
  | .variable ty _ _ _ => lookupTy st.tys ty
  | .load ty _ _ => lookupTy st.tys ty
  | .accessChain ty _ _ _ => lookupTy st.tys ty
  | .compositeExtract ty _ _ _ => lookupTy st.tys ty
  | .iadd ty _ _ _ => lookupTy st.tys ty
  | .imul ty _ _ _ => lookupTy st.tys ty
  | .select ty _ _ _ _ => lookupTy st.tys ty
  | .ult ty _ _ _ => lookupTy st.tys ty
  | .shiftRightLogical ty _ _ _ => lookupTy st.tys ty
  | .bitAnd ty _ _ _ => lookupTy st.tys ty
  | .sudot ty _ _ _ _ => lookupTy st.tys ty
  | .extInst ty _ _ _ _ => lookupTy st.tys ty
  | .convertSToF ty _ _ => lookupTy st.tys ty
  | .fadd ty _ _ _ => lookupTy st.tys ty
  | .fmul ty _ _ _ => lookupTy st.tys ty
  | _ => none

/-- Extend the scan state by an instruction's result id and type. -/
def stepInst (st : WFState) (i : Inst) : WFState :=
  match i.resultId with
  | some r =>
      { defs := r :: st.defs
        tys := match resultTag st i with
               | some t => (r, t) :: st.tys
               | none => st.tys }
  | none => st

/-- The storage classes the fragment admits: `Input`, `Workgroup`, `Function`, `StorageBuffer`. -/
def validStorageClasses : List Nat := [1, 4, 7, 12]

/-- The decorations the fragment admits. -/
def validDecorations : List Nat := [2, 6, 11, 33, 34, 35]

/-- D2 def-before-use: every used id is defined by the prefix. -/
def DefUsesB (st : WFState) (i : Inst) : Bool :=
  i.uses.all (fun x => decide (x ∈ st.defs))

/-- D2 uniqueness: an instruction's result id is fresh. -/
def UniqueB (st : WFState) (i : Inst) : Bool :=
  match i.resultId with
  | some r => !decide (r ∈ st.defs)
  | none => true

/-- D3 type agreement: on `OpSUDot` the result type is an integer of width ≥ 32 and both operands
are 32-bit integers. -/
def TypeOKB (st : WFState) (i : Inst) : Bool :=
  match i with
  | .sudot ty _ a b _ =>
      match lookupTy st.tys ty, lookupTy st.tys a, lookupTy st.tys b with
      | some (.int w _), some (.int 32 _), some (.int 32 _) => decide (32 ≤ w)
      | _, _, _ => false
  | _ => true

/-- D4 storage: storage classes and decorations are admitted. -/
def StorageOKB (_st : WFState) (i : Inst) : Bool :=
  match i with
  | .typePointer _ sc _ => decide (sc ∈ validStorageClasses)
  | .variable _ _ sc _ => decide (sc ∈ validStorageClasses)
  | .decorate _ dec _ => decide (dec ∈ validDecorations)
  | .memberDecorate _ _ dec _ => decide (dec ∈ validDecorations)
  | _ => true

/-- The per-instruction well-formedness conjunction (D2–D4). -/
def WFInst (st : WFState) (i : Inst) : Prop :=
  DefUsesB st i = true ∧ UniqueB st i = true ∧ TypeOKB st i = true ∧ StorageOKB st i = true

/-- The Boolean form of `WFInst`. -/
def wfInstB (st : WFState) (i : Inst) : Bool :=
  DefUsesB st i && UniqueB st i && TypeOKB st i && StorageOKB st i

/-- The Boolean and propositional forms agree. -/
theorem wfInstB_iff (st : WFState) (i : Inst) : wfInstB st i = true ↔ WFInst st i := by
  simp only [wfInstB, WFInst, Bool.and_eq_true, and_assoc]

/-- The well-formedness run over the linear instruction list. -/
def WFCode (st : WFState) : List Inst → Prop
  | [] => True
  | i :: t => WFInst st i ∧ WFCode (stepInst st i) t

/-- The Boolean well-formedness run. -/
def wfCheckB (st : WFState) : List Inst → Bool
  | [] => true
  | i :: t => wfInstB st i && wfCheckB (stepInst st i) t

/-- The Boolean run is sound for the propositional run. -/
theorem wfCheckB_sound : ∀ (code : List Inst) (st : WFState),
    wfCheckB st code = true → WFCode st code
  | [], _, _ => trivial
  | i :: t, st, h => by
      simp only [wfCheckB, Bool.and_eq_true] at h
      exact ⟨(wfInstB_iff st i).mp h.1, wfCheckB_sound t (stepInst st i) h.2⟩

/-- A module is well-formed when the linear scan accepts its instruction list. -/
def WellFormed (m : Module) : Prop := WFCode WFState.init m.code

/-- D2 def-before-use, per instruction. -/
theorem D2_defUses {st : WFState} {i : Inst} (h : WFInst st i) : DefUsesB st i = true := h.1

/-- D2 uniqueness, per instruction. -/
theorem D2_unique {st : WFState} {i : Inst} (h : WFInst st i) : UniqueB st i = true := h.2.1

/-- D3 type agreement, per instruction. -/
theorem D3_typeOK {st : WFState} {i : Inst} (h : WFInst st i) : TypeOKB st i = true := h.2.2.1

/-- D4 storage, per instruction. -/
theorem D4_storageOK {st : WFState} {i : Inst} (h : WFInst st i) : StorageOKB st i = true :=
  h.2.2.2

/-- The prefix states paired with the instruction each one precedes. -/
def runStatesAux (st : WFState) : List Inst → List (WFState × Inst)
  | [] => []
  | i :: t => (st, i) :: runStatesAux (stepInst st i) t

/-- Every paired instruction satisfies `WFInst` at its prefix state. -/
theorem wfCode_runStates : ∀ (code : List Inst) (st : WFState),
    WFCode st code → ∀ p ∈ runStatesAux st code, WFInst p.1 p.2
  | [], _, _, p, hp => by simp [runStatesAux] at hp
  | i :: t, st, h, p, hp => by
      simp only [WFCode] at h
      simp only [runStatesAux, List.mem_cons] at hp
      rcases hp with rfl | hp
      · exact h.1
      · exact wfCode_runStates t (stepInst st i) h.2 p hp

/-- D2 at the module level: every instruction's used ids are defined by its prefix. -/
theorem WellFormed.defUses {m : Module} (h : WellFormed m) :
    ∀ p ∈ runStatesAux WFState.init m.code, DefUsesB p.1 p.2 = true :=
  fun p hp => (wfCode_runStates m.code WFState.init h p hp).1

/-- D2 at the module level: every result id is fresh. -/
theorem WellFormed.unique {m : Module} (h : WellFormed m) :
    ∀ p ∈ runStatesAux WFState.init m.code, UniqueB p.1 p.2 = true :=
  fun p hp => (wfCode_runStates m.code WFState.init h p hp).2.1

/-- D3 at the module level. -/
theorem WellFormed.typeOK {m : Module} (h : WellFormed m) :
    ∀ p ∈ runStatesAux WFState.init m.code, TypeOKB p.1 p.2 = true :=
  fun p hp => (wfCode_runStates m.code WFState.init h p hp).2.2.1

/-- D4 at the module level. -/
theorem WellFormed.storageOK {m : Module} (h : WellFormed m) :
    ∀ p ∈ runStatesAux WFState.init m.code, StorageOKB p.1 p.2 = true :=
  fun p hp => (wfCode_runStates m.code WFState.init h p hp).2.2.2

/-! ## The decoder -/

/-- SPIR-V magic number. -/
def spirvMagic : U32Word := 0x07230203

/-- SPIR-V 1.3 version word. -/
def spirvVersion13 : U32Word := 0x00010300

/-- Validate the five-word header (`magic · version · generator · bound · schema`) and return the
instruction words. -/
def parseHeader : List U32Word → Option (List U32Word)
  | magic :: version :: _gen :: bound :: schema :: rest =>
      if (magic == spirvMagic) && (version == spirvVersion13) && (schema == 0) && (bound != 0)
      then some rest
      else none
  | _ => none

/-- Decode a SPIR-V binary into a fragment module. Malformed headers, bad instruction boundaries,
unadmitted opcodes and failed id/type/storage checks return `none`. -/
def decode (ws : List U32Word) : Option Module :=
  match parseHeader ws with
  | none => none
  | some rest =>
      match parseInsts rest with
      | none => none
      | some code => if wfCheckB WFState.init code then some ⟨code⟩ else none

/-- A successful decode passed the linear check on its own instruction list. -/
theorem decode_wf {ws : List U32Word} {m : Module} (h : decode ws = some m) :
    wfCheckB WFState.init m.code = true := by
  simp only [decode] at h
  cases hp : parseHeader ws with
  | none =>
      simp only [hp] at h
      cases h
  | some rest =>
      simp only [hp] at h
      cases hq : parseInsts rest with
      | none =>
          simp only [hq] at h
          cases h
      | some code =>
          simp only [hq] at h
          split_ifs at h with hwf
          · injection h with hm
            rw [← hm]
            exact hwf

/-- Every admitted instruction's opcode is in the served set, so any decoded module is a fragment. -/
theorem inst_opcode_mem_served (i : Inst) : i.opcode ∈ servedOpcodes := by
  cases i <;> simp only [Inst.opcode] <;> decide

/-- **Decoder soundness.** A decoded module is `WellFormed` and `IsDp4aFragment`. Completeness —
that the emitter's output decodes — is the `Spirv.Check`/emitter-run seam. -/
theorem decode_sound {ws : List U32Word} {m : Module} (h : decode ws = some m) :
    WellFormed m ∧ IsDp4aFragment m := by
  refine ⟨?_, ?_⟩
  · exact wfCheckB_sound m.code WFState.init (decode_wf h)
  · intro i _
    exact inst_opcode_mem_served i

/-! ## Decode fixtures

A hand-built served-shape module (`OpCapability`, `OpMemoryModel`, `OpTypeVoid`, `OpTypeInt`,
`OpConstant`, `OpSUDot`, `OpReturn`, `OpFunctionEnd`), a module with a duplicated result id, a module
whose `OpExecutionMode`/`OpDecorate`/`OpMemberDecorate` operands are forward references, and a module
with an undefined id outside those exempt positions. They exercise the accept and refuse paths; the
negative control corrupts the `WordCount`, the result-id, an opcode entry and the forward-reference
exemption in a disposable copy. -/

/-- Instruction word from its `WordCount` and opcode. -/
def spvInst (wc op : Nat) : U32Word := wc * 65536 + op

/-- A served-shape module with one `OpSUDot`. -/
def validTestModule : List U32Word :=
  [spirvMagic, spirvVersion13, 0, 7, 0,
   spvInst 2 17, 1,
   spvInst 3 14, 0, 1,
   spvInst 2 19, 1,
   spvInst 4 21, 2, 32, 0,
   spvInst 4 43, 2, 3, 5,
   spvInst 6 4452, 2, 4, 3, 3, 0,
   spvInst 1 253,
   spvInst 1 56]

/-- Two `OpTypeVoid` instructions defining the same result id. -/
def duplicateTestModule : List U32Word :=
  [spirvMagic, spirvVersion13, 0, 3, 0,
   spvInst 2 19, 1,
   spvInst 2 19, 1,
   spvInst 1 56]

/-- A served-shape module exercising the SPIR-V 1.3 §2.4 forward references: `OpExecutionMode` names
`%4` before `OpFunction` defines it, `OpDecorate` names `%6` before `OpVariable` defines it, and
`OpMemberDecorate` names `%7` before `OpTypeStruct` defines it. -/
def forwardRefTestModule : List U32Word :=
  [spirvMagic, spirvVersion13, 0, 8, 0,
   spvInst 2 17, 1,                 -- OpCapability Shader
   spvInst 3 14, 0, 1,              -- OpMemoryModel Logical GLSL450
   spvInst 3 16, 4, 17,             -- OpExecutionMode %4 LocalSize (forward: %4 below)
   spvInst 3 71, 6, 2,              -- OpDecorate %6 RelaxedPrecision (forward: %6 below)
   spvInst 2 19, 1,                 -- %1 = OpTypeVoid
   spvInst 4 21, 2, 32, 0,          -- %2 = OpTypeInt 32 0
   spvInst 3 33, 3, 1,              -- %3 = OpTypeFunction %1
   spvInst 4 72, 7, 0, 2,           -- OpMemberDecorate %7 0 RelaxedPrecision (forward: %7)
   spvInst 3 30, 7, 2,              -- %7 = OpTypeStruct %2
   spvInst 4 32, 8, 7, 1,           -- %8 = OpTypePointer Function %1
   spvInst 4 59, 8, 6, 7,           -- %6 = OpVariable %8 Function
   spvInst 5 54, 1, 4, 0, 3,        -- %4 = OpFunction %1 None %3
   spvInst 1 56]                    -- OpFunctionEnd

/-- An `OpVariable` whose type id is never defined: an undefined use outside the exempt positions is
still refused (D2 def-before-use). -/
def undefinedUseTestModule : List U32Word :=
  [spirvMagic, spirvVersion13, 0, 4, 0,
   spvInst 4 59, 2, 5, 7,           -- %5 = OpVariable %2 Function, %2 undefined
   spvInst 1 56]

-- The served-shape module decodes (D1 boundaries, D2 ids, D3 `OpSUDot` types all pass).
#guard (decode validTestModule).isSome

-- The duplicated result id is refused (D2 uniqueness).
#guard !(decode duplicateTestModule).isSome

-- The forward-reference module decodes: the exempt positions are not D2 uses.
#guard (decode forwardRefTestModule).isSome

-- An undefined id outside the exempt positions is still refused (D2 def-before-use).
#guard !(decode undefinedUseTestModule).isSome

end Spirv
