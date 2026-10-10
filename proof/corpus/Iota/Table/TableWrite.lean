import Iota.Table.TableRead

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Table

open Iota Iota.Decode Iota.Machine Iota.Source Iota.Refinement

/-- The decoded `tableSlotValSet` shape: one value store, returning the value. -/
def TableSlotValSetShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["base", "slot", "value"] ∧
    declaration.body =
      [.store slotValueAddressExpr (.name "value"),
       .return_ (.name "value")]

/-- The decoded `tableInsertWith` shape. -/
def TableInsertWithShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["base", "cap", "key", "value", "merge"] ∧
    declaration.body =
      [.if_ (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0))
        [.return_ (.name "TABLE_EMPTY")] none,
       .if_ (.binary .eq (.name "key") (.name "TABLE_EMPTY"))
        [.return_ (.name "TABLE_EMPTY")] none,
       .if_ (.binary .bitOr
          (.binary .lt (.name "merge") (.name "MERGE_SET"))
          (.binary .gt (.name "merge") (.name "MERGE_ACC")))
        [.return_ (.name "TABLE_EMPTY")] none,
       .bind "mask" (.binary .sub (.name "cap") (.literal 1)),
       .bind "initSlot" (.binary .bitAnd (.call "tableSpread" [.name "key"])
         (.name "mask")),
       .bind "slot" (.call "tableFindSlotGo"
         [.name "base", .name "cap", .name "mask", .name "key", .name "initSlot",
          .literal 0]),
       .if_ (.binary .lt (.name "slot") (.literal 0))
         [.return_ (.binary .sub (.literal 0) (.literal 1))] none,
       .bind "oldKey" (.call "tableSlotKey" [.name "base", .name "slot"]),
       .bind "oldVal" (.call "tableSlotVal" [.name "base", .name "slot"]),
       .if_ (.binary .eq (.name "oldKey") (.name "TABLE_EMPTY"))
         [.bind "newVal" (.conditional (.name "TABLE_PRESENT")
            (.binary .eq (.name "merge") (.name "MERGE_SET")) (.name "value")),
          .expression (.call "tableSlotSet"
            [.name "base", .name "slot", .name "key", .name "newVal"]),
          .return_ (.name "newVal")] none,
       .bind "newVal" (.conditional (.name "TABLE_PRESENT")
         (.binary .eq (.name "merge") (.name "MERGE_SET")) (.name "value")),
       .if_ (.binary .eq (.name "merge") (.name "MERGE_ACC"))
         [.bind "newVal" (.binary .add (.name "oldVal") (.name "value"))] none,
       .return_ (.call "tableSlotValSet"
         [.name "base", .name "slot", .name "newVal"])]

/-- The exact decoded body of `tableInsertWith` as a definition. `EvalCall.run` then sees the
body definitionally, so `ExecList` composition needs no `Eq.rec` transport over a body equality. -/
def insertWithBody : List Stmt :=
  [.if_ (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0))
    [.return_ (.name "TABLE_EMPTY")] none,
   .if_ (.binary .eq (.name "key") (.name "TABLE_EMPTY"))
    [.return_ (.name "TABLE_EMPTY")] none,
   .if_ (.binary .bitOr
      (.binary .lt (.name "merge") (.name "MERGE_SET"))
      (.binary .gt (.name "merge") (.name "MERGE_ACC")))
    [.return_ (.name "TABLE_EMPTY")] none,
   .bind "mask" (.binary .sub (.name "cap") (.literal 1)),
   .bind "initSlot" (.binary .bitAnd (.call "tableSpread" [.name "key"])
     (.name "mask")),
   .bind "slot" (.call "tableFindSlotGo"
     [.name "base", .name "cap", .name "mask", .name "key", .name "initSlot",
      .literal 0]),
   .if_ (.binary .lt (.name "slot") (.literal 0))
     [.return_ (.binary .sub (.literal 0) (.literal 1))] none,
   .bind "oldKey" (.call "tableSlotKey" [.name "base", .name "slot"]),
   .bind "oldVal" (.call "tableSlotVal" [.name "base", .name "slot"]),
   .if_ (.binary .eq (.name "oldKey") (.name "TABLE_EMPTY"))
     [.bind "newVal" (.conditional (.name "TABLE_PRESENT")
        (.binary .eq (.name "merge") (.name "MERGE_SET")) (.name "value")),
      .expression (.call "tableSlotSet"
        [.name "base", .name "slot", .name "key", .name "newVal"]),
      .return_ (.name "newVal")] none,
   .bind "newVal" (.conditional (.name "TABLE_PRESENT")
     (.binary .eq (.name "merge") (.name "MERGE_SET")) (.name "value")),
   .if_ (.binary .eq (.name "merge") (.name "MERGE_ACC"))
     [.bind "newVal" (.binary .add (.name "oldVal") (.name "value"))] none,
   .return_ (.call "tableSlotValSet"
     [.name "base", .name "slot", .name "newVal"])]

/-- The exact decoded `tableInsertWith` declaration. The span is the decoded byte span of the
selected body; `realWriteShapes` gates it against the exported source. -/
def insertWithDecl : FunctionDecl :=
  { name := "tableInsertWith"
    parameters := ["base", "cap", "key", "value", "merge"]
    body := insertWithBody
    span := { start := 4763, stop := 5526 } }

theorem tableInsertWithShape_insertWithDecl : TableInsertWithShape insertWithDecl := by
  refine ⟨rfl, ?_⟩
  rfl

theorem member_cons_true (head other : Word) (tail : List Word) :
    member (head :: tail) other = true ↔ other = head ∨ member tail other = true := by
  show (match Word.decidableEq head other with
      | isTrue _ => true
      | isFalse _ => member tail other) = true ↔ other = head ∨ member tail other = true
  cases decision : Word.decidableEq head other with
  | isTrue equal =>
      exact ⟨fun _ => Or.inl equal.symm, fun _ => rfl⟩
  | isFalse notEqual =>
      constructor
      · intro present
        exact Or.inr present
      · rintro (equal | present)
        · exact False.elim (notEqual equal.symm)
        · exact present

theorem member_cons_false (head other : Word) (tail : List Word) :
    member (head :: tail) other = false ↔ other ≠ head ∧ member tail other = false := by
  show (match Word.decidableEq head other with
      | isTrue _ => true
      | isFalse _ => member tail other) = false ↔ other ≠ head ∧ member tail other = false
  cases decision : Word.decidableEq head other with
  | isTrue equal =>
      constructor
      · intro absurd
        exact False.elim (Bool.noConfusion absurd)
      · rintro ⟨ne, _⟩
        exact False.elim (ne equal.symm)
  | isFalse notEqual =>
      constructor
      · intro absent
        exact ⟨fun equal => notEqual equal.symm, absent⟩
      · intro pair
        exact pair.2

theorem member_cons_ne (head other : Word) (tail : List Word) (ne : other ≠ head) :
    member (head :: tail) other = member tail other := by
  show (match Word.decidableEq head other with
      | isTrue _ => true
      | isFalse _ => member tail other) = member tail other
  cases decision : Word.decidableEq head other with
  | isTrue equal => exact False.elim (ne equal.symm)
  | isFalse _ => rfl

theorem keyCoordinate_injective (base : Word) {left right : Nat}
    (equal : keyCoordinate base left = keyCoordinate base right) : left = right := by
  rcases Nat.lt_trichotomy left right with less | same | greater
  · exact False.elim (Nat.ne_of_lt (keyCoordinate_strict_mono base less) equal)
  · exact same
  · exact False.elim (Nat.ne_of_lt (keyCoordinate_strict_mono base greater) equal.symm)

theorem valueCoordinate_injective (base : Word) {left right : Nat}
    (equal : valueCoordinate base left = valueCoordinate base right) : left = right := by
  rcases Nat.lt_trichotomy left right with less | same | greater
  · exact False.elim (Nat.ne_of_lt (valueCoordinate_strict_mono base less) equal)
  · exact same
  · exact False.elim (Nat.ne_of_lt (valueCoordinate_strict_mono base greater) equal.symm)

/-- `valueCoordinate`'s `+1` offset stated so the `rep.owned_extent` result is syntactically the
goal. Matching `rep.owned_extent (2 * slot + 1)` directly against `valueCoordinate` makes the kernel
reduce `Nat.mul`/`Nat.add` on the `Nat.mod` term inside `probeSlot`, which exceeds the kernel
recursion limit; the explicit `base.val + (2 * slot + 1)` alignment avoids that reduction. -/
theorem owned_valueCoordinate {memory : Memory} {base cap default : Word}
    {occupied : List Word} {h : Word → Word}
    (rep : RepSong memory base cap default occupied h) (slot : Nat)
    (bound : 2 * slot + 1 < 2 * cap.val) :
    memory.owned (valueCoordinate base slot) = true := by
  have hcoord : valueCoordinate base slot = base.val + (2 * slot + 1) := by
    show (base.val + 2 * slot) + 1 = base.val + (2 * slot + 1)
    rw [Nat.add_assoc]
  rw [hcoord]
  exact rep.owned_extent (2 * slot + 1) bound

/-- The `isSome` form of `owned_valueCoordinate` for the initialized value cell. -/
theorem initialized_valueCoordinate {memory : Memory} {base cap default : Word}
    {occupied : List Word} {h : Word → Word}
    (rep : RepSong memory base cap default occupied h) (slot : Nat)
    (bound : 2 * slot + 1 < 2 * cap.val) :
    (memory.initialized (valueCoordinate base slot)).isSome := by
  have hcoord : valueCoordinate base slot = base.val + (2 * slot + 1) := by
    show (base.val + 2 * slot) + 1 = base.val + (2 * slot + 1)
    rw [Nat.add_assoc]
  rw [hcoord]
  exact rep.validExtent.2 (2 * slot + 1) bound

/-- Extract a witness from an `Option` by structural case analysis, without the library iff whose
proof uses proposition extensionality. -/
theorem option_exists_of_isSome {α : Type} (value : Option α) (present : value.isSome) :
    ∃ item, value = some item := by
  cases value with
  | none => exact False.elim (Bool.noConfusion present)
  | some item => exact ⟨item, rfl⟩

theorem keyCoordinate_ne_valueCoordinate (base : Word) (left right : Nat) :
    keyCoordinate base left ≠ valueCoordinate base right := by
  rcases Nat.lt_or_ge left right with less | lower
  · exact Nat.ne_of_lt (Nat.lt_of_le_of_lt (keyCoordinate_mono base (Nat.le_of_lt less))
      (keyCoordinate_lt_valueCoordinate base right))
  · rcases Nat.eq_or_lt_of_le lower with same | rightLess
    · rw [← same]
      exact Nat.ne_of_lt (keyCoordinate_lt_valueCoordinate base right)
    · exact Nat.ne_of_gt (Nat.lt_of_lt_of_le (valueCoordinate_lt_nextKey base right)
        (keyCoordinate_mono base (Nat.succ_le_of_lt rightLess)))

/-- Memory after storing one value at a slot's value coordinate. -/
def storeValueAt (memory : Memory) (base : Word) (slot : Nat) (value : Word) : Memory :=
  { owned := memory.owned
    initialized := fun coordinate =>
      if coordinate = valueCoordinate base slot then some value
      else memory.initialized coordinate }

/-- Memory after storing a key and value at a slot's two coordinates. -/
def storeSlotAt (memory : Memory) (base : Word) (slot : Nat) (key value : Word) : Memory :=
  { owned := memory.owned
    initialized := fun coordinate =>
      if coordinate = valueCoordinate base slot then some value
      else if coordinate = keyCoordinate base slot then some key
      else memory.initialized coordinate }

@[simp] theorem storeValueAt_owned (memory : Memory) (base : Word) (slot : Nat)
    (value : Word) : (storeValueAt memory base slot value).owned = memory.owned := rfl

@[simp] theorem storeSlotAt_owned (memory : Memory) (base : Word) (slot : Nat)
    (key value : Word) : (storeSlotAt memory base slot key value).owned = memory.owned := rfl

theorem storeValueAt_keyAt (memory : Memory) (base : Word) (slot : Nat) (value : Word)
    (other : Nat) :
    keyAt (storeValueAt memory base slot value) base other = keyAt memory base other := by
  simp only [keyAt, storeValueAt]
  rw [if_neg (keyCoordinate_ne_valueCoordinate base other slot)]

theorem storeValueAt_valueAt (memory : Memory) (base : Word) (slot : Nat) (value : Word)
    (other : Nat) :
    valueAt (storeValueAt memory base slot value) base other =
      if other = slot then some value else valueAt memory base other := by
  simp only [valueAt, storeValueAt]
  by_cases same : other = slot
  · rw [if_pos (congrArg (valueCoordinate base) same), if_pos same]
  · rw [if_neg (fun equal => same (valueCoordinate_injective base equal)), if_neg same]

theorem storeSlotAt_keyAt (memory : Memory) (base : Word) (slot : Nat) (key value : Word)
    (other : Nat) :
    keyAt (storeSlotAt memory base slot key value) base other =
      if other = slot then some key else keyAt memory base other := by
  simp only [keyAt, storeSlotAt]
  rw [if_neg (keyCoordinate_ne_valueCoordinate base other slot)]
  by_cases same : other = slot
  · rw [if_pos (congrArg (keyCoordinate base) same), if_pos same]
  · rw [if_neg (fun equal => same (keyCoordinate_injective base equal)), if_neg same]

theorem storeSlotAt_valueAt (memory : Memory) (base : Word) (slot : Nat) (key value : Word)
    (other : Nat) :
    valueAt (storeSlotAt memory base slot key value) base other =
      if other = slot then some value else valueAt memory base other := by
  simp only [valueAt, storeSlotAt]
  by_cases same : other = slot
  · rw [if_pos (congrArg (valueCoordinate base) same), if_pos same]
  · rw [if_neg (fun equal => same (valueCoordinate_injective base equal)),
      if_neg (keyCoordinate_ne_valueCoordinate base slot other).symm, if_neg same]

theorem validExtent_storeValueAt {memory : Memory} {base cap : Word}
    (valid : ValidExtent memory base cap) (slot : Nat)
    (value : Word) : ValidExtent (storeValueAt memory base slot value) base cap := by
  refine ⟨⟨valid.1.1, valid.1.2.1, fun offset bound => ?_⟩, fun offset bound => ?_⟩
  · rw [storeValueAt_owned]
    exact valid.1.2.2 offset bound
  · change (if base.val + offset = valueCoordinate base slot then some value
      else memory.initialized (base.val + offset)).isSome
    by_cases same : base.val + offset = valueCoordinate base slot
    · rw [if_pos same]
      rfl
    · rw [if_neg same]
      exact valid.2 offset bound

theorem validExtent_storeSlotAt {memory : Memory} {base cap : Word}
    (valid : ValidExtent memory base cap) (slot : Nat)
    (key value : Word) : ValidExtent (storeSlotAt memory base slot key value) base cap := by
  refine ⟨⟨valid.1.1, valid.1.2.1, fun offset bound => ?_⟩, fun offset bound => ?_⟩
  · rw [storeSlotAt_owned]
    exact valid.1.2.2 offset bound
  · change (if base.val + offset = valueCoordinate base slot then some value
      else if base.val + offset = keyCoordinate base slot then some key
      else memory.initialized (base.val + offset)).isSome
    by_cases valueSame : base.val + offset = valueCoordinate base slot
    · rw [if_pos valueSame]
      rfl
    · by_cases keySame : base.val + offset = keyCoordinate base slot
      · rw [if_neg valueSame, if_pos keySame]
        rfl
      · rw [if_neg valueSame, if_neg keySame]
        exact valid.2 offset bound

theorem insertWith_apply_self (f : Word → Word → Word) (key value : Word) (h : Word → Word) :
    Apeiron.Table.insertWith f key value h key = f value (h key) := by
  show (if key = key then f value (h key) else h key) = f value (h key)
  rw [if_pos rfl]

theorem insertWith_apply_ne (f : Word → Word → Word) (key value : Word) (h : Word → Word)
    {other : Word} (ne : other ≠ key) :
    Apeiron.Table.insertWith f key value h other = h other := by
  show (if other = key then f value (h other) else h other) = h other
  rw [if_neg ne]

theorem insertWith_replacement_self (key value : Word) (h : Word → Word) :
    Apeiron.Table.insertWith (fun new _old => new) key value h key = value := by
  rw [insertWith_apply_self]

theorem insertWith_replacement_ne (key value : Word) (h : Word → Word) {other : Word}
    (ne : other ≠ key) :
    Apeiron.Table.insertWith (fun new _old => new) key value h other = h other := by
  rw [insertWith_apply_ne (fun new _old => new) key value h ne]

theorem absent_no_physical_occurrence {memory : Memory} {base cap default : Word}
    {occupied : List Word} {h : Word → Word} (rep : RepSong memory base cap default occupied h)
    {key : Word} (keyNeEmpty : key ≠ tableEmpty) (absent : member occupied key = false)
    {slot : Nat} (slotBound : slot < cap.val) (keyAtSlot : keyAt memory base slot = some key) :
    False := by
  have present := (rep.2.2.2.2.1.2 key).2 ⟨slot, slotBound, keyAtSlot, keyNeEmpty⟩
  rw [absent] at present
  exact Bool.noConfusion present

theorem repSong_storeValueAt {memory : Memory} {base cap default : Word}
    {occupied : List Word} {h : Word → Word} (rep : RepSong memory base cap default occupied h)
    (slot : Nat) (slotBound : slot < cap.val) (key value : Word)
    (present : member occupied key = true) (keyAtSlot : keyAt memory base slot = some key) :
    RepSong (storeValueAt memory base slot value) base cap default occupied
      (Apeiron.Table.insertWith (fun new _old => new) key value h) := by
  refine ⟨rep.1, validExtent_storeValueAt rep.2.1 slot value, rep.2.2.1, ?_, ?_, ?_, ?_⟩
  · intro slot' slotBound'
    rw [storeValueAt_keyAt]
    exact rep.2.2.2.1 slot' slotBound'
  · constructor
    · intro i hi j hj k hk hki hkj
      rw [storeValueAt_keyAt] at hki hkj
      exact rep.2.2.2.2.1.1 i hi j hj k hk hki hkj
    · intro k
      constructor
      · intro memberK
        rcases (rep.2.2.2.2.1.2 k).1 memberK with ⟨slot', hbound, hkeyAt, hne⟩
        exact ⟨slot', hbound, by rw [storeValueAt_keyAt]; exact hkeyAt, hne⟩
      · intro occurrence
        rcases occurrence with ⟨slot', hbound, hkeyAt, hne⟩
        rw [storeValueAt_keyAt] at hkeyAt
        exact (rep.2.2.2.2.1.2 k).2 ⟨slot', hbound, hkeyAt, hne⟩
  · intro k memberK
    rcases rep.2.2.2.2.2.1 k memberK with ⟨step, hstep, hkeyAt, hprefix⟩
    refine ⟨step, hstep, ?_, ?_⟩
    · rw [storeValueAt_keyAt]
      exact hkeyAt
    · intro earlier hlt
      rw [storeValueAt_keyAt]
      exact hprefix earlier hlt
  · constructor
    · intro slot' slotBound' k memberK hkeyAt
      rw [storeValueAt_keyAt] at hkeyAt
      rw [storeValueAt_valueAt]
      by_cases hsame : slot' = slot
      · have hkEq : k = key := Option.some.inj (by rw [hsame] at hkeyAt; exact hkeyAt.symm.trans keyAtSlot)
        rw [hsame, hkEq, if_pos rfl, insertWith_replacement_self]
      · rw [if_neg hsame]
        have hvalue := rep.2.2.2.2.2.2.1 slot' slotBound' k memberK hkeyAt
        have hkNe : k ≠ key := by
          intro hkEq
          have hkeyAtKey : keyAt memory base slot' = some key := hkEq ▸ hkeyAt
          have sameSlot := rep.2.2.2.2.1.1 slot' slotBound' slot slotBound key
            (rep.2.2.1.2 key present) hkeyAtKey keyAtSlot
          exact hsame sameSlot
        rw [insertWith_replacement_ne key value h hkNe]
        exact hvalue
    · intro k absent
      have hkNe : k ≠ key := by
        intro hkEq
        rw [hkEq] at absent
        rw [present] at absent
        exact Bool.noConfusion absent
      rw [insertWith_replacement_ne key value h hkNe]
      exact rep.2.2.2.2.2.2.2 k absent

theorem repSong_storeSlotAt {memory : Memory} {base cap default : Word}
    {occupied : List Word} {h : Word → Word} (rep : RepSong memory base cap default occupied h)
    (slot step : Nat) (stepBound : step < cap.val) (key value : Word)
    (slotEq : slot = probeSlot cap key step)
    (keyNeEmpty : key ≠ tableEmpty) (absent : member occupied key = false)
    (emptyAt : keyAt memory base (probeSlot cap key step) = some tableEmpty)
    (clearPrefix : ∀ earlier, earlier < step →
      keyAt memory base (probeSlot cap key earlier) ≠ some tableEmpty) :
    RepSong (storeSlotAt memory base slot key value) base cap default (key :: occupied)
      (Apeiron.Table.insertWith (fun new _old => new) key value h) := by
  have slotBound : slot < cap.val := by
    rw [slotEq]
    exact Nat.mod_lt _ rep.positivePowerOfTwo.pos
  have slotEmpty : keyAt memory base slot = some tableEmpty := by
    rw [slotEq]
    exact emptyAt
  refine ⟨rep.1, validExtent_storeSlotAt rep.2.1 slot key value, ?_, ?_, ?_, ?_, ?_⟩
  · constructor
    · exact ⟨absent, rep.2.2.1.1⟩
    · intro other memberOther
      rcases (member_cons_true key other occupied).1 memberOther with same | memberOccupied
      · rw [same]
        exact keyNeEmpty
      · exact rep.2.2.1.2 other memberOccupied
  · intro slot' slotBound'
    by_cases hsame : slot' = slot
    · rw [hsame, storeSlotAt_keyAt, if_pos rfl]
      exact Or.inr ⟨key, (member_cons_true key key occupied).2 (Or.inl rfl), rfl⟩
    · rw [storeSlotAt_keyAt, if_neg hsame]
      rcases rep.2.2.2.1 slot' slotBound' with hempty | ⟨k, memberK, hkeyAt⟩
      · exact Or.inl hempty
      · exact Or.inr ⟨k, (member_cons_true key k occupied).2 (Or.inr memberK), hkeyAt⟩
  · constructor
    · intro i hi j hj k hk hki hkj
      rw [storeSlotAt_keyAt] at hki hkj
      by_cases hiSlot : i = slot
      · rw [if_pos hiSlot] at hki
        have hkEq : k = key := (Option.some.inj hki).symm
        by_cases hjSlot : j = slot
        · rw [hiSlot, hjSlot]
        · rw [if_neg hjSlot] at hkj
          rw [hkEq] at hkj
          exact False.elim (absent_no_physical_occurrence rep keyNeEmpty absent hj hkj)
      · rw [if_neg hiSlot] at hki
        by_cases hjSlot : j = slot
        · rw [if_pos hjSlot] at hkj
          have hkEq : k = key := (Option.some.inj hkj).symm
          rw [hkEq] at hki
          exact False.elim (absent_no_physical_occurrence rep keyNeEmpty absent hi hki)
        · rw [if_neg hjSlot] at hkj
          exact rep.2.2.2.2.1.1 i hi j hj k hk hki hkj
    · intro k
      constructor
      · intro memberK
        rcases (member_cons_true key k occupied).1 memberK with same | memberOccupied
        · rw [same]
          exact ⟨slot, slotBound, by rw [storeSlotAt_keyAt, if_pos rfl], keyNeEmpty⟩
        · rcases (rep.2.2.2.2.1.2 k).1 memberOccupied with ⟨slot0, hbound0, hkeyAt0, hne0⟩
          refine ⟨slot0, hbound0, ?_, hne0⟩
          rw [storeSlotAt_keyAt]
          by_cases hsame : slot0 = slot
          · exfalso
            rw [hsame] at hkeyAt0
            have hkEq : k = tableEmpty := Option.some.inj (hkeyAt0.symm.trans slotEmpty)
            exact hne0 hkEq
          · rw [if_neg hsame]
            exact hkeyAt0
      · intro occurrence
        rcases occurrence with ⟨slot0, hbound0, hkeyAt0, hne0⟩
        rw [storeSlotAt_keyAt] at hkeyAt0
        by_cases hsame : slot0 = slot
        · rw [if_pos hsame] at hkeyAt0
          have hkEq : k = key := (Option.some.inj hkeyAt0).symm
          rw [hkEq]
          exact (member_cons_true key key occupied).2 (Or.inl rfl)
        · rw [if_neg hsame] at hkeyAt0
          exact (member_cons_true key k occupied).2
            (Or.inr ((rep.2.2.2.2.1.2 k).2 ⟨slot0, hbound0, hkeyAt0, hne0⟩))
  · intro k memberK
    rcases (member_cons_true key k occupied).1 memberK with same | memberOccupied
    · rw [same]
      refine ⟨step, stepBound, ?_, ?_⟩
      · rw [← slotEq, storeSlotAt_keyAt, if_pos rfl]
      · intro earlier hlt
        rw [storeSlotAt_keyAt]
        by_cases hsame : probeSlot cap key earlier = slot
        · rw [if_pos hsame]
          exact fun equal => keyNeEmpty (Option.some.inj equal)
        · rw [if_neg hsame]
          exact clearPrefix earlier hlt
    · rcases rep.2.2.2.2.2.1 k memberOccupied with ⟨step0, hstep0, hkeyAt0, hprefix0⟩
      refine ⟨step0, hstep0, ?_, ?_⟩
      · rw [storeSlotAt_keyAt]
        by_cases hsame : probeSlot cap k step0 = slot
        · rw [hsame] at hkeyAt0
          have hkEq : k = tableEmpty := Option.some.inj (hkeyAt0.symm.trans slotEmpty)
          exact False.elim (rep.2.2.1.2 k memberOccupied hkEq)
        · rw [if_neg hsame]
          exact hkeyAt0
      · intro earlier hlt
        rw [storeSlotAt_keyAt]
        by_cases hsame : probeSlot cap k earlier = slot
        · rw [if_pos hsame]
          exact fun equal => keyNeEmpty (Option.some.inj equal)
        · rw [if_neg hsame]
          exact hprefix0 earlier hlt
  · constructor
    · intro slot' slotBound' k memberK hkeyAt
      rcases (member_cons_true key k occupied).1 memberK with same | memberOccupied
      · rw [same] at hkeyAt ⊢
        rw [storeSlotAt_valueAt]
        by_cases hsame : slot' = slot
        · rw [hsame, if_pos rfl, insertWith_replacement_self]
        · rw [if_neg hsame]
          rw [storeSlotAt_keyAt, if_neg hsame] at hkeyAt
          exact False.elim (absent_no_physical_occurrence rep keyNeEmpty absent slotBound' hkeyAt)
      · rw [storeSlotAt_valueAt]
        by_cases hsame : slot' = slot
        · exfalso
          rw [hsame] at hkeyAt
          rw [storeSlotAt_keyAt, if_pos rfl] at hkeyAt
          have hkEq : k = key := (Option.some.inj hkeyAt).symm
          rw [hkEq] at memberOccupied
          rw [absent] at memberOccupied
          exact Bool.noConfusion memberOccupied
        · rw [if_neg hsame]
          have hkeyAtMemory : keyAt memory base slot' = some k := by
            rw [storeSlotAt_keyAt, if_neg hsame] at hkeyAt
            exact hkeyAt
          have hkNe : k ≠ key := by
            intro hkEq
            rw [hkEq] at memberOccupied
            rw [absent] at memberOccupied
            exact Bool.noConfusion memberOccupied
          rw [insertWith_replacement_ne key value h hkNe]
          exact rep.2.2.2.2.2.2.1 slot' slotBound' k memberOccupied hkeyAtMemory
    · intro k absent'
      have hkNe : k ≠ key := by
        intro hkEq
        rw [hkEq] at absent'
        rw [(member_cons_true key key occupied).2 (Or.inl rfl)] at absent'
        exact Bool.noConfusion absent'
      have hnotOccupied : member occupied k = false := by
        cases hmem : member occupied k with
        | false => rfl
        | true =>
            rw [(member_cons_true key k occupied).2 (Or.inr hmem)] at absent'
            exact False.elim (Bool.noConfusion absent')
      rw [insertWith_replacement_ne key value h hkNe]
      exact rep.2.2.2.2.2.2.2 k hnotOccupied

theorem tableSlotValSet_execution (context : Context) (slotValSet : FunctionDecl)
    (hfunction : context.functions "tableSlotValSet" = some slotValSet)
    (hshape : TableSlotValSetShape slotValSet)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (memory : Memory) (base slot value : Word) (depth : Nat)
    (hvalueOwned : memory.owned (valueCoordinate base slot.val) = true)
    (hvalueCoordinate : valueCoordinate base slot.val < wordSignBit) :
    EvalCall context .unbounded depth "tableSlotValSet" [base, slot, value] memory
      (.value value (storeValueAt memory base slot.val value))
      [.call "tableSlotValSet" (depth + 1), .arithmetic .mul, .arithmetic .add,
        .arithmetic .add, .store (Word.ofNat (valueCoordinate base slot.val))] := by
  let environment : Environment := [("base", base), ("slot", slot), ("value", value)]
  have hframe : Environment.ofParameters slotValSet.parameters [base, slot, value] =
      some environment := by
    rw [hshape.1]
    rfl
  have hbase : Environment.lookup environment "base" = some base := by
    simp only [environment]
    exact Environment.lookup_bind_same _ _ _
  have hslot : Environment.lookup environment "slot" = some slot := by
    simp only [environment]
    change Environment.lookup (Environment.bind [("slot", slot), ("value", value)]
      "base" base) "slot" = some slot
    rw [Environment.lookup_bind_other _ "base" "slot" base (by decide)]
    exact Environment.lookup_bind_same _ _ _
  have hvalue : Environment.lookup environment "value" = some value := by
    simp only [environment]
    change Environment.lookup (Environment.bind [("slot", slot), ("value", value)]
      "base" base) "value" = some value
    rw [Environment.lookup_bind_other _ "base" "value" base (by decide)]
    change Environment.lookup (Environment.bind [("value", value)] "slot" slot)
      "value" = some value
    rw [Environment.lookup_bind_other _ "slot" "value" slot (by decide)]
    exact Environment.lookup_bind_same _ _ _
  have hstrideLocal : Environment.lookup environment "TABLE_STRIDE" = none := by
    simp only [environment]
    change Environment.lookup (Environment.bind [("slot", slot), ("value", value)]
      "base" base) "TABLE_STRIDE" = none
    rw [Environment.lookup_bind_other _ "base" "TABLE_STRIDE" base (by decide)]
    change Environment.lookup (Environment.bind [("value", value)] "slot" slot)
      "TABLE_STRIDE" = none
    rw [Environment.lookup_bind_other _ "slot" "TABLE_STRIDE" slot (by decide)]
    change Environment.lookup (Environment.bind [] "value" value) "TABLE_STRIDE" = none
    rw [Environment.lookup_bind_other _ "value" "TABLE_STRIDE" value (by decide)]
    rfl
  have hstore : memory.store (Word.ofNat (valueCoordinate base slot.val)) value =
      some (storeValueAt memory base slot.val value) := by
    unfold Memory.store
    rw [coordinate_ofNat_emptyCone _ hvalueCoordinate]
    change (if memory.owned (valueCoordinate base slot.val) = true then
      some (storeValueAt memory base slot.val value) else none) =
        some (storeValueAt memory base slot.val value)
    rw [hvalueOwned]
    exact if_pos rfl
  have hfirst :=
    ExecStmt.store (eval_slot_value_address context .unbounded (depth + 1) environment memory
      base slot hbase hslot hstrideLocal hstride) (EvalExpr.local hvalue) hstore
  have hbody : ExecList context .unbounded (depth + 1) environment memory slotValSet.body
      (.returned value (storeValueAt memory base slot.val value))
      [.arithmetic .mul, .arithmetic .add, .arithmetic .add,
        .store (Word.ofNat (valueCoordinate base slot.val))] := by
    rw [hshape.2]
    exact ExecList.next hfirst (ExecList.returned (ExecStmt.return_ (EvalExpr.local hvalue)))
  exact EvalCall.run hfunction hframe True.intro hbody

theorem bitwise_or_zero_zero (fuel : Nat) : Word.bitwise Bool.or fuel 0 0 = 0 := by
  induction fuel with
  | zero => rfl
  | succ fuel inductionHypothesis =>
      unfold Word.bitwise
      rw [inductionHypothesis]
      rfl

theorem bitOr_zero_zero : Word.bitOr (Word.ofNat 0) (Word.ofNat 0) = Word.ofNat 0 := by
  decide

theorem wordGt_false (left right : Word) (notGreater : ¬ Word.signed left > Word.signed right) :
    Word.gt left right = Word.ofNat 0 := by
  unfold Word.gt Word.bool
  rw [decide_eq_false notGreater]
  rfl

theorem wordLt_two_one : Word.lt (Word.ofNat 2) (Word.ofNat 1) = Word.ofNat 0 := by
  apply wordLt_false
  rw [signed_ofNat_emptyCone 2 (by decide), signed_ofNat_emptyCone 1 (by decide)]
  decide

theorem wordGt_two_three : Word.gt (Word.ofNat 2) (Word.ofNat 3) = Word.ofNat 0 := by
  apply wordGt_false
  rw [signed_ofNat_emptyCone 2 (by decide), signed_ofNat_emptyCone 3 (by decide)]
  decide

theorem ofNat_two_ne_zero : Word.ofNat 2 ≠ Word.ofNat 0 := by
  intro equal
  have hval : (2 : Nat) = 0 := by
    have := congrArg Fin.val equal
    rw [ofNat_val_emptyCone 2 (by decide), ofNat_val_emptyCone 0 (by decide)] at this
    exact this
  exact absurd hval (by decide)

theorem ofNat_two_ne_one : Word.ofNat 2 ≠ Word.ofNat 1 := by
  intro equal
  have hval : (2 : Nat) = 1 := by
    have := congrArg Fin.val equal
    rw [ofNat_val_emptyCone 2 (by decide), ofNat_val_emptyCone 1 (by decide)] at this
    exact this
  exact absurd hval (by decide)

theorem ofNat_two_ne_three : Word.ofNat 2 ≠ Word.ofNat 3 := by
  intro equal
  have hval : (2 : Nat) = 3 := by
    have := congrArg Fin.val equal
    rw [ofNat_val_emptyCone 2 (by decide), ofNat_val_emptyCone 3 (by decide)] at this
    exact this
  exact absurd hval (by decide)

theorem wordEq_two_zero : Word.eq (Word.ofNat 2) (Word.ofNat 0) = Word.ofNat 0 :=
  wordEq_ne _ _ ofNat_two_ne_zero

theorem wordEq_two_one : Word.eq (Word.ofNat 2) (Word.ofNat 1) = Word.ofNat 0 :=
  wordEq_ne _ _ ofNat_two_ne_one

theorem wordEq_two_three : Word.eq (Word.ofNat 2) (Word.ofNat 3) = Word.ofNat 0 :=
  wordEq_ne _ _ ofNat_two_ne_three

/-- A `tableFindSlotGo` probe stops at `stop` on a matching key or the first empty slot. -/
def ProbeStop (memory : Memory) (base cap key : Word) (stop : Nat) : Prop :=
  stop < cap.val ∧
    (keyAt memory base (probeSlot cap key stop) = some key ∨
     keyAt memory base (probeSlot cap key stop) = some tableEmpty) ∧
    ∀ earlier, earlier < stop →
      keyAt memory base (probeSlot cap key earlier) ≠ some tableEmpty ∧
      keyAt memory base (probeSlot cap key earlier) ≠ some key

/-- A `tableFindSlotGo` probe exhausted all `cap` slots without a match or an empty slot. -/
def ProbeExhausted (memory : Memory) (base cap key : Word) : Prop :=
  ∀ stop, stop < cap.val →
    keyAt memory base (probeSlot cap key stop) ≠ some tableEmpty ∧
    keyAt memory base (probeSlot cap key stop) ≠ some key

theorem ProbeOutcome.stop_of_true {memory : Memory} {base cap key : Word} {start : Nat}
    {raw : Word} (outcome : ProbeOutcome true memory base cap key start raw) :
    ∃ stop, start ≤ stop ∧ stop ≤ cap.val ∧
      (∀ t, start ≤ t → t < stop →
        keyAt memory base (probeSlot cap key t) ≠ some tableEmpty ∧
        keyAt memory base (probeSlot cap key t) ≠ some key) ∧
      ((stop = cap.val ∧ raw = tableEmpty) ∨
       (stop < cap.val ∧ raw = Word.ofNat (probeSlot cap key stop) ∧
         (keyAt memory base (probeSlot cap key stop) = some key ∨
          keyAt memory base (probeSlot cap key stop) = some tableEmpty))) := by
  induction outcome with
  | exhausted step hstep =>
      refine ⟨step, le_rfl, Nat.le_of_eq hstep, ?_, Or.inl ⟨hstep, rfl⟩⟩
      intro t ht hlt
      exact False.elim (Nat.not_lt_of_ge ht hlt)
  | empty step hstep hempty =>
      refine ⟨step, le_rfl, Nat.le_of_lt hstep, ?_, Or.inr ⟨hstep, rfl, Or.inr hempty⟩⟩
      intro t ht hlt
      exact False.elim (Nat.not_lt_of_ge ht hlt)
  | «match» step hstep hmatch =>
      refine ⟨step, le_rfl, Nat.le_of_lt hstep, ?_, Or.inr ⟨hstep, rfl, Or.inl hmatch⟩⟩
      intro t ht hlt
      exact False.elim (Nat.not_lt_of_ge ht hlt)
  | next step raw hstep hneEmpty hneKey later inductionHypothesis =>
      rcases inductionHypothesis with ⟨stop, hstartLe, hstopLe, hclear, hleaf⟩
      refine ⟨stop, Nat.le_trans (Nat.le_succ step) hstartLe, hstopLe, ?_, hleaf⟩
      intro t ht hlt
      by_cases htEq : t = step
      · subst htEq
        exact ⟨hneEmpty, hneKey⟩
      · have hsucc : step + 1 ≤ t :=
          Nat.succ_le_of_lt (Nat.lt_of_le_of_ne ht (Ne.symm htEq))
        exact hclear t hsucc hlt

theorem probeOutcome_admitted_or_exhausted {memory : Memory} {base cap key raw : Word}
    (outcome : ProbeOutcome true memory base cap key 0 raw) :
    (∃ stop, ProbeStop memory base cap key stop) ∨ ProbeExhausted memory base cap key := by
  rcases outcome.stop_of_true with ⟨stop, _, hstopLe, hclear, hleaf⟩
  rcases hleaf with ⟨hstopEq, _⟩ | ⟨hstopLt, _, hkeyOrEmpty⟩
  · exact Or.inr fun t hlt => hclear t (Nat.zero_le t) (hstopEq ▸ hlt)
  · exact Or.inl ⟨stop, hstopLt, hkeyOrEmpty,
      fun earlier hlt => hclear earlier (Nat.zero_le earlier) hlt⟩

theorem keyAt_congr {memory memory' : Memory} {base : Word}
    (initEq : ∀ coordinate, memory'.initialized coordinate = memory.initialized coordinate)
    (slot : Nat) : keyAt memory' base slot = keyAt memory base slot := by
  unfold keyAt
  exact initEq (keyCoordinate base slot)

theorem valueAt_congr {memory memory' : Memory} {base : Word}
    (initEq : ∀ coordinate, memory'.initialized coordinate = memory.initialized coordinate)
    (slot : Nat) : valueAt memory' base slot = valueAt memory base slot := by
  unfold valueAt
  exact initEq (valueCoordinate base slot)

/-- `ValidExtent` transfers along pointwise ownership and initialization equality. -/
theorem validExtent_congr {memory memory' : Memory} {base cap : Word}
    (valid : ValidExtent memory base cap) (ownedEq : memory'.owned = memory.owned)
    (initEq : ∀ coordinate, memory'.initialized coordinate = memory.initialized coordinate) :
    ValidExtent memory' base cap := by
  have backing : ValidBacking memory' base cap :=
    ⟨valid.1.1, valid.1.2.1,
      fun offset hoffset => by rw [ownedEq]; exact valid.1.2.2 offset hoffset⟩
  exact ⟨backing, fun offset hoffset => by rw [initEq]; exact valid.2 offset hoffset⟩

/-- `RepSong` depends only on the ownership and initialization functions, so pointwise equality
transfers it. This avoids a function-extensionality axiom. -/
theorem RepSong.congr {memory memory' : Memory} {base cap default : Word}
    {occupied : List Word} {h : Word → Word}
    (rep : RepSong memory base cap default occupied h)
    (ownedEq : memory'.owned = memory.owned)
    (initEq : ∀ coordinate, memory'.initialized coordinate = memory.initialized coordinate) :
    RepSong memory' base cap default occupied h := by
  refine ⟨rep.1, validExtent_congr rep.2.1 ownedEq initEq, rep.2.2.1, ?_, ?_, ?_, ?_⟩
  · intro slot slotBound
    rcases rep.2.2.2.1 slot slotBound with empty | ⟨key, memberKey, keyAtSlot⟩
    · exact Or.inl (by rw [keyAt_congr initEq]; exact empty)
    · exact Or.inr ⟨key, memberKey, by rw [keyAt_congr initEq]; exact keyAtSlot⟩
  · constructor
    · intro i hi j hj key keyNe keyI keyJ
      rw [keyAt_congr initEq] at keyI keyJ
      exact rep.2.2.2.2.1.1 i hi j hj key keyNe keyI keyJ
    · intro key
      constructor
      · intro present
        rcases (rep.2.2.2.2.1.2 key).1 present with ⟨slot, slotBound, keyAtSlot, keyNe⟩
        exact ⟨slot, slotBound, by rw [keyAt_congr initEq]; exact keyAtSlot, keyNe⟩
      · rintro ⟨slot, slotBound, keyAtSlot, keyNe⟩
        exact (rep.2.2.2.2.1.2 key).2
          ⟨slot, slotBound, by rw [keyAt_congr initEq] at keyAtSlot; exact keyAtSlot, keyNe⟩
  · intro key present
    rcases rep.2.2.2.2.2.1 key present with ⟨step, stepBound, keyAtStep, clearPrefix⟩
    refine ⟨step, stepBound, ?_, ?_⟩
    · rw [keyAt_congr initEq]
      exact keyAtStep
    · intro earlier hlt
      rw [keyAt_congr initEq]
      exact clearPrefix earlier hlt
  · constructor
    · intro slot slotBound key present keyAtSlot
      rw [valueAt_congr initEq]
      exact rep.2.2.2.2.2.2.1 slot slotBound key present
        (by rw [keyAt_congr initEq] at keyAtSlot; exact keyAtSlot)
    · intro key absent
      exact rep.2.2.2.2.2.2.2 key absent

/-- The branch a `MERGE_MAP` `tableInsertWith` call takes on the source. -/
inductive WriteMapOutcome (memory : Memory) (base cap default : Word) (occupied : List Word)
    (h : Word → Word) (key value : Word) : Word → Memory → Prop where
  | replaced (slot : Nat) (slotBound : slot < cap.val)
      (present : member occupied key = true) (keyAtSlot : keyAt memory base slot = some key) :
      WriteMapOutcome memory base cap default occupied h key value value
        (storeValueAt memory base slot value)
  | inserted (slot step : Nat) (stepBound : step < cap.val)
      (slotEq : slot = probeSlot cap key step) (absent : member occupied key = false)
      (emptyAt : keyAt memory base (probeSlot cap key step) = some tableEmpty)
      (clearPrefix : ∀ earlier, earlier < step →
        keyAt memory base (probeSlot cap key earlier) ≠ some tableEmpty)
      (after : Memory) (ownedEq : after.owned = memory.owned)
      (keyStore : after.initialized (keyCoordinate base slot) = some key)
      (valueStore : after.initialized (valueCoordinate base slot) = some value)
      (other : ∀ coordinate, coordinate ≠ keyCoordinate base slot →
        coordinate ≠ valueCoordinate base slot →
        after.initialized coordinate = memory.initialized coordinate) :
      WriteMapOutcome memory base cap default occupied h key value value after
  | refusedFull (full : ProbeExhausted memory base cap key) :
      WriteMapOutcome memory base cap default occupied h key value tableEmpty memory

/-- A checked outcome is one of the two replacement/insertion writes, with the post-memory and
the canonical `insertWith (fun new _old => new)` map, or the explicit full-table refusal returning
raw `tableEmpty` with unchanged memory. No raw-word test decides the branch. -/
theorem WriteMapOutcome.preserves {memory : Memory} {base cap default : Word}
    {occupied : List Word} {h : Word → Word} {key value raw : Word} {memory' : Memory}
    (rep : RepSong memory base cap default occupied h) (hkey : key ≠ tableEmpty)
    (outcome : WriteMapOutcome memory base cap default occupied h key value raw memory') :
    (∃ slot, slot < cap.val ∧ member occupied key = true ∧
        keyAt memory base slot = some key ∧ raw = value ∧
        memory' = storeValueAt memory base slot value ∧
        RepSong memory' base cap default occupied
          (Apeiron.Table.insertWith (fun new _old => new) key value h)) ∨
    (∃ slot step, step < cap.val ∧ slot = probeSlot cap key step ∧
        member occupied key = false ∧
        keyAt memory base (probeSlot cap key step) = some tableEmpty ∧ raw = value ∧
        RepSong memory' base cap default (key :: occupied)
          (Apeiron.Table.insertWith (fun new _old => new) key value h)) ∨
    (raw = tableEmpty ∧ memory' = memory ∧ ProbeExhausted memory base cap key) := by
  cases outcome with
  | replaced slot slotBound present keyAtSlot =>
      exact Or.inl ⟨slot, slotBound, present, keyAtSlot, rfl, rfl,
        repSong_storeValueAt rep slot slotBound key value present keyAtSlot⟩
  | inserted slot step stepBound slotEq absent emptyAt clearPrefix after ownedEq keyStore
      valueStore other =>
      have stored := repSong_storeSlotAt rep slot step stepBound key value slotEq hkey absent
        emptyAt clearPrefix
      have hinit : ∀ coordinate, memory'.initialized coordinate =
          (storeSlotAt memory base slot key value).initialized coordinate := by
        intro coordinate
        by_cases hvalue : coordinate = valueCoordinate base slot
        · subst hvalue
          rw [valueStore]
          change some value = valueAt (storeSlotAt memory base slot key value) base slot
          rw [storeSlotAt_valueAt, if_pos rfl]
        · by_cases hkey' : coordinate = keyCoordinate base slot
          · subst hkey'
            rw [keyStore]
            change some key = keyAt (storeSlotAt memory base slot key value) base slot
            rw [storeSlotAt_keyAt, if_pos rfl]
          · rw [other coordinate hkey' hvalue]
            change memory.initialized coordinate =
              (if coordinate = valueCoordinate base slot then some value
               else if coordinate = keyCoordinate base slot then some key
               else memory.initialized coordinate)
            rw [if_neg hvalue, if_neg hkey']
      exact Or.inr (Or.inl ⟨slot, step, stepBound, slotEq, absent, emptyAt, rfl,
        stored.congr ownedEq hinit⟩)
  | refusedFull full =>
      exact Or.inr (Or.inr ⟨rfl, rfl, full⟩)

theorem wordEq_one_zero : Word.eq (Word.ofNat 1) (Word.ofNat 0) = Word.ofNat 0 := by
  apply wordEq_ne
  intro equal
  have hval : (1 : Nat) = 0 := by
    have := congrArg Fin.val equal
    rw [ofNat_val_emptyCone 1 (by decide), ofNat_val_emptyCone 0 (by decide)] at this
    exact this
  exact absurd hval (by decide)

/-- Lookup through a `bind`: either the bound name or the previous environment. -/
theorem lookup_bind_other_dec (env : Environment) (name other : String) (value : Word) :
    Environment.lookup (Environment.bind env name value) other =
      if other = name then some value else Environment.lookup env other := by
  by_cases h : other = name
  · subst h
    rw [if_pos rfl]
    exact Environment.lookup_bind_same _ _ _
  · rw [if_neg h]
    exact Environment.lookup_bind_other env name other value h

attribute [simp] lookup_bind_other_dec

/-- The `tableInsertWith` frame as a `bind` chain, so lookups reduce by the bind lemmas. -/
def writeEnvironment (base cap key value : Word) : Environment :=
  Environment.bind (Environment.bind (Environment.bind (Environment.bind (Environment.bind []
    "merge" (Word.ofNat 2)) "value" value) "key" key) "cap" cap) "base" base

@[simp] theorem writeEnvironment_base (base cap key value : Word) :
    Environment.lookup (writeEnvironment base cap key value) "base" = some base := by
  unfold writeEnvironment
  exact Environment.lookup_bind_same _ _ _

@[simp] theorem writeEnvironment_cap (base cap key value : Word) :
    Environment.lookup (writeEnvironment base cap key value) "cap" = some cap := by
  unfold writeEnvironment
  rw [Environment.lookup_bind_other _ "base" "cap" base (by decide)]
  exact Environment.lookup_bind_same _ _ _

@[simp] theorem writeEnvironment_key (base cap key value : Word) :
    Environment.lookup (writeEnvironment base cap key value) "key" = some key := by
  unfold writeEnvironment
  rw [Environment.lookup_bind_other _ "base" "key" base (by decide)]
  rw [Environment.lookup_bind_other _ "cap" "key" cap (by decide)]
  exact Environment.lookup_bind_same _ _ _

@[simp] theorem writeEnvironment_value (base cap key value : Word) :
    Environment.lookup (writeEnvironment base cap key value) "value" = some value := by
  unfold writeEnvironment
  rw [Environment.lookup_bind_other _ "base" "value" base (by decide)]
  rw [Environment.lookup_bind_other _ "cap" "value" cap (by decide)]
  rw [Environment.lookup_bind_other _ "key" "value" key (by decide)]
  exact Environment.lookup_bind_same _ _ _

@[simp] theorem writeEnvironment_merge (base cap key value : Word) :
    Environment.lookup (writeEnvironment base cap key value) "merge" = some (Word.ofNat 2) := by
  unfold writeEnvironment
  rw [Environment.lookup_bind_other _ "base" "merge" base (by decide)]
  rw [Environment.lookup_bind_other _ "cap" "merge" cap (by decide)]
  rw [Environment.lookup_bind_other _ "key" "merge" key (by decide)]
  rw [Environment.lookup_bind_other _ "value" "merge" value (by decide)]
  exact Environment.lookup_bind_same _ _ _

@[simp] theorem writeEnvironment_none_of_ne (base cap key value : Word) (name : String)
    (hbase : name ≠ "base") (hcap : name ≠ "cap") (hkey : name ≠ "key")
    (hvalue : name ≠ "value") (hmerge : name ≠ "merge") :
    Environment.lookup (writeEnvironment base cap key value) name = none := by
  unfold writeEnvironment
  rw [Environment.lookup_bind_other _ "base" name base hbase]
  rw [Environment.lookup_bind_other _ "cap" name cap hcap]
  rw [Environment.lookup_bind_other _ "key" name key hkey]
  rw [Environment.lookup_bind_other _ "value" name value hvalue]
  rw [Environment.lookup_bind_other _ "merge" name (Word.ofNat 2) hmerge]
  rfl

theorem tableInsertWith_map_execution (context : Context)
    (capacityOk spread findProbe slotKey slotVal slotSet slotValSet : FunctionDecl)
    (hcapacityOk : context.functions "tableCapacityOk" = some capacityOk)
    (hspread : context.functions "tableSpread" = some spread)
    (hfindProbe : context.functions "tableFindSlotGo" = some findProbe)
    (hslotKey : context.functions "tableSlotKey" = some slotKey)
    (hslotVal : context.functions "tableSlotVal" = some slotVal)
    (hslotSet : context.functions "tableSlotSet" = some slotSet)
    (hslotValSet : context.functions "tableSlotValSet" = some slotValSet)
    (hinsert : context.functions "tableInsertWith" = some insertWithDecl)
    (hcapacityShape : TableCapacityOkShape capacityOk)
    (hspreadShape : TableSpreadShape spread)
    (hfindShape : TableFindSlotGoShape findProbe)
    (hslotKeyShape : TableSlotKeyShape slotKey)
    (hslotValShape : TableSlotValShape slotVal)
    (hslotSetShape : TableSlotSetShape slotSet)
    (hslotValSetShape : TableSlotValSetShape slotValSet)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : context.constants "TABLE_EMPTY" = some tableEmpty)
    (hmergeSet : context.constants "MERGE_SET" = some (Word.ofNat 1))
    (hmergeAcc : context.constants "MERGE_ACC" = some (Word.ofNat 3))
    (hphi : context.constants "TABLE_PHI" = some Word.tablePhi)
    (memory : Memory) (base cap default key value : Word) (occupied : List Word)
    (h : Word → Word) (rep : RepSong memory base cap default occupied h)
    (hkey : key ≠ tableEmpty) :
    ∃ raw memory' trace,
      EvalCall context .unbounded 0 "tableInsertWith"
        [base, cap, key, value, Word.ofNat 2] memory (.value raw memory') trace ∧
      WriteMapOutcome memory base cap default occupied h key value raw memory' := by
  let environment : Environment := writeEnvironment base cap key value
  have hframe : Environment.ofParameters insertWithDecl.parameters
      [base, cap, key, value, Word.ofNat 2] = some environment := by
    rfl
  have hbase : Environment.lookup environment "base" = some base :=
    writeEnvironment_base base cap key value
  have hcap : Environment.lookup environment "cap" = some cap :=
    writeEnvironment_cap base cap key value
  have hkeyLocal : Environment.lookup environment "key" = some key :=
    writeEnvironment_key base cap key value
  have hvalueLocal : Environment.lookup environment "value" = some value :=
    writeEnvironment_value base cap key value
  have hmergeLocal : Environment.lookup environment "merge" = some (Word.ofNat 2) :=
    writeEnvironment_merge base cap key value
  have hemptyLocal : Environment.lookup environment "TABLE_EMPTY" = none :=
    writeEnvironment_none_of_ne base cap key value "TABLE_EMPTY"
      (by decide) (by decide) (by decide) (by decide) (by decide)
  have hpresentLocal : Environment.lookup environment "TABLE_PRESENT" = none :=
    writeEnvironment_none_of_ne base cap key value "TABLE_PRESENT"
      (by decide) (by decide) (by decide) (by decide) (by decide)
  have hmergeSetLocal : Environment.lookup environment "MERGE_SET" = none :=
    writeEnvironment_none_of_ne base cap key value "MERGE_SET"
      (by decide) (by decide) (by decide) (by decide) (by decide)
  have hmergeAccLocal : Environment.lookup environment "MERGE_ACC" = none :=
    writeEnvironment_none_of_ne base cap key value "MERGE_ACC"
      (by decide) (by decide) (by decide) (by decide) (by decide)
  have hstrideLocal : Environment.lookup environment "TABLE_STRIDE" = none :=
    writeEnvironment_none_of_ne base cap key value "TABLE_STRIDE"
      (by decide) (by decide) (by decide) (by decide) (by decide)
  have hphiLocal : Environment.lookup environment "TABLE_PHI" = none :=
    writeEnvironment_none_of_ne base cap key value "TABLE_PHI"
      (by decide) (by decide) (by decide) (by decide) (by decide)
  have hzero : ¬ Word.nonzero (Word.ofNat 0) := fun hz => hz rfl
  have hcapSign : cap.val < wordSignBit :=
    cap_below_signBit_of_backing memory base cap rep.validExtent.1
  -- guard 1: capacity
  have hcapArgs : EvalArgs context .unbounded 1 environment memory [.name "cap"]
      (.value [cap] memory) [] := EvalArgs.cons (EvalExpr.local hcap) EvalArgs.nil
  have hcapCall := tableCapacityOk_execution context capacityOk hcapacityOk hcapacityShape
    memory cap 1 rep.positivePowerOfTwo hcapSign
  have hcapExpr : EvalExpr context .unbounded 1 environment memory
      (.call "tableCapacityOk" [.name "cap"]) (.value (Word.ofNat 1) memory)
      [.call "tableCapacityOk" 2, .comparison .le, .arithmetic .sub, .bitwise .bitAnd,
        .comparison .eq] := EvalExpr.call hcapArgs hcapCall
  have hcapGuardExpr : EvalExpr context .unbounded 1 environment memory
      (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0))
      (.value (Word.ofNat 0) memory)
      ([.call "tableCapacityOk" 2, .comparison .le, .arithmetic .sub, .bitwise .bitAnd,
        .comparison .eq] ++ [.comparison .eq]) := by
    rw [← wordEq_one_zero]
    exact EvalExpr.binary (operator := .eq) hcapExpr (EvalExpr.literal (literalValue := 0))
  have hcapGuard :=
    ExecStmt.ifFalse (thenBody := [.return_ (.name "TABLE_EMPTY")])
      (elseBody := (none : Option (List Stmt))) hcapGuardExpr hzero ExecList.nil
  -- guard 2: reserved key
  have hkeyEqZero : Word.eq key tableEmpty = Word.ofNat 0 := wordEq_ne key tableEmpty hkey
  have hkeyGuardExpr : EvalExpr context .unbounded 1 environment memory
      (.binary .eq (.name "key") (.name "TABLE_EMPTY")) (.value (Word.ofNat 0) memory)
      [.comparison .eq] := by
    rw [← hkeyEqZero]
    exact EvalExpr.binary (operator := .eq) (EvalExpr.local hkeyLocal)
      (EvalExpr.constant hemptyLocal hempty)
  have hkeyGuard :=
    ExecStmt.ifFalse (thenBody := [.return_ (.name "TABLE_EMPTY")])
      (elseBody := (none : Option (List Stmt))) hkeyGuardExpr hzero ExecList.nil
  -- guard 3: merge range
  have hltExpr : EvalExpr context .unbounded 1 environment memory
      (.binary .lt (.name "merge") (.name "MERGE_SET")) (.value (Word.ofNat 0) memory)
      [.comparison .lt] := by
    rw [← wordLt_two_one]
    exact EvalExpr.binary (operator := .lt) (EvalExpr.local hmergeLocal)
      (EvalExpr.constant hmergeSetLocal hmergeSet)
  have hgtExpr : EvalExpr context .unbounded 1 environment memory
      (.binary .gt (.name "merge") (.name "MERGE_ACC")) (.value (Word.ofNat 0) memory)
      [.comparison .gt] := by
    rw [← wordGt_two_three]
    exact EvalExpr.binary (operator := .gt) (EvalExpr.local hmergeLocal)
      (EvalExpr.constant hmergeAccLocal hmergeAcc)
  have hbitExpr : EvalExpr context .unbounded 1 environment memory
      (.binary .bitOr (.binary .lt (.name "merge") (.name "MERGE_SET"))
        (.binary .gt (.name "merge") (.name "MERGE_ACC")))
      (.value (Word.ofNat 0) memory)
      ([.comparison .lt] ++ [.comparison .gt] ++ [.bitwise .bitOr]) := by
    rw [← bitOr_zero_zero]
    exact EvalExpr.binary (operator := .bitOr) hltExpr hgtExpr
  have hmergeGuard :=
    ExecStmt.ifFalse (thenBody := [.return_ (.name "TABLE_EMPTY")])
      (elseBody := (none : Option (List Stmt))) hbitExpr hzero ExecList.nil
  -- mask and initSlot
  let mask := Word.sub cap (Word.ofNat 1)
  let maskEnvironment := Environment.bind environment "mask" mask
  have hmaskEval : EvalExpr context .unbounded 1 environment memory
      (.binary .sub (.name "cap") (.literal 1)) (.value mask memory) [.arithmetic .sub] :=
    EvalExpr.binary (operator := .sub) (EvalExpr.local hcap) EvalExpr.literal
  have hmaskBind := ExecStmt.bind (name := "mask") hmaskEval
  have hkeyMask : Environment.lookup maskEnvironment "key" = some key := by dsimp only [maskEnvironment, environment, writeEnvironment]; rw [Environment.lookup_bind_other _ "mask" "key" _ (by decide)] ; rw [Environment.lookup_bind_other _ "base" "key" _ (by decide)] ; rw [Environment.lookup_bind_other _ "cap" "key" _ (by decide)]; exact Environment.lookup_bind_same _ _ _
  have hmaskLocal : Environment.lookup maskEnvironment "mask" = some mask := by dsimp only [maskEnvironment, environment, writeEnvironment]; exact Environment.lookup_bind_same _ _ _
  rcases tableSpread_execution context spread hspread hspreadShape hphi memory key 1 with
    ⟨spreadTrace, hspreadCall⟩
  have hspreadArgs : EvalArgs context .unbounded 1 maskEnvironment memory [.name "key"]
      (.value [key] memory) [] := EvalArgs.cons (EvalExpr.local hkeyMask) EvalArgs.nil
  have hspreadExpr : EvalExpr context .unbounded 1 maskEnvironment memory
      (.call "tableSpread" [.name "key"]) (.value (Word.tableSpread key) memory) spreadTrace :=
    EvalExpr.call hspreadArgs hspreadCall
  let initSlot := Word.bitAnd (Word.tableSpread key) mask
  have hinitEval : EvalExpr context .unbounded 1 maskEnvironment memory
      (.binary .bitAnd (.call "tableSpread" [.name "key"]) (.name "mask"))
      (.value initSlot memory) (spreadTrace ++ [] ++ [.bitwise .bitAnd]) := by
    exact EvalExpr.binary (operator := .bitAnd) hspreadExpr (EvalExpr.local hmaskLocal)
  let initEnvironment := Environment.bind maskEnvironment "initSlot" initSlot
  have hinitBind :=
    ExecStmt.bind (name := "initSlot") hinitEval
  -- probe
  rcases tableFindSlotGo_execution context slotKey findProbe hslotKey hfindProbe hslotKeyShape
    hfindShape hstride hempty memory base cap default key occupied h 0 1 rep
    (Nat.zero_le cap.val) with ⟨slotRaw, findTrace, hfindCall, houtcome⟩
  have hbaseInit : Environment.lookup initEnvironment "base" = some base := by dsimp only [initEnvironment, maskEnvironment, environment, writeEnvironment]; rw [Environment.lookup_bind_other _ "initSlot" "base" _ (by decide)] ; rw [Environment.lookup_bind_other _ "mask" "base" _ (by decide)]; exact Environment.lookup_bind_same _ _ _
  have hcapInit : Environment.lookup initEnvironment "cap" = some cap := by dsimp only [initEnvironment, maskEnvironment, environment, writeEnvironment]; rw [Environment.lookup_bind_other _ "initSlot" "cap" _ (by decide)] ; rw [Environment.lookup_bind_other _ "mask" "cap" _ (by decide)] ; rw [Environment.lookup_bind_other _ "base" "cap" _ (by decide)]; exact Environment.lookup_bind_same _ _ _
  have hmaskInit : Environment.lookup initEnvironment "mask" = some mask := by dsimp only [initEnvironment, maskEnvironment, environment, writeEnvironment]; rw [Environment.lookup_bind_other _ "initSlot" "mask" _ (by decide)]; exact Environment.lookup_bind_same _ _ _
  have hkeyInit : Environment.lookup initEnvironment "key" = some key := by dsimp only [initEnvironment, maskEnvironment, environment, writeEnvironment]; rw [Environment.lookup_bind_other _ "initSlot" "key" _ (by decide)] ; rw [Environment.lookup_bind_other _ "mask" "key" _ (by decide)] ; rw [Environment.lookup_bind_other _ "base" "key" _ (by decide)] ; rw [Environment.lookup_bind_other _ "cap" "key" _ (by decide)]; exact Environment.lookup_bind_same _ _ _
  have hinitLocal : Environment.lookup initEnvironment "initSlot" = some initSlot := by dsimp only [initEnvironment, maskEnvironment, environment, writeEnvironment]; exact Environment.lookup_bind_same _ _ _
  have hslotArgs : EvalArgs context .unbounded 1 initEnvironment memory
      [.name "base", .name "cap", .name "mask", .name "key", .name "initSlot", .literal 0]
      (.value [base, cap, mask, key, initSlot, Word.ofNat 0] memory) [] :=
    EvalArgs.cons (EvalExpr.local hbaseInit)
      (EvalArgs.cons (EvalExpr.local hcapInit)
        (EvalArgs.cons (EvalExpr.local hmaskInit)
          (EvalArgs.cons (EvalExpr.local hkeyInit)
            (EvalArgs.cons (EvalExpr.local hinitLocal)
              (EvalArgs.cons EvalExpr.literal EvalArgs.nil)))))
  have hslotExpr : EvalExpr context .unbounded 1 initEnvironment memory
      (.call "tableFindSlotGo"
        [.name "base", .name "cap", .name "mask", .name "key", .name "initSlot", .literal 0])
      (.value slotRaw memory) findTrace := EvalExpr.call hslotArgs hfindCall
  let slotEnvironment := Environment.bind initEnvironment "slot" slotRaw
  have hslotBind := ExecStmt.bind (name := "slot") hslotExpr
  have hslotLocal : Environment.lookup slotEnvironment "slot" = some slotRaw :=
    Environment.lookup_bind_same _ _ _
  rcases houtcome.stop_of_true with ⟨stop, _, hstopLe, hclear, hleaf⟩
  have hclear0 : ∀ t, t < stop →
      keyAt memory base (probeSlot cap key t) ≠ some tableEmpty ∧
      keyAt memory base (probeSlot cap key t) ≠ some key :=
    fun t hlt => hclear t (Nat.zero_le t) hlt
  rcases hleaf with ⟨hstopEq, hrawEq⟩ | ⟨hstopLt, hrawEq, hkeyOrEmpty⟩
  · -- probe exhausted: explicit full-table refusal
    subst slotRaw
    have hexhausted : ProbeExhausted memory base cap key :=
      fun t hlt => hclear0 t (by rw [hstopEq]; exact hlt)
    have hltOne : Word.lt tableEmpty (Word.ofNat 0) = Word.ofNat 1 := tableEmpty_lt_zero
    have hguardExpr : EvalExpr context .unbounded 1 slotEnvironment memory
        (.binary .lt (.name "slot") (.literal 0)) (.value (Word.ofNat 1) memory)
        [.comparison .lt] := by
      rw [← hltOne]
      exact EvalExpr.binary (operator := .lt) (EvalExpr.local hslotLocal) EvalExpr.literal
    have hnonzero : Word.nonzero (Word.ofNat 1) := wordOne_nonzero
    have hguard : ExecStmt context .unbounded 1 slotEnvironment memory
        (.if_ (.binary .lt (.name "slot") (.literal 0))
          [.return_ (.binary .sub (.literal 0) (.literal 1))] (none : Option (List Stmt)))
        (.returned tableEmpty memory) [.comparison .lt, .arithmetic .sub] := by
      apply ExecStmt.ifTrue (elseBody := (none : Option (List Stmt))) hguardExpr hnonzero
      apply ExecList.returned
      apply ExecStmt.return_
      rw [← tableEmpty_sub]
      exact EvalExpr.binary (operator := .sub) (EvalExpr.literal (literalValue := 0))
        (EvalExpr.literal (literalValue := 1))
    exact ⟨tableEmpty, memory, _,
      EvalCall.run hinsert hframe True.intro
      ((ExecList.next hcapGuard
        (ExecList.next hkeyGuard
          (ExecList.next hmergeGuard
            (ExecList.next hmaskBind
              (ExecList.next hinitBind
                (ExecList.next hslotBind
                  (ExecList.returned hguard (rest :=
                    [.bind "oldKey" (.call "tableSlotKey" [.name "base", .name "slot"]),
                     .bind "oldVal" (.call "tableSlotVal" [.name "base", .name "slot"]),
                     .if_ (.binary .eq (.name "oldKey") (.name "TABLE_EMPTY"))
                       [.bind "newVal" (.conditional (.name "TABLE_PRESENT")
                          (.binary .eq (.name "merge") (.name "MERGE_SET")) (.name "value")),
                        .expression (.call "tableSlotSet"
                          [.name "base", .name "slot", .name "key", .name "newVal"]),
                        .return_ (.name "newVal")] none,
                     .bind "newVal" (.conditional (.name "TABLE_PRESENT")
                       (.binary .eq (.name "merge") (.name "MERGE_SET")) (.name "value")),
                     .if_ (.binary .eq (.name "merge") (.name "MERGE_ACC"))
                       [.bind "newVal" (.binary .add (.name "oldVal") (.name "value"))] none,
                     .return_ (.call "tableSlotValSet"
                       [.name "base", .name "slot", .name "newVal"])])))))))))
      ,
      WriteMapOutcome.refusedFull hexhausted⟩
  · -- probe stopped on a match or the first empty slot
    subst slotRaw
    have hstopSign : probeSlot cap key stop < wordSignBit :=
      lt_trans (Nat.mod_lt _ rep.positivePowerOfTwo.pos) hcapSign
    have hltZero : Word.lt (Word.ofNat (probeSlot cap key stop)) (Word.ofNat 0) =
        Word.ofNat 0 :=
      wordLt_ofNat_zero (probeSlot cap key stop) hstopSign
    have hguardExpr : EvalExpr context .unbounded 1 slotEnvironment memory
        (.binary .lt (.name "slot") (.literal 0)) (.value (Word.ofNat 0) memory)
        [.comparison .lt] := by
      rw [← hltZero]
      exact EvalExpr.binary (operator := .lt) (EvalExpr.local hslotLocal) EvalExpr.literal
    have hguard :=
      ExecStmt.ifFalse (thenBody := [.return_ (.binary .sub (.literal 0) (.literal 1))])
      (elseBody := (none : Option (List Stmt))) hguardExpr hzero ExecList.nil
    rcases hkeyOrEmpty with hkeyAt | hemptyAt
    · -- matching key: replacement write
      have hpresent : member occupied key = true :=
        (rep.2.2.2.2.1.2 key).2 ⟨probeSlot cap key stop, Nat.mod_lt _ rep.positivePowerOfTwo.pos,
          hkeyAt, hkey⟩
      have hslotKeyCall := tableSlotKey_execution context slotKey hslotKey hslotKeyShape hstride
        memory base (Word.ofNat (probeSlot cap key stop)) 1 key
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact keyCoordinate_below_signBit_emptyCone base cap (probeSlot cap key stop)
            rep.validExtent.1.2.1 (Nat.mod_lt _ rep.positivePowerOfTwo.pos))
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact rep.owned_extent (2 * probeSlot cap key stop)
            (double_lt_double (Nat.mod_lt _ rep.positivePowerOfTwo.pos)))
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact hkeyAt)
      rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign] at hslotKeyCall
      have holdKeyArgs : EvalArgs context .unbounded 1 slotEnvironment memory
          [.name "base", .name "slot"] (.value [base, Word.ofNat (probeSlot cap key stop)] memory) [] :=
        EvalArgs.cons (EvalExpr.local (by dsimp only [slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
          (EvalArgs.cons (EvalExpr.local hslotLocal) EvalArgs.nil)
      have holdKeyExpr : EvalExpr context .unbounded 1 slotEnvironment memory
          (.call "tableSlotKey" [.name "base", .name "slot"]) (.value key memory)
          [.call "tableSlotKey" 2, .arithmetic .mul, .arithmetic .add,
            .load (Word.ofNat (keyCoordinate base (probeSlot cap key stop)))] := by
        exact EvalExpr.call holdKeyArgs hslotKeyCall
      let oldKeyEnvironment := Environment.bind slotEnvironment "oldKey" key
      have holdKeyBind :=
        ExecStmt.bind (name := "oldKey") holdKeyExpr
      have holdValCall := tableSlotVal_execution context slotVal hslotVal hslotValShape hstride
        memory base (Word.ofNat (probeSlot cap key stop)) 1 (h key)
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact valueCoordinate_below_signBit_emptyCone base cap (probeSlot cap key stop)
            rep.validExtent.1.2.1 (Nat.mod_lt _ rep.positivePowerOfTwo.pos))
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact owned_valueCoordinate rep (probeSlot cap key stop) (by
            have doubled : 2 * probeSlot cap key stop + 2 ≤ 2 * cap.val := by
              have hbound : probeSlot cap key stop + 1 ≤ cap.val :=
                Nat.succ_le_of_lt (Nat.mod_lt _ rep.positivePowerOfTwo.pos)
              calc 2 * probeSlot cap key stop + 2 = 2 * (probeSlot cap key stop + 1) := by
                    rw [Nat.mul_succ]
                _ ≤ 2 * cap.val := Nat.mul_le_mul_left 2 hbound
            exact Nat.lt_of_lt_of_le (Nat.lt_succ_self (2 * probeSlot cap key stop + 1)) doubled))
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact rep.2.2.2.2.2.2.1 (probeSlot cap key stop)
            (Nat.mod_lt _ rep.positivePowerOfTwo.pos) key hpresent hkeyAt)
      rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign] at holdValCall
      have holdValArgs : EvalArgs context .unbounded 1 oldKeyEnvironment memory
          [.name "base", .name "slot"] (.value [base, Word.ofNat (probeSlot cap key stop)] memory) [] :=
        EvalArgs.cons (EvalExpr.local (by dsimp only [oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
          (EvalArgs.cons (EvalExpr.local (by dsimp only [oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl))) EvalArgs.nil)
      have holdValExpr : EvalExpr context .unbounded 1 oldKeyEnvironment memory
          (.call "tableSlotVal" [.name "base", .name "slot"]) (.value (h key) memory)
          [.call "tableSlotVal" 2, .arithmetic .mul, .arithmetic .add, .arithmetic .add,
            .load (Word.ofNat (valueCoordinate base (probeSlot cap key stop)))] := by
        exact EvalExpr.call holdValArgs holdValCall
      let oldValEnvironment := Environment.bind oldKeyEnvironment "oldVal" (h key)
      have holdValBind :=
        ExecStmt.bind (name := "oldVal") holdValExpr
      have hkeyNeEmpty : key ≠ tableEmpty := hkey
      have hguard5Expr : EvalExpr context .unbounded 1 oldValEnvironment memory
          (.binary .eq (.name "oldKey") (.name "TABLE_EMPTY")) (.value (Word.ofNat 0) memory)
          [.comparison .eq] := by
        rw [← wordEq_ne key tableEmpty hkeyNeEmpty]
        exact EvalExpr.binary (operator := .eq) (EvalExpr.local (by dsimp only [oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
          (EvalExpr.constant (by dsimp only [oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)) hempty)
      have hguard5 :=
        ExecStmt.ifFalse (thenBody :=
            [.bind "newVal" (.conditional (.name "TABLE_PRESENT")
               (.binary .eq (.name "merge") (.name "MERGE_SET")) (.name "value")),
             .expression (.call "tableSlotSet"
               [.name "base", .name "slot", .name "key", .name "newVal"]),
             .return_ (.name "newVal")])
          (elseBody := (none : Option (List Stmt))) hguard5Expr hzero ExecList.nil
      have hnewValEval : EvalExpr context .unbounded 1 oldValEnvironment memory
          (.conditional (.name "TABLE_PRESENT")
            (.binary .eq (.name "merge") (.name "MERGE_SET")) (.name "value"))
          (.value value memory) [.comparison .eq] := by
        have hcond : EvalExpr context .unbounded 1 oldValEnvironment memory
            (.binary .eq (.name "merge") (.name "MERGE_SET")) (.value (Word.ofNat 0) memory)
            [.comparison .eq] := by
          rw [← wordEq_two_one]
          exact EvalExpr.binary (operator := .eq) (EvalExpr.local (by dsimp only [oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
            (EvalExpr.constant (by dsimp only [oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)) hmergeSet)
        exact EvalExpr.conditionalFalse hcond hzero (EvalExpr.local (by dsimp only [oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
      let newValEnvironment := Environment.bind oldValEnvironment "newVal" value
      have hnewValBind := ExecStmt.bind (name := "newVal") hnewValEval
      have haccEval : EvalExpr context .unbounded 1 newValEnvironment memory
          (.binary .eq (.name "merge") (.name "MERGE_ACC")) (.value (Word.ofNat 0) memory)
          [.comparison .eq] := by
        rw [← wordEq_two_three]
        exact EvalExpr.binary (operator := .eq) (EvalExpr.local (by dsimp only [newValEnvironment, oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
          (EvalExpr.constant (by dsimp only [newValEnvironment, oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)) hmergeAcc)
      have haccIf :=
        ExecStmt.ifFalse (thenBody := [.bind "newVal" (.binary .add (.name "oldVal") (.name "value"))])
          (elseBody := (none : Option (List Stmt))) haccEval hzero ExecList.nil
      have hvalSetCall := tableSlotValSet_execution context slotValSet hslotValSet
        hslotValSetShape hstride memory base (Word.ofNat (probeSlot cap key stop)) value 1
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact owned_valueCoordinate rep (probeSlot cap key stop) (by
            have doubled : 2 * probeSlot cap key stop + 2 ≤ 2 * cap.val := by
              have hbound : probeSlot cap key stop + 1 ≤ cap.val :=
                Nat.succ_le_of_lt (Nat.mod_lt _ rep.positivePowerOfTwo.pos)
              calc 2 * probeSlot cap key stop + 2 = 2 * (probeSlot cap key stop + 1) := by
                    rw [Nat.mul_succ]
                _ ≤ 2 * cap.val := Nat.mul_le_mul_left 2 hbound
            exact Nat.lt_of_lt_of_le (Nat.lt_succ_self (2 * probeSlot cap key stop + 1)) doubled))
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact valueCoordinate_below_signBit_emptyCone base cap (probeSlot cap key stop)
            rep.validExtent.1.2.1 (Nat.mod_lt _ rep.positivePowerOfTwo.pos))
      rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign] at hvalSetCall
      have hvalSetArgs : EvalArgs context .unbounded 1 newValEnvironment memory
          [.name "base", .name "slot", .name "newVal"]
          (.value [base, Word.ofNat (probeSlot cap key stop), value] memory) [] :=
        EvalArgs.cons (EvalExpr.local (by dsimp only [newValEnvironment, oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
          (EvalArgs.cons (EvalExpr.local (by dsimp only [newValEnvironment, oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
            (EvalArgs.cons (EvalExpr.local (by dsimp only [newValEnvironment, oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl))) EvalArgs.nil))
      have hreturn := by
        apply ExecStmt.return_
        exact EvalExpr.call hvalSetArgs hvalSetCall
      exact ⟨value, storeValueAt memory base (probeSlot cap key stop) value, _,
        EvalCall.run hinsert hframe True.intro ((ExecList.next hcapGuard
          (ExecList.next hkeyGuard
            (ExecList.next hmergeGuard
              (ExecList.next hmaskBind
                (ExecList.next hinitBind
                  (ExecList.next hslotBind
                    (ExecList.next hguard
                      (ExecList.next holdKeyBind
                        (ExecList.next holdValBind
                          (ExecList.next hguard5
                            (ExecList.next hnewValBind
                              (ExecList.next haccIf
                                (ExecList.returned hreturn (rest := []))))))))))))))),
        WriteMapOutcome.replaced (probeSlot cap key stop)
          (Nat.mod_lt _ rep.positivePowerOfTwo.pos) hpresent hkeyAt⟩
    · -- first empty slot: fresh insertion
      have habsent : member occupied key = false := by
        cases hmem : member occupied key with
        | false => rfl
        | true =>
            rcases rep.2.2.2.2.2.1 key hmem with ⟨step', hstep', hkeyAt', hprefix'⟩
            have hstepLe : step' ≤ stop := by
              by_contra hnot
              have hlt : stop < step' := Nat.lt_of_not_ge hnot
              exact (hprefix' stop hlt) hemptyAt
            by_cases hlt : step' < stop
            · exact False.elim ((hclear0 step' hlt).2 hkeyAt')
            · have hEq : step' = stop := Nat.le_antisymm hstepLe (Nat.le_of_not_gt hlt)
              exact False.elim (hkey (Option.some.inj (hkeyAt'.symm.trans (hEq ▸ hemptyAt))))
      have hslotSetCall := tableSlotSet_execution context slotSet hslotSet hslotSetShape hstride
        memory base (Word.ofNat (probeSlot cap key stop)) key value 1
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact rep.owned_extent (2 * probeSlot cap key stop)
            (double_lt_double (Nat.mod_lt _ rep.positivePowerOfTwo.pos)))
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact owned_valueCoordinate rep (probeSlot cap key stop) (by
            have doubled : 2 * probeSlot cap key stop + 2 ≤ 2 * cap.val := by
              have hbound : probeSlot cap key stop + 1 ≤ cap.val :=
                Nat.succ_le_of_lt (Nat.mod_lt _ rep.positivePowerOfTwo.pos)
              calc 2 * probeSlot cap key stop + 2 = 2 * (probeSlot cap key stop + 1) := by
                    rw [Nat.mul_succ]
                _ ≤ 2 * cap.val := Nat.mul_le_mul_left 2 hbound
            exact Nat.lt_of_lt_of_le (Nat.lt_succ_self (2 * probeSlot cap key stop + 1)) doubled))
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact keyCoordinate_below_signBit_emptyCone base cap (probeSlot cap key stop)
            rep.validExtent.1.2.1 (Nat.mod_lt _ rep.positivePowerOfTwo.pos))
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact valueCoordinate_below_signBit_emptyCone base cap (probeSlot cap key stop)
            rep.validExtent.1.2.1 (Nat.mod_lt _ rep.positivePowerOfTwo.pos))
      rcases hslotSetCall with ⟨afterSlot, hslotSetCall, hslotOwned, hslotKeyStore, hslotValueStore,
        hslotOther⟩
      rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign] at hslotKeyStore hslotValueStore hslotOther
      have hkeyAtEmpty : keyAt memory base (probeSlot cap key stop) = some tableEmpty := hemptyAt
      have hslotKeyCall := tableSlotKey_execution context slotKey hslotKey hslotKeyShape hstride
        memory base (Word.ofNat (probeSlot cap key stop)) 1 tableEmpty
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact keyCoordinate_below_signBit_emptyCone base cap (probeSlot cap key stop)
            rep.validExtent.1.2.1 (Nat.mod_lt _ rep.positivePowerOfTwo.pos))
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact rep.owned_extent (2 * probeSlot cap key stop)
            (double_lt_double (Nat.mod_lt _ rep.positivePowerOfTwo.pos)))
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact hkeyAtEmpty)
      rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign] at hslotKeyCall
      have holdKeyArgs : EvalArgs context .unbounded 1 slotEnvironment memory
          [.name "base", .name "slot"] (.value [base, Word.ofNat (probeSlot cap key stop)] memory) [] :=
        EvalArgs.cons (EvalExpr.local (by dsimp only [slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
          (EvalArgs.cons (EvalExpr.local hslotLocal) EvalArgs.nil)
      have holdKeyExpr : EvalExpr context .unbounded 1 slotEnvironment memory
          (.call "tableSlotKey" [.name "base", .name "slot"]) (.value tableEmpty memory)
          [.call "tableSlotKey" 2, .arithmetic .mul, .arithmetic .add,
            .load (Word.ofNat (keyCoordinate base (probeSlot cap key stop)))] := by
        exact EvalExpr.call holdKeyArgs hslotKeyCall
      let oldKeyEnvironment := Environment.bind slotEnvironment "oldKey" tableEmpty
      have holdKeyBind :=
        ExecStmt.bind (name := "oldKey") holdKeyExpr
      have hvalSome : (memory.initialized (valueCoordinate base (probeSlot cap key stop))).isSome :=
        initialized_valueCoordinate rep (probeSlot cap key stop) (by
          have doubled : 2 * probeSlot cap key stop + 2 ≤ 2 * cap.val := by
            have hbound : probeSlot cap key stop + 1 ≤ cap.val :=
              Nat.succ_le_of_lt (Nat.mod_lt _ rep.positivePowerOfTwo.pos)
            calc 2 * probeSlot cap key stop + 2 = 2 * (probeSlot cap key stop + 1) := by
                  rw [Nat.mul_succ]
              _ ≤ 2 * cap.val := Nat.mul_le_mul_left 2 hbound
          exact Nat.lt_of_lt_of_le (Nat.lt_succ_self (2 * probeSlot cap key stop + 1)) doubled)
      rcases option_exists_of_isSome _ hvalSome with ⟨oldVal, hvalEq⟩
      have holdValCall := tableSlotVal_execution context slotVal hslotVal hslotValShape hstride
        memory base (Word.ofNat (probeSlot cap key stop)) 1 oldVal
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact valueCoordinate_below_signBit_emptyCone base cap (probeSlot cap key stop)
            rep.validExtent.1.2.1 (Nat.mod_lt _ rep.positivePowerOfTwo.pos))
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact owned_valueCoordinate rep (probeSlot cap key stop) (by
            have doubled : 2 * probeSlot cap key stop + 2 ≤ 2 * cap.val := by
              have hbound : probeSlot cap key stop + 1 ≤ cap.val :=
                Nat.succ_le_of_lt (Nat.mod_lt _ rep.positivePowerOfTwo.pos)
              calc 2 * probeSlot cap key stop + 2 = 2 * (probeSlot cap key stop + 1) := by
                    rw [Nat.mul_succ]
                _ ≤ 2 * cap.val := Nat.mul_le_mul_left 2 hbound
            exact Nat.lt_of_lt_of_le (Nat.lt_succ_self (2 * probeSlot cap key stop + 1))
              doubled))
        (by
          rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign]
          exact hvalEq)
      rw [ofNat_val_emptyCone (probeSlot cap key stop) hstopSign] at holdValCall
      have holdValArgs : EvalArgs context .unbounded 1 oldKeyEnvironment memory
          [.name "base", .name "slot"] (.value [base, Word.ofNat (probeSlot cap key stop)] memory) [] :=
        EvalArgs.cons (EvalExpr.local (by dsimp only [oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
          (EvalArgs.cons (EvalExpr.local (by dsimp only [oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl))) EvalArgs.nil)
      have holdValExpr : EvalExpr context .unbounded 1 oldKeyEnvironment memory
          (.call "tableSlotVal" [.name "base", .name "slot"]) (.value oldVal memory)
          [.call "tableSlotVal" 2, .arithmetic .mul, .arithmetic .add, .arithmetic .add,
            .load (Word.ofNat (valueCoordinate base (probeSlot cap key stop)))] := by
        exact EvalExpr.call holdValArgs holdValCall
      let oldValEnvironment := Environment.bind oldKeyEnvironment "oldVal" oldVal
      have holdValBind :=
        ExecStmt.bind (name := "oldVal") holdValExpr
      have hguard5Expr : EvalExpr context .unbounded 1 oldValEnvironment memory
          (.binary .eq (.name "oldKey") (.name "TABLE_EMPTY"))
          (.value (Word.ofNat 1) memory) [.comparison .eq] := by
        rw [← wordEq_self tableEmpty]
        exact EvalExpr.binary (operator := .eq) (EvalExpr.local (by dsimp only [oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
          (EvalExpr.constant (by dsimp only [oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)) hempty)
      have hguard5 := by
        apply ExecStmt.ifTrue (elseBody := (none : Option (List Stmt))) hguard5Expr wordOne_nonzero
        have hnewValEval : EvalExpr context .unbounded 1 oldValEnvironment memory
            (.conditional (.name "TABLE_PRESENT")
              (.binary .eq (.name "merge") (.name "MERGE_SET")) (.name "value"))
            (.value value memory) [.comparison .eq] := by
          have hcond : EvalExpr context .unbounded 1 oldValEnvironment memory
              (.binary .eq (.name "merge") (.name "MERGE_SET"))
              (.value (Word.ofNat 0) memory) [.comparison .eq] := by
            rw [← wordEq_two_one]
            exact EvalExpr.binary (operator := .eq) (EvalExpr.local (by dsimp only [oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
              (EvalExpr.constant (by dsimp only [oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)) hmergeSet)
          exact EvalExpr.conditionalFalse hcond hzero (EvalExpr.local (by dsimp only [oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
        let newValEnvironment := Environment.bind oldValEnvironment "newVal" value
        have hnewValBind := ExecStmt.bind (name := "newVal") hnewValEval
        have hsetArgs : EvalArgs context .unbounded 1 newValEnvironment memory
            [.name "base", .name "slot", .name "key", .name "newVal"]
            (.value [base, Word.ofNat (probeSlot cap key stop), key, value] memory) [] :=
          EvalArgs.cons (EvalExpr.local (by dsimp only [newValEnvironment, oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
            (EvalArgs.cons (EvalExpr.local (by dsimp only [newValEnvironment, oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
              (EvalArgs.cons (EvalExpr.local (by dsimp only [newValEnvironment, oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl)))
                (EvalArgs.cons (EvalExpr.local (by dsimp only [newValEnvironment, oldValEnvironment, oldKeyEnvironment, slotEnvironment, initEnvironment, maskEnvironment, environment, writeEnvironment]; repeat (first | rw [Environment.lookup_bind_other _ _ _ _ (by decide)] | rw [Environment.lookup_bind_same _ _ _] | rfl))) EvalArgs.nil)))
        have hsetExpr : EvalExpr context .unbounded 1 newValEnvironment memory
            (.call "tableSlotSet" [.name "base", .name "slot", .name "key", .name "newVal"])
            (.value value afterSlot) (slotSetTrace 1 base (Word.ofNat (probeSlot cap key stop))) := by
          exact EvalExpr.call hsetArgs hslotSetCall
        have hsetStmt :=
          ExecStmt.expression hsetExpr
        have hretStmt : ExecStmt context .unbounded 1 newValEnvironment afterSlot
            (.return_ (.name "newVal")) (.returned value afterSlot) [] :=
          ExecStmt.return_ (EvalExpr.local (Environment.lookup_bind_same _ _ _))
        have hbody := ExecList.next hnewValBind
          (ExecList.next hsetStmt (ExecList.returned hretStmt (rest := [])))
        exact hbody
      exact ⟨value, afterSlot, _,
        EvalCall.run hinsert hframe True.intro ((ExecList.next hcapGuard
          (ExecList.next hkeyGuard
            (ExecList.next hmergeGuard
              (ExecList.next hmaskBind
                (ExecList.next hinitBind
                  (ExecList.next hslotBind
                    (ExecList.next hguard
                      (ExecList.next holdKeyBind
                        (ExecList.next holdValBind
                          (ExecList.returned hguard5 (rest :=
                            [.bind "newVal" (.conditional (.name "TABLE_PRESENT")
                               (.binary .eq (.name "merge") (.name "MERGE_SET")) (.name "value")),
                             .if_ (.binary .eq (.name "merge") (.name "MERGE_ACC"))
                               [.bind "newVal" (.binary .add (.name "oldVal") (.name "value"))] none,
                             .return_ (.call "tableSlotValSet"
                               [.name "base", .name "slot", .name "newVal"])]))))))))))))
        ,
        WriteMapOutcome.inserted (probeSlot cap key stop) stop
          hstopLt rfl habsent hemptyAt
          (fun earlier hlt => (hclear0 earlier hlt).1)
          afterSlot hslotOwned hslotKeyStore hslotValueStore hslotOther⟩

/-- Compiled test: the real exported source decodes to the concrete `insertWithDecl` (name,
parameters, body and span). Not acceptance; a `#guard` runs the compiled evaluator.

Temporal O(n + m) for the table/std source byte lengths: `Decode.decodeClosure` runs two linear
decodes plus one constant comparison and append, and the `functions.find?` scan is linear in the
decoded declaration count. Spatial O(n + m) for the decoded character, token and declaration lists.
Operations counted: source bytes decoded, decoded declarations scanned. -/
def realWriteShapes : Bool :=
  match Decode.decodeClosure
      (encodeBytes Iota.Generated.TableDerivation.tableSource)
      (encodeBytes Iota.Generated.TableDerivation.stdSource) with
  | .error _ => false
  | .ok value =>
      match value.val.program.functions.find?
        (fun declaration => declaration.name == "tableInsertWith") with
      | some declaration => declaration == insertWithDecl
      | none => false

#guard realWriteShapes

end Iota.Table
