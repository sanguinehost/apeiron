/- Fixed-home-shard memo storage.  Searches are structurally bounded by `N`. -/
import Slang.Memo.Value
import Slang.Address.Faces
import Apeiron.Table.Basic

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 10000
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Memo

open Slang

structure MemoEntry (sources : Slang.Source.SourceArena) {stateCapacity resultCapacity : Nat}
    (states : Slang.Arena.StateArena stateCapacity) (results : Slang.Arena.ResultArena resultCapacity) where
  key : MemoKey sources states
  parsed : Slang.Source.ParsedRef sources
  parsedSource : parsed.ref = key.source
  value : MemoValue sources states results
  consumedSource : value.consumed = key.source
  transition : TransitionDenotes parsed key value

structure MemoStore (sources : Slang.Source.SourceArena) {stateCapacity resultCapacity : Nat}
    (states : Slang.Arena.StateArena stateCapacity) (results : Slang.Arena.ResultArena resultCapacity) (N : Nat) where
  defaultValue : MemoValue sources states results
  slots : Slang.Address.Shard → Fin N → Option (MemoEntry sources states results)

def empty {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (defaultValue : MemoValue sources states results) : MemoStore sources states results N :=
  { defaultValue := defaultValue, slots := fun _ _ => none }

def update {α β : Type} [DecidableEq α] (f : α → β) (index : α) (value : β) (query : α) : β :=
  if query = index then value else f query

theorem update_same {α β : Type} [DecidableEq α] (f : α → β) (index : α) (value : β) :
    update f index value index = value := by
  unfold update
  rw [if_pos rfl]

theorem update_other {α β : Type} [DecidableEq α] (f : α → β) (index query : α) (value : β) (different : query ≠ index) :
    update f index value query = f query := by
  unfold update
  rw [if_neg different]

private def finZero (n : Nat) : Fin (Nat.succ n) := ⟨0, Nat.zero_lt_succ n⟩

private def finSucc {n : Nat} (i : Fin n) : Fin (Nat.succ n) :=
  ⟨Nat.succ i.val, Nat.succ_lt_succ i.isLt⟩

private def findKeySlotGo {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (N : Nat) (slots : Fin N → Option (MemoEntry sources states results)) (key : MemoKey sources states) :
    Option (Fin N) :=
  match N with
  | 0 => none
  | Nat.succ n =>
      match slots (finZero n) with
      | some entry => if MemoKey.equal entry.key key then some (finZero n) else
          match findKeySlotGo n (fun i => slots (finSucc i)) key with
          | some i => some (finSucc i)
          | none => none
      | none =>
          match findKeySlotGo n (fun i => slots (finSucc i)) key with
          | some i => some (finSucc i)
          | none => none
termination_by structural N

def findKeySlot {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (slots : Fin N → Option (MemoEntry sources states results)) (key : MemoKey sources states) : Option (Fin N) :=
  findKeySlotGo N slots key

private def findVacantSlotGo {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (N : Nat) (slots : Fin N → Option (MemoEntry sources states results)) : Option (Fin N) :=
  match N with
  | 0 => none
  | Nat.succ n =>
      match slots (finZero n) with
      | none => some (finZero n)
      | some _ =>
          match findVacantSlotGo n (fun i => slots (finSucc i)) with
          | some i => some (finSucc i)
          | none => none
termination_by structural N

def findVacantSlot {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (slots : Fin N → Option (MemoEntry sources states results)) : Option (Fin N) :=
  findVacantSlotGo N slots

private theorem fin_zero_or_succ {n : Nat} (index : Fin (Nat.succ n)) :
    index = finZero n ∨ ∃ rest : Fin n, index = finSucc rest := by
  cases index with
  | mk value bound =>
      cases value with
      | zero => exact Or.inl rfl
      | succ value =>
          exact Or.inr ⟨⟨value, Nat.lt_of_succ_lt_succ bound⟩, rfl⟩

private theorem findKeySlotGo_some {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity} :
    ∀ (N : Nat) (slots : Fin N → Option (MemoEntry sources states results))
      (key : MemoKey sources states) (index : Fin N),
      findKeySlotGo N slots key = some index →
        ∃ entry, slots index = some entry ∧ entry.key = key := by
  intro N
  induction N with
  | zero =>
      intro slots key index found
      cases index with
      | mk value bound => exact False.elim (Nat.not_lt_zero value bound)
  | succ n ih =>
      intro slots key index found
      unfold findKeySlotGo at found
      cases first : slots (finZero n) with
      | none =>
          rw [first] at found
          cases rest : findKeySlotGo n (fun i => slots (finSucc i)) key with
          | none =>
              rw [rest] at found
              cases found
          | some next =>
              rw [rest] at found
              cases found
              exact ih (fun i => slots (finSucc i)) key next rest
      | some entry =>
          cases equal : MemoKey.equal entry.key key with
          | false =>
              rw [first] at found
              dsimp only at found
              rw [equal] at found
              cases rest : findKeySlotGo n (fun i => slots (finSucc i)) key with
              | none =>
                  rw [rest] at found
                  cases found
              | some next =>
                  rw [rest] at found
                  cases found
                  exact ih (fun i => slots (finSucc i)) key next rest
          | true =>
              rw [first] at found
              dsimp only at found
              rw [equal] at found
              cases found
              exact ⟨entry, first, (MemoKey.equal_iff entry.key key).mp equal⟩

theorem findKeySlot_some {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    {slots : Fin N → Option (MemoEntry sources states results)} {key : MemoKey sources states}
    {index : Fin N} (found : findKeySlot slots key = some index) :
    ∃ entry, slots index = some entry ∧ entry.key = key :=
  findKeySlotGo_some N slots key index found

private theorem findKeySlotGo_none_iff {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity} :
    ∀ (N : Nat) (slots : Fin N → Option (MemoEntry sources states results))
      (key : MemoKey sources states),
      findKeySlotGo N slots key = none ↔
        ∀ index entry, slots index = some entry → entry.key ≠ key := by
  intro N
  induction N with
  | zero =>
      intro slots key
      constructor
      · intro found index
        cases index with
        | mk value bound => exact False.elim (Nat.not_lt_zero value bound)
      · intro _
        rfl
  | succ n ih =>
      intro slots key
      cases first : slots (finZero n) with
      | none =>
          constructor
          · intro found index entry present
            unfold findKeySlotGo at found
            rw [first] at found
            dsimp only at found
            cases rest : findKeySlotGo n (fun i => slots (finSucc i)) key with
            | none =>
                have tail := (ih (fun i => slots (finSucc i)) key).mp rest
                cases fin_zero_or_succ index with
                | inl zero =>
                    rw [zero, first] at present
                    cases present
                | inr successor =>
                    obtain ⟨rest, equal⟩ := successor
                    rw [equal] at present
                    exact tail rest entry present
            | some rest =>
                rw [rest] at found
                cases found
          · intro all
            unfold findKeySlotGo
            rw [first]
            dsimp only
            have tail := (ih (fun i => slots (finSucc i)) key).mpr (by
              intro index entry present
              exact all (finSucc index) entry present)
            rw [tail]
      | some firstEntry =>
          cases equal : MemoKey.equal firstEntry.key key with
          | true =>
              constructor
              · intro found
                unfold findKeySlotGo at found
                rw [first] at found
                dsimp only at found
                rw [equal] at found
                cases found
              · intro all
                unfold findKeySlotGo
                rw [first]
                dsimp only
                rw [equal]
                have unequal := all (finZero n) firstEntry first
                exact False.elim (unequal ((MemoKey.equal_iff firstEntry.key key).mp equal))
          | false =>
              constructor
              · intro found index entry present
                unfold findKeySlotGo at found
                rw [first] at found
                dsimp only at found
                rw [equal] at found
                cases restFound : findKeySlotGo n (fun i => slots (finSucc i)) key with
                | none =>
                    have tail := (ih (fun i => slots (finSucc i)) key).mp restFound
                    cases fin_zero_or_succ index with
                    | inl zero =>
                        rw [zero] at present
                        have entryEqual : firstEntry = entry :=
                          Option.some.inj (first.symm.trans present)
                        cases entryEqual
                        intro keyEqual
                        have trueEqual : MemoKey.equal firstEntry.key key = true :=
                          (MemoKey.equal_iff firstEntry.key key).mpr keyEqual
                        exact Bool.false_ne_true (equal.symm.trans trueEqual)
                    | inr successor =>
                        obtain ⟨rest, indexEqual⟩ := successor
                        rw [indexEqual] at present
                        exact tail rest entry present
                | some next =>
                    rw [restFound] at found
                    cases found
              · intro all
                unfold findKeySlotGo
                rw [first]
                dsimp only
                rw [equal]
                have tail := (ih (fun i => slots (finSucc i)) key).mpr (by
                  intro index entry present
                  exact all (finSucc index) entry present)
                rw [tail]
                rfl

theorem findKeySlot_none_iff {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (slots : Fin N → Option (MemoEntry sources states results)) (key : MemoKey sources states) :
    findKeySlot slots key = none ↔ ∀ index entry, slots index = some entry → entry.key ≠ key :=
  findKeySlotGo_none_iff N slots key

private theorem findVacantSlotGo_some {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity} :
    ∀ (N : Nat) (slots : Fin N → Option (MemoEntry sources states results)) (index : Fin N),
      findVacantSlotGo N slots = some index → slots index = none := by
  intro N
  induction N with
  | zero =>
      intro slots index found
      cases index with
      | mk value bound => exact False.elim (Nat.not_lt_zero value bound)
  | succ n ih =>
      intro slots index found
      unfold findVacantSlotGo at found
      cases first : slots (finZero n) with
      | none =>
          rw [first] at found
          dsimp only at found
          cases found
          exact first
      | some entry =>
          rw [first] at found
          dsimp only at found
          cases rest : findVacantSlotGo n (fun i => slots (finSucc i)) with
          | none =>
              rw [rest] at found
              cases found
          | some next =>
              rw [rest] at found
              cases found
              exact ih (fun i => slots (finSucc i)) next rest

theorem findVacantSlot_some {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    {slots : Fin N → Option (MemoEntry sources states results)} {index : Fin N}
    (found : findVacantSlot slots = some index) : slots index = none :=
  findVacantSlotGo_some N slots index found

private theorem findVacantSlotGo_none_iff {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity} :
    ∀ (N : Nat) (slots : Fin N → Option (MemoEntry sources states results)),
      findVacantSlotGo N slots = none ↔ ∀ index, ∃ entry, slots index = some entry := by
  intro N
  induction N with
  | zero =>
      intro slots
      constructor
      · intro found index
        cases index with
        | mk value bound => exact False.elim (Nat.not_lt_zero value bound)
      · intro _
        rfl
  | succ n ih =>
      intro slots
      cases first : slots (finZero n) with
      | none =>
          constructor
          · intro found
            unfold findVacantSlotGo at found
            rw [first] at found
            dsimp only at found
            cases found
          · intro all
            obtain ⟨entry, present⟩ := all (finZero n)
            rw [first] at present
            cases present
      | some firstEntry =>
          constructor
          · intro found index
            unfold findVacantSlotGo at found
            rw [first] at found
            dsimp only at found
            cases rest : findVacantSlotGo n (fun i => slots (finSucc i)) with
            | none =>
                have tail := (ih (fun i => slots (finSucc i))).mp rest
                cases fin_zero_or_succ index with
                | inl zero => exact ⟨firstEntry, by rw [zero]; exact first⟩
                | inr successor =>
                    obtain ⟨restIndex, equal⟩ := successor
                    obtain ⟨entry, present⟩ := tail restIndex
                    exact ⟨entry, by rw [equal]; exact present⟩
            | some rest =>
                rw [rest] at found
                cases found
          · intro all
            unfold findVacantSlotGo
            rw [first]
            dsimp only
            have tail := (ih (fun i => slots (finSucc i))).mpr (by
              intro index
              exact all (finSucc index))
            rw [tail]

theorem findVacantSlot_none_iff {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (slots : Fin N → Option (MemoEntry sources states results)) :
    findVacantSlot slots = none ↔ ∀ index, ∃ entry, slots index = some entry :=
  findVacantSlotGo_none_iff N slots

def writeSlot {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (store : MemoStore sources states results N) (shard : Slang.Address.Shard) (slot : Fin N)
    (entry : MemoEntry sources states results) : MemoStore sources states results N :=
  { store with slots := fun queryShard =>
      if queryShard = shard then update (store.slots shard) slot (some entry) else store.slots queryShard }

theorem writeSlot_same {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (store : MemoStore sources states results N) (shard : Slang.Address.Shard) (slot : Fin N)
    (entry : MemoEntry sources states results) : (writeSlot store shard slot entry).slots shard slot = some entry := by
  dsimp only [writeSlot]
  rw [if_pos rfl]
  exact update_same _ _ _

theorem writeSlot_other_slot {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (store : MemoStore sources states results N) (shard : Slang.Address.Shard) (slot other : Fin N)
    (entry : MemoEntry sources states results) (different : other ≠ slot) :
    (writeSlot store shard slot entry).slots shard other = store.slots shard other := by
  dsimp only [writeSlot]
  rw [if_pos rfl]
  exact update_other _ _ _ _ different

theorem writeSlot_other_shard {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (store : MemoStore sources states results N) (shard other : Slang.Address.Shard) (slot : Fin N)
    (entry : MemoEntry sources states results) (different : other ≠ shard) :
    (writeSlot store shard slot entry).slots other = store.slots other := by
  dsimp only [writeSlot]
  rw [if_neg different]

def readEntry? {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (store : MemoStore sources states results N) (key : MemoKey sources states) :
    Option (MemoEntry sources states results) :=
  match findKeySlot (store.slots (Slang.Address.homeShard key)) key with
  | some slot => store.slots (Slang.Address.homeShard key) slot
  | none => none

def readValue {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (store : MemoStore sources states results N) (key : MemoKey sources states) : MemoValue sources states results :=
  match readEntry? store key with | some entry => entry.value | none => store.defaultValue

theorem readEntry_key {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    {store : MemoStore sources states results N} {key : MemoKey sources states}
    {entry : MemoEntry sources states results} (read : readEntry? store key = some entry) :
    entry.key = key := by
  unfold readEntry? at read
  cases found : findKeySlot (store.slots (Slang.Address.homeShard key)) key with
  | none =>
      rw [found] at read
      cases read
  | some slot =>
      rw [found] at read
      dsimp only at read
      obtain ⟨stored, present, storedKey⟩ := findKeySlot_some found
      have same : stored = entry := Option.some.inj (present.symm.trans read)
      cases same
      exact storedKey

theorem read_dial_consistent {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    {store : MemoStore sources states results N} {key : MemoKey sources states}
    {entry : MemoEntry sources states results} (read : readEntry? store key = some entry) :
    entry.key.sigma = key.sigma ∧ entry.key.face = key.face ∧ entry.parsed.ref = key.source := by
  have keyEqual := readEntry_key read
  cases keyEqual
  exact ⟨rfl, rfl, entry.parsedSource⟩

def denote {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (store : MemoStore sources states results N) : MemoKey sources states → MemoValue sources states results :=
  fun key => readValue store key

def WellFormed {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (store : MemoStore sources states results N) : Prop :=
  (∀ shard slot entry, store.slots shard slot = some entry → shard = Slang.Address.homeShard entry.key) ∧
  ∀ shard left right leftEntry rightEntry,
    store.slots shard left = some leftEntry → store.slots shard right = some rightEntry →
    leftEntry.key = rightEntry.key → left = right

def RepMemo {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (store : MemoStore sources states results N) (h : Apeiron.Table.HMap (MemoKey sources states) (MemoValue sources states results)) : Prop :=
  WellFormed store ∧ ∀ key, denote store key = Apeiron.Table.get h key

theorem read_denotes_get {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    {store : MemoStore sources states results N}
    {h : Apeiron.Table.HMap (MemoKey sources states) (MemoValue sources states results)}
    (rep : RepMemo store h) (key : MemoKey sources states) :
    readValue store key = Apeiron.Table.get h key :=
  rep.2 key

theorem empty_rep {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (defaultValue : MemoValue sources states results) :
    RepMemo (empty (N := N) defaultValue) (fun _ => defaultValue) := by
  constructor
  · constructor
    · intro shard slot entry present
      dsimp only [empty] at present
      cases present
    · intro shard left right leftEntry rightEntry leftStored rightStored keyEqual
      dsimp only [empty] at leftStored
      cases leftStored
  · intro key
    have absent : findKeySlot ((empty (N := N) defaultValue).slots (Slang.Address.homeShard key)) key = none :=
      (findKeySlot_none_iff _ _).mpr (by
        intro slot entry present
        dsimp only [empty] at present
        cases present)
    unfold denote readValue readEntry?
    rw [absent]
    rfl

inductive InsertResult {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (store : MemoStore sources states results N) where
  | inserted (after : MemoStore sources states results N) : InsertResult store
  | shardFull : InsertResult store

def insert {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (store : MemoStore sources states results N) (entry : MemoEntry sources states results) : InsertResult store :=
  let shard := Slang.Address.homeShard entry.key
  match findKeySlot (store.slots shard) entry.key with
  | some slot => .inserted (writeSlot store shard slot entry)
  | none => match findVacantSlot (store.slots shard) with
    | some slot => .inserted (writeSlot store shard slot entry)
    | none => .shardFull

def HomeFull {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (store : MemoStore sources states results N) (key : MemoKey sources states) : Prop :=
  ∀ slot, store.slots (Slang.Address.homeShard key) slot ≠ none

end Slang.Memo
