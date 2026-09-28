import Policy.Core

/-!
Headline laws for the canonical decision (`docs/epics/policy-engine.md`, "Laws"),
plus the five closed reachability witnesses. Each theorem is audited by the
lead-owned `AxiomAudit`; this module carries no `#print`/`#eval`/`#check` lines.
-/

namespace Apeiron.Policy

theorem le_of_eq_nat {a b : Nat} (h : a = b) : a ≤ b := by omega

/-! ## Fold and by-id helpers -/

theorem foldl_perm {α : Type} (g : α → Rule → α)
    (hswap : ∀ acc r r', g (g acc r) r' = g (g acc r') r) :
    ∀ {l l' : List Rule}, l.Perm l' → ∀ acc, l.foldl g acc = l'.foldl g acc := by
  intro l l' p
  induction p with
  | nil => intro acc; rfl
  | cons a p ih => intro acc; rw [List.foldl_cons, List.foldl_cons]; exact ih (g acc a)
  | swap x y l =>
      intro acc
      rw [List.foldl_cons, List.foldl_cons, List.foldl_cons, List.foldl_cons]
      exact congrArg (l.foldl g) (hswap acc x y).symm
  | trans p q ihp ihq => intro acc; rw [ihp acc, ihq acc]

theorem permStep_swap (q : Req) (acc : Nat) (r r' : Rule) :
    permStep q (permStep q acc r) r' = permStep q (permStep q acc r') r := by
  unfold permStep
  by_cases hr : Permits q r
  · by_cases hr' : Permits q r'
    · simp only [if_pos hr, if_pos hr', bor_assoc, bor_comm r.rights r'.rights]
    · simp only [if_pos hr, if_neg hr']
  · by_cases hr' : Permits q r'
    · simp only [if_neg hr, if_pos hr']
    · simp only [if_neg hr, if_neg hr']

theorem forbStep_swap (q : Req) (acc : Nat) (r r' : Rule) :
    forbStep q (forbStep q acc r) r' = forbStep q (forbStep q acc r') r := by
  unfold forbStep
  by_cases hr : Forbids q r
  · by_cases hr' : Forbids q r'
    · simp only [if_pos hr, if_pos hr', bor_assoc, bor_comm r.rights r'.rights]
    · simp only [if_pos hr, if_neg hr']
  · by_cases hr' : Forbids q r'
    · simp only [if_neg hr, if_pos hr']
    · simp only [if_neg hr, if_neg hr']

theorem askStep_swap (q : Req) (acc : Nat) (r r' : Rule) :
    askStep q (askStep q acc r) r' = askStep q (askStep q acc r') r := by
  unfold askStep
  by_cases hr : Asks q r
  · by_cases hr' : Asks q r'
    · simp only [if_pos hr, if_pos hr', bor_assoc, bor_comm r.rights r'.rights]
    · simp only [if_pos hr, if_neg hr']
  · by_cases hr' : Asks q r'
    · simp only [if_neg hr, if_pos hr']
    · simp only [if_neg hr, if_neg hr']

theorem permRights_perm {rs rs' : List Rule} (q : Req) (hp : rs.Perm rs') :
    permRights rs q = permRights rs' q :=
  foldl_perm (permStep q) (permStep_swap q) hp 0

theorem forbRights_perm {rs rs' : List Rule} (q : Req) (hp : rs.Perm rs') :
    forbRights rs q = forbRights rs' q :=
  foldl_perm (forbStep q) (forbStep_swap q) hp 0

theorem askRights_perm {rs rs' : List Rule} (q : Req) (hp : rs.Perm rs') :
    askRights rs q = askRights rs' q :=
  foldl_perm (askStep q) (askStep_swap q) hp 0

theorem minOpt_comm (acc : Option Nat) (a b : Nat) :
    minOpt (minOpt acc a) b = minOpt (minOpt acc b) a := by
  cases acc with
  | none => simp only [minOpt]; rw [Nat.min_comm a b]
  | some c => simp only [minOpt]; rw [Nat.min_assoc, Nat.min_assoc, Nat.min_comm a b]

theorem minStep_swap (c : Rule → Prop) [DecidablePred c] (acc : Option Nat) (r r' : Rule) :
    minStep c (minStep c acc r) r' = minStep c (minStep c acc r') r := by
  unfold minStep
  by_cases hr : c r
  · by_cases hr' : c r'
    · simp only [if_pos hr, if_pos hr']
      exact minOpt_comm acc r.id r'.id
    · simp only [if_pos hr, if_neg hr']
  · by_cases hr' : c r'
    · simp only [if_neg hr, if_pos hr']
    · simp only [if_neg hr, if_neg hr']

theorem permBy_perm {rs rs' : List Rule} (q : Req) (hp : rs.Perm rs') :
    permBy rs q = permBy rs' q :=
  foldl_perm (minStep (Permits q)) (minStep_swap (Permits q)) hp none

theorem askBy_perm {rs rs' : List Rule} (q : Req) (hp : rs.Perm rs') :
    askBy rs q = askBy rs' q :=
  foldl_perm (minStep (Asks q)) (minStep_swap (Asks q)) hp none

theorem forbBy_perm {rs rs' : List Rule} (q : Req) (P : Nat) (hp : rs.Perm rs') :
    forbBy rs q P = forbBy rs' q P :=
  foldl_perm (minStep (forbCond q P)) (minStep_swap (forbCond q P)) hp none

theorem decide_perm {rs rs' : List Rule} (q : Req) (hp : rs.Perm rs') :
    decide rs q = decide rs' q := by
  unfold decide gOf
  rw [permRights_perm q hp, forbRights_perm q hp, askRights_perm q hp,
    permBy_perm q hp, askBy_perm q hp, forbBy_perm q (permRights rs' q) hp]

/-! ## The four S3a laws -/

theorem decide_default_deny (rs : List Rule) (q : Req)
    (h : ∀ r ∈ rs, r.action = q.action → r.effect = Effect.permit →
      guardEval q.facts r.guard ≠ Tri.t ∨ band r.rights q.want = 0) :
    decide rs q = Verdict.deny Cause.noPermit := by
  have hP : permRights rs q = 0 := by
    suffices aux : ∀ (l : List Rule), (∀ r ∈ l, r.action = q.action → r.effect = Effect.permit →
        guardEval q.facts r.guard ≠ Tri.t ∨ band r.rights q.want = 0) →
        ∀ acc, l.foldl (permStep q) acc = acc from aux rs h 0
    intro l
    induction l with
    | nil => intro _ acc; rfl
    | cons r rest ih =>
      intro hl acc
      rw [List.foldl_cons]
      have hn : ¬ Permits q r := by
        rintro ⟨ha, he, hg, hw⟩
        rcases hl r (by simp) ha he with h1 | h1
        · exact h1 hg
        · exact hw h1
      have hstep : permStep q acc r = acc := by rw [permStep, if_neg hn]
      rw [hstep]
      exact ih (fun r' hr' => hl r' (List.mem_cons.mpr (Or.inr hr'))) acc
  unfold decide gOf
  rw [hP]
  simp [band_zero_right, band_zero_left]

theorem decide_forbid_wins (rs : List Rule) (q : Req) (r : Rule) (hmem : r ∈ rs)
    (hrel : r.action = q.action) (heff : r.effect = Effect.forbid)
    (hg : guardEval q.facts r.guard ≠ Tri.f) :
    band (granted (decide rs q)) r.rights = 0 := by
  have hsub : Sub r.rights (forbRights rs q) := forbRights_sub_of_mem rs q r hmem ⟨hrel, heff, hg⟩
  have hz : band r.rights (ncomp (forbRights rs q)) = 0 := sub_ncomp_zero hsub
  rw [granted_decide]
  unfold gOf
  rw [band_assoc, band_comm (ncomp (forbRights rs q)) r.rights, hz, band_zero_right]

theorem decide_granted_le_want (rs : List Rule) (q : Req) :
    band (granted (decide rs q)) (ncomp (mask q.want)) = 0 := by
  rw [granted_decide]
  exact sub_ncomp_zero (gOf_sub_mask_want rs q)

theorem decide_granted_mono_completion (rs : List Rule) (q q' : Req) (hc : Completes q q') :
    band (granted (decide rs q)) (ncomp (granted (decide rs q'))) = 0 := by
  rw [granted_decide, granted_decide]
  apply sub_ncomp_zero
  unfold gOf
  have hwant : mask q'.want = mask q.want := by rw [hc.2.1]
  rw [hwant]
  exact sub_and_mul (sub_and_left (permRights_sub_completes rs q q' hc) (mask q.want))
    (sub_ncomp (forbRights_sub_completes rs q q' hc))

/-! ## Addition of a forbid never widens -/

theorem forbStep_mono (q : Req) (r : Rule) {acc acc' : Nat} (h : Sub acc acc') :
    Sub (forbStep q acc r) (forbStep q acc' r) := by
  unfold forbStep
  by_cases hf : Forbids q r
  · rw [if_pos hf, if_pos hf]; exact sub_or h
  · rw [if_neg hf, if_neg hf]; exact h

theorem foldl_sub_mono (g : Nat → Rule → Nat)
    (hg : ∀ acc acc' r, Sub acc acc' → Sub (g acc r) (g acc' r)) :
    ∀ (l : List Rule) acc acc', Sub acc acc' → Sub (l.foldl g acc) (l.foldl g acc') := by
  intro l
  induction l with
  | nil => intro acc acc' h; exact h
  | cons r rest ih =>
    intro acc acc' h
    rw [List.foldl_cons, List.foldl_cons]
    exact ih (g acc r) (g acc' r) (hg acc acc' r h)

theorem decide_add_forbid_le (rs : List Rule) (q : Req) (r : Rule)
    (heff : r.effect = Effect.forbid) :
    band (granted (decide (r :: rs) q)) (ncomp (granted (decide rs q))) = 0 := by
  have hP : permRights (r :: rs) q = permRights rs q := by
    unfold permRights
    rw [List.foldl_cons]
    have h0 : permStep q 0 r = 0 := by
      unfold permStep
      rw [if_neg]
      intro hp
      exact absurd (hp.2.1.symm.trans heff) (by decide)
    rw [h0]
  have hF : Sub (forbRights rs q) (forbRights (r :: rs) q) := by
    unfold forbRights
    rw [List.foldl_cons]
    exact foldl_sub_mono (forbStep q) (fun acc acc' r' h => forbStep_mono q r' h)
      rs 0 (forbStep q 0 r) (sub_zero_left _)
  have hgoal : Sub (gOf (r :: rs) q) (gOf rs q) := by
    unfold gOf
    rw [hP]
    exact sub_and_left (sub_ncomp hF) (band (mask q.want) (permRights rs q))
  rw [granted_decide, granted_decide]
  exact sub_ncomp_zero hgoal

/-! ## The min-id fold returns a named rule -/

theorem minOpt_prov (acc : Option Nat) (id j : Nat) (h : minOpt acc id = some j) :
    (∃ b, acc = some b ∧ j = b) ∨ j = id := by
  cases acc with
  | none => exact Or.inr (by simpa [minOpt] using h.symm)
  | some b =>
    have hb : min b id = j := by simpa [minOpt] using h
    by_cases hle : b ≤ id
    · exact Or.inl ⟨b, rfl, by rw [← hb, Nat.min_eq_left hle]⟩
    · exact Or.inr (by rw [← hb, Nat.min_eq_right (Nat.le_of_not_le hle)])

theorem minOpt_le_right (acc : Option Nat) (id j : Nat) (h : minOpt acc id = some j) :
    j ≤ id := by
  cases acc with
  | none => exact le_of_eq_nat (by simpa [minOpt] using h.symm)
  | some b =>
    have hb : min b id = j := by simpa [minOpt] using h
    rw [← hb]; exact Nat.min_le_right b id

theorem minStep_some_le {c : Rule → Prop} [DecidablePred c] (acc : Option Nat) (r : Rule)
    (b j : Nat) (hacc : acc = some b) (h : minStep c acc r = some j) : j ≤ b := by
  unfold minStep at h
  by_cases hc : c r
  · rw [if_pos hc, hacc] at h
    have : min b r.id = j := by simpa [minOpt] using h
    rw [← this]; exact Nat.min_le_left b r.id
  · rw [if_neg hc, hacc] at h
    exact le_of_eq_nat (by simpa using h.symm)

theorem minStep_le_id (c : Rule → Prop) [DecidablePred c] (acc : Option Nat) (r : Rule)
    (hc : c r) : ∀ b, minStep c acc r = some b → b ≤ r.id := by
  intro b hb
  unfold minStep at hb
  rw [if_pos hc] at hb
  exact minOpt_le_right acc r.id b hb

theorem minStep_ne_none {c : Rule → Prop} [DecidablePred c] (acc : Option Nat) (r : Rule)
    (hc : c r) : minStep c acc r ≠ none := by
  unfold minStep
  rw [if_pos hc]
  intro hh
  cases acc <;> simp [minOpt] at hh

theorem minStep_ne_none_of_ne_none {c : Rule → Prop} [DecidablePred c] (acc : Option Nat)
    (r : Rule) (h : acc ≠ none) : minStep c acc r ≠ none := by
  unfold minStep
  by_cases hc : c r
  · rw [if_pos hc]; intro hh; cases acc <;> simp [minOpt] at hh
  · rw [if_neg hc]; exact h

theorem foldl_minStep_some (c : Rule → Prop) [DecidablePred c] :
    ∀ (rs : List Rule) (acc : Option Nat) (j : Nat),
      rs.foldl (minStep c) acc = some j →
        (∀ b, acc = some b → j ≤ b) ∧
        (∀ r ∈ rs, c r → j ≤ r.id) ∧
        ((∃ b, acc = some b ∧ j = b) ∨ ∃ r ∈ rs, c r ∧ r.id = j) := by
  intro rs
  induction rs with
  | nil =>
    intro acc j h
    exact ⟨fun b hb => by rw [hb] at h; exact le_of_eq_nat (Option.some.inj h).symm,
      (fun r hr _ => absurd hr (List.not_mem_nil)), Or.inl ⟨j, h, rfl⟩⟩
  | cons r rs ih =>
    intro acc j h
    rw [List.foldl_cons] at h
    rcases ih (minStep c acc r) j h with ⟨hleft, hmin, hprov⟩
    refine ⟨?_, ?_, ?_⟩
    · intro b hb
      have hex : ∃ b', minStep c acc r = some b' := by
        cases hstep : minStep c acc r with
        | none => exact absurd hstep (minStep_ne_none_of_ne_none acc r (by rw [hb]; simp))
        | some b' => exact ⟨b', rfl⟩
      rcases hex with ⟨b', hb'⟩
      exact Nat.le_trans (hleft b' hb') (minStep_some_le acc r b b' hb hb')
    · intro r' hr' hc'
      rcases List.mem_cons.mp hr' with heq | hr''
      · rw [heq] at hc' ⊢
        cases hstep : minStep c acc r with
        | none =>
          exfalso
          unfold minStep at hstep
          rw [if_pos hc'] at hstep
          cases acc <;> simp [minOpt] at hstep
        | some b' => exact Nat.le_trans (hleft b' hstep) (minStep_le_id c acc r hc' b' hstep)
      · exact hmin r' hr'' hc'
    · rcases hprov with ⟨b, hb, hjb⟩ | hw
      · by_cases hc : c r
        · unfold minStep at hb
          rw [if_pos hc] at hb
          rcases minOpt_prov acc r.id b hb with ⟨b0, hacc, hb0⟩ | hid
          · exact Or.inl ⟨b0, hacc, hjb.trans hb0⟩
          · exact Or.inr ⟨r, by simp, hc, by rw [hjb, hid]⟩
        · unfold minStep at hb
          rw [if_neg hc] at hb
          exact Or.inl ⟨b, hb, hjb⟩
      · rcases hw with ⟨r', hr', hc', hid⟩
        exact Or.inr ⟨r', List.mem_cons.mpr (Or.inr hr'), hc', hid⟩

theorem foldl_minStep_ne_none (c : Rule → Prop) [DecidablePred c] :
    ∀ (rs : List Rule) {acc : Option Nat}, acc ≠ none → rs.foldl (minStep c) acc ≠ none := by
  intro rs
  induction rs with
  | nil => intro acc h; simpa using h
  | cons r rs ih =>
    intro acc h
    rw [List.foldl_cons]
    exact ih (minStep_ne_none_of_ne_none acc r h)

theorem foldl_minStep_ne_none_of_exists (c : Rule → Prop) [DecidablePred c] :
    ∀ (rs : List Rule) (acc : Option Nat), (∃ r ∈ rs, c r) →
      rs.foldl (minStep c) acc ≠ none := by
  intro rs
  induction rs with
  | nil => intro acc h; rcases h with ⟨r, hr, _⟩; exact absurd hr (List.not_mem_nil)
  | cons r rs ih =>
    intro acc h
    rw [List.foldl_cons]
    rcases h with ⟨r', hr', hc'⟩
    rcases List.mem_cons.mp hr' with heq | hr''
    · rw [heq] at hc'
      exact foldl_minStep_ne_none c rs (minStep_ne_none acc r hc')
    · exact ih (minStep c acc r) ⟨r', hr'', hc'⟩

theorem forbBy_mem (rs : List Rule) (q : Req) (P : Nat) {j : Nat}
    (h : forbBy rs q P = some j) : ∃ r ∈ rs, r.id = j ∧ forbCond q P r := by
  have hh := foldl_minStep_some (forbCond q P) rs none j h
  rcases hh.2.2 with ⟨b, hb, _⟩ | ⟨r, hr, hc, hid⟩
  · cases hb
  · exact ⟨r, hr, hid, hc⟩

/-! ## Deny names an applicable forbid -/

theorem forbStep_testBit_lt (q : Req) (acc : Nat) (r : Rule) {i : Nat} (hi : i < W)
    (h : (forbStep q acc r).testBit i = true) :
    acc.testBit i = true ∨ (Forbids q r ∧ r.rights.testBit i = true) := by
  unfold forbStep at h
  by_cases hf : Forbids q r
  · rw [if_pos hf, testBit_bor_lt hi] at h
    rcases Bool.or_eq_true_iff.mp h with ha | hb
    · exact Or.inl ha
    · exact Or.inr ⟨hf, hb⟩
  · rw [if_neg hf] at h
    exact Or.inl h

theorem forbRights_testBit_lt (q : Req) (i : Nat) (hi : i < W) :
    ∀ (rs : List Rule) (acc : Nat), (rs.foldl (forbStep q) acc).testBit i = true →
      acc.testBit i = true ∨ ∃ r ∈ rs, Forbids q r ∧ r.rights.testBit i = true := by
  intro rs
  induction rs with
  | nil => intro acc h; exact Or.inl (by simpa using h)
  | cons r rs ih =>
    intro acc h
    rw [List.foldl_cons] at h
    rcases ih (forbStep q acc r) h with hacc | hw
    · rcases forbStep_testBit_lt q acc r hi hacc with ha | ⟨hf, hb⟩
      · exact Or.inl ha
      · exact Or.inr ⟨r, List.mem_cons.mpr (Or.inl rfl), hf, hb⟩
    · rcases hw with ⟨r', hr', hf', hb'⟩
      exact Or.inr ⟨r', List.mem_cons.mpr (Or.inr hr'), hf', hb'⟩

theorem forbBy_ne_none (rs : List Rule) (q : Req) (P : Nat)
    (hX : band (mask q.want) P ≠ 0)
    (hg : band (band (mask q.want) P) (ncomp (forbRights rs q)) = 0) :
    forbBy rs q P ≠ none := by
  have hsub : Sub (band (mask q.want) P) (forbRights rs q) := sub_of_band_ncomp_zero hg
  obtain ⟨i, hi⟩ := exists_testBit_of_ne_zero hX
  have hiW : i < W := by
    by_cases hge : i < W
    · exact hge
    · exfalso
      rw [testBit_band_ge (Nat.le_of_not_lt hge)] at hi
      exact absurd hi (by simp)
  have hXi : (band (band (mask q.want) P) (forbRights rs q)).testBit i = true := by
    rw [hsub, mask_of_lt (band_lt (mask q.want) P)]; exact hi
  have hFi : (forbRights rs q).testBit i = true := testBit_band_right hXi
  have hXi' : (band (mask q.want) P).testBit i = true := testBit_band_left hXi
  have hwi : (mask q.want).testBit i = true := testBit_band_left hXi'
  have hPi : P.testBit i = true := testBit_band_right hXi'
  have hwi' : q.want.testBit i = true := by
    rw [testBit_mask_lt hiW] at hwi; exact hwi
  rcases forbRights_testBit_lt q i hiW rs 0 (by simpa only [forbRights] using hFi) with
    h0 | ⟨r, hr, hf, hri⟩
  · exact absurd h0 (by simp)
  · refine foldl_minStep_ne_none_of_exists (forbCond q P) rs none ⟨r, hr, ?_⟩
    refine ⟨hf.1, hf.2.1, ?_, hf.2.2⟩
    intro hzero
    have hbit : (band (band r.rights q.want) P).testBit i = true := by
      rw [testBit_band_lt hiW, testBit_band_lt hiW, hri, hwi', hPi]
      simp
    rw [hzero, Nat.zero_testBit] at hbit
    exact absurd hbit (by simp)

theorem decide_deny_named (rs : List Rule) (q : Req) (i : Nat)
    (h : decide rs q = Verdict.deny (Cause.forbid i)) :
    ∃ r ∈ rs, r.id = i ∧ r.action = q.action ∧ r.effect = Effect.forbid ∧
      guardEval q.facts r.guard ≠ Tri.f := by
  unfold decide at h
  by_cases h0 : gOf rs q = 0
  · rw [if_pos h0] at h
    by_cases h1 : band (mask q.want) (permRights rs q) = 0
    · rw [if_pos h1] at h; cases h
    · rw [if_neg h1] at h
      have hne : forbBy rs q (permRights rs q) ≠ none :=
        forbBy_ne_none rs q (permRights rs q) h1 h0
      cases hj : forbBy rs q (permRights rs q) with
      | none => exact absurd hj hne
      | some j =>
        have hji : j = i := by
          rw [hj] at h
          simp only [Option.getD_some] at h
          exact Cause.forbid.inj (Verdict.deny.inj h)
        rcases forbBy_mem rs q (permRights rs q) hj with ⟨r, hr, hrid, hcond⟩
        exact ⟨r, hr, by rw [hrid, hji], hcond.1, hcond.2.1, hcond.2.2.2⟩
  · rw [if_neg h0] at h
    by_cases h2 : band (gOf rs q) (askRights rs q) ≠ 0
    · rw [if_pos h2] at h; cases h
    · rw [if_neg h2] at h
      by_cases h3 : gOf rs q = mask q.want
      · rw [if_pos h3] at h; cases h
      · rw [if_neg h3] at h; cases h

/-! ## The two consumer-vocabulary laws -/

theorem oneBit_no_attenuate (rs : List Rule) (q : Req) (h : mask q.want ≤ 1) :
    ∀ g b, decide rs q ≠ Verdict.attenuate g b := by
  have hband : band (gOf rs q) (mask q.want) = gOf rs q :=
    sub_apply (gOf_sub_mask_want rs q) (gOf_lt rs q)
  intro g b
  unfold decide
  by_cases h0 : gOf rs q = 0
  · rw [if_pos h0]
    by_cases h1 : band (mask q.want) (permRights rs q) = 0
    · rw [if_pos h1]; intro hh; cases hh
    · rw [if_neg h1]; intro hh; cases hh
  · rw [if_neg h0]
    by_cases h2 : band (gOf rs q) (askRights rs q) ≠ 0
    · rw [if_pos h2]; intro hh; cases hh
    · rw [if_neg h2]
      have hg : gOf rs q = mask q.want := by
        rcases (by omega : mask q.want = 0 ∨ mask q.want = 1) with hz | ho
        · rw [hz, band_zero_right] at hband
          exact absurd hband.symm h0
        · rw [ho] at hband
          have hmod : band (gOf rs q) 1 = gOf rs q % 2 := band_one _
          rw [hmod] at hband
          have hlt : gOf rs q % 2 < 2 := Nat.mod_lt _ (by omega)
          omega
      rw [if_pos hg]; intro hh; cases hh

theorem noAsk_no_ask (rs : List Rule) (q : Req) (h : ∀ r ∈ rs, r.effect ≠ Effect.ask) :
    ∀ g p b, decide rs q ≠ Verdict.ask g p b := by
  have hA : askRights rs q = 0 := by
    suffices aux : ∀ (l : List Rule), (∀ r ∈ l, r.effect ≠ Effect.ask) →
        ∀ acc, l.foldl (askStep q) acc = acc from aux rs h 0
    intro l
    induction l with
    | nil => intro _ acc; rfl
    | cons r rest ih =>
      intro hl acc
      rw [List.foldl_cons]
      have hn : ¬ Asks q r := by
        rintro ⟨_, he, _⟩
        exact hl r (by simp) he
      have hstep : askStep q acc r = acc := by rw [askStep, if_neg hn]
      rw [hstep]
      exact ih (fun r' hr' => hl r' (List.mem_cons.mpr (Or.inr hr'))) acc
  intro g p b
  unfold decide
  rw [hA, band_zero_right]
  have hne : ¬ (0 ≠ 0) := fun hh => hh rfl
  rw [if_neg hne]
  by_cases h0 : gOf rs q = 0
  · rw [if_pos h0]
    by_cases h1 : band (mask q.want) (permRights rs q) = 0
    · rw [if_pos h1]; intro hh; cases hh
    · rw [if_neg h1]; intro hh; cases hh
  · rw [if_neg h0]
    by_cases h3 : gOf rs q = mask q.want
    · rw [if_pos h3]; intro hh; cases hh
    · rw [if_neg h3]; intro hh; cases hh

/-! ## Closed reachability witnesses (≤ 3 rules, `rfl`) -/

theorem witness_allow :
    decide [⟨3, Effect.permit, 7, 1, some []⟩] ⟨fun _ => none, 7, 1⟩
      = Verdict.allow 1 3 := rfl

theorem witness_attenuate :
    decide [⟨3, Effect.permit, 7, 3, some []⟩, ⟨9, Effect.forbid, 7, 2, some []⟩]
      ⟨fun _ => none, 7, 3⟩ = Verdict.attenuate 1 3 := rfl

theorem witness_ask :
    decide [⟨3, Effect.permit, 7, 1, some []⟩, ⟨5, Effect.ask, 7, 1, some []⟩]
      ⟨fun _ => none, 7, 1⟩ = Verdict.ask 1 1 5 := rfl

theorem witness_deny_noPermit :
    decide [] ⟨fun _ => none, 7, 1⟩ = Verdict.deny Cause.noPermit := rfl

theorem witness_deny_forbid :
    decide [⟨3, Effect.permit, 7, 1, some []⟩, ⟨9, Effect.forbid, 7, 1, some []⟩]
      ⟨fun _ => none, 7, 1⟩ = Verdict.deny (Cause.forbid 9) := rfl

end Apeiron.Policy
