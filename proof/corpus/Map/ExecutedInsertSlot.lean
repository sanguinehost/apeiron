-- The executed insert's three slot verdicts.
import Map.ExecutedOverflowInsert
open Aeneas Aeneas.Std Result

/-!
# The executed insert: one verdict of the capped probe at a time

`insert_core` (`src/hashtrinity/mod.rs`) runs the capped probe on a non-reserved key `k` and acts on
its verdict.  Each lemma below takes a represented table and the verdict's contract (`CappedOK`) and
shows that the verdict's write represents the last-write update `upd k v m` on `live ∪ {k}`:

* `rep_hit` — `Hit i`: the value at slot `i` is overwritten in place;
* `rep_fill` — `Vacant i`: slot `i` (an `EMPTY` slot) takes the coordinate and the value, `len` rises
  by one;
* `rep_set_ov` — `Blocked`: the overflow store is replaced by a store `o'` of shape `T'` (`OvFull`).
  `blocked_new_full` (absent key: `insert_new_at`) and `blocked_hit_full` (present key: `take_value`
  then `write_back`) produce that store.

`CoreRep`'s overflow clause (`OvInv`) carries no AVL shape and no `alive` flag; the insert needs both
(`insert_new_at_spec` takes `OInv`; `into_entries` skips non-`alive` nodes on a resize).  `OvWF` is
that extra clause: `OInv` at the tree shape, with every tree node `alive`.  `RepW` is
`CoreRep ∧ OvWF`.  `arenaLen` is the arena length; `insert_new_at` needs it below `NIL` (a pushed
node's index is `(nodes.len() - 1) as u32`, which reaches `NIL` at `2^32 - 1` nodes).
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-- The last-write update of the canonical map: `insertWith (fun new _ => new)`. -/
abbrev upd (k : U64) (v : I64) (m : Apeiron.Table.HMap U64 I64) : Apeiron.Table.HMap U64 I64 :=
  Apeiron.Table.insertWith (fun new _ => new) k v m

theorem upd_self (k : U64) (v : I64) (m : Apeiron.Table.HMap U64 I64) : upd k v m k = v := by
  simp [upd, Apeiron.Table.insertWith]

theorem upd_ne {k c : U64} (v : I64) (m : Apeiron.Table.HMap U64 I64) (h : c ≠ k) :
    upd k v m c = m c := by
  simp [upd, Apeiron.Table.insertWith, h]

theorem nd_nodup : ∀ t : OTree, ND t → (idxs t).Nodup
  | .leaf, _ => List.nodup_nil
  | .node l i r, ⟨hl, hr, hil, hir, hlr⟩ => by
    have h1 := nd_nodup l hl
    have h2 := nd_nodup r hr
    simp only [idxs]
    rw [List.nodup_append]
    refine ⟨h1, List.nodup_cons.mpr ⟨hir, h2⟩, ?_⟩
    intro a ha b hb hab
    subst hab
    rcases List.mem_cons.mp hb with e | e
    · exact hil (e ▸ ha)
    · exact hlr a ha e

section
variable {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)

/-- The overflow tree is an AVL search tree over its arena (`OInv`) whose nodes are all `alive`. -/
def OvWF (s : Tbl KS) : Prop :=
  match s.overflow with
  | none => True
  | some o => ∃ T, OInv o T ∧ ∀ i ∈ idxs T, ∃ nd, o.nodes.val[i]? = some nd ∧ nd.alive = true

/-- The overflow arena's length (`0` without an overflow store). -/
def arenaLen (s : Tbl KS) : Nat :=
  match s.overflow with
  | none => 0
  | some o => o.nodes.val.length

/-- `CoreRep` with the overflow tree's shape clause. -/
def RepW (s : Tbl KS) (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) : Prop :=
  CoreRep ks s m live ∧ OvWF s

/-! ## Frames -/

theorem placed_frame {s s' : Tbl KS} {c : U64} {i : Nat}
    (hlen : s'.coords.length = s.coords.length) (hbits : s'.bits = s.bits)
    (hkeep : ∀ (d : Nat) (x : U64), s.coords.val[d]? = some x → x.val ≠ EMPTYN →
      s'.coords.val[d]? = some x)
    (h : Placed ks s c i) : Placed ks s' c i := by
  obtain ⟨j, hj, d, hd, hdi, hall⟩ := h
  refine ⟨j, by rw [hbits]; exact hj, d, desc_frame ks s s' c j d hlen hbits hd, hdi,
    fun j' hj' => ?_⟩
  obtain ⟨d', hd', x, hx, hxe⟩ := hall j' hj'
  exact ⟨d', desc_frame ks s s' c j' d' hlen hbits hd', x, hkeep _ x hx hxe, hxe⟩

theorem ovInv_transfer {s s' : Tbl KS} {m m' : Apeiron.Table.HMap U64 I64} (hov : OvInv ks s m)
    (ho : s'.overflow = s.overflow) (hlen : s'.coords.length = s.coords.length)
    (hbits : s'.bits = s.bits)
    (hkeep : ∀ (d : Nat) (x : U64), s.coords.val[d]? = some x → x.val ≠ EMPTYN →
      s'.coords.val[d]? = some x)
    (hm : ∀ c, InOv s c → m' c = m c) : OvInv ks s' m' := by
  unfold OvInv at hov ⊢
  rw [ho]
  revert hov
  cases hso : s.overflow with
  | none => intro _; trivial
  | some o =>
    rintro ⟨T, hT, hB, hN, hF, hall⟩
    refine ⟨T, hT, hB, hN, hF, fun i hi => ?_⟩
    obtain ⟨nd, hnd, hent, hns, hP⟩ := hall i hi
    refine ⟨nd, hnd, by rw [hent, hm _ ⟨o, hso, T, hT, i, hi, nd, hnd, rfl⟩], hns, fun j hj => ?_⟩
    exact pass_frame ks s s' _ j hlen hbits
      (fun d x hx hxe hxc => ⟨x, hkeep d x hx hxe, hxc, hxe⟩) (hP j (by rw [← hbits]; exact hj))

theorem inOv_congr {s s' : Tbl KS} (ho : s'.overflow = s.overflow) (c : U64) :
    InOv s' c ↔ InOv s c := by
  unfold InOv; rw [ho]

/-- A main coordinate is not an overflow coordinate. -/
theorem not_inOv_of_main {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) {k : U64} (hns : ¬ IsSent k) {i : Nat}
    (hi : s.coords.val[i]? = some k) : ¬ InOv s k := by
  intro hov
  obtain ⟨_, hpass⟩ := ov_pass ks hR hov
  obtain ⟨j, hj, d, hd, hdi, _⟩ := hR.main_placed i k hi hns
  obtain ⟨d', hd', x, hx, hxk, _⟩ := hpass j hj
  have := desc_det ks hd' hd
  subst this
  rw [hdi, hi] at hx
  exact hxk (Option.some.inj hx).symm

/-! ## `Hit`: overwrite in place -/

theorem rep_hit {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) {k : U64} (hns : ¬ IsSent k) {i : Nat}
    (hi : s.coords.val[i]? = some k) (v : I64)
    (vals' : Slice (core.mem.maybe_uninit.MaybeUninit I64))
    (hv : vals'.val = s.vals.val.set i (some v)) :
    CoreRep ks { s with vals := vals' } (upd k v m) (fun c => live c ∨ c = k) := by
  have hlk : live k := (hR.main_val i k hi hns).1
  have hnov := not_inOv_of_main ks hR hns hi
  have hilt : i < s.vals.val.length := by
    have := (List.getElem?_eq_some_iff.mp hi).1
    have h2 : s.vals.val.length = s.coords.val.length := hR.vals_len
    omega
  refine
    { bits_pos := hR.bits_pos, len_pow := hR.len_pow, vals_len := ?_,
      main_val := fun i' c hc hcns => ?_, main_uniq := hR.main_uniq,
      main_placed := fun i' c hc hcns =>
        placed_frame ks rfl rfl (fun _ _ h _ => h) (hR.main_placed i' c hc hcns),
      len_count := hR.len_count, side_len_vals := hR.side_len_vals,
      side_val := fun i' c hc => ?_, side_uniq := hR.side_uniq,
      ov := ovInv_transfer ks hR.ov rfl rfl rfl (fun _ _ h _ => h)
        (fun c hc => upd_ne v m (fun e => hnov (e ▸ hc))),
      live_iff := fun c => ?_ }
  · show vals'.val.length = s.coords.val.length
    rw [hv, List.length_set]; exact hR.vals_len
  · have hc' : s.coords.val[i']? = some c := hc
    by_cases hii : i' = i
    · subst hii
      rw [hi] at hc'
      have := Option.some.inj hc'
      subst this
      refine ⟨Or.inr rfl, ?_⟩
      show vals'.val[i']? = _
      rw [hv, List.getElem?_set_self hilt, upd_self]
    · have hck : c ≠ k := fun e => hii (hR.main_uniq i' i c hc' (e ▸ hi) hcns)
      obtain ⟨hl, hval⟩ := hR.main_val i' c hc' hcns
      refine ⟨Or.inl hl, ?_⟩
      show vals'.val[i']? = _
      rw [hv, List.getElem?_set_ne (Ne.symm hii), hval, upd_ne v m hck]
  · obtain ⟨a, b, d⟩ := hR.side_val i' c hc
    exact ⟨a, Or.inl b, by rw [d, upd_ne v m (fun (e : c = k) => hns (e ▸ a))]⟩
  · by_cases hck : c = k
    · subst hck
      exact ⟨fun _ => Or.inr (Or.inl ⟨hns, i, hi⟩), fun _ => Or.inl hlk⟩
    · have := hR.live_iff c
      simp only [hck, or_false]
      exact this

/-! ## `Vacant`: fill an `EMPTY` slot -/

theorem rep_fill {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) {k : U64} (hns : ¬ IsSent k) (i : Usize)
    (hc : CappedOK ks s k (.Vacant i)) (v : I64)
    (coords' : Slice U64) (vals' : Slice (core.mem.maybe_uninit.MaybeUninit I64)) (keys' : KS)
    (len' : Usize)
    (hco : coords'.val = s.coords.val.set i.val k) (hv : vals'.val = s.vals.val.set i.val (some v))
    (hl : len'.val = s.len.val + 1) :
    CoreRep ks { s with coords := coords', vals := vals', keys := keys', len := len' } (upd k v m)
      (fun c => live c ∨ c = k) := by
  have hnl : ¬ live k := vacant_not_live ks hR hns i hc
  obtain ⟨⟨x0, hx0, hx0e⟩, j, hj, hd, hpre⟩ := hc
  have hilt : i.val < s.coords.val.length := (List.getElem?_eq_some_iff.mp hx0).1
  have hvl : s.vals.val.length = s.coords.val.length := hR.vals_len
  have hkeep : ∀ (d : Nat) (x : U64), s.coords.val[d]? = some x → x.val ≠ EMPTYN →
      coords'.val[d]? = some x := by
    intro d x hx hxe
    have hdi : d ≠ i.val := by
      rintro rfl
      rw [hx0] at hx
      exact hxe ((Option.some.inj hx) ▸ hx0e)
    rw [hco, List.getElem?_set_ne (Ne.symm hdi)]; exact hx
  have hnew : coords'.val[i.val]? = some k := by rw [hco, List.getElem?_set_self hilt]
  have hback : ∀ (d : Nat) (c : U64), coords'.val[d]? = some c →
      (d = i.val ∧ c = k) ∨ (d ≠ i.val ∧ s.coords.val[d]? = some c) := by
    intro d c hd
    by_cases hdi : d = i.val
    · subst hdi; rw [hnew] at hd; exact Or.inl ⟨rfl, (Option.some.inj hd).symm⟩
    · rw [hco, List.getElem?_set_ne (Ne.symm hdi)] at hd; exact Or.inr ⟨hdi, hd⟩
  have hmainK : ∀ d, s.coords.val[d]? = some k → False :=
    fun d hd => hnl (hR.main_val d k hd hns).1
  have hlen' : coords'.val.length = s.coords.val.length := by rw [hco, List.length_set]
  have hovK : ∀ c, InOv s c → c ≠ k := fun c hc e =>
    hnl (e ▸ (hR.live_iff c).mpr (Or.inr (Or.inr hc)))
  refine
    { bits_pos := hR.bits_pos, len_pow := ?_, vals_len := ?_,
      main_val := fun d c hc hcns => ?_, main_uniq := fun d1 d2 c h1 h2 hcns => ?_,
      main_placed := fun d c hc hcns => ?_,
      len_count := ?_, side_len_vals := hR.side_len_vals,
      side_val := fun i' c hc => ?_, side_uniq := hR.side_uniq,
      ov := ovInv_transfer ks hR.ov rfl hlen' rfl hkeep (fun c hc => upd_ne v m (hovK c hc)),
      live_iff := fun c => ?_ }
  · show coords'.val.length = 2 ^ s.bits.val
    rw [hlen']; exact hR.len_pow
  · show vals'.val.length = coords'.val.length
    rw [hv, List.length_set, hlen']; exact hvl
  · rcases hback d c hc with ⟨rfl, hck⟩ | ⟨hne, hd0⟩
    · subst c
      refine ⟨Or.inr rfl, ?_⟩
      show vals'.val[i.val]? = _
      rw [hv, List.getElem?_set_self (by omega), upd_self]
    · have hck : c ≠ k := fun e => hmainK d (e ▸ hd0)
      obtain ⟨hlv, hval⟩ := hR.main_val d c hd0 hcns
      refine ⟨Or.inl hlv, ?_⟩
      show vals'.val[d]? = _
      rw [hv, List.getElem?_set_ne (Ne.symm hne), hval, upd_ne v m hck]
  · rcases hback d1 c h1 with ⟨e1, rfl⟩ | ⟨hne1, h1'⟩ <;>
      rcases hback d2 c h2 with ⟨e2, hc2⟩ | ⟨hne2, h2'⟩
    · omega
    · exact absurd h2' (fun h => hmainK d2 h)
    · exact absurd h1' (fun h => hmainK d1 (hc2 ▸ h))
    · exact hR.main_uniq d1 d2 c h1' h2' hcns
  · rcases hback d c hc with ⟨rfl, hck⟩ | ⟨_, hd0⟩
    · subst c
      have hlenS : ({ s with coords := coords', vals := vals', keys := keys', len := len' } :
          Tbl KS).coords.length = s.coords.length := hlen'
      refine ⟨j, hj, i, desc_frame ks s _ k j i hlenS rfl hd, rfl, fun j' hj' => ?_⟩
      obtain ⟨d', hd', y, hy, _, hye⟩ := hpre j' hj'
      exact ⟨d', desc_frame ks s _ k j' d' hlenS rfl hd', y, hkeep _ y hy hye, hye⟩
    · exact placed_frame ks hlen' rfl hkeep (hR.main_placed d c hd0 hcns)
  · show len'.val = coords'.val.countP (fun c => decide (¬ IsSent c))
    have hx0' : s.coords.val[i.val] = x0 := (List.getElem?_eq_some_iff.mp hx0).2
    have hsx : IsSent x0 := Or.inl hx0e
    rw [hl, hco, List.countP_set hilt, hx0', hR.len_count]
    simp [hsx, hns, mainCount]
  · obtain ⟨a, b, d⟩ := hR.side_val i' c hc
    exact ⟨a, Or.inl b, by rw [d, upd_ne v m (fun (e : c = k) => hns (e ▸ a))]⟩
  · have hmain' : InMain ({ s with coords := coords', vals := vals', keys := keys', len := len' } :
        Tbl KS) c ↔ InMain s c ∨ c = k := by
      constructor
      · rintro ⟨hcns, d, hd⟩
        rcases hback d c hd with ⟨_, rfl⟩ | ⟨_, hd0⟩
        · exact Or.inr rfl
        · exact Or.inl ⟨hcns, d, hd0⟩
      · rintro (⟨hcns, d, hd⟩ | rfl)
        · exact ⟨hcns, d, hkeep d c hd (sent_ne_empty_tomb c hcns).1⟩
        · exact ⟨hns, i.val, hnew⟩
    have hside' : InSide ({ s with coords := coords', vals := vals', keys := keys', len := len' } :
        Tbl KS) c ↔ InSide s c := Iff.rfl
    have hov' := inOv_congr (s := s)
      (s' := { s with coords := coords', vals := vals', keys := keys', len := len' }) rfl c
    rw [hmain', hside', hov', hR.live_iff c]
    constructor
    · rintro ((h | h | h) | rfl)
      · exact Or.inl h
      · exact Or.inr (Or.inl (Or.inl h))
      · exact Or.inr (Or.inr h)
      · exact Or.inr (Or.inl (Or.inr rfl))
    · rintro (h | (h | rfl) | h)
      · exact Or.inl (Or.inl h)
      · exact Or.inl (Or.inr (Or.inl h))
      · exact Or.inr rfl
      · exact Or.inl (Or.inr (Or.inr h))

/-! ## `Blocked`: replace the overflow store -/

/-- The overflow store `o` of shape `T` holds `m` at its nodes' coordinates: `OInv`, every node
`alive`, a single entry `(key, some (m key))` for a non-reserved `key` whose capped prefix in `s`
holds neither `key` nor `EMPTY`. -/
def OvFull (s : Tbl KS) (m : Apeiron.Table.HMap U64 I64) (o : Ov) (T : OTree) : Prop :=
  OInv o T ∧ ∀ i ∈ idxs T, ∃ nd, o.nodes.val[i]? = some nd ∧ nd.alive = true ∧
    nd.entries.val = [(nd.key, some (m nd.key))] ∧ ¬ IsSent nd.key ∧
    ∀ j : U32, j.val < limN s.bits.val → Pass ks s nd.key j

/-- The coordinates of a tree's nodes. -/
def InT (o : Ov) (T : OTree) (c : U64) : Prop :=
  ∃ i ∈ idxs T, ∃ nd, o.nodes.val[i]? = some nd ∧ nd.key = c

theorem rep_set_ov {s : Tbl KS} {m m' : Apeiron.Table.HMap U64 I64} {live live' : U64 → Prop}
    (hR : CoreRep ks s m live) (o' : Ov) (T' : OTree) (hF : OvFull ks s m' o' T')
    (hm : ∀ c, InMain s c → m' c = m c) (hms : ∀ c, IsSent c → m' c = m c)
    (hlive : ∀ c, live' c ↔ InSide s c ∨ InMain s c ∨ InT o' T' c) :
    RepW ks { s with overflow := some o' } m' live' := by
  obtain ⟨⟨hE, hB, hND, _, _, _, hfl⟩, hall⟩ := hF
  have hmain : ∀ c, InMain s c → live' c := fun c h => (hlive c).mpr (Or.inr (Or.inl h))
  refine ⟨
    { bits_pos := hR.bits_pos, len_pow := hR.len_pow, vals_len := hR.vals_len,
      main_val := fun i c hc hcns => ?_, main_uniq := hR.main_uniq,
      main_placed := fun i c hc hcns =>
        placed_frame ks rfl rfl (fun _ _ h _ => h) (hR.main_placed i c hc hcns),
      len_count := hR.len_count, side_len_vals := hR.side_len_vals,
      side_val := fun i c hc => ?_, side_uniq := hR.side_uniq,
      ov := ?_, live_iff := fun c => ?_ }, ?_⟩
  · obtain ⟨_, hval⟩ := hR.main_val i c hc hcns
    exact ⟨hmain c ⟨hcns, i, hc⟩, by rw [hval, hm c ⟨hcns, i, hc⟩]⟩
  · obtain ⟨a, _, d⟩ := hR.side_val i c hc
    exact ⟨a, (hlive c).mpr (Or.inl ⟨i, hc⟩), by rw [d, hms c a]⟩
  · show ∃ T, Emb o'.nodes.val o'.root T ∧ _
    refine ⟨T', hE, hB, nd_nodup T' hND, fun f hf => (hfl f hf).2, fun i hi => ?_⟩
    obtain ⟨nd, hnd, _, hent, hns, hP⟩ := hall i hi
    refine ⟨nd, hnd, hent, hns, fun j hj => ?_⟩
    exact pass_frame ks s _ _ j rfl rfl (fun d x hx hxe hxc => ⟨x, hx, hxc, hxe⟩) (hP j hj)
  · rw [hlive c]
    have hov : InOv ({ s with overflow := some o' } : Tbl KS) c ↔ InT o' T' c := by
      constructor
      · rintro ⟨o, ho, T, hT, h⟩
        have hoo : o = o' := (Option.some.inj ho).symm
        subst hoo
        have := emb_unique hT hE
        subst this
        exact h
      · intro h; exact ⟨o', rfl, T', hE, h⟩
    rw [hov]
    exact Iff.rfl
  · show ∃ T, OInv o' T ∧ _
    exact ⟨T', ⟨hE, hB, hND, ‹_›, ‹_›, ‹_›, hfl⟩, fun i hi => by
      obtain ⟨nd, hnd, ha, _⟩ := hall i hi
      exact ⟨nd, hnd, ha⟩⟩

/-- The represented overflow store is `OvFull` at its tree, whose coordinates are `InOv`. -/
theorem ov_full_of_some {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) {o : Ov} (ho : s.overflow = some o) :
    ∃ T, OvFull ks s m o T ∧ ∀ c, InT o T c ↔ InOv s c := by
  obtain ⟨T, hE, _, _, _, hall⟩ := ovInv_some ks hR.1 ho
  have hW := hR.2
  unfold OvWF at hW
  rw [ho] at hW
  obtain ⟨T2, hI, hal⟩ := hW
  have := emb_unique hE hI.1
  subst this
  refine ⟨T, ⟨hI, fun i hi => ?_⟩, fun c => ⟨fun h => ⟨o, ho, T, hE, h⟩, ?_⟩⟩
  · obtain ⟨nd, hnd, hent, hns, hP⟩ := hall i hi
    obtain ⟨nd', hnd', ha⟩ := hal i hi
    rw [hnd] at hnd'
    have := Option.some.inj hnd'
    subst this
    exact ⟨nd, hnd, ha, hent, hns, hP⟩
  · rintro ⟨o', ho', T', hT', h⟩
    have hoo : o' = o := Option.some.inj (ho'.symm.trans ho)
    subst hoo
    have := emb_unique hT' hE
    subst this
    exact h

/-- The fresh overflow store (`Overflow::new`) is `OvFull` at the empty tree. -/
theorem ov_full_new (s : Tbl KS) (m : Apeiron.Table.HMap U64 I64) (o : Ov)
    (h : hashtrinity.overflow.Overflow.new I64 CI = ok o) :
    OvFull ks s m o .leaf ∧ (∀ c, ¬ InT o .leaf c) ∧ o.nodes.val.length = 0 := by
  unfold hashtrinity.overflow.Overflow.new at h
  have := (Result.ok_injective h).symm
  subst this
  refine ⟨⟨?_, fun i hi => by simp [idxs] at hi⟩, ?_, by simp [alloc.vec.Vec.new]⟩
  · exact ⟨NIL_val, trivial, trivial, trivial, trivial, by simp [alloc.vec.Vec.new],
      fun j hj => by simp [alloc.vec.Vec.new] at hj⟩
  · rintro c ⟨i, hi, _⟩
    simp [idxs] at hi

/-- `Blocked` on an absent key: `insert_new_at` adds a fresh node `(k, some v)`. -/
theorem blocked_new_full_abs {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} (ov : Ov) (T : OTree)
    (hF : OvFull ks s m ov T) {k : U64} (hns : ¬ IsSent k) (hblk : CappedOK ks s k .Blocked)
    (hcap : ov.nodes.val.length < nilN) (habs : ¬ Present ov.nodes.val T k k)
    (v : I64) (nd : U32) (e : Usize) (o' : Ov)
    (h : hashtrinity.overflow.Overflow.insert_new_at CI ov k k v = ok ((nd, e), o')) :
    ∃ T', OvFull ks s (upd k v m) o' T' ∧ (∀ c, InT o' T' c ↔ InT ov T c ∨ c = k) ∧
      ¬ InT ov T k ∧ o'.nodes.val.length ≤ ov.nodes.val.length + 1 := by
  obtain ⟨hI, hall⟩ := hF
  have hfresh : ∀ j ∈ idxs T, keyAt ov.nodes.val j ≠ k.val := by
    intro j hj hk
    obtain ⟨a, ha, _, hent, _⟩ := hall j hj
    have hak : a.key = k := UScalar.eq_of_val_eq (by rw [← keyAt_get ha]; exact hk)
    exact habs ⟨j, hj, a, ha, hak, 0, (a.key, some (m a.key)), by rw [hent]; rfl, hak⟩
  obtain ⟨T', hI', _, hlenA, hndT, ⟨nde, hnde, hkey, _, _⟩, _, hfr, _⟩ :=
    insert_new_at_spec ov o' T k k v nd e hI hcap habs h
  obtain ⟨hperm, ⟨nde', hnde', hent', hal'⟩, hframe⟩ := hfr hfresh
  rw [hnde] at hnde'
  have := Option.some.inj hnde'
  subst this
  have hmem : ∀ j, j ∈ idxs T' ↔ j = nd.val ∨ j ∈ idxs T := fun j => by
    rw [hperm.mem_iff, List.mem_cons]
  refine ⟨T', ⟨hI', fun j hj => ?_⟩, fun c => ⟨?_, ?_⟩, ?_, hlenA⟩
  · rcases (hmem j).mp hj with rfl | hjT
    · refine ⟨nde, hnde, hal', by rw [hent', hkey, upd_self], by rw [hkey]; exact hns,
        fun j' hj' => by rw [hkey]; exact hblk j' hj'⟩
    · obtain ⟨a, ha, hala, hent, hans, hP⟩ := hall j hjT
      obtain ⟨b, hb, hsk, hse, hsa⟩ := hframe j hjT a ha
      have hak : a.key ≠ k := fun e => hfresh j hjT (by rw [keyAt_get ha, e])
      refine ⟨b, hb, by rw [← hsa]; exact hala, ?_, by rw [← hsk]; exact hans,
        by rw [← hsk]; exact hP⟩
      rw [← hse, ← hsk, hent, upd_ne v m hak]
  · rintro ⟨j, hj, b, hb, hbk⟩
    rcases (hmem j).mp hj with rfl | hjT
    · rw [hnde] at hb
      have := Option.some.inj hb
      subst this
      exact Or.inr (hbk ▸ hkey)
    · obtain ⟨a, ha, _⟩ := hall j hjT
      obtain ⟨b', hb', hsk, _⟩ := hframe j hjT a ha
      rw [hb] at hb'
      have := Option.some.inj hb'
      subst this
      exact Or.inl ⟨j, hjT, a, ha, hsk.trans hbk⟩
  · rintro (⟨j, hjT, a, ha, hak⟩ | rfl)
    · obtain ⟨b, hb, hsk, _⟩ := hframe j hjT a ha
      exact ⟨j, (hmem j).mpr (Or.inr hjT), b, hb, hsk.symm.trans hak⟩
    · exact ⟨nd.val, hndT, nde, hnde, hkey⟩
  · rintro ⟨j, hj, a, ha, hak⟩
    exact hfresh j hj (by rw [keyAt_get ha, hak])

/-- `blocked_new_full_abs` with the absence stated as `find_entry = none`. -/
theorem blocked_new_full {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} (ov : Ov) (T : OTree)
    (hF : OvFull ks s m ov T) {k : U64} (hns : ¬ IsSent k) (hblk : CappedOK ks s k .Blocked)
    (hcap : ov.nodes.val.length < nilN)
    (hfe : hashtrinity.overflow.Overflow.find_entry CI core.cmp.EqU64 ov k k = ok none)
    (v : I64) (nd : U32) (e : Usize) (o' : Ov)
    (h : hashtrinity.overflow.Overflow.insert_new_at CI ov k k v = ok ((nd, e), o')) :
    ∃ T', OvFull ks s (upd k v m) o' T' ∧ (∀ c, InT o' T' c ↔ InT ov T c ∨ c = k) ∧
      ¬ InT ov T k ∧ o'.nodes.val.length ≤ ov.nodes.val.length + 1 :=
  blocked_new_full_abs ks ov T hF hns hblk hcap
    ((find_entry_none_iff ov T hF.1.1 hF.1.2.1 k k none hfe).mp rfl) v nd e o' h

/-- `Blocked` on a present key: `take_value` then `write_back` replaces its node's only entry. -/
theorem blocked_hit_full {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} (ov : Ov) (T : OTree)
    (hF : OvFull ks s m ov T) {k : U64} (n e : U32)
    (hfe : hashtrinity.overflow.Overflow.find_entry CI core.cmp.EqU64 ov k k = ok (some (n, e)))
    (old : I64) (ov1 : Ov)
    (ht : hashtrinity.overflow.Overflow.take_value CI ov n e = ok (old, ov1))
    (v : I64) (ov2 : Ov) (hw : hashtrinity.overflow.Overflow.write_back CI ov1 n e v = ok ov2) :
    OvFull ks s (upd k v m) ov2 T ∧ (∀ c, InT ov2 T c ↔ InT ov T c) ∧ InT ov T k ∧
      ov2.nodes.val.length = ov.nodes.val.length := by
  obtain ⟨hI, hall⟩ := hF
  obtain ⟨hE, hB, hND, hH, hBal, hfn, hfl⟩ := hI
  have hlen : ∀ j ∈ idxs T, ∀ nd, ov.nodes.val[j]? = some nd → nd.entries.val.length < 2 ^ 32 := by
    intro j hj nd hnd
    obtain ⟨a, ha, _, hent, _⟩ := hall j hj
    rw [ha] at hnd
    have := Option.some.inj hnd
    subst this
    rw [hent]; simp
  obtain ⟨hn, nd, hnd, hkey, ⟨⟨x, hx, _⟩, _⟩⟩ :=
    ((find_entry_spec ov T hE hB hlen k k _ hfe).2 n e).mp rfl
  obtain ⟨a, ha, hala, hent, hans, hP⟩ := hall n.val hn
  rw [hnd] at ha
  have := Option.some.inj ha
  subst a
  have he0 : e.val = 0 := by
    have := (List.getElem?_eq_some_iff.mp hx).1
    rw [hent] at this; simp at this; omega
  have hnlt : n.val < ov.nodes.val.length := (List.getElem?_eq_some_iff.mp hnd).1
  -- `take_value`
  unfold hashtrinity.overflow.Overflow.take_value at ht
  simp only [lift, bind_tc_ok] at ht
  obtain ⟨⟨n1, back⟩, hn1, ht⟩ := bind_inv ht
  obtain ⟨hn1', hback⟩ := idxm_ok hn1
  rw [hnd] at hn1'
  have := Option.some.inj hn1'
  subst n1
  obtain ⟨⟨⟨t, w⟩, back1⟩, hent1, ht⟩ := bind_inv ht
  obtain ⟨hent1', hback1⟩ := vec_index_mut_ok _ _ _ _ hent1
  rw [cast_usize_u32, he0, hent] at hent1'
  have htw : (t, w) = (nd.key, some (m nd.key)) := (Option.some.inj hent1').symm
  simp only [Prod.mk.injEq] at htw
  obtain ⟨rfl, rfl⟩ := htw
  obtain ⟨t1, -, ht⟩ := bind_inv ht
  have ht' := Result.ok_injective ht
  simp only [Prod.mk.injEq] at ht'
  obtain ⟨-, rfl⟩ := ht'
  have he1 : (back1 (nd.key, none)).val = [(nd.key, none)] := by
    rw [hback1, vec_set_val, cast_usize_u32, he0, hent]; rfl
  -- `write_back`
  unfold hashtrinity.overflow.Overflow.write_back at hw
  simp only [lift, bind_tc_ok] at hw
  obtain ⟨⟨n2, back2⟩, hn2, hw⟩ := bind_inv hw
  obtain ⟨hn2', hback2⟩ := idxm_ok hn2
  obtain ⟨⟨⟨t2, w2⟩, back3⟩, hent2, hw⟩ := bind_inv hw
  obtain ⟨hent2', hback3⟩ := vec_index_mut_ok _ _ _ _ hent2
  have hw' := (Result.ok_injective hw).symm
  subst hw'
  have hn2v : n2 = {nd with entries := back1 (nd.key, none)} := by
    have h0 : (back {nd with entries := back1 (nd.key, none)}).val[n.val]? = some n2 := hn2'
    rw [hback, List.getElem?_set_self hnlt] at h0
    exact (Option.some.inj h0).symm
  subst hn2v
  have ht2 : t2 = nd.key := by
    have h0 := hent2'
    dsimp only at h0
    rw [cast_usize_u32, he0, he1] at h0
    have := Option.some.inj h0
    simp only [Prod.mk.injEq] at this
    exact this.1.symm
  subst ht2
  have hY : (back3 (nd.key, some v)).val = [(nd.key, some v)] := by
    rw [hback3, vec_set_val, cast_usize_u32, he0]
    dsimp only
    rw [he1]; rfl
  have hnodes : (back2 { ({nd with entries := back1 (nd.key, none)} : ONode) with
      entries := back3 (nd.key, some v) }).val =
      ov.nodes.val.set n.val {nd with entries := back3 (nd.key, some v)} := by
    rw [hback2, hback, List.set_set]
  have hne : ∀ j, j ≠ n.val → (ov.nodes.val.set n.val {nd with entries := back3 (nd.key, some v)})[j]? =
      ov.nodes.val[j]? := fun j hj => List.getElem?_set_ne (Ne.symm hj)
  have hself : (ov.nodes.val.set n.val {nd with entries := back3 (nd.key, some v)})[n.val]? =
      some {nd with entries := back3 (nd.key, some v)} := List.getElem?_set_self hnlt
  have hkeys : ∀ j, keyAt (ov.nodes.val.set n.val {nd with entries := back3 (nd.key, some v)}) j =
      keyAt ov.nodes.val j := by
    intro j
    by_cases hjj : j = n.val
    · subst hjj; rw [keyAt_get hself, keyAt_get hnd]
    · unfold keyAt; rw [hne j hjj]
  have hptr : ∀ j ∈ idxs T, ∀ nd0, ov.nodes.val[j]? = some nd0 →
      ∃ nd', (ov.nodes.val.set n.val {nd with entries := back3 (nd.key, some v)})[j]? = some nd' ∧
        nd'.left = nd0.left ∧ nd'.right = nd0.right := by
    intro j _ nd0 hnd0
    by_cases hjj : j = n.val
    · subst hjj
      rw [hnd] at hnd0
      have := Option.some.inj hnd0
      subst this
      exact ⟨_, hself, rfl, rfl⟩
    · exact ⟨nd0, by rw [hne j hjj]; exact hnd0, rfl, rfl⟩
  have hhag : ∀ j, HAg ov.nodes.val (ov.nodes.val.set n.val {nd with entries := back3 (nd.key, some v)}) j := by
    intro j nd0 hnd0
    by_cases hjj : j = n.val
    · subst hjj
      rw [hnd] at hnd0
      have := Option.some.inj hnd0
      subst this
      exact ⟨_, hself, rfl⟩
    · exact ⟨nd0, by rw [hne j hjj]; exact hnd0, rfl⟩
  have hlenS : (ov.nodes.val.set n.val {nd with entries := back3 (nd.key, some v)}).length =
      ov.nodes.val.length := List.length_set
  have hInT : ∀ c, InT { { ov with nodes := back {nd with entries := back1 (nd.key, none)} } with
      nodes := back2 { ({nd with entries := back1 (nd.key, none)} : ONode) with
        entries := back3 (nd.key, some v) } } T c ↔ InT ov T c := by
    intro c
    unfold InT
    dsimp only
    rw [hnodes]
    constructor
    · rintro ⟨j, hj, b, hb, hbk⟩
      by_cases hjj : j = n.val
      · subst hjj
        rw [hself] at hb
        have := Option.some.inj hb
        subst this
        exact ⟨n.val, hj, nd, hnd, hbk⟩
      · rw [hne j hjj] at hb
        exact ⟨j, hj, b, hb, hbk⟩
    · rintro ⟨j, hj, b, hb, hbk⟩
      by_cases hjj : j = n.val
      · subst hjj
        rw [hnd] at hb
        have := Option.some.inj hb
        subst this
        exact ⟨n.val, hj, _, hself, hbk⟩
      · exact ⟨j, hj, b, by rw [hne j hjj]; exact hb, hbk⟩
  refine ⟨⟨⟨?_, ?_, hND, ?_, hBal, hfn, ?_⟩, fun j hj => ?_⟩, hInT, ⟨n.val, hn, nd, hnd, hkey⟩, ?_⟩
  · show Emb (back2 _).val ov.root T
    rw [hnodes]; exact emb_ptr_congr hE hptr
  · show BST (back2 _).val T
    rw [hnodes]; exact bst_congr' (fun j _ => hkeys j) hB
  · show HOK (back2 _).val T
    rw [hnodes]; exact hok_hcongr hH (fun j _ => hhag j)
  · intro f hf
    refine ⟨?_, (hfl f hf).2⟩
    show f.val < (back2 _).val.length
    rw [hnodes, hlenS]; exact (hfl f hf).1
  · show ∃ nd', (back2 _).val[j]? = some nd' ∧ _
    rw [hnodes]
    by_cases hjj : j = n.val
    · subst hjj
      refine ⟨_, hself, hala, ?_, hans, hP⟩
      show (back3 (nd.key, some v)).val = [(nd.key, some (upd k v m nd.key))]
      rw [hY, hkey, upd_self]
    · obtain ⟨b, hb, hbal, hbent, hbns, hbP⟩ := hall j hj
      have hbk : b.key ≠ k := by
        intro e
        apply bst_key_inj hB hj hn hjj
        rw [keyAt_get hb, keyAt_get hnd, e, hkey]
      refine ⟨b, by rw [hne j hjj]; exact hb, hbal, by rw [hbent, upd_ne v m hbk], hbns, hbP⟩
  · show (back2 _).val.length = _
    rw [hnodes, hlenS]

end

end Apeiron.MapExecuted
