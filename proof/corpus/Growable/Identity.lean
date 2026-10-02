import ApeironWord
import Apeiron.Table.Basic
import Apeiron.Table.WordIdentity

/-!
# Extracted growable-carrier identity

This module refines the identity observations of the extracted prepared-core carriers to equality
of the canonical `Apeiron.Table.HMap (Sigma × K) V` key.  It does not refine table operations.
The concrete regenerating face is admitted for every extracted `I64`; its inverse follows the
extracted `Addr.id` word through `coord.unspread`.  No statement recovers a key from `Addr.place`.
-/

namespace Apeiron.GrowableIdentity

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

abbrev Sigma := apeiron_word.cell.Sigma
abbrev StoredI64 := apeiron_word.table.keyrep.Stored I64
abbrev RegenI64 :=
  apeiron_word.table.keyrep.Regen apeiron_word.table.keyrep.I64WordCodec

abbrev wordExactIdentityInst :=
  I64.Insts.Apeiron_wordCoordLocatorLocator_coreExactIdentityU64Tuple

abbrev wordHomeInst :=
  table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma

abbrev wordCodecInst :=
  table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces
    wordHomeInst

abbrev storedKeyRep :=
  table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep wordExactIdentityInst
    core.clone.CloneI64 wordHomeInst

abbrev regenKeyRep :=
  table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep wordExactIdentityInst
    wordHomeInst wordCodecInst

/-- The carried-key face denotes its retained key. -/
def decodeStored (_sigma : Sigma) (held : StoredI64) : I64 := held.key

/-- The regenerating face denotes the result of the extracted sealed codec. -/
def decodeRegen (sigma : Sigma) (held : RegenI64) : Result I64 :=
  apeiron_word.word_i64_codec_decode sigma held.identity

/-- A canonical logical key retains `Sigma` independently of the carrier. -/
def logicalStored (sigma : Sigma) (held : StoredI64) : Sigma × I64 :=
  (sigma, decodeStored sigma held)

/-- The extracted inverse multiplier is inverse to the extracted identity multiplier. -/
theorem coord_unspread_spread_id (x : U64) :
    coord.spread x ⦃ address => coord.unspread address.id = ok x ⦄ := by
  unfold coord.spread coord.unspread
  step
  step
  step
  step
  step
  step
  step
  step
  rw [lo_post, w_post, i_post, i1_post]
  apply UScalar.eq_of_val_eq
  simp only [core.num.U64.wrapping_mul_val_eq, UScalar.cast_val_eq,
    core.num.U128.wrapping_mul_val_eq]
  simp only [UScalar.size, UScalarTy.numBits, coord.PHI, coord.PHI_INV]
  norm_num
  rw [Nat.mul_assoc]
  have hinv :
      (11400714819323198485 * 17428512612931826493) % 18446744073709551616 = 1 := by
    norm_num
  rw [Nat.mul_mod, hinv, Nat.mul_one]
  have hx : x.val < 18446744073709551616 := by simpa using x.hBounds
  simp [Nat.mod_eq_of_lt hx]

theorem coord_PHI_val : coord.PHI.val = Apeiron.Table.phi64 := by
  rw [coord.PHI]
  rfl

theorem spread_low_word (x : U64) :
    (UScalar.cast .U64
      (core.num.U128.wrapping_mul (UScalar.cast .U128 x)
        (UScalar.cast .U128 coord.PHI))).val =
      Apeiron.Table.lowWord 64 (x.val * Apeiron.Table.phi64) := by
  rw [UScalar.cast_val_eq, core.num.U128.wrapping_mul_val_eq]
  simp only [U64.cast_U128_val_eq, Apeiron.Table.lowWord]
  rw [coord_PHI_val]
  have hx := x.hBounds
  have hfit := Apeiron.Table.phi64_product_fits_u128 hx
  rw [UScalar.size]
  change ((x.val * Apeiron.Table.phi64) % 2 ^ 128) % 2 ^ 64 =
    (x.val * Apeiron.Table.phi64) % 2 ^ 64
  rw [Nat.mod_eq_of_lt hfit]

def coordSpreadId (x : U64) : U64 :=
  UScalar.cast .U64
    (core.num.U128.wrapping_mul (UScalar.cast .U128 x)
      (UScalar.cast .U128 coord.PHI))

theorem coord_spread_returns_id (x : U64) :
    coord.spread x ⦃ addr => addr.id = coordSpreadId x ⦄ := by
  unfold coord.spread
  step
  step
  step
  step
  step
  step
  step
  step
  rw [lo_post, w_post, i_post, i1_post]
  rfl

theorem coordSpreadId_injective : Function.Injective coordSpreadId := by
  intro x y hid
  apply UScalar.eq_of_val_eq
  apply Apeiron.Table.phi64_lowWord_injective x.hBounds y.hBounds
  rw [← spread_low_word x, ← spread_low_word y]
  exact congrArg UScalar.val hid

theorem coord_spread_id_injective (x y : U64) :
    coord.spread x ⦃ ax => coord.spread y ⦃ ay => ax.id = ay.id → x = y ⦄ ⦄ := by
  unfold coord.spread
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  step
  intro hid
  apply coordSpreadId_injective
  subst_vars
  simpa [coordSpreadId] using lo_post

attribute [local step] coord_spread_returns_id

/-- `coord.unspread` is the verified inverse of the extracted identity-field expression. -/
theorem coord_unspread_coordSpreadId (x : U64) :
    coord.unspread (coordSpreadId x) = ok x := by
  unfold coord.unspread coordSpreadId
  apply congrArg ok
  apply UScalar.eq_of_val_eq
  simp only [core.num.U64.wrapping_mul_val_eq, UScalar.cast_val_eq,
    core.num.U128.wrapping_mul_val_eq]
  simp only [UScalar.size, UScalarTy.numBits, coord.PHI, coord.PHI_INV]
  norm_num
  rw [Nat.mul_assoc]
  have hinv :
      (11400714819323198485 * 17428512612931826493) % 18446744073709551616 = 1 := by
    norm_num
  rw [Nat.mul_mod, hinv, Nat.mul_one]
  have hx : x.val < 18446744073709551616 := by simpa using x.hBounds
  simp [Nat.mod_eq_of_lt hx]

@[step]
theorem word_home_address_total (sigma : Sigma) (key : I64) :
    wordHomeInst.address () sigma key ⦃ _ => True ⦄ := by
  change table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address () sigma key ⦃ _ => True ⦄
  unfold table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma.address
  step
  step
  unfold core.convert.num.FromU64U32.from
  simp [lift]
  split <;> simp

/-- The extracted `I64WordCodec` round-trips every `I64` through the identity field. -/
theorem i64_codec_roundtrip (sigma : Sigma) (key : I64) :
    word_i64_codec_encode sigma key ⦃ identity =>
      word_i64_codec_decode sigma identity = ok key ⦄ := by
  unfold word_i64_codec_encode word_i64_codec_decode
  unfold table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.encode
  unfold table.keyrep.address_identity
  unfold table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.decode
  step
  step
  step
  step
  step
  step
  · rw [a_post, x_post]
    congr 1
    bv_tac
  · rw [a_post, coord_unspread_coordSpreadId]
    simp only [core.num.U64.cast_signed]
    cases i
    cases key
    simp_all

/-- Encoding returns the identity field of `coord.spread`; the supplied address's `place` is unused. -/
@[step]
theorem i64_codec_encode_identity (sigma : Sigma) (key : I64) :
    word_i64_codec_encode sigma key ⦃ identity =>
      ∃ word, word.bv = key.bv ∧
        identity = coordSpreadId word ⦄ := by
  unfold word_i64_codec_encode
  unfold table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.encode
  unfold table.keyrep.address_identity
  step
  step
  step
  step
  step
  step
  · rw [a_post, x_post]
    congr 1
    bv_tac
  · exact ⟨i, i_post, a_post⟩

/-- The extracted carried-key clone preserves the decoded key. -/
theorem stored_clone_identity (held : StoredI64) :
    table.keyrep.Stored.Insts.CoreCloneClone.clone core.clone.CloneI64 held = ok held := by
  cases held
  simp [table.keyrep.Stored.Insts.CoreCloneClone.clone]

/-- `decodeStored(s, Stored(k)) = k`. -/
theorem decodeStored_identity (sigma : Sigma) (key : I64) :
    decodeStored sigma { key := key } = key := rfl

/-- `decodeRegen` is exactly the extracted sealed-codec decode operation. -/
theorem decodeRegen_identity (sigma : Sigma) (held : RegenI64) :
    decodeRegen sigma held = word_i64_codec_decode sigma held.identity := rfl

/-- A carrier is reachable on a face when it is returned by the real extracted `hold` wrapper. -/
def ReachableRegen (sigma : Sigma) (held : RegenI64) : Prop :=
  ∃ key, word_regen_hold sigma key = ok held

/-- Every carrier returned by extracted `Regen::hold` decodes to the original key. -/
theorem regen_hold_decodes (sigma : Sigma) (key : I64) :
    word_regen_hold sigma key ⦃ held => decodeRegen sigma held = ok key ⦄ := by
  unfold word_regen_hold
  unfold table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.hold
  dsimp only [wordCodecInst,
    table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces]
  unfold table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.encode
  unfold table.keyrep.address_identity
  step
  step
  step
  step
  step
  step
  · rename_i original horiginal held hheld query hquery
    rw [hheld, x_post]
    congr 1
    apply UScalar.eq_of_val_eq
    exact congrArg BitVec.toNat (horiginal.trans hquery.symm)
  · rename_i original horiginal held hheld query hquery
    simp only [decodeRegen, word_i64_codec_decode,
      table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.decode]
    rw [hheld, coord_unspread_coordSpreadId]
    simp only [core.num.U64.cast_signed]
    cases original
    cases key
    simp_all

/-- The extracted `Regen::hold` wrapper stores exactly the extracted codec output. -/
theorem word_regen_hold_eq (sigma : Sigma) (key : I64) :
    word_regen_hold sigma key = (do
      let identity ← word_i64_codec_encode sigma key
      ok ({ identity := identity, codec := () } : RegenI64)) := by
  unfold word_regen_hold word_i64_codec_encode
  unfold table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.hold
  dsimp only [wordCodecInst,
    table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces]
  simp only [bind_assoc_eq]

/-- The extracted `Regen::matches` wrapper compares against exactly the codec output. -/
theorem word_regen_matches_eq (sigma : Sigma) (held : RegenI64) (key : I64) :
    word_regen_matches sigma held key = (do
      let identity ← word_i64_codec_encode sigma key
      ok (held.identity = identity)) := by
  unfold word_regen_matches word_i64_codec_encode
  unfold table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.matches
  dsimp only [wordCodecInst,
    table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces]
  simp only [bind_assoc_eq]

/-- Equal extracted codec outputs imply equal logical `I64` keys. -/
theorem i64_codec_encode_injective (sigma : Sigma) {left right : I64} {identity : U64}
    (hl : word_i64_codec_encode sigma left = ok identity)
    (hr : word_i64_codec_encode sigma right = ok identity) : left = right := by
  have dl : word_i64_codec_decode sigma identity = ok left := by
    simpa [hl] using i64_codec_roundtrip sigma left
  have dr : word_i64_codec_decode sigma identity = ok right := by
    simpa [hr] using i64_codec_roundtrip sigma right
  simpa using (dl.symm.trans dr)

/-- On a reachable regenerating carrier, extracted matching is logical-key equality. -/
theorem regen_matches_iff_of_reachable (sigma : Sigma) (held : RegenI64) (original query : I64)
    (hheld : word_regen_hold sigma original = ok held) :
    word_regen_matches sigma held query = ok true ↔ query = original := by
  obtain ⟨originalIdentity, hencodeOriginal, _horiginalProperty⟩ :=
    WP.spec_imp_exists (i64_codec_encode_identity sigma original)
  have hidentity : held.identity = originalIdentity := by
    rw [word_regen_hold_eq, hencodeOriginal] at hheld
    have hcarrier : ({ identity := originalIdentity, codec := () } : RegenI64) = held := by
      simpa using hheld
    exact congrArg table.keyrep.Regen.identity hcarrier.symm
  obtain ⟨queryIdentity, hencodeQuery, _hqueryProperty⟩ :=
    WP.spec_imp_exists (i64_codec_encode_identity sigma query)
  rw [word_regen_matches_eq, hencodeQuery, hidentity]
  simp
  constructor
  · intro hid
    exact i64_codec_encode_injective sigma hencodeQuery
      (hencodeOriginal.trans (congrArg ok hid))
  · intro h
    subst query
    simpa using (hencodeOriginal.symm.trans hencodeQuery)

/-- The generated `Sigma` comparison is faithful. -/
theorem sigma_eq_true_iff (left right : Sigma) :
    cell.Sigma.Insts.CoreCmpPartialEqSigma.eq left right = ok true ↔ left = right := by
  cases left <;> cases right <;>
    simp [cell.Sigma.Insts.CoreCmpPartialEqSigma.eq, cell.Sigma.read_discriminant]

/-- On the carried-key face, the real extracted slot matcher is canonical logical-key equality. -/
theorem stored_matches_iff (slotSigma querySigma : Sigma) (held : StoredI64) (value : I64)
    (query : I64) (address : table.TableAddress) :
    table.matches
        storedKeyRep
        { tag := 0#u8, sigma := slotSigma, held := held, value := value }
        querySigma query address = ok true ↔
      logicalStored slotSigma held = (querySigma, query) := by
  cases slotSigma <;> cases querySigma <;> cases held <;>
    simp [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant,
      table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.matches,
      logicalStored, decodeStored]

/-- The concrete sealed-codec call returns the spread identity for every supplied address. -/
theorem codec_encode_at_identity (sigma : Sigma) (key : I64) (address : table.TableAddress) :
    table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.encode
        wordHomeInst sigma key address ⦃ identity =>
      ∃ word, word.bv = key.bv ∧
        identity = coordSpreadId word ⦄ := by
  unfold table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces.encode
  unfold table.keyrep.address_identity
  step
  step
  step
  step
  step
  · rw [a_post, x_post]
    congr 1
    bv_tac
  · exact ⟨i, i_post, a_post⟩

/-- On a reachable carrier, the extracted regenerating-face comparison is decoded-key equality. -/
theorem regen_carrier_matches_iff_of_reachable (sigma : Sigma) (held : RegenI64)
    (original query : I64) (address : table.TableAddress)
    (hheld : word_regen_hold sigma original = ok held) :
    regenKeyRep.matches sigma held query address = ok true ↔ query = original := by
  obtain ⟨originalIdentity, hencodeOriginal, horiginalProperty⟩ :=
    WP.spec_imp_exists (i64_codec_encode_identity sigma original)
  have hidentity : held.identity = originalIdentity := by
    rw [word_regen_hold_eq, hencodeOriginal] at hheld
    have hcarrier : ({ identity := originalIdentity, codec := () } : RegenI64) = held := by
      simpa using hheld
    exact congrArg table.keyrep.Regen.identity hcarrier.symm
  obtain ⟨queryIdentity, hencodeQuery, hqueryProperty⟩ :=
    WP.spec_imp_exists (codec_encode_at_identity sigma query address)
  change
    table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.matches wordExactIdentityInst
      wordHomeInst wordCodecInst sigma held query address = ok true ↔ query = original
  unfold table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.matches
  dsimp only [regenKeyRep, wordCodecInst,
    table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces]
  rw [hencodeQuery, hidentity]
  simp
  constructor
  · intro hid
    obtain ⟨queryWord, hqueryWord, hqueryIdentity⟩ := hqueryProperty
    obtain ⟨originalWord, horiginalWord, horiginalIdentity⟩ := horiginalProperty
    have hw : queryWord = originalWord := by
      apply coordSpreadId_injective
      rw [← hqueryIdentity, ← horiginalIdentity]
      exact hid.symm
    cases query
    cases original
    simp_all
  · intro h
    subst query
    obtain ⟨queryWord, hqueryWord, hqueryIdentity⟩ := hqueryProperty
    obtain ⟨originalWord, horiginalWord, horiginalIdentity⟩ := horiginalProperty
    have hw : queryWord = originalWord := by
      apply UScalar.eq_of_val_eq
      exact congrArg BitVec.toNat (hqueryWord.trans horiginalWord.symm)
    rw [hqueryIdentity, horiginalIdentity, hw]

/-- The real extracted slot matcher preserves `Sigma` and decoded equality on the regen face. -/
theorem regen_slot_matches_iff_of_reachable (slotSigma querySigma : Sigma) (held : RegenI64)
    (original query value : I64) (address : table.TableAddress)
    (hheld : word_regen_hold slotSigma original = ok held) :
    table.matches
        regenKeyRep
        { tag := 0#u8, sigma := slotSigma, held := held, value := value }
        querySigma query address = ok true ↔
      (slotSigma, original) = (querySigma, query) := by
  cases slotSigma <;> cases querySigma
  · simpa [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant, eq_comm] using
      regen_carrier_matches_iff_of_reachable cell.Sigma.Elliptic held original query address hheld
  · simp [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant]
  · simp [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant]
  · simp [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant]
  · simpa [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant, eq_comm] using
      regen_carrier_matches_iff_of_reachable cell.Sigma.Parabolic held original query address hheld
  · simp [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant]
  · simp [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant]
  · simp [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant]
  · simpa [table.matches, cell.Sigma.Insts.CoreCmpPartialEqSigma.eq,
      cell.Sigma.read_discriminant, eq_comm] using
      regen_carrier_matches_iff_of_reachable cell.Sigma.Hyperbolic held original query address hheld

/-- Reconstructed regen carriers are addressed exactly as their decoded original key. -/
theorem regen_address_held_agrees (sigma : Sigma) (held : RegenI64) (original : I64)
    (hheld : word_regen_hold sigma original = ok held) :
    word_regen_address_held sigma held =
      wordHomeInst.address () sigma original := by
  have hdecode := regen_hold_decodes sigma original
  rw [hheld] at hdecode
  simp only [WP.spec_ok] at hdecode
  unfold word_regen_address_held
  unfold table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.address_held
  dsimp only [wordCodecInst,
    table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces]
  change (do
    let key ← decodeRegen sigma held
    wordHomeInst.address () sigma key) = _
  rw [hdecode]
  simp

/-- Stored carriers are addressed by the retained original key. -/
theorem stored_address_held_agrees (home : table.WordHome) (sigma : Sigma) (held : StoredI64) :
    storedKeyRep.address_held home sigma held =
      wordHomeInst.address home sigma
        (decodeStored sigma held) := rfl

/-- Faithful matching is equality of the exact canonical `HMap (Sigma × I64) V` key. -/
def MatchesIdentity : Prop :=
    (∀ (slotSigma querySigma : Sigma) (held : StoredI64) (value query : I64)
        (address : table.TableAddress),
      table.matches
          storedKeyRep
          { tag := 0#u8, sigma := slotSigma, held := held, value := value }
          querySigma query address = ok true ↔
        logicalStored slotSigma held = (querySigma, query)) ∧
    (∀ (slotSigma querySigma : Sigma) (held : RegenI64) (original query value : I64)
        (address : table.TableAddress), word_regen_hold slotSigma original = ok held →
      (table.matches
          regenKeyRep
          { tag := 0#u8, sigma := slotSigma, held := held, value := value }
          querySigma query address = ok true ↔
        (slotSigma, original) = (querySigma, query)))

/-- Carrier identity is the codec, equality, clone, `Sigma`, and reconstructed-address contract. -/
structure IdentityContract : Prop where
  storedDecode : ∀ (sigma : Sigma) (key : I64), decodeStored sigma { key := key } = key
  regenDecode : ∀ (sigma : Sigma) (held : RegenI64),
    decodeRegen sigma held = word_i64_codec_decode sigma held.identity
  storedClone : ∀ held : StoredI64,
    table.keyrep.Stored.Insts.CoreCloneClone.clone core.clone.CloneI64 held = ok held
  codecRoundtrip : ∀ (sigma : Sigma) (key : I64),
    word_i64_codec_encode sigma key ⦃ identity =>
      word_i64_codec_decode sigma identity = ok key ⦄
  matchesIdentity : MatchesIdentity
  storedAddress : ∀ (home : table.WordHome) (sigma : Sigma) (held : StoredI64),
    storedKeyRep.address_held home sigma held =
      wordHomeInst.address home sigma
        (decodeStored sigma held)
  regenAddress : ∀ (sigma : Sigma) (held : RegenI64) (original : I64),
    word_regen_hold sigma original = ok held →
      word_regen_address_held sigma held =
        wordHomeInst.address () sigma original
  sigmaSeparate : ∀ (left right : Sigma) (key : I64), left ≠ right →
    (left, key) ≠ (right, key)

/--
The extracted prepared-core carriers refine equality of the canonical
`Apeiron.Table.HMap (Sigma × I64) V` key.  This does not refine table operations.
-/
theorem keyRep_identity : IdentityContract := by
  refine {
    storedDecode := decodeStored_identity
    regenDecode := decodeRegen_identity
    storedClone := stored_clone_identity
    codecRoundtrip := i64_codec_roundtrip
    matchesIdentity := ?_
    storedAddress := stored_address_held_agrees
    regenAddress := regen_address_held_agrees
    sigmaSeparate := by
      intro left right key h heq
      exact h (congrArg Prod.fst heq)
  }
  exact ⟨stored_matches_iff, regen_slot_matches_iff_of_reachable⟩

/-- Distinct `Sigma` values with the same `I64` are distinct canonical `HMap` keys. -/
theorem canonical_sigma_separate (left right : Sigma) (key : I64) (h : left ≠ right) :
    (left, key) ≠ (right, key) := by
  intro heq
  exact h (congrArg Prod.fst heq)

/-- Canonical `HMap` insertion at one `Sigma` leaves the same `I64` on another face unchanged. -/
theorem canonical_sigma_insert_separate {V : Type} [DecidableEq Sigma]
    (merge : V → V → V) (map : Apeiron.Table.HMap (Sigma × I64) V)
    (left right : Sigma) (key : I64) (value : V) (h : left ≠ right) :
    Apeiron.Table.get (Apeiron.Table.insertWith merge (left, key) value map) (right, key) =
      Apeiron.Table.get map (right, key) := by
  apply Apeiron.Table.get_insertWith_ne
  exact fun heq => canonical_sigma_separate left right key h heq.symm

end Apeiron.GrowableIdentity
