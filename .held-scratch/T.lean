import Native.LocatorSpec
import Growable.Identity
namespace Apeiron.NativeLocator.Native.Address
open Aeneas Aeneas.Std Result
open apeiron_word
open Apeiron.GrowableIdentity

def storedFaceAddressHeld (profile : coord.address.H3_40) (face : cell.Sigma)
    (held : StoredI64) : Result coord.address.FaceTableAddress :=
  table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.face_address_held
    wordExactIdentityInst core.clone.CloneI64 wordHomeInst
    table.WordHome.Insts.Apeiron_wordCoordAddressFaceHomeI64Sigma () profile face held

def regenFaceAddressHeld (profile : coord.address.H3_40) (face : cell.Sigma)
    (held : RegenI64) : Result coord.address.FaceTableAddress :=
  table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.face_address_held
    wordExactIdentityInst wordHomeInst wordCodecInst
    table.WordHome.Insts.Apeiron_wordCoordAddressFaceHomeI64Sigma () profile face held

theorem test_stored (profile : coord.address.H3_40) (face : cell.Sigma) (key : I64) :
  storedFaceAddressHeld profile face {key := key} = wordFaceAddress profile face key := by
  rfl

theorem test_regen (profile : coord.address.H3_40) (face : cell.Sigma) (held : RegenI64) (key : I64)
  (h : word_regen_hold face key = ok held) :
  regenFaceAddressHeld profile face held = wordFaceAddress profile face key := by
  have hd := regen_hold_decodes face key
  rw [h] at hd
  simp only [WP.spec_ok] at hd
  unfold regenFaceAddressHeld
  unfold table.keyrep.Regen.Insts.Apeiron_wordTableKeyrepKeyRep.face_address_held
  dsimp only [wordCodecInst,
    table.keyrep.I64WordCodec.Insts.Apeiron_wordTableKeyrepExactCodecI64HU64TupleClause0_Clause0_Clause0_Faces]
  simp only [cell.Sigma.Insts.Apeiron_wordCoordAddressFaceDomain.sigma, bind_ok, bind_tc_ok]
  change (do
    let decoded ← decodeRegen face held
    wordFaceAddress profile face decoded) = wordFaceAddress profile face key
  rw [hd]
  simp
end Apeiron.NativeLocator.Native.Address
