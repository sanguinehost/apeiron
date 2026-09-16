-- The assembled table decoder: the full `DecodeTokensDerivation` and its `decodeTokens` soundness
-- equation, ported from the throwaway spike. Every guard is discharged locally on the literal
-- chunks in `ChunksA`/`ChunksB`; no whole-source reduction is used.
import Slang.Generated.SourceWitness.Decoder.ChunksB
import Slang.Generated.SourceWitness.Parsing.Table.Chunk09
import Slang.Generated.TableDerivation
import Slang.Source.Parsing.Tokens

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Slang.Generated.SourceWitness.Decoder

open Slang Slang.Decode Slang.Source Slang.Source.Parsing
open Slang.Generated.SourceWitness.Parsing.Table

set_option linter.unusedSimpArgs false

theorem table_delims : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT0 [] := by
  have d1457 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT1457 [] := .done
  have e1392 : tableC1392 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1457 = Slang.Generated.SourceWitness.Parsing.Table.tableT1392 := by
    rw [show tableC1392 = tableL1392 ++ tableC1416 from rfl, append_assoc_empty]
    rw [show tableC1416 = tableL1416 ++ tableC1440 from rfl, append_assoc_empty]
    rw [show tableC1440 = tableL1440 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1440 = tableL1440 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1457 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1416 = tableL1416 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1440 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1392 = tableL1392 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1416 from rfl]
  have d1392 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT1392 [] := by
    exact e1392 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC1392 [] (by decide)) d1457)
  have e1344 : tableC1344 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1392 = Slang.Generated.SourceWitness.Parsing.Table.tableT1344 := by
    rw [show tableC1344 = tableL1344 ++ tableC1368 from rfl, append_assoc_empty]
    rw [show tableC1368 = tableL1368 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1368 = tableL1368 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1392 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1344 = tableL1344 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1368 from rfl]
  have d1344 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT1344 [] := by
    exact e1344 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC1344 [] (by decide)) d1392)
  have e1296 : tableC1296 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1344 = Slang.Generated.SourceWitness.Parsing.Table.tableT1296 := by
    rw [show tableC1296 = tableL1296 ++ tableC1320 from rfl, append_assoc_empty]
    rw [show tableC1320 = tableL1320 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1320 = tableL1320 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1344 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1296 = tableL1296 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1320 from rfl]
  have d1296 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT1296 [] := by
    exact e1296 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC1296 [] (by decide)) d1344)
  have e1272 : tableC1272 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1296 = Slang.Generated.SourceWitness.Parsing.Table.tableT1272 := by
    rw [show tableC1272 = tableL1272 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1272 = tableL1272 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1296 from rfl]
  have d1272 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT1272 [] := by
    exact e1272 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC1272 [] (by decide)) d1296)
  have e1176 : tableC1176 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1272 = Slang.Generated.SourceWitness.Parsing.Table.tableT1176 := by
    rw [show tableC1176 = tableL1176 ++ tableC1200 from rfl, append_assoc_empty]
    rw [show tableC1200 = tableL1200 ++ tableC1224 from rfl, append_assoc_empty]
    rw [show tableC1224 = tableL1224 ++ tableC1248 from rfl, append_assoc_empty]
    rw [show tableC1248 = tableL1248 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1248 = tableL1248 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1272 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1224 = tableL1224 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1248 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1200 = tableL1200 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1224 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1176 = tableL1176 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1200 from rfl]
  have d1176 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT1176 [] := by
    exact e1176 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC1176 [] (by decide)) d1272)
  have e912 : tableC912 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1176 = Slang.Generated.SourceWitness.Parsing.Table.tableT912 := by
    rw [show tableC912 = tableL912 ++ tableC936 from rfl, append_assoc_empty]
    rw [show tableC936 = tableL936 ++ tableC960 from rfl, append_assoc_empty]
    rw [show tableC960 = tableL960 ++ tableC984 from rfl, append_assoc_empty]
    rw [show tableC984 = tableL984 ++ tableC1008 from rfl, append_assoc_empty]
    rw [show tableC1008 = tableL1008 ++ tableC1032 from rfl, append_assoc_empty]
    rw [show tableC1032 = tableL1032 ++ tableC1056 from rfl, append_assoc_empty]
    rw [show tableC1056 = tableL1056 ++ tableC1080 from rfl, append_assoc_empty]
    rw [show tableC1080 = tableL1080 ++ tableC1104 from rfl, append_assoc_empty]
    rw [show tableC1104 = tableL1104 ++ tableC1128 from rfl, append_assoc_empty]
    rw [show tableC1128 = tableL1128 ++ tableC1152 from rfl, append_assoc_empty]
    rw [show tableC1152 = tableL1152 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1152 = tableL1152 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1176 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1128 = tableL1128 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1152 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1104 = tableL1104 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1128 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1080 = tableL1080 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1104 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1056 = tableL1056 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1080 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1032 = tableL1032 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1056 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT1008 = tableL1008 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1032 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT984 = tableL984 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT1008 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT960 = tableL960 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT984 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT936 = tableL936 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT960 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT912 = tableL912 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT936 from rfl]
  have d912 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT912 [] := by
    exact e912 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC912 [] (by decide)) d1176)
  have e720 : tableC720 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT912 = Slang.Generated.SourceWitness.Parsing.Table.tableT720 := by
    rw [show tableC720 = tableL720 ++ tableC744 from rfl, append_assoc_empty]
    rw [show tableC744 = tableL744 ++ tableC768 from rfl, append_assoc_empty]
    rw [show tableC768 = tableL768 ++ tableC792 from rfl, append_assoc_empty]
    rw [show tableC792 = tableL792 ++ tableC816 from rfl, append_assoc_empty]
    rw [show tableC816 = tableL816 ++ tableC840 from rfl, append_assoc_empty]
    rw [show tableC840 = tableL840 ++ tableC864 from rfl, append_assoc_empty]
    rw [show tableC864 = tableL864 ++ tableC888 from rfl, append_assoc_empty]
    rw [show tableC888 = tableL888 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT888 = tableL888 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT912 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT864 = tableL864 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT888 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT840 = tableL840 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT864 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT816 = tableL816 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT840 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT792 = tableL792 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT816 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT768 = tableL768 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT792 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT744 = tableL744 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT768 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT720 = tableL720 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT744 from rfl]
  have d720 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT720 [] := by
    exact e720 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC720 [] (by decide)) d912)
  have e696 : tableC696 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT720 = Slang.Generated.SourceWitness.Parsing.Table.tableT696 := by
    rw [show tableC696 = tableL696 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT696 = tableL696 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT720 from rfl]
  have d696 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT696 [] := by
    exact e696 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC696 [] (by decide)) d720)
  have e600 : tableC600 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT696 = Slang.Generated.SourceWitness.Parsing.Table.tableT600 := by
    rw [show tableC600 = tableL600 ++ tableC624 from rfl, append_assoc_empty]
    rw [show tableC624 = tableL624 ++ tableC648 from rfl, append_assoc_empty]
    rw [show tableC648 = tableL648 ++ tableC672 from rfl, append_assoc_empty]
    rw [show tableC672 = tableL672 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT672 = tableL672 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT696 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT648 = tableL648 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT672 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT624 = tableL624 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT648 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT600 = tableL600 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT624 from rfl]
  have d600 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT600 [] := by
    exact e600 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC600 [] (by decide)) d696)
  have e576 : tableC576 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT600 = Slang.Generated.SourceWitness.Parsing.Table.tableT576 := by
    rw [show tableC576 = tableL576 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT576 = tableL576 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT600 from rfl]
  have d576 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT576 [] := by
    exact e576 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC576 [] (by decide)) d600)
  have e552 : tableC552 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT576 = Slang.Generated.SourceWitness.Parsing.Table.tableT552 := by
    rw [show tableC552 = tableL552 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT552 = tableL552 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT576 from rfl]
  have d552 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT552 [] := by
    exact e552 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC552 [] (by decide)) d576)
  have e408 : tableC408 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT552 = Slang.Generated.SourceWitness.Parsing.Table.tableT408 := by
    rw [show tableC408 = tableL408 ++ tableC432 from rfl, append_assoc_empty]
    rw [show tableC432 = tableL432 ++ tableC456 from rfl, append_assoc_empty]
    rw [show tableC456 = tableL456 ++ tableC480 from rfl, append_assoc_empty]
    rw [show tableC480 = tableL480 ++ tableC504 from rfl, append_assoc_empty]
    rw [show tableC504 = tableL504 ++ tableC528 from rfl, append_assoc_empty]
    rw [show tableC528 = tableL528 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT528 = tableL528 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT552 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT504 = tableL504 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT528 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT480 = tableL480 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT504 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT456 = tableL456 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT480 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT432 = tableL432 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT456 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT408 = tableL408 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT432 from rfl]
  have d408 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT408 [] := by
    exact e408 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC408 [] (by decide)) d552)
  have e384 : tableC384 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT408 = Slang.Generated.SourceWitness.Parsing.Table.tableT384 := by
    rw [show tableC384 = tableL384 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT384 = tableL384 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT408 from rfl]
  have d384 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT384 [] := by
    exact e384 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC384 [] (by decide)) d408)
  have e336 : tableC336 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT384 = Slang.Generated.SourceWitness.Parsing.Table.tableT336 := by
    rw [show tableC336 = tableL336 ++ tableC360 from rfl, append_assoc_empty]
    rw [show tableC360 = tableL360 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT360 = tableL360 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT384 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT336 = tableL336 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT360 from rfl]
  have d336 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT336 [] := by
    exact e336 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC336 [] (by decide)) d384)
  have e312 : tableC312 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT336 = Slang.Generated.SourceWitness.Parsing.Table.tableT312 := by
    rw [show tableC312 = tableL312 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT312 = tableL312 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT336 from rfl]
  have d312 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT312 [] := by
    exact e312 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC312 [] (by decide)) d336)
  have e264 : tableC264 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT312 = Slang.Generated.SourceWitness.Parsing.Table.tableT264 := by
    rw [show tableC264 = tableL264 ++ tableC288 from rfl, append_assoc_empty]
    rw [show tableC288 = tableL288 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT288 = tableL288 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT312 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT264 = tableL264 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT288 from rfl]
  have d264 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT264 [] := by
    exact e264 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC264 [] (by decide)) d312)
  have e240 : tableC240 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT264 = Slang.Generated.SourceWitness.Parsing.Table.tableT240 := by
    rw [show tableC240 = tableL240 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT240 = tableL240 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT264 from rfl]
  have d240 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT240 [] := by
    exact e240 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC240 [] (by decide)) d264)
  have e192 : tableC192 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT240 = Slang.Generated.SourceWitness.Parsing.Table.tableT192 := by
    rw [show tableC192 = tableL192 ++ tableC216 from rfl, append_assoc_empty]
    rw [show tableC216 = tableL216 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT216 = tableL216 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT240 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT192 = tableL192 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT216 from rfl]
  have d192 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT192 [] := by
    exact e192 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC192 [] (by decide)) d240)
  have e168 : tableC168 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT192 = Slang.Generated.SourceWitness.Parsing.Table.tableT168 := by
    rw [show tableC168 = tableL168 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT168 = tableL168 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT192 from rfl]
  have d168 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT168 [] := by
    exact e168 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC168 [] (by decide)) d192)
  have e120 : tableC120 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT168 = Slang.Generated.SourceWitness.Parsing.Table.tableT120 := by
    rw [show tableC120 = tableL120 ++ tableC144 from rfl, append_assoc_empty]
    rw [show tableC144 = tableL144 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT144 = tableL144 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT168 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT120 = tableL120 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT144 from rfl]
  have d120 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT120 [] := by
    exact e120 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC120 [] (by decide)) d168)
  have e96 : tableC96 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT120 = Slang.Generated.SourceWitness.Parsing.Table.tableT96 := by
    rw [show tableC96 = tableL96 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT96 = tableL96 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT120 from rfl]
  have d96 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT96 [] := by
    exact e96 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC96 [] (by decide)) d120)
  have e48 : tableC48 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT96 = Slang.Generated.SourceWitness.Parsing.Table.tableT48 := by
    rw [show tableC48 = tableL48 ++ tableC72 from rfl, append_assoc_empty]
    rw [show tableC72 = tableL72 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT72 = tableL72 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT96 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT48 = tableL48 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT72 from rfl]
  have d48 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT48 [] := by
    exact e48 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC48 [] (by decide)) d96)
  have e24 : tableC24 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT48 = Slang.Generated.SourceWitness.Parsing.Table.tableT24 := by
    rw [show tableC24 = tableL24 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT24 = tableL24 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT48 from rfl]
  have d24 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT24 [] := by
    exact e24 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC24 [] (by decide)) d48)
  have e0 : tableC0 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT24 = Slang.Generated.SourceWitness.Parsing.Table.tableT0 := by
    rw [show tableC0 = tableL0 from rfl]
    rw [← show Slang.Generated.SourceWitness.Parsing.Table.tableT0 = tableL0 ++ Slang.Generated.SourceWitness.Parsing.Table.tableT24 from rfl]
  have d0 : DelimitersBalancedDerivation Slang.Generated.SourceWitness.Parsing.Table.tableT0 [] := by
    exact e0 ▸ (DelimitersBalancedDerivation.append (DelimitersBalancedDerivation.of_bool tableC0 [] (by decide)) d24)
  exact d0
def tableConstantNames : List String := ["TABLE_PRESENT", "TABLE_EMPTY", "TABLE_STRIDE", "MERGE_SET", "MERGE_MAP", "MERGE_ACC", "MERGE_GRAPH", "TABLE_PHI", "COORD_CEILING"]

def tableFunctionNames : List String := ["tableSpread", "tableSlotKey", "tableSlotVal", "tableSlotSet", "tableSlotValSet", "tableInitGo", "tableInit", "tableCapacityOk", "tableFindSlotGo", "tableLookupSlotGo", "tableFind", "tableGet", "tableInsertWith"]

def tableConstants : List ConstantDecl := [{ name := "TABLE_PRESENT", value := .literal 1, span := ⟨204, 221⟩ }, { name := "TABLE_EMPTY", value := .binary .sub (.literal 0) (.literal 1), span := ⟨222, 241⟩ }, { name := "TABLE_STRIDE", value := .literal 2, span := ⟨242, 258⟩ }, { name := "MERGE_SET", value := .literal 1, span := ⟨295, 308⟩ }, { name := "MERGE_MAP", value := .literal 2, span := ⟨365, 378⟩ }, { name := "MERGE_ACC", value := .literal 3, span := ⟨413, 426⟩ }, { name := "MERGE_GRAPH", value := .literal 4, span := ⟨476, 491⟩ }, { name := "TABLE_PHI", value := .literal 2654435769, span := ⟨633, 655⟩ }]

def tableFunctions : List FunctionDecl := [{ name := "tableSpread", parameters := ["k"], body := [.bind "pos" (.conditional (.binary .sub (.literal 0) (.name "k")) (.binary .lt (.name "k") (.literal 0)) (.name "k")), .return_ (.binary .bitAnd (.binary .mul (.name "pos") (.name "TABLE_PHI")) (.literal 4611686018427387903))], span := ⟨738, 839⟩ }, { name := "tableSlotKey", parameters := ["base", "slot"], body := [.return_ (.memory (.binary .add (.name "base") (.binary .mul (.name "TABLE_STRIDE") (.name "slot"))))], span := ⟨1101, 1174⟩ }, { name := "tableSlotVal", parameters := ["base", "slot"], body := [.return_ (.memory (.binary .add (.binary .add (.name "base") (.binary .mul (.name "TABLE_STRIDE") (.name "slot"))) (.literal 1)))], span := ⟨1446, 1523⟩ }, { name := "tableSlotSet", parameters := ["base", "slot", "key", "value"], body := [.store (.binary .add (.name "base") (.binary .mul (.name "TABLE_STRIDE") (.name "slot"))) (.name "key"), .store (.binary .add (.binary .add (.name "base") (.binary .mul (.name "TABLE_STRIDE") (.name "slot"))) (.literal 1)) (.name "value"), .return_ (.name "value")], span := ⟨1791, 1941⟩ }, { name := "tableSlotValSet", parameters := ["base", "slot", "value"], body := [.store (.binary .add (.binary .add (.name "base") (.binary .mul (.name "TABLE_STRIDE") (.name "slot"))) (.literal 1)) (.name "value"), .return_ (.name "value")], span := ⟨2206, 2310⟩ }, { name := "tableInitGo", parameters := ["base", "cap", "i"], body := [.if_ (.binary .ge (.name "i") (.name "cap")) [.return_ (.literal 0)] (none), .expression (.call "tableSlotSet" [.name "base", .name "i", .name "TABLE_EMPTY", .literal 0]), .return_ (.call "tableInitGo" [.name "base", .name "cap", .binary .add (.name "i") (.literal 1)])], span := ⟨2568, 2706⟩ }, { name := "tableInit", parameters := ["base", "cap"], body := [.if_ (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0)) [.return_ (.name "TABLE_EMPTY")] (none), .return_ (.call "tableInitGo" [.name "base", .name "cap", .literal 0])], span := ⟨2708, 2823⟩ }, { name := "tableCapacityOk", parameters := ["cap"], body := [.if_ (.binary .le (.name "cap") (.literal 0)) [.return_ (.literal 0)] (none), .return_ (.conditional (.literal 1) (.binary .eq (.binary .bitAnd (.name "cap") (.binary .sub (.name "cap") (.literal 1))) (.literal 0)) (.literal 0))], span := ⟨2878, 2977⟩ }, { name := "tableFindSlotGo", parameters := ["base", "cap", "mask", "key", "slot", "step"], body := [.if_ (.binary .ge (.name "step") (.name "cap")) [.return_ (.binary .sub (.literal 0) (.literal 1))] (none), .bind "currSlot" (.binary .bitAnd (.binary .add (.name "slot") (.name "step")) (.name "mask")), .bind "slotKey" (.call "tableSlotKey" [.name "base", .name "currSlot"]), .if_ (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY")) [.return_ (.name "currSlot")] (none), .if_ (.binary .eq (.name "slotKey") (.name "key")) [.return_ (.name "currSlot")] (none), .return_ (.call "tableFindSlotGo" [.name "base", .name "cap", .name "mask", .name "key", .name "slot", .binary .add (.name "step") (.literal 1)])], span := ⟨3267, 3588⟩ }, { name := "tableLookupSlotGo", parameters := ["base", "cap", "mask", "key", "slot", "step"], body := [.if_ (.binary .ge (.name "step") (.name "cap")) [.return_ (.binary .sub (.literal 0) (.literal 1))] (none), .bind "currSlot" (.binary .bitAnd (.binary .add (.name "slot") (.name "step")) (.name "mask")), .bind "slotKey" (.call "tableSlotKey" [.name "base", .name "currSlot"]), .if_ (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY")) [.return_ (.binary .sub (.literal 0) (.literal 1))] (none), .if_ (.binary .eq (.name "slotKey") (.name "key")) [.return_ (.name "currSlot")] (none), .return_ (.call "tableLookupSlotGo" [.name "base", .name "cap", .name "mask", .name "key", .name "slot", .binary .add (.name "step") (.literal 1)])], span := ⟨3872, 4194⟩ }, { name := "tableFind", parameters := ["base", "cap", "key"], body := [.if_ (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0)) [.return_ (.name "TABLE_EMPTY")] (none), .if_ (.binary .eq (.name "key") (.name "TABLE_EMPTY")) [.return_ (.name "TABLE_EMPTY")] (none), .bind "mask" (.binary .sub (.name "cap") (.literal 1)), .bind "initSlot" (.binary .bitAnd (.call "tableSpread" [.name "key"]) (.name "mask")), .return_ (.call "tableLookupSlotGo" [.name "base", .name "cap", .name "mask", .name "key", .name "initSlot", .literal 0])], span := ⟨4252, 4505⟩ }, { name := "tableGet", parameters := ["base", "cap", "key"], body := [.bind "slot" (.call "tableFind" [.name "base", .name "cap", .name "key"]), .if_ (.binary .lt (.name "slot") (.literal 0)) [.return_ (.name "TABLE_EMPTY")] (none), .return_ (.call "tableSlotVal" [.name "base", .name "slot"])], span := ⟨4563, 4700⟩ }, { name := "tableInsertWith", parameters := ["base", "cap", "key", "value", "merge"], body := [.if_ (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0)) [.return_ (.name "TABLE_EMPTY")] (none), .if_ (.binary .eq (.name "key") (.name "TABLE_EMPTY")) [.return_ (.name "TABLE_EMPTY")] (none), .if_ (.binary .bitOr (.binary .lt (.name "merge") (.name "MERGE_SET")) (.binary .gt (.name "merge") (.name "MERGE_ACC"))) [.return_ (.name "TABLE_EMPTY")] (none), .bind "mask" (.binary .sub (.name "cap") (.literal 1)), .bind "initSlot" (.binary .bitAnd (.call "tableSpread" [.name "key"]) (.name "mask")), .bind "slot" (.call "tableFindSlotGo" [.name "base", .name "cap", .name "mask", .name "key", .name "initSlot", .literal 0]), .if_ (.binary .lt (.name "slot") (.literal 0)) [.return_ (.binary .sub (.literal 0) (.literal 1))] (none), .bind "oldKey" (.call "tableSlotKey" [.name "base", .name "slot"]), .bind "oldVal" (.call "tableSlotVal" [.name "base", .name "slot"]), .if_ (.binary .eq (.name "oldKey") (.name "TABLE_EMPTY")) [.bind "newVal" (.conditional (.name "TABLE_PRESENT") (.binary .eq (.name "merge") (.name "MERGE_SET")) (.name "value")), .expression (.call "tableSlotSet" [.name "base", .name "slot", .name "key", .name "newVal"]), .return_ (.name "newVal")] (none), .bind "newVal" (.conditional (.name "TABLE_PRESENT") (.binary .eq (.name "merge") (.name "MERGE_SET")) (.name "value")), .if_ (.binary .eq (.name "merge") (.name "MERGE_ACC")) [.bind "newVal" (.binary .add (.name "oldVal") (.name "value"))] (none), .return_ (.call "tableSlotValSet" [.name "base", .name "slot", .name "newVal"])], span := ⟨4763, 5526⟩ }]

def tableBytes : ByteArray :=
  Slang.Source.encodeBytes Slang.Generated.TableDerivation.tableSource

/-- The table source consumes all 7171 bytes. -/
def tableConsumed : Nat := 7171

theorem tableBytes_size : tableBytes.size = 7171 := by
  rw [show tableBytes = Slang.Source.encodeBytes Slang.Generated.Table.tableSource from rfl]
  rw [ByteRep.size_eq_length (ByteRep.encodeBytes Slang.Generated.Table.tableSource)]
  exact Slang.Generated.Table.tableSourceLength

def tableDecoded : CertifiedDecoded :=
  ⟨{ program :=
       { constants := tableConstants, functions := tableFunctions,
         sourceBytes := tableBytes.size, consumedBytes := tableConsumed },
     certificate := ⟨true, true, true, true, true⟩ },
   ⟨rfl, rfl, rfl, rfl, rfl⟩⟩

theorem tableConsumed_eq : (tableConsumed != tableBytes.size) = false := by
  rw [tableBytes_size]
  rfl

theorem tableSpans :
    SpansValidDerivation tableBytes.size
      (tableConstants.map (·.span) ++ tableFunctions.map (·.span)) := by
  rw [tableBytes_size]
  apply SpansValidDerivation.intro <;> decide

theorem tableDerivation :
    DecodeTokensDerivation tableBytes tableT0 tableConsumed tableDecoded := by
  refine DecodeTokensDerivation.assemble (bytes := tableBytes) (tokens := tableT0)
    (consumed := tableConsumed) (decoded := tableDecoded)
    ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · exact tableConsumed_eq
  · exact table_delims
  · rw [tableT0_length]
    exact table_scanselected_1073
  · decide
  · decide
  · exact DistinctDerivation.of_nodup (by decide)
  · exact DistinctDerivation.of_nodup (by decide)
  · exact tableSpans
  · exact CallsResolveDerivation.of_bool (by decide)
  · exact NamesResolveDerivation.of_bool (by decide)
  · rfl

/-- The table program's token decoder accepts, with the decoded program. -/
theorem table_decodeTokens :
    Decode.decodeTokens tableBytes tableT0 tableConsumed = .ok tableDecoded :=
  DecodeTokensDerivation.sound tableDerivation
end Slang.Generated.SourceWitness.Decoder
