# MAP-HASH-CONSTANT — receipt

Row: `APEIRON-MAP-HASH-CONSTANT` (`apeiron-map-hash-constant-DhKHlIfJUaN4sncj0LA3E7`).

`Region::home` mixes by `HOME_SPREAD = 0x517cc1b727220a95` (fxhash 0.2.1 `SEED64`, registry sha256
`885b72a1c977dc1c3ff4e0652719e62b6d8c3e713403e571afd581283f72df7b`) instead of `TAG_SPREAD =
crate::coord::PHI`. Shift/mask structure unchanged; `TAG_SPREAD` keeps the tag fingerprint.

**Paired evidence (the lane's, two arms in one campaign, each with its own forced rebuild):**
base `PHI` 1.7650 (1.7451-1.7894, sd 0.0165) vs `HOME_SPREAD` 1.7302 (1.7240-1.7431, sd 0.0087) —
per-pair ratio 0.9804 = **-1.96%**, 5/5 pairs same direction, arm ranges disjoint.

**Live-tree confirmation (five paired full-profile runs on this tree, contended, averaged):**
`hashtrinity_table` means 1.7561 / 1.7969 / 1.7237 / 1.7902 / 1.8010, **mean 1.7736, range 0.077**.
That is **inside the cross-run spread and does not confirm the -1.96%**. The paired campaign controls
the box; the live single-arm run does not. **The movement is not confirmed on the live tree.**

Adversarial control (M-M2's, carried): clearing each coordinate's low `region_bits` drives the **low**
window onto the pile-up floor (64.5x at bound 256, 1024.5x at 4096) while the landed top-window arm
does **not** invert. Two source mutations redden the gate and restore digest-identical.

Grade: `tested` (gates + the adversarial control). The scenario ratio is the campaign's number and is
**not confirmed** by the live run. `proof/regen.sh --check` will drift (`table.HOME_SPREAD` appears).
