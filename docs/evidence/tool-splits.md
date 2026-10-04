# Tool splits under the 1500-line bound — receipt

Row: `APEIRON-SPLIT-TOOLS` (`apeiron-split-tools-files-FYgQ7JVXim8eOWNXge8qxg`). Sibling rows:
`APEIRON-SPLIT-GATE` (`apeiron-split-gate-tools-BX6YWj0AN71fddeI5FQlMN`), `APEIRON-SPLIT-KEPLER`
(`apeiron-split-kepler-py-B9HQTNsxvtDpkXxw3QsY0R`).

Owner direction: break the monolithic tools into files under 1500 lines. Pure behaviour-preserving
move-and-import; no logic, message or exit code changed.

| monolith | before | after | live tree? |
|---|---|---|---|
| `tools/hooks/lane.py` | 2324 | 229 + 4 modules (max 770) | yes |
| `tools/board/board.py` | 2290 | 486 + 6 modules (max 741) | yes |
| `tools/gate/main.rs` | 2648 | 64 + 8 modules (max 1149) | yes |
| `tools/check-kepler.py` | 3686 | 398 + 9 modules (max 859) | **worktree only, not installed** |

Gates: lane `self-test` **38/38**; board `check --controls` **14/14**; `tools/gate` conformance
**byte-identical** over 18 invocations; `check-kepler.py` **17 CLI variants** byte-identical
(concatenated capture sha256 `0764fbd7…`). Controls: seeded defects fire identically, one per new
module where the lane built them. `tools/gate`'s `conformance.json`/`.md` are the frozen data record
the driver reads by a fixed path, not source — left whole deliberately.

Declared path-constant moves: `tools/board/checkpoint.py` `build_control_repo()` gained the 10 new
module paths (+10 lines, path data only); `tools/board/test_board.py` re-points two controls at
`board_check`/`board_core`. Both needed or the control suite breaks.

`docs/contracts/artifacts.toml` records `037dbd65…` for `tools/gate`; pre-split was `e6cb76b8…` and
the split tree is `dfd61f1c…` — the row was **drifted before this work**, and is now further stale.
That drift is the DIGEST-GATE lane's job.

Grade: `tested` (gates + controls at their own scope). No `proved`. No logic changed.
