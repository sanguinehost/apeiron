# Plan reset: board and plan reconciled to the owner's ladder

Source identity: `main` at `01aff6a` plus this change (`docs/plan.md`, `AGENTS.md`,
`docs/direction.md`, `tools/board/board_core.py`, `tools/board/test_board.py`, this file). Machine:
the owner's workstation (i9-14900KF, rustc 1.97.0-nightly).

## Gates run on the live tree

| gate | exit | output read |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | no diff |
| `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | 0 | no warnings |
| `cargo test --workspace --all-features --locked` | 0 | 26 test binaries, 714 passed, 0 failed |
| `python3 -m unittest discover -s tools/board -p 'test_*.py'` | 0 | 108 tests |
| the same suite with `"test"`, `"git"` or `"gh"` removed from `GATE_HEADS` in a copy | 1 each | one failure each: `test_shell_test_is_a_gate`, `test_git_and_gh_invocations_are_gates` |
| `python3 tools/check-claims.py` | 0 | |
| `python3 tools/rooting/read_gate.py --check` | 0 | |
| `python3 tools/rooting/doc_reach.py --check` | 0 | 105 removals, clean |

The three cargo gates ran under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`
with `timeout 570`.

## Board rows closed against commits

Each row below was marked Done on the board; the commit is on local `main` (ahead of `origin`).

| board row | commit | what landed |
|---|---|---|
| `p1-audit-rows-61-audit-cQNd0kLME75aiSVIlDWYv` | `01aff6a` | `lake build AxiomAudit` exit 0, 9089 jobs, zero `sorryAx` (handover measurement, not re-run here) |
| `p0-readloop-green-the-so-HieItT7wKt4SKVXrMBgWgF` | `6fd1fc5`, `d6197bc`, `01aff6a` | `Growable/ReadLoop` and `Growable/Read` build; the audit evaluates |
| `apeiron-vv-v1-c-the-entry-BhpTnZsIcBnlxn1BVXRCcI` | `b0191c8` | the tag conjunct is discharged under the named premise `SpillCoordPayloadSync` |
| `licence-apeiron-the-crate-BwaMypyGZix2X9m2B2HAYm` | `967ea28` | Apache-2.0 `LICENSE`, `NOTICE`, five manifests |
| `apeiron-board-reconcile-s-CgdnzwtZvdUJxBWVAZ8l54` | `ae097d4` | the board driver's four defects |
| `apeiron-map-parity-wire-EhXlkMtFhAOPAkVDTSGtyQ` | `b519047` | the map arm executes the vendored table ([receipt](map-parity-landed.md)) |
| `apeiron-clippy-rest-clear-BXBcoyaB24BO8jlBMXDO8l` | `01aff6a` | clippy exit 0 above |
| `apeiron-path-wire-clippy-FK7UT8NOLGOCzTTFOKOe6n` | `01aff6a` | clippy exit 0 above; the vendored feature gates carry the `ht-` prefix |
| `apeiron-board-gatehead-GRf8pkvLC4wbhMkFuiTGaf` | this change | `test`, `git`, `gh` gate heads and their controls above |

## Not covered

The Lean corpus was not rebuilt for this change; no source under `src/` or `proof/` changed. The
`AxiomAudit` figure is the handover's measurement at `01aff6a`. Cancelled rows carry their successor
in a board comment; comments are not visible to `board.py --live`.
