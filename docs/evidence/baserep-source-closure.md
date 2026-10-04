# BASEREP-SOURCE-CLOSURE — receipt

Row: `APEIRON-BASEREP-SOURCE-CLOSURE` (`apeiron-baserep-source-HIZgmcRqYLTWZCbsLfNwqU`).

`Growable/InsertRefine.lean`, `Growable/Schedule.lean`, `Growable/FaceBound.lean` and
`Growable/GaplessRead.lean` existed on disk with **no git history on any branch or tag**
(`git log --all` empty for each) while tracked sources import them:

- `Growable/BuildRefine.lean:3-4` -> InsertRefine, Schedule
- `Growable/HomeBound.lean:1` -> GaplessRead
- `Growable/GaplessRead.lean:2` -> InsertRefine

Committed as a **source-integrity checkpoint**, not a verification success (Sol
`BASE-CORPUS-SUPERSEDE` verdict). Compilation status recorded explicitly: **not independently
verified in this session.** `Growable.BuildRefine` and `Growable.HomeBound` are **not** roots of
`ApeironWordBridge` (`proof/corpus/lakefile.lean:26-32`), and no root imports them, so these four
modules are currently off the default build path — the default target is unaffected by their
presence. They are imported only by other non-root modules.

Gate (to run at the barrier, not claimed here): in a snapshot containing only declared sources,
`lake build AxiomAudit` succeeds and existing audited `Apeiron.GrowableReadLoop.*` headlines are
preserved. Control: omit `Growable/GaplessRead.lean` -> the `HomeBound` import fails (source
closure, not semantics).

Grade: `read` (provenance and import graph). No `proved`, no `measured`.
