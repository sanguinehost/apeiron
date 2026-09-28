# Lean toolchain mixing in the shared corpus cache

A local check invoked `lake -d proof/corpus build Growable.Scan Native.LocatorSpec` from the
repository root. The root has no `lean-toolchain`, so elan selected
`leanprover--lean4-nightly---nightly-2026-04-13` instead of the packages' pinned
`leanprover/lean4:v4.31.0`, and Lake began rebuilding dependency modules (`target/aeneas-model/lean`,
Mathlib, Batteries, Plausible) with the nightly toolchain. Log:
`/tmp/opencode/scan-locator-check.log` — nightly trace lines, `Aeneas/Std/WP.lean:56` `rfl` failure,
`Mathlib/Tactic/Linter/FlexibleLinter.lean:315` and `Batteries/Tactic/*` object-file failures.

Recovery: the same targets were rebuilt with the correct form

```sh
cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 \
  bash -c 'LEAN_NUM_THREADS=1 lake build Growable.Scan Native.LocatorSpec'
```

Exit 1 at `Native/LocatorSpec` only, which is the known in-progress `held_address_joint_witness`
goal (`Native/LocatorSpec.lean:1175`, `sorryAx`); `Growable.Scan` rebuilt green. The recovery log
`/tmp/opencode/scan-locator-recover.log` names `leanprover--lean4---v4.31.0` in its trace lines.

Rule: run Lean as `cd <package> && lake build …`; never `lake -d <package>` from a directory
without that package's `lean-toolchain`. A trace line naming a toolchain other than the pinned one
is a failed invocation.
