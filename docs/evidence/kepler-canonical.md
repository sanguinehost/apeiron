# Evidence: canonical elliptic Kepler solve (APEIRON-KEPLER-W7-T1)

Integration record for the W7 barrier on the Apeiron working tree. Every claim
carries one evidence label from `AGENTS.md`: tested, measured, proved or
model/anchor. This record runs no build and claims no soundness of its own; the
barrier stage `tools/check-kepler.py --stage barrier` validates the identities,
digests and receipts below against the current tree.

## Identity

- Apeiron working-tree base HEAD: `262bb0a94eff2cc7d5655122a954f44d74de96b6`
  (tested: `git rev-parse HEAD` in `/home/socol/Workspace/apeiron`).
- Resolved corpus revision: `880aa57e47e782ed7a9d9ee1974cfc87360b446b`
  (tested: read from the bridge pin files and the sibling corpus checkout).
- The bridge pin `proof/corpus/lakefile.lean` and
  `proof/corpus/lake-manifest.json` both record the resolved revision
  `880aa57e47e782ed7a9d9ee1974cfc87360b446b` (tested: `--stage pin` clean).

## Source digests (SHA-256)

Freshness is established by matching each recorded digest against the actual
working-tree file; a stale or missing digest is rejected by the barrier stage.

| source | sha256 |
|---|---|
| `src/kepler.rs` | `8414acadbe999b3d0b81fddfeaa94060c0158ff6f13e97e2a77ef8a624a5b0b6` |
| `tests/kepler.rs` | `3fe2381485b37d709260f58db87503e1f6d6b6c2f89adbb6596528c38edb558c` |
| `proof/corpus/Kepler/Certificate.lean` | `01ce4f612a22f52f1d44bc26fdba8ae3fb908ad1ea5ea5e6c8d1f9e7fa40373b` |
| `proof/corpus/AxiomAudit.lean` | `37f5fde62a6b6fcacd821b51f6baa87b833e541cf80a395a84249244d4de8d5a` |

(testing method: `sha256sum` on the four files above; the `--stage barrier`
stage recomputes the same digests and rejects a mismatch. Re-pinned 2026-09-24
~16:40 AEST after a co-owner's additive `AxiomAudit.lean` edit; the bounded
`lake build Kepler.Certificate AxiomAudit` re-run reported 3693 jobs with the
same closed cone. The shared-receipt row below retains the W5/W7 receipt.)

## Shared gate receipts

Corpus gates run under `sanguine-apeiron/proof`; bridge gates under
`apeiron/proof/corpus`.

| gate | command | result |
|---|---|---|
| corpus build | `cd proof && lake build` | 3526 jobs (tested) |
| corpus lints | `python3 scripts/check-lints.py` | clean (tested) |
| corpus style | `python3 scripts/check-style.py` | clean (tested) |
| corpus imports | `python3 scripts/check-imports.py` | clean (tested) |
| corpus checker | `python3 scripts/check-kepler.py --check` | clean (tested) |
| corpus self-test | `python3 scripts/check-kepler.py --self-test` | 16/16 (tested) |
| bridge build | `cd proof/corpus && lake build Kepler.Certificate AxiomAudit` | 3692 jobs (tested) |
| bridge cone | `#audit` over the certificate headlines | `{propext, Classical.choice, Quot.sound}` (proved) |

The bridge cone is the closed axiom cone for the audited declarations, not a
refinement of the Rust kernel (proved). No source refinement is claimed here.

## Stage outputs and control matrices

- Stage `pin`: `python3 tools/check-kepler.py --stage pin` clean; controls
  `--stage pin --controls` baseline/red/restore 0/1/0 x3 (tested).
- Stage `bridge`: `python3 tools/check-kepler.py --stage bridge` clean; controls
  `--stage bridge --controls` baseline/red/restore 0/1/0 x6 (tested).
- Stage `runtime`: `python3 tools/check-kepler.py --stage runtime` clean; controls
  `--stage runtime --controls` baseline/red/restore 0/1/0 x3 (tested).
- Stage `barrier`: controls `--stage barrier --controls` baseline/red/restore
  0/1/0 x3 (tested); the same stage is the acceptance gate for this record.

## Runtime population and SLA

- Sampled population: 12 bands x 1039 rows, 12468 solves (measured).
- Refusals under the strict sound-bound-above-ceiling `CertificateFailure`
  policy: 2 refusals at e=0.95 (measured).
- SLA row: OPEN; universal successful service needs the analytic and
  complete-domain argument (model/anchor). A capped loop is not service.
- Golden oracle: 42 rows, max `dist/(kappa eps) = 2.365`, all `dist <= r`
  (measured; this is a sampled comparison, not a proof).
- W6-T1-FIX (lead finding): `1.0 - e` is not an exact subtraction for
  e < 0.5; the fix is an exact `two_sum` split plus a conservative factor and a
  subnormal guard (tested). An independent exact-rational scan over 1,135,679
  `(e, d)` pairs found 0 violations (measured).

The population, refusal count, endpoint and SLA row above are receipts: the
barrier stage rejects the record when any of them is dropped.

## External trust inventory

- IEEE binary64 round-to-nearest (`RN`) arithmetic and `mul_add` as one
  correctly rounded `a*b+c` are external (hardware/compiler) (model/anchor).
- ASSUMPTION-SIN: the platform `f64::sin` is within 1 ulp for `|x| <= 8`; the
  guard refuses inputs outside that domain (model/anchor). This is an assumed
  platform property, not a proved one.
- `libm`, `compiler` and `hardware` are external components (tested: they are
  not inside this repository's proof boundary).
- Aeneas: no Aeneas `f64` source refinement exists for this path (non-claim).
- The bridge certificate is a conditional Lean theorem on a supplied sound
  bound, not an executable `f64` refinement of the runtime kernel (proved).

## Shared document integration notes (BLOCKED)

- `docs/flash-handoff.md` and `docs/kernel-boundary.md` have unreleased owners;
  their Kepler integration notes are BLOCKED and were not edited (tested: the
  W7-T1 change set owns only this new evidence file).

## Scope and non-claims

- No Aeneas `f64` refinement of `solve_kepler` is claimed or established here.
- The runtime refusals are the strict-policy sample above; the SLA row stays
  OPEN, so no universal service claim is made.
- The bridge theorem is conditional on a supplied sound bound; it is not
  reported as an executable `f64` result.
