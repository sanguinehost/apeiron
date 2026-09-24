# Intent — canonical elliptic Kepler solve

- **Task:** `APEIRON-KEPLER-W0-T1`; owner role: Sol (contract); supplied-checkout documentation exception, no worktree or build lane.
- **Identity:** Apeiron HEAD `262bb0a94eff2cc7d5655122a954f44d74de96b6`, dirty shared checkout; canonical corpus pin remains `3d66ea3aa2fe1de5997b6b757247454a2f771358`. HEAD alone does not identify the dirty sources.
- **Status:** contract proposal (`model/anchor`), not implementation authorization. See [spec](../spec/kepler-canonical.md) and [plan](../plan/kepler-canonical.md).

## Outcome and path

Penglai's sun–moon–planet orbit calculation MUST obtain a bounded eccentric anomaly for seasons and tides through its own time/mean-anomaly adapter → Apeiron's reusable `solve_kepler` in proposed `src/kepler.rs` → a revision-pinned certificate interpretation in `proof/corpus/Kepler/Certificate.lean` → the canonical real equation and enclosure in proposed `sanguine-apeiron/proof/Apeiron/Physics/Kepler/`. This is a **target path**, not a call-path claim: `src/kepler.rs`, the bridge, and the Penglai call site do not yet exist for this epic. Penglai owns epoch, turn count, units, elements, persistence and gameplay application; Apeiron owns input refusal and the numerical result. A call-site inventory and a gameplay gate are prerequisites for dispatching the adapter.

The intended service admits normalized finite binary64 mean anomalies and `0 ≤ e ≤ 0.95` on explicit `Sigma::Elliptic`, including the circular bit-exact identity. The design targets are `|Ê−E*| ≤ 4 κₑ ε` and certified `|F(Ê)| ≤ 4 ε` (`ε = 2^-52`, `κₑ = 1/(1−e)`), useful-science anomaly error `≤ 10^-9 rad`, at most five correction iterations for `e ≤ 0.45` and eight for `0.45 < e ≤ 0.95`, plus `≤ 10^-12 rad` phase drift over `10^6` orbits with a horizon-stable upstream clock. These are **model/anchor targets**, not measured feasibility or tested service; [spec](../spec/kepler-canonical.md) states their domain and the unresolved slots.

## Falsifier and exclusions

- A sound independent residual oracle finds `|F(Ê)|` above the reported outward bound, or a reported enclosure misses the unique exact solution for the denoted inputs; either falsifies certification. One successful return outside the exact admitted domain or on a non-elliptic face falsifies refusal. A correct cap that returns exhaustion for valid inputs instead of servicing the promised population falsifies the iteration SLA.
- A time series at the registered million-orbit horizon exceeds the drift target, or a runtime path still uses naïve cumulative binary64 phase, falsifies the integration claim. The reported ~`2.5e-5 rad` linear-growth counterexample is supplied context, **not measured in this task**; reproduce with a validated instrument before citing it as evidence.
- No import of `Substrate.StationaryPhase.{SpinorLoop,BiasInjectedRecovery}`, no new face enum, no storage work, no inference of an executable closed form from Banach existence, no claim that an abstract Lean certificate refines Rust `f64`. The elliptic trichotomy link is a scalar projection of `Trichotomy.flow (-1)`, not a claim that affine folding solves a nonlinear equation.

## Completion condition and trust

The epic remains open until every exit in [plan](../plan/kepler-canonical.md) has a baseline/red/restore semantic control and the shared gates pass on the same identities, followed by a separately gated Penglai consumer. `git diff --check` is only this documentation task's whitespace check; it cannot establish any theorem or service. The host Rust numerical path, transcendental library, rounding mode, ABI and time normalization are outside the exact-real proof until individually linked and checked. Physical orbit/season/tide semantics are **model/anchor** interpretations, not statements in the cited Lean declarations.
