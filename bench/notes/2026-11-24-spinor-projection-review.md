# The spinor/TTT projection: two reviews, both refused, and the retraction

Criterion: **Q3** (campaign cost) and the fold/TTT line. Grade: `measured` for the fit split and the
spectrum this note reuses; `model`/`anchor` for every projection the reviews refused. Result: the
projection is **refused in full** — no measured instance of it exists, the arithmetic behind the one
speed figure is a 1,000× unit slip, and two of the claims are refused by the in-tree Lean warrant
blocks that were supposed to license them.

Reviewers: `agy --dangerously-skip-permissions --model gemini-4-hydrogen-xhigh` (primary, `-xhigh`)
and `gemini-4-neon-xhigh` (second opinion), same prompt, same file list, run independently. Both
returned **REFUSED** on all seven questions. A review is critique, not evidence: every number below
resolves to a file in this repository, and the reviews' own arithmetic was re-derived here before it
was accepted.

Prompt and outputs: `/tmp/opencode/spinor-review.md`, `/tmp/opencode/spinor-review.out`,
`/tmp/opencode/spinor-review-neon.out`.

## The retraction

The claim was: block-CG over `H = ZᵀZ + stab·I` replaces `spd_inverse`, giving `inv` 256 s → 73 s, a
3.5× that moves the 4,587.4 s layer to 1.25×. It is withdrawn. Three errors, two of which cancelled:

1. **A 1,000× unit slip.** One block-CG matvec is `r²k = 1.1356e11` MAC. At `31e12` MAC/s that is
   `3.663e-3 s`. The note divided by `31e9` and wrote 3.66 s.
2. **`T = 400` was dropped.** The per-side `inv` time of 256.25 s is 400 `spd_inverse` calls of
   `2r³/3 = 1.873e10` MAC, so the measured rate is `7.492e12 / 256.25 = 29.24 GMAC/s`, not 73
   MMAC/s.
3. **The two slips cancelled** into `20 × 3.66 = 73.2` and `256 / 73 = 3.5×`. Neither factor is real.

The measured comparison runs the other way. `cb2` is the identical `r × r × k` product at the same
shape and carrier (`Rᵀ·H⁻¹`, `12288×4096`, `r = 3040`), and it costs 57.1 s two-side over 400
iterations: **1.59 TMAC/s**, 19.5× below the modelled `31 TMAC/s`. Twenty CG matvecs on that carrier
is 1,142 s two-side against `inv + cb2 = 569.6 s`, a **2.00× regression**. The 3.5× required a
carrier change (BF16) that has never run, and on that carrier direct Cholesky + `trsm`
(`r³/3 + r²k`, 1.08 CG matvecs) beats 20-iteration CG by about 18.5×.

The iteration count is also unmeasured and bounded away from 20. On `Qwen3-8B l06 down` the measured
spectrum of `H` at `stab = 0.42` is `0.35 … 2810`, `κ ≈ 8,029`
(`src/learn/nanoquant.rs:50-51`); at `t = 0` with `stab = 0.03` the same `λ_max` gives `κ ≈ 9.4e4`.
`((√κ−1)/(√κ+1))^20` is 0.640 and 0.878 at those two points, and reaching `1e-5` needs about 547 and
1,868 iterations. Diagonal preconditioning does not help: `diag(H) = (1 + stab)·I` is already a
scalar, so Jacobi is the identity.

## The layer arithmetic, corrected

`inv` is 934.0 s per side across the seven linears, **20.4%** of the 4,587.4 s layer; the 51.9%
figure is its share of `gate`'s 493.4 s loop, a different denominator. On the layer:

| change | layer | factor |
|---|---:|---:|
| `inv` at the withdrawn 3.5× | 4,587.4 → 3,920.3 s | 1.170× |
| `inv → 0` (the hard ceiling) | 4,587.4 → 3,653.4 s | 1.256× |
| all of ADMM → 0 | — | 1.658× |

The dominant term is `tune` at 2,662.4 s, **58.0%**, and this design does not touch it. Under
`AGENTS.md` § *The dominant term* the target is refused whatever its local ratio, and the word
"revolution" is refused with it: 1.170× on the layer is not a revolution, and `inv → 0` is 1.256×.

## The two claims refused by their own sources

`proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean:139-142` already says of
`rotor_adjoint_forward`: it is `conj(e^{iθ}) = (e^{iθ})⁻¹` on `ℂ`, one `Complex.exp_conj` rewrite,
and "the backward pass's operator is available at every forward step … no reverse sweep stores it"
**has no backward pass, gradient or sweep in any type**. The claim is marked `Anchor` in the file
that was cited to license it.

`SpinorDoubleCover.lean:73-84` says of `fullTurn_invisible` / `fullTurn_ne_one` /
`doubleTurn_closes`: the theorems are centrality, distinctness and order two of `-1` in
`Quaternion ℝ`; no unit group, no rotation group and no covering map is defined, and conjugation is
trivial for every nonzero central element — by `2` as much as by `-1`. The file is not in this
repository, so the citation is a pointer at best.

Neither statement maps to `W = U⁻¹B` over a real SPD `U`, to the Sherman–Morrison update
`U⁻¹ ← U⁻¹ − w(U⁻¹x)(xᵀU⁻¹)/(1 + w·xᵀU⁻¹x)`, or to a rectangular non-invertible `12288×4096` layer.
What would be needed: a projection from the network onto a unitary carrier with a proved logit
distortion bound, a theorem that two forward passes recover the exact VJP at `O(1)` passes in the
parameter count, and for the ledger a multi-step rounding-drift bound — none of which exists.

## The TTT claim, with the arms that test it

`(a4) CARRY+TTL` equals `(a1) CARRY` at 11/12 recall, 3/3 out-of-window, 5.7% self-repetition, at
78.6 against 158.9 tok/s — and both carry the full growing KV, so the fold has nothing to add in
that window. Where the arm is run without the full KV it loses:

| arm | fold | recall | tok/s |
|---|---|---|---:|
| `(j) CARRY+COLD` | off | 7/12 | 89.7 |
| `(m) CARRY+COLD+TTL` | on, 978 observations | 6/12 | 26.1 |
| `(j)` 24-turn, true decode | off | 10/11 | 124.7 |
| `(m)` 24-turn, true decode | on | 10/11 | 31.9 |
| `(h)` / `(i)` superposition read | — | 2/12, 1/12 | — |

`[SETTLE]`, the nearest thing to a demonstration, is one planted fact over 604 tokens on
Qwen2.5-0.5B-Instruct with `n = 602 < d = 896` — inside capacity. The measured sweep on real text
closes the lane: `[GATE-RIDGE]` scored 44 candidate gates over 1,499 `c4_en` boundaries and **none
reaches ≤ 0**; the oracle ceiling, a gate that already knows the answer, is **0.0915 bits/token**;
and the failure is the one `RidgeFiltration.over_capacity_crosstalk` predicts, with the null space
70–80% of what was absorbed at `d = 896`, `n ∈ [3000, 4499]`.

On the 8B at 200k the fold is 48.8× over the readout's rank, and a `SessionRidge` session needs
`xtx` 67.1 MiB + `u_inv` 134.2 MiB + a dense `4096 × 151,936` `f32` `b` at 2.318 GiB —
**2.515 GiB (2.70 GB) per session**, over both C1 (1 GiB) and M1 (2 GiB) before any model weight,
and its per-token read of `u_inv + b` is 2.62 GB, capping decode at 8.5 tok/s on its own. `(a4)`
covers none of that.

## What is left

Nothing in this design is wired, and `cg_solve`, `cgls`, `RidgeStream`, `RidgeFold` and
`SessionRidge` have zero occurrences in `apeiron-nq`. A `cg_solve` beside `spd_inverse` would be a
second solve path; a `RidgeStream` port would be dead code, because `serve/src/main.rs:345-353`
allocates a fresh `DecodeState` per request and `Engine::next_token_state` returns
`Result<u32, StepError>` with no hidden-state or logit hook. `ColdStore` already sits unwired in
`src/engine/step/cold_store.rs`; it is not precedent.

Two things from the exercise do survive, both as documents rather than code: the KV carrier
correction in `bench/cpu/1795465460-nq8b-cost-model.md`, and the paper reading in
`bench/notes/2026-11-24-ttt-is-linear-attention.md`, which refuses a covariance tier on capacity
per byte before it was written.

## What this does not cover

Neither review measured anything; their role was to locate the artifact or refuse, and this note
re-derives only the arithmetic. The fit split is one layer, once, on one host
(`2026-11-23-admm-split.md`:124-130). No CG iteration count on this system exists in any of the
three trees, so the correct statement is not "CG is slower" but "CG is slower at the measured
carrier and unknown at a BF16 carrier nobody has run". Layers 1–35, the 200k path, and C2 are
untouched.
