# The exact-copy plan

Status: plan only, no code written. Supersedes `PLAN.md` W0–W5 as they concern NanoQuant, and
`/tmp/opencode/campaign/review-final.md` §3–§4 as a remediation list. Criterion: `GOAL.md` Q3.

The old path is not repaired. `src/learn/nanoquant.rs` is a re-derivation (`fro2`, `spd_inverse`,
`admm_loss`, `export`, `tune`, `tune_sketch`) whose function boundaries do not exist in the
reference. It is deleted on the first commit of the work branch, not adapted.

## 0 · What "exactly" has to mean

Three references exist, and they are not the same artifact:

| ref | artifact | status |
|---|---|---|
| **R1** | `SamsungLabs/NanoQuant` @ `a9e0a430881ff80d83b622c3129e330dc33c04f5`, `/home/lucas/Workspace/NanoQuant` | 2,700 lines of Python, readable, runnable |
| **R2** | the resolved config of the released Qwen3-8B run, `quality-8b.json` → `experiment.resolved_config` | a JSON spec, no code |
| **R3** | arXiv 2602.06694 | describes LB-ADMM; does not describe R2's extra stages |

R1's pipeline is five stages: calibrate (`importance.collect_stats`) → uniform rank from the bit
budget (`utils.calculate_ranks`) → per layer `tune_nonfact` → `factorize_admm_nanoquant` →
`tune_fact` → after all blocks, `compress_model_recon` KD.

R2 adds, and R1 does not contain:

| key in `resolved_config` | value | in R1? |
|---|---|---|
| `factorization.scale_fit` | 2 alternating passes, rollback, ε 1e-8 | **no** |
| `factorization.shared_input` | q/k/v fused group `self_attn.attn_qkv` | **no** |
| `outliers` | 0.001, selector `residual`, 80-iteration probe, bf16 | **no** |
| `allocation` | `reconstruction_aware`, response curves, floor 0.6×/ceiling 1.4×, q/k/v/o ×1.25, down ×1.5 | **no** |
| `block_tuning.post_block_refit` | 2 epochs, scale lr 1e-5 | **no** |
| `distillation.loss` | `top_k`, top-64, teacher targets artifact | **no** (R1 is full-vocab KL) |
| `calibration.method` | `online_fisher`, 256 samples, shrinkage 0.6 | **no** (`online`, 128, 0.4) |
| `packing.embeddings.bits` | 8 | **no** (R1 has no embedding quantiser) |

`grep -rn 'scale_fit\|shared_input\|post_block_refit\|resolved_config'` over R1 returns nothing.
R2 was produced by a tree that is not public: `quality-8b.json` names `/workspace/NanoQuant/Results/029`,
`evidence/029`, `Docs/ImprovementSuggestions/…`, and an experiment number. R3 has no section for
`scale_fit`, `shared_input`, `post_block_refit` or the response curves.

**Consequence.** The 13.165 row cannot be reproduced by copying R1, because three of its stages are
absent from it. Two targets, run in this order:

- **T1 — copy R1 exactly.** Executable, dumpable, verifiable. Its score is whatever it is; measure it.
- **T2 — add R2's eight rows**, one at a time, each behind its `resolved_config` key and its own gate.
  These are reimplementations from a config key, not copies; the plan labels them `model` until a
  measurement says otherwise.

Q3 parity is claimed only after T2's last row. T1 alone does not close it and the receipt says so.

**One search before M1.** If the internal tree that produced `Results/029` survives on a retained box
image, it outranks every T2 row: rows 29–36 become copies instead of reimplementations. It is not on
this host (`grep scale_fit|shared_input|post_block_refit` over `~/Workspace` finds only an unrelated
Python variable), and `/workspace` does not exist here. Look for the image before spending the T2
weeks.

## 1 · The parity relation

Bit identity is attainable only where the reference is itself deterministic and the arithmetic order
can be reproduced. It is not attainable through 8 epochs of bf16 SGD on a different BLAS: a 1e-7
gradient difference flips a sign near zero and the factor changes. So the relation is graded by
stage class, not asserted globally:

| class | stages | criterion | why |
|---|---|---|---|
| **D** deterministic, exact | seed, ranks, calibration statistics, argmax/sign, packing, `state_dict` layout, export | equal bytes, or ≤1 bf16 ulp where the reference computes in f32 and stores bf16 | no reduction-order freedom |
| **I** iterative, fixed input | ADMM, `tune_nonfact`, `tune_fact`, KD | injected initial state; per-iterate factor agreement; **sign equality on U/V**, scales ≤1 bf16 ulp | the pack's identity is its signs |
| **M** model-level | the full pipeline | PPL and 0-shot on a shared harness, band stated ex ante (1 % PPL), plus per-block loss trajectory ≤1e-4 relative | the only observable that matters |

Any claim of "bit-exact NanoQuant" beyond class D is false and is not made. The receipt states which
class each result belongs to.

**Determinism preflight (gate G0).** Run R1 twice on the same 0.6B model, T=8, same seed, and hash
every factor. If the hashes differ, class D collapses to class M at that stage and the plan's
acceptance changes before any porting starts. This is measured first, not assumed.

## 2 · Pinning

- R1 stays at `a9e0a43`, clean tree. Instrumentation is external: a driver imports the package and
  wraps `rank1_approx`, `binary_ste`, the optimizer step and `NanoQuantLinear.forward` from the
  outside, dumping tensors. No edit to the reference source, so `git status` there stays empty and
  the oracle is the published code.
- R2's `resolved_config` is copied into `bench/notes/nq-029-config.json` with its sha256 and the
  `quality-8b.json` hash, because that file lives outside the repository.
- One new branch `nq-exact`, one commit that deletes the old fitter (§7). No second path in the
  branch at any point.

## 3 · The ledger

Every part of R1, in call order. `ours` is the new module `src/learn/nq/`; one file per reference
file, one function per reference function, same names.

| # | part | reference | ours | gate | grade at best |
|---|---|---|---|---|---|
| 1 | config + defaults | `modules/hub.py:45-83` | `nq/config.rs` | field-by-field equality with R2 JSON | D |
| 2 | seed | `utils/utils.py:15-30` | `nq/seed.rs` | reproduced `randperm`/`randn` stream | D |
| 3 | decoder/layer discovery | `utils/utils.py:62-114` | `nq/model.rs` | same 7 names, same order | D |
| 4 | rank from bit budget | `utils/utils.py:128-182` | `nq/rank.rs` | exact on `(a,b,bits)` grid, 32-floor, cap | D |
| 5 | calib dataset + loader | `utils/data_utils.py:19-51,206` | `nq/data.rs` | same `np.random.randint` indices, same padding | D |
| 6 | block-0 input cache | `utils/load_utils.py:122-177` | `nq/data.rs` | same tensor hash for 128 inputs | D |
| 7 | online calibration hook | `core/importance.py:80-111` | `nq/calib.rs` | `i_norm`/`o_norm` ≤1 bf16 ulp | D |
| 8 | shrinkage + register | `core/importance.py:181-262` | `nq/calib.rs` | exact | D |
| 9 | power iteration | `core/admm_nq.py:17-36` | `nq/admm.rs` | 5 iters, injected `v0`, ≤1e-6 rel | I |
| 10 | SVID / rank-1 | `core/admm_nq.py:40-59` | `nq/admm.rs` | sign equality | I |
| 11 | solve step | `core/admm_nq.py:63-91` | `nq/admm.rs` | f32 Cholesky + per-iter stabilizer; factor ≤1e-6 | I |
| 12 | ADMM outer loop + export | `core/admm_nq.py:95-252` | `nq/admm.rs` | per-iterate `A_z`,`B_z`; final U/V bit-equal | I |
| 13 | rho schedulers | `core/admm_nq.py:255-287` | `nq/admm.rs` | exact at 400/800 sample points | D |
| 14 | module convert + STE | `modules/linear.py:16-104` | `nq/linear.rs` | forward bit-equal on dumped `x`,`U`,`V`,`s` | D |
| 15 | forward order | `modules/linear.py:248-257` | `nq/linear.rs` | `x*s_pre → V → *s_mid → U → *s_post`, bias last | D |
| 16 | finalize + state dict | `modules/linear.py:264-321` | `nq/linear.rs` | packed bytes equal | D |
| 17 | binary packer | `kernel/utils.py:503` (`pack_binary_gemv`) | `nq/pack.rs` | byte equality on random ±1 | D |
| 18 | weighted MSE | `core/compress_block.py:17-19` | `nq/block.rs` | exact | D |
| 19 | param groups | `core/compress_block.py:22-48` | `nq/block.rs` | same three groups | D |
| 20 | `tune_nonfact` | `core/compress_block.py:51-99` | `nq/block.rs` | loss curve ≤1e-4 rel; per-step update ≤1 ulp f32 | I |
| 21 | `factorize_and_replace` | `core/compress_block.py:102-156` | `nq/block.rs` | transpose rule `out<in`; bias copy | D |
| 22 | `tune_fact` | `core/compress_block.py:159-201` | `nq/block.rs` | as #20; `finalize()` only on the just-tuned layer | I |
| 23 | block loop + two streams | `core/compress_model.py:18-122` | `nq/block.rs` | teacher targets from fp block, inputs from q block | I |
| 24 | per-block PPL | `utils/eval_utils.py:91-105` | `nq/eval.rs` | same token windows | D |
| 25 | KD | `core/compress_model.py:125-286` | `nq/kd.rs` | scale-only params, top-64 for T2, loss curve | I |
| 26 | AdamW | `optimi/adamw.py:47-75` | `nq/opt.rs` | betas (0.9, 0.99), eps 1e-6, decoupled wd | D |
| 27 | compressed state dict | `utils/load_utils.py:216-242` | `nq/pack.rs` | key set and bytes | D |
| 28 | PPL harness | `utils/eval_utils.py:13-89` | `examples/ppl.rs` | same windows, same tokens, same normaliser | D |
| 29 | `scale_fit` | R2 only | `nq/scale_fit.rs` | PPL delta on one layer, 2 passes, rollback | M |
| 30 | `shared_input` qkv | R2 only | `nq/shared.rs` | frees 0.1053 b/w at fixed PPL band | M |
| 31 | residual outliers | R2 only | `nq/outlier.rs` | 0.001 fraction, 80-iteration probe | M |
| 32 | response-curve allocation | R2 only | `nq/alloc.rs` | reproduces the 8B rank per layer from the config | M |
| 33 | `post_block_refit` | R2 only | `nq/block.rs` | 2 epochs, scale lr 1e-5 | M |
| 34 | top-64 distillation | R2 only | `nq/kd.rs` | 8 epochs, 512 tokens/batch | M |
| 35 | 8-bit embeddings | R2 only | `src/format/nq.rs` v3 | rate closes at 1.00 with bf16 head excluded | D |
| 36 | signed `scale_mid` in the pack | R2 only | `src/format/nq.rs` v3 | bf16 signed vector, 16 bits/rank | D |

Rows 1–28 are the copy. Rows 29–36 are T2 and are labelled as reimplementation until measured.

## 4 · Milestones

| id | work | gate | cost |
|---|---|---|---|
| **M0** | determinism + cost preflight on the 4090: R1 twice at T=8 on 0.6B; one 8B layer at T=8 and T=100; `tools/fit-budget.py` projection | G0: factor hashes equal; cost per layer measured | half a day |
| **M1** | external dump driver; ledger rows 1–8 | rank grid exact; `i_norm`/`o_norm` ≤1 ulp on 0.6B | 2 days |
| **M2** | `nq/admm.rs`, rows 9–13, injected init | per-iterate factor match on 3 layers (`q`,`down`,`k`), T=8 and T=400 | 4 days |
| **M3** | `nq/linear.rs`, `nq/pack.rs`, rows 14–17 | forward bit-equal; packed bytes equal | 2 days |
| **M4** | `nq/block.rs` + `nq/opt.rs`, rows 18–23 | one block, 0.6B, 8 samples: loss curve and finalised signs | 1–2 weeks |
| **M5** | `nq/kd.rs`, row 25 | full 0.6B: PPL within the G0 band, 0-shot within 0.5 pt | 1 week |
| **M6** | T2 rows 29–36, in config order | each row's own gate; no row lands without its measurement | 2–4 weeks |
| **M7** | 8B, R2 config, one process per stage | the three Q3 numbers | days of GPU |

M4 is the fork in the road. If a CPU bf16 block forward cannot hold sign equality through 8 epochs,
the tuner runs on the Vulkan device path, and the KDB NaN (`attn_kd_bwd` after `attn_kd` in one
command buffer) becomes the critical blocker rather than a side quest. That is decided by
measurement at M4, not guessed now.

## 5 · Acceptance

A Q3 receipt is one v1 record (`bench/PROTOCOL.md`) with:

1. pack bytes, replaced-weight count, rate to 9 decimals at 1.00 b/w including the 8-bit embedding;
2. PPL at ctx ≥ 2048 on the same skeleton and token stream as the paired arm;
3. six-task 0-shot, `acc_norm`/`acc` per task;
4. the comparison arm: R1 re-run, R2's published row, or our harness on their pack, labelled;
5. the class-D/I/M grade of every stage that fed the number.

The number to beat is 13.165128 at 1.00 b/w (their pack, our harness, `swap-all`) and 12.47
published. Until M7 runs, no sentence says or implies parity.

## 6 · What the copy does not prove

- It does not prove the serving kernel. Parity of the fitter says nothing about `matmat_q8act`; the
  reader's own evidence is `bench/notes/nq-proof-ladder.md` rungs 1–4.
- It does not prove Python semantics in Lean. Aeneas extracts Rust; the reference is PyTorch. The
  copy's correctness criterion is differential testing against R1, and Lean covers the fixed-point
  reader underneath it, not the equality with PyTorch.
- It does not cover rustc/LLVM, the Vulkan driver, or bf16 reductions that differ by BLAS.
- It does not make R2 exact. Rows 29–36 are reimplementations from config keys; their numerics are
  underdetermined and their grade stays `model` until a run says otherwise.

## 7 · What is deleted, and when

First commit on `nq-exact`, before any new code in that branch:

| path | disposition |
|---|---|
| `src/learn/nanoquant.rs` | delete |
| `src/learn/nanoquant/` (tests) | delete |
| `examples/nq_fit.rs` | delete; the driver becomes `examples/nq_stage.rs --stage <name> --config <json>` |
| `tests/learn_nanoquant.rs`, `tests/learn_block_tune.rs` | delete the cases that encode our conventions (unsigned `s_mid`, `INV_LIMBS`, `admm_loss`); keep the kernel equality tests |
| `src/format/nq.rs` | keep the container, add v3 (§3 rows 35–36); the v2 writer for the old fitter goes |
| `src/engine/kern/nq.rs` | keep, unchanged, until the v3 reader lands |
| `bench/notes/nq-proof-ladder.md` §6 | superseded by this ledger; the ladder's rungs 1–4 still stand |

Nothing is deleted in the current working tree. The deletions happen on the branch, so the
uncommitted work here is not touched.
