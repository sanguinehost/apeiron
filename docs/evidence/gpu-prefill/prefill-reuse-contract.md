# Batched prefill reuse: source/refinement/cost contract proposal

Task `APEIRON-GPU-PREFILL-W6-T4`. Read-only analysis. Status: proposal for lead review; nothing here
is registered, admitted, `proved`, `tested` or `measured` by this task.

Base: this worktree at commit `e481943`. The evidence grades of `!AGENTS.md:186:## 5 · Evidence grades`
apply to every claim below.

## Citation convention and gate

A citation is `path:line:anchor` or `path:line`, in backticks, and resolves when the file exists,
the line is within it and the anchor is a substring of that line. Paths outside this worktree are
marked `MAIN-UNTRACKED` when they exist only in `/home/socol/Workspace/apeiron` as untracked or
uncommitted files. The structural gate is a throwaway checker, not a build:

```sh
python3 /home/socol/oc-tmp/gpu-swarm/check_prefill_reuse_citations.py \
  --doc docs/evidence/gpu-prefill/prefill-reuse-contract.md \
  --worktree /home/socol/oc-tmp/gpu-swarm/wt-w6-t4 \
  --main /home/socol/Workspace/apeiron
```

It prints a per-citation resolution checklist and exits non-zero unless every citation resolves and
every numeric line outside a fenced block is either cited or carries an explicit
`unmeasured` / `no receipt` / `not established` marker. The observed output is in the handback,
not in this file.

## Addressed object and rate

**Addressed object.** The weight payload of each projection tensor, not the activations. For a block
of `k` columns and a tensor whose ggml type is not `Q8_0`, `Engine::proj_cols` loops one
`proj_q8act_at` per column, so each column re-resolves the resident payload and re-streams the
whole matrix:

| fact | source |
|---|---|
| the `k > 1` batched branch exists but is gated to one type | `src/engine/step/cols.rs:222:if xs.len() > 1 && self.hadamard().is_none() {`, `src/engine/step/cols.rs:224:if ggml_type == quant::GGML_TYPE_Q8_0 {` |
| otherwise one single-column projection per column | `src/engine/step/cols.rs:237:for x in xs {` |
| the module already states the per-column cost | `src/engine/step/cols.rs:195:weight bytes read once` |
| the weight payload, not the activations, is the addressed object | `src/engine/step/cols.rs:190:kernel on a resident` |
| each column re-resolves the resident payload | `src/engine/step.rs:645:fn proj_q8act_at(`, `src/engine/step.rs:607:fn payload_at(` |

**Block width.** The default prefill block width is the source's own constant, and a prompt no longer
than it is one block:

| fact | source |
|---|---|
| block width default and its environment override | `src/engine/step.rs:360:.unwrap_or(256)` |
| the default and its rationale in the module doc | `src/engine/step.rs:351:Default 256` |
| the block loop that feeds `compose_cols` | `src/engine/step.rs:1527:pub(crate) fn prefill_cols_width(`, `src/engine/step.rs:1589:let cols = self.compose_cols(` |
| the greedy entry point | `src/engine/step.rs:1561:pub fn prefill_state(` |

**Current read rate.** The Wave-5 open item records the dense Qwen3-8B-Q4_K_M prefill at
`docs/epics/gpu-prefill.md:435:Dense Q4_K prefill has no weight reuse`, stated as 16.43 tok/s over
4.68 GiB of weights and as 82.6 of an 86.2 GB/s read ceiling (`docs/evidence/bench/1790713685-ns-r027.md:7:86.2`
for the ceiling). Two of those figures have a receipt in this tree and one does not:

| figure | status |
|---|---|
| 86.2 GB/s read ceiling | measured receipt, 16 threads: `docs/evidence/bench/1790713685-ns-r027.md:7:86.2` |
| 16.43 tok/s and 4.68 GiB on Qwen3-8B-Q4_K_M | `no receipt`: the only occurrence in this worktree is the epic line above, and this task did not re-run it |
| the Q8_0 hybrid 1227 tok/s over 0.78 GiB in the same line | `no receipt` in this worktree; the corresponding figure appears in an untracked artifact of the main checkout, not here |
| ~113 GMAC/s dense `ffn` versus ~1000 GMAC/s hybrid | `unmeasured` in this tree; stated without a receipt at `docs/epics/unified-quant-kernel.md:20:the dense Qwen3-8B-Q4_K_M prefill measured` |

| cross-check | command run in this task | reading | what it is not |
|---|---|---|---|
| model file size | `ls -l /home/socol/models/gguf/q3-8b/Qwen3-8B-Q4_K_M.gguf` | 5027784512 bytes, which is the epic's 4.68 GiB to two decimals | `unmeasured` for throughput: a file size is not a prefill measurement |

**Why the ceiling is reached.** For a `Q4_K` dense tensor nothing in the block reuses a weight byte:
the batched branch is type-gated at `src/engine/step/cols.rs:224`, the group kernel
`gemm_q8_0_group` refuses every non-`Q8_0` part at `src/engine/kern/gemm.rs:481:if *ggml_type != GGML_TYPE_Q8_0`
and defers to the per-part loop, and the fused SwiGLU epilogue keeps its own `Q8_0` gate at
`src/engine/step/cols.rs:773:if gt != quant::GGML_TYPE_Q8_0 || ut != quant::GGML_TYPE_Q8_0 {`. Each
of the `k` columns therefore pays the full matrix stream, which is the shape a DRAM-bound rate
saturates.

**Reuse factor a batched driver introduces.** This is a byte count, not a speedup. With the
driver's column cap `K` the number of weight-payload passes per tensor per block falls from `k` to
`⌈k/K⌉`:

| quantity | value | source |
|---|---|---|
| existing column cap of the pooled batched kernel | `MAX_K = 16` | `src/engine/kern/matmat.rs:47:const MAX_K: usize = 16;` |
| passes today on the dense `Q4_K` path at the block width above | one per column | `src/engine/step/cols.rs:237:for x in xs {` |
| passes after, at the existing cap | `⌈k/16⌉`, i.e. thirteen for that width | arithmetic on the two rows above with `src/engine/kern/matmat.rs:47:const MAX_K: usize = 16;` |
| multiply-adds before and after | unchanged at `k·n_out·n_in` | `src/engine/trace.rs:222:pub const fn macs(` |

No wall-clock factor is claimed. The existing source states the reuse measurement only for `Q8_0`
at `src/engine/step/cols.rs:7:1.02`, and its own cost study records that in the model the exact
scale epilogue makes the two-column projection compute-bound
(`docs/evidence/perf-attribution-prefill.md:74:compute-bound`), so the byte reduction is not
evidence of a throughput reduction.

**Open.** The largest structural risk to this whole proposal is that even the `Q8_0` path loses
reuse above the column cap: `matmat_q8act` falls back to per-column when `qs.len() > MAX_K`
(`src/engine/kern/matmat.rs:356:if !n_in.is_multiple_of(ACT_QK) || qs.is_empty() || qs.len() > MAX_K {`),
and the power-of-two GEMM it tries first needs AVX2 and AVX-VNNI
(`src/engine/kern/gemm.rs:360:avx2`, `src/engine/kern/gemm.rs:361:avxvnni`) and a power-of-two
activation scale (`src/engine/kern/gemm.rs:157:if s < 0 || (s != 0 && s.count_ones() != 1) {`).
Whether the hybrid's reuse comes from the GEMM or from a `k ≤ 16` block is `unmeasured` here.

## Executing path before and after

**Before.**

| stage | symbol | source |
|---|---|---|
| prefill entry | `prefill_state` → `prefill_state_with_width` | `src/engine/step.rs:1561:pub fn prefill_state(`, `src/engine/step.rs:1570:pub(crate) fn prefill_state_with_width(` |
| layer loop over columns | `compose_cols` | `src/engine/step/cols.rs:64:pub(crate) fn compose_cols(` |
| projection per part | `proj_cols` | `src/engine/step/cols.rs:200:pub(super) fn proj_cols(` |
| dense non-`Q8_0` branch | one `proj_q8act_at` per column | `src/engine/step/cols.rs:237:for x in xs {` |
| pooled branch | `matmat_q8act` | `src/engine/step/cols.rs:232:matmat_q8act(ggml_type`, `src/engine/kern/matmat.rs:126:pub(crate) fn matmat_q8act(` |
| single-column contract path | `proj_q8act_at` → `proj_q8act_quantified` → `matvec_q8act_quant_chunked` | `src/engine/step.rs:322:fn proj_q8act_quantified(`, `src/engine/step.rs:344:matvec_q8act_quant_chunked` |
| per-call plan | `q8act_plan` | `src/engine/chart/mod.rs:1655:pub(crate) fn q8act_plan(` |
| row loop | `q8act_rows` | `src/engine/chart/mod.rs:1732:pub(crate) fn q8act_rows(` |
| per-row fast accumulation | `q8act_row_i128_fast` | `src/engine/chart/mod.rs:1086:pub(crate) fn q8act_row_i128_fast(` |
| unified per-family accumulator | `q8act_accumulate_row<D>` | `src/engine/chart/mod.rs:910:pub(crate) fn q8act_accumulate_row<D: FastBlockDecoder>(` |
| value reference | `matvec_fixed` | `src/engine/matvec.rs:149:pub fn matvec_fixed(` |

**After (intended).** `proj_cols` drops only the type equality at
`src/engine/step/cols.rs:224:if ggml_type == quant::GGML_TYPE_Q8_0 {` and forwards every admitted
family to a batched driver whose row loop is the same decoder-parameterized accumulator already in
the tree: the trait is `FastBlockDecoder` at
`src/engine/chart/mod.rs:480:pub(crate) trait FastBlockDecoder {`, its AVX2 and portable twins are
`src/engine/chart/mod.rs:933:pub(crate) unsafe fn q8act_accumulate_row_avx2<D: FastBlockDecoder>(`
and `src/engine/chart/mod.rs:965:pub(crate) fn q8act_accumulate_row_portable<D: FastBlockDecoder>(`,
and its arithmetic is the single expression at
`src/engine/chart/mod.rs:906:acc += act_scale * (w_scale * dot - w_min * xsum)`. The batched driver is
the `k`-column analogue of `q8act_row_i128_fast`, which already dispatches all four families
(`src/engine/chart/mod.rs:1096:quant::GGML_TYPE_Q4_K => q8act_q4_k_row_fast`) and refuses everything
else at `src/engine/chart/mod.rs:1098:_ => None,`. No new per-family kernel is introduced: the four
decoders are `src/engine/chart/mod.rs:517:pub(crate) struct Q8_0Decoder;`,
`src/engine/chart/mod.rs:684:impl FastBlockDecoder for Q4_KDecoder {` and
`src/engine/chart/mod.rs:810:impl FastBlockDecoder for Q6_KDecoder {`.

**Fallback and refusal contract that must not change.**

| obligation | statement | source |
|---|---|---|
| per-column value identity | column `c` of the batched call equals the single-column call bit for bit | `src/engine/kern/matmat.rs:117:bytes read once per call instead of` |
| error identity | refuse with the lowest-indexed refusing column's `MatvecError` | `src/engine/kern/matmat.rs:111:and otherwise refuses with the refusal of the lowest-indexed refusing column` |
| ineligible family or shape | per-column reference `matvec_q8act` | `src/engine/kern/matmat.rs:140:return per_column(ggml_type` |
| pooled `Q8_0` kernel refusal | `None` routes to the serial kernel that owns the refusal | `src/engine/kern/matmat.rs:240:fn pooled_q8_0(` |
| non-fused family or no Q32.32 activation | `Q8ActPlan::Reference`, i.e. `matvec_fixed` on `x̂` | `src/engine/chart/mod.rs:1660:_ => return Q8ActPlan::Reference,` |
| a row that cannot prove the bound | whole payload falls back to `matvec_fixed` on the dequantized activation | `src/engine/chart/mod.rs:1807:if fallback {`, `src/engine/chart/mod.rs:1813:return matvec_fixed(ggml_type, raw, n_out, n_in, &xhat);` |
| rounded value leaves `i64` | `MatvecError::OutputOverflow` | `src/engine/chart/mod.rs:1815:if !ok {` |
| family predicate | `fused_type` | `src/engine/fused.rs:616:pub(super) fn fused_type(ggml_type: u32) -> bool {` |
| refusal flag surface | no new refusal variant; `StepError::Matvec` carries `MatvecError` | `src/engine/step.rs:1200:out.map_err(StepError::Matvec)` |
| block-level refusal ordering | the first refusing column refuses; state has already advanced through earlier columns | `src/engine/step/cols.rs:51:for the first column that refuses` |

The fused FFN and attention group paths keep their `Q8_0` gates and their `None`-means-defer
semantics (`src/engine/step/cols.rs:295:if let Some(cols) = gemm_q8_0_group(&group, n_in, qs) {`,
`src/engine/step/cols.rs:752:fn ffn_fused_swiglu(`); the change is confined to the single-part
projection route.

## Reachable joint domain

`reachable x` is "the driver's inputs arise from a real prompt block on a real model";
`admissible x` is "the shape, quant and bound preconditions below hold". The joint witness must be
one term satisfying both, not separate inhabitants.

| condition | value | source |
|---|---|---|
| family must be fused | `Q8_0`, `Q5_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q6_K` | `src/engine/fused.rs:617:ggml_type == quant::GGML_TYPE_Q8_0` |
| family must have a decoder | `Q8_0`, `Q5_0`, `Q4_K`, `Q6_K` only | `src/engine/chart/mod.rs:1098:_ => None,` |
| `Q8_0` width and bound | `n_in % 32 == 0`, `top = 128`, narrow | `src/engine/chart/mod.rs:1664:quant::GGML_TYPE_Q8_0 if n_in.is_multiple_of(ACT_QK) => {` |
| `Q5_0` width and bound | `n_in % 32 == 0`, `top = 16`, narrow | `src/engine/chart/mod.rs:1667:quant::GGML_TYPE_Q5_0 if n_in.is_multiple_of(ACT_QK) => {` |
| `Q4_K` width and bound | `n_in % 64 == 0`, `top = 1008`, not narrow | `src/engine/chart/mod.rs:1670:quant::GGML_TYPE_Q4_K if n_in.is_multiple_of(64) => {` |
| `Q6_K` width and bound | `n_in % 256 == 0`, `top = 4096`, not narrow | `src/engine/chart/mod.rs:1673:quant::GGML_TYPE_Q6_K if n_in.is_multiple_of(QK_K) => {` |
| narrow extra bound | `S_max·(32·top·127) ≤ i64::MAX` when `narrow` | `src/engine/chart/mod.rs:356:if narrow && s_max.saturating_mul(32 * top * 127)` |
| call-level bound | `n_in·(65504·2^32)·top·127·S_max < i128::MAX`, strict | `src/engine/chart/mod.rs:366:bound < i128::MAX` |
| activation magnitude exact | `ActMagnitude::Exact` | `src/engine/chart/mod.rs:1657:ActMagnitude::Exact(v) if fused_type(ggml_type)` |
| per-column geometry | `q.len() == n_in` and `q.scales().len() == n_in/32` | `src/engine/kern/matmat.rs:367:if q.len() != n_in || q.scales().len() != blocks {` |
| column count | `1 ≤ k ≤ K`; today `K = MAX_K = 16` | `src/engine/kern/matmat.rs:356:if !n_in.is_multiple_of(ACT_QK) || qs.is_empty() || qs.len() > MAX_K {`, `src/engine/kern/matmat.rs:47:const MAX_K: usize = 16;` |
| `Q4_K` needs per-block `Σ xq` | computed per column, once per call, not per row | `src/engine/chart/mod.rs:1681:if fast && ggml_type == quant::GGML_TYPE_Q4_K {` |
| stack scratch | `[i128; MAX_K]` per row | `src/engine/kern/matmat.rs:431:let mut totals = [0i128; MAX_K];` |
| width cap on the power-of-two route | `n_in/32 ≤ 512` | `src/engine/kern/gemm.rs:81:const MAX_BLOCKS: usize = 512;`, `src/engine/kern/gemm.rs:359:n_in / ACT_QK > MAX_BLOCKS` |

**Witness.** Reachable and joint in this tree:

| witness | model | columns | source |
|---|---|---|---|
| dense `Q4_K`, two columns | `/home/socol/models/gguf/qwen05b-Q4_K_M.gguf` | two | `src/engine/tests.rs:1854:fn qwen25_two_columns_equal_two_single_steps()`, `src/engine/tests.rs:1856:cols_equal_singles(QWEN05B_PATH, &[785, 6722]);` |
| `Q8_0` hybrid, wide block | `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` | forty columns | `src/engine/tests.rs:1755:fn qwen35_two_columns_equal_two_single_steps()`, `src/engine/tests.rs:1485:const QWEN35_08B_PATH` |

**Witness gap, stated rather than filled.** No in-tree test exercises a dense `Q4_K` block with
`3 ≤ k ≤ 16`; the dense witness covers `k = 2` only. A joint witness for the whole admitted `Q4_K`
column range does not exist today and must be added by the implementation task before any
certificate claims `reachable` for `k > 2`. The synthesis-only Gram-matrix tests in
`src/engine/kern/matmat.rs:870` use `QuantAct::from_parts` fixtures and are `tested` reachability of
the driver shape, not of a model tensor.

## Preservation

**Testable predicate.** For every admitted input, with `x̂_c` the dequantized column `c`:

```text
out[c][o] = matvec_fixed(ggml_type, row o of raw, n_in, x̂_c)[0]      for all o < n_out, c < k
and on refusal:  Err is the matvec_q8act refusal of the lowest-indexed refusing column
```

This is the contract `matmat_q8act` already states (`src/engine/kern/matmat.rs:105:equals the per-column call bit for bit`,
`src/engine/kern/matmat.rs:110:exactly when every column`), and it is enforced by tests that must stay
green:

| oracle | scope | source |
|---|---|---|
| every family agrees with the fixed kernel | `chart_matches_fixed_on_random_payloads_for_every_family` | `src/engine/chart/tests.rs:223:fn chart_matches_fixed_on_random_payloads_for_every_family()` |
| batched columns equal per-column `matvec_q8act` | `columns_match_per_column_matvec_q8act` | `src/engine/kern/matmat.rs:870:fn columns_match_per_column_matvec_q8act()` |
| batched refusals equal per-column refusals | `refusals_match_per_column` | `src/engine/kern/matmat.rs:944:fn refusals_match_per_column()` |
| a block's columns equal the single steps, whole state equal after | `cols_equal_singles` | `src/engine/tests.rs:1743:assert_eq!(cols[c], single, "{path}: column {c}");` |
| AVX2 cores equal their portable twins | `fast_row_bodies_match_their_portable_twins` and siblings | `src/engine/chart/tests.rs:151:fn fast_row_bodies_match_their_portable_twins()` |

The `fused_type`/`NoQ32`/bound-failure population is excluded from `admissible`, not from
`preserves`: those inputs keep routing to `matvec_fixed` through
`src/engine/chart/mod.rs:1857:Q8ActPlan::Reference => {` and remain byte-identical by the same
oracle.

## Resource model and strict improvement

**Registered measure.** Weight-payload bytes read per projection per block, with
`B = n_out·row_bytes` the payload size. This is a source-level byte count of the loop's read set,
not a timing and not a hardware counter:

```text
beforeCost = k · B
afterCost  = ⌈k / K⌉ · B          (K = the driver's admitted column cap)
improves   = ∀ x, reachable x → admissible x → afterCost x < beforeCost x
```

The shape of that obligation is fixed by the launcher's certificate structure: `beforeCost` and
`afterCost` are `Input → Nat` and `improves` is the strict-improvement field at
`!proof/DispatchAdmission.lean:15:improves : ∀ x, reachable x → admissible x → afterCost x < beforeCost x`,
with the joint witness field at `!proof/DispatchAdmission.lean:13:joint : reachable witness ∧ admissible witness`
and preservation at `!proof/DispatchAdmission.lean:14:preserves : ∀ x, reachable x → admissible x → after x = before x`.
The launcher requires exactly those names
(`!tools/dispatch/main.rs:273:"namespace",`).

| quantity | before | after | source |
|---|---|---|---|
| weight bytes per tensor per block, `k = 200`, `K = 16` | 200·B | 13·B | derived from `src/engine/step/cols.rs:237:for x in xs {` and `src/engine/kern/matmat.rs:47:const MAX_K: usize = 16;` |
| weight-block decodes | `k·n_out·blocks` | `⌈k/K⌉·n_out·blocks` | `src/engine/kern/matmat.rs:414:The block loop decodes each weight block once` |
| multiply-adds | `k·n_out·n_in` | `k·n_out·n_in`, unchanged | `src/engine/trace.rs:222:pub const fn macs(` |
| activation code bytes read | unchanged, `k` columns | unchanged | `src/engine/kern/matmat.rs:117:bytes read once per call instead of` |

**Strict improvement.** `⌈k/K⌉ < k` holds for every `k ≥ 2` at any `K ≥ 1`, so `improves` is
discharged on the admitted domain by arithmetic once `K ≥ 1`. `reachable` alone does not give it: the
single-column case is reachable and keeps its single-column projections
(`src/engine/step/cols.rs:13:keeps the single-column projection groups`), so it admits nothing. The
certificate must therefore make the multi-column condition part of `admissible`, not merely of
`reachable`.

**What this does not claim.** It does not claim a speedup. The AGENTS rule is explicit that a
cost-model proof is not a hardware measurement
(`!AGENTS.md:359:changes are separate tasks`),
and the existing model study records that the exact per-(row, block, column) scale epilogue is what
limits this kernel (`docs/evidence/perf-attribution-prefill.md:78:Projection kernel epilogue`).

## Rejecting controls

Three controls the launcher requires for a new contract: a production-path positive, an
attributable refusal, and restoration on the same identities.

| control | exact command shape | expected |
|---|---|---|
| positive, byte identity | `cargo test --locked --lib -- engine::kern::matmat::tests::columns_match_per_column_matvec_q8act engine::kern::matmat::tests::refusals_match_per_column` (oracles at `src/engine/kern/matmat.rs:870`, `src/engine/kern/matmat.rs:944`) | exit 0 |
| positive, family parity | `cargo test --locked --lib -- engine::chart::tests::chart_matches_fixed_on_random_payloads_for_every_family` (oracle at `src/engine/chart/tests.rs:223`) | exit 0 |
| positive, whole-block identity | `cargo test --locked --lib -- engine::tests::qwen35_two_columns_equal_two_single_steps engine::tests::qwen25_two_columns_equal_two_single_steps` (oracles at `src/engine/tests.rs:1755`, `src/engine/tests.rs:1854`) | exit 0, skipped-with-message if the GGUF is absent |
| attributable refusal, source | in a throwaway copy, change the batched per-column accumulation `acc[c] +=` to `-=` at the driver's column fold; rerun the first control | non-zero with a column mismatch, not a build error |
| attributable refusal, launcher | `python3 tools/dispatch/admit.py --launch /tmp/unregistered.json` with a contract key absent from the registry | exit 1, `ADMISSION_UNREGISTERED_CONTRACT` (`!tools/dispatch/main.rs:402:ADMISSION_UNREGISTERED_CONTRACT`), worker marker absent |
| attributable refusal, certificate | corrupt one declared name in the registered certificate | exit 1, `ADMISSION_CERTIFICATE_REJECT` (`!tools/dispatch/main.rs:316:&format!("ADMISSION_CERTIFICATE_REJECT {label}: {out}"),`) |
| restoration | restore the exact bytes, rerun the positive controls above on the same source identity; expect exit 0 with byte-identical output (`src/engine/kern/matmat.rs:870:fn columns_match_per_column_matvec_q8act()`) |

A missing model file, a missing tool or an unrelated build failure is not an attributable red. The
launcher runs the bound worker with the repository root as cwd
(`!tools/dispatch/main.rs:581:let (code, out) = self.run(&worker, root, &format!("{label}-worker"), None)?;`)
inside one 530-second budget (`!tools/dispatch/main.rs:145:let mut left = 530_u64`), after a fresh
`proof/regen.sh --check` (`!tools/dispatch/main.rs:503:proof/regen.sh`) and a
`lake build <build_target>` (`!tools/dispatch/main.rs:517:"lake".into()`). The budget covering
extraction, dependency build, certificate and worker is `unmeasured` for this proposal.

## Owned paths and worker argv

| owned file | change |
|---|---|
| `src/engine/chart/mod.rs` | add the `k`-column entry over `FastBlockDecoder`, beside `q8act_row_i128_fast` (`src/engine/chart/mod.rs:1086:pub(crate) fn q8act_row_i128_fast(`) |
| `src/engine/kern/matmat.rs` | generalise `fast_family` (`src/engine/kern/matmat.rs:350:fn fast_family(`) and the row loop over the four decoders, keeping `per_column` as the refusal owner (`src/engine/kern/matmat.rs:140:return per_column(ggml_type`) |
| `src/engine/step/cols.rs` | drop the single-type gate at `src/engine/step/cols.rs:224:if ggml_type == quant::GGML_TYPE_Q8_0 {`, leaving the eligibility decision to the driver |

Three paths, which is the launcher's maximum (`!tools/dispatch/main.rs:418:paths.len() <= 3`), and
each must also appear in the registry contract's `owned_files`
(`!tools/dispatch/main.rs:423:require(registered_paths.contains(path), "ADMISSION_UNOWNED_PATH")?;`).

**Worker argv** (identical in `proposal.json` and the registered contract, or the launcher refuses
with `ADMISSION_WORKER_SUBSTITUTION` at `!tools/dispatch/main.rs:436:"ADMISSION_WORKER_SUBSTITUTION",`):

```json
["cargo", "test", "--locked", "--lib", "--",
 "engine::kern::matmat::tests",
 "engine::chart::tests",
 "engine::tests::qwen35_two_columns_equal_two_single_steps",
 "engine::tests::qwen25_two_columns_equal_two_single_steps"]
```

Residual: the multi-filter form depends on libtest accepting several filter arguments. The
single-filter fallback is `["cargo", "test", "--locked", "--lib", "engine::kern::matmat"]`, which
loses the whole-block identity control. Which form works is `unmeasured` here.

## Open and not established

| item | status |
|---|---|
| registry entry for this contract | absent: `!tools/dispatch/contracts.json:3:"contracts": {},`, and the epic records admission as closed (`docs/epics/gpu-prefill.md:98:closed to dispatch`) |
| source/refinement link for the driver | `not established` and, on this tree, unreachable: `source_root` must appear as a `--start-from crate::<root>` selection in `proof/regen.sh` (`!tools/dispatch/main.rs:447:let selected = format!("crate::{source_root}");`, refusal `ADMISSION_NOT_EXTRACTED` at `!tools/dispatch/main.rs:452:"ADMISSION_NOT_EXTRACTED",`), while the extraction source list (`proof/regen.sh:69:SOURCE_FILES="`) contains only `src/engine/matvec.rs` from the engine projections (`proof/regen.sh:102:$REPO_DIR/src/engine/matvec.rs`, selected at `proof/regen.sh:192:--start-from crate::word_matvec_fixed`) and `chart` is declared outside the extracted fragment (`docs/epics/unified-quant-kernel.md:37:is outside the extracted fragment`) |
| therefore the only `source_root` nameable today | `word_matvec_fixed`, whose wrapper calls the real body at `proof/extract/lib.rs:860:engine::matvec::matvec_fixed(`; it does not refine a batched `Q4_K` driver |
| `build_target` | must be a root of the corpus package (`!proof/corpus/lakefile.lean:21:@[default_target]`); a new root is a lakefile change outside the three owned files. The launcher requires the field (`!tools/dispatch/main.rs:456:"ADMISSION_BUILD_TARGET_REJECT",`). |
| `identity` closure | the launcher requires a fixed closure list including the `proof/dispatch` tree (`!tools/dispatch/main.rs:485:"proof/dispatch/lakefile.lean",`) and the contract's `proof_file` (`!tools/dispatch/main.rs:462:identity.contains_key(text(contract, "proof_file")?)`). Every listed file must be present in the worktree |
| launcher visibility of the harness | `tools/dispatch/`, `proof/dispatch/` and `proof/DispatchAdmission.lean` are `MAIN-UNTRACKED`: absent from this worktree, untracked in the main checkout |
| AGENTS.md admission rule | `MAIN-UNTRACKED`: the §9.1 admission section is an uncommitted addition in the main checkout (`!AGENTS.md:340:### Mandatory inference optimization admission`); it is not in the committed base of this worktree |
| registry/checker edits | separate tasks by rule (`!AGENTS.md:358:a worker cannot register its own optimization`), so the registration step is not part of the worker's three files |
| reuse on the `Q8_0` hybrid above the column cap | `unmeasured`; `matmat_q8act` falls back per column above `MAX_K` (`src/engine/kern/matmat.rs:356:if !n_in.is_multiple_of(ACT_QK) || qs.is_empty() || qs.len() > MAX_K {`) |
| register pressure of a wider column cap | `unmeasured`; the pooled row keeps `[i128; MAX_K]` stack scratch (`src/engine/kern/matmat.rs:431:let mut totals = [0i128; MAX_K];`) |
| `Q2_K` and `Q3_K` | outside the driver: `fused_type` admits them (`src/engine/fused.rs:616:pub(super) fn fused_type(ggml_type: u32) -> bool {`) but no `FastBlockDecoder` exists (`src/engine/chart/mod.rs:1098:_ => None,`), so they keep `matvec_fixed` |
| whole-prefill gain from this change | `not established`; the projection is one phase of a block whose attention, convolution and recurrent heads do not amortize over columns (`docs/evidence/perf-attribution-prefill.md:68:At 70% acceptance`) |
| docs index row | none required: the sibling evidence documents under `docs/evidence/gpu-prefill/` are not listed in `docs/README.md:71:## Evidence (`evidence/`)` |

## Handback facts

| item | value |
|---|---|
| file | `docs/evidence/gpu-prefill/prefill-reuse-contract.md` |
| files changed | this file only |
| commands run | none that build, benchmark or prove; the structural citation checker only |
| grade | planning / anchor; no source, cost or performance claim |

```text
SCOPE:      docs/evidence/gpu-prefill/prefill-reuse-contract.md; a contract proposal for human
            review. Completion: every claim cited to a resolving path:line or marked unmeasured.
REGISTER:   contract/plan registration for the batched prefill reuse; no implementation, no
            extraction, no registry edit, no optimization dispatch.
CORPUS:     src/engine/step.rs, src/engine/step/cols.rs, src/engine/chart/mod.rs,
            src/engine/kern/{gemm,matmat,tp}.rs, src/engine/matvec.rs, src/engine/fused.rs,
            src/engine/trace.rs read; proof/DispatchAdmission.lean and tools/dispatch/main.rs
            read in the main checkout. No canonical corpus theorem read or changed.
REFERENCE:  /home/socol/Workspace/apeiron/src/engine/chart/mod.rs:910 for the unified
            accumulator the proposal reuses; NONE - inventing.
GATE:       MISSING as a build gate; the structural citation checker in /home/socol/oc-tmp/gpu-swarm
            with a 540 s budget is the only gate this task ran. The implementation task is
            gate-first.
GRADE:      planning/anchor. Before: none. After: none. No speed, cost or admission claim.
```