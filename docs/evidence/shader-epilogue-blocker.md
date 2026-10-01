# Shader epilogue refinement: blocked, with a real divergence finding

Grade: **blocker + trust-boundary finding**, not a cost or timing claim. The refinement leg is not
proved and its gate is red.

## Where it stands

- Files live only in the task worktree `/home/socol/oc-tmp/gpu-swarm/wt-w7-epilogue`
  (branch `task/gpu-w7-epilogue`): `proof/native/ShaderEpilogue.lean` (1416 lines, **does not
  compile**) and `proof/native/shader_epilogue_gate.py`.
- Gate `python3 proof/native/shader_epilogue_gate.py --controls` → exit **1**, 20.9 s. The
  `positive-proof` stage fails (15.8 s), so the mutation and restore stages never ran: **there is no
  red/restore evidence for this leg.**
- `SHADER_DOT_KERNEL_OK` (the existing `ShaderDot` audit, 19 declarations) stays green in the same
  build; the gate's own closed-cone audit caught the `sorryAx` the failed elaborations left behind
  (`wrap128_bias_of_fitsI64`, `roundShift32_wrap_iff`).
- 231 declarations are cone-clean; 28 carry `sorryAx` from the failed goals. No statement was
  weakened, no `sorry` inserted, the cone was not relaxed.

## Finding 1 — a divergence outside the stated domain clause

Outside `fitsI128 (m·dot·2^(we+ae))`:

- `|d·S| ≤ 2^110` always fits `i128`, so `src/engine/kern/gemm.rs:325` never fires;
- but `|d·S·dot| ≤ 2^129` can leave `i128`. There the CPU's `?.checked_mul(dot)`
  (`gemm.rs:326`) returns `None`, while the shader's `shiftedTerm` keeps the low 128 bits and can
  emit a value with **no flag set**.

So the CPU and shader epilogues are not equivalent off the domain clause. This is stated as the
excluded population in the module docstring; it is not proved away, and it is not a claim the leg
supports. Anything relying on shader/CPU epilogue equality must carry `fitsI128 (m·dot·2^(we+ae))`.

## Finding 2 — what the shader loop's `i128` bit means

The loop's `i128` bit is "some `addChecked` call reported overflow", **not** "no prefix is out of
range"; the two coincide only while the accumulator is in range. Hence `shaderLoop_flags` needs
`fitsI128 a`, and `shaderLoop_i128_mono` carries the bit afterwards. The invariant
`i128 flag = f.i128 || !prefixesFitB a bs` is false for out-of-range `a`, which is why the loop
theorem carries that hypothesis.

## First failing goal and decomposition

First failure `ShaderEpilogue.lean:436:6` in `ediv_eq_one` — a `Mathlib.mul_le_mul_of_nonneg_*`
argument-direction detail (a lemma-signature shape, not modelling). Bounded decomposition, in order,
each ≤3 files / one gate:

1. `ediv_eq_one`, `ediv_eq_neg_one` (lemma-shape blocker);
2. `cpuStep_of_WF`, `cpuLoop_some`, `cpuRun_eq`;
3. `shaderStep_cases` + the four loop lemmas (`shaderLoop_acc/flags_nf/i128_mono/flags`);
4. `wrap128_bias_of_fitsI64`, `roundShift32_wrap_iff`;
5. `shaderRun_eq/cases/refuses_iff/value`;
6. the three main theorems (`epilogue_refusal_iff`, `epilogue_value_eq`,
   `epilogue_value_eq_round_shift_32`) and `refusal_word_eq`;
7. anchors + the `#shader_epilogue_audit` marker.

Steps 1–2 carry the only unexpected-lemma shape; 3–6 are `rw`/`simp` bookkeeping on the same
statements. The worktree is retained so the statements are not re-derived.
