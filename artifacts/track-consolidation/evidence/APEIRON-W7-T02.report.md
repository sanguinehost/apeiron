# APEIRON-W7-T02: RoPE layout from architecture; unknown architectures refused

Worktree /home/socol/oc-tmp/wt/worktree-agent-APEIRON-W7-T02 (task/w7-t02, base 04d2060, uncommitted). Started 06:10:19, finished 06:40:44 AEST (~30 min). No commit.

## Arc Block
- SCOPE: rope.rs, loader.rs, step/attn_block.rs (declared). Also touched (see Scope deviations): error.rs, step.rs, loader/tests.rs.
- REGISTER: refinement/test. Excludes: Lean (not run), speed.
- CORPUS: rope.rs is extracted by proof/extract/lib.rs (`#[path]`) and proof/ApeironWord.lean. NOT regenerated (no Lean rule). `proof/regen.sh --check` will report drift.
- REFERENCE: /home/socol/Workspace/llama.cpp (citations below).
- GATE: see results. GRADE: tested (cargo test, end-to-end ids vs llama-server); rope pairing vs direct reference: tested.

## llama.cpp citations
- `src/llama-model.cpp:2611-2647` `llama_model_rope_type`: `LLM_ARCH_LLAMA` (:2612) in the list that returns `LLAMA_ROPE_TYPE_NORM` (:2653 area; list ends at :2649 with `case ... LLAMA_EMBED/...`; T01 cited 2611-2653).
- `llama-model.cpp:2669` `LLM_ARCH_QWEN2`, `:2672` `LLM_ARCH_QWEN3` -> list returning `LLAMA_ROPE_TYPE_NEOX` (return at :2727).
- `llama-model.cpp:2737-2740` `LLM_ARCH_QWEN35` (also QWEN35MOE, QWEN3VL) -> `LLAMA_ROPE_TYPE_IMROPE`. GGUF general.architecture of /home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf is `qwen35` (strings on header; llama-arch.cpp:41).
- `ggml/src/ggml-cpu/ops.cpp:6068-6083`: `GGML_ROPE_TYPE_NORMAL` -> `rotate_pairs(n_dims, 1, ..., scale 1)` (pair (i0, i0+1)); `NEOX`, `MROPE`, `IMROPE` -> `rotate_pairs(n_dims, n_dims/2, ...)` (pair (ic, ic+n/2), ic = i0/2). Definition `ops.cpp:5934-5946`. So IMROPE pairs like NEOX; for text positions all axes equal, Qwen3.5 stays NEOX in apeiron (outputs unchanged, verified below).

## RoPE application sites (grep of src, examples)
All rotation goes through RopeTable data (`turns`, now `pair(j)`):
- rope.rs: `RopeTable::rotate`, `rotate_one`, `rotate_head` (was `(j, j+d2)`; now `self.pair(j)`).
- step/attn_block.rs `rotate_pair_prefixes` (was hardcoded `head[j]`, `head[j+d2]`; now `rope.pair(j)`).
- step.rs `rotate_head_prefixes` -> `rope.rotate_one`; `qk_norm_and_rope` -> both of the above. No pairing there.
- step/cols.rs:353 `self.rope_table()`; cols.rs:480 takes `&RopeTable`. No pairing there (not edited).
- attn/f32_core.rs: only test code (:953-1028) via `rotate_pair_prefixes`/`rotate_one`. No pairing.
- examples/decode_probe.rs:1034-1059 uses `engine.rope_table()` + `rotate_one`. No pairing.
- No rotation site outside scope hardcodes the pairing.

## Change
- rope.rs: `enum RopeLayout { Neox, Norm }`; `RopeTable.layout`; `with_layout`, `layout()`, `pair(j)`; `new`/`from_turns` default Neox. Q32 `fixed_sin_cos` arithmetic untouched; only indices.
- loader.rs: `rope_layout_for(arch)` allowlist: `qwen2`,`qwen35` -> Neox; `llama` -> Norm; else `EngineError::UnsupportedArchitecture(name)`. Called first in `resolve_config` (so `Engine::open` refuses before reading any key). `EngineConfig.rope_layout` added.
- error.rs: variant `UnsupportedArchitecture(String)` + Display.
- step.rs:962 `rope_table()` applies `.with_layout(self.config.rope_layout)` (the single constructor of the engine's table).
- attn_block.rs: `rotate_pair_prefixes` uses `rope.pair(j)`; its equivalence test now runs Neox and Norm.
- Tests: rope.rs `layouts_match_the_direct_reference` (Neox/Norm, rot 8/64, pos 0..4095, `rotate_one` and `rotate`), `partial_rotary_leaves_the_tail_and_layouts_differ` (rot 8 < hd 16), `default_constructors_build_neox`; loader/tests.rs `architecture_allowlist_maps_rope_layout_and_refuses_unknown`.

## Scope deviations (lead decide)
- step.rs (1 line) and error.rs (variant) are outside the three files. step.rs:962 is the only constructor, so carrying the layout on RopeTable needs it. error.rs holds `EngineError`; the alternative was an unnamed `ConfigMismatch`.
- loader/tests.rs edited (test appended), as brief allowed "where scope allows" only via rope.rs.
- Proof drift: rope.rs struct/body changed, so `proof/ApeironWord.lean` and any bridge theorem on `RopeTable`/`rotate_head` need `proof/regen.sh` + `lake build` (not run; Lean excluded by common rules). Extraction-fragment rules kept (no early return in loops, no new externals) but unverified by Aeneas. Open obligation.

## Gate results (cwd worktree, CARGO_TARGET_DIR=/home/socol/oc-tmp/wt/target-w7-t02, flock -s)
- d. `cargo test --lib --locked`: flag 0: exit 0, `test result: ok. 413 passed; 0 failed; 32 ignored` (409 + 4 new). flag 1: exit 101, `FAILED. 408 passed; 5 failed` (404 + 4 new); the five: attn_fused_den_num_are_independent_of_tile_count, attn_fused_refuses_zero_sum_and_zero_tile, attn_head_distinct_scores_pin_the_scale, attn_saturates_out_of_window_exp_and_refuses_dot_overflow, engine_step_greedy_decode_is_deterministic. Same five as baseline.
- d. `qwen35_on_the_one_executor_decodes_llama_cpp_ids`: flag 0 `1 passed` (82.70 s); flag 1 `1 passed` (74.63 s).
- b/a unit: rope tests 3 passed; loader test 1 passed (`13 passed` with filter `rope`).
- e. clippy `--workspace --all-features --all-targets --locked -D warnings` exit 0; `cargo fmt --all -- --check` exit 0; `cargo test --workspace --all-features --locked` (flag unset) exit 0, 610 passed / 0 failed summed over all `test result:` lines (lib 413).
- c. end-to-end: throwaway `examples/w7_ids.rs` (T01's) in a tree copy (deleted), 64 prompt ids, 16 greedy ids, llama.cpp ids from T01's `ll-*.txt` (llama-server temp 0).

| model | mode, carrier | matching prefix / 16 vs llama.cpp Q8_0 | vs T01 (before) |
|---|---|---:|---|
| SmolLM2-135M Q8_0 (llama) | Q8Act, F32 (=1) | 16 | was 0 |
| SmolLM2-135M | Q8Act, Q32 (=0) | 16 | was 0 |
| SmolLM2-135M | Exact, F32 | 16 | - |
| SmolLM2-135M | Exact, Q32 | 16 | - |
| Qwen2.5-0.5B | Q8Act, F32 | 16 | unchanged, ids byte-identical to T01 |
| Qwen2.5-0.5B | Q8Act, Q32 | 0 (ids 0 and 2 differ, T01 pre-existing near-tie) | unchanged, byte-identical to T01 |
| Qwen2.5-0.5B | Exact, both carriers | 16 | - |
| Qwen2.5-1.5B | Q8Act, F32 and Q32 | 8 | byte-identical to T01 files on both carriers |
| Qwen3.5-0.8B | test above, both flags | pass | unchanged |

## Mutation controls (throwaway copy, deleted)
- (a1) loader `"llama" => Ok(RopeLayout::Neox)`: `engine::loader::tests::architecture_allowlist_maps_rope_layout_and_refuses_unknown ... FAILED` at loader/tests.rs:318 `left: Ok(Neox) right: Ok(Norm)`. End-to-end SmolLM2 with that map: F32 `[31, 33, 31, 34, 32, 32, ...]`, Q32 `[31, 34, 32, 32, 32, ...]` (0/16).
- (a2) `RopeLayout::Norm => (j, j + head_dim/2)` in rope.rs: `engine::rope::tests::layouts_match_the_direct_reference ... FAILED` (rope.rs:395) and `partial_rotary_leaves_the_tail_and_layouts_differ ... FAILED` (rope.rs:421).
- (b) refusal arm replaced by `_ => Ok(RopeLayout::Neox)`: loader test `FAILED` at loader/tests.rs:319 `left: Ok(Neox) right: Err(UnsupportedArchitecture("mistral3"))`. Restored copies were discarded; worktree untouched by controls.

## sha256 (worktree)
rope.rs 11e13c37abaa63be551fb3249b4c5d83429b50a232cf73936f1ea0cb0db31221
loader.rs fd45e22043b3f2f670736225f0de402250d08c8cc3e7fdf88ee365f970d76914
error.rs dc389009e4239a6c09f4633598d40fed785084e0772a65d49dbbf3f651c5b49e
step.rs 2e9a8a94cb7c746faadbfe80e9fda36e3529dd0ece0449e9284ad3c8657c4e44
step/attn_block.rs d610216d62a1c13f8f4bd2e7f756d8f198d4590c8af06875d84e06076f8b411d
loader/tests.rs e9cb91523dab61759bea25077b6562ce9e59c455b6f25ca98aef9ad76d1b488f

git diff --stat: error.rs +5, loader.rs +21, loader/tests.rs +17, rope.rs +178/-, step.rs +1, attn_block.rs 16 changed; 226 insertions, 12 deletions.

## Open / trust boundary
- Lean extraction regen and bridge build not run (see deviations).
- 16 ids on 64-token prompts, one prompt; llama (NORM) only verified on SmolLM2-135M. Allowlist omits qwen3/other NEOX archs by design (untested).
- Qwen3.5 IMROPE is treated as NEOX: valid for text-only (single position axis); multimodal sections are not implemented.
- rustc/LLVM codegen and llama.cpp threading not covered. No speed measurement.
