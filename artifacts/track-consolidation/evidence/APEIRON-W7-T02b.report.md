# APEIRON-W7-T02b: llama NORM RoPE via load-time q/k row permutation; rope.rs unchanged

Worktree /home/socol/oc-tmp/wt/worktree-agent-APEIRON-W7-T02 (task/w7-t02, base 04d2060, uncommitted). Started 06:42:51, finished ~07:30 AEST. No commit.

## Arc Block
- SCOPE: loader.rs, new loader/qk_permute.rs (+ qk_permute/tests.rs), error.rs, loader/tests.rs. rope.rs, step/attn_block.rs, step.rs reverted to 04d2060 (written from `git show 04d2060:<path>`; no git mutation).
- REGISTER: refinement/test. Excludes Lean (not run; not needed, rope.rs and proof/extract unchanged), speed.
- CORPUS: proof/extract/lib.rs:54 `#[path]` rope.rs, RopeHeadWitness.lean:37, RopeHeadContent.lean: unaffected, rope.rs sha256 identical to 04d2060.
- REFERENCE: /home/socol/Workspace/llama.cpp (ece963f41), cited below.
- GATE/GRADE: below. Grade: tested (cargo, end-to-end ids vs llama-server). Q32 score equality: tested, exact integer.

## Derivation of P
Head width n (rot_dims == head_dim, even). NORM pairs (2i, 2i+1), frequency index i. apeiron NEOX pairs (i, i+n/2), same index i (rope.rs turns[j] = theta^(-2j/n)/2pi).
P: position 2i -> i, 2i+1 -> i+n/2; (P x)[j] = x[src(j)], src(j) = 2j (j < n/2), 2(j-n/2)+1 (j >= n/2).
NORM(x) = P^-1 NEOX(P x). (P q).(P k) = q.k. Permuting output rows of W_q (n_head heads), W_k (n_head_kv heads), q/k biases, and qk-norm gains (RMS is permutation invariant, gain permuted) by P per head, then running the existing NEOX RoPE, gives NORM's scores. W_v, W_o untouched. Q32: dot is an exact i128 sum of identical rotated words, bit-identical. f32: summation order only.
## llama.cpp citations
- conversion/llama.py:174-179 `LlamaModel.permute`: `reshape(n_head, 2, d/2, ..).swapaxes(1, 2)`: new[2i+s] = old[s*d/2+i] (NEOX order -> NORM order). Applied to q_proj/k_proj weight and bias at :258-262 (`if self.undo_permute`). This repo's P is its inverse (NORM -> NEOX). (No `permute` in convert_hf_to_gguf.py at this checkout; it moved to conversion/llama.py.) Partial rotary: :170 rope_dim; refused here.
- src/llama-model.cpp:2611-2653 `llama_model_rope_type`: LLM_ARCH_LLAMA NORM; QWEN2 NEOX; QWEN35 IMROPE (pairs like NEOX, ggml-cpu/ops.cpp:6068-6083, per W7-T02).

## Change
- loader.rs: `qk_row_permutation_for(arch) -> Result<bool>`: qwen2, qwen35 false; llama true; else `EngineError::UnsupportedArchitecture`. Called first in `resolve_config` (refusal before any key or tensor read). `EngineConfig.qk_row_permutation`. `Engine::open` calls `permute_qk_rows()` after `load_decode_vectors` iff true. loader.rs 1337 lines (<1500).
- loader/qk_permute.rs (146 lines): `norm_source`, `permute_heads` (generic, in place), `permute_qk` (single choke point for q/k weights and biases), `row_bytes_of` (refuses `QuantKind::Other`, rank != 2, dims[1] != rows, stored size != rows * required_bytes(type, dims[0])), `Engine::permute_qk_rows` (every trunk and MTP block's attn_q/attn_k weight in the owned `data` buffer; biases and qk-norm gains in `vectors`). Refuses rot_dims != head_dim, odd head_dim, attn_gate. Complexity comment at the function: time O(bytes of q/k weights) byte copies, space O(one head's rows) scratch.
- error.rs: `UnsupportedArchitecture(String)`. loader/tests.rs: allowlist test renamed/adapted.
- Trust note: biases/gains are read via `read_range` from the file at load, so they are permuted in the resolved Vec<i64> (done). The `cfg(test)` `Sched::Sequential` oracle reads the file, not `data`, and would see unpermuted llama q/k; no llama test uses it (open item).

## Gates (cwd worktree, CARGO_TARGET_DIR=/home/socol/oc-tmp/wt/target-w7-t02, `flock -s`)
- a. new tests (4 in qk_permute + 1 loader), `cargo test --lib --locked qk_permute`: `test result: ok. 4 passed; 0 failed`. `norm_rotation_equals_inverse_permuted_neox_of_the_permuted_vector` (hd 64 and 128, 3 heads, 6 positions 0..100000, bit-exact; reference NORM built from one-pair RopeTables); `permuted_rows_plus_neox_give_the_same_scores_as_in_place_norm` (hd 64, 4 q heads / 2 kv, exact i128 dot equality; f32 within hd*EPS*sum|terms|; also asserts a k-skip changes a score).
- c. `cargo test --lib --locked`, flag unset: exit 0, `test result: ok. 414 passed; 0 failed; 32 ignored` (409 + 5). `APEIRON_F32_CARRIER=1`: exit 101, `FAILED. 409 passed; 5 failed` (404 + 5); the five: attn_fused_den_num_are_independent_of_tile_count, attn_fused_refuses_zero_sum_and_zero_tile, attn_head_distinct_scores_pin_the_scale, attn_saturates_out_of_window_exp_and_refuses_dot_overflow, engine_step_greedy_decode_is_deterministic: same five as baseline.
- b. `qwen35_on_the_one_executor_decodes_llama_cpp_ids`: flag 0 `1 passed` (97.78 s, earlier run), flag 1 `1 passed` (80.23 s), exit 0 both.
- d. clippy `--workspace --all-features --all-targets --locked -D warnings`: exit 0. `cargo fmt --all -- --check`: exit 0. `cargo test --workspace --all-features --locked` (flag unset): exit 0, 611 passed / 0 failed summed over `test result:` lines.
- e. `git diff 04d2060 --stat`: error.rs +5, loader.rs +27, loader/tests.rs +17 (49 insertions); untracked qk_permute.rs, qk_permute/tests.rs. `git diff 04d2060 -- rope.rs attn_block.rs step.rs` = 0 bytes. rope.rs sha256 == `git show 04d2060:src/engine/rope.rs | sha256sum` = f844469b573e7989d9fa40c959c0c9e86dc1b942f18b3f416855df045d040914.

## End to end (final source; throwaway example w7_ids in a tree copy, deleted; 64 prompt ids, 16 greedy ids; llama.cpp ids = T01 ll-*.txt)
- SmolLM2-135M Q8_0 (llama): Q8Act F32 16/16, Q8Act Q32 16/16, Exact F32 16/16, Exact Q32 16/16 (was 0/16). ids 359 281 933 6801 22827 30 13738 5257 6801 22827 30 13738 19 38 423 9651.
- Qwen2.5-0.5B: Q8Act F32 16/16; Q8Act Q32 0 matching prefix (ids 41535 525 3042 ...; T01 pre-existing near-tie); Exact both carriers 16/16. Q8Act ids files byte-identical (`cmp`) to T01 ap-qwen05b-Q8_0-c0/c1.txt.
- Qwen2.5-1.5B: Q8Act F32 8/16, Q8Act Q32 8/16 (ids 8 differs, llama's own tie); both `cmp`-identical to T01 ap-qwen15b-Q8_0-c0/c1.txt.
- Qwen3.5-0.8B: repo test passes under both flags.

## Mutation controls (throwaway copy /home/socol/oc-tmp/scratch/w7/t02b-ctl, deleted; worktree untouched)
- (a) `permute_qk`: `if which == 1 { return Ok(()) }` (attn_k weights and biases not permuted): `permuted_rows_plus_neox_give_the_same_scores_as_in_place_norm ... FAILED` at qk_permute/tests.rs:109 `Q32 exact, head 0 pos 0 left: 434689179554052485028374 right: -150152767254135550593485`; other 3 pass. SmolLM2 Q8Act end to end, both carriers: `Error: Norm(OutOfRange)` (fails, no ids produced).
- (b) `"qwen2" => Ok(true)`: Qwen2.5-0.5B Q8Act ids F32 `[2750, 13, 576, 1482, 11, ...]`, Q32 `[821, 13, 576, 2701, 374, ...]` vs llama `[7032, 525, 5230, ...]`; loader test `architecture_allowlist_maps_qk_permutation_and_refuses_unknown ... FAILED` (`left: Ok(true) right: Ok(false)`).
- Baseline before each: green (above).

## sha256 (worktree, final)
rope.rs f844469b573e7989d9fa40c959c0c9e86dc1b942f18b3f416855df045d040914 (== 04d2060)
loader.rs 203c35a994ad67fa2dd818708f8f707d207ec0ef6b6bd0a63aff7bb9ea574a39
error.rs dc389009e4239a6c09f4633598d40fed785084e0772a65d49dbbf3f651c5b49e
loader/qk_permute.rs f94e727c46329e3b684a6e2a74c96fca25cc898c57d4e09702c29311e073e35f
loader/qk_permute/tests.rs 75f5285ec922f8bb314ae284f33d45edd294171a3f89593fd73d02744fa74e90
loader/tests.rs 9747eaec154d77d37b99f66bde8a0d80e840b35d94c22b9174238abc982cc0fb
step.rs da2036a4d1e2e1a791d9dd900b40260369cdd71e3adecc7b39a14c11825357d2 (== 04d2060)
step/attn_block.rs 57d3f1b823ef71fbb8363ec5e0960710e8f48921802406ed59ebe1cd1972399f (== 04d2060)

## Open / trust boundary
- Lean not run and not needed: extracted sources unchanged. `proof/regen.sh --check` not run.
- One 64-token prompt, 16 ids; NORM verified only on SmolLM2-135M. Allowlist omits qwen3 and other NEOX archs. Partial rotary llama refused, not derived. IMROPE treated as NEOX (text-only).
- Test-only `Sched::Sequential` oracle reads the file, bypassing the permuted buffer.
- `cargo` prints `duplicate key` for ~/.cargo/git/.../opsfold/Cargo.toml:61 (pre-existing, exit codes 0).
- rustc/LLVM codegen and llama.cpp threading not covered. No speed measurement.
