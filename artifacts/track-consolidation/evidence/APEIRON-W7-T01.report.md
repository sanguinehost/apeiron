# APEIRON-W7-T01: does the Wave-6 speed carry to all-attention transformers

Host: i9-14900KF, 32 CPUs, governor powersave. Binary `/home/socol/oc-tmp/wt/bin/engine_profile-04d2060` (receipts: git 04d206013f70, 0 dirty paths, src sha256 0218e7d9...f12fa), cwd `/home/socol/oc-tmp/wt/w6-integ`, `APEIRON_F32_CARRIER=1`, llama.cpp 0324696b8 (`llama_cpu`, threads 8,16,24,32). Started 05:22:39, report written about 06:12 (about 50 min of 150). No edits to w6-integ. "measured" = from a receipt. "derived" = arithmetic on measured values or on source reading.

## Answer

- No. On these standard transformers the ratio vs llama.cpp is lower than on the hybrid, and >= 2.0x on both pp and tg is reached by none of them.
- pp (measured, median): Qwen2.5-0.5B 1.55-1.57x, Qwen2.5-1.5B 1.77-2.01x (pp200) and 1.91x (pp512); hybrid Qwen3.5-0.8B 2.45x at pp512 (2.17x at pp200, W6 receipt).
- tg (measured, median): 0.5B 1.42-1.55x; 1.5B 1.10-1.21x; hybrid 1.68x at pp512 run (1.76x W6).
- tg cap (derived): at Q8_0 bytes and this host's read ceiling, tg cannot reach 2.0x llama.cpp for these models: max ratio 1.72x (0.5B) and 1.33-1.36x (1.5B). Apeiron already streams 81-90% of the ceiling; llama.cpp streams 58% (0.5B) and 74-75% (1.5B).
- SmolLM2-135M (arch `llama`): apeiron loads it without error but emits wrong tokens (0/16). Cause (derived from source): `src/engine/rope.rs` rotates pairs `(j, j+head_dim/2)` (NEOX); llama.cpp uses consecutive pairs (`LLAMA_ROPE_TYPE_NORM`, `llama-model.cpp:2611-2653`) for arch `llama`. No speed run (correctness gate failed).

## Fixtures (scratch `/home/socol/oc-tmp/w7-fixtures/`)

Quantize: `llama-quantize <f16> <out> Q8_0 8` (build-cpu), all exit 0.

| model | arch | layers | n_embd | n_ff | heads / kv | head dim | vocab | tied head | q/k/v bias | input sha256 | Q8_0 output sha256 |
|---|---|---:|---:|---:|---|---:|---:|---|---|---|---|
| Qwen2.5-0.5B `qwen05b-Q8_0.gguf` (531,067,808 B) | qwen2 | 24 | 896 | 4864 | 14 / 2 | 64 | 151936 | yes (no output.weight) | yes | 73f932c0a3a3daf1122545473973b4189ef0811cd6f2f14d73afe9fa1edd8efd (qwen05b-f16.gguf) | 62394c3af2f7e1a7db73cf86d47638a3df4d19c1fcd20b9dd2a07c40444f8a37 |
| Qwen2.5-1.5B `qwen15b-Q8_0.gguf` (1,894,532,128 B) | qwen2 | 28 | 1536 | 8960 | 12 / 2 | 128 | 151936 | no (output.weight present) | yes | fc89e330deb3fd8fa560f1c0f35a1e2b8da96d59e13445559ed190307a6f5649 (qwen2.5-1.5b-instruct-fp16.gguf) | bcbdf6b481ca6cd7197c80516b621da3238dc51d703062e0667b71c6aa08676a |
| SmolLM2-135M `smollm135m-Q8_0.gguf` (144,810,592 B) | llama | 30 | 576 | 1536 | 9 / 3 | 64 (`key_length`) | 49152 | yes | no | 686ad3458e9c8c106ab0b378b0036c97edbe9d82df5d443d69165dcaffaaea16 (smollm2135m-f16.gguf) | dbf0d1f31d3afd0864bb02a916b7e3762728fd616eebd34bd6021c7497def219 |
| Qwen3.5-0.8B (hybrid, reference row) | qwen35 | 25 (24 + MTP) | 1024 | 3584 | 8 / 2 | 256 | 248320 | yes | no | - | 37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f |

Dims read with llama.cpp `gguf-py` (`meta.py`, output `meta.txt`). head dim = n_embd / heads except SmolLM2 and Qwen3.5 (explicit key_length). rope freq_base: 1e6 (Qwen2.5), 1e5 (SmolLM2).

Prompt text: `/home/socol/Workspace/apeiron/README.md` copy `text.txt`, sha256 4da581a6d728f8b2ac8336e06b8ae2aab4a80efbb157f477e0b5a1460e71a898. `llama-tokenize -m <model> -f text.txt --ids` (build-cuda binary, `CUDA_VISIBLE_DEVICES=` empty), default BOS handling:

| ids file | tokenised with | tokens | ids sha256 (from receipts) |
|---|---|---:|---|
| `qwen25.ids` (= `qwen15.ids`, byte-identical; 1.5B tokenizer re-run checked equal) | Qwen2.5 | 2660 | 0e7b28afb1d1aca440ddb0a7aa0987ba7818d97f7b1a3fa2ba0e5b052d6b3c1e |
| `smollm.ids` | SmolLM2 | 2917 | - |
| `q35.ids` | Qwen3.5 | 2700 | 2fdda083149e7c9f9fbd065d5803cf40334f5c5063fb030342b0ffccc96d24ae |

Format: whitespace-separated ids, one line (same as `q35-t5/prompt2.ids`). engine_profile uses the first `pp` ids.

## Correctness (first 64 prompt ids, 16 greedy tokens)

Apeiron: throwaway example `examples/w7_ids.rs` in a copy `/home/socol/oc-tmp/w7-fixtures/tree-copy` (rsync of w6-integ minus target/.git; own target `tree-copy-target`; built with `flock -s`). It calls `Engine::open`, `prefill_state`, `next_token_state` with `KernelMode::Q8Act` (the mode engine_profile uses, and the one `qwen35_on_the_one_executor_decodes_llama_cpp_ids` uses) or `Exact`. Carrier from `APEIRON_F32_CARRIER`. The repo test hardcodes llama.cpp ids (`src/engine/tests.rs:1542-1556`), so no generator existed.
llama.cpp: `llama-server` (build-cuda binary, `-ngl 0 -t 8`, `CUDA_VISIBLE_DEVICES=` empty) `/completion` with the 64 ids as a token-array prompt, `n_predict 16, temperature 0, top_k 1, return_tokens` (`ll.sh`, `ll-*.txt`).

llama.cpp reference ids:
- 0.5B Q8_0: 7032 525 5230 382 565 220 16 13 15 13 15 271 14374 17684 271 12
- 1.5B Q8_0: 56480 525 3881 382 2 730 5559 271 565 34807 271 42 5559 374 264 33789
- 1.5B F16 (original fp16 gguf): 56480 525 3881 382 2 730 5559 271 42 5559 374 264 33789 6733 369 4378
- SmolLM2 Q8_0: 359 281 933 6801 22827 30 13738 5257 6801 22827 30 13738 19 38 423 9651

| model | apeiron mode, carrier | matching prefix / 16 vs llama Q8_0 | vs llama F16 | first apeiron ids |
|---|---|---:|---:|---|
| Qwen2.5-0.5B | Q8Act, F32 (=1) | 16 | - | 7032 525 5230 ... |
| Qwen2.5-0.5B | Q8Act, Q32 (=0) | 0 (ids 0 and 2 differ: 41535 vs 7032, 3042 vs 5230; 14/16 equal by position) | - | 41535 525 3042 382 565 ... |
| Qwen2.5-0.5B | Exact, Q32 and F32 | 16 and 16 | - | 7032 525 5230 ... |
| Qwen2.5-1.5B | Q8Act, F32 | 8 (id 8: 42 vs 565) | 16 | 56480 525 3881 382 2 730 5559 271 42 ... 4378 |
| Qwen2.5-1.5B | Q8Act, Q32 | 8 | 15 (last id 3238 vs 4378) | ... 3238 |
| Qwen2.5-1.5B | Exact, Q32 | 8 | 16 | ... 4378 |
| SmolLM2-135M | Q8Act, F32 | 0 | - | 31 33 31 34 32 32 ... |
| SmolLM2-135M | Q8Act, Q32 | 0 | - | 31 34 32 32 32 ... |

Note on the 0.5B Q32 Q8Act row: apeiron ids `41535 525 3042 382 565 220 16 13 15 13 15 271 14374 17684 271 12` vs llama `7032 525 5230 382 ...`. The Exact mode on the same carrier-0 path gives 16/16, so the Q8Act activation rounding flips near-tie argmaxes here.

Reading:
- 1.5B: the divergence from llama Q8_0 at id 8 is llama.cpp's own: llama.cpp F16 and apeiron Exact agree for all 16 ids (measured). Not an apeiron defect on this evidence; both Q8_0 paths pick a different member of a near tie.
- Qwen2 (biases, tied or untied head, GQA head dim 64 and 128) loads and runs. SmolLM2 is the only refusal-free failure (below).

Refusals: none at load. Silent wrong output for arch `llama`: missing feature = consecutive-pair (NORM) RoPE. Loader `src/engine/loader.rs` resolves `<arch>.*` keys generically, so it accepts `llama` and there is no error message to quote. Root-cause status: derived from `rope.rs` header comment and `llama-model.cpp`, not confirmed by an experiment with a patched pairing (no engine edits allowed).

## Speed

Command (per row): `cd /home/socol/oc-tmp/wt/w6-integ && APEIRON_F32_CARRIER=1 timeout 580 flock -x -w 900 /home/socol/oc-tmp/w6-host.lock timeout 560 /home/socol/oc-tmp/wt/bin/engine_profile-04d2060 --model <gguf> --ids <ids> --pp <P> --tg 64 --reps <R> --llama /home/socol/oc-tmp/astra-projection-profile/llama_cpu --llama-threads 8,16,24,32 --label w7-<name> --out /home/socol/oc-tmp/ds-dispatch/ab`. Tg is 64 tokens after the pp prompt. Ranges are min-max over rounds. Receipts in `/home/socol/oc-tmp/ds-dispatch/ab/`. Logs `/home/socol/oc-tmp/w7-fixtures/run-*.log`, scripts `speed.sh`, `speed2.sh`.

The receipt wording is "quiet gate busy < 0.05: every round passed" (no separate "every round passed" line). All rows below have it except where flagged. The host is the owner's desktop (Discord, sunshine, ChatGPT ran in the background at ~0.05 busy), so busy sits at the 0.05 gate edge.

| model | P | reps | apeiron pp (range) | llama pp (range) | pp ratio (range) | apeiron tg (range) | llama tg (range) | tg ratio (range) | receipt | quiet |
|---|---:|---:|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B hybrid | 512 | 5 | 1188.65 (1058.99-1214.34) | 453.90 (434.99-520.84) | 2.45 (2.30-2.79) | 85.87 (78.30-91.00) | 51.04 (50.32-52.75) | 1.68 (1.55-1.79) | 1790798160-w7-q35r2-pp512.md | every round passed |
| Qwen3.5-0.8B hybrid | 512 | 5 | 1127.18 (792.27-1146.34) | 471.00 (405.08-502.46) | 2.30 (1.83-2.83) | 79.51 (62.17-84.31) | 51.52 (49.80-52.13) | 1.53 (1.25-1.62) | 1790797395-w7-q35-pp512.md (first run) | flagged: 3 of 5 rounds busy 0.062-0.074 |
| Qwen3.5-0.8B hybrid | 200 | - | 1227.43 | 560.39 | 2.17 | 88.99 | 50.38 | 1.76 | ab/1790795647-w6e-vs-llama-r3.md (W6, cited, not re-run) | - |
| Qwen2.5-0.5B | 200 | 5 | 1646.26 (1619.12-1664.71) | 1043.87 (1017.43-1065.19) | 1.57 (1.56-1.64) | 137.31 (125.92-141.12) | 93.28 (91.66-94.73) | 1.47 (1.35-1.54) | 1790798706-w7-qwen05b-r2-pp200.md | every round passed |
| Qwen2.5-0.5B | 200 | 5 | 1609.05 (181.43-1743.56) | 882.93 (846.78-1051.98) | 1.55 (0.21-1.90) | 133.98 (13.35-146.07) | 91.43 (88.63-94.07) | 1.42 (0.15-1.63) | 1790797516-w7-qwen05b-pp200.md (first run; rounds 0-1 ran on ~11-13 busy cores, gate still read quiet) | passed, 2 outlier rounds |
| Qwen2.5-0.5B | 512 | 5 | 1627.29 (1455.46-1675.03) | 1064.36 (1027.32-1075.28) | 1.56 (1.37-1.58) | 142.93 (133.95-146.13) | 92.16 (91.99-92.34) | 1.55 (1.45-1.59) | 1790797618-w7-qwen05b-pp512.md | every round passed |
| Qwen2.5-1.5B | 200 | 5 | 543.31 (523.47-584.14) | 295.42 (289.80-308.86) | 1.77 (1.74-2.00) | 42.33 (40.76-43.39) | 38.36 (37.58-38.93) | 1.10 (1.08-1.14) | 1790798386-w7-qwen15b-r5-pp200.md | every round passed |
| Qwen2.5-1.5B | 200 | 3 | 600.79 (257.29-614.78) | 294.65 (294.34-298.76) | 2.01 (0.87-2.09) | 46.32 (44.37-46.73) | 38.42 (38.23-38.53) | 1.21 (1.15-1.22) | 1790797748-w7-qwen15b-pp200.md (earlier run; one outlier round 257 tok/s) | every round passed |
| Qwen2.5-1.5B | 512 | 3 (reps 5 exceeded 560 s, exit 143) | 584.23 (575.57-594.54) | 305.22 (303.59-306.15) | 1.91 (1.89-1.96) | 45.79 (40.67-47.39) | 38.52 (38.21-39.01) | 1.19 (1.06-1.21) | 1790797926-w7-qwen15b-pp512.md | every round passed |
| SmolLM2-135M | 200/512 | - | not run: greedy ids 0/16 vs llama.cpp (RoPE pairing) | | | | | | - | - |

Run-to-run spread is material: 1.5B pp200 ratio 1.77 (reps 5) vs 2.01 (reps 3) on the same binary; 1.5B tg 1.10 vs 1.21. Treat 0.5B/1.5B pp200 as 1.8-2.0 and 1.1-1.2 tg. The 1.5B pp512 reps-5 attempt (`qwen15b-r5-pp512`) timed out at 560 s; no receipt.

## Traced phase tables (copied from the receipts; share/rate columns as printed; receipts above)

Hybrid Qwen3.5-0.8B pp512 (1790798160): pp 0.83 ms/token wall: norm 0.057 (6.9%); residual 0.044 (5.4%); attn.proj_in 0.031 (1014 GMAC/s); attn.core 0.072 (8.7%); attn.proj_out 0.017 (750); affine.proj_in 0.137 (1105); affine.conv 0.033; affine.heads 0.084; affine.proj_out 0.049 (768); ffn 0.287 (34.7%, 922); head 0.005; pool.wait overlay 0.121 (14.6%). tg 11.16 ms/token: ffn 3.723 (33.4%, 75.4 GB/s); affine.proj_in 2.612; head 2.120 (19.0%); attn.proj_in 0.602 (55.5 GB/s); affine.proj_out 0.611; affine.heads 0.440; attn.core 0.261; affine.conv 0.289; attn.proj_out 0.226; norm 0.189; pool.wait overlay 1.688 (15.1%).

### Qwen2.5-0.5B pp200 (1790798706)

pp: 200 tokens, 0.62 ms/token wall

| phase | ms/token | share | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| norm | 0.054 | 8.8% | 49 | - | - |
| residual | 0.040 | 6.5% | 48 | - | - |
| attn.proj_in | 0.040 | 6.5% | 24 | 3.3 | 612.3 |
| attn.core | 0.069 | 11.2% | 24 | - | - |
| attn.proj_out | 0.030 | 4.8% | 24 | 3.5 | 651.6 |
| ffn | 0.372 | 59.8% | 24 | 4.5 | 844.1 |
| head | 0.010 | 1.6% | 1 | 72.3 | 68.0 |
| embed/untraced/outside | 0.006 | 1.0% | - | - | - |
| pool.wait (overlay; 317.7 tasks/dispatch) | 0.112 | 18.0% | 388 | - | - |

tg: 64 tokens, 7.34 ms/token wall

| phase | ms/token | share | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| norm | 0.168 | 2.3% | 3136 | - | - |
| residual | 0.045 | 0.6% | 3072 | - | - |
| attn.proj_in | 0.724 | 9.9% | 1536 | 36.4 | 34.2 |
| attn.core | 0.284 | 3.9% | 1536 | - | - |
| attn.proj_out | 0.414 | 5.6% | 1536 | 49.4 | 46.5 |
| ffn | 4.483 | 61.1% | 1536 | 74.4 | 70.0 |
| head | 1.184 | 16.1% | 64 | 122.1 | 114.9 |
| embed/untraced/outside | 0.037 | 0.5% | - | - | - |
| pool.wait (overlay; 38.1 tasks/dispatch) | 1.545 | 21.0% | 15552 | - | - |

### Qwen2.5-0.5B pp512 (1790797618)

pp: 512 tokens, 0.61 ms/token wall

| phase | ms/token | share | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| norm | 0.044 | 7.1% | 98 | - | - |
| residual | 0.036 | 5.9% | 96 | - | - |
| attn.proj_in | 0.038 | 6.3% | 48 | 2.7 | 643.5 |
| attn.core | 0.101 | 16.5% | 48 | - | - |
| attn.proj_out | 0.030 | 4.8% | 48 | 2.7 | 649.8 |
| ffn | 0.357 | 58.1% | 48 | 3.7 | 880.1 |
| head | 0.002 | 0.4% | 1 | 120.8 | 113.7 |
| embed/untraced/outside | 0.006 | 1.0% | - | - | - |
| pool.wait (overlay; 393.7 tasks/dispatch) | 0.101 | 16.4% | 773 | - | - |

tg: 64 tokens, 7.63 ms/token wall

| phase | ms/token | share | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| norm | 0.172 | 2.3% | 3136 | - | - |
| residual | 0.045 | 0.6% | 3072 | - | - |
| attn.proj_in | 0.637 | 8.3% | 1536 | 41.3 | 38.9 |
| attn.core | 0.232 | 3.0% | 1536 | - | - |
| attn.proj_out | 0.380 | 5.0% | 1536 | 53.9 | 50.7 |
| ffn | 4.933 | 64.7% | 1536 | 67.6 | 63.6 |
| head | 1.191 | 15.6% | 64 | 121.5 | 114.3 |
| embed/untraced/outside | 0.036 | 0.5% | - | - | - |
| pool.wait (overlay; 38.1 tasks/dispatch) | 1.831 | 24.0% | 15552 | - | - |

### Qwen2.5-1.5B pp200 (1790798386, reps 5)

pp: 200 tokens, 2.19 ms/token wall

| phase | ms/token | share | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| norm | 0.142 | 6.5% | 57 | - | - |
| residual | 0.125 | 5.7% | 56 | - | - |
| attn.proj_in | 0.129 | 5.9% | 28 | 3.6 | 682.1 |
| attn.core | 0.184 | 8.4% | 28 | - | - |
| attn.proj_out | 0.133 | 6.1% | 28 | 2.6 | 496.1 |
| ffn | 1.455 | 66.4% | 28 | 4.2 | 794.6 |
| head | 0.012 | 0.5% | 1 | 104.5 | 98.3 |
| embed/untraced/outside | 0.012 | 0.5% | - | - | - |
| pool.wait (overlay; 296.8 tasks/dispatch) | 0.347 | 15.8% | 452 | - | - |

tg: 64 tokens, 24.84 ms/token wall

| phase | ms/token | share | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| norm | 0.354 | 1.4% | 3648 | - | - |
| residual | 0.087 | 0.4% | 3584 | - | - |
| attn.proj_in | 2.198 | 8.9% | 1792 | 42.6 | 40.1 |
| attn.core | 0.532 | 2.1% | 1792 | - | - |
| attn.proj_out | 1.295 | 5.2% | 1792 | 54.2 | 51.0 |
| ffn | 17.848 | 71.9% | 1792 | 68.8 | 64.8 |
| head | 2.453 | 9.9% | 64 | 101.1 | 95.1 |
| embed/untraced/outside | 0.069 | 0.3% | - | - | - |
| pool.wait (overlay; 44.6 tasks/dispatch) | 7.302 | 29.4% | 18112 | - | - |

### Qwen2.5-1.5B pp512 (1790797926, reps 3)

pp: 512 tokens, 1.73 ms/token wall

| phase | ms/token | share | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| norm | 0.108 | 6.3% | 114 | - | - |
| residual | 0.075 | 4.3% | 112 | - | - |
| attn.proj_in | 0.113 | 6.6% | 56 | 3.2 | 776.1 |
| attn.core | 0.210 | 12.1% | 56 | - | - |
| attn.proj_out | 0.087 | 5.0% | 56 | 3.2 | 763.4 |
| ffn | 1.123 | 65.0% | 56 | 4.3 | 1029.5 |
| head | 0.004 | 0.2% | 1 | 120.5 | 113.4 |
| embed/untraced/outside | 0.010 | 0.6% | - | - | - |
| pool.wait (overlay; 365.7 tasks/dispatch) | 0.302 | 17.5% | 901 | - | - |

tg: 64 tokens, 21.66 ms/token wall

| phase | ms/token | share | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| norm | 0.318 | 1.5% | 3648 | - | - |
| residual | 0.076 | 0.4% | 3584 | - | - |
| attn.proj_in | 1.821 | 8.4% | 1792 | 51.4 | 48.4 |
| attn.core | 0.658 | 3.0% | 1792 | - | - |
| attn.proj_out | 1.075 | 5.0% | 1792 | 65.3 | 61.4 |
| ffn | 15.697 | 72.5% | 1792 | 78.3 | 73.6 |
| head | 1.958 | 9.0% | 64 | 126.6 | 119.2 |
| embed/untraced/outside | 0.061 | 0.3% | - | - | - |
| pool.wait (overlay; 44.6 tasks/dispatch) | 5.139 | 23.7% | 18112 | - | - |

## Gap to 2.0x (derived)

| model, P | llama pp | 2x target | apeiron now | cut needed | tg target (2x llama) | tg now | tg read-bound (ceiling / bytes per token) | max tg ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5B, 200 | 1043.87 | 2087.7 = 0.479 ms | 1646.3 = 0.607 ms | -0.128 ms (21%) | 186.6 | 137.3 | 161.4 (85.7 GB/s over 531.07 MB) | 1.73 |
| 0.5B, 512 | 1064.36 | 2128.7 = 0.470 ms | 1627.3 = 0.615 ms | -0.145 ms (24%) | 184.3 | 142.9 | 158.4 (84.1 over 531.07 MB) | 1.72 |
| 1.5B, 200 (r5) | 295.42 | 590.8 = 1.693 ms | 543.3 = 1.841 ms | -0.148 ms (8%) | 76.7 | 42.3 | 52.0 (85.7 over 1646.6 MB) | 1.36 |
| 1.5B, 512 (r3) | 305.22 | 610.4 = 1.638 ms | 584.2 = 1.712 ms | -0.073 ms (4%) | 77.0 | 45.8 | 51.1 (84.1 over 1646.6 MB) | 1.33 |

Bytes per token: 0.5B = whole file (tied head reads `token_embd`, 144.64 MB Q8_0). 1.5B = file minus `token_embd` (247.96 MB; head reads separate `output.weight`). Ceiling = receipt read ceiling at 16 threads (1 GiB sum, best of 3). Apeiron's tg streams 73-76 GB/s (0.5B) and 70-75 GB/s (1.5B) by this count; llama.cpp 49 and 63 GB/s. The traced `head` GB/s (101-127) exceeds the ceiling because the trace counts nominal dense bytes and stage 1 streams part (W6 note).

### Ranked gap list, pp (phases of the wall that keep pp below 2.0x; owner = file that owns the lever)

Owner rule: `step/cols.rs`, `chart/ffn.rs`, `tests.rs` = other stream; all else ours. Call sites of every phase span are in `step/cols.rs` (Norm, Residual, AttnProjIn, AttnCore, AttnProjOut, Ffn) and `step.rs` (Head); the kernels live in the files named.

0.5B (cut needed 0.13-0.15 ms/token; same ranking at P=200 and 512):
1. ffn 0.357-0.372 ms (58-60%, 844-880 GMAC/s). Alone it would need about 1480 GMAC/s at P=512 to close the gap (derived: 880*0.357/0.212). Owner: other stream (`chart/ffn.rs`, `step/cols.rs`); GEMM kernel `kern/gemm.rs` is ours.
2. attn.core 0.069 (P=200) -> 0.101 ms (P=512), 11% -> 16.5%. Grows with context (derived: +0.032 ms for 2.56x positions). Ours: `attn/f32_core.rs` (kernel), `attn.rs` (`attn_head_f32`), `rope.rs`, `step/attn_block.rs`. The span also holds bias, RoPE, KV append.
3. pool.wait overlay 0.101-0.112 ms (16-18% of wall, overlaps the compute phases): ours, `chart/pool.rs`, `par.rs` (dispatch and tail imbalance; 388-773 dispatches/pass).
4. norm 0.044-0.054 (7-9%): ours, `norm.rs`; call site `step/cols.rs`.
5. residual 0.036-0.040 (6-7%): add loop called from `step/cols.rs`; the implementation file was not traced here.
6. attn.proj_in + proj_out 0.068-0.070 combined (11-12%) at 612-652 GMAC/s against ffn's 844-880. At ffn's rate they would save about 0.018 ms (derived). Ours: `kern/gemm.rs`; calls in `step/cols.rs`.

1.5B (cut needed 0.07-0.15 ms/token, 4-8%):
1. ffn 1.12-1.46 ms (65-66%, 795-1030 GMAC/s; 795 at P=200 vs 1030 at P=512, the rate gap between P is 30%): other stream.
2. attn.core 0.184-0.210 (8-12%): ours (files above).
3. pool.wait overlay 0.30-0.35 (16-18%): ours (`chart/pool.rs`, `par.rs`).
4. norm 0.108-0.142 and residual 0.075-0.125 (together 10-12%): norm ours; `step/cols.rs` call sites.
5. attn.proj_out 0.087-0.133 at 496 (P=200) / 763 (P=512) GMAC/s: ours (`kern/gemm.rs`); attn.proj_in 0.113-0.129 at 682-776.

### Ranked gap list, tg

tg is bandwidth-limited: the 2.0x target is above the read bound for both models (table above), so no phase list can reach 2.0x at Q8_0; the remaining headroom against the bound is 10-19% of wall.
0.5B (7.3-7.6 ms/token; bound 6.2-6.3 ms):
1. ffn 4.48-4.93 ms (61-65%) at 68-74 GB/s against 84-86 GB/s ceiling: other stream.
2. pool.wait overlay 1.55-1.83 ms (21-24%, 243 dispatches/token): ours (`chart/pool.rs`, `par.rs`).
3. head 1.18-1.19 (16%): ours (`step.rs`, `lm_head.rs`).
4. attn.proj_in 0.64-0.72 at 36-41 GB/s and proj_out 0.38-0.41 at 49-54 GB/s (combined 13-15%): per-call bytes are about 1.1 MB (q,k,v) and 0.85 MB (o), so calls are dispatch-latency-bound (derived, 1536 calls per 64 tokens; about 30 us/call for proj_in); ours (`kern/gemm.rs`, `par.rs`), call sites `step/cols.rs`.
5. attn.core 0.23-0.28 (3-4%), norm 0.17, residual 0.045: ours.
1.5B (21.7-24.8 ms/token; bound about 19.2-19.6 ms, derived from 1646.6 MB / 84-86 GB/s): ffn 15.7-17.8 ms (72%, other stream); pool.wait overlay 5.1-7.3 ms (24-29%, ours); head 1.96-2.45 (ours); attn.proj_in 1.82-2.20 at 43-51 GB/s and proj_out 1.08-1.30 at 54-65 GB/s (ours); attn.core 0.53-0.66 (2-3%, ours).

## Is the f32 attention core the path taken (read-only source check)

- `APEIRON_F32_CARRIER=1` -> `mixer.rs:303-307` `carrier_from_env()` = `Carrier::F32`.
- `KvState::new` (`attn.rs:723-729`) builds `KvStorage::F32` from `carrier_from_env()`.
- `KvState::attn_head` (`attn.rs:995-996`) and `attn_head_fused` (`attn.rs:670-671`) return `attn_head_f32` when `carrier == F32`. Prefill and decode both reach it through `step/cols.rs:424` and `:562`.
- `attn_head_f32` (`attn.rs:1223-1322`) calls `f32_core::attn_core_f32_strided` (`attn/f32_core.rs:444-475`), which takes the AVX2+FMA path (`attn_core_f32_strided_avx2_inner`, with the 8-position tile `attn_tile8_avx2`) when `head_dim % 8 == 0` and AVX2+FMA are detected. Head dims here: 64 (0.5B), 128 (1.5B): tiled AVX2 path. Scalar twin otherwise.
- Stack scratch requires `head_dim <= 256` (true for all three).
- Caveat: one call is one (head, position) pair, so the tile covers 8 key positions of one query; there is no query-block batching across prefill positions (`cols.rs:424` calls per head per column). Not measured: the trace does not split f32_core from RoPE/bias/append inside `attn.core`.
- Evidence level: source read plus trace label `attn.core`; no perf/asm run.

## Open items

- SmolLM2/`llama` arch needs consecutive-pair RoPE; the loader does not refuse it (silent wrong output). Not implemented (scope).
- 1.5B reps-5 pp512 does not fit 540 s; 1.5B pp512 is reps 3.
- Run-to-run ratio spread on 1.5B pp200 (1.77 vs 2.01) shows one 3-5 round receipt is not enough to state a single ratio.
- F16-vs-Q8_0 llama.cpp difference at id 8 (1.5B) is unexplained beyond a near-tie reading; logits were not compared.
- Trust boundary: tokens compared on 64-token prompts, 16 ids; speed is one host, desktop session active; rustc/LLVM codegen and llama.cpp's own threading are outside the evidence.

## Artifacts

`/home/socol/oc-tmp/w7-fixtures/`: `*.gguf`, `*.ids`, `text.txt`, `meta.txt`, `ll-*.txt`, `ap-*.txt`, `run-*.log`, `tree-copy/` (contains `examples/w7_ids.rs`), `tree-copy-target/`, `ll.sh`, `ap.sh`, `speed.sh`, `speed2.sh`.
