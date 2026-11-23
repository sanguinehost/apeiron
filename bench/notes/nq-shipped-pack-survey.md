# Shipped NanoQuant packs: one 8B fit, instruct only, and already scored here

The question: is there a ready-made Qwen3-8B NanoQuant pack that can replace the Base campaign?
The answer is one full 8B fit, and it cannot: it is the **instruct** checkpoint, and its 0-shot
screen is below the Q2 bar. The Base campaign remains the Q3 path. The pack is usable as a
serving and performance baseline, which is a different criterion.

## Official release

`SamsungLabs/NanoQuant` (Apache-2.0, three commits) ships code under `src/nanoquant/` and no
weights. Its README links the HF *paper* page, not a model repository. The copy at
`/workspace/NanoQuant` on `nq-l4` is the same code: not a git repository, no `Results/`, no
`.pt`, `.safetensors`, `.nqp` or `.ckpt` file.

## Third-party 8B pack

`arelath/Qwen3-8B-nanoquant-GGUF` — card "Experiment 29", status `completed`, Apache-2.0.

| field | value |
|---|---|
| base_model | `Qwen/Qwen3-8B` (**instruct**, not `Qwen3-8B-Base`) |
| revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| file | `qwen3-8b-nanoquant.gguf`, 2,214,297,824 B |
| sha256 | `75d3f4f150195e0c6a7ccd2ce3372f919534fb7ec609f6b9ba52eb71af194253` |
| format | GGUF v3, 1,407 tensors, 41 KV, context 40,960 |
| tensors | standard types plus `*.nq_salient_idx`, `nq_salient_weight`, `nq_scale_pre/mid/post`, `nq_u`, `nq_v` |

The card's own protocol is WikiText-2, 64 samples x 128 tokens, and six tasks on their first 200
rows. It reports PPL 23.110475 -> 40.599726 (**1.7568x** BF16) and 0-shot ratios 0.6687 (piqa),
0.4969 (arc_easy), 0.6355 (arc_challenge), 0.6378 (hellaswag), 0.8085 (winogrande), 0.7719
(boolq); the six-task mean is **0.670x** BF16.

## Converted and scored here

`tools/nq_convert.py` reads the `nq_*` tensors and writes a v2 `.nqp`. Two carriers were built:

| pack | bytes | rate | skeleton | score on our protocol |
|---|---:|---:|---|---|
| `nq8b/model.nqp` | 976,049,680 | 1.124186 b/w | `nq8b/skeleton.gguf`, 1,329,625,696 B | **1.606603x** Q4_K_M at ctx 2048; Q8_0 head against a Q6_K bar, not same-head |
| `nq8b-late-v4/model.nqp` | 1,023,456,080 | 1.178785 b/w | `nq8b/skeleton-q6k-q4kembed.gguf`, 867,740,256 B | **1.417631** [1.355945, 1.482123] over 17 windows; **1.499754** [1.391674, 1.616228] over five; **0.7534x** on the six-task 0-shot screen |

The 17-window and five-window rows are against the official Q4_K_M file on the same token stream,
not against the card's 64x128 protocol. The six-task 0-shot row is 100 items per task and refutes
the pack: NQ 0.5600 against Q4_K_M 0.7433, paired difference -0.1833 +/- 0.0812, entirely below
the -0.0743 allowance.

Receipts: `bench/cpu/1794879503-nq8b-1124-ctx2048.md` (1.124 b/w pack),
`bench/cpu/1794885786-nq8b-uv-signs.md` (converter and layout),
`bench/cpu/1794912865-nq8b-overnight.md` and `bench/cpu/1794922136-nq8b-q2-five-windows.md`
(quality), `bench/cpu/1794937300-nq8b-q2-rented-host.md` (0-shot).

## Disposition

- **Not Q3.** Q3 names `Qwen3-8B-Base`; this fit is the instruct checkpoint, so it is a different
  model and cannot be the published row's parity artifact. No 8B **Base** NanoQuant pack was found
  on HF. The Base campaign is still required for Q3.
- **Not a Q2 product pack.** The 0-shot half is 0.7534x against the 0.90 bar; PPL reads 1.42-1.50x
  while the tasks lose a quarter of their accuracy. The next pack is judged on the 0-shot screen.
- **Usable now for serving and performance.** It is a real third-party fit of the same method, at
  1.18 b/w, and it loads through the current engine. It can replace the campaign for I1/I2 and T1
  work while the Base fit runs, and it is the same-source reference the campaign fit is compared
  against at equal rate.
- The other `arelath` conversions (`0.6B`, Llama-3.2-1B/3B, Llama-3-8B, Gemma-3-270m/1b/4b) do not
  change the 8B row. The 0.6B parity result already exists in this repository.
