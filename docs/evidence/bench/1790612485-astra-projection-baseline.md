# Engine benchmark `astra-projection-baseline`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/Workspace/apeiron/tests/fixtures/qwen35-prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `2d0b0ef13f21eccc008d860bd57a9d325dde9660` with 6 dirty paths; src sha256 `e932d0e81bc736d0be102cc3f9c1d3d6928f7a1251ed68fc1bb352fcfb402186`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 67.9 / 89.8 / 86.9 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 256.65 | 255.31 | 261.94 |
| llama.cpp pp (best threads) | 576.42 | 571.67 | 594.19 |
| ratio pp (per round) | 0.45 | 0.43 | 0.46 |
| apeiron tg | 67.31 | 62.65 | 69.30 |
| llama.cpp tg (best threads) | 52.37 | 51.11 | 55.68 |
| ratio tg (per round) | 1.28 | 1.16 | 1.32 |
| apeiron spec (e2e) | 47.43 | 44.37 | 49.72 |
| spec acceptance | 0.24 | 0.24 | 0.24 |
| apeiron pp busy cores | 23.70 | 23.68 | 23.74 |
| apeiron tg busy cores | 23.76 | 23.61 | 23.85 |
| mean MHz before pp | 2318.02 | 1312.87 | 3412.77 |
| mean MHz before tg | 4671.61 | 4451.82 | 4814.95 |

### Traced phases

#### pp: 200 tokens, 3.77 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.065 | 1.7% | 245 | - | - |
| residual | 0.050 | 1.3% | 240 | - | - |
| attn.proj_in | 0.105 | 2.8% | 30 | 7.9 | 298.6 |
| attn.core | 0.270 | 7.2% | 30 | - | - |
| attn.proj_out | 0.051 | 1.3% | 30 | 6.6 | 248.5 |
| affine.proj_in | 0.469 | 12.4% | 90 | 8.6 | 323.3 |
| affine.conv | 0.442 | 11.7% | 90 | - | - |
| affine.heads | 1.113 | 29.5% | 90 | - | - |
| affine.proj_out | 0.150 | 4.0% | 90 | 6.7 | 252.5 |
| ffn | 1.015 | 26.9% | 120 | 6.9 | 260.3 |
| head | 0.025 | 0.7% | 1 | 55.0 | 51.8 |
| untraced in pass | 0.006 | 0.2% | 0 | - | - |
| outside passes | 0.012 | 0.3% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 44.7) | 0.668 | 17.7% | 25464 | - | - |

#### tg: 64 tokens, 15.05 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.189 | 1.3% | 3136 | - | - |
| residual | 0.048 | 0.3% | 3072 | - | - |
| attn.proj_in | 0.626 | 4.2% | 384 | 53.4 | 50.2 |
| attn.core | 0.587 | 3.9% | 384 | - | - |
| attn.proj_out | 0.231 | 1.5% | 384 | 57.8 | 54.4 |
| affine.proj_in | 2.689 | 17.9% | 1152 | 59.9 | 56.4 |
| affine.conv | 0.820 | 5.4% | 1152 | - | - |
| affine.heads | 1.470 | 9.8% | 1152 | - | - |
| affine.proj_out | 0.625 | 4.2% | 1152 | 64.1 | 60.4 |
| ffn | 3.971 | 26.4% | 1536 | 70.7 | 66.5 |
| head | 3.752 | 24.9% | 64 | 72.0 | 67.8 |
| untraced in pass | 0.035 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 49.3) | 2.473 | 16.4% | 19712 | - | - |

#### spec: 64 tokens, 21.92 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 52 | - | - |
| norm | 0.572 | 2.6% | 2548 | - | - |
| residual | 0.093 | 0.4% | 2496 | - | - |
| attn.proj_in | 0.737 | 3.4% | 364 | 43.0 | 75.8 |
| attn.core | 0.694 | 3.2% | 364 | - | - |
| attn.proj_out | 0.319 | 1.5% | 364 | 39.7 | 70.0 |
| affine.proj_in | 3.051 | 13.9% | 936 | 42.9 | 80.0 |
| affine.conv | 1.102 | 5.0% | 936 | - | - |
| affine.heads | 2.263 | 10.3% | 936 | - | - |
| affine.proj_out | 0.844 | 3.9% | 936 | 38.6 | 71.9 |
| ffn | 5.611 | 25.6% | 1300 | 42.3 | 77.8 |
| head | 4.773 | 21.8% | 103 | 51.9 | 91.3 |
| restore | 1.568 | 7.2% | 39 | - | - |
| untraced in pass | 0.136 | 0.6% | 0 | - | - |
| outside passes | 0.152 | 0.7% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 92.7) | 4.233 | 19.3% | 29896 | - | - |
