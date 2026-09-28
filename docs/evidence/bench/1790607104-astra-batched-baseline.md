# Engine benchmark `astra-batched-baseline`

- model `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` (sha256 `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`)
- prompt ids `/home/socol/oc-tmp/q35-t5/prompt2.ids` (sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`), pp 200, tg 64, 5 rounds
- git `39298a1c858a11397b7a9e3d638c5a0b796b4a47` with 6 dirty paths; src sha256 `96c75f9773af6cb9d0f0ef2536cc7c0c8ccb86020759fe614c214a706300c8de`
- Intel(R) Core(TM) i9-14900KF; governor powersave; 32 CPUs; rustc 1.97.0-nightly (17584a181 2026-04-13)
- read ceiling (1 GiB sum, best of 3): 65.8 / 88.6 / 88.1 GB/s at 8 / 16 / 32 threads
- llama.cpp `0324696b8`, threads 8,16,24,32
- quiet gate busy < 0.05: every round passed

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 224.99 | 222.14 | 236.69 |
| llama.cpp pp (best threads) | 523.54 | 497.13 | 574.82 |
| ratio pp (per round) | 0.43 | 0.39 | 0.48 |
| apeiron tg | 62.71 | 62.44 | 64.65 |
| llama.cpp tg (best threads) | 49.90 | 46.88 | 50.81 |
| ratio tg (per round) | 1.27 | 1.25 | 1.33 |
| apeiron spec (e2e) | 37.95 | 37.67 | 39.14 |
| spec acceptance | 0.24 | 0.24 | 0.24 |
| apeiron pp busy cores | 23.74 | 23.70 | 23.77 |
| apeiron tg busy cores | 23.77 | 23.73 | 23.82 |
| mean MHz before pp | 2819.17 | 2207.62 | 4254.25 |
| mean MHz before tg | 4675.39 | 4634.62 | 4829.34 |

### Traced phases

#### pp: 200 tokens, 4.33 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.001 | 0.0% | 5 | - | - |
| norm | 0.064 | 1.5% | 245 | - | - |
| residual | 0.051 | 1.2% | 240 | - | - |
| attn.proj_in | 0.115 | 2.6% | 30 | 7.3 | 274.4 |
| attn.core | 0.355 | 8.2% | 30 | - | - |
| attn.proj_out | 0.057 | 1.3% | 30 | 5.9 | 222.4 |
| affine.proj_in | 0.514 | 11.9% | 90 | 7.8 | 294.8 |
| affine.conv | 0.519 | 12.0% | 90 | - | - |
| affine.heads | 1.323 | 30.5% | 90 | - | - |
| affine.proj_out | 0.164 | 3.8% | 90 | 6.1 | 229.5 |
| ffn | 1.125 | 26.0% | 120 | 6.2 | 234.9 |
| head | 0.025 | 0.6% | 1 | 53.3 | 50.2 |
| untraced in pass | 0.007 | 0.2% | 0 | - | - |
| outside passes | 0.012 | 0.3% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 44.7) | 1.137 | 26.3% | 25464 | - | - |

#### tg: 64 tokens, 15.91 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 64 | - | - |
| norm | 0.227 | 1.4% | 3136 | - | - |
| residual | 0.056 | 0.4% | 3072 | - | - |
| attn.proj_in | 0.631 | 4.0% | 384 | 52.9 | 49.8 |
| attn.core | 0.601 | 3.8% | 384 | - | - |
| attn.proj_out | 0.218 | 1.4% | 384 | 61.4 | 57.8 |
| affine.proj_in | 2.702 | 17.0% | 1152 | 59.6 | 56.1 |
| affine.conv | 0.975 | 6.1% | 1152 | - | - |
| affine.heads | 1.621 | 10.2% | 1152 | - | - |
| affine.proj_out | 0.638 | 4.0% | 1152 | 62.9 | 59.2 |
| ffn | 4.095 | 25.7% | 1536 | 68.6 | 64.5 |
| head | 4.100 | 25.8% | 64 | 65.9 | 62.0 |
| untraced in pass | 0.039 | 0.2% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 49.3) | 2.355 | 14.8% | 19712 | - | - |

#### spec: 64 tokens, 25.21 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 52 | - | - |
| norm | 0.581 | 2.3% | 2548 | - | - |
| residual | 0.096 | 0.4% | 2496 | - | - |
| attn.proj_in | 0.878 | 3.5% | 364 | 36.1 | 63.6 |
| attn.core | 0.722 | 2.9% | 364 | - | - |
| attn.proj_out | 0.337 | 1.3% | 364 | 37.7 | 66.4 |
| affine.proj_in | 3.581 | 14.2% | 936 | 36.5 | 68.1 |
| affine.conv | 1.178 | 4.7% | 936 | - | - |
| affine.heads | 2.502 | 9.9% | 936 | - | - |
| affine.proj_out | 0.980 | 3.9% | 936 | 33.2 | 62.0 |
| ffn | 6.450 | 25.6% | 1300 | 36.8 | 67.6 |
| head | 5.861 | 23.3% | 103 | 42.3 | 74.4 |
| restore | 1.620 | 6.4% | 39 | - | - |
| untraced in pass | 0.258 | 1.0% | 0 | - | - |
| outside passes | 0.160 | 0.6% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 92.7) | 6.991 | 27.7% | 29896 | - | - |
